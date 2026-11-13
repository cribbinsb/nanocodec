//! `residual_coding()` (HEVC §7.3.8.11) and its context selection
//! (HEVC §9.3.4.2.4–9.3.4.2.7), for all three scan orders, without
//! transform skip.
//!
//! The structure follows kvazaar's `kvz_encode_coeff_nxn_generic` and
//! `kvz_encode_last_significant_xy` (BSD-3-Clause, see `THIRD_PARTY.md`).

use crate::cabac::BinSink;
use crate::contexts::{
    CODED_SUB_BLOCK_CHROMA_OFFSET, Contexts, GREATER1_CHROMA_OFFSET, GREATER2_CHROMA_OFFSET,
    LAST_CHROMA_OFFSET, SIG_CHROMA_OFFSET,
};
use crate::scan::{Level, SUB_BLOCK_COEFFS, SUB_BLOCK_LOG2, Scan, ScanKind, sub_block_coded};

/// A sub-block hides a sign when its last and first non-zero levels are
/// more than this many scan positions apart (`sbhThreshold`, §7.3.8.11).
pub const SIGN_HIDING_THRESHOLD: usize = 3;
/// At most this many `coeff_abs_level_greater1_flag`s per sub-block.
pub(crate) const MAX_GREATER1_FLAGS: usize = 8;
/// 4x4 sub-blocks in the largest (32x32) transform block.
const MAX_SUB_BLOCKS: usize = 64;
/// Largest Rice parameter for `coeff_abs_level_remaining` (HEVC §9.3.3.11).
pub(crate) const MAX_RICE_PARAM: u32 = 4;
/// Prefix length below which `coeff_abs_level_remaining` is plain Rice
/// coded; longer values switch to Exp-Golomb (`cMax = 4 << cRiceParam`).
pub(crate) const RICE_PREFIX_LIMIT: u32 = 3;

/// `ctxIdxMap` of HEVC §9.3.4.2.5, for 4x4 transform blocks.
const SIG_CTX_4X4: [usize; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8];

/// Prefix group of a last-position coordinate (HEVC §9.3.3.2 TR binarisation
/// of `last_sig_coeff_*_prefix`), for coordinates 0..31.
pub(crate) const GROUP_IDX: [u32; 32] = [
    0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
];
/// Smallest coordinate in each prefix group.
const MIN_IN_GROUP: [u32; 10] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24];

/// Code the levels of one transform block. `levels` is row-major
/// (`levels[y * size + x]`) and must contain at least one non-zero level.
pub fn write_residual<L: Level>(
    w: &mut impl BinSink,
    ctx: &mut Contexts,
    scan: &Scan,
    levels: &[L],
    luma: bool,
    sign_hiding: bool,
) {
    let log2 = scan.log2;
    let size = 1usize << log2;
    let sub_side = size >> SUB_BLOCK_LOG2;

    // Coded flags of each 4x4 sub-block, raster order of sub-blocks.
    // Each 4x4 sub-block is tested in raster order (four contiguous rows);
    // the last level is then searched for only inside the last coded one.
    let mut coded = [false; MAX_SUB_BLOCKS];
    for ys in 0..sub_side {
        for xs in 0..sub_side {
            coded[ys * sub_side + xs] = sub_block_coded(levels, size, (xs, ys));
        }
    }
    let last_coded = scan
        .sub_blocks
        .iter()
        .rposition(|&(xs, ys)| coded[ys * sub_side + xs])
        .expect("write_residual needs a non-zero level");
    let first = last_coded * SUB_BLOCK_COEFFS;
    let last_scan_pos = first
        + scan.positions[first..first + SUB_BLOCK_COEFFS]
            .iter()
            .rposition(|&p| levels[p] != L::default())
            .expect("a coded sub-block has a non-zero level");
    let last_sub_block = last_scan_pos / SUB_BLOCK_COEFFS;
    let last_pos = scan.positions[last_scan_pos];
    write_last_position(w, ctx, scan, last_pos, luma);

    // `lastGreater1Ctx` carried between sub-blocks (HEVC §9.3.4.2.6).
    let mut greater1_state = 1usize;
    let mut scan_pos = last_scan_pos as isize;
    for i in (0..=last_sub_block).rev() {
        let (xs, ys) = scan.sub_blocks[i];
        let first = i * SUB_BLOCK_COEFFS;
        let mut magnitudes = [0u32; SUB_BLOCK_COEFFS];
        let mut count = 0;
        let mut signs = 0u32;
        // Scan positions of the sub-block's last and first non-zero levels.
        let (mut last_nz, mut first_nz) = (None, 0);

        if scan_pos == last_scan_pos as isize {
            let l: i32 = levels[last_pos].into();
            magnitudes[0] = l.unsigned_abs();
            count = 1;
            signs = u32::from(l < 0);
            (last_nz, first_nz) = (Some(last_scan_pos), last_scan_pos);
            scan_pos -= 1;
        }

        // coded_sub_block_flag: inferred 1 for the first and last sub-blocks.
        if i == last_sub_block || i == 0 {
            coded[ys * sub_side + xs] = true;
        } else {
            let csbf_ctx = usize::from(neighbour_coded(&coded, sub_side, xs, ys));
            let base = if luma {
                0
            } else {
                CODED_SUB_BLOCK_CHROMA_OFFSET
            };
            w.encode_bin(
                &mut ctx.coded_sub_block_flag[base + csbf_ctx],
                coded[ys * sub_side + xs],
            );
        }

        if coded[ys * sub_side + xs] {
            let pattern = sig_pattern(&coded, sub_side, xs, ys);
            while scan_pos >= first as isize {
                let p = scan.positions[scan_pos as usize];
                let level: i32 = levels[p].into();
                // The DC of a coded, non-first sub-block is inferred to be
                // significant if nothing else in it was (HEVC §7.4.9.11).
                let inferred = scan_pos as usize == first && i > 0 && count == 0;
                if !inferred {
                    let sig_ctx = sig_ctx_inc(pattern, p % size, p / size, log2, luma, scan.kind);
                    let base = if luma { 0 } else { SIG_CHROMA_OFFSET };
                    w.encode_bin(&mut ctx.sig_coeff_flag[base + sig_ctx], level != 0);
                }
                if level != 0 {
                    magnitudes[count] = level.unsigned_abs();
                    count += 1;
                    signs = (signs << 1) | u32::from(level < 0);
                    last_nz.get_or_insert(scan_pos as usize);
                    first_nz = scan_pos as usize;
                }
                scan_pos -= 1;
            }
        } else {
            scan_pos = first as isize - 1;
        }

        let Some(last_nz) = last_nz else {
            continue;
        };
        // The sign of the first (lowest scan position) level is hidden in
        // the parity of the level sum when the levels span more than
        // SIGN_HIDING_THRESHOLD positions (§7.3.8.11 signHidden).
        let hidden = sign_hiding && last_nz - first_nz > SIGN_HIDING_THRESHOLD;
        let (signs, sign_count) = if hidden {
            (signs >> 1, count - 1)
        } else {
            (signs, count)
        };
        write_levels(
            w,
            ctx,
            &magnitudes[..count],
            (signs, sign_count as u32),
            i,
            luma,
            &mut greater1_state,
        );
    }
}

/// Greater-1/greater-2 flags, signs and remaining levels of one sub-block.
/// `magnitudes` are in reverse scan order (as coded).
fn write_levels(
    w: &mut impl BinSink,
    ctx: &mut Contexts,
    magnitudes: &[u32],
    (signs, sign_count): (u32, u32),
    sub_block: usize,
    luma: bool,
    greater1_state: &mut usize,
) {
    let mut ctx_set = if sub_block > 0 && luma { 2 } else { 0 };
    if *greater1_state == 0 {
        ctx_set += 1;
    }
    *greater1_state = 1;

    let g1_base = if luma { 0 } else { GREATER1_CHROMA_OFFSET } + 4 * ctx_set;
    let flagged = magnitudes.len().min(MAX_GREATER1_FLAGS);
    let mut first_greater1 = None;
    for (idx, &m) in magnitudes[..flagged].iter().enumerate() {
        let greater1 = m > 1;
        w.encode_bin(&mut ctx.greater1_flag[g1_base + *greater1_state], greater1);
        if greater1 {
            *greater1_state = 0;
            first_greater1.get_or_insert(idx);
        } else if (1..3).contains(greater1_state) {
            *greater1_state += 1;
        }
    }
    if *greater1_state == 0
        && let Some(idx) = first_greater1
    {
        let g2_base = if luma { 0 } else { GREATER2_CHROMA_OFFSET };
        w.encode_bin(
            &mut ctx.greater2_flag[g2_base + ctx_set],
            magnitudes[idx] > 2,
        );
    }

    w.encode_bypass(signs, sign_count);

    if *greater1_state == 0 || magnitudes.len() > MAX_GREATER1_FLAGS {
        let mut rice = 0u32;
        let mut first_greater2_pending = true;
        for (idx, &m) in magnitudes.iter().enumerate() {
            // Levels already implied by the flags coded above.
            let base = if idx < MAX_GREATER1_FLAGS {
                2 + u32::from(first_greater2_pending)
            } else {
                1
            };
            if m >= base {
                write_remaining(w, m - base, rice);
                if m > 3 * (1 << rice) {
                    rice = (rice + 1).min(MAX_RICE_PARAM);
                }
            }
            if m >= 2 {
                first_greater2_pending = false;
            }
        }
    }
}

/// `coeff_abs_level_remaining` binarisation (HEVC §9.3.3.11): a Rice code,
/// escaping to Exp-Golomb beyond `RICE_PREFIX_LIMIT << rice`.
fn write_remaining(w: &mut impl BinSink, value: u32, rice: u32) {
    if value < (RICE_PREFIX_LIMIT << rice) {
        let prefix = value >> rice;
        w.encode_bypass((1 << (prefix + 1)) - 2, prefix + 1);
        w.encode_bypass(value & ((1 << rice) - 1), rice);
    } else {
        let mut length = rice;
        let mut rest = value - (RICE_PREFIX_LIMIT << rice);
        while rest >= (1 << length) {
            rest -= 1 << length;
            length += 1;
        }
        let prefix_bins = RICE_PREFIX_LIMIT + length + 1 - rice;
        w.encode_bypass((1 << prefix_bins) - 2, prefix_bins);
        w.encode_bypass(rest, length);
    }
}

/// `last_sig_coeff_{x,y}_{prefix,suffix}` (HEVC §7.3.8.11, §9.3.4.2.3).
fn write_last_position(
    w: &mut impl BinSink,
    ctx: &mut Contexts,
    scan: &Scan,
    last_pos: usize,
    luma: bool,
) {
    let log2 = scan.log2;
    let size = 1usize << log2;
    // A vertical scan codes the last position with x and y swapped.
    let (x, y) = match scan.kind {
        ScanKind::Vertical => (last_pos / size, last_pos % size),
        _ => (last_pos % size, last_pos / size),
    };
    // ctxOffset / ctxShift of HEVC §9.3.4.2.3.
    let log2 = u32::from(log2);
    let (offset, shift) = if luma {
        (
            (3 * (log2 - 2) + ((log2 - 1) >> 2)) as usize,
            (log2 + 1) >> 2,
        )
    } else {
        (LAST_CHROMA_OFFSET, log2 - 2)
    };
    let max_group = GROUP_IDX[(1usize << log2) - 1];
    for (coord, prefix_ctx) in [(x, &mut ctx.last_x_prefix), (y, &mut ctx.last_y_prefix)] {
        let group = GROUP_IDX[coord];
        for bin in 0..group {
            w.encode_bin(&mut prefix_ctx[offset + (bin >> shift) as usize], true);
        }
        if group < max_group {
            w.encode_bin(&mut prefix_ctx[offset + (group >> shift) as usize], false);
        }
    }
    for coord in [x, y] {
        let group = GROUP_IDX[coord];
        if group > 3 {
            let suffix_bits = (group - 2) / 2;
            w.encode_bypass(coord as u32 - MIN_IN_GROUP[group as usize], suffix_bits);
        }
    }
}

/// True if the sub-block to the right or below is coded (`csbfCtx`).
pub(crate) fn neighbour_coded(coded: &[bool], side: usize, xs: usize, ys: usize) -> bool {
    (xs + 1 < side && coded[ys * side + xs + 1]) || (ys + 1 < side && coded[(ys + 1) * side + xs])
}

/// `prevCsbf` of HEVC §9.3.4.2.5: bit 0 = right sub-block coded, bit 1 =
/// lower sub-block coded. `None` for 4x4 transforms (one sub-block).
pub(crate) fn sig_pattern(coded: &[bool], side: usize, xs: usize, ys: usize) -> Option<usize> {
    if side == 1 {
        return None;
    }
    let right = xs + 1 < side && coded[ys * side + xs + 1];
    let below = ys + 1 < side && coded[(ys + 1) * side + xs];
    Some(usize::from(right) | (usize::from(below) << 1))
}

/// `ctxInc` of `sig_coeff_flag` before the chroma offset (HEVC §9.3.4.2.5).
pub(crate) const fn sig_ctx_inc(
    pattern: Option<usize>,
    x: usize,
    y: usize,
    log2: u8,
    luma: bool,
    kind: ScanKind,
) -> usize {
    if x + y == 0 {
        return 0;
    }
    let Some(pattern) = pattern else {
        return SIG_CTX_4X4[(y << 2) + x];
    };
    let (xp, yp) = (x & 3, y & 3);
    let count = match pattern {
        0 => match xp + yp {
            0 => 2,
            1 | 2 => 1,
            _ => 0,
        },
        // Two in the first row (column) of the sub-block, one in the second.
        1 => match yp {
            0 => 2,
            1 => 1,
            _ => 0,
        },
        2 => match xp {
            0 => 2,
            1 => 1,
            _ => 0,
        },
        _ => 2,
    };
    // 8x8 blocks use offset 9 (diagonal scan) or 15; larger ones 21 (luma)
    // or 12.
    let offset = if log2 == 3 {
        if matches!(kind, ScanKind::Diagonal) {
            9
        } else {
            15
        }
    } else if luma {
        21
    } else {
        12
    };
    let not_first_sub_block = luma && (x >> 2) + (y >> 2) > 0;
    offset + count + if not_first_sub_block { 3 } else { 0 }
}
