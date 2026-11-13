//! `residual_coding()` (H.265 §7.3.8.11) of Main-profile streams: last
//! position, coded sub-block and significance flags, greater-1/2 flags,
//! Rice/Exp-Golomb remainders (§9.3.3.11), sign data hiding and
//! `transform_skip_flag`.
//!
//! The context selection of `sig_coeff_flag` and `coded_sub_block_flag` is
//! the encoder's (`crate::residual`), which follows §9.3.4.2.4–5, tabulated
//! once by scan index within a sub-block (as HM and `VVdeC` tabulate
//! `ctxIdxMap`), so the significance loop is one table load and one bin per
//! position. Significant positions are recorded without a branch on the
//! flag, and the greater-1 context state advances by table.

use super::cabac::Engine;
use super::ctx::{self, Contexts};
use crate::residual::{SIGN_HIDING_THRESHOLD, neighbour_coded, sig_ctx_inc, sig_pattern};
use crate::scan::{SUB_BLOCK_COEFFS, Scan, ScanKind};
use nc_common::decode::DecodeError;

/// Smallest value of each `last_sig_coeff_x/y_prefix` group: the last
/// position before its suffix is added (§7.4.9.11).
const MIN_IN_GROUP: [u32; 10] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24];
/// Greater-1 flags coded per sub-block.
const MAX_GREATER1: usize = 8;
/// Largest Rice parameter (§9.3.3.11).
const MAX_RICE: u32 = 4;
/// Prefix length below which the remainder is plain Rice coded.
const RICE_PREFIX: u32 = 3;
/// Longest Exp-Golomb suffix that keeps a level within 16 bits plus margin.
const MAX_SUFFIX: u32 = 28;
/// `CoeffMinY` magnitude for 8-bit video (§7.4.9.11).
const MAX_MAGNITUDE: u32 = 1 << 15;
/// Transform skip is allowed for 4x4 blocks only in version 1 (§7.3.8.11).
const TRANSFORM_SKIP_LOG2: u8 = 2;

/// Scan kinds, in `ScanKind as usize` order.
const SCAN_KINDS: usize = 3;
/// The raster position `4 * y + x` inside a 4x4 sub-block of each scan
/// index, by scan kind (§6.5.3 up-right diagonal, §6.5.4 horizontal,
/// §6.5.5 vertical); the test checks it against `Scan`.
const SUB_BLOCK_ORDER: [[u8; SUB_BLOCK_COEFFS]; SCAN_KINDS] = [
    [0, 4, 1, 8, 5, 2, 12, 9, 6, 3, 13, 10, 7, 14, 11, 15],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
];

/// The `sig_ctx_inc` cases a sub-block's 16 positions can fall in: the
/// transform class (4x4; 8x8 diagonal; 8x8 horizontal or vertical; larger),
/// luma or chroma, whether it is the first (DC) sub-block, and the coded
/// pattern of its right and lower neighbours (`prevCsbf`, 0..4; ignored
/// for 4x4 transforms).
const SIG_CLASSES: usize = 4;
/// `prevCsbf` values.
const SIG_PATTERNS: usize = 4;

/// The class of a transform for [`SIG_TABLE`].
const fn sig_class(log2: u8, kind: ScanKind) -> usize {
    match (log2, kind) {
        (2, _) => 0,
        (3, ScanKind::Diagonal) => 1,
        (3, _) => 2,
        _ => 3,
    }
}

/// Index of `sig_coeff_flag`'s context within `SIG_COEFF` (the chroma
/// offset included) for every case and scan index within the sub-block.
/// Index: `[scan kind][class][luma][first sub-block][prevCsbf][scan index]`.
#[allow(clippy::type_complexity)]
const SIG_TABLE: [[[[[[u8; SUB_BLOCK_COEFFS]; SIG_PATTERNS]; 2]; 2]; SIG_CLASSES]; SCAN_KINDS] = {
    // A representative log2 of each class.
    const CLASS_LOG2: [u8; SIG_CLASSES] = [2, 3, 3, 5];
    const KINDS: [ScanKind; SCAN_KINDS] =
        [ScanKind::Diagonal, ScanKind::Horizontal, ScanKind::Vertical];
    let mut table = [[[[[[0; SUB_BLOCK_COEFFS]; SIG_PATTERNS]; 2]; 2]; SIG_CLASSES]; SCAN_KINDS];
    let mut k = 0;
    while k < SCAN_KINDS {
        let mut class = 0;
        while class < SIG_CLASSES {
            let log2 = CLASS_LOG2[class];
            let mut luma = 0;
            while luma < 2 {
                let mut first = 0;
                while first < 2 {
                    let mut pattern = 0;
                    while pattern < SIG_PATTERNS {
                        let mut index = 0;
                        while index < SUB_BLOCK_COEFFS {
                            let position = SUB_BLOCK_ORDER[k][index] as usize;
                            let (xp, yp) = (position & 3, position >> 2);
                            // Any sub-block other than the first will do:
                            // only whether it is the first matters (a 4x4
                            // transform has only the first).
                            let x = if first == 1 || log2 == 2 { xp } else { xp + 4 };
                            let inc = sig_ctx_inc(
                                if log2 == 2 { None } else { Some(pattern) },
                                x,
                                yp,
                                log2,
                                luma == 1,
                                KINDS[k],
                            );
                            let base = if luma == 1 { 0 } else { ctx::SIG_CHROMA };
                            table[k][class][luma][first][pattern][index] = (base + inc) as u8;
                            index += 1;
                        }
                        pattern += 1;
                    }
                    first += 1;
                }
                luma += 1;
            }
            class += 1;
        }
        k += 1;
    }
    table
};

/// `greater1Ctx` after a `coeff_abs_level_greater1_flag` of 0 (index 0) or
/// 1 (index 1), by its value before (§9.3.4.2.6): a 1 ends the run at 0
/// for the rest of the sub-block, a 0 counts up to 3 unless already 0.
const NEXT_GREATER1: [[u8; 4]; 2] = [[0, 2, 3, 3], [0; 4]];
/// No `coeff_abs_level_greater1_flag` of the sub-block was 1 (yet).
const NO_GREATER1: usize = usize::MAX;

/// What `residual_coding()` needs besides the scan.
#[derive(Debug, Clone, Copy)]
pub(super) struct Options {
    pub luma: bool,
    /// `sign_data_hiding_enabled_flag && !cu_transquant_bypass_flag`.
    pub sign_hiding: bool,
    /// `transform_skip_enabled_flag && !cu_transquant_bypass_flag`.
    pub transform_skip: bool,
}

/// Parse one transform block's levels into `levels` (raster, all zero on
/// entry) and list the non-zero positions in `positions`. Returns
/// `transform_skip_flag`.
pub(super) fn read(
    engine: &mut Engine<'_>,
    ctx: &mut Contexts,
    scan: &Scan,
    options: Options,
    levels: &mut [i32],
    positions: &mut Vec<u16>,
) -> Result<bool, DecodeError> {
    // The engine works on a local copy, written back at the end: its
    // range, window and counters then stay in registers through the
    // block's bins instead of going through memory at each one.
    let mut local = engine.clone();
    let skip = read_levels(&mut local, ctx, scan, options, levels, positions)?;
    *engine = local;
    Ok(skip)
}

/// [`read`] on the local engine.
fn read_levels(
    e: &mut Engine<'_>,
    ctx: &mut Contexts,
    scan: &Scan,
    options: Options,
    levels: &mut [i32],
    positions: &mut Vec<u16>,
) -> Result<bool, DecodeError> {
    let luma = options.luma;
    let skip = options.transform_skip
        && scan.log2 == TRANSFORM_SKIP_LOG2
        && e.bin(ctx.at(ctx::TRANSFORM_SKIP, usize::from(!luma)));
    let side = 1usize << (scan.log2 - 2);
    let last = last_position(e, ctx, scan, luma)?;
    let last_group = last / SUB_BLOCK_COEFFS;
    let sig_cases =
        &SIG_TABLE[scan.kind as usize][sig_class(scan.log2, scan.kind)][usize::from(luma)];
    let mut coded = [false; 64];
    let mut greater1_state = 1;
    for group in (0..=last_group).rev() {
        let (xs, ys) = scan.sub_blocks[group];
        let flag = if group == last_group || group == 0 {
            true
        } else {
            let base = if luma { 0 } else { ctx::CODED_SUB_BLOCK_CHROMA };
            let inc = base + usize::from(neighbour_coded(&coded, side, xs, ys));
            e.bin(ctx.at(ctx::CODED_SUB_BLOCK, inc))
        };
        coded[ys * side + xs] = flag;
        if !flag {
            continue;
        }
        let pattern = sig_pattern(&coded, side, xs, ys).unwrap_or(0);
        let sig_contexts = &sig_cases[usize::from(xs + ys == 0)][pattern];
        // Significant scan indices within the sub-block, highest first.
        let mut found = [0u8; SUB_BLOCK_COEFFS];
        let mut count = 0;
        let top = if group == last_group {
            found[0] = (last % SUB_BLOCK_COEFFS) as u8;
            count = 1;
            last % SUB_BLOCK_COEFFS
        } else {
            SUB_BLOCK_COEFFS
        };
        for index in (1..top).rev() {
            let significant = e.bin(ctx.at(ctx::SIG_COEFF, usize::from(sig_contexts[index])));
            // Written unconditionally, kept only when significant: no
            // branch on a flag the predictor cannot guess.
            found[count] = index as u8;
            count += usize::from(significant);
        }
        if top > 0 {
            // The sub-block's DC is inferred significant when nothing else
            // in a coded (non-first, non-last) sub-block was (§7.4.9.11).
            let inferred = group > 0 && group < last_group && count == 0;
            let significant =
                inferred || e.bin(ctx.at(ctx::SIG_COEFF, usize::from(sig_contexts[0])));
            found[count] = 0;
            count += usize::from(significant);
        }
        if count == 0 {
            continue;
        }
        let found = &found[..count];
        let hidden =
            options.sign_hiding && usize::from(found[0] - found[count - 1]) > SIGN_HIDING_THRESHOLD;
        let first = group * SUB_BLOCK_COEFFS;
        sub_block_levels(
            e,
            ctx,
            found,
            group,
            luma,
            hidden,
            &mut greater1_state,
            |index, level| {
                let raster = scan.positions[first + usize::from(index)];
                levels[raster] = level;
                positions.push(raster as u16);
            },
        )?;
    }
    Ok(skip)
}

/// `last_sig_coeff_{x,y}_{prefix,suffix}` as a scan position.
fn last_position(
    e: &mut Engine<'_>,
    ctx: &mut Contexts,
    scan: &Scan,
    luma: bool,
) -> Result<usize, DecodeError> {
    let log2 = u32::from(scan.log2);
    let size = 1usize << log2;
    // §9.3.4.2.3: ctxOffset and ctxShift.
    let (offset, shift) = if luma {
        (
            (3 * (log2 - 2) + ((log2 - 1) >> 2)) as usize,
            (log2 + 1) >> 2,
        )
    } else {
        (ctx::LAST_CHROMA, log2 - 2)
    };
    let mut prefix = [0u32; 2];
    for (axis, element) in [ctx::LAST_X, ctx::LAST_Y].into_iter().enumerate() {
        while prefix[axis] < 2 * log2 - 1
            && e.bin(ctx.at(element, offset + (prefix[axis] >> shift) as usize))
        {
            prefix[axis] += 1;
        }
    }
    let mut xy = [0usize; 2];
    for axis in 0..2 {
        let g = prefix[axis];
        let suffix = if g > 3 {
            e.bypass_bits((g >> 1) - 1)
        } else {
            0
        };
        xy[axis] = (MIN_IN_GROUP[g as usize] + suffix) as usize;
        if xy[axis] >= size {
            return Err(DecodeError::Malformed("last coefficient position"));
        }
    }
    if scan.kind == ScanKind::Vertical {
        xy.swap(0, 1);
    }
    // Both coordinates are below `size`, so the raster index is in range.
    Ok(usize::from(scan.index_of[xy[1] * size + xy[0]]))
}

/// `coeff_abs_level_remaining` with Rice parameter `rice` (§9.3.3.11): a
/// unary prefix, then the suffix bits in one bypass read.
fn remaining(e: &mut Engine<'_>, rice: u32) -> Result<u32, DecodeError> {
    let prefix = e.bypass_unary(RICE_PREFIX + MAX_SUFFIX + 1);
    if prefix > RICE_PREFIX + MAX_SUFFIX {
        return Err(DecodeError::Malformed("coefficient remainder prefix"));
    }
    if prefix < RICE_PREFIX {
        Ok((prefix << rice) + e.bypass_bits(rice))
    } else {
        let length = rice + prefix - RICE_PREFIX;
        if length > MAX_SUFFIX {
            return Err(DecodeError::Malformed("coefficient remainder suffix"));
        }
        Ok((2 << rice) + (1 << length) + e.bypass_bits(length))
    }
}

/// The levels of one sub-block's significant coefficients (scan indices
/// `found` within the sub-block, highest first): greater-1/2 flags, signs,
/// remainders.
#[allow(clippy::too_many_arguments)]
fn sub_block_levels(
    e: &mut Engine<'_>,
    ctx: &mut Contexts,
    found: &[u8],
    group: usize,
    luma: bool,
    hidden: bool,
    greater1_state: &mut usize,
    mut store: impl FnMut(u8, i32),
) -> Result<(), DecodeError> {
    // ctxSet (§9.3.4.2.6): 2 for luma sub-blocks other than the first,
    // plus one when the previous sub-block ended with a greater-1 flag.
    let set = usize::from(group > 0 && luma) * 2 + usize::from(*greater1_state == 0);
    let g1_base = if luma { 0 } else { ctx::GREATER1_CHROMA } + 4 * set;
    let mut magnitude = [1u32; SUB_BLOCK_COEFFS];
    let mut first_greater1 = NO_GREATER1;
    let mut state = 1;
    for (i, m) in magnitude[..found.len().min(MAX_GREATER1)]
        .iter_mut()
        .enumerate()
    {
        let greater1 = e.bin(ctx.at(ctx::GREATER1, g1_base + usize::from(state)));
        *m += u32::from(greater1);
        if greater1 && first_greater1 == NO_GREATER1 {
            first_greater1 = i;
        }
        state = NEXT_GREATER1[usize::from(greater1)][usize::from(state)];
    }
    *greater1_state = usize::from(state);
    if first_greater1 != NO_GREATER1 {
        let base = if luma { 0 } else { ctx::GREATER2_CHROMA };
        magnitude[first_greater1] += u32::from(e.bin(ctx.at(ctx::GREATER2, base + set)));
    }
    let signed = found.len() - usize::from(hidden);
    let signs = e.bypass_bits(signed as u32);
    if first_greater1 != NO_GREATER1 || found.len() > MAX_GREATER1 {
        let mut rice = 0;
        for (i, m) in magnitude[..found.len()].iter_mut().enumerate() {
            // baseLevel at which a remainder follows (§7.3.8.11).
            let threshold = if i >= MAX_GREATER1 {
                1
            } else if i == first_greater1 {
                3
            } else {
                2
            };
            if *m == threshold {
                *m += remaining(e, rice)?;
                if *m > 3 << rice {
                    rice = (rice + 1).min(MAX_RICE);
                }
            }
        }
    }
    let mut sum = 0u32;
    for (i, &index) in found.iter().enumerate() {
        let m = magnitude[i];
        if m > MAX_MAGNITUDE {
            return Err(DecodeError::Malformed("coefficient magnitude"));
        }
        sum += m;
        let negative = if i < signed {
            (signs >> (signed - 1 - i)) & 1
        } else {
            // The hidden sign: odd sum of magnitudes means negative.
            sum & 1
        };
        // Two's complement negation by mask: no branch on the sign.
        let mask = negative.wrapping_neg() as i32;
        let level = (m as i32 ^ mask) - mask;
        if level == MAX_MAGNITUDE as i32 {
            return Err(DecodeError::Malformed("coefficient magnitude"));
        }
        store(index, level);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_block_order_is_the_4x4_scan() {
        for (kind, order) in [ScanKind::Diagonal, ScanKind::Horizontal, ScanKind::Vertical]
            .into_iter()
            .zip(SUB_BLOCK_ORDER)
        {
            let scan = Scan::new(2, kind);
            let expect: Vec<u8> = scan.positions.iter().map(|&p| p as u8).collect();
            assert_eq!(expect, order, "{kind:?}");
        }
    }
}
