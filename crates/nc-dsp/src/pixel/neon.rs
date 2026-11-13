//! NEON pixel kernels (`AArch64`, where Advanced SIMD is always present).
//!
//! SAD uses `vabd` with pairwise widening accumulation (`vpadal`) into one
//! 16-bit accumulator per 16-byte column, widened to 32 bits once per block
//! (overflow bound at [`sad_wide`]); widths 4 and 8 accumulate `vabal` into
//! one 16-bit accumulator and reduce once per block (Jetson: 8x8 9.0 → 7.7
//! ns, 8x32 34.1 → 26.1, four-candidate 8x8 42.0 → 29.9). SATD mirrors the AVX2
//! structure: vertical butterflies across row registers, then horizontal
//! stages on pairs of rows. Here pairwise add/sub is built from `uzp1`/`uzp2`
//! (even/odd lanes), which yields exactly the lane order of x86 `phaddw`.
//!
//! Kernel contract: see the `pixel` module docs. Every function carries
//! `#[target_feature(enable = "neon")]` (NEON is baseline on `AArch64`, but
//! intrinsics are only safe to call from functions that list it); then only
//! loads need `unsafe`.

use core::arch::aarch64::{
    int16x8_t, uint8x8_t, uint16x8_t, vabal_u8, vabdq_u8, vabsq_s16, vaddlvq_u16, vaddq_s16,
    vaddq_u16, vaddvq_u32, vbslq_u32, vcltq_u32, vcombine_s16, vcreate_u8, vdupq_n_u8, vdupq_n_u16,
    vdupq_n_u32, vget_low_s16, vgetq_lane_u32, vld1_u8, vld1q_u8, vld1q_u32, vminq_u32, vpadalq_u8,
    vpadalq_u16, vpaddlq_u16, vpmaxq_u16, vreinterpretq_s16_u16, vreinterpretq_u16_s16, vst1q_u32,
    vsubl_u8, vsubq_s16, vuzp1q_s16, vuzp2q_s16,
};

use super::{GRID_BLOCKS, GridBest, PixelFns, normalise_satd_4x4, normalise_satd_8x8};

/// The NEON kernel table. NEON is always available on `AArch64`.
pub(crate) fn pixel_fns() -> PixelFns {
    let mut functions = PixelFns {
        sad: [
            sad_w4,
            sad_w8,
            sad_wide::<16>,
            sad_wide::<32>,
            sad_wide::<64>,
        ],
        sad_x4: [
            sad_x4_narrow::<4>,
            sad_x4_narrow::<8>,
            sad_x4_wide::<16>,
            sad_x4_wide::<32>,
            sad_x4_wide::<64>,
        ],
        sad_quads_8x8_row,
        sad_4x4_macroblock,
        sad_scan_16x16_even_rows,
        sad_grid_best_64x64: [sad_grid_best_64x64::<1>, sad_grid_best_64x64::<2>],
        sad_grid_best_x4: None,
        sad_quads_16x16,
        satd_4x4: satd_4x4_tiles,
        satd_8x8: satd_8x8_tiles,
        satd_8x8_quads: None,
    };
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        use super::neon_dotprod as dot;
        functions.sad_grid_best_x4 = Some([dot::sad_grid_best_x4::<1>, dot::sad_grid_best_x4::<2>]);
        functions.sad_scan_16x16_even_rows = dot::sad_scan_16x16_even_rows;
        functions.sad_x4[2] = dot::sad_x4_wide::<16>;
        functions.sad_x4[3] = dot::sad_x4_wide::<32>;
        functions.sad_x4[4] = dot::sad_x4_wide::<64>;
        functions.sad_grid_best_64x64 =
            [dot::sad_grid_best_64x64::<1>, dot::sad_grid_best_64x64::<2>];
    }
    functions
}

// ---------------------------------------------------------------- loads ----

/// Two 4-byte rows packed into one 8-lane vector.
///
/// # Safety
///
/// Rows 0 and 1 (4 bytes each, `stride` apart) must be readable at `p`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_4x2(p: *const u8, stride: usize) -> uint8x8_t {
    // SAFETY: guaranteed by the caller; unaligned reads.
    let (lo, hi) = unsafe {
        (
            p.cast::<u32>().read_unaligned(),
            p.add(stride).cast::<u32>().read_unaligned(),
        )
    };
    vcreate_u8(u64::from(lo) | (u64::from(hi) << 32))
}

/// A `W`-wide row group for the narrow widths: two rows of 4, or one row of 8.
///
/// # Safety
///
/// The rows read must be in bounds (see `load_4x2` and `vld1_u8`).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_narrow<const W: usize>(p: *const u8, stride: usize) -> uint8x8_t {
    // SAFETY: forwarded caller guarantee.
    unsafe {
        if W == 4 {
            load_4x2(p, stride)
        } else {
            vld1_u8(p)
        }
    }
}

const fn narrow_rows_per_load(width: usize) -> usize {
    if width == 4 { 2 } else { 1 }
}

// ------------------------------------------------------------------ SAD ----

#[target_feature(enable = "neon")]
unsafe fn sad_w4(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize, h: usize) -> u32 {
    // SAFETY: forwarded kernel contract.
    unsafe { sad_narrow::<4>(src, src_stride, rf, ref_stride, h) }
}

#[target_feature(enable = "neon")]
unsafe fn sad_w8(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize, h: usize) -> u32 {
    // SAFETY: forwarded kernel contract.
    unsafe { sad_narrow::<8>(src, src_stride, rf, ref_stride, h) }
}

/// SAD for widths 4 and 8 (one 8-lane vector per load).
///
/// `vabal_u8` (`uabal`) adds each load's eight absolute differences into
/// u16 lanes, one instruction per load, and the block reduces once with
/// `vaddlvq_u16`. Bound: a lane gains at most 255 per load and a block is
/// at most 64 loads (`MAX_BLOCK_HEIGHT`), 64 * 255 = 16320 < 65535.
#[target_feature(enable = "neon")]
unsafe fn sad_narrow<const W: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    h: usize,
) -> u32 {
    debug_assert!(h <= MAX_BLOCK_HEIGHT);
    let mut acc = vdupq_n_u16(0);
    for y in (0..h).step_by(narrow_rows_per_load(W)) {
        // SAFETY: the rows loaded are inside the block (kernel contract).
        let (a, b) = unsafe {
            (
                load_narrow::<W>(src.as_ptr().add(y * src_stride), src_stride),
                load_narrow::<W>(rf.as_ptr().add(y * ref_stride), ref_stride),
            )
        };
        acc = vabal_u8(acc, a, b);
    }
    vaddlvq_u16(acc)
}

/// Largest block height (kernel contract), used for the overflow bound below.
const MAX_BLOCK_HEIGHT: usize = 64;

/// Columns of 16 bytes in the widest (64-sample) block.
pub(super) const MAX_COLUMNS: usize = 4;

/// SAD for widths 16, 32 and 64.
///
/// Each 16-byte column keeps its own 16-bit accumulator, widened to 32 bits
/// once at the end of the block. Bound: `vpadalq_u8` adds at most
/// 2 * 255 = 510 to a lane per row, so after `MAX_BLOCK_HEIGHT` (64) rows a
/// lane holds at most 64 * 510 = 32640 < 65535.
///
/// Measured on the Orin (Cortex-A78AE): this is no faster than widening every
/// row (64x64: 191 ns either way). Both are load-throughput bound and scale
/// linearly with block area (32x32: 47 ns). The modest speed-up over scalar
/// at 64x64 (1.8x) is because the compiler auto-vectorises the scalar
/// reference well there, not because this kernel is slow. Kept because one
/// widening per block is the simpler invariant.
#[target_feature(enable = "neon")]
unsafe fn sad_wide<const W: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    h: usize,
) -> u32 {
    debug_assert!(h <= MAX_BLOCK_HEIGHT);
    let mut cols = [vdupq_n_u16(0); MAX_COLUMNS];
    for y in 0..h {
        for (c, col) in cols.iter_mut().enumerate().take(W / 16) {
            // SAFETY: bytes 16c..16c + 16 of row y are inside the block
            // (kernel contract).
            let (a, b) = unsafe {
                (
                    vld1q_u8(src.as_ptr().add(y * src_stride + 16 * c)),
                    vld1q_u8(rf.as_ptr().add(y * ref_stride + 16 * c)),
                )
            };
            *col = vpadalq_u8(*col, vabdq_u8(a, b));
        }
    }
    widen_and_sum(&cols[..W / 16])
}

/// Sum every lane of the 16-bit column accumulators.
#[inline]
#[target_feature(enable = "neon")]
fn widen_and_sum(cols: &[uint16x8_t]) -> u32 {
    let mut acc = vdupq_n_u32(0);
    for &col in cols {
        acc = vpadalq_u16(acc, col);
    }
    vaddvq_u32(acc)
}

/// Four-candidate SAD for widths 4 and 8; same accumulator scheme and
/// overflow bound as [`sad_narrow`].
#[target_feature(enable = "neon")]
unsafe fn sad_x4_narrow<const W: usize>(
    src: &[u8],
    src_stride: usize,
    refs: [&[u8]; 4],
    ref_stride: usize,
    h: usize,
) -> [u32; 4] {
    debug_assert!(h <= MAX_BLOCK_HEIGHT);
    let mut acc = [vdupq_n_u16(0); 4];
    for y in (0..h).step_by(narrow_rows_per_load(W)) {
        // SAFETY: the rows loaded are inside the block (kernel contract).
        let a = unsafe { load_narrow::<W>(src.as_ptr().add(y * src_stride), src_stride) };
        for (sum, rf) in acc.iter_mut().zip(refs) {
            // SAFETY: as above, for each candidate.
            let b = unsafe { load_narrow::<W>(rf.as_ptr().add(y * ref_stride), ref_stride) };
            *sum = vabal_u8(*sum, a, b);
        }
    }
    // Closures do not inherit target features, so no `map` here.
    [
        vaddlvq_u16(acc[0]),
        vaddlvq_u16(acc[1]),
        vaddlvq_u16(acc[2]),
        vaddlvq_u16(acc[3]),
    ]
}

/// Four-candidate SAD for widths 16, 32 and 64; same accumulator scheme and
/// overflow bound as [`sad_wide`].
#[target_feature(enable = "neon")]
unsafe fn sad_x4_wide<const W: usize>(
    src: &[u8],
    src_stride: usize,
    refs: [&[u8]; 4],
    ref_stride: usize,
    h: usize,
) -> [u32; 4] {
    debug_assert!(h <= MAX_BLOCK_HEIGHT);
    let mut cols = [[vdupq_n_u16(0); MAX_COLUMNS]; 4];
    for y in 0..h {
        for c in 0..W / 16 {
            // SAFETY: bytes 16c..16c + 16 of row y are inside the block.
            let a = unsafe { vld1q_u8(src.as_ptr().add(y * src_stride + 16 * c)) };
            for (cand, rf) in cols.iter_mut().zip(refs) {
                // SAFETY: as above, for each candidate.
                let b = unsafe { vld1q_u8(rf.as_ptr().add(y * ref_stride + 16 * c)) };
                cand[c] = vpadalq_u8(cand[c], vabdq_u8(a, b));
            }
        }
    }
    // Closures do not inherit target features, so no `map` here.
    [
        widen_and_sum(&cols[0][..W / 16]),
        widen_and_sum(&cols[1][..W / 16]),
        widen_and_sum(&cols[2][..W / 16]),
        widen_and_sum(&cols[3][..W / 16]),
    ]
}

/// [`sad_quads_8x8`] for each block along a row, two blocks per 16-byte
/// load.
#[target_feature(enable = "neon")]
unsafe fn sad_quads_8x8_row(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    out: &mut [[u32; 4]],
) {
    let pairs = out.len() / 2;
    for (p, pair) in out[..2 * pairs]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .enumerate()
    {
        // SAFETY: blocks 2p and 2p + 1 lie inside the row in both slices.
        let [a, b] =
            unsafe { sad_quads_8x8_pair(&src[16 * p..], src_stride, &rf[16 * p..], ref_stride) };
        *pair = [a, b];
    }
    if out.len() % 2 == 1 {
        let last = out.len() - 1;
        // SAFETY: the last block lies inside the row in both slices.
        out[last] =
            unsafe { sad_quads_8x8(&src[8 * last..], src_stride, &rf[8 * last..], ref_stride) };
    }
}

/// Quadrant SADs of two horizontally adjacent 8x8 blocks. Pair sums of a
/// 16-byte row's absolute differences put each 4-column quadrant in its own
/// two u16 lanes; rows 0..4 and 4..8 accumulate apart (at most 4 * 510).
///
/// # Safety
///
/// Kernel contract for a 16-sample-wide block.
#[target_feature(enable = "neon")]
unsafe fn sad_quads_8x8_pair(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
) -> [[u32; 4]; 2] {
    let mut halves = [vdupq_n_u16(0); 2];
    for y in 0..8 {
        // SAFETY: row y (16 bytes) is inside both blocks in both slices.
        let (a, b) = unsafe {
            (
                vld1q_u8(src.as_ptr().add(y * src_stride)),
                vld1q_u8(rf.as_ptr().add(y * ref_stride)),
            )
        };
        halves[y / 4] = vpadalq_u8(halves[y / 4], vabdq_u8(a, b));
    }
    // Lanes of the u32 view: [block 0 left, block 0 right, block 1 left,
    // block 1 right].
    let (top, bottom) = (vpaddlq_u16(halves[0]), vpaddlq_u16(halves[1]));
    [
        [
            vgetq_lane_u32::<0>(top),
            vgetq_lane_u32::<1>(top),
            vgetq_lane_u32::<0>(bottom),
            vgetq_lane_u32::<1>(bottom),
        ],
        [
            vgetq_lane_u32::<2>(top),
            vgetq_lane_u32::<3>(top),
            vgetq_lane_u32::<2>(bottom),
            vgetq_lane_u32::<3>(bottom),
        ],
    ]
}

/// Quadrant SADs of an 8x8 block. `vabal_u8` accumulates each column's
/// absolute differences into its own u16 lane (lanes 0..4: left quadrant,
/// 4..8: right), one accumulator per half of the block: at most 4 * 255 per
/// lane, no overflow.
#[target_feature(enable = "neon")]
pub(super) unsafe fn sad_quads_8x8(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
) -> [u32; 4] {
    let mut halves = [vdupq_n_u16(0); 2];
    for y in 0..8 {
        // SAFETY: row y (8 bytes) is inside the 8x8 blocks.
        let (a, b) = unsafe {
            (
                vld1_u8(src.as_ptr().add(y * src_stride)),
                vld1_u8(rf.as_ptr().add(y * ref_stride)),
            )
        };
        halves[y / 4] = vabal_u8(halves[y / 4], a, b);
    }
    quad_sums(halves)
}

/// Quadrant SADs of a 16x16 block. `vpadalq_u8` of a 16-byte row's
/// absolute differences leaves the left 8 bytes' pair sums in u16 lanes 0..4
/// and the right 8 bytes' in lanes 4..8. Each half of the block (8 rows) gets
/// its own accumulator: at most 8 * 510 = 4080 per lane, no overflow.
#[target_feature(enable = "neon")]
unsafe fn sad_quads_16x16(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize) -> [u32; 4] {
    let mut halves = [vdupq_n_u16(0); 2];
    for y in 0..16 {
        // SAFETY: row y (16 bytes) is inside the 16x16 blocks.
        let (a, b) = unsafe {
            (
                vld1q_u8(src.as_ptr().add(y * src_stride)),
                vld1q_u8(rf.as_ptr().add(y * ref_stride)),
            )
        };
        halves[y / 8] = vpadalq_u8(halves[y / 8], vabdq_u8(a, b));
    }
    quad_sums(halves)
}

/// [top-left, top-right, bottom-left, bottom-right] from two half-block
/// accumulators whose u16 lanes 0..4 hold the left quadrant's partial sums
/// and lanes 4..8 the right's.
#[inline]
#[target_feature(enable = "neon")]
fn quad_sums(halves: [uint16x8_t; 2]) -> [u32; 4] {
    // Lanes of the u32 view: [left, left, right, right] pair sums.
    let top = vpaddlq_u16(halves[0]);
    let bottom = vpaddlq_u16(halves[1]);
    [
        vgetq_lane_u32::<0>(top) + vgetq_lane_u32::<1>(top),
        vgetq_lane_u32::<2>(top) + vgetq_lane_u32::<3>(top),
        vgetq_lane_u32::<0>(bottom) + vgetq_lane_u32::<1>(bottom),
        vgetq_lane_u32::<2>(bottom) + vgetq_lane_u32::<3>(bottom),
    ]
}

/// SAD scan of a 16x16 block over rows 0, 2, ..., 14. The eight source rows
/// are loaded once; per position, `vpadalq_u8` accumulates pair sums (at
/// most 8 * 510 per lane, no overflow) and one widening add reduces them.
#[target_feature(enable = "neon")]
unsafe fn sad_scan_16x16_even_rows(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    out: &mut [u32],
) {
    let mut a = [vdupq_n_u8(0); 8];
    for (k, reg) in a.iter_mut().enumerate() {
        // SAFETY: row 2k (16 bytes) is inside the block.
        *reg = unsafe { vld1q_u8(src.as_ptr().add(2 * k * src_stride)) };
    }
    for (dx, sad) in out.iter_mut().enumerate() {
        let mut acc = vdupq_n_u16(0);
        for (k, &reg) in a.iter().enumerate() {
            // SAFETY: row 2k of the candidate at offset dx is inside the
            // reference (kernel contract: 16 + n - 1 columns).
            let b = unsafe { vld1q_u8(rf.as_ptr().add(2 * k * ref_stride + dx)) };
            acc = vpadalq_u8(acc, vabdq_u8(reg, b));
        }
        *sad = vaddlvq_u16(acc);
    }
}

/// The 8x8 SADs of a 64x64 block. Each 16-byte column of a row accumulates
/// pair sums with `vpadalq_u8`: u16 lanes 0..4 belong to the left 8x8 block,
/// 4..8 to the right one (at most 8 * 510 per lane over a block row).
#[target_feature(enable = "neon")]
unsafe fn sad_grid_8x8_in_64x64<const STEP: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
) -> [u32; 64] {
    let mut grid = [0u32; 64];
    for (block_row, out) in grid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut acc = [vdupq_n_u16(0); 4];
        for y in (8 * block_row..8 * block_row + 8).step_by(STEP) {
            for (c, sums) in acc.iter_mut().enumerate() {
                // SAFETY: bytes 16c..16c + 16 of row y are inside both 64x64
                // blocks (kernel contract).
                let (a, b) = unsafe {
                    (
                        vld1q_u8(src.as_ptr().add(y * src_stride + 16 * c)),
                        vld1q_u8(rf.as_ptr().add(y * ref_stride + 16 * c)),
                    )
                };
                *sums = vpadalq_u8(*sums, vabdq_u8(a, b));
            }
        }
        for (c, &sums) in acc.iter().enumerate() {
            // Lanes of the u32 view: [left, left, right, right] pair sums.
            let wide = vpaddlq_u16(sums);
            out[2 * c] = vgetq_lane_u32::<0>(wide) + vgetq_lane_u32::<1>(wide);
            out[2 * c + 1] = vgetq_lane_u32::<2>(wide) + vgetq_lane_u32::<3>(wide);
        }
    }
    grid
}

/// [`sad_grid_8x8_in_64x64`] summed into every block size and folded into
/// `best`.
#[target_feature(enable = "neon")]
unsafe fn sad_grid_best_64x64<const STEP: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    visible: &[u32; 64],
    best: &mut GridBest,
    tag: u32,
) {
    // SAFETY: the kernel contract covers both 64x64 blocks.
    let grid = unsafe { sad_grid_8x8_in_64x64::<STEP>(src, src_stride, rf, ref_stride) };
    keep_min(best, &super::grid_costs(&grid, visible, STEP as u32), tag);
}

/// Keep-minimum over a grid update, four blocks per step: `vminq_u32`, and
/// the tag taken from `tag` wherever the cost was lower.
#[target_feature(enable = "neon")]
pub(super) fn keep_min(best: &mut GridBest, cost: &[u32; GRID_BLOCKS], tag: u32) {
    let lanes = GRID_BLOCKS / 4 * 4;
    let fresh = vdupq_n_u32(tag);
    for i in (0..lanes).step_by(4) {
        // SAFETY: elements i..i + 4 are inside all three 85-element arrays.
        unsafe {
            let b = vld1q_u32(best.sad.as_ptr().add(i));
            let c = vld1q_u32(cost.as_ptr().add(i));
            let t = vld1q_u32(best.tag.as_ptr().add(i));
            vst1q_u32(best.sad.as_mut_ptr().add(i), vminq_u32(b, c));
            vst1q_u32(
                best.tag.as_mut_ptr().add(i),
                vbslq_u32(vcltq_u32(c, b), fresh, t),
            );
        }
    }
    super::scalar::keep_min(
        &mut best.sad[lanes..],
        &mut best.tag[lanes..],
        &cost[lanes..],
        tag,
    );
}

// ----------------------------------------------------------------- SATD ----

#[target_feature(enable = "neon")]
unsafe fn satd_4x4_tiles(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut total = 0;
    for ty in (0..h).step_by(4) {
        for tx in (0..w).step_by(4) {
            // SAFETY: the 4x4 tile at (tx, ty) is inside the block.
            let sum = unsafe {
                hadamard_4x4_abs_sum(
                    src.as_ptr().add(ty * src_stride + tx),
                    src_stride,
                    rf.as_ptr().add(ty * ref_stride + tx),
                    ref_stride,
                )
            };
            total += normalise_satd_4x4(sum);
        }
    }
    total
}

#[target_feature(enable = "neon")]
unsafe fn satd_8x8_tiles(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut total = 0;
    for ty in (0..h).step_by(8) {
        for tx in (0..w).step_by(8) {
            // SAFETY: the 8x8 tile at (tx, ty) is inside the block.
            let sum = unsafe {
                hadamard_8x8_abs_sum(
                    src.as_ptr().add(ty * src_stride + tx),
                    src_stride,
                    rf.as_ptr().add(ty * ref_stride + tx),
                    ref_stride,
                )
            };
            total += normalise_satd_8x8(sum);
        }
    }
    total
}

/// Unnormalised Hadamard absolute sum of one 4x4 difference tile.
///
/// # Safety
///
/// Four rows of four bytes must be readable at `src` and `rf`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hadamard_4x4_abs_sum(
    src: *const u8,
    src_stride: usize,
    rf: *const u8,
    ref_stride: usize,
) -> u32 {
    let mut d = [vreinterpretq_s16_u16(vdupq_n_u16(0)); 4];
    for (y, row) in d.iter_mut().enumerate() {
        // SAFETY: row y of the tile is readable (caller contract). Reading
        // 4 bytes and zero-extending keeps the upper lanes zero.
        let (a, b) = unsafe {
            (
                src.add(y * src_stride).cast::<u32>().read_unaligned(),
                rf.add(y * ref_stride).cast::<u32>().read_unaligned(),
            )
        };
        *row = vreinterpretq_s16_u16(vsubl_u8(vcreate_u8(u64::from(a)), vcreate_u8(u64::from(b))));
    }
    butterflies(&mut d, 1);
    butterflies(&mut d, 2);
    let a = vcombine_s16(vget_low_s16(d[0]), vget_low_s16(d[1]));
    let b = vcombine_s16(vget_low_s16(d[2]), vget_low_s16(d[3]));
    let (s1, d1) = pairwise_add_sub(a, b);
    // The last stage pairs adjacent lanes of `s1 | d1`.
    2 * vaddlvq_u16(last_stage_abs_max(s1, d1))
}

/// Unnormalised Hadamard absolute sum of one 8x8 difference tile.
///
/// # Safety
///
/// Eight rows of eight bytes must be readable at `src` and `rf`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hadamard_8x8_abs_sum(
    src: *const u8,
    src_stride: usize,
    rf: *const u8,
    ref_stride: usize,
) -> u32 {
    let mut d = [vreinterpretq_s16_u16(vdupq_n_u16(0)); 8];
    for (y, row) in d.iter_mut().enumerate() {
        // SAFETY: row y of the tile is readable (caller contract).
        let (a, b) = unsafe {
            (
                vld1_u8(src.add(y * src_stride)),
                vld1_u8(rf.add(y * ref_stride)),
            )
        };
        *row = vreinterpretq_s16_u16(vsubl_u8(a, b));
    }
    butterflies(&mut d, 1);
    butterflies(&mut d, 2);
    butterflies(&mut d, 4);
    // Values stay below 8 * 8 * 255 = 16320, well inside i16. The four
    // last-stage vectors hold pair maxima of at most 8160 each, so their
    // sum (<= 32640) stays in u16 lanes and widens once per tile.
    let mut acc = vdupq_n_u16(0);
    for pair in d.as_chunks::<2>().0 {
        let (s1, d1) = pairwise_add_sub(pair[0], pair[1]);
        let (s2, d2) = pairwise_add_sub(s1, d1);
        acc = vaddq_u16(acc, last_stage_abs_max(s2, d2));
    }
    2 * vaddlvq_u16(acc)
}

/// Half the absolute sum of the last Hadamard stage over `a | b`, whose
/// butterflies pair adjacent lanes, as eight u16 lanes: `|x + y| + |x − y|
/// = 2 · max(|x|, |y|)`, so the stage, both absolute values and their sum
/// become one pairwise maximum of the inputs' absolute values (at most
/// 8160 before the last stage of an 8x8 tile). With the dot-product
/// variance energy, h264-fast −0.2% (crowd) / −0.7% (bodycam) Jetson
/// cycles. x86's last stage is `phaddw`, where this saves little.
#[inline]
#[target_feature(enable = "neon")]
fn last_stage_abs_max(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    vpmaxq_u16(
        vreinterpretq_u16_s16(vabsq_s16(a)),
        vreinterpretq_u16_s16(vabsq_s16(b)),
    )
}

/// One butterfly stage across row registers: rows `i` and `i + half`.
#[inline]
#[target_feature(enable = "neon")]
fn butterflies<const N: usize>(rows: &mut [int16x8_t; N], half: usize) {
    for start in (0..N).step_by(2 * half) {
        for i in start..start + half {
            let (a, b) = (rows[i], rows[i + half]);
            rows[i] = vaddq_s16(a, b);
            rows[i + half] = vsubq_s16(a, b);
        }
    }
}

/// Sums and differences of adjacent lanes of `a` then `b`; the same lane
/// order as x86 `phaddw` / `phsubw`.
#[inline]
#[target_feature(enable = "neon")]
fn pairwise_add_sub(a: int16x8_t, b: int16x8_t) -> (int16x8_t, int16x8_t) {
    let (even, odd) = (vuzp1q_s16(a, b), vuzp2q_s16(a, b));
    (vaddq_s16(even, odd), vsubq_s16(even, odd))
}

/// The 4x4 SADs of a macroblock from this ISA's row kernel.
#[target_feature(enable = "neon")]
unsafe fn sad_4x4_macroblock(
    luma: super::BlockPair<'_>,
    chroma: [super::BlockPair<'_>; 2],
) -> ([[u32; 4]; 4], [[u32; 4]; 2]) {
    // SAFETY: the row kernel's ISA is this module's (checked by the
    // dispatcher), and the slices satisfy the kernel contract.
    unsafe { super::macroblock_from_rows(sad_quads_8x8_row, luma, chroma) }
}
