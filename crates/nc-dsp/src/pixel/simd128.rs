//! WebAssembly SIMD128 pixel kernels.
//!
//! Compiled only when the crate is built with `-C target-feature=+simd128`;
//! WebAssembly has no run-time feature detection. The structure follows the
//! NEON kernels:
//!
//! - **SAD:** absolute difference as `sat(a - b) | sat(b - a)`, then
//!   pairwise widening sums accumulated in 16-bit lanes across rows and
//!   widened to 32 bits once per block (twice for 64x64; bound at
//!   `rows_per_widen`).
//! - **SATD:** vertical butterflies across rows, then horizontal stages from
//!   even/odd lane shuffles. These give the same lane order as x86
//!   `phaddw`.
//!
//! Kernel contract: see the `pixel` module docs. Arithmetic intrinsics are
//! safe inside `#[target_feature(enable = "simd128")]` functions; only
//! loads need `unsafe`.

use core::arch::wasm32::{
    i16x8_abs, i16x8_add, i16x8_shuffle, i16x8_splat, i16x8_sub, i32x4_add, u8x16_sub_sat,
    u16x8_add, u16x8_extadd_pairwise_u8x16, u16x8_extend_low_u8x16, u16x8_max, u16x8_splat, u32x4,
    u32x4_extadd_pairwise_u16x8, u32x4_extract_lane, u32x4_lt, u32x4_min, u32x4_splat, u64x2, v128,
    v128_bitselect, v128_load, v128_load64_zero, v128_or, v128_store,
};

use super::{GRID_BLOCKS, GridBest, PixelFns, normalise_satd_4x4, normalise_satd_8x8};

/// The SIMD128 kernel table (SIMD128 is a compile-time feature).
pub(crate) fn pixel_fns() -> PixelFns {
    PixelFns {
        // 4x4 SAD uses the scalar kernel: with +simd128 the compiler already
        // vectorises it, and it measured 12.5 ns vs 16.3 ns for the explicit
        // SIMD128 version (node 22, Ryzen 5950X). The explicit version stays
        // in use for the 4-wide SAD x4 (on par) and every wider block.
        sad: [
            super::scalar::sad::<4>,
            sad::<8>,
            sad::<16>,
            sad::<32>,
            sad::<64>,
        ],
        sad_x4: [
            sad_x4::<4>,
            sad_x4::<8>,
            sad_x4::<16>,
            sad_x4::<32>,
            sad_x4::<64>,
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
    }
}

// ---------------------------------------------------------------- loads ----

/// Rows covered by one load group of a `W`-wide block.
const fn rows_per_group(width: usize) -> usize {
    match width {
        4 => 4,
        8 => 2,
        _ => 1,
    }
}

/// # Safety
///
/// `N` bytes must be readable at `p`.
#[inline]
unsafe fn read<const N: usize>(p: *const u8) -> u64 {
    // SAFETY: guaranteed by the caller; unaligned reads.
    unsafe {
        match N {
            4 => u64::from(p.cast::<u32>().read_unaligned()),
            _ => p.cast::<u64>().read_unaligned(),
        }
    }
}

/// Load 16 bytes of a row group of a `W`-wide block: four rows of 4 bytes,
/// two rows of 8, or 16 bytes of one row starting at column `x`.
///
/// # Safety
///
/// The bytes read must be inside the block (kernel contract).
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn load_group<const W: usize>(p: *const u8, stride: usize, x: usize) -> v128 {
    // SAFETY: forwarded caller guarantee.
    unsafe {
        match W {
            4 => u32x4(
                read::<4>(p) as u32,
                read::<4>(p.add(stride)) as u32,
                read::<4>(p.add(2 * stride)) as u32,
                read::<4>(p.add(3 * stride)) as u32,
            ),
            8 => u64x2(read::<8>(p), read::<8>(p.add(stride))),
            _ => v128_load(p.add(x).cast()),
        }
    }
}

// ------------------------------------------------------------------ SAD ----

/// Per-lane |a - b| widened to eight u16 lanes.
#[target_feature(enable = "simd128")]
#[inline]
fn abs_diff_widened(a: v128, b: v128) -> v128 {
    u16x8_extadd_pairwise_u8x16(v128_or(u8x16_sub_sat(a, b), u8x16_sub_sat(b, a)))
}

/// 16-byte chunks per row of a `W`-wide block (one for widths 4 and 8,
/// whose row groups fill one register).
const fn chunks_per_group(width: usize) -> usize {
    if width <= 16 { 1 } else { width / 16 }
}

/// Largest pair sum of two byte differences: what one chunk adds to a u16
/// lane of [`abs_diff_widened`].
const PAIR_SUM_MAX: usize = 2 * u8::MAX as usize;

/// Rows of a `W`-wide block whose SADs fit the u16 accumulator lanes: a row
/// group adds at most `chunks_per_group(W) * 510` to a lane, so 128 groups
/// for widths up to 16, 64 for 32 and 32 for 64 (32 * 4 * 510 = 65280 <=
/// 65535). Blocks are at most 64 rows (kernel contract), so every width but
/// 64 fits one span and a 64-wide block needs at most two.
const fn rows_per_widen(width: usize) -> usize {
    u16::MAX as usize / (chunks_per_group(width) * PAIR_SUM_MAX) * rows_per_group(width)
}

/// Largest block height (kernel contract).
const MAX_BLOCK_HEIGHT: usize = 64;

/// SAD of one row group (all 16-byte chunks of it) as u16 pair sums.
///
/// # Safety
///
/// The row group at `src`/`rf` must be inside the block.
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn group_sad<const W: usize>(
    src: *const u8,
    src_stride: usize,
    rf: *const u8,
    ref_stride: usize,
) -> v128 {
    let mut row = u16x8_splat(0);
    for c in 0..chunks_per_group(W) {
        // SAFETY: forwarded caller guarantee.
        let (a, b) = unsafe {
            (
                load_group::<W>(src, src_stride, 16 * c),
                load_group::<W>(rf, ref_stride, 16 * c),
            )
        };
        row = u16x8_add(row, abs_diff_widened(a, b));
    }
    row
}

/// SAD of rows `top..end` (at most [`rows_per_widen`]) of a `W`-wide block
/// as u16 pair sums.
///
/// # Safety
///
/// The rows must be inside the block.
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn span_sad<const W: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    (top, end): (usize, usize),
) -> v128 {
    let mut narrow = u16x8_splat(0);
    for y in (top..end).step_by(rows_per_group(W)) {
        // SAFETY: rows y..y + rows_per_group(W) are inside the block.
        let part = unsafe {
            group_sad::<W>(
                src.as_ptr().add(y * src_stride),
                src_stride,
                rf.as_ptr().add(y * ref_stride),
                ref_stride,
            )
        };
        narrow = u16x8_add(narrow, part);
    }
    narrow
}

/// SAD of a `W`-wide block. Row-group sums accumulate in u16 lanes and widen
/// to u32 once per [`rows_per_widen`] rows, so once per block below 64x64
/// and twice for 64x64: per 16-pixel row that is two saturating subtracts,
/// an or, a pairwise widening add and a u16 add.
#[target_feature(enable = "simd128")]
unsafe fn sad<const W: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    h: usize,
) -> u32 {
    let span = rows_per_widen(W);
    debug_assert!(h <= MAX_BLOCK_HEIGHT && h <= 2 * span);
    // SAFETY: both spans are rows of the block.
    let mut acc = u32x4_extadd_pairwise_u16x8(unsafe {
        span_sad::<W>(src, src_stride, rf, ref_stride, (0, h.min(span)))
    });
    // Constant false below width 64, which keeps one span.
    if span < MAX_BLOCK_HEIGHT && h > span {
        // SAFETY: as above.
        let rest = unsafe { span_sad::<W>(src, src_stride, rf, ref_stride, (span, h)) };
        acc = i32x4_add(acc, u32x4_extadd_pairwise_u16x8(rest));
    }
    sum_u32_lanes(acc)
}

/// [`span_sad`] for four candidates sharing the source rows.
///
/// # Safety
///
/// The rows must be inside the block.
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn span_sad_x4<const W: usize>(
    src: &[u8],
    src_stride: usize,
    refs: [&[u8]; 4],
    ref_stride: usize,
    (top, end): (usize, usize),
) -> [v128; 4] {
    let mut narrow = [u16x8_splat(0); 4];
    for y in (top..end).step_by(rows_per_group(W)) {
        for (sum, rf) in narrow.iter_mut().zip(refs) {
            // SAFETY: rows y..y + rows_per_group(W) are inside the block.
            let part = unsafe {
                group_sad::<W>(
                    src.as_ptr().add(y * src_stride),
                    src_stride,
                    rf.as_ptr().add(y * ref_stride),
                    ref_stride,
                )
            };
            *sum = u16x8_add(*sum, part);
        }
    }
    narrow
}

/// Four-candidate SAD; the accumulation scheme of [`sad`] per candidate.
#[target_feature(enable = "simd128")]
unsafe fn sad_x4<const W: usize>(
    src: &[u8],
    src_stride: usize,
    refs: [&[u8]; 4],
    ref_stride: usize,
    h: usize,
) -> [u32; 4] {
    let span = rows_per_widen(W);
    debug_assert!(h <= MAX_BLOCK_HEIGHT && h <= 2 * span);
    // SAFETY: both spans are rows of the block.
    let first = unsafe { span_sad_x4::<W>(src, src_stride, refs, ref_stride, (0, h.min(span))) };
    let mut acc = [u32x4_splat(0); 4];
    for (wide, &sum) in acc.iter_mut().zip(&first) {
        *wide = u32x4_extadd_pairwise_u16x8(sum);
    }
    // Constant false below width 64, which keeps one span.
    if span < MAX_BLOCK_HEIGHT && h > span {
        // SAFETY: as above.
        let rest = unsafe { span_sad_x4::<W>(src, src_stride, refs, ref_stride, (span, h)) };
        for (wide, &sum) in acc.iter_mut().zip(&rest) {
            *wide = i32x4_add(*wide, u32x4_extadd_pairwise_u16x8(sum));
        }
    }
    // Closures do not inherit target features, so no `map` here.
    [
        sum_u32_lanes(acc[0]),
        sum_u32_lanes(acc[1]),
        sum_u32_lanes(acc[2]),
        sum_u32_lanes(acc[3]),
    ]
}

#[target_feature(enable = "simd128")]
#[inline]
fn sum_u32_lanes(v: v128) -> u32 {
    u32x4_extract_lane::<0>(v)
        + u32x4_extract_lane::<1>(v)
        + u32x4_extract_lane::<2>(v)
        + u32x4_extract_lane::<3>(v)
}

/// [`sad_quads_8x8`] for each block along a row.
#[target_feature(enable = "simd128")]
unsafe fn sad_quads_8x8_row(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    out: &mut [[u32; 4]],
) {
    let pairs = out.len() / 2;
    let (batches, tail) = out.as_chunks_mut::<2>();
    for (i, quads) in batches.iter_mut().enumerate() {
        // SAFETY: two adjacent blocks cover 16 bytes of each row in
        // both slices (kernel contract).
        *quads =
            unsafe { sad_quads_8x8_pair(&src[16 * i..], src_stride, &rf[16 * i..], ref_stride) };
    }
    if let Some(last) = tail.first_mut() {
        // SAFETY: the last 8x8 block is covered by both slices.
        *last = unsafe {
            sad_quads_8x8(
                &src[16 * pairs..],
                src_stride,
                &rf[16 * pairs..],
                ref_stride,
            )
        };
    }
}

/// Two adjacent blocks: pair sums of 16 byte differences accumulate in
/// u16 lanes, then four-column sums give each block's left/right quadrants.
#[target_feature(enable = "simd128")]
unsafe fn sad_quads_8x8_pair(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
) -> [[u32; 4]; 2] {
    let mut halves = [u16x8_splat(0); 2];
    for y in 0..8 {
        // SAFETY: both 8x8 blocks cover a 16-byte row (caller contract).
        let (a, b) = unsafe {
            (
                v128_load(src.as_ptr().add(y * src_stride).cast()),
                v128_load(rf.as_ptr().add(y * ref_stride).cast()),
            )
        };
        halves[y / 4] = u16x8_add(halves[y / 4], abs_diff_widened(a, b));
    }
    let top = u32x4_extadd_pairwise_u16x8(halves[0]);
    let bottom = u32x4_extadd_pairwise_u16x8(halves[1]);
    [
        [
            u32x4_extract_lane::<0>(top),
            u32x4_extract_lane::<1>(top),
            u32x4_extract_lane::<0>(bottom),
            u32x4_extract_lane::<1>(bottom),
        ],
        [
            u32x4_extract_lane::<2>(top),
            u32x4_extract_lane::<3>(top),
            u32x4_extract_lane::<2>(bottom),
            u32x4_extract_lane::<3>(bottom),
        ],
    ]
}

/// Quadrant SADs of an 8x8 block. Each row's absolute differences, widened
/// to u16, put column i in lane i (lanes 0..4: left quadrant, 4..8: right);
/// one accumulator per half of the block: at most 4 * 255 per lane.
#[target_feature(enable = "simd128")]
unsafe fn sad_quads_8x8(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize) -> [u32; 4] {
    let mut halves = [u16x8_splat(0); 2];
    for y in 0..8 {
        // SAFETY: row y (8 bytes) is inside the 8x8 blocks.
        let (a, b) = unsafe {
            (
                v128_load64_zero(src.as_ptr().add(y * src_stride).cast()),
                v128_load64_zero(rf.as_ptr().add(y * ref_stride).cast()),
            )
        };
        let diff = v128_or(u8x16_sub_sat(a, b), u8x16_sub_sat(b, a));
        halves[y / 4] = u16x8_add(halves[y / 4], u16x8_extend_low_u8x16(diff));
    }
    quad_sums(halves)
}

/// Quadrant SADs of a 16x16 block. The widened absolute differences of a
/// 16-byte row leave the left 8 bytes' pair sums in u16 lanes 0..4 and the
/// right 8 bytes' in lanes 4..8; each half of the block (8 rows) accumulates
/// separately: at most 8 * 510 = 4080 per lane.
#[target_feature(enable = "simd128")]
unsafe fn sad_quads_16x16(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize) -> [u32; 4] {
    let mut halves = [u16x8_splat(0); 2];
    for y in 0..16 {
        // SAFETY: row y (16 bytes) is inside the 16x16 blocks.
        let (a, b) = unsafe {
            (
                v128_load(src.as_ptr().add(y * src_stride).cast()),
                v128_load(rf.as_ptr().add(y * ref_stride).cast()),
            )
        };
        halves[y / 8] = u16x8_add(halves[y / 8], abs_diff_widened(a, b));
    }
    quad_sums(halves)
}

/// [top-left, top-right, bottom-left, bottom-right] from two half-block
/// accumulators whose u16 lanes 0..4 hold the left quadrant's partial sums
/// and lanes 4..8 the right's.
#[target_feature(enable = "simd128")]
#[inline]
fn quad_sums(halves: [v128; 2]) -> [u32; 4] {
    // Lanes of the u32 view: [left, left, right, right] pair sums.
    let top = u32x4_extadd_pairwise_u16x8(halves[0]);
    let bottom = u32x4_extadd_pairwise_u16x8(halves[1]);
    [
        u32x4_extract_lane::<0>(top) + u32x4_extract_lane::<1>(top),
        u32x4_extract_lane::<2>(top) + u32x4_extract_lane::<3>(top),
        u32x4_extract_lane::<0>(bottom) + u32x4_extract_lane::<1>(bottom),
        u32x4_extract_lane::<2>(bottom) + u32x4_extract_lane::<3>(bottom),
    ]
}

/// SAD scan of a 16x16 block over rows 0, 2, ..., 14: the eight source rows
/// are loaded once and reused for every position; pair sums accumulate in
/// u16 lanes (at most 8 * 510).
#[target_feature(enable = "simd128")]
unsafe fn sad_scan_16x16_even_rows(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    out: &mut [u32],
) {
    let mut a = [u16x8_splat(0); 8];
    for (k, reg) in a.iter_mut().enumerate() {
        // SAFETY: row 2k (16 bytes) is inside the block.
        *reg = unsafe { v128_load(src.as_ptr().add(2 * k * src_stride).cast()) };
    }
    for (dx, sad) in out.iter_mut().enumerate() {
        let mut acc = u16x8_splat(0);
        for (k, &reg) in a.iter().enumerate() {
            // SAFETY: row 2k of the candidate at offset dx is inside the
            // reference (kernel contract: 16 + n - 1 columns).
            let b = unsafe { v128_load(rf.as_ptr().add(2 * k * ref_stride + dx).cast()) };
            acc = u16x8_add(acc, abs_diff_widened(reg, b));
        }
        let wide = u32x4_extadd_pairwise_u16x8(acc);
        *sad = u32x4_extract_lane::<0>(wide)
            + u32x4_extract_lane::<1>(wide)
            + u32x4_extract_lane::<2>(wide)
            + u32x4_extract_lane::<3>(wide);
    }
}

/// The 8x8 SADs of a 64x64 block. Each 16-byte column of a row accumulates
/// widened pair sums: u16 lanes 0..4 belong to the left 8x8 block, 4..8 to
/// the right one (at most 8 * 510 per lane over a block row).
#[target_feature(enable = "simd128")]
unsafe fn sad_grid_8x8_in_64x64<const STEP: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
) -> [u32; 64] {
    let mut grid = [0u32; 64];
    for (block_row, out) in grid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut acc = [u16x8_splat(0); 4];
        for y in (8 * block_row..8 * block_row + 8).step_by(STEP) {
            for (c, sums) in acc.iter_mut().enumerate() {
                // SAFETY: bytes 16c..16c + 16 of row y are inside both 64x64
                // blocks (kernel contract).
                let (a, b) = unsafe {
                    (
                        v128_load(src.as_ptr().add(y * src_stride + 16 * c).cast()),
                        v128_load(rf.as_ptr().add(y * ref_stride + 16 * c).cast()),
                    )
                };
                *sums = u16x8_add(*sums, abs_diff_widened(a, b));
            }
        }
        for (c, &sums) in acc.iter().enumerate() {
            // Lanes of the u32 view: [left, left, right, right] pair sums.
            let wide = u32x4_extadd_pairwise_u16x8(sums);
            out[2 * c] = u32x4_extract_lane::<0>(wide) + u32x4_extract_lane::<1>(wide);
            out[2 * c + 1] = u32x4_extract_lane::<2>(wide) + u32x4_extract_lane::<3>(wide);
        }
    }
    grid
}

/// [`sad_grid_8x8_in_64x64`] summed into every block size and folded into
/// `best`.
#[target_feature(enable = "simd128")]
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

/// Keep-minimum over a grid update, four blocks per step: `u32x4_min`, and
/// the tag taken from `tag` wherever the cost was lower.
#[target_feature(enable = "simd128")]
fn keep_min(best: &mut GridBest, cost: &[u32; GRID_BLOCKS], tag: u32) {
    let lanes = GRID_BLOCKS / 4 * 4;
    let fresh = u32x4_splat(tag);
    for i in (0..lanes).step_by(4) {
        // SAFETY: elements i..i + 4 are inside all three 85-element arrays.
        unsafe {
            let b = v128_load(best.sad.as_ptr().add(i).cast());
            let c = v128_load(cost.as_ptr().add(i).cast());
            let t = v128_load(best.tag.as_ptr().add(i).cast());
            v128_store(best.sad.as_mut_ptr().add(i).cast(), u32x4_min(b, c));
            v128_store(
                best.tag.as_mut_ptr().add(i).cast(),
                v128_bitselect(fresh, t, u32x4_lt(c, b)),
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

#[target_feature(enable = "simd128")]
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
                hadamard_abs_sum::<4>(
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

#[target_feature(enable = "simd128")]
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
                hadamard_abs_sum::<8>(
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

/// Unnormalised Hadamard absolute sum of one `N` x `N` tile (`N` = 4 or 8).
///
/// # Safety
///
/// `N` rows of `N` bytes must be readable at `src` and `rf`.
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn hadamard_abs_sum<const N: usize>(
    src: *const u8,
    src_stride: usize,
    rf: *const u8,
    ref_stride: usize,
) -> u32 {
    let mut d = [i16x8_splat(0); 8];
    for (y, row) in d.iter_mut().take(N).enumerate() {
        // SAFETY: row y of the tile is readable (caller contract); the
        // 4-byte read zero-extends, keeping unused lanes zero.
        let (a, b) = unsafe {
            (
                read::<N>(src.add(y * src_stride)),
                read::<N>(rf.add(y * ref_stride)),
            )
        };
        *row = i16x8_sub(
            u16x8_extend_low_u8x16(u64x2(a, 0)),
            u16x8_extend_low_u8x16(u64x2(b, 0)),
        );
    }
    let mut half = 1;
    while half < N {
        butterflies(&mut d, N, half);
        half *= 2;
    }
    // Each last-stage vector holds pair maxima of at most 8160 (see
    // `last_stage_abs_max`); an 8x8 tile adds four, at most 32640, so the
    // sum stays in u16 lanes and widens once per tile.
    let mut acc = u16x8_splat(0);
    if N == 4 {
        // Two rows per register; one pairwise stage, then the last.
        let a = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(d[0], d[1]);
        let b = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(d[2], d[3]);
        let (s1, d1) = pairwise_add_sub(a, b);
        acc = last_stage_abs_max(s1, d1);
    } else {
        for pair in d.as_chunks::<2>().0 {
            let (s1, d1) = pairwise_add_sub(pair[0], pair[1]);
            let (s2, d2) = pairwise_add_sub(s1, d1);
            acc = u16x8_add(acc, last_stage_abs_max(s2, d2));
        }
    }
    2 * sum_u32_lanes(u32x4_extadd_pairwise_u16x8(acc))
}

/// Half the absolute sum of the last Hadamard stage over `a | b`, whose
/// butterflies pair adjacent lanes: `|x + y| + |x - y| = 2 * max(|x|, |y|)`,
/// so the stage, both absolute values and their sum become one maximum of
/// the even and odd lanes' absolute values (at most 8 * 4 * 255 = 8160
/// before the last stage of an 8x8 tile). The NEON kernels do the same with
/// `umaxp`.
#[target_feature(enable = "simd128")]
#[inline]
fn last_stage_abs_max(a: v128, b: v128) -> v128 {
    let (a, b) = (i16x8_abs(a), i16x8_abs(b));
    let even = i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(a, b);
    let odd = i16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(a, b);
    u16x8_max(even, odd)
}

/// One butterfly stage across the first `n` row registers.
#[target_feature(enable = "simd128")]
#[inline]
fn butterflies(rows: &mut [v128; 8], n: usize, half: usize) {
    for start in (0..n).step_by(2 * half) {
        for i in start..start + half {
            let (a, b) = (rows[i], rows[i + half]);
            rows[i] = i16x8_add(a, b);
            rows[i + half] = i16x8_sub(a, b);
        }
    }
}

/// Sums and differences of adjacent lanes of `a` then `b` (x86 `phaddw` order).
#[target_feature(enable = "simd128")]
#[inline]
fn pairwise_add_sub(a: v128, b: v128) -> (v128, v128) {
    let even = i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(a, b);
    let odd = i16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(a, b);
    (i16x8_add(even, odd), i16x8_sub(even, odd))
}

/// The 4x4 SADs of a macroblock from this ISA's row kernel.
unsafe fn sad_4x4_macroblock(
    luma: super::BlockPair<'_>,
    chroma: [super::BlockPair<'_>; 2],
) -> ([[u32; 4]; 4], [[u32; 4]; 2]) {
    // SAFETY: the row kernel's ISA is this module's (checked by the
    // dispatcher), and the slices satisfy the kernel contract.
    unsafe { super::macroblock_from_rows(sad_quads_8x8_row, luma, chroma) }
}
