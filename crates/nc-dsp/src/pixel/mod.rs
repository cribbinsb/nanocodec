//! Pixel-difference kernels: SAD, 4-candidate SAD and SATD.
//!
//! These are the cost functions of motion estimation and mode decision.
//!
//! - **SAD:** sum of absolute differences.
//! - **SAD x4:** the SAD of one source block against four candidate
//!   positions in the same reference plane. Motion search evaluates
//!   candidates in groups, so loading the source once pays.
//! - **SAD quads (8x8, 16x16):** the SADs of a block's four quadrants
//!   (top-left, top-right, bottom-left, bottom-right) in one pass; they sum to
//!   the block's SAD. Pre-analysis uses the 8x8 form, along a whole row of
//!   blocks per call, to spot small changes inside a block; motion search
//!   uses the 16x16 form to see how well a vector fits each 8x8 child.
//! - **SAD scan (16x16, even rows):** the SADs of one 16x16 source block
//!   against `n` horizontally consecutive reference positions, over rows
//!   0, 2, ..., 14 only. Hierarchical motion search scans its window one row
//!   of positions per call, so the per-call cost is paid once per row.
//! - **SAD grid best update (64x64):** the SADs of every square block of a
//!   64x64 block (64 8x8, 16 16x16, 4 32x32 and the 64x64 itself, 85 in
//!   all; see [`GRID_BLOCKS`]) at one reference position, folded into a
//!   running best per block: where a block's SAD beats its best, the best
//!   and its tag (the caller's vector) are replaced. SVT-AV1's full-pel
//!   search kernel; doing the sums and the selection inside the kernel
//!   keeps them in SIMD registers (on the Orin the separate update cost as
//!   much as the SADs).
//! - **SATD:** sum of absolute Hadamard-transformed differences, a cheap
//!   estimate of coded residual cost. Computed on 8x8 tiles when both block
//!   edges are at least 8, otherwise on 4x4 tiles. Tile sums are normalised
//!   as in the HEVC reference software: `(sum + 1) >> 1` for 4x4 and
//!   `(sum + 2) >> 2` for 8x8, so costs are comparable across tile sizes.
//!
//! # Kernel contract
//!
//! Every kernel is an `unsafe fn`. Callers (only [`crate::Dsp`]) guarantee:
//!
//! 1. each slice holds at least `(h - 1) * stride + w` bytes, i.e. it covers
//!    the whole block (`PlaneView` guarantees this); for the row kernel, the
//!    block is 8 rows of `8 * out.len()` samples; for the scan kernel, the
//!    reference covers 16 rows of `16 + out.len() - 1` samples; for the grid
//!    update, both cover a 64x64 block;
//! 2. `w` and `h` are supported block edges (4..=64, powers of two);
//! 3. the CPU supports the kernel's ISA ([`crate::Isa::is_supported`]).
//!
//! The Hadamard butterflies may be ordered differently per ISA: any ordering
//! yields the same coefficients up to sign and position, so the absolute sum
//! is identical. `nc-checkasm` verifies this.

pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx512;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx512_satd;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon_dotprod;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

/// SAD of a block whose width is fixed by the kernel; `h` rows.
pub(crate) type SadFn =
    unsafe fn(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize, h: usize) -> u32;

/// SAD of one source block against four references sharing one stride.
pub(crate) type SadX4Fn = unsafe fn(
    src: &[u8],
    src_stride: usize,
    refs: [&[u8]; 4],
    ref_stride: usize,
    h: usize,
) -> [u32; 4];

/// Quadrant SADs of a 16x16 block: [top-left, top-right, bottom-left,
/// bottom-right].
pub(crate) type SadQuadsFn =
    unsafe fn(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize) -> [u32; 4];

/// Quadrant SADs of each 8x8 block along a row: block i covers columns
/// 8i..8i+8.
pub(crate) type SadQuads8x8RowFn =
    unsafe fn(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize, out: &mut [[u32; 4]]);

/// One plane of a [`SadMacroblockFn`] call: source and reference with their
/// strides.
pub(crate) type BlockPair<'a> = (&'a [u8], usize, &'a [u8], usize);

/// The 4x4 SADs of a 4:2:0 macroblock: per luma 8x8 quadrant (raster) its
/// four 4x4 SADs, and per chroma plane the four 4x4 SADs of its 8x8 block
/// (each `[top-left, top-right, bottom-left, bottom-right]`).
pub(crate) type SadMacroblockFn =
    unsafe fn(luma: BlockPair<'_>, chroma: [BlockPair<'_>; 2]) -> ([[u32; 4]; 4], [[u32; 4]; 2]);

/// [`SadMacroblockFn`] from a [`SadQuads8x8RowFn`]: two rows of two luma
/// 8x8 blocks, then each chroma block.
///
/// # Safety
///
/// As for `row`'s ISA; the slices satisfy the kernel contract.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
#[inline]
pub(crate) unsafe fn macroblock_from_rows(
    row: SadQuads8x8RowFn,
    (src, src_stride, rf, ref_stride): BlockPair<'_>,
    chroma: [BlockPair<'_>; 2],
) -> ([[u32; 4]; 4], [[u32; 4]; 2]) {
    let mut luma = [[0u32; 4]; 4];
    let (top, bottom) = luma.split_at_mut(2);
    let mut planes = [[[0u32; 4]; 1]; 2];
    // SAFETY: forwarded from the caller (a 16x16 luma block is two rows of
    // two 8x8 blocks, each chroma block one).
    unsafe {
        row(src, src_stride, rf, ref_stride, top);
        row(
            &src[8 * src_stride..],
            src_stride,
            &rf[8 * ref_stride..],
            ref_stride,
            bottom,
        );
        for ((s, ss, r, rs), out) in chroma.into_iter().zip(&mut planes) {
            row(s, ss, r, rs, out);
        }
    }
    (luma, planes.map(|p| p[0]))
}

/// SATD over a `w` x `h` block, tiled by the kernel's transform size.
pub(crate) type SatdFn = unsafe fn(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32;

/// Independent 8x8 SATDs of a 16x16 block, in raster quadrant order.
pub(crate) type SatdQuadsFn =
    unsafe fn(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize) -> [u32; 4];

/// SADs of a 16x16 source block, rows 0, 2, ..., 14, against reference
/// blocks at `out.len()` consecutive x positions starting at `rf`.
pub(crate) type SadScan16x16Fn =
    unsafe fn(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize, out: &mut [u32]);

/// Blocks of a 64x64 grid update, in this order: 64 8x8 (row-major), then
/// 16 16x16, 4 32x32 (row-major) and the 64x64 block.
pub const GRID_BLOCKS: usize = 85;

/// Which rows of each 8x8 block a grid update reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridRows {
    /// Every row: exact SADs.
    All = 0,
    /// Rows 0, 2, 4, 6 only, doubled: about half the cost, an estimate (as
    /// SVT-AV1's subsampled searches).
    Even = 1,
}

/// Running bests of a grid update: SAD and tag per block (see
/// [`GRID_BLOCKS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridBest {
    /// Best SAD per block (`u32::MAX` before any update).
    pub sad: [u32; GRID_BLOCKS],
    /// Tag of the update that gave each best.
    pub tag: [u32; GRID_BLOCKS],
}

impl Default for GridBest {
    fn default() -> Self {
        Self {
            sad: [u32::MAX; GRID_BLOCKS],
            tag: [0; GRID_BLOCKS],
        }
    }
}

/// Fold the SADs of a 64x64 block at one position into `best`: 8x8 SADs
/// at masked-out blocks (`visible[i] == 0`) count as zero.
pub(crate) type SadGridBestFn = unsafe fn(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    visible: &[u32; 64],
    best: &mut GridBest,
    tag: u32,
);

/// The shared middle of every ISA's grid update: scale (row subsampling)
/// and mask the 8x8 SADs and sum
/// them into 16x16, 32x32 and 64x64 SADs, in [`GridBest`] order. Each ISA
/// then keeps the minimum with its own SIMD (`keep_min`): left to the
/// compiler, that step cost as much as the SADs on the Orin.
// Always inlined: it must be compiled inside each ISA's kernel, with that
// ISA's vector instructions; out of line it is built for the baseline
// target only (measured: 15% of motion-search instructions on x86).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn grid_costs(grid: &[u32; 64], visible: &[u32; 64], scale: u32) -> [u32; GRID_BLOCKS] {
    // Written as constant-length loops over whole rows (eight 8x8 SADs per
    // row) so the compiler keeps them in vector registers: add two rows,
    // then adjacent pairs. (`array::from_fn` with closures was not inlined.)
    let mut rows = [[0u32; 8]; 8];
    for (r, row) in rows.iter_mut().enumerate() {
        for (c, value) in row.iter_mut().enumerate() {
            *value = (grid[8 * r + c] * scale) & visible[8 * r + c];
        }
    }
    let mut s16 = [[0u32; 4]; 4];
    for (r, out) in s16.iter_mut().enumerate() {
        let mut sum = [0u32; 8];
        for (c, value) in sum.iter_mut().enumerate() {
            *value = rows[2 * r][c] + rows[2 * r + 1][c];
        }
        for (c, value) in out.iter_mut().enumerate() {
            *value = sum[2 * c] + sum[2 * c + 1];
        }
    }
    let mut s32 = [[0u32; 2]; 2];
    for (r, out) in s32.iter_mut().enumerate() {
        for (c, value) in out.iter_mut().enumerate() {
            *value = s16[2 * r][2 * c]
                + s16[2 * r][2 * c + 1]
                + s16[2 * r + 1][2 * c]
                + s16[2 * r + 1][2 * c + 1];
        }
    }
    let mut cost = [0u32; GRID_BLOCKS];
    for (dst, src) in cost[..64].as_chunks_mut::<8>().0.iter_mut().zip(&rows) {
        *dst = *src;
    }
    for (dst, src) in cost[64..80].as_chunks_mut::<4>().0.iter_mut().zip(&s16) {
        *dst = *src;
    }
    cost[80..82].copy_from_slice(&s32[0]);
    cost[82..84].copy_from_slice(&s32[1]);
    cost[84] = s32[0][0] + s32[0][1] + s32[1][0] + s32[1][1];
    cost
}

/// Four grid positions, folded in input order to preserve ties.
pub(crate) type SadGridBestX4Fn = unsafe fn(
    src: &[u8],
    src_stride: usize,
    refs: [&[u8]; 4],
    ref_stride: usize,
    visible: &[u32; 64],
    best: &mut GridBest,
    tags: [u32; 4],
);

/// One ISA's implementations of every pixel kernel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PixelFns {
    /// Indexed by width class (4, 8, 16, 32, 64).
    pub sad: [SadFn; 5],
    /// Indexed by width class (4, 8, 16, 32, 64).
    pub sad_x4: [SadX4Fn; 5],
    pub sad_quads_8x8_row: SadQuads8x8RowFn,
    pub sad_scan_16x16_even_rows: SadScan16x16Fn,
    /// Indexed by `GridRows` (every row, every other row).
    pub sad_grid_best_64x64: [SadGridBestFn; 2],
    pub sad_grid_best_x4: Option<[SadGridBestX4Fn; 2]>,
    pub sad_quads_16x16: SadQuadsFn,
    pub sad_4x4_macroblock: SadMacroblockFn,
    pub satd_4x4: SatdFn,
    pub satd_8x8: SatdFn,
    pub satd_8x8_quads: Option<SatdQuadsFn>,
}

/// Normalisation of a 4x4 Hadamard absolute sum (HEVC reference software).
#[inline]
pub(crate) const fn normalise_satd_4x4(sum: u32) -> u32 {
    (sum + 1) >> 1
}

/// Normalisation of an 8x8 Hadamard absolute sum (HEVC reference software).
#[inline]
pub(crate) const fn normalise_satd_8x8(sum: u32) -> u32 {
    (sum + 2) >> 2
}
