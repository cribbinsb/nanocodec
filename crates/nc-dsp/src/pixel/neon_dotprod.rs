//! NEON SAD kernels accumulating with the dot-product extension, installed
//! after runtime detection.
//!
//! `udot(|a − b|, 1)` sums four absolute differences into a u32 lane per
//! instruction. Unlike `uadalp` (pairwise accumulate, one pipe on the
//! Cortex-A78), `udot` issues on both vector pipes, and its four-byte
//! groups are halves of the 8x8 block columns the grid reports, so a row
//! of blocks needs one pairwise add instead of lane extractions. Sums stay
//! far below 2^32.
//!
//! The four-candidate SAD of blocks 16 to 64 wide (`sad_x4_wide`) runs
//! `udot` too, one u32 accumulator per candidate and 16-byte column; the
//! single-candidate SAD stays on `uadalp` (`neon::sad_wide`), where `udot`
//! measured within noise (Jetson, 16x16 to 64x64: −2.1% to +1.3%) while
//! the four-candidate form gained 16x8 −7.4%, 16x16 −6.8%, 32x32 −2.5%,
//! 64x64 −12.0% (38.2 → 35.4, 60.2 → 56.1, 185.9 → 181.2, 726 → 639 ns).
//!
//! Jetson (`nc-checkasm --bench`, ns): 33-position scan 225 → 179, grid
//! update 250 → 224 (every row) and 168 → 157 (even rows); h264-fast
//! crowd / bodycam −1.4% / −1.7% cycles (jperf, 5 rounds). The 4x4 quadrant SADs
//! (`sad_quads_8x8_row`) were slower this way (the macroblock form 60 → 66
//! ns; stores of whole result vectors feed scalar reads) and keep `uadalp`.
//!
//! Kernel contract: see the `pixel` module docs.

use core::arch::aarch64::{
    uint8x16_t, uint32x4_t, vabdq_u8, vaddvq_u32, vdupq_n_u8, vdupq_n_u32, vld1q_u8, vpaddq_u32,
    vst1q_u32,
};

use super::GridBest;
use super::neon::MAX_COLUMNS;
use crate::simd::neon_dot::udot;

/// Positions of a scan measured together: two four-result pairwise trees
/// reduce the independent accumulation chains.
const SCAN_BATCH: usize = 8;

/// `acc + Σ |a − b|` over each four-byte group of the 16 lanes.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn sad_acc(acc: uint32x4_t, a: uint8x16_t, b: uint8x16_t) -> uint32x4_t {
    udot(acc, vabdq_u8(a, b), vdupq_n_u8(1))
}

/// SAD scan of a 16x16 block over rows 0, 2, ..., 14, [`SCAN_BATCH`]
/// positions at a time. Eight independent dot-product chains cover latency
/// while requiring only one accumulator per candidate. Two pairwise trees
/// reduce the eight results without scalar lane extraction. Jetson, paired
/// locked timings for 33 positions: 186 → 176 ns (about 5.5% faster).
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn sad_scan_16x16_even_rows(
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
    let batches = out.len() / SCAN_BATCH;
    for (batch, sads) in out[..batches * SCAN_BATCH]
        .as_chunks_mut::<SCAN_BATCH>()
        .0
        .iter_mut()
        .enumerate()
    {
        let mut acc = [vdupq_n_u32(0); SCAN_BATCH];
        for (k, &reg) in a.iter().enumerate() {
            for (i, sums) in acc.iter_mut().enumerate() {
                let dx = SCAN_BATCH * batch + i;
                // SAFETY: row 2k of the candidate at offset dx is inside the
                // reference (kernel contract: 16 + n - 1 columns).
                let b = unsafe { vld1q_u8(rf.as_ptr().add(2 * k * ref_stride + dx)) };
                *sums = sad_acc(*sums, reg, b);
            }
        }
        for (group, out) in acc
            .as_chunks::<4>()
            .0
            .iter()
            .zip(sads.as_chunks_mut::<4>().0)
        {
            let total = vpaddq_u32(
                vpaddq_u32(group[0], group[1]),
                vpaddq_u32(group[2], group[3]),
            );
            // SAFETY: each output group holds four u32 results.
            unsafe { vst1q_u32(out.as_mut_ptr(), total) };
        }
    }
    for (dx, sad) in out.iter_mut().enumerate().skip(batches * SCAN_BATCH) {
        let mut acc = vdupq_n_u32(0);
        for (k, &reg) in a.iter().enumerate() {
            // SAFETY: as above.
            let b = unsafe { vld1q_u8(rf.as_ptr().add(2 * k * ref_stride + dx)) };
            acc = sad_acc(acc, reg, b);
        }
        *sad = vaddvq_u32(acc);
    }
}

/// Four-candidate SAD for widths 16, 32 and 64: one `udot` accumulator per
/// candidate and 16-byte column (independent dependency chains), summed
/// once per block. A u32 lane gains at most 4 * 255 per row, far below
/// 2^32 over 64 rows.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn sad_x4_wide<const W: usize>(
    src: &[u8],
    src_stride: usize,
    refs: [&[u8]; 4],
    ref_stride: usize,
    h: usize,
) -> [u32; 4] {
    let mut cols = [[vdupq_n_u32(0); MAX_COLUMNS]; 4];
    for y in 0..h {
        for c in 0..W / 16 {
            // SAFETY: bytes 16c..16c + 16 of row y are inside the block.
            let a = unsafe { vld1q_u8(src.as_ptr().add(y * src_stride + 16 * c)) };
            for (cand, rf) in cols.iter_mut().zip(refs) {
                // SAFETY: as above, for each candidate.
                let b = unsafe { vld1q_u8(rf.as_ptr().add(y * ref_stride + 16 * c)) };
                cand[c] = sad_acc(cand[c], a, b);
            }
        }
    }
    // Closures do not inherit target features, so no `map` here.
    [
        sum_columns(&cols[0][..W / 16]),
        sum_columns(&cols[1][..W / 16]),
        sum_columns(&cols[2][..W / 16]),
        sum_columns(&cols[3][..W / 16]),
    ]
}

/// Sum every lane of a candidate's column accumulators.
#[inline]
#[target_feature(enable = "neon")]
fn sum_columns(cols: &[uint32x4_t]) -> u32 {
    let mut acc = vdupq_n_u32(0);
    for &col in cols {
        acc = core::arch::aarch64::vaddq_u32(acc, col);
    }
    vaddvq_u32(acc)
}

/// The 8x8 SADs of a 64x64 block (rows stepping by `STEP`). Per block row,
/// each 16-byte column accumulates `[left, left, right, right]` group sums;
/// one pairwise add of two columns gives four blocks in order.
#[target_feature(enable = "neon,dotprod")]
unsafe fn sad_grid_8x8_in_64x64<const STEP: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
) -> [u32; 64] {
    let mut grid = [0u32; 64];
    for (block_row, out) in grid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut acc = [vdupq_n_u32(0); 4];
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
                *sums = sad_acc(*sums, a, b);
            }
        }
        // SAFETY: `out` holds eight u32.
        unsafe {
            vst1q_u32(out.as_mut_ptr(), vpaddq_u32(acc[0], acc[1]));
            vst1q_u32(out.as_mut_ptr().add(4), vpaddq_u32(acc[2], acc[3]));
        }
    }
    grid
}

/// [`sad_grid_8x8_in_64x64`] summed into every block size and folded into
/// `best`.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn sad_grid_best_64x64<const STEP: usize>(
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
    super::neon::keep_min(best, &super::grid_costs(&grid, visible, STEP as u32), tag);
}

/// Reuse each source load across four candidate grids. Four columns per
/// candidate keep independent dot-product chains; reductions remain ordered
/// so equal costs retain exactly the same vector as the single-position path.
/// Jetson locked paired timings: four even-row candidates 711 → 615 ns,
/// exact rows 1063 → 897 ns. The accumulation loop has no register spills.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn sad_grid_best_x4<const STEP: usize>(
    src: &[u8],
    src_stride: usize,
    refs: [&[u8]; 4],
    ref_stride: usize,
    visible: &[u32; 64],
    best: &mut GridBest,
    tags: [u32; 4],
) {
    let mut grids = [[0u32; 64]; 4];
    for block_row in 0..8 {
        let mut acc = [[vdupq_n_u32(0); 4]; 4];
        for y in (8 * block_row..8 * block_row + 8).step_by(STEP) {
            for (column, sums) in acc.iter_mut().enumerate() {
                // SAFETY: each source row covers all 64 pixels.
                let source = unsafe { vld1q_u8(src.as_ptr().add(y * src_stride + 16 * column)) };
                for candidate in 0..4 {
                    // SAFETY: each reference covers a complete 64x64 block.
                    let reference = unsafe {
                        vld1q_u8(refs[candidate].as_ptr().add(y * ref_stride + 16 * column))
                    };
                    sums[candidate] = sad_acc(sums[candidate], source, reference);
                }
            }
        }
        for candidate in 0..4 {
            let out = &mut grids[candidate][8 * block_row..8 * block_row + 8];
            let sums = acc.map(|column| column[candidate]);
            // SAFETY: this block row has eight writable u32 entries.
            unsafe {
                vst1q_u32(out.as_mut_ptr(), vpaddq_u32(sums[0], sums[1]));
                vst1q_u32(out.as_mut_ptr().add(4), vpaddq_u32(sums[2], sums[3]));
            }
        }
    }
    for (grid, tag) in grids.iter().zip(tags) {
        super::neon::keep_min(best, &super::grid_costs(grid, visible, STEP as u32), tag);
    }
}
