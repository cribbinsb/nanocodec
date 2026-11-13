//! NEON moments with the dot-product extension, installed after runtime
//! detection.
//!
//! `udot(v, 1)` sums and `udot(v, v)` squares four samples into a u32 lane
//! per instruction, and both issue on either vector pipe of the
//! Cortex-A78, where the pairwise widening adds of the generic kernel
//! (`uadalp`, three per 16 samples) share one pipe. Lanes 0..2 of each
//! accumulator hold block 0's halves and lanes 2..4 block 1's, so a pair of
//! blocks needs one pairwise add at the end. Sums stay far below 2^32.
//!
//! Kernel contract: see the `sample` module docs.

use core::arch::aarch64::{
    vaddq_u32, vdupq_n_u8, vdupq_n_u32, vgetq_lane_u32, vld1q_u8, vpaddq_u32,
};

use super::{Moments, neon};
use crate::simd::neon_dot::udot;

/// [`neon`]'s moments row (8-wide, `ROWS`-tall blocks), two blocks per
/// 16-byte load.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn moments_row<const ROWS: usize>(
    src: &[u8],
    src_stride: usize,
    out: &mut [Moments],
) {
    let pairs = out.len() / 2;
    for (p, pair) in out[..2 * pairs]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .enumerate()
    {
        // SAFETY: blocks 2p and 2p + 1 lie inside the row (kernel contract).
        *pair = unsafe { moments_pair::<ROWS>(&src[16 * p..], src_stride) };
    }
    if out.len() % 2 == 1 {
        let last = out.len() - 1;
        // SAFETY: forwarded from the caller (the last block lies inside
        // the row).
        unsafe { neon::moments_row::<ROWS>(&src[8 * last..], src_stride, &mut out[last..]) };
    }
}

/// Moments of two horizontally adjacent 8-wide, `ROWS`-tall blocks; even
/// and odd rows
/// accumulate apart, so four independent chains cover the latency.
///
/// # Safety
///
/// Kernel contract for a 16-sample-wide block.
#[target_feature(enable = "neon,dotprod")]
unsafe fn moments_pair<const ROWS: usize>(src: &[u8], src_stride: usize) -> [Moments; 2] {
    let ones = vdupq_n_u8(1);
    let mut sum = [vdupq_n_u32(0); 2];
    let mut sum_sq = [vdupq_n_u32(0); 2];
    for y in 0..ROWS {
        // SAFETY: row y (16 bytes) is inside the two blocks.
        let v = unsafe { vld1q_u8(src.as_ptr().add(y * src_stride)) };
        sum[y % 2] = udot(sum[y % 2], v, ones);
        sum_sq[y % 2] = udot(sum_sq[y % 2], v, v);
    }
    // [block 0 sum, block 1 sum, block 0 sum_sq, block 1 sum_sq].
    let totals = vpaddq_u32(vaddq_u32(sum[0], sum[1]), vaddq_u32(sum_sq[0], sum_sq[1]));
    [
        Moments {
            sum: vgetq_lane_u32::<0>(totals),
            sum_sq: vgetq_lane_u32::<2>(totals),
        },
        Moments {
            sum: vgetq_lane_u32::<1>(totals),
            sum_sq: vgetq_lane_u32::<3>(totals),
        },
    ]
}
