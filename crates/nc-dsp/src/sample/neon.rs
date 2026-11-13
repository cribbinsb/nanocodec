//! NEON sample kernels (`AArch64`).
//!
//! Moments: pairwise widening adds for the sum, `vmull_u8` squares (at most
//! 255^2 = 65025, exact in u16) accumulated pairwise into u32. Downscale:
//! `vpaddlq_u8` / `vpadalq_u8` sum each 2x2 box in u16, and `vrshrn_n_u16`
//! performs exactly the `(sum + 2) >> 2` rounding; 16 outputs per iteration,
//! with the scalar row function for the tail.
//!
//! Kernel contract: see the `sample` module docs.

use core::arch::aarch64::{
    vaddlvq_u16, vaddvq_u32, vcombine_u8, vdupq_n_u16, vdupq_n_u32, vget_high_u8, vget_low_u8,
    vgetq_lane_u32, vld1_u8, vld1q_u8, vld2q_u8, vmull_u8, vpadalq_u8, vpadalq_u16, vpaddlq_u8,
    vpaddlq_u16, vrhaddq_u8, vrshrn_n_u16, vst1q_u8,
};

use super::{Moments, SampleFns, scalar};

/// Outputs produced per iteration of the downscale loop.
const DOWNSCALE_STEP: usize = 16;

/// The NEON kernel table. NEON is always available on `AArch64`.
pub(crate) fn sample_fns() -> SampleFns {
    let mut functions = SampleFns {
        moments_8x8_row: moments_row::<8>,
        moments_8x4_row: moments_row::<4>,
        downscale_2x,
        pair_mean_2x: pairs_2x::<false>,
        pair_pick_2x: pairs_2x::<true>,
    };
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        functions.moments_8x8_row = super::neon_dotprod::moments_row::<8>;
        functions.moments_8x4_row = super::neon_dotprod::moments_row::<4>;
    }
    functions
}

/// [`moments_block`] for each 8-wide, `ROWS`-tall block along a row, two
/// blocks per 16-byte load.
#[target_feature(enable = "neon")]
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
        let [a, b] = unsafe { moments_pair::<ROWS>(&src[16 * p..], src_stride) };
        *pair = [a, b];
    }
    if out.len() % 2 == 1 {
        let last = out.len() - 1;
        // SAFETY: the last block lies inside the row.
        out[last] = unsafe { moments_block::<ROWS>(&src[8 * last..], src_stride) };
    }
}

/// Moments of two horizontally adjacent 8-wide, `ROWS`-tall blocks. Pair sums of a
/// 16-byte row put block 0 in u16 lanes 0..4 and block 1 in lanes 4..8 (at
/// most 8 * 510 per lane); squares of each 8-byte half accumulate apart.
///
/// # Safety
///
/// Kernel contract for a 16-sample-wide block.
#[target_feature(enable = "neon")]
unsafe fn moments_pair<const ROWS: usize>(src: &[u8], src_stride: usize) -> [Moments; 2] {
    let mut sum = vdupq_n_u16(0);
    let mut sum_sq = [vdupq_n_u32(0); 2];
    for y in 0..ROWS {
        // SAFETY: row y (16 bytes) is inside the two blocks.
        let v = unsafe { vld1q_u8(src.as_ptr().add(y * src_stride)) };
        sum = vpadalq_u8(sum, v);
        let (lo, hi) = (vget_low_u8(v), vget_high_u8(v));
        sum_sq[0] = vpadalq_u16(sum_sq[0], vmull_u8(lo, lo));
        sum_sq[1] = vpadalq_u16(sum_sq[1], vmull_u8(hi, hi));
    }
    // Lanes of the u32 view: [block 0, block 0, block 1, block 1].
    let sums = vpaddlq_u16(sum);
    [
        Moments {
            sum: vgetq_lane_u32::<0>(sums) + vgetq_lane_u32::<1>(sums),
            sum_sq: vaddvq_u32(sum_sq[0]),
        },
        Moments {
            sum: vgetq_lane_u32::<2>(sums) + vgetq_lane_u32::<3>(sums),
            sum_sq: vaddvq_u32(sum_sq[1]),
        },
    ]
}

/// # Safety
///
/// Kernel contract (`sample` module docs).
#[target_feature(enable = "neon")]
unsafe fn moments_block<const ROWS: usize>(src: &[u8], src_stride: usize) -> Moments {
    let mut sum = vdupq_n_u16(0);
    let mut sum_sq = vdupq_n_u32(0);
    for y in (0..ROWS).step_by(2) {
        // SAFETY: rows y and y + 1 (8 bytes each) are inside the block.
        let v = unsafe {
            vcombine_u8(
                vld1_u8(src.as_ptr().add(y * src_stride)),
                vld1_u8(src.as_ptr().add((y + 1) * src_stride)),
            )
        };
        sum = vpadalq_u8(sum, v);
        let (lo, hi) = (vget_low_u8(v), vget_high_u8(v));
        sum_sq = vpadalq_u16(sum_sq, vmull_u8(lo, lo));
        sum_sq = vpadalq_u16(sum_sq, vmull_u8(hi, hi));
    }
    Moments {
        sum: vaddlvq_u16(sum),
        sum_sq: vaddvq_u32(sum_sq),
    }
}

/// [`scalar::pairs_2x`], 16 outputs per iteration: a de-interleaving load
/// splits 32 samples into left and right samples of each pair.
///
/// # Safety
///
/// Kernel contract (`sample` module docs).
#[target_feature(enable = "neon")]
unsafe fn pairs_2x<const PICK: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    let simd_width = w - w % DOWNSCALE_STEP;
    for y in 0..h {
        let row = &src[y * src_stride..];
        let out = &mut dst[y * dst_stride..y * dst_stride + w];
        for x in (0..simd_width).step_by(DOWNSCALE_STEP) {
            // SAFETY: input bytes 2x .. 2x + 32 lie within the 2w-wide source
            // row and outputs x .. x + 16 within the output row, since
            // x + 16 <= w (kernel contract).
            unsafe {
                let pairs = vld2q_u8(row.as_ptr().add(2 * x));
                let values = if PICK {
                    pairs.0
                } else {
                    vrhaddq_u8(pairs.0, pairs.1)
                };
                vst1q_u8(out.as_mut_ptr().add(x), values);
            }
        }
        scalar::pairs_2x_row::<PICK>(&row[2 * simd_width..], &mut out[simd_width..]);
    }
}

/// # Safety
///
/// Kernel contract (`sample` module docs).
#[target_feature(enable = "neon")]
unsafe fn downscale_2x(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    let simd_width = w - w % DOWNSCALE_STEP;
    for y in 0..h {
        let top = &src[2 * y * src_stride..];
        let bottom = &src[(2 * y + 1) * src_stride..];
        let out = &mut dst[y * dst_stride..y * dst_stride + w];
        for x in (0..simd_width).step_by(DOWNSCALE_STEP) {
            // SAFETY: input bytes 2x .. 2x + 32 lie within the 2w-wide source
            // rows and outputs x .. x + 16 within the output row, since
            // x + 16 <= w (kernel contract).
            unsafe {
                let t = top.as_ptr().add(2 * x);
                let b = bottom.as_ptr().add(2 * x);
                let left = vpadalq_u8(vpaddlq_u8(vld1q_u8(t)), vld1q_u8(b));
                let right = vpadalq_u8(vpaddlq_u8(vld1q_u8(t.add(16))), vld1q_u8(b.add(16)));
                let means = vcombine_u8(vrshrn_n_u16::<2>(left), vrshrn_n_u16::<2>(right));
                vst1q_u8(out.as_mut_ptr().add(x), means);
            }
        }
        scalar::downscale_2x_row(
            &top[2 * simd_width..],
            &bottom[2 * simd_width..],
            &mut out[simd_width..],
        );
    }
}
