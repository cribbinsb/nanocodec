//! NEON intra estimate statistics: up to 32 bytes of a row per step, the
//! even samples unzipped (`uzp1`) into one 16-byte vector, every square as
//! `umull` of the absolute differences (`uabd`), pairwise-accumulated into
//! 32 bits (`uadalp`).

use core::arch::aarch64::{
    uint8x16_t, uint32x4_t, vabdq_u8, vaddvq_u32, vandq_u8, vdupq_n_u8, vdupq_n_u32, vget_low_u8,
    vld1q_u8, vmull_high_u8, vmull_u8, vpadalq_u16, vpaddlq_u8, vuzp1q_u8,
};

use super::{EstimateStats, SAMPLE_STEP};

/// Bytes of a row per step.
const CHUNK: usize = 32;
/// Bytes of one vector.
const VECTOR: usize = 16;

/// The even samples of `bytes` (`n` of them, 4 to 32; the rest zero), in
/// the low `n / 2` bytes of a vector.
#[inline]
fn even(bytes: &[u8], n: usize) -> uint8x16_t {
    let mut low = [0u8; VECTOR];
    let take = n.min(VECTOR);
    // SAFETY: NEON is enabled (callers' target feature); the loads read
    // checked 16-byte arrays or subslices.
    unsafe {
        let (first, second) = if n >= CHUNK {
            (
                vld1q_u8(bytes[..VECTOR].as_ptr()),
                vld1q_u8(bytes[VECTOR..CHUNK].as_ptr()),
            )
        } else if n == VECTOR {
            (vld1q_u8(bytes[..VECTOR].as_ptr()), vdupq_n_u8(0))
        } else {
            low[..take].copy_from_slice(&bytes[..take]);
            (vld1q_u8(low.as_ptr()), vdupq_n_u8(0))
        };
        vuzp1q_u8(first, second)
    }
}

/// `acc` plus the squares of `a - b`, pairwise in 32 bits.
#[inline]
fn add_squares(acc: uint32x4_t, a: uint8x16_t, b: uint8x16_t) -> uint32x4_t {
    // SAFETY: NEON is enabled (callers' target feature); registers only.
    unsafe {
        let d = vabdq_u8(a, b);
        let acc = vpadalq_u16(acc, vmull_u8(vget_low_u8(d), vget_low_u8(d)));
        vpadalq_u16(acc, vmull_high_u8(d, d))
    }
}

/// The NEON [`super::StatsFn`].
///
/// # Safety
///
/// None beyond NEON, which `AArch64` always has.
#[target_feature(enable = "neon")]
pub(super) unsafe fn stats(src: &[u8], stride: usize, n: usize, edge: &[u8]) -> EstimateStats {
    assert!(super::ESTIMATE_SIDES.contains(&n) && edge.len() > 3 * n);
    let width = n.min(CHUNK);
    let zero = vdupq_n_u8(0);
    // The lanes that hold samples (the horizontal prediction's splat is
    // masked to them; every other load leaves the rest zero).
    let valid = even(&[u8::MAX; CHUNK], width);
    let (mut sum, mut squares) = (vdupq_n_u32(0), vdupq_n_u32(0));
    let mut sse = [vdupq_n_u32(0); 4];
    for r in (0..n).step_by(SAMPLE_STEP) {
        let left = vandq_u8(vdupq_n_u8(edge[n - 1 - r]), valid);
        for c0 in (0..n).step_by(CHUNK) {
            let v = even(&src[r * stride + c0..], width);
            sum = vpadalq_u16(sum, vpaddlq_u8(v));
            squares = add_squares(squares, v, zero);
            let predictions = [
                even(&edge[n + 1 + c0..], width),
                left,
                even(&edge[n + 2 + r + c0..], width),
                even(&edge[n - r + c0..], width),
            ];
            for (acc, p) in sse.iter_mut().zip(predictions) {
                *acc = add_squares(*acc, v, p);
            }
        }
    }
    EstimateStats {
        sum: vaddvq_u32(sum),
        squares: vaddvq_u32(squares),
        sse: sse.map(|lanes| vaddvq_u32(lanes)),
    }
}
