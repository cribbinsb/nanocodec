//! Optional HEVC luma horizontal filtering with signed NEON dot products,
//! and the dot-product row filter ([`DotFilter`]) the AV1 kernels
//! (`av1_neon`) share.
//!
//! AV1 regular taps are halved; HEVC taps are used directly. Both fit
//! signed bytes and sum to 64.
//! Flipping the source high bit makes bytes signed; adding 8192 to the dot
//! product restores the unsigned sum exactly. The existing two-stage AV1
//! rounding and widening vertical pass are retained. HEVC uses unrounded
//! horizontal sums and its six/twelve-bit final shifts. Native five-pair
//! 300-frame tests save 2.04%/1.51% HEVC CPU on bodycam/crowd with identical
//! streams; AV1 has no measured regression. No load extends beyond
//! the documented `w + 7` reference window, including width 4 and final rows.
//!
//! Installed only after runtime dot-product detection. Paired Orin timings:
//! AV1 CPU -0.79% bodycam / -0.44% crowd, with identical streams (five runs).

use core::arch::aarch64::{
    int8x16_t, int16x8_t, int32x4_t, uint8x16_t, vaddq_s16, vcombine_s16, vcombine_u8, vdup_n_s16,
    vdupq_lane_s32, vdupq_n_s16, vdupq_n_s32, vdupq_n_u8, veorq_u8, vget_high_s16, vget_lane_u32,
    vget_low_s16, vld1_s8, vld1_u8, vld1q_u8, vmlal_n_s16, vmovn_s32, vqmovun_s16, vqtbl1q_s8,
    vreinterpret_s32_s8, vreinterpret_u32_u8, vreinterpretq_s8_s32, vreinterpretq_s8_u8,
    vrshrn_n_s32, vshrq_n_s16, vst1_u8,
};

use super::{ConvolveFn, HEVC_LUMA, TAP_CENTRE, generic};
use crate::simd::neon::{I16x4, I16x8};

/// Rust's signed-dot intrinsic postdates our Rust1.90 minimum. Keep the
/// single instruction here until that intrinsic is available at the MSRV.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot(mut sum: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    // SAFETY: dotprod is enabled; register-only signed byte products accumulate
    // into four i32 lanes. No memory, stack or condition flags are accessed.
    unsafe {
        core::arch::asm!(
            "sdot {sum:v}.4s, {a:v}.16b, {b:v}.16b",
            sum = inout(vreg) sum,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    sum
}

/// HEVC's eight-tap luma coefficients also fit signed bytes and sum to 64.
pub(crate) fn hevc_luma_fns() -> [ConvolveFn; 5] {
    [
        predict::<4>,
        predict::<8>,
        predict::<16>,
        predict::<32>,
        predict::<64>,
    ]
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DotFilter {
    lo: int8x16_t,
    hi: int8x16_t,
    gather0: uint8x16_t,
    gather4: uint8x16_t,
    gather8: uint8x16_t,
}

impl DotFilter {
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn new<const W: usize>(taps: &[i16; 8]) -> Self {
        let bytes = taps.map(|tap| tap as i8);
        // SAFETY: exactly eight coefficient bytes are readable.
        let coefficients = vreinterpret_s32_s8(unsafe { vld1_s8(bytes.as_ptr()) });
        let gather0 = [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6];
        // Two overlapping eight-byte loads cover exactly 11 or 15 bytes.
        // Width 4 duplicates source 3..7; wider blocks duplicate source 7.
        let gather4 = if W == 4 {
            [4, 5, 6, 7, 5, 6, 7, 13, 6, 7, 13, 14, 7, 13, 14, 15]
        } else {
            [4, 5, 6, 7, 5, 6, 7, 9, 6, 7, 9, 10, 7, 9, 10, 11]
        };
        let gather8 = [
            9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14, 12, 13, 14, 15,
        ];
        // SAFETY: each shuffle mask has exactly sixteen initialized bytes.
        let (gather0, gather4, gather8) = unsafe {
            (
                vld1q_u8(gather0.as_ptr()),
                vld1q_u8(gather4.as_ptr()),
                vld1q_u8(gather8.as_ptr()),
            )
        };
        Self {
            lo: vreinterpretq_s8_s32(vdupq_lane_s32::<0>(coefficients)),
            hi: vreinterpretq_s8_s32(vdupq_lane_s32::<1>(coefficients)),
            gather0,
            gather4,
            gather8,
        }
    }

    /// First-stage AV1 `(unsigned halved-tap sum +2)>>2`, or the unrounded
    /// HEVC sum. HEVC positive coefficient sums are at most 88, so even
    /// the largest 8-bit result (22440) fits signed 16-bit lanes without saturation.
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn row<const W: usize, const AV1: bool>(self, row: &[u8]) -> int16x8_t {
        let tail = if W == 4 { 3 } else { 7 };
        let row = &row[..tail + 8];
        // SAFETY: both eight-byte loads lie in the checked 11/15-byte window.
        let packed = unsafe { vcombine_u8(vld1_u8(row.as_ptr()), vld1_u8(row.as_ptr().add(tail))) };
        let signed = vreinterpretq_s8_u8(veorq_u8(packed, vdupq_n_u8(128)));
        let first = vqtbl1q_s8(signed, self.gather0);
        let middle = vqtbl1q_s8(signed, self.gather4);
        let initial = vdupq_n_s32(128 * 64);
        // SAFETY: this function enables dotprod, and all inputs are vector registers.
        let lo = unsafe { dot(dot(initial, first, self.lo), middle, self.hi) };
        let low = if AV1 {
            vrshrn_n_s32::<2>(lo)
        } else {
            vmovn_s32(lo)
        };
        if W == 4 {
            vcombine_s16(low, vdup_n_s16(0))
        } else {
            let last = vqtbl1q_s8(signed, self.gather8);
            // SAFETY: this function enables dotprod, as for the lower outputs.
            let hi = unsafe { dot(dot(initial, middle, self.lo), last, self.hi) };
            vcombine_s16(
                low,
                if AV1 {
                    vrshrn_n_s32::<2>(hi)
                } else {
                    vmovn_s32(hi)
                },
            )
        }
    }
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn store<const W: usize>(value: int16x8_t, dst: &mut [u8]) {
    let bytes = vqmovun_s16(value);
    if W == 4 {
        let word = vget_lane_u32::<0>(vreinterpret_u32_u8(bytes));
        dst[..4].copy_from_slice(&word.to_ne_bytes());
    } else {
        let dst = &mut dst[..8];
        // SAFETY: the checked destination has eight writable bytes.
        unsafe { vst1_u8(dst.as_mut_ptr(), bytes) };
    }
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn predict<const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    fx: usize,
    fy: usize,
) {
    if fx == 0 {
        // SAFETY: NEON is enabled, and the caller supplies the module contract.
        unsafe {
            if W == 4 {
                generic::hevc::<I16x4, 8>(
                    src,
                    ss,
                    dst,
                    ds,
                    W,
                    h,
                    &HEVC_LUMA[fx],
                    &HEVC_LUMA[fy],
                    (fx, fy),
                );
            } else {
                generic::hevc::<I16x8, 8>(
                    src,
                    ss,
                    dst,
                    ds,
                    W,
                    h,
                    &HEVC_LUMA[fx],
                    &HEVC_LUMA[fy],
                    (fx, fy),
                );
            }
        }
        return;
    }
    // SAFETY: dotprod is enabled and the HEVC taps fit signed bytes.
    let filter = unsafe { DotFilter::new::<W>(&HEVC_LUMA[fx]) };
    let step = if W == 4 { 4 } else { 8 };
    if fy == 0 {
        for y in 0..h {
            for x in (0..W).step_by(step) {
                // SAFETY: the validated source includes the full 8-tap window.
                let first = unsafe { filter.row::<W, false>(&src[(y + TAP_CENTRE) * ss + x..]) };
                let out = vshrq_n_s16::<6>(vaddq_s16(first, vdupq_n_s16(32)));
                // SAFETY: four/eight pixels fit this destination row.
                unsafe { store::<W>(out, &mut dst[y * ds + x..]) };
            }
        }
        return;
    }
    // SAFETY: HEVC uses its complete unchanged eight-tap vertical filter.
    unsafe { vertical_2d::<W, 8>(src, ss, dst, ds, h, filter, &HEVC_LUMA[fy]) };
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn vertical_2d<const W: usize, const N: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    filter: DotFilter,
    vertical: &[i16; N],
) {
    let step = if W == 4 { 4 } else { 8 };
    for x in (0..W).step_by(step) {
        let mut window = [vdupq_n_s16(0); N];
        for (t, row) in window.iter_mut().enumerate().take(N - 1) {
            // SAFETY: the prologue is inside the validated source window.
            *row = unsafe { filter.row::<W, false>(&src[t * ss + x..]) };
        }
        for y in 0..h {
            // SAFETY: the trimmed/full window contains the final filter row.
            window[N - 1] = unsafe { filter.row::<W, false>(&src[(y + N - 1) * ss + x..]) };
            let (mut lo, mut hi) = (vdupq_n_s32(0), vdupq_n_s32(0));
            for (&row, &tap) in window.iter().zip(vertical) {
                lo = vmlal_n_s16(lo, vget_low_s16(row), tap);
                if W != 4 {
                    hi = vmlal_n_s16(hi, vget_high_s16(row), tap);
                }
            }
            let out = vcombine_s16(vrshrn_n_s32::<12>(lo), vrshrn_n_s32::<12>(hi));
            // SAFETY: four/eight pixels fit this destination row.
            unsafe { store::<W>(out, &mut dst[y * ds + x..]) };
            window.copy_within(1.., 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_phases_match_scalar_with_minimal_unaligned_windows() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let scalar = super::super::scalar::mc_fns();
        for (reference, candidate, phases) in [
            (
                scalar.av1,
                super::super::av1_neon::put_fns::<super::super::Regular>(),
                16,
            ),
            (
                scalar.av1_smooth,
                super::super::av1_neon::put_fns::<super::super::Smooth>(),
                16,
            ),
            (
                scalar.av1_sharp,
                super::super::av1_neon::put_fns::<super::super::Sharp>(),
                16,
            ),
            (scalar.hevc_luma, hevc_luma_fns(), 4),
        ] {
            for (index, width) in [4, 8, 16, 32, 64].into_iter().enumerate() {
                for height in [4, 8, 16, 64] {
                    let ss = width + 12;
                    let source_len = (height + 6) * ss + width + 7;
                    let source: Vec<u8> = (0..=source_len)
                        .map(|i| (i * 17 + (i / ss) * 29) as u8)
                        .collect();
                    let ds = width + 11;
                    let length = (height - 1) * ds + width;
                    for fx in 0..phases {
                        for fy in 0..phases {
                            let mut expected = vec![0xa5; length + 2];
                            let mut actual = expected.clone();
                            // SAFETY: feature detection above, exact input/output windows,
                            // supported sizes and every valid phase; guard bytes excluded.
                            unsafe {
                                (reference[index])(
                                    &source[1..],
                                    ss,
                                    &mut expected[1..=length],
                                    ds,
                                    height,
                                    fx,
                                    fy,
                                );
                                (candidate[index])(
                                    &source[1..],
                                    ss,
                                    &mut actual[1..=length],
                                    ds,
                                    height,
                                    fx,
                                    fy,
                                );
                            }
                            assert_eq!(actual, expected, "{width}x{height} phase({fx},{fy})");
                        }
                    }
                }
            }
        }
    }
}
