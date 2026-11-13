//! NEON VP9 inverse lanes: wrapping butterflies, widening rotations and
//! non-saturating narrows, matching libvpx's full-range inverse semantics.

use super::{COS_BITS, Lanes, Products};
use crate::Vp9Pass;
use crate::txfm::inverse_lanes::InverseLanes;
use core::arch::aarch64::{
    int16x8_t, int32x4_t, vaddq_s16, vaddq_s32, vdupq_n_s16, vdupq_n_s32, vget_low_s16, vld1_u8,
    vld1q_s16, vld1q_s32, vmlal_high_n_s16, vmlal_n_s16, vmovl_high_s16, vmovl_s16, vmovl_u8,
    vmovn_high_s32, vmovn_s32, vmull_high_n_s16, vmull_n_s16, vqmovun_s16, vqrdmulhq_n_s16,
    vreinterpretq_s16_u16, vrshlq_s16, vrshrn_high_n_s32, vrshrn_n_s32, vshlq_s32, vst1_u8,
    vst1q_s16, vst1q_s32, vsubq_s16, vsubq_s32, vzip1q_s16, vzip2q_s16,
};

const COS_SHIFT: i32 = COS_BITS as i32;
const HALF: usize = 4;

#[derive(Clone, Copy)]
struct Narrow(int16x8_t);
#[derive(Clone, Copy)]
struct Wide([int32x4_t; 2]);

impl Products for Wide {
    type Narrow = Narrow;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: private vector types are constructed only by the NEON kernel.
        unsafe {
            Self([
                vaddq_s32(self.0[0], other.0[0]),
                vaddq_s32(self.0[1], other.0[1]),
            ])
        }
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: private vector types are constructed only by the NEON kernel.
        unsafe {
            Self([
                vsubq_s32(self.0[0], other.0[0]),
                vsubq_s32(self.0[1], other.0[1]),
            ])
        }
    }
    #[inline(always)]
    fn narrow(self) -> Narrow {
        // SAFETY: NEON is enabled; non-saturating narrowing implements WRAPLOW.
        unsafe { Narrow(vmovn_high_s32(vmovn_s32(self.0[0]), self.0[1])) }
    }
    #[inline(always)]
    fn round14(self) -> Narrow {
        // SAFETY: NEON is enabled. The module's modular proof preserves the
        // rounded low16 bits even if an ADST product sum wrapped in i32.
        unsafe {
            Narrow(vrshrn_high_n_s32::<COS_SHIFT>(
                vrshrn_n_s32::<COS_SHIFT>(self.0[0]),
                self.0[1],
            ))
        }
    }
}

impl Narrow {
    /// `round_shift(w * self, 14)` from `doubled = 2 * w`
    /// ([`super::doubled_weight`]).
    #[inline(always)]
    fn mul_doubled(self, doubled: i16) -> Self {
        // SAFETY: NEON is enabled; `doubled` is never -32768, so the
        // doubling multiply cannot saturate.
        unsafe { Self(vqrdmulhq_n_s16(self.0, doubled)) }
    }
}

impl InverseLanes for Narrow {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: private lanes are used only by the NEON entry point.
        unsafe { Self(vdupq_n_s16(value as i16)) }
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: NEON is enabled. Every DCT add is followed by WRAPLOW.
        unsafe { Self(vaddq_s16(self.0, other.0)) }
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: NEON is enabled. Every DCT subtract is followed by WRAPLOW.
        unsafe { Self(vsubq_s16(self.0, other.0)) }
    }
    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, COS_BITS);
        // Equal-weight pre-adds must also widen: adding i16 first loses
        // information for extreme inputs (libvpx idct_neon.h makes this explicit).
        Self::products(w0, a, w1, b).round14()
    }
    #[inline(always)]
    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        debug_assert_eq!(bit, COS_BITS);
        match super::doubled_weight(w) {
            Some(doubled) => a.mul_doubled(doubled),
            None => a.wide(w).round14(),
        }
    }
    #[inline(always)]
    fn clamp16(self) -> Self {
        self
    }
}

impl Lanes for Narrow {
    type Wide = Wide;
    #[inline(always)]
    fn wide(self, weight: i32) -> Wide {
        // SAFETY: NEON is enabled; all cosine weights fit i16, and each
        // individual product is exact in i32.
        unsafe {
            Wide([
                vmull_n_s16(vget_low_s16(self.0), weight as i16),
                vmull_high_n_s16(self.0, weight as i16),
            ])
        }
    }
    #[inline(always)]
    fn products(w0: i32, a: Self, w1: i32, b: Self) -> Wide {
        // SAFETY: NEON is enabled. Widen before multiply-accumulate; both
        // signed cosine weights fit i16 and the two-product sum fits i32.
        unsafe {
            Wide([
                vmlal_n_s16(
                    vmull_n_s16(vget_low_s16(a.0), w0 as i16),
                    vget_low_s16(b.0),
                    w1 as i16,
                ),
                vmlal_high_n_s16(vmull_high_n_s16(a.0, w0 as i16), b.0, w1 as i16),
            ])
        }
    }
    #[inline(always)]
    fn load_i32_wrapping(src: &[i32]) -> Self {
        let src = &src[..super::LANES];
        // SAFETY: the checked slice holds both four-element loads.
        unsafe { Wide([vld1q_s32(src.as_ptr()), vld1q_s32(src.as_ptr().add(HALF))]).narrow() }
    }
    #[inline(always)]
    fn load_i16(src: &[i16]) -> Self {
        let src = &src[..super::LANES];
        // SAFETY: the checked slice holds eight i16 samples; NEON is enabled.
        unsafe { Self(vld1q_s16(src.as_ptr())) }
    }
    #[inline(always)]
    fn store_i16(self, dst: &mut [i16]) {
        let dst = &mut dst[..super::LANES];
        // SAFETY: the checked destination holds all eight stored samples.
        unsafe {
            vst1q_s16(dst.as_mut_ptr(), self.0);
        }
    }
    #[inline(always)]
    fn store_shifted(self, dst: &mut [i32], shift: u32) {
        let dst = &mut dst[..super::LANES];
        // SAFETY: the checked destination holds both stores. Widen before
        // rounding so i16::MAX plus the rounding offset cannot wrap.
        unsafe {
            let round = vdupq_n_s32(1 << (shift - 1));
            let count = vdupq_n_s32(-(shift as i32));
            let low = vshlq_s32(vaddq_s32(vmovl_s16(vget_low_s16(self.0)), round), count);
            let high = vshlq_s32(vaddq_s32(vmovl_high_s16(self.0), round), count);
            vst1q_s32(dst.as_mut_ptr(), low);
            vst1q_s32(dst.as_mut_ptr().add(HALF), high);
        }
    }
    #[inline(always)]
    unsafe fn add_shifted(self, dst: *mut u8, shift: u32) {
        // SAFETY: the caller's destination holds the eight samples. The
        // rounding shift is exact over the whole i16 range; the residual
        // plus a pixel fits i16 and the unsigned saturating narrow clips.
        unsafe {
            let residual = vrshlq_s16(self.0, vdupq_n_s16(-(shift as i16)));
            let pixels = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(dst.cast_const())));
            vst1_u8(dst, vqmovun_s16(vaddq_s16(residual, pixels)));
        }
    }
    #[inline(always)]
    fn zip_lo(self, other: Self) -> Self {
        // SAFETY: register-only permutation inside the NEON kernel.
        unsafe { Self(vzip1q_s16(self.0, other.0)) }
    }
    #[inline(always)]
    fn zip_hi(self, other: Self) -> Self {
        // SAFETY: register-only permutation inside the NEON kernel.
        unsafe { Self(vzip2q_s16(self.0, other.0)) }
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn inverse(
    log2: u8,
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i32],
    output: &mut [i32],
) {
    super::inverse::<Narrow>(log2, columns, rows, input, output);
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn inverse_add(
    log2: u8,
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i32],
    nonzero: (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
) {
    super::inverse_add::<Narrow>(log2, (columns, rows), input, nonzero, (dst, dst_stride));
}
