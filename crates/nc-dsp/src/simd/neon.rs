//! `AArch64` Advanced SIMD vector types.
//!
//! See the `simd` module docs for why the methods are safe functions: they
//! are only ever called from kernels running under
//! `#[target_feature(enable = "neon")]`. Each method body is one `unsafe`
//! block for that reason; slices are indexed before every load and store.

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    int16x4_t, int16x8_t, int32x4_t, vabs_s16, vabsq_s16, vabsq_s32, vadd_s16, vaddlvq_s32,
    vaddq_s16, vaddq_s32, vaddvq_u32, vaddw_s16, vandq_s16, vbsl_s16, vbslq_s16, vbslq_s32,
    vceqq_s16, vceqq_s32, vceqzq_s32, vcgtq_s16, vcgtq_s32, vcltz_s16, vcltzq_s16, vcombine_s16,
    vcombine_s32, vcreate_u8, vdup_n_s16, vdup_n_u8, vdupq_n_s16, vdupq_n_s32, veorq_s32,
    vextq_s16, vget_high_s16, vget_high_s32, vget_lane_u32, vget_low_s16, vget_low_s32,
    vget_low_u16, vld1_s16, vld1_u8, vld1_u16, vld1q_s16, vld1q_s32, vld1q_u8, vld2_u8, vld2q_u16,
    vmax_s16, vmaxq_s16, vmaxq_s32, vmaxvq_u16, vmin_s16, vminq_s16, vminq_s32, vminq_u16,
    vmlal_high_n_s16, vmlal_high_s16, vmlal_n_s16, vmlal_s16, vmlal_u8, vmlaq_laneq_s32,
    vmlaq_n_s32, vmovl_high_s16, vmovl_s16, vmovl_u8, vmovl_u16, vmul_s16, vmull_high_s16,
    vmull_s16, vmull_u8, vmulq_n_s32, vmulq_s16, vmulq_s32, vneg_s16, vnegq_s16, vorrq_s32,
    vpadalq_s16, vpaddq_s32, vqadd_s16, vqaddq_s16, vqmovn_high_s32, vqmovn_s32, vqmovn_u16,
    vqmovun_s16, vqmovun_s32, vqrdmulh_s16, vqrdmulhq_s16, vqsubq_u16, vqtbl1q_u8,
    vreinterpret_s16_u16, vreinterpret_u16_s16, vreinterpret_u32_u8, vreinterpretq_s16_s32,
    vreinterpretq_s16_u8, vreinterpretq_s16_u16, vreinterpretq_s32_u32, vreinterpretq_u8_s16,
    vreinterpretq_u16_s16, vreinterpretq_u32_s32, vrshlq_s32, vshl_s16, vshl_u16, vshlq_s16,
    vshlq_s32, vshlq_u16, vshlq_u32, vshrq_n_s32, vshrq_n_u32, vst1_s16, vst1_u8, vst1_u16,
    vst1q_s16, vst1q_s32, vsub_s16, vsubq_s16, vsubq_s32, vtrn1q_s16, vtrnq_s32, vzip1q_s16,
    vzip2q_s16,
};

use core::arch::aarch64::{
    uint8x16_t, uint32x4_t, vget_low_u8, vmlal_high_lane_s16, vmlal_high_laneq_s16, vmlal_lane_s16,
    vmlal_laneq_s16, vmovl_high_u8, vmull_high_lane_s16, vmull_high_laneq_s16, vmull_lane_s16,
    vmull_laneq_s16, vqmovun_high_s16, vqshrn_high_n_s32, vqshrn_n_s32, vreinterpretq_u8_u32,
    vreinterpretq_u16_u8, vreinterpretq_u32_u16, vst1q_u8, vtrn1q_u8, vtrn1q_u16, vtrn1q_u32,
    vtrn2q_u8, vtrn2q_u16, vtrn2q_u32,
};
use core::mem::MaybeUninit;

use super::{I16FilterLanes, I16Lanes, I16Pairs, I16x8Shuffles, I32Lanes, packed_pair};

/// Four `i32` lanes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct I32x4(int32x4_t);

/// Eight `i16` lanes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct I16x8Kernel<const SVE2: bool>(int16x8_t);

pub(crate) type I16x8 = I16x8Kernel<false>;
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
pub(crate) type I16x8Sve2 = I16x8Kernel<true>;

/// Four `i16` lanes (a 64-bit register).
#[derive(Debug, Clone, Copy)]
pub(crate) struct I16x4(int16x4_t);

/// Four bytes as the low lanes of a byte vector.
#[inline(always)]
fn u8x4_bits(bytes: &[u8]) -> u64 {
    u64::from(u32::from_le_bytes(bytes[..4].try_into().expect("4 bytes")))
}

// ---------------------------------------------------------------- I32x4 ----

impl I32Lanes for I32x4 {
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vdupq_n_s32(value)) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vaddq_s32(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vsubq_s32(self.0, other.0)) }
    }

    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            let sum = vmlaq_n_s32(vmulq_n_s32(a.0, w0), b.0, w1);
            // A rounding shift by a negative count is libaom's round_shift.
            Self(vrshlq_s32(sum, vdupq_n_s32(-(bit as i32))))
        }
    }

    #[inline(always)]
    fn clamp16(self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            Self(vmaxq_s32(
                vminq_s32(self.0, vdupq_n_s32(i32::from(i16::MAX))),
                vdupq_n_s32(i32::from(i16::MIN)),
            ))
        }
    }

    #[inline(always)]
    fn round_shift(self, bits: u32) -> Self {
        // SAFETY: NEON kernels only (module docs). A shift by 0 is the identity.
        unsafe { Self(vrshlq_s32(self.0, vdupq_n_s32(-(bits as i32)))) }
    }

    #[inline(always)]
    fn shl(self, bits: u32) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vshlq_s32(self.0, vdupq_n_s32(bits as i32))) }
    }

    #[inline(always)]
    fn mul_scalar(self, c: i32) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vmulq_n_s32(self.0, c)) }
    }

    #[inline(always)]
    fn mul_add4(self, x: [Self; 4], c: &[i32; 4]) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            let c = vld1q_s32(c.as_ptr());
            let acc = vmlaq_laneq_s32::<0>(self.0, x[0].0, c);
            let acc = vmlaq_laneq_s32::<1>(acc, x[1].0, c);
            let acc = vmlaq_laneq_s32::<2>(acc, x[2].0, c);
            Self(vmlaq_laneq_s32::<3>(acc, x[3].0, c))
        }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vmulq_s32(self.0, other.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vabsq_s32(self.0)) }
    }

    #[inline(always)]
    fn shr_logical(self, bits: u32) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            Self(vreinterpretq_s32_u32(vshlq_u32(
                vreinterpretq_u32_s32(self.0),
                vdupq_n_s32(-(bits as i32)),
            )))
        }
    }

    #[inline(always)]
    fn sra(self, bits: u32) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vshlq_s32(self.0, vdupq_n_s32(-(bits as i32)))) }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vminq_s32(self.0, other.0)) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vmaxq_s32(self.0, other.0)) }
    }

    #[inline(always)]
    fn with_sign_of(self, like: Self) -> Self {
        // SAFETY: NEON kernels only (module docs). (x ^ m) - m with m = like >> 31.
        unsafe {
            let m = vshrq_n_s32::<31>(like.0);
            Self(vsubq_s32(veorq_s32(self.0, m), m))
        }
    }

    #[inline(always)]
    fn count_nonzero(self) -> u32 {
        // SAFETY: NEON kernels only (module docs).
        unsafe { 4 - vaddvq_u32(vshrq_n_u32::<31>(vceqzq_s32(self.0))) }
    }

    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        let src = &src[..4];
        // SAFETY: NEON kernels only; `src` holds 4 values.
        unsafe { Self(vld1q_s32(src.as_ptr())) }
    }

    #[inline(always)]
    fn store(self, dst: &mut [i32]) {
        let dst = &mut dst[..4];
        // SAFETY: NEON kernels only; `dst` holds 4 values.
        unsafe { vst1q_s32(dst.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn load_i16(src: &[i16]) -> Self {
        let src = &src[..4];
        // SAFETY: NEON kernels only; `src` holds 4 values.
        unsafe { Self(vmovl_s16(vld1_s16(src.as_ptr()))) }
    }

    #[inline(always)]
    fn add_to_u8(self, dst: &mut [u8]) {
        let dst = &mut dst[..4];
        // SAFETY: NEON kernels only (module docs).
        let bytes = unsafe {
            let pred = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(
                u8x4_bits(dst),
            )))));
            let sum = vqmovun_s32(vaddq_s32(self.0, pred));
            let narrow = vqmovn_u16(vcombine_u16_lo(sum));
            vget_lane_u32::<0>(vreinterpret_u32_u8(narrow))
        };
        dst.copy_from_slice(&bytes.to_le_bytes());
    }

    #[inline(always)]
    fn add_constant_u8(residual: i32, n: usize, dst: &mut [u8], stride: usize) {
        // SAFETY: NEON kernels only; caller's slice covers `n` rows at `stride`.
        unsafe {
            crate::constant_add::add::<crate::constant_add::neon::Bytes>(
                residual, n, n, dst, stride,
            );
        }
    }

    #[inline(always)]
    fn add_to_u16(self, dst: &mut [u16], max_val: i32) {
        let dst = &mut dst[..4];
        // SAFETY: NEON kernels only; `dst` holds 4 `u16` samples.
        unsafe {
            let pred = vreinterpretq_s32_u32(vmovl_u16(vld1_u16(dst.as_ptr())));
            let sum = vminq_s32(vaddq_s32(self.0, pred), vdupq_n_s32(max_val));
            vst1_u16(dst.as_mut_ptr(), vqmovun_s32(sum));
        }
    }

    #[inline(always)]
    fn store_transposed(rows: &[Self], dst: &mut [MaybeUninit<i32>], dst_stride: usize) {
        for (r, value) in transpose_i32x4(rows).into_iter().enumerate() {
            let dst = &mut dst[r * dst_stride..][..Self::LANES];
            // SAFETY: NEON enabled and dst covers one complete vector.
            unsafe {
                vst1q_s32(dst.as_mut_ptr().cast(), value.0);
            }
        }
    }

    #[inline(always)]
    fn store_sat16(self, dst: &mut [i16]) {
        let dst = &mut dst[..4];
        // SAFETY: NEON kernels only; `dst` holds 4 values.
        unsafe { vst1_s16(dst.as_mut_ptr(), vqmovn_s32(self.0)) }
    }

    #[inline(always)]
    fn load_i16_pair(src: &[i16]) -> (Self, Self) {
        let src = &src[..8];
        // SAFETY: NEON kernels only; `src` holds 8 values.
        unsafe {
            let values = vld1q_s16(src.as_ptr());
            (
                Self(vmovl_s16(vget_low_s16(values))),
                Self(vmovl_high_s16(values)),
            )
        }
    }

    #[inline(always)]
    fn store_sat16_pair(low: Self, high: Self, dst: &mut [i16]) {
        let dst = &mut dst[..8];
        // SAFETY: NEON kernels only; `dst` holds 8 values.
        unsafe { vst1q_s16(dst.as_mut_ptr(), vqmovn_high_s32(vqmovn_s32(low.0), high.0)) }
    }

    #[inline(always)]
    fn store_transposed_sat16(rows: &[Self], dst: &mut [MaybeUninit<i16>], dst_stride: usize) {
        for (r, value) in transpose_i32x4(rows).into_iter().enumerate() {
            let dst = &mut dst[r * dst_stride..][..Self::LANES];
            // SAFETY: NEON enabled and dst covers one complete 4-lane row;
            // uninitialised memory may be written.
            unsafe {
                vst1_s16(dst.as_mut_ptr().cast(), vqmovn_s32(value.0));
            }
        }
    }

    #[inline(always)]
    fn add_transposed_u8(rows: &[Self], dst: &mut [u8], dst_stride: usize) {
        let needed = dst_stride
            .checked_mul(Self::LANES - 1)
            .and_then(|n| n.checked_add(Self::LANES))
            .expect("transpose tile extent");
        let dst = &mut dst[..needed];
        for (r, value) in transpose_i32x4(rows).into_iter().enumerate() {
            // SAFETY: r < LANES and the checked span covers the last row's
            // full vector; each temporary row borrow ends in this iteration.
            let row = unsafe {
                core::slice::from_raw_parts_mut(dst.as_mut_ptr().add(r * dst_stride), Self::LANES)
            };
            value.add_to_u8(row);
        }
    }

    #[inline(always)]
    fn add_transposed_u16(rows: &[Self], dst: &mut [u16], dst_stride: usize, max_val: i32) {
        let needed = dst_stride
            .checked_mul(Self::LANES - 1)
            .and_then(|n| n.checked_add(Self::LANES))
            .expect("transpose tile extent");
        let dst = &mut dst[..needed];
        for (r, value) in transpose_i32x4(rows).into_iter().enumerate() {
            // SAFETY: r < LANES and the checked span covers the last row's
            // full vector; each temporary row borrow ends in this iteration.
            let row = unsafe {
                core::slice::from_raw_parts_mut(dst.as_mut_ptr().add(r * dst_stride), Self::LANES)
            };
            value.add_to_u16(row, max_val);
        }
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vreinterpretq_s32_u32(vcgtq_s32(self.0, other.0))) }
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vreinterpretq_s32_u32(vceqq_s32(self.0, other.0))) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vorrq_s32(self.0, other.0)) }
    }

    #[inline(always)]
    fn select(self, if_set: Self, if_clear: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            Self(vbslq_s32(
                vreinterpretq_u32_s32(self.0),
                if_set.0,
                if_clear.0,
            ))
        }
    }
}

/// `[v, v]` as a 128-bit vector of eight `u16` lanes.
#[inline(always)]
fn vcombine_u16_lo(v: core::arch::aarch64::uint16x4_t) -> core::arch::aarch64::uint16x8_t {
    // SAFETY: NEON kernels only (module docs).
    unsafe { core::arch::aarch64::vcombine_u16(v, v) }
}

// ---------------------------------------------------------------- I16x8 ----

/// `clamp((acc + round) >> shift)` of four `i32` lanes, narrowed to `i16`.
#[inline(always)]
fn round_shift_narrow(acc: int32x4_t, round: i32, shift: u32) -> int16x4_t {
    // SAFETY: NEON kernels only (module docs).
    unsafe {
        let v = if round == 0 {
            acc
        } else {
            vaddq_s32(acc, vdupq_n_s32(round))
        };
        match shift {
            0 => vqmovn_s32(v),
            2 => vqshrn_n_s32::<2>(v),
            4 => vqshrn_n_s32::<4>(v),
            6 => vqshrn_n_s32::<6>(v),
            7 => vqshrn_n_s32::<7>(v),
            _ => vqmovn_s32(vshlq_s32(v, vdupq_n_s32(-(shift as i32)))),
        }
    }
}

/// `clamp((acc + round) >> shift)` of eight `i32` lanes, narrowed to `i16`.
#[inline(always)]
fn round_shift_narrow_pair(lo: int32x4_t, hi: int32x4_t, round: i32, shift: u32) -> int16x8_t {
    // SAFETY: NEON kernels only (module docs).
    unsafe {
        let (a, b) = if round == 0 {
            (lo, hi)
        } else {
            let r = vdupq_n_s32(round);
            (vaddq_s32(lo, r), vaddq_s32(hi, r))
        };
        match shift {
            0 => vqmovn_high_s32(vqmovn_s32(a), b),
            2 => vqshrn_high_n_s32::<2>(vqshrn_n_s32::<2>(a), b),
            4 => vqshrn_high_n_s32::<4>(vqshrn_n_s32::<4>(a), b),
            6 => vqshrn_high_n_s32::<6>(vqshrn_n_s32::<6>(a), b),
            7 => vqshrn_high_n_s32::<7>(vqshrn_n_s32::<7>(a), b),
            _ => {
                let neg = vdupq_n_s32(-(shift as i32));
                vqmovn_high_s32(vqmovn_s32(vshlq_s32(a, neg)), vshlq_s32(b, neg))
            }
        }
    }
}

impl<const SVE2: bool> I16Lanes for I16x8Kernel<SVE2> {
    const LANES: usize = 8;
    type Acc = [int32x4_t; 2];

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vminq_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vmaxq_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vabsq_s16(self.0)) }
    }

    #[inline(always)]
    fn with_sign_of(self, like: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vbslq_s16(vcltzq_s16(like.0), vnegq_s16(self.0), self.0)) }
    }

    #[inline(always)]
    fn splat(value: i16) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vdupq_n_s16(value)) }
    }

    #[inline(always)]
    fn load_u8(src: &[u8]) -> Self {
        let src = &src[..8];
        // SAFETY: NEON kernels only; `src` holds 8 bytes.
        unsafe { Self(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src.as_ptr())))) }
    }

    #[inline(always)]
    fn pair_sum_u8(src: &[u8]) -> Self {
        use core::arch::aarch64::{vld1q_u8, vpaddlq_u8};
        let src = &src[..16];
        // SAFETY: NEON kernels only; `src` holds 16 bytes.
        unsafe { Self(vreinterpretq_s16_u16(vpaddlq_u8(vld1q_u8(src.as_ptr())))) }
    }

    #[inline(always)]
    fn load_i8(src: &[i8]) -> Self {
        use core::arch::aarch64::{vld1_s8, vmovl_s8};
        let src = &src[..8];
        // SAFETY: NEON kernels only; `src` holds 8 bytes.
        unsafe { Self(vmovl_s8(vld1_s8(src.as_ptr()))) }
    }

    #[inline(always)]
    fn store_i8(self, dst: &mut [i8]) {
        use core::arch::aarch64::{vqmovn_s16, vst1_s8};
        let dst = &mut dst[..8];
        // SAFETY: NEON kernels only; `dst` holds 8 bytes.
        unsafe { vst1_s8(dst.as_mut_ptr(), vqmovn_s16(self.0)) }
    }

    #[inline(always)]
    fn load(src: &[i16]) -> Self {
        let src = &src[..8];
        // SAFETY: NEON kernels only; `src` holds 8 values.
        unsafe { Self(vld1q_s16(src.as_ptr())) }
    }

    #[inline(always)]
    fn pair_avg_u16(pairs: &[u16]) -> Self {
        use core::arch::aarch64::{vld2q_u16, vrhaddq_u16};
        let pairs = &pairs[..16];
        // SAFETY: NEON kernels only; `pairs` holds 16 `u16` elements.
        unsafe {
            let deint = vld2q_u16(pairs.as_ptr());
            Self(vreinterpretq_s16_u16(vrhaddq_u16(deint.0, deint.1)))
        }
    }

    #[inline(always)]
    fn store(self, dst: &mut [i16]) {
        let dst = &mut dst[..8];
        // SAFETY: NEON kernels only; `dst` holds 8 values.
        unsafe { vst1q_s16(dst.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vaddq_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn add_sat(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vqaddq_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vsubq_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vmulq_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn mul_q15_round(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vqrdmulhq_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sra(self, bits: u32) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vshlq_s16(self.0, vdupq_n_s16(-(bits as i16)))) }
    }

    #[inline(always)]
    fn srl(self, bits: u32) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            Self(vreinterpretq_s16_u16(vshlq_u16(
                vreinterpretq_u16_s16(self.0),
                vdupq_n_s16(-(bits as i16)),
            )))
        }
    }

    #[inline(always)]
    fn clamp_u8(self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            Self(vmaxq_s16(
                vminq_s16(self.0, vdupq_n_s16(255)),
                vdupq_n_s16(0),
            ))
        }
    }

    #[inline(always)]
    fn mul_add_u8_pair(a: &[u8], b: &[u8], wa: u8, wb: u8) -> Self {
        let (a, b) = (&a[..8], &b[..8]);
        // SAFETY: NEON kernels only; `a` and `b` hold 8 bytes. The sum is at
        // most 128·255, so the unsigned 16-bit lanes reinterpret exactly.
        unsafe {
            let sum = vmlal_u8(
                vmull_u8(vld1_u8(a.as_ptr()), vdup_n_u8(wa)),
                vld1_u8(b.as_ptr()),
                vdup_n_u8(wb),
            );
            Self(vreinterpretq_s16_u16(sum))
        }
    }

    #[inline(always)]
    fn store_u8_sat(self, dst: &mut [u8]) {
        let dst = &mut dst[..8];
        // SAFETY: NEON kernels only; `dst` holds 8 bytes.
        unsafe { vst1_u8(dst.as_mut_ptr(), vqmovun_s16(self.0)) }
    }

    #[inline(always)]
    fn acc_zero() -> Self::Acc {
        // SAFETY: NEON kernels only (module docs).
        unsafe { [vdupq_n_s32(0); 2] }
    }

    #[inline(always)]
    fn acc_mul_add(acc: Self::Acc, a: Self, coeff: i16) -> Self::Acc {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            [
                vmlal_n_s16(acc[0], vget_low_s16(a.0), coeff),
                vmlal_high_n_s16(acc[1], a.0, coeff),
            ]
        }
    }

    #[inline(always)]
    fn acc_vec_mul_add2(a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            [
                vmlal_s16(
                    vmull_s16(vget_low_s16(a.0), vget_low_s16(wa.0)),
                    vget_low_s16(b.0),
                    vget_low_s16(wb.0),
                ),
                vmlal_high_s16(vmull_high_s16(a.0, wa.0), b.0, wb.0),
            ]
        }
    }

    #[inline(always)]
    fn acc_vec_fma2(acc: Self::Acc, a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            [
                vmlal_s16(
                    vmlal_s16(acc[0], vget_low_s16(a.0), vget_low_s16(wa.0)),
                    vget_low_s16(b.0),
                    vget_low_s16(wb.0),
                ),
                vmlal_high_s16(vmlal_high_s16(acc[1], a.0, wa.0), b.0, wb.0),
            ]
        }
    }

    #[inline(always)]
    fn acc_add(acc: Self::Acc, a: Self) -> Self::Acc {
        // SAFETY: NEON kernels only (module docs).
        unsafe { [vpadalq_s16(acc[0], a.0), acc[1]] }
    }

    #[inline(always)]
    fn acc_add_sq(acc: Self::Acc, a: Self) -> Self::Acc {
        #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
        if SVE2 {
            // SAFETY: the SVE2 caller selected this lane type after detection.
            // Its SSE accumulator starts at zero and holds squared byte
            // differences for an 8×64 block at most. Each low word remains
            // below i32::MAX and the other words stay zero.
            return [unsafe { super::sve2::add_squares(acc[0], a.0) }, acc[1]];
        }
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            [
                vmlal_s16(acc[0], vget_low_s16(a.0), vget_low_s16(a.0)),
                vmlal_high_s16(acc[1], a.0, a.0),
            ]
        }
    }

    #[inline(always)]
    fn acc_round_shift_store_u8(acc: Self::Acc, round: i32, shift: u32, dst: &mut [u8]) {
        let dst = &mut dst[..8];
        let words = round_shift_narrow_pair(acc[0], acc[1], round, shift);
        // SAFETY: NEON kernels only; `dst` holds 8 bytes.
        unsafe {
            vst1_u8(dst.as_mut_ptr(), vqmovun_s16(words));
        }
    }

    #[inline(always)]
    fn acc_round_shift_narrow(acc: Self::Acc, round: i32, shift: u32) -> Self {
        Self(round_shift_narrow_pair(acc[0], acc[1], round, shift))
    }

    #[inline(always)]
    fn acc_sum(acc: Self::Acc) -> i64 {
        // SAFETY: NEON kernels only (module docs).
        unsafe { vaddlvq_s32(acc[0]) + vaddlvq_s32(acc[1]) }
    }

    const FAST_HORIZ_EXT: bool = true;

    #[inline(always)]
    fn horiz_taps_acc<const N: usize>(
        s0: Self,
        s_last: Self,
        six_x: bool,
        cx: &[i16; N],
    ) -> Self::Acc {
        // SAFETY: NEON kernels only (module docs); `cx` has length `N` (4 or 8).
        unsafe {
            if N == 4 {
                let taps = vld1_s16(cx.as_ptr());
                let s1_hi = vextq_s16::<5>(s_last.0, s_last.0);
                let v1 = vextq_s16::<1>(s0.0, s1_hi);
                let v2 = vextq_s16::<2>(s0.0, s1_hi);
                let mut lo = vmull_lane_s16::<0>(vget_low_s16(s0.0), taps);
                let mut hi = vmull_high_lane_s16::<0>(s0.0, taps);
                lo = vmlal_lane_s16::<1>(lo, vget_low_s16(v1), taps);
                hi = vmlal_high_lane_s16::<1>(hi, v1, taps);
                lo = vmlal_lane_s16::<2>(lo, vget_low_s16(v2), taps);
                hi = vmlal_high_lane_s16::<2>(hi, v2, taps);
                lo = vmlal_lane_s16::<3>(lo, vget_low_s16(s_last.0), taps);
                hi = vmlal_high_lane_s16::<3>(hi, s_last.0, taps);
                [lo, hi]
            } else {
                let taps = vld1q_s16(cx.as_ptr());
                let s1_hi = vextq_s16::<1>(s_last.0, s_last.0);
                let v1 = vextq_s16::<1>(s0.0, s1_hi);
                let v2 = vextq_s16::<2>(s0.0, s1_hi);
                let v3 = vextq_s16::<3>(s0.0, s1_hi);
                let v4 = vextq_s16::<4>(s0.0, s1_hi);
                let v5 = vextq_s16::<5>(s0.0, s1_hi);
                let v6 = vextq_s16::<6>(s0.0, s1_hi);
                if six_x {
                    let mut lo = vmull_laneq_s16::<1>(vget_low_s16(v1), taps);
                    let mut hi = vmull_high_laneq_s16::<1>(v1, taps);
                    lo = vmlal_laneq_s16::<2>(lo, vget_low_s16(v2), taps);
                    hi = vmlal_high_laneq_s16::<2>(hi, v2, taps);
                    lo = vmlal_laneq_s16::<3>(lo, vget_low_s16(v3), taps);
                    hi = vmlal_high_laneq_s16::<3>(hi, v3, taps);
                    lo = vmlal_laneq_s16::<4>(lo, vget_low_s16(v4), taps);
                    hi = vmlal_high_laneq_s16::<4>(hi, v4, taps);
                    lo = vmlal_laneq_s16::<5>(lo, vget_low_s16(v5), taps);
                    hi = vmlal_high_laneq_s16::<5>(hi, v5, taps);
                    lo = vmlal_laneq_s16::<6>(lo, vget_low_s16(v6), taps);
                    hi = vmlal_high_laneq_s16::<6>(hi, v6, taps);
                    [lo, hi]
                } else {
                    let mut lo = vmull_laneq_s16::<0>(vget_low_s16(s0.0), taps);
                    let mut hi = vmull_high_laneq_s16::<0>(s0.0, taps);
                    lo = vmlal_laneq_s16::<1>(lo, vget_low_s16(v1), taps);
                    hi = vmlal_high_laneq_s16::<1>(hi, v1, taps);
                    lo = vmlal_laneq_s16::<2>(lo, vget_low_s16(v2), taps);
                    hi = vmlal_high_laneq_s16::<2>(hi, v2, taps);
                    lo = vmlal_laneq_s16::<3>(lo, vget_low_s16(v3), taps);
                    hi = vmlal_high_laneq_s16::<3>(hi, v3, taps);
                    lo = vmlal_laneq_s16::<4>(lo, vget_low_s16(v4), taps);
                    hi = vmlal_high_laneq_s16::<4>(hi, v4, taps);
                    lo = vmlal_laneq_s16::<5>(lo, vget_low_s16(v5), taps);
                    hi = vmlal_high_laneq_s16::<5>(hi, v5, taps);
                    lo = vmlal_laneq_s16::<6>(lo, vget_low_s16(v6), taps);
                    hi = vmlal_high_laneq_s16::<6>(hi, v6, taps);
                    lo = vmlal_laneq_s16::<7>(lo, vget_low_s16(s_last.0), taps);
                    hi = vmlal_high_laneq_s16::<7>(hi, s_last.0, taps);
                    [lo, hi]
                }
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    #[inline(always)]
    unsafe fn horizontal_dot_8tap(
        src: *const i16,
        taps: [i16; 8],
        round: i32,
        shift: u32,
    ) -> Option<Self> {
        if SVE2 {
            // SAFETY: the SVE2 table was selected and the caller checked the row.
            let (lo, hi) = unsafe { crate::simd::sve2::horizontal_8tap(src, &taps) };
            Some(Self::acc_round_shift_narrow([lo, hi], round, shift))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------- I16x4 ----

impl I16Lanes for I16x4 {
    const LANES: usize = 4;
    type Acc = int32x4_t;

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vmin_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vmax_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vabs_s16(self.0)) }
    }

    #[inline(always)]
    fn with_sign_of(self, like: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vbsl_s16(vcltz_s16(like.0), vneg_s16(self.0), self.0)) }
    }

    #[inline(always)]
    fn splat(value: i16) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vdup_n_s16(value)) }
    }

    #[inline(always)]
    fn load_u8(src: &[u8]) -> Self {
        let bits = u8x4_bits(src);
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            Self(vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vcreate_u8(
                bits,
            )))))
        }
    }

    #[inline(always)]
    fn load(src: &[i16]) -> Self {
        let src = &src[..4];
        // SAFETY: NEON kernels only; `src` holds 4 values.
        unsafe { Self(vld1_s16(src.as_ptr())) }
    }

    #[inline(always)]
    fn store(self, dst: &mut [i16]) {
        let dst = &mut dst[..4];
        // SAFETY: NEON kernels only; `dst` holds 4 values.
        unsafe { vst1_s16(dst.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vadd_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn add_sat(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vqadd_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vsub_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vmul_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn mul_q15_round(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vqrdmulh_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn sra(self, bits: u32) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vshl_s16(self.0, vdup_n_s16(-(bits as i16)))) }
    }

    #[inline(always)]
    fn srl(self, bits: u32) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            Self(vreinterpret_s16_u16(vshl_u16(
                vreinterpret_u16_s16(self.0),
                vdup_n_s16(-(bits as i16)),
            )))
        }
    }

    #[inline(always)]
    fn clamp_u8(self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vmax_s16(vmin_s16(self.0, vdup_n_s16(255)), vdup_n_s16(0))) }
    }

    #[inline(always)]
    fn store_u8_sat(self, dst: &mut [u8]) {
        // SAFETY: NEON kernels only (module docs).
        let bytes = unsafe {
            vget_lane_u32::<0>(vreinterpret_u32_u8(vqmovun_s16(vcombine_s16(
                self.0, self.0,
            ))))
        };
        dst[..4].copy_from_slice(&bytes.to_le_bytes());
    }

    #[inline(always)]
    fn acc_zero() -> Self::Acc {
        // SAFETY: NEON kernels only (module docs).
        unsafe { vdupq_n_s32(0) }
    }

    #[inline(always)]
    fn acc_mul_add(acc: Self::Acc, a: Self, coeff: i16) -> Self::Acc {
        // SAFETY: NEON kernels only (module docs).
        unsafe { vmlal_n_s16(acc, a.0, coeff) }
    }

    #[inline(always)]
    fn acc_vec_mul_add2(a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        // SAFETY: NEON kernels only (module docs).
        unsafe { vmlal_s16(vmull_s16(a.0, wa.0), b.0, wb.0) }
    }

    #[inline(always)]
    fn acc_vec_fma2(acc: Self::Acc, a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        // SAFETY: NEON kernels only (module docs).
        unsafe { vmlal_s16(vmlal_s16(acc, a.0, wa.0), b.0, wb.0) }
    }

    #[inline(always)]
    fn acc_add(acc: Self::Acc, a: Self) -> Self::Acc {
        // SAFETY: NEON kernels only (module docs).
        unsafe { vaddw_s16(acc, a.0) }
    }

    #[inline(always)]
    fn acc_add_sq(acc: Self::Acc, a: Self) -> Self::Acc {
        // SAFETY: NEON kernels only (module docs).
        unsafe { vmlal_s16(acc, a.0, a.0) }
    }

    #[inline(always)]
    fn acc_round_shift_store_u8(acc: Self::Acc, round: i32, shift: u32, dst: &mut [u8]) {
        let v = round_shift_narrow(acc, round, shift);
        // SAFETY: NEON kernels only (module docs).
        let bytes =
            unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(vqmovun_s16(vcombine_s16(v, v)))) };
        dst[..4].copy_from_slice(&bytes.to_le_bytes());
    }

    #[inline(always)]
    fn acc_round_shift_narrow(acc: Self::Acc, round: i32, shift: u32) -> Self {
        Self(round_shift_narrow(acc, round, shift))
    }

    #[inline(always)]
    fn acc_sum(acc: Self::Acc) -> i64 {
        // SAFETY: NEON kernels only (module docs).
        unsafe { vaddlvq_s32(acc) }
    }
}

// ------------------------------------------------------- filter lanes ----

impl<const SVE2: bool> I16FilterLanes for I16x8Kernel<SVE2> {
    #[inline(always)]
    fn load_halves(lo: &[i16], hi: &[i16]) -> Self {
        let (lo, hi) = (&lo[..4], &hi[..4]);
        // SAFETY: NEON kernels only; `lo` and `hi` hold 4 values each.
        unsafe { Self(vcombine_s16(vld1_s16(lo.as_ptr()), vld1_s16(hi.as_ptr()))) }
    }

    #[inline(always)]
    fn load_u8_halves(lo: &[u8], hi: &[u8]) -> Self {
        let lo = u64::from(u32::from_le_bytes(lo[..4].try_into().unwrap()));
        let hi = u64::from(u32::from_le_bytes(hi[..4].try_into().unwrap()));
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vreinterpretq_s16_u16(vmovl_u8(vcreate_u8(lo | (hi << 32))))) }
    }

    #[inline(always)]
    fn store_u8_sat_halves(self, lo: &mut [u8], hi: &mut [u8]) {
        // SAFETY: NEON kernels only (module docs).
        let (lo_bytes, hi_bytes) = unsafe {
            let words = vreinterpret_u32_u8(vqmovun_s16(self.0));
            (vget_lane_u32::<0>(words), vget_lane_u32::<1>(words))
        };
        lo[..4].copy_from_slice(&lo_bytes.to_le_bytes());
        hi[..4].copy_from_slice(&hi_bytes.to_le_bytes());
    }

    #[inline(always)]
    fn store_halves(self, lo: &mut [i16], hi: &mut [i16]) {
        let (lo, hi) = (&mut lo[..4], &mut hi[..4]);
        // SAFETY: NEON kernels only; `lo` and `hi` hold 4 words each.
        unsafe {
            vst1_s16(lo.as_mut_ptr(), vget_low_s16(self.0));
            vst1_s16(hi.as_mut_ptr(), vget_high_s16(self.0));
        }
    }

    #[inline(always)]
    fn combine_hi_lo(self, next: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vextq_s16::<4>(self.0, next.0)) }
    }

    #[inline(always)]
    fn min_u16(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            Self(vreinterpretq_s16_u16(vminq_u16(
                vreinterpretq_u16_s16(self.0),
                vreinterpretq_u16_s16(other.0),
            )))
        }
    }

    #[inline(always)]
    fn sub_sat_u16(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            Self(vreinterpretq_s16_u16(vqsubq_u16(
                vreinterpretq_u16_s16(self.0),
                vreinterpretq_u16_s16(other.0),
            )))
        }
    }

    #[inline(always)]
    fn eq_mask(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vreinterpretq_s16_u16(vceqq_s16(self.0, other.0))) }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vandq_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn gt_mask(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vreinterpretq_s16_u16(vcgtq_s16(self.0, other.0))) }
    }

    #[inline(always)]
    fn blend(self, if_set: Self, if_clear: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            Self(vbslq_s16(
                vreinterpretq_u16_s16(self.0),
                if_set.0,
                if_clear.0,
            ))
        }
    }

    #[inline(always)]
    fn any_nonzero(self) -> bool {
        // SAFETY: NEON kernels only (module docs).
        unsafe { vmaxvq_u16(vreinterpretq_u16_s16(self.0)) != 0 }
    }

    #[inline(always)]
    fn load_u8_deinterleaved(src: &[u8]) -> (Self, Self) {
        let src = &src[..16];
        // SAFETY: NEON kernels only; `src` holds 16 bytes.
        unsafe {
            let deint = vld2_u8(src.as_ptr());
            (
                Self(vreinterpretq_s16_u16(vmovl_u8(deint.0))),
                Self(vreinterpretq_s16_u16(vmovl_u8(deint.1))),
            )
        }
    }

    #[inline(always)]
    fn load_u16_deinterleaved(src: &[u16]) -> (Self, Self) {
        let src = &src[..16];
        // SAFETY: NEON kernels only; `src` holds 16 `u16` elements (32 bytes).
        unsafe {
            let deint = vld2q_u16(src.as_ptr());
            (
                Self(vreinterpretq_s16_u16(deint.0)),
                Self(vreinterpretq_s16_u16(deint.1)),
            )
        }
    }

    #[inline(always)]
    fn combine_mid_halves(self, next: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vextq_s16::<4>(self.0, next.0)) }
    }

    #[inline(always)]
    fn shift_quarter(self, next: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vextq_s16::<2>(self.0, next.0)) }
    }

    #[inline(always)]
    fn expand_alf_pair(blocks: &[[i16; 12]], p: usize) -> [Self; 2] {
        let b = &blocks[..2];
        let t = 2 * p;
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            [
                Self(vcombine_s16(vdup_n_s16(b[0][t]), vdup_n_s16(b[1][t]))),
                Self(vcombine_s16(
                    vdup_n_s16(b[0][t + 1]),
                    vdup_n_s16(b[1][t + 1]),
                )),
            ]
        }
    }
}

impl I16x8 {
    /// Products for a forward-transform column group, split into the low
    /// and high four lines. The caller's transform bounds keep sums in i32.
    #[inline(always)]
    pub(crate) fn matrix_dot<const H: usize>(input: &[Self; H], weights: &[i16; H]) -> [I32x4; 2] {
        // SAFETY: NEON kernels only. The weights cover each group of four.
        unsafe {
            if H == 1 {
                return input[0].mul_wide(weights[0]);
            }
            let (quads, tail) = input.as_chunks::<4>();
            let (weight_quads, tail_weights) = weights.as_chunks::<4>();
            let (mut lo, mut hi) = (vdupq_n_s32(0), vdupq_n_s32(0));
            for (x, c) in quads.iter().zip(weight_quads) {
                let c = vld1_s16(c.as_ptr());
                lo = vmlal_lane_s16::<0>(lo, vget_low_s16(x[0].0), c);
                hi = vmlal_high_lane_s16::<0>(hi, x[0].0, c);
                lo = vmlal_lane_s16::<1>(lo, vget_low_s16(x[1].0), c);
                hi = vmlal_high_lane_s16::<1>(hi, x[1].0, c);
                lo = vmlal_lane_s16::<2>(lo, vget_low_s16(x[2].0), c);
                hi = vmlal_high_lane_s16::<2>(hi, x[2].0, c);
                lo = vmlal_lane_s16::<3>(lo, vget_low_s16(x[3].0), c);
                hi = vmlal_high_lane_s16::<3>(hi, x[3].0, c);
            }
            for (x, &w) in tail.iter().zip(tail_weights) {
                lo = vmlal_n_s16(lo, vget_low_s16(x.0), w);
                hi = vmlal_high_n_s16(hi, x.0, w);
            }
            [I32x4(lo), I32x4(hi)]
        }
    }

    #[inline(always)]
    pub(crate) fn mul_wide(self, weight: i16) -> [I32x4; 2] {
        // SAFETY: NEON kernels only.
        unsafe {
            let weights = vdupq_n_s16(weight);
            [
                I32x4(vmull_s16(vget_low_s16(self.0), vget_low_s16(weights))),
                I32x4(vmull_high_s16(self.0, weights)),
            ]
        }
    }
}

impl<const SVE2: bool> I16Pairs for I16x8Kernel<SVE2> {
    const FAST_PAIR_DOT: bool = false;
    type Pairs = I32x4;

    #[inline(always)]
    fn splat_pair(first: i16, second: i16) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            Self(vreinterpretq_s16_s32(vdupq_n_s32(packed_pair(
                first, second,
            ))))
        }
    }

    #[inline(always)]
    fn pair_dot(self, other: Self) -> I32x4 {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            I32x4(vpaddq_s32(
                vmull_s16(vget_low_s16(self.0), vget_low_s16(other.0)),
                vmull_high_s16(self.0, other.0),
            ))
        }
    }

    #[inline(always)]
    fn narrow_pairs(low: I32x4, high: I32x4) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vcombine_s16(vqmovn_s32(low.0), vqmovn_s32(high.0))) }
    }

    #[inline(always)]
    fn interleave_pairs(even: I32x4, odd: I32x4) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            Self(vtrn1q_s16(
                vreinterpretq_s16_s32(even.0),
                vreinterpretq_s16_s32(odd.0),
            ))
        }
    }
}

impl I16x8Shuffles for I16x8 {
    type Wide = I32x4;

    #[inline(always)]
    fn rounded_shr<const SHIFT: i32>(self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(core::arch::aarch64::vrshrq_n_s16::<SHIFT>(self.0)) }
    }

    #[inline(always)]
    fn lanes_up(self, n: usize) -> Self {
        // SAFETY: NEON is enabled. Extracting from zero || self shifts whole
        // signed lanes without changing any bits; n=0/8 are the endpoints.
        unsafe {
            let zero = vdupq_n_s16(0);
            Self(match n {
                0 => self.0,
                1 => vextq_s16::<7>(zero, self.0),
                2 => vextq_s16::<6>(zero, self.0),
                3 => vextq_s16::<5>(zero, self.0),
                4 => vextq_s16::<4>(zero, self.0),
                5 => vextq_s16::<3>(zero, self.0),
                6 => vextq_s16::<2>(zero, self.0),
                7 => vextq_s16::<1>(zero, self.0),
                8 => zero,
                _ => panic!("CDEF lane shift exceeds eight"),
            })
        }
    }

    #[inline(always)]
    fn lanes_down(self, n: usize) -> Self {
        // SAFETY: NEON is enabled. Extracting from self || zero preserves
        // the surviving lanes exactly, with the vacated lanes set to zero.
        unsafe {
            let zero = vdupq_n_s16(0);
            Self(match n {
                0 => self.0,
                1 => vextq_s16::<1>(self.0, zero),
                2 => vextq_s16::<2>(self.0, zero),
                3 => vextq_s16::<3>(self.0, zero),
                4 => vextq_s16::<4>(self.0, zero),
                5 => vextq_s16::<5>(self.0, zero),
                6 => vextq_s16::<6>(self.0, zero),
                7 => vextq_s16::<7>(self.0, zero),
                8 => zero,
                _ => panic!("CDEF lane shift exceeds eight"),
            })
        }
    }

    #[inline(always)]
    fn shuffle_bytes(self, pattern: &[u8; 16]) -> Self {
        // SAFETY: NEON kernels only; `pattern` holds 16 bytes. `tbl` zeroes
        // the bytes whose index is out of range.
        unsafe {
            Self(vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(self.0),
                vld1q_u8(pattern.as_ptr()),
            )))
        }
    }

    #[inline(always)]
    fn zip_lo(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vzip1q_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn zip_hi(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vzip2q_s16(self.0, other.0)) }
    }

    #[inline(always)]
    fn madd(self, other: Self) -> I32x4 {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            I32x4(vpaddq_s32(
                vmull_s16(vget_low_s16(self.0), vget_low_s16(other.0)),
                vmull_high_s16(self.0, other.0),
            ))
        }
    }

    #[inline(always)]
    fn paired_squares(self, other: Self) -> [I32x4; 2] {
        // SAFETY: the ISA caller enables NEON. CDEF line sums are at most
        // 8*128 in magnitude, so two squares fit i32 before weighting.
        unsafe {
            let a = vget_low_s16(self.0);
            let b = vget_low_s16(other.0);
            [
                I32x4(vmlal_s16(vmull_s16(a, a), b, b)),
                I32x4(vmlal_high_s16(
                    vmull_high_s16(self.0, self.0),
                    other.0,
                    other.0,
                )),
            ]
        }
    }

    #[inline(always)]
    fn sum_lanes4(x: [I32x4; 4]) -> I32x4 {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            I32x4(vpaddq_s32(
                vpaddq_s32(x[0].0, x[1].0),
                vpaddq_s32(x[2].0, x[3].0),
            ))
        }
    }
}

/// Register transpose shared by residual stores and fused reconstruction.
#[inline(always)]
fn transpose_i32x4(rows: &[I32x4]) -> [I32x4; 4] {
    transpose_i32x4_vectors([rows[0].0, rows[1].0, rows[2].0, rows[3].0]).map(I32x4)
}

/// Transpose a 4x4 tile held in four NEON registers.
#[inline(always)]
pub(crate) fn transpose_i32x4_vectors([a, b, c, d]: [int32x4_t; 4]) -> [int32x4_t; 4] {
    // SAFETY: called from NEON kernels.
    unsafe {
        let ab = vtrnq_s32(a, b);
        let cd = vtrnq_s32(c, d);
        [
            vcombine_s32(vget_low_s32(ab.0), vget_low_s32(cd.0)),
            vcombine_s32(vget_low_s32(ab.1), vget_low_s32(cd.1)),
            vcombine_s32(vget_high_s32(ab.0), vget_high_s32(cd.0)),
            vcombine_s32(vget_high_s32(ab.1), vget_high_s32(cd.1)),
        ]
    }
}

/// Two 8x8 byte transposes at once: byte `j` of the low (high) half of
/// output `i` is byte `i` of the low (high) half of input `j`. Its own
/// inverse.
#[inline(always)]
pub(crate) fn transpose_u8x16_pairs(r: [uint8x16_t; 8]) -> [uint8x16_t; 8] {
    // SAFETY: register-only. `trn` pairs elements within each 64-bit
    // half for 8- and 16-bit elements, and the 32-bit stage pairs
    // element 0 with 0 and 2 with 2: the two halves never mix.
    unsafe {
        let b: [uint8x16_t; 8] = [
            vtrn1q_u8(r[0], r[1]),
            vtrn2q_u8(r[0], r[1]),
            vtrn1q_u8(r[2], r[3]),
            vtrn2q_u8(r[2], r[3]),
            vtrn1q_u8(r[4], r[5]),
            vtrn2q_u8(r[4], r[5]),
            vtrn1q_u8(r[6], r[7]),
            vtrn2q_u8(r[6], r[7]),
        ];
        // Columns {0, 4}, {2, 6}, {1, 5}, {3, 7} of rows 0..4, then of
        // rows 4..8.
        let w = [
            trn_u16::<true>(b[0], b[2]),
            trn_u16::<false>(b[0], b[2]),
            trn_u16::<true>(b[1], b[3]),
            trn_u16::<false>(b[1], b[3]),
            trn_u16::<true>(b[4], b[6]),
            trn_u16::<false>(b[4], b[6]),
            trn_u16::<true>(b[5], b[7]),
            trn_u16::<false>(b[5], b[7]),
        ];
        [
            vreinterpretq_u8_u32(vtrn1q_u32(w[0], w[4])),
            vreinterpretq_u8_u32(vtrn1q_u32(w[2], w[6])),
            vreinterpretq_u8_u32(vtrn1q_u32(w[1], w[5])),
            vreinterpretq_u8_u32(vtrn1q_u32(w[3], w[7])),
            vreinterpretq_u8_u32(vtrn2q_u32(w[0], w[4])),
            vreinterpretq_u8_u32(vtrn2q_u32(w[2], w[6])),
            vreinterpretq_u8_u32(vtrn2q_u32(w[1], w[5])),
            vreinterpretq_u8_u32(vtrn2q_u32(w[3], w[7])),
        ]
    }
}

/// `trn1` (`FIRST`) or `trn2` of the 16-bit elements of `a` and `b`, as
/// 32-bit elements.
#[inline(always)]
fn trn_u16<const FIRST: bool>(a: uint8x16_t, b: uint8x16_t) -> uint32x4_t {
    // SAFETY: register-only.
    unsafe {
        let (a, b) = (vreinterpretq_u16_u8(a), vreinterpretq_u16_u8(b));
        vreinterpretq_u32_u16(if FIRST {
            vtrn1q_u16(a, b)
        } else {
            vtrn2q_u16(a, b)
        })
    }
}

impl I16x8 {
    /// Transpose the two byte slabs together before widening the sample taps.
    /// # Safety
    /// Starting at `origin`, all eight rows have sixteen accessible bytes.
    #[inline(always)]
    pub(super) unsafe fn load_u8_vertical16(
        data: &[u8],
        origin: usize,
        stride: usize,
    ) -> [Self; 16] {
        // SAFETY: each load is within the caller's checked tile. The
        // remaining operations act only on registers.
        unsafe {
            let rows = std::array::from_fn(|i| vld1q_u8(data.as_ptr().add(origin + i * stride)));
            let packed = transpose_u8x16_pairs(rows);
            let low = packed.map(|v| Self(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)))));
            let high = packed.map(|v| Self(vreinterpretq_s16_u16(vmovl_high_u8(v))));
            std::array::from_fn(|i| if i < 8 { low[i] } else { high[i - 8] })
        }
    }

    /// Narrow two transposed slabs, then transpose the packed bytes back.
    /// # Safety
    /// Starting at `origin`, all eight rows have sixteen writable bytes.
    #[inline(always)]
    pub(super) unsafe fn store_u8_vertical16(
        data: &mut [u8],
        origin: usize,
        stride: usize,
        taps: &[Self; 16],
    ) {
        // SAFETY: each store is within the caller's checked tile.
        unsafe {
            let packed =
                std::array::from_fn(|i| vqmovun_high_s16(vqmovun_s16(taps[i].0), taps[8 + i].0));
            let rows = transpose_u8x16_pairs(packed);
            for (i, row) in rows.into_iter().enumerate() {
                vst1q_u8(data.as_mut_ptr().add(origin + i * stride), row);
            }
        }
    }
}

impl crate::sao::high_simd::SaoWordLanes for I16x8 {
    #[inline(always)]
    fn offset_table(offsets: [i16; 4]) -> Self {
        Self::load(&crate::sao::high_simd::offset_words(offsets))
    }

    #[inline(always)]
    fn select_offset(self, table: Self) -> Self {
        use core::arch::aarch64::{
            vaddq_s16, vdupq_n_s16, vmulq_n_s16, vqtbl1q_u8, vreinterpretq_s16_u8,
            vreinterpretq_u8_s16,
        };
        // SAFETY: called from the NEON SAO entry. Each sign sum -2..2
        // selects both bytes of a word in its consecutive table slot.
        unsafe {
            let index = vaddq_s16(vmulq_n_s16(self.0, 0x0202), vdupq_n_s16(0x0504));
            Self(vreinterpretq_s16_u8(vqtbl1q_u8(
                vreinterpretq_u8_s16(table.0),
                vreinterpretq_u8_s16(index),
            )))
        }
    }
}
