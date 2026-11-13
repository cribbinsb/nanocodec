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
    vextq_s16, vget_high_s32, vget_lane_u32, vget_low_s16, vget_low_s32, vget_low_u16, vld1_s16,
    vld1_u8, vld1q_s16, vld1q_s32, vld1q_u8, vmax_s16, vmaxq_s16, vmaxq_s32, vmin_s16, vminq_s16,
    vminq_s32, vminq_u16, vmlal_high_n_s16, vmlal_high_s16, vmlal_n_s16, vmlal_s16, vmlal_u8,
    vmlaq_laneq_s32, vmlaq_n_s32, vmovl_s16, vmovl_u8, vmovl_u16, vmul_s16, vmull_high_s16,
    vmull_s16, vmull_u8, vmulq_n_s32, vmulq_s16, vmulq_s32, vneg_s16, vnegq_s16, vorrq_s32,
    vpadalq_s16, vpaddq_s32, vqmovn_s32, vqmovn_u16, vqmovun_s16, vqmovun_s32, vqrdmulh_s16,
    vqrdmulhq_s16, vqsubq_u16, vqtbl1q_u8, vreinterpret_u32_u8, vreinterpretq_s16_s32,
    vreinterpretq_s16_u8, vreinterpretq_s16_u16, vreinterpretq_s32_u32, vreinterpretq_u8_s16,
    vreinterpretq_u16_s16, vreinterpretq_u32_s32, vrshlq_s32, vshl_s16, vshlq_s16, vshlq_s32,
    vshlq_u32, vshrq_n_s32, vshrq_n_u32, vst1_s16, vst1_u8, vst1q_s16, vst1q_s32, vsub_s16,
    vsubq_s16, vsubq_s32, vtrnq_s32, vzip1q_s16, vzip2q_s16,
};

use core::mem::MaybeUninit;

use super::{I16FilterLanes, I16Lanes, I16Pairs, I16x8Shuffles, I32Lanes, packed_pair};

/// Four `i32` lanes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct I32x4(int32x4_t);

/// Eight `i16` lanes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct I16x8(int16x8_t);

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
        vqmovn_s32(vshlq_s32(
            vaddq_s32(acc, vdupq_n_s32(round)),
            vdupq_n_s32(-(shift as i32)),
        ))
    }
}

impl I16Lanes for I16x8 {
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
    fn load(src: &[i16]) -> Self {
        let src = &src[..8];
        // SAFETY: NEON kernels only; `src` holds 8 values.
        unsafe { Self(vld1q_s16(src.as_ptr())) }
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
    fn acc_add(acc: Self::Acc, a: Self) -> Self::Acc {
        // SAFETY: NEON kernels only (module docs).
        unsafe { [vpadalq_s16(acc[0], a.0), acc[1]] }
    }

    #[inline(always)]
    fn acc_add_sq(acc: Self::Acc, a: Self) -> Self::Acc {
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
        let words = [
            round_shift_narrow(acc[0], round, shift),
            round_shift_narrow(acc[1], round, shift),
        ];
        // SAFETY: NEON kernels only; `dst` holds 8 bytes.
        unsafe {
            vst1_u8(
                dst.as_mut_ptr(),
                vqmovun_s16(vcombine_s16(words[0], words[1])),
            );
        }
    }

    #[inline(always)]
    fn acc_round_shift_narrow(acc: Self::Acc, round: i32, shift: u32) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            Self(vcombine_s16(
                round_shift_narrow(acc[0], round, shift),
                round_shift_narrow(acc[1], round, shift),
            ))
        }
    }

    #[inline(always)]
    fn acc_sum(acc: Self::Acc) -> i64 {
        // SAFETY: NEON kernels only (module docs).
        unsafe { vaddlvq_s32(acc[0]) + vaddlvq_s32(acc[1]) }
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

impl I16FilterLanes for I16x8 {
    #[inline(always)]
    fn load_halves(lo: &[i16], hi: &[i16]) -> Self {
        let (lo, hi) = (&lo[..4], &hi[..4]);
        // SAFETY: NEON kernels only; `lo` and `hi` hold 4 values each.
        unsafe { Self(vcombine_s16(vld1_s16(lo.as_ptr()), vld1_s16(hi.as_ptr()))) }
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
}

impl I16Pairs for I16x8 {
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
}

impl I16x8Shuffles for I16x8 {
    type Wide = I32x4;

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
    let r = [rows[0], rows[1], rows[2], rows[3]];
    // SAFETY: NEON kernels only (module docs).
    let out = unsafe {
        let t01 = vtrnq_s32(r[0].0, r[1].0);
        let t23 = vtrnq_s32(r[2].0, r[3].0);
        [
            vcombine_s32(vget_low_s32(t01.0), vget_low_s32(t23.0)),
            vcombine_s32(vget_low_s32(t01.1), vget_low_s32(t23.1)),
            vcombine_s32(vget_high_s32(t01.0), vget_high_s32(t23.0)),
            vcombine_s32(vget_high_s32(t01.1), vget_high_s32(t23.1)),
        ]
    };
    out.map(I32x4)
}
