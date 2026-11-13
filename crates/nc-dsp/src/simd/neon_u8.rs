//! `AArch64` sixteen-byte vectors for the 8-bit CDEF kernel
//! ([`U8FilterLanes`]). See the `simd` module docs for the safety argument:
//! the methods only run inside `#[target_feature(enable = "neon")]`
//! kernels; stores index their slices, loads take pointers the caller
//! checked (one window check per block).

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    int16x8_t, uint8x16_t, vabdq_u8, vaddq_s16, vaddq_u8, vcgtq_u8, vcltzq_s16, vcombine_u8,
    vcreate_u8, vdup_n_s8, vdupq_n_s8, vdupq_n_s16, vdupq_n_u8, veorq_u8, vget_high_u8,
    vget_low_s8, vget_low_u8, vgetq_lane_u32, vld1_u8, vmaxq_u8, vminq_u8, vmlal_high_s8, vmlal_s8,
    vmovl_high_u8, vmovl_u8, vqmovun_high_s16, vqmovun_s16, vqsubq_u8, vreinterpretq_s8_u8,
    vreinterpretq_s16_u16, vreinterpretq_u8_s8, vreinterpretq_u32_u8, vshlq_u8, vshrq_n_s16,
    vst1_u8, vsubq_u8,
};

use super::U8FilterLanes;

/// Sixteen `u8` lanes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct U8x16(uint8x16_t);

/// Sixteen 16-bit sums: byte lanes `0..8` in `.0`, `8..16` in `.1`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sums(int16x8_t, int16x8_t);

/// The four bytes at `bytes`, as a little-endian word.
///
/// # Safety
///
/// `bytes` points to four readable bytes.
#[inline(always)]
unsafe fn word(bytes: *const u8) -> u64 {
    // SAFETY: the caller provides four readable bytes.
    u64::from(u32::from_le(unsafe {
        bytes.cast::<u32>().read_unaligned()
    }))
}

impl U8FilterLanes for U8x16 {
    type Acc = Sums;

    #[inline(always)]
    unsafe fn load_8x2(top: *const u8, bottom: *const u8) -> Self {
        // SAFETY: the caller provides eight readable bytes at each pointer
        // and NEON.
        unsafe { Self(vcombine_u8(vld1_u8(top), vld1_u8(bottom))) }
    }

    #[inline(always)]
    unsafe fn load_4x4(rows: [*const u8; 4]) -> Self {
        // SAFETY: the caller provides four readable bytes at each pointer
        // and NEON.
        unsafe {
            let low = word(rows[0]) | word(rows[1]) << 32;
            let high = word(rows[2]) | word(rows[3]) << 32;
            Self(vcombine_u8(vcreate_u8(low), vcreate_u8(high)))
        }
    }

    #[inline(always)]
    fn store_8x2(self, top: &mut [u8], bottom: &mut [u8]) {
        let (top, bottom) = (&mut top[..8], &mut bottom[..8]);
        // SAFETY: NEON kernels only; both slices hold 8 bytes.
        unsafe {
            vst1_u8(top.as_mut_ptr(), vget_low_u8(self.0));
            vst1_u8(bottom.as_mut_ptr(), vget_high_u8(self.0));
        }
    }

    #[inline(always)]
    fn store_4x4(self, dst: &mut [u8], stride: usize) {
        // SAFETY: NEON kernels only (module docs).
        let words = unsafe {
            let w = vreinterpretq_u32_u8(self.0);
            [
                vgetq_lane_u32::<0>(w),
                vgetq_lane_u32::<1>(w),
                vgetq_lane_u32::<2>(w),
                vgetq_lane_u32::<3>(w),
            ]
        };
        for (r, w) in words.iter().enumerate() {
            dst[r * stride..r * stride + 4].copy_from_slice(&w.to_le_bytes());
        }
    }

    #[inline(always)]
    fn splat(value: u8) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vdupq_n_u8(value)) }
    }

    #[inline(always)]
    fn abs_diff(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vabdq_u8(self.0, other.0)) }
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vcgtq_u8(self.0, other.0)) }
    }

    #[inline(always)]
    fn shift_count(shift: u32) -> Self {
        // SAFETY: NEON kernels only (module docs). A negative left shift is a
        // logical right shift.
        unsafe { Self(vreinterpretq_u8_s8(vdupq_n_s8(-(shift as i8)))) }
    }

    #[inline(always)]
    fn shr(self, count: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vshlq_u8(self.0, vreinterpretq_s8_u8(count.0))) }
    }

    #[inline(always)]
    fn sub_sat(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vqsubq_u8(self.0, other.0)) }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vminq_u8(self.0, other.0)) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vmaxq_u8(self.0, other.0)) }
    }

    #[inline(always)]
    fn negate_where(self, mask: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vsubq_u8(veorq_u8(self.0, mask.0), mask.0)) }
    }

    #[inline(always)]
    fn add_wrapping(self, other: Self) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Self(vaddq_u8(self.0, other.0)) }
    }

    #[inline(always)]
    fn acc_zero() -> Sums {
        // SAFETY: NEON kernels only (module docs).
        unsafe { Sums(vdupq_n_s16(0), vdupq_n_s16(0)) }
    }

    #[inline(always)]
    fn acc_mla(acc: Sums, values: Self, weight: i8) -> Sums {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            let v = vreinterpretq_s8_u8(values.0);
            Sums(
                vmlal_s8(acc.0, vget_low_s8(v), vdup_n_s8(weight)),
                vmlal_high_s8(acc.1, v, vdupq_n_s8(weight)),
            )
        }
    }

    #[inline(always)]
    fn add_rounded(self, acc: Sums) -> Self {
        // SAFETY: NEON kernels only (module docs).
        unsafe {
            let x_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(self.0)));
            let x_hi = vreinterpretq_s16_u16(vmovl_high_u8(self.0));
            let lo = vaddq_s16(x_lo, rounded_sum(acc.0));
            let hi = vaddq_s16(x_hi, rounded_sum(acc.1));
            Self(vqmovun_high_s16(vqmovun_s16(lo), hi))
        }
    }
}

/// `(8 + sum - (sum < 0)) >> 4` per lane.
#[inline(always)]
fn rounded_sum(sum: int16x8_t) -> int16x8_t {
    // SAFETY: NEON kernels only (module docs). The comparison is -1 where the
    // sum is negative.
    unsafe {
        let negative = vreinterpretq_s16_u16(vcltzq_s16(sum));
        vshrq_n_s16::<4>(vaddq_s16(vaddq_s16(sum, vdupq_n_s16(8)), negative))
    }
}
