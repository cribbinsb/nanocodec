//! NEON byte lanes of the constant add.

use core::arch::aarch64::{
    uint8x16_t, vcombine_u8, vdup_n_u8, vdupq_n_u8, vdupq_n_u32, vget_low_u8, vgetq_lane_u32,
    vld1_u8, vld1q_u8, vqaddq_u8, vqsubq_u8, vreinterpretq_u8_u32, vreinterpretq_u32_u8,
    vsetq_lane_u32, vst1_u8, vst1q_u8,
};

use super::ByteLanes;

#[derive(Clone, Copy)]
pub(crate) struct Bytes(uint8x16_t);

impl ByteLanes for Bytes {
    #[inline(always)]
    fn splat(value: u8) -> Self {
        // SAFETY: NEON is always present on AArch64.
        unsafe { Self(vdupq_n_u8(value)) }
    }
    #[inline(always)]
    fn add_sat(self, other: Self) -> Self {
        // SAFETY: as `splat`.
        unsafe { Self(vqaddq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    fn sub_sat(self, other: Self) -> Self {
        // SAFETY: as `splat`.
        unsafe { Self(vqsubq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn load(src: *const u8, bytes: usize) -> Self {
        // SAFETY: the caller provides `bytes` readable bytes.
        unsafe {
            Self(match bytes {
                16 => vld1q_u8(src),
                8 => vcombine_u8(vld1_u8(src), vdup_n_u8(0)),
                _ => vreinterpretq_u8_u32(vsetq_lane_u32::<0>(
                    src.cast::<u32>().read_unaligned(),
                    vdupq_n_u32(0),
                )),
            })
        }
    }
    #[inline(always)]
    unsafe fn store(self, dst: *mut u8, bytes: usize) {
        // SAFETY: the caller provides `bytes` writable bytes.
        unsafe {
            match bytes {
                16 => vst1q_u8(dst, self.0),
                8 => vst1_u8(dst, vget_low_u8(self.0)),
                _ => dst
                    .cast::<u32>()
                    .write_unaligned(vgetq_lane_u32::<0>(vreinterpretq_u32_u8(self.0))),
            }
        }
    }
}

/// [`super::add`] on NEON.
///
/// # Safety
///
/// None beyond the slice contract (NEON is always present on `AArch64`).
pub(crate) unsafe fn add(
    residual: i32,
    width: usize,
    height: usize,
    dst: &mut [u8],
    stride: usize,
) {
    // SAFETY: NEON is always enabled on AArch64.
    unsafe { super::add::<Bytes>(residual, width, height, dst, stride) }
}
