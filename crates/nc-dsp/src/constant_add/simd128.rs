//! WebAssembly SIMD128 byte lanes of the constant add.

use core::arch::wasm32::{
    u8x16_add_sat, u8x16_splat, u8x16_sub_sat, v128, v128_load, v128_load32_zero, v128_load64_zero,
    v128_store, v128_store32_lane, v128_store64_lane,
};

use super::ByteLanes;

#[derive(Clone, Copy)]
pub(crate) struct Bytes(v128);

impl ByteLanes for Bytes {
    #[inline(always)]
    fn splat(value: u8) -> Self {
        Self(u8x16_splat(value))
    }
    #[inline(always)]
    fn add_sat(self, other: Self) -> Self {
        Self(u8x16_add_sat(self.0, other.0))
    }
    #[inline(always)]
    fn sub_sat(self, other: Self) -> Self {
        Self(u8x16_sub_sat(self.0, other.0))
    }
    #[inline(always)]
    unsafe fn load(src: *const u8, bytes: usize) -> Self {
        // SAFETY: the caller provides `bytes` readable bytes.
        unsafe {
            Self(match bytes {
                16 => v128_load(src.cast()),
                8 => v128_load64_zero(src.cast()),
                _ => v128_load32_zero(src.cast()),
            })
        }
    }
    #[inline(always)]
    unsafe fn store(self, dst: *mut u8, bytes: usize) {
        // SAFETY: the caller provides `bytes` writable bytes.
        unsafe {
            match bytes {
                16 => v128_store(dst.cast(), self.0),
                8 => v128_store64_lane::<0>(self.0, dst.cast()),
                _ => v128_store32_lane::<0>(self.0, dst.cast()),
            }
        }
    }
}

/// [`super::add`] on SIMD128.
///
/// # Safety
///
/// None beyond the slice contract (the build enables SIMD128).
pub(crate) unsafe fn add(
    residual: i32,
    width: usize,
    height: usize,
    dst: &mut [u8],
    stride: usize,
) {
    // SAFETY: SIMD128 is enabled for the whole build.
    unsafe { super::add::<Bytes>(residual, width, height, dst, stride) }
}
