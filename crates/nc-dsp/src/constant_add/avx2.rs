//! x86 byte lanes of the constant add (SSE2 instructions inside the AVX2
//! entry point).

use core::arch::x86_64::{
    __m128i, _mm_adds_epu8, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_set1_epi8, _mm_storel_epi64, _mm_storeu_si128, _mm_subs_epu8,
};

use super::ByteLanes;

#[derive(Clone, Copy)]
pub(crate) struct Bytes(__m128i);

impl ByteLanes for Bytes {
    #[inline(always)]
    fn splat(value: u8) -> Self {
        // SAFETY: the AVX2 entry point enables these instructions.
        unsafe { Self(_mm_set1_epi8(value as i8)) }
    }
    #[inline(always)]
    fn add_sat(self, other: Self) -> Self {
        // SAFETY: as `splat`.
        unsafe { Self(_mm_adds_epu8(self.0, other.0)) }
    }
    #[inline(always)]
    fn sub_sat(self, other: Self) -> Self {
        // SAFETY: as `splat`.
        unsafe { Self(_mm_subs_epu8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn load(src: *const u8, bytes: usize) -> Self {
        // SAFETY: the caller provides `bytes` readable bytes.
        unsafe {
            Self(match bytes {
                16 => _mm_loadu_si128(src.cast()),
                8 => _mm_loadl_epi64(src.cast()),
                _ => _mm_cvtsi32_si128(src.cast::<i32>().read_unaligned()),
            })
        }
    }
    #[inline(always)]
    unsafe fn store(self, dst: *mut u8, bytes: usize) {
        // SAFETY: the caller provides `bytes` writable bytes.
        unsafe {
            match bytes {
                16 => _mm_storeu_si128(dst.cast(), self.0),
                8 => _mm_storel_epi64(dst.cast(), self.0),
                _ => dst.cast::<i32>().write_unaligned(_mm_cvtsi128_si32(self.0)),
            }
        }
    }
}

/// [`super::add`] on x86.
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn add(
    residual: i32,
    width: usize,
    height: usize,
    dst: &mut [u8],
    stride: usize,
) {
    // SAFETY: AVX2 is enabled for this function.
    unsafe { super::add::<Bytes>(residual, width, height, dst, stride) }
}
