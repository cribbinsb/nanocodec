//! AVX2 copy/average rows and sixteen-pixel byte-to-word preparation.

use super::rows::Ops;
use core::arch::x86_64::{
    _mm_avg_epu8, _mm_cvtepu8_epi16, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_set1_epi16, _mm_slli_epi16, _mm_storel_epi64, _mm_storeu_si128,
    _mm_sub_epi16, _mm256_avg_epu8, _mm256_cvtepu8_epi16, _mm256_loadu_si256, _mm256_set1_epi16,
    _mm256_slli_epi16, _mm256_storeu_si256, _mm256_sub_epi16,
};

pub(super) struct Avx2;
impl Ops for Avx2 {
    const COPY_BYTES: usize = 32;
    const PREP_BYTES: usize = 16;
    const ROWS: usize = 1;
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    unsafe fn copy_width<const W: usize, const AVG: bool>(
        src: &[u8],
        ss: usize,
        dst: &mut [u8],
        ds: usize,
        w: usize,
        h: usize,
    ) {
        // SAFETY: forwarded rectangle and ISA contract. Only this width's
        // loop can contribute register saves to its frame.
        unsafe { super::rows::copy_width::<Self, W, AVG>(src, ss, dst, ds, w, h) }
    }
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    unsafe fn prep_width<const W: usize, const SHIFT: i32>(
        src: &[u8],
        ss: usize,
        dst: &mut [i16],
        ds: usize,
        w: usize,
        h: usize,
        bias: i16,
    ) {
        // SAFETY: validated footprint and bounded word precision.
        unsafe { super::rows::prep_width::<Self, W, SHIFT>(src, ss, dst, ds, w, h, bias) }
    }
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    unsafe fn prep_unbiased_width<const W: usize>(
        src: &[u8],
        ss: usize,
        dst: &mut [i16],
        ds: usize,
        w: usize,
        h: usize,
    ) {
        // SAFETY: validated footprint; byte samples <<4 fit signed words.
        unsafe { super::rows::prep_width::<Self, W, 4>(src, ss, dst, ds, w, h, 0) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn copy<const N: usize, const AVG: bool>(src: *const u8, dst: *mut u8) {
        // SAFETY: the validated row contains exactly the selected span.
        unsafe {
            if N == 32 {
                let a = _mm256_loadu_si256(src.cast());
                let a = if AVG {
                    _mm256_avg_epu8(a, _mm256_loadu_si256(dst.cast()))
                } else {
                    a
                };
                _mm256_storeu_si256(dst.cast(), a);
            } else if N == 16 {
                let a = _mm_loadu_si128(src.cast());
                let a = if AVG {
                    _mm_avg_epu8(a, _mm_loadu_si128(dst.cast()))
                } else {
                    a
                };
                _mm_storeu_si128(dst.cast(), a);
            } else if N == 8 {
                let a = _mm_loadl_epi64(src.cast());
                let a = if AVG {
                    _mm_avg_epu8(a, _mm_loadl_epi64(dst.cast()))
                } else {
                    a
                };
                _mm_storel_epi64(dst.cast(), a);
            } else if N == 4 {
                let a = _mm_cvtsi32_si128(src.cast::<i32>().read_unaligned());
                let a = if AVG {
                    _mm_avg_epu8(a, _mm_cvtsi32_si128(dst.cast::<i32>().read_unaligned()))
                } else {
                    a
                };
                dst.cast::<i32>().write_unaligned(_mm_cvtsi128_si32(a));
            } else {
                super::scalar::Scalar::copy::<N, AVG>(src, dst);
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn prep<const N: usize, const SHIFT: i32>(src: *const u8, dst: *mut i16, bias: i16) {
        // SAFETY: source contains N bytes and destination N words. Partial
        // vectors use matching narrow loads and stores, not padded reads.
        unsafe {
            if N == 16 {
                let a = _mm256_cvtepu8_epi16(_mm_loadu_si128(src.cast()));
                let a = _mm256_sub_epi16(_mm256_slli_epi16::<SHIFT>(a), _mm256_set1_epi16(bias));
                _mm256_storeu_si256(dst.cast(), a);
            } else if N == 8 || N == 4 {
                let a = if N == 8 {
                    _mm_loadl_epi64(src.cast())
                } else {
                    _mm_cvtsi32_si128(src.cast::<i32>().read_unaligned())
                };
                let a = _mm_sub_epi16(
                    _mm_slli_epi16::<SHIFT>(_mm_cvtepu8_epi16(a)),
                    _mm_set1_epi16(bias),
                );
                if N == 8 {
                    _mm_storeu_si128(dst.cast(), a);
                } else {
                    _mm_storel_epi64(dst.cast(), a);
                }
            } else {
                super::scalar::Scalar::prep::<N, SHIFT>(src, dst, bias);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn copy<const AVG: bool>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    // SAFETY: validated rectangle; dispatcher checked AVX2 support.
    unsafe { super::rows::copy::<Avx2, AVG>(src, ss, dst, ds, w, h) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn prep(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    size: (usize, usize),
    scale: (u32, i16),
) {
    let (w, h) = size;
    let (shift, bias) = scale;
    // SAFETY: validated rectangle and precision; AVX2 is enabled.
    unsafe { super::rows::prep_entry!(Avx2, src, ss, dst, ds, w, h, shift, bias) }
}

/// Common eight-bit AV1/AV2 compound precision: unbiased sample << 4.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn prep_unbiased(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    (w, h): (usize, usize),
) {
    // SAFETY: validated rectangle, byte << 4 fits i16, and this entry's ISA is enabled.
    unsafe { super::rows::prep_unbiased::<Avx2>(src, ss, dst, ds, w, h) }
}
