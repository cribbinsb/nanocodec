//! Ice Lake copy/average rows in 64-byte spans and 32-pixel word prep.
//! Exact smaller spans use the family's AVX2 operations. SVT-AV1's BSD-2
//! `copy_wxh_avx512.c` (Meta 2024) similarly keeps short copies narrow.
use super::rows::Ops;
use core::arch::x86_64::{
    _mm_cvtsi32_si128, _mm256_loadu_si256, _mm512_avg_epu8, _mm512_cvtepu8_epi16,
    _mm512_loadu_si512, _mm512_set1_epi16, _mm512_sll_epi16, _mm512_storeu_si512, _mm512_sub_epi16,
};
pub(super) struct Avx512;
impl Ops for Avx512 {
    const COPY_BYTES: usize = 64;
    const PREP_BYTES: usize = 32;
    const ROWS: usize = 1;
    #[inline(never)]
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
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
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
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
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
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
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
    unsafe fn copy<const N: usize, const AVG: bool>(src: *const u8, dst: *mut u8) {
        // SAFETY: exactly N readable/writable bytes; the table requires ICL.
        unsafe {
            if N == 64 {
                let a = _mm512_loadu_si512(src.cast());
                let a = if AVG {
                    _mm512_avg_epu8(a, _mm512_loadu_si512(dst.cast()))
                } else {
                    a
                };
                _mm512_storeu_si512(dst.cast(), a);
            } else {
                super::avx2::Avx2::copy::<N, AVG>(src, dst);
            }
        }
    }
    #[inline]
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
    unsafe fn prep<const N: usize, const SHIFT: i32>(src: *const u8, dst: *mut i16, bias: i16) {
        // SAFETY: N byte inputs and word outputs, bounded to signed i16.
        unsafe {
            if N == 32 {
                let a = _mm512_cvtepu8_epi16(_mm256_loadu_si256(src.cast()));
                let a = _mm512_sub_epi16(
                    _mm512_sll_epi16(a, _mm_cvtsi32_si128(SHIFT)),
                    _mm512_set1_epi16(bias),
                );
                _mm512_storeu_si512(dst.cast(), a);
            } else {
                super::avx2::Avx2::prep::<N, SHIFT>(src, dst, bias);
            }
        }
    }
}
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn copy<const AVG: bool>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    // SAFETY: validated rectangle; CPU support checked by dispatch.
    unsafe { super::rows::copy::<Avx512, AVG>(src, ss, dst, ds, w, h) }
}
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
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
    // SAFETY: validated footprint and precision; ICL includes every feature.
    unsafe { super::rows::prep_entry!(Avx512, src, ss, dst, ds, w, h, shift, bias) }
}

/// Common eight-bit AV1/AV2 compound precision: unbiased sample << 4.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn prep_unbiased(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    (w, h): (usize, usize),
) {
    // SAFETY: validated rectangle, byte << 4 fits i16, and this entry's ISA is enabled.
    unsafe { super::rows::prep_unbiased::<Avx512>(src, ss, dst, ds, w, h) }
}
