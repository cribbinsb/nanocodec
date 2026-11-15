//! AVX2 reconstruction: eight packed i16 residuals per group, saturating words.
use super::Residual16;
use core::arch::x86_64::{
    _mm_adds_epi16, _mm_cvtepu8_epi16, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_packus_epi16, _mm_set1_epi16, _mm_storel_epi64,
};

#[target_feature(enable = "avx2")]
pub(super) unsafe fn reconstruct(
    pred: &[u8],
    ps: usize,
    residual: Residual16<'_>,
    dst: &mut [u8],
    ds: usize,
    width: usize,
    height: usize,
) {
    let dc = _mm_set1_epi16(residual.constant_or_zero());
    for y in 0..height {
        let mut x = 0;
        while x + 8 <= width {
            // SAFETY: the validated row spans contain these eight pixels;
            // Samples covers eight i16 values, loaded without alignment assumptions.
            unsafe {
                let r = match residual {
                    Residual16::Constant(_) => dc,
                    Residual16::Samples(r) => _mm_loadu_si128(r.as_ptr().add(y * width + x).cast()),
                };
                let p = _mm_cvtepu8_epi16(_mm_loadl_epi64(pred.as_ptr().add(y * ps + x).cast()));
                let sum = _mm_adds_epi16(p, r);
                _mm_storel_epi64(
                    dst.as_mut_ptr().add(y * ds + x).cast(),
                    _mm_packus_epi16(sum, sum),
                );
            }
            x += 8;
        }
        if x < width {
            // SAFETY: the only tail is a complete four-pixel block row.
            unsafe {
                let r = match residual {
                    Residual16::Constant(_) => dc,
                    Residual16::Samples(r) => _mm_loadl_epi64(r.as_ptr().add(y * width + x).cast()),
                };
                let packed = pred.as_ptr().add(y * ps + x).cast::<i32>().read_unaligned();
                let p = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(packed));
                let sum = _mm_adds_epi16(p, r);
                dst.as_mut_ptr()
                    .add(y * ds + x)
                    .cast::<i32>()
                    .write_unaligned(_mm_cvtsi128_si32(_mm_packus_epi16(sum, sum)));
            }
        }
    }
}
