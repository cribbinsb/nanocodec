//! AVX2 reconstruction: eight packed i32 residuals per group, then saturating words.
use super::Residual32;
use core::arch::x86_64::{
    _mm_adds_epi16, _mm_cvtepu8_epi16, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi16, _mm_storel_epi64,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256,
};

#[target_feature(enable = "avx2")]
pub(super) unsafe fn reconstruct(
    pred: &[u8],
    ps: usize,
    residual: Residual32<'_>,
    dst: &mut [u8],
    ds: usize,
    width: usize,
    height: usize,
) {
    let dc = _mm_set1_epi16(
        (match residual {
            Residual32::Constant(dc) => dc,
            Residual32::Samples(_) => 0,
        })
        .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
    );
    for y in 0..height {
        let mut x = 0;
        while x + 8 <= width {
            // SAFETY: the validated row spans contain these eight pixels;
            // Samples covers eight i32 values, loaded without alignment assumptions.
            unsafe {
                let r = match residual {
                    Residual32::Constant(_) => dc,
                    Residual32::Samples(r) => {
                        let r = _mm256_loadu_si256(r.as_ptr().add(y * width + x).cast());
                        _mm_packs_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256::<1>(r))
                    }
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
                    Residual32::Constant(_) => dc,
                    Residual32::Samples(r) => {
                        let four = _mm_loadu_si128(r.as_ptr().add(y * width + x).cast());
                        _mm_packs_epi32(four, four)
                    }
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
