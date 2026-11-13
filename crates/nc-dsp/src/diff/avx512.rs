//! AVX-512 variance with 32 signed-16-bit differences per vector. As in
//! SVT-AV1 v4.3.0 `variance_avx512.c` (Intel 2021, BSD-2-Clause and AOM
//! patent licence), sums stay narrow until the final reduction; squared
//! differences use paired widening multiplies. Our block contract bounds
//! each narrow lane by 2*64*255=32640, including 64-wide blocks.

use super::{DiffFns, Variance, VarianceEnergy};
use core::arch::x86_64::{
    _mm256_maskz_loadu_epi8, _mm512_add_epi16, _mm512_add_epi32, _mm512_cvtepu8_epi16,
    _mm512_madd_epi16, _mm512_reduce_add_epi32, _mm512_set1_epi16, _mm512_setzero_si512,
    _mm512_sub_epi16,
};

pub(crate) fn install(table: &mut DiffFns) {
    // Zen 5 / SPR: 16-wide variance loses (0.65–0.84x); wider blocks win.
    table.variance[3..].copy_from_slice(&[variance::<32>, variance::<64>]);
    table.variance_energy[3..].copy_from_slice(&[energy::<32>, energy::<64>]);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn variance<const W: usize>(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    h: usize,
) -> Variance {
    // SAFETY: the dispatcher's validated block contract is forwarded.
    unsafe { accumulate::<W, false>(src, ss, pred, ps, h).residual }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn energy<const W: usize>(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    h: usize,
) -> VarianceEnergy {
    // SAFETY: the dispatcher's validated block contract is forwarded.
    unsafe { accumulate::<W, true>(src, ss, pred, ps, h) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn accumulate<const W: usize, const ENERGY: bool>(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    h: usize,
) -> VarianceEnergy {
    const LANES: usize = 32;
    let mask = u32::MAX >> (LANES - W.min(LANES));
    let zero = _mm512_setzero_si512();
    let (mut sum, mut squared, mut psum, mut psquared) = (zero, zero, zero, zero);
    for y in 0..h {
        for x in (0..W).step_by(LANES) {
            // SAFETY: masked loads read only W.min(32) bytes within row y.
            let (a, b) = unsafe {
                (
                    _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(
                        mask,
                        src.as_ptr().add(y * ss + x).cast(),
                    )),
                    _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(
                        mask,
                        pred.as_ptr().add(y * ps + x).cast(),
                    )),
                )
            };
            let d = _mm512_sub_epi16(a, b);
            sum = _mm512_add_epi16(sum, d);
            squared = _mm512_add_epi32(squared, _mm512_madd_epi16(d, d));
            if ENERGY {
                psum = _mm512_add_epi16(psum, b);
                psquared = _mm512_add_epi32(psquared, _mm512_madd_epi16(b, b));
            }
        }
    }
    let ones = _mm512_set1_epi16(1);
    VarianceEnergy {
        residual: Variance {
            sum: _mm512_reduce_add_epi32(_mm512_madd_epi16(sum, ones)),
            sse: _mm512_reduce_add_epi32(squared) as u32,
        },
        prediction: Variance {
            sum: _mm512_reduce_add_epi32(_mm512_madd_epi16(psum, ones)),
            sse: _mm512_reduce_add_epi32(psquared) as u32,
        },
    }
}
