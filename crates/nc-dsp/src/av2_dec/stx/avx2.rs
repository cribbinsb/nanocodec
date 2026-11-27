//! The input-pair lane schedule of dav2d's `stx4_8bpc_avx2` and
//! `stx8_8bpc_avx2` (BSD-2-Clause, `VideoLAN` and dav2d authors).

use core::arch::x86_64::{
    _mm_loadu_si128, _mm256_add_epi32, _mm256_cmpgt_epi32, _mm256_cvtepi8_epi16, _mm256_madd_epi16,
    _mm256_packs_epi32, _mm256_set1_epi32, _mm256_srai_epi32, _mm256_storeu_si256,
    _mm256_sub_epi32, _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
};

#[target_feature(enable = "avx2")]
pub(super) fn matrix<const N: usize, const GROUPS: usize>(cf: &[i16], kernel: &[i8]) -> [i16; N] {
    const { assert!(N == GROUPS * 16) };
    let inputs = cf.len();
    assert!(kernel.len() >= inputs.next_multiple_of(2) * N);
    let round = _mm256_set1_epi32(63);
    let mut low = [round; GROUPS];
    let mut high = [round; GROUPS];
    for i in (0..inputs).step_by(2) {
        let a = i32::from(cf[i] as u16);
        let b = i32::from(cf.get(i + 1).copied().unwrap_or(0));
        let pair = _mm256_set1_epi32(a | (b << 16));
        for group in 0..GROUPS {
            let offset = i * N + group * 16;
            // SAFETY: the check covers both kernel rows and all groups.
            unsafe {
                let first =
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(kernel.as_ptr().add(offset).cast()));
                let second =
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(kernel.as_ptr().add(offset + N).cast()));
                let dot0 = _mm256_madd_epi16(pair, _mm256_unpacklo_epi16(first, second));
                let dot1 = _mm256_madd_epi16(pair, _mm256_unpackhi_epi16(first, second));
                low[group] = _mm256_add_epi32(low[group], dot0);
                high[group] = _mm256_add_epi32(high[group], dot1);
            }
        }
    }
    let minus_one = _mm256_set1_epi32(-1);
    let mut out = [0; N];
    for group in 0..GROUPS {
        let a = low[group];
        let b = high[group];
        let a = _mm256_srai_epi32::<7>(_mm256_sub_epi32(a, _mm256_cmpgt_epi32(a, minus_one)));
        let b = _mm256_srai_epi32::<7>(_mm256_sub_epi32(b, _mm256_cmpgt_epi32(b, minus_one)));
        // SAFETY: the output contains one full sixteen-word vector per group.
        unsafe {
            _mm256_storeu_si256(
                out.as_mut_ptr().add(group * 16).cast(),
                _mm256_packs_epi32(a, b),
            );
        };
    }
    out
}
