//! Packed libavc-style H.264 8x8 averages and directional row selection.

use core::arch::x86_64::{
    __m128i, _mm_and_si128, _mm_avg_epu8, _mm_loadu_si128, _mm_set1_epi8, _mm_sub_epi8,
    _mm_xor_si128, _mm256_broadcastsi128_si256, _mm256_castsi128_si256, _mm256_loadu_si256,
    _mm256_or_si256, _mm256_shuffle_epi8, _mm256_storeu_si256, _mm512_castsi256_si512,
    _mm512_loadu_si512, _mm512_permutex2var_epi8, _mm512_storeu_si512,
};

use super::{H264Intra8x8Refs, h264_intra_8x8_mode as mode, rows};

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn averages(refs: &H264Intra8x8Refs, first: usize) -> (__m128i, __m128i) {
    // SAFETY: the maximum first column is ten and all three 16-byte
    // loads end inside the padded 32-byte border.
    unsafe {
        let start = refs.border.as_ptr().add(first);
        let a = _mm_loadu_si128(start.cast());
        let b = _mm_loadu_si128(start.add(1).cast());
        let c = _mm_loadu_si128(start.add(2).cast());
        let parity = _mm_and_si128(_mm_xor_si128(a, c), _mm_set1_epi8(1));
        let outer = _mm_sub_epi8(_mm_avg_epu8(a, c), parity);
        (_mm_avg_epu8(a, b), _mm_avg_epu8(outer, b))
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn predict(refs: &H264Intra8x8Refs, mode: u8, out: &mut [u8; 64]) {
    if mode < mode::DIAGONAL_DOWN_LEFT {
        return rows::plain(refs, mode, out);
    }
    let index = usize::from(mode - mode::DIAGONAL_DOWN_LEFT);
    // SAFETY: the dispatcher checks that the mode is valid. Both mask
    // and output vectors lie within their respective arrays.
    unsafe {
        let &first = rows::FIRST.get_unchecked(index);
        let [pair_mask, triple_mask] = rows::MASKS.get_unchecked(index);
        let (pair, triple) = averages(refs, usize::from(first));
        let pair = _mm256_broadcastsi128_si256(pair);
        let triple = _mm256_broadcastsi128_si256(triple);
        for offset in [0, 32] {
            let p = _mm256_loadu_si256(pair_mask.as_ptr().add(offset).cast());
            let q = _mm256_loadu_si256(triple_mask.as_ptr().add(offset).cast());
            let block =
                _mm256_or_si256(_mm256_shuffle_epi8(pair, p), _mm256_shuffle_epi8(triple, q));
            _mm256_storeu_si256(out.as_mut_ptr().add(offset).cast(), block);
        }
    }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi")]
pub(super) unsafe fn predict_icl(refs: &H264Intra8x8Refs, mode: u8, out: &mut [u8; 64]) {
    if mode < mode::DIAGONAL_DOWN_LEFT {
        return rows::plain(refs, mode, out);
    }
    let index = usize::from(mode - mode::DIAGONAL_DOWN_LEFT);
    // SAFETY: the dispatcher checks the mode and CPU. Indices are all
    // address only the first 16 bytes of either input table.
    unsafe {
        let &first = rows::FIRST.get_unchecked(index);
        let permutation = rows::ICL_INDICES.get_unchecked(index);
        let (pair, triple) = averages(refs, usize::from(first));
        let pair = _mm512_castsi256_si512(_mm256_castsi128_si256(pair));
        let triple = _mm512_castsi256_si512(_mm256_castsi128_si256(triple));
        let indices = _mm512_loadu_si512(permutation.as_ptr().cast());
        let block = _mm512_permutex2var_epi8(pair, indices, triple);
        _mm512_storeu_si512(out.as_mut_ptr().cast(), block);
    }
}
