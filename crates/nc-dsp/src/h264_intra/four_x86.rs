use core::arch::x86_64::{
    _mm_and_si128, _mm_avg_epu8, _mm_loadu_si128, _mm_or_si128, _mm_set1_epi8, _mm_shuffle_epi8,
    _mm_srli_si128, _mm_storeu_si128, _mm_sub_epi8, _mm_xor_si128,
};

use super::{Kernels, MASKS};

pub(super) const KERNELS: Kernels = Kernels {
    prepare,
    select,
    single,
};

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn prepare(line: &[u8; 16]) -> [u8; 32] {
    let mut output = [0; 32];
    // SAFETY: the input is 16 valid bytes and each store writes one half of output.
    unsafe {
        let a = _mm_loadu_si128(line.as_ptr().cast());
        let b = _mm_srli_si128::<1>(a);
        let c = _mm_srli_si128::<2>(a);
        let pairs = _mm_avg_epu8(a, b);
        // libavc forms the truncated outer average before rounding with b.
        let parity = _mm_and_si128(_mm_xor_si128(a, c), _mm_set1_epi8(1));
        let outer = _mm_sub_epi8(_mm_avg_epu8(a, c), parity);
        let triples = _mm_avg_epu8(outer, b);
        _mm_storeu_si128(output.as_mut_ptr().cast(), pairs);
        _mm_storeu_si128(output.as_mut_ptr().add(16).cast(), triples);
    }
    output
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn select(table: &[u8; 32], mode: usize) -> [u8; 16] {
    // SAFETY: the dispatcher checks the directional mode.
    let masks = unsafe { MASKS.get_unchecked(mode) };
    let mut output = [0; 16];
    // SAFETY: each table, permutation mask and destination is 16 valid bytes.
    unsafe {
        let pairs = _mm_loadu_si128(table.as_ptr().cast());
        let triples = _mm_loadu_si128(table.as_ptr().add(16).cast());
        let pair_mask = _mm_loadu_si128(masks[0].as_ptr().cast());
        let triple_mask = _mm_loadu_si128(masks[1].as_ptr().cast());
        let row = _mm_or_si128(
            _mm_shuffle_epi8(pairs, pair_mask),
            _mm_shuffle_epi8(triples, triple_mask),
        );
        _mm_storeu_si128(output.as_mut_ptr().cast(), row);
    }
    output
}

#[target_feature(enable = "avx2")]
unsafe fn single(line: &[u8; 16], mode: usize) -> [u8; 16] {
    // SAFETY: both operations share the checked ISA, line and mode.
    unsafe { select(&prepare(line), mode) }
}
