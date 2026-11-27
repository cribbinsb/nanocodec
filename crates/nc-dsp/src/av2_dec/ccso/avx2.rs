//! Byte classification and packed offset lookup from dav2d's CCSO AVX2
//! kernels, BSD-2-Clause. Copyright © 2025–2026, `VideoLAN` and dav2d
//! authors and Two Orioles, LLC. See `THIRD_PARTY.md`.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m256i, _mm_cvtsi32_si128, _mm_loadu_si128, _mm256_abs_epi8, _mm256_add_epi8,
    _mm256_adds_epu8, _mm256_and_si256, _mm256_blendv_epi8, _mm256_broadcastsi128_si256,
    _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_or_si256, _mm256_packus_epi16,
    _mm256_permute4x64_epi64, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_slli_epi16, _mm256_srl_epi16, _mm256_srli_epi16,
    _mm256_storeu_si256, _mm256_sub_epi8, _mm256_subs_epu8,
};

use super::{CcsoFilter, CcsoMode};

#[target_feature(enable = "avx2")]
unsafe fn load_luma<const CHROMA: bool>(luma: *const u8) -> __m256i {
    // SAFETY: the caller provides 32 contiguous luma bytes, or 64 for chroma.
    unsafe {
        let lo = _mm256_loadu_si256(luma.cast());
        if CHROMA {
            let hi = _mm256_loadu_si256(luma.add(32).cast());
            let even = _mm256_set1_epi16(255);
            _mm256_packus_epi16(_mm256_and_si256(lo, even), _mm256_and_si256(hi, even))
        } else {
            lo
        }
    }
}

#[target_feature(enable = "avx2")]
fn classify<const THREE: bool>(center: __m256i, neighbour: __m256i, threshold: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let low = _mm256_subs_epu8(_mm256_subs_epu8(center, neighbour), threshold);
    let not_low = _mm256_cmpeq_epi8(low, zero);
    if THREE {
        let high = _mm256_subs_epu8(_mm256_subs_epu8(neighbour, center), threshold);
        let not_high = _mm256_cmpeq_epi8(high, zero);
        _mm256_sub_epi8(_mm256_add_epi8(_mm256_set1_epi8(1), not_high), not_low)
    } else {
        _mm256_sub_epi8(zero, not_low)
    }
}

#[target_feature(enable = "avx2")]
fn add_offsets<const BANDS: bool>(
    index: __m256i,
    tables: [__m256i; 4],
    offsets: __m256i,
    pixels: __m256i,
) -> __m256i {
    let byte_index = _mm256_and_si256(_mm256_srli_epi16::<1>(index), _mm256_set1_epi8(15));
    let bank_bit = _mm256_add_epi8(index, index);
    let high_bank_bit = _mm256_add_epi8(bank_bit, bank_bit);
    let table_entry = |bank: usize| _mm256_shuffle_epi8(tables[bank], byte_index);
    let low = table_entry(0);
    let high = table_entry(1);
    let packed = if BANDS {
        _mm256_blendv_epi8(low, high, high_bank_bit)
    } else {
        let low = _mm256_blendv_epi8(low, table_entry(2), bank_bit);
        let high = _mm256_blendv_epi8(high, table_entry(3), bank_bit);
        _mm256_blendv_epi8(low, high, high_bank_bit)
    };
    let odd = _mm256_slli_epi16::<7>(index);
    let code = _mm256_blendv_epi8(packed, _mm256_srli_epi16::<4>(packed), odd);
    let correction = _mm256_shuffle_epi8(offsets, _mm256_and_si256(code, _mm256_set1_epi8(15)));
    let plus = _mm256_adds_epu8(pixels, correction);
    let minus = _mm256_subs_epu8(pixels, _mm256_abs_epi8(correction));
    _mm256_blendv_epi8(plus, minus, correction)
}

#[target_feature(enable = "avx2")]
unsafe fn span<const CHROMA: bool, const LEVELS: u8>(
    filter: &CcsoFilter,
    dst: &mut [u8],
    source: [&[u8]; 3],
    threshold: u8,
) {
    const LANES: usize = 32;
    let [center, first, second] = source.map(<[u8]>::as_ptr);
    let pitch = if CHROMA { LANES * 2 } else { LANES };
    let threshold = _mm256_set1_epi8(threshold as i8);
    let band_shift = _mm_cvtsi32_si128(i32::from(filter.band_shift));
    let band_mask = _mm256_set1_epi8((255_u16 >> filter.band_shift) as i8);

    // SAFETY: each table contains a full 16-byte lane. The last destination
    // chunk may be shorter and is left for the scalar tail.
    unsafe {
        let table = filter.codes.as_ptr();
        let tables = core::array::from_fn(|i| {
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.add(i * 16).cast()))
        });
        let offsets = _mm256_broadcastsi128_si256(_mm_loadu_si128(filter.offsets.as_ptr().cast()));
        for (chunk, dst) in dst.as_chunks_mut::<LANES>().0.iter_mut().enumerate() {
            let at = chunk * pitch;
            let center = load_luma::<CHROMA>(center.add(at));
            let bands = _mm256_and_si256(_mm256_srl_epi16(center, band_shift), band_mask);
            let index = if LEVELS == 1 {
                bands
            } else {
                let first = load_luma::<CHROMA>(first.add(at));
                let second = load_luma::<CHROMA>(second.add(at));
                let (a, b) = if LEVELS == 2 {
                    (
                        classify::<false>(center, first, threshold),
                        classify::<false>(center, second, threshold),
                    )
                } else {
                    (
                        classify::<true>(center, first, threshold),
                        classify::<true>(center, second, threshold),
                    )
                };
                _mm256_or_si256(
                    _mm256_slli_epi16::<4>(bands),
                    _mm256_or_si256(_mm256_slli_epi16::<2>(a), b),
                )
            };
            // The three deinterleaved loads share the packed qword order.
            // Classify in that order and permute only the final indices.
            let index = if CHROMA {
                _mm256_permute4x64_epi64::<0xd8>(index)
            } else {
                index
            };
            let pixels = _mm256_loadu_si256(dst.as_ptr().cast());
            let result = if LEVELS == 1 {
                add_offsets::<true>(index, tables, offsets, pixels)
            } else {
                add_offsets::<false>(index, tables, offsets, pixels)
            };
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), result);
        }
    }
}

/// # Safety
/// AVX2 must be enabled and the source slices must contain the destination's
/// complete luma footprint.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn apply(
    filter: &CcsoFilter,
    dst: &mut [u8],
    source: [&[u8]; 3],
    chroma: bool,
) -> usize {
    // SAFETY: the full SIMD chunks stay within the validated spans.
    unsafe {
        match (chroma, filter.mode) {
            (false, CcsoMode::Bands) => span::<false, 1>(filter, dst, source, 0),
            (true, CcsoMode::Bands) => span::<true, 1>(filter, dst, source, 0),
            (false, CcsoMode::Two(t)) => span::<false, 2>(filter, dst, source, t),
            (true, CcsoMode::Two(t)) => span::<true, 2>(filter, dst, source, t),
            (false, CcsoMode::Three(t)) => span::<false, 3>(filter, dst, source, t),
            (true, CcsoMode::Three(t)) => span::<true, 3>(filter, dst, source, t),
        }
    }
    dst.len() & !31
}
