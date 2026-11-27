//! Packed byte-table statistics from the dav1d 1.5.4 Ice Lake stripe
//! kernel (`src/x86/looprestoration_avx512.asm`, BSD-2-Clause).

use super::{
    B_ROUND, BLEND_SHIFT, Kept, MTABLE_BITS, ONE_ROW_SHIFT, STEP, Single, Statistics, Sums,
    TWO_ROW_SHIFT, X_BITS,
};
use crate::av1_restoration::self_guided::x_by_xplus1;
use core::arch::x86_64::{
    __m512i, _mm_loadu_si128, _mm_mask_storeu_epi8, _mm_storeu_si128, _mm512_add_epi32,
    _mm512_and_si512, _mm512_broadcast_i64x4, _mm512_castsi256_si512, _mm512_cvtepu8_epi32,
    _mm512_cvtepu16_epi32, _mm512_cvtusepi32_epi8, _mm512_inserti64x4, _mm512_loadu_si512,
    _mm512_madd_epi16, _mm512_mask_blend_epi16, _mm512_mask_blend_epi32, _mm512_max_epi32,
    _mm512_min_epu32, _mm512_mullo_epi32, _mm512_or_si512, _mm512_permutex2var_epi8,
    _mm512_set1_epi32, _mm512_setzero_si512, _mm512_shuffle_i32x4, _mm512_slli_epi32,
    _mm512_srai_epi32, _mm512_srli_epi32, _mm512_storeu_si512, _mm512_sub_epi32,
    _mm512_test_epi32_mask,
};

const FACTORS: [u8; 256] = {
    let mut table = [0; 256];
    let mut z = 0;
    while z < table.len() {
        table[z] = (256 - x_by_xplus1(z as u32)) as u8;
        z += 1;
    }
    table
};

/// Exact `256 - A` for sixteen clipped 8-bit indices in dwords.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
pub(in crate::av1_restoration) fn lookup_complement(z: __m512i) -> __m512i {
    // SAFETY: the table has four complete 64-byte chunks.
    unsafe {
        let table = FACTORS.as_ptr();
        let lo0 = _mm512_loadu_si512(table.cast());
        let lo1 = _mm512_loadu_si512(table.add(64).cast());
        let hi0 = _mm512_loadu_si512(table.add(128).cast());
        let hi1 = _mm512_loadu_si512(table.add(192).cast());
        let lo = _mm512_permutex2var_epi8(lo0, z, lo1);
        let hi = _mm512_permutex2var_epi8(hi0, z, hi1);
        let top = _mm512_test_epi32_mask(z, _mm512_set1_epi32(128));
        _mm512_and_si512(_mm512_mask_blend_epi32(top, lo, hi), _mm512_set1_epi32(255))
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
pub(super) unsafe fn store_packed(stats: Statistics, sums: Sums, dst: *mut i32) {
    let sum = _mm512_cvtepu16_epi32(sums.sum);
    let square = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(sums.squares[0]), sums.squares[1]);
    // AVX2 makes the squares in separate low/high quarters per 128-bit
    // lane. Join the quarters in pixel order for the one-vector store.
    let square = _mm512_shuffle_i32x4::<0xd8>(square, square);
    let scale = _mm512_broadcast_i64x4(stats.scale);
    let reciprocal = _mm512_broadcast_i64x4(stats.reciprocal);
    let p = _mm512_sub_epi32(
        _mm512_mullo_epi32(square, _mm512_set1_epi32(stats.count)),
        _mm512_madd_epi16(sum, sum),
    );
    let z = _mm512_srli_epi32::<MTABLE_BITS>(_mm512_add_epi32(
        _mm512_mullo_epi32(p, scale),
        _mm512_set1_epi32(1 << (MTABLE_BITS - 1)),
    ));
    let z = _mm512_min_epu32(z, _mm512_set1_epi32(255));
    // SAFETY: the caller reserves sixteen consecutive columns in the
    // packed statistics row.
    unsafe {
        let x = lookup_complement(z);
        let b = _mm512_add_epi32(
            _mm512_mullo_epi32(_mm512_madd_epi16(sum, reciprocal), x),
            _mm512_set1_epi32(B_ROUND),
        );
        let packed = _mm512_or_si512(_mm512_and_si512(b, _mm512_set1_epi32(-1 << X_BITS)), x);
        _mm512_storeu_si512(dst.cast(), packed);
    }
}

// Projection from the packed horizontal neighborhoods. Each full vector
// covers sixteen consecutive output columns and the same kept rows as AVX2.

#[target_feature(enable = "avx512f")]
#[inline]
fn unpack(values: __m512i) -> (__m512i, __m512i) {
    (
        _mm512_and_si512(values, _mm512_set1_epi32((1 << X_BITS) - 1)),
        _mm512_srli_epi32::<{ X_BITS as u32 }>(values),
    )
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn neighbours(packed: *const i32, col: usize) -> [__m512i; 3] {
    // SAFETY: a stripe row reserves the sixteen columns and their halo.
    unsafe {
        [
            _mm512_loadu_si512(packed.add(col).cast()),
            _mm512_loadu_si512(packed.add(col + 1).cast()),
            _mm512_loadu_si512(packed.add(col + 2).cast()),
        ]
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn radius1(packed: *const i32, (ab222, ab343): Kept, col: usize) -> (__m512i, __m512i) {
    // SAFETY: the stripe layout reserves space for every 16-column step.
    unsafe {
        let [left, centre, right] = neighbours(packed, col);
        let outer = _mm512_add_epi32(left, right);
        let sum = _mm512_add_epi32(outer, centre);
        let new222 = _mm512_add_epi32(sum, sum);
        let new343 = _mm512_sub_epi32(_mm512_add_epi32(new222, new222), outer);
        let p222 = ab222.add(col).cast();
        let p343 = ab343.add(col).cast();
        let old222 = _mm512_loadu_si512(p222);
        let above = unpack(_mm512_add_epi32(old222, _mm512_loadu_si512(p343)));
        let below = unpack(_mm512_add_epi32(old222, new343));
        _mm512_storeu_si512(p222, new222);
        _mm512_storeu_si512(p343, new343);
        (
            _mm512_add_epi32(above.0, below.0),
            _mm512_add_epi32(above.1, below.1),
        )
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn radius2<const BETWEEN: bool>(
    packed: *const i32,
    (x565, b565): Kept,
    col: usize,
) -> (__m512i, __m512i) {
    // SAFETY: the stripe layout reserves space for every 16-column step.
    unsafe {
        let px = x565.add(col).cast();
        let pb = b565.add(col).cast();
        let kept = (_mm512_loadu_si512(px), _mm512_loadu_si512(pb));
        if !BETWEEN {
            return kept;
        }
        let [left, centre, right] = neighbours(packed, col);
        let sum = _mm512_add_epi32(_mm512_add_epi32(left, right), centre);
        let weighed = _mm512_add_epi32(_mm512_add_epi32(_mm512_slli_epi32::<2>(sum), sum), centre);
        let (x, b) = unpack(weighed);
        _mm512_storeu_si512(px, x);
        _mm512_storeu_si512(pb, b);
        (_mm512_add_epi32(x, kept.0), _mm512_add_epi32(b, kept.1))
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn write_bytes(dst: &mut [u8], col: usize, values: __m512i) {
    let unclipped = _mm512_max_epi32(values, _mm512_setzero_si512());
    let bytes = _mm512_cvtusepi32_epi8(unclipped);
    let len = dst.len() - col;
    // SAFETY: both forms write only the bytes inside the output row.
    unsafe {
        let ptr = dst.as_mut_ptr().add(col);
        if len >= STEP {
            _mm_storeu_si128(ptr.cast(), bytes);
        } else {
            _mm_mask_storeu_epi8(ptr.cast(), (1 << len) - 1, bytes);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
pub(super) unsafe fn mix_row<const BETWEEN: bool>(
    src: &[u8],
    dst: &mut [u8],
    (packed5, five): (*const i32, Kept),
    (packed3, three): (*const i32, Kept),
    [w5, w3]: [i32; 2],
) {
    assert!(src.len() >= dst.len().next_multiple_of(STEP));
    let weights = _mm512_set1_epi32(((w3 << 2) << 16) | ((w5 << 2) & 0xffff));
    let rounding = _mm512_set1_epi32(1 << (BLEND_SHIFT + 1));
    // SAFETY: the input check and stripe layout cover every full load.
    unsafe {
        for col in (0..dst.len()).step_by(STEP) {
            let pixels = _mm512_cvtepu8_epi32(_mm_loadu_si128(src.as_ptr().add(col).cast()));
            let (x5, b5) = radius2::<BETWEEN>(packed5, five, col);
            let (x3, b3) = radius1(packed3, three, col);
            let d5 = _mm512_sub_epi32(b5, _mm512_madd_epi16(x5, pixels));
            let d5 = if BETWEEN {
                _mm512_srli_epi32::<{ TWO_ROW_SHIFT as u32 }>(d5)
            } else {
                _mm512_srli_epi32::<{ ONE_ROW_SHIFT as u32 }>(d5)
            };
            let d3 = _mm512_slli_epi32::<{ (16 - TWO_ROW_SHIFT) as u32 }>(_mm512_sub_epi32(
                b3,
                _mm512_madd_epi16(x3, pixels),
            ));
            let pair = _mm512_mask_blend_epi16(0xaaaa_aaaa, d5, d3);
            let delta = _mm512_srai_epi32::<{ (BLEND_SHIFT + 2) as u32 }>(_mm512_add_epi32(
                _mm512_madd_epi16(pair, weights),
                rounding,
            ));
            write_bytes(dst, col, _mm512_add_epi32(pixels, delta));
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
pub(super) unsafe fn single_row(
    src: &[u8],
    dst: &mut [u8],
    weight: i32,
    (kind, packed, kept): (Single, *const i32, Kept),
) {
    assert!(src.len() >= dst.len().next_multiple_of(STEP));
    let weight = _mm512_set1_epi32(weight);
    let round = _mm512_set1_epi32(1 << (BLEND_SHIFT - 1));
    // SAFETY: the input check and stripe layout cover every full load.
    unsafe {
        for col in (0..dst.len()).step_by(STEP) {
            let pixels = _mm512_cvtepu8_epi32(_mm_loadu_si128(src.as_ptr().add(col).cast()));
            let (x, b) = match kind {
                Single::FiveBetween => radius2::<true>(packed, kept, col),
                Single::FiveOn => radius2::<false>(packed, kept, col),
                Single::Three => radius1(packed, kept, col),
            };
            let delta = _mm512_sub_epi32(b, _mm512_madd_epi16(x, pixels));
            let delta = if kind == Single::FiveOn {
                _mm512_srai_epi32::<{ ONE_ROW_SHIFT as u32 }>(delta)
            } else {
                _mm512_srai_epi32::<{ TWO_ROW_SHIFT as u32 }>(delta)
            };
            let proj = _mm512_srai_epi32::<{ BLEND_SHIFT as u32 }>(_mm512_add_epi32(
                round,
                _mm512_mullo_epi32(delta, weight),
            ));
            write_bytes(dst, col, _mm512_add_epi32(pixels, proj));
        }
    }
}
