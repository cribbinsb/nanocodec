//! Two-row 4/8-wide horizontal passes from dav1d 1.5.4
//! `mc16_avx512.asm` (BSD-2-Clause).

use super::{A, AV1_FILTER_BEFORE, AV1_FILTER_TAPS, B, BitDepth, FILTER_BITS, av1_compound_bias};
use super::{dot_word, inter_round0, tap_pairs};
use core::arch::x86_64::{
    _mm_loadu_si128, _mm_storeu_si128, _mm256_castsi256_si128, _mm256_extracti128_si256,
    _mm256_loadu_si256, _mm256_min_epu16, _mm256_packus_epi32, _mm256_set1_epi16,
    _mm256_storeu_si256, _mm512_castsi128_si512, _mm512_castsi256_si512, _mm512_castsi512_si256,
    _mm512_cvtsepi32_epi16, _mm512_extracti64x4_epi64, _mm512_inserti32x4, _mm512_inserti64x4,
    _mm512_loadu_si512, _mm512_permutexvar_epi8, _mm512_set1_epi32, _mm512_shuffle_epi8,
    _mm512_srai_epi32,
};

/// Each pair of output halves fits the 256-bit pack's lane stripes.
const PAIR_SHUFFLES: [[u8; 64]; 4] = {
    let mut masks = [[0; 64]; 4];
    let mut tap = 0;
    while tap < 4 {
        let mut i = 0;
        while i < 64 {
            let lane = i / 4;
            let row = (lane / 4) % 2;
            let x = 4 * (lane / 8) + lane % 4;
            masks[tap][i] = (32 * row + 2 * (x + 2 * tap) + i % 4) as u8;
            i += 1;
        }
        tap += 1;
    }
    masks
};

/// # Safety
/// The selected CPU supports ICL, and the source/destination windows cover
/// all output rows and the adjacent source padding.
#[inline]
pub(super) unsafe fn try_filter<const W: usize, const PUT: bool>(
    src: (*const u16, usize),
    dst: (*mut u16, usize),
    height: usize,
    taps: &[i16; AV1_FILTER_TAPS],
    depth: BitDepth,
) -> bool {
    if !matches!(depth.bits(), 10 | 12) {
        return false;
    }
    let inter_shift = inter_round0(depth);
    let rounding = 1 << (inter_shift - 1);
    // SAFETY: the dispatcher checked the padded views and this CPU's ISA.
    unsafe {
        if W == 4 && !PUT && dst.1 == 4 && height.is_multiple_of(4) {
            let bias = rounding - (av1_compound_bias(depth) << inter_shift);
            if inter_shift == 3 {
                prep4::<3>(src, dst.0, height, taps, bias);
            } else {
                prep4::<5>(src, dst.0, height, taps, bias);
            }
        } else if W == 8 && PUT && height.is_multiple_of(2) {
            put8(
                src,
                dst,
                height,
                taps,
                depth,
                rounding + (1 << (FILTER_BITS - 1)),
            );
        } else {
            return false;
        }
    }
    true
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline(never)]
unsafe fn prep4<const SHIFT: u32>(
    (src, ss): (*const u16, usize),
    dst: *mut u16,
    height: usize,
    taps: &[i16; AV1_FILTER_TAPS],
    round: i32,
) {
    // SAFETY: each of the four padded source rows covers eight words.
    // The caller checked a multiple of four rows and a packed output.
    unsafe {
        let src = src.add(AV1_FILTER_BEFORE * ss + 2);
        let [a, b] =
            tap_pairs::<2>(taps).map(|(x, y)| _mm512_set1_epi32(crate::simd::packed_pair(x, y)));
        let masks = (
            _mm512_loadu_si512(A.as_ptr().cast()),
            _mm512_loadu_si512(B.as_ptr().cast()),
        );
        let bias = _mm512_set1_epi32(round);
        for y in (0..height).step_by(4) {
            let row = src.add(y * ss);
            let first = _mm512_castsi128_si512(_mm_loadu_si128(row.cast()));
            let second = _mm512_inserti32x4::<1>(first, _mm_loadu_si128(row.add(ss).cast()));
            let third = _mm512_inserti32x4::<2>(second, _mm_loadu_si128(row.add(2 * ss).cast()));
            let raw = _mm512_inserti32x4::<3>(third, _mm_loadu_si128(row.add(3 * ss).cast()));
            let sum = dot_word(bias, _mm512_shuffle_epi8(raw, masks.0), a);
            let sum = dot_word(sum, _mm512_shuffle_epi8(raw, masks.1), b);
            let words = _mm512_cvtsepi32_epi16(_mm512_srai_epi32::<SHIFT>(sum));
            _mm256_storeu_si256(dst.add(y * 4).cast(), words);
        }
    }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
unsafe fn put8(
    (src, ss): (*const u16, usize),
    (dst, ds): (*mut u16, usize),
    height: usize,
    taps: &[i16; AV1_FILTER_TAPS],
    depth: BitDepth,
    round: i32,
) {
    let packed = tap_pairs::<4>(taps).map(|(a, b)| crate::simd::packed_pair(a, b));
    // SAFETY: each vector covers the eight-tap source window. Its last
    // unused word is covered by the checked padding below the block.
    unsafe {
        let src = src.add(AV1_FILTER_BEFORE * ss);
        let masks = core::array::from_fn::<_, 4, _>(|i| {
            _mm512_loadu_si512(PAIR_SHUFFLES[i].as_ptr().cast())
        });
        let weights = packed.map(|pair| _mm512_set1_epi32(pair));
        let bias = _mm512_set1_epi32(round);
        let clip = _mm256_set1_epi16(depth.max() as i16);
        for y in (0..height).step_by(2) {
            let row = src.add(y * ss);
            let raw = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_loadu_si256(row.cast())),
                _mm256_loadu_si256(row.add(ss).cast()),
            );
            let mut sum = bias;
            for tap in 0..4 {
                let samples = _mm512_permutexvar_epi8(masks[tap], raw);
                sum = dot_word(sum, samples, weights[tap]);
            }
            let sum = _mm512_srai_epi32::<FILTER_BITS>(sum);
            let a = _mm512_castsi512_si256(sum);
            let b = _mm512_extracti64x4_epi64::<1>(sum);
            let out = _mm256_min_epu16(_mm256_packus_epi32(a, b), clip);
            if ds == 8 {
                _mm256_storeu_si256(dst.add(y * ds).cast(), out);
            } else {
                _mm_storeu_si128(dst.add(y * ds).cast(), _mm256_castsi256_si128(out));
                _mm_storeu_si128(
                    dst.add((y + 1) * ds).cast(),
                    _mm256_extracti128_si256::<1>(out),
                );
            }
        }
    }
}
