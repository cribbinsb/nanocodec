//! Ice Lake vertical 16bpc put/prep after dav1d 1.5.4 `mc16_avx512.asm`
//! (BSD-2-Clause). The six- and eight-tap inner loops retain interleaved
//! pairs and finish two output rows from only two new source rows.

use super::{
    AV1_FILTER_BEFORE, AV1_FILTER_TAPS, BitDepth, FILTER_BITS, av1_compound_bias, inter_round0,
    tap_pair_count,
};
use core::arch::x86_64::{
    __m512i, _mm256_loadu_si256, _mm256_storeu_si256, _mm512_castsi256_si512,
    _mm512_castsi512_si256, _mm512_dpwssd_epi32, _mm512_extracti64x4_epi64, _mm512_inserti64x4,
    _mm512_loadu_si512, _mm512_min_epu16, _mm512_packs_epi32, _mm512_packus_epi32,
    _mm512_permutexvar_epi8, _mm512_permutexvar_epi64, _mm512_set_epi64, _mm512_set1_epi16,
    _mm512_set1_epi32, _mm512_shrdi_epi32, _mm512_srai_epi32, _mm512_storeu_si512,
    _mm512_unpackhi_epi16, _mm512_unpacklo_epi16,
};

/// The caller checked the source and destination rectangles and the
/// vertical-only phases, and selected the complete Ice Lake tier.
#[inline(always)]
pub(super) unsafe fn try_filter<const W: usize, const PUT: bool>(
    (src, ss): (*const u16, usize),
    (dst, ds): (*mut u16, usize),
    h: usize,
    fy: &[i16; AV1_FILTER_TAPS],
    depth: BitDepth,
) -> bool {
    if !matches!(W, 16 | 32 | 64) || !h.is_multiple_of(2) {
        return false;
    }
    let pairs = tap_pair_count(fy);
    if pairs < 3 {
        return false;
    }
    let r0 = inter_round0(depth);
    macro_rules! filter_rows {
        ($pairs:literal, $put:literal, $shift:expr, $round:expr) => {
            if W == 16 {
                rows16::<$pairs, $put, $shift>((src, ss), (dst, ds), h, fy, $round, depth);
            } else {
                rows::<$pairs, $put, $shift>((src, ss), (dst, ds), (W, h), fy, $round, depth);
            }
        };
    }
    // SAFETY: the slices cover the validated window and destination,
    // the entry enables ICL, and h is even.
    unsafe {
        if PUT {
            let shift = FILTER_BITS;
            let round = 1 << (shift - 1);
            if pairs == 3 {
                filter_rows!(3, true, FILTER_BITS, round);
            } else {
                filter_rows!(4, true, FILTER_BITS, round);
            }
        } else {
            let round = (1 << (r0 - 1)) - (av1_compound_bias(depth) << r0);
            match (pairs, r0) {
                (3, 5) => filter_rows!(3, false, 5, round),
                (4, 5) => filter_rows!(4, false, 5, round),
                (3, _) => filter_rows!(3, false, 3, round),
                _ => filter_rows!(4, false, 3, round),
            }
        }
    }
    true
}

#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn rows<const PAIRS: usize, const PUT: bool, const SHIFT: u32>(
    (src, ss): (*const u16, usize),
    (dst, ds): (*mut u16, usize),
    (width, h): (usize, usize),
    taps: &[i16; AV1_FILTER_TAPS],
    round: i32,
    depth: BitDepth,
) {
    let start = AV1_FILTER_TAPS / 2 - PAIRS;
    // SAFETY: ICL is enabled and the caller validated complete rows.
    unsafe {
        let round = _mm512_set1_epi32(round);
        let max = _mm512_set1_epi16(depth.max() as i16);
        let coeffs = super::tap_pairs::<PAIRS>(taps)
            .map(|(a, b)| _mm512_set1_epi32(crate::simd::packed_pair(a, b)));
        let zip = |a, b| [_mm512_unpacklo_epi16(a, b), _mm512_unpackhi_epi16(a, b)];
        for x in (0..width).step_by(32) {
            let src = src.add(start * ss + AV1_FILTER_BEFORE + x);
            let dst = dst.add(x);
            let load = |r: usize| _mm512_loadu_si512(src.add(r * ss).cast());
            let mut tail = load(0);
            let mut even = [[round; 2]; PAIRS];
            let mut odd = [[round; 2]; PAIRS];
            for k in 0..PAIRS - 1 {
                let next = load(2 * k + 1);
                even[k] = zip(tail, next);
                tail = load(2 * k + 2);
                odd[k] = zip(next, tail);
            }
            for r in (0..h).step_by(2) {
                let next = load(r + 2 * PAIRS - 1);
                even[PAIRS - 1] = zip(tail, next);
                tail = load(r + 2 * PAIRS);
                odd[PAIRS - 1] = zip(next, tail);
                let accumulate = |window: &[[__m512i; 2]; PAIRS]| {
                    let mut low = round;
                    let mut high = round;
                    for k in 0..PAIRS {
                        low = _mm512_dpwssd_epi32(low, window[k][0], coeffs[k]);
                        high = _mm512_dpwssd_epi32(high, window[k][1], coeffs[k]);
                    }
                    let (low, high) = (
                        _mm512_srai_epi32::<SHIFT>(low),
                        _mm512_srai_epi32::<SHIFT>(high),
                    );
                    if PUT {
                        _mm512_min_epu16(_mm512_packus_epi32(low, high), max)
                    } else {
                        _mm512_packs_epi32(low, high)
                    }
                };
                _mm512_storeu_si512(dst.add(r * ds).cast(), accumulate(&even));
                _mm512_storeu_si512(dst.add((r + 1) * ds).cast(), accumulate(&odd));
                for k in 0..PAIRS - 1 {
                    even[k] = even[k + 1];
                    odd[k] = odd[k + 1];
                }
            }
        }
    }
}

// LLVM can separate the intrinsic into multiply-add and add instructions.
// Keep the independent horizontal and vertical chains in word-dot form.
#[inline]
#[target_feature(enable = "avx512f,avx512vnni")]
pub(super) unsafe fn dot_word(mut sum: __m512i, pixels: __m512i, taps: __m512i) -> __m512i {
    // SAFETY: the target feature enables the register class and instruction.
    unsafe {
        core::arch::asm!(
            "vpdpwssd {sum}, {pixels}, {taps}",
            sum = inout(zmm_reg) sum,
            pixels = in(zmm_reg) pixels,
            taps = in(zmm_reg) taps,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    sum
}

pub(super) const ZIP16: [u8; 64] = {
    let mut values = [0; 64];
    let mut index = 0;
    while index < 64 {
        values[index] = (index / 4 * 2 + index % 2 + 32 * (index % 4 / 2)) as u8;
        index += 1;
    }
    values
};

/// Two sixteen-word rows fit in one dot-product register. The odd pairs
/// are loaded together; a one-word double shift builds the even pairs.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni")]
unsafe fn rows16<const PAIRS: usize, const PUT: bool, const SHIFT: u32>(
    (src, ss): (*const u16, usize),
    (dst, ds): (*mut u16, usize),
    h: usize,
    taps: &[i16; AV1_FILTER_TAPS],
    round: i32,
    depth: BitDepth,
) {
    // SAFETY: ICL is enabled and the caller validated complete rows.
    unsafe {
        let src = src.add((AV1_FILTER_TAPS / 2 - PAIRS) * ss + AV1_FILTER_BEFORE);
        let zip = _mm512_loadu_si512(ZIP16.as_ptr().cast());
        let unstripe = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
        let load = |r: usize| _mm256_loadu_si256(src.add(r * ss).cast());
        let pair = |r| {
            let raw = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(load(r)), load(r + 1));
            _mm512_permutexvar_epi8(zip, raw)
        };
        let round = _mm512_set1_epi32(round);
        let max = _mm512_set1_epi16(depth.max() as i16);
        let coeffs = super::tap_pairs::<PAIRS>(taps)
            .map(|(a, b)| _mm512_set1_epi32(crate::simd::packed_pair(a, b)));
        let mut even = pair(0);
        let mut odd = [round; PAIRS];
        for (k, window) in odd[..PAIRS - 1].iter_mut().enumerate() {
            *window = pair(2 * k + 1);
        }
        for r in (0..h).step_by(2) {
            odd[PAIRS - 1] = pair(r + 2 * PAIRS - 1);
            let mut a = dot_word(round, even, coeffs[0]);
            let mut b = dot_word(round, odd[0], coeffs[0]);
            even = _mm512_shrdi_epi32::<16>(odd[0], odd[1]);
            for k in 1..PAIRS {
                let e = _mm512_shrdi_epi32::<16>(odd[k - 1], odd[k]);
                a = dot_word(a, e, coeffs[k]);
                b = dot_word(b, odd[k], coeffs[k]);
                odd[k - 1] = odd[k];
            }
            let a = _mm512_srai_epi32::<SHIFT>(a);
            let b = _mm512_srai_epi32::<SHIFT>(b);
            let packed = if PUT {
                _mm512_min_epu16(_mm512_packus_epi32(a, b), max)
            } else {
                _mm512_packs_epi32(a, b)
            };
            let words = _mm512_permutexvar_epi64(unstripe, packed);
            _mm256_storeu_si256(dst.add(r * ds).cast(), _mm512_castsi512_si256(words));
            _mm256_storeu_si256(
                dst.add((r + 1) * ds).cast(),
                _mm512_extracti64x4_epi64::<1>(words),
            );
        }
    }
}
