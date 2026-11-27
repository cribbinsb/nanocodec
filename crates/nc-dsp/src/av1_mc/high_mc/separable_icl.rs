//! 16bpc AV1 two-pass filters using dav1d's Ice Lake `.hv_w32` stripe
//! schedule (BSD-2-Clause). Keep interleaved vertical windows in registers
//! while computing the next two horizontal rows.

use core::arch::x86_64::{
    __m512i, _mm256_storeu_si256, _mm512_castsi512_si256, _mm512_extracti64x4_epi64,
    _mm512_loadu_si512, _mm512_min_epu16, _mm512_packs_epi32, _mm512_packus_epi32,
    _mm512_permutexvar_epi8, _mm512_permutexvar_epi64, _mm512_set_epi64, _mm512_set1_epi16,
    _mm512_set1_epi32, _mm512_shrdi_epi32, _mm512_srai_epi32, _mm512_storeu_si512,
    _mm512_unpackhi_epi16, _mm512_unpacklo_epi16,
};

use super::{
    AV1_FILTER_TAPS, Av1Taps, BitDepth, INTER_ROUND1_COMPOUND, av1_compound_bias,
    horizontal_icl::{A, B, row_sums},
    tap_pair_count, tap_pairs,
    vertical_icl::{ZIP16, dot_word},
};

/// # Safety
/// The caller checked the eight-tap source and destination blocks and
/// selected the complete Ice Lake feature set.
#[inline(always)]
pub(super) unsafe fn try_filter<const W: usize, const PUT: bool>(
    src: (*const u16, usize),
    dst: (*mut u16, usize),
    height: usize,
    (fx, fy): Av1Taps<'_>,
    depth: BitDepth,
) -> bool {
    if !matches!(W, 16 | 32 | 64) || !height.is_multiple_of(2) || !matches!(depth.bits(), 10 | 12) {
        return false;
    }
    let (hp, vp) = (tap_pair_count(fx), tap_pair_count(fy));
    macro_rules! run {
        ($hp:literal, $vp:literal, $put:literal, $h:literal, $v:literal, $r:expr) => {
            if W == 16 {
                rows16::<$hp, $vp, $put, $h, $v>(src, dst, height, (fx, fy), depth, $r);
            } else {
                rows::<$hp, $vp, $put, $h, $v>(src, dst, (W, height), (fx, fy), depth, $r);
            }
        };
    }
    macro_rules! filter_rows {
        ($hp:literal, $vp:literal) => {
            if PUT {
                if depth.bits() == 12 {
                    run!($hp, $vp, true, 5, 9, 1 << 8);
                } else {
                    run!($hp, $vp, true, 3, 11, 1 << 10);
                }
            } else {
                let round = (1 << (INTER_ROUND1_COMPOUND - 1))
                    - (av1_compound_bias(depth) << INTER_ROUND1_COMPOUND);
                if depth.bits() == 12 {
                    run!($hp, $vp, false, 5, 7, round);
                } else {
                    run!($hp, $vp, false, 3, 7, round);
                }
            }
        };
    }
    // SAFETY: the entry checked geometry, depth and ISA.
    unsafe {
        match (hp, vp) {
            (3, 3) => filter_rows!(3, 3),
            (3, 4) => filter_rows!(3, 4),
            (4, 3) => filter_rows!(4, 3),
            (4, 4) => filter_rows!(4, 4),
            _ => return false,
        }
    }
    true
}

#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn rows<const HP: usize, const VP: usize, const PUT: bool, const H: u32, const V: u32>(
    (src, ss): (*const u16, usize),
    (dst, ds): (*mut u16, usize),
    (width, height): (usize, usize),
    (fx, fy): Av1Taps<'_>,
    depth: BitDepth,
    round: i32,
) {
    let start_x = AV1_FILTER_TAPS / 2 - HP;
    let start_y = AV1_FILTER_TAPS / 2 - VP;
    // SAFETY: every stripe loads 32 horizontal samples plus the taps;
    // exactly two horizontal rows enter the sliding vertical window at
    // a time, through the last row of the checked source.
    unsafe {
        let h_round = _mm512_set1_epi32(1 << (H - 1));
        let v_round = _mm512_set1_epi32(round);
        let max = _mm512_set1_epi16(depth.max() as i16);
        let pair = |(a, b)| _mm512_set1_epi32(crate::simd::packed_pair(a, b));
        let h_coeff = tap_pairs::<HP>(fx).map(pair);
        let v_coeff = tap_pairs::<VP>(fy).map(pair);
        let mask = (
            _mm512_loadu_si512(A.as_ptr().cast()),
            _mm512_loadu_si512(B.as_ptr().cast()),
        );
        let zip = |a, b| [_mm512_unpacklo_epi16(a, b), _mm512_unpackhi_epi16(a, b)];
        for x in (0..width).step_by(32) {
            let source = src.add(start_y * ss + start_x + x);
            let horizontal = |r| {
                let (lo, hi) =
                    row_sums::<HP, false, false>(source.add(r * ss), ss, h_coeff, mask, h_round);
                _mm512_packs_epi32(_mm512_srai_epi32::<H>(lo), _mm512_srai_epi32::<H>(hi))
            };
            let mut tail = horizontal(0);
            let mut even = [[v_round; 2]; VP];
            let mut odd = [[v_round; 2]; VP];
            for k in 0..VP - 1 {
                let next = horizontal(2 * k + 1);
                even[k] = zip(tail, next);
                tail = horizontal(2 * k + 2);
                odd[k] = zip(next, tail);
            }
            let accumulate = |window: &[[__m512i; 2]; VP]| {
                let mut a = v_round;
                let mut b = v_round;
                for k in 0..VP {
                    a = dot_word(a, window[k][0], v_coeff[k]);
                    b = dot_word(b, window[k][1], v_coeff[k]);
                }
                let a = _mm512_srai_epi32::<V>(a);
                let b = _mm512_srai_epi32::<V>(b);
                if PUT {
                    _mm512_min_epu16(_mm512_packus_epi32(a, b), max)
                } else {
                    _mm512_packs_epi32(a, b)
                }
            };
            for y in (0..height).step_by(2) {
                let next = horizontal(y + 2 * VP - 1);
                even[VP - 1] = zip(tail, next);
                tail = horizontal(y + 2 * VP);
                odd[VP - 1] = zip(next, tail);
                _mm512_storeu_si512(dst.add(y * ds + x).cast(), accumulate(&even));
                _mm512_storeu_si512(dst.add((y + 1) * ds + x).cast(), accumulate(&odd));
                for k in 0..VP - 1 {
                    even[k] = even[k + 1];
                    odd[k] = odd[k + 1];
                }
            }
        }
    }
}

/// One register holds two sixteen-wide horizontal rows. Retain the odd
/// pairs, then produce the even pairs with the same word shift as dav1d.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni")]
unsafe fn rows16<const HP: usize, const VP: usize, const PUT: bool, const H: u32, const V: u32>(
    (src, ss): (*const u16, usize),
    (dst, ds): (*mut u16, usize),
    height: usize,
    (fx, fy): Av1Taps<'_>,
    depth: BitDepth,
    round: i32,
) {
    let start_x = AV1_FILTER_TAPS / 2 - HP;
    let start_y = AV1_FILTER_TAPS / 2 - VP;
    // SAFETY: checked source/destination rectangles, even height and ICL.
    unsafe {
        let source = src.add(start_y * ss + start_x);
        let h_round = _mm512_set1_epi32(1 << (H - 1));
        let v_round = _mm512_set1_epi32(round);
        let max = _mm512_set1_epi16(depth.max() as i16);
        let coeff = |(a, b)| _mm512_set1_epi32(crate::simd::packed_pair(a, b));
        let h_coeff = tap_pairs::<HP>(fx).map(coeff);
        let v_coeff = tap_pairs::<VP>(fy).map(coeff);
        let shuffle = (
            _mm512_loadu_si512(A.as_ptr().cast()),
            _mm512_loadu_si512(B.as_ptr().cast()),
        );
        let zip = _mm512_loadu_si512(ZIP16.as_ptr().cast());
        let unstripe = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
        let pair = |y| {
            let (lo, hi) =
                row_sums::<HP, true, false>(source.add(y * ss), ss, h_coeff, shuffle, h_round);
            let rows = _mm512_packs_epi32(_mm512_srai_epi32::<H>(lo), _mm512_srai_epi32::<H>(hi));
            _mm512_permutexvar_epi8(zip, rows)
        };
        let mut even = pair(0);
        let mut odd = [v_round; VP];
        for (k, window) in odd[..VP - 1].iter_mut().enumerate() {
            *window = pair(2 * k + 1);
        }
        for y in (0..height).step_by(2) {
            odd[VP - 1] = pair(y + 2 * VP - 1);
            let mut a = dot_word(v_round, even, v_coeff[0]);
            let mut b = dot_word(v_round, odd[0], v_coeff[0]);
            even = _mm512_shrdi_epi32::<16>(odd[0], odd[1]);
            for k in 1..VP {
                let even_pair = _mm512_shrdi_epi32::<16>(odd[k - 1], odd[k]);
                a = dot_word(a, even_pair, v_coeff[k]);
                b = dot_word(b, odd[k], v_coeff[k]);
                odd[k - 1] = odd[k];
            }
            let a = _mm512_srai_epi32::<V>(a);
            let b = _mm512_srai_epi32::<V>(b);
            let narrow = if PUT {
                _mm512_min_epu16(_mm512_packus_epi32(a, b), max)
            } else {
                _mm512_packs_epi32(a, b)
            };
            let rows = _mm512_permutexvar_epi64(unstripe, narrow);
            if ds == 16 {
                _mm512_storeu_si512(dst.add(y * ds).cast(), rows);
            } else {
                _mm256_storeu_si256(dst.add(y * ds).cast(), _mm512_castsi512_si256(rows));
                _mm256_storeu_si256(
                    dst.add((y + 1) * ds).cast(),
                    _mm512_extracti64x4_epi64::<1>(rows),
                );
            }
        }
    }
}
