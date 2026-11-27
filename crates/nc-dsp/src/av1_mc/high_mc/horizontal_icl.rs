//! Ice Lake horizontal 16bpc AV1 after dav1d 1.5.4 `mc16_avx512.asm`
//! (BSD-2-Clause). One 32-word stripe uses paired word dots in each
//! 128-bit lane and packs the first and second four samples together.

mod narrow;

use super::vertical_icl::dot_word;
use super::{
    AV1_FILTER_BEFORE, AV1_FILTER_TAPS, BitDepth, FILTER_BITS, av1_compound_bias, inter_round0,
    tap_pair_count, tap_pairs,
};
use core::arch::x86_64::{
    __m512i, _mm256_loadu_si256, _mm256_maskz_loadu_epi16, _mm256_storeu_si256,
    _mm512_castpd_si512, _mm512_castsi256_si512, _mm512_castsi512_pd, _mm512_castsi512_si256,
    _mm512_extracti64x4_epi64, _mm512_inserti64x4, _mm512_loadu_si512, _mm512_maskz_loadu_epi16,
    _mm512_min_epu16, _mm512_packs_epi32, _mm512_packus_epi32, _mm512_set1_epi16,
    _mm512_set1_epi32, _mm512_shuffle_epi8, _mm512_shuffle_pd, _mm512_srai_epi32,
    _mm512_storeu_si512,
};

/// The 128-bit shuffle indexes the two words of each adjacent pair for
/// four successive outputs. The two masks start at words 0 and 2.
const fn shuffle_words(offset: usize) -> [u8; 64] {
    let mut mask = [0; 64];
    let mut i = 0;
    while i < 64 {
        let lane = i % 16;
        mask[i] = (2 * (offset + lane / 4) + lane % 4) as u8;
        i += 1;
    }
    mask
}

pub(super) const A: [u8; 64] = shuffle_words(0);
pub(super) const B: [u8; 64] = shuffle_words(2);

/// # Safety
/// The verified eight-tap window and destination cover the block and the
/// selected CPU supports Ice Lake's word dots.
#[inline(always)]
pub(super) unsafe fn try_filter<const W: usize, const PUT: bool>(
    src: (*const u16, usize),
    dst: (*mut u16, usize),
    h: usize,
    fx: &[i16; AV1_FILTER_TAPS],
    depth: BitDepth,
) -> bool {
    if matches!(W, 4 | 8) {
        // SAFETY: same padded window and Ice Lake feature set.
        return unsafe { narrow::try_filter::<W, PUT>(src, dst, h, fx, depth) };
    }
    let pairs = tap_pair_count(fx);
    if !matches!(W, 16 | 32 | 64)
        || (W == 16 && !h.is_multiple_of(2))
        || pairs < 3
        || !matches!(depth.bits(), 10 | 12)
    {
        return false;
    }
    let r0 = inter_round0(depth);
    // SAFETY: checked rectangles above. Width sixteen uses pairs of rows,
    // and wider blocks use complete stripes from one row.
    unsafe {
        if W == 16 {
            filter::<true, PUT>(src, dst, (W, h), fx, pairs, depth, r0);
        } else {
            filter::<false, PUT>(src, dst, (W, h), fx, pairs, depth, r0);
        }
    }
    true
}

#[inline(always)]
unsafe fn filter<const TWO_ROWS: bool, const PUT: bool>(
    src: (*const u16, usize),
    dst: (*mut u16, usize),
    size: (usize, usize),
    fx: &[i16; AV1_FILTER_TAPS],
    pairs: usize,
    depth: BitDepth,
    r0: u32,
) {
    // SAFETY: the caller has checked the block, layout and ISA.
    unsafe {
        if PUT {
            let round = (1 << (FILTER_BITS - 1)) + (1 << (r0 - 1));
            if pairs == 3 {
                rows::<3, true, FILTER_BITS, TWO_ROWS>(src, dst, size, fx, round, depth);
            } else {
                rows::<4, true, FILTER_BITS, TWO_ROWS>(src, dst, size, fx, round, depth);
            }
        } else {
            let round = (1 << (r0 - 1)) - (av1_compound_bias(depth) << r0);
            match (pairs, r0) {
                (3, 5) => rows::<3, false, 5, TWO_ROWS>(src, dst, size, fx, round, depth),
                (4, 5) => rows::<4, false, 5, TWO_ROWS>(src, dst, size, fx, round, depth),
                (3, _) => rows::<3, false, 3, TWO_ROWS>(src, dst, size, fx, round, depth),
                _ => rows::<4, false, 3, TWO_ROWS>(src, dst, size, fx, round, depth),
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn load<const TWO_ROWS: bool>(row: *const u16, stride: usize) -> __m512i {
    // SAFETY: the caller covers 32 words, in one or two rows.
    unsafe {
        if TWO_ROWS {
            _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_loadu_si256(row.cast())),
                _mm256_loadu_si256(row.add(stride).cast()),
            )
        } else {
            _mm512_loadu_si512(row.cast())
        }
    }
}

/// The two dword vectors are in the matching left/right halves of each
/// 128-bit lane. Packing the halves restores the horizontal sample order.
/// `LAST_PADDED` allows an unused word after the taps on the final read.
/// The horizontal-only pass has another padded row, while the separable
/// pass can reach the last word of the whole checked source span.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub(super) unsafe fn row_sums<const PAIRS: usize, const TWO_ROWS: bool, const LAST_PADDED: bool>(
    ptr: *const u16,
    ss: usize,
    coeff: [__m512i; PAIRS],
    (even_pairs, odd_pairs): (__m512i, __m512i),
    round: __m512i,
) -> (__m512i, __m512i) {
    // SAFETY: caller validates the padded horizontal load footprint.
    unsafe {
        let first = load::<TWO_ROWS>(ptr, ss);
        let fa = _mm512_shuffle_epi8(first, even_pairs);
        let fb = _mm512_shuffle_epi8(first, odd_pairs);
        if PAIRS == 3 {
            let last = load::<TWO_ROWS>(ptr.add(6), ss);
            let la = _mm512_shuffle_epi8(last, even_pairs);
            let lb = _mm512_shuffle_epi8(last, odd_pairs);
            let cross = _mm512_castpd_si512(_mm512_shuffle_pd::<0x55>(
                _mm512_castsi512_pd(fb),
                _mm512_castsi512_pd(la),
            ));
            let left = dot_word(round, fa, coeff[0]);
            let left = dot_word(left, fb, coeff[1]);
            let right = dot_word(round, cross, coeff[0]);
            let right = dot_word(right, la, coeff[1]);
            (
                dot_word(left, cross, coeff[2]),
                dot_word(right, lb, coeff[2]),
            )
        } else {
            let tail = ptr.add(8);
            let last = if LAST_PADDED {
                load::<TWO_ROWS>(tail, ss)
            } else if TWO_ROWS {
                _mm512_inserti64x4::<1>(
                    _mm512_castsi256_si512(_mm256_maskz_loadu_epi16(0x7fff, tail.cast())),
                    _mm256_maskz_loadu_epi16(0x7fff, tail.add(ss).cast()),
                )
            } else {
                _mm512_maskz_loadu_epi16(0x7fff_ffff, tail.cast())
            };
            let mid = if TWO_ROWS {
                _mm512_castpd_si512(_mm512_shuffle_pd::<0x55>(
                    _mm512_castsi512_pd(first),
                    _mm512_castsi512_pd(last),
                ))
            } else {
                load::<TWO_ROWS>(ptr.add(4), ss)
            };
            let ma = _mm512_shuffle_epi8(mid, even_pairs);
            let mb = _mm512_shuffle_epi8(mid, odd_pairs);
            let la = _mm512_shuffle_epi8(last, even_pairs);
            let lb = _mm512_shuffle_epi8(last, odd_pairs);
            let left = dot_word(round, fa, coeff[0]);
            let left = dot_word(left, fb, coeff[1]);
            let left = dot_word(left, ma, coeff[2]);
            let right = dot_word(round, ma, coeff[0]);
            let right = dot_word(right, mb, coeff[1]);
            let right = dot_word(right, la, coeff[2]);
            (dot_word(left, mb, coeff[3]), dot_word(right, lb, coeff[3]))
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn rows<const PAIRS: usize, const PUT: bool, const SHIFT: u32, const TWO_ROWS: bool>(
    (src, ss): (*const u16, usize),
    (dst, ds): (*mut u16, usize),
    (width, height): (usize, usize),
    taps: &[i16; AV1_FILTER_TAPS],
    round: i32,
    depth: BitDepth,
) {
    // SAFETY: the original validated window has height + 7 rows; this pass
    // reads rows 3 through height + 2. The last 8-tap load overlaps one extra
    // word, still inside the padded source even on the last output row.
    unsafe {
        let round = _mm512_set1_epi32(round);
        let max = _mm512_set1_epi16(depth.max() as i16);
        let coeff = tap_pairs::<PAIRS>(taps).map(|(even_pairs, odd_pairs)| {
            _mm512_set1_epi32(crate::simd::packed_pair(even_pairs, odd_pairs))
        });
        let even_pairs = _mm512_loadu_si512(A.as_ptr().cast());
        let odd_pairs = _mm512_loadu_si512(B.as_ptr().cast());
        let first = AV1_FILTER_TAPS / 2 - PAIRS;
        let row_step = if TWO_ROWS { 2 } else { 1 };
        let stripe_count = if TWO_ROWS { 1 } else { width / 32 };
        for row_pair in 0..(height / row_step) {
            let y = row_pair * row_step;
            let row = src.add((AV1_FILTER_BEFORE + y) * ss + first);
            let out = dst.add(y * ds);
            for stripe in 0..stripe_count {
                let x = stripe * 32;
                let ptr = row.add(x);
                let (left, right) = row_sums::<PAIRS, TWO_ROWS, true>(
                    ptr,
                    ss,
                    coeff,
                    (even_pairs, odd_pairs),
                    round,
                );
                let left = _mm512_srai_epi32::<SHIFT>(left);
                let right = _mm512_srai_epi32::<SHIFT>(right);
                let packed = if PUT {
                    _mm512_min_epu16(_mm512_packus_epi32(left, right), max)
                } else {
                    _mm512_packs_epi32(left, right)
                };
                if TWO_ROWS {
                    _mm256_storeu_si256(out.cast(), _mm512_castsi512_si256(packed));
                    _mm256_storeu_si256(out.add(ds).cast(), _mm512_extracti64x4_epi64::<1>(packed));
                } else {
                    _mm512_storeu_si512(out.add(x).cast(), packed);
                }
            }
        }
    }
}
