//! Fused SSE2 symbol step, guided by dav1d 1.5.4 `src/x86/msac.asm`
//! `msac_decode_symbol_adapt4/8/16` (BSD-2-Clause, Copyright 2019
//! VideoLAN/dav1d authors and Two Orioles, LLC; `THIRD_PARTY.md`).
//! Interval search and CDF adaptation share the loaded probabilities and
//! comparison mask, without stack round-trips.
use super::{
    Av1SymbolStep, LANES, MIN_PROBABILITY, PROBABILITY_SHIFT, padded,
    step::{interval_4, interval_8, interval_16, normalize},
};
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_and_si128, _mm_andnot_si128, _mm_cmpeq_epi16, _mm_cmplt_epi16,
    _mm_cvtsi32_si128, _mm_cvtsi128_si64, _mm_loadl_epi64, _mm_loadu_si128, _mm_movemask_epi8,
    _mm_mulhi_epu16, _mm_set1_epi16, _mm_setr_epi16, _mm_setzero_si128, _mm_slli_epi16,
    _mm_sra_epi16, _mm_srli_epi16, _mm_storel_epi64, _mm_storeu_si128, _mm_sub_epi16,
    _mm_subs_epu16, _mm_unpackhi_epi64,
};

const PRODUCT_SHIFT: i32 = 16 - 8 - (7 - PROBABILITY_SHIFT as i32);

#[target_feature(enable = "sse2")]
#[inline]
fn lane_index(high: bool) -> __m128i {
    let base = if high { LANES as i16 } else { 0 };
    _mm_setr_epi16(
        base,
        base + 1,
        base + 2,
        base + 3,
        base + 4,
        base + 5,
        base + 6,
        base + 7,
    )
}

#[target_feature(enable = "sse2")]
#[inline]
fn bounds8(icdf: __m128i, index: __m128i, scale: __m128i, last: __m128i) -> __m128i {
    let prob = _mm_slli_epi16(
        _mm_srli_epi16(icdf, PROBABILITY_SHIFT as i32),
        PRODUCT_SHIFT,
    );
    let split = _mm_mulhi_epu16(prob, scale);
    let offset = _mm_slli_epi16(_mm_sub_epi16(last, index), MIN_PROBABILITY.ilog2() as i32);
    _mm_and_si128(_mm_add_epi16(split, offset), _mm_cmplt_epi16(index, last))
}

/// Signed biased adaptation: `p - ((p - (before & (32769 - 2^rate))) >>s rate)`,
/// exact for every `p` in `0..32768` including nonterminal zeros.
#[target_feature(enable = "sse2")]
#[inline]
fn adapt8(p: __m128i, selected: __m128i, rate: u32) -> __m128i {
    let bias = _mm_set1_epi16((32769 - (1u32 << rate)) as i16);
    let delta = _mm_sub_epi16(p, _mm_andnot_si128(selected, bias));
    _mm_sub_epi16(p, _mm_sra_epi16(delta, _mm_cvtsi32_si128(rate as i32)))
}

#[target_feature(enable = "sse2")]
#[inline]
fn halves64(v: __m128i) -> (u64, u64) {
    (
        _mm_cvtsi128_si64(v) as u64,
        _mm_cvtsi128_si64(_mm_unpackhi_epi64(v, v)) as u64,
    )
}

#[target_feature(enable = "sse2")]
#[inline]
fn four(
    icdf: &mut [u16; 4],
    symbols: usize,
    window: u64,
    range: u32,
    rate: Option<u32>,
) -> Av1SymbolStep {
    // SAFETY: `icdf` holds four initialized `u16`s (8 bytes).
    let p = unsafe { _mm_loadl_epi64(icdf.as_ptr().cast()) };
    let scale = _mm_set1_epi16((range & 0xff00) as u16 as i16);
    let last = _mm_set1_epi16((symbols - 1) as i16);
    let val = _mm_set1_epi16((window >> 48) as u16 as i16);
    let b = bounds8(p, lane_index(false), scale, last);
    let selected = _mm_cmpeq_epi16(_mm_subs_epu16(b, val), _mm_setzero_si128());
    let symbol = ((_mm_movemask_epi8(selected) as u32).trailing_zeros() / 2) as usize;
    if let Some(rate) = rate {
        // SAFETY: `icdf` is 8 writable bytes.
        unsafe { _mm_storel_epi64(icdf.as_mut_ptr().cast(), adapt8(p, selected, rate)) };
    }
    let (lower, upper) = interval_4(range, _mm_cvtsi128_si64(b) as u64, symbol);
    normalize(symbol, window, lower, upper - lower)
}

#[target_feature(enable = "sse2")]
#[inline]
fn eight<const N: usize>(
    icdf: &mut [u16; N],
    symbols: usize,
    window: u64,
    range: u32,
    rate: Option<u32>,
) -> Av1SymbolStep {
    let p = if N == LANES {
        // SAFETY: `N == 8`, so `icdf` is 16 readable bytes.
        unsafe { _mm_loadu_si128(icdf.as_ptr().cast()) }
    } else {
        let lanes: [u16; LANES] = padded(icdf);
        // SAFETY: `lanes` is 16 readable bytes.
        unsafe { _mm_loadu_si128(lanes.as_ptr().cast()) }
    };
    let scale = _mm_set1_epi16((range & 0xff00) as u16 as i16);
    let last = _mm_set1_epi16((symbols - 1) as i16);
    let val = _mm_set1_epi16((window >> 48) as u16 as i16);
    let b = bounds8(p, lane_index(false), scale, last);
    let selected = _mm_cmpeq_epi16(_mm_subs_epu16(b, val), _mm_setzero_si128());
    let symbol = ((_mm_movemask_epi8(selected) as u32).trailing_zeros() / 2) as usize;
    if let Some(rate) = rate {
        let adapted = adapt8(p, selected, rate);
        if N == LANES {
            // SAFETY: `N == 8`, so `icdf` is 16 writable bytes.
            unsafe { _mm_storeu_si128(icdf.as_mut_ptr().cast(), adapted) };
        } else {
            let mut lanes = [0u16; LANES];
            // SAFETY: `lanes` is 16 writable bytes.
            unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast(), adapted) };
            icdf.copy_from_slice(&lanes[..N]);
        }
    }
    let (lo, hi) = halves64(b);
    let (lower, upper) = interval_8(range, lo, hi, symbol);
    normalize(symbol, window, lower, upper - lower)
}

#[target_feature(enable = "sse2")]
#[inline]
fn sixteen<const N: usize>(
    icdf: &mut [u16; N],
    symbols: usize,
    window: u64,
    range: u32,
    rate: Option<u32>,
) -> Av1SymbolStep {
    let (p0, p1) = if N == 2 * LANES {
        // SAFETY: `N == 16`, so `icdf` is 32 readable bytes.
        unsafe {
            let ptr = icdf.as_ptr();
            (
                _mm_loadu_si128(ptr.cast()),
                _mm_loadu_si128(ptr.add(LANES).cast()),
            )
        }
    } else {
        let lanes: [u16; 2 * LANES] = padded(icdf);
        // SAFETY: `lanes` is 32 readable bytes.
        unsafe {
            let ptr = lanes.as_ptr();
            (
                _mm_loadu_si128(ptr.cast()),
                _mm_loadu_si128(ptr.add(LANES).cast()),
            )
        }
    };
    let scale = _mm_set1_epi16((range & 0xff00) as u16 as i16);
    let last = _mm_set1_epi16((symbols - 1) as i16);
    let val = _mm_set1_epi16((window >> 48) as u16 as i16);
    let zero = _mm_setzero_si128();
    let b0 = bounds8(p0, lane_index(false), scale, last);
    let b1 = bounds8(p1, lane_index(true), scale, last);
    let sel0 = _mm_cmpeq_epi16(_mm_subs_epu16(b0, val), zero);
    let sel1 = _mm_cmpeq_epi16(_mm_subs_epu16(b1, val), zero);
    let mask = (_mm_movemask_epi8(sel0) as u32) | ((_mm_movemask_epi8(sel1) as u32) << 16);
    let symbol = (mask.trailing_zeros() / 2) as usize;
    if let Some(rate) = rate {
        let (a0, a1) = (adapt8(p0, sel0, rate), adapt8(p1, sel1, rate));
        if N == 2 * LANES {
            // SAFETY: `N == 16`, so `icdf` is 32 writable bytes.
            unsafe {
                let ptr = icdf.as_mut_ptr();
                _mm_storeu_si128(ptr.cast(), a0);
                _mm_storeu_si128(ptr.add(LANES).cast(), a1);
            }
        } else {
            let mut lanes = [0u16; 2 * LANES];
            // SAFETY: `lanes` is 32 writable bytes.
            unsafe {
                let ptr = lanes.as_mut_ptr();
                _mm_storeu_si128(ptr.cast(), a0);
                _mm_storeu_si128(ptr.add(LANES).cast(), a1);
            }
            icdf.copy_from_slice(&lanes[..N]);
        }
    }
    let ((lo0, hi0), (lo1, hi1)) = (halves64(b0), halves64(b1));
    let (lower, upper) = interval_16(range, (lo0, hi0, lo1, hi1), symbol);
    normalize(symbol, window, lower, upper - lower)
}

#[inline]
pub(super) fn decode<const N: usize>(
    icdf: &mut [u16; N],
    symbols: usize,
    window: u64,
    range: u32,
    rate: Option<u32>,
) -> Av1SymbolStep {
    // SAFETY: SSE2 is baseline on x86_64; parent restricts `N` to `4..=16`.
    unsafe {
        if N == 4 {
            let icdf4: &mut [u16; 4] = (&mut icdf[..4]).try_into().expect("4 lanes");
            four(icdf4, symbols, window, range, rate)
        } else if N <= LANES {
            eight(icdf, symbols, window, range, rate)
        } else {
            sixteen(icdf, symbols, window, range, rate)
        }
    }
}
