//! SSE2 symbol kernels: the bounds of up to 16 symbols in two 8-lane
//! vectors (kernels only; contract in the parent module).
use super::{Av1SymbolInterval, LANES, MIN_PROBABILITY, PROB_TOP, PROBABILITY_SHIFT, padded};
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_and_si128, _mm_andnot_si128, _mm_cmpeq_epi16, _mm_cmplt_epi16,
    _mm_cvtsi32_si128, _mm_loadu_si128, _mm_movemask_epi8, _mm_mulhi_epu16, _mm_or_si128,
    _mm_set1_epi16, _mm_setr_epi16, _mm_setzero_si128, _mm_slli_epi16, _mm_srl_epi16,
    _mm_srli_epi16, _mm_storeu_si128, _mm_sub_epi16, _mm_subs_epu16,
};

/// The left shift that makes the high half of a 16x16-bit product of the
/// shifted probability and `(range >> 8) << 8` the bound's split.
const PRODUCT_SHIFT: i32 = 16 - 8 - (7 - PROBABILITY_SHIFT as i32);

/// Load 8 lanes.
#[target_feature(enable = "sse2")]
#[inline]
fn load(lanes: &[u16; LANES]) -> __m128i {
    // SAFETY: `lanes` is 16 readable bytes; the load is unaligned.
    unsafe { _mm_loadu_si128(lanes.as_ptr().cast()) }
}

/// Store 8 lanes.
#[target_feature(enable = "sse2")]
#[inline]
fn store(lanes: &mut [u16], v: __m128i) {
    let lanes: &mut [u16; LANES] = (&mut lanes[..LANES]).try_into().expect("8 lanes");
    // SAFETY: `lanes` is 16 writable bytes; the store is unaligned.
    unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast(), v) }
}

/// The two 8-lane halves of 16 lanes.
#[target_feature(enable = "sse2")]
#[inline]
fn halves(lanes: &[u16; 2 * LANES]) -> (&[u16; LANES], &[u16; LANES]) {
    let (low, high) = lanes.split_at(LANES);
    (
        low.try_into().expect("8 lanes"),
        high.try_into().expect("8 lanes"),
    )
}

/// Lane indices of the low and high vector.
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

/// The bounds (`cur`) of eight symbols, 0 from the last symbol on.
#[target_feature(enable = "sse2")]
#[inline]
fn bounds(icdf: __m128i, index: __m128i, scale: __m128i, last: __m128i) -> __m128i {
    // `((range >> 8) * (icdf >> 6)) >> 1` as one high-half product of
    // `(icdf >> 6) << 7` (below 65536 because icdf < 32768) and
    // `(range >> 8) << 8`.
    let probability = _mm_slli_epi16(
        _mm_srli_epi16(icdf, PROBABILITY_SHIFT as i32),
        PRODUCT_SHIFT,
    );
    let split = _mm_mulhi_epu16(probability, scale);
    // MIN_PROBABILITY * (last - i), wrapping past `last` (masked below).
    let offset = _mm_slli_epi16(_mm_sub_epi16(last, index), MIN_PROBABILITY.ilog2() as i32);
    let valid = _mm_cmplt_epi16(index, last);
    _mm_and_si128(_mm_add_epi16(split, offset), valid)
}

/// Bit pairs set for the lanes whose bound is above `value`.
#[target_feature(enable = "sse2")]
#[inline]
fn above(bounds: __m128i, value: __m128i) -> u32 {
    let at_or_below = _mm_cmpeq_epi16(_mm_subs_epu16(bounds, value), _mm_setzero_si128());
    !(_mm_movemask_epi8(at_or_below) as u32) & 0xffff
}

#[target_feature(enable = "sse2")]
#[inline]
fn find_symbol_tf<const N: usize>(
    icdf: &[u16; N],
    symbols: usize,
    range: u32,
    value: u32,
) -> Av1SymbolInterval {
    let scale = _mm_set1_epi16((range & 0xff00) as u16 as i16);
    let last = _mm_set1_epi16((symbols - 1) as i16);
    let value_lanes = _mm_set1_epi16(value as u16 as i16);
    // `range` then the bounds: the interval of symbol `s` is
    // `[edges[s + 1], edges[s])`.
    let mut edges = [0u16; 2 * LANES + 1];
    edges[0] = range as u16;
    let lanes: [u16; 2 * LANES] = padded(icdf);
    let (low_icdf, high_icdf) = halves(&lanes);
    let low = bounds(load(low_icdf), lane_index(false), scale, last);
    store(&mut edges[1..], low);
    let mut mask = above(low, value_lanes);
    if N > LANES {
        let high = bounds(load(high_icdf), lane_index(true), scale, last);
        store(&mut edges[1 + LANES..], high);
        mask |= above(high, value_lanes) << (2 * LANES);
    }
    // The lanes above `value` are a prefix (bounds fall with the symbol).
    let symbol = (mask.trailing_ones() / 2) as usize;
    Av1SymbolInterval {
        symbol,
        lower: u32::from(edges[symbol + 1]),
        upper: u32::from(edges[symbol]),
    }
}

/// Adapt eight lanes: towards 32768 below `symbol`, towards 0 from it on.
#[target_feature(enable = "sse2")]
#[inline]
fn adapt(icdf: __m128i, index: __m128i, symbol: __m128i, rate: __m128i) -> __m128i {
    let towards_top = _mm_add_epi16(
        icdf,
        _mm_srl_epi16(_mm_sub_epi16(_mm_set1_epi16(PROB_TOP as i16), icdf), rate),
    );
    let towards_zero = _mm_sub_epi16(icdf, _mm_srl_epi16(icdf, rate));
    let below = _mm_cmplt_epi16(index, symbol);
    _mm_or_si128(
        _mm_and_si128(below, towards_top),
        _mm_andnot_si128(below, towards_zero),
    )
}

#[target_feature(enable = "sse2")]
#[inline]
fn adapt_cdf_tf<const N: usize>(icdf: &mut [u16; N], symbol: usize, rate: u32) {
    let rate = _mm_cvtsi32_si128(rate as i32);
    let symbol = _mm_set1_epi16(symbol as i16);
    let mut lanes: [u16; 2 * LANES] = padded(icdf);
    let (low, high) = halves(&lanes);
    let (low, high) = (load(low), load(high));
    store(
        &mut lanes[..LANES],
        adapt(low, lane_index(false), symbol, rate),
    );
    if N > LANES {
        store(
            &mut lanes[LANES..],
            adapt(high, lane_index(true), symbol, rate),
        );
    }
    icdf.copy_from_slice(&lanes[..N]);
}

/// [`super::av1_find_symbol`] with sse2.
#[inline]
pub(super) fn find_symbol<const N: usize>(
    icdf: &[u16; N],
    symbols: usize,
    range: u32,
    value: u32,
) -> Av1SymbolInterval {
    // SAFETY: sse2 is the target's baseline, enabled in every build that
    // compiles this module, which is all the kernel requires.
    unsafe { find_symbol_tf(icdf, symbols, range, value) }
}

/// [`super::av1_adapt_cdf`] with sse2.
#[inline]
pub(super) fn adapt_cdf<const N: usize>(icdf: &mut [u16; N], symbol: usize, rate: u32) {
    // SAFETY: as for `find_symbol`.
    unsafe { adapt_cdf_tf(icdf, symbol, rate) }
}
