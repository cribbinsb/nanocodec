//! AVX2 [`super::run2`] and [`super::run4`]: byte multiply-adds
//! (`pmaddubsw`) of interleaved sample pairs and weight pairs, 32, 16, 8 or
//! 4 outputs per step; `pmulhrsw` for `run2`'s rounding shift (moved here
//! from the AV1 directional kernels, which it serves with HEVC and AV2),
//! two pair sums and an arithmetic shift for `run4` (`packus` clips).

#![allow(clippy::inline_always)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_maddubs_epi16, _mm_mulhrs_epi16, _mm_packus_epi16, _mm_set1_epi16,
    _mm_srai_epi16, _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
    _mm256_add_epi16, _mm256_loadu_si256, _mm256_maddubs_epi16, _mm256_mulhrs_epi16,
    _mm256_packus_epi16, _mm256_set1_epi16, _mm256_srai_epi16, _mm256_storeu_si256,
    _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
};

use super::{FOUR_TAP_BITS, QUARTER_STEP, Runs};
use crate::intra_prediction::directional::interpolate_run;

/// Outputs per step: a 256-bit, a 128-bit and a 64-bit run.
const WIDE_STEP: usize = 32;
const STEP: usize = 16;
const HALF_STEP: usize = 8;
/// The interpolation weights sum to this.
const WEIGHT_SUM: i16 = 32;
/// `_mm_mulhrs_epi16` by this is `Round2(x, 5)`: `(x · 2^10 + 2^14) >> 15`.
const ROUND2_5: i16 = 1 << 10;

/// [`super::run2`] for the dispatch tables.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn run2_entry(window: &[u8], weight: usize, out: &mut [u8]) {
    // SAFETY: AVX2 is enabled for this function.
    unsafe { run2(window, weight, out) }
}

/// The `pmaddubsw` weight pair `(32 − w, w)` of every 16-bit lane.
#[inline(always)]
fn weight_pair(weight: usize) -> i16 {
    let far = weight as i16;
    (far << 8) | (WEIGHT_SUM - far)
}

/// 32 outputs from `window[..33]` (one 256-bit step).
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline(always)]
unsafe fn step32(window: &[u8], pair: i16, out: &mut [u8]) {
    let (window, out) = (&window[..=WIDE_STEP], &mut out[..WIDE_STEP]);
    // SAFETY: AVX2 is enabled (caller contract); the loads and the store
    // cover the checked subslices.
    unsafe {
        let (weights, round) = (_mm256_set1_epi16(pair), _mm256_set1_epi16(ROUND2_5));
        let a = _mm256_loadu_si256(window.as_ptr().cast());
        let b = _mm256_loadu_si256(window[1..].as_ptr().cast());
        // Per 128-bit lane: unpack, multiply-add and pack keep the sample
        // order.
        let low = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(a, b), weights);
        let high = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(a, b), weights);
        let packed = _mm256_packus_epi16(
            _mm256_mulhrs_epi16(low, round),
            _mm256_mulhrs_epi16(high, round),
        );
        _mm256_storeu_si256(out.as_mut_ptr().cast(), packed);
    }
}

/// 16 outputs from `window[..17]` (one 128-bit step).
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline(always)]
unsafe fn step16(window: &[u8], pair: i16, out: &mut [u8]) {
    let (window, out) = (&window[..=STEP], &mut out[..STEP]);
    // SAFETY: as for `step32`.
    unsafe {
        let (weights, round) = (_mm_set1_epi16(pair), _mm_set1_epi16(ROUND2_5));
        let a = _mm_loadu_si128(window.as_ptr().cast());
        let b = _mm_loadu_si128(window[1..].as_ptr().cast());
        let low = _mm_mulhrs_epi16(_mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), weights), round);
        let high = _mm_mulhrs_epi16(_mm_maddubs_epi16(_mm_unpackhi_epi8(a, b), weights), round);
        _mm_storeu_si128(out.as_mut_ptr().cast(), _mm_packus_epi16(low, high));
    }
}

/// 8 outputs from `window[..9]` (a 64-bit step).
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline(always)]
unsafe fn step8(window: &[u8], pair: i16, out: &mut [u8]) {
    let (window, out) = (&window[..=HALF_STEP], &mut out[..HALF_STEP]);
    // SAFETY: as for `step32`.
    unsafe {
        let (weights, round) = (_mm_set1_epi16(pair), _mm_set1_epi16(ROUND2_5));
        let a = _mm_loadl_epi64(window.as_ptr().cast());
        let b = _mm_loadl_epi64(window[1..].as_ptr().cast());
        let sum = _mm_mulhrs_epi16(_mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), weights), round);
        _mm_storel_epi64(out.as_mut_ptr().cast(), _mm_packus_epi16(sum, sum));
    }
}

/// 4 outputs from `window[..5]` (a 32-bit step).
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline(always)]
unsafe fn step4(window: &[u8], pair: i16, out: &mut [u8]) {
    let (window, out) = (&window[..=QUARTER_STEP], &mut out[..QUARTER_STEP]);
    // SAFETY: as for `step32`.
    unsafe {
        let (weights, round) = (_mm_set1_epi16(pair), _mm_set1_epi16(ROUND2_5));
        let a = _mm_cvtsi32_si128(quarter(window));
        let b = _mm_cvtsi32_si128(quarter(&window[1..]));
        let sum = _mm_mulhrs_epi16(_mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), weights), round);
        out.copy_from_slice(&_mm_cvtsi128_si32(_mm_packus_epi16(sum, sum)).to_le_bytes());
    }
}

/// [`super::run2`], inlined into the caller's AVX2 code.
///
/// # Safety
///
/// AVX2 must be enabled in the caller. Every load and store indexes its
/// slice, so wrong lengths panic.
#[inline(always)]
pub(crate) unsafe fn run2(window: &[u8], weight: usize, out: &mut [u8]) {
    assert_eq!(window.len(), out.len() + 1);
    let len = out.len();
    if len < QUARTER_STEP {
        interpolate_run(window, weight, out);
        return;
    }
    let pair = weight_pair(weight);
    // SAFETY: AVX2 is enabled (caller contract).
    unsafe {
        let mut done = 0;
        while done + WIDE_STEP <= len {
            step32(&window[done..], pair, &mut out[done..]);
            done += WIDE_STEP;
        }
        if done + STEP <= len {
            step16(&window[done..], pair, &mut out[done..]);
            done += STEP;
        }
        // The rest in half (then quarter) steps, the last one ending at
        // `len` (it may recompute outputs already stored, with the same
        // values).
        while done < len {
            if len >= HALF_STEP {
                let at = done.min(len - HALF_STEP);
                step8(&window[at..], pair, &mut out[at..]);
                done = at + HALF_STEP;
            } else {
                let at = done.min(len - QUARTER_STEP);
                step4(&window[at..], pair, &mut out[at..]);
                done = at + QUARTER_STEP;
            }
        }
    }
}

/// [`super::run2`] of a run of `N` outputs (4, 8, 16, or a multiple of
/// 32), with no length dispatch: rows of a fixed block side.
///
/// # Safety
///
/// AVX2 must be enabled in the caller; `window` holds `N + 1` samples and
/// `out` `N`.
#[inline(always)]
pub(crate) unsafe fn run2_fixed<const N: usize>(window: &[u8], weight: usize, out: &mut [u8]) {
    let pair = weight_pair(weight);
    // SAFETY: AVX2 is enabled (caller contract).
    unsafe {
        match N {
            QUARTER_STEP => step4(window, pair, out),
            HALF_STEP => step8(window, pair, out),
            STEP => step16(window, pair, out),
            _ => {
                for at in (0..N).step_by(WIDE_STEP) {
                    step32(&window[at..], pair, &mut out[at..]);
                }
            }
        }
    }
}

/// The first four bytes of `samples` as one little-endian lane.
#[inline(always)]
fn quarter(samples: &[u8]) -> i32 {
    i32::from_le_bytes([samples[0], samples[1], samples[2], samples[3]])
}

/// AVX2 [`Runs`].
pub(crate) struct Avx2Runs;

impl Runs for Avx2Runs {
    #[inline(always)]
    unsafe fn run2(window: &[u8], weight: usize, out: &mut [u8]) {
        // SAFETY: forwarded contract.
        unsafe { run2(window, weight, out) }
    }

    #[inline(always)]
    unsafe fn run2_fixed<const N: usize>(window: &[u8], weight: usize, out: &mut [u8]) {
        // SAFETY: forwarded contract.
        unsafe { run2_fixed::<N>(window, weight, out) }
    }

    #[inline(always)]
    unsafe fn run4(window: &[u8], taps: [i8; 4], out: &mut [u8]) {
        // SAFETY: forwarded contract.
        unsafe { run4(window, taps, out) }
    }
}

/// The byte pair `(first, second)` as one 16-bit weight for `pmaddubsw`.
#[inline(always)]
fn tap_pair(first: i8, second: i8) -> i16 {
    i16::from_le_bytes([first as u8, second as u8])
}

/// One 256-bit `run4` step's 16-bit results (before packing): the
/// `(k, k + 1)` byte pairs by the `near` tap pair plus the `(k + 2, k + 3)`
/// pairs by the `far` pair, rounded and shifted.
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline(always)]
unsafe fn sum256(pairs01: __m256i, pairs23: __m256i, near: __m256i, far: __m256i) -> __m256i {
    // SAFETY: caller contract; register operations only.
    unsafe {
        let round = _mm256_set1_epi16(1 << (FOUR_TAP_BITS - 1));
        let sum = _mm256_add_epi16(
            _mm256_maddubs_epi16(pairs01, near),
            _mm256_maddubs_epi16(pairs23, far),
        );
        _mm256_srai_epi16::<{ FOUR_TAP_BITS as i32 }>(_mm256_add_epi16(sum, round))
    }
}

/// [`sum256`] on 128 bits.
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline(always)]
unsafe fn sum128(pairs01: __m128i, pairs23: __m128i, near: __m128i, far: __m128i) -> __m128i {
    // SAFETY: caller contract; register operations only.
    unsafe {
        let round = _mm_set1_epi16(1 << (FOUR_TAP_BITS - 1));
        let sum = _mm_add_epi16(
            _mm_maddubs_epi16(pairs01, near),
            _mm_maddubs_epi16(pairs23, far),
        );
        _mm_srai_epi16::<{ FOUR_TAP_BITS as i32 }>(_mm_add_epi16(sum, round))
    }
}

/// [`super::run4`], inlined into the caller's AVX2 code: taps 0/1 on the
/// `(k, k + 1)` byte pairs plus taps 2/3 on the `(k + 2, k + 3)` pairs;
/// each pair sum and their total fit i16 (module docs), and `packus` clips.
///
/// # Safety
///
/// AVX2 must be enabled in the caller. Every load and store indexes its
/// slice, so wrong lengths panic.
#[inline(always)]
pub(crate) unsafe fn run4(window: &[u8], taps: [i8; 4], out: &mut [u8]) {
    assert_eq!(window.len(), out.len() + 3);
    let len = out.len();
    if len < QUARTER_STEP {
        super::run4(window, taps, out);
        return;
    }
    let (near, far) = (tap_pair(taps[0], taps[1]), tap_pair(taps[2], taps[3]));
    // SAFETY: AVX2 is enabled (caller contract); every load and store
    // covers checked subslices.
    unsafe {
        let (near_w, far_w) = (_mm256_set1_epi16(near), _mm256_set1_epi16(far));
        let mut done = 0;
        while done + WIDE_STEP <= len {
            let w = &window[done..][..WIDE_STEP + 3];
            let s0 = _mm256_loadu_si256(w[..WIDE_STEP].as_ptr().cast());
            let s1 = _mm256_loadu_si256(w[1..][..WIDE_STEP].as_ptr().cast());
            let s2 = _mm256_loadu_si256(w[2..][..WIDE_STEP].as_ptr().cast());
            let s3 = _mm256_loadu_si256(w[3..][..WIDE_STEP].as_ptr().cast());
            // Per 128-bit lane: unpack, multiply-add and pack keep the
            // sample order.
            let low = sum256(
                _mm256_unpacklo_epi8(s0, s1),
                _mm256_unpacklo_epi8(s2, s3),
                near_w,
                far_w,
            );
            let high = sum256(
                _mm256_unpackhi_epi8(s0, s1),
                _mm256_unpackhi_epi8(s2, s3),
                near_w,
                far_w,
            );
            _mm256_storeu_si256(
                out[done..][..WIDE_STEP].as_mut_ptr().cast(),
                _mm256_packus_epi16(low, high),
            );
            done += WIDE_STEP;
        }
        let (near_w, far_w) = (_mm_set1_epi16(near), _mm_set1_epi16(far));
        if done + STEP <= len {
            let w = &window[done..][..STEP + 3];
            let s0 = _mm_loadu_si128(w[..STEP].as_ptr().cast());
            let s1 = _mm_loadu_si128(w[1..][..STEP].as_ptr().cast());
            let s2 = _mm_loadu_si128(w[2..][..STEP].as_ptr().cast());
            let s3 = _mm_loadu_si128(w[3..][..STEP].as_ptr().cast());
            let low = sum128(
                _mm_unpacklo_epi8(s0, s1),
                _mm_unpacklo_epi8(s2, s3),
                near_w,
                far_w,
            );
            let high = sum128(
                _mm_unpackhi_epi8(s0, s1),
                _mm_unpackhi_epi8(s2, s3),
                near_w,
                far_w,
            );
            _mm_storeu_si128(
                out[done..][..STEP].as_mut_ptr().cast(),
                _mm_packus_epi16(low, high),
            );
            done += STEP;
        }
        // The rest in half (then quarter) steps, the last one ending at
        // `len` (it may recompute outputs already stored, with the same
        // values).
        let step = if len < HALF_STEP {
            QUARTER_STEP
        } else {
            HALF_STEP
        };
        while done < len {
            let at = done.min(len - step);
            let w = &window[at..][..step + 3];
            let (s0, s1, s2, s3) = if step == HALF_STEP {
                (
                    _mm_loadl_epi64(w[..HALF_STEP].as_ptr().cast()),
                    _mm_loadl_epi64(w[1..][..HALF_STEP].as_ptr().cast()),
                    _mm_loadl_epi64(w[2..][..HALF_STEP].as_ptr().cast()),
                    _mm_loadl_epi64(w[3..][..HALF_STEP].as_ptr().cast()),
                )
            } else {
                (
                    _mm_cvtsi32_si128(quarter(w)),
                    _mm_cvtsi32_si128(quarter(&w[1..])),
                    _mm_cvtsi32_si128(quarter(&w[2..])),
                    _mm_cvtsi32_si128(quarter(&w[3..])),
                )
            };
            let sum = sum128(
                _mm_unpacklo_epi8(s0, s1),
                _mm_unpacklo_epi8(s2, s3),
                near_w,
                far_w,
            );
            let packed = _mm_packus_epi16(sum, sum);
            if step == HALF_STEP {
                _mm_storel_epi64(out[at..][..HALF_STEP].as_mut_ptr().cast(), packed);
            } else {
                out[at..at + QUARTER_STEP]
                    .copy_from_slice(&_mm_cvtsi128_si32(packed).to_le_bytes());
            }
            done = at + step;
        }
    }
}
