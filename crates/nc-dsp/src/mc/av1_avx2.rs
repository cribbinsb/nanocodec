//! AVX2 AV1 prediction at pixel precision (`put`) and compound precision
//! (`prep`), after dav1d's `put_8tap` / `prep_8tap` (`src/x86/mc_avx2.asm`, BSD-2-Clause, Copyright (c) 2018,
//! `VideoLAN` and dav1d authors; see `THIRD_PARTY.md`), reimplemented: the
//! horizontal pass multiplies unsigned pixels by signed tap pairs
//! (`vpmaddubsw`, pixels gathered by `vpshufb`), the vertical pass of a
//! 2-D filter multiplies 16-bit intermediates by tap pairs (`vpmaddwd`), and
//! roundings by powers of two are one `vpmulhrsw`.
//!
//! Arithmetic (module `mc` docs): every AV1 tap is even, so the taps are
//! halved (they then fit signed bytes, at most 64) and the shifts drop by
//! one. With halved taps a pair's products stay within ±32130 and a full
//! sum within ±30600 (`halved_sums_fit_i16`), so the 16-bit horizontal sums
//! are exact. The passes round as the reference does:
//! - horizontal only: `((sum + 2) >> 2 + 8) >> 4`;
//! - vertical only: `(sum + 32) >> 6`;
//! - both: `t = (sum + 2) >> 2`, then `(Σ t·tap + 512) >> 10`;
//!
//! each clipped to 8 bits. `prep` keeps 16-bit words (AV1 §7.11.3.4,
//! `InterRound1` = 7): `(sum + 2) >> 2` horizontally, `(sum + 32) >> 6`
//! vertically, `(Σ t·tap + 32) >> 6` after a horizontal pass, and a whole-
//! sample copy scales by 16. One `Sink` per precision does the stores, so
//! the filter paths are shared. `vpmulhrsw(x, 2^(15-k))` is
//! `(x + 2^(k-1)) >> k`.
//!
//! Layout: blocks 16 or more wide filter 16 outputs of one row per vector
//! (a 16-column strip at a time). Blocks 4 or 8 wide put two rows in one
//! vector, row `k` in the low lane and row `k + 1` in the high one; a
//! vertical tap pair then interleaves vectors `k` and `k + 1`, whose low
//! lane pairs rows (k, k+1) for output row `r` and whose high lane pairs
//! rows (k+1, k+2) for output row `r + 1`: two output rows per step. The
//! vertical pass of wide blocks also makes two rows per step, from an even
//! and an odd window of interleaved row pairs, so every new row is
//! interleaved once.
//!
//! Families whose outer taps are zero (regular and smooth, and every
//! 4-tap filter) filter six taps from one sample in, never loading the
//! outer samples.
//!
//! Kernel contract: see the `mc` module docs; the window and destination
//! are checked once on entry and the row loops then use raw pointers.

#![allow(
    clippy::inline_always,
    reason = "the rounding schemes must inline into the AVX2 kernels"
)]

use super::avx2_byte_dot::{byte_pairs, row_16, two_rows};
use super::{
    AnyTaps, Av1Family, Av1MixedFns, Av1Rounding, ConvolveFn, OuterTaps, PREP_LAST_ROUND,
    PREP_LAST_SHIFT, PrepFn, ROUND_2, ROUND_6, Rounding, SeparableRounding, SeparableTaps,
    TAP_CENTRE, TAPS, TapPair, slide,
};
use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8, _mm256_add_epi16,
    _mm256_add_epi32, _mm256_castsi128_si256, _mm256_castsi256_si128, _mm256_extracti128_si256,
    _mm256_inserti128_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_mulhrs_epi16,
    _mm256_packs_epi32, _mm256_packus_epi16, _mm256_permute2x128_si256, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_sra_epi32, _mm256_storeu_si256, _mm256_unpackhi_epi16,
    _mm256_unpacklo_epi16,
};

/// Taps of HEVC and VVC chroma filters, and the window column of their
/// centre sample.
const SHORT_TAPS: usize = 4;
const SHORT_CENTRE: usize = 1;
/// Outputs of one vector (16 lanes of 16 bits).
const LANES: usize = 16;
/// Outputs of one 128-bit lane.
const HALF: usize = 8;

/// The rounded horizontal sums of a 2-D filter ([`Rounding::FIRST`]).
#[target_feature(enable = "avx2")]
#[inline]
fn first<R: Rounding>(sum: __m256i) -> __m256i {
    match R::FIRST {
        Some(m) => round(sum, m),
        None => sum,
    }
}

/// A horizontal-only filter's 16-bit results for sink `S`.
#[target_feature(enable = "avx2")]
#[inline]
fn horizontal<R: Rounding, S: Sink>(sum: __m256i) -> __m256i {
    debug_assert!(
        !S::PREP || R::HALVED,
        "HEVC/VVC compound runs the generic kernels"
    );
    let t = first::<R>(sum);
    if S::PREP {
        t
    } else {
        round(t, R::HORIZONTAL_PUT)
    }
}

/// Pass selection after resolving window, rounding, and taps. The
/// one-axis passes for blocks four or eight wide are shared.
pub(super) trait WidePasses {
    /// Horizontal only (see [`horizontal_wide`]).
    ///
    /// # Safety
    ///
    /// As [`put_taps`]; the CPU must support the passes' ISA.
    unsafe fn horizontal<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
        window: Window,
        lead: usize,
        out: S,
        h: usize,
        taps: &[i16],
    );
    /// Vertical only (see [`vertical_wide`]).
    ///
    /// # Safety
    ///
    /// As [`WidePasses::horizontal`].
    unsafe fn vertical<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
        window: Window,
        lead: usize,
        out: S,
        h: usize,
        taps: &[i16],
    );
    /// Both directions (see [`both_wide`]).
    ///
    /// # Safety
    ///
    /// As [`WidePasses::horizontal`].
    unsafe fn both<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
        window: Window,
        lead: usize,
        out: S,
        h: usize,
        taps: (&[i16], &[i16]),
    );
}

/// The AVX2 passes of 16-output strips.
pub(super) struct Avx2Wide;

impl WidePasses for Avx2Wide {
    #[inline(always)]
    unsafe fn horizontal<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
        window: Window,
        lead: usize,
        out: S,
        h: usize,
        taps: &[i16],
    ) {
        // SAFETY: forwarded from the caller.
        unsafe { horizontal_wide::<W, P, S, R, C>(window, lead, out, h, taps) }
    }
    #[inline(always)]
    unsafe fn vertical<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
        window: Window,
        lead: usize,
        out: S,
        h: usize,
        taps: &[i16],
    ) {
        // SAFETY: forwarded from the caller.
        unsafe { vertical_wide::<W, P, S, R, C>(window, lead, out, h, taps) }
    }
    #[inline(always)]
    unsafe fn both<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
        window: Window,
        lead: usize,
        out: S,
        h: usize,
        taps: (&[i16], &[i16]),
    ) {
        // SAFETY: forwarded from the caller.
        unsafe {
            if W <= HALF {
                both_narrow::<W, P, S, R, C>(window, lead, out, h, taps);
            } else {
                both_wide::<W, P, S, R, C>(window, lead, out, h, taps);
            }
        }
    }
}

/// The AV1 compound-precision kernels of filter family `F` by width class.
pub(super) fn prep_family<F: Av1Family>() -> [PrepFn; 5] {
    [
        prep::<F, 4, Avx2Wide>,
        prep::<F, 8, Avx2Wide>,
        prep::<F, 16, Avx2Wide>,
        prep::<F, 32, Avx2Wide>,
        prep::<F, 64, Avx2Wide>,
    ]
}

/// The AV1 kernels of filter family `F` by width class (4 to 64).
pub(super) fn family<F: Av1Family>() -> [ConvolveFn; 5] {
    [
        put::<F, 4, Avx2Wide>,
        put::<F, 8, Avx2Wide>,
        put::<F, 16, Avx2Wide>,
        put::<F, 32, Avx2Wide>,
        put::<F, 64, Avx2Wide>,
    ]
}

/// The mixed-family kernels by width class (4 to 64).
pub(super) fn mixed_fns() -> Av1MixedFns {
    Av1MixedFns {
        put: [
            put_mixed::<4, Avx2Wide>,
            put_mixed::<8, Avx2Wide>,
            put_mixed::<16, Avx2Wide>,
            put_mixed::<32, Avx2Wide>,
            put_mixed::<64, Avx2Wide>,
        ],
        prep: [
            prep_mixed::<4, Avx2Wide>,
            prep_mixed::<8, Avx2Wide>,
            prep_mixed::<16, Avx2Wide>,
            prep_mixed::<32, Avx2Wide>,
            prep_mixed::<64, Avx2Wide>,
        ],
    }
}

/// `P` tap pairs of a filter from tap `first`, halved for `R::HALVED` (a
/// halved tap is even, so halving is a shift): `(t[2j], t[2j+1])`.
#[inline]
pub(super) fn tap_pairs<R: Rounding, const P: usize>(
    taps: &[i16],
    first: usize,
) -> [(i16, i16); P] {
    let shift = u32::from(R::HALVED);
    core::array::from_fn(|j| {
        (
            taps[first + 2 * j] >> shift,
            taps[first + 2 * j + 1] >> shift,
        )
    })
}

/// Tap pairs as 16-bit pairs in every 32-bit lane (`vpmaddwd`).
#[target_feature(enable = "avx2")]
#[inline]
fn word_pairs<const P: usize>(pairs: [(i16, i16); P]) -> [__m256i; P] {
    pairs.map(|(lo, hi)| {
        _mm256_set1_epi32((u32::from(lo as u16) | u32::from(hi as u16) << 16) as i32)
    })
}

/// `vpmulhrsw` by `m`: `(x + m') >> k` for `m = 2^(15 - k)`.
#[target_feature(enable = "avx2")]
#[inline]
fn round(x: __m256i, m: i16) -> __m256i {
    _mm256_mulhrs_epi16(x, _mm256_set1_epi16(m))
}

/// The vertical pass of a 2-D filter over `P` interleaved row pairs (low
/// and high halves of each lane's outputs), rounded for `put` or `prep`:
/// per lane, outputs 0..8 as 16-bit words.
#[target_feature(enable = "avx2")]
#[inline]
fn vertical_words<R: Rounding, const P: usize>(
    low: &[__m256i; P],
    high: &[__m256i; P],
    taps: &[__m256i; P],
    prep: bool,
) -> __m256i {
    let round = if prep { PREP_LAST_ROUND } else { R::LAST_ROUND };
    let mut lo = _mm256_set1_epi32(round);
    let mut hi = lo;
    for j in 0..P {
        lo = _mm256_add_epi32(lo, _mm256_madd_epi16(low[j], taps[j]));
        hi = _mm256_add_epi32(hi, _mm256_madd_epi16(high[j], taps[j]));
    }
    if prep {
        let shift = _mm_cvtsi32_si128(PREP_LAST_SHIFT);
        _mm256_packs_epi32(_mm256_sra_epi32(lo, shift), _mm256_sra_epi32(hi, shift))
    } else {
        let shift = _mm_cvtsi32_si128(R::LAST_SHIFT);
        _mm256_packs_epi32(_mm256_sra_epi32(lo, shift), _mm256_sra_epi32(hi, shift))
    }
}

/// Store the low `N` bytes of each lane of `bytes` to rows `dst` and
/// `dst + stride`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_two<const N: usize>(dst: *mut u8, stride: usize, bytes: __m256i) {
    let (a, b) = (
        _mm256_castsi256_si128(bytes),
        _mm256_extracti128_si256::<1>(bytes),
    );
    // SAFETY: the caller provides N writable bytes in both rows.
    unsafe {
        if N == 4 {
            dst.cast::<i32>().write_unaligned(_mm_cvtsi128_si32(a));
            dst.add(stride)
                .cast::<i32>()
                .write_unaligned(_mm_cvtsi128_si32(b));
        } else {
            _mm_storel_epi64(dst.cast(), a);
            _mm_storel_epi64(dst.add(stride).cast(), b);
        }
    }
}

/// Store 16 outputs packed per lane (bytes 0..8 of each lane).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_16(dst: *mut u8, bytes: __m256i) {
    let row = core::arch::x86_64::_mm_unpacklo_epi64(
        _mm256_castsi256_si128(bytes),
        _mm256_extracti128_si256::<1>(bytes),
    );
    // SAFETY: the caller provides 16 writable bytes.
    unsafe { _mm_storeu_si128(dst.cast(), row) };
}

/// Load `N` pixels of a row (4, 8 or 16) into the low bytes of a vector.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_pixels<const N: usize>(src: *const u8) -> __m128i {
    // SAFETY: the caller provides N readable bytes.
    unsafe {
        match N {
            4 => _mm_cvtsi32_si128(src.cast::<i32>().read_unaligned()),
            8 => _mm_loadl_epi64(src.cast()),
            _ => _mm_loadu_si128(src.cast()),
        }
    }
}

/// A raw window: its first used sample and row stride.
#[derive(Clone, Copy)]
pub(super) struct Window {
    src: *const u8,
    stride: usize,
}

impl Window {
    /// The sample `col` columns in on row `row`.
    ///
    /// # Safety
    ///
    /// The sample must lie inside the window the caller checked.
    #[inline]
    pub(super) unsafe fn at(self, row: usize, col: usize) -> *const u8 {
        // SAFETY: forwarded from the caller.
        unsafe { self.src.add(row * self.stride + col) }
    }
}

/// AV1 prediction of a `W`x`h` block (module and `mc` contract).
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn put<F: Av1Family, const W: usize, X: WidePasses>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    let taps = || (F::taps(W, frac_x), F::taps(h, frac_y));
    // SAFETY: AVX2 is enabled; the contract is the caller's.
    unsafe {
        put_with::<W, F, X>(src, src_stride, dst, dst_stride, h, (frac_x, frac_y), taps);
    }
}

/// [`put`] with the caller's taps per direction (`mc::Av1MixedFn`).
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn put_mixed<const W: usize, X: WidePasses>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
    fracs: (usize, usize),
    taps: TapPair<'_>,
) {
    // SAFETY: AVX2 is enabled; the contract is the caller's.
    unsafe { put_with::<W, AnyTaps, X>(src, src_stride, dst, dst_stride, h, fracs, || taps) };
}

/// [`put`] with the taps `taps` gives (asked only when the block is
/// filtered), whose outer taps are known zero when `Z` says so.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn put_with<'t, const W: usize, Z: OuterTaps, X: WidePasses>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
    (frac_x, frac_y): (usize, usize),
    taps: impl FnOnce() -> TapPair<'t>,
) {
    assert!(
        h > 0
            && h.is_multiple_of(2)
            && src.len() >= (h + TAPS - 2) * src_stride + W + TAPS - 1
            && dst.len() >= (h - 1) * dst_stride + W,
        "AV1 prediction window or destination too small"
    );
    let window = Window {
        src: src.as_ptr(),
        stride: src_stride,
    };
    let out = PutOut {
        dst: dst.as_mut_ptr(),
        stride: dst_stride,
    };
    if (frac_x, frac_y) == (0, 0) {
        // SAFETY: source/output footprints were checked above; AVX2 is enabled.
        unsafe {
            crate::block_copy::avx2::copy::<false>(
                &src[TAP_CENTRE * (src_stride + 1)..],
                src_stride,
                dst,
                dst_stride,
                W,
                h,
            );
        };
        return;
    }
    // SAFETY: AVX2 is enabled; the window covers h + 7 rows of W + 7
    // samples and the destination h rows of W (checked above). Each path
    // reads rows lead..lead + h + 2P - 1 and columns lead..lead + W + 2P
    // - 1 of the window (or the centre column/row of a one-direction
    // filter), and writes rows 0..h, columns 0..W. Taps with zero outer
    // taps use 3 pairs from tap 1, others 4 from tap 0.
    unsafe {
        if Z::OUTER_ZERO {
            put_taps::<X, W, 3, _, Av1Rounding, TAP_CENTRE>(
                window,
                1,
                out,
                h,
                (frac_x, frac_y),
                av1_taps(taps()),
            );
        } else {
            put_taps::<X, W, 4, _, Av1Rounding, TAP_CENTRE>(
                window,
                0,
                out,
                h,
                (frac_x, frac_y),
                av1_taps(taps()),
            );
        }
    }
}

/// The AV1 taps of both directions as slices ([`put_taps`] takes any
/// filter length).
#[inline]
fn av1_taps((h_taps, v_taps): TapPair<'_>) -> (&[i16], &[i16]) {
    (h_taps, v_taps)
}

/// The HEVC/VVC separable prediction (`T`'s `N`-tap table, `N` 8 or 4) of
/// a `W`x`h` block at pixel precision, on the same passes as AV1's `put`
/// (`SeparableRounding`): the `ConvolveFn` contract with an `N`-tap
/// window (`h + N - 1` rows of `W + N - 1` samples, the block at row and
/// column `N / 2 - 1`).
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn separable<
    T: SeparableTaps<N>,
    const N: usize,
    const W: usize,
    X: WidePasses,
>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    assert!(
        (N == TAPS || N == SHORT_TAPS)
            && h > 0
            && h.is_multiple_of(2)
            && src.len() >= (h + N - 2) * src_stride + W + N - 1
            && dst.len() >= (h - 1) * dst_stride + W,
        "separable prediction window or destination too small"
    );
    let window = Window {
        src: src.as_ptr(),
        stride: src_stride,
    };
    let out = PutOut {
        dst: dst.as_mut_ptr(),
        stride: dst_stride,
    };
    let taps = (&T::TABLE[frac_x][..], &T::TABLE[frac_y][..]);
    // SAFETY: AVX2 is enabled; the window covers h + N - 1 rows of W + N -
    // 1 samples and the destination h rows of W (checked above); every
    // path reads rows and columns 0..h + N - 1 / 0..W + N - 1 of the
    // window (the centre ones for one direction) and writes the block.
    unsafe {
        if (frac_x, frac_y) == (0, 0) {
            // SAFETY: checked footprint; the centre depends on the tap count.
            crate::block_copy::avx2::copy::<false>(
                &src[(N / 2 - 1) * (src_stride + 1)..],
                src_stride,
                dst,
                dst_stride,
                W,
                h,
            );
        } else if N == TAPS {
            put_taps::<X, W, 4, _, SeparableRounding, TAP_CENTRE>(
                window,
                0,
                out,
                h,
                (frac_x, frac_y),
                taps,
            );
        } else {
            put_taps::<X, W, 2, _, SeparableRounding, SHORT_CENTRE>(
                window,
                0,
                out,
                h,
                (frac_x, frac_y),
                taps,
            );
        }
    }
}

/// AV1 prediction of a `W`x`h` block at compound precision (`prep`, the
/// module table's right column; module and `mc` contract).
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn prep<F: Av1Family, const W: usize, X: WidePasses>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    let taps = || (F::taps(W, frac_x), F::taps(h, frac_y));
    // SAFETY: AVX2 is enabled; the contract is the caller's.
    unsafe {
        prep_with::<W, F, X>(src, src_stride, dst, dst_stride, h, (frac_x, frac_y), taps);
    }
}

/// [`prep`] with the caller's taps per direction (`mc::Av1MixedFn`).
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn prep_mixed<const W: usize, X: WidePasses>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    h: usize,
    fracs: (usize, usize),
    taps: TapPair<'_>,
) {
    // SAFETY: AVX2 is enabled; the contract is the caller's.
    unsafe { prep_with::<W, AnyTaps, X>(src, src_stride, dst, dst_stride, h, fracs, || taps) };
}

/// [`prep`] with taps `(h_taps, v_taps)`, as [`put_with`].
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn prep_with<'t, const W: usize, Z: OuterTaps, X: WidePasses>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    h: usize,
    (frac_x, frac_y): (usize, usize),
    taps: impl FnOnce() -> TapPair<'t>,
) {
    assert!(
        h > 0
            && h.is_multiple_of(2)
            && src.len() >= (h + TAPS - 2) * src_stride + W + TAPS - 1
            && dst.len() >= (h - 1) * dst_stride + W,
        "AV1 prediction window or destination too small"
    );
    let window = Window {
        src: src.as_ptr(),
        stride: src_stride,
    };
    let out = PrepOut {
        dst: dst.as_mut_ptr(),
        stride: dst_stride,
    };
    // SAFETY: as in `put_with`, with `i16` outputs.
    unsafe {
        if (frac_x, frac_y) == (0, 0) {
            crate::block_copy::avx2::prep_unbiased(
                &src[TAP_CENTRE * (src_stride + 1)..],
                src_stride,
                dst,
                dst_stride,
                (W, h),
            );
        } else if Z::OUTER_ZERO {
            put_taps::<X, W, 3, _, Av1Rounding, TAP_CENTRE>(
                window,
                1,
                out,
                h,
                (frac_x, frac_y),
                av1_taps(taps()),
            );
        } else {
            put_taps::<X, W, 4, _, Av1Rounding, TAP_CENTRE>(
                window,
                0,
                out,
                h,
                (frac_x, frac_y),
                av1_taps(taps()),
            );
        }
    }
}

/// A vector of one row's consecutive 16-bit results: 16 (`__m256i`, here)
/// or 32 (`__m512i`, `av1_avx512`).
pub(super) trait Words: Copy {
    /// Store the results clipped to bytes.
    ///
    /// # Safety
    ///
    /// `dst` must have one writable byte per result; the CPU must support
    /// the vector's ISA.
    unsafe fn store_bytes(self, dst: *mut u8);
    /// Store the results as they are.
    ///
    /// # Safety
    ///
    /// As [`Words::store_bytes`], one word per result.
    unsafe fn store_words(self, dst: *mut i16);
}

impl Words for __m256i {
    #[inline]
    unsafe fn store_bytes(self, dst: *mut u8) {
        // SAFETY: forwarded from the caller (AVX2 enabled, 16 bytes).
        unsafe { store_16(dst, _mm256_packus_epi16(self, self)) };
    }
    #[inline]
    unsafe fn store_words(self, dst: *mut i16) {
        // SAFETY: forwarded from the caller (AVX2 enabled, 16 words).
        unsafe { _mm256_storeu_si256(dst.cast(), self) };
    }
}

/// Where a pass's 16-bit results go: final 8-bit samples (`put`) or
/// compound-precision samples (`prep`).
pub(super) trait Sink: Copy {
    /// Compound precision: the `prep` roundings of the module table.
    const PREP: bool;
    /// A vector of consecutive results of row `r` from column `c`.
    ///
    /// # Safety
    ///
    /// The outputs must lie inside the destination the caller checked.
    unsafe fn wide<V: Words>(self, r: usize, c: usize, words: V);
    /// `N` (4 or 8) results of rows `r` (low lane) and `r + 1` (high lane).
    ///
    /// # Safety
    ///
    /// Both rows' `N` outputs must lie inside the checked destination.
    unsafe fn two<const N: usize>(self, r: usize, words: __m256i);
}

/// `put`: words clipped to bytes.
#[derive(Clone, Copy)]
pub(super) struct PutOut {
    dst: *mut u8,
    stride: usize,
}

impl Sink for PutOut {
    const PREP: bool = false;
    #[inline]
    unsafe fn wide<V: Words>(self, r: usize, c: usize, words: V) {
        // SAFETY: the vector's ISA is enabled in every caller; the outputs
        // are the caller's.
        unsafe { words.store_bytes(self.dst.add(r * self.stride + c)) };
    }
    #[inline]
    unsafe fn two<const N: usize>(self, r: usize, words: __m256i) {
        // SAFETY: as for `wide`.
        unsafe {
            store_two::<N>(
                self.dst.add(r * self.stride),
                self.stride,
                _mm256_packus_epi16(words, words),
            );
        }
    }
}

/// `prep`: words stored as they are.
#[derive(Clone, Copy)]
pub(super) struct PrepOut {
    dst: *mut i16,
    stride: usize,
}

impl Sink for PrepOut {
    const PREP: bool = true;
    #[inline]
    unsafe fn wide<V: Words>(self, r: usize, c: usize, words: V) {
        // SAFETY: as for `PutOut::wide`, in words.
        unsafe { words.store_words(self.dst.add(r * self.stride + c)) };
    }
    #[inline]
    unsafe fn two<const N: usize>(self, r: usize, words: __m256i) {
        // SAFETY: as for `wide`: N outputs of rows r and r + 1.
        unsafe {
            let (a, b) = (
                _mm256_castsi256_si128(words),
                _mm256_extracti128_si256::<1>(words),
            );
            let (first, second) = (
                self.dst.add(r * self.stride),
                self.dst.add((r + 1) * self.stride),
            );
            if N == 4 {
                _mm_storel_epi64(first.cast(), a);
                _mm_storel_epi64(second.cast(), b);
            } else {
                _mm_storeu_si128(first.cast(), a);
                _mm_storeu_si128(second.cast(), b);
            }
        }
    }
}

/// [`put`] with `P` tap pairs, the first tap `lead` samples into the
/// window.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn put_taps<
    X: WidePasses,
    const W: usize,
    const P: usize,
    S: Sink,
    R: Rounding,
    const C: usize,
>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    (frac_x, frac_y): (usize, usize),
    (h_taps, v_taps): (&[i16], &[i16]),
) {
    // SAFETY: each path keeps to the window and destination `put` checked.
    unsafe {
        match (frac_x, frac_y, W <= HALF) {
            (_, 0, true) => horizontal_narrow::<W, P, S, R, C>(window, lead, out, h, h_taps),
            (_, 0, false) => X::horizontal::<W, P, S, R, C>(window, lead, out, h, h_taps),
            (0, _, true) => vertical_narrow::<W, P, S, R, C>(window, lead, out, h, v_taps),
            (0, _, false) => X::vertical::<W, P, S, R, C>(window, lead, out, h, v_taps),
            _ => X::both::<W, P, S, R, C>(window, lead, out, h, (h_taps, v_taps)),
        }
    }
}

/// Horizontal only, 4 or 8 wide: two rows per vector.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_narrow<
    const W: usize,
    const P: usize,
    S: Sink,
    R: Rounding,
    const C: usize,
>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    taps: &[i16],
) {
    let taps = byte_pairs::<P>(tap_pairs::<R, P>(taps, lead));
    for r in (0..h).step_by(2) {
        // SAFETY: rows r + 3 and r + 4 from column `lead` (see `put`).
        unsafe {
            let sum = two_rows::<P, W>(window.at(r + C, lead), window.at(r + 1 + C, lead), &taps);
            out.two::<W>(r, horizontal::<R, S>(sum));
        }
    }
}

/// Horizontal only, 16 or more wide: a row of 16 outputs per vector.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_wide<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    taps: &[i16],
) {
    let taps = byte_pairs::<P>(tap_pairs::<R, P>(taps, lead));
    for r in 0..h {
        for c0 in (0..W).step_by(LANES) {
            // SAFETY: row r + 3, columns c0 + lead.. (see `put`).
            unsafe {
                let sum = row_16::<P>(window.at(r + C, c0 + lead), &taps);
                out.wide(r, c0, horizontal::<R, S>(sum));
            }
        }
    }
}

/// Vertical only, 4 or 8 wide: output rows `r` (low lane) and `r + 1`
/// (high lane) from byte-interleaved row pairs.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vertical_narrow<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    taps: &[i16],
) {
    let taps = byte_pairs::<P>(tap_pairs::<R, P>(taps, lead));
    // SAFETY: rows lead.. at the centre column (see `put`).
    let load = |k: usize| unsafe { load_pixels::<W>(window.at(lead + k, C)) };
    // Each vector pairs the rows for two successive outputs. Keep that
    // interleave when it becomes the next group's earlier tap pair.
    let pair = |a, b, c| {
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_unpacklo_epi8(a, b)),
            _mm_unpacklo_epi8(b, c),
        )
    };
    let mut previous = load(0);
    let mut pairs = [_mm256_set1_epi16(0); P];
    for (j, dst) in pairs.iter_mut().enumerate().take(P - 1) {
        let middle = load(2 * j + 1);
        let next = load(2 * j + 2);
        *dst = pair(previous, middle, next);
        previous = next;
    }
    for r in (0..h).step_by(2) {
        let middle = load(r + 2 * P - 1);
        let next = load(r + 2 * P);
        pairs[P - 1] = pair(previous, middle, next);
        let mut sum = _mm256_set1_epi16(0);
        for (pixels, weight) in pairs.iter().zip(&taps) {
            sum = _mm256_add_epi16(sum, _mm256_maddubs_epi16(*pixels, *weight));
        }
        let words = round(sum, if S::PREP { ROUND_2 } else { ROUND_6 });
        // SAFETY: output rows r and r + 1 (see `put`).
        unsafe { out.two::<W>(r, words) };
        slide(&mut pairs, 1, P - 1);
        previous = next;
    }
}

/// Vertical only, 16 or more wide: a row of 16 outputs per vector, from
/// byte-interleaved row pairs (low lane: outputs 0..8, high: 8..16).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vertical_wide<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    taps: &[i16],
) {
    let taps = byte_pairs::<P>(tap_pairs::<R, P>(taps, lead));
    for c0 in (0..W).step_by(LANES) {
        // SAFETY: the entry point checked these rows and the output strip.
        unsafe {
            super::byte_column_pairs::<P, _, _>(
                h,
                _mm256_set1_epi16(0),
                |k| load_pixels::<LANES>(window.at(lead + k, c0 + C)),
                |a, b| {
                    _mm256_inserti128_si256::<1>(
                        _mm256_castsi128_si256(_mm_unpacklo_epi8(a, b)),
                        _mm_unpackhi_epi8(a, b),
                    )
                },
                |r, pairs| {
                    let mut sum = _mm256_set1_epi16(0);
                    for (j, weight) in taps.iter().enumerate() {
                        sum = _mm256_add_epi16(sum, _mm256_maddubs_epi16(pairs[2 * j], *weight));
                    }
                    let words = round(sum, if S::PREP { ROUND_2 } else { ROUND_6 });
                    out.wide(r, c0, words);
                },
            );
        }
    }
}

/// Both directions, 4 or 8 wide. Vector `k` holds the rounded horizontal
/// rows `k` (low lane) and `k + 1` (high lane); the even vectors are
/// filtered, the odd ones assembled from their neighbours' lanes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn both_narrow<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    (h_taps, v_taps): (&[i16], &[i16]),
) {
    let h_taps = byte_pairs::<P>(tap_pairs::<R, P>(h_taps, lead));
    let v_taps = word_pairs::<P>(tap_pairs::<R, P>(v_taps, lead));
    // The window's last row the vertical taps reach.
    let last = h + 2 * P - 2;
    // Rows k and k + 1 (k + 1 clamped to the window: the last step's high
    // lane is never used).
    let pair = |k: usize| {
        // SAFETY: rows lead + k and lead + min(k + 1, last) from column
        // `lead` (see `put`).
        let sum = unsafe {
            two_rows::<P, W>(
                window.at(lead + k, lead),
                window.at(lead + (k + 1).min(last), lead),
                &h_taps,
            )
        };
        first::<R>(sum)
    };
    // even[j] = vector r + 2j; `next` = vector r + 2P.
    let mut even: [__m256i; P] = core::array::from_fn(|j| pair(2 * j));
    for r in (0..h).step_by(2) {
        let next = pair(r + 2 * P);
        let odd: [__m256i; P] = core::array::from_fn(|j| {
            let following = if j + 1 < P { even[j + 1] } else { next };
            _mm256_permute2x128_si256::<0x21>(even[j], following)
        });
        let low = core::array::from_fn(|j| _mm256_unpacklo_epi16(even[j], odd[j]));
        let high = core::array::from_fn(|j| _mm256_unpackhi_epi16(even[j], odd[j]));
        let words = vertical_words::<R, P>(&low, &high, &v_taps, S::PREP);
        // SAFETY: output rows r and r + 1 (see `put`).
        unsafe { out.two::<W>(r, words) };
        even = core::array::from_fn(|j| if j + 1 < P { even[j + 1] } else { next });
    }
}

/// Both directions, 16 or more wide: rounded horizontal rows of 16
/// outputs, interleaved in pairs once each into an even window (rows r,
/// r+1; r+2, r+3; ...) and an odd one (r+1, r+2; ...), giving two output
/// rows per step.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn both_wide<const W: usize, const P: usize, S: Sink, R: Rounding, const C: usize>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    (h_taps, v_taps): (&[i16], &[i16]),
) {
    let h_taps = byte_pairs::<P>(tap_pairs::<R, P>(h_taps, lead));
    let v_taps = word_pairs::<P>(tap_pairs::<R, P>(v_taps, lead));
    for c0 in (0..W).step_by(LANES) {
        // SAFETY: row lead + k, columns c0 + lead.. (see `put`).
        let row = |k: usize| {
            // SAFETY: row lead + k, columns c0 + lead.. (see `put`).
            first::<R>(unsafe { row_16::<P>(window.at(lead + k, c0 + lead), &h_taps) })
        };
        let interleave =
            |a: __m256i, b: __m256i| (_mm256_unpacklo_epi16(a, b), _mm256_unpackhi_epi16(a, b));
        // Rows 0..2P-1, interleaved: even pairs (0,1), (2,3), ...; odd
        // pairs (1,2), (3,4), ...; the last of each is made in the loop.
        let zero = _mm256_set1_epi16(0);
        let mut rows = [zero; 8];
        for (k, slot) in rows.iter_mut().enumerate().take(2 * P - 1) {
            *slot = row(k);
        }
        let mut even: [(__m256i, __m256i); P] = [(zero, zero); P];
        let mut odd = even;
        for j in 0..P - 1 {
            even[j] = interleave(rows[2 * j], rows[2 * j + 1]);
            odd[j] = interleave(rows[2 * j + 1], rows[2 * j + 2]);
        }
        let mut previous = rows[2 * P - 2];
        for r in (0..h).step_by(2) {
            let (a, b) = (row(r + 2 * P - 1), row(r + 2 * P));
            even[P - 1] = interleave(previous, a);
            odd[P - 1] = interleave(a, b);
            previous = b;
            for (k, set) in [&even, &odd].into_iter().enumerate() {
                let low = core::array::from_fn(|j| set[j].0);
                let high = core::array::from_fn(|j| set[j].1);
                let words = vertical_words::<R, P>(&low, &high, &v_taps, S::PREP);
                // SAFETY: output row r + k, columns c0..c0 + 16 (see `put`).
                unsafe { out.wide(r + k, c0, words) };
            }
            even = core::array::from_fn(|j| if j + 1 < P { even[j + 1] } else { even[j] });
            odd = core::array::from_fn(|j| if j + 1 < P { odd[j + 1] } else { odd[j] });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        HEVC_CHROMA, HEVC_LUMA, REGULAR_4, REGULAR_8, SHARP_8, SMOOTH_4, SMOOTH_8, VVC_AFFINE_LUMA,
        VVC_CHROMA, VVC_LUMA,
    };

    /// Every HEVC/VVC tap fits a signed byte, and every byte pair's and
    /// every whole filter's sums over 8-bit samples fit i16 (the exactness
    /// of `SeparableRounding`'s unhalved `vpmaddubsw` sums).
    #[test]
    fn separable_sums_fit_i16() {
        let tables: [&[i16]; 5] = [
            HEVC_LUMA.as_flattened(),
            VVC_LUMA.as_flattened(),
            VVC_AFFINE_LUMA.as_flattened(),
            HEVC_CHROMA.as_flattened(),
            VVC_CHROMA.as_flattened(),
        ];
        for (table, n) in tables.into_iter().zip([8, 8, 8, 4, 4]) {
            for taps in table.chunks(n) {
                assert!(taps.iter().all(|&t| i8::try_from(t).is_ok()));
                let fits = |set: &[i16]| {
                    let pos: i32 = set.iter().filter(|&&t| t > 0).map(|&t| i32::from(t)).sum();
                    let neg: i32 = set.iter().filter(|&&t| t < 0).map(|&t| i32::from(t)).sum();
                    pos * 255 <= i32::from(i16::MAX) && neg * 255 >= i32::from(i16::MIN)
                };
                assert!(taps.chunks(2).all(fits) && fits(taps));
            }
        }
    }

    #[test]
    fn halved_sums_fit_i16() {
        for table in [&REGULAR_4, &REGULAR_8, &SHARP_8, &SMOOTH_4, &SMOOTH_8] {
            for taps in table {
                let half: Vec<i32> = taps.iter().map(|&t| i32::from(t / 2)).collect();
                assert!(taps.iter().all(|t| t % 2 == 0));
                for pair in half.chunks(2).chain(half[1..7].chunks(2)) {
                    let pos: i32 = pair.iter().filter(|&&t| t > 0).sum();
                    let neg: i32 = pair.iter().filter(|&&t| t < 0).sum();
                    assert!(pos * 255 <= i32::from(i16::MAX) && neg * 255 >= i32::from(i16::MIN));
                }
                let pos: i32 = half.iter().filter(|&&t| t > 0).sum();
                let neg: i32 = half.iter().filter(|&&t| t < 0).sum();
                assert!(pos * 255 <= i32::from(i16::MAX) && neg * 255 >= i32::from(i16::MIN));
            }
        }
    }
}
