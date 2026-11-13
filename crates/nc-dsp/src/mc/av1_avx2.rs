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

use super::{
    AnyTaps, Av1Family, Av1MixedFns, ConvolveFn, OuterTaps, PrepFn, TAP_CENTRE, TAPS, TapPair,
    slide,
};
use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_or_si128, _mm_slli_si128, _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi8,
    _mm_unpacklo_epi8, _mm256_add_epi16, _mm256_add_epi32, _mm256_castsi128_si256,
    _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_extracti128_si256,
    _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_mulhrs_epi16, _mm256_packs_epi32, _mm256_packus_epi16, _mm256_permute2x128_si256,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_shuffle_epi8, _mm256_slli_epi16,
    _mm256_srai_epi32, _mm256_storeu_si256, _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
};

/// Outputs of one vector (16 lanes of 16 bits).
const LANES: usize = 16;
/// Outputs of one 128-bit lane.
const HALF: usize = 8;
/// `vpmulhrsw` multipliers: `(x + 2) >> 2`, `(x + 8) >> 4`, `(x + 32) >> 6`.
const ROUND_2: i16 = 1 << 13;
const ROUND_4: i16 = 1 << 11;
const ROUND_6: i16 = 1 << 9;
/// The vertical pass of a 2-D filter: `(acc + 512) >> 10` for `put`,
/// `(acc + 32) >> 6` for `prep` (`InterRound1` 11 and 7, less the halving).
const LAST_ROUND: i32 = 1 << 9;
const LAST_SHIFT: i32 = 10;
const PREP_LAST_ROUND: i32 = 1 << 5;
const PREP_LAST_SHIFT: i32 = 6;
/// `prep` scales whole samples by 16 (`<< 4`).
const PREP_COPY_SHIFT: i32 = 4;

/// The AV1 compound-precision kernels of filter family `F` by width class.
pub(super) fn prep_family<F: Av1Family>() -> [PrepFn; 5] {
    [
        prep::<F, 4>,
        prep::<F, 8>,
        prep::<F, 16>,
        prep::<F, 32>,
        prep::<F, 64>,
    ]
}

/// The AV1 kernels of filter family `F` by width class (4 to 64).
pub(super) fn family<F: Av1Family>() -> [ConvolveFn; 5] {
    [
        put::<F, 4>,
        put::<F, 8>,
        put::<F, 16>,
        put::<F, 32>,
        put::<F, 64>,
    ]
}

/// The mixed-family kernels by width class (4 to 64).
pub(super) fn mixed_fns() -> Av1MixedFns {
    Av1MixedFns {
        put: [
            put_mixed::<4>,
            put_mixed::<8>,
            put_mixed::<16>,
            put_mixed::<32>,
            put_mixed::<64>,
        ],
        prep: [
            prep_mixed::<4>,
            prep_mixed::<8>,
            prep_mixed::<16>,
            prep_mixed::<32>,
            prep_mixed::<64>,
        ],
    }
}

/// `P` halved tap pairs of a filter from tap `first` (a tap is even, so
/// halving is a shift): `(t[2j], t[2j+1])`.
#[inline]
fn halved_pairs<const P: usize>(taps: &[i16; TAPS], first: usize) -> [(i16, i16); P] {
    core::array::from_fn(|j| (taps[first + 2 * j] >> 1, taps[first + 2 * j + 1] >> 1))
}

/// Tap pairs as signed byte pairs in every 16-bit lane (`vpmaddubsw`).
#[target_feature(enable = "avx2")]
#[inline]
fn byte_pairs<const P: usize>(pairs: [(i16, i16); P]) -> [__m256i; P] {
    pairs.map(|(lo, hi)| _mm256_set1_epi16(i16::from_le_bytes([lo as i8 as u8, hi as i8 as u8])))
}

/// Tap pairs as 16-bit pairs in every 32-bit lane (`vpmaddwd`).
#[target_feature(enable = "avx2")]
#[inline]
fn word_pairs<const P: usize>(pairs: [(i16, i16); P]) -> [__m256i; P] {
    pairs.map(|(lo, hi)| {
        _mm256_set1_epi32((u32::from(lo as u16) | u32::from(hi as u16) << 16) as i32)
    })
}

/// `vpshufb` masks gathering the pixels of tap pair `p` for 8 outputs per
/// lane: output `o` of a lane takes bytes `o + 2p` and `o + 2p + 1`, from
/// lane byte `high_start` on in the high lane (8 or 16 outputs a row).
const fn masks<const P: usize>(high_offset: usize) -> [[u8; 32]; P] {
    let mut out = [[0; 32]; P];
    let mut p = 0;
    while p < P {
        let mut i = 0;
        while i < 32 {
            let output = i % 16 / 2;
            let shift = if i >= 16 { high_offset } else { 0 };
            out[p][i] = (output + 2 * p + i % 2 + shift) as u8;
            i += 1;
        }
        p += 1;
    }
    out
}

/// Source bytes `n` outputs of a row read with `P` tap pairs.
const fn span(n: usize, pairs: usize) -> usize {
    n + 2 * pairs - 1
}

/// Masks of one 16-output row: the high lane is loaded from byte
/// `span(16, P) - 16`, outputs 8..16 start 8 bytes in, so its masks start
/// `8 - (span - 16)` bytes in.
const MASKS_ROW_6: [[u8; 32]; 3] = masks::<3>(HALF + 16 - span(LANES, 3));
const MASKS_ROW_8: [[u8; 32]; 4] = masks::<4>(HALF + 16 - span(LANES, 4));
/// Masks of two rows of up to 8 outputs, one per lane.
const MASKS_TWO_6: [[u8; 32]; 3] = masks::<3>(0);
const MASKS_TWO_8: [[u8; 32]; 4] = masks::<4>(0);

/// The filter sum of the pixels in `pixels` gathered by `masks`.
#[target_feature(enable = "avx2")]
#[inline]
fn pair_sums<const P: usize>(pixels: __m256i, masks: &[[u8; 32]], taps: &[__m256i; P]) -> __m256i {
    let mut sum = _mm256_set1_epi16(0);
    for (mask, weight) in masks.iter().zip(taps) {
        // SAFETY: `mask` is 32 readable bytes.
        let mask = unsafe { _mm256_loadu_si256(mask.as_ptr().cast()) };
        let product = _mm256_maddubs_epi16(_mm256_shuffle_epi8(pixels, mask), *weight);
        sum = _mm256_add_epi16(sum, product);
    }
    sum
}

/// Unrounded horizontal sums of 16 outputs of the row at `src`, reading
/// exactly `span(16, P)` bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn row_16<const P: usize>(src: *const u8, taps: &[__m256i; P]) -> __m256i {
    // SAFETY: the caller provides span(16, P) readable bytes; both 16-byte
    // loads lie inside them.
    let pixels = unsafe {
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(src.cast())),
            _mm_loadu_si128(src.add(span(LANES, P) - 16).cast()),
        )
    };
    let masks = if P == 3 {
        &MASKS_ROW_6[..]
    } else {
        &MASKS_ROW_8[..]
    };
    pair_sums(pixels, masks, taps)
}

/// The `span(N, P)` bytes (8 to 15) of a row at `src`, from two
/// overlapping 8-byte loads.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn short_row<const P: usize, const N: usize>(src: *const u8) -> __m128i {
    let bytes = span(N, P);
    // SAFETY: the caller provides `bytes` (at least 8) readable bytes.
    unsafe {
        let low = _mm_loadl_epi64(src.cast());
        let high = _mm_loadl_epi64(src.add(bytes - 8).cast());
        let high = match bytes {
            9 => _mm_slli_si128::<1>(high),
            11 => _mm_slli_si128::<3>(high),
            13 => _mm_slli_si128::<5>(high),
            _ => _mm_slli_si128::<7>(high),
        };
        _mm_or_si128(low, high)
    }
}

/// Unrounded horizontal sums of `N` (4 or 8) outputs of two rows, `a` in
/// the low lane and `b` in the high one.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn two_rows<const P: usize, const N: usize>(
    a: *const u8,
    b: *const u8,
    taps: &[__m256i; P],
) -> __m256i {
    // SAFETY: forwarded from the caller.
    let pixels = unsafe {
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(short_row::<P, N>(a)),
            short_row::<P, N>(b),
        )
    };
    let masks = if P == 3 {
        &MASKS_TWO_6[..]
    } else {
        &MASKS_TWO_8[..]
    };
    pair_sums(pixels, masks, taps)
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
fn vertical_words<const P: usize>(
    low: &[__m256i; P],
    high: &[__m256i; P],
    taps: &[__m256i; P],
    prep: bool,
) -> __m256i {
    let round = if prep { PREP_LAST_ROUND } else { LAST_ROUND };
    let mut lo = _mm256_set1_epi32(round);
    let mut hi = lo;
    for j in 0..P {
        lo = _mm256_add_epi32(lo, _mm256_madd_epi16(low[j], taps[j]));
        hi = _mm256_add_epi32(hi, _mm256_madd_epi16(high[j], taps[j]));
    }
    if prep {
        _mm256_packs_epi32(
            _mm256_srai_epi32::<PREP_LAST_SHIFT>(lo),
            _mm256_srai_epi32::<PREP_LAST_SHIFT>(hi),
        )
    } else {
        _mm256_packs_epi32(
            _mm256_srai_epi32::<LAST_SHIFT>(lo),
            _mm256_srai_epi32::<LAST_SHIFT>(hi),
        )
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
struct Window {
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
    unsafe fn at(self, row: usize, col: usize) -> *const u8 {
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
pub(super) unsafe fn put<F: Av1Family, const W: usize>(
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
        put_with::<W, F>(src, src_stride, dst, dst_stride, h, (frac_x, frac_y), taps);
    }
}

/// [`put`] with the caller's taps per direction (`mc::Av1MixedFn`).
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn put_mixed<const W: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
    fracs: (usize, usize),
    taps: TapPair<'_>,
) {
    // SAFETY: AVX2 is enabled; the contract is the caller's.
    unsafe { put_with::<W, AnyTaps>(src, src_stride, dst, dst_stride, h, fracs, || taps) };
}

/// [`put`] with the taps `taps` gives (asked only when the block is
/// filtered), whose outer taps are known zero when `Z` says so.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn put_with<'t, const W: usize, Z: OuterTaps>(
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
        for r in 0..h {
            // SAFETY: row r + 3 of the window from column 3 (W samples) and
            // row r of the destination (W samples) lie inside what was
            // checked above; the two buffers do not overlap (`&` and
            // `&mut`).
            unsafe {
                core::ptr::copy_nonoverlapping(
                    window.at(r + TAP_CENTRE, TAP_CENTRE),
                    out.dst.add(r * dst_stride),
                    W,
                );
            }
        }
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
            put_taps::<W, 3, _>(window, 1, out, h, (frac_x, frac_y), taps());
        } else {
            put_taps::<W, 4, _>(window, 0, out, h, (frac_x, frac_y), taps());
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
pub(super) unsafe fn prep<F: Av1Family, const W: usize>(
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
        prep_with::<W, F>(src, src_stride, dst, dst_stride, h, (frac_x, frac_y), taps);
    }
}

/// [`prep`] with the caller's taps per direction (`mc::Av1MixedFn`).
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn prep_mixed<const W: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    h: usize,
    fracs: (usize, usize),
    taps: TapPair<'_>,
) {
    // SAFETY: AVX2 is enabled; the contract is the caller's.
    unsafe { prep_with::<W, AnyTaps>(src, src_stride, dst, dst_stride, h, fracs, || taps) };
}

/// [`prep`] with taps `(h_taps, v_taps)`, as [`put_with`].
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn prep_with<'t, const W: usize, Z: OuterTaps>(
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
            copy_scaled::<W>(window, out, h);
        } else if Z::OUTER_ZERO {
            put_taps::<W, 3, _>(window, 1, out, h, (frac_x, frac_y), taps());
        } else {
            put_taps::<W, 4, _>(window, 0, out, h, (frac_x, frac_y), taps());
        }
    }
}

/// The `prep` of a whole-sample block: every sample `<< 4`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn copy_scaled<const W: usize>(window: Window, out: PrepOut, h: usize) {
    for r in 0..h {
        for c0 in (0..W).step_by(LANES) {
            // SAFETY: row r + 3 from column c0 + 3 (`W.min(16)` samples) and
            // its outputs lie inside what `prep` checked.
            unsafe {
                let samples = load_pixels::<W>(window.at(r + TAP_CENTRE, c0 + TAP_CENTRE));
                let words = _mm256_slli_epi16::<PREP_COPY_SHIFT>(_mm256_cvtepu8_epi16(samples));
                let at = out.dst.add(r * out.stride + c0);
                match W {
                    4 => _mm_storel_epi64(at.cast(), _mm256_castsi256_si128(words)),
                    8 => _mm_storeu_si128(at.cast(), _mm256_castsi256_si128(words)),
                    _ => _mm256_storeu_si256(at.cast(), words),
                }
            }
        }
    }
}

/// Where a pass's 16-bit results go: final 8-bit samples (`put`) or
/// compound-precision samples (`prep`).
trait Sink: Copy {
    /// Compound precision: the `prep` roundings of the module table.
    const PREP: bool;
    /// 16 results of row `r` from column `c`, as `[0..8 | 8..16]`.
    ///
    /// # Safety
    ///
    /// The 16 outputs must lie inside the destination the caller checked.
    unsafe fn wide(self, r: usize, c: usize, words: __m256i);
    /// `N` (4 or 8) results of rows `r` (low lane) and `r + 1` (high lane).
    ///
    /// # Safety
    ///
    /// Both rows' `N` outputs must lie inside the checked destination.
    unsafe fn two<const N: usize>(self, r: usize, words: __m256i);
}

/// `put`: words clipped to bytes.
#[derive(Clone, Copy)]
struct PutOut {
    dst: *mut u8,
    stride: usize,
}

impl Sink for PutOut {
    const PREP: bool = false;
    #[inline]
    unsafe fn wide(self, r: usize, c: usize, words: __m256i) {
        // SAFETY: AVX2 is enabled in every caller; the outputs are the
        // caller's.
        unsafe {
            store_16(
                self.dst.add(r * self.stride + c),
                _mm256_packus_epi16(words, words),
            );
        }
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
struct PrepOut {
    dst: *mut i16,
    stride: usize,
}

impl Sink for PrepOut {
    const PREP: bool = true;
    #[inline]
    unsafe fn wide(self, r: usize, c: usize, words: __m256i) {
        // SAFETY: AVX2 is enabled in every caller; the 16 outputs (32
        // bytes) are the caller's.
        unsafe { _mm256_storeu_si256(self.dst.add(r * self.stride + c).cast(), words) };
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
unsafe fn put_taps<const W: usize, const P: usize, S: Sink>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    (frac_x, frac_y): (usize, usize),
    (h_taps, v_taps): (&[i16; TAPS], &[i16; TAPS]),
) {
    // SAFETY: each path keeps to the window and destination `put` checked.
    unsafe {
        match (frac_x, frac_y, W <= HALF) {
            (_, 0, true) => horizontal_narrow::<W, P, S>(window, lead, out, h, h_taps),
            (_, 0, false) => horizontal_wide::<W, P, S>(window, lead, out, h, h_taps),
            (0, _, true) => vertical_narrow::<W, P, S>(window, lead, out, h, v_taps),
            (0, _, false) => vertical_wide::<W, P, S>(window, lead, out, h, v_taps),
            (_, _, true) => both_narrow::<W, P, S>(window, lead, out, h, (h_taps, v_taps)),
            (_, _, false) => both_wide::<W, P, S>(window, lead, out, h, (h_taps, v_taps)),
        }
    }
}

/// Horizontal only, 4 or 8 wide: two rows per vector.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_narrow<const W: usize, const P: usize, S: Sink>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    taps: &[i16; TAPS],
) {
    let taps = byte_pairs::<P>(halved_pairs(taps, lead));
    for r in (0..h).step_by(2) {
        // SAFETY: rows r + 3 and r + 4 from column `lead` (see `put`).
        unsafe {
            let sum = two_rows::<P, W>(
                window.at(r + TAP_CENTRE, lead),
                window.at(r + 1 + TAP_CENTRE, lead),
                &taps,
            );
            let t = round(sum, ROUND_2);
            out.two::<W>(r, if S::PREP { t } else { round(t, ROUND_4) });
        }
    }
}

/// Horizontal only, 16 or more wide: a row of 16 outputs per vector.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_wide<const W: usize, const P: usize, S: Sink>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    taps: &[i16; TAPS],
) {
    let taps = byte_pairs::<P>(halved_pairs(taps, lead));
    for r in 0..h {
        for c0 in (0..W).step_by(LANES) {
            // SAFETY: row r + 3, columns c0 + lead.. (see `put`).
            unsafe {
                let sum = row_16::<P>(window.at(r + TAP_CENTRE, c0 + lead), &taps);
                let t = round(sum, ROUND_2);
                out.wide(r, c0, if S::PREP { t } else { round(t, ROUND_4) });
            }
        }
    }
}

/// Vertical only, 4 or 8 wide: output rows `r` (low lane) and `r + 1`
/// (high lane) from byte-interleaved row pairs.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vertical_narrow<const W: usize, const P: usize, S: Sink>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    taps: &[i16; TAPS],
) {
    let taps = byte_pairs::<P>(halved_pairs(taps, lead));
    // SAFETY: rows lead.. at the centre column (see `put`).
    let load = |k: usize| unsafe { load_pixels::<W>(window.at(lead + k, TAP_CENTRE)) };
    // Rows r..r + 2P of the step (2P + 1 rows).
    let mut rows: [__m128i; 9] = [_mm_cvtsi32_si128(0); 9];
    for (k, row) in rows.iter_mut().enumerate().take(2 * P - 1) {
        *row = load(k);
    }
    for r in (0..h).step_by(2) {
        rows[2 * P - 1] = load(r + 2 * P - 1);
        rows[2 * P] = load(r + 2 * P);
        let mut sum = _mm256_set1_epi16(0);
        for (j, weight) in taps.iter().enumerate() {
            let pairs = _mm256_inserti128_si256::<1>(
                _mm256_castsi128_si256(_mm_unpacklo_epi8(rows[2 * j], rows[2 * j + 1])),
                _mm_unpacklo_epi8(rows[2 * j + 1], rows[2 * j + 2]),
            );
            sum = _mm256_add_epi16(sum, _mm256_maddubs_epi16(pairs, *weight));
        }
        let words = round(sum, if S::PREP { ROUND_2 } else { ROUND_6 });
        // SAFETY: output rows r and r + 1 (see `put`).
        unsafe { out.two::<W>(r, words) };
        slide(&mut rows, 2, 2 * P - 1);
    }
}

/// Vertical only, 16 or more wide: a row of 16 outputs per vector, from
/// byte-interleaved row pairs (low lane: outputs 0..8, high: 8..16).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vertical_wide<const W: usize, const P: usize, S: Sink>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    taps: &[i16; TAPS],
) {
    let taps = byte_pairs::<P>(halved_pairs(taps, lead));
    for c0 in (0..W).step_by(LANES) {
        // SAFETY: rows lead.. at columns c0 + 3.. (see `put`).
        let load = |k: usize| unsafe { load_pixels::<LANES>(window.at(lead + k, c0 + TAP_CENTRE)) };
        let mut rows: [__m128i; 8] = [_mm_cvtsi32_si128(0); 8];
        for (k, row) in rows.iter_mut().enumerate().take(2 * P - 1) {
            *row = load(k);
        }
        for r in 0..h {
            rows[2 * P - 1] = load(r + 2 * P - 1);
            let mut sum = _mm256_set1_epi16(0);
            for (j, weight) in taps.iter().enumerate() {
                let (a, b) = (rows[2 * j], rows[2 * j + 1]);
                let pairs = _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm_unpacklo_epi8(a, b)),
                    _mm_unpackhi_epi8(a, b),
                );
                sum = _mm256_add_epi16(sum, _mm256_maddubs_epi16(pairs, *weight));
            }
            // The byte pairs put outputs 0..8 in the low lane and 8..16 in
            // the high one: the order `wide` takes.
            let words = round(sum, if S::PREP { ROUND_2 } else { ROUND_6 });
            // SAFETY: output row r, columns c0..c0 + 16 (see `put`).
            unsafe { out.wide(r, c0, words) };
            slide(&mut rows, 1, 2 * P - 1);
        }
    }
}

/// Both directions, 4 or 8 wide. Vector `k` holds the rounded horizontal
/// rows `k` (low lane) and `k + 1` (high lane); the even vectors are
/// filtered, the odd ones assembled from their neighbours' lanes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn both_narrow<const W: usize, const P: usize, S: Sink>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    (h_taps, v_taps): (&[i16; TAPS], &[i16; TAPS]),
) {
    let h_taps = byte_pairs::<P>(halved_pairs(h_taps, lead));
    let v_taps = word_pairs::<P>(halved_pairs(v_taps, lead));
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
        round(sum, ROUND_2)
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
        let words = vertical_words(&low, &high, &v_taps, S::PREP);
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
unsafe fn both_wide<const W: usize, const P: usize, S: Sink>(
    window: Window,
    lead: usize,
    out: S,
    h: usize,
    (h_taps, v_taps): (&[i16; TAPS], &[i16; TAPS]),
) {
    let h_taps = byte_pairs::<P>(halved_pairs(h_taps, lead));
    let v_taps = word_pairs::<P>(halved_pairs(v_taps, lead));
    for c0 in (0..W).step_by(LANES) {
        // SAFETY: row lead + k, columns c0 + lead.. (see `put`).
        let row = |k: usize| {
            // SAFETY: row lead + k, columns c0 + lead.. (see `put`).
            round(
                unsafe { row_16::<P>(window.at(lead + k, c0 + lead), &h_taps) },
                ROUND_2,
            )
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
                // SAFETY: output row r + k, columns c0..c0 + 16 (see `put`).
                unsafe { out.wide(r + k, c0, vertical_words(&low, &high, &v_taps, S::PREP)) };
            }
            even = core::array::from_fn(|j| if j + 1 < P { even[j + 1] } else { even[j] });
            odd = core::array::from_fn(|j| if j + 1 < P { odd[j + 1] } else { odd[j] });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{REGULAR_4, REGULAR_8, SHARP_8, SMOOTH_4, SMOOTH_8};

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
