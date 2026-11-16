//! WebAssembly SIMD128 AV1 prediction at pixel precision (`put`) and
//! compound precision (`prep`), in the structure of `av1_avx2` (after
//! dav1d's `put_8tap` / `prep_8tap`; BSD-2-Clause,
//! Copyright (c) 2018, `VideoLAN` and dav1d authors; see `THIRD_PARTY.md`).
//!
//! The horizontal pass multiplies widened pixels by halved taps in 16 bits
//! (SIMD128 has no unsigned-by-signed byte product). The vertical pass of a
//! 2-D filter multiplies interleaved 16-bit intermediate row pairs by tap
//! pairs (`i32x4.dot_i16x8_s`, half the multiplies of a per-tap widening
//! product), from an even and an odd window of pairs, so each new row is
//! interleaved once and every step makes two output rows. Power-of-two
//! roundings are one `i16x8.q15mulr_sat_s` (`(x·m + 2^14) >> 15`, which
//! never saturates here). Arithmetic and roundings as in `av1_avx2` and the
//! `mc` module docs: the results equal the scalar reference. One `Sink`
//! per precision does the stores (pixels clipped to bytes, or the 16-bit
//! compound words), so both share the filter paths.
//!
//! Kernel contract: see the `mc` module docs; the window and destination
//! are checked once on entry and the row loops then use raw pointers.

use super::{
    AnyTaps, Av1Family, Av1MixedFns, Av1Rounding, ConvolveFn, OuterTaps, PREP_LAST_ROUND,
    PREP_LAST_SHIFT, PrepFn, ROUND_2, ROUND_6, Rounding, SeparableRounding, SeparableTaps,
    TAP_CENTRE, TAPS, TapPair, slide,
};
use core::arch::wasm32::{
    i8x16_shuffle, i16x8_add, i16x8_mul, i16x8_narrow_i32x4, i16x8_q15mulr_sat, i16x8_shl,
    i16x8_splat, i32x4_add, i32x4_dot_i16x8, i32x4_shr, i32x4_splat, u8x16_narrow_i16x8,
    u16x8_extend_low_u8x16, u16x8_load_extend_u8x8, v128, v128_load, v128_load32_zero,
    v128_load64_zero, v128_store, v128_store32_lane, v128_store64_lane,
};

/// Outputs of one vector (8 lanes of 16 bits).
const LANES: usize = 8;
/// Bytes of one vector.
const VECTOR_BYTES: usize = 16;
/// Taps of HEVC and VVC chroma filters, and the window column of their
/// centre sample.
const SHORT_TAPS: usize = 4;
/// A whole-sample block at compound precision: each sample `<< 4`.
const PREP_COPY_SHIFT: u32 = 4;

/// The AV1 compound-precision kernels of filter family `F` by width class.
pub(super) fn prep_family<F: Av1Family>() -> [PrepFn; 5] {
    [
        prep::<F, 4, 4>,
        prep::<F, 8, LANES>,
        prep::<F, 16, LANES>,
        prep::<F, 32, LANES>,
        prep::<F, 64, LANES>,
    ]
}

/// The AV1 kernels of filter family `F` by width class (4 to 64).
pub(super) fn family<F: Av1Family>() -> [ConvolveFn; 5] {
    [
        put::<F, 4, 4>,
        put::<F, 8, LANES>,
        put::<F, 16, LANES>,
        put::<F, 32, LANES>,
        put::<F, 64, LANES>,
    ]
}

/// The mixed-family kernels (`mc::Av1MixedFn`) by width class (4 to 64).
pub(super) fn mixed_fns() -> Av1MixedFns {
    Av1MixedFns {
        put: [
            put_mixed::<4, 4>,
            put_mixed::<8, LANES>,
            put_mixed::<16, LANES>,
            put_mixed::<32, LANES>,
            put_mixed::<64, LANES>,
        ],
        prep: [
            prep_mixed::<4, 4>,
            prep_mixed::<8, LANES>,
            prep_mixed::<16, LANES>,
            prep_mixed::<32, LANES>,
            prep_mixed::<64, LANES>,
        ],
    }
}

/// The `T` taps of a filter from tap `first`, halved for `R::HALVED` (a
/// halved tap is even, so halving is a shift).
#[inline]
fn taps_of<R: Rounding, const T: usize>(taps: &[i16], first: usize) -> [i16; T] {
    let shift = u32::from(R::HALVED);
    core::array::from_fn(|k| taps[first + k] >> shift)
}

/// [`Rounding::FIRST`] on 8 sums.
#[inline]
fn first<R: Rounding>(sum: v128) -> v128 {
    match R::FIRST {
        Some(m) => round(sum, m),
        None => sum,
    }
}

/// `N` (4 or 8) pixels at `src`, widened to 16 bits.
#[inline]
unsafe fn pixels<const N: usize>(src: *const u8) -> v128 {
    // SAFETY: the caller provides N readable bytes; wasm loads need no
    // alignment.
    unsafe {
        if N == 4 {
            u16x8_extend_low_u8x16(v128_load32_zero(src.cast()))
        } else {
            u16x8_load_extend_u8x8(src)
        }
    }
}

/// Unrounded horizontal sum of `N` (4 or 8) outputs with taps `taps`, the
/// first tap's sample at `src`.
#[inline]
unsafe fn horizontal<const N: usize, const T: usize>(src: *const u8, taps: &[i16; T]) -> v128 {
    let mut sum = i16x8_splat(0);
    for (t, &tap) in taps.iter().enumerate() {
        // SAFETY: the caller provides N + T - 1 readable bytes.
        let p = unsafe { pixels::<N>(src.add(t)) };
        sum = i16x8_add(sum, i16x8_mul(p, i16x8_splat(tap)));
    }
    sum
}

/// `(x + 2^(k-1)) >> k` for `m = 2^(15 - k)`.
#[inline]
fn round(x: v128, m: i16) -> v128 {
    i16x8_q15mulr_sat(x, i16x8_splat(m))
}

/// Where a kernel's rounded 16-bit outputs go, and at which precision.
trait Sink: Copy {
    /// Compound precision (`prep`) rather than pixels (`put`).
    const PREP: bool;

    /// Store the first `N` (4 or 8) of 8 outputs at row `row`, column
    /// `col`.
    ///
    /// # Safety
    ///
    /// The outputs must lie inside the destination the caller checked.
    unsafe fn store<const N: usize>(self, row: usize, col: usize, words: v128);
}

impl Sink for Raw<*mut u8> {
    const PREP: bool = false;
    #[inline]
    unsafe fn store<const N: usize>(self, row: usize, col: usize, words: v128) {
        let bytes = u8x16_narrow_i16x8(words, words);
        // SAFETY: forwarded from the caller; wasm stores need no alignment.
        unsafe {
            let dst = self.at(row, col);
            if N == 4 {
                v128_store32_lane::<0>(bytes, dst.cast());
            } else {
                v128_store64_lane::<0>(bytes, dst.cast());
            }
        }
    }
}

impl Sink for Raw<*mut i16> {
    const PREP: bool = true;
    #[inline]
    unsafe fn store<const N: usize>(self, row: usize, col: usize, words: v128) {
        // SAFETY: forwarded from the caller; wasm stores need no alignment.
        unsafe {
            let dst = self.at(row, col);
            if N == 4 {
                v128_store64_lane::<0>(words, dst.cast());
            } else {
                v128_store(dst.cast(), words);
            }
        }
    }
}

/// Interleave the 16-bit lanes of two rows: outputs 0..4 and 4..8.
#[inline]
fn interleave(a: v128, b: v128) -> (v128, v128) {
    (
        i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(a, b),
        i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(a, b),
    )
}

/// The vertical pass of a 2-D filter over `P` interleaved row pairs, as 8
/// 16-bit outputs rounded by `(round, shift)`.
#[inline]
fn vertical<const P: usize>(
    pairs: &[(v128, v128); P],
    taps: &[v128; P],
    (round, shift): (i32, u32),
) -> v128 {
    let mut low = i32x4_splat(round);
    let mut high = low;
    for (&(lo, hi), &weight) in pairs.iter().zip(taps) {
        low = i32x4_add(low, i32x4_dot_i16x8(lo, weight));
        high = i32x4_add(high, i32x4_dot_i16x8(hi, weight));
    }
    i16x8_narrow_i32x4(i32x4_shr(low, shift), i32x4_shr(high, shift))
}

/// A raw window or destination: its first sample and row stride.
#[derive(Clone, Copy)]
struct Raw<T> {
    ptr: T,
    stride: usize,
}

impl Raw<*const u8> {
    /// # Safety
    ///
    /// The sample must lie inside the window the caller checked.
    #[inline]
    unsafe fn at(self, row: usize, col: usize) -> *const u8 {
        // SAFETY: forwarded from the caller.
        unsafe { self.ptr.add(row * self.stride + col) }
    }
}

impl<T> Raw<*mut T> {
    /// # Safety
    ///
    /// The sample must lie inside the destination the caller checked.
    #[inline]
    unsafe fn at(self, row: usize, col: usize) -> *mut T {
        // SAFETY: forwarded from the caller.
        unsafe { self.ptr.add(row * self.stride + col) }
    }
}

/// Copy `h` rows of `W` (4, 8 or a multiple of 16) bytes from `src` to `dst`
/// with vector loads and stores.
///
/// # Safety
///
/// `src` and `dst` must cover `h` rows of `W` bytes at their strides.
#[inline]
pub(super) unsafe fn copy_block<const W: usize>(
    (src, src_stride): (*const u8, usize),
    (dst, dst_stride): (*mut u8, usize),
    h: usize,
) {
    for r in 0..h {
        // SAFETY: the caller provides h rows of W bytes on both sides;
        // wasm loads and stores need no alignment.
        unsafe {
            let (from, to) = (src.add(r * src_stride), dst.add(r * dst_stride));
            match W {
                4 => v128_store32_lane::<0>(v128_load32_zero(from.cast()), to.cast()),
                8 => v128_store64_lane::<0>(v128_load64_zero(from.cast()), to.cast()),
                _ => {
                    for c in (0..W).step_by(VECTOR_BYTES) {
                        v128_store(to.add(c).cast(), v128_load(from.add(c).cast()));
                    }
                }
            }
        }
    }
}

/// The kernels of separable table `T` (`K` = `2P` taps) by width class
/// (4 to 64): [`separable`].
pub(super) fn separable_family<T: SeparableTaps<K>, const K: usize, const P: usize>()
-> [ConvolveFn; 5] {
    const { assert!(K == 2 * P) };
    [
        separable::<T, K, P, 4, 4>,
        separable::<T, K, P, 8, LANES>,
        separable::<T, K, P, 16, LANES>,
        separable::<T, K, P, 32, LANES>,
        separable::<T, K, P, 64, LANES>,
    ]
}

/// The HEVC/VVC separable prediction (`T`'s `K`-tap table) of a `W`x`h`
/// block at pixel precision on the same passes as AV1's `put`
/// ([`SeparableRounding`]: full taps, unrounded horizontal sums): the
/// `ConvolveFn` contract with a `K`-tap window (the block at row and column
/// `K / 2 - 1`), `N` outputs (4, or 8 in strips) per vector.
///
/// # Safety
///
/// None beyond the kernel contract: SIMD128 is enabled for the build.
unsafe fn separable<
    T: SeparableTaps<K>,
    const K: usize,
    const P: usize,
    const W: usize,
    const N: usize,
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
        (K == TAPS || K == SHORT_TAPS)
            && h > 0
            && h.is_multiple_of(2)
            && src.len() >= (h + K - 2) * src_stride + W + K - 1
            && dst.len() >= (h - 1) * dst_stride + W,
        "separable prediction window or destination too small"
    );
    let window = Raw {
        ptr: src.as_ptr(),
        stride: src_stride,
    };
    let out = Raw {
        ptr: dst.as_mut_ptr(),
        stride: dst_stride,
    };
    let taps = (&T::TABLE[frac_x][..], &T::TABLE[frac_y][..]);
    // SAFETY: the window covers h + K - 1 rows of W + K - 1 samples and the
    // destination h rows of W (checked above); every path reads rows and
    // columns 0..h + K - 1 / 0..W + K - 1 of the window (the centre ones
    // for one direction) and writes the block.
    unsafe {
        if (frac_x, frac_y) == (0, 0) {
            copy_block::<W>(
                (window.at(K / 2 - 1, K / 2 - 1), src_stride),
                (out.ptr, dst_stride),
                h,
            );
        } else if K == TAPS {
            put_taps::<W, N, 4, 8, _, SeparableRounding, { TAPS / 2 - 1 }>(
                window,
                0,
                out,
                h,
                (frac_x, frac_y),
                taps,
            );
        } else {
            put_taps::<W, N, 2, 4, _, SeparableRounding, { SHORT_TAPS / 2 - 1 }>(
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

/// AV1 prediction of a `W`x`h` block (module and `mc` contract), `N`
/// outputs (4, or 8 in strips) per vector.
///
/// # Safety
///
/// None beyond the kernel contract: SIMD128 is enabled for the build.
pub(super) unsafe fn put<F: Av1Family, const W: usize, const N: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    let taps = || (F::taps(W, frac_x), F::taps(h, frac_y));
    // SAFETY: forwarded from the caller.
    unsafe {
        put_with::<W, N, F>(
            (src, src_stride),
            (dst, dst_stride),
            h,
            (frac_x, frac_y),
            taps,
        );
    }
}

/// [`put`] with the caller's taps per direction.
///
/// # Safety
///
/// None beyond the kernel contract.
unsafe fn put_mixed<const W: usize, const N: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
    fracs: (usize, usize),
    taps: TapPair<'_>,
) {
    // SAFETY: forwarded from the caller.
    unsafe { put_with::<W, N, AnyTaps>((src, src_stride), (dst, dst_stride), h, fracs, || taps) };
}

/// [`put`] with the taps `taps` gives (asked only when the block is
/// filtered), whose outer taps are known zero when `Z` says so.
#[inline]
unsafe fn put_with<'t, const W: usize, const N: usize, Z: OuterTaps>(
    (src, src_stride): (&[u8], usize),
    (dst, dst_stride): (&mut [u8], usize),
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
    if (frac_x, frac_y) == (0, 0) {
        let from = TAP_CENTRE * src_stride + TAP_CENTRE;
        // SAFETY: the window covers rows TAP_CENTRE..TAP_CENTRE + h from
        // column TAP_CENTRE, W samples each, and the destination h rows of
        // W (checked above).
        unsafe {
            copy_block::<W>(
                (src[from..].as_ptr(), src_stride),
                (dst.as_mut_ptr(), dst_stride),
                h,
            );
        }
        return;
    }
    let window = Raw {
        ptr: src.as_ptr(),
        stride: src_stride,
    };
    let out = Raw {
        ptr: dst.as_mut_ptr(),
        stride: dst_stride,
    };
    // SAFETY: the window covers h + 7 rows of W + 7 samples and the
    // destination h rows of W (checked above).
    unsafe { filter::<W, N, _, Z>(window, out, h, (frac_x, frac_y), taps()) };
}

/// AV1 prediction of a `W`x`h` block at compound precision (`prep`, AV1
/// §7.11.3.4 with `InterRound1` = 7; module and `mc` contract), `N`
/// outputs (4, or 8 in strips) per vector.
///
/// # Safety
///
/// None beyond the kernel contract: SIMD128 is enabled for the build.
pub(super) unsafe fn prep<F: Av1Family, const W: usize, const N: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    let taps = || (F::taps(W, frac_x), F::taps(h, frac_y));
    // SAFETY: forwarded from the caller.
    unsafe {
        prep_with::<W, N, F>(
            (src, src_stride),
            (dst, dst_stride),
            h,
            (frac_x, frac_y),
            taps,
        );
    }
}

/// [`prep`] with the caller's taps per direction.
///
/// # Safety
///
/// None beyond the kernel contract.
unsafe fn prep_mixed<const W: usize, const N: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    h: usize,
    fracs: (usize, usize),
    taps: TapPair<'_>,
) {
    // SAFETY: forwarded from the caller.
    unsafe { prep_with::<W, N, AnyTaps>((src, src_stride), (dst, dst_stride), h, fracs, || taps) };
}

/// [`prep`] with taps `(h_taps, v_taps)`, as [`put_with`].
#[inline]
unsafe fn prep_with<'t, const W: usize, const N: usize, Z: OuterTaps>(
    (src, src_stride): (&[u8], usize),
    (dst, dst_stride): (&mut [i16], usize),
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
    let window = Raw {
        ptr: src.as_ptr(),
        stride: src_stride,
    };
    let out = Raw {
        ptr: dst.as_mut_ptr(),
        stride: dst_stride,
    };
    // SAFETY: as in `put_with`, with `i16` outputs. The copy reads row r + 3
    // from column 3 (W samples) of the window.
    unsafe {
        if (frac_x, frac_y) == (0, 0) {
            for r in 0..h {
                for c0 in (0..W).step_by(N) {
                    let samples = pixels::<N>(window.at(r + TAP_CENTRE, c0 + TAP_CENTRE));
                    out.store::<N>(r, c0, i16x8_shl(samples, PREP_COPY_SHIFT));
                }
            }
        } else {
            filter::<W, N, _, Z>(window, out, h, (frac_x, frac_y), taps());
        }
    }
}

/// The filtered (not whole-sample) prediction with taps `taps` into `out`.
///
/// # Safety
///
/// The window covers h + 7 rows of W + 7 samples and `out` h rows of W.
/// Each path reads rows lead..lead + h + 2P - 1 and columns
/// lead..lead + W + 2P - 1 of the window (or its centre column/row for a
/// one-direction filter), and writes rows 0..h, columns 0..W.
#[inline]
unsafe fn filter<const W: usize, const N: usize, S: Sink, Z: OuterTaps>(
    window: Raw<*const u8>,
    out: S,
    h: usize,
    (frac_x, frac_y): (usize, usize),
    taps: (&[i16; TAPS], &[i16; TAPS]),
) {
    // SAFETY: forwarded from the caller.
    unsafe {
        if Z::OUTER_ZERO {
            put_taps::<W, N, 3, 6, S, Av1Rounding, TAP_CENTRE>(
                window,
                1,
                out,
                h,
                (frac_x, frac_y),
                (&taps.0[..], &taps.1[..]),
            );
        } else {
            put_taps::<W, N, 4, 8, S, Av1Rounding, TAP_CENTRE>(
                window,
                0,
                out,
                h,
                (frac_x, frac_y),
                (&taps.0[..], &taps.1[..]),
            );
        }
    }
}

/// [`filter`] with `P` tap pairs (`T = 2P` taps), the first tap `lead`
/// samples into the window, rounded by `R`; `C` is the window row and
/// column of the block's first sample.
#[inline]
unsafe fn put_taps<
    const W: usize,
    const N: usize,
    const P: usize,
    const T: usize,
    S: Sink,
    R: Rounding,
    const C: usize,
>(
    window: Raw<*const u8>,
    lead: usize,
    out: S,
    h: usize,
    (frac_x, frac_y): (usize, usize),
    (h_taps, v_taps): (&[i16], &[i16]),
) {
    let (v_round, last) = if S::PREP {
        (ROUND_2, (PREP_LAST_ROUND, PREP_LAST_SHIFT))
    } else {
        (ROUND_6, (R::LAST_ROUND, R::LAST_SHIFT))
    };
    let last = (last.0, last.1.cast_unsigned());
    // SAFETY: each path keeps to the window and destination the caller
    // checked.
    unsafe {
        match (frac_x, frac_y) {
            (_, 0) => {
                let taps = taps_of::<R, T>(h_taps, lead);
                for r in 0..h {
                    for c0 in (0..W).step_by(N) {
                        let sum = horizontal::<N, T>(window.at(r + C, c0 + lead), &taps);
                        let t = first::<R>(sum);
                        out.store::<N>(
                            r,
                            c0,
                            if S::PREP {
                                t
                            } else {
                                round(t, R::HORIZONTAL_PUT)
                            },
                        );
                    }
                }
            }
            (0, _) => {
                let taps = taps_of::<R, T>(v_taps, lead);
                for c0 in (0..W).step_by(N) {
                    let mut rows = [i16x8_splat(0); 8];
                    for (k, row) in rows.iter_mut().enumerate().take(T - 1) {
                        *row = pixels::<N>(window.at(lead + k, c0 + C));
                    }
                    for r in 0..h {
                        rows[T - 1] = pixels::<N>(window.at(lead + r + T - 1, c0 + C));
                        let mut sum = i16x8_splat(0);
                        for (row, &tap) in rows.iter().zip(&taps) {
                            sum = i16x8_add(sum, i16x8_mul(*row, i16x8_splat(tap)));
                        }
                        out.store::<N>(r, c0, round(sum, v_round));
                        slide(&mut rows, 1, T - 1);
                    }
                }
            }
            _ => {
                let h_taps = taps_of::<R, T>(h_taps, lead);
                let v_row = taps_of::<R, T>(v_taps, lead);
                let v_taps: [v128; P] = core::array::from_fn(|j| {
                    let (lo, hi) = (v_row[2 * j] as u16, v_row[2 * j + 1] as u16);
                    i32x4_splat((u32::from(lo) | u32::from(hi) << 16) as i32)
                });
                for c0 in (0..W).step_by(N) {
                    let row = |k: usize| {
                        first::<R>(horizontal::<N, T>(window.at(lead + k, c0 + lead), &h_taps))
                    };
                    // Even pairs (0,1), (2,3), ...; odd pairs (1,2), (3,4),
                    // ...; the last of each made in the loop.
                    let zero = i16x8_splat(0);
                    let mut rows = [zero; 8];
                    for (k, slot) in rows.iter_mut().enumerate().take(T - 1) {
                        *slot = row(k);
                    }
                    let mut even = [(zero, zero); P];
                    let mut odd = even;
                    for j in 0..P - 1 {
                        even[j] = interleave(rows[2 * j], rows[2 * j + 1]);
                        odd[j] = interleave(rows[2 * j + 1], rows[2 * j + 2]);
                    }
                    let mut previous = rows[T - 2];
                    for r in (0..h).step_by(2) {
                        let (a, b) = (row(r + T - 1), row(r + T));
                        even[P - 1] = interleave(previous, a);
                        odd[P - 1] = interleave(a, b);
                        previous = b;
                        out.store::<N>(r, c0, vertical(&even, &v_taps, last));
                        out.store::<N>(r + 1, c0, vertical(&odd, &v_taps, last));
                        slide(&mut even, 1, P - 1);
                        slide(&mut odd, 1, P - 1);
                    }
                }
            }
        }
    }
}
