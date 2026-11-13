//! WebAssembly SIMD128 HEVC and H.266 separable prediction at pixel
//! precision (uni-prediction, default weights): `fL` / `fC` and the H.266
//! tables of [`SeparableTaps`], and the whole-sample block copy that the
//! other SIMD128 prediction paths share.
//!
//! The structure is `av1_simd128`'s (after dav1d's `put_8tap`, BSD-2-Clause;
//! see `THIRD_PARTY.md`), with HEVC's arithmetic (`mc` module docs, HEVC
//! §8.5.3.3.3): a one-direction filter is a 16-bit multiply-add per tap
//! rounded by `(x + 32) >> 6` (one `i16x8.q15mulr_sat_s`); a 2-D filter keeps
//! its unrounded 16-bit horizontal sums in a rolling window of rows per
//! column strip and runs the vertical pass on interleaved row pairs against
//! tap pairs (`i32x4.dot_i16x8_s`, half the multiplies of a per-tap
//! widening product), two output rows per step, rounded by
//! `(x + 2048) >> 12` (`shift2` = 6, then the uni-prediction `>> 6`, folded
//! into one rounding: the nested floors are equal). The taps are not
//! halved (HEVC's are odd). Results equal the scalar reference.
//!
//! A whole-sample block is copied with vector loads and stores per row:
//! `copy_from_slice` per row compiles to `memory.copy`, which V8 runs out
//! of line (11-17 ns per 16-64-byte row against 6-9 ns for the vector
//! loop, node 22).
//!
//! Kernel contract: see the `mc` module docs; the window and destination
//! are checked once on entry and the row loops then use raw pointers.

use super::{ConvolveFn, SeparableTaps};
use core::arch::wasm32::{
    i8x16_shuffle, i16x8_add, i16x8_mul, i16x8_narrow_i32x4, i16x8_q15mulr_sat, i16x8_splat,
    i32x4_add, i32x4_dot_i16x8, i32x4_shr, i32x4_splat, u8x16_narrow_i16x8, u16x8_extend_low_u8x16,
    u16x8_load_extend_u8x8, v128, v128_load, v128_load32_zero, v128_load64_zero, v128_store,
    v128_store32_lane, v128_store64_lane,
};

/// Outputs of one vector (8 lanes of 16 bits).
const LANES: usize = 8;
/// Bytes of one vector.
const VECTOR_BYTES: usize = 16;
/// `q15mulr` multiplier for `(x + 32) >> 6`.
const ROUND_6: i16 = 1 << 9;
/// The vertical pass of a 2-D filter: `(acc + 2048) >> 12`.
const LAST_ROUND: i32 = 1 << 11;
const LAST_SHIFT: u32 = 12;

/// The kernels of separable table `T` (`K` = `2P` taps) by width class (4
/// to 64).
pub(super) fn family<T: SeparableTaps<K>, const K: usize, const P: usize>() -> [ConvolveFn; 5] {
    const { assert!(K == 2 * P) };
    [
        put::<T, K, P, 4, 4>,
        put::<T, K, P, 8, LANES>,
        put::<T, K, P, 16, LANES>,
        put::<T, K, P, 32, LANES>,
        put::<T, K, P, 64, LANES>,
    ]
}

/// A raw window or destination: its first sample and row stride.
#[derive(Clone, Copy)]
struct Raw<T> {
    ptr: T,
    stride: usize,
}

impl<T> Raw<*const T> {
    /// # Safety
    ///
    /// The sample must lie inside the window the caller checked.
    #[inline]
    unsafe fn at(self, row: usize, col: usize) -> *const T {
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

/// `N` (4 or 8) pixels at `src`, widened to 16 bits.
#[inline]
unsafe fn pixels<const N: usize>(src: *const u8) -> v128 {
    // SAFETY: the caller provides N readable bytes.
    unsafe {
        if N == 4 {
            u16x8_extend_low_u8x16(v128_load32_zero(src.cast()))
        } else {
            u16x8_load_extend_u8x8(src)
        }
    }
}

/// Unrounded sum of `K` taps over `N` (4 or 8) outputs, the first tap's
/// sample at `src` and each next tap `step` bytes further.
#[inline]
unsafe fn taps_sum<const N: usize, const K: usize>(
    src: *const u8,
    step: usize,
    taps: &[i16; K],
) -> v128 {
    let mut sum = i16x8_splat(0);
    for (t, &tap) in taps.iter().enumerate() {
        // SAFETY: the caller provides the K samples of each output.
        let p = unsafe { pixels::<N>(src.add(t * step)) };
        sum = i16x8_add(sum, i16x8_mul(p, i16x8_splat(tap)));
    }
    sum
}

/// Store the first `N` (4 or 8) of 8 16-bit outputs as bytes (saturated:
/// the uni-prediction clip).
///
/// # Safety
///
/// `dst` must be valid for writing `N` bytes.
#[inline]
unsafe fn store<const N: usize>(dst: *mut u8, words: v128) {
    let bytes = u8x16_narrow_i16x8(words, words);
    // SAFETY: forwarded from the caller.
    unsafe {
        if N == 4 {
            v128_store32_lane::<0>(bytes, dst.cast());
        } else {
            v128_store64_lane::<0>(bytes, dst.cast());
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

/// The vertical pass over `P` interleaved row pairs: 8 outputs rounded by
/// `(x + 2048) >> 12`, narrowed to 16 bits.
#[inline]
fn vertical<const P: usize>(pairs: &[(v128, v128); P], taps: &[v128; P]) -> v128 {
    let mut low = i32x4_splat(LAST_ROUND);
    let mut high = low;
    for (&(lo, hi), &weight) in pairs.iter().zip(taps) {
        low = i32x4_add(low, i32x4_dot_i16x8(lo, weight));
        high = i32x4_add(high, i32x4_dot_i16x8(hi, weight));
    }
    i16x8_narrow_i32x4(i32x4_shr(low, LAST_SHIFT), i32x4_shr(high, LAST_SHIFT))
}

/// Prediction of a `W`x`h` block with table `T` (`K` = `2P` taps), `N`
/// outputs (4, or 8 in strips) per vector.
///
/// # Safety
///
/// None beyond the kernel contract: SIMD128 is enabled for the build.
unsafe fn put<
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
        h > 0
            && src_stride >= W + K - 1
            && src.len() >= (h + K - 2) * src_stride + W + K - 1
            && dst_stride >= W
            && dst.len() >= (h - 1) * dst_stride + W,
        "separable prediction window or destination too small"
    );
    let centre = K / 2 - 1;
    let window = Raw {
        ptr: src.as_ptr(),
        stride: src_stride,
    };
    let out = Raw {
        ptr: dst.as_mut_ptr(),
        stride: dst_stride,
    };
    // SAFETY: the window covers h + K - 1 rows of W + K - 1 samples and the
    // destination h rows of W (checked above). Each path reads at most
    // that: a strip of N outputs at column c0 < W reads columns c0..c0 +
    // N + K - 2, and loads of 8 pixels for 4-tap filters stay below W + 3.
    unsafe {
        match (frac_x, frac_y) {
            (0, 0) => copy_block::<W>(
                (window.at(centre, centre), src_stride),
                (out.ptr, dst_stride),
                h,
            ),
            (_, 0) => {
                let taps = &T::TABLE[frac_x];
                for r in 0..h {
                    for c0 in (0..W).step_by(N) {
                        let sum = taps_sum::<N, K>(window.at(r + centre, c0), 1, taps);
                        store::<N>(out.at(r, c0), i16x8_q15mulr_sat(sum, i16x8_splat(ROUND_6)));
                    }
                }
            }
            (0, _) => {
                // Each output row reloads its K source rows: a rolling
                // window of widened rows measured more WASM instructions
                // (V8 spills the 8-tap window).
                let taps = &T::TABLE[frac_y];
                for r in 0..h {
                    for c0 in (0..W).step_by(N) {
                        let sum = taps_sum::<N, K>(window.at(r, c0 + centre), src_stride, taps);
                        store::<N>(out.at(r, c0), i16x8_q15mulr_sat(sum, i16x8_splat(ROUND_6)));
                    }
                }
            }
            _ => two_d::<T, K, P, W, N>(window, out, h, (frac_x, frac_y)),
        }
    }
}

/// The 2-D path of [`put`]: unrounded horizontal sums, vertical pass on
/// `P` tap pairs.
///
/// # Safety
///
/// As in [`put`], after its checks.
#[inline]
unsafe fn two_d<
    T: SeparableTaps<K>,
    const K: usize,
    const P: usize,
    const W: usize,
    const N: usize,
>(
    window: Raw<*const u8>,
    out: Raw<*mut u8>,
    h: usize,
    (frac_x, frac_y): (usize, usize),
) {
    let h_taps = &T::TABLE[frac_x];
    let v_row = &T::TABLE[frac_y];
    let v_taps: [v128; P] = core::array::from_fn(|j| {
        let (lo, hi) = (v_row[2 * j] as u16, v_row[2 * j + 1] as u16);
        i32x4_splat((u32::from(lo) | u32::from(hi) << 16) as i32)
    });
    let zero = i16x8_splat(0);
    for c0 in (0..W).step_by(N) {
        // SAFETY: rows 0..h + K - 1 of the window, columns from c0 (see put).
        let row = |k: usize| unsafe { taps_sum::<N, K>(window.at(k, c0), 1, h_taps) };
        // Even pairs (0,1), (2,3), ...; odd pairs (1,2), (3,4), ...; the
        // last of each is made in the loop.
        let mut even = [(zero, zero); P];
        let mut odd = even;
        let mut previous = row(0);
        for j in 0..P - 1 {
            let (upper, lower) = (row(2 * j + 1), row(2 * j + 2));
            even[j] = interleave(previous, upper);
            odd[j] = interleave(upper, lower);
            previous = lower;
        }
        let mut r = 0;
        while r < h {
            let upper = row(r + K - 1);
            even[P - 1] = interleave(previous, upper);
            // SAFETY: row r < h of the destination.
            unsafe { store::<N>(out.at(r, c0), vertical(&even, &v_taps)) };
            if r + 1 == h {
                break;
            }
            let lower = row(r + K);
            odd[P - 1] = interleave(upper, lower);
            // SAFETY: row r + 1 < h of the destination.
            unsafe { store::<N>(out.at(r + 1, c0), vertical(&odd, &v_taps)) };
            previous = lower;
            // Shift the windows by one pair, element by element (an array
            // `copy_within` can become a `memory.copy` call).
            for j in 0..P - 1 {
                even[j] = even[j + 1];
                odd[j] = odd[j + 1];
            }
            r += 2;
        }
    }
}
