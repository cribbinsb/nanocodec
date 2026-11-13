//! WebAssembly SIMD128 HEVC 8/16/32-point inverse DCTs on 16-bit pairs.
//!
//! The generic inverse multiplies one coefficient by one basis value per
//! 32-bit lane (`i32x4.mul`, two micro-ops on x86). Here two coefficients of
//! a line share a 32-bit lane as 16-bit halves and `i32x4.dot_i16x8_s`
//! multiplies both by their basis values and adds the products, halving the
//! multiplies (`pmaddwd`, one micro-op). The structure is the generic
//! partial butterfly (even part through the half-size inverse, odd part a
//! matrix product), four lines per vector, with the same zero skipping:
//! trailing zero coefficient rows and columns are neither loaded nor
//! multiplied.
//!
//! # Exactness (range proof)
//!
//! - Both passes' inputs fit `i16`: the first pass clamps the coefficients
//!   to 16 bits, as HEVC §8.6.4.2 and the generic code do, and the second
//!   pass reads the first's output, clamped to 16 bits after its shift. So
//!   narrowing them to 16-bit lanes (`i16x8.narrow_i32x4_s`, saturating)
//!   never changes a value.
//! - Every basis value has magnitude <= 90 (`DCT32`), so each product has
//!   magnitude <= 32768 · 90 < 2^22 and a dot of two < 2^23: the 32-bit
//!   result of `dot_i16x8_s` is exact (it can only wrap for two
//!   `−32768 · −32768` products).
//! - A pass sums at most 32 products, < 2^28 in magnitude, so every 32-bit
//!   addition is exact and the sums equal the generic code's in any order.
//!   Rounding, shifts and clamps then match the generic passes exactly.

// The helpers must inline into the line transforms, which keep their
// vectors in registers only when fully inlined.
#![allow(clippy::inline_always)]

use core::arch::wasm32::{
    i16x8_narrow_i32x4, i16x8_shuffle, i32x4_add, i32x4_dot_i16x8, i32x4_extend_low_i16x8,
    i32x4_max, i32x4_min, i32x4_shr, i32x4_shuffle, i32x4_splat, i32x4_sub, u8x16_narrow_i16x8,
    u16x8_extend_low_u8x16, v128, v128_load, v128_load32_zero, v128_store, v128_store32_lane,
};
use core::mem::MaybeUninit;

use super::DCT32;
use super::generic::{HEVC_INVERSE_PASSES, MAX_AREA, nonzero_extent, written};
use super::{CodecTxfmFns, HevcTransform};

/// Lines per vector.
const LANES: usize = 4;

/// The basis value of coefficient `k` at sample `i` of the `2^log2`-point
/// DCT (`DCT32` row `k · 32 / N`).
const fn basis(log2: usize, k: usize, i: usize) -> i32 {
    DCT32[k << (5 - log2)][i] as i32
}

/// Two 16-bit basis values packed as one 32-bit lane (`low` in the low half),
/// the second operand of `dot_i16x8_s` against a coefficient pair.
const fn pack(low: i32, high: i32) -> i32 {
    (low & 0xFFFF) | (high << 16)
}

/// The odd-part weights of the `N = 2^log2`-point inverse: entry `[i][p]`
/// pairs the basis values at sample `i` of odd coefficients `4p + 1` and
/// `4p + 3`.
const fn odd_weights<const H: usize, const P: usize>(log2: usize) -> [[i32; P]; H] {
    let mut w = [[0; P]; H];
    let mut i = 0;
    while i < H {
        let mut p = 0;
        while p < P {
            w[i][p] = pack(basis(log2, 4 * p + 1, i), basis(log2, 4 * p + 3, i));
            p += 1;
        }
        i += 1;
    }
    w
}

const ODD8: [[i32; 2]; 4] = odd_weights::<4, 2>(3);
const ODD16: [[i32; 4]; 8] = odd_weights::<8, 4>(4);
const ODD32: [[i32; 8]; 16] = odd_weights::<16, 8>(5);

/// The two coefficients' lanes interleaved as 16-bit pairs `(a, b)` per
/// line (values fit `i16`, module docs).
#[inline(always)]
fn pair(a: v128, b: v128) -> v128 {
    let narrow = i16x8_narrow_i32x4(a, b);
    i16x8_shuffle::<0, 4, 1, 5, 2, 6, 3, 7>(narrow, narrow)
}

/// `a · w_low + b · w_high` per line, for a pair from [`pair`].
#[inline(always)]
fn dot(pair: v128, weights: i32) -> v128 {
    i32x4_dot_i16x8(pair, i32x4_splat(weights))
}

/// HEVC 4-point inverse of coefficients `c[0]`, `c[s]`, `c[2s]`, `c[3s]`.
#[inline(always)]
fn idct4(c: &[v128], s: usize, y: &mut [v128]) {
    let even = pair(c[0], c[2 * s]);
    let odd = pair(c[s], c[3 * s]);
    let e0 = dot(even, pack(basis(2, 0, 0), basis(2, 2, 0)));
    let e1 = dot(even, pack(basis(2, 0, 1), basis(2, 2, 1)));
    let o0 = dot(odd, pack(basis(2, 1, 0), basis(2, 3, 0)));
    let o1 = dot(odd, pack(basis(2, 1, 1), basis(2, 3, 1)));
    y[0] = i32x4_add(e0, o0);
    y[1] = i32x4_add(e1, o1);
    y[2] = i32x4_sub(e1, o1);
    y[3] = i32x4_sub(e0, o0);
}

/// Defines the `$n`-point inverse of coefficients `c[k · s]` (zero from
/// `active` on): the even part through the half-size inverse, the odd part
/// as pair dots, only over pairs holding a coefficient below `active`.
macro_rules! idct_level {
    ($name:ident, $n:literal, $half:ident, $weights:ident) => {
        #[inline(always)]
        fn $name(c: &[v128], s: usize, active: usize, y: &mut [v128]) {
            const H: usize = $n / 2;
            let mut even = [i32x4_splat(0); H];
            half!($half, c, 2 * s, active.div_ceil(2), &mut even);
            // Odd coefficient `2j + 1` is nonzero only below `active`; pair
            // `p` holds odd coefficients `4p + 1` and `4p + 3`.
            let pairs = active.div_ceil(4).min(H / 2);
            let mut odd_pairs = [i32x4_splat(0); H / 2];
            for (p, slot) in odd_pairs[..pairs].iter_mut().enumerate() {
                *slot = pair(c[(4 * p + 1) * s], c[(4 * p + 3) * s]);
            }
            for (i, weights) in $weights.iter().enumerate() {
                let mut odd = i32x4_splat(0);
                for (&p, &w) in odd_pairs[..pairs].iter().zip(weights) {
                    odd = i32x4_add(odd, dot(p, w));
                }
                y[i] = i32x4_add(even[i], odd);
                y[$n - 1 - i] = i32x4_sub(even[i], odd);
            }
        }
    };
}

/// Calls the half-size inverse (the 4-point one takes no bound).
macro_rules! half {
    (idct4, $c:expr, $s:expr, $active:expr, $y:expr) => {
        idct4($c, $s, $y)
    };
    ($half:ident, $c:expr, $s:expr, $active:expr, $y:expr) => {
        $half($c, $s, $active, $y)
    };
}

idct_level!(idct8, 8, idct4, ODD8);
idct_level!(idct16, 16, idct8, ODD16);
idct_level!(idct32, 32, idct16, ODD32);

/// `clamp16(round_shift(v, shift))` or the bare round shift.
#[inline(always)]
fn round(v: v128, shift: u32, clamp: bool) -> v128 {
    let shifted = i32x4_shr(i32x4_add(v, i32x4_splat(1 << (shift - 1))), shift);
    if clamp { clamp16(shifted) } else { shifted }
}

#[inline(always)]
fn clamp16(v: v128) -> v128 {
    i32x4_max(
        i32x4_min(v, i32x4_splat(i32::from(i16::MAX))),
        i32x4_splat(i32::from(i16::MIN)),
    )
}

/// Four `i32`s of `src` from `at`.
#[inline(always)]
fn load4(src: &[i32], at: usize) -> v128 {
    let src = &src[at..at + LANES];
    // SAFETY: `src` holds four `i32`s (16 bytes).
    unsafe { v128_load(src.as_ptr().cast()) }
}

/// Transpose four output vectors from lane-parallel lines into rows.
#[inline(always)]
fn transpose(y: &[v128]) -> [v128; LANES] {
    let [n0, n1, n2, n3] = [y[0], y[1], y[2], y[3]];
    let ab_low = i32x4_shuffle::<0, 4, 1, 5>(n0, n1);
    let ab_high = i32x4_shuffle::<2, 6, 3, 7>(n0, n1);
    let cd_low = i32x4_shuffle::<0, 4, 1, 5>(n2, n3);
    let cd_high = i32x4_shuffle::<2, 6, 3, 7>(n2, n3);
    [
        i32x4_shuffle::<0, 1, 4, 5>(ab_low, cd_low),
        i32x4_shuffle::<2, 3, 6, 7>(ab_low, cd_low),
        i32x4_shuffle::<0, 1, 4, 5>(ab_high, cd_high),
        i32x4_shuffle::<2, 3, 6, 7>(ab_high, cd_high),
    ]
}

/// Store a group of four lines from one pass, transposed.
#[inline(always)]
fn store_lines<const N: usize>(y: &[v128], first: usize, dst: &mut [MaybeUninit<i32>]) {
    for n in (0..N).step_by(LANES) {
        for (l, line) in transpose(&y[n..]).into_iter().enumerate() {
            let at = (first + l) * N + n;
            let out = &mut dst[at..at + LANES];
            // SAFETY: out holds four i32 slots (16 bytes).
            unsafe { v128_store(out.as_mut_ptr().cast(), line) };
        }
    }
}

/// Fused final transpose, add prediction and clip to 8 bits.
#[inline(always)]
fn add_lines<const N: usize>(y: &[v128], first: usize, dst: &mut [u8], stride: usize) {
    for n in (0..N).step_by(LANES) {
        for (l, line) in transpose(&y[n..]).into_iter().enumerate() {
            let at = (first + l) * stride + n;
            let out = &mut dst[at..at + LANES];
            // SAFETY: out holds exactly four bytes for the lane load/store.
            unsafe {
                let pred = i32x4_extend_low_i16x8(u16x8_extend_low_u8x16(v128_load32_zero(
                    out.as_ptr().cast(),
                )));
                let sum = i32x4_add(pred, line);
                let short = i16x8_narrow_i32x4(sum, sum);
                v128_store32_lane::<0>(u8x16_narrow_i16x8(short, short), out.as_mut_ptr().cast());
            }
        }
    }
}

/// The 2-D `N`-point inverse with the line transform `line` (columns, then
/// rows), as the generic `hevc_inverse_bounded`.
#[inline(always)]
fn inverse_passes<const N: usize>(
    coeffs: &[i32],
    extent: (usize, usize),
    line: fn(&[v128], usize, usize, &mut [v128]),
    mut store: impl FnMut(&[v128], usize),
) {
    let (rows, columns) = extent;
    let (first, second) = HEVC_INVERSE_PASSES;
    let mut c = [i32x4_splat(0); N];
    let mut y = [i32x4_splat(0); N];
    // `g[x · N + n]`: the column pass's output, column-major; columns from
    // `columns` (rounded up to whole vectors) on are zero and never read.
    let mut g = [MaybeUninit::<i32>::uninit(); MAX_AREA];
    let written_columns = columns.div_ceil(LANES) * LANES;
    for x0 in (0..written_columns).step_by(LANES) {
        for (k, lane) in c[..rows].iter_mut().enumerate() {
            *lane = clamp16(load4(coeffs, k * N + x0));
        }
        line(&c, 1, rows, &mut y);
        for lane in &mut y {
            *lane = round(*lane, first.shift, first.clamp);
        }
        store_lines::<N>(&y, x0, &mut g);
    }
    // SAFETY: `store_lines` wrote columns 0..written_columns of `g`.
    let g = unsafe { written(&g, written_columns * N) };
    c.fill(i32x4_splat(0));
    for y0 in (0..N).step_by(LANES) {
        for (k, lane) in c[..columns].iter_mut().enumerate() {
            *lane = load4(g, k * N + y0);
        }
        line(&c, 1, columns, &mut y);
        for lane in &mut y {
            *lane = round(*lane, second.shift, second.clamp);
        }
        store(&y, y0);
    }
}

/// Residual-output adapter retained for existing callers.
#[inline(always)]
fn inverse<const N: usize>(
    coeffs: &[i32],
    out: &mut [i32],
    line: fn(&[v128], usize, usize, &mut [v128]),
) {
    let extent = nonzero_extent::<N>(coeffs);
    let out = &mut out[..N * N];
    if extent.0 == 0 {
        out.fill(0);
        return;
    }
    let slots = super::generic::as_uninit(out);
    inverse_passes::<N>(coeffs, extent, line, |y, row| {
        store_lines::<N>(y, row, slots);
    });
}

/// i32-input, u8-output adapter; the paired-dot passes are unchanged.
#[inline(always)]
fn inverse_add<const N: usize>(
    coeffs: &[i32],
    extent: (usize, usize),
    dst: &mut [u8],
    stride: usize,
    line: fn(&[v128], usize, usize, &mut [v128]),
) {
    if extent.0 == 0 || extent.1 == 0 {
        return;
    }
    if extent == (1, 1) {
        super::inverse_add::dc::<crate::simd::simd128::I32x4, N>(coeffs[0], dst, stride);
        return;
    }
    inverse_passes::<N>(coeffs, extent, line, |y, row| {
        add_lines::<N>(y, row, dst, stride);
    });
}

/// # Safety
///
/// Kernel contract (`codec_txfm` module docs).
unsafe fn hevc_idct8(coeffs: &[i32], out: &mut [i32]) {
    inverse::<8>(coeffs, out, idct8);
}

/// # Safety
///
/// Kernel contract (`codec_txfm` module docs).
unsafe fn hevc_idct16(coeffs: &[i32], out: &mut [i32]) {
    inverse::<16>(coeffs, out, idct16);
}

/// # Safety
///
/// Kernel contract (`codec_txfm` module docs).
unsafe fn hevc_idct32(coeffs: &[i32], out: &mut [i32]) {
    inverse::<32>(coeffs, out, idct32);
}

unsafe fn hevc_add8(coeffs: &[i32], extent: (usize, usize), dst: &mut [u8], stride: usize) {
    inverse_add::<8>(coeffs, extent, dst, stride, idct8);
}
unsafe fn hevc_add16(coeffs: &[i32], extent: (usize, usize), dst: &mut [u8], stride: usize) {
    inverse_add::<16>(coeffs, extent, dst, stride, idct16);
}
unsafe fn hevc_add32(coeffs: &[i32], extent: (usize, usize), dst: &mut [u8], stride: usize) {
    inverse_add::<32>(coeffs, extent, dst, stride, idct32);
}

/// Install the 16-bit-pair HEVC inverses (8, 16 and 32 points).
pub(crate) fn install(table: &mut CodecTxfmFns) {
    table.hevc_inverse_add[HevcTransform::Dct8 as usize] = hevc_add8;
    table.hevc_inverse_add[HevcTransform::Dct16 as usize] = hevc_add16;
    table.hevc_inverse_add[HevcTransform::Dct32 as usize] = hevc_add32;
    table.hevc_inverse[HevcTransform::Dct8 as usize] = hevc_idct8;
    table.hevc_inverse[HevcTransform::Dct16 as usize] = hevc_idct16;
    table.hevc_inverse[HevcTransform::Dct32 as usize] = hevc_idct32;
}
