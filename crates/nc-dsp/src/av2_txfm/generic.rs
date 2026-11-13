//! The AV2 DCTs generic over [`I32Lanes`]: each 1-D pass transforms
//! `V::LANES` lines at once, with register tile transposes between passes
//! (as `codec_txfm::generic`). The `i32` instantiation is the scalar
//! reference.
//!
//! The 1-D transforms are even/odd butterflies over [`DCT_BASIS`] ("partial
//! butterflies"), one level per size `s = 2, 4, .., N` rather than one
//! function per size: the integer matrices nest (module docs of `av2_txfm`),
//! so the result is exactly AVM's matrix product. Each level's odd half is a
//! precomputed constant matrix, so every product has a fixed index and the
//! level loops unroll completely (indexing [`DCT_BASIS`] by a runtime level
//! was most of the instructions of the 32-point transforms).
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs): call them from a `#[target_feature]`
//! function of that ISA, or with the scalar `i32` lane type.

#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use core::mem::MaybeUninit;

use super::{DC_WEIGHT, DCT_BASIS, MAX_CODED_SIDE, MAX_POINTS};
use crate::codec_txfm::generic::{as_uninit, written};
use crate::simd::I32Lanes;
use crate::txfm::generic::store_pass;

/// Coded coefficients of the largest transform.
const MAX_CODED_AREA: usize = MAX_CODED_SIDE * MAX_CODED_SIDE;

/// Range of the inverse column pass output: the 8-bit residual range
/// (AVM `inv_txfm_c`: `±(1 << bd)`, upper bound exclusive).
const RESIDUAL_MIN: i32 = -256;
const RESIDUAL_MAX: i32 = 255;

/// The odd half of the `2H`-point matrix: row `r` is the first `H` samples
/// of basis vector `2r + 1` (the `N`-point vector `k` is 64-point row
/// `k · 64 / N`). Precomputed per size so that every product in the
/// butterflies below has a fixed index and the loops unroll completely.
const fn odd_matrix<const H: usize>() -> [[i32; H]; H] {
    let mut matrix = [[0; H]; H];
    let mut r = 0;
    while r < H {
        let mut j = 0;
        while j < H {
            matrix[r][j] = DCT_BASIS[(2 * r + 1) * (MAX_POINTS / (2 * H))][j] as i32;
            j += 1;
        }
        r += 1;
    }
    matrix
}

pub(super) const ODD2: [[i32; 1]; 1] = odd_matrix::<1>();
pub(super) const ODD4: [[i32; 2]; 2] = odd_matrix::<2>();
pub(super) const ODD8: [[i32; 4]; 4] = odd_matrix::<4>();
pub(super) const ODD16: [[i32; 8]; 8] = odd_matrix::<8>();
pub(super) const ODD32: [[i32; 16]; 16] = odd_matrix::<16>();
pub(super) const ODD64: [[i32; 32]; 32] = odd_matrix::<32>();

/// [`odd_matrix`] transposed: row `i` holds sample `i` of every odd basis
/// vector, the weights of one inverse output (contiguous for [`dot`]).
const fn transposed<const H: usize>(matrix: [[i32; H]; H]) -> [[i32; H]; H] {
    let mut out = [[0; H]; H];
    let mut r = 0;
    while r < H {
        let mut j = 0;
        while j < H {
            out[j][r] = matrix[r][j];
            j += 1;
        }
        r += 1;
    }
    out
}

const ODD2_T: [[i32; 1]; 1] = transposed(ODD2);
const ODD4_T: [[i32; 2]; 2] = transposed(ODD4);
const ODD8_T: [[i32; 4]; 4] = transposed(ODD8);
const ODD16_T: [[i32; 8]; 8] = transposed(ODD16);
const ODD32_T: [[i32; 16]; 16] = transposed(ODD32);

/// `Σ_j x[j] · weights[j]` over the shorter of the two: four products at a
/// time by [`I32Lanes::mul_add4`], the rest one by one. Wrapping `i32`
/// sums, so the grouping does not change the result.
#[inline(always)]
pub(super) fn dot<V: I32Lanes>(x: &[V], weights: &[i32]) -> V {
    let n = x.len().min(weights.len());
    let (x, weights) = (&x[..n], &weights[..n]);
    let (quads, x_rest) = x.as_chunks::<4>();
    let (weight_quads, weight_rest) = weights.as_chunks::<4>();
    let mut acc = V::splat(0);
    for (quad, weight_quad) in quads.iter().zip(weight_quads) {
        acc = acc.mul_add4(*quad, weight_quad);
    }
    for (value, &weight) in x_rest.iter().zip(weight_rest) {
        acc = acc.add(value.mul_scalar(weight));
    }
    acc
}

/// `live` rounded up to whole [`dot`] quads, at most `h`: the extra inputs
/// are zero, so their products add nothing, and a whole quad is one
/// multiply-accumulate by element on NEON.
#[inline(always)]
const fn quad_rounded(live: usize, h: usize) -> usize {
    let rounded = live.div_ceil(4) * 4;
    if rounded < h { rounded } else { h }
}

/// One level of [`inverse_dct`]: with `y[..H]` the `H`-point inverse of the
/// even coefficients, the odd coefficients `c[(2r + 1) · step]` give `O`,
/// and `y[i] = E[i] + O[i]`, `y[2H − 1 − i] = E[i] − O[i]` (AVM computes the
/// same sums as `a ± b` in `inv_txfm_dct2_size*_c`). Only the first
/// `live` odd inputs can be nonzero ([`live_odd_inputs`]); the products of
/// the others are skipped, exactly (they add zero), up to whole quads
/// ([`quad_rounded`]). `odd_transposed` is the level's [`transposed`] odd
/// matrix.
#[inline(always)]
fn inverse_level<V: I32Lanes, const H: usize>(
    odd_transposed: &[[i32; H]; H],
    c: &[V],
    step: usize,
    live: usize,
    y: &mut [V],
) {
    let odd_inputs: [V; H] = core::array::from_fn(|r| c[(2 * r + 1) * step]);
    let live = quad_rounded(live, H);
    for (i, weights) in odd_transposed.iter().enumerate() {
        let odd = dot(&odd_inputs[..live], weights);
        let even = y[i];
        y[i] = even.add(odd);
        y[2 * H - 1 - i] = even.sub(odd);
    }
}

/// How many of a level's `h` odd inputs `c[(2r + 1) · step]` lie below
/// `inputs` (the inputs from `inputs` on are zero): the odd multiples of
/// `step` below `inputs`.
#[inline(always)]
pub(super) const fn live_odd_inputs(h: usize, step: usize, inputs: usize) -> usize {
    if inputs <= step {
        return 0;
    }
    let live = ((inputs - 1) / step).div_ceil(2);
    if live < h { live } else { h }
}

/// `N`-point inverse DCT, `y = Mᵀ c`, built up level by level from the DC:
/// after the level of size `s`, `y[..s]` is the `s`-point inverse of the
/// even-subsampled coefficients `c[0], c[N/s], c[2N/s], ..`. Inputs from
/// `inputs` on are zero; a constant `inputs` (after inlining) drops their
/// products at compile time.
#[inline(always)]
fn inverse_dct<V: I32Lanes, const N: usize>(c: &[V; N], inputs: usize, y: &mut [V; N]) {
    y[0] = c[0].mul_scalar(DC_WEIGHT);
    inverse_level(&ODD2_T, c, N / 2, live_odd_inputs(1, N / 2, inputs), y);
    if N >= 4 {
        inverse_level(&ODD4_T, c, N / 4, live_odd_inputs(2, N / 4, inputs), y);
    }
    if N >= 8 {
        inverse_level(&ODD8_T, c, N / 8, live_odd_inputs(4, N / 8, inputs), y);
    }
    if N >= 16 {
        inverse_level(&ODD16_T, c, N / 16, live_odd_inputs(8, N / 16, inputs), y);
    }
    if N >= 32 {
        inverse_level(&ODD32_T, c, N / 32, live_odd_inputs(16, N / 32, inputs), y);
    }
}

/// One level of [`forward_dct`]: fold the `2H` working values into the sums
/// `x[i] + x[2H − 1 − i]` (the next, half-size level's input) and the
/// differences, whose products with the odd basis vectors are the outputs at
/// odd multiples of `step` below `outputs` (AVM's `a`/`b`, `c`/`d`, .. in
/// `fwd_txfm_dct2_size*_c`).
#[inline(always)]
fn forward_level<V: I32Lanes, const H: usize>(
    odd_matrix: &[[i32; H]; H],
    sums: &mut [V],
    step: usize,
    outputs: usize,
    y: &mut [V],
) {
    let mut differences = [V::splat(0); H];
    for (i, difference) in differences.iter_mut().enumerate() {
        let (a, b) = (sums[i], sums[2 * H - 1 - i]);
        sums[i] = a.add(b);
        *difference = a.sub(b);
    }
    for (r, row) in odd_matrix.iter().enumerate() {
        let k = (2 * r + 1) * step;
        if k >= outputs {
            break;
        }
        y[k] = dot(&differences, row);
    }
}

/// `N`-point forward DCT, `y = M x`, of which only outputs `y[..outputs]`
/// are computed (the rest are left unchanged): levels from `N` points down
/// to two, then the DC.
#[inline(always)]
fn forward_dct<V: I32Lanes, const N: usize>(x: &[V; N], y: &mut [V; N], outputs: usize) {
    let mut sums = *x;
    if N >= 64 {
        forward_level(&ODD64, &mut sums, N / 64, outputs, y);
    }
    if N >= 32 {
        forward_level(&ODD32, &mut sums, N / 32, outputs, y);
    }
    if N >= 16 {
        forward_level(&ODD16, &mut sums, N / 16, outputs, y);
    }
    if N >= 8 {
        forward_level(&ODD8, &mut sums, N / 8, outputs, y);
    }
    if N >= 4 {
        forward_level(&ODD4, &mut sums, N / 4, outputs, y);
    }
    forward_level(&ODD2, &mut sums, N / 2, outputs, y);
    y[0] = sums[0].mul_scalar(DC_WEIGHT);
}

/// Inverse 2-D DCT of side `C` (AVM `inv_txfm_c` without the pixel add):
/// row-major `coeffs` (`C * C`) to the row-major residual `out` (`C * C`).
/// Rows and columns from `nonzero` on must be all zero (`C` when nothing
/// is known): a row of zeros transforms to zeros, so those rows skip their
/// pass, and both passes skip the products of the zero inputs. Low-rate
/// blocks end early in the scan, so most of a large transform's
/// coefficients are zero.
///
/// # Safety
///
/// `V`'s ISA must be available (module docs); `V::LANES <= C`.
#[inline(always)]
pub(crate) unsafe fn inverse<V: I32Lanes, const C: usize>(
    coeffs: &[i32],
    nonzero: usize,
    shifts: (u32, u32),
    out: &mut [i32],
) {
    // Each arm inlines a copy of the passes with its input bound constant
    // (see `corner_bound`).
    // SAFETY: forwarded contract.
    unsafe {
        match corner_bound::<C>(nonzero) {
            CornerBound::Quarter => inverse_within::<V, C>(coeffs, nonzero, C / 4, shifts, out),
            CornerBound::Half => inverse_within::<V, C>(coeffs, nonzero, C / 2, shifts, out),
            CornerBound::Full => inverse_within::<V, C>(coeffs, nonzero, C, shifts, out),
        }
    }
}

/// The 1-D inverse passes are compiled for three input bounds: a corner
/// within a quarter or a half of the side, or any.
#[derive(Clone, Copy)]
pub(super) enum CornerBound {
    Quarter,
    Half,
    Full,
}

/// The smallest [`CornerBound`] of side `C` holding a `nonzero` corner.
#[inline(always)]
pub(super) const fn corner_bound<const C: usize>(nonzero: usize) -> CornerBound {
    if nonzero <= C / 4 {
        CornerBound::Quarter
    } else if nonzero <= C / 2 {
        CornerBound::Half
    } else {
        CornerBound::Full
    }
}

/// [`inverse`] with every 1-D pass's inputs from `inputs` (at least
/// `nonzero`) on known to be zero.
///
/// # Safety
///
/// As [`inverse`].
#[inline(always)]
unsafe fn inverse_within<V: I32Lanes, const C: usize>(
    coeffs: &[i32],
    nonzero: usize,
    inputs: usize,
    shifts: (u32, u32),
    out: &mut [i32],
) {
    debug_assert!(nonzero <= inputs);
    let lanes = V::LANES;
    let mut x = [V::splat(0); C];
    let mut y = [V::splat(0); C];
    // The input transposed, `t[k * C + r] = coeffs[r][k]`, so that the row
    // pass is lane-parallel; clamped to 16 bits (AVM `clamp_buf`). Tiles
    // outside the nonzero corner are stored as zeros without loading.
    let mut t = [MaybeUninit::<i32>::uninit(); MAX_CODED_AREA];
    for r0 in (0..C).step_by(lanes) {
        for k0 in (0..C).step_by(lanes) {
            for (i, lane) in x[..lanes].iter_mut().enumerate() {
                *lane = if r0 < nonzero && k0 < nonzero {
                    V::load(&coeffs[(r0 + i) * C + k0..]).clamp16()
                } else {
                    V::splat(0)
                };
            }
            V::store_transposed(&x[..lanes], &mut t[k0 * C + r0..], C);
        }
    }
    // SAFETY: every (r0, k0) tile of the C x C buffer was written above.
    let t = unsafe { written(&t, C * C) };
    // The row pass's output, row-major.
    let mut m = [MaybeUninit::<i32>::uninit(); MAX_CODED_AREA];
    for r0 in (0..C).step_by(lanes) {
        if r0 < nonzero {
            for (k, lane) in x.iter_mut().enumerate() {
                *lane = V::load(&t[k * C + r0..]);
            }
            inverse_dct(&x, inputs, &mut y);
            for lane in &mut y {
                *lane = lane.round_shift(shifts.0).clamp16();
            }
        } else {
            y = [V::splat(0); C];
        }
        store_pass::<V>(&y, r0, C, &mut m);
    }
    // SAFETY: rows r0..r0 + LANES of `m` were written for every r0.
    let m = unsafe { written(&m, C * C) };
    let (low, high) = (V::splat(RESIDUAL_MIN), V::splat(RESIDUAL_MAX));
    for x0 in (0..C).step_by(lanes) {
        for (r, lane) in x.iter_mut().enumerate() {
            *lane = V::load(&m[r * C + x0..]);
        }
        // Rows of `m` from `nonzero` on are zero (zero coefficient rows).
        inverse_dct(&x, inputs, &mut y);
        for (r, lane) in y.iter().enumerate() {
            let sample = lane.round_shift(shifts.1).max(low).min(high);
            sample.store(&mut out[r * C + x0..]);
        }
    }
}

/// Inverse 64x64 DCT: the 32x32 inverse of the coded coefficients, each
/// residual sample repeated in a 2x2 square (AVM `inv_txfm_c` for sides
/// above `MAX_TX_SIZE >> 1`).
///
/// # Safety
///
/// As [`inverse`].
#[inline(always)]
pub(crate) unsafe fn inverse64<V: I32Lanes>(
    coeffs: &[i32],
    nonzero: usize,
    shifts: (u32, u32),
    out: &mut [i32],
) {
    const SIDE: usize = 2 * MAX_CODED_SIDE;
    let mut coded = [0; MAX_CODED_AREA];
    // SAFETY: forwarded contract.
    unsafe { inverse::<V, MAX_CODED_SIDE>(coeffs, nonzero, shifts, &mut coded) };
    let (pairs, _) = out[..SIDE * SIDE].as_chunks_mut::<{ 2 * SIDE }>();
    let (rows, _) = coded.as_chunks::<MAX_CODED_SIDE>();
    for (pair, row) in pairs.iter_mut().zip(rows) {
        let (upper, lower) = pair.split_at_mut(SIDE);
        let (samples, _) = upper.as_chunks_mut::<2>();
        for (sample, &value) in samples.iter_mut().zip(row) {
            *sample = [value; 2];
        }
        lower.copy_from_slice(upper);
    }
}

/// Forward 2-D DCT of side `N` (AVM `fwd_txfm_c`) of the residual at
/// `stride` into the row-major coded coefficients `out` (`C * C`, `C =
/// min(N, 32)`).
///
/// # Safety
///
/// `V`'s ISA must be available (module docs); `V::LANES <= C`.
#[inline(always)]
pub(crate) unsafe fn forward<V: I32Lanes, const N: usize, const C: usize>(
    residual: &[i16],
    stride: usize,
    shifts: (u32, u32),
    out: &mut [i32],
) {
    let lanes = V::LANES;
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    // The column pass's coded outputs, column-major: `g[col * C + k]`.
    let mut g = [MaybeUninit::<i32>::uninit(); MAX_POINTS * MAX_CODED_SIDE];
    for x0 in (0..N).step_by(lanes) {
        for (r, lane) in x.iter_mut().enumerate() {
            *lane = V::load_i16(&residual[r * stride + x0..]);
        }
        forward_dct(&x, &mut y, C);
        for lane in &mut y[..C] {
            *lane = lane.round_shift(shifts.0);
        }
        store_pass::<V>(&y[..C], x0, C, &mut g);
    }
    // SAFETY: rows x0..x0 + LANES of `g` were written for every x0.
    let g = unsafe { written(&g, N * C) };
    let out = as_uninit(&mut out[..C * C]);
    for k0 in (0..C).step_by(lanes) {
        for (column, lane) in x.iter_mut().enumerate() {
            *lane = V::load(&g[column * C + k0..]);
        }
        forward_dct(&x, &mut y, C);
        for lane in &mut y[..C] {
            *lane = lane.round_shift(shifts.1);
        }
        store_pass::<V>(&y[..C], k0, C, out);
    }
}

/// Instantiates the AV2 kernels for one ISA: `$v4` lanes for 4x4, `$v`
/// otherwise.
macro_rules! av2_txfm_kernels {
    ($(#[$attr:meta])*; $v4:ty, $v:ty) => {
        use $crate::av2_txfm::generic as g;
        use $crate::av2_txfm::{Av2TxSize, Av2TxfmFns};

        /// The AV2 transform table of this ISA.
        pub(crate) fn av2_txfm_fns() -> Av2TxfmFns {
            Av2TxfmFns {
                forward: [forward4, forward8, forward16, forward32, forward64],
                inverse: [inverse4, inverse8, inverse16, inverse32, inverse64],
            }
        }

        $crate::av2_txfm::generic::av2_txfm_kernels!(@size [$(#[$attr])*]
            forward4, inverse4, $v4, Tx4, 4);
        $crate::av2_txfm::generic::av2_txfm_kernels!(@size [$(#[$attr])*]
            forward8, inverse8, $v, Tx8, 8);
        $crate::av2_txfm::generic::av2_txfm_kernels!(@size [$(#[$attr])*]
            forward16, inverse16, $v, Tx16, 16);
        $crate::av2_txfm::generic::av2_txfm_kernels!(@size [$(#[$attr])*]
            forward32, inverse32, $v, Tx32, 32);

        $(#[$attr])*
        unsafe fn forward64(residual: &[i16], stride: usize, out: &mut [i32]) {
            // SAFETY: see the module docs.
            unsafe {
                g::forward::<$v, 64, 32>(residual, stride, Av2TxSize::Tx64.forward_shifts(), out)
            }
        }

        $(#[$attr])*
        unsafe fn inverse64(coeffs: &[i32], nonzero: usize, out: &mut [i32]) {
            // SAFETY: see the module docs.
            unsafe { g::inverse64::<$v>(coeffs, nonzero, Av2TxSize::Tx64.inverse_shifts(), out) }
        }
    };
    (@size [$(#[$attr:meta])*]
     $forward:ident, $inverse:ident, $lanes:ty, $size:ident, $n:literal) => {
        $(#[$attr])*
        unsafe fn $forward(residual: &[i16], stride: usize, out: &mut [i32]) {
            // SAFETY: see the module docs.
            unsafe {
                g::forward::<$lanes, $n, $n>(
                    residual, stride, Av2TxSize::$size.forward_shifts(), out,
                )
            }
        }

        $(#[$attr])*
        unsafe fn $inverse(coeffs: &[i32], nonzero: usize, out: &mut [i32]) {
            // SAFETY: see the module docs.
            unsafe {
                g::inverse::<$lanes, $n>(coeffs, nonzero, Av2TxSize::$size.inverse_shifts(), out)
            }
        }
    };
}

pub(crate) use av2_txfm_kernels;
