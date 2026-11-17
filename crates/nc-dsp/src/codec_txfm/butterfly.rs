//! The integer-matrix DCTs of HEVC (§8.6.4.2) and AV2 (AVM
//! `inv_txfm_dct2_size*_c`, `fwd_txfm_dct2_size*_c`) written once over
//! [`I32Lanes`]: even/odd butterflies ("partial butterflies"), one level
//! per size `s = 2, 4, .., N`, parameterised by the codec's basis
//! ([`DctBasis`]). Each transforms `V::LANES` lines at once.
//!
//! The two standards' matrices have the same structure (row `k` of the
//! `N`-point matrix is row `k · M / N` of the largest one, the second half of
//! every row mirrors the first) and differ only in their entries (HEVC's
//! 32-point table, AV2's 64-point one: e.g. 36 against 35 in the 4-point
//! odd half), so the result is exactly each codec's matrix product. Each
//! level's odd half is a precomputed constant matrix, so every product has
//! a fixed index and the level loops unroll completely (indexing the basis
//! by a runtime level was most of the instructions of AV2's 32-point
//! transforms). Wrapping `i32` sums, so the grouping of products does not
//! change the result.
//!
//! The pass drivers (shifts, clamps, transposes) stay with each codec:
//! HEVC in `generic`, AV2 in `av2_txfm::generic`.

#![allow(clippy::inline_always)]

use crate::simd::I32Lanes;

/// Samples of the first half of every basis vector the tables hold.
pub(crate) const HALF_POINTS: usize = 32;

/// The odd half of the `2H`-point matrix taken from `basis`, whose row `k`
/// is the first half of vector `k` of the `ROWS`-point matrix: row `r` is
/// the first `H` samples of basis vector `2r + 1` (the `2H`-point vector
/// `k` is row `k · ROWS / 2H`).
pub(crate) const fn odd_matrix<const H: usize, const ROWS: usize>(
    basis: &[[i8; HALF_POINTS]; ROWS],
) -> [[i32; H]; H] {
    let mut matrix = [[0; H]; H];
    let mut r = 0;
    while r < H {
        let mut j = 0;
        while j < H {
            matrix[r][j] = basis[(2 * r + 1) * (ROWS / (2 * H))][j] as i32;
            j += 1;
        }
        r += 1;
    }
    matrix
}

/// `matrix` transposed: row `i` holds sample `i` of every odd basis vector,
/// the weights of one inverse output (contiguous for [`dot`]).
pub(crate) const fn transposed<const H: usize>(matrix: [[i32; H]; H]) -> [[i32; H]; H] {
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

/// A codec's DCT basis: the DC weight and each level's odd matrix
/// ([`odd_matrix`]).
pub(crate) trait DctBasis {
    /// Weight of the DC basis vector.
    const DC: i32;
    const ODD2: [[i32; 1]; 1];
    const ODD4: [[i32; 2]; 2];
    const ODD8: [[i32; 4]; 4];
    const ODD16: [[i32; 8]; 8];
    const ODD32: [[i32; 16]; 16];
    /// The 64-point level (forward only); `None` without a 64-point DCT.
    const ODD64: Option<[[i32; 32]; 32]>;
    const ODD2_T: [[i32; 1]; 1] = transposed(Self::ODD2);
    const ODD4_T: [[i32; 2]; 2] = transposed(Self::ODD4);
    const ODD8_T: [[i32; 4]; 4] = transposed(Self::ODD8);
    const ODD16_T: [[i32; 8]; 8] = transposed(Self::ODD16);
    const ODD32_T: [[i32; 16]; 16] = transposed(Self::ODD32);
}

/// `Σ_j x[j] · weights[j]` over the shorter of the two: four products at a
/// time by [`I32Lanes::mul_add4`], the rest one by one.
#[inline(always)]
pub(crate) fn dot<V: I32Lanes>(x: &[V], weights: &[i32]) -> V {
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

/// One level of [`inverse_dct`]: with `y[..H]` the `H`-point inverse of the
/// even coefficients, the odd coefficients `c[(2r + 1) · step]` give `O`,
/// and `y[i] = E[i] + O[i]`, `y[2H − 1 − i] = E[i] − O[i]`. Only the first
/// `live` odd inputs can be nonzero ([`live_odd_inputs`]); the products of
/// the others are skipped, exactly (they add zero). `odd_transposed` is
/// the level's [`transposed`] odd matrix.
///
/// The live inputs are not padded to whole [`dot`] quads: on NEON the
/// padding (a level with 2 live inputs running 4 products) cost the sparse
/// HEVC inverses 10-14 % (Jetson `nc-checkasm --bench`, 8x8 and 16x16
/// `low4`, 16x16 `split_dc`), and dropping it left AV2 within noise.
#[inline(always)]
fn inverse_level<V: I32Lanes, const H: usize>(
    odd_transposed: &[[i32; H]; H],
    c: &[V],
    step: usize,
    live: usize,
    y: &mut [V],
) {
    // A loop, not `array::from_fn`: that stayed out of line, outside the
    // caller's `#[target_feature]` context.
    let mut odd_inputs = [V::splat(0); H];
    for (r, input) in odd_inputs.iter_mut().enumerate() {
        *input = c[(2 * r + 1) * step];
    }
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
pub(crate) const fn live_odd_inputs(h: usize, step: usize, inputs: usize) -> usize {
    if inputs <= step {
        return 0;
    }
    let live = ((inputs - 1) / step).div_ceil(2);
    if live < h { live } else { h }
}

/// `N`-point inverse DCT (`N` 2 to 32), `y = Mᵀ c`, built up level by level
/// from the DC: after the level of size `s`, `y[..s]` is the `s`-point
/// inverse of the even-subsampled coefficients `c[0], c[N/s], c[2N/s], ..`.
/// Inputs from `inputs` on are zero; a constant `inputs` (after inlining)
/// drops their products at compile time.
#[inline(always)]
pub(crate) fn inverse_dct<B: DctBasis, V: I32Lanes, const N: usize>(
    c: &[V; N],
    inputs: usize,
    y: &mut [V; N],
) {
    y[0] = c[0].mul_scalar(B::DC);
    inverse_level(&B::ODD2_T, c, N / 2, live_odd_inputs(1, N / 2, inputs), y);
    if N >= 4 {
        inverse_level(&B::ODD4_T, c, N / 4, live_odd_inputs(2, N / 4, inputs), y);
    }
    if N >= 8 {
        inverse_level(&B::ODD8_T, c, N / 8, live_odd_inputs(4, N / 8, inputs), y);
    }
    if N >= 16 {
        inverse_level(
            &B::ODD16_T,
            c,
            N / 16,
            live_odd_inputs(8, N / 16, inputs),
            y,
        );
    }
    if N >= 32 {
        inverse_level(
            &B::ODD32_T,
            c,
            N / 32,
            live_odd_inputs(16, N / 32, inputs),
            y,
        );
    }
}

/// One level of [`forward_dct`]: fold the `2H` working values into the sums
/// `x[i] + x[2H − 1 − i]` (the next, half-size level's input) and the
/// differences, whose products with the odd basis vectors are the outputs at
/// odd multiples of `step` below `outputs`.
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

/// `N`-point forward DCT (`N` 2 to 64), `y = M x`, of which only outputs
/// `y[..outputs]` are computed (the rest are left unchanged): levels from
/// `N` points down to two, then the DC.
#[inline(always)]
pub(crate) fn forward_dct<B: DctBasis, V: I32Lanes, const N: usize>(
    x: &[V; N],
    y: &mut [V; N],
    outputs: usize,
) {
    let mut sums = *x;
    if N >= 64 {
        let Some(odd64) = &B::ODD64 else {
            unreachable!("no 64-point DCT in this basis")
        };
        forward_level(odd64, &mut sums, N / 64, outputs, y);
    }
    if N >= 32 {
        forward_level(&B::ODD32, &mut sums, N / 32, outputs, y);
    }
    if N >= 16 {
        forward_level(&B::ODD16, &mut sums, N / 16, outputs, y);
    }
    if N >= 8 {
        forward_level(&B::ODD8, &mut sums, N / 8, outputs, y);
    }
    if N >= 4 {
        forward_level(&B::ODD4, &mut sums, N / 4, outputs, y);
    }
    forward_level(&B::ODD2, &mut sums, N / 2, outputs, y);
    y[0] = sums[0].mul_scalar(B::DC);
}
