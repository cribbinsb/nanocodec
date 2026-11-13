//! AVX2 AV2 DCTs whose odd-half products run on 16-bit pairs (`vpmaddwd`):
//! one instruction per two multiply-adds instead of a 32-bit multiply and an
//! add per product. Mirrors the butterflies and drivers of [`super::generic`]
//! (the scalar reference) exactly; only the product grouping differs, which
//! is exact because every product sum stays in `i32` and each paired value
//! fits `i16` (ranges below). Passes whose inputs can exceed `i16` keep
//! 32-bit products.
//!
//! # Ranges (16-bit pairs)
//!
//! - Inverse, both passes: the odd inputs are coefficients clamped to 16 bits
//!   on load (row pass) or row outputs clamped to 16 bits (column pass).
//! - Forward column pass: residuals have `|r| <= 255`. The level of size `2H`
//!   folds sums of `N / (2H)` residuals, so its differences are at most
//!   `255 · N / H <= 255 · 64 = 16320`.
//! - Forward row pass: its inputs reach `255 · 5760 >> 3` (module docs of
//!   `av2_txfm`), beyond `i16`, so it uses 32-bit products.
//!
//! Matrix entries are at most 90 in magnitude, so every weight fits `i16`.

#![allow(clippy::inline_always)]

use core::mem::MaybeUninit;

use super::generic::{
    CornerBound, ODD2, ODD4, ODD8, ODD16, ODD32, ODD64, corner_bound, live_odd_inputs,
};
use super::{DC_WEIGHT, MAX_CODED_SIDE, MAX_POINTS};
use crate::codec_txfm::generic::{as_uninit, written};
use crate::simd::I32Lanes;
use crate::simd::avx2::{I32x8, WordPairs8};
use crate::txfm::generic::store_pass;

/// Lanes per vector.
const LANES: usize = 8;
/// Coded coefficients of the largest transform.
const MAX_CODED_AREA: usize = MAX_CODED_SIDE * MAX_CODED_SIDE;
/// Range of the inverse column pass output (as `generic`).
const RESIDUAL_MIN: i32 = -256;
const RESIDUAL_MAX: i32 = 255;

/// Two matrix entries as one `vpmaddwd` weight pair ([`WordPairs8::dot`]);
/// every entry is within ±90, so the `i16` casts are exact.
const fn pack(w0: i32, w1: i32) -> i32 {
    ((w1 as i16 as i32) << 16) | (w0 as i16 as u16 as i32)
}

/// Packed weights of the inverse level of `odd_matrix`: entry `[i][p]` pairs
/// rows `2p` and `2p + 1` at sample `i` (columns past `H / 2` unused).
const fn inverse_weights<const H: usize>(odd_matrix: &[[i32; H]; H]) -> [[i32; H]; H] {
    let mut packed = [[0; H]; H];
    let mut i = 0;
    while i < H {
        let mut p = 0;
        while 2 * p + 1 < H {
            packed[i][p] = pack(odd_matrix[2 * p][i], odd_matrix[2 * p + 1][i]);
            p += 1;
        }
        i += 1;
    }
    packed
}

/// Packed weights of the forward level of `odd_matrix`: entry `[r][p]` pairs
/// samples `2p` and `2p + 1` of row `r`.
const fn forward_weights<const H: usize>(odd_matrix: &[[i32; H]; H]) -> [[i32; H]; H] {
    let mut packed = [[0; H]; H];
    let mut r = 0;
    while r < H {
        let mut p = 0;
        while 2 * p + 1 < H {
            packed[r][p] = pack(odd_matrix[r][2 * p], odd_matrix[r][2 * p + 1]);
            p += 1;
        }
        r += 1;
    }
    packed
}

const INVERSE4: [[i32; 2]; 2] = inverse_weights(&ODD4);
const INVERSE8: [[i32; 4]; 4] = inverse_weights(&ODD8);
const INVERSE16: [[i32; 8]; 8] = inverse_weights(&ODD16);
const INVERSE32: [[i32; 16]; 16] = inverse_weights(&ODD32);
const FORWARD4: [[i32; 2]; 2] = forward_weights(&ODD4);
const FORWARD8: [[i32; 4]; 4] = forward_weights(&ODD8);
const FORWARD16: [[i32; 8]; 8] = forward_weights(&ODD16);
const FORWARD32: [[i32; 16]; 16] = forward_weights(&ODD32);
const FORWARD64: [[i32; 32]; 32] = forward_weights(&ODD64);

/// `Σ_p pairs[p] · weights[p]` over the first `live` pairs (at most
/// `H / 2`).
#[inline(always)]
fn paired_dot<const H: usize>(pairs: &[WordPairs8; H], weights: &[i32; H], live: usize) -> I32x8 {
    let mut acc = I32x8::splat(0);
    for p in 0..live {
        acc = acc.add(pairs[p].dot(weights[p]));
    }
    acc
}

/// Pairs `values[2p]` with `values[2p + 1]` (`H` even; slots past `H / 2`
/// are unused).
#[inline(always)]
fn pair_up<const H: usize>(values: &[I32x8; H]) -> [WordPairs8; H] {
    let zero = I32x8::splat(0).word_pairs(I32x8::splat(0));
    core::array::from_fn(|p| {
        if 2 * p + 1 < H {
            values[2 * p].word_pairs(values[2 * p + 1])
        } else {
            zero
        }
    })
}

/// Unused packed weights of the single-input first level (32-bit product).
const NO_PAIRS: [[i32; 1]; 1] = [[0]];

/// `generic::inverse_level` with paired products (`H >= 2`; `packed` from
/// [`inverse_weights`]), or a 32-bit product for the single-input level;
/// only the first `live` odd inputs can be nonzero.
#[inline(always)]
fn inverse_level<const H: usize>(
    odd_matrix: &[[i32; H]; H],
    packed: &[[i32; H]; H],
    c: &[I32x8],
    step: usize,
    live: usize,
    y: &mut [I32x8],
) {
    let odd_inputs: [I32x8; H] = core::array::from_fn(|r| c[(2 * r + 1) * step]);
    let pairs = pair_up(&odd_inputs);
    for i in 0..H {
        let odd = if live == 0 {
            I32x8::splat(0)
        } else if H == 1 {
            odd_inputs[0].mul_scalar(odd_matrix[0][i])
        } else {
            paired_dot(&pairs, &packed[i], live.div_ceil(2))
        };
        let even = y[i];
        y[i] = even.add(odd);
        y[2 * H - 1 - i] = even.sub(odd);
    }
}

/// `generic::inverse_dct` on paired levels, inputs from `inputs` on zero.
#[inline(always)]
fn inverse_dct<const N: usize>(c: &[I32x8; N], inputs: usize, y: &mut [I32x8; N]) {
    y[0] = c[0].mul_scalar(DC_WEIGHT);
    let live = |h, step| live_odd_inputs(h, step, inputs);
    inverse_level(&ODD2, &NO_PAIRS, c, N / 2, live(1, N / 2), y);
    if N >= 4 {
        inverse_level(&ODD4, &INVERSE4, c, N / 4, live(2, N / 4), y);
    }
    if N >= 8 {
        inverse_level(&ODD8, &INVERSE8, c, N / 8, live(4, N / 8), y);
    }
    if N >= 16 {
        inverse_level(&ODD16, &INVERSE16, c, N / 16, live(8, N / 16), y);
    }
    if N >= 32 {
        inverse_level(&ODD32, &INVERSE32, c, N / 32, live(16, N / 32), y);
    }
}

/// `generic::forward_level`, with paired products when `PAIRED` and `H >= 2`
/// (`packed` from [`forward_weights`]).
#[inline(always)]
fn forward_level<const H: usize, const PAIRED: bool>(
    odd_matrix: &[[i32; H]; H],
    packed: &[[i32; H]; H],
    sums: &mut [I32x8],
    step: usize,
    outputs: usize,
    y: &mut [I32x8],
) {
    let mut differences = [I32x8::splat(0); H];
    for (i, difference) in differences.iter_mut().enumerate() {
        let (a, b) = (sums[i], sums[2 * H - 1 - i]);
        sums[i] = a.add(b);
        *difference = a.sub(b);
    }
    let paired = PAIRED && H >= 2;
    let pairs = pair_up(&differences);
    for (r, row) in odd_matrix.iter().enumerate() {
        let k = (2 * r + 1) * step;
        if k >= outputs {
            break;
        }
        y[k] = if paired {
            paired_dot(&pairs, &packed[r], H / 2)
        } else {
            let mut acc = differences[0].mul_scalar(row[0]);
            for j in 1..H {
                acc = acc.add(differences[j].mul_scalar(row[j]));
            }
            acc
        };
    }
}

/// `generic::forward_dct`, paired when `PAIRED`.
#[inline(always)]
fn forward_dct<const N: usize, const PAIRED: bool>(
    x: &[I32x8; N],
    y: &mut [I32x8; N],
    outputs: usize,
) {
    let mut sums = *x;
    if N >= 64 {
        forward_level::<32, PAIRED>(&ODD64, &FORWARD64, &mut sums, N / 64, outputs, y);
    }
    if N >= 32 {
        forward_level::<16, PAIRED>(&ODD32, &FORWARD32, &mut sums, N / 32, outputs, y);
    }
    if N >= 16 {
        forward_level::<8, PAIRED>(&ODD16, &FORWARD16, &mut sums, N / 16, outputs, y);
    }
    if N >= 8 {
        forward_level::<4, PAIRED>(&ODD8, &FORWARD8, &mut sums, N / 8, outputs, y);
    }
    if N >= 4 {
        forward_level::<2, PAIRED>(&ODD4, &FORWARD4, &mut sums, N / 4, outputs, y);
    }
    forward_level::<1, PAIRED>(&ODD2, &NO_PAIRS, &mut sums, N / 2, outputs, y);
    y[0] = sums[0].mul_scalar(DC_WEIGHT);
}

/// `generic::inverse` on paired butterflies (`C >= 8`), skipping as it
/// does the row groups from `nonzero` on (all-zero coefficient rows) and
/// the products of zero inputs.
///
/// # Safety
///
/// AVX2 must be available.
#[inline(always)]
pub(crate) unsafe fn inverse<const C: usize>(
    coeffs: &[i32],
    nonzero: usize,
    shifts: (u32, u32),
    out: &mut [i32],
) {
    // SAFETY: forwarded contract.
    unsafe {
        match corner_bound::<C>(nonzero) {
            CornerBound::Quarter => inverse_within::<C>(coeffs, nonzero, C / 4, shifts, out),
            CornerBound::Half => inverse_within::<C>(coeffs, nonzero, C / 2, shifts, out),
            CornerBound::Full => inverse_within::<C>(coeffs, nonzero, C, shifts, out),
        }
    }
}

/// [`inverse`] with every 1-D pass's inputs from `inputs` (at least
/// `nonzero`) on known to be zero.
///
/// # Safety
///
/// As [`inverse`].
#[inline(always)]
unsafe fn inverse_within<const C: usize>(
    coeffs: &[i32],
    nonzero: usize,
    inputs: usize,
    shifts: (u32, u32),
    out: &mut [i32],
) {
    debug_assert!(nonzero <= inputs);
    let mut x = [I32x8::splat(0); C];
    let mut y = [I32x8::splat(0); C];
    let mut t = [MaybeUninit::<i32>::uninit(); MAX_CODED_AREA];
    for r0 in (0..C).step_by(LANES) {
        for k0 in (0..C).step_by(LANES) {
            for (i, lane) in x[..LANES].iter_mut().enumerate() {
                *lane = if r0 < nonzero && k0 < nonzero {
                    I32x8::load(&coeffs[(r0 + i) * C + k0..]).clamp16()
                } else {
                    I32x8::splat(0)
                };
            }
            I32x8::store_transposed(&x[..LANES], &mut t[k0 * C + r0..], C);
        }
    }
    // SAFETY: every (r0, k0) tile of the C x C buffer was written above.
    let t = unsafe { written(&t, C * C) };
    let mut m = [MaybeUninit::<i32>::uninit(); MAX_CODED_AREA];
    for r0 in (0..C).step_by(LANES) {
        if r0 < nonzero {
            for (k, lane) in x.iter_mut().enumerate() {
                *lane = I32x8::load(&t[k * C + r0..]);
            }
            inverse_dct(&x, inputs, &mut y);
            for lane in &mut y {
                *lane = lane.round_shift(shifts.0).clamp16();
            }
        } else {
            y = [I32x8::splat(0); C];
        }
        store_pass::<I32x8>(&y, r0, C, &mut m);
    }
    // SAFETY: rows r0..r0 + LANES of `m` were written for every r0.
    let m = unsafe { written(&m, C * C) };
    let (low, high) = (I32x8::splat(RESIDUAL_MIN), I32x8::splat(RESIDUAL_MAX));
    for x0 in (0..C).step_by(LANES) {
        for (r, lane) in x.iter_mut().enumerate() {
            *lane = I32x8::load(&m[r * C + x0..]);
        }
        inverse_dct(&x, inputs, &mut y);
        for (r, lane) in y.iter().enumerate() {
            let sample = lane.round_shift(shifts.1).max(low).min(high);
            sample.store(&mut out[r * C + x0..]);
        }
    }
}

/// `generic::forward`: paired column pass, 32-bit row pass (`N >= 8`).
///
/// # Safety
///
/// AVX2 must be available.
#[inline(always)]
pub(crate) unsafe fn forward<const N: usize, const C: usize>(
    residual: &[i16],
    stride: usize,
    shifts: (u32, u32),
    out: &mut [i32],
) {
    let mut x = [I32x8::splat(0); N];
    let mut y = [I32x8::splat(0); N];
    let mut g = [MaybeUninit::<i32>::uninit(); MAX_POINTS * MAX_CODED_SIDE];
    for x0 in (0..N).step_by(LANES) {
        for (r, lane) in x.iter_mut().enumerate() {
            *lane = I32x8::load_i16(&residual[r * stride + x0..]);
        }
        forward_dct::<N, true>(&x, &mut y, C);
        for lane in &mut y[..C] {
            *lane = lane.round_shift(shifts.0);
        }
        store_pass::<I32x8>(&y[..C], x0, C, &mut g);
    }
    // SAFETY: rows x0..x0 + LANES of `g` were written for every x0.
    let g = unsafe { written(&g, N * C) };
    let out = as_uninit(&mut out[..C * C]);
    for k0 in (0..C).step_by(LANES) {
        for (column, lane) in x.iter_mut().enumerate() {
            *lane = I32x8::load(&g[column * C + k0..]);
        }
        forward_dct::<N, false>(&x, &mut y, C);
        for lane in &mut y[..C] {
            *lane = lane.round_shift(shifts.1);
        }
        store_pass::<I32x8>(&y[..C], k0, C, out);
    }
}
