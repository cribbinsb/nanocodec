//! AV2 DCTs whose odd-half products run on 16-bit pairs ([`WordPairLanes`]:
//! AVX2 `vpmaddwd`, SIMD128 `i32x4.dot_i16x8_s`): one instruction per two
//! multiply-adds instead of a 32-bit multiply and an add per product.
//! Mirrors the butterflies and drivers of [`super::generic`] (the scalar
//! reference) exactly; only the product grouping differs, which is exact
//! because every product sum stays in `i32` and each paired value fits `i16`
//! (ranges below). Passes whose inputs can exceed `i16` keep 32-bit
//! products. The inverse serves AVX2 and SIMD128; the forward, AVX2 only so
//! far (the SIMD128 forward is the generic one).
//!
//! # Ranges (16-bit pairs)
//!
//! - Inverse, both passes: the odd inputs are 16-bit coefficients (row
//!   pass) or row outputs clamped to 16 bits (column pass).
//! - Forward column pass: residuals have `|r| <= 255`. The level of size `2H`
//!   folds sums of `N / (2H)` residuals, so its differences are at most
//!   `255 · N / H <= 255 · 64 = 16320`.
//! - Forward row pass: its inputs reach `255 · 5760 >> 3` (module docs of
//!   `av2_txfm`), beyond `i16`, so it uses 32-bit products.
//!
//! Matrix entries are at most 90 in magnitude, so every weight fits `i16`.

#![allow(clippy::inline_always)]

use core::mem::MaybeUninit;

use super::generic::{Av2Basis, CornerBound, RESIDUAL_MAX, RESIDUAL_MIN, corner_bound};
use super::{DC_WEIGHT, MAX_CODED_SIDE};
use crate::codec_txfm::butterfly::{DctBasis, live_odd_inputs};
use crate::codec_txfm::generic::written;
use crate::simd::WordPairLanes;
use crate::txfm::generic::store_pass_sat16;

#[cfg(target_arch = "x86_64")]
pub(crate) use forward_pass::forward;

/// Each level's odd matrix (`codec_txfm::butterfly`).
const ODD2: [[i32; 1]; 1] = Av2Basis::ODD2;
const ODD4: [[i32; 2]; 2] = Av2Basis::ODD4;
const ODD8: [[i32; 4]; 4] = Av2Basis::ODD8;
const ODD16: [[i32; 8]; 8] = Av2Basis::ODD16;
const ODD32: [[i32; 16]; 16] = Av2Basis::ODD32;

/// Coded coefficients of the largest transform.
const MAX_CODED_AREA: usize = MAX_CODED_SIDE * MAX_CODED_SIDE;

/// Two matrix entries as one packed weight pair
/// ([`WordPairLanes::pair_dot`]); every entry is within ±90, so the `i16`
/// casts are exact.
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

const INVERSE4: [[i32; 2]; 2] = inverse_weights(&ODD4);
const INVERSE8: [[i32; 4]; 4] = inverse_weights(&ODD8);
const INVERSE16: [[i32; 8]; 8] = inverse_weights(&ODD16);
const INVERSE32: [[i32; 16]; 16] = inverse_weights(&ODD32);

/// `Σ_p pairs[p] · weights[p]` over the first `live` pairs (at most
/// `H / 2`).
#[inline(always)]
fn paired_dot<V: WordPairLanes, const H: usize>(
    pairs: &[V::Pairs; H],
    weights: &[i32; H],
    live: usize,
) -> V {
    let mut acc = V::splat(0);
    for p in 0..live {
        acc = acc.add(V::pair_dot(pairs[p], weights[p]));
    }
    acc
}

/// Pairs `values[2p]` with `values[2p + 1]` (`H` even; slots past `H / 2`
/// are unused).
#[inline(always)]
fn pair_up<V: WordPairLanes, const H: usize>(values: &[V; H]) -> [V::Pairs; H] {
    let zero = V::splat(0).word_pairs(V::splat(0));
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
fn inverse_level<V: WordPairLanes, const H: usize>(
    odd_matrix: &[[i32; H]; H],
    packed: &[[i32; H]; H],
    c: &[V],
    step: usize,
    live: usize,
    y: &mut [V],
) {
    let odd_inputs: [V; H] = core::array::from_fn(|r| c[(2 * r + 1) * step]);
    let pairs = pair_up(&odd_inputs);
    for i in 0..H {
        let odd = if live == 0 {
            V::splat(0)
        } else if H == 1 {
            odd_inputs[0].mul_scalar(odd_matrix[0][i])
        } else {
            paired_dot::<V, H>(&pairs, &packed[i], live.div_ceil(2))
        };
        let even = y[i];
        y[i] = even.add(odd);
        y[2 * H - 1 - i] = even.sub(odd);
    }
}

/// `butterfly::inverse_dct` on paired levels, inputs from `inputs` on zero.
#[inline(always)]
fn inverse_dct<V: WordPairLanes, const N: usize>(c: &[V; N], inputs: usize, y: &mut [V; N]) {
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

/// `generic::inverse` on paired butterflies, skipping as it does the row
/// groups from `nonzero` on (all-zero coefficient rows) and the products of
/// zero inputs.
///
/// # Safety
///
/// `V`'s ISA must be available; `V::LANES <= C`.
#[inline(always)]
pub(crate) unsafe fn inverse<V: WordPairLanes, const C: usize>(
    coeffs: &[i16],
    nonzero: usize,
    shifts: (u32, u32),
    out: &mut [i16],
) {
    // SAFETY: forwarded contract.
    unsafe {
        match corner_bound::<C>(nonzero) {
            CornerBound::Quarter => inverse_within::<V, C>(coeffs, nonzero, C / 4, shifts, out),
            CornerBound::Half => inverse_within::<V, C>(coeffs, nonzero, C / 2, shifts, out),
            CornerBound::Full => inverse_within::<V, C>(coeffs, nonzero, C, shifts, out),
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
unsafe fn inverse_within<V: WordPairLanes, const C: usize>(
    coeffs: &[i16],
    nonzero: usize,
    inputs: usize,
    shifts: (u32, u32),
    out: &mut [i16],
) {
    debug_assert!(nonzero <= inputs);
    let lanes = V::LANES;
    let mut x = [V::splat(0); C];
    let mut y = [V::splat(0); C];
    let mut t = [MaybeUninit::<i16>::uninit(); MAX_CODED_AREA];
    for r0 in (0..C).step_by(lanes) {
        for k0 in (0..C).step_by(lanes) {
            for (i, lane) in x[..lanes].iter_mut().enumerate() {
                *lane = if r0 < nonzero && k0 < nonzero {
                    V::load_i16(&coeffs[(r0 + i) * C + k0..])
                } else {
                    V::splat(0)
                };
            }
            V::store_transposed_sat16(&x[..lanes], &mut t[k0 * C + r0..], C);
        }
    }
    // SAFETY: every (r0, k0) tile of the C x C buffer was written above.
    let t = unsafe { written(&t, C * C) };
    // Clamped to 16 bits by the saturating store (AVM `clamp_buf`).
    let mut m = [MaybeUninit::<i16>::uninit(); MAX_CODED_AREA];
    for r0 in (0..C).step_by(lanes) {
        if r0 < nonzero {
            for (k, lane) in x.iter_mut().enumerate() {
                *lane = V::load_i16(&t[k * C + r0..]);
            }
            inverse_dct(&x, inputs, &mut y);
            for lane in &mut y {
                *lane = lane.round_shift(shifts.0);
            }
        } else {
            y = [V::splat(0); C];
        }
        store_pass_sat16::<V>(&y, r0, C, &mut m);
    }
    // SAFETY: rows r0..r0 + LANES of `m` were written for every r0.
    let m = unsafe { written(&m, C * C) };
    let (low, high) = (V::splat(RESIDUAL_MIN), V::splat(RESIDUAL_MAX));
    for x0 in (0..C).step_by(lanes) {
        for (r, lane) in x.iter_mut().enumerate() {
            *lane = V::load_i16(&m[r * C + x0..]);
        }
        inverse_dct(&x, inputs, &mut y);
        for (r, lane) in y.iter().enumerate() {
            let sample = lane.round_shift(shifts.1).max(low).min(high);
            sample.store_sat16(&mut out[r * C + x0..]);
        }
    }
}

/// The forward DCT on pairs (AVX2 only so far).
#[cfg(target_arch = "x86_64")]
mod forward_pass {
    use core::mem::MaybeUninit;

    use super::super::generic::Av2Basis;
    use super::super::{DC_WEIGHT, MAX_CODED_SIDE, MAX_POINTS};
    use super::{NO_PAIRS, ODD2, ODD4, ODD8, ODD16, ODD32, pack, pair_up, paired_dot};
    use crate::codec_txfm::butterfly::DctBasis;

    /// The 64-point level's odd matrix.
    const ODD64: [[i32; 32]; 32] = Av2Basis::ODD64.unwrap();
    use crate::codec_txfm::generic::{as_uninit, written};
    use crate::simd::WordPairLanes;
    use crate::txfm::generic::store_pass;

    /// Packed weights of the forward level of `odd_matrix`: entry `[r][p]`
    /// pairs samples `2p` and `2p + 1` of row `r`.
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

    const FORWARD4: [[i32; 2]; 2] = forward_weights(&ODD4);
    const FORWARD8: [[i32; 4]; 4] = forward_weights(&ODD8);
    const FORWARD16: [[i32; 8]; 8] = forward_weights(&ODD16);
    const FORWARD32: [[i32; 16]; 16] = forward_weights(&ODD32);
    const FORWARD64: [[i32; 32]; 32] = forward_weights(&ODD64);

    /// `generic::forward_level`, with paired products when `PAIRED` and
    /// `H >= 2` (`packed` from [`forward_weights`]).
    #[inline(always)]
    fn forward_level<V: WordPairLanes, const H: usize, const PAIRED: bool>(
        odd_matrix: &[[i32; H]; H],
        packed: &[[i32; H]; H],
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
        let paired = PAIRED && H >= 2;
        let pairs = pair_up(&differences);
        for (r, row) in odd_matrix.iter().enumerate() {
            let k = (2 * r + 1) * step;
            if k >= outputs {
                break;
            }
            y[k] = if paired {
                paired_dot::<V, H>(&pairs, &packed[r], H / 2)
            } else {
                let mut acc = differences[0].mul_scalar(row[0]);
                for j in 1..H {
                    acc = acc.add(differences[j].mul_scalar(row[j]));
                }
                acc
            };
        }
    }

    /// `butterfly::forward_dct`, paired when `PAIRED`.
    #[inline(always)]
    fn forward_dct<V: WordPairLanes, const N: usize, const PAIRED: bool>(
        x: &[V; N],
        y: &mut [V; N],
        outputs: usize,
    ) {
        let mut sums = *x;
        if N >= 64 {
            forward_level::<V, 32, PAIRED>(&ODD64, &FORWARD64, &mut sums, N / 64, outputs, y);
        }
        if N >= 32 {
            forward_level::<V, 16, PAIRED>(&ODD32, &FORWARD32, &mut sums, N / 32, outputs, y);
        }
        if N >= 16 {
            forward_level::<V, 8, PAIRED>(&ODD16, &FORWARD16, &mut sums, N / 16, outputs, y);
        }
        if N >= 8 {
            forward_level::<V, 4, PAIRED>(&ODD8, &FORWARD8, &mut sums, N / 8, outputs, y);
        }
        if N >= 4 {
            forward_level::<V, 2, PAIRED>(&ODD4, &FORWARD4, &mut sums, N / 4, outputs, y);
        }
        forward_level::<V, 1, PAIRED>(&ODD2, &NO_PAIRS, &mut sums, N / 2, outputs, y);
        y[0] = sums[0].mul_scalar(DC_WEIGHT);
    }

    /// `generic::forward`: paired column pass, 32-bit row pass.
    ///
    /// # Safety
    ///
    /// `V`'s ISA must be available; `V::LANES <= C`.
    #[inline(always)]
    pub(crate) unsafe fn forward<V: WordPairLanes, const N: usize, const C: usize>(
        residual: &[i16],
        stride: usize,
        shifts: (u32, u32),
        out: &mut [i32],
    ) {
        let lanes = V::LANES;
        let mut x = [V::splat(0); N];
        let mut y = [V::splat(0); N];
        let mut g = [MaybeUninit::<i32>::uninit(); MAX_POINTS * MAX_CODED_SIDE];
        for x0 in (0..N).step_by(lanes) {
            for (r, lane) in x.iter_mut().enumerate() {
                *lane = V::load_i16(&residual[r * stride + x0..]);
            }
            forward_dct::<V, N, true>(&x, &mut y, C);
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
            forward_dct::<V, N, false>(&x, &mut y, C);
            for lane in &mut y[..C] {
                *lane = lane.round_shift(shifts.1);
            }
            store_pass::<V>(&y[..C], k0, C, out);
        }
    }
}
