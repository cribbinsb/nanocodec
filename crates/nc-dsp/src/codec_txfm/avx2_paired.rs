//! AVX2 HEVC 16- and 32-point forward DCT lines on 16-bit pairs
//! (`vpmaddwd`), for the generic drivers ([`Line`] on [`I32x8`]): each
//! output is one row of the HEVC matrix times the input line, two products
//! per instruction, in a loop over the outputs. The generic even/odd
//! butterflies unroll into several KB of straight-line 32-bit multiplies
//! per pass; this line is one out-of-line copy of a few hundred bytes, and
//! the hot exact column pass of HEVC Faster's 16x16 forward runs about a
//! third fewer instructions (callgrind, crowd: 2766 -> 1796 per call
//! together with the 16-bit staged rows of `txfm::avx2_forward`).
//!
//! The inverses keep the butterflies: a full paired matrix product costs
//! more instructions there than the bounded butterflies skipping zero
//! coefficients (+55% on the 16-point inverse, crowd).
//!
//! Exact: every product sum stays in `i32` and each paired value fits `i16`
//! (ranges below), so the result is the matrix product the butterflies
//! compute (checkasm compares every kernel with the scalar butterflies).
//!
//! # Ranges (16-bit pairs)
//!
//! - Row pass: residuals of 8-bit samples, `|r| <= 255`.
//! - Column pass: the row pass's output. Exact rows are at most
//!   `255 · 64 · N >> (log2 N − 1) = 32640` (the DC row has the largest L1
//!   norm, `64 · N`). Staged rows (`HevcForwardStaging::Rows`) keep the DC
//!   exact and approximate the AC rows within 0.011%, whose L1 norms are
//!   at most `922 · N / 16` (`255 · 1844 >> 4 = 29389` for 32 points).
//!
//! Matrix entries are at most 90 in magnitude, so every weight fits `i16`.

#![allow(clippy::inline_always)]

use super::DCT32;
use super::generic::Line;
use crate::simd::I32Lanes;
use crate::simd::avx2::{I32x8, WordPairs8};

/// Input pairs of the largest (32-point) line.
const MAX_PAIRS: usize = 16;

/// Two matrix entries as one `vpmaddwd` weight ([`WordPairs8::dot`]);
/// every entry is within ±90, so the casts are exact.
const fn pack(w0: i8, w1: i8) -> i32 {
    ((w1 as i32) << 16) | (w0 as i16 as u16 as i32)
}

/// Forward weights: entry `[k][p]` pairs entries `2p` and `2p + 1` of row
/// `k` of [`DCT32`]. Row `k` of the `N`-point matrix is the first `N`
/// entries of row `k · 32 / N`, so one table serves both sizes.
pub(super) const FORWARD: [[i32; MAX_PAIRS]; 32] = {
    let mut packed = [[0; MAX_PAIRS]; 32];
    let mut k = 0;
    while k < 32 {
        let mut p = 0;
        while p < MAX_PAIRS {
            packed[k][p] = pack(DCT32[k][2 * p], DCT32[k][2 * p + 1]);
            p += 1;
        }
        k += 1;
    }
    packed
};

/// Pairs `x[2p]` with `x[2p + 1]` for the first `N / 2` pairs (the rest
/// stay unused).
#[inline(always)]
fn pair_up<const N: usize>(x: &[I32x8]) -> [WordPairs8; MAX_PAIRS] {
    let zero = I32x8::splat(0).word_pairs(I32x8::splat(0));
    core::array::from_fn(|p| {
        if p < N / 2 {
            x[2 * p].word_pairs(x[2 * p + 1])
        } else {
            zero
        }
    })
}

/// `Σ_p pairs[p] · weights[p]`.
#[inline(always)]
fn dot(pairs: &[WordPairs8], weights: &[i32; MAX_PAIRS]) -> I32x8 {
    pairs
        .iter()
        .zip(weights)
        .fold(I32x8::splat(0), |acc, (pair, &w)| acc.add(pair.dot(w)))
}

/// The first `outputs` outputs of the `N`-point forward DCT. Out of line:
/// one copy serves every pass and lane group of a kernel (inlined, the
/// loop over outputs unrolled completely).
///
/// # Safety
///
/// AVX2 must be available.
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn forward<const N: usize>(x: &[I32x8], y: &mut [I32x8], outputs: usize) {
    let pairs = pair_up::<N>(x);
    let step = DCT32.len() / N;
    for (k, out) in y[..outputs].iter_mut().enumerate() {
        *out = dot(&pairs[..N / 2], &FORWARD[k * step]);
    }
}

/// Defines a paired forward line of `$n` points.
macro_rules! paired_lines {
    ($($forward:ident, $n:literal);*) => {$(
        #[doc = concat!("The ", stringify!($n), "-point HEVC forward DCT on 16-bit pairs.")]
        pub(crate) struct $forward;
        impl Line<I32x8> for $forward {
            #[inline(always)]
            fn apply(x: &[I32x8], y: &mut [I32x8]) {
                // SAFETY: `I32x8` values exist only inside AVX2 kernels.
                unsafe { forward::<$n>(x, y, $n) };
            }

            #[inline(always)]
            fn apply_low(x: &[I32x8], y: &mut [I32x8]) {
                // SAFETY: as `apply`.
                unsafe { forward::<$n>(x, y, $n / 2) };
            }
        }
    )*};
}

paired_lines!(PairedFdct16, 16; PairedFdct32, 32);
