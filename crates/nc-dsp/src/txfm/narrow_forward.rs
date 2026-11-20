//! Exact AV1/VP9 forward DCTs of 16x16, 32x32 and 64x64 blocks on 16-bit
//! lanes, written once over [`ForwardWords`] (AVX2 sixteen lanes in
//! `avx2_forward`, NEON eight in `neon_forward`, SIMD128 eight in
//! `simd128_forward`). The scalar reference is the 32-bit `generic` driver
//! (libaom `fwd_txfm2d_c` for square `DCT_DCT`); every result here equals it.
//!
//! # Ranges, stage by stage (8-bit residuals, `|r| <= 255`)
//!
//! - **Column input.** libaom shifts left by 2 (16- and 32-point; 0 for
//!   64-point): at most `1020`, an `i16`.
//! - **Column graph.** No guard: the amplitude limits of the 16-point
//!   (Q13), 32-point (Q12) and pruned 64-point (Q13) graphs, i.e. the largest
//!   input magnitude for which every node and every equal-weight pre-add of
//!   the 16-bit `half_btf` fits `i16`, are at least 1020, 1020 and 255
//!   (`range_probe::forward_16_columns_fit_i16_for_8bit_residuals`,
//!   `forward_32_64_columns_fit_i16_for_8bit_residuals`). Rotations multiply
//!   16-bit pairs into exact 32-bit sums (`|w| <= 2^13`, `|x| < 2^15`) before
//!   the rounding shift back to 16 bits.
//! - **Middle shift** (2 for 16 and 64, 4 for 32): a rounding right shift
//!   of an `i16`, exact in each lane type (no offset overflow).
//! - **Row graph.** Its inputs can exceed the row graph's amplitude limit
//!   (column outputs reach about 2886 against about 2047), so each batch of
//!   lines checks its input L1 norms while loading them: at most
//!   [`NARROW_ROW_L1_LIMIT`] (32000) keeps every node in `i16`
//!   (`range_probe::forward_row_graphs_fit_i16_for_l1_at_most_the_row_limit`,
//!   16-, 32- and pruned 64-point). A batch above it runs libaom's 32-bit
//!   graph on the same 16-bit row inputs ([`ForwardWords::wide_rows`]), so
//!   the result equals the reference for every input; real residuals are far
//!   below the limit.
//! - **Outputs.** Inside the guard the row outputs are graph nodes, so they
//!   fit `i16` and equal libaom's (whose 16-bit store never saturates,
//!   `generic::tests::forward_outputs_fit_i16_with_margin`); the 64-point
//!   final rounding shift by 2 only shrinks them.
//!
//! The 16-point forward is the hot one (VP9's 16x16 and AV1's): on sixteen
//! lanes each pass is one run of the graph on 16-bit lanes, where the 32-bit
//! drivers ran two (AVX2) with twice the register traffic.
//!
//! The functions are `unsafe` only because the lane type's instructions need
//! its ISA: call them from a `#[target_feature]` function of that ISA (the
//! drivers are `inline(always)`), or on NEON / SIMD128, which are enabled
//! for the whole build.

#![allow(clippy::inline_always)]

use core::mem::MaybeUninit;

use super::forward_lanes::{ForwardLanes, NARROW_ROW_L1_LIMIT};
use super::{cospi, dct};
use crate::codec_txfm::generic::as_uninit;
use crate::simd::I32Lanes;

/// Most inputs of a forward graph.
const MAX_POINTS: usize = 64;
/// Inter-pass buffer capacity: 64 columns of 32 kept outputs.
const BUFFER_CAPACITY: usize = MAX_POINTS * 32;

/// Coefficients kept per dimension: all of a 16x16 or 32x32 block, the low
/// 32 of a 64x64 (libaom `av1_get_adjusted_tx_size`).
pub(super) const fn kept(n: usize) -> usize {
    if n == 64 { 32 } else { n }
}

/// What the drivers need of a 16-bit lane type beyond the butterflies.
pub(crate) trait ForwardWords: ForwardLanes {
    /// Lines per vector (8 or 16); also the side of the transposed tiles.
    const LANES: usize;
    /// Per-lane L1 norms, saturating.
    type Norm: Copy;
    /// libaom-exact 32-bit lanes of the fallback row graph.
    type Wide: I32Lanes;
    /// The first `LANES` values of `src`.
    fn load(src: &[i16]) -> Self;
    /// Store the lanes at the start of `dst`.
    fn store(self, dst: &mut [MaybeUninit<i16>]);
    /// Left shift by 2 (libaom's 16- and 32-point input shift; exact for
    /// 8-bit residuals).
    fn shift_left_2(self) -> Self;
    /// `round_shift(self, SHIFT)` exactly, with no intermediate overflow.
    fn round_shift<const SHIFT: u32>(self) -> Self;
    /// Transpose the `LANES`-square tile `tile` (`LANES` vectors) in place.
    fn transpose(tile: &mut [Self]);
    fn norm_zero() -> Self::Norm;
    /// Add `|self|` to each lane's norm (saturating, so an overflowing norm
    /// still exceeds every limit; `|i16::MIN|` reads correctly as unsigned).
    fn accumulate_l1(self, norm: Self::Norm) -> Self::Norm;
    /// Every lane's norm is at most `limit`.
    fn within_limit(norm: Self::Norm, limit: u16) -> bool;
    /// [`wide_row_batch`] of the `LANES` lines from `r0`, out of line: the
    /// rare fallback stays out of the hot graph's instruction cache. An ISA
    /// enabled only per function (AVX2) overrides this with a function
    /// compiled for it, as `narrow_inverse::NarrowLanes::wide_rows`.
    fn wide_rows<const N: usize, const RASTER: bool, const KEEP: usize>(
        buf: &[i16],
        r0: usize,
        out: &mut [i16],
    ) {
        outlined_wide_rows::<Self, N, RASTER, KEEP>(buf, r0, out);
    }
}

/// [`ForwardWords::wide_rows`] for ISAs enabled for the whole build.
#[cold]
#[inline(never)]
fn outlined_wide_rows<V: ForwardWords, const N: usize, const RASTER: bool, const KEEP: usize>(
    buf: &[i16],
    r0: usize,
    out: &mut [i16],
) {
    wide_row_batch::<V::Wide, N, RASTER, KEEP>(buf, r0, V::LANES, out);
}

/// `(0..n).step_by(lanes)` as a counted loop (bounds-check friendly).
#[inline(always)]
fn groups(n: usize, lanes: usize) -> impl Iterator<Item = usize> {
    (0..n / lanes).map(move |group| group * lanes)
}

/// The column input of an `N`-point forward (libaom `fwd_shift_NxN[0]`).
#[inline(always)]
fn column_input<V: ForwardWords, const N: usize>(row: V) -> V {
    if N == 64 { row } else { row.shift_left_2() }
}

/// The column graph of an `N`-point forward keeping `KEEP` outputs
/// (`av1_fwd_cos_bit_col`: 13, 12, 13).
#[inline(always)]
fn column_dct<V: ForwardLanes, const N: usize, const KEEP: usize>(
    x: &[V; MAX_POINTS],
    y: &mut [V; MAX_POINTS],
) {
    match N {
        16 => dct::fdct16(
            x[..16].try_into().unwrap(),
            (&mut y[..16]).try_into().unwrap(),
            cospi(13),
            13,
        ),
        32 if KEEP == 16 => dct::fdct32_low16(
            x[..32].try_into().unwrap(),
            (&mut y[..32]).try_into().unwrap(),
            cospi(12),
            12,
        ),
        32 => dct::fdct32(
            x[..32].try_into().unwrap(),
            (&mut y[..32]).try_into().unwrap(),
            cospi(12),
            12,
        ),
        _ => dct::fdct64_low32(x, y, cospi(13), 13),
    }
}

/// The rounding shift between the passes (libaom `fwd_shift_NxN[1]`).
#[inline(always)]
fn mid_shift<V: ForwardWords, const N: usize>(column: V) -> V {
    if N == 32 {
        column.round_shift::<4>()
    } else {
        column.round_shift::<2>()
    }
}

/// The row graph of an `N`-point forward keeping `KEEP` outputs, on either
/// lane type (`av1_fwd_cos_bit_row`: 12, 12, 10).
#[inline(always)]
fn row_dct<V: ForwardLanes, const N: usize, const KEEP: usize>(
    x: &[V; MAX_POINTS],
    y: &mut [V; MAX_POINTS],
) {
    match N {
        16 => dct::fdct16(
            x[..16].try_into().unwrap(),
            (&mut y[..16]).try_into().unwrap(),
            cospi(12),
            12,
        ),
        32 if KEEP == 16 => dct::fdct32_low16(
            x[..32].try_into().unwrap(),
            (&mut y[..32]).try_into().unwrap(),
            cospi(12),
            12,
        ),
        32 => dct::fdct32(
            x[..32].try_into().unwrap(),
            (&mut y[..32]).try_into().unwrap(),
            cospi(12),
            12,
        ),
        _ => dct::fdct64_low32(x, y, cospi(10), 10),
    }
}

/// Exact forward `N`x`N` DCT (16, 32 or 64) keeping the `KEEP` lowest
/// frequencies per dimension (`kept(N)` for the full transform), into the
/// coded square of side `kept(N)` of `out`, column-major or (`RASTER`)
/// row-major. With `KEEP < kept(N)` the rest of the square is zeroed, which
/// equals the full forward with those coefficients discarded
/// (`txfm::low_keep`). Returns the number of row batches that ran on 16-bit
/// lanes (the rest took the wide fallback).
///
/// # Safety
///
/// The lane type's ISA must be available (module docs); `residual` holds
/// `N` rows of `N` 8-bit differences at `stride`.
#[inline(always)]
pub(crate) unsafe fn forward<
    V: ForwardWords,
    const N: usize,
    const RASTER: bool,
    const KEEP: usize,
>(
    residual: &[i16],
    stride: usize,
    out: &mut [i16],
) -> u32 {
    let coded = kept(N);
    let out = &mut out[..coded * coded];
    if KEEP < coded {
        // Only discarded coefficients need zeroing: the kept square is
        // completely overwritten below in either layout.
        for line in out[..KEEP * coded].chunks_exact_mut(coded) {
            line[KEEP..].fill(0);
        }
        out[KEEP * coded..].fill(0);
    }
    let mut buf = [MaybeUninit::<i16>::uninit(); BUFFER_CAPACITY];
    // SAFETY: forwarded contract.
    let columns = unsafe { column_pass::<V, N, KEEP>(residual, stride, &mut buf) };
    // SAFETY: forwarded contract.
    unsafe { row_pass::<V, N, RASTER, KEEP>(columns, out) }
}

/// The bounded raster forward (`generic::dct_bounded!`) of a full `N`x`N`
/// block (16 or 32): false, leaving `out` untouched, when every row-pass
/// input line's L1 norm is within `zero_limit` minus `rounding` (every
/// coefficient then has magnitude at most `zero_limit`); otherwise true
/// with the exact coefficients.
///
/// # Safety
///
/// As [`forward`].
#[inline(always)]
pub(crate) unsafe fn bounded<V: ForwardWords, const N: usize>(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    rounding: u32,
    out: &mut [i16],
) -> bool {
    let mut buf = [MaybeUninit::<i16>::uninit(); BUFFER_CAPACITY];
    // SAFETY: forwarded contract.
    let columns = unsafe { column_pass::<V, N, N>(residual, stride, &mut buf) };
    // Saturating u16 norms: a saturated norm (65535) is above every limit
    // tested here (clamped below it).
    let limit = zero_limit
        .saturating_sub(rounding)
        .min(u32::from(u16::MAX) - 1) as u16;
    let mut within = true;
    for r0 in groups(N, V::LANES) {
        let mut l1 = V::norm_zero();
        for c in 0..N {
            l1 = V::load(&columns[c * N + r0..]).accumulate_l1(l1);
        }
        within &= V::within_limit(l1, limit);
    }
    if within {
        return false;
    }
    // SAFETY: forwarded contract.
    unsafe { row_pass::<V, N, true, N>(columns, &mut out[..N * N]) };
    true
}

/// The column pass: column `c`'s `KEEP` lowest outputs, middle-shifted, at
/// `c * KEEP` of the returned buffer.
///
/// # Safety
///
/// As [`forward`].
#[inline(always)]
unsafe fn column_pass<'buf, V: ForwardWords, const N: usize, const KEEP: usize>(
    residual: &[i16],
    stride: usize,
    buf: &'buf mut [MaybeUninit<i16>; BUFFER_CAPACITY],
) -> &'buf [i16] {
    assert!(
        matches!(N, 16 | 32 | 64)
            && KEEP <= kept(N)
            && KEEP.is_multiple_of(V::LANES)
            && N.is_multiple_of(V::LANES)
    );
    let mut input = [V::splat(0); MAX_POINTS];
    let mut output = input;
    for x in groups(N, V::LANES) {
        for (r, column) in input[..N].iter_mut().enumerate() {
            *column = column_input::<V, N>(V::load(&residual[r * stride + x..]));
        }
        column_dct::<V, N, KEEP>(&input, &mut output);
        for k in groups(KEEP, V::LANES) {
            let tile = &mut output[k..k + V::LANES];
            for value in tile.iter_mut() {
                *value = mid_shift::<V, N>(*value);
            }
            // Row i of the tile: column x + i's outputs k..k + LANES.
            V::transpose(tile);
            for (i, row) in tile.iter().enumerate() {
                row.store(&mut buf[(x + i) * KEEP + k..]);
            }
        }
    }
    // SAFETY: the loop wrote, for every column 0..N, all `KEEP` outputs, and
    // `MaybeUninit<i16>` has the layout of `i16`.
    unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<i16>(), N * KEEP) }
}

/// The row pass over [`column_pass`]'s buffer, each batch of `LANES` lines
/// on 16-bit lanes inside the L1 guard (module docs), else on the wide
/// fallback. Writes the `KEEP`-square of `out` (row stride `kept(N)`).
///
/// # Safety
///
/// As [`forward`].
#[inline(always)]
unsafe fn row_pass<V: ForwardWords, const N: usize, const RASTER: bool, const KEEP: usize>(
    buf: &[i16],
    out: &mut [i16],
) -> u32 {
    let coded = kept(N);
    let mut input = [V::splat(0); MAX_POINTS];
    let mut output = input;
    let mut narrow_batches = 0;
    for r0 in groups(KEEP, V::LANES) {
        let mut l1 = V::norm_zero();
        for (c, column) in input[..N].iter_mut().enumerate() {
            let value = V::load(&buf[c * KEEP + r0..]);
            *column = value;
            l1 = value.accumulate_l1(l1);
        }
        if !V::within_limit(l1, NARROW_ROW_L1_LIMIT as u16) {
            V::wide_rows::<N, RASTER, KEEP>(buf, r0, out);
            continue;
        }
        narrow_batches += 1;
        row_dct::<V, N, KEEP>(&input, &mut output);
        if N == 64 {
            // libaom `fwd_shift_64x64[2]`; the smaller sizes have none.
            for value in &mut output[..KEEP] {
                *value = value.round_shift::<2>();
            }
        }
        let out = as_uninit(out);
        if RASTER {
            // Lane i of output[k] is coefficient (r0 + i, k): transposed
            // tiles give whole coefficient rows.
            for k0 in groups(KEEP, V::LANES) {
                let tile = &mut output[k0..k0 + V::LANES];
                V::transpose(tile);
                for (i, row) in tile.iter().enumerate() {
                    row.store(&mut out[(r0 + i) * coded + k0..]);
                }
            }
        } else {
            for (k, value) in output[..KEEP].iter().enumerate() {
                value.store(&mut out[k * coded + r0..]);
            }
        }
    }
    narrow_batches
}

/// The `lines` row-pass lines from `r0` on libaom's 32-bit graph (`W`
/// lanes at a time), from the same 16-bit row inputs: the fallback above
/// the L1 guard.
#[inline(always)]
pub(super) fn wide_row_batch<W: I32Lanes, const N: usize, const RASTER: bool, const KEEP: usize>(
    buf: &[i16],
    r0: usize,
    lines: usize,
    out: &mut [i16],
) {
    let coded = kept(N);
    let mut x = [W::splat(0); MAX_POINTS];
    let mut y = x;
    for first in groups(lines, W::LANES).map(|offset| r0 + offset) {
        for (c, column) in x[..N].iter_mut().enumerate() {
            *column = W::load_i16(&buf[c * KEEP + first..]);
        }
        row_dct::<W, N, KEEP>(&x, &mut y);
        if N == 64 {
            // libaom `fwd_shift_64x64[2]`.
            for value in &mut y[..KEEP] {
                *value = value.round_shift(2);
            }
        }
        if RASTER {
            // Lane j of y[k] is coefficient (first + j, k).
            let view = as_uninit(&mut out[..coded * coded]);
            for k0 in groups(KEEP, W::LANES) {
                W::store_transposed_sat16(
                    &y[k0..k0 + W::LANES],
                    &mut view[first * coded + k0..],
                    coded,
                );
            }
        } else {
            for (k, value) in y[..KEEP].iter().enumerate() {
                value.store_sat16(&mut out[k * coded + first..]);
            }
        }
    }
}
