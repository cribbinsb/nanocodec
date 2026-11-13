//! Fused HEVC inverse transform and reconstruction through the shared DSP.
//! The existing i32 coefficient layout is retained; only the first pass
//! has block-sized
//! scratch. Final transposes stay in registers and add directly to u8.
//! HEVC §8.6.4.2 specifies column shift 7, clamp16, then row shift 12.

#![allow(clippy::inline_always)]

use super::generic::{BoundedLine, HEVC_INVERSE_PASSES, MAX_AREA, written};
use crate::simd::I32Lanes;
use crate::txfm::generic::store_pass;
use core::mem::MaybeUninit;

/// Add a constant DCT DC residual with both normative roundings.
#[inline(always)]
pub(super) fn dc<V: I32Lanes, const N: usize>(coefficient: i32, dst: &mut [u8], stride: usize) {
    let coefficient = coefficient.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
    let first = (coefficient * 64 + 64) >> 7;
    let residual = (first * 64 + 2048) >> 12;
    if residual != 0 {
        super::generic::add_constant::<V>(residual, N, dst, stride);
    }
}

/// # Safety
/// V's ISA must be available; its lane count divides N and is at most eight.
/// The coefficient and destination slices cover `N × N` at the given stride;
/// `extent` is at most `(N, N)` and covers every nonzero coefficient.
#[inline(always)]
pub(super) unsafe fn inverse_add<V: I32Lanes, T: BoundedLine, const N: usize>(
    coeffs: &[i32],
    (rows, columns): (usize, usize),
    dst: &mut [u8],
    stride: usize,
    dct: bool,
) {
    if rows == 0 || columns == 0 {
        return;
    }
    if dct && rows == 1 && columns == 1 {
        dc::<V, N>(coeffs[0], dst, stride);
        return;
    }
    let (first, second) = HEVC_INVERSE_PASSES;
    let lanes = V::LANES;
    let written_columns = columns.div_ceil(lanes) * lanes;
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    let mut g = [MaybeUninit::<i32>::uninit(); MAX_AREA];
    for x0 in (0..written_columns).step_by(lanes) {
        for (k, lane) in x[..rows].iter_mut().enumerate() {
            *lane = V::load(&coeffs[k * N + x0..]).clamp16();
        }
        T::apply_bounded(&x, &mut y, rows);
        first.apply(&mut y);
        store_pass::<V>(&y, x0, N, &mut g);
    }
    // SAFETY: store_pass wrote all N samples of these columns.
    let g = unsafe { written(&g, written_columns * N) };
    x.fill(V::splat(0));
    for y0 in (0..N).step_by(lanes) {
        for (column, lane) in x[..columns].iter_mut().enumerate() {
            *lane = V::load(&g[column * N + y0..]);
        }
        T::apply_bounded(&x, &mut y, columns);
        second.apply(&mut y);
        for x0 in (0..N).step_by(lanes) {
            V::add_transposed_u8(&y[x0..x0 + lanes], &mut dst[y0 * stride + x0..], stride);
        }
    }
}
