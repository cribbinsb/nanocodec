//! Fused HEVC inverse transform and reconstruction through the shared DSP.
//! Coefficients come in 16 bits (the scaling process's range, §8.6.3);
//! only the first pass has block-sized scratch. Final transposes stay in
//! registers and add directly to u8.
//! HEVC §8.6.4.2 specifies column shift 7, clamp16, then row shift 12.

#![allow(clippy::inline_always)]

use super::generic::{BoundedLine, HEVC_INVERSE_PASSES, MAX_AREA, Pass, written};
use crate::simd::I32Lanes;
use crate::txfm::generic::store_pass;
use core::mem::MaybeUninit;
use nc_common::BitDepth;

/// DC basis element of HEVC DCT-II (`64`).
const BASIS_DC: i32 = 64;
/// First-pass right shift (HEVC §8.6.4.2).
const FIRST_SHIFT: u32 = 7;
/// First-pass rounding addend (`1 << 6`).
const FIRST_ROUND: i32 = 1 << (FIRST_SHIFT - 1);
/// `bdShift` of the second pass is `RESIDUAL_BITS - BitDepth` (HEVC §8.6.4.2).
const RESIDUAL_BITS: u32 = 20;

/// Add a constant DCT DC residual with both normative roundings.
#[inline(always)]
pub(super) fn dc<V: I32Lanes, const N: usize>(coefficient: i16, dst: &mut [u8], stride: usize) {
    let coefficient = i32::from(coefficient);
    let first = (coefficient * BASIS_DC + FIRST_ROUND) >> FIRST_SHIFT;
    let residual = (first * BASIS_DC + 2048) >> 12;
    if residual != 0 {
        super::generic::add_constant::<V>(residual, N, dst, stride);
    }
}

/// [`dc`] for `u16` destinations at `depth`.
#[inline(always)]
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code, reason = "u16 SIMD kernels target vector ISAs")
)]
fn dc_u16<V: I32Lanes, const N: usize>(
    coefficient: i16,
    dst: &mut [u16],
    stride: usize,
    depth: BitDepth,
) {
    let coefficient = i32::from(coefficient);
    let first = ((coefficient * BASIS_DC + FIRST_ROUND) >> FIRST_SHIFT)
        .clamp(i32::from(i16::MIN), i32::from(i16::MAX));
    let shift2 = RESIDUAL_BITS - depth.bits();
    debug_assert!(shift2 >= 1);
    let residual = ((first * BASIS_DC + (1 << (shift2 - 1))) >> shift2)
        .clamp(i32::from(i16::MIN), i32::from(i16::MAX));
    let max_val = depth.max();
    let delta = V::splat(residual);
    for row in 0..N {
        let line = &mut dst[row * stride..row * stride + N];
        for x0 in (0..N).step_by(V::LANES) {
            delta.add_to_u16(&mut line[x0..], max_val);
        }
    }
}

/// # Safety
/// V's ISA must be available; its lane count divides N and is at most eight.
/// The coefficient and destination slices cover `N × N` at the given stride;
/// `extent` is at most `(N, N)` and covers every nonzero coefficient.
#[inline(always)]
pub(super) unsafe fn inverse_add<V: I32Lanes, T: BoundedLine, const N: usize>(
    coeffs: &[i16],
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
            *lane = V::load_i16(&coeffs[k * N + x0..]);
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

/// Fused HEVC inverse transform and `u16` reconstruction at `depth`.
///
/// # Safety
/// `V`'s ISA must be available and `V::LANES` must divide `N`. The coefficient
/// and destination slices must cover `N x N` at `stride`, and `extent` must
/// cover every nonzero coefficient (`rows <= N`, `columns <= N`).
#[inline(always)]
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code, reason = "u16 SIMD kernels target vector ISAs")
)]
pub(crate) unsafe fn inverse_add_u16<V: I32Lanes, T: BoundedLine, const N: usize>(
    coeffs: &[i16],
    (rows, columns): (usize, usize),
    dst: &mut [u16],
    stride: usize,
    dct: bool,
    depth: BitDepth,
) {
    if rows == 0 || columns == 0 {
        return;
    }
    if dct && rows == 1 && columns == 1 {
        dc_u16::<V, N>(coeffs[0], dst, stride, depth);
        return;
    }
    let first = Pass {
        shift: FIRST_SHIFT,
        clamp: true,
    };
    let second = Pass {
        shift: RESIDUAL_BITS - depth.bits(),
        clamp: true,
    };
    let max_val = depth.max();
    let lanes = V::LANES;
    let written_columns = columns.div_ceil(lanes) * lanes;
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    let mut g = [MaybeUninit::<i32>::uninit(); MAX_AREA];
    for x0 in (0..written_columns).step_by(lanes) {
        for (k, lane) in x[..rows].iter_mut().enumerate() {
            *lane = V::load_i16(&coeffs[k * N + x0..]);
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
            V::add_transposed_u16(
                &y[x0..x0 + lanes],
                &mut dst[y0 * stride + x0..],
                stride,
                max_val,
            );
        }
    }
}
