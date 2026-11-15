//! Rectangular VVC DCT-II over the existing exact HEVC line graphs.
//! Tile transposes make each vector span independent rows, then columns.
//! Unlike the square HEVC driver, neither pass clamps intermediate values.

#![allow(clippy::inline_always)]

use super::generic::{self as g, HevcFdct4, HevcFdct8, HevcFdct16, HevcFdct32, Line};
use crate::simd::I32Lanes;
use core::mem::MaybeUninit;

const MAX_SIDE: usize = 32;
const MAX_AREA: usize = MAX_SIDE * MAX_SIDE;

#[inline(always)]
fn line<V: I32Lanes>(input: &[V], output: &mut [V]) {
    match input.len() {
        4 => HevcFdct4::apply(input, output),
        8 => HevcFdct8::apply(input, output),
        16 => HevcFdct16::apply(input, output),
        32 => HevcFdct32::apply(input, output),
        _ => unreachable!("validated transform side"),
    }
}

/// Both dimensions are multiples of `V::LANES`; all buffers cover the area.
/// The caller enables `V`'s ISA. Residuals follow the shared 8-bit contract.
#[inline(always)]
unsafe fn rectangular<V: I32Lanes>(width: usize, height: usize, input: &[i16], out: &mut [i16]) {
    let lanes = V::LANES;
    let area = width * height;
    let mut transposed = [MaybeUninit::<i32>::uninit(); MAX_AREA];
    let mut intermediate = [MaybeUninit::<i32>::uninit(); MAX_AREA];
    let mut x = [V::splat(0); MAX_SIDE];
    let mut y = [V::splat(0); MAX_SIDE];
    // Every input tile is complete, and together they write exactly area
    // cells of the transposed buffer, without reading source-row padding.
    for r in (0..height).step_by(lanes) {
        for c in (0..width).step_by(lanes) {
            for (i, v) in x[..lanes].iter_mut().enumerate() {
                *v = V::load_i16(&input[(r + i) * width + c..]);
            }
            V::store_transposed(&x[..lanes], &mut transposed[c * height + r..area], height);
        }
    }
    // SAFETY: the complete tile partition above initialized every cell.
    let transposed = unsafe { g::written(&transposed, area) };
    for r in (0..height).step_by(lanes) {
        for (c, v) in x[..width].iter_mut().enumerate() {
            *v = V::load(&transposed[c * height + r..]);
        }
        line(&x[..width], &mut y[..width]);
        for v in &mut y[..width] {
            *v = v.round_shift(width.ilog2() - 1);
        }
        for c in (0..width).step_by(lanes) {
            V::store_transposed(
                &y[c..c + lanes],
                &mut intermediate[r * width + c..area],
                width,
            );
        }
    }
    // SAFETY: row groups above cover every intermediate output tile.
    let intermediate = unsafe { g::written(&intermediate, area) };
    // The quantiser loads two vectors of coefficients at a time, which a
    // pair of one-vector stores cannot forward (on NEON 64 bits each; the
    // same pattern cost Jetson HEVC Default +0.5..+0.9 % cycles). Each
    // store therefore narrows two vectors: two rows when a row is one
    // vector, else two neighbouring column groups of a row.
    let column = |c: usize, x: &mut [V; MAX_SIDE], y: &mut [V; MAX_SIDE]| {
        for (r, v) in x[..height].iter_mut().enumerate() {
            *v = V::load(&intermediate[r * width + c..]);
        }
        line(&x[..height], &mut y[..height]);
        for v in &mut y[..height] {
            *v = v.round_shift(height.ilog2() + 6);
        }
    };
    // At most 32640 (`vvc_forward` module docs): the saturating stores
    // never act.
    if width == lanes {
        column(0, &mut x, &mut y);
        for r in (0..height).step_by(2) {
            V::store_sat16_pair(y[r], y[r + 1], &mut out[r * width..]);
        }
        return;
    }
    let mut z = [V::splat(0); MAX_SIDE];
    for c in (0..width).step_by(2 * lanes) {
        column(c, &mut x, &mut y);
        column(c + lanes, &mut x, &mut z);
        for r in 0..height {
            V::store_sat16_pair(y[r], z[r], &mut out[r * width + c..]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn avx2(width: usize, height: usize, input: &[i16], out: &mut [i16]) {
    // SAFETY: the shared entry checked dimensions and slice lengths; AVX2
    // is enabled here. Keep narrow strips outside the wide scratch frame.
    unsafe {
        if width < 4 || height < 4 {
            super::vvc_forward_narrow::forward(width, height, input, out);
        } else {
            avx2_wide(width, height, input, out);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn avx2_wide(width: usize, height: usize, input: &[i16], out: &mut [i16]) {
    // SAFETY: the router excludes narrow sides and enables AVX2. Both
    // dimensions divide the chosen lane count.
    unsafe {
        if width == 4 || height == 4 {
            rectangular::<crate::simd::avx2::I32x4>(width, height, input, out);
        } else {
            rectangular::<crate::simd::avx2::I32x8>(width, height, input, out);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn neon(width: usize, height: usize, input: &[i16], out: &mut [i16]) {
    // SAFETY: checked block and enabled NEON; the wide helper owns scratch.
    unsafe {
        if width < 4 || height < 4 {
            super::vvc_forward_narrow::forward(width, height, input, out);
        } else {
            neon_wide(width, height, input, out);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline(never)]
unsafe fn neon_wide(width: usize, height: usize, input: &[i16], out: &mut [i16]) {
    // SAFETY: checked wide block and enabled NEON; both sides divide four.
    unsafe { rectangular::<crate::simd::neon::I32x4>(width, height, input, out) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) unsafe fn simd128(width: usize, height: usize, input: &[i16], out: &mut [i16]) {
    // SAFETY: checked block; SIMD128 is enabled for this module.
    unsafe {
        if width < 4 || height < 4 {
            super::vvc_forward_narrow::forward(width, height, input, out);
        } else {
            simd128_wide(width, height, input, out);
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(never)]
unsafe fn simd128_wide(width: usize, height: usize, input: &[i16], out: &mut [i16]) {
    // SAFETY: checked wide block and enabled SIMD128; both sides divide four.
    unsafe { rectangular::<crate::simd::simd128::I32x4>(width, height, input, out) }
}
