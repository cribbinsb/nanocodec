//! The typed AV2 transforms of [`super::typed`] (the scalar reference) on
//! vector lanes: the same passes, shifts and clamps, with each 1-D pass a
//! matrix product on `V::LANES` lines at once. DCT axes at eight points and
//! above use the shared butterfly. At sixteen and thirty-two points NEON
//! uses lane multiply-accumulates for the AVM ADST matrices. Like the AV1
//! and VP9 forward family, the driver transposes register tiles between
//! passes when both directions mix samples.
//! Large NEON horizontal identity applies its gain directly in raster order. All
//! products and sums use the reference's exact `i32` arithmetic.
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs).

#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use core::mem::MaybeUninit;

use super::Av2TxSize;
use super::generic::Av2Basis;
use super::typed::{self, Av2Kernel, Av2TxType, Matrix};
use crate::codec_txfm::butterfly::{dot, forward_dct, inverse_dct};
use crate::codec_txfm::generic::{as_uninit, written};
use crate::simd::I32Lanes;
use crate::txfm::generic::{store_pass, store_pass_sat16};

/// Largest side of a typed transform.
const MAX_SIDE: usize = 32;
/// Sizes with typed transforms (4x4 to 32x32).
pub(crate) const TYPED_SIZES: usize = 4;
/// Range of the inverse column pass output (as `typed`).
const RESIDUAL_MIN: i32 = -256;
const RESIDUAL_MAX: i32 = 255;

/// Forward typed AV2 transform of one size (see [`typed::forward`]).
pub(crate) type Av2ForwardTypedFn =
    unsafe fn(tx_type: Av2TxType, residual: &[i16], stride: usize, out: &mut [i32]);
/// Inverse typed AV2 transform of one size (see [`typed::inverse`]).
pub(crate) type Av2InverseTypedFn = unsafe fn(tx_type: Av2TxType, coeffs: &[i16], out: &mut [i16]);

/// One ISA's typed AV2 transforms, indexed by [`Av2TxSize`] (4x4..32x32).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Av2TypedFns {
    pub forward: [Av2ForwardTypedFn; TYPED_SIZES],
    pub inverse: [Av2InverseTypedFn; TYPED_SIZES],
}

const fn use_large_neon<const N: usize>() -> bool {
    cfg!(target_arch = "aarch64") && N >= 16
}

/// `y[k] = Σ_j M[k][j] · x[j]` for every `k` (identity: `gain · x[k]`).
#[inline(always)]
fn forward_1d<V: I32Lanes, const N: usize>(
    kernel: Av2Kernel,
    matrix: Matrix,
    x: &[V; N],
    y: &mut [V; N],
) {
    if N >= 8 && kernel == Av2Kernel::Dct {
        // SAFETY: the dispatcher enabled this backend, and this line holds N vectors.
        unsafe { run_dct_line::<V, N, false>(x, y) };
        return;
    }
    match matrix {
        Matrix::Dense(m) => {
            let m: &[[i32; N]; N] = m.as_chunks().0.try_into().expect("N-point kernel");
            for (out, row) in y.iter_mut().zip(m) {
                if use_large_neon::<N>() {
                    *out = dot(x, row);
                } else {
                    let mut acc = x[0].mul_scalar(row[0]);
                    for j in 1..N {
                        acc = acc.add(x[j].mul_scalar(row[j]));
                    }
                    *out = acc;
                }
            }
        }
        Matrix::Gain(gain) => {
            for (out, value) in y.iter_mut().zip(x) {
                *out = value.mul_scalar(gain);
            }
        }
    }
}

/// Use the shared DCT basis for each DCT axis of a mixed transform.
#[inline(always)]
unsafe fn run_dct_line<V: I32Lanes, const N: usize, const INVERSE: bool>(
    x: &[V; N],
    y: &mut [V; N],
) {
    // SAFETY: the lane type comes from an ISA-specific table. On x86
    // sixteen lanes are installed only on Ice Lake.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if V::LANES == 16 {
            dct_line_icl::<V, N, INVERSE>(x, y);
            return;
        }
        dct_line::<V, N, INVERSE>(x, y);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline(never)]
unsafe fn dct_line_icl<V: I32Lanes, const N: usize, const INVERSE: bool>(
    x: &[V; N],
    y: &mut [V; N],
) {
    dct::<V, N, INVERSE>(x, y);
}

#[inline(never)]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn dct_line<V: I32Lanes, const N: usize, const INVERSE: bool>(x: &[V; N], y: &mut [V; N]) {
    dct::<V, N, INVERSE>(x, y);
}

#[inline(always)]
fn dct<V: I32Lanes, const N: usize, const INVERSE: bool>(x: &[V; N], y: &mut [V; N]) {
    if INVERSE {
        inverse_dct::<Av2Basis, V, N>(x, N, y);
    } else {
        forward_dct::<Av2Basis, V, N>(x, y, N);
    }
}

/// `y[j] = Σ_k M[k][j] · c[k]` for every `j` (identity: `gain · c[j]`).
#[inline(always)]
fn inverse_1d<V: I32Lanes, const N: usize>(
    kernel: Av2Kernel,
    matrix: Matrix,
    c: &[V; N],
    y: &mut [V; N],
) {
    if N >= 8 && kernel == Av2Kernel::Dct {
        // SAFETY: the dispatcher enabled this backend and both lines contain N vectors.
        unsafe { run_dct_line::<V, N, true>(c, y) };
        return;
    }
    match matrix {
        Matrix::Dense(m) => {
            let m: &[[i32; N]; N] = m.as_chunks().0.try_into().expect("N-point kernel");
            for (j, out) in y.iter_mut().enumerate() {
                let mut acc = c[0].mul_scalar(m[0][j]);
                for k in 1..N {
                    acc = acc.add(c[k].mul_scalar(m[k][j]));
                }
                *out = acc;
            }
        }
        Matrix::Gain(gain) => {
            for (out, value) in y.iter_mut().zip(c) {
                *out = value.mul_scalar(gain);
            }
        }
    }
}

/// [`typed::forward`] of side `N` on `V` lanes.
///
/// # Safety
///
/// `V`'s ISA must be available; `V::LANES <= N`. The dispatcher checked
/// that `tx_type` exists at this size and the slice lengths.
#[inline(always)]
pub(crate) unsafe fn forward<V: I32Lanes, const N: usize>(
    tx_type: Av2TxType,
    residual: &[i16],
    stride: usize,
    out: &mut [i32],
) {
    let size = Av2TxSize::from_side(N).expect("typed transform side");
    let (vertical, horizontal) = (
        Matrix::of(tx_type.vertical, size),
        Matrix::of(tx_type.horizontal, size),
    );
    let horizontal_gain = match horizontal {
        Matrix::Gain(gain) if use_large_neon::<N>() => Some(gain),
        _ => None,
    };
    let (first, second) = size.forward_shifts();
    let lanes = V::LANES;
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    // The column pass's output, transposed: `g[x * N + k]`.
    let mut g = [MaybeUninit::<i32>::uninit(); MAX_SIDE * MAX_SIDE];
    for x0 in (0..N).step_by(lanes) {
        for (r, lane) in x.iter_mut().enumerate() {
            *lane = V::load_i16(&residual[r * stride + x0..]);
        }
        forward_1d(tx_type.vertical, vertical, &x, &mut y);
        if let Some(gain) = horizontal_gain {
            for (row, &value) in y.iter().enumerate() {
                value
                    .round_shift(first)
                    .mul_scalar(gain)
                    .round_shift(second)
                    .store(&mut out[row * N + x0..]);
            }
            continue;
        }
        for lane in &mut y {
            *lane = lane.round_shift(first);
        }
        store_pass::<V>(&y, x0, N, &mut g);
    }
    if horizontal_gain.is_some() {
        return;
    }
    // SAFETY: with a mixing horizontal transform, every column tile wrote
    // its entries to g; identity returned after writing directly to out.
    let g = unsafe { written(&g, N * N) };
    let out = as_uninit(&mut out[..N * N]);
    for v0 in (0..N).step_by(lanes) {
        for (column, lane) in x.iter_mut().enumerate() {
            *lane = V::load(&g[column * N + v0..]);
        }
        forward_1d(tx_type.horizontal, horizontal, &x, &mut y);
        for lane in &mut y {
            *lane = lane.round_shift(second);
        }
        store_pass::<V>(&y, v0, N, out);
    }
}

/// [`typed::inverse`] of side `N` on `V` lanes.
///
/// # Safety
///
/// As [`forward`].
#[inline(always)]
pub(crate) unsafe fn inverse<V: I32Lanes, const N: usize>(
    tx_type: Av2TxType,
    coeffs: &[i16],
    out: &mut [i16],
) {
    let size = Av2TxSize::from_side(N).expect("typed transform side");
    let (vertical, horizontal) = (
        Matrix::of(tx_type.vertical, size),
        Matrix::of(tx_type.horizontal, size),
    );
    let (first, second) = size.inverse_shifts();
    let lanes = V::LANES;
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    // The input transposed, `t[k * N + r] = coeffs[r][k]`.
    let mut t = [MaybeUninit::<i16>::uninit(); MAX_SIDE * MAX_SIDE];
    for r0 in (0..N).step_by(lanes) {
        for k0 in (0..N).step_by(lanes) {
            for (i, lane) in x[..lanes].iter_mut().enumerate() {
                *lane = V::load_i16(&coeffs[(r0 + i) * N + k0..]);
            }
            V::store_transposed_sat16(&x[..lanes], &mut t[k0 * N + r0..], N);
        }
    }
    // SAFETY: every (r0, k0) tile of the N x N buffer was written above.
    let t = unsafe { written(&t, N * N) };
    // The row pass's output, row-major, clamped to 16 bits by the
    // saturating store (as `typed`).
    let mut m = [MaybeUninit::<i16>::uninit(); MAX_SIDE * MAX_SIDE];
    for r0 in (0..N).step_by(lanes) {
        for (k, lane) in x.iter_mut().enumerate() {
            *lane = V::load_i16(&t[k * N + r0..]);
        }
        inverse_1d(tx_type.horizontal, horizontal, &x, &mut y);
        for lane in &mut y {
            *lane = lane.round_shift(first);
        }
        store_pass_sat16::<V>(&y, r0, N, &mut m);
    }
    // SAFETY: rows r0..r0 + LANES of `m` were written for every r0.
    let m = unsafe { written(&m, N * N) };
    let (low, high) = (V::splat(RESIDUAL_MIN), V::splat(RESIDUAL_MAX));
    for x0 in (0..N).step_by(lanes) {
        for (r, lane) in x.iter_mut().enumerate() {
            *lane = V::load_i16(&m[r * N + x0..]);
        }
        inverse_1d(tx_type.vertical, vertical, &x, &mut y);
        for (r, lane) in y.iter().enumerate() {
            lane.round_shift(second)
                .max(low)
                .min(high)
                .store_sat16(&mut out[r * N + x0..]);
        }
    }
}

/// The scalar table: the reference functions of [`typed`].
pub(crate) fn scalar_fns() -> Av2TypedFns {
    unsafe fn forward_ref<const N: usize>(t: Av2TxType, r: &[i16], s: usize, o: &mut [i32]) {
        typed::forward(Av2TxSize::from_side(N).expect("typed side"), t, r, s, o);
    }
    unsafe fn inverse_ref<const N: usize>(t: Av2TxType, c: &[i16], o: &mut [i16]) {
        typed::inverse(Av2TxSize::from_side(N).expect("typed side"), t, c, o);
    }
    Av2TypedFns {
        forward: [
            forward_ref::<4>,
            forward_ref::<8>,
            forward_ref::<16>,
            forward_ref::<32>,
        ],
        inverse: [
            inverse_ref::<4>,
            inverse_ref::<8>,
            inverse_ref::<16>,
            inverse_ref::<32>,
        ],
    }
}

/// Instantiates one ISA's typed table: `$v4` lanes for 4x4, `$v` otherwise.
macro_rules! typed_fns {
    ($name:ident, $(#[$attr:meta])*; $v4:ty, $v:ty) => {
        /// The typed AV2 transform table of this ISA.
        pub(crate) fn $name() -> $crate::av2_txfm::typed_lanes::Av2TypedFns {
            use $crate::av2_txfm::typed::Av2TxType;
            use $crate::av2_txfm::typed_lanes as tl;
            $(#[$attr])*
            unsafe fn forward<V: $crate::simd::I32Lanes, const N: usize>(
                t: Av2TxType, r: &[i16], s: usize, o: &mut [i32],
            ) {
                // SAFETY: the ISA is enabled for this function (dispatch contract).
                unsafe { tl::forward::<V, N>(t, r, s, o) }
            }
            $(#[$attr])*
            unsafe fn inverse<V: $crate::simd::I32Lanes, const N: usize>(
                t: Av2TxType, c: &[i16], o: &mut [i16],
            ) {
                // SAFETY: as above.
                unsafe { tl::inverse::<V, N>(t, c, o) }
            }
            tl::Av2TypedFns {
                forward: [forward::<$v4, 4>, forward::<$v, 8>, forward::<$v, 16>, forward::<$v, 32>],
                inverse: [inverse::<$v4, 4>, inverse::<$v, 8>, inverse::<$v, 16>, inverse::<$v, 32>],
            }
        }
    };
}

pub(crate) use typed_fns;
