//! The SIMD kernels of `vvc_mts`, generic over [`I16Pairs`]: each pass is a
//! matrix product whose right-hand side is a basis table, interleaved so
//! that one vector holds two consecutive rows of it (`PairTable` or
//! `PairTable64`), and whose left-hand entries are broadcast as pairs: one
//! pairwise multiply-add (`pmaddwd`, `smull` + `addp`, `i32x4.dot_i16x8_s`)
//! covers two terms of every output in the vector. Both operands fit 16 bits,
//! the sums are exact, so the results equal the scalar reference's
//! (`super::scalar`).
//!
//! No closures are used inside the `#[target_feature]` lane code: matrix
//! inputs are passed as `(slice, row_stride, col_stride)` so LLVM keeps every
//! loop inlined with full target features.

#![allow(clippy::inline_always)]

use super::{
    DCT64_PAIRS, FORWARD_MATRIX_SHIFT, INVERSE_FIRST_SHIFT, MAX_INVERSE_SIDE, PairTable,
    frequency_pairs, nonzero_extent, sample_pairs,
};
use crate::VvcTransformKind;
use crate::simd::{I16Pairs, I32Lanes};

/// The most output vectors of one product row (64 outputs of 4 lanes).
const MAX_VECTORS: usize = MAX_INVERSE_SIDE / 4;

/// Storage of a product's rounded sums.
trait Sink: Copy {
    fn store<L: I32Lanes>(value: L, dst: &mut [Self]);
}

impl Sink for i32 {
    #[inline(always)]
    fn store<L: I32Lanes>(value: L, dst: &mut [i32]) {
        value.store(dst);
    }
}

impl Sink for i16 {
    #[inline(always)]
    fn store<L: I32Lanes>(value: L, dst: &mut [i16]) {
        value.store_sat16(dst);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn product<V: I16Pairs, O: Sink, const STRIDE: usize>(
    (src, row_stride, col_stride): (&[i16], usize, usize),
    right: &[[i16; STRIDE]; 16],
    (rows, depth, cols): (usize, usize, usize),
    shift: u32,
    clip: bool,
    out: &mut [O],
    out_stride: usize,
) {
    let lanes = V::Pairs::LANES;
    debug_assert!(cols.is_multiple_of(lanes) && cols / lanes <= MAX_VECTORS);
    let vectors = cols / lanes;
    let pair_count = depth.div_ceil(2);
    for r in 0..rows {
        let base = r * row_stride;
        let mut sums = [V::Pairs::splat(0); MAX_VECTORS];
        for (p, pairs) in right.iter().enumerate().take(pair_count) {
            let k = 2 * p;
            let first = src[base + k * col_stride];
            let second = if k + 1 < depth {
                src[base + (k + 1) * col_stride]
            } else {
                0
            };
            let pair = V::splat_pair(first, second);
            for (j, sum) in sums[..vectors].iter_mut().enumerate() {
                *sum = sum.add(V::load(&pairs[2 * j * lanes..]).pair_dot(pair));
            }
        }
        for (j, sum) in sums[..vectors].iter().enumerate() {
            let rounded = sum.round_shift(shift);
            let value = if clip { rounded.clamp16() } else { rounded };
            O::store(value, &mut out[r * out_stride + j * lanes..]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn product_by_width<S: I16Pairs, M: I16Pairs, W: I16Pairs, O: Sink, const STRIDE: usize>(
    left: (&[i16], usize, usize),
    right: &[[i16; STRIDE]; 16],
    shape: (usize, usize, usize),
    shift: u32,
    clip: bool,
    out: &mut [O],
    out_stride: usize,
) {
    // SAFETY: forwarded ISA contract; `shape.2` is a power of two >= 4.
    unsafe {
        if shape.2 < M::Pairs::LANES {
            product::<S, O, STRIDE>(left, right, shape, shift, clip, out, out_stride);
        } else if shape.2 < W::Pairs::LANES {
            product::<M, O, STRIDE>(left, right, shape, shift, clip, out, out_stride);
        } else {
            product::<W, O, STRIDE>(left, right, shape, shift, clip, out, out_stride);
        }
    }
}

/// `super::scalar::inverse` with lane kernels for any side in `4..=64` and
/// `second_shift` (`20 − BitDepth`).
#[inline(always)]
pub(super) unsafe fn inverse<S: I16Pairs, M: I16Pairs, W: I16Pairs>(
    (width, height): (usize, usize),
    [horizontal, vertical]: [VvcTransformKind; 2],
    coefficients: &[i16],
    out: &mut [i16],
    second_shift: u32,
) {
    let out = &mut out[..width * height];
    let (rows, cols) = nonzero_extent(
        coefficients,
        width,
        vertical.frequencies(height),
        horizontal.frequencies(width),
    );
    if rows == 0 {
        out.fill(0);
        return;
    }
    let needed = cols * height;
    let mut small = [0i16; 64];
    let mut med;
    let mut large;
    let columns: &mut [i16] = if needed <= small.len() {
        &mut small[..needed]
    } else if needed <= 256 {
        med = [0i16; 256];
        &mut med[..needed]
    } else {
        large = [0i16; 2048];
        &mut large[..needed]
    };
    // SAFETY: forwarded ISA and slice contract; no closure so target features stay active.
    unsafe {
        if height == MAX_INVERSE_SIDE {
            product_by_width::<S, M, W, i16, 128>(
                (coefficients, 1, width),
                &DCT64_PAIRS,
                (cols, rows, height),
                INVERSE_FIRST_SHIFT,
                true,
                columns,
                height,
            );
        } else {
            product_by_width::<S, M, W, i16, 64>(
                (coefficients, 1, width),
                frequency_pairs(vertical, height),
                (cols, rows, height),
                INVERSE_FIRST_SHIFT,
                true,
                columns,
                height,
            );
        }
        if width == MAX_INVERSE_SIDE {
            product_by_width::<S, M, W, i16, 128>(
                (columns, 1, height),
                &DCT64_PAIRS,
                (height, cols, width),
                second_shift,
                true,
                out,
                width,
            );
        } else {
            product_by_width::<S, M, W, i16, 64>(
                (columns, 1, height),
                frequency_pairs(horizontal, width),
                (height, cols, width),
                second_shift,
                true,
                out,
                width,
            );
        }
    }
}

/// `super::scalar::forward` with lane kernels.
#[inline(always)]
pub(super) unsafe fn forward<S: I16Pairs, M: I16Pairs, W: I16Pairs>(
    (width, height): (usize, usize),
    [horizontal, vertical]: [VvcTransformKind; 2],
    residual: &[i16],
    out: &mut [i16],
    columns: &mut [i32],
) {
    let (cols, rows) = (horizontal.frequencies(width), vertical.frequencies(height));
    let out = &mut out[..width * height];
    let right_h: &PairTable = sample_pairs(horizontal, width);
    let right_v: &PairTable = sample_pairs(vertical, height);
    // SAFETY: forwarded contract.
    unsafe {
        product_by_width::<S, M, W, i16, 64>(
            (residual, width, 1),
            right_h,
            (height, width, cols),
            width.ilog2() - 1,
            false,
            out,
            cols,
        );
        product_by_width::<S, M, W, i32, 64>(
            (out, 1, cols),
            right_v,
            (cols, height, rows),
            height.ilog2() + FORWARD_MATRIX_SHIFT,
            false,
            columns,
            rows,
        );
    }
    for (k, line) in out.chunks_exact_mut(width).enumerate() {
        if k < rows {
            for (x, value) in line[..cols].iter_mut().enumerate() {
                *value = crate::simd::saturate_i16(columns[x * rows + k]);
            }
            line[cols..].fill(0);
        } else {
            line.fill(0);
        }
    }
}

macro_rules! kernels {
    ($(#[$attr:meta])*; $small:ty, $med:ty, $wide:ty) => {
        $(#[$attr])*
        pub(super) unsafe fn inverse(
            size: (usize, usize),
            kinds: [crate::VvcTransformKind; 2],
            coefficients: &[i16],
            out: &mut [i16],
            second_shift: u32,
        ) {
            // SAFETY: target feature enabled; contract forwarded.
            unsafe {
                super::generic::inverse::<$small, $med, $wide>(
                    size,
                    kinds,
                    coefficients,
                    out,
                    second_shift,
                )
            }
        }

        $(#[$attr])*
        pub(super) unsafe fn forward(
            size: (usize, usize),
            kinds: [crate::VvcTransformKind; 2],
            residual: &[i16],
            out: &mut [i16],
        ) {
            let mut columns = [0i32; 1024];
            // SAFETY: as `inverse`.
            unsafe {
                super::generic::forward::<$small, $med, $wide>(
                    size,
                    kinds,
                    residual,
                    out,
                    &mut columns[..size.0 * size.1],
                )
            }
        }

        pub(super) fn kernels() -> super::Kernels {
            super::Kernels { inverse, forward }
        }
    };
}
pub(super) use kernels;
