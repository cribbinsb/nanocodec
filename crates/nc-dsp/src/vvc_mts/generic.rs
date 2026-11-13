//! The SIMD kernels of `vvc_mts`, generic over [`I16Pairs`]: each pass is a
//! matrix product whose right-hand side is a basis table, interleaved so
//! that one vector holds two consecutive rows of it (`PairTable`), and
//! whose left-hand entries are broadcast as pairs: one pairwise
//! multiply-add (`pmaddwd`, `smull` + `addp`, `i32x4.dot_i16x8_s`) covers
//! two terms of every output in the vector. Both operands fit 16 bits (the
//! module's range notes), the sums are exact, so the results equal the
//! scalar reference's (`super::scalar`).
//!
//! The column passes run transposed (one output row per column), so every
//! right-hand side is a basis; the forward transposes its result back.
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs): call them from a `#[target_feature]`
//! function of that ISA.

#![allow(clippy::inline_always)]

use super::{
    FORWARD_MATRIX_SHIFT, INVERSE_FIRST_SHIFT, INVERSE_SECOND_SHIFT, MAX_SIDE, PairTable,
    frequency_pairs, nonzero_extent, sample_pairs,
};
use crate::VvcTransformKind;
use crate::simd::{I16Pairs, I32Lanes};

/// The most output vectors of one product row (32 outputs of 4 lanes).
const MAX_VECTORS: usize = MAX_SIDE / 4;

/// One pass: `out[r * out_stride + c] = round(Σ_{k < depth} left(r, k) ·
/// B[k][c])` for `r < rows`, `c < cols` (a multiple of the lanes), where
/// `right` interleaves the rows of `B`; rounded by `shift` bits and
/// clipped to 16 bits when `clip`.
#[inline(always)]
#[allow(clippy::too_many_arguments, reason = "one product's operands")]
unsafe fn product<V: I16Pairs>(
    left: impl Fn(usize, usize) -> i16,
    right: &PairTable,
    (rows, depth, cols): (usize, usize, usize),
    shift: u32,
    clip: bool,
    out: &mut [i32],
    out_stride: usize,
) {
    let lanes = V::Pairs::LANES;
    debug_assert!(cols.is_multiple_of(lanes) && cols / lanes <= MAX_VECTORS);
    let vectors = cols / lanes;
    for r in 0..rows {
        let mut sums = [V::Pairs::splat(0); MAX_VECTORS];
        for (p, pairs) in right.iter().enumerate().take(depth.div_ceil(2)) {
            let k = 2 * p;
            let second = if k + 1 < depth { left(r, k + 1) } else { 0 };
            let pair = V::splat_pair(left(r, k), second);
            for (j, sum) in sums[..vectors].iter_mut().enumerate() {
                *sum = sum.add(V::load(&pairs[2 * j * lanes..]).pair_dot(pair));
            }
        }
        for (j, sum) in sums[..vectors].iter().enumerate() {
            let rounded = sum.round_shift(shift);
            let value = if clip { rounded.clamp16() } else { rounded };
            value.store(&mut out[r * out_stride + j * lanes..]);
        }
    }
}

/// [`product`] on `S` (4 outputs per vector) for 4 outputs a row, else on
/// `W`.
#[inline(always)]
#[allow(clippy::too_many_arguments, reason = "one product's operands")]
unsafe fn product_by_width<S: I16Pairs, W: I16Pairs>(
    left: impl Fn(usize, usize) -> i16,
    right: &PairTable,
    shape: (usize, usize, usize),
    shift: u32,
    clip: bool,
    out: &mut [i32],
    out_stride: usize,
) {
    // SAFETY: forwarded ISA contract; every output count is 4 or a
    // multiple of `W`'s lanes (8 at most, sides are powers of two).
    unsafe {
        if shape.2 < W::Pairs::LANES {
            product::<S>(left, right, shape, shift, clip, out, out_stride);
        } else {
            product::<W>(left, right, shape, shift, clip, out, out_stride);
        }
    }
}

/// `super::scalar::inverse` with lane kernels.
///
/// `columns` is working memory of at least the block's area.
///
/// # Safety
///
/// The ISA of `S` and `W` must be available; `S` has 4 pairs and `W` 4 or
/// 8; shapes and lengths as checked by `Dsp::vvc_inverse_kinds`.
#[inline(always)]
pub(super) unsafe fn inverse<S: I16Pairs, W: I16Pairs>(
    (width, height): (usize, usize),
    [horizontal, vertical]: [VvcTransformKind; 2],
    coefficients: &[i32],
    out: &mut [i32],
    columns: &mut [i32],
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
    // Columns, transposed: `columns[c * height + y]` is sample row `y` of
    // coefficient column `c` (only the columns holding coefficients).
    // SAFETY: forwarded contract; coefficients fit 16 bits (dequantised).
    unsafe {
        product_by_width::<S, W>(
            |c, k| coefficients[k * width + c] as i16,
            frequency_pairs(vertical, height),
            (cols, rows, height),
            INVERSE_FIRST_SHIFT,
            true,
            columns,
            height,
        );
    }
    // SAFETY: as above; the column pass clipped to 16 bits.
    unsafe {
        product_by_width::<S, W>(
            |y, c| columns[c * height + y] as i16,
            frequency_pairs(horizontal, width),
            (height, cols, width),
            INVERSE_SECOND_SHIFT,
            true,
            out,
            width,
        );
    }
}

/// `super::scalar::forward` with lane kernels.
///
/// # Safety
///
/// As [`inverse`]; residuals within ±255.
#[inline(always)]
pub(super) unsafe fn forward<S: I16Pairs, W: I16Pairs>(
    (width, height): (usize, usize),
    [horizontal, vertical]: [VvcTransformKind; 2],
    residual: &[i16],
    out: &mut [i32],
    columns: &mut [i32],
) {
    let (cols, rows) = (horizontal.frequencies(width), vertical.frequencies(height));
    let out = &mut out[..width * height];
    // Rows, in `out` until the columns are done: `transformed[y * cols +
    // k]`, frequency `k` of sample row `y`.
    let transformed = out;
    // SAFETY: forwarded contract.
    unsafe {
        product_by_width::<S, W>(
            |y, x| residual[y * width + x],
            sample_pairs(horizontal, width),
            (height, width, cols),
            width.ilog2() - 1,
            false,
            transformed,
            cols,
        );
    }
    // Columns, transposed: `columns[x * rows + k]`, vertical frequency
    // `k` of coefficient column `x`. The row pass stays within 16 bits for
    // residuals within ±255 (module docs).
    // SAFETY: as above.
    unsafe {
        product_by_width::<S, W>(
            |x, y| transformed[y * cols + x] as i16,
            sample_pairs(vertical, height),
            (cols, height, rows),
            height.ilog2() + FORWARD_MATRIX_SHIFT,
            false,
            columns,
            rows,
        );
    }
    let out = transformed;
    for (k, line) in out.chunks_exact_mut(width).enumerate() {
        if k < rows {
            for (x, value) in line[..cols].iter_mut().enumerate() {
                *value = columns[x * rows + k];
            }
            line[cols..].fill(0);
        } else {
            line.fill(0);
        }
    }
}

/// The ISA's kernels: `$small` (4 pairs) for 4 outputs a row, `$wide`
/// otherwise.
macro_rules! kernels {
    ($(#[$attr:meta])*; $small:ty, $wide:ty) => {
        $(#[$attr])*
        pub(super) unsafe fn inverse(
            size: (usize, usize),
            kinds: [crate::VvcTransformKind; 2],
            coefficients: &[i32],
            out: &mut [i32],
        ) {
            super::with_scratch(size.0 * size.1, |columns| {
                // SAFETY: target feature enabled; contract forwarded.
                unsafe {
                    super::generic::inverse::<$small, $wide>(size, kinds, coefficients, out, columns)
                }
            })
        }

        $(#[$attr])*
        pub(super) unsafe fn forward(
            size: (usize, usize),
            kinds: [crate::VvcTransformKind; 2],
            residual: &[i16],
            out: &mut [i32],
        ) {
            super::with_scratch(size.0 * size.1, |columns| {
                // SAFETY: as `inverse`.
                unsafe { super::generic::forward::<$small, $wide>(size, kinds, residual, out, columns) }
            })
        }

        pub(super) fn kernels() -> super::Kernels {
            super::Kernels { inverse, forward }
        }
    };
}
pub(super) use kernels;
