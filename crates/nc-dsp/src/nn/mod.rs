//! Integer inference kernels of a small NNUE-style network: an incrementally
//! updated 16-bit accumulator (the first layer's sums over the active
//! feature columns), its clipped activation, and dense layers of 8-bit
//! weights over 7-bit activations.
//!
//! Everything is exact integer arithmetic, so a network's decisions are the
//! same on x86, Arm and WebAssembly: every ISA returns what the scalar
//! references here return, on every input of the contracts below
//! (`nc-checkasm` `check_nn`). Sums wrap (two's complement), so the result
//! is defined, and identical on every ISA, even where a caller breaks an
//! overflow bound.
//!
//! - **Accumulator** ([`ColumnsFn`]): `acc[j] ± columns[row · W + j]` for
//!   each listed row, `W = acc.len()` a multiple of
//!   [`NN_ACCUMULATOR_ALIGN`]. The network's weight bound keeps the sums
//!   within 16 bits; the kernel does not check.
//! - **Activation** ([`ClipI16Fn`], [`ClipI32Fn`]): `clamp(v >> shift, 0,
//!   127)` as a byte, the shift arithmetic (it rounds down).
//! - **Dense** ([`DenseFn`]): `bias[o] + Σ_i w[o · IN + i] · x[i]` in 32
//!   bits, `x` in `0..=`[`NN_ACTIVATION_MAX`], `w` any `i8`, `IN` a multiple
//!   of [`NN_DENSE_INPUT_ALIGN`] (the caller pads `x` and each weight row
//!   with zeros). A bias within [`nn_dense_max_bias`] cannot overflow.
//!
//! One pattern on every ISA. The accumulator keeps a block of 64 sums in
//! registers across the whole row list (one load and one add per row and
//! vector, no store until the block is done). Dense sums run four
//! outputs at a time against one load of each input vector
//! and finish with one shared horizontal reduction; the clipped layer is
//! the linear kernel followed by the activation kernel, so there is one dot
//! product per ISA. The byte products pair up in 16 bits before widening:
//! with `x ≤ 127` a pair sum is at most `2 · 127 · 128 = 32512` in
//! magnitude, inside signed 16 bits ([`MAX_PAIR_SUM`]), so `vpmaddubsw`
//! cannot saturate and the NEON and SIMD128 16-bit pair adds cannot wrap.
//! The AVX-512 tier uses VNNI's `vpdpbusd`, which has no 16-bit stage.
//!
//! A codec-independent family (`docs/analysis/encode-kernel-sharing.md`
//! "Network inference"): written from the arithmetic above only.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128;
#[cfg(test)]
mod tests;

use crate::{Dsp, Isa};

/// Largest activation: a clipped layer's outputs, and so every dense
/// layer's inputs, lie in `0..=127`.
pub const NN_ACTIVATION_MAX: u8 = 127;

/// The accumulator width is a multiple of this (16 sums: one AVX2 vector).
pub const NN_ACCUMULATOR_ALIGN: usize = 16;

/// A dense layer's input count is a multiple of this (32 bytes: one AVX2
/// vector); the caller pads with zeros, which add nothing to the sums.
pub const NN_DENSE_INPUT_ALIGN: usize = 32;

/// Largest shift of the accumulator activation (the values are 16-bit).
pub const NN_MAX_ACCUMULATOR_SHIFT: u32 = 15;

/// Largest shift of a clipped dense layer (the sums are 32-bit).
pub const NN_MAX_DENSE_SHIFT: u32 = 31;

/// Largest magnitude of one input-weight product: `127 · |−128|`.
const MAX_PRODUCT: i32 = NN_ACTIVATION_MAX as i32 * -(i8::MIN as i32);

/// Largest magnitude of the sum of two products, which the 16-bit stage of
/// the AVX2, NEON and SIMD128 dot products holds.
const MAX_PAIR_SUM: i32 = 2 * MAX_PRODUCT;
const _: () = assert!(MAX_PAIR_SUM <= i16::MAX as i32);

// The SIMD activations clip by signed saturation to a byte, then drop the
// negatives: right only while the largest activation is the largest `i8`.
const _: () = assert!(NN_ACTIVATION_MAX == i8::MAX as u8);

/// The panic of a row index beyond the column table, on every ISA.
const ROW_IN_TABLE: &str = "a row of the column table";

/// What the SIMD kernels share: block sizes and the column lookup.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod layout {
    /// Accumulator sums a kernel keeps in registers across the row list
    /// (four AVX2, two AVX-512 or eight 128-bit vectors); the rest of a
    /// width that is not a multiple runs a vector at a time.
    pub(super) const COLUMN_BLOCK: usize = 64;

    /// Outputs whose dot products share each input load and one reduction.
    pub(super) const DENSE_GROUP: usize = 4;

    /// The `N` values from `start` of row `row` of `columns` (rows of
    /// `width`).
    #[inline]
    pub(super) fn column<const N: usize>(
        columns: &[i16],
        width: usize,
        row: u16,
        start: usize,
    ) -> &[i16; N] {
        columns
            .get(usize::from(row) * width + start..)
            .and_then(<[i16]>::first_chunk)
            .expect(super::ROW_IN_TABLE)
    }
}

/// Outputs of a clipped layer summed (on the stack) before they are
/// clipped: the widest layer expected, so a layer is usually one block.
const DENSE_BLOCK: usize = 64;

/// The largest bias magnitude with which a dense sum of `inputs` inputs
/// cannot overflow 32 bits: `i32::MAX − inputs · 127 · 128`.
///
/// # Panics
///
/// If the products alone can overflow (`inputs` above 132 104).
#[must_use]
pub const fn nn_dense_max_bias(inputs: usize) -> i32 {
    assert!(inputs <= (i32::MAX / MAX_PRODUCT) as usize);
    i32::MAX - inputs as i32 * MAX_PRODUCT
}

/// Add (or subtract) the listed rows of `columns` (rows of `acc.len()`
/// values) to `acc`, 16-bit wrapping. Panics on a row outside the table.
pub(crate) type ColumnsFn = unsafe fn(acc: &mut [i16], columns: &[i16], rows: &[u16]);

/// `out[j] = clamp(acc[j] >> shift, 0, 127)`; the lengths are equal and a
/// multiple of [`NN_ACCUMULATOR_ALIGN`], `shift` at most
/// [`NN_MAX_ACCUMULATOR_SHIFT`].
pub(crate) type ClipI16Fn = unsafe fn(acc: &[i16], shift: u32, out: &mut [u8]);

/// `out[o] = bias[o] + Σ_i w[o · x.len() + i] · x[i]`, 32-bit wrapping;
/// `x` holds a nonzero multiple of [`NN_DENSE_INPUT_ALIGN`] values within
/// [`NN_ACTIVATION_MAX`], `w` one row per output.
pub(crate) type DenseFn = unsafe fn(x: &[u8], w: &[i8], bias: &[i32], out: &mut [i32]);

/// `out[o] = clamp(sums[o] >> shift, 0, 127)`; the lengths are equal (any
/// number), `shift` at most [`NN_MAX_DENSE_SHIFT`].
pub(crate) type ClipI32Fn = unsafe fn(sums: &[i32], shift: u32, out: &mut [u8]);

/// The network kernels of one ISA.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Kernels {
    add_columns: ColumnsFn,
    sub_columns: ColumnsFn,
    clip_i16: ClipI16Fn,
    dense: DenseFn,
    clip_i32: ClipI32Fn,
}

/// The kernels of `isa`.
pub(crate) fn kernels(isa: Isa) -> Kernels {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => Kernels {
            add_columns: avx2::update_columns::<false>,
            sub_columns: avx2::update_columns::<true>,
            clip_i16: avx2::clip_i16,
            dense: avx2::dense,
            clip_i32: avx2::clip_i32,
        },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => Kernels {
            add_columns: avx512::update_columns::<false>,
            sub_columns: avx512::update_columns::<true>,
            clip_i16: avx512::clip_i16,
            dense: avx512::dense,
            clip_i32: avx512::clip_i32,
        },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => Kernels {
            add_columns: neon::update_columns::<false>,
            sub_columns: neon::update_columns::<true>,
            clip_i16: neon::clip_i16,
            dense: neon::dense,
            clip_i32: neon::clip_i32,
        },
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => Kernels {
            add_columns: simd128::update_columns::<false>,
            sub_columns: simd128::update_columns::<true>,
            clip_i16: simd128::clip_i16,
            dense: simd128::dense,
            clip_i32: simd128::clip_i32,
        },
        _ => Kernels {
            add_columns: update_columns::<false>,
            sub_columns: update_columns::<true>,
            clip_i16,
            dense,
            clip_i32,
        },
    }
}

/// The scalar reference [`ColumnsFn`], adding or subtracting.
fn update_columns<const SUBTRACT: bool>(acc: &mut [i16], columns: &[i16], rows: &[u16]) {
    let width = acc.len();
    for &row in rows {
        let column = columns
            .get(usize::from(row) * width..)
            .and_then(|from_row| from_row.get(..width))
            .expect(ROW_IN_TABLE);
        for (sum, &value) in acc.iter_mut().zip(column) {
            *sum = if SUBTRACT {
                sum.wrapping_sub(value)
            } else {
                sum.wrapping_add(value)
            };
        }
    }
}

/// The scalar reference [`ClipI16Fn`].
fn clip_i16(acc: &[i16], shift: u32, out: &mut [u8]) {
    for (out, &value) in out.iter_mut().zip(acc) {
        *out = (value >> shift).clamp(0, i16::from(NN_ACTIVATION_MAX)) as u8;
    }
}

/// The scalar reference [`ClipI32Fn`].
fn clip_i32(sums: &[i32], shift: u32, out: &mut [u8]) {
    for (out, &sum) in out.iter_mut().zip(sums) {
        *out = (sum >> shift).clamp(0, i32::from(NN_ACTIVATION_MAX)) as u8;
    }
}

/// `Σ_i row[i] · x[i]`, 32-bit wrapping: one output of the scalar reference.
fn dot(x: &[u8], row: &[i8]) -> i32 {
    row.iter().zip(x).fold(0i32, |sum, (&weight, &input)| {
        sum.wrapping_add(i32::from(weight) * i32::from(input))
    })
}

/// The scalar reference [`DenseFn`].
fn dense(x: &[u8], w: &[i8], bias: &[i32], out: &mut [i32]) {
    for ((out, &bias), row) in out.iter_mut().zip(bias).zip(w.chunks_exact(x.len())) {
        *out = bias.wrapping_add(dot(x, row));
    }
}

/// The accumulator contract of [`ColumnsFn`] (rows are checked as they are
/// read).
fn check_columns(acc: &[i16], columns: &[i16]) {
    assert!(
        !acc.is_empty() && acc.len().is_multiple_of(NN_ACCUMULATOR_ALIGN),
        "accumulator width"
    );
    assert!(columns.len().is_multiple_of(acc.len()), "column table");
}

/// The contract of [`DenseFn`] for `outputs` outputs.
fn check_dense(x: &[u8], w: &[i8], bias: &[i32], outputs: usize) {
    assert!(
        !x.is_empty() && x.len().is_multiple_of(NN_DENSE_INPUT_ALIGN),
        "dense input count"
    );
    assert_eq!(w.len(), outputs * x.len(), "one weight row per output");
    assert_eq!(bias.len(), outputs, "one bias per output");
    // Larger inputs can saturate the AVX2 pair sums, so the ISAs would differ.
    debug_assert!(x.iter().all(|&input| input <= NN_ACTIVATION_MAX));
}

impl Dsp {
    /// Add rows of a column table to a network accumulator: `acc[j] +=
    /// columns[row · W + j]` for every `row` of `rows` (repeats count
    /// again), `W = acc.len()`. The sums wrap in 16 bits; the caller's
    /// weight bound is what keeps them from doing so.
    ///
    /// # Panics
    ///
    /// If `W` is zero or not a multiple of [`NN_ACCUMULATOR_ALIGN`],
    /// `columns` is not whole rows of `W`, or a row is outside the table.
    pub fn nn_add_columns_i16(&self, acc: &mut [i16], columns: &[i16], rows: &[u16]) {
        check_columns(acc, columns);
        // SAFETY: the table matches this CPU; the kernel indexes its slices.
        unsafe { (self.nn.add_columns)(acc, columns, rows) }
    }

    /// Remove rows of a column table from a network accumulator: as
    /// [`Self::nn_add_columns_i16`] with `acc[j] -= columns[row · W + j]`
    /// (an incremental update subtracts the features that left and adds
    /// those that arrived).
    ///
    /// # Panics
    ///
    /// As [`Self::nn_add_columns_i16`].
    pub fn nn_sub_columns_i16(&self, acc: &mut [i16], columns: &[i16], rows: &[u16]) {
        check_columns(acc, columns);
        // SAFETY: as for `nn_add_columns_i16`.
        unsafe { (self.nn.sub_columns)(acc, columns, rows) }
    }

    /// The accumulator's clipped activation: `out[j] = clamp(acc[j] >>
    /// shift, 0, `[`NN_ACTIVATION_MAX`]`)`, the shift arithmetic.
    ///
    /// # Panics
    ///
    /// If the lengths differ or are not a multiple of
    /// [`NN_ACCUMULATOR_ALIGN`], or `shift` exceeds
    /// [`NN_MAX_ACCUMULATOR_SHIFT`].
    pub fn nn_clip_i16_to_u8(&self, acc: &[i16], shift: u32, out: &mut [u8]) {
        assert_eq!(acc.len(), out.len(), "one activation per sum");
        assert!(acc.len().is_multiple_of(NN_ACCUMULATOR_ALIGN));
        assert!(shift <= NN_MAX_ACCUMULATOR_SHIFT);
        // SAFETY: the table matches this CPU; the kernel indexes its slices.
        unsafe { (self.nn.clip_i16)(acc, shift, out) }
    }

    /// A linear dense layer (the output layer): `out[o] = bias[o] + Σ_i
    /// w[o · IN + i] · x[i]` in 32 bits, `IN = x.len()`. Every `x` is at
    /// most [`NN_ACTIVATION_MAX`] (checked in debug builds only: larger
    /// values give ISA-dependent results); `w` is any `i8`. A bias within
    /// ±[`nn_dense_max_bias`]`(IN)` cannot overflow; beyond it the sum
    /// wraps.
    ///
    /// # Panics
    ///
    /// If `IN` is zero or not a multiple of [`NN_DENSE_INPUT_ALIGN`], or
    /// `w` and `bias` do not hold `IN` weights and one bias per output.
    pub fn nn_dense_linear(&self, x: &[u8], w: &[i8], bias: &[i32], out: &mut [i32]) {
        check_dense(x, w, bias, out.len());
        // SAFETY: the table matches this CPU; the kernel indexes its slices.
        unsafe { (self.nn.dense)(x, w, bias, out) }
    }

    /// A dense layer with a clipped activation: `out[o] = clamp((bias[o] +
    /// Σ_i w[o · IN + i] · x[i]) >> shift, 0, `[`NN_ACTIVATION_MAX`]`)`, the
    /// sum as [`Self::nn_dense_linear`] and the shift arithmetic.
    ///
    /// # Panics
    ///
    /// As [`Self::nn_dense_linear`], or if `shift` exceeds
    /// [`NN_MAX_DENSE_SHIFT`].
    pub fn nn_dense_clipped(&self, x: &[u8], w: &[i8], bias: &[i32], shift: u32, out: &mut [u8]) {
        check_dense(x, w, bias, out.len());
        assert!(shift <= NN_MAX_DENSE_SHIFT);
        let mut block = [0i32; DENSE_BLOCK];
        let blocks = out
            .chunks_mut(DENSE_BLOCK)
            .zip(bias.chunks(DENSE_BLOCK))
            .zip(w.chunks(DENSE_BLOCK * x.len()));
        for ((out, bias), rows) in blocks {
            let sums = &mut block[..out.len()];
            // SAFETY: the table matches this CPU; the kernels index their
            // slices.
            unsafe {
                (self.nn.dense)(x, rows, bias, sums);
                (self.nn.clip_i32)(sums, shift, out);
            }
        }
    }
}
