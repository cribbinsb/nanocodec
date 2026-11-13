//! AV2 (AVM v1.0.0) 2-D transforms of any transform type, 4x4 to 32x32:
//! each direction one of the DCT, identity, ADST (AVM's tuned DST-VII) or
//! flipped ADST 1-D kernels. Scalar matrix products, monomorphized per side
//! and skipping zero inputs of the inverse: the reference, and for now the
//! only implementation (`DCT_DCT` itself takes the butterfly kernels of the
//! parent module).
//!
//! Reimplemented from AVM's `inv_txfm_c` / `inv_transform_1d_c` and the
//! `inv_txfm_{dct2,idtx,adst,fdst}_size*_c` kernels (`av2/common/idct.c`),
//! `fwd_txfm_c` / `fwd_transform_1d_c` and their `fwd_txfm_*` kernels
//! (`av2/encoder/hybrid_fwd_txfm.c`), and `g_hor_tx_type` / `g_ver_tx_type`
//! (`av2/common/common_data.h`); Alliance for Open Media,
//! BSD-3-Clause-Clear with separate AOM patent terms (see `THIRD_PARTY.md`).
//! The ADST tables are generated (`adst_tables`); every (size, type) pair is
//! checked end to end against the original C by
//! `tools/codegen/check_av2_tx_types.py` (fixture `TX_TYPE_HASHES` in
//! `nc-av2`'s `transform.rs`). The inter DDT kernels are not implemented:
//! our sequence header disables them.
//!
//! The passes, shifts and clamps are those of the parent module's DCTs: the
//! forward transforms columns (vertical kernel) then rows (horizontal
//! kernel), the inverse rows then columns, clamping coefficients and the
//! row output to 16 bits and the residual to `-256..=255`.

use super::adst_tables::{ADST4, ADST8, ADST16, FDST4, FDST8, FDST16, IDENTITY_SCALES};
use super::{Av2TxSize, DCT_BASIS, MAX_POINTS};

/// Largest side of a typed transform (64x64 is `DCT_DCT` only).
const MAX_SIDE: usize = 32;
/// Largest side of the ADST kernels.
const MAX_ADST_SIDE: usize = 16;
/// 16-bit clamp of the inverse input and row pass (AVM `clamp_buf`, `bd + 8`).
const COEFF_MIN: i32 = -32768;
const COEFF_MAX: i32 = 32767;
/// Range of the inverse column pass output (AVM `inv_txfm_c`, 8-bit video).
const RESIDUAL_MIN: i32 = -256;
const RESIDUAL_MAX: i32 = 255;

/// One 1-D AV2 transform kernel (AVM `TX1D_TYPE`, without the DDTs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Av2Kernel {
    /// `DCT2`.
    Dct,
    /// `IDT`: a per-size gain, no mixing.
    Identity,
    /// `DST7`: the tuned DST-VII AVM calls ADST (4..16 points).
    Adst,
    /// `DCT8`: the flipped ADST (4..16 points).
    FlipAdst,
}

impl Av2Kernel {
    /// Whether this kernel exists at `size`.
    #[must_use]
    pub const fn supports(self, size: Av2TxSize) -> bool {
        match self {
            Self::Dct => true,
            Self::Identity => size.side() <= MAX_SIDE,
            Self::Adst | Self::FlipAdst => size.side() <= MAX_ADST_SIDE,
        }
    }
}

/// A 2-D AV2 transform type: the kernel of each direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Av2TxType {
    /// Kernel of the vertical (column) transform.
    pub vertical: Av2Kernel,
    /// Kernel of the horizontal (row) transform.
    pub horizontal: Av2Kernel,
}

impl Av2TxType {
    /// `DCT_DCT`.
    pub const DCT: Self = Self {
        vertical: Av2Kernel::Dct,
        horizontal: Av2Kernel::Dct,
    };

    /// Whether both kernels exist at `size`.
    #[must_use]
    pub const fn supports(self, size: Av2TxSize) -> bool {
        self.vertical.supports(size) && self.horizontal.supports(size)
    }
}

/// The `N`-point DCT as a dense matrix, row `k` basis vector `k`
/// (`N <= 32`, so the stored half of [`DCT_BASIS`] covers every sample).
const fn dct_matrix<const N: usize>() -> [[i32; N]; N] {
    let mut matrix = [[0; N]; N];
    let mut k = 0;
    while k < N {
        let mut j = 0;
        while j < N {
            matrix[k][j] = DCT_BASIS[k * (MAX_POINTS / N)][j] as i32;
            j += 1;
        }
        k += 1;
    }
    matrix
}

const DCT4: [[i32; 4]; 4] = dct_matrix::<4>();
const DCT8: [[i32; 8]; 8] = dct_matrix::<8>();
const DCT16: [[i32; 16]; 16] = dct_matrix::<16>();
const DCT32: [[i32; 32]; 32] = dct_matrix::<32>();

/// A kernel at one size: a dense matrix (row `k` is basis vector `k`, row
/// major) or the identity's gain.
#[derive(Clone, Copy)]
pub(super) enum Matrix {
    Dense(&'static [i32]),
    Gain(i32),
}

impl Matrix {
    /// # Panics
    /// If `kernel` does not exist at `size` (or `size` is 64x64).
    pub(super) fn of(kernel: Av2Kernel, size: Av2TxSize) -> Self {
        let n = size.side();
        assert!(
            n <= MAX_SIDE && kernel.supports(size),
            "AV2 {kernel:?} kernel has no {n}-point form"
        );
        match (kernel, n) {
            (Av2Kernel::Identity, _) => Self::Gain(IDENTITY_SCALES[size as usize]),
            (Av2Kernel::Dct, 4) => Self::Dense(DCT4.as_flattened()),
            (Av2Kernel::Dct, 8) => Self::Dense(DCT8.as_flattened()),
            (Av2Kernel::Dct, 16) => Self::Dense(DCT16.as_flattened()),
            (Av2Kernel::Dct, _) => Self::Dense(DCT32.as_flattened()),
            (Av2Kernel::Adst, 4) => Self::Dense(ADST4.as_flattened()),
            (Av2Kernel::Adst, 8) => Self::Dense(ADST8.as_flattened()),
            (Av2Kernel::Adst, _) => Self::Dense(ADST16.as_flattened()),
            (Av2Kernel::FlipAdst, 4) => Self::Dense(FDST4.as_flattened()),
            (Av2Kernel::FlipAdst, 8) => Self::Dense(FDST8.as_flattened()),
            (Av2Kernel::FlipAdst, _) => Self::Dense(FDST16.as_flattened()),
        }
    }
}

/// AVM's pass rounding: add half, shift right (arithmetic).
#[inline]
fn round_shift(value: i32, shift: u32) -> i32 {
    (value + (1 << (shift - 1))) >> shift
}

/// Forward 2-D transform of type `tx_type` (AVM `fwd_txfm_c`) of the
/// residual at `stride` into the row-major coefficients `out` (`n * n`).
///
/// # Panics
/// If `tx_type` does not exist at `size` or the slices are short.
pub(crate) fn forward(
    size: Av2TxSize,
    tx_type: Av2TxType,
    residual: &[i16],
    stride: usize,
    out: &mut [i32],
) {
    match size.side() {
        4 => forward_n::<4>(size, tx_type, residual, stride, out),
        8 => forward_n::<8>(size, tx_type, residual, stride, out),
        16 => forward_n::<16>(size, tx_type, residual, stride, out),
        _ => forward_n::<32>(size, tx_type, residual, stride, out),
    }
}

/// [`forward`] at side `N`. Both passes run along contiguous rows: the
/// column pass accumulates whole residual rows scaled by one matrix entry,
/// the row pass takes dot products of coefficient rows with basis rows
/// (integer sums, so the order matches AVM's exactly).
fn forward_n<const N: usize>(
    size: Av2TxSize,
    tx_type: Av2TxType,
    residual: &[i16],
    stride: usize,
    out: &mut [i32],
) {
    let (first, second) = size.forward_shifts();
    // The column pass's output, row-major by vertical frequency.
    let mut columns = [[0; N]; N];
    let vertical = Matrix::of(tx_type.vertical, size);
    for (k, line) in columns.iter_mut().enumerate() {
        match vertical {
            Matrix::Dense(m) => {
                for (y, &weight) in m[k * N..(k + 1) * N].iter().enumerate() {
                    let row = &residual[y * stride..y * stride + N];
                    for (sum, &sample) in line.iter_mut().zip(row) {
                        *sum += weight * i32::from(sample);
                    }
                }
            }
            Matrix::Gain(gain) => {
                let row = &residual[k * stride..k * stride + N];
                for (sum, &sample) in line.iter_mut().zip(row) {
                    *sum = gain * i32::from(sample);
                }
            }
        }
        for sum in line.iter_mut() {
            *sum = round_shift(*sum, first);
        }
    }
    let horizontal = Matrix::of(tx_type.horizontal, size);
    for (line, out) in columns.iter().zip(out.as_chunks_mut::<N>().0) {
        for (u, coefficient) in out.iter_mut().enumerate() {
            let sum = match horizontal {
                Matrix::Dense(m) => m[u * N..(u + 1) * N]
                    .iter()
                    .zip(line)
                    .map(|(&weight, &x)| weight * x)
                    .sum(),
                Matrix::Gain(gain) => gain * line[u],
            };
            *coefficient = round_shift(sum, second);
        }
    }
}

/// Inverse 2-D transform of type `tx_type` (AVM `inv_txfm_c` without the
/// pixel add) of the row-major `coeffs` into the row-major residual `out`
/// (`n * n`, each sample within `-256..=255`).
///
/// # Panics
/// If `tx_type` does not exist at `size` or the slices are short.
pub(crate) fn inverse(size: Av2TxSize, tx_type: Av2TxType, coeffs: &[i32], out: &mut [i32]) {
    match size.side() {
        4 => inverse_n::<4>(size, tx_type, coeffs, out),
        8 => inverse_n::<8>(size, tx_type, coeffs, out),
        16 => inverse_n::<16>(size, tx_type, coeffs, out),
        _ => inverse_n::<32>(size, tx_type, coeffs, out),
    }
}

/// [`inverse`] at side `N`. Both passes accumulate contiguous basis rows
/// scaled by one input value and skip zero inputs, which leave the sums
/// unchanged: quantized blocks are mostly zero, so the rows below the
/// last nonzero one are never transformed (bit-exact; integer sums).
fn inverse_n<const N: usize>(size: Av2TxSize, tx_type: Av2TxType, coeffs: &[i32], out: &mut [i32]) {
    let (first, second) = size.inverse_shifts();
    // The row pass's output, row-major; rows from `used` on are zero.
    let mut rows = [[0; N]; N];
    let mut used = 0;
    let horizontal = Matrix::of(tx_type.horizontal, size);
    for (r, (line, input)) in rows.iter_mut().zip(coeffs.as_chunks::<N>().0).enumerate() {
        if input.iter().all(|&c| c == 0) {
            continue;
        }
        used = r + 1;
        for (k, &c) in input.iter().enumerate() {
            let c = c.clamp(COEFF_MIN, COEFF_MAX);
            match horizontal {
                Matrix::Dense(m) if c != 0 => {
                    for (sum, &weight) in line.iter_mut().zip(&m[k * N..(k + 1) * N]) {
                        *sum += weight * c;
                    }
                }
                Matrix::Dense(_) => {}
                Matrix::Gain(gain) => line[k] = gain * c,
            }
        }
        for sum in line.iter_mut() {
            *sum = round_shift(*sum, first).clamp(COEFF_MIN, COEFF_MAX);
        }
    }
    let mut columns = [[0; N]; N];
    match Matrix::of(tx_type.vertical, size) {
        Matrix::Dense(m) => {
            for (k, line) in rows[..used].iter().enumerate() {
                for (y, sums) in columns.iter_mut().enumerate() {
                    let weight = m[k * N + y];
                    for (sum, &value) in sums.iter_mut().zip(line) {
                        *sum += weight * value;
                    }
                }
            }
        }
        Matrix::Gain(gain) => {
            for (sums, line) in columns.iter_mut().zip(&rows) {
                for (sum, &value) in sums.iter_mut().zip(line) {
                    *sum = gain * value;
                }
            }
        }
    }
    for (sums, out) in columns.iter().zip(out.as_chunks_mut::<N>().0) {
        for (sample, &sum) in out.iter_mut().zip(sums) {
            *sample = round_shift(sum, second).clamp(RESIDUAL_MIN, RESIDUAL_MAX);
        }
    }
}
