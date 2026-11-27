//! The secondary transforms of the general AV2 decoder (AVM's intra
//! secondary transform, IST): a non-separable matrix that maps the first 8
//! (4x4 kernel) or 32 (8x8 kernel) dequantized coefficients of an intra
//! block to its top-left 4x4 or 8x8 low frequencies, applied before the
//! primary inverse transform ([`super::itx`]).
//!
//! Ported from dav2d `src/stx_tmpl.c` (`stxfm_c`, `stxfm4_c`, `stxfm8_c`),
//! `src/stx_tables.c` (generated into [`super::stx_tables`]) and the kernel
//! selection of `src/recon_tmpl.c`, commit 237229e; Copyright © 2018-2026
//! `VideoLAN` and dav2d authors, BSD-2-Clause (`docs/licenses/dav2d`).
//! dav2d is bit-exact with AVM v1.0.0.
//!
//! Coefficients are in [`super::itx`]'s column-major layout with column
//! stride `min(h, 32)`; the inputs are the first `eob + 1` entries of the
//! buffer, as dav2d's coefficient reader leaves them for a secondary
//! transform block.

use super::itx::RectTxSize;
use crate::{Dsp, Isa};

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx512;
use super::stx_tables::{COEFF8X8_MAPPING, COEFF8X8_MAPPING_IDX, STX4_KERNELS, STX8_KERNELS};

/// Inputs of the 4x4 kernels.
pub const STX4_INPUTS: usize = 8;
/// Outputs of a 4x4 kernel: the top-left 4x4 coefficients.
pub const STX4_OUTPUTS: usize = 16;
/// Inputs of the 8x8 kernels.
pub const STX8_INPUTS: usize = 32;
/// Outputs of an 8x8 kernel: 48 of the top-left 8x8 coefficients, placed
/// by [`COEFF8X8_MAPPING`].
pub const STX8_OUTPUTS: usize = 48;

/// Side of the 4x4 and 8x8 output squares.
const SIDE4: usize = 4;
const SIDE8: usize = 8;
/// Largest column stride (the largest coded height).
const MAX_STRIDE: usize = 32;

/// Kernel types per set (dav2d `stx & 3`, minus one).
pub const STX_TYPES: usize = 3;
/// Sets of the 4x4 kernels.
pub const STX4_SETS: usize = 14;
/// Sets of the 8x8 kernels.
pub const STX8_SETS: usize = 11;

/// Fractional bits of the kernel weights.
const KERNEL_SHIFT: u32 = 7;

/// The end of block [`super::itx::inverse_transform_add`] takes after a 4x4
/// kernel: dav2d signals only the 4x4 corner (scan index 9 closes its first
/// four rows) so the primary transform takes its fast path.
pub const EOB_AFTER_STX4: u16 = 9;
/// The end of block after an 8x8 kernel (dav2d `recon_tmpl.c`).
pub const EOB_AFTER_STX8: u16 = 35;

/// dav2d `enum IntraPredMode` numbers of the horizontal-ish modes whose
/// secondary transform output is not transposed (`recon_tmpl.c`).
const HOR_PRED: u8 = 2;
const HOR_DOWN_PRED: u8 = 6;
const VERT_LEFT_PRED: u8 = 8;
const SMOOTH_H_PRED: u8 = 11;

/// The bits of dav2d's packed `stx` value: the type (1-based) in bits 0-1,
/// the set in bits 2-5.
const STX_TYPE_MASK: u8 = 0x3;
const STX_SET_SHIFT: u32 = 2;
const STX_SET_MASK: u8 = 0xf;

/// dav2d `stxfm_c`: the matrix product of the first `eob + 1` inputs with a
/// kernel of `N` outputs, each rounded half away from zero and clamped to
/// 16 bits (8-bit video: `±128 · 256`).
fn matrix_product<const N: usize>(cf: &[i16], kernel: &[i8]) -> [i16; N] {
    let mut out = [0; N];
    for (x, o) in out.iter_mut().enumerate() {
        let sum: i32 = cf
            .iter()
            .zip(kernel[x..].iter().step_by(N))
            .map(|(&c, &k)| i32::from(c) * i32::from(k))
            .sum();
        let magnitude = (sum.abs() + (1 << (KERNEL_SHIFT - 1))) >> KERNEL_SHIFT;
        *o = (magnitude * sum.signum()).clamp(i16::MIN.into(), i16::MAX.into()) as i16;
    }
    out
}

/// dav2d `stxfm4_c`: the 4x4 kernel `kernel` (8 inputs by 16 outputs) on
/// the first `eob + 1 <= 8` coefficients of `cf`, written to the top-left
/// 4x4 of the column-major block of column stride `stride` (4 to 32);
/// `transpose` writes output `y * 4 + x` to row `x`, column `y` instead of
/// row `y`, column `x`. Inputs outside the 4x4 are zeroed.
pub fn secondary4(
    cf: &mut [i16],
    stride: usize,
    transpose: bool,
    kernel: &[i8; STX4_INPUTS * STX4_OUTPUTS],
    eob: u16,
) {
    secondary4_for_isa(Isa::Scalar, cf, stride, transpose, kernel, eob);
}

fn secondary4_for_isa(
    isa: Isa,
    cf: &mut [i16],
    stride: usize,
    transpose: bool,
    kernel: &[i8; STX4_INPUTS * STX4_OUTPUTS],
    eob: u16,
) {
    check_stride(stride, SIDE4);
    assert!(usize::from(eob) < STX4_INPUTS, "4x4 IST end of block {eob}");
    let sums = secondary_matrix::<STX4_OUTPUTS, 1>(isa, &cf[..=usize::from(eob)], kernel);
    cf[SIDE4..STX4_INPUTS].fill(0);
    for (n, &sum) in sums.iter().enumerate() {
        let (y, x) = (n / SIDE4, n % SIDE4);
        let position = if transpose {
            y * stride + x
        } else {
            x * stride + y
        };
        cf[position] = sum;
    }
}

/// dav2d `stxfm8_c`: the 8x8 kernel `kernel` (32 inputs by 48 outputs) on
/// the first `eob + 1 <= 32` coefficients of `cf`, written to the top-left
/// 8x8 at the raster positions of mapping `map_idx` (see [`kernel8`]) in a
/// column-major block of column stride `stride` (8 to 32), transposed as
/// for [`secondary4`]. The first 32 entries are zeroed first.
pub fn secondary8(
    cf: &mut [i16],
    stride: usize,
    transpose: bool,
    map_idx: usize,
    kernel: &[i8; STX8_INPUTS * STX8_OUTPUTS],
    eob: u16,
) {
    secondary8_for_isa(Isa::Scalar, cf, stride, (transpose, map_idx), kernel, eob);
}

#[allow(unsafe_code)]
fn secondary8_for_isa(
    isa: Isa,
    cf: &mut [i16],
    stride: usize,
    (transpose, map_idx): (bool, usize),
    kernel: &[i8; STX8_INPUTS * STX8_OUTPUTS],
    eob: u16,
) {
    check_stride(stride, SIDE8);
    assert!(usize::from(eob) < STX8_INPUTS, "8x8 IST end of block {eob}");
    let inputs = usize::from(eob) + 1;
    #[cfg(target_arch = "x86_64")]
    if isa == Isa::Avx512 {
        // SAFETY: the dispatch validated the full Ice Lake feature set.
        unsafe { avx512::secondary8(cf, stride, (transpose, map_idx), kernel, inputs) };
        return;
    }
    let sums = secondary_matrix::<STX8_OUTPUTS, 3>(isa, &cf[..inputs], kernel);
    cf[..STX8_INPUTS].fill(0);
    for (&raster, &sum) in COEFF8X8_MAPPING[map_idx].iter().zip(&sums) {
        let (y, x) = (usize::from(raster) / SIDE8, usize::from(raster) % SIDE8);
        let position = if transpose {
            y * stride + x
        } else {
            x * stride + y
        };
        cf[position] = sum;
    }
}

#[allow(unsafe_code)]
fn secondary_matrix<const N: usize, const GROUPS: usize>(
    isa: Isa,
    cf: &[i16],
    kernel: &[i8],
) -> [i16; N] {
    #[cfg(target_arch = "x86_64")]
    if matches!(isa, Isa::Avx2 | Isa::Avx512) {
        // SAFETY: both tiers include AVX2 and were validated by Dsp.
        return unsafe { avx2::matrix::<N, GROUPS>(cf, kernel) };
    }
    let _ = isa;
    matrix_product::<N>(cf, kernel)
}

fn check_stride(stride: usize, side: usize) {
    assert!(
        stride.is_power_of_two() && (side..=MAX_STRIDE).contains(&stride),
        "IST column stride {stride}"
    );
}

/// The 4x4 kernel of set `set` and 0-based type `ty` (dav2d
/// `dav2d_stx_4x4_kernel[set][type]`).
#[must_use]
pub fn kernel4(set: usize, ty: usize) -> &'static [i8; STX4_INPUTS * STX4_OUTPUTS] {
    &STX4_KERNELS[set][ty]
}

/// The 8x8 kernel of set `set` and 0-based type `ty` with its output
/// mapping index (dav2d `dav2d_stx_8x8_kernel[set][type]`,
/// `dav2d_coeff8x8_mapping_idx[set * 3 + type]`).
#[must_use]
pub fn kernel8(set: usize, ty: usize) -> (&'static [i8; STX8_INPUTS * STX8_OUTPUTS], usize) {
    (
        &STX8_KERNELS[set][ty],
        usize::from(COEFF8X8_MAPPING_IDX[set * STX_TYPES + ty]),
    )
}

/// Whether the secondary transform output is transposed (dav2d
/// `recon_tmpl.c`): always for inter and intra block copy blocks
/// (`intra_mode` `None`), else unless the luma intra mode (dav2d
/// `enum IntraPredMode` number) is horizontal-ish.
#[must_use]
pub fn transposed(intra_mode: Option<u8>) -> bool {
    !matches!(
        intra_mode,
        Some(HOR_PRED | HOR_DOWN_PRED | VERT_LEFT_PRED | SMOOTH_H_PRED)
    )
}

/// The secondary transform step of dav2d `recon_tmpl.c` for a `size` block
/// whose packed `stx` value (dav2d's: 1-based type in bits 0-1, set in bits
/// 2-5) is nonzero: picks the 8x8 kernel when both sides are at least 8,
/// else the 4x4, runs it on `cf` (column stride `size.coded_height()`), and
/// returns the end of block to pass to the primary inverse transform.
///
/// # Panics
/// If `stx` has no kernel or the arguments break [`secondary4`] /
/// [`secondary8`]'s contracts.
pub fn apply_secondary(
    cf: &mut [i16],
    size: RectTxSize,
    stx: u8,
    transpose: bool,
    eob: u16,
) -> u16 {
    apply_secondary_for_isa(Isa::Scalar, cf, size, stx, transpose, eob)
}

/// As [`apply_secondary`], with the secondary transform of the selected ISA.
/// # Panics
/// As [`apply_secondary`].
pub fn apply_secondary_with_dsp(
    dsp: &Dsp,
    cf: &mut [i16],
    size: RectTxSize,
    stx: u8,
    transpose: bool,
    eob: u16,
) -> u16 {
    apply_secondary_for_isa(dsp.isa(), cf, size, stx, transpose, eob)
}

fn apply_secondary_for_isa(
    isa: Isa,
    cf: &mut [i16],
    size: RectTxSize,
    stx: u8,
    transpose: bool,
    eob: u16,
) -> u16 {
    let ty = usize::from(stx & STX_TYPE_MASK)
        .checked_sub(1)
        .expect("IST type 0 is no secondary transform");
    let set = usize::from((stx >> STX_SET_SHIFT) & STX_SET_MASK);
    let stride = size.coded_height();
    if size.width() >= SIDE8 && size.height() >= SIDE8 {
        let (kernel, map_idx) = kernel8(set, ty);
        secondary8_for_isa(isa, cf, stride, (transpose, map_idx), kernel, eob);
        EOB_AFTER_STX8
    } else {
        secondary4_for_isa(isa, cf, stride, transpose, kernel4(set, ty), eob);
        EOB_AFTER_STX4
    }
}

#[cfg(test)]
mod tests;
