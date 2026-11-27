//! AV1 forward transforms of every type and size: the encoder-side mirror of
//! [`crate::av1_inverse`]. The 1-D forward DCT4–64 (shared with the tuned
//! `DCT_DCT` kernels of the `txfm` module), ADST4/8/16, flipped ADST and
//! identity4–32, combined into the 2-D forward of all 19 transform sizes
//! and 16 transform types.
//!
//! [`crate::Dsp::av1_forward`] is the single entry point. Square `DCT_DCT`
//! goes to the `txfm` module's tuned SIMD kernels; every other size and type
//! dispatches through [`kernel`] to the active ISA's SIMD driver
//! (`av1_forward_simd`, generic over [`I32Lanes`]) or this module's scalar
//! reference. The encoder's forward is not normative (the
//! decoder inverts whatever was coded), but its scale must be libaom's: the
//! quantiser tables assume it.
//!
//! Arithmetic follows libaom's `fwd_txfm2d_c` and `av1_fadst4/8/16`,
//! `av1_fidentity*_c` (`av1/encoder/av1_fwd_txfm2d.c`, `av1_fwd_txfm1d.c`,
//! BSD-2-Clause, Copyright (c) 2016, Alliance for Open Media; see
//! `THIRD_PARTY.md`), for 8-bit video:
//!
//! - **Column pass:** the residual column (reversed for a vertical
//!   `FLIPADST`) shifted left by the size's input shift, the 1-D column
//!   transform, a rounding shift by the size's middle shift.
//! - **Row pass:** the row (reversed for a horizontal `FLIPADST`), the 1-D
//!   row transform, a rounding shift by the size's output shift, and for 2:1
//!   rectangles a scale by `sqrt(2)` (`Round2(x * 5793, 12)`).
//! - **64-point zero-out:** only the lowest 32 frequencies of a 64-sample
//!   dimension are kept (AV1 §5.11.39 codes no others).
//!
//! The ADST graphs are written as the inverse's are (`av1_inverse::one_d`):
//! a signed input permutation, alternating rotations and add/sub butterflies
//! on ever larger blocks, an output permutation. VP9's ADST (`vp9_txfm`) is
//! the same family but a different integer graph (14-bit constants, no
//! intermediate rounding shifts), and AV2's (`av2_txfm`) is a matrix
//! transform: the arithmetic differs, so they are not shared.
//!
//! Coefficient layout is libaom's column-major one, as in the `txfm` and
//! `av1_inverse` modules: coefficient (row `r`, column `c`) of the coded
//! `coded_width x coded_height` area is `out[c * coded_height + r]`, each
//! saturated to 16 bits (8-bit residuals never reach the bound; `txfm`
//! module "Ranges").

use core::mem::MaybeUninit;

use crate::av1_inverse::{Av1TxSize, Av1TxType, Kernel1d};
use crate::simd::I32Lanes;
use crate::txfm::{cospi, dct};

/// `round(2^12 * sqrt(2))`: the 2:1 rectangle and identity4/16 scale.
pub(crate) const SQRT2: i32 = 5793;
/// Fractional bits of [`SQRT2`].
pub(crate) const SQRT2_BITS: u32 = 12;
/// Largest transform side.
const MAX_SIDE: usize = 64;

/// libaom `fwd_shift_*`: (input left shift, middle rounding shift, output
/// rounding shift), indexed like [`Av1TxSize::ALL`].
pub(crate) const SHIFTS: [(u32, u32, u32); 19] = [
    (2, 0, 0), // 4x4
    (2, 1, 0), // 8x8
    (2, 2, 0), // 16x16
    (2, 4, 0), // 32x32
    (0, 2, 2), // 64x64
    (2, 1, 0), // 4x8
    (2, 1, 0), // 8x4
    (2, 2, 0), // 8x16
    (2, 2, 0), // 16x8
    (2, 4, 0), // 16x32
    (2, 4, 0), // 32x16
    (0, 2, 2), // 32x64
    (2, 4, 2), // 64x32
    (2, 1, 0), // 4x16
    (2, 1, 0), // 16x4
    (2, 2, 0), // 8x32
    (2, 2, 0), // 32x8
    (0, 2, 0), // 16x64
    (2, 4, 0), // 64x16
];

/// libaom `av1_fwd_cos_bit_col` / `_row`, indexed by `[width_log2 -
/// 2][height_log2 - 2]` (0 where the size does not exist).
pub(crate) const COS_BIT_COL: [[u32; 5]; 5] = [
    [13, 13, 13, 0, 0],
    [13, 13, 13, 12, 0],
    [13, 13, 13, 12, 13],
    [0, 13, 13, 12, 13],
    [0, 0, 13, 12, 13],
];
pub(crate) const COS_BIT_ROW: [[u32; 5]; 5] = [
    [13, 13, 12, 0, 0],
    [13, 13, 13, 12, 0],
    [13, 13, 12, 13, 12],
    [0, 12, 13, 12, 11],
    [0, 0, 12, 11, 10],
];

/// libaom `av1_sinpi_arr_data` at 12 and 13 bits (the 4-point ADST's
/// constants; [`SINPI_13`] is the 4-point forward's usual precision).
const SINPI_12: [i32; 5] = [0, 1321, 2482, 3344, 3803];
const SINPI_13: [i32; 5] = [0, 2642, 4964, 6689, 7606];

/// Input order and signs of the 8- and 16-point ADST (libaom stage 1):
/// `x[k] = sign[k] * input[order[k]]`.
const ADST8_INPUT: [usize; 8] = [0, 7, 3, 4, 1, 6, 2, 5];
const ADST8_NEGATED: [bool; 8] = [false, true, true, false, true, false, false, true];
const ADST16_INPUT: [usize; 16] = [0, 15, 7, 8, 3, 12, 4, 11, 1, 14, 6, 9, 2, 13, 5, 10];
const ADST16_NEGATED: [bool; 16] = [
    false, true, true, false, true, false, false, true, true, false, false, true, false, true,
    true, false,
];
/// Output order of the 8- and 16-point ADST (libaom's last stage):
/// `output[k] = x[order[k]]`.
const ADST8_OUTPUT: [usize; 8] = [1, 6, 3, 4, 5, 2, 7, 0];
const ADST16_OUTPUT: [usize; 16] = [1, 14, 3, 12, 5, 10, 7, 8, 9, 6, 11, 4, 13, 2, 15, 0];
/// The first rotation of every ADST: by `pi / 4`.
const QUARTER_PI: usize = 32;

/// Rotate the pair `x[i], x[i + 1]` by `angle` (in `pi / 128`):
/// `(cos * a + sin * b, sin * a - cos * b)`.
#[allow(clippy::inline_always)]
#[inline(always)]
fn rotate<V: I32Lanes>(x: &mut [V], i: usize, angle: usize, cospi: &[i32; 64], bit: u32) {
    let (a, b) = (x[i], x[i + 1]);
    x[i] = V::half_btf(cospi[angle], a, cospi[64 - angle], b, bit);
    x[i + 1] = V::half_btf(cospi[64 - angle], a, -cospi[angle], b, bit);
}

/// The mirrored rotation of the second half of a rotation stage:
/// `(-sin * a + cos * b, cos * a + sin * b)`.
#[allow(clippy::inline_always)]
#[inline(always)]
fn rotate_mirrored<V: I32Lanes>(x: &mut [V], i: usize, angle: usize, cospi: &[i32; 64], bit: u32) {
    let (a, b) = (x[i], x[i + 1]);
    x[i] = V::half_btf(-cospi[64 - angle], a, cospi[angle], b, bit);
    x[i + 1] = V::half_btf(cospi[angle], a, cospi[64 - angle], b, bit);
}

/// Add/sub butterflies within each block of `block` values: the first half
/// becomes the sums, the second half the differences.
#[allow(clippy::inline_always)]
#[inline(always)]
fn butterflies<V: I32Lanes>(x: &mut [V], block: usize) {
    let half = block / 2;
    for start in (0..x.len()).step_by(block) {
        for i in start..start + half {
            let (a, b) = (x[i], x[i + half]);
            x[i] = a.add(b);
            x[i + half] = a.sub(b);
        }
    }
}

/// The signed input permutation of the 8- and 16-point ADST.
#[allow(clippy::inline_always)]
#[inline(always)]
fn permute_input<V: I32Lanes, const N: usize>(
    input: &[V; N],
    order: &[usize; N],
    negated: &[bool; N],
) -> [V; N] {
    let mut out = [V::splat(0); N];
    for k in 0..N {
        let x = input[order[k]];
        out[k] = if negated[k] { V::splat(0).sub(x) } else { x };
    }
    out
}

/// Forward ADST4 (libaom `av1_fadst4`).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn fadst4<V: I32Lanes>(input: &[V; 4], output: &mut [V; 4], bit: u32) {
    let sinpi = if bit == 12 { &SINPI_12 } else { &SINPI_13 };
    let [x0, x1, x2, x3] = *input;
    let a = x0
        .mul_scalar(sinpi[1])
        .add(x1.mul_scalar(sinpi[2]))
        .add(x3.mul_scalar(sinpi[4]));
    let b = x0.add(x1).sub(x3).mul_scalar(sinpi[3]);
    let c = x0
        .mul_scalar(sinpi[4])
        .sub(x1.mul_scalar(sinpi[1]))
        .add(x3.mul_scalar(sinpi[2]));
    let d = x2.mul_scalar(sinpi[3]);
    output[0] = a.add(d).round_shift(bit);
    output[1] = b.round_shift(bit);
    output[2] = c.sub(d).round_shift(bit);
    output[3] = c.sub(a).add(d).round_shift(bit);
}

/// Forward ADST8 (libaom `av1_fadst8`).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn fadst8<V: I32Lanes>(input: &[V; 8], output: &mut [V; 8], bit: u32) {
    let cospi = cospi(bit);
    let mut x = permute_input(input, &ADST8_INPUT, &ADST8_NEGATED);
    rotate(&mut x, 2, QUARTER_PI, cospi, bit);
    rotate(&mut x, 6, QUARTER_PI, cospi, bit);
    butterflies(&mut x, 4);
    rotate(&mut x, 4, 16, cospi, bit);
    rotate_mirrored(&mut x, 6, 16, cospi, bit);
    butterflies(&mut x, 8);
    for (pair, angle) in [4, 20, 36, 52].into_iter().enumerate() {
        rotate(&mut x, 2 * pair, angle, cospi, bit);
    }
    for (dst, &src) in output.iter_mut().zip(&ADST8_OUTPUT) {
        *dst = x[src];
    }
}

/// Forward ADST16 (libaom `av1_fadst16`).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn fadst16<V: I32Lanes>(input: &[V; 16], output: &mut [V; 16], bit: u32) {
    let cospi = cospi(bit);
    let mut x = permute_input(input, &ADST16_INPUT, &ADST16_NEGATED);
    for i in [2, 6, 10, 14] {
        rotate(&mut x, i, QUARTER_PI, cospi, bit);
    }
    butterflies(&mut x, 4);
    rotate(&mut x, 4, 16, cospi, bit);
    rotate_mirrored(&mut x, 6, 16, cospi, bit);
    rotate(&mut x, 12, 16, cospi, bit);
    rotate_mirrored(&mut x, 14, 16, cospi, bit);
    butterflies(&mut x, 8);
    rotate(&mut x, 8, 8, cospi, bit);
    rotate(&mut x, 10, 40, cospi, bit);
    rotate_mirrored(&mut x, 12, 8, cospi, bit);
    rotate_mirrored(&mut x, 14, 40, cospi, bit);
    butterflies(&mut x, 16);
    for (pair, angle) in [2, 10, 18, 26, 34, 42, 50, 58].into_iter().enumerate() {
        rotate(&mut x, 2 * pair, angle, cospi, bit);
    }
    for (dst, &src) in output.iter_mut().zip(&ADST16_OUTPUT) {
        *dst = x[src];
    }
}

/// Forward identity of `n` points (libaom `av1_fidentity*_c`): a scale by
/// `sqrt(2)` (4), 2 (8), `2 * sqrt(2)` (16) or 4 (32).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn fidentity<V: I32Lanes>(input: &[V], output: &mut [V]) {
    let n = input.len();
    for (out, &x) in output.iter_mut().zip(input) {
        *out = match n {
            4 => x.mul_scalar(SQRT2).round_shift(SQRT2_BITS),
            8 => x.shl(1),
            16 => x.mul_scalar(2 * SQRT2).round_shift(SQRT2_BITS),
            _ => x.shl(2),
        };
    }
}

/// One 1-D forward transform of `input.len()` points (4 to 64) with
/// `cos_bit`-bit constants.
fn forward_1d(kernel: Kernel1d, cos_bit: u32, input: &[i32], output: &mut [i32]) {
    fn fixed<const N: usize>(x: &[i32]) -> &[i32; N] {
        x.try_into().expect("transform length")
    }
    fn fixed_mut<const N: usize>(x: &mut [i32]) -> &mut [i32; N] {
        x.try_into().expect("transform length")
    }
    let cospi = cospi(cos_bit);
    match (kernel, input.len()) {
        (Kernel1d::Dct, 4) => dct::fdct4(fixed(input), fixed_mut(output), cospi, cos_bit),
        (Kernel1d::Dct, 8) => dct::fdct8(fixed(input), fixed_mut(output), cospi, cos_bit),
        (Kernel1d::Dct, 16) => dct::fdct16(fixed(input), fixed_mut(output), cospi, cos_bit),
        (Kernel1d::Dct, 32) => dct::fdct32(fixed(input), fixed_mut(output), cospi, cos_bit),
        (Kernel1d::Dct, 64) => dct::fdct64(fixed(input), fixed_mut(output), cospi, cos_bit),
        (Kernel1d::Adst | Kernel1d::FlipAdst, 4) => {
            fadst4(fixed(input), fixed_mut(output), cos_bit);
        }
        (Kernel1d::Adst | Kernel1d::FlipAdst, 8) => {
            fadst8(fixed(input), fixed_mut(output), cos_bit);
        }
        (Kernel1d::Adst | Kernel1d::FlipAdst, 16) => {
            fadst16(fixed(input), fixed_mut(output), cos_bit);
        }
        (Kernel1d::Identity, _) => fidentity(input, output),
        (kernel, n) => unreachable!("no {n}-point {kernel:?} (checked by Av1TxType::fits)"),
    }
}

/// One ISA's general AV1 2-D forward transform kernel.
pub(crate) type Av1ForwardFn = unsafe fn(Av1TxSize, Av1TxType, &[i16], usize, &mut [i16]);

/// Select the general AV1 2-D forward transform kernel for `isa`.
pub(crate) fn kernel(isa: crate::Isa) -> Av1ForwardFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        crate::Isa::Avx512 => crate::av1_forward_simd::avx512,
        #[cfg(target_arch = "x86_64")]
        crate::Isa::Avx2 => crate::av1_forward_simd::avx2,
        #[cfg(target_arch = "aarch64")]
        crate::Isa::Neon => crate::av1_forward_simd::neon,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        crate::Isa::Simd128 => crate::av1_forward_simd::simd128,
        _ => forward,
    }
}

/// The scalar reference 2-D forward of `tx` and `kind` (module docs): the
/// `width x height` residual, row-major at `stride`, into the coded area of
/// column-major `out`.
///
/// The caller checks `kind.fits(tx)` and the slice extents.
pub(crate) fn forward(
    tx: Av1TxSize,
    kind: Av1TxType,
    residual: &[i16],
    stride: usize,
    out: &mut [i16],
) {
    let (width, height) = (tx.width(), tx.height());
    let index = tx.index().expect("a valid transform size");
    let (input_shift, mid_shift, output_shift) = SHIFTS[index];
    let w_idx = usize::from(tx.width_log2 - 2);
    let h_idx = usize::from(tx.height_log2 - 2);
    let (cos_bit_col, cos_bit_row) = (COS_BIT_COL[w_idx][h_idx], COS_BIT_ROW[w_idx][h_idx]);
    let (vertical, horizontal) = kind.kernels();
    let two_to_one = width == 2 * height || height == 2 * width;

    // Column outputs, row-major (`columns[r * width + c]`), left
    // uninitialised as the `txfm` drivers leave theirs: zeroing 16 KiB per
    // call was a fifth of a key's instructions with rectangles on (memset,
    // callgrind 2026-11-22); the column pass writes every one the row pass
    // reads.
    let mut columns = [MaybeUninit::<i32>::uninit(); MAX_SIDE * MAX_SIDE];
    let mut line = [0_i32; MAX_SIDE];
    let mut transformed = [0_i32; MAX_SIDE];
    for c in 0..width {
        for (r, value) in line[..height].iter_mut().enumerate() {
            let source = if vertical == Kernel1d::FlipAdst {
                height - 1 - r
            } else {
                r
            };
            *value = i32::from(residual[source * stride + c]) << input_shift;
        }
        forward_1d(
            vertical,
            cos_bit_col,
            &line[..height],
            &mut transformed[..height],
        );
        let column = if horizontal == Kernel1d::FlipAdst {
            width - 1 - c
        } else {
            c
        };
        for (r, &value) in transformed[..height].iter().enumerate() {
            columns[r * width + column].write(value.round_shift(mid_shift));
        }
    }
    // SAFETY: the column pass wrote all `width * height` leading entries
    // (every column `c`, each row `r`), and `MaybeUninit<i32>` has the
    // layout of `i32`.
    let columns: &[i32] =
        unsafe { core::slice::from_raw_parts(columns.as_ptr().cast(), width * height) };

    let coded_height = tx.coded_height();
    for r in 0..coded_height {
        forward_1d(
            horizontal,
            cos_bit_row,
            &columns[r * width..(r + 1) * width],
            &mut transformed[..width],
        );
        for c in 0..tx.coded_width() {
            let mut value = transformed[c].round_shift(output_shift);
            if two_to_one {
                value = value.mul_scalar(SQRT2).round_shift(SQRT2_BITS);
            }
            out[c * coded_height + r] =
                value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dsp;
    use nc_common::PlaneViewMut;

    /// A deterministic 8-bit residual of `width x height`, row-major.
    fn residual(width: usize, height: usize, seed: u32) -> Vec<i16> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..width * height)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state % 511) as i16) - 255
            })
            .collect()
    }

    #[test]
    fn dct_matches_the_tuned_square_kernels() {
        let dsp = Dsp::new();
        for log2 in 2..=6_u8 {
            let tx = Av1TxSize {
                width_log2: log2,
                height_log2: log2,
            };
            let n = tx.width();
            let input = residual(n, n, u32::from(log2));
            let mut expected = vec![0_i16; tx.coded_area()];
            match crate::DctSize::from_side(n) {
                Some(size) => dsp.av1_forward_dct(size, &input, n, &mut expected),
                None => dsp.av1_forward_dct_64(&input, n, &mut expected),
            }
            let mut actual = vec![0_i16; tx.coded_area()];
            forward(tx, Av1TxType::DctDct, &input, n, &mut actual);
            assert_eq!(actual, expected, "{n}x{n}");
        }
    }

    /// Forward, then the normative inverse of the unquantised coefficients
    /// (libaom's forward has the inverse's scale: its quantiser and
    /// dequantiser cancel), gives the residual back up to the rounding of
    /// both passes, for every size and type: the forward has the inverse's
    /// orientation, flips and scale.
    #[test]
    fn every_size_and_type_inverts() {
        const BASE: u8 = 128;
        const TOLERANCE: i32 = 2;
        let dsp = Dsp::new();
        for tx in Av1TxSize::ALL {
            let (width, height) = (tx.width(), tx.height());
            // 64-point dimensions drop their upper frequencies, so a smooth
            // residual (which has none) is the fair test there.
            let large = width == MAX_SIDE || height == MAX_SIDE;
            let input: Vec<i16> = residual(width, height, 7)
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if large {
                        ((i % width) as i16 + (i / width) as i16) - 60
                    } else {
                        v / 2
                    }
                })
                .collect();
            for kind in Av1TxType::ALL.into_iter().filter(|kind| kind.fits(tx)) {
                let mut coeffs = vec![0_i16; tx.coded_area()];
                forward(tx, kind, &input, width, &mut coeffs);
                let mut pixels = vec![BASE; width * height];
                let mut view = PlaneViewMut::new(&mut pixels, width, width, height).unwrap();
                dsp.av1_inverse_add_i16(tx, kind, &mut coeffs, tx.coded_area(), &mut view);
                let worst = pixels
                    .iter()
                    .zip(&input)
                    .map(|(&p, &s)| (i32::from(p) - i32::from(BASE) - i32::from(s)).abs())
                    .max()
                    .unwrap();
                assert!(worst <= TOLERANCE, "{tx:?} {kind:?}: worst error {worst}");
            }
        }
    }
}
