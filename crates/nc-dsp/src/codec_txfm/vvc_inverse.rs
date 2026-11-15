//! H.266 rectangular inverse DCT-II through the shared HEVC scaling/DCT kernels.
//! Decoder-only parameterization follows BSD-Clear `VVdeC` Quant (`THIRD_PARTY.md`).
use super::{HevcTransform, VvcQuantizer};
use crate::Dsp;

/// Shared flat inverse scaling, including narrow ISP and 64-point transforms.
pub(crate) fn scaling(
    (width, height): (usize, usize),
    qp: u8,
    dependent: bool,
) -> super::HevcQuantizer {
    let log2_area = width.ilog2() + height.ilog2();
    let odd = log2_area & 1;
    let mean = log2_area / 2;
    let adjusted_qp = qp + u8::from(dependent);
    let mut scaling = VvcQuantizer::new(qp, 2).0;
    let scales = if odd == 0 {
        [40, 45, 51, 57, 64, 72]
    } else {
        [57, 64, 72, 80, 90, 102]
    };
    scaling.level_scale = 16 * scales[usize::from(adjusted_qp % 6)];
    scaling.per = u32::from(adjusted_qp / 6);
    scaling.bd_shift = 3 + mean + odd + u32::from(dependent);
    scaling
}

pub(crate) fn inverse(
    dsp: &Dsp,
    width: usize,
    height: usize,
    levels: &[i32],
    qp: u8,
    dependent: bool,
    out: &mut [i16],
) {
    assert!([2, 4, 8, 16, 32].contains(&width) && [2, 4, 8, 16, 32].contains(&height));
    assert!(qp <= 63 && levels.len() >= width * height && out.len() >= width * height);
    // Scratch sized to the transform's square: zeroing 32x32 buffers for
    // every small block was 4.5 % of VVC Default's x86 instructions.
    match width.max(height).max(4) {
        4 => inverse_sized::<16>(dsp, (width, height), levels, (qp, dependent), out),
        8 => inverse_sized::<64>(dsp, (width, height), levels, (qp, dependent), out),
        16 => inverse_sized::<256>(dsp, (width, height), levels, (qp, dependent), out),
        _ => inverse_sized::<1024>(dsp, (width, height), levels, (qp, dependent), out),
    }
}

/// [`inverse`] of a block whose enclosing square transform has `AREA`
/// samples.
fn inverse_sized<const AREA: usize>(
    dsp: &Dsp,
    (width, height): (usize, usize),
    levels: &[i32],
    (qp, dependent): (u8, bool),
    out: &mut [i16],
) {
    let scaling = scaling((width, height), qp, dependent);
    let mut scaled = [0; AREA];
    dsp.vvc_scale(
        &levels[..width * height],
        &scaling,
        &mut scaled[..width * height],
    );
    let side = AREA.isqrt();
    let transform = HevcTransform::dct(side.ilog2() as u8).unwrap();
    if width == height && width >= 4 {
        dsp.hevc_inverse(transform, &scaled, out);
        return;
    }
    // DCT-II bases nest: row k of the W-point basis is row k*N/W
    // of the N-point basis, restricted to its first W samples. Inserting
    // zero frequencies preserves both normative rounding/clipping stages.
    // The square/own-stream path never pays this rectangular adapter cost.
    let mut embedded = [0; AREA];
    for y in 0..height {
        for x in 0..width {
            embedded[(y * side / height) * side + x * side / width] = scaled[y * width + x];
        }
    }
    let mut square = [0; AREA];
    dsp.hevc_inverse(transform, &embedded, &mut square);
    for y in 0..height {
        out[y * width..(y + 1) * width].copy_from_slice(&square[y * side..y * side + width]);
    }
}

/// H.266 §8.7.3 scaling with an explicit scaling matrix: `matrix` holds the
/// factor `m` of each coefficient (row-major, 16 is the flat value) and
/// `flat` the parameters of [`scaling`]. The arithmetic and both saturating
/// clamps are those of the shared flat kernel (`hevc_scale`); a scalar loop,
/// since each coefficient has its own multiplier and only streams with
/// explicit scaling lists come here.
fn scale_with_matrix(levels: &[i32], flat: &super::HevcQuantizer, matrix: &[u8], out: &mut [i16]) {
    let unit = flat.level_scale / FLAT_SCALING_FACTOR;
    let product_limit = super::generic::hevc_dequant_product_limit(flat);
    let level_limit = super::generic::HEVC_DEQUANT_LEVEL_LIMIT;
    let round = 1 << (flat.bd_shift - 1);
    for ((scaled, &level), &factor) in out.iter_mut().zip(levels).zip(matrix) {
        // |level| <= 2^16 and factor * unit <= 255 * 102: the product fits.
        let product = (level.clamp(-level_limit, level_limit) * unit * i32::from(factor))
            .clamp(-product_limit, product_limit);
        let value = ((product << flat.per) + round) >> flat.bd_shift;
        *scaled = value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
}

/// The scaling factor `m` of a flat scaling list, folded into
/// `HevcQuantizer::level_scale`.
const FLAT_SCALING_FACTOR: i32 = 16;

impl Dsp {
    /// VVC 8-bit inverse scaling with an explicit scaling matrix, then the
    /// optional inverse LFNST and the inverse primary transform, for the
    /// decoder of streams with explicit scaling lists. `matrix` holds the
    /// scaling factor of each coefficient of the block, row-major.
    /// # Panics
    /// If a dimension is not a power of two in 2..=64, the block has fewer
    /// than four samples, QP exceeds 63, or a slice is short.
    #[allow(clippy::too_many_arguments)] // one transform block's parameters
    pub fn vvc_inverse_with_matrix(
        &self,
        size: (usize, usize),
        kinds: [crate::VvcTransformKind; 2],
        lfnst: Option<crate::VvcLfnst>,
        levels: &[i32],
        (qp, dependent): (u8, bool),
        matrix: &[u8],
        out: &mut [i16],
    ) {
        let (width, height) = size;
        let length = width * height;
        assert!(kinds[0].valid(width) && kinds[1].valid(height) && length >= 4 && qp <= 63);
        assert!(levels.len() >= length && matrix.len() >= length && out.len() >= length);
        let flat = scaling(size, qp, dependent);
        crate::vvc_mts::with_scratch(length, |scaled| {
            scale_with_matrix(&levels[..length], &flat, matrix, &mut scaled[..length]);
            if let Some(transform) = lfnst {
                super::vvc_lfnst::inverse(size, transform, &mut scaled[..length]);
            }
            super::vvc_primary::inverse(size, kinds, &scaled[..length], out);
        });
    }

    /// Invert transform skip and optional block differential residuals using
    /// the shared dequantizer. `bdpcm` is zero, one (horizontal), or two (vertical).
    /// # Panics
    /// If dimensions/buffers, QP or differential direction are invalid.
    pub fn vvc_inverse_skip(
        &self,
        (width, height): (usize, usize),
        levels: &[i32],
        qp: u8,
        bdpcm: u8,
        out: &mut [i16],
    ) {
        let valid = |n: usize| n.is_power_of_two() && (2..=32).contains(&n);
        assert!(valid(width) && valid(height) && qp <= 63 && bdpcm <= 2);
        let length = width * height;
        assert!(levels.len() >= length && out.len() >= length);
        let mut accumulated = [0i32; 1024];
        for i in 0..length {
            let previous = match bdpcm {
                1 if i % width != 0 => accumulated[i - 1],
                2 if i >= width => accumulated[i - width],
                _ => 0,
            };
            accumulated[i] =
                (i64::from(levels[i]) + i64::from(previous)).clamp(-32768, 32767) as i32;
        }
        let mut scaling = VvcQuantizer::new(qp.max(4), 2).0;
        // The existing HEVC scale includes a factor of sixteen; transform
        // skip's normative six-bit inverse scale therefore uses ten bits.
        scaling.bd_shift = 10;
        self.vvc_scale(&accumulated[..length], &scaling, &mut out[..length]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Isa;
    fn reference(width: usize, height: usize, levels: &[i32], qp: u8, dependent: bool) -> Vec<i16> {
        let area_log = width.ilog2() + height.ilog2();
        let adjusted = u32::from(qp) + u32::from(dependent);
        let shift = area_log as i32 / 2 - 1 + (area_log as i32 & 1) + i32::from(dependent)
            - (adjusted / 6) as i32;
        let scale = if area_log & 1 == 0 {
            [40, 45, 51, 57, 64, 72]
        } else {
            [57, 64, 72, 80, 90, 102]
        }[(adjusted % 6) as usize];
        let scaled: Vec<i64> = levels
            .iter()
            .map(|&v| {
                let product = i64::from(v) * scale;
                (if shift > 0 {
                    (product + (1 << (shift - 1))) >> shift
                } else {
                    product << -shift
                })
                .clamp(-32768, 32767)
            })
            .collect();
        let mut intermediate = vec![0i64; width * height];
        for y in 0..height {
            for x in 0..width {
                let sum: i64 = (0..height)
                    .map(|k| {
                        i64::from(super::super::DCT32[k * 32 / height][y]) * scaled[k * width + x]
                    })
                    .sum();
                intermediate[y * width + x] = ((sum + 64) >> 7).clamp(-32768, 32767);
            }
        }
        let mut out = vec![0; width * height];
        for y in 0..height {
            for x in 0..width {
                let sum: i64 = (0..width)
                    .map(|k| {
                        i64::from(super::super::DCT32[k * 32 / width][x])
                            * intermediate[y * width + k]
                    })
                    .sum();
                out[y * width + x] = i16::try_from((sum + 2048) >> 12).unwrap();
            }
        }
        out
    }
    #[test]
    fn rectangular_inverse_matches_direct_matrix_and_scaling() {
        let dsps = [Dsp::with_isa(Isa::Scalar).unwrap(), Dsp::new()];
        for width in [2, 4, 8, 16, 32] {
            for height in [2, 4, 8, 16, 32] {
                for qp in [0, 5, 22, 37, 63] {
                    for dependent in [false, true] {
                        for pattern in 0..3 {
                            let levels: Vec<_> = (0..width * height)
                                .map(|i| match pattern {
                                    0 => (i as i32 * 37 % 29) - 14,
                                    1 => {
                                        if i % 2 == 0 {
                                            -32768
                                        } else {
                                            32767
                                        }
                                    }
                                    _ => {
                                        if i == width * height - 1 {
                                            -13
                                        } else {
                                            0
                                        }
                                    }
                                })
                                .collect();
                            let expected = reference(width, height, &levels, qp, dependent);
                            for dsp in &dsps {
                                let mut actual = vec![0; width * height];
                                inverse(dsp, width, height, &levels, qp, dependent, &mut actual);
                                assert_eq!(
                                    actual,
                                    expected,
                                    "{width}x{height} qp={qp} dependent={dependent} pattern={pattern} isa={:?}",
                                    dsp.isa()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "vvc_skip_tests.rs"]
mod skip_tests;

#[cfg(test)]
mod matrix_tests {
    use super::*;
    use crate::{Isa, VvcTransformKind};

    /// H.266 §8.7.3 in 64-bit integers, for one coefficient.
    fn reference(level: i32, factor: u8, size: (usize, usize), qp: u8, dependent: bool) -> i16 {
        let log2_area = size.0.ilog2() + size.1.ilog2();
        let rectangular = log2_area & 1;
        let shift = 3 + log2_area / 2 + rectangular + u32::from(dependent);
        let scales = [[40, 45, 51, 57, 64, 72], [57, 64, 72, 80, 90, 102]];
        let qp = i64::from(qp) + i64::from(dependent);
        let scale = scales[rectangular as usize][(qp % 6) as usize];
        let product = (i64::from(level) * i64::from(factor) * scale) << (qp / 6);
        ((product + (1 << (shift - 1))) >> shift).clamp(-32768, 32767) as i16
    }

    #[test]
    fn matrix_scaling_matches_the_specification_equation() {
        let levels: Vec<i32> = (0..64)
            .map(|i| [0, 1, -1, 37, -915, 32767, -32768, 6000][i % 8])
            .collect();
        let matrix: Vec<u8> = (0..64).map(|i| [16, 1, 255, 8, 97][i % 5]).collect();
        for size in [(8, 8), (16, 4), (4, 8), (2, 32)] {
            for qp in [0, 5, 22, 37, 63] {
                for dependent in [false, true] {
                    let mut scaled = [0; 64];
                    scale_with_matrix(&levels, &scaling(size, qp, dependent), &matrix, &mut scaled);
                    for (index, &value) in scaled.iter().enumerate() {
                        let expected = reference(levels[index], matrix[index], size, qp, dependent);
                        assert_eq!(value, expected, "{size:?} qp {qp} {dependent} {index}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_flat_matrix_decodes_as_the_flat_path() {
        let dsp = Dsp::with_isa(Isa::Scalar).unwrap();
        let flat = [16u8; 4096];
        for size in [(4, 4), (8, 16), (32, 32), (64, 64), (16, 2)] {
            let length = size.0 * size.1;
            let levels: Vec<i32> = (0..length)
                .map(|i| {
                    if i % 64 < 6 && i < 32 * size.0 {
                        (i as i32 % 9 - 4) * 30
                    } else {
                        0
                    }
                })
                .collect();
            let (mut expected, mut actual) = (vec![0i16; length], vec![0i16; length]);
            let kinds = [VvcTransformKind::Dct2; 2];
            dsp.vvc_inverse_primary(size, kinds, &levels, 30, true, &mut expected);
            dsp.vvc_inverse_with_matrix(size, kinds, None, &levels, (30, true), &flat, &mut actual);
            assert_eq!(actual, expected, "{size:?}");
        }
    }
}
