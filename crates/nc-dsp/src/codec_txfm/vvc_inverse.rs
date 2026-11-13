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
    out: &mut [i32],
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
    out: &mut [i32],
) {
    let scaling = scaling((width, height), qp, dependent);
    let mut scaled = [0; AREA];
    dsp.hevc_dequantize(
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

impl Dsp {
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
        out: &mut [i32],
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
        self.hevc_dequantize(&accumulated[..length], &scaling, &mut out[..length]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Isa;
    fn reference(width: usize, height: usize, levels: &[i32], qp: u8, dependent: bool) -> Vec<i32> {
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
                out[y * width + x] = ((sum + 2048) >> 12) as i32;
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
