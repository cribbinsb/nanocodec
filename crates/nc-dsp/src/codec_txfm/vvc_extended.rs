//! H.266 inverse scaling and transforms with extended precision
//! (`sps_extended_precision_flag`, §7.4.3.22): coefficients and the first
//! transform stage span `log2TransformRange` bits (`Max(15, Min(20,
//! BitDepth + 6))`), wider than the 16 bits of the shared kernels
//! (`vvc_inverse`, `vvc_primary`, `vvc_lfnst`), which therefore cannot
//! serve these blocks. Scalar references only: the range extension
//! profiles that use them are not speed-critical here. Arithmetic follows
//! VTM `Quant::dequant` and `TrQuant::xIT` (BSD-3-Clause; `THIRD_PARTY.md`).
use super::vvc_primary::VvcTransformKind;
use super::{HEVC_LEVEL_SCALE, VVC_RECT_LEVEL_SCALE, VvcLfnst};
use crate::DepthKernels;

/// The scaling factor `m` of a flat scaling list (H.266 §8.7.3).
const FLAT_FACTOR: i64 = 16;
/// `bdShift = BitDepth + rectNonTsFlag + (Log2(nTbW) + Log2(nTbH)) / 2 +
/// 10 − log2TransformRange + sh_dep_quant_used_flag` (§8.7.3): the constant.
const SCALING_SHIFT_BASE: i32 = 10;
/// Transform skip's `bdShift` (§8.7.3: `rectNonTsFlag` and the size term
/// vanish, the transform shift is zero).
const SKIP_SCALING_SHIFT: i32 = 10;
/// The first (vertical) stage's shift (§8.7.4.1).
const FIRST_SHIFT: u32 = 7;
/// The second stage's shift is `log2TransformRange + SECOND_SHIFT_BASE −
/// BitDepth` (VTM `shift_2nd`: matrix shift 6, less one moved to the first).
const SECOND_SHIFT_BASE: u32 = 5;
/// The frequencies a transform keeps in each direction (§7.3.11.11 zero-out).
const MAX_FREQUENCIES: usize = 32;

/// One transform block's extended-precision inverse.
#[derive(Clone, Copy, Debug)]
pub struct VvcExtendedBlock<'a> {
    /// Width and height.
    pub size: (usize, usize),
    /// `Qp′` and `sh_dep_quant_used_flag`.
    pub quantizer: (u8, bool),
    /// The explicit scaling factors `m` (row-major), if any.
    pub matrix: Option<&'a [u8]>,
    /// `log2TransformRange`.
    pub log2_range: u32,
}

impl VvcExtendedBlock<'_> {
    /// `CoeffMin` and `CoeffMax`.
    fn range(&self) -> (i64, i64) {
        (-(1 << self.log2_range), (1 << self.log2_range) - 1)
    }

    /// Scale `levels` (§8.7.3) into `out`, `bd_shift` given.
    fn scale(&self, levels: &[i32], bd_shift: i32, rect: bool, out: &mut [i32]) {
        let (qp, dependent) = self.quantizer;
        let qp = qp + u8::from(dependent);
        let scales = if rect {
            VVC_RECT_LEVEL_SCALE
        } else {
            HEVC_LEVEL_SCALE
        };
        let scale = i64::from(scales[usize::from(qp % 6)]) << (qp / 6);
        let (low, high) = self.range();
        for (i, (out, &level)) in out.iter_mut().zip(levels).enumerate() {
            let factor = self.matrix.map_or(FLAT_FACTOR, |m| i64::from(m[i]));
            let product = i64::from(level) * factor * scale;
            let value = if bd_shift > 0 {
                (product + (1 << (bd_shift - 1))) >> bd_shift
            } else {
                product << -bd_shift
            };
            *out = value.clamp(low, high) as i32;
        }
    }
}

/// Round `sum` by `shift` bits.
fn rounded(sum: i64, shift: u32) -> i64 {
    (sum + (1 << (shift - 1))) >> shift
}

/// Saturate a residual to the 16 bits of the residual buffers (VTM `Pel`).
fn residual(value: i64) -> i16 {
    value.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

impl DepthKernels<'_> {
    /// Scaling, the optional inverse LFNST and the inverse primary transform
    /// `kinds` of one block with extended precision (§8.7.2 to §8.7.4):
    /// the residual, saturated to 16 bits.
    ///
    /// # Panics
    /// If the size, kinds or buffers are invalid.
    pub fn vvc_inverse_extended(
        self,
        block: VvcExtendedBlock<'_>,
        (kinds, lfnst): ([VvcTransformKind; 2], Option<VvcLfnst>),
        levels: &[i32],
        out: &mut [i16],
    ) {
        let (width, height) = block.size;
        assert!(kinds[0].valid(width) && kinds[1].valid(height));
        let length = width * height;
        assert!(levels.len() >= length && out.len() >= length);
        let depth = self.depth().bits() as i32;
        let (log2_w, log2_h) = (width.ilog2() as i32, height.ilog2() as i32);
        let rect = (log2_w + log2_h) & 1 == 1;
        // VTM clips the transform shift (`log2TransformRange − BitDepth −
        // mean`) at zero with extended precision.
        let transform_shift =
            (block.log2_range as i32 - depth - i32::midpoint(log2_w, log2_h)).max(0);
        let bd_shift =
            SCALING_SHIFT_BASE - transform_shift + i32::from(rect) + i32::from(block.quantizer.1);
        let mut coefficients = vec![0i32; length];
        block.scale(&levels[..length], bd_shift, rect, &mut coefficients);
        let (low, high) = block.range();
        if let Some(lfnst) = lfnst {
            super::vvc_lfnst::inverse_clipped(block.size, lfnst, &mut coefficients, |value| {
                i64::from(value).clamp(low, high) as i32
            });
        }
        let second = block.log2_range + SECOND_SHIFT_BASE - depth as u32;
        if width == 1 || height == 1 {
            let (size, kind) = if width == 1 {
                (height, kinds[1])
            } else {
                (width, kinds[0])
            };
            for (x, target) in out[..size].iter_mut().enumerate() {
                let sum = (0..kind.frequencies(size))
                    .map(|k| i64::from(kind.basis(size, k, x)) * i64::from(coefficients[k]))
                    .sum();
                *target = residual(rounded(sum, second + 1));
            }
            return;
        }
        let [horizontal, vertical] = kinds;
        let columns = horizontal.frequencies(width).min(MAX_FREQUENCIES);
        let rows = vertical.frequencies(height).min(MAX_FREQUENCIES);
        let mut intermediate = vec![0i64; height * columns];
        for y in 0..height {
            for x in 0..columns {
                let sum: i64 = (0..rows)
                    .map(|k| {
                        i64::from(vertical.basis(height, k, y))
                            * i64::from(coefficients[k * width + x])
                    })
                    .sum();
                intermediate[y * columns + x] = rounded(sum, FIRST_SHIFT).clamp(low, high);
            }
        }
        for y in 0..height {
            for x in 0..width {
                let sum: i64 = (0..columns)
                    .map(|k| {
                        i64::from(horizontal.basis(width, k, x)) * intermediate[y * columns + k]
                    })
                    .sum();
                out[y * width + x] = residual(rounded(sum, second));
            }
        }
    }

    /// Transform skip with extended precision (§8.7.2, §8.7.3): optional
    /// BDPCM accumulation of the levels (`bdpcm` 1 horizontal, 2 vertical,
    /// clipped to the coefficient range), scaling with `bdShift` 10, and the
    /// scaled coefficients as the residual, saturated to 16 bits.
    ///
    /// # Panics
    /// If the size or buffers are invalid.
    pub fn vvc_inverse_skip_extended(
        self,
        block: VvcExtendedBlock<'_>,
        bdpcm: u8,
        levels: &[i32],
        out: &mut [i16],
    ) {
        let (width, height) = block.size;
        let length = width * height;
        assert!(levels.len() >= length && out.len() >= length && bdpcm <= 2);
        let (low, high) = block.range();
        let mut accumulated = vec![0i32; length];
        for i in 0..length {
            let previous = match bdpcm {
                1 if i % width != 0 => accumulated[i - 1],
                2 if i >= width => accumulated[i - width],
                _ => 0,
            };
            accumulated[i] = (i64::from(levels[i]) + i64::from(previous)).clamp(low, high) as i32;
        }
        let flat = VvcExtendedBlock {
            matrix: None,
            quantizer: (block.quantizer.0, false),
            ..block
        };
        let mut scaled = vec![0i32; length];
        flat.scale(&accumulated, SKIP_SCALING_SHIFT, false, &mut scaled);
        for (out, &value) in out[..length].iter_mut().zip(&scaled) {
            *out = residual(i64::from(value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::BitDepth;

    /// At `log2TransformRange` 15 the extended path is the 16-bit one.
    #[test]
    fn range_fifteen_matches_the_shared_kernels() {
        let dsp = crate::Dsp::new();
        let kernels = dsp.at(BitDepth::TEN);
        for &(width, height) in &[(4, 4), (8, 4), (16, 16), (32, 8)] {
            let levels: Vec<i32> = (0..width * height)
                .map(|i| ((i * 37 % 23) as i32 - 11) * i32::from(i < 24))
                .collect();
            let kinds = [VvcTransformKind::Dct2; 2];
            let mut want = vec![0i16; width * height];
            kernels.vvc_inverse_primary((width, height), kinds, &levels, 40, false, &mut want);
            let mut got = vec![0i16; width * height];
            let block = VvcExtendedBlock {
                size: (width, height),
                quantizer: (40, false),
                matrix: None,
                log2_range: 15,
            };
            kernels.vvc_inverse_extended(block, (kinds, None), &levels, &mut got);
            assert_eq!(got, want, "{width}x{height}");
        }
    }
}
