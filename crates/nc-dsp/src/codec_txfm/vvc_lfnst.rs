//! H.266 low-frequency non-separable inverse transform in shared DSP.
//! Normative tables and rearrangement follow BSD-Clear `VVdeC` (`THIRD_PARTY.md`).
use super::vvc_primary::{self, VvcTransformKind};
use crate::Dsp;
#[path = "vvc_lfnst_tables.rs"]
mod tables;

/// Low-frequency transform selection and the associated intra prediction mode.
#[derive(Clone, Copy, Debug)]
pub struct VvcLfnst {
    /// Signalled nonzero LFNST index (one or two).
    pub index: u8,
    /// Ordinary intra direction, or planar for matrix prediction.
    pub direction: u8,
    /// Prediction block dimensions; ISP uses its CU size for wide angles.
    pub prediction_size: (usize, usize),
}

fn inverse(size: (usize, usize), transform: VvcLfnst, coefficients: &mut [i32]) {
    let (width, height) = size;
    let (pred_width, pred_height) = transform.prediction_size;
    let wide =
        crate::vvc_intra::rectangular::wide_angle(pred_width, pred_height, transform.direction);
    // The spectral wide-angle numbering uses a 67-mode subtraction for
    // tall blocks; prediction uses 65. Preserve that two-mode difference.
    let wide = if transform.direction > 1 && wide < i32::from(transform.direction) {
        wide - 2
    } else {
        wide
    };
    let mode = if wide < 0 {
        wide + 81
    } else if wide >= 67 {
        wide + 14
    } else {
        wide
    } as usize;
    let transpose = mode >= 81 || (mode < 67 && mode > 34);
    let set = usize::from(tables::MODE_SET[mode]);
    let index = usize::from(transform.index - 1);
    let large = width >= 8 && height >= 8;
    let matrix: &[[i8; 16]] = if large {
        &tables::LFNST8[set][index]
    } else {
        &tables::LFNST4[set][index]
    };
    let mut input = [0; 16];
    let mut i = 0;
    for diagonal in 0usize..7 {
        for x in diagonal.saturating_sub(3)..=diagonal.min(3) {
            let y = diagonal - x;
            input[i] = coefficients[y * width + x];
            i += 1;
        }
    }
    let active = if width == height && [4, 8].contains(&width) {
        8
    } else {
        16
    };
    let side = if large { 8 } else { 4 };
    let mut position = 0;
    for y in 0..side {
        for x in 0..if y < 4 { side } else { 4 } {
            let sum: i32 = matrix[position][..active]
                .iter()
                .zip(&input[..active])
                .map(|(&basis, &value)| i32::from(basis) * value)
                .sum();
            let destination = if transpose {
                x * width + y
            } else {
                y * width + x
            };
            coefficients[destination] = ((sum + 64) >> 7).clamp(-32768, 32767);
            position += 1;
        }
    }
}

impl Dsp {
    /// Apply inverse scaling, LFNST and DCT-II to an 8-bit VVC transform block.
    /// # Panics
    /// If dimensions, LFNST selection, QP or buffer sizes are invalid.
    pub fn vvc_inverse_lfnst(
        &self,
        size: (usize, usize),
        transform: VvcLfnst,
        levels: &[i32],
        qp: u8,
        dependent: bool,
        out: &mut [i32],
    ) {
        let (width, height) = size;
        let valid = |n: usize| n.is_power_of_two() && (4..=64).contains(&n);
        assert!(valid(width) && valid(height) && qp <= 63);
        assert!([1, 2].contains(&transform.index) && transform.direction <= 66);
        assert!(valid(transform.prediction_size.0) && valid(transform.prediction_size.1));
        let length = width * height;
        assert!(levels.len() >= length && out.len() >= length);
        let mut scaled = [0; 4096];
        let scaling = super::vvc_inverse::scaling(size, qp, dependent);
        self.hevc_dequantize(&levels[..length], &scaling, &mut scaled[..length]);
        inverse(size, transform, &mut scaled[..length]);
        vvc_primary::inverse(size, [VvcTransformKind::Dct2; 2], &scaled[..length], out);
    }
}

#[cfg(test)]
#[path = "vvc_lfnst_tests.rs"]
mod tests;
