//! The key-frame planning cost of an intra leaf
//! ([`nc_encoder::CodecBackend::intra_leaf_cost`], `hevc.intra_leaf_cost`):
//! what lets the engine's key-frame planner (`planner.intra_plan` 2) choose
//! HEVC key-frame CU sizes from transform cost instead of coding one fixed
//! size, so detail gets 8x8 CUs (where `PART_NxN` can then apply) and flat
//! areas 64x64 ones.
//!
//! Each luma transform block of the leaf is predicted by DC from
//! the *source* neighbours (the reconstruction does not exist yet; at key
//! quality the two are close), then transformed, quantised as
//! `code_residual` would and priced by the CABAC estimate; the remaining
//! distortion is measured on the coefficients. The mode syntax is priced as the cheapest
//! candidate-list code. This is the HEVC form of the AV1 backend's leaf
//! cost (SVT-AV1's I-frame "light PD0 with a transform").

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::{BlockSize, HevcTransform};
use nc_encoder::Block;

use super::{HevcBackend, Scratch, transform_units};
use crate::headers::LOG2_MIN_CB;
use crate::predict::{self, ModeCode};
use crate::scan::ScanKind;

/// The forward transform's coefficients are `2^(7 − log2 n)` times the
/// orthonormal ones (HM scaling, 8-bit: `15 − BitDepth − log2 n`).
const FORWARD_GAIN_LOG2: u32 = 7;
/// Sample value of a prediction with no available neighbour (8-bit
/// `1 << (BitDepth − 1)`, §8.4.4.2.2).
const MID_SAMPLE: u32 = 128;

impl HevcBackend {
    /// [`nc_encoder::CodecBackend::intra_leaf_cost`] of `block`: luma only.
    /// Chroma follows the luma partition and adds little to its choice;
    /// leaving it out cut planning instructions by about a third.
    pub(super) fn key_leaf_cost(&self, source: &Picture, block: &Block) -> f32 {
        let part_bits = if block.log2_size() == LOG2_MIN_CB as u8 {
            self.bin_bits(self.contexts.part_mode, true)
        } else {
            0.0
        };
        let mut bits = part_bits + self.intra_mode_bits(ModeCode::Mpm(0));
        let mut distortion = 0u64;
        for &(x, y, log2) in transform_units(block, self.info.max_transform_log2).as_slice() {
            let (d, b) = self.dc_luma_cost(source, (x, y, 1 << log2));
            distortion += d;
            bits += b;
        }
        distortion as f32 + self.rd_lambda() * bits
    }

    /// Distortion and bits of one luma block (`(x, y, n)`, n ≥ 8) coded with
    /// a DC prediction from the source neighbours. The distortion is
    /// measured on the coefficients (`Σ (c − c′)²` scaled back to samples
    /// by the forward transform's gain): HEVC's DCT is orthogonal to within
    /// rounding, and this saves the inverse transform.
    fn dc_luma_cost(&self, source: &Picture, (x, y, n): (usize, usize, usize)) -> (u64, f32) {
        let log2 = n.trailing_zeros() as u8;
        let src_plane = source.plane(PlaneIndex::Y).view();
        let src = src_plane
            .sub_view(x, y, n, n)
            .expect("block inside the picture");
        let dc = source_dc(&src_plane, x, y, n);
        let flat = [dc; crate::transform::MAX_COEFFS];
        let pred = PlaneView::new(&flat, n, n, n).expect("flat prediction");
        let size = BlockSize::new(n, n).expect("transform sizes 8..32");
        let quantizer =
            &self.frame_quantization.planes[0].transforms[1][usize::from(log2 - 2)].quantizer;
        let transform = HevcTransform::dct(log2).expect("transform sizes 8..32");
        let mut scratch = self.scratch.borrow_mut();
        let Scratch {
            residual,
            levels,
            coeffs,
            scaled,
            ..
        } = &mut *scratch;
        let area = n * n;
        self.dsp
            .residual_sad(size, &src, &pred, &mut residual[..], n);
        self.dsp
            .hevc_forward(transform, &residual[..area], n, &mut coeffs[..area]);
        let nonzero = self
            .dsp
            .hevc_quantize_from(&coeffs[..area], quantizer, &mut levels[..area]);
        let mut bits = self.bin_bits(self.contexts.cbf_luma[1], nonzero > 0);
        let gain_shift = 2 * (FORWARD_GAIN_LOG2 - u32::from(log2));
        let error = if nonzero == 0 {
            coeffs[..area]
                .iter()
                .map(|&c| i64::from(c).pow(2) as u64)
                .sum::<u64>()
        } else {
            let scan = ScanKind::for_intra(predict::DC, log2, true);
            bits += self.levels_bits(&levels[..area], log2, scan, true);
            self.dsp
                .hevc_dequantize(&levels[..area], quantizer, &mut scaled[..area]);
            coeffs[..area]
                .iter()
                .zip(&scaled[..area])
                .map(|(&c, &d)| (i64::from(c) - i64::from(d)).pow(2) as u64)
                .sum::<u64>()
        };
        (error >> gain_shift, bits)
    }
}

/// The DC value (§8.4.4.2.5) of the `n`-square block at `(x, y)` predicted
/// from its source neighbours: the mean of the row above and the column to
/// the left, either side alone where the other is outside the picture, or
/// mid-grey at the picture's corner.
fn source_dc(plane: &PlaneView<'_>, x: usize, y: usize, side: usize) -> u8 {
    let above = (y > 0).then(|| {
        plane.row(y - 1)[x..x + side]
            .iter()
            .map(|&s| u32::from(s))
            .sum::<u32>()
    });
    let left = (x > 0).then(|| {
        (0..side)
            .map(|row| u32::from(plane.row(y + row)[x - 1]))
            .sum::<u32>()
    });
    let count = side as u32;
    let dc = match (above, left) {
        (Some(above), Some(left)) => (above + left + count) / (2 * count),
        (Some(sum), None) | (None, Some(sum)) => (sum + count / 2) / count,
        (None, None) => MID_SAMPLE,
    };
    dc as u8
}

#[cfg(test)]
mod tests {
    use super::FORWARD_GAIN_LOG2;
    use nc_dsp::{Dsp, HevcTransform};

    /// The coefficient-domain energy, scaled by the forward gain, equals
    /// the residual's energy (the DCT is orthogonal to within rounding).
    #[test]
    fn forward_gain_maps_coefficient_energy_to_samples() {
        let dsp = Dsp::new();
        for log2 in 3..=5u8 {
            let n = 1usize << log2;
            let residual: Vec<i16> = (0..n * n)
                .map(|i| ((i * 7919 + i / n * 31) % 101) as i16 - 50)
                .collect();
            let mut coeffs = vec![0; n * n];
            let transform = HevcTransform::dct(log2).unwrap();
            dsp.hevc_forward(transform, &residual, n, &mut coeffs);
            let samples: u64 = residual
                .iter()
                .map(|&r| (i64::from(r) * i64::from(r)) as u64)
                .sum();
            let coefficients: u64 = coeffs
                .iter()
                .map(|&c| (i64::from(c) * i64::from(c)) as u64)
                .sum();
            let scaled = coefficients >> (2 * (FORWARD_GAIN_LOG2 - u32::from(log2)));
            let ratio = scaled as f64 / samples as f64;
            assert!((0.98..1.02).contains(&ratio), "log2 {log2}: ratio {ratio}");
        }
    }
}
