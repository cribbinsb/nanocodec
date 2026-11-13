//! H.266 8-bit luma mapping and chroma-scaling tables (LMCS).
//! Piecewise-linear arithmetic follows BSD-Clear `VVdeC` Reshape; see `THIRD_PARTY.md`.
use crate::Dsp;
use nc_common::PlaneViewMut;

const BINS: usize = 16;
const BIN_WIDTH: i32 = 16;
const PRECISION: u32 = 11;

/// Validated 8-bit LMCS mapping, independent of codec header storage.
#[derive(Debug)]
pub struct VvcLmcs {
    forward: [u8; 256],
    inverse: [u8; 256],
    chroma: [i32; 256],
}
impl VvcLmcs {
    /// Derive a model from the active bin interval and signed bin-width deltas.
    /// Returns `None` for a model outside the normative 8-bit bounds.
    #[must_use]
    pub fn new(bins: [u8; 2], deltas: [i16; BINS], chroma_offset: i16) -> Option<Self> {
        let [first, last] = bins.map(usize::from);
        if first > last || last >= BINS {
            return None;
        }
        let mut widths = [0; BINS];
        let mut pivots = [0; BINS + 1];
        for i in 0..BINS {
            if (first..=last).contains(&i) {
                widths[i] = BIN_WIDTH + i32::from(deltas[i]);
                if !(2..=127).contains(&widths[i])
                    || !(2..=127).contains(&(widths[i] + i32::from(chroma_offset)))
                {
                    return None;
                }
            }
            pivots[i + 1] = pivots[i] + widths[i];
            if (first..=last).contains(&i)
                && pivots[i] % 8 != 0
                && pivots[i] / 8 == pivots[i + 1] / 8
            {
                return None;
            }
        }
        if pivots[BINS] > 255 {
            return None;
        }
        let scale: [i32; BINS] = widths.map(|w| {
            if w == 0 {
                0
            } else {
                (BIN_WIDTH << PRECISION) / w
            }
        });
        let chroma_scale: [i32; BINS] = widths.map(|w| {
            if w == 0 {
                1 << PRECISION
            } else {
                (BIN_WIDTH << PRECISION) / (w + i32::from(chroma_offset))
            }
        });
        let forward = std::array::from_fn(|sample| {
            let bin = sample / BIN_WIDTH as usize;
            (pivots[bin]
                + ((widths[bin] * 128 * (sample as i32 % BIN_WIDTH) + (1 << (PRECISION - 1)))
                    >> PRECISION))
                .clamp(0, 255) as u8
        });
        let indices: [usize; 256] = std::array::from_fn(|sample| {
            (first..=last)
                .find(|&i| (sample as i32) < pivots[i + 1])
                .unwrap_or((last + 1).min(BINS - 1))
        });
        let inverse = std::array::from_fn(|sample| {
            let bin = indices[sample];
            (BIN_WIDTH * bin as i32
                + ((scale[bin] * (sample as i32 - pivots[bin]) + (1 << (PRECISION - 1)))
                    >> PRECISION))
                .clamp(0, 255) as u8
        });
        Some(Self {
            forward,
            inverse,
            chroma: indices.map(|i| chroma_scale[i]),
        })
    }
    /// Eleven-fraction-bit chroma residual multiplier at the given luma level.
    #[must_use]
    pub fn chroma_scale(&self, luma: u8) -> i32 {
        self.chroma[usize::from(luma)]
    }
}
impl Dsp {
    /// Scale 8-bit chroma residuals with the LMCS eleven-fraction-bit multiplier.
    /// Input clipping precedes symmetric rounding, as required by H.266 §8.7.5.
    pub fn vvc_scale_chroma_residual(&self, scale: i32, residual: &mut [i32]) {
        assert!((0..=16384).contains(&scale));
        for value in residual {
            let input = (*value).clamp(-256, 255);
            let scaled = input.signum() * ((input.abs() * scale + 1024) >> 11);
            *value = scaled.clamp(-32768, 32767);
        }
    }

    /// Map an inter prediction into the LMCS reconstruction domain.
    pub fn vvc_forward_lmcs(&self, model: &VvcLmcs, samples: &mut [u8]) {
        for value in samples {
            *value = model.forward[usize::from(*value)];
        }
    }
    /// Return reconstructed luma to the original domain before in-loop filtering.
    pub fn vvc_inverse_lmcs(&self, model: &VvcLmcs, plane: &mut PlaneViewMut<'_>) {
        for y in 0..plane.height() {
            for value in plane.row_mut(y) {
                *value = model.inverse[usize::from(*value)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chroma_scaling_clips_before_symmetric_rounding() {
        let mut residual = [-32768, -257, -256, -3, -1, 0, 1, 3, 255, 256, 32767];
        Dsp::new().vvc_scale_chroma_residual(3072, &mut residual);
        assert_eq!(residual, [-384, -384, -384, -5, -2, 0, 2, 5, 383, 383, 383]);
    }
    #[test]
    fn shifted_identity_preserves_rounding_and_clamps_inactive_bins() {
        let model = VvcLmcs::new([1, 15], [0; 16], 0).unwrap();
        let dsp = Dsp::new();
        let mut samples: Vec<u8> = (0..=255).collect();
        dsp.vvc_forward_lmcs(&model, &mut samples);
        assert_eq!(
            samples,
            (0..=255u8)
                .map(|v| v.saturating_sub(16))
                .collect::<Vec<_>>()
        );
        dsp.vvc_inverse_lmcs(
            &model,
            &mut PlaneViewMut::new(&mut samples, 256, 256, 1).unwrap(),
        );
        assert_eq!(samples, (0..=255u8).map(|v| v.max(16)).collect::<Vec<_>>());
        assert!((0..=255).all(|v| model.chroma_scale(v) == 2048));
    }
    #[test]
    fn model_rejects_invalid_bin_widths_and_total_range() {
        assert!(VvcLmcs::new([0, 15], [0; 16], 0).is_none());
        assert!(VvcLmcs::new([15, 1], [0; 16], 0).is_none());
        assert!(VvcLmcs::new([1, 15], [-15; 16], 0).is_none());
        assert!(VvcLmcs::new([1, 15], [0; 16], -15).is_none());
    }
}
