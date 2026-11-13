//! The psychovisual term of mode decisions (`decision.psy_strength`), shared
//! by the engine's inter ranking and any backend decision made on
//! predictions (AV2's interpolation filter choice).
//!
//! As x264/x265's psy-rd: a candidate whose prediction holds less (or more)
//! AC energy than the source pays `strength · |E(source) − E(prediction)|`,
//! with E a block's `n · variance` (its energy about its own mean). That is
//! in squared-error units, the same as the ranked distortion, so it adds to
//! a cost directly. Mean-removed residual distortion alone prefers smooth
//! predictions (sub-sample interpolation, smoothing filters, DC intra) that
//! blur texture; the term keeps candidates that preserve its energy.

use nc_common::PlaneView;
use nc_dsp::{BlockSize, Dsp};

/// `n · variance` of the `size` block `view` (at most 64x64): its AC energy.
#[must_use]
pub fn ac_energy(dsp: &Dsp, size: BlockSize, view: &PlaneView<'_>) -> u64 {
    let area = (size.width() * size.height()) as u32;
    u64::from(dsp.block_sums(size, view).n_variance(area))
}

/// The psychovisual penalty, in squared-error units, of a prediction with
/// AC energy `prediction_energy` for a source block with `source_energy`.
#[must_use]
pub fn psy_penalty(strength: f32, source_energy: u64, prediction_energy: u64) -> f32 {
    strength * (source_energy as f32 - prediction_energy as f32).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_is_the_variance_about_the_mean() {
        let dsp = Dsp::new();
        let size = BlockSize::new(8, 8).unwrap();
        let flat = [77u8; 64];
        let flat = PlaneView::new(&flat, 8, 8, 8).unwrap();
        assert_eq!(ac_energy(&dsp, size, &flat), 0);
        // Alternating 0/2: deviation 1 everywhere, energy 64.
        let alternating: Vec<u8> = (0..64)
            .map(|i| if (i + i / 8) % 2 == 0 { 0 } else { 2 })
            .collect();
        let alternating = PlaneView::new(&alternating, 8, 8, 8).unwrap();
        assert_eq!(ac_energy(&dsp, size, &alternating), 64);
        assert_eq!(psy_penalty(0.5, 64, 0), 32.0);
        assert_eq!(psy_penalty(0.5, 0, 64), 32.0);
    }
}
