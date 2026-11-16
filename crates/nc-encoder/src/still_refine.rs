//! What refining a still picture costs (`rate.refine_cost`), and how far its
//! inter frames stand above the last key (the quiet run-up's strength).
//!
//! On a still scene an inter frame's size is set by how much finer its
//! quantizer is than the quality the picture already holds, not by the
//! inter model: a frame at the held quality costs a few hundred bytes, one
//! a few QP finer re-codes the whole picture. The per-role models learn the
//! cheap frames and miss the refinements by 5-60x (AV1 Fast,
//! `long_static_chirla_s024c2` 300 kbps: 150-260 kbit against 2-37 kbit
//! predicted).
//!
//! [`RefineCost`] tracks the held quality as a quantizer step, and prices a
//! refinement to step `s` as an intra coding of the difference: the last
//! key's cost scale `bits · step^e` times `s^-e − held^-e`, times the still
//! share and [`RefineCost::scale`]. On the chirla probe it predicts frames
//! over 100 kbit within 0.8-1.5x at scale 0.5 (the role models: 0.02-0.26x).
//!
//! [`KeyQuality`] measures the dip at a periodic key as it builds: a key can
//! take at most the strict buffer, so on a still scene the inter frames
//! refine for a whole key period past what the key reaches (AV1 Fast,
//! `pool_otw_homes_00004` 300 kbps: P frames 31 -> 37 dB over 10 s; the key
//! fills the buffer from empty and codes 31.1 dB).

/// Growth of the held step per frame: camera noise and small changes coded
/// at the coarser level wear the refinement away. Offline fit on the same
/// rate logs: 1.02 gave the smallest error on the large frames of chirla and
/// `otw_homes` (1.0 and 1.04 were 0.1-0.5 octave worse).
const HELD_DECAY_PER_FRAME: f64 = 1.02;

/// The held quality of a still picture and the price of refining it.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RefineCost {
    /// `rate.refine_cost`: share of an intra coding a refinement costs
    /// (0: no prediction, the role models alone).
    pub scale: f64,
    /// Finest step the still picture holds (`None` before the first key).
    held_step: Option<f64>,
    /// The last key's `bits · step^e`: what intra coding costs at a step.
    key_cost: Option<f64>,
}

impl RefineCost {
    pub fn new(scale: f64) -> Self {
        Self {
            scale,
            ..Self::default()
        }
    }

    /// A frame at `step` was coded in `bits`; `exponent` is the rate model's.
    pub fn observe(&mut self, key: bool, step: f64, bits: f64, exponent: f64) {
        if key {
            self.key_cost = Some(bits * step.powf(exponent));
            self.held_step = Some(step);
            return;
        }
        // Every coded frame holds at least its own step: on moving content
        // the held step follows the frame's, and cannot wear away without
        // bound (2026-11-16: grown 50x over a crowd scene's 300 frames, it
        // held wildtrack_cam3's inter frames 4-6 dB down).
        let worn = self.held_step.map(|held| held * HELD_DECAY_PER_FRAME);
        self.held_step = Some(worn.map_or(step, |held| held.min(step)));
    }

    /// The finest step the still picture holds.
    #[cfg(test)]
    pub fn held_step(&self) -> Option<f64> {
        self.held_step
    }

    /// Predicted bits of refining the still picture to `step` on a frame
    /// with `still_share` unchanged blocks (0 when off or not finer than the
    /// held quality).
    pub fn price(&self, step: f64, still_share: f64, exponent: f64) -> f64 {
        match (self.held_step, self.key_cost) {
            (Some(held), Some(cost)) => {
                self.scale
                    * still_share
                    * cost
                    * (step.powf(-exponent) - held.powf(-exponent)).max(0.)
            }
            _ => 0.,
        }
    }

    /// The finest step an inter frame with `still_share` unchanged blocks
    /// may take so that its refinement fits `room` bits (`None`: any).
    pub fn finest_step(&self, room: f64, still_share: f64, exponent: f64) -> Option<f64> {
        let (held, cost) = (self.held_step?, self.key_cost?);
        let price = self.scale * still_share * cost;
        if price <= 0. || !room.is_finite() {
            return None;
        }
        // s^-e - held^-e <= room / price.
        Some((held.powf(-exponent) + room / price).powf(-1. / exponent))
    }
}

/// Frames the inter frames' running PSNR-Y averages over: the pulse guard's
/// 8-frame median before a key, over a cadence cycle's base and its others.
const INTER_DB_FRAMES: f64 = 8.;
/// dB the last key may stand above the inter frames' running PSNR-Y before
/// the quiet run-up is off; it is whole from equal quality up. Measured
/// against a run-up off from equal quality (a 1 dB ramp from 0): that left
/// HEVC Default's crowd keys (`mot17_04`, keys every 2 s) unquieted where they
/// stood level with their P frames, worst-clip pulse +0.93 dB.
const QUIET_SPAN_DB: f64 = 2.;

/// The last key's PSNR-Y and the inter frames' running PSNR-Y since it.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct KeyQuality {
    key_db: Option<f64>,
    inter_db: Option<f64>,
}

impl KeyQuality {
    /// How far the quiet run-up to a key applies (`rate.prekey_quiet`),
    /// 0..=1: by how much the inter frames' running PSNR-Y stands above
    /// the last key's: whole from equal quality up, none once the key
    /// stands [`QUIET_SPAN_DB`] above; 1 before both are measured. Where the key stands above its inter frames (lowlight:
    /// noise the inter frames do not keep) a quiet run-up only lifts the
    /// key further (AV1 Fast, `bdd_night_new` 300 kbps: pulse +2.87 ->
    /// +4.05 dB); where the inter frames refined past it (still scenes) it
    /// leaves the key the buffer (f-c3 `otw_homes_00004` 300 kbps: key
    /// 25.0 -> 30.4 dB).
    pub fn quiet_strength(&self) -> f64 {
        match (self.key_db, self.inter_db) {
            (Some(key_db), Some(inter_db)) => {
                (1. + (inter_db - key_db) / QUIET_SPAN_DB).clamp(0., 1.)
            }
            _ => 1.,
        }
    }

    /// dB the inter frames' running PSNR-Y stands above the last key's
    /// (negative below), once both are measured.
    pub fn inter_gain_db(&self) -> Option<f64> {
        Some(self.inter_db? - self.key_db?)
    }

    /// A frame was coded at luma MSE `mse`.
    pub fn observe(&mut self, key: bool, mse: f64) {
        let db = psnr(mse);
        if key {
            self.key_db = Some(db);
            self.inter_db = None;
        } else {
            let running = self.inter_db.unwrap_or(db);
            self.inter_db = Some(running + (db - running) / INTER_DB_FRAMES);
        }
    }
}

/// PSNR of an 8-bit picture at luma MSE `mse`.
fn psnr(mse: f64) -> f64 {
    const PEAK_SQUARED: f64 = 255. * 255.;
    10. * (PEAK_SQUARED / mse.max(f64::EPSILON)).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refinement_is_priced_from_the_held_quality() {
        let mut cost = RefineCost::new(0.5);
        cost.observe(true, 100., 200_000., 1.);
        // Held at the key's step: a frame may refine to the step whose
        // half-intra difference fits the room.
        let finest = cost.finest_step(50_000., 1., 1.).unwrap();
        // 0.5 x 2e7 x (1/s - 1/100) = 5e4 -> 1/s = 0.015.
        assert!((finest - 1. / 0.015).abs() < 1e-9, "{finest}");
    }

    #[test]
    fn the_held_quality_wears_away_and_follows_finer_frames() {
        let mut cost = RefineCost::new(0.5);
        cost.observe(true, 100., 1., 1.);
        cost.observe(false, 200., 1., 1.);
        assert_eq!(cost.held_step(), Some(100. * HELD_DECAY_PER_FRAME));
        cost.observe(false, 50., 1., 1.);
        assert_eq!(cost.held_step(), Some(50.));
        // A coarser frame leaves it to wear away.
        cost.observe(false, 400., 1., 1.);
        assert_eq!(cost.held_step(), Some(50. * HELD_DECAY_PER_FRAME));
    }

    #[test]
    fn the_quiet_follows_how_far_the_inter_frames_stand_above_the_key() {
        let mut quality = KeyQuality::default();
        assert_eq!(quality.quiet_strength(), 1.);
        quality.observe(true, psnr_mse(30.));
        quality.observe(false, psnr_mse(29.));
        assert!((quality.quiet_strength() - 0.5).abs() < 1e-9);
        // A key above its inter frames: no quiet.
        quality.observe(true, psnr_mse(40.));
        quality.observe(false, psnr_mse(35.));
        assert_eq!(quality.quiet_strength(), 0.);
    }

    fn psnr_mse(db: f64) -> f64 {
        255. * 255. / 10f64.powf(db / 10.)
    }
}
