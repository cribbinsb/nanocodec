//! Scene-cut detection: insert a key frame where the content changes.
//!
//! A cut leaves nothing for motion prediction to find, so an inter frame
//! spends at least as many bits as a key frame and then propagates a poor
//! picture. The analysis already measures what is needed: the best
//! motion-compensated SAD of every 8x8 block (`fine` motion field) and each
//! block's spatial activity (pre-analysis `sum_sq_dev`). A frame is a cut
//! when both
//!
//! 1. motion-compensated SAD exceeds `scene.cut_ratio` times the frame's
//!    intra activity (estimated SAD against the block mean), i.e. intra
//!    prediction would do at least as well; and
//! 2. that SAD per pixel jumps `scene.history_ratio` times above its recent
//!    average at the same reference distance (layered structures predict
//!    over 1, 2 and 4 frames, which differ in SAD), so sustained hard
//!    content (sensor noise, fast motion, flat noisy scenes with little
//!    activity) never fires repeatedly;
//!
//! and at least `scene.min_key_interval` frames have passed since the last
//! key (flashes and strobes).
//!
//! Measured with the defaults (1.8, 3, 8) on the 39 cut-free TRAIN+VAL
//! clips (11,001 frames): two fires, both in `bodycam_antare_0815` on an
//! abrupt exposure change and a whip pan, where motion SAD is 3.6–9.8× the
//! intra estimate (a key costs no more there). All 32 synthetic cuts
//! (pairs of different 720p clips joined at frame 60) fire on the cut frame
//! and nowhere else; real cuts measured 2.1–27× intra and 3.7–284× history.
//! Details: `docs/handoff/realuse.md`.

use nc_analysis::FrameAnalysis;

use crate::params::SceneParams;

/// Mean absolute deviation per root-mean-square deviation of Gaussian
/// samples (`sqrt(2 / pi)`): turns a block's sum of squared deviations into
/// an estimated SAD against its mean, comparable with motion SADs.
const MEAN_ABS_PER_RMS: f64 = 0.797_884_560_8;
/// Samples in an 8x8 analysis block.
const BLOCK_SAMPLES: f64 = 64.;
/// Smallest intra-activity estimate per sample: keeps flat frames (whose
/// activity is near zero) from turning sensor noise into a huge ratio.
const MIN_ACTIVITY_PER_SAMPLE: f64 = 1.;
/// Weight of the newest frame in the running SAD-per-pixel average.
const HISTORY_WEIGHT: f64 = 0.25;

/// Frame-level costs a cut decision compares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FrameCosts {
    /// Motion-compensated SAD per pixel.
    pub inter_per_pixel: f64,
    /// Estimated intra SAD per pixel (activity against block means).
    pub intra_per_pixel: f64,
}

impl FrameCosts {
    /// The costs of an analysed inter frame, or `None` without motion.
    pub fn of(analysis: &FrameAnalysis) -> Option<Self> {
        let motion = analysis.motion.as_ref()?;
        let blocks = analysis.pre.blocks.cells();
        let samples = BLOCK_SAMPLES * blocks.len() as f64;
        let inter: f64 = motion.fine.cells().iter().map(|b| f64::from(b.sad)).sum();
        let intra: f64 = blocks
            .iter()
            .map(|b| MEAN_ABS_PER_RMS * (BLOCK_SAMPLES * f64::from(b.sum_sq_dev)).sqrt())
            .sum();
        Some(Self {
            inter_per_pixel: inter / samples,
            intra_per_pixel: (intra / samples).max(MIN_ACTIVITY_PER_SAMPLE),
        })
    }
}

/// Running state of the detector.
#[derive(Debug, Clone)]
pub(crate) struct SceneDetector {
    cut_ratio: f64,
    history_ratio: f64,
    min_key_interval: u64,
    /// Recent motion-compensated SAD per pixel, per reference distance
    /// (cleared at key frames).
    history: Vec<(u32, f64)>,
}

impl SceneDetector {
    /// A detector for `params`, or `None` when detection is off.
    pub fn new(params: &SceneParams) -> Option<Self> {
        (params.detect >= 0.5).then(|| Self {
            cut_ratio: f64::from(params.cut_ratio),
            history_ratio: f64::from(params.history_ratio),
            min_key_interval: params.min_key_interval.round() as u64,
            history: Vec::new(),
        })
    }

    /// Whether an inter frame with `costs`, predicted from `distance`
    /// frames back and `since_key` frames after the last key, is a cut.
    /// Updates the history either way.
    pub fn is_cut(&mut self, costs: FrameCosts, distance: u32, since_key: u64) -> bool {
        let unpredictable = costs.inter_per_pixel > self.cut_ratio * costs.intra_per_pixel;
        let entry = self.history.iter_mut().find(|(d, _)| *d == distance);
        let jump = entry
            .as_ref()
            .is_some_and(|(_, h)| costs.inter_per_pixel > self.history_ratio * *h);
        let cut = unpredictable && jump && since_key >= self.min_key_interval;
        let value = costs.inter_per_pixel;
        match (cut, entry) {
            (true, _) => self.reset(),
            (false, Some((_, h))) => *h = *h * (1. - HISTORY_WEIGHT) + value * HISTORY_WEIGHT,
            (false, None) => self.history.push((distance, value)),
        }
        cut
    }

    /// A key frame was coded: motion SADs against the old scene say nothing
    /// about the new one.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameCosts, SceneDetector};
    use crate::params::{ParamOverrides, SceneParams};

    fn detector() -> SceneDetector {
        let mut params = ParamOverrides::default();
        params.set("scene.detect", 1.0);
        SceneDetector::new(&SceneParams::from_overrides(&params)).unwrap()
    }

    fn costs(inter: f64, intra: f64) -> FrameCosts {
        FrameCosts {
            inter_per_pixel: inter,
            intra_per_pixel: intra,
        }
    }

    impl SceneDetector {
        fn is_cut_flat(&mut self, costs: FrameCosts, since_key: u64) -> bool {
            self.is_cut(costs, 1, since_key)
        }
    }

    #[test]
    fn history_is_per_reference_distance() {
        let mut d = detector();
        for i in 1..20 {
            // Distance-4 frames are naturally costlier than distance-1 ones.
            d.is_cut(costs(1., 10.), 1, i);
            assert!(!d.is_cut(costs(6., 4.), 4, i));
        }
        assert!(d.is_cut(costs(20., 4.), 4, 20));
    }

    #[test]
    fn off_by_default() {
        assert!(
            SceneDetector::new(&SceneParams::from_overrides(&ParamOverrides::default())).is_none()
        );
    }

    #[test]
    fn fires_on_a_jump_to_unpredictable_content_only() {
        let mut d = detector();
        for i in 1..20 {
            assert!(!d.is_cut_flat(costs(2., 10.), i));
        }
        // Unpredictable but no jump (sustained noise): no cut.
        let mut noisy = detector();
        for i in 1..20 {
            assert!(!noisy.is_cut_flat(costs(12., 10.), i));
        }
        // Too soon after the last key: no cut, and the history absorbs it.
        assert!(!d.is_cut_flat(costs(25., 10.), 3));
        let mut d = detector();
        for i in 1..20 {
            d.is_cut_flat(costs(2., 10.), i);
        }
        assert!(d.is_cut_flat(costs(25., 10.), 20));
        // The history restarts after the cut.
        assert!(!d.is_cut_flat(costs(25., 10.), 1));
    }
}
