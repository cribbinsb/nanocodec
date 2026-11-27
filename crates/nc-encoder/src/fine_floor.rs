//! The fine-floor gate (`planner.fine_floor_*`): per inter superblock,
//! whether the smallest inter block (`planner.inter_min_log2`) is allowed
//! there, or the floor is one level higher.
//!
//! Why: 8x8 inter CUs (HEVC slow and slower, `planner.inter_min_log2` 3)
//! gain on average and lose on whole content classes (tune-slow,
//! 2026-11-23: HEVC slow quick L1 J -3.39 % with videocall +3.33 %; HEVC
//! slower L3 bodycam +4.6 %, videocall +3.6 %), at about +30 % CPU. The
//! cause, measured per clip and frame, is the camera: they lose where it
//! moves (docs/handoff/tuning.md "Fine-floor gate (gate8x8, 2026-11-23)").
//! A floor that one constant sets for every superblock cannot follow it; a
//! gate on what the motion search already measured can, and it skips the
//! planning and RD trials of the finer blocks where it closes.
//!
//! How: the floor holds where `bias + w · features ≥ 0`
//! ([`LinearThreshold`], the heuristic-containing form: all terms 0 is the
//! gate off and the plain floor everywhere, byte-identical), else it is
//! raised one level for the whole superblock, for the planner
//! (`partition_plan.rs`) and the RD partition search (`rd_split.rs`) alike.
//! The features are known before the superblock is planned:
//! - the camera's motion over the frame, ln(1 + samples per frame) of the
//!   global motion fit (translation plus zoom at the picture edge, as the
//!   moving-camera structure reads it, `moving_camera.rs`);
//! - the frame's static-block share (pre-analysis);
//! - what finer motion buys in the superblock: ln of the motion search's
//!   best 16x16 SAD sum over its best 8x8 SAD sum (SVT's `me_dist`), near 0
//!   where the 8x8 vectors predict no better than the 16x16 ones;
//! - the residual left at 16x16, ln(1 + SAD per sample / quantizer step):
//!   small where the residual quantises away whatever the block size.
//!
//! The first two carry the measured cause: a moving camera replaces the
//! picture within a second, so the finer partition's bits are not inherited
//! by the frames after it, as a fixed camera's are.
//!
//! Its parameters are registered and tuned on long-window encodes
//! (`nceval.tune`), never fitted offline.

use nc_analysis::MotionAnalysis;
use nc_analysis::linear_threshold::LinearThreshold;

use crate::params::PlannerParams;
use crate::partition::Block;

/// The gate's features: camera motion, static share, fine-motion gain,
/// residual.
pub(crate) const FEATURES: usize = 4;
/// Index of the 8x8 and 16x16 sums in [`nc_analysis::motion::SbStats::me_dist`].
const ME_DIST_8X8: usize = 0;
const ME_DIST_16X16: usize = 1;
/// Added to both SAD sums before their ratio, so a perfectly predicted
/// superblock has gain 0, not an undefined ratio.
const SAD_FLOOR: f32 = 1.0;

/// The gate for one frame, `None` while every term is 0.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FineFloorGate {
    shift: LinearThreshold<FEATURES>,
    /// The frame's camera-motion and static-share features, the same for
    /// every superblock.
    camera: f32,
    static_share: f32,
    /// The frame quantizer's sample step.
    step: f32,
}

impl FineFloorGate {
    /// The gate of an inter frame whose camera moved `camera_motion` luma
    /// samples per frame, with `static_share` of its blocks static, coded at
    /// quantizer sample `step`; `None` while it is off.
    pub fn of(
        params: &PlannerParams,
        (camera_motion, static_share): (f32, f32),
        step: f32,
    ) -> Option<Self> {
        LinearThreshold::new(
            params.fine_floor_bias,
            [
                params.fine_floor_camera,
                params.fine_floor_static,
                params.fine_floor_motion_gain,
                params.fine_floor_residual,
            ],
        )
        .map(|shift| Self {
            shift,
            camera: camera_motion.max(0.0).ln_1p(),
            static_share,
            step: step.max(f32::MIN_POSITIVE),
        })
    }

    /// Whether the floor holds in superblock `sb` of a `width` x `height`
    /// picture: its features summed over the motion search's superblocks it
    /// covers. A superblock the search has no statistics for keeps the
    /// floor (the plain planner).
    pub fn opens(
        &self,
        motion: Option<&MotionAnalysis>,
        sb: &Block,
        (width, height): (usize, usize),
    ) -> bool {
        let Some(motion) = motion else {
            return true;
        };
        let size = nc_analysis::motion::SB;
        let last = (
            (sb.x + sb.width()).min(width).saturating_sub(1) / size,
            (sb.y + sb.height()).min(height).saturating_sub(1) / size,
        );
        let (mut fine, mut whole, mut samples) = (0.0_f32, 0.0_f32, 0usize);
        for row in sb.y / size..=last.1 {
            for col in sb.x / size..=last.0 {
                let Some(stats) = motion.sb.try_get(col as isize, row as isize) else {
                    return true;
                };
                fine += stats.me_dist[ME_DIST_8X8] as f32;
                whole += stats.me_dist[ME_DIST_16X16] as f32;
                // The search's sums cover the visible samples only.
                let (x, y) = (col * size, row * size);
                samples += size.min(width - x) * size.min(height - y);
            }
        }
        let features = [
            self.camera,
            self.static_share,
            ((whole + SAD_FLOOR) / (fine + SAD_FLOOR)).ln(),
            (whole / samples.max(1) as f32 / self.step).ln_1p(),
        ];
        self.shift.at(&features) >= 0.0
    }
}

/// `params` with the floor raised one level (where the gate closes).
#[must_use]
pub(crate) fn raised(params: &PlannerParams) -> PlannerParams {
    PlannerParams {
        inter_min_log2: params.inter_min_log2.round() + 1.0,
        ..*params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(settings: &[(&str, f32)]) -> PlannerParams {
        let mut overrides = crate::params::ParamOverrides::default();
        for &(name, value) in settings {
            overrides.set(name, value);
        }
        PlannerParams::default().with_overrides(&overrides)
    }

    #[test]
    fn all_zero_is_off() {
        assert!(FineFloorGate::of(&params(&[]), (3.0, 0.5), 4.0).is_none());
    }

    #[test]
    fn a_superblock_without_motion_statistics_keeps_the_floor() {
        let gate = FineFloorGate::of(
            &params(&[("planner.fine_floor_bias", -1.0)]),
            (0.0, 0.0),
            4.0,
        )
        .expect("on");
        assert!(gate.opens(None, &Block::square(0, 0, 6), (64, 64)));
    }

    #[test]
    fn camera_motion_closes_the_floor() {
        let settings = [
            ("planner.fine_floor_bias", 1.0),
            ("planner.fine_floor_camera", -1.0),
        ];
        let p = params(&settings);
        let still = FineFloorGate::of(&p, (0.5, 0.8), 4.0).expect("on");
        let moving = FineFloorGate::of(&p, (8.0, 0.1), 4.0).expect("on");
        assert!(
            still
                .shift
                .at(&[still.camera, still.static_share, 0.0, 0.0])
                >= 0.0
        );
        assert!(
            moving
                .shift
                .at(&[moving.camera, moving.static_share, 0.0, 0.0])
                < 0.0
        );
    }

    #[test]
    fn raising_moves_the_floor_one_level() {
        let p = params(&[("planner.inter_min_log2", 3.0)]);
        assert_eq!(raised(&p).inter_min_log2, 4.0);
    }
}
