//! Adaptive coded resolution (`resolution.adaptive`): code at half size
//! per dimension while the half size shows the viewer a better picture than
//! the full size at the stream's rate.
//!
//! Past the codec's largest quantizer (AV1 qindex 255) rate control can
//! only spend fewer bits by raising λ (`rate.lambda_headroom_octaves`):
//! more skipped blocks and dropped residual at a step that cannot get
//! coarser, so each coded block stays dear. There a half-size picture,
//! upscaled by the player, can beat the full size (720p vs 360p at 100
//! kbps: `docs/handoff/res-indep.md`), but only on content that loses
//! little to the halving. The policy switches the coded size, without a key
//! frame, for backends whose references may differ in size from the frame
//! (`CodecBackend::supports_scaled_references`;
//! `Encoder::set_coded_resolution`).
//!
//! Two conditions, each with hysteresis:
//! - *Rate*: a moving average of log2(planned step / largest step) of rate
//!   control's inter level. Down needs the full-size level `down_step_ratio`
//!   past the ceiling; the half size goes back up once the level the full
//!   size would need (the half-size level times `half_step_gain`) is below
//!   `up_step_ratio` of it.
//! - *Evidence* (`resolution_evidence.rs`): while the rate condition holds
//!   at full size, and all the time at half size, every
//!   [`EVIDENCE_PERIOD`]th frame measures the luma error the viewer sees
//!   (`coded`) and the half size's floor (the source halved and shown
//!   upscaled). Down also needs at least [`MIN_EVIDENCE`] samples with the
//!   averaged floor at most `down_floor_ratio` of the full size's coded
//!   error; a half size whose averaged floor reaches `up_floor_ratio` of its
//!   own displayed error goes back up whatever the rate. The rate level
//!   alone used to decide: a coding tool that spends a few more bits per
//!   level (`refs.second`) tipped `pool_chad_4_109` and
//!   `pool_pan_jaad_video_0194` (150 kbps 720p) into a half size 2.5 dB
//!   worse (`docs/handoff/adaptive-resolution.md`, "Evidence").
//!
//! Each size is kept at least `dwell_seconds`; the first decision waits
//! only for one averaging window. Codec-neutral: the thresholds are step
//! and error ratios.

use crate::params::ResolutionParams;
use crate::resolution_evidence::Distortions;

/// Inter frames between two evidence measurements: each costs a display
/// upscale and two error sums over the stream-size luma (a half-size
/// downscale too at full size), a few ms at 720p on one core, so it runs
/// on about four frames a second at 30 fps, and only while the rate
/// condition holds or the stream is at half size.
pub(crate) const EVIDENCE_PERIOD: u32 = 8;
/// Evidence samples before the evidence may switch the size: about a
/// second at 30 fps.
const MIN_EVIDENCE: u32 = 4;
/// Weight of each new sample in the evidence's moving averages.
const EVIDENCE_WEIGHT: f64 = 0.25;

/// The two coded sizes the policy chooses between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodedScale {
    /// The stream size.
    Full,
    /// Half the stream size per dimension ([`half_size`]).
    Half,
}

/// The half-size picture of a `width` x `height` stream: rounded up (a
/// reference may be at most twice the frame, AV1 §7.11.3.3) to even sizes
/// (4:2:0).
#[must_use]
pub(crate) fn half_size(width: usize, height: usize) -> (usize, usize) {
    let half = |n: usize| n.div_ceil(2).next_multiple_of(2);
    (half(width), half(height))
}

/// The switching policy's state.
#[derive(Debug, Clone)]
pub(crate) struct ResolutionPolicy {
    /// log2 of `resolution.down_step_ratio`.
    down_log2: f64,
    /// log2 of `resolution.up_step_ratio`.
    up_log2: f64,
    /// log2 of `resolution.half_step_gain`.
    gain_log2: f64,
    /// Weight of each new frame in the moving average.
    weight: f64,
    /// Frames to keep a size before switching again.
    dwell_frames: u32,
    /// Frames the current size must be observed before a switch: the
    /// averaging window at the start, then `dwell_frames`.
    hold_frames: u32,
    scale: CodedScale,
    /// Inter frames observed at the current size.
    frames: u32,
    /// Moving average of log2(planned step / largest step) at the
    /// current size; `None` before the first observation.
    average: Option<f64>,
    /// `resolution.down_floor_ratio`.
    down_floor_ratio: f64,
    /// `resolution.up_floor_ratio`.
    up_floor_ratio: f64,
    /// Moving averages of the measured distortions at the current size,
    /// since the evidence last started.
    evidence: Option<Distortions>,
    /// Samples in `evidence`.
    samples: u32,
}

impl ResolutionPolicy {
    /// The policy `params` ask for at `frame_rate` frames per second;
    /// `None` when `resolution.adaptive` is off.
    #[must_use]
    pub(crate) fn new(params: &ResolutionParams, frame_rate: f64) -> Option<Self> {
        let window = (f64::from(params.window_seconds) * frame_rate).max(1.0);
        (params.adaptive >= 0.5).then(|| Self {
            down_log2: f64::from(params.down_step_ratio).log2(),
            up_log2: f64::from(params.up_step_ratio).log2(),
            gain_log2: f64::from(params.half_step_gain).log2(),
            weight: 1.0 / window,
            dwell_frames: (f64::from(params.dwell_seconds) * frame_rate).round() as u32,
            hold_frames: window.round() as u32,
            scale: CodedScale::Full,
            frames: 0,
            average: None,
            down_floor_ratio: f64::from(params.down_floor_ratio),
            up_floor_ratio: f64::from(params.up_floor_ratio),
            evidence: None,
            samples: 0,
        })
    }

    /// Whether this inter frame's distortions should be measured
    /// ([`Self::add_evidence`]): every [`EVIDENCE_PERIOD`]th frame at half
    /// size, and at full size while the rate condition for going down holds
    /// (measuring nothing on streams that never get there).
    pub(crate) fn wants_evidence(&self) -> bool {
        let due = self.frames.is_multiple_of(EVIDENCE_PERIOD);
        match self.scale {
            CodedScale::Half => due,
            CodedScale::Full => due && self.rate_says_down(),
        }
    }

    /// A frame's measured distortions at the current size.
    pub(crate) fn add_evidence(&mut self, sample: Distortions) {
        let blend = |old: f64, new: f64| old + EVIDENCE_WEIGHT * (new - old);
        self.evidence = Some(self.evidence.map_or(sample, |e| Distortions {
            floor: blend(e.floor, sample.floor),
            coded: blend(e.coded, sample.coded),
        }));
        self.samples += 1;
    }

    fn rate_says_down(&self) -> bool {
        self.average.is_some_and(|a| a >= self.down_log2)
    }

    /// The averaged evidence, once there are enough samples.
    fn settled_evidence(&self) -> Option<Distortions> {
        self.evidence.filter(|_| self.samples >= MIN_EVIDENCE)
    }

    /// After an inter frame whose rate-control level plans a quantizer step
    /// `step_ratio` times the codec's largest: the size the next frame
    /// should be coded at, when it changes.
    pub(crate) fn observe(&mut self, step_ratio: f64) -> Option<CodedScale> {
        let sample = step_ratio.max(f64::MIN_POSITIVE).log2();
        let average = self
            .average
            .map_or(sample, |a| a + self.weight * (sample - a));
        self.average = Some(average);
        self.frames += 1;
        if self.scale == CodedScale::Full && !self.rate_says_down() {
            // Evidence gathered while the rate was starved says nothing
            // about the next time it is.
            self.evidence = None;
            self.samples = 0;
        }
        if self.frames < self.hold_frames {
            return None;
        }
        let evidence = self.settled_evidence();
        let next = match self.scale {
            CodedScale::Full
                if self.rate_says_down()
                    && evidence.is_some_and(|e| e.floor <= self.down_floor_ratio * e.coded) =>
            {
                CodedScale::Half
            }
            CodedScale::Half
                if average + self.gain_log2 <= self.up_log2
                    || evidence.is_some_and(|e| e.floor >= self.up_floor_ratio * e.coded) =>
            {
                CodedScale::Full
            }
            _ => return None,
        };
        // The level is expected to move by the gain at the new size.
        let shift = match next {
            CodedScale::Half => -self.gain_log2,
            CodedScale::Full => self.gain_log2,
        };
        self.average = Some(average + shift);
        self.scale = next;
        self.frames = 0;
        self.evidence = None;
        self.samples = 0;
        self.hold_frames = self.dwell_frames;
        Some(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ParamOverrides;

    fn policy(dwell_seconds: f32) -> ResolutionPolicy {
        let mut params = ResolutionParams::from_overrides(&ParamOverrides::default());
        params.adaptive = 1.0;
        params.dwell_seconds = dwell_seconds;
        ResolutionPolicy::new(&params, 30.0).expect("enabled")
    }

    #[test]
    fn half_sizes_round_up_to_even() {
        assert_eq!(half_size(1280, 720), (640, 360));
        assert_eq!(half_size(1920, 1080), (960, 540));
        assert_eq!(half_size(854, 480), (428, 240));
        assert_eq!(half_size(1278, 718), (640, 360));
    }

    #[test]
    fn off_by_default() {
        let params = ResolutionParams::from_overrides(&ParamOverrides::default());
        assert!(ResolutionPolicy::new(&params, 30.0).is_none());
    }

    /// The half size's floor and the full size's error on content where
    /// the half size wins (measured on Crowd Run at 100 kbps, 720p: floor
    /// about 120, displayed error about 630) and where it loses
    /// (`pool_chad_4_109` at 150 kbps: floor about 310, error about 235).
    const HALF_WINS: Distortions = Distortions {
        floor: 120.0,
        coded: 630.0,
    };
    const HALF_LOSES: Distortions = Distortions {
        floor: 310.0,
        coded: 235.0,
    };

    /// One inter frame at `step_ratio`, measured as `evidence` when the
    /// policy asks.
    fn frame(
        p: &mut ResolutionPolicy,
        step_ratio: f64,
        evidence: Distortions,
    ) -> Option<CodedScale> {
        if p.wants_evidence() {
            p.add_evidence(evidence);
        }
        p.observe(step_ratio)
    }

    #[test]
    fn switches_down_past_the_ceiling_after_the_dwell_and_back_up() {
        let mut p = policy(1.0);
        // Below the ceiling: never switches, and measures nothing.
        assert!((0..100).all(|_| !p.wants_evidence() && p.observe(0.8).is_none()));
        // Far past it, on content the half size wins: switches once the
        // average crosses and the evidence has settled, not on one frame.
        assert_eq!(frame(&mut p, 4.0, HALF_WINS), None);
        let down = (0..100).position(|_| frame(&mut p, 4.0, HALF_WINS).is_some());
        assert!(down.is_some());
        // At half size the level falls; it must stay for the dwell time.
        let up = (0..200).position(|_| frame(&mut p, 0.2, HALF_WINS) == Some(CodedScale::Full));
        assert!(up.is_some_and(|frames| frames + 1 >= 30));
    }

    #[test]
    fn stays_at_full_size_where_the_half_size_loses() {
        let mut p = policy(1.0);
        assert!((0..600).all(|_| frame(&mut p, 8.0, HALF_LOSES).is_none()));
    }

    #[test]
    fn half_size_goes_back_up_when_it_stops_winning_whatever_the_rate() {
        let mut p = policy(1.0);
        while frame(&mut p, 8.0, HALF_WINS).is_none() {}
        // Still starved, but the half size now shows little more than its
        // floor: back to full size after the dwell.
        let shown_near_floor = Distortions {
            floor: 120.0,
            coded: 200.0,
        };
        let up = (0..300).position(|_| frame(&mut p, 8.0, shown_near_floor).is_some());
        assert!(up.is_some_and(|frames| frames + 1 >= 30));
    }

    #[test]
    fn hysteresis_holds_between_the_thresholds() {
        let mut p = policy(0.0);
        let params = ResolutionParams::from_overrides(&ParamOverrides::default());
        // Switched down, a half-size level that maps just above the up
        // threshold keeps the half size, as does evidence between the
        // down and up ratios.
        while frame(&mut p, 8.0, HALF_WINS).is_none() {}
        let hold = f64::from(params.up_step_ratio) / f64::from(params.half_step_gain) * 1.05;
        let between = Distortions {
            floor: 100.0,
            coded: 100.0 / (0.5 * f64::from(params.down_floor_ratio + params.up_floor_ratio)),
        };
        assert!((0..300).all(|_| frame(&mut p, hold, between).is_none()));
    }
}
