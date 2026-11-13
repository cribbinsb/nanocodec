//! Adaptive coded resolution (`resolution.adaptive`): code at half size
//! per dimension while the bitrate is too low for the full size.
//!
//! Past the codec's largest quantizer (AV1 qindex 255) rate control can
//! only spend fewer bits by raising λ (`rate.lambda_headroom_octaves`):
//! more skipped blocks and dropped residual at a step that cannot get
//! coarser, so each coded block stays dear. There a half-size picture,
//! upscaled by the player, beats the full size (720p vs 360p at 100 kbps:
//! `docs/handoff/res-indep.md`). This policy watches the planned quantizer
//! step of rate control's inter level relative to the largest step, and
//! switches the coded size, without a key frame, for backends whose
//! references may differ in size from the frame
//! (`CodecBackend::supports_scaled_references`; `Encoder::set_coded_resolution`).
//!
//! Decisions use a moving average of the log step ratio, so single frames
//! do not switch, and two thresholds with a gap between them (hysteresis):
//! down when the full-size level sits `down_step_ratio` past the ceiling,
//! up when the level the full size would need (the half-size level times
//! `half_step_gain`, the step ratio between the two sizes at the same
//! rate) is back below `up_step_ratio` of it. Each size is kept at least
//! `dwell_seconds`; the first decision waits only for one averaging window,
//! so a starved stream leaves the full size within its first second.
//! Codec-neutral: the thresholds are step ratios.

use crate::params::ResolutionParams;

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
        })
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
        if self.frames < self.hold_frames {
            return None;
        }
        let next = match self.scale {
            CodedScale::Full if average >= self.down_log2 => CodedScale::Half,
            CodedScale::Half if average + self.gain_log2 <= self.up_log2 => CodedScale::Full,
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

    #[test]
    fn switches_down_past_the_ceiling_after_the_dwell_and_back_up() {
        let mut p = policy(1.0);
        // Below the ceiling: never switches.
        assert!((0..100).all(|_| p.observe(0.8).is_none()));
        // Far past it: switches once the average crosses, not on one frame.
        assert_eq!(p.observe(4.0), None);
        let down = (0..100).position(|_| p.observe(4.0).is_some());
        assert!(down.is_some());
        // At half size the level falls; it must stay for the dwell time.
        let up = (0..200).position(|_| p.observe(0.2) == Some(CodedScale::Full));
        assert!(up.is_some_and(|frames| frames + 1 >= 30));
    }

    #[test]
    fn hysteresis_holds_between_the_thresholds() {
        let mut p = policy(0.0);
        let params = ResolutionParams::from_overrides(&ParamOverrides::default());
        // Switched down, a half-size level that maps just above the up
        // threshold keeps the half size.
        while p.observe(8.0).is_none() {}
        let hold = f64::from(params.up_step_ratio) / f64::from(params.half_step_gain) * 1.05;
        assert!((0..300).all(|_| p.observe(hold).is_none()));
    }
}
