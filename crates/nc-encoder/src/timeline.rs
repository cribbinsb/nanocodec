//! Frame timestamps: capture times in, per-frame intervals out.
//!
//! Callers may give each frame its capture time in microseconds (camera
//! clocks, variable frame rate, frames dropped before the encoder). The
//! interval since the previous frame is what rate control budgets for the
//! frame. Frames without a timestamp are placed one nominal frame duration
//! after the previous one and budgeted at exactly the nominal rate, so a
//! stream coded without timestamps is unchanged.

use crate::config::FrameRate;

/// Microseconds per second.
pub const MICROS_PER_SECOND: i64 = 1_000_000;
/// Shortest interval a frame is budgeted for, in seconds: timestamps that
/// repeat or run backwards (clock jumps) count as this, not as zero time.
const MIN_INTERVAL_SECONDS: f64 = 0.001;
/// Longest interval a frame is budgeted for, in seconds: after a capture
/// stall the next frame gets at most one second of bitrate, not a burst
/// sized to the whole gap.
const MAX_INTERVAL_SECONDS: f64 = 1.0;

/// When a frame was captured and how long it covers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FrameTime {
    /// Capture time in microseconds.
    pub timestamp_us: i64,
    /// Seconds since the previous frame for rate control; `None` means
    /// the nominal frame duration.
    pub interval: Option<f64>,
}

/// The timestamps of the frames coded so far.
#[derive(Debug, Clone)]
pub(crate) struct Timeline {
    /// One nominal frame duration, rounded to whole microseconds.
    nominal_us: i64,
    last_us: Option<i64>,
}

impl Timeline {
    /// A timeline for a stream of nominal rate `rate` (non-zero).
    pub fn new(rate: FrameRate) -> Self {
        let num = i64::from(rate.num);
        let nominal_us = (MICROS_PER_SECOND * i64::from(rate.den) + num / 2) / num;
        Self {
            nominal_us,
            last_us: None,
        }
    }

    /// Untimed frames from now on are one nominal frame of `rate` apart.
    pub fn set_rate(&mut self, rate: FrameRate) {
        self.nominal_us = Self::new(rate).nominal_us;
    }

    /// Time the next frame: `given` is its capture time, if known.
    pub fn next(&mut self, given: Option<i64>) -> FrameTime {
        let time = match (given, self.last_us) {
            (Some(timestamp_us), Some(last)) => {
                let seconds = (timestamp_us - last) as f64 / MICROS_PER_SECOND as f64;
                FrameTime {
                    timestamp_us,
                    interval: Some(seconds.clamp(MIN_INTERVAL_SECONDS, MAX_INTERVAL_SECONDS)),
                }
            }
            (Some(timestamp_us), None) => FrameTime {
                timestamp_us,
                interval: None,
            },
            (None, last) => FrameTime {
                timestamp_us: last.map_or(0, |l| l + self.nominal_us),
                interval: None,
            },
        };
        self.last_us = Some(time.timestamp_us);
        time
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameTime, Timeline};
    use crate::config::FrameRate;

    #[test]
    fn untimed_frames_advance_nominally() {
        let mut t = Timeline::new(FrameRate {
            num: 30000,
            den: 1001,
        });
        assert_eq!(t.next(None).timestamp_us, 0);
        assert_eq!(
            t.next(None),
            FrameTime {
                timestamp_us: 33367,
                interval: None
            }
        );
    }

    #[test]
    fn timestamps_give_clamped_intervals() {
        let mut t = Timeline::new(FrameRate { num: 30, den: 1 });
        assert_eq!(t.next(Some(1_000_000)).interval, None);
        assert_eq!(t.next(Some(1_066_666)).interval, Some(0.066_666));
        assert_eq!(t.next(Some(1_066_666)).interval, Some(0.001));
        assert_eq!(t.next(Some(9_000_000)).interval, Some(1.0));
        assert_eq!(t.next(None).timestamp_us, 9_033_333);
    }
}
