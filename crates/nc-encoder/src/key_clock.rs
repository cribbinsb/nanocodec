//! The periodic key-frame schedule, in time.
//!
//! `intra_period` frames apart at the nominal rate: a key every `intra_period`
//! nominal frame intervals of capture time. Untimed frames are one nominal
//! interval apart ([`crate::timeline::Timeline`]), so their keys fall every
//! `intra_period` coded frames exactly as a frame-count schedule puts them.
//! Timestamped frames from a live source may skip captures (a busy sender
//! drops frames before they reach the encoder): the first coded frame at or
//! past each period boundary is the key, so keys stay about `intra_period`
//! intervals of real time apart instead of `intra_period` coded frames, which
//! with half the frames skipped would be twice as long.
//!
//! A frame counts as at a boundary from half a nominal interval before it, so
//! jitter around the nominal spacing never moves a key by a frame.

/// Where frames stand against the schedule.
#[derive(Debug, Clone)]
pub(crate) struct KeyClock {
    /// One nominal frame interval, in microseconds.
    nominal_us: i64,
    /// The first coded frame's time: periods count from it.
    origin_us: Option<i64>,
    /// The period the previous coded frame fell in.
    last_period: u64,
}

impl KeyClock {
    pub fn new(nominal_us: i64) -> Self {
        Self {
            nominal_us,
            origin_us: None,
            last_period: 0,
        }
    }

    /// Untimed frames from now on are `nominal_us` apart (a frame rate change).
    pub fn set_nominal(&mut self, nominal_us: i64) {
        self.nominal_us = nominal_us;
    }

    /// The period (counted from 0) a frame at `time_us` falls in, for a key
    /// period of `period_frames` nominal frames.
    fn period_of(&self, time_us: i64, period_frames: u64) -> u64 {
        let origin = self.origin_us.unwrap_or(time_us);
        let span = self
            .nominal_us
            .saturating_mul(i64::try_from(period_frames).unwrap_or(i64::MAX))
            .max(1);
        u64::try_from((time_us - origin + self.nominal_us / 2).max(0) / span).unwrap_or(0)
    }

    /// Whether the next coded frame, at `time_us`, is a scheduled key: the
    /// first frame, or the first in a later period.
    pub fn is_due(&self, time_us: i64, period_frames: u64) -> bool {
        self.origin_us.is_none() || self.period_of(time_us, period_frames) > self.last_period
    }

    /// Nominal frames from the frame at `time_us` to the next scheduled key.
    pub fn frames_to_next(&self, time_us: i64, period_frames: u64) -> u64 {
        let origin = self.origin_us.unwrap_or(time_us);
        let elapsed = u64::try_from(
            ((time_us - origin) + self.nominal_us / 2).max(0) / self.nominal_us.max(1),
        )
        .unwrap_or(0);
        period_frames - elapsed % period_frames
    }

    /// A frame at `time_us` was coded.
    pub fn coded(&mut self, time_us: i64, period_frames: u64) {
        self.last_period = self.period_of(time_us, period_frames);
        self.origin_us.get_or_insert(time_us);
    }
}

#[cfg(test)]
mod tests {
    use super::KeyClock;

    const NOMINAL_US: i64 = 33_333;
    const PERIOD: u64 = 30;

    /// Keys of frames at `times`, coding each.
    fn keys(times: impl IntoIterator<Item = i64>) -> Vec<usize> {
        let mut clock = KeyClock::new(NOMINAL_US);
        times
            .into_iter()
            .enumerate()
            .filter_map(|(i, t)| {
                let due = clock.is_due(t, PERIOD);
                clock.coded(t, PERIOD);
                due.then_some(i)
            })
            .collect()
    }

    #[test]
    fn nominal_spacing_keys_every_period_frames() {
        assert_eq!(keys((0..100).map(|i| i * NOMINAL_US)), [0, 30, 60, 90]);
    }

    #[test]
    fn skipped_frames_keep_keys_on_time() {
        // Every other capture skipped: keys every 15 coded frames, about a second apart.
        assert_eq!(
            keys((0..100).step_by(2).map(|i| i * NOMINAL_US)),
            [0, 15, 30, 45]
        );
    }

    #[test]
    fn jitter_does_not_move_a_key() {
        let jittered = (0..61).map(|i| i * NOMINAL_US + if i % 2 == 0 { 3000 } else { -3000 });
        assert_eq!(keys(jittered), [0, 30, 60]);
    }

    #[test]
    fn a_long_gap_gives_one_key() {
        assert_eq!(
            keys([0, NOMINAL_US, 200 * NOMINAL_US, 201 * NOMINAL_US]),
            [0, 2]
        );
    }

    #[test]
    fn frames_to_next_counts_nominal_frames() {
        let mut clock = KeyClock::new(NOMINAL_US);
        clock.coded(0, PERIOD);
        assert_eq!(clock.frames_to_next(0, PERIOD), 30);
        assert_eq!(clock.frames_to_next(10 * NOMINAL_US, PERIOD), 20);
    }
}
