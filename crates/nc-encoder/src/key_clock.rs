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
//!
//! A requested or scene-cut key restarts the schedule: the next scheduled key
//! comes `intra_period` after it, not at the old boundary (x264 `keyint`,
//! SVT-AV1 and libaom `kf_max_dist` also count from the last key). Why: a
//! scheduled key a few frames after a forced one refreshes a picture just
//! refreshed, and its bits buy nothing. A scheduled key keeps the time grid,
//! so late captures do not drift it. This is the one source of the frames to
//! the next key: the encoder's frame types, its run-up to a key and rate
//! control's key planning all read it (`Encoder::frames_to_next_key`).

/// Where frames stand against the schedule.
#[derive(Debug, Clone)]
pub(crate) struct KeyClock {
    /// One nominal frame interval, in microseconds.
    nominal_us: i64,
    /// The first coded frame's time: periods count from it.
    origin_us: Option<i64>,
    /// The period the previous coded frame fell in.
    last_period: u64,
    /// The frame index of the last key: untimed frames count periods from it.
    origin_index: u64,
}

impl KeyClock {
    pub fn new(nominal_us: i64) -> Self {
        Self {
            nominal_us,
            origin_us: None,
            last_period: 0,
            origin_index: 0,
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

    /// Whether the next coded frame, `index` (at `time_us` when timed), is a
    /// scheduled key: the first frame, or the first in a later period.
    pub fn is_due(&self, index: u64, time_us: Option<i64>, period_frames: u64) -> bool {
        match time_us {
            Some(time_us) => {
                self.origin_us.is_none()
                    || self.period_of(time_us, period_frames) > self.last_period
            }
            None => (index - self.origin_index).is_multiple_of(period_frames),
        }
    }

    /// Nominal frames from frame `index` (at `time_us` when timed) to the
    /// next scheduled key.
    pub fn frames_to_next(&self, index: u64, time_us: Option<i64>, period_frames: u64) -> u64 {
        let Some(time_us) = time_us else {
            return period_frames - (index - self.origin_index) % period_frames;
        };
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

    /// Frame `index` (at `time_us` when timed) was coded as a key that was
    /// not scheduled (requested, a scene cut): the schedule restarts from it.
    pub fn restart(&mut self, index: u64, time_us: Option<i64>) {
        self.origin_index = index;
        if let Some(time_us) = time_us {
            self.origin_us = Some(time_us);
            self.last_period = 0;
        }
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
                let due = clock.is_due(i as u64, Some(t), PERIOD);
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
        assert_eq!(clock.frames_to_next(0, Some(0), PERIOD), 30);
        assert_eq!(clock.frames_to_next(10, Some(10 * NOMINAL_US), PERIOD), 20);
    }

    #[test]
    fn a_forced_key_restarts_the_schedule() {
        let clock = KeyClock::new(NOMINAL_US);
        assert!(clock.is_due(30, None, PERIOD));
        let mut forced = clock.clone();
        forced.restart(40, None);
        assert!(!forced.is_due(60, None, PERIOD));
        assert!(forced.is_due(70, None, PERIOD));
        assert_eq!(forced.frames_to_next(45, None, PERIOD), 25);
        // Timed: a key requested at frame 40 moves the next one to frame 70.
        let mut timed = KeyClock::new(NOMINAL_US);
        for i in 0..=40 {
            timed.coded(i * NOMINAL_US, PERIOD);
        }
        timed.restart(40, Some(40 * NOMINAL_US));
        assert!(!timed.is_due(60, Some(60 * NOMINAL_US), PERIOD));
        assert!(timed.is_due(70, Some(70 * NOMINAL_US), PERIOD));
        assert_eq!(timed.frames_to_next(45, Some(45 * NOMINAL_US), PERIOD), 25);
    }
}
