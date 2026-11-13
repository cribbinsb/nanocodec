//! Short-window rate bound (`rate.window_seconds`): the bits rate control may
//! still spend on the next frame so that no window of that length (1 s by
//! default) codes more than the target times its length plus a tolerance
//! (`rate.window_tolerance`).
//!
//! The long-term bound is the debit term's (`RateController::debit`); this
//! ledger only keeps short stretches accurate. Rate control plans each inter
//! frame within the room and re-codes a frame that overran it
//! (`RateController::strict_room`). Definitions and measurements:
//! `docs/handoff/rate-control.md`, "Target as a ceiling".
use std::collections::VecDeque;

/// Slack on interval sums, in seconds, so that floating-point sums of frame
/// intervals land on whole windows (30 intervals of 1/30 s are 1 s).
const TIME_EPSILON: f64 = 1e-9;

/// The bits of the frames of the trailing window.
#[derive(Debug, Clone)]
pub(crate) struct WindowLedger {
    /// Window length in seconds.
    seconds: f64,
    /// (interval seconds, bits) of the latest frames, newest last: at most a
    /// window's worth.
    recent: VecDeque<(f64, f64)>,
}

impl WindowLedger {
    pub(crate) fn new(seconds: f64) -> Self {
        Self {
            seconds,
            recent: VecDeque::new(),
        }
    }

    /// A frame interval of `seconds` passed with `bits` counted against the
    /// window (0 for a held or dropped frame).
    pub(crate) fn record(&mut self, seconds: f64, bits: f64) {
        self.recent.push_back((seconds, bits));
        let mut kept = 0.;
        let mut keep = 0;
        for &(interval, _) in self.recent.iter().rev() {
            if kept >= self.seconds - TIME_EPSILON {
                break;
            }
            kept += interval;
            keep += 1;
        }
        let stale = self.recent.len() - keep;
        self.recent.drain(..stale);
    }

    /// Bits added to the latest frame after it was recorded (another spatial
    /// layer's unspent bits it forfeited).
    pub(crate) fn adjust_latest(&mut self, bits: f64) {
        if let Some(latest) = self.recent.back_mut() {
            latest.1 += bits;
        }
    }

    /// Most bits a frame of `seconds` may take so that the window ending
    /// with it holds at most `allowed` bits; negative when the frames
    /// before it already hold more.
    pub(crate) fn room(&self, allowed: f64, seconds: f64) -> f64 {
        let span = self.seconds - seconds;
        let mut covered = 0.;
        let mut trailing = 0.;
        for &(interval, bits) in self.recent.iter().rev() {
            if covered >= span - TIME_EPSILON {
                break;
            }
            covered += interval;
            trailing += bits;
        }
        allowed - trailing
    }

    /// The window length, in seconds.
    pub(crate) fn seconds(&self) -> f64 {
        self.seconds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FPS: f64 = 30.;
    const TARGET: f64 = 300_000.;

    #[test]
    fn room_counts_the_trailing_window_only() {
        let mut ledger = WindowLedger::new(1.);
        ledger.record(1. / FPS, TARGET / 2.);
        for _ in 1..29 {
            ledger.record(1. / FPS, TARGET / FPS);
        }
        // Frame 29 closes the first window: the first frame's half second
        // and 28 uniform frames are in it.
        let room = ledger.room(TARGET, 1. / FPS);
        let expected = TARGET / 2. - 28. * TARGET / FPS;
        assert!((room - expected).abs() < 1e-6, "{room} vs {expected}");
        ledger.record(1. / FPS, TARGET / FPS);
        // Frame 30's window no longer holds the first frame.
        let room = ledger.room(TARGET, 1. / FPS);
        assert!((room - TARGET / FPS).abs() < 1e-6, "{room}");
    }

    #[test]
    fn adjusting_the_latest_frame_moves_the_room() {
        let mut ledger = WindowLedger::new(1.);
        for _ in 0..30 {
            ledger.record(1. / FPS, TARGET / FPS);
        }
        let before = ledger.room(TARGET, 1. / FPS);
        ledger.adjust_latest(1000.);
        assert!((ledger.room(TARGET, 1. / FPS) - (before - 1000.)).abs() < 1e-6);
    }
}
