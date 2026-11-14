//! The key-period ledger (`rate.period_ceiling`): the target as a ceiling
//! over each key period, enforced on frame sizes.
//!
//! Rate control plans budgets around a virtual buffer and corrects debt
//! proportionally, with a floor (a quarter of the budget). On still scenes
//! that is not enough: refinement bursts overspend whatever budget they get
//! (their size is set by the quality the picture already holds, not by the
//! plan), so a stream can stay a buffer in debt and end a 20 s window over
//! its target (AV1 Default, 600 f, key every 10 s: chirla cam 7 at 93-187
//! kbps 1.015-1.037 of target;
//! `docs/analysis/persistent-region-coding-2026-11-14.md`).
//!
//! The ledger counts what the stream has spent against the target line
//! (bits over `target × elapsed time`) and gives every inter frame at most
//! what the rest of the ledger period (the key period's duration, or
//! `rate.key_repay_seconds` without periodic keys, in stream time) can
//! still repay: the period's remaining
//! allowance less a minimal spend reserved for each later frame. A frame
//! over it is re-coded coarser through the existing re-code path, so the
//! debt is repaid by the next scheduled key whatever the bursts would
//! cost. Keys are not bounded here: they start a period and are repaid
//! inside it.

/// Uniform budgets kept back for each later frame of the period: what an
/// inter frame of a still scene costs at the inter level at most (0.1-0.3
/// U on the audit's still points), so the frames after a capped one are not
/// starved to repay it.
const RESERVE_PER_FRAME: f64 = 0.25;
/// Least room the ledger leaves any inter frame, in uniform budgets: a frame
/// that small cannot be coded coarser to any use, and refusing it would
/// only add re-codes; what it costs beyond the allowance is a tiny miss.
const MIN_ROOM: f64 = 0.25;

/// Share of unchanged blocks above which a due key keeps strict-buffer
/// room, growing to [`STILL_KEY_GROWTH`] times the last key at 1 (a fixed
/// camera): a crowd scene keeping it re-coded the frames before every key
/// (AV1 crowd mot20 85 kbps: 16 re-codes, frames before the key -3 dB).
const STILL_FROM: f64 = 0.8;
/// How much more than the last key a key of a wholly still scene needs: it
/// codes the quality the inter frames refined since (chirla 93 kbps: 31
/// budgets after a 17-budget key, 1.8x).
const STILL_KEY_GROWTH: f64 = 2.;

/// The ledger of one stream.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PeriodLedger {
    /// Frames from the frame being planned to the debt's deadline,
    /// counting it (`None`: not planned yet).
    frames_left: Option<u64>,
    /// Coded frames to the next scheduled key, when it falls inside the
    /// period or starts the next one (`None`: no key by the deadline).
    key_due: Option<u64>,
    /// Whether that key falls before the deadline, so that its bits count
    /// against this period's allowance; a key that starts the next period
    /// is repaid there.
    key_in_period: bool,
    /// Bits of the last key: what the next one is expected to need.
    last_key_bits: f64,
}

impl PeriodLedger {
    /// Plan the frame at stream time `elapsed` (seconds) of a stream at
    /// `fps` whose ledger periods last `period` seconds: the debt is due at
    /// the next multiple of `period` of stream time. Stream time, not the
    /// key schedule, so that held frames (which do not count towards the
    /// key period) cannot carry debt past a boundary, and every window of
    /// whole periods from the start ends repaid.
    pub fn plan(&mut self, elapsed: f64, period: f64, fps: f64, to_key: Option<u64>) {
        let into = elapsed.rem_euclid(period);
        let left = ((period - into) * fps).round().max(1.) as u64;
        self.frames_left = Some(left);
        self.key_due = to_key.filter(|&frames| frames <= left && self.last_key_bits > 0.);
        self.key_in_period = self.key_due.is_some_and(|frames| frames < left);
    }

    /// A key of `bits` was coded.
    pub fn observe_key(&mut self, bits: f64) {
        self.last_key_bits = bits;
    }

    /// Most bits the inter frame being planned may take: with a uniform
    /// frame budget `uniform` (target over frame rate) and `debt` bits spent
    /// above the target line so far, the period's remaining allowance less
    /// the reserve of its later frames, at least [`MIN_ROOM`] budgets.
    ///
    /// A key due inside the period is reserved the last key's size from the
    /// period's allowance, and on still content (`still_share` of unchanged
    /// blocks) room in the strict buffer (`strict`: what that buffer could
    /// take now, unbounded when off): a burst just before a key would
    /// otherwise starve it, and the frames after it inherit the starved key
    /// (AV1 chirla 93 kbps: a 26-budget frame four frames before the key
    /// left it 1.0 buffer and qindex 264).
    pub fn room(&self, uniform: f64, debt: f64, (strict, still_share): (f64, f64)) -> f64 {
        let Some(frames) = self.frames_left else {
            return f64::INFINITY;
        };
        let frames = frames as f64;
        // A periodic key starts the next period (the periods are the key
        // period), and the debt it leaves is repaid there. Charging it to
        // the period before as well squeezed the frames before every key
        // (AV1 Default, keys every 2 s: periodic-key pulse +2.25 dB, crowd
        // +3.9, lowlight +4.5); only a key inside the period (requested
        // mid-period) is paid for in it.
        let still = ((still_share - STILL_FROM) / (1. - STILL_FROM)).clamp(0., 1.);
        let key = if self.key_in_period {
            self.last_key_bits - uniform
        } else if self.key_due.is_some() {
            (self.last_key_bits - uniform) * still
        } else {
            0.
        };
        let allowance = frames * uniform - debt - (frames - 1.) * RESERVE_PER_FRAME * uniform - key;
        let before_key = self.key_due.map_or(f64::INFINITY, |to_key| {
            // The strict buffer drains a budget per frame until the key. A
            // still scene's key holds refined quality and costs more than
            // the last one, so it keeps up to [`STILL_KEY_GROWTH`] times the
            // last key on a fixed camera ([`STILL_FROM`]). The room scales
            // with the measured key, not the buffer: a whole buffer kept
            // for every fixed-camera key starved the frames before the keys
            // of crowd scenes, whose moving part keeps their keys small
            // (AV1 Default 600 f: crowd worst-1% +101%, pan +42%, re-codes
            // 6.0 -> 22.8 per 1000 on static and crowd).
            let need = self.last_key_bits * STILL_KEY_GROWTH * still;
            strict + (to_key as f64 - 1.) * uniform - need
        });
        allowance.min(before_key).max(MIN_ROOM * uniform)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_last_frames_of_a_period_repay_the_debt() {
        let mut ledger = PeriodLedger::default();
        ledger.plan(299. / 30., 10., 30., None);
        // One frame left, 0.5 budgets of debt: the frame may take half a budget.
        assert_eq!(ledger.room(1000., 500., (f64::INFINITY, 1.)), 500.);
        // Deep debt: the floor.
        assert_eq!(
            ledger.room(1000., 5000., (f64::INFINITY, 1.)),
            MIN_ROOM * 1000.
        );
    }

    #[test]
    fn early_frames_keep_room_for_a_burst() {
        let mut ledger = PeriodLedger::default();
        ledger.plan(1. / 30., 10., 30., None);
        // 299 frames left, one buffer (30 budgets) of debt.
        let room = ledger.room(1000., 30_000., (f64::INFINITY, 1.));
        assert!(room > 100. * 1000., "{room}");
    }

    #[test]
    fn periods_follow_stream_time() {
        let mut ledger = PeriodLedger::default();
        // 9.99 s into a 10 s period at 30 fps: the last frame.
        ledger.plan(9.99, 10., 30., None);
        assert_eq!(ledger.room(1000., 500., (f64::INFINITY, 1.)), 500.);
        ledger.plan(10.0, 10., 30., None);
        assert!(ledger.room(1000., 500., (f64::INFINITY, 1.)) > 200. * 1000.);
    }

    #[test]
    fn a_key_starting_the_next_period_is_not_charged_to_this_one() {
        let mut ledger = PeriodLedger::default();
        ledger.observe_key(20_000.);
        // The last frame of the period, the key right after it: one budget
        // of room with no debt, whatever the key costs.
        ledger.plan(299. / 30., 10., 30., Some(1));
        assert_eq!(ledger.room(1000., 0., (f64::INFINITY, 0.)), 1000.);
        // A requested key two frames into a period with 100 frames left
        // counts: 100 - 99 x 0.25 - 19 = 56.25 budgets.
        ledger.plan(10. - 100. / 30., 10., 30., Some(2));
        assert_eq!(ledger.room(1000., 0., (f64::INFINITY, 0.)), 56_250.);
    }

    #[test]
    fn a_due_key_keeps_its_room_in_the_strict_buffer() {
        let mut ledger = PeriodLedger::default();
        ledger.observe_key(20_000.);
        // Four frames to the key, 50 kbit of strict room, a fixed camera:
        // 50 + 3 - 2 x 20 = 13.
        ledger.plan(0.5, 10., 30., Some(4));
        assert_eq!(ledger.room(1000., 0., (50_000., 1.)), 13_000.);
        // Moving content in a fixed frame (90% unchanged) keeps half that.
        assert_eq!(ledger.room(1000., 0., (50_000., 0.9)), 33_000.);
    }
}
