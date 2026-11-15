//! Rate control's part of still-scene coding (`RateController::still`):
//! the quiet run-up to a scheduled key, the held key of a still scene and
//! the key-period ceiling. The refinement loan that once lived here
//! (`rate.refine`) failed its screens and was removed (2026-11-15,
//! docs/handoff/tuning.md "Removed levers").

use crate::params::RateParams;
use crate::period_ledger::PeriodLedger;

/// Rate control's part of still-scene coding: the quiet run-up to a key
/// (`rate.prekey_quiet`), the held key (`rate.key_hold_still`) and the key-period ceiling that repays
/// the bursts of still scenes (`rate.period_ceiling`, which bounds every
/// inter frame in debt, still or not).
#[derive(Debug, Clone)]
pub(crate) struct StillRate {
    /// `rate.prekey_quiet` in frames (0: off).
    prekey_quiet_frames: f64,
    /// `rate.key_hold_still` (buffers; 0: off).
    pub key_hold_buffers: f64,
    /// The key-period ceiling (`rate.period_ceiling`; `None`: off).
    ledger: Option<PeriodLedger>,
}

impl StillRate {
    pub fn new(params: &RateParams, fps: f64) -> Self {
        Self {
            prekey_quiet_frames: f64::from(params.prekey_quiet) * fps,
            key_hold_buffers: f64::from(params.key_hold_still),
            ledger: (params.period_ceiling >= 0.5).then(PeriodLedger::default),
        }
    }

    /// Plan the key-period ledger for the frame at stream time `elapsed`
    /// (`PeriodLedger::plan`).
    pub fn plan_period(&mut self, elapsed: f64, period: f64, fps: f64, to_key: Option<u64>) {
        if let Some(ledger) = self.ledger.as_mut() {
            ledger.plan(elapsed, period, fps, to_key);
        }
    }

    /// A key of `bits` was coded (the ledger reserves its size for the next).
    pub fn observe_key(&mut self, bits: f64) {
        if let Some(ledger) = self.ledger.as_mut() {
            ledger.observe_key(bits);
        }
    }

    /// The ledger's room for an inter frame (`PeriodLedger::room`;
    /// unbounded when off).
    pub fn ledger_room(&self, uniform: f64, debt: f64, strict: (f64, f64)) -> f64 {
        self.ledger
            .as_ref()
            .map_or(f64::INFINITY, |ledger| ledger.room(uniform, debt, strict))
    }

    /// A cadence `offset` with no base boost `to_key` frames before a
    /// scheduled key, within `rate.prekey_quiet`.
    pub fn quieted(&self, to_key: Option<u64>, offset: f64) -> f64 {
        match to_key {
            Some(frames) if offset < 0. && (frames as f64) <= self.prekey_quiet_frames => 0.,
            _ => offset,
        }
    }
}
