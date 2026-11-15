//! The hold of still blocks before a scheduled key (`rate.prekey_quiet`).
//!
//! In the last `rate.prekey_quiet` seconds before a scheduled periodic key,
//! still blocks keep the quality they hold (`ReuseTerm::hold_still`):
//! whatever a frame refined there the key codes again. Planned refinements
//! (`rate.refine`) and the hold on every frame (`rate.hold_still`,
//! `rate.hold_recover`) that once lived here failed their screens and were
//! removed (2026-11-15, docs/handoff/tuning.md "Removed levers").

use crate::params::RateParams;
use crate::reuse::{ReuseFrame, ReuseTerm};

/// Fewest frames a cell's source must have been still to be held: a shorter
/// run predicts too little reuse to price it finer than the frame (its
/// price, `λ / (1 + R)`, is within 6 QP of it).
pub(crate) const HOLD_MIN_STILL_RUN: u8 = 4;

/// The hold of still blocks before a scheduled key.
#[derive(Debug, Clone)]
pub(crate) struct StillPlan {
    prekey_quiet_frames: u64,
    /// Whether the frame being planned holds its still blocks: its level is
    /// planned, and its model learnt, without its block offsets (the held
    /// blocks are copies).
    pub without_offsets: bool,
}

impl StillPlan {
    /// The plan of `params` at `fps`; `available`: the stream is flat,
    /// single-layer and codes block quantizers with the reuse term on.
    pub fn new(params: &RateParams, fps: f64, available: bool) -> Self {
        Self {
            prekey_quiet_frames: if available {
                (f64::from(params.prekey_quiet) * fps).round() as u64
            } else {
                0
            },
            without_offsets: false,
        }
    }

    /// Plan inter frame `frame` (after `ReuseTerm::plan`; `flat`: it
    /// predicts from the frame before in a flat structure): the hold of its
    /// still blocks before a scheduled key (the key codes every block again).
    pub fn plan(&mut self, term: &mut ReuseTerm, frame: &ReuseFrame<'_>, flat: bool) {
        self.without_offsets = false;
        if frame.key || !flat {
            return;
        }
        let quiet = frame
            .horizon
            .is_some_and(|to_key| to_key <= self.prekey_quiet_frames);
        if quiet {
            term.hold_still(frame, HOLD_MIN_STILL_RUN);
            self.without_offsets = true;
        }
    }
}
