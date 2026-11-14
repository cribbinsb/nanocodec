//! In-frame rate control: steering a frame's size while it is coded.
//!
//! Frame-level rate control (`rate_control.rs`) fixes a frame's quantizer
//! before the frame and learns from its size afterwards; a frame its model
//! mispredicts (static refinement bursts: 13-22x in AV1,
//! `docs/results/rate-control-audit-2026-11-13.md`) is only caught after the
//! fact, by coding it again (`rate.recode_overflow`). This module watches
//! the bits a frame has spent after each superblock row, projects the rest
//! from the pre-analysis' bit weight still to code, and coarsens the
//! remaining rows when the projection passes a limit: the room a re-code
//! would enforce (`inframe.room`) and, for inter frames, a multiple of the
//! uncorrected frame budget (`inframe.budgets`). The idea is openh264's
//! group-of-macroblock control (`ratectl.cpp`, `RcCalculateGomQp`: ±1-2 QP
//! per group of 2-4 macroblock rows, within the frame QP ± 3; BSD) and HM /
//! kvazaar's CTU feedback (JCTVC-K0103), reimplemented on the engine's own
//! block-QP plumbing; x264's row-level VBV is the same idea (known from its
//! documentation only). Unlike openh264 it steers only toward a limit, not
//! toward the frame's budget, so frames the plan predicted well are coded
//! exactly as planned.
//!
//! What it never does: refine a row (the target is a ceiling and the plan
//! chose the frame quantizer), or move more than `inframe.row_step` QP units
//! between rows (spatial smoothness), or exceed `inframe.max_offset`. Rows
//! coarsened mid-frame are reported to rate control as a realised block
//! offset, so the size model learns the frame's real step.
//!
//! Off by default (`inframe.room` and `inframe.budgets` 0): every preset of
//! every codec is byte-identical without it. Measurements:
//! `docs/analysis/within-frame-rate-control-2026-11-14.md`.

use nc_analysis::preanalysis::PreAnalysis;

use crate::params::InFrameParams;
use crate::qp_cadence::{QP_UNITS_PER_OCTAVE, step_ratio};

/// log2 of the side of a pre-analysis cell (8x8 luma).
const CELL_LOG2: u8 = 3;

/// Expected bit weight of each of `rows` superblock rows of `1 << sb_log2`
/// luma lines: the sum over its 8x8 cells of `1 + sqrt(activity)`, the
/// motion-compensated SAD where the analysis has a previous frame and the
/// spatial deviation otherwise (block AQ's bit weight, `aq.rs`); one per
/// cell column where no analysis covers the row (keys without analysis,
/// padding), i.e. by area.
pub(crate) fn row_weights(pre: Option<&PreAnalysis>, sb_log2: u8, rows: usize) -> Vec<f64> {
    let cell_rows_per_sb = 1usize << sb_log2.saturating_sub(CELL_LOG2);
    let Some(pre) = pre.filter(|p| p.blocks.cols() > 0) else {
        return vec![1.; rows];
    };
    let cols = pre.blocks.cols();
    let cells = pre.blocks.cells();
    (0..rows)
        .map(|row| {
            let first = row * cell_rows_per_sb * cols;
            let last = ((row + 1) * cell_rows_per_sb * cols).min(cells.len());
            let covered = cells.get(first..last).unwrap_or(&[]);
            let measured: f64 = covered
                .iter()
                .map(|b| 1. + f64::from(b.sad_prev.unwrap_or(b.sum_sq_dev)).sqrt())
                .sum();
            let uncovered = cell_rows_per_sb * cols - covered.len();
            measured + uncovered as f64
        })
        .collect()
}

/// One superblock row of the frame: its expected share of the frame's bits
/// and the offset it was coded at.
#[derive(Debug, Clone, Copy)]
struct Row {
    weight: f64,
    offset: f64,
}

/// In-frame controller of one encoder (`None` in the encoder when off).
#[derive(Debug)]
pub(crate) struct InFrameControl {
    /// Share of the re-code room a frame is steered to land in (0 = no
    /// hard limit).
    room: f64,
    /// Inter frame soft limit in uncorrected frame budgets (0 = none).
    budgets: f64,
    /// Whether keys are steered (by the hard limit) too.
    keys: bool,
    /// Whether the soft limit applies only during the cadence start-up.
    startup_only: bool,
    /// Largest coarsening, QP units.
    max_offset: f64,
    /// Largest change between consecutive rows, QP units.
    row_step: f64,
    /// Slope of bits against step the projection assumes for the rows
    /// still to code.
    exponent: f64,
    /// Bit limit of the frame being coded (`None`: not steering it).
    limit: Option<f64>,
    /// Bits the backend reported before the first row (headers).
    start_bits: f64,
    /// Every row's weight; offsets are filled in as rows are coded.
    rows: Vec<Row>,
    /// Rows coded so far.
    coded: usize,
    /// Offset of the row being coded, QP units.
    offset: f64,
}

impl InFrameControl {
    /// `None` when both limits are off.
    pub fn new(params: &InFrameParams) -> Option<Self> {
        (params.room > 0. || params.budgets > 0.).then(|| Self {
            room: f64::from(params.room),
            budgets: f64::from(params.budgets),
            keys: params.keys >= 0.5,
            startup_only: params.startup_only >= 0.5,
            max_offset: f64::from(params.max_offset),
            row_step: f64::from(params.row_step),
            exponent: f64::from(params.exponent),
            limit: None,
            start_bits: 0.,
            rows: Vec::new(),
            coded: 0,
            offset: 0.,
        })
    }

    /// The frame's bit limit from the re-code room `room_bits` (infinite
    /// when re-coding is off) and the uncorrected frame budget
    /// `uniform_bits`; the soft limit applies to inter frames only (keys
    /// are planned loans), with `inframe.startup_only` only to frames in the
    /// cadence start-up (`startup`), and keys are steered only with
    /// `inframe.keys`.
    /// `None` when neither limit is finite.
    pub fn frame_limit(
        &self,
        (key, startup): (bool, bool),
        room_bits: f64,
        uniform_bits: f64,
    ) -> Option<f64> {
        if key && !self.keys {
            return None;
        }
        let hard = if self.room > 0. {
            self.room * room_bits
        } else {
            f64::INFINITY
        };
        let soft = if self.budgets > 0. && !key && (startup || !self.startup_only) {
            self.budgets * uniform_bits
        } else {
            f64::INFINITY
        };
        let limit = hard.min(soft);
        limit.is_finite().then_some(limit.max(0.))
    }

    /// Plan the next frame: limited to `limit` bits (`None`: coded as
    /// planned, with no block-QP syntax needed for it), with one expected
    /// bit weight per superblock row (`row_weights`).
    pub fn plan_frame(&mut self, limit: Option<f64>, row_weights: impl Iterator<Item = f64>) {
        self.limit = limit;
        self.rows.clear();
        if limit.is_some() {
            self.rows
                .extend(row_weights.map(|weight| Row { weight, offset: 0. }));
        }
        self.coded = 0;
        self.offset = 0.;
    }

    /// Start coding the planned frame (again, after a re-code), after the
    /// backend has written `start_bits` (headers).
    pub fn begin_frame(&mut self, start_bits: u64) {
        self.start_bits = start_bits as f64;
        for row in &mut self.rows {
            row.offset = 0.;
        }
        self.coded = 0;
        self.offset = 0.;
    }

    /// Whether the current frame is steered.
    pub fn steering(&self) -> bool {
        self.limit.is_some()
    }

    /// Offset of the row being coded, QP units (0 or coarser).
    pub fn offset_qp(&self) -> f64 {
        self.offset
    }

    /// The row just coded is done and the frame has `bits` so far: choose
    /// the next row's offset.
    pub fn end_row(&mut self, bits: u64) {
        let Some(limit) = self.limit else {
            return;
        };
        if let Some(row) = self.rows.get_mut(self.coded) {
            row.offset = self.offset;
            self.coded += 1;
        }
        let needed = self.needed_offset(bits as f64, limit);
        self.offset = needed
            .clamp(self.offset - self.row_step, self.offset + self.row_step)
            .clamp(0., self.max_offset);
    }

    /// The offset at which the rows still to code would end the frame at
    /// `limit`, given `bits` spent so far: the bits per unit of weight so
    /// far (normalised to offset 0 through `exponent`) priced over the
    /// remaining weight. Unbounded above when the limit is already spent.
    fn needed_offset(&self, bits: f64, limit: f64) -> f64 {
        let (done, rest) = self.rows.split_at(self.coded);
        let remaining: f64 = rest.iter().map(|r| r.weight).sum();
        let normalised: f64 = done
            .iter()
            .map(|r| r.weight * step_ratio(r.offset).powf(-self.exponent))
            .sum();
        let data = bits - self.start_bits;
        if remaining <= 0. || normalised <= 0. || data <= 0. {
            return 0.;
        }
        let available = limit - bits;
        if available <= 0. {
            return f64::INFINITY;
        }
        let projected = data / normalised * remaining;
        QP_UNITS_PER_OCTAVE / self.exponent * (projected / available).log2()
    }

    /// `(weight, offset)` of every coded row with a non-zero offset, and
    /// the frame's total weight: what rate control folds into one realised
    /// offset (`RateController::effective_block_offset`).
    pub fn realised(&self) -> (f64, impl Iterator<Item = (f64, f64)> + '_) {
        let total = self.rows.iter().map(|r| r.weight).sum();
        let offsets = self.rows[..self.coded]
            .iter()
            .filter(|r| r.offset != 0.)
            .map(|r| (r.weight, r.offset));
        (total, offsets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ParamOverrides;

    fn control(room: f32, budgets: f32) -> InFrameControl {
        let mut o = ParamOverrides::default();
        o.set("inframe.room", room);
        o.set("inframe.budgets", budgets);
        InFrameControl::new(&InFrameParams::from_overrides(&o)).expect("on")
    }

    #[test]
    fn off_by_default() {
        assert!(
            InFrameControl::new(&InFrameParams::from_overrides(&ParamOverrides::default()))
                .is_none()
        );
    }

    #[test]
    fn limits_take_the_tighter_and_spare_keys_the_soft_one() {
        let mut c = control(0.5, 8.);
        assert_eq!(c.frame_limit((true, false), 100_000., 1_000.), None);
        c.keys = true;
        assert_eq!(
            c.frame_limit((false, false), 100_000., 1_000.),
            Some(8_000.)
        );
        assert_eq!(
            c.frame_limit((true, false), 100_000., 1_000.),
            Some(50_000.)
        );
        assert_eq!(c.frame_limit((true, false), f64::INFINITY, 1_000.), None);
        c.startup_only = true;
        c.room = 0.;
        assert_eq!(c.frame_limit((false, false), 100_000., 1_000.), None);
        assert_eq!(c.frame_limit((false, true), 100_000., 1_000.), Some(8_000.));
    }

    #[test]
    fn a_frame_on_budget_is_left_alone() {
        let mut c = control(1., 0.);
        c.plan_frame(Some(10_000.), std::iter::repeat_n(1., 10));
        c.begin_frame(0);
        for row in 1..=10 {
            c.end_row(row * 900);
            assert_eq!(c.offset_qp(), 0.);
        }
        assert_eq!(c.realised().1.count(), 0);
    }

    #[test]
    fn a_burst_is_coarsened_in_bounded_steps() {
        let mut c = control(1., 0.);
        c.plan_frame(Some(10_000.), std::iter::repeat_n(1., 10));
        c.begin_frame(0);
        // The first row alone spends 40% of the limit: the frame heads for 4x.
        c.end_row(4_000);
        let first = c.offset_qp();
        assert!(first > 0. && first <= 3.);
        let mut bits = 4_000;
        let mut previous = first;
        for _ in 1..9 {
            bits += 300;
            c.end_row(bits);
            assert!((c.offset_qp() - previous).abs() <= 3. + 1e-9);
            assert!(c.offset_qp() <= 12.);
            previous = c.offset_qp();
        }
        // Once the projection fits, the offset relaxes again (by the same
        // step), so only some rows end up coarser.
        let (total, offsets) = c.realised();
        assert_eq!(total, 10.);
        assert!(offsets.count() >= 1);
    }

    #[test]
    fn a_spent_limit_coarsens_at_full_step() {
        let mut c = control(1., 0.);
        c.plan_frame(Some(1_000.), std::iter::repeat_n(1., 4));
        c.begin_frame(0);
        c.end_row(2_000);
        assert_eq!(c.offset_qp(), 3.);
    }
}
