//! Layer propagation: per-block modulation of the temporal-layer quantizer
//! ladder on base (T0) frames, from how much of each base block the frames
//! of the last layer period predicted from (`layers.propagation`).
//!
//! A layered structure (L1T2, L1T3) codes T0 finer than the upper layers
//! (`qp_cadence::LayerLadder`) because later frames predict from it. The
//! ladder assumes every T0 block is reused alike. MB-tree (x264) and TPL
//! (libaom, SVT-AV1) measure the reuse per block from a lookahead: each
//! block of frame `t` sends `(intra + received) · (1 - inter / intra)`
//! back along its vector to its reference's blocks, by area, and a block's
//! received share sets its offset. With no lookahead the measurement comes
//! one period late: when a T0 frame arrives, every frame that referenced
//! the previous T0 (directly, or through T1) has been analysed, so the
//! previous T0's propagation is known. Its reuse share, normalised by what
//! a perfectly still block receives over the period (one per frame), is
//! averaged into a history and carried to the current T0's blocks along the
//! current frame's own vectors (which point into the previous T0); that is
//! the forecast. Costs are the analysis's: intra from the cell's sum of
//! squared deviations (Gaussian SAD estimate, as in the reuse forecast of
//! `docs/handoff/block-aq.md`), inter the 8x8 motion-search SAD.
//!
//! **Offset** (one-sided): `strength · gate · top · (1 - forecast)`, where
//! `top` is the ladder's top-layer offset: a block nothing reused goes up to
//! `strength` of the way to the top layer's quantizer, a block reused like
//! a still one keeps the T0 quantizer. Rate control budgets the offsets as
//! any block AQ (`RateController::effective_block_offset`), so the bits
//! move inside the T0 frame from unreferenced to referenced blocks; under
//! fixed QP they are saved.
//!
//! **Gate**: each period compares the offsets its forecast planned for the
//! previous T0 with those the measured propagation implies; the moving
//! average of the mean absolute difference (QP units) closes the gate
//! linearly up to `layers.propagation_gate`, so content whose reuse the
//! history does not predict (a shaking camera) keeps the static ladder.
//!
//! **Before a key** (`layers.propagation_prekey`): no offsets in the last
//! `rate.prekey_quiet` seconds before a scheduled periodic key. The offsets
//! coarsen a T0 frame's blocks, so rate control lowers its level, and a key
//! planned from that level came out finer than the frames before it (AV1
//! Faster final check: long-window key pulse +0.40 dB; with the stop +0.00).
//!
//! Analysis-only: no pixel pass; a few passes over the 8x8 cells per frame.

use nc_analysis::MotionAnalysis;
use nc_analysis::preanalysis::PreAnalysis;

use crate::params::LayerParams;

/// Side of a cell (the 8x8 analysis grid).
const CELL: f32 = 8.;
/// Samples per cell.
const CELL_SAMPLES: f32 = CELL * CELL;
/// Mean absolute per squared deviation of a Gaussian (`2/π`): the intra
/// cost estimate from the cell's sum of squared deviations.
const GAUSSIAN_ABS_PER_SQUARE: f32 = std::f32::consts::FRAC_2_PI;
/// Intra cost floor per sample (sensor noise): a flat cell costs at least
/// this much to code from scratch.
const MIN_INTRA_PER_SAMPLE: f32 = 1.;
/// Quantile of a frame's 8x8 motion-search SADs taken as its temporal noise
/// floor: residual at or below it is noise no coding removes, so the block
/// is copied and its reference's quality persists. Without it, flat noisy
/// still cells (inter SAD near their small intra cost) counted as
/// unreferenced and went coarse (AV1 Veryfast probe, `bodycam_airport`
/// 300 kbps: -4 dB windows on its near-still stretches).
const NOISE_QUANTILE: f64 = 0.25;
/// Weight of the gate's error history per period.
const GATE_MEMORY: f64 = 0.75;
/// Overlap weight below which a warped cell (mostly off the picture) keeps
/// its co-located history instead.
const MIN_WARP_WEIGHT: f32 = 0.5;

/// One frame of the current layer period, as the propagation needs it.
#[derive(Debug)]
struct PeriodFrame {
    index: u64,
    /// The frame it predicts from (`None`: a key, or a base it restarts at).
    reference: Option<u64>,
    intra: Vec<f32>,
    /// Vector (luma samples) and SAD of each cell against `reference`.
    motion: Vec<(i16, i16, u32)>,
    /// The frame's temporal noise floor (a cell SAD, [`NOISE_QUANTILE`]).
    noise: u32,
    /// Propagation received from later frames (scratch).
    received: Vec<f32>,
}

/// What the propagation needs of the frame being planned.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerFrame<'a> {
    pub index: u64,
    /// The frame's primary reference, `index - distance`.
    pub reference: u64,
    pub key: bool,
    /// Layered structure (`LayerPosition::layers > 1`).
    pub layered: bool,
    /// Temporal layer (0 is the base).
    pub layer: u8,
    /// The ladder's top-layer offset (QP units), the coarsest it codes.
    pub top_offset: f64,
    /// The frame lies in the run-up to a scheduled periodic key
    /// (`rate.prekey_quiet`).
    pub before_key: bool,
    pub pre: &'a PreAnalysis,
    pub motion: Option<&'a MotionAnalysis>,
}

/// Per-stream propagation state.
#[derive(Debug)]
pub(crate) struct LayerPropagation {
    strength: f64,
    /// Whether offsets stop in the run-up to a key (`layers.propagation_prekey`).
    prekey: bool,
    memory: f32,
    gate_error: f64,
    cols: usize,
    /// The previous base frame and the frames since.
    period: Vec<PeriodFrame>,
    /// The previous base frame's forecast reuse, `None` before one exists.
    forecast: Option<Vec<f32>>,
    /// Moving average of the forecast's offset error (QP units).
    error: f64,
    /// The current frame's offsets (QP units), empty when it has none.
    offsets: Vec<f64>,
    /// Offset error of the last measured period (for reports and tests).
    last_error: Option<f64>,
}

impl LayerPropagation {
    /// `None` when `layers.propagation` is 0.
    pub fn new(params: &LayerParams) -> Option<Self> {
        (params.propagation > 0.).then(|| Self {
            strength: f64::from(params.propagation),
            prekey: params.propagation_prekey >= 0.5,
            memory: params.propagation_memory,
            gate_error: f64::from(params.propagation_gate),
            cols: 0,
            period: Vec::new(),
            forecast: None,
            error: 0.,
            offsets: Vec::new(),
            last_error: None,
        })
    }

    /// The current frame's offsets, if it has any.
    pub fn offsets(&self) -> Option<&[f64]> {
        (!self.offsets.is_empty()).then_some(self.offsets.as_slice())
    }

    /// Fold in the frame being planned and plan its offsets.
    pub fn plan(&mut self, frame: &LayerFrame<'_>) {
        self.offsets.clear();
        let cols = frame.pre.blocks.cols();
        let intra: Vec<f32> = frame
            .pre
            .blocks
            .cells()
            .iter()
            .map(|b| intra_cost(b.sum_sq_dev))
            .collect();
        let motion = frame
            .motion
            .filter(|m| m.fine.cols() == cols && m.fine.cells().len() == intra.len())
            .filter(|_| !frame.key);
        let restart = frame.key || !frame.layered || motion.is_none() || cols != self.cols;
        if restart {
            // A key, a flat frame or a new geometry: this frame starts a
            // period with no history (a flat chain has no layer period).
            self.cols = cols;
            self.forecast = None;
            self.period.clear();
            self.period
                .push(PeriodFrame::new(frame.index, None, intra, None));
            return;
        }
        let motion = motion.expect("checked above");
        let reference = self
            .period
            .iter()
            .any(|f| f.index == frame.reference)
            .then_some(frame.reference);
        self.period.push(PeriodFrame::new(
            frame.index,
            reference,
            intra,
            Some(motion),
        ));
        if frame.layer != 0 {
            return;
        }
        let measured = self.propagate();
        if let (Some(measured), Some(forecast)) = (&measured, &self.forecast) {
            let top = frame.top_offset.max(0.);
            let error = forecast
                .iter()
                .zip(measured)
                .map(|(f, m)| f64::from((f - m).abs()))
                .sum::<f64>()
                / measured.len().max(1) as f64
                * top;
            self.error = match self.last_error {
                Some(_) => GATE_MEMORY * self.error + (1. - GATE_MEMORY) * error,
                None => error,
            };
            self.last_error = Some(error);
        }
        let history = match (measured, self.forecast.take()) {
            (Some(measured), Some(forecast)) => measured
                .iter()
                .zip(&forecast)
                .map(|(m, f)| self.memory * f + (1. - self.memory) * m)
                .collect(),
            (Some(measured), None) => measured,
            (None, forecast) => forecast.unwrap_or_default(),
        };
        let current = self.period.pop().expect("pushed above");
        self.period.clear();
        if history.len() == current.intra.len() {
            let forecast = warp(&history, &current.motion, cols);
            // A finer key would be planned from the level the offsets
            // lowered (the frame's blocks went coarser, so its level finer).
            let quiet = self.prekey && frame.before_key;
            let scale = if quiet {
                0.
            } else {
                self.strength * self.gate() * frame.top_offset.max(0.)
            };
            if scale > 0. {
                self.offsets
                    .extend(forecast.iter().map(|&f| scale * f64::from(1. - f)));
            }
            self.forecast = Some(forecast);
        }
        self.period.push(PeriodFrame {
            reference: None,
            motion: Vec::new(),
            ..current
        });
    }

    /// The gate (0..1) from the forecast's recent error.
    fn gate(&self) -> f64 {
        if self.gate_error <= 0. {
            return 1.;
        }
        (1. - self.error / self.gate_error).clamp(0., 1.)
    }

    /// Propagate the period's frames, newest first, into the previous base
    /// frame (the period's first entry) and return its reuse per cell in
    /// `[0, 1]`: received over intra, normalised by a still cell's
    /// (`log2(1 + received / intra) / log2(1 + frames)`).
    fn propagate(&mut self) -> Option<Vec<f32>> {
        let frames = self.period.len().checked_sub(1).filter(|&n| n > 0)?;
        let cols = self.cols;
        for f in &mut self.period {
            f.received.clear();
            f.received.resize(f.intra.len(), 0.);
        }
        for newest in (1..self.period.len()).rev() {
            let (older, rest) = self.period.split_at_mut(newest);
            let frame = &rest[0];
            let Some(target) = frame
                .reference
                .and_then(|r| older.iter_mut().find(|f| f.index == r))
            else {
                continue;
            };
            let rows = frame.intra.len() / cols.max(1);
            for (cell, &(mvx, mvy, sad)) in frame.motion.iter().enumerate() {
                let intra = frame.intra[cell];
                let residual = sad.saturating_sub(frame.noise) as f32;
                let reused = 1. - residual.min(intra) / intra;
                let amount = (intra + frame.received[cell]) * reused;
                if amount <= 0. {
                    continue;
                }
                for (source, weight) in overlap(cell, cols, rows, mvx, mvy) {
                    target.received[source] += amount * weight;
                }
            }
        }
        let base = &self.period[0];
        let full = (1. + frames as f32).log2();
        Some(
            base.received
                .iter()
                .zip(&base.intra)
                .map(|(received, intra)| ((1. + received / intra).log2() / full).min(1.))
                .collect(),
        )
    }
}

impl PeriodFrame {
    fn new(
        index: u64,
        reference: Option<u64>,
        intra: Vec<f32>,
        motion: Option<&MotionAnalysis>,
    ) -> Self {
        let motion = motion.map_or_else(Vec::new, |m| {
            m.fine
                .cells()
                .iter()
                .map(|b| (b.mv.x, b.mv.y, b.sad))
                .collect()
        });
        let mut sads: Vec<u32> = motion.iter().map(|&(_, _, sad)| sad).collect();
        let noise = if sads.is_empty() {
            0
        } else {
            let at = ((sads.len() - 1) as f64 * NOISE_QUANTILE) as usize;
            *sads.select_nth_unstable(at).1
        };
        Self {
            index,
            noise,
            reference: reference.filter(|_| !motion.is_empty()),
            intra,
            motion,
            received: Vec::new(),
        }
    }
}

/// `history` of the previous base frame carried to the current one's
/// cells along their vectors (`motion`, into the previous base), by area.
fn warp(history: &[f32], motion: &[(i16, i16, u32)], cols: usize) -> Vec<f32> {
    let rows = history.len() / cols.max(1);
    motion
        .iter()
        .enumerate()
        .map(|(cell, &(mvx, mvy, _))| {
            let (sum, weight) = overlap(cell, cols, rows, mvx, mvy)
                .fold((0., 0.), |(s, w), (source, weight)| {
                    (s + history[source] * weight, w + weight)
                });
            if weight < MIN_WARP_WEIGHT {
                history[cell]
            } else {
                sum / weight
            }
        })
        .collect()
}

/// Intra cost estimate of a cell from its sum of squared deviations.
fn intra_cost(sum_sq_dev: u32) -> f32 {
    (GAUSSIAN_ABS_PER_SQUARE * CELL_SAMPLES * sum_sq_dev as f32).sqrt()
        + CELL_SAMPLES * MIN_INTRA_PER_SAMPLE
}

/// The cells of a `cols` x `rows` grid that the block of `cell` displaced
/// by `(mvx, mvy)` samples overlaps, with the share of its area in each.
fn overlap(
    cell: usize,
    cols: usize,
    rows: usize,
    mvx: i16,
    mvy: i16,
) -> impl Iterator<Item = (usize, f32)> {
    let y = (cell / cols) as f32 + f32::from(mvy) / CELL;
    let x = (cell % cols) as f32 + f32::from(mvx) / CELL;
    let (y0, x0) = (y.floor(), x.floor());
    let (fy, fx) = (y - y0, x - x0);
    [(0., 1. - fy), (1., fy)]
        .into_iter()
        .flat_map(move |(dy, wy)| {
            [(0., 1. - fx), (1., fx)]
                .into_iter()
                .filter_map(move |(dx, wx)| {
                    let (r, c) = (y0 + dy, x0 + dx);
                    let weight = wy * wx;
                    (weight > 0.
                        && r >= 0.
                        && c >= 0.
                        && (r as usize) < rows
                        && (c as usize) < cols)
                        .then(|| (r as usize * cols + c as usize, weight))
                })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_splits_by_area() {
        let parts: Vec<_> = overlap(0, 3, 1, 4, 0).collect();
        assert_eq!(parts, [(0, 0.5), (1, 0.5)]);
        assert_eq!(overlap(0, 3, 1, -8, 0).count(), 0);
    }

    #[test]
    fn a_still_cell_is_fully_reused_and_a_new_one_not() {
        // Two cells; an L1T3 period of four frames after base 0. Cell 0 is
        // still (SAD 0), cell 1 changes completely (SAD at its intra cost).
        let intra = vec![intra_cost(64 * 100); 2];
        let still = |sad1| vec![(0, 0, 0), (0, 0, sad1)];
        let full = intra[1] as u32 * 2;
        let mut p = LayerPropagation {
            strength: 1.,
            prekey: false,
            memory: 0.,
            gate_error: 0.,
            cols: 2,
            period: vec![PeriodFrame {
                index: 0,
                reference: None,
                intra: intra.clone(),
                motion: Vec::new(),
                received: Vec::new(),
                noise: 0,
            }],
            forecast: None,
            error: 0.,
            offsets: Vec::new(),
            last_error: None,
        };
        for (index, reference) in [(1, 0), (2, 0), (3, 2), (4, 0)] {
            p.period.push(PeriodFrame {
                index,
                reference: Some(reference),
                intra: intra.clone(),
                motion: still(full),
                received: Vec::new(),
                noise: 0,
            });
        }
        let reuse = p.propagate().unwrap();
        assert!((reuse[0] - 1.).abs() < 1e-6);
        assert!(reuse[1].abs() < 1e-6);
    }
}
