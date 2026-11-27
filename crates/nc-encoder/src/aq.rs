//! Block adaptive quantization: a per-block quantizer offset, in QP units,
//! from two terms that add.
//!
//! - **Variance** (`rate.aq_strength`): detailed (high-variance) blocks mask
//!   quantization noise, flat blocks show it, so moving bits from the first
//!   to the second at equal rate improves perceived quality (x264's and
//!   libvpx's variance AQ; the idea, not their code). The term is
//!   `strength * (log2(variance) - mean log2 variance of the frame)`, finer
//!   where flatter than average.
//! - **Persistence** (`persist.strength`, `persistence.rs`): blocks whose
//!   content later frames predict well get finer, unpredictable ones coarser.
//! - **Reuse** (`persist.reuse`, `reuse.rs`): still blocks whose measured
//!   reconstruction fell below their long-run quality are refined, at the
//!   step their reuse warrants. Its offsets follow from λ and are not
//!   clamped by `rate.aq_max_offset`, which bounds the two heuristic terms.
//! - **Layer propagation** (`layers.propagation`, `layer_propagation.rs`):
//!   on base frames of a layered structure, blocks the last period's frames
//!   reused little go towards the top layer's quantizer. Its offsets are
//!   pooled, rounded to `layers.propagation_grid` and not clamped (they are
//!   bounded by the ladder).
//!
//! The analysis already has every 8x8 block's sum of squared deviations
//! and motion error, so no pixel pass is added. Cell offsets are pooled per
//! aligned square of the codec's quantizer granularity
//! (`CodecBackend::BLOCK_QP_LOG2`), so every leaf inside one shares its
//! offset and rate control budgets the offsets the codec can code, then
//! clamped to `±rate.aq_max_offset`. Offsets are converted to the codec's
//! native quantizer through its step table, so a QP unit means the same step
//! ratio in every codec. Only backends with block-level QP syntax
//! (`CodecBackend::SUPPORTS_BLOCK_QP`) receive non-zero offsets.

use nc_analysis::MotionAnalysis;
use nc_analysis::preanalysis::{BlockStats, PreAnalysis};

use crate::layer_propagation::{LayerFrame, LayerPropagation};
use crate::params::{LayerParams, ParamOverrides, PersistParams, RateParams};
use crate::partition::Block;
use crate::persistence::PersistenceTracker;
use crate::qp_cadence::{nearest_quantizer, step_ratio};

/// Side of a pre-analysis cell (8x8 luma).
const CELL_LOG2: u8 = 3;
/// Samples per cell, the divisor of `sum_sq_dev` for the variance.
const CELL_SAMPLES: f64 = 64.;
/// Variance floor, so flat blocks do not dominate the log mean.
const MIN_VARIANCE: f64 = 1.;
/// Largest spread of `log2(variance)` over a frame's cells: an 8-bit 8x8
/// block's variance is at most `255² / 4` (half its samples at 0, half at
/// 255), below `2^14`, and [`MIN_VARIANCE`] puts the floor at `2^0`.
const LOG2_VARIANCE_SPAN: f64 = 14.;
/// Distinct offsets kept free for the reuse term (added after the grid)
/// when the codec codes only a few per frame.
const RESERVED_REUSE_OFFSETS: usize = 1;

/// Per-frame block AQ state of one encoder.
#[derive(Debug)]
pub(crate) struct BlockAq {
    /// Variance term strength (`rate.aq_strength`, 0 = off).
    strength: f64,
    /// Largest offset either way, in QP units (`rate.aq_max_offset`).
    max_offset: f64,
    /// log2 of the side of the square sharing one offset
    /// (`CodecBackend::BLOCK_QP_LOG2`).
    unit_log2: u8,
    /// Grid (QP units) the heuristic terms are rounded to when the codec
    /// codes only a few distinct offsets per frame
    /// (`CodecBackend::BLOCK_QP_OFFSETS`), else `None`.
    offset_step: Option<f64>,
    persistence: Option<PersistenceTracker>,
    /// The layer propagation term (`layers.propagation`), if on.
    propagation: Option<LayerPropagation>,
    /// Grid (QP units) the propagation term is rounded to, 0 for none.
    propagation_grid: f64,
    cols: usize,
    /// Offset of every cell of the current frame in QP units, pooled per
    /// quantizer square (the heuristic terms clamped).
    offset: Vec<f64>,
    /// The reuse term's share of `offset`, before pooling (scratch).
    reuse_offset: Vec<f64>,
    /// The propagation term's share of `offset`, before pooling (scratch).
    layer_offset: Vec<f64>,
    /// The cells of `offset` that are not zero.
    nonzero: Vec<usize>,
    /// Cells already refined on R1 (masked out on R2 in split mode).
    r1_refined: Vec<bool>,
    /// Whether any cell of the planned frame has a non-zero offset.
    active: bool,
    /// Sum of `bit_weight` over the frame.
    total_weight: f64,
    /// Expected share of the frame's bits of every cell (rate control's
    /// block-offset weight): its residual activity, temporal where the
    /// pre-analysis has a previous frame, spatial otherwise.
    bit_weight: Vec<f32>,
}

impl BlockAq {
    /// `None` when every term is off. Offsets are planned per aligned
    /// `1 << unit_log2` square (cells of 8x8 at 3 or below), at most
    /// `max_offsets` distinct non-zero ones per frame.
    pub fn new(overrides: &ParamOverrides, unit_log2: u8, max_offsets: usize) -> Option<Self> {
        let params = RateParams::from_overrides(overrides);
        let persist = PersistParams::from_overrides(overrides);
        let persistence = PersistenceTracker::new(&persist);
        let reuse = persist.reuse >= 0.5;
        let layers = LayerParams::from_overrides(overrides);
        let propagation = LayerPropagation::new(&layers);
        let any = params.aq_strength > 0. || persistence.is_some() || propagation.is_some();
        (any || reuse).then(|| Self {
            strength: f64::from(params.aq_strength),
            max_offset: f64::from(params.aq_max_offset),
            unit_log2,
            offset_step: heuristic_offset_step(params.aq_max_offset, max_offsets),
            persistence,
            propagation,
            propagation_grid: f64::from(layers.propagation_grid),
            cols: 0,
            offset: Vec::new(),
            reuse_offset: Vec::new(),
            layer_offset: Vec::new(),
            nonzero: Vec::new(),
            r1_refined: Vec::new(),
            active: false,
            total_weight: 0.,
            bit_weight: Vec::new(),
        })
    }

    /// Whether the layer propagation term is on.
    pub fn propagates(&self) -> bool {
        self.propagation.is_some()
    }

    /// Fold the frame being planned into the layer propagation term, if on
    /// (before [`Self::plan`]).
    pub fn plan_layer(&mut self, frame: &LayerFrame<'_>) {
        if let Some(propagation) = self.propagation.as_mut() {
            propagation.plan(frame);
        }
    }

    /// Measure the frame from its analysis: `motion` is `None` on frames
    /// without a motion search (key frames, the first frame); `reuse` is
    /// the reuse term's offset of every cell (`reuse.rs`), if it is on.
    pub fn plan(
        &mut self,
        pre: &PreAnalysis,
        motion: Option<&MotionAnalysis>,
        reuse: Option<&[f64]>,
    ) {
        self.cols = pre.blocks.cols();
        let reuse = reuse.filter(|r| r.iter().any(|&offset| offset != 0.));
        let cells = pre.blocks.cells().len();
        let layer = self
            .propagation
            .as_ref()
            .and_then(LayerPropagation::offsets)
            .filter(|l| l.len() == cells && l.iter().any(|&offset| offset != 0.));
        let heuristic = self.strength > 0. || self.persistence.is_some() || layer.is_some();
        if !heuristic && reuse.is_none() {
            // Nothing to plan (most frames with only the reuse term on):
            // the frame codes as without block AQ.
            self.active = false;
            return;
        }
        if !heuristic && let Some(reuse) = reuse.filter(|r| r.len() == cells) {
            self.plan_reuse_only(pre, reuse);
            return;
        }
        self.bit_weight.clear();
        // Gathered first, then the square roots on their own: that loop
        // vectorises (in one pass it cost ~20 instructions per cell). f32
        // is ample for a weight and halves the passes' memory.
        self.bit_weight
            .extend(pre.blocks.cells().iter().map(cell_activity));
        for weight in &mut self.bit_weight {
            *weight = bits_weight(*weight);
        }
        self.total_weight = self.bit_weight.iter().map(|&w| f64::from(w)).sum();
        self.offset.clear();
        if self.strength > 0. && !self.variance_rounds_away() {
            self.offset.extend(pre.blocks.cells().iter().map(|b| {
                (f64::from(b.sum_sq_dev) / CELL_SAMPLES)
                    .max(MIN_VARIANCE)
                    .log2()
            }));
            let count = self.offset.len().max(1) as f64;
            let mean = self.offset.iter().sum::<f64>() / count;
            for offset in &mut self.offset {
                *offset = self.strength * (*offset - mean);
            }
        } else {
            self.offset.resize(pre.blocks.cells().len(), 0.);
        }
        if let Some(persistence) = self.persistence.as_mut() {
            persistence.observe(motion);
            persistence.add_offsets(self.cols, &mut self.offset);
        }
        self.reuse_offset.clear();
        match reuse.filter(|r| r.len() == self.offset.len()) {
            Some(reuse) => self.reuse_offset.extend_from_slice(reuse),
            None => self.reuse_offset.resize(self.offset.len(), 0.),
        }
        // Pooling all-zero offsets (most frames with only the reuse term
        // on) changes nothing.
        let pool = |values: &mut Vec<f64>| {
            if values.iter().any(|&v| v != 0.) {
                pool_units(values, self.cols, self.unit_log2);
            }
        };
        self.layer_offset.clear();
        match layer {
            Some(layer) => self.layer_offset.extend_from_slice(layer),
            None => self.layer_offset.resize(self.offset.len(), 0.),
        }
        if self.unit_log2 > CELL_LOG2 {
            pool(&mut self.offset);
            pool(&mut self.reuse_offset);
            pool(&mut self.layer_offset);
        }
        let grid = self.propagation_grid;
        if grid > 0. {
            for offset in &mut self.layer_offset {
                *offset = (*offset / grid).round() * grid;
            }
        }
        let max = self.max_offset;
        for ((offset, reuse), layer) in self
            .offset
            .iter_mut()
            .zip(&self.reuse_offset)
            .zip(&self.layer_offset)
        {
            let clamped = offset.clamp(-max, max);
            let coded = self
                .offset_step
                .map_or(clamped, |step| (clamped / step).round() * step);
            *offset = coded + reuse + layer;
        }
        self.nonzero.clear();
        self.nonzero
            .extend((0..self.offset.len()).filter(|&cell| self.offset[cell] != 0.));
        self.active = !self.nonzero.is_empty();
    }

    /// Whether the variance term is too weak to move any cell by a coded
    /// step: with only a few distinct offsets per frame its values are
    /// rounded to `offset_step`, and when its largest possible magnitude
    /// (`strength` x [`LOG2_VARIANCE_SPAN`]; pooling only averages them) is
    /// under half a step, every cell rounds to 0, as if it were off. Only
    /// when no persistence offset is added before the rounding. (VP9 Fast's
    /// `rate.aq_strength` is 0.0003: its per-cell `log2` cost ~1 % of the
    /// rung's instructions for offsets that were always 0.)
    fn variance_rounds_away(&self) -> bool {
        self.persistence.is_none()
            && self
                .offset_step
                .is_some_and(|step| self.strength * LOG2_VARIANCE_SPAN < step / 2.)
    }

    /// Plan a frame whose only offsets are the reuse term's `reuse`: a few
    /// refined cells, so only their quantizer squares are pooled and
    /// written (the general path pooled and scanned every cell, ~2% of
    /// Jetson H.264 Faster cycles on the frames that refine anything).
    fn plan_reuse_only(&mut self, pre: &PreAnalysis, reuse: &[f64]) {
        if self.bit_weight.len() != reuse.len() {
            self.bit_weight.clear();
            self.bit_weight.resize(reuse.len(), 0.);
        }
        if self.offset.len() == reuse.len() {
            for &cell in &self.nonzero {
                self.offset[cell] = 0.;
            }
        } else {
            self.offset.clear();
            self.offset.resize(reuse.len(), 0.);
        }
        self.nonzero.clear();
        let (cols, rows) = (self.cols, reuse.len() / self.cols.max(1));
        let side = 1usize << self.unit_log2.saturating_sub(CELL_LOG2);
        for cell in (0..reuse.len()).filter(|&cell| reuse[cell] != 0.) {
            if self.offset[cell] != 0. {
                // Its square is pooled already.
                continue;
            }
            let (row0, col0) = ((cell / cols) / side * side, (cell % cols) / side * side);
            let (row1, col1) = ((row0 + side).min(rows), (col0 + side).min(cols));
            let index = |row: usize, col: usize| row * cols + col;
            let sum: f64 = (row0..row1)
                .flat_map(|row| (col0..col1).map(move |col| (row, col)))
                .map(|(row, col)| reuse[index(row, col)])
                .sum();
            let mean = sum / ((row1 - row0) * (col1 - col0)) as f64;
            for row in row0..row1 {
                self.offset[index(row, col0)..index(row, col1)].fill(mean);
                self.nonzero.extend(index(row, col0)..index(row, col1));
            }
        }
        // Only the offset squares' weights are read (the leaves coded off
        // the frame quantizer lie inside them: a square is the codec's
        // largest block); the frame's total comes from rate control's
        // complexity pass ([`Self::set_total_weight`]).
        for &cell in &self.nonzero {
            self.bit_weight[cell] = bit_weight(&pre.blocks.cells()[cell]);
        }
        self.active = !self.nonzero.is_empty();
    }

    /// The frame's total expected bits weight: the sum of every cell's
    /// [`bit_weight`], which rate control's complexity pass computes
    /// anyway (`Complexity`: its temporal sum, or spatial without a
    /// previous frame).
    pub fn set_total_weight(&mut self, total: f64) {
        self.total_weight = total;
    }

    /// Zero-centered per-block MC-residual first-key refinement offsets on R1/R2.
    /// `pass`: `0` = single-shot, `1` = split R1 (certain zero-MV blocks only, records mask),
    /// `2` = split R2 (skips cells already refined in pass 1).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_first_key_mc_refine(
        &mut self,
        pre: &PreAnalysis,
        motion: &MotionAnalysis,
        refine_qp: f64,
        from: f64,
        full: f64,
        mv_shoulder: f64,
        mv_scale: f64,
        cam_shoulder: f64,
        cam_scale: f64,
        camera_motion: f64,
        pass: u8,
    ) -> Option<f64> {
        let pre_cells = pre.blocks.cells();
        let fine_cells = motion.fine.cells();
        if pre_cells.len() != fine_cells.len() || pre_cells.is_empty() || refine_qp <= 0. {
            return None;
        }
        let n = pre_cells.len();
        if pass == 1 {
            self.r1_refined.clear();
            self.r1_refined.resize(n, false);
            if camera_motion > 0.5 {
                return None;
            }
        }
        let cam_decay = if cam_scale > 0. {
            let excess = (camera_motion - cam_shoulder).max(0.);
            (-excess / cam_scale).exp()
        } else {
            1.0
        };
        if cam_decay * refine_qp < 0.1 {
            return None;
        }
        self.cols = pre.blocks.cols();
        if !self.active || self.offset.len() != n {
            self.offset.clear();
            self.offset.resize(n, 0.);
            self.nonzero.clear();
        }
        let span = full - from;
        let mut refine = vec![0.0f64; n];
        let mut any = false;
        for i in 0..n {
            if pass == 2 && self.r1_refined.get(i).copied().unwrap_or(false) {
                continue;
            }
            let mv_mag = f64::from(fine_cells[i].mv.x).hypot(f64::from(fine_cells[i].mv.y));
            if pass == 1 && mv_mag > 0.5 {
                continue;
            }
            let sigma = (f64::from(pre_cells[i].sum_sq_dev) / CELL_SAMPLES).sqrt();
            let mc_mae = f64::from(fine_cells[i].sad) / CELL_SAMPLES;
            let good = if sigma < 4.0 {
                if mc_mae <= 1.0 { 1.0 } else { 0.0 }
            } else {
                (1.0 - mc_mae / sigma).clamp(0.0, 1.0)
            };
            if pass == 1 && good < full {
                continue;
            }
            let mc_weight = if span > 0. {
                ((good - from) / span).clamp(0.0, 1.0)
            } else if good >= full {
                1.0
            } else {
                0.0
            };
            let mv_decay = if mv_scale > 0. {
                let excess = (mv_mag - mv_shoulder).max(0.);
                (-excess / mv_scale).exp()
            } else {
                1.0
            };
            let weight = mc_weight * mv_decay * cam_decay;
            if weight * refine_qp >= 0.1 {
                refine[i] = -refine_qp * weight;
                any = true;
                if pass == 1 {
                    self.r1_refined[i] = true;
                }
            }
        }
        if !any {
            return None;
        }
        if self.unit_log2 > CELL_LOG2 {
            pool_units(&mut refine, self.cols, self.unit_log2);
        }
        let mean_delta = refine.iter().sum::<f64>() / n as f64;
        for (dst, &delta) in self.offset.iter_mut().zip(&refine) {
            let centered = delta - mean_delta;
            if centered.abs() > 1e-6 {
                *dst += centered;
            }
        }
        self.nonzero.clear();
        self.nonzero
            .extend((0..self.offset.len()).filter(|&cell| self.offset[cell] != 0.));
        self.active = !self.nonzero.is_empty();
        Some(-mean_delta)
    }

    /// Zero-centered per-block MC-confidence QP concentration on cadence base
    /// frames: shifts high-MC-confidence, low-MV blocks finer and unpredictable
    /// blocks coarser at ~zero net bit change.
    pub fn apply_base_mc_concentration(
        &mut self,
        pre: &PreAnalysis,
        motion: &MotionAnalysis,
        strength_qp: f64,
        camera_motion: f64,
    ) {
        let pre_cells = pre.blocks.cells();
        let fine_cells = motion.fine.cells();
        if pre_cells.len() != fine_cells.len() || pre_cells.is_empty() || strength_qp <= 0. {
            return;
        }
        let cam_decay = (-(camera_motion - 1.0).max(0.) / 2.0).exp();
        let eff_qp = strength_qp * cam_decay;
        if eff_qp < 0.25 {
            return;
        }
        let n = pre_cells.len();
        self.cols = pre.blocks.cols();
        if !self.active || self.offset.len() != n {
            self.offset.clear();
            self.offset.resize(n, 0.);
            self.nonzero.clear();
        }
        if self.bit_weight.len() != n {
            self.bit_weight.clear();
            self.bit_weight.resize(n, 0.);
        }
        let mut w = vec![0.0_f64; n];
        let mut sum_w = 0.0_f64;
        let span = 0.35_f64; // 0.50 -> 0.85
        for i in 0..n {
            let b = &pre_cells[i];
            let m = &fine_cells[i];
            let sigma = (f64::from(b.sum_sq_dev) / 64.0).sqrt();
            let mc_mae = f64::from(m.sad) / 64.0;
            let mv_mag = (f64::from(m.mv.x).powi(2) + f64::from(m.mv.y).powi(2)).sqrt();
            let good = if sigma < 4.0 {
                if mc_mae <= 1.0 { 1.0 } else { 0.0 }
            } else {
                (1.0 - mc_mae / sigma).clamp(0.0, 1.0)
            };
            let mc_w = ((good - 0.50) / span).clamp(0.0, 1.0);
            let mv_decay = (-(mv_mag - 1.0).max(0.) / 3.0).exp();
            let wi = mc_w * mv_decay;
            w[i] = wi;
            sum_w += wi;
        }
        let mean_w = sum_w / n as f64;
        // Skip if frame is uniform (all still or all chaotic).
        let mut deltas = vec![0.0_f64; n];
        let mut any = false;
        for i in 0..n {
            let d = -eff_qp * (w[i] - mean_w);
            if d.abs() >= 0.1 {
                deltas[i] = d;
                any = true;
            }
        }
        if !any {
            return;
        }
        if self.unit_log2 > CELL_LOG2 {
            pool_units(&mut deltas, self.cols, self.unit_log2);
        }
        let mean_d = deltas.iter().sum::<f64>() / n as f64;
        for i in 0..n {
            let raw = (deltas[i] - mean_d).clamp(-self.max_offset, self.max_offset);
            let centered = self
                .offset_step
                .map_or(raw, |step| (raw / step).round() * step);
            if centered.abs() > 1e-6 {
                self.offset[i] += centered;
                if self.bit_weight[i] == 0. {
                    self.bit_weight[i] = bit_weight(&pre_cells[i]);
                }
            }
        }
        self.nonzero.clear();
        self.nonzero
            .extend((0..self.offset.len()).filter(|&cell| self.offset[cell] != 0.));
        self.active = !self.nonzero.is_empty();
    }

    /// Zero-centered per-block temporal-persistence QP concentration on
    /// non-initial (periodic) key frames (`key && index > 0`): uses each 8x8
    /// cell's frame-to-frame temporal similarity (`1 - sad_prev / (64*sigma)`)
    /// and still run length (`still.runs`) to concentrate intra bits onto
    /// persistent background blocks while coarsening fast-changing blocks at
    /// ~zero net bit change.
    pub fn apply_periodic_key_aq(
        &mut self,
        pre: &PreAnalysis,
        still: &nc_analysis::still::StillMap,
        strength_qp: f64,
        last_camera_motion: f64,
    ) {
        let pre_cells = pre.blocks.cells();
        let runs = still.runs.cells();
        if pre_cells.is_empty() || pre_cells.len() != runs.len() || strength_qp <= 0. {
            return;
        }
        // Require temporal sad_prev to be present (not frame 0).
        if pre_cells[0].sad_prev.is_none() {
            return;
        }
        let cam_decay = (-(last_camera_motion - 1.0).max(0.) / 2.0).exp();
        let eff_qp = strength_qp * cam_decay;
        if eff_qp < 0.25 {
            return;
        }
        let n = pre_cells.len();
        self.cols = pre.blocks.cols();
        if !self.active || self.offset.len() != n {
            self.offset.clear();
            self.offset.resize(n, 0.);
            self.nonzero.clear();
        }
        if self.bit_weight.len() != n {
            self.bit_weight.clear();
            self.bit_weight.resize(n, 0.);
        }
        let mut w = vec![0.0_f64; n];
        let mut sum_w = 0.0_f64;
        let span = 0.35_f64; // 0.50 -> 0.85
        for i in 0..n {
            let b = &pre_cells[i];
            let sigma = (f64::from(b.sum_sq_dev) / 64.0).sqrt();
            let mae = f64::from(b.sad_prev.unwrap_or(0)) / 64.0;
            let good = if sigma < 4.0 {
                if mae <= 1.0 { 1.0 } else { 0.0 }
            } else {
                (1.0 - mae / sigma).clamp(0.0, 1.0)
            };
            let temp_w = ((good - 0.50) / span).clamp(0.0, 1.0);
            let run_w = (f64::from(runs[i]) / 8.0).clamp(0.0, 1.0);
            let wi = temp_w.max(run_w);
            w[i] = wi;
            sum_w += wi;
        }
        let mean_w = sum_w / n as f64;
        // Skip scene cuts (nearly everything changed) or uniform static scenes.
        if !(0.05..=0.95).contains(&mean_w) {
            return;
        }
        let mut deltas = vec![0.0_f64; n];
        let mut any = false;
        for i in 0..n {
            let d = -eff_qp * (w[i] - mean_w);
            if d.abs() >= 0.1 {
                deltas[i] = d;
                any = true;
            }
        }
        if !any {
            return;
        }
        if self.unit_log2 > CELL_LOG2 {
            pool_units(&mut deltas, self.cols, self.unit_log2);
        }
        let mean_d = deltas.iter().sum::<f64>() / n as f64;
        let max_pos = self.max_offset.min(2.0);
        for i in 0..n {
            let raw = (deltas[i] - mean_d).clamp(-self.max_offset, max_pos);
            let centered = self
                .offset_step
                .map_or(raw, |step| (raw / step).round() * step);
            if centered.abs() > 1e-6 {
                self.offset[i] += centered;
                if self.bit_weight[i] == 0. {
                    self.bit_weight[i] = bit_weight(&pre_cells[i]);
                }
            }
        }
        self.nonzero.clear();
        self.nonzero
            .extend((0..self.offset.len()).filter(|&cell| self.offset[cell] != 0.));
        self.active = !self.nonzero.is_empty();
    }

    /// Whether the planned frame has any non-zero offset.
    pub fn active(&self) -> bool {
        self.active
    }

    /// `(expected bits weight, offset in QP units)` of every cell with an
    /// offset: the frame's planned block offsets, for rate control's
    /// budgeting with [`Self::total_weight`]. Bits concentrate in active
    /// cells, so an area weight would let the many cheap flat cells that AQ
    /// refines dominate the estimate.
    pub fn offset_cells(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.nonzero
            .iter()
            .map(|&cell| (f64::from(self.bit_weight[cell]), self.offset[cell]))
    }

    /// Whether the reuse term is the only term (its offsets are planned
    /// sparsely, and the frame's total weight is set from outside).
    pub fn reuse_only(&self) -> bool {
        self.strength == 0. && self.persistence.is_none()
    }

    /// Expected bits weight of the whole frame (every cell's).
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Expected bits weight of `block`: the sum over the cells it covers.
    pub fn block_weight(&self, block: &Block) -> f64 {
        if self.cols == 0 {
            return 0.;
        }
        let rows = self.bit_weight.len() / self.cols;
        let (width, height) = cell_extent(block);
        let (col0, row0) = (block.x >> CELL_LOG2, block.y >> CELL_LOG2);
        (row0..(row0 + height).min(rows))
            .flat_map(|row| (col0..(col0 + width).min(self.cols)).map(move |col| (row, col)))
            .map(|(row, col)| f64::from(self.bit_weight[row * self.cols + col]))
            .sum()
    }

    /// Offset of `block` in QP units: the mean offset of the cells it
    /// covers inside the picture.
    #[inline]
    pub fn offset_qp(&self, block: &Block) -> f64 {
        if self.cols == 0 {
            return 0.;
        }
        let (width, height) = cell_extent(block);
        let (col0, row0) = (block.x >> CELL_LOG2, block.y >> CELL_LOG2);
        // A block inside one quantizer square has the square's pooled
        // offset in every cell: read one (the mean over its cells, below,
        // was a few dozen instructions per leaf on every frame with an
        // offset).
        let side = 1usize << self.unit_log2.saturating_sub(CELL_LOG2);
        if width.max(height) <= side && self.unit_log2 > CELL_LOG2 {
            return self
                .offset
                .get(row0 * self.cols + col0)
                .copied()
                .unwrap_or(0.);
        }
        let rows = self.offset.len() / self.cols;
        let (mut sum, mut count) = (0., 0.);
        for row in row0..(row0 + height).min(rows) {
            for col in col0..(col0 + width).min(self.cols) {
                sum += self.offset[row * self.cols + col];
                count += 1.;
            }
        }
        if count == 0. {
            return 0.;
        }
        sum / count
    }
}

/// Give every cell of the `cols`-wide grid `values` the mean of its aligned
/// `1 << unit_log2` quantizer square (the cells of it inside the picture),
/// so each leaf and each cell of rate control's estimate carries the offset
/// the codec can code there.
fn pool_units(values: &mut [f64], cols: usize, unit_log2: u8) {
    let rows = values.len() / cols;
    let side = 1usize << (unit_log2 - CELL_LOG2);
    for row0 in (0..rows).step_by(side) {
        // A band of all-zero offsets pools to itself: the reuse term's few
        // refined cells leave nearly every band so (pooling them all cost
        // ~17 instructions per cell on every frame it refines anything).
        let band = row0 * cols..(row0 + side).min(rows) * cols;
        if values[band].iter().all(|&v| v == 0.) {
            continue;
        }
        for col0 in (0..cols).step_by(side) {
            let (row1, col1) = ((row0 + side).min(rows), (col0 + side).min(cols));
            let index = |row: usize, col: usize| row * cols + col;
            let sum: f64 = (row0..row1)
                .flat_map(|row| (col0..col1).map(move |col| (row, col)))
                .map(|(row, col)| values[index(row, col)])
                .sum();
            let mean = sum / ((row1 - row0) * (col1 - col0)) as f64;
            for row in row0..row1 {
                values[index(row, col0)..index(row, col1)].fill(mean);
            }
        }
    }
}

/// Expected share of the frame's bits of a cell: its residual activity,
/// temporal where the pre-analysis has a previous frame, spatial otherwise.
fn bit_weight(block: &BlockStats) -> f32 {
    1. + (block.sad_prev.map_or(block.sum_sq_dev, u32::from) as f32).sqrt()
}

/// `(width, height)` of `block` in cells (at least one each).
fn cell_extent(block: &Block) -> (usize, usize) {
    let cells = |log2: u8| (1usize << log2 >> CELL_LOG2).max(1);
    (cells(block.log2_width), cells(block.log2_height))
}

/// A QP-unit offset as a native quantizer offset from `quantizer`, within
/// `bounds`, through the codec's step table.
/// The grid step (QP units) that leaves the heuristic terms
/// `max_offsets - RESERVED_REUSE_OFFSETS` distinct non-zero values, evenly
/// either side of zero over `±max_offset`; `None` when the codec codes any
/// offset. Without it, a codec with a few per-frame offsets (VP9's
/// segments) handed them to the first values in coding order and coded the
/// rest at the nearest: the realised offsets missed rate control's plan
/// (VP9 variance AQ 1: key frames 17% over their budget, `J_real` +23%).
fn heuristic_offset_step(max_offset: f32, max_offsets: usize) -> Option<f64> {
    if max_offsets == usize::MAX {
        return None;
    }
    let per_side = max_offsets.saturating_sub(RESERVED_REUSE_OFFSETS) / 2;
    (per_side > 0).then(|| f64::from(max_offset) / per_side as f64)
}

/// A cell's activity: its motion error against the reference where the
/// analysis has one, else its sum of squared deviations.
#[inline]
pub(crate) fn cell_activity(b: &BlockStats) -> f32 {
    b.sad_prev.map_or(b.sum_sq_dev, u32::from) as f32
}

/// The expected-bits weight of a cell of `activity` ([`cell_activity`]):
/// bits grow about with the residual's magnitude, its square root.
#[inline]
pub(crate) fn bits_weight(activity: f32) -> f32 {
    1. + activity.sqrt()
}

pub(crate) fn native_offset(steps: &[f64], bounds: (u32, u32), quantizer: u32, offset: f64) -> i16 {
    if offset == 0. {
        return 0;
    }
    let target = steps[quantizer as usize] * step_ratio(offset);
    (i64::from(nearest_quantizer(steps, bounds, target)) - i64::from(quantizer)) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_analysis::grid::Grid;
    use nc_analysis::preanalysis::BlockStats;

    #[test]
    fn flat_blocks_get_finer_and_busy_blocks_coarser() {
        let mut o = ParamOverrides::default();
        o.set("rate.aq_strength", 1.);
        o.set("persist.reuse", 0.);
        let mut aq = BlockAq::new(&o, 0, usize::MAX).unwrap();
        // Left half flat (variance 4), right half busy (variance 256).
        let cells = (0..4)
            .map(|i| BlockStats {
                sum_sq_dev: if i % 2 == 0 { 4 * 64 } else { 256 * 64 },
                ..BlockStats::default()
            })
            .collect();
        aq.plan(
            &PreAnalysis {
                blocks: Grid::from_cells(2, 2, cells),
                static_fraction: 0.,
                mean_luma: 0.,
            },
            None,
            None,
        );
        let block = |x| Block::square(x, 0, 3);
        assert!((aq.offset_qp(&block(0)) + 3.).abs() < 1e-9);
        assert!((aq.offset_qp(&block(8)) - 3.).abs() < 1e-9);
        let steps: Vec<f64> = (0..52).map(|q| step_ratio(f64::from(q))).collect();
        assert_eq!(native_offset(&steps, (0, 51), 30, -3.), -3);
        assert_eq!(native_offset(&steps, (0, 51), 50, 3.), 1);
        let mut off = ParamOverrides::default();
        off.set("persist.reuse", 0.);
        assert!(BlockAq::new(&off, 0, usize::MAX).is_none());
    }

    #[test]
    fn few_offset_codecs_get_offsets_on_a_grid_that_fits() {
        let mut o = ParamOverrides::default();
        o.set("rate.aq_strength", 1.);
        o.set("persist.reuse", 0.);
        // Seven offsets (VP9's segments): three a side of zero, 8/3 apart.
        let mut aq = BlockAq::new(&o, 0, 7).unwrap();
        let cells = (0..16)
            .map(|i| BlockStats {
                sum_sq_dev: (1u32 << i) * 64,
                ..BlockStats::default()
            })
            .collect();
        aq.plan(
            &PreAnalysis {
                blocks: Grid::from_cells(16, 1, cells),
                static_fraction: 0.,
                mean_luma: 0.,
            },
            None,
            None,
        );
        let step = 8. / 3.;
        let mut levels: Vec<i64> = (0..16)
            .map(|x| aq.offset_qp(&Block::square(x * 8, 0, 3)) / step)
            .inspect(|level| assert!((level - level.round()).abs() < 1e-9))
            .map(|level| level.round() as i64)
            .collect();
        levels.dedup();
        assert_eq!(levels, [-3, -2, -1, 0, 1, 2, 3]);
        assert_eq!(heuristic_offset_step(8., usize::MAX), None);
    }

    #[test]
    fn leaves_share_the_offset_of_their_quantizer_square() {
        let mut o = ParamOverrides::default();
        o.set("rate.aq_strength", 1.);
        o.set("persist.reuse", 0.);
        let mut aq = BlockAq::new(&o, 4, usize::MAX).unwrap();
        // One 16x16 square: a flat and a busy column; a busy square beside it.
        let cells = (0..8)
            .map(|i| BlockStats {
                sum_sq_dev: if i % 4 == 0 { 4 * 64 } else { 256 * 64 },
                ..BlockStats::default()
            })
            .collect();
        aq.plan(
            &PreAnalysis {
                blocks: Grid::from_cells(4, 2, cells),
                static_fraction: 0.,
                mean_luma: 0.,
            },
            None,
            None,
        );
        let leaf = |x, y| aq.offset_qp(&Block::square(x, y, 3));
        let square = aq.offset_qp(&Block::square(0, 0, 4));
        for (x, y) in [(0, 0), (8, 0), (0, 8), (8, 8)] {
            assert!((leaf(x, y) - square).abs() < 1e-9);
        }
        assert!(leaf(0, 0) < leaf(16, 0));
    }

    #[test]
    fn r2_clears_stale_inactive_offsets_before_adding_deltas() {
        let mut o = ParamOverrides::default();
        o.set("rate.aq_strength", 0.);
        o.set("persist.reuse", 1.);
        let mut aq = BlockAq::new(&o, 3, usize::MAX).unwrap();
        let pre = PreAnalysis {
            blocks: Grid::from_cells(
                2,
                1,
                vec![
                    BlockStats {
                        sum_sq_dev: 100 * 64,
                        ..BlockStats::default()
                    },
                    BlockStats {
                        sum_sq_dev: 100 * 64,
                        ..BlockStats::default()
                    },
                ],
            ),
            static_fraction: 1.,
            mean_luma: 128.,
        };
        // Frame 1: reuse sets a non-zero offset on cell 0.
        aq.plan(&pre, None, Some(&[-8.0, 0.0]));
        assert!(aq.active());
        assert!((aq.offset_qp(&Block::square(0, 0, 3)) + 8.0).abs() < 1e-6);
        // Frame 2: no reuse offsets -> AQ becomes inactive, leaving stale array behind.
        aq.plan(&pre, None, None);
        assert!(!aq.active());
        // R2 adds zero-centered deltas on frame 2; stale -8.0 must not reappear.
        let mut analyzer = nc_analysis::Analyzer::new(16, 8, nc_dsp::Dsp::new()).unwrap();
        let pic = nc_common::Picture::new(16, 8, 128).unwrap();
        let _ = analyzer.analyze(&pic);
        let f2 = analyzer.analyze(&pic);
        let motion = f2.motion.as_ref().expect("second frame has motion");
        let mean_qp =
            aq.apply_first_key_mc_refine(&pre, motion, 6.0, 0.5, 0.85, 1.0, 3.0, 1.0, 2.0, 0.0, 0);
        assert!(mean_qp.is_some());
        // Both cells are identical -> zero-centered R2 deltas must be 0, not -8.0!
        assert!(aq.offset_qp(&Block::square(0, 0, 3)).abs() < 1e-6);
        assert!(aq.offset_qp(&Block::square(8, 0, 3)).abs() < 1e-6);
    }

    #[test]
    fn base_mc_concentration_respects_camera_motion_and_offset_step() {
        let mut o = ParamOverrides::default();
        o.set("rate.aq_strength", 0.);
        o.set("rate.aq_max_offset", 6.);
        o.set("persist.reuse", 1.);
        let mut aq = BlockAq::new(&o, 3, 8).unwrap(); // 8-segment grid (like VP9)
        let mut analyzer = nc_analysis::Analyzer::new(128, 64, nc_dsp::Dsp::new()).unwrap();
        let mut p0 = nc_common::Picture::new(128, 64, 128).unwrap();
        let mut p1 = nc_common::Picture::new(128, 64, 128).unwrap();
        // Left 64x64 SB: static smooth gradient (high MC confidence, sad == 0).
        // Right 64x64 SB: scrambled texture in frame 1 (low MC confidence).
        for y in 0..64 {
            for x in 0..128 {
                let v0 = (60 + (x % 64) * 2 + y) as u8;
                p0.plane_mut(nc_common::PlaneIndex::Y).view_mut().row_mut(y)[x] = v0;
                p1.plane_mut(nc_common::PlaneIndex::Y).view_mut().row_mut(y)[x] = if x < 64 {
                    v0
                } else {
                    ((x * 37 + y * 59) % 200 + 20) as u8
                };
            }
        }
        let _ = analyzer.analyze(&p0);
        let f1 = analyzer.analyze(&p1);
        let motion = f1.motion.as_ref().unwrap();
        // High camera motion shuts it off cleanly.
        aq.apply_base_mc_concentration(&f1.pre, motion, 6.0, 10.0);
        assert!(!aq.active());
        // Still camera concentrates finer QP on left (high MC) vs right (low MC).
        aq.apply_base_mc_concentration(&f1.pre, motion, 6.0, 0.0);
        assert!(aq.offset_qp(&Block::square(0, 0, 3)) < aq.offset_qp(&Block::square(80, 0, 3)));
    }

    #[test]
    fn periodic_key_aq_concentrates_on_persistent_blocks_and_caps_coarsening() {
        let mut o = ParamOverrides::default();
        o.set("rate.aq_strength", 0.);
        o.set("rate.aq_max_offset", 6.);
        o.set("persist.reuse", 1.);
        let mut aq = BlockAq::new(&o, 3, usize::MAX).unwrap();
        let mut analyzer = nc_analysis::Analyzer::new(128, 64, nc_dsp::Dsp::new()).unwrap();
        let mut p0 = nc_common::Picture::new(128, 64, 128).unwrap();
        let mut p1 = nc_common::Picture::new(128, 64, 128).unwrap();
        for y in 0..64 {
            for x in 0..128 {
                let v0 = (60 + (x % 64) * 2 + y) as u8;
                p0.plane_mut(nc_common::PlaneIndex::Y).view_mut().row_mut(y)[x] = v0;
                p1.plane_mut(nc_common::PlaneIndex::Y).view_mut().row_mut(y)[x] = if x < 64 {
                    v0
                } else {
                    ((x * 37 + y * 59) % 200 + 20) as u8
                };
            }
        }
        let _ = analyzer.analyze(&p0);
        let f1 = analyzer.analyze(&p1);
        // High camera motion shuts it off cleanly.
        aq.apply_periodic_key_aq(&f1.pre, &f1.still, 4.0, 10.0);
        assert!(!aq.active());
        // Still camera sharpens persistent left half and caps right-half coarsening <= +2.0.
        aq.apply_periodic_key_aq(&f1.pre, &f1.still, 4.0, 0.0);
        let left = aq.offset_qp(&Block::square(0, 0, 3));
        let right = aq.offset_qp(&Block::square(80, 0, 3));
        assert!(
            left < 0.0 && right > 0.0 && right <= 2.0 + 1e-6,
            "left={left} right={right}"
        );
    }
}
