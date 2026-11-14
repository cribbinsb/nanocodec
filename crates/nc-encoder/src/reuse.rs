//! The temporal-reuse term of block AQ: a still block whose reconstruction
//! has become worse than its long-run quality is refined back, at the price
//! its reuse by later frames warrants.
//!
//! A still block is coded once and then copied, so a poor copy stays poor:
//! a starved key, a scene cut, a loss recovery, an intra refresh, a looping
//! source or a rate dip all leave still content worse than the stream can
//! afford, and nothing in ordinary mode decision notices (the residual of a
//! copy sits under the frame quantizer's dead zone). This term tracks the
//! reconstruction error of the 8x8 cells and makes the long-run quality a
//! fixed point the encoder returns to:
//!
//! - **Measurement.** A key measures every cell. On inter frames the
//!   engine marks every block it coded *dirty*, unless it copied the frame
//!   before at the zero vector (a static skip, or a block its backend
//!   reports as a copy); a clean copy keeps its error. Each inter frame
//!   tracks one row of cells in [`MEASURE_PERIOD`]: a cell whose source
//!   moved forgets its history, a still cell that is dirty, unknown or
//!   moved since its measurement is measured. Refined cells are measured
//!   in the frame that refines them. Measuring every row with a still cell
//!   on every frame cost H.264 `h264-faster` ~10% of its instructions and
//!   ~8% of its Jetson cycles; now ~3% (`docs/handoff/rate-control.md`).
//!
//! - **Target.** A cell's long-run quality is the better of what it had
//!   before a coding made it markedly worse (`restore`: the error before a
//!   jump beyond the measurement spread, kept until the cell is back or its
//!   source moves) and what coding it at the inter level would deliver: the
//!   error of its last coding scaled by the squared ratio of the steps
//!   (distortion grows with the squared step). The first restores content
//!   a later frame degraded; the second lifts content that never was coded
//!   well (a starved first key). Slow drift is not a jump and is left
//!   alone: refining it on `crowd_pp22_027` re-refined a fifth of the still
//!   cells every frame (H.264 600 kbps: -0.55 dB at equal rate).
//! - **Gate.** An inter frame refines a still cell when it is worse than
//!   its target beyond the measurement spread and its source's own
//!   frame-to-frame change (sensor noise: a copy holds an earlier frame's),
//!   when its price step can still lower its error, and at most once per
//!   doubling of its reuse. Only the degraded cells get an offset (block AQ
//!   averages them over the codec's quantizer square), so the term costs
//!   nothing where nothing degraded.
//! - **Price.** A block reused by `R` later frames carries its distortion
//!   into each of them while its bits are paid once: minimising
//!   `D·(1 + R) + λ·bits` is the frame's problem at `λ / (1 + R)`, and since
//!   λ grows with the squared step in every codec, its step may be the
//!   frame's over `sqrt(1 + R)` ([`REUSE_QP_PER_OCTAVE`] per octave of
//!   `1 + R`; x264's mb-tree and libaom's TPL derive the same weight from a
//!   lookahead). `R` is measured: a block still for `r` frames is expected
//!   to stay still `r` more (the only scale-free estimate from a run
//!   length), never past the next scheduled key. The step never goes below
//!   the cell's noise, which the next frame replaces.
//! - **Keys** code every block again: a still cell gets the step that keeps
//!   it no worse than it is (or its target, if better), no finer than its
//!   price.
//!
//! Measured: `crates/nc-cli/tests/static_recovery.rs`; the looped waterfall
//! in `docs/handoff/rate-control.md`.

use nc_analysis::preanalysis::{BlockStats, PreAnalysis};
use nc_analysis::still::StillMap;
use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::Dsp;

use crate::qp_cadence::{QP_UNITS_PER_OCTAVE, step_ratio};

/// QP units finer per octave of `1 + R` (reuse count): the step scales with
/// `(1 + R)^(-1/2)` because λ is proportional to the squared step.
const REUSE_QP_PER_OCTAVE: f64 = QP_UNITS_PER_OCTAVE / 2.;
/// log2 of the side of a measurement cell (the pre-analysis 8x8 grid).
const CELL_LOG2: u8 = 3;
/// Side of a measurement cell.
const CELL: usize = 1 << CELL_LOG2;
/// Samples per cell.
const CELL_SAMPLES: f64 = (CELL * CELL) as f64;
/// Mean squared error of a uniform quantiser per squared step (`Δ²/12`):
/// how fine a step stays above a cell's noise.
const UNIFORM_NOISE_PER_SQUARED_STEP: f64 = 1. / 12.;
/// Mean squared value per squared mean absolute value of a zero-mean
/// Gaussian (`π/2`): turns a cell's frame-to-frame SAD into the mean
/// squared change of its source.
const GAUSSIAN_SQUARE_PER_SQUARED_ABS: f64 = std::f64::consts::FRAC_PI_2;
/// Spread of a cell's error measurement: three standard deviations of a
/// mean squared error over a cell's 64 samples (a Gaussian error's square
/// has relative deviation `sqrt(2)`, so the mean of 64 has `sqrt(2/64)`).
/// Copies of an unchanged reconstruction fluctuate within it with the
/// source's noise; a degraded re-coding (the looped waterfall's periodic
/// key: 3.5x) is far outside it.
const SPREAD: f64 = 1. + TOLERANCE_SIGMAS * std::f64::consts::SQRT_2 / CELL as f64;
/// Standard deviations of [`SPREAD`].
const TOLERANCE_SIGMAS: f64 = 3.;
/// How much looser [`ReuseTerm::plan`]'s prefilter is than the test it
/// guards: it computes the level's error in another order, which may round
/// differently in the last bits.
const PREFILTER_MARGIN: f64 = 1. - 1e-9;
/// Rounds [`degrade_below`] up past the f32 it is kept in (relative
/// rounding 2^-24).
const PREFILTER_ROUNDING: f64 = 1. + 1e-6;
/// [`ReuseTerm::flags`]: the cell's error is known.
const KNOWN: u8 = 1;
/// [`ReuseTerm::flags`]: the cell is owed an earlier error or has a
/// refinement attempt on record.
const OWES: u8 = 2;
/// [`ReuseTerm::flags`]: the cell is refined in the frame being coded.
const REFINED: u8 = 4;
/// [`ReuseTerm::flags`]: the cell was coded since its error was measured
/// (the measurement waits for its row's turn, [`MEASURE_PERIOD`]).
const DIRTY: u8 = 8;
/// An inter frame tracks one row of cells in this many (refined cells at
/// once; keys every cell). Noisy still content is re-coded with a small
/// residual every frame, and an error fluctuates around a level: tracking
/// every cell every frame cost ~5% of Jetson h264-faster cycles, and a
/// degradation is now caught, and refined, at most this many frames late.
const MEASURE_PERIOD: usize = 4;
/// Independent noise realisations in a cell's frame-to-frame change (its
/// source's noise variance is half the change).
const NOISE_REALISATIONS: f64 = 2.;

/// Most QP units [`ReuseTerm::hold_still`] codes a held cell coarser than
/// the frame: two octaves of step put any residual of its error deep in the
/// dead zone.
const MAX_HOLD_OFFSET: f64 = 12.;
/// Bits per sample per octave of a residual's error removed (rate-distortion
/// of a Gaussian at high rate: `R = ½·log2(σ²/D)`).
const RATE_PER_OCTAVE_OF_DISTORTION: f64 = 0.5;

/// Bounds of one planned refinement ([`ReuseTerm::plan_refinement`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct RefinementLimits {
    /// Smallest estimated bits worth a refinement.
    pub least: f64,
    /// Estimated bits at which no more cells are added.
    pub most: f64,
    /// Most QP units a cell is coded finer than the frame.
    pub max_offset: f64,
    /// Fewest frames a cell's source has been still.
    pub min_run: u8,
}

/// What the reuse term needs to know about the frame being planned.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReuseFrame<'a> {
    /// The analysis' stillness of this frame (also on key frames).
    pub still: &'a StillMap,
    /// The frame's pre-analysis (each cell's change since the previous
    /// source frame).
    pub pre: &'a PreAnalysis,
    /// Whether every block is coded again (a key frame).
    pub key: bool,
    /// Frames until the next scheduled key frame, which ends every reuse
    /// (`None`: no key is scheduled).
    pub horizon: Option<u64>,
    /// The inter level's quantiser step in sample units (an orthonormal
    /// transform coefficient's step).
    pub sample_step: f64,
}

impl ReuseFrame<'_> {
    /// Later frames expected to reuse a block still for `run` frames.
    fn reuse(&self, run: u8) -> u64 {
        self.horizon
            .map_or(u64::from(run), |h| h.min(u64::from(run)))
    }
}

/// The coding of the frame just coded, for [`ReuseTerm::observe`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct CodedFrame<'a> {
    /// Whether every block was coded again (a key frame).
    pub key: bool,
    /// The frame quantizer's step in sample units.
    pub sample_step: f64,
    /// The frame's pre-analysis (each cell's source change), if the frame
    /// was analysed.
    pub pre: Option<&'a PreAnalysis>,
    /// The frame's stillness (its static runs: 0 where the source moved),
    /// if the frame was analysed.
    pub still: Option<&'a StillMap>,
}

/// The offsets being planned, and which cells have one.
struct PlannedOffsets {
    offsets: Vec<f64>,
    cells: Vec<usize>,
}

impl PlannedOffsets {
    /// Plan `cell` at `offset` (QP units).
    fn set(&mut self, cell: usize, offset: f64) {
        if offset != 0. {
            self.offsets[cell] = offset;
            self.cells.push(cell);
        }
    }
}

/// Per-cell reconstruction history of one stream.
#[derive(Debug)]
pub(crate) struct ReuseTerm {
    /// Whether the codec codes the offsets per block (else one per frame).
    per_block: bool,
    cols: usize,
    /// Frames observed so far: the index of the frame being coded.
    frame: u32,
    /// [`MEASURE_PERIOD`] (1 in tests that follow every frame).
    period: usize,
    /// Mean squared luma error of every cell's reconstruction against its
    /// source when last measured (infinite: unknown); empty before the
    /// first frame.
    error: Vec<f64>,
    /// The frame of every cell's last measurement: its error holds while
    /// its source has been static since (`StillMap::static_runs`).
    measured: Vec<u32>,
    /// The error to restore every cell to: what it had before a coding
    /// made it worse beyond [`SPREAD`]; infinite when nothing is owed.
    restore: Vec<f64>,
    /// The offset of each cell's last refinement since its source last
    /// moved or it was restored (infinite: none). A further refinement at
    /// the same price buys no more, so the next waits until the cell's
    /// reuse, and with it the price, has doubled: without this,
    /// `crowd_pp22_027` H.264 refined a sixth of its still cells every
    /// frame, cells some decision keeps degrading again.
    attempt: Vec<f64>,
    /// Error and sample step of every cell's last coding (key frames and
    /// refinements; other inter frames copy still cells).
    coded_error: Vec<f64>,
    coded_step: Vec<f64>,
    /// The finest sample step every cell was coded at since its source
    /// last moved (infinite: unknown), for [`Self::hold_still`].
    held: Vec<f64>,
    /// The squared inter-level step below which each cell may be degraded
    /// (rounded up; 0: never, an unknown error): the planner's prefilter,
    /// one small array instead of the cells' history.
    degrade_below: Vec<f32>,
    /// [`KNOWN`], [`OWES`], [`REFINED`] and [`DIRTY`] of every cell: what a
    /// pass over the grid needs, without reading the history.
    flags: Vec<u8>,
    /// Offset of every cell in the frame being coded, QP units.
    offsets: Vec<f64>,
    /// The cells of [`Self::offsets`] that are not zero.
    offset_cells: Vec<usize>,
    /// Whether any cell of the frame being coded has an offset.
    any_offset: bool,
    /// Whether the frame being coded predicts from the frame before, so
    /// that a block copied from its reference at the zero vector is as the
    /// term last saw it.
    copies_previous: bool,
    /// The last frame that measured every cell (a key): the plan after it
    /// looks at every row, so that a starved key is refined the frame
    /// after it.
    measured_all: Option<u32>,
    /// Cells `plan` looks at (scratch).
    candidates: Vec<usize>,
    /// Cells `observe` looks at (scratch).
    pending: Vec<usize>,
    /// One run of cells' squared errors (scratch).
    row_scratch: Vec<u32>,
}

impl ReuseTerm {
    /// A term for a codec with (`per_block`) or without block quantizers.
    /// Offsets are per cell: block AQ averages them over the codec's
    /// quantizer squares, so a few degraded cells refine their square only
    /// a little (applying a degraded cell's price to every still cell of
    /// its square refined 10-25x as many AV1 cells as were degraded).
    pub fn new(per_block: bool) -> Self {
        Self {
            per_block,
            cols: 0,
            frame: 0,
            period: MEASURE_PERIOD,
            error: Vec::new(),
            measured: Vec::new(),
            restore: Vec::new(),
            attempt: Vec::new(),
            coded_error: Vec::new(),
            coded_step: Vec::new(),
            held: Vec::new(),
            degrade_below: Vec::new(),
            flags: Vec::new(),
            offsets: Vec::new(),
            offset_cells: Vec::new(),
            any_offset: false,
            copies_previous: false,
            measured_all: None,
            candidates: Vec::new(),
            pending: Vec::new(),
            row_scratch: Vec::new(),
        }
    }

    /// Every cell's offset in the planned frame, in QP units relative to
    /// the inter level (negative is finer), on the pre-analysis grid; empty
    /// when no cell has one.
    pub fn offsets(&self) -> &[f64] {
        if self.any_offset { &self.offsets } else { &[] }
    }

    /// The planned offsets as one offset for the whole frame (their mean),
    /// for codecs without block quantizers.
    pub fn frame_offset(&self) -> f64 {
        if !self.any_offset || self.offsets.is_empty() {
            return 0.;
        }
        self.offsets.iter().sum::<f64>() / self.offsets.len() as f64
    }

    /// Whether `row` of cells takes its turn in the frame being coded.
    fn turn(&self, row: usize) -> bool {
        (row + self.frame as usize).is_multiple_of(self.period)
    }

    /// Plan the offsets of `frame`. Only still cells can carry one: a key,
    /// and the frame after one, look at all of them, other inter frames at
    /// the rows whose turn came last frame (those just tracked), a few
    /// operations per still cell after a branch-free prefilter.
    pub fn plan(&mut self, frame: &ReuseFrame<'_>) {
        let cols = frame.still.blocks.cols();
        let runs = frame.still.runs.cells();
        let pre = frame.pre.blocks.cells();
        if self.offsets.len() == runs.len() {
            for &cell in &self.offset_cells {
                self.offsets[cell] = 0.;
                if let Some(flags) = self.flags.get_mut(cell) {
                    *flags &= !REFINED;
                }
            }
        } else {
            self.offsets.clear();
            self.offsets.resize(runs.len(), 0.);
        }
        self.offset_cells.clear();
        self.any_offset = false;
        let measured = self.cols == cols && self.error.len() == runs.len();
        if !measured || pre.len() != runs.len() || cols == 0 {
            return;
        }
        // What coding the cell at the inter level would deliver.
        let at_level = |cell: usize| {
            self.coded_error[cell] * (frame.sample_step / self.coded_step[cell]).powi(2)
        };
        let target = |cell: usize| self.restore[cell].min(at_level(cell));
        // A copy is worse than its target only beyond the measurements'
        // spread.
        let degraded_from = |cell: usize| target(cell) * SPREAD;
        // A still cell's price: its reuse-weighted offset, never finer than
        // the step whose quantisation noise is its source's own noise.
        let price = |cell: usize, change: f64| {
            let reuse = frame.reuse(runs[cell]);
            let noise_step = step_for_distortion(change / NOISE_REALISATIONS);
            let offset = reuse_offset(reuse as f64).max(offset_to(noise_step, frame.sample_step));
            (reuse > 0 && offset < 0.).then_some(offset)
        };
        // The cells to look at: on a key every still cell; on an inter
        // frame those of the rows tracked last frame that may be degraded
        // at this frame's step ([`degrade_below`], tested in f32 rounded
        // down so that it stays a necessary condition).
        let squared_step = frame.sample_step * frame.sample_step;
        let mut squared_step_f32 = squared_step as f32;
        if f64::from(squared_step_f32) > squared_step {
            squared_step_f32 = squared_step_f32.next_down();
        }
        let mut candidates = std::mem::take(&mut self.candidates);
        candidates.clear();
        let rows = runs.len() / cols;
        // The rows tracked last frame: their turn was at frame - 1.
        let last =
            |row: usize| (row + self.frame as usize + self.period - 1).is_multiple_of(self.period);
        let every_row = frame.key || self.measured_all == Some(self.frame.wrapping_sub(1));
        for row in (0..rows).filter(|&row| every_row || last(row)) {
            let cells = row * cols..(row + 1) * cols;
            let wanted = |(&run, &below): (&u8, &f32)| {
                u8::from(run != 0) & (u8::from(frame.key) | u8::from(squared_step_f32 < below))
            };
            let (runs, below) = (&runs[cells.clone()], &self.degrade_below[cells]);
            // A branch-free test of the whole row first: nearly every row
            // holds no candidate.
            if runs.iter().zip(below).map(wanted).fold(0, |any, w| any | w) == 0 {
                continue;
            }
            for (i, cell) in runs.iter().zip(below).enumerate() {
                if wanted(cell) != 0 && (frame.key || squared_step < f64::from(*cell.1)) {
                    candidates.push(row * cols + i);
                }
            }
        }
        let mut planned = PlannedOffsets {
            offsets: std::mem::take(&mut self.offsets),
            cells: std::mem::take(&mut self.offset_cells),
        };
        for &cell in &candidates {
            let change = source_change(pre[cell].sad_prev);
            if frame.key {
                let Some(price) = price(cell, change) else {
                    continue;
                };
                // Keep the cell no worse than it is (or at its target), at
                // no more than its price: the step of its last coding
                // scaled to that error.
                let keep = self.error[cell].min(target(cell));
                let keep_step = self.coded_step[cell]
                    * (keep / self.coded_error[cell].max(f64::MIN_POSITIVE)).sqrt();
                planned.set(
                    cell,
                    price.max(offset_to(keep_step, frame.sample_step).min(0.)),
                );
                continue;
            }
            // Refine a degraded cell, and only where its price step can
            // still lower its error (a residual whose error is below that
            // step's quantisation noise codes to nothing). An unknown or
            // dirty cell is skipped: its error is not its reconstruction's.
            let error = self.error[cell];
            let current = self.flags[cell] & (KNOWN | DIRTY) == KNOWN;
            if !current || error <= degraded_from(cell) + change {
                continue;
            }
            let Some(price) = price(cell, change) else {
                continue;
            };
            let refinable = error > expected_distortion(frame.sample_step * step_ratio(price));
            let finer = price < self.attempt[cell] - REUSE_QP_PER_OCTAVE;
            if refinable && finer {
                planned.set(cell, price);
            }
        }
        self.any_offset = !planned.cells.is_empty();
        if self.per_block && self.flags.len() == planned.offsets.len() {
            for &cell in &planned.cells {
                self.flags[cell] |= REFINED;
            }
        }
        (self.offsets, self.offset_cells) = (planned.offsets, planned.cells);
        self.candidates = candidates;
    }

    /// Plan a refinement of `frame` (after [`Self::plan`]): still cells
    /// reused for long (`R`: their still run, up to the next key) whose
    /// reconstruction is worse than their price step `λ / (1 + R)` would
    /// deliver, above their source's noise, get that price (at most
    /// `limits.max_offset` QP finer), in order of forward value per
    /// estimated bit (`R` times the error removed, over the bits that
    /// remove it) until the estimated bits reach `limits.most`. The bits
    /// of a cell are the rate of its residual at its step (half a bit per
    /// sample per halving of its error, the high-rate rule), which the loan
    /// scales by what refinements really cost. Returns the estimate, or 0
    /// (and plans nothing more) when it would be below `limits.least`.
    pub fn plan_refinement(&mut self, frame: &ReuseFrame<'_>, limits: RefinementLimits) -> f64 {
        let runs = frame.still.runs.cells();
        let pre = frame.pre.blocks.cells();
        if frame.key || self.error.len() != runs.len() || pre.len() != runs.len() {
            return 0.;
        }
        let mut chosen: Vec<(f64, usize, f64, f64)> = Vec::new();
        for cell in 0..runs.len() {
            let run = runs[cell];
            if run < limits.min_run
                || self.flags[cell] & (KNOWN | DIRTY) != KNOWN
                || self.offsets[cell] != 0.
            {
                continue;
            }
            let reuse = frame.reuse(run) as f64;
            let noise = source_change(pre[cell].sad_prev) / NOISE_REALISATIONS;
            let offset = reuse_offset(reuse)
                .max(offset_to(step_for_distortion(noise), frame.sample_step))
                .max(-limits.max_offset);
            if offset >= 0. || offset >= self.attempt[cell] - REUSE_QP_PER_OCTAVE {
                continue;
            }
            let error = self.error[cell];
            let distortion = expected_distortion(frame.sample_step * step_ratio(offset));
            if error <= (distortion + noise) * SPREAD {
                continue;
            }
            let bits = CELL_SAMPLES * RATE_PER_OCTAVE_OF_DISTORTION * (error / distortion).log2();
            let value = reuse * (error - distortion) * CELL_SAMPLES;
            chosen.push((value / bits, cell, offset, bits));
        }
        chosen.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let mut estimate = 0.;
        let taken = chosen
            .iter()
            .take_while(|&&(_, _, _, bits)| {
                let fits = estimate < limits.most;
                estimate += bits;
                fits
            })
            .count();
        if estimate < limits.least {
            return 0.;
        }
        for &(_, cell, offset, _) in &chosen[..taken] {
            self.offsets[cell] = offset;
            self.offset_cells.push(cell);
            if self.per_block {
                self.flags[cell] |= REFINED;
            }
        }
        self.any_offset = !self.offset_cells.is_empty();
        estimate
    }

    /// Hold the still cells of an ordinary frame of `frame` (after
    /// [`Self::plan`]) at the finest step they were coded at since their
    /// source settled (`held`): a cell still for `min_run` frames whose
    /// held step is coarser than the frame's gets the offset to it, so its
    /// residual (that step's quantisation noise) codes to nothing and it is
    /// copied. Without this a frame whose level fell below the quality a
    /// still picture holds (cheap still frames lower the level) refines the
    /// whole picture unplanned: the bursts planned refinements replace (AV1
    /// `chirla` 93 kbps with the cadence flat: 10-14-budget frames at the
    /// level). A cell whose source moved forgets its held step.
    ///
    /// With `recover`, a cell is held only when its measured error is no
    /// worse than what the frame's step would deliver: a cell left poor (a
    /// starved key, a moved-and-settled block) is coded at the level and
    /// recovers, where the held step alone kept it at its coarse coding.
    pub fn hold_still(&mut self, frame: &ReuseFrame<'_>, min_run: u8, recover: bool) {
        let runs = frame.still.runs.cells();
        if frame.key || self.held.len() != runs.len() {
            return;
        }
        let level = expected_distortion(frame.sample_step) * SPREAD;
        for (cell, &run) in runs.iter().enumerate() {
            if run == 0 {
                self.held[cell] = f64::INFINITY;
                continue;
            }
            let held = self.held[cell];
            let poor = recover
                && (self.flags[cell] & (KNOWN | DIRTY) != KNOWN || self.error[cell] > level);
            if run < min_run
                || self.offsets[cell] != 0.
                || !held.is_finite()
                || held <= frame.sample_step
                || poor
            {
                continue;
            }
            self.offsets[cell] = offset_to(held, frame.sample_step).min(MAX_HOLD_OFFSET);
            self.offset_cells.push(cell);
        }
        self.any_offset = !self.offset_cells.is_empty();
    }

    /// Whether the planned frame is an inter frame predicting from the
    /// frame before (`previous_reference`), so that a block copied from its
    /// reference at the zero vector leaves its cells as the term last saw
    /// them.
    pub fn set_reference(&mut self, previous_reference: bool) {
        self.copies_previous = previous_reference;
    }

    /// The engine coded the `width` x `height` luma block at (`x`, `y`),
    /// as a copy of its reference at the zero vector (`copy`: a static skip,
    /// or a block the backend reports as one) or not. Every cell it changed
    /// is marked dirty, to be measured in its row's turn.
    pub fn coded_block(
        &mut self,
        (x, y): (usize, usize),
        (width, height): (usize, usize),
        (copy, step): (bool, f64),
    ) {
        let cols = self.cols;
        if (copy && self.copies_previous) || cols == 0 {
            return;
        }
        let (col0, col1) = ((x / CELL).min(cols), ((x + width) / CELL).min(cols));
        for row in (y / CELL)..(y + height) / CELL {
            // Rows past the picture (a block over its padding) end it.
            let cells = row * cols + col0..row * cols + col1;
            let Some(flags) = self.flags.get_mut(cells.clone()) else {
                break;
            };
            for flags in flags {
                *flags |= DIRTY;
            }
            if let Some(held) = self.held.get_mut(cells) {
                for held in held {
                    *held = held.min(step);
                }
            }
        }
    }

    /// Track the coded frame (`coded`) over the cells of the pre-analysis
    /// grid of `width` x `height` visible luma samples, `source` and `recon`
    /// being the coded pictures (which cover whole cells, so an edge cell is
    /// measured over its padding).
    pub fn observe(
        &mut self,
        dsp: &Dsp,
        (source, recon): (&Picture, &Picture),
        (width, height): (usize, usize),
        coded: CodedFrame<'_>,
    ) {
        let (source, recon) = (
            source.plane(PlaneIndex::Y).view(),
            recon.plane(PlaneIndex::Y).view(),
        );
        let (cols, rows) = (width.div_ceil(CELL), height.div_ceil(CELL));
        let cells = cols * rows;
        let fresh = self.cols != cols || self.restore.len() != cells;
        self.cols = cols;
        let pre = coded
            .pre
            .map(|pre| pre.blocks.cells())
            .filter(|pre| pre.len() == cells);
        let runs = coded
            .still
            .map(|still| (still.static_runs.cells(), still.runs.cells()))
            .filter(|(runs, _)| runs.len() == cells);
        if let Some((pre, runs)) = pre.zip(runs).filter(|_| !fresh && !coded.key) {
            self.observe_inter(dsp, (&source, &recon), (pre, runs), coded);
        } else {
            self.observe_all(dsp, (&source, &recon), rows, pre, coded, fresh);
            self.measured_all = Some(self.frame);
        }
        self.frame = self.frame.wrapping_add(1);
        self.copies_previous = false;
    }

    /// Measure every cell (a key, the first frame, or a frame without
    /// analysis).
    fn observe_all(
        &mut self,
        dsp: &Dsp,
        (source, recon): (&PlaneView<'_>, &PlaneView<'_>),
        rows: usize,
        pre: Option<&[BlockStats]>,
        coded: CodedFrame<'_>,
        fresh: bool,
    ) {
        let cols = self.cols;
        let cells = cols * rows;
        if fresh {
            for values in [
                &mut self.error,
                &mut self.restore,
                &mut self.attempt,
                &mut self.coded_error,
            ] {
                values.clear();
                values.resize(cells, f64::INFINITY);
            }
            self.coded_step.clear();
            self.coded_step.resize(cells, coded.sample_step);
            self.held.clear();
            self.held.resize(cells, f64::INFINITY);
            self.measured.clear();
            self.measured.resize(cells, self.frame);
            self.degrade_below.clear();
            self.degrade_below.resize(cells, 0.);
            self.flags.clear();
            self.flags.resize(cells, 0);
        }
        let mut scratch = std::mem::take(&mut self.row_scratch);
        scratch.resize(cols, 0);
        for row in 0..rows {
            let (y, width) = (row * CELL, cols * CELL);
            dsp.sse_8x8_row(
                &cell_row(source, (0, y), width),
                &cell_row(recon, (0, y), width),
                &mut scratch,
            );
            for (col, &sse) in scratch.iter().enumerate() {
                let cell = row * cols + col;
                // The first frame has no history to follow.
                let before = if fresh {
                    f64::INFINITY
                } else {
                    self.error[cell]
                };
                self.error[cell] = f64::from(sse) / CELL_SAMPLES;
                self.measured[cell] = self.frame;
                self.coded(cell, coded);
                let change = pre.map_or(f64::INFINITY, |pre| source_change(pre[cell].sad_prev));
                self.follow(cell, before, change);
                self.refresh(cell);
            }
        }
        self.row_scratch = scratch;
    }

    /// Track the cells of an inter frame in the rows whose turn it is, and
    /// the refined cells. Only still cells are planned, so only they are
    /// measured: a dirty one (coded since its measurement), an unknown one,
    /// one whose source moved since its measurement, and a refined one; a
    /// copy keeps its error; a moved cell forgets it. A degradation is so
    /// caught at most [`MEASURE_PERIOD`] frames late, and each frame reads
    /// a fraction of the grid and pictures.
    fn observe_inter(
        &mut self,
        dsp: &Dsp,
        pictures: (&PlaneView<'_>, &PlaneView<'_>),
        (pre, (static_runs, still_runs)): (&[BlockStats], (&[u8], &[u8])),
        coded: CodedFrame<'_>,
    ) {
        let cols = self.cols;
        let rows = pre.len() / cols;
        let mut pending = std::mem::take(&mut self.pending);
        pending.clear();
        for row in (0..rows).filter(|&row| self.turn(row)) {
            let cells = row * cols..(row + 1) * cols;
            let flags = &self.flags[cells.clone()];
            let runs = static_runs[cells.clone()]
                .iter()
                .zip(&still_runs[cells.clone()]);
            let measured = &self.measured[cells.clone()];
            let frame = self.frame;
            // Nothing to do for a known, clean copy measured since its
            // source settled, for a cell not still (not planned), nor for
            // a moved cell without history.
            let needs = |((&flags, (&run, &still)), &measured): ((&u8, (&u8, &u8)), &u32)| {
                let stale = frame.wrapping_sub(measured) > u32::from(run);
                (run == 0 && flags != 0) || (still != 0 && (flags != KNOWN || stale))
            };
            pending.extend(
                flags
                    .iter()
                    .zip(runs)
                    .zip(measured)
                    .enumerate()
                    .filter(|&(_, cell)| needs(cell))
                    .map(|(i, _)| cells.start + i),
            );
        }
        self.track_cells(dsp, pictures, &pending, (pre, static_runs), coded);
        if self.per_block && self.any_offset {
            // Refined cells are measured in the frame that refines them.
            pending.clear();
            let refined = self
                .offset_cells
                .iter()
                .filter(|&&cell| !self.turn(cell / cols));
            pending.extend(refined);
            self.track_cells(dsp, pictures, &pending, (pre, static_runs), coded);
        }
        self.pending = pending;
    }

    /// [`Self::track`] the ascending `cells`, measured in runs along their
    /// rows (one kernel call each).
    fn track_cells(
        &mut self,
        dsp: &Dsp,
        (source, recon): (&PlaneView<'_>, &PlaneView<'_>),
        cells: &[usize],
        (pre, static_runs): (&[BlockStats], &[u8]),
        coded: CodedFrame<'_>,
    ) {
        let cols = self.cols;
        let mut scratch = std::mem::take(&mut self.row_scratch);
        for run in cells.chunk_by(|&a, &b| b == a + 1 && b % cols != 0) {
            let (first, row) = (run[0] % cols, run[0] / cols);
            let span = (first * CELL, row * CELL);
            scratch.resize(run.len(), 0);
            dsp.sse_8x8_row(
                &cell_row(source, span, run.len() * CELL),
                &cell_row(recon, span, run.len() * CELL),
                &mut scratch,
            );
            for (&cell, &sse) in run.iter().zip(&scratch) {
                self.track(cell, sse, (pre, static_runs[cell]), coded);
            }
        }
        self.row_scratch = scratch;
    }

    /// Track `cell` of an inter frame, whose squared error is now `sse`;
    /// its source has been static for `run` frames.
    fn track(
        &mut self,
        cell: usize,
        sse: u32,
        (pre, run): (&[BlockStats], u8),
        coded: CodedFrame<'_>,
    ) {
        let flags = self.flags[cell];
        if run == 0 || self.frame.wrapping_sub(self.measured[cell]) > u32::from(run) {
            // The source moved since the last measurement: that history
            // means nothing for the new content.
            self.error[cell] = f64::INFINITY;
            self.restore[cell] = f64::INFINITY;
            self.attempt[cell] = f64::INFINITY;
            if run == 0 {
                self.degrade_below[cell] = 0.;
                self.flags[cell] = 0;
                return;
            }
        }
        let before = self.error[cell];
        // A clean copy keeps its error (its source's noise aside).
        if flags & DIRTY != 0 || !before.is_finite() || flags & REFINED != 0 {
            self.error[cell] = f64::from(sse) / CELL_SAMPLES;
            self.measured[cell] = self.frame;
        }
        self.coded(cell, coded);
        self.follow(cell, before, source_change(pre[cell].sad_prev));
        self.refresh(cell);
    }

    /// Refresh `cell`'s [`Self::flags`] (clean, unless it is refined in the
    /// frame being coded) and [`Self::degrade_below`] after its history
    /// changed.
    fn refresh(&mut self, cell: usize) {
        let (error, restore) = (self.error[cell], self.restore[cell]);
        let known = if error.is_finite() { KNOWN } else { 0 };
        let owes = if restore.is_finite() || self.attempt[cell].is_finite() {
            OWES
        } else {
            0
        };
        self.flags[cell] = known | owes;
        self.degrade_below[cell] = degrade_below(
            error,
            restore,
            (self.coded_error[cell], self.coded_step[cell]),
        );
    }

    /// Record the coding of `cell` if it was refined in the frame just
    /// coded (every cell on a key).
    fn coded(&mut self, cell: usize, coded: CodedFrame<'_>) {
        let refined = self.per_block && self.any_offset && self.offsets.len() == self.error.len();
        let offset = if refined { self.offsets[cell] } else { 0. };
        if coded.key || offset < 0. {
            self.coded_error[cell] = self.error[cell];
            self.coded_step[cell] = coded.sample_step * step_ratio(offset);
            if coded.key {
                self.held[cell] = self.coded_step[cell];
            }
        }
        if !coded.key && offset < 0. {
            self.attempt[cell] = offset;
        }
    }

    /// Follow `cell`'s error from `before` to its new one, its source
    /// having changed by `change` (mean squared).
    fn follow(&mut self, cell: usize, before: f64, change: f64) {
        let error = self.error[cell];
        if change > before {
            // The source moved by more than the old reconstruction's
            // error: that history means nothing for the new content.
            self.restore[cell] = f64::INFINITY;
            self.attempt[cell] = f64::INFINITY;
        } else if error > before * SPREAD + change {
            // A coding made the cell markedly worse.
            self.restore[cell] = self.restore[cell].min(before);
        } else if error <= self.restore[cell] * SPREAD {
            self.restore[cell] = f64::INFINITY;
            self.attempt[cell] = f64::INFINITY;
        }
    }
}

/// The `width` samples by one cell of rows from (`x`, `y`) of a coded
/// plane.
fn cell_row<'a>(view: &PlaneView<'a>, (x, y): (usize, usize), width: usize) -> PlaneView<'a> {
    view.sub_view(x, y, width, CELL)
        .expect("the coded picture covers whole cells")
}

/// Offset of a block reused by `reuse` later frames, in QP units.
fn reuse_offset(reuse: f64) -> f64 {
    -REUSE_QP_PER_OCTAVE * (1. + reuse).log2()
}

/// QP-unit offset from step `from` to step `to`.
fn offset_to(to: f64, from: f64) -> f64 {
    QP_UNITS_PER_OCTAVE * (to / from).log2()
}

/// Mean squared frame-to-frame change of a cell's source, from its SAD
/// against the previous source frame (0 without one).
fn source_change(sad_prev: Option<u32>) -> f64 {
    let mean_abs = f64::from(sad_prev.unwrap_or(0)) / CELL_SAMPLES;
    GAUSSIAN_SQUARE_PER_SQUARED_ABS * mean_abs * mean_abs
}

/// The squared inter-level step below which a cell with this history may
/// be degraded in [`ReuseTerm::plan`]: a necessary condition for its test
/// (which adds the source change, never negative), rounded up. Zero for an
/// unknown error (never planned on an inter frame).
fn degrade_below(error: f64, restore: f64, (coded_error, coded_step): (f64, f64)) -> f32 {
    let spread = SPREAD * PREFILTER_MARGIN;
    if !error.is_finite() {
        return 0.;
    }
    // Owed an earlier error, or coded perfectly: degraded at any step.
    if error > restore * spread || coded_error <= 0. {
        return f32::INFINITY;
    }
    // At the inter level the cell would have `coded_error · s² /
    // coded_step²`; it is degraded only beyond that times the spread.
    let below = error * coded_step * coded_step / (coded_error * spread);
    (below * PREFILTER_ROUNDING) as f32
}

/// Mean squared error of a block coded at `sample_step`.
fn expected_distortion(sample_step: f64) -> f64 {
    UNIFORM_NOISE_PER_SQUARED_STEP * sample_step * sample_step
}

/// The step whose expected distortion is `distortion`.
fn step_for_distortion(distortion: f64) -> f64 {
    (distortion / UNIFORM_NOISE_PER_SQUARED_STEP).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_analysis::grid::Grid;
    use nc_analysis::preanalysis::BlockStats;

    fn still(runs: Vec<u8>) -> StillMap {
        let cols = runs.len();
        StillMap {
            blocks: Grid::from_cells(cols, 1, runs.iter().map(|&r| r > 0).collect()),
            static_runs: Grid::from_cells(cols, 1, runs.clone()),
            runs: Grid::from_cells(cols, 1, runs),
            fraction: 0.,
        }
    }

    /// A one-row pre-analysis whose static cells changed by `sad_prev`
    /// each.
    fn pre(sad_prev: &[u32]) -> PreAnalysis {
        let cells = sad_prev
            .iter()
            .map(|&sad| BlockStats {
                sad_prev: Some(sad),
                is_static: true,
                ..BlockStats::default()
            })
            .collect();
        PreAnalysis {
            blocks: Grid::from_cells(sad_prev.len(), 1, cells),
            static_fraction: 0.,
            mean_luma: 0.,
        }
    }

    /// What coding at step 16 delivers in these tests (`Δ²/12`).
    const LEVEL: f64 = 256. / 12.;

    /// A term over `error.len()` cells owed `restore`, that last coded
    /// every cell at step 16 with error [`LEVEL`] and now measures `error`.
    fn term(restore: Vec<f64>, error: Vec<f64>) -> ReuseTerm {
        let cells = error.len();
        ReuseTerm {
            cols: cells,
            frame: 1,
            period: 1,
            measured: vec![0; cells],
            coded_error: vec![LEVEL; cells],
            coded_step: vec![16.; cells],
            degrade_below: error
                .iter()
                .zip(&restore)
                .map(|(&error, &restore)| degrade_below(error, restore, (LEVEL, 16.)))
                .collect(),
            flags: vec![KNOWN; cells],
            attempt: vec![f64::INFINITY; cells],
            error,
            restore,
            ..ReuseTerm::new(true)
        }
    }

    #[test]
    fn reuse_weight_follows_lambda_over_one_plus_reuse() {
        // Reused by 3 later frames: λ/4, so half the step (6 QP units).
        assert!((reuse_offset(3.) + QP_UNITS_PER_OCTAVE).abs() < 1e-12);
    }

    #[test]
    fn inter_frames_refine_still_cells_worse_than_their_target() {
        // At step 16 the level delivers 21.3 (32.6 with the tolerance).
        let map = still(vec![3, 3, 3, 0]);
        let unchanged = pre(&[0; 4]);
        let frame = ReuseFrame {
            still: &map,
            pre: &unchanged,
            key: false,
            horizon: None,
            sample_step: 16.,
        };
        // Owed 5 at 9, worse than the level (50), owed 5 at 5, and owed 5 at
        // 50 but not still.
        let inf = f64::INFINITY;
        let mut t = term(vec![5., inf, 5., 5.], vec![9., 50., 5., 50.]);
        t.plan(&frame);
        let price = reuse_offset(3.);
        assert_eq!(t.offsets(), &[price, price, 0., 0.]);
        assert!((t.frame_offset() - price / 2.).abs() < 1e-12);
        // Source noise no coding removes (a mean change of 3 per sample,
        // 14 squared) keeps the gate shut on the first cell, and bounds how
        // fine the second may go.
        let noisy_source = pre(&[3 * 64; 4]);
        let noisy = ReuseFrame {
            pre: &noisy_source,
            ..frame
        };
        t.plan(&noisy);
        let noise_floor = offset_to(step_for_distortion(source_change(Some(192)) / 2.), 16.);
        assert_eq!(t.offsets(), &[0., noise_floor.max(price), 0., 0.]);
    }

    #[test]
    fn keys_keep_each_still_cells_target_within_its_price() {
        let map = still(vec![200, 200, 0]);
        let unchanged = pre(&[0; 3]);
        let frame = ReuseFrame {
            still: &map,
            pre: &unchanged,
            key: true,
            horizon: Some(3),
            sample_step: 16.,
        };
        // Error 21.3/4 needs half the step to keep (-6, the price at reuse
        // 3); 21.3/64 would need an eighth, more than the price allows.
        let inf = f64::INFINITY;
        let mut t = term(vec![inf; 3], vec![LEVEL / 4., LEVEL / 64., 1.]);
        t.plan(&frame);
        let offsets = t.offsets();
        assert!((offsets[0] + QP_UNITS_PER_OCTAVE).abs() < 1e-9);
        assert!((offsets[1] + QP_UNITS_PER_OCTAVE).abs() < 1e-9);
        assert_eq!(offsets[2], 0.);
    }

    /// One inter frame of a one-row term over a zero source: the
    /// reconstruction holds `values`, `coded` cells were coded (the others
    /// copied) and `runs` are the cells' static runs; every row takes its
    /// turn. Returns the errors.
    fn code(
        t: &mut ReuseTerm,
        values: &[u8],
        (coded, runs): (&[bool], &[u8]),
        sad: u32,
    ) -> Vec<f64> {
        let width = CELL * values.len();
        let source = Picture::new(width, CELL, 0).unwrap();
        let mut recon = Picture::new(width, CELL, 0).unwrap();
        let mut plane = recon.plane_mut(PlaneIndex::Y).view_mut();
        for y in 0..CELL {
            for (cell, &value) in values.iter().enumerate() {
                plane.row_mut(y)[cell * CELL..(cell + 1) * CELL].fill(value);
            }
        }
        t.set_reference(true);
        for (cell, &coded) in coded.iter().enumerate() {
            t.coded_block((cell * CELL, 0), (CELL, CELL), (!coded, 1.));
        }
        let analysis = pre(&vec![sad; values.len()]);
        let map = still(runs.to_vec());
        let coded = CodedFrame {
            key: false,
            sample_step: 16.,
            pre: Some(&analysis),
            still: Some(&map),
        };
        t.observe(&Dsp::new(), (&source, &recon), (width, CELL), coded);
        t.error.clone()
    }

    fn tracker() -> ReuseTerm {
        ReuseTerm {
            period: 1,
            ..ReuseTerm::new(true)
        }
    }

    #[test]
    fn a_jump_is_owed_until_restored_or_the_source_moves() {
        let mut t = tracker();
        let mut restore = |value: u8, sad: u32| {
            code(&mut t, &[value], (&[true], &[9]), sad);
            t.restore[0]
        };
        assert_eq!(restore(1, 0), f64::INFINITY);
        // Error 1 -> 1.21 is within the spread, 1 -> 9 a jump...
        assert_eq!(restore(1, 0), f64::INFINITY);
        assert_eq!(restore(3, 0), 1.);
        // ...owed while it lasts, cleared once restored...
        assert_eq!(restore(3, 0), 1.);
        assert_eq!(restore(1, 0), f64::INFINITY);
        // ...or once the source moves by more than the old error (a mean
        // change of 4 per sample, 25 squared).
        assert_eq!(restore(3, 0), 1.);
        assert_eq!(restore(3, 4 * 64), f64::INFINITY);
    }

    #[test]
    fn copies_keep_their_error_and_moved_cells_forget_it() {
        let mut t = tracker();
        let inf = f64::INFINITY;
        assert_eq!(code(&mut t, &[1, 1], (&[true, true], &[9, 9]), 0), [1., 1.]);
        // A copy keeps its error; a moved cell has none.
        assert_eq!(
            code(&mut t, &[2, 2], (&[false, true], &[9, 0]), 0),
            [1., inf]
        );
        // Coded cells are measured, also one without an error.
        assert_eq!(code(&mut t, &[2, 2], (&[true, true], &[9, 9]), 0), [4., 4.]);
        assert_eq!(
            code(&mut t, &[3, 3], (&[false, true], &[9, 9]), 0),
            [4., 9.]
        );
        // A copy whose source moved and settled since its measurement is
        // measured again.
        assert_eq!(
            code(&mut t, &[3, 3], (&[false, false], &[1, 9]), 0),
            [9., 9.]
        );
    }

    #[test]
    fn rows_are_tracked_in_turn() {
        let mut t = ReuseTerm::new(true);
        code(&mut t, &[1], (&[true], &[9]), 0);
        // Coded again: measured only in its row's turn, every
        // MEASURE_PERIOD frames.
        let errors: Vec<f64> = (0..MEASURE_PERIOD)
            .map(|_| code(&mut t, &[2], (&[true], &[9]), 0)[0])
            .collect();
        let mut want = vec![1.; MEASURE_PERIOD];
        want[MEASURE_PERIOD - 1] = 4.;
        assert_eq!(errors, want);
    }
}
