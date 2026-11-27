//! In-frame rate control: watch a frame's bits superblock by superblock
//! while it is coded, so a frame that would pass the buffer's re-code
//! ceiling either lands inside it in one coding or is abandoned early.
//!
//! Under the buffer bound (`rate.recode_overflow`, docs/handoff/rate-control.md
//! "Buffer as a hard bound") a frame past its room is coded again, coarser
//! (`RateController::recode_quantizer`): static refinement bursts can cost
//! 30-100x their prediction, and each re-code is a whole further coding of
//! the frame (1.7-7.6 % of encode time at the fast and medium rungs,
//! 4-17 % on drone and static clips; census 2026-11-23 in the handoff).
//!
//! The frame's bits are predicted per superblock before coding (each
//! superblock's share from the analysis' complexity, blended with the last
//! frame's actual shares), and after each superblock the bits the entropy
//! coder has written (`CodecBackend::coded_bits`) are compared with that
//! prediction. Four uses, each off by default:
//!
//! - **Early re-code** (`rate.recode_early`): when the frame has already
//!   spent its room, or after `rate.recode_early_share` of it projects past
//!   `rate.recode_early_margin` times its room, the coding stops there and
//!   the frame is re-coded whole at the step the projection implies. The
//!   result is the whole-frame re-code's (evenly coarser); the saving is the
//!   unfinished rest of the discarded coding.
//! - **Reactive steering** (`rate.in_frame`): when the projected total
//!   passes `rate.in_frame_aim` of the room, the superblocks after it are
//!   coded coarser, by at most `rate.in_frame_step_qp` per superblock and
//!   `rate.in_frame_max_qp` in all. Row steering of this kind was rejected
//!   as a re-code replacement before (docs/analysis/
//!   within-frame-rate-control-2026-11-14.md section 4.2: frames after a key
//!   kept their lower rows up to 12 QP coarser, worst-1 % PSNR-Y
//!   -1.5..-5.5 dB).
//! - **Proactive steering** (`rate.in_frame_proactive`, with `rate.in_frame`):
//!   before the first superblock the frame's size is also predicted from the
//!   last frame of its kind (its bits per unit of complexity in each
//!   superblock, at its step); where the larger prediction passes the aim
//!   the whole frame starts evenly coarser by the offset that fits it, and
//!   superblocks then correct it by at most `rate.in_frame_correct_qp`
//!   either way.
//! - **Wind-back** (`rate.in_frame_windback`, with `rate.in_frame`): a
//!   superblock that takes the projection past that multiple of the aim is
//!   undone (the backend's node trial, `CodecBackend::undo_node_trial`) and
//!   coded again at the raised offset, so the overshoot is caught at the
//!   superblock that caused it rather than in the ones after it.
//!
//! The steering offset rides on the block-quantizer path block AQ uses
//! (H.264 `mb_qp_delta`, HEVC/VVC `cu_qp_delta`, AV1/AV2 `delta_q`; VP9 codes
//! it only on frames with segmentation on), so λ follows it.

use nc_analysis::preanalysis::PreAnalysis;

use nc_analysis::preanalysis::BlockStats;

use crate::aq::{bits_weight, cell_activity};
use crate::params::RateParams;
use crate::qp_cadence::QP_UNITS_PER_OCTAVE;

/// Side of a pre-analysis cell (8x8 luma), log2.
const CELL_LOG2: u8 = 3;

/// Share of the aim a frame's larger prediction (rate control's, or the
/// last frame of its kind's) must reach for the frame to be watched (and
/// steered) at all. Far below it the watch is skipped: its per-superblock
/// reading of the coder cost HEVC fast 1.6 % of its instructions in
/// pricing alone (callgrind, 2026-11-23), and a steered AV1 frame codes a
/// `delta_q` symbol in every superblock (with every frame steered the
/// screens' clips that never re-code lost +0.3..+0.8 % J, `nceval.quick`
/// probe, bodycam and lowlight, 2026-11-23).
const WATCH_ARMING: f64 = 0.5;

/// Fewest watched units a frame needs to be steered: steering acts on the
/// units after the one that shows the overshoot, so a picture of a few
/// superblocks gains nothing, yet still codes block-QP syntax. On the
/// 128x64 starved-key test (`static_recovery`, AV1 veryfast, two units) the
/// syntax alone, with no offset ever moved, coded the starved key 10 %
/// smaller at qindex 177, its re-code stopped a step finer, and the
/// recovery frame after it lost 1.15 dB.
const STEER_MIN_UNITS: usize = 8;

/// Share of the wind-back threshold the projection must reach before the
/// engine opens a node trial around each superblock: a trial on every
/// superblock of every frame cost AV1 +1.6..+6.7 % cycles (`aws_cpu_ab`,
/// fast and medium, both key shares, 2026-11-23), and frames far under
/// their room never wind back.
const WINDBACK_ARMING: f64 = 0.75;

/// log2 of the smallest superblock watched on its own: smaller ones (H.264
/// macroblocks) are watched a row at a time. Per macroblock the watch cost
/// H.264 medium 0.95 % of its instructions (`end_superblock` and `begin`,
/// callgrind, 150 frames of 720p, 2026-11-23).
const WATCH_MIN_LOG2: u8 = 6;

/// Whether superblocks of side `1 << superblock_log2` are watched a row at
/// a time ([`WATCH_MIN_LOG2`]) rather than one by one.
pub(crate) fn watches_rows(superblock_log2: u8) -> bool {
    superblock_log2 < WATCH_MIN_LOG2
}

/// The watched units of a `width` x `height` picture of superblocks of
/// side `1 << superblock_log2`, as `(columns, rows)`: one column of whole
/// rows where [`watches_rows`].
pub(crate) fn watch_grid(width: usize, height: usize, superblock_log2: u8) -> (usize, usize) {
    let side = 1 << superblock_log2;
    let rows = height.div_ceil(side);
    if watches_rows(superblock_log2) {
        (1, rows)
    } else {
        (width.div_ceil(side), rows)
    }
}

/// Key frames and inter frames keep their own shares: a key's bits follow
/// its texture, an inter frame's its motion error.
const KINDS: usize = 2;

/// How a frame is steered (`rate.in_frame*`).
#[derive(Debug, Clone, Copy)]
struct Steer {
    /// Share of the buffer room the projection must stay under.
    aim: f64,
    /// Largest offset, in QP units.
    max_qp: f64,
    /// Largest change of the offset after one superblock, in QP units.
    step_qp: f64,
    /// Bits-versus-step slope that turns a projected overshoot into QP units.
    exponent: f64,
    /// With proactive steering, how far superblocks may move the starting
    /// offset, in QP units; reactive steering moves it within `[0, max_qp]`.
    proactive_correction: Option<f64>,
    /// With wind-back, the multiple of the aim a superblock must take the
    /// projection past to be coded again.
    windback: Option<f64>,
}

/// When a coding is abandoned for an early re-code (`rate.recode_early*`).
#[derive(Debug, Clone, Copy)]
struct EarlyRecode {
    /// Predicted share of the frame coded before a projection may decide.
    share: f64,
    /// Multiple of the room the projection must pass.
    margin: f64,
}

/// The settings of [`InFrameRate`].
#[derive(Debug, Clone, Copy)]
struct Settings {
    steer: Option<Steer>,
    early: Option<EarlyRecode>,
    /// Weight of the last frame's actual shares against the analysis'
    /// complexity shares.
    history: f64,
    /// Whether key frames are watched too (`rate.in_frame_keys`).
    keys: bool,
    /// Share of the room the projection aims under (`rate.in_frame_aim`).
    aim: f64,
    /// Bits-versus-step slope of the predictions (`rate.in_frame_exponent`).
    exponent: f64,
}

/// The last coded frame of a kind: what the next one's prediction reads.
#[derive(Debug, Default)]
struct Past {
    /// Actual bits of each superblock.
    bits: Vec<f64>,
    /// Analysis complexity of each superblock.
    weights: Vec<f64>,
    /// The frame's quantizer step.
    step: f64,
}

/// What a coding of a frame starts from ([`InFrameRate::begin`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameStart {
    pub key: bool,
    /// Whether its blocks may be coded off its quantizer (`FramePlan::block_qp`).
    pub steerable: bool,
    /// Watched units in the frame ([`watch_grid`]; 0: not watched).
    pub units: usize,
    /// Rate control's predicted bits for the coding.
    pub predicted: f64,
    /// Bits the frame may take before it is re-coded.
    pub room: f64,
    /// Whether rate control would grant a re-code of this coding.
    pub recodable: bool,
    /// The frame quantizer's step.
    pub step: f64,
    /// The coder's position before the first superblock.
    pub start: f64,
}

/// What follows a finished superblock ([`InFrameRate::end_superblock`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Verdict {
    Continue,
    /// Undo the superblock and code it again at the raised offset.
    WindBack,
    /// Stop coding and re-code the frame, as if it had taken this many bits.
    Recode(f64),
}

/// One coding's state.
#[derive(Debug)]
struct Coding {
    kind: usize,
    /// Predicted share of the frame's bits by the end of each superblock.
    cumulative: Vec<f64>,
    /// Predicted bits of the frame at the starting offset.
    predicted: f64,
    room: f64,
    recodable: bool,
    step: f64,
    weights: Vec<f64>,
    /// The coder's position when the frame started, and at the end of the
    /// last finished superblock.
    start: f64,
    last: f64,
    /// Actual bits of each finished superblock.
    bits: Vec<f64>,
    /// The offset the next blocks take, and the one the frame started at,
    /// in QP units.
    offset: f64,
    initial: f64,
    /// The superblock last wound back (it is not wound back twice).
    wound: Option<usize>,
    /// Whether the projection has come near the wind-back threshold.
    armed: bool,
    /// How this coding is steered (`None`: watched for an early re-code only).
    steer: Option<Steer>,
}

/// In-frame rate control of one encoder (see the module doc).
#[derive(Debug)]
pub(crate) struct InFrameRate {
    settings: Settings,
    /// The last coded frame of each kind (inter, key).
    past: [Past; KINDS],
    /// The analysis complexity of each superblock of the frame being coded
    /// ([`Self::set_frame_weights`]).
    weights: Vec<f64>,
    coding: Option<Coding>,
}

impl InFrameRate {
    /// The control `params` configure, or `None` when every use is off.
    pub(crate) fn new(params: &RateParams) -> Option<Self> {
        let steer = (params.in_frame >= 0.5).then(|| Steer {
            aim: f64::from(params.in_frame_aim),
            max_qp: f64::from(params.in_frame_max_qp),
            step_qp: f64::from(params.in_frame_step_qp),
            exponent: f64::from(params.in_frame_exponent),
            proactive_correction: (params.in_frame_proactive >= 0.5)
                .then_some(f64::from(params.in_frame_correct_qp)),
            windback: (params.in_frame_windback > 0.)
                .then_some(f64::from(params.in_frame_windback)),
        });
        let early = (params.recode_early >= 0.5).then(|| EarlyRecode {
            share: f64::from(params.recode_early_share),
            margin: f64::from(params.recode_early_margin),
        });
        (steer.is_some() || early.is_some()).then(|| Self {
            settings: Settings {
                steer,
                early,
                history: f64::from(params.in_frame_history),
                keys: params.in_frame_keys >= 0.5,
                aim: f64::from(params.in_frame_aim),
                exponent: f64::from(params.in_frame_exponent),
            },
            past: [Past::default(), Past::default()],
            weights: Vec::new(),
            coding: None,
        })
    }

    /// The frame about to be coded is not watched: its codings, should a
    /// re-code be watched after all, share the frame's bits equally.
    pub(crate) fn clear_frame_weights(&mut self) {
        self.weights.clear();
    }

    /// The frame about to be coded has these superblock complexities
    /// ([`superblock_weights`]), read by every coding of it.
    pub(crate) fn set_frame_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    /// Whether blocks may be coded off the frame quantizer (the backend
    /// needs its block-QP syntax on).
    pub(crate) fn steers(&self) -> bool {
        self.settings.steer.is_some()
    }

    /// Whether a frame (`key` or inter) may be coded off its quantizer, so
    /// its blocks must carry block-QP syntax (`FramePlan::block_qp`).
    pub(crate) fn steers_frame(&self, key: bool) -> bool {
        self.steers() && (!key || self.settings.keys)
    }

    /// Whether a frame (`key` or inter) with rate control's `predicted` bits
    /// and `room`, planned at `step`, is watched (a key only with
    /// `rate.in_frame_keys`): its larger prediction
    /// (rate control's, or the last watched frame of its kind's bits at this
    /// step, [`past_frame_size`]) comes within [`WATCH_ARMING`] of the aim.
    /// It reads no superblock weights, so a frame that is not watched costs
    /// none ([`superblock_weights`] is computed only for watched frames).
    pub(crate) fn watches(&self, key: bool, (predicted, room): (f64, f64), step: f64) -> bool {
        if key && !self.settings.keys || !room.is_finite() {
            return false;
        }
        let past = &self.past[usize::from(key)];
        let from_past = past_frame_size(past, step, self.settings.exponent);
        predicted.max(from_past) > WATCH_ARMING * self.settings.aim * room
    }

    /// Whether a frame [`Self::watches`] of `units` watched units is also
    /// steered (at least [`STEER_MIN_UNITS`]; its blocks then carry
    /// block-QP syntax, `FramePlan::block_qp`).
    pub(crate) fn steers_watched(
        &self,
        key: bool,
        bounds: (f64, f64),
        step: f64,
        units: usize,
    ) -> bool {
        units >= STEER_MIN_UNITS && self.steers_frame(key) && self.watches(key, bounds, step)
    }

    /// The frame about to be coded is not watched ([`Self::watches`]).
    pub(crate) fn skip_frame(&mut self) {
        self.coding = None;
    }

    /// Whether a coding is being watched.
    pub(crate) fn watching(&self) -> bool {
        self.coding.is_some()
    }

    /// Whether the next superblock of the coding under way may be wound
    /// back (the engine then opens a node trial around it): only once the
    /// projection has come near the threshold ([`WINDBACK_ARMING`]).
    pub(crate) fn winds_back(&self) -> bool {
        self.coding
            .as_ref()
            .is_some_and(|c| c.armed && c.steer.is_some_and(|s| s.windback.is_some()))
    }

    /// Start watching a coding of a frame. Without a finite room there is
    /// nothing to watch for, and the frame codes as without this control.
    pub(crate) fn begin(&mut self, frame: &FrameStart) {
        self.coding = None;
        let units = frame.units;
        if self.weights.len() != units {
            self.weights = vec![1.; units];
        }
        let watched = !frame.key || self.settings.keys;
        if !watched
            || units == 0
            || !frame.room.is_finite()
            || frame.room <= 0.
            || frame.predicted <= 0.
        {
            return;
        }
        let kind = usize::from(frame.key);
        let past = &self.past[kind];
        let blend = if past.bits.len() == units {
            self.settings.history
        } else {
            0.
        };
        let past_total: f64 = past.bits.iter().sum();
        let mut total = 0.;
        let cumulative = normalised(&self.weights)
            .iter()
            .enumerate()
            .map(|(unit, &c)| {
                let share = past.bits.get(unit).map_or(0., |b| b / past_total.max(1.));
                total += blend * share + (1. - blend) * c;
                total
            })
            .collect();
        let steer = self
            .settings
            .steer
            .filter(|_| frame.steerable && units >= STEER_MIN_UNITS);
        let (initial, predicted) = match steer {
            Some(steer) if steer.proactive_correction.is_some() => {
                let from_past = past_size(past, &self.weights, frame.step, steer.exponent);
                let predicted = frame.predicted.max(from_past.unwrap_or(0.));
                let fits = steer.aim * frame.room;
                let offset = qp_for_ratio(predicted / fits, steer.exponent).clamp(0., steer.max_qp);
                (offset, predicted * ratio_for_qp(offset, steer.exponent))
            }
            _ => (0., frame.predicted),
        };
        self.coding = Some(Coding {
            kind,
            cumulative,
            predicted,
            room: frame.room,
            recodable: frame.recodable,
            step: frame.step,
            weights: self.weights.clone(),
            start: frame.start,
            last: frame.start,
            bits: Vec::with_capacity(units),
            offset: initial,
            initial,
            wound: None,
            armed: false,
            steer,
        });
    }

    /// The offset, in QP units, the blocks coded next take (0 when idle).
    pub(crate) fn offset_qp(&self) -> f64 {
        self.coding.as_ref().map_or(0., |c| c.offset)
    }

    /// Superblock `unit` (in coding order) is coded and the coder stands at
    /// `coded` bits: decide what follows (a wind-back only where `undoable`),
    /// and with steering move the offset towards what keeps the projected
    /// frame under the aim.
    pub(crate) fn end_superblock(&mut self, unit: usize, coded: f64, undoable: bool) -> Verdict {
        let settings = self.settings;
        let Some(c) = self.coding.as_mut() else {
            return Verdict::Continue;
        };
        c.bits.push((coded - c.last).max(0.));
        c.last = coded;
        let done_share = c.cumulative.get(unit).copied().unwrap_or(1.);
        let spent = coded - c.start;
        if done_share <= 0. || done_share >= 1. {
            return Verdict::Continue;
        }
        // The frame so far against its prediction says how far the rest
        // will miss too; the superblocks coded so far carry their offsets.
        let projected = spent / done_share;
        if let Some(early) = settings.early.filter(|_| c.recodable) {
            let certain = spent > c.room;
            let likely = done_share >= early.share && projected > early.margin * c.room;
            if certain || likely {
                self.coding = None;
                return Verdict::Recode(projected.max(spent));
            }
        }
        let Some(steer) = c.steer else {
            return Verdict::Continue;
        };
        let pace = (spent / (done_share * c.predicted)).max(f64::MIN_POSITIVE);
        let projected_rest = (1. - done_share) * c.predicted * pace;
        let allowed_rest = steer.aim * c.room - spent;
        let wanted = if allowed_rest <= 0. {
            steer.max_qp
        } else {
            qp_for_ratio(projected_rest / allowed_rest, steer.exponent)
        };
        // Reactive: only the overshoot steers, and under the aim the offset
        // relaxes towards the frame's quantizer, never finer. Proactive:
        // within a small band around the starting offset.
        let (low, high) = match steer.proactive_correction {
            Some(band) => (
                (c.initial - band).max(0.),
                (c.initial + band).min(steer.max_qp),
            ),
            None => (0., steer.max_qp),
        };
        if let Some(margin) = steer.windback {
            c.armed = projected > WINDBACK_ARMING * margin * steer.aim * c.room;
        }
        let before = c.offset;
        c.offset = (c.offset + wanted.clamp(-steer.step_qp, steer.step_qp)).clamp(low, high);
        let wind = undoable
            && c.wound != Some(unit)
            && c.offset > before
            && steer
                .windback
                .is_some_and(|margin| projected > margin * steer.aim * c.room);
        if wind {
            c.wound = Some(unit);
            Verdict::WindBack
        } else {
            Verdict::Continue
        }
    }

    /// The last superblock was undone for a wind-back and the coder stands
    /// at `coded` bits again: forget its bits, keep the raised offset.
    pub(crate) fn wound_back(&mut self, coded: f64) {
        if let Some(c) = self.coding.as_mut() {
            c.bits.pop();
            c.last = coded;
        }
    }

    /// The coding is done: keep its bits for the next frame of its kind.
    pub(crate) fn finish(&mut self) {
        let Some(c) = self.coding.take() else { return };
        if c.bits.len() == c.cumulative.len() {
            self.past[c.kind] = Past {
                bits: c.bits,
                weights: c.weights,
                step: c.step,
            };
        }
    }
}

/// The size `past` predicts for a frame of superblock complexity `weights`
/// at `step`: each superblock at the past frame's bits per unit of
/// complexity in it, scaled by the step ratio under `exponent` (`None`
/// when the layouts differ).
fn past_size(past: &Past, weights: &[f64], step: f64, exponent: f64) -> Option<f64> {
    if past.bits.len() != weights.len() || past.step <= 0. || step <= 0. {
        return None;
    }
    let units = past.bits.iter().zip(&past.weights).zip(weights);
    let at_past_step: f64 = units
        .filter(|&((_, &before), _)| before > 0.)
        .map(|((&bits, &before), &now)| bits * now / before)
        .sum();
    Some(at_past_step * (past.step / step).powf(exponent))
}

/// The size `past` predicts for a frame at `step`: its bits, scaled by the
/// step ratio under `exponent` (0 without a past frame).
fn past_frame_size(past: &Past, step: f64, exponent: f64) -> f64 {
    if past.step <= 0. || step <= 0. {
        return 0.;
    }
    past.bits.iter().sum::<f64>() * (past.step / step).powf(exponent)
}

/// QP units that scale a frame's bits down by `ratio` under `exponent`.
fn qp_for_ratio(ratio: f64, exponent: f64) -> f64 {
    QP_UNITS_PER_OCTAVE / exponent * ratio.log2()
}

/// The bits ratio an offset of `qp` units gives under `exponent`.
fn ratio_for_qp(qp: f64, exponent: f64) -> f64 {
    (-qp * exponent / QP_UNITS_PER_OCTAVE).exp2()
}

/// The expected-bits weight of each watched unit (a [`watch_grid`] of
/// superblocks of side `1 << superblock_log2`, `columns` x `rows`, raster
/// order): the sum of its cells' block AQ weights (`aq::bits_weight`) of
/// their motion error, or of their spatial detail in a `key` (coded
/// without reference), or equal weights without an analysis. One pass over
/// the cell rows with no per-cell index arithmetic (with a division and a
/// clamp per cell it cost 0.6 M instructions a 720p frame, 1.1-1.5 % of the
/// encode).
pub(crate) fn superblock_weights(
    pre: Option<&PreAnalysis>,
    (columns, rows): (usize, usize),
    superblock_log2: u8,
    key: bool,
) -> Vec<f64> {
    let units = columns * rows;
    let Some(pre) = pre.filter(|p| p.blocks.cols() > 0) else {
        return vec![1.; units];
    };
    let mut weights = vec![0.; units];
    let shift = superblock_log2.saturating_sub(CELL_LOG2);
    let cells_per_unit = if columns == 1 {
        pre.blocks.cols()
    } else {
        1 << shift
    };
    let activity = |cell| {
        if key {
            cell_activity_spatial(cell)
        } else {
            cell_activity(cell)
        }
    };
    for (cell_row, cells) in pre.blocks.cells().chunks(pre.blocks.cols()).enumerate() {
        let row = (cell_row >> shift).min(rows - 1);
        let units = &mut weights[row * columns..(row + 1) * columns];
        for (unit, chunk) in units.iter_mut().zip(cells.chunks(cells_per_unit)) {
            let sum: f32 = chunk.iter().map(|cell| bits_weight(activity(cell))).sum();
            *unit += f64::from(sum);
        }
    }
    weights
}

/// A key's cell activity: its spatial detail (a key has no reference).
fn cell_activity_spatial(cell: &BlockStats) -> f32 {
    cell.sum_sq_dev as f32
}

/// `weights` scaled to sum to 1 (equal shares when they sum to nothing).
fn normalised(weights: &[f64]) -> Vec<f64> {
    let total: f64 = weights.iter().sum();
    if total > 0. {
        weights.iter().map(|w| w / total).collect()
    } else {
        vec![1. / weights.len() as f64; weights.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ParamOverrides;

    fn control(options: &[(&str, f32)]) -> InFrameRate {
        let mut o = ParamOverrides::default();
        for &(name, value) in options {
            o.set(name, value);
        }
        InFrameRate::new(&RateParams::from_overrides(&o)).unwrap()
    }

    fn frame(weights: &[f64], predicted: f64, room: f64) -> FrameStart {
        FrameStart {
            key: false,
            steerable: true,
            units: weights.len(),
            predicted,
            room,
            recodable: true,
            step: 10.,
            start: 0.,
        }
    }

    #[test]
    fn a_frame_on_its_prediction_keeps_offset_zero() {
        let mut r = control(&[("rate.in_frame", 1.)]);
        r.begin(&frame(&[1.; 4], 1000., 4000.));
        for unit in 0..4 {
            assert_eq!(
                r.end_superblock(unit, 250. * (unit + 1) as f64, true),
                Verdict::Continue
            );
            assert_eq!(r.offset_qp(), 0.);
        }
    }

    #[test]
    fn a_burst_past_its_room_is_coded_coarser_superblock_by_superblock() {
        let mut r = control(&[("rate.in_frame", 1.)]);
        r.begin(&frame(&[1.; 8], 1000., 4000.));
        // Ten times its prediction: the projection passes the room.
        r.end_superblock(0, 1250., true);
        let first = r.offset_qp();
        assert!(first > 0.);
        r.end_superblock(1, 2500., true);
        assert!(r.offset_qp() > first);
        assert!(r.offset_qp() <= f64::from(RateParams::default().in_frame_max_qp));
    }

    #[test]
    fn a_frame_that_has_spent_its_room_is_recoded_at_once() {
        let mut r = control(&[("rate.recode_early", 1.)]);
        assert!(!r.steers());
        r.begin(&frame(&[1.; 8], 1000., 4000.));
        assert_eq!(
            r.end_superblock(0, 4100., true),
            Verdict::Recode(8. * 4100.)
        );
    }

    #[test]
    fn a_projection_decides_only_after_enough_of_the_frame() {
        let mut r = control(&[("rate.recode_early", 1.)]);
        r.begin(&frame(&[1.; 8], 1000., 4000.));
        // 1/8 coded, projecting 2.4 rooms: too early to decide.
        assert_eq!(r.end_superblock(0, 1200., true), Verdict::Continue);
        // 1/4 coded, projecting 2.4 rooms: past the 1.5 margin.
        assert_eq!(
            r.end_superblock(1, 2400., true),
            Verdict::Recode(4. * 2400.)
        );
    }

    #[test]
    fn a_coding_that_cannot_be_recoded_is_never_stopped() {
        let mut r = control(&[("rate.recode_early", 1.)]);
        r.begin(&FrameStart {
            recodable: false,
            ..frame(&[1.; 8], 1000., 4000.)
        });
        assert_eq!(r.end_superblock(0, 1e9, true), Verdict::Continue);
    }

    #[test]
    fn proactive_steering_starts_evenly_coarser_from_the_last_frame() {
        let mut r = control(&[("rate.in_frame", 1.), ("rate.in_frame_proactive", 1.)]);
        // The last frame took 8000 bits at the same step and complexity.
        r.begin(&frame(&[1.; 8], 1000., 1e6));
        for unit in 0..8 {
            r.end_superblock(unit, 1000. * (unit + 1) as f64, true);
        }
        r.finish();
        r.begin(&frame(&[1.; 8], 1000., 4000.));
        let start = r.offset_qp();
        assert!(start > 0.);
        // Superblocks move it at most the correction band.
        r.end_superblock(0, 1e5, true);
        assert!(r.offset_qp() <= start + f64::from(RateParams::default().in_frame_correct_qp));
    }

    #[test]
    fn a_superblock_that_bursts_is_wound_back_once() {
        let mut r = control(&[("rate.in_frame", 1.), ("rate.in_frame_windback", 1.2)]);
        r.begin(&frame(&[1.; 8], 1000., 4000.));
        assert_eq!(r.end_superblock(0, 125., true), Verdict::Continue);
        assert_eq!(r.end_superblock(1, 3000., true), Verdict::WindBack);
        r.wound_back(125.);
        // Coded again, still over: kept this time.
        assert_eq!(r.end_superblock(1, 2500., true), Verdict::Continue);
        assert_eq!(
            r.coding.as_ref().map(|c| c.bits.clone()),
            Some(vec![125., 2375.])
        );
    }

    #[test]
    fn a_picture_of_a_few_superblocks_is_not_steered() {
        let mut r = control(&[("rate.in_frame", 1.)]);
        r.begin(&frame(&[1.; 2], 1000., 4000.));
        r.end_superblock(0, 1e5, true);
        assert_eq!(r.offset_qp(), 0.);
    }

    #[test]
    fn without_a_room_nothing_is_steered() {
        let mut r = control(&[("rate.in_frame", 1.)]);
        r.begin(&frame(&[1.; 8], 1000., f64::INFINITY));
        r.end_superblock(0, 1e9, true);
        assert_eq!(r.offset_qp(), 0.);
    }
}
