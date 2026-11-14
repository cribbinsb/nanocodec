//! Low-delay frame-level rate control: actual packet feedback and a virtual VBV.
//!
//! The target is a long-term ceiling (`rate.debit`, the default): the debt
//! (bits spent over the target line) beyond a key's scheduled loan is repaid,
//! and keys' loans are repaid within `rate.key_repay_seconds`, so the
//! average over windows longer than that horizon lands at or under the
//! target. Single frames and seconds may exceed it (keys, static refinement
//! bursts) within the strict-CBR buffer's re-code ceiling
//! (`rate.recode_overflow`); `rate.window_seconds` bounds short windows too.
//! The guarantee and its measurements: `docs/handoff/rate-control.md`,
//! "Target as a long-term ceiling".
//!
//! No padding, lookahead or re-encoding is performed. Bounds and scene cuts can
//! make a requested rate unattainable; counters expose that fact. With
//! [`RateControlConfig::frame_drop`] the encoder skips inter frames that would
//! overflow the buffer even at the maximum quantizer ([`RateController::should_drop`]).
use crate::layer::LayerPosition;
use crate::params::RateParams;
use crate::qp_cadence::{
    CadenceRole, LayerLadder, MotionHint, QP_UNITS_PER_OCTAVE, QpCadence, nearest_quantizer,
    step_ratio,
};
use crate::quality_cap::QualityFloor;
use crate::rate_window::WindowLedger;
use crate::refinement_gain::RefinementGain;
use crate::refinement_loan::StillRate;
use crate::skip_budget::{self, SkipBudget};
use crate::{Codec, FrameRate};
use std::fmt;

/// Runtime target and low-delay controller tuning; quantizer bounds are actual QP/qindex.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateControlConfig {
    /// Long-term encoded packet bitrate, in bits/second.
    pub target_bps: u64,
    /// Peak drain rate; `None` allows twice the target, with VBV headroom for keys.
    pub peak_bps: Option<u64>,
    /// Virtual decoder buffer duration in milliseconds.
    pub buffer_ms: u32,
    /// Minimum actual QP/qindex; `None` uses the codec minimum.
    pub min_quantizer: Option<u32>,
    /// Maximum actual QP/qindex; `None` uses the codec maximum.
    pub max_quantizer: Option<u32>,
    /// Maximum native quantizer change between inter frames; defaults to4QP/20AV1indices.
    /// Keys use their own budget/model directly within the actual QP bounds.
    /// Live explicit bound changes take priority over this movement limit.
    pub max_step: Option<u32>,
    /// Keyframe budget relative to an inter frame, amortized over the GOP.
    pub keyframe_boost: f64,
    /// EWMA weight of normal packet-complexity feedback.
    pub model_smoothing: f64,
    /// Model error ratio that triggers immediate scene/spike adaptation.
    pub scene_change_ratio: f64,
    /// Bitrate model exponent: bits are approximately proportional to step^-exponent.
    pub rate_exponent: f64,
    /// Initial intra/inter complexity ratio, before both have been observed.
    pub initial_key_complexity: f64,
    /// Bootstrap inter-frame bits/pixel at the configured seed quantizer; codec default if None.
    pub initial_bits_per_pixel: Option<f64>,
    /// Skip (emit nothing for) an inter frame whose predicted size at the
    /// maximum quantizer would overflow the buffer: the only way to hold an
    /// unreachable target. Off by default.
    pub frame_drop: bool,
    /// Measure each frame's temporal noise (`RateControlStats::residual_sigma`, a
    /// rate-log diagnostic that no decision reads). Off by default: its
    /// median over the frame's 8x8 blocks cost about 1% of an H.264 speed
    /// preset's instructions (x86 callgrind, h264-faster 720p).
    pub temporal_noise_stat: bool,
}
/// Default [`RateControlConfig::buffer_ms`].
pub const DEFAULT_BUFFER_MS: u32 = 1000;
impl RateControlConfig {
    /// Construct a low-delay target with a 1 s buffer, the size the browser
    /// panels and every measurement use; all settings remain adjustable.
    /// The earlier 500 ms default had no recorded reason and made CLI and
    /// C API runs differ from measured ones: AV2's inter re-code ceiling
    /// (`rate.recode_overflow_inter` 1.1) re-coded 17-26 of 150 frames on
    /// `crowd_mot20_02` at 600 kbps with 500 ms, none with 1000 ms.
    #[must_use]
    pub const fn new(target_bps: u64) -> Self {
        Self {
            target_bps,
            peak_bps: None,
            buffer_ms: DEFAULT_BUFFER_MS,
            min_quantizer: None,
            max_quantizer: None,
            max_step: None,
            keyframe_boost: 8.,
            model_smoothing: 0.25,
            scene_change_ratio: 3.,
            rate_exponent: 1.1,
            initial_key_complexity: 8.,
            initial_bits_per_pixel: None,
            frame_drop: false,
            temporal_noise_stat: false,
        }
    }
}
/// A codec's rate-model calibration, from [`crate::CodecBackend::rate_model`]:
/// its lossy quantizer range, how far the inter quantizer may move per frame
/// and the first-frame bits guess. These depend on how the codec's quantizer
/// maps to a step, so they are the backend's, not the engine's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateModel {
    /// Smallest lossy quantizer (QP 0; qindex 1, qindex 0 being lossless).
    pub min_quantizer: u32,
    /// Largest quantizer.
    pub max_quantizer: u32,
    /// Default [`RateControlConfig::max_step`], in native quantizer units:
    /// about two thirds of an octave of step (4 QP; 20 qindex of the
    /// AV1/VP9 AC table, 0.44-0.55 octave in its working range).
    pub max_step: u32,
    /// Default [`RateControlConfig::initial_bits_per_pixel`]: the bits per
    /// pixel at the default seed quantizer (`nanocodec --bitrate` without
    /// `--qp`: QP 32, qindex 180) that the bootstrap of a frame without a
    /// model assumes. Codecs whose step at the seed differs need their own
    /// (AV2: `nc-av2` `RATE_MODEL`).
    pub bootstrap_bits_per_pixel: f64,
    /// Scale of the QP cadence's offsets for this backend
    /// (`QpCadence::scaled`): codecs whose bits and quality react more
    /// strongly to a step change between consecutive frames need a gentler
    /// cadence for the same frame-to-frame stability.
    pub cadence_scale: f64,
    /// Default `rate.first_key_lookahead`: the buffers of bits the first key
    /// may take when the start-up lookahead sees camera motion (0 = off).
    pub first_key_lookahead: f64,
    /// Quantiser step in sample units (the step of an orthonormal transform
    /// coefficient) per unit of `CodecBackend::rate_control_quantizer_step`:
    /// what the reuse term's distortion model (`reuse.rs`) needs.
    pub sample_step_scale: f64,
}
impl RateModel {
    /// H.264/HEVC QP scale (step doubles every 6 QP).
    pub const QP: Self = Self {
        min_quantizer: 0,
        max_quantizer: 51,
        max_step: 4,
        bootstrap_bits_per_pixel: 0.08,
        cadence_scale: 1.,
        first_key_lookahead: 0.,
        sample_step_scale: H26X_SAMPLE_STEP_AT_QP0,
    };
    /// AV1 and VP9 qindex: both use the same 8-bit AC step table.
    pub const AC_LOOKUP_QINDEX: Self = Self {
        min_quantizer: 1,
        max_quantizer: 255,
        max_step: 20,
        bootstrap_bits_per_pixel: 0.03,
        cadence_scale: 1.,
        first_key_lookahead: 0.,
        sample_step_scale: 1. / AC_LOOKUP_PER_SAMPLE_STEP,
    };
}
/// H.264/HEVC/VVC quantiser step at QP 0 in sample units (`Qstep(QP) =
/// 0.625 · 2^(QP/6)`, H.264 §8.5.12 scaling: `Qstep(4) = 1`), against the
/// unit-step `2^(QP/6)` of the default rate-control step.
const H26X_SAMPLE_STEP_AT_QP0: f64 = 0.625;
/// AV1 and VP9 AC quantiser values are in eighths of a sample step (their
/// transforms scale coefficients by 8 relative to orthonormal).
const AC_LOOKUP_PER_SAMPLE_STEP: f64 = 8.;
/// The first-key cap of the start-up lookahead in force (0 = off):
/// `rate.first_key_lookahead` when set, else the backend's
/// (`RateModel::first_key_lookahead`); `rate.first_key_lookahead_off`
/// turns it off.
pub(crate) fn first_key_lookahead(params: &RateParams, model: &RateModel) -> f64 {
    let explicit = f64::from(params.first_key_lookahead);
    if params.first_key_lookahead_off >= 0.5 {
        0.
    } else if explicit > 0. {
        explicit
    } else {
        model.first_key_lookahead
    }
}
/// The QP cadence rate control plans with: the configured one, scaled for
/// the backend's `calibration` and flattened by [`spatial_aware`].
fn rate_cadence(params: &RateParams, calibration: &RateModel) -> QpCadence {
    spatial_aware(
        QpCadence::from_params(params).scaled(cadence_scale(params, calibration)),
        params,
    )
}

/// `cadence`, flattened when the cyclic spatial cadence
/// (`rate.cadence_spatial`) gives its boost to blocks instead of frames,
/// or when planned refinements (`rate.refine`) refine still blocks instead
/// of whole base frames (their start-up cadence too with `refine_startup` 0).
pub(crate) fn spatial_aware(cadence: QpCadence, params: &RateParams) -> QpCadence {
    if params.refine >= 0.5 && params.refine_startup < 0.5 {
        cadence.flat()
    } else if params.cadence_spatial >= 0.5 || params.refine >= 0.5 {
        cadence.flattened()
    } else {
        cadence
    }
}

/// The cadence scale in force: `rate.cadence_scale` when set, else the
/// backend's (`RateModel::cadence_scale`).
pub(crate) fn cadence_scale(params: &RateParams, model: &RateModel) -> f64 {
    let explicit = f64::from(params.cadence_scale);
    if explicit > 0. {
        explicit
    } else {
        model.cadence_scale
    }
}
/// Invalid rate-control settings; rejected without changing the active controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateControlError(pub &'static str);
impl fmt::Display for RateControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for RateControlError {}
/// Actual measured packet totals and best-effort VBV state since RC was enabled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateControlStats {
    /// Actual frame quantizer, including keys; live bound changes clamp this choice
    /// immediately before the next frame is encoded.
    pub current_quantizer: u32,
    /// Clamped virtual buffer occupancy,0..1; see overflow counter for excursions.
    pub buffer_fullness: f64,
    /// Average packet bitrate since RC was enabled, not a rolling-window estimate.
    pub actual_bps: f64,
    /// Actual packet bits, without synthetic padding.
    pub total_bits: u64,
    /// Frames encoded with this controller.
    pub frames: u64,
    /// Current configured target bitrate.
    pub target_bps: u64,
    /// Physical VBV overflow events (including buffer shrink on reconfiguration).
    pub buffer_overflows: u64,
    /// Largest actual packet observed, in bits.
    pub max_frame_bits: u64,
    /// Frames exceeding their peak-rate-plus-buffer-headroom allowance.
    pub peak_overflows: u64,
    /// Frames at maximum QP that still exceeded their assigned frame budget.
    pub target_unreachable_frames: u64,
    /// Input frames skipped by [`RateControlConfig::frame_drop`] (not in `frames`).
    pub dropped_frames: u64,
    /// Frames coded a second time, coarser, because the first coding would
    /// have overflowed the buffer (`rate.recode_overflow`).
    pub recoded_frames: u64,
    /// Bits of the last periodic key's planned loan not yet repaid by the
    /// following inter frames: `actual_bps` minus this over the elapsed
    /// time is the rate the controller is converging to, so a rate check
    /// on a stream shorter than the repayment horizon can tell the planned
    /// key burst from a real overshoot.
    pub key_loan_bits: f64,
    /// Smoothed temporal residual noise (luma samples), a diagnostic
    /// (`quality_cap::residual_sigma`); 0 unless
    /// [`RateControlConfig::temporal_noise_stat`] is on.
    pub residual_sigma: f64,
    /// How the latest frame was planned, a rate-log diagnostic.
    pub plan: RatePlanTrace,
}
/// The controller's plan of the latest frame and its state after it: a
/// diagnostic for the rate log (`nanocodec --rate-log`), read by no decision.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RatePlanTrace {
    /// Bits the frame was budgeted, after the debt correction.
    pub budget_bits: f64,
    /// Bits the frame's model predicted at its quantizer.
    pub predicted_bits: f64,
    /// Inter level quantizer (before cadence offsets; the key's own on a key).
    pub level_quantizer: u32,
    /// Cadence offset applied, in QP units (negative is finer).
    pub cadence_offset: f64,
    /// Share of unchanged blocks the frame's analysis saw.
    pub static_fraction: f64,
    /// Debt beyond the scheduled key loan after the frame, in bits.
    pub excess_bits: f64,
    /// The debit term after the frame, in bits.
    pub debit_bits: f64,
    /// The buffer capacity in bits (what the correction divides by).
    pub capacity_bits: f64,
    /// Times the frame was coded: 1, plus its re-codes
    /// (`rate.recode_overflow`, at most `MAX_RECODES`).
    pub codings: u32,
}
#[derive(Debug, Clone, Copy)]
struct Settings {
    config: RateControlConfig,
    min: u32,
    max: u32,
    step: u32,
    peak: f64,
    capacity: f64,
}
impl Settings {
    fn new(config: RateControlConfig, model: &RateModel) -> Result<Self, RateControlError> {
        let (lo, hi, step) = (model.min_quantizer, model.max_quantizer, model.max_step);
        let min = config.min_quantizer.unwrap_or(lo);
        let max = config.max_quantizer.unwrap_or(hi);
        let step = config.max_step.unwrap_or(step);
        if config.target_bps == 0 || config.target_bps > 100_000_000_000 {
            return Err(RateControlError("target bitrate outside1..100Gbps"));
        }
        if config.buffer_ms == 0 || config.buffer_ms > 60_000 {
            return Err(RateControlError("buffer duration outside1..60000ms"));
        }
        if min < lo || max > hi || min > max || step == 0 || step > hi {
            return Err(RateControlError("invalid codec quantizer bounds or step"));
        }
        if config
            .initial_bits_per_pixel
            .is_some_and(|b| !b.is_finite() || b <= 0. || b > 32.)
        {
            return Err(RateControlError("invalid bootstrap bits/pixel"));
        }
        let peak = config
            .peak_bps
            .unwrap_or(config.target_bps.saturating_mul(2));
        if peak < config.target_bps || peak > 100_000_000_000 {
            return Err(RateControlError("peak bitrate must cover target"));
        }
        for (v, low, high) in [
            (config.keyframe_boost, 1., 64.),
            (config.model_smoothing, 0.01, 1.),
            (config.scene_change_ratio, 1.1, 100.),
            (config.rate_exponent, 0.25, 3.),
            (config.initial_key_complexity, 1., 64.),
        ] {
            if !v.is_finite() || v < low || v > high {
                return Err(RateControlError("invalid rate-model tuning"));
            }
        }
        Ok(Self {
            config,
            min,
            max,
            step,
            peak: peak as f64,
            capacity: config.target_bps as f64 * f64::from(config.buffer_ms) / 1000.,
        })
    }
}
/// Same-frame preanalysis hints. No additional pixel pass is needed.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Complexity {
    pub spatial: f64,
    pub temporal: f64,
    /// Share of 8x8 blocks unchanged since the previous source frame.
    pub static_fraction: f64,
    /// Planned block-level quantizer offsets of the frame, as one effective
    /// offset in QP units (`RateController::effective_block_offset`); 0 when
    /// every block is coded at the frame quantizer.
    pub block_qp_offset: f64,
    /// A key frame's whole-frame reuse offset in QP units (`reuse.rs`), for
    /// codecs that cannot give each block its own; 0 otherwise.
    pub key_reuse_offset: f64,
    /// The frame's temporal residual noise (`quality_cap::residual_sigma`),
    /// or `None` without motion analysis (first frame).
    pub residual_sigma: Option<f64>,
    /// The frame's motion against its predecessor, for the cadence's motion
    /// gate (`None` without it).
    pub motion: Option<MotionHint>,
}
impl Complexity {
    pub fn from_preanalysis(pre: Option<&nc_analysis::preanalysis::PreAnalysis>) -> Self {
        pre.map_or(
            Self {
                spatial: 1.,
                temporal: 1.,
                static_fraction: 0.,
                block_qp_offset: 0.,
                key_reuse_offset: 0.,
                residual_sigma: None,
                motion: None,
            },
            |pre| {
                pre.blocks.cells().iter().fold(
                    Self {
                        spatial: 0.,
                        temporal: 0.,
                        static_fraction: f64::from(pre.static_fraction),
                        block_qp_offset: 0.,
                        key_reuse_offset: 0.,
                        residual_sigma: None,
                        motion: None,
                    },
                    |mut sum, b| {
                        sum.spatial += 1. + f64::from(b.sum_sq_dev).sqrt();
                        sum.temporal += 1. + f64::from(b.sad_prev.unwrap_or(0)).sqrt();
                        sum
                    },
                )
            },
        )
    }

    fn value(self, key: bool) -> f64 {
        let n = if key { self.spatial } else { self.temporal };
        if n.is_finite() { n.max(1.) } else { 1. }
    }
}
/// Inter frames between one-quantizer climbs while refinement is
/// exhausted: a quarter of the gain window, so the slope can respond to
/// each climb before the next.
const CAP_STEP_FRAMES: u32 = 8;
/// Weight of a frame's residual noise in its running estimate (as the
/// analyzer smooths its sensor-noise estimate): keys, which have no motion
/// analysis of their own, keep the estimate of the frames before them.
const RESIDUAL_SIGMA_SMOOTHING: f64 = 0.25;
/// Peak camera displacement per frame, in luma samples, from which the
/// start-up lookahead treats the scene as moving
/// ([`RateController::observe_startup_motion`]). At frame 1 of the TRAIN and
/// VAL clips, still cameras (static, crowd) read at most 0.5 samples and
/// panning, driving or handheld cameras 1.3-12; a stopped car reads 0.
const STARTUP_CAMERA_MOTION: f64 = 1.0;
/// Range of the debt correction of a frame budget (`budget`): a buffer of
/// debt quarters it, a buffer of credit doubles it at most.
const MIN_BUDGET_CORRECTION: f64 = 0.25;
const MAX_BUDGET_CORRECTION: f64 = 2.;
/// Coded frames over which the time of held frames (`hold.rs`) is spread
/// ([`RateController::hold_frame`]): the smoothing horizon of the held
/// time per coded frame. A single held frame's interval is spent over about
/// this many frames; a steady repeat (every other frame) reaches its two
/// intervals per coded frame within about as many.
const HELD_TIME_FRAMES: f64 = 8.;
/// Re-codes `rate.recode_overflow` may spend on one frame.
pub(crate) const MAX_RECODES: u32 = 2;
/// Bits-versus-step slope the first re-code of a frame assumes. Static
/// refinement bursts, which cause most overflows, are steep: VP9
/// `static_chirla_s000c3_360p` qindex 177 -> 92 took 74 -> 12075 bytes, an
/// exponent near 4. With the model's 1.1 the re-code came out 26x smaller
/// than its room (H.264 `static_cam22` frame 4: 52.8 kbit, room 14.9 kbit,
/// QP 44 -> 54, 2.0 kbit), wasting the refinement.
pub(crate) const RECODE_FIRST_EXPONENT: f64 = 3.;
/// Range of the slope a second re-code measures between two codings.
const RECODE_EXPONENT_RANGE: (f64, f64) = (1., 4.);
/// Share of the buffer room a re-code aims at, a margin for the slope's error.
const RECODE_FILL: f64 = 0.9;
/// Largest debit term, in buffers: the bias it can cancel (frames coming in
/// a third over their budgets leave the proportional correction a quarter
/// buffer of excess). Unbounded, the excess of a key's start-up seconds
/// wound it up and static scenes then ran 6% under the target for the next
/// 15 s (`long_static_chirla_s024c2` AV1 400 kbps, 5 s horizon).
const DEBIT_MAX_BUFFERS: f64 = 0.25;
/// Time constant of the running share of unchanged blocks that sets each
/// key's repay horizon (`repay_seconds_for_key`): about a second of history,
/// so one frame's flash or pan does not decide a 10 s phase.
const STILL_AVERAGE_SECONDS: f64 = 1.0;
/// Smallest change of the first key's horizon worth re-planning for.
const REPLAN_MIN_SECONDS: f64 = 1e-6;
/// Model slot of key frames; inter roles follow at `INTER_MODEL + role`.
const KEY_MODEL: usize = 0;
const INTER_MODEL: usize = 1;
const MODEL_CLASSES: usize = INTER_MODEL + CadenceRole::COUNT;

/// One frame's share of a buffer it shares with other frames of the same
/// instant (a spatial layer of a temporal unit), planned by the combined
/// buffer model (`spatial.rs`, [`crate::spatial::CombinedCbrBuffer`]) and
/// applied by [`RateController::select`]. [`LayerShare::WHOLE`] is a
/// stream's whole budget: no change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LayerShare {
    /// Multiplier of the inter budget, `(0, 1]`.
    pub budget_scale: f64,
    /// Bits another layer left unspent this unit: added to this inter
    /// frame's budget and to its drain, so the combined drain is unchanged.
    pub reallocated_bits: f64,
    /// Room left in the combined buffer this unit (`INFINITY`: no limit).
    pub room_bits: f64,
    /// Finest quantizer an inter frame may take.
    pub min_inter_quantizer: Option<u32>,
    /// Quantizer range of a key picture anchored to another layer's.
    pub key_anchor: Option<KeyAnchor>,
    /// Drop the key loan before planning (a key picture predicted from
    /// another layer).
    pub clear_key_loan: bool,
}

impl LayerShare {
    /// The whole budget: nothing shared.
    pub const WHOLE: Self = Self {
        budget_scale: 1.0,
        reallocated_bits: 0.0,
        room_bits: f64::INFINITY,
        min_inter_quantizer: None,
        key_anchor: None,
        clear_key_loan: false,
    };
}

/// Quantizer range of a key picture anchored to another layer's quantizer
/// (spatial SVC `S1` key or repair pictures predicted from `S0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KeyAnchor {
    /// Finest quantizer.
    pub min: u32,
    /// Coarsest quantizer (at least `min`).
    pub max: u32,
    /// When the layer has no inter quantizer yet, start its inter level this
    /// far above the key's.
    pub seed_inter_offset: Option<u32>,
}

#[derive(Debug)]
pub(crate) struct RateController {
    settings: Settings,
    /// The codec's calibration (quantizer range, step limit, bootstrap).
    calibration: RateModel,
    /// Nominal frame rate.
    fps: f64,
    /// Rate of the frame being coded: `fps`, or the inverse of its real
    /// capture interval ([`RateController::set_frame_interval`]). Budgets
    /// and buffer drain follow it, so variable-rate input and dropped
    /// camera frames are budgeted by time, not by frame count.
    frame_fps: f64,
    /// Seconds of input coded so far (for the measured bitrate).
    elapsed: f64,
    period: u32,
    steps: Vec<f64>,
    pixels: f64,
    current: u32,
    /// Inter-frame *level* quantizer (before cadence offsets): what smoothing limits.
    last_inter_quantizer: Option<u32>,
    /// The quality ceiling: the finest step any frame may use.
    quality_floor: QualityFloor,
    /// Whether the quality cap raised the current frame's quantizer.
    quality_capped: bool,
    /// Whether finer inter frames still buy quality (`rate.min_refinement_gain`).
    refinement: RefinementGain,
    /// Consecutive inter frames held by `hold_exhausted_refinement`.
    cap_frames: u32,
    seed: u32,
    /// Complexity hint of each model slot's last update (scene detection).
    model_hint: [f64; MODEL_CLASSES],
    /// `rate.scene_hint_ratio`.
    scene_hint_ratio: f64,
    /// Key-frame bits model, then one inter model per cadence role (or per
    /// temporal layer role in a layered structure).
    model: [Option<f64>; MODEL_CLASSES],
    /// Model slots whose next observation replaces them outright: after a
    /// resolution change bits per unit of the complexity hint differ (a
    /// larger picture is smoother per cell), and smoothing the old value in
    /// kept the first 10 P frames after a 360p -> 720p up-switch 3-5 QP
    /// above their settled level.
    relearn: [bool; MODEL_CLASSES],
    cadence: QpCadence,
    /// Per-layer quantizer offsets of a layered structure.
    ladder: LayerLadder,
    /// Budget of the first key relative to an inter frame
    /// (`rate.first_key_boost_scale` times the configured boost), before any
    /// inter level exists to size it by quality ([`RateController::key_step`]).
    first_key_boost: f64,
    /// Planned boost of the key whose loan is being repaid.
    loan_boost: f64,
    /// Frames coded since the last key (the loan repayment phase).
    frames_since_key: u32,
    /// Inter budget of the frames held since the last coded frame
    /// (`hold_frame`).
    held_pending: f64,
    /// Running mean of the held frames' inter budget per coded frame, over
    /// [`HELD_TIME_FRAMES`]: what inter budgets add (`held_budget`).
    held_per_frame: f64,
    /// The held time's bits in the current frame's budget (`held_budget`).
    held_spent: f64,
    /// Effective step ratio of the frame's block-level offsets (planned at
    /// `select`, replaced by the realised one before `feedback`); 1 without.
    block_ratio: f64,
    /// Effective step ratio of the rows in-frame rate control coarsened
    /// (`inframe.rs`; set before `feedback` or a re-code learns from a
    /// coding, reset after `feedback`); 1 without.
    inframe_ratio: f64,
    /// `rate.key_repay_seconds`: the moving-scene repay horizon and the
    /// debit term's horizon.
    key_repay_seconds: f64,
    /// `rate.key_repay_still_seconds` (0: the same as `key_repay_seconds`),
    /// `rate.key_repay_still_from` and `rate.key_repay_still_full`.
    key_repay_still_seconds: f64,
    key_repay_still_from: f64,
    key_repay_still_full: f64,
    /// Running share of unchanged 8x8 blocks over the inter frames: their
    /// mean until [`STILL_AVERAGE_SECONDS`] of them are in, then an average
    /// with that time constant; 0 before the first one.
    still_share: f64,
    /// Inter frames counted into `still_share` (saturating).
    still_frames: u32,
    /// The current key's repay horizon, chosen when the key is planned
    /// (`repay_seconds_for_key`).
    phase_repay_seconds: f64,
    /// The first key's re-plan, while pending: the inter frame after the
    /// first key at which a second of stillness history exists
    /// (`replan_first_key`).
    first_key_replan_at: Option<u32>,
    /// `rate.key_cap_buffer` and `rate.key_cap_gop` (`key_bits_cap`).
    key_cap_buffer: f64,
    key_cap_gop: f64,
    /// `rate.first_key_lookahead`, and the buffer share it caps the first
    /// GOP's key at once the start-up lookahead saw camera motion (0 = no
    /// cap; [`RateController::observe_startup_motion`]).
    first_key_lookahead: f64,
    first_key_cap_buffer: f64,
    /// `rate.recode_overflow`: the buffer fullness re-codes keep frames
    /// under, as a multiple of the capacity; 0 is off.
    recode_ceiling: f64,
    /// `rate.recode_overflow_inter`: the same ceiling for inter frames
    /// (`recode_ceiling` when 0).
    recode_ceiling_inter: f64,
    /// Re-codes of the current frame so far, and the (step, bits) of its
    /// previous coding (`recode_quantizer`).
    recodes_this_frame: u32,
    recode_trial: Option<(f64, f64)>,
    last_key_index: u64,
    /// Model slot of the frame awaiting feedback.
    current_slot: usize,
    fullness: f64,
    /// Outstanding key loan (what the last key borrowed), amortized by the
    /// frames after it. Subtracting it from actual target debt keeps a key's
    /// planned burst from a second, unintended penalty on those frames.
    scheduled_key_debt: f64,
    last_index: u64,
    occupancy: f64,
    /// The same leaky bucket drained at the target rather than the peak
    /// rate: what a strict-CBR receiver holds. Only frame dropping reads it.
    target_occupancy: f64,
    /// That bucket replayed as `nceval`'s strict-CBR check does: not
    /// clipped at the capacity, so its peak is the buffer fullness a
    /// receiver sees (`rate.recode_overflow`).
    strict_occupancy: f64,
    last_hint: f64,
    /// Complexity hint of the most recent inter frame (drop prediction).
    last_inter_hint: Option<f64>,
    last_budget: f64,
    /// `rate.level_step_scale`.
    level_step_scale: f64,
    last_peak: f64,
    last_key: bool,
    last_predicted: f64,
    /// The debit term (`rate.debit`): bits of excess (debt beyond the
    /// scheduled key loan) not yet repaid, accumulated over the repay
    /// horizon and never negative; `None` when off ([`Self::debit`]).
    debit: Option<f64>,
    /// `rate.debit_credit` times the buffer: most undershoot the debt keeps.
    debit_credit: f64,
    /// `rate.debit_margin` with the debit term, else 0 (`uniform_budget`).
    debit_margin: f64,
    /// The short-window bound (`rate.window_seconds`; `None`: off), its
    /// tolerance and whether keys are bounded by it.
    window: Option<WindowLedger>,
    window_tolerance: f64,
    window_keys: bool,
    /// The current frame's share of a combined buffer ([`Self::select`]):
    /// its budget scale and room apply to the frame plan only, its
    /// reallocated bits also to the frame's drain ([`Self::feedback`]).
    share: LayerShare,
    stats: RateControlStats,
    /// The skip-aware budget (`rate.skip_budget`).
    skip_budget: SkipBudget,
    /// The current frame's skip-aware budget scale (1 for keys and when
    /// off): its budget and what its interval drains from the debt.
    frame_skip_scale: f64,
    /// The current frame's budget without the skip-aware scale: what it
    /// would have been planned, debt repayment included.
    unscaled_budget: f64,
    /// The refinement loan, quiet run-up and held key of still scenes.
    still: StillRate,
}
impl RateController {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: RateControlConfig,
        codec: Codec,
        calibration: RateModel,
        rate: FrameRate,
        period: u32,
        seed: u32,
        steps: Vec<f64>,
        pixels: usize,
        params: &RateParams,
        ladder: LayerLadder,
    ) -> Result<Self, RateControlError> {
        let s = Settings::new(config, &calibration)?;
        if steps.len() <= s.max as usize || steps.iter().any(|v| !v.is_finite() || *v <= 0.) {
            return Err(RateControlError("invalid backend quantizer steps"));
        }
        let q = seed.clamp(s.min, s.max);
        Ok(Self {
            settings: s,
            calibration,
            fps: f64::from(rate.num) / f64::from(rate.den),
            frame_fps: f64::from(rate.num) / f64::from(rate.den),
            elapsed: 0.,
            period,
            steps,
            pixels: pixels as f64,
            current: q,
            last_inter_quantizer: None,
            quality_floor: QualityFloor::new(params, codec),
            quality_capped: false,
            refinement: RefinementGain::new(f64::from(params.min_refinement_gain)),
            cap_frames: 0,
            seed: q,
            model: [None; MODEL_CLASSES],
            relearn: [false; MODEL_CLASSES],
            model_hint: [1.; MODEL_CLASSES],
            scene_hint_ratio: f64::from(params.scene_hint_ratio),
            cadence: rate_cadence(params, &calibration),
            first_key_boost: config.keyframe_boost * f64::from(params.first_key_boost_scale),
            loan_boost: config.keyframe_boost * f64::from(params.first_key_boost_scale),
            frames_since_key: 0,
            held_pending: 0.,
            held_per_frame: 0.,
            held_spent: 0.,
            block_ratio: 1.,
            inframe_ratio: 1.,
            key_repay_seconds: f64::from(params.key_repay_seconds),
            key_repay_still_seconds: f64::from(params.key_repay_still_seconds),
            key_repay_still_from: f64::from(params.key_repay_still_from),
            key_repay_still_full: f64::from(params.key_repay_still_full),
            still_share: 0.,
            still_frames: 0,
            phase_repay_seconds: f64::from(params.key_repay_seconds),
            first_key_replan_at: None,
            key_cap_buffer: f64::from(params.key_cap_buffer),
            key_cap_gop: f64::from(params.key_cap_gop),
            first_key_lookahead: first_key_lookahead(params, &calibration),
            first_key_cap_buffer: 0.,
            recode_ceiling: f64::from(params.recode_overflow),
            recode_ceiling_inter: f64::from(params.recode_overflow_inter),
            recodes_this_frame: 0,
            recode_trial: None,
            ladder,
            last_key_index: 0,
            current_slot: KEY_MODEL,
            fullness: s.capacity / 2.,
            scheduled_key_debt: 0.,
            last_index: 0,
            occupancy: 0.,
            target_occupancy: 0.,
            strict_occupancy: 0.,
            last_hint: 1.,
            last_inter_hint: None,
            last_budget: 1.,
            level_step_scale: f64::from(params.level_step_scale),
            last_peak: 1.,
            last_key: false,
            last_predicted: 1.,
            debit: (params.debit >= 0.5).then_some(0.),
            debit_credit: f64::from(params.debit_credit) * s.capacity,
            debit_margin: if params.debit >= 0.5 {
                f64::from(params.debit_margin)
            } else {
                0.
            },
            window: (params.window_seconds > 0.)
                .then(|| WindowLedger::new(f64::from(params.window_seconds))),
            window_tolerance: f64::from(params.window_tolerance),
            window_keys: params.window_keys >= 0.5,
            share: LayerShare::WHOLE,
            stats: RateControlStats {
                current_quantizer: q,
                buffer_fullness: 0.,
                actual_bps: 0.,
                total_bits: 0,
                frames: 0,
                target_bps: config.target_bps,
                buffer_overflows: 0,
                max_frame_bits: 0,
                peak_overflows: 0,
                target_unreachable_frames: 0,
                dropped_frames: 0,
                recoded_frames: 0,
                key_loan_bits: 0.,
                residual_sigma: 0.,
                plan: RatePlanTrace::default(),
            },
            skip_budget: SkipBudget::new(params),
            frame_skip_scale: 1.,
            unscaled_budget: 0.,
            still: StillRate::new(params, f64::from(rate.num) / f64::from(rate.den)),
        })
    }
    /// Apply a changed target, peak or buffer mid-stream (a transport's
    /// new bandwidth estimate), keeping the models.
    ///
    /// The target debt (and the key loan inside it) scales with the target:
    /// it is kept as time at the target rate. The correction reads debt
    /// relative to the buffer (target × buffer time), so a debt kept in bits
    /// would correct twice as hard after halving the target, repaying bits
    /// the old, faster channel already carried; a changed buffer alone
    /// keeps the debt in bits (a larger buffer corrects it more gently).
    /// Measured (2x down at frame 150 of 300, single key, `crowd_mot20` and
    /// `pan_jaad` 720p, H.264/HEVC/AV1): with the debt kept in bits the first
    /// 1 s window reached ±10% of the new target only after 55–67 frames,
    /// QP overshot its settled level by 3 (H.264) for the first 10 frames
    /// with a 0.7–1.6 dB dip, and the 5 s after the change coded 509–534
    /// kbps for 600; scaled, the first window is within 10%, there is no
    /// overshoot or dip, and they code 562–564 kbps. 2x up is unchanged.
    pub fn reconfigure(&mut self, config: RateControlConfig) -> Result<(), RateControlError> {
        let s = Settings::new(config, &self.calibration)?;
        if self.steps.len() <= s.max as usize {
            return Err(RateControlError("quantizer table too short"));
        }
        let debt = self.fullness - self.settings.capacity / 2.;
        let scale = config.target_bps as f64 / self.settings.config.target_bps as f64;
        self.fullness = s.capacity / 2. + debt * scale;
        self.scheduled_key_debt *= scale;
        self.debit = self.debit.map(|integral| integral * scale);
        if self.occupancy > s.capacity {
            self.stats.buffer_overflows += 1;
        }
        // A smaller modeled VBV clips excess occupancy after counting overflow;
        // it does not claim to preserve bytes already queued by a transport.
        self.occupancy = self.occupancy.min(s.capacity);
        self.target_occupancy = self.target_occupancy.min(s.capacity);
        self.strict_occupancy = self.strict_occupancy.min(s.capacity);
        if config.rate_exponent.to_bits() != self.settings.config.rate_exponent.to_bits() {
            self.model = [None; MODEL_CLASSES];
        }
        // Keep the first key's scale while applying a changed boost.
        let boost_change = config.keyframe_boost / self.settings.config.keyframe_boost;
        self.first_key_boost *= boost_change;
        self.loan_boost *= boost_change;
        self.settings = s;
        // Preserve real accumulated target debt. Only the part of the old
        // key reservation repayable by the remaining Ps is still scheduled;
        // shrinking the target must not forgive an already emitted large key.
        // Increasing it cannot invent credit for a key that was never emitted.
        let remaining = if self.stats.frames == 0 {
            0
        } else {
            u64::from(self.repay_frames().saturating_sub(self.frames_since_key))
        };
        let repayment = (self.uniform_budget() - self.loan_inter_budget()).max(0.);
        self.scheduled_key_debt = self.scheduled_key_debt.min(remaining as f64 * repayment);
        self.current = self.current.clamp(s.min, s.max);
        self.last_inter_quantizer = self.last_inter_quantizer.map(|q| q.clamp(s.min, s.max));
        self.stats.current_quantizer = self.current;
        self.stats.buffer_fullness = (self.occupancy / s.capacity).clamp(0., 1.);
        self.stats.target_bps = config.target_bps;
        Ok(())
    }
    fn quantizer_for(&self, step: f64) -> u32 {
        quantizer_for_step(&self.steps, (self.settings.min, self.settings.max), step)
    }
    /// The frame budget the stream aims at on average: the target's bits
    /// per frame, less `rate.debit_margin` with the debit term. Budgets,
    /// key loans and the debt all read it, so the stream converges to it.
    fn uniform_budget(&self) -> f64 {
        self.channel_drain() * (1. - self.debit_margin)
    }
    /// The target's bits per frame interval: what a strict-CBR channel
    /// drains from a receiver's buffer.
    fn channel_drain(&self) -> f64 {
        self.settings.config.target_bps as f64 / self.frame_fps
    }
    /// The skip-aware budget scale of the next frame: 1 for keys.
    fn skip_scale(&self, key: bool) -> f64 {
        if key {
            1.
        } else {
            self.skip_budget.scale(self.bits_per_pixel())
        }
    }
    /// The target's bits per frame per luma sample (`bpp_curve.rs`).
    fn bits_per_pixel(&self) -> f64 {
        self.channel_drain() / self.pixels.max(1.)
    }
    /// Frames over which a key's loan is repaid: the key period,
    /// capped at `rate.key_repay_seconds` so very long or infinite periods
    /// (single-key streams, intra refresh) still repay the first key.
    fn repay_frames(&self) -> u32 {
        let horizon = (self.phase_repay_seconds * self.frame_fps).ceil().max(1.);
        self.period.max(1).min(horizon as u32)
    }
    /// The repay horizon of a key planned now: `rate.key_repay_seconds`,
    /// blended linearly towards `rate.key_repay_still_seconds` as the running
    /// share of unchanged blocks rises from `rate.key_repay_still_from` to
    /// `rate.key_repay_still_full`. Chosen once per key, so a phase's loan
    /// and repayment stay consistent; the first key has no history, takes
    /// the moving-scene horizon and is re-planned once a second of history
    /// exists (`replan_first_key`). Safe for the rate: the loan is repaid in
    /// full either way, only more slowly, and the debit term keeps the
    /// moving-scene horizon, so the long-run rate holds.
    fn repay_seconds_for_key(&self) -> f64 {
        let still = if self.key_repay_still_seconds > 0. {
            self.key_repay_still_seconds
        } else {
            self.key_repay_seconds
        };
        let span = self.key_repay_still_full - self.key_repay_still_from;
        let weight = if span > 0. {
            ((self.still_share - self.key_repay_still_from) / span).clamp(0., 1.)
        } else if self.still_share >= self.key_repay_still_full {
            1.
        } else {
            0.
        };
        self.key_repay_seconds + weight * (still - self.key_repay_seconds)
    }
    /// Count an inter frame's share of unchanged blocks into `still_share`,
    /// and re-plan the first key once a second of it is in.
    fn track_stillness(&mut self, static_fraction: f64) {
        self.still_frames = self.still_frames.saturating_add(1);
        let weight = (1. / f64::from(self.still_frames))
            .max(1. / (STILL_AVERAGE_SECONDS * self.frame_fps))
            .min(1.);
        self.still_share += (static_fraction - self.still_share) * weight;
        if self
            .first_key_replan_at
            .is_some_and(|at| self.frames_since_key >= at)
        {
            self.first_key_replan_at = None;
            self.replan_first_key();
        }
    }
    /// Re-plan the first key's repayment once [`STILL_AVERAGE_SECONDS`] of
    /// inter frames show how still the scene is: single-key and long-GOP
    /// streams have no history at their first key, which is the key a still
    /// scene keeps referencing longest. The remaining loan is spread over the
    /// rest of the new horizon by choosing the boost whose repayment phase
    /// repays exactly that loan in exactly those frames (with uniform budget
    /// U and per-frame repayment r = debt / remaining frames, the phase's
    /// inter budget U·F/(F-1+b) repays U-r when b = (U + r(F-1)) / (U - r)),
    /// so no loan is forgiven or double-counted. A horizon the loan cannot
    /// be spread over (the phase already over, or r >= U) is left alone.
    fn replan_first_key(&mut self) {
        let horizon = self.repay_seconds_for_key();
        if (horizon - self.phase_repay_seconds).abs() < REPLAN_MIN_SECONDS {
            return;
        }
        let frames = (horizon * self.frame_fps)
            .ceil()
            .max(1.)
            .min(f64::from(self.period.max(1)));
        let remaining = frames - f64::from(self.frames_since_key);
        let uniform = self.uniform_budget();
        let repayment = self.scheduled_key_debt / remaining.max(1.);
        if remaining < 1. || repayment >= uniform {
            return;
        }
        self.phase_repay_seconds = horizon;
        self.loan_boost = (uniform + repayment * (frames - 1.)) / (uniform - repayment);
    }
    /// Inter-frame budget of the repayment phase of a key at `boost` times
    /// it: the key plus `repay_frames - 1` inter frames at it spend exactly
    /// the uniform budget over the phase.
    fn phase_inter_budget(&self, boost: f64) -> f64 {
        let frames = f64::from(self.repay_frames());
        self.uniform_budget() * frames / (frames - 1. + boost)
    }
    /// Inter-frame budget of the current key's repayment phase.
    fn loan_inter_budget(&self) -> f64 {
        self.phase_inter_budget(self.loan_boost)
    }
    /// Most bits a key may take (`rate.key_cap_buffer`: a share of the
    /// buffer; `rate.key_cap_gop`: a share of its repayment phase's bits);
    /// unbounded when both are off. Keys plan within it (`key_boost`), and
    /// the key-level floor may not push a key past it (`key_step_floor`).
    fn key_bits_cap(&self) -> f64 {
        let phase = f64::from(self.repay_frames()) * self.uniform_budget();
        [
            (self.key_cap_buffer, self.settings.capacity),
            (self.first_key_cap_buffer, self.settings.capacity),
            (self.key_cap_gop, phase),
        ]
        .into_iter()
        .filter(|&(share, _)| share > 0.)
        .map(|(share, whole)| share * whole)
        .fold(self.share.room_bits, f64::min)
    }
    /// The key budget's multiple of a repayment-phase inter frame: the
    /// first key's boost or the configured one, lowered so it stays within
    /// `key_bits_cap`. The loan and the repayment then follow the capped
    /// key, not bits it never gets. With F repayment frames and uniform
    /// budget U, a boost b gives the key b·U·F/(F-1+b), which meets a cap C
    /// at b = C·(F-1)/(U·F - C).
    fn key_boost(&self) -> f64 {
        let cap = self.key_bits_cap();
        let phase_frames = f64::from(self.repay_frames());
        let phase = phase_frames * self.uniform_budget();
        let boost = if self.stats.frames == 0 {
            self.first_key_boost
        } else {
            self.settings.config.keyframe_boost
        };
        if cap >= phase {
            return boost;
        }
        boost.min(cap * (phase_frames - 1.) / (phase - cap)).max(1.)
    }
    /// The held frames' budget each coded inter frame spends.
    fn held_budget(&self) -> f64 {
        self.held_per_frame
    }

    /// Budget of the next inter frame: the repayment-phase budget, then the
    /// uniform budget once the phase is over.
    fn inter_budget(&self) -> f64 {
        if self.frames_since_key >= self.repay_frames() {
            self.uniform_budget()
        } else {
            self.loan_inter_budget()
        }
    }
    fn budget(&self, key: bool) -> f64 {
        self.budget_at(key, self.frame_skip_scale)
    }
    /// [`Self::budget`] at skip-budget scale `skip_scale`.
    fn budget_at(&self, key: bool, skip_scale: f64) -> f64 {
        let s = self.settings;
        let mut scheduled = if key {
            let boost = self.key_boost();
            self.phase_inter_budget(boost) * boost
        } else {
            (self.inter_budget() - self.still.repayment())
                .max(self.inter_budget() * MIN_BUDGET_CORRECTION)
                * self.share.budget_scale
                * skip_scale
                + self.held_budget()
        };
        if !key && self.share.reallocated_bits > 0. {
            scheduled += self.share.reallocated_bits;
        }
        let excess = self.excess() + self.debit.unwrap_or(0.);
        let correction =
            (1. - excess / s.capacity).clamp(MIN_BUDGET_CORRECTION, MAX_BUDGET_CORRECTION);
        let external_cap = if key {
            self.share.room_bits
        } else {
            self.share
                .room_bits
                .max(self.uniform_budget() + self.share.reallocated_bits)
        };
        (scheduled * correction)
            .min(s.peak / self.frame_fps + (s.capacity - self.occupancy))
            .min(self.strict_room(key))
            .min(external_cap)
            .max(1.)
    }
    /// Debt beyond the scheduled key loan: what the correction repays.
    fn excess(&self) -> f64 {
        self.fullness - self.settings.capacity / 2. - self.scheduled_debt()
    }
    /// Debt repaid on schedule: the key's loan and a refinement's.
    fn scheduled_debt(&self) -> f64 {
        self.scheduled_key_debt + self.still.outstanding()
    }
    /// The debit term, after a frame interval: the debt correction alone
    /// is proportional, and on content whose bits come in one-sided bursts
    /// (static refinement: 0.5-1.1 buffers every 1-3 s, each repaid over
    /// about 1 s) its excess settles at a positive mean, so the stream runs
    /// above the target line (`long_static_chirla_s024c2` AV1 400 kbps:
    /// +0.4 buffer on average over 30 s). The term integrates the excess
    /// over the repay horizon and adds it to the correction (never below
    /// 0: undershoot is not credit; at most [`DEBIT_MAX_BUFFERS`]), so a
    /// lasting excess is repaid; and the debt keeps at most
    /// `rate.debit_credit` buffers of undershoot, so cheap stretches do not
    /// fund unbounded later bursts. With the 1% `rate.debit_margin` and a
    /// 5 s horizon, 600-frame long-clip runs (keys every 10 s) land at a
    /// median 0.99-1.00 of the target instead of 1.009-1.013, at AV1 J
    /// +0.12% (static -0.05%).
    fn debit(&mut self) {
        let Some(integral) = self.debit else {
            return;
        };
        let floor = self.settings.capacity / 2. + self.scheduled_debt() - self.debit_credit;
        self.fullness = self.fullness.max(floor);
        let horizon = (self.key_repay_seconds * self.frame_fps).max(1.);
        let most = DEBIT_MAX_BUFFERS * self.settings.capacity;
        self.debit = Some((integral + self.excess() / horizon).clamp(0., most));
    }
    /// Count `bits` of the frame interval just passed in the short window.
    /// A key outside the window bound counts as one uniform frame: its
    /// excess is the loan its repayment phase repays.
    fn record_window(&mut self, bits: f64, key: bool) {
        let uniform = self.uniform_budget();
        let seconds = 1. / self.frame_fps;
        let exempt = key && !self.window_keys;
        if let Some(window) = &mut self.window {
            window.record(seconds, if exempt { uniform.min(bits) } else { bits });
        }
    }
    /// Bits the next frame may take, with `rate.recode_overflow`, before the
    /// buffer drained at the target rate (what a strict-CBR receiver holds)
    /// would pass the frame's ceiling (`recode_ceiling`) times its capacity,
    /// and with `rate.window_seconds` before the short window would pass the
    /// target plus its tolerance; unbounded otherwise. The frame budget
    /// stays within it too, so a planned frame never aims past the room that
    /// a re-code would enforce.
    ///
    /// An inter frame may always take the drain of its interval: one that
    /// small does not raise the bucket, so a buffer a planned key left above
    /// the inter ceiling drains at the key's repayment pace instead of
    /// starving the frames after it (AV2 `static_cam22` at 300 kbps, key at
    /// 1.18 buffers, inter ceiling 1.1: worst-1% PSNR-Y BD +22%).
    fn strict_room(&self, key: bool) -> f64 {
        let ceiling = self.recode_ceiling(key);
        let buffer = if ceiling <= 0. {
            f64::INFINITY
        } else {
            let room = ceiling * self.settings.capacity - self.strict_occupancy;
            if key && self.holds_still_key() {
                // A still scene's key may take `rate.key_hold_still` buffers whatever the
                // debt: the frames after it repay it.
                room.max(self.settings.capacity * self.still.key_hold_buffers)
            } else if key {
                room
            } else {
                room.max(self.channel_drain())
            }
        };
        buffer.min(self.window_room(key))
    }
    /// The short window's room for the next frame (unbounded when off, and
    /// for a key the window exempts).
    fn window_room(&self, key: bool) -> f64 {
        match &self.window {
            Some(window) if !key || self.window_keys => {
                let allowed = self.settings.config.target_bps as f64
                    * window.seconds()
                    * (1. + self.window_tolerance);
                window.room(allowed, 1. / self.frame_fps)
            }
            _ => f64::INFINITY,
        }
    }
    /// The re-code ceiling of a key or inter frame (0 = off). Keys are
    /// planned loans whose size the budget already bounds; inter frames
    /// overflow through unplanned static refinement bursts, which a backend
    /// may hold to a tighter ceiling (`rate.recode_overflow_inter`).
    fn recode_ceiling(&self, key: bool) -> f64 {
        if key || self.recode_ceiling_inter <= 0. {
            self.recode_ceiling
        } else {
            self.recode_ceiling_inter
        }
    }
    /// Model slot of a frame: one for keys; one per temporal-layer role in
    /// a layered structure; one per cadence role for flat inter frames, or a
    /// single shared one when the cadence is flat.
    fn model_slot(&self, key: bool, index: u64, layer: LayerPosition) -> usize {
        if key {
            KEY_MODEL
        } else if !layer.is_flat() {
            INTER_MODEL + CadenceRole::of_layer(layer).index()
        } else if self.cadence.is_flat() {
            INTER_MODEL
        } else {
            INTER_MODEL + self.cadence.role(index - self.last_key_index).index()
        }
    }

    /// Inter level step that spends `budget` per frame on average over one
    /// cadence cycle (or mini-GOP of a layered structure):
    /// `Σ_roles m_r·h / (s·ratio_r)^e = n·budget`, solved for `s`.
    /// A role not yet observed borrows the mean of the observed roles.
    /// `distance`: frames since the last key (which cadence cycle applies).
    fn inter_level_step(
        &self,
        hint: f64,
        budget: f64,
        strength: f64,
        layer: LayerPosition,
        distance: u64,
    ) -> Option<f64> {
        let exponent = self.settings.config.rate_exponent;
        if layer.is_flat() && self.cadence.is_flat() {
            return self.model[INTER_MODEL].map(|m| (m * hint / budget).powf(1. / exponent));
        }
        let observed: Vec<f64> = self.model[INTER_MODEL..]
            .iter()
            .flatten()
            .copied()
            .collect();
        if observed.is_empty() {
            return None;
        }
        let fallback = observed.iter().sum::<f64>() / observed.len() as f64;
        let (mut weighted, mut frames) = (0., 0.);
        let mut add = |role: CadenceRole, offset: f64| {
            let model = self.model[INTER_MODEL + role.index()].unwrap_or(fallback);
            weighted += model * step_ratio(offset).powf(-exponent);
            frames += 1.;
        };
        if layer.is_flat() {
            for d in self.cadence.cycle(distance) {
                add(self.cadence.role(d), self.cadence.offset(d, strength));
            }
        } else {
            for position in layer.cycle() {
                add(
                    CadenceRole::of_layer(position),
                    self.ladder.offset(position),
                );
            }
        }
        Some((weighted * hint / (frames * budget)).powf(1. / exponent))
    }

    /// The step of a key frame: the finer of what its budget affords (the
    /// configured key boost, repaid on schedule) and what the quality
    /// around it requires, coarsened only where its bits would not fit
    /// `key_allowance`. The quality requirement is the inter level's step,
    /// the λ of the frames that will predict from the key, made finer
    /// where later frames reuse it by the reuse term (`reuse.rs`: per block
    /// where the codec codes block quantizers, else `reuse_offset` for the
    /// whole frame), so a key never re-codes persistent content coarser
    /// than the frames before it held it. The first key has no level, so
    /// its budget (the first-key boost) alone sizes it. Requested,
    /// periodic and resolution-change keys all take this one path.
    ///
    /// Ignoring the budget (the level alone) lost 0.14 dB AV1 / 0.03 dB
    /// H.264 whole-frame PSNR-Y on the looped waterfall (360p15, 300 kbps,
    /// key every 100 f) against this rule: a budget-sized key's finer
    /// moving content pays off in the frames that predict from it. This
    /// rule replaces the former `rate.key_level_offset` floor (inter level
    /// minus 6 QP times the static share) and per-codec periodic boosts.
    fn key_step(&self, budget_step: f64, reuse_offset: f64, hint: f64) -> f64 {
        let Some(level) = self.last_inter_quantizer else {
            return budget_step;
        };
        let level_step = self.steps[level as usize] * step_ratio(reuse_offset);
        let allowance_step = self.model[KEY_MODEL].map_or(0., |m| {
            (m * hint / self.key_allowance()).powf(1. / self.settings.config.rate_exponent)
                / self.block_ratio
        });
        if self.holds_still_key() {
            // The held quality, within the buffers alone: the frames after
            // the key repay what it borrows.
            return level_step.max(allowance_step);
        }
        level_step.min(budget_step).max(allowance_step)
    }

    /// Whether the key being planned is a periodic key of a still scene
    /// planned at the quality its inter frames hold (`rate.key_hold_still`).
    fn holds_still_key(&self) -> bool {
        self.still.key_hold_buffers > 0.
            && self.stats.frames > 0
            && self.still_share >= self.key_repay_still_from
    }

    /// Most bits a key may take: what the peak channel and the physical
    /// buffer can hold, the re-code ceiling's room, `key_bits_cap`, and the
    /// largest loan its repayment phase can repay while leaving its inter
    /// frames the debt correction's floor ([`MIN_BUDGET_CORRECTION`] of the
    /// uniform budget): without the last, a static scene's key at the fine inter
    /// level took a whole 1 s GOP's bits (VP9 `static_chirla_s000c3_360p`
    /// 150 kbps, key every 30 frames: keys of 141-150 kbit against a budget
    /// of 32 kbit, every GOP 13-23% over).
    ///
    /// A still scene's key with `rate.key_hold_still` is bounded by the
    /// strict buffer's room alone (at least that many buffers): its loan is
    /// repaid by the frames after it.
    fn key_allowance(&self) -> f64 {
        if self.holds_still_key() {
            return self.strict_room(true).max(1.);
        }
        let frames = f64::from(self.repay_frames());
        let repayable = self.uniform_budget() * (frames - (frames - 1.) * MIN_BUDGET_CORRECTION);
        (self.settings.peak / self.frame_fps + (self.settings.capacity - self.occupancy))
            .min(self.strict_room(true))
            .min(self.key_bits_cap())
            .min(repayable)
            .max(1.)
    }

    /// While finer inter frames no longer buy quality
    /// (`RefinementGain::exhausted`), keep the inter level from getting
    /// finer than the previous one and coarsen it by one quantizer every
    /// `CAP_STEP_FRAMES` frames; the hold ends as soon as the measured gain
    /// returns. Returns the level and whether the hold raised it.
    fn hold_exhausted_refinement(&mut self, key: bool, level: u32) -> (u32, bool) {
        if key {
            self.refinement.reset();
            self.cap_frames = 0;
            return (level, false);
        }
        let Some(previous) = self
            .last_inter_quantizer
            .filter(|_| self.refinement.exhausted())
        else {
            self.cap_frames = 0;
            return (level, false);
        };
        self.cap_frames += 1;
        let climb = u32::from(self.cap_frames.is_multiple_of(CAP_STEP_FRAMES));
        let held = (previous + climb).min(self.settings.max);
        (level.max(held), held > level)
    }

    /// The inter level quantizer of the latest inter frame (`None` before
    /// the first): the quantizer the frames around a key are coded at.
    pub fn inter_level(&self) -> Option<u32> {
        self.last_inter_quantizer
    }

    /// Whether the encoder should measure each frame's temporal noise
    /// ([`Complexity::residual_sigma`]).
    pub fn wants_temporal_noise(&self) -> bool {
        self.settings.config.temporal_noise_stat || self.quality_floor.wants_noise()
    }

    /// Whether the encoder should report each inter frame's luma distortion
    /// (`observe_distortion`).
    pub fn wants_distortion(&self) -> bool {
        self.refinement.enabled() && !self.last_key
    }

    /// The share (0..=1) of the frame just coded that the noise skip took,
    /// for the skip-aware budget of the next inter frame; a key resets it.
    pub fn observe_noise_skip(&mut self, skipped_share: f64) {
        if self.last_key {
            self.skip_budget.reset();
        } else {
            self.skip_budget.observe(skipped_share);
        }
    }

    /// The luma mean squared error of the frame just coded, for the
    /// refinement-gain measure of the quality cap.
    pub fn observe_distortion(&mut self, mse: f64) {
        if !self.last_key {
            let qp_units = QP_UNITS_PER_OCTAVE * self.steps[self.current as usize].log2();
            self.refinement.observe(qp_units, mse);
        }
    }

    /// Finest quantizer the quality cap allows the next frame
    /// (`quality_cap.rs`).
    fn quality_floor(&self) -> u32 {
        self.quantizer_for(self.quality_floor.step(self.bits_per_pixel()))
    }

    /// The QP offset of the next frame from the inter level: its layer's
    /// ladder offset, its cadence offset, or 0 (keys and flat streams).
    fn frame_offset(&self, key: bool, layer: LayerPosition, index: u64, strength: f64) -> f64 {
        if key || (layer.is_flat() && self.cadence.is_flat()) {
            0.
        } else if !layer.is_flat() {
            self.ladder.offset(layer)
        } else {
            self.cadence_offset(index, strength)
        }
    }

    /// The cadence offset of flat inter frame `index` at `strength`, quieted
    /// before a key.
    fn cadence_offset(&self, index: u64, strength: f64) -> f64 {
        self.quieted(
            index,
            self.cadence.offset(index - self.last_key_index, strength),
        )
    }

    /// The frame just selected on `budget`: its plan is its ordinary budget
    /// and prediction plus its refinement loan (0 on an ordinary frame).
    fn add_refinement_loan(&mut self, key: bool, budget: f64) {
        let loan = self.still.on_select(key, self.last_predicted, budget);
        self.last_predicted += loan;
        self.last_budget = budget + loan;
    }

    /// `offset` with no base boost in the last `rate.prekey_quiet` before a
    /// scheduled periodic key: whatever a finer frame refines there the key
    /// codes again (AV1 Default, 600 f still points: 8-20% of the base
    /// bursts' bits fell in the second before a key).
    fn quieted(&self, index: u64, offset: f64) -> f64 {
        let period = u64::from(self.period);
        let to_key = (period > 0 && period != u64::from(u32::MAX))
            .then(|| period - (index - self.last_key_index) % period);
        self.still.quieted(to_key, offset)
    }

    /// The largest loan a refinement of the next inter frame may take: its
    /// `rate.refine_budgets` uniform budgets, within the room the strict
    /// buffer leaves beyond the frame's own budget (so a refinement that
    /// costs its loan is never re-coded); 0 when off or below
    /// `rate.refine_min_budgets`. With the smallest worthwhile loan.
    pub(crate) fn refinement_room(&self) -> Option<(f64, f64)> {
        let room = (self.strict_room(false) - self.budget(false)).max(0.);
        self.still.room(self.uniform_budget(), room)
    }

    /// Plan the next inter frame as a refinement whose blocks were
    /// estimated at `estimate` bits: its loan comes on top of its ordinary
    /// budget (`RefinementLoan`).
    pub(crate) fn plan_refinement(&mut self, estimate: f64) {
        self.still.plan(estimate);
    }

    /// The floor of a frame at `offset` QP from the inter level: the quality
    /// floor `floor`, or with `rate.quality_floor_cadence` the floor moved by
    /// a finer frame's offset, so the cadence's base frames keep their boost
    /// while the floor bounds the level.
    fn frame_floor(&self, floor: u32, offset: f64) -> u32 {
        let step = self.quality_floor.step(self.bits_per_pixel());
        if self.quality_floor.keeps_cadence() && offset < 0. && step > 0. {
            self.quantizer_for(step * step_ratio(offset))
        } else {
            floor
        }
    }

    /// Step for a frame with no observed model of its kind: a bits/pixel
    /// guess at the seed quantizer, scaled by spatial activity.
    fn bootstrap_step(&self, key: bool, hint: Complexity, budget: f64) -> f64 {
        let exponent = self.settings.config.rate_exponent;
        let activity = (hint.spatial / (self.pixels / 64.).max(1.) / 128.).clamp(0.25, 4.);
        let bpp = self
            .settings
            .config
            .initial_bits_per_pixel
            .unwrap_or(self.calibration.bootstrap_bits_per_pixel);
        let reference_bits = self.pixels
            * bpp
            * activity
            * if key {
                self.settings.config.initial_key_complexity
            } else {
                1.
            };
        self.steps[self.seed as usize] * (reference_bits / budget).powf(1. / exponent)
    }

    /// The picture has `pixels` luma samples from the next frame on (a
    /// resolution change). The models are bits per unit of the complexity
    /// hint, which sums over the picture's cells, so they carry over as a
    /// start, but each is replaced by its first observation at the new
    /// size (`relearn`); the first-frame guess reads the size.
    /// Whether the start-up lookahead is wanted (`rate.first_key_lookahead`).
    pub fn wants_startup_motion(&self) -> bool {
        self.first_key_lookahead > 0.
    }

    /// Before the first frame: the peak camera displacement (luma samples)
    /// from the first to the second source frame. Camera motion caps the
    /// first GOP's key at `rate.first_key_lookahead` buffers, through
    /// `key_bits_cap`, so the loan and its repayment follow the capped key.
    /// The first key's 2.5x boost (`rate.first_key_boost_scale`) pays where
    /// the picture persists: static and crowd scenes (a still camera) keep
    /// it. Under camera motion the picture is replaced within seconds and a
    /// smaller key (cap 0.2-0.3) won pan by 12-15% in a blanket screen,
    /// which lost crowd by 2-11% (`rate.key_cap_buffer`).
    pub fn observe_startup_motion(&mut self, peak_displacement: f64) {
        if self.stats.frames == 0 && peak_displacement >= STARTUP_CAMERA_MOTION {
            self.first_key_cap_buffer = self.first_key_lookahead;
        }
    }

    /// Start without the initial keyframe loan: a layer whose first picture
    /// predicts from another layer (spatial SVC `S1` with
    /// `spatial.inter_layer = 1`) codes no intra key to repay.
    pub(crate) fn clear_key_loan(&mut self) {
        self.loan_boost = 1.0;
        self.scheduled_key_debt = 0.0;
    }

    /// Nominal scheduled inter-frame budget (before any layer share).
    pub(crate) fn inter_frame_budget(&self) -> f64 {
        self.inter_budget()
    }

    /// Forfeit `unspent_bits` that `S0` handed to `S1` on an inter temporal
    /// unit so `S0` does not bank them as negative debt credit to force a
    /// finer `S0` quantizer on subsequent frames.
    pub(crate) fn forfeit_unspent_inter_bits(&mut self, unspent_bits: f64) {
        if unspent_bits <= 0.0 {
            return;
        }
        self.fullness += unspent_bits;
        self.target_occupancy =
            (self.target_occupancy + unspent_bits).clamp(0.0, self.settings.capacity);
        self.strict_occupancy = (self.strict_occupancy + unspent_bits).max(0.0);
        if let Some(window) = &mut self.window {
            window.adjust_latest(unspent_bits);
        }
    }

    /// Sync `last_inter_quantizer` to `current` after an inter-frame recode so
    /// subsequent inter frames step from the post-recode quantizer rather than
    /// the discarded trial's.
    pub(crate) fn sync_inter_level_after_recode(&mut self) {
        if !self.last_key {
            self.last_inter_quantizer = Some(self.current);
        }
    }

    /// Record a recode triggered by the combined temporal-unit CBR buffer
    /// (`CombinedCbrBuffer`) so the layer controller learns from the discarded
    /// trial at `self.current` and updates its active quantizer to `new_q`.
    pub(crate) fn record_external_recode(&mut self, new_q: u32, discarded_bits: u64) {
        let current_step = self.steps[self.current as usize];
        let size = discarded_bits as f64;
        self.learn(discarded_bits);
        if self.recodes_this_frame == 0 {
            self.stats.recoded_frames += 1;
        }
        self.recodes_this_frame += 1;
        self.recode_trial = Some((current_step, size));
        self.current = new_q.clamp(self.settings.min, self.settings.max);
        if !self.last_key {
            self.last_inter_quantizer = Some(self.current);
        }
    }

    pub fn set_picture_size(&mut self, pixels: usize) {
        self.pixels = pixels as f64;
        self.relearn = [true; MODEL_CLASSES];
    }

    /// Largest change of the inter level from one frame to the next: the
    /// codec's step limit times `rate.level_step_scale`. The level re-solves
    /// every frame from noisy complexity hints, and on steady content that
    /// jitter alone is visible flicker (AV1 `lowlight_pp22_004` at 500 kbps,
    /// cadence off: mean |dPSNR| 0.115 dB with the 20-qindex limit, 0.063
    /// with 2). On the CBR flicker set (3 lowlight, bodycam, crowd, pan; 150
    /// f) 0.5 gave AV1 `J_real` -0.1%, mean |dPSNR| -0.04 dB (bodycam -0.10)
    /// and bodycam worst-1% -31%; 0.25 reacted too slowly (`J_real` +1.1%).
    fn level_step(&self) -> u32 {
        ((f64::from(self.settings.step) * self.level_step_scale).round() as u32).max(1)
    }

    /// `level`, raised to `floor`, within the bounds `share` sets the frame
    /// (an inter frame's finest quantizer, a key's anchor), and whether the
    /// share capped it (its budget scale or its quantizer floor). The frame's
    /// plan caps end here: recode rooms after the plan see only its
    /// reallocated drain.
    fn share_bounded_level(
        &mut self,
        key: bool,
        level: u32,
        floor: u32,
        share: &LayerShare,
    ) -> (u32, bool) {
        let (min, max) = (self.settings.min, self.settings.max);
        let min_inter_q = share.min_inter_quantizer.filter(|_| !key);
        let capped =
            !key && (share.budget_scale < 1.0 || min_inter_q.is_some_and(|min_q| level < min_q));
        self.share = LayerShare {
            reallocated_bits: share.reallocated_bits,
            ..LayerShare::WHOLE
        };
        let mut level = level.max(floor);
        if let Some(min_q) = min_inter_q {
            level = level.max(min_q.clamp(min, max));
        }
        if let Some(anchor) = share.key_anchor.filter(|_| key) {
            let lo = anchor.min.clamp(min, max);
            level = level.clamp(lo, anchor.max.clamp(lo, max));
        }
        (level, capped)
    }

    /// Choose the quantizer of the next frame, at temporal-layer position
    /// `layer` (layered structures apply the per-layer ladder offsets around
    /// the level quantizer, like the flat pseudo-layer cadence).
    /// `share` is the frame's share of a buffer it shares with other frames
    /// of the same instant ([`LayerShare::WHOLE`]: none).
    pub(crate) fn select(
        &mut self,
        key: bool,
        index: u64,
        hint: Complexity,
        layer: LayerPosition,
        share: &LayerShare,
    ) -> u32 {
        if share.clear_key_loan {
            self.clear_key_loan();
        }
        self.share = *share;
        if key {
            if self.stats.frames > 0 {
                self.first_key_cap_buffer = 0.;
            }
            self.last_key_index = index;
            self.phase_repay_seconds = self.repay_seconds_for_key();
            self.first_key_replan_at = (self.stats.frames == 0
                && self.key_repay_still_seconds > 0.)
                .then(|| (STILL_AVERAGE_SECONDS * self.frame_fps).ceil() as u32);
        } else {
            self.track_stillness(hint.static_fraction);
        }
        let h = hint.value(key);
        let held = std::mem::take(&mut self.held_pending);
        self.held_per_frame += (held - self.held_per_frame) / HELD_TIME_FRAMES;
        self.held_spent = if key { 0. } else { self.held_budget() };
        self.frame_skip_scale = self.skip_scale(key);
        let budget = self.budget(key);
        self.unscaled_budget = self.budget_at(key, 1.);
        let slot = self.model_slot(key, index, layer);
        let strength = self.cadence.strength(hint.static_fraction, hint.motion);
        let exponent = self.settings.config.rate_exponent;
        let level_step = if key {
            self.model[KEY_MODEL].map(|m| (m * h / budget).powf(1. / exponent))
        } else {
            self.inter_level_step(h, budget, strength, layer, index - self.last_key_index)
        }
        .unwrap_or_else(|| self.bootstrap_step(key, hint, budget));
        // Block-level offsets change the frame's effective step: the frame
        // quantizer is chosen so that the effective step meets the budget,
        // and the bits of finer blocks are budgeted, not read as complexity.
        self.block_ratio = step_ratio(hint.block_qp_offset);
        let level_step = level_step / self.block_ratio;
        let goal = self.quantizer_for(if key {
            self.key_step(level_step, hint.key_reuse_offset, h)
        } else {
            level_step
        });
        // Smooth comparable frames, not the deliberately different I/P budget.
        // Cross-class limits otherwise force a low-QP key's following P frames
        // to overspend for several frames and then visibly over-correct.
        let level = if let Some(previous) = self.last_inter_quantizer.filter(|_| !key) {
            goal.clamp(
                previous.saturating_sub(self.level_step()),
                previous.saturating_add(self.level_step()),
            )
        } else {
            goal
        }
        .clamp(self.settings.min, self.settings.max);
        if let Some(sigma) = hint.residual_sigma {
            self.stats.residual_sigma +=
                RESIDUAL_SIGMA_SMOOTHING * (sigma - self.stats.residual_sigma);
        }
        self.quality_floor.observe(key, hint.residual_sigma);
        let floor = self.quality_floor();
        let (level, share_capped) = self.share_bounded_level(key, level, floor, share);
        let (level, held) = self.hold_exhausted_refinement(key, level);
        let offset = self.frame_offset(key, layer, index, strength);
        let q = if offset == 0. {
            level
        } else {
            let step = self.steps[level as usize] * step_ratio(offset);
            nearest_quantizer(&self.steps, (self.settings.min, self.settings.max), step)
        };
        self.quality_capped = q < floor || goal < floor || held || share_capped;
        let mut q = q.max(self.frame_floor(floor, offset));
        if let Some(min_q) = share.min_inter_quantizer.filter(|_| !key) {
            q = q.max(min_q.clamp(self.settings.min, self.settings.max));
        }
        if key {
            self.loan_boost = self.key_boost();
            if let Some(seed_offset) = share.key_anchor.and_then(|a| a.seed_inter_offset)
                && self.last_inter_quantizer.is_none()
            {
                self.last_inter_quantizer =
                    Some(q.saturating_add(seed_offset).min(self.settings.max));
            }
        } else {
            self.last_inter_quantizer = Some(level);
            self.last_inter_hint = Some(h);
        }
        self.current = q;
        self.recodes_this_frame = 0;
        self.recode_trial = None;
        self.current_slot = slot;
        self.last_hint = h;
        self.last_index = index;
        self.last_key = key;
        self.last_peak =
            self.settings.peak / self.frame_fps + (self.settings.capacity - self.occupancy);
        self.last_predicted = self.model[slot].map_or(budget, |m| {
            m * h / (self.steps[q as usize] * self.block_ratio).powf(exponent)
        });
        self.add_refinement_loan(key, budget);
        let cadence_offset = if key || !layer.is_flat() {
            0.
        } else {
            self.cadence_offset(index, strength)
        };
        self.trace_plan(level, cadence_offset, hint.static_fraction);
        q
    }
    /// Record the plan of the frame just selected in the rate-log trace.
    fn trace_plan(&mut self, level: u32, cadence_offset: f64, static_fraction: f64) {
        self.stats.plan = RatePlanTrace {
            budget_bits: self.last_budget,
            predicted_bits: self.last_predicted,
            level_quantizer: level,
            cadence_offset,
            static_fraction,
            ..self.stats.plan
        };
    }
    /// With `rate.recode_overflow`, the coarser quantizer to code the
    /// current frame again at when its `bits` would overflow the buffer
    /// drained at the target rate (what a strict-CBR receiver holds), or
    /// `None` to keep it. On static content a frame finer than its
    /// predecessors refines everything they left and can cost 30-100x its
    /// prediction (VP9 `static_chirla_s000c3_360p`: qindex 177 -> 92, 74 ->
    /// 12075 bytes); no model saw those bursts coming, and they caused most
    /// strict-CBR overruns, but they also buy real quality, so the frame is
    /// re-coded only when the buffer cannot absorb it, and only as much
    /// coarser as it needs: the first re-code assumes the steep slope of
    /// such bursts (`RECODE_FIRST_EXPONENT`), a second one (rare) the slope
    /// measured between the two codings. At most `MAX_RECODES` per frame.
    pub fn recode_quantizer(&mut self, bits: u64) -> Option<u32> {
        let room = self.strict_room(self.last_key);
        let size = bits as f64;
        if size <= room
            || self.current >= self.settings.max
            || self.recodes_this_frame >= MAX_RECODES
        {
            return None;
        }
        let current_step = self.steps[self.current as usize];
        let exponent = self
            .recode_trial
            .map_or(RECODE_FIRST_EXPONENT, |(step, trial_bits)| {
                ((trial_bits / size).ln() / (current_step / step).ln())
                    .clamp(RECODE_EXPONENT_RANGE.0, RECODE_EXPONENT_RANGE.1)
            });
        let goal = RECODE_FILL * room.max(1.);
        let step = current_step * (size / goal).powf(1. / exponent);
        let quantizer = self.quantizer_for(step).max(self.current + 1);
        // The discarded coding is a real observation at its quantizer:
        // without it the models saw only the cheap re-codes and kept
        // planning frames that overflow (static_london_bus H.264 150 kbps
        // re-coded every frame from 37 on).
        self.learn(bits);
        if self.recodes_this_frame == 0 {
            self.stats.recoded_frames += 1;
        }
        self.recodes_this_frame += 1;
        self.recode_trial = Some((current_step, size));
        self.current = quantizer;
        self.last_predicted = goal;
        Some(quantizer)
    }

    /// The bounds in-frame rate control (`inframe.rs`) steers the planned
    /// frame by: the room `recode_quantizer` would enforce on it (infinite
    /// when re-coding is off) and the uncorrected frame budget, which does
    /// not shrink with the debt (bounds scaled by the corrected budget
    /// starved the frames after a key: audit addendum, `rate.burst_cycles`).
    pub(crate) fn inframe_bounds(&self) -> (f64, f64) {
        (self.strict_room(self.last_key), self.uniform_budget())
    }

    /// Frames between scheduled key frames from now on (`u32::MAX`: none,
    /// as with intra refresh), for the key budget's amortization.
    pub fn set_key_period(&mut self, period: u32) {
        self.period = period;
    }

    /// Time the next frame covers: the seconds since the previous frame's
    /// capture, or `None` for the nominal `1 / fps` (whose arithmetic is
    /// exactly that of a controller without timestamps).
    /// Untimed frames from now on are `1 / fps` apart (a camera's new
    /// nominal rate; timestamped frames follow their own intervals).
    pub fn set_nominal_rate(&mut self, fps: f64) {
        self.fps = fps;
    }
    pub fn set_frame_interval(&mut self, seconds: Option<f64>) {
        self.frame_fps = seconds.map_or(self.fps, |s| 1. / s);
    }
    /// Whether the next inter frame should be skipped: frame dropping is on
    /// and, even at the maximum quantizer, the frame plus the `protected`
    /// frames that must follow it before the next droppable one (lower
    /// temporal layers) would overflow the buffer after their drain, in
    /// either the peak-rate bucket or the target-rate one (the tighter
    /// bound: with the default peak of twice the target, a sustained
    /// overshoot would otherwise stall a strict-CBR receiver long before
    /// the peak bucket fills). Each
    /// is predicted with the costliest observed inter model and the last
    /// inter frame's complexity: underestimating would let a frame through
    /// that overflows, and an overflow forgets bits, so the stream would
    /// exceed the target. Below the maximum quantizer rate control itself
    /// can still shrink frames, so nothing is dropped there.
    pub fn should_drop(&self, protected: u32) -> bool {
        if !self.settings.config.frame_drop {
            return false;
        }
        let Some(smallest) = self.smallest_inter_bits() else {
            return false;
        };
        let s = self.settings;
        let frames = f64::from(protected + 1);
        let overflows = |occupancy: f64, drain_bps: f64| {
            occupancy + frames * (smallest - drain_bps / self.frame_fps) > s.capacity
        };
        overflows(self.occupancy, s.peak)
            || overflows(self.target_occupancy, s.config.target_bps as f64)
    }

    /// Whether [`RateControlConfig::frame_drop`] is on.
    pub(crate) fn frame_drop_enabled(&self) -> bool {
        self.settings.config.frame_drop
    }

    /// The bits of the next inter frame at the maximum quantizer, predicted
    /// with the costliest observed inter model and the last inter frame's
    /// complexity (the floor [`Self::should_drop`] tests); `None` before
    /// any inter frame has been observed.
    pub(crate) fn smallest_inter_bits(&self) -> Option<f64> {
        let costliest = self.model[INTER_MODEL..]
            .iter()
            .flatten()
            .copied()
            .reduce(f64::max);
        let (Some(model), Some(hint)) = (costliest, self.last_inter_hint) else {
            return None;
        };
        let s = self.settings;
        Some(model * hint / self.steps[s.max as usize].powf(s.config.rate_exponent))
    }

    /// The single offset, in QP units, whose step gives a frame the same
    /// bits as its blocks' `(weight, offset)` mix, `weight` being each
    /// block's expected share of the bits: bits scale as step^-exponent,
    /// so the weighted mean of `ratio^-exponent` is what matters.
    /// `total_weight` covers every block; `offsets` lists those with a
    /// non-zero offset (the others' ratio is exactly 1, so a frame with a
    /// few refined blocks costs a few terms).
    pub fn effective_block_offset(
        &self,
        total_weight: f64,
        offsets: impl Iterator<Item = (f64, f64)>,
    ) -> f64 {
        let exponent = self.settings.config.rate_exponent;
        let mut weighted = total_weight;
        for (weight, offset) in offsets.filter(|&(_, offset)| offset != 0.) {
            weighted += weight * (step_ratio(offset).powf(-exponent) - 1.);
        }
        if total_weight <= 0. || weighted <= 0. {
            return 0.;
        }
        -QP_UNITS_PER_OCTAVE / exponent * (weighted / total_weight).log2()
    }

    /// Replace the planned block offset with the one the frame realised
    /// (what the blocks were actually coded with), before `feedback`.
    pub fn set_realised_block_offset(&mut self, offset_qp: f64) {
        self.block_ratio = step_ratio(offset_qp);
    }

    /// The realised offset, in QP units, of the rows in-frame rate control
    /// coarsened in the coding about to be learnt from (`feedback` or
    /// `recode_quantizer`), on top of the block offsets.
    pub(crate) fn set_inframe_offset(&mut self, offset_qp: f64) {
        self.inframe_ratio = step_ratio(offset_qp);
    }

    /// Account for a dropped frame: its interval passes with no bits, so the
    /// buffer drains and the target debt falls by an under-spent inter
    /// budget. The models are untouched (nothing was coded).
    pub fn drop_frame(&mut self) {
        self.pass_interval();
        self.stats.dropped_frames += 1;
    }

    /// Account for a held frame (`hold.rs`): as for a dropped one, its
    /// interval passes with no bits, drains the buffers and repays its share
    /// of a key's loan. Its own inter budget is not credited to the debt at
    /// once (the correction would then spend it on top of everything): the
    /// inter frames that follow spend it over [`HELD_TIME_FRAMES`]
    /// (`held_budget`), each counting what it took against the debt
    /// (`held_spent`), instead of the next frame alone (whose interval
    /// covered the held frame's at first: on
    /// `bodycam_antare_0815`, an exact repeat after the first key coded the
    /// next frame at QP 27 instead of 30, `J_real` +2.5% H.264, +4.4% AV1).
    pub fn hold_frame(&mut self) {
        let budget = self.inter_budget();
        self.held_pending += budget;
        self.fullness -= self.uniform_budget() - budget;
        self.drain_interval();
        self.debit();
    }

    /// The current frame's interval passes with no bits coded.
    fn pass_interval(&mut self) {
        self.fullness -= self.uniform_budget();
        self.drain_interval();
        self.debit();
    }

    /// The current frame's interval drains the buffers and the key's loan
    /// schedule (`pass_interval` without the debt credit).
    fn drain_interval(&mut self) {
        let uniform = self.uniform_budget();
        let drain = self.channel_drain();
        self.scheduled_key_debt =
            (self.scheduled_key_debt - (uniform - self.inter_budget())).max(0.);
        self.frames_since_key = self.frames_since_key.saturating_add(1);
        self.occupancy = (self.occupancy - self.settings.peak / self.frame_fps).max(0.);
        self.target_occupancy = (self.target_occupancy - drain).max(0.);
        self.strict_occupancy = (self.strict_occupancy - drain).max(0.);
        self.record_window(0., false);
        self.elapsed += 1. / self.frame_fps;
        self.stats.key_loan_bits = self.scheduled_key_debt;
        self.stats.buffer_fullness = (self.occupancy / self.settings.capacity).clamp(0., 1.);
        self.stats.actual_bps = self.stats.total_bits as f64 / self.elapsed;
    }
    /// Update the current frame's model slot with `bits` coded at the
    /// current quantizer.
    fn learn(&mut self, bits: u64) {
        let class = self.current_slot;
        let value = bits as f64
            * (self.steps[self.current as usize] * self.block_ratio * self.inframe_ratio)
                .powf(self.settings.config.rate_exponent)
            / self.last_hint;
        let ratio = (bits as f64 / self.last_predicted.max(1.))
            .max(self.last_predicted / bits.max(1) as f64);
        // A large miss restarts the model only when the content changed too.
        // On static content each frame's bits depend on how finely its
        // reference was coded, so misses of 5x and more alternate frame to
        // frame on unchanged content; restarting the model on each of them
        // made the quantizer bang between its step limits and overspend
        // (H.264 static_otw_homes390: +16..+19% over target, +2..+5% without).
        let hint_ratio =
            (self.last_hint / self.model_hint[class]).max(self.model_hint[class] / self.last_hint);
        let relearn = std::mem::take(&mut self.relearn[class]);
        let weight = if relearn
            || (ratio > self.settings.config.scene_change_ratio
                && hint_ratio >= self.scene_hint_ratio)
        {
            1.
        } else {
            self.settings.config.model_smoothing
        };
        self.model[class] =
            Some(self.model[class].map_or(value, |old| old * (1. - weight) + value * weight));
        self.model_hint[class] = self.last_hint;
    }
    pub fn feedback(&mut self, bits: u64) {
        // A refinement frame's bits are its ordinary plan plus the refined
        // blocks: the role model learns from ordinary frames only, the loan
        // its cost scale.
        if !self.still.refining() {
            self.learn(bits);
        }
        self.still.settle(bits as f64);
        self.inframe_ratio = 1.;
        let uniform = self.uniform_budget();
        let bonus = std::mem::take(&mut self.share.reallocated_bits);
        let p = self.inter_budget();
        self.fullness += bits as f64 - uniform - std::mem::take(&mut self.held_spent)
            + skip_budget::forgiven(bits as f64, self.unscaled_budget, self.frame_skip_scale);
        if bonus > 0.0 {
            self.fullness -= bonus;
        }
        // Every key, periodic or requested, borrows its planned budget and
        // the frames after it repay that on schedule; what it took beyond
        // (or short of) the plan, and what is left of an earlier key's loan,
        // the debt correction repays as ordinary debt. (Scheduling the
        // actual size instead cost J_real +0.8% AV1 / +0.3% H.264 on the
        // TRAIN proxy: an undershooting first key no longer handed its
        // unspent budget to the frames after it.)
        if self.last_key {
            self.frames_since_key = 0;
            self.scheduled_key_debt =
                (self.loan_inter_budget() * self.loan_boost - uniform).max(0.);
        } else {
            self.scheduled_key_debt = (self.scheduled_key_debt - (uniform - p)).max(0.);
        }
        self.frames_since_key = self.frames_since_key.saturating_add(1);
        if self.quality_capped {
            // The target is a ceiling: bits a capped frame left unspent are
            // banked only up to the correction's full response (one buffer
            // below the half-full level), never for a later burst.
            let credit_limit =
                self.settings.capacity / 2. + self.scheduled_key_debt - self.settings.capacity;
            self.fullness = self.fullness.max(credit_limit);
        }
        self.occupancy += bits as f64 - self.settings.peak / self.frame_fps;
        if self.occupancy > self.settings.capacity {
            self.stats.buffer_overflows += 1;
        }
        if bits as f64 > self.last_peak {
            self.stats.peak_overflows += 1;
        }
        if self.current == self.settings.max && bits as f64 > self.last_budget {
            self.stats.target_unreachable_frames += 1;
        }
        self.occupancy = self.occupancy.clamp(0., self.settings.capacity);
        let drain = self.channel_drain();
        self.target_occupancy =
            (self.target_occupancy + bits as f64 - drain).clamp(0., self.settings.capacity);
        self.strict_occupancy = (self.strict_occupancy + bits as f64 - drain).max(0.);
        if bonus > 0.0 {
            self.target_occupancy = (self.target_occupancy - bonus).max(0.);
            self.strict_occupancy = (self.strict_occupancy - bonus).max(0.);
        }
        self.record_window(bits as f64 - bonus.max(0.), self.last_key);
        self.debit();
        // Keep actual target debt even when it exceeds the physical buffer.
        // Clipping it independently of a large scheduled key loan invents
        // negative excess (and undeserved extra P budget). Response is already
        // bounded by budget()'s correction clamp; physical VBV is separate.
        self.stats.frames += 1;
        self.stats.total_bits = self.stats.total_bits.saturating_add(bits);
        self.stats.current_quantizer = self.current;
        self.stats.buffer_fullness = (self.occupancy / self.settings.capacity).clamp(0., 1.);
        self.elapsed += 1. / self.frame_fps;
        self.stats.actual_bps = self.stats.total_bits as f64 / self.elapsed;
        self.stats.max_frame_bits = self.stats.max_frame_bits.max(bits);
        self.stats.key_loan_bits = self.scheduled_key_debt;
        self.stats.plan.excess_bits = self.excess();
        self.stats.plan.debit_bits = self.debit.unwrap_or(0.);
        self.stats.plan.capacity_bits = self.settings.capacity;
        self.stats.plan.codings = 1 + self.recodes_this_frame;
    }
    pub const fn stats(&self) -> RateControlStats {
        self.stats
    }
}

/// Smallest quantizer in `bounds.0..=bounds.1` whose step in `steps` is at
/// least `step`, or `bounds.1` when `step` exceeds every entry.
pub(crate) fn quantizer_for_step(steps: &[f64], bounds: (u32, u32), step: f64) -> u32 {
    (bounds.0..=bounds.1)
        .find(|&q| steps[q as usize] >= step)
        .unwrap_or(bounds.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// A controller whose first key gets the normal key budget, so the
    /// key-loan tests see one boost (`rate.first_key_boost_scale` 1), and
    /// whose budgets are not bounded by the strict-buffer ceiling
    /// (`rate.recode_overflow` 0) or steered by the debit term
    /// (`rate.debit` 0), so they test the loan and the proportional
    /// debt correction alone.
    fn controller(target: u64) -> RateController {
        controller_with(target, &[])
    }
    /// `controller` with extra parameter overrides.
    fn controller_with(target: u64, overrides: &[(&str, f32)]) -> RateController {
        let mut params = crate::ParamOverrides::default();
        params.set("rate.first_key_boost_scale", 1.);
        params.set("rate.recode_overflow", 0.);
        params.set("rate.debit", 0.);
        for &(name, value) in overrides {
            params.set(name, value);
        }
        RateController::new(
            RateControlConfig::new(target),
            Codec::Hevc,
            RateModel::QP,
            FrameRate { num: 30, den: 1 },
            30,
            32,
            (0..=51).map(|q| 2f64.powf(f64::from(q) / 6.)).collect(),
            640 * 360,
            &RateParams::from_overrides(&params),
            crate::LayerParams::default().rate_control_ladder(1),
        )
        .unwrap()
    }
    const HINT: Complexity = Complexity {
        spatial: 460_800.,
        temporal: 460_800.,
        static_fraction: 0.,
        block_qp_offset: 0.,
        key_reuse_offset: 0.,
        residual_sigma: None,
        motion: None,
    };
    #[test]
    fn rate_log_plan_reports_the_frame_and_the_state_after_it() {
        let mut r = controller(200_000);
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(40_000);
        let still = Complexity {
            static_fraction: 1.,
            ..HINT
        };
        let q = r.select(false, 4, still, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(5_000);
        let plan = r.stats().plan;
        assert_eq!(plan.budget_bits, r.last_budget);
        assert!(plan.predicted_bits > 0.);
        // Frame 4 is a base frame of the default four-frame cadence: its
        // quantizer is finer than the level by the full boost.
        assert!(plan.cadence_offset < 0. && q < plan.level_quantizer);
        assert_eq!(plan.static_fraction, 1.);
        assert_eq!(plan.excess_bits, r.excess());
        assert_eq!(plan.capacity_bits, r.settings.capacity);
    }
    #[test]
    fn tiny_buffer_large_key_loan_does_not_invent_credit() {
        let mut r = controller(200_000);
        let mut cfg = RateControlConfig::new(200_000);
        cfg.buffer_ms = 1;
        cfg.keyframe_boost = 64.;
        r.period = 300;
        r.reconfigure(cfg).unwrap();
        let key = r.inter_budget() * 64.;
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(key.round() as u64);
        assert!(r.scheduled_key_debt > r.settings.capacity * 100.);
        // A nominally sized key has no excess: it must not double P budgets
        // just because its actual target debt exceeds the buffer-size clamp.
        assert!((r.budget(false) / r.inter_budget() - 1.).abs() < 0.01);
        for i in 1..900 {
            r.select(
                i % 300 == 0,
                i,
                HINT,
                LayerPosition::FLAT,
                &LayerShare::WHOLE,
            );
            r.feedback(r.last_budget.round() as u64);
            let expected = r.stats.total_bits as f64 - r.uniform_budget() * (i + 1) as f64;
            assert!((r.fullness - r.settings.capacity / 2. - expected).abs() < 0.001);
        }
        assert!(r.stats.actual_bps / 200_000. < 1.01);
        let before = r.fullness - r.settings.capacity / 2.;
        cfg.target_bps = 100_000;
        r.reconfigure(cfg).unwrap();
        // The debt follows the halved target.
        assert!((r.fullness - r.settings.capacity / 2. - before / 2.).abs() < 0.001);
    }

    #[test]
    fn reserved_key_cycle_does_not_penalize_following_frames() {
        let mut r = controller(500_000);
        for i in 0..90 {
            let key = i % 30 == 0;
            let nominal = if key {
                r.loan_inter_budget() * 8.
            } else {
                r.inter_budget()
            };
            // Rounded packet lengths accumulate less than one bit per frame.
            assert!((r.budget(key) / nominal - 1.).abs() < 0.001, "frame{i}");
            r.select(key, i, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
            r.feedback(nominal.round() as u64);
            if key {
                assert!(r.scheduled_key_debt > r.uniform_budget() * 5.);
            }
        }
        assert!(r.scheduled_key_debt < 0.001);
    }

    #[test]
    fn every_key_borrows_its_plan_and_a_new_key_leaves_the_rest_as_debt() {
        let mut r = controller(500_000);
        let p = r.inter_budget();
        let planned = p * 8.;
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        let overspend = 10_000.;
        r.feedback((planned + overspend).round() as u64);
        // The plan is the loan; the overspend is corrected as debt.
        assert!((r.scheduled_key_debt - (planned - r.uniform_budget())).abs() < 1.);
        let corrected = 1. - overspend / r.settings.capacity;
        assert!((r.budget(false) / p - corrected).abs() < 0.001);
        r.select(false, 1, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(p.round() as u64);
        let unpaid = r.scheduled_key_debt;
        let before = r.fullness - r.settings.capacity / 2. - unpaid;
        // A requested key borrows its plan too; what was left of the old
        // loan becomes ordinary debt.
        r.select(true, 2, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        let planned = r.loan_inter_budget() * r.loan_boost;
        r.feedback(planned.round() as u64);
        assert!((r.scheduled_key_debt - (planned - r.uniform_budget())).abs() < 1.);
        let after = r.fullness - r.settings.capacity / 2. - r.scheduled_key_debt;
        assert!((after - before - unpaid).abs() < 1.);
    }

    #[test]
    fn period_one_has_no_key_loan_or_phase_correction() {
        let mut r = controller(500_000);
        r.period = 1;
        for i in 0..12 {
            let nominal = r.uniform_budget();
            assert!((r.budget(true) / nominal - 1.).abs() < 0.001);
            r.select(true, i, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
            r.feedback(nominal.round() as u64);
            assert!(r.scheduled_key_debt < 0.001);
        }
    }

    #[test]
    fn live_target_change_scales_the_debt_with_the_target() {
        let mut r = controller(500_000);
        let p = r.inter_budget();
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback((p * 8.).round() as u64);
        for i in 1..5 {
            r.select(false, i, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
            r.feedback(p.round() as u64);
        }
        let share = |r: &RateController| {
            let s = r.settings;
            (r.fullness - s.capacity / 2. - r.scheduled_key_debt) / s.capacity
        };
        let (debt_share, correction) = (share(&r), r.budget(false) / r.inter_budget());
        let actual_debt = r.fullness - r.settings.capacity / 2.;
        r.reconfigure(RateControlConfig::new(250_000)).unwrap();
        assert!((r.fullness - r.settings.capacity / 2. - actual_debt / 2.).abs() < 0.001);
        assert!((share(&r) - debt_share).abs() < 1e-9);
        // The same relative correction, not a doubled one.
        assert!((r.budget(false) / r.inter_budget() - correction).abs() < 1e-6);
        assert!(r.budget(false) < r.inter_budget());
        r.reconfigure(RateControlConfig::new(1_000_000)).unwrap();
        // No fabricated key credit: the excess keeps its share.
        assert!((share(&r) - debt_share).abs() < 1e-9);
    }

    #[test]
    fn enabling_mid_gop_does_not_invent_a_previous_key_reservation() {
        let mut r = controller(500_000);
        r.select(false, 15, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(r.inter_budget().round() as u64);
        assert_eq!(r.scheduled_key_debt, 0.);
    }

    #[test]
    fn first_key_responds_to_target_and_current_complexity() {
        let mut low = controller(100_000);
        let mut high = controller(2_000_000);
        assert!(
            low.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE)
                > high.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE)
        );
        let mut flat = controller(500_000);
        let mut busy = controller(500_000);
        assert!(
            busy.select(
                true,
                0,
                Complexity {
                    spatial: HINT.spatial * 4.,
                    ..HINT
                },
                LayerPosition::FLAT,
                &LayerShare::WHOLE,
            ) > flat.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE)
        );
    }
    #[test]
    fn tracks_budget_scene_spikes_and_bounded_quantizer_changes() {
        let mut r = controller(500_000);
        let mut previous = [None, None];
        let mut bits = 0u64;
        for i in 0..300 {
            let key = i % 30 == 0;
            let multiplier = if i >= 150 { 3. } else { 1. };
            let hint = Complexity {
                spatial: HINT.spatial * multiplier,
                temporal: HINT.temporal * multiplier,
                ..HINT
            };
            let q = r.select(key, i, hint, LayerPosition::FLAT, &LayerShare::WHOLE);
            let class = usize::from(key);
            if !key && let Some(old) = previous[class] {
                assert!(q.abs_diff(old) <= 4);
            }
            previous[class] = Some(q);
            let step = 2f64.powf(f64::from(q) / 6.);
            let reference = 2f64.powf(32. / 6.);
            let actual = (500_000. / 30.
                * (reference / step).powf(1.1)
                * multiplier
                * if key { 4. } else { 1. }) as u64;
            r.feedback(actual);
            if i >= 210 {
                bits += actual;
            }
        }
        let rate = bits as f64 / 3.;
        assert!((rate / 500_000. - 1.).abs() < 0.12, "{rate}");
        assert!(r.stats().buffer_fullness.is_finite());
    }
    #[test]
    fn peak_headroom_is_physical_and_live_reconfigure_keeps_history() {
        let mut r = controller(500_000);
        r.fullness = -r.settings.capacity;
        r.occupancy = 0.;
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        assert!(r.last_peak <= r.settings.capacity + r.settings.peak / r.fps);
        r.feedback(20_000);
        let old = r.stats();
        r.model[0] = Some(12.);
        let mut cfg = RateControlConfig::new(250_000);
        cfg.buffer_ms = 200;
        r.reconfigure(cfg).unwrap();
        assert_eq!(r.stats().total_bits, old.total_bits);
        assert_eq!(r.stats().frames, old.frames);
        assert_eq!(r.model[0], Some(12.));
        assert_eq!(r.stats().target_bps, 250_000);
        let before = r.stats();
        cfg.target_bps = 0;
        assert!(r.reconfigure(cfg).is_err());
        assert_eq!(r.stats(), before);
    }
    #[test]
    fn key_budget_does_not_drag_following_inter_quantizer() {
        let mut r = controller(500_000);
        r.last_inter_quantizer = Some(40);
        r.current = 20;
        r.model[INTER_MODEL] = Some(
            r.steps[40].powf(r.settings.config.rate_exponent) * r.budget(false) / HINT.temporal,
        );
        let inter = r.select(false, 1, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        assert!(inter >= 36);
        r.model[KEY_MODEL] =
            Some(r.steps[10].powf(r.settings.config.rate_exponent) * r.budget(true) / HINT.spatial);
        assert!(
            r.select(true, 30, HINT, LayerPosition::FLAT, &LayerShare::WHOLE) <= 11,
            "key budget must not wait for inter-QP ramps"
        );
        assert_eq!(r.last_inter_quantizer, Some(inter));
    }

    #[test]
    fn keys_are_never_coarser_than_the_inter_level_and_its_reuse() {
        // A deep buffer, so the physical allowance does not bind.
        let mut config = RateControlConfig::new(500_000);
        config.buffer_ms = 10_000;
        let mut r = controller(500_000);
        r.reconfigure(config).unwrap();
        // A long repayment phase, so the repayable loan does not bind either.
        r.period = 300;
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(100_000);
        r.last_inter_quantizer = Some(34);
        // A key model that, on budget alone, would code the key at `q`.
        let budget_at = |r: &mut RateController, q: usize| {
            r.model[KEY_MODEL] = Some(
                r.steps[q].powf(r.settings.config.rate_exponent) * r.budget(true) / HINT.spatial,
            );
        };
        budget_at(&mut r, 40);
        assert_eq!(
            r.select(true, 30, HINT, LayerPosition::FLAT, &LayerShare::WHOLE),
            34
        );
        let reused = Complexity {
            key_reuse_offset: -6.,
            ..HINT
        };
        assert_eq!(
            r.select(true, 60, reused, LayerPosition::FLAT, &LayerShare::WHOLE),
            28
        );
        // A budget finer than the level is spent (QP 20, give or take the
        // rounding of the model's step).
        budget_at(&mut r, 20);
        assert!(r.select(true, 90, HINT, LayerPosition::FLAT, &LayerShare::WHOLE) <= 21);
    }

    #[test]
    fn misses_on_unchanged_content_do_not_restart_the_model() {
        let mut r = controller(500_000);
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(100_000);
        // Frames 1, 3 and 5 share a cadence role, so one model slot.
        r.select(false, 1, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(10_000);
        let settled = r.model[r.current_slot].unwrap();
        // Same hint, a 10x miss: smoothed, not replaced.
        r.select(false, 3, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        let slot = r.current_slot;
        r.feedback(100_000);
        let after = r.model[slot].unwrap();
        assert!(after < settled * 4., "{settled} -> {after}");
        // A changed hint with a large miss is a scene change: replaced.
        let busy = Complexity {
            temporal: HINT.temporal * 3.,
            ..HINT
        };
        r.select(false, 5, busy, LayerPosition::FLAT, &LayerShare::WHOLE);
        let slot = r.current_slot;
        let before = r.model[slot];
        r.feedback(1_000_000);
        assert!(r.model[slot] > before.map(|m| m * 4.));
    }

    #[test]
    fn a_still_scene_key_repays_over_the_still_horizon() {
        let still = Complexity {
            static_fraction: 0.95,
            ..HINT
        };
        let half = Complexity {
            static_fraction: 0.7,
            ..HINT
        };
        let plan = |overrides: &[(&str, f32)], hint: Complexity| {
            let mut r = controller_with(500_000, overrides);
            r.period = 600;
            r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
            r.feedback(10_000);
            for i in 1..300 {
                r.select(false, i, hint, LayerPosition::FLAT, &LayerShare::WHOLE);
                r.feedback(1_000);
            }
            r.select(true, 300, hint, LayerPosition::FLAT, &LayerShare::WHOLE);
            r.repay_frames()
        };
        // Default (still horizon 0): every key repays over rate.key_repay_seconds.
        assert_eq!(plan(&[], still), 150);
        let longer = [("rate.key_repay_still_seconds", 10.)];
        // The first key has no history: the moving-scene horizon until a
        // second of inter frames is in.
        let mut first = controller_with(500_000, &longer);
        first.period = 600;
        first.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        assert_eq!(first.repay_frames(), 150);
        // A still scene's later key takes the still horizon; half-way between
        // `still_from` (0.5) and `still_full` (0.9), half of the extra time.
        assert_eq!(plan(&longer, still), 300);
        assert!((224..=226).contains(&plan(&longer, half))); // 225 up to rounding
        assert_eq!(plan(&longer, HINT), 150);
    }

    #[test]
    fn a_still_scene_replans_its_first_key_and_repays_it_in_full() {
        let still = Complexity {
            static_fraction: 0.95,
            ..HINT
        };
        let run = |overrides: &[(&str, f32)], hint: Complexity| {
            let mut r = controller_with(500_000, overrides);
            r.period = 900;
            let key = r.loan_inter_budget() * 8.;
            r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
            r.feedback(key.round() as u64);
            let mut debts = Vec::new();
            for i in 1..400 {
                r.select(false, i, hint, LayerPosition::FLAT, &LayerShare::WHOLE);
                r.feedback(r.inter_budget().round() as u64);
                debts.push(r.scheduled_key_debt);
            }
            (r.repay_frames(), debts)
        };
        let (moving, _) = run(&[("rate.key_repay_still_seconds", 10.)], HINT);
        assert_eq!(moving, 150);
        let (frames, debts) = run(&[("rate.key_repay_still_seconds", 10.)], still);
        assert_eq!(frames, 300);
        // Still owed half-way through the longer phase, repaid at its end.
        assert!(debts[200] > 0.);
        assert!(debts[310] < 1.);
        // Without a still horizon the first key is never re-planned.
        assert_eq!(run(&[], still).0, 150);
    }

    #[test]
    fn an_infinite_key_period_repays_the_key_within_the_horizon() {
        let mut r = controller(500_000);
        r.period = u32::MAX;
        let horizon = r.repay_frames();
        assert_eq!(horizon, 150); // rate.key_repay_seconds 5 at 30 fps
        let key = r.loan_inter_budget() * 8.;
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(key.round() as u64);
        assert!(r.scheduled_key_debt > r.uniform_budget() * 5.);
        for i in 1..u64::from(horizon) {
            let p = r.inter_budget();
            r.select(false, i, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
            r.feedback(p.round() as u64);
        }
        assert!(r.scheduled_key_debt < 1.);
        // After the phase, inter frames get the uniform budget again.
        assert!((r.inter_budget() / r.uniform_budget() - 1.).abs() < 1e-9);
    }

    #[test]
    fn the_first_key_gets_a_larger_budget_by_default() {
        let r = RateController::new(
            RateControlConfig::new(500_000),
            Codec::Hevc,
            RateModel::QP,
            FrameRate { num: 30, den: 1 },
            30,
            32,
            (0..=51).map(|q| 2f64.powf(f64::from(q) / 6.)).collect(),
            640 * 360,
            &RateParams::default(),
            crate::LayerParams::default().rate_control_ladder(1),
        )
        .unwrap();
        assert!((r.first_key_boost / r.settings.config.keyframe_boost - 2.5).abs() < 1e-9);
    }

    #[test]
    fn startup_camera_motion_caps_only_the_first_key() {
        // A binary fraction, exact as the f32 param.
        let cap = 0.25;
        let mut still = controller_with(500_000, &[("rate.first_key_lookahead", cap as f32)]);
        let mut moving = controller_with(500_000, &[("rate.first_key_lookahead", cap as f32)]);
        assert!(still.wants_startup_motion());
        still.observe_startup_motion(STARTUP_CAMERA_MOTION / 2.);
        moving.observe_startup_motion(STARTUP_CAMERA_MOTION * 2.);
        assert!(still.key_bits_cap().is_infinite());
        assert!((moving.key_bits_cap() - cap * moving.settings.capacity).abs() < 1e-6);
        // A later scheduled key starts a GOP of its own, uncapped.
        moving.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        moving.feedback(10_000);
        moving.select(true, 30, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        assert!(moving.key_bits_cap().is_infinite());
    }

    #[test]
    fn saturation_is_reported_instead_of_fake_padding_or_dropping() {
        let mut cfg = RateControlConfig::new(1000);
        cfg.min_quantizer = Some(51);
        cfg.max_quantizer = Some(51);
        let mut r = RateController::new(
            cfg,
            Codec::Hevc,
            RateModel::QP,
            FrameRate { num: 30, den: 1 },
            30,
            51,
            (0..=51).map(|q| 2f64.powf(f64::from(q) / 6.)).collect(),
            640 * 360,
            &RateParams::default(),
            crate::LayerParams::default().rate_control_ladder(1),
        )
        .unwrap();
        for i in 0..5 {
            assert_eq!(
                r.select(i == 0, i, HINT, LayerPosition::FLAT, &LayerShare::WHOLE),
                51
            );
            r.feedback(100_000);
        }
        assert_eq!(r.stats().frames, 5);
        assert_eq!(r.stats().total_bits, 500_000);
        assert!(r.stats().buffer_overflows > 0);
        assert!(r.stats().target_unreachable_frames > 0);
    }

    #[test]
    fn layered_structure_gives_base_frames_finer_quantizers() {
        let mut r = controller(3_000_000);
        r.ladder = crate::LayerParams::default().rate_control_ladder(3);
        let structure = LayerPosition {
            layer: 0,
            layers: 3,
            position: 0,
            period: 4,
        };
        let positions = structure.cycle();
        r.select(true, 0, HINT, structure, &LayerShare::WHOLE);
        r.feedback(60_000);
        let mut last = Vec::new();
        for i in 1..=40u64 {
            let position = positions[(i % 4) as usize];
            let q = r.select(false, i, HINT, position, &LayerShare::WHOLE);
            r.feedback(if position.layer == 0 { 20_000 } else { 8_000 });
            last.push((position.layer, q));
        }
        let mean = |layer| {
            let qs: Vec<u32> = last[20..]
                .iter()
                .filter(|e| e.0 == layer)
                .map(|e| e.1)
                .collect();
            f64::from(qs.iter().sum::<u32>()) / qs.len() as f64
        };
        assert!(mean(0) < mean(1) && mean(1) < mean(2), "{last:?}");
        assert!(r.model[INTER_MODEL + CadenceRole::Middle.index()].is_some());
    }

    #[test]
    fn frame_intervals_budget_by_time() {
        let mut r = controller(300_000);
        let nominal = r.uniform_budget();
        assert!((nominal - 10_000.).abs() < 1e-9);
        // A frame covering two nominal intervals (half rate, or one dropped
        // camera frame) gets twice the budget and drains twice the buffer.
        r.set_frame_interval(Some(2. / 30.));
        assert!((r.uniform_budget() - 2. * nominal).abs() < 1e-6);
        r.select(false, 1, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(20_000);
        assert!((r.stats().actual_bps - 300_000.).abs() < 1e-6);
        r.set_frame_interval(None);
        assert_eq!(r.uniform_budget().to_bits(), nominal.to_bits());
    }

    #[test]
    fn drops_only_when_the_maximum_quantizer_overflows() {
        let mut cfg = RateControlConfig::new(30_000);
        cfg.peak_bps = Some(30_000);
        let mut r = controller(30_000);
        r.reconfigure(cfg).unwrap();
        r.select(false, 1, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(200_000); // far above the 1000-bit drain, even at max QP
        assert!(!r.should_drop(0), "frame dropping is off by default");
        cfg.frame_drop = true;
        r.reconfigure(cfg).unwrap();
        assert!(r.should_drop(0));
        let occupancy = r.occupancy;
        r.drop_frame();
        assert!((occupancy - r.occupancy - 1_000.).abs() < 1e-6);
        assert_eq!((r.stats().dropped_frames, r.stats().frames), (1, 1));
    }

    #[test]
    fn drops_hold_the_target_rate_bucket_under_a_higher_peak() {
        let mut cfg = RateControlConfig::new(30_000);
        cfg.frame_drop = true; // default peak: twice the target
        let mut r = controller(30_000);
        r.reconfigure(cfg).unwrap();
        r.select(false, 1, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(1_500); // 1.5x the target per frame, under the peak
        let mut index = 2;
        while !r.should_drop(0) {
            r.select(false, index, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
            r.feedback(1_500);
            index += 1;
            assert!(index < 100, "never dropped");
        }
        assert_eq!(
            r.stats().buffer_overflows,
            0,
            "the peak bucket never filled"
        );
    }

    #[test]
    fn the_inter_ceiling_recodes_bursts_but_not_the_key() {
        let mut r = controller_with(
            300_000,
            &[
                ("rate.recode_overflow", 1.4),
                ("rate.recode_overflow_inter", 1.1),
            ],
        );
        // A long repayment phase, so the key's loan leaves the inter frames
        // most of their budget.
        r.period = 300;
        let capacity = r.settings.capacity;
        // A key at 1.2 buffers stays (only the 1.4 ceiling applies to keys).
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        assert_eq!(r.recode_quantizer((1.2 * capacity) as u64), None);
        r.feedback((1.2 * capacity) as u64);
        // The next inter frame may still take its drain, above the ceiling.
        r.select(false, 1, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        assert!(r.strict_room(false) >= r.uniform_budget());
        assert_eq!(r.recode_quantizer(r.uniform_budget() as u64), None);
        // A burst past 1.1 buffers is re-coded coarser.
        let q = r.current;
        let recoded = r.recode_quantizer((0.5 * capacity) as u64);
        assert!(recoded.is_some_and(|coarser| coarser > q));
    }
    #[test]
    fn exhausted_refinement_stops_the_level_getting_finer() {
        let mut r = controller_with(
            300_000,
            &[
                ("rate.min_refinement_gain", 0.1),
                ("rate.level_step_scale", 1.),
            ],
        );
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(20_000);
        let mut previous = None;
        for i in 1..120 {
            r.select(false, i, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
            // Nearly free inter frames whose quality never changes.
            r.feedback(50);
            r.observe_distortion(10.);
            let level = r.last_inter_quantizer;
            if r.refinement.exhausted() {
                assert!(level >= previous, "frame {i}: {level:?} < {previous:?}");
            }
            previous = level;
        }
        assert!(r.refinement.exhausted());
    }

    #[test]
    fn capped_underspend_is_banked_for_at_most_one_buffer() {
        let mut r = controller_with(300_000, &[("rate.max_quality_qp", 30.)]);
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(20_000);
        for i in 1..300 {
            r.select(false, i, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
            r.feedback(50);
        }
        let limit = r.settings.capacity / 2. + r.scheduled_key_debt - r.settings.capacity;
        assert!(r.fullness >= limit - 1e-6, "{} < {limit}", r.fullness);
    }

    #[test]
    fn no_frame_is_finer_than_the_max_quality_qp() {
        // HEVC steps: H.264 QP 24 is quantizer 24 here.
        let mut r = controller_with(3_000_000, &[("rate.max_quality_qp", 24.)]);
        assert!(r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE) >= 24);
        r.feedback(1_000);
        for i in 1..30 {
            assert!(r.select(false, i, HINT, LayerPosition::FLAT, &LayerShare::WHOLE) >= 24);
            r.feedback(100);
        }
    }

    #[test]
    fn debit_term_repays_a_lasting_overshoot() {
        // Every inter frame comes in 10% over its budget: the proportional
        // correction alone settles at a positive debt (the stream stays
        // above the target line); with the debit term the excess returns
        // to about zero.
        let run = |debit: f32| {
            let mut r = controller_with(300_000, &[("rate.debit", debit)]);
            r.period = 300;
            r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
            r.feedback((r.loan_inter_budget() * r.loan_boost).round() as u64);
            for i in 1..900 {
                r.select(false, i, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
                r.feedback((r.last_budget * 1.1).round() as u64);
            }
            r.excess() / r.settings.capacity
        };
        let proportional = run(0.);
        let debit = run(1.);
        assert!(proportional > 0.05, "{proportional}");
        assert!(debit.abs() < 0.01, "{debit}");
    }

    #[test]
    fn debit_term_banks_little_undershoot() {
        let mut r = controller_with(300_000, &[("rate.debit", 1.), ("rate.debit_credit", 0.25)]);
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(20_000);
        for i in 1..300 {
            r.select(false, i, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
            r.feedback(50);
        }
        assert!(r.excess() >= -0.25 * r.settings.capacity - 1e-6);
        assert!(r.budget(false) <= r.inter_budget() * 1.25 + 1e-6);
    }

    #[test]
    fn short_window_bounds_inter_frames_and_exempts_keys() {
        let mut r = controller_with(
            300_000,
            &[("rate.window_seconds", 1.), ("rate.window_tolerance", 0.1)],
        );
        r.period = 300;
        let drain = r.uniform_budget();
        // A key of most of a buffer leaves the frames after it their whole
        // window: its excess is a loan, not window spend.
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        assert!(r.recode_quantizer(250_000).is_none());
        r.select(true, 0, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(250_000);
        let room = r.strict_room(false);
        assert!((room - (330_000. - drain)).abs() < 1., "{room}");
        // An inter frame of half a buffer then leaves the next ones the
        // rest of the window's allowance.
        r.select(false, 1, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        assert!(r.budget(false) <= room);
        assert!(r.current == r.settings.max || r.recode_quantizer(340_000).is_some());
        r.select(false, 1, HINT, LayerPosition::FLAT, &LayerShare::WHOLE);
        r.feedback(150_000);
        let room = r.strict_room(false);
        assert!((room - (330_000. - drain - 150_000.)).abs() < 1., "{room}");
    }
}
