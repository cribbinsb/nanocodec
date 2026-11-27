//! The HEVC [`CodecBackend`]: IDR pictures and P pictures of 2Nx2N CUs
//! (and optionally `PART_NxN` 8x8 intra CUs).
//!
//! Per frame: one access unit holding one slice segment that covers the
//! picture — an IRAP I slice (preceded by VPS, SPS and PPS: `IDR_W_RADL`
//! for the first key frame and any that changes the SPS, otherwise `CRA_NUT`,
//! see `headers::irap_slice_header`) or a
//! `TRAIL_R` P slice predicted from the previous picture. Per CTU,
//! `coding_quadtree()` (HEVC §7.3.8.4); per CU, `coding_unit()` (§7.3.8.5)
//! with a single 2Nx2N prediction unit:
//! - intra: planar, DC or angular luma (SATD shortlist), chroma in DM mode,
//!   optionally four 4x4 prediction units and split transform trees
//!   (`backend/intra.rs`);
//! - inter: skip (merge, no residual), merge with residual, or AMVP with a
//!   motion vector difference, one reference picture.
//!
//! Transform units equal the CU up to the configured16x16 or32x32 maximum;
//! larger CUs split implicitly. An optional inter heuristic further splits
//! diffuse chroma residuals to16x16 luma TUs (§7.3.8.8), and optional
//! rate-distortion trees split any inter TU down to 8x8
//! (`backend/inter_tree.rs`).
//!
//! Decisions and reconstruction happen during the engine's partition walk;
//! the syntax is only recorded ([`crate::syntax`]). After the last CU the
//! deblocking filter and SAO run, and then the slice data is entropy coded
//! with each CTU's SAO parameters in front of it.

use std::cell::RefCell;

use nc_common::{BitWriter, Picture, PlaneIndex, PlaneView, PlaneViewMut, nal};
use nc_dsp::{
    BlockSize, Dsp, HevcForwardMode, HevcForwardStaging, HevcQuantizer, HevcTransform, Residual16,
};
use nc_encoder::chroma_policy::ChromaPolicy;
use nc_encoder::{
    Block, BlockDecision, CodecBackend, CodedGroups, ConfigError, EncoderConfig, FramePlan,
    FrameReferences, FrameType, IntraChoice, MAX_REFERENCE_SLOTS, Mv, PlaneZeroLimits,
    PredictionMode, QuantParams, ReferenceChoice, RefreshWindow, StaticSkipTest, TrellisSettings,
    ZeroLimits, copy_zero_vector_prediction, holds_second_reference, long_term_slot, luma_level,
    reference_slots, uses_long_term_slot, zero_vector_residual_is_zero,
};

use crate::collocated::MotionStore;
use crate::contexts::{Contexts, SliceKind};
use crate::deblock::{Strength, deblock};
use crate::grid::{Cell, Grid};
use crate::headers::ReferenceSet;
use crate::headers::{
    CraPoc, KeptSlot, LOG2_CTB, LOG2_MAX_TB, LOG2_MIN_CB, LONG_POC_LSB_BITS, LongTermEntry,
    NalType, POC_LSB_BITS, SIGN_DATA_HIDING, SliceDeblocking, SliceQp, SliceReferenceSet,
    StreamInfo, inter_slice_header, irap_slice_header, nal_header, pps, sps, vps,
};
use crate::inter::{Component, predict_bi_into, predict_chroma, predict_luma, predict_samples};
use crate::mvpred::{
    AMVP_CANDIDATES, DEFAULT_MERGE_CANDIDATES, List, ListMotion, MAX_MERGE_CANDIDATES, MergeList,
    Motion, ReferenceDistances, SliceLists, amvp_candidates, candidate_lists,
};
use crate::predict::{Refs, Sides, most_probable};
use crate::quant;
use crate::quant::{
    INTER_ROUNDING_512THS, INTRA_ROUNDING_512THS, chroma_qp, hide_signs, largest_zero_coefficient,
    shave_isolated_ones,
};
use crate::rate::BinCosts;
use crate::residual_trellis::{BlockQuantizer, RateModel, TrellisQuantizer};
use crate::sao::{self, Candidates, CtbTypes, PictureSao};
use crate::scan::{Scan, ScanKind, Scans, coded_sub_blocks_counted};
use crate::syntax::{
    AmvpList, CuKind, CuRecord, Item, LevelStore, MAX_TUS, MVD_EG_ORDER, PlaneResidual,
    TransformSyntax, TransformUnit, TransformUnits, TuRange, split_ctx_inc, split_flag_present,
    write_slice_data,
};
use crate::transform::{MAX_COEFFS, forward_bound, inverse_dc};

mod block_qp;
mod coded_snapshot;
mod cu_trial;
mod final_cost;
mod inter_tree;
mod intra;
mod key_cost;
mod leaf_alternatives;
mod node_trial;
mod rd;
mod stress;
use block_qp::{BlockQp, QuantizationSettings};
use stress::RandomSyntax;

/// Side of the minimum (4x4) luma transform block: the unit of z-scan
/// availability (§6.4.1).
const MIN_TB_SIDE: usize = 4;
/// `max_transform_hierarchy_depth_inter` when inter CUs may split their
/// transforms: two levels keep every CU within `syntax::MAX_TUS` units.
const MAX_INTER_TRANSFORM_DEPTH: u8 = 2;
/// Luma samples of the largest CU (the CTU).
const CTU_SAMPLES: usize = 1 << (2 * LOG2_CTB);
/// Largest leaf CU: the CTU.
const MAX_BLOCK_LOG2: u8 = LOG2_CTB as u8;
/// HEVC's rate model: the shared QP calibration, plus the start-up
/// lookahead's first-key cap (`rate.first_key_lookahead`) of 0.5 buffers
/// under camera motion, looser than H.264's 0.3. `J_real` TRAIN, 150 f, vs
/// off: 0.5 -0.1% (pan -0.6%, worst-1% Y -0.5%, no clip worse); 0.3 -0.1%
/// with 5 of 18 clips worse at low rate (texture +0.2%).
///
/// QP cadence 1.2x the engine's, re-tuned once MS-SSIM was scored on every
/// frame instead of every 4th (which saw mostly base frames). `J_real`
/// (k 0.1) vs 1.0: TRAIN 18 clips -1.9%, VAL 11 clips -1.1%, held-out
/// (`broad_independent_v1`, 13 clips, every class) -0.3%, no guard; 1.4
/// gains more on the mean but trips worst-1% VMAF (+8% proxy-6:
/// `static_cam22` +27%), 0.5 WORSE (+6%). HEVC Fast (k 0.6): TRAIN -1.3%,
/// VAL -1.1%; the old hevc-better preset: TRAIN -0.1%, VAL -0.6%. Both VALs sit on the
/// worst-1% VMAF guard (+2.0 / +2.1%): the stronger cadence's coarser leaf
/// windows (`lowlight_pp22_049`, `pan_bdd_00cef86b` +7%), with 8 of 11 clips
/// better on the mean and no flicker or pulse change; adopted. Long GOP
/// (900 f, one key): chirla, crowd and pan within -0.01..+0.21 dB Y per
/// third.
const RATE_MODEL: nc_encoder::RateModel = nc_encoder::RateModel {
    cadence_scale: 1.2,
    first_key_lookahead: 0.5,
    first_key: Some(nc_encoder::first_key_model::FirstKeyCalibration {
        weights: [
            0.640_684, 0.926_629, 0.12901, 0.476_181, 0.152_473, -0.77976, 0.420_794, 0.455_062,
            0.398_692,
        ],
        fast_bias: 0.000_763,
    }),
    ..nc_encoder::RateModel::QP
};
/// QP range of 8-bit HEVC.
const QP_MIN: u32 = 0;
const QP_MAX: u32 = 51;
/// Lagrange multiplier at QP 12 in squared-error units per bit; it doubles
/// every 3 QP (the HM model, λ = 0.57 · 2^((QP − 12) / 3)).
const LAMBDA_AT_QP12: f32 = 0.57;
/// Quantiser rounding offsets are in 512ths of a step.
const ROUNDING_UNITS_PER_STEP: f32 = 512.0;
/// Fixed-QP key-frame QP offset at the 60-frame reference period
/// (`nc_encoder::fixed_qp_key_offset`). Fitted to the quick clips with SAO
/// on: 12 is best at 60 frames (2 s), 15 at 300 frames (10 s).
const KEY_QP_OFFSET_AT_REFERENCE: f64 = 12.0;
/// Approximate signalling cost of the CU-level syntax, in bits, for mode
/// ranking only.
mod cost {
    /// `cu_skip_flag` (P slices).
    pub const SKIP_FLAG: f32 = 1.0;
    /// `pred_mode_flag` + `part_mode`.
    pub const MODE_AND_PART: f32 = 2.0;
    /// `merge_flag` or `mvp_l0_flag`.
    pub const FLAG: f32 = 1.0;
    /// DC intra: MPM flag and index, chroma mode.
    pub const INTRA_DC: f32 = 4.0;
    /// `inter_pred_idc` of a B-slice AMVP unit: one bin for `PRED_BI`, two
    /// for `PRED_L0` (§9.3.4.2.2).
    pub const INTER_PRED_IDC_BI: f32 = 1.0;
    pub const INTER_PRED_IDC_UNI: f32 = 2.0;
    /// `ref_idx_l0` of a slice with two references (§9.3.3.2 truncated
    /// rice, cMax 1): one context-coded bin.
    pub const REF_IDX: f32 = 1.0;
    /// `merge_idx` (§9.3.3.2 truncated rice, cMax `candidates` − 1): one
    /// bin per index step, the last index needing no terminating bin.
    pub fn merge_idx(idx: usize, candidates: usize) -> f32 {
        (idx + 1).min(candidates - 1) as f32
    }
}

/// Predictor lists of the block the engine is currently deciding: the
/// merge list, and the AMVP list of reference `ref_idx`.
#[derive(Debug, Clone, Copy)]
struct Predictors {
    x: usize,
    y: usize,
    log2: u8,
    ref_idx: u8,
    merge: MergeList,
    amvp: [Mv; AMVP_CANDIDATES],
}

impl Predictors {
    /// Whether these are the lists of `block` predicted from `ref_idx`.
    fn of(&self, block: &Block, ref_idx: u8) -> bool {
        (self.x, self.y, self.log2, self.ref_idx) == (block.x, block.y, block.log2_size(), ref_idx)
    }
}

/// The predictor lists of the block being decided, one per reference
/// index: the engine switches references within a block
/// (`CodecBackend::select_reference`) and back, and each switch would
/// otherwise derive the lists again.
#[derive(Debug, Clone, Copy, Default)]
struct PredictorCache {
    entries: [Option<Predictors>; crate::collocated::MAX_REF_IDX],
    /// The reference index last stored.
    last: Option<u8>,
}

impl PredictorCache {
    /// The lists of `block` for `ref_idx`, if cached.
    fn get(&self, block: &Block, ref_idx: u8) -> Option<Predictors> {
        self.entries[usize::from(ref_idx)].filter(|p| p.of(block, ref_idx))
    }

    /// As [`Self::get`], by reference.
    fn get_ref(&self, block: &Block, ref_idx: u8) -> Option<&Predictors> {
        self.entries[usize::from(ref_idx)]
            .as_ref()
            .filter(|p| p.of(block, ref_idx))
    }

    /// The lists last stored.
    fn latest(&self) -> Option<Predictors> {
        self.last.and_then(|r| self.entries[usize::from(r)])
    }

    /// Keep `p`; lists of another block are dropped.
    fn set(&mut self, p: Predictors) {
        for entry in &mut self.entries {
            if entry.is_some_and(|e| (e.x, e.y, e.log2) != (p.x, p.y, p.log2)) {
                *entry = None;
            }
        }
        self.entries[usize::from(p.ref_idx)] = Some(p);
        self.last = Some(p.ref_idx);
    }

    fn clear(&mut self) {
        *self = Self::default();
    }
}

/// Frame-constant quantisation and decision weights. A plane has one QP;
/// intra/inter dead zones and four transform sizes are prepared once.
/// Exact bodycam/crowd P instruction savings:0.23%/0.33% at QP28/39.
#[derive(Debug)]
struct FrameQuantization {
    lambda: f32,
    sqrt_lambda: f32,
    planes: [PlaneQuantization; 2],
}

#[derive(Debug)]
struct PlaneQuantization {
    qp: i32,
    transforms: [[TransformQuantization; 4]; 2],
    /// Largest orthonormal inter coefficient that quantises to zero
    /// (`hevc.still_skip`'s zero test).
    inter_zero_limit: f32,
}

#[derive(Debug, Clone, Copy)]
struct TransformQuantization {
    quantizer: HevcQuantizer,
    zero_limit: u64,
    pre_transform_sad_limit: u32,
    column_zero_limit: u32,
    /// Smallest level kept: 1 normally; past QP 51 the smallest level a
    /// quantiser at the planned step (`headroom_step` times the coded one,
    /// `FramePlan::lambda_scale`) would keep, `ceil((1 − rounding) ·
    /// headroom_step)`.
    planned_min_level: i32,
}

impl TransformQuantization {
    /// Quantise `coeffs` into `levels` with the dead zone of the planned
    /// step (past QP 51); returns the number of non-zero levels. The coding
    /// path and the key-frame cost estimate share it, so decisions are
    /// priced as blocks are coded.
    fn quantize(&self, dsp: &Dsp, coeffs: &[i16], levels: &mut [i16]) -> u32 {
        let nonzero = dsp.hevc_quantize_from(coeffs, &self.quantizer, levels);
        if self.planned_min_level > 1 && nonzero > 0 {
            nc_encoder::apply_planned_dead_zone(levels, self.planned_min_level) as u32
        } else {
            nonzero
        }
    }
}

impl FrameQuantization {
    /// The quantisation of `params` at `qp` for a frame planned at its
    /// coded quantizer (no λ headroom).
    fn without_headroom(
        qp: i32,
        chroma_offset: i32,
        rounding: [(i32, i32); 2],
        params: &HevcParams,
    ) -> Self {
        Self::new(
            qp,
            chroma_offset,
            rounding,
            params.lambda_scale,
            params.column_zero_scale,
            params.pre_transform_zero_scale,
            1.0,
        )
    }

    fn new(
        qp: i32,
        chroma_offset: i32,
        rounding: [(i32, i32); 2],
        lambda_scale: f32,
        column_zero_scale: f32,
        pre_transform_zero_scale: f32,
        headroom_step: f32,
    ) -> Self {
        let lambda = lambda_scale * LAMBDA_AT_QP12 * 2f32.powf((qp - 12) as f32 / 3.0);
        Self {
            lambda,
            sqrt_lambda: lambda.sqrt(),
            planes: [
                (qp, rounding[0]),
                (chroma_qp(qp + chroma_offset), rounding[1]),
            ]
            .map(|(qp, rounding)| PlaneQuantization {
                qp,
                inter_zero_limit: quant::orthonormal_zero_limit(qp, rounding.1),
                transforms: std::array::from_fn(|intra| {
                    let rounding = [rounding.1, rounding.0][intra];
                    std::array::from_fn(|index| {
                        let log2 = index as u8 + 2;
                        let zero_limit = largest_zero_coefficient(qp, log2, rounding);
                        let keep = 1.0 - rounding as f32 / ROUNDING_UNITS_PER_STEP;
                        TransformQuantization {
                            planned_min_level: nc_encoder::planned_min_level(keep, headroom_step),
                            column_zero_limit: (zero_limit as f32 * column_zero_scale) as u32,
                            quantizer: HevcQuantizer::with_rounding(qp as u8, log2, rounding),
                            zero_limit,
                            pre_transform_sad_limit: crate::transform::forward_zero_sad_limit(
                                log2,
                                if intra == 0 {
                                    (zero_limit as f64 * f64::from(pre_transform_zero_scale)) as u64
                                } else {
                                    zero_limit
                                },
                            ),
                        }
                    })
                }),
            }),
        }
    }
}

/// The forward quantiser rounding offsets (`hevc.*_rounding`), intra and
/// inter, for luma and for chroma.
fn rounding_of(params: &HevcParams) -> [(i32, i32); 2] {
    let intra = params.intra_rounding.round() as i32;
    [
        (intra, params.inter_rounding.round() as i32),
        (intra, params.chroma_inter_rounding.round() as i32),
    ]
}

/// The SAO components and their decision settings (`hevc.sao_*`).
fn sao_components_of(params: &HevcParams) -> sao::Components {
    sao::Components {
        luma: params.sao_luma >= 0.5,
        chroma: params.sao_chroma >= 0.5,
        luma_lambda_scale: f64::from(params.sao_luma_lambda_scale),
        stats_row_step: params.sao_stats_row_step.round() as usize,
        skip_unchanged: params.sao_skip_unchanged >= 0.5,
        search: sao::TypeSearch {
            band: params.sao_band >= 0.5,
            interval: params.sao_search_interval.round().max(1.0) as u64,
        },
    }
}

/// The `hevc.*` settings of non-base frames: the stream's, overlaid with
/// its `nonbase.hevc.*` values (`frame_role.rs`).
fn nonbase_params(config: &EncoderConfig) -> HevcParams {
    let (nonbase, ordinary) = config.params.split_prefix(nc_encoder::NONBASE_PREFIX);
    HevcParams::from_overrides(&ordinary.overlaid_with(&nonbase))
}

/// `hevc.still_skip_limit_scale` while `hevc.still_skip` is on, else 0.
/// How inter pictures are coded: generalised B slices when the engine tries
/// low-delay bi-prediction (`decision.bi_pair`), else P slices.
fn inter_slice_of(params: &nc_encoder::ParamOverrides) -> SliceKind {
    if nc_encoder::DecisionParams::from_overrides(params).bi_pair >= 0.5 {
        SliceKind::Bi
    } else {
        SliceKind::Predicted
    }
}

/// The per-picture slice type of a stream with B slices (`hevc.b_slice_*`):
/// P pictures where bi-prediction did not pay in the last B picture.
#[derive(Debug, Clone, Copy)]
struct BSchedule {
    min_share: f32,
    retry: u32,
    /// P pictures still to code before the next B picture.
    p_left: u32,
}

impl BSchedule {
    fn new(params: &HevcParams) -> Self {
        let mut schedule = Self {
            min_share: 0.0,
            retry: 0,
            p_left: 0,
        };
        schedule.configure(params);
        schedule
    }

    fn configure(&mut self, params: &HevcParams) {
        self.min_share = params.b_slice_min_share;
        self.retry = params.b_slice_retry.round() as u32;
    }

    /// The next inter picture's slice type, for a stream coding `inter`.
    fn slice(&self, inter: SliceKind) -> SliceKind {
        if inter == SliceKind::Bi && self.p_left > 0 {
            SliceKind::Predicted
        } else {
            inter
        }
    }

    /// An inter picture of a B stream was committed as `slice`, with
    /// `bi_share` of its inter area bi-predicted.
    fn coded(&mut self, slice: SliceKind, bi_share: f32) {
        if slice == SliceKind::Bi {
            if bi_share < self.min_share {
                self.p_left = self.retry;
            }
        } else {
            self.p_left = self.p_left.saturating_sub(1);
        }
    }
}

/// List 1's AMVP list of one block and reference index.
#[derive(Debug, Clone, Copy)]
struct L1Amvp {
    /// The block's `(x, y, log2)` and `ref_idx_l1`.
    key: (usize, usize, u8, u8),
    list: [Mv; AMVP_CANDIDATES],
}

fn still_skip_limit_scale(params: &HevcParams) -> f32 {
    if params.still_skip >= 0.5 {
        params.still_skip_limit_scale
    } else {
        0.0
    }
}

/// The staged (AV1-butterfly) passes of the 16- and 32-point forwards:
/// rows (`hevc.fast_forward16`, `hevc.fast_forward32`) and, on top of the
/// 32-point rows, columns (`hevc.fast_forward32_columns`); and whether the
/// exact column pass skips proven-zero column groups
/// (`hevc.forward_zero_columns`, [`HevcForwardMode::zero_columns`]); and
/// which sizes keep only their low frequencies on inter residuals
/// (`hevc.partial_forward`, [`HevcForwardMode::low_frequencies`]).
#[derive(Debug, Clone, Copy)]
struct FastForward {
    mode16: HevcForwardMode,
    mode32: HevcForwardMode,
    partial16: bool,
    partial32: bool,
}

impl FastForward {
    fn new(params: &HevcParams) -> Self {
        let zero_columns = params.forward_zero_columns >= 0.5;
        let mode = |staging| HevcForwardMode {
            staging,
            zero_columns,
            low_frequencies: false,
        };
        Self {
            // No 16-point column lever: see `fast_forward32_columns`.
            mode16: mode(Self::staging_of(params.fast_forward16, 0.0)),
            mode32: mode(Self::staging_of(
                params.fast_forward32,
                params.fast_forward32_columns,
            )),
            partial16: params.partial_forward >= 1.5,
            partial32: params.partial_forward >= 0.5,
        }
    }

    /// From the rows and columns levers of one size.
    fn staging_of(rows: f32, columns: f32) -> HevcForwardStaging {
        match (rows >= 0.5, columns >= 0.5) {
            (false, _) => HevcForwardStaging::Exact,
            (true, false) => HevcForwardStaging::Rows,
            (true, true) => HevcForwardStaging::RowsAndColumns,
        }
    }

    /// The mode of an `n`-point forward; `partial` allows the low-frequency
    /// one.
    fn mode(self, n: usize, partial: bool) -> HevcForwardMode {
        let (mode, sized_partial) = match n {
            16 => (self.mode16, self.partial16),
            32 => (self.mode32, self.partial32),
            _ => return HevcForwardMode::EXACT,
        };
        HevcForwardMode {
            low_frequencies: partial && sized_partial,
            ..mode
        }
    }
}

/// Exact engine prediction stored directly in the reusable luma CU buffer.
#[derive(Debug, Clone, Copy)]
struct CachedLuma {
    block: Block,
    motion: Motion,
    filter: nc_dsp::Filter,
    frame_index: u64,
}

/// The coded picture size for a visible `width` x `height`: whole minimum
/// coding blocks (`pic_width_in_luma_samples` must be a multiple of
/// `MinCbSizeY`, §7.4.3.2.1).
fn coded_picture_size(width: usize, height: usize) -> (usize, usize) {
    let min_cb = 1usize << LOG2_MIN_CB;
    (
        width.next_multiple_of(min_cb),
        height.next_multiple_of(min_cb),
    )
}

/// The HEVC backend.
#[derive(Debug)]
pub struct HevcBackend {
    info: StreamInfo,
    slice: SliceKind,
    /// QP of the current slice.
    qp: i32,
    /// Picture order count of the current picture (0 at each IDR; CRA
    /// pictures keep counting).
    poc: u32,
    /// Frame index of the most recent IDR picture (POC origin).
    key_index: u64,
    /// The VPS and SPS of the coded video sequence (the last IDR's); a key
    /// frame with the same ones is a CRA picture.
    sequence_sets: Option<Vec<u8>>,
    /// Whether the current key frame is an IDR picture.
    idr: bool,
    frame_index: u64,
    /// Intra-refresh window of the current frame (`CodecBackend::set_refresh`).
    refresh: Option<RefreshWindow>,
    scans: Scans,
    /// Context states used for rate estimates during decisions (the slice's
    /// initial states; the real coding pass starts from these too).
    contexts: Contexts,
    /// The slice data recorded so far.
    items: Vec<Item>,
    /// In-frame rate control's running estimate of `items`
    /// (`CodecBackend::coded_bits`), priced once each.
    item_pricing: Option<ItemPricing>,
    /// SAO decision of the current picture (made in `filter_frame`).
    sao: Option<PictureSao>,
    /// Per-CTB SAO types of the last committed SAO search frame, for reuse.
    sao_classes: Vec<CtbTypes>,
    /// This picture's searched types, which become `sao_classes` in
    /// `commit_frame` (the recode contract: a re-coded or discarded coding
    /// must not change what later pictures reuse).
    pending_sao_classes: Option<Vec<CtbTypes>>,
    /// The SAO search's outcome per CTB against the last picture's types
    /// (`nc_encoder::FilterOutcome`).
    sao_outcome: nc_encoder::OutcomeTracker<CtbTypes>,
    /// The engine's usage gate for this frame's SAO statistics
    /// (`filter.usage_*`, `nc_encoder::UsageGate`).
    sao_usage: Option<nc_encoder::UsageGate>,
    grid: Grid,
    /// Each reference slot's motion: the collocated motion for TMVP.
    motion: MotionStore,
    /// The current picture's place in the temporal structure.
    references: FrameReferences,
    /// The slice's reference list (`refs.second` adds a second entry).
    reference_list: ReferenceDistances,
    /// `ref_idx_l0` the engine is deciding or coding the current CU with
    /// (`CodecBackend::select_reference`).
    active_ref: u8,
    /// How inter pictures are coded: P slices, or with `decision.bi_pair`
    /// generalised B slices (both lists the same past pictures), so the
    /// engine may code low-delay bi-prediction
    /// ([`ReferenceChoice::PrimaryPair`], and [`ReferenceChoice::Compound`]
    /// with a second reference).
    inter_slice: SliceKind,
    /// When inter pictures fall back to P slices (`hevc.b_slice_*`).
    b_schedule: BSchedule,
    /// List 1's AMVP list of the block last priced for bi-prediction.
    l1_amvp: Option<L1Amvp>,
    /// List 0's centred 14-bit prediction of a bi-predicted CU (reused).
    bi_first: Vec<i16>,
    /// POC of the picture each reference slot holds (checks the SPS
    /// reference picture sets against the engine's structure).
    slot_poc: [u32; MAX_REFERENCE_SLOTS],
    /// The slot of the long-term reference, when one is kept.
    long_term_slot: usize,
    /// POC of the picture the stream names as a long-term picture
    /// (`long_term_ref_pics_present_flag`), once a picture has.
    long_term_poc: Option<u32>,
    next_cu: u32,
    next_tu: u32,
    current: PredictorCache,
    /// Key frames are coded this many QP finer than P frames.
    rate_control_enabled: bool,
    key_qp_offset: i32,
    /// Forward quantiser rounding offsets (1/512 step), intra and inter,
    /// for luma and for chroma.
    rounding: [(i32, i32); 2],
    /// Multiple of HM's λ (`hevc.lambda_scale`).
    lambda_scale: f32,
    /// Boxed: a block QP swaps it with a parked one per CU
    /// (`block_qp.rs`), and by value each swap moved its 792 bytes.
    frame_quantization: Box<FrameQuantization>,
    /// Per-CU QPs (`cu_qp_delta`), once the engine enables block QP.
    block_qp: Option<BlockQp>,
    column_zero_scale: f32,
    fast_forward: FastForward,
    pre_transform_zero_scale: f32,
    tu16_chroma_concentration_q16: u32,
    /// Minimum scan gap before a trailing ±1 level is dropped (0: never).
    shave_gap: usize,
    /// SAO components and luma λ multiple (`hevc.sao_*`).
    sao_components: sao::Components,
    /// Pixel kernels (motion compensation), chosen for this CPU.
    dsp: Dsp,
    /// Inter prediction of the current CU, per plane (allocated once).
    inter_pred: [Vec<u8>; 3],
    cached_luma: Option<CachedLuma>,
    /// The luma modes of the key-frame CU last asked about by
    /// [`CodecBackend::intra_alternatives`], the mode search's choice first.
    key_luma_ranking: Option<(Block, Vec<(u8, f32)>)>,
    early_merge_lambdas: f32,
    /// How P-frame intra CUs choose their luma mode
    /// (`hevc.coarse_inter_intra`, `hevc.inter_intra_dc_only`).
    inter_intra: InterIntraSearch,
    /// How key-frame intra CUs choose their luma mode: `Full`, or `Coarse`
    /// (`hevc.key_intra_coarse`).
    key_intra: InterIntraSearch,
    early_merge: Option<EarlyMergeHint>,
    /// `hevc.still_skip` and its limit scale (0 = off).
    still_skip_limit_scale: f32,
    /// The CU `try_static_skip` accepted: the next `encode_block` codes it
    /// with no residual.
    still_skipped: Option<(usize, usize)>,
    /// The encoder information SEI RBSP written after every IRAP
    /// picture's parameter sets (`CodecBackend::set_encoder_info`).
    encoder_info: Option<Vec<u8>>,
    /// Transform-block scratch (allocated once; only the `n * n` prefix a
    /// block uses is touched, so no per-block clearing of 32x32 buffers).
    scratch: RefCell<Scratch>,
    /// Levels of every coded transform block of the picture.
    level_store: RefCell<LevelStore>,
    /// Transform units of every recorded CU of the picture, in coding order
    /// (each CU's [`TuRange`] points here; cleared per picture).
    tu_store: Vec<TransformUnit>,
    /// The current slice's deblocking control ([`SliceDeblocking::Fixed`]
    /// unless `hevc.deblock_inter` is 0 or in the syntax stress test).
    deblocking: SliceDeblocking,
    /// `hevc.deblock_inter`: P slices are deblocked.
    deblock_inter: bool,
    /// `debug.random_decisions` only (`CodecBackend::enable_random_decisions`):
    /// random syntax choices replace the backend's own.
    random: Option<RandomSyntax>,
    /// The final decision stage's pricing scratch (`final_cost.rs`).
    final_scratch: final_cost::FinalScratch,
    /// Bin costs of the CABAC rate estimates (`crate::rate`).
    bin_costs: BinCosts,
    /// The optional intra/inter coding tools.
    tools: CodingTools,
    /// `quant.trellis`: trellis quantisation of every coded transform block.
    trellis: Option<TrellisQuant>,
    /// `quant.rd_trim`: rate-aware trimming of the coded (winning) blocks'
    /// plain luma levels, where the trellis would run, while it is off.
    rd_trim: Option<nc_encoder::RateTrim>,
    /// While the frame has light superblocks (`set_light_params`), the
    /// per-block settings not in use.
    alternate: Option<Alternate>,
    /// The engine's whole-CU pricing trial in progress (`cu_trial.rs`).
    cu_trial: cu_trial::CuTrial,
    /// The partition search's open node trials (`backend/node_trial.rs`).
    node_trials: node_trial::NodeTrials,
    /// The luma mode search of intra CUs coded in a pricing trial
    /// (`hevc.trial_full_intra`: the full one), or `None` to keep the
    /// P-frame search.
    trial_intra_search: Option<InterIntraSearch>,
    /// `hevc.intra_rd_candidates`: luma modes of the full intra search
    /// priced by transform cost (1: the SATD-best is taken).
    intra_rd_candidates: usize,
    /// The staged intra decision's first cut (`decision.intra_stage_*`):
    /// where on, it replaces `intra_rd_candidates`.
    intra_stage: nc_encoder::IntraStage,
    /// Shared quantizer gates (`gate.partial_transform_min_quantizer`).
    gates: nc_encoder::GateParams,
}

/// The settings HEVC reads per CU, swapped between light and regular
/// superblocks (`nc_encoder` `effort_class.rs`).
#[derive(Debug)]
struct BlockSettings {
    tools: CodingTools,
    trellis: Option<TrellisQuant>,
    fast_forward: FastForward,
    column_zero_scale: f32,
    pre_transform_zero_scale: f32,
    shave_gap: usize,
    early_merge_lambdas: f32,
    inter_intra: InterIntraSearch,
}

impl BlockSettings {
    /// The light settings of `overrides`. The parameter sets fix whether
    /// transform trees and `NxN` intra may be coded at all (`regular`'s
    /// tools): a light CU may drop those searches, never add them.
    fn light(overrides: &nc_encoder::ParamOverrides, regular: &CodingTools) -> Self {
        let params = HevcParams::from_overrides(overrides);
        let light = CodingTools::new(&params);
        Self {
            tools: CodingTools {
                inter_tu_split: light.inter_tu_split && regular.inter_tu_split,
                inter_tu_split_spread: light.inter_tu_split_spread,
                inter_tu_split_min_cost: light.inter_tu_split_min_cost,
                intra_nxn: light.intra_nxn && regular.intra_nxn,
                intra_nxn_min_cost: light.intra_nxn_min_cost,
                ..*regular
            },
            trellis: TrellisQuant::from_params(&QuantParams::from_overrides(overrides)),
            fast_forward: FastForward::new(&params),
            column_zero_scale: params.column_zero_scale,
            pre_transform_zero_scale: params.pre_transform_zero_scale,
            shave_gap: params.shave_gap.round() as usize,
            early_merge_lambdas: params.early_merge_lambdas,
            inter_intra: InterIntraSearch::new(&params),
        }
    }
}

/// The per-CU settings the backend is not coding with: the light ones while
/// it codes a regular superblock, the regular ones while it codes a light
/// one.
#[derive(Debug)]
struct Alternate {
    settings: BlockSettings,
    /// Whether the backend's own settings are the light ones.
    light_in_use: bool,
}

/// Which quantiser codes a transform block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Quantisation {
    /// The plain dead-zone quantiser: a rate-distortion trial whose result
    /// may be discarded.
    Plain,
    /// The block as it will be coded: the trellis where `quant.trellis`
    /// enables it for the slice and plane (winner only, as x264's
    /// `trellis 1`; trials keep the plain quantiser).
    Final,
}

/// The QP whose quantisation step is one sample (`Qstep = 2^((qp − 4) / 6)`,
/// HEVC §8.6.2 with flat scaling lists).
const QSTEP_UNIT_QP: i32 = 4;

/// The registered trellis controls (`nc_encoder::QuantParams`) and the
/// search's buffers.
#[derive(Debug)]
struct TrellisQuant {
    settings: TrellisSettings,
    search: RefCell<TrellisQuantizer>,
}

impl TrellisQuant {
    fn from_params(quant: &QuantParams) -> Option<Self> {
        TrellisSettings::from_params(quant).map(|settings| Self {
            settings,
            search: RefCell::default(),
        })
    }
}

/// Where a transform block's prediction comes from, and so what an
/// uncoded block's reconstruction needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Predicted {
    /// Intra prediction: an uncoded block copies its prediction.
    Intra,
    /// Inter prediction: an uncoded block copies its prediction.
    Inter,
    /// Inter prediction already placed in the reconstruction for the whole
    /// CU: an uncoded block leaves it there.
    InterInPlace,
}

/// The luma mode search of intra CUs (P frames; key frames use `Full` or
/// `Coarse`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InterIntraSearch {
    /// The key frames' search (angular sweep and refinement).
    Full,
    /// Planar, DC, pure horizontal/vertical and the MPMs; no trials.
    Coarse,
    /// DC (the mode the engine's intra test priced), no search or trials,
    /// for CUs up to `2^max_log2`; larger CUs search as [`Self::Coarse`].
    /// With `smooth` (`hevc.inter_intra_smooth`) the CUs up to that size
    /// take planar or DC, the cheaper by SATD, otherwise as DC.
    DcOnly { max_log2: u8, smooth: bool },
}

impl InterIntraSearch {
    /// The key-frame search (`hevc.key_intra_coarse`).
    fn key(params: &HevcParams) -> Self {
        if params.key_intra_coarse >= 0.5 {
            Self::Coarse
        } else {
            Self::Full
        }
    }

    fn new(params: &HevcParams) -> Self {
        if params.inter_intra_dc_only >= 0.5 {
            Self::DcOnly {
                max_log2: params.inter_intra_dc_only_max_log2.round() as u8,
                smooth: params.inter_intra_smooth >= 0.5,
            }
        } else if params.coarse_inter_intra >= 0.5 {
            Self::Coarse
        } else {
            Self::Full
        }
    }
}

/// The optional coding tools (`hevc.intra_*`, `hevc.inter_tu_*`).
#[derive(Debug, Clone, Copy)]
struct CodingTools {
    /// `hevc.intra_leaf_cost`: key-frame CU sizes are planned.
    intra_leaf_cost: bool,
    /// `hevc.intra_nxn` and its cost gate (λ per sample).
    intra_nxn: bool,
    intra_nxn_min_cost: f32,
    /// `hevc.intra_tu_split_spread` and `_min_cost` (the depth is in the
    /// SPS).
    intra_tu_split_spread: f32,
    intra_tu_split_min_cost: f32,
    /// `hevc.inter_tu_split` and its spread gate.
    inter_tu_split: bool,
    inter_tu_split_spread: f32,
    inter_tu_split_min_cost: f32,
}

impl CodingTools {
    fn new(params: &HevcParams) -> Self {
        Self {
            intra_leaf_cost: params.intra_leaf_cost >= 0.5,
            intra_nxn: params.intra_nxn >= 0.5,
            intra_nxn_min_cost: params.intra_nxn_min_cost,
            intra_tu_split_spread: params.intra_tu_split_spread,
            intra_tu_split_min_cost: params.intra_tu_split_min_cost,
            inter_tu_split: params.inter_tu_split >= 0.5,
            inter_tu_split_spread: params.inter_tu_split_spread,
            inter_tu_split_min_cost: params.inter_tu_split_min_cost,
        }
    }
}

/// A confidence hint belongs to exactly one integer merge prediction.
#[derive(Debug, Clone, Copy)]
struct EarlyMergeHint {
    frame_index: u64,
    block: Block,
    motion: Motion,
    sse_per_sample: f32,
}

impl EarlyMergeHint {
    fn matches(self, frame_index: u64, block: &Block, motion: Motion) -> bool {
        self.frame_index == frame_index && self.block == *block && self.motion == motion
    }
}

/// Working buffers of [`HevcBackend::code_residual`].
#[derive(Debug)]
struct Scratch {
    /// The luma prediction the smooth P-frame intra choice
    /// (`hevc.inter_intra_smooth`) built for the first transform block of
    /// its CU, at `(x, y, n)` with `mode`: that block's coding takes it
    /// instead of gathering the references and predicting again.
    smooth_prediction: Option<((usize, usize, usize), u8)>,
    smooth_samples: Box<[u8; MAX_COEFFS]>,
    residual: Box<[i16; MAX_COEFFS]>,
    levels: Box<[i16; MAX_COEFFS]>,
    coeffs: Box<[i16; MAX_COEFFS]>,
    scaled: Box<[i16; MAX_COEFFS]>,
    decoded: Box<[i16; MAX_COEFFS]>,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            smooth_prediction: None,
            smooth_samples: Box::new([0; MAX_COEFFS]),
            residual: Box::new([0; MAX_COEFFS]),
            levels: Box::new([0; MAX_COEFFS]),
            coeffs: Box::new([0; MAX_COEFFS]),
            scaled: Box::new([0; MAX_COEFFS]),
            decoded: Box::new([0; MAX_COEFFS]),
        }
    }
}

nc_encoder::param_group! {
    /// HEVC backend heuristics: quantiser dead zones, chroma QP and the
    /// in-loop filters' strength (perceptual vs PSNR trade-offs).
    pub struct HevcParams ("hevc") {
        // Optional TU16 trades transform work for compression: native CPU
        // -20.64%/-9.95% (bodycam/crowd); full TRAIN BD +4.886% PSNR,
        // -0.673% VMAF. Bodycam PSNR class +14.19%; keep TU32 default.
        max_transform_log2: LOG2_MAX_TB as f32, [4.0, LOG2_MAX_TB as f32],
            "largest luma transform log2:4 uses16x16,5 uses32x32; smaller trades compression for transform speed",
        tu16_chroma_concentration: 0.5, [0.0, 1.0],
            "optional inter TU16 when chroma low-frequency energy fraction is below threshold;0 disables",
        // Default -3. Staged TRAIN proxy-6 (150f) PSNR-YUV / VMAF v1 BD: -3 vs
        // 0 -4.97% / -4.87% (VAL-150 -3.88% / -7.24%, long-GOP -1.80% /
        // -1.49%). -6 vs -3 won TRAIN (-3.25% / -2.45%) but on
        // bodycam_airport, measured with a cap-safe QP grid, it costs +15% bits
        // at matched QP and +8.9% PSNR-YUV BD for no VMAF gain (a hidden
        // per-class loss), so -6 is not the default.
        chroma_qp_offset: -3.0, [-12.0, 12.0],
            "pps_cb/cr_qp_offset: higher spends fewer bits on chroma, more on luma",
        intra_rounding: INTRA_ROUNDING_512THS as f32, [0.0, 256.0],
            "intra quantiser rounding offset (1/512 step, HM 171): lower widens the dead zone",
        inter_rounding: INTER_ROUNDING_512THS as f32, [0.0, 256.0],
            "inter quantiser rounding offset (1/512 step; HM 85, default 128 keeps texture SVT keeps): lower widens the dead zone",
        chroma_inter_rounding: INTER_ROUNDING_512THS as f32, [0.0, 256.0],
            "inter chroma quantiser rounding offset (1/512 step): higher keeps the colour detail VMAF's chroma feature measures",
        // MaxNumMergeCand: more spatial (and the temporal) candidates for
        // merge and skip, each ranked by mode decision and costing about a
        // bypass bin more to index. VVC's six-candidate list (with TMVP)
        // measured -1.9% J_real at +6.9% instructions (docs/handoff/vvc.md
        // "vvc-tmvp"); HEVC P slices have no history or pairwise candidates.
        merge_candidates: DEFAULT_MERGE_CANDIDATES as f32, [1.0, MAX_MERGE_CANDIDATES as f32],
            "merge candidates per P slice (MaxNumMergeCand, HEVC §7.4.7.1): more cheap vectors for skip and merge, a slightly longer merge_idx",
        sao_luma: 1.0, [0.0, 1.0],
            "luma SAO on (1) or off (0)",
        sao_chroma: 1.0, [0.0, 1.0],
            "chroma SAO on (1) or off (0)",
        // Default 4 (as Fast): TRAIN proxy-6 (150f) PSNR-YUV / VMAF v1 BD
        // +0.51% / +0.75% vs 2 for -2.7% instructions (crowd09 / city): J = -1.0.
        sao_stats_row_step: 4.0, [1.0, 4.0],
            "sample every Nth row for SAO statistics; higher trades decision precision for speed",
        // A Fast lever (default off, streams unchanged). Fast P instructions
        // (callgrind, 13 f): lowlight_city_night QP 43 -8.4%, bodycam_bc1
        // QP 47 -2.5%, crowd09 QP 47 -2.8%; bytes at the same QP within 0.04%.
        sao_skip_unchanged: 0.0, [0.0, 1.0],
            "1 switches SAO off without statistics in P-frame CTBs whose CUs are all skipped (they copy already filtered samples): faster",
        // Off: VC-1K 6 (fixed QP, 150 f) J -0.0 [-0.3, +0.3] for +2.6%
        // instructions (band statistics; KristenAndSara 720p QP 38).
        sao_band: 0.0, [0.0, 1.0],
            "1 evaluates SAO band offset (position search, §8.7.3.2) beside the four edge classes on searched pictures",
        // 4: searching every picture (1) gained VC-1K 6 J -0.6 [-1.0, -0.1]
        // for +4.7% instructions (edge statistics of all four classes).
        sao_search_interval: 4.0, [1.0, 8.0],
            "SAO types are searched on key pictures and every Nth picture (by POC), reused with re-derived offsets in between: higher is faster, coarser SAO",
        sao_luma_lambda_scale: 1.0, [0.25, 16.0],
            "λ multiple for luma SAO decisions: higher applies fewer luma offsets (less smoothing)",
        deblock_beta_offset: 0.0, [-6.0, 6.0],
            "pps_beta_offset_div2: higher treats more edges as blocking (more filtering)",
        deblock_tc_offset: 0.0, [-6.0, 6.0],
            "pps_tc_offset_div2: higher allows larger deblocking corrections",
        /// A speed lever for the Pareto search: the deblocking filter is
        /// about 6% of HEVC P-frame instructions (crowd09, QP 42). Off, P
        /// slices override the PPS (`deblocking_filter_override_flag`,
        /// `slice_deblocking_filter_disabled_flag`, §7.3.6.1); key frames
        /// keep it.
        /// Per-picture slice type under `decision.bi_pair`: a B slice
        /// whose bi-predicted share falls below this sends the next
        /// `hevc.b_slice_retry` inter pictures as P slices (no
        /// `inter_pred_idc` on uni units), then B again.
        b_slice_min_share: 0.0, [0.0, 1.0],
            "with decision.bi_pair: a B picture whose bi-predicted inter area share is below this codes the next hevc.b_slice_retry inter pictures as P (0: every inter picture B)",
        b_slice_retry: 8.0, [1.0, 64.0],
            "with hevc.b_slice_min_share: P pictures before B slices are tried again",
        deblock_inter: 1.0, [0.0, 1.0],
            "1 deblocks P slices; 0 switches the filter off on them (faster, blockier P frames)",
        lambda_scale: 1.0, [0.25, 4.0],
            "multiple of HM's λ for mode decisions and SAO: lower codes more residual (fewer skips)",
        column_zero_scale: 1.0, [1.0, 4.0],
            "intermediate-column zero bound multiplier: higher skips more residual detail for speed;1 is exact",
        // Default 1 for both (as Fast): TRAIN proxy-6 (150f) PSNR-YUV / VMAF
        // v1 BD -0.20% / -0.11% vs 0 for -0.9% instructions: J = -0.7.
        fast_forward16: 1.0, [0.0, 1.0],
            "encoder-only staged16-point row approximation retaining column zero bound",
        fast_forward32: 1.0, [0.0, 1.0],
            "encoder-only staged32-point row approximation retaining column zero bound",
        // Staged 32-point columns on top of staged rows (needs the rows
        // lever on): relative coefficient error 1.0% against 0.73% for rows
        // alone (codec_txfm tests); the 2-D DC stays exact. Jetson
        // (crowd 720p HEVC Fast QP 47): staged32_bounded 2.69% -> 1.72% of
        // cycles. Staged 16-point columns saved nothing there (6.46% ->
        // 6.56%), so they have no lever. Default off (byte-identical)
        // until screened.
        fast_forward32_columns: 0.0, [0.0, 1.0],
            "encoder-only staged32-point column approximation on top of staged rows",
        // Skip the exact column pass's 8-column groups the bound proves
        // all quantise to zero, writing zero coefficients (levels unchanged;
        // sign hiding and the trellis see zeros there). Applies with staged
        // rows (`fast_forward16/32`). Lowlight 360p HEVC Fast: the upper
        // eight columns of 57-85% of transformed 16x16 blocks qualify.
        // Default on: on vs off was byte-identical in 48/48 encodes (HEVC
        // Default and Fast, 8 clips, QP 32/40/47, 60f); P instructions at
        // QP 42 -1.0% Default / -1.7% Fast (x86), Jetson Fast -0.9..-1.8%.
        forward_zero_columns: 1.0, [0.0, 1.0],
            "encoder-only: zero proven-zero column groups of the staged-row forward instead of computing them",
        // SVT-AV1's partial-frequency forward (`av1.partial_forward`) for
        // inter residuals: only the low quarter of the coefficients is
        // computed and coded; block-QP refinements keep every frequency
        // (static_recovery.rs: HEVC Faster's static half stayed 8 dB short
        // after a starved key without that exception). Off in Default and
        // Fast.
        partial_forward: 0.0, [0.0, 2.0],
            "inter residuals keep only their low-frequency quarter: 1 32x32 transforms, 2 also 16x16 (faster, loses fine detail)",
        // Optional0.03: three native150f paired rounds save1.52% lowlight,
        // 3.83% MEVA and5.76% crowd CPU. Same three TRAIN4-QP curves cost
        // Full21 TRAIN PSNR/V1 BD+1.525%/+1.161%; held-out+.771%/+.011%.
        // TRAIN crowd027 V1+7.976%; crowd-low noise -2.476V1 at200kbps.
        // Default off; Fast selects .03. See HEVC handoff for full evidence.
        // Default 1: HEVC Default TRAIN proxy-6 (150f) PSNR-YUV / VMAF v1 BD
        // +0.24% / +0.56% for -3.7% (crowd09) / -0.6% (city) instructions:
        // J = -0.86. Full TRAIN confirmation follows.
        coarse_inter_intra: 1.0, [0.0, 1.0],
            "1 limits P-frame intra CUs to planar, DC, pure horizontal/vertical and the MPMs (no angular sweep): faster, coarser intra",
        // A Fast lever (default off, streams unchanged): it skips the
        // coarse P-frame mode search. Fast QP 47 P instructions -5.4%
        // crowd09 / -4.6% bodycam_bc1 for +0.9% / +0.3% bytes at the same
        // QP (callgrind, 13 frames). J_real (preset agent, CBR TRAIN-6 60f vs
        // SVT M12): every class -9..-35%, but bodycam worst-1% Y +29.9% vs
        // shipped Fast's +24.7% (allowance +5).
        // A Fast lever (default off, streams unchanged). Fast key-frame
        // instructions (callgrind) -19% lowlight_city_night QP 43 / -26%
        // bodycam_bc1 / -24% crowd09 QP 47, for key bytes +1.6% / +2.4% /
        // +8.8% at the same QP.
        // Whole-CU pricing (engine `cu_pricing.rs`, `backend/cu_trial.rs`).
        trial_full_intra: 1.0, [0.0, 1.0],
            "1 gives intra CUs coded in a whole-CU pricing trial the full (key-frame) luma mode search, 0 the P-frame search: 0 is faster",
        // 1 by default (SATD decides alone).
        intra_rd_candidates: 1.0, [1.0, 8.0],
            "luma modes of the full intra search (key frames) whose first transform block is transformed, plainly quantised and priced by CABAC bits plus coefficient-domain error before one is chosen; 1 takes the SATD-best: higher is slower, better intra",
        // The engine's smooth set for P-frame intra
        // (`IntraModeSet::Smooth`): most P-frame intra CUs repair brightness
        // changes and occlusions, where planar does most of the work.
        inter_intra_smooth: 0.0, [0.0, 1.0],
            "with hevc.inter_intra_dc_only, 1 codes those P-frame intra CUs with planar or DC, the cheaper by SATD + sqrt(lambda) mode bits, otherwise as DC only (no trials, inter fallback unconditional): finer than DC alone at about its cost",
        key_intra_coarse: 0.0, [0.0, 1.0],
            "1 gives key-frame intra CUs the coarse P-frame mode search (planar, DC, pure horizontal/vertical, MPMs): faster key frames, coarser intra",
        inter_intra_dc_only: 0.0, [0.0, 1.0],
            "1 codes P-frame intra CUs with DC, the mode the engine's intra test priced, without a mode search: faster, coarser intra",
        // 6 (every CU size) keeps `inter_intra_dc_only` as measured; lower
        // values leave the larger intra CUs, where bodycam's worst frames
        // come from, with the coarse search.
        inter_intra_dc_only_max_log2: 6.0, [3.0, 6.0],
            "with hevc.inter_intra_dc_only, the largest P-frame intra CU (log2 side) coded DC-only; larger ones keep the coarse search: higher is faster",
        early_merge_lambdas: 0.0, [0.0, 1.0],
            "cached integer merge SSE/sample lambda threshold for early mode and all-plane residual bypass: 0 disables",
        // Static-content Fast lever (static-detect, 2026-11-09; default off,
        // screen in docs/handoff/static-detect.md): see
        // `nc_encoder::still_skip`.
        still_skip: 0.0, [0.0, 1.0],
            "1 codes CUs of at least 16x16 that the stillness detector marks still with the zero vector and no residual, without mode decision or transforms, when every plane's zero-vector residual would quantise to zero",
        still_skip_limit_scale: 1.0, [1.0, 8.0],
            "still_skip's zero-residual test: multiple of the quantiser's zero limit (1 is nearly a proof that nothing would be coded): higher skips more",
        // Scale2: native CPU -3.56% body/-4.16% crowd; natural TRAIN
        // PSNR/V1 BD -0.017%/+0.338%, held-out +0.039%/+0.129%.
        pre_transform_zero_scale: 2.0, [1.0, 4.0],
            "inter residual L1 zero-bound multiplier before transform: 1 exact, higher skips near-zero detail for speed; intra unchanged",
        shave_gap: 8.0, [0.0, 16.0],
            "drop trailing ±1 levels that are at least this many scan positions past the previous level (SVT shave_coeff; 0 off): lower drops more",
        key_qp_offset_delta: 0.0, [-6.0, 6.0],
            "added to the intra-period-derived key-frame QP offset: higher spends more on key frames",
        // Intra and inter coding tools (`backend/intra.rs`,
        // `backend/inter_tree.rs`, `backend/key_cost.rs`); measurements in
        // docs/handoff/hevc.md.
        // Default on (planned keys + NxN + intra depth 2, the intra
        // bundle): J_real full TRAIN (18 clips, CBR ladder, 150f) quality
        // -1.1% full / -0.8% low, BETTER at k 0.2 (J -0.4%, dCPU +2.9%: key
        // frames 2.2x, about 8x a P frame), no guard; long-GOP (300f single
        // key) PSNR-YUV / VMAF v1 -2.7% / -0.3%. Fast pins all three off.
        intra_leaf_cost: 1.0, [0.0, 1.0],
            "1 plans key-frame CU sizes by transform cost (the engine's planner.intra_plan 2) instead of the fixed --block size: finer CUs on detail, at extra planning work",
        intra_nxn: 1.0, [0.0, 1.0],
            "1 lets 8x8 intra CUs code four 4x4 prediction units (PART_NxN) when that is cheaper: finer intra prediction of detail, at extra search work",
        // 2 vs 0 (with planned keys and intra depth 2): key-frame BD-YUV
        // -7.61% vs -7.67% (6 clips, frame 0, QP 22-37) for -11% key
        // instructions (crowd09 QP 42: 1184M -> 1049M).
        intra_nxn_min_cost: 2.0, [0.0, 64.0],
            "NxN is tried only on 8x8 CUs whose 2Nx2N cost exceeds this many λ per sample: higher is faster, tries NxN on less detail",
        intra_tu_depth: 2.0, [0.0, 2.0],
            "max_transform_hierarchy_depth_intra: intra transform blocks may split this many levels below the CU when cheaper (finer prediction of detail, more search work)",
        // 1 (no gate): 1.5 gives key BD -7.27% vs -7.67% for -7% key
        // instructions; 2 saves -15%.
        intra_tu_split_spread: 1.0, [1.0, 4.0],
            "an intra transform split is tried only when the residual's largest quadrant holds at least this fraction x 1/4 of its energy (1 always, 4 never): higher is faster",
        // 1 (planned keys, NxN, depth 2): key BD-YUV -6.94% vs -7.54% ungated
        // for key instructions 750M vs 915M (crowd09 QP 42; 2: -5.82%, 681M).
        intra_tu_split_min_cost: 1.0, [0.0, 64.0],
            "an intra transform split is tried only when the unsplit block costs more than this many λ per sample: higher is faster, splits less detail",
        // Default 1 with spread 2 (quality point, PROCEDURES "Preset
        // roles", k 0.1): J_real full TRAIN quality -1.1% full / -0.8% low,
        // 17/18 clips better, no guard, for +10.2% / +7.8% P instructions
        // (crowd09 / bc1, QP 42); homes390 -0.6% / -0.1%. HEVC Fast keeps 0.
        inter_tu_split: 1.0, [0.0, 1.0],
            "1 decides every inter CU's transform tree by rate-distortion trials (down to 8x8): better residual coding, more transform work",
        // A gate at 1 λ/sample removes most of the gain (crowd09 QP 37: P
        // +0.018 dB vs +0.150 dB ungated) for P instructions +4.5% vs +12%.
        inter_tu_split_min_cost: 0.0, [0.0, 64.0],
            "an inter transform split is tried only when the unsplit block costs more than this many λ per sample: higher is faster",
        inter_tu_split_spread: 2.0, [1.0, 4.0],
            "an inter transform split is tried only when the luma residual's largest quadrant holds at least this fraction x 1/4 of its energy: higher is faster",
    }
}

impl CodecBackend for HevcBackend {
    // A long-held second reference widens the POC LSBs to 16 bits
    // (`headers::LONG_POC_LSB_BITS`), so it reaches the engine's reference age.
    const SECOND_REFERENCE_REACH: u64 = nc_encoder::MAX_REFERENCE_AGE - 1;
    fn filter_outcomes(&self) -> Vec<(nc_encoder::FilterTool, nc_encoder::FilterOutcome)> {
        self.sao_outcome
            .frame()
            .map(|outcome| (nc_encoder::FilterTool::Sao, outcome))
            .into_iter()
            .collect()
    }

    // λ and the zero-residual bounds follow the planned step (`begin_frame`).
    // CBR 150 f, crowd_mot20_06 at 150 kbps: 315 -> 156 kbps.
    const SUPPORTS_LAMBDA_HEADROOM: bool = true;
    // cu_qp_delta (§7.3.8.14), one per CTB quantization group.
    const SUPPORTS_BLOCK_QP: bool = true;
    const BLOCK_QP_LOG2: u8 = LOG2_CTB as u8;

    const PARAM_SPECS: &'static [&'static [nc_encoder::ParamSpec]] = &[HevcParams::SPECS];
    /// The `hevc.*` heuristics are re-read per frame (`set_frame_params`),
    /// so frames nothing references can run cheaper ones
    /// (`nonbase.hevc.*`, `decision.nonbase_disposable_only`).
    const FRAME_PARAM_SPECS: &'static [&'static [nc_encoder::ParamSpec]] = &[HevcParams::SPECS];
    /// Parameter-set syntax (SPS/PPS: transform sizes and depths, chroma QP
    /// and deblocking offsets) and key-frame settings, read once.
    const STREAM_FIXED_PARAMS: &'static [&'static str] = &[
        "hevc.max_transform_log2",
        "hevc.tu16_chroma_concentration",
        "hevc.chroma_qp_offset",
        "hevc.deblock_beta_offset",
        "hevc.deblock_tc_offset",
        "hevc.key_qp_offset_delta",
        "hevc.key_intra_coarse",
        "hevc.intra_rd_candidates",
        "hevc.intra_leaf_cost",
        "hevc.intra_nxn",
        "hevc.intra_nxn_min_cost",
        "hevc.intra_tu_depth",
        "hevc.intra_tu_split_spread",
        "hevc.intra_tu_split_min_cost",
        "hevc.inter_tu_split",
        "hevc.inter_tu_split_min_cost",
        "hevc.inter_tu_split_spread",
        "hevc.trial_full_intra",
    ];
    /// Read per CU (`BlockSettings`); transform trees and `NxN` intra can only
    /// be switched off, as the parameter sets allow them or not.
    const SUPERBLOCK_PARAMS: &'static [&'static str] = &[
        "hevc.inter_tu_split",
        "hevc.inter_tu_split_spread",
        "hevc.inter_tu_split_min_cost",
        "hevc.intra_nxn",
        "hevc.intra_nxn_min_cost",
        "hevc.fast_forward16",
        "hevc.fast_forward32",
        "hevc.fast_forward32_columns",
        "hevc.forward_zero_columns",
        "hevc.partial_forward",
        "hevc.column_zero_scale",
        "hevc.pre_transform_zero_scale",
        "hevc.shave_gap",
        "hevc.early_merge_lambdas",
        "hevc.coarse_inter_intra",
        "hevc.inter_intra_dc_only",
        "hevc.inter_intra_dc_only_max_log2",
        "hevc.inter_intra_smooth",
        "quant.trellis",
        "quant.trellis_lambda_scale",
    ];
    /// `begin_frame` derives the picture from its plan; the collocated
    /// motion and the slots' POCs are committed in `commit_frame`.
    const RECODE: bool = true;
    // Intra refresh: dirty-edge CUs keep to DC and TMVP is off under it
    // (`slice_temporal_mvp_enabled_flag`, HEVC §7.4.7.1).
    const INTRA_REFRESH: bool = true;
    const NODE_TRIALS: bool = true;

    /// Shared decision values for HEVC:
    /// - `intra_gate_variance` 0: with the DC-aware intra test, trying intra
    ///   on every inter block adds to its gain (TRAIN 150f PSNR-YUV/VMAF v1
    ///   BD vs the test alone -1.6%/-0.5%, +1.5% instructions).
    /// - `codec_subpel_seed_lambdas` 0.05 (as HEVC Fast): skips NEW sub-pel
    ///   refinement when the seed residual is already small. On Default,
    ///   full TRAIN 150f +0.87%/+0.97% BD for -6.7% instructions (crowd09
    ///   Q47 / city Q42): J = -3.1.
    /// - `refine_cheap_vector` 1 (gate 0.05) and `codec_fullpel_probe` 1
    ///   (now the engine defaults), as AV1's quality-first Default: under coherent camera motion the
    ///   merge/AMVP predictor refined to sub-sample precision beats the
    ///   source search vector. `J_real` 15-clip TRAIN (CBR ladder, 150f):
    ///   quality -2.4% full / -2.5% low, pan -3.7%, worst-1% Y -3.3%, one
    ///   clip worse (homes432 +0.2%), for +11% instructions (crowd09 / bc1
    ///   Q42): BETTER at k 0.2. Fast keeps both off.
    /// - sub-pel move margins 1.1: the legacy engine values, kept until a
    ///   screen measures HEVC at the engine's 1.0.
    /// - `planner.depth_removal_min_quantizer` 0.86: no depth removal at
    ///   QP <= 43, so fine texture can split below 32x32. `J_real` TRAIN
    ///   (18 clips incl. texture): BETTER, quality -1.7% full / -0.0% low,
    ///   texture -2.1% / +1.4%, no guards, +3.1% instructions (crowd09 /
    ///   bc1 Q42). Removal off at every QP lost the low half (+0.4%).
    ///
    /// See docs/handoff/hevc.md.
    // The fixed-QP profile (`--qp`): the QP cadence at full strength on every
    // frame, a static per-position pattern as the standards' test conditions
    // use, instead of scaling it by each frame's still-block share. Fixed QP,
    // matched QP (nceval.fixedqp, two pool draws of 15 clips): J -1.1%
    // [-2.1, +0.1] and -1.0% [-2.0, -0.2]; CPU unchanged.
    // docs/analysis/fixed-qp-gap-2026-11-14.md "fixed-QP profile".
    // With it, the full mode decision against the unified plan's second
    // reference (at full cadence strength every high-quality frame already
    // carries it): two draws of 15-16 clips J -1.3% [-2.1, -0.6] and -1.7%
    // [-3.0, -0.5]. The other HEVC presets pin the plan and the profile off.
    //
    // Under camera motion the key is 6 QP coarser (`rate.fixed_qp_moving_key`, now the engine default,
    // engine `moving_camera.rs`): the profile's key, 15 QP finer than the P
    // frames at a 10 s period, is inherited by a fixed camera's frames, not
    // by a handheld or driving one's. Fixed QP, matched QP (nceval.fixedqp):
    // the six worst VC-1K phone-call clips J -3.9 [-5.2, -2.7]; pool draw
    // fixedqp-2026-11-18 quick (24 clips) -1.3 [-2.1, -0.6], adopt (45)
    // -0.8 [-1.4, -0.4], one clip worse (+1.1 %) of the 23 that change;
    // final (107) -0.8 [-1.0, -0.5]: pan -1.9, video call -1.9, bodycam
    // -1.3, static +0.3 (`static_meva_hospital` +9.1 %, the worst of five
    // clips worse). A long single key (450 f from a moving start): rate
    // -6.0 %, P-frame PSNR unchanged. CBR is untouched. 12 coarser (x265's
    // key offset): VC-1K -4.7, pool quick -1.2. HEVC Slow, Fast and Faster
    // pin 0 (not screened).
    // docs/handoff/hevc.md "VC-1K phone video calls".
    const FIXED_QP_PARAMS: &'static [(&'static str, f32)] = &[
        ("rate.cadence_static_scaling", 0.0),
        ("refs.second_full", 1.0),
    ];

    const DEFAULT_PARAMS: &'static [(&'static str, f32)] = &[
        // Periodic keys at their own quantizer, without the reuse term's finer
        // still cells (`reuse.rs`; owner, 2026-11-27: smaller periodic keys at
        // neutral J). Final check (medium, 600 f keys every 150 and 900 f keys
        // every 300, draw 2026-11-27, --strict-buffer): J -0.03% [-0.13, +0.06],
        // no guard; FourPeople 854x480 400 kbps, keys every 150: 26.8 -> 19.4 KB.
        // The other presets pin 2 until screened.
        ("persist.reuse_keys", 0.0),
        // Start-up QP cadence (period 8, base frames 6 QP finer, for the
        // first 100 frames after a key, then the engine's period 4 / -3 / +1)
        // and the cadence motion gate (`qp_cadence.rs`; docs/handoff/vvc.md,
        // qp-cascade). J_real (k 0.1, dCPU 0): proxy-6 BETTER -4.5% (static
        // -8.1%, crowd -6.4%, pan -3.4%, bodycam +0.6%); VAL J_real -9.7%
        // (static_otw_homes390 -29.8%, crowd -14.6%), periodic-key pulse
        // +0.27 dB against the 0.25 guard (coordinator: negligible, adopted);
        // conference (adv12) BETTER -20.5%, worst-1% Y -5.6%, window flicker
        // +0.12 dB, within the headroom below SVT-AV1 rtc. Long GOP (900 f,
        // one key, CBR 300): chirla Y 37.66 -> 37.63 at -0.4% rate, pan_bdd
        // 30.37 -> 30.40.
        ("rate.cadence_startup_period", 8.0),
        ("rate.cadence_startup_base_boost", 6.0),
        ("rate.cadence_motion_gate", 1.0),
        // The target as a ceiling over every key period (`rate.period_ceiling`,
        // `period_ledger.rs`; owner, 2026-11-14: steady sizes for still scenes).
        // 600-frame long-window L3 (keys every 10 s, 31 points, draw 2026-11-14):
        // on top of the quiet second before a key (`rate.prekey_quiet`): J -0.20%
        // [-0.40, -0.00] BETTER, no guard; static -0.78%, moving classes
        // unchanged; re-codes on the 22 static and crowd points 6.8 -> 8.0 per
        // 1000 frames. (Before the quiet second: J -0.84%, static -3.4%,
        // achieved/target max 1.008 -> 1.000.)
        // Inert within one key period. The other HEVC presets pin it off.
        ("rate.period_ceiling", 1.0),
        // `rate.still_level_damping` (engine 0.5) off: its final-check (2026-11-19,
        // 600 f, keys every 10 s) was J -0.36 % [-0.59, -0.14] with a pulse
        // guard of +1.03 dB from a real loss on meva_hospital_g301 (75 kbps:
        // key -0.93 dB, the 30 frames after it -1.42 dB, whole clip -0.19 dB).
        ("rate.still_level_damping", 0.0),
        // A fixed camera's key coded finer (`rate.key_still_boost`, QP units
        // times how still the scene is): its later frames inherit it. Pool levels,
        // draw 2026-11-14: L1 -1.64% [-1.88, -0.28]; L2 -1.02% [-1.57, -0.52];
        // 600-frame L3 (keys every 10 s) -0.42% [-0.90, -0.00], no guard; static
        // -1.75%, crowd +0.07%, pan and lowlight unchanged; achieved/target max
        // 0.993 -> 0.994; re-codes on the 22 static and crowd points 8.0 -> 10.6
        // per 1000 frames. 6 QP (the other Defaults' value) trips the
        // periodic-key pulse guard (+0.36 dB on draw 2026-11-14; landed on
        // 2026-11-15 as a497b590 and reverted with the inheritance-sized key
        // after the final-check showed HEVC keys pulsing at +1.61 dB).
        // The other HEVC presets pin it off.
        ("rate.key_still_boost", 4.0),
        // Variance AQ (`rate.aq_strength`, `nc-encoder` `aq.rs`): gentle
        // 0.25 QP/octave per-64x64-CTU redistribution from detailed to flat
        // blocks. HEVC CABAC-codes `cu_qp_delta` only on non-zero-CBF blocks
        // and adapts deblocking per CU. Pool draw 2026-11-26 (`--local`,
        // `--strict-buffer`): L1 (n=24) J -1.30% [-1.73, -0.57], 0 guards;
        // formal L3 (n=98, 150 f) J -0.40% [-0.76, -0.03] BETTER, unweighted
        // -0.54%, 0 guards (drone -2.03%, pan -1.86%, static -0.98%, crowd
        // -0.89%, bodycam +0.29%, videocall +0.29%); periodic-key pulse 2 s
        // -0.20 dB, 10 s +0.14 dB; ~0% extra CPU.
        ("rate.aq_strength", 0.25),
        // Key-frame trellis λ scaling (`quant.trellis_key_lambda_scale`, as
        // VVC's `vvc.trellis_key_lambda_scale = 0.5`): keeps more key-frame
        // detail for subsequent inter frames to inherit. Pool draw 2026-11-27
        // (`--local`, `--strict-buffer`): L1 (n=24) J -0.22% [-0.47, +0.02],
        // 0 guards; formal L3 (n=98, 150 f) J -0.22% [-0.58, +0.16],
        // unweighted -0.29%, 0 guards (static -1.28%, drone -0.26%, lowlight
        // -0.19%, pan -0.14%); periodic-key pulse 2 s -0.01 dB, 10 s +0.00 dB;
        // ~0% extra CPU.
        ("quant.trellis_key_lambda_scale", 0.5),
        // The unified structure's reference plan (`structure.unified`,
        // `nc-encoder` `second_reference.rs` `TierPlan`): high-quality frames
        // (key, cadence base) also predict from the high-quality frame before
        // them where the cadence strength is 0.75 or more. Build 78191a7a,
        // pool levels of draw 2026-11-15: L1 -0.52% [-0.88, -0.19]; L2
        // -0.35% [-0.66, -0.10]; L3 (ladder, 126 clips) J -0.33% [-0.62,
        // -0.09], no guard; Jetson cycles +2.2% (lowlight +5.2, crowd +2.1,
        // bodycam +0.2), instructions +1.6%. 600-frame final-check (27 clips,
        // keys every 10 s): J -0.19% [-0.63, +0.25], no guard; crowd -1.2%,
        // static -0.4%, lowlight +0.5%: a small gain. Landing conformance with it on:
        // every stream bit-exact in HM, ffmpeg and ours. The other HEVC
        // presets pin it off.
        ("structure.unified", 1.0),
        // The QP cadence at least as strong as the frames between measurably
        // inherit the high-quality frame (`structure.inherited_depth`,
        // `nc-encoder` `inheritance.rs`), beside the static share: it adds
        // the cadence on moving content that inherits (pans, crowds) and
        // leaves noise alone. Build 51b0ec30, pool levels of draw 2026-11-15:
        // L1 -1.19% [-2.86, -0.02]; L2 -1.86% [-3.54, -0.48]; L3 (single
        // point, 126 clips) J -1.33% [-2.74, -0.18], no guard (pan -3.4%,
        // crowd -0.9%, lowlight -0.3%, static +0.1%, bodycam +0.3%, texture
        // -7.1% on 4 clips, drone +1.7% on 3); 600-frame final (27 clips,
        // keys every 10 s) J -0.49% [-0.74, -0.23], no guard. Jetson cycles
        // -0.7% (instructions +0.4%). static_recovery passes; long single key
        // (900 f): no drift, PSNR-Y per third +0.00 to +0.19 dB. A first
        // version that gave unmeasured content the full cadence tripped the
        // worst-1% guards on pans (`docs/handoff/unified-structure.md` 13.4).
        // The other HEVC presets pin it off.
        ("structure.inherited_depth", 1.0),
        ("decision.intra_gate_variance", 0.0),
        // Sub-sample refinement without its early exits (seed residual,
        // negligible skip residual, cheap-vector residual): each skips
        // refining blocks whose residual looked small, where HEVC's exact
        // 8-tap ranking still finds better quarter-sample vectors. Fixed QP
        // (with exact_rank_margin 0, draw fixedqp-2026-11-14) J -0.4% for
        // +0.6% Jetson cycles. Adoption screen (k 0.1, dCPU +0.6%, pool draw
        // 2026-11-15): L1 J -0.93% [-1.28, -0.28] n 24, L2 -0.02% [-0.42,
        // +0.40] n 48, L3 (600 f, ladder) -0.56% [-0.85, -0.30] n 27 BETTER,
        // no guard. HEVC Fast and Faster pin the old values.
        ("decision.codec_subpel_seed_lambdas", 0.0),
        ("decision.skip_no_refine_lambdas", 0.0),
        ("decision.refine_cheap_min_lambdas", 0.0),
        ("decision.subpel_move_bias", 1.1),
        ("decision.subpel_move_bias_nonzero", 1.1),
        ("decision.subpel_fractional_bias", 1.1),
        ("planner.depth_removal_min_quantizer", 0.86),
        // Directional P-frame intra estimate (`decision.intra_directional`,
        // engine `intra_estimate.rs`) with this backend's SATD fallback
        // (`inter_fallback`), estimated only where DC intra costs under 2x
        // inter: J_real staged TRAIN (18 clips, k 0.1) BETTER, quality -0.4%
        // full / -0.4% low (PSNR-YUV -0.6%, NEG -0.0%, VMAF v1 -0.2%), 5/18
        // clips worse (lowlight_city_night +1.1%), no guard, periodic-key
        // pulse -0.04 dB. Fixed QP 60 f (QP 41-51): Riverbed -4.3% PSNR-YUV
        // / -3.4% VMAF, ParkJoy -1.0%, TreesAndGrass -0.3%. CPU: cost-clip P
        // frames +1.8% mid / +1.4% hi, key frames 0. HEVC Fast keeps it off
        // (its small P-frame intra CUs are DC only). docs/handoff/h264.md,
        // "P-frame intra: directional estimate".
        ("decision.intra_directional", 1.0),
        ("decision.intra_directional_dc_ratio", 2.0),
        // Tuned jointly (eval/nceval/tune_pool.py, study
        // hevc-default-eq2-2026-11-11, equal x86 CPU): coarser quantiser
        // rounding, a higher planner lambda with a cheaper split, less TU16
        // chroma concentration. Study clips J -1.51%, fresh clips -0.94%,
        // L3 of the study's draw -0.98% [-1.52, -0.49] (overfit gap +0.53);
        // re-checked at head 7fd52f29 on the disjoint L3 (79 clips,
        // single-point) J -0.89% [-1.43, -0.40], L2 (ladder) -0.60%, no
        // guard; CPU x86 -2.0%, Jetson cycles -0.3%. HEVC Fast and Faster
        // pin the previous values they do not set.
        ("hevc.inter_rounding", 109.0),
        ("hevc.intra_rounding", 209.0),
        ("planner.lambda_scale", 2.581_25),
        ("planner.split_rate_factor", 2.262_5),
        // The zero-trial split selector at half its fitted shift (engine
        // `split_selector.rs`, docs/handoff/planner-selector.md): fixed QP
        // J -0.6% [-0.9, -0.4] (quick draw, n 24); pool k 0.1 (dCPU +0.35%):
        // L1 -0.24% [-0.48, -0.01], L3 -0.30% [-0.51, -0.10] BETTER,
        // final-check (600 f) -0.61% [-0.90, -0.43] BETTER, no guard. The
        // other HEVC presets pin it off until screened on their own.
        ("planner.split_selector", 1.0),
        ("planner.split_selector_scale", 0.5),
        // The diffuse-chroma TU16 heuristic off: it predates the RD inter
        // transform tree (`inter_tree.rs`), which now decides 32x32 against
        // 16x16 by cost; forcing the split made 26-48 % of the inter transform
        // area 16x16 on VC-1K phone calls (x265: 3-7 %). Six worst VC-1K
        // clips (fixed QP) J -1.5 [-1.9, -1.1]; instructions -0.1 to -2.7 %.
        // Pool (CBR, k 0.1, dCPU -1, draw 2026-11-19, head 3db6e2c1): L1
        // -1.45 % [-1.99, -0.95], L2 -0.57 % [-1.31, +0.09] (drone +3.35),
        // L3 -0.31 % [-0.69, +0.06] BETTER, no guard (videocall -1.57, drone
        // +2.53, static +0.23); final-check (600 f, 36 clips) -0.68 % [-0.97,
        // -0.38], no guard. docs/handoff/hevc.md "VC-1K phone video calls".
        // Fast and Faster pin 0.5, Slow 0.35 (not screened).
        ("hevc.tu16_chroma_concentration", 0.0),
        // Four merge candidates (MaxNumMergeCand, `mvpred.rs`) and no depth
        // removal (16x16 and 8x8 always on offer to the planner). What they
        // add: the third spatial candidate and TMVP for skip and merge, and
        // planned splits the motion-search costs had ruled out. Fixed QP,
        // matched QP (nceval.fixedqp, 15 pool clips): merge 4 J -1.7%
        // [-2.0, -1.0] (3: -1.3%, 5: -1.7%), no depth removal -0.4%. CPU: x86
        // P instructions +2.2%, Jetson cycles +1.7%. Adoption screen (k 0.1,
        // dCPU 2.2, the day's pool draw): L2 J -1.12% [-1.53, -0.75], L3
        // (full ladder) BETTER, J -1.01% [-1.34, -0.72] (n 126, L2+L3
        // pooled), no guard. docs/analysis/fixed-qp-gap-2026-11-14.md.
        // HEVC Fast and Faster pin the old values. No depth removal is the
        // engine default since 2026-11-15.
        ("hevc.merge_candidates", 4.0),
        // Whole-CU pricing of intra leaves (engine `cu_pricing.rs`): an
        // intra leaf (not pre-analysis static) that beat inter by under 0.1
        // λ per sample is also coded at merge candidate 0, and the cheaper
        // coded CU (all planes) is kept; trial intra takes the full search.
        // The shadow RD log found intra -> inter worth 1.2 % of the coded
        // cost on the six worst VC-1K clips, 69 % of it within 0.05 λ.
        // Worst six (fixed QP, matched QP) J -2.1 [-2.6, -1.7] (VMAF -3.2);
        // instructions (QP 36, 30 f: Johnny / KristenAndSara /
        // WestWindEasy / ControlledBurn) +0.5 / +1.0 / +6.8 / +3.3 %. Pool
        // (draw 2026-11-19, k 0.1, dCPU 2.8): L1 -1.52 %, L2 -1.23 %, L3
        // (single point, n 108) -1.00 % [-1.39, -0.61]; the periodic-key
        // pulse guard (+0.56 dB, external_otw_square_native) comes from a
        // better key (rate control gave it more bits: key +2.9 dB, P frames
        // before and after it better). final-check (36 clips, 600 f, keys
        // every 10 s) J -0.80 % [-0.96, -0.64], no guard.
        // docs/handoff/hevc.md "VC-1K: shadow
        // RD ceiling and whole-CU pricing". The other HEVC presets pin it off.
        ("decision.cu_price_intra_lambdas", 0.1),
        // SAO decisions from every second row's statistics (the registered
        // 4 samples a quarter): edge classes and offsets fit the CTB better.
        // Fixed QP (draw fixedqp-2026-11-21, 23 clips) J -1.0 % [-1.2,
        // -0.6] at +2.6 % P instructions (every row: -1.2 % at +7.9 %);
        // whole encode (chart timing commands, callgrind) +2.75 %. Quick
        // (k 0.1) L1 -0.75 % [-1.22, -0.28], probe -0.36 %; nceval.train
        // (k 0.1): L3 (n 108) -0.43 % [-0.68, -0.19] BETTER, final-check
        // (n 34, 600 f) -0.94 % [-1.11, -0.65] BETTER, no guard
        // (docs/analysis/unused-tools-2026-11-21.md). Fast and Faster pin 4,
        // Slow sets 1.
        ("hevc.sao_stats_row_step", 2.0),
        // P-frame CTUs whose CUs are all skipped (they copy filtered
        // reference samples) keep SAO off without gathering statistics, as
        // Fast and Faster do: the SAO decision is 6-7 % of the encode
        // (callgrind, crowd / static 720p) and these CTUs rarely take it.
        // Fixed QP (24 clips) J +0.0 [-0.2, +0.2]; whole encode Zen 5
        // -2.15 %, Graviton3 -1.62 %, Graviton4 -1.80 %. nceval.train
        // (k 0.1): L3 (n 108) -0.20 % [-0.39, -0.01] BETTER, final-check
        // (n 34) +0.20 % [-0.07, +0.44] AMBIGUOUS, passed
        // (docs/analysis/unused-tools-2026-11-21.md); Veryslow too (k 0:
        // docs/handoff/filter-effort.md).
        ("hevc.sao_skip_unchanged", 1.0),
        // Filter-search effort by usage (`nc_encoder::UsageGate`): on non-base
        // P frames, CTUs coded as skip over at least 3/4 of their area keep SAO
        // off without statistics (each CTU refreshed every 8 frames). HEVC
        // reason: its SAO is wired to the gate (with VVC's SAO and ALF).
        // nceval.train (k 0.1, base 1cb80965c): L3 (n 108) -0.17 % [-0.44,
        // +0.09] BETTER, final-check (n 34) -0.01 % [-0.15, +0.19] AMBIGUOUS;
        // whole encode cycles Graviton3 -0.34 %, Graviton4 -0.48 %, Zen 5
        // -0.36 % (docs/handoff/filter-effort.md). Fast, Faster, Slow,
        // Slower and Veryslow pin 0.
        ("filter.usage_quiet_share", 0.75),
        // Periodic keys planned 2 QP units finer than the inter level, adopted
        // with the key-frame RD partition search (3ebca4466). The search itself
        // is off at medium (reverted 2026-11-22): on the published front's chart
        // timing (150 f, one key) the head cost Zen 5 cycles AV1 +5.3 %, AV2
        // +9.1 %, VVC +5.9 % against the same head with the search, these finer
        // keys, the AV1 / AV2 key trellis and H.264's key Intra_8x8 off, which
        // outweighs its J (L3 -0.33 to -0.55 %). The finer key adds no search
        // and stays (coordinator decision); a screen of it alone is queued.
        ("rate.key_level_finer", 2.0),
        // Motion and sub-pel exits as linear threshold shifts (G2,
        // `linear_threshold.rs`; docs/analysis/g2-motion-thresholds-2026-11-22.md):
        // exits looser on non-base frames, zero-vector seeds and noisy
        // sources, base frames slightly stricter. Tuned for HEVC medium on
        // 600 f windows (trial hevc_600f_t15); other codecs: AV1 medium's
        // tuned point was flat (held-out L3 +0.01 %, 600 f -0.05 %). Judged
        // at the front slope k 0.1827, with Zen 5 cycles (Graviton 4 saves
        // more: -2.32 / -1.65 %): CPU 1/150 -1.96 %, 1/300 -1.19 %; final
        // check (2026-11-22 draw) 1/150 J -0.42 % [-0.65, -0.19] (n 39),
        // 1/300 J -0.27 % [-0.54, -0.01] (n 29); L3 150 f J -0.40 %
        // [-0.68, -0.13] (n 108); quality alone -0.06 / -0.05 / -0.04 %; key
        // pulse guarded <= +0.03 dB. Fast, Faster, Slow, Slower and Veryslow
        // pin 0 (not screened).
        ("analysis.window_exit_shift_bias", -0.103_662),
        ("analysis.window_exit_shift_lambda", -0.291_233),
        ("analysis.window_exit_shift_noise", 0.394_298),
        ("analysis.window_exit_shift_nonbase", 0.444_587),
        ("analysis.zero_gate_shift_bias", -0.020_961_7),
        ("analysis.zero_gate_shift_lambda", 0.233_67),
        ("analysis.zero_gate_shift_neighbour_mv", -0.061_071),
        ("analysis.zero_gate_shift_noise", -0.521_622),
        ("analysis.zero_gate_shift_nonbase", 0.749_672),
        ("decision.skip_no_refine_shift_bias", 0.049_152_1),
        ("decision.skip_no_refine_shift_lambda", -0.042_977_6),
        ("decision.skip_no_refine_shift_noise", 0.021_140_3),
        ("decision.skip_no_refine_shift_nonbase", -0.056_853_5),
        ("decision.subpel_seed_shift_bias", -0.015_777_7),
        ("decision.subpel_seed_shift_lambda", -0.028_532),
        ("decision.subpel_seed_shift_noise", 0.011_911_3),
        ("decision.subpel_seed_shift_nonbase", 0.046_694_7),
        ("decision.subpel_seed_shift_size", -0.005_502_83),
        ("decision.subpel_seed_shift_zero_mv", 0.041_685_9),
    ];

    // A flat field-by-field initialisation of the backend's state.
    #[allow(clippy::too_many_lines)]
    fn new(config: &EncoderConfig) -> Result<Self, ConfigError> {
        if !(QP_MIN..=QP_MAX).contains(&config.quantizer) {
            return Err(ConfigError::Quantizer {
                value: config.quantizer,
                min: QP_MIN,
                max: QP_MAX,
            });
        }
        if !(LOG2_MIN_CB as u8..=MAX_BLOCK_LOG2).contains(&config.block_log2) {
            return Err(ConfigError::BlockSize(config.block_log2));
        }
        let qp = config.quantizer as i32;
        let params = HevcParams::from_overrides(&config.params);
        let info = stream_info(config, &params);
        let (width, height) = (info.width, info.height);
        let rounding = rounding_of(&params);
        let frame_quantization = Box::new(FrameQuantization::without_headroom(
            qp,
            info.chroma_qp_offset,
            rounding,
            &params,
        ));
        Ok(Self {
            info,
            slice: SliceKind::Intra,
            qp,
            poc: 0,
            key_index: 0,
            sequence_sets: None,
            idr: true,
            frame_index: 0,
            refresh: None,
            scans: Scans::new(),
            contexts: Contexts::for_slice(SliceKind::Intra, qp),
            items: Vec::new(),
            item_pricing: None,
            sao: None,
            sao_classes: Vec::new(),
            pending_sao_classes: None,
            sao_outcome: nc_encoder::OutcomeTracker::default(),
            sao_usage: None,
            grid: Grid::new(width, height),
            motion: MotionStore::new(
                width,
                height,
                if info.kept_slot.is_kept() {
                    long_term_slot(config.temporal_layers) + 1
                } else {
                    reference_slots(config.temporal_layers)
                },
            ),
            references: FrameReferences::FLAT,
            reference_list: ReferenceDistances::one(1),
            active_ref: 0,
            inter_slice: inter_slice_of(&config.params),
            l1_amvp: None,
            bi_first: Vec::new(),
            slot_poc: [0; MAX_REFERENCE_SLOTS],
            long_term_slot: long_term_slot(config.temporal_layers),
            long_term_poc: None,
            next_cu: 0,
            next_tu: 0,
            current: PredictorCache::default(),
            rate_control_enabled: false,
            key_qp_offset: config.key_qp_offset.unwrap_or_else(|| {
                nc_encoder::fixed_qp_key_offset(config.intra_period, KEY_QP_OFFSET_AT_REFERENCE)
                    + params.key_qp_offset_delta.round() as i32
            }),
            rounding,
            frame_quantization,
            block_qp: None,
            column_zero_scale: params.column_zero_scale,
            fast_forward: FastForward::new(&params),
            pre_transform_zero_scale: params.pre_transform_zero_scale,
            tu16_chroma_concentration_q16: (params.tu16_chroma_concentration * 65536.0).round()
                as u32,
            shave_gap: params.shave_gap.round() as usize,
            lambda_scale: params.lambda_scale,
            sao_components: sao_components_of(&params),
            dsp: Dsp::new(),
            inter_pred: [
                vec![0; CTU_SAMPLES],
                vec![0; CTU_SAMPLES / 4],
                vec![0; CTU_SAMPLES / 4],
            ],
            cached_luma: None,
            key_luma_ranking: None,
            early_merge_lambdas: params.early_merge_lambdas,
            inter_intra: InterIntraSearch::new(&params),
            key_intra: InterIntraSearch::key(&params),
            early_merge: None,
            still_skip_limit_scale: still_skip_limit_scale(&params),
            still_skipped: None,
            encoder_info: None,
            scratch: RefCell::default(),
            level_store: RefCell::default(),
            tu_store: Vec::new(),
            deblocking: SliceDeblocking::Fixed,
            deblock_inter: params.deblock_inter >= 0.5,
            b_schedule: BSchedule::new(&params),
            random: None,
            final_scratch: final_cost::FinalScratch::default(),
            bin_costs: BinCosts::new(),
            tools: CodingTools::new(&params),
            trellis: TrellisQuant::from_params(&QuantParams::from_overrides(&config.params)),
            rd_trim: nc_encoder::RateTrim::from_overrides(&config.params),
            alternate: None,
            cu_trial: cu_trial::CuTrial::default(),
            node_trials: node_trial::NodeTrials::default(),
            trial_intra_search: (params.trial_full_intra >= 0.5).then_some(InterIntraSearch::Full),
            intra_rd_candidates: params.intra_rd_candidates.round().max(1.0) as usize,
            intra_stage: nc_encoder::IntraStage::from_overrides(&config.params),
            gates: nc_encoder::GateParams::from_overrides(&config.params),
        })
    }

    fn superblock_log2(&self) -> u8 {
        LOG2_CTB as u8
    }

    fn min_block_log2(&self) -> u8 {
        LOG2_MIN_CB as u8
    }

    fn rate_model(&self) -> nc_encoder::RateModel {
        RATE_MODEL
    }

    fn supports_inter(&self) -> bool {
        true
    }

    fn coded_size(&self, width: usize, height: usize) -> (usize, usize) {
        coded_picture_size(width, height)
    }

    fn supports_temporal_layers(&self) -> bool {
        true
    }

    fn enable_random_decisions(&mut self, rng: nc_encoder::DebugRng) {
        self.enable_random_syntax(rng);
    }

    /// `cu_qp_delta_enabled_flag` from the next PPS (every key frame
    /// repeats it; this is called before the first).
    fn enable_block_qp(&mut self) {
        self.info.block_qp = true;
        self.block_qp = Some(BlockQp::new());
    }

    /// An estimate: the slice is entropy coded only at `end_frame`, so the
    /// recorded items are priced as the coding will adapt its contexts,
    /// from where the last reading stopped (afresh when a node trial cut
    /// them back past it).
    fn coded_bits(&mut self) -> Option<f64> {
        let mut pricing = self
            .item_pricing
            .take()
            .filter(|p| p.next <= self.items.len())
            .unwrap_or_else(|| ItemPricing {
                next: 0,
                bits: 0.,
                contexts: self.contexts.clone(),
            });
        let added = self.estimated_item_bits(&self.items[pricing.next..], &mut pricing.contexts);
        pricing.bits += f64::from(added);
        pricing.next = self.items.len();
        let bits = pricing.bits;
        self.item_pricing = Some(pricing);
        Some(bits)
    }

    /// Slices then carry explicit reference picture sets keeping the
    /// long-term picture (`long_term_set`).
    fn supports_long_term_reference(&self) -> bool {
        true
    }

    /// The second reference (`refs.second`) is a second short-term entry of
    /// the slice's reference picture set and list (`ref_idx_l0` 1, HEVC
    /// §7.3.7, §8.3.2), chosen per prediction unit.
    fn supports_second_reference(&self) -> bool {
        true
    }

    fn select_reference(&mut self, _block: &Block, reference: ReferenceChoice) {
        self.active_ref = reference.single_index();
    }

    /// Generalised B slices code bi-prediction from the previous picture at
    /// two vectors (list 0 and list 1 `ref_idx` 0).
    fn supports_primary_pair(&self) -> bool {
        self.slice == SliceKind::Bi
    }

    /// In a generalised B slice with the second reference, list 0's first
    /// picture and list 1's second (`ref_idx_l1` 1) average.
    fn supports_compound_reference(&self) -> bool {
        self.slice == SliceKind::Bi && self.reference_list.count() > 1
    }

    /// The merge list's genuinely bi-predicted candidates whose list 0 half
    /// is the first reference: list 1 at `ref_idx` 0 is a primary pair, at
    /// 1 (the second reference) a compound.
    fn pair_merge_candidates(&mut self, block: &Block, out: &mut Vec<(Mv, ReferenceChoice)>) {
        let merge = self
            .current
            .get(block, 0)
            .unwrap_or_else(|| self.predictors_for(block, 0))
            .merge;
        for candidate in merge.motions() {
            let Some(l1) = candidate.l1.filter(|_| candidate.as_uni().is_none()) else {
                continue;
            };
            let second = match (candidate.ref_idx, l1.ref_idx) {
                (0, 0) => ReferenceChoice::PrimaryPair { second_mv: l1.mv },
                (0, 1) => ReferenceChoice::Compound {
                    secondary_mv: l1.mv,
                },
                _ => continue,
            };
            out.push((candidate.mv, second));
        }
    }

    fn pair_coded_cost(
        &mut self,
        source: &Picture,
        references: [&Picture; 2],
        block: &Block,
        pair: (Mv, ReferenceChoice),
    ) -> Option<f32> {
        self.price_pair(source, references, block, pair)
    }

    fn primary_pair_mode_bits(&mut self, block: &Block, first_mv: Mv, second_mv: Mv) -> f32 {
        self.bi_mode_bits(block, bi_motion(first_mv, (second_mv, 0)))
    }

    fn compound_mode_bits(&mut self, block: &Block, primary_mv: Mv, secondary_mv: Mv) -> f32 {
        self.bi_mode_bits(block, bi_motion(primary_mv, (secondary_mv, 1)))
    }

    fn encode_compound_block(
        &mut self,
        block: &Block,
        decision: &BlockDecision,
        source: &Picture,
        primary_reference: &Picture,
        secondary_reference: &Picture,
        recon: &mut Picture,
    ) {
        let (second_mv, l1_ref) = match decision.reference {
            ReferenceChoice::PrimaryPair { second_mv } => (second_mv, 0),
            ReferenceChoice::Compound { secondary_mv } => (secondary_mv, 1),
            ReferenceChoice::Primary | ReferenceChoice::Secondary => {
                unreachable!("the engine codes single-reference blocks with encode_block")
            }
        };
        let PredictionMode::Inter(first_mv) = decision.mode else {
            unreachable!("two-hypothesis blocks are inter")
        };
        self.begin_block_qp(decision.qp_offset);
        self.active_ref = 0;
        self.encode_bi_cu(
            block,
            bi_motion(first_mv, (second_mv, l1_ref)),
            source,
            [primary_reference, secondary_reference],
            recon,
        );
        self.end_block_qp();
        self.current.clear();
        self.cached_luma = None;
        self.early_merge = None;
    }

    /// Rank inter candidates with HEVC's own 8-tap luma filter (the engine's
    /// SVT light-PD1 flow). Quick clips, 10 s IP, PSNR-YUV BD vs svtav1-m12
    /// (fast check): −3.0% → −4.9%.
    fn luma_filter(&self) -> Option<nc_dsp::Filter> {
        Some(nc_dsp::Filter::HevcLuma)
    }

    // Exact candidate ranking has already filtered the winning fractional luma.
    // Keep those pixels in inter_pred; the following CU predicts chroma normally.
    // With selector0.5/SAO row2, five native paired runs save1.89% bodycam
    // and0.57% crowd CPU. Streams are identical, so quality is unchanged.
    const CACHE_EXACT_LUMA: bool = true;

    fn cache_exact_luma(
        &mut self,
        block: &Block,
        mv: Mv,
        filter: nc_dsp::Filter,
        prediction: &[u8],
        _stats: nc_dsp::Variance,
    ) {
        let area = block.size() * block.size();
        self.inter_pred[0][..area].copy_from_slice(prediction);
        self.cached_luma = Some(CachedLuma {
            block: *block,
            motion: Motion::uni(mv, self.active_ref),
            filter,
            frame_index: self.frame_index,
        });
    }

    fn set_frame_params(&mut self, overrides: &nc_encoder::ParamOverrides) {
        self.intra_stage = nc_encoder::IntraStage::from_overrides(overrides);
        self.gates = nc_encoder::GateParams::from_overrides(overrides);
        let params = HevcParams::from_overrides(overrides);
        // The frame's quantisation is derived from these in `begin_frame`.
        self.rounding = rounding_of(&params);
        self.lambda_scale = params.lambda_scale;
        self.column_zero_scale = params.column_zero_scale;
        self.pre_transform_zero_scale = params.pre_transform_zero_scale;
        self.fast_forward = FastForward::new(&params);
        self.shave_gap = params.shave_gap.round() as usize;
        self.sao_components = sao_components_of(&params);
        self.early_merge_lambdas = params.early_merge_lambdas;
        self.inter_intra = InterIntraSearch::new(&params);
        self.still_skip_limit_scale = still_skip_limit_scale(&params);
        // Switching the filter off needs the PPS override flag, which
        // `stream_info` sets when any frame's settings do.
        self.deblock_inter = params.deblock_inter >= 0.5;
        self.b_schedule.configure(&params);
        self.inter_slice = inter_slice_of(overrides);
    }

    fn set_light_params(&mut self, params: Option<&nc_encoder::ParamOverrides>) {
        debug_assert!(
            self.alternate.as_ref().is_none_or(|a| !a.light_in_use),
            "set between frames, regular settings in use"
        );
        self.alternate = params.map(|overrides| Alternate {
            settings: BlockSettings::light(overrides, &self.tools),
            light_in_use: false,
        });
    }

    fn use_light_params(&mut self, light: bool) {
        let Some(other) = self.alternate.as_mut().filter(|a| a.light_in_use != light) else {
            return;
        };
        let theirs = &mut other.settings;
        std::mem::swap(&mut self.tools, &mut theirs.tools);
        std::mem::swap(&mut self.trellis, &mut theirs.trellis);
        std::mem::swap(&mut self.fast_forward, &mut theirs.fast_forward);
        std::mem::swap(&mut self.column_zero_scale, &mut theirs.column_zero_scale);
        std::mem::swap(
            &mut self.pre_transform_zero_scale,
            &mut theirs.pre_transform_zero_scale,
        );
        std::mem::swap(&mut self.shave_gap, &mut theirs.shave_gap);
        std::mem::swap(
            &mut self.early_merge_lambdas,
            &mut theirs.early_merge_lambdas,
        );
        std::mem::swap(&mut self.inter_intra, &mut theirs.inter_intra);
        other.light_in_use = light;
    }

    fn lambda(&self) -> f32 {
        self.frame_quantization.lambda
    }

    fn prefer_cached_inter(
        &mut self,
        block: &Block,
        mv: Mv,
        stats: nc_dsp::Variance,
        lambda: f32,
    ) -> bool {
        self.early_merge = None;
        let motion = Motion::uni(mv, self.active_ref);
        if self.early_merge_lambdas <= 0.0
            || !self.slice.is_inter()
            || !mv.is_full_pel()
            || self
                .current
                .latest()
                .is_none_or(|p| p.merge.get(0).as_uni() != Some(motion))
        {
            return false;
        }
        let limit = self.early_merge_lambdas * lambda;
        if stats.sse as f32 > limit * (block.size() * block.size()) as f32 {
            return false;
        }
        self.early_merge = Some(EarlyMergeHint {
            frame_index: self.frame_index,
            block: *block,
            motion,
            sse_per_sample: limit,
        });
        true
    }

    fn relative_quantizer(&self) -> f32 {
        self.qp as f32 / QP_MAX as f32
    }

    fn partition_bits(&self, node: &Block, split: bool) -> f32 {
        if split_flag_present(&self.grid, node) {
            self.contexts.split_cu_flag[split_ctx_inc(&self.grid, node)].bits(split)
        } else {
            0.0
        }
    }

    fn supports_intra_leaf_cost(&self) -> bool {
        self.tools.intra_leaf_cost
    }

    fn intra_leaf_cost(&mut self, source: &Picture, block: &Block) -> Option<f32> {
        self.tools
            .intra_leaf_cost
            .then(|| self.key_leaf_cost(source, block))
    }

    fn begin_cu_trial(&mut self, block: &Block, recon: &Picture) -> bool {
        self.save_cu_trial(block, recon)
    }

    fn cu_trial_cost(&mut self, block: &Block, source: &Picture, recon: &Picture) -> f32 {
        self.coded_cu_cost(block, source, recon)
    }

    fn undo_cu_trial(&mut self, recon: &mut Picture) {
        self.restore_cu_trial(recon);
    }

    fn end_cu_trial(&mut self) {
        self.finish_cu_trial();
    }

    fn begin_node_trial(&mut self, node: &Block, recon: &Picture) {
        self.open_node_trial(node, recon);
    }

    fn begin_node_pricing(&mut self, node: &Block, _recon: &Picture) {
        self.open_node_pricing(node);
    }

    fn node_trial_cost(&mut self, source: &Picture, recon: &Picture) -> f32 {
        self.node_trial_price(source, recon)
    }

    fn node_trial_parts(
        &mut self,
        source: &Picture,
        recon: &Picture,
    ) -> Option<nc_encoder::NodeTrialParts> {
        Some(self.node_trial_split_price(source, recon))
    }

    fn node_trial_summary(&self) -> Option<nc_encoder::NodeTrialSummary> {
        Some(self.node_trial_coded_summary())
    }

    fn undo_node_trial(&mut self, recon: &mut Picture) {
        self.undo_node(recon);
    }

    fn keep_node_candidate(&mut self, recon: &Picture) {
        self.keep_node(recon);
    }

    fn restore_node_candidate(&mut self, recon: &mut Picture) {
        self.restore_kept_node(recon);
    }

    fn end_node_trial(&mut self) {
        self.close_node_trial();
    }

    fn last_cu_skipped(&self) -> bool {
        self.last_cu_was_skip()
    }

    fn final_intra_screen(
        &mut self,
        source: &Picture,
        recon: &Picture,
        block: &Block,
    ) -> Option<(f32, f32)> {
        Some(self.screen_intra_cu(source, recon, block))
    }

    fn final_coded_cost(
        &mut self,
        source: &Picture,
        reference: &Picture,
        block: &Block,
        candidate: nc_encoder::FinalCandidate,
    ) -> Option<f32> {
        self.prices_final_candidates()
            .then(|| self.price_final(source, reference, block, candidate))
            .flatten()
    }

    /// Random decisions bypass the pricing.
    fn prices_final_candidates(&self) -> bool {
        self.random.is_none()
    }

    /// Merge candidate 0 (`final_cost.rs`): the winner walk keeps it, and
    /// the final decision stage sets it against the ranking's winner.
    fn skip_vector(&mut self, block: &Block) -> Option<Mv> {
        self.merge_skip_vector(block)
    }

    fn cu_trial_skip_vector(&mut self, block: &Block) -> Option<Mv> {
        self.first_merge_vector(block)
    }

    fn static_skip_vector(&mut self, _block: &Block) -> Option<Mv> {
        // Also the vector of the engine's noise skip, which needs no
        // `hevc.still_skip`.
        Some(Mv::ZERO)
    }

    /// `hevc.still_skip`: a still CU whose zero-vector residual would
    /// quantise to zero in every plane is coded as the zero vector with no
    /// residual (`nc_encoder::still_skip`); so is one the engine's noise
    /// skip passes (`test.within_noise`).
    fn try_static_skip(
        &mut self,
        block: &Block,
        source: &Picture,
        reference: &Picture,
        recon: &mut Picture,
        test: StaticSkipTest,
    ) -> bool {
        if self.random.is_some() {
            return false;
        }
        let accepted = test.within_noise
            || test.repeat
            || (test.still && self.still_skip_limit_scale > 0.0 && {
                let [luma, chroma] =
                    self.frame_quantization
                        .planes
                        .each_ref()
                        .map(|plane| ZeroLimits {
                            dc: plane.inter_zero_limit,
                            ac: plane.inter_zero_limit,
                        });
                let limits = PlaneZeroLimits { luma, chroma }.scaled(self.still_skip_limit_scale);
                zero_vector_residual_is_zero(&self.dsp, block, source, reference, limits)
            });
        if accepted {
            copy_zero_vector_prediction(&self.dsp, block, reference, recon);
            self.still_skipped = Some((block.x, block.y));
        }
        accepted
    }

    fn mv_candidates(&mut self, block: &Block, out: &mut Vec<Mv>) {
        self.ranking_vectors(block, out);
    }

    fn ranking_vectors(&mut self, block: &Block, out: &mut Vec<Mv>) -> Option<Mv> {
        let p = self.predictors(block);
        self.current.set(p);
        // Merge candidates cost almost nothing to signal; AMVP predictors
        // cost only a zero vector difference. With a second reference, a
        // merge vector of the other reference is still a likely vector:
        // `mode_bits` prices it as AMVP.
        out.extend(p.merge.vectors().iter().chain(&p.amvp).copied());
        p.merge
            .get(0)
            .as_uni()
            .filter(|first| first.ref_idx == self.active_ref)
            .map(|first| first.mv)
    }

    fn mode_bits(&self, block: &Block, mode: &PredictionMode) -> f32 {
        let min_size = block.log2_size() == LOG2_MIN_CB as u8;
        match *mode {
            PredictionMode::IntraDc => {
                // part_mode is only coded for minimum-size intra CUs.
                let part = if min_size { 0.0 } else { -1.0 };
                cost::SKIP_FLAG + cost::MODE_AND_PART + cost::INTRA_DC + part
            }
            PredictionMode::Inter(mv) => {
                let fallback;
                let p = if let Some(p) = self.current.get_ref(block, self.active_ref) {
                    p
                } else {
                    fallback = self.predictors(block);
                    &fallback
                };
                let motion = Motion::uni(mv, self.active_ref);
                if let Some((idx, _)) = p.merge.predicting(motion) {
                    // Skip, or merge with residual.
                    cost::SKIP_FLAG + cost::merge_idx(idx, self.info.merge_candidates)
                } else {
                    let mvd = p
                        .amvp
                        .iter()
                        .map(|&pred| mvd_bits(mv, pred))
                        .fold(f32::INFINITY, f32::min);
                    cost::SKIP_FLAG
                        + cost::MODE_AND_PART
                        + 2.0 * cost::FLAG
                        + self.uni_inter_pred_idc_bits()
                        + self.ref_idx_bits()
                        + mvd
                }
            }
        }
    }

    fn set_rate_control_enabled(&mut self, enabled: bool) {
        self.rate_control_enabled = enabled;
    }

    fn set_refresh(&mut self, window: Option<&RefreshWindow>) {
        self.refresh = window.cloned();
    }

    fn set_filter_effort(&mut self, effort: nc_encoder::FilterEffort) {
        self.sao_usage = effort.usage;
    }

    /// A `user_data_unregistered` SEI (§D.2.7) after the IRAP pictures'
    /// parameter sets.
    fn set_encoder_info(&mut self, payload: Option<Vec<u8>>) {
        self.encoder_info =
            payload.map(|payload| nal::sei_rbsp(nal::SEI_USER_DATA_UNREGISTERED, &payload));
    }

    /// The VPS and SPS level from the next IRAP picture on covers `rate`.
    fn set_stream_rate(&mut self, rate: Option<nc_encoder::StreamRate>) {
        let info = &mut self.info;
        info.level_idc = luma_level(info.width, info.height, info.frame_rate, rate).hevc_idc();
    }

    fn begin_frame(&mut self, plan: &FramePlan) {
        self.pending_sao_classes = None;
        self.sao_outcome.begin_picture();
        self.references = plan.references;
        self.active_ref = 0;
        self.frame_index = plan.index;
        self.cached_luma = None;
        match plan.frame_type {
            FrameType::Key => {
                self.slice = SliceKind::Intra;
                // Recoding repeats begin_frame: the sets are committed with
                // the frame (`commit_frame`).
                self.idr = self.sequence_sets.as_ref() != Some(&self.sequence_set_bytes());
                if self.idr {
                    self.key_index = plan.index;
                }
                self.poc = self.picture_order_count(plan.index);
                self.reference_list = ReferenceDistances::one(1);
            }
            FrameType::Inter => {
                self.slice = self.b_schedule.slice(self.inter_slice);
                self.poc = self.picture_order_count(plan.index);
                self.reference_list = self.slice_reference_list();
            }
        }
        self.qp = match plan.frame_type {
            FrameType::Key if !self.rate_control_enabled => {
                nc_encoder::offset_key_quantizer(plan.quantizer, self.key_qp_offset, QP_MIN, QP_MAX)
                    as i32
            }
            _ => plan.quantizer as i32,
        };
        // Planned past QP 51 (`FramePlan::lambda_scale`): λ follows the
        // planned step, and residual is proven zero against it (√ of the λ
        // scale times the coded step), as if quantised there.
        let headroom_step = plan.lambda_scale.sqrt();
        let settings = QuantizationSettings {
            chroma_offset: self.chroma_offset(),
            rounding: self.rounding,
            lambda_scale: self.lambda_scale * plan.lambda_scale,
            column_zero_scale: self.column_zero_scale * headroom_step,
            pre_transform_zero_scale: self.pre_transform_zero_scale * headroom_step,
            headroom_step,
        };
        *self.frame_quantization = settings.at(self.qp);
        if let Some(block_qp) = self.block_qp.as_mut() {
            block_qp.begin_frame(self.qp, settings);
        }
        // Contexts are initialised at the start of every slice (§9.3.2.2).
        self.contexts = Contexts::for_slice(self.slice, self.qp);
        self.items.clear();
        self.item_pricing = None;
        self.level_store.get_mut().clear();
        self.tu_store.clear();
        self.sao = None;
        self.grid.reset();
        self.next_cu = 0;
        self.next_tu = 0;
        self.current.clear();
        self.l1_amvp = None;
        self.early_merge = None;
        if let Some(random) = self.random.as_mut() {
            self.deblocking = random.slice_deblocking();
        } else if !self.deblock_inter {
            self.deblocking = match self.slice {
                SliceKind::Intra => SliceDeblocking::Inherited,
                SliceKind::Predicted | SliceKind::Bi => SliceDeblocking::Override {
                    disabled: true,
                    beta_offset_div2: 0,
                    tc_offset_div2: 0,
                },
            };
        } else if self.info.deblock_override {
            // A frame of a stream whose other frames switch the filter off
            // (`nonbase.hevc.deblock_inter`): the PPS settings.
            self.deblocking = SliceDeblocking::Inherited;
        }
    }

    fn begin_superblock(&mut self, x: usize, y: usize) {
        if let Some(block_qp) = self.block_qp.as_mut() {
            block_qp.begin_group();
        }
        self.items.push(Item::Ctu { x, y });
    }

    fn write_partition(&mut self, node: &Block, split: bool) {
        self.items.push(Item::Split { node: *node, split });
    }

    fn intra_alternatives(
        &mut self,
        block: &Block,
        _decision: &BlockDecision,
        source: &Picture,
        recon: &mut Picture,
    ) -> IntraChoice {
        self.key_intra_alternatives(block, source, recon)
    }

    fn intra_alternative_estimate(&self, rank: u8) -> Option<f32> {
        self.key_alternative_estimate(rank)
    }

    fn encode_block(
        &mut self,
        block: &Block,
        decision: &BlockDecision,
        source: &Picture,
        reference: Option<&Picture>,
        recon: &mut Picture,
    ) {
        self.begin_block_qp(decision.qp_offset);
        self.active_ref = decision.reference.single_index();
        match decision.mode {
            mode if self.random.is_some() => {
                self.encode_random_cu(block, mode, source, reference, recon);
            }
            PredictionMode::IntraDc => {
                let fallback = decision.intra_fallback;
                match self.inter_fallback(block, fallback, source, reference, recon) {
                    Ok((mv, reference)) => {
                        self.encode_inter_cu(block, mv, source, reference, recon);
                    }
                    Err(searched) => {
                        let choice = self.leaf_alternative(block, decision.intra, searched);
                        self.encode_intra_cu(block, source, recon, choice);
                    }
                }
            }
            PredictionMode::Inter(mv) => {
                let reference = reference.expect("inter blocks need a reference picture");
                self.encode_inter_cu(block, mv, source, reference, recon);
            }
        }
        self.end_block_qp();
        self.current.clear();
        self.cached_luma = None;
        self.early_merge = None;
        // The next CU's decision starts on the structure's reference.
        self.active_ref = 0;
    }

    fn filter_frame(&mut self, source: &Picture, recon: &mut Picture) {
        if let Some((beta_offset_div2, tc_offset_div2)) =
            self.deblocking.offsets_div2(self.info.deblock_offsets_div2)
        {
            let strength = Strength {
                chroma_qp_offset: self.info.chroma_qp_offset,
                beta_offset_div2,
                tc_offset_div2,
            };
            deblock(recon, &self.grid, &strength);
        }
        if self.random.is_some() {
            self.random_sao(recon);
            return;
        }
        if !(self.sao_components.luma || self.sao_components.chroma) {
            return;
        }
        // SAO types are searched on key frames and every
        // `hevc.sao_search_interval`-th frame, and reused (offsets
        // re-derived) in between.
        let search = self.slice == SliceKind::Intra
            || u64::from(self.poc).is_multiple_of(self.sao_components.search.interval)
            || self.sao_classes.is_empty();
        let candidates = if search {
            Candidates::Search
        } else {
            Candidates::Reuse(&self.sao_classes)
        };
        let predicted = self.slice.is_inter();
        let skip_shares = self.grid.skip_shares_ctbs();
        let usage = self.sao_usage.filter(|_| predicted).map(|gate| {
            self.sao_outcome
                .usage_skips(&gate, skip_shares.len(), |ctb| skip_shares[ctb])
        });
        let skipped = nc_encoder::merge_skips(
            (self.sao_components.skip_unchanged && predicted).then(|| self.grid.skipped_ctbs()),
            usage,
        );
        let lambda = f64::from(self.lambda());
        let decision = sao::decide(
            (source, recon),
            [lambda, lambda * self.chroma_lambda_ratio()],
            self.sao_components,
            candidates,
            skipped.as_deref(),
            sao::SaoModels::for_slice(self.slice, self.qp),
        );
        if search && self.sao_components.search.interval > 1 {
            // A skipped CTB's "off" says nothing about its types: keep the
            // last search's until the CTB is coded again.
            let mut classes = sao::types(&decision);
            if let Some(skipped) = &skipped
                && self.sao_classes.len() == classes.len()
            {
                for ((class, old), &skip) in classes.iter_mut().zip(&self.sao_classes).zip(skipped)
                {
                    if skip {
                        *class = *old;
                    }
                }
            }
            self.pending_sao_classes = Some(classes);
        }
        self.sao_outcome.observe(
            sao::types(&decision),
            |ctb| !skipped.as_ref().is_some_and(|s| s[ctb]),
            |types| types.iter().any(Option::is_some),
            |ctb| skip_shares[ctb],
        );
        sao::apply(recon, &decision);
        self.sao = Some(decision);
    }

    fn end_frame(&mut self) -> Vec<u8> {
        let data = write_slice_data(
            &self.items,
            (self.level_store.get_mut(), &self.tu_store),
            self.slice,
            self.qp,
            self.info.block_qp,
            &self.grid,
            TransformSyntax {
                scans: &self.scans,
                max_log2: self.info.max_transform_log2,
                inter_depth: self.info.inter_transform_depth,
                intra_depth: self.info.intra_transform_depth,
            },
            self.sao.as_ref().map(|sao| (sao, self.sao_components)),
            self.info.merge_candidates,
        );

        let qp = SliceQp {
            delta: self.qp - self.info.init_qp,
            chroma_offset: self.info.chroma_policy.map(|_| self.chroma_offset()),
        };
        let mut header = BitWriter::new();
        let nal_type = match self.slice {
            SliceKind::Intra => {
                let cra = (!self.idr).then(|| CraPoc {
                    lsb: self.poc & ((1 << self.info.poc_lsb_bits) - 1),
                    lsb_bits: self.info.poc_lsb_bits,
                    sps_sets: self.info.reference_sets().len(),
                    long_term_refs: self.info.kept_slot.long_term_refs(),
                });
                irap_slice_header(&mut header, cra, qp, self.slice_sao(), self.deblocking);
                if self.idr {
                    NalType::IdrWRadl
                } else {
                    NalType::Cra
                }
            }
            SliceKind::Predicted | SliceKind::Bi => {
                let lsb = self.poc & ((1 << self.info.poc_lsb_bits) - 1);
                let sets = self.info.reference_sets();
                let (short_term, long_term) = if self.info.kept_slot.is_kept() {
                    self.long_term_set()
                } else {
                    (Vec::new(), Vec::new())
                };
                let references = if self.info.kept_slot.is_kept() {
                    SliceReferenceSet::Explicit {
                        entries: &short_term,
                        sps_count: sets.len(),
                        long_term: self
                            .info
                            .kept_slot
                            .long_term_refs()
                            .then_some(&long_term[..]),
                    }
                } else {
                    SliceReferenceSet::Sps {
                        index: self.reference_set_index(sets),
                        count: sets.len(),
                    }
                };
                inter_slice_header(
                    &mut header,
                    (lsb, self.info.poc_lsb_bits),
                    references,
                    qp,
                    self.slice_sao(),
                    (self.temporal_mvp(), self.slice == SliceKind::Bi),
                    (self.reference_list.count(), self.info.merge_candidates),
                    self.deblocking,
                );
                if self.references.position.is_top_layer() {
                    NalType::TrailN
                } else {
                    NalType::TrailR
                }
            }
        };
        let temporal_id = self.references.position.layer;
        let mut rbsp = header.into_bytes();
        rbsp.extend_from_slice(&data.into_bytes());

        let mut unit = Vec::with_capacity(rbsp.len() + 128);
        if self.slice == SliceKind::Intra {
            nal::write_nal_unit(&nal_header(NalType::Vps, 0), &vps(&self.info), &mut unit);
            nal::write_nal_unit(&nal_header(NalType::Sps, 0), &sps(&self.info), &mut unit);
            nal::write_nal_unit(&nal_header(NalType::Pps, 0), &pps(&self.info), &mut unit);
            // A prefix SEI precedes the picture's first slice (§7.4.2.4.4).
            if let Some(sei) = &self.encoder_info {
                nal::write_nal_unit(&nal_header(NalType::PrefixSei, 0), sei, &mut unit);
            }
        } else if let Some(sei) = self.recovery_point() {
            nal::write_nal_unit(
                &nal_header(NalType::PrefixSei, temporal_id),
                &sei,
                &mut unit,
            );
        }
        nal::write_nal_unit(&nal_header(nal_type, temporal_id), &rbsp, &mut unit);
        unit
    }

    fn commit_frame(&mut self) {
        if let Some(classes) = self.pending_sao_classes.take() {
            self.sao_classes = classes;
        }
        self.sao_outcome.commit();
        if self.slice == SliceKind::Intra && self.idr {
            self.sequence_sets = Some(self.sequence_set_bytes());
        }
        if self.inter_slice == SliceKind::Bi && self.slice.is_inter() {
            self.b_schedule.coded(self.slice, self.grid.bi_share());
        }
        // The picture's motion becomes the collocated motion of the slots
        // it refreshed (the grid is reset for the next picture).
        self.motion.save(
            &mut self.grid,
            self.reference_list.by_index(),
            self.references.refresh,
        );
        // A key frame's empty set releases every picture (§8.3.2).
        self.long_term_poc = if self.slice.is_inter() && self.long_term_signalled() {
            Some(self.slot_poc[self.long_term_slot])
        } else if self.slice.is_inter() {
            self.long_term_poc
        } else {
            None
        };
        for (slot, poc) in self.slot_poc.iter_mut().enumerate() {
            if self.references.refresh.contains(slot) {
                *poc = self.poc;
            }
        }
    }
}

impl HevcBackend {
    /// The recovery point SEI (§D.2.8) of a picture starting an intra-refresh
    /// sweep: decoding from it, pictures are exact from the one the sweep
    /// completes on, `recovery_poc_cnt` pictures later (POC follows the
    /// frame index, as the sweep's count does). None past the POC LSB range
    /// the count must stay within (§D.3.8).
    fn recovery_point(&self) -> Option<Vec<u8>> {
        let window = self.refresh.as_ref().filter(|w| w.sweep_start)?;
        let frames = i32::try_from(window.recovery_frames).ok()?;
        if frames >= 1 << (self.info.poc_lsb_bits - 1) {
            return None;
        }
        let mut w = BitWriter::new();
        w.put_se(frames); // recovery_poc_cnt
        w.put_bit(true); // exact_match_flag: clean blocks never read dirty samples
        w.put_bit(false); // broken_link_flag
        Some(nal::sei_rbsp(
            nal::SEI_RECOVERY_POINT,
            &nal::sei_payload_bytes(w),
        ))
    }

    /// The VPS and SPS this picture's coded video sequence needs.
    fn sequence_set_bytes(&self) -> Vec<u8> {
        [vps(&self.info), sps(&self.info)].concat()
    }

    /// POC of frame `index`: its distance from the last IDR picture. POC
    /// follows the frame index, not a count of coded pictures: when rate
    /// control drops a top-layer picture its POC is simply absent, and the
    /// SPS reference sets (POC deltas of the structure) stay valid.
    fn picture_order_count(&self, index: u64) -> u32 {
        u32::try_from(index - self.key_index)
            .expect("a coded video sequence is shorter than 2^32 frames")
    }

    /// The slice reference set with a long-term slot (§7.4.8): the picture
    /// this one predicts from (used), and, kept for later pictures unless
    /// this one replaces them, slot 0's T0 picture and the long-term slot's
    /// picture. A recovery picture (predicting from the long-term slot)
    /// replaces every short-term slot, so the pictures it follows, which
    /// the receiver may have lost, leave the set. Returns the short-term
    /// entries (distance, used), nearest first, and the long-term ones.
    ///
    /// With an LTR ([`KeptSlot::LongTerm`]) the long-term slot's picture is a long-term
    /// entry ([`Self::long_term_signalled`]), so its distance is not bound
    /// by the POC LSBs (§8.3.2); otherwise (the second reference,
    /// `refs.second`) it is one more short-term entry.
    fn long_term_set(&self) -> (Vec<(u32, bool)>, Vec<LongTermEntry>) {
        let distance = |slot: usize| self.poc - self.slot_poc[slot];
        let is_long_term = |slot: usize| self.is_long_term(slot);
        let mut set: Vec<(u32, bool)> = Vec::new();
        let mut long_term = Vec::new();
        let mut list = |slot: usize, used: bool| {
            if is_long_term(slot) {
                if long_term.is_empty() {
                    long_term.push(LongTermEntry::new(
                        self.slot_poc[slot],
                        self.poc,
                        self.info.poc_lsb_bits,
                        used,
                    ));
                }
            } else if set.iter().all(|&(listed, _)| listed != distance(slot)) {
                set.push((distance(slot), used));
            }
        };
        list(self.references.reference.slot, true);
        // The second reference (`refs.second`) is the long-term slot's
        // picture, used: its POC distance is larger than the first
        // reference's, so the sorted set lists it second (`ref_idx_l0` 1).
        if let Some(second) = self.references.secondary {
            list(second.slot, true);
        }
        for kept in [0, self.long_term_slot] {
            if !self.references.refresh.contains(kept) {
                list(kept, false);
            }
        }
        set.sort_unstable();
        (set, long_term)
    }

    /// Whether the picture in `slot` is a long-term entry of the current
    /// picture: the long-term slot's picture when it is signalled as one.
    fn is_long_term(&self, slot: usize) -> bool {
        self.long_term_signalled() && self.slot_poc[slot] == self.slot_poc[self.long_term_slot]
    }

    /// Whether the current picture names the long-term slot's picture as a
    /// long-term entry (`long_term_ref_pics_present_flag`): with an LTR,
    /// once an earlier picture has, or once neither the picture's reference
    /// nor its kept T0 picture is that same picture. The pictures right
    /// after it is marked predict from it as their short-term reference (or
    /// keep it as T0) and list it so; a picture marked long-term never
    /// returns to short-term (§8.3.2).
    fn long_term_signalled(&self) -> bool {
        if !self.info.kept_slot.long_term_refs() {
            return false;
        }
        let held = self.slot_poc[self.long_term_slot];
        if self.long_term_poc == Some(held) {
            return true;
        }
        let slot = self.references.reference.slot;
        let reference_is_held = slot != self.long_term_slot && self.slot_poc[slot] == held;
        let t0_is_held = !self.references.refresh.contains(0) && self.slot_poc[0] == held;
        !(reference_is_held || t0_is_held)
    }

    /// The SPS reference picture set of the current picture: its reference
    /// (used), plus slot 0's T0 picture (kept, not used) when predicting from
    /// another slot, since the next T0 predicts from it. Adaptive structures
    /// change the T0 distance per mini-GOP; every combination is in the
    /// stream's set list.
    ///
    /// # Panics
    ///
    /// If the engine's structure produced a set the SPS does not list.
    fn reference_set_index(&self, sets: &[ReferenceSet]) -> usize {
        let slot = self.references.reference.slot;
        let used = (self.poc - self.slot_poc[slot], true);
        let kept = (slot != 0).then(|| (self.poc - self.slot_poc[0], false));
        let wanted = || std::iter::once(used).chain(kept);
        sets.iter()
            .position(|set| set.iter().copied().eq(wanted()))
            .expect("the SPS lists every reference set the structure produces")
    }

    fn predictors(&self, block: &Block) -> Predictors {
        self.predictors_for(block, self.active_ref)
    }

    /// The merge list of `block` and its list 0 AMVP list for `ref_idx`.
    fn predictors_for(&self, block: &Block, ref_idx: u8) -> Predictors {
        let size = block.size();
        // The collocated picture is `ref_idx_l0` 0 (`collocated_ref_idx` 0).
        let collocated = self.motion.collocated(self.references.reference.slot);
        let collocated = self.temporal_mvp().then_some(&collocated);
        let (merge, amvp) = candidate_lists(
            &self.grid,
            collocated,
            (block.x, block.y, size),
            (self.slice_lists(), ref_idx),
        );
        Predictors {
            x: block.x,
            y: block.y,
            log2: block.log2_size(),
            ref_idx,
            merge,
            amvp,
        }
    }

    /// What the current slice's candidate lists depend on.
    fn slice_lists(&self) -> SliceLists<'_> {
        SliceLists {
            merge_len: self.info.merge_candidates,
            references: &self.reference_list,
            bi: self.slice == SliceKind::Bi,
        }
    }

    /// List 1's AMVP list of `block` for reference `ref_idx` (cached for the
    /// block being priced).
    fn l1_amvp(&mut self, block: &Block, ref_idx: u8) -> [Mv; AMVP_CANDIDATES] {
        let key = (block.x, block.y, block.log2_size(), ref_idx);
        if let Some(cached) = self.l1_amvp.filter(|c| c.key == key) {
            return cached.list;
        }
        let collocated = self.motion.collocated(self.references.reference.slot);
        let list = amvp_candidates(
            &self.grid,
            self.temporal_mvp().then_some(&collocated),
            (block.x, block.y, block.size()),
            &self.reference_list,
            (List::L1, ref_idx),
        );
        self.l1_amvp = Some(L1Amvp { key, list });
        list
    }

    /// Estimated bits of a bi-predicted unit of `block` with `motion`:
    /// merge where a candidate is that motion, else AMVP of both lists.
    fn bi_mode_bits(&mut self, block: &Block, motion: Motion) -> f32 {
        let l1 = motion.l1.expect("bi-predicted motion");
        let p = self
            .current
            .get(block, motion.ref_idx)
            .unwrap_or_else(|| self.predictors_for(block, motion.ref_idx));
        if let Some((idx, _)) = p.merge.predicting(motion) {
            return cost::SKIP_FLAG + cost::merge_idx(idx, self.info.merge_candidates);
        }
        let best = |mv: Mv, list: &[Mv]| {
            list.iter()
                .map(|&pred| mvd_bits(mv, pred))
                .fold(f32::INFINITY, f32::min)
        };
        let l1_amvp = self.l1_amvp(block, l1.ref_idx);
        cost::SKIP_FLAG
            + cost::MODE_AND_PART
            + cost::FLAG
            + cost::INTER_PRED_IDC_BI
            + 2.0 * (cost::FLAG + self.ref_idx_bits())
            + best(motion.mv, &p.amvp)
            + best(l1.mv, &l1_amvp)
    }

    /// The reference list of the current P slice: the structure's reference,
    /// and the second reference (`refs.second`) when the engine gave one, by
    /// POC distance, and which is a long-term picture.
    fn slice_reference_list(&self) -> ReferenceDistances {
        let distance = |slot: usize| (self.poc - self.slot_poc[slot]) as i32;
        let slot = self.references.reference.slot;
        let primary = distance(slot);
        let list = match self.references.secondary {
            Some(second) => ReferenceDistances::two(primary, distance(second.slot)),
            None => ReferenceDistances::one(primary),
        };
        // A recovery picture predicts from the long-term picture (§8.5.3.2:
        // its vectors are never scaled).
        if self.is_long_term(slot) {
            list.with_long_term(0)
        } else {
            list
        }
    }

    /// Estimated bits of a uni-predicted AMVP unit's `inter_pred_idc` (B
    /// slices only).
    fn uni_inter_pred_idc_bits(&self) -> f32 {
        if self.slice == SliceKind::Bi {
            cost::INTER_PRED_IDC_UNI
        } else {
            0.0
        }
    }

    /// Estimated bits of `ref_idx_l0` in this slice (none with one
    /// reference).
    fn ref_idx_bits(&self) -> f32 {
        if self.reference_list.count() > 1 {
            cost::REF_IDX
        } else {
            0.0
        }
    }

    /// Whether this picture uses temporal motion vector prediction (TMVP,
    /// §8.5.3.2.8). Not under intra refresh: after a loss the decoder's
    /// collocated motion differs from the encoder's, so a merge index would
    /// select a different vector and the clean region would not recover.
    fn temporal_mvp(&self) -> bool {
        self.refresh.is_none()
    }

    /// The candidate list (§8.4.2) of the prediction block at `(x, y)`.
    fn most_probable_modes(&self, x: usize, y: usize) -> [u8; 3] {
        let (left, above) = self.grid.neighbour_intra_modes(x, y);
        most_probable(left, above)
    }

    /// The reference samples of a transform block of `plane` at `(x, y)`
    /// (plane coordinates), with availability in z-scan order (§6.4.1).
    fn intra_refs(&self, recon: &Picture, plane: PlaneIndex, x: usize, y: usize, n: usize) -> Refs {
        // Chroma positions map to luma at twice the coordinates (4:2:0).
        let scale = usize::from(plane != PlaneIndex::Y);
        let (lx, ly) = (x << scale, y << scale);
        // Availability is per 4x4 luma block: 4 luma or 2 chroma samples.
        let unit = MIN_TB_SIDE >> scale;
        // One z-scan test per neighbouring region instead of one per unit:
        // exact; Fast instructions fall 1.0% per P frame and 3.8% per key
        // frame (callgrind, crowd09 QP 47).
        // The block's own left and above neighbours always precede it
        // ([`ReferenceAvailability`]), so one answer covers each of those
        // sides: exact.
        let availability = self.grid.reference_availability(lx, ly, n << scale);
        Refs::gather(
            recon.plane(plane),
            (x, y),
            n,
            unit,
            Sides::Uniform,
            |xn, yn| availability.at(xn << scale, yn << scale),
        )
    }

    /// Consume only a matching exact luma prediction. The one-reference frame
    /// contract guarantees its reference pixels stay unchanged until this CU.
    fn predict_inter_cu(&mut self, block: &Block, mv: Mv, reference: &Picture) {
        let motion = Motion::uni(mv, self.active_ref);
        let reuse_luma = self.cached_luma.take().is_some_and(|cached| {
            cached.block == *block
                && cached.motion == motion
                && cached.filter == nc_dsp::Filter::HevcLuma
                && cached.frame_index == self.frame_index
        });
        predict_cu(
            &self.dsp,
            block,
            mv,
            reference,
            &mut self.inter_pred,
            reuse_luma,
        );
    }

    fn early_merge_chroma_confident(&self, block: &Block, source: &Picture, limit: f32) -> bool {
        [PlaneIndex::U, PlaneIndex::V].into_iter().all(|plane| {
            let (x, y, n) = plane_rect(plane, block.x, block.y, block.log2_size());
            let source = source.plane(plane).view();
            let src = source.sub_view(x, y, n, n).expect("CU inside picture");
            let pred =
                PlaneView::new(&self.inter_pred[plane as usize], n, n, n).expect("CU prediction");
            let size = BlockSize::new(n, n).expect("chroma CU size");
            self.dsp.variance(size, &src, &pred).sse as f32 <= limit * (n * n) as f32
        })
    }

    /// An inter CU predicted with `mv` from `reference`.
    fn encode_inter_cu(
        &mut self,
        block: &Block,
        mv: Mv,
        source: &Picture,
        reference: &Picture,
        recon: &mut Picture,
    ) {
        let predictors = self
            .current
            .get(block, self.active_ref)
            .unwrap_or_else(|| self.predictors(block));
        let motion = Motion::uni(mv, self.active_ref);
        if self.still_skipped.take() == Some((block.x, block.y)) {
            // `try_static_skip` wrote the reconstruction (the co-located
            // reference samples); only the syntax remains.
            let tus = self.uncoded_transform_units(block);
            self.record_inter_cu(block, motion, &predictors, &tus);
            return;
        }
        self.predict_inter_cu(block, mv, reference);
        let hint = self
            .early_merge
            .take()
            .filter(|hint| hint.matches(self.frame_index, block, motion));
        if hint.is_some_and(|hint| {
            self.early_merge_chroma_confident(block, source, hint.sse_per_sample)
        }) {
            // Prediction remains normative. Chroma must independently pass:
            // quiet luma does not imply quiet colour residuals. A failed UV
            // check retains normal residual coding for the early-selected MV.
            self.place_prediction(block, recon);
            let merged = predictors.merge.get(0);
            self.record_cu(
                block,
                Cell {
                    skip: true,
                    mv,
                    ref_idx: self.active_ref,
                    l1: merged.l1,
                    tu: u32::MAX,
                    ..Cell::default()
                },
                &[],
            );
            self.push_cu(block, CuKind::Skip { merge_idx: 0 }, &[]);
            return;
        }
        let tus = self.code_inter_residual(block, source, recon);
        self.record_inter_cu(block, motion, &predictors, &tus);
    }

    /// A bi-predicted CU (B slices): `motion`'s list 0 half from
    /// `references[0]`, its list 1 half from `references[1]`.
    fn encode_bi_cu(
        &mut self,
        block: &Block,
        motion: Motion,
        source: &Picture,
        references: [&Picture; 2],
        recon: &mut Picture,
    ) {
        let predictors = self
            .current
            .get(block, motion.ref_idx)
            .unwrap_or_else(|| self.predictors_for(block, motion.ref_idx));
        predict_bi_cu(
            &self.dsp,
            block,
            motion,
            references,
            (&mut self.bi_first, &mut self.inter_pred),
        );
        let tus = self.code_inter_residual(block, source, recon);
        self.record_inter_cu(block, motion, &predictors, &tus);
    }

    /// Code the residual of an inter CU from its prediction (`inter_pred`)
    /// and write its reconstruction.
    fn code_inter_residual(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
    ) -> TransformUnits {
        if self.tools.inter_tu_split {
            self.code_inter_tree(block, source, recon)
        } else {
            // The whole CU's prediction is copied once (long rows); its
            // uncoded transform blocks then need no copy of their own. Exact;
            // HEVC Fast Jetson time -1.3% (4 clips), x86 P instructions
            // 0.0% crowd09 / -1.3% lowlight_city_night.
            self.place_prediction(block, recon);
            let mut tus = TransformUnits::default();
            for &position in self.inter_transform_units(block, source).as_slice() {
                tus.push(self.code_inter_tu(block, position, source, recon));
            }
            tus
        }
    }

    /// The transform units of an inter CU coded with no residual (as the
    /// residual path lays them out, every block uncoded).
    fn uncoded_transform_units(&self, block: &Block) -> TransformUnits {
        let mut tus = TransformUnits::default();
        for &(x, y, log2) in transform_units(block, self.info.max_transform_log2).as_slice() {
            tus.push(TransformUnit {
                x,
                y,
                log2,
                luma: PlaneResidual::default(),
                cb: PlaneResidual::default(),
                cr: PlaneResidual::default(),
            });
        }
        tus
    }

    /// Choose the syntax of an inter CU with `motion` and residual `tus`
    /// (skip, merge or AMVP) and record it. `predictors` hold the AMVP list
    /// of `motion`'s reference.
    fn record_inter_cu(
        &mut self,
        block: &Block,
        motion: Motion,
        predictors: &Predictors,
        tus: &TransformUnits,
    ) {
        debug_assert_eq!(predictors.ref_idx, motion.ref_idx);
        let residual = tus.as_slice().iter().any(TransformUnit::coded);
        // A merge candidate predicting the same samples is coded instead,
        // and its motion is the unit's (what later units derive from).
        let (merge_idx, motion) = match predictors.merge.predicting(motion) {
            Some((index, candidate)) => (Some(index), candidate),
            None => (None, motion),
        };
        let kind = match merge_idx {
            Some(merge_idx) if !residual => CuKind::Skip { merge_idx },
            Some(merge_idx) => CuKind::Merge { merge_idx },
            None => {
                let l0 = self.amvp_choice(motion.mv, motion.ref_idx, &predictors.amvp);
                let l1 = motion.l1.map(|l1| {
                    let candidates = self.l1_amvp(block, l1.ref_idx);
                    self.amvp_choice(l1.mv, l1.ref_idx, &candidates)
                });
                CuKind::Amvp { l0, l1 }
            }
        };
        let skip = matches!(kind, CuKind::Skip { .. });
        let cell = Cell {
            skip,
            mv: motion.mv,
            ref_idx: motion.ref_idx,
            l1: motion.l1,
            ..Cell::default()
        };
        self.record_cu(block, cell, tus.as_slice());
        self.push_cu(block, kind, tus.as_slice());
    }

    /// One list's AMVP syntax for `mv` into reference `ref_idx`: the
    /// candidate of `candidates` with the cheapest difference.
    fn amvp_choice(&self, mv: Mv, ref_idx: u8, candidates: &[Mv]) -> AmvpList {
        let (mvp_idx, pred) = candidates
            .iter()
            .copied()
            .enumerate()
            .min_by(|a, b| mvd_bits(mv, a.1).total_cmp(&mvd_bits(mv, b.1)))
            .expect("two AMVP candidates");
        AmvpList {
            mv,
            ref_idx: (self.reference_list.count() > 1).then_some(ref_idx),
            pred,
            mvp_idx,
        }
    }

    /// Copy the inter prediction of `block` (all planes) into `recon`.
    fn place_prediction(&self, block: &Block, recon: &mut Picture) {
        for plane in PlaneIndex::ALL {
            let (x, y, n) = plane_rect(plane, block.x, block.y, block.log2_size());
            let mut out = recon.plane_mut(plane).view_mut();
            copy_square(&mut out, (x, y), n, &self.inter_pred[plane as usize], n);
        }
    }

    /// Code the three planes of the inter transform unit `(x, y, log2)` of
    /// `block` from the CU's prediction.
    #[inline]
    fn code_inter_tu(
        &self,
        block: &Block,
        (x, y, log2): (usize, usize, u8),
        source: &Picture,
        recon: &mut Picture,
    ) -> TransformUnit {
        let [luma, cb, cr] = PlaneIndex::ALL.map(|plane| {
            self.code_inter_plane(
                block,
                plane,
                (x, y, log2),
                (source, recon),
                Predicted::InterInPlace,
                Quantisation::Final,
            )
        });
        TransformUnit {
            x,
            y,
            log2,
            luma,
            cb,
            cr,
        }
    }

    /// Code `plane` of the inter transform unit `(x, y, log2)` (luma
    /// position and size) of `block` from the CU's prediction.
    #[inline]
    fn code_inter_plane(
        &self,
        block: &Block,
        plane: PlaneIndex,
        (x, y, log2): (usize, usize, u8),
        (source, recon): (&Picture, &mut Picture),
        predicted: Predicted,
        quantisation: Quantisation,
    ) -> PlaneResidual {
        let (px, py, size) = plane_rect(plane, x, y, log2);
        // The transform unit's part of the CU prediction.
        let (cu_x, cu_y, cu_n) = plane_rect(plane, block.x, block.y, block.log2_size());
        let start = (py - cu_y) * cu_n + px - cu_x;
        let pred = &self.inter_pred[plane as usize][start..];
        self.code_residual(
            plane,
            px,
            py,
            size,
            (pred, cu_n),
            source,
            recon,
            predicted,
            ScanKind::Diagonal,
            quantisation,
        )
    }

    /// Preserve large chroma transforms for coherent residuals, while
    /// allowing smaller/faster transforms for diffuse fine detail. This is an
    /// approximate size decision; reconstruction and syntax remain normative.
    /// Threshold0.5 saves12.36%/8.07% native CPU (bodycam/crowd). Full21 TRAIN
    /// BD-rate is +0.104% PSNR/-1.199% VMAF v1; worst fixed-noise losses are
    /// 0.0941dB/0.2411V1. Together with SAO row2, supported VAL improves
    /// 0.562% PSNR/1.648% VMAF v1 and native CPU falls17.53%/12.82%;
    /// one high-noise400kbps point loses0.246V1. This measured combination
    /// is the default; both controls remain tunable.
    fn inter_transform_units(&self, block: &Block, source: &Picture) -> TuPositions {
        let base = transform_units(block, self.info.max_transform_log2);
        if self.info.inter_transform_depth == 0 || block.log2_size() < 5 {
            return base;
        }
        let mut out = TuPositions::default();
        for &(x, y, log2) in base.as_slice() {
            if log2 == 5 && self.prefer_tu16(block, source, x, y) {
                for child in (Block::square(x, y, 5)).quadrants() {
                    out.push((child.x, child.y, child.log2_size()));
                }
            } else {
                out.push((x, y, log2));
            }
        }
        out
    }

    fn prefer_tu16(&self, block: &Block, source: &Picture, x: usize, y: usize) -> bool {
        let zero_limit = self.frame_quantization.planes[1].transforms[0][2].zero_limit;
        // The four 8x8 quadrants' statistics of each plane: their SSEs add
        // up to the 16x16 SSE of the zero bound, and their sums give the
        // coarse energy. (One 16x16 variance per plane, then the quadrants
        // for nearly every TU, measured 10 kernel calls and ~900
        // instructions of glue per call; HEVC Faster.)
        let quadrants = [PlaneIndex::U, PlaneIndex::V].map(|plane| {
            let (px, py, _) = plane_rect(plane, x, y, 5);
            let (cu_x, cu_y, stride) = plane_rect(plane, block.x, block.y, block.log2_size());
            let offset = (py - cu_y) * stride + px - cu_x;
            let src = source
                .plane(plane)
                .view()
                .sub_view(px, py, 16, 16)
                .expect("chroma TU");
            let pred = PlaneView::new(&self.inter_pred[plane as usize][offset..], stride, 16, 16)
                .expect("chroma prediction");
            chroma_quadrants(&self.dsp, &src, &pred)
        });
        let negligible = quadrants.iter().all(|quads| {
            let sse: u64 = quads.iter().map(|q| u64::from(q.sse)).sum();
            // Cauchy bounds L1 by sqrt(area*SSE), so this avoids quadrant
            // analysis for residuals whose large chroma transform is zero.
            let squared_bound = sse * 256;
            let root = squared_bound.isqrt();
            let abs_bound = root + u64::from(root * root != squared_bound);
            forward_bound(4, abs_bound) <= zero_limit
        });
        if negligible {
            return false;
        }
        let (energy, squared_error) = quadrants
            .iter()
            .map(coarse_energy)
            .fold((0, 0), |(e, s), (qe, qs)| (e + qe, s + qs));
        energy * 65536 < squared_error * 64 * u64::from(self.tu16_chroma_concentration_q16)
    }

    // Optional staged rows approximate forward coefficients; normative inverse
    // reconstruction is unchanged. Packed16+32 alone saved 3.28%/1.70% native
    // bodycam/crowd CPU, with +0.05% PSNR/+0.10% V1 BD-rate on natural TRAIN.
    // Evidence: docs/handoff/overnight.md (hevc-row-narrow-agent).
    fn forward_residual(
        &self,
        transform: HevcTransform,
        intra: bool,
        residual: &[i16],
        limit: u32,
        out: &mut [i16],
    ) -> bool {
        let n = transform.side();
        // Intra residuals and block-QP refinements (the reuse term restoring
        // the detail a degraded reference lost) keep every frequency.
        let partial = !intra
            && !self.refining_block()
            && nc_encoder::WorkGate::partial_transform(&self.gates)
                .admits_quantizer(self.qp as f32 / QP_MAX as f32);
        let mode = self.fast_forward.mode(n, partial);
        if mode.staging != HevcForwardStaging::Exact || mode.low_frequencies {
            self.dsp
                .hevc_forward_staged_unless_zero(transform, residual, n, limit, mode, out)
        } else if n >= 16 {
            self.dsp
                .hevc_forward_unless_zero(transform, residual, n, limit, out)
        } else {
            self.dsp.hevc_forward(transform, residual, n, out);
            true
        }
    }

    /// Transform, quantise and reconstruct one plane of a transform block
    /// whose prediction is `pred` (samples, row stride). Scratch is reused;
    /// a coded block's levels are appended to the picture's level store for
    /// the syntax writer.
    fn code_residual(
        &self,
        plane: PlaneIndex,
        x: usize,
        y: usize,
        n: usize,
        (pred, pred_stride): (&[u8], usize),
        source: &Picture,
        recon: &mut Picture,
        predicted: Predicted,
        scan: ScanKind,
        quantisation: Quantisation,
    ) -> PlaneResidual {
        let intra = predicted == Predicted::Intra;
        let log2 = n.trailing_zeros() as u8;
        let quantization = &self.frame_quantization.planes[usize::from(plane != PlaneIndex::Y)]
            .transforms[usize::from(intra)][usize::from(log2 - 2)];
        let size = BlockSize::new(n, n).expect("transform sizes 4..32");
        let src = source.plane(plane).view();
        let src_block = src.sub_view(x, y, n, n).expect("block inside the picture");
        let pred_block =
            PlaneView::new(pred, pred_stride, n, n).expect("prediction covers the block");
        // Most blocks are proven zero by the SAD bound below, so the SAD is
        // taken first and they return before any transform set-up (scratch,
        // quantiser, reconstruction view): the zero path runs for most
        // transform blocks of every P frame. Exact; HEVC Faster x86
        // instructions -1.7% (callgrind, chart clips, fixed QP 150 f).
        //
        // On the default exact transform path, scale1 proves no coefficient
        // survives quantisation. Larger inter thresholds deliberately discard
        // near-zero detail. Scale2 together with packed rows and SAO row4 saved
        // 12.71%/8.97% native bodycam/crowd CPU at +0.81% PSNR/+0.84% V1 BD-rate
        // over natural TRAIN; these are joint, not isolated, threshold results.
        // See docs/handoff/overnight.md (control-fulltrain).
        let abs_sum = self.dsp.sad(size, &src_block, &pred_block);
        if abs_sum < quantization.pre_transform_sad_limit {
            if predicted != Predicted::InterInPlace {
                let mut out = recon.plane_mut(plane).view_mut();
                copy_square(&mut out, (x, y), n, pred, pred_stride);
            }
            return PlaneResidual {
                log2,
                scan,
                ..PlaneResidual::default()
            };
        }
        let qp = self.frame_quantization.planes[usize::from(plane != PlaneIndex::Y)].qp;
        let mut scratch = self.scratch.borrow_mut();
        let Scratch {
            residual,
            levels,
            coeffs,
            scaled,
            decoded,
            ..
        } = &mut *scratch;
        // 4x4 intra luma uses the DST (§8.6.4.2).
        let transform = crate::transform::kind(log2, intra && plane == PlaneIndex::Y)
            .expect("transform sizes 4..32");
        let quantizer = &quantization.quantizer;
        let levels = &mut levels[..n * n];
        self.dsp
            .residual(size, &src_block, &pred_block, &mut residual[..], n);
        let residual = &residual[..n * n];
        // A second exact bound after the row pass skips column work
        // and quantisation on proven-zero blocks. Including the bound,
        // bodycam/crowd P instructions fall3.44%/3.04% at QP28/39.
        let transformed = self.forward_residual(
            transform,
            intra,
            residual,
            quantization.column_zero_limit,
            &mut coeffs[..n * n],
        );
        let (nonzero, sub_blocks) = if transformed {
            self.quantise_levels(
                plane,
                (&coeffs[..n * n], levels),
                quantization,
                self.scans.get(log2, scan),
                quantisation,
            )
        } else {
            (0, CodedGroups::default())
        };
        let coded = nonzero > 0;
        // Parity adjustment never zeroes or creates a coded block, so an
        // all-zero block (most transformed inter blocks) skips it.
        // A lone DC has no sign-hiding span and, under a DCT, reconstructs
        // to a constant (the DST's first basis is not flat).
        // Exact shortcut: bodycam/crowd P instructions fall 3.79%/0.99%
        // at QP28/39, with unchanged streams and reconstruction.
        let dc_only = nonzero == 1 && levels[0] != 0 && transform != HevcTransform::Dst4;
        if coded && SIGN_DATA_HIDING && !dc_only {
            hide_signs(
                &coeffs[..n * n],
                (levels, sub_blocks),
                self.scans.get(log2, scan),
                qp,
            );
        }

        let mut out = recon.plane_mut(plane).view_mut();
        if coded {
            let residual = self.decode_levels(
                transform,
                quantizer,
                levels,
                dc_only,
                (&mut scaled[..n * n], &mut decoded[..n * n]),
            );
            let mut block = out
                .sub_view_mut(x, y, n, n)
                .expect("block inside the picture");
            self.dsp
                .reconstruct_i16(size, &pred_block, residual, &mut block);
        } else if predicted != Predicted::InterInPlace {
            copy_square(&mut out, (x, y), n, pred, pred_stride);
        }
        let entry = coded.then(|| self.store_levels(levels, sub_blocks));
        PlaneResidual {
            log2,
            scan,
            levels: entry.unwrap_or(0),
            coded,
        }
    }

    /// Quantise a transformed block's `coeffs` into `levels` (the dead
    /// zone, shaving and, for final blocks, the trellis); returns the
    /// number of non-zero levels (or an upper bound once shaving dropped
    /// some; 0 exactly when every level is zero) and the coded sub-blocks.
    fn quantise_levels(
        &self,
        plane: PlaneIndex,
        (coeffs, levels): (&[i16], &mut [i16]),
        quantization: &TransformQuantization,
        scan: &Scan,
        quantisation: Quantisation,
    ) -> (u32, CodedGroups) {
        let mut nonzero = quantization.quantize(&self.dsp, coeffs, levels);
        if nonzero == 0 {
            return (0, CodedGroups::default());
        }
        let mut sub_blocks = quantised_sub_blocks(levels, nonzero, scan);
        // A lone level is the block's first and is never shaved;
        // skipping the shave there (most coded inter blocks: a DC)
        // saves its search for the last level. Exact: HEVC Faster x86
        // instructions -0.6% (callgrind, chart clips, CBR 600).
        if self.shave_gap > 0 && nonzero > 1 {
            sub_blocks = shave_isolated_ones(levels, scan, sub_blocks, self.shave_gap);
        }
        let trellis = (quantisation == Quantisation::Final)
            .then(|| self.trellis_for(plane))
            .flatten();
        if let Some(trellis) = trellis {
            nonzero = self.trellis_quantize(trellis, plane, coeffs, scan.log2, scan.kind, levels);
            sub_blocks = quantised_sub_blocks(levels, nonzero, scan);
        } else if let Some(trim) = self
            .rd_trim_for(plane)
            .filter(|_| quantisation == Quantisation::Final)
        {
            nonzero = self.rate_trim(trim, coeffs, scan, (levels, nonzero, sub_blocks));
            sub_blocks = quantised_sub_blocks(levels, nonzero, scan);
        }
        (nonzero, sub_blocks)
    }

    /// The rate-aware trimming for a transform block of `plane` in this
    /// slice: luma only (chroma keeps the plain quantiser, as the trellis
    /// does by default), and only while the trellis is off.
    fn rd_trim_for(&self, plane: PlaneIndex) -> Option<&nc_encoder::RateTrim> {
        if self.trellis.is_some() || plane != PlaneIndex::Y {
            return None;
        }
        self.rd_trim
            .as_ref()
            .filter(|t| t.covers(self.slice == SliceKind::Intra))
    }

    /// `quant.rd_trim` on one luma block: the shared rules
    /// (`nc_encoder::trim_levels`), with distortion in quantiser steps as
    /// the trellis prices it (`trellis_quantize`), from the plain levels'
    /// `nonzero` count (or bound) and coded sub-blocks. Returns the number
    /// of non-zero levels, or a bound on it as `quantise_levels` does.
    fn rate_trim(
        &self,
        trim: &nc_encoder::RateTrim,
        coeffs: &[i16],
        scan: &Scan,
        (levels, nonzero, sub_blocks): (&mut [i16], u32, CodedGroups),
    ) -> u32 {
        // A lone DC above one is untouched by the ±1 rules: only the
        // whole-block rule could change it.
        if nonzero == 1 && levels[0].unsigned_abs() > 1 && !trim.block {
            return nonzero;
        }
        let Some(last) = quant::previous_nonzero(levels, scan, sub_blocks, scan.positions.len())
        else {
            return 0;
        };
        let qp = self.frame_quantization.planes[0].qp;
        let (qbits, scale) = quant::step_terms(qp, scan.log2);
        let step_sse = 2f64.powf(f64::from(qp - QSTEP_UNIT_QP) / 3.0);
        // 2^-qbits: exact, so multiplying equals dividing by 2^qbits.
        let unit = 1.0 / (1_i64 << qbits) as f64;
        let eob = nc_encoder::trim_levels(
            trim,
            levels,
            &scan.positions,
            last + 1,
            f64::from(self.lambda()),
            |p, magnitude| {
                let value = (i64::from(coeffs[p].unsigned_abs()) * scale) as f64 * unit;
                nc_encoder::level_removal_sse(value, magnitude, step_sse)
            },
        );
        // A bound, as `quantise_levels` returns: trimming only removes
        // levels, and an end of block of one is the DC alone.
        match eob {
            0 | 1 => eob as u32,
            _ => nonzero,
        }
    }

    /// The decoded residual of a coded block's `levels` (`scaled` and
    /// `decoded` are scratch of the block's size). A lone DC (`dc_only`)
    /// dequantises alone and decodes to one constant: exact, as the other
    /// levels are zero and dequantise to zero, and it skips scaling and
    /// filling the whole block for most coded inter blocks.
    fn decode_levels<'a>(
        &self,
        transform: HevcTransform,
        quantizer: &HevcQuantizer,
        levels: &[i16],
        dc_only: bool,
        (scaled, decoded): (&mut [i16], &'a mut [i16]),
    ) -> Residual16<'a> {
        if dc_only {
            self.dsp
                .hevc_dequantize(&levels[..1], quantizer, &mut scaled[..1]);
            return Residual16::Constant(inverse_dc(scaled[0]));
        }
        self.dsp.hevc_dequantize(levels, quantizer, scaled);
        self.dsp.hevc_inverse(transform, scaled, decoded);
        Residual16::Samples(decoded)
    }

    /// Append a coded block's levels and coded sub-blocks to the picture's
    /// level store; returns the block's entry.
    ///
    /// Levels are 16-bit throughout (§7.4.9.11): coefficients are at most
    /// 2^15 in magnitude (`nc_dsp::codec_txfm` ranges), the quantiser scale
    /// at most 2^15 and its shift at least 16 at 8 bits, so `|level| <=
    /// 2^14`, and sign hiding and the trellis move a level by one.
    fn store_levels(&self, levels: &[i16], sub_blocks: CodedGroups) -> u32 {
        self.level_store.borrow_mut().push(levels, sub_blocks)
    }

    /// `slice_sao_luma_flag` and `slice_sao_chroma_flag` of the current
    /// picture (the SPS always enables SAO).
    fn slice_sao(&self) -> (bool, bool) {
        (self.sao_components.luma, self.sao_components.chroma)
    }

    /// The current picture's chroma QP offset (`qPiCb` − `QpY`): the PPS
    /// offset, or the engine chroma policy's for the picture QP (PPS 0
    /// plus the slice offset).
    fn chroma_offset(&self) -> i32 {
        match self.info.chroma_policy {
            Some(policy) => policy_chroma_offset(policy, self.qp),
            None => self.info.chroma_qp_offset,
        }
    }

    /// Chroma's λ over luma's for SAO decisions: the squared chroma step's
    /// share, `2^((QpC − QpY) / 3)` (HM's chroma weight, as VVC's
    /// `chroma_lambda`), so chroma offsets are priced at chroma's own step.
    fn chroma_lambda_ratio(&self) -> f64 {
        let chroma = chroma_qp(self.qp + self.chroma_offset());
        2f64.powf(f64::from(chroma - self.qp) / 3.0)
    }

    /// The trellis settings for a transform block of `plane` in this
    /// slice, with that plane's λ multiple, or `None` while it is off.
    fn trellis_for(&self, plane: PlaneIndex) -> Option<(&TrellisQuant, f32)> {
        let trellis = self
            .trellis
            .as_ref()
            .filter(|t| t.settings.covers(self.slice == SliceKind::Intra))?;
        let settings = &trellis.settings;
        let base_scale = settings.lambda_scale_for(self.slice == SliceKind::Intra);
        let scale = if plane == PlaneIndex::Y {
            base_scale
        } else if self.info.chroma_policy.is_some() {
            // The policy weighs chroma error by the squared step ratio,
            // 2^((QPy - QPc)/3) (as HM): chroma λ is luma's over it.
            let [luma, chroma] = self
                .frame_quantization
                .planes
                .each_ref()
                .map(|plane| plane.qp);
            base_scale
                * settings.chroma_scale?
                * 2f32.powf((chroma - luma) as f32 / QP_PER_LAMBDA_DOUBLING)
        } else {
            base_scale * settings.chroma_scale?
        };
        Some((trellis, scale))
    }

    /// Trellis quantisation (`quant.trellis`) of one transform block's
    /// coefficients `coeffs`, starting from the plain quantiser's `levels`
    /// and priced with the slice's contexts as they stand. Returns the
    /// number of non-zero levels. λ is per sample SSE, and a level step of
    /// the plane's QP costs `Qstep² = 2^((qp − 4) / 3)` of it (HEVC
    /// scaling keeps transform coefficients orthonormal in step units), so
    /// chroma at its own QP is traded at its own step, as luma is.
    fn trellis_quantize(
        &self,
        (trellis, lambda_scale): (&TrellisQuant, f32),
        plane: PlaneIndex,
        coeffs: &[i16],
        log2: u8,
        scan: ScanKind,
        levels: &mut [i16],
    ) -> u32 {
        let luma = plane == PlaneIndex::Y;
        let qp = self.frame_quantization.planes[usize::from(!luma)].qp;
        let (qbits, scale) = quant::step_terms(qp, log2);
        let quantizer = BlockQuantizer {
            step_sse: 2f32.powf((qp - QSTEP_UNIT_QP) as f32 / 3.0),
            scale,
            qbits,
        };
        let scan = self.scans.get(log2, scan);
        let mut model = RateModel::new(&self.contexts, &self.bin_costs, scan, luma);
        let lambda = self.lambda() * lambda_scale;
        trellis.search.borrow_mut().quantize(
            &mut model,
            coeffs,
            quantizer,
            (lambda, trellis.settings.max_states),
            levels,
        )
    }

    /// Store what later CUs and the deblocking filter need about a CU.
    /// Record CU `block` coded as `kind` with transform units `tus` for the
    /// slice writer.
    fn push_cu(&mut self, block: &Block, kind: CuKind, tus: &[TransformUnit]) {
        let range = TuRange::new(self.tu_store.len(), tus.len());
        self.tu_store.extend_from_slice(tus);
        self.items.push(Item::Cu(CuRecord {
            block: *block,
            kind,
            tus: range,
        }));
    }

    fn record_cu(&mut self, block: &Block, mut cell: Cell, tus: &[TransformUnit]) {
        cell.qp = self.cu_qp(tus.iter().any(TransformUnit::coded));
        cell.depth = LOG2_CTB as u8 - block.log2_size();
        cell.cu = self.next_cu;
        self.next_cu += 1;
        self.grid.fill(block.x, block.y, block.size(), cell);
        for tu in tus {
            self.grid
                .set_transform(tu.x, tu.y, 1 << tu.log2, self.next_tu, tu.luma.coded);
            self.next_tu += 1;
        }
    }
}

/// Fixed row lengths let the compiler use short vector copies rather than a
/// dynamic memcpy call per row. This only changes all-zero TU reconstruction;
/// prediction samples and the surrounding plane padding remain unchanged.
/// Three native150f pairs save0.76% body/0.98% crowd/0.70% lowlight CPU with
/// exact packets/reconstruction (docs/handoff/hevc.md). On x86 the per-row
/// memcpy call measured ~35 instructions per row (callgrind, `lowlight_city_night`).
/// Skipped CUs copy whole CU predictions the same way.
#[inline]
fn copy_square(
    out: &mut PlaneViewMut<'_>,
    (x, y): (usize, usize),
    n: usize,
    pred: &[u8],
    stride: usize,
) {
    match n {
        4 => copy_prediction::<4>(out, x, y, pred, stride),
        8 => copy_prediction::<8>(out, x, y, pred, stride),
        16 => copy_prediction::<16>(out, x, y, pred, stride),
        32 => copy_prediction::<32>(out, x, y, pred, stride),
        64 => copy_prediction::<64>(out, x, y, pred, stride),
        _ => unreachable!("block sides are powers of two from 4 to 64"),
    }
}

/// The coded sub-blocks of `levels` holding `nonzero` non-zero levels: a
/// lone DC (most coded inter blocks) without a scan.
fn quantised_sub_blocks(levels: &[i16], nonzero: u32, scan: &Scan) -> CodedGroups {
    match nonzero {
        0 => CodedGroups::default(),
        1 if levels[0] != 0 => CodedGroups::FIRST,
        _ => coded_sub_blocks_counted(levels, scan, nonzero),
    }
}

/// One `N`x`N` copy through `nc_dsp::copy_square`: a single range check,
/// then one fixed-size row move per row (the per-row slice checks here
/// cost about as much as the copies; HEVC Faster x86 callgrind).
#[inline]
fn copy_prediction<const N: usize>(
    out: &mut PlaneViewMut<'_>,
    x: usize,
    y: usize,
    pred: &[u8],
    stride: usize,
) {
    let mut dst = out.sub_view_mut(x, y, N, N).expect("block inside the view");
    nc_dsp::copy_square::<N>(&mut dst, pred, stride);
}

/// The residual statistics of the four 8x8 quadrants of a 16x16 chroma TU,
/// in raster order.
fn chroma_quadrants(dsp: &Dsp, src: &PlaneView<'_>, pred: &PlaneView<'_>) -> [nc_dsp::Variance; 4] {
    let mut quads = [nc_dsp::Variance::default(); 4];
    let (top, bottom) = quads.split_at_mut(2);
    dsp.variance_8x8_row(src, pred, top);
    let lower_src = src.sub_view(0, 8, 16, 8).expect("lower quadrants");
    let lower_pred = pred.sub_view(0, 8, 16, 8).expect("lower quadrants");
    dsp.variance_8x8_row(&lower_src, &lower_pred, bottom);
    quads
}

/// Energy captured by the four 8x8 residual means of one 16x16 chroma TU
/// (its [`chroma_quadrants`]), and the TU's SSE.
fn coarse_energy(quads: &[nc_dsp::Variance; 4]) -> (u64, u64) {
    quads.iter().fold((0, 0), |(coarse, sse), stats| {
        let sum = i64::from(stats.sum);
        (coarse + (sum * sum) as u64, sse + u64::from(stats.sse))
    })
}

/// The transform units of a CU, `(x, y, log2)` in luma samples, in coding
/// (z) order.
fn transform_units(block: &Block, max_log2: u8) -> TuPositions {
    fn append(block: &Block, max_log2: u8, out: &mut TuPositions) {
        if block.log2_size() <= max_log2 {
            out.push((block.x, block.y, block.log2_size()));
        } else {
            for child in block.quadrants() {
                append(&child, max_log2, out);
            }
        }
    }
    let mut out = TuPositions::default();
    append(block, max_log2, &mut out);
    out
}

/// Positions of a CU's transform units (no heap allocation).
#[derive(Debug, Default)]
struct TuPositions {
    positions: [(usize, usize, u8); MAX_TUS],
    len: usize,
}

impl TuPositions {
    fn push(&mut self, position: (usize, usize, u8)) {
        self.positions[self.len] = position;
        self.len += 1;
    }

    fn as_slice(&self) -> &[(usize, usize, u8)] {
        &self.positions[..self.len]
    }
}

/// Position and side of a luma transform block's area in `plane`.
fn plane_rect(plane: PlaneIndex, x: usize, y: usize, log2: u8) -> (usize, usize, usize) {
    let shift = usize::from(plane != PlaneIndex::Y);
    (x >> shift, y >> shift, 1 << (usize::from(log2) - shift))
}

/// A coded vector difference component lies in −2^15..2^15 − 1 quarter
/// samples (HEVC §7.4.9.9, `lMvd`).
const MVD_RANGE: std::ops::RangeInclusive<i32> = -(1 << 15)..=(1 << 15) - 1;

/// Bits of `mvd_coding()` for `mv − pred` (exact count of its bins):
/// infinite where a component is outside [`MVD_RANGE`], so that no mode
/// decision chooses a difference the syntax cannot carry. Two engine vectors
/// lie that far apart only on a picture over 4096 samples wide or tall (a
/// second-reference seed at the picture's size against a predicted vector
/// of the other sign; AV1's narrower range was passed at 1280 wide).
fn mvd_bits(mv: Mv, pred: Mv) -> f32 {
    let difference = [
        i32::from(mv.x) - i32::from(pred.x),
        i32::from(mv.y) - i32::from(pred.y),
    ];
    if difference.iter().any(|d| !MVD_RANGE.contains(d)) {
        return f32::INFINITY;
    }
    let component = |d: i32| -> u32 {
        match d.unsigned_abs() {
            0 => 1,
            1 => 3, // greater0, greater1, sign
            a => {
                // The order-k tail adds two bins for every doubling of
                // the quotient, as in VVC's order-one MVD estimate.
                let prefix = (((a - 2) >> MVD_EG_ORDER) + 1).ilog2();
                3 + 2 * prefix + 1 + MVD_EG_ORDER
            }
        }
    };
    (component(difference[0]) + component(difference[1])) as f32
}

/// Motion-compensated prediction of all three planes of `block` from
/// `reference` with `mv`, into `planes` (row-major, one per plane, each at
/// least the CU's size; reused across CUs).
fn predict_cu(
    dsp: &Dsp,
    block: &Block,
    mv: Mv,
    reference: &Picture,
    planes: &mut [Vec<u8>; 3],
    reuse_luma: bool,
) {
    for (plane, pred) in [PlaneIndex::Y, PlaneIndex::U, PlaneIndex::V]
        .into_iter()
        .zip(planes)
    {
        if plane == PlaneIndex::Y && reuse_luma {
            continue;
        }
        let (x, y, n) = plane_rect(plane, block.x, block.y, block.log2_size());
        let predict = if plane == PlaneIndex::Y {
            predict_luma
        } else {
            predict_chroma
        };
        predict(
            dsp,
            reference.plane(plane),
            x,
            y,
            n,
            n,
            mv,
            &mut pred[..n * n],
        );
    }
}

/// Bi-prediction from list 0's first reference at `first_mv` and list 1's
/// reference `ref_idx` at `mv`.
fn bi_motion(first_mv: Mv, (mv, ref_idx): (Mv, u8)) -> Motion {
    Motion {
        mv: first_mv,
        ref_idx: 0,
        l1: Some(ListMotion { mv, ref_idx }),
    }
}

/// Default-weight bi-prediction (HEVC eq. 8-251) of all three planes of
/// `block`: list 0's half from `references[0]` at `motion.mv`, list 1's
/// from `references[1]`, into `planes` (as [`predict_cu`]); `first` holds
/// list 0's centred 14-bit samples (reused across CUs).
fn predict_bi_cu(
    dsp: &Dsp,
    block: &Block,
    motion: Motion,
    references: [&Picture; 2],
    (first, planes): (&mut Vec<i16>, &mut [Vec<u8>; 3]),
) {
    let l1 = motion.l1.expect("bi-predicted motion");
    let kernels = dsp.at(nc_common::BitDepth::EIGHT);
    for (plane, pred) in PlaneIndex::ALL.into_iter().zip(planes) {
        let (x, y, n) = plane_rect(plane, block.x, block.y, block.log2_size());
        let component = if plane == PlaneIndex::Y {
            Component::Luma
        } else {
            Component::CHROMA_420
        };
        predict_samples(
            kernels,
            references[0].plane(plane),
            (x, y),
            (n, n),
            motion.mv,
            component,
            first,
        );
        let mut dst = PlaneViewMut::new(&mut pred[..n * n], n, n, n).expect("CU prediction");
        predict_bi_into(
            kernels,
            references[1].plane(plane),
            (x, y),
            l1.mv,
            component,
            &first[..n * n],
            &mut dst,
        );
    }
}

/// The parameter-set contents of a stream for `config` with `params`.
/// QP steps over which λ (a squared step) doubles.
const QP_PER_LAMBDA_DOUBLING: f32 = 3.0;
/// HEVC's chroma QP offset range (`pps_cb_qp_offset` + `slice_cb_qp_offset`, §7.4.7.1).
const CHROMA_OFFSET_LIMIT: i32 = 12;

/// The chroma QP offset (`qPiCb` − `QpY`) whose Table 8-10 chroma QP at
/// luma QP `qp` is nearest the engine chroma policy's
/// (`nc_encoder::chroma_policy`; HEVC QPs are its unit).
fn policy_chroma_offset(policy: ChromaPolicy, qp: i32) -> i32 {
    policy.closest_qp_offset(
        -CHROMA_OFFSET_LIMIT..=CHROMA_OFFSET_LIMIT,
        f64::from(qp),
        |offset| f64::from(chroma_qp(qp + offset)),
    )
}

fn stream_info(config: &EncoderConfig, params: &HevcParams) -> StreamInfo {
    let policy = ChromaPolicy::from_params(&config.params);
    // Pictures are coded in whole minimum CUs; a conformance window
    // crops the visible size back out (§7.4.3.2.1).
    let (width, height) = coded_picture_size(config.width, config.height);
    StreamInfo {
        width,
        height,
        crop: (width - config.width, height - config.height),
        level_idc: luma_level(width, height, config.frame_rate, None).hevc_idc(),
        init_qp: config.quantizer as i32,
        max_transform_log2: params.max_transform_log2.round() as u8,
        inter_transform_depth: if params.inter_tu_split >= 0.5
            || (params.max_transform_log2.round() >= 5.0
                && (params.tu16_chroma_concentration * 65536.0).round() > 0.0)
        {
            MAX_INTER_TRANSFORM_DEPTH
        } else {
            0
        },
        // With the engine chroma policy the PPS offset is 0 and each slice
        // carries its picture's offset; deblocking reads the PPS offset
        // alone (§8.7.2.5.5), as the policy's map intends.
        chroma_qp_offset: if policy.is_some() {
            0
        } else {
            params.chroma_qp_offset.round() as i32
        },
        chroma_policy: policy,
        deblock_offsets_div2: (
            params.deblock_beta_offset.round() as i32,
            params.deblock_tc_offset.round() as i32,
        ),
        // Only `hevc.deblock_inter` 0, on every frame or on non-base frames
        // alone, needs per-slice overrides (one more bit per slice), so the
        // default PPS stays as it was.
        deblock_override: params.deblock_inter < 0.5 || nonbase_params(config).deblock_inter < 0.5,
        intra_transform_depth: params.intra_tu_depth.round() as u8,
        temporal_layers: config.temporal_layers,
        poc_lsb_bits: if holds_second_reference(&config.params) {
            LONG_POC_LSB_BITS
        } else {
            POC_LSB_BITS
        },
        signal: config.signal,
        frame_rate: config.frame_rate,
        kept_slot: if nc_encoder::long_term_requested(config) {
            KeptSlot::LongTerm
        } else if uses_long_term_slot(&config.params) {
            KeptSlot::ShortTerm
        } else {
            KeptSlot::None
        },
        block_qp: false,
        merge_candidates: params.merge_candidates.round() as usize,
    }
}

/// In-frame rate control's running estimate of a frame's recorded items
/// (`CodecBackend::coded_bits`).
#[derive(Debug)]
struct ItemPricing {
    /// The first item not yet priced.
    next: usize,
    /// Estimated bits of the items before it.
    bits: f64,
    /// The contexts as the items before it left them.
    contexts: Contexts,
}

#[cfg(test)]
mod mvd_range_tests {
    use super::{Mv, mvd_bits};

    #[test]
    fn a_difference_past_the_coded_range_costs_infinite_bits() {
        // 4100 samples right against 4100 samples left: 32800 quarters.
        let right = Mv { x: 16_400, y: 0 };
        let left = Mv { x: -16_400, y: 0 };
        assert!(mvd_bits(right, left).is_infinite());
        assert!(mvd_bits(left, right).is_infinite());
        // The ends of the range itself are coded.
        let pred = Mv { x: -16_384, y: 0 };
        assert!(mvd_bits(Mv { x: 16_383, y: 0 }, pred).is_finite());
        assert!(mvd_bits(pred, Mv { x: 16_384, y: 0 }).is_finite());
        assert!(mvd_bits(Mv { x: 0, y: 16_384 }, Mv { x: 0, y: -16_384 }).is_infinite());
    }
}

#[cfg(test)]
mod frame_quantization_tests {
    use super::{
        FrameQuantization, HevcQuantizer, LAMBDA_AT_QP12, chroma_qp, largest_zero_coefficient,
    };

    #[test]
    fn scaled_pre_transform_bound_changes_only_inter_threshold() {
        for qp in [0, 12, 28, 40, 51] {
            let exact = FrameQuantization::new(qp, -3, [(171, 128); 2], 1.0, 1.0, 1.0, 1.0);
            for scale in [1.5, 2.0, 3.0, 4.0] {
                let scaled = FrameQuantization::new(qp, -3, [(171, 128); 2], 1.0, 1.0, scale, 1.0);
                for (a, b) in exact.planes.iter().zip(&scaled.planes) {
                    for intra in 0..2 {
                        for (index, (x, y)) in a.transforms[intra]
                            .iter()
                            .zip(&b.transforms[intra])
                            .enumerate()
                        {
                            assert_eq!(x.zero_limit, y.zero_limit);
                            assert_eq!(x.column_zero_limit, y.column_zero_limit);
                            assert_eq!(x.quantizer, y.quantizer);
                            let log2 = index as u8 + 2;
                            let limit = if intra == 1 {
                                x.zero_limit
                            } else {
                                (x.zero_limit as f64 * f64::from(scale)) as u64
                            };
                            for (threshold, limit) in [
                                (x.pre_transform_sad_limit, x.zero_limit),
                                (y.pre_transform_sad_limit, limit),
                            ] {
                                for sad in [0, threshold.saturating_sub(1), threshold] {
                                    assert_eq!(
                                        sad < threshold,
                                        crate::transform::forward_bound(log2, u64::from(sad))
                                            <= limit,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cached_weights_and_quantizers_match_uncached_formulas() {
        for qp in 0..=51 {
            for offset in [-12, 0, 12] {
                for rounding in [(0, 0), (171, 128), (256, 256)] {
                    for scale in [0.25, 1.0, 4.0] {
                        let frame =
                            FrameQuantization::new(qp, offset, [rounding; 2], scale, 1.0, 1.0, 1.0);
                        let lambda = scale * LAMBDA_AT_QP12 * 2f32.powf((qp - 12) as f32 / 3.0);
                        assert_eq!(frame.lambda.to_bits(), lambda.to_bits());
                        assert_eq!(frame.sqrt_lambda.to_bits(), lambda.sqrt().to_bits());
                        for (plane, q) in frame.planes.iter().zip([qp, chroma_qp(qp + offset)]) {
                            assert_eq!(plane.qp, q);
                            for (transforms, rounding) in
                                plane.transforms.iter().zip([rounding.1, rounding.0])
                            {
                                for (i, t) in transforms.iter().enumerate() {
                                    let log2 = i as u8 + 2;
                                    assert_eq!(
                                        t.quantizer,
                                        HevcQuantizer::with_rounding(q as u8, log2, rounding)
                                    );
                                    assert_eq!(u64::from(t.column_zero_limit), t.zero_limit);
                                    assert_eq!(
                                        t.zero_limit,
                                        largest_zero_coefficient(q, log2, rounding)
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod transform_unit_tests {
    use super::transform_units;
    use nc_encoder::Block;

    #[test]
    fn transform_tiles_follow_morton_order_and_cover_each_cu() {
        for max_log2 in [4, 5] {
            for log2_size in 3..=6 {
                let block = Block::square(8, 24, log2_size);
                let positions = transform_units(&block, max_log2);
                let leaf_log2 = log2_size.min(max_log2);
                let depth = log2_size - leaf_log2;
                assert_eq!(positions.as_slice().len(), 1 << (2 * depth));
                for (i, &(x, y, log2)) in positions.as_slice().iter().enumerate() {
                    let (mut tx, mut ty) = (0, 0);
                    for bit in 0..depth {
                        tx |= ((i >> (2 * bit)) & 1) << bit;
                        ty |= ((i >> (2 * bit + 1)) & 1) << bit;
                    }
                    assert_eq!(
                        (x, y, log2),
                        (
                            block.x + (tx << leaf_log2),
                            block.y + (ty << leaf_log2),
                            leaf_log2
                        )
                    );
                }
            }
        }
    }

    #[test]
    fn forced_cu_boundary_splits_keep_every_transform_inside_picture() {
        fn visit(block: Block, width: usize, height: usize, cap: u8, coverage: &mut [u8]) {
            if block.x >= width || block.y >= height {
                return;
            }
            if !block.is_inside(width, height) {
                assert!(block.log2_size() > 3);
                for child in block.quadrants() {
                    visit(child, width, height, cap, coverage);
                }
                return;
            }
            for &(x, y, log2) in transform_units(&block, cap).as_slice() {
                let n = 1usize << log2;
                assert!(x + n <= width && y + n <= height);
                for row in y..y + n {
                    for pixel in &mut coverage[row * width + x..row * width + x + n] {
                        *pixel += 1;
                    }
                }
            }
        }
        for (width, height) in [(8, 8), (64, 64), (72, 72), (136, 80), (120, 104)] {
            for cap in [4, 5] {
                let mut coverage = vec![0; width * height];
                for y in (0..height).step_by(64) {
                    for x in (0..width).step_by(64) {
                        visit(Block::square(x, y, 6), width, height, cap, &mut coverage);
                    }
                }
                assert!(
                    coverage.iter().all(|&count| count == 1),
                    "{width}x{height}, cap{cap}"
                );
            }
        }
    }
}

#[cfg(test)]
mod selective_transform_tests {
    use super::{chroma_quadrants, coarse_energy};
    use nc_common::PlaneView;
    use nc_dsp::Dsp;

    #[test]
    fn chroma_concentration_distinguishes_coherent_and_fine_residuals() {
        let pred = [128u8; 256];
        let reference = PlaneView::new(&pred, 16, 16, 16).unwrap();
        let dsp = Dsp::new();
        for pattern in 0..4 {
            let source: Vec<u8> = (0..256)
                .map(|i| {
                    let (x, y) = (i % 16, i / 16);
                    match pattern {
                        0 => 128,
                        1 => 136,
                        2 => {
                            if (x / 8 + y / 8) % 2 == 0 {
                                120
                            } else {
                                136
                            }
                        }
                        _ => {
                            if (x + y) % 2 == 0 {
                                120
                            } else {
                                136
                            }
                        }
                    }
                })
                .collect();
            let src = PlaneView::new(&source, 16, 16, 16).unwrap();
            let (energy, sse) = coarse_energy(&chroma_quadrants(&dsp, &src, &reference));
            match pattern {
                0 => assert_eq!((energy, sse), (0, 0)),
                1 | 2 => assert_eq!(energy, 64 * sse),
                _ => {
                    assert_eq!(energy, 0);
                    assert!(sse > 0);
                }
            }
        }
    }
}

#[cfg(test)]
mod mixed_transform_tests {
    use super::HevcBackend;
    use nc_common::{Picture, PlaneIndex};
    use nc_encoder::{Block, Codec, CodecBackend, EncoderConfig, FrameRate, ParamOverrides};

    #[test]
    fn selective_transform_tiles_mix_sizes_in_morton_order() {
        let mut params = ParamOverrides::default();
        params.set("hevc.tu16_chroma_concentration", 0.5);
        // The selective TU16 tiling is the one-transform-per-CU path; the RD
        // inter transform tree (Default since the quality-point change)
        // decides splits itself.
        params.set("hevc.inter_tu_split", 0.0);
        let mut config = EncoderConfig {
            codec: Codec::Hevc,
            width: 64,
            height: 64,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 28,
            intra_period: 300,
            block_log2: 6,
            key_qp_offset: Some(0),
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal::default(),
            params,
        };
        let mut backend = HevcBackend::new(&config).unwrap();
        for plane in &mut backend.inter_pred {
            plane.fill(128);
        }
        let mut source = Picture::new(64, 64, 0).unwrap();
        for plane in [PlaneIndex::U, PlaneIndex::V] {
            let mut view = source.plane_mut(plane).view_mut();
            for y in 0..32 {
                for x in 0..32 {
                    view.row_mut(y)[x] = match x / 16 + 2 * (y / 16) {
                        0 => {
                            if (x + y) % 2 == 0 {
                                96
                            } else {
                                160
                            }
                        }
                        1 => 144,
                        2 => 128,
                        _ => {
                            if (x / 8 + y / 8) % 2 == 0 {
                                96
                            } else {
                                160
                            }
                        }
                    };
                }
            }
        }
        let block = Block::square(0, 0, 6);
        let positions = backend.inter_transform_units(&block, &source);
        assert_eq!(
            positions.as_slice(),
            &[
                (0, 0, 4),
                (16, 0, 4),
                (0, 16, 4),
                (16, 16, 4),
                (32, 0, 5),
                (0, 32, 5),
                (32, 32, 5),
            ]
        );
        backend.info.inter_transform_depth = 0;
        assert_eq!(
            backend
                .inter_transform_units(&block, &source)
                .as_slice()
                .len(),
            4
        );
        config.params.set("hevc.max_transform_log2", 4.0);
        let forced16 = HevcBackend::new(&config).unwrap();
        assert_eq!(forced16.info.inter_transform_depth, 0);
        assert_eq!(
            forced16
                .inter_transform_units(&block, &source)
                .as_slice()
                .len(),
            16
        );
    }
}

#[cfg(test)]
mod exact_prediction_cache_tests {
    use super::{HevcBackend, predict_cu};
    use nc_common::{Picture, PlaneIndex};
    use nc_dsp::{Filter, Variance};
    use nc_encoder::{
        Block, Codec, CodecBackend, EncoderConfig, FramePlan, FrameRate, FrameType, Mv,
        ParamOverrides,
    };

    fn backend() -> HevcBackend {
        HevcBackend::new(&EncoderConfig {
            codec: Codec::Hevc,
            width: 128,
            height: 128,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 28,
            intra_period: 300,
            block_log2: 6,
            key_qp_offset: None,
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal::default(),
            params: ParamOverrides::default(),
        })
        .unwrap()
    }

    fn reference() -> Picture {
        let mut picture = Picture::new(128, 128, 0).unwrap();
        for plane in [PlaneIndex::Y, PlaneIndex::U, PlaneIndex::V] {
            let mut view = picture.plane_mut(plane).view_mut();
            for y in 0..view.height() {
                for (x, value) in view.row_mut(y).iter_mut().enumerate() {
                    *value = ((x * 17 + y * 29 + x * y + plane as usize * 43) & 255) as u8;
                }
            }
        }
        picture
    }

    #[test]
    fn matching_cache_preserves_canonical_pixels_and_chroma_at_edges() {
        let mut backend = backend();
        let reference = reference();
        for log2_size in 3..=6 {
            let n = 1 << log2_size;
            for (x, y) in [(0, 0), (128 - n, 128 - n)] {
                let block = Block::square(x, y, log2_size);
                for mv in [
                    Mv::ZERO,
                    Mv { x: -1, y: 3 },
                    Mv { x: 3, y: -7 },
                    Mv { x: 36, y: 2 },
                ] {
                    let mut expected = backend.inter_pred.clone();
                    predict_cu(&backend.dsp, &block, mv, &reference, &mut expected, false);
                    backend.cache_exact_luma(
                        &block,
                        mv,
                        Filter::HevcLuma,
                        &expected[0][..n * n],
                        Variance { sse: 0, sum: 0 },
                    );
                    backend.predict_inter_cu(&block, mv, &reference);
                    for (plane, want) in expected.iter().enumerate() {
                        let area = (n * n) >> (2 * usize::from(plane != 0));
                        assert_eq!(&backend.inter_pred[plane][..area], &want[..area]);
                    }
                    assert!(backend.cached_luma.is_none());
                }
            }
        }
    }

    #[test]
    fn cache_hits_skip_luma_and_misses_restore_canonical_prediction() {
        let mut backend = backend();
        let reference = reference();
        let block = Block::square(16, 16, 4);
        let mv = Mv { x: -1, y: 3 };
        let mut expected = backend.inter_pred.clone();
        predict_cu(&backend.dsp, &block, mv, &reference, &mut expected, false);
        for miss in 0..7 {
            backend.cache_exact_luma(
                &block,
                mv,
                Filter::HevcLuma,
                &[47; 256],
                Variance { sse: 0, sum: 0 },
            );
            let cached = backend.cached_luma.as_mut().unwrap();
            match miss {
                0 => {}
                1 => cached.block.x += 16,
                2 => {
                    cached.block.log2_width += 1;
                    cached.block.log2_height += 1;
                }
                3 => cached.motion.mv.x += 1,
                4 => cached.filter = Filter::Av1,
                5 => cached.motion.ref_idx = 1,
                _ => cached.frame_index += 1,
            }
            backend.predict_inter_cu(&block, mv, &reference);
            assert_eq!(
                &backend.inter_pred[0][..256],
                if miss == 0 {
                    &[47; 256]
                } else {
                    &expected[0][..256]
                }
            );
            for (got, want) in backend.inter_pred[1..].iter().zip(&expected[1..]) {
                assert_eq!(&got[..64], &want[..64]);
            }
            assert!(backend.cached_luma.is_none());
        }
    }

    #[test]
    fn every_new_frame_invalidates_matching_cache_even_when_poc_resets() {
        let mut backend = backend();
        let block = Block::square(0, 0, 4);
        for (index, frame_type) in [(1, FrameType::Inter), (2, FrameType::Key)] {
            backend.cache_exact_luma(
                &block,
                Mv::ZERO,
                Filter::HevcLuma,
                &[47; 256],
                Variance { sse: 0, sum: 0 },
            );
            backend.begin_frame(&FramePlan {
                index,
                frame_type,
                quantizer: 28,
                references: nc_encoder::FrameReferences::FLAT,
                lambda_scale: 1.0,
                block_qp: false,
            });
            assert!(backend.cached_luma.is_none());
            assert_eq!(backend.frame_index, index);
        }
    }
}

#[cfg(test)]
mod early_merge_tests {
    use super::*;
    use nc_encoder::{Codec, FrameRate, ParamOverrides};

    fn backend() -> HevcBackend {
        let mut backend = HevcBackend::new(&EncoderConfig {
            codec: Codec::Hevc,
            width: 64,
            height: 64,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 28,
            intra_period: 300,
            block_log2: 6,
            key_qp_offset: None,
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal::default(),
            params: ParamOverrides::default(),
        })
        .unwrap();
        backend.begin_frame(&FramePlan {
            index: 1,
            frame_type: FrameType::Inter,
            quantizer: 28,
            references: nc_encoder::FrameReferences::FLAT,
            lambda_scale: 1.0,
            block_qp: false,
        });
        backend
    }

    #[test]
    fn early_merge_cached_sse_boundary_and_prediction_keys() {
        let mut backend = backend();
        let block = Block::square(0, 0, 4);
        backend.mv_candidates(&block, &mut Vec::new());
        let stats = nc_dsp::Variance { sse: 256, sum: 256 };
        assert!(!backend.prefer_cached_inter(&block, Mv::ZERO, stats, 4.0));
        backend.early_merge_lambdas = 0.25;
        assert!(backend.prefer_cached_inter(&block, Mv::ZERO, stats, 4.0));
        let hint = backend.early_merge.unwrap();
        let on_first = |mv| Motion::uni(mv, 0);
        assert!(hint.matches(1, &block, on_first(Mv::ZERO)));
        assert!(!hint.matches(2, &block, on_first(Mv::ZERO)));
        assert!(!hint.matches(1, &Block { x: 16, ..block }, on_first(Mv::ZERO)));
        assert!(!hint.matches(1, &block, on_first(Mv { x: 4, y: 0 })));
        assert!(!hint.matches(1, &block, Motion::uni(Mv::ZERO, 1)));
        assert!(!backend.prefer_cached_inter(
            &block,
            Mv::ZERO,
            nc_dsp::Variance { sse: 257, sum: 0 },
            4.0
        ));
        assert!(!backend.prefer_cached_inter(&block, Mv { x: 1, y: 0 }, stats, 4.0));
        backend.begin_frame(&FramePlan {
            index: 2,
            frame_type: FrameType::Key,
            quantizer: 28,
            references: nc_encoder::FrameReferences::FLAT,
            lambda_scale: 1.0,
            block_qp: false,
        });
        assert!(backend.early_merge.is_none());
        assert!(!backend.prefer_cached_inter(&block, Mv::ZERO, stats, 4.0));
    }

    #[test]
    fn chroma_confidence_retains_residuals_and_consumes_hint() {
        let block = Block::square(0, 0, 5);
        for chroma_changed in [false, true] {
            let mut backend = backend();
            backend.early_merge_lambdas = 0.03;
            backend.mv_candidates(&block, &mut Vec::new());
            let mut source = Picture::new(64, 64, 64).unwrap();
            let mut reference = Picture::new(64, 64, 64).unwrap();
            for plane in PlaneIndex::ALL {
                source.plane_mut(plane).view_mut().fill(128);
                reference.plane_mut(plane).view_mut().fill(128);
            }
            reference.extend_borders();
            if chroma_changed {
                source.plane_mut(PlaneIndex::U).view_mut().fill(180);
            }
            let mut recon = Picture::new(64, 64, 64).unwrap();
            assert!(backend.prefer_cached_inter(
                &block,
                Mv::ZERO,
                nc_dsp::Variance { sse: 0, sum: 0 },
                100.0
            ));
            backend.encode_inter_cu(&block, Mv::ZERO, &source, &reference, &mut recon);
            assert!(backend.early_merge.is_none());
            let Some(Item::Cu(cu)) = backend.items.last() else {
                panic!("CU recorded")
            };
            if chroma_changed {
                assert!(cu.tus.units(&backend.tu_store).iter().any(|t| t.cb.coded));
                assert!(!matches!(cu.kind, CuKind::Skip { .. }));
            } else {
                assert!(cu.tus.units(&backend.tu_store).is_empty());
                assert!(matches!(cu.kind, CuKind::Skip { merge_idx: 0 }));
                assert_eq!(recon.plane(PlaneIndex::Y).view().row(0)[0], 128);
            }
        }
    }
}
