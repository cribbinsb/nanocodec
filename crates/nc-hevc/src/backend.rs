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
use nc_dsp::{BlockSize, Dsp, HevcForwardMode, HevcForwardStaging, HevcQuantizer, HevcTransform};
use nc_encoder::chroma_policy::ChromaPolicy;
use nc_encoder::{
    Block, BlockDecision, CodecBackend, ConfigError, EncoderConfig, FramePlan, FrameReferences,
    FrameType, LtrParams, MAX_REFERENCE_SLOTS, Mv, PlaneZeroLimits, PredictionMode, QuantParams,
    RefreshWindow, StaticSkipTest, ZeroLimits, copy_zero_vector_prediction, long_term_slot,
    reference_slots, zero_vector_residual_is_zero,
};

use crate::collocated::MotionStore;
use crate::contexts::{Contexts, SliceKind};
use crate::deblock::{Strength, deblock};
use crate::grid::{Cell, Grid};
use crate::headers::ReferenceSet;
use crate::headers::{
    CraPoc, LOG2_CTB, LOG2_MAX_TB, LOG2_MIN_CB, NalType, POC_LSB_BITS, SIGN_DATA_HIDING,
    SliceDeblocking, SliceQp, SliceReferenceSet, StreamInfo, irap_slice_header, level_idc,
    nal_header, p_slice_header, pps, sps, vps,
};
use crate::inter::{predict_chroma, predict_luma};
use crate::mvpred::{AMVP_CANDIDATES, MAX_MERGE_CANDIDATES, candidate_lists};
use crate::predict::{Refs, Sides, most_probable};
use crate::quant;
use crate::quant::{
    INTER_ROUNDING_512THS, INTRA_ROUNDING_512THS, chroma_qp, hide_signs, largest_zero_coefficient,
    shave_isolated_ones,
};
use crate::rate::BinCosts;
use crate::residual_trellis::{BlockQuantizer, RateModel, TrellisQuantizer};
use crate::sao::{self, PictureSao, SEARCH_INTERVAL};
use crate::scan::{ScanKind, Scans};
use crate::syntax::{
    CuKind, CuRecord, Item, MAX_TUS, MVD_EG_ORDER, PlaneResidual, TransformSyntax, TransformUnit,
    TransformUnits, TuRange, split_ctx_inc, split_flag_present, write_slice_data,
};
use crate::transform::{MAX_COEFFS, forward_bound, inverse_dc};

mod block_qp;
mod inter_tree;
mod intra;
mod key_cost;
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
/// `static_cam22` +27%), 0.5 WORSE (+6%). hevc-fast (k 0.6): TRAIN -1.3%,
/// VAL -1.1%; hevc-better: TRAIN -0.1%, VAL -0.6%. Both VALs sit on the
/// worst-1% VMAF guard (+2.0 / +2.1%): the stronger cadence's coarser leaf
/// windows (`lowlight_pp22_049`, `pan_bdd_00cef86b` +7%), with 8 of 11 clips
/// better on the mean and no flicker or pulse change; adopted. Long GOP
/// (900 f, one key): chirla, crowd and pan within -0.01..+0.21 dB Y per
/// third.
const RATE_MODEL: nc_encoder::RateModel = nc_encoder::RateModel {
    cadence_scale: 1.2,
    first_key_lookahead: 0.5,
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
    /// `merge_idx` with two candidates: one bin.
    pub const MERGE_IDX: f32 = 1.0;
}

/// Predictor lists of the block the engine is currently deciding.
#[derive(Debug, Clone, Copy)]
struct Predictors {
    x: usize,
    y: usize,
    log2: u8,
    merge: [Mv; MAX_MERGE_CANDIDATES],
    amvp: [Mv; AMVP_CANDIDATES],
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
    pre_transform_zero_limit: u64,
    column_zero_limit: u32,
    /// Smallest level kept: 1 normally; past QP 51 the smallest level a
    /// quantiser at the planned step (`headroom_step` times the coded one,
    /// `FramePlan::lambda_scale`) would keep, `ceil((1 − rounding) ·
    /// headroom_step)`.
    planned_min_level: i32,
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
                            planned_min_level: (keep * headroom_step).ceil().max(1.0) as i32,
                            column_zero_limit: (zero_limit as f32 * column_zero_scale) as u32,
                            quantizer: HevcQuantizer::with_rounding(qp as u8, log2, rounding),
                            zero_limit,
                            pre_transform_zero_limit: if intra == 0 {
                                (zero_limit as f64 * f64::from(pre_transform_zero_scale)) as u64
                            } else {
                                zero_limit
                            },
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
    }
}

/// The `hevc.*` settings of non-base frames: the stream's, overlaid with
/// its `nonbase.hevc.*` values (`frame_role.rs`).
fn nonbase_params(config: &EncoderConfig) -> HevcParams {
    let (nonbase, ordinary) = config.params.split_prefix(nc_encoder::NONBASE_PREFIX);
    HevcParams::from_overrides(&ordinary.overlaid_with(&nonbase))
}

/// `hevc.still_skip_limit_scale` while `hevc.still_skip` is on, else 0.
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
    mv: Mv,
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
    /// SAO decision of the current picture (made in `filter_frame`).
    sao: Option<PictureSao>,
    /// Per-CTB SAO classes of the last SAO search frame, for reuse.
    sao_classes: Vec<(Option<u8>, Option<u8>)>,
    grid: Grid,
    /// Each reference slot's motion: the collocated motion for TMVP.
    motion: MotionStore,
    /// The current picture's place in the temporal structure.
    references: FrameReferences,
    /// POC of the picture each reference slot holds (checks the SPS
    /// reference picture sets against the engine's structure).
    slot_poc: [u32; MAX_REFERENCE_SLOTS],
    /// The slot of the long-term reference, when one is kept.
    long_term_slot: usize,
    next_cu: u32,
    next_tu: u32,
    current: Option<Predictors>,
    /// Key frames are coded this many QP finer than P frames.
    rate_control_enabled: bool,
    key_qp_offset: i32,
    /// Forward quantiser rounding offsets (1/512 step), intra and inter,
    /// for luma and for chroma.
    rounding: [(i32, i32); 2],
    /// Multiple of HM's λ (`hevc.lambda_scale`).
    lambda_scale: f32,
    frame_quantization: FrameQuantization,
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
    early_merge_lambdas: f32,
    /// How P-frame intra CUs choose their luma mode
    /// (`hevc.coarse_inter_intra`, `hevc.inter_intra_dc_only`).
    inter_intra: InterIntraSearch,
    /// How key-frame intra CUs choose their luma mode: `Full`, or `Coarse`
    /// (`hevc.key_intra_coarse`).
    key_intra: InterIntraSearch,
    early_merge: Option<EarlyMergeHint>,
    /// The CU (x, y, log2 size) whose intra the engine chose only through
    /// its directional estimate, and the inter vector it beat
    /// (`CodecBackend::intra_fallback`), consumed by its `encode_block`.
    intra_fallback: Option<((usize, usize, u8), Mv)>,
    /// `hevc.still_skip` and its limit scale (0 = off).
    still_skip_limit_scale: f32,
    /// The CU `try_static_skip` accepted: the next `encode_block` codes it
    /// with no residual.
    still_skipped: Option<(usize, usize)>,
    /// Transform-block scratch (allocated once; only the `n * n` prefix a
    /// block uses is touched, so no per-block clearing of 32x32 buffers).
    scratch: RefCell<Scratch>,
    /// Levels of every coded transform block of the picture, appended in
    /// coding order (cleared per picture, so no per-block allocation).
    /// Stored as `i16` (every HEVC level fits): the store is written per
    /// coded block and read back by the slice writer a frame's coding later,
    /// so its size is memory traffic.
    level_store: RefCell<Vec<i16>>,
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
    /// Bin costs of the CABAC rate estimates (`crate::rate`).
    bin_costs: BinCosts,
    /// The optional intra/inter coding tools.
    tools: CodingTools,
    /// `quant.trellis`: trellis quantisation of every coded transform block.
    trellis: Option<TrellisQuant>,
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
    /// Multiple of the mode-decision λ.
    lambda_scale: f32,
    /// Further multiple for chroma; `None` leaves chroma unoptimised.
    chroma_scale: Option<f32>,
    /// Whether key (IDR) slices are trellised too.
    key_frames: bool,
    /// Survivors per scan position.
    max_states: usize,
    search: RefCell<TrellisQuantizer>,
}

impl TrellisQuant {
    fn from_params(quant: &QuantParams) -> Option<Self> {
        (quant.trellis >= 0.5).then(|| Self {
            lambda_scale: quant.trellis_lambda_scale,
            chroma_scale: (quant.trellis_chroma > 0.0).then_some(quant.trellis_chroma),
            key_frames: quant.trellis_key >= 0.5,
            max_states: quant.trellis_states.round() as usize,
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
    DcOnly { max_log2: u8 },
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
    mv: Mv,
    sse_per_sample: f32,
}

impl EarlyMergeHint {
    fn matches(self, frame_index: u64, block: &Block, mv: Mv) -> bool {
        self.frame_index == frame_index && self.block == *block && self.mv == mv
    }
}

/// Working buffers of [`HevcBackend::code_residual`].
#[derive(Debug)]
struct Scratch {
    residual: Box<[i16; MAX_COEFFS]>,
    levels: Box<[i32; MAX_COEFFS]>,
    coeffs: Box<[i32; MAX_COEFFS]>,
    scaled: Box<[i32; MAX_COEFFS]>,
    decoded: Box<[i32; MAX_COEFFS]>,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
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
        sao_luma: 1.0, [0.0, 1.0],
            "luma SAO edge offset on (1) or off (0)",
        sao_chroma: 1.0, [0.0, 1.0],
            "chroma SAO edge offset on (1) or off (0)",
        // Default 4 (as Fast): TRAIN proxy-6 (150f) PSNR-YUV / VMAF v1 BD
        // +0.51% / +0.75% vs 2 for -2.7% instructions (crowd09 / city): J = -1.0.
        sao_stats_row_step: 4.0, [1.0, 4.0],
            "sample every Nth row for SAO statistics; higher trades decision precision for speed",
        // A Fast lever (default off, streams unchanged). Fast P instructions
        // (callgrind, 13 f): lowlight_city_night QP 43 -8.4%, bodycam_bc1
        // QP 47 -2.5%, crowd09 QP 47 -2.8%; bytes at the same QP within 0.04%.
        sao_skip_unchanged: 0.0, [0.0, 1.0],
            "1 switches SAO off without statistics in P-frame CTBs whose CUs are all skipped (they copy already filtered samples): faster",
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
        // (crowd 720p hevc-fast QP 47): staged32_bounded 2.69% -> 1.72% of
        // cycles. Staged 16-point columns saved nothing there (6.46% ->
        // 6.56%), so they have no lever. Default off (byte-identical)
        // until screened.
        fast_forward32_columns: 0.0, [0.0, 1.0],
            "encoder-only staged32-point column approximation on top of staged rows",
        // Skip the exact column pass's 8-column groups the bound proves
        // all quantise to zero, writing zero coefficients (levels unchanged;
        // sign hiding and the trellis see zeros there). Applies with staged
        // rows (`fast_forward16/32`). Lowlight 360p hevc-fast: the upper
        // eight columns of 57-85% of transformed 16x16 blocks qualify.
        // Default on: on vs off was byte-identical in 48/48 encodes (HEVC
        // Default and Fast, 8 clips, QP 32/40/47, 60f); P instructions at
        // QP 42 -1.0% Default / -1.7% Fast (x86), Jetson Fast -0.9..-1.8%.
        forward_zero_columns: 1.0, [0.0, 1.0],
            "encoder-only: zero proven-zero column groups of the staged-row forward instead of computing them",
        // SVT-AV1's partial-frequency forward (`av1.partial_forward`) for
        // inter residuals: only the low quarter of the coefficients is
        // computed and coded; block-QP refinements keep every frequency
        // (static_recovery.rs: hevc-faster's static half stayed 8 dB short
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
    fn intra_fallback(&mut self, block: &Block, mv: Mv) {
        self.intra_fallback = Some(((block.x, block.y, block.log2_size()), mv));
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
        "hevc.intra_leaf_cost",
        "hevc.intra_nxn",
        "hevc.intra_nxn_min_cost",
        "hevc.intra_tu_depth",
        "hevc.intra_tu_split_spread",
        "hevc.intra_tu_split_min_cost",
        "hevc.inter_tu_split",
        "hevc.inter_tu_split_min_cost",
        "hevc.inter_tu_split_spread",
    ];
    /// `begin_frame` derives the picture from its plan; the collocated
    /// motion and the slots' POCs are committed in `commit_frame`.
    const RECODE: bool = true;

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
    const DEFAULT_PARAMS: &'static [(&'static str, f32)] = &[
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
        ("decision.intra_gate_variance", 0.0),
        ("decision.codec_subpel_seed_lambdas", 0.05),
        ("decision.subpel_move_bias", 1.1),
        ("decision.subpel_move_bias_nonzero", 1.1),
        ("decision.subpel_fractional_bias", 1.1),
        ("planner.depth_removal_min_quantizer", 0.86),
        // Exact-rank pruning off: at 1.1, J_real proxy-6 (150f) J-full
        // +0.4% / J-low +0.1% (all 6 clips slightly worse) for -3.8% P
        // instructions; outside the near-free bar the engine default met
        // on AV1. `work/engine-cpu/screens/hevc-prune11.log`.
        ("decision.exact_rank_margin", 0.0),
        // Trellis quantisation of the coded blocks (`quant.*`,
        // docs/handoff/trellis.md), key frames included, one survivor, luma
        // only: J_real staged TRAIN (18 clips) BETTER, quality -2.4% full /
        // -2.0% low (PSNR-YUV -2.1%, NEG -5.1%, VMAF v1 -4.2%, MS-SSIM 0.0%),
        // 3/18 clips worse (bodycam_bc1 +2.7%), worst-1% PSNR-Y +1.9 (guard
        // +2), no guard; periodic-key pulse +0.02 dB. CPU: P frames +15.6%
        // host (crowd09 / bc1 QP36), key frames +52%; Jetson +23.2% (crowd
        // 150 frames, one key); J ~ -0.1 at k 0.1 (2:1 weighted). Single-key
        // long GOPs at equal rate (YUV per third): airport 600 kbps
        // +0.04/+0.07/+0.03 dB, homes390 300 kbps +0.11/+0.26/+0.27,
        // chirla 150 kbps +0.33/-0.16/-0.16. hevc-fast keeps it off.
        ("quant.trellis", 1.0),
        ("quant.trellis_key", 1.0),
        ("quant.trellis_states", 1.0),
        ("quant.trellis_chroma", 0.0),
        // Directional P-frame intra estimate (`decision.intra_directional`,
        // engine `intra_estimate.rs`) with this backend's SATD fallback
        // (`inter_fallback`), estimated only where DC intra costs under 2x
        // inter: J_real staged TRAIN (18 clips, k 0.1) BETTER, quality -0.4%
        // full / -0.4% low (PSNR-YUV -0.6%, NEG -0.0%, VMAF v1 -0.2%), 5/18
        // clips worse (lowlight_city_night +1.1%), no guard, periodic-key
        // pulse -0.04 dB. Fixed QP 60 f (QP 41-51): Riverbed -4.3% PSNR-YUV
        // / -3.4% VMAF, ParkJoy -1.0%, TreesAndGrass -0.3%. CPU: cost-clip P
        // frames +1.8% mid / +1.4% hi, key frames 0. hevc-fast keeps it off
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
        // guard; CPU x86 -2.0%, Jetson cycles -0.3%. hevc-fast, -faster
        // and -better pin the previous values they do not set.
        ("hevc.inter_rounding", 109.0),
        ("hevc.intra_rounding", 209.0),
        ("planner.lambda_scale", 2.581_25),
        ("planner.split_rate_factor", 2.262_5),
        ("hevc.tu16_chroma_concentration", 0.35),
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
        let fps = f64::from(config.frame_rate.num) / f64::from(config.frame_rate.den.max(1));
        let qp = config.quantizer as i32;
        let params = HevcParams::from_overrides(&config.params);
        let info = stream_info(config, &params, fps);
        let (width, height) = (info.width, info.height);
        let rounding = rounding_of(&params);
        let frame_quantization =
            FrameQuantization::without_headroom(qp, info.chroma_qp_offset, rounding, &params);
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
            sao: None,
            sao_classes: Vec::new(),
            grid: Grid::new(width, height),
            motion: MotionStore::new(
                width,
                height,
                if info.long_term {
                    long_term_slot(config.temporal_layers) + 1
                } else {
                    reference_slots(config.temporal_layers)
                },
            ),
            references: FrameReferences::FLAT,
            slot_poc: [0; MAX_REFERENCE_SLOTS],
            long_term_slot: long_term_slot(config.temporal_layers),
            next_cu: 0,
            next_tu: 0,
            current: None,
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
            early_merge_lambdas: params.early_merge_lambdas,
            inter_intra: InterIntraSearch::new(&params),
            key_intra: InterIntraSearch::key(&params),
            early_merge: None,
            intra_fallback: None,
            still_skip_limit_scale: still_skip_limit_scale(&params),
            still_skipped: None,
            scratch: RefCell::default(),
            level_store: RefCell::default(),
            tu_store: Vec::new(),
            deblocking: SliceDeblocking::Fixed,
            deblock_inter: params.deblock_inter >= 0.5,
            random: None,
            bin_costs: BinCosts::new(),
            tools: CodingTools::new(&params),
            trellis: TrellisQuant::from_params(&QuantParams::from_overrides(&config.params)),
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

    /// Slices then carry explicit reference picture sets keeping the
    /// long-term picture (`long_term_set`).
    fn supports_long_term_reference(&self) -> bool {
        true
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
            mv,
            filter,
            frame_index: self.frame_index,
        });
    }

    fn set_frame_params(&mut self, params: &nc_encoder::ParamOverrides) {
        let params = HevcParams::from_overrides(params);
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
        if self.early_merge_lambdas <= 0.0
            || self.slice != SliceKind::Predicted
            || !mv.is_full_pel()
            || self.current.is_none_or(|p| p.merge[0] != mv)
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
            mv,
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
            copy_zero_vector_prediction(block, reference, recon);
            self.still_skipped = Some((block.x, block.y));
        }
        accepted
    }

    fn mv_candidates(&mut self, block: &Block, out: &mut Vec<Mv>) {
        let p = self.predictors(block);
        self.current = Some(p);
        // Merge candidates cost almost nothing to signal; AMVP predictors
        // cost only a zero vector difference.
        out.extend(p.merge.iter().chain(&p.amvp).copied());
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
                let p = self
                    .current
                    .filter(|c| (c.x, c.y, c.log2) == (block.x, block.y, block.log2_size()))
                    .unwrap_or_else(|| self.predictors(block));
                if p.merge.contains(&mv) {
                    // Skip, or merge with residual: `merge_idx` is one bin
                    // with two candidates.
                    cost::SKIP_FLAG + cost::MERGE_IDX
                } else {
                    let mvd = p
                        .amvp
                        .iter()
                        .map(|&pred| mvd_bits(mv, pred))
                        .fold(f32::INFINITY, f32::min);
                    cost::SKIP_FLAG + cost::MODE_AND_PART + 2.0 * cost::FLAG + mvd
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

    fn begin_frame(&mut self, plan: &FramePlan) {
        self.references = plan.references;
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
            }
            FrameType::Inter => {
                self.slice = SliceKind::Predicted;
                self.poc = self.picture_order_count(plan.index);
            }
        }
        self.qp = match plan.frame_type {
            FrameType::Key if !self.rate_control_enabled => {
                (plan.quantizer as i32 - self.key_qp_offset).max(QP_MIN as i32)
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
        self.frame_quantization = settings.at(self.qp);
        if let Some(block_qp) = self.block_qp.as_mut() {
            block_qp.begin_frame(self.qp, settings);
        }
        // Contexts are initialised at the start of every slice (§9.3.2.2).
        self.contexts = Contexts::for_slice(self.slice, self.qp);
        self.items.clear();
        self.level_store.get_mut().clear();
        self.tu_store.clear();
        self.sao = None;
        self.grid.reset();
        self.next_cu = 0;
        self.next_tu = 0;
        self.current = None;
        self.early_merge = None;
        if let Some(random) = self.random.as_mut() {
            self.deblocking = random.slice_deblocking();
        } else if !self.deblock_inter {
            self.deblocking = match self.slice {
                SliceKind::Intra => SliceDeblocking::Inherited,
                SliceKind::Predicted => SliceDeblocking::Override {
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

    fn encode_block(
        &mut self,
        block: &Block,
        decision: &BlockDecision,
        source: &Picture,
        reference: Option<&Picture>,
        recon: &mut Picture,
    ) {
        self.begin_block_qp(decision.qp_offset);
        match decision.mode {
            mode if self.random.is_some() => {
                self.encode_random_cu(block, mode, source, reference, recon);
            }
            PredictionMode::IntraDc => match self.inter_fallback(block, source, reference, recon) {
                Some((mv, reference)) => self.encode_inter_cu(block, mv, source, reference, recon),
                None => self.encode_intra_cu(block, source, recon),
            },
            PredictionMode::Inter(mv) => {
                let reference = reference.expect("inter blocks need a reference picture");
                self.encode_inter_cu(block, mv, source, reference, recon);
            }
        }
        self.end_block_qp();
        self.current = None;
        self.cached_luma = None;
        self.early_merge = None;
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
        // SAO classes are searched on key frames and every SEARCH_INTERVAL-th
        // frame, and reused (offsets re-derived) in between.
        let search =
            self.slice == SliceKind::Intra || u64::from(self.poc).is_multiple_of(SEARCH_INTERVAL);
        let reuse =
            (!search && !self.sao_classes.is_empty()).then_some(self.sao_classes.as_slice());
        let skipped = (self.sao_components.skip_unchanged && self.slice == SliceKind::Predicted)
            .then(|| self.grid.skipped_ctbs());
        let decision = sao::decide(
            source,
            recon,
            f64::from(self.lambda()),
            self.sao_components,
            reuse,
            skipped.as_deref(),
        );
        if search {
            // A skipped CTB's "off" says nothing about its classes: keep the
            // last search's until the CTB is coded again.
            let mut classes = sao::classes(&decision);
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
            self.sao_classes = classes;
        }
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
        );

        let qp = SliceQp {
            delta: self.qp - self.info.init_qp,
            chroma_offset: self.info.chroma_policy.map(|_| self.chroma_offset()),
        };
        let mut header = BitWriter::new();
        let nal_type = match self.slice {
            SliceKind::Intra => {
                let cra = (!self.idr).then(|| CraPoc {
                    lsb: self.poc & ((1 << POC_LSB_BITS) - 1),
                    sps_sets: self.info.reference_sets().len(),
                });
                irap_slice_header(&mut header, cra, qp, self.slice_sao(), self.deblocking);
                if self.idr {
                    NalType::IdrWRadl
                } else {
                    NalType::Cra
                }
            }
            SliceKind::Predicted => {
                let lsb = self.poc & ((1 << POC_LSB_BITS) - 1);
                let sets = self.info.reference_sets();
                let explicit = if self.info.long_term {
                    self.long_term_set()
                } else {
                    Vec::new()
                };
                let references = if self.info.long_term {
                    SliceReferenceSet::Explicit {
                        entries: &explicit,
                        sps_count: sets.len(),
                    }
                } else {
                    SliceReferenceSet::Sps {
                        index: self.reference_set_index(sets),
                        count: sets.len(),
                    }
                };
                p_slice_header(
                    &mut header,
                    lsb,
                    references,
                    qp,
                    self.slice_sao(),
                    self.temporal_mvp(),
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
        }
        nal::write_nal_unit(&nal_header(nal_type, temporal_id), &rbsp, &mut unit);
        unit
    }

    fn commit_frame(&mut self) {
        if self.slice == SliceKind::Intra && self.idr {
            self.sequence_sets = Some(self.sequence_set_bytes());
        }
        // The picture's motion becomes the collocated motion of the slots
        // it refreshed (the grid is reset for the next picture).
        let distance = self.references.reference.distance as i32;
        self.motion
            .save(&mut self.grid, distance, self.references.refresh);
        for (slot, poc) in self.slot_poc.iter_mut().enumerate() {
            if self.references.refresh.contains(slot) {
                *poc = self.poc;
            }
        }
    }
}

impl HevcBackend {
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

    /// The SPS reference picture set of the current picture: its reference
    /// (used), plus slot 0's T0 picture (kept, not used) when predicting from
    /// another slot, since the next T0 predicts from it. Adaptive structures
    /// change the T0 distance per mini-GOP; every combination is in the
    /// stream's set list.
    ///
    /// # Panics
    ///
    /// If the engine's structure produced a set the SPS does not list.
    /// The slice reference set with a long-term reference (§7.4.8): the
    /// picture this one predicts from (used), and, kept for later pictures
    /// unless this one replaces them, slot 0's T0 picture and the long-term
    /// picture. A recovery picture (predicting from the long-term slot)
    /// replaces every short-term slot, so the pictures it follows, which
    /// the receiver may have lost, leave the set.
    fn long_term_set(&self) -> Vec<(u32, bool)> {
        let slot = self.references.reference.slot;
        let mut set = vec![(self.poc - self.slot_poc[slot], true)];
        for kept in [0, self.long_term_slot] {
            let distance = self.poc - self.slot_poc[kept];
            if !self.references.refresh.contains(kept) && set.iter().all(|&(d, _)| d != distance) {
                set.push((distance, false));
            }
        }
        set.sort_unstable();
        set
    }

    fn reference_set_index(&self, sets: &[ReferenceSet]) -> usize {
        let slot = self.references.reference.slot;
        let used = (self.poc - self.slot_poc[slot], true);
        let kept = (slot != 0).then(|| (self.poc - self.slot_poc[0], false));
        let wanted: Vec<(u32, bool)> = std::iter::once(used).chain(kept).collect();
        sets.iter()
            .position(|set| *set == wanted.as_slice())
            .expect("the SPS lists every reference set the structure produces")
    }

    fn predictors(&self, block: &Block) -> Predictors {
        let size = block.size();
        let reference = self.references.reference;
        let collocated = self
            .motion
            .collocated(reference.slot, reference.distance as i32);
        let collocated = self.temporal_mvp().then_some(&collocated);
        let (merge, amvp) = candidate_lists(&self.grid, collocated, block.x, block.y, size);
        Predictors {
            x: block.x,
            y: block.y,
            log2: block.log2_size(),
            merge,
            amvp,
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
        let availability = self.grid.reference_availability(lx, ly, n << scale);
        Refs::gather(
            recon.plane(plane),
            (x, y),
            n,
            unit,
            Sides::PerUnit,
            |xn, yn| availability.at(xn << scale, yn << scale),
        )
    }

    /// Consume only a matching exact luma prediction. The one-reference frame
    /// contract guarantees its reference pixels stay unchanged until this CU.
    fn predict_inter_cu(&mut self, block: &Block, mv: Mv, reference: &Picture) {
        let reuse_luma = self.cached_luma.take().is_some_and(|cached| {
            cached.block == *block
                && cached.mv == mv
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
            .filter(|c| (c.x, c.y, c.log2) == (block.x, block.y, block.log2_size()))
            .unwrap_or_else(|| self.predictors(block));
        if self.still_skipped.take() == Some((block.x, block.y)) {
            // `try_static_skip` wrote the reconstruction (the co-located
            // reference samples); only the syntax remains.
            let tus = self.uncoded_transform_units(block);
            self.record_inter_cu(block, mv, &predictors, &tus);
            return;
        }
        self.predict_inter_cu(block, mv, reference);
        let hint = self
            .early_merge
            .take()
            .filter(|hint| hint.matches(self.frame_index, block, mv));
        if hint.is_some_and(|hint| {
            self.early_merge_chroma_confident(block, source, hint.sse_per_sample)
        }) {
            // Prediction remains normative. Chroma must independently pass:
            // quiet luma does not imply quiet colour residuals. A failed UV
            // check retains normal residual coding for the early-selected MV.
            self.place_prediction(block, recon);
            self.record_cu(
                block,
                Cell {
                    skip: true,
                    mv,
                    tu: u32::MAX,
                    ..Cell::default()
                },
                &[],
            );
            self.push_cu(block, CuKind::Skip { merge_idx: 0 }, &[]);
            return;
        }
        let tus = if self.tools.inter_tu_split {
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
        };
        self.record_inter_cu(block, mv, &predictors, &tus);
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

    /// Choose the syntax of an inter CU with vector `mv` and residual `tus`
    /// (skip, merge or AMVP) and record it.
    fn record_inter_cu(
        &mut self,
        block: &Block,
        mv: Mv,
        predictors: &Predictors,
        tus: &TransformUnits,
    ) {
        let residual = tus.as_slice().iter().any(TransformUnit::coded);
        let merge_idx = predictors.merge.iter().position(|&m| m == mv);

        let kind = match merge_idx {
            Some(merge_idx) if !residual => CuKind::Skip { merge_idx },
            Some(merge_idx) => CuKind::Merge { merge_idx },
            None => {
                let (mvp_idx, pred) = predictors
                    .amvp
                    .iter()
                    .copied()
                    .enumerate()
                    .min_by(|a, b| mvd_bits(mv, a.1).total_cmp(&mvd_bits(mv, b.1)))
                    .expect("two AMVP candidates");
                CuKind::Amvp { mv, pred, mvp_idx }
            }
        };
        let skip = matches!(kind, CuKind::Skip { .. });
        let cell = Cell {
            skip,
            mv,
            ..Cell::default()
        };
        self.record_cu(block, cell, tus.as_slice());
        self.push_cu(block, kind, tus.as_slice());
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
        let size16 = BlockSize::new(16, 16).expect("chroma16");
        let mut views = [None, None];
        let zero_limit = self.frame_quantization.planes[1].transforms[0][2].zero_limit;
        let mut negligible = true;
        for (slot, plane) in [PlaneIndex::U, PlaneIndex::V].into_iter().enumerate() {
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
            let stats = self.dsp.variance(size16, &src, &pred);
            // Cauchy bounds L1 by sqrt(area*SSE), so this avoids quadrant
            // analysis for residuals whose large chroma transform is zero.
            let squared_bound = u64::from(stats.sse) * 256;
            let root = squared_bound.isqrt();
            let abs_bound = root + u64::from(root * root != squared_bound);
            negligible &= forward_bound(4, abs_bound) <= zero_limit;
            views[slot] = Some((src, pred));
        }
        if negligible {
            return false;
        }
        let mut energy = 0;
        let mut squared_error = 0;
        for (src, pred) in views.iter().flatten() {
            let (e, s) = chroma_coarse_energy(&self.dsp, src, pred);
            energy += e;
            squared_error += s;
        }
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
        out: &mut [i32],
    ) -> bool {
        let n = transform.side();
        // Intra residuals and block-QP refinements (the reuse term restoring
        // the detail a degraded reference lost) keep every frequency.
        let partial = !intra && !self.refining_block();
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
        if forward_bound(log2, u64::from(abs_sum)) <= quantization.pre_transform_zero_limit {
            if predicted != Predicted::InterInPlace {
                let mut out = recon.plane_mut(plane).view_mut();
                copy_square(&mut out, (x, y), n, pred, pred_stride);
            }
            return PlaneResidual {
                log2,
                scan,
                levels: 0,
                coded: false,
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
        let mut nonzero = 0;
        let coded = if transformed {
            nonzero = self
                .dsp
                .hevc_quantize_from(&coeffs[..n * n], quantizer, levels);
            if quantization.planned_min_level > 1 {
                nonzero = apply_planned_dead_zone(levels, quantization.planned_min_level);
            }
            // A lone level is the block's first and is never shaved;
            // skipping the shave there (most coded inter blocks: a DC)
            // saves its search for the last level through every empty
            // sub-block. Exact: HEVC Faster x86 instructions -0.6% (callgrind,
            // chart clips, CBR 600).
            let kept = nonzero > 0
                && (self.shave_gap == 0
                    || nonzero == 1
                    || shave_isolated_ones(levels, self.scans.get(log2, scan), self.shave_gap));
            let trellis = (quantisation == Quantisation::Final)
                .then(|| self.trellis_for(plane))
                .flatten();
            if kept && let Some(trellis) = trellis {
                nonzero =
                    self.trellis_quantize(trellis, plane, &coeffs[..n * n], log2, scan, levels);
            }
            nonzero > 0
        } else {
            false
        };
        // Parity adjustment never zeroes or creates a coded block, so an
        // all-zero block (most transformed inter blocks) skips it.
        // A lone DC has no sign-hiding span and, under a DCT, reconstructs
        // to a constant (the DST's first basis is not flat).
        // Exact shortcut: bodycam/crowd P instructions fall 3.79%/0.99%
        // at QP28/39, with unchanged streams and reconstruction.
        let dc_only = nonzero == 1 && levels[0] != 0 && transform != HevcTransform::Dst4;
        if coded && SIGN_DATA_HIDING && !dc_only {
            hide_signs(&coeffs[..n * n], levels, self.scans.get(log2, scan), qp);
        }

        let mut out = recon.plane_mut(plane).view_mut();
        if coded {
            self.dsp
                .hevc_dequantize(levels, quantizer, &mut scaled[..n * n]);
            if dc_only {
                decoded[..n * n].fill(inverse_dc(scaled[0]));
            } else {
                self.dsp
                    .hevc_inverse(transform, &scaled[..n * n], &mut decoded[..n * n]);
            }
            reconstruct_square(&mut out, (x, y), n, (pred, pred_stride), &decoded[..n * n]);
        } else if predicted != Predicted::InterInPlace {
            copy_square(&mut out, (x, y), n, pred, pred_stride);
        }
        let start = if coded { self.store_levels(levels) } else { 0 };
        PlaneResidual {
            log2,
            scan,
            levels: start,
            coded,
        }
    }

    /// Append a coded block's levels to the picture's level store; returns
    /// where they start.
    ///
    /// Levels always fit 16 bits (§7.4.9.11): coefficients are at most
    /// 2^15 in magnitude (`nc_dsp::codec_txfm` ranges), the quantiser scale
    /// at most 2^15 and its shift at least 16 at 8 bits, so `|level| <=
    /// 2^14`, and sign hiding moves a level by one. The narrowing is a
    /// plain truncation so the copy vectorises: a checked conversion per
    /// level (every coefficient of every coded block, mostly zeros) was
    /// ~3% of HEVC Faster's P-frame instructions on lowlight.
    fn store_levels(&self, levels: &[i32]) -> u32 {
        debug_assert!(
            levels.iter().all(|&level| i16::try_from(level).is_ok()),
            "HEVC levels fit 16 bits (§7.4.9.11)"
        );
        let mut store = self.level_store.borrow_mut();
        let start = u32::try_from(store.len()).expect("level store under 4 Gi entries");
        store.extend(levels.iter().map(|&level| level as i16));
        start
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

    /// The trellis settings for a transform block of `plane` in this
    /// slice, with that plane's λ multiple, or `None` while it is off.
    fn trellis_for(&self, plane: PlaneIndex) -> Option<(&TrellisQuant, f32)> {
        let trellis = self
            .trellis
            .as_ref()
            .filter(|t| t.key_frames || self.slice != SliceKind::Intra)?;
        let scale = if plane == PlaneIndex::Y {
            trellis.lambda_scale
        } else if self.info.chroma_policy.is_some() {
            // The policy weighs chroma error by the squared step ratio,
            // 2^((QPy - QPc)/3) (as HM): chroma λ is luma's over it.
            let [luma, chroma] = self
                .frame_quantization
                .planes
                .each_ref()
                .map(|plane| plane.qp);
            trellis.lambda_scale
                * trellis.chroma_scale?
                * 2f32.powf((chroma - luma) as f32 / QP_PER_LAMBDA_DOUBLING)
        } else {
            trellis.lambda_scale * trellis.chroma_scale?
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
        coeffs: &[i32],
        log2: u8,
        scan: ScanKind,
        levels: &mut [i32],
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
            (lambda, trellis.max_states),
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

/// `clamp(prediction + residual)` into the `n`x`n` block at `(x, y)` of
/// `out` (`residual` row-major, `n` per row).
#[inline]
fn reconstruct_square(
    out: &mut PlaneViewMut<'_>,
    (x, y): (usize, usize),
    n: usize,
    (pred, stride): (&[u8], usize),
    residual: &[i32],
) {
    match n {
        4 => reconstruct::<4>(out, x, y, pred, stride, residual),
        8 => reconstruct::<8>(out, x, y, pred, stride, residual),
        16 => reconstruct::<16>(out, x, y, pred, stride, residual),
        32 => reconstruct::<32>(out, x, y, pred, stride, residual),
        _ => unreachable!("transform sides are powers of two from 4 to 32"),
    }
}

#[inline]
fn reconstruct<const N: usize>(
    out: &mut PlaneViewMut<'_>,
    x: usize,
    y: usize,
    pred: &[u8],
    stride: usize,
    residual: &[i32],
) {
    let out_stride = out.stride();
    let dst = block_mut::<N>(out, x, y);
    let pred = &pred[..(N - 1) * stride + N];
    for (row, residual) in residual[..N * N].as_chunks::<N>().0.iter().enumerate() {
        let dst = &mut dst[row * out_stride..row * out_stride + N];
        let pred = &pred[row * stride..row * stride + N];
        for ((d, &p), &r) in dst.iter_mut().zip(pred).zip(residual) {
            *d = (i32::from(p) + r).clamp(0, 255) as u8;
        }
    }
}

/// The samples of `out` spanned by the `N`x`N` block at `(x, y)`, from its
/// first sample to its last, bounds-checked once: the per-row checks of
/// `row_mut` cost about as much as the short copies themselves. With
/// fixed-size reconstruction rows, Fast instructions fell 1.9% per P frame
/// and 2.5% per key frame (exact; callgrind, crowd09 QP 47).
///
/// # Panics
///
/// If the block does not lie inside `out`.
#[inline]
fn block_mut<'a, const N: usize>(
    out: &'a mut PlaneViewMut<'_>,
    x: usize,
    y: usize,
) -> &'a mut [u8] {
    assert!(
        x + N <= out.width() && y + N <= out.height(),
        "block outside the view"
    );
    let stride = out.stride();
    let start = y * stride + x;
    &mut out.data_mut()[start..start + (N - 1) * stride + N]
}

/// Energy captured by four8x8 residual means in one16x16 chroma TU.
fn chroma_coarse_energy(dsp: &Dsp, src: &PlaneView<'_>, pred: &PlaneView<'_>) -> (u64, u64) {
    let mut coarse = 0;
    let mut sse = 0;
    let size = BlockSize::new(8, 8).expect("chroma quadrant");
    for y in [0, 8] {
        for x in [0, 8] {
            let a = src.sub_view(x, y, 8, 8).expect("quadrant");
            let b = pred.sub_view(x, y, 8, 8).expect("quadrant");
            let stats = dsp.variance(size, &a, &b);
            let sum = i64::from(stats.sum);
            coarse += (sum * sum) as u64;
            sse += u64::from(stats.sse);
        }
    }
    (coarse, sse)
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

/// Bits of `mvd_coding()` for `mv − pred` (exact count of its bins).
fn mvd_bits(mv: Mv, pred: Mv) -> f32 {
    let component = |d: i32| -> u32 {
        match d.unsigned_abs() {
            0 => 1,
            1 => 3, // greater0, greater1, sign
            a => {
                let mut value = a - 2;
                let mut k = MVD_EG_ORDER;
                let mut bins = 0;
                while value >= (1 << k) {
                    value -= 1 << k;
                    k += 1;
                    bins += 1;
                }
                3 + bins + 1 + k
            }
        }
    };
    (component(i32::from(mv.x) - i32::from(pred.x))
        + component(i32::from(mv.y) - i32::from(pred.y))) as f32
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

/// The parameter-set contents of a stream for `config` with `params`.
/// QP steps over which λ (a squared step) doubles.
const QP_PER_LAMBDA_DOUBLING: f32 = 3.0;
/// HEVC's chroma QP offset range (`pps_cb_qp_offset` + `slice_cb_qp_offset`, §7.4.7.1).
const CHROMA_OFFSET_LIMIT: i32 = 12;

/// The chroma QP offset (`qPiCb` − `QpY`) whose Table 8-10 chroma QP at
/// luma QP `qp` is nearest the engine chroma policy's
/// (`nc_encoder::chroma_policy`; HEVC QPs are its unit).
fn policy_chroma_offset(policy: ChromaPolicy, qp: i32) -> i32 {
    let target = policy.chroma_q(f64::from(qp));
    let step = |offset: i32| 2f64.powf(f64::from(chroma_qp(qp + offset)) / 6.0);
    nc_encoder::chroma_policy::closest_index(
        -CHROMA_OFFSET_LIMIT..=CHROMA_OFFSET_LIMIT,
        step,
        2f64.powf(target / 6.0),
    )
}

fn stream_info(config: &EncoderConfig, params: &HevcParams, fps: f64) -> StreamInfo {
    let policy = ChromaPolicy::from_params(&config.params);
    // Pictures are coded in whole minimum CUs; a conformance window
    // crops the visible size back out (§7.4.3.2.1).
    let (width, height) = coded_picture_size(config.width, config.height);
    StreamInfo {
        width,
        height,
        crop: (width - config.width, height - config.height),
        level_idc: level_idc(width, height, fps),
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
        signal: config.signal,
        frame_rate: config.frame_rate,
        long_term: LtrParams::from_overrides(&config.params).interval > 0.0,
        block_qp: false,
    }
}

/// Zero the levels below `min_level` in magnitude (the planned step's dead
/// zone past QP 51, `TransformQuantization::planned_min_level`); returns how
/// many remain non-zero.
fn apply_planned_dead_zone(levels: &mut [i32], min_level: i32) -> u32 {
    let mut nonzero = 0;
    for level in levels.iter_mut() {
        if level.abs() < min_level {
            *level = 0;
        }
        nonzero += u32::from(*level != 0);
    }
    nonzero
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
                        for (x, y) in a.transforms[intra].iter().zip(&b.transforms[intra]) {
                            assert_eq!(x.zero_limit, y.zero_limit);
                            assert_eq!(x.column_zero_limit, y.column_zero_limit);
                            assert_eq!(x.quantizer, y.quantizer);
                            assert_eq!(x.pre_transform_zero_limit, x.zero_limit);
                            if intra == 1 {
                                assert_eq!(x.pre_transform_zero_limit, y.pre_transform_zero_limit);
                            } else {
                                assert_eq!(
                                    y.pre_transform_zero_limit,
                                    (x.zero_limit as f64 * f64::from(scale)) as u64
                                );
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
    use super::chroma_coarse_energy;
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
            let (energy, sse) = chroma_coarse_energy(&dsp, &src, &reference);
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
        for miss in 0..6 {
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
                3 => cached.mv.x += 1,
                4 => cached.filter = Filter::Av1,
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
        assert!(hint.matches(1, &block, Mv::ZERO));
        assert!(!hint.matches(2, &block, Mv::ZERO));
        assert!(!hint.matches(1, &Block { x: 16, ..block }, Mv::ZERO));
        assert!(!hint.matches(1, &block, Mv { x: 4, y: 0 }));
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
