//! The H.264 [`CodecBackend`]: IDR pictures of `I_16x16` DC macroblocks and
//! P pictures of `P_L0_16x16`, `P_Skip` and `I_16x16` macroblocks, each with
//! one slice, CABAC, and the deblocking filter on.
//!
//! Per frame one access unit: SPS + PPS + IDR slice for key frames (NAL
//! unit types 7, 8, 5; parameter sets have constant ids and content), a
//! non-IDR P slice (type 1) otherwise, predicted from the previous picture.
//! The engine walks 16x16 "superblocks", which are exactly the macroblocks in
//! raster (= decoding) order.

use core::cell::Cell;

use nc_common::nal::write_nal_unit;
use nc_common::{BitWriter, Picture, PlaneIndex, PlaneView};
use nc_dsp::{BlockSize, Dsp};
use nc_encoder::chroma_policy::ChromaPolicy;
use nc_encoder::{
    Block, BlockDecision, CodecBackend, ConfigError, EncoderConfig, FramePlan, FrameType, Mv,
    PredictionMode, QuantParams, ReferenceChoice, RefreshWindow, StaticSkipTest, holds_frames,
    long_term_slot, reference_slots, uses_long_term_slot,
};

use crate::cabac::{BinCosts, BinSink, CabacEncoder, MbPart, SliceKind};
use crate::cavlc::CavlcEncoder;
use crate::chroma_offsets::{ChromaOffsets, pps_id};
use crate::deblock::{DeblockControl, deblock_picture};
use crate::intra4x4::{Intra4x4Rdo, Intra4x4Result, ModeRd, code_intra_4x4};
use crate::intra8x8::{Intra8x8Result, code_intra_8x8};
use crate::macroblock::{
    CBP_CHROMA_AC, CBP_CHROMA_NONE, ChromaTrellis, Elimination, IntraModes, IntraNxnLevels,
    LumaTrellis, MbQuantizers, Prediction, Residual, ResidualKind, ResidualSkip, TrellisSettings,
    inter_residual_skip, intra_nxn_residual, predict_inter, predict_inter_chroma, predict_intra,
    reconstruct_macroblock, residual_skip_views, residual_skip_within, transform_macroblock,
    trellis_chroma_ac, trellis_luma, write_prediction, write_prediction_views,
};
use crate::mbinfo::{MbInfo, MbPartition, Neighbours};
use crate::mbqp::QpTracker;
use crate::mc::{McKernels, whole_sample_macroblock};
use crate::mvpred::{predict_16x16, skip_mv_from};
use crate::params::{
    Entropy, StreamParams, idr_slice_header, p_slice_header, picture_parameter_set,
    sequence_parameter_set,
};
use crate::partition::{SplitSearch, choose_partition, mvd_bits};
use crate::predict::ChromaMode;
use crate::references::{NAL_REF_IDC_IDR, ReferenceTracker};
use crate::residual::trellis::TrellisQuantizer;
use crate::residual_rd::{choose_quadrants, luma_gains, luma_sse, prunable_quadrants};
use crate::syntax::{CodedMb, ref_idx_ctx, ref_idx_first_inc, write_macroblock};

mod stress;
use stress::RandomSyntax;

/// Macroblock size.
const MB_LOG2: u8 = 4;
const MB_SIZE: usize = 1 << MB_LOG2;
/// The directional intra estimate's sub-block in P macroblocks (`Intra_8x8`).
const INTRA_ESTIMATE_LOG2: u8 = 3;
/// QP range (8-bit video: 0..=51).
const QP_MAX: u32 = 51;

nc_encoder::param_group! {
    /// H.264's tunable heuristics (`nanocodec params`). Measured trade-offs
    /// of the defaults are in `docs/handoff/h264.md`. The defaults are the
    /// TRAIN-selected joint-search set (h264-overnight-20261107-v2 trial 51,
    /// formerly `--preset h264-balanced`): full VAL natural BD-rate -14.5%
    /// PSNR-YUV / -12.6% VMAF-NEG and -11.7% instructions versus the earlier
    /// hand-tuned values; one noise point loses 0.036 dB (disclosed trade-off).
    pub struct H264Params ("h264") {
        /// Key frames are the reference every later frame predicts from
        /// (directly or through skips), so their quality is paid for once and
        /// inherited many times. Rounded to whole QP steps. The tuned 10.97
        /// left a key pulse 7.5 dB above steady state decaying over about 12
        /// frames; 6 (fixed QP, 4 clips x 4 QPs, 240 f single key):
        /// PSNR-YUV / VMAF BD -0.24% / -0.68%, worst-1% PSNR (key excluded)
        /// -1.5%, key pulse 7.5 -> 4.4 dB, frame-to-frame flicker unchanged
        /// (150 f: -0.7% / -1.2%). Rate control sizes keys itself.
        keyframe_qp_offset: 6.0, [0.0, 12.0],
            "key frames this many QP steps finer than inter frames: higher spends more on key frames",
        intra4x4_gate_satd: 24.314_306, [0.0, 64.0],
            "Intra_4x4 tried only above this Intra_16x16 SATD/sample: higher is faster",
        split_gate_satd: 33.347_725, [0.0, 96.0],
            "16x8/8x16/8x8 tried only above this 16x16 SATD/sample: higher is faster",
        /// A fixed SATD gate admits the same macroblocks at every QP, but a
        /// split pays for its extra vector bits more easily at fine
        /// quantisers (small λ). Positive values replace `split_gate_satd`
        /// with this multiple of √λ per sample, so the search widens at high
        /// rates and narrows at low ones.
        split_gate_lambda: 0.0, [0.0, 4.0],
            "if > 0, split search gate is this many sqrt(lambda) of 16x16 SATD/sample (replaces split_gate_satd): lower searches more",
        /// Macroblocks whose quadrant search vectors all lie within this
        /// many quarter samples of the 16x16 vector stay 16x16 without a
        /// split search (0: always search). The search vectors are
        /// whole-sample, so the exit misses the sub-sample motion
        /// differences fine texture has (docs/handoff/h264.md, "Fine
        /// texture vs x264 veryfast").
        split_uniform_qpel: 4.0, [0.0, 8.0],
            "16x16 kept without split search when every quadrant search vector is within this many quarter samples of it (rounded; 0 always searches): higher is faster",
        /// Quarter-sample refinement steps of each quadrant's vector in the
        /// split search (half then quarter sample). Each step costs four 8x8
        /// motion-compensated SATDs per quadrant.
        split_refine_steps: 2.0, [0.0, 2.0],
            "split search refinement steps per quadrant (rounded): 2 half+quarter sample, 1 quarter only, 0 none; lower is faster",
        /// Off by default since candidate C (with `analysis.window_half_width`
        /// 3 and `analysis.mixed_motion` 0, see `DEFAULT_PARAMS`): crowd P
        /// frames 43.89M -> 42.71M instructions; capped TRAIN BD +0.67% PSNR /
        /// +0.49% VMAF-NEG, VAL +0.36% / +0.28%.
        split_refine_mv16: 0.0, [0.0, 1.0],
            "1 also refines quadrants whose best vector is the (already refined) 16x16 one; 0 skips them: faster",
        /// A speed lever (h264-fastest): 0 codes every inter macroblock as
        /// `P_L0_16x16` (or `P_Skip`), with no split search and no 8x8
        /// motion vectors gathered for it.
        partition_search: 1.0, [0.0, 1.0],
            "1 searches 16x8/8x16/8x8 partitions above the split gate; 0 codes every inter macroblock as 16x16: faster",
        /// A speed lever (h264-fastest): intra macroblocks of P slices
        /// predict `Intra_16x16` and chroma from DC only (`Intra_4x4` keeps
        /// its own gate), sparing three luma and six chroma SATDs per trial.
        inter_intra_dc_only: 0.0, [0.0, 1.0],
            "1 tries only the DC Intra_16x16 and chroma modes for intra macroblocks in P slices: faster, coarser intra in P frames",
        subpel_skip_variance: 24.0, [0.0, 64.0],
            "no sub-pel refinement below this residual variance/sample: higher is faster",
        /// A fixed variance gate skips refinement on the same blocks at every
        /// QP, but at fine quantisers (small λ) the residual a sub-sample
        /// vector removes is worth far more bits: on grainy high-rate crowd
        /// content (crowd_pp22_027, QP 25) the fixed gate costs ~16% rate at
        /// equal PSNR. Positive values cap the gate at this multiple of λ
        /// per sample, so refinement widens at high rates only.
        subpel_skip_lambda: 0.0, [0.0, 1.0],
            "if > 0, the sub-pel skip gate is at most this many lambda of residual variance/sample: lower refines more at fine QPs",
        /// Ranks candidate vectors with the exact H.264 6-tap prediction
        /// (the engine's SVT light-PD1 flow, as HEVC and AV1 do) instead of
        /// libaom's bilinear sub-pixel variance, which mis-ranks fractional
        /// vectors on grainy and clean content alike.
        /// With `decision.codec_subpel_seed_lambdas` 0.1 (`DEFAULT_PARAMS`),
        /// 150-frame capped screen of 5 TRAIN clips: −9.7% PSNR-YUV / −6.5%
        /// VMAF v1 BD (lowlight_pp22_004 −21.1/−14.7, crowd_mot17_09
        /// −1.8/−0.3) for +7.3% crowd, +2.8% static, +1.9% bodycam P-frame
        /// instructions at QP37. The bilinear path's `subpel_skip_variance`,
        /// `subpel_skip_lambda` and `decision.skip_no_refine_lambdas` apply
        /// only when this is 0.
        exact_ranking: 1.0, [0.0, 1.0],
            "1 ranks inter candidates on exact 6-tap predictions (engine light-PD1 flow): slower, decides on the coded prediction",
        inter_zero_luma_scale: 1.0, [1.0, 4.0],
            "inter luma SAD zero-bound multiplier: higher skips more residual and partition work",
        // Default -3: staged TRAIN proxy-6 (150f) PSNR-YUV / VMAF v1 BD
        // -2.56% / -2.17% vs 0 (antare V1 +0.92%, others better); host
        // instructions +1.15% (crowd09 QP36): J = -1.68.
        chroma_qp_offset: -3.0, [-12.0, 12.0],
            "PPS chroma_qp_index_offset (both components): negative spends more bits on chroma",
        inter_zero_chroma_scale: 1.0, [1.0, 4.0],
            "inter chroma SAD zero-bound multiplier: higher skips more chroma residual work",
        inter_rounding_divisor: 5.602_792_3, [2.0, 16.0],
            "inter quantisation rounding is 1/N of a step (N rounded to integer): lower preserves more texture at greater rate",
        eliminate_luma_8x8: 2.303_169, [0.0, 10.0],
            "drop inter 8x8 luma blocks costing less (JM scale): higher drops more",
        eliminate_luma_mb: 6.0, [0.0, 16.0],
            "drop all inter luma below this macroblock cost: higher drops more",
        eliminate_chroma_ac: 3.915_538, [0.0, 10.0],
            "drop inter chroma AC costing less: higher drops more",
        /// Static macroblocks (pre-analysis, noise-aware) are offered
        /// `P_Skip` before the engine's mode decision; accepted only when
        /// the skip prediction passes the zero-residual SAD test below, so
        /// a declined block takes the normal decision. Saves the candidate
        /// measurements on still backgrounds.
        static_skip: 1.0, [0.0, 1.0],
            "1 offers static macroblocks P_Skip before mode decision (accepted only if the skip prediction's residual quantises to zero); 0 never: faster with 1",
        static_skip_luma_scale: 1.0, [0.25, 4.0],
            "static P_Skip luma 4x4 SAD bound, as a multiple of the inter zero bound: higher skips more",
        static_skip_chroma_scale: 1.0, [0.25, 4.0],
            "static P_Skip chroma 4x4 and DC SAD bounds, as a multiple of the inter zero bounds: higher skips more",
        /// Rate-distortion pruning of inter luma residuals (`residual_rd`):
        /// sparse luma quadrants are kept only when the squared error they
        /// remove exceeds this multiple of λ times their CABAC bits.
        residual_rd_lambda: 0.0, [0.0, 4.0],
            "sparse inter luma quadrants kept only if their squared-error gain exceeds this many lambda per CABAC bit (0: off): higher drops more",
        /// With `quant.trellis` on, also trellis-quantise `I_16x16`
        /// macroblocks (luma AC; with `quant.trellis_chroma` > 0 their
        /// chroma AC), in key frames only when `quant.trellis_key` is 1.
        trellis_i16x16: 0.0, [0.0, 1.0],
            "1 also trellis-quantises I_16x16 macroblocks' luma AC (and chroma AC with quant.trellis_chroma): slower, fewer bits per intra macroblock",
        /// As `trellis_i16x16` for `I_NxN` macroblocks: each 4x4 block is
        /// trellised before it is reconstructed, so later blocks predict
        /// from the trellised reconstruction.
        trellis_i4x4: 0.0, [0.0, 1.0],
            "1 also trellis-quantises Intra_4x4 luma blocks (and the macroblock's chroma AC with quant.trellis_chroma): slower, fewer bits per intra macroblock",
        /// Coded inter luma uses the 8x8 transform (`inter_residual_kind`);
        /// this also codes it with 4x4 transforms and keeps the one with the
        /// lower squared error plus λ times its CABAC bits.
        transform_size_rd: 0.0, [0.0, 1.0],
            "1 chooses each coded inter macroblock's luma transform size (4x4 or 8x8) by rate-distortion cost: slower, better compression",
        /// Intra macroblocks by squared error plus λ times their CABAC bits
        /// instead of SATD: each `Intra_4x4` block's mode among the
        /// `intra_rd_candidates` best by SATD, and `Intra_4x4` against the
        /// `Intra_16x16` mode SATD chose.
        intra_rd: 0.0, [0.0, 2.0],
            "rate-distortion intra decisions (rounded): 0 off (SATD), 1 in key frames, 2 in every frame; higher is slower, better intra",
        intra_rd_candidates: 9.0, [1.0, 9.0],
            "with intra_rd, the Intra_4x4 modes per block (lowest SATD cost first, rounded) coded and priced: lower is faster",
        /// The shortlist (module comment of `intra_nxn`) evaluates 5-7 of
        /// the nine `Intra4x4PredMode`s per block.
        intra4x4_all_modes: 0.0, [0.0, 1.0],
            "1 evaluates all nine Intra_4x4 modes per block instead of the 5-7 mode shortlist: slower, better intra prediction",
        /// `I_NxN` with `Intra_8x8` prediction and the 8x8 transform
        /// (`transform_size_8x8_flag`, High profile), tried above its own
        /// gate and before `Intra_4x4`, which then stops as soon as it
        /// costs more than the best so far.
        intra8x8: 0.0, [0.0, 2.0],
            "Intra_8x8 macroblocks (8x8 transform; rounded): 0 never, 1 in key frames, 2 in every frame; higher is slower, better intra",
        intra8x8_gate_satd: 24.314_306, [0.0, 64.0],
            "with intra8x8, Intra_8x8 tried only above this Intra_16x16 SATD/sample: higher is faster",
        intra8x8_all_modes: 0.0, [0.0, 1.0],
            "1 evaluates all nine Intra_8x8 modes per block instead of the 5-7 mode shortlist: slower, better intra prediction",
        /// A speed lever (h264-faster): `disable_deblocking_filter_idc` 1 in P
        /// slices skips the boundary-strength pass and every edge filter of
        /// the frame. Key frames stay filtered: every later frame predicts
        /// from them. Legal (§7.4.3), but blockier references at low rates.
        /// Measured vs h264-fast: J_real (proxy-4, 60 f, CBR ladder) +3.4%
        /// (crowd +8.9); Jetson instructions at fixed QP +0.2 / -2.4%
        /// (crowd 39 / bodycam 45: the blockier references code more
        /// residual), CPU time at the 600 kbps rung -3.5% (on top of the
        /// h264-faster search levers). Not in h264-faster: blocking then
        /// accumulates over long GOPs (single key, 300 f, 600 kbps,
        /// static_meva_hospital -0.83 / -1.15 / -1.25 dB per third, against
        /// -0.64 / -0.69 / -0.70 with the filter on).
        deblock_inter: 1.0, [0.0, 1.0],
            "0 switches the deblocking filter off in P slices (key frames stay filtered): faster, blockier at low rates",
        /// A speed lever (h264-fastest): CAVLC slice data
        /// (`entropy_coding_mode_flag` 0, `cavlc.rs`) instead of CABAC. Mode
        /// decision and the rate-distortion tools keep pricing with the
        /// CABAC model (its contexts at their slice-initial state).
        cavlc: 0.0, [0.0, 1.0],
            "1 codes the slice data with CAVLC instead of CABAC: faster entropy coding, more bits",
    }
}

/// NAL unit headers (§7.3.1): `nal_ref_idc` and the unit type.
const NAL_SPS: u8 = 0x67;
const NAL_PPS: u8 = 0x68;
/// `nal_unit_type` of IDR and non-IDR slices (Table 7-1).
const NAL_TYPE_IDR_SLICE: u8 = 5;
const NAL_TYPE_NON_IDR_SLICE: u8 = 1;
/// Position of `nal_ref_idc` in the NAL unit header byte.
const NAL_REF_IDC_SHIFT: u8 = 5;

/// H.264's rate model: the shared QP calibration, plus the start-up
/// lookahead's first-key cap (`rate.first_key_lookahead`) of 0.3 buffers
/// under camera motion. `J_real` TRAIN, 150 f, vs off: -0.7% (pan -2.2%,
/// bodycam -1.1%, lowlight -0.8%, worst-1% Y -2.1%; crowd and static
/// byte-identical, no guard tripped).
const RATE_MODEL: nc_encoder::RateModel = nc_encoder::RateModel {
    first_key_lookahead: 0.3,
    ..nc_encoder::RateModel::QP
};

/// The reference encoder's mode-decision multiplier for squared error,
/// `λ = 0.85 · 2^((QP − 12) / 3)`.
const LAMBDA_SCALE: f32 = 0.85;

/// Rough signalling cost estimates (bits) for mode ranking, from the
/// binarizations of §9.3.2 with typical context probabilities. Provisional.
const SKIP_BITS: f32 = 1.0;
const INTER_16X16_OVERHEAD_BITS: f32 = 4.0; // skip flag, mb_type, a cheap cbp
const INTRA_16X16_BITS: f32 = 9.0; // skip flag, prefix, mb_type, chroma mode

/// The H.264 backend.
#[derive(Debug)]
pub struct H264Backend {
    /// Pixel kernels, chosen for this CPU.
    dsp: Dsp,
    /// The inter prediction kernels, resolved once from `dsp`.
    mc: McKernels,
    params: StreamParams,
    /// The configured QP, used for inter frames.
    rate_control_enabled: bool,
    base_qp: u8,
    /// QP of the current frame (finer for key frames).
    qp: u8,
    /// Squared-error and SATD weights at `mb_qp`, scaled by `lambda_scale`.
    rd_lambda: f32,
    satd_lambda: f32,
    /// The frame's [`FramePlan::lambda_scale`].
    lambda_scale: f32,
    /// Quantisers at `mb_qp`.
    quantizers: MbQuantizers,
    /// The QP `rd_lambda`, `satd_lambda` and `quantizers` belong to: the
    /// frame QP between macroblocks, a macroblock's own QP while it is coded
    /// (`BlockDecision::qp_offset`).
    mb_qp: u8,
    /// Quantisers of every QP used so far at `chroma_offset` (the tuning
    /// never changes).
    quantizer_cache: Vec<Option<MbQuantizers>>,
    /// The engine chroma policy's offsets, one PPS each (`chroma.policy`
    /// 1); `None` keeps the single PPS of `h264.chroma_qp_offset`.
    chroma_offsets: Option<ChromaOffsets>,
    /// The current picture's `chroma_qp_index_offset`.
    chroma_offset: i8,
    /// `QP_Y` prediction of the current slice (`mbqp.rs`).
    qp_tracker: QpTracker,
    /// State of every coded macroblock of the current picture.
    mbs: Vec<Option<MbInfo>>,
    slice: SliceKind,
    /// `frame_num` and reference slot bookkeeping (`references.rs`) as
    /// committed by the previous picture.
    references: ReferenceTracker,
    /// The bookkeeping after the current picture, committed by
    /// `commit_frame` (the recode contract: `begin_frame` may run again).
    next_references: Option<ReferenceTracker>,
    /// `nal_ref_idc` of the current picture.
    nal_ref_idc: u8,
    /// `num_ref_idx_l0_active_minus1 + 1` of the current slice: 2 when the
    /// frame has the engine's second reference (index 1, the long-term
    /// picture; `references.rs`).
    active_references: u8,
    /// `ref_idx_l0` of the reference the engine selected for the
    /// macroblock being decided or coded (`CodecBackend::select_reference`).
    selected_ref: u8,
    /// `idr_pic_id`, alternating so consecutive IDR pictures differ (§7.4.3).
    idr_pic_id: u32,
    /// The slice's CABAC encoder: its writer, and in CAVLC streams only
    /// the rate model decisions price with (never written).
    cabac: Option<CabacEncoder>,
    /// The slice's CAVLC writer in CAVLC streams (`h264.cavlc`).
    cavlc: Option<CavlcEncoder>,
    /// CABAC bin costs by probability state, for pricing macroblocks.
    bin_costs: BinCosts,
    access_unit: Vec<u8>,
    /// The motion search's 8x8 vectors of the macroblock about to be coded
    /// (`sub_block_motion`).
    sub_vectors: Option<[Mv; 4]>,
    /// Predicted vectors of the macroblock being decided; cleared when a
    /// macroblock is coded (its neighbours' state changes).
    predicted: Cell<Option<PredictionCache>>,
    /// The levels of the macroblock being coded, reused across macroblocks
    /// (see [`Residual::reset`]).
    residual: Box<Residual>,
    /// The 4x4-transform alternative of a coded inter macroblock
    /// (`h264.transform_size_rd`), allocated on first use.
    alt_residual: Option<Box<Residual>>,
    /// The macroblock accepted by `try_static_skip` and its skip
    /// prediction, consumed by its `encode_block`.
    static_skip: Option<(usize, usize)>,
    static_pred: Box<Prediction>,
    /// The macroblock (column, row) whose intra the engine chose only
    /// through its directional estimate, and the inter vector it beat
    /// (`CodecBackend::intra_fallback`), consumed by its `encode_block`.
    intra_fallback: Option<((usize, usize), Mv)>,
    /// The prediction buffer of the macroblock being coded, reused: every
    /// mode writes all of it, and a fresh zeroed one per macroblock was a
    /// 384-byte memset (crowd QP 36: 1.2M of 689M instructions).
    mb_pred: Option<Box<Prediction>>,
    /// The exact 16x16 luma prediction of the engine's winning fractional
    /// candidate (`h264.exact_ranking`), taken when that macroblock is coded.
    exact_luma: Cell<Option<ExactLuma>>,
    /// Tunable heuristics.
    tuning: H264Params,
    /// Intra-refresh window of the current frame (`CodecBackend::set_refresh`).
    refresh: Option<RefreshWindow>,
    /// The current slice's `cabac_init_idc` and deblocking control (0 and
    /// the default except in the syntax stress test).
    cabac_init_idc: u8,
    deblock: DeblockControl,
    /// `debug.random_decisions` only (`CodecBackend::enable_random_decisions`):
    /// random syntax choices replace the backend's own.
    random: Option<RandomSyntax>,
    /// `quant.trellis`: trellis quantisation of coded residuals.
    trellis: Option<H264Trellis>,
}

/// The registered trellis controls (`nc_encoder::QuantParams`,
/// `h264.trellis_i16x16`, `h264.trellis_i4x4`) and the search's buffers. Inter macroblocks' luma,
/// optionally their chroma AC and intra macroblocks' AC; the DC blocks and
/// `Intra_4x4` luma (quantised while predicting) keep the plain quantiser.
#[derive(Debug)]
struct H264Trellis {
    /// Multiple of the mode-decision λ.
    lambda_scale: f32,
    /// Further multiple of λ for chroma AC (`None`: chroma untouched),
    /// before the squared chroma/luma step ratio.
    chroma_scale: Option<f32>,
    /// `I_16x16` and `I_NxN` macroblocks too (`h264.trellis_i16x16`,
    /// `h264.trellis_i4x4`).
    intra: [bool; 2],
    /// Key frames too (`quant.trellis_key`; only intra macroblocks there).
    key_frames: bool,
    /// Survivors per scan position.
    max_states: usize,
    search: crate::residual::trellis::TrellisQuantizer,
}

impl H264Trellis {
    fn from_params(quant: &QuantParams, tuning: &H264Params) -> Option<Self> {
        (quant.trellis >= 0.5).then(|| Self {
            lambda_scale: quant.trellis_lambda_scale,
            chroma_scale: (quant.trellis_chroma > 0.0).then_some(quant.trellis_chroma),
            intra: [tuning.trellis_i16x16 >= 0.5, tuning.trellis_i4x4 >= 0.5],
            key_frames: quant.trellis_key >= 0.5,
            max_states: quant.trellis_states.round() as usize,
            search: crate::residual::trellis::TrellisQuantizer::default(),
        })
    }

    /// Whether macroblocks coded as `kind` in a `slice` are trellised.
    fn covers(&self, kind: ResidualKind, slice: SliceKind) -> bool {
        let kind_covered = match kind {
            ResidualKind::Inter | ResidualKind::Inter8x8 => true,
            ResidualKind::Intra16x16 => self.intra[0],
            ResidualKind::Intra4x4 | ResidualKind::Intra8x8 => self.intra[1],
        };
        kind_covered && (slice != SliceKind::I || self.key_frames)
    }
}

/// The cached predicted vectors of macroblock `index` (`(col, row)`). The
/// 16x16 prediction is only computed when needed: a skip vector that is
/// zero by the neighbour rule (§8.4.1.1, nearly every static macroblock)
/// does not need it, and a static skip then never asks for it.
#[derive(Debug, Clone, Copy)]
struct PredictionCache {
    index: (usize, usize),
    /// The reference index `mvp` predicts for.
    ref_idx: u8,
    skip: Mv,
    mvp: Option<Mv>,
}

/// An exact 6-tap luma prediction the engine already computed while ranking:
/// the macroblock at `origin` predicted at `mv` from reference `ref_idx`.
#[derive(Debug, Clone, Copy)]
struct ExactLuma {
    origin: (usize, usize),
    mv: Mv,
    ref_idx: u8,
    pixels: [u8; MB_SIZE * MB_SIZE],
}

/// The vectors a macroblock can signal most cheaply.
#[derive(Debug, Clone, Copy)]
struct PredictedVectors {
    /// The `P_Skip` vector; `None` for a reference index other than 0,
    /// which `P_Skip` cannot use (§8.4.1.1).
    skip: Option<Mv>,
    /// The 16x16 predicted vector for the selected reference index.
    mvp: Mv,
}

/// How a macroblock is coded once its intra choice is made: the intra
/// choice, or inter at a vector (the engine's, or the fallback its intra
/// choice lost to).
enum MbPlan {
    Intra(MbChoice),
    Inter(Mv),
}

/// How a macroblock is going to be predicted.
// One short-lived value per macroblock on the stack; boxing the large
// Intra_4x4 variant would only add a heap allocation per intra macroblock.
#[allow(clippy::large_enum_variant)]
enum MbChoice {
    Intra16x16(IntraModes),
    Intra4x4 {
        modes: IntraModes,
        /// Boxed: its levels are 1 KB, and an unboxed variant made every
        /// `MbChoice` (inter ones included) that size, copied on each return.
        i4: Box<Intra4x4Result>,
    },
    Intra8x8 {
        modes: IntraModes,
        /// Boxed as `Intra4x4`'s.
        i8: Box<Intra8x8Result>,
    },
    Inter {
        partition: MbPartition,
        mvs: [Mv; 4],
    },
}

/// The luma of an `I_NxN` macroblock as its trial coded it.
#[derive(Clone, Copy)]
enum NxnLuma<'a> {
    Blocks4x4(&'a Intra4x4Result),
    Blocks8x8(&'a Intra8x8Result),
}

impl<'a> NxnLuma<'a> {
    fn levels(self) -> IntraNxnLevels<'a> {
        match self {
            Self::Blocks4x4(i4) => IntraNxnLevels::Blocks4x4(&i4.levels),
            Self::Blocks8x8(i8) => IntraNxnLevels::Blocks8x8(&i8.levels),
        }
    }

    /// The macroblock to write, with chroma mode `chroma` and `residual`.
    fn coded(self, chroma: ChromaMode, residual: &Residual) -> CodedMb<'_> {
        match self {
            Self::Blocks4x4(i4) => CodedMb::Intra4x4 {
                luma_modes: i4.modes,
                chroma,
                residual,
            },
            Self::Blocks8x8(i8) => CodedMb::Intra8x8 {
                luma_modes: i8.modes,
                chroma,
                residual,
            },
        }
    }
}

/// The luma samples of the macroblock at `(x, y)` of `picture`.
fn copy_luma(picture: &Picture, (x, y): (usize, usize)) -> [u8; MB_SIZE * MB_SIZE] {
    let region = picture
        .plane(nc_common::PlaneIndex::Y)
        .region(x as isize, y as isize, MB_SIZE, MB_SIZE)
        .expect("macroblock inside the picture");
    let mut out = [0; MB_SIZE * MB_SIZE];
    for (r, row) in out.as_chunks_mut::<MB_SIZE>().0.iter_mut().enumerate() {
        row.copy_from_slice(region.row(r));
    }
    out
}

/// Write `luma` (from [`copy_luma`]) back over the macroblock at `(x, y)`.
fn restore_luma(picture: &mut Picture, (x, y): (usize, usize), luma: &[u8; MB_SIZE * MB_SIZE]) {
    let mut region = picture
        .plane_mut(nc_common::PlaneIndex::Y)
        .region_mut(x as isize, y as isize, MB_SIZE, MB_SIZE)
        .expect("macroblock inside the picture");
    for (r, row) in luma.as_chunks::<MB_SIZE>().0.iter().enumerate() {
        region.row_mut(r).copy_from_slice(row);
    }
}

/// `ref_idx_l0` of the engine's reference choice (`references.rs`: the
/// secondary reference is list index 1).
fn ref_idx_of(choice: ReferenceChoice) -> u8 {
    match choice {
        ReferenceChoice::Primary => 0,
        ReferenceChoice::Secondary => 1,
        ReferenceChoice::Compound { .. } => unreachable!("H.264 uses single-reference P slices"),
    }
}

impl H264Backend {
    fn mb_index(block: &Block) -> (usize, usize) {
        (block.x / MB_SIZE, block.y / MB_SIZE)
    }

    /// The intra prediction of macroblock `(col, row)`: the best
    /// `Intra_16x16` and chroma modes, and `Intra_8x8` or `Intra_4x4` if
    /// they beat them. Trying `I_NxN` reconstructs its luma into `recon` as
    /// it goes; the winner's reconstruction is left there (`Intra_16x16`'s
    /// overwrites it when the macroblock is coded).
    fn choose_intra(
        &mut self,
        source: &Picture,
        recon: &mut Picture,
        (col, row): (usize, usize),
        pred: &mut Prediction,
    ) -> (MbChoice, f32) {
        let (x, y) = (col * MB_SIZE, row * MB_SIZE);
        let dc_only = self.slice == SliceKind::P && self.tuning.inter_intra_dc_only >= 0.5;
        let modes = predict_intra(&self.dsp, source, recon, (x, y), pred, dc_only);
        // Lambda is in the squared-error domain, SATD is not.
        // I_NxN is only searched where Intra_16x16 predicts poorly (a gate
        // per block size): a
        // smooth macroblock gains nothing from per-block directions. Quick
        // clips, 10 s IP: gate 8 costs +0.06% BD-rate for -4M instructions
        // per 720p frame (gate 4: +0.09%, -2M; gate 16: +0.54%, -6M).
        // I_NxN may read the macroblock above-right (§8.3.1.2, §8.3.2.2),
        // which is dirty on intra refresh's dirty edge; Intra_16x16 never
        // does.
        let block = Block::square(x, y, MB_LOG2);
        let refresh_edge = self
            .refresh
            .as_ref()
            .is_some_and(|w| w.on_dirty_edge(&block));
        let above = |gate: f32| modes.luma_satd as f32 >= gate * 256.0;
        let try_4x4 = above(self.tuning.intra4x4_gate_satd);
        let try_8x8 = self.intra_8x8_covers(self.slice) && above(self.tuning.intra8x8_gate_satd);
        if refresh_edge || !(try_4x4 || try_8x8) {
            return (MbChoice::Intra16x16(modes), modes.luma_satd as f32);
        }
        let rd = self.intra_rd_covers(self.slice);
        // The best choice so far and its cost: rate-distortion cost with
        // `rd`, SATD plus λ times mode bits otherwise. By SATD an I_NxN trial
        // stops once its cost reaches the best (it can only grow), which
        // changes no decision.
        let mut best_cost = if rd {
            self.intra_16x16_rd_cost(&block, source, recon, modes, pred)
        } else {
            modes.luma_satd as f32
        };
        let abort_above = |best: f32| if rd { f32::INFINITY } else { best };
        let mut best = MbChoice::Intra16x16(modes);
        let mut luma_8x8 = None;
        if try_8x8
            && let Some(i8) =
                self.code_intra_8x8_mb(source, recon, (col, row), abort_above(best_cost))
        {
            let cost = if rd {
                let luma = NxnLuma::Blocks8x8(&i8);
                self.intra_nxn_rd_cost(&block, source, recon, modes, luma, pred)
            } else {
                i8.cost
            };
            if cost < best_cost {
                best_cost = cost;
                luma_8x8 = try_4x4.then(|| copy_luma(recon, (x, y)));
                best = MbChoice::Intra8x8 {
                    modes,
                    i8: Box::new(i8),
                };
            }
        }
        let i4 = try_4x4
            .then(|| self.code_intra_4x4_mb(source, recon, (col, row), rd, abort_above(best_cost)))
            .flatten();
        if let Some(i4) = i4 {
            let cost = if rd {
                let luma = NxnLuma::Blocks4x4(&i4);
                self.intra_nxn_rd_cost(&block, source, recon, modes, luma, pred)
            } else {
                i4.cost
            };
            if cost < best_cost {
                let i4 = MbChoice::Intra4x4 {
                    modes,
                    i4: Box::new(i4),
                };
                return (i4, cost);
            }
        }
        if let Some(luma) = luma_8x8 {
            // The Intra_4x4 trial overwrote the Intra_8x8 reconstruction.
            restore_luma(recon, (x, y), &luma);
        }
        (best, best_cost)
    }

    /// How macroblock `block` is predicted (into `pred`) for the engine's
    /// `decision`, and which residual parts may be skipped; pictures and
    /// `sub_vectors` as in [`Self::code_macroblock`].
    fn choose_macroblock(
        &mut self,
        block: &Block,
        decision: &BlockDecision,
        (source, reference, recon): (&Picture, Option<&Picture>, &mut Picture),
        sub_vectors: Option<[Mv; 4]>,
        pred: &mut Prediction,
    ) -> (MbChoice, ResidualSkip) {
        if self.random.is_some() {
            return self.random_choice(
                block,
                decision.mode,
                source,
                reference,
                recon,
                sub_vectors,
                pred,
            );
        }
        let plan = match decision.mode {
            PredictionMode::IntraDc => {
                self.intra_or_fallback(block, (source, reference, recon), pred)
            }
            PredictionMode::Inter(mv) => MbPlan::Inter(mv),
        };
        match plan {
            MbPlan::Intra(choice) => (choice, ResidualSkip::NONE),
            MbPlan::Inter(mv) => {
                let reference = reference.expect("inter blocks need a reference picture");
                let predicted = self.predicted_vectors(block); // still cached
                self.choose_inter(
                    source,
                    reference,
                    (block.x, block.y),
                    mv,
                    sub_vectors,
                    predicted,
                    pred,
                )
            }
        }
    }

    /// The intra choice of macroblock `block` (pictures as in
    /// [`Self::code_macroblock`]), or its inter fallback
    /// ([`CodecBackend::intra_fallback`]) where that costs less. The
    /// engine's directional estimate chose intra from source samples; on
    /// the reconstruction intra may cost more than the inter vector it beat
    /// (x264 compares both by SATD). Under `intra_rd` the intra cost is not
    /// in SATD units, so intra stands.
    fn intra_or_fallback(
        &mut self,
        block: &Block,
        (source, reference, recon): (&Picture, Option<&Picture>, &mut Picture),
        pred: &mut Prediction,
    ) -> MbPlan {
        let position = Self::mb_index(block);
        let fallback = self
            .intra_fallback
            .take()
            .and_then(|(at, mv)| (at == position).then_some(mv));
        let (choice, intra_cost) = self.choose_intra(source, recon, position, pred);
        match fallback.zip(reference) {
            Some((mv, reference))
                if !self.intra_rd_covers(self.slice)
                    && self.inter_satd_cost(block, source, reference, mv) < intra_cost =>
            {
                MbPlan::Inter(mv)
            }
            _ => MbPlan::Intra(choice),
        }
    }

    /// Luma SATD plus λ times mode bits of predicting `block` from
    /// `reference` at `mv`: the scale of [`Self::choose_intra`]'s costs.
    // Out of line: it runs only for estimate-won intra macroblocks.
    #[inline(never)]
    fn inter_satd_cost(&self, block: &Block, source: &Picture, reference: &Picture, mv: Mv) -> f32 {
        // Through the macroblock prediction (chroma included, unused): a
        // second caller of the 16x16 luma kernel moved it out of line in
        // the inter path (+0.5% instructions on crowd).
        let mut prediction = Prediction::ZERO;
        predict_inter(
            &self.mc,
            reference,
            block.x,
            block.y,
            &[mv; 4],
            &mut prediction,
        );
        let src = source
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, MB_SIZE, MB_SIZE)
            .expect("macroblock inside the picture");
        let pred = PlaneView::new(&prediction.luma, MB_SIZE, MB_SIZE, MB_SIZE)
            .expect("buffer covers the block");
        let size = BlockSize::new(MB_SIZE, MB_SIZE).expect("16x16 is a kernel size");
        self.dsp.satd(size, &src, &pred) as f32
            + self.satd_lambda * self.mode_bits(block, &PredictionMode::Inter(mv))
    }

    /// Whether `Intra_8x8` is tried in a `slice` (`h264.intra8x8`: 1 in key
    /// frames, 2 in every frame).
    fn intra_8x8_covers(&self, slice: SliceKind) -> bool {
        match self.tuning.intra8x8.round() as u8 {
            0 => false,
            1 => slice == SliceKind::I,
            _ => true,
        }
    }

    /// Whether intra macroblocks of a `slice` are decided by rate-distortion
    /// cost (`h264.intra_rd`: 1 in key frames, 2 in every frame).
    fn intra_rd_covers(&self, slice: SliceKind) -> bool {
        match self.tuning.intra_rd.round() as u8 {
            0 => false,
            1 => slice == SliceKind::I,
            _ => true,
        }
    }

    /// Code the luma of macroblock `(col, row)` as `Intra_4x4` into `recon`,
    /// each block trellised if the trellis covers `Intra_4x4` and its mode
    /// chosen by rate-distortion cost if `rd`.
    // Out of line, so the block loop and its predictors are optimised as
    // one unit: inlined into `choose_intra`, the Intra_4x4 predictor stayed
    // a call and key frames cost 5% more instructions.
    #[inline(never)]
    fn code_intra_4x4_mb(
        &mut self,
        source: &Picture,
        recon: &mut Picture,
        (col, row): (usize, usize),
        rd: bool,
        abort_above: f32,
    ) -> Option<Intra4x4Result> {
        let (x, y) = (col * MB_SIZE, row * MB_SIZE);
        let nb = Neighbours::of(&self.mbs, self.params.mb_cols, col, row);
        let top_right_mb = row > 0 && col + 1 < self.params.mb_cols;
        let covered = self.lambda_scale <= 1.0
            && self
                .trellis
                .as_ref()
                .is_some_and(|t| t.covers(ResidualKind::Intra4x4, self.slice));
        let cabac = self.cabac.as_ref().filter(|_| covered || rd);
        let bins = cabac.map(|c| c.bit_counter(&self.bin_costs));
        let luma = bins.as_ref().map(|bins| {
            let coded = |m: Option<&MbInfo>, bx: usize, by: usize| {
                cond_term_flag(m, true, |m| m.luma_block_coded(bx, by))
            };
            let (lambda_scale, max_states) = self
                .trellis
                .as_ref()
                .map_or((1.0, 1), |t| (t.lambda_scale, t.max_states));
            LumaTrellis {
                settings: TrellisSettings {
                    bins,
                    lambda: self.rd_lambda * lambda_scale,
                    max_states,
                },
                left_coded: std::array::from_fn(|by| coded(nb.left, 3, by)),
                above_coded: std::array::from_fn(|bx| coded(nb.above, bx, 3)),
            }
        });
        let mut own_search = None;
        let search = match self.trellis.as_mut() {
            Some(t) => &mut t.search,
            None => own_search.get_or_insert_with(TrellisQuantizer::default),
        };
        let rdo = luma.as_ref().map(|luma| Intra4x4Rdo {
            luma,
            search,
            trellis: covered,
            mode_rd: rd.then(|| ModeRd {
                lambda: self.rd_lambda,
                satd_lambda: self.satd_lambda,
                candidates: self.tuning.intra_rd_candidates.round() as usize,
            }),
        });
        code_intra_4x4(
            &self.dsp,
            source,
            recon,
            (x, y),
            top_right_mb,
            self.quantizers.luma_intra(),
            (self.satd_lambda, self.tuning.intra4x4_all_modes >= 0.5),
            |bx, by, own| nb.predicted_intra_4x4_mode(bx, by, own),
            rdo,
            abort_above,
        )
    }

    /// Code the luma of macroblock `(col, row)` as `Intra_8x8` into `recon`
    /// (`None` once its cost reaches `abort_above`, see [`code_intra_8x8`]).
    // Out of line, so the block loop and its predictors are optimised as
    // one unit: inlined into `choose_intra`, the Intra_4x4 predictor stayed
    // a call and key frames cost 5% more instructions.
    #[inline(never)]
    fn code_intra_8x8_mb(
        &self,
        source: &Picture,
        recon: &mut Picture,
        (col, row): (usize, usize),
        abort_above: f32,
    ) -> Option<Intra8x8Result> {
        let nb = Neighbours::of(&self.mbs, self.params.mb_cols, col, row);
        code_intra_8x8(
            &self.dsp,
            source,
            recon,
            (col * MB_SIZE, row * MB_SIZE),
            row > 0 && col + 1 < self.params.mb_cols,
            self.quantizers.luma_intra(),
            (self.satd_lambda, self.tuning.intra8x8_all_modes >= 0.5),
            |bx, by, own| nb.predicted_intra_4x4_mode(bx, by, own),
            abort_above,
        )
    }

    /// Luma squared error plus λ times the CABAC bits of the macroblock
    /// `block` coded as `Intra_16x16` in `modes` (prediction in `pred`),
    /// reconstructed into `recon`. Uses `self.residual`.
    fn intra_16x16_rd_cost(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
        modes: IntraModes,
        pred: &Prediction,
    ) -> f32 {
        let choice = MbChoice::Intra16x16(modes);
        self.transform_residual(
            &choice,
            ResidualSkip::NONE,
            source,
            pred,
            (block.x, block.y),
        );
        self.trellis_residual(block, source, pred);
        let residual = &*self.residual;
        reconstruct_macroblock(
            &self.dsp,
            recon,
            pred,
            block.x,
            block.y,
            residual,
            &self.quantizers,
        );
        let distortion = luma_sse(&self.dsp, source, recon, (block.x, block.y));
        distortion + self.rd_lambda * self.mb_bits(block, &CodedMb::Intra16x16 { modes, residual })
    }

    /// [`Self::intra_16x16_rd_cost`] of the macroblock coded as `I_NxN` with
    /// luma `luma` (already reconstructed into `recon`) and chroma in
    /// `modes`.
    fn intra_nxn_rd_cost(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &Picture,
        modes: IntraModes,
        luma: NxnLuma<'_>,
        pred: &Prediction,
    ) -> f32 {
        intra_nxn_residual(
            &self.dsp,
            source,
            pred,
            block.x,
            block.y,
            &self.quantizers,
            luma.levels(),
            &mut self.residual,
        );
        self.trellis_residual(block, source, pred);
        let coded = luma.coded(modes.chroma, &self.residual);
        luma_sse(&self.dsp, source, recon, (block.x, block.y))
            + self.rd_lambda * self.mb_bits(block, &coded)
    }

    /// CABAC bits of macroblock `block` coded as `coded`, priced with the
    /// slice's contexts as they stand.
    fn mb_bits(&self, block: &Block, coded: &CodedMb<'_>) -> f32 {
        let (col, row) = Self::mb_index(block);
        let nb = Neighbours::of(&self.mbs, self.params.mb_cols, col, row);
        let cabac = self.cabac.as_ref().expect("encode_block outside a frame");
        let mut counter = cabac.bit_counter(&self.bin_costs);
        let mut qp = self.qp_tracker;
        write_macroblock(
            &mut counter,
            &self.dsp,
            self.slice,
            coded,
            &nb,
            &mut qp,
            self.mb_qp,
        );
        counter.bits()
    }

    /// The motion partitioning and prediction of the inter macroblock at
    /// `(x, y)` whose engine-chosen vector is `mv`, and which parts of its
    /// residual the configured SAD thresholds permit skipping.
    /// `sub_vectors` are the motion search's 8x8 vectors of the macroblock.
    fn choose_inter(
        &self,
        source: &Picture,
        reference: &Picture,
        (x, y): (usize, usize),
        mv: Mv,
        sub_vectors: Option<[Mv; 4]>,
        predicted: PredictedVectors,
        pred: &mut Prediction,
    ) -> (MbChoice, ResidualSkip) {
        let unsplit = MbChoice::Inter {
            partition: MbPartition::P16x16,
            mvs: [mv; 4],
        };
        // A cached prediction belongs to one macroblock of this frame (the
        // cache is cleared per frame), so position and vector identify it.
        let exact = self.exact_luma.take().filter(|cached| {
            cached.origin == (x, y) && cached.mv == mv && cached.ref_idx == self.selected_ref
        });
        if let Some(exact) = exact {
            pred.luma = exact.pixels;
            predict_inter_chroma(&self.mc, reference, x, y, mv, pred);
        } else {
            predict_inter(&self.mc, reference, x, y, &[mv; 4], pred);
        }
        let zero16 = inter_residual_skip(&self.dsp, source, pred, x, y, &self.quantizers);
        // P_Skip as early as possible (the reference encoder's order): the
        // skip vector passing all residual thresholds needs no partition search.
        // With threshold scales above 1 this is a deliberate speed/quality tradeoff.
        if zero16.all() && Some(mv) == predicted.skip {
            return (unsplit, zero16);
        }
        if self.tuning.partition_search < 0.5 {
            return (unsplit, zero16);
        }
        let sub_vectors = sub_vectors.unwrap_or([mv; 4]);
        let (partition, mvs) = choose_partition(
            &self.dsp,
            &self.mc,
            source,
            reference,
            (x, y),
            (mv, &pred.luma),
            &sub_vectors,
            predicted.mvp,
            self.satd_lambda,
            SplitSearch {
                uniform_qpel: match self.tuning.split_uniform_qpel.round() as i16 {
                    0 => None,
                    limit => Some(limit),
                },
                gate_satd: if self.tuning.split_gate_lambda > 0.0 {
                    self.tuning.split_gate_lambda * self.satd_lambda
                } else {
                    self.tuning.split_gate_satd
                },
                refine_steps: self.tuning.split_refine_steps.round() as usize,
                refine_mv16: self.tuning.split_refine_mv16 >= 0.5,
            },
        );
        if partition == MbPartition::P16x16 || !self.refresh_allows_quadrants(x, &mvs) {
            return (unsplit, zero16);
        }
        predict_inter(&self.mc, reference, x, y, &mvs, pred);
        let zero = inter_residual_skip(&self.dsp, source, pred, x, y, &self.quantizers);
        (MbChoice::Inter { partition, mvs }, zero)
    }

    /// Whether intra refresh allows each 8x8 quadrant's vector of the
    /// macroblock at column `x` (partition refinement may move a vector the
    /// engine checked; the unsplit vector always passed).
    fn refresh_allows_quadrants(&self, x: usize, mvs: &[Mv; 4]) -> bool {
        const QUADRANT: usize = MB_SIZE / 2;
        self.refresh.as_ref().is_none_or(|w| {
            mvs.iter()
                .enumerate()
                .all(|(i, &mv)| w.allows_area(x + (i % 2) * QUADRANT, QUADRANT, mv))
        })
    }

    /// Transforms and quantises the residual of the chosen macroblock mode
    /// into `self.residual` (`Intra_4x4` luma against the reconstruction
    /// built block by block in `choose_intra`).
    fn transform_residual(
        &mut self,
        choice: &MbChoice,
        zero: ResidualSkip,
        source: &Picture,
        pred: &Prediction,
        (x, y): (usize, usize),
    ) {
        let residual = &mut *self.residual;
        match choice {
            MbChoice::Intra4x4 { i4, .. } => intra_nxn_residual(
                &self.dsp,
                source,
                pred,
                x,
                y,
                &self.quantizers,
                IntraNxnLevels::Blocks4x4(&i4.levels),
                residual,
            ),
            MbChoice::Intra8x8 { i8, .. } => intra_nxn_residual(
                &self.dsp,
                source,
                pred,
                x,
                y,
                &self.quantizers,
                IntraNxnLevels::Blocks8x8(&i8.levels),
                residual,
            ),
            MbChoice::Intra16x16(_) => transform_macroblock(
                &self.dsp,
                source,
                pred,
                x,
                y,
                ResidualKind::Intra16x16,
                &self.quantizers,
                ResidualSkip::NONE,
                residual,
            ),
            MbChoice::Inter { .. } => {
                if zero.all() {
                    residual.reset(ResidualKind::Inter);
                } else {
                    transform_macroblock(
                        &self.dsp,
                        source,
                        pred,
                        x,
                        y,
                        self.random
                            .as_ref()
                            .map_or_else(|| inter_residual_kind(zero), RandomSyntax::inter_kind),
                        &self.quantizers,
                        zero,
                        residual,
                    );
                }
            }
        }
    }

    /// Writes macroblock `(col, row)` as `coded` (see [`write_coded`]).
    /// Code the macroblock `block` at the QP currently loaded (`use_qp`).
    fn code_macroblock(
        &mut self,
        block: &Block,
        decision: BlockDecision,
        source: &Picture,
        reference: Option<&Picture>,
        recon: &mut Picture,
        sub_vectors: Option<[Mv; 4]>,
    ) {
        let (x, y) = (block.x, block.y);
        let (col, row) = Self::mb_index(block);
        if self.static_skip.take() == Some((col, row))
            && let PredictionMode::Inter(mv) = decision.mode
        {
            // Accepted by `try_static_skip`, which already wrote the
            // reconstruction: P_Skip (§7.4.4) with nothing else to do.
            self.write_coded(&CodedMb::Skip(mv), (col, row));
            return;
        }
        let mut pred = self
            .mb_pred
            .take()
            .unwrap_or_else(|| Box::new(Prediction::ZERO));
        let (choice, zero) = self.choose_macroblock(
            block,
            &decision,
            (source, reference, &mut *recon),
            sub_vectors,
            &mut pred,
        );
        self.transform_residual(&choice, zero, source, &pred, (x, y));
        self.trellis_residual(block, source, &pred);
        if let MbChoice::Inter { partition, mvs } = choice {
            self.transform_size_rd(block, (partition, mvs), zero, source, &pred, recon);
        }
        let reconstructed = match choice {
            MbChoice::Inter { partition, mvs }
                if self.residual_rd_multiplier() > 0.0
                    && self.random.is_none()
                    && !self.residual.is_empty() =>
            {
                self.prune_inter_residual(block, (partition, mvs), source, &pred, recon)
            }
            _ => false,
        };
        let residual = &*self.residual;
        let coded = match choice {
            MbChoice::Intra4x4 { modes, ref i4 } => {
                NxnLuma::Blocks4x4(i4).coded(modes.chroma, residual)
            }
            MbChoice::Intra8x8 { modes, ref i8 } => {
                NxnLuma::Blocks8x8(i8).coded(modes.chroma, residual)
            }
            MbChoice::Intra16x16(modes) => CodedMb::Intra16x16 { modes, residual },
            // P_Skip: the skip vector and nothing to code (§7.4.4).
            MbChoice::Inter { partition, mvs }
                if partition == MbPartition::P16x16
                    && Some(mvs[0]) == self.skip_mv_of(block)
                    && residual.is_empty()
                    && !self.random.as_ref().is_some_and(RandomSyntax::keeps_coded) =>
            {
                CodedMb::Skip(mvs[0])
            }
            MbChoice::Inter { partition, mvs } => CodedMb::Inter {
                partition,
                mvs,
                ref_idx: self.coded_ref_idx(),
                mvp: self.known_mvp(block),
                residual,
            },
        };
        // Nothing coded: the prediction is the reconstruction. (I_NxN luma
        // was reconstructed block by block and `pred` holds the Intra_16x16
        // prediction, so this only applies to inter.)
        if reconstructed {
            // Pruning left the reconstruction of what is coded.
        } else if residual.kind.is_inter() && residual.is_empty() {
            write_prediction(recon, &pred, x, y);
        } else {
            reconstruct_macroblock(&self.dsp, recon, &pred, x, y, residual, &self.quantizers);
        }

        if let Some(cavlc) = self.cavlc.as_mut() {
            write_coded_cavlc(
                cavlc,
                &coded,
                &mut self.mbs,
                &self.params,
                (&mut self.qp_tracker, self.mb_qp),
                (col, row),
            );
        } else {
            write_coded(
                (
                    self.cabac.as_mut().expect("encode_block outside a frame"),
                    &self.dsp,
                ),
                self.slice,
                &coded,
                &mut self.mbs,
                &self.params,
                (&mut self.qp_tracker, self.mb_qp),
                (col, row),
            );
        }
        self.predicted.set(None);
        self.mb_pred = Some(pred);
    }

    /// Trellis quantisation (`quant.trellis`) of the coded residual in
    /// `self.residual` of the macroblock `block` predicted by `pred`, priced
    /// with the slice's contexts as they stand. Not past the λ-headroom cap,
    /// where the planned dead zone decides levels. Only the gate is inlined
    /// into the macroblock coder: inlining the search's setup there cost
    /// ~0.1% of host instructions in spills on every macroblock.
    #[inline]
    fn trellis_residual(&mut self, block: &Block, source: &Picture, pred: &Prediction) {
        let Some(trellis) = self.trellis.as_ref() else {
            return;
        };
        let residual = &*self.residual;
        let kind = residual.kind;
        let luma = residual.cbp_luma != 0 && !kind.is_intra_nxn();
        let chroma = trellis.chroma_scale.is_some() && residual.cbp_chroma == CBP_CHROMA_AC;
        if (luma || chroma)
            && trellis.covers(kind, self.slice)
            && self.lambda_scale <= 1.0
            && self.random.is_none()
        {
            self.trellis_coded_residual(block, source, pred, luma);
        }
    }

    /// [`Self::trellis_residual`] past its gate: the luma if `luma`, and the
    /// chroma AC if `quant.trellis_chroma` asks for it.
    #[inline(never)]
    fn trellis_coded_residual(
        &mut self,
        block: &Block,
        source: &Picture,
        pred: &Prediction,
        luma: bool,
    ) {
        let trellis = self.trellis.as_mut().expect("gated by trellis_residual");
        let kind = self.residual.kind;
        let chroma = trellis
            .chroma_scale
            .filter(|_| self.residual.cbp_chroma == CBP_CHROMA_AC);
        let cabac = self.cabac.as_ref().expect("encode_block outside a frame");
        let bins = cabac.bit_counter(&self.bin_costs);
        let (col, row) = Self::mb_index(block);
        let nb = Neighbours::of(&self.mbs, self.params.mb_cols, col, row);
        let intra = !kind.is_inter();
        let lambda = self.rd_lambda * trellis.lambda_scale;
        let settings = |lambda| TrellisSettings {
            bins: &bins,
            lambda,
            max_states: trellis.max_states,
        };
        if luma {
            let luma = LumaTrellis {
                settings: settings(lambda),
                left_coded: std::array::from_fn(|by| {
                    cond_term_flag(nb.left, intra, |m| m.luma_block_coded(3, by))
                }),
                above_coded: std::array::from_fn(|bx| {
                    cond_term_flag(nb.above, intra, |m| m.luma_block_coded(bx, 3))
                }),
            };
            trellis_luma(
                &self.dsp,
                &mut self.residual,
                &self.quantizers,
                &luma,
                &mut trellis.search,
            );
        }
        if let Some(chroma_scale) = chroma {
            let ratio = self.quantizers.chroma_step_ratio(kind);
            let flag = |bits: u8, bx: usize, by: usize| (bits >> (by * 2 + bx)) & 1 != 0;
            let chroma = ChromaTrellis {
                settings: settings(lambda * chroma_scale * ratio * ratio),
                left_coded: std::array::from_fn(|c| {
                    std::array::from_fn(|by| {
                        cond_term_flag(nb.left, intra, |m| flag(m.chroma_ac_coded[c], 1, by))
                    })
                }),
                above_coded: std::array::from_fn(|c| {
                    std::array::from_fn(|bx| {
                        cond_term_flag(nb.above, intra, |m| flag(m.chroma_ac_coded[c], bx, 1))
                    })
                }),
            };
            trellis_chroma_ac(
                &self.dsp,
                source,
                pred,
                (block.x, block.y),
                &self.quantizers,
                &chroma,
                &mut trellis.search,
                &mut self.residual,
            );
        }
    }

    /// [`Self::choose_transform_size`] if `h264.transform_size_rd` asks for
    /// it and the macroblock codes luma with the 8x8 transform.
    #[inline]
    fn transform_size_rd(
        &mut self,
        block: &Block,
        motion: (MbPartition, [Mv; 4]),
        zero: ResidualSkip,
        source: &Picture,
        pred: &Prediction,
        recon: &mut Picture,
    ) {
        if self.tuning.transform_size_rd >= 0.5
            && self.residual.kind == ResidualKind::Inter8x8
            && self.residual.cbp_luma != 0
            && self.lambda_scale <= 1.0
            && self.random.is_none()
        {
            self.choose_transform_size(block, motion, zero, source, pred, recon);
        }
    }

    /// `h264.transform_size_rd`: code the luma of the inter macroblock
    /// `block` (in `self.residual` with 8x8 transforms) with 4x4 transforms
    /// too, and keep whichever has the lower luma squared error plus λ times
    /// the macroblock's CABAC bits. The chroma is the same in both. Leaves
    /// `recon` holding some reconstruction of the macroblock; the caller
    /// reconstructs the winner.
    #[inline(never)]
    fn choose_transform_size(
        &mut self,
        block: &Block,
        (partition, mvs): (MbPartition, [Mv; 4]),
        zero: ResidualSkip,
        source: &Picture,
        pred: &Prediction,
        recon: &mut Picture,
    ) {
        let (x, y) = (block.x, block.y);
        let mut alt = self
            .alt_residual
            .take()
            .unwrap_or_else(|| Box::new(Residual::empty(ResidualKind::Inter)));
        transform_macroblock(
            &self.dsp,
            source,
            pred,
            x,
            y,
            ResidualKind::Inter,
            &self.quantizers,
            ResidualSkip {
                chroma: [true; 2],
                ..zero
            },
            &mut alt,
        );
        alt.copy_chroma_from(&self.residual);
        core::mem::swap(&mut self.residual, &mut alt);
        // `self.residual` holds the 4x4 alternative here, `alt` the 8x8 one.
        if self.residual.cbp_luma != 0 {
            self.trellis_residual_luma(block);
        }
        let ref_idx = self.coded_ref_idx();
        let mvp = self.known_mvp(block);
        let mut cost = |residual: &Residual| {
            reconstruct_macroblock(&self.dsp, recon, pred, x, y, residual, &self.quantizers);
            let coded = CodedMb::Inter {
                partition,
                mvs,
                ref_idx,
                mvp,
                residual,
            };
            luma_sse(&self.dsp, source, recon, (x, y))
                + self.rd_lambda * self.mb_bits(block, &coded)
        };
        if cost(&alt) <= cost(&self.residual) {
            core::mem::swap(&mut self.residual, &mut alt);
        }
        self.alt_residual = Some(alt);
    }

    /// Trellis quantisation of the coded luma in `self.residual` (an inter
    /// macroblock's), as [`Self::trellis_coded_residual`] does it.
    fn trellis_residual_luma(&mut self, block: &Block) {
        let Some(trellis) = self.trellis.as_mut() else {
            return;
        };
        let cabac = self.cabac.as_ref().expect("encode_block outside a frame");
        let bins = cabac.bit_counter(&self.bin_costs);
        let (col, row) = Self::mb_index(block);
        let nb = Neighbours::of(&self.mbs, self.params.mb_cols, col, row);
        let luma = LumaTrellis {
            settings: TrellisSettings {
                bins: &bins,
                lambda: self.rd_lambda * trellis.lambda_scale,
                max_states: trellis.max_states,
            },
            left_coded: std::array::from_fn(|by| {
                cond_term_flag(nb.left, false, |m| m.luma_block_coded(3, by))
            }),
            above_coded: std::array::from_fn(|bx| {
                cond_term_flag(nb.above, false, |m| m.luma_block_coded(bx, 3))
            }),
        };
        trellis_luma(
            &self.dsp,
            &mut self.residual,
            &self.quantizers,
            &luma,
            &mut trellis.search,
        );
    }

    /// Rate-distortion pruning of the coded inter residual in
    /// `self.residual` (see `residual_rd`). Returns whether it considered
    /// any part, in which case `recon` holds the reconstruction of what
    /// remains.
    fn prune_inter_residual(
        &mut self,
        block: &Block,
        (partition, mvs): (MbPartition, [Mv; 4]),
        source: &Picture,
        pred: &Prediction,
        recon: &mut Picture,
    ) -> bool {
        let skip_mv = self.skip_mv_of(block);
        let ref_idx = self.coded_ref_idx();
        let mvp = self.known_mvp(block);
        let lambda = self.residual_rd_multiplier() * self.rd_lambda;
        let residual = &mut *self.residual;
        let considered = prunable_quadrants(residual, self.lambda_scale > 1.0);
        if considered == 0 {
            return false;
        }
        let (x, y) = (block.x, block.y);
        let (col, row) = Self::mb_index(block);
        reconstruct_macroblock(&self.dsp, recon, pred, x, y, residual, &self.quantizers);
        let gains = luma_gains(&self.dsp, source, recon, pred, (x, y), considered);
        let cabac = self.cabac.as_ref().expect("encode_block outside a frame");
        let nb = Neighbours::of(&self.mbs, self.params.mb_cols, col, row);
        let price = |coded: &CodedMb<'_>| {
            let mut counter = cabac.bit_counter(&self.bin_costs);
            let mut qp = self.qp_tracker;
            write_macroblock(
                &mut counter,
                &self.dsp,
                self.slice,
                coded,
                &nb,
                &mut qp,
                self.mb_qp,
            );
            counter
        };
        // Only the considered quadrants are priced: dense quadrants and
        // chroma are kept whatever they cost (instructions, crowd QP 32:
        // +6.4% pricing everything, see `residual_rd`).
        let coded_pattern = (residual.cbp_luma, residual.cbp_chroma);
        (residual.cbp_luma, residual.cbp_chroma) = (considered, CBP_CHROMA_NONE);
        let priced = price(&CodedMb::Inter {
            partition,
            mvs,
            ref_idx,
            mvp,
            residual,
        });
        (residual.cbp_luma, residual.cbp_chroma) = coded_pattern;
        // Without any residual a skip-vector macroblock becomes P_Skip and
        // saves its header too.
        let uncoded_saving_bits = match skip_mv {
            Some(skip_mv) if partition == MbPartition::P16x16 && mvs[0] == skip_mv => {
                priced.part_bits(MbPart::Header) - price(&CodedMb::Skip(skip_mv)).bits()
            }
            _ => 0.0,
        };
        let keep = choose_quadrants(
            residual,
            (considered, &gains),
            &priced,
            uncoded_saving_bits,
            lambda,
        );
        if keep != residual.cbp_luma {
            let residual = &mut *self.residual;
            residual.retain_luma_quadrants(keep);
            if residual.is_empty() {
                write_prediction(recon, pred, x, y);
            } else {
                reconstruct_macroblock(&self.dsp, recon, pred, x, y, residual, &self.quantizers);
            }
        }
        true
    }

    /// Load the weights and quantisers of `qp` (no-op when already loaded).
    /// Inlined: called twice per macroblock, nearly always a no-op.
    #[inline]
    fn use_qp(&mut self, qp: u8) {
        if qp != self.mb_qp {
            self.mb_qp = qp;
            (self.rd_lambda, self.satd_lambda) = self.weights_at(qp);
            self.quantizers = self.quantizers_at(qp);
        }
    }

    /// The λ multiple of inter residual pruning (`residual_rd`):
    /// `h264.residual_rd_lambda`, or 1 on frames planned past QP 51
    /// (`FramePlan::lambda_scale`), where every residual decision follows
    /// the scaled λ. As a default at every QP it failed `J_real` (staged
    /// proxy-4: quality -0.3%, +3.9% instructions, worst-1% NEG +3.4%).
    fn residual_rd_multiplier(&self) -> f32 {
        if self.tuning.residual_rd_lambda > 0.0 {
            self.tuning.residual_rd_lambda
        } else if self.lambda_scale > 1.0 {
            1.0
        } else {
            0.0
        }
    }

    /// The squared-error and SATD weights at `qp`, with the frame's λ scale.
    fn weights_at(&self, qp: u8) -> (f32, f32) {
        let (rd, satd) = frame_weights(qp);
        (rd * self.lambda_scale, satd * self.lambda_scale.sqrt())
    }

    /// The quantisers of `qp`, cached.
    fn quantizers_at(&mut self, qp: u8) -> MbQuantizers {
        let (tuning, chroma_offset) = (&self.tuning, self.chroma_offset);
        let new = |zero_scale: f32| {
            MbQuantizers::new(
                qp,
                tuning.elimination(),
                tuning.inter_rounding_divisor.round() as u8,
                tuning.inter_zero_luma_scale * zero_scale,
                tuning.inter_zero_chroma_scale * zero_scale,
                chroma_offset,
            )
        };
        if self.lambda_scale > 1.0 {
            // Planned past QP 51 (`FramePlan::lambda_scale`): residual is
            // proven zero, and quantised, against the planned step, √(λ
            // scale) times the coded one. Not cached: rare.
            let step_ratio = self.lambda_scale.sqrt();
            return new(step_ratio)
                .with_planned_step(step_ratio, tuning.inter_rounding_divisor.round() as u8);
        }
        *self.quantizer_cache[usize::from(qp)].get_or_insert_with(|| new(1.0))
    }

    fn write_coded(&mut self, coded: &CodedMb<'_>, pos: (usize, usize)) {
        if let Some(cavlc) = self.cavlc.as_mut() {
            write_coded_cavlc(
                cavlc,
                coded,
                &mut self.mbs,
                &self.params,
                (&mut self.qp_tracker, self.mb_qp),
                pos,
            );
        } else {
            write_coded(
                (
                    self.cabac.as_mut().expect("encode_block outside a frame"),
                    &self.dsp,
                ),
                self.slice,
                coded,
                &mut self.mbs,
                &self.params,
                (&mut self.qp_tracker, self.mb_qp),
                pos,
            );
        }
        self.predicted.set(None);
    }

    /// The skip and 16x16 predicted vectors of `block`. The engine asks for
    /// them once per candidate vector (about five times per macroblock), so
    /// they are computed once per macroblock and cached; the cache hit is
    /// inlined (out of line it cost ~40 instructions per call).
    #[inline]
    fn predicted_vectors(&self, block: &Block) -> PredictedVectors {
        let index = Self::mb_index(block);
        if let Some(PredictionCache {
            index: cached,
            ref_idx,
            skip,
            mvp: Some(mvp),
        }) = self.predicted.get()
            && cached == index
            && ref_idx == self.selected_ref
        {
            return PredictedVectors {
                skip: self.usable_skip(skip),
                mvp,
            };
        }
        self.compute_predicted_vectors(block)
    }

    /// `skip` where the selected reference allows `P_Skip` (index 0).
    fn usable_skip(&self, skip: Mv) -> Option<Mv> {
        (self.selected_ref == 0).then_some(skip)
    }

    /// The `P_Skip` vector of `block` if the selected reference allows
    /// `P_Skip`, from the cache when present (with or without the 16x16
    /// prediction), otherwise with both vectors.
    #[inline]
    fn skip_mv_of(&self, block: &Block) -> Option<Mv> {
        match self.predicted.get() {
            Some(cache) if cache.index == Self::mb_index(block) => self.usable_skip(cache.skip),
            _ => self.predicted_vectors(block).skip,
        }
    }

    /// The 16x16 predicted vector of `block` for its coded reference index,
    /// if mode decision already derived it (cached, see `predicted_vectors`).
    fn known_mvp(&self, block: &Block) -> Option<Mv> {
        let cache = self.predicted.get()?;
        let coded_ref = self.coded_ref_idx().unwrap_or(0);
        (cache.index == Self::mb_index(block) && cache.ref_idx == coded_ref)
            .then_some(cache.mvp)
            .flatten()
    }

    /// The coded `ref_idx_l0` of the macroblock being coded: `None` when
    /// the slice has one active reference.
    fn coded_ref_idx(&self) -> Option<u8> {
        (self.active_references > 1).then_some(self.selected_ref)
    }

    /// Estimated bits of `ref_idx_l0` of a 16x16 inter macroblock using the
    /// selected reference, from the slice's contexts as they stand; 0 with
    /// one active reference.
    fn ref_idx_bits(&self, block: &Block) -> f32 {
        if self.active_references <= 1 {
            return 0.0;
        }
        if self.cavlc.is_some() {
            // te(v) (§9.1): one bit for two references, else ue(v).
            return match self.active_references {
                2 => 1.0,
                _ => BitWriter::ue_len(u32::from(self.selected_ref)) as f32,
            };
        }
        let cabac = self.cabac.as_ref().expect("mode decision outside a frame");
        let counter = cabac.bit_counter(&self.bin_costs);
        let inc = ref_idx_first_inc(
            MbPartition::P16x16,
            self.selected_ref,
            &self.neighbours(block),
        );
        (0..=usize::from(self.selected_ref))
            .map(|bin| {
                counter.bin_bits(
                    ref_idx_ctx(bin, inc[0]),
                    bin < usize::from(self.selected_ref),
                )
            })
            .sum()
    }

    /// The `P_Skip` vector of a static candidate, leaving the 16x16
    /// prediction uncomputed unless the skip vector is that prediction:
    /// accepted static skips never need it (-0.6M instructions per static
    /// 1080x720 P frame). Only the static path uses this; elsewhere the
    /// prediction is needed anyway and computing both at once is cheaper.
    fn static_skip_mv(&self, block: &Block) -> Mv {
        let index = Self::mb_index(block);
        if let Some(cache) = self.predicted.get()
            && cache.index == index
        {
            return cache.skip;
        }
        let nb = self.neighbours(block);
        let mut mvp = None;
        let skip = skip_mv_from(&nb, || *mvp.insert(predict_16x16(&nb, 0)));
        self.predicted.set(Some(PredictionCache {
            index,
            ref_idx: 0,
            skip,
            mvp,
        }));
        skip
    }

    #[inline(never)]
    fn compute_predicted_vectors(&self, block: &Block) -> PredictedVectors {
        let index = Self::mb_index(block);
        let nb = self.neighbours(block);
        let ref_idx = self.selected_ref;
        let mvp = predict_16x16(&nb, ref_idx);
        let skip = match self.predicted.get() {
            Some(cache) if cache.index == index => cache.skip,
            _ if ref_idx == 0 => skip_mv_from(&nb, || mvp),
            _ => skip_mv_from(&nb, || predict_16x16(&nb, 0)),
        };
        self.predicted.set(Some(PredictionCache {
            index,
            ref_idx,
            skip,
            mvp: Some(mvp),
        }));
        PredictedVectors {
            skip: self.usable_skip(skip),
            mvp,
        }
    }

    fn neighbours(&self, block: &Block) -> Neighbours<'_> {
        let (col, row) = Self::mb_index(block);
        Neighbours::of(&self.mbs, self.params.mb_cols, col, row)
    }
}

impl H264Params {
    /// The slice data's entropy coder (`h264.cavlc`).
    fn entropy(&self) -> Entropy {
        if self.cavlc >= 0.5 {
            Entropy::Cavlc
        } else {
            Entropy::Cabac
        }
    }

    /// The PPS `chroma_qp_index_offset`, rounded to a whole QP step.
    fn chroma_qp_offset(&self) -> i8 {
        self.chroma_qp_offset.round() as i8
    }

    /// The slice deblocking control of a `frame_type` frame.
    fn deblock_control(&self, frame_type: FrameType) -> DeblockControl {
        match frame_type {
            FrameType::Inter if self.deblock_inter < 0.5 => DeblockControl::OFF,
            _ => DeblockControl::DEFAULT,
        }
    }
}

impl CodecBackend for H264Backend {
    // mb_qp_delta (§7.3.5) carries `BlockDecision::qp_offset`.
    const SUPPORTS_BLOCK_QP: bool = true;
    const BLOCK_QP_LOG2: u8 = MB_LOG2;
    /// `begin_frame` reads only committed state (reference bookkeeping,
    /// `idr_pic_id`); `commit_frame` advances it.
    const RECODE: bool = true;
    const REPORTS_CODED_BITS: bool = true;

    /// The slice's bits so far: the CAVLC writer's, or the CABAC coder's.
    fn coded_bits(&mut self) -> Option<u64> {
        match (&self.cavlc, &self.cabac) {
            (Some(cavlc), _) => Some(cavlc.bit_len()),
            (None, Some(cabac)) => Some(cabac.bit_len()),
            (None, None) => None,
        }
    }

    // λ past QP 51 also widens the zero-residual bounds and prunes residual
    // (`quantizers_at`, `residual_rd_multiplier`). CBR 150 f, headroom 2
    // octaves: crowd_mot20_06 233 kbps 432 -> 242, pan_jaad_0117 200 kbps
    // 309 -> 204, handheld_mot17_10 200 kbps 301 -> 207.
    const SUPPORTS_LAMBDA_HEADROOM: bool = true;

    const PARAM_SPECS: &'static [&'static [nc_encoder::ParamSpec]] = &[H264Params::SPECS];
    /// Every `h264.*` heuristic is re-read per frame (`set_frame_params`),
    /// so disposable frames can run cheaper ones (`nonbase.h264.*`).
    const FRAME_PARAM_SPECS: &'static [&'static [nc_encoder::ParamSpec]] = &[H264Params::SPECS];
    /// The chroma QP offset is PPS syntax, and the intra trellis switches
    /// are read once into [`H264Trellis`].
    const STREAM_FIXED_PARAMS: &'static [&'static str] = &[
        "h264.chroma_qp_offset",
        "h264.trellis_i16x16",
        "h264.trellis_i4x4",
        "h264.cavlc",
    ];

    /// The shared decision values of the same joint-search set as the
    /// `H264Params` defaults (measured together; validated for H.264 only).
    const DEFAULT_PARAMS: &'static [(&'static str, f32)] = &[
        // Start-up QP cadence (period 8, base frames 4 QP finer, for the
        // first 100 frames after a key, then the engine's period 4 / -3 / +1)
        // and the cadence motion gate (`qp_cadence.rs`; docs/handoff/vvc.md,
        // qp-cascade). J_real (k 0.1, dCPU 0): boost 6 TRAIN WORSE (worst-1%
        // Y +2.4%, lowlight +15.8%); boost 4 TRAIN (18 clips) BETTER -1.8%
        // (lowlight -5.4%, static -2.6%), pulse -0.06 dB; VAL BETTER -5.0%
        // (static -22.5%, crowd -4.5%, bodycam +2.2%), pulse +0.05 dB. Long
        // GOP (900 f, one key, CBR 300): chirla Y 38.14 -> 38.18, pan_bdd
        // 27.04 -> 27.06.
        ("rate.cadence_startup_period", 8.0),
        ("rate.cadence_startup_base_boost", 4.0),
        ("rate.cadence_motion_gate", 1.0),
        // QP cadence 1.4x the engine's (base -4.2 QP, leaves +1.4, start-up
        // base -5.6), re-tuned once MS-SSIM was scored on every frame
        // instead of every 4th (mostly base frames). J_real (k 0.1) vs 1:
        // TRAIN -0.5%, VAL -0.8%, held-out `broad_independent_v1` (13
        // clips, every class) -1.0%. The held-out guard misses are tenths
        // (worst-1% Y +2.19, VMAF +2.06, flicker +0.06 dB): per-frame
        // checks show a start-up transient on a 60 fps pan (uvg_campus at
        // 1200 kbps: 6 frames ~1 dB lower after a +2 dB base frame, mean
        // +0.16 dB) and noise-level worst windows on a 10 fps bodycam
        // (within 0.26 dB). 900-frame single-key runs within 0.00..+0.14
        // dB Y per third. h264-fast and h264-faster keep 1 (held-out
        // worst-1% VMAF +6.6%; VAL worst-1% +4%).
        ("rate.cadence_scale", 1.4),
        ("decision.skip_no_refine_lambdas", 0.209_950_97),
        ("decision.subpel_move_bias", 1.201_243_3),
        ("decision.subpel_move_bias_nonzero", 1.066_726),
        // With `h264.exact_ranking` (default 1): skip refining the search
        // vector when its whole-sample residual variance is under 0.1·λ
        // per sample, where refinement rarely changes the coded result.
        // Seed 0.2 halves the crowd instruction cost (+2.5% vs +7.3% at
        // QP37) but keeps only about two thirds of the BD gain (150-frame
        // screen, 5 TRAIN clips: −6.9/−2.8% vs −9.7/−6.5% PSNR/VMAF v1).
        ("decision.codec_subpel_seed_lambdas", 0.1),
        // The legacy sub-sample move biases (the engine defaults moved to
        // AV1's quality point).
        ("decision.subpel_fractional_bias", 1.1),
        // Exact-rank pruning off: at 1.1 it saves only 0.3% of P
        // instructions (few sub-sample candidates) for J_real proxy-4
        // +0.2% / +0.2%.
        ("decision.exact_rank_margin", 0.0),
        // Psychovisual term (`nc_encoder::psy`): J_real full TRAIN (18 clips,
        // CBR ladder) quality -1.1% (full -0.8%, low -1.4%; NEG -1.7%,
        // MS-SSIM -1.0%, PSNR-YUV +0.1%; pan -1.7/-3.3%), no guards, for
        // +2.7% P-frame instructions: J -0.56% at H.264's Default k 0.2.
        // Long single-key runs (crowd_mot20_02 1500 f at 600 kbps,
        // static_chirla 1800 f at 300 kbps): per-third PSNR-Y within
        // ±0.1 dB, no drift. HEVC 0.25 lost (proxy-4 +1.3%), so it is
        // H.264's only.
        ("decision.psy_strength", 0.25),
        // Trellis quantisation of inter macroblocks' luma (`quant.*`,
        // docs/handoff/trellis.md), one survivor: J_real staged TRAIN (18
        // clips) at k 0.1 BETTER, quality -2.0% (full -2.3%, low -1.7%),
        // 0/18 clips worse, every class better (lowlight -3.5%, static -3.0%,
        // pan -2.9%, crowd -2.8%, texture -1.1%, bodycam -0.7%), NEG -3.4%,
        // VMAF v1 -2.5%, no guard; periodic-key pulse -0.01 dB. CPU: P frames
        // +18.9% host (66.7 -> 79.3 M, QP32), Jetson +17.7% (crowd QP32): J
        // -0.2 with 2:1 weighting, a marginal arm. Single-key long GOPs, YUV
        // per third: airport 600 kbps +0.04/+0.02/+0.04 dB, homes390 300
        // kbps +0.02/-0.05/0.00, chirla 150 kbps -0.24/-0.25/-0.08 at 2.7%
        // less rate (CBR undershoot).
        ("quant.trellis", 1.0),
        ("quant.trellis_states", 1.0),
        // Chroma AC keeps the plain quantiser.
        ("quant.trellis_chroma", 0.0),
        // The Default's quality set (2026-11-10, docs/handoff/h264.md
        // "Default quality set"): CPU spent where J_real at k 0.1 pays,
        // toward Default = 1.4x h264-fast. Single levers, J_real proxy-6
        // quality / host P instructions: full-pel probe -2.3% / +8.0%,
        // refining the cheap vector -1.3% / +7.5%, quarter-sample walk
        // -0.8% / +0.8%, intra gate 25 -0.7% / +0.4%, all Intra_4x4 modes
        // -0.5% / 0, full-pel window 4 -0.3% / +0.8%, trellis λ x1.3
        // -0.3% / +1.7%, mixed motion -0.2% / +0.2%, λ split gate 0.8 -0.3%
        // / +2.8% (TRAIN). h264-fast pins every one of them back.
        ("decision.codec_fullpel_probe", 1.0),
        ("decision.refine_cheap_vector", 1.0),
        ("decision.refine_cheap_min_lambdas", 0.0),
        ("decision.subpel_walk", 1.0),
        ("decision.intra_gate_variance", 25.0),
        ("h264.intra4x4_all_modes", 1.0),
        ("analysis.window_half_width", 4.0),
        ("analysis.mixed_motion", 1.0),
        ("quant.trellis_lambda_scale", 1.3),
        ("h264.split_gate_lambda", 0.8),
        // Intra_8x8 in every frame, on every macroblock that tries intra
        // (docs/handoff/h264.md "Intra_8x8"): J_real staged TRAIN (18 clips,
        // 150 f) at k 0.1 BETTER, J -0.2% (quality full -0.7%, low -0.3%;
        // NEG -1.8%, worst-1% NEG -2.5%; static -2.0%, crowd and lowlight
        // -1.3%, bodycam and pan +0.4%), no guard. Fixed QP 26-38: Riverbed
        // -7.8% PSNR-YUV BD; key frames -5..-12%. CPU: P frames host +2.7%
        // (mean of two QP sets), Jetson crowd +2.2% instructions; key frames
        // +28%. h264-fast and h264-faster pin it off.
        ("h264.intra8x8", 2.0),
        // Tuned jointly (eval/nceval/tune_pool.py, study
        // h264-default-eq2-2026-11-11, equal x86 CPU): Intra_8x8 only above
        // 9.6 SATD/sample, and a mode-decision lambda 0.8875x. Study clips J
        // -0.89%, L3 of the study's draw -0.78% [-1.07, -0.48]; re-checked at
        // head 34dfb1d2 on the disjoint L3 (79 clips, single-point) J -0.71%
        // [-1.03, -0.37], no guard; CPU x86 -0.3%, Jetson cycles -1.7%.
        // h264-fast, -faster, -fastest and -balanced pin the old values.
        ("h264.intra8x8_gate_satd", 9.6),
        // A quality arm, jointly tuned (Default/Fast CPU separation was 1.37x
        // against the 1.5x bar; study h264-default-arms-2026-11-13, (J, CPU)
        // front at k 0.1): intra modes chosen by coded cost in every frame
        // and a lower mode-decision lambda (0.8875 -> 0.6625). Disjoint L3
        // (79 clips, full ladder: the change reshapes the curve) at head
        // 34d6f18b: quality J -3.56% [-3.98, -3.15], J at k 0.1 -2.34%
        // [-2.76, -1.93], no guard; CPU x86 +6.8%, Jetson cycles +14.9%.
        // h264-fast, -faster, -fastest and -balanced pin intra_rd off.
        ("h264.intra_rd", 2.0),
        ("decision.lambda_scale", 0.6625),
        // Gated temporal layers (docs/handoff/hierarchical-p.md "Layer
        // gate", fitted on 44 clips in neither TRAIN nor VAL): L1T3 unless
        // the running averages show repeated inputs (share > 0.25), a mostly
        // static scene (static fraction > 0.68), a shaking camera (jitter
        // > 2) or motion past the search reach (16 px/frame); the engine's
        // CBR layer step ratios. An explicit `--temporal-layers` replaces
        // these; `--param layers.count=1` keeps a stream flat.
        // J_real vs flat (k 0.1, PSNR-YUV / VMAF v1 / MS-SSIM thirds, full
        // gate): TRAIN 18 clips -8.6% (-6.7 / -6.7 / -10.9), VAL 11 clips
        // -5.7% (-2.3 / -3.1 / -10.9), no guard; 900-frame single-key runs
        // within -0.09..+0.03 dB per third (no drift); x86 instructions
        // +0.5%. The disposable-frame cuts of h264-fast stay out (pan
        // -0.32 dB in the long run's middle third).
        // Noisy-still gate (re-fit, 2026-11-11): a still scene whose frame
        // difference is large uncompensable noise stays flat
        // (static_otw_homes390 -5.6% J_real; TRAIN identical, VAL -0.3%,
        // docs/handoff/hierarchical-p.md "Layer gate re-fit").
        ("layers.count", 3.0),
        ("layers.adaptive", 1.0),
        ("layers.adaptive_max_camera_motion", 16.0),
        ("layers.adaptive_max_compensable", 1.0),
        ("layers.adaptive_max_repeat_share", 0.25),
        ("layers.adaptive_max_static", 0.68),
        ("layers.adaptive_max_camera_jitter", 2.0),
        ("layers.adaptive_max_noise_residual", 1.8),
    ];

    fn new(config: &EncoderConfig) -> Result<Self, ConfigError> {
        if config.quantizer > QP_MAX {
            return Err(ConfigError::Quantizer {
                value: config.quantizer,
                min: 0,
                max: QP_MAX,
            });
        }
        if config.block_log2 != MB_LOG2 {
            return Err(ConfigError::BlockSize(config.block_log2));
        }
        let qp = config.quantizer as u8;
        let tuning = H264Params::from_overrides(&config.params);
        let (rd_lambda, satd_lambda) = frame_weights(qp);
        let long_term = uses_long_term_slot(&config.params);
        let mut params = StreamParams::new(config.width, config.height, config.frame_rate)
            .with_structure(
                config.temporal_layers,
                reference_slots(config.temporal_layers),
            )
            .with_signal(config.signal, holds_frames(config));
        if long_term {
            params = params.with_long_term_reference();
        }
        let dsp = Dsp::new();
        let chroma_offset = tuning.chroma_qp_offset();
        Ok(Self {
            tuning,
            refresh: None,
            mc: McKernels::new(&dsp),
            dsp,
            mbs: vec![None; params.mb_count()],
            params,
            rate_control_enabled: false,
            base_qp: qp,
            qp,
            rd_lambda,
            satd_lambda,
            lambda_scale: 1.0,
            mb_qp: qp,
            chroma_offsets: ChromaPolicy::from_params(&config.params).map(ChromaOffsets::new),
            chroma_offset,
            quantizer_cache: vec![None; QP_MAX as usize + 1],
            qp_tracker: QpTracker::new(qp),
            quantizers: MbQuantizers::new(
                qp,
                tuning.elimination(),
                tuning.inter_rounding_divisor.round() as u8,
                tuning.inter_zero_luma_scale,
                tuning.inter_zero_chroma_scale,
                tuning.chroma_qp_offset(),
            ),
            slice: SliceKind::I,
            references: ReferenceTracker::new(
                long_term.then(|| long_term_slot(config.temporal_layers)),
                reference_slots(config.temporal_layers),
            ),
            next_references: None,
            nal_ref_idc: NAL_REF_IDC_IDR,
            active_references: 1,
            selected_ref: 0,
            idr_pic_id: 0,
            cabac: None,
            cavlc: None,
            bin_costs: BinCosts::new(),
            access_unit: Vec::new(),
            sub_vectors: None,
            predicted: Cell::new(None),
            residual: Box::new(Residual::empty(ResidualKind::Inter)),
            alt_residual: None,
            static_skip: None,
            static_pred: Box::new(Prediction::ZERO),
            intra_fallback: None,
            mb_pred: None,
            exact_luma: Cell::new(None),
            cabac_init_idc: 0,
            deblock: DeblockControl::DEFAULT,
            random: None,
            trellis: H264Trellis::from_params(
                &QuantParams::from_overrides(&config.params),
                &tuning,
            ),
        })
    }

    fn set_frame_params(&mut self, params: &nc_encoder::ParamOverrides) {
        let tuning = H264Params::from_overrides(params);
        // The cached quantisers embed the zero-residual bounds and rounding.
        if !tuning.same_quantizers(&self.tuning) {
            self.quantizer_cache.fill(None);
        }
        self.tuning = tuning;
    }

    fn superblock_log2(&self) -> u8 {
        MB_LOG2
    }

    fn min_block_log2(&self) -> u8 {
        MB_LOG2
    }

    fn coded_size(&self, width: usize, height: usize) -> (usize, usize) {
        // Pictures are whole macroblocks; the SPS cropping window hides the
        // padding (§7.4.2.1.1, frame_cropping_flag).
        (
            width.next_multiple_of(MB_SIZE),
            height.next_multiple_of(MB_SIZE),
        )
    }

    fn rate_model(&self) -> nc_encoder::RateModel {
        RATE_MODEL
    }

    fn supports_inter(&self) -> bool {
        true
    }

    fn enable_random_decisions(&mut self, rng: nc_encoder::DebugRng) {
        self.random = Some(RandomSyntax::new(rng));
    }

    fn supports_temporal_layers(&self) -> bool {
        true
    }

    /// Consecutive IDR pictures need different `idr_pic_id`s (§7.4.3): the
    /// first IDR after a resolution change continues the alternation. The
    /// changed SPS takes the other parameter set id (`params.rs`).
    fn continue_from(&mut self, previous: &Self) {
        self.idr_pic_id = previous.idr_pic_id;
        self.params.parameter_set_id = previous.params.parameter_set_id ^ 1;
    }

    /// Slot 1 is then a long-term picture (`references.rs`).
    fn supports_long_term_reference(&self) -> bool {
        true
    }

    /// The long-term picture becomes list index 1 (`references.rs`); each
    /// macroblock codes `ref_idx_l0` for all its partitions.
    fn supports_second_reference(&self) -> bool {
        true
    }

    fn select_reference(&mut self, _block: &Block, reference: ReferenceChoice) {
        self.selected_ref = ref_idx_of(reference);
    }

    fn intra_fallback(&mut self, block: &Block, mv: Mv) {
        self.intra_fallback = Some((Self::mb_index(block), mv));
    }

    /// `I_NxN` picks a direction per 8x8 (`Intra_8x8`) or 4x4 block; the
    /// estimate's 8x8 unit stands for both.
    fn intra_estimate_log2(&self) -> u8 {
        INTRA_ESTIMATE_LOG2
    }

    /// SVT's level-8 exit (`h264.subpel_skip_variance`, default 10): refining
    /// below 10 per sample gains H.264 nothing measurable on the quick clips
    /// and costs 6 M instructions/frame.
    fn subpel_skip_variance(&self) -> f32 {
        let fixed = self.tuning.subpel_skip_variance;
        if self.tuning.subpel_skip_lambda > 0.0 {
            fixed.min(self.tuning.subpel_skip_lambda * self.rd_lambda)
        } else {
            fixed
        }
    }

    fn luma_filter(&self) -> Option<nc_dsp::Filter> {
        (self.tuning.exact_ranking >= 0.5).then_some(nc_dsp::Filter::H264Luma)
    }

    // Exact ranking has already filtered the winning fractional luma; the
    // macroblock reuses it instead of predicting it again. Only the
    // `luma_filter` flow calls `cache_exact_luma`, so bilinear ranking is
    // unaffected.
    const CACHE_EXACT_LUMA: bool = true;

    fn cache_exact_luma(
        &mut self,
        block: &Block,
        mv: Mv,
        filter: nc_dsp::Filter,
        prediction: &[u8],
        _stats: nc_dsp::Variance,
    ) {
        debug_assert_eq!(filter, nc_dsp::Filter::H264Luma);
        let mut pixels = [0; MB_SIZE * MB_SIZE];
        pixels.copy_from_slice(prediction);
        self.exact_luma.set(Some(ExactLuma {
            origin: (block.x, block.y),
            mv,
            ref_idx: self.selected_ref,
            pixels,
        }));
    }

    fn lambda(&self) -> f32 {
        self.rd_lambda
    }

    fn relative_quantizer(&self) -> f32 {
        f32::from(self.qp) / QP_MAX as f32
    }

    fn mv_candidates(&mut self, block: &Block, out: &mut Vec<Mv>) {
        let predicted = self.predicted_vectors(block);
        out.extend(predicted.skip);
        out.push(predicted.mvp);
    }

    fn skip_vector(&mut self, block: &Block) -> Option<Mv> {
        self.skip_mv_of(block)
    }

    // Inline hint: once the engine's leaf walk grew (rectangular leaves),
    // LLVM stopped inlining this into the static-skip test of
    // `decide_inter_leaf`. Without the hint h264-fast costs 0.1% more
    // x86 instructions and 0.3-0.7% more Jetson cycles (crowd and bodycam
    // 720p, 60 frames).
    #[inline]
    fn try_static_skip(
        &mut self,
        block: &Block,
        source: &Picture,
        reference: &Picture,
        recon: &mut Picture,
        test: StaticSkipTest,
    ) -> bool {
        let pos = (block.x, block.y);
        if test.within_noise {
            // The engine's noise skip tested the zero vector; P_Skip codes
            // it only where the skip vector predicted from the neighbours
            // (H.264 §8.4.1.1) is zero.
            let skip = self.static_skip_mv(block);
            if skip == Mv::ZERO
                && let Some((luma, chroma)) = whole_sample_macroblock(reference, pos, skip)
            {
                write_prediction_views(recon, pos, &luma, &chroma);
                self.static_skip = Some(Self::mb_index(block));
                return true;
            }
        }
        if self.tuning.static_skip < 0.5 {
            return false;
        }
        let skip = self.static_skip_mv(block);
        let limits = self.quantizers.inter_zero_limits_scaled(
            self.tuning.static_skip_luma_scale,
            self.tuning.static_skip_chroma_scale,
        );
        // A whole-sample skip vector (nearly always zero on still content)
        // is tested and copied straight from the reference: no prediction
        // buffer, one copy instead of two.
        let accepted = if let Some((luma, chroma)) = whole_sample_macroblock(reference, pos, skip) {
            let zero = residual_skip_views(&self.dsp, source, pos, &luma, &chroma, limits);
            if zero.all() {
                write_prediction_views(recon, pos, &luma, &chroma);
            }
            zero.all()
        } else {
            let pred = &mut self.static_pred;
            predict_inter(&self.mc, reference, pos.0, pos.1, &[skip; 4], pred);
            let zero = residual_skip_within(&self.dsp, source, pred, pos.0, pos.1, limits);
            if zero.all() {
                write_prediction(recon, pred, pos.0, pos.1);
            }
            zero.all()
        };
        if accepted {
            self.static_skip = Some(Self::mb_index(block));
        }
        accepted
    }

    fn copied_reference(&self, block: &Block) -> bool {
        let (col, row) = Self::mb_index(block);
        self.mbs[row * self.params.mb_cols + col]
            .as_ref()
            .is_some_and(|mb| {
                !mb.is_intra() && mb.cbp_luma == 0 && mb.ref_idx == 0 && mb.mvs == [Mv::ZERO; 4]
            })
    }

    fn sub_block_motion(&mut self, _block: &Block, vectors: &[Mv]) {
        // A macroblock has four 8x8 sub-blocks.
        self.sub_vectors = vectors.try_into().ok();
    }

    fn wants_sub_block_motion(&self) -> bool {
        self.tuning.partition_search >= 0.5
    }

    fn mode_bits(&self, block: &Block, mode: &PredictionMode) -> f32 {
        match *mode {
            PredictionMode::IntraDc => INTRA_16X16_BITS,
            PredictionMode::Inter(mv) => {
                let PredictedVectors { skip, mvp } = self.predicted_vectors(block);
                if Some(mv) == skip {
                    return SKIP_BITS;
                }
                INTER_16X16_OVERHEAD_BITS
                    + self.ref_idx_bits(block)
                    + mvd_bits(i32::from(mv.x) - i32::from(mvp.x))
                    + mvd_bits(i32::from(mv.y) - i32::from(mvp.y))
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
        self.access_unit.clear();
        self.predicted.set(None);
        self.exact_luma.set(None);
        self.base_qp = plan.quantizer as u8;
        self.qp = match plan.frame_type {
            FrameType::Key if !self.rate_control_enabled => self
                .base_qp
                .saturating_sub(self.tuning.keyframe_qp_offset.round() as u8),
            _ => self.base_qp,
        };
        self.mb_qp = self.qp;
        self.lambda_scale = plan.lambda_scale;
        let (pps_index, chroma_offset) = match &self.chroma_offsets {
            Some(offsets) => offsets.at(self.qp),
            None => (0, self.tuning.chroma_qp_offset()),
        };
        if chroma_offset != self.chroma_offset {
            self.quantizer_cache.fill(None);
        }
        self.chroma_offset = chroma_offset;
        let pps = pps_id(self.params.parameter_set_id, pps_index);
        (self.rd_lambda, self.satd_lambda) = self.weights_at(self.qp);
        self.quantizers = self.quantizers_at(self.qp);
        self.qp_tracker = QpTracker::new(self.qp);
        let mut next_references = self.references.clone();
        let refs = next_references.next(plan);
        self.active_references = refs.active_references;
        self.selected_ref = 0;
        self.next_references = Some(next_references);
        self.nal_ref_idc = refs.nal_ref_idc;
        self.deblock = self.tuning.deblock_control(plan.frame_type);
        if let Some(random) = self.random.as_mut() {
            self.cabac_init_idc = random.cabac_init_idc();
            self.deblock = random.deblock_control();
        }
        let entropy = self.tuning.entropy();
        let header = match plan.frame_type {
            FrameType::Key => {
                write_nal_unit(
                    &[NAL_SPS],
                    &sequence_parameter_set(&self.params),
                    &mut self.access_unit,
                );
                let offsets = match &self.chroma_offsets {
                    Some(offsets) => offsets.offsets().to_vec(),
                    None => vec![self.chroma_offset],
                };
                for (index, offset) in offsets.into_iter().enumerate() {
                    write_nal_unit(
                        &[NAL_PPS],
                        &picture_parameter_set(
                            pps_id(self.params.parameter_set_id, index),
                            self.params.parameter_set_id,
                            offset,
                            entropy,
                        ),
                        &mut self.access_unit,
                    );
                }
                let header = idr_slice_header(
                    pps,
                    self.idr_pic_id,
                    &refs.marking,
                    self.qp,
                    self.deblock,
                    entropy,
                );
                self.slice = SliceKind::I;
                header
            }
            FrameType::Inter => {
                let init_idc = (entropy == Entropy::Cabac).then_some(self.cabac_init_idc);
                let header = p_slice_header(pps, &refs, self.qp, init_idc, self.deblock);
                self.slice = SliceKind::P;
                header
            }
        };
        let cabac_out = match entropy {
            Entropy::Cabac => header,
            Entropy::Cavlc => {
                self.cavlc = Some(CavlcEncoder::new(
                    header,
                    self.slice,
                    self.active_references,
                    (self.params.mb_count(), self.params.mb_cols),
                ));
                BitWriter::new()
            }
        };
        self.cabac = Some(CabacEncoder::new(
            cabac_out,
            self.slice,
            self.cabac_init_idc,
            self.qp,
        ));
        self.mbs.fill(None);
    }

    fn begin_superblock(&mut self, _x: usize, _y: usize) {}

    fn write_partition(&mut self, _node: &Block, _split: bool) {
        // Macroblocks have no partition syntax at this level.
    }

    fn encode_block(
        &mut self,
        block: &Block,
        decision: &BlockDecision,
        source: &Picture,
        reference: Option<&Picture>,
        recon: &mut Picture,
    ) {
        debug_assert_eq!(block.log2_size(), MB_LOG2);
        let sub_vectors = self.sub_vectors.take();
        let mb_qp = (i32::from(self.qp) + i32::from(decision.qp_offset)).clamp(0, QP_MAX as i32);
        self.use_qp(mb_qp as u8);
        self.selected_ref = ref_idx_of(decision.reference);
        self.code_macroblock(block, *decision, source, reference, recon, sub_vectors);
        self.selected_ref = 0;
        self.use_qp(self.qp);
    }

    fn loop_filter_frame(&mut self, recon: &mut Picture) {
        deblock_picture(
            &self.dsp,
            recon,
            &self.mbs,
            self.params.mb_cols,
            self.chroma_offset,
            self.deblock,
        );
    }

    fn end_frame(&mut self) -> Vec<u8> {
        let cabac = self.cabac.take().expect("end_frame without begin_frame");
        let slice_data = match self.cavlc.take() {
            Some(cavlc) => cavlc.finish(),
            None => cabac.finish(),
        };
        let nal_type = if self.slice == SliceKind::I {
            NAL_TYPE_IDR_SLICE
        } else {
            NAL_TYPE_NON_IDR_SLICE
        };
        let nal = (self.nal_ref_idc << NAL_REF_IDC_SHIFT) | nal_type;
        write_nal_unit(&[nal], &slice_data, &mut self.access_unit);
        std::mem::take(&mut self.access_unit)
    }

    fn commit_frame(&mut self) {
        if let Some(next) = self.next_references.take() {
            self.references = next;
        }
        if self.slice == SliceKind::I {
            self.idr_pic_id ^= 1;
        }
    }
}

/// Writes the syntax of macroblock `(col, row)` as `coded` and records its
/// state for its neighbours and the loop filter.
fn write_coded(
    (cabac, dsp): (&mut CabacEncoder, &Dsp),
    slice: SliceKind,
    coded: &CodedMb<'_>,
    mbs: &mut [Option<MbInfo>],
    params: &StreamParams,
    (qp, mb_qp): (&mut QpTracker, u8),
    (col, row): (usize, usize),
) {
    let index = row * params.mb_cols + col;
    let nb = Neighbours::of(mbs, params.mb_cols, col, row);
    let info = cabac.write(|writer| {
        let info = write_macroblock(writer, dsp, slice, coded, &nb, qp, mb_qp);
        // end_of_slice_flag (§7.3.4): the single slice ends after the last
        // macroblock of the picture.
        writer.encode_terminate(index + 1 == params.mb_count());
        info
    });
    mbs[index] = Some(info);
}

/// [`write_coded`] for a CAVLC stream (`h264.cavlc`). Out of line, so the
/// CABAC path inlines as it did before CAVLC existed (+0.5% instructions
/// at the Default otherwise).
#[inline(never)]
fn write_coded_cavlc(
    cavlc: &mut CavlcEncoder,
    coded: &CodedMb<'_>,
    mbs: &mut [Option<MbInfo>],
    params: &StreamParams,
    qp: (&mut QpTracker, u8),
    (col, row): (usize, usize),
) {
    let index = row * params.mb_cols + col;
    let nb = Neighbours::of(mbs, params.mb_cols, col, row);
    let info = cavlc.write(coded, &nb, qp, index);
    mbs[index] = Some(info);
}

/// condTermFlagN of a `coded_block_flag` (§9.3.3.1.1.9) for a block in a
/// neighbouring macroblock `m`: unavailable counts as coded for an intra
/// current macroblock and uncoded for an inter one, skipped as uncoded,
/// otherwise the neighbour block's `flag`.
fn cond_term_flag(m: Option<&MbInfo>, current_intra: bool, flag: impl Fn(&MbInfo) -> bool) -> bool {
    match m {
        None => current_intra,
        Some(m) => !m.is_skip() && flag(m),
    }
}

/// The residual kind of a coded inter macroblock. Coded inter luma always
/// uses the 8x8 transform: on the quick clips choosing per macroblock by 4x4-
/// vs 8x8-Hadamard SATD gave -8.0% BD-rate against openh264-baseline and
/// always-8x8 -8.1% (always-4x4 +1.6%), so the two SATDs (about 5M
/// instructions per 720p frame) buy nothing. Proven-zero luma is coded empty
/// (the 4x4 kind, no levels).
fn inter_residual_kind(zero: ResidualSkip) -> ResidualKind {
    if zero.luma {
        ResidualKind::Inter
    } else {
        ResidualKind::Inter8x8
    }
}

impl H264Params {
    /// Whether `other` builds the same [`MbQuantizers`] as `self`.
    fn same_quantizers(&self, other: &Self) -> bool {
        // The values come from the same overrides: equal exactly, or changed.
        let same = |a: f32, b: f32| a.to_bits() == b.to_bits();
        self.elimination() == other.elimination()
            && same(self.inter_rounding_divisor, other.inter_rounding_divisor)
            && same(self.inter_zero_luma_scale, other.inter_zero_luma_scale)
            && same(self.inter_zero_chroma_scale, other.inter_zero_chroma_scale)
    }

    /// The sparse-residual elimination thresholds, as whole cost units.
    fn elimination(&self) -> Elimination {
        let units = |v: f32| v.round() as u32;
        Elimination {
            luma_8x8: units(self.eliminate_luma_8x8),
            luma_mb: units(self.eliminate_luma_mb),
            chroma_ac: units(self.eliminate_chroma_ac),
        }
    }
}

/// QP changes only at frame boundaries. Keep the old operation order so
/// caching removes repeated exponentials and square roots without new weights.
/// Two legal TRAIN cost points: P -0.39%, GOP300 -0.38% instructions, exact bytes.
fn frame_weights(qp: u8) -> (f32, f32) {
    let lambda = LAMBDA_SCALE * 2f32.powf((f32::from(qp) - 12.0) / 3.0);
    (lambda, lambda.sqrt())
}

#[cfg(test)]
mod weight_tests {
    use super::*;
    use nc_encoder::{Codec, FrameRate, ParamOverrides};

    #[test]
    fn frame_cache_retains_old_bits_across_key_inter_transitions_and_clipping() {
        for qp in 0..=QP_MAX as u8 {
            for offset in [0.0, 6.0, 12.0] {
                let mut params = ParamOverrides::default();
                params.set("h264.keyframe_qp_offset", offset);
                let config = EncoderConfig {
                    codec: Codec::H264,
                    width: 16,
                    height: 16,
                    frame_rate: FrameRate { num: 30, den: 1 },
                    quantizer: u32::from(qp),
                    intra_period: 300,
                    block_log2: MB_LOG2,
                    key_qp_offset: None,
                    temporal_layers: 1,
                    signal: nc_encoder::StreamSignal::default(),
                    params,
                };
                let mut backend = H264Backend::new(&config).unwrap();
                for (index, frame_type) in [FrameType::Key, FrameType::Inter, FrameType::Key]
                    .into_iter()
                    .enumerate()
                {
                    backend.begin_frame(&FramePlan {
                        index: index as u64,
                        frame_type,
                        quantizer: u32::from(qp),
                        references: nc_encoder::FrameReferences::FLAT,
                        lambda_scale: 1.0,
                        block_qp: false,
                    });
                    let expected_qp = if frame_type == FrameType::Key {
                        qp.saturating_sub(offset as u8)
                    } else {
                        qp
                    };
                    let expected = LAMBDA_SCALE * 2f32.powf((f32::from(expected_qp) - 12.0) / 3.0);
                    assert_eq!(backend.lambda().to_bits(), expected.to_bits());
                    assert_eq!(backend.satd_lambda.to_bits(), expected.sqrt().to_bits());
                }
            }
        }
    }
}
