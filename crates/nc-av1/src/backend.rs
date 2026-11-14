//! The AV1 [`CodecBackend`]: key frames of DC-predicted blocks, and inter
//! frames predicted from the previous frame.
//!
//! Per frame: one temporal unit = temporal delimiter + (key frames only)
//! sequence header + `OBU_FRAME` (frame header + one tile). Per block, the
//! syntax of AV1 §5.11.5 (`decode_block`) with every optional tool off:
//!
//! - key frames (`intra_frame_mode_info`): `skip`, `intra_frame_y_mode`,
//!   `uv_mode`;
//! - inter frames (`inter_frame_mode_info`): `skip`, `is_inter`, then for
//!   inter blocks the reference (`LAST_FRAME`), the inter mode
//!   (`NEARESTMV` / `NEARMV` / `GLOBALMV` / `NEWMV`), `drl_mode` and the
//!   vector difference; for intra blocks `y_mode` and `uv_mode`;
//!
//! then the residual of Y, U, V (one transform block each:
//! `TX_MODE_LARGEST`, square blocks ≤ 16x16).

use std::rc::Rc;

use nc_common::{BitWriter, Picture, PlaneIndex, PlaneView};
use nc_encoder::{
    Block, BlockDecision, CodecBackend, ConfigError, EncoderConfig, FramePlan, FrameReferences,
    FrameType, IntraModeSet, MAX_REFERENCE_SLOTS, Mv, PlaneZeroLimits, PredictionMode, QuantParams,
    ReferenceChoice, RefreshWindow, SlotMask, SpatialConfig, SpatialLayerFrame, StaticSkipTest,
    ZeroLimits, copy_zero_vector_prediction, directions_by_orientation, residual_quantises_to_zero,
    zero_vector_residual_is_zero,
};

use crate::cdef::{self, CdefParams};
use crate::cdf::CdfContext;
use crate::cfl::ChromaFromLuma;
use crate::coeff::trellis::{RateModel, TrellisQuantizer};
use crate::coeff::{
    CoeffCdfs, PlaneType, TxbContext, coefficient_scan, end_of_block_within, entropy_context,
};
use crate::grain::{GrainParams, GrainState};
use crate::inter;
use crate::intra::{
    FILTER_INTRA_MODES, LumaIntra, MAX_ANGLE_DELTA, filter_intra_allowed, filter_intra_size_context,
};
use crate::loopfilter::{self, LoopFilterParams};
use crate::mds3;
use crate::modeinfo::{Av1Mv, BlockMode, InterRefs, MI_LOG2, ModeInfo, ModeInfoGrid};
use crate::mvref::{
    CompoundMvStack, LastOnly, MvStack, SingleRef, find_compound_mv_stack, find_mv_stack,
};
use crate::obu::{
    FrameHeader, FrameIdLengths, FrameIds, FrameKind, FrameSizeSyntax, IntraRefresh, ObuType,
    RestorationTool, SequenceTools, SlotUse, sequence_header, spatial_sequence_header, write_obu,
    write_obu_layer,
};
use crate::predict::{FilterIntraMode, IntraMode, dc_value};
use crate::quant::QuantSteps;
use crate::range_coder::{LiteralSlot, SpanSlot, SymbolWriter};
use crate::reference_store::{HeldReference, ReferenceStore, reference_scale_allowed};
use crate::restoration::{self, FrameRestoration, SearchSettings, WienerReferences};
use crate::superres::{SUPERRES_DENOM_MAX, SUPERRES_NUM, Superres, downscaled_width};
use crate::tables::{
    AC_QLOOKUP, CFL_ALLOWED, CFL_NOT_ALLOWED, DC_PRED, INTRA_MODE_CONTEXT, INTRA_MODES, Partition,
    UV_DC_PRED, UV_INTRA_MODES_CFL_ALLOWED, UV_INTRA_MODES_CFL_NOT_ALLOWED, coeff_cdf_q_ctx,
};
use crate::transform::{
    ForwardPolicy, TxSize, dequantized_extent, forward_dct_with, inverse_dct_add_eob,
    inverse_dct_dc_add, quantize_coded, quantize_coded_from,
};
use crate::txsize::{self, TxSplit, TxfmContexts};
use nc_dsp::{Av1Quantizer, BlockSize, CdefSource, Dsp};
use nc_encoder::symbol_cost::symbol_bits;

mod delta_q;
mod p_intra;
mod stress;
use delta_q::DeltaQ;
use p_intra::InterFrameIntra;
use stress::RandomSyntax;

/// 64x64 superblocks (`use_128x128_superblock = 0`).
const SUPERBLOCK_LOG2: u8 = 6;
/// Luma samples AV1 in-loop filtering spreads a sample's influence (see
/// `CodecBackend::loop_filter_reach` below).
const AV1_LOOP_FILTER_REACH: usize = 16;
/// Chroma-from-luma mode in the `CfL`-enabled UV alphabet.
const UV_CFL_PRED: usize = 13;
/// Smallest leaf block. 4:2:0 chroma of an 8x8 block is 4x4, the smallest
/// AV1 transform; smaller luma blocks would need chroma merged across blocks.
const MIN_BLOCK_LOG2: u8 = 3;
/// Largest leaf block: a whole 64x64 superblock (64x64 transform, of which
/// the top-left 32x32 coefficients are coded).
const MAX_BLOCK_LOG2: u8 = 6;
/// Range of `base_q_idx`; 0 would make frames lossless (a different tool set).
const QINDEX_MIN: u32 = 1;
const QINDEX_MAX: u32 = 255;
/// AV1's rate model: the shared qindex calibration, plus the start-up
/// lookahead's first-key cap (`rate.first_key_lookahead`) of 0.3 buffers
/// under camera motion. `J_real` TRAIN, 150 f, vs off: -0.5% (pan -1.9%,
/// bodycam -0.9%, lowlight -0.8%, texture +0.5%, worst-1% Y -1.2%; crowd
/// and static byte-identical, no guard tripped).
///
/// QP cadence 1.4x the engine's, re-tuned once MS-SSIM was scored on every
/// frame instead of every 4th (mostly base frames). `J_real` (k 0.1) vs 1:
/// TRAIN -1.5%, VAL -0.8%; 1.8 WORSE (worst-1% +6..+9%), 0.5 WORSE (+3%).
/// av1-fast TRAIN -1.4% / VAL -0.4%, av1-faster -0.2% / -0.7% (k 0.6).
/// Pool L3 (79 clips, native 10-30 fps): J -0.81% [-1.24, -0.39]. The 10 fps
/// clips are neutral (+0.3%), so the held-out 10 fps bodycam loss (+7.6%) is
/// that clip's alone; a cadence shortened to keep its duration at low frame
/// rates did worse there (H.264 10 fps +0.5%). The periodic-key pulse rises
/// +0.29 dB (guard 0.25; a 15 fps lowlight clip +0.8 dB).
const RATE_MODEL: nc_encoder::RateModel = nc_encoder::RateModel {
    cadence_scale: 1.4,
    first_key_lookahead: 0.3,
    ..nc_encoder::RateModel::AC_LOOKUP_QINDEX
};
/// Engine vectors are in quarter samples, AV1's in eighths.
const AV1_MV_PER_ENGINE_MV: i32 = 2;
/// Highest list position NEARMV can name (`RefMvIdx` 1 + two `drl_mode` bits).
const MAX_NEAR_INDEX: usize = 3;
const AC_Q_PER_SAMPLE_STEP: f32 = 8.0;
/// One whole quantiser step in the 1/128-step rounding offsets.
const ROUNDING_Q7_ONE: f32 = 128.0;
/// Per-mille representation keeps the default key-step scale exactly equal
/// to the original f64 0.24 when converting the registered f32 parameter.
const KEY_STEP_SCALE_DENOMINATOR: f64 = 1000.0;
/// Range of a coded `delta_q` value (AV1 §5.9.13: `su(1+6)`).
const DELTA_Q_MIN: i32 = -64;
const DELTA_Q_MAX: i32 = 63;
/// `angle_delta_y` symbols: deltas −3..=3.
const ANGLE_DELTA_SYMBOLS: usize = 2 * MAX_ANGLE_DELTA as usize + 1;
/// Selects equivalent forward/reverse EOB scans for sparse coefficient blocks.
/// This TRAIN-derived speed cutoff changes scan work, never encoding decisions.
const SPARSE_EOB_MAX_NONZERO: u32 = 8;

/// How an inter block's vector is signalled (AV1 `YMode` of inter blocks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InterMode {
    /// List position 0.
    Nearest,
    /// List position 1, 2 or 3 (`RefMvIdx`: 1 + the `drl_mode` run).
    Near(usize),
    /// The global (zero) vector.
    Global,
    /// A coded difference from list position 0, 1 or 2 (`RefMvIdx`: the
    /// `drl_mode` run). The encoder's own choice always predicts from 0.
    New(usize),
}

impl InterMode {
    /// The mode as recorded for later blocks.
    fn block_mode(self) -> BlockMode {
        match self {
            Self::Nearest => BlockMode::Nearest,
            Self::Near(_) => BlockMode::Near,
            Self::Global => BlockMode::Global,
            Self::New(_) => BlockMode::New,
        }
    }

    /// Whether this is `NEWMV`.
    fn is_new(self) -> bool {
        matches!(self, Self::New(_))
    }

    /// The list position the mode names or predicts from, and the first
    /// `drl_mode` position: `NEWMV` codes the run from 0, `NEARMV` from 1
    /// (AV1 §5.11.23 `read_drl_mode`). `None` for modes without `drl_mode`.
    fn drl(self) -> Option<(usize, usize)> {
        match self {
            Self::New(index) => Some((index, 0)),
            Self::Near(index) => Some((index, 1)),
            Self::Nearest | Self::Global => None,
        }
    }
}

/// The `drl_mode` symbols coding list position `index` for a mode whose run
/// starts at `first` (AV1 §5.11.23): one symbol per position the stack can
/// still move past, until the run stops at `index`. Each item is the
/// position (for the context) and the symbol.
fn drl_symbols(
    stack: &MvStack,
    (index, first): (usize, usize),
) -> impl Iterator<Item = (usize, bool)> {
    (first..first + DRL_SYMBOLS)
        .filter(move |&position| stack.count > position + 1)
        .take_while(move |&position| position <= index)
        .map(move |position| (position, index > position))
}

/// How a compound inter block's vector pair is signalled (AV1 §5.11.23, §5.11.26
/// `compound_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompoundInterMode {
    /// `NEAREST_NEARESTMV` (symbol 0): stack position 0 for both lists.
    NearestNearest,
    /// `NEAR_NEARMV` (symbol 1): stack position `1..=3` for both lists.
    NearNear(usize),
    /// `NEAREST_NEWMV` (symbol 2): position 0 for list 0, coded delta from 0 for list 1.
    NearestNew,
    /// `NEW_NEARESTMV` (symbol 3): coded delta from 0 for list 0, position 0 for list 1.
    NewNearest,
    /// `NEAR_NEWMV` (symbol 4): position `1..=3` for list 0, coded delta from that position for list 1.
    NearNew(usize),
    /// `NEW_NEARMV` (symbol 5): coded delta from position `1..=3` for list 0, that position for list 1.
    NewNear(usize),
    /// `GLOBAL_GLOBALMV` (symbol 6): zero vector for both lists.
    GlobalGlobal,
    /// `NEW_NEWMV` (symbol 7): coded delta from position `0..=2` for both lists.
    NewNew(usize),
}

impl CompoundInterMode {
    const fn symbol(self) -> usize {
        match self {
            Self::NearestNearest => 0,
            Self::NearNear(_) => 1,
            Self::NearestNew => 2,
            Self::NewNearest => 3,
            Self::NearNew(_) => 4,
            Self::NewNear(_) => 5,
            Self::GlobalGlobal => 6,
            Self::NewNew(_) => 7,
        }
    }

    const fn block_mode(self) -> BlockMode {
        match self {
            Self::NearestNearest => BlockMode::Nearest,
            Self::NearNear(_) => BlockMode::Near,
            Self::GlobalGlobal => BlockMode::Global,
            Self::NearestNew
            | Self::NewNearest
            | Self::NearNew(_)
            | Self::NewNear(_)
            | Self::NewNew(_) => BlockMode::New,
        }
    }

    const fn drl(self) -> Option<(usize, usize)> {
        match self {
            Self::NewNew(index) => Some((index, 0)),
            Self::NearNear(index) | Self::NearNew(index) | Self::NewNear(index) => Some((index, 1)),
            Self::NearestNearest | Self::NearestNew | Self::NewNearest | Self::GlobalGlobal => None,
        }
    }

    const fn new_mv_base(self) -> [Option<usize>; 2] {
        match self {
            Self::NearestNearest | Self::NearNear(_) | Self::GlobalGlobal => [None, None],
            Self::NearestNew => [None, Some(0)],
            Self::NewNearest => [Some(0), None],
            Self::NearNew(index) => [None, Some(index)],
            Self::NewNear(index) => [Some(index), None],
            Self::NewNew(index) => [Some(index), Some(index)],
        }
    }
}

fn drl_symbols_compound(
    stack: &CompoundMvStack,
    (index, first): (usize, usize),
) -> impl Iterator<Item = (usize, bool)> {
    (first..first + DRL_SYMBOLS)
        .filter(move |&position| stack.count > position + 1)
        .take_while(move |&position| position <= index)
        .map(move |position| (position, index > position))
}

/// Most `drl_mode` symbols of one block.
const DRL_SYMBOLS: usize = 2;

/// Coefficient entropy contexts (AV1 `AboveLevelContext` / `AboveDcContext`,
/// stored combined as in libaom), per plane, one byte per 4x4 unit.
#[derive(Debug)]
struct CoeffContexts {
    above: [Vec<u8>; 3],
    /// Left contexts of the current superblock row only.
    left: [Vec<u8>; 3],
}

impl CoeffContexts {
    /// Contexts for a frame `mi_cols` mode-info units wide.
    fn new(mi_cols: usize) -> Self {
        let sb_mi = 1usize << (SUPERBLOCK_LOG2 as usize - MI_LOG2);
        Self {
            above: [vec![0; mi_cols], vec![0; mi_cols / 2], vec![0; mi_cols / 2]],
            left: [vec![0; sb_mi], vec![0; sb_mi / 2], vec![0; sb_mi / 2]],
        }
    }
}

/// One CDEF syntax slot per 64x64 filter block of a coded picture.
fn cdef_slots(coded_width: usize, coded_height: usize) -> Vec<Option<LiteralSlot>> {
    vec![None; coded_width.div_ceil(64) * coded_height.div_ceil(64)]
}

/// The reference list of the block currently being decided and coded.
#[derive(Debug, Clone, Copy)]
struct CurrentBlock {
    x: usize,
    y: usize,
    secondary_ref: bool,
    stack: MvStack,
    /// Candidate-independent inter syntax cost, before any block CDF updates.
    /// Caching saves 0.22% P instructions on two legal TRAIN points, exact streams.
    inter_base_bits: f32,
}

/// Coding state that depends on the coded frame size: its size, mode-info and
/// context grids and CDEF state. Spatial SVC keeps one per layer and swaps
/// them whole when the coded layer changes, so switching layers inside a
/// temporal unit never reallocates.
#[derive(Debug)]
struct LayerGrids {
    /// Coded frame size (`FrameWidth`, `FrameHeight`): under superres the
    /// width is the reduced one the engine codes.
    width: usize,
    height: usize,
    mode_info: ModeInfoGrid,
    contexts: CoeffContexts,
    /// Transform-size contexts of the coded neighbours.
    txfm: TxfmContexts,
    cdef_slots: Vec<Option<LiteralSlot>>,
    /// CDEF choices the current frame reuses from earlier frames and updates.
    cdef_memory: CdefMemory,
    /// `cdef_memory` as the last committed frame left it (the recode
    /// contract: each coding of a frame starts from it).
    committed_cdef_memory: CdefMemory,
    /// The kind of the last committed frame.
    committed_kind: FrameKind,
}

impl LayerGrids {
    fn new(width: usize, height: usize) -> Self {
        // The mode-info grid covers whole 8x8 luma units
        // (MiCols = 2 * ((frame_width + 7) >> 3), §7.21).
        let (coded_width, coded_height) = coded_picture_size(width, height);
        let mi_cols = coded_width >> MI_LOG2;
        let mi_rows = coded_height >> MI_LOG2;
        Self {
            width,
            height,
            mode_info: ModeInfoGrid::new(mi_cols, mi_rows),
            contexts: CoeffContexts::new(mi_cols),
            txfm: TxfmContexts::new(2 * width.div_ceil(8)),
            cdef_slots: cdef_slots(coded_width, coded_height),
            cdef_memory: CdefMemory::default(),
            committed_cdef_memory: CdefMemory::default(),
            committed_kind: FrameKind::Key,
        }
    }
}

/// `(coded_size, render_size)` stored in one DPB reference slot.
type SlotDimensions = ((usize, usize), (usize, usize));

/// Spatial SVC state on [`Av1Backend`] (`2` or `3` spatial layers).
#[derive(Debug)]
struct BackendSpatial {
    config: SpatialConfig,
    /// The grids of the layers not currently being coded (the active one is
    /// [`Av1Backend::grids`]).
    parked: [Option<LayerGrids>; nc_encoder::MAX_SPATIAL_LAYERS],
    active_spatial_id: usize,
    current_layer: SpatialLayerFrame,
    /// Coded and render dimensions stored in each DPB slot (`(coded, render)`).
    slot_size: [Option<SlotDimensions>; MAX_REFERENCE_SLOTS],
    /// Spatial layer id (`0..config.layers`) of the picture currently occupying
    /// each DPB slot, so a layer distinguishes its own unscaled temporal slots
    /// from lower-layer inter-layer slots held in `store`.
    slot_layer: [Option<u8>; MAX_REFERENCE_SLOTS],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantizerMode {
    Fixed,
    RateControlled,
}

/// The area coded for a `width` x `height` frame: whole 8x8 luma units, as
/// the mode-info grid covers (§7.21). Headers carry the true frame size;
/// decoders output only that.
pub(crate) fn coded_picture_size(width: usize, height: usize) -> (usize, usize) {
    (width.next_multiple_of(8), height.next_multiple_of(8))
}

/// The AV1 backend.
#[derive(Debug)]
pub struct Av1Backend {
    /// The frame's [`FramePlan::lambda_scale`], applied to every λ.
    lambda_scale: f32,
    /// The frame-size state of the layer being coded (one per spatial
    /// layer under spatial SVC, swapped whole by [`BackendSpatial`]).
    grids: LayerGrids,
    /// The stream size: the sequence header's maximum frame size and every
    /// frame's render size.
    stream_size: (usize, usize),
    /// `av1.superres_denom` above 8: superres state.
    superres: Option<Superres>,
    /// Reference slots whose normative picture is not the engine's
    /// reconstruction (superres, coded-size switches).
    store: ReferenceStore,
    /// This inter frame's reference is held in `store` (set per frame: the
    /// hot paths test one flag).
    reference_held: bool,
    steps: QuantSteps,
    /// Chroma steps (`frame_q_idx + chroma_delta_q`).
    chroma_steps: QuantSteps,
    /// The CDFs of the frame being coded, adapted symbol by symbol. Boxed,
    /// like the saved copies, so the backend stays small to move (a CDF
    /// context is about 16 KiB; unoptimised builds copy it through the stack).
    cdfs: Box<CdfContext>,
    /// The CDFs saved with each reference slot; an inter frame loads its
    /// reference's (`primary_ref_frame` = `LAST_FRAME`). Slots refreshed by
    /// one frame share its saved context.
    slot_cdfs: [Option<Rc<CdfContext>>; MAX_REFERENCE_SLOTS],
    /// The current frame's place in the temporal structure.
    references: FrameReferences,
    /// Frame ids of the next frame and of each slot (`obu.rs`), when the
    /// stream codes them.
    frame_ids: Option<FrameIds>,
    /// Temporal layers of the stream (1: flat, no OBU extension headers).
    temporal_layers: u8,
    /// Colour and timing signalling written in the sequence header.
    signal: nc_encoder::StreamSignal,
    /// Nominal frame rate (the sequence header's timing clock).
    frame_rate: nc_encoder::FrameRate,
    /// The configured `base_q_idx` (inter frames).
    quantizer_mode: QuantizerMode,
    base_q_idx: u32,
    /// `base_q_idx` of the frame being coded.
    frame_q_idx: u32,
    /// `CurrentQIndex` of the block being coded (`frame_q_idx` outside
    /// `encode_block`, so the engine's decisions see the frame λ and scale
    /// it themselves): `steps`, `chroma_steps` and `lambda()` follow it.
    block_q_idx: u32,
    /// Superblock quantizer deltas (`BlockDecision::qp_offset`).
    delta_q: DeltaQ,
    /// One offset shared by chroma quantization and the eventual frame header.
    frame_chroma_delta_q: i32,
    /// Registered frame-kind rounding, converted once rather than per block.
    rounding_q7: i32,
    /// The same for chroma (`av1.inter_chroma_rounding_q7` on P frames).
    chroma_rounding_q7: i32,
    /// Integer MDS3 controls, converted once at encoder creation.
    shave_min_zero_run: usize,
    chroma_skip_sad_num: u32,
    /// Registered backend base-frame cadence and this frame's cached flag.
    base_period: u64,
    is_base_frame: bool,
    /// Effective optional pre-transform threshold, selected once per frame.
    pre_tx_score_this_frame: f32,
    /// CDEF strengths of the frame being coded.
    cdef: CdefParams,
    /// Padded input retained across planes; every fill rewrites samples and borders.
    cdef_input: CdefSource,
    /// A second strength and its deferred per-filter-block syntax slots.
    cdef_alternate: Option<CdefParams>,
    cdef_frame: CdefFrameChoice,
    noise_sigma: f32,
    /// The engine chroma policy (`chroma.policy` 1): replaces
    /// `av1.chroma_delta_q` and its noise and key-frame adaptations.
    chroma_policy: Option<nc_encoder::chroma_policy::ChromaPolicy>,
    kind: FrameKind,
    current: Option<CurrentBlock>,
    writer: SymbolWriter,
    dsp: Dsp,
    scratch: Scratch,
    /// The block `try_static_skip` accepted (`av1.still_skip`): the next
    /// `encode_block` codes it with no residual.
    still_skipped: Option<(usize, usize)>,
    /// The block [`CodecBackend::encode_block`] last coded as a copy of the
    /// reference's luma (zero motion, no luma residual), by position.
    luma_copy: Option<(usize, usize)>,
    /// The block (x, y, log2 size) whose intra the engine chose only
    /// through its directional estimate, and the inter vector it beat
    /// (`CodecBackend::intra_fallback`), consumed by its `encode_block`
    /// (`p_intra.rs`).
    intra_fallback: Option<((usize, usize, u8), Mv)>,
    /// This frame's intra-refresh window (`CodecBackend::set_refresh`):
    /// the inter-frame intra search keeps DC on its dirty edge.
    refresh: Option<RefreshWindow>,
    params: Av1Params,
    /// This frame's deblocking levels (`loop_filter_params()`).
    loop_filter: LoopFilterParams,
    /// This frame's `TxMode`: inter frames with `av1.inter_tx_split` signal
    /// transform sizes (inter transform trees, intra `tx_depth`).
    tx_mode: TxMode,
    /// `debug.random_decisions` only (`CodecBackend::enable_random_decisions`):
    /// random syntax choices replace the backend's own.
    random: Option<RandomSyntax>,
    /// `quant.trellis`: trellis quantisation of the coded block.
    trellis: Option<TrellisSettings>,
    /// Film grain signalling (`grain.enable`); `None` when off.
    grain: Option<GrainState>,
    /// Reference currently selected by the engine (`Primary` = `LAST_FRAME`,
    /// `Secondary` = `LAST2_FRAME` on enhancement frames).
    active_ref: ReferenceChoice,
    /// Spatial SVC state; `None` on single-layer streams.
    spatial: Option<Box<BackendSpatial>>,
    /// Loop restoration (`restoration.rs`): the sequence header enables
    /// it, the frame searches it, the place each superblock's unit syntax
    /// goes, and the frame's choice.
    restoration: RestorationState,
}

/// The registered trellis controls (`nc_encoder::QuantParams`).
/// Loop restoration state of the stream and of the frame being coded.
#[derive(Debug, Default)]
struct RestorationState {
    /// `enable_restoration` (sequence header): `av1.restoration` at
    /// configuration, without superres or spatial layers (whose unit grids
    /// the search does not map).
    sequence: bool,
    /// This frame searches restoration (and so reserves unit syntax).
    frame: bool,
    /// Each superblock's reserved unit syntax and its luma position.
    spans: Vec<(SpanSlot, (usize, usize))>,
    /// The frame's restoration, when any plane uses it.
    chosen: Option<FrameRestoration>,
    /// The filter's working buffers, kept across frames.
    scratch: nc_dsp::av1_restoration::Av1LrScratch,
}

/// The CDEF choices made once per frame (`Av1Backend::cdef_frame_choice`).
#[derive(Debug, Clone, Copy)]
struct CdefFrameChoice {
    /// Strengths are chosen per superblock (deferred `cdef_idx` slots).
    per_superblock: bool,
    /// The per-superblock search runs in full instead of reusing earlier
    /// frames' selection.
    search: bool,
}

#[derive(Debug, Clone, Copy)]
struct TrellisSettings {
    /// Multiple of the mode-decision λ.
    lambda_scale: f32,
    /// Further multiple for chroma; `None` leaves chroma unoptimised.
    chroma_scale: Option<f32>,
    /// Whether key frames are trellised too.
    key_frames: bool,
    /// Survivors per scan position.
    max_states: usize,
}

impl TrellisSettings {
    fn from_params(quant: &QuantParams) -> Option<Self> {
        (quant.trellis >= 0.5).then(|| Self {
            lambda_scale: quant.trellis_lambda_scale,
            chroma_scale: (quant.trellis_chroma > 0.0).then_some(quant.trellis_chroma),
            key_frames: quant.trellis_key >= 0.5,
            max_states: quant.trellis_states.round() as usize,
        })
    }
}

/// What the trellis needs to price one transform block's coefficients.
#[derive(Debug, Clone, Copy)]
struct TrellisBlock {
    lambda: f32,
    max_states: usize,
    plane: PlaneType,
    is_inter: bool,
    intra_mode: usize,
    /// Where the block's `TxbContext` comes from: plane, its 4x4 position
    /// and side in the context arrays, and whether it fills its block.
    /// Derived only for blocks the plain quantiser leaves coded.
    context: (usize, usize, usize, usize, bool),
}

nc_encoder::param_group! {
    /// AV1 backend frame, mode-rate, residual and CDEF heuristics.
    pub struct Av1Params ("av1") {
        /// Initial weight is the old H.264-derived0.135 in sample-SSE/bit
        /// units; SVT stage-specific distortion normalization needs separate
        /// auditing before adopting a different model.
        lambda_per_step_squared: 0.135, [0.01, 1.0],
            "bit cost per squared quantizer sample step: higher favours cheaper modes and residual skipping",
        /// Encoder-side forward approximation (SVT's TX shortcut): 32x32
        /// and 64x64 residuals are transformed to their low 16x16
        /// frequencies only. 1 applies it to the RD-cost trials only (the
        /// coded residual stays exact); 2 also to the coded residual, which
        /// failed the worst-1% guard on AV1 Fast (+6.1% balanced, static and
        /// pan worst; preset agent, 150 f). Default off (byte-identical).
        partial_forward: 0.0, [0.0, 2.0],
            "low-frequency forward of 32x32/64x64 residuals: 1 in RD trials only, 2 also in coding",
        /// The same for 16x16 residuals (low 8x8), from this frame q_idx up,
        /// in the scope `partial_forward` sets; 256 never applies it.
        partial_forward16_min_q: 256.0, [0.0, 256.0],
            "frame q_idx from which 16x16 residuals keep only their low 8x8 frequencies (256: never)",
        /// SVT CQP `cqp_qindex_calc` boosts key references by q · (1 − 0.76).
        keyframe_step_permille: 240.0, [100.0, 500.0],
            "key-frame AC step in per mille of the inter step (SVT 240): lower spends more on the reference key frame",
        keyframe_rounding_q7: 48.0, [16.0, 64.0],
            "key-frame coefficient rounding offset in 1/128 step: higher preserves more small coefficients",
        inter_rounding_q7: 48.0, [16.0, 64.0],
            "P-frame coefficient rounding offset in 1/128 step: higher preserves more small coefficients",
        inter_chroma_rounding_q7: 48.0, [16.0, 64.0],
            "P-frame chroma rounding offset in 1/128 step: higher keeps the colour detail VMAF's chroma feature measures",
        base_period: 4.0, [1.0, 16.0],
            "backend base-frame period: higher searches global CDEF and uses the gentler residual margin less often",
        cdef_search_row_step: 4.0, [1.0, 8.0],
            "luma CDEF search row stride: lower samples more distortion rows at extra search cost",
        /// Per-64x64 luma CDEF with the strong candidate (`cdef_local_strong_*`)
        /// is on by default: the QP-model strengths are nearly off at high
        /// quality (Y primary 1, UV 0 at qindex 90), so coding noise built up
        /// over P frames (bodycam_airport/static: -0.5 dB/frame, flat with
        /// strong CDEF, as SVT M12, which drifts the same way with
        /// `--enable-cdef 0`). 6-clip TRAIN screen, PSNR-YUV/VMAF-v1 BD versus
        /// the previous default: -3.52%/-2.43% (homes432 -7.1/-7.4, PP22 low
        /// light -8.2/-7.3; worst bodycam BC1 +0.5/+1.1); callgrind +2.3% to
        /// +5.9%. Evidence `docs/analysis/av1-heldout-gap.md`.
        cdef_per_superblock: 1.0, [0.0, 1.0],
            "CDEF luma off/model selection per 64x64 block: 1 spends filter search and literal bits for local distortion decisions",
        cdef_local_search_period: 1.0, [1.0, 16.0],
            "local CDEF search period: higher reuses the previous block choices between searched frames to save work",
        cdef_local_noise_full_search_sigma: 0.0, [0.0, 32.0],
            "noise sigma at which local CDEF searches every frame despite its period; 0 disables adaptation",
        cdef_local_strong_luma: 1.0, [0.0, 1.0],
            "local CDEF high-variance low-noise P-frame search: 1 tests off/the configured primary at its cadence, retaining QP-model UV",
        cdef_local_strong_primary: 15.0, [0.0, 15.0],
            "primary strength for strong local CDEF trials: lower uses a gentler luma candidate; rounded to an integer",
        /// Optional empirical cap, without another filter pass or palette entry.
        /// TRAIN model primaries homes432/PP22/MEVA/body/pan are1/2/4/5/5;
        /// scale3 retains measured body/pan15 while limiting lower-QP filtering.
        /// Fixed TRAIN scale3 vs15: homes432300f LPIPS -8.86%, V1 +0.116
        /// at +2.48% rate; PP22/MEVA V1 -0.225/-0.276 at +1.95/+0.78% rate.
        /// Body241/pan238 packets/recons unchanged. Single-QP, not BD.
        /// No extra filtering passes;0 preserves the fixed-primary policy.
        cdef_local_strong_model_scale: 0.0, [0.0, 16.0],
            "cap strong primary by rounded modeled primary times this scale: 0 retains configured fixed primary; secondary stays0",
        /// Coded secondary strength of the strong candidate (SVT level 7 adds
        /// a secondary pass). Default 2. Staged TRAIN screens, PSNR-YUV/VMAF-v1
        /// BD: 1 vs 0 -1.89%/-1.45% (full 15-clip TRAIN proxy); 2 vs 1
        /// -0.97%/-0.53% (full set). Evidence
        /// `work/av1-heldout/cdefsearch/{sec1,sec2v1}.{txt,json}`.
        cdef_local_strong_secondary: 2.0, [0.0, 3.0],
            "coded secondary strength (0..3, 3 means 4) of the strong local CDEF candidate: 0 keeps primary only; rounded to an integer",
        /// SVT level 7 searches chroma strengths where the QP model is off at
        /// high quality; 0 keeps the QP model.
        cdef_uv_search_primary: 0.0, [0.0, 15.0],
            "on searched per-superblock CDEF frames, choose chroma among off, the QP model and this primary by sampled U+V error: 0 keeps the QP model; rounded",
        /// 0 by default: the former 350 gate excluded clean, low-variance
        /// content (bodycam_airport), exactly where the drift is worst.
        cdef_local_strong_variance_min: 0.0, [0.0, 16384.0],
            "minimum sampled source picture variance for strong local CDEF: higher restricts the strong candidate to more textured frames",
        keyframe_cfl: 0.0, [0.0, 1.0],
            "key-frame chroma from reconstructed luma: 1 searches signed slopes on blocks up to 32; extra key-frame work for chroma quality",
        /// TRAIN6 150f calibrated QPs vs 0: PSNR-YUV/V1 BD −0.01%/+0.26%
        /// (all within ±0.8%); high-quality grid (pp22_004) −0.57%/−0.52%.
        /// A consistency fix: 0 sent key chroma to qindex 0 at inter q108.
        keyframe_chroma_scaled: 1.0, [0.0, 1.0],
            "key-frame chroma offset: 1 keeps the inter chroma/luma step ratio on key frames (fixed QP and rate control), 0 subtracts the inter qindex offset from the key qindex (much finer key chroma)",
        /// 2 vs 0 (transform-RD mode ranking), staged screen, full TRAIN15
        /// 60f: PSNR-YUV/V1 BD −1.28%/−2.19%, 13/15 clips better on both;
        /// +172M instructions per 720p key frame at q200 (key ~1.9x).
        keyframe_intra_modes: 2.0, [0.0, 2.0],
            "key-frame intra set: 0 DC only, 1 smooth/Paeth, 2 adds vertical/horizontal; higher spends more mode-search effort",
        /// The fuller key-frame intra set (SVT M8-M10): the extra candidates
        /// below are ranked by prediction SATD plus √λ · mode bits, and the
        /// best `keyframe_intra_rd_candidates` of them join the
        /// `keyframe_intra_modes` set in the transform-cost ranking.
        keyframe_directional: 0.0, [0.0, 1.0],
            "key-frame oblique directional modes: 1 adds D45/D67/D113/D135/D157/D203 as SATD-ranked candidates (AV1 §7.11.2.4)",
        keyframe_angle_deltas: 0.0, [0.0, 2.0],
            "key-frame angle deltas around the best SATD-ranked direction: 0 off, 1 ±2 then ±1/±3 beside the better side, 2 all six; rounded",
        /// Also sets `enable_filter_intra` in the sequence header, so every
        /// DC intra block up to 32x32 (key or inter frame) codes
        /// `use_filter_intra`.
        keyframe_filter_intra: 0.0, [0.0, 1.0],
            "key-frame filter intra: 1 enables the tool in the sequence header and adds its five modes as SATD-ranked candidates on blocks up to 32x32 (AV1 §7.11.2.3)",
        keyframe_directions: 8.0, [1.0, 8.0],
            "key-frame directions predicted for the SATD ranking: the likeliest this many by gradient orientation (8: all allowed); rounded, lower is faster",
        keyframe_intra_rd_candidates: 3.0, [1.0, 16.0],
            "SATD-ranked extra key-frame intra candidates (directions, deltas, filter intra) that are also transform-costed; rounded, higher searches more",
        intra_rate_scale: 1.0, [0.25, 4.0],
            "intra mode-bit cost relative to prediction distortion: higher favours cheaper prediction modes",
        inter_intra_search: 0.0, [0.0, 2.0],
            "inter-frame intra luma modes (`p_intra.rs`): 0 DC only, 1 a short SATD-ranked, transform-costed search (and the inter fallback) for blocks the engine's directional estimate sent to intra, 2 the search for every inter-frame intra block; rounded",
        inter_intra_rd_candidates: 2.0, [1.0, 7.0],
            "SATD-ranked inter-frame intra candidates (DC, V, H, D45, D135, smooth, Paeth) that are also transform-costed; rounded, higher searches more",
        transform_rd_skip: 0.0, [0.0, 2.0],
            "reject near-MV luma before inverse using transform gain: 0 off, 1 current CDF rate, 2 SVT EOB proxy",
        transform_rd_skip_chroma: 0.0, [0.0, 1.0],
            "on transform-domain luma rejection, retain UV prediction and bypass UV residual work; ordinary zero luma is unaffected",
        transform_rd_uv_max_noise_sigma: 255.0, [0.0, 255.0],
            "allow transform-RD UV bypass only at or below this analyzed noise sigma; 255 keeps the unrestricted policy",
        transform_rd_gain_scale: 1.0, [0.25, 2.0],
            "estimated spatial distortion gain multiplier before inverse: higher retains more luma residuals",
        transform_rd_rate_scale: 1.0, [0.0, 4.0],
            "coefficient-rate multiplier in pre-inverse luma RD: higher rejects more residuals",
        /// On by default since 2026-11-08 with score 70, energy 15 and RD
        /// 50 (Fast sets SVT's 70/30/100 explicitly). Proxy-6 (TRAIN, 6
        /// clips, 150f) vs off: PSNR-YUV/V1 BD +0.82%/+0.86%, worst clip
        /// bodycam_antare +2.10%/+0.34% (SVT's energy 30 / RD 100 lost
        /// +10.3% PSNR there: low-energy texture on clean video); host P-frame
        /// instructions crowd/bodycam −4.1%/−6.9% (J = dBD + 0.6 dCPU ≈ −2.5).
        pre_tx_skip: 1.0, [0.0, 1.0],
            "enable SVT light-PD1 luma pre-transform skip using prediction SSE and current syntax costs",
        pre_tx_skip_energy: 15.0, [0.0, 100.0],
            "pre-transform energy threshold: 100*SSE < energy*area*qindex; higher skips more",
        pre_tx_skip_score: 70.0, [0.0, 250.0],
            "base/inherited pre-transform skip score (0 disables this score); positive nonbase override takes precedence; lower skips more",
        // TRAIN five-class70base/50nonbase vs uniform70: V1 BD -0.209%,
        // PSNR BD +1.408%. Host GOP300 instructions body/crowd -2.00/-2.35%;
        // Jetson150f three-pair median CPU -1.24/-2.18%. Optional tradeoff only.
        pre_tx_skip_score_nonbase: 0.0, [0.0, 250.0],
            "when pre_tx_skip is enabled, positive score overrides nonbase frames only; 0 inherits pre_tx_skip_score (fractional values preserved)",
        pre_tx_skip_rd: 50.0, [0.0, 400.0],
            "pre-transform RD threshold: 100*skip_cost < threshold*reference_cost; 0 disables RD signal",
        /// 1 replaces the RD signal's fixed guess at the coded alternative
        /// (distortion 1 per sample, 11.7 bits) with the shared quantiser
        /// test `nc_encoder::residual_quantises_to_zero` (Default since
        /// 2026-11-08; Fast keeps 0). Proxy-6 (TRAIN, 6 clips, 150f), model
        /// 1 at texture scale 8 vs 0: PSNR-YUV/V1 BD +0.54%/+0.61%, worst
        /// bodycam_antare +2.69%/+1.33%; host P-frame instructions
        /// crowd/bodycam −2.3%/−5.2% (J ≈ −1.7). Antare at scale 4/8/16:
        /// PSNR BD +0.90/+2.69/+3.45% for −0.8/−3.8/−6.3% CPU.
        pre_tx_skip_model: 1.0, [0.0, 1.0],
            "pre-transform RD signal: 0 prices coding with a fixed guess (SVT), 1 tests whether the residual would quantise to zero",
        pre_tx_texture_scale: 8.0, [1.0, 16.0],
            "zero-residual test (pre_tx_skip_model 1): AC energy allowance in squared zero limits; higher skips more texture",
        // Static-content Fast lever (static-detect, 2026-11-09; default off,
        // screen in docs/handoff/static-detect.md): see
        // `nc_encoder::still_skip`.
        still_skip: 0.0, [0.0, 1.0],
            "1 codes blocks of at least 16x16 that the stillness detector marks still with the zero vector and no residual, without mode decision or transforms, when every plane's zero-vector residual would quantise to zero",
        still_skip_limit_scale: 1.0, [1.0, 8.0],
            "still_skip's zero-residual test: multiple of the quantiser's zero limits (DC and AC; 1 is nearly a proof that nothing would be coded): higher skips more",
        pre_tx_luma_dominant: 1.0, [0.0, 1.0],
            "conservative low-qindex protection: 1 treats every input as luma-dominant; 0 uses SVT nondominant penalties",
        shave_min_zero_run: 8.0, [0.0, 32.0],
            "minimum zeros before shaving a trailing +/-1 coefficient, rounded to an integer: 0 shaves contiguous ones (including DC), higher preserves more coefficients",
        chroma_skip_sad_num: 5.0, [0.0, 32.0],
            "inter-chroma sampled-SAD threshold numerator over 8, rounded to an integer: 0 disables skipping, higher skips more transforms",
        residual_rd_max_mv: 8.0, [0.0, 64.0],
            "luma residual RD check only where |vector| ≤ this (1/8 samples): higher checks more blocks",
        skip_bias_pct: 60.0, [40.0, 100.0],
            "biased skip on low-energy blocks: drop the residual if pct% of the skip distortion is below the coded RD cost (SVT lpd1_blk_skip_luma_rd); lower skips more",
        skip_bias_pct_base: 60.0, [40.0, 100.0],
            "skip_bias_pct on every 4th frame (SVT flat-IPP pseudo-layer base frames, gentler light-PD1): higher codes more residual there",
        skip_bias_energy: 0.0625, [0.0, 1.0],
            "biased skip applies when the prediction error per sample is below this × λ: higher applies it to more blocks",
        // Default -32 (was -24): staged TRAIN proxy-6 (150f, AV1 Fast)
        // PSNR-YUV / VMAF v1 BD -2.10% / -1.50% (static_cam22 V1 +1.33%,
        // others better); host instructions +0.36% (crowd09 q180): J = -1.58.
        chroma_delta_q: -32.0, [-64.0, 63.0],
            "chroma quantiser offset in qindex units (DeltaQUDc = DeltaQUAc): higher codes chroma more coarsely",
        chroma_noise_scale: 2.0, [0.0, 32.0],
            "attenuate chroma_delta_q linearly to zero at this analyzed noise sigma: 0 disables adaptation; fractional values allowed",
        // Default 1 with energy 0.25 (quality point, PROCEDURES "Preset
        // roles", k 0.1), after the RD costing fix (b41273d; the first
        // enabling grew homes390 P frames 4.45x). Proxy-6 + homes390 150f:
        // PSNR-YUV / VMAF v1 -1.38% / -0.90%, every clip <= +0.30%; P
        // instructions +10.3% (q180) / +5.0% (q220). AV1 Fast keeps 0.
        inter_tx_split: 1.0, [0.0, 1.0],
            "inter transform split: 1 codes inter frames with TX_MODE_SELECT and splits an inter block's transform into four where an RD estimate prefers it (AV1 §5.11.17)",
        /// See `Av1Backend::choose_tx_split`.
        ceiling_tx_split: 2.0, [0.0, 64.0],
            "code a 64x64 inter block as four 32x32 transforms where rate control plans at least this multiple of the λ at qindex 255 (0 = never): a 64x64 transform cannot quantise coarser; lower splits earlier past the ceiling, fewer stale skipped patches at starved rates",
        inter_tx_split_energy: 0.25, [0.0, 4.0],
            "skip the transform-split trial where the prediction error per sample is below this × λ: higher is faster, splitting fewer weak residuals",
        inter_tx_split_concentration: 0.0, [0.0, 4.0],
            "try the transform split only where the largest quarter's prediction error exceeds this multiple of the quarters' mean (0 tries every block): higher is faster",
        // Default 1 (was 0, rejected on VMAF v1 alone): J_real staged TRAIN
        // (18 clips, real-use CBR, PSNR-YUV / VMAF NEG / MS-SSIM BD) -6.8% /
        // -1.2% / -6.9%, quality -5.1%, 0/18 clips worse, no guard; VMAF v1
        // +0.2% is what rejected it before. Callgrind +4.5% (crowd09) /
        // +10.5% (bodycam_airport) at q200 after the SIMD kernel, so J_real
        // -3.6% at the Default's k 0.2. VAL (11 clips, full gate) -6.6%,
        // 0/11 worse; long-GOP single-key gain holds over 900 frames. Fast
        // pins 0 (+9.0% there, J ≈ 0 at k 0.6). Evidence
        // `docs/handoff/av1.md` (J_real recheck).
        loop_filter: 1.0, [0.0, 1.0],
            "deblocking loop filter: 1 filters block edges at libaom's pick-from-Q level before CDEF (AV1 §7.14)",
        loop_filter_level_scale: 1.0, [0.0, 2.0],
            "multiplier on the pick-from-Q deblocking level: lower smooths block edges less (keeps more texture)",
        loop_filter_sharpness: 0.0, [0.0, 7.0],
            "deblocking sharpness (rounded): higher lowers the inner-edge limit, filtering textured edges less",
        near_count: 3.0, [0.0, 3.0],
            "NEARMV list positions offered to mode decision (SVT near_count; AV1 allows 3): higher tries more vectors",
        /// SVT M11+ sets `disable_cdf_update` on 7 of 8 real-time frames
        /// (`enc_mode_config.c`, `rtc` CDF control); a Fast preset applies it
        /// to non-base frames with `nonbase.av1.cdf_update = 0`.
        /// SVT M11+ prices coefficients in mode decision with a linear
        /// end-of-block model (`coeff_rate_est_lvl` 0); here it replaces the
        /// live-CDF walk of the residual-drop check
        /// (`drop_unprofitable_luma_residual`), the one P-frame decision
        /// that walks the coefficient CDFs in the Fast preset.
        coeff_rate_linear: 0.0, [0.0, 1.0],
            "1 prices the P-frame residual-drop check's coefficients with SVT's linear end-of-block estimate instead of the live CDFs: faster, coarser rate",
        cdf_update: 1.0, [0.0, 1.0],
            "1 adapts every CDF per coded symbol within the frame; 0 codes the frame with disable_cdf_update (AV1 §5.9.2): symbols are priced and coded with the CDFs the frame loaded, which it passes on unchanged (faster encode and decode, less adapted probabilities)",
        /// Frame ids (`obu.rs`): off by default, so streams without loss
        /// resilience pay nothing; [`configured_frame_ids`] also turns them
        /// on with a long-term reference.
        frame_ids: 0.0, [0.0, 1.0],
            "1 codes frame ids (frame_id_numbers_present_flag, AV1 §5.9.2) so a decoder can tell a lost reference from the stream alone (receivers without the RTP dependency descriptor); on anyway with ltr.interval; costs idLen + 7 delta lengths per inter frame (23 bits flat or layered, up to 65 with a long-term reference)",
        /// Super-resolution (`superres.rs`): fewer coded samples per frame
        /// at starved bitrates, upscaled normatively by the decoder.
        superres_denom: 8.0, [8.0, 16.0],
            "AV1 superres (§7.16): every frame is coded 8/denom as wide as the stream and upscaled horizontally by the decoder; 8 is off, 9..16 code narrower frames (the coded width must come out even)",
        /// Loop restoration (`restoration.rs`): sets `enable_restoration`
        /// at configuration (not with superres); a frame override of 0
        /// skips the search on those frames. Off: on the 2026-11-12 pool
        /// levels (5-tap luma, chroma, 256 units, row step 4, no refine)
        /// J +0.37% [-0.48, +1.25] at L3 (PSNR-YUV better, VMAF v1 and
        /// MS-SSIM worse) for +9% encode and +13% decode instructions
        /// (`docs/handoff/av1.md`, loop restoration).
        restoration: 0.0, [0.0, 1.0],
            "AV1 loop restoration (§7.17): 1 fits a least-squares Wiener filter per restoration unit after CDEF and codes it where its error saving pays for its bits",
        restoration_luma_taps: 7.0, [3.0, 7.0],
            "Wiener taps of luma restoration filters (3, 5 or 7; chroma at most 5): fewer taps cut the search's statistics roughly quadratically and fit less",
        restoration_chroma: 1.0, [0.0, 1.0],
            "1 also restores the chroma planes (SVT-AV1's fast levels restore luma only)",
        restoration_unit_size: 0.0, [0.0, 256.0],
            "restoration unit size: 64, 128 or 256 luma samples; anything else chooses among them per frame by RD cost",
        restoration_row_step: 2.0, [1.0, 8.0],
            "restoration statistics take every Nth row: higher searches faster on a sparser sample",
        restoration_refine: 1.0, [0.0, 4.0],
            "passes of ±1 refinement of the quantized Wiener taps (libaom finer_tile_search_wiener); 0 keeps the least-squares rounding",
        restoration_lambda_scale: 1.0, [0.25, 4.0],
            "multiplier on the frame λ when restoration units weigh their tap bits against squared error: higher restores fewer units",
        restoration_top_layer: 1.0, [0.0, 1.0],
            "0 skips restoration on the top temporal layer of layered streams (frames nothing predicts from), as SVT-AV1 does",
        /// Compound dual-reference inter prediction on frames with a secondary reference.
        compound: 1.0, [0.0, 1.0],
            "1 enables compound single/dual-reference prediction (reference_select = 1) on inter frames that carry a secondary reference (AV1 §5.9.23, §5.11.25); 0 restricts blocks to single-reference prediction",
    }
}

/// The superres `av1.superres_denom` asks for (`superres.rs`, rounded to
/// a whole denominator), if any.
///
/// # Errors
/// [`ConfigError::Resolution`] when the coded width comes out odd (4:2:0
/// coding needs even widths) or with film grain (whose synthesis would run
/// at the upscaled size: not supported yet).
fn configured_superres(
    params: &Av1Params,
    config: &EncoderConfig,
) -> Result<Option<Superres>, ConfigError> {
    let denom = (params.superres_denom.round() as usize).clamp(SUPERRES_NUM, SUPERRES_DENOM_MAX);
    let Some(superres) = Superres::new(denom, config.width, config.height) else {
        return Ok(None);
    };
    if !downscaled_width(config.width, denom).is_multiple_of(2) {
        return Err(ConfigError::Resolution(
            "av1.superres_denom gives an odd coded width for this stream width",
        ));
    }
    if GrainParams::from_overrides(&config.params).enable >= 0.5 {
        return Err(ConfigError::Resolution(
            "av1.superres_denom with film grain is not supported",
        ));
    }
    Ok(Some(superres))
}

/// `av1.base_period`, checked by `new`: the base cadence decides which
/// frames are base frames.
const BASE_PERIOD_PARAM: &str = "av1.base_period";

/// `av1.frame_ids`: code frame ids without a long-term reference (a
/// receiver that sends loss feedback but has no RTP dependency descriptor).
pub const FRAME_IDS_PARAM: &str = "av1.frame_ids";

/// The frame ids of a stream with `config`: when `av1.frame_ids` asks for
/// them or a long-term reference is in use (loss resilience, where a
/// browser receiver must tell a lost reference from the stream), with
/// fields as short as the stream's farthest reference allows.
fn configured_frame_ids(params: &Av1Params, config: &EncoderConfig) -> Option<FrameIds> {
    (params.frame_ids >= 0.5 || nc_encoder::long_term_requested(config)).then(|| {
        FrameIds::new(FrameIdLengths::for_reach(nc_encoder::reference_reach(
            config,
        )))
    })
}

impl CodecBackend for Av1Backend {
    // λ, the zero-residual limits and the luma residual RD check follow the
    // planned step. CBR 150 f, crowd_mot20_06 at 150 kbps: 273 -> 159 kbps.
    const SUPPORTS_LAMBDA_HEADROOM: bool = true;
    // Superblock `delta_qindex` (AV1 §5.11.12, `delta_q`).
    const SUPPORTS_BLOCK_QP: bool = true;
    const BLOCK_QP_LOG2: u8 = SUPERBLOCK_LOG2;

    const PARAM_SPECS: &'static [&'static [nc_encoder::ParamSpec]] =
        &[Av1Params::SPECS, GrainParams::SPECS];
    /// Every `av1.*` parameter is read per frame (`set_frame_params`) except
    /// `av1.base_period`, which `new` rejects as a non-base setting, and
    /// the stream-fixed ones below.
    const FRAME_PARAM_SPECS: &'static [&'static [nc_encoder::ParamSpec]] = &[Av1Params::SPECS];
    /// Frame ids are sequence-header syntax.
    const STREAM_FIXED_PARAMS: &'static [&'static str] = &[FRAME_IDS_PARAM];
    /// `begin_frame` starts from committed state (slot CDFs, CDEF memory,
    /// the previous frame's kind); `commit_frame` advances it.
    const RECODE: bool = true;
    const REPORTS_CODED_BITS: bool = true;

    /// The tile's bits so far (the frame header is written at
    /// `end_frame` and not counted).
    fn coded_bits(&mut self) -> Option<u64> {
        Some(self.writer.coded_bits())
    }

    /// The DC-aware intra test stays off for AV1: on VAL-150 (full gate) it
    /// loses VMAF v1 although PSNR gains. With every block also trying intra
    /// (gate 0): PSNR-YUV -8.20%, VMAF v1 +6.06% on val-proxy-5 (airport
    /// +24.5% V1); with the default gate: -0.92% / +0.26% on all 11 natural
    /// clips. H.264 and HEVC keep it (see docs/handoff/hevc.md).
    ///
    /// The motion-decision values below are now the engine's defaults
    /// (`nc_encoder` `DecisionParams`); their AV1 measurements stay here.
    ///
    /// NEW sub-sample refinement is skipped when the search seed already
    /// predicts within 0.03 λ per sample (a speed heuristic: it saves the
    /// bilinear refinement evaluations on blocks whose residual is already
    /// negligible). Callgrind 31 f: -7.0% bodycam BC1, -4.0% crowd09
    /// instructions; staged TRAIN screen proxy-6 PSNR-YUV/VMAF-v1 BD
    /// +0.59%/+0.52%, every class within +1.2% (0.1 saved -9.6% but cost
    /// crowd/pan over +3%). Evidence `work/av1-heldout/cpu/seed003.txt`.
    ///
    /// Default is the quality point of the preset pair (docs/PROCEDURES.md,
    /// "Preset roles"); AV1 Fast pins the engine values back. Measured
    /// alone on proxy-6 (150f, PSNR-YUV / VMAF v1 BD, P instructions
    /// crowd09 + bc1 at q180):
    /// - sub-pel move margins 1 (no 10% margin for half/quarter moves):
    ///   -1.17% / -1.03%, +3.4%;
    /// - the four-neighbour integer probe, ungated: -1.76% / -2.10%, +2.6%;
    /// - refining the best cheap (NEAREST/NEAR) vector as a NEW candidate,
    ///   gate 0.05 λ: -1.41% / -0.82%, +10.1%.
    ///
    /// `planner.depth_removal_min_quantizer` 0.8: no depth removal at
    /// `base_q_idx` <= 204, so fine texture can split below 32x32.
    /// `J_real` TRAIN (18 clips incl. texture): BETTER, quality -3.5% full /
    /// -1.5% low, no clip worse on the full curve, no guards, for +4.6%
    /// (crowd09) / +9.7% (bc1) instructions at q160. Fast keeps 0.
    ///
    /// Trellis quantisation (`quant.*`, `docs/handoff/trellis.md`) on
    /// P frames: λ×0.6, one survivor per position, chroma at 0.25. `J_real`
    /// staged TRAIN (18 clips) BETTER: quality −2.2% full / −2.2% low, every
    /// class and every metric better (PSNR-YUV −2.4%, NEG −2.2%, MS-SSIM
    /// −2.1%, VMAF v1 −2.5%), worst clip `bodycam_bc1` +1.3%, no guard, for
    /// +15% P-frame instructions (q180, key frames unchanged): J −0.7 at
    /// k 0.1. λ×1 with two survivors lost on TRAIN (bodycam NEG +5.4%,
    /// J +0.3) and drifted on long GOPs (airport −0.15 dB Y by the last
    /// third); key frames trellised tripped the static `quality_low` guard.
    /// Single-key long GOP vs off at equal rate: airport 600 kbps YUV
    /// +0.09/+0.07/+0.02 dB per third, chirla 150 kbps −0.01/+0.05/+0.07,
    /// homes390 300 kbps −0.04/−0.05/−0.06. Fast keeps the trellis off.
    ///
    /// Key-frame intra set (`av1.keyframe_*`, SVT M8-M10 key frames;
    /// `docs/handoff/av1.md`): oblique directions, coarse-to-fine angle
    /// deltas and filter intra, the best 3 SATD-ranked extras transform
    /// costed. Filter intra also adds a `use_filter_intra` flag to P-frame
    /// DC blocks. `J_real` staged TRAIN (18 clips) BETTER: quality −0.6%
    /// full / −0.4% low, static −1.3%, worst clip `bodycam_bc1` (low +4.0%),
    /// no guard, periodic-key pulse +0.18 dB: J −0.3 at k 0.1 with dCPU
    /// 2.5%. Key/P instructions (q200, x86) 8.3x → 13.3x crowd09, 7.7x →
    /// 12.8x bc1, 12.0x → 20.8x airport; Jetson CPU 6.6x → 9.8x crowd09,
    /// 8.7x → 13.3x airport (the Default's key/P guard is ~15x, Fast's
    /// ~10x). Fast keeps all three off.
    const DEFAULT_PARAMS: &'static [(&'static str, f32)] = &[
        // A coarser inter rounding offset (48 -> 41, AV1 Fast's value): a CPU
        // cut at equal quality (study av1-default-md-2026-11-13 trial #1,
        // mode-decision levers only). Head 0cdb619d (rate.debit), pool
        // levels of rotation 2026-11-13: L1 -0.14%, L2 (ladder) -0.09%,
        // disjoint L3 (79 clips, single-point) J +0.16% [-0.17, +0.50], no
        // guard; x86 instructions -4.2%, Jetson cycles -3.1%. av1-svc and
        // av1-svc3 pin 48.
        ("av1.inter_rounding_q7", 41.0),
        // Each sub-sample seed refined only when its whole-sample distortion
        // is within 1.1x the other's (`decision.seed_refine_ratio`, the
        // engine's seed gate from vvc-default-prune): a CPU cut at equal
        // quality. Head 91c6823e, pool levels of rotation 2026-11-14:
        // disjoint L3 (79 clips, single-point) J +0.07% [-0.23, +0.38],
        // pooled +0.03%, no guard; x86 instructions -2.9%, Jetson cycles
        // -2.4%. The other AV1 presets pin 0 (both seeds refined).
        ("decision.seed_refine_ratio", 1.1),
        // Start-up QP cadence (period 8, base frames 4 QP finer, for the
        // first 100 frames after a key, then the engine's period 4 / -3 / +1)
        // and the cadence motion gate (`qp_cadence.rs`; docs/handoff/vvc.md,
        // qp-cascade). J_real (k 0.1, dCPU 0): boost 6 proxy-4 WORSE (crowd
        // worst-1% +8..+15%); boost 4 TRAIN (18 clips) BETTER -2.6% (static
        // -5.8%, lowlight -6.1%, crowd -3.1%), worst-1% Y -0.4%, pulse -0.12
        // dB; VAL BETTER -6.5% (static_otw_homes390 -25%, crowd -6.9%,
        // bodycam +1.2%), pulse -0.08 dB. Long GOP (900 f, one key, CBR
        // 300): chirla Y 37.46 -> 37.58, pan_bdd 30.96 -> 30.98.
        ("rate.cadence_startup_period", 8.0),
        ("rate.cadence_startup_base_boost", 4.0),
        ("rate.cadence_motion_gate", 1.0),
        ("decision.intra_test_dc", 0.0),
        ("planner.depth_removal_min_quantizer", 0.8),
        ("quant.trellis", 1.0),
        ("quant.trellis_key", 0.0),
        ("quant.trellis_states", 1.0),
        ("quant.trellis_lambda_scale", 0.6),
        ("av1.keyframe_directional", 1.0),
        ("av1.keyframe_angle_deltas", 1.0),
        ("av1.keyframe_filter_intra", 1.0),
        // Directional P-frame intra estimate (`decision.intra_directional`,
        // engine `intra_estimate.rs`) with the inter-frame intra search and
        // fallback for the blocks it sends to intra (`av1.inter_intra_search`
        // 1, `p_intra.rs`): inter frames used to code intra DC only. J_real
        // (k 0.1, dCPU 1.1): TRAIN (18 clips) BETTER -0.2%, quality -0.3%
        // full / -0.3% low (PSNR -0.3%, NEG -0.3%, VMAF v1 -0.3%), 5/18
        // clips worse (at most +0.3%), pulse -0.02 dB; VAL (11 clips)
        // BETTER -0.2%, quality -0.3% / -0.4%, pulse +0.01 dB. Fixed QP 60 f:
        // Riverbed -4.0% PSNR-YUV / -1.4% VMAF. Cost-clip P frames +1.1%,
        // key 0. Single-key 600 f CBR (chirla, crowd_mot20, pan_bdd): no
        // drift. av1-fast keeps it off (not screened at k 0.6).
        ("decision.intra_directional", 1.0),
        ("av1.inter_intra_search", 1.0),
        // Adaptive coded resolution (`nc-encoder` `resolution.rs`): half
        // size per dimension, no key frame, while rate control plans past
        // qindex 255. 720p CBR 100 kbps, 300 f, scored at 720p as displayed
        // (PSNR-Y / MS-SSIM): CrowdRun 20.40 / 0.622 -> 20.50 / 0.636,
        // ParkJoy 19.86 / 0.612 -> 20.02 / 0.625 (VMAF v1 5.4 -> 2.3, both
        // near zero), FoodMarket unswitched; at 300 kbps nothing switches
        // (streams identical). docs/handoff/adaptive-resolution.md.
        ("resolution.adaptive", 1.0),
    ];

    fn new(config: &EncoderConfig) -> Result<Self, ConfigError> {
        if !(QINDEX_MIN..=QINDEX_MAX).contains(&config.quantizer) {
            return Err(ConfigError::Quantizer {
                value: config.quantizer,
                min: QINDEX_MIN,
                max: QINDEX_MAX,
            });
        }
        if !(MIN_BLOCK_LOG2..=MAX_BLOCK_LOG2).contains(&config.block_log2) {
            return Err(ConfigError::BlockSize(config.block_log2));
        }
        let params = Av1Params::from_overrides(&config.params);
        // The backend's base cadence decides which frames are base, so it
        // cannot itself change on non-base frames.
        let (nonbase, _) = config.params.split_prefix(nc_encoder::NONBASE_PREFIX);
        if nonbase.names().any(|name| name == BASE_PERIOD_PARAM) {
            return Err(ConfigError::Param(nc_encoder::ParamError::NotPerFrame(
                format!("{}{BASE_PERIOD_PARAM}", nc_encoder::NONBASE_PREFIX),
            )));
        }
        let base_period = (params.base_period.round() as u64).max(1);
        let superres = configured_superres(&params, config)?;
        let restoration_sequence = params.restoration >= 0.5 && superres.is_none();
        let width = superres
            .as_ref()
            .map_or(config.width, |s| downscaled_width(config.width, s.denom()));
        Ok(Self {
            lambda_scale: 1.0,
            grids: LayerGrids::new(width, config.height),
            stream_size: (config.width, config.height),
            superres,
            store: ReferenceStore::default(),
            reference_held: false,
            steps: QuantSteps::for_qindex(config.quantizer),
            chroma_steps: QuantSteps::for_qindex(config.quantizer),
            cdfs: Box::new(CdfContext::new(coeff_cdf_q_ctx(config.quantizer))),
            slot_cdfs: std::array::from_fn(|_| None),
            references: FrameReferences::FLAT,
            frame_ids: configured_frame_ids(&params, config),
            temporal_layers: config.temporal_layers,
            signal: config.signal,
            frame_rate: config.frame_rate,
            quantizer_mode: QuantizerMode::Fixed,
            base_q_idx: config.quantizer,
            frame_q_idx: config.quantizer,
            block_q_idx: config.quantizer,
            delta_q: DeltaQ::default(),
            frame_chroma_delta_q: noise_adapted_chroma_delta(
                params.chroma_delta_q,
                0.0,
                params.chroma_noise_scale,
            ),
            rounding_q7: params.keyframe_rounding_q7.round() as i32,
            chroma_rounding_q7: params.keyframe_rounding_q7.round() as i32,
            base_period,
            is_base_frame: true,
            pre_tx_score_this_frame: params.pre_tx_skip_score,
            cdef: CdefParams::default(),
            cdef_input: CdefSource::default(),
            cdef_alternate: None,
            cdef_frame: CdefFrameChoice {
                per_superblock: false,
                search: true,
            },
            noise_sigma: 0.0,
            chroma_policy: nc_encoder::chroma_policy::ChromaPolicy::from_params(&config.params),
            kind: FrameKind::Key,
            current: None,
            writer: SymbolWriter::new(),
            dsp: Dsp::new(),
            scratch: Scratch::new(),
            still_skipped: None,
            luma_copy: None,
            intra_fallback: None,
            refresh: None,
            shave_min_zero_run: params.shave_min_zero_run.round() as usize,
            chroma_skip_sad_num: params.chroma_skip_sad_num.round() as u32,
            params,
            loop_filter: LoopFilterParams::OFF,
            tx_mode: TxMode::Largest,
            random: None,
            trellis: TrellisSettings::from_params(&QuantParams::from_overrides(&config.params)),
            grain: GrainState::new(GrainParams::from_overrides(&config.params)),
            active_ref: ReferenceChoice::Primary,
            spatial: None,
            restoration: RestorationState {
                sequence: restoration_sequence,
                ..RestorationState::default()
            },
        })
    }

    /// Deblocking modifies up to 6 samples (13-tap filter, AV1 §7.14.6.4);
    /// CDEF then picks each 8x8 block's direction from all its samples and
    /// reads 2 more beyond it (§7.15.2), so influence spreads about 11
    /// samples; 16 with margin.
    fn loop_filter_reach(&self) -> usize {
        AV1_LOOP_FILTER_REACH
    }

    fn superblock_log2(&self) -> u8 {
        SUPERBLOCK_LOG2
    }

    fn min_block_log2(&self) -> u8 {
        MIN_BLOCK_LOG2
    }

    fn supports_inter(&self) -> bool {
        true
    }

    fn coded_size(&self, width: usize, height: usize) -> (usize, usize) {
        coded_picture_size(width, height)
    }

    fn frame_size(&self, width: usize, height: usize) -> (usize, usize) {
        match &self.superres {
            Some(superres) => (downscaled_width(width, superres.denom()), height),
            None => (width, height),
        }
    }

    fn output_picture(&self) -> Option<&Picture> {
        self.superres.as_ref().and_then(Superres::output)
    }

    fn supports_scaled_references(&self) -> bool {
        self.superres.is_none()
    }

    /// The new size takes effect from the next frame (§5.9.5 – §5.9.7
    /// `frame_size_override_flag`); the old pictures stay the references
    /// until refreshed.
    fn set_frame_size(
        &mut self,
        width: usize,
        height: usize,
        references: &[Option<&Picture>],
    ) -> Result<(), ConfigError> {
        if self.superres.is_some() {
            return Err(ConfigError::Resolution(
                "not with av1.superres_denom (references are upscaled)",
            ));
        }
        if (width, height) == (self.grids.width, self.grids.height) {
            return Ok(());
        }
        let (stream_width, stream_height) = self.stream_size;
        if width > stream_width || height > stream_height {
            return Err(ConfigError::Resolution(
                "a frame cannot be larger than the sequence header's maximum",
            ));
        }
        let current = (self.grids.width, self.grids.height);
        let scalable = references
            .iter()
            .enumerate()
            .filter(|(_, picture)| picture.is_some())
            .map(|(slot, _)| self.store.held(slot).map_or(current, HeldReference::size))
            .all(|reference| reference_scale_allowed(reference, (width, height)));
        if !scalable {
            return Err(ConfigError::Resolution(
                "AV1 scales a reference at most 2x down and 16x up (§7.11.3.3)",
            ));
        }
        self.store.hold_engine_pictures(references, current);
        let (coded_width, coded_height) = coded_picture_size(width, height);
        let mi_cols = coded_width >> MI_LOG2;
        self.grids.width = width;
        self.grids.height = height;
        self.grids.mode_info = ModeInfoGrid::new(mi_cols, coded_height >> MI_LOG2);
        self.grids.contexts = CoeffContexts::new(mi_cols);
        self.grids.txfm = TxfmContexts::new(2 * width.div_ceil(8));
        self.grids.cdef_slots = cdef_slots(coded_width, coded_height);
        // The remembered per-filter-block CDEF choices index the old grid.
        self.grids.cdef_memory = CdefMemory::default();
        self.grids.committed_cdef_memory = CdefMemory::default();
        Ok(())
    }

    fn supports_temporal_layers(&self) -> bool {
        true
    }

    fn supports_second_reference(&self) -> bool {
        self.spatial.is_some()
    }

    fn supports_compound_reference(&self) -> bool {
        self.frame_reference_select()
    }

    fn select_reference(&mut self, _block: &Block, reference: ReferenceChoice) {
        self.active_ref = reference;
    }

    fn supports_spatial_layers(&self) -> bool {
        true
    }

    fn max_spatial_layers(&self) -> u8 {
        3
    }

    fn configure_spatial(&mut self, spatial: &SpatialConfig) -> Result<(), ConfigError> {
        if self.superres.is_some() {
            return Err(ConfigError::SpatialScalability(
                "spatial SVC cannot be combined with av1.superres_denom",
            ));
        }
        if self.grain.is_some() {
            return Err(ConfigError::SpatialScalability(
                "spatial SVC cannot be combined with film grain synthesis",
            ));
        }
        if !spatial
            .layer_sizes
            .windows(2)
            .all(|w| reference_scale_allowed(w[0], w[1]))
        {
            return Err(ConfigError::SpatialScalability(
                "AV1 spatial layer scale exceeds §7.11.3.3 limits",
            ));
        }
        // Layers of different sizes would need a unit grid per layer, and
        // `spatial_sequence_header` omits frame_id_numbers_present_flag.
        self.restoration.sequence = false;
        self.frame_ids = None;
        let top_id = spatial.layers.saturating_sub(1);
        let top_idx = usize::from(top_id);
        let mut parked: [Option<LayerGrids>; nc_encoder::MAX_SPATIAL_LAYERS] =
            std::array::from_fn(|_| None);
        for (idx, &(w, h)) in spatial.layer_sizes.iter().enumerate().take(top_idx) {
            parked[idx] = Some(LayerGrids::new(w, h));
        }
        self.spatial = Some(Box::new(BackendSpatial {
            config: spatial.clone(),
            parked,
            active_spatial_id: top_idx,
            current_layer: SpatialLayerFrame {
                spatial_id: top_id,
                temporal_id: 0,
                width: spatial.full_size.0,
                height: spatial.full_size.1,
                key_unit: true,
                intra_only: false,
                from_base_only: false,
                max_bitrate_bps: None,
            },
            slot_size: [None; MAX_REFERENCE_SLOTS],
            slot_layer: [None; MAX_REFERENCE_SLOTS],
        }));
        Ok(())
    }

    fn prepare_spatial_layer(&mut self, layer: SpatialLayerFrame) {
        self.switch_spatial_layer(layer);
    }

    fn commit_spatial_reference(
        &mut self,
        spatial_id: u8,
        recon: &Picture,
        size: (usize, usize),
        refresh: SlotMask,
        refresh_all: bool,
    ) {
        // Hold the normative picture only when the layer above predicts from
        // it in this unit (AV1 §7.11.3.3 scaled prediction).
        let store_for_upper = self.spatial.as_ref().is_some_and(|sp| {
            spatial_id + 1 < sp.config.layers
                && sp
                    .config
                    .inter_layer
                    .active_in_unit(sp.current_layer.key_unit)
        });
        let held = (store_for_upper && (refresh_all || !refresh.is_empty()))
            .then(|| HeldReference::new(Rc::new(recon.clone()), size.0, size.1));
        self.store
            .commit_spatial(refresh, refresh_all, held.as_ref());
    }

    /// Every reference names a slot (`ref_frame_idx`) and every frame its
    /// `refresh_frame_flags`, so the long-term slot needs nothing new.
    fn enable_block_qp(&mut self) {
        self.delta_q.enable();
    }

    fn enable_random_decisions(&mut self, rng: nc_encoder::DebugRng) {
        self.random = Some(RandomSyntax::new(rng));
    }

    fn supports_long_term_reference(&self) -> bool {
        true
    }

    // Exact candidate ranking already produced these pixels and statistics.
    // Retaining the improving winner avoids filtering it again for initial
    // prediction and residual rejection; the one-reference frame contract
    // plus block/MV/filter checks makes reuse exact. Joint-fast GOP300 host
    // instructions fall5.00%/2.54% on bodycam/crowd; paired native CPU
    // falls4.51%/0.83%, with identical streams.
    const CACHE_EXACT_LUMA: bool = true;

    fn cache_exact_luma(
        &mut self,
        block: &Block,
        mv: Mv,
        filter: nc_dsp::Filter,
        prediction: &[u8],
        stats: nc_dsp::Variance,
    ) {
        let area = block.size() * block.size();
        let secondary_ref = self.is_secondary();
        self.scratch.luma_prediction[..area].copy_from_slice(prediction);
        self.scratch.cached_luma = Some(CachedLuma {
            block: *block,
            mv: to_av1_mv(mv),
            filter,
            secondary_ref,
            stats,
        });
    }

    fn luma_filter(&self) -> Option<nc_dsp::Filter> {
        Some(nc_dsp::Filter::Av1)
    }

    fn lambda(&self) -> f32 {
        let q = (self.block_q_idx as usize).min(AC_QLOOKUP.len() - 1);
        let step = f32::from(AC_QLOOKUP[q]) / AC_Q_PER_SAMPLE_STEP;
        self.params.lambda_per_step_squared * step * step * self.lambda_scale
    }

    fn relative_quantizer(&self) -> f32 {
        self.base_q_idx as f32 / QINDEX_MAX as f32
    }

    /// The cheapest inter leaf at the current probabilities: skipped,
    /// inter, `LAST_FRAME`, `NEARESTMV`, each symbol at its cheapest
    /// context (VP9's estimate averages over contexts; for AV1 that charged
    /// rarely used, dear contexts and over-merged crowd: `J_real` +1.8%).
    /// Without it a split paid only its partition symbols, so whole-sample
    /// variance gains worth well under one bit split 64x64 blocks at low
    /// rates (`FoodMarket` 720p, 100 kbps: a median gain of 0.2–0.6 λ for 3
    /// extra leaves).
    fn planner_leaf_bits(&self) -> f32 {
        fn cheapest(costs: impl Iterator<Item = f32>) -> f32 {
            costs.fold(f32::INFINITY, f32::min)
        }
        let cdfs = &self.cdfs;
        cheapest(cdfs.skip.iter().map(|c| c.bits(1)))
            + cheapest(cdfs.intra_inter.iter().map(|c| c.bits(1)))
            + cheapest(cdfs.single_ref.iter().map(|ctx| {
                SINGLE_REF_SYMBOLS_FOR_LAST
                    .iter()
                    .map(|&p| ctx[p].bits(0))
                    .sum::<f32>()
            }))
            + cheapest(cdfs.new_mv.iter().map(|c| c.bits(1)))
            + cheapest(cdfs.zero_mv.iter().map(|c| c.bits(1)))
            + cheapest(cdfs.ref_mv.iter().map(|c| c.bits(0)))
    }

    fn partition_bits(&self, node: &Block, split: bool) -> f32 {
        let partition = if split {
            Partition::Split
        } else {
            Partition::None
        };
        match self.partition_symbol(node, partition) {
            PartitionSymbol::Full { ctx, nsyms: _ } => {
                self.cdfs.partition[ctx].bits(partition as usize)
            }
            PartitionSymbol::Binary(cdf) => symbol_bits(&cdf, usize::from(split)),
            PartitionSymbol::Implied => 0.0,
        }
    }

    fn supports_intra_leaf_cost(&self) -> bool {
        true
    }

    fn intra_leaf_cost(&mut self, source: &Picture, block: &Block) -> Option<f32> {
        let (above, left) = self.key_mode_context(block);
        let (cfl, _) = uv_mode_alphabet(block);
        let mode_bits = self.cdfs.kf_y_mode[above][left].bits(DC_PRED)
            + self.cdfs.uv_mode[cfl][DC_PRED].bits(UV_DC_PRED);
        let cost = [PlaneIndex::Y, PlaneIndex::U, PlaneIndex::V]
            .into_iter()
            .map(|plane| self.intra_plane_cost(source, block, plane))
            .sum::<f32>();
        Some(cost + self.lambda() * mode_bits)
    }

    fn static_skip_vector(&mut self, _block: &Block) -> Option<Mv> {
        // Also the vector of the engine's noise skip, which needs no
        // `av1.still_skip`.
        Some(Mv::ZERO)
    }

    fn copied_reference(&self, block: &Block) -> bool {
        // A held reference's prediction is not a copy of the engine's.
        self.luma_copy == Some((block.x, block.y)) && self.held_reference().is_none()
    }

    /// `av1.still_skip`: a still block whose zero-vector residual would
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
        // A held reference predicts scaled, not the engine's proxy.
        if self.held_reference().is_some() {
            return false;
        }
        let accepted = test.within_noise
            || (test.still && self.params.still_skip >= 0.5 && {
                let limits = PlaneZeroLimits {
                    luma: self.zero_limits(PlaneIndex::Y),
                    chroma: self.zero_limits(PlaneIndex::U),
                }
                .scaled(self.params.still_skip_limit_scale);
                zero_vector_residual_is_zero(&self.dsp, block, source, reference, limits)
            });
        if accepted {
            copy_zero_vector_prediction(block, reference, recon);
            self.still_skipped = Some((block.x, block.y));
        }
        accepted
    }

    fn mv_candidates(&mut self, block: &Block, out: &mut Vec<Mv>) {
        let stack = self.stack_for(block);
        // NEARESTMV denotes list position 0 and NEARMV positions 1..=3 (SVT's
        // light-PD1 `near_count`; AV1 allows three through `drl_mode`). Where
        // the list is shorter they are the global vector, zero, which the
        // engine adds itself. Offering positions 2 and 3 as well as 1:
        // quick clips, 10 s IP, PSNR-YUV BD vs svtav1-m12 (fast check)
        // +35.5% → +35.0% (bodycam −1.6, dashcam −1.3 points).
        let near_count = (self.params.near_count as usize).min(MAX_NEAR_INDEX);
        for index in 0..stack.count.min(1 + near_count) {
            out.push(to_engine_mv(stack.mvs[index]));
        }
    }

    fn mode_bits(&self, block: &Block, mode: &PredictionMode) -> f32 {
        match *mode {
            PredictionMode::IntraDc => {
                let (above, left) = self.neighbours(block);
                let intra_inter = intra_inter_context(above, left);
                let cdfs = &self.cdfs;
                let filter_flag = if self.codes_filter_intra(block, IntraMode::Dc) {
                    cdfs.filter_intra[filter_intra_size_context(block.log2_size())].bits(0)
                } else {
                    0.0
                };
                cdfs.intra_inter[intra_inter].bits(0)
                    + cdfs.y_mode[size_group(block)].bits(DC_PRED)
                    + cdfs.uv_mode[uv_mode_alphabet(block).0][DC_PRED].bits(UV_DC_PRED)
                    + filter_flag
            }
            PredictionMode::Inter(mv) => {
                let secondary_ref = self.is_secondary();
                // Mode decision prices every candidate vector of the block
                // [`Self::stack_for`] cached: read the cache in place (a
                // copy of it cost more than the pricing).
                if let Some(c) = self.current.as_ref().filter(|c| {
                    (c.x, c.y) == (block.x, block.y) && c.secondary_ref == secondary_ref
                }) {
                    return c.inter_base_bits
                        + inter_mode_bits(&self.cdfs, &c.stack, to_av1_mv(mv));
                }
                let (above, left) = self.neighbours(block);
                let base_bits = self.cdfs.intra_inter[intra_inter_context(above, left)].bits(1)
                    + self.single_ref_bits(above, left, secondary_ref);
                let stack = self.single_ref_stack(block, secondary_ref);
                base_bits + inter_mode_bits(&self.cdfs, &stack, to_av1_mv(mv))
            }
        }
    }

    fn compound_mode_bits(&mut self, block: &Block, primary_mv: Mv, secondary_mv: Mv) -> f32 {
        let (above, left) = self.neighbours(block);
        let stack =
            find_compound_mv_stack(&self.grids.mode_info, block.x, block.y, block.log2_size());
        let mvs = [to_av1_mv(primary_mv), to_av1_mv(secondary_mv)];
        self.cdfs.intra_inter[intra_inter_context(above, left)].bits(1)
            + comp_ref_bits(&self.cdfs, above, left)
            + compound_inter_mode_bits(&self.cdfs, &stack, mvs)
    }

    fn set_refresh(&mut self, window: Option<&RefreshWindow>) {
        self.refresh = window.cloned();
    }

    fn intra_fallback(&mut self, block: &Block, mv: Mv) {
        self.intra_fallback = Some(((block.x, block.y, block.log2_size()), mv));
    }

    fn set_noise_sigma(&mut self, sigma: f32) {
        self.noise_sigma = sigma;
    }

    fn set_frame_params(&mut self, params: &nc_encoder::ParamOverrides) {
        let params = Av1Params::from_overrides(params);
        // Everything `begin_frame` does not re-derive from `self.params`.
        self.shave_min_zero_run = params.shave_min_zero_run.round() as usize;
        self.chroma_skip_sad_num = params.chroma_skip_sad_num.round() as u32;
        self.params = params;
    }

    fn rate_model(&self) -> nc_encoder::RateModel {
        RATE_MODEL
    }

    fn rate_control_quantizer_step(&self, quantizer: u32) -> f64 {
        f64::from(AC_QLOOKUP[(quantizer as usize).min(255)])
    }

    fn set_rate_control_enabled(&mut self, enabled: bool) {
        self.quantizer_mode = if enabled {
            QuantizerMode::RateControlled
        } else {
            QuantizerMode::Fixed
        };
    }

    fn begin_frame(&mut self, plan: &FramePlan) {
        self.lambda_scale = plan.lambda_scale;
        self.active_ref = ReferenceChoice::Primary;
        // The engine uses one immutable reference throughout this frame.
        self.scratch.cached_luma = None;
        // Key frames start from the default CDFs; inter frames from those
        // `LAST_FRAME` saved (`load_cdfs`, AV1 §5.9.2). No block information
        // crosses frames (no temporal vectors).
        let previous_was_key = self.grids.committed_kind == FrameKind::Key;
        self.grids
            .cdef_memory
            .clone_from(&self.grids.committed_cdef_memory);
        self.kind = match plan.frame_type {
            FrameType::Key => FrameKind::Key,
            FrameType::Inter => FrameKind::Inter,
        };
        self.base_q_idx = plan.quantizer;
        let fixed_key = self.kind == FrameKind::Key && self.quantizer_mode == QuantizerMode::Fixed;
        self.frame_q_idx = if fixed_key {
            scaled_qindex(self.base_q_idx, self.key_step_scale())
        } else {
            self.base_q_idx
        };
        self.references = plan.references;
        self.reference_held = self.kind == FrameKind::Inter
            && self.store.held(self.references.reference.slot).is_some();
        // With real temporal layers the base frames are the T0 frames.
        self.is_base_frame = if plan.references.position.is_flat() {
            plan.index.is_multiple_of(self.base_period)
        } else {
            plan.references.position.layer == 0
        };
        self.pre_tx_score_this_frame = if self.kind == FrameKind::Inter
            && self.params.pre_tx_skip > 0.0
            && !self.is_base_frame
            && self.params.pre_tx_skip_score_nonbase > 0.0
        {
            self.params.pre_tx_skip_score_nonbase
        } else {
            self.params.pre_tx_skip_score
        };
        self.set_rounding();

        self.frame_chroma_delta_q = noise_adapted_chroma_delta(
            self.params.chroma_delta_q,
            self.noise_sigma,
            self.params.chroma_noise_scale,
        );
        if let Some(policy) = self.chroma_policy {
            self.frame_chroma_delta_q = policy_chroma_delta(policy, self.frame_q_idx);
        } else if self.kind == FrameKind::Key && self.params.keyframe_chroma_scaled >= 0.5 {
            // Under rate control the controller picks the key qindex itself;
            // its inter equivalent undoes the key-frame step factor.
            let inter_q_idx = if fixed_key {
                self.base_q_idx
            } else {
                scaled_qindex(self.frame_q_idx, 1.0 / self.key_step_scale())
            };
            self.frame_chroma_delta_q = key_chroma_delta(
                inter_q_idx,
                self.frame_q_idx,
                self.frame_chroma_delta_q,
                self.key_step_scale(),
            );
        }
        if let Some(random) = self.random.as_mut() {
            self.frame_chroma_delta_q = random.chroma_delta_q();
        }
        self.set_block_qindex(self.frame_q_idx);
        self.delta_q.start_frame(self.frame_q_idx, plan.block_qp);
        self.loop_filter = self.frame_loop_filter();
        self.tx_mode = self.frame_tx_mode();
        self.grids.txfm.start_frame();
        match self.kind {
            FrameKind::Key => *self.cdfs = CdfContext::new(coeff_cdf_q_ctx(self.frame_q_idx)),
            FrameKind::Inter => {
                let saved = self.slot_cdfs[plan.references.reference.slot]
                    .as_deref()
                    .expect("inter frames follow a key frame");
                self.cdfs.as_mut().clone_from(saved);
            }
        }
        self.cdef_frame = self.cdef_frame_choice(plan.index, previous_was_key);
        self.cdef_alternate = None;
        self.grids.cdef_slots.fill(None);
        self.restoration.frame = self.restoration_frame(plan);
        self.restoration.spans.clear();
        self.restoration.chosen = None;
        self.writer = if self.cdef_frame.per_superblock || self.restoration.frame {
            SymbolWriter::new_deferred()
        } else {
            SymbolWriter::new()
        };
        if !self.adapts_cdfs() {
            self.writer = std::mem::take(&mut self.writer).without_adaptation();
        }
        self.grids.mode_info.clear();
        self.current = None;
        for above in &mut self.grids.contexts.above {
            above.fill(0);
        }
    }

    fn begin_superblock(&mut self, x: usize, y: usize) {
        self.delta_q.start_superblock();
        // Restoration units precede the partition (AV1 §5.11.57) but are
        // chosen after CDEF: keep their place.
        if self.restoration.frame {
            let slot = self.writer.reserve_span();
            self.restoration.spans.push((slot, (x, y)));
        }
        if x == 0 {
            // AV1 §5.11.2: left contexts are cleared at the start of every
            // superblock row of the tile.
            for left in &mut self.grids.contexts.left {
                left.fill(0);
            }
            self.grids.txfm.start_sb_row();
        }
    }

    fn write_partition(&mut self, node: &Block, split: bool) {
        self.write_partition_symbol(
            node,
            if split {
                Partition::Split
            } else {
                Partition::None
            },
        );
    }

    fn encode_block(
        &mut self,
        block: &Block,
        decision: &BlockDecision,
        source: &Picture,
        reference: Option<&Picture>,
        recon: &mut Picture,
    ) {
        self.active_ref = decision.reference;
        let block_q_idx = self
            .delta_q
            .block_qindex(self.frame_q_idx, decision.qp_offset);
        self.use_block_qindex(block_q_idx);
        let (inter, intra_mode) =
            self.choose_block_prediction(block, decision, source, reference, recon);
        let coded = match inter {
            None => CodedPrediction::Intra(intra_mode),
            Some((mv, reference)) => CodedPrediction::Single { reference, mv },
        };
        self.code_block(block, block_q_idx, &coded, source, recon);
    }

    fn encode_compound_block(
        &mut self,
        block: &Block,
        decision: &BlockDecision,
        source: &Picture,
        primary_ref: &Picture,
        secondary_ref: &Picture,
        recon: &mut Picture,
    ) {
        self.active_ref = decision.reference;
        let block_q_idx = self
            .delta_q
            .block_qindex(self.frame_q_idx, decision.qp_offset);
        self.use_block_qindex(block_q_idx);
        let (PredictionMode::Inter(primary_mv), ReferenceChoice::Compound { secondary_mv }) =
            (decision.mode, decision.reference)
        else {
            unreachable!("compound blocks carry an inter decision and a compound reference");
        };
        let compound = CompoundPrediction {
            primary: primary_ref,
            secondary: secondary_ref,
            mvs: [to_av1_mv(primary_mv), to_av1_mv(secondary_mv)],
        };
        let coded = CodedPrediction::Compound(&compound);
        self.code_block(block, block_q_idx, &coded, source, recon);
    }

    fn filter_frame(&mut self, source: &Picture, recon: &mut Picture) {
        // Deblocking precedes CDEF (§7.14, §7.15), so CDEF chooses its
        // strengths on the deblocked picture the decoder will filter.
        loopfilter::apply(
            &self.dsp,
            recon,
            &self.grids.mode_info,
            self.loop_filter,
            self.grids.width,
            self.grids.height,
        );
        let boundaries = self
            .restoration
            .frame
            .then(|| restoration::stripe_boundaries(recon, (self.grids.width, self.grids.height)));
        if self.random.is_some() {
            self.random_cdef(recon);
        } else {
            self.apply_cdef(source, recon);
        }
        if let Some(boundaries) = boundaries {
            self.restore(source, recon, &boundaries);
        }
        // Superres upscales the CDEF output, reading the coded area past
        // the frame width as the decoder reconstructed it (§7.16).
        if let Some(superres) = &mut self.superres {
            superres.upscale(recon, self.grids.width);
        }
        // Motion compensation clamps reference positions to the frame size
        // (§7.11.3.3), not to the 8x8-aligned area coded beyond it.
        if (recon.width(), recon.height()) != (self.grids.width, self.grids.height) {
            recon.replicate_beyond(self.grids.width, self.grids.height);
        }
        if let Some(grain) = &mut self.grain {
            let reference =
                (self.kind == FrameKind::Inter).then_some(self.references.reference.slot as u8);
            grain.plan(
                source,
                recon,
                (self.grids.width, self.grids.height),
                reference,
            );
        }
    }

    fn end_frame(&mut self) -> Vec<u8> {
        let refs = self.references;
        let intra_refresh = if self
            .spatial
            .as_ref()
            .is_some_and(|sp| sp.current_layer.intra_only)
        {
            IntraRefresh::SelectedSlots
        } else {
            IntraRefresh::AllSlots
        };
        let tile = std::mem::take(&mut self.writer).finish();
        let mut frame = BitWriter::new();
        FrameHeader {
            kind: self.kind,
            intra_refresh,
            width: self.grids.width,
            height: self.grids.height,
            base_q_idx: self.frame_q_idx,
            chroma_delta_q: self.chroma_delta_q(),
            delta_q_present: self.delta_q.present(),
            loop_filter: self.loop_filter,
            tx_mode_select: self.tx_mode == TxMode::Select,
            disable_cdf_update: !self.adapts_cdfs(),
            reference_select: self.frame_reference_select(),
            cdef: self.cdef,
            cdef_alternate: self.cdef_alternate,
            restoration: self.restoration.sequence.then(|| {
                self.restoration
                    .chosen
                    .as_ref()
                    .map(FrameRestoration::params)
                    .unwrap_or_default()
            }),
            references: SlotUse {
                reference: refs.reference.slot as u8,
                secondary: refs.secondary.map(|s| s.slot as u8),
                refresh: refs.refresh.0,
            },
            film_grain: self.grain.as_ref().map(GrainState::syntax),
            superres_denom: self.superres.as_ref().map(Superres::denom),
            size: self.frame_size_syntax(),
            frame_id: self
                .frame_ids
                .as_ref()
                .map(|ids| ids.syntax(self.kind, refs.reference.slot)),
        }
        .write(&mut frame);
        // tile_group_obu() with one tile: no tile_start_and_end_present_flag,
        // and the last tile carries no size field (AV1 §5.11.1).
        frame.put_bytes(&tile);
        let frame_bytes = frame.into_bytes();
        if self.spatial.is_some() {
            return self.write_spatial_unit(&frame_bytes);
        }

        let mut unit = Vec::with_capacity(tile.len() + 64);
        write_obu(&mut unit, ObuType::TemporalDelimiter, None, &[]);
        if self.kind == FrameKind::Key {
            write_obu(
                &mut unit,
                ObuType::SequenceHeader,
                None,
                &sequence_header(
                    self.stream_size.0,
                    self.stream_size.1,
                    self.temporal_layers,
                    self.signal,
                    self.frame_rate,
                    SequenceTools {
                        filter_intra: self.filter_intra_enabled(),
                        film_grain: self.grain.is_some(),
                        superres: self.superres.is_some(),
                        restoration: if self.restoration.sequence {
                            RestorationTool::Enabled
                        } else {
                            RestorationTool::Disabled
                        },
                        frame_ids: self.frame_ids.as_ref().map(FrameIds::lengths),
                    },
                ),
            );
        }
        let temporal_id = (self.temporal_layers > 1).then_some(refs.position.layer);
        write_obu(&mut unit, ObuType::Frame, temporal_id, &frame_bytes);
        unit
    }

    fn commit_frame(&mut self) {
        let intra_only = self
            .spatial
            .as_ref()
            .is_some_and(|sp| sp.current_layer.intra_only);
        let refresh_all = self.kind == FrameKind::Key && !intra_only;
        // The refreshed slots now hold this frame's final CDFs (AV1 §7.20);
        // key frames refresh every slot, whereas intra-only frames refresh
        // only their explicit mask.
        let saved = Rc::new(self.cdfs.saved());
        for (slot, cdfs) in self.slot_cdfs.iter_mut().enumerate() {
            if refresh_all || self.references.refresh.contains(slot) {
                *cdfs = Some(Rc::clone(&saved));
            }
        }
        std::mem::swap(
            &mut self.grids.cdef_memory,
            &mut self.grids.committed_cdef_memory,
        );
        if let Some(sp) = &mut self.spatial {
            let size = (self.grids.width, self.grids.height);
            let layer_id = sp.current_layer.spatial_id;
            for slot in 0..MAX_REFERENCE_SLOTS {
                if refresh_all || self.references.refresh.contains(slot) {
                    sp.slot_size[slot] = Some((size, size));
                    sp.slot_layer[slot] = Some(layer_id);
                }
            }
            self.grids.committed_kind = self.kind;
            return;
        }
        if let Some(grain) = &mut self.grain {
            let (key, refresh) = (self.kind == FrameKind::Key, self.references.refresh);
            grain.commit(|slot| key || refresh.contains(slot));
        }
        let upscaled = self.superres.as_mut().map(Superres::commit);
        self.store.commit(
            self.kind == FrameKind::Key,
            self.references.refresh,
            upscaled.as_ref(),
        );
        if let Some(ids) = &mut self.frame_ids {
            ids.commit(self.kind == FrameKind::Key, self.references.refresh);
        }
        self.grids.committed_kind = self.kind;
    }
}

impl Av1Backend {
    /// Code `block` at `block_q_idx` with its chosen prediction: residuals,
    /// mode syntax and mode info. Every block goes through here, so each
    /// coding step has this one caller.
    fn code_block(
        &mut self,
        block: &Block,
        block_q_idx: u32,
        coded: &CodedPrediction<'_>,
        source: &Picture,
        recon: &mut Picture,
    ) {
        let (prediction, intra_mode, single) = match *coded {
            CodedPrediction::Intra(intra_mode) => (Prediction::Intra(intra_mode), intra_mode, None),
            CodedPrediction::Single { reference, mv } => (
                Prediction::Inter { reference, mv },
                LumaIntra::DC,
                Some((mv, reference)),
            ),
            CodedPrediction::Compound(compound) => {
                (Prediction::Compound(compound), LumaIntra::DC, None)
            }
        };
        let compound = match *coded {
            CodedPrediction::Compound(compound) => Some(compound),
            _ => None,
        };
        let is_inter = single.is_some() || compound.is_some();
        let still_skipped =
            single.is_some() && self.still_skipped.take() == Some((block.x, block.y));
        // CfL depends on reconstructed luma, so choose it after coding Y.
        let (luma, tx_split, luma_leaves) = if still_skipped {
            let luma = self.predict_without_residual(block, PlaneIndex::Y, &prediction, recon);
            (luma, TxSplit::NONE, Vec::new())
        } else {
            self.code_luma(block, &prediction, is_inter, source, recon)
        };
        let cfl = if compound.is_some() {
            None
        } else {
            self.choose_cfl(block, intra_mode.mode, !is_inter, source, recon)
        };
        let chroma_prediction = cfl.as_ref().map_or_else(
            || match *coded {
                CodedPrediction::Intra(intra_mode) => Prediction::Intra(intra_mode),
                CodedPrediction::Single { reference, mv } => Prediction::Inter { reference, mv },
                CodedPrediction::Compound(compound) => Prediction::Compound(compound),
            },
            Prediction::ChromaFromLuma,
        );
        let skip_chroma = still_skipped || self.bypass_uv_residual(&luma);
        let [u, v] = self.code_chroma(block, &chroma_prediction, source, recon, skip_chroma);
        let mut residuals = [luma, u, v];
        if !tx_split.split {
            self.drop_unprofitable_luma_residual(block, &prediction, &mut residuals[0], recon);
        }
        if self.random.is_some() {
            let predictions = [&prediction, &chroma_prediction, &chroma_prediction];
            // Luma stays when chroma is CfL (predicted from it) or when it
            // is a transform tree (each leaf has its own residual).
            let keep_luma = cfl.is_some() || tx_split.split;
            self.random_residual_drops(block, predictions, &mut residuals, keep_luma, recon);
        }
        let skip = residuals.iter().all(|r| r.eob == 0);

        let (above, left) = self.neighbours(block);
        self.writer
            .write_bool(skip, &mut self.cdfs.skip[skip_context(above, left)]);
        self.write_cdef_index(block, skip);
        self.write_delta_qindex(block, block_q_idx, skip);
        let mut info = ModeInfo {
            size_log2_mi: block.log2_size() - MI_LOG2 as u8,
            skip,
            mode_detail: intra_mode.mode.symbol() as u8,
            tx_log2: block.log2_size(),
            ..ModeInfo::default()
        };
        self.write_block_mode(
            block,
            coded,
            intra_mode,
            cfl.as_ref(),
            (above, left),
            &mut info,
        );

        // A skipped block codes no tree: its transform is the block.
        let (tx_split, luma_leaves) = if skip {
            (TxSplit::NONE, Vec::new())
        } else {
            (tx_split, luma_leaves)
        };
        if self.tx_mode == TxMode::Select {
            self.write_tx_size(block, is_inter, skip, tx_split, above, left);
        }
        self.write_residuals(block, &residuals, skip, &luma_leaves, &info, intra_mode);
        self.luma_copy = match *coded {
            CodedPrediction::Single { mv, .. } if !info.is_last2_only() => {
                luma_copy_at(block, Some(mv), &residuals[0], &luma_leaves)
            }
            CodedPrediction::Intra(_) => luma_copy_at(block, None, &residuals[0], &luma_leaves),
            CodedPrediction::Single { .. } | CodedPrediction::Compound(_) => None,
        };
        if let Some(compound) = compound {
            self.grids.mode_info.fill_mv2(
                block.y >> MI_LOG2,
                block.x >> MI_LOG2,
                info.size_mi(),
                compound.mvs[1],
            );
        }
        self.record_block(block, info, tx_split);
        self.current = None;
        self.active_ref = ReferenceChoice::Primary;
        self.use_block_qindex(self.frame_q_idx);
    }

    /// The mode syntax of a block coded with `coded` (intra modes, or its
    /// references and vectors), recording the inter choice in `info`.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn write_block_mode(
        &mut self,
        block: &Block,
        coded: &CodedPrediction<'_>,
        intra_mode: LumaIntra,
        cfl: Option<&ChromaFromLuma>,
        (above, left): (Option<ModeInfo>, Option<ModeInfo>),
        info: &mut ModeInfo,
    ) {
        match (self.kind, *coded) {
            (FrameKind::Key, _) => {
                self.write_key_frame_intra_modes(block, intra_mode, cfl);
            }
            (FrameKind::Inter, CodedPrediction::Intra(_)) => {
                self.writer.write_bool(
                    false,
                    &mut self.cdfs.intra_inter[intra_inter_context(above, left)],
                );
                self.write_intra_modes(block, intra_mode, cfl);
            }
            (FrameKind::Inter, CodedPrediction::Single { mv, .. }) => {
                self.writer.write_bool(
                    true,
                    &mut self.cdfs.intra_inter[intra_inter_context(above, left)],
                );
                let secondary_ref = self.is_secondary();
                self.write_ref_frame(above, left, secondary_ref);
                let stack = self.stack_for(block);
                let mode = self.choose_inter_mode(&stack, mv);
                self.write_inter_mode(&stack, mv, mode);
                info.mv = mv;
                info.mode = mode.block_mode();
                info.mode_detail = InterRefs::single(secondary_ref).code();
            }
            (FrameKind::Inter, CodedPrediction::Compound(compound)) => {
                self.writer.write_bool(
                    true,
                    &mut self.cdfs.intra_inter[intra_inter_context(above, left)],
                );
                self.write_comp_ref_frame(above, left);
                let stack = find_compound_mv_stack(
                    &self.grids.mode_info,
                    block.x,
                    block.y,
                    block.log2_size(),
                );
                let mode = self.choose_compound_inter_mode(&stack, compound.mvs);
                self.write_compound_inter_mode(&stack, compound.mvs, mode);
                info.mv = compound.mvs[0];
                info.mode = mode.block_mode();
                info.mode_detail = InterRefs::LastLast2.code();
            }
        }
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn choose_block_prediction<'a>(
        &mut self,
        block: &Block,
        decision: &BlockDecision,
        source: &Picture,
        reference: Option<&'a Picture>,
        recon: &mut Picture,
    ) -> (Option<(Av1Mv, &'a Picture)>, LumaIntra) {
        match decision.mode {
            PredictionMode::Inter(mv) => {
                let reference = reference.expect("inter blocks need a reference");
                (Some((to_av1_mv(mv), reference)), LumaIntra::DC)
            }
            PredictionMode::IntraDc if self.kind == FrameKind::Key || self.random.is_some() => {
                (None, self.choose_intra_mode(block, source, recon))
            }
            PredictionMode::IntraDc => {
                match self.inter_frame_intra(block, source, reference, recon) {
                    InterFrameIntra::Intra(luma) => (None, luma),
                    InterFrameIntra::Inter(mv, reference) => (Some((mv, reference)), LumaIntra::DC),
                }
            }
        }
    }
}

/// CDEF choices carried from frame to frame: the per-superblock selection
/// and strengths of the last searched frame, reused until the next search.
#[derive(Debug, Clone, Default)]
struct CdefMemory {
    /// Per-filter-block on/off of the last per-superblock search.
    cached_selection: Vec<bool>,
    /// Luma candidate of the previous per-superblock frame.
    previous_candidate: Option<(u8, u8)>,
    /// Chroma strength chosen on the last searched frame (`cdef_uv_search_primary`).
    uv_choice: Option<(u8, u8)>,
    /// Luma CDEF strength chosen on the last searched frame, reused until
    /// the next (SVT level 9).
    luma: (u8, u8),
}

/// How one plane of a block is predicted.
enum Prediction<'a> {
    /// Luma intra; chroma intra is DC in our subset (or `CfL`, below).
    Intra(LumaIntra),
    ChromaFromLuma(&'a ChromaFromLuma),
    Inter {
        reference: &'a Picture,
        mv: Av1Mv,
    },
    /// Behind a reference so the enum stays as small as the
    /// single-reference variants every other block uses.
    Compound(&'a CompoundPrediction<'a>),
}

/// The two references and vectors of a compound block (`LAST_FRAME`,
/// `LAST2_FRAME`; AV1 §7.11.3.15).
struct CompoundPrediction<'a> {
    primary: &'a Picture,
    secondary: &'a Picture,
    mvs: [Av1Mv; 2],
}

/// The prediction a block is coded with ([`Av1Backend::code_block`]).
#[derive(Clone, Copy)]
enum CodedPrediction<'a> {
    Intra(LumaIntra),
    /// One reference (`LAST_FRAME` or `LAST2_FRAME`, [`Av1Backend::active_ref`]).
    Single {
        reference: &'a Picture,
        mv: Av1Mv,
    },
    Compound(&'a CompoundPrediction<'a>),
}

/// One plane's quantised residual for a block (levels in `Scratch`).
struct PlaneResidual {
    tx: TxSize,
    eob: usize,
    /// Present only for a coded luma residual eligible for the RD check.
    distortion: Option<ResidualDistortion>,
    /// Specific transform-domain rejection; ordinary eob0 never enables UV bypass.
    transform_rd_rejected: bool,
}

impl PlaneResidual {
    fn uncoded(tx: TxSize) -> Self {
        Self {
            tx,
            eob: 0,
            distortion: None,
            transform_rd_rejected: false,
        }
    }
}

/// Squared errors used only by the near-static inter-luma residual decision.
struct ResidualDistortion {
    /// Squared error of the prediction alone (what skipping the residual gives).
    sse_predicted: u32,
    /// Squared error of the reconstruction with the residual coded.
    sse_coded: u32,
    /// Reuse the exact early CDF estimate on surviving residuals.
    coefficient_bits: Option<f32>,
}

/// One improving exact candidate, valid only in the current frame/reference.
#[derive(Debug, Clone, Copy)]
struct CachedLuma {
    block: Block,
    mv: Av1Mv,
    filter: nc_dsp::Filter,
    secondary_ref: bool,
    stats: nc_dsp::Variance,
}

/// Buffers reused by every block (no per-block allocation).
#[derive(Debug)]
struct Scratch {
    /// Exact filtered prediction, untouched by forward/inverse transforms.
    luma_prediction: Vec<u8>,
    cached_luma: Option<CachedLuma>,
    /// Residual of the plane being coded, row-major.
    residual: Vec<i16>,
    /// Quantised levels of the block, per plane (libaom layout).
    /// Contents are unspecified when eob is zero: coefficient coding/context
    /// return before reading them, and the next forward overwrites its area.
    levels: [Vec<i32>; 3],
    /// Dequantised coefficients of the plane being coded.
    dequant: Vec<i32>,
    /// Allocated only for enabled transform-domain luma RD and key-frame
    /// transform-based planning or mode search.
    original_coefficients: Vec<i32>,
    /// Constant DC prediction for key-frame partition planning.
    flat_prediction: Vec<u8>,
    /// Quantised luma levels of the intra mode search's winner.
    intra_levels: Vec<i32>,
    /// Their unquantised coefficients, where `CachedIntraLevels::originals`
    /// says they were kept (for the trellis).
    intra_originals: Vec<i32>,
    cached_intra: Option<CachedIntraLevels>,
    /// Luma prediction residual of a whole block, for the transform-split
    /// decision.
    tree_residual: Vec<i16>,
    /// Quantised levels of every leaf of a split luma transform tree.
    tree_levels: Vec<i32>,
    /// The block whose one-level split the decision quantised into
    /// `tree_levels`, with each quarter's nonzero count and whether
    /// `tree_originals` holds the quarters' unquantised coefficients.
    tree_trial: Option<(Block, [u32; 4], bool)>,
    /// Unquantised coefficients of the trial's quarters, laid out as
    /// `tree_levels` (for the trellis).
    tree_originals: Vec<i32>,
    /// `quant.trellis` search buffers.
    trellis: TrellisQuantizer,
}

/// `TxMode` of a frame (AV1 §5.9.21 `read_tx_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TxMode {
    /// `TX_MODE_LARGEST`: every transform is its block.
    Largest,
    /// `TX_MODE_SELECT`: blocks signal their transform sizes.
    Select,
}

/// One coded leaf of a split luma transform tree.
#[derive(Debug, Clone, Copy)]
struct LumaLeaf {
    /// Offset in luma samples from the block's top-left corner.
    dx: usize,
    dy: usize,
    /// log2 of the transform side.
    log2: u8,
    eob: usize,
    /// Start of its levels in `Scratch::tree_levels`.
    offset: usize,
}

/// Smallest block whose inter transform may split (16x16 into 8x8).
const MIN_TX_SPLIT_LOG2: u8 = 4;
/// Luma 4x4 units per superblock side (the left context's extent).
const SB_MI_UNITS: usize = 16;

/// Which block and prediction [`Scratch::intra_levels`] belong to: the
/// winner of the key-frame intra mode search, or (`mode` `None`) the
/// whole-block trial of the inter transform-split decision.
#[derive(Debug, Clone, Copy)]
struct CachedIntraLevels {
    block: Block,
    mode: Option<LumaIntra>,
    nonzero: u32,
    /// `Scratch::intra_originals` holds the levels' unquantised
    /// coefficients ([`Av1Backend::trial_coefficients_reusable`]).
    originals: bool,
}

/// Largest block and largest coded coefficient array.
const MAX_BLOCK_AREA: usize = 64 * 64;
const MAX_CODED_AREA: usize = 32 * 32;

impl Scratch {
    fn new() -> Self {
        Self {
            luma_prediction: vec![0; MAX_BLOCK_AREA],
            cached_luma: None,
            residual: vec![0; MAX_BLOCK_AREA],
            levels: std::array::from_fn(|_| vec![0; MAX_CODED_AREA]),
            dequant: vec![0; MAX_CODED_AREA],
            original_coefficients: Vec::new(),
            flat_prediction: vec![0; MAX_BLOCK_AREA],
            intra_levels: vec![0; MAX_CODED_AREA],
            intra_originals: vec![0; MAX_CODED_AREA],
            cached_intra: None,
            tree_residual: vec![0; MAX_BLOCK_AREA],
            tree_levels: vec![0; MAX_BLOCK_AREA],
            tree_originals: vec![0; MAX_BLOCK_AREA],
            tree_trial: None,
            trellis: TrellisQuantizer::default(),
        }
    }
}

impl Av1Backend {
    /// This frame's forward approximation (`av1.partial_forward`,
    /// `av1.partial_forward16_min_q`) for an RD-cost trial or (`coding`)
    /// the residual that is coded.
    /// Whether a trial's unquantised coefficients can stand in for the
    /// coding transform's, so the trellis can reuse them: the trellis is
    /// configured and trials transform as coding does (exact: same residual,
    /// same forward). Recomputing them cost the trellis path a second
    /// forward transform per coded block.
    fn trial_coefficients_reusable(&self) -> bool {
        self.trellis.is_some() && self.forward_policy(false) == self.forward_policy(true)
    }

    fn forward_policy(&self, coding: bool) -> ForwardPolicy {
        let scope = if coding { 1.5 } else { 0.5 };
        // A block quantised finer than the frame is being refined (the
        // reuse term restoring detail a degraded reference lost), and an
        // enhancement picture coded from the upscaled base only adds the
        // detail the base lacks: the high frequencies the approximation
        // drops are what both are for. Approximated, av1-svc-fast's
        // enhancement layer kept the blurred base on still content
        // (static_recovery.rs: 32.3 dB vs 44.5 exact).
        let applies = self.params.partial_forward >= scope
            && !self.refining_block()
            && !self.coding_from_base_only();
        ForwardPolicy {
            large: applies,
            medium: applies && self.frame_q_idx as f32 >= self.params.partial_forward16_min_q,
        }
    }

    /// The strong luma CDEF candidate: the configured primary (optionally
    /// capped by the model, `cdef_local_strong_model_scale`) and secondary.
    fn strong_luma_candidate(&self, model: (u8, u8)) -> (u8, u8) {
        (
            strong_luma_primary(
                self.params.cdef_local_strong_primary,
                model.0,
                self.params.cdef_local_strong_model_scale,
            ),
            self.params.cdef_local_strong_secondary.round() as u8,
        )
    }

    /// This frame's deblocking levels: off, or libaom's pick-from-Q level
    /// for every plane and direction (`av1.loop_filter`). The syntax stress
    /// test picks random levels where the filter is on, so
    /// `av1.loop_filter = 0` keeps testing streams without it.
    fn frame_loop_filter(&mut self) -> LoopFilterParams {
        if self.params.loop_filter < 0.5 {
            return LoopFilterParams::OFF;
        }
        if let Some(random) = self.random.as_mut() {
            return random.loop_filter();
        }
        let ac_q = i32::from(AC_QLOOKUP[self.frame_q_idx as usize]);
        let fitted = loopfilter::level_from_q(ac_q, self.kind == FrameKind::Key);
        let level = (f32::from(fitted) * self.params.loop_filter_level_scale).round();
        LoopFilterParams {
            level: [level.clamp(0.0, f32::from(loopfilter::MAX_LOOP_FILTER)) as u8; 4],
            sharpness: self.params.loop_filter_sharpness.round() as u8,
        }
    }

    /// This frame's `TxMode`: inter frames signal transform sizes with
    /// `av1.inter_tx_split`, and always in the syntax stress test. Without
    /// the split search they still do when blocks may be refined
    /// (`delta_q_present`): a refined 64x64 block needs four 32x32
    /// transforms (see [`Self::refined_tx_split`]).
    fn frame_tx_mode(&self) -> TxMode {
        let select =
            self.params.inter_tx_split >= 0.5 || self.random.is_some() || self.delta_q.present();
        if self.kind == FrameKind::Inter && select {
            TxMode::Select
        } else {
            TxMode::Largest
        }
    }

    /// Whether this frame searches loop restoration. SVT-AV1 (non-RTC
    /// presets) restores only frames other frames predict from; with
    /// `restoration_top_layer` 0 the top temporal layer of a layered
    /// stream skips the search (and its cost) likewise.
    fn restoration_frame(&self, plan: &FramePlan) -> bool {
        let position = plan.references.position;
        let top_layer = self.temporal_layers > 1 && position.layer + 1 == self.temporal_layers;
        self.restoration.sequence
            && self.random.is_none()
            && self.params.restoration >= 0.5
            && (self.params.restoration_top_layer >= 0.5 || !top_layer)
    }

    /// Choose this frame's restoration on `recon` (after CDEF), filter
    /// `recon` with it, and code each superblock's units into its place.
    fn restore(
        &mut self,
        source: &Picture,
        recon: &mut Picture,
        boundaries: &[nc_dsp::av1_restoration::Av1StripeBoundaries; 3],
    ) {
        let unit_size = self.params.restoration_unit_size.round() as usize;
        let settings = SearchSettings {
            luma_reach: (self.params.restoration_luma_taps.round() as usize).saturating_sub(1) / 2,
            chroma: self.params.restoration_chroma >= 0.5,
            unit_size: restoration::UNIT_SIZES
                .iter()
                .copied()
                .find(|&size| size == unit_size),
            row_step: (self.params.restoration_row_step.round() as usize).max(1),
            refine_passes: self.params.restoration_refine.round() as usize,
            lambda: f64::from(self.lambda() * self.params.restoration_lambda_scale),
        };
        let size = (self.grids.width, self.grids.height);
        let Some(chosen) = restoration::search(
            &self.dsp,
            &settings,
            (source, recon),
            size,
            &self.cdfs.use_wiener,
        ) else {
            return;
        };
        chosen.apply(
            (&self.dsp, &mut self.restoration.scratch),
            recon,
            boundaries,
            size,
        );
        let mut references = WienerReferences::default();
        for &(slot, position) in &self.restoration.spans {
            chosen.write_superblock(
                &mut self.writer.span(slot),
                &mut self.cdfs.use_wiener,
                &mut references,
                position,
            );
        }
        self.restoration.chosen = Some(chosen);
    }

    /// Choose and apply this frame's CDEF strengths (the in-loop filter).
    fn apply_cdef(&mut self, source: &Picture, recon: &mut Picture) {
        let q = AC_QLOOKUP[self.frame_q_idx as usize];
        let (luma_model, chroma) =
            cdef::strengths_from_q(f32::from(q), self.kind == FrameKind::Key);
        let mut params = CdefParams {
            damping_minus_3: (self.frame_q_idx >> 6) as u8,
            y_pri: self.grids.cdef_memory.luma.0,
            y_sec: self.grids.cdef_memory.luma.1,
            uv_pri: chroma.0,
            uv_sec: chroma.1,
        };
        if self.cdef_frame.per_superblock {
            let strong_enabled = self.params.cdef_local_strong_luma >= 0.5;
            let noise_limit = self.params.cdef_local_noise_full_search_sigma;
            // A variance minimum of 0 or less admits every frame, so the
            // picture variance (a pass over the source luma) is only
            // measured when it can decide.
            let variance_min = self.params.cdef_local_strong_variance_min;
            let strong = strong_enabled
                && self.kind == FrameKind::Inter
                && (noise_limit <= 0.0 || self.noise_sigma < noise_limit)
                && (variance_min <= 0.0
                    || cdef::picture_variance(&self.dsp, &source.plane(PlaneIndex::Y).view())
                        as f32
                        >= variance_min);
            let candidate = if strong {
                self.strong_luma_candidate(luma_model)
            } else {
                luma_model
            };
            (params.y_pri, params.y_sec) = candidate;
            let must_search = self.cdef_frame.search
                || (strong_enabled && !strong)
                || self.grids.cdef_memory.previous_candidate != Some(candidate);
            self.grids.cdef_memory.previous_candidate = Some(candidate);
            (params.uv_pri, params.uv_sec) =
                self.chroma_strength(source, recon, params, chroma, must_search);
            let selected = if must_search {
                cdef::choose_superblocks(
                    source,
                    recon,
                    &self.grids.mode_info,
                    params,
                    self.params.cdef_search_row_step.round() as usize,
                    &mut self.cdef_input,
                )
            } else {
                cdef::apply_selected_with_source(
                    recon,
                    &self.grids.mode_info,
                    params,
                    &self.grids.cdef_memory.cached_selection,
                    &mut self.cdef_input,
                );
                self.grids.cdef_memory.cached_selection.clone()
            };
            self.grids
                .cdef_memory
                .cached_selection
                .clone_from(&selected);
            self.finish_cdef_indices(params, &selected);
            return;
        }
        let mut search = None;
        if self.kind == FrameKind::Key || self.is_base_frame {
            let candidate = luma_model;
            let found = cdef::choose_luma(
                source,
                recon,
                &self.grids.mode_info,
                params,
                candidate,
                self.params.cdef_search_row_step.round() as usize,
                &mut self.cdef_input,
            );
            self.grids.cdef_memory.luma = found.strength;
            (params.y_pri, params.y_sec) = found.strength;
            search = Some(found);
        }
        cdef::apply_with_source(
            recon,
            &self.grids.mode_info,
            params,
            search.as_ref(),
            &mut self.cdef_input,
        );
        self.cdef = params;
    }

    /// The chroma CDEF strength of a per-superblock frame: the QP model, or
    /// with `cdef_uv_search_primary` the best of off / model / that primary,
    /// searched with the luma choices and reused between searched frames.
    fn chroma_strength(
        &mut self,
        source: &Picture,
        recon: &Picture,
        params: CdefParams,
        model: (u8, u8),
        searched_frame: bool,
    ) -> (u8, u8) {
        let primary = self.params.cdef_uv_search_primary.round() as u8;
        if primary == 0 {
            return model;
        }
        if searched_frame || self.grids.cdef_memory.uv_choice.is_none() {
            self.grids.cdef_memory.uv_choice = Some(cdef::choose_chroma(
                source,
                recon,
                &self.grids.mode_info,
                params,
                &[model, (primary, model.1)],
                self.params.cdef_search_row_step.round() as usize,
                &mut self.cdef_input,
            ));
        }
        self.grids.cdef_memory.uv_choice.unwrap_or(model)
    }

    /// Uniform choices need no per-block syntax: discard the reserved bits
    /// and use their common strength as the only frame-header strength set.
    fn finish_cdef_indices(&mut self, model: CdefParams, selected: &[bool]) {
        debug_assert_eq!(self.grids.cdef_slots.len(), selected.len());
        let choices = || {
            self.grids
                .cdef_slots
                .iter()
                .zip(selected)
                .filter_map(|(slot, &choice)| slot.map(|_| choice))
        };
        let first = choices().next().unwrap_or(false);
        let uniform = choices().all(|choice| choice == first);
        for (slot, &choice) in self.grids.cdef_slots.iter().zip(selected) {
            if let Some(slot) = slot {
                if uniform {
                    self.writer.omit_literal(*slot);
                } else {
                    self.writer.set_literal(*slot, choice);
                }
            }
        }
        let off = CdefParams {
            y_pri: 0,
            y_sec: 0,
            ..model
        };
        self.cdef = if uniform && first { model } else { off };
        self.cdef_alternate = (!uniform).then_some(model);
    }

    /// Reserve the filter index immediately after `skip_txfm` (AV1 `read_cdef`).
    fn write_cdef_index(&mut self, block: &Block, skip: bool) {
        if self.cdef_frame.per_superblock && !skip {
            let index = block.y / 64 * self.grids.width.div_ceil(64) + block.x / 64;
            if self.grids.cdef_slots[index].is_none() {
                self.grids.cdef_slots[index] = Some(self.writer.reserve_literal());
            }
        }
    }

    /// Quantize the following blocks at `q_idx` (`CurrentQIndex`); chroma
    /// adds the frame's `chroma_delta_q`, clamped as the decoder's
    /// `get_dc_quant` does (AV1 §7.12.2).
    fn set_block_qindex(&mut self, q_idx: u32) {
        self.block_q_idx = q_idx;
        self.steps = QuantSteps::for_qindex(q_idx);
        self.chroma_steps = QuantSteps::for_qindex(
            (q_idx as i32 + self.chroma_delta_q()).clamp(0, QINDEX_MAX as i32) as u32,
        );
    }

    /// Whether the current block is quantised finer than the frame (a
    /// block-QP refinement, `nc_encoder` `reuse.rs`).
    fn coding_from_base_only(&self) -> bool {
        self.spatial
            .as_ref()
            .is_some_and(|sp| sp.current_layer.from_base_only)
    }

    fn refining_block(&self) -> bool {
        self.block_q_idx < self.frame_q_idx
    }

    /// [`Self::set_block_qindex`] where `q_idx` differs from the current
    /// one (most blocks of most frames keep the frame quantizer).
    fn use_block_qindex(&mut self, q_idx: u32) {
        if q_idx != self.block_q_idx {
            self.set_block_qindex(q_idx);
        }
    }

    /// `read_delta_qindex()` of `block`, coded at `q_idx`, after its `skip`
    /// and `cdef_idx` (AV1 §5.11.5, §5.11.12).
    fn write_delta_qindex(&mut self, block: &Block, q_idx: u32, skip: bool) {
        let whole_superblock =
            block.log2_width == SUPERBLOCK_LOG2 && block.log2_height == SUPERBLOCK_LOG2;
        self.delta_q.write(
            &mut self.writer,
            &mut self.cdfs.delta_q,
            q_idx,
            whole_superblock && skip,
        );
    }

    /// The effective offset cached at frame start for quantization and header
    /// syntax, in the range of `delta_q` (su(1+6)).
    fn chroma_delta_q(&self) -> i32 {
        self.frame_chroma_delta_q
    }

    /// This frame's CDEF choices, once its kind and qindex are set: strengths
    /// per superblock when the model's strength is not zero (or strong luma
    /// may apply), and a full search on keys, every local search period,
    /// after a key, on noisy frames, or with nothing to reuse.
    fn cdef_frame_choice(&self, index: u64, previous_was_key: bool) -> CdefFrameChoice {
        let model = cdef::strengths_from_q(
            f32::from(AC_QLOOKUP[self.frame_q_idx as usize]),
            self.kind == FrameKind::Key,
        )
        .0;
        let per_superblock = self.random.is_some()
            || (self.params.cdef_per_superblock >= 0.5
                && (model != (0, 0)
                    || (self.params.cdef_local_strong_luma >= 0.5
                        && self.kind == FrameKind::Inter)));
        let local_period = self.params.cdef_local_search_period.round() as u64;
        let noise_limit = self.params.cdef_local_noise_full_search_sigma;
        let search = self.kind == FrameKind::Key
            || index.is_multiple_of(local_period.max(1))
            || previous_was_key
            || (noise_limit > 0.0 && self.noise_sigma >= noise_limit)
            || self.grids.cdef_memory.cached_selection.is_empty();
        CdefFrameChoice {
            per_superblock,
            search,
        }
    }

    /// Fixed-QP key-frame quantiser step relative to the inter step.
    fn key_step_scale(&self) -> f64 {
        f64::from(self.params.keyframe_step_permille) / KEY_STEP_SCALE_DENOMINATOR
    }

    fn is_secondary(&self) -> bool {
        matches!(self.active_ref, ReferenceChoice::Secondary)
    }

    /// Whether this frame enables per-block compound single/dual-reference
    /// selection (`reference_select = 1` in the frame header, AV1 §5.9.23).
    fn frame_reference_select(&self) -> bool {
        self.references.secondary.is_some()
            && self.kind == FrameKind::Inter
            && self.params.compound >= 0.5
    }

    fn single_ref_bits(
        &self,
        above: Option<ModeInfo>,
        left: Option<ModeInfo>,
        secondary_ref: bool,
    ) -> f32 {
        let comp_flag = if self.frame_reference_select() {
            self.cdfs.comp_mode[comp_mode_context(above, left)].bits(0)
        } else {
            0.0
        };
        comp_flag
            + ref_frame_bits(
                &self.cdfs,
                self.single_ref_contexts(above, left),
                secondary_ref,
            )
    }

    /// `(near, p4)` contexts of the single-reference symbols. Without a
    /// `LAST2_FRAME` reference this frame every inter neighbour uses
    /// `LAST_FRAME`, and libaom's contexts reduce to one (the single-layer
    /// path does no counting).
    fn single_ref_contexts(
        &self,
        above: Option<ModeInfo>,
        left: Option<ModeInfo>,
    ) -> (usize, usize) {
        if self.references.secondary.is_none() {
            let ctx = last_only_ref_context(above, left);
            return (ctx, ctx);
        }
        let (near_ctx, p4_ctx, _) = ref_count_contexts(above, left);
        (near_ctx, p4_ctx)
    }

    /// The single-reference list of `block` for `LAST2_FRAME` when
    /// `secondary_ref`, else `LAST_FRAME`. A frame without `LAST2_FRAME`
    /// scans as it did before spatial SVC ([`LastOnly`]).
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn single_ref_stack(&self, block: &Block, secondary_ref: bool) -> MvStack {
        let (grid, x, y, log2) = (&self.grids.mode_info, block.x, block.y, block.log2_size());
        if self.references.secondary.is_none() {
            find_mv_stack(grid, x, y, log2, LastOnly)
        } else {
            find_mv_stack(
                grid,
                x,
                y,
                log2,
                SingleRef {
                    last2: secondary_ref,
                },
            )
        }
    }

    fn active_slot(&self) -> usize {
        if self.is_secondary() {
            self.references
                .secondary
                .expect("secondary reference present")
                .slot
        } else {
            self.references.reference.slot
        }
    }

    fn switch_spatial_layer(&mut self, layer: SpatialLayerFrame) {
        let Some(mut sp) = self.spatial.take() else {
            return;
        };
        let target = usize::from(layer.spatial_id);
        if sp.active_spatial_id != target {
            let next = sp.parked[target]
                .take()
                .expect("target spatial layer grids parked");
            let prev = std::mem::replace(&mut self.grids, next);
            sp.parked[sp.active_spatial_id] = Some(prev);
            sp.active_spatial_id = target;
        }
        sp.current_layer = layer;
        self.spatial = Some(sp);
    }

    fn write_spatial_unit(&self, frame_bytes: &[u8]) -> Vec<u8> {
        let sp = self.spatial.as_ref().expect("spatial mode active");
        let layer = sp.current_layer;
        let mut unit = Vec::with_capacity(frame_bytes.len() + 64);
        if layer.spatial_id == 0 {
            write_obu_layer(&mut unit, ObuType::TemporalDelimiter, None, &[]);
            if layer.key_unit {
                let fixed_q =
                    (self.quantizer_mode == QuantizerMode::Fixed).then_some(self.base_q_idx);
                let seq = spatial_sequence_header(
                    &sp.config,
                    self.signal,
                    self.frame_rate,
                    SequenceTools {
                        filter_intra: self.filter_intra_enabled(),
                        film_grain: self.grain.is_some(),
                        superres: self.superres.is_some(),
                        restoration: RestorationTool::Disabled,
                        frame_ids: None,
                    },
                    fixed_q,
                    layer.max_bitrate_bps,
                    self.random.is_some(),
                );
                write_obu_layer(&mut unit, ObuType::SequenceHeader, None, &seq);
            }
        }
        write_obu_layer(
            &mut unit,
            ObuType::Frame,
            Some((layer.temporal_id, layer.spatial_id)),
            frame_bytes,
        );
        unit
    }

    /// The reference list of `block`, built once per block and kept until
    /// the block is coded.
    fn stack_for(&mut self, block: &Block) -> MvStack {
        let secondary_ref = self.is_secondary();
        if let Some(c) = self
            .current
            .filter(|c| (c.x, c.y) == (block.x, block.y) && c.secondary_ref == secondary_ref)
        {
            return c.stack;
        }
        let stack = self.single_ref_stack(block, secondary_ref);
        let (above, left) = self.neighbours(block);
        let inter_base_bits = self.cdfs.intra_inter[intra_inter_context(above, left)].bits(1)
            + self.single_ref_bits(above, left, secondary_ref);
        self.current = Some(CurrentBlock {
            x: block.x,
            y: block.y,
            secondary_ref,
            stack,
            inter_base_bits,
        });
        stack
    }

    /// The mode info above and to the left of `block`, where they exist in
    /// the tile (AV1 `AvailU` / `AvailL`).
    fn neighbours(&self, block: &Block) -> (Option<ModeInfo>, Option<ModeInfo>) {
        let (mi_row, mi_col) = (block.y >> MI_LOG2, block.x >> MI_LOG2);
        (
            (mi_row > 0).then(|| *self.grids.mode_info.at(mi_row - 1, mi_col)),
            (mi_col > 0).then(|| *self.grids.mode_info.at(mi_row, mi_col - 1)),
        )
    }

    /// Key-frame mode CDF contexts come from the reconstructed neighbours'
    /// actual luma modes (AV1 `Intra_Mode_Context`).
    fn key_mode_context(&self, block: &Block) -> (usize, usize) {
        let (above, left) = self.neighbours(block);
        // Every block of a key frame is intra: the detail byte is the mode.
        let context = |info: Option<ModeInfo>| {
            info.map_or(0, |info| {
                usize::from(INTRA_MODE_CONTEXT[usize::from(info.mode_detail)])
            })
        };
        (context(above), context(left))
    }

    /// One plane's share of [`CodecBackend::intra_leaf_cost`]: DC prediction
    /// from the *source* neighbours (at key-frame quality they are close to
    /// the reconstruction, which does not exist yet), then
    /// [`Self::residual_rd_cost`].
    fn intra_plane_cost(&mut self, source: &Picture, block: &Block, plane: PlaneIndex) -> f32 {
        let shift = u8::from(plane != PlaneIndex::Y);
        let (x, y) = (block.x >> shift, block.y >> shift);
        let n = block.size() >> shift;
        let src_plane = source.plane(plane);
        let dc = dc_value(src_plane, x, y, n, y > 0, x > 0);
        let src = src_plane
            .region(x as isize, y as isize, n, n)
            .expect("block inside source");
        let flat = &mut self.scratch.flat_prediction[..n * n];
        flat.fill(dc);
        let pred = PlaneView::new(flat, n, n, n).expect("flat prediction is tightly packed");
        let size = BlockSize::new(n, n).expect("square AV1 block size");
        self.dsp
            .residual(size, &src, &pred, &mut self.scratch.residual, n);
        let sse = u64::from(self.dsp.variance(size, &src, &pred).sse);
        let shift = u8::from(plane != PlaneIndex::Y);
        let tx = TxSize::from_log2(block.log2_size() - shift).expect("supported transform size");
        self.residual_rd_cost(tx, plane, false, DC_PRED, sse).0
    }

    /// Estimated cost of coding the residual in `scratch.residual` (row-major,
    /// block-sized; `sse` its energy) as one transform block of `plane` of an
    /// intra block predicted with `intra_mode` (its symbol): forward transform and quantisation as `code_plane` does them,
    /// the coefficient bits (current CDFs, neighbour contexts taken as empty)
    /// and the remaining distortion estimated in the transform domain. That
    /// estimate keeps the energy a 64x64 transform cannot code
    /// (`coefficient_gain`), which keeps text and edges out of 64x64 leaves.
    fn residual_rd_cost(
        &mut self,
        tx: TxSize,
        plane: PlaneIndex,
        is_inter: bool,
        intra_mode: usize,
        sse: u64,
    ) -> (f32, u32) {
        let n = tx.size();
        let coded_area = tx.coded_size() * tx.coded_size();
        let quantizer = self.plane_quantizer(plane, tx);
        let policy = self.forward_policy(false);
        let scratch = &mut self.scratch;
        let levels = &mut scratch.levels[plane as usize][..coded_area];
        if scratch.original_coefficients.is_empty() {
            scratch.original_coefficients.resize(MAX_CODED_AREA, 0);
        }
        // The RD cost keeps the unquantised coefficients: transform into
        // them and quantise out of place (no copy).
        let original = &mut scratch.original_coefficients[..coded_area];
        forward_dct_with(&self.dsp, policy, tx, &scratch.residual, n, original);
        let nonzero = quantize_coded_from(&self.dsp, policy, tx, original, &quantizer, levels);
        let eob = quantized_eob(tx, levels, nonzero, policy);
        let plane_type = if plane == PlaneIndex::Y {
            PlaneType::Luma
        } else {
            PlaneType::Chroma
        };
        let ctx = TxbContext::derive(plane_type, true, &[], &[]);
        let bits = self
            .cdfs
            .coeff
            .cost(tx, plane_type, is_inter, intra_mode, ctx, levels, eob);
        // Only the levels the EOB leaves room for can be nonzero: the others
        // dequantise to zero and add nothing to the gain.
        let dequant_area = if eob == 1 {
            1
        } else {
            dequantized_extent(tx, eob)
        };
        let gain = if eob == 0 {
            0.0
        } else {
            let dequant = &mut scratch.dequant[..dequant_area];
            self.dsp
                .av1_dequantize(&levels[..dequant_area], &quantizer, dequant);
            crate::transform::coefficient_gain(&self.dsp, tx, &original[..dequant_area], dequant)
        };
        (
            (sse as f64 - gain).max(0.0) as f32 + self.lambda() * bits,
            nonzero,
        )
    }

    /// Coefficients and entropy contexts of the three planes (luma leaf by
    /// leaf where its transform tree split; `luma_leaves` is empty
    /// otherwise).
    fn write_residuals(
        &mut self,
        block: &Block,
        residuals: &[PlaneResidual; 3],
        skip: bool,
        luma_leaves: &[LumaLeaf],
        info: &ModeInfo,
        intra_mode: LumaIntra,
    ) {
        for (plane, residual) in residuals.iter().enumerate() {
            let tx = residual.tx;
            let (x4, y4, units) = Self::plane_units(block, plane);
            if plane == 0 && !luma_leaves.is_empty() {
                self.write_luma_tree(block, luma_leaves);
                continue;
            }
            if !skip {
                let plane_type = if plane == 0 {
                    PlaneType::Luma
                } else {
                    PlaneType::Chroma
                };
                let (above_ctx, left_ctx) = self.context_slices(plane, x4, y4, units);
                // The transform always covers the whole plane block.
                let ctx = TxbContext::derive(plane_type, true, above_ctx, left_ctx);
                self.cdfs.coeff.write(
                    &mut self.writer,
                    tx,
                    plane_type,
                    info.is_inter(),
                    intra_mode.intra_dir(),
                    ctx,
                    &self.scratch.levels[plane][..tx.coded_size() * tx.coded_size()],
                    residual.eob,
                );
            }
            let value = if skip {
                0
            } else {
                entropy_context(tx, &self.scratch.levels[plane], residual.eob)
            };
            self.set_contexts(plane, x4, y4, units, value);
        }
    }

    /// Store the coded block's mode info, with each leaf's transform size
    /// where its transform tree split.
    fn record_block(&mut self, block: &Block, info: ModeInfo, tx_split: TxSplit) {
        self.grids
            .mode_info
            .fill(block.y >> MI_LOG2, block.x >> MI_LOG2, info);
        for &(dx, dy, tx_log2) in &tx_split.leaves(block.log2_size()) {
            if tx_log2 != block.log2_size() {
                let leaf = ModeInfo { tx_log2, ..info };
                let (row, col) = ((block.y + dy) >> MI_LOG2, (block.x + dx) >> MI_LOG2);
                self.grids
                    .mode_info
                    .fill_square(row, col, 1 << (tx_log2 - MI_LOG2 as u8), leaf);
            }
        }
    }

    /// Code the luma of `block`: one transform, or for an inter block the
    /// transform tree [`Self::choose_tx_split`] picks (its leaves are then
    /// returned, and the returned residual only says whether any is coded).
    fn code_luma(
        &mut self,
        block: &Block,
        prediction: &Prediction<'_>,
        is_inter: bool,
        source: &Picture,
        recon: &mut Picture,
    ) -> (PlaneResidual, TxSplit, Vec<LumaLeaf>) {
        let split = if is_inter {
            self.choose_tx_split(block, prediction, source, recon)
        } else {
            TxSplit::NONE
        };
        if !split.split {
            let luma = self.code_plane(block, PlaneIndex::Y, prediction, source, recon);
            return (luma, split, Vec::new());
        }
        let leaves = self.code_luma_tree(block, prediction, source, recon, split);
        let tx = TxSize::from_log2(block.log2_size()).expect("supported transform size");
        let mut luma = PlaneResidual::uncoded(tx);
        luma.eob = usize::from(leaves.iter().any(|leaf| leaf.eob > 0));
        (luma, split, leaves)
    }

    /// The inter transform tree of `block` (AV1 §5.11.17), the single
    /// decision point for it: one level of split where four quarter
    /// transforms beat the block transform on the estimated cost (current
    /// CDF coefficient bits plus the distortion left after quantisation, as
    /// [`Self::residual_rd_cost`] estimates it) plus the split flag's bits.
    /// The largest quarter's prediction error exceeds
    /// `inter_tx_split_concentration` times the quarters' mean.
    fn error_concentrated(&self, n: usize, src: &PlaneView<'_>, pred: &PlaneView<'_>) -> bool {
        let half = n / 2;
        let quarter = BlockSize::new(half, half).expect("square AV1 block size");
        let quarters: [u32; 4] = std::array::from_fn(|i| {
            let (qx, qy) = ((i % 2) * half, (i / 2) * half);
            let s = src
                .sub_view(qx, qy, half, half)
                .expect("quarter inside block");
            let p = pred
                .sub_view(qx, qy, half, half)
                .expect("quarter inside block");
            self.dsp.variance(quarter, &s, &p).sse
        });
        let total: u64 = quarters.iter().map(|&q| u64::from(q)).sum();
        let largest = quarters.iter().copied().max().unwrap_or(0);
        f64::from(largest) * 4.0
            > f64::from(self.params.inter_tx_split_concentration) * total as f64
    }

    /// Past qindex 255 (`lambda_scale` > 1) rate control wants a coarser
    /// step than AV1 has, and `TX_64X64` is the size that cannot follow:
    /// its dequantiser divides by 4 against `TX_32X32`'s 2 (§7.12.3), so at
    /// qindex 255 one level is 1828/512 = 3.6 sample values of basis
    /// amplitude, a `TX_32X32` level twice that. The 180p and 360p encodes of
    /// the same `CrowdRun` at 100 kbps quantise at about 6.3 (`TX_16X16` at
    /// qindex 212); 720p `TX_64X64` at 3.6 skipped 83% of the picture
    /// (stale patches). Four `TX_32X32`s at 7.1 code the same low
    /// frequencies coarser. Splitting from just past the ceiling (720p
    /// AV1, 100 kbps, PSNR-Y / MS-SSIM): `CrowdRun` 20.00/.584 →
    /// 20.22/.609, `ParkJoy` 19.75/.600 → 19.82/.609, but `FoodMarket`
    /// 26.83/.769 → 26.83/.767 and VMAF −1.6, where the planned step is
    /// barely past the largest and `TX_32X32` is too coarse. From twice the
    /// λ (√2 the step) `FoodMarket` is unchanged and `CrowdRun` keeps the gain.
    fn split_past_ceiling(&self, block: &Block) -> Option<TxSplit> {
        (self.params.ceiling_tx_split > 0.0
            && self.lambda_scale >= self.params.ceiling_tx_split
            && block.log2_size() == CEILING_SPLIT_LOG2)
            .then_some(TxSplit {
                split: true,
                quadrants: [false; 4],
            })
    }

    /// The transform split of a block when the split search is off: a
    /// refined 64x64 block codes four 32x32 transforms, because the 64x64
    /// transform codes only its low 32x32 frequencies (AV1 §7.13.3) and
    /// cannot restore the detail the refinement is for (a still region
    /// copied from a starved key stayed 9 dB short for good:
    /// `nc-cli/tests/static_recovery.rs`).
    fn refined_tx_split(&self, block: &Block) -> TxSplit {
        if self.refining_block() && block.log2_size() == CEILING_SPLIT_LOG2 {
            TxSplit {
                split: true,
                quadrants: [false; 4],
            }
        } else {
            TxSplit::NONE
        }
    }

    /// A split decided without the search: the stress test's draw, the
    /// refined-block rule when the search is off, the ceiling split.
    fn decided_tx_split(&mut self, block: &Block) -> Option<TxSplit> {
        if let Some(random) = self.random.as_mut() {
            return Some(random.tx_split());
        }
        if self.params.inter_tx_split < 0.5 {
            return Some(self.refined_tx_split(block));
        }
        self.split_past_ceiling(block)
    }

    fn choose_tx_split(
        &mut self,
        block: &Block,
        prediction: &Prediction<'_>,
        source: &Picture,
        recon: &mut Picture,
    ) -> TxSplit {
        if self.tx_mode != TxMode::Select || block.log2_size() < MIN_TX_SPLIT_LOG2 {
            return TxSplit::NONE;
        }
        if let Some(split) = self.decided_tx_split(block) {
            return split;
        }
        let n = block.size();
        self.predict_plane(block, PlaneIndex::Y, prediction, recon);
        let size = BlockSize::new(n, n).expect("square AV1 block size");
        let src = source
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, n, n)
            .expect("block inside source");
        let pred = recon
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, n, n)
            .expect("block inside reconstruction");
        // Blocks whose prediction error is below `inter_tx_split_energy` · λ
        // per sample keep the block transform without trial transforms:
        // their residual quantises to little or nothing at any size.
        let sse = self.dsp.variance(size, &src, &pred).sse;
        if (sse as f32) < self.params.inter_tx_split_energy * self.lambda() * (n * n) as f32 {
            return TxSplit::NONE;
        }
        // A block whose luma the unsplit path drops before the transform
        // (`code_plane`'s light-PD1 skip) keeps one transform: a split would
        // code a residual the encoder has already decided is not worth
        // coding (static_otw_homes390: 4.45x P-frame bits for +0.6 dB).
        if let Prediction::Inter { mv, .. } = prediction
            && self.pre_tx_skip_enabled()
            && self.skip_luma_before_transform(block, *mv, sse, &src, &pred)
        {
            return TxSplit::NONE;
        }
        // Splitting pays where the error is concentrated in part of the
        // block: quarter transforms code that part and leave the rest
        // empty. With `inter_tx_split_concentration` > 0 only blocks whose
        // largest quarter error exceeds that multiple of the quarters' mean
        // get the trial transforms.
        if self.params.inter_tx_split_concentration > 0.0
            && !self.error_concentrated(n, &src, &pred)
        {
            return TxSplit::NONE;
        }
        self.dsp
            .residual(size, &src, &pred, &mut self.scratch.tree_residual, n);
        let (whole, whole_nonzero) = self.tree_transform_cost(n, (0, 0), block.log2_size());
        // `code_plane` reuses these levels if the block keeps one transform.
        let coded = TxSize::from_log2(block.log2_size())
            .expect("supported transform size")
            .coded_size();
        let originals = self.trial_coefficients_reusable();
        let scratch = &mut self.scratch;
        scratch.intra_levels[..coded * coded].copy_from_slice(&scratch.levels[0][..coded * coded]);
        if originals {
            scratch.intra_originals[..coded * coded]
                .copy_from_slice(&scratch.original_coefficients[..coded * coded]);
        }
        scratch.cached_intra = Some(CachedIntraLevels {
            block: *block,
            mode: None,
            nonzero: whole_nonzero,
            originals,
        });
        // Where the block transform quantises to nothing, the quarters are not
        // tried: a split only pays for coefficients the block transform
        // cannot code compactly, and most inter blocks have none (skips
        // four trial transforms on 80-90% of P blocks).
        if whole_nonzero == 0 {
            return TxSplit::NONE;
        }
        let half = block.log2_size() - 1;
        let step = 1 << half;
        let quarter_area = {
            let coded = TxSize::from_log2(half)
                .expect("supported transform size")
                .coded_size();
            coded * coded
        };
        let mut quarters = 0.0;
        let mut nonzero = [0_u32; 4];
        for (i, count) in nonzero.iter_mut().enumerate() {
            let (cost, n_quarter) =
                self.tree_transform_cost(n, ((i % 2) * step, (i / 2) * step), half);
            quarters += cost;
            *count = n_quarter;
            // `code_luma_tree` reuses each quarter's levels (and, for the
            // trellis, coefficients) if the split wins.
            let scratch = &mut self.scratch;
            let quarter = i * quarter_area..(i + 1) * quarter_area;
            scratch.tree_levels[quarter.clone()]
                .copy_from_slice(&scratch.levels[0][..quarter_area]);
            if originals {
                scratch.tree_originals[quarter]
                    .copy_from_slice(&scratch.original_coefficients[..quarter_area]);
            }
        }
        self.scratch.tree_trial = Some((*block, nonzero, originals));
        let ctx = txsize::split_context(
            &self.grids.txfm,
            block.x >> MI_LOG2,
            block.y >> MI_LOG2,
            block.log2_size(),
            block.log2_size(),
        );
        let flag = &self.cdfs.txfm_split[ctx];
        let split_bits = flag.bits(1) - flag.bits(0);
        // The unsplit block may also drop its residual (distortion = the
        // prediction SSE, as the residual RD checks in `code_plane` price
        // it), so the split must beat the cheaper of coding and not coding.
        let unsplit = whole.min(sse as f32);
        if quarters + self.lambda() * split_bits < unsplit {
            TxSplit {
                split: true,
                quadrants: [false; 4],
            }
        } else {
            TxSplit::NONE
        }
    }

    /// Estimated cost (and nonzero count) of the `tx_log2` inter luma
    /// transform at `offset` in the block residual (`scratch.tree_residual`,
    /// row stride `stride`).
    fn tree_transform_cost(
        &mut self,
        stride: usize,
        (x, y): (usize, usize),
        tx_log2: u8,
    ) -> (f32, u32) {
        let tx = TxSize::from_log2(tx_log2).expect("supported transform size");
        let t = tx.size();
        let mut sse = 0_u64;
        for row in 0..t {
            let from = &self.scratch.tree_residual[(y + row) * stride + x..][..t];
            self.scratch.residual[row * t..][..t].copy_from_slice(from);
            sse += from
                .iter()
                .map(|&r| u64::from((i32::from(r) * i32::from(r)).unsigned_abs()))
                .sum::<u64>();
        }
        self.residual_rd_cost(tx, PlaneIndex::Y, true, DC_PRED, sse)
    }

    /// Predict, transform, quantise and reconstruct the luma of an inter
    /// block coded with the transform tree `split`; the leaves' levels go
    /// to `scratch.tree_levels` in coding order.
    fn code_luma_tree(
        &mut self,
        block: &Block,
        prediction: &Prediction<'_>,
        source: &Picture,
        recon: &mut Picture,
        split: TxSplit,
    ) -> Vec<LumaLeaf> {
        self.predict_plane(block, PlaneIndex::Y, prediction, recon);
        let policy = self.forward_policy(true);
        // Quarter levels the split decision already quantised, if this tree
        // is exactly its one-level split.
        let trial = self
            .scratch
            .tree_trial
            .take()
            .filter(|(b, _, _)| *b == *block && split.quadrants == [false; 4])
            // The trellis needs the unquantised coefficients.
            .filter(|&(_, _, originals)| originals || self.trellis.is_none());
        let mut offset = 0;
        let mut leaves = Vec::with_capacity(4);
        for (index, (dx, dy, log2)) in split.leaves(block.log2_size()).into_iter().enumerate() {
            let tx = TxSize::from_log2(log2).expect("supported transform size");
            let t = tx.size();
            let coded_area = tx.coded_size() * tx.coded_size();
            let (x, y) = ((block.x + dx) as isize, (block.y + dy) as isize);
            let size = BlockSize::new(t, t).expect("square transform size");
            let src = source
                .plane(PlaneIndex::Y)
                .region(x, y, t, t)
                .expect("transform inside source");
            let mut dst = recon
                .plane_mut(PlaneIndex::Y)
                .region_mut(x, y, t, t)
                .expect("transform inside reconstruction");
            let leaf = Block::square(block.x + dx, block.y + dy, log2);
            let trellis = self.trellis_block(&leaf, PlaneIndex::Y, prediction, false);
            let scratch = &mut self.scratch;
            let quantizer = self.steps.quantizer(tx, self.rounding_q7);
            let nonzero = if let Some((_, counts, _)) = trial {
                match trellis {
                    // The split decision's coefficients (a fresh transform
                    // of the same residual under the same forward policy),
                    // quantised by the trellis.
                    Some(leaf_trellis) => trellis_quantize(
                        &self.dsp,
                        (&mut self.cdfs.coeff, &self.grids.contexts),
                        &mut scratch.trellis,
                        leaf_trellis,
                        tx,
                        policy,
                        &quantizer,
                        &scratch.tree_originals[offset..offset + coded_area],
                        &mut scratch.tree_levels[offset..offset + coded_area],
                    ),
                    None => counts[index],
                }
            } else {
                self.dsp
                    .residual(size, &src, &dst.as_view(), &mut scratch.residual, t);
                let levels = &mut scratch.tree_levels[offset..offset + coded_area];
                if let Some(leaf_trellis) = trellis {
                    if scratch.original_coefficients.is_empty() {
                        scratch.original_coefficients.resize(MAX_CODED_AREA, 0);
                    }
                    let original = &mut scratch.original_coefficients[..coded_area];
                    forward_dct_with(&self.dsp, policy, tx, &scratch.residual, t, original);
                    trellis_quantize(
                        &self.dsp,
                        (&mut self.cdfs.coeff, &self.grids.contexts),
                        &mut scratch.trellis,
                        leaf_trellis,
                        tx,
                        policy,
                        &quantizer,
                        original,
                        levels,
                    )
                } else {
                    forward_dct_with(&self.dsp, policy, tx, &scratch.residual, t, levels);
                    quantize_coded(&self.dsp, policy, tx, levels, &quantizer)
                }
            };
            let levels = &mut scratch.tree_levels[offset..offset + coded_area];
            let mut eob = quantized_eob(tx, levels, nonzero, policy);
            if eob > 0 {
                eob = mds3::shave_trailing_ones(
                    coefficient_scan(tx),
                    levels,
                    eob,
                    self.shave_min_zero_run,
                );
            }
            if eob > 0 {
                let area = dequantized_extent(tx, eob);
                let dequant = &mut scratch.dequant[..coded_area];
                self.dsp
                    .av1_dequantize(&levels[..area], &quantizer, &mut dequant[..area]);
                inverse_dct_add_eob(&self.dsp, tx, dequant, eob, &mut dst);
            }
            leaves.push(LumaLeaf {
                dx,
                dy,
                log2,
                eob,
                offset,
            });
            offset += coded_area;
        }
        leaves
    }

    /// Transform-size syntax of a block in a `TX_MODE_SELECT` frame (AV1
    /// §5.11.15–17) and its transform-size contexts: an inter block's tree,
    /// an intra block's `tx_depth` (always 0: the block transform), nothing
    /// for a skipped inter block.
    fn write_tx_size(
        &mut self,
        block: &Block,
        is_inter: bool,
        skip: bool,
        split: TxSplit,
        above: Option<ModeInfo>,
        left: Option<ModeInfo>,
    ) {
        let (mi_col, mi_row) = (block.x >> MI_LOG2, block.y >> MI_LOG2);
        let side_mi = block.size() >> MI_LOG2;
        let side = block.size() as u8;
        if is_inter && skip {
            self.grids.txfm.fill(mi_col, mi_row, side_mi, side);
            return;
        }
        if !is_inter {
            let category = txsize::tx_size_category(block.log2_size());
            let ctx = txsize::tx_size_context(
                &self.grids.txfm,
                above,
                left,
                mi_col,
                mi_row,
                block.log2_size(),
            );
            self.writer.write_symbol(
                0,
                &mut self.cdfs.tx_depth[category][ctx],
                txsize::tx_depth_symbols(block.log2_size()),
            );
            self.grids.txfm.fill(mi_col, mi_row, side_mi, side);
            return;
        }
        let ctx = txsize::split_context(
            &self.grids.txfm,
            mi_col,
            mi_row,
            block.log2_size(),
            block.log2_size(),
        );
        self.writer
            .write_bool(split.split, &mut self.cdfs.txfm_split[ctx]);
        if !split.split {
            self.grids.txfm.fill(mi_col, mi_row, side_mi, side);
            return;
        }
        let half = block.log2_size() - 1;
        let half_mi = side_mi / 2;
        for (i, &quadrant) in split.quadrants.iter().enumerate() {
            let (col, row) = (mi_col + (i % 2) * half_mi, mi_row + (i / 2) * half_mi);
            if TxSplit::flag_coded(1, half) {
                let ctx =
                    txsize::split_context(&self.grids.txfm, col, row, block.log2_size(), half);
                self.writer
                    .write_bool(quadrant, &mut self.cdfs.txfm_split[ctx]);
            }
            if quadrant && TxSplit::flag_coded(1, half) {
                let quarter_mi = half_mi / 2;
                for j in 0..4 {
                    let (c, r) = (col + (j % 2) * quarter_mi, row + (j / 2) * quarter_mi);
                    self.grids
                        .txfm
                        .fill(c, r, quarter_mi.max(1), 1 << (half - 1));
                }
            } else {
                self.grids.txfm.fill(col, row, half_mi, 1 << half);
            }
        }
    }

    /// Coefficients of a split luma transform tree, leaf by leaf, each with
    /// its own neighbour contexts (AV1 §5.11.35 `transform_tree`).
    fn write_luma_tree(&mut self, block: &Block, leaves: &[LumaLeaf]) {
        for leaf in leaves {
            let tx = TxSize::from_log2(leaf.log2).expect("supported transform size");
            let coded_area = tx.coded_size() * tx.coded_size();
            let units = tx.size() >> MI_LOG2;
            let x4 = (block.x + leaf.dx) >> MI_LOG2;
            let y4 = ((block.y + leaf.dy) >> MI_LOG2) & (SB_MI_UNITS - 1);
            let (above, left) = self.context_slices(0, x4, y4, units);
            let ctx = TxbContext::derive(PlaneType::Luma, false, above, left);
            let levels = &self.scratch.tree_levels[leaf.offset..leaf.offset + coded_area];
            self.cdfs.coeff.write(
                &mut self.writer,
                tx,
                PlaneType::Luma,
                true,
                DC_PRED,
                ctx,
                levels,
                leaf.eob,
            );
            let value = entropy_context(tx, levels, leaf.eob);
            self.set_contexts(0, x4, y4, units, value);
        }
    }

    /// Prediction-SSE search with syntax pricing before either chroma transform.
    /// Restrict this optional tool to key frames until its amortised CPU and
    /// both quality metrics have been measured.
    #[inline]
    /// The luma mode of an intra block: the key-frame search, DC in inter
    /// frames, or a random one in the syntax stress test.
    fn choose_intra_mode(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
    ) -> LumaIntra {
        let filter_intra = self.filter_intra_enabled();
        match self.random.as_mut() {
            Some(random) => random.luma_intra(block.log2_size(), filter_intra),
            None if self.kind == FrameKind::Key => self.choose_key_intra_mode(block, source, recon),
            None => LumaIntra::DC,
        }
    }

    /// `CfL` for a block whose luma is reconstructed: key-frame intra blocks
    /// where it pays, or random slopes in the syntax stress test.
    fn choose_cfl(
        &mut self,
        block: &Block,
        mode: IntraMode,
        intra: bool,
        source: &Picture,
        recon: &Picture,
    ) -> Option<ChromaFromLuma> {
        if self.random.is_some() {
            self.random_cfl(block, intra, source, recon)
        } else {
            self.choose_key_cfl(block, mode, source, recon)
        }
    }

    /// How an inter block signals `mv`: the cheapest mode, or a random legal
    /// one in the syntax stress test.
    fn choose_inter_mode(&mut self, stack: &MvStack, mv: Av1Mv) -> InterMode {
        match self.random.as_mut() {
            Some(random) => random.inter_mode(stack, mv),
            None => inter_mode_for(stack, mv),
        }
    }

    /// How a compound inter block signals `mvs`: the cheapest mode, or a
    /// random legal one in the syntax stress test.
    fn choose_compound_inter_mode(
        &mut self,
        stack: &CompoundMvStack,
        mvs: [Av1Mv; 2],
    ) -> CompoundInterMode {
        match self.random.as_mut() {
            Some(random) => random.compound_mode(stack, mvs),
            None => compound_inter_mode_for(stack, mvs),
        }
    }

    fn choose_key_cfl(
        &self,
        block: &Block,
        mode: IntraMode,
        source: &Picture,
        recon: &Picture,
    ) -> Option<ChromaFromLuma> {
        if self.kind != FrameKind::Key || self.params.keyframe_cfl < 0.5 || block.log2_size() > 5 {
            return None;
        }
        let cfl = ChromaFromLuma::search(block, source, recon);
        if cfl.alpha == [0, 0] {
            return None;
        }
        let uv = &self.cdfs.uv_mode[CFL_ALLOWED][mode.symbol()];
        let mut bits =
            uv.bits(UV_CFL_PRED) - uv.bits(UV_DC_PRED) + self.cdfs.cfl_sign.bits(cfl.joint_sign());
        for (p, &alpha) in cfl.alpha.iter().enumerate() {
            if alpha != 0 {
                bits += self.cdfs.cfl_alpha[cfl.alpha_context(p)]
                    .bits(usize::from(alpha.unsigned_abs() - 1));
            }
        }
        (cfl.improvement as f32 > self.lambda() * self.params.intra_rate_scale * bits)
            .then_some(cfl)
    }

    /// Key-frame luma mode search: each candidate's prediction is costed by
    /// [`Self::residual_rd_cost`] plus λ · its mode bits; transform and
    /// reconstruction then run once for the winner. Ranking by prediction
    /// SATD instead picked SMOOTH/PAETH where their residual cost more to code
    /// and lost more to the quantiser than DC's (`bodycam_airport` key, planned
    /// partition: +7% bytes at −1.6 dB luma with the non-directional set), so
    /// SATD only shortlists the fuller set's extra candidates
    /// ([`Self::key_intra_shortlist`]).
    fn choose_key_intra_mode(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
    ) -> LumaIntra {
        let set = IntraModeSet::from_param(self.params.keyframe_intra_modes);
        let shortlist = self.key_intra_shortlist(block, source, recon, set.modes());
        if set == IntraModeSet::DcOnly && shortlist.is_empty() {
            return LumaIntra::DC;
        }
        let candidates: Vec<LumaIntra> = set
            .modes()
            .iter()
            .map(|&mode| LumaIntra::plain(mode))
            .chain(shortlist)
            .collect();
        self.rd_best_luma(block, source, recon, &candidates, Self::key_mode_bits)
            .1
    }

    /// The cheapest of `candidates` for the luma of intra `block` and its
    /// cost: each prediction costed by [`Self::residual_rd_cost`] plus λ ·
    /// its `mode_bits`. The winner's levels are cached for `code_plane`.
    fn rd_best_luma(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
        candidates: &[LumaIntra],
        mode_bits: impl Fn(&Self, &Block, LumaIntra) -> f32,
    ) -> (f32, LumaIntra) {
        let lambda = self.lambda() * self.params.intra_rate_scale;
        let tx = TxSize::from_log2(block.log2_size()).expect("supported transform size");
        let mut best = (f32::INFINITY, LumaIntra::DC);
        for &luma in candidates {
            self.predict_plane(block, PlaneIndex::Y, &Prediction::Intra(luma), recon);
            let n = block.size();
            let size = BlockSize::new(n, n).expect("intra blocks are kernel sizes");
            let src = source
                .plane(PlaneIndex::Y)
                .region(block.x as isize, block.y as isize, n, n)
                .expect("block inside source");
            let pred = recon
                .plane(PlaneIndex::Y)
                .region(block.x as isize, block.y as isize, n, n)
                .expect("block inside reconstruction");
            self.dsp
                .residual(size, &src, &pred, &mut self.scratch.residual, n);
            let sse = u64::from(self.dsp.variance(size, &src, &pred).sse);
            let (coefficient_cost, nonzero) =
                self.residual_rd_cost(tx, PlaneIndex::Y, false, luma.intra_dir(), sse);
            let cost = coefficient_cost + lambda * mode_bits(self, block, luma);
            if cost < best.0 {
                best = (cost, luma);
                // `code_plane` reuses the winner's levels instead of
                // transforming and quantising the same residual again.
                let coded = tx.coded_size();
                let originals = self.trial_coefficients_reusable();
                let scratch = &mut self.scratch;
                scratch.intra_levels[..coded * coded]
                    .copy_from_slice(&scratch.levels[0][..coded * coded]);
                if originals {
                    scratch.intra_originals[..coded * coded]
                        .copy_from_slice(&scratch.original_coefficients[..coded * coded]);
                }
                scratch.cached_intra = Some(CachedIntraLevels {
                    block: *block,
                    mode: Some(luma),
                    nonzero,
                    originals,
                });
            }
        }
        best
    }

    /// The fuller key-frame set's extra candidates worth transform costing
    /// (SVT M8-M10 key frames): oblique directions
    /// (`av1.keyframe_directional`), angle deltas around the best direction
    /// (`av1.keyframe_angle_deltas`) and filter intra
    /// (`av1.keyframe_filter_intra`), ranked by prediction SATD plus
    /// √λ · mode bits (as AV2 ranks its full set), best
    /// `av1.keyframe_intra_rd_candidates` first. Candidates already in
    /// `base` are left out. Empty when every tool is off.
    fn key_intra_shortlist(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
        base: &[IntraMode],
    ) -> Vec<LumaIntra> {
        let directional = self.params.keyframe_directional >= 0.5;
        let delta_search = self.params.keyframe_angle_deltas.round() as i32;
        let filter = self.codes_filter_intra(block, IntraMode::Dc);
        if !directional && delta_search == 0 && !filter {
            return Vec::new();
        }
        let mut ranked: Vec<(f32, LumaIntra)> = Vec::new();
        let mut score = |this: &mut Self, luma: LumaIntra| {
            let cost = this.key_intra_satd_cost(block, source, recon, luma);
            ranked.push((cost, luma));
            cost
        };
        // The directions the search allows (the likeliest
        // `av1.keyframe_directions` by gradient orientation), nominal angles
        // first: the best of them anchors the delta search.
        let mut best_direction = (f32::INFINITY, IntraMode::Vertical);
        for mode in self.key_directions(block, source, directional) {
            let cost = score(self, LumaIntra::plain(mode));
            if cost < best_direction.0 {
                best_direction = (cost, mode);
            }
        }
        let (nominal_cost, direction) = best_direction;
        let mut try_delta = |this: &mut Self, angle_delta: i8| {
            score(
                this,
                LumaIntra {
                    angle_delta,
                    ..LumaIntra::plain(direction)
                },
            )
        };
        match delta_search {
            0 => {}
            1 => {
                // Delta costs are roughly convex (AV2 `DeltaSearch::
                // CoarseToFine`): refine beside the better of −2, 0, +2.
                let (lower, upper) = (try_delta(self, -2), try_delta(self, 2));
                let refine: [i8; 2] = if lower < nominal_cost && lower <= upper {
                    [-1, -3]
                } else if upper < nominal_cost {
                    [1, 3]
                } else {
                    [-1, 1]
                };
                for delta in refine {
                    try_delta(self, delta);
                }
            }
            _ => {
                for delta in (-MAX_ANGLE_DELTA..=MAX_ANGLE_DELTA).filter(|&d| d != 0) {
                    try_delta(self, delta);
                }
            }
        }
        if filter {
            for mode in FilterIntraMode::ALL {
                score(
                    self,
                    LumaIntra {
                        filter: Some(mode),
                        ..LumaIntra::DC
                    },
                );
            }
        }
        let in_base =
            |luma: &LumaIntra| *luma == LumaIntra::plain(luma.mode) && base.contains(&luma.mode);
        ranked.retain(|(_, luma)| !in_base(luma));
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        let keep = self.params.keyframe_intra_rd_candidates.round().max(1.0) as usize;
        ranked
            .into_iter()
            .take(keep)
            .map(|(_, luma)| luma)
            .collect()
    }

    /// The directional modes the key search predicts: vertical and
    /// horizontal, plus the oblique ones with `directional`; with
    /// `av1.keyframe_directions` below 8 only the likeliest that many of the
    /// eight by the source block's gradient orientation (the shared
    /// engine's [`directions_by_orientation`]).
    fn key_directions(&self, block: &Block, source: &Picture, directional: bool) -> Vec<IntraMode> {
        let allowed = |mode: &IntraMode| {
            directional || matches!(mode, IntraMode::Vertical | IntraMode::Horizontal)
        };
        let limit = self.params.keyframe_directions.round() as usize;
        let all_directions = IntraMode::OBLIQUE.len() + 2;
        if limit >= all_directions {
            return [IntraMode::Vertical, IntraMode::Horizontal]
                .into_iter()
                .chain(IntraMode::OBLIQUE)
                .filter(allowed)
                .collect();
        }
        let n = block.size();
        let src = source
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, n, n)
            .expect("block inside source");
        directions_by_orientation(&src)
            .into_iter()
            .filter(allowed)
            .take(limit)
            .collect()
    }

    /// Prediction SATD of `luma` on the block plus √λ · its mode bits (the
    /// prediction is written into `recon`, which coding overwrites).
    fn key_intra_satd_cost(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
        luma: LumaIntra,
    ) -> f32 {
        let bit_cost = (self.lambda() * self.params.intra_rate_scale).sqrt();
        self.intra_prediction_satd(block, source, recon, luma)
            + bit_cost * self.key_mode_bits(block, luma)
    }

    /// Prediction SATD of `luma` on the block (the prediction is written
    /// into `recon`, which coding overwrites).
    fn intra_prediction_satd(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
        luma: LumaIntra,
    ) -> f32 {
        self.predict_plane(block, PlaneIndex::Y, &Prediction::Intra(luma), recon);
        let n = block.size();
        let size = BlockSize::new(n, n).expect("intra blocks are kernel sizes");
        let src = source
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, n, n)
            .expect("block inside source");
        let pred = recon
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, n, n)
            .expect("block inside reconstruction");
        self.dsp.satd(size, &src, &pred) as f32
    }

    /// Key-frame luma mode bits: `y_mode`, its angle delta, `uv_mode` DC
    /// under it and the filter intra syntax.
    fn key_mode_bits(&self, block: &Block, luma: LumaIntra) -> f32 {
        let (above, left) = self.key_mode_context(block);
        self.cdfs.kf_y_mode[above][left].bits(luma.mode.symbol())
            + self.luma_mode_tail_bits(block, luma)
    }

    /// The bits that follow an intra block's `y_mode` in key and inter
    /// frames alike: `uv_mode` DC under it, the angle delta and the filter
    /// intra syntax.
    fn luma_mode_tail_bits(&self, block: &Block, luma: LumaIntra) -> f32 {
        let (cfl, _) = uv_mode_alphabet(block);
        let mut bits = self.cdfs.uv_mode[cfl][luma.mode.symbol()].bits(UV_DC_PRED);
        if let Some(ctx) = luma.mode.angle_context() {
            bits += self.cdfs.angle_delta[ctx].bits(luma.angle_delta_symbol());
        }
        if self.codes_filter_intra(block, luma.mode) {
            let ctx = filter_intra_size_context(block.log2_size());
            bits += self.cdfs.filter_intra[ctx].bits(usize::from(luma.filter.is_some()));
            if let Some(filter) = luma.filter {
                bits += self.cdfs.filter_intra_mode.bits(filter.symbol());
            }
        }
        bits
    }

    /// How the frame header signals this frame's size (§5.9.5 – §5.9.7):
    /// the stream size, or under a coded-size switch its own size, copied
    /// from the reference when that has it (`found_ref`). Superres frames
    /// keep the stream's upscaled size; spatial SVC layers signal their own
    /// display size as `RenderWidth` × `RenderHeight`.
    fn frame_size_syntax(&self) -> FrameSizeSyntax {
        let size = (self.grids.width, self.grids.height);
        if let Some(sp) = &self.spatial {
            let stream = sp.config.full_size;
            let override_size = size != stream;
            let found_ref = override_size
                && self.kind == FrameKind::Inter
                && sp.slot_size[self.references.reference.slot] == Some((size, size));
            return FrameSizeSyntax {
                stream,
                render: size,
                override_size,
                found_ref,
            };
        }
        let override_size = self.superres.is_none() && size != self.stream_size;
        let found_ref = override_size
            && self.kind == FrameKind::Inter
            && self.held_reference().is_none_or(|held| held.size() == size);
        FrameSizeSyntax {
            stream: self.stream_size,
            render: self.stream_size,
            override_size,
            found_ref,
        }
    }

    /// The normative picture of this inter frame's active reference when the
    /// backend holds it (`reference_store.rs`): blocks then predict from it
    /// scaled instead of from the engine's reconstruction.
    #[inline]
    fn held_reference(&self) -> Option<&HeldReference> {
        if self.spatial.is_none() && !self.reference_held {
            return None;
        }
        self.held_reference_for_slot_slow(self.active_slot())
    }

    #[inline]
    fn held_reference_for_slot(&self, slot: usize) -> Option<&HeldReference> {
        if self.spatial.is_none() && !self.reference_held {
            return None;
        }
        self.held_reference_for_slot_slow(slot)
    }

    #[inline(never)]
    fn held_reference_for_slot_slow(&self, slot: usize) -> Option<&HeldReference> {
        if let Some(sp) = &self.spatial {
            if sp.current_layer.spatial_id == 0
                || self.kind != FrameKind::Inter
                || sp.slot_layer[slot] == Some(sp.current_layer.spatial_id)
            {
                return None;
            }
            return self.store.held(slot);
        }
        self.store.held(slot)
    }

    /// Statistics for the exact cached prediction, if its key matches.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn cached_luma_stats(
        &self,
        block: &Block,
        prediction: &Prediction<'_>,
    ) -> Option<nc_dsp::Variance> {
        let Prediction::Inter { mv, .. } = prediction else {
            return None;
        };
        // The engine ranked candidates on its proxy of a held reference.
        if self.held_reference().is_some() {
            return None;
        }
        let secondary_ref = self.is_secondary();
        self.scratch
            .cached_luma
            .filter(|cached| {
                cached.block == *block
                    && cached.mv == *mv
                    && cached.filter == nc_dsp::Filter::Av1
                    && cached.secondary_ref == secondary_ref
            })
            .map(|cached| cached.stats)
    }

    /// Write the prediction, reusing exact luma pixels when available.
    fn predict_plane(
        &self,
        block: &Block,
        plane: PlaneIndex,
        prediction: &Prediction<'_>,
        recon: &mut Picture,
    ) {
        if let Prediction::Compound(compound) = prediction {
            let shift = usize::from(plane != PlaneIndex::Y);
            let (x, y) = (block.x >> shift, block.y >> shift);
            let n = 1usize << (block.log2_size() - shift as u8);
            self.predict_compound_plane(plane, (x, y, n, shift), compound, recon);
            return;
        }
        self.predict_single_plane(block, plane, prediction, recon);
    }

    /// [`Self::predict_plane`] for every prediction but compound.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn predict_single_plane(
        &self,
        block: &Block,
        plane: PlaneIndex,
        prediction: &Prediction<'_>,
        recon: &mut Picture,
    ) {
        let shift = usize::from(plane != PlaneIndex::Y);
        let (x, y) = (block.x >> shift, block.y >> shift);
        let n = 1usize << (block.log2_size() - shift as u8);
        if plane == PlaneIndex::Y && self.cached_luma_stats(block, prediction).is_some() {
            let cached = PlaneView::new(&self.scratch.luma_prediction[..n * n], n, n, n)
                .expect("cached prediction is tightly packed");
            recon
                .plane_mut(plane)
                .region_mut(x as isize, y as isize, n, n)
                .expect("block inside reconstruction")
                .copy_from(&cached);
            return;
        }
        match prediction {
            Prediction::Intra(luma) => {
                let luma = if plane == PlaneIndex::Y {
                    *luma
                } else {
                    LumaIntra::DC
                };
                if luma == LumaIntra::DC {
                    let value = dc_value(recon.plane(plane), x, y, n, y > 0, x > 0);
                    recon
                        .plane_mut(plane)
                        .region_mut(x as isize, y as isize, n, n)
                        .expect("blocks lie inside the picture")
                        .fill(value);
                } else {
                    let edges = luma.edges(recon.plane(plane), x, y, n);
                    let mut dst = recon
                        .plane_mut(plane)
                        .region_mut(x as isize, y as isize, n, n)
                        .expect("blocks lie inside the picture");
                    luma.predict(&self.dsp, &edges, &mut dst);
                }
            }
            Prediction::ChromaFromLuma(cfl) => {
                assert_ne!(plane, PlaneIndex::Y);
                let mut dst = recon
                    .plane_mut(plane)
                    .region_mut(x as isize, y as isize, n, n)
                    .expect("block inside reconstruction");
                cfl.predict(plane as usize - 1, &mut dst);
            }
            Prediction::Inter { reference, mv } => {
                let mut dst = recon
                    .plane_mut(plane)
                    .region_mut(x as isize, y as isize, n, n)
                    .expect("blocks lie inside the picture");
                if let Some(held) = self.held_reference() {
                    let (reference, reference_size) = held.plane(plane);
                    inter::predict_scaled(
                        &self.dsp,
                        reference,
                        reference_size,
                        held.scale_sizes((self.grids.width, self.grids.height)),
                        (x, y),
                        n,
                        *mv,
                        shift == 1,
                        &mut dst,
                    );
                    return;
                }
                inter::predict(
                    &self.dsp,
                    reference.plane(plane),
                    x,
                    y,
                    n,
                    *mv,
                    shift == 1,
                    &mut dst,
                );
            }
            Prediction::Compound(_) => unreachable!("dispatched by predict_plane"),
        }
    }

    /// Compound prediction of one plane of a block. Kept out of line: its two
    /// 8 KiB prediction buffers would otherwise be set up in every
    /// [`Self::predict_plane`] call, compound or not.
    #[inline(never)]
    fn predict_compound_plane(
        &self,
        plane: PlaneIndex,
        (x, y, n, shift): (usize, usize, usize, usize),
        compound: &CompoundPrediction<'_>,
        recon: &mut Picture,
    ) {
        let CompoundPrediction {
            primary,
            secondary,
            mvs,
        } = *compound;
        let mut comp0 = [0i16; MAX_BLOCK_AREA];
        let mut comp1 = [0i16; MAX_BLOCK_AREA];
        let area = n * n;
        let primary_slot = self.references.reference.slot;
        let secondary_slot = self
            .references
            .secondary
            .expect("compound prediction requires a secondary reference")
            .slot;
        for (ref_pic, slot, mv, buf) in [
            (primary, primary_slot, mvs[0], &mut comp0[..area]),
            (secondary, secondary_slot, mvs[1], &mut comp1[..area]),
        ] {
            if let Some(held) = self.held_reference_for_slot(slot) {
                let (plane_ref, ref_size) = held.plane(plane);
                inter::predict_scaled_compound(
                    &self.dsp,
                    plane_ref,
                    ref_size,
                    held.scale_sizes((self.grids.width, self.grids.height)),
                    (x, y),
                    n,
                    mv,
                    shift == 1,
                    buf,
                );
            } else {
                inter::predict_compound(
                    &self.dsp,
                    ref_pic.plane(plane),
                    x,
                    y,
                    n,
                    mv,
                    shift == 1,
                    buf,
                );
            }
        }
        let mut dst = recon
            .plane_mut(plane)
            .region_mut(x as isize, y as isize, n, n)
            .expect("blocks lie inside the picture");
        let size = BlockSize::new(n, n).expect("square AV1 block size");
        self.dsp
            .average_compound(size, (&comp0[..area], &comp1[..area]), n, &mut dst);
    }

    fn bypass_uv_residual(&self, luma: &PlaneResidual) -> bool {
        luma.transform_rd_rejected
            && self.params.transform_rd_skip_chroma >= 0.5
            && self.noise_sigma <= self.params.transform_rd_uv_max_noise_sigma
    }

    fn code_chroma(
        &mut self,
        block: &Block,
        prediction: &Prediction<'_>,
        source: &Picture,
        recon: &mut Picture,
        skip_residual: bool,
    ) -> [PlaneResidual; 2] {
        [PlaneIndex::U, PlaneIndex::V].map(|plane| {
            if skip_residual {
                self.predict_without_residual(block, plane, prediction, recon)
            } else {
                self.code_plane(block, plane, prediction, source, recon)
            }
        })
    }

    fn predict_without_residual(
        &mut self,
        block: &Block,
        plane: PlaneIndex,
        prediction: &Prediction<'_>,
        recon: &mut Picture,
    ) -> PlaneResidual {
        self.predict_plane(block, plane, prediction, recon);
        let shift = u8::from(plane != PlaneIndex::Y);
        let tx = TxSize::from_log2(block.log2_size() - shift).expect("supported transform size");
        PlaneResidual::uncoded(tx)
    }

    #[allow(clippy::too_many_lines)] // Keep the borrowed prediction/transform scratch pipeline together.
    fn code_plane(
        &mut self,
        block: &Block,
        plane: PlaneIndex,
        prediction: &Prediction<'_>,
        source: &Picture,
        recon: &mut Picture,
    ) -> PlaneResidual {
        let shift = usize::from(plane != PlaneIndex::Y);
        let (x, y) = (block.x >> shift, block.y >> shift);
        let log2 = block.log2_size() - shift as u8;
        let tx = TxSize::from_log2(log2).expect("block sizes map to supported transforms");
        let n = tx.size();
        let coded_area = tx.coded_size() * tx.coded_size();
        let size = BlockSize::new(n, n).expect("square AV1 block size");
        let policy = self.forward_policy(true);

        // Prediction goes straight into the reconstruction; the residual is
        // added on top of it.
        self.predict_plane(block, plane, prediction, recon);
        let mut dst = recon
            .plane_mut(plane)
            .region_mut(x as isize, y as isize, n, n)
            .expect("blocks lie inside the picture");
        let src = source
            .plane(plane)
            .region(x as isize, y as isize, n, n)
            .expect("blocks lie inside the picture");
        let max_mv = self.params.residual_rd_max_mv as i32;
        // Planned past qindex 255 (`FramePlan::lambda_scale`) every inter
        // luma residual is RD-checked at the scaled λ: the vector limit
        // keeps the check off moving blocks where it costs quality at the
        // coded λ, but past the cap the residual must pay for itself.
        let headroom = self.lambda_scale > 1.0;
        let needs_distortion = plane == PlaneIndex::Y
            && match prediction {
                Prediction::Inter { mv, .. } => {
                    headroom
                        || (i32::from(mv.row).abs() <= max_mv && i32::from(mv.col).abs() <= max_mv)
                }
                Prediction::Compound(CompoundPrediction { mvs, .. }) => {
                    headroom
                        || mvs.iter().all(|mv| {
                            i32::from(mv.row).abs() <= max_mv && i32::from(mv.col).abs() <= max_mv
                        })
                }
                _ => false,
            };
        // Full SSE preserves mean residual energy; variance is unsafe here.
        let mut pre_tx_sse = if plane == PlaneIndex::Y {
            self.cached_luma_stats(block, prediction)
                .map(|stats| stats.sse)
        } else {
            None
        };
        let pre_tx_skip = plane == PlaneIndex::Y
            && self.pre_tx_skip_enabled()
            && match prediction {
                Prediction::Inter { mv, .. } => {
                    let sse = pre_tx_sse
                        .unwrap_or_else(|| self.dsp.variance(size, &src, &dst.as_view()).sse);
                    pre_tx_sse = Some(sse);
                    self.skip_luma_before_transform(block, *mv, sse, &src, &dst.as_view())
                }
                _ => false,
            };
        let transform_rd_context =
            (needs_distortion && self.params.transform_rd_skip >= 0.5).then(|| {
                let (x4, y4, units) = Self::plane_units(block, 0);
                let (above, left) = self.context_slices(0, x4, y4, units);
                TxbContext::derive(PlaneType::Luma, true, above, left)
            });
        let transform_rd_lambda = transform_rd_context.map(|_| self.lambda());
        let quantizer = self.plane_quantizer(plane, tx);
        let planned_min_level = self.planned_min_level(plane);
        let trellis = self.trellis_block(block, plane, prediction, true);
        let scratch = &mut self.scratch;
        let levels = &mut scratch.levels[plane as usize][..coded_area];
        let inter = matches!(
            prediction,
            Prediction::Inter { .. } | Prediction::Compound(_)
        );
        let uncoded = PlaneResidual::uncoded(tx);
        if pre_tx_skip {
            return uncoded;
        }
        let cached = match (plane, prediction) {
            (PlaneIndex::Y, Prediction::Intra(mode)) => scratch
                .cached_intra
                .take()
                .filter(|c| c.block == *block && c.mode == Some(*mode)),
            // The transform-RD check needs the unquantised coefficients,
            // which the cache does not keep.
            (PlaneIndex::Y, Prediction::Inter { .. } | Prediction::Compound(_))
                if transform_rd_context.is_none() =>
            {
                scratch
                    .cached_intra
                    .take()
                    .filter(|c| c.block == *block && c.mode.is_none())
            }
            _ => None,
        };
        // The trellis needs the unquantised coefficients too.
        let cached = cached.filter(|c| c.originals || trellis.is_none());
        let nonzero = if let Some(cached) = cached {
            if let Some(t) = trellis {
                // The trial's coefficients (a fresh transform: same
                // residual and forward policy), quantised by the trellis.
                trellis_quantize(
                    &self.dsp,
                    (&mut self.cdfs.coeff, &self.grids.contexts),
                    &mut scratch.trellis,
                    t,
                    tx,
                    policy,
                    &quantizer,
                    &scratch.intra_originals[..coded_area],
                    levels,
                )
            } else {
                levels.copy_from_slice(&scratch.intra_levels[..coded_area]);
                cached.nonzero
            }
        } else {
            self.dsp
                .residual(size, &src, &dst.as_view(), &mut scratch.residual, n);
            // SVT light-PD1 shortcut for inter chroma (`mds3` module docs).
            if inter
                && plane != PlaneIndex::Y
                && mds3::skip_chroma_residual(&scratch.residual, n, self.chroma_skip_sad_num)
            {
                return uncoded;
            }
            if transform_rd_context.is_some() || trellis.is_some() {
                // The RD paths keep the unquantised coefficients: transform
                // into them and quantise out of place (no copy).
                if scratch.original_coefficients.is_empty() {
                    scratch.original_coefficients.resize(MAX_CODED_AREA, 0);
                }
                let original = &mut scratch.original_coefficients[..coded_area];
                forward_dct_with(&self.dsp, policy, tx, &scratch.residual, n, original);
                match trellis {
                    Some(t) => trellis_quantize(
                        &self.dsp,
                        (&mut self.cdfs.coeff, &self.grids.contexts),
                        &mut scratch.trellis,
                        t,
                        tx,
                        policy,
                        &quantizer,
                        original,
                        levels,
                    ),
                    None => {
                        quantize_coded_from(&self.dsp, policy, tx, original, &quantizer, levels)
                    }
                }
            } else {
                forward_dct_with(&self.dsp, policy, tx, &scratch.residual, n, levels);
                // Quantization already counts every nonzero level. Avoid
                // walking the diagonal scan again for all-zero and DC-only
                // blocks.
                quantize_coded(&self.dsp, policy, tx, levels, &quantizer)
            }
        };
        let nonzero = if headroom {
            apply_planned_dead_zone(levels, planned_min_level)
        } else {
            nonzero
        };
        if nonzero == 0 {
            return uncoded;
        }

        let mut eob = quantized_eob(tx, levels, nonzero, policy);
        if inter && eob > 0 {
            eob = mds3::shave_trailing_ones(
                coefficient_scan(tx),
                levels,
                eob,
                self.shave_min_zero_run,
            );
        }
        if eob == 0 {
            return uncoded;
        }
        // Isolated legal-rate probes: AV1 P −1.2%, key −2.4% instructions;
        // bitstream parity is unchanged (zero BD-rate effect).
        // Quantisation does not change the prediction. Delay its SSE until
        // we know this is a coded inter-luma residual whose RD decision uses
        // it; chroma, key frames and larger vectors never consume these stats.
        // The pre-transform gate already measured this exact prediction.
        // Forward transform and quantisation only modify coefficient scratch.
        let sse_predicted = needs_distortion.then(|| {
            pre_tx_sse.unwrap_or_else(|| self.dsp.variance(size, &src, &dst.as_view()).sse)
        });
        let dequant = &mut scratch.dequant[..coded_area];
        let dequant_area = if eob == 1 {
            1
        } else {
            dequantized_extent(tx, eob)
        };
        self.dsp.av1_dequantize(
            &levels[..dequant_area],
            &quantizer,
            &mut dequant[..dequant_area],
        );
        let mut cached_coefficient_bits = None;
        // Approximate spatial gain avoids inverse/reconstruction work for weak
        // residuals. Mode2, gain1.25, UV sigma1 and maxMV8 measured native CPU
        // -12.87%/-9.76% (bodycam/crowd), with full21 TRAIN PSNR/VMAF-v1 BD
        // +2.16%/-0.52%. Useful residuals can be lost, especially on noisy input;
        // docs/handoff/overnight.md records the tradeoffs. Disabled by default.
        if let Some(ctx) = transform_rd_context {
            let lambda = transform_rd_lambda.expect("enabled luma RD caches lambda");
            let predicted = f64::from(sse_predicted.expect("enabled luma RD needs prediction SSE"));
            let gain = crate::transform::coefficient_gain(
                &self.dsp,
                tx,
                &scratch.original_coefficients[..dequant_area],
                &dequant[..dequant_area],
            ) * f64::from(self.params.transform_rd_gain_scale);
            let estimated_coded = (predicted - gain).max(0.0);
            let bits = if self.params.transform_rd_skip >= 1.5 {
                // Pinned SVT product_coding_loop.c:5530, coeff_rate_est_lvl0.
                self.cdfs.coeff.eob_linear_bits(tx, ctx, eob)
            } else {
                self.cdfs
                    .coeff
                    .cost(tx, PlaneType::Luma, true, DC_PRED, ctx, levels, eob)
                    - self
                        .cdfs
                        .coeff
                        .cost(tx, PlaneType::Luma, true, DC_PRED, ctx, levels, 0)
            };
            if self.params.transform_rd_skip < 1.5 {
                cached_coefficient_bits = Some(bits);
            }
            let pct = if (predicted as f32) < self.params.skip_bias_energy * lambda * (n * n) as f32
            {
                if self.is_base_frame {
                    self.params.skip_bias_pct_base
                } else {
                    self.params.skip_bias_pct
                }
            } else {
                100.0
            };
            let coded_cost =
                estimated_coded + f64::from(lambda * bits * self.params.transform_rd_rate_scale);
            if f64::from(pct) * predicted < 100.0 * coded_cost {
                // Prediction remains untouched: no inverse or re-predict.
                return PlaneResidual {
                    transform_rd_rejected: true,
                    ..uncoded
                };
            }
        }
        if eob == 1 {
            inverse_dct_dc_add(&self.dsp, tx, dequant[0], &mut dst, &mut scratch.residual);
        } else {
            inverse_dct_add_eob(&self.dsp, tx, dequant, eob, &mut dst);
        }
        let distortion = sse_predicted.map(|sse_predicted| ResidualDistortion {
            sse_predicted,
            sse_coded: self.dsp.variance(size, &src, &dst.as_view()).sse,
            coefficient_bits: cached_coefficient_bits,
        });
        PlaneResidual {
            eob,
            distortion,
            ..uncoded
        }
    }

    /// Pinned SVT `lpd1_should_perform_tx` score with configurable controls.
    /// Rate units here are bits, not SVT's 1/512 bits. The current encoder's
    /// lambda and syntax CDFs supply the rate-distortion units.
    /// The light-PD1 pre-transform luma skip runs this frame.
    fn pre_tx_skip_enabled(&self) -> bool {
        self.params.pre_tx_skip > 0.0 && self.pre_tx_score_this_frame > 0.0
    }

    fn skip_luma_before_transform(
        &self,
        block: &Block,
        mv: Av1Mv,
        sse: u32,
        src: &PlaneView<'_>,
        pred: &PlaneView<'_>,
    ) -> bool {
        let area = (block.size() * block.size()) as f32;
        let (above, left) = self.neighbours(block);
        let energy =
            100.0 * (sse as f32) < self.params.pre_tx_skip_energy * area * self.block_q_idx as f32;
        let neighbours = above.zip(left).map(|(above, left)| {
            (
                above.skip && left.skip,
                above.mode == BlockMode::Nearest && left.mode == BlockMode::Nearest,
            )
        });
        let rd = || {
            if self.params.pre_tx_skip_model >= 0.5 {
                residual_quantises_to_zero(
                    &self.dsp,
                    block.size(),
                    src,
                    pred,
                    self.zero_limits(PlaneIndex::Y),
                    self.params.pre_tx_texture_scale,
                )
            } else if self.params.pre_tx_skip_rd > 0.0 {
                let mode_bits = self.mode_bits(block, &PredictionMode::Inter(to_engine_mv(mv)));
                let skip_cdf = &self.cdfs.skip[skip_context(above, left)];
                let skip_cost = sse as f32 + self.lambda() * (mode_bits + skip_cdf.bits(1));
                let reference_cost =
                    area + self.lambda() * (mode_bits + skip_cdf.bits(0) + 6000.0 / 512.0);
                100.0 * skip_cost < self.params.pre_tx_skip_rd * reference_cost
            } else {
                false
            }
        };
        nc_encoder::pre_tx_skips(
            energy,
            neighbours,
            rd,
            nc_encoder::QuantizerBand::from_av1_qindex(self.block_q_idx),
            self.params.pre_tx_luma_dominant > 0.0,
            self.pre_tx_score_this_frame,
        )
    }

    /// The trellis inputs of the transform block covering `block` in
    /// `plane`, or `None` while `quant.trellis` is off. `fills_block` is
    /// false for the leaves of a luma transform tree: their `all_zero`
    /// context then reads the neighbour contexts as they stand, before the
    /// earlier leaves of the same tree have written theirs (an
    /// approximation of the coded-block flag's price only).
    fn trellis_block(
        &self,
        block: &Block,
        plane: PlaneIndex,
        prediction: &Prediction<'_>,
        fills_block: bool,
    ) -> Option<TrellisBlock> {
        let settings = self
            .trellis
            .filter(|t| t.key_frames || self.kind != FrameKind::Key)?;
        let p = plane as usize;
        let plane_type = if plane == PlaneIndex::Y {
            PlaneType::Luma
        } else {
            PlaneType::Chroma
        };
        let (x4, y4, units) = Self::plane_units(block, p);
        // λ is per squared luma step; chroma is quantised at its own
        // (delta-q) step, and keeps the quality that step buys only if its
        // λ scales with it (as x264 and HM scale chroma λ by the chroma QP
        // offset). At the luma λ the trellis crushed chroma: crowd 30
        // frames lost 2.9 dB U PSNR for 0.45 dB Y.
        let plane_scale = match plane_type {
            PlaneType::Luma => 1.0,
            PlaneType::Chroma => {
                let step_ratio = self.chroma_steps.ac as f32 / self.steps.ac as f32;
                settings.chroma_scale? * step_ratio * step_ratio
            }
        };
        Some(TrellisBlock {
            lambda: self.lambda() * settings.lambda_scale * plane_scale,
            max_states: settings.max_states,
            plane: plane_type,
            is_inter: matches!(
                prediction,
                Prediction::Inter { .. } | Prediction::Compound(_)
            ),
            intra_mode: match prediction {
                Prediction::Intra(luma) => luma.intra_dir(),
                _ => DC_PRED,
            },
            context: (p, x4, y4, units, fills_block),
        })
    }

    /// The frame kind's registered luma and chroma rounding offsets.
    /// Whether this frame's symbols adapt their CDFs (`av1.cdf_update`).
    fn adapts_cdfs(&self) -> bool {
        self.params.cdf_update >= 0.5
    }

    fn set_rounding(&mut self) {
        let (luma, chroma) = match self.kind {
            FrameKind::Key => (
                self.params.keyframe_rounding_q7,
                self.params.keyframe_rounding_q7,
            ),
            FrameKind::Inter => (
                self.params.inter_rounding_q7,
                self.params.inter_chroma_rounding_q7,
            ),
        };
        self.rounding_q7 = luma.round() as i32;
        self.chroma_rounding_q7 = chroma.round() as i32;
    }

    /// The quantiser of `plane` for transform size `tx` in this frame.
    fn plane_quantizer(&self, plane: PlaneIndex, tx: TxSize) -> Av1Quantizer {
        if plane == PlaneIndex::Y {
            self.steps.quantizer(tx, self.rounding_q7)
        } else {
            self.chroma_steps.quantizer(tx, self.chroma_rounding_q7)
        }
    }

    /// The smallest level that survives quantisation at the planned step
    /// past qindex 255 (√ of [`FramePlan::lambda_scale`] times the coded
    /// one): a coefficient needs `(1 − rounding) · planned step`, i.e.
    /// `(1 − rounding) · √scale` coded steps.
    fn planned_min_level(&self, plane: PlaneIndex) -> i32 {
        let rounding = if plane == PlaneIndex::Y {
            self.rounding_q7
        } else {
            self.chroma_rounding_q7
        };
        let keep = 1.0 - rounding as f32 / ROUNDING_Q7_ONE;
        (keep * self.lambda_scale.sqrt()).ceil() as i32
    }

    /// The largest orthonormal coefficients of `plane` that quantise to
    /// zero: AV1 transforms scale orthonormal coefficients by 8 (with
    /// `dqDenom` folded into the step), and a level is zero below
    /// `(1 − rounding) · step`. Past qindex 255 the limits follow the
    /// planned step (√ of [`FramePlan::lambda_scale`] times the coded one).
    fn zero_limits(&self, plane: PlaneIndex) -> ZeroLimits {
        let (steps, rounding) = if plane == PlaneIndex::Y {
            (&self.steps, self.rounding_q7)
        } else {
            (&self.chroma_steps, self.chroma_rounding_q7)
        };
        let keep = (1.0 - rounding as f32 / ROUNDING_Q7_ONE) * self.lambda_scale.sqrt();
        let limit = |step: i32| keep * step as f32 / AC_Q_PER_SAMPLE_STEP;
        ZeroLimits {
            dc: limit(steps.dc),
            ac: limit(steps.ac),
        }
    }

    /// Rate–distortion check of a coded inter luma residual: if its
    /// coefficients cost more (λ · bits, priced with the current CDFs) than
    /// the squared error they remove, they are not coded and the luma
    /// reconstruction goes back to the prediction. On blocks whose
    /// prediction error is already tiny (below `skip_bias_energy` · λ per
    /// sample) the test is biased towards skipping as SVT's
    /// `lpd1_blk_skip_luma_rd` is: drop if `pct · D_skip < 100 · (D_coded +
    /// λ·R)`. Coding such residuals often makes the picture *worse*: their
    /// sub-unit mean corrections round to scattered ±1 samples.
    ///
    /// Restricted to near-static blocks (`residual_rd_max_mv`) and to luma.
    /// Quick clips, 10 s IP, PSNR-YUV BD vs svtav1-m12 (fast check):
    /// - the plain check: static −36 points, bodycam −120, low-light 0,
    ///   dashcam +4, crowd +6 (without the vector limit dashcam lost 10
    ///   points; with chroma included every clip got worse);
    /// - the bias at 60% below λ/16: static −11, bodycam −15, crowd +3.7,
    ///   dashcam +1.2, low-light −0.5 (mean −4.3). Unconditional 75% gave
    ///   static −36 but low-light +29 and crowd +12.
    fn drop_unprofitable_luma_residual(
        &mut self,
        block: &Block,
        prediction: &Prediction<'_>,
        luma: &mut PlaneResidual,
        recon: &mut Picture,
    ) {
        let Some(distortion) = luma.distortion.as_ref() else {
            return;
        };
        let tx = luma.tx;
        let (x4, y4, units) = Self::plane_units(block, 0);
        let (above_ctx, left_ctx) = self.context_slices(0, x4, y4, units);
        let ctx = TxbContext::derive(PlaneType::Luma, true, above_ctx, left_ctx);
        let levels = &self.scratch.levels[0][..tx.coded_size() * tx.coded_size()];
        let coefficient_bits = distortion.coefficient_bits.unwrap_or_else(|| {
            if self.params.coeff_rate_linear >= 0.5 {
                self.cdfs.coeff.eob_linear_bits(tx, ctx, luma.eob)
            } else {
                self.cdfs
                    .coeff
                    .cost(tx, PlaneType::Luma, true, DC_PRED, ctx, levels, luma.eob)
                    - self
                        .cdfs
                        .coeff
                        .cost(tx, PlaneType::Luma, true, DC_PRED, ctx, levels, 0)
            }
        });
        let lambda = self.lambda();
        let area = (block.size() * block.size()) as f32;
        let skip_distortion = distortion.sse_predicted as f32;
        let pct = if skip_distortion < self.params.skip_bias_energy * lambda * area {
            // SVT's pseudo-layer cadence: every 4th frame ("base") runs a
            // gentler light-PD1 level, whose skip margin is wider.
            if self.is_base_frame {
                self.params.skip_bias_pct_base
            } else {
                self.params.skip_bias_pct
            }
        } else {
            100.0
        };
        let coded_cost = distortion.sse_coded as f32 + lambda * coefficient_bits;
        if pct * skip_distortion < 100.0 * coded_cost {
            // The reconstruction holds prediction + residual: predict again.
            self.predict_plane(block, PlaneIndex::Y, prediction, recon);
            luma.eob = 0;
            luma.distortion = None;
        }
    }

    /// Key-frame luma mode, directional angle delta and chroma mode, followed
    /// by `CfL` slopes when selected (AV1 §5.11.7 `intra_frame_mode_info`).
    fn write_key_frame_intra_modes(
        &mut self,
        block: &Block,
        mode: LumaIntra,
        cfl: Option<&ChromaFromLuma>,
    ) {
        let (above, left) = self.key_mode_context(block);
        self.writer.write_symbol(
            mode.mode.symbol(),
            &mut self.cdfs.kf_y_mode[above][left],
            INTRA_MODES,
        );
        self.write_angle_and_uv_modes(block, mode, cfl);
    }

    /// `y_mode` and the chroma mode of an intra block in an inter frame
    /// (AV1 §5.11.22 `intra_block_mode_info`). The encoder's own choice is DC
    /// for both.
    fn write_intra_modes(&mut self, block: &Block, mode: LumaIntra, cfl: Option<&ChromaFromLuma>) {
        self.writer.write_symbol(
            mode.mode.symbol(),
            &mut self.cdfs.y_mode[size_group(block)],
            INTRA_MODES,
        );
        self.write_angle_and_uv_modes(block, mode, cfl);
    }

    /// `angle_delta_y` for directional luma modes, `uv_mode` (DC, or `CfL`
    /// with its slopes; chroma angles are never coded for these), then
    /// filter intra where the sequence enables it (AV1 §5.11.7: palette is
    /// off, so `filter_intra_mode_info` follows the chroma syntax).
    fn write_angle_and_uv_modes(
        &mut self,
        block: &Block,
        luma: LumaIntra,
        cfl: Option<&ChromaFromLuma>,
    ) {
        if let Some(ctx) = luma.mode.angle_context() {
            self.writer.write_symbol(
                luma.angle_delta_symbol(),
                &mut self.cdfs.angle_delta[ctx],
                ANGLE_DELTA_SYMBOLS,
            );
        }
        let (allowed, nsyms) = uv_mode_alphabet(block);
        self.writer.write_symbol(
            if cfl.is_some() {
                UV_CFL_PRED
            } else {
                UV_DC_PRED
            },
            &mut self.cdfs.uv_mode[allowed][luma.mode.symbol()],
            nsyms,
        );
        if let Some(cfl) = cfl {
            self.writer
                .write_symbol(cfl.joint_sign(), &mut self.cdfs.cfl_sign, 8);
            for (p, &alpha) in cfl.alpha.iter().enumerate() {
                if alpha != 0 {
                    self.writer.write_symbol(
                        usize::from(alpha.unsigned_abs() - 1),
                        &mut self.cdfs.cfl_alpha[cfl.alpha_context(p)],
                        16,
                    );
                }
            }
        }
        if self.codes_filter_intra(block, luma.mode) {
            let ctx = filter_intra_size_context(block.log2_size());
            self.writer
                .write_bool(luma.filter.is_some(), &mut self.cdfs.filter_intra[ctx]);
            if let Some(filter) = luma.filter {
                self.writer.write_symbol(
                    filter.symbol(),
                    &mut self.cdfs.filter_intra_mode,
                    FILTER_INTRA_MODES,
                );
            }
        }
    }

    /// The sequence enables filter intra (`av1.keyframe_filter_intra`).
    fn filter_intra_enabled(&self) -> bool {
        self.params.keyframe_filter_intra >= 0.5
    }

    /// Whether an intra block with luma `mode` codes `use_filter_intra`.
    fn codes_filter_intra(&self, block: &Block, mode: IntraMode) -> bool {
        self.filter_intra_enabled()
            && mode == IntraMode::Dc
            && filter_intra_allowed(block.log2_size())
    }

    /// `LAST_FRAME` (`secondary_ref = false`) or `LAST2_FRAME`
    /// (`secondary_ref = true`) as a single reference (AV1 §5.11.25
    /// `read_ref_frames`: `single_ref_p1 = 0`, `single_ref_p3 = 0`,
    /// `single_ref_p4 = secondary_ref`).
    fn write_ref_frame(
        &mut self,
        above: Option<ModeInfo>,
        left: Option<ModeInfo>,
        secondary_ref: bool,
    ) {
        if self.frame_reference_select() {
            self.writer.write_bool(
                false,
                &mut self.cdfs.comp_mode[comp_mode_context(above, left)],
            );
        }
        let (near_ctx, p4_ctx) = self.single_ref_contexts(above, left);
        self.writer.write_bool(
            false,
            &mut self.cdfs.single_ref[near_ctx][SINGLE_REF_P1_NODE],
        );
        self.writer.write_bool(
            false,
            &mut self.cdfs.single_ref[near_ctx][SINGLE_REF_P3_NODE],
        );
        self.writer.write_bool(
            secondary_ref,
            &mut self.cdfs.single_ref[p4_ctx][SINGLE_REF_P4_NODE],
        );
    }

    /// `(LAST_FRAME, LAST2_FRAME)` as a unidirectional compound reference pair
    /// (AV1 §5.11.25 `read_ref_frames`: `comp_mode = 1`, `comp_ref_type = 0`,
    /// `uni_comp_ref = 0`, `uni_comp_ref_p1 = 0`).
    fn write_comp_ref_frame(&mut self, above: Option<ModeInfo>, left: Option<ModeInfo>) {
        let (near_ctx, _, uni_p1_ctx) = ref_count_contexts(above, left);
        self.writer.write_bool(
            true,
            &mut self.cdfs.comp_mode[comp_mode_context(above, left)],
        );
        self.writer.write_bool(
            false,
            &mut self.cdfs.comp_ref_type[comp_ref_type_context(above, left)],
        );
        self.writer
            .write_bool(false, &mut self.cdfs.uni_comp_ref[near_ctx][0]);
        self.writer
            .write_bool(false, &mut self.cdfs.uni_comp_ref[uni_p1_ctx][1]);
    }

    /// The inter mode `mode`, `drl_mode` and vector difference for vector
    /// `mv` (AV1 §5.11.23 `inter_block_mode_info`).
    fn write_inter_mode(&mut self, stack: &MvStack, mv: Av1Mv, mode: InterMode) {
        let (w, cdfs) = (&mut self.writer, &mut self.cdfs);
        w.write_bool(!mode.is_new(), &mut cdfs.new_mv[stack.new_mv_context()]);
        if !mode.is_new() {
            w.write_bool(
                mode != InterMode::Global,
                &mut cdfs.zero_mv[stack.zero_mv_context()],
            );
            if mode != InterMode::Global {
                w.write_bool(
                    mode != InterMode::Nearest,
                    &mut cdfs.ref_mv[stack.ref_mv_context()],
                );
            }
        }
        for (position, symbol) in mode
            .drl()
            .into_iter()
            .flat_map(|drl| drl_symbols(stack, drl))
        {
            w.write_bool(symbol, &mut cdfs.drl[stack.drl_context(position)]);
        }
        if let InterMode::New(index) = mode {
            cdfs.mv.write(w, mv, stack.vector(index));
        }
    }

    /// The compound inter mode `mode`, `drl_mode` and vector differences for
    /// `mvs` (AV1 §5.11.23, §5.11.26 `read_compound_mode`, `assign_mv`).
    fn write_compound_inter_mode(
        &mut self,
        stack: &CompoundMvStack,
        mvs: [Av1Mv; 2],
        mode: CompoundInterMode,
    ) {
        let (w, cdfs) = (&mut self.writer, &mut self.cdfs);
        w.write_symbol(
            mode.symbol(),
            &mut cdfs.compound_mode[stack.compound_mode_context()],
            8,
        );
        for (position, symbol) in mode
            .drl()
            .into_iter()
            .flat_map(|drl| drl_symbols_compound(stack, drl))
        {
            w.write_bool(symbol, &mut cdfs.drl[stack.drl_context(position)]);
        }
        let bases = mode.new_mv_base();
        for list in 0..2 {
            if let Some(index) = bases[list] {
                let ref_mv = stack.new_base_vector(index)[list];
                cdfs.mv.write(w, mvs[list], ref_mv);
            }
        }
    }

    /// Position (in 4x4 units of the plane) and side length of a block's
    /// transform: column within the frame, row within the superblock.
    fn plane_units(block: &Block, plane: usize) -> (usize, usize, usize) {
        let shift = MI_LOG2 + usize::from(plane > 0);
        let sb_rows_mask = (1usize << (SUPERBLOCK_LOG2 as usize - shift)) - 1;
        let units = 1usize << (usize::from(block.log2_size()) - shift);
        (block.x >> shift, (block.y >> shift) & sb_rows_mask, units)
    }

    fn context_slices(&self, plane: usize, x4: usize, y4: usize, units: usize) -> (&[u8], &[u8]) {
        (
            &self.grids.contexts.above[plane][x4..x4 + units],
            &self.grids.contexts.left[plane][y4..y4 + units],
        )
    }

    fn set_contexts(&mut self, plane: usize, x4: usize, y4: usize, units: usize, value: u8) {
        self.grids.contexts.above[plane][x4..x4 + units].fill(value);
        self.grids.contexts.left[plane][y4..y4 + units].fill(value);
    }

    /// How the `partition` symbol (AV1 §5.11.4) of a square node is coded,
    /// including the reduced forms used where the frame edge cuts the node
    /// (libaom `write_partition`). `partition` must be valid for the node.
    fn partition_symbol(&self, node: &Block, partition: Partition) -> PartitionSymbol {
        let size_log2_mi = node.log2_size() - MI_LOG2 as u8;
        let (mi_row, mi_col) = (node.y >> MI_LOG2, node.x >> MI_LOG2);
        let half = 1usize << (size_log2_mi - 1);
        let has_rows = mi_row + half < self.grids.mode_info.mi_rows;
        let has_cols = mi_col + half < self.grids.mode_info.mi_cols;
        // Context: whether the above / left neighbours are smaller than this node.
        let above =
            mi_row > 0 && self.grids.mode_info.at(mi_row - 1, mi_col).size_log2_mi < size_log2_mi;
        let left =
            mi_col > 0 && self.grids.mode_info.at(mi_row, mi_col - 1).size_log2_mi < size_log2_mi;
        let bsl = usize::from(node.log2_size() - MIN_BLOCK_LOG2);
        let ctx = bsl * 4 + usize::from(left) * 2 + usize::from(above);
        let cdf = &self.cdfs.partition[ctx].icdf;
        match (has_rows, has_cols) {
            (true, true) => PartitionSymbol::Full {
                ctx,
                nsyms: if node.log2_size() == MIN_BLOCK_LOG2 {
                    4
                } else {
                    10
                },
            },
            (false, true) => {
                // Bottom half outside: only HORZ or SPLIT are possible; libaom
                // gathers the "vertical-alike" probabilities for this case.
                debug_assert!(matches!(partition, Partition::Split | Partition::Horz));
                PartitionSymbol::Binary(gather(cdf, &VERT_ALIKE))
            }
            (true, false) => {
                // Right half outside: only VERT or SPLIT are possible.
                debug_assert!(matches!(partition, Partition::Split | Partition::Vert));
                PartitionSymbol::Binary(gather(cdf, &HORZ_ALIKE))
            }
            (false, false) => {
                debug_assert_eq!(partition, Partition::Split, "split is implied");
                PartitionSymbol::Implied
            }
        }
    }

    fn write_partition_symbol(&mut self, node: &Block, partition: Partition) {
        match self.partition_symbol(node, partition) {
            PartitionSymbol::Full { ctx, nsyms } => {
                self.writer
                    .write_symbol(partition as usize, &mut self.cdfs.partition[ctx], nsyms);
            }
            PartitionSymbol::Binary(icdf) => {
                self.writer
                    .write_fixed(usize::from(partition == Partition::Split), &icdf, 2);
            }
            PartitionSymbol::Implied => {}
        }
    }
}

/// How one `partition` symbol is coded.
enum PartitionSymbol {
    /// The full partition alphabet, with partition CDF `ctx`.
    Full { ctx: usize, nsyms: usize },
    /// `split_or_horz` / `split_or_vert` with a gathered binary CDF, which
    /// is not adapted (AV1 §8.3.2).
    Binary([u16; 2]),
    /// Nothing coded: the split is implied by the frame edge.
    Implied,
}

/// End of block of freshly quantised `levels` holding `nonzero` nonzero
/// values. Sparse AC blocks usually end near the start of the scan: the
/// quantizer's exact count tells us when every nonzero was found; dense
/// blocks keep the reverse scan's cheap high-EOB exit.
/// Zero the levels below `min_level` in magnitude (the planned step's dead
/// zone, see `planned_min_level`); returns how many remain non-zero.
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

/// Trellis quantisation (`quant.trellis`) of one transform block's forward
/// coefficients `coeffs` into `levels`, priced with `cdfs` as they stand.
/// Returns the number of nonzero levels.
#[allow(clippy::too_many_arguments)]
fn trellis_quantize(
    dsp: &Dsp,
    (cdfs, contexts): (&mut CoeffCdfs, &CoeffContexts),
    trellis: &mut TrellisQuantizer,
    block: TrellisBlock,
    tx: TxSize,
    policy: ForwardPolicy,
    quantizer: &Av1Quantizer,
    coeffs: &[i32],
    levels: &mut [i32],
) -> u32 {
    let nonzero = dsp.av1_quantize_from(coeffs, quantizer, levels);
    let eob = quantized_eob(tx, levels, nonzero, policy);
    if eob == 0 {
        return 0;
    }
    let (p, x4, y4, units, fills_block) = block.context;
    let ctx = TxbContext::derive(
        block.plane,
        fills_block,
        &contexts.above[p][x4..x4 + units],
        &contexts.left[p][y4..y4 + units],
    );
    let mut model = RateModel::new(
        cdfs,
        tx,
        block.plane,
        block.is_inter,
        block.intra_mode,
        ctx,
        coeffs[0] < 0,
    );
    let (lambda, states) = (block.lambda, block.max_states);
    trellis.quantize(&mut model, coeffs, quantizer, lambda, states, levels, eob)
}

/// End of block of `levels` with `nonzero` nonzero levels, quantised from
/// a forward transform under `policy` (so zero past its
/// [`ForwardPolicy::scan_len`]).
fn quantized_eob(tx: TxSize, levels: &[i32], nonzero: u32, policy: ForwardPolicy) -> usize {
    if nonzero == 0 {
        0
    } else if nonzero == 1 && levels[0] != 0 {
        1
    } else if nonzero <= SPARSE_EOB_MAX_NONZERO {
        let mut remaining = nonzero;
        coefficient_scan(tx)
            .iter()
            .position(|&pos| {
                remaining -= u32::from(levels[usize::from(pos)] != 0);
                remaining == 0
            })
            .expect("quantizer count matches the coefficient levels")
            + 1
    } else {
        end_of_block_within(tx, levels, policy.scan_len(tx))
    }
}

/// The `base_q_idx` whose AC quantiser step is closest to, without going
/// below, `scale` times that of `base_q_idx` (SVT-AV1 / libaom
/// `compute_qdelta`), never below the encoder's minimum.
fn scaled_qindex(base_q_idx: u32, scale: f64) -> u32 {
    let target = f64::from(AC_QLOOKUP[base_q_idx as usize]) * scale;
    let found = AC_QLOOKUP
        .iter()
        .position(|&q| f64::from(q) >= target)
        .unwrap_or(AC_QLOOKUP.len() - 1);
    (found as u32).max(QINDEX_MIN)
}

/// Block size (log2) whose transform splits past the largest quantizer
/// (`av1.ceiling_tx_split`): the 64x64 transform's dequantiser is the finest.
const CEILING_SPLIT_LOG2: u8 = 6;

/// `single_ref_p1`, `single_ref_p3`, `single_ref_p4`: the three binary
/// choices that select `LAST_FRAME`, as indices into `single_ref[ctx]`.
const SINGLE_REF_SYMBOLS_FOR_LAST: [usize; 3] = [0, 2, 3];

/// Which signalling denotes `mv`, cheapest first.
fn inter_mode_for(stack: &MvStack, mv: Av1Mv) -> InterMode {
    if mv == stack.vector(0) {
        InterMode::Nearest
    } else if let Some(index) = (1..=MAX_NEAR_INDEX)
        .filter(|&i| i == 1 || i < stack.count)
        .find(|&i| mv == stack.vector(i))
    {
        InterMode::Near(index)
    } else if mv == Av1Mv::ZERO {
        InterMode::Global
    } else {
        InterMode::New(0)
    }
}

/// Bits of the inter mode, `drl_mode` and vector difference for `mv`.
fn inter_mode_bits(cdfs: &CdfContext, stack: &MvStack, mv: Av1Mv) -> f32 {
    let mode = inter_mode_for(stack, mv);
    let mut bits = cdfs.new_mv[stack.new_mv_context()].bits(usize::from(!mode.is_new()));
    if !mode.is_new() {
        bits += cdfs.zero_mv[stack.zero_mv_context()].bits(usize::from(mode != InterMode::Global));
        if mode != InterMode::Global {
            bits +=
                cdfs.ref_mv[stack.ref_mv_context()].bits(usize::from(mode != InterMode::Nearest));
        }
    }
    for (position, symbol) in mode
        .drl()
        .into_iter()
        .flat_map(|drl| drl_symbols(stack, drl))
    {
        bits += cdfs.drl[stack.drl_context(position)].bits(usize::from(symbol));
    }
    if let InterMode::New(index) = mode {
        bits += cdfs.mv.bits(mv, stack.vector(index));
    }
    bits
}

/// Which compound signalling denotes `mvs`, cheapest first.
fn compound_inter_mode_for(stack: &CompoundMvStack, mvs: [Av1Mv; 2]) -> CompoundInterMode {
    let v0 = stack.vector(0);
    if mvs == v0 {
        return CompoundInterMode::NearestNearest;
    }
    for index in (1..=MAX_NEAR_INDEX).filter(|&i| i == 1 || i < stack.count) {
        if mvs == stack.vector(index) {
            return CompoundInterMode::NearNear(index);
        }
    }
    if mvs == [Av1Mv::ZERO, Av1Mv::ZERO] {
        return CompoundInterMode::GlobalGlobal;
    }
    if mvs[0] == v0[0] {
        return CompoundInterMode::NearestNew;
    }
    if mvs[1] == v0[1] {
        return CompoundInterMode::NewNearest;
    }
    for index in (1..=MAX_NEAR_INDEX).filter(|&i| i == 1 || i < stack.count) {
        let vk = stack.vector(index);
        if mvs[0] == vk[0] {
            return CompoundInterMode::NearNew(index);
        }
        if mvs[1] == vk[1] {
            return CompoundInterMode::NewNear(index);
        }
    }
    CompoundInterMode::NewNew(0)
}

/// Bits of the compound inter mode, `drl_mode` and vector differences for `mvs`.
fn compound_inter_mode_bits(cdfs: &CdfContext, stack: &CompoundMvStack, mvs: [Av1Mv; 2]) -> f32 {
    let mode = compound_inter_mode_for(stack, mvs);
    let mut bits = cdfs.compound_mode[stack.compound_mode_context()].bits(mode.symbol());
    for (position, symbol) in mode
        .drl()
        .into_iter()
        .flat_map(|drl| drl_symbols_compound(stack, drl))
    {
        bits += cdfs.drl[stack.drl_context(position)].bits(usize::from(symbol));
    }
    let bases = mode.new_mv_base();
    for list in 0..2 {
        if let Some(index) = bases[list] {
            bits += cdfs.mv.bits(mvs[list], stack.new_base_vector(index)[list]);
        }
    }
    bits
}

/// `single_ref` binary tree node indices (AV1 §5.11.25, §8.3.2):
/// `p1` (`0`) splits `{LAST, LAST2, LAST3, GOLDEN}` vs `{BWDREF, ALTREF2, ALTREF}`;
/// `p3` (`2`) splits `{LAST, LAST2}` vs `{LAST3, GOLDEN}`;
/// `p4` (`3`) splits `LAST_FRAME` (`0`) vs `LAST2_FRAME` (`1`).
const SINGLE_REF_P1_NODE: usize = 0;
const SINGLE_REF_P3_NODE: usize = 2;
const SINGLE_REF_P4_NODE: usize = 3;

/// `ref_count_ctx(counts[0], counts[1])` values (AV1 §8.3.2).
const REF_COUNT_CTX_LESS: usize = 0;
const REF_COUNT_CTX_EQUAL: usize = 1;
const REF_COUNT_CTX_GREATER: usize = 2;

/// Bits of the `LAST_FRAME` (`secondary_ref = false`) or `LAST2_FRAME`
/// (`secondary_ref = true`) single-reference choice (AV1 §5.11.25, §8.3.2).
fn ref_frame_bits(
    cdfs: &CdfContext,
    (near_ctx, p4_ctx): (usize, usize),
    secondary_ref: bool,
) -> f32 {
    cdfs.single_ref[near_ctx][SINGLE_REF_P1_NODE].bits(0)
        + cdfs.single_ref[near_ctx][SINGLE_REF_P3_NODE].bits(0)
        + cdfs.single_ref[p4_ctx][SINGLE_REF_P4_NODE].bits(usize::from(secondary_ref))
}

/// Bits of the `(LAST_FRAME, LAST2_FRAME)` unidirectional compound reference
/// choice (AV1 §5.11.25, §8.3.2).
fn comp_ref_bits(cdfs: &CdfContext, above: Option<ModeInfo>, left: Option<ModeInfo>) -> f32 {
    let (near_ctx, _, uni_p1_ctx) = ref_count_contexts(above, left);
    cdfs.comp_mode[comp_mode_context(above, left)].bits(1)
        + cdfs.comp_ref_type[comp_ref_type_context(above, left)].bits(0)
        + cdfs.uni_comp_ref[near_ctx][0].bits(0)
        + cdfs.uni_comp_ref[uni_p1_ctx][1].bits(0)
}

/// Context of `comp_mode` (AV1 §8.3.2 `comp_mode` ctx).
/// Our encoder never uses backward references (`BWDREF`..=`ALTREF`), so
/// `check_backward` is always false and `single(n)` is simply `!n.is_inter()`.
fn comp_mode_context(above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
    match (above, left) {
        (Some(a), Some(l)) => match (a.is_compound(), l.is_compound()) {
            (false, false) => 0,
            (false, true) => 2 + usize::from(!a.is_inter()),
            (true, false) => 2 + usize::from(!l.is_inter()),
            (true, true) => 4,
        },
        (Some(n), None) | (None, Some(n)) => {
            if n.is_compound() {
                3
            } else {
                0
            }
        }
        (None, None) => 1,
    }
}

/// Context of `comp_ref_type` (AV1 §8.3.2 `comp_ref_type` ctx).
/// Every compound block we emit is unidirectional `(LAST_FRAME, LAST2_FRAME)`
/// and no block uses backward references.
fn comp_ref_type_context(above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
    let comp = |n: Option<ModeInfo>| n.is_some_and(|m| m.is_compound());
    match (above, left) {
        (Some(a), Some(l)) if a.is_inter() && l.is_inter() => {
            match (a.is_compound(), l.is_compound()) {
                (false, false) => 3,
                (false | true, true) | (true, false) => 4,
            }
        }
        (Some(_), Some(_)) => {
            if comp(above) || comp(left) {
                3
            } else {
                2
            }
        }
        _ => {
            if comp(above) || comp(left) {
                4
            } else {
                2
            }
        }
    }
}

/// Context of `skip`: the number of skipped neighbours.
fn skip_context(above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
    usize::from(above.is_some_and(|m| m.skip)) + usize::from(left.is_some_and(|m| m.skip))
}

/// Context of `is_inter` (libaom `av1_get_intra_inter_context`).
fn intra_inter_context(above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
    match (above, left) {
        (Some(a), Some(l)) => match (!a.is_inter(), !l.is_inter()) {
            (true, true) => 3,
            (false, false) => 0,
            _ => 1,
        },
        (Some(n), None) | (None, Some(n)) => 2 * usize::from(!n.is_inter()),
        (None, None) => 0,
    }
}

/// Context of every single-reference symbol when all inter blocks use
/// `LAST_FRAME` (libaom `av1_get_pred_context_single_ref_p1/p3/p4`, whose
/// counts then tie at zero without an inter neighbour).
fn last_only_ref_context(above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
    if above.is_some_and(|m| m.is_inter()) || left.is_some_and(|m| m.is_inter()) {
        REF_COUNT_CTX_GREATER
    } else {
        REF_COUNT_CTX_EQUAL
    }
}

/// Contexts of `(single_ref_p1 / single_ref_p3 / uni_comp_ref, single_ref_p4, uni_comp_ref_p1)`
/// (AV1 §8.3.2 `ref_count_ctx`, libaom `av1_get_pred_context_single_ref_p1/p3/p4`).
/// Every inter neighbour uses `LAST_FRAME` and/or `LAST2_FRAME`
/// ([`ModeInfo::uses_last`], [`ModeInfo::uses_last2`]).
fn ref_count_contexts(above: Option<ModeInfo>, left: Option<ModeInfo>) -> (usize, usize, usize) {
    let last = [above, left]
        .iter()
        .filter(|n| n.is_some_and(|m| m.uses_last()))
        .count();
    let last2 = [above, left]
        .iter()
        .filter(|n| n.is_some_and(|m| m.uses_last2()))
        .count();
    let near_ctx = if last + last2 == 0 {
        REF_COUNT_CTX_EQUAL
    } else {
        REF_COUNT_CTX_GREATER
    };
    let p4_ctx = match last.cmp(&last2) {
        std::cmp::Ordering::Less => REF_COUNT_CTX_LESS,
        std::cmp::Ordering::Equal => REF_COUNT_CTX_EQUAL,
        std::cmp::Ordering::Greater => REF_COUNT_CTX_GREATER,
    };
    let uni_p1_ctx = if last2 == 0 {
        REF_COUNT_CTX_EQUAL
    } else {
        REF_COUNT_CTX_GREATER
    };
    (near_ctx, p4_ctx, uni_p1_ctx)
}

/// The `uv_mode` alphabet of a block: `CfL` is allowed for blocks up to
/// 32x32 (AV1 §5.11.24 `is_cfl_allowed`). Returns the CDF index and the
/// number of symbols.
fn uv_mode_alphabet(block: &Block) -> (usize, usize) {
    if block.log2_size() <= 5 {
        (CFL_ALLOWED, UV_INTRA_MODES_CFL_ALLOWED)
    } else {
        (CFL_NOT_ALLOWED, UV_INTRA_MODES_CFL_NOT_ALLOWED)
    }
}

/// `size_group_lookup` of a square block (8x8 → 1, 16x16 → 2, 32x32 → 3).
fn size_group(block: &Block) -> usize {
    usize::from(block.log2_size() - 2).min(3)
}

/// The position of `block` if it copies the reference's luma: inter (at
/// `mv`) with zero motion and no luma residual (`luma`, transform `leaves`).
fn luma_copy_at(
    block: &Block,
    mv: Option<Av1Mv>,
    luma: &PlaneResidual,
    leaves: &[LumaLeaf],
) -> Option<(usize, usize)> {
    let copy = mv == Some(Av1Mv::default()) && luma.eob == 0 && leaves.is_empty();
    copy.then_some((block.x, block.y))
}

fn to_av1_mv(mv: Mv) -> Av1Mv {
    Av1Mv::checked(
        i32::from(mv.y) * AV1_MV_PER_ENGINE_MV,
        i32::from(mv.x) * AV1_MV_PER_ENGINE_MV,
    )
    .expect("engine vectors stay within AV1's vector range")
}

/// An AV1 vector in engine units. List vectors in this encoder are always
/// whole samples, so the division is exact.
fn to_engine_mv(mv: Av1Mv) -> Mv {
    Mv {
        x: (i32::from(mv.col) / AV1_MV_PER_ENGINE_MV) as i16,
        y: (i32::from(mv.row) / AV1_MV_PER_ENGINE_MV) as i16,
    }
}

/// Partitions summed for `split_or_horz` (libaom `partition_gather_vert_alike`;
/// the 4-way split is included because superblocks are never 128x128).
const VERT_ALIKE: [Partition; 6] = [
    Partition::Vert,
    Partition::Split,
    Partition::HorzA,
    Partition::VertA,
    Partition::VertB,
    Partition::Vert4,
];
/// Partitions summed for `split_or_vert` (libaom `partition_gather_horz_alike`).
const HORZ_ALIKE: [Partition; 6] = [
    Partition::Horz,
    Partition::Split,
    Partition::HorzA,
    Partition::HorzB,
    Partition::VertA,
    Partition::Horz4,
];

/// A boolean inverse CDF whose "1" probability is the summed probability of
/// `parts` in the partition CDF (libaom `partition_gather_{horz,vert}_alike`;
/// AV1 §8.3.2 `split_or_horz` / `split_or_vert`).
fn gather(icdf: &[u16], parts: &[Partition]) -> [u16; 2] {
    let prob = |p: Partition| -> u32 {
        let i = p as usize;
        let before = if i > 0 {
            u32::from(icdf[i - 1])
        } else {
            1 << 15
        };
        before - u32::from(icdf[i])
    };
    let sum: u32 = parts.iter().map(|&p| prob(p)).sum();
    [sum as u16, 0]
}

#[path = "decoder/mod.rs"]
pub(crate) mod decoder;

/// A cap only: preserve the configured candidate when the optional scale is0.
/// Applied only after the existing strong-P-frame content guards have passed.
fn strong_luma_primary(configured: f32, modeled: u8, scale: f32) -> u8 {
    let configured = configured.round() as u8;
    if scale > 0.0 {
        configured.min((f32::from(modeled) * scale).round() as u8)
    } else {
        configured
    }
}

/// Chroma offset for a key frame that keeps the inter frames'
/// chroma-to-luma step ratio: the chroma qindex inter frames use at the
/// equivalent inter quality (`inter_q_idx + inter_delta`) scaled by the same key-frame step factor as
/// luma, expressed relative to the key frame's own qindex. Subtracting
/// `inter_delta` from the boosted key qindex instead lands where the AC table
/// is steep: at inter q120 the key chroma step became 0.35x the key luma
/// step (0.68x on inter frames), and at q108 chroma qindex clamped to 0
/// (`bodycam_airport` key 70.6 KB against 26.2 KB with no chroma offset, same
/// luma PSNR).
fn key_chroma_delta(inter_q_idx: u32, key_q_idx: u32, inter_delta: i32, key_scale: f64) -> i32 {
    let inter_chroma =
        (inter_q_idx as i32 + inter_delta).clamp(QINDEX_MIN as i32, QINDEX_MAX as i32);
    let key_chroma = scaled_qindex(inter_chroma as u32, key_scale);
    (key_chroma as i32 - key_q_idx as i32).clamp(DELTA_Q_MIN, DELTA_Q_MAX)
}

/// The `delta_q` that puts the chroma AC step of frame qindex `q_idx`
/// nearest the engine chroma policy's step (`nc_encoder::chroma_policy`;
/// AV1 8-bit steps are 8x the H.264-style sample step).
fn policy_chroma_delta(policy: nc_encoder::chroma_policy::ChromaPolicy, q_idx: u32) -> i32 {
    let ac = |q: i32| f64::from(QuantSteps::for_qindex(q.clamp(0, QINDEX_MAX as i32) as u32).ac);
    let luma = ac(q_idx as i32);
    let target = policy.chroma_step(luma / AV1_STEP_SCALE) * AV1_STEP_SCALE;
    nc_encoder::chroma_policy::closest_index(
        DELTA_Q_MIN..=DELTA_Q_MAX,
        |d| ac(q_idx as i32 + d),
        target,
    )
}

/// AV1 8-bit quantiser steps over the H.264-normalised sample step
/// (`nc_encoder::chroma_policy::q_of_step`): qindex 255 maps to about QP 51.
const AV1_STEP_SCALE: f64 = 8.0;

/// Retain the chroma allocation on clean input, without spending it on noise.
/// Scale 2 puts five TRAIN noise families above the cutoff; natural lowlight
/// is 0.22–0.49, though natural motion can also cause high spikes. Reuses frame
/// analysis without another pixel pass. With base -24, the 15-natural-clip TRAIN
/// mean PSNR/VMAF-v1 BD-rate is -7.77%/-6.28% versus offset 0. Noise protection
/// trades some clean-content gain for avoiding chroma allocation to noise.
/// Native 360p CBR CPU changes are within noise (-0.14% body, +0.09% crowd).
/// See docs/handoff/av1.md for the scope, reuse proofs and held-out validation.
fn noise_adapted_chroma_delta(base: f32, sigma: f32, scale: f32) -> i32 {
    (nc_encoder::noise_attenuated_chroma_offset(base, sigma, scale).round() as i32)
        .clamp(DELTA_Q_MIN, DELTA_Q_MAX)
}

#[cfg(test)]
mod parameter_tests {
    use super::*;
    use nc_encoder::ParamOverrides;

    #[test]
    fn strong_model_cap_preserves_default_and_accepts_fractional_scales() {
        assert_eq!(Av1Params::default().cdef_local_strong_model_scale, 0.0);
        for model in 0..=15 {
            assert_eq!(strong_luma_primary(15.0, model, 0.0), 15);
            assert!(strong_luma_primary(9.0, model, 3.0) <= 9);
        }
        assert_eq!(strong_luma_primary(15.0, 1, 3.0), 3);
        assert_eq!(strong_luma_primary(15.0, 2, 3.0), 6);
        assert_eq!(strong_luma_primary(15.0, 4, 3.0), 12);
        assert_eq!(strong_luma_primary(15.0, 5, 3.0), 15);
        assert_eq!(strong_luma_primary(15.0, 1, 2.5), 3);
        assert_eq!(strong_luma_primary(15.0, 0, 3.0), 0);
        for value in [0.0, 0.125, 2.5, 3.0, 16.0] {
            let mut overrides = ParamOverrides::default();
            overrides.set("av1.cdef_local_strong_model_scale", value);
            assert!(overrides.validate(Av1Backend::PARAM_SPECS).is_ok());
        }
        for value in [-0.1, 16.1, f32::NAN, f32::INFINITY] {
            let mut overrides = ParamOverrides::default();
            overrides.set("av1.cdef_local_strong_model_scale", value);
            assert!(overrides.validate(Av1Backend::PARAM_SPECS).is_err());
        }
    }

    #[test]
    fn noise_chroma_defaults_and_scale_are_range_checked() {
        assert_eq!(Av1Params::default().chroma_delta_q, -32.0);
        assert_eq!(Av1Params::default().chroma_noise_scale, 2.0);
        for value in [0.0, 0.125, 2.0, 32.0] {
            let mut overrides = ParamOverrides::default();
            overrides.set("av1.chroma_noise_scale", value);
            assert!(overrides.validate(Av1Backend::PARAM_SPECS).is_ok());
        }
        for value in [-0.1, 32.1, f32::NAN, f32::INFINITY] {
            let mut overrides = ParamOverrides::default();
            overrides.set("av1.chroma_noise_scale", value);
            assert!(overrides.validate(Av1Backend::PARAM_SPECS).is_err());
        }
        for sigma in [0.0, 0.25, 2.0, 12.0] {
            assert_eq!(noise_adapted_chroma_delta(0.0, sigma, 2.0), 0);
            assert_eq!(noise_adapted_chroma_delta(-24.0, sigma, 0.0), -24);
        }
        assert_eq!(noise_adapted_chroma_delta(-24.0, 0.0, 2.0), -24);
        assert_eq!(noise_adapted_chroma_delta(-24.0, 1.0, 2.0), -12);
        assert_eq!(noise_adapted_chroma_delta(-24.0, 2.0, 2.0), 0);
        assert_eq!(noise_adapted_chroma_delta(-24.0, 12.0, 2.0), 0);
        assert_eq!(noise_adapted_chroma_delta(24.0, 1.25, 2.5), 12);
    }

    #[test]
    fn scaled_key_chroma_keeps_the_inter_step_ratio() {
        // Inter q120: chroma qindex 96. Key luma 30; scaling chroma 96 by the
        // same 0.24 gives qindex 18 (delta -12), not 30 - 24 = 6.
        assert_eq!(key_chroma_delta(120, 30, -24, 0.24), -12);
        // q108: the unscaled offset would clamp chroma to qindex 0.
        assert_eq!(key_chroma_delta(108, 24, -24, 0.24), -9);
        assert_eq!(key_chroma_delta(120, 30, 0, 0.24), 0);
        let mut backend = prediction_backend();
        backend.params.chroma_delta_q = -24.0;
        backend.params.keyframe_chroma_scaled = 1.0;
        let key = |quantizer| FramePlan {
            index: 0,
            frame_type: FrameType::Key,
            quantizer,
            references: nc_encoder::FrameReferences::FLAT,
            lambda_scale: 1.0,
            block_qp: false,
        };
        backend.begin_frame(&key(120));
        assert_eq!((backend.frame_q_idx, backend.chroma_delta_q()), (30, -12));
        let _ = backend.end_frame();
        backend.commit_frame();
        // Rate control picks the key qindex directly; its inter equivalent
        // (the first qindex at or above step 37 / 0.24) is 121, one above.
        backend.set_rate_control_enabled(true);
        backend.begin_frame(&key(30));
        assert_eq!((backend.frame_q_idx, backend.chroma_delta_q()), (30, -11));
    }

    #[test]
    fn chroma_quantization_and_header_keep_one_frame_offset() {
        let mut backend = prediction_backend();
        backend.params.chroma_delta_q = -24.0;
        // The unscaled key offset, so every frame carries the noise offset.
        backend.params.keyframe_chroma_scaled = 0.0;
        backend.params.chroma_noise_scale = 2.0;
        for (index, sigma, delta) in [(0, 0.0, -24), (1, 1.0, -12), (2, 5.0, 0)] {
            backend.set_noise_sigma(sigma);
            backend.begin_frame(&FramePlan {
                index,
                frame_type: if index == 0 {
                    FrameType::Key
                } else {
                    FrameType::Inter
                },
                quantizer: 100,
                references: nc_encoder::FrameReferences::FLAT,
                lambda_scale: 1.0,
                block_qp: false,
            });
            assert_eq!(backend.chroma_delta_q(), delta);
            assert_eq!(
                backend.chroma_steps,
                QuantSteps::for_qindex((backend.frame_q_idx as i32 + delta).clamp(0, 255) as u32)
            );
            // Header emission must retain the value used for this frame's pixels.
            backend.set_noise_sigma(100.0);
            assert_eq!(backend.chroma_delta_q(), delta);
            let _ = backend.end_frame();
            backend.commit_frame();
        }
    }

    #[test]
    fn mds3_threshold_defaults_and_limits_are_registered() {
        let defaults = Av1Params::default();
        assert_eq!(defaults.shave_min_zero_run, 8.0);
        assert_eq!(defaults.chroma_skip_sad_num, 5.0);
        for name in ["av1.shave_min_zero_run", "av1.chroma_skip_sad_num"] {
            for value in [0.0, 8.0, 32.0] {
                let mut overrides = ParamOverrides::default();
                overrides.set(name, value);
                assert!(overrides.validate(Av1Backend::PARAM_SPECS).is_ok());
            }
            for value in [-1.0, 33.0, f32::NAN, f32::INFINITY] {
                let mut overrides = ParamOverrides::default();
                overrides.set(name, value);
                assert!(overrides.validate(Av1Backend::PARAM_SPECS).is_err());
            }
        }
    }

    #[test]
    fn pre_transform_controls_are_optional_and_range_checked() {
        assert_eq!(Av1Params::default().pre_tx_skip, 1.0);
        assert_eq!(Av1Params::default().pre_tx_skip_score_nonbase, 0.0);
        for (name, low, high) in [
            ("av1.pre_tx_skip", 0.0, 1.0),
            ("av1.pre_tx_skip_energy", 0.0, 100.0),
            ("av1.pre_tx_skip_score", 0.0, 250.0),
            ("av1.pre_tx_skip_score_nonbase", 0.0, 250.0),
            ("av1.pre_tx_skip_rd", 0.0, 400.0),
            ("av1.pre_tx_luma_dominant", 0.0, 1.0),
        ] {
            for value in [low, high] {
                let mut overrides = ParamOverrides::default();
                overrides.set(name, value);
                assert!(overrides.validate(Av1Backend::PARAM_SPECS).is_ok());
            }
            for value in [low - 1.0, high + 1.0, f32::NAN, f32::INFINITY] {
                let mut overrides = ParamOverrides::default();
                overrides.set(name, value);
                assert!(overrides.validate(Av1Backend::PARAM_SPECS).is_err());
            }
        }
    }
    fn prediction_backend() -> Av1Backend {
        Av1Backend::new(&EncoderConfig {
            codec: nc_encoder::Codec::Av1,
            width: 128,
            height: 128,
            frame_rate: nc_encoder::FrameRate { num: 30, den: 1 },
            quantizer: 100,
            intra_period: 300,
            block_log2: 6,
            key_qp_offset: None,
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal::default(),
            params: ParamOverrides::default(),
        })
        .unwrap()
    }

    #[test]
    fn nonbase_pre_transform_score_inherits_and_preserves_fractional_thresholds() {
        let mut backend = prediction_backend();
        backend.params.pre_tx_skip_score = 70.0;
        backend.params.pre_tx_skip_score_nonbase = 50.25;
        for enabled in [0.0, 1.0] {
            backend.params.pre_tx_skip = enabled;
            for index in 0..8 {
                backend.begin_frame(&FramePlan {
                    index,
                    frame_type: if index == 0 {
                        FrameType::Key
                    } else {
                        FrameType::Inter
                    },
                    quantizer: 100,
                    references: nc_encoder::FrameReferences::FLAT,
                    lambda_scale: 1.0,
                    block_qp: false,
                });
                let expected = if enabled > 0.0 && index % 4 != 0 {
                    50.25
                } else {
                    70.0
                };
                assert_eq!(backend.pre_tx_score_this_frame, expected);
                let _ = backend.end_frame();
                backend.commit_frame();
            }
        }
        backend.params.pre_tx_skip_score_nonbase = 0.0;
        backend.begin_frame(&FramePlan {
            index: 9,
            frame_type: FrameType::Inter,
            quantizer: 100,
            references: nc_encoder::FrameReferences::FLAT,
            lambda_scale: 1.0,
            block_qp: false,
        });
        assert_eq!(backend.pre_tx_score_this_frame, 70.0);
    }

    #[test]
    fn exact_luma_cache_is_keyed_and_invalidated_at_frame_boundary() {
        let mut backend = prediction_backend();
        let block = Block::square(16, 16, 4);
        let mv = Mv { x: -1, y: 3 };
        let reference = Picture::new(128, 128, 0).unwrap();
        let prediction = Prediction::Inter {
            reference: &reference,
            mv: to_av1_mv(mv),
        };
        let stats = nc_dsp::Variance { sse: 99, sum: -7 };
        backend.cache_exact_luma(&block, mv, nc_dsp::Filter::Av1, &[47; 256], stats);
        assert_eq!(backend.cached_luma_stats(&block, &prediction), Some(stats));
        for other in [Block { x: 32, ..block }, Block::square(block.x, block.y, 5)] {
            assert_eq!(backend.cached_luma_stats(&other, &prediction), None);
        }
        let other_mv = Prediction::Inter {
            reference: &reference,
            mv: to_av1_mv(Mv { x: 1, ..mv }),
        };
        assert_eq!(backend.cached_luma_stats(&block, &other_mv), None);
        backend.cache_exact_luma(&block, mv, nc_dsp::Filter::HevcLuma, &[47; 256], stats);
        assert_eq!(backend.cached_luma_stats(&block, &prediction), None);
        backend.cache_exact_luma(&block, mv, nc_dsp::Filter::Av1, &[47; 256], stats);
        backend.begin_frame(&FramePlan {
            index: 2,
            frame_type: FrameType::Key,
            quantizer: 100,
            references: nc_encoder::FrameReferences::FLAT,
            lambda_scale: 1.0,
            block_qp: false,
        });
        assert_eq!(backend.cached_luma_stats(&block, &prediction), None);
    }

    /// The transform-based key-frame leaf cost keeps flat superblocks whole
    /// and splits detail a 64x64 transform cannot code (its top half of the
    /// frequencies is never coded, AV1 §5.11.39).
    #[test]
    fn intra_leaf_cost_splits_only_uncodable_detail() {
        let mut backend = prediction_backend();
        backend.begin_frame(&FramePlan {
            index: 0,
            frame_type: FrameType::Key,
            quantizer: 100,
            references: nc_encoder::FrameReferences::FLAT,
            lambda_scale: 1.0,
            block_qp: false,
        });
        let mut source = Picture::new(128, 128, 0).unwrap();
        source.plane_mut(PlaneIndex::Y).view_mut().fill(100);
        // Top-left superblock: columns alternating 40/160, Nyquist detail.
        for y in 0..64 {
            for (x, v) in source.plane_mut(PlaneIndex::Y).view_mut().row_mut(y)[..64]
                .iter_mut()
                .enumerate()
            {
                *v = if x % 2 == 0 { 40 } else { 160 };
            }
        }
        let mut split_and_whole = |x: usize, y: usize| {
            let sb = Block::square(x, y, 6);
            let whole = backend.intra_leaf_cost(&source, &sb).unwrap();
            let split: f32 = sb
                .quadrants()
                .iter()
                .map(|child| backend.intra_leaf_cost(&source, child).unwrap())
                .sum();
            (split, whole)
        };
        let (split, whole) = split_and_whole(0, 0);
        assert!(split < whole, "detail: split {split} vs whole {whole}");
        let (split, whole) = split_and_whole(64, 64);
        assert!(whole < split, "flat: whole {whole} vs split {split}");
    }

    #[test]
    fn cached_luma_initial_prediction_restore_and_misses_match_canonical_filter() {
        let mut backend = prediction_backend();
        let mut reference = Picture::new(128, 128, 64).unwrap();
        for y in 0..128 {
            for (x, v) in reference
                .plane_mut(PlaneIndex::Y)
                .view_mut()
                .row_mut(y)
                .iter_mut()
                .enumerate()
            {
                *v = ((x * 17 + y * 31 + (x ^ y) * 7) % 256) as u8;
            }
        }
        reference.extend_borders();
        let mut recon = Picture::new(128, 128, 0).unwrap();
        for log2_size in [3, 4, 5, 6] {
            let n = 1 << log2_size;
            for mv in [
                Mv { x: -3, y: 1 },
                Mv { x: 5, y: -7 },
                Mv { x: -300, y: 350 },
            ] {
                let block = Block::square(0, 0, log2_size);
                let prediction = Prediction::Inter {
                    reference: &reference,
                    mv: to_av1_mv(mv),
                };
                let mut expected = vec![0; n * n];
                let mut dst = nc_common::PlaneViewMut::new(&mut expected, n, n, n).unwrap();
                inter::predict(
                    &backend.dsp,
                    reference.plane(PlaneIndex::Y),
                    0,
                    0,
                    n,
                    to_av1_mv(mv),
                    false,
                    &mut dst,
                );
                backend.cache_exact_luma(
                    &block,
                    mv,
                    nc_dsp::Filter::Av1,
                    &expected,
                    nc_dsp::Variance::default(),
                );
                for _ in 0..2 {
                    recon.plane_mut(PlaneIndex::Y).view_mut().fill(255);
                    backend.predict_plane(&block, PlaneIndex::Y, &prediction, &mut recon);
                    for y in 0..n {
                        assert_eq!(
                            &recon.plane(PlaneIndex::Y).view().row(y)[..n],
                            &expected[y * n..][..n]
                        );
                    }
                }
                // A different vector must follow the unchanged interpolation fallback.
                let miss = Prediction::Inter {
                    reference: &reference,
                    mv: to_av1_mv(Mv { x: mv.x + 1, ..mv }),
                };
                backend.predict_plane(&block, PlaneIndex::Y, &miss, &mut recon);
                let actual: Vec<_> = (0..n)
                    .flat_map(|y| recon.plane(PlaneIndex::Y).view().row(y)[..n].to_vec())
                    .collect();
                backend.scratch.cached_luma = None;
                backend.predict_plane(&block, PlaneIndex::Y, &miss, &mut recon);
                for y in 0..n {
                    assert_eq!(
                        &recon.plane(PlaneIndex::Y).view().row(y)[..n],
                        &actual[y * n..][..n]
                    );
                }
            }
        }
    }
}
