//! The VP9 backend of the shared engine: the engine plans partitions and
//! chooses each block's prediction (intra DC or an inter vector); this
//! backend writes VP9 syntax, transforms, quantizes and reconstructs
//! (see `nc_encoder::CodecBackend` for the call order).
//!
//! Coding tools: 64x64 superblocks with square partitions down to 8x8,
//! the largest transform of each block (`ALLOW_32X32`), DC intra, single
//! reference inter prediction (NEAREST/NEAR/ZERO/NEW, 8-tap regular), skip,
//! deadzone quantization and the deblocking filter with a quantizer-derived
//! level. Frames are error resilient with default probabilities.
//!
//! The real-time shortcuts `vp9.nonrd_skip_scale` and
//! `vp9.residual_bits_per_level` reimplement libvpx's `vp9_pickmode.c`
//! (BSD-3-Clause; copyright notice and licence in `THIRD_PARTY.md`).

use std::rc::Rc;

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_encoder::{
    Block, BlockDecision, Codec, CodecBackend, ConfigError, EncoderConfig, FramePlan, FrameType,
    MAX_REFERENCE_SLOTS, Mv as EngineMv, PredictionMode, QuantParams, ReferenceChoice, SlotMask,
    SpatialConfig, SpatialLayerFrame, spatial::INTER_LAYER_SCRATCH_MASK,
};

use crate::block_qp::SegmentQuantizers;
use crate::bool_coder::BitSink;
use crate::coeff::{self, Model, PlaneType};
use crate::contexts::{self, EntropyContexts};
use crate::header::{ALL_SLOTS, FrameHeader, MAX_DIMENSION};
use crate::inter_pred::{self, BlockPosition};
use crate::mvref::{self, MvRefs};
use crate::probs::{CoefUpdates, Counts, FrameContext, SWITCHABLE_CONTEXTS, SWITCHABLE_FILTERS};
use crate::recon::{self, TxBlock};
use crate::recorder::{self, TileRecorder};
use crate::segmentation::{self, SegmentFeatures, SegmentMap, Segmentation};
use crate::stress::RandomSyntax;
use crate::tables::{self, AC_QUANT};
use crate::transform::{self, Rounding, TxSize};
use crate::tree;
use crate::types::{
    InterMode, IntraMode, MI_LOG2, MI_SIZE, MIN_BLOCK_LOG2, ModeInfo, ModeInfoGrid, Mv,
    SUPERBLOCK_LOG2, max_tx_size, size_group, uv_tx_size,
};
use crate::{loopfilter, motion};

/// The non-directional modes and V/H a pruned intra search starts with.
const BASE_INTRA_MODES: [IntraMode; 4] = [IntraMode::Dc, IntraMode::V, IntraMode::H, IntraMode::Tm];
/// The luma modes libvpx's real-time speeds try (`INTRA_DC_H_V`,
/// `vp9_speed_features.c` `intra_y_mode_mask`; their chroma mask is
/// `INTRA_DC`).
const REAL_TIME_LUMA_MODES: [IntraMode; 3] = [IntraMode::Dc, IntraMode::V, IntraMode::H];

/// How an intra block chooses its modes (`vp9.intra_mode_search`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntraSearch {
    /// All ten modes.
    All,
    /// The four basic modes, then the two directions beside the winner.
    Pruned,
    /// libvpx real-time: luma DC, V or H, chroma DC.
    RealTime,
}

impl IntraSearch {
    fn of(param: f32) -> Self {
        match param.round() as u8 {
            0 => Self::All,
            1 => Self::Pruned,
            _ => Self::RealTime,
        }
    }
}

/// The two directional modes nearest `mode`'s direction (diagonals for DC
/// and TM).
fn directions_beside(mode: IntraMode) -> [IntraMode; 2] {
    match mode {
        IntraMode::V => [IntraMode::D63, IntraMode::D117],
        IntraMode::H => [IntraMode::D153, IntraMode::D207],
        _ => [IntraMode::D45, IntraMode::D135],
    }
}

/// Intra modes the encoder chooses from: all ten.
const CODED_INTRA_MODES: [IntraMode; 10] = [
    IntraMode::Dc,
    IntraMode::V,
    IntraMode::H,
    IntraMode::D45,
    IntraMode::D135,
    IntraMode::D117,
    IntraMode::D153,
    IntraMode::D207,
    IntraMode::D63,
    IntraMode::Tm,
];
/// VP9's rate model: the shared qindex calibration, plus the start-up
/// lookahead's first-key cap (`rate.first_key_lookahead`) of 0.3 buffers
/// under camera motion. `J_real` TRAIN, 150 f, vs off: -0.2% (pan -0.9%,
/// lowlight -0.9%, bodycam -0.5%, texture +0.3%, worst-1% Y -1.2%; crowd
/// and static byte-identical, no guard tripped).
///
/// QP cadence 1.2x the engine's, re-tuned once MS-SSIM was scored on every
/// frame instead of every 4th (mostly base frames). `J_real` (k 0.1) vs 1:
/// TRAIN -1.3%, VAL -0.6%, held-out `broad_independent_v1` (13 clips, every
/// class) -1.5%; worst-1% VMAF +2.6% there (a 60 fps pan's start-up
/// transient). 1.4 gains more (TRAIN -2.0%, VAL -1.4%, held-out -2.1%) but
/// its held-out worst-1% VMAF is +3.8% (the same pan +44%). `vp9-fast`
/// takes 1.4 (TRAIN -1.9%, VAL -1.7%, no guard).
const RATE_MODEL: nc_encoder::RateModel = nc_encoder::RateModel {
    cadence_scale: 1.2,
    first_key_lookahead: 0.3,
    ..nc_encoder::RateModel::AC_LOOKUP_QINDEX
};
/// Lowest and highest `base_q_idx`.
const QINDEX_MAX: u32 = 255;
/// VP9 vectors are in 1/8 samples, the engine's in 1/4.
const MV_PER_ENGINE_MV: i32 = 2;
/// AC quantizer table entries per sample-domain quantizer step (the DCT
/// coefficients are 8x orthonormal).
const AC_Q_PER_SAMPLE_STEP: f32 = 8.0;
/// Denominator of `vp9.keyframe_step_permille`.
const KEY_STEP_SCALE_DENOMINATOR: f64 = 1000.0;
/// libvpx `LPF_PICK_FROM_Q` (`vp9_picklpf.c`): filter level
/// `round((q * 20723 + 1015158) / 2^18)` from the AC step `q` (about
/// `0.316 q + 3.87`), 4 lower on key frames.
const FILTER_FROM_Q_SLOPE: i32 = 20_723;
const FILTER_FROM_Q_OFFSET: i32 = 1_015_158;
const FILTER_FROM_Q_SHIFT: u32 = 18;
const KEY_FILTER_REDUCTION: i32 = 4;
/// Samples of the largest block in all planes (64x64 luma, 4:2:0).
const MAX_BLOCK_SAMPLES: usize = 64 * 64 * 3 / 2;
/// Luma samples of the largest block.
const MAX_BLOCK_AREA: usize = 64 * 64;
/// The inter rounding divisor of `Rounding::Inter`.
const INTER_ROUNDING_DIVISOR: u8 = 6;
/// Quadrants of a block, for the residual-spread test.
const QUADRANTS: f32 = 4.0;
/// Vector units (1/8 sample) per luma sample.
const MV_UNITS_PER_SAMPLE: i32 = 8;
/// Largest transform block area.
const MAX_TX_AREA: usize = 32 * 32;
/// Transform blocks of one coded block: 64x64 luma at 4x4 (lossless) plus chroma.
const MAX_TX_BLOCKS: usize = 256 + 2 * 64;
/// The segment of skipped inter blocks under `vp9.skip_filter_off` and
/// its filter-level delta (down to level 0).
const SKIP_SEGMENT: usize = 1;
const MAX_FILTER_DELTA: i8 = 63;
/// Segment-tree probabilities of a frame header before its ids are coded:
/// the frame's own replace them (`own_segment_tree_probs`).
const PENDING_TREE_PROBS: [u8; segmentation::TREE_PROBS] = [255; segmentation::TREE_PROBS];
/// Side (log2) of the square sharing one block quantizer: a superblock,
/// so a frame's offsets stay few enough for its eight segments.
const BLOCK_QP_LOG2: u8 = 6;

nc_encoder::param_group! {
    pub(crate) struct Vp9Params ("vp9") {
        lambda_per_step_squared: 0.09, [0.01, 1.0],
            "mode-decision lambda per squared sample-domain AC step (TRAIN screen; AV1 uses 0.135)",
        keyframe_step_permille: 240.0, [100.0, 1000.0],
            "fixed-QP key frame AC step relative to the inter step, per mille (AV1's value)",
        chroma_delta_q: -8.0, [-15.0, 15.0],
            "chroma DC/AC qindex offset from the frame qindex (negative is finer)",
        chroma_delta_q_coarse: 0.0, [-15.0, 0.0],
            "extra chroma qindex offset reached at vp9.chroma_delta_q_coarse_end, ramped linearly from zero at vp9.chroma_delta_q_coarse_start (0 off; AV2's): coarse frames take finer chroma, fine-quantizer frames keep vp9.chroma_delta_q",
        chroma_delta_q_coarse_start: 150.0, [0.0, 255.0],
            "frame qindex at which vp9.chroma_delta_q_coarse starts to apply",
        chroma_delta_q_coarse_end: 210.0, [1.0, 255.0],
            "frame qindex from which the whole vp9.chroma_delta_q_coarse applies",
        chroma_delta_gap_slope: 0.0, [0.0, 4.0],
            "chroma qindex offset change per dB that the previous frame's chroma was cleaner than its luma beyond vp9.chroma_delta_gap_pivot (finer above it, coarser below; 0 off; AV2's)",
        chroma_delta_gap_pivot: 10.0, [0.0, 30.0],
            "luma-chroma PSNR gap (dB) at which vp9.chroma_delta_gap_slope leaves the chroma offset unchanged",
        filter_level_scale: 1.0, [0.0, 2.0],
            "multiplies the quantizer-derived loop-filter level (libvpx LPF_PICK_FROM_Q)",
        interp_filter: 3.0, [0.0, 3.0],
            "inter frames' interpolation filter: 0 smooth, 1 regular, 2 sharp, 3 switchable per block",
        high_precision_qindex: 0.0, [0.0, 256.0],
            "inter frames below this qindex allow 1/8-sample vectors (libvpx HIGH_PRECISION_MV_QTHRESH)",
        /// 1 (Default): `J_real --clips adopt` full TRAIN -0.9% / VAL -1.2%
        /// (both BETTER, no guard); keys alone (crowd, 4 intra frames,
        /// qindex 130-240) BD-rate PSNR-YUV -3.3%; -0.6% instructions
        /// (crowd 720p 20 f CBR 400). Periodic-key pulse +0.13 / +0.16 dB.
        /// vp9-fast keeps 0 (not screened there).
        intra_mode_estimate: 1.0, [0.0, 1.0],
            "intra mode choice measures: 0 the first transform block's prediction SSE, 1 the whole block's prediction SATD (up to 32x32; HEVC's CU-level search)",
        /// 2 is a speed lever (libvpx real-time intra masks, 2026-11-13):
        /// measured in docs/handoff/vp9-faster.md.
        intra_mode_search: 0.0, [0.0, 2.0],
            "intra mode choice: 0 tries all ten modes, 1 the four basic ones and the two directions beside the best, 2 luma DC/V/H and chroma DC only (libvpx real-time)",
        /// 4 with `shave_gap` 8 (HEVC's pair): J_real TRAIN proxy-6 -1.2%
        /// (full -1.5%, low -1.0%; NEG -3.3%, MS-SSIM -1.4%, VMAF -1.3%;
        /// every class better), no guards; +8.2% Jetson / +5.3% x86 CPU
        /// (more coded levels, more transform trials): -0.5% at Default's
        /// CPU weight (k 0.1). Long GOP (one key, 300 frames, crowd and
        /// low-light at 400 kbps) holds its trend. Shaving alone measured
        /// +0.1%; rounding 4 alone -1.0% on crowd and low-light, -0.1% on
        /// TRAIN.
        inter_rounding_divisor: 4.0, [2.0, 6.0],
            "inter luma quantizer rounding offset 1/n of the step (HM 6; HEVC's encoder uses 4)",
        filter_sharpness: 0.0, [0.0, 7.0],
            "loop-filter sharpness (VP9 sharpness_level): higher filters textured edges less",
        chroma_rounding: 2.0, [0.0, 2.0],
            "chroma quantizer rounding: 0 the block's (1/3 intra, 1/6 inter), 1 at least 1/3, 2 nearest (1/2)",
        coef_updates: 1.0, [0.0, 1.0],
            "forward coefficient probability updates in the compressed header",
        intra_tx_depth: 2.0, [0.0, 3.0],
            "intra blocks try this many smaller transforms below the largest",
        inter_tx_depth: 2.0, [0.0, 3.0],
            "inter blocks try this many smaller transforms below the largest (TX_MODE_SELECT)",
        /// A speed lever for Fast (default off): crowd 720p at q230, 2:
        /// Jetson -14.4%, x86 -8.0%; 1.5: -7.2% / -3.9%. J_real TRAIN quality
        /// at 1.5 (proxy-6) +1.6% full / +1.2% low; at 2 (proxy-4) +3.5% /
        /// +2.0% with class guards tripped.
        inter_tx_split_spread: 1.0, [1.0, 4.0],
            "smaller inter transforms are tried only when the luma residual's largest quadrant holds at least this fraction x 1/4 of its energy (1 always): higher is faster, keeps evenly spread residual in one transform",
        /// See `inter_rounding_divisor`. Default no longer shaves (with
        /// key trellis, ungated P-frame intra and fine-quantizer small
        /// blocks: see `DEFAULT_PARAMS`); vp9-fast keeps 8.
        shave_gap: 0.0, [0.0, 16.0],
            "drop trailing ±1 levels at least this many scan positions past the previous level (SVT shave_coeff; 0 off): lower drops more",
        /// A speed lever for Fast: crowd 720p q230 Jetson -3.9%, x86 +1.6%
        /// (full forward then zeroed there); J_real TRAIN quality +0.2% /
        /// +0.3% (proxy-6, no guards): about 0 at Default's k 0.1.
        partial_forward: 0.0, [0.0, 1.0],
            "32x32 transforms code only their low 16x16 frequencies (an encoder-side forward approximation, SVT's TX shortcut): faster, drops fine detail in large blocks",
        /// A speed lever (HEVC's `pre_transform_zero_scale`, default 1:
        /// exact). Measured for vp9-faster: docs/handoff/vp9-faster.md.
        pre_transform_zero_scale: 1.0, [1.0, 4.0],
            "inter residual zero-bound multiplier before the transform: 1 exact (only blocks proven to quantise to zero skip it), higher drops near-zero inter residual untransformed",
        /// A speed lever (HEVC's `max_transform_log2`, default 5: 32x32).
        /// Measured for vp9-faster: docs/handoff/vp9-faster.md.
        inter_max_tx_log2: 5.0, [3.0, 5.0],
            "largest transform of inter blocks (log2: 3 8x8, 4 16x16, 5 32x32): smaller is faster (the 32x32 forward and its scans) and codes fine detail of large blocks dearer",
        /// A speed lever (AV1's `pre_tx_skip_energy`, SVT's light-PD1
        /// energy test; default 0: off). Measured for vp9-faster:
        /// docs/handoff/vp9-faster.md.
        pre_tx_skip_energy: 0.0, [0.0, 100.0],
            "inter blocks whose luma prediction error is below this energy (100 x SSE < energy x area x qindex) code no luma residual, without transforms: higher skips more",
        /// A speed lever (HEVC's and H.264's `inter_intra_dc_only`,
        /// default 0). Measured for vp9-faster: docs/handoff/vp9-faster.md.
        inter_intra_dc_only: 0.0, [0.0, 1.0],
            "1 codes intra blocks of inter frames with DC luma and chroma prediction, without a mode search",
        /// A speed lever (libvpx's real-time intra blocks, which code one
        /// transform size; default 3: no cap). Measured for vp9-faster:
        /// docs/handoff/vp9-faster.md.
        inter_intra_tx_depth: 3.0, [0.0, 3.0],
            "intra blocks of inter frames try at most this many smaller transforms (the lesser of this and intra_tx_depth): lower is faster",
        /// A speed lever (libvpx's real-time skip test, default 0: off).
        /// Alone on vp9-fast at 1: Jetson -5.0% CPU, J_real quality +2.0%
        /// (pool L1, 2026-11-12); vp9-faster takes it (handoff).
        nonrd_skip_scale: 0.0, [0.0, 8.0],
            "inter transform blocks whose residual variance is below this x libvpx's AC threshold (ac_q^2 / 64) skip the transform: coding only DC from the residual sum, or nothing when its mean is below the DC threshold too (0 off)",
        /// A speed lever (libvpx's real-time rate, default 0: off). Alone
        /// on vp9-fast at 4: Jetson -8.0% CPU, J_real quality +3.1% (pool
        /// L1, 2026-11-12); vp9-faster takes it (handoff).
        /// A speed lever (libvpx's block-level skip test, default 0).
        /// Measured for vp9-faster: docs/handoff/vp9-faster.md.
        nonrd_block_test: 0.0, [0.0, 1.0],
            "with nonrd_skip_scale, 1 first tests each inter plane block as a whole (libvpx's averaged test): a block with nothing to code skips its transform blocks unvisited",
        residual_bits_per_level: 0.0, [0.0, 16.0],
            "residual RD and transform-size trials price inter levels at this many bits per unit of level magnitude (libvpx block_yrd: 4) instead of their token costs (0)",
        /// A speed lever (libvpx's real-time filter search, default 0).
        /// Measured for vp9-faster: docs/handoff/vp9-faster.md.
        filter_search_rt: 0.0, [0.0, 1.0],
            "1 tries only the regular and smooth filters on switchable-filter blocks (libvpx real-time), not sharp",
        tx_search_min_levels: 0.0, [0.0, 16.0],
            "inter blocks whose largest transform codes at most this many nonzero levels skip the smaller sizes (0 searches all)",
        leaf_cost_transform_domain: 1.0, [0.0, 1.0],
            "key-frame planning measures a leaf's distortion on its coefficients (no inverse transform)",
        skip_filter_off: 0.0, [0.0, 1.0],
            "inter frames put skipped inter blocks in a segment with loop-filter level 0",
        residual_rd: 1.0, [0.0, 1.0],
            "inter blocks drop their residual when skipping costs less (SSE + lambda * bits)",
        chroma_sse_weight: 4.0, [0.5, 8.0],
            "weight of chroma squared error against luma in the residual-drop decision",
        chroma_weight_adapt: 0.0, [0.0, 4.0],
            "residual-drop chroma weight from the previous frame's MSE_Y / MSE_C times this gain (clamped 0.25..16) in place of vp9.chroma_sse_weight (0 off; AV2's)",
    }
}

/// Whether the frame quantizer is fixed or chosen by rate control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantizerMode {
    Fixed,
    RateControlled,
}

/// One coded transform block's levels, kept until its tokens are written.
/// Compact (12 bytes): trial saves and restores copy every block's.
#[derive(Clone, Copy)]
struct CodedTx {
    plane: PlaneIndex,
    tx: TxSize,
    ty: transform::TxType,
    eob_nonzero: bool,
    /// Top-left in 4x4 units of the plane (the coefficient-context grid).
    unit: (u16, u16),
    levels_offset: u32,
}

impl CodedTx {
    fn new(
        block: &TxBlock,
        ty: transform::TxType,
        eob_nonzero: bool,
        levels_offset: usize,
    ) -> Self {
        let unit = |sample: usize| u16::try_from(sample >> 2).expect("VP9 frames fit 16-bit units");
        Self {
            plane: block.plane,
            tx: block.tx,
            ty,
            eob_nonzero,
            unit: (unit(block.x), unit(block.y)),
            levels_offset: u32::try_from(levels_offset).expect("per-block levels fit 32 bits"),
        }
    }

    /// The coefficient probability model of the block.
    fn model(&self, inter: bool) -> Model {
        Model::of(self.plane, self.tx, self.ty, inter)
    }

    /// The block's levels in the scratch buffer.
    fn levels(&self) -> std::ops::Range<usize> {
        let start = self.levels_offset as usize;
        start..start + self.tx.area()
    }

    fn units(&self) -> (usize, usize) {
        (usize::from(self.unit.0), usize::from(self.unit.1))
    }
}

/// Reusable per-block buffers.
struct Scratch {
    levels: Vec<i32>,
    coded: Vec<CodedTx>,
    /// One transform block's residual and coefficients (only the block's
    /// area is used; reused, not zeroed).
    residual: Vec<i16>,
    coefficients: Vec<i32>,
    /// The current inter block's prediction, all planes packed.
    prediction: Vec<u8>,
    /// One block's luma prediction while choosing its filter, and the best
    /// one so far.
    filter_trial: Vec<u8>,
    filter_best: Vec<u8>,
    /// The best transform-size trial so far: its reconstruction (packed
    /// like `prediction`), levels and transform blocks.
    best_recon: Vec<u8>,
    best_levels: Vec<i32>,
    best_coded: Vec<CodedTx>,
    /// Chroma transform blocks kept while a trial recodes only luma.
    kept_chroma: Vec<CodedTx>,
    /// One inter plane block's 4x4 SADs against its prediction, raster
    /// order ([`plane_quad_sads`]).
    quad_sads: Vec<u32>,
}

impl Scratch {
    /// Forget the last trial's transform blocks, keeping its chroma ones
    /// aside when `planes` recodes luma only.
    fn start_trial(&mut self, planes: TrialPlanes) {
        self.kept_chroma.clear();
        if planes == TrialPlanes::LumaOnly {
            let first_chroma = self.coded.partition_point(|c| c.plane == PlaneIndex::Y);
            self.kept_chroma.extend(self.coded.drain(first_chroma..));
        }
        self.coded.clear();
    }

    /// Put back the chroma blocks [`Self::start_trial`] kept, after luma.
    fn finish_trial(&mut self) {
        self.coded.append(&mut self.kept_chroma);
    }
}

/// Coding state that depends on the coded frame size: its size, mode-info and
/// context grids, and segmentation state. Spatial SVC keeps one per layer and
/// swaps them whole when the coded layer changes, so switching layers inside a
/// temporal unit never reallocates.
#[derive(Debug)]
struct LayerGrids {
    width: usize,
    height: usize,
    /// Segment ids of the last segmented frame (`last_frame_seg_map`).
    segment_map: SegmentMap,
    /// The segment features of the last committed frame's header, which a
    /// frame that is not past-independent continues from.
    committed_segment_features: SegmentFeatures,
    grid: ModeInfoGrid,
    /// The previous frame's mode info, for its co-located vectors.
    previous: Option<ModeInfoGrid>,
    /// Whether the previous frame was a key frame.
    last_key: bool,
    contexts: EntropyContexts,
    /// `MSE_Y / MSE_C` of the layer's last committed frame
    /// (`nc_encoder::plane_error_ratio`), when a chroma control reads it.
    chroma_error_ratio: Option<f64>,
    /// The current frame's ratio, committed by `commit_frame` (a recoded
    /// frame measures it again).
    frame_chroma_error_ratio: Option<f64>,
    /// Spatial SVC: the decoder's probability context of this layer
    /// (`frame_context_idx` = its spatial id), which its next frame starts
    /// from. A single-layer stream keeps it in [`Vp9Backend::context`].
    context: FrameContext,
}

impl LayerGrids {
    fn new(width: usize, height: usize) -> Self {
        let (coded_width, coded_height) = coded_size(width, height);
        let (mi_cols, mi_rows) = (coded_width >> MI_LOG2, coded_height >> MI_LOG2);
        Self {
            width,
            height,
            segment_map: SegmentMap::cleared(mi_cols, mi_rows),
            committed_segment_features: SegmentFeatures::default(),
            grid: ModeInfoGrid::new(mi_cols, mi_rows),
            previous: None,
            last_key: true,
            contexts: EntropyContexts::new(mi_cols),
            chroma_error_ratio: None,
            frame_chroma_error_ratio: None,
            context: FrameContext::default(),
        }
    }
}

/// Normative reference picture and size (`RefFrameWidth`, `RefFrameHeight`,
/// VP9 §8.5.2.3) held for base-layer (`S0`) slots so `S1` predicts from `S0`
/// via scaled convolution (`vp9_convolve_scaled`) while the engine's `S1`
/// pool holds an upscaled full-size proxy for motion estimation.
#[derive(Debug, Clone)]
struct HeldReference {
    picture: Rc<Picture>,
    width: usize,
    height: usize,
}

impl HeldReference {
    fn new(picture: Rc<Picture>, width: usize, height: usize) -> Self {
        Self {
            picture,
            width,
            height,
        }
    }

    fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

/// Held reference pictures by DPB slot (`None`: the engine's pool picture is
/// exact).
#[derive(Debug, Default)]
struct ReferenceStore {
    slots: [Option<HeldReference>; MAX_REFERENCE_SLOTS],
}

impl ReferenceStore {
    fn held(&self, slot: usize) -> Option<&HeldReference> {
        self.slots[slot].as_ref()
    }

    fn commit_spatial(
        &mut self,
        refresh: SlotMask,
        refresh_all: bool,
        picture: Option<&HeldReference>,
    ) {
        for (slot, held) in self.slots.iter_mut().enumerate() {
            if refresh_all || refresh.contains(slot) {
                match picture {
                    Some(pic) => {
                        if let Some(existing) = held {
                            existing.clone_from(pic);
                        } else {
                            *held = Some(pic.clone());
                        }
                    }
                    None => *held = None,
                }
            }
        }
    }
}

/// The largest enhancement/base size ratio VP9 spatial SVC encodes: the
/// scaled prediction is stress-tested (`syntax_stress.py --spatial`) at
/// 1.25x to 2x; VP9 itself allows up to 16x.
const MAX_SPATIAL_UPSCALE: usize = 2;

/// Whether a frame of `frame` size may predict from a reference of
/// `reference` size in VP9 (§7.2, frame size with refs semantics; libvpx
/// `valid_ref_frame_size`): `2 * width >= ref_width` and
/// `width <= 16 * ref_width`, and the same vertically. The reference is at
/// most twice the frame's size and at least 1/16 of it.
fn reference_scale_allowed(reference: (usize, usize), frame: (usize, usize)) -> bool {
    let fits = |r: usize, f: usize| 2 * f >= r && f <= 16 * r;
    fits(reference.0, frame.0) && fits(reference.1, frame.1)
}

/// Two-layer spatial SVC state on [`Vp9Backend`].
#[derive(Debug)]
struct BackendSpatial {
    /// The grids of the layer not being coded (the active one is
    /// [`Vp9Backend::grids`]).
    parked: LayerGrids,
    active_spatial_id: u8,
    current_layer: SpatialLayerFrame,
    /// The last coded picture, of either layer, was a key frame: the
    /// decoder adapts the next frame's probabilities at the after-key rate
    /// (`COUNT_SAT_AFTER_KEY`), whatever its layer.
    previous_key: bool,
    /// A base temporal-layer-0 picture has been coded since the last key
    /// (see [`Self::context_use`]).
    base_since_key: bool,
    /// An error-resilient enhancement picture reset every context of the
    /// receivers that decoded it but not of base-only receivers, so the
    /// base layer's pictures are error resilient too, up to the next one of
    /// temporal layer 0.
    base_context_diverged: bool,
    /// The probabilities the current frame is coded with: its starting
    /// context plus its forward updates (`end_frame`), which adaptation
    /// keeps where it does not adapt.
    coded_context: FrameContext,
    /// The size of the picture in each reference slot (the decoder's
    /// `RefFrameWidth` / `RefFrameHeight`).
    slot_sizes: [Option<(usize, usize)>; MAX_REFERENCE_SLOTS],
}

impl BackendSpatial {
    /// How the current layer's frame uses the probability contexts. Each
    /// layer adapts its own context (`frame_context_idx` = its spatial id)
    /// from its own earlier pictures, as a single-layer stream does, so
    /// neither restarts from the defaults every frame (error resilience
    /// cost the 360p base 6-18% at equal quantizer). Every operating point
    /// must decode a frame identically, which a frame's context and
    /// adaptation rate allow only when they do not depend on frames some
    /// receivers drop:
    /// - temporal layers above 0 read their layer's context but do not
    ///   refresh it (a receiver may drop them);
    /// - the base's first temporal-layer-0 picture after a key does not
    ///   refresh: whether it adapts at the after-key rate depends on
    ///   whether the receiver decoded the key unit's enhancement picture;
    /// - an enhancement repair is error resilient (the receiver lost the
    ///   picture that refreshed its context), which resets every context of
    ///   receivers of both layers, so the base's pictures are error
    ///   resilient too, up to its next temporal-layer-0 one.
    ///
    /// The previous frame's vectors are never used (`previous_vectors`):
    /// every base picture is hidden and every enhancement picture follows
    /// one, so no decoder uses them (`use_prev_frame_mvs`, VP9 §7.2).
    fn context_use(&self, key: bool) -> ContextUse {
        let layer = &self.current_layer;
        let base = layer.spatial_id == 0;
        if key {
            return ContextUse {
                error_resilient: false,
                index: 0,
                refresh: true,
            };
        }
        // An enhancement picture coded afresh (from the base, or intra)
        // outside a key unit repairs a lost one (`Encoder::report_layer_loss`).
        let repair = !base && !layer.key_unit && (layer.from_base_only || layer.intra_only);
        if (base && self.base_context_diverged) || repair {
            return ContextUse {
                error_resilient: true,
                index: 0,
                refresh: false,
            };
        }
        ContextUse {
            error_resilient: false,
            index: layer.spatial_id,
            refresh: layer.temporal_id == 0 && (!base || self.base_since_key),
        }
    }

    /// Advance the cross-layer state past a committed frame.
    fn committed(&mut self, header: &FrameHeader) {
        let (key, error_resilient) = (header.key, header.error_resilient);
        for (slot, size) in self.slot_sizes.iter_mut().enumerate() {
            if header.refresh & (1 << slot) != 0 {
                *size = Some((header.width, header.height));
            }
        }
        let base = self.current_layer.spatial_id == 0;
        self.previous_key = key;
        if key {
            self.base_since_key = false;
            self.base_context_diverged = false;
        } else if base {
            // Every base picture stays error resilient until one of
            // temporal layer 0, which every base receiver decodes.
            let lowest = self.current_layer.temporal_id == 0;
            self.base_since_key |= lowest;
            self.base_context_diverged &= !lowest;
        } else if error_resilient {
            self.base_context_diverged = true;
        }
    }
}

/// How a frame uses the decoder's probability contexts (VP9 §7.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ContextUse {
    /// `error_resilient_mode`: every context is reset first, nothing is
    /// adapted, and the previous frame's vectors are not used.
    error_resilient: bool,
    /// `frame_context_idx`.
    index: u8,
    /// `refresh_frame_context` (with backward adaptation).
    refresh: bool,
}

/// One block's reference vectors and inter reference bits under one
/// reference, cached for the mode decision's candidates.
#[derive(Debug, Clone, Copy)]
struct BlockRefs {
    /// `(row, col, size_log2, secondary reference)`.
    key: (usize, usize, u8, bool),
    refs: MvRefs,
    /// [`Vp9Backend::reference_bits`].
    reference_bits: f32,
}

/// VP9 encoder backend.
pub struct Vp9Backend {
    /// The frame's [`FramePlan::lambda_scale`], applied to every λ.
    lambda_scale: f32,
    /// The frame-size state of the layer being coded (one per spatial
    /// layer under spatial SVC, swapped whole by [`BackendSpatial`]).
    grids: LayerGrids,
    /// Reference slots whose normative picture is at `S0` resolution while
    /// the engine's `S1` pool holds an upscaled proxy.
    store: ReferenceStore,
    /// Spatial SVC configuration and parked layer state when active.
    spatial: Option<BackendSpatial>,
    /// Active reference choice (`Primary` = `LAST_FRAME`, `Secondary` =
    /// `GOLDEN_FRAME`) during mode evaluation and block coding.
    active_ref: ReferenceChoice,
    params: Vp9Params,
    /// Whether block quantizers are enabled (`enable_block_qp`).
    block_qp_enabled: bool,
    /// Block quantizers through segmentation, on frames whose blocks may
    /// carry offsets (`FramePlan::block_qp`): the segment of each offset
    /// this frame codes.
    block_qp: Option<SegmentQuantizers>,
    /// The qindex the current block is coded at: the frame's outside
    /// `encode_block`, its segment's inside.
    coding_q_idx: u8,
    quantizer_mode: QuantizerMode,
    header: FrameHeader,
    /// The segment features in force before the current frame's header.
    segment_state: SegmentFeatures,
    lossless: bool,
    /// The engine chroma policy (`chroma.policy` 1): replaces
    /// `vp9.chroma_delta_q` and `vp9.chroma_sse_weight`.
    chroma_policy: Option<nc_encoder::chroma_policy::ChromaPolicy>,
    /// Probabilities of the current frame (adapted after each frame).
    context: FrameContext,
    /// Mode symbol bits under `context`, priced once per frame.
    mode_costs: Box<ModeCosts>,
    /// Symbols coded in the current frame.
    counts: Counts,
    writer: TileRecorder,
    /// Reference vectors of the block being decided, cached per position and
    /// reference (`(row, col, log2_size, secondary_ref)`).
    refs: Option<BlockRefs>,
    scratch: Scratch,
    buffers: Box<recon::ResidualBuffers>,
    /// Inter prediction scratch.
    inter: recon::InterScratch,
    /// Kernel dispatch (forward transforms).
    dsp: nc_dsp::Dsp,
    /// Random syntax for the stress test (`debug.random_decisions`) only.
    random: Option<RandomSyntax>,
    /// `quant.trellis`: trellis quantisation of every transform block.
    trellis: Option<Vp9Trellis>,
}

/// log2 of the 4x4-sample units of the entropy contexts.
const TX_UNIT_LOG2: usize = 2;

/// The registered trellis controls (`nc_encoder::QuantParams`) and the
/// search's buffers.
#[derive(Debug)]
struct Vp9Trellis {
    /// Multiple of the mode-decision λ.
    lambda_scale: f32,
    /// Further multiple for chroma; `None` leaves chroma unoptimised.
    chroma_scale: Option<f32>,
    /// Whether key frames are trellised too.
    key_frames: bool,
    /// Survivors per scan position.
    max_states: usize,
    search: coeff::trellis::TrellisQuantizer,
}

impl Vp9Trellis {
    /// Trellis-quantise the levels of `tx_block` (transform type `ty` of
    /// an inter or intra block) from its `coefficients`, starting from the
    /// plain quantiser's `levels` (some nonzero) and priced with the
    /// frame's probabilities and the neighbours' nonzero contexts as they
    /// stand. Returns whether any level is left.
    fn quantize(
        &mut self,
        (fc, contexts, tx_block, inter): (&FrameContext, &EntropyContexts, &TxBlock, bool),
        ty: transform::TxType,
        (lambda, coefficients, steps): (f32, &[i32], transform::Steps),
        levels: &mut [i32],
    ) -> bool {
        let model = Model::of(tx_block.plane, tx_block.tx, ty, inter);
        let probs = &fc.coef[model.tx as usize][model.plane as usize][usize::from(model.inter)];
        let ctx = contexts.coefficient(
            tx_block.plane,
            tx_block.x >> TX_UNIT_LOG2,
            tx_block.y >> TX_UNIT_LOG2,
            model.tx,
        );
        let mut rates = coeff::trellis::RateModel::new(model, probs, ctx);
        self.search.quantize(
            &mut rates,
            coefficients,
            steps,
            (lambda, self.max_states),
            levels,
        )
    }

    fn from_params(quant: &QuantParams) -> Option<Self> {
        (quant.trellis >= 0.5).then(|| Self {
            lambda_scale: quant.trellis_lambda_scale,
            chroma_scale: (quant.trellis_chroma > 0.0).then_some(quant.trellis_chroma),
            key_frames: quant.trellis_key >= 0.5,
            max_states: quant.trellis_states.round() as usize,
            search: coeff::trellis::TrellisQuantizer::default(),
        })
    }
}

impl std::fmt::Debug for Vp9Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vp9Backend")
            .field("header", &self.header)
            .finish_non_exhaustive()
    }
}

/// `vp9.interp_filter` values.
const INTERP_FILTERS: [nc_dsp::Vp9Filter; 3] = [
    nc_dsp::Vp9Filter::Smooth,
    nc_dsp::Vp9Filter::Regular,
    nc_dsp::Vp9Filter::Sharp,
];

/// Visible sizes are any 4:2:0 size (the engine requires even ones).
const VISIBLE_MULTIPLE: usize = 1;

/// The 8-aligned coded size (`MiCols * 8`): the frame header carries the
/// visible size and the decoder walks whole 8x8 units.
fn coded_size(width: usize, height: usize) -> (usize, usize) {
    (
        width.next_multiple_of(MI_SIZE),
        height.next_multiple_of(MI_SIZE),
    )
}

impl CodecBackend for Vp9Backend {
    /// `begin_frame` continues from committed state only (probability
    /// context, segment map and features, previous grid, last key);
    /// `commit_frame` adapts and advances it.
    const RECODE: bool = true;

    // Scaling λ alone reaches low targets here: crowd_mot20_06 CBR 150 kbps,
    // 150 f, headroom 2 octaves: 309 -> 159 kbps.
    const SUPPORTS_LAMBDA_HEADROOM: bool = true;

    const PARAM_SPECS: &'static [&'static [nc_encoder::ParamSpec]] = &[Vp9Params::SPECS];
    /// Every `vp9.*` parameter is read per frame (`set_frame_params`): each
    /// is a block decision or a choice the frame header carries, so a
    /// non-base frame (`nonbase.vp9.*`, `frame_role.rs`) may take its own.
    const FRAME_PARAM_SPECS: &'static [&'static [nc_encoder::ParamSpec]] = &[Vp9Params::SPECS];

    const SUPPORTS_BLOCK_QP: bool = true;
    const BLOCK_QP_LOG2: u8 = BLOCK_QP_LOG2;
    const BLOCK_QP_OFFSETS: usize = segmentation::MAX_SEGMENTS - 1;

    /// The engine's motion decisions (AV1's quality point: sub-sample move
    /// margins 1, the four-neighbour integer probe, refining the best cheap
    /// vector as a NEW candidate at gate 0.05) except the sub-sample seed
    /// gate, which VP9's screen kept at 0. `J_real` TRAIN proxy-6 against
    /// the legacy search (CBR ladder, 150 frames): -1.4% (full -1.8%, low
    /// -1.1%), no clip worse (pan -3.7%, bodycam -2.2%), PSNR -3.2%,
    /// MS-SSIM -2.2%, worst-1% Y -2.3%, VMAF NEG +0.2%; +4.3% instructions
    /// (30 frames 720p crowd q200).
    ///
    /// Trellis quantisation (`quant.*`, docs/handoff/trellis.md) of P-frame
    /// luma, one survivor: `J_real` staged TRAIN (18 clips) at k 0.1 BETTER,
    /// quality -1.3% full / -1.2% low, every class better (crowd -2.1%,
    /// lowlight -2.0%, bodycam -1.4%, texture -1.1%, pan -0.9%, static
    /// -0.5%), 1/18 clips worse (`static_meva_hospital` +0.4%), NEG -2.2%,
    /// VMAF v1 -2.7%, no guard; periodic-key pulse 0.00 dB. CPU: P frames
    /// +6.4% host (148.2 -> 157.7 M, q180), Jetson +11.2% (crowd q180): J
    /// -0.3 with 2:1 weighting. Single-key long GOPs at equal rate, YUV per
    /// third: airport 600 kbps +0.04/+0.07/+0.06 dB, chirla 150 kbps
    /// +0.01/+0.04/-0.02, homes390 300 kbps -0.05/-0.08/-0.10. vp9-fast keeps
    /// it off.
    ///
    /// Key-frame trellis (`quant.trellis_key` 1), no level shaving
    /// (`vp9.shave_gap` 0), P-frame intra tried on every block
    /// (`decision.intra_gate_variance` 0) and small blocks kept on offer
    /// below relative quantizer 0.86 (`planner.depth_removal_min_quantizer`),
    /// as one bundle: `J_real --clips adopt` (proxy-6 + val-proxy-5, k 0.6)
    /// BETTER -2.9% (full -3.0%, low -2.8%), no guard (val-proxy-5 alone
    /// -1.9%, bodycam +4.0%; proxy-6 -3.3%); periodic-key pulse -0.01 /
    /// +0.02 dB. Alone (6 TRAIN clips, 60 f): `trellis_key` + shave 0 -1.7%,
    /// intra gate 0 -0.7%, depth removal 0.86 -0.6%. Instructions (crowd
    /// 720p 20 f CBR 400, callgrind): +1.7%, +0.1%, +0.8%. vp9-fast keeps
    /// the engine values.
    const DEFAULT_PARAMS: &'static [(&'static str, f32)] = &[
        // Tuned jointly (eval/nceval/tune_pool.py, study
        // vp9-default-eq2-2026-11-11, equal x86 CPU): planner lambda 2 ->
        // 3.16 (fewer splits). Study clips J -1.97%, L3 of the study's draw
        // -0.87% [-1.31, -0.43] (overfit gap +1.09); re-checked at head
        // cd9ea802 on the study's rotation: disjoint L3 (79 clips,
        // single-point) J -0.74% [-1.18, -0.30], L2 (ladder) -1.21% [-1.58,
        // -0.83], no guard. CPU x86 -1.5%, Jetson cycles -1.1%. vp9-fast pins
        // the previous value.
        ("planner.lambda_scale", 3.162_5),
        // Start-up QP cadence (period 8, base frames 4 QP finer, for the
        // first 100 frames after a key, then the engine's period 4 / -3 / +1)
        // and the cadence motion gate (`qp_cadence.rs`; docs/handoff/vvc.md,
        // qp-cascade). J_real (k 0.1, dCPU 0): boost 6 proxy-4 worst-1% Y
        // +3.0%; boost 4 TRAIN (18 clips) BETTER -2.6% (static -6.4%,
        // lowlight -5.2%, crowd -2.9%), worst-1% Y -1.2%, pulse -0.02 dB;
        // VAL BETTER -6.5% (static -22.0%, crowd -7.8%, bodycam -2.2%), pulse
        // +0.19 dB. Long GOP (900 f, one key, CBR 300): chirla Y 37.14 ->
        // 37.18, pan_bdd 30.67 -> 30.70.
        ("rate.cadence_startup_period", 8.0),
        ("rate.cadence_startup_base_boost", 4.0),
        ("rate.cadence_motion_gate", 1.0),
        ("decision.codec_subpel_seed_lambdas", 0.0),
        ("quant.trellis", 1.0),
        ("quant.trellis_key", 1.0),
        ("quant.trellis_states", 1.0),
        ("quant.trellis_chroma", 0.0),
        ("decision.intra_gate_variance", 0.0),
        ("planner.depth_removal_min_quantizer", 0.86),
        // AV2's error-gap chroma offset: chroma 3 qindex finer per dB that
        // the previous frame's chroma was cleaner than its luma beyond 10 dB
        // (coarser below). VP9 trails our HEVC mostly on VMAF v1's chroma
        // feature. J_real --clips adopt (k 0.6, 2026-11-12): L1 -1.30%, L2
        // -0.98% [-1.32, -0.68], L3 (ladder, L2+L3 pooled, n 127) BETTER
        // -0.84% [-1.13, -0.59]; every class and every term better (L3
        // PSNR-YUV -1.1%, VMAF v1 -0.9%, MS-SSIM -0.9%), no guard. Slope 2 /
        // 4 and pivot 8 / 12 ranked below 3 / 10 at L1. Long GOP (900 f,
        // one key, CBR 400): luma unchanged per third, chroma crowd -0.2 dB,
        // pan +0.1..+0.3 dB, no drift. No extra work (one row-subsampled
        // plane error per frame). vp9-fast, -faster and -fastest pin 0 (not
        // screened there).
        ("vp9.chroma_delta_gap_slope", 3.0),
        // VP9 spatial SVC's enhancement-layer intra gate (the engine's
        // `spatial.enhancement_intra_gate`): its measured results include it
        // (docs/handoff/spatial-svc.md). Single-layer streams never read it.
        ("spatial.enhancement_intra_gate", 1.0),
    ];

    fn new(config: &EncoderConfig) -> Result<Self, ConfigError> {
        if config.codec != Codec::Vp9 {
            return Err(ConfigError::Unsupported("VP9 backend requires VP9 codec"));
        }
        if config.signal.colour != nc_common::ColourInfo::default() {
            return Err(ConfigError::Unsupported(
                "VP9 colour metadata overrides are not yet supported",
            ));
        }
        if config.width == 0
            || config.height == 0
            || config.width > MAX_DIMENSION
            || config.height > MAX_DIMENSION
        {
            return Err(ConfigError::Dimensions {
                width: config.width,
                height: config.height,
                multiple: VISIBLE_MULTIPLE,
            });
        }
        if config.quantizer > QINDEX_MAX {
            return Err(ConfigError::Quantizer {
                value: config.quantizer,
                min: 0,
                max: QINDEX_MAX,
            });
        }
        if !(MIN_BLOCK_LOG2..=SUPERBLOCK_LOG2).contains(&config.block_log2) {
            return Err(ConfigError::BlockSize(config.block_log2));
        }
        if config.key_qp_offset.is_some() {
            return Err(ConfigError::Unsupported(
                "VP9 sets its key-frame quantizer with vp9.keyframe_step_permille",
            ));
        }
        let qindex = config.quantizer as u8;
        Ok(Self {
            lambda_scale: 1.0,
            grids: LayerGrids::new(config.width, config.height),
            store: ReferenceStore::default(),
            spatial: None,
            active_ref: ReferenceChoice::Primary,
            params: Vp9Params::from_overrides(&config.params),
            block_qp_enabled: false,
            block_qp: None,
            coding_q_idx: qindex,
            quantizer_mode: QuantizerMode::Fixed,
            segment_state: SegmentFeatures::default(),
            header: FrameHeader {
                key: true,
                show_frame: true,
                error_resilient: false,
                context_index: 0,
                refresh_context: true,
                width: config.width,
                height: config.height,
                ref_slots: [0; 3],
                size_from_ref: Some(0),
                refresh: ALL_SLOTS,
                filter_level: 0,
                sharpness: 0,
                interp_filter: Some(nc_dsp::Vp9Filter::Regular),
                allow_hp: false,
                base_q_idx: qindex,
                segmentation: Segmentation::disabled(SegmentFeatures::default()),
                chroma_delta_q: 0,
            },
            lossless: qindex == 0,
            chroma_policy: nc_encoder::chroma_policy::ChromaPolicy::from_params(&config.params),
            context: FrameContext::default(),
            mode_costs: ModeCosts::new(&FrameContext::default()),
            counts: Counts::default(),
            writer: TileRecorder::default(),
            refs: None,
            scratch: Scratch {
                levels: vec![0; MAX_TX_BLOCKS * 16],
                coded: Vec::with_capacity(MAX_TX_BLOCKS),
                residual: vec![0; MAX_TX_AREA],
                coefficients: vec![0; MAX_TX_AREA],
                prediction: vec![0; MAX_BLOCK_SAMPLES],
                best_recon: vec![0; MAX_BLOCK_SAMPLES],
                filter_trial: vec![0; MAX_BLOCK_AREA],
                filter_best: vec![0; MAX_BLOCK_AREA],
                best_levels: vec![0; MAX_TX_BLOCKS * 16],
                best_coded: Vec::with_capacity(MAX_TX_BLOCKS),
                kept_chroma: Vec::with_capacity(MAX_TX_BLOCKS),
                quad_sads: vec![0; QUAD_SADS],
            },
            random: None,
            trellis: Vp9Trellis::from_params(&QuantParams::from_overrides(&config.params)),
            buffers: recon::ResidualBuffers::new(&nc_dsp::Dsp::new()),
            inter: recon::InterScratch::new(),
            dsp: nc_dsp::Dsp::new(),
        })
    }

    fn superblock_log2(&self) -> u8 {
        SUPERBLOCK_LOG2
    }

    fn min_block_log2(&self) -> u8 {
        MIN_BLOCK_LOG2
    }

    /// Temporal layers and LTR recovery reference frames other than the
    /// previous one; those frames are error resilient (`begin_frame`).
    fn supports_temporal_layers(&self) -> bool {
        true
    }

    fn supports_second_reference(&self) -> bool {
        self.spatial.is_some()
    }

    fn select_reference(&mut self, _block: &Block, reference: ReferenceChoice) {
        self.active_ref = reference;
    }

    fn supports_spatial_layers(&self) -> bool {
        true
    }

    fn configure_spatial(&mut self, spatial: &SpatialConfig) -> Result<(), ConfigError> {
        if !reference_scale_allowed(spatial.base_size, spatial.full_size) {
            return Err(ConfigError::SpatialScalability(
                "VP9 spatial layer scale outside §7.2 reference limits (base at least 1/16 of the full size)",
            ));
        }
        let (base, full) = (spatial.base_size, spatial.full_size);
        if full.0 > MAX_SPATIAL_UPSCALE * base.0 || full.1 > MAX_SPATIAL_UPSCALE * base.1 {
            return Err(ConfigError::SpatialScalability(
                "VP9 spatial layers are verified up to a 2:1 base (spatial.base_scale >= 0.5)",
            ));
        }
        self.grids = LayerGrids::new(spatial.full_size.0, spatial.full_size.1);
        self.spatial = Some(BackendSpatial {
            parked: LayerGrids::new(spatial.base_size.0, spatial.base_size.1),
            active_spatial_id: 1,
            current_layer: SpatialLayerFrame {
                spatial_id: 1,
                temporal_id: 0,
                width: spatial.full_size.0,
                height: spatial.full_size.1,
                key_unit: true,
                intra_only: false,
                from_base_only: false,
                max_bitrate_bps: None,
            },
            previous_key: false,
            base_since_key: false,
            base_context_diverged: false,
            coded_context: FrameContext::default(),
            slot_sizes: [None; MAX_REFERENCE_SLOTS],
        });
        Ok(())
    }

    fn prepare_spatial_layer(&mut self, layer: SpatialLayerFrame) {
        let Some(sp) = self.spatial.as_mut() else {
            return;
        };
        if sp.active_spatial_id != layer.spatial_id {
            std::mem::swap(&mut self.grids, &mut sp.parked);
            sp.active_spatial_id = layer.spatial_id;
        }
        debug_assert_eq!(
            (self.grids.width, self.grids.height),
            (layer.width, layer.height)
        );
        sp.current_layer = layer;
        self.active_ref = ReferenceChoice::Primary;
    }

    fn commit_spatial_reference(
        &mut self,
        spatial_id: u8,
        recon: &Picture,
        size: (usize, usize),
        refresh: SlotMask,
        refresh_all: bool,
    ) {
        let held =
            (spatial_id == 0).then(|| HeldReference::new(Rc::new(recon.clone()), size.0, size.1));
        self.store
            .commit_spatial(refresh, refresh_all, held.as_ref());
    }

    fn finish_spatial_tu(&mut self, data: &mut Vec<u8>, layer_lengths: &[usize]) {
        crate::superframe::pack_spatial_tu(data, layer_lengths);
    }

    fn supports_long_term_reference(&self) -> bool {
        true
    }

    fn supports_inter(&self) -> bool {
        true
    }

    fn coded_size(&self, width: usize, height: usize) -> (usize, usize) {
        coded_size(width, height)
    }

    /// At the current block's quantizer inside `encode_block` (the engine
    /// scales its own ranking λ for block offsets, so it sees the frame's).
    fn lambda(&self) -> f32 {
        let step = AC_QUANT[usize::from(self.coding_q_idx)] as f32 / AC_Q_PER_SAMPLE_STEP;
        self.params.lambda_per_step_squared * step * step * self.lambda_scale
    }

    fn relative_quantizer(&self) -> f32 {
        f32::from(self.header.base_q_idx) / QINDEX_MAX as f32
    }

    /// The cheapest inter leaf's mode bits, plus its filter symbol when
    /// the frame switches filters per block.
    fn planner_leaf_bits(&self) -> f32 {
        let filter = if self.header.interp_filter.is_none() {
            self.mode_costs
                .switchable
                .iter()
                .flatten()
                .copied()
                .fold(f32::INFINITY, f32::min)
        } else {
            0.0
        };
        self.mode_costs.leaf + filter
    }

    fn set_frame_params(&mut self, params: &nc_encoder::ParamOverrides) {
        self.params = Vp9Params::from_overrides(params);
    }

    fn rate_model(&self) -> nc_encoder::RateModel {
        RATE_MODEL
    }

    fn rate_control_quantizer_step(&self, quantizer: u32) -> f64 {
        f64::from(AC_QUANT[quantizer.min(QINDEX_MAX) as usize])
    }

    /// Frames whose blocks may carry offsets then code segment ids,
    /// segment 0 at the frame quantizer (`block_qp.rs`). Other frames
    /// leave segmentation to `vp9.skip_filter_off`: segment ids on every
    /// block of a frame that refines a few cost a 4-clip TRAIN proxy +2.9%
    /// `J_real` (`static_cam22` key 48.7 KB against 46.0 KB).
    fn enable_block_qp(&mut self) {
        self.block_qp_enabled = true;
    }

    fn block_qp_offset(&mut self, requested: i16) -> i16 {
        match self.block_qp.as_mut() {
            Some(segments) if self.header.segmentation.enabled => segments.assign(requested).1,
            _ => 0,
        }
    }

    fn enable_random_decisions(&mut self, rng: nc_encoder::DebugRng) {
        self.random = Some(RandomSyntax::new(rng));
    }

    fn set_rate_control_enabled(&mut self, enabled: bool) {
        self.quantizer_mode = if enabled {
            QuantizerMode::RateControlled
        } else {
            QuantizerMode::Fixed
        };
    }

    fn partition_bits(&self, node: &Block, split: bool) -> f32 {
        let (row, col) = (node.y >> MI_LOG2, node.x >> MI_LOG2);
        if !node.is_inside(
            self.grids.grid.cols * MI_SIZE,
            self.grids.grid.rows * MI_SIZE,
        ) {
            return 0.0;
        }
        let probs = self.partition_probs(row, col, node.log2_size());
        let symbol = if split {
            tree::PARTITION_SPLIT
        } else {
            tree::PARTITION_NONE
        };
        tree::cost(tree::PARTITION, &probs, symbol)
    }

    fn supports_intra_leaf_cost(&self) -> bool {
        true
    }

    /// Key-frame partition planning (`planner.intra_plan = 2`): the block
    /// coded as one DC-intra leaf at its transform size, predicted from the
    /// source neighbours (the reconstruction does not exist yet), with the
    /// coefficient bits of the current probabilities (neighbour contexts
    /// taken as empty) and the reconstructed residual's error.
    fn intra_leaf_cost(&mut self, source: &Picture, block: &Block) -> Option<f32> {
        let tx = if self.lossless {
            TxSize::Tx4
        } else {
            max_tx_size(block.log2_size())
        };
        let mut bits = tree::cost(
            tree::INTRA_MODE,
            &tables::KF_Y_MODE[0][0],
            IntraMode::Dc as usize,
        ) + tree::cost(
            tree::INTRA_MODE,
            &tables::KF_UV_MODE[0],
            IntraMode::Dc as usize,
        );
        let mut sse = 0u64;
        // Speed heuristic (`vp9.leaf_cost_transform_domain`): the planner
        // only ranks leaves, and the coefficient-domain error (Parseval)
        // differs from the reconstruction's by transform rounding alone;
        // it skips the inverse transform of every planned leaf.
        let transform_domain = self.params.leaf_cost_transform_domain >= 0.5 && !self.lossless;
        for plane in PlaneIndex::ALL {
            let plane_tx = if plane == PlaneIndex::Y || self.lossless {
                tx
            } else {
                uv_tx_size(block.log2_size(), tx)
            };
            let ss = usize::from(plane != PlaneIndex::Y);
            let (x, y, n) = (block.x >> ss, block.y >> ss, block.size() >> ss);
            let dc = recon::source_dc(source.plane(plane), x, y, n);
            let model = Model {
                tx: plane_tx,
                ty: transform::TxType::DctDct,
                plane: if plane == PlaneIndex::Y {
                    PlaneType::Y
                } else {
                    PlaneType::Uv
                },
                inter: false,
            };
            let steps = self.header.steps(plane);
            let probs = &self.context.coef[plane_tx as usize][model.plane as usize][0];
            for tx_block in recon::tx_blocks(plane, block.x, block.y, block.size(), plane_tx) {
                let side = plane_tx.side();
                let area = plane_tx.area();
                let src = source.plane(plane).view();
                let scratch = &mut self.scratch;
                let residual = &mut scratch.residual[..area];
                for (dy, line) in residual.chunks_exact_mut(side).enumerate() {
                    let row = &src.row(tx_block.y + dy)[tx_block.x..tx_block.x + side];
                    for (r, &s) in line.iter_mut().zip(row) {
                        *r = i16::from(s) - i16::from(dc);
                    }
                }
                let coefficients = &mut scratch.coefficients[..area];
                transform::forward(
                    &self.dsp,
                    plane_tx,
                    model.ty,
                    self.lossless,
                    residual,
                    coefficients,
                );
                let levels = &mut scratch.levels[..area];
                let nonzero = transform::quantize(
                    &self.dsp,
                    plane_tx,
                    self.lossless,
                    coefficients,
                    steps,
                    (Rounding::Intra, 1),
                    levels,
                );
                let levels = if nonzero {
                    &*levels
                } else {
                    &transform::ZERO_LEVELS[..area]
                };
                bits += coeff::cost(levels, model, probs, 0);
                if transform_domain {
                    sse += transform::coefficient_sse(plane_tx, coefficients, levels, steps);
                    continue;
                }
                let reconstructed =
                    self.buffers
                        .residual_of(plane_tx, model.ty, self.lossless, levels, steps);
                sse += residual
                    .iter()
                    .zip(reconstructed)
                    .map(|(&r, &q)| (i64::from(r) - i64::from(q)).unsigned_abs().pow(2))
                    .sum::<u64>();
            }
        }
        Some(sse as f32 + self.lambda() * bits)
    }

    fn mv_candidates(&mut self, block: &Block, out: &mut Vec<EngineMv>) {
        let refs = self.refs_for(block);
        for mv in [refs.nearest, refs.near] {
            let engine = to_engine_mv(mv);
            if !out.contains(&engine) {
                out.push(engine);
            }
        }
    }

    fn mode_bits(&self, block: &Block, mode: &PredictionMode) -> f32 {
        match *mode {
            PredictionMode::IntraDc => {
                let (row, col) = (block.y >> MI_LOG2, block.x >> MI_LOG2);
                let context = contexts::intra_inter(&self.grids.grid, row, col);
                let intra_inter = self.context.intra_inter[context];
                let costs = &self.mode_costs;
                tree::bit_cost(false, intra_inter)
                    + costs.y[size_group(block.log2_size())][IntraMode::Dc as usize]
                    + costs.uv[IntraMode::Dc as usize][IntraMode::Dc as usize]
            }
            PredictionMode::Inter(mv) => {
                // Mode decision prices every candidate of a block after
                // `refs_for` cached its vectors and reference bits.
                let key = self.refs_key(block);
                let cached = self
                    .refs
                    .filter(|r| r.key == key)
                    .unwrap_or_else(|| BlockRefs {
                        key,
                        refs: self.find_refs(block),
                        reference_bits: self.reference_bits(block),
                    });
                cached.reference_bits
                    + cheapest_inter_mode(
                        &self.mode_costs,
                        &cached.refs,
                        to_vp9_mv(mv),
                        self.header.allow_hp,
                    )
                    .1
            }
        }
    }

    fn begin_frame(&mut self, plan: &FramePlan) {
        self.lambda_scale = plan.lambda_scale;
        let is_spatial = self.spatial.is_some();
        // On a spatial SVC enhancement (`S1`) frame with `intra_only = true`
        // (`spatial.inter_layer = 0`), VP9 cannot use `KEY_FRAME` (which
        // refreshes all 8 slots `0xff`, overwriting `S0`'s slots `0, 1`) or
        // `intra_only` (`show_frame = 0` only, VP9 §6.2). Instead it codes an
        // `INTER_FRAME` (`show_frame = 1`) whose blocks are all intra,
        // refreshing only `S1`'s slots (`0x0c`).
        let spatial_intra_only = self
            .spatial
            .as_ref()
            .is_some_and(|sp| sp.current_layer.intra_only);
        let key = plan.frame_type == FrameType::Key && !spatial_intra_only;
        let base = plan.quantizer.min(QINDEX_MAX);
        let qindex = if plan.frame_type == FrameType::Key
            && self.quantizer_mode == QuantizerMode::Fixed
            && base != 0
        {
            scaled_qindex(
                base,
                f64::from(self.params.keyframe_step_permille) / KEY_STEP_SCALE_DENOMINATOR,
            )
        } else {
            base
        } as u8;
        self.lossless = qindex == 0;
        self.coding_q_idx = qindex;
        let context_use = match &self.spatial {
            Some(sp) => sp.context_use(key),
            None => ContextUse {
                error_resilient: !key
                    && (plan.references.position.layers > 1
                        || plan.references.reference.distance != 1),
                index: 0,
                refresh: true,
            },
        };
        let past_independent = key || context_use.error_resilient;
        // A spatial layer's frame sends its segment features whenever it is
        // segmented (`Segmentation::standalone`): the features in force
        // before it are the other layer's on a receiver of both.
        let segment_state = if past_independent || is_spatial {
            SegmentFeatures::default()
        } else {
            self.grids.committed_segment_features
        };
        let (primary_slot, secondary_slot) = if spatial_intra_only {
            (0, 0)
        } else {
            let p = plan.references.reference.slot as u8;
            let s = plan.references.secondary.map_or(p, |sec| sec.slot as u8);
            (p, s)
        };
        self.header = self.build_frame_header(
            plan,
            (key, context_use, qindex),
            [primary_slot, secondary_slot, primary_slot],
            segment_state,
        );
        self.grids.contexts.reset_frame();
        self.writer = TileRecorder::default();
        self.refs = None;
        self.active_ref = ReferenceChoice::Primary;
        if past_independent {
            // `vp9_setup_past_independence` resets every context.
            self.context = FrameContext::default();
        } else if is_spatial {
            self.context = self.grids.context.clone();
        }
        self.mode_costs = ModeCosts::new(&self.context);
        self.block_qp =
            (self.block_qp_enabled && plan.block_qp).then(|| SegmentQuantizers::new(qindex));
        self.segment_state = segment_state;
        self.header.segmentation = self.plan_segmentation(segment_state);
        if is_spatial {
            self.header.segmentation = self.header.segmentation.standalone();
        }
        if past_independent {
            self.grids.segment_map =
                SegmentMap::cleared(self.grids.grid.cols, self.grids.grid.rows);
        }
        self.counts = Counts {
            high_precision: self.header.allow_hp,
            ..Counts::default()
        };
    }

    fn begin_superblock(&mut self, x: usize, _y: usize) {
        if x == 0 {
            self.grids.contexts.reset_left();
        }
    }

    fn write_partition(&mut self, node: &Block, split: bool) {
        let (row, col) = (node.y >> MI_LOG2, node.x >> MI_LOG2);
        let ctx = self.grids.contexts.partition(row, col, node.log2_size());
        let probs = self.partition_probs(row, col, node.log2_size());
        // An 8x8 node has no half: it always codes the full tree (VP9 `hbs`).
        let half = if node.log2_size() == MIN_BLOCK_LOG2 {
            0
        } else {
            1usize << (node.log2_size() - MI_LOG2 - 1)
        };
        let has_rows = row + half < self.grids.grid.rows;
        let has_cols = col + half < self.grids.grid.cols;
        let symbol = if split {
            tree::PARTITION_SPLIT
        } else {
            tree::PARTITION_NONE
        };
        self.counts.partition[ctx][symbol] += 1;
        if self.header.key {
            match (has_rows, has_cols) {
                (true, true) => tree::write(&mut self.writer, tree::PARTITION, &probs, symbol),
                // The other halves are outside the frame: SPLIT versus HORZ/VERT.
                (false, true) => self.writer.write(split, probs[1]),
                (true, false) => self.writer.write(split, probs[2]),
                (false, false) => debug_assert!(split, "a straddling node must split"),
            }
        } else {
            let base = recorder::partition_slot(ctx);
            match (has_rows, has_cols) {
                (true, true) => {
                    self.writer
                        .write_mode_tree(tree::PARTITION, &probs, base, symbol);
                }
                (false, true) => self.writer.write_mode(split, probs[1], base + 1),
                (true, false) => self.writer.write_mode(split, probs[2], base + 2),
                (false, false) => debug_assert!(split, "a straddling node must split"),
            }
        }
        if !split {
            self.grids
                .contexts
                .set_partition(row, col, node.log2_size());
        }
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
        let secondary_ref = self.active_ref == ReferenceChoice::Secondary;
        let position = self.position(block);
        let (row, col) = (position.row, position.col);
        let block_segment = self.block_segment(decision.qp_offset);
        let tx = self.largest_tx(block, decision);
        let inter = match decision.mode {
            PredictionMode::IntraDc => None,
            PredictionMode::Inter(mv) => {
                let engine_ref = reference.expect("inter blocks need a reference");
                Some(self.predict_inter_block(block, &position, mv, source, engine_ref, recon))
            }
        };
        let intra_modes = inter.is_none().then(|| match &mut self.random {
            Some(random) => random.intra_modes(&CODED_INTRA_MODES),
            None => self.choose_intra_modes(block, tx, source, recon),
        });
        let drop_residual = self
            .random
            .as_mut()
            .is_some_and(RandomSyntax::drop_residual);
        // An inter block's prediction is already saved in
        // `scratch.prediction` (predicted there, then copied into `recon`).
        let (tx, known_bits) = if inter.is_some() && !drop_residual {
            self.code_inter_residuals(block, tx, source, recon)
        } else if let (Some(modes), false) = (intra_modes, drop_residual) {
            (
                self.code_intra_residuals(block, tx, modes, source, recon),
                None,
            )
        } else {
            let dropped = if drop_residual {
                ResidualDrop::All
            } else {
                ResidualDrop::None
            };
            self.code_residuals(
                block,
                (tx, TrialPlanes::All),
                intra_modes,
                dropped,
                source,
                recon,
            );
            (tx, None)
        };
        if inter.is_some() && self.params.residual_rd >= 0.5 {
            self.drop_unprofitable_residual(block, source, recon, known_bits);
        }
        let skip = self.scratch.coded.iter().all(|c| !c.eob_nonzero);
        let (segment_id, segment_predicted) = self.segment_of(
            row,
            col,
            block.log2_size(),
            block_segment.unwrap_or(u8::from(inter.is_some() && skip)),
        );
        let info = ModeInfo {
            size_log2: block.log2_size(),
            inter: inter.map(|(mode, mv, _, _)| (mode, mv)),
            secondary_ref: inter.is_some() && secondary_ref,
            y_mode: intra_modes.map_or(IntraMode::Dc, |(y, _)| y),
            uv_mode: intra_modes.map_or(IntraMode::Dc, |(_, uv)| uv),
            skip,
            // A skipped inter block codes no size: the decoder takes the
            // largest (its residual is empty either way).
            tx_size: if skip && inter.is_some() {
                max_tx_size(block.log2_size())
            } else {
                tx
            },
            segment_id,
            segment_predicted,
            filter: inter.map_or(nc_dsp::Vp9Filter::Regular, |(_, _, _, filter)| filter),
        };
        self.write_mode_info(row, col, &info, inter.map(|(_, _, refs, _)| refs));
        if skip {
            self.grids
                .contexts
                .clear_block(block.x, block.y, block.size());
        } else {
            self.write_tokens(info.is_inter());
        }
        self.grids.grid.fill(row, col, info);
        self.refs = None;
        self.active_ref = ReferenceChoice::Primary;
        self.coding_q_idx = self.header.base_q_idx;
    }

    fn filter_frame(&mut self, source: &Picture, recon: &mut Picture) {
        self.loop_filter_frame(recon);
        if self.measures_chroma_error_ratio() && !self.lossless {
            self.grids.frame_chroma_error_ratio =
                nc_encoder::plane_error_ratio(&self.dsp, source, recon);
        }
    }

    fn loop_filter_frame(&mut self, recon: &mut Picture) {
        loopfilter::apply(
            &self.dsp,
            recon,
            &self.grids.grid,
            (self.header.filter_level, self.header.sharpness),
            self.header
                .segmentation
                .active_filter_levels(self.header.filter_level),
        );
    }

    fn end_frame(&mut self) -> Vec<u8> {
        self.finish_segment_quantizers();
        // Forward updates of this frame's coefficient models, chosen on
        // its counts; the tile is written with them.
        let sizes = if self.lossless {
            1
        } else {
            TxSize::Tx32 as usize + 1
        };
        let is_spatial = self.spatial.is_some();
        // The context the decoder starts the frame from (`begin_frame`).
        let base_context = &self.context;
        let updates = if self.params.coef_updates >= 0.5 {
            CoefUpdates::choose(base_context, &self.counts, sizes)
        } else {
            CoefUpdates::none(base_context)
        };
        let mode_updates = is_spatial.then(|| {
            let old = recorder::mode_slot_probs(base_context);
            let new = crate::probs::choose_mode_updates(base_context, &self.writer.mode_counts());
            (old, new)
        });
        let compressed = self.header.write_compressed_with_modes(
            &base_context.coef,
            &updates,
            mode_updates.as_ref().map(|(old, new)| (old, new)),
        );
        if let Some(sp) = self.spatial.as_mut() {
            sp.coded_context.clone_from(&self.context);
            sp.coded_context.coef = updates.coef;
            if let Some((_, new)) = &mode_updates {
                recorder::set_mode_slot_probs(&mut sp.coded_context, new);
            }
        }
        if self.own_segment_tree_probs() {
            self.header.segmentation.tree_probs = self.writer.segment_tree_probs();
        }
        let tile = std::mem::take(&mut self.writer).finish_with_modes(
            &recorder::slot_probs(&updates.coef),
            self.header.segmentation.tree_probs,
            mode_updates.as_ref().map(|(_, new)| new),
        );
        let mut packet = self.header.write(compressed.len());
        packet.extend(compressed);
        packet.extend(tile);
        packet
    }

    fn commit_frame(&mut self) {
        let key = self.header.key;
        if let Some(ratio) = self.grids.frame_chroma_error_ratio.take() {
            self.grids.chroma_error_ratio = Some(ratio);
        }
        if self.header.segmentation.enabled {
            self.grids.segment_map.update(&self.grids.grid);
        }
        // Backward adaptation starts from the context before the forward
        // updates; error-resilient frames do not adapt and leave the default.
        if let Some(sp) = self.spatial.as_mut() {
            if key || self.header.error_resilient {
                // `vp9_setup_past_independence` reset every context.
                self.grids.context = FrameContext::default();
                sp.parked.context = FrameContext::default();
            }
            if self.header.refresh_context && !self.header.error_resilient {
                self.grids.context = self.context.adapted_from_coded(
                    &sp.coded_context,
                    &self.counts,
                    !key,
                    !key && sp.previous_key,
                    !self.lossless,
                );
            }
            sp.committed(&self.header);
        } else if !self.header.error_resilient {
            self.context = self.context.adapted(
                &self.counts,
                !key,
                !key && self.grids.last_key,
                !self.lossless,
            );
        }
        self.grids.committed_segment_features = self.header.segmentation.features;
        self.grids.last_key = key;
        self.grids.previous = Some(self.grids.grid.clone());
    }
}

impl Vp9Backend {
    /// Normative reference picture (when held in `self.store` for an `S0` slot
    /// referenced by `S1`) and `(RefFrameWidth, RefFrameHeight)` for the active
    /// reference (`LAST_FRAME` or `GOLDEN_FRAME`).
    fn active_reference(&self) -> (Option<Rc<Picture>>, (usize, usize)) {
        let own_size = (self.grids.width, self.grids.height);
        if self
            .spatial
            .as_ref()
            .is_some_and(|sp| sp.active_spatial_id == 1)
        {
            let slot = if self.active_ref == ReferenceChoice::Secondary {
                usize::from(self.header.ref_slots[1])
            } else {
                usize::from(self.header.ref_slots[0])
            };
            if let Some(held) = self.store.held(slot) {
                return (Some(Rc::clone(&held.picture)), held.size());
            }
        }
        (None, own_size)
    }

    /// Quantizer steps of `plane` for the block being coded.
    fn block_steps(&self, plane: PlaneIndex) -> transform::Steps {
        self.header.steps_at(self.coding_q_idx, plane)
    }

    fn position(&self, block: &Block) -> BlockPosition {
        BlockPosition {
            row: block.y >> MI_LOG2,
            col: block.x >> MI_LOG2,
            size: block.size() >> MI_LOG2,
            rows: self.grids.grid.rows,
            cols: self.grids.grid.cols,
            width: self.grids.width,
            height: self.grids.height,
        }
    }

    /// The block's reference vectors, cached until it is coded.
    fn refs_for(&mut self, block: &Block) -> MvRefs {
        let key = self.refs_key(block);
        if let Some(cached) = self.refs.filter(|r| r.key == key) {
            return cached.refs;
        }
        let cached = BlockRefs {
            key,
            refs: self.find_refs(block),
            reference_bits: self.reference_bits(block),
        };
        self.refs = Some(cached);
        cached.refs
    }

    /// The [`BlockRefs`] key of `block` under the active reference.
    fn refs_key(&self, block: &Block) -> (usize, usize, u8, bool) {
        (
            block.y >> MI_LOG2,
            block.x >> MI_LOG2,
            block.log2_size(),
            self.active_ref == ReferenceChoice::Secondary,
        )
    }

    /// The reference vectors of `block` under the active reference
    /// (`mvref::find`).
    fn find_refs(&self, block: &Block) -> MvRefs {
        mvref::find(
            &self.grids.grid,
            self.previous_vectors(),
            &self.position(block),
            block.log2_size(),
            self.header.allow_hp,
            self.active_ref == ReferenceChoice::Secondary,
        )
    }

    /// Bits of coding `block` as inter with the active reference: the
    /// intra/inter flag and the single-reference bits (VP9 §6.4.17-18).
    fn reference_bits(&self, block: &Block) -> f32 {
        let (row, col) = (block.y >> MI_LOG2, block.x >> MI_LOG2);
        let (fc, grid) = (&self.context, &self.grids.grid);
        let intra_inter = fc.intra_inter[contexts::intra_inter(grid, row, col)];
        let p1 = fc.single_ref[contexts::single_ref_p1(grid, row, col)][0];
        let reference = if self.active_ref == ReferenceChoice::Secondary {
            let p2 = fc.single_ref[contexts::single_ref_p2(grid, row, col)][1];
            tree::bit_cost(true, p1) + tree::bit_cost(false, p2)
        } else {
            tree::bit_cost(false, p1)
        };
        tree::bit_cost(true, intra_inter) + reference
    }

    /// The previous frame's mode info when this frame may read its
    /// vectors (`use_prev_frame_mvs`: an inter frame after an inter frame).
    fn previous_vectors(&self) -> Option<&ModeInfoGrid> {
        self.grids
            .previous
            .as_ref()
            // Spatial layers never use them (`BackendSpatial::context_use`).
            .filter(|_| {
                self.spatial.is_none()
                    && !self.header.key
                    && !self.header.error_resilient
                    && !self.grids.last_key
            })
    }

    fn partition_probs(&self, row: usize, col: usize, log2: u8) -> [u8; 3] {
        let ctx = self.grids.contexts.partition(row, col, log2);
        if self.header.key {
            tables::KF_PARTITION[ctx]
        } else {
            self.context.partition[ctx]
        }
    }

    /// Write a block's segment id (`intra_segment_id` / `inter_segment_id`,
    /// VP9 §6.4): predicted from the last map or explicit.
    fn write_segment_id(&mut self, row: usize, col: usize, info: &ModeInfo) {
        let own_tree_probs = self.own_segment_tree_probs();
        let segmentation = &self.header.segmentation;
        if !segmentation.enabled {
            return;
        }
        if let Some(pred_probs) = segmentation.temporal {
            let ctx = segmentation::prediction_context(&self.grids.grid, row, col);
            self.writer.write(info.segment_predicted, pred_probs[ctx]);
        }
        if info.segment_predicted {
            return;
        }
        if own_tree_probs {
            self.writer.write_segment_id(info.segment_id);
        } else {
            tree::write(
                &mut self.writer,
                segmentation::SEGMENT_TREE,
                &segmentation.tree_probs,
                usize::from(info.segment_id),
            );
        }
    }

    /// Write a block's mode info (VP9 §6.4.3 key frames, §6.4.11 inter
    /// frames): skip, then intra modes or the inter reference, mode and
    /// vector difference.
    #[allow(clippy::too_many_lines)]
    fn write_mode_info(&mut self, row: usize, col: usize, info: &ModeInfo, refs: Option<MvRefs>) {
        self.write_segment_id(row, col, info);
        let fc = &self.context;
        let counts = &mut self.counts;
        let skip_ctx = contexts::skip(&self.grids.grid, row, col);
        self.writer
            .write_mode(info.skip, fc.skip[skip_ctx], recorder::skip_slot(skip_ctx));
        counts.skip[skip_ctx][usize::from(info.skip)] += 1;
        if self.header.key {
            write_tx_size(
                &mut self.writer,
                fc,
                counts,
                &self.grids.grid,
                row,
                col,
                info,
                self.lossless,
            );
            let (above, left) = contexts::key_frame_modes(&self.grids.grid, row, col);
            let y_probs = &tables::KF_Y_MODE[above as usize][left as usize];
            let uv_probs = &tables::KF_UV_MODE[info.y_mode as usize];
            tree::write(
                &mut self.writer,
                tree::INTRA_MODE,
                y_probs,
                info.y_mode as usize,
            );
            tree::write(
                &mut self.writer,
                tree::INTRA_MODE,
                uv_probs,
                info.uv_mode as usize,
            );
            return;
        }
        let intra_inter = contexts::intra_inter(&self.grids.grid, row, col);
        self.writer.write_mode(
            info.is_inter(),
            fc.intra_inter[intra_inter],
            recorder::intra_inter_slot(intra_inter),
        );
        counts.intra_inter[intra_inter][usize::from(info.is_inter())] += 1;
        write_tx_size(
            &mut self.writer,
            fc,
            counts,
            &self.grids.grid,
            row,
            col,
            info,
            self.lossless,
        );
        if let (Some((mode, mv)), Some(refs)) = (info.inter, refs) {
            let p1 = contexts::single_ref_p1(&self.grids.grid, row, col);
            self.writer.write_mode(
                info.secondary_ref,
                fc.single_ref[p1][0],
                recorder::single_ref_slot(p1, 0),
            );
            counts.single_ref[p1][0][usize::from(info.secondary_ref)] += 1;
            if info.secondary_ref {
                let p2 = contexts::single_ref_p2(&self.grids.grid, row, col);
                self.writer.write_mode(
                    false,
                    fc.single_ref[p2][1],
                    recorder::single_ref_slot(p2, 1),
                ); // GOLDEN_FRAME
                counts.single_ref[p2][1][0] += 1;
            }
            let mode_probs = &fc.inter_mode[refs.mode_context];
            self.writer.write_mode_tree(
                tree::INTER_MODE,
                mode_probs,
                recorder::inter_mode_slot(refs.mode_context),
                mode as usize,
            );
            counts.inter_mode[refs.mode_context][mode as usize] += 1;
            if self.header.interp_filter.is_none() {
                let ctx = contexts::switchable_interp(&self.grids.grid, row, col);
                let symbol = contexts::filter_symbol(info.filter);
                self.writer.write_mode_tree(
                    tree::SWITCHABLE_INTERP,
                    &fc.switchable_interp[ctx],
                    recorder::switchable_interp_slot(ctx),
                    symbol,
                );
                counts.switchable_interp[ctx][symbol] += 1;
            }
            if mode == InterMode::New {
                let (dr, dc) = (mv.row - refs.nearest.row, mv.col - refs.nearest.col);
                let usehp = self.header.allow_hp && motion::uses_high_precision(refs.nearest);
                motion::encode(&mut self.writer, dr, dc, fc, counts, usehp)
                    .expect("engine vectors are within the coded range");
            }
            return;
        }
        let group = size_group(info.size_log2);
        counts.y_mode[group][info.y_mode as usize] += 1;
        counts.uv_mode[info.y_mode as usize][info.uv_mode as usize] += 1;
        self.writer.write_mode_tree(
            tree::INTRA_MODE,
            &fc.y_mode[group],
            recorder::y_mode_slot(group),
            info.y_mode as usize,
        );
        tree::write(
            &mut self.writer,
            tree::INTRA_MODE,
            &fc.uv_mode[info.y_mode as usize],
            info.uv_mode as usize,
        );
    }

    /// With 1/8 precision allowed near the block's reference vector, the
    /// engine's vector (quarter-sample) or one of its 1/8 neighbours,
    /// whichever predicts luma best (SSE plus lambda-weighted mode and
    /// vector bits; regular filter). Off by default
    /// (`vp9.high_precision_qindex` 0): 30 frames with it below qindex 200
    /// gave crowd q120 +0.1 dB at -0.75% bytes, but `bodycam_airport` q160
    /// -0.14 dB at +4.9%, for +1-4% instructions.
    fn refine_high_precision(
        &mut self,
        source: &Picture,
        (reference, ref_size): (&Picture, (usize, usize)),
        position: &BlockPosition,
        refs: &MvRefs,
        mv: Mv,
    ) -> Mv {
        if !self.header.allow_hp || !motion::uses_high_precision(refs.nearest) {
            return mv;
        }
        if let Some(random) = &mut self.random {
            return random.high_precision_vector(mv);
        }
        let lambda = self.lambda();
        let side = position.size * MI_SIZE;
        let (x, y) = (
            (position.col * MI_SIZE) as isize,
            (position.row * MI_SIZE) as isize,
        );
        let src = source
            .plane(PlaneIndex::Y)
            .region(x, y, side, side)
            .expect("block inside the source");
        let size = nc_dsp::BlockSize::new(side, side).expect("square VP9 block");
        // Speed heuristic: a vector a cheaper mode already signals stays
        // (an eighth step would turn it into NEWMV), and the search walks the
        // rows, then the columns of the best row: five predictions, not nine.
        if [Mv::ZERO, refs.nearest, refs.near].contains(&mv) {
            return mv;
        }
        let cost_of = |backend: &mut Self, candidate: Mv| {
            let bits = cheapest_inter_mode(&backend.mode_costs, refs, candidate, true).1;
            if !bits.is_finite() {
                return f32::INFINITY;
            }
            let prediction = &mut backend.scratch.filter_trial[..side * side];
            inter_pred::predict_with_ref_size(
                &backend.dsp,
                reference,
                ref_size,
                PlaneIndex::Y,
                position,
                candidate,
                nc_dsp::Vp9Filter::Regular,
                &mut backend.inter.window,
                prediction,
            );
            let pred = PlaneView::new(prediction, side, side, side).expect("prediction block");
            backend.dsp.variance(size, &src, &pred).sse as f32 + lambda * bits
        };
        let mut best = (cost_of(self, mv), mv);
        for (dr, dc) in [(-1, 0), (1, 0)] {
            let candidate = Mv {
                row: mv.row + dr,
                col: mv.col + dc,
            };
            let cost = cost_of(self, candidate);
            if cost < best.0 {
                best = (cost, candidate);
            }
        }
        let row_best = best.1;
        for dc in [-1, 1] {
            let candidate = Mv {
                row: row_best.row,
                col: row_best.col + dc,
            };
            let cost = cost_of(self, candidate);
            if cost < best.0 {
                best = (cost, candidate);
            }
        }
        best.1
    }

    /// The block's interpolation filter: the frame's, or with switchable
    /// filters the one whose luma prediction costs least (SSE plus
    /// lambda-weighted filter bits). A whole-sample vector predicts the same
    /// with every filter, so it takes the cheapest symbol.
    /// The largest transform the block's residual search starts from.
    fn largest_tx(&self, block: &Block, decision: &BlockDecision) -> TxSize {
        if self.lossless {
            TxSize::Tx4
        } else if matches!(decision.mode, PredictionMode::Inter(_)) {
            // Speed lever (`vp9.inter_max_tx_log2`): smaller transforms for
            // large inter blocks. Measured in docs/handoff/vp9-faster.md.
            let cap = self.params.inter_max_tx_log2.round() as u8;
            max_tx_size(block.log2_size().min(cap))
        } else {
            max_tx_size(block.log2_size())
        }
    }

    fn choose_filter(
        &mut self,
        source: &Picture,
        (reference, ref_size): (&Picture, (usize, usize)),
        position: &BlockPosition,
        mv: Mv,
    ) -> (nc_dsp::Vp9Filter, bool) {
        if let Some(filter) = self.header.interp_filter {
            return (filter, false);
        }
        if let Some(random) = &mut self.random {
            return (random.block_filter(), false);
        }
        let ctx = contexts::switchable_interp(&self.grids.grid, position.row, position.col);
        let bits = self.mode_costs.switchable[ctx];
        let lambda = self.lambda();
        let unscaled = ref_size == (position.width, position.height);
        let whole_sample =
            unscaled && mv.row % MV_UNITS_PER_SAMPLE == 0 && mv.col % MV_UNITS_PER_SAMPLE == 0;
        if whole_sample {
            let cheapest = (0..SWITCHABLE_FILTERS)
                .min_by(|&a, &b| bits[a].total_cmp(&bits[b]))
                .expect("three filters");
            return (contexts::symbol_filter(cheapest), false);
        }
        let side = position.size * MI_SIZE;
        let (x, y) = (
            (position.col * MI_SIZE) as isize,
            (position.row * MI_SIZE) as isize,
        );
        let src = source
            .plane(PlaneIndex::Y)
            .region(x, y, side, side)
            .expect("block inside the source");
        let size = nc_dsp::BlockSize::new(side, side).expect("square VP9 block");
        // The best trial's luma prediction stays in `filter_best`, so the
        // block's prediction does not filter it again.
        let mut best = (f32::INFINITY, nc_dsp::Vp9Filter::Regular);
        // Speed heuristic (`vp9.filter_search_rt`, libvpx `vp9_pickmode.c`
        // real-time filter search, which tries only EIGHTTAP and
        // EIGHTTAP_SMOOTH): the sharp filter is not tried. Measured in
        // docs/handoff/vp9-faster.md.
        let skip_sharp = self.params.filter_search_rt >= 0.5;
        for (symbol, &symbol_bits) in bits.iter().enumerate() {
            let filter = contexts::symbol_filter(symbol);
            if skip_sharp && filter == nc_dsp::Vp9Filter::Sharp {
                continue;
            }
            let prediction = &mut self.scratch.filter_trial[..side * side];
            inter_pred::predict_with_ref_size(
                &self.dsp,
                reference,
                ref_size,
                PlaneIndex::Y,
                position,
                mv,
                filter,
                &mut self.inter.window,
                prediction,
            );
            let pred = PlaneView::new(prediction, side, side, side).expect("prediction block");
            let cost = self.dsp.variance(size, &src, &pred).sse as f32 + lambda * symbol_bits;
            if cost < best.0 {
                best = (cost, filter);
                std::mem::swap(
                    &mut self.scratch.filter_trial,
                    &mut self.scratch.filter_best,
                );
            }
        }
        (best.1, true)
    }

    /// Chroma rounding for a block quantized with `block`
    /// (`vp9.chroma_rounding`). A smaller chroma deadzone keeps the colour
    /// detail VMAF's chroma feature measures: VP9 cannot quantize chroma
    /// much finer than luma (`delta_q` stops at -15), unlike HEVC's chroma
    /// QP table. `crowd_pp22_051` q219: nearest rounding +0.5% bytes, VMAF
    /// 52.6 -> 56.0, chroma PSNR +0.5 dB.
    fn chroma_rounding(&self, block: Rounding) -> Rounding {
        match self.params.chroma_rounding.round() as u8 {
            0 => block,
            1 => Rounding::Intra,
            _ => Rounding::Nearest,
        }
    }

    /// Luma and chroma modes of an intra block: of the ten modes, the one
    /// whose prediction costs least plus weighted mode bits (chroma
    /// conditioned on luma), measured as `vp9.intra_mode_estimate` says:
    /// on the plane's first transform block (SSE) or on the whole block
    /// (SATD).
    fn choose_intra_modes(
        &self,
        block: &Block,
        tx: TxSize,
        source: &Picture,
        recon: &Picture,
    ) -> (IntraMode, IntraMode) {
        // Speed lever (`vp9.inter_intra_dc_only`): intra blocks of inter
        // frames take DC without a search. Measured in
        // docs/handoff/vp9-faster.md.
        if !self.header.key && self.params.inter_intra_dc_only >= 0.5 {
            return (IntraMode::Dc, IntraMode::Dc);
        }
        let (row, col) = (block.y >> MI_LOG2, block.x >> MI_LOG2);
        let whole_block = self.params.intra_mode_estimate >= 0.5;
        let (metric, bit_weight) = if whole_block {
            (recon::IntraMetric::Satd, self.lambda().sqrt())
        } else {
            (recon::IntraMetric::Sse, self.lambda())
        };
        let search = IntraSearch::of(self.params.intra_mode_search);
        let pick = |plane: PlaneIndex, plane_tx: TxSize, costs: &[f32; INTRA_MODES]| {
            // Speed heuristic (`vp9.intra_mode_search` 2, libvpx real-time
            // speeds' intra masks): chroma takes DC unsearched and luma
            // chooses among DC, V and H. Measured in
            // docs/handoff/vp9-faster.md.
            if search == IntraSearch::RealTime && plane != PlaneIndex::Y {
                return IntraMode::Dc;
            }
            let estimate_tx = if whole_block {
                let side_log2 = block.log2_size() - u8::from(plane != PlaneIndex::Y);
                TxSize::from_log2(side_log2.clamp(TxSize::Tx4.log2(), TxSize::Tx32.log2()))
            } else {
                plane_tx
            };
            let first = recon::tx_blocks(plane, block.x, block.y, block.size(), estimate_tx)
                .next()
                .expect("a block has transform blocks");
            let cost = |mode: IntraMode| {
                let distortion = recon::intra_prediction_cost(
                    &self.dsp,
                    recon,
                    source,
                    &first,
                    (mode, metric),
                    row > 0,
                    col > 0,
                );
                (distortion as f32 + bit_weight * costs[mode as usize], mode)
            };
            let best = |modes: &mut dyn Iterator<Item = IntraMode>| {
                modes
                    .map(cost)
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .expect("at least one mode")
            };
            match search {
                IntraSearch::All => return best(&mut CODED_INTRA_MODES.into_iter()).1,
                IntraSearch::RealTime => return best(&mut REAL_TIME_LUMA_MODES.into_iter()).1,
                IntraSearch::Pruned => {}
            }
            // Speed heuristic (`vp9.intra_mode_search`): the four
            // non-directional modes, then the two directions beside the
            // winner's. Measured in the commit message.
            let base = best(&mut BASE_INTRA_MODES.into_iter());
            let refined = best(&mut directions_beside(base.1).into_iter());
            if refined.0 < base.0 {
                refined.1
            } else {
                base.1
            }
        };
        let costs = &self.mode_costs;
        let (y_costs, uv_costs) = if self.header.key {
            let (above, left) = contexts::key_frame_modes(&self.grids.grid, row, col);
            (&costs.kf_y[above as usize][left as usize], &costs.kf_uv)
        } else {
            (&costs.y[size_group(block.log2_size())], &costs.uv)
        };
        let uv_tx = if self.lossless {
            tx
        } else {
            uv_tx_size(block.log2_size(), tx)
        };
        let y = pick(PlaneIndex::Y, tx, y_costs);
        let uv = pick(PlaneIndex::U, uv_tx, &uv_costs[y as usize]);
        (y, uv)
    }

    /// Keep the coded state of the current transform-size trial (its
    /// reconstruction, levels and transform blocks) as the best so far.
    /// Only the `planes` the next trial recodes are kept: the others stay
    /// as they are in `recon` and the level buffer.
    fn save_best_trial(&mut self, block: &Block, recon: &Picture, planes: TrialPlanes) {
        let scratch = &mut self.scratch;
        copy_block_out(block, recon, &mut scratch.best_recon, planes);
        let used = trial_levels(block, planes);
        scratch.best_levels[..used].copy_from_slice(&scratch.levels[..used]);
        scratch.best_coded.clone_from(&scratch.coded);
    }

    /// Return to the trial kept by [`Self::save_best_trial`] without coding
    /// it again.
    /// `planes` are the ones the save kept (the failed trial's).
    fn restore_best_trial(&mut self, block: &Block, recon: &mut Picture, planes: TrialPlanes) {
        let scratch = &mut self.scratch;
        copy_block_in(block, &scratch.best_recon, recon, planes);
        let used = trial_levels(block, planes);
        scratch.levels[..used].copy_from_slice(&scratch.best_levels[..used]);
        scratch.coded.clone_from(&scratch.best_coded);
    }

    /// Residual RD for an inter block: when the prediction alone costs no
    /// more than the coded residual (SSE + lambda * bits, with token bits at
    /// the current contexts and the skip flag), restore the prediction and
    /// code the block as skipped. `known_bits` are the coded blocks' token
    /// bits when the transform-size search already priced them.
    fn drop_unprofitable_residual(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
        known_bits: Option<f32>,
    ) {
        if self.scratch.coded.iter().all(|c| !c.eob_nonzero) {
            return;
        }
        let (row, col) = (block.y >> MI_LOG2, block.x >> MI_LOG2);
        let skip_prob = self.context.skip[contexts::skip(&self.grids.grid, row, col)];
        let lambda = self.lambda();
        let (coded_sse, skipped_sse) = self.block_sse(block, source, recon);
        let token_bits = known_bits.unwrap_or_else(|| self.token_bits(true));
        let coded = coded_sse + lambda * (token_bits + tree::bit_cost(false, skip_prob));
        let skipped = skipped_sse + lambda * tree::bit_cost(true, skip_prob);
        if skipped <= coded {
            self.restore_prediction(block, recon, TrialPlanes::All);
            for coded in &mut self.scratch.coded {
                coded.eob_nonzero = false;
            }
        }
    }

    /// Estimated token bits of the coded transform blocks (current
    /// probabilities and contexts; inter models).
    fn token_bits(&self, inter: bool) -> f32 {
        let per_level = self.params.residual_bits_per_level;
        if inter && per_level > 0.0 {
            return per_level * self.level_magnitude_sum() as f32;
        }
        self.scratch
            .coded
            .iter()
            .filter(|c| c.eob_nonzero)
            .map(|coded| {
                let model = coded.model(inter);
                let probs =
                    &self.context.coef[coded.tx as usize][model.plane as usize][usize::from(inter)];
                let (ux, uy) = coded.units();
                let ctx = self
                    .grids
                    .contexts
                    .coefficient(coded.plane, ux, uy, coded.tx);
                coeff::cost(&self.scratch.levels[coded.levels()], model, probs, ctx)
            })
            .sum()
    }

    /// Speed heuristic (`vp9.residual_bits_per_level`, libvpx
    /// `vp9_pickmode.c` `block_yrd`, BSD): the sum of the coded levels'
    /// magnitudes, which libvpx's real-time mode decision prices at 4 bits
    /// a unit instead of walking the token trees.
    fn level_magnitude_sum(&self) -> u32 {
        self.scratch
            .coded
            .iter()
            .filter(|c| c.eob_nonzero)
            .map(|coded| {
                self.scratch.levels[coded.levels()]
                    .iter()
                    .map(|level| level.unsigned_abs())
                    .sum::<u32>()
            })
            .sum()
    }

    /// SSE of the block's reconstruction against the source, all planes
    /// (shared DSP kernels).
    fn reconstruction_sse(&self, block: &Block, source: &Picture, recon: &Picture) -> u64 {
        PlaneIndex::ALL
            .into_iter()
            .map(|plane| {
                let ss = usize::from(plane != PlaneIndex::Y);
                let (x, y, n) = (block.x >> ss, block.y >> ss, block.size() >> ss);
                let size = nc_dsp::BlockSize::new(n, n).expect("square VP9 plane block");
                let (src, rec) = (source.plane(plane), recon.plane(plane));
                let (xi, yi) = (x as isize, y as isize);
                let src = src.region(xi, yi, n, n).expect("block inside the picture");
                let rec = rec.region(xi, yi, n, n).expect("block inside the picture");
                u64::from(self.dsp.variance(size, &src, &rec).sse)
            })
            .sum()
    }

    /// The frame's `delta_q_uv` without the engine policy:
    /// `vp9.chroma_delta_q`, plus the coarse-quantizer extra ramped in by
    /// `qindex`, moved by the previous frame's luma/chroma error gap
    /// (AV2's controls; `nc_encoder::chroma_offset`).
    fn own_chroma_delta(&self, qindex: u8) -> i8 {
        let p = &self.params;
        let delta = p.chroma_delta_q
            + nc_encoder::coarse_chroma_extra(
                f32::from(qindex),
                p.chroma_delta_q_coarse,
                p.chroma_delta_q_coarse_start,
                p.chroma_delta_q_coarse_end,
            )
            + nc_encoder::error_gap_offset(
                self.grids.chroma_error_ratio,
                p.chroma_delta_gap_slope,
                p.chroma_delta_gap_pivot,
            );
        (delta.round() as i32).clamp(-CHROMA_DELTA_LIMIT, CHROMA_DELTA_LIMIT) as i8
    }

    /// Chroma's squared-error weight in the residual drop without the
    /// engine policy: under `vp9.chroma_weight_adapt`, the previous
    /// frame's `MSE_Y / MSE_C` times the gain (PSNR-YUV's marginal value of
    /// chroma error is about 4/6 of that ratio), else `vp9.chroma_sse_weight`.
    fn own_chroma_weight(&self) -> f32 {
        let gain = self.params.chroma_weight_adapt;
        match self.grids.chroma_error_ratio.filter(|_| gain > 0.0) {
            Some(ratio) => {
                (f64::from(gain) * ratio).clamp(ADAPTIVE_WEIGHT_MIN, ADAPTIVE_WEIGHT_MAX) as f32
            }
            None => self.params.chroma_sse_weight,
        }
    }

    /// Whether a chroma control reads the luma/chroma error ratio.
    fn measures_chroma_error_ratio(&self) -> bool {
        self.chroma_policy.is_none()
            && (self.params.chroma_delta_gap_slope > 0.0 || self.params.chroma_weight_adapt > 0.0)
    }

    /// SSE of the block's reconstruction and of its saved prediction
    /// against the source, all planes (shared DSP kernels).
    /// Chroma is weighted by `vp9.chroma_sse_weight`, or by the chroma
    /// policy's squared step ratio.
    fn block_sse(&self, block: &Block, source: &Picture, recon: &Picture) -> (f32, f32) {
        let (mut coded_total, mut predicted_total) = (0f32, 0f32);
        let mut offset = 0;
        for plane in PlaneIndex::ALL {
            let ss = usize::from(plane != PlaneIndex::Y);
            let (x, y, n) = (block.x >> ss, block.y >> ss, block.size() >> ss);
            let size = nc_dsp::BlockSize::new(n, n).expect("square VP9 plane block");
            let (xi, yi) = (x as isize, y as isize);
            let src = source.plane(plane).region(xi, yi, n, n);
            let rec = recon.plane(plane).region(xi, yi, n, n);
            let (src, rec) = (
                src.expect("block inside the source"),
                rec.expect("block inside the picture"),
            );
            let packed = &self.scratch.prediction[offset..offset + n * n];
            let predicted = PlaneView::new(packed, n, n, n).expect("packed prediction");
            offset += n * n;
            let weight = match (plane, self.chroma_policy) {
                (PlaneIndex::Y, _) => 1.0,
                (_, Some(_)) => {
                    let (luma, chroma) = (
                        self.header.steps_at(self.header.base_q_idx, PlaneIndex::Y),
                        self.header.steps_at(self.header.base_q_idx, plane),
                    );
                    nc_encoder::chroma_policy::distortion_weight(
                        f64::from(luma.ac),
                        f64::from(chroma.ac),
                    ) as f32
                }
                (_, None) => self.own_chroma_weight(),
            };
            coded_total += weight * self.dsp.variance(size, &src, &rec).sse as f32;
            predicted_total += weight * self.dsp.variance(size, &src, &predicted).sse as f32;
        }
        (coded_total, predicted_total)
    }

    /// Put the saved inter prediction of `planes` back into `recon`.
    fn restore_prediction(&self, block: &Block, recon: &mut Picture, planes: TrialPlanes) {
        copy_block_in(block, &self.scratch.prediction, recon, planes);
    }

    /// Speed heuristic (`vp9.pre_tx_skip_energy`, AV1's light-PD1 energy
    /// test): the inter prediction's luma error is so small for the block's
    /// quantizer that its residual is not worth transforming. Not for a
    /// refined block (it exists to restore small detail).
    fn negligible_luma_prediction_error(&self, block: &Block, source: &Picture) -> bool {
        let energy = self.params.pre_tx_skip_energy;
        if self.lossless || energy <= 0.0 || self.refining_block() {
            return false;
        }
        let n = block.size();
        let size = nc_dsp::BlockSize::new(n, n).expect("square VP9 block");
        let pred =
            PlaneView::new(&self.scratch.prediction[..n * n], n, n, n).expect("packed prediction");
        let src = source
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, n, n)
            .expect("block inside the source");
        let sse = self.dsp.variance(size, &src, &pred).sse as f32;
        ENERGY_SCALE * sse < energy * (n * n) as f32 * f32::from(self.coding_q_idx)
    }

    /// Speed heuristic (`vp9.tx_search_min_levels`): the coded block has
    /// at most that many nonzero levels, so smaller transforms (which pay
    /// more per coefficient) are not tried.
    fn sparse_residual(&self) -> bool {
        let limit = self.params.tx_search_min_levels.round() as usize;
        if limit == 0 {
            return false;
        }
        let scratch = &self.scratch;
        scratch
            .coded
            .iter()
            .filter(|c| c.eob_nonzero)
            .map(|c| {
                scratch.levels[c.levels()]
                    .iter()
                    .filter(|&&l| l != 0)
                    .count()
            })
            .sum::<usize>()
            <= limit
    }

    /// Speed heuristic (`vp9.inter_tx_split_spread`, HEVC's
    /// `inter_tu_split_spread`): the luma prediction error is spread evenly
    /// over the block's quadrants, so smaller transforms, which pay off
    /// where the residual is local, are not tried.
    fn evenly_spread_residual(&self, block: &Block, source: &Picture) -> bool {
        let spread = self.params.inter_tx_split_spread;
        if spread <= 1.0 {
            return false;
        }
        let (n, half) = (block.size(), block.size() / 2);
        let size = nc_dsp::BlockSize::new(half, half).expect("square VP9 quadrant");
        let quadrants = [(0, 0), (half, 0), (0, half), (half, half)].map(|(dx, dy)| {
            let packed = &self.scratch.prediction[dy * n + dx..];
            let pred = PlaneView::new(packed, n, half, half).expect("packed prediction");
            let (x, y) = ((block.x + dx) as isize, (block.y + dy) as isize);
            let src = source
                .plane(PlaneIndex::Y)
                .region(x, y, half, half)
                .expect("block inside the source");
            u64::from(self.dsp.variance(size, &src, &pred).sse)
        });
        let total: u64 = quadrants.iter().sum();
        let largest = quadrants.into_iter().max().unwrap_or(0);
        (largest as f32) * QUADRANTS < spread * total as f32
    }

    /// This frame's segmentation. With `vp9.skip_filter_off`, inter
    /// frames put skipped inter blocks in segment 1, whose loop-filter
    /// level is 0: the samples they copy were filtered in the frame they
    /// come from, and filtering them again every frame smooths texture into
    /// bands. Off by default: `crowd_pp22_051` q226 gains VMAF 54.3 -> 56.5
    /// (cambi 0.90 -> 0.58), but the explicit segment map (redundant with
    /// the skip flag) costs +12% bytes there and +20% on `bodycam_airport`;
    /// without the map the same filtering change costs only +2%, and
    /// temporal map prediction costs more still.
    fn plan_segmentation(&mut self, state: SegmentFeatures) -> Segmentation {
        if self.block_qp.is_some() && !self.lossless {
            return self.block_qp_segmentation(state);
        }
        if let Some(random) = &mut self.random {
            return random.segmentation(state, self.header.key);
        }
        let header = &self.header;
        if header.key
            || header.filter_level == 0
            || self.lossless
            || self.params.skip_filter_off < 0.5
        {
            return Segmentation::disabled(state);
        }
        let mut features = SegmentFeatures::default();
        features.filter[SKIP_SEGMENT] = Some(-(MAX_FILTER_DELTA));
        Segmentation {
            enabled: true,
            tree_probs: PENDING_TREE_PROBS,
            // Skip patterns change too much between frames for predicted
            // ids to pay: crowd 107.7k -> 112.0k bytes, airport 42.2k ->
            // 45.8k with temporal_update (150 frames).
            temporal: None,
            update_data: features != state,
            features,
        }
    }

    /// Whether this frame's segment ids are coded with its own tree
    /// probabilities, set once the tile is known (`end_frame`): always but
    /// in the syntax stress test, whose header probabilities are random.
    fn own_segment_tree_probs(&self) -> bool {
        self.header.segmentation.enabled && self.random.is_none()
    }

    /// The segment of a block (skipped inter blocks to the filter-off
    /// segment) and whether it is coded as predicted from the last map.
    fn segment_of(&mut self, row: usize, col: usize, log2: u8, planned: u8) -> (u8, bool) {
        let segmentation = self.header.segmentation;
        if !segmentation.enabled {
            return (0, false);
        }
        let id = match &mut self.random {
            Some(random) if self.block_qp.is_none() => random.segment_id(),
            _ => planned,
        };
        let predicted = segmentation.temporal.is_some()
            && id
                == self
                    .grids
                    .segment_map
                    .predicted(row, col, 1 << (log2 - MI_LOG2));
        (id, predicted)
    }

    /// The segmentation of a frame with block quantizers: always on, so
    /// any block can take a segment; the quantizer features are filled in
    /// once the frame's offsets are known (`finish_segment_quantizers`).
    /// It replaces `vp9.skip_filter_off`'s use of the segments. In the
    /// stress test the tree and prediction probabilities and the
    /// loop-filter features stay random.
    fn block_qp_segmentation(&mut self, state: SegmentFeatures) -> Segmentation {
        let key = self.header.key;
        match self.random.as_mut().map(|r| r.segmentation(state, key)) {
            Some(random) if random.enabled => random,
            _ => Segmentation {
                enabled: true,
                tree_probs: PENDING_TREE_PROBS,
                // As for skip segments, ids are coded explicitly.
                temporal: None,
                update_data: false,
                features: SegmentFeatures {
                    quantizer: [None; segmentation::MAX_SEGMENTS],
                    ..state
                },
            },
        }
    }

    /// Whether the block being coded is quantised finer than the frame (a
    /// block-QP refinement, `nc_encoder` `reuse.rs`).
    fn refining_block(&self) -> bool {
        self.coding_q_idx < self.header.base_q_idx
    }

    /// Whether the picture is an enhancement layer coded from the same
    /// instant's upscaled base only (a key or repair unit's S1): its residual
    /// is the detail the base lacks, all high frequencies, so it keeps every
    /// frequency as a refined block does (as `nc-av1`'s). With
    /// `vp9.partial_forward` (vp9-svc-fast) still content stayed at the
    /// blurred base: the undisturbed static half of
    /// `nc-cli/tests/static_recovery.rs` `vp9_svc_presets_recover` at 32.4 dB,
    /// 43.2 exact.
    fn coding_from_base_only(&self) -> bool {
        self.spatial
            .as_ref()
            .is_some_and(|sp| sp.current_layer.from_base_only)
    }

    /// The segment of a block coded `offset` from the frame quantizer
    /// (the engine asked `block_qp_offset` first, so it is in use), or
    /// `None` without block quantizers; sets the block's coding qindex.
    fn block_segment(&mut self, offset: i16) -> Option<u8> {
        let segments = self.block_qp.as_mut()?;
        if !self.header.segmentation.enabled {
            return None;
        }
        let (segment, coded) = segments.assign(offset);
        debug_assert_eq!(coded, offset, "offsets come from block_qp_offset");
        self.coding_q_idx = (i16::from(self.header.base_q_idx) + coded) as u8;
        Some(segment)
    }

    /// Write the frame's segment quantizers into its features, sending them
    /// when they changed.
    fn finish_segment_quantizers(&mut self) {
        let (Some(segments), true) = (&self.block_qp, self.header.segmentation.enabled) else {
            return;
        };
        let segmentation = &mut self.header.segmentation;
        segmentation.features.quantizer = segments.feature_data(segmentation.features.absolute);
        segmentation.update_data |= segmentation.features != self.segment_state;
    }

    /// The transform size of an inter block: from the largest down by up to
    /// `vp9.inter_tx_depth` sizes while each is cheaper than the last
    /// (SSE + lambda * token bits); leaves the block coded with it and
    /// returns its token bits when the search priced them.
    fn code_inter_residuals(
        &mut self,
        block: &Block,
        max: TxSize,
        source: &Picture,
        recon: &mut Picture,
    ) -> (TxSize, Option<f32>) {
        let dropped = if self.negligible_luma_prediction_error(block, source) {
            ResidualDrop::Luma
        } else {
            ResidualDrop::None
        };
        self.code_residuals(block, (max, TrialPlanes::All), None, dropped, source, recon);
        let depth = self.params.inter_tx_depth.round() as u8;
        // Speed heuristic: a block whose largest transform codes nothing
        // keeps it (smaller transforms rarely pay for a residual the larger
        // one rounds away). Measured with the search: see the commit.
        let coded = self.scratch.coded.iter().any(|c| c.eob_nonzero);
        if self.lossless
            || depth == 0
            || dropped == ResidualDrop::Luma
            || !coded
            || self.sparse_residual()
            || self.evenly_spread_residual(block, source)
        {
            return (max, None);
        }
        let lambda = self.lambda();
        let cost = |backend: &Self, recon: &Picture| {
            let bits = backend.token_bits(true);
            (
                backend.reconstruction_sse(block, source, recon) as f32 + lambda * bits,
                bits,
            )
        };
        let ((mut best_cost, mut best_bits), mut best) = (cost(self, recon), max);
        let (mut last, mut planes) = (max, TrialPlanes::All);
        let smallest = max.log2().saturating_sub(depth).max(TxSize::Tx4.log2());
        for log2 in (smallest..max.log2()).rev() {
            let candidate = TxSize::from_log2(log2);
            planes = self.trial_planes(block, last, candidate);
            if last == best {
                self.save_best_trial(block, recon, planes);
            }
            self.restore_prediction(block, recon, planes);
            self.code_residuals(
                block,
                (candidate, planes),
                None,
                ResidualDrop::None,
                source,
                recon,
            );
            last = candidate;
            let (candidate_cost, candidate_bits) = cost(self, recon);
            if candidate_cost >= best_cost {
                break;
            }
            (best, best_cost, best_bits) = (candidate, candidate_cost, candidate_bits);
        }
        if last != best {
            self.restore_best_trial(block, recon, planes);
        }
        (best, Some(best_bits))
    }

    /// The transform size of an intra block: from the largest down by up to
    /// `vp9.intra_tx_depth` sizes while each is cheaper (SSE + lambda *
    /// token bits; intra prediction follows the transform blocks, so every
    /// trial rewrites the whole block); leaves the block coded with it.
    fn code_intra_residuals(
        &mut self,
        block: &Block,
        max: TxSize,
        modes: (IntraMode, IntraMode),
        source: &Picture,
        recon: &mut Picture,
    ) -> TxSize {
        self.code_residuals(
            block,
            (max, TrialPlanes::All),
            Some(modes),
            ResidualDrop::None,
            source,
            recon,
        );
        let depth = self.intra_tx_depth();
        if self.lossless || depth == 0 {
            return max;
        }
        let lambda = self.lambda();
        let cost = |backend: &Self, recon: &Picture| {
            backend.reconstruction_sse(block, source, recon) as f32
                + lambda * backend.token_bits(false)
        };
        let (mut best, mut best_cost, mut last) = (max, cost(self, recon), max);
        let mut planes = TrialPlanes::All;
        let smallest = max.log2().saturating_sub(depth).max(TxSize::Tx4.log2());
        for log2 in (smallest..max.log2()).rev() {
            let candidate = TxSize::from_log2(log2);
            planes = self.trial_planes(block, last, candidate);
            if last == best {
                self.save_best_trial(block, recon, planes);
            }
            self.code_residuals(
                block,
                (candidate, planes),
                Some(modes),
                ResidualDrop::None,
                source,
                recon,
            );
            last = candidate;
            let candidate_cost = cost(self, recon);
            if candidate_cost >= best_cost {
                break;
            }
            (best, best_cost) = (candidate, candidate_cost);
        }
        if last != best {
            self.restore_best_trial(block, recon, planes);
        }
        best
    }

    /// How many smaller transforms below the largest an intra block tries.
    /// Speed heuristic (`vp9.inter_intra_tx_depth`, libvpx `vp9_pickmode.c`,
    /// whose real-time intra blocks code one transform size): intra blocks of
    /// inter frames search fewer sizes. Inter frames code few intra blocks
    /// and lean on their reference for detail, so their transform size
    /// matters less than on key frames, which keep `vp9.intra_tx_depth`.
    /// Measured in docs/handoff/vp9-faster.md.
    fn intra_tx_depth(&self) -> u8 {
        let depth = self.params.intra_tx_depth.round() as u8;
        if self.header.key {
            depth
        } else {
            depth.min(self.params.inter_intra_tx_depth.round() as u8)
        }
    }

    /// The trellis λ for `plane` in this frame (`None`: no trellis): the
    /// mode-decision λ times `quant.trellis_lambda_scale`, and for chroma
    /// also `quant.trellis_chroma` and the squared chroma/luma step ratio
    /// (x264/HM chroma λ practice).
    fn trellis_lambda(&self, plane: PlaneIndex) -> Option<f32> {
        let trellis = self
            .trellis
            .as_ref()
            .filter(|t| !self.lossless && (t.key_frames || !self.header.key))?;
        let scale = if plane == PlaneIndex::Y {
            1.0
        } else {
            let ratio =
                self.block_steps(plane).ac as f32 / self.block_steps(PlaneIndex::Y).ac as f32;
            trellis.chroma_scale? * ratio * ratio
        };
        Some(self.lambda() * trellis.lambda_scale * scale)
    }

    /// The rounding and planned minimum level (past qindex 255) of the
    /// luma and chroma of an intra or inter block, and the shave gap.
    fn quantization_controls(&self, intra: bool) -> ([(Rounding, i32); 2], usize) {
        let rounding = if intra {
            Rounding::Intra
        } else {
            match self.params.inter_rounding_divisor.round() as u8 {
                INTER_ROUNDING_DIVISOR => Rounding::Inter,
                n => Rounding::Fraction(n),
            }
        };
        let chroma_rounding = self.chroma_rounding(rounding);
        let shave_gap = if self.lossless {
            0
        } else {
            self.params.shave_gap.round() as usize
        };
        let with_min_level = |r| (r, transform::planned_min_level(r, self.lambda_scale));
        (
            [with_min_level(rounding), with_min_level(chroma_rounding)],
            shave_gap,
        )
    }

    fn build_frame_header(
        &mut self,
        plan: &FramePlan,
        (key, context_use, qindex): (bool, ContextUse, u8),
        ref_slots: [u8; 3],
        segment_state: SegmentFeatures,
    ) -> FrameHeader {
        // A spatial layer's frame takes its size from a reference of its own
        // layer (VP9 §6.2.6: 4 bytes a frame); a key unit's enhancement
        // picture, whose references are all the base, writes it.
        let size_from_ref = match &self.spatial {
            Some(sp) => {
                let size = (self.grids.width, self.grids.height);
                ref_slots
                    .iter()
                    .position(|&slot| sp.slot_sizes[usize::from(slot)] == Some(size))
            }
            None => Some(0),
        };
        // A base-layer picture is never shown itself (`superframe.rs`).
        let base_layer = self
            .spatial
            .as_ref()
            .is_some_and(|sp| sp.current_layer.spatial_id == 0);
        let interp_filter = match &mut self.random {
            Some(random) => random.interp_filter(),
            None => INTERP_FILTERS
                .get(self.params.interp_filter.round() as usize)
                .copied(),
        };
        let refresh = match plan.references.refresh {
            _ if key => ALL_SLOTS,
            // A unit without an enhancement picture shows its base picture
            // with `show_existing_frame`, which needs a slot holding it: one
            // that refreshes none (a disposable top temporal layer picture)
            // refreshes the inter-layer scratch slot, which only the same
            // unit's enhancement picture reads.
            mask if base_layer && mask.is_empty() => INTER_LAYER_SCRATCH_MASK.0,
            mask => mask.0,
        };
        FrameHeader {
            key,
            show_frame: !base_layer,
            error_resilient: context_use.error_resilient,
            context_index: context_use.index,
            refresh_context: context_use.refresh,
            width: self.grids.width,
            height: self.grids.height,
            ref_slots,
            size_from_ref,
            refresh,
            interp_filter,
            allow_hp: !key
                && match &mut self.random {
                    Some(random) => random.allow_hp(),
                    None => f64::from(qindex) < f64::from(self.params.high_precision_qindex),
                },
            sharpness: match &mut self.random {
                Some(random) => random.sharpness(),
                None => self.params.filter_sharpness.round() as u8,
            },
            filter_level: match (&mut self.random, self.lossless) {
                (_, true) => 0,
                (Some(random), false) => random.filter_level(),
                (None, false) => filter_level(
                    qindex,
                    plan.frame_type == FrameType::Key,
                    self.params.filter_level_scale,
                ),
            },
            base_q_idx: qindex,
            segmentation: Segmentation::disabled(segment_state),
            chroma_delta_q: match (self.lossless, self.chroma_policy) {
                (true, _) => 0,
                (false, Some(policy)) => policy_chroma_delta(policy, qindex),
                (false, None) => self.own_chroma_delta(qindex),
            },
        }
    }

    fn predict_inter_block(
        &mut self,
        block: &Block,
        position: &BlockPosition,
        mv: EngineMv,
        source: &Picture,
        engine_ref: &Picture,
        recon: &mut Picture,
    ) -> (InterMode, Mv, MvRefs, nc_dsp::Vp9Filter) {
        let refs = self.refs_for(block);
        let (held_pic, ref_size) = self.active_reference();
        let ref_pic = held_pic.as_deref().unwrap_or(engine_ref);
        let mv =
            self.refine_high_precision(source, (ref_pic, ref_size), position, &refs, to_vp9_mv(mv));
        let allow_hp = self.header.allow_hp;
        let mode = match &mut self.random {
            Some(random) => random.inter_mode(&refs, mv),
            None => cheapest_inter_mode(&self.mode_costs, &refs, mv, allow_hp).0,
        };
        let (filter, luma_ready) = self.choose_filter(source, (ref_pic, ref_size), position, mv);
        let side = block.size();
        let scratch = &mut self.scratch;
        recon::predict_inter(
            &self.dsp,
            &mut self.inter,
            &mut scratch.prediction,
            (ref_pic, ref_size, position),
            (mv, filter),
            luma_ready.then_some(&scratch.filter_best[..side * side]),
        );
        copy_block_in(block, &scratch.prediction, recon, TrialPlanes::All);
        (mode, mv, refs, filter)
    }

    /// Transform, quantize and reconstruct every transform block of the
    /// block (intra blocks predict each transform block first). Levels
    /// stay in the scratch buffers for [`Vp9Backend::write_tokens`].
    /// `dropped` planes code no residual (stress test: all, the block is
    /// skipped; `vp9.pre_tx_skip_energy`: luma).
    /// With [`TrialPlanes::LumaOnly`] the chroma blocks of the last call
    /// (and their reconstruction) stay as they are.
    #[allow(clippy::too_many_lines)]
    fn code_residuals(
        &mut self,
        block: &Block,
        (tx, planes): (TxSize, TrialPlanes),
        intra: Option<(IntraMode, IntraMode)>,
        dropped: ResidualDrop,
        source: &Picture,
        recon: &mut Picture,
    ) {
        let (row, col) = (block.y >> MI_LOG2, block.x >> MI_LOG2);
        let (top, left) = (row > 0, col > 0);
        let (plane_quantization, shave_gap) = self.quantization_controls(intra.is_some());
        let shortcuts = self.residual_shortcuts(intra.is_some());
        self.scratch.start_trial(planes);
        // Exact shortcut ([`Self::quad_sads`]): only for inter blocks coded
        // once, not where a transform-size search recodes the block, nor
        // where the real-time skip test (`vp9.nonrd_skip_scale`) measures
        // each transform block first (there it cost +0.2..0.4%).
        let quad_planes = intra.is_none()
            && !self.lossless
            && self.params.inter_tx_depth < 0.5
            && shortcuts.nonrd_scale <= 0.0;
        let mut offset = 0;
        for plane in PlaneIndex::ALL.into_iter().filter(|&p| planes.includes(p)) {
            let trellis_lambda = self.trellis_lambda(plane);
            let plane_tx = self.plane_tx(block, tx, plane);
            let skip_plane = dropped.includes(plane)
                || self.plane_skip(block, (plane, plane_tx), &shortcuts, (source, recon));
            let quads = if quad_planes && !skip_plane {
                self.quad_sads(block, (plane, plane_tx), (source, recon))
            } else {
                None
            };
            for tx_block in recon::tx_blocks(plane, block.x, block.y, block.size(), plane_tx) {
                let mode = intra.map(|(y, uv)| if plane == PlaneIndex::Y { y } else { uv });
                if let Some(mode) = mode {
                    recon::predict_intra(&self.dsp, recon, &tx_block, mode, top, left);
                }
                let ty = recon::tx_type(plane, mode, plane_tx, self.lossless);
                let area = plane_tx.area();
                if skip_plane {
                    // The prediction in `recon` is the reconstruction.
                    self.scratch
                        .coded
                        .push(CodedTx::new(&tx_block, ty, false, offset));
                    offset += area;
                    continue;
                }
                let steps = self.header.steps_at(self.coding_q_idx, plane);
                let (rounding, min_level) = plane_quantization[usize::from(plane != PlaneIndex::Y)];
                let tx_coding = (plane_tx, offset, (rounding, min_level));
                let nonrd = (steps, shortcuts.nonrd_scale);
                if self.code_nonrd(source, recon, &tx_block, tx_coding, nonrd) {
                    offset += area;
                    continue;
                }
                let side = plane_tx.side();
                if let Some(quads) = quads {
                    let sad = quads.tx_sad(&self.scratch.quad_sads, &tx_block, side);
                    if ty == transform::TxType::DctDct
                        && transform::dct_levels_are_zero(
                            plane_tx,
                            sad,
                            steps,
                            (rounding, min_level),
                            shortcuts.zero_scale,
                        )
                    {
                        self.scratch
                            .coded
                            .push(CodedTx::new(&tx_block, ty, false, offset));
                        offset += area;
                        continue;
                    }
                }
                let residual = &mut self.scratch.residual[..area];
                let (bx, by) = (tx_block.x as isize, tx_block.y as isize);
                let src = source
                    .plane(plane)
                    .region(bx, by, side, side)
                    .expect("block inside the source");
                let rec = recon
                    .plane(plane)
                    .region(bx, by, side, side)
                    .expect("block inside the picture");
                let size = nc_dsp::BlockSize::new(side, side).expect("square VP9 transform block");
                let sad = self.dsp.residual_sad(size, &src, &rec, residual, side);
                let coefficients = &mut self.scratch.coefficients[..area];
                let levels = &mut self.scratch.levels[offset..offset + area];
                let certainly_zero = ty == transform::TxType::DctDct
                    && !self.lossless
                    && transform::dct_levels_are_zero(
                        plane_tx,
                        sad,
                        steps,
                        (rounding, min_level),
                        shortcuts.zero_scale,
                    );
                let zero_limit =
                    transform::zero_coefficient_limit(plane_tx, steps, (rounding, min_level));
                let transformed = !certainly_zero
                    && transform_residual(
                        &self.dsp,
                        (plane_tx, ty, self.lossless, shortcuts.partial_forward),
                        residual,
                        zero_limit,
                        coefficients,
                    );
                let nonzero = transformed
                    && transform::quantize(
                        &self.dsp,
                        plane_tx,
                        self.lossless,
                        coefficients,
                        steps,
                        (rounding, min_level),
                        levels,
                    );
                let nonzero = nonzero && trim_levels(plane_tx, ty, (min_level, shave_gap), levels);
                let nonzero = match (&mut self.trellis, trellis_lambda) {
                    (Some(trellis), Some(lambda)) if nonzero => {
                        let block = (
                            &self.context,
                            &self.grids.contexts,
                            &tx_block,
                            intra.is_none(),
                        );
                        trellis.quantize(block, ty, (lambda, coefficients, steps), levels)
                    }
                    _ => nonzero,
                };
                if nonzero {
                    self.buffers
                        .add_residual(recon, &tx_block, ty, self.lossless, levels, steps);
                }
                self.scratch
                    .coded
                    .push(CodedTx::new(&tx_block, ty, nonzero, offset));
                offset += area;
            }
        }
        self.scratch.finish_trial();
    }

    /// The inexact residual shortcuts in force for an intra or inter block.
    fn residual_shortcuts(&self, intra: bool) -> ResidualShortcuts {
        // A block quantised finer than the frame is being refined (the
        // reuse term restoring detail a degraded reference lost): the high
        // frequencies the approximation drops are what it is for. With
        // `vp9.inter_tx_split_spread` keeping such blocks in one 32x32
        // transform, a still region copied from a starved key stayed 8 dB
        // short for good (`vp9-fast`, `nc-cli/tests/static_recovery.rs`).
        // An enhancement picture coded from the upscaled base only keeps every
        // frequency too (`coding_from_base_only`).
        let exact = self.refining_block() || self.coding_from_base_only();
        let partial_forward = self.params.partial_forward >= 0.5 && !self.lossless && !exact;
        // Speed heuristic (`vp9.pre_transform_zero_scale`): inter residual
        // within a multiple of the zero proof's limit is dropped untransformed.
        // Intra blocks and refined blocks keep the exact proof (a refined
        // block exists to restore small detail).
        let zero_scale = if !intra && !exact {
            self.params.pre_transform_zero_scale
        } else {
            1.0
        };
        // Speed heuristic (`vp9.nonrd_skip_scale`): libvpx's real-time skip
        // test, inter blocks only; refined blocks keep every frequency.
        let nonrd_scale = if !intra && !self.lossless && !exact {
            self.params.nonrd_skip_scale
        } else {
            0.0
        };
        ResidualShortcuts {
            partial_forward,
            zero_scale,
            nonrd_scale,
        }
    }

    /// Exact shortcut: an inter plane block's 4x4 and 8x8 transform blocks
    /// take their SADs from one pass over the block (into
    /// `scratch.quad_sads`), so those the zero proof drops (most of them at
    /// real-time rates) build no views or residual. Only small transforms,
    /// several to the plane block, pay for the pass. vp9-fastest (8x8
    /// inter transforms, no size search): -2.4% x86 instructions.
    fn quad_sads(
        &mut self,
        block: &Block,
        (plane, tx): (PlaneIndex, TxSize),
        pictures: (&Picture, &Picture),
    ) -> Option<QuadSads> {
        let ss = usize::from(plane != PlaneIndex::Y);
        let (x, y, n) = (block.x >> ss, block.y >> ss, block.size() >> ss);
        if tx.side() > QUAD_STRIP || n <= tx.side() {
            return None;
        }
        plane_quad_sads(
            &self.dsp,
            pictures,
            plane,
            (x, y, n),
            &mut self.scratch.quad_sads,
        );
        Some(QuadSads {
            x,
            y,
            columns: n / QUAD_SIDE,
        })
    }

    /// The transform size of `plane` for luma transform size `tx`.
    fn plane_tx(&self, block: &Block, tx: TxSize, plane: PlaneIndex) -> TxSize {
        if plane == PlaneIndex::Y || self.lossless {
            tx
        } else {
            uv_tx_size(block.log2_size(), tx)
        }
    }

    /// Speed heuristic (`vp9.nonrd_block_test`, libvpx `model_rd_for_sb_y`
    /// as written): the whole plane block's residual, averaged over its
    /// transform blocks, passes [`nonrd_skip`] with nothing to code, so
    /// no transform block is visited. Inter blocks only (`scale` 0: off).
    /// Measured in docs/handoff/vp9-faster.md.
    fn plane_skip(
        &self,
        block: &Block,
        (plane, tx): (PlaneIndex, TxSize),
        shortcuts: &ResidualShortcuts,
        (source, recon): (&Picture, &Picture),
    ) -> bool {
        let scale = shortcuts.nonrd_scale;
        if scale <= 0.0 || self.params.nonrd_block_test < 0.5 {
            return false;
        }
        let ss = usize::from(plane != PlaneIndex::Y);
        let (x, y, n) = (block.x >> ss, block.y >> ss, block.size() >> ss);
        if n == tx.side() {
            // One transform block: its own test is the same.
            return false;
        }
        let (xi, yi) = (x as isize, y as isize);
        let src = source.plane(plane).region(xi, yi, n, n);
        let rec = recon.plane(plane).region(xi, yi, n, n);
        let (src, rec) = (
            src.expect("block inside the source"),
            rec.expect("block inside the picture"),
        );
        let size = nc_dsp::BlockSize::new(n, n).expect("square VP9 plane block");
        let whole = self.dsp.variance(size, &src, &rec);
        // libvpx averages the block's sums over its transform blocks.
        let blocks = ((n / tx.side()) * (n / tx.side())) as i64;
        let average = nc_dsp::Variance {
            sse: (i64::from(whole.sse) / blocks) as u32,
            sum: (i64::from(whole.sum) / blocks) as i32,
        };
        let steps = self.header.steps_at(self.coding_q_idx, plane);
        nonrd_skip(average, tx, steps, scale) == NonrdSkip::All
    }

    /// [`nonrd_skip`] of an inter DCT transform block (`scale` 0: off).
    fn nonrd_skip_of(
        &self,
        tx_block: &TxBlock,
        tx: TxSize,
        steps: transform::Steps,
        scale: f32,
        source: &Picture,
        recon: &Picture,
    ) -> NonrdSkip {
        if scale <= 0.0 {
            return NonrdSkip::None;
        }
        let side = tx.side();
        let (x, y) = (tx_block.x as isize, tx_block.y as isize);
        let src = source.plane(tx_block.plane).region(x, y, side, side);
        let rec = recon.plane(tx_block.plane).region(x, y, side, side);
        let (src, rec) = (
            src.expect("block inside the source"),
            rec.expect("block inside the picture"),
        );
        let size = nc_dsp::BlockSize::new(side, side).expect("square VP9 transform block");
        nonrd_skip(self.dsp.variance(size, &src, &rec), tx, steps, scale)
    }

    /// Code an inter DCT transform block that [`nonrd_skip`] judges (at
    /// `scale`, 0 off) to need no AC coefficient: its DC coefficient alone
    /// (libvpx `vp9_xform_quant_dc`) or nothing. Returns whether it did;
    /// `false` leaves the block to the ordinary path.
    fn code_nonrd(
        &mut self,
        source: &Picture,
        recon: &mut Picture,
        tx_block: &TxBlock,
        (tx, offset, quantization): (TxSize, usize, (Rounding, i32)),
        (steps, scale): (transform::Steps, f32),
    ) -> bool {
        let skip = self.nonrd_skip_of(tx_block, tx, steps, scale, source, recon);
        let nonzero = match skip {
            NonrdSkip::None => return false,
            NonrdSkip::All => false,
            NonrdSkip::DcOnly(dc) => {
                self.code_dc(recon, tx_block, (tx, offset, quantization), steps, dc)
            }
        };
        let ty = transform::TxType::DctDct;
        self.scratch
            .coded
            .push(CodedTx::new(tx_block, ty, nonzero, offset));
        true
    }

    /// Code only the DC coefficient `dc` of a transform block; returns
    /// whether its level is nonzero.
    fn code_dc(
        &mut self,
        recon: &mut Picture,
        tx_block: &TxBlock,
        (tx, offset, quantization): (TxSize, usize, (Rounding, i32)),
        steps: transform::Steps,
        dc: i32,
    ) -> bool {
        let area = tx.area();
        let coefficients = &mut self.scratch.coefficients[..area];
        coefficients.fill(0);
        coefficients[0] = dc;
        let levels = &mut self.scratch.levels[offset..offset + area];
        let nonzero = transform::quantize(
            &self.dsp,
            tx,
            false,
            coefficients,
            steps,
            quantization,
            levels,
        );
        if nonzero {
            let ty = transform::TxType::DctDct;
            self.buffers
                .add_residual(recon, tx_block, ty, false, levels, steps);
        }
        nonzero
    }

    /// The planes a transform-size trial from `last` to `candidate` codes:
    /// chroma whose transform size does not change would code exactly as
    /// before (its prediction and quantization do not depend on luma).
    fn trial_planes(&self, block: &Block, last: TxSize, candidate: TxSize) -> TrialPlanes {
        let chroma = |tx| uv_tx_size(block.log2_size(), tx);
        if !self.lossless && chroma(last) == chroma(candidate) {
            TrialPlanes::LumaOnly
        } else {
            TrialPlanes::All
        }
    }

    /// Write the coefficient tokens of every transform block, updating the
    /// above/left nonzero contexts.
    fn write_tokens(&mut self, inter: bool) {
        for coded in &self.scratch.coded {
            let (ux, uy) = coded.units();
            let ctx = self
                .grids
                .contexts
                .coefficient(coded.plane, ux, uy, coded.tx);
            let model = coded.model(inter);
            let levels = if coded.eob_nonzero {
                &self.scratch.levels[coded.levels()]
            } else {
                &transform::ZERO_LEVELS[..coded.tx.area()]
            };
            let mut coder = coeff::Coder::new(model, &self.context, &mut self.counts);
            let eob = coeff::write(&mut self.writer, levels, model, &mut coder, ctx);
            self.grids
                .contexts
                .set_coefficient(coded.plane, ux, uy, coded.tx, eob > 0);
        }
    }
}

/// Write a block's transform size (`TX_MODE_SELECT`, VP9 §6.4.6 / libvpx
/// `write_selected_tx_size`): coded unless lossless or a skipped inter
/// block, which take the largest.
#[allow(clippy::too_many_arguments)]
fn write_tx_size(
    writer: &mut TileRecorder,
    fc: &FrameContext,
    counts: &mut Counts,
    grid: &ModeInfoGrid,
    row: usize,
    col: usize,
    info: &ModeInfo,
    lossless: bool,
) {
    let max = max_tx_size(info.size_log2);
    if lossless || (info.skip && info.is_inter()) {
        return;
    }
    let ctx = contexts::tx_size(grid, row, col, max);
    let probs = fc.tx_probs(max, ctx);
    let base = recorder::tx_slot(max, ctx);
    let size = info.tx_size as usize;
    writer.write_mode(size != 0, probs[0], base);
    if size != 0 && max >= TxSize::Tx16 {
        writer.write_mode(size != 1, probs[1], base + 1);
        if size != 1 && max >= TxSize::Tx32 {
            writer.write_mode(size != 2, probs[2], base + 2);
        }
    }
    counts.count_tx(max, ctx, info.tx_size);
}

/// Drop the quantized levels the encoder chooses not to code; returns
/// whether any is left. Past qindex 255 the residual is quantised as if at
/// the planned step (`FramePlan::lambda_scale`, levels below `min_level`):
/// λ alone left blocks drifting into dense residual (`pan_bdd_00cef86b` at
/// 150 kbps: 283 kbps with 2 octaves, 403 with 3). `shave_gap`
/// (`vp9.shave_gap`) drops isolated trailing ±1 levels.
fn trim_levels(
    tx: TxSize,
    ty: transform::TxType,
    (min_level, shave_gap): (i32, usize),
    levels: &mut [i32],
) -> bool {
    (min_level <= 1 || transform::apply_planned_dead_zone(tx, min_level, levels))
        && (shave_gap == 0 || coeff::shave_isolated_ones(levels, tx, ty, shave_gap))
}

/// VP9's delta range (`delta_q` is 4 bits and a sign, §6.2.9).
const CHROMA_DELTA_LIMIT: i32 = 15;
/// Bounds of the adaptive chroma weight (`vp9.chroma_weight_adapt`): the
/// error ratio of a near-flat chroma plane must not drive it to extremes.
const ADAPTIVE_WEIGHT_MIN: f64 = 0.25;
const ADAPTIVE_WEIGHT_MAX: f64 = 16.0;
/// VP9 8-bit steps over the H.264-normalised sample step (as AV1's).
const VP9_STEP_SCALE: f64 = 8.0;

/// The `delta_q_uv` (DC and AC) that puts chroma's AC step at frame qindex
/// `qindex` nearest the engine chroma policy's (`nc_encoder::chroma_policy`).
/// The inexact residual shortcuts of one block
/// ([`Vp9Backend::residual_shortcuts`]).
struct ResidualShortcuts {
    /// `vp9.partial_forward`.
    partial_forward: bool,
    /// `vp9.pre_transform_zero_scale` (1: the exact proof).
    zero_scale: f32,
    /// `vp9.nonrd_skip_scale` (0: off).
    nonrd_scale: f32,
}

/// What libvpx's real-time skip test leaves of a transform block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NonrdSkip {
    /// Code it normally.
    None,
    /// Only its DC coefficient (this value) can be nonzero.
    DcOnly(i32),
    /// Nothing: every coefficient would quantize to zero.
    All,
}

/// libvpx's real-time skip test (`vp9_pickmode.c` `model_rd_for_sb_y`,
/// BSD-licensed; reimplemented): a transform block whose residual variance
/// is below `ac_q^2 / 64` has no AC coefficient worth coding, and one whose
/// squared mean is also below `dc_q^2 / 64` none at all; `scale` multiplies
/// both thresholds. libvpx averages the test over a block's transform units;
/// here each unit tests its own residual. The DC coefficient is libvpx's
/// `vpx_fdctNxN_1` (8x the orthonormal DC, 32x32 4x): the residual sum
/// times 8 / side (32x32: / 8).
fn nonrd_skip(
    variance: nc_dsp::Variance,
    tx: TxSize,
    steps: transform::Steps,
    scale: f32,
) -> NonrdSkip {
    let area = tx.area() as i64;
    let (sse, sum) = (i64::from(variance.sse), i64::from(variance.sum));
    let var = sse - sum * sum / area;
    let threshold =
        |step: i32| scale * ((i64::from(step) * i64::from(step)) >> NONRD_THRESHOLD_SHIFT) as f32;
    if var as f32 >= threshold(steps.ac) {
        return NonrdSkip::None;
    }
    if ((sse - var) as f32) < threshold(steps.dc) {
        return NonrdSkip::All;
    }
    let dc = if tx == TxSize::Tx32 {
        sum / i64::from(DC_SUM_DIVISOR_32)
    } else {
        sum * DCT_DC_SCALE / tx.side() as i64
    };
    NonrdSkip::DcOnly(dc as i32)
}

/// libvpx's skip thresholds are the squared dequantizer step `>> 6`.
const NONRD_THRESHOLD_SHIFT: u32 = 6;
/// Forward DCT outputs are 8x orthonormal below 32x32 ...
const DCT_DC_SCALE: i64 = 8;
/// ... and the 32x32 DC is the residual sum / 8 (`vpx_fdct32x32_1`).
const DC_SUM_DIVISOR_32: i32 = 8;

fn policy_chroma_delta(policy: nc_encoder::chroma_policy::ChromaPolicy, qindex: u8) -> i8 {
    let ac = |d: i32| f64::from(transform::Steps::new(qindex, 0, d as i8).ac);
    let target = policy.chroma_step(ac(0) / VP9_STEP_SCALE) * VP9_STEP_SCALE;
    nc_encoder::chroma_policy::closest_index(-CHROMA_DELTA_LIMIT..=CHROMA_DELTA_LIMIT, ac, target)
        as i8
}

/// The frame filter level from the quantizer (libvpx `LPF_PICK_FROM_Q`).
/// `scale` (`vp9.filter_level_scale`) multiplies the guess for tuning.
fn filter_level(qindex: u8, key: bool, scale: f32) -> u8 {
    let q = AC_QUANT[usize::from(qindex)];
    let guess = (q * FILTER_FROM_Q_SLOPE + FILTER_FROM_Q_OFFSET + (1 << (FILTER_FROM_Q_SHIFT - 1)))
        >> FILTER_FROM_Q_SHIFT;
    let guess = if key {
        guess - KEY_FILTER_REDUCTION
    } else {
        guess
    };
    ((guess as f32 * scale).round() as i32).clamp(0, i32::from(loopfilter::MAX_LEVEL)) as u8
}

/// The qindex whose AC step is closest to, without going below, `scale`
/// times that of `qindex` (libvpx `vp9_compute_qdelta`), at least 1 so the
/// frame stays lossy.
fn scaled_qindex(qindex: u32, scale: f64) -> u32 {
    let target = f64::from(AC_QUANT[qindex as usize]) * scale;
    let found = AC_QUANT
        .iter()
        .position(|&q| f64::from(q) >= target)
        .unwrap_or(AC_QUANT.len() - 1);
    (found as u32).max(1)
}

/// The cheapest inter mode signalling `mv`, with its bits (mode, plus the
/// vector difference for NEWMV).
fn cheapest_inter_mode(
    costs: &ModeCosts,
    refs: &MvRefs,
    mv: Mv,
    allow_hp: bool,
) -> (InterMode, f32) {
    let mode_cost = |mode: InterMode| costs.inter[refs.mode_context][mode as usize];
    let usehp = allow_hp && motion::uses_high_precision(refs.nearest);
    let (dr, dc) = (mv.row - refs.nearest.row, mv.col - refs.nearest.col);
    // Without 1/8 precision an odd difference cannot be coded as NEWMV.
    let new_cost = if usehp || (dr & 1 == 0 && dc & 1 == 0) {
        mode_cost(InterMode::New) + costs.mv.cost(dr, dc, usehp)
    } else {
        f32::INFINITY
    };
    let mut best = (InterMode::New, new_cost);
    for (mode, vector) in [
        (InterMode::Zero, Mv::ZERO),
        (InterMode::Nearest, refs.nearest),
        (InterMode::Near, refs.near),
    ] {
        let cost = mode_cost(mode);
        if vector == mv && cost < best.1 {
            best = (mode, cost);
        }
    }
    best
}

fn to_vp9_mv(mv: EngineMv) -> Mv {
    Mv {
        row: i32::from(mv.y) * MV_PER_ENGINE_MV,
        col: i32::from(mv.x) * MV_PER_ENGINE_MV,
    }
}

/// Reference vectors are even eighths (no high precision), so exact.
fn to_engine_mv(mv: Mv) -> EngineMv {
    EngineMv {
        x: (mv.col / MV_PER_ENGINE_MV) as i16,
        y: (mv.row / MV_PER_ENGINE_MV) as i16,
    }
}

/// Copy the block's samples of every plane out of `picture`, packed plane
/// after plane.
fn copy_block_out(block: &Block, picture: &Picture, packed: &mut [u8], planes: TrialPlanes) {
    let mut offset = 0;
    for plane in PlaneIndex::ALL.into_iter().filter(|&p| planes.includes(p)) {
        let ss = usize::from(plane != PlaneIndex::Y);
        let (x, y, n) = (block.x >> ss, block.y >> ss, block.size() >> ss);
        let view = picture.plane(plane).view();
        for dy in 0..n {
            packed[offset..offset + n].copy_from_slice(&view.row(y + dy)[x..x + n]);
            offset += n;
        }
    }
}

/// Forward-transform a block's `residual` into raster `coefficients`;
/// false when the block certainly quantizes to zero (every coefficient at
/// most `zero_limit`, `transform::zero_coefficient_limit`), which the
/// bounded 16x16/32x32 DCT proves exactly before its row pass, leaving
/// `coefficients` unspecified. With `partial` a 32x32 block keeps only its
/// low 16x16 frequencies (`vp9.partial_forward`).
fn transform_residual(
    dsp: &nc_dsp::Dsp,
    (tx, ty, lossless, partial): (TxSize, transform::TxType, bool, bool),
    residual: &[i16],
    zero_limit: u32,
    coefficients: &mut [i32],
) -> bool {
    let side = tx.side();
    if partial && tx == TxSize::Tx32 {
        // 32x32 is DCT only; its low 16x16 frequencies alone.
        dsp.av1_forward_dct_raster_low(side, residual, side, coefficients);
        return true;
    }
    let bounded = match tx {
        TxSize::Tx16 => Some(nc_dsp::DctSize::N16),
        TxSize::Tx32 => Some(nc_dsp::DctSize::N32),
        TxSize::Tx4 | TxSize::Tx8 => None,
    };
    match bounded {
        Some(size) if ty == transform::TxType::DctDct && !lossless => {
            dsp.av1_forward_dct_raster_unless_zero(size, residual, side, zero_limit, coefficients)
        }
        _ => {
            transform::forward(dsp, tx, ty, lossless, residual, coefficients);
            true
        }
    }
}

/// Levels a trial of `planes` codes: luma first, then chroma.
fn trial_levels(block: &Block, planes: TrialPlanes) -> usize {
    let luma = block.size() * block.size();
    if planes == TrialPlanes::All {
        luma * 3 / 2
    } else {
        luma
    }
}

/// Write the `planes` samples packed by [`copy_block_out`] back into
/// `picture`.
fn copy_block_in(block: &Block, packed: &[u8], picture: &mut Picture, planes: TrialPlanes) {
    let mut offset = 0;
    for plane in PlaneIndex::ALL.into_iter().filter(|&p| planes.includes(p)) {
        let ss = usize::from(plane != PlaneIndex::Y);
        let (x, y, n) = (block.x >> ss, block.y >> ss, block.size() >> ss);
        let mut view = picture.plane_mut(plane).view_mut();
        let mut dst = view
            .sub_view_mut(x, y, n, n)
            .expect("block inside the picture");
        copy_square(&mut dst, &packed[offset..offset + n * n], n);
        offset += n * n;
    }
}

/// Side of the cells of [`plane_quad_sads`] (the 4x4 quadrants of
/// `Dsp::sad_quads_8x8_row`) and the strip height of one kernel call.
const QUAD_SIDE: usize = 4;
const QUAD_STRIP: usize = 8;
/// Cells of the largest plane block (64x64 luma).
const QUAD_SADS: usize = (64 / QUAD_SIDE) * (64 / QUAD_SIDE);

/// The 4x4 SADs of the `n`x`n` plane block at (`x`, `y`) of `source`
/// against `recon` (the block's prediction), raster order, into `out`.
fn plane_quad_sads(
    dsp: &nc_dsp::Dsp,
    (source, recon): (&Picture, &Picture),
    plane: PlaneIndex,
    (x, y, n): (usize, usize, usize),
    out: &mut [u32],
) {
    let columns = n / QUAD_SIDE;
    let blocks = n / QUAD_STRIP;
    let mut strip = [[0u32; 4]; 64 / QUAD_STRIP];
    for row in (0..n).step_by(QUAD_STRIP) {
        let (xi, yi) = (x as isize, (y + row) as isize);
        let src = source.plane(plane).region(xi, yi, n, QUAD_STRIP);
        let rec = recon.plane(plane).region(xi, yi, n, QUAD_STRIP);
        let (src, rec) = (
            src.expect("block inside the source"),
            rec.expect("block inside the picture"),
        );
        dsp.sad_quads_8x8_row(&src, &rec, &mut strip[..blocks]);
        let top = (row / QUAD_SIDE) * columns;
        for (i, [tl, tr, bl, br]) in strip[..blocks].iter().copied().enumerate() {
            out[top + 2 * i] = tl;
            out[top + 2 * i + 1] = tr;
            out[top + columns + 2 * i] = bl;
            out[top + columns + 2 * i + 1] = br;
        }
    }
}

/// Where [`plane_quad_sads`] measured a plane block: its origin and the
/// width of its rows of cells.
#[derive(Debug, Clone, Copy)]
struct QuadSads {
    x: usize,
    y: usize,
    columns: usize,
}

impl QuadSads {
    /// The SAD of the `side`x`side` transform block `tx_block`.
    fn tx_sad(self, quads: &[u32], tx_block: &TxBlock, side: usize) -> u32 {
        let (cx, cy) = (
            (tx_block.x - self.x) / QUAD_SIDE,
            (tx_block.y - self.y) / QUAD_SIDE,
        );
        let cells = side / QUAD_SIDE;
        (cy..cy + cells)
            .map(|row| {
                let start = row * self.columns + cx;
                quads[start..start + cells].iter().sum::<u32>()
            })
            .sum()
    }
}

/// One `n`x`n` packed block into `dst` through `nc_dsp::copy_square`: one
/// range check, then a fixed-size move per row (HEVC's prediction copies;
/// the per-row slice checks cost about as much as the copies).
fn copy_square(dst: &mut nc_common::PlaneViewMut<'_>, packed: &[u8], n: usize) {
    match n {
        4 => nc_dsp::copy_square::<4>(dst, packed, n),
        8 => nc_dsp::copy_square::<8>(dst, packed, n),
        16 => nc_dsp::copy_square::<16>(dst, packed, n),
        32 => nc_dsp::copy_square::<32>(dst, packed, n),
        64 => nc_dsp::copy_square::<64>(dst, packed, n),
        _ => unreachable!("VP9 plane blocks are 4 to 64 samples square"),
    }
}

/// The planes of a block that code no residual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidualDrop {
    None,
    Luma,
    All,
}

impl ResidualDrop {
    fn includes(self, plane: PlaneIndex) -> bool {
        self == Self::All || (self == Self::Luma && plane == PlaneIndex::Y)
    }
}

/// Denominator of `vp9.pre_tx_skip_energy` (AV1's units: percent).
const ENERGY_SCALE: f32 = 100.0;

/// The planes a transform-size trial codes afresh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrialPlanes {
    All,
    /// Luma only: chroma keeps the last trial's coding.
    LumaOnly,
}

impl TrialPlanes {
    fn includes(self, plane: PlaneIndex) -> bool {
        self == Self::All || plane == PlaneIndex::Y
    }
}

/// Luma/chroma intra modes and inter modes.
const INTRA_MODES: usize = 10;
const INTER_MODES: usize = 4;
/// `INTER_MODE_CONTEXTS`, `BLOCK_SIZE_GROUPS`.
const INTER_MODE_CONTEXTS: usize = 7;
const SIZE_GROUPS: usize = 4;

/// Bits of every mode symbol under one frame's probabilities. Mode
/// decision prices these per candidate (hundreds of thousands of times a
/// frame) while the probabilities only change between frames.
struct ModeCosts {
    inter: [[f32; INTER_MODES]; INTER_MODE_CONTEXTS],
    mv: motion::MvCosts,
    switchable: [[f32; SWITCHABLE_FILTERS]; SWITCHABLE_CONTEXTS],
    y: [[f32; INTRA_MODES]; SIZE_GROUPS],
    uv: [[f32; INTRA_MODES]; INTRA_MODES],
    kf_y: [[[f32; INTRA_MODES]; INTRA_MODES]; INTRA_MODES],
    kf_uv: [[f32; INTRA_MODES]; INTRA_MODES],
    /// The cheapest inter leaf's skip, intra/inter, reference and mode
    /// bits (each at its cheapest context).
    leaf: f32,
}

impl ModeCosts {
    fn new(context: &FrameContext) -> Box<Self> {
        fn symbols<const N: usize>(tree: &tree::Tree, probs: &[u8]) -> [f32; N] {
            std::array::from_fn(|value| tree::cost(tree, probs, value))
        }
        let intra = |probs: &[u8; 9]| symbols::<INTRA_MODES>(tree::INTRA_MODE, probs);
        let mean = |costs: &[f32]| costs.iter().sum::<f32>() / costs.len() as f32;
        let inter = context
            .inter_mode
            .map(|probs| symbols::<INTER_MODES>(tree::INTER_MODE, &probs));
        let leaf = mean(&context.skip.map(|p| tree::bit_cost(true, p)))
            + mean(&context.intra_inter.map(|p| tree::bit_cost(true, p)))
            + mean(&context.single_ref.map(|p| tree::bit_cost(false, p[0])))
            + mean(&inter.map(|modes| modes.into_iter().fold(f32::INFINITY, f32::min)));
        Box::new(Self {
            mv: motion::MvCosts::new(context),
            inter,
            switchable: context
                .switchable_interp
                .map(|probs| symbols::<SWITCHABLE_FILTERS>(tree::SWITCHABLE_INTERP, &probs)),
            y: context.y_mode.map(|probs| intra(&probs)),
            uv: context.uv_mode.map(|probs| intra(&probs)),
            kf_y: tables::KF_Y_MODE.map(|row| row.map(|probs| intra(&probs))),
            kf_uv: tables::KF_UV_MODE.map(|probs| intra(&probs)),
            leaf,
        })
    }
}
