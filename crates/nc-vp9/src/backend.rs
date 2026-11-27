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
    Block, BlockDecision, Codec, CodecBackend, ConfigError, EncoderConfig, FilterEffort, FramePlan,
    FrameType, InterLayerMode, MAX_REFERENCE_SLOTS, MAX_SPATIAL_LAYERS, Mv as EngineMv,
    PredictionMode, QuantParams, ReferenceChoice, SlotMask, SpatialConfig, SpatialLayerFrame,
    StaticSkipTest, copy_zero_vector_prediction, spatial::layer_slot_offset,
};

use crate::bool_coder::BitSink;
use crate::coeff::{self, Model, PlaneType};
use crate::contexts::{self, EntropyContexts};
use crate::header::{ALL_SLOTS, ColourSpace, FrameHeader, MAX_DIMENSION};
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
use nc_encoder::segment_quantizers::SegmentQuantizers;

mod final_cost;
mod node_trial;

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
/// its held-out worst-1% VMAF is +3.8% (the same pan +44%). `VP9 Fast`
/// takes 1.4 (TRAIN -1.9%, VAL -1.7%, no guard).
const RATE_MODEL: nc_encoder::RateModel = nc_encoder::RateModel {
    cadence_scale: 1.2,
    first_key_lookahead: 0.3,
    first_key: Some(nc_encoder::first_key_model::FirstKeyCalibration {
        weights: [
            0.512_549, 0.948_092, 0.126_744, 0.605_836, 0.144_182, -0.910_492, 0.286_988,
            0.403_286, 0.581_013,
        ],
        fast_bias: 0.016_524,
    }),
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
/// Minimum `|offset|` (qindex units) accepted on `S0` of a three-layer spatial
/// stream (`16` qindex): `Segmentation::standalone` (`VP9 §6.2.11 / §6.4.8`)
/// sends `update_data = 1`, seven tree probabilities, and an explicit segment
/// id on every block (including `skip = 1`), costing ~120-180 bits/frame on
/// `320x180`, so small row-steering and cyclic-band nudges cost more in map
/// syntax than they save, while large `ReuseTerm` static-recovery refinements
/// still pass through. With the idle-segmentation switch-off
/// (`finish_segment_quantizers`), the `L3T1_KEY` 180p view went from +7.0% to
/// +1.3% BD-J vs 180p coded alone (11 chart clips; 720p view +1.0 pp,
/// svc-next merge review, 2026-11-14).
const THREE_LAYER_S0_MIN_SEGMENT_OFFSET: u16 = 16;
/// Minimum picture area (`640x360`) for `vp9.inter_intra_dc_only` mode 2:
/// sub-`360p` layers (`320x180` `S0`, `<0.15%` of three-layer instructions)
/// keep directional intra (`180p` pan borders otherwise lose `+5..+10%` BD-J;
/// with mode 2, 11-clip `L3T1` `180p` stays `+2.3% -> +2.3%`, `L3T1_KEY`
/// `+2.4% -> +2.5%`).
const MIN_INTER_INTRA_DC_AREA: usize = 640 * 360;

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
        /// 1 hands the engine VP9's own 8-tap filter (`luma_filter`), as
        /// H.264's `h264.exact_ranking` does: inter candidates are ranked
        /// on exact predictions (the engine's light-PD1 flow) and the
        /// winner walk (`decision.winner_walk_steps`) can run. 0 keeps the
        /// bilinear ranking every VP9 rung below Slow was tuned on.
        exact_ranking: 0.0, [0.0, 1.0],
            "1 ranks inter candidates on exact VP9 8-tap predictions (engine light-PD1 flow, enables the winner walk): slower, decides on the coded prediction",
        /// 1 (Default): `J_real --clips adopt` full TRAIN -0.9% / VAL -1.2%
        /// (both BETTER, no guard); keys alone (crowd, 4 intra frames,
        /// qindex 130-240) BD-rate PSNR-YUV -3.3%; -0.6% instructions
        /// (crowd 720p 20 f CBR 400). Periodic-key pulse +0.13 / +0.16 dB.
        /// VP9 Fast keeps 0 (not screened there).
        intra_mode_estimate: 1.0, [0.0, 1.0],
            "intra mode choice measures: 0 the first transform block's prediction SSE, 1 the whole block's prediction SATD (up to 32x32; HEVC's CU-level search)",
        /// 2 is a speed lever (libvpx real-time intra masks, 2026-11-13):
        /// measured in docs/handoff/vp9-faster.md.
        intra_mode_search: 0.0, [0.0, 2.0],
            "intra mode choice: 0 tries all ten modes, 1 the four basic ones and the two directions beside the best, 2 luma DC/V/H and chroma DC only (libvpx real-time)",
        /// 1 by default (the prediction measure decides alone).
        intra_rd_candidates: 1.0, [1.0, 10.0],
            "intra luma modes (the cheapest by the intra_mode_estimate measure, each with its best chroma mode) coded at the largest transform and decided by squared error plus lambda times token and mode bits; 1 takes the cheapest uncoded: higher is slower, better intra",
        /// 1 by default (trials quantise as the final coding does).
        intra_rd_trellis: 1.0, [0.0, 1.0],
            "with vp9.intra_rd_candidates above 1, 1 quantises the mode trials with the trellis where the final coding uses it, 0 with plain quantisation: 0 is faster",
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
        /// blocks: see `DEFAULT_PARAMS`); VP9 Fast keeps 8.
        shave_gap: 0.0, [0.0, 16.0],
            "drop trailing ±1 levels at least this many scan positions past the previous level (SVT shave_coeff; 0 off): lower drops more",
        /// A speed lever for Fast: crowd 720p q230 Jetson -3.9%, x86 +1.6%
        /// (full forward then zeroed there); J_real TRAIN quality +0.2% /
        /// +0.3% (proxy-6, no guards): about 0 at Default's k 0.1.
        partial_forward: 0.0, [0.0, 1.0],
            "32x32 transforms code only their low 16x16 frequencies (an encoder-side forward approximation, SVT's TX shortcut): faster, drops fine detail in large blocks",
        /// A speed lever (HEVC's `pre_transform_zero_scale`, default 1:
        /// exact). Measured for VP9 Faster: docs/handoff/vp9-faster.md.
        pre_transform_zero_scale: 1.0, [1.0, 4.0],
            "inter residual zero-bound multiplier before the transform: 1 exact (only blocks proven to quantise to zero skip it), higher drops near-zero inter residual untransformed",
        /// A speed lever (HEVC's `max_transform_log2`, default 5: 32x32).
        /// Measured for VP9 Faster: docs/handoff/vp9-faster.md.
        inter_max_tx_log2: 5.0, [3.0, 5.0],
            "largest transform of inter blocks (log2: 3 8x8, 4 16x16, 5 32x32): smaller is faster (the 32x32 forward and its scans) and codes fine detail of large blocks dearer",
        /// A speed lever (AV1's `pre_tx_skip_energy`, SVT's light-PD1
        /// energy test; default 0: off). Measured for VP9 Faster:
        /// docs/handoff/vp9-faster.md.
        pre_tx_skip_energy: 0.0, [0.0, 100.0],
            "inter blocks whose luma prediction error is below this energy (100 x SSE < energy x area x qindex) code no luma residual, without transforms: higher skips more",
        /// A speed lever (HEVC's and H.264's `inter_intra_dc_only`,
        /// default 0). Measured for VP9 Faster: docs/handoff/vp9-faster.md.
        /// Mode 2 (VP9 L2 / L3 Medium) keeps directional intra on
        /// key/repair enhancement pictures (`from_base_only` / `intra_only`),
        /// refined blocks, sub-`360p` layers, and blocks touching a frame edge:
        /// `L3T1_KEY` Callgrind `-3.6%` crowd / `-2.1%` bodycam (`-1.9%` 3-clip
        /// mean), 11-clip `[8]` `720p` `+2.0% -> +2.9%` (`[+1.5% -> +1.6%]`),
        /// `L3T1` `+33.0% -> +32.8%` (`[+32.9% -> +32.8%]`).
        inter_intra_dc_only: 0.0, [0.0, 2.0],
            "1 codes intra blocks of inter frames with DC luma and chroma prediction without a mode search; 2 restricts that shortcut to interior blocks of >=360p non-reset inter frames",
        /// A speed lever (libvpx's real-time intra blocks, which code one
        /// transform size; default 3: no cap). Measured for VP9 Faster:
        /// docs/handoff/vp9-faster.md.
        inter_intra_tx_depth: 3.0, [0.0, 3.0],
            "intra blocks of inter frames try at most this many smaller transforms (the lesser of this and intra_tx_depth): lower is faster",
        /// A speed lever (libvpx's real-time skip test, default 0: off).
        /// Alone on VP9 Fast at 1: Jetson -5.0% CPU, J_real quality +2.0%
        /// (pool L1, 2026-11-12); VP9 Faster takes it (handoff).
        nonrd_skip_scale: 0.0, [0.0, 8.0],
            "inter transform blocks whose residual variance is below this x libvpx's AC threshold (ac_q^2 / 64) skip the transform: coding only DC from the residual sum, or nothing when its mean is below the DC threshold too (0 off)",
        /// A speed lever (libvpx's real-time rate, default 0: off). Alone
        /// on VP9 Fast at 4: Jetson -8.0% CPU, J_real quality +3.1% (pool
        /// L1, 2026-11-12); VP9 Faster takes it (handoff).
        /// A speed lever (libvpx's block-level skip test, default 0).
        /// Measured for VP9 Faster: docs/handoff/vp9-faster.md.
        nonrd_block_test: 0.0, [0.0, 1.0],
            "with nonrd_skip_scale, 1 first tests each inter plane block as a whole (libvpx's averaged test): a block with nothing to code skips its transform blocks unvisited",
        residual_bits_per_level: 0.0, [0.0, 16.0],
            "residual RD and transform-size trials price inter levels at this many bits per unit of level magnitude (libvpx block_yrd: 4) instead of their token costs (0)",
        /// A speed lever (libvpx's real-time filter search, default 0).
        /// Measured for VP9 Faster: docs/handoff/vp9-faster.md.
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
/// Compact (16 bytes): trial saves and restores copy every block's.
#[derive(Clone, Copy)]
struct CodedTx {
    plane: PlaneIndex,
    tx: TxSize,
    ty: transform::TxType,
    eob_nonzero: bool,
    /// End of block of the levels (0 without them), found once for the
    /// rate estimate and the token writer.
    eob: u16,
    /// Top-left in 4x4 units of the plane (the coefficient-context grid).
    unit: (u16, u16),
    levels_offset: u32,
}

impl CodedTx {
    /// A block without levels to code.
    fn uncoded(block: &TxBlock, ty: transform::TxType, levels_offset: usize) -> Self {
        Self::new(block, ty, 0, levels_offset)
    }

    /// A block whose levels in scratch have the given end of block.
    fn new(block: &TxBlock, ty: transform::TxType, eob: usize, levels_offset: usize) -> Self {
        let unit = |sample: usize| u16::try_from(sample >> 2).expect("VP9 frames fit 16-bit units");
        Self {
            plane: block.plane,
            tx: block.tx,
            ty,
            eob_nonzero: eob != 0,
            eob: u16::try_from(eob).expect("end of block fits 1024"),
            unit: (unit(block.x), unit(block.y)),
            levels_offset: u32::try_from(levels_offset).expect("per-block levels fit 32 bits"),
        }
    }

    /// Code no levels after all (the block is coded as skipped).
    fn drop_levels(&mut self) {
        self.eob_nonzero = false;
        self.eob = 0;
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
    coefficients: Vec<i16>,
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
    quad_sads: Vec<u16>,
    /// Mode trials quantise without the trellis (`vp9.intra_rd_trellis` 0).
    plain_trial: bool,
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
/// VP9 §8.5.2.3) held for lower-layer (`S0`, and `S1` of three) slots so an
/// upper spatial layer predicts from the exact lower-layer reconstruction via
/// scaled convolution (`vp9_convolve_scaled`) while the engine's pool holds an
/// upscaled proxy for motion estimation.
#[derive(Debug, Clone)]
struct HeldReference {
    picture: Rc<Picture>,
    width: usize,
    height: usize,
    spatial_id: u8,
}

impl HeldReference {
    fn new(picture: Rc<Picture>, width: usize, height: usize, spatial_id: u8) -> Self {
        Self {
            picture,
            width,
            height,
            spatial_id,
        }
    }

    fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

/// The held reference pictures by slot (`None`: the engine's is exact).
type ReferenceStore = nc_encoder::HeldSlots<HeldReference>;

/// Disposable scratch slot refreshed by a hidden spatial layer picture (`S0`,
/// `S1` of three, or `S_top` in K-SVC) whose temporal `refresh` mask is empty
/// (`T2`), so `show_existing_frame` has a slot holding it without overwriting
/// any layer's temporal hierarchy (`0..2*N`): `(2,0)->4`, `(2,1)->5`,
/// `(3,0)->6`, `(3,1)->7`, `(3,2)->7`.
const fn hidden_scratch_slot(num_layers: u8, spatial_id: u8) -> usize {
    let slot = (num_layers as usize) * 2 + (spatial_id as usize);
    if slot < MAX_REFERENCE_SLOTS {
        slot
    } else {
        MAX_REFERENCE_SLOTS - 1
    }
}

/// Multi-layer (`2` or `3`) spatial SVC and K-SVC state on [`Vp9Backend`]
/// (structure after libvpx `vp9/encoder/vp9_svc_layercontext.c`,
/// `INTER_LAYER_PRED_ON` and `INTER_LAYER_PRED_OFF_NONKEY`, BSD-3-Clause,
/// `THIRD_PARTY.md`).
#[derive(Debug)]
struct BackendSpatial {
    num_layers: u8,
    inter_layer: InterLayerMode,
    /// The grids of the layers not currently being coded (the active layer is
    /// [`Vp9Backend::grids`]).
    parked: [Option<LayerGrids>; MAX_SPATIAL_LAYERS],
    active_spatial_id: u8,
    current_layer: SpatialLayerFrame,
    /// The last coded picture in the temporal unit was a key frame: the
    /// decoder adapts the next frame's probabilities at the after-key rate
    /// (`COUNT_SAT_AFTER_KEY`, VP9 §8.4), whatever its layer.
    previous_key: bool,
    /// A base (`S0`) temporal-layer-0 picture has been coded since the last
    /// key (see [`Self::context_use`]).
    base_since_key: bool,
    /// Per-layer flag set when an error-resilient repair on one layer reset all
    /// four contexts (`vp9_setup_past_independence`, VP9 §7.2) on receivers
    /// that decoded it, forcing each diverged layer to emit error-resilient
    /// frames up to and including its next temporal-layer-0 picture.
    context_diverged: [bool; MAX_SPATIAL_LAYERS],
    /// The probabilities the current frame is coded with: its starting
    /// context plus its forward updates (`end_frame`), which adaptation
    /// keeps where it does not adapt.
    coded_context: FrameContext,
    /// The size of the picture in each reference slot (the decoder's
    /// `RefFrameWidth` / `RefFrameHeight`).
    slot_sizes: [Option<(usize, usize)>; MAX_REFERENCE_SLOTS],
    /// Precomputed `EngineMv` scaled-reference phase offsets (`[primary, secondary]`)
    /// for the current frame (`VP9 §8.5.2.3`).
    scaled_mv_offsets: [EngineMv; 2],
}

impl BackendSpatial {
    fn scaled_mv_offset(&self, choice: ReferenceChoice) -> EngineMv {
        self.scaled_mv_offsets[usize::from(choice == ReferenceChoice::Secondary)]
    }

    fn to_vp9_mv(&self, choice: ReferenceChoice, mv: EngineMv) -> Mv {
        let offset = self.scaled_mv_offset(choice);
        to_vp9_mv(EngineMv {
            x: mv.x.saturating_add(offset.x),
            y: mv.y.saturating_add(offset.y),
        })
    }
    /// How the current layer's frame uses the four probability contexts
    /// (`frame_context_idx` $\in \{0, 1, 2\}$ = `spatial_id`, VP9 §6.2 / §7.2):
    /// - temporal layers above `0` read their layer's context without
    ///   refreshing it (a lower-frame-rate receiver drops them);
    /// - `S0`'s first `T0` picture after a key does not refresh: whether it
    ///   adapts at `COUNT_SAT_AFTER_KEY` depends on whether the receiver
    ///   decoded the key unit's upper layers;
    /// - an enhancement repair (`from_base_only` or `intra_only` outside a key
    ///   unit) is error resilient, resetting all four contexts on receivers
    ///   that decode it, so every other layer (and the repaired layer if
    ///   repaired above `T0`) emits error-resilient frames through its next
    ///   `T0` picture.
    ///
    /// Every spatial layer frame has `use_prev_frame_mvs = 0` (`VP9 §7.2`):
    /// lower layers are always hidden (`show_frame = 0`), and in K-SVC the top
    /// layer is hidden too (displayed by `show_existing_frame`, which leaves
    /// `last_show_frame = 0`) so stripping lower layers on non-key units keeps
    /// `use_prev_frame_mvs = 0` on every receiver.
    fn context_use(&self, key: bool) -> ContextUse {
        let layer = &self.current_layer;
        let s = usize::from(layer.spatial_id);
        if key {
            return ContextUse {
                error_resilient: false,
                index: 0,
                refresh: true,
            };
        }
        let repair = s > 0 && !layer.key_unit && (layer.from_base_only || layer.intra_only);
        if self.context_diverged[s] || repair {
            return ContextUse {
                error_resilient: true,
                index: 0,
                refresh: false,
            };
        }
        ContextUse {
            error_resilient: false,
            index: layer.spatial_id,
            refresh: layer.temporal_id == 0 && (s > 0 || self.base_since_key),
        }
    }

    /// Advance the cross-layer state past a committed frame.
    fn committed(&mut self, header: &FrameHeader) {
        let key = header.key;
        for (slot, size) in self.slot_sizes.iter_mut().enumerate() {
            if header.refresh & (1 << slot) != 0 {
                *size = Some((header.width, header.height));
            }
        }
        let s = usize::from(self.current_layer.spatial_id);
        let lowest = self.current_layer.temporal_id == 0;
        let repair = s > 0
            && !self.current_layer.key_unit
            && (self.current_layer.from_base_only || self.current_layer.intra_only);
        self.previous_key = key;
        if key {
            self.base_since_key = false;
            self.context_diverged = [false; MAX_SPATIAL_LAYERS];
            return;
        }
        if s == 0 {
            self.base_since_key |= lowest;
        }
        if repair {
            for (id, div) in self
                .context_diverged
                .iter_mut()
                .enumerate()
                .take(usize::from(self.num_layers))
            {
                *div = (id != s) || !lowest;
            }
        } else if header.error_resilient {
            self.context_diverged[s] &= !lowest;
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

/// The matching, unscaled luma prediction already filtered during ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RankedLuma {
    origin: (usize, usize),
    side: usize,
    reference: ReferenceChoice,
    mv: Mv,
    filter: nc_dsp::Vp9Filter,
}

impl RankedLuma {
    fn for_block(
        position: &BlockPosition,
        reference: ReferenceChoice,
        mv: Mv,
        filter: nc_dsp::Vp9Filter,
    ) -> Self {
        Self {
            origin: (position.col * MI_SIZE, position.row * MI_SIZE),
            side: position.size * MI_SIZE,
            reference,
            mv,
            filter,
        }
    }
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
    spatial: Option<Box<BackendSpatial>>,
    /// Active reference choice (`Primary` = `LAST_FRAME`, `Secondary` =
    /// `GOLDEN_FRAME`) during mode evaluation and block coding.
    active_ref: ReferenceChoice,
    params: Vp9Params,
    /// The staged intra decision's first cut (`decision.intra_stage_*`).
    intra_stage: nc_encoder::IntraStage,
    /// Shared quantizer gates (`gate.partial_transform_min_quantizer`).
    gates: nc_encoder::GateParams,
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
    /// `EncoderConfig::key_qp_offset` in qindex steps: replaces the
    /// `vp9.keyframe_step_permille` step factor for fixed-QP key frames.
    key_qindex_offset: Option<i32>,
    header: FrameHeader,
    /// What every key frame signals (`config.signal.colour`).
    colour: ColourSpace,
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
    counts: Box<Counts>,
    writer: TileRecorder,
    /// Reference vectors of the block being decided, cached per position and
    /// reference (`(row, col, log2_size, secondary_ref)`).
    refs: Option<BlockRefs>,
    ranked_luma: Option<RankedLuma>,
    ranked_luma_samples: Vec<u8>,
    scratch: Scratch,
    buffers: Box<recon::ResidualBuffers>,
    /// Inter prediction scratch.
    inter: recon::InterScratch,
    /// Kernel dispatch (forward transforms).
    dsp: nc_dsp::Dsp,
    /// Random syntax for the stress test (`debug.random_decisions`) only.
    random: Option<RandomSyntax>,
    /// The final decision stage's pricing scratch (`final_cost.rs`).
    final_scratch: final_cost::FinalScratch,
    /// `quant.trellis`: trellis quantisation of every transform block.
    trellis: Option<Vp9Trellis>,
    /// `quant.rd_trim`: rate-aware trimming of the plain quantiser's luma
    /// levels, in every residual coding, while the trellis is off.
    rd_trim: Option<nc_encoder::RateTrim>,
    /// The RD partition search's open node trials (`node_trial.rs`).
    node_trials: node_trial::NodeTrials,
    /// Whether the block last coded is an inter block without residual
    /// predicted from a candidate vector (`CodecBackend::last_cu_skipped`).
    last_block_skipped: bool,
    /// The block of a repeat frame `try_static_skip` accepted
    /// (`StaticSkipTest::repeat`), coded next with no residual.
    repeat_block: Option<(usize, usize)>,
    /// The engine's loop-filter decision for this frame (`filter_effort.rs`;
    /// VP9 reads whether an inter frame is deblocked).
    filter_effort: FilterEffort,
}

/// log2 of the 4x4-sample units of the entropy contexts.
const TX_UNIT_LOG2: usize = 2;

/// The registered trellis controls (`nc_encoder::QuantParams`) and the
/// search's buffers.
#[derive(Debug)]
struct Vp9Trellis {
    settings: nc_encoder::TrellisSettings,
    search: coeff::trellis::TrellisQuantizer,
}

impl Vp9Trellis {
    /// Trellis-quantise the levels of `tx_block` (transform type `ty` of
    /// an inter or intra block) from its `coefficients`, starting from the
    /// plain quantiser's `levels` (some nonzero) and priced with the
    /// frame's probabilities and the neighbours' nonzero contexts as they
    /// stand. Returns the new end of block.
    fn quantize(
        &mut self,
        dsp: &nc_dsp::Dsp,
        (fc, contexts, tx_block, inter): (&FrameContext, &EntropyContexts, &TxBlock, bool),
        ty: transform::TxType,
        (lambda, coefficients, steps): (f32, &[i16], transform::Steps),
        levels: (&mut [i32], Option<usize>),
    ) -> usize {
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
            dsp,
            &mut rates,
            coefficients,
            steps,
            (lambda, self.settings.max_states),
            levels,
        )
    }

    fn from_params(quant: &QuantParams) -> Option<Self> {
        nc_encoder::TrellisSettings::from_params(quant).map(|settings| Self {
            settings,
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
    /// A vector component lies strictly within ±2048 samples (libvpx
    /// `is_mv_valid`: `MV_LOW` < component < `MV_UPP`, 14 bits of eighths; a
    /// decoder marks the frame corrupt otherwise).
    const VECTOR_REACH: usize = 2047;
    const RECODE: bool = true;
    // No intra refresh: backward-adapted frame contexts (VP9 §8.4.2) and
    // `use_prev_frame_mvs` carry a lost frame's state into every later
    // frame until a key, so a sweep cannot repair the picture (measured: a
    // dropped frame leaves vpxdec refusing every later frame and FFmpeg at
    // 12-15 dB for the rest of the stream).
    const INTRA_REFRESH: bool = false;

    // Scaling λ alone reaches low targets here: crowd_mot20_06 CBR 150 kbps,
    // 150 f, headroom 2 octaves: 309 -> 159 kbps.
    const SUPPORTS_LAMBDA_HEADROOM: bool = true;
    const NODE_TRIALS: bool = true;

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
    /// +0.01/+0.04/-0.02, homes390 300 kbps -0.05/-0.08/-0.10. VP9 Fast keeps
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
    /// 720p 20 f CBR 400, callgrind): +1.7%, +0.1%, +0.8%. VP9 Fast keeps
    /// the engine values.
    const DEFAULT_PARAMS: &'static [(&'static str, f32)] = &[
        // Periodic keys at the engine's `rate.key_cap_buffer` 1.0, the first key at 0.8
        // (`rate.key_cap_buffer_first`): cap 1.0 on the first key too was ejected on L3 at medium
        // and faster (+0.10 / +0.23 %, the larger first key at 150 f with one key). nceval.train at
        // both key shares (base 29f3c5a22, front k): L3 byte-identical (one-key streams keep their
        // first key), final check BETTER at 1/150 and 1/300: medium -0.99 / -0.38 %, faster -1.09 /
        // -0.39 %. VP9 fast sets the first key at 1.0 (passed).
        ("rate.key_cap_buffer_first", 0.8),
        // Key intra modes by coded cost among the SATD-best three, trials
        // without the trellis: all-intra PSNR-YUV BD -3.00 % at +43 % key
        // instructions (CBR 150 f +1.7-1.9 %). Pool (k 0.1, dCPU 1.8, draw
        // 2026-11-19) L3 J -0.44 % [-0.72, -0.17] n 108; final-check -0.70 %
        // [-0.99, -0.42], no guard (docs/analysis/rd-mode-audit-2026-11-19.md).
        ("vp9.intra_rd_candidates", 3.0),
        ("vp9.intra_rd_trellis", 0.0),
        // Tuned jointly (eval/nceval/tune_pool.py, study
        // vp9-default-eq2-2026-11-11, equal x86 CPU): planner lambda 2 ->
        // 3.16 (fewer splits). Study clips J -1.97%, L3 of the study's draw
        // -0.87% [-1.31, -0.43] (overfit gap +1.09); re-checked at head
        // cd9ea802 on the study's rotation: disjoint L3 (79 clips,
        // single-point) J -0.74% [-1.18, -0.30], L2 (ladder) -1.21% [-1.58,
        // -0.83], no guard. CPU x86 -1.5%, Jetson cycles -1.1%. VP9 Fast pins
        // the previous value. (Re-tuned to 2.38 on 2026-11-14: the CPU cut
        // at the end of this list.)
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
        // The target as a ceiling over every key period (`rate.period_ceiling`,
        // `period_ledger.rs`; owner, 2026-11-14: steady sizes for still scenes).
        // 600-frame long-window L3 (keys every 10 s, 31 points, draw 2026-11-14):
        // J -1.33% [-2.53, -0.32] BETTER, no guard; static -5.5%, crowd +0.2%,
        // pan and lowlight unchanged; achieved/target max 1.007 -> 1.000; re-codes
        // on the 22 static and crowd points 5.5 -> 10.8 per 1000 frames.
        // Inert within one key period. The other VP9 presets pin it off.
        ("rate.period_ceiling", 1.0),
        // `rate.still_level_damping` (engine 0.5) was screened on H.264 Default
        // only (2026-11-18); pinned off here until this codec is screened.
        ("rate.still_level_damping", 0.0),
        // A fixed camera's key coded finer (`rate.key_still_boost`, QP units
        // times how still the scene is): its later frames inherit it. Pool levels,
        // draw 2026-11-14: L1 -0.16% [-0.62, +0.10]; L2 -1.69% [-2.20, -1.16];
        // 600-frame L3 (keys every 10 s) -1.84% [-2.42, -1.30], no guard; static
        // -6.8%, crowd -0.6%, pan and lowlight unchanged; periodic-key pulse
        // -0.26 dB; achieved/target max 1.000 -> 1.000; re-codes on the 22 static
        // and crowd points 10.8 -> 15.7 per 1000 frames.
        // The other VP9 presets pin it off.
        ("rate.key_still_boost", 6.0),
        // The unified structure's reference plan (`structure.unified`,
        // `nc-encoder` `second_reference.rs` `TierPlan`): high-quality frames
        // (key, cadence base) also predict from the high-quality frame before
        // them where the cadence strength is 0.75 or more. Build 78191a7a,
        // pool levels of draw 2026-11-15: L1 -1.38% [-2.14, -0.73]; L2
        // -0.74% [-0.97, -0.50]; L3 (ladder, 126 clips) J -0.67% [-0.88,
        // -0.45], no guard; Jetson cycles +0.8% (lowlight +1.3, crowd +1.1,
        // bodycam 0.0), instructions +0.6%. 600-frame final-check (27 clips,
        // keys every 10 s): J -0.58% [-0.78, -0.38], no guard. Landing conformance with it on:
        // every stream bit-exact in vpxdec, ffmpeg and ours. The other VP9
        // presets pin it off.
        ("structure.unified", 1.0),
        // The QP cadence at least as strong as the frames between measurably
        // inherit the high-quality frame (`structure.inherited_depth`,
        // `nc-encoder` `inheritance.rs`), beside the static share, with
        // still-block refinements priced against the inter level
        // (`persist.reuse_level_price`, `reuse.rs`): the frame after a key
        // is a cadence leaf, and with the stronger cadence alone the starved
        // key's still half recovered 0.85 dB short (`static_recovery`: 38.6
        // against 39.4 dB; with both 43.9 against 40.0). Build 51b0ec30,
        // pool levels of draw 2026-11-15, both together: L1 -1.62% [-2.78,
        // -0.68]; L2 -1.57% [-2.61, -0.71]; L3 (single point, 126 clips) J
        // -1.21% [-2.03, -0.53], no guard (pan -3.5%, crowd -0.9%, lowlight
        // -0.6%, static -0.4%, bodycam +0.9%, texture -3.6% on 4 clips,
        // drone -0.5% on 3); 600-frame final (27 clips, keys every 10 s) J
        // -0.56% [-0.95, -0.16], no guard (lowlight -1.3%, crowd -0.7%,
        // static +0.2%). Jetson cycles +0.5% (instructions +0.7%). The other
        // VP9 presets pin both off.
        ("structure.inherited_depth", 1.0),
        ("persist.reuse_level_price", 1.0),
        ("decision.codec_subpel_seed_lambdas", 0.0),
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
        // plane error per frame). VP9 Fast, Faster and Veryfast pin 0 (not
        // screened there).
        ("vp9.chroma_delta_gap_slope", 3.0),
        // VP9 spatial SVC's enhancement-layer intra gate (the engine's
        // `spatial.enhancement_intra_gate`): its measured results include it
        // (docs/handoff/spatial-svc.md). Single-layer streams never read it.
        ("spatial.enhancement_intra_gate", 1.0),
        // Jointly re-tuned as a CPU cut at equal quality (2026-11-14; study
        // vp9-default-speed-2026-11-13, trial #23, mode-decision / quantiser /
        // filter levers): P-frame intra DC only, the transform search from 4
        // levels, the inter rounding divisor 4 -> 3.78, a lower
        // planner lambda (3.16 -> 2.38), a weaker loop filter (0.63), more
        // chroma (delta q -14) and the values below. Head 7c0d98dc, pool levels
        // of rotation 2026-11-14: disjoint L3 (79 clips, single-point) J -0.02%
        // [-0.94, +0.75], pooled -0.12%, no guard; x86 instructions -12.2%,
        // Jetson cycles -8.0%. The other VP9 presets pin the previous values.
        ("vp9.tx_search_min_levels", 4.0),
        ("vp9.inter_rounding_divisor", 3.783),
        ("vp9.inter_tx_split_spread", 1.047),
        ("vp9.lambda_per_step_squared", 0.1006),
        ("planner.lambda_scale", 2.379),
        // The zero-trial split selector at half its fitted shift (engine
        // `split_selector.rs`, docs/handoff/planner-selector.md). nceval.train
        // against 9a0ff2c7c (draw 2026-11-21, k 0.1, cycles zen5 / graviton3
        // / graviton4 +2.1 / +1.4 / +1.3%): L3 J -0.55% [-0.78, -0.32]
        // BETTER; final-check (600 f) -0.86% [-1.11, -0.69], its worst-clip
        // periodic-key pulse guard tripped (+0.58 dB > +0.50), adopted on
        // judgement: that clip (meva_bus_detail, static) codes the same keys
        // to 0.01 dB in both arms while its P frames gain +0.06..+1.0 dB. The
        // other VP9 presets pin it off (Fast L1 +0.06, Faster +0.81,
        // Veryfast +1.53 with their CPU at k 0.6; Slower +0.1 fixed QP).
        ("planner.split_selector", 1.0),
        ("planner.split_selector_scale", 0.5),
        ("quant.trellis_lambda_scale", 1.139),
        ("decision.intra_gate_variance", 6.857),
        ("vp9.filter_level_scale", 0.6274),
        ("vp9.chroma_delta_q", -14.0),
        ("vp9.inter_intra_dc_only", 1.0),
        // Exact candidate ranking on VP9's own filter, and the winner walk
        // over the winners whose residual is at least 0.25 lambda per sample
        // (H.264's gate). Fixed QP (23 clips, draw fixedqp-2026-11-20) J -2.1%
        // [-3.0, -1.2] at +13.6% x86 P instructions; whole encode (CBR 600,
        // the three chart timing clips, 150 f, callgrind) +9.9%. Pool k 0.1
        // (draw 2026-11-20): L2 (ladder) -0.78% [-1.23, -0.26], L3 (108 clips,
        // single-point) -1.24% [-1.69, -0.90], no guard; final-check (33
        // clips, 600 f, ladder) -1.53% [-1.78, -1.24]. Its periodic-key pulse
        // guard (+0.57 dB on the worst clip, 2 s keys) comes from better P
        // frames: the absolute key PSNR-Y moves -0.04..+0.13 dB on all six
        // clips, the eight frames before each key +0.0..+0.16 dB. The other
        // VP9 presets pin it off (not screened there).
        ("vp9.exact_ranking", 1.0),
        ("decision.winner_walk_steps", 3.0),
        ("decision.winner_walk_min_lambdas", 0.25),
        // The winner walk also walks a winner at the skip vector (NEARESTMV,
        // `decision.winner_walk_keeps_skip` 0). Codec reason: H.264 P_Skip and
        // HEVC cu_skip_flag tie residual-free coding to the inherited vector,
        // so moving it off costs the skip syntax; in VP9 any vector may skip
        // its residual and a moved NEARESTMV costs only its vector bits.
        // Kept unwalked, fixed QP (draw fixedqp-2026-11-20, quick): VP9
        // Default J +0.3 % [+0.1, +0.4], n 22.
        ("decision.winner_walk_keeps_skip", 0.0),
        // Periodic keys planned 2 QP units finer than the inter level, adopted
        // with the key-frame RD partition search (3ebca4466). The search itself
        // is off at medium (reverted 2026-11-22): on the published front's chart
        // timing (150 f, one key) the head cost Zen 5 cycles AV1 +5.3 %, AV2
        // +9.1 %, VVC +5.9 % against the same head with the search, these finer
        // keys, the AV1 / AV2 key trellis and H.264's key Intra_8x8 off, which
        // outweighs its J (L3 -0.33 to -0.55 %). The finer key adds no search
        // and stays (coordinator decision); a screen of it alone is queued.
        ("rate.key_level_finer", 2.0),
    ];

    // The fixed-QP profile (`--qp`): every high-quality frame of the unified
    // plan carries the second reference (no cadence-strength gate), with the
    // full mode decision against it, a static reference pattern as the
    // standards' test conditions use. Fixed QP, matched QP (nceval.fixedqp,
    // 16 pool clips): draw fixedqp-2026-11-15 J -1.5% [-2.4, -0.7], draw
    // fixedqp-2026-11-15-second -1.4% [-2.6, -0.3]; each half alone -0.2 to
    // -1.0%. docs/analysis/fixed-qp-gap-2026-11-14.md "fixed-QP profile".
    // The other VP9 presets pin the plan off, so the profile is inert there.
    // It also codes a period-8 QP cadence at base boost 4 (leaf 1, the engine's), no
    // start-up cadence: medium, 600 f fixed QP, nceval.fixedqp draw
    // fixedqp-2026-11-22: AV1 adopt draw J -1.34 % [-1.67, -0.99] n 48, VP9
    // -1.53 % [-1.74, -0.76] n 22, AV2 -1.31 % [-1.62, -0.65] n 24 (quick);
    // AV1 at 150 f -0.21 %. Slow / slower (quick, 600 f, base 57f355565):
    // AV1 -0.65 / -1.84 %, VP9 -0.79 / -1.46 %, AV2 -1.10 / -1.74 %, every
    // interval below zero. The fast rungs and the SVC files, not screened, pin
    // the old cadence. docs/handoff/vvc.md "Fixed-QP cascade sweep" (VVC, HEVC
    // and H.264 did not hold).
    const FIXED_QP_PARAMS: &'static [(&'static str, f32)] = &[
        ("structure.secondary_min_depth", 0.0),
        ("refs.second_full", 1.0),
        ("rate.cadence_period", 8.0),
        ("rate.cadence_base_boost", 4.0),
        ("rate.cadence_startup_period", 0.0),
    ];

    fn new(config: &EncoderConfig) -> Result<Self, ConfigError> {
        if config.codec != Codec::Vp9 {
            return Err(ConfigError::Unsupported("VP9 backend requires VP9 codec"));
        }
        let colour = ColourSpace::of(config.signal.colour).ok_or(ConfigError::Unsupported(
            "VP9 signals only the BT.601 and BT.709 colour matrices",
        ))?;
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
        let qindex = config.quantizer as u8;
        Ok(Self {
            lambda_scale: 1.0,
            grids: LayerGrids::new(config.width, config.height),
            store: ReferenceStore::default(),
            spatial: None,
            active_ref: ReferenceChoice::Primary,
            params: Vp9Params::from_overrides(&config.params),
            intra_stage: nc_encoder::IntraStage::from_overrides(&config.params),
            gates: nc_encoder::GateParams::from_overrides(&config.params),
            block_qp_enabled: false,
            block_qp: None,
            coding_q_idx: qindex,
            quantizer_mode: QuantizerMode::Fixed,
            key_qindex_offset: config.key_qp_offset,
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
                colour,
            },
            colour,
            lossless: qindex == 0,
            chroma_policy: nc_encoder::chroma_policy::ChromaPolicy::from_params(&config.params),
            context: FrameContext::default(),
            mode_costs: ModeCosts::new(&FrameContext::default()),
            counts: Box::default(),
            writer: TileRecorder::default(),
            refs: None,
            ranked_luma: None,
            ranked_luma_samples: Vec::new(),
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
                plain_trial: false,
            },
            random: None,
            final_scratch: final_cost::FinalScratch::default(),
            trellis: Vp9Trellis::from_params(&QuantParams::from_overrides(&config.params)),
            rd_trim: nc_encoder::RateTrim::from_overrides(&config.params),
            buffers: recon::ResidualBuffers::new(&nc_dsp::Dsp::new()),
            inter: recon::InterScratch::new(),
            dsp: nc_dsp::Dsp::new(),
            node_trials: node_trial::NodeTrials::default(),
            last_block_skipped: false,
            repeat_block: None,
            filter_effort: FilterEffort::FULL,
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
        true
    }

    fn select_reference(&mut self, _block: &Block, reference: ReferenceChoice) {
        self.active_ref = reference;
    }

    fn supports_spatial_layers(&self) -> bool {
        true
    }

    fn max_spatial_layers(&self) -> u8 {
        MAX_SPATIAL_LAYERS as u8
    }

    fn configure_spatial(&mut self, spatial: &SpatialConfig) -> Result<(), ConfigError> {
        if spatial.layer_sizes.len() != usize::from(spatial.layers)
            || spatial.layer_sizes.first() != Some(&spatial.base_size)
            || spatial.layer_sizes.last() != Some(&spatial.full_size)
        {
            return Err(ConfigError::SpatialScalability(
                "VP9 spatial layer dimensions do not match layer_sizes",
            ));
        }
        for pair in spatial.layer_sizes.windows(2) {
            let (low, high) = (pair[0], pair[1]);
            if !nc_encoder::scaled_reference_allowed(low, high) {
                return Err(ConfigError::SpatialScalability(
                    "VP9 spatial layer scale outside §7.2 reference limits (base at least 1/16 of the upper layer)",
                ));
            }
        }
        let top_id = spatial.layers - 1;
        let mut parked: [Option<LayerGrids>; MAX_SPATIAL_LAYERS] = std::array::from_fn(|_| None);
        for (s, &(w, h)) in spatial.layer_sizes[..usize::from(top_id)]
            .iter()
            .enumerate()
        {
            parked[s] = Some(LayerGrids::new(w, h));
        }
        self.grids = LayerGrids::new(spatial.full_size.0, spatial.full_size.1);
        self.spatial = Some(Box::new(BackendSpatial {
            num_layers: spatial.layers,
            inter_layer: spatial.inter_layer,
            parked,
            active_spatial_id: top_id,
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
            previous_key: false,
            base_since_key: false,
            context_diverged: [false; MAX_SPATIAL_LAYERS],
            coded_context: FrameContext::default(),
            slot_sizes: [None; MAX_REFERENCE_SLOTS],
            scaled_mv_offsets: [EngineMv { x: 0, y: 0 }; 2],
        }));
        Ok(())
    }

    fn prepare_spatial_layer(&mut self, layer: SpatialLayerFrame) {
        let Some(sp) = self.spatial.as_mut() else {
            return;
        };
        if sp.active_spatial_id != layer.spatial_id {
            let next = sp.parked[usize::from(layer.spatial_id)]
                .take()
                .expect("configured spatial layer grids");
            let prev = std::mem::replace(&mut self.grids, next);
            sp.parked[usize::from(sp.active_spatial_id)] = Some(prev);
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
        _refresh: SlotMask,
        refresh_all: bool,
    ) {
        let keep = self
            .spatial
            .as_ref()
            .is_some_and(|sp| spatial_id + 1 < sp.num_layers);
        let held =
            keep.then(|| HeldReference::new(Rc::new(recon.clone()), size.0, size.1, spatial_id));
        self.store
            .commit(refresh_all, SlotMask(self.header.refresh), held.as_ref());
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

    /// The frame's filter, or regular where blocks switch filters (the
    /// filter most blocks code), with `vp9.exact_ranking`.
    fn luma_filter(&self) -> Option<nc_dsp::Filter> {
        if self.params.exact_ranking < 0.5 {
            return None;
        }
        self.exact_luma_filter()
    }

    fn exact_luma_filter(&self) -> Option<nc_dsp::Filter> {
        Some(match self.header.interp_filter {
            Some(nc_dsp::Vp9Filter::Smooth) => nc_dsp::Filter::Vp9Smooth,
            Some(nc_dsp::Vp9Filter::Sharp) => nc_dsp::Filter::Vp9Sharp,
            // The encoder never codes bilinear frames (`INTERP_FILTERS`).
            Some(nc_dsp::Vp9Filter::Regular | nc_dsp::Vp9Filter::Bilinear) | None => {
                nc_dsp::Filter::Vp9
            }
        })
    }

    const CACHE_UNWEIGHTED_LUMA: bool = true;

    fn cache_exact_luma(
        &mut self,
        block: &Block,
        mv: EngineMv,
        filter: nc_dsp::Filter,
        prediction: &[u8],
        _stats: nc_dsp::Variance,
    ) {
        // Spatial references can be resampled proxies, and near the edge
        // the engine and the bitstream predictor may extend different crops.
        if self.spatial.is_some() || self.random.is_some() {
            return;
        }
        let filter = match filter {
            nc_dsp::Filter::Vp9 => nc_dsp::Vp9Filter::Regular,
            nc_dsp::Filter::Vp9Smooth => nc_dsp::Vp9Filter::Smooth,
            nc_dsp::Filter::Vp9Sharp => nc_dsp::Vp9Filter::Sharp,
            _ => return,
        };
        let (dx, dy) = mv.full_pel();
        let before = nc_dsp::VP9_TAPS_BEFORE as isize;
        let width = (block.size() + nc_dsp::VP9_TAPS - 1) as isize;
        let x = block.x as isize + dx as isize - before;
        let y = block.y as isize + dy as isize - before;
        if x < 0
            || y < 0
            || x + width > self.grids.width as isize
            || y + width > self.grids.height as isize
        {
            return;
        }
        self.ranked_luma = Some(RankedLuma::for_block(
            &self.position(block),
            self.active_ref,
            to_vp9_mv(mv),
            filter,
        ));
        self.ranked_luma_samples.clear();
        self.ranked_luma_samples.extend_from_slice(prediction);
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
        self.intra_stage = nc_encoder::IntraStage::from_overrides(params);
        self.gates = nc_encoder::GateParams::from_overrides(params);
    }

    fn rate_model(&self) -> nc_encoder::RateModel {
        RATE_MODEL
    }

    fn rate_control_quantizer_step(&self, quantizer: u32) -> f64 {
        f64::from(AC_QUANT[quantizer.min(QINDEX_MAX) as usize])
    }

    /// Frames whose blocks may carry offsets then code segment ids,
    /// segment 0 at the frame quantizer (`nc_encoder::segment_quantizers`). Other frames
    /// leave segmentation to `vp9.skip_filter_off`: segment ids on every
    /// block of a frame that refines a few cost a 4-clip TRAIN proxy +2.9%
    /// `J_real` (`static_cam22` key 48.7 KB against 46.0 KB).
    fn enable_block_qp(&mut self) {
        self.block_qp_enabled = true;
    }

    fn block_qp_offset(&mut self, requested: i16) -> i16 {
        if self.random.is_none()
            && self
                .spatial
                .as_ref()
                .is_some_and(|sp| sp.num_layers >= 3 && sp.current_layer.spatial_id == 0)
            && requested.unsigned_abs() < THREE_LAYER_S0_MIN_SEGMENT_OFFSET
        {
            return 0;
        }
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
        // Priced as the leaf is coded (`trim_levels`'s planned dead zone).
        let min_level = transform::planned_min_level(Rounding::Intra, self.lambda_scale);
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
                let nonzero = transform::quantize_planned(
                    &self.dsp,
                    plane_tx,
                    self.lossless,
                    coefficients,
                    steps,
                    (Rounding::Intra, min_level),
                    levels,
                );
                let levels = if nonzero {
                    &*levels
                } else {
                    &transform::ZERO_LEVELS[..area]
                };
                let eob = if nonzero {
                    coeff::end_of_block(&self.dsp, levels, plane_tx, model.ty)
                } else {
                    0
                };
                bits += coeff::cost((levels, eob), model, probs, 0);
                if transform_domain {
                    sse += transform::coefficient_sse(
                        plane_tx,
                        coefficients,
                        nonzero.then_some(levels),
                        steps,
                    );
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

    /// `NEARESTMV` (`final_cost.rs`): the winner walk keeps it, and the
    /// final decision stage sets it against the ranking's winner.
    fn skip_vector(&mut self, block: &Block) -> Option<EngineMv> {
        Some(self.nearest_vector(block))
    }

    /// VP9 takes no static skips (the default would offer the skip vector).
    fn static_skip_vector(&mut self, _block: &Block) -> Option<EngineMv> {
        None
    }

    /// Only a repeat frame's blocks (`StaticSkipTest::repeat`): coded at
    /// the zero vector with no residual, a skip block (VP9 §6.4.6).
    fn try_static_skip(
        &mut self,
        block: &Block,
        _source: &Picture,
        reference: &Picture,
        recon: &mut Picture,
        test: StaticSkipTest,
    ) -> bool {
        if !test.repeat || self.random.is_some() {
            return false;
        }
        copy_zero_vector_prediction(&self.dsp, block, reference, recon);
        self.repeat_block = Some((block.x, block.y));
        true
    }

    fn set_filter_effort(&mut self, effort: FilterEffort) {
        self.filter_effort = effort;
    }

    fn final_intra_screen(
        &mut self,
        source: &Picture,
        recon: &Picture,
        block: &Block,
    ) -> Option<(f32, f32)> {
        self.prices_final_candidates()
            .then(|| self.screen_intra(source, recon, block))
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
        self.random.is_none() && !self.lossless
    }

    fn mv_candidates(&mut self, block: &Block, out: &mut Vec<EngineMv>) {
        let refs = self.refs_for(block);
        for mv in [refs.nearest, refs.near] {
            let engine = self.engine_candidate(mv);
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
                let vp9_mv = self.coded_vector(&cached.refs, mv);
                cached.reference_bits
                    + cheapest_inter_mode(
                        &self.mode_costs,
                        &cached.refs,
                        vp9_mv,
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
        // A lossless (qindex 0) stream keeps lossless keys, and a lossy
        // key stays lossy (qindex 1 and up).
        let fixed_key = plan.frame_type == FrameType::Key
            && self.quantizer_mode == QuantizerMode::Fixed
            && base != 0;
        let qindex = match self.key_qindex_offset {
            Some(offset) if fixed_key => {
                nc_encoder::offset_key_quantizer(base, offset, 1, QINDEX_MAX)
            }
            _ if fixed_key => scaled_qindex(
                base,
                f64::from(self.params.keyframe_step_permille) / KEY_STEP_SCALE_DENOMINATOR,
            ),
            _ => base,
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
            let slot = self.spatial.as_ref().map_or(0, |sp| {
                if sp.current_layer.key_unit {
                    0
                } else {
                    layer_slot_offset(sp.active_spatial_id) as u8
                }
            });
            (slot, slot)
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
        if self.spatial.is_some() {
            let p_off = self.slot_scaled_mv_offset(usize::from(primary_slot));
            let s_off = self.slot_scaled_mv_offset(usize::from(secondary_slot));
            if let Some(sp) = self.spatial.as_mut() {
                sp.scaled_mv_offsets = [p_off, s_off];
            }
        }
        self.grids.contexts.reset_frame();
        self.writer = TileRecorder::default();
        self.refs = None;
        self.ranked_luma = None;
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
        *self.counts = Counts {
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
        let intra_modes = inter
            .is_none()
            .then(|| self.intra_modes(block, tx, source, recon));
        let repeat = self.repeat_block.take() == Some((block.x, block.y));
        let drop_residual = repeat
            || self
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
        // A skip CU: no residual, predicted from a candidate vector
        // (`NEARESTMV`, `NEARMV`, `ZEROMV`; VP9 §6.4.16).
        self.last_block_skipped =
            skip && info.inter.is_some_and(|(mode, _)| mode != InterMode::New);
        self.refs = None;
        self.ranked_luma = None;
        self.active_ref = ReferenceChoice::Primary;
        self.coding_q_idx = self.header.base_q_idx;
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

    fn undo_node_trial(&mut self, recon: &mut Picture) {
        // The leaf and quad require just one undo, which keeps the first
        // candidate's entropy state in the buffer that held the starting state.
        const { assert!(!Self::RECTANGULAR_LEAVES) };
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
        self.last_block_skipped
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
            self.header
                .segmentation
                .enabled
                .then_some(self.header.segmentation.tree_probs),
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
                for grids in sp.parked.iter_mut().flatten() {
                    grids.context = FrameContext::default();
                }
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
        // The finished grid becomes the last frame's by a swap, not a copy
        // (a 720p grid is 345 KB): the grid then holds an older frame's
        // entries, which are stale either way ([`ModeInfoGrid`]: readers
        // only look at blocks already coded in the current frame).
        match &mut self.grids.previous {
            Some(previous) if previous.same_shape(&self.grids.grid) => {
                std::mem::swap(previous, &mut self.grids.grid);
            }
            previous => *previous = Some(self.grids.grid.clone()),
        }
    }
}

impl Vp9Backend {
    /// Normative reference picture (when held in `self.store` for a lower
    /// spatial layer's slot referenced by an upper layer) and
    /// `(RefFrameWidth, RefFrameHeight)` for the active reference (`LAST_FRAME`
    /// or `GOLDEN_FRAME`).
    fn active_reference(&self) -> (Option<Rc<Picture>>, (usize, usize)) {
        let own_size = (self.grids.width, self.grids.height);
        if let Some(sp) = &self.spatial
            && sp.active_spatial_id > 0
        {
            let slot = if self.active_ref == ReferenceChoice::Secondary {
                usize::from(self.header.ref_slots[1])
            } else {
                usize::from(self.header.ref_slots[0])
            };
            if let Some(held) = self.store.held(slot)
                && held.spatial_id != sp.active_spatial_id
            {
                return (Some(Rc::clone(&held.picture)), held.size());
            }
        }
        (None, own_size)
    }

    /// Quarter-sample (`EngineMv`) offset aligning the engine's centre-mapped
    /// resampling proxy (`PlaneResampler`, `in = (out + 0.5) * r - 0.5`) with
    /// VP9's top-left-mapped scaled reference (`VP9 §8.5.2.3`,
    /// `in = (out + mv / 8) * r`; `libvpx` `vp9_pickmode.c` `(-4, -4)` at
    /// `2:1`). Cached per frame in [`BackendSpatial::scaled_mv_offsets`] so
    /// single-layer `mode_bits` never runs it.
    /// Measured (11 VAL clips `[8 chart clips]`): `VP9 SVC L3T1` `720p`
    /// `+36.2% -> +34.7%` (`[+37.6% -> +35.3%]`), `360p` `+25.2% -> +24.2%`;
    /// `VP9 SVC L2T1` `720p` `+24.2% -> +23.0%` (`[+25.3% -> +23.8%]`).
    fn slot_scaled_mv_offset(&self, slot: usize) -> EngineMv {
        let Some(sp) = &self.spatial else {
            return EngineMv { x: 0, y: 0 };
        };
        if sp.active_spatial_id == 0 {
            return EngineMv { x: 0, y: 0 };
        }
        let Some(held) = self
            .store
            .held(slot)
            .filter(|h| h.spatial_id != sp.active_spatial_id)
        else {
            return EngineMv { x: 0, y: 0 };
        };
        let (ref_w, ref_h) = held.size();
        let (own_w, own_h) = (self.grids.width, self.grids.height);
        if (ref_w, ref_h) == (own_w, own_h) {
            return EngineMv { x: 0, y: 0 };
        }
        let delta = |ref_s: usize, own_s: usize| -> i16 {
            let num = 2 * (ref_s as i64 - own_s as i64);
            let half = ref_s as i64 / 2;
            let rounded = if num >= 0 {
                (num + half) / ref_s as i64
            } else {
                (num - half) / ref_s as i64
            };
            rounded as i16
        };
        EngineMv {
            x: delta(ref_w, own_w),
            y: delta(ref_h, own_h),
        }
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
            let key = RankedLuma::for_block(position, self.active_ref, mv, filter);
            let cached = unscaled && self.ranked_luma == Some(key);
            let prediction = if cached {
                &self.ranked_luma_samples[..side * side]
            } else {
                let trial = &mut self.scratch.filter_trial[..side * side];
                inter_pred::predict_with_ref_size(
                    &self.dsp,
                    reference,
                    ref_size,
                    PlaneIndex::Y,
                    position,
                    mv,
                    filter,
                    &mut self.inter.window,
                    trial,
                );
                trial
            };
            let pred = PlaneView::new(prediction, side, side, side).expect("prediction block");
            let cost = self.dsp.variance(size, &src, &pred).sse as f32 + lambda * symbol_bits;
            if cost < best.0 {
                best = (cost, filter);
                if cached {
                    self.scratch.filter_best[..side * side].copy_from_slice(prediction);
                } else {
                    std::mem::swap(
                        &mut self.scratch.filter_trial,
                        &mut self.scratch.filter_best,
                    );
                }
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

    /// Mode 2 of `vp9.inter_intra_dc_only` (`vp9-svc` / `vp9-svc3`): skips
    /// directional intra search only on interior blocks of `>= 360p` non-reset,
    /// unrefined inter pictures, preserving directional intra on key/repair
    /// enhancement pictures (`coding_from_base_only` or `intra_only`, coded as
    /// `INTER_FRAME` syntax), block-QP refinements, sub-`360p` layers
    /// (`320x180`), and blocks touching any frame edge.
    fn interior_spatial_inter_block(&self, block: &Block) -> bool {
        if self.coding_from_base_only()
            || self.refining_block()
            || self
                .spatial
                .as_ref()
                .is_some_and(|sp| sp.current_layer.intra_only)
            || self.grids.width * self.grids.height < MIN_INTER_INTRA_DC_AREA
        {
            return false;
        }
        let size = block.size();
        block.x > 0
            && block.y > 0
            && block.x + size < self.grids.width
            && block.y + size < self.grids.height
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
        self.intra_mode_candidates(block, tx, (source, recon), 1)[0].0
    }

    /// The `keep` cheapest luma modes of an intra block by the measure of
    /// [`Self::choose_intra_modes`] (of those, the staged cut's survivors
    /// where it is on), cheapest first, each with its cheapest chroma mode
    /// (conditioned on that luma mode) and its luma cost.
    fn intra_mode_candidates(
        &self,
        block: &Block,
        tx: TxSize,
        (source, recon): (&Picture, &Picture),
        keep: usize,
    ) -> Vec<((IntraMode, IntraMode), f32)> {
        // Speed lever (`vp9.inter_intra_dc_only`): intra blocks of inter
        // frames take DC without a search (`1.0` = all non-key frames in VP9
        // Default; `2.0` = interior blocks of >=360p non-reset inter frames).
        if !self.header.key
            && self.params.inter_intra_dc_only >= 0.5
            && (self.params.inter_intra_dc_only < 1.5 || self.interior_spatial_inter_block(block))
        {
            return vec![((IntraMode::Dc, IntraMode::Dc), 0.0)];
        }
        let (row, col) = (block.y >> MI_LOG2, block.x >> MI_LOG2);
        let whole_block = self.params.intra_mode_estimate >= 0.5;
        let (metric, bit_weight) = if whole_block {
            (recon::IntraMetric::Satd, self.lambda().sqrt())
        } else {
            (recon::IntraMetric::Sse, self.lambda())
        };
        let search = IntraSearch::of(self.params.intra_mode_search);
        let rank =
            |plane: PlaneIndex, plane_tx: TxSize, costs: &[f32; INTRA_MODES], keep: usize| {
                // Speed heuristic (`vp9.intra_mode_search` 2, libvpx real-time
                // speeds' intra masks): chroma takes DC unsearched and luma
                // chooses among DC, V and H. Measured in
                // docs/handoff/vp9-faster.md.
                if search == IntraSearch::RealTime && plane != PlaneIndex::Y {
                    return vec![(IntraMode::Dc, 0.0)];
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
                    distortion as f32 + bit_weight * costs[mode as usize]
                };
                // The engine's shared ranking keeps the cheapest.
                let ranked = |modes: &[IntraMode], keep: usize| {
                    nc_encoder::rank_intra(modes.iter().copied(), keep, &cost)
                };

                match search {
                    IntraSearch::All => return ranked(&CODED_INTRA_MODES, keep),
                    IntraSearch::RealTime => return ranked(&REAL_TIME_LUMA_MODES, keep),
                    IntraSearch::Pruned => {}
                }
                // Speed heuristic (`vp9.intra_mode_search`): the four
                // non-directional modes, then the two directions beside the
                // winner's. Measured in the commit message.
                let mut base = ranked(&BASE_INTRA_MODES, keep);
                let refined = ranked(&directions_beside(base[0].0), keep);
                base.extend(refined);
                base.sort_by(|a, b| a.1.total_cmp(&b.1));
                base.truncate(keep);
                base
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
        let lumas = rank(PlaneIndex::Y, tx, y_costs, keep);
        // The staged cut's survivors only get a chroma mode.
        let cut = self.intra_stage.cut(self.header.key);
        let lumas = if cut.is_active() {
            cut.survivors(lumas)
        } else {
            lumas
        };
        lumas
            .into_iter()
            .map(|(y, cost)| {
                (
                    (y, rank(PlaneIndex::U, uv_tx, &uv_costs[y as usize], 1)[0].0),
                    cost,
                )
            })
            .collect()
    }

    /// The modes of an intra block: random syntax, the RD choice among
    /// `vp9.intra_rd_candidates`, or the prediction measure alone.
    fn intra_modes(
        &mut self,
        block: &Block,
        tx: TxSize,
        source: &Picture,
        recon: &mut Picture,
    ) -> (IntraMode, IntraMode) {
        if let Some(random) = &mut self.random {
            return random.intra_modes(&CODED_INTRA_MODES);
        }
        // The staged cut ranks every luma mode and codes its survivors.
        let keep = if self.intra_stage.cut(self.header.key).is_active() {
            INTRA_MODES
        } else {
            self.params.intra_rd_candidates.round().max(1.0) as usize
        };
        if keep > 1 && !self.lossless {
            self.rd_intra_modes(block, tx, source, recon, keep)
        } else {
            self.choose_intra_modes(block, tx, source, recon)
        }
    }

    /// Mode bits of an intra block's luma and chroma modes at its contexts.
    fn intra_mode_bits(&self, block: &Block, (y, uv): (IntraMode, IntraMode)) -> f32 {
        let costs = &self.mode_costs;
        if self.header.key {
            let (row, col) = (block.y >> MI_LOG2, block.x >> MI_LOG2);
            let (above, left) = contexts::key_frame_modes(&self.grids.grid, row, col);
            costs.kf_y[above as usize][left as usize][y as usize]
                + costs.kf_uv[y as usize][uv as usize]
        } else {
            costs.y[size_group(block.log2_size())][y as usize] + costs.uv[y as usize][uv as usize]
        }
    }

    /// The intra modes of a block decided by coded cost
    /// (`vp9.intra_rd_candidates`): the cheapest few by the prediction
    /// measure are each coded at the largest transform, all planes, and the
    /// one with the lowest squared error plus λ · (token and mode bits) is
    /// kept. Why: SATD of the prediction scores residual energy the
    /// quantiser removes; on AV1 keys SATD top 1 left 70 % of DC-only's RD
    /// loss against a transform-costed search, top 2 + RD 29 %
    /// (docs/handoff/vp9-av2.md "AV2 on the Pareto frontier"). Leaves
    /// `recon` holding some trial; the caller codes the winner.
    fn rd_intra_modes(
        &mut self,
        block: &Block,
        tx: TxSize,
        source: &Picture,
        recon: &mut Picture,
        keep: usize,
    ) -> (IntraMode, IntraMode) {
        let candidates = self.intra_mode_candidates(block, tx, (source, recon), keep);
        if candidates.len() == 1 {
            return candidates[0].0;
        }
        let lambda = self.lambda();
        // Speed heuristic (`vp9.intra_rd_trellis` 0): the trials compare
        // modes, so plain quantisation ranks them nearly as the trellis
        // would at a fraction of the work. Measured in
        // docs/analysis/rd-mode-audit-2026-11-19.md.
        self.scratch.plain_trial = self.params.intra_rd_trellis < 0.5;
        let mut best = (f32::INFINITY, candidates[0].0);
        for (modes, _) in candidates {
            self.code_residuals(
                block,
                (tx, TrialPlanes::All),
                Some(modes),
                ResidualDrop::None,
                source,
                recon,
            );
            let cost = self.reconstruction_sse(block, source, recon) as f32
                + lambda * (self.token_bits(false) + self.intra_mode_bits(block, modes));
            if cost < best.0 {
                best = (cost, modes);
            }
        }
        self.scratch.plain_trial = false;
        best.1
    }

    /// Keep the coded state of the current transform-size trial (its
    /// reconstruction, levels and transform blocks) as the best so far.
    /// Only the `planes` the next trial recodes are kept: the others stay
    /// as they are in `recon` and the level buffer.
    fn save_best_trial(&mut self, block: &Block, recon: &Picture, planes: TrialPlanes) {
        let scratch = &mut self.scratch;
        copy_block_out(&self.dsp, block, recon, &mut scratch.best_recon, planes);
        let used = trial_levels(block, planes);
        scratch.best_levels[..used].copy_from_slice(&scratch.levels[..used]);
        scratch.best_coded.clone_from(&scratch.coded);
    }

    /// Return to the trial kept by [`Self::save_best_trial`] without coding
    /// it again.
    /// `planes` are the ones the save kept (the failed trial's).
    fn restore_best_trial(&mut self, block: &Block, recon: &mut Picture, planes: TrialPlanes) {
        let scratch = &mut self.scratch;
        copy_block_in(&self.dsp, block, &scratch.best_recon, recon, planes);
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
                coded.drop_levels();
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
                let levels = &self.scratch.levels[coded.levels()];
                coeff::cost((levels, usize::from(coded.eob)), model, probs, ctx)
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
                nc_encoder::block_sse(&self.dsp, (source, recon), plane, (x, y, n))
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
            Some(ratio) => nc_encoder::adaptive_error_weight(gain, ratio) as f32,
            None => self.params.chroma_sse_weight,
        }
    }

    /// Whether a chroma control reads the luma/chroma error ratio.
    fn measures_chroma_error_ratio(&self) -> bool {
        self.chroma_policy.is_none()
            && (self.params.chroma_delta_gap_slope > 0.0 || self.params.chroma_weight_adapt > 0.0)
    }

    /// SSE of the block's reconstruction and of its saved prediction
    /// against the source, all planes (shared DSP kernels), each weighted
    /// by [`Self::distortion_weight`].
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
            let weight = self.distortion_weight(plane);
            coded_total += weight * self.dsp.variance(size, &src, &rec).sse as f32;
            predicted_total += weight * self.dsp.variance(size, &src, &predicted).sse as f32;
        }
        (coded_total, predicted_total)
    }

    /// The squared-error weight of `plane` in the backend's rate-distortion
    /// decisions: 1 for luma; for chroma `vp9.chroma_sse_weight` (or its
    /// adaptive form), or the chroma policy's squared step ratio.
    fn distortion_weight(&self, plane: PlaneIndex) -> f32 {
        match (plane, self.chroma_policy) {
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
        }
    }

    /// Put the saved inter prediction of `planes` back into `recon`.
    fn restore_prediction(&self, block: &Block, recon: &mut Picture, planes: TrialPlanes) {
        copy_block_in(&self.dsp, block, &self.scratch.prediction, recon, planes);
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
        let n = block.size();
        let pred = PlaneView::new(&self.scratch.prediction, n, n, n).expect("packed prediction");
        let src = source
            .plane(PlaneIndex::Y)
            .region(block.x as isize, block.y as isize, n, n)
            .expect("block inside the source");
        let quadrants = nc_encoder::quadrant_sse(&self.dsp, &src, &pred, n);
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
    /// `vp9.partial_forward` (VP9 L2 Fast) still content stayed at the
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
    /// when they changed; on a spatial frame where no block took a non-default
    /// feature (`SegmentFeatures::default`), turn segmentation off so the
    /// header and tile omit `Segmentation::standalone`'s empty table and map.
    fn finish_segment_quantizers(&mut self) {
        let (Some(segments), true) = (&self.block_qp, self.header.segmentation.enabled) else {
            return;
        };
        let segmentation = &mut self.header.segmentation;
        segmentation.features.quantizer = segments.feature_data(segmentation.features.absolute);
        if self.spatial.is_some()
            && self.random.is_none()
            && segmentation.features == SegmentFeatures::default()
        {
            *segmentation = Segmentation::disabled(self.segment_state);
            return;
        }
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
        if self.scratch.plain_trial {
            return None;
        }
        let trellis = self
            .trellis
            .as_ref()
            .filter(|t| !self.lossless && t.settings.covers(self.header.key))?;
        let scale = if plane == PlaneIndex::Y {
            1.0
        } else {
            let ratio =
                self.block_steps(plane).ac as f32 / self.block_steps(PlaneIndex::Y).ac as f32;
            trellis.settings.chroma_scale? * ratio * ratio
        };
        Some(self.lambda() * trellis.settings.lambda_scale_for(self.header.key) * scale)
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
        // picture (whose references are the base) and a standalone `intra_only`
        // repair (whose receiver slot may still hold `S0` from the key unit)
        // write the explicit frame size.
        let size_from_ref = match &self.spatial {
            Some(sp) if !sp.current_layer.key_unit && sp.current_layer.intra_only => None,
            Some(sp) => {
                let size = (self.grids.width, self.grids.height);
                ref_slots
                    .iter()
                    .position(|&slot| sp.slot_sizes[usize::from(slot)] == Some(size))
            }
            None => Some(0),
        };
        // Lower spatial layers (`spatial_id + 1 < num_layers`) and K-SVC top
        // layers (`InterLayerMode::KeyOnly`) are coded with `show_frame = 0`
        // and displayed via `show_existing_frame` (`superframe.rs`), keeping
        // `use_prev_frame_mvs = 0` (`VP9 §7.2`) across all extracted views.
        let hidden_layer = self.spatial.as_ref().is_some_and(|sp| {
            sp.current_layer.spatial_id + 1 < sp.num_layers
                || sp.inter_layer == InterLayerMode::KeyOnly
        });
        let interp_filter = match &mut self.random {
            Some(random) => random.interp_filter(),
            None => INTERP_FILTERS
                .get(self.params.interp_filter.round() as usize)
                .copied(),
        };
        let refresh = match plan.references.refresh {
            _ if key => ALL_SLOTS,
            // A hidden spatial layer picture (`S0`, `S1` of three, or `S_top`
            // in K-SVC) is shown via `show_existing_frame` when it is the top
            // layer of a unit or extracted view, which needs a slot holding
            // it: a disposable `T2` picture with empty `refresh` refreshes its
            // scratch slot (`hidden_scratch_slot`), leaving temporal slots
            // `0..2*N` untouched.
            mask if hidden_layer && mask.is_empty() => {
                let sp = self.spatial.as_ref().expect("hidden_layer implies spatial");
                1u8 << hidden_scratch_slot(sp.num_layers, sp.current_layer.spatial_id)
            }
            mask => mask.0,
        };
        FrameHeader {
            key,
            show_frame: !hidden_layer,
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
                (None, false) if !key && !self.filter_effort.deblock => 0,
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
            colour: self.colour,
        }
    }

    /// The engine's (quarter-sample) candidate for VP9 reference vector
    /// `mv`: an odd (1/8) component moves towards zero, and a spatial
    /// layer's reference offset comes off.
    fn engine_candidate(&self, mv: Mv) -> EngineMv {
        let mut engine = to_engine_mv(mv);
        if let Some(sp) = &self.spatial {
            let offset = sp.scaled_mv_offset(self.active_ref);
            engine.x = engine.x.saturating_sub(offset.x);
            engine.y = engine.y.saturating_sub(offset.y);
        }
        engine
    }

    /// The VP9 vector the engine's `mv` codes for a block with reference
    /// vectors `refs`. With 1/8 precision allowed the block's NEARESTMV or
    /// NEARMV vector can be odd, which the engine sees only as its
    /// quarter-sample candidate ([`Self::engine_candidate`]); that
    /// candidate stands for the reference vector itself, so it is priced
    /// and coded as NEARESTMV / NEARMV. (Converting it as a plain vector
    /// made it a NEWMV one eighth away: with `vp9.high_precision_qindex`
    /// 256 the cheap modes of blocks next to 1/8 vectors were lost,
    /// `bodycam_airport` fixed QP BD-rate PSNR-YUV +7.2%.)
    /// The zero vector stays ZEROMV.
    fn coded_vector(&self, refs: &MvRefs, mv: EngineMv) -> Mv {
        let plain = match &self.spatial {
            None => to_vp9_mv(mv),
            Some(sp) => sp.to_vp9_mv(self.active_ref, mv),
        };
        if plain == Mv::ZERO {
            return plain;
        }
        [refs.nearest, refs.near]
            .into_iter()
            .find(|&r| (r.row & 1 != 0 || r.col & 1 != 0) && self.engine_candidate(r) == mv)
            .unwrap_or(plain)
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
        let base_mv = self.coded_vector(&refs, mv);
        let mv = self.refine_high_precision(source, (ref_pic, ref_size), position, &refs, base_mv);
        let allow_hp = self.header.allow_hp;
        let new_codable = new_mv_codable(&refs, mv, allow_hp);
        let mode = match &mut self.random {
            Some(random) => random.inter_mode(&refs, mv, new_codable),
            None => cheapest_inter_mode(&self.mode_costs, &refs, mv, allow_hp).0,
        };
        // A vector no mode can code (NEWMV out of range or of precision, and
        // no reference vector equal to it) is replaced by NEARESTMV before it
        // predicts anything, so the stream stays decodable: the engine never
        // hands over such a vector on the shipped presets (its NEWMV price is
        // infinite), this keeps an encoder abort impossible if one ever did.
        let (mode, mv) = if mode == InterMode::New && !new_codable {
            (InterMode::Nearest, refs.nearest)
        } else {
            (mode, mv)
        };
        let (filter, luma_ready) = self.choose_filter(source, (ref_pic, ref_size), position, mv);
        let side = block.size();
        let scratch = &mut self.scratch;
        let key = RankedLuma::for_block(position, self.active_ref, mv, filter);
        let luma = if luma_ready {
            Some(&scratch.filter_best[..side * side])
        } else if ref_size == (position.width, position.height) && self.ranked_luma == Some(key) {
            Some(&self.ranked_luma_samples[..side * side])
        } else {
            None
        };
        recon::predict_inter(
            &self.dsp,
            &mut self.inter,
            &mut scratch.prediction,
            (ref_pic, ref_size, position),
            (mv, filter),
            luma,
        );
        copy_block_in(
            &self.dsp,
            block,
            &scratch.prediction,
            recon,
            TrialPlanes::All,
        );
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
            let rd_trim = self.rd_trim_lambda(plane);
            let plane_tx = self.plane_tx(block, tx, plane);
            let skip_plane = dropped.includes(plane)
                || self.plane_skip(block, (plane, plane_tx), &shortcuts, (source, recon));
            let quads = if quad_planes && !skip_plane {
                self.quad_sads(block, (plane, plane_tx), (source, recon))
            } else {
                None
            };
            let steps = self.header.steps_at(self.coding_q_idx, plane);
            let (rounding, min_level) = plane_quantization[usize::from(plane != PlaneIndex::Y)];
            let quantization =
                transform::PlaneQuantization::new(plane_tx, steps, (rounding, min_level));
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
                        .push(CodedTx::uncoded(&tx_block, ty, offset));
                    offset += area;
                    continue;
                }
                let tx_coding = (plane_tx, offset, &quantization);
                let nonrd = (steps, shortcuts.nonrd_scale);
                if self.code_nonrd(source, recon, &tx_block, tx_coding, nonrd) {
                    offset += area;
                    continue;
                }
                let side = plane_tx.side();
                // Exact shortcut: the zero proof takes the SAD alone (from
                // the quad pass, else one SAD kernel call), so a block it
                // drops forms no residual (nor, from the quads, any views);
                // a kept block's residual is then formed without its SAD.
                // Intra blocks and blocks past the real-time skip test
                // (`vp9.nonrd_skip_scale`) seldom prove zero, so they form
                // residual and SAD in one pass as before (SAD first for every
                // block: VP9 Faster +0.4 % x86 instructions, Veryfast -1.2 %).
                let provable = ty == transform::TxType::DctDct && !self.lossless;
                let proven_zero =
                    |sad: u32| quantization.dct_levels_are_zero(sad, shortcuts.zero_scale);
                let quad_sad = quads.map(|q| q.tx_sad(&self.scratch.quad_sads, &tx_block, side));
                let uncoded = CodedTx::uncoded(&tx_block, ty, offset);
                if provable && quad_sad.is_some_and(proven_zero) {
                    self.scratch.coded.push(uncoded);
                    offset += area;
                    continue;
                }
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
                let unproven = provable && quad_sad.is_none();
                let fused = unproven && (intra.is_some() || shortcuts.nonrd_scale > 0.0);
                if unproven && !fused && proven_zero(self.dsp.sad(size, &src, &rec)) {
                    self.scratch.coded.push(uncoded);
                    offset += area;
                    continue;
                }
                let residual = &mut self.scratch.residual[..area];
                if fused {
                    if proven_zero(self.dsp.residual_sad(size, &src, &rec, residual, side)) {
                        self.scratch.coded.push(uncoded);
                        offset += area;
                        continue;
                    }
                } else {
                    self.dsp.residual(size, &src, &rec, residual, side);
                }
                let coefficients = &mut self.scratch.coefficients[..area];
                let levels = &mut self.scratch.levels[offset..offset + area];
                let zero_limit = quantization.zero_limit();
                let transformed = transform_residual(
                    &self.dsp,
                    (plane_tx, ty, self.lossless, shortcuts.partial_forward),
                    residual,
                    zero_limit,
                    coefficients,
                );
                let nonzero = transformed
                    && quantization.quantize(&self.dsp, self.lossless, coefficients, levels);
                let trimmed_eob = if nonzero {
                    trim_levels(&self.dsp, plane_tx, ty, (min_level, shave_gap), levels)
                } else {
                    Some(0)
                };
                let nonzero = trimmed_eob != Some(0);
                let eob = match (&mut self.trellis, trellis_lambda) {
                    (Some(trellis), Some(lambda)) if nonzero => {
                        let block = (
                            &self.context,
                            &self.grids.contexts,
                            &tx_block,
                            intra.is_none(),
                        );
                        trellis.quantize(
                            &self.dsp,
                            block,
                            ty,
                            (lambda, coefficients, steps),
                            (levels, trimmed_eob),
                        )
                    }
                    _ => match rd_trim {
                        Some((trim, lambda)) if nonzero => coeff::rate_trim(
                            &self.dsp,
                            &trim,
                            (levels, coefficients),
                            (plane_tx, ty),
                            steps,
                            lambda,
                        ),
                        _ => trimmed_eob.unwrap_or_else(|| {
                            coeff::end_of_block(&self.dsp, levels, plane_tx, ty)
                        }),
                    },
                };
                if eob != 0 {
                    self.buffers.add_residual(
                        recon,
                        &tx_block,
                        ty,
                        self.lossless,
                        (levels, eob),
                        steps,
                    );
                }
                let coded = CodedTx::new(&tx_block, ty, eob, offset);
                self.scratch.coded.push(coded);
                offset += area;
            }
        }
        self.scratch.finish_trial();
    }

    /// The rate-aware trimming of a transform block of `plane` and its λ:
    /// luma only, while the trellis is off (`trellis_lambda` covers it then).
    fn rd_trim_lambda(&self, plane: PlaneIndex) -> Option<(nc_encoder::RateTrim, f64)> {
        if self.trellis.is_some() || self.lossless || plane != PlaneIndex::Y {
            return None;
        }
        let trim = self.rd_trim.filter(|t| t.covers(self.header.key))?;
        Some((trim, f64::from(self.lambda())))
    }

    /// The inexact residual shortcuts in force for an intra or inter block.
    fn residual_shortcuts(&self, intra: bool) -> ResidualShortcuts {
        // A block quantised finer than the frame is being refined (the
        // reuse term restoring detail a degraded reference lost): the high
        // frequencies the approximation drops are what it is for. With
        // `vp9.inter_tx_split_spread` keeping such blocks in one 32x32
        // transform, a still region copied from a starved key stayed 8 dB
        // short for good (`VP9 Fast`, `nc-cli/tests/static_recovery.rs`).
        // An enhancement picture coded from the upscaled base only keeps every
        // frequency too (`coding_from_base_only`).
        let exact = self.refining_block() || self.coding_from_base_only();
        let partial_forward = self.params.partial_forward >= 0.5
            && !self.lossless
            && !exact
            && nc_encoder::WorkGate::partial_transform(&self.gates)
                .admits_quantizer(f32::from(self.coding_q_idx) / QINDEX_MAX as f32);
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
    /// several to the plane block, pay for the pass. VP9 Veryfast (8x8
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
        (tx, offset, quantization): (TxSize, usize, &transform::PlaneQuantization),
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
        let coded = CodedTx::new(tx_block, ty, usize::from(nonzero), offset);
        self.scratch.coded.push(coded);
        true
    }

    /// Code only the DC coefficient `dc` of a transform block; returns
    /// whether its level is nonzero.
    fn code_dc(
        &mut self,
        recon: &mut Picture,
        tx_block: &TxBlock,
        (tx, offset, quantization): (TxSize, usize, &transform::PlaneQuantization),
        steps: transform::Steps,
        dc: i16,
    ) -> bool {
        let area = tx.area();
        let levels = &mut self.scratch.levels[offset..offset + area];
        let nonzero = quantization.quantize_dc(&self.dsp, dc, &mut levels[0]);
        if nonzero {
            // Later pricing and reconstruction still see a raster block.
            levels[1..].fill(0);
            let ty = transform::TxType::DctDct;
            self.buffers
                .add_residual(recon, tx_block, ty, false, (levels, 1), steps);
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
            let eob = usize::from(coded.eob);
            let mut coder = coeff::Coder::new(model, &self.context, &mut self.counts);
            coeff::write(&mut self.writer, (levels, eob), model, &mut coder, ctx);
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

/// Drop the quantized levels the encoder chooses not to code. Returns
/// the end of block when known. Past qindex 255 the residual is quantised as if at
/// the planned step (`FramePlan::lambda_scale`, levels below `min_level`):
/// λ alone left blocks drifting into dense residual (`pan_bdd_00cef86b` at
/// 150 kbps: 283 kbps with 2 octaves, 403 with 3). `shave_gap`
/// (`vp9.shave_gap`) drops isolated trailing ±1 levels.
fn trim_levels(
    dsp: &nc_dsp::Dsp,
    tx: TxSize,
    ty: transform::TxType,
    (min_level, shave_gap): (i32, usize),
    levels: &mut [i32],
) -> Option<usize> {
    if min_level > 1 && !transform::apply_planned_dead_zone(tx, min_level, levels) {
        return Some(0);
    }
    (shave_gap != 0).then(|| coeff::shave_isolated_ones(dsp, levels, tx, ty, shave_gap))
}

/// VP9's delta range (`delta_q` is 4 bits and a sign, §6.2.9).
const CHROMA_DELTA_LIMIT: i32 = 15;
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
    DcOnly(i16),
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
/// times 8 / side (32x32: / 8), at most `2040 * 16` for 8-bit residuals:
/// a 16-bit coefficient like the forward's.
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
    NonrdSkip::DcOnly(i16::try_from(dc).expect("the DC of 8-bit residuals fits 16 bits"))
}

/// libvpx's skip thresholds are the squared dequantizer step `>> 6`.
const NONRD_THRESHOLD_SHIFT: u32 = 6;
/// Forward DCT outputs are 8x orthonormal below 32x32 ...
const DCT_DC_SCALE: i64 = 8;
/// ... and the 32x32 DC is the residual sum / 8 (`vpx_fdct32x32_1`).
const DC_SUM_DIVISOR_32: i32 = 8;

fn policy_chroma_delta(policy: nc_encoder::chroma_policy::ChromaPolicy, qindex: u8) -> i8 {
    let ac = |d: i32| f64::from(transform::Steps::new(qindex, 0, d as i8).ac);
    policy.closest_step_delta(-CHROMA_DELTA_LIMIT..=CHROMA_DELTA_LIMIT, VP9_STEP_SCALE, ac) as i8
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
    let last = AC_QUANT.len() as u32 - 1;
    let step = |q: u32| f64::from(AC_QUANT[q as usize]);
    nc_encoder::scaled_quantizer(0..=last, step, qindex, scale)
        .unwrap_or(last)
        .max(1)
}

/// The cheapest inter mode signalling `mv`, with its bits (mode, plus the
/// vector difference for NEWMV).
/// Whether NEWMV can code `mv` for a block with reference vectors `refs`
/// (VP9 §6.4.19): without 1/8 precision (`allow_hp` and a small nearest
/// vector) an odd difference cannot be coded, nor one past `MV_CLASS_10`
/// (±2048 samples), which a vector and a nearest vector of opposite sign
/// reach on wide pictures (1080p and up). Mode choices offer NEWMV only
/// where this holds ([`cheapest_inter_mode`], the stress chooser), so the
/// writer never meets a difference it cannot code.
fn new_mv_codable(refs: &MvRefs, mv: Mv, allow_hp: bool) -> bool {
    let usehp = allow_hp && motion::uses_high_precision(refs.nearest);
    let (dr, dc) = (mv.row - refs.nearest.row, mv.col - refs.nearest.col);
    let in_range = |v: i32| (-motion::MAX_DIFFERENCE..=motion::MAX_DIFFERENCE).contains(&v);
    (usehp || (dr & 1 == 0 && dc & 1 == 0)) && in_range(dr) && in_range(dc)
}

fn cheapest_inter_mode(
    costs: &ModeCosts,
    refs: &MvRefs,
    mv: Mv,
    allow_hp: bool,
) -> (InterMode, f32) {
    let mode_cost = |mode: InterMode| costs.inter[refs.mode_context][mode as usize];
    let usehp = allow_hp && motion::uses_high_precision(refs.nearest);
    let (dr, dc) = (mv.row - refs.nearest.row, mv.col - refs.nearest.col);
    let new_cost = if new_mv_codable(refs, mv, allow_hp) {
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

/// The engine vector of a VP9 vector: exact for even eighths, an odd
/// (1/8) component moves towards zero.
fn to_engine_mv(mv: Mv) -> EngineMv {
    EngineMv {
        x: (mv.col / MV_PER_ENGINE_MV) as i16,
        y: (mv.row / MV_PER_ENGINE_MV) as i16,
    }
}

/// Copy the block's samples of every plane out of `picture`, packed plane
/// after plane.
fn copy_block_out(
    dsp: &nc_dsp::Dsp,
    block: &Block,
    picture: &Picture,
    packed: &mut [u8],
    planes: TrialPlanes,
) {
    let mut offset = 0;
    for plane in PlaneIndex::ALL.into_iter().filter(|&p| planes.includes(p)) {
        let ss = usize::from(plane != PlaneIndex::Y);
        let (x, y, n) = (block.x >> ss, block.y >> ss, block.size() >> ss);
        let view = picture.plane(plane).view();
        let from = view.sub_view(x, y, n, n).expect("block inside picture");
        dsp.block_copies().copy_rect(
            from.data(),
            from.stride(),
            &mut packed[offset..offset + n * n],
            n,
            (n, n),
            false,
        );
        offset += n * n;
    }
}

/// Forward-transform a block's `residual` into raster `coefficients`;
/// false when the block certainly quantizes to zero (every coefficient at
/// most `zero_limit`, `PlaneQuantization::zero_limit`), which the
/// bounded 16x16/32x32 DCT proves exactly before its row pass, leaving
/// `coefficients` unspecified. With `partial` a 32x32 block keeps only its
/// low 16x16 frequencies (`vp9.partial_forward`).
fn transform_residual(
    dsp: &nc_dsp::Dsp,
    (tx, ty, lossless, partial): (TxSize, transform::TxType, bool, bool),
    residual: &[i16],
    zero_limit: u32,
    coefficients: &mut [i16],
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
fn copy_block_in(
    dsp: &nc_dsp::Dsp,
    block: &Block,
    packed: &[u8],
    picture: &mut Picture,
    planes: TrialPlanes,
) {
    let mut offset = 0;
    for plane in PlaneIndex::ALL.into_iter().filter(|&p| planes.includes(p)) {
        let ss = usize::from(plane != PlaneIndex::Y);
        let (x, y, n) = (block.x >> ss, block.y >> ss, block.size() >> ss);
        let mut view = picture.plane_mut(plane).view_mut();
        let mut dst = view
            .sub_view_mut(x, y, n, n)
            .expect("block inside the picture");
        let ds = dst.stride();
        dsp.block_copies().copy_rect(
            &packed[offset..offset + n * n],
            n,
            dst.data_mut(),
            ds,
            (n, n),
            false,
        );
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
    out: &mut [u16],
) {
    let columns = n / QUAD_SIDE;
    let blocks = n / QUAD_STRIP;
    let mut strip = [[0u16; 4]; 64 / QUAD_STRIP];
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
    fn tx_sad(self, quads: &[u16], tx_block: &TxBlock, side: usize) -> u32 {
        let (cx, cy) = (
            (tx_block.x - self.x) / QUAD_SIDE,
            (tx_block.y - self.y) / QUAD_SIDE,
        );
        let cells = side / QUAD_SIDE;
        (cy..cy + cells)
            .map(|row| {
                let start = row * self.columns + cx;
                quads[start..start + cells]
                    .iter()
                    .map(|&sad| u32::from(sad))
                    .sum::<u32>()
            })
            .sum()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_odd_difference_is_new_mv_only_with_high_precision() {
        // 2026-11-21 syntax stress: an odd (1/8) NEARMV vector chosen as NEWMV
        // beside an even nearest vector could not be coded (writer panic).
        let refs = MvRefs {
            nearest: Mv { row: 0, col: 8 },
            near: Mv { row: 3, col: 8 },
            mode_context: 0,
        };
        assert!(!new_mv_codable(&refs, refs.near, false));
        assert!(new_mv_codable(&refs, refs.near, true));
        assert!(new_mv_codable(&refs, Mv { row: 2, col: 8 }, false));
    }

    #[test]
    fn a_difference_past_the_coded_range_is_not_new_mv() {
        let costs = ModeCosts::new(&FrameContext::default());
        let refs = MvRefs {
            nearest: Mv { row: 0, col: -8000 },
            near: Mv::ZERO,
            mode_context: 0,
        };
        // 2048 samples right of a nearest vector 1000 samples left.
        let far = Mv { row: 0, col: 8384 };
        assert!(
            cheapest_inter_mode(&costs, &refs, far, false)
                .1
                .is_infinite()
        );
        let near = Mv { row: 0, col: 0 };
        assert!(
            cheapest_inter_mode(&costs, &refs, near, false)
                .1
                .is_finite()
        );
    }
}
