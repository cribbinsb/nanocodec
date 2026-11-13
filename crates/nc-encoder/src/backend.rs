//! The interface between the shared engine and a codec.
//!
//! The engine decides *what* to code (frame type, partitioning, per-block
//! modes); a backend decides *how* it is written: syntax, entropy coding,
//! transforms, quantisation and reconstruction, all of which differ between
//! codecs. Distortions are sample-domain squared error and `lambda()` is per
//! bit, so the engine compares costs across backends without knowing their
//! syntax.
//!
//! Call order for one frame:
//!
//! ```text
//! begin_frame
//!   for each superblock (raster order):
//!     begin_superblock
//!     partition walk (depth first, children in raster order):
//!       write_partition for every node (write_shape instead for a node
//!       coded as rectangular leaves), then either recurse or, per leaf:
//!         (inter frames) mv_candidates, mode_bits for each candidate mode,
//!                        sub_block_motion (if wants_sub_block_motion)
//!         encode_block
//! loop_filter_frame
//! end_frame
//! ```
//!
//! On inter frames the engine ranks candidate modes itself (prediction error
//! plus `lambda() * mode_bits(...)`) and hands the winner to `encode_block`;
//! the backend only says which vectors it can signal cheaply and what each
//! mode costs to signal. *How* a vector is signalled (AV1 NEAREST/NEAR/NEW,
//! HEVC merge/AMVP, H.264 predicted MV) is the backend's business.

use nc_common::Picture;

use crate::config::{ConfigError, EncoderConfig};
use crate::mv::Mv;
use crate::partition::Block;
use crate::refresh::RefreshWindow;
use crate::structure::FrameReferences;

/// [`CodecBackend::loop_filter_reach`] of codecs whose filters modify at
/// most 3 samples across an edge (H.264, HEVC deblocking; SAO reads one
/// more), with margin.
pub const DEFAULT_LOOP_FILTER_REACH: usize = 8;

/// Kind of frame being coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// Intra-only random access point (AV1 `KEY_FRAME`, H.264/HEVC IDR).
    Key,
    /// Predicted from one earlier frame (the previous one in a flat chain;
    /// see [`FramePlan::references`]), forward only.
    Inter,
}

/// Engine-level decisions for one frame, handed to the backend up front.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FramePlan {
    /// Index of the frame in coding (= display) order.
    pub index: u64,
    /// Frame type.
    pub frame_type: FrameType,
    /// Quantiser in the codec's native scale (see `EncoderConfig::quantizer`).
    pub quantizer: u32,
    /// Temporal layer, the reference slot inter prediction reads and the
    /// slots this frame refreshes ([`FrameReferences::FLAT`] in a flat
    /// chain). Backends map these onto their reference syntax.
    pub references: FrameReferences,
    /// Multiple of the backend's decision λ (1 normally). Rate control
    /// may plan past the codec's largest quantizer (`rate.lambda_headroom_octaves`):
    /// the frame is then coded at the largest, with λ grown by the squared
    /// step ratio of the planned quantizer, so decisions keep trading
    /// quality for rate (more skip, less residual) where the quantizer
    /// cannot go further. Backends multiply every λ they use by it.
    pub lambda_scale: f32,
    /// Whether blocks of this frame may carry quantizer offsets
    /// ([`BlockDecision::qp_offset`]): block AQ planned a non-zero offset,
    /// or the syntax stress test may draw one. Backends whose block-QP
    /// syntax costs bits on every block (VP9 segment ids) leave it off
    /// otherwise.
    pub block_qp: bool,
}

/// How a block is predicted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionMode {
    /// Intra DC prediction for luma and chroma.
    IntraDc,
    /// Motion-compensated from the previous frame with this vector.
    Inter(Mv),
}

/// Which of the frame's references an inter block predicts from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferenceChoice {
    /// [`FrameReferences::reference`]: the previous frame in a flat chain.
    #[default]
    Primary,
    /// [`FrameReferences::secondary`]: the last high-quality frame
    /// (`second_reference.rs`).
    Secondary,
    /// Equal-weight compound average of [`ReferenceChoice::Primary`] (with
    /// [`PredictionMode::Inter`]'s vector) and [`ReferenceChoice::Secondary`]
    /// (with `secondary_mv`; `AV1 §7.11.3.15`).
    Compound {
        /// The vector into the secondary reference.
        secondary_mv: Mv,
    },
}

/// Everything the engine decided about one leaf block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockDecision {
    /// Prediction mode.
    pub mode: PredictionMode,
    /// The reference an inter mode predicts from (always
    /// [`ReferenceChoice::Primary`] unless the frame has a secondary one).
    pub reference: ReferenceChoice,
    /// Quantizer offset of this block from the frame quantizer, in the
    /// codec's native units (QP; AV1 qindex). Backends signal it with their
    /// block-level syntax (H.264 `mb_qp_delta`); 0 codes at frame QP.
    pub qp_offset: i16,
}

impl BlockDecision {
    /// A decision for `mode` at the frame quantizer.
    #[must_use]
    pub const fn new(mode: PredictionMode) -> Self {
        Self {
            mode,
            reference: ReferenceChoice::Primary,
            qp_offset: 0,
        }
    }
}

/// Per-picture spatial-layer context passed to [`CodecBackend::prepare_spatial_layer`]
/// immediately before [`CodecBackend::begin_frame`] when `spatial.layers > 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialLayerFrame {
    /// Spatial layer (`0` = scaled base, `1` = full-size enhancement; `AV1 §5.3.3`).
    pub spatial_id: u8,
    /// Temporal layer (`0..=2`; `AV1 §5.3.3`).
    pub temporal_id: u8,
    /// True when this temporal unit starts a random-access GOP (`SequenceHeader`
    /// emitted before `S0`'s frame OBU).
    pub key_unit: bool,
    /// True when a `FrameType::Key` picture is an `INTRA_ONLY_FRAME`
    /// (`frame_type = 2`, refreshing only `references.refresh`, `AV1 §5.9.2`).
    pub intra_only: bool,
    /// An enhancement-layer inter picture predicted only from the same
    /// instant's upscaled base (a key or repair unit's `S1`): its residual is
    /// the detail the base lacks, all high frequencies.
    pub from_base_only: bool,
    /// Visible width of this spatial layer's picture.
    pub width: usize,
    /// Visible height of this spatial layer's picture.
    pub height: usize,
    /// Configured stream peak/target bitrate (`bps`) for Annex A.3 `MaxBitrate`
    /// and `MinCompRatio` level selection (`None` on fixed-QP streams).
    pub max_bitrate_bps: Option<u64>,
}

/// A codec as seen by the engine.
pub trait CodecBackend {
    /// Whether the codec owns picture-level analysis and block traversal.
    /// Such codecs still use shared GOP planning, rate control and reference
    /// storage, but must not pay for a second unused motion/partition analysis.
    const OWN_PICTURE_PIPELINE: bool = false;

    /// Encode the whole picture for codecs opting into `OWN_PICTURE_PIPELINE`.
    /// The engine supplies its planned quantizer/reference and owns feedback.
    /// Other codecs use the ordinary block callbacks and never call this hook.
    fn encode_picture(
        &mut self,
        _plan: &FramePlan,
        _source: &Picture,
        _reference: Option<&Picture>,
        _recon: &mut Picture,
    ) -> Vec<u8> {
        unreachable!("picture pipeline requires an implementation")
    }

    /// The backend's own tunable parameters (declared with
    /// [`param_group!`](crate::param_group), read from `config.params` in
    /// [`CodecBackend::new`]). Validated by the engine together with its own.
    const PARAM_SPECS: &'static [&'static [crate::ParamSpec]] = &[];

    /// The groups of [`CodecBackend::PARAM_SPECS`] this backend re-reads per
    /// frame in [`CodecBackend::set_frame_params`]: only these may take a
    /// non-base frame value (`nonbase.<name>`, `frame_role.rs`). None by
    /// default.
    const FRAME_PARAM_SPECS: &'static [&'static [crate::ParamSpec]] = &[];

    /// Parameters of the [`CodecBackend::FRAME_PARAM_SPECS`] groups that
    /// stay fixed for the stream (parameter-set syntax, state built once):
    /// they may not take a non-base frame value.
    const STREAM_FIXED_PARAMS: &'static [&'static str] = &[];

    /// This codec's own defaults for shared engine parameters (`decision.*`,
    /// `planner.*`), applied beneath the caller's settings. A value measured
    /// for one codec must not silently change the others, so each backend
    /// opts in here with the evidence in a comment.
    const DEFAULT_PARAMS: &'static [(&'static str, f32)] = &[];

    /// Whether the backend codes [`BlockDecision::qp_offset`] with block-level
    /// QP syntax. The engine sends non-zero offsets only to backends that do.
    const SUPPORTS_BLOCK_QP: bool = false;

    /// log2 of the side of the aligned square that shares one block
    /// quantizer (H.264 a macroblock, AV1 a superblock, whose first block
    /// sets `CurrentQIndex`): the engine plans one offset per such square,
    /// so every leaf inside it gets the same [`BlockDecision::qp_offset`].
    /// 0: each leaf has its own.
    const BLOCK_QP_LOG2: u8 = 0;

    /// Distinct non-zero block offsets one frame can code (VP9: the seven
    /// segments besides the frame quantizer's); unlimited where the syntax
    /// codes any offset per block. Block AQ rounds its heuristic terms onto
    /// a grid that fits, so the offsets rate control plans are the ones the
    /// codec codes (`aq.rs`).
    const BLOCK_QP_OFFSETS: usize = usize::MAX;

    /// Whether the backend applies [`FramePlan::lambda_scale`]: rate control
    /// plans past the largest quantizer (`rate.lambda_headroom_octaves`)
    /// only for backends that do. Scaling λ alone is not enough: residual
    /// decisions must follow the planned step too (zero-residual bounds or
    /// dead zone × √`lambda_scale`, RD residual checks at the scaled λ), or
    /// skipped blocks drift until their residual costs more than before
    /// (HEVC and AV1 with λ alone: `crowd_mot20_06` at 150 kbps, rate +25%
    /// and +1%, -2 to -3 dB).
    const SUPPORTS_LAMBDA_HEADROOM: bool = false;

    /// The engine will send non-zero [`BlockDecision::qp_offset`]s (AQ or
    /// the syntax stress test is on), called once before the first frame.
    /// Backends whose block-QP syntax costs bits even at offset 0 (AV1
    /// `delta_q_present`) enable it only then, so other streams are unchanged.
    fn enable_block_qp(&mut self) {}

    /// The native quantizer offset the backend will code for a block the
    /// engine wants coded `requested` steps from the frame quantizer
    /// (non-zero; asked before each such block is decided). Codecs that can
    /// code any offset return it unchanged; VP9, with at most eight
    /// segment quantizers a frame, returns the nearest it can code. The
    /// engine ranks, codes and budgets the block with the answer.
    fn block_qp_offset(&mut self, requested: i16) -> i16 {
        requested
    }

    /// Create a backend for `config`, rejecting anything the codec (or this
    /// implementation of it) cannot code.
    ///
    /// # Errors
    ///
    /// A [`ConfigError`] naming the offending setting.
    fn new(config: &EncoderConfig) -> Result<Self, ConfigError>
    where
        Self: Sized;

    /// Luma samples this codec's in-loop filters (deblocking, SAO, CDEF) can
    /// carry a sample's influence across an edge: intra refresh keeps clean
    /// predictions that far from the dirty region (`refresh.rs`).
    fn loop_filter_reach(&self) -> usize {
        DEFAULT_LOOP_FILTER_REACH
    }

    /// Continue the stream `previous` was coding, after a mid-stream
    /// reconfiguration rebuilt this backend for a new picture size
    /// (`Encoder::set_resolution`): carry over state whose values must
    /// change between consecutive pictures (H.264 `idr_pic_id`). Nothing by
    /// default.
    fn continue_from(&mut self, previous: &Self)
    where
        Self: Sized,
    {
        let _ = previous;
    }

    /// Whether this backend codes an inter frame predicted from any
    /// reference slot and refreshing any set of slots, as the long-term
    /// reference needs (`ltr.rs`: slot 1 beside a flat chain's slot 0).
    fn supports_long_term_reference(&self) -> bool {
        false
    }

    /// Whether inter frames may carry a second reference
    /// ([`FrameReferences::secondary`], `refs.second`): the backend then
    /// signals both references, a per-block reference choice
    /// ([`BlockDecision::reference`]) and prediction from either. The
    /// second one lives in the long-term slot, as with
    /// [`CodecBackend::supports_long_term_reference`].
    fn supports_second_reference(&self) -> bool {
        false
    }

    /// Whether the current inter frame supports compound inter prediction
    /// ([`ReferenceChoice::Compound`], averaging [`ReferenceChoice::Primary`]
    /// and [`ReferenceChoice::Secondary`]; `AV1 §5.11.25`, `AV1 §7.11.3.15`).
    fn supports_compound_reference(&self) -> bool {
        false
    }

    /// Estimated bits to signal compound inter prediction of `block` from
    /// `primary_mv` and `secondary_mv` (mode, reference-pair, compound-mode,
    /// DRL and MV syntax; residual excluded). Used by `decision.rs` when
    /// [`CodecBackend::supports_compound_reference`] is true.
    fn compound_mode_bits(&mut self, block: &Block, primary_mv: Mv, secondary_mv: Mv) -> f32 {
        let _ = (block, primary_mv, secondary_mv);
        0.0
    }

    /// The reference the following [`CodecBackend::mv_candidates`],
    /// [`CodecBackend::mv_seeds`], [`CodecBackend::skip_vector`] and
    /// [`CodecBackend::mode_bits`] calls for `block` mean: their vectors
    /// are predicted from neighbours using that reference, and the mode's
    /// bits include its reference-index syntax. It is
    /// [`ReferenceChoice::Primary`] at the start of every block, and the
    /// engine selects it again before coding a block that predicts from
    /// the primary reference after trying the secondary one. Only called
    /// on backends that support a second reference.
    fn select_reference(&mut self, block: &Block, reference: ReferenceChoice) {
        let _ = (block, reference);
    }

    /// The intra-refresh window of the next frame, or `None` (the default:
    /// no refresh, or a key frame), before [`CodecBackend::begin_frame`].
    /// Backends that choose vectors or intra modes themselves keep them
    /// inside its rules (`refresh.rs`): no above-right samples for intra
    /// blocks on the dirty edge, clean-block vectors allowed by
    /// [`RefreshWindow::allows_area`], and no motion predicted from the
    /// reference's motion field.
    fn set_refresh(&mut self, window: Option<&RefreshWindow>) {
        let _ = window;
    }

    /// Supply the existing analyzer's running sensor-noise estimate once per frame.
    /// Backends that do not use noise adaptation keep the default no-op.
    fn set_noise_sigma(&mut self, _sigma: f32) {}

    /// Re-read the [`CodecBackend::FRAME_PARAM_SPECS`] groups from `params`
    /// for the next frame, before [`CodecBackend::begin_frame`]. The engine
    /// calls it on every frame, and only when a preset gives some parameter
    /// a non-base frame value (`frame_role.rs`), so backends without such
    /// groups keep the default no-op.
    fn set_frame_params(&mut self, params: &crate::ParamOverrides) {
        let _ = params;
    }

    /// log2 of the top-level coding unit: AV1 superblock, HEVC CTU, H.264
    /// macroblock. The engine walks the frame in units of this size.
    fn superblock_log2(&self) -> u8;

    /// log2 of the smallest block the partition walk may produce.
    fn min_block_log2(&self) -> u8;

    /// log2 of the largest leaf the partition walk may produce; larger nodes
    /// always split. For a backend whose transforms do not yet cover every
    /// block size its syntax allows (AV2 codes the chroma transform at the
    /// chroma block size). Default: the superblock, i.e. no limit.
    fn max_leaf_log2(&self) -> u8 {
        self.superblock_log2()
    }

    /// The picture size the codec actually codes for a `width` x `height`
    /// input. Codecs whose syntax can only describe whole units larger than
    /// the minimum block (H.264 macroblocks) code a padded picture and
    /// signal a cropping window; the engine then pads the source by edge
    /// replication, walks the coded size, and crops the reconstruction back.
    /// The default is no padding.
    fn coded_size(&self, width: usize, height: usize) -> (usize, usize) {
        (width, height)
    }

    /// The size of the pictures the engine codes for a stream of `width` x
    /// `height`: the stream size, or smaller for a backend that codes
    /// reduced pictures and has the decoder scale them back normatively
    /// (AV1 superres, §7.16). The engine then resamples each source frame
    /// to this size (`source_scale.rs`) and works at it throughout;
    /// [`CodecBackend::coded_size`] pads it. The default is the stream size.
    fn frame_size(&self, width: usize, height: usize) -> (usize, usize) {
        (width, height)
    }

    /// The decoder's output for the last committed frame when it is not
    /// the engine's reconstruction: the upscaled picture of a backend whose
    /// [`CodecBackend::frame_size`] is smaller than the stream. `None` (the
    /// default) when the reconstruction is the output.
    fn output_picture(&self) -> Option<&Picture> {
        None
    }

    /// Whether inter frames may predict from references of another picture
    /// size (AV1 scaled references, §7.11.3.3), so the engine can change the
    /// coded picture size without a key frame
    /// ([`CodecBackend::set_frame_size`], `Encoder::set_coded_resolution`).
    fn supports_scaled_references(&self) -> bool {
        false
    }

    /// Code the following frames at `width` x `height` (the engine's coded
    /// picture size; the stream size and sequence header stay), predicting
    /// from references of the old size by scaled prediction. `references`
    /// is the reconstruction each engine reference slot holds right now,
    /// at the current size: the engine replaces its own copies with
    /// resampled proxies for its decisions, so the backend keeps these (or
    /// the normative pictures it already holds) as the references it
    /// predicts from until each slot is refreshed. Called only between
    /// frames and only when [`CodecBackend::supports_scaled_references`].
    ///
    /// # Errors
    ///
    /// A [`ConfigError`] when this backend or its configuration cannot
    /// switch sizes (the default).
    fn set_frame_size(
        &mut self,
        width: usize,
        height: usize,
        references: &[Option<&Picture>],
    ) -> Result<(), ConfigError> {
        let _ = (width, height, references);
        Err(ConfigError::Resolution(
            "this codec cannot change the picture size without a key frame",
        ))
    }

    /// Whether this backend can code [`FrameType::Inter`] frames yet. If not,
    /// the engine codes every frame as a key frame.
    fn supports_inter(&self) -> bool {
        false
    }

    /// Test only (`debug.random_decisions`, the syntax-coverage stress test):
    /// randomise the backend's own syntax choices with `rng` (how a vector
    /// is signalled, transform sizes and splits, skip, loop-filter
    /// parameters), keeping every stream legal. Called once, right after
    /// [`CodecBackend::new`], and only when the parameter is set; real
    /// encodes never call it. The default keeps the normal choices.
    fn enable_random_decisions(&mut self, _rng: crate::DebugRng) {}

    /// Whether this backend signals layered reference structures
    /// (`EncoderConfig::temporal_layers` above 1). If not, the engine
    /// rejects such configurations.
    fn supports_temporal_layers(&self) -> bool {
        false
    }

    /// Whether this backend supports spatial scalability (`spatial.layers > 1`).
    fn supports_spatial_layers(&self) -> bool {
        false
    }

    /// Maximum number of spatial layers (`spatial.layers`) this backend can code.
    fn max_spatial_layers(&self) -> u8 {
        if self.supports_spatial_layers() { 2 } else { 1 }
    }

    /// Configure multi-layer spatial scalability once at encoder construction.
    ///
    /// # Errors
    ///
    /// Rejects codec-specific incompatible tools (e.g. `av1.superres_denom > 8`
    /// or `grain.enable` if unsupported with spatial layers).
    fn configure_spatial(
        &mut self,
        config: &crate::spatial::SpatialConfig,
    ) -> Result<(), ConfigError> {
        let _ = config;
        Err(ConfigError::SpatialScalability(
            "not supported by this codec yet",
        ))
    }

    /// Prepare the backend for the next spatial layer picture inside a
    /// multi-layer temporal unit, called immediately before `begin_frame`.
    fn prepare_spatial_layer(&mut self, layer: SpatialLayerFrame) {
        let _ = layer;
    }

    /// Record the committed normative reconstruction of a spatial layer picture
    /// (`size` visible samples, refreshing `refresh`, or all slots when
    /// `refresh_all` is true) so inter-layer scaled references read exact
    /// normative pixels (`AV1 §7.11.3.3`, `AV1 §7.20`).
    fn commit_spatial_reference(
        &mut self,
        spatial_id: u8,
        recon: &Picture,
        size: (usize, usize),
        refresh: nc_analysis::link::SlotMask,
        refresh_all: bool,
    ) {
        let _ = (spatial_id, recon, size, refresh, refresh_all);
    }

    /// Finalize a multi-layer spatial temporal unit's coded bytes (`data`, the
    /// layers' pictures of `layer_lengths` bytes each) once all scheduled
    /// layers of the temporal unit have been encoded (VP9: the Annex B
    /// superframe index, and the picture showing a unit's lone hidden base).
    fn finish_spatial_tu(&mut self, data: &mut Vec<u8>, layer_lengths: &[usize]) {
        let _ = (data, layer_lengths);
    }

    /// log2 of the sub-block side at which this codec's intra prediction
    /// can change direction inside an inter-frame leaf, for the engine's
    /// directional intra estimate (`decision.intra_directional`,
    /// `intra_estimate.rs`). The default gives each leaf (up to 64x64) one
    /// direction.
    fn intra_estimate_log2(&self) -> u8 {
        6
    }

    /// Whole-sample residual variance per sample below which the engine
    /// does not refine a vector to sub-sample precision (the refinement then
    /// rarely pays for its search cost). 0 refines every block that is not
    /// predicted exactly. See `decision::refine_subpel` for measurements.
    fn subpel_skip_variance(&self) -> f32 {
        0.0
    }

    /// The codec's luma interpolation filter. `Some` selects the engine's
    /// SVT light-PD1 candidate flow, which ranks vectors with this filter
    /// (`decision::choose_inter_mode`); `None` keeps the bilinear ranking.
    fn luma_filter(&self) -> Option<nc_dsp::Filter> {
        None
    }

    /// Preserve exact fractional luma candidates for the following block encode.
    /// Other backends retain their existing prediction path at compile time.
    const CACHE_EXACT_LUMA: bool = false;

    /// Save an improving codec-filter candidate, tightly packed at block width.
    /// The reference picture stays unchanged through `encode_block`. Implementors
    /// must key reuse by block/vector and invalidate it when a new frame begins.
    fn cache_exact_luma(
        &mut self,
        _block: &Block,
        _mv: Mv,
        _filter: nc_dsp::Filter,
        _prediction: &[u8],
        _stats: nc_dsp::Variance,
    ) {
    }

    /// Lagrange multiplier at the frame's quantiser: squared-error units per
    /// bit, used to trade prediction error against signalling cost when the
    /// engine ranks modes.
    fn lambda(&self) -> f32 {
        1.0
    }

    /// Vectors the codec can signal cheaply for `block` in an inter frame
    /// (AV1: the NEAREST/NEAR stack and the global vector; HEVC: merge
    /// candidates; H.264: the predicted vector). Appended to `out`; the engine
    /// evaluates them alongside its own motion search result. Called for a
    /// block before [`CodecBackend::mode_bits`] and
    /// [`CodecBackend::encode_block`] of that same block.
    fn mv_candidates(&mut self, block: &Block, out: &mut Vec<Mv>) {
        let _ = (block, out);
    }

    /// Further predicted vectors that only seed sub-sample refinement: the
    /// engine measures each with its cheap (bilinear) distortion beside the
    /// [`CodecBackend::mv_candidates`] vectors and refines the best of them
    /// all, but ranks a seed with the exact filter only if refinement keeps
    /// it. For long predictor lists whose later entries rarely win as they
    /// stand (AV2's DRL). Called right after `mv_candidates`.
    fn mv_seeds(&mut self, block: &Block, out: &mut Vec<Mv>) {
        let _ = (block, out);
    }

    /// Optionally accept an integer predicted vector using already measured
    /// residual statistics, before normal candidate refinement/ranking. The
    /// default preserves full decisions; a backend owns any approximate
    /// confidence policy and validates its cached hint again during coding.
    fn prefer_cached_inter(
        &mut self,
        _block: &Block,
        _mv: Mv,
        _stats: nc_dsp::Variance,
        _lambda: f32,
    ) -> bool {
        false
    }

    /// The frame's quantiser as a fraction of the codec's range (0 finest,
    /// 1 coarsest: AV1 `base_q_idx / 255`, H.264/HEVC `QP / 51`), for the
    /// engine's quantiser-dependent decision thresholds.
    fn relative_quantizer(&self) -> f32 {
        0.5
    }

    /// Estimated mode and vector bits that every inter leaf pays however it
    /// is predicted (the cheapest leaf coding at the current probabilities),
    /// for partition planning: four children pay it four times, so it
    /// weighs against splitting. Codecs whose leaves signal almost nothing
    /// extra return 0 (the default).
    fn planner_leaf_bits(&self) -> f32 {
        0.0
    }

    /// Estimated bits of the partition decision for `node` (split or not),
    /// for partition planning on inter frames. Codecs whose partition syntax
    /// is implicit at this level (H.264 macroblocks) return 0.
    fn partition_bits(&self, node: &Block, split: bool) -> f32 {
        let _ = (node, split);
        0.0
    }

    /// Whether this backend can receive rectangular leaves (it declares
    /// [`CodecBackend::partition_shapes`]). The engine's per-leaf
    /// arithmetic is monomorphised on it: for square-only backends a leaf's
    /// width is its height at compile time, so they keep the square path's
    /// cost (callgrind: the general width/height path cost H.264 +0.35%).
    const RECTANGULAR_LEAVES: bool = false;

    /// The rectangular partitions (`shape.rs`) this backend can code for
    /// square inter-frame `node` as leaves, besides the quadtree split.
    /// The engine plans them only on inter frames, only for nodes wholly
    /// inside the picture, and never under intra refresh. Rectangular
    /// leaves are always predicted inter: the engine never chooses intra
    /// for them. None by default, which keeps the square quadtree; a
    /// backend declaring shapes sets [`CodecBackend::RECTANGULAR_LEAVES`].
    fn partition_shapes(&self, node: &Block) -> crate::ShapeSet {
        let _ = node;
        crate::ShapeSet::EMPTY
    }

    /// Estimated bits of coding square `node` as `shape` (every partition
    /// symbol of the node and its leaves), for partition planning.
    fn shape_bits(&self, node: &Block, shape: crate::PartitionShape) -> f32 {
        let _ = (node, shape);
        0.0
    }

    /// Record that square `node` is partitioned as `shape`; called instead
    /// of [`CodecBackend::write_partition`] for such a node, and followed by
    /// the leaves in [`crate::PartitionShape::parts`] order. Only shapes
    /// from [`CodecBackend::partition_shapes`] are ever written.
    fn write_shape(&mut self, node: &Block, shape: crate::PartitionShape) {
        let _ = (node, shape);
        unreachable!("the engine writes only declared shapes")
    }

    /// Estimated rate-distortion cost of coding key-frame `block` as one
    /// DC-intra leaf at its full transform size, in sample-SSE units with
    /// bits weighted by [`CodecBackend::lambda`]. It is measured on the source
    /// alone (no reconstruction), before the superblock is coded, for
    /// transform-aware key-frame partition planning. Takes `&mut self` only so
    /// a backend can reuse its coefficient scratch; it must not change coded
    /// state. `None` exactly when [`CodecBackend::supports_intra_leaf_cost`]
    /// is false (the default).
    fn intra_leaf_cost(&mut self, source: &Picture, block: &Block) -> Option<f32> {
        let _ = (source, block);
        None
    }

    /// Estimated rate-distortion cost of coding inter-frame `block` as one
    /// leaf predicted from `reference` at `mv`: its luma reconstruction
    /// squared error plus [`CodecBackend::lambda`] times its mode and
    /// coefficient bits, with the entropy state as it stands. For partition
    /// planning (`planner.rect_rd`): it must not change coded state. `None`
    /// (the default) without such an estimate.
    fn inter_leaf_cost(
        &self,
        source: &Picture,
        reference: &Picture,
        block: &Block,
        mv: Mv,
    ) -> Option<f32> {
        let _ = (source, reference, block, mv);
        None
    }

    /// Whether [`CodecBackend::intra_leaf_cost`] is implemented. Without it,
    /// `planner.intra_plan = 2` keeps the configured fixed key-frame size.
    fn supports_intra_leaf_cost(&self) -> bool {
        false
    }

    /// The vector this block would get in the codec's skip mode (H.264
    /// `P_Skip`), if the codec wants the engine's skip shortcut: when that
    /// vector wins candidate ranking with a residual negligible against λ,
    /// sub-sample refinement is not run (see `decision.rs`,
    /// `SKIP_NO_REFINE_LAMBDAS`). `None` (the default) disables it.
    fn skip_vector(&mut self, block: &Block) -> Option<Mv> {
        let _ = block;
        None
    }

    /// The vector a static skip ([`CodecBackend::try_static_skip`]) of
    /// `block` would code, or `None` when the backend takes no static
    /// skips. The default is [`CodecBackend::skip_vector`] (H.264 `P_Skip`).
    fn static_skip_vector(&mut self, block: &Block) -> Option<Mv> {
        self.skip_vector(block)
    }

    /// Whether the [`CodecBackend::encode_block`] just made of inter `block`
    /// left its luma an exact copy of the primary reference at the zero
    /// vector (zero motion, no luma residual), before the loop filters. The
    /// engine's reuse term then keeps the block's measured error instead
    /// of measuring it again (`reuse.rs`). The default reports no copies,
    /// which is always correct, only slower.
    fn copied_reference(&self, block: &Block) -> bool {
        let _ = block;
        false
    }

    /// Offered for inter blocks that pre-analysis marks static (unchanged
    /// since the previous source frame, against a noise-aware bound), before
    /// any mode decision; `still` says whether the stillness detector
    /// (`nc_analysis::still`: static for several frames, away from motion)
    /// also marks the whole block. Return true to code `block` in the
    /// codec's skip mode with its [`CodecBackend::static_skip_vector`], which
    /// the engine then passes to [`CodecBackend::encode_block`] as the inter
    /// vector without measuring candidates. A backend should accept only when its own
    /// residual test of the skip prediction passes, so the shortcut saves
    /// decision work without forcing skips on blocks that need a residual
    /// (the forced static skip removed in `ac3cd9d` hurt clean content).
    /// An accepting backend writes the block's reconstruction into `recon`
    /// here; a declining one may also have written it (the normal path then
    /// overwrites it). The default declines, leaving the normal decision.
    fn try_static_skip(
        &mut self,
        block: &Block,
        source: &Picture,
        reference: &Picture,
        recon: &mut Picture,
        still: bool,
    ) -> bool {
        let _ = (block, source, reference, recon, still);
        false
    }

    /// The motion search's whole-sample vectors of the 8x8 sub-blocks of an
    /// inter `block` (at least 16x16), raster order within the block
    /// (`(size / 8)²` entries; zero where the search has no result). Called
    /// on inter frames right before [`CodecBackend::encode_block`] of that
    /// block, whatever mode the engine chose. Codecs that can split a block's
    /// motion finer than the engine's partition (H.264 16x8 / 8x16 / 8x8
    /// macroblock partitions) use these as candidate vectors; the default
    /// ignores them.
    fn sub_block_motion(&mut self, block: &Block, vectors: &[Mv]) {
        let _ = (block, vectors);
    }

    /// Whether [`CodecBackend::sub_block_motion`] is worth calling: gathering
    /// the vectors walks the motion field per leaf (about 1% of H.264 P-frame
    /// instructions), so the engine skips it for codecs that ignore them.
    fn wants_sub_block_motion(&self) -> bool {
        false
    }

    /// The inter vector `mv` (on the primary reference) that intra beat in
    /// `block` only through the engine's directional intra estimate
    /// (`decision.intra_directional`), which predicts from source samples.
    /// Called after that intra block's mode decision, before its
    /// [`CodecBackend::encode_block`], and only for such blocks. A codec whose own intra search, on
    /// the reconstruction, finds `mv` cheaper may code it instead; the
    /// default ignores it.
    fn intra_fallback(&mut self, block: &Block, mv: Mv) {
        let _ = (block, mv);
    }

    /// Estimated bits to signal `mode` for `block` (mode and vector syntax
    /// only; the residual is not included). Only used for ranking.
    fn mode_bits(&self, block: &Block, mode: &PredictionMode) -> f32 {
        let _ = (block, mode);
        0.0
    }

    /// Whether frame quantizers are owned by rate control rather than fixed-QP boosts.
    fn set_rate_control_enabled(&mut self, _enabled: bool) {}

    /// Quantization step at an actual frame QP/qindex, for bitrate modelling.
    fn rate_control_quantizer_step(&self, quantizer: u32) -> f64 {
        2f64.powf(f64::from(quantizer) / 6.)
    }

    /// The rate model's calibration for this codec's quantizer scale; the
    /// default suits the H.264/HEVC QP scale of the default
    /// [`CodecBackend::rate_control_quantizer_step`].
    fn rate_model(&self) -> crate::RateModel {
        crate::RateModel::QP
    }

    /// Start a frame, resetting entropy state and using the planned quantizer.
    fn begin_frame(&mut self, plan: &FramePlan);

    /// Start the superblock whose top-left luma sample is `(x, y)`.
    fn begin_superblock(&mut self, x: usize, y: usize);

    /// Record the partition decision for `node`: `split` into four square
    /// quadrants or not. Called for every node the walk visits, including
    /// leaves (`split == false`) and nodes whose split is forced by the
    /// frame edge; the backend writes whatever its syntax requires.
    fn write_partition(&mut self, node: &Block, split: bool);

    /// Code one leaf block: predict, transform, quantise, write its syntax
    /// and write the reconstruction of all three planes into `recon`.
    /// `reference` is the reconstruction of the reference slot
    /// `decision.reference` names (borders extended); it is `Some` exactly
    /// on inter frames.
    fn encode_block(
        &mut self,
        block: &Block,
        decision: &BlockDecision,
        source: &Picture,
        reference: Option<&Picture>,
        recon: &mut Picture,
    );

    /// Code one compound-predicted leaf block (`decision.reference` is
    /// [`ReferenceChoice::Compound`]), averaging `primary_reference` (at
    /// `decision.mode`) and `secondary_reference` (at its `secondary_mv`). Only called when
    /// [`CodecBackend::supports_compound_reference`] is true.
    fn encode_compound_block(
        &mut self,
        block: &Block,
        decision: &BlockDecision,
        source: &Picture,
        primary_reference: &Picture,
        secondary_reference: &Picture,
        recon: &mut Picture,
    ) {
        let _ = secondary_reference;
        self.encode_block(block, decision, source, Some(primary_reference), recon);
    }

    /// In-loop filtering of the finished picture, called after the last
    /// block and before [`CodecBackend::end_frame`]. `recon` holds the
    /// unfiltered reconstruction of every block (intra prediction inside
    /// the frame must see unfiltered samples); the backend filters it in
    /// place, so the filtered picture becomes the next frame's reference
    /// (e.g. the H.264 deblocking filter, §8.7). The default does nothing.
    fn loop_filter_frame(&mut self, recon: &mut Picture) {
        let _ = recon;
    }

    /// The engine's in-loop filtering entry point: like
    /// [`CodecBackend::loop_filter_frame`], with the coded source picture
    /// for filters whose strength the encoder chooses by measuring
    /// distortion (AV1 CDEF). The default ignores the source and calls
    /// `loop_filter_frame`.
    fn filter_frame(&mut self, source: &Picture, recon: &mut Picture) {
        let _ = source;
        self.loop_filter_frame(recon);
    }

    /// Finish the frame and return its bitstream unit (one AV1 temporal
    /// unit, or the H.264/HEVC access unit's NAL units in Annex B form).
    /// With [`CodecBackend::RECODE`] this must not commit state that later
    /// frames read: the engine may discard the unit and code the frame
    /// again from [`CodecBackend::begin_frame`].
    fn end_frame(&mut self) -> Vec<u8>;

    /// Whether this backend honours the recode contract: `begin_frame`
    /// derives the frame's state from its plan and committed state only
    /// (so calling it again restarts the same frame), `end_frame` commits
    /// nothing, and every change later frames read happens in
    /// [`CodecBackend::commit_frame`]. Only then may rate control re-code a
    /// frame that would overflow the buffer (`rate.recode_overflow`).
    const RECODE: bool = false;

    /// Commit the state later frames read (reference contexts, counters,
    /// the previous frame's syntax) once the unit `end_frame` returned is
    /// final. Called exactly once per coded frame. Nothing by default.
    fn commit_frame(&mut self) {}
}
