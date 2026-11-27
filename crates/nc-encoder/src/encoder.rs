//! The frame loop: drives a [`CodecBackend`] over every frame and block.
//!
//! Per frame: choose the frame type (a key frame every `intra_period`
//! frames, inter frames in between when the backend supports them) and its
//! place in the temporal structure (`structure.rs`: which reference slot it
//! predicts from, which slots it refreshes), run source analysis (motion
//! search against the reference's *source* frame), walk the partition,
//! decide each leaf's mode, and keep the reconstruction in the slots it
//! refreshes.

use std::cell::{OnceCell, RefCell};

use nc_analysis::motion::{SbSearch, SearchConfig};
use nc_analysis::preanalysis::PreAnalysis;
use nc_analysis::still::StillMap;
use nc_analysis::{Analyzer, FrameAnalysis, MotionAnalysis, NoTimer};
use nc_common::{ExtendedPicture, Picture, PlaneIndex, PlaneView, RawFrame};
use nc_dsp::Dsp;

use crate::aq::{BlockAq, native_offset};
use crate::backend::{
    BlockDecision, CodecBackend, FinalCandidate, FramePlan, FrameType, IntraChoice, PredictionMode,
    ReferenceChoice,
};
use crate::config::{Codec, ConfigError, EncoderConfig, FrameRate};
use crate::cu_pricing;
use crate::decision::{
    Choice, DecisionScratch, PricingHint, choose_compound_mode, choose_inter_mode,
    choose_pair_merge, choose_primary_pair, choose_secondary_mode, is_square_leaf, is_static,
    is_still, leaf_area, leaf_extent, leaf_log2_min_side,
};
use crate::depth_context::DepthContext;
use crate::effort_class::{Classifier, KeyClassifier, LIGHT_PREFIX, LightOverlay, LightParams};
use crate::encoder_info::EncoderInfo;
use crate::feedback::{LossFeedback, ReferencedFrames};
use crate::filter_effort::{CdefEffort, FilterEffort, FrameActivity};
use crate::filter_outcome::{FilterOutcomeStats, OutcomeRole};
use crate::fine_floor::{self, FineFloorGate};
use crate::frame_role::{EffortOverlay, FrameRole, KEY_PREFIX, NONBASE_PREFIX};
use crate::hold::{FrameHold, LastCoded};
use crate::in_frame_rate::{
    FrameStart, InFrameRate, Verdict, superblock_weights, watch_grid, watches_rows,
};
use crate::inheritance::{Inheritance, Tier};
use crate::input::write_raw;
use crate::key_clock::KeyClock;
use crate::key_oracle::{CellAnalysis, with_key_map};
use crate::key_plan::{KeyPlanMode, KeyPlanner};
use crate::labels::{FeatureInputs, FrameContext, FrameLabels, LabelConfig, LabelTool, Labeller};
use crate::layer::LayerPosition;
use crate::layer_propagation::{LayerFrame, Period};
use crate::ltr::{LongTermReference, LtrAction, long_term_slot};
use crate::moving_camera::MovingCamera;
use crate::mv::{MV_UNITS_PER_SAMPLE, Mv};
use crate::noise_skip::{NoiseMap, camera_is_still};
use crate::params::RefParams;
use crate::params::{
    AdaptParams, AnalysisParams, DebugParams, DecisionParams, EffortParams, FilterParams,
    LayerParams, LtrParams, ParamOverrides, PersistParams, PlannerParams, RateParams,
    RefreshParams, ResolutionParams, SceneParams, SpatialParams, StillParams, StructureParams,
    engine_specs,
};
use crate::partition::{Block, should_split};
use crate::partition_plan::{LeafStats, NodeMargins, NoiseTest, Partition, Planner};
use crate::qp_cadence::{
    CadenceRole, LayerLadder, MotionHint, QP_UNITS_PER_OCTAVE, QpCadence, nearest_quantizer,
    step_ratio,
};
use crate::quality_cap::residual_sigma;
use crate::random_decisions::{DebugRng, RandomDecisions};
use crate::rate_control::{
    Complexity, ForcedQuantizer, LayerShare, RateControlConfig, RateControlError, RateControlStats,
    RateController, RateModel, cadence_scale,
};
use crate::references::ReferencePool;
use crate::refinement_gain::luma_mse;
use crate::refresh::{
    IntraRefresh, MIN_REFRESH_PERIOD, RefreshRegion, RefreshSweep, RefreshWindow,
};
use crate::resolution::{CodedScale, ResolutionPolicy, half_size};
use crate::resolution_evidence::HalfSizeProbe;
use crate::reuse::{CodedFrame, ReuseFrame, ReuseTerm};
use crate::scene::{FrameCosts, SceneDetector};
use crate::second_reference::{
    SecondReference, TierPlan, second_reference_enabled, second_reference_steps_aside,
};
use crate::shape::{MAX_SHAPE_PARTS, PartitionShape, ShapeSet};
use crate::source_scale::{SourceScaler, scale_source_picture};
use crate::spatial::SpatialConfig;
use crate::startup_lookahead::{StartupLookahead, startup_scene_profile};
use crate::still_plan::StillPlan;
use crate::still_skip::StaticSkipTest;
use crate::structure::{
    FrameReferences, MAX_TEMPORAL_LAYERS, MotionSignal, TemporalStructure,
    longest_reference_distance,
};
use crate::threshold_control::{FrameEvidence, Switch, ThresholdControl, Tool};
use crate::timeline::{FrameTime, Timeline};
use crate::transplant::{PartitionLog, Transplant};
use leaf_rd::LeafSearch;
use nc_analysis::link::{MAX_REFERENCE_SLOTS, SlotMask};
use rd_split::LeafTally;

#[path = "leaf_rd.rs"]
mod leaf_rd;
#[path = "rd_split.rs"]
mod rd_split;
#[path = "spatial_encode.rs"]
mod spatial_encode;

use spatial_encode::SpatialSession;
pub use spatial_encode::{LayerFrameMetadata, SpatialReferenceStats};

/// Border kept around reconstructions. Prediction clamps reference
/// coordinates itself, so this only needs to cover the codecs' own
/// interpolation reach when they read borders directly.
const RECON_BORDER: usize = 64;
/// 4:2:0 chroma subsampling: visible sizes must be multiples of this.
const CHROMA_SUBSAMPLING: usize = 2;
/// Bounds of the learned searched-to-plan key size ratio
/// (`Encoder::key_search_ratio`): the key searches made keys 3.6-11.4 %
/// smaller at equal quality (docs/analysis/key-frame-rdo-2026-11-21.md
/// sections 3-4), so a ratio outside these is a measurement of something
/// else (a scene the step change moved) and is not learned past them.
const KEY_SEARCH_RATIO_RANGE: (f64, f64) = (0.5, 1.25);
/// 8x8 luma cells darker than this mean count as dark for the depth model
/// (`structure.depth_model`).
const DARK_CELL_MEAN: u8 = 32;

/// What the encoder knows about a frame it produced, without parsing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameMetadata {
    /// Index in coding (= display) order among coded frames: a frame
    /// dropped from a flat chain does not take an index.
    pub index: u64,
    /// Frame type.
    pub frame_type: FrameType,
    /// Temporal layer (0 = base). Frames of the top layer of a layered
    /// structure are never referenced, so transports may drop them.
    pub temporal_layer: u8,
    /// Capture time in microseconds: as given to an `encode_*_at` method,
    /// otherwise one nominal frame duration after the previous frame.
    pub timestamp_us: i64,
    /// The frame was dropped by rate control
    /// ([`RateControlConfig::frame_drop`]): `data` is empty and nothing
    /// must be sent; the decoder keeps showing the previous frame.
    pub dropped: bool,
    /// The frame would not change the picture and was held (`hold.rs`):
    /// `data` is empty, nothing must be sent, and the decoder keeps showing
    /// the previous frame. It takes no place in the stream: `index` is the
    /// next coded frame's.
    pub held: bool,
    /// Requested frame quantizer in the codec's native scale. Fixed-QP keys
    /// may apply a backend quality boost; under rate control this is actual QP.
    pub quantizer: u32,
    /// The coded frames this one predicts from (none for key frames and
    /// frames that are not coded). A transport carries them as its
    /// dependency signal (as the AV1 RTP Dependency Descriptor's frame
    /// diffs do) so a receiver that lost one of them knows not to decode
    /// this frame: AV1, VP9 and AV2 streams carry no frame numbers that
    /// would tell their decoders.
    pub references: ReferencedFrames,
}

/// One coded frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedFrame {
    /// The bitstream unit (AV1 temporal unit / H.264, HEVC access unit).
    pub data: Vec<u8>,
    /// Per-frame metadata for transports and diagnostics.
    pub metadata: FrameMetadata,
    /// Per-spatial-layer picture metadata when `spatial.layers > 1` (empty for
    /// single-layer streams, held frames, and dropped temporal units).
    pub layers: Vec<LayerFrameMetadata>,
}

/// Engine state that belongs to one coded layer: its picture size, buffers,
/// analysis and rate/quantizer adaptation. A single-layer stream has one;
/// spatial SVC has one per spatial layer, the active one in
/// [`Encoder::layer`] and the other parked in [`SpatialSession`] (swapped
/// whole, never field by field). Everything else on [`Encoder`] is shared
/// by all layers or per frame: a new field goes here only if each layer
/// needs its own.
#[derive(Debug)]
pub(super) struct LayerState {
    /// Size of the pictures the engine codes ([`CodecBackend::frame_size`]):
    /// the stream size, or smaller (AV1 superres).
    pub(super) width: usize,
    pub(super) height: usize,
    /// Coded picture size ([`CodecBackend::coded_size`]); the partition walk
    /// covers this, not just the visible picture.
    pub(super) coded_width: usize,
    pub(super) coded_height: usize,
    /// Reconstructions per reference slot plus the frame being coded;
    /// moved out of `self` while a frame is coded (always present between
    /// frames).
    pub(super) references: Option<ReferencePool>,
    /// Present only when the coded size exceeds the visible size: the source
    /// padded to the coded size, and the visible part of the reconstruction.
    pub(super) padding: Option<Padding>,
    /// Source analysis; `None` when the backend codes intra frames only or
    /// the picture is too small for the analysis pyramid (motion search then
    /// falls back to the zero vector).
    pub(super) analyzer: Option<Analyzer>,
    /// Without an analyzer: the buffer [`Encoder::source_mut`] hands out.
    pub(super) input: Option<Picture>,
    /// Rate control, when enabled ([`Encoder::set_rate_control`]).
    pub(super) rate_control: Option<RateController>,
    /// The last high-quality frame as a second reference (`refs.second`).
    pub(super) second: Option<SecondReference>,
    /// How much of a high-quality frame the frames after it keep
    /// (`inheritance.rs`), when the shape or the depth reads it.
    pub(super) inheritance: Option<Inheritance>,
    /// Variance AQ (`rate.aq_strength`), for backends with block-level QP.
    pub(super) aq: Option<BlockAq>,
    /// The reuse term (`persist.reuse`, `reuse.rs`): per block through
    /// block AQ, or one offset per key frame for codecs without block QP.
    pub(super) reuse: Option<ReuseTerm>,
    /// The last frame the reuse term saw (the reference its state describes).
    pub(super) reuse_last_frame: Option<u64>,
    pub(super) base_mc_aq_qp: f64,
    pub(super) key_aq_qp: f64,
    pub(super) last_camera_motion: f64,
    /// Per-layer quantizer offsets of a layered structure: fixed QP.
    pub(super) ladder: LayerLadder,
    /// The same for rate control (`layers.cbr_*`).
    pub(super) rate_control_ladder: LayerLadder,
    /// Pseudo-layer QP offsets of inter frames.
    pub(super) cadence: QpCadence,
    /// Index of the most recent key frame: the origin of the QP cadence.
    pub(super) last_key_index: u64,
    /// Quantizer of the last key or high-quality frame (`is_high_quality`):
    /// the quality held frames build on.
    pub(super) last_quantizer: u32,
    /// Per-cell temporal noise of the still picture, for the noise skip
    /// (`decision.noise_skip`); updated only while that is on.
    pub(super) noise_map: NoiseMap,
    /// Luma area the noise skip took in the frame being coded, for the
    /// skip-aware budget (`rate.skip_budget`).
    pub(super) noise_skipped_area: usize,
}

/// The encoder: configuration, a codec backend and the reconstructions.
#[derive(Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent switches of the encoder and of the frame being coded"
)]
pub struct Encoder<B: CodecBackend> {
    config: EncoderConfig,
    backend: B,
    /// The settings the encoder information message names
    /// (`encoder_info.rs`).
    info: EncoderInfo,
    /// Engine state of the layer being coded: the stream itself, or one
    /// spatial layer while [`SpatialSession`] parks the other.
    layer: LayerState,
    /// Present when the pictures are coded smaller than the stream: the
    /// stream-size input, resampled to `width` x `height` per frame.
    scaler: Option<SourceScaler>,
    /// Adaptive coded resolution (`resolution.adaptive`), under rate
    /// control with a backend that scales references.
    resolution: Option<ResolutionPolicy>,
    /// The measurements `resolution` decides on; present with it.
    resolution_probe: Option<HalfSizeProbe>,
    /// The last frame's reconstruction, kept when a coded-size switch
    /// (`set_coded_resolution`) resampled the pictures it lived in; cleared
    /// by the next frame.
    switched_recon: Option<Picture>,
    /// The reference/GOP manager.
    structure: TemporalStructure,
    /// Pixel kernels for mode decision.
    dsp: Dsp,
    next_index: u64,
    /// Frame timestamps and the intervals rate control budgets.
    timeline: Timeline,
    /// The time of the frame being coded (set before its type is chosen).
    frame_time: FrameTime,
    /// The periodic key schedule, restarted by every unscheduled key
    /// (`key_clock.rs`): the one source of the frames to the next key.
    key_clock: KeyClock,
    /// Quantizer step of every native quantizer, for step-domain QP offsets.
    quantizer_steps: Vec<f64>,
    /// `quantizer_steps` extended past the largest quantizer by
    /// `rate.lambda_headroom_octaves` (see [`FramePlan::lambda_scale`]):
    /// the quantizers rate control may plan.
    rate_steps: Vec<f64>,
    /// `rate.cadence_fixed_qp`: the cadence also applies without rate control.
    cadence_fixed_qp: bool,
    /// The second source frame, handed over before the first is coded
    /// ([`Encoder::look_ahead`]) and consumed by the first frame, unless
    /// `rate.startup_lookahead` turns the lookahead off.
    lookahead: StartupLookahead,
    /// The fixed-QP structure under camera motion (`moving_camera.rs`).
    moving_camera: MovingCamera,
    /// `rate.fixed_qp_key_ln_interval`: QP units finer a fixed-QP key is
    /// planned per unit of the key interval feature (`key_interval.rs`).
    fixed_qp_key_per_ln: f64,
    /// The frame being coded, when it is on the reuse term's chain
    /// ([`on_reuse_chain`]): the term sees only those frames.
    reuse_frame: Option<u64>,
    /// `(expected bits weight, offset in QP units)` of every block coded with AQ this frame:
    /// the realised offsets rate control normalizes its model with.
    block_offsets: Vec<(f64, f64)>,
    /// In-frame rate control (`rate.in_frame`, `in_frame_rate.rs`): the
    /// block offset that steers a frame inside its re-code room.
    in_frame: Option<InFrameRate>,
    /// Area of the leaves coded off the frame quantizer by in-frame rate
    /// control alone, with their offsets in QP units: the realised block
    /// offset rate control learns from when block AQ is off.
    in_frame_offsets: Vec<(f64, f64)>,
    /// The hold of still blocks before a scheduled key (`rate.prekey_quiet`),
    /// for flat single-layer streams with block quantizers and the reuse term.
    still: StillPlan,
    /// Quantizer of the frame being coded (block offsets are relative to it).
    frame_quantizer: u32,
    force_keyframe: bool,
    /// A quantizer imposed on the next frame under rate control
    /// ([`Encoder::force_next_quantizer`]).
    forced_quantizer: Option<ForcedQuantizer>,
    /// QP offsets per 8x8 cell imposed on the next key frame
    /// ([`Encoder::set_next_key_offsets`], `key_oracle.rs`).
    key_offsets: Option<Vec<f64>>,
    /// The partition transplant oracle, when a caller asked for it
    /// (`transplant.rs`).
    transplant: Option<Transplant>,
    /// Long-term reference with feedback (`ltr.interval`), when enabled.
    ltr: Option<LongTermReference>,
    /// Receiver feedback: slot contents and repairs (`feedback.rs`).
    feedback: LossFeedback,
    /// Intra-refresh sweep, when enabled ([`Encoder::set_intra_refresh`]).
    refresh: Option<RefreshSweep>,
    /// The refresh window of the frame being coded.
    window: Option<RefreshWindow>,
    /// Scene-cut detection (`scene.detect`), when enabled.
    scene: Option<SceneDetector>,
    /// Frame hold (`hold.rs`), when enabled.
    hold: Option<FrameHold>,
    /// Whether the frame being coded is a repeat frame (`hold.code_skip`,
    /// `hold.rs` "Repeat frames"): a held frame coded as a copy of its
    /// reference.
    repeating: bool,
    /// Scratch list of candidate vectors, reused for every inter block.
    decision_scratch: DecisionScratch,
    /// The planner's margins of the inter superblock being coded, the RD
    /// partition search's gate (`rd_split.rs`).
    plan_margins: Option<Box<NodeMargins>>,
    /// The coded leaf sizes the RD partition search's context gate reads
    /// (`depth_context.rs`).
    depths: DepthContext,
    /// Leaves the RD partition search coded and how many were skips, for
    /// `planner.rd_split_merge_skipped` (`rd_split.rs`).
    rd_leaf_tally: LeafTally,
    /// The label dump (`labels.rs`), while on.
    labels: Option<Box<Labeller>>,
    /// Whether key leaves are searched as they are coded (`leaf_rd.rs`).
    leaf_search: LeafSearch,
    /// The committed frames' filter-search outcomes (`filter_outcome.rs`).
    filter_outcomes: FilterOutcomeStats,
    /// Tunable partition-planner parameters.
    planner_params: PlannerParams,
    /// Tunable mode-decision parameters.
    decision_params: DecisionParams,
    /// `decision_params` as adjusted for the frame being coded (cadence).
    frame_decision_params: DecisionParams,
    /// Cheaper settings for non-base frames (`nonbase.*`, `frame_role.rs`);
    /// `None` without such settings.
    effort_overlay: Option<EffortOverlay>,
    /// Light-superblock settings (`light.*`, `effort_class.rs`); `None`
    /// while `effort.regular_fraction` is 1.
    light_overlay: Option<LightOverlay>,
    /// The light superblocks of the frame being coded and their engine
    /// parameters; `None` when it has none.
    frame_light: Option<FrameLight>,
    /// The benefit proxy's settings (`effort.*`), for the frame-level
    /// filter decision as well as the superblock classes.
    effort_params: EffortParams,
    /// Frame-level loop-filter effort (`filter.*`, `filter_effort.rs`).
    filter_params: FilterParams,
    /// The last coded frame's skipped share of its area, from its backend.
    previous_skip_share: Option<f32>,
    /// The last committed frame's relative quantizer (`0..=1`), for high-rate
    /// gates (`decision.nonbase_min_quantizer`).
    last_relative_quantizer: Option<f32>,
    /// Consecutive frames, up to the last, whose CDEF search the filter
    /// effort skipped.
    cdef_reused: u32,
    /// Adaptive tool thresholds (`adapt.*`, `threshold_control.rs`);
    /// `None` while every tool is off.
    threshold_control: Option<ThresholdControl>,
    /// The stream's analysis settings when its motion exits carry linear
    /// shifts (`analysis.{zero_gate,window_exit}_shift_*`), applied to each
    /// frame's search settings; `None` while every shift term is 0.
    motion_shifts: Option<AnalysisParams>,
    /// `debug.recode_identity`: every frame is coded twice (recode test).
    recode_identity: bool,
    /// How a frame's first coding may stop: halfway under
    /// `debug.stop_identity` (the test of stopped codings), else by in-frame
    /// rate control.
    first_stop: StopRule,
    /// `rate.key_plan_first` with `rate.key_recode_miss`: a key with RD
    /// searches is coded first without them, to learn its size. It holds
    /// the last searched key's size over its plan coding's at the same
    /// step (1 before any): the plan coding's size times it estimates the
    /// searched key's. `None` while off.
    key_search_ratio: Option<f64>,
    /// `debug.random_decisions`: random legal decisions replace the normal
    /// ones (syntax-coverage stress test). `None` in every real encode.
    random: Option<RandomDecisions>,
    /// Multi-layer spatial scalability session (`spatial.layers == 2`);
    /// `None` on every single-layer stream.
    spatial: Option<Box<SpatialSession>>,
}

/// The light superblocks of one frame (`effort_class.rs`).
#[derive(Debug, Clone)]
struct FrameLight {
    /// The engine parameters light superblocks are coded with.
    params: LightParams,
    /// Whether each superblock, in raster order, is light.
    map: Vec<bool>,
}

/// Buffers needed when the codec codes a padded picture.
#[derive(Debug)]
pub(super) struct Padding {
    source: Option<Picture>,
    visible_recon: OnceCell<Picture>,
    spare_recon: RefCell<Option<Picture>>,
}

impl Padding {
    fn copy_source(&mut self, source: &Picture, (width, height): (usize, usize)) -> &Picture {
        let padded = self.source.get_or_insert_with(|| {
            Picture::new(width, height, 0).expect("the coded dimensions were validated")
        });
        copy_padded(source, padded);
        padded
    }

    /// Bitstream-only callers never need the cropped copy. A caller asking
    /// for reconstruction gets the same picture until the next coded frame.
    fn reconstruction(&self, dsp: &Dsp, coded: &Picture) -> &Picture {
        self.visible_recon.get_or_init(|| {
            let mut visible = self
                .spare_recon
                .take()
                .expect("the crop buffer is available");
            copy_visible(dsp, coded, &mut visible);
            visible
        })
    }

    fn reconstructed(&mut self) {
        if let Some(visible) = self.visible_recon.take() {
            *self.spare_recon.get_mut() = Some(visible);
        }
    }
}

/// The analysis has already replicated its source edges. Expose those
/// samples for coding when they cover the padded area, then restore the
/// original dimensions before that frame can serve as an analysis reference.
enum CodedSource<'a> {
    Picture(&'a Picture),
    Extended(ExtendedPicture<'a, u8>),
}

impl core::ops::Deref for CodedSource<'_> {
    type Target = Picture;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Picture(picture) => picture,
            Self::Extended(picture) => picture,
        }
    }
}

fn coded_source<'a>(
    analyzed: Option<&'a mut Picture>,
    input: Option<&'a Picture>,
    padding: Option<&'a mut Padding>,
    (width, height): (usize, usize),
) -> CodedSource<'a> {
    match (analyzed, padding) {
        (Some(source), Some(_)) if source.can_extend_to(width, height) => {
            CodedSource::Extended(source.extended(width, height))
        }
        (analyzed, padding) => {
            let source = analyzed
                .map(|picture| &*picture)
                .or(input)
                .expect("the encoder owns a source buffer");
            CodedSource::Picture(match padding {
                Some(padding) => padding.copy_source(source, (width, height)),
                None => source,
            })
        }
    }
}

fn analyzed_frame(
    analyzer: Option<&mut Analyzer>,
) -> (Option<&mut Picture>, Option<&FrameAnalysis>) {
    match analyzer {
        Some(analyzer) => {
            let (source, analysis) = analyzer.frame_and_analysis();
            (Some(source), analysis)
        }
        None => (None, None),
    }
}

impl<B: CodecBackend> Encoder<B> {
    /// Create an encoder for `config`.
    ///
    /// # Errors
    ///
    /// A [`ConfigError`] if the engine or the backend rejects the settings.
    #[allow(clippy::too_many_lines)]
    pub fn new(mut config: EncoderConfig) -> Result<Self, ConfigError> {
        let caller_params = config.params.clone();
        config.params =
            rate_mode_settings(&config.params.layered_over(&backend_defaults::<B>(&config)));
        crate::constant_qp::apply(&mut config);
        config.temporal_layers = configured_temporal_layers(&config);
        let info = EncoderInfo::new(&config, &caller_params);
        validate_stream(&config)?;
        let (effort_overlay, light_overlay) = validated_params::<B>(&config.params)?;
        let planner_params = PlannerParams::from_overrides(&config.params);
        let decision_params = DecisionParams::from_overrides(&config.params);
        let effort_params = EffortParams::from_overrides(&config.params);
        let filter_params = FilterParams::from_overrides(&config.params);
        let threshold_control = ThresholdControl::new(AdaptParams::from_overrides(&config.params));
        let motion_shifts = Some(AnalysisParams::from_overrides(&config.params))
            .filter(AnalysisParams::has_motion_shifts);
        let mut backend = B::new(&config)?;
        check_temporal_layers(&config, &backend)?;
        let spatial_cfg =
            SpatialConfig::from_encoder_config_max(&config, backend.max_spatial_layers())?;
        if let Some(ref scfg) = spatial_cfg {
            backend.configure_spatial(scfg)?;
        }
        let structure = temporal_structure(&config);
        let multiple = 1usize << backend.min_block_log2();
        let (picture, (coded_width, coded_height)) = picture_geometry(&config, &backend)?;
        let (width, height) = (picture.width, picture.height);
        let bad_dims = |_| ConfigError::Dimensions {
            width: config.width,
            height: config.height,
            multiple,
        };
        let ltr = long_term_reference(&config, &backend)?;
        let second = second_reference(&config, &backend, ltr.is_some())?;
        let (quantizer_steps, rate_steps) = quantizer_step_tables(&backend, &config);
        let cadence = configured_cadence(&config, &backend);
        let (qp_log2, qp_offsets) = backend.block_qp_grid();
        let aq = BlockAq::new(&config.params, qp_log2, qp_offsets).filter(|_| B::SUPPORTS_BLOCK_QP);
        let persist = PersistParams::from_overrides(&config.params);
        let reuse = (persist.reuse >= 0.5).then(|| {
            ReuseTerm::new(B::SUPPORTS_BLOCK_QP)
                .level_priced(persist.reuse_level_price >= 0.5)
                .key_cells(f64::from(persist.reuse_keys))
        });
        let cadence_fixed_qp = RateParams::from_overrides(&config.params).cadence_fixed_qp >= 0.5;
        let lookahead = StartupLookahead::new(&RateParams::from_overrides(&config.params));
        let moving_camera = MovingCamera::new(
            &RateParams::from_overrides(&config.params),
            config.key_qp_offset.is_some(),
        );
        let fixed_qp_key_per_ln =
            f64::from(RateParams::from_overrides(&config.params).fixed_qp_key_ln_interval);
        let scene = SceneDetector::new(&SceneParams::from_overrides(&config.params));
        let use_aq = aq.is_some();
        let still = StillPlan::new(
            &RateParams::from_overrides(&config.params),
            f64::from(config.frame_rate.num) / f64::from(config.frame_rate.den),
            spatial_cfg.is_none() && use_aq && reuse.is_some(),
        );
        let (spatial, layer) = if let Some(scfg) = spatial_cfg {
            let (session, enhancement) =
                SpatialSession::new(&config, &backend, scfg, (use_aq, reuse.is_some()))?;
            (Some(Box::new(session)), enhancement)
        } else {
            let slots = pool_slots(&structure, &config, ltr.is_some() || second.is_some());
            let (references, padding) =
                frame_buffers(&picture, (coded_width, coded_height), slots).map_err(bad_dims)?;
            let analyzer = backend
                .supports_inter()
                .then(|| source_analyzer(&picture, slots, structure.reads_repeats()))
                .flatten();
            let input = analyzer
                .is_none()
                .then(|| Picture::new(width, height, 0))
                .transpose()
                .map_err(bad_dims)?;
            let layer_params = LayerParams::from_overrides(&config.params);
            let layer = LayerState {
                width,
                height,
                coded_width,
                coded_height,
                references: Some(references),
                padding,
                analyzer,
                input,
                rate_control: None,
                second,
                inheritance: Inheritance::new(
                    &layer_params,
                    &StructureParams::from_overrides(&config.params),
                ),
                aq,
                reuse,
                reuse_last_frame: None,
                base_mc_aq_qp: f64::from(RateParams::from_overrides(&config.params).base_mc_aq_qp),
                key_aq_qp: f64::from(RateParams::from_overrides(&config.params).key_aq_qp),
                last_camera_motion: 0.,
                ladder: layer_params.fixed_ladder(config.temporal_layers),
                rate_control_ladder: layer_params.rate_control_ladder(config.temporal_layers),
                cadence,
                last_key_index: 0,
                noise_map: NoiseMap::new(),
                noise_skipped_area: 0,
                last_quantizer: 0,
            };
            (None, layer)
        };
        let (scaler, resolution) = if spatial.is_some() {
            (None, None)
        } else {
            let scaler = SourceScaler::between((config.width, config.height), (width, height))
                .map_err(bad_dims)?;
            let resolution = adaptive_resolution(&config, &backend, (width, height))?;
            (scaler, resolution)
        };
        let resolution_probe = match resolution {
            Some(_) => Some(
                HalfSizeProbe::new(
                    (config.width, config.height),
                    half_size(config.width, config.height),
                )
                .map_err(bad_dims)?,
            ),
            None => None,
        };
        let timeline = Timeline::new(config.frame_rate);
        let key_clock = KeyClock::new(timeline.nominal_us());
        let debug = DebugParams::from_overrides(&config.params);
        let random = RandomDecisions::from_seed(debug.random_decisions);
        let recode_identity = debug.recode_identity >= 0.5;
        let first_stop = if debug.stop_identity >= 0.5 && B::RECODE && random.is_none() {
            StopRule::Halfway
        } else {
            StopRule::InFrame
        };
        let rate_params = RateParams::from_overrides(&config.params);
        let key_search_ratio =
            (rate_params.key_plan_first >= 0.5 && rate_params.key_recode_miss > 0.).then_some(1.);
        // Steered frames need block QP syntax; random decisions choose
        // their own offsets, and spatial layers share one buffer that this
        // per-layer steering does not see.
        let in_frame = InFrameRate::new(&rate_params)
            .filter(|control| !control.steers() || B::SUPPORTS_BLOCK_QP)
            .filter(|_| random.is_none() && spatial.is_none());
        let steers = in_frame.as_ref().is_some_and(InFrameRate::steers);
        enable_backend_options(&mut backend, random.as_ref(), use_aq || steers);
        backend.set_encoder_info(info.payload(config.quantizer));
        let encoder = Self {
            info,
            recode_identity,
            first_stop,
            key_search_ratio,
            hold: FrameHold::new(&config),
            repeating: false,
            config,
            backend,
            layer,
            scaler,
            resolution,
            resolution_probe,
            switched_recon: None,
            structure,
            dsp: Dsp::new(),
            next_index: 0,
            timeline,
            frame_time: FrameTime {
                timestamp_us: 0,
                interval: None,
                timed: false,
            },
            key_clock,
            quantizer_steps,
            rate_steps,
            cadence_fixed_qp,
            lookahead,
            moving_camera,
            fixed_qp_key_per_ln,
            reuse_frame: None,
            block_offsets: Vec::new(),
            in_frame,
            in_frame_offsets: Vec::new(),
            still,
            frame_quantizer: 0,
            force_keyframe: false,
            forced_quantizer: None,
            key_offsets: None,
            transplant: None,
            ltr,
            feedback: LossFeedback::default(),
            refresh: None,
            window: None,
            scene,
            decision_scratch: DecisionScratch::default(),
            plan_margins: None,
            depths: DepthContext::default(),
            rd_leaf_tally: LeafTally::default(),
            labels: None,
            leaf_search: LeafSearch::default(),
            filter_outcomes: FilterOutcomeStats::default(),
            planner_params,
            decision_params,
            frame_decision_params: decision_params,
            effort_overlay,
            light_overlay,
            frame_light: None,
            effort_params,
            filter_params,
            previous_skip_share: None,
            last_relative_quantizer: None,
            cdef_reused: 0,
            threshold_control,
            motion_shifts,
            random,
            spatial,
        };
        encoder.with_parameter_refresh()
    }

    /// This encoder with the intra refresh `refresh.period` asks for.
    fn with_parameter_refresh(mut self) -> Result<Self, ConfigError> {
        let period = RefreshParams::from_overrides(&self.config.params)
            .period
            .round() as u32;
        if period > 0 {
            self.set_intra_refresh(Some(IntraRefresh { period }))?;
        }
        Ok(self)
    }

    /// The configuration in use.
    #[must_use]
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// The reconstruction of the most recently encoded frame: exactly what a
    /// conforming decoder outputs for it (the visible picture, after any
    /// cropping).
    #[must_use]
    pub fn reconstruction(&self) -> &Picture {
        if let Some(picture) = self.reconstruction_override() {
            return picture;
        }
        let coded = self
            .layer
            .references
            .as_ref()
            .expect("references are present between frames")
            .last();
        match &self.layer.padding {
            Some(padding) => padding.reconstruction(&self.dsp, coded),
            None => coded,
        }
    }

    /// Visible size of the last reconstruction, without copying its pixels.
    #[must_use]
    pub fn reconstruction_size(&self) -> (usize, usize) {
        self.reconstruction_override()
            .map_or((self.layer.width, self.layer.height), |picture| {
                (picture.width(), picture.height())
            })
    }

    fn reconstruction_override(&self) -> Option<&Picture> {
        self.spatial
            .as_deref()
            .map(SpatialSession::output_reconstruction)
            .or_else(|| self.backend.output_picture())
            .or(self.switched_recon.as_ref())
    }

    /// Enable or update rate control without resetting references or learned rate models.
    /// # Errors
    /// Rejects invalid targets, buffers or actual codec quantizer bounds.
    pub fn set_rate_control(&mut self, config: RateControlConfig) -> Result<(), RateControlError> {
        if self.spatial.is_some() {
            self.set_spatial_rate_control(config)?;
        } else {
            self.set_layer_rate_control(config)?;
        }
        self.info.set_rate(Some(config));
        self.send_encoder_info();
        Ok(())
    }

    /// [`Encoder::set_rate_control`] of a single-layer stream.
    fn set_layer_rate_control(
        &mut self,
        config: RateControlConfig,
    ) -> Result<(), RateControlError> {
        if let Some(controller) = self.layer.rate_control.as_mut() {
            controller.reconfigure(config)?;
        } else {
            let steps = self.rate_steps.clone();
            let model = self.backend.rate_model();
            let headroom = (self.rate_steps.len() - self.quantizer_steps.len()) as u32;
            let model = RateModel {
                max_quantizer: model.max_quantizer + headroom,
                ..model
            };
            self.layer.rate_control = Some(RateController::new(
                config,
                model,
                self.config.frame_rate,
                u32::try_from(self.key_period()).unwrap_or(u32::MAX),
                self.config.quantizer,
                steps,
                self.layer.width * self.layer.height,
                &RateParams::from_overrides(&self.config.params),
                self.layer.rate_control_ladder,
            )?);
        }
        self.backend.set_rate_control_enabled(true);
        Ok(())
    }

    /// Restore fixed-QP operation and its configured keyframe quality boost.
    pub fn disable_rate_control(&mut self) {
        if self.spatial.is_some() {
            self.disable_spatial_rate_control();
        } else {
            self.layer.rate_control = None;
            self.backend.set_rate_control_enabled(false);
        }
        self.info.set_rate(None);
        self.send_encoder_info();
    }

    /// Hand the backend the encoder information message of the settings
    /// now in force, for the next key frame on.
    fn send_encoder_info(&mut self) {
        self.backend
            .set_encoder_info(self.info.payload(self.config.quantizer));
    }

    /// What the frame-level filter searches decided on the committed
    /// single-layer frames, by tool and frame role (`filter_outcome.rs`).
    #[must_use]
    pub fn filter_outcome_stats(&self) -> &FilterOutcomeStats {
        &self.filter_outcomes
    }

    /// Packet feedback and VBV statistics, or `None` in fixed-QP mode.
    #[must_use]
    pub fn rate_control_stats(&self) -> Option<RateControlStats> {
        if let Some(spatial) = self.spatial.as_ref() {
            return spatial.combined_rate_control_stats(&self.layer);
        }
        self.layer.rate_control.as_ref().map(RateController::stats)
    }

    /// Code the next frame at `forced` instead of the quantizer rate
    /// control plans for it (`None`: the plan), without re-coding it; the
    /// controller otherwise runs as usual. An oracle for measuring a
    /// quantizer sequence (`nanocodec encode --qp-replay`), not a control:
    /// the target is not held. It applies to the next frame planned under
    /// rate control (single-layer streams), so a caller replaying a
    /// sequence sets it before every input frame.
    pub fn force_next_quantizer(&mut self, forced: Option<ForcedQuantizer>) {
        self.forced_quantizer = forced;
    }

    /// Impose `offsets` (QP units per 8x8 luma cell, raster order) on the
    /// next frame if it is a key, on top of the block AQ terms; `None`
    /// clears it. An oracle for measuring a key's bit allocation
    /// (`key_oracle.rs`, `nanocodec encode --key-offsets`), not a control: a
    /// caller sets it before the frame it is for. Backends without block
    /// quantizers, and maps of another size, ignore it.
    pub fn set_next_key_offsets(&mut self, offsets: Option<Vec<f64>>) {
        self.key_offsets = offsets;
    }

    /// Record the partition coded in every inter superblock from now on
    /// (`transplant.rs`, `nanocodec encode --partition-record`): an oracle
    /// for measuring a partition guide, not a control. Read it back with
    /// [`Encoder::take_partition_log`].
    pub fn record_partitions(&mut self) {
        self.transplant = Some(Transplant::Record(PartitionLog::default()));
    }

    /// Code `log`'s partitions in place of the plan in the inter superblocks
    /// it holds (`transplant.rs`, `nanocodec encode --partition-replay`): an
    /// oracle for measuring a partition guide, not a control. Those
    /// superblocks skip the RD partition search.
    pub fn replay_partitions(&mut self, log: PartitionLog) {
        self.transplant = Some(Transplant::Replay(log));
    }

    /// The partitions recorded since [`Encoder::record_partitions`], ending
    /// the recording; `None` when not recording.
    pub fn take_partition_log(&mut self) -> Option<PartitionLog> {
        match self.transplant.take() {
            Some(Transplant::Record(log)) => Some(log),
            other => {
                self.transplant = other;
                None
            }
        }
    }

    /// The analysis of the last frame the analyzer saw, per 8x8 luma cell:
    /// a diagnostic for the key-frame inheritance oracle (`key_oracle.rs`,
    /// `nanocodec encode --analysis-log`). `None` without an analyzer.
    #[must_use]
    pub fn last_cell_analysis(&self) -> Option<CellAnalysis> {
        self.layer
            .analyzer
            .as_ref()
            .and_then(Analyzer::analysis)
            .map(CellAnalysis::of)
    }

    /// Record decision labels from the next frame on (`labels.rs`): every
    /// superblock's inputs and coded cost, every searched node's candidate
    /// costs. Leaves the stream unchanged; costs encode time.
    ///
    /// # Errors
    ///
    /// A tool overriding a backend parameter the backend cannot change per
    /// superblock (not in [`CodecBackend::SUPERBLOCK_PARAMS`]), or a name no
    /// parameter has.
    pub fn enable_labels(&mut self, config: LabelConfig) -> Result<(), String> {
        let specs = crate::params::engine_specs();
        for tool in &config.tools {
            tool.overrides
                .validate(&[specs.as_slice(), B::PARAM_SPECS].concat())
                .map_err(|e| format!("label tool {}: {e}", tool.name))?;
            let (_, backend) = tool.split();
            if let Some(name) = backend.names().find(|n| !B::SUPERBLOCK_PARAMS.contains(n)) {
                return Err(format!(
                    "label tool {}: {name} cannot change per superblock in this codec",
                    tool.name
                ));
            }
        }
        self.labels = Some(Box::new(Labeller::new(config)));
        Ok(())
    }

    /// The labels of the frames coded since the last call (none while the
    /// dump is off).
    pub fn take_labels(&mut self) -> Vec<FrameLabels> {
        self.labels
            .as_mut()
            .map_or_else(Vec::new, |l| std::mem::take(&mut l.done))
    }
    /// Request an intra refresh on the next input frame, without resetting RC history.
    pub fn request_keyframe(&mut self) {
        self.force_keyframe = true;
    }

    /// `motion`: the frame's analysis, whose temporal noise only rate
    /// control reads (measured here, so fixed QP skips it).
    fn frame_plan(
        &mut self,
        index: u64,
        frame_type: FrameType,
        references: FrameReferences,
        pre: Option<&PreAnalysis>,
        motion: Option<&MotionAnalysis>,
        key_reuse_offset: f64,
    ) -> FramePlan {
        self.frame_plan_with_rc_key(
            index,
            frame_type,
            frame_type == FrameType::Key,
            references,
            pre,
            motion,
            key_reuse_offset,
            &LayerShare::WHOLE,
        )
    }

    #[expect(
        clippy::too_many_lines,
        reason = "build the frame plan in coding order"
    )]
    fn frame_plan_with_rc_key(
        &mut self,
        index: u64,
        frame_type: FrameType,
        rc_key: bool,
        references: FrameReferences,
        pre: Option<&PreAnalysis>,
        motion: Option<&MotionAnalysis>,
        key_reuse_offset: f64,
        share: &LayerShare,
    ) -> FramePlan {
        let cadence_motion = motion
            .and_then(|m| cadence_motion_hint(m, references.reference.distance, self.layer.width));
        if index == 0 {
            self.layer.last_camera_motion = 0.;
        } else if let Some(m) = cadence_motion {
            self.layer.last_camera_motion = m.camera_motion;
        }
        let depth = self.inherited_depth(index, cadence_motion);
        let cadence_base = self.is_cadence_base(index, references.position);
        let quantizer = if let Some(controller) = self.layer.rate_control.as_mut() {
            controller.set_cadence_depth(depth);
            // Reuse this frame's existing8x8 statistics; no lookahead/pixel pass.
            let mut hint = Complexity::from_preanalysis(pre);
            if controller.wants_temporal_noise() {
                hint.residual_sigma = motion.and_then(|m| residual_sigma(&m.fine));
            }
            hint.motion = cadence_motion;
            hint.key_reuse_offset = key_reuse_offset;
            if !rc_key
                && index > 2
                && self.forced_quantizer.is_none()
                && self.layer.base_mc_aq_qp > 0.
                && cadence_base
                && let (Some(pre), Some(motion), Some(aq)) = (pre, motion, self.layer.aq.as_mut())
            {
                aq.apply_base_mc_concentration(
                    pre,
                    motion,
                    self.layer.base_mc_aq_qp,
                    hint.motion.map_or(0.0, |m| m.camera_motion),
                );
            }
            if let Some(aq) = self.layer.aq.as_mut().filter(|aq| aq.active()) {
                if aq.reuse_only() {
                    // The cells' bit weights summed (`BlockAq::set_total_weight`).
                    let previous = pre.is_some_and(|p| {
                        p.blocks
                            .cells()
                            .first()
                            .is_some_and(|b| b.sad_prev.is_some())
                    });
                    aq.set_total_weight(if previous {
                        hint.temporal
                    } else {
                        hint.spatial
                    });
                }
                // A frame holding its still blocks is planned as without
                // their offsets: the held blocks are copies.
                if !self.still.without_offsets {
                    hint.block_qp_offset =
                        controller.effective_block_offset(aq.total_weight(), aq.offset_cells());
                }
            }
            let mut quantizer = controller.select(rc_key, index, hint, references.position, share);
            let forced = self.forced_quantizer.take();
            if let Some(forced) = forced {
                quantizer = controller.force(forced);
            }
            if !rc_key
                && forced.is_none()
                && let Some((refine_qp, from, full, mv_sh, mv_sc, cam_sh, cam_sc, max_cam, pass)) =
                    controller.first_key_mc_block_refine(quantizer)
                && let (Some(pre), Some(motion), Some(aq)) = (pre, motion, self.layer.aq.as_mut())
                && let Some(mean_qp) = aq.apply_first_key_mc_refine(
                    pre,
                    motion,
                    refine_qp,
                    from,
                    full,
                    mv_sh,
                    mv_sc,
                    cam_sh,
                    cam_sc,
                    hint.motion.map_or(0.0, |m| m.camera_motion).max(max_cam),
                    pass,
                )
            {
                quantizer = controller.apply_first_key_mc_frame_shift(mean_qp);
            }
            quantizer
        } else if self.config.quantizer < self.quantizer_bounds().0 {
            // A quantizer below the lossy range (qindex 0: lossless in VP9
            // and AV2) is never offset.
            self.config.quantizer
        } else if frame_type == FrameType::Key {
            // Keys keep the configured (T0) quantizer, coarser under camera
            // motion (`MovingCamera`) and finer by the key interval term;
            // backends apply their own key boost relative to it.
            let offset = self.moving_camera.key_offset() - self.fixed_qp_key_finer();
            if offset == 0. {
                self.config.quantizer
            } else {
                let step =
                    self.quantizer_steps[self.config.quantizer as usize] * step_ratio(offset);
                nearest_quantizer(&self.quantizer_steps, self.quantizer_bounds(), step)
            }
        } else if !references.position.is_flat() {
            let step = self.quantizer_steps[self.config.quantizer as usize]
                * step_ratio(self.layer.ladder.offset(references.position));
            nearest_quantizer(&self.quantizer_steps, self.quantizer_bounds(), step)
        } else if self.cadence_fixed_qp && !self.layer.cadence.is_flat() {
            let strength = self.own_cadence_strength(
                pre.map_or(0., |p| f64::from(p.static_fraction)),
                cadence_motion,
            );
            let step = self.quantizer_steps[self.config.quantizer as usize]
                * step_ratio(
                    self.layer
                        .cadence
                        .offset(index - self.layer.last_key_index, strength),
                );
            nearest_quantizer(&self.quantizer_steps, self.quantizer_bounds(), step)
        } else {
            self.config.quantizer
        };
        let (quantizer, lambda_scale) = self.coded_quantizer(quantizer);
        FramePlan {
            index,
            frame_type,
            quantizer,
            references,
            lambda_scale,
            block_qp: self.random.is_some()
                || self.active_aq().is_some()
                || self.in_frame_at_risk(frame_type == FrameType::Key, quantizer),
        }
    }

    /// Whether in-frame rate control may steer a frame (`key` or inter)
    /// planned at `quantizer`: its block-QP syntax is then coded
    /// (`FramePlan::block_qp`), and only then, as AV1 pays a `delta_q`
    /// symbol per superblock for it.
    fn in_frame_at_risk(&self, key: bool, quantizer: u32) -> bool {
        let (Some(control), Some(controller)) =
            (self.in_frame.as_ref(), self.layer.rate_control.as_ref())
        else {
            return false;
        };
        let bounds = controller.in_frame_bounds();
        let step = self.quantizer_steps[quantizer as usize];
        let sb_log2 = self.backend.superblock_log2();
        let (columns, rows) = watch_grid(self.layer.coded_width, self.layer.coded_height, sb_log2);
        control.steers_watched(key, (bounds.predicted, bounds.room), step, columns * rows)
    }

    /// The coded quantizer and [`FramePlan::lambda_scale`] of a planned
    /// quantizer: one past the codec's largest is coded at the largest,
    /// with λ grown by the squared step ratio.
    fn coded_quantizer(&self, planned: u32) -> (u32, f32) {
        let coded = planned.min(max_quantizer(self.config.codec));
        let step_ratio = self.rate_steps[planned as usize] / self.rate_steps[coded as usize];
        (coded, (step_ratio * step_ratio) as f32)
    }

    /// The codec's lossy quantizer range.
    fn quantizer_bounds(&self) -> (u32, u32) {
        (
            min_quantizer(self.config.codec),
            max_quantizer(self.config.codec),
        )
    }

    /// Frame `index`'s type: a key frame every `intra_period` frames, on
    /// request, or always if the backend has no inter frames.
    fn frame_type(&mut self, index: u64) -> FrameType {
        let forced = std::mem::take(&mut self.force_keyframe);
        if !forced && self.backend.supports_inter() && !self.periodic_key(index) {
            FrameType::Inter
        } else {
            FrameType::Key
        }
    }

    /// Whether frame `index` is a scheduled key frame: every `intra_period`
    /// frames from the last key, or only the first while intra refresh
    /// replaces periodic keys (`key_clock.rs`). A timestamped frame follows
    /// time instead: a live source's skipped captures do not stretch the key
    /// interval; untimed frames keep the frame-count schedule exactly.
    fn periodic_key(&self, index: u64) -> bool {
        self.key_clock
            .is_due(index, self.schedule_time(), self.key_period())
    }

    /// The frame being planned's capture time when it is timestamped (the
    /// key schedule then follows time).
    fn schedule_time(&self) -> Option<i64> {
        self.frame_time
            .timed
            .then_some(self.frame_time.timestamp_us)
    }

    /// Block AQ, when the frame being coded has any non-zero offset: a
    /// frame whose offsets are all zero (most frames with only the reuse
    /// term on) codes, budgets and ranks exactly as without block AQ, and
    /// skips the per-block offset work (`pan_jaad_0117` H.264: 4% of the
    /// instructions).
    fn active_aq(&self) -> Option<&BlockAq> {
        self.layer.aq.as_ref().filter(|aq| aq.active())
    }

    /// The run-up to a scheduled key (`rate.prekey_quiet`) in frames, when
    /// the layer propagation term reads it (0 otherwise: the parameters
    /// are not looked up on every frame).
    fn prekey_window(&self) -> f64 {
        if !self.layer.aq.as_ref().is_some_and(BlockAq::propagates) {
            return 0.;
        }
        let fps = f64::from(self.config.frame_rate.num) / f64::from(self.config.frame_rate.den);
        f64::from(RateParams::from_overrides(&self.config.params).prekey_quiet) * fps
    }

    /// Frames from `index` to the next scheduled key frame (`None` when no
    /// later key is scheduled; `key_clock.rs`).
    fn frames_to_next_key(&self, index: u64) -> Option<u64> {
        let period = self.key_period();
        (period != u64::MAX && period > 0).then(|| {
            self.key_clock
                .frames_to_next(index, self.schedule_time(), period)
        })
    }

    /// The sample-unit quantiser step the reuse term's error gate compares
    /// against: the inter level of rate control, or the configured fixed
    /// quantizer (the frame's own is chosen after the block offsets).
    fn reuse_sample_step(&self) -> f64 {
        let level = self
            .layer
            .rate_control
            .as_ref()
            .and_then(RateController::inter_level)
            .unwrap_or(self.config.quantizer);
        self.sample_step(level)
    }

    /// Whether key `index` (buffer share `share`) is bounded by rate
    /// control's allowance (`ReuseFrame::key_bound`): asked only when the
    /// reuse term needs it, and false under fixed QP.
    pub(crate) fn reuse_key_bound(
        &mut self,
        index: u64,
        pre: &PreAnalysis,
        share: &LayerShare,
    ) -> bool {
        if !self
            .layer
            .reuse
            .as_ref()
            .is_some_and(ReuseTerm::wants_key_bound)
        {
            return false;
        }
        let hint = Complexity::from_preanalysis(Some(pre));
        self.layer
            .rate_control
            .as_mut()
            .is_some_and(|c| c.key_bound_by_allowance(index, hint, share))
    }

    /// QP units inter frame `index` is planned coarser than the inter level
    /// by its cadence role at strength `depth`, for the reuse term
    /// (`persist.reuse_level_price`; 0 when off, on keys, on layered frames
    /// and on frames the cadence does not coarsen).
    pub(super) fn reuse_role_offset(
        &self,
        index: u64,
        key: bool,
        references: &FrameReferences,
        depth: Option<f64>,
    ) -> f64 {
        let level_priced = self
            .layer
            .reuse
            .as_ref()
            .is_some_and(ReuseTerm::prices_at_level);
        let (Some(strength), true) = (depth, level_priced && !key) else {
            return 0.;
        };
        let position = references.position;
        let offset = match self.layer.rate_control.as_ref() {
            Some(controller) => controller.role_offset(position, index, strength),
            None if self.cadence_fixed_qp && position.is_flat() => self
                .layer
                .cadence
                .offset(index - self.layer.last_key_index, strength),
            None => 0.,
        };
        offset.max(0.)
    }

    /// The sample-unit step of native `quantizer`.
    fn sample_step(&self, quantizer: u32) -> f64 {
        let quantizer = (quantizer as usize).min(self.quantizer_steps.len() - 1);
        self.quantizer_steps[quantizer] * self.backend.rate_model().sample_step_scale
    }

    /// QP units finer a fixed-QP key is planned by the key interval
    /// (`rate.fixed_qp_key_ln_interval`; none with an explicit key QP
    /// offset, which holds as given).
    fn fixed_qp_key_finer(&self) -> f64 {
        if self.config.key_qp_offset.is_some() {
            return 0.;
        }
        crate::key_interval::finer_qp(0., self.fixed_qp_key_per_ln, self.key_ln_interval())
    }

    /// The key interval feature of key-frame settings (`key_interval.rs`):
    /// every key restarts the schedule (`key_clock.rs`), so each serves the
    /// key period.
    fn key_ln_interval(&self) -> f64 {
        let fps = f64::from(self.config.frame_rate.num) / f64::from(self.config.frame_rate.den);
        crate::key_interval::ln_interval(self.key_period(), fps)
    }

    /// Frames between scheduled key frames (effectively never with intra
    /// refresh, whose sweeps replace them).
    fn key_period(&self) -> u64 {
        if self.refresh.is_some() {
            u64::MAX
        } else {
            u64::from(self.config.intra_period)
        }
    }

    /// Untimed frames from the next one on are one frame of `rate` apart
    /// (a camera changing its rate without passing timestamps): rate
    /// control budgets each for that time and `FrameMetadata::timestamp_us`
    /// advances by it. Timestamped frames ([`Encoder::encode_at`]) follow
    /// their own intervals, so a rate change needs nothing there. The
    /// signalled timing is unchanged; streams whose rate changes should use
    /// `TimingSignal::Variable`.
    ///
    /// # Errors
    ///
    /// [`ConfigError::FrameRate`] for a zero numerator or denominator.
    pub fn set_frame_rate(&mut self, rate: FrameRate) -> Result<(), ConfigError> {
        if rate.num == 0 || rate.den == 0 {
            return Err(ConfigError::FrameRate);
        }
        self.timeline.set_rate(rate);
        self.key_clock.set_nominal(self.timeline.nominal_us());
        if let Some(spatial) = self.spatial.as_mut() {
            spatial.set_frame_rate(&mut self.layer, rate);
        } else if let Some(controller) = self.layer.rate_control.as_mut() {
            controller.set_nominal_rate(f64::from(rate.num) / f64::from(rate.den));
        }
        Ok(())
    }

    /// Code the next frame at `width` x `height` (visible size): a key frame
    /// starting a new coded video sequence with new parameter sets or
    /// sequence header, every reference replaced. Rate control keeps its
    /// models (normalised by the content complexity, which follows the
    /// picture size) and its buffer and debt, so the first frames after the
    /// switch are sized from what it has learned; the frame index,
    /// timestamps, the periodic key schedule and intra refresh continue. A
    /// long-term reference restarts with the key frame. Input frames from
    /// then on must have the new size.
    ///
    /// # Errors
    ///
    /// The dimensions the backend rejects (the encoder is unchanged).
    pub fn set_resolution(&mut self, width: usize, height: usize) -> Result<(), ConfigError> {
        if (width, height) == (self.config.width, self.config.height) {
            return Ok(());
        }
        let mut next = Self::new(EncoderConfig {
            width,
            height,
            ..self.config.clone()
        })?;
        next.backend.continue_from(&self.backend);
        next.info = self.info.clone();
        next.send_encoder_info();
        if let Some(mut controller) = self.layer.rate_control.take() {
            controller.set_picture_size(width * height);
            next.backend.set_rate_control_enabled(true);
            next.layer.rate_control = Some(controller);
        } else if let Some(rc_cfg) = self.spatial.as_ref().and_then(|s| s.rate_config) {
            next.set_rate_control(rc_cfg)
                .expect("rate control validated earlier");
        }
        if let Some(sweep) = self.refresh.as_ref() {
            next.set_intra_refresh(Some(sweep.settings()))?;
        }
        next.next_index = self.next_index;
        next.timeline = self.timeline.clone();
        next.layer.last_key_index = self.layer.last_key_index;
        next.random = self.random.take();
        next.force_keyframe = true;
        *self = next;
        Ok(())
    }

    /// Code the following frames at `width` x `height` (at most the stream
    /// size) without a key frame: inter frames predict from references of
    /// the old size by the codec's scaled prediction (AV1 §7.11.3.3), and
    /// the stream keeps its size, which the backend signals as every
    /// frame's render size so players display at a constant size. Rate
    /// control, the temporal structure, the reference slots, timestamps and
    /// key schedule carry on. The engine's own reference pictures and the
    /// analyzer's reference sources are resampled to the new size as
    /// proxies for its decisions; the backend keeps the exact pictures it
    /// predicts from. Input frames stay at the stream size and are
    /// resampled to the coded size.
    ///
    /// # Errors
    ///
    /// A backend without scaled references (`set_resolution` switches
    /// with a key frame instead), intra refresh (its sweep is laid out for
    /// one size), or dimensions the engine or backend rejects; the encoder
    /// is unchanged.
    pub fn set_coded_resolution(&mut self, width: usize, height: usize) -> Result<(), ConfigError> {
        if self.spatial.is_some() {
            return Err(ConfigError::SpatialScalability(
                "set_coded_resolution is not supported with spatial layers",
            ));
        }
        if (width, height) == (self.layer.width, self.layer.height) {
            return Ok(());
        }
        if !self.backend.supports_scaled_references() {
            return Err(ConfigError::Resolution(
                "this codec cannot predict across picture sizes: use set_resolution",
            ));
        }
        if self.refresh.is_some() {
            return Err(ConfigError::Resolution("not with intra refresh"));
        }
        let multiple = 1usize << self.backend.min_block_log2();
        let picture = EncoderConfig {
            width,
            height,
            ..self.config.clone()
        };
        let coded = self.backend.coded_size(width, height);
        check_dimensions(&picture, coded, multiple)?;
        let bad_dims = |_| ConfigError::Dimensions {
            width,
            height,
            multiple,
        };
        let pool = self
            .layer
            .references
            .as_ref()
            .expect("references are present between frames");
        let resampled = pool
            .resampled(coded.0, coded.1, RECON_BORDER)
            .map_err(bad_dims)?;
        let padding = padding_buffers(&picture, coded).map_err(bad_dims)?;
        let scaler =
            SourceScaler::between((self.config.width, self.config.height), (width, height))
                .map_err(bad_dims)?;
        let input = match self.layer.input {
            Some(_) => Some(Picture::new(width, height, 0).map_err(bad_dims)?),
            None => None,
        };
        // Before the first frame no slot holds a picture.
        let held = if self.next_index > 0 {
            pool.slot_pictures()
        } else {
            [None; MAX_REFERENCE_SLOTS]
        };
        self.backend.set_frame_size(width, height, &held)?;
        if self.next_index > 0 && self.switched_recon.is_none() {
            self.switched_recon = Some(self.reconstruction().clone());
        }
        if let Some(analyzer) = self.layer.analyzer.as_mut() {
            analyzer.resize(width, height).map_err(bad_dims)?;
        }
        self.layer.references = Some(resampled);
        self.layer.padding = padding;
        self.scaler = scaler;
        self.layer.input = input;
        (self.layer.width, self.layer.height) = (width, height);
        (self.layer.coded_width, self.layer.coded_height) = coded;
        if let Some(controller) = self.layer.rate_control.as_mut() {
            controller.set_picture_size(width * height);
        }
        // The reuse term's per-cell history is laid out for the old size.
        if let Some(old) = self.layer.reuse.as_ref() {
            let level_priced = old.prices_at_level();
            self.layer.reuse =
                Some(ReuseTerm::new(B::SUPPORTS_BLOCK_QP).level_priced(level_priced));
            self.layer.reuse_last_frame = None;
        }
        Ok(())
    }

    /// Under `resolution.adaptive`, measure an inter frame's displayed
    /// error and the half size's floor when the policy asks for evidence
    /// (`resolution_evidence.rs`). `stream_input` is the caller's
    /// stream-size picture, if it passed one; `coded_source` and `recon`
    /// are the frame's coded-size source and reconstruction.
    fn gather_resolution_evidence(
        &mut self,
        stream_input: Option<&Picture>,
        coded_source: &Picture,
        recon: &Picture,
    ) {
        let (Some(policy), Some(probe)) =
            (self.resolution.as_mut(), self.resolution_probe.as_mut())
        else {
            return;
        };
        if !policy.wants_evidence() {
            return;
        }
        let coded = (self.layer.width, self.layer.height);
        let sample = match self.scaler.as_ref() {
            None => probe.full_size(
                &self.dsp,
                &visible_luma(coded_source, coded),
                &visible_luma(recon, coded),
            ),
            Some(scaler) => {
                let stream = (self.config.width, self.config.height);
                let source = stream_input.unwrap_or_else(|| scaler.input());
                probe.half_size(
                    &self.dsp,
                    &visible_luma(source, stream),
                    &visible_luma(coded_source, coded),
                    &visible_luma(recon, coded),
                )
            }
        };
        policy.add_evidence(sample);
    }

    /// After an inter frame under `resolution.adaptive`: hand rate
    /// control's planned step to the policy (`resolution.rs`) and switch
    /// the coded size for the next frame when it says so.
    fn adapt_coded_resolution(&mut self) {
        let (Some(policy), Some(controller)) =
            (self.resolution.as_mut(), self.layer.rate_control.as_ref())
        else {
            return;
        };
        // An intra-refresh sweep is laid out for the size it started at.
        if self.refresh.is_some() {
            return;
        }
        let Some(level) = controller.inter_level() else {
            return;
        };
        let largest = self.quantizer_steps[max_quantizer(self.config.codec) as usize];
        let planned = self.rate_steps[(level as usize).min(self.rate_steps.len() - 1)];
        let Some(scale) = policy.observe(planned / largest) else {
            return;
        };
        let (width, height) = match scale {
            CodedScale::Full => (self.config.width, self.config.height),
            CodedScale::Half => half_size(self.config.width, self.config.height),
        };
        // The policy only runs where switching is supported (checked at
        // creation), and both sizes passed the dimension checks then.
        self.set_coded_resolution(width, height)
            .expect("adaptive resolution sizes are valid");
    }

    /// Enable intra refresh (`refresh.rs`) instead of periodic key frames, or
    /// disable it (`None`); takes effect at the next frame. An adaptive
    /// temporal structure is held flat while it is on (the sweep starts with
    /// the first flat frame); a fixed layered structure sweeps on its
    /// base-layer frames only, over as many of them as cover the period.
    ///
    /// # Errors
    ///
    /// [`ConfigError::IntraRefresh`] with spatial layers, without inter
    /// frames, for a backend that cannot recover by it, beside an explicit
    /// second reference, or for a period below [`MIN_REFRESH_PERIOD`].
    pub fn set_intra_refresh(&mut self, refresh: Option<IntraRefresh>) -> Result<(), ConfigError> {
        if let Some(settings) = refresh {
            if self.spatial.is_some() {
                return Err(ConfigError::IntraRefresh(
                    "not supported with spatial layers",
                ));
            }
            if !self.backend.supports_inter() {
                return Err(ConfigError::IntraRefresh(
                    "the backend codes no inter frames",
                ));
            }
            // A backend that does not keep its own choices inside the
            // window's rules (`CodecBackend::INTRA_REFRESH`) would let a
            // decoder stay wrong after a loss: the sweep would silently
            // replace keys with nothing.
            if !B::INTRA_REFRESH {
                return Err(ConfigError::IntraRefresh(
                    "the codec cannot recover by intra refresh (VP9, AV2)",
                ));
            }
            // A block predicting from the HQ frame may reach back past the
            // sweep's clean region: an explicit `refs.second` refuses the
            // sweep, the unified plan (a default) is suspended for it below.
            if self.layer.second.as_ref().is_some_and(|s| !s.is_planned()) {
                return Err(ConfigError::IntraRefresh("not with a second reference"));
            }
            if settings.period < MIN_REFRESH_PERIOD {
                return Err(ConfigError::IntraRefresh("period below 2 frames"));
            }
        }
        if let Some(second) = self.layer.second.as_mut() {
            second.suspend(refresh.is_some());
        }
        self.structure.set_hold_flat(refresh.is_some());
        if refresh != self.refresh.as_ref().map(RefreshSweep::settings) {
            // An adaptive structure is held flat (the sweep runs on every
            // frame); a fixed layered one sweeps on its base-layer frames.
            let base_spacing = if self.structure.is_adaptive() {
                1
            } else {
                longest_reference_distance(self.config.temporal_layers)
            };
            self.refresh = refresh.map(|settings| {
                RefreshSweep::new(
                    settings,
                    self.layer.coded_width,
                    1 << self.backend.min_block_log2(),
                    self.backend.loop_filter_reach(),
                )
                .with_frame_spacing(base_spacing)
            });
        }
        let period = u32::try_from(self.key_period()).unwrap_or(u32::MAX);
        if let Some(controller) = self.layer.rate_control.as_mut() {
            controller.set_key_period(period);
        }
        Ok(())
    }

    /// Whether frame `index` at `position` is a base frame of SVT's
    /// flat-IPP pseudo-layer cadence: every `base_period`th frame, or with
    /// real temporal layers the T0 frames.
    fn is_cadence_base(&self, index: u64, position: LayerPosition) -> bool {
        let period = (self.decision_params.base_period.round() as u64).max(1);
        if position.layers > 1 {
            position.layer == 0
        } else {
            index.is_multiple_of(period)
        }
    }

    /// The effort role of frame `index` (`frame_role.rs`): key frames and
    /// cadence base frames are base, and with
    /// `decision.nonbase_disposable_only` so is every referenced frame.
    fn frame_role(&self, index: u64, references: &FrameReferences, key: bool) -> FrameRole {
        let disposable = references.refresh == SlotMask::NONE;
        let referenced_nonbase = self.decision_params.nonbase_disposable_only >= 0.5 && !disposable;
        let high_rate_base = self.decision_params.nonbase_min_quantizer > 0.0
            && self
                .last_relative_quantizer
                .is_some_and(|q| q < self.decision_params.nonbase_min_quantizer);
        if key
            || referenced_nonbase
            || high_rate_base
            || self.is_cadence_base(index, references.position)
        {
            FrameRole::Base
        } else {
            FrameRole::NonBase
        }
    }

    /// Before analysing frame `index`, the motion search settings of its
    /// role, when the effort overlay or the adaptive exits (`adapt.motion`)
    /// are on. A scene cut found by this
    /// analysis makes the frame a key frame after its search ran with the
    /// non-base settings; only the cut decision sees that search.
    fn apply_search_role(
        &self,
        analyzer: &mut Analyzer,
        index: u64,
        planned: &FrameReferences,
        key: bool,
    ) {
        let role = self.frame_role(index, planned, key);
        if let Some(search) = self.search_config(role) {
            analyzer.set_search_config(search);
        }
    }

    /// The motion search settings of frames of `role` when they differ from
    /// the stream's: the effort overlay's, with the adapted exits and the
    /// exits' linear shifts on top.
    fn search_config(&self, role: FrameRole) -> Option<SearchConfig> {
        let control = self
            .threshold_control
            .as_ref()
            .filter(|control| control.adapts_motion());
        let record = self.labels.is_some();
        if self.effort_overlay.is_none()
            && control.is_none()
            && self.motion_shifts.is_none()
            && !record
        {
            return None;
        }
        let mut search = match &self.effort_overlay {
            Some(overlay) => overlay.params(role).search,
            None => AnalysisParams::from_overrides(&self.config.params).search_config(),
        };
        if let Some(control) = control {
            control.apply_search(role, &mut search);
        }
        if let Some(shifts) = &self.motion_shifts {
            shifts.with_motion_shifts(
                &mut search,
                role == FrameRole::NonBase,
                self.backend.lambda(),
            );
        }
        if record {
            // The label dump records the zero-gate network's inputs.
            search.gate_net.record_inputs = true;
            search.gate_net.frame = nc_analysis::gate_net::GateFrame {
                lambda: self.backend.lambda(),
                nonbase: role == FrameRole::NonBase,
            };
        }
        Some(search)
    }

    /// Before coding frame `index`, the backend's settings of its `role`
    /// (`key`: a key frame), when the effort overlay is on.
    fn apply_backend_role(&mut self, role: FrameRole, key: bool) {
        if let Some(overlay) = &self.effort_overlay {
            self.backend.set_frame_params(overlay.backend_overrides(
                &self.config.params,
                role,
                key,
            ));
        }
    }

    /// Before coding frame `index` of `role`, the settings of its light
    /// superblocks (`effort_class.rs`): the engine's adjusted as the
    /// frame's own are, and the backend's.
    fn apply_light_role(
        &mut self,
        index: u64,
        references: &FrameReferences,
        (role, key): (FrameRole, bool),
        analysis: Option<&FrameAnalysis>,
    ) {
        self.frame_light = None;
        let Some(overlay) = self.light_overlay.as_ref() else {
            return;
        };
        let step = self.sample_step(self.frame_quantizer) as f32;
        let (width, height) = (self.layer.coded_width, self.layer.coded_height);
        let sb_log2 = self.backend.superblock_log2();
        let map = analysis.and_then(|analysis| {
            if key {
                overlay.applies_to_key().then(|| {
                    KeyClassifier {
                        pre: &analysis.pre,
                        effort: &crate::key_interval::key_effort_params(
                            overlay.effort,
                            self.key_ln_interval(),
                        ),
                        step,
                        width,
                        height,
                    }
                    .light_map(sb_log2)
                })
            } else {
                let motion = analysis.motion.as_ref()?;
                overlay.applies_to(role).then(|| {
                    Classifier {
                        motion,
                        effort: &overlay.effort,
                        step,
                        width,
                        height,
                    }
                    .light_map(sb_log2)
                })
            }
        });
        let Some(map) = map else {
            return;
        };
        let light = overlay.params(role);
        let overrides = overlay.backend_overrides(role).clone();
        let (decision, planner) = self.adjusted_for_frame(
            (light.decision, light.planner),
            index,
            references,
            (role, key, analysis),
        );
        self.frame_light = Some(FrameLight {
            params: LightParams { decision, planner },
            map,
        });
        self.backend.set_light_params(Some(&overrides));
    }

    /// Before coding a frame of `role`, the backend's loop-filter effort
    /// (`filter_effort.rs`): decided from `motion` on inter frames, the
    /// full filters on key frames.
    fn apply_filter_effort(
        &mut self,
        (role, index): (FrameRole, u64),
        inter: bool,
        motion: Option<&MotionAnalysis>,
    ) {
        let mut effort = match motion {
            Some(motion) => {
                let proxy = Classifier {
                    motion,
                    effort: &self.effort_params,
                    step: self.sample_step(self.frame_quantizer) as f32,
                    width: self.layer.coded_width,
                    height: self.layer.coded_height,
                }
                .frame_proxy();
                let activity = FrameActivity {
                    role,
                    proxy: Some(proxy),
                    previous_skip_share: self.previous_skip_share,
                    cdef_reused: self.cdef_reused,
                };
                let mut effort = crate::filter_effort::decide(&self.filter_params, activity);
                if let Some(reuse) = self
                    .threshold_control
                    .as_ref()
                    .and_then(ThresholdControl::cdef_reuse_frames)
                {
                    // The adaptive interval (`adapt.cdef`): base frames search,
                    // and a non-base frame reuses while the frames since the
                    // last search are fewer than the controller allows.
                    effort.cdef = if role == FrameRole::NonBase && self.cdef_reused < reuse {
                        CdefEffort::Reuse
                    } else {
                        CdefEffort::Search
                    };
                }
                effort
            }
            None => FilterEffort::FULL,
        };
        if inter {
            effort.usage = crate::filter_effort::usage_gate(&self.filter_params, role, index);
        }
        self.cdef_reused = match effort.cdef {
            CdefEffort::Reuse => self.cdef_reused + 1,
            CdefEffort::Search => 0,
        };
        self.backend.set_filter_effort(effort);
    }

    /// The decision and planner parameters as adjusted for one frame.
    fn frame_params(
        &self,
        index: u64,
        references: &FrameReferences,
        (role, key): (FrameRole, bool),
        analysis: Option<&FrameAnalysis>,
    ) -> (DecisionParams, PlannerParams) {
        let params = match &self.effort_overlay {
            Some(overlay) => {
                let params = overlay.params(role);
                (params.decision, params.planner)
            }
            None => (self.decision_params, self.planner_params),
        };
        self.adjusted_for_frame(params, index, references, (role, key, analysis))
    }

    /// `decision_params` and `planner_params` with frame `index`'s
    /// adjustments: the cadence base λ, the frame-level content gates, the
    /// thresholds adapted for its `role` and, on a `key` frame, its searches
    /// at the key interval (`key_interval.rs`).
    fn adjusted_for_frame(
        &self,
        (mut decision_params, mut planner_params): (DecisionParams, PlannerParams),
        index: u64,
        references: &FrameReferences,
        (role, key, analysis): (FrameRole, bool, Option<&FrameAnalysis>),
    ) -> (DecisionParams, PlannerParams) {
        if key {
            (decision_params, planner_params) = crate::key_interval::key_search_params(
                decision_params,
                planner_params,
                self.key_ln_interval(),
            );
        }
        if let Some(control) = &self.threshold_control {
            control.apply(role, &mut decision_params, &mut planner_params);
        }
        let noise_sigma = analysis.map(|a| a.noise_sigma);
        decision_params = decision_params
            .with_frame_shift_features(role == FrameRole::NonBase, noise_sigma.unwrap_or(0.0));
        // A cadence base frame may get a lower decision/planner λ.
        if self.is_cadence_base(index, references.position) {
            decision_params.lambda_scale *= decision_params.base_lambda_scale;
            planner_params.lambda_scale *= decision_params.base_lambda_scale;
        }
        // R1-only coding tool overrides when frame 1 is refining a capped first key.
        if !key
            && let Some((inter_min_3, rd_split, lam_scale)) = self
                .layer
                .rate_control
                .as_ref()
                .and_then(RateController::r1_tool_overrides)
        {
            if inter_min_3 {
                planner_params.inter_min_log2 = 3.0;
            }
            if rd_split {
                planner_params.rd_split = 1.0;
            }
            if lam_scale > 0.0 {
                decision_params.lambda_scale *= lam_scale as f32;
                planner_params.lambda_scale *= lam_scale as f32;
            }
        }
        // A changed search can fit sensor noise rather than real motion.
        // The noise estimate already exists; apply this content gate once per
        // frame, preserving the legacy search when the configured limit wins.
        if noise_sigma.is_some_and(|sigma| sigma > decision_params.subpel_walk_max_noise_sigma) {
            decision_params.subpel_walk = 0.0;
        }
        // A frame motion search predicts well needs no intra (SVT skip_intra).
        let skip_limit = decision_params.intra_frame_skip_me_dist;
        if skip_limit > 0.0
            && analysis
                .and_then(|a| a.motion.as_ref())
                .is_some_and(|m| m.mean_me_dist() <= skip_limit)
        {
            decision_params.intra_gate_variance = f32::INFINITY;
            decision_params.intra_gate_lambdas = f32::INFINITY;
        }
        (decision_params, planner_params)
    }

    /// Encode one frame, timed one nominal frame after the previous one.
    /// `source` must match the configured dimensions.
    pub fn encode(&mut self, source: &Picture) -> EncodedFrame {
        self.encode_picture(source, None)
    }

    /// [`Encoder::encode`] for a frame captured at `timestamp_us`
    /// (microseconds, any origin). Rate control budgets the frame for the
    /// time since the previous frame, so variable-rate input and dropped
    /// camera frames keep the bitrate in time.
    pub fn encode_at(&mut self, source: &Picture, timestamp_us: i64) -> EncodedFrame {
        self.encode_picture(source, Some(timestamp_us))
    }

    fn encode_picture(&mut self, source: &Picture, timestamp_us: Option<i64>) -> EncodedFrame {
        assert_eq!(
            (source.width(), source.height()),
            (self.config.width, self.config.height),
            "source picture does not match the encoder configuration"
        );
        self.encode_frame(Some(source), timestamp_us)
    }

    /// The picture the next source frame can be written into (its visible
    /// area) before [`Encoder::encode_written`]. This saves the copy that
    /// [`Encoder::encode`] makes into the analyzer's bordered buffer (about
    /// 1% of HEVC encode time on the Jetson).
    pub fn source_mut(&mut self) -> &mut Picture {
        if let Some(spatial) = self.spatial.as_mut() {
            return spatial.source_mut();
        }
        if let Some(scaler) = self.scaler.as_mut() {
            return scaler.input_mut();
        }
        match (&mut self.layer.analyzer, &mut self.layer.input) {
            (Some(analyzer), _) => analyzer.next_frame_mut(),
            (None, Some(input)) => input,
            (None, None) => unreachable!("an encoder without an analyzer owns an input buffer"),
        }
    }

    /// Encode a borrowed camera frame (I420, YV12 or NV12, any strides; see
    /// [`RawFrame`]). It is converted straight into the source buffer of
    /// [`Encoder::source_mut`], so this costs no more than [`Encoder::encode`].
    ///
    /// # Errors
    ///
    /// [`nc_common::Error::InvalidDimensions`] if the frame's planes do not
    /// have the configured size; nothing is encoded then.
    pub fn encode_raw(&mut self, frame: &RawFrame<'_>) -> Result<EncodedFrame, nc_common::Error> {
        self.encode_raw_timed(frame, None)
    }

    /// [`Encoder::encode_raw`] for a frame captured at `timestamp_us` (see
    /// [`Encoder::encode_at`]).
    ///
    /// # Errors
    ///
    /// As [`Encoder::encode_raw`].
    pub fn encode_raw_at(
        &mut self,
        frame: &RawFrame<'_>,
        timestamp_us: i64,
    ) -> Result<EncodedFrame, nc_common::Error> {
        self.encode_raw_timed(frame, Some(timestamp_us))
    }

    fn encode_raw_timed(
        &mut self,
        frame: &RawFrame<'_>,
        timestamp_us: Option<i64>,
    ) -> Result<EncodedFrame, nc_common::Error> {
        frame.check(self.config.width, self.config.height)?;
        let dsp = self.dsp;
        write_raw(&dsp, frame, self.source_mut());
        Ok(self.encode_frame(None, timestamp_us))
    }

    /// Before the first frame: hand over the second source frame, which the
    /// caller still encodes in turn. Rate control then sizes the stream's
    /// first key for the camera motion between the two
    /// (`rate.first_key_lookahead`), and a fixed-QP stream offsets it
    /// (`rate.fixed_qp_moving_key`); ignored when the one in force is off,
    /// or after the first frame. The caller waits one frame
    /// longer before the first; the encoder spends one extra source
    /// analysis.
    ///
    /// # Errors
    ///
    /// [`nc_common::Error::InvalidDimensions`] if the frame's planes do not
    /// have the configured size.
    pub fn look_ahead(&mut self, next: &RawFrame<'_>) -> Result<(), nc_common::Error> {
        next.check(self.config.width, self.config.height)?;
        if self.wants_look_ahead() {
            let mut picture = Picture::new(self.config.width, self.config.height, 0)?;
            write_raw(&self.dsp, next, &mut picture);
            if self.scaler.is_some() {
                let mut scaled = Picture::new(self.layer.width, self.layer.height, 0)?;
                scale_source_picture(&self.dsp, &picture, &mut scaled);
                picture = scaled;
            }
            self.lookahead.hold(picture);
        }
        Ok(())
    }

    /// Whether [`Encoder::look_ahead`] would be used: before the first
    /// frame, with rate control and `rate.first_key_lookahead` on, or at
    /// fixed QP with `rate.fixed_qp_moving_key`, unless
    /// `rate.startup_lookahead` is 0. A capture loop holds the first frame
    /// back only then.
    #[must_use]
    pub fn wants_look_ahead(&self) -> bool {
        if !self.lookahead.is_on() {
            return false;
        }
        if let Some(spatial) = self.spatial.as_ref() {
            return spatial.wants_look_ahead(&self.layer);
        }
        self.next_index == 0
            && self.layer.rate_control.as_ref().map_or_else(
                || self.moving_camera.enabled(),
                RateController::wants_startup_motion,
            )
    }

    /// Encode the frame written into [`Encoder::source_mut`].
    pub fn encode_written(&mut self) -> EncodedFrame {
        self.encode_frame(None, None)
    }

    /// [`Encoder::encode_written`] for a frame captured at `timestamp_us`
    /// (see [`Encoder::encode_at`]).
    pub fn encode_written_at(&mut self, timestamp_us: i64) -> EncodedFrame {
        self.encode_frame(None, Some(timestamp_us))
    }

    /// Encode `input`, or the frame written into [`Encoder::source_mut`],
    /// captured at `timestamp_us` if known.
    #[allow(clippy::too_many_lines)]
    fn encode_frame(&mut self, input: Option<&Picture>, timestamp_us: Option<i64>) -> EncodedFrame {
        if self.spatial.is_some() {
            return self.encode_spatial_tu(input, timestamp_us);
        }
        // The input goes into the analyzer's buffer first (resampled when
        // frames are coded smaller than the stream): frame hold decides
        // from its pre-analysis.
        let stream_input = input;
        let input = if self.scaler.is_some() {
            self.scale_source(input);
            None
        } else {
            input
        };
        if let (Some(source), Some(a)) = (input, self.layer.analyzer.as_mut()) {
            a.write_next(source);
        }
        let held = self.hold_due();
        if held && !self.hold.as_ref().is_some_and(FrameHold::emits_repeats) {
            return self.hold_frame(timestamp_us);
        }
        self.switched_recon = None;
        let index = self.next_index;
        let time = self.timeline.next(timestamp_us);
        self.frame_time = time;
        let frames_to_key = self.frames_to_next_key(index);
        if let Some(controller) = self.layer.rate_control.as_mut() {
            controller.set_frame_interval(time.interval);
            controller.set_frames_to_key(frames_to_key);
        }
        if self.drop_due(index) {
            return self.drop_frame(index, time);
        }
        self.next_index += 1;
        let (scheduled, ltr_action) = self.ltr_plan(index);
        let on_schedule = self.periodic_key(index);
        if time.timed {
            self.key_clock.coded(time.timestamp_us, self.key_period());
        }
        // A recovery frame restarts the layer pattern, as a key frame does.
        let recovery = ltr_action == LtrAction::Recover;
        let planned = self.with_ltr(
            ltr_action,
            index,
            self.structure
                .peek(index, scheduled == FrameType::Key || recovery),
        );
        let planned = self.with_second(index, planned, (scheduled == FrameType::Key, None));
        // Shared analysis, when enabled, runs on every frame so that the
        // next frame always has its reference's source to search against.
        // The analyzer is moved out of `self` for the frame so its results
        // can be borrowed while the partition walk mutates the encoder.
        let mut analyzer = self.layer.analyzer.take();
        let own_input = self.layer.input.take();
        if let Some(a) = analyzer.as_mut() {
            self.apply_search_role(a, index, &planned, scheduled == FrameType::Key);
            a.analyze_written_timed(planned.link(), &mut NoTimer);
        }
        // A scene cut turns a scheduled inter frame into a key frame, which
        // refreshes every reference slot (the analyzer's sources too).
        let cut =
            scheduled == FrameType::Inter && self.is_scene_cut(index, &planned, analyzer.as_ref());
        let frame_type = if cut { FrameType::Key } else { scheduled };
        if frame_type == FrameType::Key {
            self.layer.last_key_index = index;
            if !on_schedule {
                self.key_clock
                    .restart(index, time.timed.then_some(time.timestamp_us));
            }
            if let Some(scene) = self.scene.as_mut() {
                scene.reset();
            }
        }
        let key = frame_type == FrameType::Key;
        if let Some(inheritance) = self.layer.inheritance.as_ref() {
            self.structure
                .observe_inheritance(inheritance.loss().map(|loss| loss as f32));
        }
        let references = self.structure.next(index, key || recovery);
        let ltr_action = if key { LtrAction::None } else { ltr_action };
        let references = self.with_ltr(ltr_action, index, references);
        // The motion was measured against the planned reference.
        if let Some(a) = analyzer.as_ref().and_then(Analyzer::analysis)
            && let Some(hint) = a
                .motion
                .as_ref()
                .and_then(|m| cadence_motion_hint(m, planned.reference.distance, self.layer.width))
        {
            self.moving_camera
                .observe(hint.camera_motion, f64::from(a.pre.static_fraction));
        }
        if let Some(second) = self.layer.second.as_mut() {
            second.camera_moving(self.moving_camera.moving());
        }
        self.observe_depth_model_features(analyzer.as_ref(), &planned);
        let depth = analyzer
            .as_ref()
            .and_then(Analyzer::analysis)
            .map(|a| self.cadence_depth(index, a, &references));
        let references = self.with_second(index, references, (key, depth));
        // A held frame is coded as a copy of the frame just before it, the
        // picture the receiver shows (`hold.rs` "Repeat frames").
        self.repeating =
            held && !key && references.reference.distance == 1 && self.refresh.is_none();
        if let Some(ltr) = self.ltr.as_mut() {
            ltr.coded(index, key, ltr_action);
        }
        if let Some(second) = self.layer.second.as_mut() {
            second.coded(index, &references);
        }
        let referenced = self.feedback.coded(index, (key, recovery), &references);
        if let (true, Some(a)) = (cut, analyzer.as_mut()) {
            a.refresh_current(references.refresh);
        }
        let (analyzed, analysis) = analyzed_frame(analyzer.as_mut());
        // The motion was measured against the planned reference.
        if let Some(analysis) = analysis
            && let Some(motion) = &analysis.motion
        {
            let distance = planned.reference.distance;
            self.structure.observe(motion_signal(
                motion,
                Some(analysis),
                distance,
                self.layer.width,
            ));
        }
        let input = input.or(own_input.as_ref());
        let source = analyzed
            .as_deref()
            .or(input)
            .expect("the encoder has an input buffer");
        if let Some(next) = self.lookahead.take()
            && let Some((peak, still)) = startup_scene_profile(
                &self.config,
                (self.layer.width, self.layer.height),
                source,
                &next,
            )
        {
            self.moving_camera.observe_startup(peak, f64::from(still));
            if let Some(controller) = self.layer.rate_control.as_mut() {
                controller.observe_startup_scene(peak, f64::from(still));
            }
        }
        let motion = analysis.and_then(|a| a.motion.as_ref());
        // Static detection compares with the reference's source frame, so it
        // means nothing on the first frame.
        let pre = analysis.filter(|a| a.motion.is_some()).map(|a| &a.pre);
        let still = analysis.filter(|a| a.motion.is_some()).map(|a| &a.still);

        let mut key_reuse_offset = 0.;
        let reuse_frame = on_reuse_chain(key, &references);
        self.reuse_frame = reuse_frame.then_some(index);
        if let Some(analysis) = analysis {
            let key_bound = key && self.reuse_key_bound(index, &analysis.pre, &LayerShare::WHOLE);
            let frame = ReuseFrame {
                still: &analysis.still,
                pre: &analysis.pre,
                key,
                horizon: self.frames_to_next_key(index),
                sample_step: self.reuse_sample_step(),
                role_offset: self.reuse_role_offset(index, key, &references, depth),
                key_bound,
            };
            // The term's state is the frame before, or on a layered stream
            // the previous base frame (the last one it saw).
            let reference_index = index.checked_sub(u64::from(references.reference.distance));
            let follows = references.reference.distance == 1
                || (reference_index.is_some() && reference_index == self.layer.reuse_last_frame);
            self.still.without_offsets = false;
            if let Some(term) = self.layer.reuse.as_mut().filter(|_| reuse_frame) {
                term.plan(&frame);
                term.set_reference(!key && follows);
                let flat = references.position.is_flat() && follows;
                let strength = self
                    .layer
                    .rate_control
                    .as_ref()
                    .map_or(1., RateController::prekey_quiet_strength);
                self.still.plan(term, &frame, flat, strength);
            }
            let offsets = self
                .layer
                .reuse
                .as_ref()
                .filter(|_| reuse_frame)
                .map(ReuseTerm::offsets);
            let cells = analysis.pre.blocks.cells().len();
            let key_map = self.key_offsets.take().filter(|m| key && m.len() == cells);
            let keyed = key_map.as_deref().map(|map| with_key_map(offsets, map));
            let offsets = keyed.as_deref().or(offsets);
            let window = self.prekey_window();
            let before_key = window > 0.
                && self
                    .frames_to_next_key(index)
                    .is_some_and(|frames| frames as f64 <= window);
            let (layer, top_offset) = self.propagation_role(index, &references, depth);
            if let Some(aq) = self.layer.aq.as_mut() {
                aq.plan_layer(&LayerFrame {
                    index,
                    reference: index.saturating_sub(u64::from(references.reference.distance)),
                    key,
                    period: if !references.position.is_flat() {
                        Period::Layered
                    } else if self.layer.cadence.is_flat() {
                        Period::None
                    } else {
                        Period::Cadence
                    },
                    layer,
                    top_offset,
                    before_key,
                    pre: &analysis.pre,
                    motion,
                });
                aq.plan(&analysis.pre, motion, offsets);
                if key && index > 0 && self.forced_quantizer.is_none() && self.layer.key_aq_qp > 0.
                {
                    aq.apply_periodic_key_aq(
                        &analysis.pre,
                        &analysis.still,
                        self.layer.key_aq_qp,
                        self.layer.last_camera_motion,
                    );
                }
            } else if key {
                // Without block quantizers a key carries the reuse term as
                // one frame offset.
                key_reuse_offset = self
                    .layer
                    .reuse
                    .as_ref()
                    .map_or(0., ReuseTerm::frame_offset);
            }
        }
        self.block_offsets.clear();
        self.in_frame_offsets.clear();
        if key
            && !self.repeating
            && let Some(controller) = self.layer.rate_control.as_mut()
            && controller.wants_first_key_picture()
        {
            let q = crate::params::QuantParams::from_overrides(&self.config.params);
            let key_trellis_off = q.trellis < 0.5 || q.trellis_key < 0.5;
            controller
                .observe_first_key_picture(source.plane(PlaneIndex::Y).view(), key_trellis_off);
        }
        let mut plan = if self.repeating {
            self.repeat_plan(index, references)
        } else {
            self.frame_plan(
                index,
                frame_type,
                references,
                analysis.map(|a| &a.pre),
                motion,
                key_reuse_offset,
            )
        };
        self.frame_quantizer = plan.quantizer;
        self.set_in_frame_weights(analysis.map(|a| &a.pre), key, plan.quantizer);

        let mut padding = self.layer.padding.take();
        let source_for_coding = coded_source(
            analyzed,
            input,
            padding.as_mut(),
            (self.layer.coded_width, self.layer.coded_height),
        );
        let coded_source = &*source_for_coding;
        let secondary_ratio = references
            .secondary
            .map(|link| link.distance as f32 / references.reference.distance.max(1) as f32);
        let inter = (frame_type == FrameType::Inter).then_some(FrameInputs {
            motion,
            pre,
            still,
            secondary_ratio,
        });
        let role = self.frame_role(index, &references, key);
        let (decision_params, planner_params) =
            self.frame_params(index, &references, (role, key), analysis);
        self.frame_decision_params = decision_params;
        self.apply_backend_role(role, key);
        self.apply_light_role(index, &references, (role, key), analysis);
        self.note_label_overrides(role, key);
        let inter_frame = frame_type == FrameType::Inter;
        if self.repeating {
            self.backend.set_filter_effort(FilterEffort::REPEAT);
        } else {
            self.apply_filter_effort((role, index), inter_frame, motion.filter(|_| inter_frame));
            let r1_active = inter_frame
                && self
                    .layer
                    .rate_control
                    .as_ref()
                    .is_some_and(|rc| rc.r1_tool_overrides().is_some());
            self.backend.set_r1_refine(r1_active);
        }
        if decision_params.noise_skip > 0.
            && let (Some(pre), Some(motion)) = (pre, motion)
        {
            let still = camera_is_still(motion, (self.layer.width, self.layer.height));
            self.layer.noise_map.update(pre, &motion.fine, still);
        }

        self.backend
            .set_noise_sigma(analysis.map_or(0.0, |a| a.noise_sigma));
        if let Some(inheritance) = self.layer.inheritance.as_mut() {
            inheritance.set_noise_sigma(f64::from(analysis.map_or(0.0, |a| a.noise_sigma)));
        }
        if let Some(labeller) = self.labels.as_mut() {
            labeller.noise_sigma = analysis.map_or(0.0, |a| a.noise_sigma);
        }
        // The sweep assumes each frame with a window references the
        // previous one: only base-layer frames code one. Upper-layer frames
        // (of a fixed layered structure, or of a layered mini-GOP still in
        // progress when an adaptive structure was held flat) leave the sweep
        // where it is; they predict only from frames of their own mini-GOP
        // and its T0 frame, and no base-layer frame predicts from them, so
        // they are clean once the base layer is.
        let base = references.position.layer == 0;
        self.window = match (self.refresh.as_mut(), frame_type) {
            (Some(sweep), FrameType::Inter) if base => {
                // A recovery frame predicts from the long-term reference,
                // coded earlier in the sweep or before it: a receiver that
                // never lost a frame may hold it partly dirty, so the sweep
                // starts again from the recovery frame.
                if recovery {
                    sweep.restart();
                }
                Some(sweep.next_window())
            }
            (Some(sweep), FrameType::Key) => {
                sweep.restart();
                None
            }
            _ => None,
        };
        self.backend.set_refresh(self.window.as_ref());
        let mut pool = self
            .layer
            .references
            .take()
            .expect("references are present between frames");
        let secondary_slot = references
            .secondary
            .filter(|_| frame_type == FrameType::Inter)
            .map(|link| link.slot);
        let (reference, secondary, recon) = pool.frames(references.reference.slot, secondary_slot);
        let pictures = FramePictures {
            source: coded_source,
            reference,
            secondary,
            recon,
        };
        let data = if self.repeating {
            self.code_repeat_picture(&plan, pictures, inter.as_ref())
        } else {
            let first = self.identity_trial(plan).unwrap_or(plan);
            self.frame_quantizer = first.quantizer;
            self.begin_second_usage();
            // `rate.key_plan_first`: the first coding of a key leaves out its
            // RD searches, the coding rate control then settles keeps them.
            let searched = (self.frame_decision_params, planner_params);
            let plan_first = key && self.key_search_ratio.is_some() && self.key_searches(&searched);
            let first_params = if plan_first {
                self.frame_decision_params = without_key_leaf_searches(&searched.0);
                without_key_partition_search(&planner_params)
            } else {
                planner_params
            };
            // A plan-only first coding measures the whole key: never stopped.
            let first_stop = if plan_first && self.first_stop == StopRule::InFrame {
                StopRule::Never
            } else {
                self.first_stop
            };
            let mut coded =
                self.code_picture(&first, pictures, inter.as_ref(), &first_params, first_stop);
            let mut codings = 1;
            let mut plan_bits = None;
            loop {
                let (bits, stopped) = match &coded {
                    Ok(data) => (bit_count(data), false),
                    Err(stopped) => (stopped.bits as u64, true),
                };
                let plan_coding = plan_first && plan_bits.is_none();
                // Rate control judges the plan's coding by the searched coding
                // it stands for.
                let judged = if plan_coding {
                    (bits as f64 * self.key_search_ratio.unwrap_or(1.)) as u64
                } else {
                    bits
                };
                let recode = self.recode_plan(plan, (judged, stopped), codings);
                let recode = if plan_coding {
                    // The plan's coding only measured the key: code it searched,
                    // at the step rate control asks for, or at the same one.
                    plan_bits = Some(bits);
                    self.frame_decision_params = searched.0;
                    Some(recode.unwrap_or(plan))
                } else {
                    recode
                };
                // A stopped coding has no stream to keep: it is always coded
                // again, at its own plan if rate control would keep it.
                let recode = match recode {
                    Some(recode) => recode,
                    None if stopped => plan,
                    None => break,
                };
                codings += 1;
                // The frame would overflow the buffer: code it again,
                // coarser (`RateController::recode_quantizer`).
                plan = recode;
                self.frame_quantizer = plan.quantizer;
                self.block_offsets.clear();
                self.in_frame_offsets.clear();
                let pictures = FramePictures {
                    source: coded_source,
                    reference,
                    secondary,
                    recon: &mut *recon,
                };
                self.begin_second_usage();
                coded = self.code_picture(
                    &plan,
                    pictures,
                    inter.as_ref(),
                    &planner_params,
                    StopRule::InFrame,
                );
                if let (Some(plan_bits), Ok(data)) = (plan_bits.filter(|_| codings == 2), &coded) {
                    let same_step = plan.quantizer == first.quantizer;
                    self.learn_key_search_ratio(plan_bits, bit_count(data), same_step);
                }
            }
            let Ok(data) = coded else {
                unreachable!("a stopped coding is always coded again");
            };
            data
        };
        if !B::FILTER_BEFORE_END {
            self.backend.filter_frame(coded_source, recon);
        }
        self.backend.commit_frame();
        if let Some(labeller) = self.labels.as_mut() {
            labeller.finish_frame();
        }
        if self.repeating {
            // A repeat frame decided nothing: the adaptive tools and the
            // statistics the next frames read keep the last coded frame's.
            self.depths.hold();
        } else {
            self.observe_coded_decisions((key, role), motion);
            self.observe_hq_quality(index, key, &references, (coded_source, &*recon));
        }
        self.depths.commit();
        self.finish_coded_frame_feedback(
            index,
            key,
            references.position,
            reuse_frame,
            plan.quantizer,
            analysis,
            (coded_source, recon),
            &data,
        );
        if frame_type == FrameType::Inter && !self.repeating {
            self.gather_resolution_evidence(stream_input, coded_source, recon);
        }
        drop(source_for_coding);
        self.layer.analyzer = analyzer;
        self.layer.input = own_input;
        pool.finish(references.refresh);
        if let Some(p) = padding.as_mut() {
            p.reconstructed();
        }
        self.layer.references = Some(pool);
        self.layer.padding = padding;
        if frame_type == FrameType::Inter && !self.repeating {
            self.adapt_coded_resolution();
        }
        self.repeating = false;
        EncodedFrame {
            data,
            metadata: FrameMetadata {
                index: plan.index,
                frame_type: plan.frame_type,
                temporal_layer: references.position.layer,
                timestamp_us: time.timestamp_us,
                dropped: false,
                held: false,
                quantizer: plan.quantizer,
                references: referenced,
            },
            layers: Vec::new(),
        }
    }

    /// What a coded frame's decisions tell the adaptive tools: the filter
    /// outcomes, the second reference's usage, the skip share the next
    /// frame's filter effort reads, and the evidence of threshold control.
    fn observe_coded_decisions(
        &mut self,
        (key, role): (bool, FrameRole),
        motion: Option<&MotionAnalysis>,
    ) {
        let outcome_role = match (key, role) {
            (true, _) => OutcomeRole::Key,
            (false, FrameRole::Base) => OutcomeRole::Base,
            (false, FrameRole::NonBase) => OutcomeRole::NonBase,
        };
        self.filter_outcomes
            .add(outcome_role, &self.backend.filter_outcomes());
        if let Some(second) = self.layer.second.as_mut().filter(|s| s.watches_usage()) {
            second.finish_usage();
        }
        self.previous_skip_share = crate::filter_effort::reads_skip_share(&self.filter_params)
            .then(|| self.backend.coded_skip_share())
            .flatten();
        let relative_quantizer = self.backend.relative_quantizer();
        self.last_relative_quantizer = Some(relative_quantizer);
        self.structure.observe_quantizer(relative_quantizer);
        if !key
            && let (Some(motion), Some(search)) = (motion, self.search_config(role))
            && self.decision_scratch.evidence.gathers(Tool::Motion)
        {
            record_motion_evidence(motion, &search, &mut self.decision_scratch.evidence);
        }
        if let (Some(control), false) = (self.threshold_control.as_mut(), key) {
            control.observe(
                role,
                &self.decision_scratch.evidence,
                self.backend.filter_search_evidence(),
            );
        }
    }

    /// The plan of repeat frame `index` (`hold.rs` "Repeat frames"): at the
    /// last coded frame's quantizer, which only its headers carry, without
    /// asking rate control for one (its bits are charged after coding,
    /// [`RateController::repeat_frame`]).
    fn repeat_plan(&self, index: u64, references: FrameReferences) -> FramePlan {
        FramePlan {
            index,
            frame_type: FrameType::Inter,
            quantizer: self.frame_quantizer,
            references,
            lambda_scale: 1.,
            block_qp: false,
        }
    }

    /// Code repeat frame `plan` (`hold.rs` "Repeat frames"): every leaf at
    /// the largest size the picture allows, each the backend's skip at the
    /// zero vector ([`StaticSkipTest::repeat`]); a leaf the backend declines
    /// is decided as usual. Never stopped or re-coded.
    fn code_repeat_picture(
        &mut self,
        plan: &FramePlan,
        pictures: FramePictures<'_>,
        inter: Option<&FrameInputs<'_>>,
    ) -> Vec<u8> {
        let FramePictures {
            source,
            reference,
            secondary,
            recon,
        } = pictures;
        self.begin_coding(plan);
        let sb_log2 = self.backend.superblock_log2();
        let sb = 1usize << sb_log2;
        for y in (0..self.layer.coded_height).step_by(sb) {
            for x in (0..self.layer.coded_width).step_by(sb) {
                self.backend.begin_superblock(x, y);
                let pictures = FramePictures {
                    source,
                    reference,
                    secondary,
                    recon: &mut *recon,
                };
                self.code_node(Block::square(x, y, sb_log2), None, pictures, inter);
            }
        }
        if B::FILTER_BEFORE_END {
            self.backend.filter_frame(source, recon);
        }
        self.backend.end_frame()
    }

    pub(super) fn finish_coded_frame_feedback(
        &mut self,
        index: u64,
        key: bool,
        position: LayerPosition,
        reuse_frame: bool,
        quantizer: u32,
        analysis: Option<&nc_analysis::FrameAnalysis>,
        (coded_source, recon): (&Picture, &Picture),
        data: &[u8],
    ) {
        // Frames coded coarser by design (upper layers, cadence) do not
        // set the quality a hold compares with.
        let high_quality = self.is_high_quality(index, position);
        if (key || high_quality) && !self.repeating {
            self.layer.last_quantizer = quantizer;
        }
        let wants_distortion = self.layer.inheritance.is_some()
            || self
                .layer
                .rate_control
                .as_ref()
                .is_some_and(RateController::wants_distortion);
        let mse = wants_distortion.then(|| {
            let (width, height) = (self.layer.coded_width, self.layer.coded_height);
            luma_mse(&self.dsp, coded_source, recon, width, height)
        });
        let realised_offset = self.realised_block_offset();
        let repeating = self.repeating;
        let inheriting = self.layer.inheritance.as_mut().filter(|_| !repeating);
        if let (Some(inheritance), Some(mse)) = (inheriting, mse) {
            let tier = if key {
                Tier::Key
            } else if high_quality {
                Tier::HighQuality
            } else {
                Tier::Other
            };
            let step =
                self.quantizer_steps[(quantizer as usize).min(self.quantizer_steps.len() - 1)];
            // The quantizer the blocks were coded at, not the frame's alone: the
            // block offsets (layer propagation, reuse) move a frame's real
            // quantizer, and the loss per QP unit read against the frame QP alone
            // made the inherited depth follow them (AV2 slow, propagation off,
            // carried no second reference on crowd where medium did, 2026-11-24).
            let qp_units = QP_UNITS_PER_OCTAVE * step.log2() + realised_offset.unwrap_or(0.);
            inheritance.observe(tier, qp_units, mse);
        }
        if let Some(term) = self.layer.reuse.as_mut().filter(|_| reuse_frame) {
            self.layer.reuse_last_frame = Some(index);
            let coded = CodedFrame {
                key,
                sample_step: self.quantizer_steps
                    [(quantizer as usize).min(self.quantizer_steps.len() - 1)]
                    * self.backend.rate_model().sample_step_scale,
                pre: analysis.map(|a| &a.pre),
                still: analysis.map(|a| &a.still),
            };
            let size = (self.layer.width, self.layer.height);
            term.observe(&self.dsp, (coded_source, recon), size, coded);
        }
        if let (Some(controller), true) = (self.layer.rate_control.as_mut(), self.repeating) {
            controller.repeat_frame(bit_count(data));
        } else if let Some(controller) = self.layer.rate_control.as_mut() {
            if let Some(realised) = realised_offset {
                controller.set_realised_block_offset(realised);
            }
            if let Some(hold) = self.hold.as_mut() {
                let budget = controller.inter_frame_budget().max(1.);
                hold.coded(bit_count(data) as f64 / budget, controller.inter_level());
            }
            controller.feedback(bit_count(data));
            let area = (self.layer.width * self.layer.height).max(1);
            controller.observe_noise_skip(self.layer.noise_skipped_area as f64 / area as f64);
            if let Some(mse) = mse.filter(|_| controller.wants_distortion()) {
                controller.observe_distortion(mse);
            }
        }
        self.layer.noise_skipped_area = 0;
    }

    /// Resample `input`, or the frame written into the scaler's stream-size
    /// buffer, into the buffer [`Encoder::encode_written`] codes.
    fn scale_source(&mut self, input: Option<&Picture>) {
        let scaler = self.scaler.as_ref().expect("only with a source scaler");
        let full = input.unwrap_or_else(|| scaler.input());
        let target = match (&mut self.layer.analyzer, &mut self.layer.input) {
            (Some(analyzer), _) => analyzer.next_frame_mut(),
            (None, Some(input)) => input,
            (None, None) => unreachable!("an encoder without an analyzer owns an input buffer"),
        };
        scale_source_picture(&self.dsp, full, target);
    }

    /// Under `debug.recode_identity`, the plan of a throw-away first coding
    /// of the frame: at the other end of the quantizer range, so every
    /// choice it makes and state it touches differs from the real coding,
    /// and any of it leaking into the re-code (state not restored from the
    /// committed copy) changes the stream.
    fn identity_trial(&self, plan: FramePlan) -> Option<FramePlan> {
        if !self.recode_identity || !B::RECODE || self.random.is_some() {
            return None;
        }
        let (lowest, highest) = self.quantizer_bounds();
        let quantizer = if plan.quantizer < highest {
            highest
        } else {
            lowest
        };
        Some(FramePlan { quantizer, ..plan })
    }

    /// Whether a key coded with `params` runs a key RD search (the
    /// partition search or the leaf search), so a plan-only first coding
    /// (`rate.key_plan_first`) saves time.
    fn key_searches(&self, params: &(DecisionParams, PlannerParams)) -> bool {
        !self.recode_identity
            && (params.1.key_rd_split >= 0.5 && self.rd_split_applies()
                || without_key_leaf_searches(&params.0) != params.0 && self.leaf_rd_applies())
    }

    /// Learn [`Self::key_search_ratio`] from a key's plan-only and searched
    /// codings where both were at one step (`same_step`), kept within
    /// [`KEY_SEARCH_RATIO_RANGE`].
    fn learn_key_search_ratio(&mut self, plan_bits: u64, searched_bits: u64, same_step: bool) {
        if plan_bits > 0 && same_step {
            let (low, high) = KEY_SEARCH_RATIO_RANGE;
            self.key_search_ratio =
                Some((searched_bits as f64 / plan_bits as f64).clamp(low, high));
        }
    }

    /// The plan to code the current frame at once more, after `codings`
    /// codings, if any: the real `plan` after an `identity_trial`, or a
    /// coarser one when rate control finds that its `bits` (`projected` for
    /// a stopped coding) would overflow the buffer
    /// (`RateController::recode_quantizer` bounds how often).
    /// Only backends that honour the recode contract are re-coded, and
    /// never under random decisions (whose generator has moved on).
    fn recode_plan(
        &mut self,
        plan: FramePlan,
        (bits, projected): (u64, bool),
        codings: u32,
    ) -> Option<FramePlan> {
        if !B::RECODE || self.random.is_some() {
            return None;
        }
        if self.recode_identity && codings == 1 {
            return Some(plan);
        }
        let controller = self.layer.rate_control.as_mut()?;
        let planned = if projected {
            controller.recode_quantizer_projected(bits)
        } else {
            controller.recode_quantizer(bits)
        }?;
        let (quantizer, lambda_scale) = self.coded_quantizer(planned);
        Some(FramePlan {
            quantizer,
            lambda_scale,
            ..plan
        })
    }

    /// [`Self::code_picture`] of a coding that is never stopped (spatial
    /// layers, which have no in-frame rate control).
    fn code_whole_picture(
        &mut self,
        plan: &FramePlan,
        pictures: FramePictures<'_>,
        inter: Option<&FrameInputs<'_>>,
        planner_params: &PlannerParams,
    ) -> Vec<u8> {
        let FramePictures {
            source,
            reference,
            secondary,
            recon,
        } = pictures;
        let pictures = FramePictures {
            source,
            reference,
            secondary,
            recon: &mut *recon,
        };
        let Ok(data) = self.code_picture(plan, pictures, inter, planner_params, StopRule::Never)
        else {
            unreachable!("a coding that may not stop always finishes");
        };
        if !B::FILTER_BEFORE_END {
            self.backend.filter_frame(source, recon);
        }
        data
    }

    /// Give in-frame rate control (`in_frame_rate.rs`) the superblock
    /// complexities of the frame about to be coded at `quantizer`, from its
    /// analysis `pre` (a `key` reads its spatial detail), when it watches
    /// the frame; a frame it does not watch is spared the pass over its
    /// cells.
    fn set_in_frame_weights(&mut self, pre: Option<&PreAnalysis>, key: bool, quantizer: u32) {
        let (Some(control), Some(controller)) =
            (self.in_frame.as_ref(), self.layer.rate_control.as_ref())
        else {
            return;
        };
        let bounds = controller.in_frame_bounds();
        let step = self.quantizer_steps[quantizer as usize];
        let watched = control.watches(key, (bounds.predicted, bounds.room), step);
        let sb_log2 = self.backend.superblock_log2();
        let grid = watch_grid(self.layer.coded_width, self.layer.coded_height, sb_log2);
        if let Some(control) = self.in_frame.as_mut() {
            if watched {
                control.set_frame_weights(superblock_weights(pre, grid, sb_log2, key));
            } else {
                control.clear_frame_weights();
            }
        }
    }

    /// Start in-frame rate control (`in_frame_rate.rs`) on a coding of the
    /// frame at `plan`, whose superblocks have side `1 << sb_log2`; a coding
    /// that may not be stopped (`may_stop` false) is only steered.
    fn begin_in_frame(&mut self, plan: &FramePlan, sb_log2: u8, may_stop: bool) {
        if self.in_frame.is_none() {
            return;
        }
        let Some(bounds) = self
            .layer
            .rate_control
            .as_ref()
            .map(RateController::in_frame_bounds)
        else {
            return;
        };
        let key = plan.frame_type == FrameType::Key;
        let step = self.quantizer_steps[plan.quantizer as usize];
        let watched = self
            .in_frame
            .as_ref()
            .is_some_and(|control| control.watches(key, (bounds.predicted, bounds.room), step));
        let start = watched.then(|| self.backend.coded_bits()).flatten();
        let Some(start) = start else {
            if let Some(control) = self.in_frame.as_mut() {
                control.skip_frame();
            }
            return;
        };
        let (columns, rows) = watch_grid(self.layer.coded_width, self.layer.coded_height, sb_log2);
        let units = columns * rows;
        let frame = FrameStart {
            key,
            steerable: plan.block_qp,
            units,
            predicted: bounds.predicted,
            room: bounds.room,
            recodable: bounds.recodable && may_stop && B::RECODE && !self.recode_identity,
            step,
            start,
        };
        if let Some(control) = self.in_frame.as_mut() {
            control.begin(&frame);
        }
    }

    /// Superblock `unit` (raster order) is coded: in-frame rate control
    /// reads the bits so far and says what follows (a wind-back only where
    /// the superblock is `undoable`).
    fn end_in_frame_superblock(&mut self, unit: usize, undoable: bool) -> Verdict {
        let Some(control) = self.in_frame.as_mut().filter(|c| c.watching()) else {
            return Verdict::Continue;
        };
        match self.backend.coded_bits() {
            Some(bits) => control.end_superblock(unit, bits, undoable),
            None => Verdict::Continue,
        }
    }

    /// Start a coding of the frame at `plan`: everything a coding gathers
    /// starts empty, so a re-coded frame keeps nothing of a discarded one.
    fn begin_coding(&mut self, plan: &FramePlan) {
        self.backend.begin_frame(plan);
        if let Some(term) = self.layer.reuse.as_mut() {
            term.begin_coding();
        }
        let control = self
            .threshold_control
            .as_ref()
            .filter(|_| plan.frame_type == FrameType::Inter);
        let role = self.frame_role(plan.index, &plan.references, false);
        let wanted = self
            .threshold_control
            .iter()
            .flat_map(ThresholdControl::adapted_tools);
        self.decision_scratch.evidence.clear(
            self.threshold_control.as_ref().map(ThresholdControl::band),
            wanted,
            control.map_or(Switch::Preset, |control| control.quarter(role)),
        );
    }

    /// Code one picture from `begin_frame` to `end_frame` (the partition
    /// walk, or a picture-owning backend's own pipeline), writing its
    /// reconstruction into `pictures.recon`; returns the unit, not yet
    /// committed (`CodecBackend::commit_frame`), or, when `stop` lets
    /// in-frame rate control stop it and it finds it past its room part-way
    /// (`rate.recode_early`), the size to re-code it from, without
    /// finishing it (`begin_frame` of the next coding starts afresh).
    #[allow(
        clippy::too_many_lines,
        reason = "the frame walk owns per-superblock planning"
    )]
    fn code_picture(
        &mut self,
        plan: &FramePlan,
        pictures: FramePictures<'_>,
        inter: Option<&FrameInputs<'_>>,
        planner_params: &PlannerParams,
        stop: StopRule,
    ) -> Result<Vec<u8>, StoppedCoding> {
        let FramePictures {
            source: coded_source,
            reference,
            secondary,
            recon,
        } = pictures;
        self.begin_coding(plan);
        let sb_log2 = self.backend.superblock_log2();
        self.begin_label_coding(plan, sb_log2);
        self.begin_in_frame(plan, sb_log2, stop == StopRule::InFrame);
        let columns = self.layer.coded_width.div_ceil(1 << sb_log2);
        let units = columns * self.layer.coded_height.div_ceil(1 << sb_log2);
        let halfway = (stop == StopRule::Halfway).then_some(units / 2);
        let mut stopped = None;
        let sb = 1usize << sb_log2;
        let regular_decision = self.frame_decision_params;
        let step = self.sample_step(self.frame_quantizer) as f32;
        // Taken for the frame (and put back for a recode).
        let light = self.frame_light.take();
        // The fine-floor gate's frame features (`fine_floor.rs`).
        let fine_floor_frame = inter.map_or((0.0, 0.0), |frame| {
            (
                frame
                    .motion
                    .map_or(0.0, |m| camera_displacement(m, self.layer.width)),
                frame.pre.map_or(0.0, |p| p.static_fraction),
            )
        });
        let mut light_superblocks = light.iter().flat_map(|l| l.map.iter().copied());
        // The class the settings are in: they change only where the class
        // does (H.264 classes every macroblock; copying the decision
        // parameters per superblock cost ~1.5 % there).
        let mut coded_light = false;
        'rows: for y in (0..self.layer.coded_height).step_by(sb) {
            for x in (0..self.layer.coded_width).step_by(sb) {
                self.backend.begin_superblock(x, y);
                let sb = Block::square(x, y, sb_log2);
                let light_params = light
                    .as_ref()
                    .filter(|_| light_superblocks.next() == Some(true))
                    .map(|l| &l.params);
                if light.is_some() && light_params.is_some() != coded_light {
                    coded_light = light_params.is_some();
                    self.frame_decision_params =
                        light_params.map_or(regular_decision, |p| p.decision);
                    self.backend.use_light_params(coded_light);
                }
                let planner_params = light_params.map_or(planner_params, |p| &p.planner);
                // The fine-floor gate (`fine_floor.rs`) raises this
                // superblock's floor one level where it closes.
                let raised_floor;
                let planner_params =
                    match inter.zip(FineFloorGate::of(planner_params, fine_floor_frame, step)) {
                        Some((frame, gate))
                            if !gate.opens(
                                frame.motion,
                                &sb,
                                (self.layer.coded_width, self.layer.coded_height),
                            ) =>
                        {
                            raised_floor = fine_floor::raised(planner_params);
                            &raised_floor
                        }
                        _ => planner_params,
                    };
                // Inter frames choose their partition per superblock first
                // (VLPD0); key frames code the configured fixed size, or
                // with `planner.intra_plan` a planned one (`key_plan`). A
                // superblock that cannot be split (H.264 macroblocks) has
                // nothing to plan.
                let splittable = sb_log2 > self.backend.min_block_log2();
                // Intra refresh constrains leaves by region; rectangular
                // shapes stay out of it (`CodecBackend::partition_shapes`).
                let shapes = self.window.is_none();
                let decision = &self.frame_decision_params;
                let noise = (planner_params.noise_split >= 0.5 && decision.noise_skip > 0.)
                    .then_some(NoiseTest {
                        map: &self.layer.noise_map,
                        scale: decision.noise_skip,
                        chroma_ratio: decision.noise_skip_chroma_ratio,
                        step: self.sample_step(self.frame_quantizer) as f32,
                    });
                let planner = |motion, pre| Planner {
                    dsp: &self.dsp,
                    backend: &self.backend,
                    source: coded_source,
                    reference,
                    motion,
                    pre,
                    noise,
                    width: self.layer.coded_width,
                    height: self.layer.coded_height,
                    params: planner_params,
                    shapes,
                    lambda_headroom: plan.lambda_scale,
                    step,
                    leaf_bits: OnceCell::new(),
                    margins: RefCell::default(),
                    // The label dump records the value network's inputs; the
                    // stop network reads them.
                    record_inputs: self.labels.is_some()
                        || (planner_params.rd_split >= 0.5
                            && rd_split::stop_reads_inputs(planner_params)),
                    selector_terms: crate::split_selector::selector_terms(planner_params),
                };
                let key_mode = KeyPlanMode::resolve(planner_params.intra_plan, &self.backend);
                let plan = match inter {
                    // Random decisions choose every split themselves.
                    _ if self.random.is_some() => None,
                    Some(frame) if splittable => {
                        let planner = planner(frame.motion, frame.pre);
                        let record_margins = planner_params.rd_split >= 0.5
                            && (planner_params.rd_split_margin > 1.0
                                || rd_split::context_reads_margins(planner_params))
                            && self.rd_split_applies()
                            && !self.replays_partitions();
                        let partition = planner.plan(sb, record_margins);
                        self.plan_margins = planner.take_margins();
                        record_depth_evidence(
                            &planner,
                            &sb,
                            &partition,
                            &mut self.decision_scratch.evidence,
                        );
                        Some(self.replayed(&sb).unwrap_or(partition))
                    }
                    None if splittable && key_mode != KeyPlanMode::Fixed => {
                        Some(self.plan_key_superblock(sb, coded_source, planner_params, key_mode))
                    }
                    _ => None,
                };
                // A superblock in-frame rate control may wind back is coded
                // inside a node trial it can undo.
                let undoable = inter.is_some()
                    && !watches_rows(sb_log2)
                    && self.rd_split_applies()
                    && sb.is_inside(self.layer.coded_width, self.layer.coded_height)
                    && self.in_frame.as_ref().is_some_and(InFrameRate::winds_back);
                let offsets = (self.block_offsets.len(), self.in_frame_offsets.len());
                if undoable {
                    self.backend.begin_node_trial(&sb, recon);
                }
                let pictures = FramePictures {
                    source: coded_source,
                    reference,
                    secondary,
                    recon: &mut *recon,
                };
                let light = light_params.is_some();
                self.code_labelled_superblock(
                    sb,
                    plan.as_ref(),
                    pictures,
                    inter,
                    (planner_params, light),
                );
                let row_watch = watches_rows(sb_log2);
                let unit = if row_watch {
                    y >> sb_log2
                } else {
                    (y >> sb_log2) * columns + (x >> sb_log2)
                };
                let row_end = x + (1 << sb_log2) >= self.layer.coded_width;
                let mut verdict = if row_watch && !row_end {
                    Verdict::Continue
                } else {
                    self.end_in_frame_superblock(unit, undoable)
                };
                if verdict == Verdict::WindBack {
                    // Coded again from where it began, at the raised offset,
                    // with the same planned partition.
                    self.backend.undo_node_trial(recon);
                    self.block_offsets.truncate(offsets.0);
                    self.in_frame_offsets.truncate(offsets.1);
                    if let (Some(control), Some(bits)) =
                        (self.in_frame.as_mut(), self.backend.coded_bits())
                    {
                        control.wound_back(bits);
                    }
                    let pictures = FramePictures {
                        source: coded_source,
                        reference,
                        secondary,
                        recon: &mut *recon,
                    };
                    self.code_labelled_superblock(
                        sb,
                        plan.as_ref(),
                        pictures,
                        inter,
                        (planner_params, light),
                    );
                    verdict = self.end_in_frame_superblock(unit, false);
                }
                if undoable {
                    self.backend.end_node_trial();
                }
                if halfway == Some((y >> sb_log2) * columns + (x >> sb_log2)) {
                    stopped = Some(StoppedCoding { bits: 0. });
                    break 'rows;
                }
                if let Verdict::Recode(bits) = verdict {
                    stopped = Some(StoppedCoding { bits });
                    break 'rows;
                }
            }
        }
        if let Some(control) = self.in_frame.as_mut() {
            control.finish();
        }
        if light.is_some() {
            self.frame_decision_params = regular_decision;
            self.backend.use_light_params(false);
        }
        self.frame_light = light;
        if let Some(stopped) = stopped {
            return Err(stopped);
        }
        if B::FILTER_BEFORE_END {
            self.backend.filter_frame(coded_source, recon);
        }
        Ok(self.backend.end_frame())
    }

    /// Inter superblock `sb`'s partition from the transplant log, when
    /// replaying one that holds it in a form this encode can code
    /// (`transplant.rs`).
    fn replayed(&self, sb: &Block) -> Option<Partition> {
        let Some(Transplant::Replay(log)) = &self.transplant else {
            return None;
        };
        log.replay(self.next_index.saturating_sub(1), sb, |node| {
            self.legal_shapes(node)
        })
    }

    /// File inter superblock `sb`'s coded `partition` when recording
    /// (`transplant.rs`).
    pub(super) fn record_partition(&mut self, sb: &Block, partition: &Partition) {
        let frame = self.next_index.saturating_sub(1);
        if let Some(Transplant::Record(log)) = self.transplant.as_mut() {
            log.record(frame, sb, partition);
        }
    }

    /// Whether inter superblocks report their searched partition for the
    /// transplant log.
    pub(super) fn records_partitions(&self) -> bool {
        matches!(self.transplant, Some(Transplant::Record(_)))
    }

    /// Whether inter superblocks code a transplanted partition, without the
    /// RD partition search.
    pub(super) fn replays_partitions(&self) -> bool {
        matches!(self.transplant, Some(Transplant::Replay(_)))
    }

    /// The key plan of superblock `sb` (`key_plan.rs`); its margins are kept
    /// for the key RD search's gate (`planner.key_rd_margin`).
    fn plan_key_superblock(
        &mut self,
        sb: Block,
        source: &Picture,
        params: &PlannerParams,
        mode: KeyPlanMode,
    ) -> Partition {
        let (partition, margins) = KeyPlanner {
            dsp: &self.dsp,
            backend: &mut self.backend,
            source,
            width: self.layer.coded_width,
            height: self.layer.coded_height,
            params,
            mode,
        }
        .plan(sb);
        self.plan_margins = Some(margins);
        partition
    }

    /// Whether inter frame `index` (analysed against `planned`'s reference)
    /// is a scene cut (`scene.rs`).
    fn is_scene_cut(
        &mut self,
        index: u64,
        planned: &FrameReferences,
        analyzer: Option<&Analyzer>,
    ) -> bool {
        let costs = analyzer
            .and_then(Analyzer::analysis)
            .and_then(FrameCosts::of);
        match (self.scene.as_mut(), costs) {
            (Some(scene), Some(costs)) => scene.is_cut(
                costs,
                planned.reference.distance,
                index - self.layer.last_key_index,
            ),
            _ => false,
        }
    }

    /// Frame `index`'s scheduled type and what the long-term reference
    /// policy does to it (`ltr.rs`): a loss without a usable LTR forces a
    /// key frame.
    fn ltr_plan(&mut self, index: u64) -> (FrameType, LtrAction) {
        let scheduled = self.frame_type(index);
        let base = self.structure.peek(index, false).position.layer == 0;
        match (self.ltr.as_ref(), scheduled) {
            (Some(ltr), FrameType::Inter) => match ltr.action(index, base) {
                LtrAction::Key => (FrameType::Key, LtrAction::None),
                action => (FrameType::Inter, action),
            },
            _ => (scheduled, LtrAction::None),
        }
    }

    /// With `refs.second_hold_rearm`: an HQ frame's coded luma error, for
    /// the held reference to follow a still scene's refinement.
    fn observe_hq_quality(
        &mut self,
        index: u64,
        key: bool,
        references: &FrameReferences,
        (source, recon): (&Picture, &Picture),
    ) {
        if !self
            .layer
            .second
            .as_ref()
            .is_some_and(SecondReference::rearms)
            || !(key || self.is_high_quality(index, references.position))
        {
            return;
        }
        let size = (self.layer.width, self.layer.height);
        let sse = self
            .dsp
            .sse(&visible_luma(source, size), &visible_luma(recon, size));
        let error = sse as f64 / (size.0 * size.1) as f64;
        if let Some(second) = self.layer.second.as_mut() {
            second.hq_coded(references, error);
        }
    }

    /// Start counting the coding of a frame's use of the second reference
    /// (`refs.second_hold_min_share`; again when the frame is re-coded).
    fn begin_second_usage(&mut self) {
        if let Some(second) = self.layer.second.as_mut() {
            second.begin_usage();
        }
    }

    /// `references` of frame `index` (a key frame if `key`, at the cadence
    /// strength `depth` if known) with the second reference, when it is on
    /// (`second_reference.rs`).
    /// The block-quantizer offset the frame realised, in QP units, under rate
    /// control: block AQ's offsets weighted as the rate model reads them, or the
    /// in-frame steering's by area (`None` without rate control or offsets).
    fn realised_block_offset(&self) -> Option<f64> {
        let controller = self.layer.rate_control.as_ref()?;
        if self.repeating {
            return None;
        }
        if let Some(aq) = self
            .layer
            .aq
            .as_ref()
            .filter(|aq| aq.active() && !self.still.without_offsets)
        {
            let offsets = self.block_offsets.iter().copied();
            Some(controller.effective_block_offset(aq.total_weight(), offsets))
        } else if !self.in_frame_offsets.is_empty() {
            // Steered without block AQ: the offsets are weighted by area.
            let area = (self.layer.coded_width * self.layer.coded_height) as f64;
            let offsets = self.in_frame_offsets.iter().copied();
            Some(controller.effective_block_offset(area, offsets))
        } else {
            None
        }
    }

    pub(super) fn with_second(
        &self,
        index: u64,
        references: FrameReferences,
        (key, depth): (bool, Option<f64>),
    ) -> FrameReferences {
        match self.layer.second.as_ref() {
            Some(second) => {
                let high_quality = key || self.is_high_quality(index, references.position);
                let applied = second.apply(index, high_quality, references);
                second.planned(applied, high_quality, depth)
            }
            None => references,
        }
    }

    /// The QP cadence's strength (0..1) on the frame `analysis` describes:
    /// how much finer than its neighbours a high-quality frame is coded
    /// here, the depth the unified reference plan reads (`TierPlan`).
    pub(super) fn cadence_depth(
        &self,
        index: u64,
        analysis: &FrameAnalysis,
        references: &FrameReferences,
    ) -> f64 {
        let motion = analysis
            .motion
            .as_ref()
            .and_then(|m| cadence_motion_hint(m, references.reference.distance, self.layer.width));
        let own = self.own_cadence_strength(f64::from(analysis.pre.static_fraction), motion);
        self.inherited_depth(index, motion)
            .map_or(own, |depth| own.max(depth))
    }

    /// The cadence's own strength for a frame's still-block share and
    /// motion: at fixed QP under camera motion it follows the scene
    /// (`MovingCamera`), whatever the fixed-QP profile's scaling.
    fn own_cadence_strength(&self, static_fraction: f64, motion: Option<MotionHint>) -> f64 {
        let cadence = &self.layer.cadence;
        if self.layer.rate_control.is_none() && self.moving_camera.cadence_follows_scene() {
            cadence.scene_strength(static_fraction, motion)
        } else {
            cadence.strength(static_fraction, motion)
        }
    }

    /// Feed the frame's content features to the inheritance's depth model
    /// (`structure.depth_model`): still-block share, camera motion, the share of
    /// dark 8x8 cells and mean luma.
    fn observe_depth_model_features(
        &mut self,
        analyzer: Option<&Analyzer>,
        planned: &FrameReferences,
    ) {
        let width = self.layer.width;
        let Some(inheritance) = self
            .layer
            .inheritance
            .as_mut()
            .filter(|i| i.uses_depth_model())
        else {
            return;
        };
        let Some(analysis) = analyzer.and_then(Analyzer::analysis) else {
            return;
        };
        let camera_motion = analysis
            .motion
            .as_ref()
            .and_then(|m| cadence_motion_hint(m, planned.reference.distance, width))
            .map_or(0.0, |hint| hint.camera_motion);
        let cells = analysis.pre.blocks.cells();
        let dark_share = if cells.is_empty() {
            0.0
        } else {
            cells
                .iter()
                .filter(|cell| cell.mean < DARK_CELL_MEAN)
                .count() as f64
                / cells.len() as f64
        };
        inheritance.observe_frame_features(
            f64::from(analysis.pre.static_fraction),
            camera_motion,
            dark_share,
            f64::from(analysis.pre.mean_luma),
        );
    }

    /// The least cadence strength the measured inheritance asks for
    /// (`structure.inherited_depth`), under rate control only: at a fixed
    /// quantizer the tiers never follow content.
    fn inherited_depth(&self, index: u64, motion: Option<MotionHint>) -> Option<f64> {
        let rc = self.layer.rate_control.as_ref()?;
        let inheritance = self.layer.inheritance.as_ref()?;
        let fsk = index.saturating_sub(self.layer.last_key_index);
        let pixels = (self.layer.width * self.layer.height).max(1) as f64;
        let ctx = crate::inheritance::DepthContext {
            key_period: self.key_period(),
            frames_since_key: fsk,
            in_startup: self.layer.cadence.in_startup(fsk),
            bpp: rc.inter_frame_budget() / pixels,
            camera_motion: motion.map(|m| m.camera_motion),
        };
        inheritance.depth_for(ctx)
    }

    /// The layer propagation term's view of frame `index`
    /// (`layer_propagation.rs`): its layer and the room above it, the
    /// ladder's top-layer offset on a layered frame; on a flat one 0 for a
    /// cadence base frame (1 otherwise) and the cadence's base-to-leaf span
    /// at the frame's strength `depth`, under rate control only. At a fixed
    /// quantizer the flat form's offsets only coarsen blocks, with nothing
    /// to spend the saved bits on: fixed-QP A/B (draw fixedqp-2026-11-21,
    /// 24 clips) J +1.3 % AV2 Default, +1.0 % VP9 Fast, +0.6 % HEVC Default,
    /// +0.5 % VVC Default, +0.2 % AV1 Fast, against CBR probe gains
    /// (docs/handoff/temporal-lambda.md).
    fn propagation_role(
        &self,
        index: u64,
        references: &FrameReferences,
        depth: Option<f64>,
    ) -> (u8, f64) {
        let position = references.position;
        if !position.is_flat() {
            let ladder = if self.layer.rate_control.is_some() {
                self.layer.rate_control_ladder
            } else {
                self.layer.ladder
            };
            return (position.layer, ladder.top_offset());
        }
        let layer = u8::from(!self.is_high_quality(index, position));
        let strength = depth.unwrap_or(0.);
        let span = self
            .layer
            .rate_control
            .as_ref()
            .map_or(0., |controller| controller.cadence_span(index, strength));
        (layer, span)
    }

    /// Whether inter frame `index` is coded finer than its neighbours: a
    /// base-layer frame of a layered structure, or a base frame of the QP
    /// cadence (`qp_cadence.rs`) in a flat one.
    fn is_high_quality(&self, index: u64, position: LayerPosition) -> bool {
        if position.layers > 1 {
            position.layer == 0
        } else {
            self.layer
                .cadence
                .role(index.saturating_sub(self.layer.last_key_index))
                == CadenceRole::Base
        }
    }

    /// `references` changed by the LTR policy's `action` for frame `index`.
    fn with_ltr(
        &self,
        action: LtrAction,
        index: u64,
        references: FrameReferences,
    ) -> FrameReferences {
        match self.ltr.as_ref() {
            Some(ltr) => ltr.apply(action, index, references),
            None => references,
        }
    }

    /// The receiver decoded frame `index` (`FrameMetadata::index`): with a
    /// long-term reference (`ltr.interval`), a marked frame becomes the
    /// one recovery can predict from. Ignored otherwise.
    pub fn acknowledge(&mut self, index: u64) {
        if let Some(ltr) = self.ltr.as_mut() {
            ltr.acknowledge(index);
        }
    }

    /// The receiver decoded spatial layer `spatial_id` (`0` = base, `1` =
    /// enhancement) of temporal unit `tu_index` (`FrameMetadata::index`).
    /// On single-layer streams (`spatial_id == 0`), delegates to
    /// [`Self::acknowledge`].
    pub fn acknowledge_layer(&mut self, tu_index: u64, spatial_id: u8) {
        if self.spatial.is_none() && spatial_id == 0 {
            self.acknowledge(tu_index);
        }
    }

    /// The receiver is missing frame `index` (`FrameMetadata::index`): the
    /// next frame predicts from the acknowledged long-term reference, or is a
    /// key frame when there is none (or LTR is off). On a spatial stream,
    /// reporting a whole temporal unit lost checks both `S0` and `S1` at
    /// `tu_index = index` (use [`Self::report_layer_loss`] when only one
    /// spatial layer was lost). A lost top-layer frame, which no frame
    /// references, needs nothing, nor does a frame older than the last key
    /// or recovery frame: that frame already repairs it. Receivers repeat
    /// reports while their picture stays frozen (the report itself may be
    /// lost), so without this every repeat that arrives while the repair is
    /// in flight would cost another key frame.
    pub fn report_loss(&mut self, index: u64) {
        if let Some(num_layers) = self.spatial.as_ref().map(|s| s.config.layers) {
            for sid in 0..num_layers {
                self.report_layer_loss(index, sid);
            }
            return;
        }
        if !self.feedback.needs_repair(index) {
            return;
        }
        match self.ltr.as_mut() {
            Some(ltr) => ltr.report_loss(index),
            None => self.request_keyframe(),
        }
    }

    /// The receiver is missing spatial layer `spatial_id` (`0..layers`) of
    /// temporal unit `tu_index` (`FrameMetadata::index`):
    ///
    /// - Losing a referenced `S0` picture (`spatial_id == 0`) forces a full
    ///   key temporal unit.
    /// - Losing a referenced `S1` or `S2` picture while lower layers are intact
    ///   schedules an enhancement-only repair on the next temporal unit (`S0`
    ///   and any intact lower layer code as ordinary inter frames). In full
    ///   SVC it repairs that layer and every layer above it from the layer
    ///   immediately below; in K-SVC and independent layers it repairs that
    ///   layer alone, intra-only, unless the loss was in the key unit, which
    ///   the layers above predicted from.
    /// - Losing a non-reference picture (top temporal layer) or a picture
    ///   older than the layer's last repair needs no action.
    pub fn report_layer_loss(&mut self, tu_index: u64, spatial_id: u8) {
        let Some(spatial) = self.spatial.as_mut() else {
            if spatial_id == 0 {
                self.report_loss(tu_index);
            }
            return;
        };
        let num_layers = spatial.config.layers;
        if spatial_id >= num_layers {
            return;
        }
        let structure = &mut spatial.structure;
        let lost_layer_needs_repair = structure.needs_layer_repair(spatial_id, tu_index);
        if lost_layer_needs_repair {
            if spatial_id == 0 {
                self.request_keyframe();
                return;
            }
            structure.request_layer_repair(spatial_id);
        }
        // The layers above predicted from the lost picture only in a unit that
        // crosses layers: every unit in full SVC, the key unit alone in K-SVC.
        let lost_unit_is_key = tu_index == structure.last_key_tu();
        if !spatial.config.inter_layer.active_in_unit(lost_unit_is_key) {
            return;
        }
        for upper in (spatial_id + 1)..num_layers {
            if structure.needs_layer_repair(upper, tu_index) {
                structure.request_layer_repair(upper);
                // A full-SVC repair already cascades upward from `upper`.
                if !lost_layer_needs_repair || spatial.config.inter_layer.active_on_inter_unit() {
                    break;
                }
            }
        }
    }

    /// Whether the next input, written into the analyzer's buffer, is held
    /// (`hold.rs`). Never the first frame, a requested key frame or a loss
    /// recovery, which the receiver needs.
    fn hold_due(&mut self) -> bool {
        let waiting = self.force_keyframe
            || self
                .ltr
                .as_ref()
                .is_some_and(LongTermReference::recovery_pending);
        if self.hold.is_none() || waiting || self.next_index == 0 {
            return false;
        }
        let planned = self.structure.peek(self.next_index, false);
        let link = planned.link();
        let steps = (
            self.reuse_sample_step(),
            self.sample_step(self.layer.last_quantizer),
        );
        let mut analyzer = self.layer.analyzer.take();
        if let Some(a) = analyzer.as_mut() {
            // The search settings of the frame's expected role, so its
            // pre-analysis builds the half-resolution level only when its
            // search will read it (`Analyzer::pre_analyze_next`); the frame's
            // own analysis applies its final role.
            self.apply_search_role(a, self.next_index, &planned, false);
        }
        self.layer.analyzer = analyzer;
        let (Some(hold), Some(analyzer)) = (self.hold.as_mut(), self.layer.analyzer.as_mut())
        else {
            return false;
        };
        if analyzer.analysis().is_none() {
            return false;
        }
        // The frame's analysis reuses this pre-analysis if it is coded.
        analyzer.pre_analyze_next(link);
        let referenced = analyzer.references_last(link);
        let repeat = match (analyzer.next_pre(), analyzer.analysis()) {
            (Some(next), Some(last)) => {
                let last = LastCoded {
                    pre: &last.pre,
                    referenced,
                };
                hold.repeats(next, last, steps)
            }
            _ => false,
        };
        let fullness = self
            .layer
            .rate_control
            .as_ref()
            .map(|rc| rc.stats().buffer_fullness);
        let held = hold.holds(repeat);
        held && !hold.refine_instead(fullness)
    }

    /// Hold the last picture instead of coding a frame: only time passes,
    /// which rate control accounts as an interval without bits.
    fn hold_frame(&mut self, timestamp_us: Option<i64>) -> EncodedFrame {
        let time = self.timeline.next(timestamp_us);
        if let Some(controller) = self.layer.rate_control.as_mut() {
            controller.set_frame_interval(time.interval);
            controller.hold_frame();
        }
        EncodedFrame {
            data: Vec::new(),
            metadata: FrameMetadata {
                index: self.next_index,
                frame_type: FrameType::Inter,
                temporal_layer: 0,
                timestamp_us: time.timestamp_us,
                dropped: false,
                held: true,
                quantizer: 0,
                references: ReferencedFrames::NONE,
            },
            layers: Vec::new(),
        }
    }

    /// Whether rate control drops the next frame (see
    /// [`RateControlConfig::frame_drop`]). Key frames, periodic or
    /// requested, are never dropped, nor are frames other frames reference.
    fn drop_due(&self, index: u64) -> bool {
        let inter = self.backend.supports_inter()
            && !self.force_keyframe
            && !self.periodic_key(index)
            && !self.ltr.as_ref().is_some_and(|l| l.protects(index));
        let protected = self.structure.protected_after_next();
        inter
            && self.structure.next_is_droppable()
            && self
                .layer
                .rate_control
                .as_ref()
                .is_some_and(|rc| rc.should_drop(protected))
    }

    /// Skip frame `index`: nothing is coded and no reference, analysis or
    /// backend state changes. A flat chain's next frame takes the same
    /// index (frame numbers and POCs stay contiguous); in a layered
    /// structure the dropped top-layer frame keeps its place, so the
    /// pattern continues unchanged.
    fn drop_frame(&mut self, index: u64, time: FrameTime) -> EncodedFrame {
        if let Some(controller) = self.layer.rate_control.as_mut() {
            controller.drop_frame();
        }
        // A layered mini-GOP keeps the dropped frame's place; a flat chain
        // gives its index to the next frame.
        let layer = match self.structure.drop_next() {
            Some(layer) => {
                self.next_index += 1;
                layer
            }
            None => 0,
        };
        EncodedFrame {
            data: Vec::new(),
            metadata: FrameMetadata {
                index,
                frame_type: FrameType::Inter,
                temporal_layer: layer,
                timestamp_us: time.timestamp_us,
                dropped: true,
                held: false,
                quantizer: 0,
                references: ReferencedFrames::NONE,
            },
            layers: Vec::new(),
        }
    }

    /// How `node` is partitioned. A node across an edge of the intra-refresh
    /// range splits where it can, so its clean or dirty part is not coded
    /// intra as well.
    fn children<'p>(
        &mut self,
        node: &Block,
        plan: Option<&'p Partition>,
        inter: bool,
    ) -> NodeSplit<'p> {
        let straddles = inter
            && node.log2_size() > self.backend.min_block_log2()
            && self
                .window
                .as_ref()
                .is_some_and(|w| w.straddles_intra_edge(node));
        let splittable = node.log2_size() > self.backend.min_block_log2();
        let too_large = node.log2_size() > self.backend.max_leaf_log2();
        let split = match plan {
            Some(Partition::Split(children)) => return NodeSplit::Split(Some(children)),
            Some(Partition::Shaped(shape, parts)) => return NodeSplit::Shaped(*shape, Some(parts)),
            Some(Partition::Leaf(_)) => straddles,
            // Outside the picture or across the refresh edge the split is
            // forced; elsewhere any legal size, chosen at random.
            None if self.random.is_some() => {
                splittable
                    && (straddles
                        || too_large
                        || !node.is_inside(self.layer.coded_width, self.layer.coded_height)
                        || self.random.as_mut().is_some_and(RandomDecisions::split))
            }
            None => {
                // A repeat frame's leaves are as large as the codec allows:
                // each costs a skip, whatever its size.
                let leaf_log2 = if self.repeating {
                    self.backend.max_leaf_log2()
                } else {
                    self.config.block_log2.min(self.backend.max_leaf_log2())
                };
                straddles
                    || should_split(
                        node,
                        leaf_log2,
                        self.backend.min_block_log2(),
                        self.layer.coded_width,
                        self.layer.coded_height,
                    )
            }
        };
        if split {
            return NodeSplit::Split(None);
        }
        let random_shape = if B::RECTANGULAR_LEAVES
            && inter
            && self.window.is_none()
            && node.is_inside(self.layer.coded_width, self.layer.coded_height)
        {
            let shapes = self.legal_shapes(node);
            self.random.as_mut().and_then(|random| random.shape(shapes))
        } else {
            None
        };
        random_shape.map_or(NodeSplit::Leaf, |shape| NodeSplit::Shaped(shape, None))
    }

    /// The rectangular shapes the backend declares for `node` whose leaves
    /// are no smaller than its minimum block.
    fn legal_shapes(&self, node: &Block) -> ShapeSet {
        let min = self.backend.min_block_log2();
        self.backend
            .partition_shapes(node)
            .iter()
            .filter(|shape| shape.log2_min_side(node.log2_size()) >= min)
            .fold(ShapeSet::EMPTY, ShapeSet::with)
    }

    /// The mode of inter-frame leaf `node` and the reference it predicts
    /// from, or
    /// `None` when the backend accepted it as a static skip and has already
    /// coded it. `clean`: the node is in the intra-refresh clean region,
    /// whose vectors the window constrains.
    fn decide_inter_leaf(
        &mut self,
        node: &Block,
        planned: Option<&LeafStats>,
        frame: &FrameInputs<'_>,
        pictures: FramePictures<'_>,
        clean: bool,
        refined: bool,
    ) -> Option<LeafChoice> {
        let FramePictures {
            source,
            reference,
            secondary,
            recon,
        } = pictures;
        let window = self.window.take();
        let refresh = window.as_ref().filter(|_| clean);
        if self.random.is_some() {
            let inter_layer = self
                .spatial
                .as_ref()
                .is_some_and(|s| s.coding_inter_layer_step.is_some());
            let mode = self.random_leaf_choice(node, inter_layer && secondary.is_some(), refresh);
            self.window = window;
            return Some((mode.0, mode.1, None, PricingHint::NONE));
        }
        if self.repeating && self.code_repeat_leaf(node, (source, reference), recon) {
            self.window = window;
            return None;
        }
        // The static skip proves a zero residual at the frame quantizer; a
        // block coded finer than the frame (`refined`) may well code one.
        let skip_allowed = !refined
            && refresh.is_none_or(|w| {
                self.backend
                    .static_skip_vector(node)
                    .is_some_and(|mv| w.allows(node, mv))
            });
        // Skip testing, motion ranking and pricing share one analysis verdict.
        let static_block = frame.pre.is_some_and(|pre| is_static::<B>(pre, node));
        let mode = if skip_allowed
            && self.try_static_skip(node, frame, (source, reference), recon, static_block)
        {
            let skip = self
                .backend
                .static_skip_vector(node)
                .expect("a backend accepting static skips has a skip vector");
            self.code_static_skip(node, skip, (source, reference), recon);
            None
        } else if self.frame_decision_params.static_zero_vector >= 0.5
            && frame.still.is_some_and(|still| is_still::<B>(still, node))
            && refresh.is_none_or(|w| w.allows(node, Mv::ZERO))
        {
            // A reliably still block (static for several frames, away from
            // motion; `still.*`): the zero vector, with no candidate or
            // intra trials.
            Some(ZERO_VECTOR_LEAF)
        } else {
            let search_mv =
                planned.map_or_else(|| search_vector::<B>(frame.motion, node), |p| p.search_mv);
            // The pre-analysis moments give the source block's energy
            // exactly, saving a variance pass, where its 8x8 cells hold only
            // picture samples (not replicated border).
            let (width, height) = leaf_extent::<B>(node);
            let inside = node.x + width <= self.layer.width && node.y + height <= self.layer.height;
            let analysis = (frame.pre.filter(|_| inside), inside && static_block);
            let primary = choose_inter_mode(
                &self.dsp,
                &mut self.backend,
                node,
                (source, reference, Some(&*recon)),
                search_mv,
                planned,
                &mut self.decision_scratch,
                analysis,
                &self.frame_decision_params,
                refresh,
            );
            let (choice, chosen) = self.choose_reference(
                node,
                (source, reference),
                secondary.zip(frame.secondary_ratio),
                (search_mv, primary),
                analysis,
                refresh,
            );
            if self.backend.wants_sub_block_motion()
                && is_square_leaf::<B>(node)
                && node.log2_size() > SUB_BLOCK_LOG2
            {
                let mut vectors = [Mv::ZERO; MAX_SUB_BLOCKS];
                // The motion search measured the primary reference only.
                let count = if chosen == ReferenceChoice::Primary {
                    sub_block_vectors(frame.motion, node, &mut vectors)
                } else {
                    0
                };
                if let Some(window) = refresh {
                    forbid_sub_block_vectors(window, node, &mut vectors[..count]);
                }
                self.backend.sub_block_motion(node, &vectors[..count]);
            }
            // No pricing against the secondary reference (no hint), nor on
            // blocks pre-analysis marks static: there an intra leaf
            // refreshes what later frames copy, which the coded cost of
            // one CU does not see (pool L3, CBR: still-camera clips lost up
            // to +10 % J with it).
            let pricing = if chosen == ReferenceChoice::Primary && !static_block {
                choice.pricing
            } else {
                PricingHint::NONE
            };
            Some((
                choice.mode.prediction(),
                chosen,
                choice.mode.intra_fallback(),
                pricing,
            ))
        };
        self.window = window;
        mode
    }

    /// Code leaf `node` of a repeat frame as the backend's skip at the zero
    /// vector ([`StaticSkipTest::repeat`]), unless the backend declines.
    fn code_repeat_leaf(
        &mut self,
        node: &Block,
        (source, reference): (&Picture, &Picture),
        recon: &mut Picture,
    ) -> bool {
        let repeat = StaticSkipTest {
            repeat: true,
            ..StaticSkipTest::default()
        };
        let accepted = self
            .backend
            .try_static_skip(node, source, reference, recon, repeat);
        if accepted {
            self.code_static_skip(node, Mv::ZERO, (source, reference), recon);
        }
        accepted
    }

    /// The syntax stress test's random leaf choice (`debug.random_decisions`):
    /// out of line, away from every real encode's leaf decision.
    #[cold]
    #[inline(never)]
    fn random_leaf_choice(
        &mut self,
        node: &Block,
        allow_secondary: bool,
        refresh: Option<&RefreshWindow>,
    ) -> (PredictionMode, ReferenceChoice) {
        let allow_compound = allow_secondary && self.backend.supports_compound_reference();
        let allow_pair = self.backend.supports_primary_pair();
        let random = self.random.as_mut().expect("random mode checked");
        let mut chosen = if allow_compound && random.compound_reference() {
            ReferenceChoice::Compound {
                secondary_mv: Mv::ZERO,
            }
        } else if allow_pair && random.compound_reference() {
            ReferenceChoice::PrimaryPair {
                second_mv: Mv::ZERO,
            }
        } else if allow_secondary && random.secondary_reference() {
            ReferenceChoice::Secondary
        } else {
            ReferenceChoice::Primary
        };
        if chosen == ReferenceChoice::Secondary {
            self.backend.select_reference(node, chosen);
        }
        // Far vectors go as far as the scaled seed does (`within_reach`).
        let reach = |size: usize| size.min(B::VECTOR_REACH) as i32;
        let mode = random_inter_leaf(
            &mut self.backend,
            random,
            node,
            self.decision_scratch.candidates_mut(),
            (
                reach(self.layer.coded_width),
                reach(self.layer.coded_height),
            ),
            refresh,
        );
        if let ReferenceChoice::Compound { .. } | ReferenceChoice::PrimaryPair { .. } = chosen {
            chosen = if matches!(mode, PredictionMode::IntraDc) {
                ReferenceChoice::Primary
            } else {
                let candidate = self.random.as_mut().expect("random mode checked").vector();
                let second = if refresh.is_none_or(|w| w.allows(node, candidate)) {
                    candidate
                } else {
                    Mv::ZERO
                };
                let PredictionMode::Inter(first) = mode else {
                    unreachable!("intra was handled above")
                };
                // A pair the codec cannot signal (a far first vector whose
                // difference has no code in the pair's syntax) stays single.
                match chosen {
                    ReferenceChoice::Compound { .. }
                        if self
                            .backend
                            .compound_mode_bits(node, first, second)
                            .is_finite() =>
                    {
                        ReferenceChoice::Compound {
                            secondary_mv: second,
                        }
                    }
                    ReferenceChoice::PrimaryPair { .. }
                        if self
                            .backend
                            .primary_pair_mode_bits(node, first, second)
                            .is_finite() =>
                    {
                        ReferenceChoice::PrimaryPair { second_mv: second }
                    }
                    _ => ReferenceChoice::Primary,
                }
            };
        } else if chosen == ReferenceChoice::Secondary && matches!(mode, PredictionMode::IntraDc) {
            self.backend
                .select_reference(node, ReferenceChoice::Primary);
            chosen = ReferenceChoice::Primary;
        }
        (mode, chosen)
    }

    /// `primary`, the choice of `node` against the primary reference, the
    /// `secondary` reference's (with its distance ratio to the primary's), or
    /// their equal-weight compound average where that costs less; `search_mv`
    /// seeds the secondary's vectors. The backend's reference selection is
    /// left on the returned choice.
    fn with_secondary(
        &mut self,
        node: &Block,
        source: &Picture,
        primary_reference: &Picture,
        secondary: Option<(&Picture, f32)>,
        (search_mv, primary): (Mv, Choice),
        analysis: (Option<&PreAnalysis>, bool),
        refresh: Option<&RefreshWindow>,
    ) -> (Choice, ReferenceChoice) {
        // What it skips: the second reference's mode decision (and the
        // compound trial) where `decision.second_reference` is 0, as on
        // light superblocks. Why safe: the primary reference is the closer
        // one and always tried; the choice only narrows. Effect:
        // `docs/handoff/light-pipeline.md`.
        let tried = self.frame_decision_params.second_reference >= 0.5;
        let Some((secondary, ratio)) = secondary.filter(|_| tried) else {
            return (primary, ReferenceChoice::Primary);
        };
        // Against an older temporal secondary reference (`ratio > 0`), an
        // intra primary choice already beat the closer primary inter reference
        // with the residual's DC error counted (`decision.intra_test_dc`),
        // which the cheap secondary ranking omits (bodycam_bc1 J_real +16%).
        // A co-located same-instant secondary reference (`ratio == 0`) can
        // still predict disoccluded or fast-moving blocks that forced the
        // temporal primary reference into intra, counting DC error when
        // `!primary.mode.is_inter()`.
        // A cached planner choice (cost `-inf`) was accepted as good
        // without ranking: there is no cost for the secondary to beat.
        let ranked = primary.cost > 0.0;
        if (!primary.mode.is_inter() && ratio > 0.0) || !ranked {
            return (primary, ReferenceChoice::Primary);
        }
        let seed = within_reach(
            scaled_vector(search_mv, ratio),
            (self.layer.coded_width, self.layer.coded_height),
            B::VECTOR_REACH,
        );
        let include_dc = !primary.mode.is_inter();
        let Some(second) = self.secondary_choice(
            node,
            source,
            secondary,
            (seed, primary.cost, include_dc),
            analysis,
            refresh,
        ) else {
            self.backend
                .select_reference(node, ReferenceChoice::Primary);
            return (primary, ReferenceChoice::Primary);
        };
        let (best, chosen) = if second.cost < primary.cost {
            (second, ReferenceChoice::Secondary)
        } else {
            self.backend
                .select_reference(node, ReferenceChoice::Primary);
            (primary, ReferenceChoice::Primary)
        };
        if self.backend.supports_compound_reference()
            && let (PredictionMode::Inter(primary_mv), PredictionMode::Inter(secondary_mv)) =
                (primary.mode.prediction(), second.mode.prediction())
        {
            let compound_margin = self
                .layer
                .second
                .as_ref()
                .map_or(0.0, |s| s.compound_margin);
            let worst_single = primary.cost.max(second.cost);
            let best_single = primary.cost.min(second.cost);
            if worst_single.is_finite()
                && best_single.is_finite()
                && worst_single <= best_single * (1.0 + compound_margin)
                && let Some(compound) = choose_compound_mode(
                    &self.dsp,
                    &mut self.backend,
                    node,
                    source,
                    primary_reference,
                    secondary,
                    (primary_mv, secondary_mv, include_dc),
                    &mut self.decision_scratch,
                    (&self.frame_decision_params, analysis.0),
                    refresh,
                )
                && compound.cost < best.cost
            {
                return (compound, ReferenceChoice::Compound { secondary_mv });
            }
            self.backend.select_reference(node, chosen);
        }
        (best, chosen)
    }

    /// The reference choice of `node` from its `primary` choice: the
    /// secondary reference or compound ([`Self::with_secondary`]), then the
    /// primary pair ([`Self::with_primary_pair`]).
    fn choose_reference(
        &mut self,
        node: &Block,
        (source, reference): (&Picture, &Picture),
        secondary: Option<(&Picture, f32)>,
        (search_mv, primary): (Mv, Choice),
        analysis: (Option<&PreAnalysis>, bool),
        refresh: Option<&RefreshWindow>,
    ) -> (Choice, ReferenceChoice) {
        let best = self.with_secondary(
            node,
            source,
            reference,
            secondary,
            (search_mv, primary),
            analysis,
            refresh,
        );
        self.with_primary_pair(
            node,
            source,
            (reference, secondary.map(|(picture, _)| picture)),
            best,
            analysis,
            refresh,
        )
    }

    /// `best`, or low-delay bi-prediction from the primary reference at
    /// `best`'s vector and a searched second one where that costs less
    /// (`decision.bi_pair`, [`ReferenceChoice::PrimaryPair`]).
    fn with_primary_pair(
        &mut self,
        node: &Block,
        source: &Picture,
        (reference, secondary): (&Picture, Option<&Picture>),
        best: (Choice, ReferenceChoice),
        analysis: (Option<&PreAnalysis>, bool),
        refresh: Option<&RefreshWindow>,
    ) -> (Choice, ReferenceChoice) {
        let (choice, chosen) = best;
        // What it skips: the pair trials on static blocks, on blocks whose
        // single-reference winner is intra, the secondary reference or an
        // unranked cached planner choice. Why safe: a static block's zero
        // vector already predicts it, and the pair extends the primary
        // winner only. Effect: not yet measured (off by default).
        let (static_block, tried) = (
            analysis.1,
            self.frame_decision_params.bi_pair >= 0.5 && self.backend.supports_primary_pair(),
        );
        let PredictionMode::Inter(first_mv) = choice.mode.prediction() else {
            return best;
        };
        if !tried || static_block || chosen != ReferenceChoice::Primary || choice.cost <= 0.0 {
            return best;
        }
        let searched = choose_primary_pair(
            &self.dsp,
            &mut self.backend,
            node,
            source,
            reference,
            (first_mv, false),
            &mut self.decision_scratch,
            (&self.frame_decision_params, analysis.0),
            refresh,
        )
        .map(|(pair, second_mv, blurred)| {
            (pair, ReferenceChoice::PrimaryPair { second_mv }, blurred)
        });
        let secondary = secondary.filter(|_| self.backend.supports_compound_reference());
        let merged = choose_pair_merge(
            &self.dsp,
            &mut self.backend,
            node,
            source,
            (reference, secondary),
            first_mv,
            &mut self.decision_scratch,
            (&self.frame_decision_params, analysis.0),
            refresh,
        );
        let Some((pair, second, blurred)) = [searched, merged]
            .into_iter()
            .flatten()
            .filter(|(pair, ..)| pair.cost < choice.cost)
            .min_by(|a, b| a.0.cost.total_cmp(&b.0.cost))
        else {
            return best;
        };
        if let PredictionMode::Inter(pair_mv) = pair.mode.prediction()
            && self.frame_decision_params.bi_pair_rd >= 0.5
            && !self.pair_pays_when_coded(
                node,
                source,
                (reference, secondary),
                (first_mv, pair_mv, second, blurred),
            )
        {
            return best;
        }
        (pair, second)
    }

    /// `decision.bi_pair_rd`: whether the two-hypothesis winner (first
    /// vector `pair_mv`, second half `second`) costs less than the
    /// single-reference choice at `single_mv` when both are coded (luma
    /// transform, quantisation and bits; the pair also pays its ranking's
    /// blur term `blurred`); `true` where the backend cannot price either
    /// (the ranking stands).
    fn pair_pays_when_coded(
        &mut self,
        node: &Block,
        source: &Picture,
        (reference, secondary): (&Picture, Option<&Picture>),
        (single_mv, pair_mv, second, blurred): (Mv, Mv, ReferenceChoice, f32),
    ) -> bool {
        let second_reference = match second {
            ReferenceChoice::Compound { .. } => secondary.unwrap_or(reference),
            _ => reference,
        };
        let paired = self.backend.pair_coded_cost(
            source,
            [reference, second_reference],
            node,
            (pair_mv, second),
        );
        let single = self.backend.final_coded_cost(
            source,
            reference,
            node,
            FinalCandidate::Inter(single_mv),
        );
        match (paired, single) {
            (Some(paired), Some(single)) => paired + blurred < single,
            _ => true,
        }
    }

    /// The best inter mode of `node` predicted from the `secondary`
    /// reference, from the codec's predicted vectors and `seed` (the
    /// primary search vector scaled to the secondary's distance), to beat
    /// `primary_cost`: the cheap
    /// ranking (`decision::choose_secondary_mode`), or with
    /// `refs.second_full` the full decision. `None` when intra beats it.
    /// Leaves the backend's reference selected as
    /// [`ReferenceChoice::Secondary`].
    ///
    /// The secondary reference gets no motion search of its own: its best
    /// vectors are near zero where it pays most (the HQ frame's still and
    /// slowly moving detail), and the scaled primary vector covers steady
    /// motion.
    fn secondary_choice(
        &mut self,
        node: &Block,
        source: &Picture,
        secondary: &Picture,
        (seed, primary_cost, include_dc): (Mv, f32, bool),
        analysis: (Option<&PreAnalysis>, bool),
        refresh: Option<&RefreshWindow>,
    ) -> Option<Choice> {
        self.backend
            .select_reference(node, ReferenceChoice::Secondary);
        if !self.layer.second.as_ref().is_some_and(|s| s.full_decision) {
            let refine_margin = self.layer.second.as_ref().map_or(0.0, |s| s.refine_margin);
            return choose_secondary_mode(
                &self.dsp,
                &mut self.backend,
                node,
                source,
                secondary,
                seed,
                (primary_cost, refine_margin, include_dc),
                &mut self.decision_scratch,
                (&self.frame_decision_params, analysis.0),
                refresh,
            );
        }
        let choice = choose_inter_mode(
            &self.dsp,
            &mut self.backend,
            node,
            (source, secondary, None),
            seed,
            None,
            &mut self.decision_scratch,
            analysis,
            &self.frame_decision_params,
            refresh,
        );
        choice.mode.is_inter().then_some(choice)
    }

    /// Depth-first partition walk of one node, following `plan` where there
    /// is one (inter frames) and the fixed configured size otherwise. `inter`
    /// is `Some` on inter frames.
    fn code_node(
        &mut self,
        node: Block,
        plan: Option<&Partition>,
        pictures: FramePictures<'_>,
        inter: Option<&FrameInputs<'_>>,
    ) {
        let FramePictures {
            source,
            reference,
            secondary,
            recon,
        } = pictures;
        let (width, height) = (self.layer.coded_width, self.layer.coded_height);
        if node.x >= width || node.y >= height {
            return; // entirely outside the frame: no syntax in any codec
        }
        let children = self.children(&node, plan, inter.is_some());
        match children {
            NodeSplit::Split(child_plans) => {
                self.backend.write_partition(&node, true);
                for (i, child) in node.quadrants().into_iter().enumerate() {
                    let pictures = FramePictures {
                        source,
                        reference,
                        secondary,
                        recon: &mut *recon,
                    };
                    self.code_node(child, child_plans.map(|c| &c[i]), pictures, inter);
                }
            }
            NodeSplit::Shaped(shape, part_plans) => {
                self.backend.write_shape(&node, shape);
                let parts = shape.parts(&node);
                for (i, part) in parts.into_iter().take(shape.part_count()).enumerate() {
                    let pictures = FramePictures {
                        source,
                        reference,
                        secondary,
                        recon: &mut *recon,
                    };
                    let planned = part_plans.and_then(|p| p[i].as_ref());
                    self.code_leaf(part, planned, pictures, inter);
                }
            }
            NodeSplit::Leaf => {
                self.backend.write_partition(&node, false);
                let planned = match plan {
                    Some(Partition::Leaf(stats)) => stats.as_ref(),
                    _ => None,
                };
                let pictures = FramePictures {
                    source,
                    reference,
                    secondary,
                    recon,
                };
                self.code_leaf(node, planned, pictures, inter);
            }
        }
    }

    /// Decide and code leaf `node`; `planned` are the residual statistics
    /// the planner measured for it, if any.
    fn code_leaf(
        &mut self,
        node: Block,
        planned: Option<&LeafStats>,
        pictures: FramePictures<'_>,
        inter: Option<&FrameInputs<'_>>,
    ) {
        let FramePictures {
            source,
            reference,
            secondary,
            recon,
        } = pictures;
        let region = self
            .window
            .as_ref()
            .map_or(RefreshRegion::Dirty, |w| w.region(&node));
        let qp_offset = self.leaf_qp_offset(&node);
        // The leaf decision's search positions count from here (`labels.rs`).
        let positions = self.decision_scratch.search_positions();
        let (mode, chosen, intra_fallback, selected_ref, pricing) = match inter {
            None => (
                PredictionMode::IntraDc,
                ReferenceChoice::Primary,
                None,
                None,
                PricingHint::NONE,
            ),
            Some(_) if region == RefreshRegion::Intra => (
                PredictionMode::IntraDc,
                ReferenceChoice::Primary,
                None,
                Some(reference),
                PricingHint::NONE,
            ),
            Some(frame) => {
                let pictures = FramePictures {
                    source,
                    reference,
                    secondary,
                    recon: &mut *recon,
                };
                let clean = region == RefreshRegion::Clean;
                // Rank modes with the λ of the quantizer the block is coded
                // at: λ scales with the squared step (one frame λ would rank
                // a block coded 8 QP finer as if residual bits cost 6x more).
                // Only the λ scale changes, so only it is saved and restored
                // (copying the whole parameter set twice per leaf showed up
                // as memcpy calls).
                let frame_lambda_scale = self.frame_decision_params.lambda_scale;
                if qp_offset != 0 {
                    self.frame_decision_params.lambda_scale *= self.block_lambda_ratio(qp_offset);
                }
                let decided =
                    self.decide_inter_leaf(&node, planned, frame, pictures, clean, qp_offset < 0);
                if self.spatial.is_some() {
                    let choice = decided.map(|(mode, chosen, _, _)| (mode, chosen));
                    self.observe_spatial_leaf_choice(leaf_area::<B>(&node), choice);
                }
                self.frame_decision_params.lambda_scale = frame_lambda_scale;
                if secondary.is_some()
                    && let Some(second) = self.layer.second.as_mut().filter(|s| s.watches_usage())
                {
                    let from_secondary = decided.as_ref().is_some_and(|(mode, chosen, ..)| {
                        matches!(mode, PredictionMode::Inter(_)) && chosen.uses_secondary()
                    });
                    second.observe_leaf(leaf_area::<B>(&node), from_secondary);
                }
                let Some((mode, chosen, intra_fallback, pricing)) = decided else {
                    // A static skip: coded at the frame quantizer.
                    return self.record_leaf_label(&node, None, (0, positions));
                };
                let picture = match chosen {
                    ReferenceChoice::Primary
                    | ReferenceChoice::Compound { .. }
                    | ReferenceChoice::PrimaryPair { .. } => reference,
                    ReferenceChoice::Secondary => {
                        secondary.expect("only frames with a secondary reference choose it")
                    }
                };
                (mode, chosen, intra_fallback, Some(picture), pricing)
            }
        };
        // Rate control's realised offset needs only the leaves coded off the
        // frame quantizer (the others' ratio is 1; the frame's total weight
        // covers them): on a frame where the reuse term refines a few cells,
        // a few leaves.
        if qp_offset != 0 {
            let frame_step = self.quantizer_steps[self.frame_quantizer as usize];
            let coded = (i64::from(self.frame_quantizer) + i64::from(qp_offset)) as usize;
            let offset = QP_UNITS_PER_OCTAVE * (self.quantizer_steps[coded] / frame_step).log2();
            if let Some(aq) = self.active_aq() {
                self.block_offsets.push((aq.block_weight(&node), offset));
            } else if self.in_frame.is_some() {
                let area = (1usize << (node.log2_width + node.log2_height)) as f64;
                self.in_frame_offsets.push((area, offset));
            }
        }
        self.reuse_planned_luma_stats(&node, planned, mode, chosen);
        let decision = BlockDecision {
            mode,
            reference: chosen,
            qp_offset,
            intra_fallback,
            intra: IntraChoice::default(),
        };
        let decision = self.encode_leaf(
            &node,
            decision,
            (pricing, region == RefreshRegion::Dirty),
            (source, selected_ref, reference, secondary),
            recon,
        );
        self.record_leaf_label(&node, Some(&decision), (qp_offset, positions));
        self.note_reused_leaf(&node, decision.reference, qp_offset, inter.is_some());
    }

    /// Hand the final inter coder the planner's measurement for the chosen vector.
    #[inline]
    fn reuse_planned_luma_stats(
        &mut self,
        node: &Block,
        planned: Option<&LeafStats>,
        mode: PredictionMode,
        chosen: ReferenceChoice,
    ) {
        if B::REUSE_PLANNED_LUMA_STATS
            && chosen == ReferenceChoice::Primary
            && node.is_square()
            && let (Some(stats), PredictionMode::Inter(mv)) = (planned, mode)
            && mv.is_full_pel()
        {
            let residual = if mv == stats.search_mv {
                Some(stats.search)
            } else {
                (mv == Mv::ZERO).then_some(stats.zero)
            };
            if let Some(residual) = residual {
                self.backend.reuse_planned_luma_stats(node, mv, residual);
            }
        }
    }

    /// Tell the reuse term (inter frames it runs on) that leaf `node` was
    /// coded from `chosen` at block quantizer offset `qp_offset`.
    fn note_reused_leaf(
        &mut self,
        node: &Block,
        chosen: ReferenceChoice,
        qp_offset: i16,
        inter: bool,
    ) {
        let reuse_frame = self.reuse_frame.is_some();
        let copy = chosen == ReferenceChoice::Primary && self.backend.copied_reference(node);
        let step = self
            .sample_step((i64::from(self.frame_quantizer) + i64::from(qp_offset)).max(0) as u32);
        if let Some(term) = self.layer.reuse.as_mut().filter(|_| inter && reuse_frame) {
            term.coded_block((node.x, node.y), leaf_extent::<B>(node), (copy, step));
        }
    }

    /// Code leaf `node` as `decision` (compound, whole-CU priced where
    /// `pricing` allows and the leaf is outside intra refresh's constrained
    /// region, by the key-frame leaf search of `leaf_rd.rs`, or plain);
    /// returns what was coded. The pictures are the source, the decision's
    /// reference (`None` only for a key frame's leaf), the primary and the
    /// secondary.
    fn encode_leaf(
        &mut self,
        node: &Block,
        decision: BlockDecision,
        (pricing, unconstrained): (PricingHint, bool),
        (source, selected_ref, reference, secondary): (
            &Picture,
            Option<&Picture>,
            &Picture,
            Option<&Picture>,
        ),
        recon: &mut Picture,
    ) -> BlockDecision {
        if let ReferenceChoice::Compound { .. } = decision.reference {
            self.backend.encode_compound_block(
                node,
                &decision,
                source,
                reference,
                secondary.expect("only frames with a secondary reference choose compound"),
                recon,
            );
            decision
        } else if let ReferenceChoice::PrimaryPair { .. } = decision.reference {
            self.backend
                .encode_compound_block(node, &decision, source, reference, reference, recon);
            decision
        } else if selected_ref.is_none() && self.leaf_rd_applies() {
            self.code_key_leaf_by_rd(node, decision, source, recon)
        } else if unconstrained && pricing != PricingHint::NONE {
            cu_pricing::code_leaf(
                &mut self.backend,
                node,
                decision,
                pricing,
                &self.frame_decision_params,
                (source, selected_ref, reference),
                recon,
            )
        } else {
            self.backend
                .encode_block(node, &decision, source, selected_ref, recon);
            decision
        }
    }

    fn leaf_qp_offset(&mut self, node: &Block) -> i16 {
        let random_qp = self
            .random
            .as_mut()
            .filter(|_| B::SUPPORTS_BLOCK_QP)
            .map(|random| {
                let codec = self.config.codec;
                random.qp_offset(
                    self.frame_quantizer,
                    (min_quantizer(codec), max_quantizer(codec)),
                )
            });
        let qp_offset = random_qp.unwrap_or_else(|| {
            let aq = self.active_aq().map_or(0., |aq| aq.offset_qp(node));
            let steered = self.in_frame.as_ref().map_or(0., InFrameRate::offset_qp);
            let codec = self.config.codec;
            native_offset(
                &self.quantizer_steps,
                (min_quantizer(codec), max_quantizer(codec)),
                self.frame_quantizer,
                aq + steered,
            )
        });
        if qp_offset == 0 {
            0
        } else {
            self.backend.block_qp_offset(qp_offset)
        }
    }
}

/// The label dump's superblock hooks (`labels.rs`): each does nothing
/// while the dump is off.
impl<B: CodecBackend> Encoder<B> {
    /// Start the labels of a coding at `plan`.
    fn begin_label_coding(&mut self, plan: &FramePlan, superblock_log2: u8) {
        if self.labels.is_none() || self.spatial.is_some() {
            return;
        }
        let context = FrameContext {
            index: plan.index,
            key: plan.frame_type == FrameType::Key,
            temporal_layer: plan.references.position.layer,
            quantizer: plan.quantizer,
            lambda: self.backend.lambda(),
            noise_sigma: self.labels.as_ref().map_or(0.0, |l| l.noise_sigma),
            width: self.layer.coded_width as u32,
            height: self.layer.coded_height as u32,
            superblock_log2,
            min_block_log2: self.backend.min_block_log2(),
            key_distance: u32::try_from(plan.index.saturating_sub(self.layer.last_key_index))
                .unwrap_or(u32::MAX),
            rate: self
                .layer
                .rate_control
                .as_ref()
                .map_or([f32::NAN; 4], |rc| {
                    let stats = rc.stats();
                    [
                        stats.buffer_fullness as f32,
                        stats.plan.budget_bits as f32,
                        stats.plan.level_quantizer as f32,
                        stats.plan.cadence_offset as f32,
                    ]
                }),
        };
        if let Some(labeller) = self.labels.as_mut() {
            labeller.begin_coding(context);
        }
    }

    /// Code superblock `sb` (`code_superblock`); while the dump is on,
    /// recording its inputs first, its tool trials, and its coded J and time.
    fn code_labelled_superblock(
        &mut self,
        sb: Block,
        plan: Option<&Partition>,
        pictures: FramePictures<'_>,
        inter: Option<&FrameInputs<'_>>,
        (planner_params, light): (&PlannerParams, bool),
    ) {
        let FramePictures {
            source,
            reference,
            secondary,
            recon,
        } = pictures;
        let priced = self.open_superblock_label(&sb, (source, reference, secondary), inter);
        if priced == Some(true) {
            let pictures = FramePictures {
                source,
                reference,
                secondary,
                recon: &mut *recon,
            };
            self.label_tools(sb, plan, pictures, inter, (planner_params, light));
        }
        let started = priced.map(|priced| self.start_superblock_label(&sb, priced, recon));
        let pictures = FramePictures {
            source,
            reference,
            secondary,
            recon: &mut *recon,
        };
        self.code_superblock(sb, plan, pictures, inter, planner_params);
        if let (Some(started), Some(priced)) = (started, priced) {
            self.close_superblock_label((started, priced), source, recon);
        }
    }

    /// Record superblock `sb`'s inputs before it is coded; returns whether
    /// it is priced (inside the picture, the backend has node trials), or
    /// `None` while the dump is off.
    fn open_superblock_label(
        &mut self,
        sb: &Block,
        (source, reference, secondary): (&Picture, &Picture, Option<&Picture>),
        inter: Option<&FrameInputs<'_>>,
    ) -> Option<bool> {
        if self.labels.is_none() || self.spatial.is_some() {
            return None;
        }
        let inside = sb.is_inside(self.layer.coded_width, self.layer.coded_height);
        let inputs = FeatureInputs {
            source,
            reference: inter.map(|frame| (reference, search_vector::<B>(frame.motion, sb))),
            secondary: inter.and_then(|frame| {
                let picture = secondary?;
                let ratio = frame.secondary_ratio?;
                Some((
                    picture,
                    scaled_vector(search_vector::<B>(frame.motion, sb), ratio),
                ))
            }),
            pre: inter.and_then(|frame| frame.pre),
            still: inter.and_then(|frame| frame.still),
            motion: inter.and_then(|frame| frame.motion),
        };
        let labeller = self.labels.as_mut()?;
        let tools = labeller.config.tools.len();
        let record = crate::labels::superblock_record(&inputs, sb, inside, tools);
        labeller.current.superblocks.push(record);
        Some(inside && self.rd_split_applies())
    }

    /// Open the pricing of the superblock's real coding (where `priced`: a
    /// superblock is priced as one node) and return the clock.
    fn start_superblock_label(&mut self, sb: &Block, priced: bool, recon: &Picture) -> u64 {
        if priced {
            self.backend.begin_node_pricing(sb, recon);
        }
        self.labels.as_ref().map_or(0, |l| l.now())
    }

    /// Before superblock `sb` is coded, code it once as the preset does and
    /// once per labelled tool, each a trial that is undone (`labels.rs`),
    /// noting each coding's J and time.
    fn label_tools(
        &mut self,
        sb: Block,
        plan: Option<&Partition>,
        pictures: FramePictures<'_>,
        inter: Option<&FrameInputs<'_>>,
        (planner_params, light): (&PlannerParams, bool),
    ) {
        let Some(labeller) = self.labels.as_mut() else {
            return;
        };
        let tools: Vec<(ParamOverrides, ParamOverrides)> = std::iter::once(Default::default())
            .chain(labeller.config.tools.iter().map(LabelTool::split))
            .collect();
        labeller.in_tool_trial = true;
        let FramePictures {
            source,
            reference,
            secondary,
            recon,
        } = pictures;
        for (slot, (engine, backend)) in tools.iter().enumerate() {
            self.backend.begin_node_trial(&sb, recon);
            let mark = self.engine_mark();
            let depths = self.depths.clone();
            let decision = self.frame_decision_params;
            self.frame_decision_params = decision.with_overrides(engine);
            let planner = planner_params.with_overrides(engine);
            if !backend.is_empty() {
                self.switch_label_backend(Some(backend), light);
            }
            let started = self.label_mark();
            let pictures = FramePictures {
                source,
                reference,
                secondary,
                recon: &mut *recon,
            };
            self.code_superblock(sb, plan, pictures, inter, &planner);
            let cost = self.backend.node_trial_cost(source, recon);
            let (ns, work, positions) = started.until(self.label_mark());
            if !backend.is_empty() {
                self.switch_label_backend(None, light);
            }
            self.frame_decision_params = decision;
            self.backend.undo_node_trial(recon);
            self.undo_engine(&mark);
            self.depths = depths;
            self.backend.end_node_trial();
            if let Some(record) = self
                .labels
                .as_mut()
                .and_then(|l| l.current.superblocks.last_mut())
            {
                record.tool_cost[slot] = cost;
                record.tool_ns[slot] = ns;
                record.tool_work[slot] = work;
                record.tool_positions[slot] = positions;
            }
        }
        if let Some(labeller) = self.labels.as_mut() {
            labeller.in_tool_trial = false;
        }
    }

    /// Code the next blocks with the backend's settings of this superblock
    /// (`light` or regular) under `tool`'s overrides, or (`None`) put the
    /// frame's settings back. The backend's light slot carries the tool's
    /// settings meanwhile ([`CodecBackend::set_light_params`]).
    fn switch_label_backend(&mut self, tool: Option<&ParamOverrides>, light: bool) {
        let Some(labeller) = self.labels.as_ref() else {
            return;
        };
        let frame_light = labeller.backend_light.clone();
        let base = if light {
            frame_light
                .clone()
                .unwrap_or_else(|| labeller.backend_regular.clone())
        } else {
            labeller.backend_regular.clone()
        };
        self.backend.use_light_params(false);
        if let Some(tool) = tool {
            self.backend
                .set_light_params(Some(&base.overlaid_with(tool)));
            self.backend.use_light_params(true);
        } else {
            self.backend.set_light_params(frame_light.as_ref());
            self.backend.use_light_params(light);
        }
    }

    /// Keep the backend overrides the frame's superblocks are coded with,
    /// for the tool trials ([`Encoder::switch_label_backend`]).
    fn note_label_overrides(&mut self, role: FrameRole, key: bool) {
        let Some(labeller) = self.labels.as_mut() else {
            return;
        };
        labeller.backend_regular = match &self.effort_overlay {
            Some(overlay) => overlay
                .backend_overrides(&self.config.params, role, key)
                .clone(),
            None => self.config.params.clone(),
        };
        labeller.backend_light = self
            .frame_light
            .as_ref()
            .and(self.light_overlay.as_ref())
            .map(|overlay| overlay.backend_overrides(role).clone());
    }

    /// Record leaf `node` of the coding in progress (`None`: a static skip),
    /// whose decision began at search position count `positions`; undone
    /// with the engine's trial state (`rd_split.rs` `EngineMark`).
    fn record_leaf_label(
        &mut self,
        node: &Block,
        decision: Option<&BlockDecision>,
        (qp_offset, positions): (i16, u64),
    ) {
        let skip = decision.is_some() && self.backend.last_cu_skipped();
        let Some(labeller) = self.labels.as_mut() else {
            return;
        };
        let measured = self.decision_scratch.search_positions() - positions;
        let (mode, mv) = match decision.map(|d| (d.mode, d.reference)) {
            None => (crate::labels::MODE_STATIC_SKIP, [0, 0]),
            Some((PredictionMode::IntraDc, _)) => (crate::labels::MODE_INTRA, [0, 0]),
            Some((PredictionMode::Inter(mv), reference)) => (
                match reference {
                    ReferenceChoice::Primary => crate::labels::MODE_INTER,
                    ReferenceChoice::Secondary => crate::labels::MODE_SECONDARY,
                    ReferenceChoice::Compound { .. } => crate::labels::MODE_COMPOUND,
                    ReferenceChoice::PrimaryPair { .. } => crate::labels::MODE_PRIMARY_PAIR,
                },
                [mv.x, mv.y],
            ),
        };
        // Each leaf decided and coded is a unit of the work counter.
        labeller.add_work(node.width() * node.height());
        let superblock = labeller.current.superblocks.len().saturating_sub(1) as u32;
        labeller.current.leaves.push(crate::labels::LeafLabel {
            superblock,
            x: node.x as u32,
            y: node.y as u32,
            log2: [node.log2_width, node.log2_height],
            mode,
            skip,
            mv,
            qp_offset,
            positions: u16::try_from(measured).unwrap_or(u16::MAX),
        });
    }

    /// Note the coded cost and time of the superblock last opened.
    fn close_superblock_label(
        &mut self,
        (started, priced): (u64, bool),
        source: &Picture,
        recon: &Picture,
    ) {
        let cost = if priced {
            let cost = self.backend.node_trial_cost(source, recon);
            self.backend.end_node_trial();
            cost
        } else {
            f32::NAN
        };
        if let Some(labeller) = self.labels.as_mut() {
            let now = labeller.now();
            if let Some(record) = labeller.current.superblocks.last_mut() {
                record.ns = now.saturating_sub(started);
                record.cost = cost;
            }
            let superblock = labeller.current.superblocks.len().saturating_sub(1) as u32;
            labeller.record_final_leaves(superblock);
        }
    }
}

impl<B: CodecBackend> Encoder<B> {
    /// Offer `node` to the backend's static skip when pre-analysis marks it
    /// static, when the noise skip (`noise_skip.rs`, while on) finds its
    /// zero-vector residual within the camera's noise.
    fn try_static_skip(
        &mut self,
        node: &Block,
        frame: &FrameInputs<'_>,
        (source, reference): (&Picture, &Picture),
        recon: &mut Picture,
        static_block: bool,
    ) -> bool {
        let params = &self.frame_decision_params;
        let within_noise = params.noise_skip > 0.
            && self.layer.noise_map.explains(
                &self.dsp,
                node,
                (source, reference),
                params.noise_skip,
                params.noise_skip_chroma_ratio,
                self.sample_step(self.frame_quantizer) as f32,
            );
        if !within_noise && !static_block {
            return false;
        }
        let still = frame.still.is_some_and(|still| is_still::<B>(still, node));
        let accepted = self.backend.try_static_skip(
            node,
            source,
            reference,
            recon,
            StaticSkipTest {
                still,
                within_noise,
                repeat: false,
            },
        );
        if accepted && within_noise {
            self.layer.noise_skipped_area += node.width() * node.height();
        }
        accepted
    }

    /// Code the static skip of `node` at vector `skip`, and report it to the
    /// reuse term (a copy of the reference at the zero vector).
    fn code_static_skip(
        &mut self,
        node: &Block,
        skip: Mv,
        (source, reference): (&Picture, &Picture),
        recon: &mut Picture,
    ) {
        self.backend.encode_block(
            node,
            &BlockDecision::new(PredictionMode::Inter(skip)),
            source,
            Some(reference),
            recon,
        );
        let reuse_frame = self.reuse_frame.is_some();
        let step = self.sample_step(self.frame_quantizer);
        if let Some(term) = self.layer.reuse.as_mut().filter(|_| reuse_frame) {
            term.coded_block(
                (node.x, node.y),
                leaf_extent::<B>(node),
                (skip == Mv::ZERO, step),
            );
        }
    }

    /// λ of a block coded `qp_offset` native quantizer steps from the
    /// frame's, relative to the frame λ: λ is proportional to the squared
    /// quantizer step in every codec.
    fn block_lambda_ratio(&self, qp_offset: i16) -> f32 {
        let frame = self.quantizer_steps[self.frame_quantizer as usize];
        let coded = (i64::from(self.frame_quantizer) + i64::from(qp_offset)) as usize;
        let ratio = self.quantizer_steps[coded] / frame;
        (ratio * ratio) as f32
    }
}

/// Turn on the backend options the engine's settings need: the syntax
/// stress test's random choices, and block-QP syntax when blocks may carry
/// quantizer offsets (block AQ or random offsets).
fn enable_backend_options<B: CodecBackend>(
    backend: &mut B,
    random: Option<&RandomDecisions>,
    aq: bool,
) {
    if let Some(random) = random {
        backend.enable_random_decisions(DebugRng::new(random.backend_seed()));
    }
    if B::SUPPORTS_BLOCK_QP && (aq || random.is_some()) {
        backend.enable_block_qp();
    }
}

/// A random legal mode for inter-frame leaf `node`
/// (`debug.random_decisions`). The backend sees the same calls as in a normal
/// decision (`mv_candidates`, `sub_block_motion`) so its per-block state is
/// prepared exactly as for a real encode.
fn random_inter_leaf<B: CodecBackend>(
    backend: &mut B,
    random: &mut RandomDecisions,
    node: &Block,
    candidates: &mut Vec<Mv>,
    reach: (i32, i32),
    refresh: Option<&RefreshWindow>,
) -> PredictionMode {
    candidates.clear();
    candidates.push(Mv::ZERO);
    backend.mv_candidates(node, candidates);
    // Legal: allowed by the refresh window, and a vector the codec can
    // signal for this block (infinite bits otherwise, as mode decision reads
    // them: a vector range, a difference with no code).
    let mode = match random.inter_mode(candidates, reach, |mv| {
        refresh.is_none_or(|w| w.allows(node, mv))
            && backend
                .mode_bits(node, &PredictionMode::Inter(mv))
                .is_finite()
    }) {
        // Rectangular leaves are inter only (`CodecBackend::partition_shapes`);
        // shapes are never planned under refresh, so zero is allowed.
        PredictionMode::IntraDc if !is_square_leaf::<B>(node) => PredictionMode::Inter(Mv::ZERO),
        mode => mode,
    };
    if is_square_leaf::<B>(node) && node.log2_size() > SUB_BLOCK_LOG2 {
        let mut vectors = [Mv::ZERO; MAX_SUB_BLOCKS];
        let count = (node.size() >> SUB_BLOCK_LOG2).pow(2);
        random.sub_block_vectors(&mut vectors[..count]);
        if let Some(window) = refresh {
            forbid_sub_block_vectors(window, node, &mut vectors[..count]);
        }
        backend.sub_block_motion(node, &vectors[..count]);
    }
    mode
}

/// log2 of the sub-blocks whose search vectors
/// [`CodecBackend::sub_block_motion`] reports (the analysis's 8x8 field).
const SUB_BLOCK_LOG2: u8 = 3;
/// Sub-blocks of the largest (64x64) block.
const MAX_SUB_BLOCKS: usize = 64;

/// The fine (8x8) search vectors of the sub-blocks of `block`, raster order,
/// written to the front of `out`; returns how many. Zero where no search
/// result exists.
// Out of line: inlined into `code_leaf` (once its call became conditional)
// it moved other inlining there, +0.5% H.264 Default instructions.
#[inline(never)]
fn sub_block_vectors(motion: Option<&MotionAnalysis>, block: &Block, out: &mut [Mv]) -> usize {
    let per_side = block.size() >> SUB_BLOCK_LOG2;
    let (col0, row0) = (block.x >> SUB_BLOCK_LOG2, block.y >> SUB_BLOCK_LOG2);
    let count = per_side * per_side;
    // Whole rows of the field at a time: a per-cell `try_get` cost ~47
    // instructions per sub-block (~0.6M per static 1080x720 P frame). Each
    // vector is written once (cells past the field are zero), with no
    // separate zero fill: for a macroblock's four vectors that fill was a
    // `memset` call.
    for (r, out_row) in out[..count].chunks_exact_mut(per_side).enumerate() {
        let cells = motion.map_or(&[][..], |m| m.fine.row_run(col0, row0 + r, per_side));
        for (i, v) in out_row.iter_mut().enumerate() {
            *v = cells.get(i).map_or(Mv::ZERO, |b| {
                Mv::from_full_pel(i32::from(b.mv.x), i32::from(b.mv.y))
            });
        }
    }
    count
}

/// A partition-walk node's fate.
enum NodeSplit<'p> {
    /// Coded as one leaf block.
    Leaf,
    /// Split into four quadrants, following these plans if any.
    Split(Option<&'p [Partition; 4]>),
    /// Coded as the rectangular leaves of a shape, with their planned
    /// statistics if any.
    Shaped(
        PartitionShape,
        Option<&'p [Option<LeafStats>; MAX_SHAPE_PARTS]>,
    ),
}

/// Replace the sub-block vectors of clean `block` that intra refresh
/// forbids with zero (always allowed for a clean block).
fn forbid_sub_block_vectors(window: &RefreshWindow, block: &Block, vectors: &mut [Mv]) {
    let side = 1usize << SUB_BLOCK_LOG2;
    let per_row = block.size() >> SUB_BLOCK_LOG2;
    for (i, mv) in vectors.iter_mut().enumerate() {
        if !window.allows_area(block.x + (i % per_row) * side, side, *mv) {
            *mv = Mv::ZERO;
        }
    }
}

/// The pictures one frame's partition walk reads and writes.
struct FramePictures<'a> {
    /// The coded source (padded to the coded size if needed).
    source: &'a Picture,
    /// The reconstruction of this frame's reference slot (unused on key
    /// frames).
    reference: &'a Picture,
    /// The reconstruction of the secondary reference slot, on inter frames
    /// that have one ([`FrameReferences::secondary`]).
    secondary: Option<&'a Picture>,
    /// The reconstruction being written.
    recon: &'a mut Picture,
}

/// Refuse temporal layers a backend cannot code.
fn check_temporal_layers<B: CodecBackend>(
    config: &EncoderConfig,
    backend: &B,
) -> Result<(), ConfigError> {
    if config.temporal_layers > 1
        && !(backend.supports_inter() && backend.supports_temporal_layers())
    {
        return Err(ConfigError::TemporalLayers(config.temporal_layers));
    }
    Ok(())
}

/// The configuration of the pictures the engine codes (the stream's, at
/// [`CodecBackend::frame_size`]) and their coded size, both checked, as is
/// the block size.
fn picture_geometry<B: CodecBackend>(
    config: &EncoderConfig,
    backend: &B,
) -> Result<(EncoderConfig, (usize, usize)), ConfigError> {
    if config.block_log2 < backend.min_block_log2() || config.block_log2 > backend.superblock_log2()
    {
        return Err(ConfigError::BlockSize(config.block_log2));
    }
    let (width, height) = backend.frame_size(config.width, config.height);
    let picture = EncoderConfig {
        width,
        height,
        ..config.clone()
    };
    let coded = backend.coded_size(width, height);
    check_dimensions(config, (config.width, config.height), CHROMA_SUBSAMPLING)?;
    check_dimensions(&picture, coded, 1usize << backend.min_block_log2())?;
    Ok((picture, coded))
}

/// Check the visible size (even for 4:2:0) and the backend's coded size
/// (whole minimum blocks; backends pad to it, the engine replicates edges).
fn check_dimensions(
    config: &EncoderConfig,
    (coded_width, coded_height): (usize, usize),
    multiple: usize,
) -> Result<(), ConfigError> {
    let fits = |d: usize, m: usize| d > 0 && d.is_multiple_of(m);
    let error = |multiple| ConfigError::Dimensions {
        width: config.width,
        height: config.height,
        multiple,
    };
    if !fits(config.width, CHROMA_SUBSAMPLING) || !fits(config.height, CHROMA_SUBSAMPLING) {
        Err(error(CHROMA_SUBSAMPLING))
    } else if !fits(coded_width, multiple) || !fits(coded_height, multiple) {
        Err(error(multiple))
    } else {
        Ok(())
    }
}

/// Check the frame rate, intra period, temporal structure and colour
/// signalling of `config`.
/// The stream's temporal layer count: `layers.count` when a preset or
/// `--param` sets it, so that presets can turn layering on, otherwise
/// `EncoderConfig::temporal_layers`. Every later reader sees this one value;
/// `validate_stream` rejects counts outside `1..=MAX_TEMPORAL_LAYERS`.
fn configured_temporal_layers(config: &EncoderConfig) -> u8 {
    if config.params.names().any(|name| name == LAYER_COUNT_PARAM) {
        let count = LayerParams::from_overrides(&config.params).count;
        count.round().clamp(0.0, f32::from(u8::MAX)) as u8
    } else {
        config.temporal_layers
    }
}

/// The parameter that sets the temporal layer count from presets.
const LAYER_COUNT_PARAM: &str = "layers.count";

/// Whether the reuse term (`reuse.rs`) sees a frame: keys and base-layer
/// frames. Its state describes the picture later frames copy from; in a
/// layered mini-GOP that is the T0 chain (each T0 predicts from the one
/// before), while T1 and T2 frames are reused by at most the rest of their
/// mini-GOP: refining there is spent where nothing keeps it, and the T0
/// after them would find the term believing it done (a starved key left
/// static content 7.5 dB down, `crates/nc-cli/tests/static_recovery.rs`).
/// An inter-frame leaf's decided mode, the reference it predicts from and,
/// for intra that won only through the directional estimate, the inter
/// vector it beat (`BlockDecision::intra_fallback`).
type LeafChoice = (PredictionMode, ReferenceChoice, Option<Mv>, PricingHint);

/// The zero vector on the primary reference, with no search.
const ZERO_VECTOR_LEAF: LeafChoice = (
    PredictionMode::Inter(Mv::ZERO),
    ReferenceChoice::Primary,
    None,
    PricingHint::NONE,
);

fn on_reuse_chain(key: bool, references: &FrameReferences) -> bool {
    key || references.position.layer == 0
}

/// The prefix of the temporal-structure parameters (`LayerParams`).
const LAYER_PARAM_PREFIX: &str = "layers.";

/// Prefix of settings (a preset's or `--param`'s) that apply only under the
/// fixed-QP profile (`rate.fixed_qp_profile`), over the plain ones: a rung's
/// fixed-QP values beside its rate-controlled ones, where the two modes were
/// screened apart (2026-11-24: AV1 and VP9 fast code fixed QP at period 8, base
/// boost 4 and no start-up cadence while their rate control keeps its pins).
const FIXED_QP_PREFIX: &str = "fixed_qp.";

/// `params` for the stream's rate mode: the `fixed_qp.` settings over the rest
/// under the fixed-QP profile, and dropped otherwise.
fn rate_mode_settings(params: &ParamOverrides) -> ParamOverrides {
    let (fixed_qp_only, plain) = params.split_prefix(FIXED_QP_PREFIX);
    if RateParams::from_overrides(&plain).fixed_qp_profile >= 0.5 {
        plain.overlaid_with(&fixed_qp_only)
    } else {
        plain
    }
}

/// The backend's `DEFAULT_PARAMS` that apply to `config`. The backend's
/// temporal-structure defaults (H.264 Default's gated L1T3) step aside when
/// the caller sets the structure itself (a layer count above 1: an explicit
/// `--temporal-layers` keeps the engine's own layer settings) or asks for a
/// second reference in a codec that cannot signal one beside layers
/// (`CodecBackend::SECOND_REFERENCE_WITH_LAYERS`).
/// With `rate.fixed_qp_profile` the backend's `FIXED_QP_PARAMS` follow.
fn backend_defaults<B: CodecBackend>(config: &EncoderConfig) -> Vec<(&'static str, f32)> {
    let flat_only_second =
        second_reference_enabled(&config.params) && !B::SECOND_REFERENCE_WITH_LAYERS;
    let own_structure = config.temporal_layers > 1 || flat_only_second;
    let spatial = SpatialParams::from_overrides(&config.params).layers.round() as u8 > 1;
    let fixed_qp = RateParams::from_overrides(&config.params).fixed_qp_profile >= 0.5;
    // Later entries win (`ParamOverrides::layered_over`): the fixed-QP
    // profile replaces the plain defaults it names.
    let profile: &[(&str, f32)] = if fixed_qp { B::FIXED_QP_PARAMS } else { &[] };
    B::DEFAULT_PARAMS
        .iter()
        .chain(profile)
        .copied()
        .filter(|(name, _)| !(own_structure && name.starts_with(LAYER_PARAM_PREFIX)))
        .filter(|(name, _)| !(spatial && *name == "resolution.adaptive"))
        .collect()
}

fn validate_stream(config: &EncoderConfig) -> Result<(), ConfigError> {
    if config.frame_rate.num == 0 || config.frame_rate.den == 0 {
        Err(ConfigError::FrameRate)
    } else if config.intra_period == 0 {
        Err(ConfigError::IntraPeriod)
    } else if !(1..=MAX_TEMPORAL_LAYERS).contains(&config.temporal_layers) {
        Err(ConfigError::TemporalLayers(config.temporal_layers))
    } else if !config.signal.colour.is_valid_for_420() {
        Err(ConfigError::Colour)
    } else {
        Ok(())
    }
}

/// The camera's displacement against the frame's reference (luma samples):
/// the global motion fit's translation plus its zoom at the picture edge.
fn camera_displacement(motion: &MotionAnalysis, width: usize) -> f32 {
    let global = &motion.global;
    global.tx.hypot(global.ty) + global.zoom.abs() * width as f32 / 2.0
}

/// What the adaptive temporal structure reads from one frame's motion:
/// camera motion per frame (global translation plus zoom at the picture
/// edge, over the reference distance), for distance-1 frames the
/// compensable share of the frame difference, the static-block fraction
/// and the motion-compensated residual, and from the frame's analysis (`frame`, when there is one) the
/// still-block fraction and whether the input repeats the previous one.
fn motion_signal(
    motion: &MotionAnalysis,
    frame: Option<&FrameAnalysis>,
    distance: u32,
    width: usize,
) -> MotionSignal {
    let adjacent = distance == 1;
    MotionSignal {
        camera_motion: camera_displacement(motion, width) / distance.max(1) as f32,
        compensable: adjacent.then(|| motion.compensable_share()),
        still_fraction: frame.map_or(0.0, |f| f.still.fraction),
        repeat: frame.and_then(|f| f.repeat),
        static_fraction: frame.filter(|_| adjacent).map(|f| f.pre.static_fraction),
        residual: adjacent.then(|| motion.residual_per_sample()),
    }
}

/// The motion the QP cadence's motion gate reads, when the frame was
/// analysed against its immediate predecessor (the compensable share is
/// not comparable over longer distances).
fn cadence_motion_hint(motion: &MotionAnalysis, distance: u32, width: usize) -> Option<MotionHint> {
    let signal = motion_signal(motion, None, distance, width);
    signal.compensable.map(|compensable| MotionHint {
        camera_motion: f64::from(signal.camera_motion),
        compensable: f64::from(compensable),
    })
}

/// The reconstruction pool (`slots` reference slots) and, when the coded
/// size exceeds the visible one, the padding buffers.
fn frame_buffers(
    config: &EncoderConfig,
    (coded_width, coded_height): (usize, usize),
    slots: usize,
) -> nc_common::Result<(ReferencePool, Option<Padding>)> {
    let references = ReferencePool::new(coded_width, coded_height, RECON_BORDER, slots)?;
    let padding = padding_buffers(config, (coded_width, coded_height))?;
    Ok((references, padding))
}

/// The padding buffers of pictures of `config`'s size coded at
/// `(coded_width, coded_height)`; `None` when the two are equal.
fn padding_buffers(
    config: &EncoderConfig,
    (coded_width, coded_height): (usize, usize),
) -> nc_common::Result<Option<Padding>> {
    if (coded_width, coded_height) == (config.width, config.height) {
        return Ok(None);
    }
    Ok(Some(Padding {
        source: None,
        visible_recon: OnceCell::new(),
        spare_recon: RefCell::new(Some(Picture::new(config.width, config.height, 0)?)),
    }))
}

/// The top-left `width` x `height` luma of `picture` (its visible area).
fn visible_luma(picture: &Picture, (width, height): (usize, usize)) -> PlaneView<'_> {
    picture
        .plane(PlaneIndex::Y)
        .region(0, 0, width, height)
        .expect("the visible area lies inside the picture")
}

/// The adaptive-resolution policy `resolution.adaptive` asks for, checked
/// against the backend (scaled references), intra refresh and the half
/// size's dimensions; `None` when off or when the backend codes reduced
/// pictures already (AV1 superres).
fn adaptive_resolution<B: CodecBackend>(
    config: &EncoderConfig,
    backend: &B,
    picture: (usize, usize),
) -> Result<Option<ResolutionPolicy>, ConfigError> {
    let frame_rate = f64::from(config.frame_rate.num) / f64::from(config.frame_rate.den);
    let Some(policy) = ResolutionPolicy::new(
        &ResolutionParams::from_overrides(&config.params),
        frame_rate,
    ) else {
        return Ok(None);
    };
    // A backend already coding reduced pictures (AV1 superres) keeps its
    // own scheme.
    if picture != (config.width, config.height) {
        return Ok(None);
    }
    if !backend.supports_scaled_references() {
        return Err(ConfigError::Resolution(
            "resolution.adaptive needs a codec with scaled references",
        ));
    }
    // Intra refresh lays its sweep out for one picture size: it keeps the
    // full size (`Encoder::adapt_coded_resolution` holds the size while a
    // sweep runs).
    if RefreshParams::from_overrides(&config.params).period > 0.0 {
        return Ok(None);
    }
    let (width, height) = half_size(config.width, config.height);
    let half = EncoderConfig {
        width,
        height,
        ..config.clone()
    };
    check_dimensions(
        &half,
        backend.coded_size(width, height),
        1usize << backend.min_block_log2(),
    )?;
    Ok(Some(policy))
}

/// The long-term reference `ltr.interval` asks for, if the configuration
/// and backend allow it (flat structures, backends that can reference any
/// slot).
fn long_term_reference<B: CodecBackend>(
    config: &EncoderConfig,
    backend: &B,
) -> Result<Option<LongTermReference>, ConfigError> {
    let interval = LtrParams::from_overrides(&config.params).interval.round() as u32;
    if interval == 0 {
        return Ok(None);
    }
    if long_term_slot(config.temporal_layers) >= MAX_REFERENCE_SLOTS {
        return Err(ConfigError::LongTermReference(
            "no reference slot left beside this many temporal layers",
        ));
    }
    if !backend.supports_long_term_reference() {
        return Err(ConfigError::LongTermReference(
            "not supported by this codec yet",
        ));
    }
    Ok(Some(LongTermReference::new(
        interval,
        long_term_slot(config.temporal_layers),
        config.intra_period,
    )))
}

/// Reference slots the pool keeps: the structure's, and the long-term slot
/// beside them when the long-term or second reference (`long_term`) uses it.
fn pool_slots(structure: &TemporalStructure, config: &EncoderConfig, long_term: bool) -> usize {
    let long_term_slots = if long_term {
        long_term_slot(config.temporal_layers) + 1
    } else {
        1
    };
    structure.slot_count().max(long_term_slots)
}

/// The second reference `refs.second` asks for, if the configuration and
/// backend allow it. It takes the long-term slot, so not beside a
/// long-term reference (`ltr_active`) yet.
fn second_reference<B: CodecBackend>(
    config: &EncoderConfig,
    backend: &B,
    ltr_active: bool,
) -> Result<Option<SecondReference>, ConfigError> {
    if !second_reference_enabled(&config.params) || !backend.supports_inter() {
        return Ok(None);
    }
    // Beside temporal layers the HQ frame's slot follows the layer
    // pattern's, which each codec must be able to signal.
    if config.temporal_layers > 1 && !B::SECOND_REFERENCE_WITH_LAYERS {
        // An explicit `refs.second` 1 is refused; the unified plan, a default
        // way of coding, and `refs.second` 2 (a preset's) step aside and the
        // stream is coded without it.
        return if RefParams::from_overrides(&config.params).second >= 0.5
            && !second_reference_steps_aside(&config.params)
        {
            Err(ConfigError::SecondReference(
                "not with temporal layers in this codec yet",
            ))
        } else {
            Ok(None)
        };
    }
    let slot = long_term_slot(config.temporal_layers);
    if !backend.supports_second_reference() {
        return Err(ConfigError::SecondReference(
            "not supported by this codec yet",
        ));
    }
    if ltr_active {
        return Err(ConfigError::SecondReference(
            "not with a long-term reference yet",
        ));
    }
    if RefreshParams::from_overrides(&config.params).period > 0.0 {
        return Err(ConfigError::SecondReference("not with intra refresh"));
    }
    let refs = RefParams::from_overrides(&config.params);
    let rate = config.frame_rate;
    let hold = (f64::from(refs.second_hold) * f64::from(rate.num) / f64::from(rate.den)).round();
    Ok(Some(
        SecondReference::new(slot, &refs)
            .with_plan(TierPlan::new(&config.params))
            .with_hold(hold as u64, B::SECOND_REFERENCE_REACH),
    ))
}

/// `mv` scaled by `ratio` (to another reference distance), rounded to the
/// nearest whole sample: it seeds a search that refines it.
fn scaled_vector(mv: Mv, ratio: f32) -> Mv {
    let unit = MV_UNITS_PER_SAMPLE as f32;
    let scale = |v: i16| {
        let samples = (f32::from(v) * ratio / unit).round() * unit;
        samples.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
    };
    Mv {
        x: scale(mv.x),
        y: scale(mv.y),
    }
}

/// `mv` limited in each direction to the picture's size and to `reach`
/// whole samples ([`CodecBackend::VECTOR_REACH`]): a scaled seed beyond the
/// picture (a long-held second reference scales by its distance, up to 255)
/// points wholly into the padding, and on a picture wider than the codec's
/// vector range (AV1 and VP9: 2048 samples) past what the codec can code.
fn within_reach(mv: Mv, (width, height): (usize, usize), reach: usize) -> Mv {
    let clamp = |v: i16, size: usize| {
        let limit = i32::try_from(size.min(reach))
            .unwrap_or(i32::MAX)
            .saturating_mul(MV_UNITS_PER_SAMPLE);
        let limit = i16::try_from(limit).unwrap_or(i16::MAX);
        v.clamp(-limit, limit)
    };
    Mv {
        x: clamp(mv.x, width),
        y: clamp(mv.y, height),
    }
}

/// The reference structure `config` asks for: fixed or adaptive layers.
fn temporal_structure(config: &EncoderConfig) -> TemporalStructure {
    match LayerParams::from_overrides(&config.params).adaptive_rule() {
        Some(rule) if config.temporal_layers > 1 => {
            TemporalStructure::adaptive(config.temporal_layers, rule)
        }
        _ => TemporalStructure::new(config.temporal_layers),
    }
}

/// Check every setting in `params` (non-base `nonbase.*` settings too)
/// against the engine's and `B`'s registries, exactly as [`Encoder::new`]
/// does: front ends use it to report a bad setting before encoding.
///
/// # Errors
///
/// [`ConfigError::Param`] for the first unknown or out-of-range setting.
pub fn validate_params<B: CodecBackend>(params: &ParamOverrides) -> Result<(), ConfigError> {
    validated_params::<B>(params).map(|_| ())
}

/// Check every setting in `params` against the engine's and `B`'s
/// registries, and build the non-base effort overlay they ask for.
fn validated_params<B: CodecBackend>(
    params: &ParamOverrides,
) -> Result<(Option<EffortOverlay>, Option<LightOverlay>), ConfigError> {
    let specs = [engine_specs(), B::PARAM_SPECS.to_vec()].concat();
    // Fixed-QP-only settings are checked whatever the stream's rate mode.
    let (fixed_qp_only, params) = params.split_prefix(FIXED_QP_PREFIX);
    fixed_qp_only.validate(&specs).map_err(ConfigError::Param)?;
    let (light, framed) = params.split_prefix(LIGHT_PREFIX);
    let (nonbase, ordinary) = framed.split_prefix(NONBASE_PREFIX);
    let (_, ordinary) = ordinary.split_prefix(KEY_PREFIX);
    ordinary.validate(&specs).map_err(ConfigError::Param)?;
    let effort =
        EffortOverlay::from_overrides(&framed, B::FRAME_PARAM_SPECS, B::STREAM_FIXED_PARAMS)
            .map_err(ConfigError::Param)?;
    let light = LightOverlay::from_overrides(
        &light,
        &ordinary,
        // Key-only settings leave the non-base frames' backend values alone.
        effort
            .as_ref()
            .filter(|_| !nonbase.is_empty())
            .map(|e| &e.nonbase_overrides),
        (B::PARAM_SPECS, B::SUPERBLOCK_PARAMS),
    )
    .map_err(ConfigError::Param)?;
    Ok((effort, light))
}

/// The QP cadence of `config`, scaled for `backend` (`RateModel::cadence_scale`).
fn configured_cadence<B: CodecBackend>(config: &EncoderConfig, backend: &B) -> QpCadence {
    let params = RateParams::from_overrides(&config.params);
    QpCadence::from_params(&params).scaled(cadence_scale(&params, &backend.rate_model()))
}

/// The motion search's band evidence (`threshold_control.rs`, `adapt.motion`):
/// each superblock that searched past the one-position exit `search` set is a
/// run when the zero-vector SAD per sample the gates judged was just above that
/// exit, and a win
/// when its best 64x64 prediction saves at least [`MOTION_WIN_SHARE`] of the
/// zero vector's SAD, which the exit would have kept.
fn record_motion_evidence(
    motion: &MotionAnalysis,
    search: &SearchConfig,
    evidence: &mut FrameEvidence,
) {
    let threshold = search.gates.one_position_per_pixel;
    for stats in motion.sb.cells() {
        if matches!(stats.search, SbSearch::Static | SbSearch::OnePosition) {
            continue;
        }
        let zero = stats.zz_sad as f32;
        let best = stats.me_dist[ME_DIST_64X64] as f32;
        evidence.record_above(
            Tool::Motion,
            stats.gated_per_pixel,
            threshold,
            best < zero * (1.0 - MOTION_WIN_SHARE),
        );
    }
}

/// Depth removal's band evidence (`threshold_control.rs`, `adapt.depth`):
/// a superblock just above removal's thresholds is a run, and a win when
/// `partition` still plans it below 32x32.
fn record_depth_evidence<B: CodecBackend>(
    planner: &Planner<'_, B>,
    sb: &Block,
    partition: &Partition,
    evidence: &mut FrameEvidence,
) {
    if !evidence.gathers(Tool::Depth) {
        return;
    }
    if let Some(margin) = planner.depth_removal_margin(sb) {
        let small = partition.smallest_side_log2(sb.log2_size()) < DEPTH_REMOVAL_LOG2;
        evidence.record_above(Tool::Depth, margin, DEPTH_MARGIN_THRESHOLD, small);
    }
}

/// Depth removal keeps nothing below 32x32 (log2 5).
const DEPTH_REMOVAL_LOG2: u8 = 5;
/// Depth removal's margin at its thresholds ([`Planner::depth_removal_margin`]
/// is the improvements over them: removal fires below 1).
const DEPTH_MARGIN_THRESHOLD: f32 = 1.0;

/// The share of the zero vector's SAD a searched superblock's best 64x64
/// prediction must save to count as the search changing its prediction.
const MOTION_WIN_SHARE: f32 = 0.125;
/// Index of the 64x64 sum in [`nc_analysis::motion::SbStats::me_dist`].
const ME_DIST_64X64: usize = 3;

/// The source analyzer for `config`, keeping `slots` reference sources and
/// detecting repeated inputs if the structure reads them; `None` when the
/// picture is too small for the analysis pyramid.
/// The source analyzer.
fn source_analyzer(config: &EncoderConfig, slots: usize, detect_repeats: bool) -> Option<Analyzer> {
    let mut analyzer =
        Analyzer::with_reference_slots(config.width, config.height, Dsp::new(), slots).ok()?;
    analyzer.set_repeat_detection(detect_repeats);
    let analysis = AnalysisParams::from_overrides(&config.params);
    analyzer.set_search_config(analysis.search_config());
    analyzer.set_sampling(analysis.stat_rows(), analysis.downsample());
    analyzer.set_still_config(StillParams::from_overrides(&config.params).still_config());
    Some(analyzer)
}

/// Whether a coding may stop part-way ([`Encoder::code_picture`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopRule {
    /// It codes the whole picture.
    Never,
    /// In-frame rate control may stop it (`rate.recode_early`).
    InFrame,
    /// It stops after half its superblock rows (`debug.stop_identity`).
    Halfway,
}

/// A coding in-frame rate control stopped part-way (`rate.recode_early`):
/// the frame is re-coded as if it had taken `bits`.
#[derive(Debug, Clone, Copy)]
struct StoppedCoding {
    bits: f64,
}

/// Per-frame inputs to the inter-frame block decision.
#[derive(Debug, Clone, Copy)]
struct FrameInputs<'a> {
    /// Motion search against the reference's source frame (`None` without
    /// an analyzer or on the first frame).
    motion: Option<&'a MotionAnalysis>,
    /// Pre-analysis (static map) against the reference's source frame;
    /// `None` on the first frame or without an analyzer.
    pre: Option<&'a PreAnalysis>,
    /// Reliably static blocks (`nc_analysis::still`); `None` as `pre`.
    still: Option<&'a StillMap>,
    /// The secondary reference's distance over the primary's, when the
    /// frame has one: motion measured against the primary is scaled by it
    /// to seed the secondary's search.
    secondary_ratio: Option<f32>,
}

/// The motion search's whole-sample vector for `block`: the field matching
/// the block size (8x8 fine field, or the 16/32/64 aggregated fields); for a
/// rectangle, that of its top-left square of the shorter side. Zero where no
/// search result exists (no analysis, or a block in the padded area outside
/// the visible picture).
pub(crate) fn search_vector<B: CodecBackend>(motion: Option<&MotionAnalysis>, block: &Block) -> Mv {
    square_search_vector(motion, block.x, block.y, leaf_log2_min_side::<B>(block))
}

/// The distinct motion-search vectors of the squares of `block`'s shorter
/// side that tile it (one for a square; the first is [`search_vector`]),
/// written to the front of `out`; returns how many.
pub(crate) fn search_vectors(
    motion: Option<&MotionAnalysis>,
    block: &Block,
    out: &mut [Mv; MAX_TILING_SQUARES],
) -> usize {
    let side = 1usize << block.log2_min_side();
    let tiles = block.area() / (side * side);
    let mut count = 0;
    for tile in 0..tiles.min(MAX_TILING_SQUARES) {
        let (x, y) = if block.log2_width > block.log2_height {
            (block.x + tile * side, block.y)
        } else {
            (block.x, block.y + tile * side)
        };
        let mv = square_search_vector(motion, x, y, block.log2_min_side());
        if !out[..count].contains(&mv) {
            out[count] = mv;
            count += 1;
        }
    }
    count
}

/// Most squares of a block's shorter side that tile one engine block (a
/// quarter-high ternary leaf).
pub(crate) const MAX_TILING_SQUARES: usize = 4;

/// [`search_vector`] of the square of side `1 << log2` at (`x`, `y`).
fn square_search_vector(motion: Option<&MotionAnalysis>, x: usize, y: usize, log2: u8) -> Mv {
    let Some(motion) = motion else {
        return Mv::ZERO;
    };
    let (col, row) = ((x >> log2) as isize, (y >> log2) as isize);
    let mv = match log2 {
        3 => motion.fine.try_get(col, row).map(|b| b.mv),
        4..=6 => motion.aggregated[usize::from(log2 - 4)]
            .blocks
            .try_get(col, row)
            .map(|b| b.mv),
        _ => None,
    };
    mv.map_or(Mv::ZERO, |v| {
        Mv::from_full_pel(i32::from(v.x), i32::from(v.y))
    })
}

/// Copy `src` into the larger `dst`, filling the extra columns and rows by
/// repeating the last visible sample (what the padded area should look like
/// to keep its residual small; it is cropped away after decoding).
fn copy_padded(src: &Picture, dst: &mut Picture) {
    for index in PlaneIndex::ALL {
        let from = src.plane(index).view();
        let mut to = dst.plane_mut(index).view_mut();
        let (w, h) = (from.width(), from.height());
        for y in 0..to.height() {
            let row = from.row(y.min(h - 1));
            let out = to.row_mut(y);
            out[..w].copy_from_slice(row);
            let last = row[w - 1];
            out[w..].fill(last);
        }
    }
}

/// Copy the top-left (visible) part of `src` into the smaller `dst`.
fn copy_visible(dsp: &Dsp, src: &Picture, dst: &mut Picture) {
    for index in PlaneIndex::ALL {
        let from = src.plane(index).view();
        let mut to = dst.plane_mut(index).view_mut();
        dsp.copy_plane(&from, &mut to);
    }
}

/// The step of every native quantizer, and that table extended by
/// `rate.lambda_headroom_octaves` for rate control ([`extend_steps`]).
fn quantizer_step_tables<B: CodecBackend>(
    backend: &B,
    config: &EncoderConfig,
) -> (Vec<f64>, Vec<f64>) {
    let steps: Vec<f64> = (0..=max_quantizer(config.codec))
        .map(|q| backend.rate_control_quantizer_step(q))
        .collect();
    let headroom = if B::SUPPORTS_LAMBDA_HEADROOM {
        RateParams::from_overrides(&config.params).lambda_headroom_octaves
    } else {
        0.0
    };
    let rate_steps = extend_steps(&steps, f64::from(headroom));
    (steps, rate_steps)
}

/// Native quantizers used to extrapolate the step table past its end.
const STEP_EXTRAPOLATION_UNITS: usize = 8;

/// `steps` continued past its last quantizer until the step is `octaves`
/// octaves larger, at the mean per-unit ratio of its last
/// [`STEP_EXTRAPOLATION_UNITS`] (exactly `2^(1/6)` for the QP scale; the
/// AC-table slope for qindex).
fn extend_steps(steps: &[f64], octaves: f64) -> Vec<f64> {
    let mut extended = steps.to_vec();
    let last = steps.len() - 1;
    let ratio = (steps[last] / steps[last - STEP_EXTRAPOLATION_UNITS])
        .powf(1.0 / STEP_EXTRAPOLATION_UNITS as f64);
    let limit = steps[last] * 2f64.powf(octaves) * (1.0 + 1e-9);
    loop {
        let next = extended[extended.len() - 1] * ratio;
        if next > limit {
            break extended;
        }
        extended.push(next);
    }
}

/// Bits of a coded unit.
/// `params` without the key-frame leaf search (`decision.intra_rd_*`).
fn without_key_leaf_searches(params: &DecisionParams) -> DecisionParams {
    DecisionParams {
        intra_rd_modes: 0.,
        intra_rd_transforms: 0.,
        intra_rd_chroma: 0.,
        ..*params
    }
}

/// `params` without the key-frame partition search (`planner.key_rd_split`).
fn without_key_partition_search(params: &PlannerParams) -> PlannerParams {
    PlannerParams {
        key_rd_split: 0.,
        ..*params
    }
}

fn bit_count(data: &[u8]) -> u64 {
    (data.len() as u64).saturating_mul(8)
}

/// Largest native quantizer: AV1/VP9/AV2 qindex 255, H.264/HEVC QP 51, VVC QP 63.
const fn max_quantizer(codec: Codec) -> u32 {
    match codec {
        Codec::Av1 | Codec::Vp9 | Codec::Av2 => 255,
        Codec::Vvc => 63,
        _ => 51,
    }
}

/// Smallest lossy native quantizer (AV1, VP9 and AV2 qindex 0 is lossless coding).
const fn min_quantizer(codec: Codec) -> u32 {
    match codec {
        Codec::Av1 | Codec::Vp9 | Codec::Av2 => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scaled_seed_past_the_picture_is_held_at_its_edge() {
        // A 40-sample vector to a second reference 255 frames back.
        let seed = scaled_vector(Mv { x: 160, y: -40 }, 255.0);
        let held = within_reach(seed, (1280, 720), usize::MAX);
        assert_eq!(
            held,
            Mv {
                x: 1280 * 4,
                y: -720 * 4
            }
        );
        let small = Mv { x: 12, y: -8 };
        assert_eq!(within_reach(small, (64, 64), usize::MAX), small);
    }
    #[test]
    fn a_scaled_seed_stays_within_the_codec_reach_on_a_wider_picture() {
        // AV1 and VP9 code components below 2048 samples; the picture is wider.
        let seed = scaled_vector(Mv { x: 160, y: -40 }, 255.0);
        let held = within_reach(seed, (2560, 360), 2047);
        assert_eq!(
            held,
            Mv {
                x: 2047 * 4,
                y: -360 * 4
            }
        );
    }
    #[test]
    fn fixed_qp_settings_apply_only_under_the_fixed_qp_profile() {
        let mut params = ParamOverrides::default();
        params.set("rate.cadence_period", 4.0);
        params.set("fixed_qp.rate.cadence_period", 8.0);
        let rate_controlled = rate_mode_settings(&params);
        assert_eq!(rate_controlled.get("rate.cadence_period"), Some(4.0));
        assert!(
            rate_controlled
                .names()
                .all(|name| !name.starts_with(FIXED_QP_PREFIX))
        );
        params.set("rate.fixed_qp_profile", 1.0);
        assert_eq!(
            rate_mode_settings(&params).get("rate.cadence_period"),
            Some(8.0)
        );
    }
}
