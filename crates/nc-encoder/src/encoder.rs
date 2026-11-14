//! The frame loop: drives a [`CodecBackend`] over every frame and block.
//!
//! Per frame: choose the frame type (a key frame every `intra_period`
//! frames, inter frames in between when the backend supports them) and its
//! place in the temporal structure (`structure.rs`: which reference slot it
//! predicts from, which slots it refreshes), run source analysis (motion
//! search against the reference's *source* frame), walk the partition,
//! decide each leaf's mode, and keep the reconstruction in the slots it
//! refreshes.

use std::cell::OnceCell;

use nc_analysis::preanalysis::PreAnalysis;
use nc_analysis::still::StillMap;
use nc_analysis::{Analyzer, FrameAnalysis, MotionAnalysis, NoTimer};
use nc_common::{Picture, PlaneIndex, PlaneView, RawFrame};
use nc_dsp::Dsp;

use crate::aq::{BlockAq, native_offset};
use crate::backend::{
    BlockDecision, CodecBackend, FramePlan, FrameType, PredictionMode, ReferenceChoice,
};
use crate::config::{Codec, ConfigError, EncoderConfig, FrameRate};
use crate::cyclic_cadence::CyclicCadence;
use crate::decision::{
    BlockSads, Choice, DecisionScratch, choose_compound_mode, choose_inter_mode,
    choose_secondary_mode, is_square_leaf, is_static, is_still, leaf_area, leaf_extent,
    leaf_log2_min_side,
};
use crate::feedback::{LossFeedback, ReferencedFrames};
use crate::frame_role::{EffortOverlay, FrameRole, NONBASE_PREFIX};
use crate::hold::{FrameHold, LastCoded};
use crate::inframe::{InFrameControl, row_weights};
use crate::input::write_raw;
use crate::key_clock::KeyClock;
use crate::key_plan::{KeyPlanMode, KeyPlanner};
use crate::layer::LayerPosition;
use crate::ltr::{LongTermReference, LtrAction, long_term_slot};
use crate::mv::{MV_UNITS_PER_SAMPLE, Mv};
use crate::noise_skip::{NoiseMap, camera_is_still};
use crate::params::RefParams;
use crate::params::{
    AnalysisParams, DebugParams, DecisionParams, InFrameParams, LayerParams, LtrParams,
    ParamOverrides, PersistParams, PlannerParams, RateParams, RefreshParams, ResolutionParams,
    SceneParams, SpatialParams, StillParams, engine_specs,
};
use crate::partition::{Block, should_split};
use crate::partition_plan::{LeafStats, NoiseTest, Partition, Planner};
use crate::planned_refinement::StillPlan;
use crate::qp_cadence::{
    CadenceRole, LayerLadder, MotionHint, QP_UNITS_PER_OCTAVE, QpCadence, nearest_quantizer,
    step_ratio,
};
use crate::quality_cap::residual_sigma;
use crate::random_decisions::{DebugRng, RandomDecisions};
use crate::rate_control::{
    Complexity, LayerShare, RateControlConfig, RateControlError, RateControlStats, RateController,
    RateModel, cadence_scale, spatial_aware,
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
use crate::second_reference::{SecondReference, second_reference_enabled};
use crate::shape::{MAX_SHAPE_PARTS, PartitionShape, ShapeSet};
use crate::source_scale::{SourceScaler, scale_source_picture};
use crate::spatial::SpatialConfig;
use crate::startup_lookahead::startup_camera_motion;
use crate::still_skip::StaticSkipTest;
use crate::structure::{FrameReferences, MAX_TEMPORAL_LAYERS, MotionSignal, TemporalStructure};
use crate::timeline::{FrameTime, Timeline};
use nc_analysis::link::{MAX_REFERENCE_SLOTS, SlotMask};

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
    /// Variance AQ (`rate.aq_strength`), for backends with block-level QP.
    pub(super) aq: Option<BlockAq>,
    /// The reuse term (`persist.reuse`, `reuse.rs`): per block through
    /// block AQ, or one offset per key frame for codecs without block QP.
    pub(super) reuse: Option<ReuseTerm>,
    /// The last frame the reuse term saw (the reference its state describes).
    pub(super) reuse_last_frame: Option<u64>,
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
pub struct Encoder<B: CodecBackend> {
    config: EncoderConfig,
    backend: B,
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
    /// The periodic key schedule for timestamped frames (`key_clock.rs`).
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
    /// ([`Encoder::look_ahead`]); consumed by the first frame.
    lookahead: Option<Picture>,
    /// The frame being coded, when it is on the reuse term's chain
    /// ([`on_reuse_chain`]): the term sees only those frames.
    reuse_frame: Option<u64>,
    /// `(expected bits weight, offset in QP units)` of every block coded with AQ this frame:
    /// the realised offsets rate control normalizes its model with.
    block_offsets: Vec<(f64, f64)>,
    /// In-frame rate control (`inframe.*`), for backends that code block
    /// quantizers and report their bits mid-frame; `None` when off.
    inframe: Option<InFrameControl>,
    /// The cyclic spatial cadence (`rate.cadence_spatial`), for flat
    /// single-layer streams with block quantizers and the reuse term.
    cyclic: Option<CyclicCadence>,
    /// Planned refinements and the hold of still blocks (`rate.refine`,
    /// `rate.hold_still`, `rate.prekey_quiet`), for flat single-layer
    /// streams with block quantizers and the reuse term.
    still: StillPlan,
    /// Scratch: the reuse term's and the band's offsets of every cell.
    block_terms: Vec<f64>,
    /// Quantizer of the frame being coded (block offsets are relative to it).
    frame_quantizer: u32,
    force_keyframe: bool,
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
    /// Scratch list of candidate vectors, reused for every inter block.
    decision_scratch: DecisionScratch,
    /// Tunable partition-planner parameters.
    planner_params: PlannerParams,
    /// Tunable mode-decision parameters.
    decision_params: DecisionParams,
    /// `decision_params` as adjusted for the frame being coded (cadence).
    frame_decision_params: DecisionParams,
    /// Cheaper settings for non-base frames (`nonbase.*`, `frame_role.rs`);
    /// `None` without such settings.
    effort_overlay: Option<EffortOverlay>,
    /// `debug.recode_identity`: every frame is coded twice (recode test).
    recode_identity: bool,
    /// `debug.random_decisions`: random legal decisions replace the normal
    /// ones (syntax-coverage stress test). `None` in every real encode.
    random: Option<RandomDecisions>,
    /// Multi-layer spatial scalability session (`spatial.layers == 2`);
    /// `None` on every single-layer stream.
    spatial: Option<Box<SpatialSession>>,
}

/// Buffers needed when the codec codes a padded picture.
#[derive(Debug)]
pub(super) struct Padding {
    source: Picture,
    visible_recon: Picture,
}

impl<B: CodecBackend> Encoder<B> {
    /// Create an encoder for `config`.
    ///
    /// # Errors
    ///
    /// A [`ConfigError`] if the engine or the backend rejects the settings.
    #[allow(clippy::too_many_lines)]
    pub fn new(mut config: EncoderConfig) -> Result<Self, ConfigError> {
        config.params = config.params.layered_over(&backend_defaults::<B>(&config));
        config.temporal_layers = configured_temporal_layers(&config);
        validate_stream(&config)?;
        let effort_overlay = validated_params::<B>(&config.params)?;
        let planner_params = PlannerParams::from_overrides(&config.params);
        let decision_params = DecisionParams::from_overrides(&config.params);
        let mut backend = B::new(&config)?;
        check_temporal_layers(&config, &backend)?;
        let spatial_cfg =
            SpatialConfig::from_encoder_config_max(&config, backend.max_spatial_layers())?;
        if let Some(ref scfg) = spatial_cfg {
            if B::OWN_PICTURE_PIPELINE {
                return Err(ConfigError::SpatialScalability(
                    "not supported by picture-owning backends yet",
                ));
            }
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
        let reuse = (PersistParams::from_overrides(&config.params).reuse >= 0.5)
            .then(|| ReuseTerm::new(B::SUPPORTS_BLOCK_QP));
        let cadence_fixed_qp = RateParams::from_overrides(&config.params).cadence_fixed_qp >= 0.5;
        let scene = SceneDetector::new(&SceneParams::from_overrides(&config.params));
        let use_aq = aq.is_some();
        let cyclic = CyclicCadence::new(
            RateParams::from_overrides(&config.params).cadence_spatial >= 0.5
                && spatial_cfg.is_none()
                && use_aq
                && reuse.is_some(),
            unflattened_cadence(&config, &backend),
            qp_log2,
        );
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
            // Picture-owning codecs need shared motion costs only for opt-in cuts.
            let analyzer = (backend.supports_inter()
                && (!B::OWN_PICTURE_PIPELINE || scene.is_some()))
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
                aq,
                reuse,
                reuse_last_frame: None,
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
        let inframe = InFrameControl::new(&InFrameParams::from_overrides(&config.params))
            .filter(|_| B::SUPPORTS_BLOCK_QP && B::REPORTS_CODED_BITS);
        enable_backend_options(&mut backend, random.as_ref(), use_aq || inframe.is_some());
        let encoder = Self {
            recode_identity,
            hold: FrameHold::new(&config),
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
            lookahead: None,
            reuse_frame: None,
            block_offsets: Vec::new(),
            inframe,
            cyclic,
            still,
            block_terms: Vec::new(),
            frame_quantizer: 0,
            force_keyframe: false,
            ltr,
            feedback: LossFeedback::default(),
            refresh: None,
            window: None,
            scene,
            decision_scratch: DecisionScratch::default(),
            planner_params,
            decision_params,
            frame_decision_params: decision_params,
            effort_overlay,
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
        if let Some(spatial) = self.spatial.as_ref() {
            return spatial.output_reconstruction();
        }
        if let Some(output) = self.backend.output_picture() {
            return output;
        }
        if let Some(previous) = self.switched_recon.as_ref() {
            return previous;
        }
        match self.layer.padding.as_ref() {
            Some(p) => &p.visible_recon,
            None => self
                .layer
                .references
                .as_ref()
                .expect("references are present between frames")
                .last(),
        }
    }

    /// Enable or update rate control without resetting references or learned rate models.
    /// # Errors
    /// Rejects invalid targets, buffers or actual codec quantizer bounds.
    pub fn set_rate_control(&mut self, config: RateControlConfig) -> Result<(), RateControlError> {
        if self.spatial.is_some() {
            return self.set_spatial_rate_control(config);
        }
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
                self.config.codec,
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
            return;
        }
        self.layer.rate_control = None;
        self.backend.set_rate_control_enabled(false);
    }

    /// Packet feedback and VBV statistics, or `None` in fixed-QP mode.
    #[must_use]
    pub fn rate_control_stats(&self) -> Option<RateControlStats> {
        if let Some(spatial) = self.spatial.as_ref() {
            return spatial.combined_rate_control_stats(&self.layer);
        }
        self.layer.rate_control.as_ref().map(RateController::stats)
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

    #[allow(clippy::too_many_arguments)]
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
        let quantizer = if let Some(controller) = self.layer.rate_control.as_mut() {
            // Reuse this frame's existing8x8 statistics; no lookahead/pixel pass.
            let mut hint = Complexity::from_preanalysis(pre);
            if controller.wants_temporal_noise() {
                hint.residual_sigma = motion.and_then(|m| residual_sigma(&m.fine));
            }
            hint.motion = cadence_motion;
            hint.key_reuse_offset = key_reuse_offset;
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
                // A refinement frame's level is planned as without its
                // refined blocks, whose bits are its loan.
                if !self.still.without_offsets {
                    hint.block_qp_offset =
                        controller.effective_block_offset(aq.total_weight(), aq.offset_cells());
                }
            }
            let quantizer = controller.select(rc_key, index, hint, references.position, share);
            if let Some(inframe) = self.inframe.as_mut() {
                let (room, uniform) = controller.inframe_bounds();
                let sb_log2 = self.backend.superblock_log2();
                let rows = self.layer.coded_height.div_ceil(1 << sb_log2);
                let startup = self
                    .layer
                    .cadence
                    .in_startup(index - self.layer.last_key_index);
                let limit = inframe.frame_limit((rc_key, startup), room, uniform);
                let weights = limit.map(|_| row_weights(pre, sb_log2, rows));
                inframe.plan_frame(limit, weights.into_iter().flatten());
            }
            quantizer
        } else if frame_type == FrameType::Key || self.config.quantizer < self.quantizer_bounds().0
        {
            // Keys keep the configured (T0) quantizer; backends apply their
            // own key boost relative to it. A quantizer below the lossy range
            // (qindex 0: lossless in VP9 and AV2) is never offset.
            self.config.quantizer
        } else if !references.position.is_flat() {
            let step = self.quantizer_steps[self.config.quantizer as usize]
                * step_ratio(self.layer.ladder.offset(references.position));
            nearest_quantizer(&self.quantizer_steps, self.quantizer_bounds(), step)
        } else if self.cadence_fixed_qp && !self.layer.cadence.is_flat() {
            let strength = self.layer.cadence.strength(
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
                || self.inframe.as_ref().is_some_and(InFrameControl::steering),
        }
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
    /// frames, or only the first while intra refresh replaces periodic keys.
    /// A timestamped frame follows time instead (`key_clock.rs`): a live
    /// source's skipped captures do not stretch the key interval; untimed
    /// frames keep the frame-count schedule exactly.
    fn periodic_key(&self, index: u64) -> bool {
        if self.frame_time.timed {
            self.key_clock
                .is_due(self.frame_time.timestamp_us, self.key_period())
        } else {
            index.is_multiple_of(self.key_period())
        }
    }

    /// Block AQ, when the frame being coded has any non-zero offset: a
    /// frame whose offsets are all zero (most frames with only the reuse
    /// term on) codes, budgets and ranks exactly as without block AQ, and
    /// skips the per-block offset work (`pan_jaad_0117` H.264: 4% of the
    /// instructions).
    fn active_aq(&self) -> Option<&BlockAq> {
        self.layer.aq.as_ref().filter(|aq| aq.active())
    }

    /// Frames from `index` to the next scheduled key frame (`None` when no
    /// later key is scheduled).
    fn frames_to_next_key(&self, index: u64) -> Option<u64> {
        let period = self.key_period();
        (period != u64::MAX && period > 0).then(|| {
            if self.frame_time.timed {
                self.key_clock
                    .frames_to_next(self.frame_time.timestamp_us, period)
            } else {
                period - index % period
            }
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

    /// Plan the cyclic band of inter frame `index` (`cyclic_cadence.rs`)
    /// into `block_terms`, beside the reuse term's offsets: whether any
    /// cell has a band offset. Keys, layered frames and frames without a
    /// still map plan none.
    fn plan_band(
        &mut self,
        index: u64,
        key: bool,
        references: &FrameReferences,
        analysis: &FrameAnalysis,
        motion: Option<&MotionAnalysis>,
    ) -> bool {
        if key || !references.position.is_flat() || self.cyclic.is_none() {
            return false;
        }
        let level_step = self.reuse_sample_step();
        let distance = index - self.layer.last_key_index;
        let hint = motion
            .and_then(|m| cadence_motion_hint(m, references.reference.distance, self.layer.width));
        let reuse = self
            .layer
            .reuse
            .as_ref()
            .filter(|_| self.reuse_frame.is_some())
            .map(ReuseTerm::offsets);
        let cyclic = self.cyclic.as_mut().expect("checked above");
        cyclic.observe(&analysis.pre, motion);
        let strength = cyclic.strength(f64::from(analysis.pre.static_fraction), hint);
        let size = (analysis.pre.blocks.cols(), analysis.pre.blocks.rows());
        let band = cyclic.plan(size, distance, level_step, strength);
        if band.is_empty() {
            return false;
        }
        self.block_terms.clear();
        match reuse.filter(|r| r.len() == band.len()) {
            Some(reuse) => self
                .block_terms
                .extend(reuse.iter().zip(band).map(|(r, b)| r + b)),
            None => self.block_terms.extend_from_slice(band),
        }
        true
    }

    /// The sample-unit step of native `quantizer`.
    fn sample_step(&self, quantizer: u32) -> f64 {
        let quantizer = (quantizer as usize).min(self.quantizer_steps.len() - 1);
        self.quantizer_steps[quantizer] * self.backend.rate_model().sample_step_scale
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
    /// The dimensions the backend rejects (the encoder is unchanged), or a
    /// backend that owns its picture pipeline (VP9, AV2: not supported).
    pub fn set_resolution(&mut self, width: usize, height: usize) -> Result<(), ConfigError> {
        if (width, height) == (self.config.width, self.config.height) {
            return Ok(());
        }
        if B::OWN_PICTURE_PIPELINE {
            return Err(ConfigError::Resolution(
                "not yet with a picture-owning backend (VP9, AV2)",
            ));
        }
        let mut next = Self::new(EncoderConfig {
            width,
            height,
            ..self.config.clone()
        })?;
        next.backend.continue_from(&self.backend);
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
        if self.layer.reuse.is_some() {
            self.layer.reuse = Some(ReuseTerm::new(B::SUPPORTS_BLOCK_QP));
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
    /// the first flat frame); fixed layered structures are not supported yet.
    ///
    /// # Errors
    ///
    /// [`ConfigError::IntraRefresh`] with fixed temporal layers, without inter
    /// frames, for a picture-owning backend, or for a period below
    /// [`MIN_REFRESH_PERIOD`].
    pub fn set_intra_refresh(&mut self, refresh: Option<IntraRefresh>) -> Result<(), ConfigError> {
        if let Some(settings) = refresh {
            if self.spatial.is_some() {
                return Err(ConfigError::IntraRefresh(
                    "not supported with spatial layers",
                ));
            }
            // An adaptive structure is held flat instead (the sweep runs on
            // flat frames only); a fixed layered one cannot be.
            if self.config.temporal_layers > 1 && !self.structure.is_adaptive() {
                return Err(ConfigError::IntraRefresh("not yet with temporal layers"));
            }
            if !self.backend.supports_inter() {
                return Err(ConfigError::IntraRefresh(
                    "the backend codes no inter frames",
                ));
            }
            // A picture-owning backend (VP9, AV2) never sees the refresh
            // window, so the sweep would silently replace keys with nothing.
            if B::OWN_PICTURE_PIPELINE {
                return Err(ConfigError::IntraRefresh(
                    "not yet with a picture-owning backend (VP9, AV2)",
                ));
            }
            if self.ltr.is_some() {
                return Err(ConfigError::IntraRefresh("not with a long-term reference"));
            }
            // A block predicting from the HQ frame may reach back past the
            // sweep's clean region.
            if self.layer.second.is_some() {
                return Err(ConfigError::IntraRefresh("not with a second reference"));
            }
            if settings.period < MIN_REFRESH_PERIOD {
                return Err(ConfigError::IntraRefresh("period below 2 frames"));
            }
        }
        self.structure.set_hold_flat(refresh.is_some());
        if refresh != self.refresh.as_ref().map(RefreshSweep::settings) {
            self.refresh = refresh.map(|settings| {
                RefreshSweep::new(
                    settings,
                    self.layer.coded_width,
                    1 << self.backend.min_block_log2(),
                    self.backend.loop_filter_reach(),
                )
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
        if key || referenced_nonbase || self.is_cadence_base(index, references.position) {
            FrameRole::Base
        } else {
            FrameRole::NonBase
        }
    }

    /// Before analysing frame `index`, the motion search settings of its
    /// role, when the effort overlay is on. A scene cut found by this
    /// analysis makes the frame a key frame after its search ran with the
    /// non-base settings; only the cut decision sees that search.
    fn apply_search_role(
        &self,
        analyzer: &mut Analyzer,
        index: u64,
        planned: &FrameReferences,
        key: bool,
    ) {
        if let Some(overlay) = &self.effort_overlay {
            let role = self.frame_role(index, planned, key);
            analyzer.set_search_config(overlay.params(role).search);
        }
    }

    /// Before coding frame `index`, the backend's settings of its `role`,
    /// when the effort overlay is on.
    fn apply_backend_role(&mut self, role: FrameRole) {
        if let Some(overlay) = &self.effort_overlay {
            match role {
                FrameRole::Base => self.backend.set_frame_params(&self.config.params),
                FrameRole::NonBase => self.backend.set_frame_params(&overlay.nonbase_overrides),
            }
        }
    }

    /// The decision and planner parameters as adjusted for one frame.
    fn frame_params(
        &self,
        index: u64,
        references: &FrameReferences,
        role: FrameRole,
        analysis: Option<&FrameAnalysis>,
    ) -> (DecisionParams, PlannerParams) {
        let noise_sigma = analysis.map(|a| a.noise_sigma);
        let (mut decision_params, mut planner_params) = match &self.effort_overlay {
            Some(overlay) => {
                let params = overlay.params(role);
                (params.decision, params.planner)
            }
            None => (self.decision_params, self.planner_params),
        };
        // A cadence base frame may get a lower decision/planner λ.
        if self.is_cadence_base(index, references.position) {
            decision_params.lambda_scale *= decision_params.base_lambda_scale;
            planner_params.lambda_scale *= decision_params.base_lambda_scale;
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
    /// (`rate.first_key_lookahead`; ignored when that is off, without rate
    /// control, or after the first frame). The caller waits one frame
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
            self.lookahead = Some(picture);
        }
        Ok(())
    }

    /// Whether [`Encoder::look_ahead`] would be used: before the first
    /// frame, with rate control and `rate.first_key_lookahead` on. A capture
    /// loop holds the first frame back only then.
    #[must_use]
    pub fn wants_look_ahead(&self) -> bool {
        if let Some(spatial) = self.spatial.as_ref() {
            return spatial.wants_look_ahead(&self.layer);
        }
        self.next_index == 0
            && self
                .layer
                .rate_control
                .as_ref()
                .is_some_and(RateController::wants_startup_motion)
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
        if self.hold_due() {
            return self.hold_frame(timestamp_us);
        }
        self.switched_recon = None;
        let index = self.next_index;
        let time = self.timeline.next(timestamp_us);
        self.frame_time = time;
        if let Some(controller) = self.layer.rate_control.as_mut() {
            controller.set_frame_interval(time.interval);
        }
        if self.drop_due(index) {
            return self.drop_frame(index, time);
        }
        self.next_index += 1;
        let (scheduled, ltr_action) = self.ltr_plan(index);
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
            if let Some(scene) = self.scene.as_mut() {
                scene.reset();
            }
        }
        let key = frame_type == FrameType::Key;
        let references = self.structure.next(index, key || recovery);
        let ltr_action = if key { LtrAction::None } else { ltr_action };
        let references = self.with_ltr(ltr_action, index, references);
        let still_fraction = analyzer
            .as_ref()
            .and_then(Analyzer::analysis)
            .filter(|a| a.motion.is_some())
            .map(|a| a.still.fraction);
        let references = self.with_second(index, references, (key, still_fraction));
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
        let analysis = analyzer.as_ref().and_then(Analyzer::analysis);
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
        // With an analyzer the source is its copy of the frame (the same
        // visible samples as `input`).
        let source = match (analyzer.as_ref(), input) {
            (Some(a), _) => a.frame(),
            (None, Some(source)) => source,
            (None, None) => own_input
                .as_ref()
                .expect("an input buffer without an analyzer"),
        };
        if let (Some(next), Some(controller)) =
            (self.lookahead.take(), self.layer.rate_control.as_mut())
            && let Some(peak) = startup_camera_motion(
                &self.config,
                (self.layer.width, self.layer.height),
                source,
                &next,
            )
        {
            controller.observe_startup_motion(peak);
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
            let frame = ReuseFrame {
                still: &analysis.still,
                pre: &analysis.pre,
                key,
                horizon: self.frames_to_next_key(index),
                sample_step: self.reuse_sample_step(),
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
                let controller = self.layer.rate_control.as_mut();
                self.still.plan(term, &frame, controller, (index, flat));
            }
            let band = self.plan_band(index, key, &references, analysis, motion);
            let reuse = self
                .layer
                .reuse
                .as_ref()
                .filter(|_| reuse_frame)
                .map(ReuseTerm::offsets);
            let offsets = if band {
                Some(self.block_terms.as_slice())
            } else {
                reuse
            };
            if let Some(aq) = self.layer.aq.as_mut() {
                aq.plan(&analysis.pre, motion, offsets);
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
        let mut plan = self.frame_plan(
            index,
            frame_type,
            references,
            analysis.map(|a| &a.pre),
            motion,
            key_reuse_offset,
        );
        self.frame_quantizer = plan.quantizer;
        if key {
            let step = self.sample_step(plan.quantizer);
            if let Some(cyclic) = self.cyclic.as_mut() {
                cyclic.key(step);
            }
        }

        // The coded source: the input itself, or a padded copy of it.
        let mut padding = self.layer.padding.take();
        let coded_source = match padding.as_mut() {
            Some(p) => {
                copy_padded(source, &mut p.source);
                &p.source
            }
            None => source,
        };
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
            self.frame_params(index, &references, role, analysis);
        self.frame_decision_params = decision_params;
        self.apply_backend_role(role);
        if decision_params.noise_skip > 0.
            && let (Some(pre), Some(motion)) = (pre, motion)
        {
            let still = camera_is_still(motion, (self.layer.width, self.layer.height));
            self.layer.noise_map.update(pre, &motion.fine, still);
        }

        self.backend
            .set_noise_sigma(analysis.map_or(0.0, |a| a.noise_sigma));
        // The sweep assumes each frame references the previous one: frames
        // of a layered mini-GOP still in progress when an adaptive structure
        // was held flat (`set_intra_refresh`) code no window and leave the
        // sweep where it is.
        let flat = references.position.layers == 1;
        self.window = match (self.refresh.as_mut(), frame_type) {
            (Some(sweep), FrameType::Inter) if flat => Some(sweep.next_window()),
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
        let first = self.identity_trial(plan).unwrap_or(plan);
        self.frame_quantizer = first.quantizer;
        let mut data = self.code_picture(&first, pictures, inter.as_ref(), &planner_params);
        let mut codings = 1;
        while let Some(recode) = self.recode_plan(plan, bit_count(&data), codings) {
            codings += 1;
            // The frame would overflow the buffer: code it again,
            // coarser (`RateController::recode_quantizer`).
            plan = recode;
            self.frame_quantizer = plan.quantizer;
            self.block_offsets.clear();
            let pictures = FramePictures {
                source: coded_source,
                reference,
                secondary,
                recon: &mut *recon,
            };
            data = self.code_picture(&plan, pictures, inter.as_ref(), &planner_params);
        }
        self.backend.commit_frame();
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
        if frame_type == FrameType::Inter {
            self.gather_resolution_evidence(stream_input, coded_source, recon);
        }
        self.layer.analyzer = analyzer;
        self.layer.input = own_input;
        pool.finish(references.refresh, &self.dsp);
        if let Some(p) = padding.as_mut() {
            copy_visible(&self.dsp, pool.last(), &mut p.visible_recon);
        }
        self.layer.references = Some(pool);
        self.layer.padding = padding;
        if frame_type == FrameType::Inter {
            self.adapt_coded_resolution();
        }
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

    #[allow(clippy::too_many_arguments)]
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
        if key || self.is_high_quality(index, position) {
            self.layer.last_quantizer = quantizer;
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
        let steering = self.inframe.as_ref().is_some_and(InFrameControl::steering);
        if steering {
            self.report_inframe_offset();
        }
        if let Some(controller) = self.layer.rate_control.as_mut() {
            let planned_without_offsets = self.still.without_offsets;
            if let Some(aq) = self
                .layer
                .aq
                .as_ref()
                .filter(|aq| aq.active() && !steering && !planned_without_offsets)
            {
                let offsets = self.block_offsets.iter().copied();
                let realised = controller.effective_block_offset(aq.total_weight(), offsets);
                controller.set_realised_block_offset(realised);
            }
            controller.feedback(bit_count(data));
            let area = (self.layer.width * self.layer.height).max(1);
            controller.observe_noise_skip(self.layer.noise_skipped_area as f64 / area as f64);
            if controller.wants_distortion() {
                let (width, height) = (self.layer.coded_width, self.layer.coded_height);
                controller.observe_distortion(luma_mse(
                    &self.dsp,
                    coded_source,
                    recon,
                    width,
                    height,
                ));
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

    /// The plan to code the current frame at once more, after `codings`
    /// codings, if any: the real `plan` after an `identity_trial`, or a
    /// coarser one when rate control finds that its `bits` would overflow
    /// the buffer (`RateController::recode_quantizer` bounds how often).
    /// Only backends that honour the recode contract are re-coded, and
    /// never under random decisions (whose generator has moved on).
    fn recode_plan(&mut self, plan: FramePlan, bits: u64, codings: u32) -> Option<FramePlan> {
        if !B::RECODE || self.random.is_some() {
            return None;
        }
        if self.recode_identity && codings == 1 {
            return Some(plan);
        }
        if self.inframe.as_ref().is_some_and(InFrameControl::steering) {
            self.report_inframe_offset();
        }
        let planned = self.layer.rate_control.as_mut()?.recode_quantizer(bits)?;
        let (quantizer, lambda_scale) = self.coded_quantizer(planned);
        Some(FramePlan {
            quantizer,
            lambda_scale,
            ..plan
        })
    }

    /// Tell rate control how much coarser the rows in-frame control
    /// coarsened made the coding just finished, before it learns from it.
    fn report_inframe_offset(&mut self) {
        let (Some(controller), Some(inframe)) =
            (self.layer.rate_control.as_mut(), self.inframe.as_ref())
        else {
            return;
        };
        let (total, offsets) = inframe.realised();
        controller.set_inframe_offset(controller.effective_block_offset(total, offsets));
    }

    /// Code one picture from `begin_frame` to `end_frame` (the partition
    /// walk, or a picture-owning backend's own pipeline), writing its
    /// reconstruction into `pictures.recon`; returns the unit, not yet
    /// committed (`CodecBackend::commit_frame`).
    fn code_picture(
        &mut self,
        plan: &FramePlan,
        pictures: FramePictures<'_>,
        inter: Option<&FrameInputs<'_>>,
        planner_params: &PlannerParams,
    ) -> Vec<u8> {
        let FramePictures {
            source: coded_source,
            reference,
            secondary,
            recon,
        } = pictures;
        self.backend.begin_frame(plan);
        let coded_size = (self.layer.coded_width, self.layer.coded_height);
        if let Some(screen) = self.layer.second.as_mut().and_then(|s| s.screen.as_mut()) {
            screen.begin_picture(plan.index, coded_size);
        }
        let steering = self.inframe.as_ref().is_some_and(InFrameControl::steering);
        if steering && let Some(inframe) = self.inframe.as_mut() {
            inframe.begin_frame(self.backend.coded_bits().unwrap_or(0));
        }
        if B::OWN_PICTURE_PIPELINE {
            self.backend.encode_picture(
                plan,
                coded_source,
                (plan.frame_type == FrameType::Inter).then_some(reference),
                recon,
            )
        } else {
            let sb_log2 = self.backend.superblock_log2();
            let sb = 1usize << sb_log2;
            for y in (0..self.layer.coded_height).step_by(sb) {
                for x in (0..self.layer.coded_width).step_by(sb) {
                    self.backend.begin_superblock(x, y);
                    let sb = Block::square(x, y, sb_log2);
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
                        leaf_bits: OnceCell::new(),
                    };
                    let key_mode =
                        KeyPlanMode::resolve(self.planner_params.intra_plan, &self.backend);
                    let plan = match inter {
                        // Random decisions choose every split themselves.
                        _ if self.random.is_some() => None,
                        Some(frame) if splittable => {
                            Some(planner(frame.motion, frame.pre).plan(sb))
                        }
                        None if splittable && key_mode != KeyPlanMode::Fixed => Some(
                            KeyPlanner {
                                dsp: &self.dsp,
                                backend: &mut self.backend,
                                source: coded_source,
                                width: self.layer.coded_width,
                                height: self.layer.coded_height,
                                params: planner_params,
                                mode: key_mode,
                            }
                            .plan(sb),
                        ),
                        _ => None,
                    };
                    let pictures = FramePictures {
                        source: coded_source,
                        reference,
                        secondary,
                        recon: &mut *recon,
                    };
                    self.code_node(sb, plan.as_ref(), pictures, inter);
                }
                if steering && let Some(bits) = self.backend.coded_bits() {
                    self.inframe
                        .as_mut()
                        .expect("steering implies in-frame control")
                        .end_row(bits);
                }
            }
            self.backend.filter_frame(coded_source, recon);
            self.backend.end_frame()
        }
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

    /// `references` of frame `index` (a key frame if `key`, with the
    /// analysis' still-block fraction if known) with the second reference,
    /// when it is on (`second_reference.rs`).
    fn with_second(
        &self,
        index: u64,
        references: FrameReferences,
        (key, still_fraction): (bool, Option<f32>),
    ) -> FrameReferences {
        match self.layer.second.as_ref() {
            Some(second) => {
                let high_quality = key || self.is_high_quality(index, references.position);
                second.apply(index, (high_quality, still_fraction), references)
            }
            None => references,
        }
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
        hold.holds(repeat)
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
                straddles
                    || should_split(
                        node,
                        self.config.block_log2.min(self.backend.max_leaf_log2()),
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
    ) -> Option<(PredictionMode, ReferenceChoice)> {
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
            return Some(mode);
        }
        // The static skip proves a zero residual at the frame quantizer; a
        // block coded finer than the frame (`refined`) may well code one.
        let band = refined
            && self
                .cyclic
                .as_ref()
                .is_some_and(|c| c.in_band((node.x, node.y)));
        let skip_allowed = !refined
            && refresh.is_none_or(|w| {
                self.backend
                    .static_skip_vector(node)
                    .is_some_and(|mv| w.allows(node, mv))
            });
        let mode = if skip_allowed && self.try_static_skip(node, frame, (source, reference), recon)
        {
            let skip = self
                .backend
                .static_skip_vector(node)
                .expect("a backend accepting static skips has a skip vector");
            self.code_static_skip(node, skip, (source, reference), recon);
            None
        } else if band && refresh.is_none_or(|w| w.allows(node, Mv::ZERO)) {
            // A square of the cyclic band: static for several frames and
            // re-coded only to refine it, so the zero vector with no search
            // (what the references' cyclic refresh costs). Instructions,
            // x86 callgrind, static_cam22 300 f 150 kbps: see
            // `docs/analysis/within-frame-rate-control-2026-11-14.md`.
            Some((PredictionMode::Inter(Mv::ZERO), ReferenceChoice::Primary))
        } else if self.frame_decision_params.static_zero_vector >= 0.5
            && frame.still.is_some_and(|still| is_still::<B>(still, node))
            && refresh.is_none_or(|w| w.allows(node, Mv::ZERO))
        {
            // A reliably still block (static for several frames, away from
            // motion; `still.*`): the zero vector, with no candidate or
            // intra trials.
            Some((PredictionMode::Inter(Mv::ZERO), ReferenceChoice::Primary))
        } else {
            let skip_vector = self.backend.skip_vector(node);
            let search_mv =
                planned.map_or_else(|| search_vector::<B>(frame.motion, node), |p| p.search_mv);
            // The pre-analysis moments give the source block's energy
            // exactly, saving a variance pass, where its 8x8 cells hold only
            // picture samples (not replicated border).
            let (width, height) = leaf_extent::<B>(node);
            let inside = node.x + width <= self.layer.width && node.y + height <= self.layer.height;
            let source_moments = frame.pre.filter(|_| inside);
            let primary = choose_inter_mode(
                &self.dsp,
                &mut self.backend,
                node,
                source,
                reference,
                search_mv,
                planned,
                &mut self.decision_scratch,
                skip_vector,
                source_moments,
                &self.frame_decision_params,
                refresh,
            );
            let (choice, chosen) = self.with_secondary(
                node,
                source,
                reference,
                secondary.zip(frame.secondary_ratio),
                (search_mv, primary),
                source_moments,
                refresh,
            );
            if chosen != ReferenceChoice::Primary
                && let Some(screen) = self.layer.second.as_mut().and_then(|s| s.screen.as_mut())
            {
                screen.record_win((node.x, node.y), leaf_extent::<B>(node));
            }
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
            if let Some(mv) = choice.mode.intra_fallback() {
                self.backend.intra_fallback(node, mv);
            }
            Some((choice.mode.prediction(), chosen))
        };
        self.window = window;
        mode
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
        let random = self.random.as_mut().expect("random mode checked");
        let mut chosen = if allow_compound && random.compound_reference() {
            ReferenceChoice::Compound {
                secondary_mv: Mv::ZERO,
            }
        } else if allow_secondary && random.secondary_reference() {
            ReferenceChoice::Secondary
        } else {
            ReferenceChoice::Primary
        };
        if chosen == ReferenceChoice::Secondary {
            self.backend.select_reference(node, chosen);
        }
        let mode = random_inter_leaf(
            &mut self.backend,
            random,
            node,
            self.decision_scratch.candidates_mut(),
            refresh,
        );
        if let ReferenceChoice::Compound { .. } = chosen {
            chosen = if matches!(mode, PredictionMode::IntraDc) {
                ReferenceChoice::Primary
            } else {
                let candidate = self.random.as_mut().expect("random mode checked").vector();
                let secondary_mv = if refresh.is_none_or(|w| w.allows(node, candidate)) {
                    candidate
                } else {
                    Mv::ZERO
                };
                ReferenceChoice::Compound { secondary_mv }
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
    #[allow(clippy::too_many_arguments)]
    fn with_secondary(
        &mut self,
        node: &Block,
        source: &Picture,
        primary_reference: &Picture,
        secondary: Option<(&Picture, f32)>,
        (search_mv, primary): (Mv, Choice),
        source_moments: Option<&PreAnalysis>,
        refresh: Option<&RefreshWindow>,
    ) -> (Choice, ReferenceChoice) {
        let Some((secondary, ratio)) = secondary else {
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
        if (!primary.mode.is_inter() && ratio > 0.0)
            || !self.secondary_worth_trying(node, primary.cost)
        {
            return (primary, ReferenceChoice::Primary);
        }
        let seed = scaled_vector(search_mv, ratio);
        if !self.passes_secondary_screen(
            node,
            (source, primary_reference, secondary),
            seed,
            primary,
        ) {
            return (primary, ReferenceChoice::Primary);
        }
        let include_dc = !primary.mode.is_inter();
        let Some(second) = self.secondary_choice(
            node,
            source,
            secondary,
            (seed, primary.cost, include_dc),
            source_moments,
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
                    (&self.frame_decision_params, source_moments),
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

    /// Whether leaf `node`, whose primary choice cost `primary_cost`, tries
    /// the second reference (`refs.second_gate_lambdas`): a block the
    /// primary reference already predicts within the gate has little left
    /// for a finer reference to win. A cached planner choice (`-inf`) was
    /// accepted as good without ranking.
    fn secondary_worth_trying(&self, node: &Block, primary_cost: f32) -> bool {
        let gate = self.layer.second.as_ref().map_or(0.0, |s| s.gate_lambdas);
        let lambda = self.backend.lambda() * self.frame_decision_params.lambda_scale;
        primary_cost > gate * lambda * leaf_area::<B>(node) as f32
            && primary_cost > f32::NEG_INFINITY
    }

    /// Whether the second-reference screen (`second_screen.rs`), when on,
    /// lets leaf `node` try the `secondary` reference: from the whole-sample
    /// SADs of the secondary at the zero vector and `seed` against the
    /// primary reference's at the `primary` choice's vector, and that
    /// choice's cost. An intra primary choice, or a vector whose window
    /// leaves the padded picture, is not screened.
    fn passes_secondary_screen(
        &self,
        node: &Block,
        (source, primary_reference, secondary): (&Picture, &Picture, &Picture),
        seed: Mv,
        primary: Choice,
    ) -> bool {
        let Some(screen) = self.layer.second.as_ref().and_then(|s| s.screen.as_ref()) else {
            return true;
        };
        let PredictionMode::Inter(primary_mv) = primary.mode.prediction() else {
            return true;
        };
        let sads = BlockSads::new::<B>(&self.dsp, source, node);
        let primary_sad = sads.at(primary_reference, primary_mv);
        let zero_sad = sads.at(secondary, Mv::ZERO);
        let seed_sad = if seed.full_pel() == (0, 0) {
            None
        } else {
            sads.at(secondary, seed)
        };
        // The better of the two (`Option::min` would rank `None` lowest).
        let secondary_sad = match (zero_sad, seed_sad) {
            (Some(zero), Some(seed)) => Some(zero.min(seed)),
            (zero, seed) => zero.or(seed),
        };
        let (Some(primary_sad), Some(secondary_sad)) = (primary_sad, secondary_sad) else {
            return true;
        };
        let lambda = self.backend.lambda() * self.frame_decision_params.lambda_scale;
        let area = leaf_area::<B>(node);
        screen.worth_trying(
            ((node.x, node.y), leaf_extent::<B>(node)),
            (secondary_sad, primary_sad),
            primary.cost / (lambda * area as f32),
        )
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
        source_moments: Option<&PreAnalysis>,
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
                (&self.frame_decision_params, source_moments),
                refresh,
            );
        }
        let skip_vector = self.backend.skip_vector(node);
        let choice = choose_inter_mode(
            &self.dsp,
            &mut self.backend,
            node,
            source,
            secondary,
            seed,
            None,
            &mut self.decision_scratch,
            skip_vector,
            source_moments,
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
        let qp_offset = if inter.is_some() && self.band_within_noise(&node, (source, reference)) {
            0
        } else {
            self.leaf_qp_offset(&node)
        };
        let (mode, chosen, selected_ref) = match inter {
            None => (PredictionMode::IntraDc, ReferenceChoice::Primary, None),
            Some(_) if region == RefreshRegion::Intra => (
                PredictionMode::IntraDc,
                ReferenceChoice::Primary,
                Some(reference),
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
                    self.observe_spatial_leaf_choice(leaf_area::<B>(&node), decided);
                }
                self.frame_decision_params.lambda_scale = frame_lambda_scale;
                let Some((mode, chosen)) = decided else {
                    // A static skip: coded at the frame quantizer.
                    return;
                };
                let picture = match chosen {
                    ReferenceChoice::Primary | ReferenceChoice::Compound { .. } => reference,
                    ReferenceChoice::Secondary => {
                        secondary.expect("only frames with a secondary reference choose it")
                    }
                };
                (mode, chosen, Some(picture))
            }
        };
        // Rate control's realised offset needs only the leaves coded off the
        // frame quantizer (the others' ratio is 1; the frame's total weight
        // covers them): on a frame where the reuse term refines a few cells,
        // a few leaves. A frame in-frame control steers keeps its planned
        // block offset and reports the rows it coarsened instead
        // (`Encoder::report_inframe_offset`).
        let steering = self.inframe.as_ref().is_some_and(InFrameControl::steering);
        if let Some(aq) = self.active_aq().filter(|_| qp_offset != 0 && !steering) {
            let frame_step = self.quantizer_steps[self.frame_quantizer as usize];
            let coded = (i64::from(self.frame_quantizer) + i64::from(qp_offset)) as usize;
            let offset = QP_UNITS_PER_OCTAVE * (self.quantizer_steps[coded] / frame_step).log2();
            self.block_offsets.push((aq.block_weight(&node), offset));
        }
        let decision = BlockDecision {
            mode,
            reference: chosen,
            qp_offset,
        };
        if let ReferenceChoice::Compound { .. } = chosen {
            self.backend.encode_compound_block(
                &node,
                &decision,
                source,
                reference,
                secondary.expect("only frames with a secondary reference choose compound"),
                recon,
            );
        } else {
            self.backend
                .encode_block(&node, &decision, source, selected_ref, recon);
        }
        let reuse_frame = self.reuse_frame.is_some();
        let copy = chosen == ReferenceChoice::Primary && self.backend.copied_reference(&node);
        let step = self
            .sample_step((i64::from(self.frame_quantizer) + i64::from(qp_offset)).max(0) as u32);
        if let Some(term) = self
            .layer
            .reuse
            .as_mut()
            .filter(|_| inter.is_some() && reuse_frame)
        {
            term.coded_block((node.x, node.y), leaf_extent::<B>(&node), (copy, step));
        }
        if !copy && let Some(cyclic) = self.cyclic.as_mut() {
            cyclic.coded((node.x, node.y), step);
        }
    }

    /// Whether `node` lies in the cyclic band but its zero-vector residual
    /// is already within the camera's noise (`noise_skip.rs`, while on):
    /// refining it would code noise, so it is coded at the frame quantizer
    /// and its square is settled (not offered again until coded coarser).
    /// Without this, AV1 (whose backend often declines the noise skip on a
    /// block planned finer) refined noise: synthetic noisy still at sigma
    /// 2.9, band + noise skip 390 kbps / 37.33 dB against the clean scene
    /// vs noise skip alone 265 kbps / 37.94 dB.
    fn band_within_noise(&mut self, node: &Block, pictures: (&Picture, &Picture)) -> bool {
        let params = &self.frame_decision_params;
        if params.noise_skip <= 0.
            || !self
                .cyclic
                .as_ref()
                .is_some_and(|c| c.in_band((node.x, node.y)))
        {
            return false;
        }
        let within = self.layer.noise_map.explains(
            &self.dsp,
            node,
            pictures,
            params.noise_skip,
            params.noise_skip_chroma_ratio,
            self.sample_step(self.frame_quantizer) as f32,
        );
        if within && let Some(cyclic) = self.cyclic.as_mut() {
            cyclic.settled((node.x, node.y));
        }
        within
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
            let inframe = self
                .inframe
                .as_ref()
                .filter(|c| c.steering())
                .map_or(0., InFrameControl::offset_qp);
            let aq = self.active_aq().map(|aq| aq.offset_qp(node));
            if aq.is_none() && inframe == 0. {
                return 0;
            }
            let codec = self.config.codec;
            native_offset(
                &self.quantizer_steps,
                (min_quantizer(codec), max_quantizer(codec)),
                self.frame_quantizer,
                aq.unwrap_or(0.) + inframe,
            )
        });
        if qp_offset == 0 {
            0
        } else {
            self.backend.block_qp_offset(qp_offset)
        }
    }
}

impl<B: CodecBackend> Encoder<B> {
    /// Offer `node` to the backend's static skip when pre-analysis marks it
    /// static, or when the noise skip (`noise_skip.rs`, while on) finds its
    /// zero-vector residual within the camera's noise.
    fn try_static_skip(
        &mut self,
        node: &Block,
        frame: &FrameInputs<'_>,
        (source, reference): (&Picture, &Picture),
        recon: &mut Picture,
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
        if !within_noise && !frame.pre.is_some_and(|pre| is_static::<B>(pre, node)) {
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
    refresh: Option<&RefreshWindow>,
) -> PredictionMode {
    candidates.clear();
    candidates.push(Mv::ZERO);
    backend.mv_candidates(node, candidates);
    let mode = match random.inter_mode(candidates, |mv| refresh.is_none_or(|w| w.allows(node, mv)))
    {
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
fn on_reuse_chain(key: bool, references: &FrameReferences) -> bool {
    key || references.position.layer == 0
}

/// The prefix of the temporal-structure parameters (`LayerParams`).
const LAYER_PARAM_PREFIX: &str = "layers.";

/// The backend's `DEFAULT_PARAMS` that apply to `config`. The backend's
/// temporal-structure defaults (H.264 Default's gated L1T3) step aside when
/// the caller sets the structure itself (a layer count above 1: an explicit
/// `--temporal-layers` keeps the engine's own layer settings) or asks for a
/// feature that needs a flat stream (the second reference, `refs.second`).
fn backend_defaults<B: CodecBackend>(config: &EncoderConfig) -> Vec<(&'static str, f32)> {
    let own_structure = config.temporal_layers > 1 || second_reference_enabled(&config.params);
    let spatial = SpatialParams::from_overrides(&config.params).layers.round() as u8 > 1;
    B::DEFAULT_PARAMS
        .iter()
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
    let global = &motion.global;
    let translation = global.tx.hypot(global.ty);
    let zoom_at_edge = global.zoom.abs() * width as f32 / 2.0;
    let adjacent = distance == 1;
    MotionSignal {
        camera_motion: (translation + zoom_at_edge) / distance.max(1) as f32,
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
        source: Picture::new(coded_width, coded_height, 0)?,
        visible_recon: Picture::new(config.width, config.height, 0)?,
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
    if RefreshParams::from_overrides(&config.params).period > 0.0 {
        return Err(ConfigError::LongTermReference("not with intra refresh"));
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
    // Layered structures would also need the HQ frame placed after an
    // older short-term reference in every codec's list; not tested yet.
    if config.temporal_layers > 1 {
        return Err(ConfigError::SecondReference("not with temporal layers yet"));
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
    Ok(Some(SecondReference::new(
        slot,
        &RefParams::from_overrides(&config.params),
    )))
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
) -> Result<Option<EffortOverlay>, ConfigError> {
    let (_, ordinary) = params.split_prefix(NONBASE_PREFIX);
    ordinary
        .validate(&[engine_specs(), B::PARAM_SPECS.to_vec()].concat())
        .map_err(ConfigError::Param)?;
    EffortOverlay::from_overrides(params, B::FRAME_PARAM_SPECS, B::STREAM_FIXED_PARAMS)
        .map_err(ConfigError::Param)
}

/// The QP cadence of `config`, scaled for `backend` (`RateModel::cadence_scale`).
/// Flattened when the cyclic spatial cadence gives the boost to blocks
/// (`rate_control::spatial_aware`).
fn configured_cadence<B: CodecBackend>(config: &EncoderConfig, backend: &B) -> QpCadence {
    let params = RateParams::from_overrides(&config.params);
    spatial_aware(unflattened_cadence(config, backend), &params)
}

/// The configured QP cadence, scaled for `backend`, whichever of frames or
/// blocks it applies to.
fn unflattened_cadence<B: CodecBackend>(config: &EncoderConfig, backend: &B) -> QpCadence {
    let params = RateParams::from_overrides(&config.params);
    QpCadence::from_params(&params).scaled(cadence_scale(&params, &backend.rate_model()))
}

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
