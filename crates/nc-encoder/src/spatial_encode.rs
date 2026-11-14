//! Multi-layer spatial SVC execution loop (`spatial.layers == 2` or `3`) for
//! [`Encoder`].
//!
//! When `spatial.layers == 1`, [`Encoder::spatial`] is `None` and the single-layer
//! frame loop in `encoder.rs` runs untouched (byte-identical across all codecs).
//! When `spatial.layers > 1`, each input frame is encoded as one multi-layer
//! temporal unit (`S0` lowest spatial layer first, then `S1`, and in 3-layer
//! mode `S2` full-size top enhancement). Each layer has its own [`LayerState`]:
//! the active one is [`Encoder::layer`], the inactive ones live in
//! [`SpatialSession::parked`] indexed by `spatial_id`, and [`SpatialSession::activate`]
//! swaps the whole [`LayerState`] on layer switch so all layers share the exact
//! partition planner, leaf mode decision, AQ, reuse term and recode machinery.

use nc_analysis::link::FrameLink;
use nc_analysis::{Analyzer, NoTimer};
use nc_common::Picture;

use crate::aq::BlockAq;
use crate::backend::{
    CodecBackend, FramePlan, FrameType, PredictionMode, ReferenceChoice, SpatialLayerFrame,
};
use crate::config::{ConfigError, EncoderConfig, FrameRate};
use crate::encoder::{
    EncodedFrame, Encoder, FrameInputs, FrameMetadata, FramePictures, LayerState, Padding,
    bit_count, check_dimensions, configured_cadence, copy_padded, copy_visible, frame_buffers,
    max_quantizer, motion_signal, on_reuse_chain, source_analyzer,
};
use crate::feedback::ReferencedFrames;
use crate::hold::LastCoded;
use crate::params::{LayerParams, RateParams, RefParams};
use crate::rate_control::{
    LayerShare, MAX_RECODES, RECODE_FIRST_EXPONENT, RateControlConfig, RateControlError,
    RateControlStats, RateController, RateModel, quantizer_for_step,
};
use crate::reuse::{ReuseFrame, ReuseTerm};
use crate::second_reference::SecondReference;
use crate::source_scale::PictureResampler;
use crate::spatial::{
    BASE_SLOT_OFFSET, CombinedCbrBuffer, EnhancementClock, EnhancementIntra, InterLayerMode,
    LayerDropFacts, MAX_SPATIAL_LAYERS, ShareRequest, SpatialConfig, SpatialLayerPlan,
    SpatialStructure, SpatialUnitDrop, TemporalUnitPlan, decide_unit_drop_layers,
    layer_slot_offset, layer_temporal_slots, spatial_pool_slots, split_rate_control,
};
use crate::startup_lookahead::startup_scene_profile;
use crate::timeline::FrameTime;

/// Metadata for one coded spatial layer picture inside [`EncodedFrame::layers`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerFrameMetadata {
    /// Spatial layer (`0` = base, `1` = first enhancement, `2` = top enhancement; `AV1 §5.3.3`).
    pub spatial_id: u8,
    /// Temporal layer (`0..=2`; `AV1 §5.3.3`).
    pub temporal_layer: u8,
    /// Monotone index among coded layer pictures across the stream.
    pub picture_index: u64,
    /// Engine frame type (`Key` on `S0` keyframes and `S1`/`S2` `INTRA_ONLY_FRAME`s; `Inter` otherwise).
    pub frame_type: FrameType,
    /// True when `spatial_id >= 1` is an enhancement-only repair picture.
    pub enhancement_repair: bool,
    /// Visible width of this layer's picture.
    pub width: usize,
    /// Visible height of this layer's picture.
    pub height: usize,
    /// Start offset of this layer's OBU slice in [`EncodedFrame::data`].
    pub byte_offset: usize,
    /// Byte length of this layer's OBU slice in [`EncodedFrame::data`].
    pub byte_len: usize,
    /// Actual frame quantizer in the codec's native scale.
    pub quantizer: u32,
    /// Earlier `picture_index` values this layer picture predicts from.
    pub references: ReferencedFrames,
    /// True when `spatial_id >= 1` predicts from `spatial_id - 1` in this
    /// temporal unit (`AV1 §7.18` / `VP9` inter-layer dependency; true on all
    /// units in `InterLayerMode::On` and on key units in `InterLayerMode::KeyOnly`,
    /// false on `S0`, non-key units in `KeyOnly`, `INTRA_ONLY` repairs, and `Off`).
    pub inter_layer_ref: bool,
    /// True when this layer picture can be dropped relative to base operating
    /// points (true on enhancement layers `spatial_id >= 1`, and on non-reference
    /// top temporal layer `S0` pictures).
    pub droppable: bool,
    /// Bitmask over [`SpatialConfig::operating_points`] (`0..operating_points.len()`,
    /// up to 9 operating points in `L3T3`): bit `i` is `1` when operating point
    /// `i` includes this layer picture, and `0` when operating point `i` drops it.
    pub operating_point_mask: u32,
}

impl LayerFrameMetadata {
    /// Whether this layer picture is an enhancement-layer `INTRA_ONLY_FRAME`
    /// (`AV1 §5.9.2` / `§6.8.2`: `spatial_id >= 1` and `frame_type == Key`).
    #[must_use]
    pub const fn is_intra_only(&self) -> bool {
        self.spatial_id > 0 && matches!(self.frame_type, FrameType::Key)
    }

    /// Whether operating point `op_index` (`0` = highest quality) includes this
    /// layer picture.
    #[must_use]
    pub const fn included_in_operating_point(&self, op_index: u8) -> bool {
        op_index < 32 && (self.operating_point_mask & (1u32 << op_index)) != 0
    }

    /// Whether operating point `op_index` (`0` = highest quality) drops this
    /// layer picture.
    #[must_use]
    pub const fn droppable_at_operating_point(&self, op_index: u8) -> bool {
        !self.included_in_operating_point(op_index)
    }
}

fn operating_point_mask(config: &SpatialConfig, spatial_id: u8, temporal_id: u8) -> u32 {
    let mut mask = 0u32;
    for (idx, op) in config.operating_points.iter().enumerate().take(32) {
        if spatial_id <= op.max_spatial_id && temporal_id <= op.max_temporal_id {
            mask |= 1u32 << idx;
        }
    }
    mask
}

/// Cumulative reference-selection tally on dual-reference enhancement inter
/// frames (where both the layer's temporal reference and the same-instant
/// lower-layer reference are available; §2.6, §4).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpatialReferenceStats {
    /// Leaf blocks coded from the temporal enhancement reference (`Primary`).
    pub temporal_blocks: u64,
    /// Leaf blocks coded from the same-instant lower-layer reference (`Secondary`).
    pub inter_layer_blocks: u64,
    /// Leaf blocks coded intra on dual-reference enhancement frames.
    pub intra_blocks: u64,
    /// Luma sample area coded from the temporal enhancement reference.
    pub temporal_area: u64,
    /// Luma sample area coded from the same-instant lower-layer reference.
    pub inter_layer_area: u64,
    /// Luma sample area coded intra on dual-reference enhancement frames.
    pub intra_area: u64,
}

impl SpatialReferenceStats {
    fn add(self, other: Self) -> Self {
        Self {
            temporal_blocks: self.temporal_blocks + other.temporal_blocks,
            inter_layer_blocks: self.inter_layer_blocks + other.inter_layer_blocks,
            intra_blocks: self.intra_blocks + other.intra_blocks,
            temporal_area: self.temporal_area + other.temporal_area,
            inter_layer_area: self.inter_layer_area + other.inter_layer_area,
            intra_area: self.intra_area + other.intra_area,
        }
    }
}

/// Multi-layer spatial scalability session owned by [`Encoder`] when
/// `spatial.layers > 1`.
#[derive(Debug)]
pub(super) struct SpatialSession {
    pub(super) config: SpatialConfig,
    pub(super) structure: SpatialStructure,
    /// Inactive layers' states indexed by `spatial_id` (`None` at `active_spatial_id`,
    /// which lives in [`Encoder::layer`]). Boxed so [`Encoder`] stays compact.
    parked: Box<[Option<LayerState>]>,
    /// `spatial_id` currently swapped into [`Encoder::layer`].
    active_spatial_id: u8,
    /// Each layer's visible reconstruction of its last coded picture, lent
    /// to the layer while it codes its next one.
    visible_recon: [Option<Picture>; MAX_SPATIAL_LAYERS],
    full_source: Option<Picture>,
    /// Source downscalers for each adjacent step `s + 1 -> s` (`0..num_layers - 1`).
    down_resamplers: Vec<PictureResampler>,
    /// Inter-layer proxy upscalers for each adjacent step `s -> s + 1` (`0..num_layers - 1`).
    up_resamplers: Vec<PictureResampler>,
    pub(super) combined_cbr: Option<CombinedCbrBuffer>,
    pub(super) rate_config: Option<RateControlConfig>,
    nominal_tu_dt: f64,
    /// Time since the last coded or rate-control-dropped `S1` picture.
    s1_clock: EnhancementClock,
    /// Highest `spatial_id` coded in the most recent temporal unit (`0..num_layers`).
    pub(super) output_spatial_id: u8,
    /// Enhancement step (`0` = `S1`, `1` = `S2`) currently coding a dual-reference
    /// inter frame, if any.
    pub(super) coding_inter_layer_step: Option<usize>,
    /// Whether the layer currently being planned/coded has a higher coded layer
    /// above it in the same temporal unit.
    current_layer_has_coded_upper: bool,
    /// Bits already emitted by lower spatial layers `< active_spatial_id` in this TU.
    current_tu_spent_bits: u64,
    /// Most recent coded quantizer of each spatial layer (`0..MAX_SPATIAL_LAYERS`).
    last_layer_quantizer: [u32; MAX_SPATIAL_LAYERS],
    /// Unspent bits reallocated upward to the layer currently being planned.
    current_tu_reallocated_bits: f64,
    /// Unspent `S0` bits reserved directly for `S2` when `S1` and `S2` both code.
    current_tu_carry_to_s2: f64,
    current_tu_interval: Option<f64>,
    /// Start-up `(full_res_peak_displacement, static_fraction)` measured once on
    /// `TU 0` from `S0`'s lookahead pair.
    startup_profile: Option<(f64, f32)>,
    recode_ceiling: f64,
    recode_ceiling_inter: f64,
    pub(super) step_ref_stats: [SpatialReferenceStats; MAX_SPATIAL_LAYERS - 1],
}

/// A spatial layer's engine state at `(w, h)`, and its visible
/// reconstruction buffer.
fn spatial_layer_state<B: CodecBackend>(
    config: &EncoderConfig,
    backend: &B,
    (w, h): (usize, usize),
    pool_slots: usize,
    temporal_layers: u8,
    reads_repeats: bool,
    second: Option<SecondReference>,
    (use_aq, use_reuse): (bool, bool),
) -> Result<(LayerState, Picture), ConfigError> {
    let multiple = 1usize << backend.min_block_log2();
    let bad_dims = |_| ConfigError::Dimensions {
        width: config.width,
        height: config.height,
        multiple,
    };
    let layer_cfg = EncoderConfig {
        width: w,
        height: h,
        temporal_layers,
        ..config.clone()
    };
    let coded = backend.coded_size(w, h);
    check_dimensions(&layer_cfg, coded, multiple)?;
    let (references, padding) = frame_buffers(&layer_cfg, coded, pool_slots).map_err(bad_dims)?;
    let analyzer = backend
        .supports_inter()
        .then(|| source_analyzer(&layer_cfg, pool_slots, reads_repeats))
        .flatten();
    let input = analyzer
        .is_none()
        .then(|| Picture::new(w, h, 0))
        .transpose()
        .map_err(bad_dims)?;
    let aq = use_aq
        .then(|| BlockAq::new(&config.params, B::BLOCK_QP_LOG2, B::BLOCK_QP_OFFSETS))
        .flatten();
    let reuse = use_reuse.then(|| ReuseTerm::new(B::SUPPORTS_BLOCK_QP));
    let last_visible_recon = Picture::new(w, h, 0).map_err(bad_dims)?;
    let layer_params = LayerParams::from_overrides(&config.params);
    let state = LayerState {
        width: w,
        height: h,
        coded_width: coded.0,
        coded_height: coded.1,
        references: Some(references),
        padding,
        analyzer,
        input,
        rate_control: None,
        second,
        aq,
        reuse,
        reuse_last_frame: None,
        ladder: layer_params.fixed_ladder(temporal_layers),
        rate_control_ladder: layer_params.rate_control_ladder(temporal_layers),
        cadence: configured_cadence(&layer_cfg, backend),
        last_key_index: 0,
        last_quantizer: 0,
    };
    Ok((state, last_visible_recon))
}

fn build_step_resamplers(
    spatial: &SpatialConfig,
) -> (Vec<PictureResampler>, Vec<PictureResampler>) {
    let steps = usize::from(spatial.layers).saturating_sub(1);
    let mut down_resamplers = Vec::with_capacity(steps);
    let mut up_resamplers = Vec::with_capacity(steps);
    for step in 0..steps {
        let lower = spatial.layer_sizes[step];
        let upper = spatial.layer_sizes[step + 1];
        let exact_2x = (lower.0 * 2, lower.1 * 2) == upper;
        down_resamplers.push(if spatial.base_box_downscale && exact_2x {
            PictureResampler::box_downscale_2x(upper, lower)
        } else {
            PictureResampler::new(upper, lower)
        });
        up_resamplers.push(if spatial.proxy_bilinear && exact_2x {
            PictureResampler::bilinear_upscale_2x(lower, upper)
        } else {
            PictureResampler::new(lower, upper)
        });
    }
    (down_resamplers, up_resamplers)
}

#[derive(Clone, Copy)]
struct UpperLayerDone {
    sid: u8,
    step: usize,
    temporal_id: u8,
    has_coded_upper: bool,
    layer_reset: bool,
    prev_ref_stats: SpatialReferenceStats,
    nominal_inter_budget: f64,
    incoming_reallocated: f64,
    layer_bits: u64,
    layer_quantizer: u32,
}

impl SpatialSession {
    pub(super) fn new<B: CodecBackend>(
        config: &EncoderConfig,
        backend: &B,
        spatial: SpatialConfig,
        flags: (bool, bool),
    ) -> Result<(Self, LayerState), ConfigError> {
        let num_layers = usize::from(spatial.layers);
        let pool_slots = spatial_pool_slots(spatial.layers, spatial.inter_layer);
        let multiple = 1usize << backend.min_block_log2();
        let bad_dims = |_| ConfigError::Dimensions {
            width: config.width,
            height: config.height,
            multiple,
        };
        let adaptive_rule = LayerParams::from_overrides(&config.params).adaptive_rule();
        let structure = SpatialStructure::with_layers(
            spatial.layers,
            spatial.temporal_layers,
            spatial.enhancement_interval,
            spatial.inter_layer,
            adaptive_rule,
        );
        let rate_params = RateParams::from_overrides(&config.params);
        let ref_params = RefParams::from_overrides(&config.params);
        let enh_temporal_layers = if spatial.enhancement_interval == 1 {
            spatial.temporal_layers
        } else {
            1
        };

        let mut parked: Box<[Option<LayerState>]> = (0..MAX_SPATIAL_LAYERS)
            .map(|_| None)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut visible_recon: [Option<Picture>; MAX_SPATIAL_LAYERS] = [None, None, None];
        let mut top_state = None;
        for s in 0..num_layers {
            let tl = if s == 0 {
                spatial.temporal_layers
            } else {
                enh_temporal_layers
            };
            let reads_repeats = s == 0 && structure.reads_repeats();
            let second = (s > 0 && spatial.inter_layer.active_on_inter_unit())
                .then(|| SecondReference::new(0, &ref_params));
            let (state, recon) = spatial_layer_state(
                config,
                backend,
                spatial.layer_sizes[s],
                pool_slots,
                tl,
                reads_repeats,
                second,
                flags,
            )?;
            visible_recon[s] = Some(recon);
            if s + 1 == num_layers {
                top_state = Some(state);
            } else {
                parked[s] = Some(state);
            }
        }
        let top_state = top_state.expect("at least 2 spatial layers");

        let full_source = Picture::new(config.width, config.height, 0).map_err(bad_dims)?;
        let (down_resamplers, up_resamplers) = build_step_resamplers(&spatial);
        let nominal_tu_dt =
            f64::from(config.frame_rate.den.max(1)) / f64::from(config.frame_rate.num.max(1));
        let top_spatial_id = spatial.layers - 1;

        let session = Self {
            config: spatial,
            structure,
            parked,
            active_spatial_id: top_spatial_id,
            visible_recon,
            full_source: Some(full_source),
            down_resamplers,
            up_resamplers,
            combined_cbr: None,
            rate_config: None,
            nominal_tu_dt,
            s1_clock: EnhancementClock::default(),
            output_spatial_id: 0,
            coding_inter_layer_step: None,
            current_layer_has_coded_upper: false,
            current_tu_spent_bits: 0,
            last_layer_quantizer: [config.quantizer; MAX_SPATIAL_LAYERS],
            current_tu_reallocated_bits: 0.0,
            current_tu_carry_to_s2: 0.0,
            current_tu_interval: None,
            startup_profile: None,
            recode_ceiling: f64::from(rate_params.recode_overflow),
            recode_ceiling_inter: f64::from(rate_params.recode_overflow_inter),
            step_ref_stats: [SpatialReferenceStats::default(); MAX_SPATIAL_LAYERS - 1],
        };
        Ok((session, top_state))
    }

    /// Layer `spatial_id`'s state, `active` being [`Encoder::layer`].
    pub(super) fn layer<'a>(&'a self, active: &'a LayerState, spatial_id: u8) -> &'a LayerState {
        if spatial_id == self.active_spatial_id {
            active
        } else {
            self.parked[usize::from(spatial_id)]
                .as_ref()
                .expect("inactive spatial layer is parked")
        }
    }

    /// Layer `spatial_id`'s state, mutably, `active` being [`Encoder::layer`].
    pub(super) fn layer_mut<'a>(
        &'a mut self,
        active: &'a mut LayerState,
        spatial_id: u8,
    ) -> &'a mut LayerState {
        if spatial_id == self.active_spatial_id {
            active
        } else {
            self.parked[usize::from(spatial_id)]
                .as_mut()
                .expect("inactive spatial layer is parked")
        }
    }

    /// Swap `spatial_id` whole into `active` ([`Encoder::layer`]), parking the
    /// previously active layer back into `self.parked`.
    fn activate(&mut self, active: &mut LayerState, spatial_id: u8) {
        if spatial_id != self.active_spatial_id {
            let next = self.parked[usize::from(spatial_id)]
                .take()
                .expect("target spatial layer is parked");
            let prev = std::mem::replace(active, next);
            self.parked[usize::from(self.active_spatial_id)] = Some(prev);
            self.active_spatial_id = spatial_id;
        }
    }

    pub(super) fn source_mut(&mut self) -> &mut Picture {
        self.full_source
            .as_mut()
            .expect("full_source is present between temporal units")
    }

    pub(super) fn output_reconstruction(&self) -> &Picture {
        self.layer_reconstruction(self.output_spatial_id)
            .expect("the output layer exists")
    }

    pub(super) fn layer_reconstruction(&self, spatial_id: u8) -> Option<&Picture> {
        if usize::from(spatial_id) >= usize::from(self.config.layers) {
            return None;
        }
        self.visible_recon[usize::from(spatial_id)].as_ref()
    }

    pub(super) fn layer_rate_control_stats(
        &self,
        active: &LayerState,
        spatial_id: u8,
    ) -> Option<RateControlStats> {
        if usize::from(spatial_id) >= usize::from(self.config.layers) {
            return None;
        }
        let layer = self.layer(active, spatial_id);
        layer.rate_control.as_ref().map(RateController::stats)
    }

    pub(super) fn combined_rate_control_stats(
        &self,
        active: &LayerState,
    ) -> Option<RateControlStats> {
        let cbr = self.combined_cbr.as_ref()?;
        let s0 = self.layer_rate_control_stats(active, 0)?;
        let mut peak_overflows = 0u64;
        let mut target_unreachable = 0u64;
        let mut recoded_frames = 0u64;
        let mut key_loan_bits = 0.0f64;
        let mut out_q = s0.current_quantizer;
        let mut residual_sigma = s0.residual_sigma;
        for sid in 0..self.config.layers {
            let st = self.layer_rate_control_stats(active, sid)?;
            peak_overflows += st.peak_overflows;
            target_unreachable += st.target_unreachable_frames;
            recoded_frames += st.recoded_frames;
            key_loan_bits += st.key_loan_bits;
            if sid == self.output_spatial_id && st.frames > 0 {
                out_q = st.current_quantizer;
            }
            if st.frames > 0 {
                residual_sigma = st.residual_sigma;
            }
        }
        Some(RateControlStats {
            current_quantizer: out_q,
            buffer_fullness: cbr.buffer_fullness(),
            actual_bps: cbr.actual_bps(),
            total_bits: cbr.total_bits(),
            frames: cbr.coded_tus(),
            target_bps: cbr.target_bps(),
            buffer_overflows: cbr.buffer_overflows(),
            max_frame_bits: cbr.peak_tu_bits(),
            peak_overflows,
            target_unreachable_frames: target_unreachable,
            dropped_frames: s0.dropped_frames,
            recoded_frames,
            key_loan_bits,
            residual_sigma,
            plan: s0.plan,
        })
    }

    pub(super) fn wants_look_ahead(&self, active: &LayerState) -> bool {
        self.structure.next_tu_index() == 0
            && self
                .layer(active, 0)
                .rate_control
                .as_ref()
                .is_some_and(RateController::wants_startup_motion)
    }

    pub(super) fn set_frame_rate(&mut self, active: &mut LayerState, rate: FrameRate) {
        let fps = f64::from(rate.num) / f64::from(rate.den);
        self.nominal_tu_dt = 1.0 / fps;
        let enhancement_interval = self.config.enhancement_interval.max(1);
        for sid in 0..self.config.layers {
            let layer_fps = if sid == 1 && self.config.layers == 2 {
                fps / f64::from(enhancement_interval)
            } else {
                fps
            };
            if let Some(rc) = self.layer_mut(active, sid).rate_control.as_mut() {
                rc.set_nominal_rate(layer_fps);
            }
        }
    }

    fn recode_ceiling(&self, key: bool) -> f64 {
        if key || self.recode_ceiling_inter <= 0.0 {
            self.recode_ceiling
        } else {
            self.recode_ceiling_inter
        }
    }

    pub(super) fn combined_recode_step(
        &self,
        plan: FramePlan,
        layer_bits: u64,
        codings: u32,
        rate_steps: &[f64],
        max_q: u32,
    ) -> Option<u32> {
        if codings > MAX_RECODES || plan.quantizer >= max_q {
            return None;
        }
        let cbr = self.combined_cbr.as_ref()?;
        let ceiling = self.recode_ceiling(plan.frame_type == FrameType::Key);
        let spent = self.current_tu_spent_bits;
        let total_tu_bits = spent.saturating_add(layer_bits);
        if !cbr.would_exceed_ceiling(total_tu_bits, self.current_tu_interval, ceiling) {
            return None;
        }
        let room_layer = cbr.remaining_tu_room(self.current_tu_interval, ceiling, spent);
        if (layer_bits as f64) <= room_layer {
            return None;
        }
        let current_step = rate_steps[plan.quantizer as usize];
        let target_step =
            current_step * (layer_bits as f64 / room_layer).powf(1.0 / RECODE_FIRST_EXPONENT);
        Some(quantizer_for_step(
            rate_steps,
            (plan.quantizer + 1, max_q),
            target_step,
        ))
    }
}

impl<B: CodecBackend> Encoder<B> {
    /// Apply [`EnhancementIntra::Gated`] to an enhancement-layer inter
    /// frame's decision parameters.
    fn gate_enhancement_intra(
        &self,
        layer_plan: &SpatialLayerPlan,
        params: &mut crate::DecisionParams,
    ) {
        let Some(scfg) = self.spatial_config() else {
            return;
        };
        // The gate leans on the lower layer's prediction to stand in for
        // intra, so K-SVC gates only the pictures that have it. Independent
        // layers keep their original rule (every inter picture), which their
        // streams were tuned with.
        let has_gate_reference = match scfg.inter_layer {
            InterLayerMode::Off => !layer_plan.intra_only,
            _ => layer_plan.predicts_from_lower,
        };
        let apply_gate = scfg.enhancement_intra == EnhancementIntra::Gated
            && layer_plan.spatial_id > 0
            && has_gate_reference;
        if apply_gate {
            params.intra_test_dc = 0.0;
            params.intra_gate_variance = params
                .intra_gate_variance
                .max(crate::DecisionParams::default().intra_gate_variance);
        }
    }

    /// Spatial scalability configuration when `spatial.layers > 1`, or `None`
    /// for single-layer streams.
    #[must_use]
    pub fn spatial_config(&self) -> Option<&SpatialConfig> {
        self.spatial.as_ref().map(|s| &s.config)
    }

    /// Visible reconstruction of `spatial_id` (`0..layers`) from the last
    /// temporal unit that coded that layer.
    #[must_use]
    pub fn layer_reconstruction(&self, spatial_id: u8) -> Option<&Picture> {
        self.spatial
            .as_ref()
            .and_then(|s| s.layer_reconstruction(spatial_id))
    }

    /// Per-layer rate-control statistics for `spatial_id` (`0..layers`) when
    /// spatial scalability and CBR rate control are both active.
    #[must_use]
    pub fn layer_rate_control_stats(&self, spatial_id: u8) -> Option<RateControlStats> {
        self.spatial
            .as_ref()
            .and_then(|s| s.layer_rate_control_stats(&self.layer, spatial_id))
    }

    /// Cumulative reference-selection counts summed across all enhancement
    /// spatial layers (`S1`, plus `S2` in 3-layer mode; `None` in single-layer).
    #[must_use]
    pub fn spatial_reference_stats(&self) -> Option<SpatialReferenceStats> {
        let spatial = self.spatial.as_ref()?;
        let steps = usize::from(spatial.config.layers) - 1;
        Some(
            spatial.step_ref_stats[..steps]
                .iter()
                .fold(SpatialReferenceStats::default(), |sum, &step| sum.add(step)),
        )
    }

    /// Cumulative reference-selection counts for enhancement `spatial_id`
    /// (`1` = `S1` from `S0`, `2` = `S2` from `S1`).
    #[must_use]
    pub fn layer_reference_stats(&self, spatial_id: u8) -> Option<SpatialReferenceStats> {
        let spatial = self.spatial.as_ref()?;
        if spatial_id == 0 || spatial_id >= spatial.config.layers {
            return None;
        }
        Some(spatial.step_ref_stats[usize::from(spatial_id - 1)])
    }

    pub(super) fn set_spatial_rate_control(
        &mut self,
        config: RateControlConfig,
    ) -> Result<(), RateControlError> {
        let spatial = self.spatial.as_mut().expect("spatial mode checked");
        let num_layers = spatial.config.layers;
        let budgets = split_rate_control(
            config,
            &spatial.config.layer_rate_shares,
            self.config.frame_rate,
            self.config.intra_period,
            spatial.config.enhancement_interval,
            spatial.config.inter_layer,
        )?;
        let rate_params = RateParams::from_overrides(&self.config.params);
        let model = self.backend.rate_model();
        let headroom = (self.rate_steps.len() - self.quantizer_steps.len()) as u32;
        let model = RateModel {
            max_quantizer: model.max_quantizer + headroom,
            ..model
        };
        let clear_enh_loan = spatial.config.inter_layer.active_on_inter_unit()
            && spatial.config.enhancement_interval > 1;
        for (sid, budget) in (0..num_layers).zip(budgets) {
            let clear_loan = sid == 1 && clear_enh_loan;
            let layer = spatial.layer_mut(&mut self.layer, sid);
            if let Some(rc) = layer.rate_control.as_mut() {
                rc.reconfigure(budget.config)?;
            } else {
                let mut rc = RateController::new(
                    budget.config,
                    self.config.codec,
                    model,
                    budget.frame_rate,
                    budget.intra_period,
                    self.config.quantizer,
                    self.rate_steps.clone(),
                    layer.width * layer.height,
                    &rate_params,
                    layer.rate_control_ladder,
                )?;
                if clear_loan {
                    rc.clear_key_loan();
                }
                layer.rate_control = Some(rc);
            }
        }
        if let Some(cbr) = spatial.combined_cbr.as_mut() {
            cbr.reconfigure(config);
        } else {
            spatial.combined_cbr = Some(
                CombinedCbrBuffer::with_share_caps(
                    config,
                    self.config.frame_rate,
                    &spatial.config.layer_rate_shares,
                )
                .with_inter_layer(spatial.config.inter_layer),
            );
        }
        spatial.rate_config = Some(config);
        self.backend.set_rate_control_enabled(true);
        Ok(())
    }

    pub(super) fn disable_spatial_rate_control(&mut self) {
        if let Some(spatial) = self.spatial.as_mut() {
            for sid in 0..spatial.config.layers {
                spatial.layer_mut(&mut self.layer, sid).rate_control = None;
            }
            spatial.combined_cbr = None;
            spatial.rate_config = None;
        }
        self.backend.set_rate_control_enabled(false);
    }

    pub(super) fn observe_spatial_leaf_choice(
        &mut self,
        area: usize,
        mode: Option<(PredictionMode, ReferenceChoice)>,
    ) {
        let Some(spatial) = self.spatial.as_mut() else {
            return;
        };
        let Some(step) = spatial.coding_inter_layer_step else {
            return;
        };
        let stats = &mut spatial.step_ref_stats[step];
        let area = area as u64;
        match mode {
            None | Some((PredictionMode::Inter(_), ReferenceChoice::Primary)) => {
                stats.temporal_blocks += 1;
                stats.temporal_area += area;
            }
            Some((
                PredictionMode::Inter(_),
                ReferenceChoice::Secondary | ReferenceChoice::Compound { .. },
            )) => {
                stats.inter_layer_blocks += 1;
                stats.inter_layer_area += area;
            }
            Some((PredictionMode::IntraDc, _)) => {
                stats.intra_blocks += 1;
                stats.intra_area += area;
            }
        }
    }

    /// Make `spatial_id` the active layer ([`Encoder::layer`]) and lend it
    /// its visible reconstruction buffer for the picture it codes next.
    fn activate_spatial_layer(&mut self, spatial_id: u8) -> Picture {
        let spatial = self.spatial.as_mut().expect("spatial mode");
        spatial.activate(&mut self.layer, spatial_id);
        spatial.visible_recon[usize::from(spatial_id)]
            .take()
            .expect("visible reconstructions are present between temporal units")
    }

    /// Give the active layer's visible reconstruction back to the session.
    fn store_visible_recon(&mut self, recon: Picture) {
        let spatial = self.spatial.as_mut().expect("spatial mode");
        spatial.visible_recon[usize::from(spatial.active_spatial_id)] = Some(recon);
    }

    pub(super) fn encode_spatial_tu(
        &mut self,
        input: Option<&Picture>,
        timestamp_us: Option<i64>,
    ) -> EncodedFrame {
        let own_full = input.is_none().then(|| {
            self.spatial
                .as_mut()
                .expect("spatial mode")
                .full_source
                .take()
                .expect("full_source present")
        });
        let full_src = input.unwrap_or_else(|| own_full.as_ref().expect("set above"));
        self.downscale_lower_sources(full_src);

        let tu_index = self
            .spatial
            .as_ref()
            .expect("spatial mode")
            .structure
            .next_tu_index();
        let scheduled_key =
            self.force_keyframe || !self.backend.supports_inter() || self.periodic_key(tu_index);
        let mut top_written = false;
        if self.spatial_hold_due(tu_index, scheduled_key, full_src, &mut top_written) {
            if let Some(full) = own_full {
                self.spatial.as_mut().expect("spatial mode").full_source = Some(full);
            }
            return self.hold_spatial_tu(tu_index, timestamp_us);
        }

        // A held unit keeps a requested key for the next coded one.
        self.force_keyframe = false;
        let time = self.timeline.next(timestamp_us);
        let (s1_due_sched, unit_drop) = self.plan_unit_drop(tu_index, scheduled_key, time.interval);
        let kept_layers = match unit_drop {
            SpatialUnitDrop::Unit => {
                if let Some(full) = own_full {
                    self.spatial.as_mut().expect("spatial mode").full_source = Some(full);
                }
                return self.drop_spatial_tu(tu_index, time, s1_due_sched);
            }
            SpatialUnitDrop::KeepLayers(k) => k,
        };

        let (s0_recon_buf, s0_analyzer, tu_plan, s1_interval_for_code) =
            self.analyze_and_commit_s0(tu_index, scheduled_key, kept_layers);
        let full_lookahead = self.lookahead.take();
        let (mut data, mut out_quantizer, s0_meta) = self.encode_s0_layer(
            &tu_plan,
            s0_analyzer,
            s0_recon_buf,
            full_lookahead.as_ref(),
            time.interval,
        );
        let mut layers = vec![s0_meta];
        let mut output_spatial_id = 0u8;
        let top_sid = self.spatial.as_ref().expect("spatial mode").config.layers - 1;
        for (s, plan) in tu_plan.upper.iter().enumerate() {
            let Some(plan) = plan else { break };
            let sid = plan.spatial_id;
            // Lower layers' sources were downscaled into their analyzers up
            // front; the top layer's is written here unless the hold check
            // already wrote it. `S1` keeps its own clock (decimated cadence).
            let source_written = if sid == top_sid { top_written } else { true };
            let interval = if sid == 1 {
                s1_interval_for_code
            } else {
                time.interval
            };
            let has_coded_upper = tu_plan.upper.get(s + 1).is_some_and(Option::is_some);
            let (bytes, quantizer, meta) = self.encode_upper_layer(
                tu_plan.key_unit,
                plan,
                has_coded_upper,
                full_src,
                source_written,
                interval,
                data.len(),
            );
            data.extend_from_slice(&bytes);
            layers.push(meta);
            out_quantizer = quantizer;
            output_spatial_id = sid;
        }
        self.finish_spatial_tu(
            &tu_plan,
            data,
            layers,
            out_quantizer,
            output_spatial_id,
            time,
            own_full,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_spatial_tu(
        &mut self,
        tu_plan: &TemporalUnitPlan,
        mut data: Vec<u8>,
        layers: Vec<LayerFrameMetadata>,
        out_quantizer: u32,
        output_spatial_id: u8,
        time: FrameTime,
        own_full: Option<Picture>,
    ) -> EncodedFrame {
        let layer_lengths: Vec<usize> = layers.iter().map(|l| l.byte_len).collect();
        self.backend.finish_spatial_tu(&mut data, &layer_lengths);
        let tu_refs = union_tu_references(&layers);
        let out_temporal_layer = layers
            .iter()
            .map(|l| l.temporal_layer)
            .max()
            .unwrap_or(tu_plan.base.temporal_id);
        let tu_bits = bit_count(&data);
        let spatial = self.spatial.as_mut().expect("spatial mode");
        spatial.output_spatial_id = output_spatial_id;
        if let Some(cbr) = spatial.combined_cbr.as_mut() {
            cbr.observe_tu(tu_bits, time.interval, true);
        }
        if let Some(full) = own_full {
            spatial.full_source = Some(full);
        }
        EncodedFrame {
            data,
            metadata: FrameMetadata {
                index: tu_plan.tu_index,
                frame_type: if tu_plan.key_unit {
                    FrameType::Key
                } else {
                    FrameType::Inter
                },
                temporal_layer: out_temporal_layer,
                timestamp_us: time.timestamp_us,
                dropped: false,
                held: false,
                quantizer: out_quantizer,
                references: tu_refs,
            },
            layers,
        }
    }

    /// Downscale `full_src` through each spatial step, top down (`S2 -> S1
    /// -> S0`), into every lower layer's next source picture, leaving `S0`
    /// active.
    fn downscale_lower_sources(&mut self, full_src: &Picture) {
        let dsp = &self.dsp;
        let spatial = self.spatial.as_mut().expect("spatial mode");
        let top = usize::from(spatial.config.layers) - 1;
        for s in (0..top).rev() {
            spatial.activate(&mut self.layer, s as u8);
            let dst = next_source_mut(&mut self.layer);
            let src: &Picture = if s + 1 == top {
                full_src
            } else {
                next_source_mut(
                    spatial.parked[s + 1]
                        .as_mut()
                        .expect("upper layer is parked"),
                )
            };
            spatial.down_resamplers[s].scale(dsp, src, dst);
        }
    }

    /// Swap `S0` into `self`, run base-layer analysis and scene-cut detection,
    /// and commit the temporal unit schedule.
    fn analyze_and_commit_s0(
        &mut self,
        tu_index: u64,
        scheduled_key: bool,
        mut kept_layers: u8,
    ) -> (Picture, Option<Analyzer>, TemporalUnitPlan, Option<f64>) {
        let (s0_recon_buf, planned_base, base_coded_idx, num_layers) = {
            let spatial = self.spatial.as_mut().expect("spatial mode");
            let planned = spatial.structure.peek_base(scheduled_key);
            let idx = spatial.structure.layer_coded_count(0);
            let nl = spatial.config.layers;
            (self.activate_spatial_layer(0), planned, idx, nl)
        };

        let mut s0_analyzer = self.layer.analyzer.take();
        if let Some(a) = s0_analyzer.as_mut() {
            self.apply_search_role(a, base_coded_idx, &planned_base, scheduled_key);
            a.analyze_written_timed(planned_base.link(), &mut NoTimer);
        }
        let cut = !scheduled_key
            && self.is_scene_cut(base_coded_idx, &planned_base, s0_analyzer.as_ref());
        let key_unit = scheduled_key || cut;
        if key_unit {
            kept_layers = num_layers;
            if let Some(scene) = self.scene.as_mut() {
                scene.reset();
            }
        }
        if let (true, Some(a)) = (cut, s0_analyzer.as_mut()) {
            a.refresh_current(nc_analysis::link::SlotMask::ALL);
        }
        if let Some(analysis) = s0_analyzer.as_ref().and_then(Analyzer::analysis)
            && let Some(motion) = &analysis.motion
        {
            let sig = motion_signal(
                motion,
                Some(analysis),
                planned_base.reference.distance,
                self.layer.width,
            );
            self.spatial
                .as_mut()
                .expect("spatial mode")
                .structure
                .observe_motion(sig);
        }

        let (tu_plan, s1_interval_for_code) = {
            let spatial = self.spatial.as_mut().expect("spatial mode");
            let s1_due_final = spatial.structure.layer_due(1, key_unit);
            let s1_interval = if !s1_due_final {
                None
            } else if tu_index == 0 {
                spatial.s1_clock = EnhancementClock::default();
                None
            } else {
                spatial.s1_clock.take(spatial.config.enhancement_interval)
            };
            let plan = spatial
                .structure
                .next_temporal_unit_layers(key_unit, kept_layers);
            // Notify rate controllers of any upper layers dropped on this TU.
            if s1_due_final
                && plan.upper[0].is_none()
                && let Some(rc) = spatial.layer_mut(&mut self.layer, 1).rate_control.as_mut()
            {
                rc.set_frame_interval(s1_interval);
                rc.drop_frame();
            }
            // Layers above `S1` are due every unit, their frame interval
            // already set for this one (`plan_unit_drop`).
            for sid in 2..num_layers {
                if plan.layer(sid).is_none()
                    && let Some(rc) = spatial
                        .layer_mut(&mut self.layer, sid)
                        .rate_control
                        .as_mut()
                {
                    rc.drop_frame();
                }
            }
            (plan, s1_interval)
        };
        self.next_index = self
            .spatial
            .as_ref()
            .expect("spatial mode")
            .structure
            .next_tu_index();
        (s0_recon_buf, s0_analyzer, tu_plan, s1_interval_for_code)
    }

    /// Encode `S0` (the active layer), and commit its reference.
    fn encode_s0_layer(
        &mut self,
        tu_plan: &TemporalUnitPlan,
        s0_analyzer: Option<Analyzer>,
        mut s0_recon_buf: Picture,
        full_lookahead: Option<&Picture>,
        interval: Option<f64>,
    ) -> (Vec<u8>, u32, LayerFrameMetadata) {
        if let Some(next_full) = full_lookahead {
            self.observe_s0_startup_lookahead(s0_analyzer.as_ref(), next_full);
        }
        {
            let spatial = self.spatial.as_mut().expect("spatial mode");
            spatial.coding_inter_layer_step = None;
            spatial.current_layer_has_coded_upper = tu_plan.upper[0].is_some();
            spatial.current_tu_spent_bits = 0;
            spatial.current_tu_reallocated_bits = 0.0;
            spatial.current_tu_carry_to_s2 = 0.0;
            spatial.current_tu_interval = interval;
        }
        let s0_nominal_inter_budget = (!tu_plan.key_unit)
            .then(|| {
                self.layer
                    .rate_control
                    .as_ref()
                    .map(RateController::inter_frame_budget)
            })
            .flatten()
            .unwrap_or(0.0);
        let (s0_bytes, s0_quantizer) = self.code_spatial_layer(
            tu_plan.key_unit,
            &tu_plan.base,
            s0_analyzer,
            &mut s0_recon_buf,
        );
        let s0_bits = bit_count(&s0_bytes);
        self.backend.commit_spatial_reference(
            0,
            &s0_recon_buf,
            (self.layer.width, self.layer.height),
            tu_plan.base.references.refresh,
            tu_plan.key_unit,
        );
        self.store_visible_recon(s0_recon_buf);
        let ((base_w, base_h), op_mask) = {
            let spatial = self.spatial.as_mut().expect("spatial mode");
            spatial.current_tu_spent_bits = s0_bits;
            spatial.last_layer_quantizer[0] = s0_quantizer;
            if !tu_plan.key_unit
                && tu_plan.upper[0].is_some()
                && spatial.config.inter_layer.active_on_key_unit()
                && spatial.config.enhancement_interval == 1
                && let (Some(cbr), Some(s0_rc)) = (
                    spatial.combined_cbr.as_ref(),
                    self.layer.rate_control.as_mut(),
                )
            {
                let u0 =
                    cbr.reallocate_unspent_lower_bits(0, s0_nominal_inter_budget, 0.0, s0_bits);
                s0_rc.forfeit_unspent_inter_bits(u0);
                spatial.current_tu_reallocated_bits = u0;
            }
            (
                spatial.config.base_size,
                operating_point_mask(&spatial.config, 0, tu_plan.base.temporal_id),
            )
        };
        let meta = LayerFrameMetadata {
            spatial_id: 0,
            temporal_layer: tu_plan.base.temporal_id,
            picture_index: tu_plan.base.picture_index,
            frame_type: tu_plan.base.frame_type,
            enhancement_repair: false,
            width: base_w,
            height: base_h,
            byte_offset: 0,
            byte_len: s0_bytes.len(),
            quantizer: s0_quantizer,
            references: tu_plan.base.referenced_pictures,
            inter_layer_ref: false,
            droppable: tu_plan.base.references.refresh.is_empty(),
            operating_point_mask: op_mask,
        };
        (s0_bytes, s0_quantizer, meta)
    }

    fn observe_s0_startup_lookahead(
        &mut self,
        s0_analyzer: Option<&Analyzer>,
        next_full: &Picture,
    ) {
        if self.layer.rate_control.is_none() {
            return;
        }
        let (next_base, scale_x) = {
            let spatial = self.spatial.as_mut().expect("spatial mode");
            // The base layer's look-ahead picture, through the same steps as
            // its source (`downscale_lower_sources`).
            let mut next: Option<Picture> = None;
            for s in (0..usize::from(spatial.config.layers) - 1).rev() {
                let (w, h) = spatial.config.layer_sizes[s];
                let mut dst = Picture::new(w, h, 0).expect("valid layer dimensions");
                let src = next.as_ref().unwrap_or(next_full);
                spatial.down_resamplers[s].scale(&self.dsp, src, &mut dst);
                next = Some(dst);
            }
            (
                next.expect("at least two spatial layers"),
                spatial.config.full_size.0 as f64 / spatial.config.base_size.0.max(1) as f64,
            )
        };
        let s0_src = match (s0_analyzer, self.layer.input.as_ref()) {
            (Some(a), _) => a.frame(),
            (None, Some(buf)) => buf,
            (None, None) => unreachable!("S0 owns analyzer or input"),
        };
        if let Some((peak, static_fraction)) = startup_scene_profile(
            &self.config,
            (self.layer.width, self.layer.height),
            s0_src,
            &next_base,
        ) {
            if let Some(rc) = self.layer.rate_control.as_mut() {
                rc.observe_startup_motion(peak);
            }
            self.spatial.as_mut().expect("spatial mode").startup_profile =
                Some((peak * scale_x, static_fraction));
        }
    }

    /// Make enhancement layer `layer_plan.spatial_id` (`1` or `2`) the active
    /// layer, inject the upscaled lower-layer proxy, run analysis, code the
    /// picture, commit its reference, and propagate unspent inter bits upward.
    #[allow(clippy::too_many_arguments)]
    fn encode_upper_layer(
        &mut self,
        key_unit: bool,
        layer_plan: &SpatialLayerPlan,
        has_coded_upper: bool,
        full_src: &Picture,
        source_already_written: bool,
        layer_interval: Option<f64>,
        byte_offset: usize,
    ) -> (Vec<u8>, u32, LayerFrameMetadata) {
        let sid = layer_plan.spatial_id;
        let step = usize::from(sid - 1);
        let layer_reset = key_unit || layer_plan.enhancement_repair;
        let mut recon_buf = self.prepare_upper_layer_state(layer_plan, layer_interval);
        let analyzer =
            self.analyze_upper_layer(layer_plan, layer_reset, full_src, source_already_written);

        // In 3-layer mode when S1 (`sid == 1`) codes with S2 (`has_coded_upper`)
        // on an inter TU, split incoming unspent S0 bits `u0` between S1 (`r1`)
        // and direct carry to S2 (`u0 - r1`) before S1 selects its quantizer:
        let s1_static = analyzer
            .as_ref()
            .and_then(Analyzer::analysis)
            .map_or(0.0, |a| a.pre.static_fraction);
        let prev_ref_stats = {
            let spatial = self.spatial.as_mut().expect("spatial mode");
            spatial.current_layer_has_coded_upper = has_coded_upper;
            if sid == 1
                && has_coded_upper
                && !layer_reset
                && spatial.current_tu_reallocated_bits > 0.0
            {
                let s2_inter = spatial
                    .layer(&self.layer, 2)
                    .rate_control
                    .as_ref()
                    .and_then(RateController::inter_level);
                if let Some(cbr) = spatial.combined_cbr.as_ref() {
                    let (r1, carry_s2) = cbr.split_s0_reallocation_for_s1(
                        spatial.current_tu_reallocated_bits,
                        s1_static,
                        self.layer.width * self.layer.height,
                        s2_inter,
                    );
                    spatial.current_tu_reallocated_bits = r1;
                    spatial.current_tu_carry_to_s2 = carry_s2;
                }
            }
            spatial.coding_inter_layer_step = layer_plan.references.secondary.map(|_| step);
            spatial.step_ref_stats[step]
        };

        let nominal_inter_budget = (!layer_reset && has_coded_upper)
            .then(|| {
                self.layer
                    .rate_control
                    .as_ref()
                    .map(RateController::inter_frame_budget)
            })
            .flatten()
            .unwrap_or(0.0);
        let incoming_reallocated = self
            .spatial
            .as_ref()
            .map_or(0.0, |s| s.current_tu_reallocated_bits);

        let (layer_bytes, layer_quantizer) =
            self.code_spatial_layer(key_unit, layer_plan, analyzer, &mut recon_buf);
        let layer_bits = bit_count(&layer_bytes);
        self.backend.commit_spatial_reference(
            sid,
            &recon_buf,
            (self.layer.width, self.layer.height),
            layer_plan.references.refresh,
            false,
        );
        self.store_visible_recon(recon_buf);
        let ((layer_w, layer_h), op_mask) = self.finish_upper_layer_reallocation(UpperLayerDone {
            sid,
            step,
            temporal_id: layer_plan.temporal_id,
            has_coded_upper,
            layer_reset,
            prev_ref_stats,
            nominal_inter_budget,
            incoming_reallocated,
            layer_bits,
            layer_quantizer,
        });
        let meta = LayerFrameMetadata {
            spatial_id: sid,
            temporal_layer: layer_plan.temporal_id,
            picture_index: layer_plan.picture_index,
            frame_type: layer_plan.frame_type,
            enhancement_repair: layer_plan.enhancement_repair,
            width: layer_w,
            height: layer_h,
            byte_offset,
            byte_len: layer_bytes.len(),
            quantizer: layer_quantizer,
            references: layer_plan.referenced_pictures,
            inter_layer_ref: layer_plan.predicts_from_lower,
            droppable: true,
            operating_point_mask: op_mask,
        };
        (layer_bytes, layer_quantizer, meta)
    }

    fn analyze_upper_layer(
        &mut self,
        layer_plan: &SpatialLayerPlan,
        layer_reset: bool,
        full_src: &Picture,
        source_already_written: bool,
    ) -> Option<Analyzer> {
        let sid = layer_plan.spatial_id;
        let mut analyzer = self.layer.analyzer.take();
        if let Some(a) = analyzer.as_mut() {
            if !source_already_written {
                a.write_next(full_src);
            }
            let link = if layer_plan.intra_only {
                FrameLink {
                    reference: nc_analysis::link::ReferenceLink {
                        slot: BASE_SLOT_OFFSET,
                        distance: 1,
                    },
                    refresh: layer_temporal_slots(sid),
                }
            } else if layer_reset {
                // Key/repair enhancement frames code from the upscaled lower layer,
                // but analyse against this layer's own last temporal slot so
                // stillness runs carry across starved key TUs (`static_recovery.rs`).
                FrameLink {
                    reference: nc_analysis::link::ReferenceLink {
                        slot: layer_slot_offset(sid),
                        distance: 1,
                    },
                    refresh: layer_plan.references.link().refresh,
                }
            } else {
                layer_plan.references.link()
            };
            self.apply_search_role(
                a,
                layer_plan.layer_frame_index,
                &layer_plan.references,
                layer_reset,
            );
            a.analyze_written_timed(link, &mut NoTimer);
        } else if !source_already_written && let Some(buf) = self.layer.input.as_mut() {
            for plane in nc_common::PlaneIndex::ALL {
                buf.plane_mut(plane)
                    .view_mut()
                    .copy_from(&full_src.plane(plane).view());
            }
        }
        if layer_plan.layer_frame_index == 0
            && let (Some((full_peak, _)), Some(rc)) = (
                self.spatial.as_ref().and_then(|s| s.startup_profile),
                self.layer.rate_control.as_mut(),
            )
        {
            let full_w = self
                .spatial
                .as_ref()
                .map_or(self.layer.width, |s| s.config.full_size.0)
                .max(1);
            let layer_peak = full_peak * (self.layer.width as f64 / full_w as f64);
            rc.observe_startup_motion(layer_peak);
        }
        analyzer
    }

    fn finish_upper_layer_reallocation(&mut self, done: UpperLayerDone) -> ((usize, usize), u32) {
        let spatial = self.spatial.as_mut().expect("spatial mode");
        if spatial.coding_inter_layer_step == Some(done.step)
            && let Some(cbr) = spatial.combined_cbr.as_mut()
        {
            let cur = spatial.step_ref_stats[done.step];
            let d_il = cur.inter_layer_area - done.prev_ref_stats.inter_layer_area;
            let d_t = cur.temporal_area - done.prev_ref_stats.temporal_area;
            let d_intra = cur.intra_area - done.prev_ref_stats.intra_area;
            cbr.observe_step_inter_layer_hit_rate(done.step, d_il, d_il + d_t + d_intra);
        }
        spatial.coding_inter_layer_step = None;
        spatial.current_tu_spent_bits = spatial
            .current_tu_spent_bits
            .saturating_add(done.layer_bits);
        spatial.last_layer_quantizer[usize::from(done.sid)] = done.layer_quantizer;
        if done.has_coded_upper
            && !done.layer_reset
            && spatial.config.inter_layer.active_on_key_unit()
            && let (Some(cbr), Some(rc)) = (
                spatial.combined_cbr.as_ref(),
                self.layer.rate_control.as_mut(),
            )
        {
            let f1 = cbr.reallocate_unspent_lower_bits(
                done.sid,
                done.nominal_inter_budget,
                done.incoming_reallocated,
                done.layer_bits,
            );
            rc.forfeit_unspent_inter_bits(f1);
            spatial.current_tu_reallocated_bits = spatial.current_tu_carry_to_s2 + f1;
            spatial.current_tu_carry_to_s2 = 0.0;
        } else {
            spatial.current_tu_reallocated_bits = 0.0;
        }
        (
            spatial.config.layer_sizes[usize::from(done.sid)],
            operating_point_mask(&spatial.config, done.sid, done.temporal_id),
        )
    }

    /// Make enhancement layer `layer_plan.spatial_id` active, injecting the
    /// upscaled `(spatial_id - 1)` proxy into its reference pool only on units
    /// where `layer_plan` actually predicts from `(spatial_id - 1)`.
    fn prepare_upper_layer_state(
        &mut self,
        layer_plan: &SpatialLayerPlan,
        layer_interval: Option<f64>,
    ) -> Picture {
        let sid = layer_plan.spatial_id;
        let step = usize::from(sid - 1);
        let recon_buf = self.activate_spatial_layer(sid);
        if let Some(rc) = self.layer.rate_control.as_mut() {
            rc.set_frame_interval(layer_interval);
        }
        if layer_plan.predicts_from_lower {
            let spatial = self.spatial.as_mut().expect("spatial mode");
            let lower_visible = spatial.visible_recon[step]
                .as_ref()
                .expect("lower layer's visible reconstruction is present");
            let lower_slot = match layer_plan.references.secondary {
                Some(sec) => sec.slot,
                None => layer_plan.references.reference.slot,
            };
            let pool = self.layer.references.as_mut().expect("enhancement pool");
            spatial.up_resamplers[step].scale_luma_only(
                &self.dsp,
                lower_visible,
                pool.current_mut(),
            );
            pool.store_proxy(nc_analysis::link::SlotMask::only(lower_slot));
        }
        recon_buf
    }

    /// Code one picture of the active spatial layer, writing its visible
    /// reconstruction into `visible_out`.
    fn code_spatial_layer(
        &mut self,
        key_unit: bool,
        layer_plan: &SpatialLayerPlan,
        analyzer: Option<Analyzer>,
        visible_out: &mut Picture,
    ) -> (Vec<u8>, u32) {
        let own_input = self.layer.input.take();
        let index = layer_plan.layer_frame_index;
        let rc_key =
            layer_plan.frame_type == FrameType::Key || key_unit || layer_plan.enhancement_repair;
        if rc_key {
            self.layer.last_key_index = index;
        }
        let from_lower_key = rc_key && layer_plan.predicts_from_lower;
        let analysis = analyzer.as_ref().and_then(Analyzer::analysis);
        let source = match (analyzer.as_ref(), own_input.as_ref()) {
            (Some(a), _) => a.frame(),
            (None, Some(buf)) => buf,
            (None, None) => unreachable!("layer owns analyzer or input"),
        };
        // On an inter-layer key/repair TU, enhancement layers (`S1`, `S2`) predict
        // from the same instant's lower layer: motion measured against their own
        // previous temporal frame is for the reuse stillness run only, not coding.
        let motion = analysis
            .filter(|_| !from_lower_key)
            .and_then(|a| a.motion.as_ref());
        let pre = analysis.filter(|_| motion.is_some()).map(|a| &a.pre);
        let still = analysis.filter(|_| motion.is_some()).map(|a| &a.still);

        let (reuse_frame, key_reuse_offset) =
            self.plan_spatial_reuse_and_aq(layer_plan, rc_key, analysis);

        self.block_offsets.clear();
        let plan = self.spatial_frame_plan(
            layer_plan.picture_index,
            index,
            layer_plan.frame_type,
            rc_key,
            from_lower_key,
            layer_plan.references,
            analysis.map(|a| &a.pre),
            motion,
            key_reuse_offset,
        );
        self.frame_quantizer = plan.quantizer;

        let mut padding = self.layer.padding.take();
        let coded_source = match padding.as_mut() {
            Some(p) => {
                copy_padded(source, &mut p.source);
                &p.source
            }
            None => source,
        };
        let secondary_ratio = layer_plan.references.secondary.map(|link| {
            link.distance as f32 / layer_plan.references.reference.distance.max(1) as f32
        });
        let inter = (layer_plan.frame_type == FrameType::Inter).then_some(FrameInputs {
            motion,
            pre,
            still,
            secondary_ratio,
        });
        let role = self.frame_role(index, &layer_plan.references, rc_key);
        let (mut decision_params, planner_params) =
            self.frame_params(index, &layer_plan.references, role, analysis);
        self.gate_enhancement_intra(layer_plan, &mut decision_params);
        self.frame_decision_params = decision_params;
        let layer_ctx = self.prepare_spatial_backend(
            layer_plan,
            key_unit,
            role,
            analysis.map_or(0.0, |a| a.noise_sigma),
        );

        let mut pool = self.layer.references.take().expect("layer pool present");
        let secondary_slot = layer_plan
            .references
            .secondary
            .filter(|_| layer_plan.frame_type == FrameType::Inter)
            .map(|link| link.slot);
        let (reference, secondary, recon) =
            pool.frames(layer_plan.references.reference.slot, secondary_slot);
        let (data, final_plan) = self.code_spatial_layer_with_recode(
            plan,
            layer_ctx,
            (coded_source, reference, secondary, recon),
            inter.as_ref(),
            &planner_params,
        );
        self.backend.commit_frame();
        self.finish_coded_frame_feedback(
            layer_plan.layer_frame_index,
            rc_key,
            layer_plan.references.position,
            reuse_frame,
            final_plan.quantizer,
            analysis,
            (coded_source, recon),
            &data,
        );
        self.layer.analyzer = analyzer;
        self.layer.input = own_input;
        pool.finish(layer_plan.references.refresh, &self.dsp);
        copy_layer_visible_recon(&self.dsp, padding.as_mut(), pool.last(), visible_out);
        self.layer.references = Some(pool);
        self.layer.padding = padding;
        (data, final_plan.quantizer)
    }

    fn prepare_spatial_backend(
        &mut self,
        layer_plan: &SpatialLayerPlan,
        key_unit: bool,
        role: crate::encoder::FrameRole,
        noise_sigma: f32,
    ) -> SpatialLayerFrame {
        self.apply_backend_role(role);
        self.backend.set_noise_sigma(noise_sigma);
        self.window = None;
        self.backend.set_refresh(None);
        let max_bitrate_bps = self
            .spatial
            .as_ref()
            .and_then(|s| s.rate_config)
            .map(|rc| rc.peak_bps.unwrap_or(rc.target_bps).max(rc.target_bps));
        let layer_ctx = SpatialLayerFrame {
            spatial_id: layer_plan.spatial_id,
            temporal_id: layer_plan.temporal_id,
            key_unit,
            intra_only: layer_plan.intra_only,
            from_base_only: layer_plan.spatial_id > 0
                && (key_unit || layer_plan.enhancement_repair)
                && !layer_plan.intra_only,
            width: self.layer.width,
            height: self.layer.height,
            max_bitrate_bps,
        };
        self.backend.prepare_spatial_layer(layer_ctx);
        layer_ctx
    }

    fn code_spatial_layer_with_recode(
        &mut self,
        mut plan: FramePlan,
        layer_ctx: SpatialLayerFrame,
        (coded_source, reference, secondary, recon): (
            &Picture,
            &Picture,
            Option<&Picture>,
            &mut Picture,
        ),
        inter: Option<&FrameInputs<'_>>,
        planner_params: &crate::PlannerParams,
    ) -> (Vec<u8>, FramePlan) {
        let pictures = FramePictures {
            source: coded_source,
            reference,
            secondary,
            recon: &mut *recon,
        };
        let first = self.identity_trial(plan).unwrap_or(plan);
        self.frame_quantizer = first.quantizer;
        let mut data = self.code_picture(&first, pictures, inter, planner_params);
        let mut codings = 1;
        while let Some(recode) = self.spatial_recode_plan(plan, bit_count(&data), codings) {
            codings += 1;
            plan = recode;
            self.frame_quantizer = plan.quantizer;
            self.block_offsets.clear();
            self.backend.prepare_spatial_layer(layer_ctx);
            let pictures = FramePictures {
                source: coded_source,
                reference,
                secondary,
                recon: &mut *recon,
            };
            data = self.code_picture(&plan, pictures, inter, planner_params);
        }
        (data, plan)
    }

    fn plan_spatial_reuse_and_aq(
        &mut self,
        layer_plan: &SpatialLayerPlan,
        rc_key: bool,
        analysis: Option<&nc_analysis::FrameAnalysis>,
    ) -> (bool, f64) {
        let index = layer_plan.layer_frame_index;
        let reuse_frame = on_reuse_chain(rc_key, &layer_plan.references);
        self.reuse_frame = reuse_frame.then_some(index);
        let Some(analysis) = analysis else {
            return (reuse_frame, 0.0);
        };
        let rframe = ReuseFrame {
            still: &analysis.still,
            pre: &analysis.pre,
            key: rc_key,
            horizon: self.spatial_frames_to_next_key(layer_plan.spatial_id),
            sample_step: self.reuse_sample_step(),
        };
        let step_dist = if layer_plan.spatial_id == 1 {
            let interval = self
                .spatial
                .as_ref()
                .expect("spatial mode")
                .config
                .enhancement_interval
                .max(1);
            layer_plan.references.reference.distance / interval
        } else {
            layer_plan.references.reference.distance
        };
        let reference_index = index.checked_sub(u64::from(step_dist.max(1)));
        let follows = step_dist <= 1
            || (reference_index.is_some() && reference_index == self.layer.reuse_last_frame);
        if let Some(term) = self.layer.reuse.as_mut().filter(|_| reuse_frame) {
            term.plan(&rframe);
            term.set_reference(!rc_key && follows);
        }
        let offsets = self
            .layer
            .reuse
            .as_ref()
            .filter(|_| reuse_frame)
            .map(ReuseTerm::offsets);
        let mut key_reuse_offset = 0.0;
        if let Some(aq) = self.layer.aq.as_mut() {
            aq.plan(&analysis.pre, analysis.motion.as_ref(), offsets);
        } else if rc_key {
            key_reuse_offset = self
                .layer
                .reuse
                .as_ref()
                .map_or(0.0, ReuseTerm::frame_offset);
        }
        (reuse_frame, key_reuse_offset)
    }

    #[allow(clippy::too_many_arguments)]
    fn spatial_frame_plan(
        &mut self,
        picture_index: u64,
        layer_index: u64,
        frame_type: FrameType,
        rc_key: bool,
        from_lower_key: bool,
        references: crate::structure::FrameReferences,
        pre: Option<&nc_analysis::preanalysis::PreAnalysis>,
        motion: Option<&nc_analysis::MotionAnalysis>,
        key_reuse_offset: f64,
    ) -> FramePlan {
        let share = self.layer_share(layer_index, rc_key, from_lower_key, pre);
        let mut plan = self.frame_plan_with_rc_key(
            layer_index,
            frame_type,
            rc_key,
            references,
            pre,
            motion,
            key_reuse_offset,
            &share,
        );
        plan.index = picture_index;
        plan
    }

    /// The active layer's share of this temporal unit, from the combined
    /// buffer ([`CombinedCbrBuffer::plan_share`]); the whole budget without
    /// rate control.
    fn layer_share(
        &self,
        layer_index: u64,
        rc_key: bool,
        from_lower_key: bool,
        pre: Option<&nc_analysis::preanalysis::PreAnalysis>,
    ) -> LayerShare {
        let spatial = self.spatial.as_ref().expect("spatial mode");
        let Some(cbr) = spatial.combined_cbr.as_ref() else {
            return LayerShare::WHOLE;
        };
        if self.layer.rate_control.is_none() {
            return LayerShare::WHOLE;
        }
        let sid = spatial.active_spatial_id;
        let ceiling = spatial.recode_ceiling(rc_key);
        let upper_inter_level = (sid + 1 < spatial.config.layers)
            .then(|| {
                spatial
                    .layer(&self.layer, sid + 1)
                    .rate_control
                    .as_ref()
                    .and_then(RateController::inter_level)
            })
            .flatten();
        let lower_quantizer = if from_lower_key {
            spatial.last_layer_quantizer[usize::from(sid - 1)]
        } else {
            spatial.last_layer_quantizer[usize::from(sid)]
        };
        let (w, h) = spatial.config.layer_sizes[usize::from(sid)];
        cbr.plan_share(&ShareRequest {
            spatial_id: sid,
            key: rc_key,
            from_lower_key,
            has_coded_upper_layer: spatial.current_layer_has_coded_upper,
            layer_index,
            inter_layer: spatial.config.inter_layer,
            enhancement_interval: spatial.config.enhancement_interval,
            interval: spatial.current_tu_interval,
            ceiling,
            prior_layer_bits: spatial.current_tu_spent_bits,
            lower_quantizer,
            reallocated_bits: spatial.current_tu_reallocated_bits,
            static_fraction: pre.map_or(0.0, |p| p.static_fraction),
            layer_pixels: w * h,
            upper_inter_level,
            startup_profile: spatial.startup_profile,
        })
    }

    fn spatial_recode_plan(
        &mut self,
        plan: FramePlan,
        bits: u64,
        codings: u32,
    ) -> Option<FramePlan> {
        if let Some(next) = self.recode_plan(plan, bits, codings) {
            if let Some(rc) = self.layer.rate_control.as_mut() {
                rc.sync_inter_level_after_recode();
            }
            return Some(next);
        }
        if !B::RECODE || self.random.is_some() {
            return None;
        }
        let max_q = max_quantizer(self.config.codec);
        let planned = self.spatial.as_ref()?.combined_recode_step(
            plan,
            bits,
            codings,
            &self.rate_steps,
            max_q,
        )?;
        let (quantizer, lambda_scale) = self.coded_quantizer(planned);
        if quantizer <= plan.quantizer {
            return None;
        }
        if let Some(rc) = self.layer.rate_control.as_mut() {
            rc.record_external_recode(planned, bits);
        }
        Some(FramePlan {
            quantizer,
            lambda_scale,
            ..plan
        })
    }

    fn spatial_frames_to_next_key(&self, spatial_id: u8) -> Option<u64> {
        let spatial = self.spatial.as_ref()?;
        let tu = spatial.structure.next_tu_index().saturating_sub(1);
        let rem_tus = self.frames_to_next_key(tu)?;
        if spatial_id == 1 && spatial.config.layers == 2 {
            let interval = u64::from(spatial.config.enhancement_interval.max(1));
            Some(rem_tus.div_ceil(interval).max(1))
        } else {
            Some(rem_tus)
        }
    }

    fn spatial_hold_due(
        &mut self,
        tu_index: u64,
        scheduled_key: bool,
        full_src: &Picture,
        top_written: &mut bool,
    ) -> bool {
        let enh_repair = self
            .spatial
            .as_ref()
            .is_some_and(|s| s.structure.enhancement_repair_pending());
        if self.hold.is_none() || scheduled_key || enh_repair || tu_index == 0 {
            return false;
        }
        let (top_sid, top_due, output_is_top) = {
            let spatial = self.spatial.as_ref().expect("spatial mode");
            let top = spatial.config.layers - 1;
            let due = spatial.structure.layer_due(top, false);
            (top, due, spatial.output_spatial_id == top)
        };
        if top_due && !output_is_top {
            return false;
        }
        let layer_id = if top_due { top_sid } else { 0 };
        let link = {
            let spatial = self.spatial.as_ref().expect("spatial mode");
            if top_due {
                spatial.structure.peek_layer_temporal(top_sid).link()
            } else {
                spatial.structure.peek_base(false).link()
            }
        };
        let steps = {
            let layer = self
                .spatial
                .as_ref()
                .expect("spatial mode")
                .layer(&self.layer, layer_id);
            let inter_level = layer
                .rate_control
                .as_ref()
                .and_then(RateController::inter_level)
                .unwrap_or(self.config.quantizer);
            (
                self.sample_step(inter_level),
                self.sample_step(layer.last_quantizer),
            )
        };
        let (Some(hold), Some(spatial)) = (self.hold.as_mut(), self.spatial.as_mut()) else {
            return false;
        };
        let Some(analyzer) = spatial
            .layer_mut(&mut self.layer, layer_id)
            .analyzer
            .as_mut()
        else {
            return false;
        };
        if analyzer.analysis().is_none() {
            return false;
        }
        if top_due {
            analyzer.write_next(full_src);
            *top_written = true;
        }
        analyzer.pre_analyze_next(link);
        let referenced = analyzer.references_last(link);
        let repeat = match (analyzer.next_pre(), analyzer.analysis()) {
            (Some(next), Some(last)) => hold.repeats(
                next,
                LastCoded {
                    pre: &last.pre,
                    referenced,
                },
                steps,
            ),
            _ => false,
        };
        hold.holds(repeat)
    }

    fn hold_spatial_tu(&mut self, tu_index: u64, timestamp_us: Option<i64>) -> EncodedFrame {
        let time = self.timeline.next(timestamp_us);
        let spatial = self.spatial.as_mut().expect("spatial mode");
        let num_layers = spatial.config.layers;
        let enhancement_interval = spatial.config.enhancement_interval;
        if let Some(rc) = spatial.layer_mut(&mut self.layer, 0).rate_control.as_mut() {
            rc.set_frame_interval(time.interval);
            rc.hold_frame();
        }
        if enhancement_interval == 1 {
            for sid in 1..num_layers {
                if let Some(rc) = spatial
                    .layer_mut(&mut self.layer, sid)
                    .rate_control
                    .as_mut()
                {
                    rc.set_frame_interval(time.interval);
                    rc.hold_frame();
                }
            }
        } else {
            spatial
                .s1_clock
                .advance(time.interval, spatial.nominal_tu_dt);
        }
        if let Some(cbr) = spatial.combined_cbr.as_mut() {
            cbr.observe_tu(0, time.interval, false);
        }
        EncodedFrame {
            data: Vec::new(),
            metadata: FrameMetadata {
                index: tu_index,
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

    /// Advance `S1`'s clock by this unit and decide what rate control drops
    /// from the top down (`S2 -> S1 -> Unit`, [`decide_unit_drop_layers`]).
    fn plan_unit_drop(
        &mut self,
        tu_index: u64,
        scheduled_key: bool,
        interval: Option<f64>,
    ) -> (bool, SpatialUnitDrop) {
        let spatial = self.spatial.as_mut().expect("spatial mode");
        let num_layers = usize::from(spatial.config.layers);
        if tu_index > 0 {
            spatial.s1_clock.advance(interval, spatial.nominal_tu_dt);
        }
        let s1_due = spatial.structure.layer_due(1, scheduled_key);
        let s1_interval = (tu_index > 0)
            .then(|| {
                spatial
                    .s1_clock
                    .interval(spatial.config.enhancement_interval)
            })
            .flatten();
        if let Some(rc) = spatial.layer_mut(&mut self.layer, 0).rate_control.as_mut() {
            rc.set_frame_interval(interval);
        }
        if let (true, Some(rc)) = (
            s1_due,
            spatial.layer_mut(&mut self.layer, 1).rate_control.as_mut(),
        ) {
            rc.set_frame_interval(s1_interval);
        }
        for sid in 2..spatial.config.layers {
            if let Some(rc) = spatial
                .layer_mut(&mut self.layer, sid)
                .rate_control
                .as_mut()
            {
                rc.set_frame_interval(interval);
            }
        }

        let mut due = [false; MAX_SPATIAL_LAYERS];
        let mut drop_facts = [LayerDropFacts {
            droppable: false,
            controller_drop: false,
            floor_bits: None,
        }; MAX_SPATIAL_LAYERS];
        for s in 0..num_layers {
            let sid = s as u8;
            due[s] = spatial.structure.layer_due(sid, scheduled_key);
            let droppable = spatial.structure.layer_is_droppable(sid, false);
            let protected = spatial.structure.layer_protected_after_next(sid);
            let rc = spatial.layer(&self.layer, sid).rate_control.as_ref();
            drop_facts[s] = LayerDropFacts {
                droppable,
                controller_drop: rc.is_some_and(|rc| rc.should_drop(protected)),
                floor_bits: rc.and_then(RateController::smallest_inter_bits),
            };
        }
        let frame_drop = spatial
            .layer(&self.layer, 0)
            .rate_control
            .as_ref()
            .is_some_and(RateController::frame_drop_enabled);
        let cbr = spatial.combined_cbr.as_ref();
        let drop = decide_unit_drop_layers(
            frame_drop,
            scheduled_key,
            &due[..num_layers],
            &drop_facts[..num_layers],
            |bits| cbr.is_some_and(|c| c.would_overflow(bits, interval)),
        );
        (s1_due, drop)
    }

    /// Skip a whole temporal unit (all upper layers absent or dropped first).
    fn drop_spatial_tu(&mut self, tu_index: u64, time: FrameTime, s1_due: bool) -> EncodedFrame {
        let spatial = self.spatial.as_mut().expect("spatial mode");
        let num_layers = spatial.config.layers;
        if s1_due && spatial.config.enhancement_interval == 1 {
            for sid in 1..num_layers {
                if let Some(rc) = spatial
                    .layer_mut(&mut self.layer, sid)
                    .rate_control
                    .as_mut()
                {
                    rc.set_frame_interval(time.interval);
                    rc.drop_frame();
                }
            }
            spatial.s1_clock = EnhancementClock::default();
        }
        if let Some(rc) = spatial.layer_mut(&mut self.layer, 0).rate_control.as_mut() {
            rc.drop_frame();
        }
        if let Some(cbr) = spatial.combined_cbr.as_mut() {
            cbr.observe_tu(0, time.interval, false);
        }
        let layer = spatial.structure.drop_temporal_unit();
        self.next_index = spatial.structure.next_tu_index();
        EncodedFrame {
            data: Vec::new(),
            metadata: FrameMetadata {
                index: tu_index,
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
}

/// External `picture_index` dependencies of a temporal unit as a whole
/// (excluding intra-TU `S1 -> S0` and `S2 -> S1` dependencies so single-unit
/// callers see earlier TUs' picture indices).
fn union_tu_references(layers: &[LayerFrameMetadata]) -> ReferencedFrames {
    let mut local = [None; MAX_SPATIAL_LAYERS];
    for (dst, l) in local.iter_mut().zip(layers.iter()) {
        *dst = Some(l.picture_index);
    }
    let mut deps = [None; 2];
    let mut n = 0;
    for layer in layers {
        for dep in layer.references.iter() {
            if !local.contains(&Some(dep)) && !deps[..n].contains(&Some(dep)) && n < 2 {
                deps[n] = Some(dep);
                n += 1;
            }
        }
    }
    ReferencedFrames::from_pair(deps[0], deps[1])
}

fn copy_layer_visible_recon(
    dsp: &nc_dsp::Dsp,
    padding: Option<&mut Padding>,
    coded_last: &Picture,
    out: &mut Picture,
) {
    if let Some(p) = padding {
        copy_visible(dsp, coded_last, &mut p.visible_recon);
        copy_visible(dsp, &p.visible_recon, out);
    } else {
        copy_visible(dsp, coded_last, out);
    }
}

/// The picture a layer's next source is written into: its analyzer's next
/// frame, or its plain input buffer when the backend has no analysis.
fn next_source_mut(layer: &mut LayerState) -> &mut Picture {
    match (&mut layer.analyzer, &mut layer.input) {
        (Some(a), _) => a.next_frame_mut(),
        (None, Some(buf)) => buf,
        (None, None) => unreachable!("layer owns analyzer or input"),
    }
}
