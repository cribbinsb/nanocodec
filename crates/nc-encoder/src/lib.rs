//! The shared nanocodec encoder engine.
//!
//! One engine drives every codec backend. It owns everything
//! that is codec-independent — configuration, the frame loop, the square
//! block-partition walk, block decisions, the reconstruction buffer — and
//! reaches each codec only through the [`CodecBackend`] trait. Codec crates
//! sit one layer above this crate (see `docs/ARCHITECTURE.md`); this crate
//! never depends on any of them.
//!
//! The pipeline follows SVT-AV1 real-time M12 (`docs/ARCHITECTURE.md` §2,
//! `docs/reference/svt-av1-m12-rtc.md`; layout in `docs/handoff/engine.md`):
//! - frame planning: key frames (periodic, on request or at scene cuts),
//!   single-reference inter frames in a flat chain or hierarchical-P of up to
//!   three temporal layers (`structure`), a long-term reference with
//!   receiver feedback (`ltr`) and intra refresh (`refresh`);
//! - per superblock, a statistics-driven partition plan (`key_plan`,
//!   `partition_plan`, with rectangular shapes where the backend declares
//!   them), then a non-RD mode decision per leaf (`decision`): ranked inter
//!   candidates with sub-pel refinement, intra from a shortlist, still-block
//!   skips;
//! - shared quantisation and filter tools (`aq`, `trellis`, CDEF strength
//!   model and search) and the quality-capped rate control.
//!
//! - [`EncoderConfig`]: what to encode.
//! - [`CodecBackend`]: what a codec implements.
//! - [`Encoder`]: the frame loop; returns an [`EncodedFrame`] per input.
//! - [`ParamOverrides`]: tunable heuristic parameters (see `params`).
//! - [`RateControlConfig`]: optional runtime target/peak/buffer control, enabled
//!   by [`Encoder::set_rate_control`], with measured [`RateControlStats`].
//!   Fixed-QP remains the default. Targets are best-effort at quantizer limits
//!   and scene cuts; no padding conceals missed budgets, and frames are
//!   dropped only with the opt-in `RateControlConfig::frame_drop`.

#![forbid(unsafe_code)]

mod aq;
mod backend;
mod bpp_curve;
mod cdef_model;
mod cdef_search;
mod chroma_offset;
pub mod chroma_policy;
mod config;
mod cyclic_cadence;
mod decision;
mod encoder;
mod feedback;
mod frame_role;
mod held_quality;
mod hold;
mod inframe;
mod input;
mod intra_candidates;
mod intra_estimate;
mod key_clock;
mod key_plan;
mod key_qp;
mod layer;
mod ltr;
mod mv;
mod noise_skip;
// Public so `param_group!` (which names `$crate::params::…`) works in backend crates.
pub mod params;
mod partition;
mod partition_plan;
mod period_ledger;
mod persistence;
mod planned_refinement;
mod pre_tx_skip;
mod psy;
mod qp_cadence;
mod quality_cap;
mod random_decisions;
mod rate_control;
mod rate_window;
mod references;
mod refinement_gain;
mod refinement_loan;
mod refresh;
mod resolution;
mod reuse;
mod scene;
mod second_reference;
pub mod segment_quantizers;
mod shape;
mod skip_budget;
mod source_scale;
pub mod spatial;
mod startup_lookahead;
mod still_camera;
mod still_skip;
mod structure;
pub mod symbol_cost;
mod timeline;
mod trellis;
mod zero_residual;

pub use backend::{
    BlockDecision, CodecBackend, DEFAULT_LOOP_FILTER_REACH, FramePlan, FrameType, PredictionMode,
    ReferenceChoice, SpatialLayerFrame,
};
pub use cdef_model::{CDEF_MODEL_MAX_Q, cdef_strengths_from_q};
pub use cdef_search::{StrengthSelection, select_strength_sets};
pub use chroma_offset::{
    coarse_chroma_extra, error_gap_offset, noise_attenuated_chroma_offset, plane_error_ratio,
};
pub use config::{Codec, ConfigError, EncoderConfig, FrameRate, StreamSignal, TimingSignal};
pub use encoder::{
    EncodedFrame, Encoder, FrameMetadata, LayerFrameMetadata, SpatialReferenceStats,
    validate_params,
};
pub use feedback::ReferencedFrames;
pub use frame_role::{FRAME_PARAM_SPECS, FrameRole, NONBASE_PREFIX};
pub use hold::holds_frames;
pub use intra_candidates::{
    IntraModeSet, directions_by_orientation, shortlist_intra, with_neighbours,
};
pub use key_qp::{KEY_OFFSET_REFERENCE_PERIOD, fixed_qp_key_offset};
pub use layer::LayerPosition;
pub use ltr::{MAX_REFERENCE_AGE, long_term_requested, long_term_slot, reference_reach};
pub use mv::{MV_UNITS_PER_SAMPLE, Mv};
pub use nc_analysis::grain;
pub use nc_analysis::link::{MAX_REFERENCE_SLOTS, ReferenceLink, SlotMask};
pub use nc_common::{ColourDescription, ColourInfo};
pub use params::{
    AnalysisParams, ChromaParams, DebugParams, DecisionParams, InFrameParams, LayerParams,
    LtrParams, ParamError, ParamOverrides, ParamSpec, PersistParams, PlannerParams, QuantParams,
    RateParams, RefParams, RefreshParams, SceneParams, SpatialParams, StillParams, engine_specs,
};
pub use partition::Block;
pub use pre_tx_skip::{QuantizerBand, pre_tx_skip_score, pre_tx_skips};
pub use psy::{ac_energy, psy_penalty};
pub use random_decisions::DebugRng;
pub use refresh::{IntraRefresh, MIN_REFRESH_PERIOD, RefreshRegion, RefreshWindow};
pub use second_reference::uses_long_term_slot;
pub use shape::{MAX_SHAPE_PARTS, PartitionShape, ShapeSet};
pub use still_camera::still_camera_params;
pub use still_skip::{
    MIN_STILL_SKIP_SIZE, PlaneZeroLimits, StaticSkipTest, copy_zero_vector_prediction,
    zero_vector_residual_is_zero,
};
pub use structure::{FrameReferences, MAX_TEMPORAL_LAYERS, reference_slots};
pub use timeline::MICROS_PER_SECOND;
pub use trellis::{
    Candidates, CoeffRateModel, ForwardCoeffRateModel, ForwardTrellis, MAX_CANDIDATES, Trellis,
    ZERO_CANDIDATE_MAX_LEVEL,
};
pub use zero_residual::{
    TextureScales, ZeroLimits, residual_quantises_to_zero, residual_quantises_to_zero_scaled,
};

pub use rate_control::{
    RateControlConfig, RateControlError, RateControlStats, RateModel, RatePlanTrace,
};
pub use spatial::{
    CombinedCbrBuffer, EnhancementIntra, InterLayerMode, MAX_SPATIAL_LAYERS, OperatingPointSpec,
    ScalabilityMode, SpatialConfig,
};
