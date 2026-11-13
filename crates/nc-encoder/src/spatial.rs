//! Spatial scalability (`spatial.*`): scheduling, reference-slot allocation,
//! operating-point descriptors and CBR rate splitting for two- and three-layer
//! spatial SVC streams (`spatial_id = 0..layers`, `AV1 §5.3.3`, `§5.5.1`,
//! `§6.4.1`, `§7.11.3.3`, `§7.18`, `§7.20`).
//!
//! # Reference-Slot Map (AV1's 8 slots, AV1 §7.20)
//!
//! AV1 provides 8 reference frame slots (`0..8`, `AV1 §7.20`). Slots are
//! partitioned with named constants so every lower spatial operating point
//! (`S0` alone, or `S0 + S1`) is strictly reference-closed and never depends
//! on a higher spatial layer (`libaom` `examples/svc_encoder_rtc.cc`,
//! `set_layer_pattern` modes `6` = `L3T1`, `8` = `L3T2`, `9` = `L3T3`,
//! `11` = `L3T3` K-SVC):
//!
//! - Slots `0..=1` ([`BASE_TEMPORAL_SLOTS`], `layer_temporal_slots(0)`):
//!   `S0` temporal hierarchy (slot `0` for `T0` in `T1`/`T2`/`T3`; slot `1`
//!   for `T1` in `T3`).
//! - Slots `2..=3` ([`ENHANCEMENT_TEMPORAL_SLOTS`], `layer_temporal_slots(1)`):
//!   `S1` temporal hierarchy (slot `2` for `T0`; slot `3` for `T1` in `T3`).
//! - Slots `4..=5` ([`TOP_ENHANCEMENT_TEMPORAL_SLOTS`], `layer_temporal_slots(2)`):
//!   in 3-layer mode (`L3T1`/`L3T2`/`L3T3`), `S2`'s temporal hierarchy (slot
//!   `4` for `T0`; slot `5` for `T1` in `L3T3`).
//! - Disposable (`T2`) inter-layer scratch slots ([`inter_layer_scratch_slot`]):
//!   on non-reference `T2` temporal units (`T3` hierarchies) with
//!   `spatial.inter_layer = 1`, a disposable lower-layer picture refreshes its
//!   dedicated scratch slot so the layer immediately above can predict from the
//!   same instant's lower picture without overwriting `T0`/`T1` temporal slots:
//!   - In 2-layer mode (`num_layers = 2`): `S0 -> S1` uses slot `4`
//!     ([`INTER_LAYER_SCRATCH_SLOT`]); slots `5..=7` stay unused (`SPATIAL_POOL_SLOTS = 5`).
//!   - In 3-layer mode (`num_layers = 3`): `S0 -> S1` uses slot `6`
//!     (`inter_layer_scratch_slot(3, 0)`), `S1 -> S2` uses slot `7`
//!     (`inter_layer_scratch_slot(3, 1)`), and `S2` refreshes `SlotMask::NONE`,
//!     filling all 8 DPB slots (`spatial_pool_slots(3) = 8`) with zero
//!     cross-layer collisions across `L3T1`, `L3T2`, and `L3T3`.
//! - Why `S2` predicts from `S1` + earlier `S2` (and not directly from `S0`):
//!   matching `libaom` `svc_encoder_rtc.cc` modes `6`/`8`/`9`, `S1` already
//!   reconstructs both upscaled `S0` content and `360p` mid-frequency detail,
//!   so a `2:1` upscale from `S1` dominates a `4:1` upscale from `S0` while
//!   keeping `S2` in the fast two-reference (`LAST_FRAME`, `LAST2_FRAME`)
//!   signalling path.
//! - Long-term references (`ltr.interval > 0`), intra refresh
//!   (`refresh.period > 0`), temporal second reference (`refs.second = 1`),
//!   adaptive coded resolution (`resolution.adaptive = 1`), and decimated
//!   `enhancement_interval > 1` on 3 spatial layers are rejected cleanly with
//!   [`ConfigError`].
//!
//! # Key Temporal Units and Display Semantics
//!
//! On a key temporal unit (`tu_index == 0`, periodic `intra_period`, scene
//! cut, or [`crate::Encoder::request_keyframe`]):
//!
//! - `S0` is a standard `KEY_FRAME` (`frame_type = 0`, refreshing all 8 slots
//!   with `S0`, `AV1 §5.9.2` / `§7.20`).
//! - When `spatial.inter_layer = 1`, `S1` is an `INTER_FRAME` (`frame_type = 1`)
//!   predicting solely from the same temporal unit's `S0` picture upscaled via
//!   `AV1 §7.11.3.3` and refreshing [`ENHANCEMENT_TEMPORAL_SLOTS`]; in 3-layer
//!   mode, `S2` is an `INTER_FRAME` predicting solely from the same temporal
//!   unit's reconstructed `S1` picture upscaled via `AV1 §7.11.3.3` and
//!   refreshing [`TOP_ENHANCEMENT_TEMPORAL_SLOTS`]. Any receiver at any of the
//!   up to 9 operating points can therefore join cleanly at any key TU.
//! - When `spatial.inter_layer = 0`, `S1` and `S2` are `INTRA_ONLY_FRAME`s
//!   (`frame_type = 2`, `AV1 §5.9.2` / `§6.8.2`) refreshing only their own
//!   temporal slots (`2..=3` for `S1`, `4..=5` for `S2`).
//!
//! Every coded spatial layer picture carries `show_frame = 1`: per `AV1 §7.5`
//! ("Ordering of OBUs") and `§7.18` ("Choosing an operating point"), a decoder
//! targeting operating point `op` drops pictures outside `operating_point_idc`
//! (`AV1 §6.4.1`) and outputs the highest spatial layer kept in each temporal
//! unit.

use std::collections::VecDeque;

use nc_analysis::link::{MAX_REFERENCE_SLOTS, ReferenceLink, SlotMask};

use crate::backend::FrameType;
use crate::config::{ConfigError, EncoderConfig, FrameRate};
use crate::feedback::ReferencedFrames;
use crate::layer::LayerPosition;
use crate::params::{LtrParams, RefParams, RefreshParams, ResolutionParams, SpatialParams};
use crate::rate_control::{KeyAnchor, LayerShare, RateControlConfig, RateControlError};
use crate::resolution::half_size;
use crate::structure::{FrameReferences, MAX_TEMPORAL_LAYERS, TemporalStructure};

/// Maximum spatial layers supported by the shared engine (`S0`, `S1`, `S2`).
pub const MAX_SPATIAL_LAYERS: usize = 3;
/// Temporal DPB slots reserved per spatial layer (`T0` and `T1`).
pub const SLOTS_PER_SPATIAL_LAYER: usize = 2;

/// First temporal slot offset (`2 * spatial_id`) for `spatial_id` (`0..=2`).
#[must_use]
pub const fn layer_slot_offset(spatial_id: u8) -> usize {
    (spatial_id as usize) * SLOTS_PER_SPATIAL_LAYER
}

/// Temporal DPB slot mask (`{2 * spatial_id, 2 * spatial_id + 1}`) for `spatial_id`.
#[must_use]
pub const fn layer_temporal_slots(spatial_id: u8) -> SlotMask {
    SlotMask(0b11 << layer_slot_offset(spatial_id))
}

/// Scratch DPB slot refreshed by a disposable (`T2`) picture on `lower_spatial_id`
/// (`< num_layers - 1`) so `lower_spatial_id + 1` can predict from it without
/// overwriting `lower_spatial_id`'s `T0`/`T1` slots: the slots after every
/// layer's temporal pair (`4` in 2-layer; `6, 7` in 3-layer).
#[must_use]
pub const fn inter_layer_scratch_slot(num_layers: u8, lower_spatial_id: u8) -> usize {
    (num_layers as usize) * SLOTS_PER_SPATIAL_LAYER + (lower_spatial_id as usize)
}

/// Total active DPB slots needed by `num_layers` spatial SVC (`5` for 2 layers,
/// all `8` for 3): a temporal pair per layer and a scratch slot per step.
#[must_use]
pub const fn spatial_pool_slots(num_layers: u8) -> usize {
    (num_layers as usize) * SLOTS_PER_SPATIAL_LAYER + (num_layers.saturating_sub(1) as usize)
}

/// First reference slot belonging to the base spatial layer (`spatial_id = 0`).
pub const BASE_SLOT_OFFSET: usize = layer_slot_offset(0);
/// Reference slots used by the base spatial layer's temporal hierarchy (`0, 1`).
pub const BASE_TEMPORAL_SLOTS: SlotMask = layer_temporal_slots(0);
/// Reference slots used by the first enhancement layer's temporal hierarchy (`2, 3`).
pub const ENHANCEMENT_TEMPORAL_SLOTS: SlotMask = layer_temporal_slots(1);
/// Reference slots used by the top enhancement layer of three (`4, 5`).
pub const TOP_ENHANCEMENT_TEMPORAL_SLOTS: SlotMask = layer_temporal_slots(2);
/// Two-layer scratch slot refreshed by a non-reference `S0` picture when `S1` predicts from it.
pub const INTER_LAYER_SCRATCH_SLOT: usize = inter_layer_scratch_slot(2, 0);
/// Mask containing only [`INTER_LAYER_SCRATCH_SLOT`].
pub const INTER_LAYER_SCRATCH_MASK: SlotMask = SlotMask(1 << INTER_LAYER_SCRATCH_SLOT);
/// Active reference-pool slots kept by two-layer spatial SVC (`0..=4`).
pub const SPATIAL_POOL_SLOTS: usize = spatial_pool_slots(2);

/// `operating_point_idc` spatial-layer mask for `S0` only (`AV1 §6.4.1`).
const SPATIAL_LAYER_0_IDC: u16 = 0b0001 << 8;
/// `operating_point_idc` spatial-layer mask for `S0 + S1` (`AV1 §6.4.1`).
const SPATIAL_LAYERS_0_AND_1_IDC: u16 = 0b0011 << 8;
/// `operating_point_idc` spatial-layer mask for `S0 + S1 + S2` (`AV1 §6.4.1`).
const SPATIAL_LAYERS_0_1_AND_2_IDC: u16 = 0b0111 << 8;

/// Minimum allowed `spatial.base_rate_share`.
const MIN_BASE_RATE_SHARE: f64 = 0.1;
/// Maximum allowed `spatial.base_rate_share`.
const MAX_BASE_RATE_SHARE: f64 = 0.8;
/// Minimum allowed `spatial.s0_rate_share` in 3-layer mode.
const MIN_S0_RATE_SHARE: f64 = 0.05;
/// Maximum allowed `spatial.s0_rate_share` in 3-layer mode.
const MAX_S0_RATE_SHARE: f64 = 0.60;
/// Minimum allowed `spatial.s1_rate_share` in 3-layer mode.
const MIN_S1_RATE_SHARE: f64 = 0.10;
/// Maximum allowed `spatial.s1_rate_share` in 3-layer mode.
const MAX_S1_RATE_SHARE: f64 = 0.70;
/// Maximum combined `s0_rate_share + s1_rate_share` so `S2` retains at least `10%`.
const MAX_LOWER_RATE_SHARE_SUM: f64 = 0.90;
/// Maximum allowed `spatial.enhancement_interval` (`1..=30` TUs).
const MAX_ENHANCEMENT_INTERVAL: f32 = 30.0;

/// Maximum reference downscale (`RefWidth <= 2 * FrameWidth`, `AV1 §7.11.3.3`).
const AV1_MAX_REF_DOWNSCALE: usize = 2;
/// Maximum reference upscale (`FrameWidth <= 16 * RefWidth`, `AV1 §7.11.3.3`).
const AV1_MAX_REF_UPSCALE: usize = 16;

/// WebRTC-style scalability mode identifiers (`--scalability`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalabilityMode {
    /// 1 spatial layer, 1 temporal layer (flat single-layer).
    L1T1,
    /// 1 spatial layer, 2 temporal layers.
    L1T2,
    /// 1 spatial layer, 3 temporal layers.
    L1T3,
    /// 2 spatial layers, 1 temporal layer.
    L2T1,
    /// 2 spatial layers, 2 temporal layers.
    L2T2,
    /// 2 spatial layers, 3 temporal layers.
    L2T3,
    /// 3 spatial layers, 1 temporal layer.
    L3T1,
    /// 3 spatial layers, 2 temporal layers.
    L3T2,
    /// 3 spatial layers, 3 temporal layers.
    L3T3,
}

impl ScalabilityMode {
    /// Parse a case-insensitive mode name (`L1T1`..`L3T3`).
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_uppercase().as_str() {
            "L1T1" => Some(Self::L1T1),
            "L1T2" => Some(Self::L1T2),
            "L1T3" => Some(Self::L1T3),
            "L2T1" => Some(Self::L2T1),
            "L2T2" => Some(Self::L2T2),
            "L2T3" => Some(Self::L2T3),
            "L3T1" => Some(Self::L3T1),
            "L3T2" => Some(Self::L3T2),
            "L3T3" => Some(Self::L3T3),
            _ => None,
        }
    }

    /// Number of spatial layers (`1`, `2`, or `3`).
    #[must_use]
    pub const fn spatial_layers(self) -> u8 {
        match self {
            Self::L1T1 | Self::L1T2 | Self::L1T3 => 1,
            Self::L2T1 | Self::L2T2 | Self::L2T3 => 2,
            Self::L3T1 | Self::L3T2 | Self::L3T3 => 3,
        }
    }

    /// Number of temporal layers (`1`, `2`, or `3`).
    #[must_use]
    pub const fn temporal_layers(self) -> u8 {
        match self {
            Self::L1T1 | Self::L2T1 | Self::L3T1 => 1,
            Self::L1T2 | Self::L2T2 | Self::L3T2 => 2,
            Self::L1T3 | Self::L2T3 | Self::L3T3 => 3,
        }
    }

    /// Canonical uppercase mode name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::L1T1 => "L1T1",
            Self::L1T2 => "L1T2",
            Self::L1T3 => "L1T3",
            Self::L2T1 => "L2T1",
            Self::L2T2 => "L2T2",
            Self::L2T3 => "L2T3",
            Self::L3T1 => "L3T1",
            Self::L3T2 => "L3T2",
            Self::L3T3 => "L3T3",
        }
    }
}

/// One AV1 operating point in the sequence header (`AV1 §5.5.1`, `§6.4.1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperatingPointSpec {
    /// `operating_point_idc`: bits `0..=7` select temporal layers, bits
    /// `8..=11` select spatial layers (`AV1 §6.4.1`).
    pub idc: u16,
    /// Highest `spatial_id` included in this operating point (`0..=2`).
    pub max_spatial_id: u8,
    /// Highest `temporal_id` included in this operating point (`0..=2`).
    pub max_temporal_id: u8,
    /// Maximum visible width of any picture decoded at this operating point.
    pub width: usize,
    /// Maximum visible height of any picture decoded at this operating point.
    pub height: usize,
    /// Effective output frame rate of this operating point, for `seq_level_idx`.
    pub frame_rate: FrameRate,
}

/// The intra search of enhancement-layer (`S1` / `S2`) inter frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnhancementIntra {
    /// The frame's own decision parameters.
    AsFrame,
    /// No DC intra test, and intra gated by at least the engine's default
    /// variance (`spatial.enhancement_intra_gate`): the upscaled lower
    /// picture already predicts the low frequencies an intra block adds.
    Gated,
}

/// Validated spatial scalability configuration (`spatial.layers == 2` or `3`).
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialConfig {
    /// Number of spatial layers (`2` or `3`).
    pub layers: u8,
    /// Visible dimensions `(width, height)` per spatial layer (`0..layers`).
    pub layer_sizes: Vec<(usize, usize)>,
    /// Nominal CBR bitrate share per spatial layer (`0..layers`, summing to
    /// `1.0`): every lower layer's share is a cap, the top layer takes the
    /// rest (`spatial.base_rate_share` in two-layer mode,
    /// `spatial.s0_rate_share` / `spatial.s1_rate_share` in three-layer mode).
    /// The one place the configured split is resolved: the rate controllers,
    /// the combined buffer and the level signalling all read it.
    pub layer_rate_shares: Vec<f64>,
    /// Visible dimensions `(width, height)` of the lowest spatial layer (`spatial_id = 0`).
    pub base_size: (usize, usize),
    /// Visible dimensions `(width, height)` of the highest spatial layer (`spatial_id = layers - 1`).
    pub full_size: (usize, usize),
    /// Whether each enhancement layer predicts from the same temporal unit's lower picture.
    pub inter_layer: bool,
    /// `S1` is coded in every `enhancement_interval`-th temporal unit (`1..=30`, 2-layer only).
    pub enhancement_interval: u32,
    /// Whether exact `2:1` inter-layer steps use a bilinear `[1 3]/4` search proxy
    /// (`spatial.proxy_bilinear`) instead of Lanczos-3.
    pub proxy_bilinear: bool,
    /// Whether exact `2:1` source downscales use the `2x2` box mean
    /// (`spatial.base_box_downscale`) instead of Lanczos-3.
    pub base_box_downscale: bool,
    /// The intra search of enhancement-layer inter frames
    /// (`spatial.enhancement_intra_gate`).
    pub enhancement_intra: EnhancementIntra,
    /// Temporal layer count (`1..=3`).
    pub temporal_layers: u8,
    /// Operating points ordered from highest quality (`0`) to lowest (`AV1 §5.5.1`).
    pub operating_points: Vec<OperatingPointSpec>,
}

impl SpatialConfig {
    /// Parse and validate `spatial.*` from `config`; returns `Ok(None)` when
    /// `spatial.layers == 1` so single-layer streams take their untouched path.
    ///
    /// # Errors
    ///
    /// Rejects unsupported codecs, incompatible engine tools (`ltr.interval`,
    /// `refresh.period`, `refs.second`, `resolution.adaptive`), or dimensions
    /// outside AV1's reference-scaling limits.
    pub fn from_encoder_config(
        config: &EncoderConfig,
        supports_spatial_layers: bool,
    ) -> Result<Option<Self>, ConfigError> {
        let max_layers = if supports_spatial_layers {
            MAX_SPATIAL_LAYERS as u8
        } else {
            1
        };
        Self::from_encoder_config_max(config, max_layers)
    }

    /// Parse and validate `spatial.*` against a backend supporting up to
    /// `max_spatial_layers` (`1`, `2` for VP9, `3` for AV1).
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] on unsupported layer counts, incompatible engine
    /// tools, invalid rate-share caps, or step ratios outside `AV1 §7.11.3.3`.
    pub fn from_encoder_config_max(
        config: &EncoderConfig,
        max_spatial_layers: u8,
    ) -> Result<Option<Self>, ConfigError> {
        let params = SpatialParams::from_overrides(&config.params);
        let layers = params.layers.round() as u8;
        if layers <= 1 {
            return Ok(None);
        }
        let enhancement_interval = params
            .enhancement_interval
            .round()
            .clamp(1.0, MAX_ENHANCEMENT_INTERVAL) as u32;
        validate_spatial_compatible_tools(
            config,
            layers,
            max_spatial_layers,
            enhancement_interval,
        )?;
        let full_size = (config.width, config.height);
        let inter_layer = params.inter_layer >= 0.5;
        let (layer_sizes, layer_rate_shares, operating_points) = if layers == 2 {
            let base_size = base_layer_size(full_size, params.base_scale)?;
            let s0_share =
                f64::from(params.base_rate_share).clamp(MIN_BASE_RATE_SHARE, MAX_BASE_RATE_SHARE);
            let ops = build_operating_points(
                base_size,
                full_size,
                config.frame_rate,
                config.temporal_layers,
                enhancement_interval,
                inter_layer,
            );
            (
                vec![base_size, full_size],
                vec![s0_share, (1.0 - s0_share).max(0.0)],
                ops,
            )
        } else {
            let sizes = three_layer_sizes(full_size, params.s0_scale, params.s1_scale)?;
            let s0_share =
                f64::from(params.s0_rate_share).clamp(MIN_S0_RATE_SHARE, MAX_S0_RATE_SHARE);
            let s1_share =
                f64::from(params.s1_rate_share).clamp(MIN_S1_RATE_SHARE, MAX_S1_RATE_SHARE);
            if s0_share + s1_share > MAX_LOWER_RATE_SHARE_SUM {
                return Err(ConfigError::SpatialScalability(
                    "s0_rate_share + s1_rate_share must not exceed 0.90 (leave >= 10% for S2)",
                ));
            }
            let s2_share = 1.0 - s0_share - s1_share;
            let ops = build_three_layer_operating_points(
                sizes,
                config.frame_rate,
                config.temporal_layers,
            );
            (sizes.to_vec(), vec![s0_share, s1_share, s2_share], ops)
        };
        let base_size = layer_sizes[0];
        let any_exact_2x = layer_sizes
            .windows(2)
            .any(|pair| (pair[0].0 * 2, pair[0].1 * 2) == pair[1]);
        Ok(Some(Self {
            layers,
            layer_sizes,
            layer_rate_shares,
            base_size,
            full_size,
            inter_layer,
            enhancement_interval,
            proxy_bilinear: params.proxy_bilinear >= 0.5 && any_exact_2x,
            base_box_downscale: params.base_box_downscale >= 0.5 && any_exact_2x,
            enhancement_intra: if params.enhancement_intra_gate >= 0.5 {
                EnhancementIntra::Gated
            } else {
                EnhancementIntra::AsFrame
            },
            temporal_layers: config.temporal_layers,
            operating_points,
        }))
    }
}

fn validate_spatial_compatible_tools(
    config: &EncoderConfig,
    layers: u8,
    max_spatial_layers: u8,
    enhancement_interval: u32,
) -> Result<(), ConfigError> {
    if max_spatial_layers <= 1 {
        return Err(ConfigError::SpatialScalability(
            "not supported by this codec yet",
        ));
    }
    if layers > MAX_SPATIAL_LAYERS as u8 {
        return Err(ConfigError::SpatialScalability(
            "at most 3 spatial layers are supported",
        ));
    }
    if layers > max_spatial_layers {
        return Err(ConfigError::SpatialScalability(
            "this codec supports at most 2 spatial layers",
        ));
    }
    if !(1..=MAX_TEMPORAL_LAYERS).contains(&config.temporal_layers) {
        return Err(ConfigError::TemporalLayers(config.temporal_layers));
    }
    if LtrParams::from_overrides(&config.params).interval.round() as u32 > 0 {
        return Err(ConfigError::LongTermReference(
            "not supported with spatial layers",
        ));
    }
    if RefreshParams::from_overrides(&config.params).period > 0.0 {
        return Err(ConfigError::IntraRefresh(
            "not supported with spatial layers",
        ));
    }
    if RefParams::from_overrides(&config.params).second >= 0.5 {
        return Err(ConfigError::SecondReference(
            "not supported with spatial layers (inter-layer prediction uses the second reference)",
        ));
    }
    if ResolutionParams::from_overrides(&config.params).adaptive >= 0.5 {
        return Err(ConfigError::SpatialScalability(
            "not supported with adaptive coded resolution (resolution.adaptive)",
        ));
    }
    if layers == 3 && enhancement_interval > 1 {
        return Err(ConfigError::SpatialScalability(
            "decimated enhancement_interval > 1 is only supported with 2 spatial layers",
        ));
    }
    Ok(())
}

fn legal_step_axis(lower: usize, upper: usize) -> bool {
    lower >= 2
        && lower.is_multiple_of(2)
        && lower <= AV1_MAX_REF_DOWNSCALE * upper
        && upper <= AV1_MAX_REF_UPSCALE * lower
}

/// Even 4:2:0 base-layer dimensions for `full_size` at `base_scale`, checked
/// against AV1's `[1/2x, 16x]` reference-scaling bounds (`AV1 §7.11.3.3`).
///
/// Exact `0.5` uses [`half_size`] so 2:1 spatial SVC matches adaptive
/// resolution's half-size grid (`1280x720 -> 640x360`); other scales round to
/// the nearest even sample count (`1280x720` at `2/3 -> 854x480`, at
/// `0.8 -> 1024x576`).
///
/// # Errors
///
/// Returns [`ConfigError::SpatialScalability`] if the scaled dimensions violate
/// 4:2:0 alignment or AV1's reference-scaling ratio limits.
pub fn base_layer_size(
    (full_w, full_h): (usize, usize),
    base_scale: f32,
) -> Result<(usize, usize), ConfigError> {
    if full_w < 4 || full_h < 4 || !full_w.is_multiple_of(2) || !full_h.is_multiple_of(2) {
        return Err(ConfigError::SpatialScalability(
            "stream dimensions must be even and at least 4x4 for spatial layers",
        ));
    }
    let scale = f64::from(base_scale);
    let (base_w, base_h) = if (scale - 0.5).abs() < 1e-6 {
        half_size(full_w, full_h)
    } else {
        let even = |full: usize| {
            let min_even = full
                .div_ceil(AV1_MAX_REF_UPSCALE)
                .next_multiple_of(2)
                .max(2);
            (((full as f64 * scale) / 2.0).round() as usize * 2).clamp(min_even, full)
        };
        (even(full_w), even(full_h))
    };
    if !legal_step_axis(base_w, full_w) || !legal_step_axis(base_h, full_h) {
        return Err(ConfigError::SpatialScalability(
            "base_scale produces dimensions outside AV1 §7.11.3.3 scaling bounds",
        ));
    }
    Ok((base_w, base_h))
}

/// Validated even 4:2:0 dimensions `[S0, S1, S2]` for three-layer spatial SVC.
///
/// Default `(0.25, 0.5)` cascades [`half_size`] twice (`1280x720 -> 640x360 -> 320x180`);
/// custom `(s0_scale, s1_scale)` round to even sample counts and validate both
/// `S0 -> S1` and `S1 -> S2` against `AV1 §7.11.3.3` (`[1/2x, 16x]`).
///
/// # Errors
///
/// Returns [`ConfigError::SpatialScalability`] if dimensions are odd, `< 8x8`,
/// `s0_scale > s1_scale`, or either adjacent step violates `AV1 §7.11.3.3`.
pub fn three_layer_sizes(
    (full_w, full_h): (usize, usize),
    s0_scale: f32,
    s1_scale: f32,
) -> Result<[(usize, usize); 3], ConfigError> {
    if full_w < 8 || full_h < 8 || !full_w.is_multiple_of(2) || !full_h.is_multiple_of(2) {
        return Err(ConfigError::SpatialScalability(
            "stream dimensions must be even and at least 8x8 for 3 spatial layers",
        ));
    }
    if s0_scale > s1_scale {
        return Err(ConfigError::SpatialScalability(
            "s0_scale must not exceed s1_scale",
        ));
    }
    let s1 = base_layer_size((full_w, full_h), s1_scale)?;
    let s0 =
        if (f64::from(s0_scale) - 0.25).abs() < 1e-6 && (f64::from(s1_scale) - 0.5).abs() < 1e-6 {
            half_size(s1.0, s1.1)
        } else {
            let even = |full: usize, limit: usize| {
                let min_even = limit
                    .div_ceil(AV1_MAX_REF_UPSCALE)
                    .next_multiple_of(2)
                    .max(2);
                (((full as f64 * f64::from(s0_scale)) / 2.0).round() as usize * 2)
                    .clamp(min_even, limit)
            };
            (even(full_w, s1.0), even(full_h, s1.1))
        };
    if !legal_step_axis(s0.0, s1.0)
        || !legal_step_axis(s0.1, s1.1)
        || !legal_step_axis(s1.0, full_w)
        || !legal_step_axis(s1.1, full_h)
    {
        return Err(ConfigError::SpatialScalability(
            "three-layer scales violate AV1 §7.11.3.3 adjacent-layer scaling bounds",
        ));
    }
    Ok([s0, s1, (full_w, full_h)])
}

/// Minimum `temporal_id` that `S1` pictures carry when `enhancement_interval > 1`:
/// if `S1` predicts from `S0` (`inter_layer`), any operating point that keeps
/// `S1` must also keep every `S0` picture at `k * enhancement_interval`.
#[must_use]
pub const fn decimated_enhancement_temporal_id(
    temporal_layers: u8,
    enhancement_interval: u32,
    inter_layer: bool,
) -> u8 {
    if !inter_layer || temporal_layers <= 1 {
        return 0;
    }
    match temporal_layers {
        2 => {
            if enhancement_interval.is_multiple_of(2) {
                0
            } else {
                1
            }
        }
        _ => {
            // 3-layer dyadic pattern `[T0, T2, T1, T2]`: multiples of 4 hit T0
            // only; even multiples hit `{T0, T1}`; odd intervals hit `T2`.
            if enhancement_interval.is_multiple_of(4) {
                0
            } else if enhancement_interval.is_multiple_of(2) {
                1
            } else {
                2
            }
        }
    }
}

/// Build all valid two-layer `(spatial, temporal)` operating points ordered
/// highest quality first (`AV1 §5.5.1`, `§6.4.1`).
#[must_use]
pub fn build_operating_points(
    base_size: (usize, usize),
    full_size: (usize, usize),
    frame_rate: FrameRate,
    temporal_layers: u8,
    enhancement_interval: u32,
    inter_layer: bool,
) -> Vec<OperatingPointSpec> {
    let mut points = Vec::with_capacity(usize::from(temporal_layers) * 2);
    let min_enh_t = if enhancement_interval == 1 {
        0
    } else {
        decimated_enhancement_temporal_id(temporal_layers, enhancement_interval, inter_layer)
    };
    let subset_rate = |max_t: u8| {
        let drop_levels = temporal_layers.saturating_sub(1).saturating_sub(max_t);
        FrameRate {
            num: frame_rate.num,
            den: frame_rate.den.saturating_mul(1u32 << drop_levels),
        }
    };
    // Full spatial (`S0 + S1`, spatial bits `0b11 << 8`), highest frame rate first.
    for max_t in (min_enh_t..temporal_layers).rev() {
        let t_mask = (1u16 << (max_t + 1)) - 1;
        points.push(OperatingPointSpec {
            idc: SPATIAL_LAYERS_0_AND_1_IDC | t_mask,
            max_spatial_id: 1,
            max_temporal_id: max_t,
            width: full_size.0,
            height: full_size.1,
            frame_rate: subset_rate(max_t),
        });
    }
    // Base spatial only (`S0`, spatial bit `0b01 << 8`), highest frame rate first.
    for max_t in (0..temporal_layers).rev() {
        let t_mask = (1u16 << (max_t + 1)) - 1;
        points.push(OperatingPointSpec {
            idc: SPATIAL_LAYER_0_IDC | t_mask,
            max_spatial_id: 0,
            max_temporal_id: max_t,
            width: base_size.0,
            height: base_size.1,
            frame_rate: subset_rate(max_t),
        });
    }
    points
}

/// Build all valid three-layer `(spatial, temporal)` operating points (`3` in
/// `L3T1`, `6` in `L3T2`, `9` in `L3T3`) ordered highest quality first (`op 0`
/// = `S0 + S1 + S2` at full frame rate; last op = `S0` at `T0`; `AV1 §5.5.1`,
/// `§6.4.1`).
#[must_use]
pub fn build_three_layer_operating_points(
    sizes: [(usize, usize); 3],
    frame_rate: FrameRate,
    temporal_layers: u8,
) -> Vec<OperatingPointSpec> {
    let mut points = Vec::with_capacity(usize::from(temporal_layers) * 3);
    let subset_rate = |max_t: u8| {
        let drop_levels = temporal_layers.saturating_sub(1).saturating_sub(max_t);
        FrameRate {
            num: frame_rate.num,
            den: frame_rate.den.saturating_mul(1u32 << drop_levels),
        }
    };
    for (max_s, s_idc) in [
        (2u8, SPATIAL_LAYERS_0_1_AND_2_IDC),
        (1u8, SPATIAL_LAYERS_0_AND_1_IDC),
        (0u8, SPATIAL_LAYER_0_IDC),
    ] {
        let (w, h) = sizes[usize::from(max_s)];
        for max_t in (0..temporal_layers).rev() {
            let t_mask = (1u16 << (max_t + 1)) - 1;
            points.push(OperatingPointSpec {
                idc: s_idc | t_mask,
                max_spatial_id: max_s,
                max_temporal_id: max_t,
                width: w,
                height: h,
                frame_rate: subset_rate(max_t),
            });
        }
    }
    points
}

/// Planned coding metadata for one spatial layer within a temporal unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialLayerPlan {
    /// Spatial layer (`0` = base, `1` = first enhancement, `2` = top enhancement).
    pub spatial_id: u8,
    /// `temporal_id` carried in `obu_extension_header()` (`AV1 §5.3.3`).
    pub temporal_id: u8,
    /// Monotone index among coded layer pictures across the whole stream.
    pub picture_index: u64,
    /// Layer-local coded picture index (`0, 1, 2, ...` within this spatial layer).
    pub layer_frame_index: u64,
    /// Engine frame type (`Key` for `S0` keys and `S1`/`S2` `IntraOnly`; `Inter` otherwise).
    pub frame_type: FrameType,
    /// True when `frame_type == Key` represents an AV1 `INTRA_ONLY_FRAME`
    /// (`frame_type = 2`, refreshing only `references.refresh`, `AV1 §5.9.2`).
    pub intra_only: bool,
    /// True when this enhancement picture (`spatial_id >= 1`) resets its own
    /// temporal reference chain on a non-key repair temporal unit.
    pub enhancement_repair: bool,
    /// Reference slot links and refresh mask in the 8-slot DPB (`0..8`).
    pub references: FrameReferences,
    /// Earlier coded `picture_index` values this picture predicts from.
    pub referenced_pictures: ReferencedFrames,
}

/// Planned coding schedule for one multi-layer temporal unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalUnitPlan {
    /// Capture-order index among coded temporal units.
    pub tu_index: u64,
    /// Whether this temporal unit starts a random-access GOP (`SequenceHeader` + `S0` key).
    pub key_unit: bool,
    /// Lowest spatial layer (`1` or `2`) repaired on this non-key TU, if any.
    pub repair_from_layer: Option<u8>,
    /// Base spatial layer (`spatial_id = 0`), always present in a coded TU.
    pub base: SpatialLayerPlan,
    /// Enhancement layers `spatial_id = 1..layers` (index `spatial_id - 1`),
    /// each present when scheduled and not dropped.
    pub upper: [Option<SpatialLayerPlan>; MAX_SPATIAL_LAYERS - 1],
}

impl TemporalUnitPlan {
    /// The plan of `spatial_id`, if that layer is coded in this unit.
    #[must_use]
    pub fn layer(&self, spatial_id: u8) -> Option<&SpatialLayerPlan> {
        match spatial_id {
            0 => Some(&self.base),
            s => self.upper.get(usize::from(s) - 1)?.as_ref(),
        }
    }

    /// Iterate over all coded layer plans in ascending `spatial_id` order.
    pub fn layer_plans(&self) -> impl Iterator<Item = &SpatialLayerPlan> {
        std::iter::once(&self.base).chain(self.upper.iter().flatten())
    }
}

/// Recent non-reference picture or TU indices remembered for loss-repair filtering.
const NON_REFERENCE_MEMORY: usize = 256;

/// Stateful scheduler and slot tracker for two- and three-layer spatial SVC,
/// testable in isolation without a codec backend.
#[derive(Debug, Clone)]
pub struct SpatialStructure {
    num_layers: u8,
    temporal: [TemporalStructure; MAX_SPATIAL_LAYERS],
    temporal_layers: u8,
    enhancement_interval: u32,
    inter_layer: bool,
    s1_min_temporal_id: u8,
    tu_index: u64,
    last_key_tu: u64,
    next_picture_index: u64,
    coded_counts: [u64; MAX_SPATIAL_LAYERS],
    slot_picture: [Option<u64>; MAX_REFERENCE_SLOTS],
    repaired_from: u64,
    non_reference: VecDeque<u64>,
    layer_repaired_from_tu: [u64; MAX_SPATIAL_LAYERS],
    layer_non_reference_tus: [VecDeque<u64>; MAX_SPATIAL_LAYERS],
    repair_from_layer: Option<u8>,
}

impl SpatialStructure {
    /// Create an `N`-layer (`2` or `3`) spatial scheduler.
    #[must_use]
    pub fn with_layers(
        num_layers: u8,
        temporal_layers: u8,
        enhancement_interval: u32,
        inter_layer: bool,
        adaptive_rule: Option<crate::structure::AdaptiveRule>,
    ) -> Self {
        let num_layers = num_layers.clamp(2, MAX_SPATIAL_LAYERS as u8);
        let interval = if num_layers == 2 {
            enhancement_interval.max(1)
        } else {
            1
        };
        let enh_temporal_layers = if interval == 1 { temporal_layers } else { 1 };
        let make_ts = |tl: u8| match adaptive_rule {
            Some(rule) if interval == 1 => TemporalStructure::adaptive(tl, rule),
            _ => TemporalStructure::new(tl),
        };
        Self {
            num_layers,
            temporal: [
                make_ts(temporal_layers),
                make_ts(enh_temporal_layers),
                make_ts(enh_temporal_layers),
            ],
            temporal_layers,
            enhancement_interval: interval,
            inter_layer,
            s1_min_temporal_id: decimated_enhancement_temporal_id(
                temporal_layers,
                interval,
                inter_layer,
            ),
            tu_index: 0,
            last_key_tu: 0,
            next_picture_index: 0,
            coded_counts: [0; MAX_SPATIAL_LAYERS],
            slot_picture: [None; MAX_REFERENCE_SLOTS],
            repaired_from: 0,
            non_reference: VecDeque::with_capacity(NON_REFERENCE_MEMORY),
            layer_repaired_from_tu: [0; MAX_SPATIAL_LAYERS],
            layer_non_reference_tus: std::array::from_fn(|_| {
                VecDeque::with_capacity(NON_REFERENCE_MEMORY)
            }),
            repair_from_layer: None,
        }
    }

    /// Configured spatial layer count (`2` or `3`).
    #[must_use]
    pub const fn num_layers(&self) -> u8 {
        self.num_layers
    }

    /// Configured temporal layer count on the base layer (`1..=3`).
    #[must_use]
    pub const fn temporal_layers(&self) -> u8 {
        self.temporal_layers
    }

    /// Whether adaptive temporal gating reads repeat detection on the base analyzer.
    #[must_use]
    pub fn reads_repeats(&self) -> bool {
        self.temporal[0].reads_repeats()
    }

    /// Whether losing `picture_index` breaks a later temporal unit's references.
    #[must_use]
    pub fn needs_repair(&self, picture_index: u64) -> bool {
        picture_index >= self.repaired_from
            && picture_index < self.next_picture_index
            && !self.non_reference.contains(&picture_index)
    }

    /// Whether losing `spatial_id` (`0..num_layers`) on `tu_index` breaks future
    /// references of `spatial_id` and is not already covered by an earlier or
    /// pending repair.
    #[must_use]
    pub fn needs_layer_repair(&self, spatial_id: u8, tu_index: u64) -> bool {
        let s = usize::from(spatial_id);
        if s >= usize::from(self.num_layers) {
            return false;
        }
        if spatial_id > 0 && self.repair_from_layer.is_some_and(|r| r <= spatial_id) {
            return false;
        }
        tu_index >= self.layer_repaired_from_tu[s]
            && tu_index < self.tu_index
            && !self.layer_non_reference_tus[s].contains(&tu_index)
    }

    /// Schedule an enhancement-only repair from `spatial_id` (`1` or `2`) upward
    /// on the next coded temporal unit: layers `< spatial_id` code as ordinary
    /// inter frames, while layers `>= spatial_id` reset their temporal slots
    /// predicting only from the layer immediately below (`inter_layer = 1`) or
    /// `INTRA_ONLY_FRAME` (`inter_layer = 0`).
    pub fn request_layer_repair(&mut self, spatial_id: u8) {
        if spatial_id >= 1 && spatial_id < self.num_layers {
            self.repair_from_layer = Some(
                self.repair_from_layer
                    .map_or(spatial_id, |cur| cur.min(spatial_id)),
            );
        }
    }

    /// Whether any enhancement-layer repair (`S1` or `S2`) is pending.
    #[must_use]
    pub const fn enhancement_repair_pending(&self) -> bool {
        self.repair_from_layer.is_some()
    }

    /// Next temporal unit's index.
    #[must_use]
    pub const fn next_tu_index(&self) -> u64 {
        self.tu_index
    }

    /// Coded-frame count of `spatial_id` (`0..num_layers`) so far.
    #[must_use]
    pub const fn layer_coded_count(&self, spatial_id: u8) -> u64 {
        let s = spatial_id as usize;
        if s < MAX_SPATIAL_LAYERS {
            self.coded_counts[s]
        } else {
            0
        }
    }

    /// Whether `spatial_id` (`0..num_layers`) is scheduled in the next temporal
    /// unit if coded (`key_unit` and a pending repair always schedule all layers).
    #[must_use]
    pub const fn layer_due(&self, spatial_id: u8, key_unit: bool) -> bool {
        if spatial_id >= self.num_layers {
            return false;
        }
        if spatial_id == 0 || self.num_layers == 3 {
            return true;
        }
        if key_unit || self.tu_index == 0 || self.repair_from_layer.is_some() {
            return true;
        }
        (self.tu_index - self.last_key_tu).is_multiple_of(self.enhancement_interval as u64)
    }

    /// Preview the base layer's references for the next temporal unit (for
    /// pre-analysis / scene-cut detection before committing the TU).
    #[must_use]
    pub fn peek_base(&self, key_unit: bool) -> FrameReferences {
        let key = key_unit || self.tu_index == 0;
        if !key && self.repair_from_layer.is_some() && self.temporal_layers > 1 {
            let mut refs = self.temporal[0].peek(self.coded_counts[0], true);
            refs.refresh = BASE_TEMPORAL_SLOTS;
            return refs;
        }
        self.temporal[0].peek(self.coded_counts[0], key)
    }

    /// Preview `spatial_id`'s temporal references (`spatial_id >= 1`, with slots
    /// shifted into `layer_temporal_slots(spatial_id)`).
    #[must_use]
    pub fn peek_layer_temporal(&self, spatial_id: u8) -> FrameReferences {
        let s = usize::from(spatial_id).min(MAX_SPATIAL_LAYERS - 1);
        shift_references(
            self.temporal[s].peek(self.coded_counts[s], false),
            layer_slot_offset(spatial_id),
        )
    }

    /// Whether `spatial_id` (`0..num_layers`) may be dropped by rate control on
    /// this non-key TU without breaking later references in `spatial_id`.
    #[must_use]
    pub fn layer_is_droppable(&self, spatial_id: u8, key_unit: bool) -> bool {
        let s = usize::from(spatial_id);
        s < usize::from(self.num_layers)
            && !key_unit
            && self.repair_from_layer.is_none()
            && self.tu_index > 0
            && self.temporal[s].next_is_droppable()
    }

    /// Protected frames after `spatial_id`'s next picture in its temporal structure.
    #[must_use]
    pub fn layer_protected_after_next(&self, spatial_id: u8) -> u32 {
        let s = usize::from(spatial_id).min(MAX_SPATIAL_LAYERS - 1);
        self.temporal[s].protected_after_next()
    }

    /// Feed motion observation into adaptive temporal gating across all layers.
    pub fn observe_motion(&mut self, motion: crate::structure::MotionSignal) {
        for s in 0..usize::from(self.num_layers) {
            self.temporal[s].observe(motion);
        }
    }

    /// Drop the entire temporal unit (only valid when `!key_unit`, all upper
    /// layers are absent or dropped, and [`Self::layer_is_droppable`]`(0)`): returns
    /// the dropped `temporal_id`.
    pub fn drop_temporal_unit(&mut self) -> u8 {
        let dropped_tu = self.tu_index;
        for s in 0..usize::from(self.num_layers) {
            push_bounded(&mut self.layer_non_reference_tus[s], dropped_tu);
        }
        for s in 1..usize::from(self.num_layers) {
            if self.layer_due(s as u8, false) && self.enhancement_interval == 1 {
                let _ = self.temporal[s].drop_next();
            }
        }
        let layer = self.temporal[0].drop_next().unwrap_or(0);
        self.tu_index += 1;
        layer
    }

    /// Advance the `N`-layer schedule by one coded temporal unit, keeping at most
    /// `kept_layers` (`1..=num_layers`) spatial layers when rate control drops
    /// upper layers from the top down (`S2` first, then `S1`).
    pub fn next_temporal_unit_layers(
        &mut self,
        key_unit: bool,
        kept_layers: u8,
    ) -> TemporalUnitPlan {
        let key = key_unit || self.tu_index == 0;
        let repair_from = if key {
            self.repair_from_layer = None;
            self.last_key_tu = self.tu_index;
            None
        } else {
            self.repair_from_layer.take()
        };
        let n = usize::from(self.num_layers);
        let kept = kept_layers.clamp(1, self.num_layers);
        let mut code_layer = [false; MAX_SPATIAL_LAYERS];
        code_layer[0] = true;
        for (s, coded) in code_layer.iter_mut().enumerate().take(n).skip(1) {
            let sid = s as u8;
            let due = repair_from.is_some() || self.layer_due(sid, key);
            let can_drop = sid >= kept
                && !key
                && repair_from.is_none()
                && self.temporal[s].next_is_droppable();
            *coded = due && !can_drop;
        }
        // Never code a higher spatial layer when the layer below it was dropped.
        for s in 2..n {
            if !code_layer[s - 1] {
                code_layer[s] = false;
            }
        }
        for (s, &coded) in code_layer.iter().enumerate().take(n).skip(1) {
            let due = repair_from.is_some() || self.layer_due(s as u8, key);
            if due && !coded && self.enhancement_interval == 1 {
                let _ = self.temporal[s].drop_next();
            }
        }

        let any_repair = repair_from.is_some();
        let (base, mut lower_slot) = self.advance_base_layer(key, any_repair, code_layer[1]);
        let mut lower_temporal_id = base.temporal_id;
        let mut upper = [None; MAX_SPATIAL_LAYERS - 1];
        // Upper layers code in order while the layer below them was coded
        // (`code_layer` is already closed downward).
        for s in 1..n {
            if !code_layer[s] {
                break;
            }
            let sid = s as u8;
            let mode = match repair_from {
                _ if key => UpperLayerMode::KeyUnit,
                Some(r) if r <= sid => UpperLayerMode::LayerRepair,
                // A repair above this layer restarts the temporal hierarchy
                // below it (`advance_base_layer`), so this intact layer
                // restarts too, keeping its operating points closed.
                Some(_) if self.temporal_layers > 1 => UpperLayerMode::RestartIntactTemporal,
                _ => UpperLayerMode::Normal,
            };
            let (plan, slot_for_next) = self.advance_upper_layer(UpperLayerAdvance {
                spatial_id: sid,
                mode,
                code_next_upper: s + 1 < n && code_layer[s + 1],
                lower_temporal_id,
                lower_slot,
            });
            lower_slot = slot_for_next;
            lower_temporal_id = plan.temporal_id;
            upper[s - 1] = Some(plan);
        }

        let current_tu = self.tu_index;
        self.record_repair_tracking(current_tu, key, repair_from, &base, &upper);
        self.tu_index += 1;
        TemporalUnitPlan {
            tu_index: current_tu,
            key_unit: key,
            repair_from_layer: repair_from,
            base,
            upper,
        }
    }

    fn advance_base_layer(
        &mut self,
        key: bool,
        any_repair: bool,
        code_s1: bool,
    ) -> (SpatialLayerPlan, usize) {
        let base_layer_index = self.coded_counts[0];
        self.coded_counts[0] += 1;
        let restart_base = key || (any_repair && self.temporal_layers > 1);
        let mut base_refs = self.temporal[0].next(base_layer_index, restart_base);
        if key {
            base_refs.refresh = SlotMask::ALL;
        } else if any_repair && self.temporal_layers > 1 {
            base_refs.refresh = BASE_TEMPORAL_SLOTS;
        } else if code_s1 && self.inter_layer && base_refs.refresh.is_empty() {
            // Disposable S0 frame needed as S1's inter-layer reference in this TU:
            // place it in S0's dedicated scratch slot (4 in 2-layer, 6 in 3-layer)
            // so slots 0..=1 stay untouched.
            let scratch = inter_layer_scratch_slot(self.num_layers, 0);
            base_refs.refresh = SlotMask::only(scratch);
        }
        let base_picture_index = self.next_picture_index;
        self.next_picture_index += 1;
        let base_referenced = if key {
            ReferencedFrames::NONE
        } else {
            ReferencedFrames::from_pair(self.slot_picture[base_refs.reference.slot], None)
        };
        for (slot, held) in self.slot_picture.iter_mut().enumerate() {
            if key || base_refs.refresh.contains(slot) {
                *held = Some(base_picture_index);
            }
        }
        let base_slot_for_s1 = if key {
            BASE_SLOT_OFFSET
        } else {
            first_slot(base_refs.refresh).unwrap_or(BASE_SLOT_OFFSET)
        };
        let plan = SpatialLayerPlan {
            spatial_id: 0,
            temporal_id: base_refs.position.layer,
            picture_index: base_picture_index,
            layer_frame_index: base_layer_index,
            frame_type: if key {
                FrameType::Key
            } else {
                FrameType::Inter
            },
            intra_only: false,
            enhancement_repair: false,
            references: base_refs,
            referenced_pictures: base_referenced,
        };
        (plan, base_slot_for_s1)
    }

    fn advance_upper_layer(&mut self, req: UpperLayerAdvance) -> (SpatialLayerPlan, usize) {
        let UpperLayerAdvance {
            spatial_id,
            mode,
            code_next_upper,
            lower_temporal_id,
            lower_slot,
        } = req;
        let s = usize::from(spatial_id);
        let offset = layer_slot_offset(spatial_id);
        let layer_reset = matches!(mode, UpperLayerMode::KeyUnit | UpperLayerMode::LayerRepair);
        let restart_temporal = layer_reset || matches!(mode, UpperLayerMode::RestartIntactTemporal);
        let layer_index = self.coded_counts[s];
        self.coded_counts[s] += 1;
        let raw = self.temporal[s].next(layer_index, restart_temporal);
        let mut shifted_refresh = if restart_temporal {
            layer_temporal_slots(spatial_id)
        } else {
            shift_mask(raw.refresh, offset)
        };
        if !layer_reset && code_next_upper && self.inter_layer && shifted_refresh.is_empty() {
            // Disposable (`T2`) picture the layer above predicts from in this
            // TU: place it in this layer's scratch slot so its own `T0`/`T1`
            // slots stay untouched.
            let scratch = inter_layer_scratch_slot(self.num_layers, spatial_id);
            shifted_refresh = SlotMask::only(scratch);
        }
        let temporal_id = if spatial_id == 1 && self.enhancement_interval > 1 {
            self.s1_min_temporal_id.max(if self.inter_layer {
                lower_temporal_id
            } else {
                0
            })
        } else {
            raw.position.layer
        };
        let position = LayerPosition {
            layer: temporal_id,
            ..raw.position
        };
        let interval = if spatial_id == 1 {
            self.enhancement_interval
        } else {
            1
        };
        let (frame_type, intra_only, references, referenced_pictures) = self
            .build_upper_layer_references(
                position,
                raw.reference,
                shifted_refresh,
                offset,
                interval,
                lower_slot,
                layer_reset,
            );
        let pic_index = self.next_picture_index;
        self.next_picture_index += 1;
        for (slot, held) in self.slot_picture.iter_mut().enumerate() {
            if references.refresh.contains(slot) {
                *held = Some(pic_index);
            }
        }
        let slot_for_next = first_slot(references.refresh).unwrap_or(offset);
        let plan = SpatialLayerPlan {
            spatial_id,
            temporal_id,
            picture_index: pic_index,
            layer_frame_index: layer_index,
            frame_type,
            intra_only,
            enhancement_repair: matches!(mode, UpperLayerMode::LayerRepair),
            references,
            referenced_pictures,
        };
        (plan, slot_for_next)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_upper_layer_references(
        &self,
        position: LayerPosition,
        raw_ref: ReferenceLink,
        shifted_refresh: SlotMask,
        offset: usize,
        interval: u32,
        lower_slot: usize,
        layer_reset: bool,
    ) -> (FrameType, bool, FrameReferences, ReferencedFrames) {
        if layer_reset && self.inter_layer {
            let refs = FrameReferences {
                position,
                reference: ReferenceLink {
                    slot: lower_slot,
                    distance: 1,
                },
                refresh: shifted_refresh,
                secondary: None,
            };
            let dep = ReferencedFrames::from_pair(self.slot_picture[lower_slot], None);
            (FrameType::Inter, false, refs, dep)
        } else if layer_reset {
            let refs = FrameReferences {
                position,
                reference: ReferenceLink {
                    slot: offset,
                    distance: 1,
                },
                refresh: shifted_refresh,
                secondary: None,
            };
            (FrameType::Key, true, refs, ReferencedFrames::NONE)
        } else {
            let primary_slot = raw_ref.slot + offset;
            let primary_distance = raw_ref.distance.saturating_mul(interval);
            let secondary = self.inter_layer.then_some(ReferenceLink {
                slot: lower_slot,
                distance: 0,
            });
            let refs = FrameReferences {
                position,
                reference: ReferenceLink {
                    slot: primary_slot,
                    distance: primary_distance,
                },
                refresh: shifted_refresh,
                secondary,
            };
            let dep = ReferencedFrames::from_pair(
                self.slot_picture[primary_slot],
                secondary.and_then(|sec| self.slot_picture[sec.slot]),
            );
            (FrameType::Inter, false, refs, dep)
        }
    }

    fn record_repair_tracking(
        &mut self,
        current_tu: u64,
        key: bool,
        repair_from: Option<u8>,
        base: &SpatialLayerPlan,
        upper: &[Option<SpatialLayerPlan>; MAX_SPATIAL_LAYERS - 1],
    ) {
        let n = usize::from(self.num_layers);
        if key {
            self.repaired_from = base.picture_index;
            self.layer_repaired_from_tu[..n].fill(current_tu);
        } else if let Some(r) = repair_from {
            self.layer_repaired_from_tu[usize::from(r)..n].fill(current_tu);
        }
        let plans = [Some(base), upper[0].as_ref(), upper[1].as_ref()];
        let mut layer_ref = [false; MAX_SPATIAL_LAYERS];
        layer_ref[0] = key || (base.references.refresh.0 & BASE_TEMPORAL_SLOTS.0) != 0;
        for (s, (&plan, is_ref)) in plans
            .iter()
            .zip(layer_ref.iter_mut())
            .enumerate()
            .take(n)
            .skip(1)
        {
            let mask = layer_temporal_slots(s as u8);
            *is_ref = plan.is_some_and(|p| (p.references.refresh.0 & mask.0) != 0);
        }
        for (non_ref, &is_ref) in self
            .layer_non_reference_tus
            .iter_mut()
            .zip(layer_ref.iter())
            .take(n)
        {
            if !is_ref {
                push_bounded(non_ref, current_tu);
            }
        }
        let mut pic_ref = [false; MAX_SPATIAL_LAYERS];
        for s in (0..n).rev() {
            pic_ref[s] = plans[s].is_some()
                && (layer_ref[s] || (self.inter_layer && s + 1 < n && pic_ref[s + 1]));
        }
        for (&plan, &is_pic_ref) in plans.iter().zip(pic_ref.iter()).take(n) {
            if let (Some(p), false) = (plan, is_pic_ref) {
                push_bounded(&mut self.non_reference, p.picture_index);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum UpperLayerMode {
    Normal,
    RestartIntactTemporal,
    LayerRepair,
    KeyUnit,
}

#[derive(Clone, Copy)]
struct UpperLayerAdvance {
    spatial_id: u8,
    mode: UpperLayerMode,
    code_next_upper: bool,
    lower_temporal_id: u8,
    lower_slot: usize,
}

fn push_bounded(queue: &mut VecDeque<u64>, value: u64) {
    if queue.len() == NON_REFERENCE_MEMORY {
        queue.pop_front();
    }
    queue.push_back(value);
}

// Start-up settings of the enhancement controller ([`split_rate_control`]),
// tuned by Gemini so `TU 0` and `TU 1` bootstrap at a balanced step ratio
// (an unscaled start left `TU 0` `S1` above qindex 240); not measured one by
// one.

/// Bits per pixel a controller starts from when the stream sets none.
const DEFAULT_INITIAL_BITS_PER_PIXEL: f64 = 0.03;
/// Every-unit inter-layer enhancement: its key boost is this share of the stream's...
const ENHANCEMENT_KEY_BOOST_SCALE: f64 = 0.5;
/// ...its initial key complexity this share...
const ENHANCEMENT_KEY_COMPLEXITY_SCALE: f64 = 0.35;
/// ...and its initial bits per pixel this share.
const ENHANCEMENT_BPP_SCALE: f64 = 0.60;
/// A decimated `S1` (every `N`-th unit) starts at `N` to this power times
/// the base bits per pixel: its reference is `N` units old.
const DECIMATED_BPP_EXPONENT: f64 = 0.85;

/// Per-layer CBR controller configuration produced by [`split_rate_control`].
#[derive(Debug, Clone, Copy)]
pub struct LayerRateBudget {
    /// Layer CBR rate-control settings.
    pub config: RateControlConfig,
    /// Layer nominal frame rate (`frame_rate / N` for decimated enhancement).
    pub frame_rate: FrameRate,
    /// Layer intra period in layer frames.
    pub intra_period: u32,
}

/// Split a stream CBR `RateControlConfig` across the spatial layers'
/// controllers at `shares` ([`SpatialConfig::layer_rate_shares`]): each lower
/// layer gets its rounded share (at least `1` bps, leaving `1` bps for every
/// layer above it) and the top layer the exact remainder, so the targets sum
/// to `total.target_bps`.
///
/// # Errors
///
/// Returns [`RateControlError`] if `total.target_bps` is below one bit per
/// second per layer or `buffer_ms == 0`.
pub fn split_rate_control(
    total: RateControlConfig,
    shares: &[f64],
    frame_rate: FrameRate,
    intra_period: u32,
    enhancement_interval: u32,
    inter_layer: bool,
) -> Result<Vec<LayerRateBudget>, RateControlError> {
    let num_layers = shares.len();
    if total.target_bps < num_layers as u64 {
        return Err(RateControlError(
            "spatial CBR target_bps must be at least 1 bps per spatial layer",
        ));
    }
    if total.buffer_ms == 0 {
        return Err(RateControlError("buffer_ms must be non-zero"));
    }
    let mut layer_bps = Vec::with_capacity(num_layers);
    let mut remaining = total.target_bps;
    for (s, &share) in shares.iter().enumerate() {
        let layers_above = (num_layers - 1 - s) as u64;
        let bps = if layers_above == 0 {
            remaining
        } else {
            ((total.target_bps as f64 * share).round() as u64).clamp(1, remaining - layers_above)
        };
        remaining -= bps;
        layer_bps.push(bps);
    }

    let split_peak = |bps: u64| {
        total.peak_bps.map(|peak| {
            let ratio = bps as f64 / total.target_bps as f64;
            ((peak as f64 * ratio).round() as u64).max(bps)
        })
    };
    let base_bpp = total
        .initial_bits_per_pixel
        .unwrap_or(DEFAULT_INITIAL_BITS_PER_PIXEL);
    let top = num_layers - 1;
    Ok(layer_bps
        .into_iter()
        .enumerate()
        .map(|(s, bps)| {
            let mut cfg = total;
            cfg.target_bps = bps;
            cfg.peak_bps = split_peak(bps);
            // Only `S1` of a two-layer stream is decimated (validated config).
            let interval = if s == 1 {
                enhancement_interval.max(1)
            } else {
                1
            };
            if s > 0 && inter_layer {
                if interval == 1 {
                    let key_boost_scale = if s == top {
                        ENHANCEMENT_KEY_BOOST_SCALE
                    } else {
                        MIDDLE_LAYER_KEY_BOOST_SCALE
                    };
                    cfg.keyframe_boost = (total.keyframe_boost * key_boost_scale).max(1.0);
                    cfg.initial_key_complexity =
                        (total.initial_key_complexity * ENHANCEMENT_KEY_COMPLEXITY_SCALE).max(1.0);
                    cfg.initial_bits_per_pixel = Some(base_bpp * ENHANCEMENT_BPP_SCALE);
                } else {
                    let n = f64::from(interval);
                    cfg.keyframe_boost = 1.0;
                    cfg.initial_key_complexity = 1.0;
                    cfg.initial_bits_per_pixel = Some(base_bpp * n.powf(DECIMATED_BPP_EXPONENT));
                }
            }
            LayerRateBudget {
                config: cfg,
                frame_rate: FrameRate {
                    num: frame_rate.num,
                    den: frame_rate.den.saturating_mul(interval),
                },
                intra_period: intra_period.div_ceil(interval).max(1),
            }
        })
        .collect())
}

/// Keyframe boost scale of a middle enhancement layer (`S1` of three, `360p`;
/// the top layer takes [`ENHANCEMENT_KEY_BOOST_SCALE`], `0.5`): `S0` already
/// codes the intra `KEY_FRAME` and `S2` follows in the same TU; `0.65` spent
/// `42.4%` (`156 kb`) of `S1`'s 5 s budget on `TU 0` on `pool_wildtrack_cam1`
/// at `500 kbps` (Gemini, av1-svc3).
const MIDDLE_LAYER_KEY_BOOST_SCALE: f64 = 0.45;

/// Minimum adaptive `S0` share of the total stream bitrate on static inter
/// temporal units in two-layer mode when `spatial.base_rate_share` acts as a cap.
const MIN_ADAPTIVE_BASE_SHARE: f64 = 0.15;

/// In three-layer mode, `S0` (`180p`) static inter-frame share floor as a
/// fraction of `s0_rate_share` (`0.50`: passes unused `180p` bits upward on
/// static scenes).
const THREE_LAYER_S0_STATIC_FLOOR_RATIO: f64 = 0.50;
/// In three-layer mode, `S0` (`180p`) moving-scene inter-frame share floor as a
/// fraction of `s0_rate_share` (`0.82`: keeps `S0` P-frames matched to its key
/// budget; `0.45` starved `wildtrack` `S0` P-frames to `408` b vs `619` b
/// standalone `180p`, costing `+12%` BD-rate on `op 2`).
const THREE_LAYER_S0_MOVING_FLOOR_RATIO: f64 = 0.82;
/// In three-layer mode, `S1` (`360p`) static inter-frame share floor as a
/// fraction of `s1_rate_share` (`0.70`: passes unused `360p` bits to `S2`).
const THREE_LAYER_S1_STATIC_FLOOR_RATIO: f64 = 0.70;
/// In three-layer mode, `S1` (`360p`) moving-scene inter-frame share floor as a
/// fraction of `s1_rate_share` (`0.82`: preserves `360p` OP quality when `S2`
/// mostly uses temporal prediction; `0.45` starved `wildtrack` `S1` to `1422` b/f).
const THREE_LAYER_S1_MOVING_FLOOR_RATIO: f64 = 0.82;
/// In three-layer mode, low-bitrate guard raises a lower layer's share floor by
/// up to this fraction of its cap (`0.18`).
const THREE_LAYER_LOW_RATE_RAISE_RATIO: f64 = 0.18;
/// Moving-scene quantizer lead allowed for `180p` `S0` below `360p` `S1` in
/// three-layer mode (`32` qindex; `14` clamped `320x180` P-frames to `360p`-like
/// quantizers at `1/4` the area).
const THREE_LAYER_S0_MOVING_LEAD: f64 = 32.0;
/// Extra static-scene quantizer lead for `180p` `S0` below `360p` `S1` (`+8` qindex).
const THREE_LAYER_S0_STATIC_EXTRA_LEAD: f64 = 8.0;
/// Moving-scene quantizer lead allowed for `360p` `S1` below `720p` `S2` in
/// three-layer mode (`20` qindex).
const THREE_LAYER_S1_MOVING_LEAD: f64 = 20.0;

/// Nominal lower-layer bits/pixel/frame threshold below which the adaptive
/// share floor is raised to guard lower operating points at low rungs.
const LOW_RATE_GUARD_BPP: f64 = 0.015;

/// Initial EMA estimate of an enhancement layer's inter-layer reference area
/// fraction before its first dual-reference inter frame is observed.
const INITIAL_INTER_LAYER_HIT_RATE: f64 = 0.05;

/// EMA smoothing weight for an enhancement layer's inter-layer hit rate.
const INTER_LAYER_HIT_EMA: f64 = 0.35;

/// A temporal unit is a static scene from this static fraction on.
const STATIC_SCENE_FRACTION: f32 = 0.50;
/// Enhancement inter-layer hit rate at which reliance on the lower layer starts...
const RELIANCE_HIT_RATE_START: f64 = 0.06;
/// ...and the span over which it grows to full reliance.
const RELIANCE_HIT_RATE_SPAN: f64 = 0.10;
/// Weight of the active (non-static) area in a static scene's lower-layer share.
const STATIC_ACTIVITY_WEIGHT: f64 = 1.4;
/// Weight of the active area in a moving scene's lower-layer share...
const MOVING_ACTIVITY_WEIGHT: f64 = 0.35;
/// ...and of the upper layer's reliance on the lower layer.
const MOVING_RELIANCE_WEIGHT: f64 = 0.85;
/// Below [`LOW_RATE_GUARD_BPP`], the two-layer share floor rises by up to this...
const LOW_RATE_SHARE_RAISE: f64 = 0.08;
/// ...the active area's weight by up to this on static scenes...
const STATIC_LOW_RATE_ACTIVITY: f64 = 0.25;
/// ...and on moving ones by up to this.
const MOVING_LOW_RATE_ACTIVITY: f64 = 0.20;
/// A lower layer's budget never drops below this share of its nominal budget.
const MIN_BASE_BUDGET_SCALE: f64 = 0.1;
/// A lower layer's quantizer is floored near the upper layer's only while
/// reliance on it stays below this.
const RELIANCE_FLOOR_LIMIT: f64 = 0.50;
/// Most qindex units a lower layer may code finer than the layer above on a static scene...
const STATIC_MAX_LEAD: f64 = 16.0;
/// ...and on a moving scene.
const MOVING_BASE_LEAD: f64 = 14.0;
/// Further lead at full reliance...
const RELIANCE_LEAD: f64 = 12.0;
/// ...and at the deepest low-rate guard.
const LOW_RATE_LEAD: f64 = 6.0;

/// Combined temporal-unit CBR bucket (`S0 + S1 [+ S2]` together) drained at the
/// total target bitrate, enforcing stream-level buffer compliance, tracking peak
/// temporal-unit size, and deciding the single-place upward `S0 -> S1 [-> S2]`
/// rate split ([`Self::plan_share`]).
#[derive(Debug, Clone)]
pub struct CombinedCbrBuffer {
    num_layers: u8,
    target_bps: u64,
    capacity_bits: f64,
    occupancy_bits: f64,
    nominal_dt: f64,
    layer_share_caps: [f64; MAX_SPATIAL_LAYERS],
    inter_layer_hit_rates: [f64; MAX_SPATIAL_LAYERS - 1],
    elapsed: f64,
    total_bits: u64,
    tu_count: u64,
    buffer_overflows: u64,
    peak_tu_bits: u64,
}

impl CombinedCbrBuffer {
    /// Create a combined CBR bucket for `config` at `frame_rate` over the
    /// spatial layers' nominal `shares` ([`SpatialConfig::layer_rate_shares`]):
    /// each lower layer's share is its cap, and the top layer takes the rest
    /// plus the lower layers' unspent bits.
    #[must_use]
    pub fn with_share_caps(
        config: RateControlConfig,
        frame_rate: FrameRate,
        shares: &[f64],
    ) -> Self {
        let capacity_bits =
            (config.target_bps as f64 * f64::from(config.buffer_ms) / 1000.0).max(1.0);
        let nominal_dt = f64::from(frame_rate.den.max(1)) / f64::from(frame_rate.num.max(1));
        let mut layer_share_caps = [0.0; MAX_SPATIAL_LAYERS];
        layer_share_caps[..shares.len()].copy_from_slice(shares);
        Self {
            num_layers: shares.len() as u8,
            target_bps: config.target_bps,
            capacity_bits,
            occupancy_bits: 0.0,
            nominal_dt,
            layer_share_caps,
            inter_layer_hit_rates: [INITIAL_INTER_LAYER_HIT_RATE; MAX_SPATIAL_LAYERS - 1],
            elapsed: 0.0,
            total_bits: 0,
            tu_count: 0,
            buffer_overflows: 0,
            peak_tu_bits: 0,
        }
    }

    /// Reconfigure target bitrate and buffer duration mid-stream.
    pub fn reconfigure(&mut self, config: RateControlConfig) {
        self.target_bps = config.target_bps;
        self.capacity_bits =
            (config.target_bps as f64 * f64::from(config.buffer_ms) / 1000.0).max(1.0);
        if self.occupancy_bits > self.capacity_bits {
            self.buffer_overflows += 1;
            self.occupancy_bits = self.capacity_bits;
        }
    }

    /// Update the smoothed inter-layer hit rate for enhancement step `step`
    /// (`0` = `S1` from `S0`, `1` = `S2` from `S1`).
    pub fn observe_step_inter_layer_hit_rate(
        &mut self,
        step: usize,
        inter_layer_area: u64,
        total_area: u64,
    ) {
        if total_area == 0 || step >= self.inter_layer_hit_rates.len() {
            return;
        }
        let hit = (inter_layer_area as f64 / total_area as f64).clamp(0.0, 1.0);
        let rate = &mut self.inter_layer_hit_rates[step];
        *rate += INTER_LAYER_HIT_EMA * (hit - *rate);
    }

    fn lower_share_floor(&self, spatial_id: u8) -> (f64, f64) {
        let cap = self.layer_share_caps[usize::from(spatial_id)];
        if self.num_layers <= 2 {
            (MIN_ADAPTIVE_BASE_SHARE.min(cap), LOW_RATE_SHARE_RAISE)
        } else {
            let ratio = if spatial_id == 0 {
                THREE_LAYER_S0_STATIC_FLOOR_RATIO
            } else {
                THREE_LAYER_S1_STATIC_FLOOR_RATIO
            };
            (
                (ratio * cap).min(cap),
                THREE_LAYER_LOW_RATE_RAISE_RATIO * cap,
            )
        }
    }

    /// Adaptive inter-frame share, budget scale factor, and optional finest
    /// quantizer floor for lower spatial layer `spatial_id` (`0` in 2-layer;
    /// `0` or `1` in 3-layer) relative to `spatial_id + 1`.
    #[must_use]
    pub fn plan_lower_layer_inter_constraints(
        &self,
        spatial_id: u8,
        static_fraction: f32,
        layer_pixels: usize,
        upper_inter_level: Option<u32>,
    ) -> (f64, f64, Option<u32>) {
        let s = usize::from(spatial_id);
        let cap = self.layer_share_caps[s];
        let static_scene = static_fraction >= STATIC_SCENE_FRACTION;
        let (static_floor, low_rate_raise) = self.lower_share_floor(spatial_id);
        let min_floor = if self.num_layers > 2 && !static_scene {
            let moving_ratio = if spatial_id == 0 {
                THREE_LAYER_S0_MOVING_FLOOR_RATIO
            } else {
                THREE_LAYER_S1_MOVING_FLOOR_RATIO
            };
            (moving_ratio * cap).min(cap)
        } else {
            static_floor
        };
        let active_fraction = (1.0 - f64::from(static_fraction)).clamp(0.0, 1.0);
        let nominal_bits_per_tu = self.target_bps as f64 * cap * self.nominal_dt;
        let bpp = nominal_bits_per_tu / (layer_pixels.max(1) as f64);
        let low_rate_guard =
            (LOW_RATE_GUARD_BPP / bpp.max(f64::MIN_POSITIVE) - 1.0).clamp(0.0, 1.0);
        let min_share = (min_floor + low_rate_raise * low_rate_guard).min(cap);

        let hit_rate = self.inter_layer_hit_rates[s.min(MAX_SPATIAL_LAYERS - 2)];
        let il_reliance =
            ((hit_rate - RELIANCE_HIT_RATE_START) / RELIANCE_HIT_RATE_SPAN).clamp(0.0, 1.0);
        let activity_weight = if static_scene {
            active_fraction * STATIC_ACTIVITY_WEIGHT
                + il_reliance
                + STATIC_LOW_RATE_ACTIVITY * low_rate_guard * active_fraction
        } else {
            active_fraction * MOVING_ACTIVITY_WEIGHT
                + il_reliance * MOVING_RELIANCE_WEIGHT
                + MOVING_LOW_RATE_ACTIVITY * low_rate_guard * active_fraction
        }
        .clamp(0.0, 1.0);
        let effective_share =
            (min_share + (cap - min_share) * activity_weight).clamp(min_share, cap);
        let budget_scale = (effective_share / cap).clamp(MIN_BASE_BUDGET_SCALE, 1.0);

        let lead = if il_reliance >= RELIANCE_FLOOR_LIMIT {
            None
        } else if static_scene {
            let motion = ((1.0 - f64::from(static_fraction))
                / (1.0 - f64::from(STATIC_SCENE_FRACTION)))
            .clamp(0.0, 1.0);
            let max_lead = if self.num_layers > 2 && spatial_id == 0 {
                STATIC_MAX_LEAD + THREE_LAYER_S0_STATIC_EXTRA_LEAD
            } else {
                STATIC_MAX_LEAD
            };
            Some(max_lead * motion)
        } else if self.num_layers > 2 {
            Some(if spatial_id == 0 {
                THREE_LAYER_S0_MOVING_LEAD
            } else {
                THREE_LAYER_S1_MOVING_LEAD
            })
        } else {
            Some(MOVING_BASE_LEAD)
        };
        let min_q = lead.zip(upper_inter_level).map(|(lead, q_up)| {
            let lead = lead + RELIANCE_LEAD * il_reliance + LOW_RATE_LEAD * low_rate_guard;
            q_up.saturating_sub(lead.round() as u32)
        });
        (effective_share, budget_scale, min_q)
    }

    /// Split unspent `S0` inter bits `u0` between `S1` (`r1`) and direct carry
    /// to `S2` (`u0 - r1`) in three-layer mode when `S2` is coded in the same TU,
    /// proportional to `S1`'s effective share vs `S2`'s share.
    #[must_use]
    pub fn split_s0_reallocation_for_s1(
        &self,
        u0: f64,
        s1_static_fraction: f32,
        s1_pixels: usize,
        s2_inter_level: Option<u32>,
    ) -> (f64, f64) {
        if u0 <= 0.0 || self.num_layers < 3 {
            return (u0.max(0.0), 0.0);
        }
        let (s1_eff, _, _) = self.plan_lower_layer_inter_constraints(
            1,
            s1_static_fraction,
            s1_pixels,
            s2_inter_level,
        );
        let s2_cap = self.layer_share_caps[2];
        let denom = (s1_eff + s2_cap).max(f64::MIN_POSITIVE);
        let r1 = (u0 * (s1_eff / denom)).clamp(0.0, u0);
        (r1, (u0 - r1).max(0.0))
    }

    /// One layer frame's share of the temporal unit: the single place the
    /// combined buffer's rate split is decided across 2 or 3 spatial layers.
    ///
    /// - Any enhancement layer (`spatial_id >= 1`) is bounded by
    ///   [`Self::remaining_tu_room`] after prior layers' spent bits; an inter
    ///   enhancement picture also receives `request.reallocated_bits` forfeited
    ///   by lower layers (`S0 -> S1 -> S2`).
    /// - An inter-layer enhancement key or repair picture (`S1` or `S2`)
    ///   anchors its quantizer to the immediately-lower layer (`S0` for `S1`,
    ///   `S1` for `S2`; [`enhancement_key_anchor`]).
    /// - Every lower layer (`S0` in 2-layer; `S0` and `S1` in 3-layer when the
    ///   layer above it is coded in this TU) gets its static-aware cap
    ///   ([`Self::plan_lower_layer_inter_constraints`]).
    #[must_use]
    pub(crate) fn plan_share(&self, request: &ShareRequest) -> LayerShare {
        let mut share = LayerShare::WHOLE;
        if request.spatial_id > 0 {
            share.room_bits =
                self.remaining_tu_room(request.interval, request.ceiling, request.prior_layer_bits);
            if request.key && request.inter_layer {
                let (anchor, clear_key_loan) = enhancement_key_anchor(self.num_layers, request);
                share.key_anchor = Some(anchor);
                share.clear_key_loan = clear_key_loan;
                return share;
            }
            if !request.key && request.reallocated_bits > 0.0 {
                share.reallocated_bits = request.reallocated_bits;
            }
        }
        if !request.key
            && request.has_coded_upper_layer
            && request.inter_layer
            && request.enhancement_interval == 1
        {
            let upper_level = request.upper_inter_level.unwrap_or(
                request
                    .lower_quantizer
                    .saturating_add(ASSUMED_ENHANCEMENT_QUANTIZER_LEAD),
            );
            let (_, budget_scale, min_q) = self.plan_lower_layer_inter_constraints(
                request.spatial_id,
                request.static_fraction,
                request.layer_pixels,
                Some(upper_level),
            );
            share.budget_scale = budget_scale;
            share.min_inter_quantizer = min_q;
        }
        share
    }

    /// Unspent inter bits of lower layer `spatial_id` (`0` or `1`) to forfeit
    /// and pass upward (`S0 -> S1 -> S2`), including any `incoming_reallocated`
    /// bits `S1` received from `S0` that `S1` itself left unspent, preserving
    /// exact total drain `sum_s drain_s == total_bps * dt`.
    #[must_use]
    pub fn reallocate_unspent_lower_bits(
        &self,
        spatial_id: u8,
        nominal_budget: f64,
        incoming_reallocated: f64,
        actual_bits: u64,
    ) -> f64 {
        let s = usize::from(spatial_id);
        let cap = self.layer_share_caps[s];
        if nominal_budget <= 0.0 || cap <= 0.0 {
            return 0.0;
        }
        let (min_floor, _) = self.lower_share_floor(spatial_id);
        let max_own_forfeit = ((1.0 - min_floor / cap) * nominal_budget).max(0.0);
        let max_reallocate = max_own_forfeit + incoming_reallocated.max(0.0);
        (nominal_budget + incoming_reallocated.max(0.0) - actual_bits as f64)
            .clamp(0.0, max_reallocate)
    }

    /// Remaining bits available in the current temporal unit before the
    /// combined buffer exceeds `ceiling * capacity_bits` after draining
    /// `target_bps * dt` and accounting for `already_spent_tu_bits`.
    #[must_use]
    pub fn remaining_tu_room(
        &self,
        interval: Option<f64>,
        ceiling: f64,
        already_spent_tu_bits: u64,
    ) -> f64 {
        let eff_ceiling = if ceiling > 0.0 { ceiling } else { 1.0 };
        let room_tu = eff_ceiling * self.capacity_bits - self.drained_occupancy(interval);
        (room_tu - already_spent_tu_bits as f64).max(1.0)
    }

    /// Whether adding `tu_bits` on a TU of duration `interval` seconds would
    /// exceed `ceiling * capacity_bits` after draining `target_bps * dt`.
    #[must_use]
    pub fn would_exceed_ceiling(&self, tu_bits: u64, interval: Option<f64>, ceiling: f64) -> bool {
        if ceiling <= 0.0 {
            return false;
        }
        self.drained_occupancy(interval) + tu_bits as f64 > ceiling * self.capacity_bits
    }

    /// Whether `tu_bits` (a prediction, so fractional) on a TU of duration
    /// `interval` would overflow the strict-CBR buffer.
    #[must_use]
    pub fn would_overflow(&self, tu_bits: f64, interval: Option<f64>) -> bool {
        self.drained_occupancy(interval) + tu_bits > self.capacity_bits
    }

    /// Occupancy after a TU of duration `interval` drains at the target rate.
    fn drained_occupancy(&self, interval: Option<f64>) -> f64 {
        let dt = interval.unwrap_or(self.nominal_dt).max(0.0);
        (self.occupancy_bits - self.target_bps as f64 * dt).max(0.0)
    }

    /// Record a coded, dropped (`0` bits), or held (`0` bits) temporal unit.
    pub fn observe_tu(&mut self, tu_bits: u64, interval: Option<f64>, coded: bool) {
        let dt = interval.unwrap_or(self.nominal_dt).max(0.0);
        self.elapsed += dt;
        if coded {
            self.tu_count += 1;
            self.total_bits += tu_bits;
            self.peak_tu_bits = self.peak_tu_bits.max(tu_bits);
        }
        let next = (self.occupancy_bits + tu_bits as f64 - self.target_bps as f64 * dt).max(0.0);
        if next > self.capacity_bits {
            self.buffer_overflows += 1;
            self.occupancy_bits = self.capacity_bits;
        } else {
            self.occupancy_bits = next;
        }
    }

    /// Total target bitrate in bits/second.
    #[must_use]
    pub const fn target_bps(&self) -> u64 {
        self.target_bps
    }

    /// Fraction of the modeled receiver buffer currently occupied (`[0.0, 1.0]`).
    #[must_use]
    pub fn buffer_fullness(&self) -> f64 {
        (self.occupancy_bits / self.capacity_bits).clamp(0.0, 1.0)
    }

    /// Mean achieved bitrate across elapsed stream time.
    #[must_use]
    pub fn actual_bps(&self) -> f64 {
        if self.elapsed > 0.0 {
            self.total_bits as f64 / self.elapsed
        } else {
            0.0
        }
    }

    /// Total bits emitted across all temporal units.
    #[must_use]
    pub const fn total_bits(&self) -> u64 {
        self.total_bits
    }

    /// Number of coded temporal units observed.
    #[must_use]
    pub const fn coded_tus(&self) -> u64 {
        self.tu_count
    }

    /// Number of temporal units that overflowed `capacity_bits`.
    #[must_use]
    pub const fn buffer_overflows(&self) -> u64 {
        self.buffer_overflows
    }

    /// Largest single temporal unit (`S0 + S1 [+ S2]`) in bits observed so far.
    #[must_use]
    pub const fn peak_tu_bits(&self) -> u64 {
        self.peak_tu_bits
    }
}

/// What [`CombinedCbrBuffer::plan_share`] plans one layer frame's share from.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ShareRequest {
    /// Spatial layer (`0..=2`).
    pub spatial_id: u8,
    /// Planned as a key picture (`S0` key, or `S1`/`S2` key or repair picture).
    pub key: bool,
    /// Whether `spatial_id + 1` is also coded in this temporal unit (so
    /// `spatial_id` acts as a capped lower layer).
    pub has_coded_upper_layer: bool,
    /// The frame's index within its layer.
    pub layer_index: u64,
    /// `spatial.inter_layer`.
    pub inter_layer: bool,
    /// `spatial.enhancement_interval`.
    pub enhancement_interval: u32,
    /// This temporal unit's duration (`None`: the nominal one).
    pub interval: Option<f64>,
    /// Share of the combined buffer the unit may fill.
    pub ceiling: f64,
    /// Bits already coded by lower spatial layers `< spatial_id` in this TU.
    pub prior_layer_bits: u64,
    /// Quantizer of `spatial_id - 1` in this TU when planning an enhancement
    /// key/repair picture, or this layer's last quantizer when planning a
    /// lower layer before its upper layer has an `inter_level`.
    pub lower_quantizer: u32,
    /// Unspent bits reallocated upward to this layer on this TU.
    pub reallocated_bits: f64,
    /// Static fraction of the picture being planned.
    pub static_fraction: f32,
    /// Luma samples of this layer's picture.
    pub layer_pixels: usize,
    /// Inter quantizer level of `spatial_id + 1`, once it has coded an inter frame.
    pub upper_inter_level: Option<u32>,
    /// Start-up `(full_res_peak_displacement, static_fraction)` measured on `TU 0`.
    pub startup_profile: Option<(f64, f32)>,
}

/// An upper layer's inter quantizer assumed this far above the lower layer's
/// until the upper layer has coded an inter frame.
const ASSUMED_ENHANCEMENT_QUANTIZER_LEAD: u32 = 16;

/// A decimated `S1` key picture (`enhancement_interval > 1`) may code this
/// much finer than the same-instant `S0` key...
const DECIMATED_KEY_FINER: u32 = 12;
/// ...or this much coarser.
const DECIMATED_KEY_COARSER: u32 = 8;
/// A decimated `S1` with no inter level yet starts it this far above its key.
const DECIMATED_KEY_INTER_SEED: u32 = 8;
/// Start-up profile of a scene that moves from its first frame: a static
/// fraction below this...
const MOVING_STARTUP_MAX_STATIC: f32 = 0.20;
/// ...or a peak camera displacement (full-resolution luma samples per frame)
/// of at least this.
const MOVING_STARTUP_MIN_PEAK: f64 = 1.5;
/// In two-layer mode, a moving start-up's first `S1` codes at least this much
/// coarser than `S0`...
const MOVING_STARTUP_KEY_MIN_LEAD: u32 = 28;
/// ...and at most this much coarser.
const MOVING_STARTUP_KEY_MAX_LEAD: u32 = 56;
/// In three-layer mode, each 2:1 step (`S0 -> S1`, `S1 -> S2`) on a moving
/// start-up adds half the single-step lead (`+16..+36` per step -> `+32..+72`
/// across both steps, avoiding saturating `S2` at q255).
const MOVING_STARTUP_THREE_LAYER_MIN_LEAD: u32 = 16;
const MOVING_STARTUP_THREE_LAYER_MAX_LEAD: u32 = 36;
/// Any other inter-layer enhancement key codes at most this much finer than
/// the layer immediately below it.
const KEY_MAX_FINER_THAN_BASE: u32 = 4;

fn enhancement_key_anchor(num_layers: u8, request: &ShareRequest) -> (KeyAnchor, bool) {
    let lower_q = request.lower_quantizer;
    let moving_startup = request.layer_index == 0
        && request
            .startup_profile
            .is_some_and(|(peak, static_fraction)| {
                static_fraction < MOVING_STARTUP_MAX_STATIC || peak >= MOVING_STARTUP_MIN_PEAK
            });
    let anchor = |min: u32, max: u32, seed_inter_offset| KeyAnchor {
        min,
        max: max.max(min),
        seed_inter_offset,
    };
    if request.enhancement_interval > 1 {
        let anchor = anchor(
            lower_q.saturating_sub(DECIMATED_KEY_FINER),
            lower_q.saturating_add(DECIMATED_KEY_COARSER),
            Some(DECIMATED_KEY_INTER_SEED),
        );
        (anchor, false)
    } else if moving_startup {
        let (min_lead, max_lead) = if num_layers >= 3 {
            (
                MOVING_STARTUP_THREE_LAYER_MIN_LEAD,
                MOVING_STARTUP_THREE_LAYER_MAX_LEAD,
            )
        } else {
            (MOVING_STARTUP_KEY_MIN_LEAD, MOVING_STARTUP_KEY_MAX_LEAD)
        };
        let anchor = anchor(
            lower_q.saturating_add(min_lead),
            lower_q.saturating_add(max_lead),
            Some(0),
        );
        (anchor, true)
    } else {
        let anchor = anchor(
            lower_q.saturating_sub(KEY_MAX_FINER_THAN_BASE),
            u32::MAX,
            None,
        );
        (anchor, false)
    }
}

/// Top-down layer drop decision for an `N`-layer temporal unit
/// ([`decide_unit_drop_layers`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialUnitDrop {
    /// Keep scheduled layers `< kept_layers` (`1..=num_layers`) and drop any
    /// scheduled layers `>= kept_layers` (`S2` first, then `S1`).
    KeepLayers(u8),
    /// Skip the whole temporal unit (`S2`, `S1`, and `S0`).
    Unit,
}

/// One layer's rate-control facts for the next temporal unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerDropFacts {
    /// The layer's picture may be dropped without breaking later references.
    pub droppable: bool,
    /// The layer's own controller would drop it ([`crate::RateController::should_drop`]).
    pub controller_drop: bool,
    /// Predicted bits at the maximum quantizer (`None` before any inter model).
    pub floor_bits: Option<f64>,
}

/// Top-down enhancement-first frame dropping across `N` spatial layers (`2` or
/// `3`, brief §2.5): under buffer pressure, `S2` drops first, then `S1`, and
/// `S0` drops only if every upper layer is absent or already dropped.
#[must_use]
pub fn decide_unit_drop_layers(
    frame_drop: bool,
    key_unit: bool,
    due: &[bool],
    facts: &[LayerDropFacts],
    overflows: impl Fn(f64) -> bool,
) -> SpatialUnitDrop {
    let num_layers = due.len().min(facts.len());
    if !frame_drop || key_unit || num_layers == 0 {
        return SpatialUnitDrop::KeepLayers(num_layers as u8);
    }
    let mut floor_sum: f64 = (0..num_layers)
        .filter(|&s| due[s])
        .map(|s| facts[s].floor_bits.unwrap_or(0.0))
        .sum();
    for s in (1..num_layers).rev() {
        if !due[s] {
            continue;
        }
        let drop_s = facts[s].droppable && (facts[s].controller_drop || overflows(floor_sum));
        if drop_s {
            floor_sum = (floor_sum - facts[s].floor_bits.unwrap_or(0.0)).max(0.0);
        } else {
            return SpatialUnitDrop::KeepLayers((s + 1) as u8);
        }
    }
    if facts[0].droppable && (facts[0].controller_drop || overflows(floor_sum)) {
        SpatialUnitDrop::Unit
    } else {
        SpatialUnitDrop::KeepLayers(1)
    }
}

/// Time since the last coded (or rate-control-dropped) `S1` picture: the
/// interval `S1`'s controller budgets one picture over when
/// `enhancement_interval > 1`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EnhancementClock {
    elapsed_sec: f64,
    units: u32,
    timed: bool,
}

impl EnhancementClock {
    /// One temporal unit of `interval` seconds (`None` = the nominal
    /// `nominal_dt`, untimed input) passed.
    pub fn advance(&mut self, interval: Option<f64>, nominal_dt: f64) {
        self.units += 1;
        if let Some(dt) = interval {
            self.elapsed_sec += dt;
            self.timed = true;
        } else {
            self.elapsed_sec += nominal_dt;
        }
    }

    /// `S1`'s frame interval if its picture were coded now: `None` (the
    /// controller's nominal `frame_rate / N`) when exactly `expected_units`
    /// untimed units passed, else the elapsed seconds.
    #[must_use]
    pub fn interval(&self, expected_units: u32) -> Option<f64> {
        (self.timed || self.units != expected_units)
            .then_some(self.elapsed_sec.max(MIN_INTERVAL_SEC))
    }

    /// [`Self::interval`], then restart the clock (an `S1` picture was coded
    /// or dropped by rate control).
    pub fn take(&mut self, expected_units: u32) -> Option<f64> {
        let interval = self.interval(expected_units);
        *self = Self::default();
        interval
    }
}

/// Shortest `S1` frame interval passed to its controller (guards `1 / dt`).
const MIN_INTERVAL_SEC: f64 = 1e-6;

fn shift_mask(mask: SlotMask, offset: usize) -> SlotMask {
    SlotMask(mask.0 << offset)
}

fn shift_references(refs: FrameReferences, offset: usize) -> FrameReferences {
    FrameReferences {
        position: refs.position,
        reference: ReferenceLink {
            slot: refs.reference.slot + offset,
            distance: refs.reference.distance,
        },
        refresh: shift_mask(refs.refresh, offset),
        secondary: refs.secondary.map(|s| ReferenceLink {
            slot: s.slot + offset,
            distance: s.distance,
        }),
    }
}

fn first_slot(mask: SlotMask) -> Option<usize> {
    (0..MAX_REFERENCE_SLOTS).find(|&slot| mask.contains(slot))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Codec, StreamSignal};
    use crate::params::ParamOverrides;
    use std::collections::HashSet;

    fn sample_config() -> EncoderConfig {
        let mut params = ParamOverrides::default();
        params.set("spatial.layers", 2.0);
        EncoderConfig {
            codec: Codec::Av1,
            width: 1280,
            height: 720,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 120,
            intra_period: 30,
            temporal_layers: 1,
            block_log2: 6,
            key_qp_offset: None,
            signal: StreamSignal::default(),
            params,
        }
    }

    #[test]
    fn scalability_modes_parse_and_map_layers() {
        for (name, s, t) in [
            ("L1T1", 1, 1),
            ("l1t2", 1, 2),
            ("L1T3", 1, 3),
            ("L2T1", 2, 1),
            ("l2t2", 2, 2),
            ("L2T3", 2, 3),
            ("L3T1", 3, 1),
            ("l3t2", 3, 2),
            ("L3T3", 3, 3),
        ] {
            let mode = ScalabilityMode::parse(name).unwrap();
            assert_eq!((mode.spatial_layers(), mode.temporal_layers()), (s, t));
        }
        assert!(ScalabilityMode::parse("L4T1").is_none());
        assert!(ScalabilityMode::parse("L2T4").is_none());
        assert!(ScalabilityMode::parse("L3T4").is_none());
    }

    #[test]
    fn base_and_three_layer_dimensions_cover_dyadic_and_custom_ratios() {
        assert_eq!(base_layer_size((1280, 720), 0.5).unwrap(), (640, 360));
        assert_eq!(base_layer_size((1280, 720), 2.0 / 3.0).unwrap(), (854, 480));
        assert_eq!(base_layer_size((1280, 720), 0.6667).unwrap(), (854, 480));
        assert_eq!(base_layer_size((1280, 720), 0.8).unwrap(), (1024, 576));
        assert_eq!(base_layer_size((1280, 720), 1.0).unwrap(), (1280, 720));
        assert!(base_layer_size((2, 720), 0.5).is_err());

        assert_eq!(
            three_layer_sizes((1280, 720), 0.25, 0.5).unwrap(),
            [(320, 180), (640, 360), (1280, 720)]
        );
        assert_eq!(
            three_layer_sizes((1280, 720), 0.375, 2.0 / 3.0).unwrap(),
            [(480, 270), (854, 480), (1280, 720)]
        );
        assert_eq!(
            three_layer_sizes((1884, 64), 0.25, 0.5).unwrap(),
            [(472, 16), (942, 32), (1884, 64)]
        );
        assert!(three_layer_sizes((1280, 720), 0.6, 0.5).is_err());
        assert!(three_layer_sizes((6, 720), 0.25, 0.5).is_err());
    }

    #[test]
    fn spatial_config_rejects_incompatible_tools_and_invalid_three_layer_combos() {
        let base = sample_config();
        assert!(
            SpatialConfig::from_encoder_config(&base, true)
                .unwrap()
                .is_some()
        );
        assert!(matches!(
            SpatialConfig::from_encoder_config(&base, false),
            Err(ConfigError::SpatialScalability(_))
        ));
        for (param, value) in [
            ("ltr.interval", 10.0),
            ("refresh.period", 30.0),
            ("refs.second", 1.0),
            ("resolution.adaptive", 1.0),
        ] {
            let mut cfg = base.clone();
            cfg.params.set(param, value);
            assert!(
                SpatialConfig::from_encoder_config(&cfg, true).is_err(),
                "{param}={value} must be rejected with spatial layers"
            );
        }

        // 3-layer rejected on max_spatial_layers == 2 (e.g. VP9), with enhancement_interval > 1,
        // or when s0_rate_share + s1_rate_share > 0.90:
        let mut l3 = sample_config();
        l3.params.set("spatial.layers", 3.0);
        let ok_l3 = SpatialConfig::from_encoder_config_max(&l3, 3)
            .unwrap()
            .unwrap();
        assert_eq!(ok_l3.layers, 3);
        assert_eq!(ok_l3.layer_sizes, vec![(320, 180), (640, 360), (1280, 720)]);
        assert!(matches!(
            SpatialConfig::from_encoder_config_max(&l3, 2),
            Err(ConfigError::SpatialScalability(_))
        ));
        let mut l3_decim = l3.clone();
        l3_decim.params.set("spatial.enhancement_interval", 2.0);
        assert!(matches!(
            SpatialConfig::from_encoder_config_max(&l3_decim, 3),
            Err(ConfigError::SpatialScalability(_))
        ));
        let mut l3_shares = l3;
        l3_shares.params.set("spatial.s0_rate_share", 0.45);
        l3_shares.params.set("spatial.s1_rate_share", 0.50);
        assert!(matches!(
            SpatialConfig::from_encoder_config_max(&l3_shares, 3),
            Err(ConfigError::SpatialScalability(_))
        ));
    }

    /// Verify that for EVERY operating point advertised in `operating_points`,
    /// filtering coded layer pictures by `idc` leaves a self-contained stream
    /// whose `referenced_pictures` were all kept earlier!
    fn assert_all_operating_points_reference_closed(
        temporal_layers: u8,
        enhancement_interval: u32,
        inter_layer: bool,
    ) {
        let mut s = SpatialStructure::with_layers(
            2,
            temporal_layers,
            enhancement_interval,
            inter_layer,
            None,
        );
        let ops = build_operating_points(
            (854, 480),
            (1280, 720),
            FrameRate { num: 30, den: 1 },
            temporal_layers,
            enhancement_interval,
            inter_layer,
        );
        let mut tus = Vec::new();
        for tu in 0..36 {
            tus.push(s.next_temporal_unit_layers(tu == 0 || tu == 18, 2));
        }
        assert_eq!(ops[0].max_spatial_id, 1);
        assert_eq!(ops[0].max_temporal_id, temporal_layers - 1);
        assert_eq!(ops.last().unwrap().idc, 0x0101);

        for op in &ops {
            let in_op = |p: &SpatialLayerPlan| {
                ((op.idc >> p.temporal_id) & 1 != 0) && ((op.idc >> (8 + p.spatial_id)) & 1 != 0)
            };
            let mut decoded: HashSet<u64> = HashSet::new();
            for tu in &tus {
                for plan in tu.layer_plans() {
                    if in_op(plan) {
                        for dep in plan.referenced_pictures.iter() {
                            assert!(
                                decoded.contains(&dep),
                                "OP {:#x} missing S{} dep {dep} at TU {}",
                                op.idc,
                                plan.spatial_id,
                                tu.tu_index
                            );
                        }
                        decoded.insert(plan.picture_index);
                    }
                }
            }
        }
    }

    #[test]
    fn operating_points_are_closed_across_all_modes_and_intervals() {
        for temporal_layers in 1..=3 {
            for inter_layer in [false, true] {
                for interval in [1, 2, 3, 4, 5, 6, 8] {
                    assert_all_operating_points_reference_closed(
                        temporal_layers,
                        interval,
                        inter_layer,
                    );
                }
            }
        }
    }

    #[test]
    fn three_layer_dpb_map_and_operating_points_are_closed_across_l3t1_l3t2_l3t3() {
        assert_eq!(spatial_pool_slots(2), 5);
        assert_eq!(spatial_pool_slots(3), 8);
        assert_eq!(inter_layer_scratch_slot(2, 0), 4);
        assert_eq!(inter_layer_scratch_slot(3, 0), 6);
        assert_eq!(inter_layer_scratch_slot(3, 1), 7);

        for temporal_layers in 1..=3 {
            for inter_layer in [false, true] {
                let mut s = SpatialStructure::with_layers(3, temporal_layers, 1, inter_layer, None);
                let ops = build_three_layer_operating_points(
                    [(320, 180), (640, 360), (1280, 720)],
                    FrameRate { num: 30, den: 1 },
                    temporal_layers,
                );
                assert_eq!(ops.len(), usize::from(temporal_layers) * 3);
                assert_eq!(ops[0].max_spatial_id, 2);
                assert_eq!(ops[0].max_temporal_id, temporal_layers - 1);
                assert_eq!(ops.last().unwrap().idc, 0x0101);

                let mut tus = Vec::new();
                for tu in 0..36 {
                    // Exercise S2-only repair on TU 9, S1+S2 repair on TU 15, S2 drop on TU 21,
                    // and key TU on TU 24:
                    if tu == 9 {
                        s.request_layer_repair(2);
                    } else if tu == 15 {
                        s.request_layer_repair(1);
                    }
                    let kept = if tu == 21 { 2 } else { 3 };
                    let plan = s.next_temporal_unit_layers(tu == 0 || tu == 24, kept);
                    if temporal_layers == 3 && inter_layer && tu == 1 {
                        // Disposable T2 TU in L3T3: S0 -> slot 6, S1 -> slot 7, S2 -> NONE!
                        assert_eq!(
                            plan.base.references.refresh,
                            SlotMask::only(inter_layer_scratch_slot(3, 0))
                        );
                        let s1 = plan.upper[0].unwrap();
                        assert_eq!(
                            s1.references.refresh,
                            SlotMask::only(inter_layer_scratch_slot(3, 1))
                        );
                        assert_eq!(
                            s1.references.secondary,
                            Some(ReferenceLink {
                                slot: inter_layer_scratch_slot(3, 0),
                                distance: 0,
                            })
                        );
                        let s2 = plan.upper[1].unwrap();
                        assert_eq!(s2.references.refresh, SlotMask::NONE);
                        assert_eq!(
                            s2.references.secondary,
                            Some(ReferenceLink {
                                slot: inter_layer_scratch_slot(3, 1),
                                distance: 0,
                            })
                        );
                    }
                    tus.push(plan);
                }

                for op in &ops {
                    let in_op = |p: &SpatialLayerPlan| {
                        ((op.idc >> p.temporal_id) & 1 != 0)
                            && ((op.idc >> (8 + p.spatial_id)) & 1 != 0)
                    };
                    let mut decoded: HashSet<u64> = HashSet::new();
                    for tu in &tus {
                        for plan in tu.layer_plans() {
                            if in_op(plan) {
                                for dep in plan.referenced_pictures.iter() {
                                    assert!(
                                        decoded.contains(&dep),
                                        "L3T{temporal_layers} (inter={inter_layer}) OP {:#x} missing S{} dep {dep} at TU {}",
                                        op.idc,
                                        plan.spatial_id,
                                        tu.tu_index
                                    );
                                }
                                decoded.insert(plan.picture_index);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn l2t1_l2t2_l2t3_slot_isolation_and_key_units() {
        let mut s = SpatialStructure::with_layers(2, 3, 1, true, None);
        let tu0 = s.next_temporal_unit_layers(true, 2);
        assert_eq!(
            (tu0.base.frame_type, tu0.base.intra_only),
            (FrameType::Key, false)
        );
        assert_eq!(tu0.base.references.refresh, SlotMask::ALL);
        let enh0 = tu0.upper[0].unwrap();
        assert_eq!(
            (enh0.frame_type, enh0.intra_only),
            (FrameType::Inter, false)
        );
        assert_eq!(enh0.references.reference.slot, BASE_SLOT_OFFSET);
        assert_eq!(enh0.references.secondary, None);
        assert_eq!(enh0.references.refresh, ENHANCEMENT_TEMPORAL_SLOTS);
        assert_eq!(
            enh0.referenced_pictures.iter().collect::<Vec<_>>(),
            vec![tu0.base.picture_index]
        );

        let tu1 = s.next_temporal_unit_layers(false, 2);
        assert_eq!(tu1.base.temporal_id, 2);
        assert_eq!(tu1.base.references.refresh, INTER_LAYER_SCRATCH_MASK);
        let enh1 = tu1.upper[0].unwrap();
        assert_eq!(enh1.temporal_id, 2);
        assert_eq!(enh1.references.refresh, SlotMask::NONE);
        assert!(ENHANCEMENT_TEMPORAL_SLOTS.contains(enh1.references.reference.slot));
        assert_eq!(
            enh1.references.secondary,
            Some(ReferenceLink {
                slot: INTER_LAYER_SCRATCH_SLOT,
                distance: 0,
            })
        );

        let mut indep = SpatialStructure::with_layers(2, 2, 1, false, None);
        let k0 = indep.next_temporal_unit_layers(true, 2);
        let e0 = k0.upper[0].unwrap();
        assert_eq!((e0.frame_type, e0.intra_only), (FrameType::Key, true));
        assert_eq!(e0.references.refresh, ENHANCEMENT_TEMPORAL_SLOTS);
        assert_eq!(e0.referenced_pictures, ReferencedFrames::NONE);
    }

    #[test]
    fn decimated_enhancement_cadence_and_enhancement_first_drops() {
        let mut s = SpatialStructure::with_layers(2, 1, 6, true, None);
        let mut coded_s1 = Vec::new();
        for tu in 0..19 {
            let drop_s1 = tu == 12;
            let plan = s.next_temporal_unit_layers(tu == 0, if drop_s1 { 1 } else { 2 });
            if let Some(enh) = plan.upper[0] {
                coded_s1.push((tu, enh.references.reference.distance));
            }
        }
        assert_eq!(coded_s1, vec![(0, 1), (6, 6), (18, 6)]);
    }

    #[test]
    fn rate_split_and_combined_cbr_buffer_handle_720p5_bursts() {
        let mut total = RateControlConfig::new(500_000);
        total.buffer_ms = 1000;
        let [base, enh] = split_rate_control(
            total,
            &[0.40, 0.60],
            FrameRate { num: 30, den: 1 },
            300,
            6,
            true,
        )
        .unwrap()[..] else {
            panic!("two layer budgets");
        };
        assert_eq!(base.config.target_bps, 200_000);
        assert_eq!(enh.config.target_bps, 300_000);
        assert_eq!(base.frame_rate, FrameRate { num: 30, den: 1 });
        assert_eq!(enh.frame_rate, FrameRate { num: 30, den: 6 });
        assert_eq!((base.intra_period, enh.intra_period), (300, 50));
        assert_eq!(enh.config.keyframe_boost, 1.0);

        let mut cbr =
            CombinedCbrBuffer::with_share_caps(total, FrameRate { num: 30, den: 1 }, &[0.35, 0.65]);
        for tu in 0..150 {
            let s0_bits = 6_666;
            let s1_bits = if tu % 6 == 0 { 60_000 } else { 0 };
            cbr.observe_tu(s0_bits + s1_bits, None, true);
        }
        assert_eq!(cbr.buffer_overflows(), 0);
        assert_eq!(cbr.peak_tu_bits(), 66_666);
        assert!((cbr.actual_bps() - 500_000.0).abs() < 100.0);
    }

    #[test]
    fn three_layer_rate_split_and_upward_bit_flow_conserve_total_budget() {
        let mut total = RateControlConfig::new(600_000);
        total.buffer_ms = 1000;
        total.peak_bps = Some(1_200_000);
        let shares = [0.12, 0.25, 0.63];
        let [s0, s1, s2] =
            split_rate_control(total, &shares, FrameRate { num: 30, den: 1 }, 300, 1, true)
                .unwrap()[..]
        else {
            panic!("three layer budgets");
        };
        assert_eq!(
            s0.config.target_bps + s1.config.target_bps + s2.config.target_bps,
            600_000
        );
        assert_eq!(s0.config.target_bps, 72_000);
        assert_eq!(s1.config.target_bps, 150_000);
        assert_eq!(s2.config.target_bps, 378_000);

        let mut cbr =
            CombinedCbrBuffer::with_share_caps(total, FrameRate { num: 30, den: 1 }, &shares);
        for _ in 0..8 {
            cbr.observe_step_inter_layer_hit_rate(0, 4, 100);
            cbr.observe_step_inter_layer_hit_rate(1, 4, 100);
        }
        // Static scene: both S0 (180p) and S1 (360p) undershoot, flowing unspent bits to S2 (720p):
        let u0_nom = 72_000.0 / 30.0; // 2400 bits
        let u1_nom = 150_000.0 / 30.0; // 5000 bits
        let u2_nom = 378_000.0 / 30.0; // 12600 bits
        let u0 = cbr.reallocate_unspent_lower_bits(0, u0_nom, 0.0, 400);
        assert!(u0 > 1_000.0);
        let (r1, carry_s2) = cbr.split_s0_reallocation_for_s1(u0, 0.95, 640 * 360, Some(150));
        let f1 = cbr.reallocate_unspent_lower_bits(1, u1_nom, r1, 1_200);
        let to_s2 = carry_s2 + f1;
        assert!(
            to_s2 > 2_500.0,
            "S2 must receive unspent S0 + S1 bits ({to_s2})"
        );
        let drain_sum = (u0_nom - u0) + (u1_nom + r1 - f1) + (u2_nom + to_s2);
        assert!((drain_sum - 600_000.0 / 30.0).abs() < 1e-6);
    }

    #[test]
    fn layer_peaks_split_in_proportion_to_their_targets() {
        let mut total = RateControlConfig::new(600_000);
        total.peak_bps = Some(1_200_000);
        for interval in [1, 6] {
            let [base, enh] = split_rate_control(
                total,
                &[0.35, 0.65],
                FrameRate { num: 30, den: 1 },
                300,
                interval,
                true,
            )
            .unwrap()[..] else {
                panic!("two layer budgets");
            };
            assert_eq!(base.config.peak_bps, Some(420_000));
            assert_eq!(enh.config.peak_bps, Some(780_000));
        }
    }

    fn facts(droppable: bool, controller_drop: bool, floor: f64) -> LayerDropFacts {
        LayerDropFacts {
            droppable,
            controller_drop,
            floor_bits: Some(floor),
        }
    }

    #[test]
    fn unit_drop_drops_s2_then_s1_before_s0() {
        let room = |limit: f64| move |bits: f64| bits > limit;
        let calm = (facts(true, false, 1_000.0), facts(true, false, 4_000.0));
        assert_eq!(
            decide_unit_drop_layers(false, false, &[true, true], &[calm.0, calm.1], room(0.0)),
            SpatialUnitDrop::KeepLayers(2)
        );
        assert_eq!(
            decide_unit_drop_layers(true, true, &[true, true], &[calm.0, calm.1], room(0.0)),
            SpatialUnitDrop::KeepLayers(2)
        );
        assert_eq!(
            decide_unit_drop_layers(
                true,
                false,
                &[true, true],
                &[calm.0, calm.1],
                room(10_000.0)
            ),
            SpatialUnitDrop::KeepLayers(2)
        );
        assert_eq!(
            decide_unit_drop_layers(true, false, &[true, true], &[calm.0, calm.1], room(2_000.0)),
            SpatialUnitDrop::KeepLayers(1)
        );
        assert_eq!(
            decide_unit_drop_layers(true, false, &[true, true], &[calm.0, calm.1], room(500.0)),
            SpatialUnitDrop::Unit
        );
        let base_wants = (facts(true, true, 1_000.0), facts(true, false, 4_000.0));
        assert_eq!(
            decide_unit_drop_layers(
                true,
                false,
                &[true, true],
                &[base_wants.0, base_wants.1],
                room(10_000.0)
            ),
            SpatialUnitDrop::KeepLayers(2)
        );
        assert_eq!(
            decide_unit_drop_layers(
                true,
                false,
                &[true, false],
                &[base_wants.0, base_wants.1],
                room(10_000.0)
            ),
            SpatialUnitDrop::Unit
        );
        let pinned = (facts(true, true, 1_000.0), facts(false, true, 4_000.0));
        assert_eq!(
            decide_unit_drop_layers(true, false, &[true, true], &[pinned.0, pinned.1], room(0.0)),
            SpatialUnitDrop::KeepLayers(2)
        );
        let enh_wants = (facts(true, false, 1_000.0), facts(true, true, 4_000.0));
        assert_eq!(
            decide_unit_drop_layers(
                true,
                false,
                &[true, true],
                &[enh_wants.0, enh_wants.1],
                room(10_000.0)
            ),
            SpatialUnitDrop::KeepLayers(1)
        );

        // 3-layer top-down dropping: S0=800, S1=2200, S2=6000 (total 9000):
        let l3_due = [true, true, true];
        let l3_calm = [
            facts(true, false, 800.0),
            facts(true, false, 2_200.0),
            facts(true, false, 6_000.0),
        ];
        // Room 10_000: keep all 3 layers.
        assert_eq!(
            decide_unit_drop_layers(true, false, &l3_due, &l3_calm, room(10_000.0)),
            SpatialUnitDrop::KeepLayers(3)
        );
        // Room 4_000: S0+S1+S2 (9000) overflows, S0+S1 (3000) fits -> drop S2 only!
        assert_eq!(
            decide_unit_drop_layers(true, false, &l3_due, &l3_calm, room(4_000.0)),
            SpatialUnitDrop::KeepLayers(2)
        );
        // Room 1_500: S0+S1 (3000) also overflows, S0 (800) fits -> drop S2 and S1, keep S0!
        assert_eq!(
            decide_unit_drop_layers(true, false, &l3_due, &l3_calm, room(1_500.0)),
            SpatialUnitDrop::KeepLayers(1)
        );
        // Room 400: even S0 overflows -> drop Unit!
        assert_eq!(
            decide_unit_drop_layers(true, false, &l3_due, &l3_calm, room(400.0)),
            SpatialUnitDrop::Unit
        );
        // S1 wants to drop on its own while S2 does NOT: S2 stays so S1 must stay too!
        let s1_only_wants = [
            facts(true, false, 800.0),
            facts(true, true, 2_200.0),
            facts(true, false, 6_000.0),
        ];
        assert_eq!(
            decide_unit_drop_layers(true, false, &l3_due, &s1_only_wants, room(10_000.0)),
            SpatialUnitDrop::KeepLayers(3)
        );
    }

    #[test]
    fn enhancement_clock_reports_explicit_intervals_only_off_cadence() {
        let nominal = 1.0 / 30.0;
        let mut clock = EnhancementClock::default();
        for _ in 0..6 {
            clock.advance(None, nominal);
        }
        assert_eq!(clock.take(6), None);
        for _ in 0..7 {
            clock.advance(None, nominal);
        }
        let seven = clock.take(6).expect("off cadence");
        assert!((seven - 7.0 * nominal).abs() < 1e-12);
        clock.advance(Some(0.1), nominal);
        assert_eq!(clock.take(1), Some(0.1));
    }

    #[test]
    fn per_layer_loss_distinguishes_s0_key_s1_repair_s2_repair_and_non_reference_tus() {
        let mut s = SpatialStructure::with_layers(3, 3, 1, true, None);
        let _tu0 = s.next_temporal_unit_layers(true, 3);
        let _tu1 = s.next_temporal_unit_layers(false, 3); // T2 non-reference
        let _tu2 = s.next_temporal_unit_layers(false, 3); // T1 reference

        // Losing T2 (TU 1) needs no repair on any layer.
        assert!(!s.needs_layer_repair(0, 1));
        assert!(!s.needs_layer_repair(1, 1));
        assert!(!s.needs_layer_repair(2, 1));

        // Losing S2 only on TU 2 repairs S2 from same-TU S1 while keeping S0 and S1 temporal!
        assert!(s.needs_layer_repair(2, 2));
        s.request_layer_repair(2);
        assert!(!s.needs_layer_repair(2, 2));
        assert!(s.needs_layer_repair(1, 2));
        let tu3 = s.next_temporal_unit_layers(false, 1);
        assert!(!tu3.key_unit);
        assert_eq!(tu3.repair_from_layer, Some(2));
        let s1_3 = tu3.upper[0].unwrap();
        let s2_3 = tu3.upper[1].unwrap();
        assert!(!s1_3.enhancement_repair);
        assert!(s1_3.references.secondary.is_some()); // S1 still uses temporal + S0!
        assert!(s2_3.enhancement_repair);
        assert_eq!(s2_3.references.secondary, None); // S2 predicts solely from S1!
        assert_eq!(s2_3.references.refresh, TOP_ENHANCEMENT_TEMPORAL_SLOTS);
        assert_eq!(
            s2_3.referenced_pictures.iter().collect::<Vec<_>>(),
            vec![s1_3.picture_index]
        );

        // Losing S1 on TU 3 repairs BOTH S1 (from S0) and S2 (from repaired S1)!
        assert!(s.needs_layer_repair(1, 3));
        s.request_layer_repair(1);
        let tu4 = s.next_temporal_unit_layers(false, 1);
        assert_eq!(tu4.repair_from_layer, Some(1));
        let s1_4 = tu4.upper[0].unwrap();
        let s2_4 = tu4.upper[1].unwrap();
        assert!(s1_4.enhancement_repair);
        assert!(s2_4.enhancement_repair);
        assert_eq!(
            s1_4.referenced_pictures.iter().collect::<Vec<_>>(),
            vec![tu4.base.picture_index]
        );
        assert_eq!(
            s2_4.referenced_pictures.iter().collect::<Vec<_>>(),
            vec![s1_4.picture_index]
        );

        // Independent layers (inter_layer=0) repair S1 with an INTRA_ONLY_FRAME.
        let mut indep = SpatialStructure::with_layers(2, 1, 6, false, None);
        let _ = indep.next_temporal_unit_layers(true, 2);
        let _ = indep.next_temporal_unit_layers(false, 2);
        assert!(!indep.needs_layer_repair(1, 1));
        assert!(indep.needs_layer_repair(1, 0));
        indep.request_layer_repair(1);
        let repair_tu = indep.next_temporal_unit_layers(false, 2);
        assert!(!repair_tu.key_unit);
        assert_eq!(repair_tu.repair_from_layer, Some(1));
        let enh_rep = repair_tu.upper[0].unwrap();
        assert_eq!(
            (enh_rep.frame_type, enh_rep.intra_only),
            (FrameType::Key, true)
        );
        assert_eq!(enh_rep.references.refresh, ENHANCEMENT_TEMPORAL_SLOTS);
        assert_eq!(enh_rep.referenced_pictures, ReferencedFrames::NONE);
    }

    fn share_request(spatial_id: u8, key: bool) -> ShareRequest {
        ShareRequest {
            spatial_id,
            key,
            has_coded_upper_layer: spatial_id == 0,
            layer_index: 0,
            inter_layer: true,
            enhancement_interval: 1,
            interval: None,
            ceiling: 1.0,
            prior_layer_bits: 0,
            lower_quantizer: 100,
            reallocated_bits: 0.0,
            static_fraction: 0.0,
            layer_pixels: 640 * 360,
            upper_inter_level: None,
            startup_profile: None,
        }
    }

    #[test]
    fn layer_shares_follow_the_combined_buffer_plan() {
        let cbr = CombinedCbrBuffer::with_share_caps(
            RateControlConfig::new(500_000),
            FrameRate { num: 30, den: 1 },
            &[0.35, 0.65],
        );
        assert_eq!(cbr.plan_share(&share_request(0, true)), LayerShare::WHOLE);

        let inter_s1 = ShareRequest {
            prior_layer_bits: 4_000,
            reallocated_bits: 1_500.0,
            ..share_request(1, false)
        };
        let share = cbr.plan_share(&inter_s1);
        assert_eq!(share.reallocated_bits, 1_500.0);
        assert_eq!(share.room_bits, cbr.remaining_tu_room(None, 1.0, 4_000));
        assert_eq!(share.key_anchor, None);

        let moving = ShareRequest {
            startup_profile: Some((4.0, 0.05)),
            ..share_request(1, true)
        };
        let share = cbr.plan_share(&moving);
        assert!(share.clear_key_loan);
        assert_eq!(
            share.key_anchor.map(|a| (a.min, a.max)),
            Some((
                100 + MOVING_STARTUP_KEY_MIN_LEAD,
                100 + MOVING_STARTUP_KEY_MAX_LEAD
            ))
        );
        let still = ShareRequest {
            startup_profile: Some((0.2, 0.9)),
            ..share_request(1, true)
        };
        let share = cbr.plan_share(&still);
        assert!(!share.clear_key_loan);
        assert_eq!(
            share.key_anchor.map(|a| a.min),
            Some(100 - KEY_MAX_FINER_THAN_BASE)
        );
    }

    #[test]
    fn static_aware_rate_split_caps_s0_and_reallocates_unspent_bits_to_s1() {
        let mut total = RateControlConfig::new(500_000);
        total.buffer_ms = 1000;
        let mut cbr =
            CombinedCbrBuffer::with_share_caps(total, FrameRate { num: 30, den: 1 }, &[0.35, 0.65]);
        for _ in 0..10 {
            cbr.observe_step_inter_layer_hit_rate(0, 5, 100);
        }
        let base_pixels = 640 * 360;
        let (static_share, static_scale, min_q) =
            cbr.plan_lower_layer_inter_constraints(0, 0.95, base_pixels, Some(160));
        assert!(
            (0.15..=0.21).contains(&static_share),
            "static share should drop near the 0.15 floor, got {static_share}"
        );
        assert!(static_scale < 0.60);
        assert!(
            min_q.unwrap() >= 156,
            "static S0 quantizer should be floored near S1 (160), got {min_q:?}"
        );

        let s0_nominal = 500_000.0 * 0.35 / 30.0;
        let s1_nominal = 500_000.0 * 0.65 / 30.0;
        let s0_actual = 2_000u64;
        let reallocated = cbr.reallocate_unspent_lower_bits(0, s0_nominal, 0.0, s0_actual);
        assert!(reallocated > 2_500.0);
        let combined_drain = (s0_nominal - reallocated) + (s1_nominal + reallocated);
        assert!((combined_drain - 500_000.0 / 30.0).abs() < 1e-6);

        let low_cbr = CombinedCbrBuffer::with_share_caps(
            RateControlConfig::new(150_000),
            FrameRate { num: 30, den: 1 },
            &[0.35, 0.65],
        );
        let (low_share, _, low_min_q) =
            low_cbr.plan_lower_layer_inter_constraints(0, 0.95, base_pixels, Some(160));
        assert!(
            low_share > static_share,
            "low-bitrate guard should raise S0 floor ({low_share} vs {static_share})"
        );
        assert!(low_min_q.unwrap() < min_q.unwrap());

        for _ in 0..10 {
            cbr.observe_step_inter_layer_hit_rate(0, 40, 100);
        }
        let (active_share, active_scale, active_min_q) =
            cbr.plan_lower_layer_inter_constraints(0, 0.05, base_pixels, Some(160));
        assert!((active_share - 0.35).abs() < 1e-6);
        assert!((active_scale - 1.0).abs() < 1e-6);
        assert_eq!(active_min_q, None);
    }
}
