//! Tunable heuristic parameters.
//!
//! The encoder's thresholds and multipliers interact, so tuning them one at a
//! time misses the joint optimum, and different speed/quality trade-offs want
//! different sets of values. Every such constant is therefore a named,
//! range-checked parameter:
//!
//! - each area declares a typed group with [`param_group!`] (its fields are
//!   plain `f32`s read once per encoder, so a lookup never sits in a hot
//!   loop);
//! - [`ParamOverrides`] holds the values a user (CLI, preset file, tuner)
//!   changed; everything else keeps its default, and the defaults are the
//!   values the encoder was tuned with by hand;
//! - [`engine_specs`] (plus a backend's own [`CodecBackend::PARAM_SPECS`],
//!   declared with the same exported macro)
//!   lists every parameter with its default, range and what it trades, for
//!   `nanocodec params` and the tuner (`eval/nceval/tune.py`).
//!
//! [`CodecBackend::PARAM_SPECS`]: crate::backend::CodecBackend::PARAM_SPECS

use std::collections::BTreeMap;
use std::fmt;

/// One tunable parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamSpec {
    /// Dotted name, `area.parameter`.
    pub name: &'static str,
    /// The hand-tuned value.
    pub default: f32,
    /// Smallest accepted value.
    pub min: f32,
    /// Largest accepted value.
    pub max: f32,
    /// What raising the value trades, in one line.
    pub doc: &'static str,
}

/// Values set away from their defaults, by parameter name.
#[derive(Debug, Clone, Default)]
pub struct ParamOverrides(BTreeMap<String, f32>);

impl PartialEq for ParamOverrides {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(&other.0)
                .all(|((a, x), (b, y))| a == b && x.to_bits() == y.to_bits())
    }
}

impl Eq for ParamOverrides {}

/// A parameter override that does not fit the registry.
#[derive(Debug, Clone)]
pub enum ParamError {
    /// No parameter has this name.
    Unknown(String),
    /// A non-base frame setting (`nonbase.<name>`, `frame_role.rs`) names a
    /// parameter that is not re-read per frame, or one that defines the
    /// base cadence itself.
    NotPerFrame(String),
    /// The value is outside the parameter's range.
    OutOfRange {
        /// Parameter name.
        name: String,
        /// Requested value.
        value: f32,
        /// Accepted range.
        min: f32,
        /// Accepted range.
        max: f32,
    },
}

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(name) => write!(f, "unknown parameter '{name}' (see `nanocodec params`)"),
            Self::NotPerFrame(name) => write!(
                f,
                "parameter '{name}' cannot take a non-base frame value: it is not re-read per frame"
            ),
            Self::OutOfRange {
                name,
                value,
                min,
                max,
            } => write!(f, "parameter {name} = {value} outside {min}..={max}"),
        }
    }
}

impl PartialEq for ParamError {
    fn eq(&self, other: &Self) -> bool {
        // Compare values bit for bit, so the type can be `Eq` (configs are).
        self.to_string() == other.to_string()
    }
}

impl Eq for ParamError {}

impl std::error::Error for ParamError {}

impl ParamOverrides {
    /// Set `name` to `value` (checked against the registry later, by
    /// [`ParamOverrides::validate`]).
    pub fn set(&mut self, name: &str, value: f32) {
        self.0.insert(name.to_owned(), value);
    }

    /// Whether `name` has an explicit override in this map.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    /// The explicit override of `name`, if this map has one.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<f32> {
        self.0.get(name).copied()
    }

    /// These overrides with every input frame coded (`hold.enable` = 0):
    /// for tests that encode a repeated picture to exercise coding itself.
    #[must_use]
    pub fn coding_every_frame(mut self) -> Self {
        self.set("hold.enable", 0.);
        self
    }

    /// The value of `spec`: its override if set, else its default.
    #[must_use]
    pub fn value(&self, spec: &ParamSpec) -> f32 {
        self.0.get(spec.name).copied().unwrap_or(spec.default)
    }

    /// `base` settings with every override in `self` taking precedence: how a
    /// codec's own defaults for shared engine parameters
    /// ([`CodecBackend::DEFAULT_PARAMS`](crate::CodecBackend::DEFAULT_PARAMS))
    /// sit beneath the caller's presets and `--param` settings.
    #[must_use]
    pub fn layered_over(&self, base: &[(&str, f32)]) -> Self {
        let mut merged: BTreeMap<String, f32> = base
            .iter()
            .map(|&(name, value)| (name.to_owned(), value))
            .collect();
        merged.extend(self.0.iter().map(|(name, &value)| (name.clone(), value)));
        Self(merged)
    }

    /// The overrides whose names start with `prefix`, with the prefix
    /// removed, and the rest: how the non-base frame settings
    /// (`nonbase.<name>`) are separated from the ordinary ones.
    #[must_use]
    pub fn split_prefix(&self, prefix: &str) -> (Self, Self) {
        let (prefixed, rest): (BTreeMap<_, _>, BTreeMap<_, _>) = self
            .0
            .iter()
            .map(|(name, &value)| (name.clone(), value))
            .partition(|(name, _)| name.starts_with(prefix));
        let stripped = prefixed
            .into_iter()
            .map(|(name, value)| (name[prefix.len()..].to_owned(), value))
            .collect();
        (Self(stripped), Self(rest))
    }

    /// `self` with every override in `top` taking precedence.
    #[must_use]
    pub fn overlaid_with(&self, top: &Self) -> Self {
        let mut merged = self.0.clone();
        merged.extend(top.0.iter().map(|(name, &value)| (name.clone(), value)));
        Self(merged)
    }

    /// The names of the overridden parameters, in order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }

    /// Whether no parameter is overridden.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Every override must name a parameter in `specs` and lie in its range.
    ///
    /// # Errors
    /// The first override that does not.
    pub fn validate(&self, specs: &[&[ParamSpec]]) -> Result<(), ParamError> {
        for (name, &value) in &self.0 {
            let spec = specs
                .iter()
                .flat_map(|group| group.iter())
                .find(|s| s.name == name)
                .ok_or_else(|| ParamError::Unknown(name.clone()))?;
            if !(spec.min..=spec.max).contains(&value) {
                return Err(ParamError::OutOfRange {
                    name: name.clone(),
                    value,
                    min: spec.min,
                    max: spec.max,
                });
            }
        }
        Ok(())
    }
}

/// Declare a group of tunable parameters: a struct of `f32` fields, its
/// [`ParamSpec`]s (`SPECS`, named `prefix.field`) and a constructor that
/// reads overrides. Each field is `name: default, [min, max], "doc"`.
#[macro_export]
macro_rules! param_group {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident ($prefix:literal) {
            $( $(#[$fmeta:meta])* $field:ident : $default:expr, [$min:expr, $max:expr], $doc:literal ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq)]
        $vis struct $name {
            $( $(#[$fmeta])* #[doc = $doc] pub $field: f32, )*
        }

        impl $name {
            /// The group's parameters, in declaration order.
            pub const SPECS: &'static [$crate::params::ParamSpec] = &[
                $( $crate::params::ParamSpec {
                    name: concat!($prefix, ".", stringify!($field)),
                    default: $default,
                    min: $min,
                    max: $max,
                    doc: $doc,
                }, )*
            ];

            /// The group's values: overrides where set, defaults elsewhere.
            #[must_use]
            pub fn from_overrides(overrides: &$crate::params::ParamOverrides) -> Self {
                let mut specs = Self::SPECS.iter();
                Self {
                    $( $field: overrides.value(specs.next().expect("one spec per field")), )*
                }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::from_overrides(&$crate::params::ParamOverrides::default())
            }
        }
    };
}

param_group! {
    /// The partition planner (`partition_plan.rs`, SVT's very-light PD0 and
    /// depth removal).
    pub struct PlannerParams ("planner") {
        /// See `partition_plan.rs` for the measurements behind the default.
        lambda_scale: 2.0, [0.25, 8.0],
            "planner λ multiple of the mode-decision λ: higher splits less",
        distortion_cap_per_lambda: 0.125, [0.0, 1.0],
            "per-sample distortion cap as a fraction of planner λ: higher lets noisy blocks split",
        split_rate_factor: 2.0, [0.5, 4.0],
            "multiple of the split symbol's bits: higher splits less",
        /// See `Planner::leaf_bits`.
        leaf_bits: 0.0, [0.0, 64.0],
            "mode and vector bits charged to every planned inter leaf (0 = the backend's estimate): higher splits less, as each extra block pays its signalling",
        /// See `Planner::distortion`.
        dc_bits: 4.0, [0.0, 256.0],
            "bits a leaf's mean residual costs to code, in planner λ: its mean error counts as distortion up to this cost (0 = free, SVT's variance): higher splits less where coarse quantisation leaves block means uncoded",
        bias_base: 1100.0, [800.0, 1400.0],
            "parent cost multiplier (‰) before quantiser slope: higher favours splitting",
        bias_q_slope: 150.0, [0.0, 400.0],
            "parent cost multiplier reduction (‰) at the top quantiser: higher keeps more parents at high QP",
        bias_min: 900.0, [700.0, 1100.0],
            "lowest parent bias (‰)",
        bias_max: 1200.0, [1000.0, 1500.0],
            "highest parent bias (‰)",
        /// SVT's `HIGH_8x8_DIST_VAR_TH` (50000) capped too often for us; 3x
        /// keeps more 64x64 leaves where the planner's cost already prefers
        /// them. Real-use screen at k 0.1: AV1 proxy-6 `J_real` −0.6%, HEVC
        /// proxy-4 −0.3%, no guards (500000: AV1 −0.7%); 720p AV1
        /// `FoodMarket` 100 kbps MS-SSIM .769 → .773.
        high_8x8_cost_variance: 150_000.0, [5_000.0, 1_000_000.0],
            "8x8 cost spread above which blocks are capped at 32x32: lower caps more",
        depth_removal_th_32_16: 25.0, [0.0, 200.0],
            "depth removal 32→16 improvement threshold (‰): higher removes small blocks more",
        depth_removal_th_16_8: 50.0, [0.0, 400.0],
            "depth removal 16→8 improvement threshold (‰): higher removes small blocks more",
        depth_removal_qp_scale: 2.0, [0.0, 8.0],
            "depth removal threshold multiplier: higher removes small blocks more",
        depth_removal_min_quantizer: 0.0, [0.0, 1.0],
            "relative quantizer (0 finest, 1 coarsest) below which depth removal is skipped: higher keeps small blocks on offer at fine quantizers, at more planning work",
        /// 4 (16x16) is SVT M10+ (`partition_plan.rs`); 3 re-allows 8x8
        /// inter blocks, 5 stops at 32x32.
        inter_min_log2: 4.0, [3.0, 5.0],
            "smallest planned inter block inside the frame (log2, rounded): lower splits finer at extra planning and mode-decision work",
        /// 2 vs fixed 16x16 (with `av1.keyframe_chroma_scaled`), AV1 TRAIN6
        /// 150f: PSNR-YUV/V1 BD −3.78%/−5.04%, every clip better on both;
        /// about +110–215M instructions per 720p key frame.
        intra_plan: 2.0, [0.0, 2.0],
            "key frames: 0 codes the configured fixed block size, 1 plans the partition by source variance, 2 by the backend's transform-based leaf cost where it has one (AV1; others keep the fixed size), at extra planning work",
        /// 1050: the transform leaf cost predicts DC only, so it misses what
        /// smaller blocks gain from their own directions (a 64x64 VVC CU
        /// with 32-point units costs the same as its four 32x32 CUs). `VVenC`'s
        /// all-intra partitions in our encoder gain ~5% YUV; 1050, all-intra
        /// fixed-QP BD (4 TRAIN clips, PSNR-YUV): VVC −1.1%, AV1 −1.1%, HEVC
        /// −1.6%, VP9 −1.1%, AV2 −1.9%. Real-use screen at k 0.1, proxy-6:
        /// VVC `J_real` −0.6%, AV1 −0.4%, no guards. Best-of-four-directions
        /// leaf costs gain luma but lose chroma (+0.2% to +2.4% YUV), also
        /// with a reference-noise term (docs/handoff/res-indep.md).
        intra_bias: 1050.0, [800.0, 1400.0],
            "key-frame parent cost multiplier (‰): higher favours smaller intra blocks, spending more work and partition bits for finer prediction",
        intra_min_log2: 3.0, [3.0, 6.0],
            "smallest planned key-frame block (log2): higher is faster and cheaper to signal, but predicts detail worse",
        intra_flat_stop: 0.0, [0.0, 16.0],
            "key frames: a node whose leaf costs at most this many λ per 8x8 of its area is kept whole without planning its quadrants (0: off): higher is faster, keeping smooth areas in large blocks",
        /// Rectangular leaves (`partition_plan.rs` "Rectangular shapes"),
        /// only for backends declaring shapes.
        rect_min_log2: 3.0, [2.0, 6.0],
            "smallest side (log2) of a planned rectangular inter leaf: higher plans fewer, larger shapes at less planning work",
        rect_against_leaf: 1.0, [0.0, 1.0],
            "1 offers rectangular shapes where the square plan is one leaf as well as where it splits; 0 only replaces quad splits (fewer, cheaper shapes)",
        rect_bias: 1000.0, [800.0, 3000.0],
            "rectangular shape cost multiplier (‰) against the plan it competes with: higher plans fewer shapes (each extra leaf's signalling is not in the planner's cost)",
        rect_motion_gate: 1.0, [0.0, 16.0],
            "measure a rectangular shape only where every motion-search vector of one leaf differs from every one of the next by at least this many whole samples (0 measures every declared shape): higher is faster, with fewer shapes",
        rect_rd: 0.0, [0.0, 1.0],
            "1 has the backend's inter rate-distortion cost (sub-sample vector, transform, quantisation, coefficient and mode bits) confirm each shape the variance model plans against the square plan, 0 trusts the variance model: 1 rejects shapes that cost more to code, at a coded-cost trial per planned shape",
    }
}

param_group! {
    /// Inter-frame mode decision (`decision.rs`, SVT's light PD1).
    pub struct DecisionParams ("decision") {
        intra_gate_variance: 64.0, [0.0, 1024.0],
            "residual variance/sample above which DC intra is tried: lower tries intra more (slower)",
        // Default 1: 6 TRAIN clips x 4 QPs, 150f, PSNR-YUV/VMAF v1 BD vs 0:
        // H.264 -5.96%/-3.07% (+4.6% instructions), HEVC Fast -3.94%/-3.19%
        // (+0.7%), AV1 Fast -1.99%/-0.50% (-0.5%); antare -14% PSNR on HEVC.
        // Holds over whole single-key clips. See docs/handoff/hevc.md.
        intra_test_dc: 1.0, [0.0, 1.0],
            "1 adds the best inter residual's DC error to the intra gate and the intra-vs-inter cost (brightness changes, flashes, fades): more intra",
        intra_directional: 0.0, [0.0, 1.0],
            "1 also prices intra by a cheap directional estimate (best of DC, vertical, horizontal and the two diagonals per sub-block, from source neighbours; `intra_estimate.rs`), not DC alone: textured blocks with a direction reach the codec's intra search (slower where the intra gate opens)",
        intra_directional_dc_ratio: 4.0, [1.0, 16.0],
            "the directional intra estimate runs only where DC intra costs less than this multiple of the best inter candidate: higher estimates more blocks (slower)",
        intra_direction_bits: 3.0, [0.0, 64.0],
            "bits charged per sub-block direction in the directional intra estimate: higher sends fewer blocks to intra",
        skip_no_refine_lambdas: 0.25, [0.0, 1.0],
            "skip-vector residual (λ/sample) below which sub-pel refinement is skipped: higher is faster",
        // Engine motion defaults (2026-11-09): the measured quality point
        // AV1 and AV2 had each copied into their own `DEFAULT_PARAMS`
        // (measurements in `nc_av1`'s `DEFAULT_PARAMS`). A codec overrides
        // only what it has measured or not yet screened (H.264, VP9 and
        // HEVC's margins keep the legacy values), so a new codec no longer
        // silently runs the legacy search.
        codec_fullpel_probe: 1.0, [0.0, 1.0],
            "exact-filter codecs: test four neighboring integer predictors on reconstruction and retain the best as an extra NEW candidate; 0 disables",
        codec_fullpel_probe_min_lambdas: 0.0, [0.0, 64.0],
            "skip integer-neighbor probes below this seed residual variance/sample lambda multiple: 0 preserves unrestricted probes; higher saves work and can lose prediction quality",
        codec_subpel_seed_lambdas: 0.03, [0.0, 64.0],
            "exact-filter codecs: skip NEW sub-pel refinement when seed residual variance/sample is below this lambda multiple (0 disables)",
        codec_subpel_seed_lambdas_large: 0.0, [0.0, 64.0],
            "exact-filter codecs: the seed gate for blocks of 32x32 and larger (the larger of this and the general seed gate): higher refines fewer large blocks (faster)",
        subpel_quarter_needs_half_move: 0.0, [0.0, 1.0],
            "1 skips the quarter-sample stage when no half-sample move won (four of the eight refinement evaluations): faster, fewer quarter-sample vectors",
        // A disposable-frame lever (h264-faster3): stopping after the
        // half-sample cross saves the quarter stage's four evaluations per
        // refined block; the coded predictions stay at whole or half
        // samples. Measured in docs/handoff/h264-faster.md.
        subpel_half_only: 0.0, [0.0, 1.0],
            "1 ends the fixed-stage sub-sample search after its half-sample stage (the `decision.subpel_walk` search is unaffected): faster, coarser vectors",
        subpel_walk_max_noise_sigma: 255.0, [0.0, 255.0],
            "retain the legacy sub-pel search above this estimated sensor-noise sigma: lower limits coordinate-walk refinement to cleaner input",
        subpel_walk: 0.0, [0.0, 2.0],
            "exact-filter motion search: 0 fixed crosses, 1 quarter-pel coordinate walk, 2 SVT initial-offset bias and walk",
        subpel_move_bias: 1.0, [1.0, 1.5],
            "initial-move margin: legacy search biases absolute-zero moves; SVT walk biases the first move from the ME seed",
        subpel_move_bias_nonzero: 1.0, [1.0, 1.5],
            "legacy search factor for moves from a non-zero vector; source-correct SVT walk does not use this field",
        subpel_fractional_bias: 1.0, [1.0, 1.5],
            "legacy search factor for moves from a sub-sample position (no added interpolation work); lower accepts more quarter-sample refinements",
        lambda_scale: 1.0, [0.5, 2.0],
            "mode-decision λ multiple: higher favours cheap (predicted) vectors and DC intra less",
        base_period: 4.0, [1.0, 16.0],
            "every Nth frame is a 'base' frame (SVT flat-IPP pseudo-layers): `base_lambda_scale` applies there",
        base_lambda_scale: 1.0, [0.5, 1.0],
            "decision and planner λ multiple on base frames: lower spends more bits on them (finer references for the next frames)",
        // Disposable frames (temporal layers): their errors reach no later
        // frame, so cheap settings there cost no propagated quality.
        nonbase_disposable_only: 0.0, [0.0, 1.0],
            "frames the nonbase.* settings apply to: 0 every non-base frame of the cadence, 1 only frames no later frame references (the top temporal layer)",
        refine_cheap_vector: 1.0, [0.0, 1.0],
            "exact-filter codecs: 1 also refines the best cheap (NEAREST/NEAR) vector to sub-sample precision and ranks it as a NEW candidate; 0 refines only the search seed",
        refine_cheap_min_lambdas: 0.05, [0.0, 64.0],
            "with the cheap-vector refinement on, refine the cheap vector only when its residual variance/sample is at least this lambda multiple: higher is faster",
        seed_refine_ratio: 0.0, [0.0, 4.0],
            "with the cheap-vector refinement on, each of the two sub-sample seeds (search vector, best cheap vector) is refined only when its whole-sample distortion is at most this multiple of the other's; 0 refines both: lower is faster",
        // Speed levers for the Fast preset search (engine-cpu, 2026-11-09;
        // defaults unchanged). P-frame instructions, crowd_mot17_09 720p,
        // AV1 q180 / HEVC QP 36: max_cheap_vectors 2 -5.0% / -1.7%, 1
        // -10.5% / -7.6%; exact_rank_margin 1.3 -4.8% / -2.8%, 1.1 -6.5% /
        // -3.8% (AV2 -2.0%), 1.0 -8.3% / -5.2%. Quality:
        // docs/handoff/engine-cpu.md.
        max_cheap_vectors: 16.0, [1.0, 16.0],
            "most codec-predicted (NEAREST/NEAR, merge) vectors ranked per block, in the codec's order (16 = all): lower is faster, misses cheap predictors",
        // Default 1.1 (2026-11-09): J_real (CBR ladder) AV1 proxy-6 150f
        // J-full +0.1% / J-low -0.1%, no guard, -6.5% P instructions: BETTER.
        // HEVC, AV2 and H.264 pin 0 (their screens at the backend).
        exact_rank_margin: 1.1, [0.0, 4.0],
            "exact-filter ranking: skip a sub-sample candidate's exact prediction when its bilinear cost exceeds this multiple of the best exact cost so far (0 = never): lower is faster",
        new_rate_scale: 1.0, [0.0, 1.0],
            "multiple of a NEW vector's mode+MV bits in the ranking (SVT `approx_inter_rate` prices NEW cheaply): lower picks NEW more",
        // SVT M9+ `skip_intra` (enc_mode_config.c: norm_me_dist <= 5000,
        // the mean over 64x64 blocks of the summed best 8x8 SADs, about 1.2
        // SAD per sample). A Fast lever; default off.
        intra_frame_skip_me_dist: 0.0, [0.0, 65_536.0],
            "P frames whose mean per-superblock 8x8 motion-search distortion (SVT `norm_me_dist`) is at most this try no intra (0 = off): higher skips intra in more frames (faster)",
        // A Fast lever (default off): static content should cost almost
        // nothing to decide. Since 2026-11-09 it reads the stillness
        // detector (`still.*`: static for several frames, away from
        // motion), not the one-frame pre-analysis test, which also fired on
        // crowd backgrounds next to moving people (AV1 Fast train6 crowd
        // +7.3 J points). The residual is still coded, so the backend's own
        // skip tests decide whether anything is sent.
        // The noise skip (`noise_skip.rs`): off by default; calibration in
        // docs/analysis/static-noise-2026-11-14.md.
        noise_skip: 0.0, [0.0, 8.0],
            "codes a still block as a zero-vector skip when its zero-vector residual's mean squared error is at most this multiple of the cell's temporal noise variance, in every 8x8 cell (0 = off): higher stops coding sensor noise sooner, at a reconstruction further from the scene",
        noise_skip_chroma_ratio: 1.0, [0.25, 4.0],
            "the noise skip's chroma bound: the luma noise sigma times this (pre-analysis measures no chroma noise): higher lets chroma noise pass",
        static_zero_vector: 0.0, [0.0, 1.0],
            "1 codes inter blocks the stillness detector (`still.*`) marks still with the zero vector, skipping candidate search, sub-sample refinement and the intra trial: faster on static content",
        // Encoder-side approximation (a Fast lever; default exact): the
        // ranking only needs relative distortions, and image rows are
        // strongly correlated. Sub-sample bilinear measurements keep every
        // row (their vertical filter needs adjacent rows).
        distortion_row_step: 1.0, [1.0, 2.0],
            "mode decision measures whole-sample and exact-filter candidate distortion on every Nth luma row (1 or 2, rounded): 2 halves that work, coarser ranking",
        psy_strength: 0.0, [0.0, 2.0],
            "psychovisual term: inter candidates pay this multiple of the AC energy their prediction loses (or adds) against the source, DC intra its whole energy (x264/x265 psy-rd; exact-filter codecs; 0 = off): higher keeps texture",
    }
}

param_group! {
    /// Source motion search (`nc_analysis::motion`, SVT M12's HME + full-pel
    /// window). The defaults are SVT's search; codecs that only use small
    /// block vectors may run a lighter one (`CodecBackend::DEFAULT_PARAMS`).
    pub struct AnalysisParams ("analysis") {
        window_half_width: 4.0, [1.0, 4.0],
            "full-pel window half-width (x in -w..=w-1 around the HME result): lower is faster, worse vectors",
        window_half_height: 1.0, [0.0, 1.0],
            "full-pel window half-height (y in -h..=h): lower is faster, worse vectors",
        hme_level1: 1.0, [0.0, 1.0],
            "1 refines the quarter-resolution HME vector at half resolution before the full-pel window; 0 is faster",
        mixed_motion: 1.0, [0.0, 1.0],
            "1 adds full-pel positions for superblocks with mixed motion (crowds); 0 is faster",
        // Speed levers for the Fast preset search (engine-cpu, 2026-11-09);
        // defaults are the unchanged search. Measured P-frame instructions
        // (callgrind, crowd_mot17_09 720p, AV1 q180, analysis 8.76M of 104.3M):
        // HME range 8x2 -0.98% (analysis -0.82M); window exit 3 -0.36%;
        // zz gates 2/3/4 -1.11%; level-1 variance gate 1000: no change on
        // crowd (its mean variance is below 1000). For comparison, the
        // existing levers (HEVC QP 36): window 8x3 -> 2x1 -2.3%, level 1 off
        // -0.13%, mixed motion off -0.23%. Quality: docs/handoff/engine-cpu.md.
        window_exit_sad_per_pixel: 0.0, [0.0, 8.0],
            "full-pel window stops after its centre when the 64x64 SAD per pixel there is below this (0 = never): higher is faster, worse vectors",
        hme_range_x: 16.0, [1.0, 16.0],
            "HME window half-width in quarter-resolution samples (x4 luma): lower is faster, misses fast motion until it is seeded",
        hme_range_y: 4.0, [0.0, 4.0],
            "HME window half-height in quarter-resolution samples (x4 luma): lower is faster, misses vertical motion until it is seeded",
        hme_level1_max_variance: 0.0, [0.0, 10000.0],
            "HME level 1 only on frames whose mean 8x8 luma variance is below this (0 = every frame; SVT M12 uses 1000): lower is faster",
        static_sad_per_pixel: nc_analysis::motion::STATIC_SAD_PER_PIXEL, [0.0, 4.0],
            "superblocks whose zero-vector SAD per pixel is below this are static (no search): higher is faster",
        one_position_sad_per_pixel: nc_analysis::motion::ONE_POSITION_SAD_PER_PIXEL, [0.0, 6.0],
            "below this zero-vector SAD per pixel the full-pel search is the zero vector alone: higher is faster",
        skip_hme_sad_per_pixel: nc_analysis::motion::SKIP_HME_SAD_PER_PIXEL, [0.0, 8.0],
            "below this zero-vector SAD per pixel HME is skipped and the window centres on zero: higher is faster",
        // A disposable-frame lever (`nonbase.analysis.hme_from_seed` with
        // `decision.nonbase_disposable_only`): such a frame then builds no
        // pyramid either. Measured in docs/handoff/h264-faster.md.
        hme_from_seed: 0.0, [0.0, 1.0],
            "1 centres the full-pel window on the previous frame's HME vectors rescaled to this frame's distance instead of a new HME search (frames nothing references then build no pyramid): faster, misses changes of motion",
        // Speed lever (`nc_analysis::motion::ZeroSads`):
        // skips re-reading both frames for the zero-vector SADs, whose
        // values become exact instead of even-row estimates. Measured in
        // docs/handoff/lean-preanalysis.md.
        // Speed levers (`nc_analysis::preanalysis` "Even rows",
        // `nc_analysis::pyramid::Downsample`): each halves the luma rows its
        // pass reads; the pre-analysis pass is memory-bound on the
        // Cortex-A78. Measured in docs/handoff/lean-preanalysis.md.
        preanalysis_rows: 0.0, [0.0, 1.0],
            "1 takes pre-analysis' statistics (moments, change against the previous frame, chroma) from even rows only, scaled: faster, noisier statistics",
        pyramid_downsample: 0.0, [0.0, 2.0],
            "half-resolution search level: 0 2x2 box means, 1 even rows with horizontal pair means, 2 even rows and columns (decimated): higher reads fewer rows, aliases more",
        zero_sads_from_pre: 0.0, [0.0, 1.0],
            "1 takes the motion search's zero-vector SADs (zz gates, static and one-position SB grids) from pre-analysis' exact 8x8 SADs instead of measuring them again: faster",
    }
}

param_group! {
    /// The stillness detector (`nc_analysis::still`) the static-content
    /// levers read (`decision.static_zero_vector`,
    /// `layers.adaptive_min_still`). It changes nothing on its own.
    /// Classifier measurements against a long-horizon oracle (TRAIN, 150
    /// frames; `docs/handoff/static-detect.md`): per-frame pre-analysis
    /// static (run 1, guard 0) has precision 0.45 (crowd 0.31), run 8 with
    /// guard 1 0.82 (crowd 0.56, static 0.97) at recall 0.80.
    pub struct StillParams ("still") {
        min_run: nc_analysis::still::DEFAULT_MIN_RUN as f32, [1.0, 32.0],
            "consecutive static frames before a block counts as still: higher is stricter",
        guard: nc_analysis::still::DEFAULT_GUARD as f32, [0.0, 3.0],
            "radius in 8x8 blocks that must also be static this frame: higher keeps further from motion",
    }
}

impl StillParams {
    /// The analysis crate's stillness configuration for these values.
    #[must_use]
    pub fn still_config(&self) -> nc_analysis::still::StillConfig {
        nc_analysis::still::StillConfig {
            min_run: self.min_run.round().clamp(1.0, f32::from(u8::MAX)) as u8,
            guard: self.guard.round().clamp(0.0, f32::from(u8::MAX)) as u8,
        }
    }
}

param_group! {
    /// The persistence term of block AQ (`persistence.rs`, `aq.rs`): finer
    /// quantizers where past frames predict a block well, so it is coded
    /// well once and then copied, coarser where nothing predicts it. Only
    /// backends with block-level QP use it; rate control budgets the
    /// offsets with the variance term's.
    pub struct PersistParams ("persist") {
        strength: 0.0, [0.0, 3.0],
            "QP units per octave of a block's average motion-search error away from the frame's mean (0 = off): higher moves more bits from unpredictable to predictable blocks",
        reuse: 1.0, [0.0, 1.0],
            "1 quantizes still blocks at the step their measured reuse warrants (lambda / (1 + reuse)) whenever their reconstruction is worse than that step delivers (`reuse.rs`): 0 is off",
        memory: 0.75, [0.0, 0.95],
            "weight of the history in each block's moving-average motion error, per frame: higher reacts slower and flickers less",
    }
}

param_group! {
    /// Per-layer fixed-QP ladder of hierarchical-P structures
    /// (`structure.rs`; only read when `temporal_layers` > 1 and rate
    /// control is off; rate control applies its own `cbr_*` ladder around
    /// its level). Expressed as quantizer-step
    /// ratios to the top layer's step so one setting serves every codec. T0
    /// keeps the configured quantizer and upper layers are coarser
    /// (`qp_cadence.rs::LayerLadder`). The starting point was SVT-AV1's
    /// layered CQP ladder (`docs/reference/svt-av1-m12-rtc.md` §6: TL0
    /// x0.40, TL1 x0.70 of the top layer); fixed QP and rate control are
    /// tuned separately since they prefer different T0 boosts.
    pub struct LayerParams ("layers") {
        // SVT's 0.40. Fixed-QP screens (adaptive 3 layers): 0.3 is worse in
        // all three codecs (4 TRAIN clips x 60 frames). 0.5 looked better
        // at 60 frames (H.264 -0.75% PSNR / -1.55% VMAF), but loses in
        // full-length single-key runs (H.264 meva + crowd_pp22_027: +0.88% /
        // +0.34%; crowd +2.9%): over long GOPs a finer T0 pays back. 0.6 is
        // worse still.
        base_step_ratio: 0.40, [0.1, 1.0],
            "fixed QP: T0 quantizer step as a fraction of the top layer's: lower codes the upper layers coarser relative to T0, spending fewer bits on frames nothing references",
        middle_step_ratio: 0.70, [0.1, 1.0],
            "fixed QP: T1 quantizer step as a fraction of the top layer's in 3-layer structures: lower spends more bits on the frames the top layer predicts from",
        // CBR, L1T3, single key, 240 frames, staged proxy-6 vs 0.40 (PSNR /
        // VMAF BD, worst-1% PSNR / VMAF): AV1 -2.2% / -1.7% (+0.1% / +0.6%),
        // H.264 -2.1% / -1.6% (-0.3% / +1.7%), HEVC -3.4% / -2.1% (-0.2% /
        // +0.3%). 0.30 gains more on the mean
        // (AV1 -3.5% / -2.3%) but costs worst-1% VMAF up to +5.5% (flicker).
        cbr_base_step_ratio: 0.35, [0.1, 1.0],
            "rate control: T0 step fraction of the top layer's around the controller's level (CBR prefers more T0 boost than fixed QP)",
        cbr_middle_step_ratio: 0.70, [0.1, 1.0],
            "rate control: T1 step fraction of the top layer's in 3-layer structures",
        // VAL-150 (staged, full gate) vs flat: H.264 -11.5% PSNR / -9.0%
        // VMAF, HEVC -6.9% / -10.9%, AV1 -5.3% / -2.2%
        // (docs/handoff/hierarchical-p.md); only read with --temporal-layers > 1.
        adaptive: 1.0, [0.0, 1.0],
            "1 chooses flat or the full --temporal-layers pattern per mini-GOP from camera and scene motion (switching at T0 frames); 0 keeps the configured structure",
        adaptive_max_camera_motion: 2.0, [0.0, 16.0],
            "adaptive: largest camera motion (luma samples/frame) still layered: higher layers more moving-camera content",
        adaptive_max_compensable: 0.68, [0.0, 1.0],
            "adaptive: largest compensable share of the frame difference still layered: higher layers more scene motion",
        adaptive_enter_ratio: 0.9, [0.5, 1.0],
            "adaptive: switch from flat to layered only below this fraction of both limits: lower switches less often (hysteresis)",
        // Static-content gate (static-detect agent): crowd scenes pass the
        // motion limits part of the time and then flicker in the coarse top
        // layer; few of their blocks are still (`still.*`).
        adaptive_min_still: 0.0, [0.0, 1.0],
            "adaptive: smallest still-block fraction (`still.*`, a running average) of a layered mini-GOP (0 = no gate): higher layers only more static scenes",
        // Layer gate (layer-gate agent, 2026-11-11; fitted on 44 fresh
        // clips in neither TRAIN nor VAL, docs/handoff/hierarchical-p.md
        // "Layer gate").
        adaptive_max_repeat_share: 1.0, [0.0, 1.0],
            "adaptive: largest share of repeated input frames (running average) of a layered mini-GOP (1 = no gate, not detected): lower keeps sources with fewer repeats flat",
        adaptive_max_static: 1.0, [0.0, 1.0],
            "adaptive: largest static-block fraction (distance-1 frames, running average) of a layered mini-GOP (1 = no gate): lower keeps more static scenes flat",
        adaptive_max_camera_jitter: 64.0, [0.0, 64.0],
            "adaptive: largest camera jitter (change in camera motion between distance-1 frames two apart, luma samples/frame, running average) of a layered mini-GOP (64 = no gate): lower keeps more shaking cameras flat",
        // Noisy-still gate (layer-gate re-fit, 2026-11-11): a still scene
        // whose frame difference is large uncompensable temporal noise
        // (`static_otw_homes390`) loses with layering in every gated preset.
        adaptive_max_noise_residual: 64.0, [0.0, 64.0],
            "adaptive: largest motion-compensated residual (distance-1 best-match SAD per luma sample, running average) of a layered mini-GOP whose compensable share is below `adaptive_noise_compensable` (64 = no gate): lower keeps more noisy still scenes flat",
        adaptive_noise_compensable: 0.2, [0.0, 1.0],
            "adaptive: compensable share (running average) below which `adaptive_max_noise_residual` applies: higher treats more moving content as noisy still",
        // A parameter so that presets can turn layering on; the stream
        // setting (`EncoderConfig::temporal_layers`, `--temporal-layers`)
        // applies only while this is not set (`Encoder::new`).
        count: 1.0, [1.0, crate::structure::MAX_TEMPORAL_LAYERS as f32],
            "temporal layers when set (a preset or --param): 1 flat, 2 L1T2, 3 L1T3; it takes precedence over --temporal-layers",
    }
}

impl LayerParams {
    /// The fixed-QP ladder for `layers` temporal layers.
    #[must_use]
    pub(crate) fn fixed_ladder(&self, layers: u8) -> crate::qp_cadence::LayerLadder {
        crate::qp_cadence::LayerLadder::new((self.base_step_ratio, self.middle_step_ratio), layers)
    }

    /// The rate-control ladder for `layers` temporal layers.
    #[must_use]
    pub(crate) fn rate_control_ladder(&self, layers: u8) -> crate::qp_cadence::LayerLadder {
        crate::qp_cadence::LayerLadder::new(
            (self.cbr_base_step_ratio, self.cbr_middle_step_ratio),
            layers,
        )
    }

    /// The adaptive structure rule, if enabled.
    #[must_use]
    pub fn adaptive_rule(&self) -> Option<crate::structure::AdaptiveRule> {
        (self.adaptive >= 0.5).then_some(crate::structure::AdaptiveRule {
            max_camera_motion: self.adaptive_max_camera_motion,
            max_compensable: self.adaptive_max_compensable,
            min_still: self.adaptive_min_still,
            max_repeat_share: self.adaptive_max_repeat_share,
            max_static: self.adaptive_max_static,
            max_camera_jitter: self.adaptive_max_camera_jitter,
            max_noise_residual: self.adaptive_max_noise_residual,
            noise_compensable: self.adaptive_noise_compensable,
            enter_ratio: self.adaptive_enter_ratio,
        })
    }
}

impl AnalysisParams {
    /// Which rows pre-analysis' statistics read.
    #[must_use]
    pub fn stat_rows(&self) -> nc_analysis::preanalysis::StatRows {
        if self.preanalysis_rows >= 0.5 {
            nc_analysis::preanalysis::StatRows::Even
        } else {
            nc_analysis::preanalysis::StatRows::All
        }
    }

    /// How the pyramid's half level is made.
    #[must_use]
    pub fn downsample(&self) -> nc_analysis::pyramid::Downsample {
        match self.pyramid_downsample.round() as u8 {
            0 => nc_analysis::pyramid::Downsample::Filtered,
            1 => nc_analysis::pyramid::Downsample::EvenRowsMean,
            _ => nc_analysis::pyramid::Downsample::Decimated,
        }
    }

    /// The analysis crate's search configuration for these values.
    #[must_use]
    pub fn search_config(&self) -> nc_analysis::motion::SearchConfig {
        nc_analysis::motion::SearchConfig {
            window_half_width: self.window_half_width.round() as i16,
            window_half_height: self.window_half_height.round() as i16,
            window_exit_sad_per_pixel: self.window_exit_sad_per_pixel,
            hme_range_x: self.hme_range_x.round() as i16,
            hme_range_y: self.hme_range_y.round() as i16,
            hme_level1: self.hme_level1 >= 0.5,
            hme_level1_max_variance: if self.hme_level1_max_variance > 0.0 {
                self.hme_level1_max_variance
            } else {
                f32::INFINITY
            },
            mixed_motion: self.mixed_motion >= 0.5,
            hme_from_seed: self.hme_from_seed >= 0.5,
            zero_sads: if self.zero_sads_from_pre >= 0.5 {
                nc_analysis::motion::ZeroSads::PreAnalysis
            } else {
                nc_analysis::motion::ZeroSads::Measured
            },
            gates: nc_analysis::motion::ZeroVectorGates {
                static_per_pixel: self.static_sad_per_pixel,
                one_position_per_pixel: self.one_position_sad_per_pixel,
                skip_hme_per_pixel: self.skip_hme_sad_per_pixel,
            },
        }
        .clamped()
    }
}

param_group! {
    /// Frame-level quantizer policy shared by fixed QP and rate control
    /// (`qp_cadence.rs`, `rate_control.rs`).
    pub struct RateParams ("rate") {
        cadence_period: 4.0, [1.0, 16.0],
            "pseudo-layer period (SVT flat IPP uses 4); 1 codes every P frame at the same quantizer",
        cadence_base_boost: 3.0, [0.0, 12.0],
            "QP units finer on every period-th P frame: higher refines inherited background more but spends more there",
        cadence_startup_period: 0.0, [0.0, 16.0],
            "pseudo-layer period of the first `rate.cadence_startup_frames` after a key (0 = `rate.cadence_period` throughout): a deeper start-up cadence refines what the frames after a key inherit, the ordinary period keeps long static scenes from spending most bits on rare base frames",
        cadence_startup_base_boost: 0.0, [0.0, 12.0],
            "with `rate.cadence_startup_period`, QP units finer on its base frames (0 = `rate.cadence_base_boost`)",
        cadence_startup_frames: 100.0, [0.0, 3600.0],
            "with `rate.cadence_startup_period`, frames after a key it covers (rounded up to its next base frame)",
        cadence_motion_gate: 0.0, [0.0, 1.0],
            "1 also gives the full cadence to frames with little camera motion and a low compensable share (clean low-motion scenes whose still-block share is low, e.g. conference); 0 scales it by the still share alone",
        cadence_max_camera_motion: 2.0, [0.0, 16.0],
            "motion gate: camera motion (luma samples per frame) at which it stops adding strength (full below 75% of it): higher keeps the cadence under more camera motion",
        cadence_max_compensable: 0.68, [0.0, 1.0],
            "motion gate: compensable share of the frame difference at which it stops adding strength (full below 75% of it): higher keeps the cadence on more compensable motion",
        cadence_leaf_penalty: 1.0, [0.0, 6.0],
            "QP units coarser on odd-position P frames: higher saves bits on frames nothing at base level refers back to",
        first_key_boost_scale: 2.5, [1.0, 8.0],
            "rate control: first key budget multiple of the normal key budget: higher codes the picture every later frame inherits finer, at more start-up debt",
        first_key_lookahead: 0.0, [0.0, 2.0],
            "rate control: when the caller hands over the second frame before the first (`Encoder::look_ahead`) and it shows camera motion, the first key takes at most this many buffers of bits (0 = the backend's value): lower starts moving scenes coarser and spends the bits on the frames that follow",
        first_key_lookahead_off: 0.0, [0.0, 1.0],
            "1 turns the start-up lookahead's first-key cap off whatever the backend's value",
        // Off: H.264 J_real proxy-4 at 0.3 -2.9% but crowd +5.6% (guard),
        // at 0.2 -2.2%, crowd +9.2% (pan -12..-14%).
        key_cap_buffer: 0.0, [0.0, 4.0],
            "rate control: most bits any key may take, as a multiple of the buffer (0 = none): lower keeps the first key from filling the receiver's buffer, at a coarser start",
        // Off: CBR GOP 60 (2 s), 150 f, 6 TRAIN clips, 0.3 vs off,
        // PSNR/VMAF BD: H.264 +1.8/+1.4% (overrun frames 7 -> 17), VP9
        // +2.5/+1.5% (60 -> 4); lowlight_city_night +6..+11%. Since
        // rate.recode_overflow the static short-GOP keys are already smaller.
        key_cap_gop: 0.0, [0.0, 1.0],
            "rate control: most bits a key may take, as a share of its repayment phase's bits (the key period, at most the repay horizon; 0 = none): lower keeps short GOPs of static content on target, where the key level floor would otherwise spend the whole GOP on the key",
        key_repay_seconds: 5.0, [0.1, 60.0],
            "rate control: longest time over which a key's extra bits are repaid by the following P frames (the key period if shorter), and the debit term's horizon (`rate.debit`): lower recovers sooner after a key in long or single-key streams",
        // A still scene's key is referenced for its whole period, so it may
        // be repaid more slowly; a moving one's frames after the key need
        // the bits back soon. Tuner, 720-frame long-horizon clips (H.264
        // Default, repay 7.7-8.1 s for every scene): static J -5.1..-8.0%
        // with worst-1% much better, crowd worst-1% VMAF +8.2..+8.3%.
        key_repay_still_seconds: 0.0, [0.0, 60.0],
            "rate control: the key repay horizon of a still scene (0 = `rate.key_repay_seconds`); each key's horizon blends linearly from `rate.key_repay_seconds` to this as the running share of unchanged blocks rises from `rate.key_repay_still_from` to `rate.key_repay_still_full`; the debit term keeps `rate.key_repay_seconds`: higher repays still scenes' keys more slowly, keeping their quality up for longer",
        key_repay_still_from: 0.5, [0.0, 1.0],
            "rate control, with `rate.key_repay_still_seconds`: share of unchanged 8x8 blocks below which a key repays over `rate.key_repay_seconds`",
        key_repay_still_full: 0.9, [0.0, 1.0],
            "rate control, with `rate.key_repay_still_seconds`: share of unchanged 8x8 blocks from which a key repays over the still horizon",
        // The horizon 10 -> 5 s with the debit term: 600-frame long clips
        // (keys every 10 s) AV1 J +0.12% at 5 s, +0.15% at 10 s against
        // the proportional controller; at 150 frames 5 s leaves 3-12 of
        // 50 VAL ladder points over the target, 10 s 40-46.
        debit: 1.0, [0.0, 1.0],
            "rate control: 1 adds the long-term debit term, which repays any spend above the target line (beyond the scheduled key loan) over `rate.key_repay_seconds` and banks at most `rate.debit_credit` of undershoot, so the long-run rate stays at or under the target; 0 corrects the debt proportionally only, as before",
        // 0.25 broke the H.264 layered static_recovery check: a starved
        // key's static half recovers on the undershoot it banks.
        debit_credit: 1.0, [0.0, 2.0],
            "rate control, with `rate.debit`: most buffers of undershoot the debt keeps as credit for later frames: lower holds the rate further under the target after cheap stretches",
        // 600-frame long clips, AV1: 0.01 J +0.12% (static -0.05%),
        // 0.02 J +1.54% (static +4.5%); H.264/HEVC/AV1 points over the
        // target 15-18 -> 3-9 of 20 at 0.01 (worst +1.4%).
        debit_margin: 0.01, [0.0, 0.1],
            "rate control, with `rate.debit`: share of the target the debt line aims under it: higher lands long runs further under the target, at fewer bits",
        window_seconds: 0.0, [0.0, 10.0],
            "rate control: length of the short window no stretch of inter frames may exceed the target over by more than `rate.window_tolerance` (re-coded to fit; 0 = off)",
        window_tolerance: 0.03, [0.0, 2.0],
            "rate control, with `rate.window_seconds`: share of the target a window may exceed it by: lower holds short stretches more accurately, at smaller static refinement bursts",
        window_keys: 0.0, [0.0, 1.0],
            "rate control, with `rate.window_seconds`: 1 also bounds key frames by the window (they then borrow nothing beyond it); 0 exempts a key's window, its excess being a loan repaid on schedule",
        // x264's rate control does the same: its QP may reach 69 for
        // analysis while the coded QP stops at 51, which is how it reaches
        // targets our H.264 overshoots by 70% with every P frame at QP 51.
        // On by default (coordinator, 2026-11-08: a CBR target is a
        // ceiling, landing at 2x it is a rate-control failure whatever BD
        // says); it changes nothing until rate control hits the cap.
        // J_real proxy-4 (H.264, off vs 2): full -0.2%, low +2.4%. 3 octaves:
        // at 2, fast pans still saturated (pan_bdd_00cef86b at 150 kbps:
        // HEVC 191, H.264 172 kbps with every frame at the headroom's end;
        // at 3: 148 and 147).
        lambda_headroom_octaves: 3.0, [0.0, 4.0],
            "rate control may plan this many octaves of step past the codec's largest quantizer: frames stay coded at the largest, the decision lambda grows with the squared step (0 = off): higher reaches lower targets",
        scene_hint_ratio: 1.5, [1.0, 10.0],
            "rate control: a packet far off its prediction resets the rate model only if the frame's complexity hint also changed by this ratio (1 = any miss resets): higher ignores misses on unchanged content",
        // J_real VAL (H.264, 11 clips, CBR ladder, 150 f) 1.4 vs 0: J_real
        // +0.2% (full +0.1%, low +0.2%), no guard; non-compliant points
        // 2 -> 0, overrun frames 184 -> 99; only static_london_bus changes
        // (J +1.7/+3.2%, worst-1% -1.1/-1.6%), the other 10 clips are
        // byte-identical (no re-code). At 1, static_cam22 cost +3.5%.
        recode_overflow: 1.4, [0.0, 2.0],
            "rate control: codes a frame again, coarser, when it would fill a buffer drained at the target rate (a strict-CBR receiver, 1 s by default) past this multiple of its capacity, for backends that support re-coding (0 = never): 1 keeps the buffer, higher tolerates short overflows and re-codes less",
        recode_overflow_inter: 0.0, [0.0, 2.0],
            "rate control: the re-code ceiling of inter frames, as a multiple of the buffer (0 = the main re-code ceiling): lower keeps static refinement bursts inside a strict-CBR receiver's buffer, at the refinement they would have bought",
        aq_max_offset: 8.0, [1.0, 12.0],
            "variance AQ: largest block offset either way, in QP units: higher allows stronger per-block redistribution",
        aq_strength: 0.0, [0.0, 3.0],
            "variance AQ: QP units per octave of block variance away from the frame mean (0 = off): higher moves more bits from detailed to flat blocks",
        // One codec-independent form: the H.264 QP of the pixel-domain step
        // (0.625 * 2^(QP/6)), mapped per codec by `quality_cap.rs`, so one
        // value is one step in every codec. Equivalents (AV1 and VP9 by
        // their AC quantizer tables): QP 10 = step 2.0 = AV1/VP9 qindex 9;
        // QP 14 = 3.2 = qindex 19; QP 18 = 5.0 = qindex 33; QP 20 = 6.3 =
        // qindex 44; QP 22 = 7.9 = qindex 57; HEVC and VVC QP equal H.264
        // QP. Census of the 2026-11-14 L1 draw (every Default, every rung,
        // 150 f, plus the 600 f long list): no stream reaches QP 10 (42/42
        // sampled points byte-identical with it on); QP 14 binds on one
        // static long-window H.264 point (docs/analysis/static-noise-2026-11-14.md).
        max_quality_qp: 10.0, [0.0, 51.0],
            "rate control quality ceiling: finest quantizer step of any frame, as the H.264 QP of the same pixel-domain step (0 = none): higher stops spending bits sooner on easy content",
        noise_floor_scale: 0.0, [0.0, 32.0],
            "rate control quality ceiling: finest pixel-domain quantizer step of any frame, as this multiple of the temporal noise sigma measured on still blocks (0 = none; no floor while too little of the picture is still): higher stops coding sensor noise sooner",
        quality_floor_cadence: 0.0, [0.0, 1.0],
            "1 lets QP-cadence base frames go finer than the quality floor by their cadence offset (the floor bounds the inter level); 0 clamps every frame at the floor",
        // CBR 240 f single key, PSNR-YUV / VMAF v1 BD vs 0 (lower rate at
        // the same quality wins; the target is a ceiling): static/lowlight
        // set H.264 -5.3/-2.8%, HEVC -16.1/-11.9%, AV1 -11.1/-7.1%
        // (static_otw_homes390 -31..-46%, worst-1% -19..-27%); TRAIN staged
        // proxy-6 neutral (+0.1..-0.6%; moving clips never engage it); VAL
        // H.264 -10.1/-8.4%, HEVC -9.5/-9.2%, AV1 -5.6/-6.6%. Not tuned as a
        // quality lever: the long-GOP check failed. With no bound on the climb,
        // inter frames that inherit their key read as a flat slope at any QP,
        // so the level climbs to the maximum (long_static_chirla_s024c2 1800 f
        // at 150 kbps: 20 kbps, VMAF 92 -> 71). The default 10 is only a
        // backstop against coding noise (owner-approved, 2026-11-14): no
        // benchmark stream reaches it (static-noise census, 588 encodes; every
        // preset of every codec byte-identical on the 2026-11-14 sample).
        min_refinement_gain: 0.0, [0.0, 1.0],
            "rate control quality ceiling: luma PSNR (dB) per finer QP below which inter frames stop getting finer, measured online over the cadence (0 = off): higher underspends sooner on content whose quality has stopped improving",
        // CBR flicker set (3 lowlight, bodycam, crowd, pan; 150 f), J_real
        // vs 1: AV1 -0.1%, H.264 -0.0%, HEVC -0.3%, no guard tripped; mean
        // |dPSNR| -0.03..-0.05 dB (bodycam -0.10..-0.16), bodycam worst-1%
        // -26..-31%. 0.25 reacts too slowly (AV1 J_real +1.1%).
        level_step_scale: 0.5, [0.05, 2.0],
            "rate control: the per-frame limit on inter-level changes is the codec's step limit times this: lower smooths frame-to-frame quality, higher reacts faster",
        cadence_scale: 0.0, [0.0, 2.0],
            "QP cadence offsets multiplied by this (0 = the backend's own scale): lower gives a gentler cadence",
        cadence_static_scaling: 1.0, [0.0, 1.0],
            "1 scales cadence offsets by the frame's static-block share: higher removes the cadence under camera motion",
        // Fixed-QP TRAIN, staged proxy-6 (150 f) vs 0, PSNR-YUV / VMAF v1 BD:
        // H.264 -2.9% / -2.4%, HEVC -3.7% / -3.7%, AV1 -2.5% / -2.8%; full
        // length single key (4 clips, 240-300 f): H.264 -4.3% / -3.9%, HEVC
        // -5.2% / -5.7%, AV1 -3.4% / -3.7%; every clip gains in every codec.
        // The base frames' finer step raises the rate of a given --qp by
        // 2-12% at matched QP.
        cadence_fixed_qp: 1.0, [0.0, 1.0],
            "1 also applies the cadence at fixed QP; 0 keeps fixed-QP frames flat (the cadence then only shapes rate control)",
    }
}

param_group! {
    /// Scene-cut detection (`scene.rs`): a key frame where motion
    /// prediction stops working. Off by default.
    pub struct SceneParams ("scene") {
        detect: 0.0, [0.0, 1.0],
            "1 inserts a key frame at a detected scene cut; 0 codes cuts as inter frames (default streams unchanged)",
        cut_ratio: 1.8, [0.2, 8.0],
            "motion-compensated SAD over the intra-activity estimate above which a frame counts as a cut: higher fires only on cuts with nothing predictable left",
        history_ratio: 3.0, [1.0, 20.0],
            "jump in motion-compensated SAD per pixel over the recent average needed as well: higher ignores sustained hard content (noise, fast motion) more",
        min_key_interval: 8.0, [1.0, 300.0],
            "fewest frames after any key frame before a cut may insert another: higher avoids key bursts on flashes and strobes",
    }
}

param_group! {
    /// Frame hold (`hold.rs`): an input frame that would not change the
    /// picture is not coded and takes no place in the stream.
    pub struct HoldParams ("hold") {
        enable: 1.0, [0.0, 1.0],
            "1 holds the last picture instead of coding a frame that would not change it (nothing is sent; rate control, layers, key period and analysis skip it); 0 codes every input frame",
        change: 0.5, [0.0, 4.0],
            "largest SAD of any 8x8 luma block against the last coded frame's source, as a fraction of the current quantizer step, at which a frame is held (flat chains): at 0.5 no transform coefficient of the change reaches half a step; 0 holds exact repeats only",
        max_mean_abs: 1.0, [0.0, 16.0],
            "largest mean absolute change per sample of any 8x8 luma block at which a frame is held, whatever the quantizer: a coarse step must not hide real motion; 1 keeps holds to exact and near-exact (recompressed) repeats",
        recur_frames: 8.0, [0.0, 64.0],
            "a repeat is held only if another repeat came at most this many input frames before it (a repeat source); an isolated repeat (a duplicated first frame, one stall) is coded: it costs a near-empty frame, while holding it shifts rate control (held-out AV1 `J_real` -0.6..-6.3% when coded); 0 holds no frame",
        refine_octaves: 1.0, [0.0, 8.0],
            "octaves the last coded frame's step may exceed the current inter level's for a hold: a coarser last frame (a starved key) is coded so the encoder can refine it",
    }
}

param_group! {
    /// Intra refresh (`refresh.rs`) from the parameter registry, so presets
    /// and the evaluation harness can select it; `Encoder::set_intra_refresh`
    /// changes it at run time.
    pub struct RefreshParams ("refresh") {
        period: 0.0, [0.0, 600.0],
            "frames per intra-refresh sweep instead of periodic key frames (0 = off): higher spreads the refresh thinner but recovers from loss later",
    }
}

param_group! {
    /// Long-term reference with receiver feedback (`ltr.rs`); flat
    /// structures, codecs that support it.
    pub struct LtrParams ("ltr") {
        interval: 0.0, [0.0, 600.0],
            "frames between long-term reference marks (0 = off): lower keeps a more recent recovery point, at more marks awaiting acknowledgement",
    }
}

param_group! {
    /// A second reference beside the structure's (`second_reference.rs`):
    /// the last high-quality frame, kept in the long-term slot.
    pub struct RefParams ("refs") {
        second: 0.0, [0.0, 1.0],
            "1 lets inter blocks also predict from the last high-quality frame (key or cadence base), held in the long-term slot: helps where that frame is coded finer than the previous one, at a second mode decision per block",
        second_gate_lambdas: 0.0, [0.0, 64.0],
            "try the second reference only on blocks whose primary choice costs more than this many λ per sample (0 tries every block): higher is faster",
        second_refine_margin: 0.3, [0.0, 4.0],
            "cheap ranking: refine the second reference's best vector to sub-sample precision only when it costs within this fraction above the primary choice (0 refines only outright winners): higher is slower",
        second_max_distance: 64.0, [2.0, 64.0],
            "frames after the high-quality frame that may still reference it (64 = no limit): higher reaches further, at a second mode decision on more frames",
        second_min_still: 0.0, [0.0, 1.0],
            "still-block fraction below which a frame does not reference the high-quality frame (0 = every frame): higher limits it to fixed cameras, faster",
        second_full: 0.0, [0.0, 1.0],
            "1 runs the full mode decision (sub-sample refinement, intra trial) against the second reference; 0 ranks its zero, predicted and scaled-seed vectors only: 0 is faster",
        second_compound_margin: 0.15, [0.0, 4.0],
            "compound inter trial gate: try the equal-weight average of primary and secondary inter winners only when the worse single-reference cost is within this fraction of the better one (0 = off when neither ties; higher tries compound on more blocks)",
    }
}

param_group! {
    /// Trellis quantisation of the coded block (`trellis.rs`), in backends
    /// that provide a coefficient rate model.
    pub struct QuantParams ("quant") {
        trellis: 0.0, [0.0, 1.0],
            "trellis quantisation (0 off, 1 the winning block's coefficients after mode decision): 1 spends a rate-distortion search per coded transform block to round levels down or to zero where the bits are worth more",
        trellis_lambda_scale: 1.0, [0.25, 4.0],
            "multiple of the mode-decision λ the trellis trades bits at: higher drops more levels",
        trellis_chroma: 0.25, [0.0, 4.0],
            "multiple of the trellis λ for chroma (0 leaves chroma to the plain quantiser): lower keeps more chroma levels",
        trellis_key: 1.0, [0.0, 1.0],
            "1 also trellises key frames (0: P frames only): 0 keeps the key frame every later frame predicts from at the plain quantiser's quality",
        trellis_states: 4.0, [1.0, 16.0],
            "entropy-context states kept per scan position: higher searches more level combinations",
    }
}

param_group! {
    /// The engine's chroma policy (`chroma_policy.rs`): one luma/chroma
    /// balance for every codec, mapped by each backend to its own syntax.
    pub struct ChromaParams ("chroma") {
        policy: 0.0, [0.0, 1.0],
            "1 quantises chroma at the engine policy's step (luma's up to QP-eq 36, then up to 3 QP finer, the JVET CTC map) and weighs chroma error by the squared step ratio; 0 keeps the codec's own chroma offsets and weights",
        /// -1 (chroma 1 QP finer than the CTC map at every rate): better
        /// than 0 in every codec screened at both (staged TRAIN `J_real`,
        /// k 0.1: AV1 -0.3% -> -0.4%, H.264 +0.1% -> -0.2%, VVC +0.3% ->
        /// +0.1%, VP9 +0.2% -> +0.1%; docs/handoff/chroma-policy.md).
        offset: -1.0, [-6.0, 6.0],
            "QP-equivalent shift of the policy's chroma step (6 doubles it): higher codes chroma more coarsely",
    }
}

param_group! {
    /// Adaptive coded resolution (`resolution.rs`): half size per dimension
    /// while rate control plans past the codec's largest quantizer, for
    /// codecs with scaled references (AV1), switched without key frames.
    pub struct ResolutionParams ("resolution") {
        adaptive: 0.0, [0.0, 1.0],
            "1 codes at half size per dimension (no key frame; players still display the stream size) while rate control's level plans past the largest quantizer, and back at full size when the rate allows; codecs with scaled references only (AV1)",
        down_step_ratio: 1.25, [1.0, 8.0],
            "switch down when the full-size level's planned step averages this multiple of the largest quantizer's step",
        up_step_ratio: 0.8, [0.25, 1.0],
            "switch up when the half-size level's step times `half_step_gain` averages below this multiple of the largest step",
        half_step_gain: 2.0, [1.0, 8.0],
            "ratio of the full-size to the half-size planned step at the same rate (moves the average at a switch, and maps half-size levels for switching up)",
        window_seconds: 1.0, [0.1, 10.0],
            "time constant of the moving average of the planned step",
        dwell_seconds: 2.0, [0.0, 30.0],
            "shortest time at one size before switching again",
    }
}

param_group! {
    /// Spatial scalability (`spatial.rs`): a scaled base layer (`spatial_id = 0`)
    /// plus up to two enhancement layers (`spatial_id = 1, 2`) in one stream,
    /// for codecs that support it (AV1 and VP9: up to 3 layers).
    pub struct SpatialParams ("spatial") {
        layers: 1.0, [1.0, 3.0],
            "number of spatial layers (1 = single layer; 2 = S0 + full-size S1; 3 = quarter S0 + half S1 + full-size S2 on AV1 and VP9)",
        base_scale: 0.5, [0.25, 1.0],
            "two-layer base spatial layer size relative to the stream dimensions (0.5 = 2:1; e.g. 0.6667 = 720p->480p via polyphase Lanczos-3)",
        s0_scale: 0.25, [0.0625, 1.0],
            "three-layer S0 spatial layer size relative to the stream dimensions (0.25 with `s1_scale=0.5` = cascaded 2:1 steps, 1280x720->640x360->320x180)",
        s1_scale: 0.5, [0.125, 1.0],
            "three-layer S1 spatial layer size relative to the stream dimensions (0.5 = 2:1 step from full-size S2, 1280x720->640x360)",
        base_rate_share: 0.35, [0.1, 0.8],
            "two-layer fraction of the total CBR target capped to the base spatial layer S0: higher strengthens the base operating point, lower leaves more bits for S1",
        s0_rate_share: 0.08, [0.05, 0.6],
            "three-layer fraction of the total CBR target capped to the lowest spatial layer S0 (180p); unspent inter bits flow upward to S1 and S2",
        s1_rate_share: 0.26, [0.1, 0.7],
            "three-layer fraction of the total CBR target capped to the middle spatial layer S1 (360p); unspent inter bits flow upward to S2",
        // K-SVC nominal rate shares (`spatial.inter_layer = 2`, 3-clip J_real
        // screen on static_otw_homes390, pan_jaad_0101, crowd_pp22_051):
        // - 2-layer `0.32` (vs `0.35` / `0.28`): 720p view +5.50% (+5.90% / +5.23%),
        //   360p view -13.44% (-13.32% / -12.11%), total vs simulcast -18.58%
        //   (-18.18% / -19.01%).
        // - 3-layer `(0.08, 0.24)` (vs `(0.08, 0.26)` / `(0.07, 0.22)`): 720p view
        //   +4.21% (+3.76% / +6.76%), 360p view -8.43% (-7.53% / -7.00%), 180p view
        //   -1.62% (-1.77% / +0.93%), total vs simulcast3 -11.59% (-10.50% / -11.55%).
        key_base_rate_share: 0.32, [0.1, 0.8],
            "two-layer K-SVC (`inter_layer = 2`) fraction of the total CBR target capped to S0: without P-frame inter-layer prediction S1 does not amortise S0 inter bits, so a slightly lower cap than full SVC balances the 360p and 720p views",
        key_s0_rate_share: 0.08, [0.05, 0.6],
            "three-layer K-SVC (`inter_layer = 2`) fraction of the total CBR target capped to S0 (180p); unspent static inter bits flow upward to S1 and S2",
        key_s1_rate_share: 0.24, [0.1, 0.7],
            "three-layer K-SVC (`inter_layer = 2`) fraction of the total CBR target capped to S1 (360p), which a 360p K-SVC viewer receives without S0 on non-key units",
        inter_layer: 1.0, [0.0, 2.0],
            "inter-layer prediction mode: 1 = full SVC (every TU, AV1 §7.11.3.3); 2 = KEY / K-SVC (key TUs only, single-layer temporal on P-frames); 0 = independent layers",
        // Speed lever (svc-fast, 2026-11-12): the proxy is a search aid only
        // (the backend predicts from the base picture by AV1 §7.11.3.3), and
        // the Lanczos-3 upscale was ~10% of an av1-svc-fast encode's instructions.
        // Speed lever (svc-fast, 2026-11-12): the Lanczos-3 base downscale was
        // ~14% of an av1-svc-fast encode's instructions; the 2x2 mean is the
        // analysis pyramid's SIMD kernel.
        base_box_downscale: 0.0, [0.0, 1.0],
            "1 makes the base layer's source with the 2x2 mean instead of Lanczos-3 (exact 2:1 base only): much cheaper, softer and more aliased base pictures",
        proxy_bilinear: 0.0, [0.0, 1.0],
            "1 builds the enhancement layer's search proxy of the upscaled base picture with a [1 3]/4 bilinear 2:1 upscale instead of Lanczos-3 (exact 2:1 base only): cheaper, slightly blurrier proxy for motion and mode search",
        enhancement_interval: 1.0, [1.0, 30.0],
            "code the enhancement spatial layer in every Nth temporal unit only (1 = every unit; 6 with a 30 fps base = a 5 fps enhancement): higher concentrates enhancement bits into fewer full-size frames",
        // Speed / quality heuristic of VP9 spatial SVC (vp9-svc, 2026-11-12;
        // on by default for VP9 only, through its DEFAULT_PARAMS): the
        // enhancement layer's inter frames predict from the upscaled base
        // picture, which already carries the low frequencies an intra block
        // would add, so the DC intra test is skipped and intra is gated by
        // at least the engine's default variance. Not separately screened:
        // VP9 SVC's measured results (docs/handoff/spatial-svc.md §7.5)
        // include it.
        enhancement_intra_gate: 0.0, [0.0, 1.0],
            "1 skips the DC intra test on enhancement-layer inter frames and gates their intra by at least the engine default variance: faster, fewer intra blocks where the upscaled base predicts well",
    }
}

param_group! {
    /// In-frame rate control (`inframe.rs`): coarsening the rest of a frame
    /// whose size, projected after each superblock row, passes its limit.
    /// Off by default (both limits 0).
    pub struct InFrameParams ("inframe") {
        room: 0.0, [0.0, 1.0],
            "in-frame rate control: steer each frame to end within this share of the room a re-code would enforce (`rate.recode_overflow`; 0 = no hard limit): higher re-codes more, lower coarsens more frames mid-way",
        budgets: 0.0, [0.0, 64.0],
            "in-frame rate control: soft limit of an inter frame in uncorrected frame budgets (target / frame rate; 0 = none): lower bounds static refinement bursts harder",
        // AV1 Default, the audit's six still points, 600 f, room 0.9 on
        // keys too: meva 97 kbps worst-1% PSNR-Y 29.14 -> 26.96 dB, chirla
        // 47 kbps 28.81 -> 27.31 (a key coarsened mid-way leaves its lower
        // rows coarse for the whole GOP; a re-code coarsens it evenly).
        keys: 0.0, [0.0, 1.0],
            "in-frame rate control: 1 steers key frames by the re-code room too; 0 leaves keys to the whole-frame re-code",
        max_offset: 12.0, [1.0, 24.0],
            "in-frame rate control: largest coarsening of a row, in QP units",
        row_step: 3.0, [0.5, 12.0],
            "in-frame rate control: largest offset change between consecutive superblock rows, in QP units: lower is smoother across the frame, higher reacts faster",
        exponent: 2.0, [0.5, 4.0],
            "in-frame rate control: slope of bits against step assumed for the rows still to code: higher coarsens less per projected excess",
    }
}

param_group! {
    /// Test-only modes (`random_decisions.rs`); never set in real encodes.
    pub struct DebugParams ("debug") {
        random_decisions: 0.0, [0.0, 16_777_215.0],
            "nonzero: seed for random legal partition/mode/vector/block-QP choices (syntax-coverage stress test; compression is meaningless); 0 = normal decisions",
        recode_identity: 0.0, [0.0, 1.0],
            "1 codes every frame of a backend that supports re-coding twice, first at the other end of the quantizer range, keeping the second (the recode contract's test: the stream must be byte-identical to 0)",
    }
}

/// Every engine parameter group's specs.
#[must_use]
pub fn engine_specs() -> Vec<&'static [ParamSpec]> {
    vec![
        DecisionParams::SPECS,
        PlannerParams::SPECS,
        AnalysisParams::SPECS,
        StillParams::SPECS,
        RateParams::SPECS,
        InFrameParams::SPECS,
        PersistParams::SPECS,
        LayerParams::SPECS,
        SceneParams::SPECS,
        HoldParams::SPECS,
        RefreshParams::SPECS,
        LtrParams::SPECS,
        RefParams::SPECS,
        QuantParams::SPECS,
        ChromaParams::SPECS,
        ResolutionParams::SPECS,
        SpatialParams::SPECS,
        DebugParams::SPECS,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_in_range_and_names_unique() {
        let specs: Vec<&ParamSpec> = engine_specs().into_iter().flatten().collect();
        for s in &specs {
            assert!((s.min..=s.max).contains(&s.default), "{}", s.name);
        }
        let mut names: Vec<&str> = specs.iter().map(|s| s.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), specs.len());
    }

    #[test]
    fn overrides_apply_and_validate() {
        let mut o = ParamOverrides::default();
        o.set("planner.lambda_scale", 3.0);
        assert_eq!(PlannerParams::from_overrides(&o).lambda_scale, 3.0);
        assert!(o.validate(&engine_specs()).is_ok());
        o.set("planner.lambda_scale", 100.0);
        assert!(matches!(
            o.validate(&engine_specs()),
            Err(ParamError::OutOfRange { .. })
        ));
        let mut u = ParamOverrides::default();
        u.set("planner.nope", 1.0);
        assert!(matches!(
            u.validate(&engine_specs()),
            Err(ParamError::Unknown(_))
        ));
    }

    #[test]
    fn prefix_split_and_overlay() {
        let mut o = ParamOverrides::default();
        o.set("planner.lambda_scale", 3.0);
        o.set("nonbase.planner.lambda_scale", 4.0);
        o.set("nonbase.decision.lambda_scale", 1.5);
        let (nonbase, rest) = o.split_prefix("nonbase.");
        assert_eq!(rest.names().collect::<Vec<_>>(), ["planner.lambda_scale"]);
        assert_eq!(
            nonbase.names().collect::<Vec<_>>(),
            ["decision.lambda_scale", "planner.lambda_scale"]
        );
        let overlay = rest.overlaid_with(&nonbase);
        assert_eq!(PlannerParams::from_overrides(&overlay).lambda_scale, 4.0);
        assert_eq!(DecisionParams::from_overrides(&overlay).lambda_scale, 1.5);
        assert_eq!(PlannerParams::from_overrides(&rest).lambda_scale, 3.0);
    }
}
