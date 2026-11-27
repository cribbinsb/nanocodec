//! Constant QP (`rate.constant_qp`): a fixed-QP stream with every picture and
//! block at the configured quantizer, the way reference encoders are pinned
//! for a quantizer-point comparison (libaom and libvpx `--min-q=Q --max-q=Q
//! --aq-mode=0`).
//!
//! Our fixed-QP streams otherwise vary the quantizer on purpose: the QP
//! cadence (`qp_cadence.rs`), the temporal-layer ladder, a finer key frame
//! (`key_qp.rs`, `moving_camera.rs`, `key_interval.rs`) and the block-offset
//! terms (`aq.rs`: variance, persistence, reuse, layer propagation). The
//! switch turns each off, over the caller's and the codec's settings, so the
//! one parameter is enough whatever the preset sets. It is a comparison
//! setting, not a tuning: it codes worse than the default pattern.

use crate::config::EncoderConfig;
use crate::params::RateParams;

/// The engine parameters that vary the quantizer at fixed QP, at their
/// values that do not: the frame cadence at fixed QP, the layer ladder's step
/// ratios (1 codes every layer at the top layer's, the configured, step), the
/// four block-offset terms, and the key-frame terms the backends' own key
/// offset does not already cover.
pub(crate) const FLAT_SETTINGS: &[(&str, f32)] = &[
    ("rate.cadence_fixed_qp", 0.0),
    ("layers.base_step_ratio", 1.0),
    ("layers.middle_step_ratio", 1.0),
    ("rate.aq_strength", 0.0),
    ("persist.strength", 0.0),
    ("persist.reuse", 0.0),
    ("layers.propagation", 0.0),
    ("rate.fixed_qp_moving_key", 0.0),
    ("rate.fixed_qp_key_ln_interval", 0.0),
];

/// `config` made constant-QP when `rate.constant_qp` is set: the
/// [`FLAT_SETTINGS`] over its parameters and a key-frame offset of 0
/// (`EncoderConfig::key_qp_offset`, which replaces every backend's own key
/// offset). Unchanged otherwise.
pub(crate) fn apply(config: &mut EncoderConfig) {
    if RateParams::from_overrides(&config.params).constant_qp < 0.5 {
        return;
    }
    for &(name, value) in FLAT_SETTINGS {
        config.params.set(name, value);
    }
    config.key_qp_offset = Some(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Codec, FrameRate, StreamSignal};
    use crate::params::{LayerParams, ParamOverrides, PersistParams};

    fn config(params: &[(&str, f32)]) -> EncoderConfig {
        let mut overrides = ParamOverrides::default();
        for &(name, value) in params {
            overrides.set(name, value);
        }
        EncoderConfig {
            codec: Codec::H264,
            width: 64,
            height: 64,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 30,
            intra_period: 30,
            temporal_layers: 1,
            block_log2: 4,
            key_qp_offset: None,
            signal: StreamSignal::default(),
            params: overrides,
        }
    }

    #[test]
    fn switch_flattens_every_quantizer_term() {
        let mut config = config(&[("rate.aq_strength", 1.0), ("rate.constant_qp", 1.0)]);
        apply(&mut config);
        let rate = RateParams::from_overrides(&config.params);
        let persist = PersistParams::from_overrides(&config.params);
        let layers = LayerParams::from_overrides(&config.params);
        assert_eq!(config.key_qp_offset, Some(0));
        assert_eq!(rate.cadence_fixed_qp, 0.0);
        assert_eq!(rate.aq_strength, 0.0);
        assert_eq!((persist.strength, persist.reuse), (0.0, 0.0));
        assert_eq!(
            (layers.base_step_ratio, layers.middle_step_ratio),
            (1.0, 1.0)
        );
        assert_eq!(layers.propagation, 0.0);
    }

    #[test]
    fn off_by_default() {
        let mut config = config(&[("rate.aq_strength", 1.0)]);
        let before = config.clone();
        apply(&mut config);
        assert_eq!(config, before);
    }
}
