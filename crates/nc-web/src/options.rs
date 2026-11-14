//! Encoder options for the demo's Advanced section: what each codec
//! supports, its registered heuristic parameters (name, range, the codec's
//! default, description) and free-form `name=value` overrides checked
//! against the engine's registry, exactly as encoder construction checks
//! them. The page reads all of this from here, so it stays in sync with
//! the engine.

use std::fmt::Write as _;

use nc_av1::Av1Backend;
use nc_av2::Av2Backend;
use nc_encoder::{
    Codec, CodecBackend, ConfigError, Encoder, EncoderConfig, FrameRate, ParamOverrides, ParamSpec,
    StreamSignal, engine_specs, validate_params,
};
use nc_h264::H264Backend;
use nc_hevc::HevcBackend;
use nc_vp9::Vp9Backend;
use nc_vvc::VvcBackend;

/// The codec named `name` ("h264", "hevc", "av1", "vp9", "av2", "vvc").
pub(crate) fn parse_codec(name: &str) -> Option<Codec> {
    match name {
        "h264" => Some(Codec::H264),
        "hevc" => Some(Codec::Hevc),
        "av1" => Some(Codec::Av1),
        "vp9" => Some(Codec::Vp9),
        "av2" => Some(Codec::Av2),
        "vvc" => Some(Codec::Vvc),
        _ => None,
    }
}

/// The curated options the page's Advanced section may offer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CuratedOption {
    /// Film grain synthesis (`nc_av1::enable_film_grain`).
    FilmGrain,
    /// Persistence AQ (`persist.*`), which acts only through block
    /// quantizer offsets (`CodecBackend::SUPPORTS_BLOCK_QP`).
    BlockQp,
    /// Temporal layers (`layers.count` > 1).
    TemporalLayers,
    /// The second reference (`refs.second`).
    SecondReference,
}

impl CuratedOption {
    const ALL: [Self; 4] = [
        Self::FilmGrain,
        Self::BlockQp,
        Self::TemporalLayers,
        Self::SecondReference,
    ];

    /// The option's key in the page's JSON.
    fn key(self) -> &'static str {
        match self {
            Self::FilmGrain => "filmGrain",
            Self::BlockQp => "blockQp",
            Self::TemporalLayers => "temporalLayers",
            Self::SecondReference => "secondReference",
        }
    }
}

/// One codec's parameter registry.
struct Backend {
    specs: Vec<&'static [ParamSpec]>,
    defaults: &'static [(&'static str, f32)],
    block_qp: bool,
    validate: fn(&ParamOverrides) -> Result<(), ConfigError>,
}

fn backend_of<B: CodecBackend>() -> Backend {
    Backend {
        specs: [engine_specs(), B::PARAM_SPECS.to_vec()].concat(),
        defaults: B::DEFAULT_PARAMS,
        block_qp: B::SUPPORTS_BLOCK_QP,
        validate: validate_params::<B>,
    }
}

fn backend(codec: Codec) -> Backend {
    match codec {
        Codec::H264 => backend_of::<H264Backend>(),
        Codec::Hevc => backend_of::<HevcBackend>(),
        Codec::Av1 => backend_of::<Av1Backend>(),
        Codec::Vp9 => backend_of::<Vp9Backend>(),
        Codec::Av2 => backend_of::<Av2Backend>(),
        Codec::Vvc => backend_of::<VvcBackend>(),
    }
}

/// Whether `codec` offers `option`. Layers and the second reference are
/// backend capabilities the engine checks only when it builds an encoder
/// (`supports_temporal_layers`, `supports_second_reference`), so this
/// builds a small one with the option on: the answer cannot drift from
/// what encoding accepts.
pub(crate) fn offers(codec: Codec, option: CuratedOption) -> bool {
    match option {
        CuratedOption::FilmGrain => {
            nc_av1::enable_film_grain(codec, None, &mut ParamOverrides::default()).is_ok()
        }
        CuratedOption::BlockQp => backend(codec).block_qp,
        CuratedOption::TemporalLayers => accepts(codec, LAYER_COUNT, 2.0),
        CuratedOption::SecondReference => accepts(codec, SECOND_REFERENCE, 1.0),
    }
}

/// The parameters behind the layer and second-reference options.
const LAYER_COUNT: &str = "layers.count";
const SECOND_REFERENCE: &str = "refs.second";
/// The probe encoder's size: the smallest the page allows.
const PROBE_SIZE: usize = 64;
/// Any valid fixed quantizer for the probe (every codec's range holds it).
const PROBE_QUANTIZER: u32 = 30;

/// Whether the engine builds a `codec` encoder with `name = value`.
fn accepts(codec: Codec, name: &str, value: f32) -> bool {
    let mut params = ParamOverrides::default();
    params.set(name, value);
    let config = EncoderConfig {
        codec,
        width: PROBE_SIZE,
        height: PROBE_SIZE,
        frame_rate: FrameRate { num: 30, den: 1 },
        quantizer: PROBE_QUANTIZER,
        intra_period: 30,
        block_log2: crate::block_log2(codec),
        key_qp_offset: None,
        temporal_layers: 1,
        signal: StreamSignal::default(),
        params,
    };
    match codec {
        Codec::H264 => Encoder::<H264Backend>::new(config).is_ok(),
        Codec::Hevc => Encoder::<HevcBackend>::new(config).is_ok(),
        Codec::Av1 => Encoder::<Av1Backend>::new(config).is_ok(),
        Codec::Vp9 => Encoder::<Vp9Backend>::new(config).is_ok(),
        Codec::Av2 => Encoder::<Av2Backend>::new(config).is_ok(),
        Codec::Vvc => Encoder::<VvcBackend>::new(config).is_ok(),
    }
}

/// A bad line of a free-form override list.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LineError {
    /// 1-based line number.
    pub line: usize,
    /// Why it was rejected.
    pub message: String,
}

/// `text` as overrides for `codec`: one `name = value` per line, `#`
/// comments and blank lines ignored (the CLI's parameter-file format). Each
/// setting is checked against the engine's and the codec's registries.
///
/// # Errors
///
/// Every bad line, with its reason.
pub(crate) fn parse_overrides(codec: Codec, text: &str) -> Result<ParamOverrides, Vec<LineError>> {
    let backend = backend(codec);
    let mut overrides = ParamOverrides::default();
    let mut errors = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let setting = line.split('#').next().unwrap_or("").trim();
        if setting.is_empty() {
            continue;
        }
        match parse_setting(setting, backend.validate) {
            Ok((name, value)) => overrides.set(name, value),
            Err(message) => errors.push(LineError {
                line: index + 1,
                message,
            }),
        }
    }
    if errors.is_empty() {
        Ok(overrides)
    } else {
        Err(errors)
    }
}

/// One `name = value` setting, checked on its own so that every bad line
/// gets its own message.
fn parse_setting(
    setting: &str,
    validate: fn(&ParamOverrides) -> Result<(), ConfigError>,
) -> Result<(&str, f32), String> {
    let (name, value) = setting
        .split_once('=')
        .ok_or_else(|| format!("expected name=value, got '{setting}'"))?;
    let name = name.trim();
    let value: f32 = value
        .trim()
        .parse()
        .map_err(|_| format!("'{}' is not a number", value.trim()))?;
    let mut single = ParamOverrides::default();
    single.set(name, value);
    validate(&single).map_err(|e| e.to_string())?;
    Ok((name, value))
}

/// `codec`'s parameters as a JSON array of `{name, default, min, max,
/// doc}`; a default is the value the codec encodes with (its backend's
/// `DEFAULT_PARAMS` over the engine's), as `nanocodec params --codec`.
pub(crate) fn registry_json(codec: Codec) -> String {
    let backend = backend(codec);
    let entries: Vec<String> = backend
        .specs
        .iter()
        .flat_map(|group| group.iter())
        .map(|spec| {
            let default = backend
                .defaults
                .iter()
                .find(|(name, _)| *name == spec.name)
                .map_or(spec.default, |&(_, value)| value);
            format!(
                r#"{{"name":{},"default":{},"min":{},"max":{},"doc":{}}}"#,
                json_string(spec.name),
                default,
                spec.min,
                spec.max,
                json_string(spec.doc)
            )
        })
        .collect();
    format!("[{}]", entries.join(","))
}

/// What `codec` offers, as a JSON object of [`CuratedOption`] keys.
pub(crate) fn options_json(codec: Codec) -> String {
    let entries: Vec<String> = CuratedOption::ALL
        .iter()
        .map(|&option| format!(r#""{}":{}"#, option.key(), offers(codec, option)))
        .collect();
    format!("{{{}}}", entries.join(","))
}

/// `errors` as a JSON array of `{line, message}`.
pub(crate) fn errors_json(errors: &[LineError]) -> String {
    let entries: Vec<String> = errors
        .iter()
        .map(|e| {
            format!(
                r#"{{"line":{},"message":{}}}"#,
                e.line,
                json_string(&e.message)
            )
        })
        .collect();
    format!("[{}]", entries.join(","))
}

/// `text` as a JSON string literal (RFC 8259 §7).
fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_options_follow_the_codecs() {
        assert!(offers(Codec::Av1, CuratedOption::FilmGrain));
        assert!(!offers(Codec::H264, CuratedOption::FilmGrain));
        // AV1 and VP9 take the second reference on flat streams since 9fe4dd1b;
        // HEVC has no second-reference syntax in our backend.
        for codec in [Codec::H264, Codec::Av1, Codec::Vp9] {
            assert!(offers(codec, CuratedOption::SecondReference));
        }
        assert!(!offers(Codec::Hevc, CuratedOption::SecondReference));
        assert!(offers(Codec::H264, CuratedOption::TemporalLayers));
        assert!(offers(Codec::Av1, CuratedOption::BlockQp));
    }

    #[test]
    fn overrides_report_every_bad_line() {
        let text =
            "# comment\nrefs.second = 1\n\nnope.nothing=1\nlayers.count=9\nrate.aq_strength x\n";
        let errors = parse_overrides(Codec::H264, text).unwrap_err();
        assert_eq!(errors.iter().map(|e| e.line).collect::<Vec<_>>(), [4, 5, 6]);
        assert!(errors[0].message.contains("unknown parameter"));
        assert!(errors[1].message.contains("outside"));
        let good = parse_overrides(
            Codec::H264,
            "refs.second=1 # on\nnonbase.decision.codec_subpel_seed_lambdas=0.05",
        );
        assert_eq!(good.map(|o| o.names().count()), Ok(2));
    }

    #[test]
    fn the_json_has_codec_defaults_and_escapes_text() {
        let json = registry_json(Codec::Vp9);
        assert!(json.starts_with("[{\"name\":") && json.ends_with("}]"));
        assert!(json.contains("\"name\":\"decision.codec_subpel_seed_lambdas\",\"default\":0,"));
        assert_eq!(json_string("a\"b\\c\n"), r#""a\"b\\c\n""#);
        assert!(options_json(Codec::Av1).starts_with(r#"{"filmGrain":true,"blockQp":true,"#));
    }
}
