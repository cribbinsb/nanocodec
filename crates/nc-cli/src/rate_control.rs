//! CLI bitrate/VBV options; omitted bitrate preserves the fixed-QP path.
//! `--rate-change FRAME:KBPS[:PEAK_KBPS[:BUFFER_MS]]` (repeatable) changes
//! the target, peak and buffer before coding frame FRAME, as a transport
//! reacting to a bandwidth estimate would (`Encoder::set_rate_control`).
use nc_encoder::RateControlConfig;

/// A mid-stream rate-control change (`--rate-change`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RateChange {
    /// The frame (input index) the change applies before.
    pub frame: u32,
    /// The configuration from that frame on.
    pub config: RateControlConfig,
}

const RATE_FLAGS: &[&str] = &[
    "--buffer-ms",
    "--peak-bitrate",
    "--min-qp",
    "--max-qp",
    "--max-qp-step",
    "--rate-log",
    "--frame-drop",
    "--rate-change",
];

fn value<'a>(args: &'a [String], flag: &str) -> Result<Option<&'a str>, String> {
    let Some(index) = args.iter().position(|arg| arg == flag) else {
        return Ok(None);
    };
    args.get(index + 1)
        .map(String::as_str)
        .filter(|value| !value.starts_with("--"))
        .map(Some)
        .ok_or_else(|| format!("missing value after {flag}"))
}

fn bitrate(value: &str, flag: &str) -> Result<u64, String> {
    let kbps: f64 = value.parse().map_err(|_| format!("bad {flag}: {value}"))?;
    let bps = kbps * 1000.0;
    if !bps.is_finite() || !(1.0..=1_000_000_000.0).contains(&bps) {
        return Err(format!("{flag} must be positive and at most 1000000 kbps"));
    }
    Ok(bps.round() as u64)
}

fn number(args: &[String], flag: &str) -> Result<Option<u32>, String> {
    value(args, flag)?
        .map(|value| value.parse().map_err(|_| format!("bad {flag}: {value}")))
        .transpose()
}

pub(super) fn parse(args: &[String]) -> Result<Option<RateControlConfig>, String> {
    let Some(target) = value(args, "--bitrate")? else {
        if let Some(flag) = RATE_FLAGS
            .iter()
            .find(|flag| args.iter().any(|arg| arg == **flag))
        {
            return Err(format!("{flag} requires --bitrate"));
        }
        return Ok(None);
    };
    let _ = value(args, "--rate-log")?;
    let mut config = RateControlConfig::new(bitrate(target, "--bitrate")?);
    if let Some(buffer_ms) = number(args, "--buffer-ms")? {
        config.buffer_ms = buffer_ms;
    }
    config.peak_bps = value(args, "--peak-bitrate")?
        .map(|value| bitrate(value, "--peak-bitrate"))
        .transpose()?;
    config.min_quantizer = number(args, "--min-qp")?;
    config.max_quantizer = number(args, "--max-qp")?;
    config.max_step = number(args, "--max-qp-step")?;
    config.frame_drop = args.iter().any(|arg| arg == "--frame-drop");
    // Only the rate log reports the temporal noise statistic.
    config.temporal_noise_stat = args.iter().any(|arg| arg == "--rate-log");
    Ok(Some(config))
}

/// Every `--rate-change`, in frame order, each applied on top of the
/// configuration before it (starting from `initial`).
pub(super) fn changes(
    args: &[String],
    initial: Option<RateControlConfig>,
) -> Result<Vec<RateChange>, String> {
    let mut specs: Vec<&str> = args
        .windows(2)
        .filter(|pair| pair[0] == "--rate-change")
        .map(|pair| pair[1].as_str())
        .collect();
    if specs.is_empty() {
        return Ok(Vec::new());
    }
    let mut config = initial.ok_or("--rate-change requires --bitrate")?;
    let bad = |spec: &str| format!("bad --rate-change {spec} (FRAME:KBPS[:PEAK_KBPS[:BUFFER_MS]])");
    let frame_of = |spec: &str| spec.split(':').next().and_then(|f| f.parse::<u32>().ok());
    specs.sort_by_key(|spec| frame_of(spec));
    let mut out = Vec::with_capacity(specs.len());
    for spec in specs {
        let mut fields = spec.split(':');
        let frame = fields
            .next()
            .and_then(|f| f.parse().ok())
            .ok_or_else(|| bad(spec))?;
        let target = fields.next().ok_or_else(|| bad(spec))?;
        config.target_bps = bitrate(target, "--rate-change")?;
        if let Some(peak) = fields.next() {
            config.peak_bps = Some(bitrate(peak, "--rate-change")?);
        }
        if let Some(buffer) = fields.next() {
            config.buffer_ms = buffer.parse().map_err(|_| bad(spec))?;
        }
        if fields.next().is_some() {
            return Err(bad(spec));
        }
        out.push(RateChange { frame, config });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).into()).collect()
    }
    #[test]
    fn fixed_quantizer_remains_unconfigured_and_rate_options_are_explicit() {
        assert!(parse(&args(&["--qp", "32"])).unwrap().is_none());
        for input in [
            vec!["--buffer-ms", "500"],
            vec!["--rate-log", "x.csv"],
            vec!["--bitrate"],
            vec!["--bitrate", "NaN"],
            vec!["--bitrate", "0"],
            vec!["--bitrate", "-1"],
            vec!["--bitrate", "200", "--max-qp", "no"],
        ] {
            assert!(parse(&args(&input)).is_err());
        }
        let config = parse(&args(&[
            "--bitrate",
            "123.5",
            "--peak-bitrate",
            "250",
            "--buffer-ms",
            "750",
            "--min-qp",
            "12",
            "--max-qp",
            "48",
            "--max-qp-step",
            "3",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(config.target_bps, 123_500);
        assert_eq!(config.peak_bps, Some(250_000));
        assert_eq!(config.buffer_ms, 750);
        assert_eq!(config.min_quantizer, Some(12));
        assert_eq!(config.max_quantizer, Some(48));
        assert_eq!(config.max_step, Some(3));
    }

    #[test]
    fn rate_changes_accumulate_in_frame_order() {
        let a = args(&[
            "--bitrate",
            "500",
            "--rate-change",
            "90:250:500:300",
            "--rate-change",
            "30:1000",
        ]);
        let initial = parse(&a).unwrap();
        let parsed = changes(&a, initial).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(
            (parsed[0].frame, parsed[0].config.target_bps),
            (30, 1_000_000)
        );
        assert_eq!(parsed[0].config.peak_bps, None);
        let later = parsed[1].config;
        assert_eq!(
            (later.target_bps, later.peak_bps, later.buffer_ms),
            (250_000, Some(500_000), 300)
        );
        assert!(changes(&args(&["--qp", "30", "--rate-change", "5:100"]), None).is_err());
        assert!(
            changes(
                &args(&["--bitrate", "5", "--rate-change", "x:100"]),
                initial
            )
            .is_err()
        );
    }
}
