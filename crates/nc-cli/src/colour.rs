//! `--colour` and `--range`: colour signalling options.
//!
//! `--colour bt709|bt601|bt601-625` picks a preset (BT.601 means the
//! 525-line primaries common to webcams; `bt601-625` the PAL ones), or
//! `P,T,M` gives the three ISO/IEC 23091-2 code points directly.
//! `--range full|limited` sets the sample range. With neither option
//! nothing is signalled and streams are unchanged.

use nc_encoder::{ColourDescription, ColourInfo};

/// Colour signalling from the command line (`get` looks up a flag's value).
pub fn parse(get: impl Fn(&str) -> Option<String>) -> Result<ColourInfo, String> {
    let description = get("--colour").map(|v| description(&v)).transpose()?;
    let full_range = match get("--range").as_deref() {
        None | Some("limited") => false,
        Some("full") => true,
        Some(other) => return Err(format!("bad --range {other} (full, limited)")),
    };
    Ok(ColourInfo {
        description,
        full_range,
    })
}

fn description(text: &str) -> Result<ColourDescription, String> {
    match text {
        "bt709" => return Ok(ColourDescription::BT709),
        "bt601" => return Ok(ColourDescription::BT601_525),
        "bt601-625" => return Ok(ColourDescription::BT601_625),
        _ => {}
    }
    let bad = || format!("bad --colour {text} (bt709, bt601, bt601-625 or P,T,M)");
    let codes: Vec<u8> = text
        .split(',')
        .map(|v| v.parse().map_err(|_| bad()))
        .collect::<Result<_, _>>()?;
    match codes[..] {
        [primaries, transfer, matrix] => Ok(ColourDescription {
            primaries,
            transfer,
            matrix,
        }),
        _ => Err(bad()),
    }
}

#[cfg(test)]
mod tests {
    use super::parse;
    use nc_encoder::{ColourDescription, ColourInfo};

    fn run(args: &[(&str, &str)]) -> Result<ColourInfo, String> {
        parse(|flag| {
            args.iter()
                .find(|(f, _)| *f == flag)
                .map(|(_, v)| (*v).to_string())
        })
    }

    #[test]
    fn presets_code_points_and_range() {
        assert_eq!(run(&[]), Ok(ColourInfo::default()));
        let colour = run(&[("--colour", "bt709"), ("--range", "full")]).unwrap();
        assert_eq!(colour.description, Some(ColourDescription::BT709));
        assert!(colour.full_range);
        let custom = run(&[("--colour", "9,16,9")]).unwrap().description.unwrap();
        assert_eq!(
            (custom.primaries, custom.transfer, custom.matrix),
            (9, 16, 9)
        );
        assert!(run(&[("--colour", "1,2")]).is_err());
        assert!(run(&[("--range", "tv")]).is_err());
    }
}
