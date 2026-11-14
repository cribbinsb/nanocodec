//! `--qp-replay FILE`: a quantizer per input frame, imposed on rate
//! control's plan (`Encoder::force_next_quantizer`). An oracle for the
//! rate-control cost decomposition (`eval/nceval/rc_oracle.py`): a stream's
//! own logged quantizers, or a smoothed version of them, coded with every
//! other part of the encoder unchanged.
//!
//! One line per input frame: `QUANTIZER [LEVEL]` (the frame's quantizer as
//! `--rate-log` reports it in `qp`, and the inter level it stands for, its
//! `level_q`), or `-` to keep the controller's plan. Frames past the last
//! line keep the plan too.

use nc_encoder::ForcedQuantizer;

/// The replayed quantizers, by input frame.
pub(crate) type Replay = Vec<Option<ForcedQuantizer>>;

/// Parse a replay file's `text`.
pub(crate) fn parse(text: &str) -> Result<Replay, String> {
    text.lines()
        .enumerate()
        .map(|(line, row)| {
            let bad = || format!("--qp-replay line {}: {row:?}", line + 1);
            let mut fields = row.split_whitespace();
            match fields.next() {
                None | Some("-") => Ok(None),
                Some(quantizer) => {
                    let quantizer = quantizer.parse().map_err(|_| bad())?;
                    let level = fields
                        .next()
                        .map(|level| level.parse().map_err(|_| bad()))
                        .transpose()?;
                    if fields.next().is_some() {
                        return Err(bad());
                    }
                    Ok(Some(ForcedQuantizer { quantizer, level }))
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_quantizers_levels_and_kept_plans() {
        let replay = parse("40\n44 42\n-\n\n").unwrap();
        assert_eq!(
            replay,
            vec![
                Some(ForcedQuantizer {
                    quantizer: 40,
                    level: None
                }),
                Some(ForcedQuantizer {
                    quantizer: 44,
                    level: Some(42)
                }),
                None,
                None,
            ]
        );
        assert!(parse("40 41 42").is_err());
        assert!(parse("x").is_err());
    }
}
