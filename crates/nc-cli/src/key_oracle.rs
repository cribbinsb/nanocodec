//! The key-frame inheritance oracle's two diagnostics (`nc_encoder`
//! `key_oracle.rs`; `eval/keystudy/inherit.py` computes the maps):
//!
//! - `--analysis-log FILE`: after every input frame, the analysis the
//!   analyzer made of it, per 8x8 luma cell. Binary, little-endian, one
//!   record per frame: `u32` input frame, `u16` cell columns, `u16` cell
//!   rows, `u8` 1 when the frame has motion analysis, then per cell in
//!   raster order `u32` sum of squared deviations, `u32` motion SAD (0
//!   without motion), `i16` vector x, `i16` vector y (luma samples).
//! - `--key-offsets FILE`: QP offsets per 8x8 cell imposed on key frames
//!   (`Encoder::set_next_key_offsets`). Text: a line `frame N` (input
//!   frame), then one line per cell row of whitespace-separated offsets;
//!   repeat for further keys. Frames without a section get none.

use std::collections::HashMap;
use std::io::Write;

use nc_encoder::CellAnalysis;

/// Key offset maps by input frame, each in raster order.
pub(crate) type KeyOffsets = HashMap<u32, Vec<f64>>;

/// Parse a `--key-offsets` file's `text`.
pub(crate) fn parse(text: &str) -> Result<KeyOffsets, String> {
    let mut maps = KeyOffsets::new();
    let mut current: Option<u32> = None;
    for (line, row) in text.lines().enumerate() {
        let bad = || format!("--key-offsets line {}: {row:?}", line + 1);
        let row = row.trim();
        if row.is_empty() {
            continue;
        }
        if let Some(frame) = row.strip_prefix("frame ") {
            let frame = frame.trim().parse().map_err(|_| bad())?;
            maps.insert(frame, Vec::new());
            current = Some(frame);
            continue;
        }
        let map = current
            .and_then(|frame| maps.get_mut(&frame))
            .ok_or_else(bad)?;
        for value in row.split_whitespace() {
            map.push(value.parse().map_err(|_| bad())?);
        }
    }
    Ok(maps)
}

/// Append one `--analysis-log` record for input frame `index`.
pub(crate) fn write_record(
    out: &mut impl Write,
    index: u32,
    analysis: &CellAnalysis,
) -> std::io::Result<()> {
    let too_large = |_| std::io::Error::other("cell grid larger than 65535");
    let cols = u16::try_from(analysis.cols).map_err(too_large)?;
    let rows = u16::try_from(analysis.rows).map_err(too_large)?;
    let has_motion = analysis.cells.iter().any(|c| c.motion_sad.is_some());
    out.write_all(&index.to_le_bytes())?;
    out.write_all(&cols.to_le_bytes())?;
    out.write_all(&rows.to_le_bytes())?;
    out.write_all(&[u8::from(has_motion)])?;
    for cell in &analysis.cells {
        out.write_all(&cell.sum_sq_dev.to_le_bytes())?;
        out.write_all(&cell.motion_sad.unwrap_or(0).to_le_bytes())?;
        out.write_all(&cell.mv.0.to_le_bytes())?;
        out.write_all(&cell.mv.1.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_encoder::AnalysisCell;

    #[test]
    fn parses_maps_by_frame() {
        let maps = parse("frame 0\n1 -2\n0.5 0\n\nframe 300\n3\n").unwrap();
        assert_eq!(maps[&0], vec![1., -2., 0.5, 0.]);
        assert_eq!(maps[&300], vec![3.]);
        assert!(parse("1 2\n").is_err());
        assert!(parse("frame x\n").is_err());
    }

    #[test]
    fn writes_one_record_per_frame() {
        let analysis = CellAnalysis {
            cols: 1,
            rows: 1,
            cells: vec![AnalysisCell {
                sum_sq_dev: 7,
                motion_sad: Some(3),
                mv: (-1, 2),
            }],
        };
        let mut out = Vec::new();
        write_record(&mut out, 5, &analysis).unwrap();
        assert_eq!(
            out,
            [
                5, 0, 0, 0, 1, 0, 1, 0, 1, 7, 0, 0, 0, 3, 0, 0, 0, 255, 255, 2, 0
            ]
        );
    }
}
