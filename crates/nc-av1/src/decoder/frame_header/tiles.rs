//! `tile_info()` (AV1 §5.9.15): the tile grid in 4x4 (mode-info) units.
use super::FrameSize;
use crate::backend::decoder::availability::TileBounds;
use crate::backend::decoder::sequence::Sequence;
use nc_common::decode::{BitReader, DecodeError};

/// `MAX_TILE_WIDTH`, `MAX_TILE_AREA`, `MAX_TILE_COLS`, `MAX_TILE_ROWS`
/// (AV1 §3).
const MAX_TILE_WIDTH: usize = 4096;
const MAX_TILE_AREA: usize = 4096 * 2304;
pub(crate) const MAX_TILE_COLS: usize = 64;
pub(crate) const MAX_TILE_ROWS: usize = 64;

/// The tile grid of a frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TileInfo {
    /// `MiColStarts[0..=TileCols]`, `MiRowStarts[0..=TileRows]`.
    pub col_starts: Vec<usize>,
    pub row_starts: Vec<usize>,
    pub cols_log2: u32,
    pub rows_log2: u32,
    /// `context_update_tile_id`: the tile whose final CDFs the frame keeps.
    pub context_update_tile_id: usize,
    /// `TileSizeBytes`: bytes of each coded `tile_size_minus_1`.
    pub tile_size_bytes: usize,
}
impl Default for TileInfo {
    fn default() -> Self {
        Self {
            col_starts: vec![0, 0],
            row_starts: vec![0, 0],
            cols_log2: 0,
            rows_log2: 0,
            context_update_tile_id: 0,
            tile_size_bytes: 4,
        }
    }
}
impl TileInfo {
    pub(crate) fn cols(&self) -> usize {
        self.col_starts.len() - 1
    }
    pub(crate) fn rows(&self) -> usize {
        self.row_starts.len() - 1
    }
    pub(crate) fn count(&self) -> usize {
        self.cols() * self.rows()
    }
    /// The bounds of tile `index` (`TileNum`, raster order).
    pub(crate) fn bounds(&self, index: usize) -> TileBounds {
        let (row, col) = (index / self.cols(), index % self.cols());
        TileBounds {
            col_start: self.col_starts[col],
            col_end: self.col_starts[col + 1],
            row_start: self.row_starts[row],
            row_end: self.row_starts[row + 1],
        }
    }
}

/// `tile_log2(blkSize, target)`: the smallest `k` with
/// `blkSize << k >= target`.
fn tile_log2(block: usize, target: usize) -> u32 {
    let mut k = 0;
    while (block << k) < target {
        k += 1;
    }
    k
}

/// `ns(n)` (AV1 §4.10.10): a value below `n` in non-symmetric unsigned
/// coding.
pub(crate) fn read_ns(r: &mut BitReader<'_>, n: u32) -> Result<u32, DecodeError> {
    let width = 32 - n.leading_zeros();
    let short_values = (1 << width) - n;
    let value = r.read_bits(width - 1)?;
    if value < short_values {
        return Ok(value);
    }
    Ok((value << 1) - short_values + u32::from(r.read_bit()?))
}

/// Read `increment_*_log2` bits from `min` up to `max`.
fn increments(r: &mut BitReader<'_>, min: u32, max: u32) -> Result<u32, DecodeError> {
    let mut log2 = min;
    while log2 < max && r.read_bit()? {
        log2 += 1;
    }
    Ok(log2)
}

/// Starts of tiles `size_sb` superblocks apart, then `end`.
fn uniform_starts(sb_count: usize, size_sb: usize, sb_shift: u32, end: usize) -> Vec<usize> {
    let mut starts: Vec<usize> = (0..sb_count)
        .step_by(size_sb.max(1))
        .map(|sb| sb << sb_shift)
        .collect();
    starts.push(end);
    starts
}

pub(crate) fn parse(
    r: &mut BitReader<'_>,
    seq: &Sequence,
    size: &FrameSize,
) -> Result<TileInfo, DecodeError> {
    let (mi_cols, mi_rows) = (size.mi_cols(), size.mi_rows());
    let sb_shift: u32 = if seq.use_128x128_superblock { 5 } else { 4 };
    let sb_cols = (mi_cols + (1 << sb_shift) - 1) >> sb_shift;
    let sb_rows = (mi_rows + (1 << sb_shift) - 1) >> sb_shift;
    let sb_size_log2 = sb_shift + 2;
    let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size_log2;
    let mut max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size_log2);
    let min_log2_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles = min_log2_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));
    let mut info = TileInfo::default();
    if r.read_bit()? {
        // uniform_tile_spacing_flag
        info.cols_log2 = increments(r, min_log2_cols, max_log2_cols)?;
        let width_sb = (sb_cols + (1 << info.cols_log2) - 1) >> info.cols_log2;
        info.col_starts = uniform_starts(sb_cols, width_sb, sb_shift, mi_cols);
        let min_log2_rows = min_log2_tiles.saturating_sub(info.cols_log2);
        info.rows_log2 = increments(r, min_log2_rows, max_log2_rows)?;
        let height_sb = (sb_rows + (1 << info.rows_log2) - 1) >> info.rows_log2;
        info.row_starts = uniform_starts(sb_rows, height_sb, sb_shift, mi_rows);
    } else {
        let mut widest_sb = 0;
        info.col_starts = explicit_starts(r, sb_cols, max_tile_width_sb, sb_shift, mi_cols, |w| {
            widest_sb = widest_sb.max(w);
        })?;
        info.cols_log2 = tile_log2(1, info.cols());
        max_tile_area_sb = if min_log2_tiles > 0 {
            (sb_rows * sb_cols) >> (min_log2_tiles + 1)
        } else {
            sb_rows * sb_cols
        };
        let max_tile_height_sb = (max_tile_area_sb / widest_sb.max(1)).max(1);
        info.row_starts =
            explicit_starts(r, sb_rows, max_tile_height_sb, sb_shift, mi_rows, |_| {})?;
        info.rows_log2 = tile_log2(1, info.rows());
    }
    if info.cols() > MAX_TILE_COLS || info.rows() > MAX_TILE_ROWS {
        return Err(DecodeError::Malformed("AV1 tile count"));
    }
    if info.cols_log2 > 0 || info.rows_log2 > 0 {
        info.context_update_tile_id = r.read_bits(info.cols_log2 + info.rows_log2)? as usize;
        if info.context_update_tile_id >= info.count() {
            return Err(DecodeError::Malformed("AV1 context_update_tile_id"));
        }
        info.tile_size_bytes = r.read_bits(2)? as usize + 1;
    }
    Ok(info)
}

/// Explicitly sized tiles (`width_in_sbs_minus_1` / `height_in_sbs_minus_1`):
/// their starts, then `end`; `each` sees every tile's size.
fn explicit_starts(
    r: &mut BitReader<'_>,
    sb_count: usize,
    max_size_sb: usize,
    sb_shift: u32,
    end: usize,
    mut each: impl FnMut(usize),
) -> Result<Vec<usize>, DecodeError> {
    let mut starts = Vec::new();
    let mut start_sb = 0;
    while start_sb < sb_count {
        if starts.len() >= MAX_TILE_COLS.max(MAX_TILE_ROWS) {
            return Err(DecodeError::Malformed("AV1 tile count"));
        }
        starts.push(start_sb << sb_shift);
        let max = (sb_count - start_sb).min(max_size_sb) as u32;
        let size_sb = read_ns(r, max)? as usize + 1;
        each(size_sb);
        start_sb += size_sb;
    }
    starts.push(end);
    Ok(starts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ns(5)`: values 0..2 take 2 bits, 3 and 4 take 3 (§4.10.10).
    #[test]
    fn non_symmetric_values_round_trip() {
        let cases: [(&[u8], u32); 4] = [
            (&[0b0000_0000], 0),
            (&[0b1000_0000], 2),
            (&[0b1100_0000], 3),
            (&[0b1110_0000], 4),
        ];
        for (data, expected) in cases {
            let mut r = BitReader::new(data);
            assert_eq!(read_ns(&mut r, 5).unwrap(), expected);
        }
    }

    #[test]
    fn tile_log2_is_the_smallest_covering_shift() {
        assert_eq!(tile_log2(1, 1), 0);
        assert_eq!(tile_log2(1, 30), 5);
        assert_eq!(tile_log2(64, 30), 0);
        assert_eq!(uniform_starts(30, 8, 4, 480), vec![0, 128, 256, 384, 480]);
    }
}
