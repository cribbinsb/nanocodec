//! Tile layout syntax (`tile_info`), shared by the sequence header and the
//! frame header (dav2d `obu.c` `parse_tile_info`, AVM `read_tile_info`).
use super::bits::uniform;
use nc_common::decode::{BitReader, Result};

/// Most tile columns or rows of a frame (AVM `MAX_TILE_COLS`/`MAX_TILE_ROWS`).
pub const MAX_TILES_PER_AXIS: usize = 64;

/// Widest tile in luma samples before the level adjustment (AVM
/// `MAX_TILE_WIDTH`).
const MAX_TILE_WIDTH: u32 = 4096;
/// Largest tile in luma samples before the level adjustment (AVM
/// `MAX_TILE_AREA`).
const MAX_TILE_AREA: u32 = 4096 * 2304;
/// Levels from which tiles may be larger (AVM `av2_get_max_tile_*`): 6.0
/// and 7.0.
const LARGE_TILE_LEVEL: u8 = 14;
const LARGER_TILE_LEVEL: u8 = 18;

/// What sizes a tile layout: the frame (or sequence maximum) size, the
/// superblock size the limits are counted in, and the level.
#[derive(Debug, Clone, Copy)]
pub struct TileGeometry {
    /// Frame width and height in luma samples.
    pub size: (u32, u32),
    /// log2 of the frame's superblock size (6, 7 or 8).
    pub superblock_log2: u8,
    /// log2 of the sequence's superblock size: uniform tiles are spaced in
    /// these even when a key frame codes smaller superblocks.
    pub sequence_superblock_log2: u8,
    /// `seq_level_idx` and `seq_tier` (the tile size limits depend on them).
    pub level: u8,
    /// `seq_tier`.
    pub high_tier: bool,
}

/// A frame's tile grid: starts of each column and row in superblocks, with
/// the frame's extent as the last entry.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TileLayout {
    /// `uniform_tile_spacing_flag`.
    pub uniform: bool,
    /// Column starts, then the frame width, in superblocks.
    pub column_starts: Vec<u16>,
    /// Row starts, then the frame height, in superblocks.
    pub row_starts: Vec<u16>,
}

impl TileLayout {
    /// Tile columns.
    #[must_use]
    pub fn columns(&self) -> usize {
        self.column_starts.len() - 1
    }

    /// Tile rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.row_starts.len() - 1
    }
}

/// The smallest `k` with `size << k >= target` (AVM `tile_log2`).
fn tile_log2(size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (size << k) < target {
        k += 1;
    }
    k
}

/// Starts of `1 << log2_count` uniform tiles (fewer when the frame is
/// short of superblocks) along an axis of `size` luma samples, in units of
/// `step` frame superblocks per sequence superblock (dav2d
/// `calc_uniform_tiles`). The spacing counts whole sequence superblocks
/// only; the remainder goes to the first tiles.
fn uniform_starts(log2_count: u32, step: u16, sequence_superblock_log2: u8, size: u32) -> Vec<u16> {
    let shift = u32::from(sequence_superblock_log2);
    // Whole superblocks (rounded to 8-sample units, as AVM counts mi units).
    let whole = ((size + 7) >> shift).max(1);
    let extended = 1 + ((size - 1) >> shift);
    let tile = whole >> log2_count;
    let mut extra = i64::from(whole) - i64::from(tile << log2_count);
    if tile == 0 {
        extra += i64::from(extended - whole);
    }
    let mut starts = Vec::new();
    let mut superblock = 0;
    while superblock < extended && starts.len() < 1 << log2_count {
        starts.push(superblock as u16 * step);
        superblock += tile + u32::from(extra > 0);
        extra -= 1;
    }
    starts
}

/// log2 of a tile count rounded up (bits of a tile index along an axis).
#[must_use]
pub fn log2(count: usize) -> u32 {
    tile_log2(1, count as u32)
}

/// Superblocks of `geometry` across and down.
fn superblocks(geometry: TileGeometry) -> (u32, u32) {
    let log2 = u32::from(geometry.superblock_log2);
    let count = |size: u32| (size + (1 << log2) - 1) >> log2;
    (count(geometry.size.0), count(geometry.size.1))
}

/// Whether a frame of `geometry` may reuse the sequence's tile `layout`:
/// a non-uniform one when the frame has the sequence's superblock grid, a
/// uniform one when its tile count still fits (dav2d
/// `parse_tile_info_frmhdr`).
#[must_use]
pub fn reusable(
    layout: &TileLayout,
    sequence: &super::sequence::SequenceHeader,
    geometry: TileGeometry,
) -> bool {
    let (columns, rows) = superblocks(geometry);
    if layout.uniform {
        let fits = |count: usize, superblocks: u32| {
            let log2 = log2(count);
            let tile = (superblocks + (1 << log2) - 1) >> log2;
            tile * ((1 << log2) - 1) < superblocks
        };
        fits(layout.columns(), columns) && fits(layout.rows(), rows)
    } else {
        let sequence_grid = superblocks(TileGeometry {
            size: (sequence.max_width, sequence.max_height),
            superblock_log2: geometry.sequence_superblock_log2,
            ..geometry
        });
        sequence_grid == (columns, rows)
    }
}

/// The sequence's tile `layout` for a frame of `geometry`: a uniform
/// layout keeps its tile count, spread over the frame's superblocks; a
/// non-uniform one keeps its starts, in frame superblocks.
#[must_use]
pub fn reuse(layout: &TileLayout, geometry: TileGeometry) -> TileLayout {
    let step = 1_u16 << (geometry.sequence_superblock_log2 - geometry.superblock_log2);
    let (columns, rows) = superblocks(geometry);
    let (mut column_starts, mut row_starts) = if layout.uniform {
        (
            uniform_starts(
                log2(layout.columns()),
                step,
                geometry.sequence_superblock_log2,
                geometry.size.0,
            ),
            uniform_starts(
                log2(layout.rows()),
                step,
                geometry.sequence_superblock_log2,
                geometry.size.1,
            ),
        )
    } else {
        let scaled = |starts: &[u16]| -> Vec<u16> {
            starts[..starts.len() - 1]
                .iter()
                .map(|&start| start * step)
                .collect()
        };
        (scaled(&layout.column_starts), scaled(&layout.row_starts))
    };
    // (A reused non-uniform layout's extent equals the frame's: see
    // `reusable`.)
    column_starts.push(columns as u16);
    row_starts.push(rows as u16);
    TileLayout {
        uniform: layout.uniform,
        column_starts,
        row_starts,
    }
}

/// Parse `tile_info` for a frame of `geometry`.
/// # Errors
/// Truncation.
pub fn parse(bits: &mut BitReader<'_>, geometry: TileGeometry) -> Result<TileLayout> {
    let uniform_spacing = bits.read_bit()?;
    let log2 = u32::from(geometry.superblock_log2);
    let superblocks = |size: u32| (size + (1 << log2) - 1) >> log2;
    let (columns, rows) = (superblocks(geometry.size.0), superblocks(geometry.size.1));
    let level = geometry.level;
    let width_adjust = u32::from(level >= LARGER_TILE_LEVEL)
        + u32::from(level >= LARGE_TILE_LEVEL && geometry.high_tier);
    let max_tile_width = MAX_TILE_WIDTH >> (log2 - width_adjust);
    let area_adjust = u32::from(level >= LARGE_TILE_LEVEL) + width_adjust;
    let max_tile_area = MAX_TILE_AREA >> (2 * log2 - area_adjust);
    let min_log2_columns = tile_log2(max_tile_width, columns);
    let max_log2_columns = tile_log2(1, columns.min(MAX_TILES_PER_AXIS as u32));
    let max_log2_rows = tile_log2(1, rows.min(MAX_TILES_PER_AXIS as u32));
    let min_log2_tiles = tile_log2(max_tile_area, columns * rows).max(min_log2_columns);
    // Frame superblocks per sequence superblock (key frames may code
    // smaller superblocks than the sequence's).
    let step = 1_u16 << (geometry.sequence_superblock_log2 - geometry.superblock_log2);

    let (mut column_starts, mut row_starts);
    if uniform_spacing {
        let mut log2_columns = min_log2_columns;
        while log2_columns < max_log2_columns && bits.read_bit()? {
            log2_columns += 1;
        }
        column_starts = uniform_starts(
            log2_columns,
            step,
            geometry.sequence_superblock_log2,
            geometry.size.0,
        );
        let log2_columns = tile_log2(1, column_starts.len() as u32);
        let mut log2_rows = min_log2_tiles.saturating_sub(log2_columns);
        while log2_rows < max_log2_rows && bits.read_bit()? {
            log2_rows += 1;
        }
        row_starts = uniform_starts(
            log2_rows,
            step,
            geometry.sequence_superblock_log2,
            geometry.size.1,
        );
    } else {
        column_starts = Vec::new();
        let mut widest = 0;
        let mut start = 0;
        while start < columns && column_starts.len() < MAX_TILES_PER_AXIS {
            let most = (columns - start).min(max_tile_width);
            let width = if most > 1 {
                1 + uniform(bits, most)?
            } else {
                1
            };
            column_starts.push(start as u16);
            start += width;
            widest = widest.max(width);
        }
        let mut area = columns * rows;
        if min_log2_tiles != 0 {
            area >>= min_log2_tiles + 1;
        }
        let max_tile_height = (area / widest.max(1)).max(1);
        row_starts = Vec::new();
        let mut start = 0;
        while start < rows && row_starts.len() < MAX_TILES_PER_AXIS {
            let most = (rows - start).min(max_tile_height);
            let height = if most > 1 {
                1 + uniform(bits, most)?
            } else {
                1
            };
            row_starts.push(start as u16);
            start += height;
        }
    }
    column_starts.push(columns as u16);
    row_starts.push(rows as u16);
    Ok(TileLayout {
        uniform: uniform_spacing,
        column_starts,
        row_starts,
    })
}
