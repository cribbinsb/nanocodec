//! Which AV2 edges are deblocked, with which lengths and thresholds (AVM
//! v1.0.0 `set_lpf_parameters`, `get_transform_size` and
//! `av2_filter_block_plane_{vert,horz}` in `av2/common/av2_loopfilter.c`;
//! BSD-3-Clause-Clear, see `THIRD_PARTY.md`), driving nc-dsp's
//! [`Dsp::av2_deblock_edge`] over a reconstructed picture.
//!
//! The coder records each block in an [`EdgeGrid`] (its square transform
//! sizes, whether it is a skipped inter block, its qindex). An edge segment
//! (four samples along a transform edge) is filtered when:
//!
//! - it is not on the picture's left or top border;
//! - the block on its q side has a transform edge there;
//! - either side's thresholds are nonzero;
//! - the q-side block is not a skipped inter block, or the edge is that
//!   block's own boundary (a skipped block has no residual edges inside).
//!
//! The maximum filter lengths come from the smaller of the two sides'
//! transform sizes (clipped near the right/bottom border), shortened on the
//! p side of chroma edges on 64-row luma superblock boundaries.
//!
//! AVM filters superblock by superblock, interleaving vertical edges of one
//! superblock with horizontal edges of the previous. Every filter stays
//! inside its own superblock row and column span except for the p side of
//! edges on a superblock boundary, which it reaches only after those
//! samples' own edges are done, so filtering all vertical edges of a plane
//! (left to right in each row) and then all horizontal edges (top to bottom
//! in each column) gives identical samples.
//!
//! The subset has no sub-PU edges (`enable_lf_sub_pu` off), no lossless
//! segments (a lossless frame has no deblocking syntax) and one tile.

use super::thresholds::ThresholdTable;
use super::{DeblockParams, FilterChannel};
use crate::tx_size::TxSize;
use nc_common::{Picture, PlaneIndex};
use nc_dsp::{AV2_EDGE_LINES, Av2EdgeFilter, Dsp};
use std::ops::Range;

/// Mode-information unit (AVM `MI_SIZE`): 4×4 luma samples.
const MI_LOG2: u32 = 2;
/// Spacing of the edges the filter visits, in samples of any plane.
const EDGE_STEP: usize = 4;
/// AVM `DF_DELTA_SCALE`: qindex offset per signalled delta unit.
pub(crate) const DELTA_SCALE: i32 = 8;
/// Rows of the 64×64 luma superblock grid whose chroma edges use a short
/// p side (`horz_superblock_edge`), in luma samples.
const SUPERBLOCK_ROW: usize = 64;
/// Within this many samples of the right/bottom border the luma (chroma)
/// transform size is clipped to 16 (8) so the filter stays inside.
const LUMA_BORDER_CLIP: usize = 16;
const CHROMA_BORDER_CLIP: usize = 8;

/// Maximum filter lengths `(p side, q side)` by the smaller transform side.
const LENGTH_4: (u8, u8) = (4, 4);
const LENGTH_8: (u8, u8) = (8, 8);
const LUMA_LENGTH_16: (u8, u8) = (14, 14);
const LUMA_LENGTH_32: (u8, u8) = (18, 18);
/// Luma 32+ edges on a superblock row keep a 14-tap p side.
const LUMA_LENGTH_32_SUPERBLOCK: (u8, u8) = (14, 18);
const CHROMA_LENGTH_16: (u8, u8) = (10, 10);
/// Chroma edges on a superblock row: at most two p samples change, which
/// keeps the line buffer above the superblock row short.
const CHROMA_LENGTH_8_SUPERBLOCK: (u8, u8) = (6, 8);
const CHROMA_LENGTH_16_SUPERBLOCK: (u8, u8) = (6, 10);

/// What one coded block contributes to its edges. Positions are `u16`
/// (pictures stay below 2^16 samples) and the side `u8` (at most 64), so a
/// record is 8 bytes instead of 32 (read for every filtered edge segment).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockEdges {
    /// Top-left luma sample.
    pub x: u16,
    pub y: u16,
    /// Side of the square block in luma samples.
    pub size: u8,
    /// Luma transform size (the whole block for a skipped inter block,
    /// AVM `max_txsize_rect_lookup`).
    pub luma_tx: TxSize,
    /// Chroma transform size (AVM `av2_get_max_uv_txsize`).
    pub chroma_tx: TxSize,
    /// An inter block without residual (`skip_txfm`).
    pub skipped_inter: bool,
    /// The block's luma AC qindex (`final_qindex_ac[0]`).
    pub luma_qindex: u8,
    /// The block's chroma AC qindex (`final_qindex_ac[1]`, equal for V:
    /// the frame chroma delta applied, [`crate::transform::plane_qindex`]).
    pub chroma_qindex: u8,
}

/// Per-4×4 record of the blocks of one picture.
///
/// Each 4×4 unit holds the index of its block in `blocks` rather than the
/// block itself: recording a 64×64 block then writes 256 four-byte indices
/// instead of 256 32-byte records, which made `record` most of
/// `encode_inter`'s self time on the Jetson (perf, 7c7bf5b9).
#[derive(Debug, Clone)]
pub(crate) struct EdgeGrid {
    columns: usize,
    rows: usize,
    blocks: Vec<BlockEdges>,
    units: Vec<u32>,
}

/// Index of a unit no block has been recorded over yet.
const UNRECORDED: u32 = u32::MAX;

impl EdgeGrid {
    /// An empty grid for a `width` x `height` luma picture.
    pub fn new(width: usize, height: usize) -> Self {
        let columns = width.div_ceil(1 << MI_LOG2);
        let rows = height.div_ceil(1 << MI_LOG2);
        Self {
            columns,
            rows,
            blocks: Vec::new(),
            units: vec![UNRECORDED; columns * rows],
        }
    }

    /// Record a coded block.
    pub fn record(&mut self, block: BlockEdges) {
        let index = u32::try_from(self.blocks.len()).expect("a picture has fewer than 2^32 blocks");
        self.blocks.push(block);
        let first_column = usize::from(block.x) >> MI_LOG2;
        let first_row = usize::from(block.y) >> MI_LOG2;
        let units = usize::from(block.size) >> MI_LOG2;
        for row in first_row..(first_row + units).min(self.rows) {
            let start = row * self.columns + first_column;
            let end = row * self.columns + (first_column + units).min(self.columns);
            // Blocks never overlap, so every block in `blocks` is the block
            // of all its units ([`filter_picture`] walks the list).
            debug_assert!(
                self.units[start..end]
                    .iter()
                    .all(|&unit| unit == UNRECORDED)
            );
            self.units[start..end].fill(index);
        }
    }

    /// Every recorded block.
    fn blocks(&self) -> impl Iterator<Item = BlockEdges> + '_ {
        self.blocks.iter().copied()
    }

    /// The block covering luma sample `(x, y)`.
    fn block_at(&self, x: usize, y: usize) -> BlockEdges {
        let index = self.units[(y >> MI_LOG2) * self.columns + (x >> MI_LOG2)];
        assert!(
            index != UNRECORDED,
            "every block of the picture is recorded before deblocking"
        );
        self.blocks[index as usize]
    }
}

/// One plane's view of a block: transform side, block origin and side, in
/// that plane's samples.
#[derive(Debug, Clone, Copy)]
struct PlaneBlock {
    tx: usize,
    origin: usize,
    /// The block's AC qindex in this plane.
    qindex: u8,
    block: BlockEdges,
}

/// Filter `picture` (the complete reconstruction) in place; `map` is kept
/// from picture to picture (clear between uses), so its picture-sized
/// storage is allocated and zeroed once.
pub(crate) fn filter_picture(
    picture: &mut Picture,
    grid: &EdgeGrid,
    params: DeblockParams,
    dsp: &Dsp,
    map: &mut EdgeMap,
) {
    for plane in PlaneIndex::ALL {
        for vertical in [true, false] {
            let channel = FilterChannel::of(plane, vertical);
            if let Some(delta) = params.delta(channel) {
                let thresholds = ThresholdTable::new(i32::from(delta) * DELTA_SCALE);
                filter_direction(picture, grid, plane, vertical, &thresholds, dsp, map);
            }
        }
    }
}

/// All edges of one plane and direction, in AVM's order along each line.
fn filter_direction(
    picture: &mut Picture,
    grid: &EdgeGrid,
    plane: PlaneIndex,
    vertical: bool,
    thresholds: &ThresholdTable,
    dsp: &Dsp,
    map: &mut EdgeMap,
) {
    let mut view = picture.plane_mut(plane).view_mut();
    let (width, height) = (view.width(), view.height());
    debug_assert!(width.is_multiple_of(AV2_EDGE_LINES) && height.is_multiple_of(AV2_EDGE_LINES));
    let lines = EdgeLines {
        grid,
        shift: usize::from(plane != PlaneIndex::Y),
        luma: plane == PlaneIndex::Y,
        vertical,
        across_extent: if vertical { width } else { height },
        along_extent: if vertical { height } else { width },
    };
    lines.mark(thresholds, map);
    // Edge line by edge line (vertical edges column by column, horizontal
    // edges row by row) keeps AVM's order along every line: a vertical
    // edge's segments of different rows share no sample, nor do a
    // horizontal edge's segments of different columns. The picture border
    // (line 0) is never marked.
    for line in 1..map.lines {
        if map.active[line] {
            dsp.av2_deblock_line(&mut view, line * EDGE_STEP, vertical, map.row(line));
            map.clear_row(line);
        }
    }
}

/// The filter of every edge segment of one plane and direction: a row of
/// segments per edge line (every [`EDGE_STEP`] samples across), `None`
/// where a segment is left alone. Kept clear between directions: only the
/// rows of active lines are written, and each is cleared once filtered.
#[derive(Debug, Default)]
pub(crate) struct EdgeMap {
    lines: usize,
    segments: usize,
    filters: Vec<Option<Av2EdgeFilter>>,
    /// Lines with at least one filtered segment.
    active: Vec<bool>,
}

impl EdgeMap {
    /// Size the (clear) map for `lines` edge lines of `segments` segments.
    /// Storage only grows: a chroma plane's smaller map followed by the
    /// next picture's luma map would otherwise zero it again.
    fn reset(&mut self, lines: usize, segments: usize) {
        debug_assert!(self.active.iter().all(|&active| !active));
        self.lines = lines;
        self.segments = segments;
        if self.filters.len() < lines * segments {
            self.filters.resize(lines * segments, None);
        }
        if self.active.len() < lines {
            self.active.resize(lines, false);
        }
    }

    fn row(&self, line: usize) -> &[Option<Av2EdgeFilter>] {
        &self.filters[line * self.segments..(line + 1) * self.segments]
    }

    /// Set segments `segments` of `line` to `filter`.
    fn set(&mut self, line: usize, segments: Range<usize>, filter: Option<Av2EdgeFilter>) {
        if filter.is_some() {
            let row = line * self.segments;
            self.filters[row + segments.start..row + segments.end].fill(filter);
            self.active[line] = true;
        }
    }

    fn clear_row(&mut self, line: usize) {
        self.filters[line * self.segments..(line + 1) * self.segments].fill(None);
        self.active[line] = false;
    }
}

/// The edge lines of one plane and direction.
struct EdgeLines<'a> {
    grid: &'a EdgeGrid,
    /// Chroma subsampling shift of the plane.
    shift: usize,
    luma: bool,
    vertical: bool,
    /// Plane size across and along the edges.
    across_extent: usize,
    along_extent: usize,
}

impl EdgeLines<'_> {
    /// The block covering plane sample `along` of the edge line at
    /// `across`.
    fn block_at(&self, across: usize, along: usize) -> BlockEdges {
        let (x, y) = if self.vertical {
            (across, along)
        } else {
            (along, across)
        };
        self.grid.block_at(x << self.shift, y << self.shift)
    }

    /// This plane's view of `block`.
    fn plane_block(&self, block: BlockEdges) -> PlaneBlock {
        let origin = usize::from(if self.vertical { block.x } else { block.y }) >> self.shift;
        let (tx, qindex) = if self.luma {
            (block.luma_tx, block.luma_qindex)
        } else {
            (block.chroma_tx, block.chroma_qindex)
        };
        PlaneBlock {
            tx: tx.width(),
            origin,
            qindex,
            block,
        }
    }

    /// End of `block` across the edge lines, in plane samples.
    fn across_end(&self, block: BlockEdges) -> usize {
        (usize::from(if self.vertical { block.x } else { block.y }) + usize::from(block.size))
            >> self.shift
    }

    /// Start of `block` along the edge lines, in plane samples.
    fn along_start(&self, block: BlockEdges) -> usize {
        usize::from(if self.vertical { block.y } else { block.x }) >> self.shift
    }

    /// End of `block` along the edge lines, in plane samples.
    fn along_end(&self, block: BlockEdges) -> usize {
        (usize::from(if self.vertical { block.y } else { block.x }) + usize::from(block.size))
            >> self.shift
    }

    /// The segments whose first sample lies inside `block` along the edge
    /// lines: the segments `block` is the q side of on its edge lines.
    fn segments(&self, block: BlockEdges) -> Range<usize> {
        let end = self.along_end(block).min(self.along_extent);
        self.along_start(block).div_ceil(EDGE_STEP)..end.div_ceil(EDGE_STEP)
    }

    /// Fill the (clear) `map` with the filter of every segment.
    ///
    /// Block by block, only the lines on a block's transform edges are
    /// visited; walking every edge line along all its blocks visited each
    /// block once per line it crosses.
    fn mark(&self, thresholds: &ThresholdTable, map: &mut EdgeMap) {
        map.reset(
            self.across_extent / EDGE_STEP,
            self.along_extent / EDGE_STEP,
        );
        for block in self.grid.blocks() {
            self.mark_block(block, thresholds, map);
        }
    }

    /// Mark the segments `block` is the q side of on each of its transform
    /// edges. A segment is filtered when its line is a transform edge of
    /// the block covering its first sample (the q side), with the block
    /// covering the sample before it across the line as the p side.
    fn mark_block(&self, block: BlockEdges, thresholds: &ThresholdTable, map: &mut EdgeMap) {
        let segments = self.segments(block);
        if segments.is_empty() {
            return;
        }
        let current = self.plane_block(block);
        let end = self.across_end(block).min(self.across_extent);
        // Transform edges lie every `tx` samples from the block's origin;
        // only those on the EDGE_STEP grid start a line, and the picture
        // border (coordinate 0) is not filtered.
        let edges = (current.origin..end).step_by(current.tx);
        for coordinate in edges.filter(|&c| c != 0 && c.is_multiple_of(EDGE_STEP)) {
            let position = EdgePosition {
                coordinate,
                extent: self.across_extent,
                luma: self.luma,
                superblock_row: !self.vertical
                    && (coordinate << self.shift).is_multiple_of(SUPERBLOCK_ROW),
            };
            let line = coordinate / EDGE_STEP;
            if coordinate != current.origin {
                // Inside the block both sides are the block itself.
                let filter = edge_filter(current, current, position, thresholds);
                map.set(line, segments.clone(), filter);
                continue;
            }
            // On the block's own edge the p side changes with the
            // neighbouring blocks; segments facing one neighbour share a
            // filter.
            let mut segment = segments.start;
            while segment < segments.end {
                let previous = self.block_at(coordinate - 1, segment * EDGE_STEP);
                let run_end = self.segments(previous).end.min(segments.end);
                let filter = edge_filter(current, self.plane_block(previous), position, thresholds);
                map.set(line, segment..run_end, filter);
                segment = run_end;
            }
        }
    }
}

/// Where an edge lies in its plane.
#[derive(Debug, Clone, Copy)]
struct EdgePosition {
    /// Sample index of q0 across the edge.
    coordinate: usize,
    /// Plane size across the edge.
    extent: usize,
    luma: bool,
    /// A horizontal edge on a 64-row luma superblock boundary.
    superblock_row: bool,
}

/// The filter of one transform edge between `previous` (p side) and
/// `current` (q side), or `None` if it is not filtered.
fn edge_filter(
    current: PlaneBlock,
    previous: PlaneBlock,
    position: EdgePosition,
    thresholds: &ThresholdTable,
) -> Option<Av2EdgeFilter> {
    let (current_q, current_side) = thresholds.get(current.qindex);
    let (previous_q, previous_side) = thresholds.get(previous.qindex);
    let enabled = (current_q != 0 && current_side != 0) || (previous_q != 0 && previous_side != 0);
    let block_edge = position.coordinate == current.origin;
    if !enabled || (current.block.skipped_inter && !block_edge) {
        return None;
    }
    let clip = if position.luma {
        LUMA_BORDER_CLIP
    } else {
        CHROMA_BORDER_CLIP
    };
    let current_tx = if position.extent < position.coordinate + clip {
        current.tx.min(clip)
    } else {
        current.tx
    };
    let (length_neg, length_pos) = lengths(current_tx.min(previous.tx), position);
    Some(Av2EdgeFilter {
        length_neg,
        length_pos,
        q_threshold: mean(current_q, previous_q),
        side_threshold: mean(current_side, previous_side),
    })
}

/// Maximum filter lengths for the smaller transform side `tx`.
fn lengths(tx: usize, position: EdgePosition) -> (u8, u8) {
    let superblock = position.superblock_row;
    match (tx, position.luma) {
        (4, _) => LENGTH_4,
        (8, false) if superblock => CHROMA_LENGTH_8_SUPERBLOCK,
        (8, _) => LENGTH_8,
        (16, true) => LUMA_LENGTH_16,
        (_, true) if superblock => LUMA_LENGTH_32_SUPERBLOCK,
        (_, true) => LUMA_LENGTH_32,
        (_, false) if superblock => CHROMA_LENGTH_16_SUPERBLOCK,
        (_, false) => CHROMA_LENGTH_16,
    }
}

/// Rounded mean of two sides' thresholds where both are nonzero, otherwise
/// the nonzero one (a skipped side does not weaken the filter).
fn mean(current: u16, previous: u16) -> u16 {
    match (current, previous) {
        (0, other) | (other, 0) => other,
        (a, b) => (a + b).div_ceil(2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_encoder::DebugRng;

    /// Luma size of the test pictures: not a whole number of 64×64
    /// superblocks, so blocks cross the right and bottom borders.
    const WIDTH: usize = 136;
    const HEIGHT: usize = 104;
    const SUPERBLOCK: usize = 64;

    /// Split the square at `(x, y)` at random down to 4×4 and record its
    /// blocks, skipping those entirely outside the picture.
    fn record_square(grid: &mut EdgeGrid, random: &mut DebugRng, x: usize, y: usize, size: usize) {
        if x >= WIDTH || y >= HEIGHT {
            return;
        }
        if size > 4 && random.chance(150) {
            let half = size / 2;
            for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
                record_square(grid, random, x + dx, y + dy, half);
            }
            return;
        }
        let log2 = size.trailing_zeros();
        let random_tx = |random: &mut DebugRng, max_log2: u32| {
            let log2 = 2 + random.below(u64::from(max_log2 - 1)) as u32;
            TxSize::from_log2(log2).expect("square transform")
        };
        let luma_tx = random_tx(random, log2);
        let chroma_tx = random_tx(random, log2.saturating_sub(1).max(2));
        grid.record(BlockEdges {
            x: x as u16,
            y: y as u16,
            size: size as u8,
            luma_tx,
            chroma_tx,
            skipped_inter: random.chance(40),
            luma_qindex: random.below(256) as u8,
            chroma_qindex: random.below(256) as u8,
        });
    }

    /// The filter of segment `segment` of the edge line at `coordinate`
    /// straight from the definition (module docs): the q side is the
    /// block covering the segment's first sample, the p side the block
    /// before it across the line.
    fn reference(
        lines: &EdgeLines<'_>,
        thresholds: &ThresholdTable,
        coordinate: usize,
        segment: usize,
    ) -> Option<Av2EdgeFilter> {
        let along = segment * EDGE_STEP;
        let current = lines.plane_block(lines.block_at(coordinate, along));
        if !(coordinate - current.origin).is_multiple_of(current.tx) {
            return None;
        }
        let previous = lines.plane_block(lines.block_at(coordinate - 1, along));
        let position = EdgePosition {
            coordinate,
            extent: lines.across_extent,
            luma: lines.luma,
            superblock_row: !lines.vertical
                && (coordinate << lines.shift).is_multiple_of(SUPERBLOCK_ROW),
        };
        edge_filter(current, previous, position, thresholds)
    }

    #[test]
    fn edge_map_matches_the_per_segment_definition() {
        let mut random = DebugRng::new(0x5eed_ed9e);
        let mut map = EdgeMap::default();
        for _ in 0..20 {
            let mut grid = EdgeGrid::new(WIDTH, HEIGHT);
            for y in (0..HEIGHT).step_by(SUPERBLOCK) {
                for x in (0..WIDTH).step_by(SUPERBLOCK) {
                    record_square(&mut grid, &mut random, x, y, SUPERBLOCK);
                }
            }
            let thresholds = ThresholdTable::new(random.symmetric(2) * DELTA_SCALE);
            for (shift, vertical) in [(0, true), (0, false), (1, true), (1, false)] {
                let (width, height) = (WIDTH >> shift, HEIGHT >> shift);
                let lines = EdgeLines {
                    grid: &grid,
                    shift,
                    luma: shift == 0,
                    vertical,
                    across_extent: if vertical { width } else { height },
                    along_extent: if vertical { height } else { width },
                };
                lines.mark(&thresholds, &mut map);
                for line in 0..map.lines {
                    let expected: Vec<_> = (0..map.segments)
                        .map(|segment| {
                            (line > 0)
                                .then(|| reference(&lines, &thresholds, line * EDGE_STEP, segment))
                                .flatten()
                        })
                        .collect();
                    assert_eq!(
                        map.row(line),
                        expected,
                        "line {line}, shift {shift}, vertical {vertical}"
                    );
                    assert_eq!(map.active[line], expected.iter().any(Option::is_some));
                    map.clear_row(line);
                }
            }
        }
    }
}
