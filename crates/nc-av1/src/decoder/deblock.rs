//! Deblocking of decoded frames (AV1 §7.14) from the decoder's own edge
//! state. Each block is recorded once: its position and size, its uniform
//! luma transform size (`LoopfilterTxSizes`; the leaves of an inter
//! transform tree are recorded as luma-only areas of their own), whether it
//! is a skipped inter block, whether it carries chroma, and its filter
//! levels (§7.14.4, with segment-free reference and mode deltas). Chroma
//! sizes follow from the block size (`get_plane_residual_size`,
//! `get_tx_size`), so U and V share one record. What a later edge reads
//! across from an area (the transform side, the area's extent along the
//! edge and its levels) goes into its right column, stored column-major
//! for vertical edges, and its bottom row for horizontal ones: two
//! contiguous runs, not the area. The filter visits each area's transform
//! edges: the first against its neighbours' records, in runs over which
//! the filter length and level stay the same, the inner ones (both sides
//! this area's) whole, as libaom's `av1/common/av1_loopfilter.c` decides
//! them (BSD-2-Clause, Copyright (c) 2016, Alliance for Open Media; see
//! `THIRD_PARTY.md`).
//!
//! Order (§7.14.1): per plane, every vertical edge, then every horizontal
//! one. Within a pass edges never interact (a filter's reach on each side
//! stays within the smaller transform side across the edge), so areas are
//! visited in decoding order.
use super::block_size::BlockSize;
use super::frame_header::{INTRA_FRAME, LoopFilter, TOTAL_REFS};
use super::transform::{TxSize, chroma_block};
use nc_common::{Picture, PlaneIndex, PlaneViewMut};
use nc_dsp::{AV1_EDGE_LINES, Av1EdgeFilter, Av1FilterLength, Dsp, EdgeRun};

/// Luma samples per 4x4 unit, log2.
const UNIT_LOG2: u8 = 2;
/// `MAX_LOOP_FILTER`.
const MAX_LEVEL: i32 = 63;
/// Level above which the deltas are scaled up (`nShift = lvlSeg >> 5`).
const DELTA_SCALE_SHIFT: u32 = 5;
/// `blimit = 2 · (level + 2) + limit`, `thresh = level >> 4` (§7.14.4).
const BLIMIT_LEVEL_OFFSET: u32 = 2;
const THRESH_SHIFT: u32 = 4;
/// Sharpness above which the inner limit shifts by two, not one.
const SHARPNESS_DOUBLE_SHIFT: u8 = 4;
/// `limit` is capped at `9 − sharpness` when sharpness is nonzero.
const SHARPNESS_LIMIT_BASE: u8 = 9;
/// The sides (4 and 8 samples, log2) below which the filter shortens.
const SIDE_4: u8 = 2;
const SIDE_8: u8 = 3;
/// Level index (`loop_filter_level[i]`) of U; V follows.
const U_LEVEL: usize = 2;

/// The levels of one block: `loop_filter_level[i]`-based for luma
/// vertical edges, luma horizontal edges, U and V.
pub(super) type BlockLevels = [u8; 4];

/// What an edge reads across from one 4x4 unit of an area's right column
/// (vertical edges) or bottom row (horizontal edges): the transform side
/// across the edge and the area's extent along it (log2 samples, 3 bits
/// each), and two levels of 6 bits: luma's for this direction, or U's and
/// V's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
struct Side(u32);

const SIDE_BITS: u32 = 3;
const SIDE_MASK: u32 = (1 << SIDE_BITS) - 1;
const LEVEL_BITS: u32 = 6;
const LEVEL_MASK: u32 = (1 << LEVEL_BITS) - 1;
const EXTENT_SHIFT: u32 = SIDE_BITS;
const LEVEL_SHIFT: u32 = 2 * SIDE_BITS;

impl Side {
    fn new(tx_side: u8, extent: u8, levels: [u8; 2]) -> Self {
        Self(
            u32::from(tx_side)
                | u32::from(extent) << EXTENT_SHIFT
                | u32::from(levels[0]) << LEVEL_SHIFT
                | u32::from(levels[1]) << (LEVEL_SHIFT + LEVEL_BITS),
        )
    }
    fn tx_side(self) -> u8 {
        (self.0 & SIDE_MASK) as u8
    }
    fn extent(self) -> u8 {
        ((self.0 >> EXTENT_SHIFT) & SIDE_MASK) as u8
    }
    /// Level 0 (luma, or U) or 1 (V).
    fn level(self, which: usize) -> u8 {
        ((self.0 >> (LEVEL_SHIFT + LEVEL_BITS * which as u32)) & LEVEL_MASK) as u8
    }
}

/// A block (or an inter transform tree leaf) as the loop filter sees it.
#[derive(Debug, Clone, Copy)]
pub(super) struct EdgeBlock {
    /// The block's size.
    pub size: BlockSize,
    /// A skipped inter block: only its block edges are filtered.
    pub skipped_inter: bool,
    /// The block's levels.
    pub levels: BlockLevels,
}

/// An area of a plane tiled by one transform size: its top-left unit,
/// its size, transform size and block size there (log2 samples), and two
/// levels (luma: vertical and horizontal edges; chroma: U and V).
#[derive(Debug, Clone, Copy)]
struct Area {
    origin: (u16, u16),
    size: (u8, u8),
    tx: (u8, u8),
    block: (u8, u8),
    skipped_inter: bool,
    levels: [u8; 2],
}

/// The areas of one plane (U and V share theirs), with their right-column
/// and bottom-row records.
#[derive(Debug, Default)]
struct PlaneSides {
    /// Every area, in decoding order.
    areas: Vec<Area>,
    /// Units per row and per column.
    columns: usize,
    rows: usize,
    /// Right-column records, column-major (`x * rows + y`).
    right: Vec<Side>,
    /// Bottom-row records, row-major (`y * columns + x`).
    bottom: Vec<Side>,
}

impl PlaneSides {
    fn start_frame(&mut self, columns: usize, rows: usize) {
        (self.columns, self.rows) = (columns, rows);
        self.areas.clear();
        self.right.resize(columns * rows, Side::default());
        self.bottom.resize(columns * rows, Side::default());
    }
    /// Record `area`, whose sides read `levels` for vertical and for
    /// horizontal edges; units outside the frame are left out.
    #[allow(
        clippy::inline_always,
        reason = "twice per block; out of line its set-up cost ~28M instructions on pan 360p"
    )]
    #[inline(always)]
    fn record(&mut self, area: Area, levels: [[u8; 2]; 2]) {
        self.areas.push(area);
        let (columns, rows) = (self.columns, self.rows);
        let (x4, y4) = (usize::from(area.origin.0), usize::from(area.origin.1));
        let (size, tx) = (area.size, area.tx);
        let (w4, h4) = (1 << (size.0 - UNIT_LOG2), 1 << (size.1 - UNIT_LOG2));
        let right = x4 + w4 - 1;
        if right < columns && y4 < rows {
            let first = right * rows;
            self.right[first + y4..first + (y4 + h4).min(rows)]
                .fill(Side::new(tx.0, size.1, levels[0]));
        }
        let bottom = y4 + h4 - 1;
        if bottom < rows && x4 < columns {
            let first = bottom * columns;
            self.bottom[first + x4..first + (x4 + w4).min(columns)]
                .fill(Side::new(tx.1, size.0, levels[1]));
        }
    }
}

/// The edge state of a frame, reused across frames.
#[derive(Debug, Default)]
pub(super) struct EdgeMap {
    /// The frame is deblocked (a luma level is nonzero, as [`apply`]
    /// tests): otherwise nothing is recorded, as nothing reads the map.
    recording: bool,
    luma: PlaneSides,
    /// U and V share their areas (and sizes).
    chroma: PlaneSides,
}

/// A block's chroma plane size and transform size (log2 samples).
fn chroma_sizes(block: BlockSize) -> ((u8, u8), (u8, u8)) {
    let tx = TxSize::chroma(block);
    (chroma_block(block), (tx.width_log2(), tx.height_log2()))
}

impl EdgeMap {
    /// Shape the map for a frame of `mi_cols` x `mi_rows` luma units
    /// (4:2:0, both even). Stale records are never read: a unit an edge
    /// reads across from is in the right column or bottom row of an area
    /// inside the frame, written before the filter runs. Frames that are
    /// not deblocked (`lf`'s luma levels both zero) record nothing.
    pub(super) fn start_frame(&mut self, mi_cols: usize, mi_rows: usize, lf: &LoopFilter) {
        self.recording = deblocked(lf);
        self.luma.start_frame(mi_cols, mi_rows);
        self.chroma.start_frame(mi_cols / 2, mi_rows / 2);
    }
    /// Record a block at luma unit (`mi_col`, `mi_row`), whose luma is one
    /// area of `luma_tx` transforms (`None` when its transform tree's
    /// leaves are recorded with [`EdgeMap::leaf`]) and which carries its
    /// chroma or not. Blocks starting outside the frame are not recorded.
    #[inline]
    pub(super) fn block(
        &mut self,
        (mi_col, mi_row): (usize, usize),
        block: EdgeBlock,
        luma_tx: Option<(u8, u8)>,
        chroma: bool,
    ) {
        if !self.recording || mi_col >= self.luma.columns || mi_row >= self.luma.rows {
            return;
        }
        let l = block.levels;
        if let Some(tx) = luma_tx {
            let size = (block.size.width_log2(), block.size.height_log2());
            let area = Area {
                origin: (mi_col as u16, mi_row as u16),
                size,
                tx,
                block: size,
                skipped_inter: block.skipped_inter,
                levels: [l[0], l[1]],
            };
            self.luma.record(area, [[l[0]; 2], [l[1]; 2]]);
        }
        if chroma {
            let (size, tx) = chroma_sizes(block.size);
            let levels = [l[U_LEVEL], l[U_LEVEL + 1]];
            let area = Area {
                origin: ((mi_col >> 1) as u16, (mi_row >> 1) as u16),
                size,
                tx,
                block: size,
                skipped_inter: block.skipped_inter,
                levels,
            };
            self.chroma.record(area, [levels, levels]);
        }
    }
    /// Record one leaf of an inter block's transform tree: a luma area of
    /// one `tx` transform at luma unit (`x4`, `y4`).
    pub(super) fn leaf(&mut self, (x4, y4): (usize, usize), tx: (u8, u8), block: EdgeBlock) {
        if !self.recording || x4 >= self.luma.columns || y4 >= self.luma.rows {
            return;
        }
        let l = block.levels;
        let area = Area {
            origin: (x4 as u16, y4 as u16),
            size: tx,
            tx,
            block: (block.size.width_log2(), block.size.height_log2()),
            skipped_inter: block.skipped_inter,
            levels: [l[0], l[1]],
        };
        self.luma.record(area, [[l[0]; 2], [l[1]; 2]]);
    }
}

/// The levels of every (reference, mode type) pair of a frame (§7.14.4
/// without segment features or block deltas): dav1d's level table.
#[derive(Debug, Clone, Copy)]
pub(super) struct LevelTable {
    levels: [[BlockLevels; 2]; TOTAL_REFS],
}

impl LevelTable {
    pub(super) fn new(lf: &LoopFilter) -> Self {
        let mut levels = [[[0; 4]; 2]; TOTAL_REFS];
        for (reference, by_mode) in levels.iter_mut().enumerate() {
            for (mode_type, out) in by_mode.iter_mut().enumerate() {
                for (i, level) in out.iter_mut().enumerate() {
                    *level = filter_level(lf, i, reference, mode_type);
                }
            }
        }
        Self { levels }
    }
    /// The levels of a block predicted from `reference` (`INTRA_FRAME` for
    /// intra blocks); `mode_type` is 1 for inter modes other than
    /// `GLOBALMV` (`NEARESTMV`, `NEARMV`, `NEWMV`).
    pub(super) fn get(&self, reference: usize, mode_type: usize) -> BlockLevels {
        self.levels[reference][mode_type]
    }
}

/// `get_filter_level` (§7.14.4) of edge class `i` (luma vertical, luma
/// horizontal, U, V).
fn filter_level(lf: &LoopFilter, i: usize, reference: usize, mode_type: usize) -> u8 {
    let base = i32::from(lf.level[i]).clamp(0, MAX_LEVEL);
    if !lf.delta_enabled {
        return base as u8;
    }
    let shift = base >> DELTA_SCALE_SHIFT;
    let mut level = base + (i32::from(lf.deltas.refs[reference]) << shift);
    if reference != INTRA_FRAME {
        level += i32::from(lf.deltas.modes[mode_type]) << shift;
    }
    level.clamp(0, MAX_LEVEL) as u8
}

/// `limit`, `blimit` and `thresh` of `level` (§7.14.4).
fn thresholds(sharpness: u8, level: u8, length: Av1FilterLength) -> Av1EdgeFilter {
    let shift = u32::from(sharpness > 0) + u32::from(sharpness > SHARPNESS_DOUBLE_SHIFT);
    let mut limit = level >> shift;
    if sharpness > 0 {
        limit = limit.min(SHARPNESS_LIMIT_BASE - sharpness);
    }
    let limit = limit.max(1);
    Av1EdgeFilter {
        length,
        limit,
        blimit: (2 * (u32::from(level) + BLIMIT_LEVEL_OFFSET) + u32::from(limit)) as u8,
        thresh: level >> THRESH_SHIFT,
    }
}

/// [`thresholds`] of every level of a frame (length 4 as a placeholder).
type Limits = [Av1EdgeFilter; MAX_LEVEL as usize + 1];

fn limits(sharpness: u8) -> Limits {
    std::array::from_fn(|level| thresholds(sharpness, level as u8, Av1FilterLength::Four))
}

/// Filter length from the smaller transform side across the edge (log2,
/// §7.14.3): luma 4 / 8 / 14 taps for 4 / 8 / 16+ samples, chroma 4 / 6.
fn filter_length(luma: bool, side_log2: u8) -> Av1FilterLength {
    match (luma, side_log2) {
        (_, SIDE_4) => Av1FilterLength::Four,
        (false, _) => Av1FilterLength::Six,
        (true, SIDE_8) => Av1FilterLength::Eight,
        (true, _) => Av1FilterLength::Fourteen,
    }
}

/// Whether a frame is deblocked at all: §7.14.1 skips the whole process
/// when both luma levels are zero (chroma levels are then not coded).
fn deblocked(lf: &LoopFilter) -> bool {
    lf.level[0] != 0 || lf.level[1] != 0
}

/// Deblock `recon` (§7.14) with the frame's `lf` parameters. `width` x
/// `height` is the frame size: edges at or beyond it are not filtered.
pub(super) fn apply(
    dsp: &Dsp,
    recon: &mut Picture,
    map: &EdgeMap,
    lf: &LoopFilter,
    (width, height): (usize, usize),
    runs: &mut Vec<EdgeRun<Av1EdgeFilter>>,
) {
    if !deblocked(lf) {
        return;
    }
    let limits = limits(lf.sharpness);
    for plane in PlaneIndex::ALL {
        let index = plane as usize;
        if plane != PlaneIndex::Y && lf.level[index + 1] == 0 {
            continue;
        }
        let ss = usize::from(plane != PlaneIndex::Y);
        let mut view = recon.plane_mut(plane).view_mut();
        let size = ((width + ss) >> ss, (height + ss) >> ss);
        runs.clear();
        PlanePass::<true>::new(map, plane, &limits, size, runs).filter(dsp, &mut view);
        runs.clear();
        PlanePass::<false>::new(map, plane, &limits, size, runs).filter(dsp, &mut view);
    }
}

/// Lines of one edge waiting to be filtered: `start..end` along it, with
/// one filter.
#[derive(Debug, Clone, Copy)]
struct Run {
    start: usize,
    end: usize,
    filter: Av1EdgeFilter,
}

/// One pass over one plane: its vertical edges (`VERTICAL`) or its
/// horizontal ones, each pass compiled on its own.
///
/// Areas come in decoding order, which visits the areas along any one
/// edge line in order; so each line keeps one pending run, which the next
/// run on it extends when it continues it with the same filter. Edges
/// within a pass do not interact, so filtering later changes nothing, and
/// the filter is called once per run of lines rather than once per area.
struct PlanePass<'a, const VERTICAL: bool> {
    map: &'a EdgeMap,
    plane: PlaneIndex,
    limits: &'a Limits,
    /// The plane's part of the frame, in samples.
    size: (usize, usize),
    /// The pending run of each edge line (unit across the edges).
    pending: Vec<Option<Run>>,
    /// The pass's finished runs, in order, for one filter call.
    runs: &'a mut Vec<EdgeRun<Av1EdgeFilter>>,
}

impl<'a, const VERTICAL: bool> PlanePass<'a, VERTICAL> {
    fn new(
        map: &'a EdgeMap,
        plane: PlaneIndex,
        limits: &'a Limits,
        size: (usize, usize),
        runs: &'a mut Vec<EdgeRun<Av1EdgeFilter>>,
    ) -> Self {
        let sides = if plane == PlaneIndex::Y {
            &map.luma
        } else {
            &map.chroma
        };
        let lines = if VERTICAL { sides.columns } else { sides.rows };
        Self {
            map,
            plane,
            limits,
            size,
            pending: vec![None; lines],
            runs,
        }
    }

    fn luma(&self) -> bool {
        self.plane == PlaneIndex::Y
    }

    fn sides(&self) -> &'a PlaneSides {
        if self.luma() {
            &self.map.luma
        } else {
            &self.map.chroma
        }
    }

    /// Which of an area's two levels this pass reads: luma's vertical or
    /// horizontal edges, or U's or V's.
    fn level_index(&self) -> usize {
        match self.plane {
            PlaneIndex::Y => usize::from(!VERTICAL),
            PlaneIndex::U => 0,
            PlaneIndex::V => 1,
        }
    }

    fn filter(mut self, dsp: &Dsp, view: &mut PlaneViewMut<'_>) {
        let (plane_width, plane_height) = self.size;
        let which = self.level_index();
        for area in &self.sides().areas {
            let (x4, y4) = (usize::from(area.origin.0), usize::from(area.origin.1));
            let level = area.levels[which];
            // Coordinates across (x for a vertical edge) and along the edge.
            let (across, along, across_limit, along_limit) = if VERTICAL {
                (x4, y4, plane_width, plane_height)
            } else {
                (y4, x4, plane_height, plane_width)
            };
            let (size_across, size_along, tx_side, block_side) = if VERTICAL {
                (area.size.0, area.size.1, area.tx.0, area.block.0)
            } else {
                (area.size.1, area.size.0, area.tx.1, area.block.1)
            };
            let start = along << UNIT_LOG2;
            let end = (start + (1 << size_along)).min(along_limit);
            let first = across << UNIT_LOG2;
            let last = (first + (1 << size_across)).min(across_limit);
            // A skipped inter block filters only its block edges.
            let filtered =
                |edge: usize| !area.skipped_inter || edge.trailing_zeros() >= u32::from(block_side);
            // Edges at or beyond the frame's last sample are off screen
            // (§7.14.2 `onScreen`), though blocks there are decoded.
            if (1..across_limit).contains(&first) && filtered(first) {
                self.edge((across, start, end), tx_side, level);
            }
            // The inner edges have this area's transforms on both sides.
            let mut edge = first + (1 << tx_side);
            while edge < last && level != 0 {
                if filtered(edge) {
                    let length = filter_length(self.luma(), tx_side);
                    self.run(edge >> UNIT_LOG2, start..end, (level, length));
                }
                edge += 1 << tx_side;
            }
        }
        for across in 0..self.pending.len() {
            if let Some(run) = self.pending[across].take() {
                self.finish(across, run);
            }
        }
        dsp.av1_loop_filter_runs(view, VERTICAL, self.runs);
    }

    /// The edge at unit `across` from sample `start` to `end` along it, of
    /// an area whose transform side across the edge is `tx_side` (log2)
    /// and whose level is `level`, in segments of the areas on the other
    /// side. Each is aligned to its size, so it ends at the next multiple
    /// of its extent along the edge.
    fn edge(&mut self, (across, start, end): (usize, usize, usize), tx_side: u8, level: u8) {
        let sides = self.sides();
        let (records, stride) = if VERTICAL {
            (&sides.right, sides.rows)
        } else {
            (&sides.bottom, sides.columns)
        };
        let line = &records[(across - 1) * stride..across * stride];
        let which = usize::from(self.plane == PlaneIndex::V);
        let mut along = start;
        while along < end {
            let other = line[along >> UNIT_LOG2];
            let next = (((along >> other.extent()) + 1) << other.extent()).min(end);
            let edge_level = if level == 0 {
                other.level(which)
            } else {
                level
            };
            let length = filter_length(self.luma(), tx_side.min(other.tx_side()));
            self.run(across, along..next, (edge_level, length));
            along = next;
        }
    }

    /// Queue the lines `lines` of the edge at unit `across` with `level`
    /// (none at level 0) and filter length `length`: extend the line's
    /// pending run, or filter that and start a new one.
    // Inlined: called per edge segment, mostly to extend a run.
    #[inline]
    fn run(
        &mut self,
        across: usize,
        lines: core::ops::Range<usize>,
        (level, length): (u8, Av1FilterLength),
    ) {
        if level == 0 {
            return;
        }
        let filter = Av1EdgeFilter {
            length,
            ..self.limits[usize::from(level)]
        };
        let pending = &mut self.pending[across];
        if let Some(run) = pending
            && run.end == lines.start
            && run.filter == filter
        {
            run.end = lines.end;
            return;
        }
        let next = Run {
            start: lines.start,
            end: lines.end,
            filter,
        };
        if let Some(run) = pending.replace(next) {
            self.finish(across, run);
        }
    }

    /// Add the finished `run` of the edge line at unit `across` to the
    /// pass's run list (whole 4-line units, as the edge filter takes them).
    fn finish(&mut self, across: usize, run: Run) {
        let (x, y) = if VERTICAL {
            (across << UNIT_LOG2, run.start)
        } else {
            (run.start, across << UNIT_LOG2)
        };
        self.runs.push(EdgeRun {
            x,
            y,
            lines: (run.end - run.start).next_multiple_of(AV1_EDGE_LINES),
            filter: run.filter,
        });
    }
}

/// Lines one edge call filters, as the unit size.
const _: () = assert!(AV1_EDGE_LINES == 1 << UNIT_LOG2);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_follow_the_reference_and_mode_deltas() {
        let mut lf = LoopFilter {
            level: [40, 20, 10, 0],
            sharpness: 0,
            delta_enabled: true,
            ..LoopFilter::default()
        };
        lf.deltas.modes = [0, -2];
        // Default deltas: INTRA_FRAME +1, LAST 0, GOLDEN -1. Level 40 has
        // nShift 1, so the deltas double.
        let table = LevelTable::new(&lf);
        assert_eq!(table.get(INTRA_FRAME, 0), [42, 21, 11, 1]);
        assert_eq!(table.get(1, 1), [36, 18, 8, 0]);
        assert_eq!(table.get(4, 0), [38, 19, 9, 0]);
        lf.delta_enabled = false;
        assert_eq!(LevelTable::new(&lf).get(4, 1), [40, 20, 10, 0]);
    }

    #[test]
    fn thresholds_follow_the_sharpness_rules() {
        let t = thresholds(0, 10, Av1FilterLength::Four);
        assert_eq!((t.limit, t.blimit, t.thresh), (10, 34, 0));
        // 40 >> 2 = 10, capped at 9 - 5 = 4.
        let t = thresholds(5, 40, Av1FilterLength::Four);
        assert_eq!((t.limit, t.blimit, t.thresh), (4, 88, 2));
        assert_eq!(thresholds(5, 1, Av1FilterLength::Four).limit, 1);
    }

    #[test]
    fn chroma_sizes_follow_get_tx_size() {
        // 4x16 luma: 4x8 chroma; 128x128: 64x64 plane, 32x32 transforms;
        // 16x64: 8x32; 128x64: 64x32 plane, 32x32 transforms.
        assert_eq!(
            chroma_sizes(BlockSize::new(2, 4).unwrap()),
            ((2, 3), (2, 3))
        );
        assert_eq!(
            chroma_sizes(BlockSize::new(7, 7).unwrap()),
            ((6, 6), (5, 5))
        );
        assert_eq!(
            chroma_sizes(BlockSize::new(4, 6).unwrap()),
            ((3, 5), (3, 5))
        );
        assert_eq!(
            chroma_sizes(BlockSize::new(7, 6).unwrap()),
            ((6, 5), (5, 5))
        );
    }

    #[test]
    fn frames_not_deblocked_record_nothing() {
        let mut map = EdgeMap::default();
        map.start_frame(8, 4, &LoopFilter::default());
        let block = EdgeBlock {
            size: BlockSize::new(4, 3).unwrap(),
            skipped_inter: false,
            levels: [0; 4],
        };
        map.block((0, 0), block, Some((3, 3)), true);
        map.leaf((0, 0), (3, 3), block);
        assert!(map.luma.areas.is_empty() && map.chroma.areas.is_empty());
    }

    #[test]
    fn records_cover_the_area_inside_the_frame() {
        let mut map = EdgeMap::default();
        let deblocked = LoopFilter {
            level: [1, 1, 1, 1],
            ..LoopFilter::default()
        };
        map.start_frame(8, 4, &deblocked);
        let block = EdgeBlock {
            size: BlockSize::new(5, 4).unwrap(),
            skipped_inter: true,
            levels: [3, 4, 5, 6],
        };
        // A 32x16 block at luma unit (4, 2): its right column (unit 11)
        // and bottom row (unit 5) are outside the 8x4-unit frame.
        map.block((4, 2), block, Some((5, 4)), true);
        assert!(
            map.luma
                .right
                .iter()
                .chain(&map.luma.bottom)
                .all(|&r| r == Side::default())
        );
        assert_eq!(map.luma.areas.len(), 1);
        // A 16x8 block of 8x8 transforms at (0, 0): luma right column unit
        // 3 (rows 0 and 1), bottom row unit 1 (columns 0 to 3); its 8x4
        // chroma at chroma unit (0, 0).
        let block = EdgeBlock {
            size: BlockSize::new(4, 3).unwrap(),
            ..block
        };
        map.block((0, 0), block, Some((3, 3)), true);
        let right = map.luma.right[3 * 4 + 1];
        assert_eq!((right.tx_side(), right.extent(), right.level(0)), (3, 3, 3));
        let bottom = map.luma.bottom[8 + 3];
        assert_eq!(
            (bottom.tx_side(), bottom.extent(), bottom.level(0)),
            (3, 4, 4)
        );
        assert_eq!(map.luma.right[2 * 4], Side::default());
        let chroma = map.chroma.right[2];
        assert_eq!(
            (chroma.tx_side(), chroma.level(0), chroma.level(1)),
            (3, 5, 6)
        );
    }
}
