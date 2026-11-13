//! In-loop deblocking of a decoded frame with every Profile 0 feature:
//! per-segment levels, reference/mode deltas, rectangular and sub-8x8
//! blocks (VP9 §8.8; libvpx `vp9_loopfilter.c` `vp9_loop_filter_frame_init`
//! and `vp9_filter_block_plane_non420`, BSD-3-Clause, see
//! `THIRD_PARTY.md`).
//!
//! Superblocks run in raster order; in each, every plane filters all its
//! vertical edges, then all horizontal ones. Each 8x8 unit's edges come
//! from its block's transform size: 16-wide on 16x16/32x32 transform
//! edges, 8-wide on 8x8 edges and every 32-sample boundary, 4-wide
//! otherwise, plus the internal edge of 4x4 transforms. Skipped inter
//! blocks filter only their outer edges; a unit of level 0 filters none.

use nc_common::{Picture, PlaneIndex, PlaneViewMut};
use nc_dsp::{Vp9FilterLimits as Limits, Vp9FilterWidth as Width};

use super::block::{BlockInfo, INTRA_FRAME, PredMode};
use super::frame::SB_MI;
use super::header::{LoopFilter, MAX_LOOP_FILTER, SegFeature, Segmentation};
use crate::segmentation::MAX_SEGMENTS;
use crate::transform::TxSize;
use crate::types::MI_SIZE;

/// Reference frames (intra, LAST, GOLDEN, ALTREF) and mode classes
/// (`MAX_MODE_LF_DELTAS`: ZEROMV, other inter modes).
const REFERENCES: usize = 4;
const MODE_CLASSES: usize = 2;

/// `lfi_n->lvl`: the level of every segment, reference and mode class
/// (`vp9_loop_filter_frame_init`).
pub(super) struct Levels([[[u8; MODE_CLASSES]; REFERENCES]; MAX_SEGMENTS]);

impl Levels {
    pub(super) fn new(lf: &LoopFilter, seg: &Segmentation) -> Self {
        let base = i32::from(lf.level);
        // Deltas count double from level 32 (`n_shift`).
        let scale = 1 << (base >> 5);
        let mut levels = [[[0u8; MODE_CLASSES]; REFERENCES]; MAX_SEGMENTS];
        for (segment, table) in levels.iter_mut().enumerate() {
            let level = match seg.feature(segment as u8, SegFeature::AltLf) {
                Some(data) if seg.absolute => i32::from(data),
                Some(data) => base + i32::from(data),
                None => base,
            }
            .clamp(0, MAX_LOOP_FILTER);
            if !lf.deltas_enabled {
                *table = [[level as u8; MODE_CLASSES]; REFERENCES];
                continue;
            }
            let clamp = |value: i32| value.clamp(0, MAX_LOOP_FILTER) as u8;
            table[0][0] = clamp(level + i32::from(lf.ref_deltas[0]) * scale);
            for (reference, modes) in table.iter_mut().enumerate().skip(1) {
                for (mode, entry) in modes.iter_mut().enumerate() {
                    *entry = clamp(
                        level
                            + i32::from(lf.ref_deltas[reference]) * scale
                            + i32::from(lf.mode_deltas[mode]) * scale,
                    );
                }
            }
        }
        Self(levels)
    }

    /// `get_filter_level` (`mode_lf_lut`: ZEROMV and intra modes are class 0).
    fn of(&self, info: &BlockInfo) -> u8 {
        let class = usize::from(info.ref_frame[0] > INTRA_FRAME && info.mode != PredMode::ZeroMv);
        self.0[usize::from(info.segment_id)][info.ref_frame[0] as usize][class]
    }
}

/// The edges of one 8x8 unit along one direction in one plane type,
/// packed in a byte: the filter at its left/top edge (none, 4, 8 or 16
/// wide) and whether a 4-wide edge runs 4 samples in (4x4 transforms).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct EdgeCode(u8);

impl EdgeCode {
    const WIDTH_MASK: u8 = 0b11;
    const INTERNAL: u8 = 0b100;

    fn new(edge: Option<Width>, internal: bool) -> Self {
        let width = match edge {
            None => 0,
            Some(Width::Four) => 1,
            Some(Width::Eight) => 2,
            Some(Width::Sixteen) => 3,
        };
        Self(width | if internal { Self::INTERNAL } else { 0 })
    }

    fn edge(self) -> Option<Width> {
        match self.0 & Self::WIDTH_MASK {
            0 => None,
            1 => Some(Width::Four),
            2 => Some(Width::Eight),
            _ => Some(Width::Sixteen),
        }
    }

    fn internal(self) -> bool {
        self.0 & Self::INTERNAL != 0
    }
}

/// The field of a unit's edge codes: luma vertical, luma horizontal,
/// chroma vertical, chroma horizontal.
fn code_index(chroma: bool, vertical: bool) -> usize {
    2 * usize::from(chroma) + usize::from(!vertical)
}

/// What the filter needs of each 8x8 unit, recorded as blocks decode: the
/// filter level and the unit's edges, decided once per block
/// (`vp9_filter_block_plane_non420` mask building). Packed in one word,
/// the level in the low byte and each [`EdgeCode`] in a nibble above it
/// (indexed by [`code_index`]), so a block writes each unit with a single
/// store. Chroma codes exist at even units only (one per 4:2:0 chroma unit).
#[derive(Debug, Clone, Copy, Default)]
struct Unit(u32);

impl Unit {
    const LEVEL_BITS: u32 = 8;
    const CODE_BITS: u32 = 4;

    /// `code` in the field of edge codes `index`.
    fn code_bits(index: usize, code: EdgeCode) -> u32 {
        u32::from(code.0) << (Self::LEVEL_BITS + Self::CODE_BITS * index as u32)
    }

    fn level(self) -> u8 {
        self.0 as u8
    }

    fn code(self, index: usize) -> EdgeCode {
        let shift = Self::LEVEL_BITS + Self::CODE_BITS * index as u32;
        EdgeCode((self.0 >> shift) as u8 & ((1 << Self::CODE_BITS) - 1))
    }
}

/// What a block contributes to its units' filtering.
pub(super) struct BlockEdges {
    level: u8,
    tx: TxSize,
    uv_tx: TxSize,
    /// Skipped inter block: only its outer edges are filtered.
    skip_inter: bool,
    /// Sub-8x8 blocks start a block at every unit.
    sub8x8: bool,
}

impl BlockEdges {
    pub(super) fn of(info: &BlockInfo, levels: &Levels) -> Self {
        Self {
            level: levels.of(info),
            tx: info.tx_size,
            uv_tx: info.size.uv_tx(info.tx_size),
            skip_inter: info.skip && info.is_inter(),
            sub8x8: info.size.is_sub8x8(),
        }
    }
}

/// [`Unit`]s of every 8x8 unit of the frame.
#[derive(Default)]
pub(super) struct Units {
    cols: usize,
    rows: usize,
    cells: Vec<Unit>,
}

impl Units {
    /// These units resized for a frame of `cols` x `rows` 8x8 units. Every
    /// unit inside the frame is recorded by the block covering it before the
    /// filter reads it, so recycled cells need no clearing.
    pub(super) fn reuse(mut self, cols: usize, rows: usize) -> Self {
        self.cells.resize(cols * rows, Unit::default());
        (self.cols, self.rows) = (cols, rows);
        self
    }

    /// Record a block covering the 8x8 units of rows `rows` and columns
    /// `cols`.
    pub(super) fn record(
        &mut self,
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
        block: &BlockEdges,
    ) {
        if block.level == 0 {
            // Units start unfiltered; a level of 0 filters no edge.
            for row in rows {
                let start = row * self.cols;
                self.cells[start + cols.start..start + cols.end].fill(Unit::default());
            }
            return;
        }
        // Vertical edges vary only along the block's columns and
        // horizontal ones only along its rows: decide each line once.
        let mut vertical = [[EdgeCode::default(); 2]; SB_MI];
        for (code, col) in vertical.iter_mut().zip(cols.clone()) {
            let block_left = block.sub8x8 || col == cols.start;
            *code = Self::line_codes(block, col, col == self.cols - 1, block_left);
        }
        let mut horizontal = [[EdgeCode::default(); 2]; SB_MI];
        for (code, row) in horizontal.iter_mut().zip(rows.clone()) {
            let block_top = block.sub8x8 || row == rows.start;
            *code = Self::line_codes(block, row, row == self.rows - 1, block_top);
        }
        let vertical_bits = vertical.map(|[luma, chroma]| {
            Unit::code_bits(code_index(false, true), luma)
                | Unit::code_bits(code_index(true, true), chroma)
        });
        let last_col = self.cols - 1;
        for (row, [luma, chroma]) in rows.zip(horizontal) {
            let row_bits = u32::from(block.level)
                | Unit::code_bits(code_index(false, false), luma)
                | Unit::code_bits(code_index(true, false), chroma);
            // `mask_4x4_int` follows the column border; the horizontal
            // pass also drops it on the last chroma row.
            let border_bits =
                row_bits & !Unit::code_bits(code_index(true, false), EdgeCode(EdgeCode::INTERNAL));
            let start = row * self.cols;
            for ((cell, col), vertical) in self.cells[start + cols.start..start + cols.end]
                .iter_mut()
                .zip(cols.clone())
                .zip(vertical_bits)
            {
                let row_bits = if col == last_col {
                    border_bits
                } else {
                    row_bits
                };
                *cell = Unit(row_bits | vertical);
            }
        }
    }

    /// The luma and chroma codes of a block's edges at unit `index` (a
    /// column for vertical edges, a row for horizontal ones): `last` if
    /// it is the frame's last unit, `block_edge` if it starts the block.
    /// The chroma code applies at even units (4:2:0 chroma reads only
    /// those).
    fn line_codes(block: &BlockEdges, index: usize, last: bool, block_edge: bool) -> [EdgeCode; 2] {
        [false, true].map(|chroma| {
            let ss = usize::from(chroma);
            let tx = if chroma { block.uv_tx } else { block.tx };
            let border = chroma && last;
            // Position in plane 8x8 units within the superblock.
            let (edge, internal) = unit_edges(
                block.skip_inter,
                tx,
                (index % SB_MI) >> ss,
                block_edge,
                border,
            );
            EdgeCode::new(edge, internal && !border)
        })
    }

    fn at(&self, row: usize, col: usize) -> &Unit {
        &self.cells[row * self.cols + col]
    }
}

/// The frame's deblocking, one superblock row at a time.
///
/// Filtering a superblock row touches only that row and the bottom of the
/// one above (already filtered), and intra prediction reads unfiltered
/// samples of the row above only, so the decoder filters each row as soon
/// as the row below it has decoded, while its samples are still in cache
/// (as libvpx's decoder does); the result equals filtering the whole frame
/// afterwards.
pub(super) struct Deblocker {
    /// Limits per filter level; empty when the frame is not filtered.
    limits: Vec<Limits>,
}

impl Deblocker {
    pub(super) fn new(lf: &LoopFilter) -> Self {
        let limits = if lf.level == 0 {
            Vec::new()
        } else {
            (0..=MAX_LOOP_FILTER as u8)
                .map(|level| Limits::with_sharpness(level, lf.sharpness))
                .collect()
        };
        Self { limits }
    }

    /// Deblock the superblock row starting at 8x8 unit row `sb_row` in
    /// place; its blocks must all be decoded.
    pub(super) fn filter_row(
        &self,
        dsp: &nc_dsp::Dsp,
        recon: &mut Picture,
        units: &Units,
        sb_row: usize,
    ) {
        if self.limits.is_empty() {
            return;
        }
        // Planes are independent: one view per plane and row, not per edge.
        for plane in PlaneIndex::ALL {
            let mut view = recon.plane_mut(plane).view_mut();
            for sb_col in (0..units.cols).step_by(SB_MI) {
                filter_superblock(
                    dsp,
                    &mut view,
                    plane != PlaneIndex::Y,
                    units,
                    (sb_row, sb_col),
                    &self.limits,
                );
            }
        }
    }
}

/// The edges of a unit along one direction (`vp9_filter_block_plane_non420`
/// mask building): the filter at its left/top edge and whether it has an
/// internal 4x4 edge. `index` is the unit's position along the direction in
/// plane 8x8 units, `block_edge` whether the unit starts its block,
/// `border` whether a chroma unit is the frame's last mode-info unit.
fn unit_edges(
    skip_inter: bool,
    tx: TxSize,
    index: usize,
    block_edge: bool,
    border: bool,
) -> (Option<Width>, bool) {
    let edge = if skip_inter && !block_edge {
        None
    } else {
        match tx {
            TxSize::Tx32 if index.is_multiple_of(4) => {
                Some(if border { Width::Eight } else { Width::Sixteen })
            }
            TxSize::Tx16 if index.is_multiple_of(2) => {
                Some(if border { Width::Eight } else { Width::Sixteen })
            }
            TxSize::Tx32 | TxSize::Tx16 => None,
            // 8-wide filtering on every 32-sample boundary.
            TxSize::Tx8 => Some(Width::Eight),
            TxSize::Tx4 => Some(if index.is_multiple_of(4) {
                Width::Eight
            } else {
                Width::Four
            }),
        }
    };
    (edge, !skip_inter && tx == TxSize::Tx4)
}

/// The plane's unit offsets (every `step`-th 8x8 unit) of a superblock at
/// `start` that lie before `limit`, and their count.
fn units_inside(start: usize, limit: usize, step: usize) -> ([usize; SB_MI], usize) {
    let mut units = [0; SB_MI];
    let mut count = 0;
    for unit in (0..SB_MI).step_by(step).take_while(|&u| start + u < limit) {
        units[count] = unit;
        count += 1;
    }
    (units, count)
}

/// Filter the `units` x 8 lines across the edge at plane sample `(x, y)`;
/// lines beyond the plane are skipped.
fn filter_edge(
    dsp: &nc_dsp::Dsp,
    view: &mut PlaneViewMut<'_>,
    (x, y): (usize, usize),
    (vertical, units): (bool, usize),
    width: Width,
    limits: Limits,
) {
    let (plane_w, plane_h) = (view.width(), view.height());
    let stride = view.stride();
    let span = units * MI_SIZE;
    let (step, along, lines) = if vertical {
        (1, stride, span.min(plane_h - y))
    } else {
        (stride, 1, span.min(plane_w - x))
    };
    dsp.vp9_filter_edge(
        view.data_mut(),
        y * stride + x,
        step,
        along,
        lines,
        width,
        limits,
    );
}

/// Deblock one plane of the superblock at 8x8 unit `(sb_row, sb_col)`:
/// every vertical edge, then every horizontal one. Runs of units with the
/// same edges and level along an edge are one kernel call (the same
/// per-line order: an 8-line stripe is touched only by its own edges).
fn filter_superblock(
    dsp: &nc_dsp::Dsp,
    plane: &mut PlaneViewMut<'_>,
    chroma: bool,
    frame: &Units,
    (sb_row, sb_col): (usize, usize),
    limits: &[Limits],
) {
    let ss = usize::from(chroma);
    let step = 1 << ss;
    let (row_units, row_count) = units_inside(sb_row, frame.rows, step);
    let (col_units, col_count) = units_inside(sb_col, frame.cols, step);
    let (rows, cols) = (&row_units[..row_count], &col_units[..col_count]);
    // Each unit's edges and level as one comparable key.
    let key = |unit: &Unit, vertical: bool| {
        u16::from(unit.level()) << 8 | u16::from(unit.code(code_index(chroma, vertical)).0)
    };
    let mut vertical = [[0u16; SB_MI]; SB_MI];
    let mut horizontal = [[0u16; SB_MI]; SB_MI];
    for (i, &r) in rows.iter().enumerate() {
        for (j, &c) in cols.iter().enumerate() {
            let unit = frame.at(sb_row + r, sb_col + c);
            vertical[j][i] = key(unit, true);
            horizontal[i][j] = key(unit, false);
        }
    }
    let to_plane = |units: usize| (units * MI_SIZE) >> ss;
    for (j, &c) in cols.iter().enumerate() {
        let x = to_plane(sb_col + c);
        let mut i = 0;
        for run in vertical[j][..rows.len()].chunk_by(|a, b| a == b) {
            let y = to_plane(sb_row + rows[i]);
            filter_run(dsp, plane, (x, y), (true, run.len()), run[0], limits);
            i += run.len();
        }
    }
    for (i, &r) in rows.iter().enumerate() {
        let y = to_plane(sb_row + r);
        let mut j = 0;
        for run in horizontal[i][..cols.len()].chunk_by(|a, b| a == b) {
            let x = to_plane(sb_col + cols[j]);
            filter_run(dsp, plane, (x, y), (false, run.len()), run[0], limits);
            j += run.len();
        }
    }
}

/// Filter a run of `units` units with edge key `key` starting at plane
/// sample `(x, y)`: the outer edge (none at the frame's left or top) and
/// the internal 4x4 edge.
fn filter_run(
    dsp: &nc_dsp::Dsp,
    plane: &mut PlaneViewMut<'_>,
    (x, y): (usize, usize),
    (vertical, units): (bool, usize),
    key: u16,
    limits: &[Limits],
) {
    let level = key >> 8;
    if level == 0 {
        return;
    }
    let code = EdgeCode(key as u8);
    let limits = limits[usize::from(level)];
    let at_frame_edge = if vertical { x == 0 } else { y == 0 };
    if let Some(width) = code.edge()
        && !at_frame_edge
    {
        filter_edge(dsp, plane, (x, y), (vertical, units), width, limits);
    }
    if code.internal() {
        let inner = if vertical { (x + 4, y) } else { (x, y + 4) };
        filter_edge(dsp, plane, inner, (vertical, units), Width::Four, limits);
    }
}
