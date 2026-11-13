//! In-loop deblocking of a decoded frame with every Profile 0 feature:
//! per-segment levels, reference/mode deltas, rectangular and sub-8x8
//! blocks (VP9 §8.8). The edges are libvpx's per-superblock bit masks
//! (`vp9_loopfilter.c` `vp9_build_mask`, `vp9_adjust_mask`,
//! `vp9_filter_block_plane_ss00` / `_ss11`; BSD-3-Clause, see
//! `THIRD_PARTY.md`): each block ORs its edges into its superblock's masks
//! as it is stored, and the filter walks only the set bits.
//!
//! Superblocks run in raster order; in each, every plane filters all its
//! vertical edges, then all horizontal ones. Each 8x8 unit's edges come
//! from its block's transform size: 16-wide on 16x16/32x32 transform
//! edges, 8-wide on 8x8 edges and every 32-sample boundary, 4-wide
//! otherwise, plus the internal edge of 4x4 transforms. Skipped inter
//! blocks filter only their outer edges; a block of level 0 filters none.

use nc_common::{Picture, PlaneViewMut};
use nc_dsp::{Vp9FilterLimits as Limits, Vp9FilterWidth as Width};

use super::block::{BlockInfo, BlockSize, INTRA_FRAME, PredMode};
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

const BLOCK_SIZES: usize = BlockSize::ALL.len();
const TX_SIZES: usize = 4;

// Luma masks: bit `row * 8 + col` is the 8x8 unit at (row, col) of a
// superblock. Chroma (4:2:0) masks: bit `row * 4 + col` of its 4x4 units.

/// `left_64x64_txform_mask` / `above_64x64_txform_mask`: the units of a
/// 64x64 block whose left / top edge is a transform edge.
const LEFT_TX: [u64; TX_SIZES] = [
    u64::MAX,
    u64::MAX,
    0x5555_5555_5555_5555,
    0x1111_1111_1111_1111,
];
const ABOVE_TX: [u64; TX_SIZES] = [
    u64::MAX,
    u64::MAX,
    0x00ff_00ff_00ff_00ff,
    0x0000_00ff_0000_00ff,
];
/// `left_prediction_mask` / `above_prediction_mask`: a block's own left /
/// top units, at the superblock's origin.
const LEFT_BLOCK: [u64; BLOCK_SIZES] = [
    0x01,
    0x01,
    0x01,
    0x01,
    0x0101,
    0x01,
    0x0101,
    0x0101_0101,
    0x0101,
    0x0101_0101,
    0x0101_0101_0101_0101,
    0x0101_0101,
    0x0101_0101_0101_0101,
];
const ABOVE_BLOCK: [u64; BLOCK_SIZES] = [
    0x01, 0x01, 0x01, 0x01, 0x01, 0x03, 0x03, 0x03, 0x0f, 0x0f, 0x0f, 0xff, 0xff,
];
/// `size_mask`: every unit of a block, at the superblock's origin.
const BLOCK_UNITS: [u64; BLOCK_SIZES] = [
    0x01,
    0x01,
    0x01,
    0x01,
    0x0101,
    0x03,
    0x0303,
    0x0303_0303,
    0x0f0f,
    0x0f0f_0f0f,
    0x0f0f_0f0f_0f0f_0f0f,
    0xffff_ffff,
    u64::MAX,
];
/// Units on a 32-sample boundary (`left_border`, `above_border`): 4-wide
/// edges there filter 8 wide.
const LEFT_BORDER: u64 = 0x1111_1111_1111_1111;
const ABOVE_BORDER: u64 = 0x0000_00ff_0000_00ff;

/// The chroma counterparts (`*_uv`).
const LEFT_TX_UV: [u16; TX_SIZES] = [0xffff, 0xffff, 0x5555, 0x1111];
const ABOVE_TX_UV: [u16; TX_SIZES] = [0xffff, 0xffff, 0x0f0f, 0x000f];
const LEFT_BLOCK_UV: [u16; BLOCK_SIZES] = [
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x11, 0x01, 0x11, 0x1111, 0x11, 0x1111,
];
const ABOVE_BLOCK_UV: [u16; BLOCK_SIZES] = [
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x03, 0x03, 0x03, 0x0f, 0x0f,
];
const BLOCK_UNITS_UV: [u16; BLOCK_SIZES] = [
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x11, 0x03, 0x33, 0x3333, 0xff, 0xffff,
];
const LEFT_BORDER_UV: u16 = 0x1111;
const ABOVE_BORDER_UV: u16 = 0x000f;
/// Luma units of the frame's first column / row, which have no left / top edge.
const FIRST_COLUMN: u64 = 0x0101_0101_0101_0101;
const FIRST_COLUMN_UV: u16 = 0x1111;

/// `LOOP_FILTER_MASK`: the edges of one superblock, per transform size
/// (32x32 is folded into 16x16 by [`SuperblockMask::adjusted`]), and the
/// level of each luma unit.
#[derive(Debug, Clone, Copy)]
struct SuperblockMask {
    left_y: [u64; TX_SIZES],
    above_y: [u64; TX_SIZES],
    int_4x4_y: u64,
    left_uv: [u16; TX_SIZES],
    above_uv: [u16; TX_SIZES],
    int_4x4_uv: u16,
    levels: [u8; SB_MI * SB_MI],
}

impl Default for SuperblockMask {
    fn default() -> Self {
        Self {
            left_y: [0; TX_SIZES],
            above_y: [0; TX_SIZES],
            int_4x4_y: 0,
            left_uv: [0; TX_SIZES],
            above_uv: [0; TX_SIZES],
            int_4x4_uv: 0,
            levels: [0; SB_MI * SB_MI],
        }
    }
}

/// `vp9_build_mask`: whether a block at (row, col) in its superblock
/// writes the chroma masks (`first_block_in_16x16`: 4:2:0 chroma units
/// take the block at their top-left luma unit).
fn builds_chroma(row: usize, col: usize) -> bool {
    row.is_multiple_of(2) && col.is_multiple_of(2)
}

/// The masks of one direction of one plane, ready to filter: per unit the
/// edge width (16, 8 or 4 wide), the internal 4x4 edge, and the level.
/// `N` units per side, bit `index * N + position` where `index` is the
/// line of edges (a column for vertical edges, a row for horizontal ones).
struct EdgeMasks {
    sixteen: u64,
    eight: u64,
    four: u64,
    internal: u64,
}

impl EdgeMasks {
    fn any(&self) -> u64 {
        self.sixteen | self.eight | self.four | self.internal
    }
}

/// Transpose an 8x8 bit matrix (bit `row * 8 + col` to `col * 8 + row`;
/// Hacker's Delight 7-3).
fn transpose8(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00aa_00aa_00aa_00aa;
    x ^= t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_cccc_0000_cccc;
    x ^= t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_f0f0_f0f0;
    x ^ t ^ (t << 28)
}

/// Transpose a 4x4 bit matrix (bit `row * 4 + col` to `col * 4 + row`).
fn transpose4(x: u16) -> u16 {
    let mut x = u32::from(x);
    let t = (x ^ (x >> 3)) & 0x0a0a;
    x ^= t ^ (t << 3);
    let t = (x ^ (x >> 6)) & 0x00cc;
    x ^= t ^ (t << 6);
    x as u16
}

impl SuperblockMask {
    /// `vp9_build_mask` of a block at unit (row, col) of this superblock.
    fn add_block(&mut self, (row, col): (usize, usize), block: &BlockInfo, level: u8) {
        if level == 0 {
            return;
        }
        let size = block.size as usize;
        let (w, h) = (block.size.width_mi(), block.size.height_mi());
        for r in row..(row + h).min(SB_MI) {
            let start = r * SB_MI + col;
            self.levels[start..start + w.min(SB_MI - col)].fill(level);
        }
        let tx = block.tx_size as usize;
        let uv_tx = block.size.uv_tx(block.tx_size) as usize;
        let shift = row * SB_MI + col;
        let shift_uv = (row >> 1) * (SB_MI / 2) + (col >> 1);
        let chroma = builds_chroma(row, col);
        self.above_y[tx] |= ABOVE_BLOCK[size] << shift;
        self.left_y[tx] |= LEFT_BLOCK[size] << shift;
        if chroma {
            self.above_uv[uv_tx] |= ABOVE_BLOCK_UV[size] << shift_uv;
            self.left_uv[uv_tx] |= LEFT_BLOCK_UV[size] << shift_uv;
        }
        // Skipped inter blocks filter only their outer edges.
        if block.skip && block.is_inter() {
            return;
        }
        self.above_y[tx] |= (BLOCK_UNITS[size] & ABOVE_TX[tx]) << shift;
        self.left_y[tx] |= (BLOCK_UNITS[size] & LEFT_TX[tx]) << shift;
        if chroma {
            self.above_uv[uv_tx] |= (BLOCK_UNITS_UV[size] & ABOVE_TX_UV[uv_tx]) << shift_uv;
            self.left_uv[uv_tx] |= (BLOCK_UNITS_UV[size] & LEFT_TX_UV[uv_tx]) << shift_uv;
        }
        if tx == TxSize::Tx4 as usize {
            self.int_4x4_y |= BLOCK_UNITS[size] << shift;
        }
        if chroma && uv_tx == TxSize::Tx4 as usize {
            self.int_4x4_uv |= BLOCK_UNITS_UV[size] << shift_uv;
        }
    }

    /// `vp9_adjust_mask` for the superblock at unit (`mi_row`, `mi_col`)
    /// of a frame of `mi_rows` x `mi_cols` units: 32x32 edges filter as
    /// 16x16, 4-wide edges on 32-sample boundaries filter 8 wide, units
    /// outside the frame filter nothing, and so on.
    fn adjusted(
        mut self,
        (mi_row, mi_col): (usize, usize),
        (mi_rows, mi_cols): (usize, usize),
    ) -> Self {
        let (t4, t8, t16, t32) = (0, 1, 2, 3);
        self.left_y[t16] |= self.left_y[t32];
        self.above_y[t16] |= self.above_y[t32];
        self.left_uv[t16] |= self.left_uv[t32];
        self.above_uv[t16] |= self.above_uv[t32];
        self.left_y[t8] |= self.left_y[t4] & LEFT_BORDER;
        self.left_y[t4] &= !LEFT_BORDER;
        self.above_y[t8] |= self.above_y[t4] & ABOVE_BORDER;
        self.above_y[t4] &= !ABOVE_BORDER;
        self.left_uv[t8] |= self.left_uv[t4] & LEFT_BORDER_UV;
        self.left_uv[t4] &= !LEFT_BORDER_UV;
        self.above_uv[t8] |= self.above_uv[t4] & ABOVE_BORDER_UV;
        self.above_uv[t4] &= !ABOVE_BORDER_UV;
        if mi_row + SB_MI > mi_rows {
            let rows = mi_rows - mi_row;
            let mask_y = (1u64 << (rows << 3)) - 1;
            let mask_uv = ((1u32 << (((rows + 1) >> 1) << 2)) - 1) as u16;
            for i in [t4, t8, t16] {
                self.left_y[i] &= mask_y;
                self.above_y[i] &= mask_y;
                self.left_uv[i] &= mask_uv;
                self.above_uv[i] &= mask_uv;
            }
            self.int_4x4_y &= mask_y;
            self.int_4x4_uv &= mask_uv;
            // No wide filter on the last chroma row.
            if rows == 1 {
                self.above_uv[t8] |= self.above_uv[t16];
                self.above_uv[t16] = 0;
            }
            if rows == 5 {
                self.above_uv[t8] |= self.above_uv[t16] & 0xff00;
                self.above_uv[t16] &= !0xff00;
            }
        }
        if mi_col + SB_MI > mi_cols {
            let columns = mi_cols - mi_col;
            let mask_y = ((1u64 << columns) - 1) * FIRST_COLUMN;
            let mask_uv = ((1u16 << ((columns + 1) >> 1)) - 1) * FIRST_COLUMN_UV;
            // Internal edges do not apply on the frame's last chroma column.
            let mask_uv_int = ((1u16 << (columns >> 1)) - 1) * FIRST_COLUMN_UV;
            for i in [t4, t8, t16] {
                self.left_y[i] &= mask_y;
                self.above_y[i] &= mask_y;
                self.left_uv[i] &= mask_uv;
                self.above_uv[i] &= mask_uv;
            }
            self.int_4x4_y &= mask_y;
            self.int_4x4_uv &= mask_uv_int;
            // No wide filter on the last chroma column.
            if columns == 1 {
                self.left_uv[t8] |= self.left_uv[t16];
                self.left_uv[t16] = 0;
            }
            if columns == 5 {
                self.left_uv[t8] |= self.left_uv[t16] & 0xcccc;
                self.left_uv[t16] &= !0xcccc;
            }
        }
        // No filter on the frame's left column.
        if mi_col == 0 {
            for i in [t4, t8, t16] {
                self.left_y[i] &= !FIRST_COLUMN;
                self.left_uv[i] &= !FIRST_COLUMN_UV;
            }
        }
        self
    }

    /// The luma vertical (`vertical`) or horizontal edges, indexed by line
    /// of edges: columns for vertical ones (transposed), rows otherwise.
    /// `top_row`: the superblock starts the frame (no top edges in row 0).
    fn luma(&self, vertical: bool, top_row: bool) -> EdgeMasks {
        if vertical {
            EdgeMasks {
                sixteen: transpose8(self.left_y[2]),
                eight: transpose8(self.left_y[1]),
                four: transpose8(self.left_y[0]),
                internal: transpose8(self.int_4x4_y),
            }
        } else {
            let keep = if top_row { !0xff } else { u64::MAX };
            EdgeMasks {
                sixteen: self.above_y[2] & keep,
                eight: self.above_y[1] & keep,
                four: self.above_y[0] & keep,
                internal: self.int_4x4_y,
            }
        }
    }

    /// The chroma counterpart of [`Self::luma`]; `last_row`: the chroma row
    /// (0..4) holding the frame's last unit row when it is odd, whose
    /// internal horizontal edges are not filtered.
    fn chroma(&self, vertical: bool, top_row: bool, last_row: Option<usize>) -> EdgeMasks {
        if vertical {
            EdgeMasks {
                sixteen: u64::from(transpose4(self.left_uv[2])),
                eight: u64::from(transpose4(self.left_uv[1])),
                four: u64::from(transpose4(self.left_uv[0])),
                internal: u64::from(transpose4(self.int_4x4_uv)),
            }
        } else {
            let keep: u16 = if top_row { !0xf } else { u16::MAX };
            let internal =
                last_row.map_or(self.int_4x4_uv, |row| self.int_4x4_uv & !(0xf << (row * 4)));
            EdgeMasks {
                sixteen: u64::from(self.above_uv[2] & keep),
                eight: u64::from(self.above_uv[1] & keep),
                four: u64::from(self.above_uv[0] & keep),
                internal: u64::from(internal),
            }
        }
    }
}

/// The edge masks of every superblock of the frame, built as blocks are
/// stored.
#[derive(Default)]
pub(super) struct Masks {
    mi_cols: usize,
    mi_rows: usize,
    sb_cols: usize,
    cells: Vec<SuperblockMask>,
}

impl Masks {
    /// These masks cleared for a frame of `mi_cols` x `mi_rows` units
    /// (`vp9_reset_lfm`).
    pub(super) fn reuse(mut self, mi_cols: usize, mi_rows: usize) -> Self {
        let sb_cols = mi_cols.div_ceil(SB_MI);
        let sb_rows = mi_rows.div_ceil(SB_MI);
        self.cells.clear();
        self.cells
            .resize(sb_cols * sb_rows, SuperblockMask::default());
        (self.mi_cols, self.mi_rows, self.sb_cols) = (mi_cols, mi_rows, sb_cols);
        self
    }

    /// Add the edges of a block at unit (`mi_row`, `mi_col`).
    pub(super) fn record(
        &mut self,
        (mi_row, mi_col): (usize, usize),
        block: &BlockInfo,
        levels: &Levels,
    ) {
        let sb = (mi_row / SB_MI) * self.sb_cols + mi_col / SB_MI;
        self.cells[sb].add_block((mi_row % SB_MI, mi_col % SB_MI), block, levels.of(block));
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

    /// Deblock the superblock row starting at 8x8 unit row `mi_row` in
    /// place; its blocks must all be decoded.
    pub(super) fn filter_row(
        &self,
        dsp: &nc_dsp::Dsp,
        recon: &mut Picture,
        masks: &Masks,
        mi_row: usize,
    ) {
        if self.limits.is_empty() {
            return;
        }
        let frame = (masks.mi_rows, masks.mi_cols);
        let sb_row = mi_row / SB_MI;
        let row_masks = &masks.cells[sb_row * masks.sb_cols..][..masks.sb_cols];
        let [y, u, v] = recon.planes_mut();
        let (mut y, mut u, mut v) = (y.view_mut(), u.view_mut(), v.view_mut());
        // `mi_row + r == mi_rows - 1` for an even r: the chroma row of an odd
        // last unit row.
        let last_chroma_row = (mi_row + SB_MI >= masks.mi_rows
            && (masks.mi_rows - mi_row) % 2 == 1)
            .then(|| (masks.mi_rows - 1 - mi_row) / 2);
        for (sb_col, mask) in row_masks.iter().enumerate() {
            let mi_col = sb_col * SB_MI;
            let mask = mask.adjusted((mi_row, mi_col), frame);
            let origin = (mi_col * MI_SIZE, mi_row * MI_SIZE);
            let top = mi_row == 0;
            let level = |index: usize, position: usize, vertical: bool, ss: usize| {
                let (row, col) = if vertical {
                    (position, index)
                } else {
                    (index, position)
                };
                mask.levels[(row << ss) * SB_MI + (col << ss)]
            };
            let ctx = FilterContext {
                dsp,
                limits: &self.limits,
                origin,
            };
            ctx.plane(&mut y, &mask.luma(true, top), true, 0, &level);
            ctx.plane(&mut y, &mask.luma(false, top), false, 0, &level);
            for plane in [&mut u, &mut v] {
                ctx.plane(
                    plane,
                    &mask.chroma(true, top, last_chroma_row),
                    true,
                    1,
                    &level,
                );
                ctx.plane(
                    plane,
                    &mask.chroma(false, top, last_chroma_row),
                    false,
                    1,
                    &level,
                );
            }
        }
    }
}

/// What filtering one superblock needs besides its masks.
struct FilterContext<'a> {
    dsp: &'a nc_dsp::Dsp,
    limits: &'a [Limits],
    /// The superblock's luma top-left sample.
    origin: (usize, usize),
}

impl FilterContext<'_> {
    /// Filter one direction of one plane of the superblock: lines of edges
    /// in order (columns left to right, or rows top to bottom). Within a
    /// line, units cover different lines across the edge, so each edge
    /// class can run as a whole: runs of units of one width and level are
    /// one kernel call each, and the internal 4x4 edges follow their units'
    /// outer edges, as in libvpx's per-row passes.
    fn plane(
        &self,
        view: &mut PlaneViewMut<'_>,
        edges: &EdgeMasks,
        vertical: bool,
        ss: usize,
        level: &impl Fn(usize, usize, bool, usize) -> u8,
    ) {
        let side = SB_MI >> ss;
        let line_mask = (1u64 << side) - 1;
        let (x0, y0) = (self.origin.0 >> ss, self.origin.1 >> ss);
        let classes = [
            (edges.sixteen, Width::Sixteen, 0),
            (edges.eight, Width::Eight, 0),
            (edges.four, Width::Four, 0),
            (edges.internal, Width::Four, INTERNAL_OFFSET),
        ];
        for index in 0..side {
            let shift = index * side;
            if (edges.any() >> shift) & line_mask == 0 {
                continue;
            }
            for (mask, width, inset) in classes {
                let mut bits = (mask >> shift) & line_mask;
                while bits != 0 {
                    let start = bits.trailing_zeros() as usize;
                    let unit_level = level(index, start, vertical, ss);
                    let mut end = start + 1;
                    while end < side
                        && (bits >> end) & 1 != 0
                        && level(index, end, vertical, ss) == unit_level
                    {
                        end += 1;
                    }
                    bits &= !((1u64 << end) - (1u64 << start));
                    let along = start * MI_SIZE;
                    let across = index * MI_SIZE + inset;
                    let (x, y) = if vertical {
                        (x0 + across, y0 + along)
                    } else {
                        (x0 + along, y0 + across)
                    };
                    let lines = (end - start) * MI_SIZE;
                    self.edge(view, (x, y), (vertical, lines), width, unit_level);
                }
            }
        }
    }

    /// Filter `lines` lines of the edge at plane sample `(x, y)`.
    fn edge(
        &self,
        view: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        (vertical, lines): (bool, usize),
        width: Width,
        level: u8,
    ) {
        let limits = self.limits[usize::from(level)];
        let stride = view.stride();
        let (step, along) = if vertical { (1, stride) } else { (stride, 1) };
        self.dsp.vp9_filter_edge(
            view.data_mut(),
            y * stride + x,
            step,
            along,
            lines,
            width,
            limits,
        );
    }
}

/// Internal 4x4 edges lie 4 samples into their unit.
const INTERNAL_OFFSET: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transposes_match_the_definition() {
        for seed in 0..64u64 {
            let x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (seed << 17);
            let mut expected = 0u64;
            for bit in 0..64 {
                if x >> bit & 1 != 0 {
                    expected |= 1 << ((bit % 8) * 8 + bit / 8);
                }
            }
            assert_eq!(transpose8(x), expected);
            let small = x as u16;
            let mut expected = 0u16;
            for bit in 0..16 {
                if small >> bit & 1 != 0 {
                    expected |= 1 << ((bit % 4) * 4 + bit / 4);
                }
            }
            assert_eq!(transpose4(small), expected);
        }
    }
}
