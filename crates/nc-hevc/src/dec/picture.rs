//! Per-picture state of the decoder: the CTB layout (tiles, scan orders,
//! §6.5.1), what parsing needs about blocks already decoded (§6.4
//! availability, context selection, MPM and QP prediction), what the
//! in-loop filters need (§8.7), and the motion that later pictures read
//! through TMVP (§8.5.3.2.8).
//!
//! Block information is kept on the 4x4 luma grid, the minimum transform
//! and prediction block granularity of the Main profile.

#![allow(
    clippy::many_single_char_names,
    reason = "H.265 sample and block variables keep their specification names (x, y, w, h, n)"
)]

use nc_common::decode::DecodeError;
use std::rc::Rc;
use std::sync::Arc;

use nc_common::{Picture, PictureRecycler};
use nc_encoder::Mv;

use super::loop_filter::Progress;
use super::params::{Pps, Sps, Tiles};
use super::sao::CtbSao;

/// log2 of the grid of collocated motion TMVP reads (16x16, §8.5.3.2.8).
pub(super) const TMVP_GRID_LOG2: u32 = 4;
/// log2 of the information grid (4x4 luma samples).
pub(super) const GRID_LOG2: u8 = 2;
/// Luma border of decoded pictures, enough for most motion vectors to
/// read in place (farther windows are clamped explicitly). A multiple of
/// the row alignment, so that no wider border is allocated and extended.
pub(super) const BORDER: usize = 32;

/// Motion of one prediction block (§8.5.3.2): per list a vector and a
/// reference index, −1 when the list is unused. Intra blocks use neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PuMotion {
    pub mv: [Mv; 2],
    pub ref_idx: [i8; 2],
}

impl PuMotion {
    /// No prediction from either list.
    pub const NONE: Self = Self {
        mv: [Mv { x: 0, y: 0 }; 2],
        ref_idx: [-1; 2],
    };

    /// `predFlagLX`.
    pub fn uses(&self, list: usize) -> bool {
        self.ref_idx[list] >= 0
    }
}

impl Default for PuMotion {
    fn default() -> Self {
        Self::NONE
    }
}

/// Where a 4x4 block's motion is: an index into the picture's table of
/// prediction-block motion ([`PictureState::pus`]). Storing the 10-byte
/// [`PuMotion`] in every 4x4 block made the motion field the largest write
/// stream of inter decoding (2.8 MB per 2460x1848 picture, 12 % of the
/// first-level write misses); a prediction block now writes its motion
/// once and a 4-byte index per 4x4 block. Indices exceed 16 bits: a 4K
/// picture can hold 259,200 8x4 prediction blocks.
pub(super) type PuIndex = u32;
/// The index of [`PuMotion::NONE`], the table's first entry.
const NO_MOTION: PuIndex = 0;

/// Flags of one 4x4 block.
pub(super) mod flag {
    /// `CuPredMode == MODE_INTRA`.
    pub const INTRA: u8 = 1;
    /// `cu_skip_flag`.
    pub const SKIP: u8 = 1 << 1;
    /// Deblocking and SAO leave the samples alone: `cu_transquant_bypass_flag`,
    /// or `pcm_flag` with `pcm_loop_filter_disabled_flag` (§8.7.2.5.7, §8.7.3).
    pub const NO_FILTER: u8 = 1 << 2;
    /// The luma transform block has non-zero coefficients (§8.7.2.4).
    pub const CODED: u8 = 1 << 3;
    /// A transform block edge runs along the left side (§8.7.2.3).
    pub const TU_EDGE_LEFT: u8 = 1 << 4;
    /// A transform block edge runs along the top side.
    pub const TU_EDGE_TOP: u8 = 1 << 5;
    /// A prediction block edge runs along the left side (§8.7.2.3).
    pub const PU_EDGE_LEFT: u8 = 1 << 6;
    /// A prediction block edge runs along the top side.
    pub const PU_EDGE_TOP: u8 = 1 << 7;
}

/// Either kind of edge along a block's left side.
const LEFT_EDGES: u8 = flag::TU_EDGE_LEFT | flag::PU_EDGE_LEFT;
/// Either kind of edge along a block's top side.
const TOP_EDGES: u8 = flag::TU_EDGE_TOP | flag::PU_EDGE_TOP;
/// Luma deblocking edges lie on the 8x8 grid (§8.7.2.3).
pub(super) const EDGE_GRID: usize = 8;

/// The luma deblocking edges of a picture as 8-sample pieces on the 8x8
/// grid, listed per CTB row as decoding marks them, so that the strength
/// derivation and the filters visit edges only instead of scanning every
/// grid position.
/// A piece is the index of its first q-side 4x4 block: for a vertical edge
/// the block in its even block row (the piece covers that row and the
/// next), for a horizontal edge the block in its even block column. Each
/// piece is listed once, when the first of its two segments is marked;
/// the other segment's flags say whether it is an edge too.
#[derive(Debug, Default)]
pub(super) struct EdgePieces {
    pub vertical: Vec<Vec<u32>>,
    pub horizontal: Vec<Vec<u32>>,
}

impl EdgePieces {
    /// Empty lists for `rows` CTB rows, reusing `spare` storage.
    fn new(spare: Option<Self>, rows: usize) -> Self {
        let mut pieces = spare.unwrap_or_default();
        for lists in [&mut pieces.vertical, &mut pieces.horizontal] {
            lists.resize_with(rows, Vec::new);
            lists.iter_mut().for_each(Vec::clear);
        }
        pieces
    }
}

/// Which block edges [`PictureState::mark_edges`] records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Edges {
    Transform,
    Prediction,
    /// A coding block's boundary: both.
    Both,
}

/// What is known about one 4x4 block.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct BlockInfo {
    /// The slice that decoded it, plus one (stale until decoded: see
    /// `PictureState::new`).
    pub slice: u16,
    /// `CtDepth`.
    pub depth: u8,
    /// [`flag`] bits.
    pub flags: u8,
    /// `IntraPredModeY` (DC for inter and PCM blocks, as MPM derivation
    /// treats them, §8.4.2).
    pub intra_mode: u8,
    /// `QpY` of its coding unit.
    pub qp: i8,
}

/// A reference picture as a slice saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RefPic {
    /// The DPB picture's unique id: pictures are told apart by identity
    /// (§8.7.2.4), and DPB slots are reused.
    pub id: u64,
    pub poc: i32,
    pub long_term: bool,
}

/// What a slice contributes to its picture's filtering and to TMVP of
/// later pictures.
#[derive(Debug, Clone)]
pub(super) struct SliceInfo {
    /// `SliceAddrRs`: the first CTB of the slice in raster order.
    pub address: usize,
    pub deblock_disabled: bool,
    pub beta_offset_div2: i32,
    pub tc_offset_div2: i32,
    pub loop_filter_across_slices: bool,
    /// `RefPicList0` / `RefPicList1`.
    pub lists: [Vec<RefPic>; 2],
}

/// CTB addressing of a picture (§6.5.1): raster/tile scan conversion and
/// tile membership.
#[derive(Debug, Clone)]
pub(super) struct Layout {
    pub log2_ctb: u8,
    pub ctb_w: usize,
    pub ctb_h: usize,
    pub rs_to_ts: Vec<usize>,
    pub ts_to_rs: Vec<usize>,
    /// `TileId` by raster address.
    pub tile_of: Vec<u16>,
    /// More than one tile.
    pub tiled: bool,
    /// z-scan order of the 4x4 blocks inside a CTB (`MinTbAddrZs`
    /// relative to the CTB, §6.5.2), by raster position in the CTB.
    zscan: Vec<u16>,
}

impl Layout {
    pub fn new(sps: &Sps, pps: &Pps) -> Result<Self, DecodeError> {
        let (ctb_w, ctb_h) = sps.ctbs();
        let (columns, rows) = match &pps.tiles {
            Some(t) => tile_grid(t, ctb_w, ctb_h)?,
            None => (vec![0, ctb_w], vec![0, ctb_h]),
        };
        let count = ctb_w * ctb_h;
        let mut ts_to_rs = Vec::with_capacity(count);
        let mut tile_of = vec![0u16; count];
        let mut tile = 0u16;
        for r in rows.windows(2) {
            for c in columns.windows(2) {
                for y in r[0]..r[1] {
                    for x in c[0]..c[1] {
                        ts_to_rs.push(y * ctb_w + x);
                        tile_of[y * ctb_w + x] = tile;
                    }
                }
                tile += 1;
            }
        }
        let mut rs_to_ts = vec![0; count];
        for (ts, &rs) in ts_to_rs.iter().enumerate() {
            rs_to_ts[rs] = ts;
        }
        let side = 1usize << (sps.log2_ctb - GRID_LOG2);
        let mut zscan = vec![0u16; side * side];
        for (i, z) in zscan.iter_mut().enumerate() {
            let (x, y) = (i % side, i / side);
            // Interleave the bits of x and y (y above x).
            let mut v = 0;
            for bit in 0..8 {
                v |= ((x >> bit) & 1) << (2 * bit);
                v |= ((y >> bit) & 1) << (2 * bit + 1);
            }
            *z = v as u16;
        }
        Ok(Self {
            log2_ctb: sps.log2_ctb,
            ctb_w,
            ctb_h,
            rs_to_ts,
            ts_to_rs,
            tiled: tile > 1,
            tile_of,
            zscan,
        })
    }

    /// Raster address of the CTB containing luma `(x, y)`.
    pub fn ctb_of(&self, x: usize, y: usize) -> usize {
        (y >> self.log2_ctb) * self.ctb_w + (x >> self.log2_ctb)
    }

    /// z-order of the 4x4 block at luma `(x, y)` inside its CTB
    /// (`MinTbAddrZs` refined to 4x4 blocks, which orders blocks the same
    /// way, §6.5.2).
    #[inline]
    pub fn z_in_ctb(&self, x: usize, y: usize) -> u16 {
        let mask = (1 << self.log2_ctb) - 1;
        let shift = self.log2_ctb - GRID_LOG2;
        self.zscan[(((y & mask) >> GRID_LOG2) << shift) | ((x & mask) >> GRID_LOG2)]
    }
}

/// Tile column and row boundaries in CTBs.
fn tile_grid(
    t: &Tiles,
    ctb_w: usize,
    ctb_h: usize,
) -> Result<(Vec<usize>, Vec<usize>), DecodeError> {
    if t.columns > ctb_w || t.rows > ctb_h {
        return Err(DecodeError::Malformed("more tiles than CTBs"));
    }
    Ok((
        Tiles::boundaries(t.columns, t.widths.as_deref(), ctb_w)?,
        Tiles::boundaries(t.rows, t.heights.as_deref(), ctb_h)?,
    ))
}

/// Everything decoded into one picture before it enters the DPB.
#[derive(Debug)]
pub(super) struct PictureState {
    pub samples: Picture,
    pub width: usize,
    pub height: usize,
    /// Shared by the pictures of one SPS/PPS pair.
    pub layout: Rc<Layout>,
    /// Width of the 4x4 grid.
    pub stride: usize,
    pub info: Vec<BlockInfo>,
    /// Each 4x4 block's index into `pus` (stale, like `info`, until the
    /// block is decoded).
    pub motion: Vec<PuIndex>,
    /// The motion of every prediction block decoded so far, after
    /// [`PuMotion::NONE`].
    pub pus: Vec<PuMotion>,
    pub slices: Vec<SliceInfo>,
    /// SAO parameters of every CTB, raster order.
    pub sao: Vec<CtbSao>,
    /// How far decoding and in-loop filtering have come.
    pub progress: Progress,
    /// Deblocking-grid edge pieces of each CTB row.
    pub edges: EdgePieces,
}

/// Buffers of pictures that left the DPB, reused by new pictures so that
/// decoding a stream does not allocate (and clear) memory per picture.
/// Sample buffers are not cleared: decoding writes every sample.
#[derive(Debug, Default)]
pub(super) struct Pool {
    pictures: PictureRecycler,
    /// Pictures that left the DPB while output still held them.
    shared: Vec<Arc<Picture>>,
    /// TMVP motion fields of pictures that left the DPB.
    fields: Vec<Vec<PuMotion>>,
    /// The per-picture index map and motion table.
    motion: Option<Vec<PuIndex>>,
    pus: Option<Vec<PuMotion>>,
    slice_of: Vec<Vec<u16>>,
    info: Vec<Vec<BlockInfo>>,
    sao: Vec<Vec<CtbSao>>,
    edges: Option<EdgePieces>,
}

/// Spare motion fields kept at most (one per picture in flight).
const SPARE_FIELDS: usize = 4;

/// A vector of `len` copies of `value`, reusing `spare` storage.
fn refill<T: Clone>(spare: Option<Vec<T>>, len: usize, value: T) -> Vec<T> {
    let mut v = spare.unwrap_or_default();
    v.clear();
    v.resize(len, value);
    v
}

/// A vector of `len` elements reusing `spare` storage, whose elements are
/// stale: for state that decoding writes for every block before anything
/// reads it, which saves clearing it per picture.
fn reuse<T: Clone>(spare: Option<Vec<T>>, len: usize, value: T) -> Vec<T> {
    let mut v = spare.unwrap_or_default();
    v.resize(len, value);
    v
}

impl Pool {
    /// A `width` x `height` picture with the decoder's border, contents
    /// unspecified.
    pub fn picture(&mut self, width: usize, height: usize) -> Result<Picture, DecodeError> {
        // Pictures the caller has since released from output.
        let mut i = 0;
        while i < self.shared.len() {
            if Arc::strong_count(&self.shared[i]) == 1 {
                if let Ok(picture) = Arc::try_unwrap(self.shared.swap_remove(i)) {
                    self.pictures.give(picture);
                }
            } else {
                i += 1;
            }
        }
        Ok(self.pictures.take(width, height, BORDER)?)
    }

    /// Take back the buffers of a picture leaving the DPB; samples still
    /// shared with output are reused once released.
    pub fn recycle(&mut self, samples: Arc<Picture>, motion: Motion) {
        match Arc::try_unwrap(samples) {
            Ok(picture) => self.pictures.give(picture),
            Err(shared) => self.shared.push(shared),
        }
        // A few spare motion fields cover the pictures in flight.
        if self.fields.len() < SPARE_FIELDS {
            self.fields.push(motion.field);
            self.slice_of.push(motion.slice_of);
        }
    }
}

impl PictureState {
    pub fn new(sps: &Sps, layout: Rc<Layout>, pool: &mut Pool) -> Result<Self, DecodeError> {
        let stride = sps.width >> GRID_LOG2;
        let blocks = stride * (sps.height >> GRID_LOG2);
        let ctbs = layout.ctb_w * layout.ctb_h;
        let progress = Progress::new(layout.ctb_w, layout.ctb_h);
        let rows = layout.ctb_h;
        Ok(Self {
            samples: pool.picture(sps.width, sps.height)?,
            width: sps.width,
            height: sps.height,
            layout,
            stride,
            // Every coding unit assigns its blocks' information and motion
            // (intra ones too) before anything reads them: neighbours are
            // read only once availability (§6.4.1) admits them, which
            // takes decoded blocks only, and the in-loop filters and TMVP
            // run on complete pictures.
            info: reuse(pool.info.pop(), blocks, BlockInfo::default()),
            motion: reuse(pool.motion.take(), blocks, NO_MOTION),
            pus: refill(pool.pus.take(), 1, PuMotion::NONE),
            slices: Vec::new(),
            sao: refill(pool.sao.pop(), ctbs, CtbSao::default()),
            progress,
            edges: EdgePieces::new(pool.edges.take(), rows),
        })
    }

    /// The samples, and everything else read-only (the in-loop filters
    /// read block state while writing samples).
    pub fn split(&mut self) -> (&mut Picture, Blocks<'_>) {
        (
            &mut self.samples,
            Blocks {
                info: &self.info,
                motion: &self.motion,
                pus: &self.pus,
                slices: &self.slices,
                layout: &self.layout,
                sao: &self.sao,
                edges: &self.edges,
                stride: self.stride,
                width: self.width,
                height: self.height,
            },
        )
    }

    /// Index of the 4x4 block at luma `(x, y)`.
    #[inline]
    pub fn block(&self, x: usize, y: usize) -> usize {
        (y >> GRID_LOG2) * self.stride + (x >> GRID_LOG2)
    }

    /// The availability process of §6.4.1 for the block at `(xn, yn)` seen
    /// from `(x, y)` by slice `slice` (index + 1).
    #[inline]
    pub fn available(&self, (x, y): (usize, usize), (xn, yn): (isize, isize), slice: u16) -> bool {
        if xn < 0 || yn < 0 || xn as usize >= self.width || yn as usize >= self.height {
            return false;
        }
        let (xn, yn) = (xn as usize, yn as usize);
        if self.info[self.block(xn, yn)].slice != slice {
            return false;
        }
        if xn <= x && yn <= y && !self.layout.tiled {
            // Left, above or above-left: earlier in z-order within a CTB
            // and in raster order across CTBs (one tile).
            return true;
        }
        let layout = &self.layout;
        let l = layout.log2_ctb;
        let (own, other) = ((x >> l, y >> l), (xn >> l, yn >> l));
        if own == other {
            // Same CTB, so same tile: decoding order inside it.
            return layout.z_in_ctb(xn, yn) < layout.z_in_ctb(x, y);
        }
        if !layout.tiled {
            // One tile: tile scan is raster scan.
            return (other.1, other.0) < (own.1, own.0);
        }
        let (own, other) = (layout.ctb_of(x, y), layout.ctb_of(xn, yn));
        layout.tile_of[other] == layout.tile_of[own]
            && layout.rs_to_ts[other] < layout.rs_to_ts[own]
    }

    /// Set `f` on every 4x4 block of the `w`x`h` area at `(x, y)`.
    pub fn fill(
        &mut self,
        (x, y): (usize, usize),
        (w, h): (usize, usize),
        mut f: impl FnMut(&mut BlockInfo),
    ) {
        for by in (y >> GRID_LOG2)..((y + h) >> GRID_LOG2) {
            let row = by * self.stride;
            for b in &mut self.info[row + (x >> GRID_LOG2)..row + ((x + w) >> GRID_LOG2)] {
                f(b);
            }
        }
    }

    /// Mark a transform (`tu`) or prediction block's left and top edges,
    /// and list the deblocking-grid pieces they start (see [`EdgePieces`]).
    pub fn mark_edges(&mut self, (x, y): (usize, usize), (w, h): (usize, usize), kind: Edges) {
        let (left, top) = match kind {
            Edges::Transform => (flag::TU_EDGE_LEFT, flag::TU_EDGE_TOP),
            Edges::Prediction => (flag::PU_EDGE_LEFT, flag::PU_EDGE_TOP),
            Edges::Both => (
                flag::TU_EDGE_LEFT | flag::PU_EDGE_LEFT,
                flag::TU_EDGE_TOP | flag::PU_EDGE_TOP,
            ),
        };
        let (bx, by) = (x >> GRID_LOG2, y >> GRID_LOG2);
        let first = by * self.stride + bx;
        // A side whose first block already carries these flags lies on
        // its coding block's boundary, which `Edges::Both` marked (and
        // listed) whole before any of the block's prediction or transform
        // blocks: nothing to add.
        if self.info[first].flags & left != left {
            // The picture's own left boundary is not filtered.
            let listed = x > 0 && x.is_multiple_of(EDGE_GRID);
            self.mark_left((bx, by), h >> GRID_LOG2, left, listed);
        }
        if self.info[first].flags & top != top {
            let listed = y > 0 && y.is_multiple_of(EDGE_GRID);
            self.mark_top(first, w >> GRID_LOG2, top, listed);
        }
    }

    /// Set `bits` on the left side of `rows` blocks down from block
    /// `(bx, by)`; with `listed`, list the vertical pieces none of whose
    /// blocks had a left edge yet. A piece's two blocks lie in one coding
    /// block (pieces are 8-aligned, coding blocks at least 8x8), so the one
    /// outside `rows` is decoded state, not stale.
    fn mark_left(&mut self, (bx, by): (usize, usize), rows: usize, bits: u8, listed: bool) {
        let stride = self.stride;
        if listed {
            let list = &mut self.edges.vertical[by >> (self.layout.log2_ctb - GRID_LOG2)];
            for piece_row in ((by & !1)..by + rows).step_by(2) {
                let i = piece_row * stride + bx;
                if (self.info[i].flags | self.info[i + stride].flags) & LEFT_EDGES == 0 {
                    list.push(i as u32);
                }
            }
        }
        for row in by..by + rows {
            self.info[row * stride + bx].flags |= bits;
        }
    }

    /// Set `bits` on the top side of `count` blocks from block `first`;
    /// with `listed`, list the horizontal pieces none of whose blocks had a
    /// top edge yet (as [`Self::mark_left`]).
    fn mark_top(&mut self, first: usize, count: usize, bits: u8, listed: bool) {
        if listed {
            let row = first / self.stride;
            let list = &mut self.edges.horizontal[row >> (self.layout.log2_ctb - GRID_LOG2)];
            // Pieces start on even block columns (the grid width is even).
            for i in ((first & !1)..first + count).step_by(2) {
                if (self.info[i].flags | self.info[i + 1].flags) & TOP_EDGES == 0 {
                    list.push(i as u32);
                }
            }
        }
        for b in &mut self.info[first..first + count] {
            b.flags |= bits;
        }
    }

    /// Motion of the 4x4 block with index `i` (decoded, inter).
    #[inline]
    pub fn motion_at(&self, i: usize) -> PuMotion {
        motion_of(&self.motion, &self.pus, i)
    }

    /// Record the intra coding block of side `size` at `(x, y)` in the
    /// motion field: as no motion, but only where TMVP of later pictures
    /// reads it, the 4x4 blocks on the 16x16 grid (§8.5.3.2.8 rounds the
    /// collocated position to it). This picture's own readers of the field
    /// (merge and AMVP candidates, deblocking strengths) test
    /// [`flag::INTRA`] first, so writing every block would be wasted work.
    pub fn set_intra_motion(&mut self, (x, y): (usize, usize), size: usize) {
        let grid = 1 << TMVP_GRID_LOG2;
        for yy in (y.next_multiple_of(grid)..y + size).step_by(grid) {
            for xx in (x.next_multiple_of(grid)..x + size).step_by(grid) {
                let i = self.block(xx, yy);
                self.motion[i] = NO_MOTION;
            }
        }
    }

    /// Store `motion` for the `w`x`h` prediction block at `(x, y)`: once
    /// in the table, and its index in every 4x4 block.
    pub fn set_motion(&mut self, (x, y): (usize, usize), (w, h): (usize, usize), motion: PuMotion) {
        let index = PuIndex::try_from(self.pus.len()).expect("fewer than 2^32 prediction blocks");
        self.pus.push(motion);
        for by in (y >> GRID_LOG2)..((y + h) >> GRID_LOG2) {
            let row = by * self.stride;
            self.motion[row + (x >> GRID_LOG2)..row + ((x + w) >> GRID_LOG2)].fill(index);
        }
    }

    /// The decoded picture's motion, as TMVP of later pictures reads it;
    /// the parsing state goes back to `pool`.
    pub fn into_motion(self, pool: &mut Pool) -> (Picture, Motion) {
        // TMVP reads the 4x4 block at the top left of each 16x16 grid
        // cell: sample those, so the DPB keeps one entry per cell.
        let step = 1 << (TMVP_GRID_LOG2 - u32::from(GRID_LOG2));
        let rows = self.motion.len() / self.stride;
        let (grid_w, grid_h) = (self.stride.div_ceil(step), rows.div_ceil(step));
        let cells = || {
            (0..grid_h).flat_map(move |gy| {
                (0..grid_w).map(move |gx| (gy * step) * self.stride + gx * step)
            })
        };
        let mut field = pool.fields.pop().unwrap_or_default();
        field.clear();
        field.extend(cells().map(|i| motion_of(&self.motion, &self.pus, i)));
        let mut slice_of = pool.slice_of.pop().unwrap_or_default();
        slice_of.clear();
        if self.slices.len() > 1 {
            slice_of.extend(cells().map(|i| self.info[i].slice.saturating_sub(1)));
        }
        pool.info.push(self.info);
        pool.sao.push(self.sao);
        pool.edges = Some(self.edges);
        pool.motion = Some(self.motion);
        pool.pus = Some(self.pus);
        let lists = self.slices.into_iter().map(|s| s.lists).collect();
        (
            self.samples,
            Motion {
                stride: grid_w,
                field,
                slice_of,
                lists,
            },
        )
    }
}

/// Read-only block state of a picture (see [`PictureState::split`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct Blocks<'a> {
    pub info: &'a [BlockInfo],
    pub motion: &'a [PuIndex],
    pub pus: &'a [PuMotion],
    pub slices: &'a [SliceInfo],
    pub layout: &'a Layout,
    pub sao: &'a [CtbSao],
    pub edges: &'a EdgePieces,
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

impl Blocks<'_> {
    /// Index of the 4x4 block at luma `(x, y)`.
    #[inline]
    pub fn block(&self, x: usize, y: usize) -> usize {
        (y >> GRID_LOG2) * self.stride + (x >> GRID_LOG2)
    }

    /// Motion of the 4x4 block with index `i` (decoded, inter).
    #[inline]
    pub fn motion_at(&self, i: usize) -> PuMotion {
        motion_of(self.motion, self.pus, i)
    }
}

/// The motion of 4x4 block `i` through its table index. An index never
/// written in this picture (only possible after a decoding error) reads
/// as no motion instead of failing.
#[inline]
fn motion_of(motion: &[PuIndex], pus: &[PuMotion], i: usize) -> PuMotion {
    pus.get(motion[i] as usize)
        .copied()
        .unwrap_or(PuMotion::NONE)
}

/// What TMVP needs from a decoded picture (§8.5.3.2.8, §8.5.3.2.9): the
/// motion of the 4x4 block at the top left of every 16x16 cell, the only
/// ones it reads.
#[derive(Debug, Default)]
pub(super) struct Motion {
    /// Width of the 16x16 grid.
    pub stride: usize,
    pub field: Vec<PuMotion>,
    /// Slice index of each cell's block (to find its reference lists);
    /// empty for a single-slice picture.
    pub slice_of: Vec<u16>,
    /// Each slice's `RefPicList0` / `RefPicList1`.
    pub lists: Vec<[Vec<RefPic>; 2]>,
}

impl Motion {
    /// Motion and reference pictures of the cell's block at luma `(x, y)`
    /// (a multiple of 16 in both coordinates, §8.5.3.2.8).
    pub fn at(&self, x: usize, y: usize) -> Option<(PuMotion, &[Vec<RefPic>; 2])> {
        let i = (y >> TMVP_GRID_LOG2) * self.stride + (x >> TMVP_GRID_LOG2);
        let motion = *self.field.get(i)?;
        // A single-slice picture keeps no per-block slice index.
        let slice = self.slice_of.get(i).copied().unwrap_or(0);
        let lists = self.lists.get(usize::from(slice))?;
        Some((motion, lists))
    }
}
