//! Decoding one frame's tiles: the tile layout, the partition tree, and
//! per-block parse-then-reconstruct (VP9 §6.4, §8.4; libvpx
//! `vp9_decodeframe.c` `decode_tiles`, `decode_partition`, `decode_block`,
//! BSD-3-Clause, see `THIRD_PARTY.md`).
//!
//! [`FrameDecoder`] owns everything one frame produces (reconstruction,
//! mode info, vectors for the next frame, the segment map, symbol counts)
//! and the above contexts. Mode parsing (`modes.rs`), reference vectors
//! (`mvref.rs`) and reconstruction (`recon.rs`) extend it.

use nc_common::decode::{DecodeError, Result};
use nc_common::{Picture, PictureRecycler};

use super::block::{BlockInfo, BlockSize, MvRef, Partition, REFS_PER_FRAME};
use super::fill::fill_run;
use super::header::{CompressedHeader, FrameHeader};
use super::loopfilter::{Deblocker, Levels, Masks};
use super::recon::{InterScratch, REFERENCE_BORDER, Reference};
use crate::bool_coder::BoolReader;
use crate::coeff::TokenCache;
use crate::probs::{Counts, FrameContext};
use crate::recon::ResidualBuffers;
use crate::segmentation::MAX_SEGMENTS;
use crate::tables;
use crate::transform::Steps;
use crate::tree;
use crate::types::MI_SIZE;

/// Mode-info units per superblock side, and its log2.
pub(super) const SB_MI: usize = 8;
const SB_MI_LOG2: u32 = 3;
/// 4x4 units per superblock side (left coefficient contexts).
const SB_4X4: usize = 16;
/// Bytes of a tile size marker (all tiles but the last).
const TILE_SIZE_BYTES: usize = 4;

/// The inputs one frame decodes from.
pub(super) struct FrameInputs<'a> {
    pub header: &'a FrameHeader,
    pub compressed: CompressedHeader,
    pub context: &'a FrameContext,
    /// LAST, GOLDEN and ALTREF (inter frames).
    pub references: [Option<Reference<'a>>; REFS_PER_FRAME],
    /// The previous frame's vectors (`use_prev_frame_mvs`).
    pub previous_mvs: Option<&'a [MvRef]>,
    /// `last_frame_seg_map`.
    pub last_segment_map: &'a [u8],
    pub dsp: &'a nc_dsp::Dsp,
}

/// One tile's column range and its left contexts.
pub(super) struct Tile {
    pub col_start: usize,
    pub col_end: usize,
    /// Left coefficient contexts per plane (4x4 units of one superblock).
    pub left_nonzero: [[u8; SB_4X4]; 3],
    pub left_partition: [u8; SB_MI],
}

/// The state and output of decoding one frame.
pub(super) struct FrameDecoder<'a> {
    pub inputs: FrameInputs<'a>,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub recon: Picture,
    /// Every decoded block; `grid` holds each 8x8 unit's block index.
    pub blocks: Vec<BlockInfo>,
    pub grid: Vec<u32>,
    pub mvs: Vec<MvRef>,
    pub segment_map: Vec<u8>,
    pub counts: Counts,
    /// Above coefficient contexts per plane (4x4 units, superblock-aligned).
    pub above_nonzero: [Vec<u8>; 3],
    pub above_partition: Vec<u8>,
    pub scratch: Box<BlockScratch>,
    /// Quantizer steps per segment, luma and chroma (libvpx `seg_dequant`):
    /// resolved once per frame instead of per transform block.
    pub steps: [[Steps; 2]; MAX_SEGMENTS],
    /// Loop-filter levels of the frame and each superblock's edge masks.
    pub filter_levels: Levels,
    pub filter_masks: Masks,
}

/// Working memory kept by the decoder across frames: allocating and
/// clearing it for every frame cost about 2 % of decoding (720p: the
/// inter-prediction window, the vector and block grids, the growing block
/// list), while no frame depends on its contents.
pub(super) struct BlockScratch {
    /// The block list, grid and vector buffers of the next frame. Every
    /// grid and vector cell is written by the block covering it before any
    /// read (neighbours and later frames read decoded blocks only), so a
    /// recycled buffer needs no clearing.
    pub blocks: Vec<BlockInfo>,
    pub grid: Vec<u32>,
    pub mvs: Vec<MvRef>,
    pub filter_masks: Masks,
    /// Dequantized coefficients of one transform block, zero between
    /// blocks (and so between frames).
    pub coefficients: [i32; 32 * 32],
    pub token_cache: TokenCache,
    pub buffers: Box<ResidualBuffers>,
    pub inter: InterScratch,
    /// One transform block's intra prediction (zeroing a fresh one per
    /// block cost 0.3 % of decoding).
    pub intra: [u8; 32 * 32],
}

impl std::fmt::Debug for BlockScratch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockScratch").finish_non_exhaustive()
    }
}

impl BlockScratch {
    pub(super) fn new(dsp: &nc_dsp::Dsp) -> Box<Self> {
        Box::new(Self {
            blocks: Vec::new(),
            grid: Vec::new(),
            mvs: Vec::new(),
            filter_masks: Masks::default(),
            coefficients: [0; 32 * 32],
            token_cache: TokenCache::new(),
            buffers: ResidualBuffers::new(dsp),
            inter: InterScratch::new(),
            intra: [0; 32 * 32],
        })
    }
}

/// The quantizer steps of every segment's luma and chroma.
fn segment_steps(header: &FrameHeader) -> [[Steps; 2]; MAX_SEGMENTS] {
    let quantizer = &header.quantizer;
    core::array::from_fn(|segment| {
        let q = header.segmentation.qindex(segment as u8, quantizer.base);
        [
            Steps::new(q, quantizer.y_dc_delta, 0),
            Steps::new(q, quantizer.uv_dc_delta, quantizer.uv_ac_delta),
        ]
    })
}

/// What a decoded frame leaves for later frames.
pub(super) struct FrameOutput {
    pub recon: Picture,
    pub mvs: Vec<MvRef>,
    pub segment_map: Vec<u8>,
    pub counts: Counts,
}

/// Unit index of a new grid cell (never read: neighbours are decoded).
const NO_BLOCK: u32 = u32::MAX;

impl<'a> FrameDecoder<'a> {
    /// A decoder for one frame, its reconstruction from `pictures` (a
    /// released picture is reused without clearing when its size fits).
    pub(super) fn new(
        inputs: FrameInputs<'a>,
        pictures: &mut PictureRecycler,
        mut scratch: Box<BlockScratch>,
    ) -> Result<Self> {
        let header = inputs.header;
        let mi_cols = header.width.div_ceil(MI_SIZE);
        let mi_rows = header.height.div_ceil(MI_SIZE);
        // Planes cover whole superblocks: blocks and transforms that pass
        // the frame edge write there (never read back).
        let sb_cols = mi_cols.div_ceil(SB_MI);
        let sb_rows = mi_rows.div_ceil(SB_MI);
        // Every sample a later step reads is written first: blocks cover
        // the 8-aligned frame, intra edges stop at it and the loop filter
        // stays inside it. A reference is extended beyond its visible size
        // once it is complete (`extend_reference`).
        let (width, height) = (sb_cols * 64, sb_rows * 64);
        let recon = pictures
            .take(width, height, REFERENCE_BORDER)
            .map_err(|_| DecodeError::Unsupported("VP9 frame size"))?;
        let units = sb_cols * SB_4X4;
        let mut blocks = std::mem::take(&mut scratch.blocks);
        blocks.clear();
        let mut grid = std::mem::take(&mut scratch.grid);
        grid.resize(mi_cols * mi_rows, NO_BLOCK);
        let filter_masks = std::mem::take(&mut scratch.filter_masks).reuse(mi_cols, mi_rows);
        let mut mvs = std::mem::take(&mut scratch.mvs);
        mvs.resize(mi_cols * mi_rows, MvRef::default());
        if header.is_intra() {
            // Intra frames store no vectors ([`Self::store_block`]).
            mvs.fill(MvRef::default());
        }
        Ok(Self {
            inputs,
            mi_cols,
            mi_rows,
            recon,
            blocks,
            grid,
            mvs,
            segment_map: vec![0; mi_cols * mi_rows],
            counts: Counts {
                high_precision: header.allow_hp,
                ..Counts::default()
            },
            above_nonzero: [vec![0; units], vec![0; units / 2], vec![0; units / 2]],
            above_partition: vec![0; sb_cols * SB_MI],
            scratch,
            steps: segment_steps(header),
            filter_levels: Levels::new(&header.loop_filter, &header.segmentation),
            filter_masks,
        })
    }

    /// The frame's output, and its working memory (block list, grids and
    /// deblocking units back in the scratch) for the next frame.
    pub(super) fn finish(self) -> (FrameOutput, Box<BlockScratch>) {
        let mut scratch = self.scratch;
        (scratch.blocks, scratch.grid) = (self.blocks, self.grid);
        scratch.filter_masks = self.filter_masks;
        let output = FrameOutput {
            recon: self.recon,
            mvs: self.mvs,
            segment_map: self.segment_map,
            counts: self.counts,
        };
        (output, scratch)
    }

    /// Decode every tile of `data` (the bytes after the compressed header).
    pub(super) fn decode_tiles(&mut self, data: &[u8]) -> Result<()> {
        let header = self.inputs.header;
        let (cols, rows) = (
            1usize << header.tile_cols_log2,
            1usize << header.tile_rows_log2,
        );
        let deblocker = Deblocker::new(&header.loop_filter);
        let mut tiles = Vec::with_capacity(cols);
        let mut rest = data;
        for tile_row in 0..rows {
            let (row_start, row_end) = tile_range(tile_row, self.mi_rows, header.tile_rows_log2);
            tiles.clear();
            for tile_col in 0..cols {
                let last = tile_row == rows - 1 && tile_col == cols - 1;
                let size = if last {
                    rest.len()
                } else {
                    let marker = rest.get(..TILE_SIZE_BYTES).ok_or(DecodeError::Truncated)?;
                    rest = &rest[TILE_SIZE_BYTES..];
                    u32::from_be_bytes(marker.try_into().expect("four bytes")) as usize
                };
                let tile_data = rest.get(..size).ok_or(DecodeError::Truncated)?;
                rest = &rest[size..];
                let (col_start, col_end) =
                    tile_range(tile_col, self.mi_cols, header.tile_cols_log2);
                let tile = Tile {
                    col_start,
                    col_end,
                    left_nonzero: [[0; SB_4X4]; 3],
                    left_partition: [0; SB_MI],
                };
                tiles.push((BoolReader::new(tile_data)?, tile));
            }
            // Superblock rows across the tile columns (independent within a
            // tile row), so that each row is complete before it is filtered.
            for mi_row in (row_start..row_end).step_by(SB_MI) {
                for (reader, tile) in &mut tiles {
                    tile.left_nonzero = [[0; SB_4X4]; 3];
                    tile.left_partition = [0; SB_MI];
                    for mi_col in (tile.col_start..tile.col_end).step_by(SB_MI) {
                        self.decode_partition(reader, tile, mi_row, mi_col, 3)?;
                    }
                }
                if mi_row >= SB_MI {
                    let above = mi_row - SB_MI;
                    deblocker.filter_row(
                        self.inputs.dsp,
                        &mut self.recon,
                        &self.filter_masks,
                        above,
                    );
                }
            }
            for (reader, _) in &tiles {
                reader.finish()?;
            }
        }
        let last_row = (self.mi_rows - 1) & !(SB_MI - 1);
        deblocker.filter_row(
            self.inputs.dsp,
            &mut self.recon,
            &self.filter_masks,
            last_row,
        );
        Ok(())
    }

    /// `decode_partition` of the square block of side `8 << log2_8x8` at
    /// `(mi_row, mi_col)`.
    fn decode_partition(
        &mut self,
        reader: &mut BoolReader<'_>,
        tile: &mut Tile,
        mi_row: usize,
        mi_col: usize,
        log2_8x8: u8,
    ) -> Result<()> {
        if mi_row >= self.mi_rows || mi_col >= self.mi_cols {
            return Ok(());
        }
        let bsize = BlockSize::square(log2_8x8);
        let side = 1usize << log2_8x8;
        let half = side >> 1;
        let has_rows = mi_row + half < self.mi_rows;
        let has_cols = mi_col + half < self.mi_cols;
        let partition = self.read_partition(
            reader,
            tile,
            (mi_row, mi_col),
            (has_rows, has_cols),
            log2_8x8,
        );
        let subsize = bsize.subsize(partition);
        if half == 0 {
            self.decode_block(reader, tile, mi_row, mi_col, subsize)?;
        } else {
            match partition {
                Partition::None => self.decode_block(reader, tile, mi_row, mi_col, subsize)?,
                Partition::Horizontal => {
                    self.decode_block(reader, tile, mi_row, mi_col, subsize)?;
                    if has_rows {
                        self.decode_block(reader, tile, mi_row + half, mi_col, subsize)?;
                    }
                }
                Partition::Vertical => {
                    self.decode_block(reader, tile, mi_row, mi_col, subsize)?;
                    if has_cols {
                        self.decode_block(reader, tile, mi_row, mi_col + half, subsize)?;
                    }
                }
                Partition::Split => {
                    for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
                        self.decode_partition(
                            reader,
                            tile,
                            mi_row + dr,
                            mi_col + dc,
                            log2_8x8 - 1,
                        )?;
                    }
                }
            }
        }
        // `dec_update_partition_context`: whole blocks set the context;
        // split blocks leave it to their children (8x8 always sets it).
        if half == 0 || partition != Partition::Split {
            let (above, left) = subsize.partition_context();
            let end = (mi_col + side).min(self.above_partition.len());
            fill_run(&mut self.above_partition[mi_col..end], above);
            let row = mi_row & (SB_MI - 1);
            fill_run(&mut tile.left_partition[row..(row + side).min(SB_MI)], left);
        }
        Ok(())
    }

    /// `read_partition`: blocks crossing the frame edge code a reduced
    /// choice (VP9 §9.3.1).
    fn read_partition(
        &mut self,
        reader: &mut BoolReader<'_>,
        tile: &Tile,
        (mi_row, mi_col): (usize, usize),
        (has_rows, has_cols): (bool, bool),
        log2_8x8: u8,
    ) -> Partition {
        let above = (self.above_partition[mi_col] >> log2_8x8) & 1;
        let left = (tile.left_partition[mi_row & (SB_MI - 1)] >> log2_8x8) & 1;
        let ctx = usize::from(left * 2 + above) + usize::from(log2_8x8) * 4;
        let probs = if self.inputs.header.is_intra() {
            &tables::KF_PARTITION[ctx]
        } else {
            &self.inputs.context.partition[ctx]
        };
        let partition = match (has_rows, has_cols) {
            (true, true) => Partition::from_index(tree::read(reader, tree::PARTITION, probs)),
            (false, true) => {
                if reader.read(probs[1]) {
                    Partition::Split
                } else {
                    Partition::Horizontal
                }
            }
            (true, false) => {
                if reader.read(probs[2]) {
                    Partition::Split
                } else {
                    Partition::Vertical
                }
            }
            (false, false) => Partition::Split,
        };
        self.counts.partition[ctx][partition as usize] += 1;
        partition
    }

    /// `decode_block`: mode info, prediction and residual of one block.
    fn decode_block(
        &mut self,
        reader: &mut BoolReader<'_>,
        tile: &mut Tile,
        mi_row: usize,
        mi_col: usize,
        bsize: BlockSize,
    ) -> Result<()> {
        let position = BlockPosition::new(self, tile, mi_row, mi_col, bsize);
        let mut info = self.read_mode_info(reader, &position)?;
        if info.skip {
            self.reset_skip_context(tile, &position);
        }
        let coded = if info.is_inter() {
            self.predict_inter_block(&info, &position)?;
            !info.skip && self.reconstruct_inter(reader, tile, &info, &position)
        } else {
            self.reconstruct_intra(reader, tile, &info, &position);
            true
        };
        // `decode_block`: an inter block of 8x8 or more without
        // coefficients counts as skipped from here on (contexts, filter).
        if info.is_inter() && !bsize.is_sub8x8() && !coded {
            info.skip = true;
        }
        self.store_block(info, &position);
        Ok(())
    }

    /// Record the block for its neighbours, the loop filter and the next
    /// frame's vectors.
    fn store_block(&mut self, info: BlockInfo, position: &BlockPosition) {
        let index = self.blocks.len() as u32;
        self.blocks.push(info);
        self.filter_masks.record(
            (position.mi_row, position.mi_col),
            &info,
            &self.filter_levels,
        );
        let mv = MvRef {
            ref_frame: info.ref_frame,
            mv: info.mv,
        };
        for row in position.mi_row..position.mi_row + position.y_mis {
            let start = row * self.mi_cols + position.mi_col;
            fill_run(&mut self.grid[start..start + position.x_mis], index);
            if !self.inputs.header.is_intra() {
                fill_run(&mut self.mvs[start..start + position.x_mis], mv);
            }
        }
    }

    /// `dec_reset_skip_context`: a skipped block clears the coefficient
    /// contexts it covers (beyond the frame edge too).
    fn reset_skip_context(&mut self, tile: &mut Tile, position: &BlockPosition) {
        for plane in 0..3 {
            let ss = usize::from(plane > 0);
            let (x, y) = position.plane_4x4(ss);
            let (w, h) = position.plane_size_4x4(ss);
            fill_run(&mut self.above_nonzero[plane][x..x + w], 0);
            let y = y & ((SB_4X4 >> ss) - 1);
            fill_run(&mut tile.left_nonzero[plane][y..y + h], 0);
        }
    }
}

/// `get_tile_offset`: tile `index` of `1 << log2` spans these 8x8 units.
fn tile_range(index: usize, mis: usize, log2: u32) -> (usize, usize) {
    let sbs = mis.div_ceil(SB_MI);
    let offset = |i: usize| (((i * sbs) >> log2) << SB_MI_LOG2).min(mis);
    (offset(index), offset(index + 1))
}

/// Where a block lies: position, size, frame edges and neighbours.
#[derive(Debug, Clone, Copy)]
pub(super) struct BlockPosition {
    pub mi_row: usize,
    pub mi_col: usize,
    pub size: BlockSize,
    /// 8x8 units inside the frame (`x_mis`, `y_mis`).
    pub x_mis: usize,
    pub y_mis: usize,
    /// Distances to the frame edges in 1/8 luma samples
    /// (`mb_to_left_edge` .. `mb_to_bottom_edge`).
    pub to_left: i32,
    pub to_right: i32,
    pub to_top: i32,
    pub to_bottom: i32,
    /// Block indices of the above and left neighbours (`above_mi`,
    /// `left_mi`; left stops at the tile's first column).
    pub above: Option<u32>,
    pub left: Option<u32>,
    /// The tile's columns (`is_inside`).
    pub tile_cols: (usize, usize),
    /// Per plane type (luma, 4:2:0 chroma), in 4x4 units: the block's
    /// top-left, its size and the part inside the frame. Resolved once per
    /// block; every transform block and prediction reads them.
    planes: [PlaneGeometry; 2],
}

/// A block's extent in one plane type, in 4x4 units.
#[derive(Debug, Clone, Copy)]
struct PlaneGeometry {
    origin: (usize, usize),
    size: (usize, usize),
    limits: (usize, usize),
}

impl BlockPosition {
    fn new(
        frame: &FrameDecoder<'_>,
        tile: &Tile,
        mi_row: usize,
        mi_col: usize,
        size: BlockSize,
    ) -> Self {
        let (bw, bh) = (size.width_mi(), size.height_mi());
        let eighths = |units: isize| (units * MI_SIZE as isize * 8) as i32;
        let cell = |row: usize, col: usize| frame.grid[row * frame.mi_cols + col];
        let to_right = eighths(frame.mi_cols as isize - bw as isize - mi_col as isize);
        let to_bottom = eighths(frame.mi_rows as isize - bh as isize - mi_row as isize);
        let planes = [0, 1].map(|ss| {
            let size = ((bw * 2) >> ss, (bh * 2) >> ss);
            // `max_blocks_wide` / `max_blocks_high`.
            let cut = |edge: i32, size: usize| {
                if edge >= 0 {
                    size
                } else {
                    (size as i32 + (edge >> (5 + ss))) as usize
                }
            };
            PlaneGeometry {
                origin: ((mi_col * 2) >> ss, (mi_row * 2) >> ss),
                size,
                limits: (cut(to_right, size.0), cut(to_bottom, size.1)),
            }
        });
        Self {
            mi_row,
            mi_col,
            size,
            x_mis: bw.min(frame.mi_cols - mi_col),
            y_mis: bh.min(frame.mi_rows - mi_row),
            to_left: -eighths(mi_col as isize),
            to_right,
            to_top: -eighths(mi_row as isize),
            to_bottom,
            above: (mi_row > 0).then(|| cell(mi_row - 1, mi_col)),
            left: (mi_col > tile.col_start).then(|| cell(mi_row, mi_col - 1)),
            tile_cols: (tile.col_start, tile.col_end),
            planes,
        }
    }

    /// Top-left in 4x4 units of a plane with subsampling `ss`.
    pub(super) fn plane_4x4(&self, ss: usize) -> (usize, usize) {
        self.planes[ss].origin
    }

    /// Block size in 4x4 units of the plane (`n4_w`, `n4_h`; sub-8x8
    /// blocks count as 8x8).
    pub(super) fn plane_size_4x4(&self, ss: usize) -> (usize, usize) {
        self.planes[ss].size
    }

    /// `max_blocks_wide` / `max_blocks_high`: 4x4 units of the plane
    /// inside the 8x8-aligned frame.
    pub(super) fn plane_limits_4x4(&self, ss: usize) -> (usize, usize) {
        self.planes[ss].limits
    }
}
