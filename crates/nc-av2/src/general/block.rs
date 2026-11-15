//! The general block layer: a tile's superblocks, their partition trees
//! ([`partition`]) and each coding block's syntax ([`syntax`]), for every
//! AV2 block size and tree arrangement (dav2d `decode.c`:
//! `dav2d_decode_tile_sbrow`, `decode_sb`, `decode_b`).
//!
//! Decoder-only: the encoder's block layer (`crate::block_state`,
//! `crate::partition`) is untouched. The leaves are shared: the entropy
//! reader, [`crate::entropy::Cdf`] (models in [`super::cdfs`]), and in time
//! the coefficient reader, transforms and kernels.
//!
//! Each tool group reached from a block (intra, inter, residual, loop
//! filters, segmentation and tiles) answers
//! [`nc_common::decode::DecodeError::Unsupported`] naming itself until it
//! is built; docs/handoff/av2.md "The general block layer" lists the
//! interfaces each group plugs into.
/// Emit a block-trace line at a depth (dav2d indents by depth) when the
/// tile is traced. A macro so the arguments may borrow the tile's fields
/// beside its trace sink.
macro_rules! trace {
    ($tile:expr, $depth:expr, $($arguments:tt)*) => {
        if let Some(trace) = &mut $tile.trace {
            trace(format_args!("{:depth$}{}", "", format_args!($($arguments)*), depth = $depth));
        }
    };
}

pub mod context;
mod partition;
pub mod record;
pub mod size;
mod syntax;

use std::fmt;

use super::cdfs::Cdfs;
use super::frame_header::{FrameHeader, FrameType};
use super::sequence::SequenceHeader;
use crate::entropy::{Cdf, Reader};
use context::Contexts;
use nc_common::decode::{DecodeError, Result};
use size::BlockSize;

/// Receives the block layer's trace lines, in dav2d's `DEBUG_BLOCK_INFO`
/// format (one call per line, without the newline).
pub type BlockTrace<'t> = &'t mut dyn FnMut(fmt::Arguments<'_>);

/// Luma samples per 4x4 unit, as a shift.
const UNIT_LOG2: u32 = 2;
/// 4x4 units per 64x64 block, as a shift (filter blocks and the CDEF
/// index map are counted in 64x64 blocks).
const UNITS64_LOG2: u32 = 4;

/// What the block layer needs of the frame, in 4x4 units.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    /// Width and height in 4x4 units, rounded up to 8x8 (dav2d `bw`,
    /// `bh`).
    columns: u32,
    rows: u32,
    /// The superblock and its side in units.
    root: BlockSize,
    superblock_units: u32,
}

/// The tile's extent in 4x4 units.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    column_start: u32,
    column_end: u32,
    row_start: u32,
    row_end: u32,
}

/// The state of one tile being decoded (dav2d `Dav2dTaskContext` with its
/// `Dav2dTileState`): the symbol reader and models, the contexts, and
/// where the current block is.
struct Tile<'a, 't> {
    frame: &'a FrameHeader,
    sequence: &'a SequenceHeader,
    reader: Reader<'a>,
    cdfs: Box<Cdfs>,
    /// The models adapt (`disable_cdf_update` 0).
    adapt: bool,
    contexts: Contexts,
    geometry: Geometry,
    bounds: Bounds,
    /// The current block's top-left luma position and its chroma block's.
    row: u32,
    column: u32,
    chroma_row: u32,
    chroma_column: u32,
    /// Inside an intra-only region of an inter frame (separate trees).
    intra_region: bool,
    /// The chroma tree of a 64x64 block split differently from luma, so
    /// chroma from luma may not use it (dav2d `sdp_cfl_disallowed`).
    #[allow(dead_code, reason = "the intra tool group reads it")]
    sdp_cfl_disallowed: bool,
    /// The running superblock quantizer (dav2d `last_qidx`).
    qindex: u16,
    /// Per 64x64 filter block of the frame, the CDEF strength set read
    /// (`None` until a block reads it; for the loop-filter group).
    cdef_index: Vec<Option<u8>>,
    /// Per 64x64 filter block and plane, the CCSO flag read; and the GDF
    /// flag (for the loop-filter group).
    ccso_flags: Vec<[bool; 3]>,
    gdf_flags: Vec<bool>,
    trace: Option<BlockTrace<'t>>,
}

/// Decode the one tile of a frame through the general block layer.
/// # Errors
/// Names the first tool the layer does not have yet; malformed and
/// truncated data.
pub fn decode_tile(
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    cdfs: Box<Cdfs>,
    data: &[u8],
    trace: Option<BlockTrace<'_>>,
) -> Result<()> {
    if frame.tiles.columns() * frame.tiles.rows() != 1 {
        return Err(DecodeError::Unsupported("AV2 multiple tiles"));
    }
    let units = |size: u32| ((size + 7) >> 3) << 1;
    let root = BlockSize::superblock(frame.superblock_log2);
    let geometry = Geometry {
        columns: units(frame.width),
        rows: units(frame.height),
        root,
        superblock_units: root.units().0,
    };
    let blocks64 = |count: u32| (count as usize).div_ceil(1 << UNITS64_LOG2);
    let filter_blocks = blocks64(geometry.columns) * blocks64(geometry.rows);
    let size = data.len();
    let mut tile = Tile {
        frame,
        sequence,
        reader: Reader::new(data)?,
        cdfs,
        adapt: !frame.disable_cdf_update,
        contexts: Contexts::new(geometry.columns as usize),
        geometry,
        bounds: Bounds {
            column_start: 0,
            column_end: geometry.columns,
            row_start: 0,
            row_end: geometry.rows,
        },
        row: 0,
        column: 0,
        chroma_row: 0,
        chroma_column: 0,
        intra_region: false,
        sdp_cfl_disallowed: false,
        qindex: frame.quantizer.base,
        cdef_index: vec![None; filter_blocks],
        ccso_flags: vec![[false; 3]; filter_blocks],
        gdf_flags: vec![false; filter_blocks],
        trace,
    };
    tile.decode(size)
}

impl Tile<'_, '_> {
    /// Every superblock of the tile in raster order.
    fn decode(&mut self, size: usize) -> Result<()> {
        let range = self.reader.range();
        trace!(
            self,
            0,
            "decode_tile[tilerow=0/col=0,y={}-{},x={}-{},size={size}]: r={range}",
            self.bounds.row_start,
            self.bounds.row_end,
            self.bounds.column_start,
            self.bounds.column_end
        );
        if self
            .frame
            .restoration
            .planes
            .iter()
            .any(|plane| plane.kind != 0)
        {
            return Err(DecodeError::Unsupported("AV2 loop restoration units"));
        }
        let intra_frame = self.frame.is_intra();
        self.contexts.above.reset(intra_frame);
        let step = self.geometry.superblock_units;
        let mut row = self.bounds.row_start;
        while row < self.bounds.row_end {
            self.contexts.left.reset(intra_frame);
            let mut column = self.bounds.column_start;
            while column < self.bounds.column_end {
                (self.row, self.column) = (row, column);
                self.sdp_cfl_disallowed = false;
                let root = self.geometry.root;
                let mut direction = 0;
                self.superblock(1, Some(root), Some(root), &mut direction)?;
                column += step;
            }
            row += step;
        }
        Ok(())
    }

    /// Read a binary symbol with the model `select` picks.
    fn bool(&mut self, select: impl FnOnce(&mut Cdfs) -> &mut Cdf<2>) -> Result<bool> {
        self.symbol(select).map(|symbol| symbol != 0)
    }

    /// Read a symbol with the model `select` picks: adapting it, or with
    /// its fixed probabilities when the frame does not adapt.
    fn symbol<const N: usize>(
        &mut self,
        select: impl FnOnce(&mut Cdfs) -> &mut Cdf<N>,
    ) -> Result<usize> {
        let cdf = select(&mut self.cdfs);
        if self.adapt {
            self.reader.symbol(cdf)
        } else {
            self.reader.fixed(cdf.inverse())
        }
    }

    /// One equiprobable bit.
    fn bypass(&mut self) -> Result<bool> {
        self.reader.literal(1).map(|bit| bit != 0)
    }

    /// Whether the frame is key or intra-only.
    fn intra_frame(&self) -> bool {
        matches!(self.frame.frame_type, FrameType::Key | FrameType::Intra)
    }

    /// The index of the 64x64 filter block holding unit (`row`, `column`).
    fn filter_block(&self, row: u32, column: u32) -> usize {
        let stride = (self.geometry.columns as usize).div_ceil(1 << UNITS64_LOG2);
        (row >> UNITS64_LOG2) as usize * stride + (column >> UNITS64_LOG2) as usize
    }
}

/// Luma samples of `units` 4x4 units.
fn samples(units: u32) -> u32 {
    units << UNIT_LOG2
}
