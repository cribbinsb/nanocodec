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

mod coefficients;
pub mod context;
mod intra;
mod intrabc;
mod loop_filters;
pub mod motion;
mod palette;
mod partition;
pub mod pixels;
mod quantizer_matrices;
mod reconstruction;
pub mod record;
mod restoration;
pub mod size;
mod syntax;
pub mod transform;

use std::fmt;

use std::rc::Rc;

use super::cdfs::Cdfs;
use super::filters::{FilterMaps, UNITS64_LOG2};
use super::frame_header::{FrameHeader, FrameType};
use super::sequence::SequenceHeader;
use crate::entropy::{Cdf, Reader};
use coefficients::{LEVELS, Scans};
use context::Contexts;
use intra::LUMA_MAP_SIDE;
use motion::{MotionStore, StoreGeometry};
use nc_common::decode::{DecodeError, Result};
use pixels::Planes;
use reconstruction::ChromaResiduals;
use size::BlockSize;
use transform::TxSize;

/// Receives the block layer's trace lines, in dav2d's `DEBUG_BLOCK_INFO`
/// format (one call per line, without the newline).
pub type BlockTrace<'t> = &'t mut dyn FnMut(fmt::Arguments<'_>);

/// Luma samples per 4x4 unit, as a shift.
const UNIT_LOG2: u32 = 2;

/// Coefficients of one transform (at most 32x32 are coded).
const COEFFICIENTS: usize = 32 * 32;
/// A block copy's source window: up to 64x64 samples and a column and a
/// row more for the bilinear filter.
const BLOCK_COPY_WINDOW: usize = 65 * 65;

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
#[allow(clippy::struct_excessive_bools, reason = "the tile's running flags")]
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
    /// What the loop filters need of the blocks: per 64x64 filter block
    /// the quantizer, CDEF set, CCSO and GDF flags; the deblocking edges;
    /// the units with a luma residual.
    filters: FilterMaps,
    /// The non-separable Wiener filters the tile's restoration units
    /// coded last.
    restoration_banks: restoration::Banks,
    /// The dequantizer steps (DC, AC) per plane at `qindex` (dav2d
    /// `ts->dq`).
    dequant: [[u32; 2]; 3],
    /// Per superblock row of 4x4 units (luma, chroma), the units already
    /// coded (dav2d `is_coded`), for above-right and below-left edges.
    coded: [[u64; 64]; 2],
    /// Per 4x4 unit of the current 64x64 (16x16 units): the luma
    /// directional mode index, forward skip coding and transform type
    /// (dav2d `luma_intra_dir_mode_map`, `luma_fsc_map`, `txtp_map`).
    luma_direction_map: [u8; LUMA_MAP_SIDE * LUMA_MAP_SIDE],
    luma_fsc_map: [u8; LUMA_MAP_SIDE * LUMA_MAP_SIDE],
    luma_type_map: [u8; LUMA_MAP_SIDE * LUMA_MAP_SIDE],
    /// Coefficient scratch: one luma transform, a block's chroma, the
    /// level buffer, the scans.
    luma_coefficients: Vec<i16>,
    chroma_residuals: ChromaResiduals,
    levels: Vec<i8>,
    scans: Rc<Scans>,
    /// Whether the last U transform had coefficients (V's skip context).
    u_has_coefficients: bool,
    /// The frame being reconstructed.
    planes: Planes,
    /// The current palette block's index map, unpacked and packed (dav2d
    /// `scratch.pal_idx_y`).
    palette_map: Vec<u8>,
    palette_indices: Vec<u8>,
    /// Intra prediction's edge buffer (dav2d `scratch.edge`).
    edge: Vec<u8>,
    /// The motion of decoded blocks, in frames whose blocks may predict
    /// vectors (inter frames, or frames allowing intra block copy).
    motion: Option<MotionStore>,
    /// A block copy's source window (dav2d `scratch.emu_edge`).
    window: Vec<u8>,
    trace: Option<BlockTrace<'t>>,
}

/// A decoded tile: the reconstructed planes (padded to whole
/// superblocks), what the loop filters need, and the final models.
#[derive(Debug)]
pub struct DecodedTile {
    /// The reconstruction before loop filters.
    pub planes: Planes,
    /// The loop filters' maps.
    pub filters: FilterMaps,
    /// The models after the tile.
    pub cdfs: Box<Cdfs>,
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
) -> Result<DecodedTile> {
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
        filters: FilterMaps::new(geometry.columns, geometry.rows, frame.quantizer.base),
        restoration_banks: restoration::Banks::new(),
        dequant: [[0; 2]; 3],
        coded: [[0; 64]; 2],
        luma_direction_map: [context::NO_DIRECTION; LUMA_MAP_SIDE * LUMA_MAP_SIDE],
        luma_fsc_map: [0; LUMA_MAP_SIDE * LUMA_MAP_SIDE],
        luma_type_map: [0; LUMA_MAP_SIDE * LUMA_MAP_SIDE],
        luma_coefficients: vec![0; COEFFICIENTS],
        chroma_residuals: ChromaResiduals::default(),
        levels: vec![0; LEVELS],
        scans: Rc::new(Scans::new()),
        u_has_coefficients: false,
        planes: Planes::new(geometry.columns, geometry.rows, geometry.superblock_units),
        palette_map: Vec::with_capacity(palette::MAX_SIDE * palette::MAX_SIDE),
        palette_indices: Vec::with_capacity(palette::PACKED_INDICES),
        edge: vec![0; nc_dsp::av2_dec::ipred::EDGE_LEN],
        motion: (!frame.is_intra() || frame.intra_block_copy.is_some()).then(|| {
            MotionStore::new(
                StoreGeometry {
                    columns: geometry.columns,
                    rows: geometry.rows,
                    superblock_log2: frame.superblock_log2,
                    column_start: 0,
                    column_end: geometry.columns,
                    row_start: 0,
                    row_end: geometry.rows,
                },
                sequence.references.mv_bank,
            )
        }),
        window: vec![0; BLOCK_COPY_WINDOW],
        trace,
    };
    tile.dequant = tile.dequantizers(tile.qindex);
    tile.decode(size)?;
    Ok(DecodedTile {
        planes: tile.planes,
        filters: tile.filters,
        cdfs: tile.cdfs,
    })
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
        let intra_frame = self.frame.is_intra();
        self.contexts.above.reset(intra_frame);
        let step = self.geometry.superblock_units;
        let mut row = self.bounds.row_start;
        while row < self.bounds.row_end {
            self.contexts.left.reset(intra_frame);
            // Inter frames' superblocks below the tile's first row seed the
            // MV bank from the row above (dav2d `dav2d_refmvs_reset_sb`).
            let refill = !intra_frame && row != self.bounds.row_start && self.frame.tip.mode != 2;
            if let Some(motion) = &mut self.motion {
                motion.start_row();
            }
            let mut column = self.bounds.column_start;
            while column < self.bounds.column_end {
                (self.row, self.column) = (row, column);
                self.sdp_cfl_disallowed = false;
                self.coded = [[0; 64]; 2];
                if let Some(motion) = &mut self.motion {
                    motion.start_superblock(row, column, refill);
                }
                let root = self.geometry.root;
                // With separate trees from a 64x64 root, the chroma units
                // come between the trees (`partition`).
                let planes = if self.separate_root_trees() { 1 } else { 3 };
                self.restoration_units(0..planes)?;
                let mut direction = 0;
                self.superblock(1, Some(root), Some(root), &mut direction)?;
                if let Some(motion) = &mut self.motion {
                    motion.end_superblock(row, column);
                }
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

    /// A value below `n` (at least 2) in equiprobable bits, the shorter
    /// codes first (dav2d `dav2d_msac_decode_uniform`).
    fn uniform(&mut self, n: u32) -> Result<u32> {
        let bits = n.ilog2() + 1;
        let short = (1 << bits) - n;
        let value = self.reader.literal(bits - 1)?;
        Ok(if value < short {
            value
        } else {
            (value << 1) - short + u32::from(self.bypass()?)
        })
    }

    /// Whether a key or intra-only frame splits each 64x64 superblock into
    /// its luma and chroma trees at the root (dav2d `sdp_root_split`).
    fn separate_root_trees(&self) -> bool {
        self.geometry.root == BlockSize::B64x64
            && self.sequence.partition.separate_trees
            && self.intra_frame()
    }

    /// Whether the frame is key or intra-only.
    fn intra_frame(&self) -> bool {
        matches!(self.frame.frame_type, FrameType::Key | FrameType::Intra)
    }

    /// The index of the 64x64 filter block holding unit (`row`, `column`).
    fn filter_block(&self, row: u32, column: u32) -> usize {
        self.filters.filter_block(row, column)
    }
}

/// Luma samples of `units` 4x4 units.
fn samples(units: u32) -> u32 {
    units << UNIT_LOG2
}

/// The dequantizer step of a quantizer index (dav2d `dav2d_dq_lookup`):
/// 64 for lossless, else a 24-entry mantissa table doubling every 24
/// indices.
pub(crate) fn dequantizer_step(qindex: i32) -> u32 {
    const LOSSLESS_STEP: u32 = 64;
    const OCTAVE: i32 = 24;
    const MANTISSAS: [u32; 24] = [
        40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 54, 55, 57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76,
        78,
    ];
    if qindex == 0 {
        return LOSSLESS_STEP;
    }
    let index = qindex - 1;
    MANTISSAS[(index % OCTAVE) as usize] << (index / OCTAVE)
}

/// A plane quantizer index from the luma AC index and a delta (dav2d
/// `clip_qidx`): lossless stays lossless unless the delta is positive.
pub(crate) fn delta_qindex(qindex: i32, delta: i32) -> i32 {
    const MAX: i32 = 255;
    if qindex == 0 && delta <= 0 {
        0
    } else {
        (qindex + delta).clamp(1, MAX)
    }
}

impl Tile<'_, '_> {
    /// The quantizer matrix of a plane's transforms of size `tx`, when the
    /// frame weights that plane (dav2d `f->qm[tx][plane]`: the first
    /// set's levels; level 15 is flat).
    fn quantizer_matrix(&self, plane: usize, tx: TxSize) -> Option<&'static [u8]> {
        use quantizer_matrices::{MATRICES, MAX_SIDE_LOG2, OFFSETS};
        let matrices = self.frame.quantizer.matrices?;
        let level = [matrices.y, matrices.u, matrices.v][plane][0];
        let tables = MATRICES.get(usize::from(level))?;
        let (w, h) = (tx.w.min(MAX_SIDE_LOG2), tx.h.min(MAX_SIDE_LOG2));
        let offset = usize::from(OFFSETS[usize::from(w)][usize::from(h)]);
        let entries = 1 << (2 * UNIT_LOG2 + u32::from(w + h));
        Some(&tables[usize::from(plane > 0)][offset..offset + entries])
    }

    /// The DC and AC steps of each plane at superblock quantizer `qindex`
    /// (dav2d `init_quant_tables`, one segment).
    fn dequantizers(&self, qindex: u16) -> [[u32; 2]; 3] {
        let quantizer = &self.frame.quantizer;
        let base = &self.sequence.quant;
        let yac = i32::from(qindex);
        let step = |delta: i32| dequantizer_step(delta_qindex(yac, delta));
        [
            [
                step(i32::from(quantizer.y_dc) + i32::from(base.base_y_dc_delta)),
                dequantizer_step(yac),
            ],
            [
                step(i32::from(quantizer.u_dc) + i32::from(base.base_uv_dc_delta)),
                step(i32::from(quantizer.u_ac) + i32::from(base.base_uv_ac_delta)),
            ],
            [
                step(i32::from(quantizer.v_dc) + i32::from(base.base_uv_dc_delta)),
                step(i32::from(quantizer.v_ac) + i32::from(base.base_uv_ac_delta)),
            ],
        ]
    }
}
