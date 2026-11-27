//! The above and left contexts of the general block layer, per 4x4 luma
//! unit (dav2d `BlockContext`, `reset_context`): what later blocks' symbol
//! contexts read of the blocks already decoded beside them.
//!
//! The above edge spans the frame width and is reset at each tile; the left
//! edge spans one superblock row (64 units, enough for 256x256) and is
//! reset at each superblock row of a tile. Each tool group adds the fields
//! its contexts read here (docs/handoff/av2.md "The general block layer").

use super::palette::MAX_COLORS;
use crate::general::filters::edges::{RESET_CHROMA_CONTEXT, RESET_LUMA_CONTEXT};

/// Units of the left edge: one 256x256 superblock.
pub const LEFT_UNITS: usize = 64;

/// One edge's per-unit context.
#[derive(Debug, Clone, Default)]
pub struct Edge {
    /// Partition context bits per plane tree (0 luma or shared, 1 chroma):
    /// `!(size - 1)` of the last block in 4x4 units, so bit `k` says the
    /// neighbour is narrower (above) or shorter (left) than `2^(k+1)`.
    pub partition: [Vec<u8>; 2],
    /// The neighbour skipped its residual (`skip_txfm`).
    pub skip_txfm: Vec<u8>,
    /// The neighbour is in skip mode.
    pub skip_mode: Vec<u8>,
    /// The neighbour is intra (1 at the start of key frames).
    pub intra: Vec<u8>,
    /// The neighbour uses intra block copy.
    pub intrabc: Vec<u8>,
    /// The neighbour is a block copy scaled by block-adaptive weights
    /// (`morph_pred`).
    pub morph_pred: Vec<u8>,
    /// The neighbour's segment id was predicted.
    pub segment_predicted: Vec<u8>,
    /// The neighbour's luma intra mode (dav2d `IntraPredMode`), or an
    /// inter block's mode (dav2d `InterPredMode` numbering).
    pub mode: Vec<u8>,
    /// The neighbour's references as `i8` bits ([`NO_REFERENCE_BITS`] for
    /// intra and none; dav2d `ref`).
    pub reference: [Vec<u8>; 2],
    /// The neighbour's compound blend (dav2d `CompInterType`).
    pub comp_type: Vec<u8>,
    /// The neighbour's interpolation filter ([`NO_FILTER`] unset).
    pub filter: Vec<u8>,
    /// The neighbour's motion mode (dav2d `MotionMode`).
    pub motion_mode: Vec<u8>,
    /// The neighbour coded an adaptive residual.
    pub amvd: Vec<u8>,
    /// The neighbour's residual precision choice (dav2d `mvprec`: 0
    /// none, 1 the frame's, 2 reduced).
    pub mv_precision: Vec<u8>,
    /// The neighbour's directional luma mode index (0..56: direction and
    /// angle delta), [`NO_DIRECTION`] when not directional.
    pub direction: Vec<u8>,
    /// The neighbour used forward skip coding.
    pub fsc: Vec<u8>,
    /// The neighbour used a reference line beyond the first.
    pub mrl: Vec<u8>,
    /// The neighbour blended two reference lines.
    pub multi_mrl: Vec<u8>,
    /// The neighbour used data-driven intra prediction.
    pub dip: Vec<u8>,
    /// The neighbour's luma palette size, 0 without one (dav2d `pal_sz`).
    pub palette_size: Vec<u8>,
    /// The neighbour's palette colours, valid only with a size (dav2d
    /// `al_pal`).
    pub palette: Vec<[u16; MAX_COLORS]>,
    /// The neighbour's chroma intra mode (per chroma 4x4 unit).
    pub uv_mode: Vec<u8>,
    /// The luma coefficient context of the transform covering the unit:
    /// the level sum capped at 63, plus the DC sign in bits 6-7 (dav2d
    /// `lcoef`; 0x40 when nothing is coded).
    pub luma_coefficients: Vec<u8>,
    /// The same per chroma 4x4 unit, for U and V (dav2d `ccoef`).
    pub chroma_coefficients: [Vec<u8>; 2],
    /// The deblocking level of the transform touching this edge of the
    /// unit, luma and (per chroma 4x4 unit) chroma (dav2d `tx_lpf_y`,
    /// `tx_lpf_uv`; [`crate::general::filters::edges`]).
    pub deblock_luma: Vec<u8>,
    /// See `deblock_luma`.
    pub deblock_chroma: Vec<u8>,
}

/// [`Edge::reference`] of intra blocks: -1 as a byte.
pub const NO_REFERENCE_BITS: u8 = 0xff;
/// [`Edge::filter`] before an inter block sets it (dav2d
/// `DAV2D_N_SWITCHABLE_FILTERS`).
pub const NO_FILTER: u8 = 3;
/// [`Edge::mode`] at the start of inter frames (dav2d `NEARMV`).
const INTER_FRAME_MODE: u8 = 13;

/// [`Edge::direction`] of a block without a directional luma mode.
pub const NO_DIRECTION: u8 = 0xff;
/// [`Edge::luma_coefficients`] of a transform without coefficients (a
/// zero sum with the "no sign" DC code).
pub const NO_COEFFICIENTS: u8 = 0x40;

impl Edge {
    /// An edge of `units` 4x4 units.
    #[must_use]
    pub fn new(units: usize) -> Self {
        Self {
            partition: [vec![0; units], vec![0; units]],
            skip_txfm: vec![0; units],
            skip_mode: vec![0; units],
            intra: vec![0; units],
            intrabc: vec![0; units],
            morph_pred: vec![0; units],
            segment_predicted: vec![0; units],
            mode: vec![0; units],
            reference: [
                vec![NO_REFERENCE_BITS; units],
                vec![NO_REFERENCE_BITS; units],
            ],
            comp_type: vec![0; units],
            filter: vec![NO_FILTER; units],
            motion_mode: vec![0; units],
            amvd: vec![0; units],
            mv_precision: vec![0; units],
            direction: vec![NO_DIRECTION; units],
            fsc: vec![0; units],
            mrl: vec![0; units],
            multi_mrl: vec![0; units],
            dip: vec![0; units],
            palette_size: vec![0; units],
            palette: vec![[0; MAX_COLORS]; units],
            uv_mode: vec![0; units],
            luma_coefficients: vec![NO_COEFFICIENTS; units],
            chroma_coefficients: [vec![NO_COEFFICIENTS; units], vec![NO_COEFFICIENTS; units]],
            deblock_luma: vec![RESET_LUMA_CONTEXT; units],
            deblock_chroma: vec![RESET_CHROMA_CONTEXT; units],
        }
    }

    /// Back to the state before any block (dav2d `reset_context`); key and
    /// intra-only frames start with intra neighbours.
    pub fn reset(&mut self, intra_frame: bool) {
        for plane in &mut self.partition {
            plane.fill(0);
        }
        self.skip_txfm.fill(0);
        self.skip_mode.fill(0);
        self.intra.fill(u8::from(intra_frame));
        self.intrabc.fill(0);
        self.morph_pred.fill(0);
        self.segment_predicted.fill(0);
        self.mode
            .fill(if intra_frame { 0 } else { INTER_FRAME_MODE });
        if !intra_frame {
            for side in &mut self.reference {
                side.fill(NO_REFERENCE_BITS);
            }
            self.comp_type.fill(0);
        }
        self.filter.fill(NO_FILTER);
        self.direction.fill(NO_DIRECTION);
        self.fsc.fill(0);
        self.mrl.fill(0);
        self.multi_mrl.fill(0);
        self.dip.fill(0);
        self.palette_size.fill(0);
        self.uv_mode.fill(0);
        self.luma_coefficients.fill(NO_COEFFICIENTS);
        for plane in &mut self.chroma_coefficients {
            plane.fill(NO_COEFFICIENTS);
        }
        self.deblock_luma.fill(RESET_LUMA_CONTEXT);
        self.deblock_chroma.fill(RESET_CHROMA_CONTEXT);
    }
}

/// The above and left edges of a tile.
#[derive(Debug, Clone, Default)]
pub struct Contexts {
    /// Indexed by the absolute column in 4x4 units.
    pub above: Edge,
    /// Indexed by the row in 4x4 units within the superblock row
    /// (`row & 63`).
    pub left: Edge,
}

impl Contexts {
    /// Contexts for a frame `columns` 4x4 units wide (the above edge
    /// rounded up to whole 256x256 superblocks: blocks overhang the frame).
    #[must_use]
    pub fn new(columns: usize) -> Self {
        Self {
            above: Edge::new(columns.next_multiple_of(LEFT_UNITS)),
            left: Edge::new(LEFT_UNITS),
        }
    }
}

/// The above edge's references and motion modes over one 256-wide
/// column of units, copied at each superblock of an inter frame: warp
/// contexts at a superblock's top read them at 8x8 resolution, which may
/// reach units the block to the left has rewritten since (dav2d
/// `SBEdgeCtx`, AVM #1091).
#[derive(Debug, Clone)]
pub struct AboveCache {
    /// See [`Edge::reference`].
    pub reference: [[u8; LEFT_UNITS]; 2],
    /// See [`Edge::motion_mode`].
    pub motion_mode: [u8; LEFT_UNITS],
}

impl Default for AboveCache {
    fn default() -> Self {
        Self {
            reference: [[NO_REFERENCE_BITS; LEFT_UNITS]; 2],
            motion_mode: [0; LEFT_UNITS],
        }
    }
}

/// The left-edge index of row `row` (in 4x4 units).
#[must_use]
pub fn left_index(row: u32) -> usize {
    (row as usize) & (LEFT_UNITS - 1)
}
