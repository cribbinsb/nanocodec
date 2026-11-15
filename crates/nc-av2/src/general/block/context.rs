//! The above and left contexts of the general block layer, per 4x4 luma
//! unit (dav2d `BlockContext`, `reset_context`): what later blocks' symbol
//! contexts read of the blocks already decoded beside them.
//!
//! The above edge spans the frame width and is reset at each tile; the left
//! edge spans one superblock row (64 units, enough for 256x256) and is
//! reset at each superblock row of a tile. Each tool group adds the fields
//! its contexts read here (docs/handoff/av2.md "The general block layer").

use super::palette::MAX_COLORS;

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
    /// The neighbour's segment id was predicted.
    pub segment_predicted: Vec<u8>,
    /// The neighbour's luma intra mode (dav2d `IntraPredMode`; DC for
    /// inter blocks).
    pub mode: Vec<u8>,
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
    pub palette: Vec<[u8; MAX_COLORS]>,
    /// The neighbour's chroma intra mode (per chroma 4x4 unit).
    pub uv_mode: Vec<u8>,
    /// The luma coefficient context of the transform covering the unit:
    /// the level sum capped at 63, plus the DC sign in bits 6-7 (dav2d
    /// `lcoef`; 0x40 when nothing is coded).
    pub luma_coefficients: Vec<u8>,
    /// The same per chroma 4x4 unit, for U and V (dav2d `ccoef`).
    pub chroma_coefficients: [Vec<u8>; 2],
}

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
            segment_predicted: vec![0; units],
            mode: vec![0; units],
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
        self.segment_predicted.fill(0);
        self.mode.fill(0);
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

/// The left-edge index of row `row` (in 4x4 units).
#[must_use]
pub fn left_index(row: u32) -> usize {
    (row as usize) & (LEFT_UNITS - 1)
}
