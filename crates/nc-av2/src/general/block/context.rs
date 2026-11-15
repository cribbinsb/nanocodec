//! The above and left contexts of the general block layer, per 4x4 luma
//! unit (dav2d `BlockContext`, `reset_context`): what later blocks' symbol
//! contexts read of the blocks already decoded beside them.
//!
//! The above edge spans the frame width and is reset at each tile; the left
//! edge spans one superblock row (64 units, enough for 256x256) and is
//! reset at each superblock row of a tile. Each tool group adds the fields
//! its contexts read here (docs/handoff/av2.md "The general block layer").

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
}

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
