//! The coded leaf sizes around a node, for the RD partition search's
//! context gate (`rd_split.rs`, `planner.rd_split_context`).
//!
//! Per 8x8 cell of the picture: the log2 of the smallest side of the leaf
//! that covers it, as the search decided it, in the frame being coded and
//! in the last committed one. A node then reads its left and above
//! neighbours' leaf sizes (already coded: the search runs in raster order of
//! superblocks and z-order inside them) and the smallest leaf the previous
//! frame coded over its area, both relative to its own size.
//!
//! Why these: on the node dump of the search (HEVC, VVC and AV1, Default and
//! Slow, 6 clips x 2 QP, 1.16 M trialled nodes; docs/handoff/rd-split-gate.md)
//! the neighbours' and co-located depth relative to the
//! node predicted where the search changes the plan better than any other
//! cheap feature (the planner's margin, the quarters' vector spread, the
//! children's skips and vectors, the split SAD ratio), the features
//! `VVenC` / HM-style fast split decisions also lean on.

use crate::partition::Block;

/// Cell size of the map (log2): the smallest inter leaf any backend codes.
const CELL_LOG2: u8 = 3;
/// A cell with no leaf recorded (a frame without the search, or none yet).
const UNKNOWN: u8 = 0;

/// The leaf-size maps of the current and the last committed frame.
#[derive(Debug, Default)]
pub(crate) struct DepthContext {
    cols: usize,
    rows: usize,
    current: Vec<u8>,
    previous: Vec<u8>,
    /// Whether the frame being coded recorded leaves (it ran the search).
    recorded: bool,
    /// Whether the frame being coded skipped the search by its frame
    /// period: the co-located record stays the last searched frame's.
    held: bool,
    /// Frames committed so far (the gate's refresh rotation).
    frames: u64,
}

/// What the maps say around one node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NodeContext {
    /// The smaller of the left and above neighbours' leaf log2 minus the
    /// node's (one of them where the other is unknown; 0 where both are).
    pub neighbours: i8,
    /// The smallest leaf log2 the last frame coded over the node, minus the
    /// node's (0 where unknown).
    pub co_located: i8,
}

impl DepthContext {
    /// Size the maps for a `width` x `height` picture; a new size forgets
    /// everything recorded.
    pub fn fit(&mut self, width: usize, height: usize) {
        let (cols, rows) = (
            width.div_ceil(1 << CELL_LOG2),
            height.div_ceil(1 << CELL_LOG2),
        );
        if (cols, rows) != (self.cols, self.rows) {
            *self = Self {
                cols,
                rows,
                current: vec![UNKNOWN; cols * rows],
                previous: vec![UNKNOWN; cols * rows],
                recorded: false,
                held: false,
                frames: self.frames,
            };
        }
    }

    /// Record that the square `node` is coded with leaves whose smallest
    /// side is `1 << leaf_log2` everywhere in it.
    pub fn record(&mut self, node: &Block, leaf_log2: u8) {
        let value = leaf_log2.max(CELL_LOG2);
        let (col0, row0) = (node.x >> CELL_LOG2, node.y >> CELL_LOG2);
        if col0 >= self.cols || row0 >= self.rows {
            return;
        }
        let cells = 1usize << node.log2_size().saturating_sub(CELL_LOG2);
        let col_end = (col0 + cells).min(self.cols);
        for row in row0..(row0 + cells).min(self.rows) {
            self.current[row * self.cols + col0..row * self.cols + col_end].fill(value);
        }
        self.recorded = true;
    }

    /// Keep the co-located record through the frame being coded: an inter
    /// frame the search's frame period leaves out.
    pub fn hold(&mut self) {
        self.held = true;
    }

    /// The coded frame is final: its leaves become the co-located record
    /// (none when it did not run the search, unless it was held).
    pub fn commit(&mut self) {
        if self.recorded {
            std::mem::swap(&mut self.current, &mut self.previous);
        } else if !self.held {
            self.previous.fill(UNKNOWN);
        }
        self.recorded = false;
        self.held = false;
        self.frames += 1;
    }

    /// Frames committed so far.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// The context of the square `node` (inside the maps).
    pub fn of(&self, node: &Block) -> NodeContext {
        let log2 = node.log2_size().max(CELL_LOG2);
        let (col0, row0) = (node.x >> CELL_LOG2, node.y >> CELL_LOG2);
        let cell = |map: &[u8], col: usize, row: usize| {
            (col < self.cols && row < self.rows).then(|| map[row * self.cols + col])
        };
        let known = |v: Option<u8>| v.filter(|&v| v != UNKNOWN);
        let left = col0
            .checked_sub(1)
            .and_then(|col| known(cell(&self.current, col, row0)));
        let above = row0
            .checked_sub(1)
            .and_then(|row| known(cell(&self.current, col0, row)));
        let neighbours = match (left, above) {
            (Some(l), Some(a)) => Some(l.min(a)),
            (one, other) => one.or(other),
        };
        let cells = 1usize << (log2 - CELL_LOG2);
        let co_located = (row0..(row0 + cells).min(self.rows))
            .flat_map(|row| {
                let start = row * self.cols;
                self.previous[start + col0.min(self.cols)..start + (col0 + cells).min(self.cols)]
                    .iter()
                    .copied()
            })
            .filter(|&v| v != UNKNOWN)
            .min();
        let relative = |v: Option<u8>| v.map_or(0, |v| v as i8 - log2 as i8);
        NodeContext {
            neighbours: relative(neighbours),
            co_located: relative(co_located),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_context_reads_as_the_node_size() {
        let mut map = DepthContext::default();
        map.fit(128, 64);
        let ctx = map.of(&Block::square(32, 32, 5));
        assert_eq!((ctx.neighbours, ctx.co_located), (0, 0));
    }

    #[test]
    fn neighbours_and_co_located_leaves_relative_to_the_node() {
        let mut map = DepthContext::default();
        map.fit(128, 64);
        map.record(&Block::square(0, 0, 6), 4); // 16x16 leaves on the left
        map.commit();
        map.record(&Block::square(0, 0, 6), 6);
        map.record(&Block::square(64, 0, 5), 5);
        let ctx = map.of(&Block::square(96, 0, 5));
        // Left: the 32x32 leaf at (64, 0); no row above.
        assert_eq!(ctx.neighbours, 0);
        // The last frame coded nothing over (96, 0).
        assert_eq!(ctx.co_located, 0);
        let ctx = map.of(&Block::square(0, 32, 5));
        // Above: the 64x64 leaf; co-located: the 16x16 leaves of the last frame.
        assert_eq!((ctx.neighbours, ctx.co_located), (1, -1));
    }

    #[test]
    fn a_frame_without_leaves_forgets_the_co_located_record() {
        let mut map = DepthContext::default();
        map.fit(64, 64);
        map.record(&Block::square(0, 0, 6), 3);
        map.commit();
        map.commit();
        assert_eq!(map.of(&Block::square(0, 0, 5)).co_located, 0);
    }
}
