//! Blocks and the square block-partition walk shared by all codecs.
//!
//! Every codec splits its top-level unit (superblock / CTU / macroblock)
//! with a quadtree; the decision of *where* to split belongs to the engine.
//! Nodes that straddle the frame edge must split until every leaf lies
//! inside the frame; nodes entirely outside are not coded at all (in all
//! three codecs, blocks outside the frame carry no syntax).
//!
//! Quadtree nodes are square. Leaves may be rectangular where a backend
//! declares rectangular partitions; every side is a power of two.

/// A block (partition node or leaf) in luma sample coordinates: square for
/// every quadtree node, possibly rectangular for a leaf of a rectangular
/// partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    /// Left edge, in luma samples.
    pub x: usize,
    /// Top edge, in luma samples.
    pub y: usize,
    /// log2 of the width, in luma samples.
    pub log2_width: u8,
    /// log2 of the height, in luma samples.
    pub log2_height: u8,
}

impl Block {
    /// The square block of side `1 << log2_size` at (`x`, `y`).
    #[must_use]
    #[inline]
    pub const fn square(x: usize, y: usize, log2_size: u8) -> Self {
        Self {
            x,
            y,
            log2_width: log2_size,
            log2_height: log2_size,
        }
    }

    /// The `1 << log2_width` x `1 << log2_height` block at (`x`, `y`).
    #[must_use]
    #[inline]
    pub const fn rect(x: usize, y: usize, log2_width: u8, log2_height: u8) -> Self {
        Self {
            x,
            y,
            log2_width,
            log2_height,
        }
    }

    /// Whether width and height are equal.
    #[must_use]
    #[inline]
    pub const fn is_square(&self) -> bool {
        self.log2_width == self.log2_height
    }

    /// log2 of the side of a square block. Backends that declare no
    /// rectangular shapes only ever receive squares.
    #[must_use]
    #[inline]
    pub fn log2_size(&self) -> u8 {
        debug_assert!(self.is_square(), "log2_size of a rectangle {self:?}");
        self.log2_width
    }

    /// Side length of a square block, in luma samples.
    #[must_use]
    #[inline]
    pub fn size(&self) -> usize {
        1 << self.log2_size()
    }

    /// Width in luma samples.
    #[must_use]
    #[inline]
    pub const fn width(&self) -> usize {
        1 << self.log2_width
    }

    /// Height in luma samples.
    #[must_use]
    #[inline]
    pub const fn height(&self) -> usize {
        1 << self.log2_height
    }

    /// Area in luma samples.
    #[must_use]
    #[inline]
    pub const fn area(&self) -> usize {
        self.width() * self.height()
    }

    /// log2 of the shorter side.
    #[must_use]
    #[inline]
    pub fn log2_min_side(&self) -> u8 {
        self.log2_width.min(self.log2_height)
    }

    /// The four quadrants of a square block, in raster (coding) order.
    #[must_use]
    #[inline]
    pub fn quadrants(&self) -> [Block; 4] {
        let half = self.size() / 2;
        let child = |dx, dy| Block::square(self.x + dx, self.y + dy, self.log2_size() - 1);
        [
            child(0, 0),
            child(half, 0),
            child(0, half),
            child(half, half),
        ]
    }

    /// True if the whole block lies inside a `width` x `height` frame.
    #[must_use]
    #[inline]
    pub fn is_inside(&self, width: usize, height: usize) -> bool {
        self.x + self.width() <= width && self.y + self.height() <= height
    }
}

/// Split policy for intra bring-up: split down to `target_log2`, and further
/// only where the frame edge forces it.
pub(crate) fn should_split(
    node: &Block,
    target_log2: u8,
    min_log2: u8,
    width: usize,
    height: usize,
) -> bool {
    node.log2_size() > min_log2
        && (node.log2_size() > target_log2 || !node.is_inside(width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadrants_tile_the_parent() {
        let b = Block::square(64, 0, 6);
        let q = b.quadrants();
        assert_eq!(q[0], Block::square(64, 0, 5));
        assert_eq!(q[3], Block::square(96, 32, 5));
    }

    #[test]
    fn edge_nodes_split_inner_nodes_stop_at_target() {
        // 1280x720: the 64x64 superblock at y=704 straddles the bottom edge.
        let edge = Block::square(0, 704, 6);
        assert!(should_split(&edge, 6, 3, 1280, 720));
        let inner = Block::square(0, 0, 4);
        assert!(!should_split(&inner, 4, 3, 1280, 720));
        assert!(should_split(
            &Block::square(inner.x, inner.y, 5),
            4,
            3,
            1280,
            720
        ));
    }
}
