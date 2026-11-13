//! Rectangular partitions of a square node: the shapes a backend may
//! declare beside the quadtree split.
//!
//! A shape divides a square partition node into rectangular leaves, all
//! with power-of-two sides: binary halves (VVC `SPLIT_BT_HOR/VER`, H.266
//! §7.3.11.4; AV1 `PARTITION_HORZ/VERT`, AV1 §5.11.4) and ternary
//! quarter-half-quarter splits (VVC `SPLIT_TT_HOR/VER`). The engine plans
//! shapes only for backends that declare them
//! ([`crate::CodecBackend::partition_shapes`]); everything else keeps the
//! square quadtree.

use crate::partition::Block;

/// The most leaves any shape produces.
pub const MAX_SHAPE_PARTS: usize = 3;

/// A rectangular partition of a square node into leaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitionShape {
    /// Top and bottom halves.
    Horizontal,
    /// Left and right halves.
    Vertical,
    /// Top quarter, middle half, bottom quarter.
    HorizontalTriple,
    /// Left quarter, middle half, right quarter.
    VerticalTriple,
}

impl PartitionShape {
    /// Every shape, in [`ShapeSet`] bit order.
    pub const ALL: [PartitionShape; 4] = [
        PartitionShape::Horizontal,
        PartitionShape::Vertical,
        PartitionShape::HorizontalTriple,
        PartitionShape::VerticalTriple,
    ];

    /// Whether the cut lines run horizontally (the leaves are stacked).
    #[must_use]
    pub const fn is_horizontal(self) -> bool {
        matches!(
            self,
            PartitionShape::Horizontal | PartitionShape::HorizontalTriple
        )
    }

    /// Number of leaves.
    #[must_use]
    pub const fn part_count(self) -> usize {
        match self {
            PartitionShape::Horizontal | PartitionShape::Vertical => 2,
            PartitionShape::HorizontalTriple | PartitionShape::VerticalTriple => 3,
        }
    }

    /// log2 of the shorter side of the smallest leaf of a square node of
    /// side `1 << log2_size`.
    #[must_use]
    pub const fn log2_min_side(self, log2_size: u8) -> u8 {
        match self {
            PartitionShape::Horizontal | PartitionShape::Vertical => log2_size - 1,
            PartitionShape::HorizontalTriple | PartitionShape::VerticalTriple => log2_size - 2,
        }
    }

    /// The leaves of square `node`, in coding order (top to bottom, left
    /// to right); only the first [`Self::part_count`] are meaningful.
    #[must_use]
    pub fn parts(self, node: &Block) -> [Block; MAX_SHAPE_PARTS] {
        let log2 = node.log2_size();
        let size = node.size();
        // (offset along the cut axis, log2 extent along it) per leaf.
        let spans: [(usize, u8); MAX_SHAPE_PARTS] = match self {
            PartitionShape::Horizontal | PartitionShape::Vertical => {
                [(0, log2 - 1), (size / 2, log2 - 1), (0, log2 - 1)]
            }
            PartitionShape::HorizontalTriple | PartitionShape::VerticalTriple => [
                (0, log2 - 2),
                (size / 4, log2 - 1),
                (3 * size / 4, log2 - 2),
            ],
        };
        spans.map(|(offset, extent)| {
            if self.is_horizontal() {
                Block::rect(node.x, node.y + offset, log2, extent)
            } else {
                Block::rect(node.x + offset, node.y, extent, log2)
            }
        })
    }

    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// A set of [`PartitionShape`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShapeSet(u8);

impl ShapeSet {
    /// No shapes: the square quadtree only.
    pub const EMPTY: ShapeSet = ShapeSet(0);

    /// The set holding `shapes`.
    #[must_use]
    pub fn of(shapes: &[PartitionShape]) -> Self {
        Self(shapes.iter().fold(0, |bits, s| bits | s.bit()))
    }

    /// This set with `shape` added.
    #[must_use]
    pub fn with(self, shape: PartitionShape) -> Self {
        Self(self.0 | shape.bit())
    }

    /// Whether `shape` is in the set.
    #[must_use]
    pub fn contains(self, shape: PartitionShape) -> bool {
        self.0 & shape.bit() != 0
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The shapes in the set, in [`PartitionShape::ALL`] order.
    pub fn iter(self) -> impl Iterator<Item = PartitionShape> {
        PartitionShape::ALL
            .into_iter()
            .filter(move |&s| self.contains(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_tile_the_node() {
        let node = Block::square(64, 32, 5);
        for shape in PartitionShape::ALL {
            let parts = shape.parts(&node);
            let parts = &parts[..shape.part_count()];
            assert_eq!(parts.iter().map(Block::area).sum::<usize>(), node.area());
            for part in parts {
                assert!(part.is_inside(64 + 32, 32 + 32), "{shape:?} {part:?}");
                assert!(part.log2_min_side() >= shape.log2_min_side(5));
            }
        }
        assert_eq!(
            PartitionShape::VerticalTriple.parts(&node)[1],
            Block::rect(72, 32, 4, 5)
        );
        assert_eq!(
            PartitionShape::Horizontal.parts(&node)[1],
            Block::rect(64, 48, 5, 4)
        );
    }

    #[test]
    fn sets_hold_what_was_added() {
        let set = ShapeSet::EMPTY.with(PartitionShape::Vertical);
        assert!(set.contains(PartitionShape::Vertical));
        assert!(!set.contains(PartitionShape::Horizontal));
        assert_eq!(set.iter().collect::<Vec<_>>(), [PartitionShape::Vertical]);
        assert!(ShapeSet::EMPTY.is_empty());
    }
}
