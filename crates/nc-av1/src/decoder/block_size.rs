//! Block sizes of the decoder (AV1 §6.10.4 `subSize`, §9.3 size tables):
//! every one of the 22 sizes, square and rectangular, as width and height
//! in samples (log2), and the sizes each partition type produces
//! (`Partition_Subsize`).
use crate::tables::Partition;

/// `MI_SIZE_LOG2`: mode info is kept per 4x4 luma unit.
pub(crate) const MI_SIZE_LOG2: u8 = 2;
/// Smallest and largest block side (log2 samples).
const MIN_SIDE_LOG2: u8 = 2;
const MAX_SIDE_LOG2: u8 = 7;
/// The widest aspect ratio (4:1) as a log2 difference.
const MAX_ASPECT_LOG2: u8 = 2;

/// `MiSize` order (§6.10.4): (width, height) log2 of `BLOCK_4X4` ..
/// `BLOCK_64X16`. Syntax compares sizes in this order and indexes CDFs by it.
const MI_SIZES: [(u8, u8); 22] = [
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
    (4, 5),
    (5, 4),
    (5, 5),
    (5, 6),
    (6, 5),
    (6, 6),
    (6, 7),
    (7, 6),
    (7, 7),
    (2, 4),
    (4, 2),
    (3, 5),
    (5, 3),
    (4, 6),
    (6, 4),
];
/// `Size_Group` of each `MiSize` (the inter-frame `y_mode` CDF).
const SIZE_GROUPS: [u8; 22] = [
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2,
];

/// A block size (`BLOCK_4X4` .. `BLOCK_128X128`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockSize {
    width_log2: u8,
    height_log2: u8,
}

impl BlockSize {
    pub(crate) const BLOCK_4X4: Self = Self::square(2);
    pub(crate) const BLOCK_8X8: Self = Self::square(3);
    pub(crate) const BLOCK_64X64: Self = Self::square(6);
    pub(crate) const BLOCK_128X128: Self = Self::square(MAX_SIDE_LOG2);

    pub(crate) const fn square(side_log2: u8) -> Self {
        Self {
            width_log2: side_log2,
            height_log2: side_log2,
        }
    }

    /// The size `width_log2` x `height_log2` if AV1 has it (sides 4..=128,
    /// aspect at most 4:1, no 4:1 shapes at 128).
    pub(crate) fn new(width_log2: u8, height_log2: u8) -> Option<Self> {
        let sides = MIN_SIDE_LOG2..=MAX_SIDE_LOG2;
        let valid = sides.contains(&width_log2)
            && sides.contains(&height_log2)
            && width_log2.abs_diff(height_log2) <= MAX_ASPECT_LOG2
            && (width_log2.max(height_log2) < MAX_SIDE_LOG2
                || width_log2.abs_diff(height_log2) < MAX_ASPECT_LOG2);
        valid.then_some(Self {
            width_log2,
            height_log2,
        })
    }

    pub(crate) fn width_log2(self) -> u8 {
        self.width_log2
    }
    pub(crate) fn height_log2(self) -> u8 {
        self.height_log2
    }
    /// The spec's `MiSize` value.
    pub(crate) fn index(self) -> usize {
        MI_SIZES
            .iter()
            .position(|&s| s == (self.width_log2, self.height_log2))
            .expect("a block size")
    }
    /// `Size_Group[MiSize]`.
    pub(crate) fn size_group(self) -> usize {
        usize::from(SIZE_GROUPS[self.index()])
    }
    pub(crate) fn width(self) -> usize {
        1 << self.width_log2
    }
    pub(crate) fn height(self) -> usize {
        1 << self.height_log2
    }
    /// `Num_4x4_Blocks_Wide`.
    pub(crate) fn width_mi(self) -> usize {
        1 << (self.width_log2 - MI_SIZE_LOG2)
    }
    /// `Num_4x4_Blocks_High`.
    pub(crate) fn height_mi(self) -> usize {
        1 << (self.height_log2 - MI_SIZE_LOG2)
    }
    /// `Mi_Width_Log2`, `Mi_Height_Log2`.
    pub(crate) fn width_mi_log2(self) -> u8 {
        self.width_log2 - MI_SIZE_LOG2
    }
    pub(crate) fn height_mi_log2(self) -> u8 {
        self.height_log2 - MI_SIZE_LOG2
    }
    pub(crate) fn is_square(self) -> bool {
        self.width_log2 == self.height_log2
    }

    /// `Partition_Subsize[partition][self]` of a square block; `None` where
    /// the table has `BLOCK_INVALID` (4-way splits of 8x8 and 128x128,
    /// anything of 4x4).
    pub(crate) fn subsize(self, partition: Partition) -> Option<Self> {
        debug_assert!(self.is_square(), "partitions split square blocks");
        let side = self.width_log2;
        let (w, h) = match partition {
            Partition::None => (side, side),
            Partition::Horz | Partition::HorzA | Partition::HorzB => (side, side - 1),
            Partition::Vert | Partition::VertA | Partition::VertB => (side - 1, side),
            Partition::Split => (side - 1, side - 1),
            Partition::Horz4 => (side, side.checked_sub(2)?),
            Partition::Vert4 => (side.checked_sub(2)?, side),
        };
        Self::new(w, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AV1 has exactly 22 block sizes (§6.10.4).
    #[test]
    fn mi_size_order_and_size_groups() {
        assert_eq!(BlockSize::BLOCK_4X4.index(), 0);
        assert_eq!(BlockSize::BLOCK_8X8.index(), 3);
        assert_eq!(BlockSize::BLOCK_128X128.index(), 15);
        assert_eq!(BlockSize::new(6, 4).unwrap().index(), 21);
        assert_eq!(BlockSize::new(3, 5).unwrap().size_group(), 1);
        assert_eq!(BlockSize::BLOCK_64X64.size_group(), 3);
        assert_eq!(MI_SIZES.len(), 22);
    }

    #[test]
    fn there_are_twenty_two_sizes() {
        let count = (0..=8)
            .flat_map(|w| (0..=8).map(move |h| (w, h)))
            .filter(|&(w, h)| BlockSize::new(w, h).is_some())
            .count();
        assert_eq!(count, 22);
        assert!(BlockSize::new(7, 5).is_none());
        assert!(BlockSize::new(6, 4).is_some());
    }

    /// Spot checks against `Partition_Subsize` (§9.3).
    #[test]
    fn subsizes_follow_the_specification_table() {
        let s = |side| BlockSize::square(side);
        assert_eq!(s(3).subsize(Partition::Horz), BlockSize::new(3, 2));
        assert_eq!(s(3).subsize(Partition::Split), BlockSize::new(2, 2));
        assert_eq!(s(3).subsize(Partition::Horz4), None);
        assert_eq!(s(4).subsize(Partition::Vert4), BlockSize::new(2, 4));
        assert_eq!(s(6).subsize(Partition::HorzB), BlockSize::new(6, 5));
        assert_eq!(s(7).subsize(Partition::Vert4), None);
        assert_eq!(s(7).subsize(Partition::VertA), BlockSize::new(6, 7));
        assert_eq!(s(2).subsize(Partition::Split), None);
    }
}
