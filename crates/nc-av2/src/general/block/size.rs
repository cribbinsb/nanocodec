//! Block sizes and partition types of the general block layer: every AV2
//! block from 256x256 to 4x4 (dav2d `enum BlockSize`, `enum
//! BlockPartition`, `dav2d_block_dimensions`), and what each partition of
//! each size splits into (dav2d `decode_sb`'s `subb` table).

/// A block size, largest first (dav2d `enum BlockSize` order, which the
/// subdivision table indexes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
#[allow(missing_docs, reason = "named by their width x height in luma samples")]
pub enum BlockSize {
    B256x256,
    B256x128,
    B128x256,
    B128x128,
    B128x64,
    B64x128,
    B64x64,
    B64x32,
    B64x16,
    B64x8,
    B64x4,
    B32x64,
    B32x32,
    B32x16,
    B32x8,
    B32x4,
    B16x64,
    B16x32,
    B16x16,
    B16x8,
    B16x4,
    B8x64,
    B8x32,
    B8x16,
    B8x8,
    B8x4,
    B4x64,
    B4x32,
    B4x16,
    B4x8,
    B4x4,
}

/// Every size, in [`BlockSize`] order.
#[allow(clippy::enum_glob_use, reason = "a table of every variant")]
const SIZES: [BlockSize; 31] = {
    use BlockSize::*;
    [
        B256x256, B256x128, B128x256, B128x128, B128x64, B64x128, B64x64, B64x32, B64x16, B64x8,
        B64x4, B32x64, B32x32, B32x16, B32x8, B32x4, B16x64, B16x32, B16x16, B16x8, B16x4, B8x64,
        B8x32, B8x16, B8x8, B8x4, B4x64, B4x32, B4x16, B4x8, B4x4,
    ]
};

/// log2 of each size's width and height in 4x4 units.
const LOG2: [(u8, u8); 31] = [
    (6, 6),
    (6, 5),
    (5, 6),
    (5, 5),
    (5, 4),
    (4, 5),
    (4, 4),
    (4, 3),
    (4, 2),
    (4, 1),
    (4, 0),
    (3, 4),
    (3, 3),
    (3, 2),
    (3, 1),
    (3, 0),
    (2, 4),
    (2, 3),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 4),
    (1, 3),
    (1, 2),
    (1, 1),
    (1, 0),
    (0, 4),
    (0, 3),
    (0, 2),
    (0, 1),
    (0, 0),
];

impl BlockSize {
    /// The square superblock of side `1 << log2` luma samples (6, 7 or 8).
    #[must_use]
    pub fn superblock(log2: u8) -> Self {
        match log2 {
            8 => Self::B256x256,
            7 => Self::B128x128,
            _ => Self::B64x64,
        }
    }

    /// Width and height in 4x4 units.
    #[must_use]
    pub fn units(self) -> (u32, u32) {
        let (w, h) = self.log2_units();
        (1 << w, 1 << h)
    }

    /// log2 of the width and height in 4x4 units.
    #[must_use]
    pub fn log2_units(self) -> (u8, u8) {
        LOG2[self as usize]
    }

    /// The size of `log2` width and height in 4x4 units, if AV2 has it.
    #[must_use]
    pub fn of_log2_units(log2: (u8, u8)) -> Option<Self> {
        LOG2.iter()
            .position(|&size| size == log2)
            .map(|index| SIZES[index])
    }
}

/// How a block splits (dav2d `enum BlockPartition`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Partition {
    /// Not split: one coding block.
    None,
    /// Two halves, one above the other.
    Horizontal,
    /// Two halves side by side.
    Vertical,
    /// A quarter on top, two quarter squares side by side, a quarter below.
    Horizontal3,
    /// The transpose of [`Partition::Horizontal3`].
    Vertical3,
    /// Horizontal strips of 1/8, 2/8, 4/8 and 1/8 of the height.
    Horizontal4A,
    /// Horizontal strips of 1/8, 4/8, 2/8 and 1/8.
    Horizontal4B,
    /// The transpose of [`Partition::Horizontal4A`].
    Vertical4A,
    /// The transpose of [`Partition::Horizontal4B`].
    Vertical4B,
    /// Four squares (128x128 and 256x256 blocks only).
    Split,
}

impl Partition {
    /// The children's minimum side in 4x4 units below which they cannot
    /// split again, (width, height) (dav2d `decode_sb` `lim`).
    #[must_use]
    pub fn child_limit(self) -> (u32, u32) {
        match self {
            Self::None => (1, 1),
            Self::Horizontal => (1, 2),
            Self::Vertical => (2, 1),
            Self::Horizontal3 => (2, 4),
            Self::Vertical3 => (4, 2),
            Self::Horizontal4A | Self::Horizontal4B => (1, 8),
            Self::Vertical4A | Self::Vertical4B => (8, 1),
            Self::Split => (2, 2),
        }
    }

    /// The value dav2d records in a partition-direction word (its `bp`).
    #[must_use]
    pub fn code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Horizontal => 1,
            Self::Vertical => 2,
            Self::Horizontal3 => 3,
            Self::Vertical3 => 4,
            Self::Horizontal4A => 5,
            Self::Horizontal4B => 6,
            Self::Vertical4A => 7,
            Self::Vertical4B => 8,
            Self::Split => 9,
        }
    }

    /// The partition of a dav2d `bp` code.
    #[must_use]
    pub fn of_code(code: u32) -> Option<Self> {
        Some(match code {
            0 => Self::None,
            1 => Self::Horizontal,
            2 => Self::Vertical,
            3 => Self::Horizontal3,
            4 => Self::Vertical3,
            5 => Self::Horizontal4A,
            6 => Self::Horizontal4B,
            7 => Self::Vertical4A,
            8 => Self::Vertical4B,
            9 => Self::Split,
            _ => return None,
        })
    }
}

/// What a block size splits into, per direction (horizontal, vertical):
/// the half, quarter and eighth strips and the split's quarter block; and
/// its partition-context offsets (plain, direction).
#[derive(Debug, Clone, Copy)]
pub struct Subdivision {
    /// `parts[direction][kind]`: kind 0 half, 1 quarter, 2 eighth, 3 the
    /// square (or rectangle) a four-way split gives.
    pub parts: [[Option<BlockSize>; 4]; 2],
    /// Context offsets of the split and direction symbols (in units of
    /// four contexts); `None` where the symbol is never coded.
    pub context: [Option<u8>; 2],
}

impl BlockSize {
    /// What this size splits into (dav2d `subb`).
    #[must_use]
    pub fn subdivision(self) -> Subdivision {
        SUBDIVISIONS[self as usize]
    }
}

/// [`Subdivision`] of every size, in [`BlockSize`] order.
#[allow(clippy::enum_glob_use, reason = "a table of every variant")]
const SUBDIVISIONS: [Subdivision; 31] = {
    use BlockSize::*;
    const fn s(
        horizontal: [Option<BlockSize>; 4],
        vertical: [Option<BlockSize>; 4],
        context: [Option<u8>; 2],
    ) -> Subdivision {
        Subdivision {
            parts: [horizontal, vertical],
            context,
        }
    }
    const N: Option<BlockSize> = None;
    [
        s(
            [Some(B256x128), N, N, Some(B128x128)],
            [Some(B128x256), N, N, Some(B128x128)],
            [Some(9), Some(12)],
        ),
        s([N; 4], [Some(B128x128), N, N, N], [Some(8), None]),
        s([Some(B128x128), N, N, N], [N; 4], [Some(7), None]),
        s(
            [Some(B128x64), N, N, Some(B64x64)],
            [Some(B64x128), N, N, Some(B64x64)],
            [Some(6), Some(9)],
        ),
        s([N; 4], [Some(B64x64), N, N, N], [Some(5), None]),
        s([Some(B64x64), N, N, N], [N; 4], [Some(4), None]),
        s(
            [Some(B64x32), Some(B64x16), Some(B64x8), Some(B32x32)],
            [Some(B32x64), Some(B16x64), Some(B8x64), Some(B32x32)],
            [Some(3), Some(6)],
        ),
        s(
            [Some(B64x16), Some(B64x8), Some(B64x4), Some(B32x16)],
            [Some(B32x32), Some(B16x32), Some(B8x32), Some(B32x16)],
            [Some(3), Some(5)],
        ),
        s(
            [Some(B64x8), Some(B64x4), N, Some(B32x8)],
            [Some(B32x16), Some(B16x16), Some(B8x16), Some(B32x8)],
            [Some(15), Some(14)],
        ),
        s([N; 4], [N; 4], [Some(0), Some(0)]),
        s([N; 4], [N; 4], [Some(0), None]),
        s(
            [Some(B32x32), Some(B32x16), Some(B32x8), Some(B16x32)],
            [Some(B16x64), Some(B8x64), Some(B4x64), Some(B16x32)],
            [Some(3), Some(4)],
        ),
        s(
            [Some(B32x16), Some(B32x8), Some(B32x4), Some(B16x16)],
            [Some(B16x32), Some(B8x32), Some(B4x32), Some(B16x16)],
            [Some(2), Some(3)],
        ),
        s(
            [Some(B32x8), Some(B32x4), N, Some(B16x8)],
            [Some(B16x16), Some(B8x16), Some(B4x16), Some(B16x8)],
            [Some(2), Some(2)],
        ),
        s(
            [Some(B32x4), N, N, N],
            [Some(B16x8), Some(B8x8), Some(B4x8), Some(B16x4)],
            [Some(13), Some(14)],
        ),
        s([N; 4], [N; 4], [Some(0), None]),
        s(
            [Some(B16x32), Some(B16x16), Some(B16x8), Some(B8x32)],
            [Some(B8x64), Some(B4x64), N, Some(B8x32)],
            [Some(14), Some(13)],
        ),
        s(
            [Some(B16x16), Some(B16x8), Some(B16x4), Some(B8x16)],
            [Some(B8x32), Some(B4x32), N, Some(B8x16)],
            [Some(2), Some(1)],
        ),
        s(
            [Some(B16x8), Some(B16x4), N, Some(B8x8)],
            [Some(B8x16), Some(B4x16), N, Some(B8x8)],
            [Some(1), Some(0)],
        ),
        s(
            [Some(B16x4), N, N, N],
            [Some(B8x8), Some(B4x8), N, Some(B8x4)],
            [Some(1), Some(2)],
        ),
        s([N; 4], [Some(B8x4), N, N, N], [Some(11), None]),
        s([N; 4], [N; 4], [Some(0), Some(0)]),
        s(
            [Some(B8x16), Some(B8x8), Some(B8x4), Some(B4x16)],
            [Some(B4x32), N, N, N],
            [Some(12), Some(13)],
        ),
        s(
            [Some(B8x8), Some(B8x4), N, Some(B4x8)],
            [Some(B4x16), N, N, N],
            [Some(1), Some(1)],
        ),
        s(
            [Some(B8x4), N, N, N],
            [Some(B4x8), N, N, N],
            [Some(0), Some(0)],
        ),
        s([N; 4], [Some(B4x4), N, N, N], [Some(0), None]),
        s([N; 4], [N; 4], [Some(0), None]),
        s([N; 4], [N; 4], [Some(0), None]),
        s([Some(B4x8), N, N, N], [N; 4], [Some(10), None]),
        s([Some(B4x4), N, N, N], [N; 4], [Some(0), None]),
        s([N; 4], [N; 4], [None, None]),
    ]
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_subdivisions_agree() {
        for (index, &size) in SIZES.iter().enumerate() {
            assert_eq!(size as usize, index);
            assert_eq!(BlockSize::of_log2_units(size.log2_units()), Some(size));
            let (w, h) = size.units();
            let parts = size.subdivision().parts;
            // A horizontal half halves the height, a vertical one the width.
            if let Some(half) = parts[0][0] {
                assert_eq!(half.units(), (w, h / 2));
            }
            if let Some(half) = parts[1][0] {
                assert_eq!(half.units(), (w / 2, h));
            }
        }
        assert_eq!(BlockSize::superblock(7), BlockSize::B128x128);
    }
}
