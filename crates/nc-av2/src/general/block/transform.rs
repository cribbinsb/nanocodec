//! Transform sizes and the luma transform partition of the general block
//! layer (dav2d `TxfmInfo`, `dav2d_txfm_dimensions`, `dav2d_tx_part_tbl`,
//! `dav2d_max_txfm_size_for_bs`, `read_tx_part`; AVM `read_tx_partition`).
//!
//! dav2d tabulates these per size; each is a rule of the side logs, so
//! here they are computed: a transform is 4..64 samples a side with
//! aspect ratios up to 16:1, and codes at most 32x32 coefficients.
use super::Tile;
use super::record::BlockRecord;
use super::size::BlockSize;
use nc_common::decode::{DecodeError, Result};

/// log2 of the largest transform side in 4x4 units (64 samples) and of the
/// largest coded side (32 samples).
const MAX_LOG2: u8 = 4;
const MAX_CODED_LOG2: u8 = 3;
/// Lossless blocks use transforms up to 32x32.
const LOSSLESS_MAX_LOG2: u8 = 3;

/// A transform size: log2 of its width and height in 4x4 units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TxSize {
    /// log2 of the width in 4x4 units.
    pub w: u8,
    /// log2 of the height.
    pub h: u8,
}

/// dav2d `enum RectTxfmSize` order: the squares, then the rectangles.
const DAV2D_ORDER: [(u8, u8); 25] = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (0, 1),
    (1, 0),
    (1, 2),
    (2, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (0, 2),
    (2, 0),
    (1, 3),
    (3, 1),
    (2, 4),
    (4, 2),
    (0, 3),
    (3, 0),
    (1, 4),
    (4, 1),
    (0, 4),
    (4, 0),
];

impl TxSize {
    /// The 4x4 transform.
    pub const TX4: Self = Self { w: 0, h: 0 };

    /// Width and height in 4x4 units.
    #[must_use]
    pub fn units(self) -> (u32, u32) {
        (1 << self.w, 1 << self.h)
    }

    /// Width and height in samples.
    #[must_use]
    pub fn samples(self) -> (usize, usize) {
        (4 << self.w, 4 << self.h)
    }

    /// dav2d's `enum RectTxfmSize` value, which the kernels take.
    #[must_use]
    pub fn dav2d_index(self) -> usize {
        DAV2D_ORDER
            .iter()
            .position(|&size| size == (self.w, self.h))
            .unwrap_or(0)
    }

    /// The smaller and larger side logs (dav2d `min`, `max`).
    #[must_use]
    pub fn min(self) -> u8 {
        self.w.min(self.h)
    }

    /// See [`TxSize::min`].
    #[must_use]
    pub fn max(self) -> u8 {
        self.w.max(self.h)
    }

    /// The size context of the coefficient models (dav2d `ctx`: the
    /// rounded-up mean of the side logs).
    #[must_use]
    pub fn context(self) -> usize {
        usize::from((self.w + self.h).div_ceil(2))
    }

    /// The coded side logs: 64-sample sides code 32 coefficients.
    #[must_use]
    pub fn coded(self) -> (u8, u8) {
        (self.w.min(MAX_CODED_LOG2), self.h.min(MAX_CODED_LOG2))
    }

    /// A 64-long transform whose short side is at least 32 (dav2d `sub ==
    /// TX_32X32`): it has DCT only.
    #[must_use]
    pub fn is_large(self) -> bool {
        self.max() == MAX_LOG2 && self.min() >= MAX_CODED_LOG2
    }

    /// The size of `w` x `h` units logs, when AV2 has one.
    fn of(w: i8, h: i8) -> Option<Self> {
        let valid = |side: i8| (0..=MAX_LOG2 as i8).contains(&side);
        (valid(w) && valid(h)).then_some(Self {
            w: w as u8,
            h: h as u8,
        })
    }
}

/// The largest luma transform of a block (dav2d
/// `dav2d_max_txfm_size_for_bs[][I444]`), or the lossless one.
#[must_use]
pub fn luma_max(size: BlockSize, lossless: bool) -> TxSize {
    let (w, h) = size.log2_units();
    let cap = if lossless {
        LOSSLESS_MAX_LOG2
    } else {
        MAX_LOG2
    };
    TxSize {
        w: w.min(cap),
        h: h.min(cap),
    }
}

/// The 4:2:0 chroma transform of a chroma block of luma size `size`
/// (dav2d `dav2d_max_txfm_size_for_bs[][I420]`): half the sides, from 4
/// to 64 samples.
#[must_use]
pub fn chroma_max(size: BlockSize) -> TxSize {
    let (w, h) = size.log2_units();
    TxSize {
        w: w.saturating_sub(1).min(MAX_LOG2),
        h: h.saturating_sub(1).min(MAX_LOG2),
    }
}

/// How a luma block splits into transforms (dav2d `enum TxPartition`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxPartition {
    /// One transform.
    None,
    /// Four quarters.
    Split,
    /// Two of half height.
    Horizontal,
    /// Two of half width.
    Vertical,
    /// Four of quarter height.
    Horizontal4,
    /// Four of quarter width.
    Vertical4,
    /// Two quarters, a half, two quarters (the half across the middle),
    /// stacked.
    Horizontal5,
    /// The same side by side.
    Vertical5,
}

impl TxPartition {
    /// Every partition, in dav2d order.
    const ALL: [Self; 8] = [
        Self::None,
        Self::Split,
        Self::Horizontal,
        Self::Vertical,
        Self::Horizontal4,
        Self::Vertical4,
        Self::Horizontal5,
        Self::Vertical5,
    ];

    /// The transform size of each part (for the five-part splits, the
    /// small parts) of a block of `size`, if the split exists (dav2d
    /// `dav2d_tx_part_tbl`).
    #[must_use]
    pub fn tx_size(self, size: BlockSize) -> Option<TxSize> {
        let (w, h) = size.log2_units();
        let (w, h) = (w as i8, h as i8);
        if w.max(h) > MAX_LOG2 as i8 {
            return (self == Self::None).then_some(TxSize {
                w: MAX_LOG2,
                h: MAX_LOG2,
            });
        }
        let (dw, dh) = match self {
            Self::None => (0, 0),
            Self::Split => (1, 1),
            Self::Horizontal => (0, 1),
            Self::Vertical => (1, 0),
            Self::Horizontal4 => (0, 2),
            Self::Vertical4 => (2, 0),
            Self::Horizontal5 => (1, 2),
            Self::Vertical5 => (2, 1),
        };
        TxSize::of(w - dw, h - dh)
    }
}

/// How a luma block's transforms are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LumaTransforms {
    /// A lossy block's partition.
    Partition(TxPartition),
    /// A lossless block: 4x4 transforms, or its largest lossless one.
    Lossless {
        /// The block's largest lossless transform instead of 4x4.
        large: bool,
    },
}

impl LumaTransforms {
    /// The partition, which a lossless layout has none of.
    #[must_use]
    pub fn partition(self) -> TxPartition {
        match self {
            Self::Partition(partition) => partition,
            Self::Lossless { .. } => TxPartition::None,
        }
    }
}

/// The transform split context of a block size (dav2d
/// `size_to_tx_part_group_lookup`); 8 for sizes that cannot split.
fn split_context(size: BlockSize) -> usize {
    let (w, h) = size.log2_units();
    match (w, h) {
        (0, 0 | 1) | (1, 0) => 0,
        (1, 1) => 1,
        (1, 2) | (2, 1) => 2,
        (2, 2) => 3,
        (2, 3) | (3, 2) => 4,
        (3, 3) => 5,
        (3, 4) | (4, 3) => 6,
        (4, 4) => 7,
        _ => 8,
    }
}

/// The two-dimensional split type context of a block size (dav2d
/// `size_to_tx_type_group_vh_lookup`).
fn split_type_context(size: BlockSize) -> usize {
    let (w, h) = size.log2_units();
    match (w, h) {
        (1, 1) => 0,
        (1, 2) => 1,
        (2, 1) => 2,
        (2, 2) => 3,
        (2, 3) => 4,
        (3, 2) => 5,
        (3, 3) => 6,
        (3, 4) => 7,
        (4, 3) => 8,
        (4, 4) => 9,
        (1, 3 | 4) => 10,
        (3 | 4, 1) => 11,
        (2, 4) => 12,
        _ => 13,
    }
}

/// Splits up to this side (64 samples) are signalled.
const SPLIT_MAX_UNITS: u32 = 16;

impl Tile<'_, '_> {
    /// The luma transform layout of a block (dav2d `read_tx_part`): its
    /// partition, or a lossless block's transform size choice.
    pub(super) fn transform_partition(
        &mut self,
        depth: usize,
        size: BlockSize,
        record: &BlockRecord,
        fsc: bool,
    ) -> Result<LumaTransforms> {
        let (bw4, bh4) = size.units();
        let inter = !record.intra || record.intrabc;
        if self.lossless() {
            let coded = if record.intra && !record.intrabc {
                fsc
            } else {
                !record.skip_txfm
            };
            let mut large = false;
            if size != BlockSize::B4x4 && coded {
                let group = super::intra::size_group(size);
                large = self.bool(|m| &mut m.lossless_tx_size[group][usize::from(inter)])?;
                let range = self.reader.range();
                trace!(
                    self,
                    depth,
                    "Post-lltxsz[ctx={group}|{},{}]: r={range}",
                    u8::from(inter),
                    u8::from(large)
                );
            }
            return Ok(LumaTransforms::Lossless { large });
        }
        let mut partition = TxPartition::None;
        if !record.skip_txfm
            && self.frame.tx_mode_select
            && size != BlockSize::B4x4
            && bw4.max(bh4) <= SPLIT_MAX_UNITS
        {
            let (f, i) = (usize::from(fsc), usize::from(inter));
            let context = split_context(size);
            if self.bool(|m| &mut m.tx_split[f][i][context])? {
                partition = if bw4.min(bh4) >= 2 {
                    let ctx = split_type_context(size);
                    let kind = if self.sequence.transform.reduced_partition_set {
                        self.symbol(|m| &mut m.tx_part_2d_reduced[f][i][ctx])?
                    } else {
                        self.symbol(|m| &mut m.tx_part_2d[f][i][ctx])?
                    };
                    TxPartition::ALL[1 + kind]
                } else if bw4.max(bh4) >= 4 {
                    let wide = usize::from(bw4 >= 4);
                    let four = !self.sequence.transform.reduced_partition_set
                        && self.bool(|m| &mut m.tx_part_1d[f][i][wide])?;
                    TxPartition::ALL[2 + wide + 2 * usize::from(four)]
                } else if size == BlockSize::B4x8 {
                    TxPartition::Horizontal
                } else {
                    TxPartition::Vertical
                };
            }
            let range = self.reader.range();
            trace!(
                self,
                depth,
                "Post-txpart[ctx={f}|{i}|{context},{}]: r={range}",
                TxPartition::ALL
                    .iter()
                    .position(|&p| p == partition)
                    .unwrap_or(0)
            );
        }
        if partition.tx_size(size).is_none() {
            return Err(DecodeError::Malformed("AV2 transform partition"));
        }
        Ok(LumaTransforms::Partition(partition))
    }
}
