//! Transform layout of one square coding block and its luma partition syntax.
//!
//! A lossy luma block is coded either as one transform of the block size or,
//! after a 4-way split, as four transforms of half the side (AVM
//! `TX_PARTITION_SPLIT`; with `reduced_tx_part_set` on, the other partition
//! types are never needed for square blocks). Chroma always uses one
//! transform of the chroma block size, capped at 32×32 (AVM
//! `av2_get_tx_size`). Lossless blocks use 4×4 transforms in every plane
//! (AVM `get_lossless_tx_size`). Within a plane, transforms are coded in
//! raster order, luma first, then U, then V (AVM `decode_token_recon_block`
//! for blocks up to 64×64).
//!
//! The partition syntax reimplements AVM v1.0.0 `read_tx_partition`
//! (`av2/decoder/decodeframe.c`) and its context lookups
//! `size_to_tx_part_group_lookup` and
//! `size_to_tx_type_group_vert_and_horz_lookup` (`av2/common/common_data.h`);
//! see `THIRD_PARTY.md` for attribution and patent terms.
use crate::coeff_tables as t;
use crate::entropy::{Reader, Writer};
use crate::tx_size::TxSize;
use nc_common::decode::{DecodeError, Result};

/// Smallest coding block side (log2): 8×8.
const MIN_BLOCK_LOG2: u32 = 3;
/// Largest coding block side (log2): 64×64.
const MAX_BLOCK_LOG2: u32 = 6;
/// Largest chroma transform side (log2): AVM caps chroma at TX32.
const MAX_CHROMA_TX_LOG2: u32 = 5;
/// The `TX_PARTITION_SPLIT` symbol of the 4-way partition type
/// (AVM codes `partition - 1`).
const SPLIT_TYPE_SYMBOL: usize = 0;

/// Transform sizes of one square coding block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Layout {
    /// Luma block side, log2.
    pub block_log2: u32,
    /// Luma transform size.
    pub luma: TxSize,
    /// Size of every chroma transform.
    pub chroma: TxSize,
}

impl Layout {
    /// The lossy layout of a block of side `1 << block_log2`: one luma
    /// transform of the block size, or four of half its side when `split`.
    #[must_use]
    pub fn lossy(block_log2: u32, split: bool) -> Self {
        Self {
            block_log2,
            luma: TxSize::from_log2(block_log2 - u32::from(split)).expect("square block size"),
            chroma: TxSize::from_log2((block_log2 - 1).min(MAX_CHROMA_TX_LOG2))
                .expect("square chroma size"),
        }
    }

    /// The lossless layout: 4×4 transforms in every plane.
    #[must_use]
    pub fn lossless(block_log2: u32) -> Self {
        Self {
            block_log2,
            luma: TxSize::Tx4,
            chroma: TxSize::Tx4,
        }
    }

    /// Whether the luma block holds more than one transform.
    #[must_use]
    pub fn split(self) -> bool {
        self.luma.log2() < self.block_log2
    }

    /// Side (log2) of `plane`'s block.
    #[must_use]
    pub fn plane_log2(self, plane: usize) -> u32 {
        self.block_log2 - u32::from(plane != 0)
    }

    /// Transform size of `plane`.
    #[must_use]
    pub fn tx(self, plane: usize) -> TxSize {
        if plane == 0 { self.luma } else { self.chroma }
    }

    /// Plane positions of `plane`'s transforms in coding (raster) order, for
    /// the block whose luma origin is `(x, y)`.
    pub fn positions(
        self,
        plane: usize,
        (x, y): (usize, usize),
    ) -> impl Iterator<Item = (usize, usize)> {
        let shift = usize::from(plane != 0);
        let (px, py) = (x >> shift, y >> shift);
        let side = 1 << self.plane_log2(plane);
        let step = self.tx(plane).width();
        (0..side)
            .step_by(step)
            .flat_map(move |row| (0..side).step_by(step).map(move |col| (px + col, py + row)))
    }
}

/// `txfm_do_partition_cdf` group of a square block (AVM
/// `size_to_tx_part_group_lookup`: 8×8 → 1, 16×16 → 3, 32×32 → 5, 64×64 → 7).
fn split_group(block_log2: u32) -> usize {
    2 * (block_log2 - MIN_BLOCK_LOG2) as usize + 1
}

/// `txfm_4way_partition_type_cdf` group of a square block (AVM
/// `size_to_tx_type_group_vert_and_horz_lookup`: 8×8 → 0, 16×16 → 3,
/// 32×32 → 6, 64×64 → 9).
fn type_group(block_log2: u32) -> usize {
    3 * (block_log2 - MIN_BLOCK_LOG2) as usize
}

fn check(block_log2: u32) -> Result<()> {
    if (MIN_BLOCK_LOG2..=MAX_BLOCK_LOG2).contains(&block_log2) {
        Ok(())
    } else {
        Err(DecodeError::Unsupported(
            "AV2 transform partition block size",
        ))
    }
}

/// Write the luma transform partition of a lossy block: unsplit, or the
/// 4-way split. Call after the modes, for intra blocks and non-skipped
/// inter blocks in `TX_MODE_SELECT` frames.
/// # Errors
/// Rejects block sizes outside 8×8..64×64 and propagates entropy errors.
pub fn write(writer: &mut Writer, block_log2: u32, is_inter: bool, split: bool) -> Result<()> {
    check(block_log2)?;
    let (inter, group) = (usize::from(is_inter), split_group(block_log2));
    writer.model(usize::from(split), &t::TX_SPLIT[inter][group], |m| {
        &mut m.tx_split[inter][group]
    })?;
    if split {
        let group = type_group(block_log2);
        writer.model(SPLIT_TYPE_SYMBOL, &t::TX_PARTITION[inter][group], |m| {
            &mut m.tx_partition[inter][group]
        })?;
    }
    Ok(())
}

/// Read the luma transform partition written by [`write`]; true for the split.
/// # Errors
/// Rejects block sizes outside the subset, partition types other than the
/// 4-way split, and truncated data.
pub fn read(reader: &mut Reader<'_>, block_log2: u32, is_inter: bool) -> Result<bool> {
    check(block_log2)?;
    let (inter, group) = (usize::from(is_inter), split_group(block_log2));
    let split = reader.model(&t::TX_SPLIT[inter][group], |m| {
        &mut m.tx_split[inter][group]
    })? != 0;
    if split {
        let group = type_group(block_log2);
        let kind = reader.model(&t::TX_PARTITION[inter][group], |m| {
            &mut m.tx_partition[inter][group]
        })?;
        if kind != SPLIT_TYPE_SYMBOL {
            return Err(DecodeError::Unsupported(
                "AV2 transform partition other than the 4-way split",
            ));
        }
    }
    Ok(split)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a over (symbol, fixed CDF) pairs, as traced from original AVM
    /// for the 8×8 intra and inter splits.
    fn trace(symbols: &[(u64, &[u16])]) -> u64 {
        let mut hash = 14_695_981_039_346_656_037_u64;
        for &(symbol, cdf) in symbols {
            for value in std::iter::once(symbol).chain(cdf.iter().map(|&v| u64::from(v))) {
                hash = (hash ^ value).wrapping_mul(1_099_511_628_211);
            }
        }
        hash
    }

    #[test]
    fn original_avm_8x8_transform_partition_traces() {
        const INTRA: u64 = 6_304_109_073_305_073_772;
        const INTER: u64 = 15_431_112_097_797_065_015;
        let (intra_split, inter_split) = (
            &t::TX_SPLIT[0][split_group(3)],
            &t::TX_SPLIT[1][split_group(3)],
        );
        let partition = |inter: usize| &t::TX_PARTITION[inter][type_group(3)][..];
        assert_eq!(trace(&[(1, intra_split), (0, partition(0))]), INTRA);
        assert_eq!(trace(&[(1, inter_split), (0, partition(1))]), INTER);
    }

    #[test]
    fn groups_follow_avm_lookups() {
        assert_eq!([3, 4, 5, 6].map(split_group), [1, 3, 5, 7]);
        assert_eq!([3, 4, 5, 6].map(type_group), [0, 3, 6, 9]);
    }

    #[test]
    fn every_size_round_trips() {
        let mut writer = Writer::new();
        writer.enable_adaptation(100);
        let cases: Vec<(u32, bool, bool)> = (3..=6)
            .flat_map(|log2| {
                [false, true]
                    .into_iter()
                    .flat_map(move |inter| [(log2, inter, false), (log2, inter, true)])
            })
            .collect();
        for &(log2, inter, split) in &cases {
            write(&mut writer, log2, inter, split).unwrap();
        }
        let packet = writer.finish();
        let mut reader = Reader::new(&packet).unwrap();
        reader.enable_adaptation(100);
        for &(log2, inter, split) in &cases {
            assert_eq!(read(&mut reader, log2, inter).unwrap(), split);
        }
        assert!(write(&mut Writer::new(), 7, false, false).is_err());
    }

    #[test]
    fn layouts_cover_each_plane_in_raster_order() {
        let layout = Layout::lossy(6, true);
        assert_eq!((layout.luma, layout.chroma), (TxSize::Tx32, TxSize::Tx32));
        assert_eq!(
            layout.positions(0, (64, 0)).collect::<Vec<_>>(),
            [(64, 0), (96, 0), (64, 32), (96, 32)]
        );
        assert_eq!(layout.positions(2, (64, 0)).collect::<Vec<_>>(), [(32, 0)]);
        let small = Layout::lossy(3, false);
        assert_eq!(
            (small.luma, small.chroma, small.split()),
            (TxSize::Tx8, TxSize::Tx4, false)
        );
        let lossless = Layout::lossless(4);
        assert_eq!(lossless.positions(0, (0, 0)).count(), 16);
        assert_eq!(lossless.positions(1, (0, 0)).count(), 4);
    }
}
