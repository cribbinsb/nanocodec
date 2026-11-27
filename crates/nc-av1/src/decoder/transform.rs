//! Transform sizes and transform types as the decoder's syntax sees them
//! (AV1 §6.10.19, §5.11.47): the 19 sizes with the derived quantities the
//! coefficient syntax indexes by (`Tx_Size_Sqr`, `Tx_Size_Sqr_Up`, the
//! coded 32x32-capped area), the transform-type sets (`get_tx_set`) and the
//! symbol-to-type mapping of each set.
//!
//! The set tables follow libaom `av1/common/blockd.h` and `entropymode.h`
//! (`av1_ext_tx_set_lookup`, `av1_ext_tx_used`, `av1_ext_tx_inv`;
//! BSD-2-Clause, Copyright (c) 2016, Alliance for Open Media; see
//! `THIRD_PARTY.md`).
use super::block_size::BlockSize;
use nc_dsp::Av1TxType;

pub(crate) use crate::tx_size::TxSize;

/// log2 of the smallest transform side (`TX_4X4`).
const MIN_SIDE_LOG2: u8 = 2;
/// `Tx_Size_Sqr_Up` of `TX_32X32`, log2.
const SIDE_32_LOG2: u8 = 5;
/// `Tx_Size_Sqr` of `TX_16X16`, log2.
const SIDE_16_LOG2: u8 = 4;
/// The coefficient CDF sets indexed by `txSzCtx` (`TX_SIZES`).
pub(super) const TX_SIZE_CONTEXTS: usize = 5;
/// The largest transform side (64), log2.
const MAX_SIDE_LOG2: u8 = 6;

/// The sizes of a block's transforms.
impl TxSize {
    /// `Max_Tx_Size_Rect`: the largest transform of a block (64 at most).
    pub(crate) fn largest(block: BlockSize) -> Self {
        Self::new(
            block.width_log2().min(MAX_SIDE_LOG2),
            block.height_log2().min(MAX_SIDE_LOG2),
        )
        .expect("a block's largest transform is a transform size")
    }
    /// The chroma transform of a block whose chroma is subsampled by
    /// `subsampling` (`get_tx_size`): the largest of its chroma block
    /// (`get_plane_residual_size`, at least 4x4), where 64-sample sides
    /// become 32.
    pub(crate) fn chroma(block: BlockSize, subsampling: (usize, usize)) -> Self {
        let (w, h) = chroma_block(block, subsampling);
        let (w, h) = (w.min(MAX_SIDE_LOG2), h.min(MAX_SIDE_LOG2));
        let (w, h) = match (w, h) {
            (w, MAX_SIDE_LOG2) if w < SIDE_32_LOG2 => (w, SIDE_32_LOG2),
            (MAX_SIDE_LOG2, h) if h < SIDE_32_LOG2 => (SIDE_32_LOG2, h),
            (w, h) if w == MAX_SIDE_LOG2 || h == MAX_SIDE_LOG2 => (SIDE_32_LOG2, SIDE_32_LOG2),
            size => size,
        };
        Self::new(w, h).expect("a chroma transform is a transform size")
    }
}

/// A block's chroma size, log2, in a format subsampled by `subsampling`
/// (`get_plane_residual_size`: the luma size shifted down per axis, at
/// least 4 samples; the 4:2:2 sizes the table marks invalid are refused
/// at the partition).
pub(crate) fn chroma_block(block: BlockSize, (ss_x, ss_y): (usize, usize)) -> (u8, u8) {
    (
        (block.width_log2() - ss_x as u8).max(MIN_SIDE_LOG2),
        (block.height_log2() - ss_y as u8).max(MIN_SIDE_LOG2),
    )
}

/// The transform types a block may use (`TxSetType`, §5.11.48
/// `get_tx_set`), in libaom's `TxSetType` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TxSet {
    /// `DCT_DCT` only: nothing is coded.
    DctOnly = 0,
    /// `DCT_DCT` and `IDTX`.
    DctIdtx = 1,
    /// The four DCT/ADST combinations and `IDTX`.
    Dtt4Idtx = 2,
    /// [`TxSet::Dtt4Idtx`] plus `V_DCT` and `H_DCT`.
    Dtt4Idtx1dDct = 3,
    /// The nine DCT/ADST/FLIPADST combinations, `IDTX`, `V_DCT`, `H_DCT`.
    Dtt9Idtx1dDct = 4,
    /// All sixteen.
    All16 = 5,
}

/// Bit `t` set where set `s` allows type `t` (libaom `av1_ext_tx_used_flag`).
const SET_TYPES: [u16; 6] = [0x0001, 0x0201, 0x020F, 0x0E0F, 0x0FFF, 0xFFFF];
/// Symbols of each set (`av1_num_ext_tx_set`).
const SET_SYMBOLS: [usize; 6] = [1, 2, 5, 7, 12, 16];
/// Type of each symbol of each set (`av1_ext_tx_inv`).
const SET_INVERSE: [[u8; 16]; 6] = [
    [0; 16],
    [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 0, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 0, 10, 11, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 10, 11, 0, 1, 2, 4, 5, 3, 6, 7, 8, 0, 0, 0, 0],
    [9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 4, 5, 3, 6, 7, 8],
];
/// The CDF array of each set, intra and inter (`ext_tx_set_index`; `None`
/// where the set is never used on that side).
const SET_CDF_INDEX: [[Option<usize>; 6]; 2] = [
    [None, None, Some(2), Some(1), None, None],
    [None, Some(3), None, None, Some(2), Some(1)],
];

impl TxSet {
    /// The set of a `tx` transform of an intra or inter block (§5.11.48).
    pub(crate) fn of(tx: TxSize, inter: bool, reduced: bool) -> Self {
        let sqr_up = tx.square_up_log2();
        if sqr_up > SIDE_32_LOG2 {
            return Self::DctOnly;
        }
        match (inter, sqr_up == SIDE_32_LOG2, reduced) {
            (true, true, _) | (true, false, true) => Self::DctIdtx,
            (false, true, _) => Self::DctOnly,
            (false, false, true) => Self::Dtt4Idtx,
            (inter, false, false) => {
                let sixteen = tx.square_log2() == SIDE_16_LOG2;
                match (inter, sixteen) {
                    (false, false) => Self::Dtt4Idtx1dDct,
                    (false, true) => Self::Dtt4Idtx,
                    (true, false) => Self::All16,
                    (true, true) => Self::Dtt9Idtx1dDct,
                }
            }
        }
    }
    /// Whether the set allows `kind`.
    pub(crate) fn contains(self, kind: Av1TxType) -> bool {
        SET_TYPES[self as usize] & (1 << kind as u16) != 0
    }
    /// Symbols of the `inter_tx_type` / `intra_tx_type` alphabet.
    pub(crate) fn symbols(self) -> usize {
        SET_SYMBOLS[self as usize]
    }
    /// The type a coded symbol stands for.
    pub(crate) fn kind(self, symbol: usize) -> Av1TxType {
        Av1TxType::ALL[usize::from(SET_INVERSE[self as usize][symbol])]
    }
    /// The CDF array index (`set` of `inter_tx_type[set][txSzSqr]`), and
    /// the transform-size index within it.
    pub(crate) fn cdf_index(self, inter: bool, tx: TxSize) -> Option<(usize, usize)> {
        SET_CDF_INDEX[usize::from(inter)][self as usize].map(|set| (set, tx.type_cdf_size()))
    }
}

/// The scan class of a transform type (libaom `tx_type_to_class`): which
/// neighbours the level contexts read, and the scan order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TxClass {
    /// Two-dimensional (diagonal scan).
    TwoD,
    /// Identity across rows: `H_*` types, scanned column by column.
    Horizontal,
    /// Identity down columns: `V_*` types, scanned row by row.
    Vertical,
}

impl TxClass {
    pub(crate) fn of(kind: Av1TxType) -> Self {
        use Av1TxType::{HAdst, HDct, HFlipadst, VAdst, VDct, VFlipadst};
        match kind {
            VDct | VAdst | VFlipadst => Self::Vertical,
            HDct | HAdst | HFlipadst => Self::Horizontal,
            _ => Self::TwoD,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tx(w: u8, h: u8) -> TxSize {
        TxSize::new(w, h).unwrap()
    }

    #[test]
    fn derived_quantities_follow_the_specification_tables() {
        // txSzCtx: TX_4X8 -> 1, TX_16X64 -> 3, TX_64X64 -> 4.
        assert_eq!(tx(2, 3).context(), 1);
        assert_eq!(tx(4, 6).context(), 3);
        assert_eq!(tx(6, 6).context(), 4);
        // dqDenom: 2 for TX_16X32 and TX_64X16, 4 for TX_32X64.
        assert_eq!(tx(4, 5).dequant_shift(), 1);
        assert_eq!(tx(6, 4).dequant_shift(), 1);
        assert_eq!(tx(5, 6).dequant_shift(), 2);
        assert_eq!(tx(3, 5).dequant_shift(), 0);
        assert_eq!(tx(6, 6).eob_alphabet(), 6);
        assert_eq!(tx(2, 4).eob_alphabet(), 2);
    }

    #[test]
    fn sizes_of_blocks_and_splits() {
        const YUV420: (usize, usize) = (1, 1);
        let block = |w, h| BlockSize::new(w, h).unwrap();
        assert_eq!(TxSize::largest(block(7, 6)), tx(6, 6));
        assert_eq!(TxSize::largest(block(2, 4)), tx(2, 4));
        assert_eq!(TxSize::chroma(block(7, 7), YUV420), tx(5, 5));
        assert_eq!(TxSize::chroma(block(4, 6), YUV420), tx(3, 5));
        assert_eq!(TxSize::chroma(block(7, 6), YUV420), tx(5, 5));
        assert_eq!(TxSize::chroma(block(2, 4), YUV420), tx(2, 3));
        // 4:2:2 and 4:4:4 (libaom `av1_get_adjusted_tx_size` of the plane
        // block's largest transform).
        assert_eq!(TxSize::chroma(block(7, 6), (1, 0)), tx(5, 5));
        assert_eq!(TxSize::chroma(block(6, 4), (1, 0)), tx(5, 4));
        assert_eq!(TxSize::chroma(block(4, 6), (0, 0)), tx(4, 5));
        assert_eq!(TxSize::chroma(block(6, 4), (0, 0)), tx(5, 4));
        assert_eq!(TxSize::chroma(block(5, 6), (0, 0)), tx(5, 5));
        assert_eq!(TxSize::chroma(block(2, 2), (0, 0)), tx(2, 2));
        assert_eq!(tx(6, 6).split(), Some(tx(5, 5)));
        assert_eq!(tx(4, 3).split(), Some(tx(3, 3)));
        assert_eq!(tx(2, 4).split(), Some(tx(2, 3)));
        assert_eq!(tx(6, 4).split(), Some(tx(5, 4)));
        assert_eq!(tx(3, 2).split(), Some(tx(2, 2)));
        assert_eq!(tx(2, 2).split(), None);
    }

    #[test]
    fn sets_follow_get_tx_set() {
        assert_eq!(TxSet::of(tx(2, 2), false, false), TxSet::Dtt4Idtx1dDct);
        assert_eq!(TxSet::of(tx(4, 4), false, false), TxSet::Dtt4Idtx);
        assert_eq!(TxSet::of(tx(4, 5), false, false), TxSet::DctOnly);
        assert_eq!(TxSet::of(tx(2, 4), true, false), TxSet::All16);
        assert_eq!(TxSet::of(tx(4, 5), true, false), TxSet::DctIdtx);
        assert_eq!(TxSet::of(tx(4, 3), true, true), TxSet::DctIdtx);
        assert_eq!(TxSet::of(tx(6, 4), true, false), TxSet::DctOnly);
        // DCT_DCT is the libaom encoder's symbol 1 of the intra sets and 7
        // of ALL16.
        assert_eq!(TxSet::Dtt4Idtx1dDct.kind(1), Av1TxType::DctDct);
        assert_eq!(TxSet::All16.kind(7), Av1TxType::DctDct);
        assert_eq!(TxSet::Dtt9Idtx1dDct.kind(3), Av1TxType::DctDct);
        for set in [TxSet::Dtt4Idtx, TxSet::All16, TxSet::DctIdtx] {
            for symbol in 0..set.symbols() {
                assert!(set.contains(set.kind(symbol)));
            }
        }
        assert_eq!(TxSet::DctIdtx.cdf_index(true, tx(5, 5)), Some((3, 3)));
        assert_eq!(TxSet::Dtt4Idtx.cdf_index(false, tx(4, 5)), Some((2, 2)));
    }
}
