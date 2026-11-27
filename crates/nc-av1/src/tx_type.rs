//! Intra transform types (AV1 §5.11.47, §6.10.19): the type an intra
//! prediction mode implies (`Mode_To_Txfm`, which the decoder applies to
//! chroma), and the luma types the encoder may code on an intra block.
//!
//! The encoder codes only types of transform class 2D (`DCT_DCT`,
//! `ADST_DCT`, `DCT_ADST`, `ADST_ADST`, `IDTX`): they share `DCT_DCT`'s scan
//! and coefficient contexts (`coeff.rs`), and all five are in both intra
//! sets (`TX_SET_INTRA_1` below 16x16, `TX_SET_INTRA_2` at 16x16). 32x32
//! and larger intra transforms are `DCT_DCT` only.

use nc_dsp::Av1TxType;

use crate::transform::TxSize;

/// The intra types the encoder may code on a luma block up to 16x16.
const CODED_INTRA_TYPES: [Av1TxType; 5] = [
    Av1TxType::DctDct,
    Av1TxType::AdstAdst,
    Av1TxType::AdstDct,
    Av1TxType::DctAdst,
    Av1TxType::Idtx,
];
/// Largest intra transform side with a choice of type (16).
const MAX_TYPED_LOG2: u8 = 4;

/// The transform type of an intra block's chroma (`Mode_To_Txfm`, libaom
/// `intra_mode_to_tx_type`), by `UVMode` (`UV_CFL_PRED` as `DC_PRED`), and
/// the encoder's mode-dependent luma default.
pub(crate) fn intra_mode_tx_type(mode: usize) -> Av1TxType {
    use Av1TxType::{AdstAdst, AdstDct, DctAdst, DctDct};
    const BY_MODE: [Av1TxType; 13] = [
        DctDct, AdstDct, DctAdst, DctDct, AdstAdst, AdstDct, DctAdst, DctAdst, AdstDct, AdstAdst,
        AdstDct, DctAdst, AdstAdst,
    ];
    BY_MODE.get(mode).copied().unwrap_or(DctDct)
}

/// The luma types an intra block of mode `mode` (its `YMode` symbol) may
/// code at `tx`, in the encoder's order: its own choice first (`DCT_DCT`,
/// or with `mode_default` the type `mode` implies), then the type the mode
/// implies, then the rest. A single `DCT_DCT` above 16x16.
pub(crate) fn intra_luma_types(tx: TxSize, mode: usize, mode_default: bool) -> Vec<Av1TxType> {
    if tx.square_up_log2() > MAX_TYPED_LOG2 {
        return vec![Av1TxType::DctDct];
    }
    let implied = intra_mode_tx_type(mode);
    let first = if mode_default {
        implied
    } else {
        Av1TxType::DctDct
    };
    let mut types = vec![first];
    for kind in std::iter::once(implied).chain(CODED_INTRA_TYPES) {
        if !types.contains(&kind) {
            types.push(kind);
        }
    }
    types
}

/// How many types [`intra_luma_types`] lists at `tx`.
pub(crate) fn intra_luma_type_count(tx: TxSize) -> usize {
    if tx.square_up_log2() > MAX_TYPED_LOG2 {
        1
    } else {
        CODED_INTRA_TYPES.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mode_lists_each_coded_type_once_with_its_choice_first() {
        for tx in [TxSize::TX_8X8, TxSize::TX_16X16] {
            for mode in 0..13 {
                for mode_default in [false, true] {
                    let types = intra_luma_types(tx, mode, mode_default);
                    assert_eq!(types.len(), intra_luma_type_count(tx));
                    assert!(CODED_INTRA_TYPES.iter().all(|kind| types.contains(kind)));
                    let first = if mode_default {
                        intra_mode_tx_type(mode)
                    } else {
                        Av1TxType::DctDct
                    };
                    assert_eq!(types[0], first);
                }
            }
        }
        assert_eq!(
            intra_luma_types(TxSize::TX_32X32, 4, true),
            [Av1TxType::DctDct]
        );
    }
}
