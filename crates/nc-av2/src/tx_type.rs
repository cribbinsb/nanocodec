//! AV2 transform types (AVM `TX_TYPE`) and the frame's reduced transform set
//! (`reduced_tx_set_used`), which limits the types each block may signal.
//!
//! Follows AVM v1.0.0 `av2/common/enums.h` (`TX_TYPE`), `common_data.h`
//! (`g_hor_tx_type`, `g_ver_tx_type`), `blockd.h` (`av2_get_ext_tx_set_type`,
//! `get_ext_tx_set`, `av2_num_reduced_tx_set`), `entropymode.h`
//! (`av2_ext_tx_inv`, `av2_md_idx2type`) and `decodemv.c`
//! (`av2_read_tx_type`); see `THIRD_PARTY.md`.
//!
//! Our sequence header disables secondary transforms, forward skip coding,
//! inter DDT and cross-component transforms, and restricts chroma to the DCT
//! (`enable_chroma_dctonly`), so only the primary luma type is coded.

use crate::tx_size::TxSize;
use nc_common::decode::{DecodeError, Result};
use nc_dsp::{Av2Kernel, Av2TxType};

/// A primary transform type, in AVM `TX_TYPE` order. The first word names
/// the vertical kernel, the second the horizontal one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TxType {
    /// `DCT_DCT`.
    DctDct,
    /// `ADST_DCT`.
    AdstDct,
    /// `DCT_ADST`.
    DctAdst,
    /// `ADST_ADST`.
    AdstAdst,
    /// `FLIPADST_DCT`.
    FlipadstDct,
    /// `DCT_FLIPADST`.
    DctFlipadst,
    /// `FLIPADST_FLIPADST`.
    FlipadstFlipadst,
    /// `ADST_FLIPADST`.
    AdstFlipadst,
    /// `FLIPADST_ADST`.
    FlipadstAdst,
    /// `IDTX`.
    Idtx,
    /// `V_DCT`: vertical DCT, horizontal identity.
    VDct,
    /// `H_DCT`: vertical identity, horizontal DCT.
    HDct,
    /// `V_ADST`.
    VAdst,
    /// `H_ADST`.
    HAdst,
    /// `V_FLIPADST`.
    VFlipadst,
    /// `H_FLIPADST`.
    HFlipadst,
}

impl TxType {
    /// Every type, in AVM `TX_TYPE` order.
    pub const ALL: [Self; 16] = [
        Self::DctDct,
        Self::AdstDct,
        Self::DctAdst,
        Self::AdstAdst,
        Self::FlipadstDct,
        Self::DctFlipadst,
        Self::FlipadstFlipadst,
        Self::AdstFlipadst,
        Self::FlipadstAdst,
        Self::Idtx,
        Self::VDct,
        Self::HDct,
        Self::VAdst,
        Self::HAdst,
        Self::VFlipadst,
        Self::HFlipadst,
    ];

    /// The 1-D kernels (AVM `g_ver_tx_type` / `g_hor_tx_type`).
    #[must_use]
    pub const fn kernels(self) -> Av2TxType {
        use Av2Kernel::{Adst, Dct, FlipAdst as Flip, Identity as Id};
        let (vertical, horizontal) = match self {
            Self::DctDct => (Dct, Dct),
            Self::AdstDct => (Adst, Dct),
            Self::DctAdst => (Dct, Adst),
            Self::AdstAdst => (Adst, Adst),
            Self::FlipadstDct => (Flip, Dct),
            Self::DctFlipadst => (Dct, Flip),
            Self::FlipadstFlipadst => (Flip, Flip),
            Self::AdstFlipadst => (Adst, Flip),
            Self::FlipadstAdst => (Flip, Adst),
            Self::Idtx => (Id, Id),
            Self::VDct => (Dct, Id),
            Self::HDct => (Id, Dct),
            Self::VAdst => (Adst, Id),
            Self::HAdst => (Id, Adst),
            Self::VFlipadst => (Flip, Id),
            Self::HFlipadst => (Id, Flip),
        };
        Av2TxType {
            vertical,
            horizontal,
        }
    }
}

/// The frame header's `reduced_tx_set_used` choices our subset codes.
/// Chroma and TX64 are DCT-only under each; inter TX32 always uses
/// `EXT_TX_SET_DCT_IDTX` and intra TX32 infers `DCT_DCT`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TxSetMode {
    /// `reduced_tx_set_used = 0`: inter luma TX4/TX8 use `EXT_TX_SET_ALL16`
    /// and TX16 `EXT_TX_SET_DTT9_IDTX_1DDCT`; intra luma TX4..TX16 signal an
    /// index into the mode-dependent `EXT_NEW_TX_SET`, of which our subset
    /// codes only index 0 (`DCT_DCT` in every `av2_md_idx2type` row).
    Full,
    /// `reduced_tx_set_used = 1`: intra luma TX4..TX16 choose `DCT_DCT` or
    /// `ADST_ADST` (AVM `EXT_NEW_TX_SET` cut to `av2_num_reduced_tx_set[0]`
    /// entries, whose second entry is `ADST_ADST` for every mode); inter
    /// luma uses `EXT_TX_SET_DCT_IDTX`.
    IntraDctAdst,
    /// `reduced_tx_set_used = 2`: intra blocks infer `DCT_DCT`; inter luma
    /// uses `EXT_TX_SET_DCT_IDTX`.
    IntraDct,
}

/// How a lossy luma transform block signals its type (AVM
/// `av2_write_tx_type` / `av2_read_tx_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signalling {
    /// Nothing is coded: the type is `DCT_DCT`.
    Inferred,
    /// One symbol indexes these types (inter `inter_ext_tx_cdf`, intra
    /// `intra_ext_tx_cdf[2]`).
    Symbol(&'static [TxType]),
    /// A set flag, then an index within the first
    /// [`INTER_TX_TYPE_INDEX_COUNT`] types or an offset past them.
    SetIndex(InterSet),
    /// A 7-symbol index into the intra mode-dependent `EXT_NEW_TX_SET`
    /// (`intra_ext_tx_cdf[1]`); our subset codes index 0, `DCT_DCT`.
    IntraModeDependent,
}

/// The inter sets of `reduced_tx_set_used = 0` signalled by a set flag and
/// index (AVM `eset` 1 and 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterSet {
    /// `EXT_TX_SET_ALL16` (TX4, TX8).
    All16,
    /// `EXT_TX_SET_DTT9_IDTX_1DDCT` (TX16).
    Dtt9Idtx1dDct,
}

/// Types signalled by an index below this; later types by an offset
/// (AVM `INTER_TX_TYPE_INDEX_COUNT`).
pub const INTER_TX_TYPE_INDEX_COUNT: usize = 8;

impl InterSet {
    /// The set's types by index (AVM `av2_ext_tx_inv`).
    #[must_use]
    pub const fn types(self) -> &'static [TxType] {
        match self {
            Self::All16 => &ALL16_TYPES,
            Self::Dtt9Idtx1dDct => &DTT9_IDTX_1DDCT_TYPES,
        }
    }

    /// The signalling-set axis of the flag and index models (AVM `eset - 1`).
    #[must_use]
    pub const fn model_index(self) -> usize {
        match self {
            Self::All16 => 0,
            Self::Dtt9Idtx1dDct => 1,
        }
    }
}

/// `reduced_tx_set_used` of [`TxSetMode::Full`].
const HEADER_FULL: u32 = 0;
/// `reduced_tx_set_used` of [`TxSetMode::IntraDctAdst`].
const HEADER_INTRA_DCT_ADST: u32 = 1;
/// `reduced_tx_set_used` of [`TxSetMode::IntraDct`].
const HEADER_INTRA_DCT: u32 = 2;
/// Bits of `reduced_tx_set_used`.
pub const HEADER_BITS: u32 = 2;
/// Intra types of [`TxSetMode::IntraDctAdst`] by symbol.
const INTRA_REDUCED_TYPES: [TxType; 2] = [TxType::DctDct, TxType::AdstAdst];
/// The intra type our subset codes in `EXT_NEW_TX_SET` (index 0).
const INTRA_FULL_TYPES: [TxType; 1] = [TxType::DctDct];
/// Inter types of `EXT_TX_SET_DCT_IDTX` by symbol (`av2_ext_tx_inv[1]`).
const INTER_DCT_IDTX_TYPES: [TxType; 2] = [TxType::Idtx, TxType::DctDct];
/// `EXT_TX_SET_ALL16` by index (`av2_ext_tx_inv[EXT_TX_SET_ALL16]`).
const ALL16_TYPES: [TxType; 16] = [
    TxType::Idtx,
    TxType::VDct,
    TxType::HDct,
    TxType::VAdst,
    TxType::HAdst,
    TxType::VFlipadst,
    TxType::HFlipadst,
    TxType::DctDct,
    TxType::AdstDct,
    TxType::DctAdst,
    TxType::FlipadstDct,
    TxType::DctFlipadst,
    TxType::AdstAdst,
    TxType::FlipadstFlipadst,
    TxType::AdstFlipadst,
    TxType::FlipadstAdst,
];
/// `EXT_TX_SET_DTT9_IDTX_1DDCT` by index
/// (`av2_ext_tx_inv[EXT_TX_SET_DTT9_IDTX_1DDCT]`).
const DTT9_IDTX_1DDCT_TYPES: [TxType; 12] = [
    TxType::Idtx,
    TxType::VDct,
    TxType::HDct,
    TxType::DctDct,
    TxType::AdstDct,
    TxType::DctAdst,
    TxType::FlipadstDct,
    TxType::DctFlipadst,
    TxType::AdstAdst,
    TxType::FlipadstFlipadst,
    TxType::AdstFlipadst,
    TxType::FlipadstAdst,
];
/// The types with the 2-D coefficient class (AVM `tx_type_to_class`):
/// [`TxType::ALL`] up to and including `IDTX`. Both [`InterSet`]s contain
/// all of them.
const TWO_D_TYPES: [TxType; 10] = [
    TxType::DctDct,
    TxType::AdstDct,
    TxType::DctAdst,
    TxType::AdstAdst,
    TxType::FlipadstDct,
    TxType::DctFlipadst,
    TxType::FlipadstFlipadst,
    TxType::AdstFlipadst,
    TxType::FlipadstAdst,
    TxType::Idtx,
];

impl TxType {
    /// Whether the type uses the default scan and 2-D coefficient contexts
    /// (AVM `TX_CLASS_2D`); the 1-D `V_*` / `H_*` types need the row and
    /// column scans, which our subset does not implement.
    #[must_use]
    pub fn is_two_dimensional(self) -> bool {
        TWO_D_TYPES.contains(&self)
    }
}

impl TxSetMode {
    /// The `reduced_tx_set_used` value.
    #[must_use]
    pub const fn header_value(self) -> u32 {
        match self {
            Self::Full => HEADER_FULL,
            Self::IntraDctAdst => HEADER_INTRA_DCT_ADST,
            Self::IntraDct => HEADER_INTRA_DCT,
        }
    }

    /// The mode a `reduced_tx_set_used` value selects.
    /// # Errors
    /// Rejects the set our decoder does not implement (3).
    pub fn from_header(value: u32) -> Result<Self> {
        match value {
            HEADER_FULL => Ok(Self::Full),
            HEADER_INTRA_DCT_ADST => Ok(Self::IntraDctAdst),
            HEADER_INTRA_DCT => Ok(Self::IntraDct),
            _ => Err(DecodeError::Unsupported("AV2 reduced transform set")),
        }
    }

    /// How a lossy luma block of transform size `tx` with end of block `eob`
    /// (> 0) signals its type (AVM `av2_get_ext_tx_set_type`,
    /// `get_ext_tx_set`). TX64 is DCT-only; intra TX32 uses
    /// `EXT_TX_SET_DCT_IDTX`, which has no intra entry; intra blocks with
    /// only a DC coefficient infer `DCT_DCT` (`av2_read_tx_type`'s
    /// `dc_skip`).
    #[must_use]
    pub fn signalling(self, tx: TxSize, is_inter: bool, eob: usize) -> Signalling {
        match (is_inter, tx, self) {
            (_, TxSize::Tx64, _) | (false, TxSize::Tx32, _) => Signalling::Inferred,
            (false, _, _) if eob <= 1 => Signalling::Inferred,
            (true, TxSize::Tx4 | TxSize::Tx8, Self::Full) => Signalling::SetIndex(InterSet::All16),
            (true, TxSize::Tx16, Self::Full) => Signalling::SetIndex(InterSet::Dtt9Idtx1dDct),
            (true, _, _) => Signalling::Symbol(&INTER_DCT_IDTX_TYPES),
            (false, _, Self::Full) => Signalling::IntraModeDependent,
            (false, _, Self::IntraDctAdst) => Signalling::Symbol(&INTRA_REDUCED_TYPES),
            (false, _, Self::IntraDct) => Signalling::Inferred,
        }
    }

    /// The luma types our encoder may give a lossy block of transform size
    /// `tx` with more than a DC coefficient: the frame's set limited to the
    /// 2-D types for inter blocks, `DCT_DCT` alone for intra blocks of
    /// [`Self::Full`]; empty when the type is inferred.
    #[must_use]
    pub fn luma_types(self, tx: TxSize, is_inter: bool) -> &'static [TxType] {
        // Any end of block above one: intra blocks then signal like inter.
        const AC_EOB: usize = 2;
        match self.signalling(tx, is_inter, AC_EOB) {
            Signalling::Inferred => &[],
            Signalling::Symbol(types) => types,
            Signalling::SetIndex(_) => &TWO_D_TYPES,
            Signalling::IntraModeDependent => &INTRA_FULL_TYPES,
        }
    }
}
