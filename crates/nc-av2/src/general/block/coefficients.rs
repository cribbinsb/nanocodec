//! One transform's coefficients (dav2d `decode_coefs`; AVM
//! `av2_read_coeffs_txb`, `av2_read_sig_txtype`, `read_tx_type`,
//! `av2_read_cctx_type`, `read_stx_type`): the all-zero flag, the end of
//! block, the transform type, the secondary or cross-chroma transform,
//! the levels in reverse scan order (low and high frequency models, TCQ
//! states, parity hiding), forward skip coding's forward walk, signs, the
//! high-range remainders and the dequantization; and the context later
//! transforms read (level sum and DC sign).
//!
//! Decoder-only: the general layer's reader for every transform size and
//! type (our own encoder's coefficient syntax is `crate::coeff`, square
//! transforms only). The level buffers and their padded, flat indexing
//! follow dav2d exactly, since its neighbour reads past a row's end are
//! what AVM's contexts see.
use super::Tile;
use super::context::NO_COEFFICIENTS;
use super::size::BlockSize;
use super::transform::TxSize;
use crate::general::sample::{Av2Sample, Coefficient};
use nc_common::decode::{DecodeError, Result};
use nc_dsp::Av1Coefficient;

/// dav2d `enum TxfmType`: horizontal 1-D type | class << 3 | vertical
/// 1-D type << 5 (`levels.h`).
pub mod tx_type {
    /// 1-D types.
    const DCT: u16 = 0;
    const IDENTITY: u16 = 1;
    const ADST: u16 = 2;
    const FLIPADST: u16 = 3;
    const WHT: u16 = 6;
    /// Classes.
    pub const CLASS_2D: u16 = 0;
    pub const CLASS_2D_INV: u16 = 1;
    pub const CLASS_H: u16 = 2;
    pub const CLASS_V: u16 = 3;
    /// The class field.
    pub const CLASS_SHIFT: u16 = 3;
    pub const CLASS_MASK: u16 = 3;

    const fn tx(horizontal: u16, vertical: u16, class: u16) -> u16 {
        horizontal | (class << CLASS_SHIFT) | (vertical << 5)
    }
    pub const DCT_DCT: u16 = tx(DCT, DCT, CLASS_2D);
    pub const ADST_DCT: u16 = tx(DCT, ADST, CLASS_2D);
    pub const DCT_ADST: u16 = tx(ADST, DCT, CLASS_2D);
    pub const ADST_ADST: u16 = tx(ADST, ADST, CLASS_2D);
    pub const FLIPADST_DCT: u16 = tx(DCT, FLIPADST, CLASS_2D);
    pub const DCT_FLIPADST: u16 = tx(FLIPADST, DCT, CLASS_2D);
    pub const FLIPADST_FLIPADST: u16 = tx(FLIPADST, FLIPADST, CLASS_2D);
    pub const ADST_FLIPADST: u16 = tx(FLIPADST, ADST, CLASS_2D);
    pub const FLIPADST_ADST: u16 = tx(ADST, FLIPADST, CLASS_2D);
    pub const IDTX: u16 = tx(IDENTITY, IDENTITY, CLASS_2D);
    pub const IDTX_INV: u16 = tx(IDENTITY, IDENTITY, CLASS_2D_INV);
    pub const V_DCT: u16 = tx(IDENTITY, DCT, CLASS_V);
    pub const H_DCT: u16 = tx(DCT, IDENTITY, CLASS_H);
    pub const V_ADST: u16 = tx(IDENTITY, ADST, CLASS_V);
    pub const H_ADST: u16 = tx(ADST, IDENTITY, CLASS_H);
    pub const V_FLIPADST: u16 = tx(IDENTITY, FLIPADST, CLASS_V);
    pub const H_FLIPADST: u16 = tx(FLIPADST, IDENTITY, CLASS_H);
    pub const WHT_WHT: u16 = tx(WHT, WHT, CLASS_2D);
    /// The horizontal (bit 1) and vertical (bit 6) "is an ADST" bits.
    pub const HORIZONTAL_ADST_BIT: u16 = 0x02;
    pub const VERTICAL_ADST_BIT: u16 = 0x40;
    /// The bits of a horizontal or vertical (flip)ADST or identity.
    pub const HORIZONTAL_NOT_DCT_BITS: u16 = 0x03;
    pub const VERTICAL_NOT_DCT_BITS: u16 = 0x60;
    /// The type's low byte; the secondary or cross-chroma transform rides
    /// above it.
    pub const TYPE_MASK: u16 = 0xff;
    pub const EXTRA_SHIFT: u16 = 8;
}

use tx_type::{
    ADST_ADST, ADST_DCT, ADST_FLIPADST, CLASS_2D, CLASS_2D_INV, CLASS_H, CLASS_MASK, CLASS_SHIFT,
    CLASS_V, DCT_ADST, DCT_DCT, DCT_FLIPADST, EXTRA_SHIFT, FLIPADST_ADST, FLIPADST_DCT,
    FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, HORIZONTAL_ADST_BIT, HORIZONTAL_NOT_DCT_BITS,
    IDTX, IDTX_INV, V_ADST, V_DCT, V_FLIPADST, VERTICAL_ADST_BIT, VERTICAL_NOT_DCT_BITS, WHT_WHT,
};

/// What a transform's coefficients decode to.
#[derive(Debug, Clone, Copy)]
pub struct Residual {
    /// The end of block (the last nonzero scan position), `None` when the
    /// transform has no coefficients.
    pub eob: Option<u16>,
    /// The transform type, with the secondary transform (`type | set <<
    /// 2`) or the cross-chroma type above [`tx_type::EXTRA_SHIFT`].
    pub tx_type: u16,
    /// The context the transform leaves (dav2d `res_ctx`).
    pub context: u8,
}

/// The block-level inputs of one transform's coefficients.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools, reason = "the block's flags")]
pub struct TransformInput {
    /// 0 luma, 1 U, 2 V.
    pub plane: usize,
    pub tx: TxSize,
    /// The block size the skip context compares against (luma or chroma
    /// tree size).
    pub block: BlockSize,
    /// Intra prediction (not block copy).
    pub intra: bool,
    /// Forward skip coding of the luma block (or, for chroma of separate
    /// trees, of the co-located luma).
    pub fsc: bool,
    /// The co-located luma's forward skip coding (lossless chroma).
    pub luma_fsc: bool,
    /// Luma and chroma intra modes (dav2d `IntraPredMode`).
    pub y_mode: u8,
    pub uv_mode: u8,
    /// Whether U of this chroma transform has coefficients (V's skip
    /// context).
    pub u_has_coefficients: bool,
    /// The luma transform type an inter chroma transform inherits.
    pub luma_type: u16,
    /// Dequantizer steps: DC, AC.
    pub dequant: [u32; 2],
}

/// The transform types of the intra luma set, by size context, mode and
/// coded index (dav2d `md_idx2type`).
const INTRA_TYPES: [[[u16; 7]; 13]; 3] = intra_types();

#[allow(clippy::too_many_lines, reason = "a table")]
const fn intra_types() -> [[[u16; 7]; 13]; 3] {
    const A: u16 = DCT_DCT;
    const B: u16 = ADST_ADST;
    const C: u16 = ADST_DCT;
    const D: u16 = DCT_ADST;
    [
        [
            [A, B, C, D, ADST_FLIPADST, FLIPADST_ADST, H_ADST],
            [A, B, C, D, ADST_FLIPADST, V_DCT, V_ADST],
            [A, B, C, D, FLIPADST_ADST, H_DCT, H_ADST],
            [A, B, C, D, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, ADST_FLIPADST, FLIPADST_ADST, H_ADST],
            [A, B, C, D, ADST_FLIPADST, V_ADST, V_FLIPADST],
            [A, B, C, D, ADST_FLIPADST, FLIPADST_ADST, H_ADST],
            [A, B, C, D, FLIPADST_ADST, H_DCT, H_ADST],
            [A, B, C, D, ADST_FLIPADST, V_DCT, V_ADST],
            [A, B, C, D, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, ADST_FLIPADST, FLIPADST_ADST, V_ADST],
            [A, B, C, D, ADST_FLIPADST, FLIPADST_ADST, H_ADST],
            [A, B, D, V_DCT, H_DCT, V_ADST, H_ADST],
        ],
        [
            [A, B, C, D, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, ADST_FLIPADST, FLIPADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, FLIPADST_ADST, FLIPADST_DCT, ADST_FLIPADST],
            [A, B, C, D, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, FLIPADST_DCT, FLIPADST_ADST, ADST_FLIPADST],
            [A, B, C, D, DCT_FLIPADST, FLIPADST_FLIPADST, ADST_FLIPADST],
            [A, B, C, D, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, V_DCT, H_DCT, H_ADST],
        ],
        [
            [A, B, C, D, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, FLIPADST_DCT, FLIPADST_ADST, ADST_FLIPADST],
            [A, B, C, D, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, FLIPADST_DCT, FLIPADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, FLIPADST_DCT, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, DCT_FLIPADST, ADST_FLIPADST, FLIPADST_ADST],
            [A, B, C, D, V_DCT, H_DCT, V_ADST],
        ],
    ]
}

/// The type of a long (32 or 64) transform by long-side DCT, which side is
/// long (0 wide, 1 tall) and the short-side index (dav2d
/// `txtp_long_tbl`).
const LONG_TYPES: [[[u16; 4]; 2]; 2] = [
    [
        [V_DCT, V_ADST, V_FLIPADST, IDTX],
        [H_DCT, H_ADST, H_FLIPADST, IDTX],
    ],
    [
        [DCT_DCT, ADST_DCT, FLIPADST_DCT, H_DCT],
        [DCT_DCT, DCT_ADST, DCT_FLIPADST, V_DCT],
    ],
];

/// The inter transform types by coded index: the first set's eight, then
/// the second set's from [`INTER_SECOND_SET`] (dav2d `txtp_inv_tbl[0]`);
/// 16x16 transforms have their own twelve (`txtp_inv_tbl[1]`).
const INTER_TYPES: [u16; 16] = [
    IDTX,
    V_DCT,
    H_DCT,
    V_ADST,
    H_ADST,
    V_FLIPADST,
    H_FLIPADST,
    DCT_DCT,
    ADST_DCT,
    DCT_ADST,
    FLIPADST_DCT,
    DCT_FLIPADST,
    ADST_ADST,
    FLIPADST_FLIPADST,
    ADST_FLIPADST,
    FLIPADST_ADST,
];
const INTER_TYPES_16X16: [u16; 12] = [
    IDTX,
    V_DCT,
    H_DCT,
    DCT_DCT,
    ADST_DCT,
    DCT_ADST,
    FLIPADST_DCT,
    DCT_FLIPADST,
    ADST_ADST,
    FLIPADST_FLIPADST,
    ADST_FLIPADST,
    FLIPADST_ADST,
];
/// The inter second set's indices start here.
const INTER_SECOND_SET: usize = 8;
/// The types of the reduced inter set 3 (dav2d `txtp_dct_idtx_iddct_tbl`).
const INTER_DCT_IDTX_IDDCT: [u16; 4] = [DCT_DCT, V_DCT, H_DCT, IDTX];
/// `reduced_tx_set` values of the inter DCT-or-identity sets, and of the
/// DCT, identity and one-direction DCT set.
const REDUCED_DCT_IDTX: [u8; 2] = [1, 2];
const REDUCED_DCT_IDTX_IDDCT: u8 = 3;
/// Inter end-of-block positions this close to the origin (row + column)
/// have context 1; past the block's extent minus this, context 2.
const INTER_EOB_NEAR: u32 = 2;
const INTER_EOB_FAR_MARGIN: u32 = 4;
/// Inter type contexts count at most 8 units a side.
const INTER_EOB_MAX_UNITS: u32 = 8;
/// The 32x32 transform and the long (32- or 64-long) side, in side logs.
const TX32_LOG2: u8 = 3;
const TX64_LOG2: u8 = 4;

/// The chroma transform type an intra chroma mode implies (dav2d
/// `dav2d_txtp_from_uvmode`).
const CHROMA_TYPES: [u16; 14] = [
    DCT_DCT, ADST_DCT, DCT_ADST, DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, DCT_ADST, ADST_DCT,
    ADST_ADST, ADST_DCT, DCT_ADST, ADST_ADST, DCT_DCT,
];

/// The three largest end-of-block classes escape at 7 with one or two
/// bypass bits.
const EOB_ESCAPE: usize = 7;
/// The secondary transform set by mode and coded index (dav2d
/// `inv_most_probable_stx_mapping` and its `ADST_ADST` variant).
const STX_SETS: [[u8; 7]; 12] = [
    [6, 1, 0, 5, 4, 3, 2],
    [1, 6, 0, 4, 2, 5, 3],
    [1, 6, 0, 4, 2, 5, 3],
    [2, 6, 0, 5, 1, 4, 3],
    [3, 4, 6, 1, 0, 2, 5],
    [4, 1, 3, 6, 0, 5, 2],
    [4, 1, 3, 6, 0, 5, 2],
    [5, 0, 6, 2, 1, 4, 3],
    [5, 0, 6, 2, 1, 4, 3],
    [6, 1, 0, 5, 4, 3, 2],
    [1, 6, 0, 4, 2, 5, 3],
    [1, 6, 0, 4, 2, 5, 3],
];
const STX_SETS_ADST: [[u8; 4]; 12] = [
    [3, 1, 0, 2],
    [1, 3, 0, 2],
    [1, 3, 0, 2],
    [1, 3, 0, 2],
    [0, 2, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 0, 3],
    [1, 0, 3, 2],
    [1, 0, 3, 2],
    [3, 1, 0, 2],
    [1, 3, 0, 2],
    [1, 3, 0, 2],
];
/// `ADST_ADST` sets follow the `DCT_DCT` ones.
const STX_ADST_SET_OFFSET: u8 = 7;
/// Secondary transforms apply to intra blocks below these end-of-block
/// limits (8x8 DCT, min side 8 DCT, otherwise) when the end of block is
/// at least 1.
const STX_EOB_LIMITS: [u16; 3] = [20, 32, 8];
const STX_EOB_LIMIT_ADST: u16 = 20;
/// The skip context of forward skip coding blocks.
const FSC_SKIP_CONTEXT: usize = 9;
/// Coefficient tokens: high-frequency levels saturate at 3, low-frequency
/// at 5, then a range symbol adds up to 3.
const HF_LIMIT: i32 = 3;
const LF_LIMIT: i32 = 5;
/// The range model of the last level beyond the low-frequency limit.
const LF_EOB_RANGE_CONTEXT: usize = 7;
/// Remainder thresholds: low frequencies luma 8 / chroma 5, high 6.
const LUMA_LF_MAX_BASE_RANGE: i32 = 8;
const CHROMA_LF_MAX_BASE_RANGE: i32 = 5;
const HF_MAX_BASE_RANGE: i32 = 6;
/// Forward skip coding's tokens beyond 6 have remainders.
const IDTX_MAX_BASE_RANGE: i32 = 6;
/// Remainder magnitudes are kept in 20 bits, dequantized products in 24.
const TOKEN_MASK: u32 = 0xf_ffff;
const PRODUCT_MASK: u32 = 0xff_ffff;
/// The largest dequantized magnitude is `bd + 7` bits (dav2d `cf_max =
/// ~(~127U << bpc)`: `0x7fff` at 8 bits); this is the 7.
const COEFFICIENT_EXTRA_BITS: u32 = 7;
/// Dequantization rounds by 4 before its shift of 3 (+1 with TCQ, + the
/// size context beyond 2).
const DEQUANT_ROUNDING: u32 = 4;
const DEQUANT_SHIFT: u32 = 3;
const DEQUANT_SIZE_CONTEXT_FREE: usize = 2;
/// Level contexts: the coded sum is capped at 63; the DC sign code sits
/// in bits 6-7 (0 negative, 1 none, 2 positive).
const LEVEL_SUM_MAX: u32 = 63;
const DC_SIGN_SHIFT: u32 = 6;
const DC_SIGN_NONE: u8 = 1 << DC_SIGN_SHIFT;
const DC_SIGN_POSITIVE: u8 = 2 << DC_SIGN_SHIFT;
/// Parity hiding is on for transforms with an end of block above 3; its
/// state counts nonzero levels from bit 16 and keeps the parity in bit 0.
const PH_MIN_EOB: i32 = 3;
const PH_COUNT_ONE: u32 = 0x1_0000;
const PH_ACTIVE_SHIFT: u32 = 18;
/// TCQ is off when the state's sign bit is clear.
const TCQ_OFF: i32 = 0;
/// The level buffer's largest extent: 32 + 2 rows of up to 33 entries.
pub const LEVELS: usize = 34 * 34;
/// The 2D classes' high-to-low frequency boundary in scan positions
/// (luma; chroma 1); the 1-D classes' is twice the coded length (chroma:
/// once).
const LUMA_LOW_FREQUENCIES: i32 = 10;
/// The 1-D classes' level stride.
const ONE_D_STRIDE: usize = 32;

/// An up-right diagonal scan of `w` x `h` (coded, at most 32x32)
/// coefficients, stored column-major (`x * h + y`): every dav2d `scan_*`
/// table.
const fn diagonal_scan<const N: usize>(w: usize, h: usize) -> [u16; N] {
    assert!(N == w * h);
    let mut scan = [0; N];
    let mut index = 0;
    let mut diagonal = 0;
    while diagonal < w + h - 1 {
        let mut x = diagonal.saturating_sub(h - 1);
        while x < w && x <= diagonal {
            let y = diagonal - x;
            scan[index] = (x * h + y) as u16;
            index += 1;
            x += 1;
        }
        diagonal += 1;
    }
    scan
}

/// Every coded scan, by log2 width and height (0..=3 each).
static SCANS: [[&[u16]; 4]; 4] = [
    [
        &diagonal_scan::<16>(4, 4),
        &diagonal_scan::<32>(4, 8),
        &diagonal_scan::<64>(4, 16),
        &diagonal_scan::<128>(4, 32),
    ],
    [
        &diagonal_scan::<32>(8, 4),
        &diagonal_scan::<64>(8, 8),
        &diagonal_scan::<128>(8, 16),
        &diagonal_scan::<256>(8, 32),
    ],
    [
        &diagonal_scan::<64>(16, 4),
        &diagonal_scan::<128>(16, 8),
        &diagonal_scan::<256>(16, 16),
        &diagonal_scan::<512>(16, 32),
    ],
    [
        &diagonal_scan::<128>(32, 4),
        &diagonal_scan::<256>(32, 8),
        &diagonal_scan::<512>(32, 16),
        &diagonal_scan::<1024>(32, 32),
    ],
];

fn scan_for(tx: TxSize) -> &'static [u16] {
    let (w, h) = tx.coded();
    SCANS[usize::from(w)][usize::from(h)]
}

/// The TCQ state after a level (dav2d `tcq_next_state`): bits 1-2 shift
/// down, bit 2 from bit 0 and bit 2 against the level's parity; a state
/// with its sign bit clear (TCQ off) stays 0.
fn tcq_next(state: i32, level: i32) -> i32 {
    (((state & 4) ^ (((level & 1) ^ (state & 1)) << 2)) | ((state & 6) >> 1) | i32::MIN)
        & (state >> 31)
}

/// The quantizer index of TCQ's current state (bit 1).
fn tcq_quantizer(state: i32) -> i32 {
    (state & 2) >> 1
}

impl<T: Av2Sample> Tile<'_, '_, T> {
    /// The largest dequantized magnitude at the sequence's depth (dav2d
    /// `cf_max`; a constant in 8-bit tiles).
    fn coefficient_max(&self) -> u32 {
        (1 << (self.depth().bits() + COEFFICIENT_EXTRA_BITS)) - 1
    }

    /// Decode one transform's coefficients into `coefficients` (dav2d's
    /// layout: column-major at the coded size) given its above and left
    /// level contexts.
    #[allow(clippy::too_many_lines, reason = "one syntax structure, as dav2d's")]
    pub(super) fn coefficients(
        &mut self,
        depth: usize,
        input: &TransformInput,
        (above, left): (&[u8], &[u8]),
        coefficients: &mut [Coefficient<T>],
    ) -> Result<Residual> {
        let chroma = input.plane > 0;
        let tx = input.tx;
        let lossless = self.lossless();
        let range = self.reader.range();
        let (tw, th) = tx.samples();
        trace!(
            self,
            depth - 1,
            "decode_cf[y={},x={},pl={},tx={tw}x{th}]: r={range}",
            self.row,
            self.column,
            input.plane
        );
        let fsc_luma = input.fsc && !chroma && self.sequence.transform.fsc;
        let skip_context = if fsc_luma {
            FSC_SKIP_CONTEXT
        } else {
            skip_context(
                tx,
                input.block,
                above,
                left,
                input.plane,
                input.u_has_coefficients,
            )
        };
        let inter_or_fsc = usize::from(!input.intra || input.fsc);
        let size_context = tx.context();
        let all_zero = if input.plane == 2 {
            self.bool(|m| &mut m.txb_skip_v[skip_context])?
        } else {
            self.bool(|m| &mut m.txb_skip[inter_or_fsc][size_context][skip_context])?
        };
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-all_zero[ctx={}|{size_context}|{skip_context},{}]: r={range}",
            if input.plane == 2 {
                -1
            } else {
                inter_or_fsc as i32
            },
            u8::from(all_zero)
        );
        if all_zero {
            let tx_type = if !chroma && input.fsc {
                IDTX
            } else if lossless {
                if tx == TxSize::TX4 { WHT_WHT } else { IDTX }
            } else {
                DCT_DCT
            };
            return Ok(Residual {
                eob: None,
                tx_type,
                context: NO_COEFFICIENTS,
            });
        }
        let eob = self.end_of_block(depth, tx, chroma, input.intra)?;
        let mut tx_type = self.transform_type(depth, input, eob, lossless)?;
        let class = (tx_type >> CLASS_SHIFT) & CLASS_MASK;
        let (extra, stx) = self.secondary_or_cross(depth, input, tx_type, eob, lossless)?;
        let mut state = LevelState {
            sum: 0,
            dc_sign: DC_SIGN_NONE,
            dequant_shift: u32::from(self.frame.tcq && !chroma && class == CLASS_2D && !lossless)
                + DEQUANT_SHIFT
                + tx.context().saturating_sub(DEQUANT_SIZE_CONTEXT_FREE) as u32,
            // Identity, Walsh-Hadamard and one-dimensional classes are unweighted.
            matrix: if class == CLASS_2D && tx_type != IDTX && tx_type != WHT_WHT {
                self.quantizer_matrix(input.plane, tx)
            } else {
                None
            },
        };
        if self.sequence.transform.fsc && (!input.intra || input.fsc) && tx_type == IDTX && !chroma
        {
            tx_type = IDTX_INV;
            self.forward_skip_levels(depth, input, eob, coefficients, &mut state)?;
        } else {
            let tcq = !chroma && self.frame.tcq && class == CLASS_2D && !lossless;
            let parity = !chroma
                && self.frame.parity_hiding
                && tx_type != IDTX
                && !lossless
                && i32::from(eob) > PH_MIN_EOB;
            let flags = LevelFlags {
                class,
                stx: stx != 0,
                tcq,
                parity,
            };
            let dc_sign_context = if chroma {
                0
            } else {
                dc_sign_context(tx, above, left)
            };
            self.levels(
                depth,
                input,
                eob,
                flags,
                dc_sign_context,
                coefficients,
                &mut state,
            )?;
        }
        Ok(Residual {
            eob: Some(eob),
            tx_type: tx_type | (extra << EXTRA_SHIFT),
            context: (state.sum.min(LEVEL_SUM_MAX) as u8) | state.dc_sign,
        })
    }

    /// The end of block: a class symbol, then its high bit and the rest
    /// in bypass bits (dav2d `decode_coefs` "find end-of-block").
    fn end_of_block(&mut self, depth: usize, tx: TxSize, chroma: bool, intra: bool) -> Result<u16> {
        let (slw, slh) = tx.coded();
        let class_context = slw + slh;
        let context = if chroma { 2 } else { usize::from(!intra) };
        let mut class = match class_context {
            0 => self.symbol(|m| &mut m.eob16[context])?,
            1 => self.symbol(|m| &mut m.eob32[context])?,
            2 => self.symbol(|m| &mut m.eob64[context])?,
            3 => self.symbol(|m| &mut m.eob128[context])?,
            4 => self.symbol(|m| &mut m.eob256[context])?,
            5 => self.symbol(|m| &mut m.eob512[context])?,
            _ => self.symbol(|m| &mut m.eob1024[context])?,
        };
        if class == EOB_ESCAPE && class_context >= 4 {
            let bits = if class_context == 4 { 1 } else { 2 };
            class += self.reader.literal(bits)? as usize;
            if class_context == 5 && class == 10 {
                return Err(DecodeError::Malformed("AV2 end-of-block class"));
            }
        }
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-eob_bin_{}[ctx={context},{class}]: r={range}",
            16 << class_context
        );
        let mut eob = class as u32;
        if class > 1 {
            let high = u32::from(self.bool(|m| &mut m.eob_extra)?);
            let low_bits = class as u32 - 2;
            eob = high | 2;
            if low_bits > 0 {
                eob = (eob << low_bits) | self.reader.literal(low_bits)?;
            }
            let range = self.reader.range();
            trace!(self, depth, "Post-eob[{eob}]: r={range}");
        }
        if eob >= 16 << class_context {
            return Err(DecodeError::Malformed("AV2 end of block"));
        }
        Ok(eob as u16)
    }

    /// The transform type: implied for chroma, lossless and large
    /// transforms, coded for intra luma.
    fn transform_type(
        &mut self,
        depth: usize,
        input: &TransformInput,
        eob: u16,
        lossless: bool,
    ) -> Result<u16> {
        let tx = input.tx;
        let chroma = input.plane > 0;
        let tx_type = if lossless {
            if chroma {
                if input.intra {
                    if input.luma_fsc { IDTX } else { WHT_WHT }
                } else {
                    input.luma_type & !(CLASS_2D_INV << CLASS_SHIFT)
                }
            } else if input.intra {
                if input.fsc { IDTX } else { WHT_WHT }
            } else if tx.max() == 0 {
                if self.bool(|m| &mut m.tx_type_lossless)? {
                    IDTX
                } else {
                    WHT_WHT
                }
            } else {
                IDTX
            }
        } else if chroma {
            self.chroma_type(input)
        } else if input.intra {
            self.intra_luma_type(input, eob)?
        } else {
            self.inter_luma_type(input, eob)?
        };
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-txtp[{}/{}]: r={range}",
            one_d_name(tx_type & 7),
            one_d_name((tx_type >> 5) & 7)
        );
        Ok(tx_type)
    }

    /// A chroma transform's type: DCT only, or the intra mode's or the
    /// luma's type where the size allows it.
    fn chroma_type(&self, input: &TransformInput) -> u16 {
        if self.sequence.transform.chroma_dct_only {
            return DCT_DCT;
        }
        let tx_type = if input.intra {
            CHROMA_TYPES[usize::from(input.uv_mode)]
        } else {
            input.luma_type
        };
        // The chroma size's transform set (AVM v1.0.0 `av2_get_tx_type`
        // takes DCT_DCT for a type outside it; dav2d ec810ff1): no
        // (flip)ADST from 32 samples, no identity along a 64-sample side,
        // only DCT_DCT and IDTX at 32x32, only DCT_DCT from 32x64 up.
        let (tw, th) = (input.tx.w, input.tx.h);
        let excluded = |side: u8, adst: u16, not_dct: u16| {
            side >= TX32_LOG2 && tx_type & if side == TX64_LOG2 { not_dct } else { adst } != 0
        };
        let square16 = input.tx == TxSize { w: 2, h: 2 };
        let square32 = input.tx
            == TxSize {
                w: TX32_LOG2,
                h: TX32_LOG2,
            };
        if excluded(tw, HORIZONTAL_ADST_BIT, HORIZONTAL_NOT_DCT_BITS)
            || excluded(th, VERTICAL_ADST_BIT, VERTICAL_NOT_DCT_BITS)
            || (square16 && ((tx_type & 0x47) == 0x41 || (tx_type & 0xe2) == 0x22))
            || (square32 && (tx_type == V_DCT || tx_type == H_DCT))
            || (tw.min(th) >= TX32_LOG2 && tw.max(th) == TX64_LOG2)
        {
            DCT_DCT
        } else if tx_type == IDTX_INV {
            IDTX
        } else {
            tx_type
        }
    }

    /// An intra luma transform's type.
    fn intra_luma_type(&mut self, input: &TransformInput, eob: u16) -> Result<u16> {
        let tx = input.tx;
        if tx.is_large() {
            return Ok(DCT_DCT);
        }
        if input.fsc {
            return Ok(IDTX);
        }
        if eob == 0 || tx == (TxSize { w: 3, h: 3 }) || self.frame.reduced_tx_set == 2 {
            return Ok(DCT_DCT);
        }
        let short = usize::from(tx.min());
        if tx.max() >= 3 {
            let long_dct = tx.max() == 4 || self.bool(|m| &mut m.tx_type_long32_dct[0])?;
            let index = self.symbol(|m| &mut m.tx_type_intra_short[short])?;
            return Ok(LONG_TYPES[usize::from(long_dct)][usize::from(tx.w < tx.h)][index]);
        }
        let size_context = usize::from(u8::midpoint(tx.w, tx.h));
        let index = if self.frame.reduced_tx_set != 0 {
            self.symbol(|m| &mut m.tx_type_intra_reduced[short])?
        } else {
            self.symbol(|m| &mut m.tx_type_intra[short])?
        };
        Ok(INTRA_TYPES[size_context][usize::from(input.y_mode)][index])
    }

    /// An inter (or block-copy) luma transform's type, with a context of
    /// where the end of block lies (dav2d `decode_coefs`, inter branch).
    fn inter_luma_type(&mut self, input: &TransformInput, eob: u16) -> Result<u16> {
        let tx = input.tx;
        if tx.is_large() {
            return Ok(DCT_DCT);
        }
        let (slw, _) = tx.coded();
        let eob = u32::from(eob);
        let (y, x) = (eob >> (2 + slw), eob & ((4 << slw) - 1));
        let (tw4, th4) = tx.units();
        let extent = 4 * (tw4.min(INTER_EOB_MAX_UNITS) + th4.min(INTER_EOB_MAX_UNITS));
        let context = if x + y < INTER_EOB_NEAR {
            1
        } else if x + y > extent - INTER_EOB_FAR_MARGIN {
            2
        } else {
            0
        };
        let short = usize::from(tx.min());
        let square32 = TxSize {
            w: TX32_LOG2,
            h: TX32_LOG2,
        };
        let dct_or_identity = |dct: bool| if dct { DCT_DCT } else { IDTX };
        if tx == square32 {
            let dct =
                self.bool(|m| &mut m.tx_type_inter_dct_idtx[context][usize::from(TX32_LOG2)])?;
            return Ok(dct_or_identity(dct));
        }
        if tx.max() >= TX32_LOG2 {
            let long_dct = tx.max() == TX64_LOG2 || self.bool(|m| &mut m.tx_type_long32_dct[1])?;
            let index = self.symbol(|m| &mut m.tx_type_inter_short[context][short])?;
            return Ok(LONG_TYPES[usize::from(long_dct)][usize::from(tx.w < tx.h)][index]);
        }
        let reduced = self.frame.reduced_tx_set;
        if REDUCED_DCT_IDTX.contains(&reduced) {
            let dct = self.bool(|m| &mut m.tx_type_inter_dct_idtx[context][short])?;
            return Ok(dct_or_identity(dct));
        }
        if reduced == REDUCED_DCT_IDTX_IDDCT {
            let index = self.symbol(|m| &mut m.tx_type_inter_dct_idtx_iddct[context][short])?;
            return Ok(INTER_DCT_IDTX_IDDCT[index]);
        }
        let square16 = tx == TxSize { w: 2, h: 2 };
        let set = usize::from(square16);
        let second = self.bool(|m| &mut m.tx_type_inter_set[set][context][short])?;
        Ok(match (second, square16) {
            (false, false) => {
                INTER_TYPES[self.symbol(|m| &mut m.tx_type_inter_index[set][context])?]
            }
            (false, true) => {
                INTER_TYPES_16X16[self.symbol(|m| &mut m.tx_type_inter_index[set][context])?]
            }
            (true, false) => {
                INTER_TYPES
                    [INTER_SECOND_SET + self.symbol(|m| &mut m.tx_type_inter_offset1[context])?]
            }
            (true, true) => {
                INTER_TYPES_16X16
                    [INTER_SECOND_SET + self.symbol(|m| &mut m.tx_type_inter_offset2[context])?]
            }
        })
    }

    /// The secondary transform of a luma transform, or the cross-chroma
    /// transform of a U transform: the value stored above the type, and
    /// the secondary kernel (0 none).
    fn secondary_or_cross(
        &mut self,
        depth: usize,
        input: &TransformInput,
        tx_type: u16,
        eob: u16,
        lossless: bool,
    ) -> Result<(u16, u16)> {
        let tx = input.tx;
        let enabled = if input.intra {
            self.sequence.transform.intra_secondary
        } else {
            self.sequence.transform.inter_secondary
        };
        if enabled && input.plane == 0 {
            let mut kernel = false;
            if input.intra {
                if eob >= 1
                    && input.y_mode != super::intra::mode::PAETH
                    && (tx_type == DCT_DCT || tx_type == ADST_ADST)
                {
                    let limit = if tx == (TxSize { w: 1, h: 1 }) && tx_type == DCT_DCT {
                        STX_EOB_LIMITS[0]
                    } else if tx.min() >= 1 {
                        if tx_type == DCT_DCT {
                            STX_EOB_LIMITS[1]
                        } else {
                            STX_EOB_LIMIT_ADST
                        }
                    } else {
                        STX_EOB_LIMITS[2]
                    };
                    kernel = eob < limit;
                }
            } else {
                kernel = tx.min() >= 2 && tx_type == DCT_DCT && (3..32).contains(&eob);
            }
            if !kernel {
                return Ok((0, 0));
            }
            let inter = usize::from(!input.intra);
            let size = usize::from(tx.min());
            let kernel = self.symbol(|m| &mut m.stx[inter][size])? as u16;
            let mut set = 0;
            if kernel != 0 && input.intra {
                let mode = usize::from(input.y_mode);
                set = if tx.min() >= 1 && tx_type == ADST_ADST {
                    let index = self.symbol(|m| &mut m.stx_set_adst)?;
                    STX_SETS_ADST[mode][index]
                } else {
                    let index = self.symbol(|m| &mut m.stx_set)?;
                    STX_SETS[mode][index]
                };
                set += STX_ADST_SET_OFFSET * u8::from(tx_type == ADST_ADST);
            }
            let range = self.reader.range();
            trace!(self, depth, "Post-stx[type={kernel},set={set}]: r={range}");
            return Ok(((u16::from(set) << 2) | kernel, kernel));
        }
        if self.sequence.transform.cctx
            && input.plane == 1
            && eob >= u16::from(input.intra)
            && !lossless
        {
            let cross = self.symbol(|m| &mut m.cctx)? as u16;
            let range = self.reader.range();
            trace!(self, depth, "Post-cctx[{cross}]: r={range}");
            return Ok((cross, 0));
        }
        Ok((0, 0))
    }
}

/// Which level syntax a transform uses.
#[derive(Debug, Clone, Copy)]
struct LevelFlags {
    class: u16,
    stx: bool,
    tcq: bool,
    parity: bool,
}

/// What the level walk accumulates for the context it leaves.
#[derive(Debug, Clone, Copy)]
struct LevelState {
    sum: u32,
    dc_sign: u8,
    dequant_shift: u32,
    /// The quantizer matrix weighting each step (dav2d `qm_tbl`).
    matrix: Option<&'static [u8]>,
}

/// The geometry of the level buffer and scan of one class.
#[derive(Debug, Clone, Copy)]
struct Walk {
    class: u16,
    stride: usize,
    shift: u32,
    shift2: u32,
    mask: u32,
    /// Scan positions below this are low frequencies.
    low: i32,
}

impl Walk {
    /// The coefficient index and level index of scan position `i`.
    fn at(self, scan: &[u16], i: u32) -> (usize, usize, u32) {
        match self.class {
            CLASS_2D => {
                let rc = u32::from(scan[i as usize]);
                (
                    rc as usize,
                    rc as usize,
                    (rc >> self.shift) + (rc & self.mask),
                )
            }
            CLASS_H => {
                let (x, y) = (i & self.mask, i >> self.shift);
                (i as usize, x as usize * self.stride + y as usize, y)
            }
            _ => {
                let (x, y) = (i & self.mask, i >> self.shift);
                let rc = (x << self.shift2) | y;
                (rc as usize, x as usize * self.stride + y as usize, y)
            }
        }
    }
}

impl<T: Av2Sample> Tile<'_, '_, T> {
    /// The level walk of every class but forward skip coding (dav2d
    /// `DECODE_COEFS_CLASS` and the DC that follows).
    #[allow(clippy::too_many_lines, reason = "dav2d's walk")]
    fn levels(
        &mut self,
        depth: usize,
        input: &TransformInput,
        eob: u16,
        flags: LevelFlags,
        dc_sign_context: usize,
        coefficients: &mut [Coefficient<T>],
        state: &mut LevelState,
    ) -> Result<()> {
        let chroma = input.plane > 0;
        let tx = input.tx;
        let (slw, slh) = tx.coded();
        let size_context = tx.context();
        let class_context = u32::from(slw + slh);
        let mut levels = std::mem::take(&mut self.levels);
        let walk = match flags.class {
            CLASS_2D => {
                let stride = 4 << slh;
                levels[..stride * ((4 << slw) + 2)].fill(0);
                Walk {
                    class: CLASS_2D,
                    stride,
                    shift: u32::from(slh) + 2,
                    shift2: 0,
                    mask: (4 << slh) - 1,
                    low: if chroma { 1 } else { LUMA_LOW_FREQUENCIES },
                }
            }
            CLASS_H => {
                levels[..ONE_D_STRIDE * ((4 << slh) + 2)].fill(0);
                Walk {
                    class: CLASS_H,
                    stride: ONE_D_STRIDE,
                    shift: u32::from(slh) + 2,
                    shift2: 0,
                    mask: (4 << slh) - 1,
                    low: (8 << slh) >> u8::from(chroma),
                }
            }
            _ => {
                levels[..ONE_D_STRIDE * ((4 << slw) + 2)].fill(0);
                Walk {
                    class: CLASS_V,
                    stride: ONE_D_STRIDE,
                    shift: u32::from(slw) + 2,
                    shift2: u32::from(slh) + 2,
                    mask: (4 << slw) - 1,
                    low: (8 << slw) >> u8::from(chroma),
                }
            }
        };
        let scan = scan_for(tx);
        let mut tcq_state = if flags.tcq { i32::MIN } else { TCQ_OFF };
        let mut parity: u32 = if flags.parity { 2 } else { 0 };
        let mut hr_average = 0;
        let mut dc_token;
        if eob > 0 {
            let eob = i32::from(eob);
            let context =
                1 + usize::from(eob > 2 << class_context) + usize::from(eob > 4 << class_context);
            let mut high = eob >= walk.low;
            let mut limit = if high { HF_LIMIT } else { LF_LIMIT };
            let mut token = 1 + if high {
                if chroma {
                    self.symbol(|m| &mut m.eob_base_uv_hf[context])?
                } else {
                    self.symbol(|m| &mut m.eob_base_hf[size_context][context])?
                }
            } else if chroma {
                self.symbol(|m| &mut m.eob_base_uv_lf[context])?
            } else {
                self.symbol(|m| &mut m.eob_base_lf[size_context][context])?
            } as i32;
            let has_range = !chroma || high;
            let (rc, level_index, _) = walk.at(scan, eob as u32);
            let mut range_context = None;
            if token == limit && has_range {
                let ctx = if limit == LF_LIMIT {
                    LF_EOB_RANGE_CONTEXT
                } else {
                    0
                };
                range_context = Some(ctx);
                token += self.range_symbol(chroma, high, ctx)? as i32;
                if parity != 0 {
                    parity += PH_COUNT_ONE + token as u32 + u32::from(token == 7);
                }
            } else if parity != 0 {
                parity += PH_COUNT_ONE + token as u32;
            }
            let range = self.reader.range();
            trace!(
                self,
                depth,
                "Post-eob_tok[pos={eob},ctx={size_context}|{context}|{},freq={},plane={},{token}]: r={range}",
                range_context.map_or(-1, |c| c as i32),
                if limit == LF_LIMIT { "lo" } else { "hi" },
                if chroma { "uv" } else { "y" }
            );
            tcq_state = tcq_next(tcq_state, token);
            coefficients[if flags.stx { eob as usize } else { rc }] =
                Av1Coefficient::from_i32(token);
            levels[level_index] = token as i8;
            let mut i = eob - 1;
            loop {
                if i == walk.low - 1 {
                    high = false;
                    limit = LF_LIMIT;
                }
                if i == 0 {
                    break;
                }
                let (rc, level_index, xy) = walk.at(scan, i as u32);
                let (context, hr_context) = lo_context(
                    &levels[level_index..],
                    walk.class,
                    xy,
                    input.plane,
                    walk.stride,
                );
                let tcq = tcq_quantizer(tcq_state) as usize;
                let mut token = if limit == LF_LIMIT {
                    if chroma {
                        self.symbol(|m| &mut m.base_uv_lf[context])?
                    } else {
                        self.symbol(|m| &mut m.base_lf[size_context][context][tcq])?
                    }
                } else if chroma {
                    self.symbol(|m| &mut m.base_uv_hf[context])?
                } else {
                    self.symbol(|m| &mut m.base_hf[size_context][context][tcq])?
                } as i32;
                let has_range = !chroma || high;
                let mut shown_hr = None;
                if token == limit && has_range {
                    shown_hr = Some(hr_context);
                    token += self.range_symbol(chroma, high, hr_context)? as i32;
                    if parity != 0 {
                        parity += PH_COUNT_ONE + token as u32 + u32::from(token == 7);
                    }
                } else if parity != 0 {
                    parity += (token.wrapping_neg() as u32) & (PH_COUNT_ONE | 1);
                }
                let range = self.reader.range();
                trace!(
                    self,
                    depth,
                    "Post-tok[pos={i},ctx={size_context}|{context}|{tcq}|{},freq={},plane={},{token}]: r={range}",
                    shown_hr.map_or(-1, |c| c as i32),
                    if limit == LF_LIMIT { "lo" } else { "hi" },
                    if chroma { "uv" } else { "y" }
                );
                tcq_state = tcq_next(tcq_state, token);
                levels[level_index] = token as i8;
                coefficients[if flags.stx { i as usize } else { rc }] =
                    Av1Coefficient::from_i32(token);
                i -= 1;
            }
            // DC
            if parity >> PH_ACTIVE_SHIFT != 0 {
                let context = parity_context(&levels, walk.class, walk.stride);
                let index = self.symbol(|m| &mut m.parity_dc[context])? as i32;
                dc_token = (index << 1) + (parity & 1) as i32;
                let range = self.reader.range();
                trace!(
                    self,
                    depth,
                    "Post-dc_tok_ph[pos=0,ctx={size_context}|{context}|{index},plane=y,{dc_token}]: r={range}"
                );
            } else {
                let tcq = tcq_quantizer(tcq_state) as usize;
                let (context, hr_context) =
                    lo_context(&levels, walk.class, 0, input.plane, walk.stride);
                dc_token = if limit == LF_LIMIT {
                    if chroma {
                        self.symbol(|m| &mut m.base_uv_lf[context])?
                    } else {
                        self.symbol(|m| &mut m.base_lf[size_context][context][tcq])?
                    }
                } else if chroma {
                    self.symbol(|m| &mut m.base_uv_hf[context])?
                } else {
                    self.symbol(|m| &mut m.base_hf[size_context][context][tcq])?
                } as i32;
                let has_range = !chroma || high;
                let mut shown_hr = None;
                if dc_token == limit && has_range {
                    shown_hr = Some(hr_context);
                    dc_token += self.range_symbol(chroma, high, hr_context)? as i32;
                }
                let range = self.reader.range();
                trace!(
                    self,
                    depth,
                    "Post-dc_tok[pos=0,ctx={size_context}|{context}|{tcq}|{},freq={},plane={},{dc_token}]: r={range}",
                    shown_hr.map_or(-1, |c| c as i32),
                    if limit == LF_LIMIT { "lo" } else { "hi" },
                    if chroma { "uv" } else { "y" }
                );
                tcq_state = if flags.tcq { i32::MIN } else { TCQ_OFF };
            }
            self.signs_and_remainders(
                depth,
                input,
                (eob, walk, scan, flags),
                (&mut tcq_state, &mut hr_average, parity),
                coefficients,
                state,
            )?;
        } else if chroma {
            dc_token = 1 + self.symbol(|m| &mut m.eob_base_uv_lf[0])? as i32;
            let range = self.reader.range();
            trace!(
                self,
                depth,
                "Post-eob_tok[pos=0,ctx={size_context}|0|-1,freq=lo,plane=uv,{dc_token}]: r={range}"
            );
        } else {
            dc_token = 1 + self.symbol(|m| &mut m.eob_base_lf[size_context][0])? as i32;
            let ctx = if flags.class == CLASS_2D {
                0
            } else {
                LF_EOB_RANGE_CONTEXT
            };
            if dc_token == LF_LIMIT {
                dc_token += self.symbol(|m| &mut m.range_lf[ctx])? as i32;
            }
            let range = self.reader.range();
            trace!(
                self,
                depth,
                "Post-eob_tok[pos=0,ctx={size_context}|0|{},freq=lo,plane=y,{dc_token}]: r={range}",
                if dc_token < LF_LIMIT { -1 } else { ctx as i32 }
            );
        }
        self.levels = levels;
        if dc_token != 0 {
            self.dc(
                depth,
                input,
                dc_token,
                (tcq_state, hr_average, parity),
                dc_sign_context,
                coefficients,
                state,
            )?;
        }
        Ok(())
    }

    /// A low-range symbol: luma high or low frequency, or chroma high.
    fn range_symbol(&mut self, chroma: bool, high: bool, context: usize) -> Result<usize> {
        match (chroma, high) {
            (true, _) => self.symbol(|m| &mut m.range_uv_hf[context]),
            (false, true) => self.symbol(|m| &mut m.range_hf[context]),
            (false, false) => self.symbol(|m| &mut m.range_lf[context]),
        }
    }

    /// The signs, remainders and dequantization of the AC levels, from the
    /// end of block down (dav2d's second loop of `DECODE_COEFS_CLASS`).
    #[allow(clippy::type_complexity, reason = "the walk's running state")]
    fn signs_and_remainders(
        &mut self,
        depth: usize,
        input: &TransformInput,
        (eob, walk, scan, flags): (i32, Walk, &[u16], LevelFlags),
        (tcq_state, hr_average, parity): (&mut i32, &mut i32, u32),
        coefficients: &mut [Coefficient<T>],
        state: &mut LevelState,
    ) -> Result<()> {
        let chroma = input.plane > 0;
        let tcq_on = u32::from(flags.tcq);
        for i in (1..=eob).rev() {
            let (rc, y) = match walk.class {
                CLASS_2D => {
                    let rc = if flags.stx {
                        i as usize
                    } else {
                        usize::from(scan[i as usize])
                    };
                    (rc, 0)
                }
                CLASS_H => (i as usize, i as u32 >> walk.shift),
                _ => {
                    let (x, y) = (i as u32 & walk.mask, i as u32 >> walk.shift);
                    (((x << walk.shift2) | y) as usize, y)
                }
            };
            let mut token = coefficients[rc].to_i32();
            if token == 0 {
                *tcq_state = tcq_next(*tcq_state, 0);
                continue;
            }
            let sign = if walk.class == CLASS_2D || y > 0 || chroma {
                let sign = self.bypass()?;
                let range = self.reader.range();
                let prefix = if walk.class != CLASS_2D && y == 0 {
                    "dc_"
                } else {
                    ""
                };
                trace!(
                    self,
                    depth,
                    "Post-{prefix}sign[pos={i},{}]: r={range}",
                    u8::from(sign)
                );
                sign
            } else {
                let hidden = usize::from(parity >> PH_ACTIVE_SHIFT != 0);
                let sign = self.bool(|m| &mut m.dc_sign[usize::from(chroma)][hidden][0])?;
                let range = self.reader.range();
                trace!(
                    self,
                    depth,
                    "Post-dc_sign[pos={i},ctx=0,{}]: r={range}",
                    u8::from(sign)
                );
                sign
            };
            let tcq = tcq_quantizer(*tcq_state);
            *tcq_state = tcq_next(*tcq_state, token);
            let max_base_range = if i < walk.low {
                if chroma {
                    CHROMA_LF_MAX_BASE_RANGE
                } else {
                    LUMA_LF_MAX_BASE_RANGE
                }
            } else {
                HF_MAX_BASE_RANGE
            };
            let step = weighted(input.dequant[1], state.matrix, || scan[i as usize]);
            let value = if token >= max_base_range - tcq_on as i32 {
                let remainder = self.high_range(*hr_average)?;
                token += remainder << tcq_on;
                let range = self.reader.range();
                trace!(
                    self,
                    depth, "Post-residual[pos={i},{remainder}->{token}]: r={range}"
                );
                *hr_average = (*hr_average + remainder) >> 1;
                token &= TOKEN_MASK as i32;
                let level = ((token << tcq_on) - tcq) as u32;
                let value = ((level.wrapping_mul(step) & PRODUCT_MASK) + DEQUANT_ROUNDING)
                    >> state.dequant_shift;
                value.min(self.coefficient_max() + u32::from(sign))
            } else {
                let level = ((token << tcq_on) - tcq) as u32;
                (level.wrapping_mul(step) + DEQUANT_ROUNDING) >> state.dequant_shift
            };
            state.sum += token as u32;
            coefficients[rc] = signed(value, sign);
        }
        Ok(())
    }

    /// The DC sign, remainder and dequantization.
    fn dc(
        &mut self,
        depth: usize,
        input: &TransformInput,
        mut token: i32,
        (tcq_state, hr_average, parity): (i32, i32, u32),
        dc_sign_context: usize,
        coefficients: &mut [Coefficient<T>],
        state: &mut LevelState,
    ) -> Result<()> {
        let chroma = input.plane > 0;
        let sign = if chroma {
            let sign = self.bypass()?;
            let range = self.reader.range();
            trace!(
                self,
                depth,
                "Post-dc_sign[pos=0,{}]: r={range}",
                u8::from(sign)
            );
            sign
        } else {
            let hidden = usize::from(parity >> PH_ACTIVE_SHIFT != 0);
            let sign = self.bool(|m| &mut m.dc_sign[0][hidden][dc_sign_context])?;
            let range = self.reader.range();
            trace!(
                self,
                depth,
                "Post-dc_sign[pos=0,ctx={dc_sign_context},{}]: r={range}",
                u8::from(sign)
            );
            sign
        };
        let step = weighted(input.dequant[0], state.matrix, || 0);
        state.dc_sign = if sign { 0 } else { DC_SIGN_POSITIVE };
        let max_base_range = if chroma {
            CHROMA_LF_MAX_BASE_RANGE
        } else {
            LUMA_LF_MAX_BASE_RANGE
        };
        let tcq_on = i32::from(tcq_state != TCQ_OFF);
        let tcq = tcq_quantizer(tcq_state);
        let value = if parity >> PH_ACTIVE_SHIFT != 0 && token > 5 {
            let remainder = self.high_range(hr_average >> 1)? << 1;
            token += remainder;
            let range = self.reader.range();
            trace!(
                self,
                depth, "Post-residual_ph[pos=0,{remainder}->{token}]: r={range}"
            );
            // Masked and clamped as the other high-range path (dav2d
            // b11a8fd0).
            token &= TOKEN_MASK as i32;
            let value = (((token as u32).wrapping_mul(step) & PRODUCT_MASK) + DEQUANT_ROUNDING)
                >> state.dequant_shift;
            value.min(self.coefficient_max() + u32::from(sign))
        } else if token >= max_base_range - tcq_on {
            let remainder = self.high_range(hr_average)?;
            token += remainder << tcq_on;
            let range = self.reader.range();
            trace!(
                self,
                depth, "Post-residual[pos=0,{remainder}->{token}]: r={range}"
            );
            token &= TOKEN_MASK as i32;
            let level = ((token << tcq_on) - tcq) as u32;
            let value = ((level.wrapping_mul(step) & PRODUCT_MASK) + DEQUANT_ROUNDING)
                >> state.dequant_shift;
            value.min(self.coefficient_max() + u32::from(sign))
        } else {
            let level = ((token << tcq_on) - tcq) as u32;
            (level.wrapping_mul(step) + DEQUANT_ROUNDING) >> state.dequant_shift
        };
        state.sum += token as u32;
        coefficients[0] = signed(value, sign);
        Ok(())
    }

    /// Forward skip coding's levels: the first level ("beginning of
    /// block") counted from the end, then every later position forwards,
    /// then signs from the beginning (dav2d's `IDTX_INV` branch).
    #[allow(clippy::too_many_lines, reason = "dav2d's forward walk")]
    fn forward_skip_levels(
        &mut self,
        depth: usize,
        input: &TransformInput,
        eob: u16,
        coefficients: &mut [Coefficient<T>],
        state: &mut LevelState,
    ) -> Result<()> {
        let tx = input.tx;
        let (slw, slh) = tx.coded();
        let class_context = u32::from(slw + slh);
        let stride = 1 + (4 << slh);
        let mut levels = std::mem::take(&mut self.levels);
        levels[..stride * ((4 << slw) + 1)].fill(0);
        let scan = scan_for(tx);
        let size_context = tx.context().min(2);
        let last = (16 << class_context) - 1;
        let bob = last - i32::from(eob);
        let context = usize::from(bob > 2 << class_context) + usize::from(bob > 4 << class_context);
        let mut token = 1 + self.symbol(|m| &mut m.bob_base[size_context][context])? as i32;
        if token == 3 {
            token += self.symbol(|m| &mut m.range_idtx[size_context][0])? as i32;
        }
        let shift = u32::from(slh) + 2;
        let mask = (4 << slh) - 1;
        let at = |rc: u16| {
            let (x, y) = (usize::from(rc) >> shift, usize::from(rc) & mask);
            (1 + x) * stride + 1 + y
        };
        let rc = scan[bob as usize];
        coefficients[usize::from(rc)] = Av1Coefficient::from_i32(token);
        levels[at(rc)] = token as i8;
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-bob_tok[pos={bob},ctx={size_context}|{context}|{},plane=y,{token}]: r={range}",
            if token < 3 { -1 } else { 0 }
        );
        for i in bob + 1..=last {
            let rc = scan[i as usize];
            let index = at(rc);
            let (context, hr_context) = idtx_context(&levels, index, stride);
            let mut token = self.symbol(|m| &mut m.base_idtx[size_context][context])? as i32;
            if token == 3 {
                token += self.symbol(|m| &mut m.range_idtx[size_context][hr_context])? as i32;
            }
            coefficients[usize::from(rc)] = Av1Coefficient::from_i32(token);
            levels[index] = token as i8;
            let range = self.reader.range();
            trace!(
                self,
                depth,
                "Post-tok[pos={i},ctx={size_context}|{context}|{},plane=y,{token}]: r={range}",
                if token < 3 { -1 } else { hr_context as i32 }
            );
        }
        let mut hr_average = 0;
        let step = input.dequant[1];
        // Forward skip coding never uses TCQ's extra shift.
        let shift = state.dequant_shift - u32::from(self.frame.tcq && !self.lossless());
        for i in bob..=last {
            let rc = scan[i as usize];
            let mut token = coefficients[usize::from(rc)].to_i32();
            if token == 0 {
                continue;
            }
            let index = at(rc);
            let context = idtx_sign_context(&levels, index, stride);
            let sign = self.bool(|m| &mut m.sign_idtx[size_context][context])?;
            if i == 0 {
                state.dc_sign = if sign { 0 } else { DC_SIGN_POSITIVE };
            }
            let range = self.reader.range();
            trace!(
                self,
                depth,
                "Post-sign[pos={i},ctx={size_context}|{context},plane=y,{}]: r={range}",
                u8::from(sign)
            );
            levels[index] = if sign { -1 } else { 1 };
            let value = if token >= IDTX_MAX_BASE_RANGE {
                let remainder = self.high_range(hr_average)?;
                token += remainder;
                let range = self.reader.range();
                trace!(
                    self,
                    depth, "Post-residual[pos={i},{remainder}->{token}]: r={range}"
                );
                hr_average = (hr_average + remainder) >> 1;
                token &= TOKEN_MASK as i32;
                let value = (((token as u32).wrapping_mul(step) & PRODUCT_MASK) + DEQUANT_ROUNDING)
                    >> shift;
                value.min(self.coefficient_max() + u32::from(sign))
            } else {
                ((token as u32).wrapping_mul(step) + DEQUANT_ROUNDING) >> shift
            };
            state.sum += token as u32;
            coefficients[usize::from(rc)] = signed(value, sign);
        }
        self.levels = levels;
        Ok(())
    }

    /// A high-range remainder (dav2d `decode_hr`): a unary prefix (zeros
    /// ended by a one) of up to
    /// 5 or 6, then `m` bits or an Exp-Golomb escape, `m` from the running
    /// average.
    fn high_range(&mut self, average: i32) -> Result<i32> {
        const MAX_PREFIX: u32 = 6;
        const PREFIX_BASE: u32 = 4;
        let m = (average.clamp(2, 64) as u32).ilog2();
        let cap = (m + PREFIX_BASE).min(MAX_PREFIX);
        let mut prefix = 0;
        while prefix < cap && !self.bypass()? {
            prefix += 1;
        }
        let remainder = if prefix == cap {
            self.exp_golomb(m + 1)?
        } else {
            self.reader.literal(m)?
        };
        Ok((remainder + (prefix << m)) as i32)
    }

    /// An Exp-Golomb value of order `k` (dav2d `decode_exp_golomb`: a unary
    /// length of at most 21 bypass bits).
    fn exp_golomb(&mut self, k: u32) -> Result<u32> {
        const MAX_LENGTH: u32 = 21;
        let mut length = 0;
        while length < MAX_LENGTH && !self.bypass()? {
            length += 1;
        }
        let length = length + k;
        let value = (1 << length) + self.reader.literal(length)?;
        Ok(value - (1 << k))
    }
}

/// A dequantizer step weighted by a quantizer matrix's entry for the
/// coefficient at `position` (dav2d `ac_dq`, `dc_dq`).
fn weighted(step: u32, matrix: Option<&[u8]>, position: impl FnOnce() -> u16) -> u32 {
    matrix.map_or(step, |matrix| {
        nc_dsp::quantizer_matrix::weighted_step(step, matrix[usize::from(position())])
    })
}

/// A dequantized magnitude with its sign, as dav2d's `coef` store (16
/// bits at 8 bpc, where the magnitude is at most `0x8000`).
fn signed<C: Av1Coefficient>(value: u32, negative: bool) -> C {
    let value = value as i32;
    C::from_i32(if negative { -value } else { value })
}

/// The all-zero flag's context (dav2d `get_skip_ctx`).
fn skip_context(
    tx: TxSize,
    block: BlockSize,
    above: &[u8],
    left: &[u8],
    plane: usize,
    u_has_coefficients: bool,
) -> usize {
    let (tw, th) = tx.units();
    let (above, left) = (&above[..tw as usize], &left[..th as usize]);
    let (bw, bh) = block.log2_units();
    if plane > 0 {
        let not_one = bw.saturating_sub(u8::from(bw > 0)) > tx.w
            || bh.saturating_sub(u8::from(bh > 0)) > tx.h;
        let coded = |edge: &[u8]| usize::from(edge.iter().any(|&c| c != NO_COEFFICIENTS));
        let offset = if plane == 1 {
            6
        } else {
            6 * usize::from(u_has_coefficients) + 3 * usize::from(not_one)
        };
        return offset + coded(above) + coded(left);
    }
    if (bw, bh) == (tx.w, tx.h) {
        return 0;
    }
    let merged = |edge: &[u8]| usize::from((edge.iter().fold(0, |a, &c| a | c) & 0x3f).min(4));
    (merged(above) + merged(left) + 3) >> 1
}

/// The DC sign context (dav2d `get_dc_sign_ctx`): the balance of positive
/// and negative DC signs along both edges.
fn dc_sign_context(tx: TxSize, above: &[u8], left: &[u8]) -> usize {
    let (tw, th) = tx.units();
    let sum: i32 = above[..tw as usize]
        .iter()
        .chain(&left[..th as usize])
        .map(|&c| i32::from(c >> DC_SIGN_SHIFT))
        .sum();
    let balance = sum - (tw + th) as i32;
    usize::from(balance != 0) + usize::from(balance > 0)
}

/// The base level context and the range context of a level (dav2d
/// `get_lo_ctx`); `levels` starts at the level.
fn lo_context(levels: &[i8], class: u16, xy: u32, plane: usize, stride: usize) -> (usize, usize) {
    let chroma = plane > 0;
    let at = |offset: usize| u32::from(levels[offset] as u8);
    let low_frequency = xy
        < if chroma {
            1
        } else if class == CLASS_2D {
            4
        } else {
            2
        };
    let mut limit = if low_frequency { 5 } else { 3 };
    let (mut lo, mut hi) = (0, 0);
    let mut add = |value: u32, limit: u32, lo: &mut u32| {
        *lo += value.min(limit);
        hi += value.min(5);
    };
    add(at(1), limit, &mut lo);
    add(at(stride), limit, &mut lo);
    let offset;
    if class == CLASS_2D {
        add(at(stride + 1), limit, &mut lo);
        if chroma {
            limit = 3;
            offset = if plane == 1 { 0 } else { 4 };
        } else {
            lo += at(2).min(limit) + at(2 * stride).min(limit);
            if low_frequency {
                (offset, limit) = if xy == 0 {
                    (0, 8)
                } else if xy < 2 {
                    (9, 6)
                } else {
                    (16, 4)
                };
            } else {
                offset = if xy < 6 {
                    0
                } else if xy < 8 {
                    5
                } else {
                    10
                };
                limit = 4;
            }
        }
    } else if chroma {
        offset = 8;
        limit = 3;
    } else {
        add(at(2), 3, &mut lo);
        lo += at(3).min(3) + at(4).min(3);
        if low_frequency {
            (offset, limit) = if xy == 0 { (21, 6) } else { (28, 4) };
        } else {
            offset = 15;
            limit = 4;
        }
    }
    let range_offset = if !chroma && low_frequency && (xy > 0 || class != CLASS_2D) {
        7
    } else {
        0
    };
    let range = range_offset + ((hi + 1) >> 1).min(if chroma { 3 } else { 6 });
    (offset + ((lo + 1) >> 1).min(limit) as usize, range as usize)
}

/// Forward skip coding's level context: the left and above neighbours
/// (dav2d `get_lo_ctx_idtx`).
fn idtx_context(levels: &[i8], index: usize, stride: usize) -> (usize, usize) {
    let left = u32::from(levels[index - 1] as u8);
    let above = u32::from(levels[index - stride] as u8);
    let lo = left.min(3) + above.min(3);
    let hi = (left.min(5) + above.min(5)).min(6);
    (lo as usize, hi as usize)
}

/// Forward skip coding's sign context (dav2d `get_sign_ctx_idtx`): the
/// signs of the three causal neighbours (stored as ±1) and the level's
/// size.
fn idtx_sign_context(levels: &[i8], index: usize, stride: usize) -> usize {
    let sum = i32::from(levels[index - 1])
        + i32::from(levels[index - stride])
        + i32::from(levels[index - stride - 1]);
    let offset = if levels[index] > 3 { 2 } else { 0 };
    match sum {
        -3 => offset + 6,
        -2 | -1 => offset + 2,
        1 | 2 => offset + 1,
        3 => offset + 5,
        _ => 0,
    }
}

/// The parity-hidden DC's context (dav2d `get_ph_ctx`).
fn parity_context(levels: &[i8], class: u16, stride: usize) -> usize {
    let at = |offset: usize| u32::from(levels[offset] as u8).min(3);
    let mut sum = at(1) + at(2) + at(stride);
    sum += if class == CLASS_2D {
        at(stride + 1) + at(2 * stride)
    } else {
        at(3) + at(4)
    };
    if sum < 7 {
        ((sum + 1) >> 1) as usize
    } else {
        4
    }
}

/// dav2d `dav2d_tx1d_names` of a 1-D type (for traces; reconstruction's
/// traces use it too).
pub(super) fn one_d_name(kind: u16) -> &'static str {
    [
        "dct", "identity", "adst", "flipadst", "ddt", "flipddt", "wht", "?",
    ][usize::from(kind & 7)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The diagonal rule reproduces dav2d's tabulated scans (spot checks
    /// of `scan_4x4` and `scan_4x8`; tools checked every table).
    #[test]
    fn scans_match_dav2d() {
        assert_eq!(
            diagonal_scan::<16>(4, 4),
            [0, 1, 4, 2, 5, 8, 3, 6, 9, 12, 7, 10, 13, 11, 14, 15]
        );
        assert_eq!(&diagonal_scan::<32>(4, 8)[..8], [0, 1, 8, 2, 9, 16, 3, 10]);
    }
}
