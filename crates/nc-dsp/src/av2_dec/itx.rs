//! Inverse transforms of the general AV2 decoder: every rectangular size
//! from 4x4 to 64x64 with every transform type dav2d supports (DCT, ADST,
//! flipped ADST, identity, the inter DDTs, the lossless WHT and the
//! lossless DPCM residual), added to 8-bit pixels; and the cross-chroma
//! transform (CCTX) that mixes the U and V coefficients before them.
//!
//! Ported from dav2d `src/itx_tmpl.c` (`inv_txfm_add_c`, `residual_add`,
//! `cctx_c`), `src/levels.h` (the size and type numbering), `src/tables.c`
//! (`dav2d_txfm_dimensions`, `dav2d_tx_shift`, `dav2d_tx_ddt_mask`,
//! `dav2d_cctx_angle`) and `src/scan.c` (`dav2d_last_eob_per_col`), commit
//! 237229e; Copyright © 2018-2026 `VideoLAN` and dav2d authors,
//! BSD-2-Clause (`docs/licenses/dav2d`). dav2d is bit-exact with AVM v1.0.0.
//!
//! # Coefficient layout
//!
//! A `w`x`h` transform codes at most `cw = min(w, 32)` by `ch = min(h, 32)`
//! coefficients (64-point sides keep their 32 lowest frequencies). They are
//! stored **column-major**, as dav2d's coefficient reader writes them: the
//! coefficient of horizontal frequency `x` and vertical frequency `y` is
//! `coeffs[x * ch + y]`. [`inverse_transform_add`] reads and then **zeroes**
//! the first `cw * ch` entries (16 for the WHT), so the caller's buffer is
//! clean for the next block, as in dav2d.

use super::itx_1d::{self, MAX_POINTS, Transform1d};

/// Largest coded side ([`MAX_POINTS`]); also the intermediate block's side.
const MAX_CODED: usize = MAX_POINTS;

/// Largest 8-bit sample.
const PIXEL_MAX: i32 = 255;

/// `1 / sqrt(2)` in 8 fractional bits: the extra input scale of a 2:1
/// rectangle, whose 1-D gains multiply to `sqrt(2)` times a square's.
const RECT2_SCALE: i32 = 181;
const RECT2_SHIFT: u32 = 8;

/// The DC-only shortcut folds the two DCT passes' DC gain, `64 · 64 =
/// 2^12`, into one shift (dav2d `inv_txfm_add_c`).
const DC_GAIN_LOG2: u32 = 12;
/// dav2d adds `shift0 - DC_ROUND_BIAS` to the DC rounding term so the
/// shortcut matches the two separately rounded passes.
const DC_ROUND_BIAS: i32 = 6;

/// The lossless WHT's input scale: coefficients carry 2 fractional bits
/// plus the WHT's own gain of 2 (AV1 §7.13.3, `>> UNIT_QUANT_SHIFT`).
const WHT_INPUT_SHIFT: u32 = 3;
/// Side of the (only) lossless WHT size.
const WHT_SIDE: usize = 4;

/// Rows of the first pass are processed in groups of this many before the
/// end of block is checked (dav2d `dav2d_last_eob_per_col`).
const ROW_GROUP: usize = 4;

/// `enum TxfmSize` / `enum RectTxfmSize` of dav2d `src/levels.h`, with the
/// same numbering: the squares first, then the rectangles (`RTX_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[allow(missing_docs)]
pub enum RectTxSize {
    Tx4x4 = 0,
    Tx8x8,
    Tx16x16,
    Tx32x32,
    Tx64x64,
    Tx4x8,
    Tx8x4,
    Tx8x16,
    Tx16x8,
    Tx16x32,
    Tx32x16,
    Tx32x64,
    Tx64x32,
    Tx4x16,
    Tx16x4,
    Tx8x32,
    Tx32x8,
    Tx16x64,
    Tx64x16,
    Tx4x32,
    Tx32x4,
    Tx8x64,
    Tx64x8,
    Tx4x64,
    Tx64x4,
}

/// Number of [`RectTxSize`] values (dav2d `N_RECT_TX_SIZES`).
pub const RECT_TX_SIZES: usize = 25;

/// Per-size constants, by [`RectTxSize`].
#[derive(Clone, Copy)]
struct SizeInfo {
    /// log2 of the width and height in 4-sample units (dav2d `lw`, `lh`).
    log2_w4: u8,
    log2_h4: u8,
    /// Rounding shifts after the row and the column pass (dav2d
    /// `dav2d_tx_shift`, AVM `inv_tx_shift`).
    shifts: [u8; 2],
    /// The type bits that turn (flipped) ADST into (flipped) DDT in inter
    /// blocks (dav2d `dav2d_tx_ddt_mask`).
    ddt_mask: u8,
    /// Start of this size's row-group thresholds in [`LAST_EOB_TABLE`].
    last_eob_offset: u8,
}

const fn info(log2_w4: u8, log2_h4: u8, shifts: [u8; 2], ddt_mask: u8, offset: u8) -> SizeInfo {
    SizeInfo {
        log2_w4,
        log2_h4,
        shifts,
        ddt_mask,
        last_eob_offset: offset,
    }
}

/// dav2d's tables by [`RectTxSize`].
#[rustfmt::skip]
const SIZE_INFO: [SizeInfo; RECT_TX_SIZES] = [
    info(0, 0, [7, 10], 0x00,  1), // 4x4
    info(1, 1, [7, 11], 0x42,  0), // 8x8
    info(2, 2, [6, 13], 0x42, 10), // 16x16
    info(3, 3, [6, 13], 0x00, 38), // 32x32
    info(4, 4, [6, 13], 0x00, 38), // 64x64
    info(0, 1, [7, 10], 0x40,  0), // 4x8
    info(1, 0, [7, 10], 0x02,  1), // 8x4
    info(1, 2, [7, 11], 0x42,  6), // 8x16
    info(2, 1, [7, 11], 0x42,  0), // 16x8
    info(2, 3, [6, 12], 0x02, 30), // 16x32
    info(3, 2, [6, 12], 0x40, 10), // 32x16
    info(3, 4, [6, 12], 0x00, 38), // 32x64
    info(4, 3, [6, 12], 0x00, 38), // 64x32
    info(0, 2, [6, 12], 0x40,  2), // 4x16
    info(2, 0, [6, 12], 0x02,  1), // 16x4
    info(1, 3, [6, 13], 0x02, 22), // 8x32
    info(3, 1, [6, 13], 0x40,  0), // 32x8
    info(2, 4, [6, 13], 0x02, 30), // 16x64
    info(4, 2, [6, 13], 0x40, 10), // 64x16
    info(0, 3, [7, 11], 0x00, 14), // 4x32
    info(3, 0, [7, 11], 0x00,  1), // 32x4
    info(1, 4, [6, 12], 0x02, 22), // 8x64
    info(4, 1, [6, 12], 0x40,  0), // 64x8
    info(0, 4, [6, 13], 0x00, 14), // 4x64
    info(4, 0, [6, 13], 0x00,  1), // 64x4
];

/// Terminates each size's run of thresholds in [`LAST_EOB_TABLE`].
const NO_MORE_ROWS: u16 = u16::MAX;

/// dav2d `dav2d_last_eob_per_col.table`: after each group of
/// [`ROW_GROUP`] rows, the first pass continues only while the end of
/// block exceeds the next threshold (the scan position of the last
/// coefficient in the rows done so far).
#[rustfmt::skip]
const LAST_EOB_TABLE: [u16; 46] = [
    9,                               NO_MORE_ROWS, // 4x8, 8x8, 16x8, 32x8
    9,  25,  41,                     NO_MORE_ROWS, // 4x16
    9,  35,  67,                     NO_MORE_ROWS, // 8x16
    9,  35,  77,                     NO_MORE_ROWS, // 16x16, 32x16
    9,  25,  41,  57,  73,  89, 105, NO_MORE_ROWS, // 4x32
    9,  35,  67,  99, 131, 163, 195, NO_MORE_ROWS, // 8x32
    9,  35,  77, 135, 199, 263, 327, NO_MORE_ROWS, // 16x32
    9,  35,  77, 135, 209, 299, 405, NO_MORE_ROWS, // 32x32
];

impl RectTxSize {
    /// Every size, in dav2d's numbering order.
    pub const ALL: [Self; RECT_TX_SIZES] = [
        Self::Tx4x4,
        Self::Tx8x8,
        Self::Tx16x16,
        Self::Tx32x32,
        Self::Tx64x64,
        Self::Tx4x8,
        Self::Tx8x4,
        Self::Tx8x16,
        Self::Tx16x8,
        Self::Tx16x32,
        Self::Tx32x16,
        Self::Tx32x64,
        Self::Tx64x32,
        Self::Tx4x16,
        Self::Tx16x4,
        Self::Tx8x32,
        Self::Tx32x8,
        Self::Tx16x64,
        Self::Tx64x16,
        Self::Tx4x32,
        Self::Tx32x4,
        Self::Tx8x64,
        Self::Tx64x8,
        Self::Tx4x64,
        Self::Tx64x4,
    ];

    /// The size with dav2d's number `raw`.
    #[must_use]
    pub fn from_raw(raw: u8) -> Option<Self> {
        Self::ALL.get(usize::from(raw)).copied()
    }

    fn info(self) -> SizeInfo {
        SIZE_INFO[self as usize]
    }

    /// log2 of the width in 4-sample units (dav2d `t_dim->lw`).
    #[must_use]
    pub fn log2_width4(self) -> u8 {
        self.info().log2_w4
    }

    /// log2 of the height in 4-sample units (dav2d `t_dim->lh`).
    #[must_use]
    pub fn log2_height4(self) -> u8 {
        self.info().log2_h4
    }

    /// Width in samples.
    #[must_use]
    pub fn width(self) -> usize {
        4 << self.log2_width4()
    }

    /// Height in samples.
    #[must_use]
    pub fn height(self) -> usize {
        4 << self.log2_height4()
    }

    /// Coded width: the width, at most 32.
    #[must_use]
    pub fn coded_width(self) -> usize {
        self.width().min(MAX_CODED)
    }

    /// Coded height: the height, at most 32; also the column stride of the
    /// coefficient buffer (module docs).
    #[must_use]
    pub fn coded_height(self) -> usize {
        self.height().min(MAX_CODED)
    }

    /// dav2d `dav2d_tx_ddt_mask`: the [`TxType`] bits that select the DDT in
    /// place of the ADST in inter blocks; see [`TxType::with_inter_ddt`].
    #[must_use]
    pub fn ddt_mask(self) -> u8 {
        self.info().ddt_mask
    }

    /// Whether the sides differ by a factor of two (an odd log2 area),
    /// which scales the input by `1 / sqrt(2)`.
    fn is_rect2(self) -> bool {
        (self.log2_width4() + self.log2_height4()) & 1 == 1
    }

    /// Rows of the first pass a 2-D class transform with end of block `eob`
    /// must compute: the rest are zero.
    fn rows_2d(self, eob: u16) -> usize {
        let thresholds = &LAST_EOB_TABLE[usize::from(self.info().last_eob_offset)..];
        ROW_GROUP * (1 + thresholds.iter().take_while(|&&t| eob > t).count())
    }
}

/// `enum Tx1dType` of dav2d `src/levels.h`: one direction's kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Tx1d {
    /// The DCT.
    Dct = 0,
    /// The identity (a per-size gain).
    Identity = 1,
    /// AVM's tuned DST-VII.
    Adst = 2,
    /// The flipped ADST.
    FlipAdst = 3,
    /// The inter data-driven transform (8 and 16 points).
    Ddt = 4,
    /// The flipped DDT.
    FlipDdt = 5,
    /// The lossless Walsh-Hadamard transform (4x4 only).
    Wht = 6,
}

impl Tx1d {
    fn from_bits(bits: u16) -> Self {
        match bits {
            0 => Self::Dct,
            1 => Self::Identity,
            2 => Self::Adst,
            3 => Self::FlipAdst,
            4 => Self::Ddt,
            5 => Self::FlipDdt,
            6 => Self::Wht,
            _ => panic!("invalid AV2 1-D transform type {bits}"),
        }
    }
}

/// `enum TxClass` of dav2d `src/levels.h`: which coefficients the scan
/// visits, which bounds the rows the first pass must compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TxClass {
    /// Two-dimensional scan.
    TwoD = 0,
    /// Two-dimensional scan in reverse coefficient order (`IDTX_INV`, the
    /// forward-skip-coding identity).
    TwoDInverse = 1,
    /// Horizontal-only (1-D across rows) types.
    Horizontal = 2,
    /// Vertical-only types.
    Vertical = 3,
}

/// The lossless DPCM direction of an identity or WHT residual (dav2d
/// `txtp >> 8`): the residual is a running sum along rows or columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Dpcm {
    /// Plain residual.
    None = 0,
    /// Each sample adds the residual left of it (horizontal prediction).
    Horizontal = 1,
    /// Each sample adds the residual above it (`VERT_PRED`).
    Vertical = 2,
}

/// dav2d's `enum TxfmType` value as `itxfm_add` receives it, bit for bit:
/// the horizontal [`Tx1d`] in bits 0-2, the [`TxClass`] in bits 3-4, the
/// vertical [`Tx1d`] in bits 5-7 and the lossless [`Dpcm`] direction in
/// bits 8-9. Constants are named as dav2d's, vertical kernel first
/// (`ADST_DCT` is a vertical ADST and a horizontal DCT).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TxType(pub u16);

const HOR_MASK: u16 = 0x7;
const CLASS_SHIFT: u32 = 3;
const CLASS_MASK: u16 = 0x3;
const VER_SHIFT: u32 = 5;
const VER_MASK: u16 = 0x7;
const DPCM_SHIFT: u32 = 8;
/// The plain type (no DPCM) bits (dav2d `txtp & 0xff`).
const TYPE_MASK: u16 = 0xff;

/// dav2d `TX_TYPE_ENUM`.
const fn tx_type(hor: Tx1d, ver: Tx1d, class: TxClass) -> TxType {
    TxType(hor as u16 | (class as u16) << CLASS_SHIFT | (ver as u16) << VER_SHIFT)
}

/// dav2d `TX_TYPE_ENUM_2D`.
const fn two_d(hor: Tx1d, ver: Tx1d) -> TxType {
    tx_type(hor, ver, TxClass::TwoD)
}

#[allow(missing_docs)]
impl TxType {
    pub const DCT_DCT: Self = two_d(Tx1d::Dct, Tx1d::Dct);
    pub const ADST_DCT: Self = two_d(Tx1d::Dct, Tx1d::Adst);
    pub const DCT_ADST: Self = two_d(Tx1d::Adst, Tx1d::Dct);
    pub const ADST_ADST: Self = two_d(Tx1d::Adst, Tx1d::Adst);
    pub const FLIPADST_DCT: Self = two_d(Tx1d::Dct, Tx1d::FlipAdst);
    pub const DCT_FLIPADST: Self = two_d(Tx1d::FlipAdst, Tx1d::Dct);
    pub const FLIPADST_FLIPADST: Self = two_d(Tx1d::FlipAdst, Tx1d::FlipAdst);
    pub const ADST_FLIPADST: Self = two_d(Tx1d::FlipAdst, Tx1d::Adst);
    pub const FLIPADST_ADST: Self = two_d(Tx1d::Adst, Tx1d::FlipAdst);
    pub const IDTX: Self = tx_type(Tx1d::Identity, Tx1d::Identity, TxClass::TwoD);
    pub const IDTX_INV: Self = tx_type(Tx1d::Identity, Tx1d::Identity, TxClass::TwoDInverse);
    pub const V_DCT: Self = tx_type(Tx1d::Identity, Tx1d::Dct, TxClass::Vertical);
    pub const H_DCT: Self = tx_type(Tx1d::Dct, Tx1d::Identity, TxClass::Horizontal);
    pub const V_ADST: Self = tx_type(Tx1d::Identity, Tx1d::Adst, TxClass::Vertical);
    pub const H_ADST: Self = tx_type(Tx1d::Adst, Tx1d::Identity, TxClass::Horizontal);
    pub const V_FLIPADST: Self = tx_type(Tx1d::Identity, Tx1d::FlipAdst, TxClass::Vertical);
    pub const H_FLIPADST: Self = tx_type(Tx1d::FlipAdst, Tx1d::Identity, TxClass::Horizontal);
    pub const WHT_WHT: Self = two_d(Tx1d::Wht, Tx1d::Wht);
    pub const DDT_DCT: Self = two_d(Tx1d::Dct, Tx1d::Ddt);
    pub const FLIPDDT_DCT: Self = two_d(Tx1d::Dct, Tx1d::FlipDdt);
    pub const DDT_IDENTITY: Self = two_d(Tx1d::Identity, Tx1d::Ddt);
    pub const FLIPDDT_IDENTITY: Self = two_d(Tx1d::Identity, Tx1d::FlipDdt);
    pub const DDT_DDT: Self = two_d(Tx1d::Ddt, Tx1d::Ddt);
    pub const FLIPDDT_DDT: Self = two_d(Tx1d::Ddt, Tx1d::FlipDdt);
    pub const DCT_DDT: Self = two_d(Tx1d::Ddt, Tx1d::Dct);
    pub const IDENTITY_DDT: Self = two_d(Tx1d::Ddt, Tx1d::Identity);
    pub const DDT_FLIPDDT: Self = two_d(Tx1d::FlipDdt, Tx1d::Ddt);
    pub const FLIPDDT_FLIPDDT: Self = two_d(Tx1d::FlipDdt, Tx1d::FlipDdt);
    pub const DCT_FLIPDDT: Self = two_d(Tx1d::FlipDdt, Tx1d::Dct);
    pub const IDENTITY_FLIPDDT: Self = two_d(Tx1d::FlipDdt, Tx1d::Identity);
    pub const DDT_ADST: Self = two_d(Tx1d::Adst, Tx1d::Ddt);
    pub const FLIPDDT_ADST: Self = two_d(Tx1d::Adst, Tx1d::FlipDdt);
    pub const DDT_FLIPADST: Self = two_d(Tx1d::FlipAdst, Tx1d::Ddt);
    pub const FLIPDDT_FLIPADST: Self = two_d(Tx1d::FlipAdst, Tx1d::FlipDdt);
    pub const ADST_DDT: Self = two_d(Tx1d::Ddt, Tx1d::Adst);
    pub const ADST_FLIPDDT: Self = two_d(Tx1d::FlipDdt, Tx1d::Adst);
    pub const FLIPADST_DDT: Self = two_d(Tx1d::Ddt, Tx1d::FlipAdst);
    pub const FLIPADST_FLIPDDT: Self = two_d(Tx1d::FlipDdt, Tx1d::FlipAdst);
}

impl TxType {
    /// The horizontal (row) kernel.
    #[must_use]
    pub fn horizontal(self) -> Tx1d {
        Tx1d::from_bits(self.0 & HOR_MASK)
    }

    /// The vertical (column) kernel.
    #[must_use]
    pub fn vertical(self) -> Tx1d {
        Tx1d::from_bits((self.0 >> VER_SHIFT) & VER_MASK)
    }

    /// The scan class.
    #[must_use]
    pub fn class(self) -> TxClass {
        match (self.0 >> CLASS_SHIFT) & CLASS_MASK {
            0 => TxClass::TwoD,
            1 => TxClass::TwoDInverse,
            2 => TxClass::Horizontal,
            _ => TxClass::Vertical,
        }
    }

    /// The lossless DPCM direction.
    #[must_use]
    pub fn dpcm(self) -> Dpcm {
        match self.0 >> DPCM_SHIFT {
            0 => Dpcm::None,
            1 => Dpcm::Horizontal,
            2 => Dpcm::Vertical,
            bits => panic!("invalid AV2 DPCM direction {bits}"),
        }
    }

    /// This type with lossless DPCM `dpcm` (dav2d `recon_tmpl.c`:
    /// `txtp += (1 + (mode == VERT_PRED)) << 8`; only for `IDTX` and
    /// `WHT_WHT`).
    #[must_use]
    pub fn with_dpcm(self, dpcm: Dpcm) -> Self {
        Self((self.0 & TYPE_MASK) | (dpcm as u16) << DPCM_SHIFT)
    }

    /// The inter DDT substitution of dav2d `recon_tmpl.c` when the sequence
    /// enables it: `txtp += txtp & dav2d_tx_ddt_mask[tx]`, which moves a
    /// (flipped) ADST kernel of an 8- or 16-point side to the (flipped) DDT.
    #[must_use]
    pub fn with_inter_ddt(self, size: RectTxSize) -> Self {
        Self(self.0 + (self.0 & u16::from(size.ddt_mask())))
    }
}

/// dav2d `inv_txfm_add_c` (8-bit): inverse transforms the column-major
/// coefficients (module docs) of a `size` block of type `tx_type` (the full
/// dav2d value, DPCM and DDT bits included) and adds the residual to the
/// 8-bit pixels at `dst[offset..]` (rows `stride` apart), clipping to
/// `0..=255`. `eob` is the scan index of the last nonzero coefficient
/// (dav2d's `eob`; after a secondary transform, the value
/// [`super::stx::apply_secondary`] returns): it only lets the first pass
/// skip rows that are zero. Zeroes the coefficients it read.
///
/// # Panics
/// If a kernel of `tx_type` does not exist at `size` (dav2d's syntax never
/// codes one), `WHT_WHT` is not 4x4, or a slice is too short.
pub fn inverse_transform_add(
    dst: &mut [u8],
    offset: usize,
    stride: usize,
    coeffs: &mut [i16],
    size: RectTxSize,
    tx_type: TxType,
    eob: u16,
) {
    let block = Block::new(dst, offset, stride, size.width(), size.height());
    if TxType(tx_type.0 & TYPE_MASK) == TxType::WHT_WHT {
        assert_eq!(size, RectTxSize::Tx4x4, "AV2 WHT_WHT is 4x4 only");
        inverse_wht_add(block, coeffs, tx_type.dpcm());
    } else if eob == 0 && tx_type == TxType::DCT_DCT {
        inverse_dc_add(block, coeffs, size);
    } else {
        inverse_2d_add(block, coeffs, size, tx_type, eob);
    }
}

/// The destination of one transform block.
struct Block<'a> {
    dst: &'a mut [u8],
    offset: usize,
    stride: usize,
    width: usize,
    height: usize,
}

impl<'a> Block<'a> {
    fn new(dst: &'a mut [u8], offset: usize, stride: usize, width: usize, height: usize) -> Self {
        assert!(stride >= width, "AV2 transform stride below its width");
        assert!(
            offset + (height - 1) * stride + width <= dst.len(),
            "AV2 transform block outside its plane"
        );
        Self {
            dst,
            offset,
            stride,
            width,
            height,
        }
    }

    /// Adds `residual(x, y)` to every pixel, clipping to 8 bits (dav2d
    /// `residual_add` and the upsampling loops); with DPCM the residual
    /// accumulates along rows or columns first.
    fn add(self, dpcm: Dpcm, residual: impl Fn(usize, usize) -> i32) {
        let mut column_sums = [0; MAX_CODED];
        for y in 0..self.height {
            let row = &mut self.dst[self.offset + y * self.stride..][..self.width];
            let mut row_sum = 0;
            for (x, pixel) in row.iter_mut().enumerate() {
                let r = residual(x, y);
                let r = match dpcm {
                    Dpcm::None => r,
                    Dpcm::Horizontal => {
                        row_sum += r;
                        row_sum
                    }
                    Dpcm::Vertical => {
                        column_sums[x] += r;
                        column_sums[x]
                    }
                };
                *pixel = (i32::from(*pixel) + r).clamp(0, PIXEL_MAX) as u8;
            }
        }
    }
}

/// dav2d `inv_txfm_add_wht_wht_4x4_c`: the lossless 4x4 WHT, rows then
/// columns, unrounded.
fn inverse_wht_add(block: Block, coeffs: &mut [i16], dpcm: Dpcm) {
    let coeffs = &mut coeffs[..WHT_SIDE * WHT_SIDE];
    let mut tmp = [[0; WHT_SIDE]; WHT_SIDE];
    for (y, row) in tmp.iter_mut().enumerate() {
        for (x, c) in row.iter_mut().enumerate() {
            *c = i32::from(coeffs[x * WHT_SIDE + y]) >> WHT_INPUT_SHIFT;
        }
        itx_1d::wht4(row);
    }
    coeffs.fill(0);
    for x in 0..WHT_SIDE {
        let mut column = tmp.map(|row| row[x]);
        itx_1d::wht4(&mut column);
        for (row, v) in tmp.iter_mut().zip(column) {
            row[x] = v;
        }
    }
    block.add(dpcm, |x, y| tmp[y][x]);
}

/// The DC-only `DCT_DCT` shortcut of dav2d `inv_txfm_add_c`: both passes'
/// DC gains and roundings folded into one constant add.
fn inverse_dc_add(block: Block, coeffs: &mut [i16], size: RectTxSize) {
    let [shift0, shift1] = size.info().shifts.map(u32::from);
    let shift = shift0 + shift1 - DC_GAIN_LOG2;
    let rnd = (1 << (shift - 1)) + shift0 as i32 - DC_ROUND_BIAS;
    let mut dc = i32::from(coeffs[0]);
    coeffs[0] = 0;
    if size.is_rect2() {
        dc = rect2_scale(dc);
    }
    let dc = (dc + rnd) >> shift;
    block.add(Dpcm::None, |_, _| dc);
}

/// dav2d's `1 / sqrt(2)` input scaling of 2:1 rectangles.
fn rect2_scale(c: i32) -> i32 {
    (c * RECT2_SCALE + (1 << (RECT2_SHIFT - 1))) >> RECT2_SHIFT
}

/// AVM's pass rounding: add half (none for a zero shift), shift right.
fn round_shift(v: i32, shift: u32) -> i32 {
    (v + ((1 << shift) >> 1)) >> shift
}

/// The general path of dav2d `inv_txfm_add_c`: rows (only those the end of
/// block can reach), round and clamp to 16 bits, columns, round, add (each
/// residual sample repeated over 2 pixels along a 64-point side).
fn inverse_2d_add(block: Block, coeffs: &mut [i16], size: RectTxSize, tx_type: TxType, eob: u16) {
    let first = kernel(tx_type.horizontal(), size.log2_width4(), tx_type);
    let second = kernel(tx_type.vertical(), size.log2_height4(), tx_type);
    let (cw, ch) = (size.coded_width(), size.coded_height());
    let coeffs = &mut coeffs[..cw * ch];
    let [shift0, shift1] = size.info().shifts.map(u32::from);

    // The row pass's output, row-major `cw` wide; rows not computed stay 0.
    let mut tmp = [0; MAX_CODED * MAX_CODED];
    let rows = first_pass_rows(size, tx_type.class(), eob);
    for (y, row) in tmp.chunks_exact_mut(cw).take(rows).enumerate() {
        for (x, c) in row.iter_mut().enumerate() {
            let coeff = i32::from(coeffs[x * ch + y]);
            *c = if size.is_rect2() {
                rect2_scale(coeff)
            } else {
                coeff
            };
        }
        first(row);
    }
    coeffs.fill(0);
    // 8-bit video clamps the intermediate to 16 bits (dav2d
    // `row_clip_min = INT16_MIN`, AVM `clamp_buf` at `bd + 8`).
    for v in &mut tmp[..cw * ch] {
        *v = round_shift(*v, shift0).clamp(i16::MIN.into(), i16::MAX.into());
    }

    let mut column = [0; MAX_CODED];
    for x in 0..cw {
        for (c, row) in column.iter_mut().zip(tmp.chunks_exact(cw)).take(ch) {
            *c = row[x];
        }
        second(&mut column[..ch]);
        for (&c, row) in column.iter().zip(tmp.chunks_exact_mut(cw)).take(ch) {
            row[x] = c;
        }
    }

    // A 64-point side holds 32 residual samples, each covering 2 pixels.
    let x_up = u32::from(size.width() > cw);
    let y_up = u32::from(size.height() > ch);
    block.add(tx_type.dpcm(), |x, y| {
        round_shift(tmp[(y >> y_up) * cw + (x >> x_up)], shift1)
    });
}

/// The rows of the first pass that can be nonzero (dav2d's loop bounds).
fn first_pass_rows(size: RectTxSize, class: TxClass, eob: u16) -> usize {
    let ch = size.coded_height();
    let rows = match class {
        TxClass::TwoD => size.rows_2d(eob),
        // One row per coefficient up to the end of block.
        TxClass::Horizontal => usize::from(eob).min(ch - 1) + 1,
        // Row-major scan: the end of block's row.
        TxClass::Vertical => (usize::from(eob) >> (size.log2_width4() + 2)) + 1,
        TxClass::TwoDInverse => ch,
    };
    assert!(rows <= ch, "AV2 end of block beyond the transform");
    rows
}

/// The 1-D kernel `kind` along a side of `log2_side4`.
fn kernel(kind: Tx1d, log2_side4: u8, tx_type: TxType) -> Transform1d {
    itx_1d::transform(kind, log2_side4)
        .unwrap_or_else(|| panic!("AV2 {tx_type:?} has no {}-point {kind:?}", 4 << log2_side4))
}

/// dav2d `dav2d_cctx_angle`: `sin`, `cos` and `-sin` of the six CCTX
/// rotations (45, 30, 60, -45, -30, -60 degrees) in 8 fractional bits, by
/// CCTX type minus one.
pub const CCTX_ANGLES: [[i16; 3]; 6] = [
    [181, 181, -181],
    [128, 222, -128],
    [222, 128, -222],
    [-181, 181, 181],
    [-128, 222, 128],
    [-222, 128, 222],
];

/// Fractional bits of the CCTX rotation.
const CCTX_SHIFT: u32 = 8;

/// dav2d `cctx_c` (8-bit): rotates each pair of U and V coefficients by
/// `angle` (a row of [`CCTX_ANGLES`]: `sin`, `cos`, `-sin`), rounding half
/// away from zero and clamping to 16 bits. `u` and `v` hold the coded `min(w, 32) * min(h, 32)`
/// coefficients (a power of two from 16 to 1024).
///
/// # Panics
/// If the lengths differ or are not such a power of two, or `angle[2]` is
/// not `-angle[0]`.
pub fn cross_chroma(u: &mut [i16], v: &mut [i16], angle: &[i16; 3]) {
    assert_eq!(u.len(), v.len(), "CCTX planes differ in size");
    assert!(
        u.len().is_power_of_two() && (16..=MAX_CODED * MAX_CODED).contains(&u.len()),
        "CCTX size {} is not a coded transform area",
        u.len()
    );
    let [sin, cos, minus_sin] = angle.map(i32::from);
    assert_eq!(minus_sin, -sin, "CCTX angle is not a rotation");
    for (u, v) in u.iter_mut().zip(v.iter_mut()) {
        let (cu, cv) = (i32::from(*u), i32::from(*v));
        *u = cctx_round(cu * cos - cv * sin);
        *v = cctx_round(cu * sin + cv * cos);
    }
}

/// dav2d's CCTX rounding, half away from zero (`(a + 128 - (a < 0)) >> 8`),
/// then the 16-bit clamp of 8-bit video.
fn cctx_round(a: i32) -> i16 {
    let half = 1 << (CCTX_SHIFT - 1);
    let r = (a + half - i32::from(a < 0)) >> CCTX_SHIFT;
    r.clamp(i16::MIN.into(), i16::MAX.into()) as i16
}

#[cfg(test)]
pub(super) mod tests;
