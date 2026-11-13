//! AV1 inverse transforms of every type and size (AV1 §7.13.2–3): the 1-D
//! DCT4–64, ADST4/8/16, flipped ADST and identity4–32, combined into the
//! 2-D inverse of all 19 transform sizes and 16 transform types, added to
//! the prediction with 8-bit clipping.
//!
//! [`crate::Dsp::av1_inverse_add`] is the single entry point. Square
//! `DCT_DCT`, the only kind our own encoder emits, goes to the `txfm`
//! module's tuned kernels (including the EOB-sparse 32/64 graphs). Every
//! other size and type with both sides at least 8 runs the same guarded
//! 16-bit drivers (`txfm::narrow_inverse`) with this module's ADST and
//! identity graphs ([`one_d`], generic over the lane policy); sizes with a
//! 4-sample side, and the scalar reference, run the 32-bit driver of
//! [`generic`]. All DCT sizes share a DC-only shortcut. The generic
//! driver skips rows beyond conservative EOB bounds for each scan class.
//! These shortcuts follow dav1d 1.5.4 `src/itx_tmpl.c` (BSD-2-Clause,
//! `VideoLAN` and dav1d authors; ideas only, see `THIRD_PARTY.md`).
//!
//! Arithmetic follows libaom's `inv_txfm2d_add_c` (BSD-2-Clause, Copyright
//! (c) 2016, Alliance for Open Media; see `THIRD_PARTY.md`), the reference
//! the conformance MD5s come from, for 8-bit video:
//!
//! - **Row pass:** coefficients are clamped to 16 bits (`bd + 8`); for 2:1
//!   rectangles they are first scaled by `1/sqrt(2)`
//!   (`Round2(x * 2896, 12)`, §7.13.3). Then the 1-D row transform and a
//!   rounding shift by the size's [`Av1TxSize::row_shift`].
//! - **Column pass:** row outputs are clamped to 16 bits
//!   (`max(bd + 6, 16)`), then the 1-D column transform and a rounding
//!   shift by [`COLUMN_SHIFT`].
//! - **1-D stages:** every add/sub stage of the DCT and of ADST8/16 is
//!   clamped to 16 bits (libaom's 8-bit stage range); ADST4 and identity
//!   have no stage clamps. For conformant streams none of these clamps
//!   changes a value (§7.13.3 requires every intermediate to fit), but
//!   matching libaom keeps non-conformant input deterministic.
//! - **64-point zero-out (§7.13.3):** only the top-left 32x32 coefficients
//!   of a 64-sample dimension are coded; the rest are zero, so 64-point
//!   passes run the DCT64 graph with its upper 32 inputs removed.
//! - **Flips:** `FLIPADST` vertically reverses the column outputs before the
//!   add (`ud_flip`); horizontally it reverses each row's outputs
//!   (`lr_flip`).
//!
//! # Coefficient layout
//!
//! libaom's column-major layout, as in the `txfm` module: the coded area is
//! `coded_width x coded_height` (each side capped at 32), and coefficient
//! (row `r`, column `c`) is `coeffs[c * coded_height + r]`.
//!
//! # Input range
//!
//! Coefficients are dequantised values, which §7.12.3 clamps to 16 bits.
//! The row pass clamps them to 16 bits before (not after) the rectangular
//! scaling; this differs from libaom only for coefficients the dequantiser
//! cannot produce, and it keeps every SIMD product in 32 bits.

#[allow(unsafe_code)]
mod generic;
#[allow(unsafe_code)]
pub(crate) mod kernels;
pub(crate) mod one_d;
pub(crate) mod wht;

/// Inverse transforms use 12-bit cosines throughout (libaom `INV_COS_BIT`).
const COS_BIT: u32 = 12;
/// Rounding shift after the column pass, every size (libaom
/// `inv_shift_*[1]`, §7.13.3 `colShift`).
const COLUMN_SHIFT: u32 = 4;
/// Largest coded side: 64-point dimensions code only their lowest 32
/// frequencies (§5.11.39, §7.13.3).
pub const AV1_MAX_CODED_SIDE: usize = 32;
/// `round(2^12 / sqrt(2))`: the 2:1 rectangle input scale (§7.13.3).
const INV_SQRT2: i32 = 2896;
/// `round(2^12 * sqrt(2))`: the identity4/16 scale (§7.13.2.15).
const SQRT2: i32 = 5793;
/// Fractional bits of [`INV_SQRT2`] and [`SQRT2`].
const SQRT2_BITS: u32 = 12;

/// An AV1 transform size (AV1 §6.10.18 `TX_SIZES_ALL`): sides of 4 to 64
/// samples with an aspect ratio of at most 4:1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Av1TxSize {
    /// log2 of the width in samples (2..=6).
    pub width_log2: u8,
    /// log2 of the height in samples (2..=6).
    pub height_log2: u8,
}

impl Av1TxSize {
    /// Every size in the spec's `TX_SIZES_ALL` order (`TX_4X4` .. `TX_64X16`).
    pub const ALL: [Av1TxSize; 19] = [
        Self::of(2, 2),
        Self::of(3, 3),
        Self::of(4, 4),
        Self::of(5, 5),
        Self::of(6, 6),
        Self::of(2, 3),
        Self::of(3, 2),
        Self::of(3, 4),
        Self::of(4, 3),
        Self::of(4, 5),
        Self::of(5, 4),
        Self::of(5, 6),
        Self::of(6, 5),
        Self::of(2, 4),
        Self::of(4, 2),
        Self::of(3, 5),
        Self::of(5, 3),
        Self::of(4, 6),
        Self::of(6, 4),
    ];

    const fn of(width_log2: u8, height_log2: u8) -> Self {
        Self {
            width_log2,
            height_log2,
        }
    }

    /// Whether this is one of the 19 AV1 sizes.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        let (w, h) = (self.width_log2, self.height_log2);
        w >= 2 && w <= 6 && h >= 2 && h <= 6 && w.abs_diff(h) <= 2
    }

    /// Width in samples.
    #[must_use]
    pub const fn width(self) -> usize {
        1 << self.width_log2
    }

    /// Height in samples.
    #[must_use]
    pub const fn height(self) -> usize {
        1 << self.height_log2
    }

    /// Coded coefficient columns: the width, capped at 32.
    #[must_use]
    pub const fn coded_width(self) -> usize {
        min_side(self.width())
    }

    /// Coded coefficient rows: the height, capped at 32.
    #[must_use]
    pub const fn coded_height(self) -> usize {
        min_side(self.height())
    }

    /// Coded coefficients (the length `coeffs` must have at least).
    #[must_use]
    pub const fn coded_area(self) -> usize {
        self.coded_width() * self.coded_height()
    }

    /// Position in [`Self::ALL`]; `None` for an invalid size.
    #[must_use]
    pub const fn index(self) -> Option<usize> {
        let mut i = 0;
        while i < Self::ALL.len() {
            let size = Self::ALL[i];
            if size.width_log2 == self.width_log2 && size.height_log2 == self.height_log2 {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// Rounding shift after the row pass (§7.13.3 `Transform_Row_Shift`,
    /// libaom `inv_shift_*[0]`), which keeps the column input within 16
    /// bits.
    ///
    /// # Panics
    ///
    /// For an invalid size.
    pub(crate) const fn row_shift(self) -> u32 {
        match self.index() {
            Some(index) => ROW_SHIFT[index],
            None => panic!("not an AV1 transform size"),
        }
    }

    /// Whether the row pass prescales by `1/sqrt(2)` (2:1 rectangles).
    pub(crate) const fn is_two_to_one(self) -> bool {
        self.width_log2.abs_diff(self.height_log2) == 1
    }
}

/// §7.13.3 `Transform_Row_Shift` in [`Av1TxSize::ALL`] order.
const ROW_SHIFT: [u32; 19] = [0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2];

const fn min_side(side: usize) -> usize {
    if side < AV1_MAX_CODED_SIDE {
        side
    } else {
        AV1_MAX_CODED_SIDE
    }
}

/// A 1-D inverse transform kind (AV1 §7.13.2), per direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kernel1d {
    Dct,
    Adst,
    /// ADST with its outputs reversed.
    FlipAdst,
    Identity,
}

impl Kernel1d {
    /// Whether this kernel exists at `side` samples (libaom
    /// `av1_txfm_type_ls`): DCT 4–64, ADST 4–16, identity 4–32.
    const fn supports(self, side: usize) -> bool {
        match self {
            Self::Dct => side <= 64,
            Self::Adst | Self::FlipAdst => side <= 16,
            Self::Identity => side <= 32,
        }
    }
}

/// An AV1 transform type (AV1 §6.10.19 `TxType`), named vertical
/// (column) kernel first; the discriminant is the spec's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum Av1TxType {
    DctDct = 0,
    AdstDct = 1,
    DctAdst = 2,
    AdstAdst = 3,
    FlipadstDct = 4,
    DctFlipadst = 5,
    FlipadstFlipadst = 6,
    AdstFlipadst = 7,
    FlipadstAdst = 8,
    /// Identity in both directions.
    Idtx = 9,
    /// Vertical DCT, horizontal identity.
    VDct = 10,
    /// Horizontal DCT, vertical identity.
    HDct = 11,
    VAdst = 12,
    HAdst = 13,
    VFlipadst = 14,
    HFlipadst = 15,
}

impl Av1TxType {
    /// Every type in spec order.
    pub const ALL: [Av1TxType; 16] = [
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

    /// The type with spec value `value`, if any.
    #[must_use]
    pub fn from_value(value: u8) -> Option<Self> {
        Self::ALL.get(usize::from(value)).copied()
    }

    /// The column (vertical) and row (horizontal) kernels (§7.13.3, libaom
    /// `vtx_tab` / `htx_tab` with the flips of `set_flip_cfg`).
    pub(crate) const fn kernels(self) -> (Kernel1d, Kernel1d) {
        use Kernel1d::{Adst, Dct, FlipAdst, Identity};
        match self {
            Self::DctDct => (Dct, Dct),
            Self::AdstDct => (Adst, Dct),
            Self::DctAdst => (Dct, Adst),
            Self::AdstAdst => (Adst, Adst),
            Self::FlipadstDct => (FlipAdst, Dct),
            Self::DctFlipadst => (Dct, FlipAdst),
            Self::FlipadstFlipadst => (FlipAdst, FlipAdst),
            Self::AdstFlipadst => (Adst, FlipAdst),
            Self::FlipadstAdst => (FlipAdst, Adst),
            Self::Idtx => (Identity, Identity),
            Self::VDct => (Dct, Identity),
            Self::HDct => (Identity, Dct),
            Self::VAdst => (Adst, Identity),
            Self::HAdst => (Identity, Adst),
            Self::VFlipadst => (FlipAdst, Identity),
            Self::HFlipadst => (Identity, FlipAdst),
        }
    }

    /// Whether both 1-D kernels of this type exist at `size`'s sides. The
    /// transform-set syntax (§5.11.47–48) only ever selects such types; a
    /// 64-sample side is always `DCT_DCT` there.
    #[must_use]
    pub const fn fits(self, size: Av1TxSize) -> bool {
        let (vertical, horizontal) = self.kernels();
        size.is_valid() && vertical.supports(size.height()) && horizontal.supports(size.width())
    }
}

/// The 2-D inverse of one size: column then row kernel, column-major
/// `coeffs` (module docs), added to `dst` at `dst_stride` with 8-bit
/// clipping.
pub(crate) type InverseAddFn = unsafe fn(
    vertical: Kernel1d,
    horizontal: Kernel1d,
    coeffs: &[i32],
    nonzero_rows: usize,
    dst: &mut [u8],
    dst_stride: usize,
);

/// One ISA's 2-D inverses, indexed like [`Av1TxSize::ALL`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Av1InverseFns {
    pub add: [InverseAddFn; 19],
    pub dc: unsafe fn(Av1TxSize, i32, &mut [u8], usize),
}

/// Conservative coded-row extent of the first `eob` positions of `kind`'s
/// scan (AV1 §5.11.41): every nonzero coefficient lies in a row below it.
/// Diagonal scans complete all positions with r+c < k before reaching row k;
/// count that triangle clipped to the coded width. Rounding to 4/8/16 rows
/// avoids a scan walk per block, while the driver rounds to its lane count.
/// One-dimensional transform classes use their raster scans: `H_*` column
/// by column, `V_*` row by row. `nc-av1` checks this bound against its scan
/// tables for every size, class and end of block.
#[must_use]
pub fn av1_inverse_nonzero_rows(tx: Av1TxSize, kind: Av1TxType, eob: usize) -> usize {
    use Av1TxType::{HAdst, HDct, HFlipadst, VAdst, VDct, VFlipadst};
    let (w, h) = (tx.coded_width(), tx.coded_height());
    if eob == 1 {
        return 1;
    }
    match kind {
        HDct | HAdst | HFlipadst => eob.min(h),
        VDct | VAdst | VFlipadst => eob.div_ceil(w).min(h),
        _ => {
            for rows in [4, 8, 16] {
                let cols = w.min(rows);
                let complete_diagonals = cols * rows - cols * (cols - 1) / 2;
                if rows < h && eob <= complete_diagonals {
                    return rows;
                }
            }
            h
        }
    }
}

/// Highest end-of-block of a `DCT_DCT` 32x32 or 64x64 whose nonzero
/// coefficients all lie in the low 8x8 (the first 36 positions of the
/// default zig-zag scan, §5.11.41, cover exactly the `r + c < 8` triangle).
pub const AV1_LOW8_MAX_EOB: usize = 36;
/// The same for the low 16x16 (`16 * 17 / 2` positions).
pub const AV1_LOW16_MAX_EOB: usize = 136;

/// The side of the low-frequency square that holds every nonzero
/// coefficient of a 32x32 or 64x64 `DCT_DCT` with end-of-block `eob`
/// (8, 16 or 32), for the EOB-sparse inverse kernels.
#[must_use]
pub const fn av1_dct_nonzero_side(eob: usize) -> usize {
    if eob <= AV1_LOW8_MAX_EOB {
        8
    } else if eob <= AV1_LOW16_MAX_EOB {
        16
    } else {
        AV1_MAX_CODED_SIDE
    }
}

#[cfg(test)]
mod tests {
    use super::Av1TxSize;

    #[test]
    fn sizes_are_the_nineteen_valid_ones() {
        let mut valid = 0;
        for w in 0..8 {
            for h in 0..8 {
                let size = Av1TxSize {
                    width_log2: w,
                    height_log2: h,
                };
                valid += usize::from(size.is_valid());
                assert_eq!(size.is_valid(), size.index().is_some(), "{w} {h}");
            }
        }
        assert_eq!(valid, Av1TxSize::ALL.len());
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx512;
