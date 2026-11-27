//! AV1 transform sizes (AV1 §6.10.19 `TX_SIZES_ALL`), shared by the
//! encoder and the decoder: the 19 sizes with the derived quantities the
//! coefficient syntax indexes by (`Tx_Size_Sqr`, `Tx_Size_Sqr_Up`,
//! `txSzCtx`, the coded 32x32-capped area, `dqDenom`).
//!
//! The encoder's square-block paths use the square constructors and
//! accessors ([`TxSize::from_log2`], [`TxSize::size`]); everything that
//! may meet a rectangle reads the width and height.

use nc_dsp::Av1TxSize;

/// log2 of the smallest transform side (`TX_4X4`).
const MIN_SIDE_LOG2: u8 = 2;
/// log2 of the largest coded side: 64-point transforms code 32 coefficients.
const MAX_CODED_SIDE_LOG2: u8 = 5;
/// Transform areas (samples) above which `dqDenom` doubles (AV1 §7.12.3).
const DEQUANT_AREA_STEPS: [usize; 2] = [256, 1024];
/// The transform-type CDF sizes (`EXT_TX_SIZES`: square 4x4 .. 32x32).
const EXT_TX_SIZES: u8 = 4;

/// A transform size (`TX_4X4` .. `TX_64X16`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TxSize(Av1TxSize);

impl TxSize {
    /// `TX_4X4`, the only transform of lossless blocks.
    pub const TX_4X4: Self = Self::square(MIN_SIDE_LOG2);
    /// The `width` x `height` transform (log2 sides), if AV1 has it.
    pub(crate) fn new(width_log2: u8, height_log2: u8) -> Option<Self> {
        let size = Av1TxSize {
            width_log2,
            height_log2,
        };
        size.is_valid().then_some(Self(size))
    }
    /// `Split_Tx_Size`: squares split into four, 2:1 rectangles into two
    /// squares, 4:1 rectangles into two 2:1 halves. `None` for 4x4.
    pub(crate) fn split(self) -> Option<Self> {
        let (w, h) = (self.width_log2(), self.height_log2());
        let (w, h) = match w.cmp(&h) {
            std::cmp::Ordering::Equal => (w.checked_sub(1)?, h.checked_sub(1)?),
            _ if w.abs_diff(h) == 1 => (w.min(h), w.min(h)),
            std::cmp::Ordering::Greater => (w - 1, h),
            std::cmp::Ordering::Less => (w, h - 1),
        };
        Self::new(w, h)
    }
    /// The size as the inverse-transform kernels take it.
    pub(crate) fn dsp(self) -> Av1TxSize {
        self.0
    }
    pub(crate) fn width_log2(self) -> u8 {
        self.0.width_log2
    }
    pub(crate) fn height_log2(self) -> u8 {
        self.0.height_log2
    }
    pub(crate) fn width(self) -> usize {
        self.0.width()
    }
    pub(crate) fn height(self) -> usize {
        self.0.height()
    }
    /// `Tx_Size_Sqr`: the smaller side, log2.
    pub(crate) fn square_log2(self) -> u8 {
        self.0.width_log2.min(self.0.height_log2)
    }
    /// `Tx_Size_Sqr_Up`: the larger side, log2.
    pub(crate) fn square_up_log2(self) -> u8 {
        self.0.width_log2.max(self.0.height_log2)
    }
    /// `txSzCtx` (§5.11.39): the coefficient CDF set, the mean of the
    /// square sizes below and above, rounded up.
    pub(crate) fn context(self) -> usize {
        let sqr = usize::from(self.square_log2() - MIN_SIDE_LOG2);
        let sqr_up = usize::from(self.square_up_log2() - MIN_SIDE_LOG2);
        (sqr + sqr_up + 1) >> 1
    }
    /// log2 of the coded coefficient columns (the width, capped at 32).
    pub(crate) fn coded_width_log2(self) -> u8 {
        self.0.width_log2.min(MAX_CODED_SIDE_LOG2)
    }
    /// log2 of the coded coefficient rows (the height, capped at 32).
    pub(crate) fn coded_height_log2(self) -> u8 {
        self.0.height_log2.min(MAX_CODED_SIDE_LOG2)
    }
    /// Coded coefficients.
    pub(crate) fn coded_area(self) -> usize {
        self.0.coded_area()
    }
    /// `eobMultisize`: which `eob_pt_*` alphabet codes the end of block
    /// (0 for 16 coefficients .. 6 for 1024).
    pub(crate) fn eob_alphabet(self) -> usize {
        usize::from(self.coded_width_log2() + self.coded_height_log2() - 2 * MIN_SIDE_LOG2)
    }
    /// `log2(dqDenom)` (§7.12.3): 1 above 256 samples, 2 above 1024.
    pub(crate) fn dequant_shift(self) -> u32 {
        let area = self.width() * self.height();
        DEQUANT_AREA_STEPS
            .iter()
            .map(|&step| u32::from(area > step))
            .sum()
    }
    /// The transform-type CDF size index (`txSzSqr`, 4x4 .. 32x32).
    pub(crate) fn type_cdf_size(self) -> usize {
        usize::from(self.square_log2().min(MIN_SIDE_LOG2 + EXT_TX_SIZES - 1) - MIN_SIDE_LOG2)
    }
}

/// The square transforms, by side.
impl TxSize {
    /// `TX_8X8`.
    #[cfg(test)]
    pub const TX_8X8: Self = Self::square(3);
    /// `TX_16X16`.
    pub const TX_16X16: Self = Self::square(4);
    /// `TX_32X32`.
    pub const TX_32X32: Self = Self::square(5);
    /// `TX_64X64` (only the top-left 32x32 coefficients are coded).
    pub const TX_64X64: Self = Self::square(6);

    const fn square(log2: u8) -> Self {
        Self(Av1TxSize {
            width_log2: log2,
            height_log2: log2,
        })
    }

    /// The transform for an `n`x`n` block, `n = 1 << log2`.
    #[must_use]
    pub fn from_log2(log2: u8) -> Option<Self> {
        Self::new(log2, log2)
    }

    /// Whether the transform is square.
    #[must_use]
    pub fn is_square(self) -> bool {
        self.0.width_log2 == self.0.height_log2
    }

    /// log2 of the side of a square transform.
    #[must_use]
    pub fn log2(self) -> u8 {
        debug_assert!(self.is_square(), "{self:?} has no single side");
        self.0.width_log2
    }

    /// Side length of a square transform in samples.
    #[must_use]
    pub fn size(self) -> usize {
        1 << self.log2()
    }

    /// Side length of a square transform's coded coefficient array: 64x64
    /// transforms code only their top-left 32x32 (libaom
    /// `av1_get_adjusted_tx_size`).
    #[must_use]
    pub fn coded_size(self) -> usize {
        1 << self.coded_log2()
    }

    /// log2 of [`TxSize::coded_size`].
    #[must_use]
    pub fn coded_log2(self) -> u8 {
        self.log2().min(MAX_CODED_SIDE_LOG2)
    }
}
