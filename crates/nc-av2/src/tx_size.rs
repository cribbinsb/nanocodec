//! Square AV2 transform sizes and their coefficient-coding geometry.
//!
//! The subset codes square transforms only (AVM `TX_4X4` .. `TX_64X64`).
//! A 64-point transform keeps only its top-left 32×32 coefficients
//! (AVM `av2_get_adjusted_tx_size`), so its coded geometry is that of TX32
//! while its entropy-model identity stays its own.

/// Largest coded side: AVM zeroes the high-frequency half of 64-point transforms.
const MAX_CODED_LOG2: u32 = 5;
/// Side of the smallest transform and of one entropy-context unit, in pixels.
const UNIT_LOG2: u32 = 2;

/// A square transform size, ordered as AVM's `TX_SIZE` for square sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TxSize {
    /// 4×4 (AVM `TX_4X4`).
    Tx4,
    /// 8×8 (AVM `TX_8X8`).
    Tx8,
    /// 16×16 (AVM `TX_16X16`).
    Tx16,
    /// 32×32 (AVM `TX_32X32`).
    Tx32,
    /// 64×64 with the top-left 32×32 coded (AVM `TX_64X64`).
    Tx64,
}

impl TxSize {
    /// Every size, smallest first.
    pub const ALL: [Self; 5] = [Self::Tx4, Self::Tx8, Self::Tx16, Self::Tx32, Self::Tx64];

    /// The size whose side is `1 << log2` pixels, for `log2` in 2..=6.
    #[must_use]
    pub const fn from_log2(log2: u32) -> Option<Self> {
        match log2 {
            2 => Some(Self::Tx4),
            3 => Some(Self::Tx8),
            4 => Some(Self::Tx16),
            5 => Some(Self::Tx32),
            6 => Some(Self::Tx64),
            _ => None,
        }
    }

    /// Base-two logarithm of the side in pixels.
    #[must_use]
    pub const fn log2(self) -> u32 {
        self.index() as u32 + UNIT_LOG2
    }

    /// Side in pixels.
    #[must_use]
    pub const fn width(self) -> usize {
        1 << self.log2()
    }

    /// Side in 4×4 entropy-context units (AVM `tx_size_wide_unit`).
    #[must_use]
    pub const fn units(self) -> usize {
        1 << (self.log2() - UNIT_LOG2)
    }

    /// Base-two logarithm of the coded side (AVM `get_txb_bwl`).
    #[must_use]
    pub const fn coded_log2(self) -> u32 {
        if self.log2() < MAX_CODED_LOG2 {
            self.log2()
        } else {
            MAX_CODED_LOG2
        }
    }

    /// Side of the coded coefficient block (AVM `get_txb_wide`).
    #[must_use]
    pub const fn coded_width(self) -> usize {
        1 << self.coded_log2()
    }

    /// Number of coded coefficients, also the largest end of block (AVM `av2_get_max_eob`).
    #[must_use]
    pub const fn coded_len(self) -> usize {
        1 << (2 * self.coded_log2())
    }

    /// AVM `TX_SIZE` value, which is also the square entropy context
    /// (`get_txsize_entropy_ctx`) used to select coefficient models.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_avm_adjusted_sizes() {
        let expected = [
            (4, 1, 4, 16),
            (8, 2, 8, 64),
            (16, 4, 16, 256),
            (32, 8, 32, 1024),
        ];
        for (tx, (width, units, coded, len)) in TxSize::ALL.into_iter().zip(expected) {
            assert_eq!(
                (tx.width(), tx.units(), tx.coded_width(), tx.coded_len()),
                (width, units, coded, len)
            );
        }
        let tx64 = TxSize::Tx64;
        assert_eq!(
            (
                tx64.width(),
                tx64.units(),
                tx64.coded_width(),
                tx64.coded_len()
            ),
            (64, 16, 32, 1024)
        );
        for tx in TxSize::ALL {
            assert_eq!(TxSize::from_log2(tx.log2()), Some(tx));
        }
        assert_eq!(TxSize::from_log2(1), None);
        assert_eq!(TxSize::from_log2(7), None);
    }
}
