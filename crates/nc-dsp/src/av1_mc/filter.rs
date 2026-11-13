//! The AV1 interpolation filters (AV1 §7.11.3.4, `Subpel_Filters`): four
//! families of 8-tap filters at 1/16-sample positions, chosen per direction
//! (`dual_filter`), with 4-tap variants for block edges of at most 4
//! samples.
//!
//! The regular, smooth and sharp taps are shared with the encoder's
//! prediction kernels (`crate::mc`); bilinear is defined here. Taps: libaom
//! `av1/common/filter.h` (BSD-2-Clause, see `THIRD_PARTY.md`).

use crate::mc::{REGULAR_4, REGULAR_8, SHARP_8, SMOOTH_4, SMOOTH_8, TAPS};

/// Sub-sample positions of the interpolation filters (`1 << SUBPEL_BITS`).
pub const AV1_SUBPEL_POSITIONS: usize = 16;
/// Filter taps of every family (the 4-tap variants pad with zeros).
pub const AV1_FILTER_TAPS: usize = TAPS;
/// Taps before the centre sample: a prediction window starts this many
/// samples up and left of the block's integer reference position.
pub const AV1_FILTER_BEFORE: usize = 3;

/// One filter family's taps by 1/16-sample position.
pub(crate) type Kernel = [[i16; TAPS]; AV1_SUBPEL_POSITIONS];

/// Edges of at most this many samples take a family's 4-tap variant
/// (`filterIdx` 4 and 5).
const SHORT_EDGE: usize = 4;

/// `BILINEAR` (`filterIdx` 3; libaom `av1_bilinear_filters`): two taps,
/// `128 - 8p` and `8p`, on the centre sample and the next.
const BILINEAR: Kernel = {
    let mut kernel = [[0; TAPS]; AV1_SUBPEL_POSITIONS];
    let mut position = 0;
    while position < AV1_SUBPEL_POSITIONS {
        let weight = (position * 8) as i16;
        kernel[position][AV1_FILTER_BEFORE] = 128 - weight;
        kernel[position][AV1_FILTER_BEFORE + 1] = weight;
        position += 1;
    }
    kernel
};

/// An interpolation filter family; the discriminants are the AV1
/// `interp_filter` / `interpolation_filter` syntax values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Av1Filter {
    /// `EIGHTTAP` (regular).
    Regular = 0,
    /// `EIGHTTAP_SMOOTH`.
    Smooth = 1,
    /// `EIGHTTAP_SHARP`.
    Sharp = 2,
    /// `BILINEAR`.
    Bilinear = 3,
}

impl Av1Filter {
    /// Every family, in syntax order.
    pub const ALL: [Av1Filter; 4] = [Self::Regular, Self::Smooth, Self::Sharp, Self::Bilinear];

    /// The taps by position for a block edge of `length` samples: the
    /// family's own filter, or for edges of at most 4 samples the 4-tap
    /// regular filter (regular, sharp: `filterIdx` 4) or the 4-tap smooth
    /// filter (smooth: `filterIdx` 5). Bilinear has no 4-tap variant.
    #[must_use]
    pub(crate) fn kernel(self, length: usize) -> &'static Kernel {
        let short = length <= SHORT_EDGE;
        match self {
            Self::Regular | Self::Sharp if short => &REGULAR_4,
            Self::Smooth if short => &SMOOTH_4,
            Self::Regular => &REGULAR_8,
            Self::Smooth => &SMOOTH_8,
            Self::Sharp => &SHARP_8,
            Self::Bilinear => &BILINEAR,
        }
    }
}

/// The filters of one block in each direction (AV1 §7.11.3.4: the
/// horizontal pass uses `interp_filter[1]`, the vertical `interp_filter[0]`;
/// without `enable_dual_filter` both are the same).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Av1Filters {
    /// Horizontal filter (`interp_filter[1]`).
    pub x: Av1Filter,
    /// Vertical filter (`interp_filter[0]`).
    pub y: Av1Filter,
}

impl Av1Filters {
    /// The same family in both directions.
    #[must_use]
    pub const fn both(filter: Av1Filter) -> Self {
        Self {
            x: filter,
            y: filter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every filter is unity-gain, and position 0 is the identity.
    #[test]
    fn every_kernel_sums_to_128_with_identity_at_zero() {
        for filter in Av1Filter::ALL {
            for length in [2, 4, 8, 128] {
                let kernel = filter.kernel(length);
                assert_eq!(kernel[0][AV1_FILTER_BEFORE], 128);
                for taps in kernel {
                    assert_eq!(taps.iter().map(|&t| i32::from(t)).sum::<i32>(), 128);
                }
            }
        }
    }
}
