//! The luma intra prediction of one block as AV1 signals it (a `YMode`, its
//! angle delta, and filter intra), and which edge samples beyond the block
//! a directional prediction may read. Shared by the encoder and decoder.

use nc_common::{Plane, PlaneViewMut};
use nc_dsp::Dsp;

use crate::predict::{EdgeExtension, FilterIntraMode, IntraEdges, IntraMode};

/// Largest `|angle_delta_y|` (AV1 `MAX_ANGLE_DELTA`).
pub const MAX_ANGLE_DELTA: i8 = 3;
/// Degrees per angle-delta step (AV1 `ANGLE_STEP`).
const ANGLE_STEP: i16 = 3;
/// `filter_intra_mode` symbols.
pub const FILTER_INTRA_MODES: usize = FilterIntraMode::ALL.len();
/// Largest block side (log2) that may use filter intra (32x32).
const FILTER_INTRA_MAX_LOG2: u8 = 5;
/// Superblock side in 4x4 units: availability is decided within one.
const SB_UNITS_LOG2: u32 = 4;
const SB_UNITS: usize = 1 << SB_UNITS_LOG2;
/// 4x4 units per sample, log2.
const UNIT_LOG2: u32 = 2;

/// One block's luma intra prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LumaIntra {
    /// The `YMode`.
    pub mode: IntraMode,
    /// `angle_delta_y` in −3..=3 (directional modes only, else 0).
    pub angle_delta: i8,
    /// Filter intra (DC blocks of at most 32x32 only).
    pub filter: Option<FilterIntraMode>,
}

impl LumaIntra {
    /// DC prediction without filter intra.
    pub const DC: Self = Self::plain(IntraMode::Dc);

    /// `mode` at its nominal angle, without filter intra.
    #[must_use]
    pub const fn plain(mode: IntraMode) -> Self {
        Self {
            mode,
            angle_delta: 0,
            filter: None,
        }
    }

    /// The prediction angle in degrees of a directional mode.
    #[must_use]
    pub fn angle(self) -> Option<u16> {
        self.mode
            .nominal_angle()
            .map(|nominal| (nominal as i16 + ANGLE_STEP * i16::from(self.angle_delta)) as u16)
    }

    /// The `YMode` symbol whose transform-type CDFs the block's luma
    /// transform uses (AV1 §5.11.47: filter intra maps to a direction).
    #[must_use]
    pub fn intra_dir(self) -> usize {
        self.filter
            .map_or(self.mode.symbol(), FilterIntraMode::intra_dir)
    }

    /// The `angle_delta_y` symbol (0..=6).
    #[must_use]
    pub fn angle_delta_symbol(self) -> usize {
        (self.angle_delta + MAX_ANGLE_DELTA) as usize
    }

    /// Whether this prediction reads samples beyond the block's own edges.
    #[must_use]
    pub fn needs_extension(self) -> bool {
        self.angle()
            .is_some_and(|angle| angle != 90 && angle != 180)
    }

    /// The edges of the `size` square at (`x`, `y`) of the luma `plane`
    /// this prediction reads.
    #[must_use]
    pub fn edges(self, plane: &Plane, x: usize, y: usize, size: usize) -> IntraEdges {
        let extension = if self.needs_extension() {
            edge_extension(x, y, size)
        } else {
            EdgeExtension::default()
        };
        IntraEdges::extended(plane, x, y, size, extension)
    }

    /// Predict the block from its [`Self::edges`], with `dsp`'s kernels.
    pub fn predict(self, dsp: &Dsp, edges: &IntraEdges, dst: &mut PlaneViewMut<'_>) {
        match (self.filter, self.angle()) {
            (Some(filter), _) => dsp.av1_intra_predict_filter(edges, filter, dst),
            (None, Some(angle)) if self.needs_extension() => {
                dsp.av1_intra_predict_angle(edges, angle, dst);
            }
            _ => dsp.av1_intra_predict(edges, self.mode, dst),
        }
    }
}

/// Whether filter intra may be signalled on a block of side `1 << log2`.
#[must_use]
pub fn filter_intra_allowed(log2_size: u8) -> bool {
    log2_size <= FILTER_INTRA_MAX_LOG2
}

/// The `filter_intra_cdfs` index of a square block (AV1 `MiSize`:
/// `BLOCK_8X8` 3, `BLOCK_16X16` 6, `BLOCK_32X32` 9).
#[must_use]
pub fn filter_intra_size_context(log2_size: u8) -> usize {
    const SQUARE_SIZES_PER_STEP: usize = 3;
    usize::from(log2_size - UNIT_LOG2 as u8) * SQUARE_SIZES_PER_STEP
}

/// Which luma samples beyond the square block at (`x`, `y`) are decoded
/// when it is predicted (AV1 §5.11.35, `BlockDecoded` as set by
/// `clear_block_decoded_flags` and each transform block).
///
/// Our blocks are squares of a quadtree with one transform per intra block,
/// so blocks complete in Z order within each 64x64 superblock: a 4x4 unit
/// inside the superblock is decoded exactly when its Z-order index is below
/// the block's. Above the superblock everything is decoded; to its left
/// everything down to its bottom edge; to its right and below nothing.
/// Units beyond the picture edge count as decoded here (the spec leaves
/// them unset), which reads the same samples: a block never crosses the
/// picture edge, so such a unit is only asked about by a block that ends at
/// that edge, where the edge reads clamp to the last sample either way.
#[must_use]
pub fn edge_extension(x: usize, y: usize, size: usize) -> EdgeExtension {
    let unit = |v: usize| (v >> UNIT_LOG2) % SB_UNITS;
    let (col, row, side) = (unit(x), unit(y), size >> UNIT_LOG2);
    let own = z_order(row, col);
    let above_right = y > 0
        && match (row, col + side) {
            (0, _) => true,
            (_, c) if c >= SB_UNITS => false,
            (r, c) => z_order(r - 1, c) < own,
        };
    let below_left = x > 0
        && match (row + side, col) {
            (r, _) if r >= SB_UNITS => false,
            (_, 0) => true,
            (r, c) => z_order(r, c - 1) < own,
        };
    EdgeExtension {
        above_right,
        below_left,
    }
}

/// Z-order (Morton) index of a 4x4 unit in its superblock: row bits above
/// column bits at each level, so quadrants run top-left, top-right,
/// bottom-left, bottom-right.
fn z_order(row: usize, col: usize) -> usize {
    (0..SB_UNITS_LOG2).fold(0, |z, bit| {
        z | (((col >> bit) & 1) << (2 * bit)) | (((row >> bit) & 1) << (2 * bit + 1))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ext(above_right: bool, below_left: bool) -> EdgeExtension {
        EdgeExtension {
            above_right,
            below_left,
        }
    }

    #[test]
    fn quadtree_neighbours_follow_decode_order() {
        // Top-left 32x32 of a superblock below the first row: the row above
        // is decoded, the left superblock too.
        assert_eq!(edge_extension(64, 64, 32), ext(true, true));
        // Top-right 32x32: its above-right is the next superblock's part of
        // the row above (decoded), its below-left the bottom-left quadrant
        // (not yet).
        assert_eq!(edge_extension(96, 64, 32), ext(true, false));
        assert_eq!(edge_extension(96, 0, 32), ext(false, false));
        // Bottom-left 32x32: above-right is the top-right quadrant (done),
        // below-left is the next superblock row (not).
        assert_eq!(edge_extension(64, 96, 32), ext(true, false));
        // Bottom-right 32x32: neither.
        assert_eq!(edge_extension(96, 96, 32), ext(false, false));
        // The first 8x8 of the bottom-left 16x16 of the top-left 32x32.
        assert_eq!(edge_extension(64, 80, 8), ext(true, true));
        // A 64x64 block reads on along the row above; below it is not
        // decoded.
        assert_eq!(edge_extension(64, 64, 64), ext(true, false));
        // Picture edges: nothing above or to the left.
        assert_eq!(edge_extension(0, 0, 8), ext(false, false));
    }

    #[test]
    fn filter_intra_mapping_and_sizes() {
        assert_eq!(filter_intra_size_context(3), 3);
        assert_eq!(filter_intra_size_context(5), 9);
        assert!(filter_intra_allowed(5) && !filter_intra_allowed(6));
        let filtered = LumaIntra {
            filter: Some(FilterIntraMode::D157),
            ..LumaIntra::DC
        };
        assert_eq!(filtered.intra_dir(), IntraMode::D157.symbol());
        let delta = LumaIntra {
            angle_delta: -2,
            ..LumaIntra::plain(IntraMode::D203)
        };
        assert_eq!(delta.angle(), Some(197));
        assert_eq!(delta.angle_delta_symbol(), 1);
    }
}
