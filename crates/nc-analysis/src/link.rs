//! How an analysed frame relates to the source frames kept for reference.
//!
//! A flat IPP chain always searches against the previous source frame. A
//! hierarchical-P structure (`docs/ARCHITECTURE.md` §6.1) searches against
//! whichever earlier frame the codec will predict from, possibly several
//! frames back. The [`Analyzer`](crate::Analyzer) therefore keeps the source
//! of each reference *slot* the encoder keeps a reconstruction for, and each
//! frame names the slot it is searched against ([`ReferenceLink`]) and the
//! slots it replaces.
//!
//! Temporal seeds (last frame's vectors and camera motion) were measured over
//! the last frame's own reference distance; they are rescaled linearly to
//! this frame's distance, as SVT-AV1 scales its temporal candidates by
//! picture-distance ratio.

use crate::global::GlobalMotion;
use crate::grid::Grid;
use crate::motion::AggregatedField;
use crate::mv::Mv;

/// Most reference slots the analyzer keeps a source frame for (AV1's
/// `NUM_REF_FRAMES = 8`, AV1 §7.20).
pub const MAX_REFERENCE_SLOTS: usize = 8;

/// A set of reference slots, one bit per slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SlotMask(pub u8);

impl SlotMask {
    /// No slot.
    pub const NONE: SlotMask = SlotMask(0);
    /// Every slot.
    pub const ALL: SlotMask = SlotMask(((1u16 << MAX_REFERENCE_SLOTS) - 1) as u8);

    /// The mask holding only `slot`.
    #[must_use]
    pub const fn only(slot: usize) -> Self {
        SlotMask(1 << slot)
    }

    /// Whether `slot` is in the mask.
    #[must_use]
    pub const fn contains(self, slot: usize) -> bool {
        self.0 & (1 << slot) != 0
    }

    /// Whether the mask is empty (a non-reference frame).
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// The reference a frame is searched against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceLink {
    /// Slot holding the reference's source frame.
    pub slot: usize,
    /// Frames between the reference and this frame (1 = previous frame).
    pub distance: u32,
}

/// One frame's place in the reference structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameLink {
    /// What this frame is searched against.
    pub reference: ReferenceLink,
    /// Slots whose source this frame replaces once analysed.
    pub refresh: SlotMask,
}

impl FrameLink {
    /// The flat chain: every frame searched against, and replacing, the
    /// previous one in slot 0.
    pub const PREVIOUS: FrameLink = FrameLink {
        reference: ReferenceLink {
            slot: 0,
            distance: 1,
        },
        refresh: SlotMask(1),
    };
}

/// A ratio of reference distances, `to / from`, for rescaling vectors
/// measured over distance `from` to distance `to`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistanceRatio {
    to: u32,
    from: u32,
}

impl DistanceRatio {
    /// The ratio `to / from` (both at least 1).
    #[must_use]
    pub fn new(to: u32, from: u32) -> Self {
        Self {
            to: to.max(1),
            from: from.max(1),
        }
    }

    /// Whether vectors keep their values.
    #[must_use]
    pub fn is_identity(self) -> bool {
        self.to == self.from
    }

    /// `mv` rescaled, rounding half away from zero and saturating.
    #[must_use]
    pub fn scale_mv(self, mv: Mv) -> Mv {
        Mv::new(self.scale(mv.x), self.scale(mv.y))
    }

    fn scale(self, v: i16) -> i16 {
        let (to, from) = (i32::try_from(self.to), i32::try_from(self.from));
        let (to, from) = (to.unwrap_or(i32::MAX), from.unwrap_or(i32::MAX));
        let product = i32::from(v).saturating_mul(to);
        let rounded = (product.abs() + from / 2) / from * product.signum();
        rounded.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
    }

    /// The ratio as a float, for continuous models.
    #[must_use]
    pub fn factor(self) -> f32 {
        self.to as f32 / self.from as f32
    }
}

/// `grid` with every vector rescaled.
pub(crate) fn scale_mv_grid(grid: &Grid<Mv>, ratio: DistanceRatio) -> Grid<Mv> {
    let cells = grid.cells().iter().map(|&mv| ratio.scale_mv(mv)).collect();
    Grid::from_cells(grid.cols(), grid.rows(), cells)
}

/// `field` with every block's vector rescaled (its SADs are kept: they only
/// rank, and are never compared across frames).
pub(crate) fn scale_field(field: &AggregatedField, ratio: DistanceRatio) -> AggregatedField {
    let blocks = &field.blocks;
    let cells = blocks
        .cells()
        .iter()
        .map(|b| {
            let mut b = *b;
            b.mv = ratio.scale_mv(b.mv);
            b
        })
        .collect();
    AggregatedField {
        size: field.size,
        blocks: Grid::from_cells(blocks.cols(), blocks.rows(), cells),
    }
}

/// `global` with its displacement rescaled.
pub(crate) fn scale_global(global: GlobalMotion, ratio: DistanceRatio) -> GlobalMotion {
    global.scaled(ratio.factor())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_rounds_half_away_from_zero() {
        let half = DistanceRatio::new(1, 2);
        assert_eq!(half.scale_mv(Mv::new(3, -3)), Mv::new(2, -2));
        assert_eq!(half.scale_mv(Mv::new(4, -1)), Mv::new(2, -1));
        let quad = DistanceRatio::new(4, 1);
        assert_eq!(quad.scale_mv(Mv::new(-5, 7)), Mv::new(-20, 28));
        assert!(DistanceRatio::new(2, 2).is_identity());
    }

    #[test]
    fn ratio_saturates() {
        let big = DistanceRatio::new(4, 1);
        assert_eq!(
            big.scale_mv(Mv::new(i16::MAX, i16::MIN)),
            Mv::new(i16::MAX, i16::MIN)
        );
    }

    #[test]
    fn slot_masks() {
        assert!(SlotMask::ALL.contains(0) && SlotMask::ALL.contains(1));
        assert!(SlotMask::only(1).contains(1) && !SlotMask::only(1).contains(0));
        assert!(SlotMask::NONE.is_empty());
    }
}
