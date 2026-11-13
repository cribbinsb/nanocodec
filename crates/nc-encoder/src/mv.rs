//! Motion vectors as the engine sees them.
//!
//! The engine works in quarter luma samples: the precision H.264 and HEVC
//! signal. AV1 signals eighths and its backend scales by two. During inter
//! bring-up every vector the engine chooses is whole-sample; sub-sample
//! refinement comes later, in the same type.

/// Quarter-sample units per luma sample.
pub const MV_UNITS_PER_SAMPLE: i32 = 1 << MV_FRACTION_BITS;

/// Bits of a vector component below the whole sample.
const MV_FRACTION_BITS: u32 = 2;

/// A motion vector in quarter luma samples.
///
/// Positive `x` means the reference block lies to the right of the current
/// block, positive `y` below it (the convention of all three codecs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Mv {
    /// Horizontal displacement, quarter luma samples.
    pub x: i16,
    /// Vertical displacement, quarter luma samples.
    pub y: i16,
}

impl Mv {
    /// The zero vector.
    pub const ZERO: Mv = Mv { x: 0, y: 0 };

    /// A vector from whole-sample components.
    #[must_use]
    pub fn from_full_pel(x: i32, y: i32) -> Self {
        let unit = |v: i32| (v * MV_UNITS_PER_SAMPLE) as i16;
        Self {
            x: unit(x),
            y: unit(y),
        }
    }

    /// True if both components are whole samples.
    #[must_use]
    pub fn is_full_pel(self) -> bool {
        self.fraction() == (0, 0)
    }

    /// Both components in one word: equal vectors have equal bits, and one
    /// integer comparison needs no branch between the components.
    #[must_use]
    pub fn bits(self) -> u32 {
        u32::from(self.x as u16) | (u32::from(self.y as u16) << 16)
    }

    /// The whole-sample displacement `(x, y)`, rounding towards minus
    /// infinity for sub-sample vectors.
    #[must_use]
    pub fn full_pel(self) -> (i32, i32) {
        // An arithmetic shift rounds towards minus infinity: `div_euclid` by
        // the power-of-two unit, without its sign fix-ups (mode decision
        // takes this per measured vector).
        (
            i32::from(self.x) >> MV_FRACTION_BITS,
            i32::from(self.y) >> MV_FRACTION_BITS,
        )
    }

    /// The sub-sample phase `(x, y)` in `0..MV_UNITS_PER_SAMPLE`: the
    /// remainder of [`Mv::full_pel`] (`rem_euclid` by the unit).
    #[must_use]
    pub fn fraction(self) -> (i32, i32) {
        let mask = MV_UNITS_PER_SAMPLE - 1;
        (i32::from(self.x) & mask, i32::from(self.y) & mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_pel_round_trip() {
        let mv = Mv::from_full_pel(-3, 7);
        assert_eq!(mv, Mv { x: -12, y: 28 });
        assert!(mv.is_full_pel());
        assert_eq!(mv.full_pel(), (-3, 7));
        assert!(!Mv { x: 1, y: 0 }.is_full_pel());
    }

    #[test]
    fn full_pel_and_fraction_are_euclidean() {
        for x in i16::MIN..=i16::MAX {
            let mv = Mv {
                x,
                y: x.wrapping_neg(),
            };
            let units = MV_UNITS_PER_SAMPLE;
            let (vx, vy) = (i32::from(mv.x), i32::from(mv.y));
            assert_eq!(mv.full_pel(), (vx.div_euclid(units), vy.div_euclid(units)));
            assert_eq!(mv.fraction(), (vx.rem_euclid(units), vy.rem_euclid(units)));
            assert_eq!(mv.is_full_pel(), vx % units == 0 && vy % units == 0);
        }
    }
}
