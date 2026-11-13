//! Integer-pel motion vectors, as produced by source-frame analysis.
//!
//! Analysis works in whole samples; codecs refine to sub-pel precision with
//! their own vector types later.

/// A motion vector in whole samples of the plane it refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Mv {
    /// Horizontal displacement (positive: reference block is to the right).
    pub x: i16,
    /// Vertical displacement (positive: reference block is below).
    pub y: i16,
}

impl Mv {
    /// The zero vector.
    pub const ZERO: Mv = Mv { x: 0, y: 0 };

    /// A vector from its components.
    #[must_use]
    pub const fn new(x: i16, y: i16) -> Self {
        Self { x, y }
    }

    /// City-block distance to `other`, the usual proxy for MV coding cost.
    #[must_use]
    pub fn l1_distance(self, other: Mv) -> u32 {
        u32::from(self.x.abs_diff(other.x)) + u32::from(self.y.abs_diff(other.y))
    }

    /// Component-wise median of three vectors (the H.264-style predictor).
    #[must_use]
    pub fn median3(a: Mv, b: Mv, c: Mv) -> Mv {
        Mv::new(median3(a.x, b.x, c.x), median3(a.y, b.y, c.y))
    }

    /// Each component multiplied by `factor` (scaling between pyramid levels),
    /// saturating at the `i16` range.
    #[must_use]
    pub fn scaled(self, factor: i16) -> Mv {
        Mv::new(self.x.saturating_mul(factor), self.y.saturating_mul(factor))
    }

    /// Each component clamped to `[-limit, limit]`.
    #[must_use]
    pub fn clamped(self, limit: i16) -> Mv {
        Mv::new(self.x.clamp(-limit, limit), self.y.clamp(-limit, limit))
    }
}

fn median3(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.min(b).max(c))
}

#[cfg(test)]
mod tests {
    use super::Mv;

    #[test]
    fn median_and_distance() {
        let m = Mv::median3(Mv::new(1, 9), Mv::new(5, -3), Mv::new(3, 4));
        assert_eq!(m, Mv::new(3, 4));
        assert_eq!(Mv::new(-2, 3).l1_distance(Mv::new(1, 1)), 5);
        assert_eq!(Mv::new(40, -40).clamped(32), Mv::new(32, -32));
        assert_eq!(Mv::new(3, -2).scaled(4), Mv::new(12, -8));
    }
}
