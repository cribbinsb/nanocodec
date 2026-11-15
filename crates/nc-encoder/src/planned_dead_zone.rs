//! The dead zone of a frame planned past the codec's coarsest quantizer,
//! shared by the backends that take λ headroom (`FramePlan::lambda_scale`;
//! H.264, HEVC, AV1, VP9).
//!
//! Rate control may plan a frame coarser than the syntax can code
//! (`CodecBackend::SUPPORTS_LAMBDA_HEADROOM`): the frame is coded at the
//! coarsest quantizer, its decisions priced at the planned λ, and its levels
//! thinned to those a quantizer at the planned step would keep. The planned
//! step is √`lambda_scale` times the coded one, so a level survives only if
//! its coefficient reaches the quantizer's keep fraction (one minus its
//! rounding offset) of that many coded steps.

/// The smallest level magnitude kept at a planned step `step_ratio` times
/// the coded one, for a quantizer whose rounding offset leaves the `keep`
/// fraction of a step below each level (`1 − rounding`). At least 1: at or
/// below the coded step every level is kept.
#[must_use]
pub fn planned_min_level(keep: f32, step_ratio: f32) -> i32 {
    ((keep * step_ratio).ceil() as i32).max(1)
}

/// Zero the `levels` below `min_level` in magnitude ([`planned_min_level`]);
/// returns how many remain non-zero.
pub fn apply_planned_dead_zone<L>(levels: &mut [L], min_level: i32) -> usize
where
    L: Copy + Default + PartialEq + Into<i32>,
{
    let mut nonzero = 0;
    for level in levels.iter_mut() {
        if (*level).into().abs() < min_level {
            *level = L::default();
        }
        nonzero += usize::from(*level != L::default());
    }
    nonzero
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_levels_reaching_the_planned_step() {
        // HEVC inter rounding 128/512: a level survives from 0.75 planned steps.
        assert_eq!(planned_min_level(0.75, 2.0), 2);
        assert_eq!(planned_min_level(0.75, 4.0), 3);
        assert_eq!(planned_min_level(0.75, 1.0), 1);
        assert_eq!(planned_min_level(0.0, 1.0), 1);
        let mut levels = [3i16, -2, 1, -1, 0, 5];
        assert_eq!(apply_planned_dead_zone(&mut levels, 2), 3);
        assert_eq!(levels, [3, -2, 0, 0, 0, 5]);
        let mut wide = [1i32, -1, 2];
        assert_eq!(apply_planned_dead_zone(&mut wide, 1), 3);
    }
}
