//! Motion vector scaling (AV1 §7.11.3.3): where a block's prediction starts
//! in a reference frame of a different size, and how far it steps per
//! output sample. For a reference of the frame's own size this gives the
//! plain unscaled position (`step` 1024).

use super::convolve::{AV1_SCALE_SUBPEL_BITS, Av1McAxis};
use super::round2_signed;

/// `REF_SCALE_SHIFT`: precision of the reference-to-frame size ratio.
const REF_SCALE_SHIFT: u32 = 14;
/// `SUBPEL_BITS`: motion vectors address 1/16 plane samples once doubled.
const SUBPEL_BITS: u32 = 4;
/// Mask of the 1/1024 fraction of a scaled position.
const SCALE_FRACTION_MASK: i64 = (1 << AV1_SCALE_SUBPEL_BITS) - 1;

/// The sizes a reference is scaled between, in luma samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1ScaleSizes {
    /// `RefUpscaledWidth[refIdx]` and `RefFrameHeight[refIdx]`.
    pub reference: (u32, u32),
    /// `FrameWidth` and `FrameHeight` of the frame being predicted.
    pub frame: (u32, u32),
}

/// A scaled block position: the integer reference sample of the first
/// output sample (`startX >> 10`, `startY >> 10`) and each direction's
/// fraction and step. The prediction window starts 3 samples up and left
/// of `sample`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1ScaledPosition {
    /// Integer reference sample (x, y) of the first output sample; may lie
    /// outside the reference (its samples are then clamped).
    pub sample: (i64, i64),
    /// Horizontal fraction and `xStep`.
    pub x: Av1McAxis,
    /// Vertical fraction and `yStep`.
    pub y: Av1McAxis,
}

/// One direction of §7.11.3.3: the start position in 1/1024 reference
/// samples and the step, for plane position `position`, motion vector
/// component `mv` (1/8 luma samples) and subsampling `sub`.
fn scale_axis(position: u32, mv: i32, sub: bool, reference: u32, frame: u32) -> (i64, u32) {
    let scale =
        ((i64::from(reference) << REF_SCALE_SHIFT) + i64::from(frame / 2)) / i64::from(frame);
    let half_sample = 1i64 << (SUBPEL_BITS - 1);
    let orig =
        (i64::from(position) << SUBPEL_BITS) + i64::from((2 * mv) >> u32::from(sub)) + half_sample;
    let base = orig * scale - (half_sample << REF_SCALE_SHIFT);
    let off = (1i64 << (AV1_SCALE_SUBPEL_BITS - SUBPEL_BITS)) / 2;
    let start = round2_signed(base, REF_SCALE_SHIFT + SUBPEL_BITS - AV1_SCALE_SUBPEL_BITS) + off;
    let step = round2_signed(scale, REF_SCALE_SHIFT - AV1_SCALE_SUBPEL_BITS);
    (start, step as u32)
}

/// The scaled start and steps (AV1 §7.11.3.3) of a block at plane sample
/// `(x, y)` with motion vector `mv` = `(row, col)` in 1/8 luma samples, in a
/// plane subsampled by `(sub_x, sub_y)`.
#[must_use]
pub fn av1_scale_position(
    (x, y): (u32, u32),
    (mv_row, mv_col): (i32, i32),
    (sub_x, sub_y): (bool, bool),
    sizes: Av1ScaleSizes,
) -> Av1ScaledPosition {
    let (start_x, step_x) = scale_axis(x, mv_col, sub_x, sizes.reference.0, sizes.frame.0);
    let (start_y, step_y) = scale_axis(y, mv_row, sub_y, sizes.reference.1, sizes.frame.1);
    Av1ScaledPosition {
        sample: (
            start_x >> AV1_SCALE_SUBPEL_BITS,
            start_y >> AV1_SCALE_SUBPEL_BITS,
        ),
        x: Av1McAxis {
            position: (start_x & SCALE_FRACTION_MASK) as u32,
            step: step_x,
        },
        y: Av1McAxis {
            position: (start_y & SCALE_FRACTION_MASK) as u32,
            step: step_y,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1_mc::convolve::AV1_UNSCALED_STEP;

    /// An unscaled reference gives the plain 1/16 position: integer sample
    /// `pos >> 4`, phase `pos & 15`.
    #[test]
    fn unscaled_reference_gives_the_plain_position() {
        let sizes = Av1ScaleSizes {
            reference: (640, 360),
            frame: (640, 360),
        };
        for (mv_row, mv_col) in [(0, 0), (-13, 7), (100, -255)] {
            for sub in [false, true] {
                let p = av1_scale_position((24, 40), (mv_row, mv_col), (sub, sub), sizes);
                let pos_x = (24 << 4) + ((2 * mv_col) >> u32::from(sub));
                let pos_y = (40 << 4) + ((2 * mv_row) >> u32::from(sub));
                assert_eq!(p.sample, (i64::from(pos_x >> 4), i64::from(pos_y >> 4)));
                assert_eq!(p.x.step, AV1_UNSCALED_STEP);
                assert_eq!(
                    (p.x.position >> 6, p.y.position >> 6),
                    ((pos_x & 15) as u32, (pos_y & 15) as u32)
                );
            }
        }
    }
}
