//! References of another size than the frame: whether one scales
//! (`is_scaled`), and where a block reads it (AV1 §7.11.3.3 motion vector
//! scaling; the arithmetic is nc-dsp's `av1_scale_position`).
use crate::modeinfo::Av1Mv;
use nc_common::decode::DecodeError;
use nc_dsp::av1_mc::convolve::Av1McAxis;
use nc_dsp::av1_mc::filter::AV1_FILTER_BEFORE;
use nc_dsp::av1_mc::scale::{Av1ScaleSizes, av1_scale_position};

/// `REF_SCALE_SHIFT`: scale factors are in 1/16384; one is no scaling.
const REF_SCALE_SHIFT: u32 = 14;
const NO_SCALE: usize = 1 << REF_SCALE_SHIFT;
/// A reference may be at most twice the frame's size and at least a
/// sixteenth of it (§7.11.3.3, §7.21).
const MAX_DOWNSCALE: usize = 2;
const MAX_UPSCALE: usize = 16;

/// The sizes a reference of `size` scales from a frame of `frame` size,
/// or `None` when `is_scaled` is false.
///
/// # Errors
/// A reference more than twice the frame's size or less than a sixteenth
/// of it.
pub(crate) fn scale_sizes(
    size: (usize, usize),
    frame: (usize, usize),
) -> Result<Option<Av1ScaleSizes>, DecodeError> {
    let fits = |reference: usize, frame: usize| {
        reference <= MAX_DOWNSCALE * frame && frame <= MAX_UPSCALE * reference
    };
    if !fits(size.0, frame.0) || !fits(size.1, frame.1) {
        return Err(DecodeError::Malformed("AV1 reference size"));
    }
    // `is_scaled`: either factor (`xScale`, `yScale`) differs from one;
    // widths past 32768 can differ by a sample and still not scale.
    let factor =
        |reference: usize, frame: usize| ((reference << REF_SCALE_SHIFT) + frame / 2) / frame;
    let scaled = factor(size.0, frame.0) != NO_SCALE || factor(size.1, frame.1) != NO_SCALE;
    Ok(scaled.then_some(Av1ScaleSizes {
        reference: (size.0 as u32, size.1 as u32),
        frame: (frame.0 as u32, frame.1 as u32),
    }))
}

/// Where the block at plane sample (`x`, `y`) of a plane subsampled by
/// `sub`, displaced by luma vector `mv`, reads a reference scaled by
/// `sizes`: the top-left of its filter window and each direction's start
/// fraction and step.
pub(crate) fn scaled_window(
    sizes: Av1ScaleSizes,
    (x, y): (usize, usize),
    mv: Av1Mv,
    sub: (bool, bool),
) -> ((isize, isize), (Av1McAxis, Av1McAxis)) {
    let position = av1_scale_position(
        (x as u32, y as u32),
        (i32::from(mv.row), i32::from(mv.col)),
        sub,
        sizes,
    );
    let before = AV1_FILTER_BEFORE as isize;
    (
        (
            position.sample.0 as isize - before,
            position.sample.1 as isize - before,
        ),
        (position.x, position.y),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaling_follows_the_scale_factors_within_their_limits() {
        assert_eq!(scale_sizes((640, 360), (640, 360)).unwrap(), None);
        let half = scale_sizes((640, 360), (1280, 720)).unwrap().unwrap();
        assert_eq!(half.reference, (640, 360));
        assert_eq!(half.frame, (1280, 720));
        // A sample apart at 40000 wide rounds to the same factor.
        assert_eq!(scale_sizes((40001, 64), (40000, 64)).unwrap(), None);
        assert!(scale_sizes((1281, 720), (640, 360)).is_err());
        assert!(scale_sizes((64, 64), (1040, 64)).is_err());
    }
}
