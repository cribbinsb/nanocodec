//! The border-and-rows H.264 `Intra_8x8` predictors compiled for NEON.

use super::H264Intra8x8Refs;

#[target_feature(enable = "neon")]
pub(super) unsafe fn predict(refs: &H264Intra8x8Refs, mode: u8, out: &mut [u8; 64]) {
    super::rows::predict(refs, mode, out);
}
