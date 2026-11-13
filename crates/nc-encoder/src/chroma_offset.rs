//! Frame-level chroma quantizer offset adjustments, shared by the codecs that
//! code a frame chroma delta (AV1 `DeltaQUAc`, AV2 `u_ac_delta_q`, VP9
//! `delta_q_uv`). Each moves the codec's own chroma offset (codec qindex
//! units); callers round and clamp to their syntax range.
//!
//! - [`noise_attenuated_chroma_offset`]: a finer chroma quantizer pays on
//!   clean content but would spend bits coding sensor noise, so the offset
//!   fades out as the analyzer's noise estimate rises.
//! - [`coarse_chroma_extra`]: a further offset ramped in by the frame
//!   quantizer. At coarse quantizers chroma is cheap to keep, while clips
//!   coding at fine quantizers (bodycam) lose MS-SSIM to finer chroma, so
//!   one offset for all rates is a compromise (docs/handoff/av2.md).
//! - [`error_gap_offset`] with [`plane_error_ratio`]: the offset follows how
//!   much cleaner chroma was than luma in the previous frame. Under PSNR-YUV
//!   (6:1:1) the marginal value of chroma SSE is about (4/6)·`MSE_Y/MSE_C`
//!   of luma's, so chroma far cleaner than luma (crowd, pan: 12-18 dB) is
//!   cheap to improve and chroma near luma's error (bodycam) is not.

use nc_common::{Picture, PlaneIndex};
use nc_dsp::{BlockSize, Dsp};

/// Side of the blocks [`plane_error_ratio`] sums plane errors over.
const ERROR_BLOCK: usize = 16;
/// Every n-th row of error blocks is measured: the ratio needs no more.
const ERROR_ROW_PERIOD: usize = 4;
/// Largest change [`error_gap_offset`] makes, either way: the ratio of a
/// near-flat chroma plane must not drive it to the ends of a delta range.
const GAP_OFFSET_LIMIT: f32 = 16.0;

/// `base` (a chroma qindex offset, codec units) attenuated linearly to zero
/// at noise sigma `scale`; a non-positive `scale` disables the adaptation.
/// Callers round and clamp to their syntax range.
///
/// AV1 evidence (scale 2): five TRAIN noise families sit above the cutoff,
/// natural lowlight is 0.22–0.49; with base −24 the 15-natural-clip TRAIN
/// mean PSNR/VMAF-v1 BD-rate is −7.77%/−6.28% versus offset 0 (see
/// docs/handoff/av1.md).
#[must_use]
pub fn noise_attenuated_chroma_offset(base: f32, sigma: f32, scale: f32) -> f32 {
    let attenuation = if scale > 0.0 {
        (1.0 - sigma.max(0.0) / scale).clamp(0.0, 1.0)
    } else {
        1.0
    };
    base * attenuation
}

/// `extra` (a chroma qindex offset) ramped in linearly by the frame
/// quantizer `qindex`: none at or below `start`, all of it from `end` on
/// (a step at `start` when `end` is not above it).
#[must_use]
pub fn coarse_chroma_extra(qindex: f32, extra: f32, start: f32, end: f32) -> f32 {
    let span = (end - start).max(1.0);
    extra * ((qindex - start) / span).clamp(0.0, 1.0)
}

/// The chroma qindex change for a frame whose last measured luma/chroma
/// error ratio is `error_ratio` ([`plane_error_ratio`]): `slope` qindex per
/// dB that chroma is cleaner than luma beyond `pivot_db`, negative (finer
/// chroma) above the pivot, within [`GAP_OFFSET_LIMIT`]. Zero before a
/// ratio is measured.
#[must_use]
pub fn error_gap_offset(error_ratio: Option<f64>, slope: f32, pivot_db: f32) -> f32 {
    error_ratio.map_or(0.0, |ratio| {
        let gap_db = 10.0 * ratio.log10() as f32;
        (-slope * (gap_db - pivot_db)).clamp(-GAP_OFFSET_LIMIT, GAP_OFFSET_LIMIT)
    })
}

/// `MSE_Y / MSE_C` of `recon` against `source` on a row subsample, or `None`
/// when chroma has no error to compare with.
#[must_use]
pub fn plane_error_ratio(dsp: &Dsp, source: &Picture, recon: &Picture) -> Option<f64> {
    let mse = |plane| plane_mse(dsp, source, recon, plane);
    let chroma = f64::midpoint(mse(PlaneIndex::U), mse(PlaneIndex::V));
    (chroma > 0.0).then(|| mse(PlaneIndex::Y) / chroma)
}

/// Mean squared error of `plane` over every [`ERROR_ROW_PERIOD`]-th row of
/// whole [`ERROR_BLOCK`] blocks.
fn plane_mse(dsp: &Dsp, source: &Picture, recon: &Picture, plane: PlaneIndex) -> f64 {
    let size = BlockSize::new(ERROR_BLOCK, ERROR_BLOCK).expect("square block size");
    let (source, recon) = (source.plane(plane).view(), recon.plane(plane).view());
    let columns = source.width().min(recon.width()) / ERROR_BLOCK;
    let rows = source.height().min(recon.height()) / ERROR_BLOCK;
    let (mut sse, mut samples) = (0u64, 0u64);
    for row in (0..rows).step_by(ERROR_ROW_PERIOD) {
        for column in 0..columns {
            let (x, y) = (column * ERROR_BLOCK, row * ERROR_BLOCK);
            let source = source.sub_view(x, y, ERROR_BLOCK, ERROR_BLOCK);
            let recon = recon.sub_view(x, y, ERROR_BLOCK, ERROR_BLOCK);
            let (source, recon) = source
                .zip(recon)
                .expect("whole blocks lie inside the plane");
            sse += u64::from(dsp.variance(size, &source, &recon).sse);
            samples += (ERROR_BLOCK * ERROR_BLOCK) as u64;
        }
    }
    if samples == 0 {
        0.0
    } else {
        sse as f64 / samples as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_extra_ramps_between_start_and_end() {
        assert_eq!(coarse_chroma_extra(100.0, -12.0, 120.0, 180.0), 0.0);
        assert_eq!(coarse_chroma_extra(150.0, -12.0, 120.0, 180.0), -6.0);
        assert_eq!(coarse_chroma_extra(255.0, -12.0, 120.0, 180.0), -12.0);
        assert_eq!(coarse_chroma_extra(121.0, -12.0, 120.0, 120.0), -12.0);
    }

    #[test]
    fn gap_offset_follows_the_gap_within_its_limit() {
        // MSE_Y / MSE_C = 100 is a 20 dB gap: 10 dB past a 10 dB pivot.
        assert_eq!(error_gap_offset(Some(100.0), 1.0, 10.0), -10.0);
        // A 0 dB gap is 10 dB short of the pivot: coarser chroma.
        assert_eq!(error_gap_offset(Some(1.0), 1.0, 10.0), 10.0);
        assert_eq!(error_gap_offset(Some(1e6), 2.0, 10.0), -GAP_OFFSET_LIMIT);
        assert_eq!(error_gap_offset(None, 2.0, 10.0), 0.0);
        assert_eq!(error_gap_offset(Some(100.0), 0.0, 10.0), 0.0);
    }

    #[test]
    fn offset_fades_linearly_to_zero_at_the_scale() {
        assert_eq!(noise_attenuated_chroma_offset(-24.0, 0.0, 2.0), -24.0);
        assert_eq!(noise_attenuated_chroma_offset(-24.0, 1.0, 2.0), -12.0);
        assert_eq!(noise_attenuated_chroma_offset(-24.0, 2.0, 2.0), 0.0);
        assert_eq!(noise_attenuated_chroma_offset(-24.0, 9.0, 2.0), 0.0);
        assert_eq!(noise_attenuated_chroma_offset(-24.0, 9.0, 0.0), -24.0);
    }
}
