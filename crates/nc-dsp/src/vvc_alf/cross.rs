//! H.266 cross-component ALF correction of chroma from prepared luma rows:
//! horizontally subsampled chroma (4:2:0, 4:2:2; SIMD tiers) and 4:4:4.
//! Arithmetic follows BSD-Clear `VVdeC` `AdaptiveLoopFilter`; see `THIRD_PARTY.md`.
//! The reference is generic over the sample type with a runtime depth.
use nc_common::{BitDepth, Sample};

/// Taps of the CC-ALF filter (§8.8.5.7 `CcAlfCoeff`).
const CROSS_TAPS: usize = 7;
/// The range of a CC-ALF coefficient (§7.4.3.18: `±2^6`).
const COEFFICIENTS: std::ops::RangeInclusive<i16> = -64..=64;
/// `log2` of the weight of a CC-ALF coefficient (§8.8.5.7: the sum is
/// rounded by 7 bits).
const SUM_SHIFT: u32 = 7;

/// The checks of the CC-ALF row (any sample type).
pub(crate) fn check_cross_row<T: Sample>(
    rows: &[&[T]; 4],
    coefficients: [i16; CROSS_TAPS],
    out: &[T],
) {
    check_cross_row_step(rows, coefficients, out, SUBSAMPLED_STEP);
}

/// The checks of a CC-ALF row whose chroma samples are `step` luma samples
/// apart.
pub(crate) fn check_cross_row_step<T: Sample>(
    rows: &[&[T]; 4],
    coefficients: [i16; CROSS_TAPS],
    out: &[T],
    step: usize,
) {
    assert!(rows.iter().all(|r| r.len() >= step * out.len() + 2));
    assert!(coefficients.iter().all(|c| COEFFICIENTS.contains(c)));
}

/// Luma samples per chroma sample of horizontally subsampled chroma
/// (`SubWidthC` 2).
const SUBSAMPLED_STEP: usize = 2;

/// The CC-ALF row of horizontally subsampled chroma (the kernel tables'
/// entry; [`cross_row_step_at`]).
pub(crate) fn cross_row_at<T: Sample>(
    rows: &[&[T]; 4],
    coefficients: [i16; CROSS_TAPS],
    out: &mut [T],
    depth: BitDepth,
) {
    cross_row_step_at(rows, coefficients, (out, SUBSAMPLED_STEP), depth);
}

/// The CC-ALF row at any sample type and depth (§8.8.5.7) of chroma whose
/// samples are `step` (`SubWidthC`) luma samples apart, `rows` starting one
/// luma sample left of the first one's: the correction is clipped to
/// `−(1 << (BitDepth − 1))..=(1 << (BitDepth − 1)) − 1`, the corrected
/// sample to the depth. VVC has one `BitDepth` for luma and chroma
/// (`sps_bitdepth_minus8`), so `depth` serves both.
pub(crate) fn cross_row_step_at<T: Sample>(
    rows: &[&[T]; 4],
    coefficients: [i16; CROSS_TAPS],
    (out, step): (&mut [T], usize),
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    let half = depth.mid();
    for (x, pixel) in out.iter_mut().enumerate() {
        let x = step * x + 1;
        let centre = rows[1][x].to_i32();
        let neighbours = [
            rows[0][x],
            rows[1][x - 1],
            rows[1][x + 1],
            rows[2][x - 1],
            rows[2][x],
            rows[2][x + 1],
            rows[3][x],
        ];
        let sum: i32 = coefficients
            .iter()
            .zip(neighbours)
            .map(|(&c, sample)| i32::from(c) * (sample.to_i32() - centre))
            .sum();
        let correction = ((sum + (1 << (SUM_SHIFT - 1))) >> SUM_SHIFT).clamp(-half, half - 1);
        *pixel = depth.clip_to(pixel.to_i32() + correction);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn correction_clips_before_addition_and_handles_both_signs() {
        let dsp = crate::Dsp::new();
        let middle = [0, 0, 0, 255, 0, 100];
        let upper = [0, 255, 0, 0, 0, 0];
        let lower = [255; 6];
        let rows = [&upper[..], &middle[..], &lower[..], &lower[..]];
        let mut output = [40, 200];
        dsp.vvc_alf_cross_row(&rows, [64, 0, 0, 0, 0, 0, 0], &mut output);
        assert_eq!(output, [167, 73]);
        let mut output = [40, 200];
        dsp.vvc_alf_cross_row(&rows, [64, 64, 64, 64, 64, 64, 64], &mut output);
        assert_eq!(output, [167, 72]);
    }

    #[test]
    fn unsubsampled_rows_step_one_luma_sample() {
        let depth = nc_common::BitDepth::EIGHT;
        let middle = [0, 0, 255, 0];
        let upper = [0, 255, 0, 0];
        let rows = [&upper[..], &middle[..], &middle[..], &middle[..]];
        let mut output = [40u8, 200];
        super::cross_row_step_at(&rows, [64, 0, 0, 0, 0, 0, 0], (&mut output, 1), depth);
        // Chroma 0 sees luma 1 (above it 255), chroma 1 luma 2 (255, above 0).
        assert_eq!(output, [167, 73]);
    }
}
