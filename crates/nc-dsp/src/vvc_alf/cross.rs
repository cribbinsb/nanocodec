//! H.266 cross-component ALF correction of 4:2:0 chroma from prepared luma rows.
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

impl crate::Dsp {
    /// Apply seven luma differences to a chroma row, with 8-bit correction clipping.
    /// Rows are ordered y−1, y, y+1, y+2 and have one horizontal padding sample.
    /// The codec prepares virtual-boundary row substitutions before this call.
    /// # Panics
    /// If rows are short or coefficients exceed the normative ±64 range.
    pub fn vvc_alf_cross_row(&self, rows: &[&[u8]; 4], coefficients: [i16; 7], out: &mut [u8]) {
        check_cross_row(rows, coefficients, out);
        cross_row_at(rows, coefficients, out, BitDepth::EIGHT);
    }
}

/// The checks of the CC-ALF row (any sample type).
pub(crate) fn check_cross_row<T: Sample>(
    rows: &[&[T]; 4],
    coefficients: [i16; CROSS_TAPS],
    out: &[T],
) {
    assert!(rows.iter().all(|r| r.len() >= 2 * out.len() + 2));
    assert!(coefficients.iter().all(|c| COEFFICIENTS.contains(c)));
}

/// The CC-ALF row at any sample type and depth (§8.8.5.7): the correction
/// is clipped to `−(1 << (BitDepth − 1))..=(1 << (BitDepth − 1)) − 1`, the
/// corrected sample to the depth. VVC has one `BitDepth` for luma and
/// chroma (`sps_bitdepth_minus8`), so `depth` serves both.
pub(crate) fn cross_row_at<T: Sample>(
    rows: &[&[T]; 4],
    coefficients: [i16; CROSS_TAPS],
    out: &mut [T],
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    let half = depth.mid();
    for (x, pixel) in out.iter_mut().enumerate() {
        let x = 2 * x + 1;
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
}
