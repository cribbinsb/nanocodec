//! H.266 cross-component ALF correction of 4:2:0 chroma from prepared luma rows.
//! Arithmetic follows BSD-Clear `VVdeC` `AdaptiveLoopFilter`; see `THIRD_PARTY.md`.
impl crate::Dsp {
    /// Apply seven luma differences to a chroma row, with 8-bit correction clipping.
    /// Rows are ordered y−1, y, y+1, y+2 and have one horizontal padding sample.
    /// The codec prepares virtual-boundary row substitutions before this call.
    /// # Panics
    /// If rows are short or coefficients exceed the normative ±64 range.
    pub fn vvc_alf_cross_row(&self, rows: &[&[u8]; 4], coefficients: [i16; 7], out: &mut [u8]) {
        assert!(rows.iter().all(|r| r.len() >= 2 * out.len() + 2));
        assert!(coefficients.iter().all(|&c| (-64..=64).contains(&c)));
        for (x, pixel) in out.iter_mut().enumerate() {
            let x = 2 * x + 1;
            let centre = i32::from(rows[1][x]);
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
                .map(|(&c, sample)| i32::from(c) * (i32::from(sample) - centre))
                .sum();
            let correction = ((sum + 64) >> 7).clamp(-128, 127);
            *pixel = (i32::from(*pixel) + correction).clamp(0, 255) as u8;
        }
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
