//! VVC prediction refinement arithmetic shared by decoder motion controllers.
//! PROF follows BSD-Clear `VVdeC` `InterPrediction`; see `THIRD_PARTY.md`.
#![allow(unsafe_code)]
pub(super) fn gradient(samples: &[i16], stride: usize, at: usize) -> [i32; 2] {
    [
        i32::from(samples[at + 1] >> 6) - i32::from(samples[at - 1] >> 6),
        i32::from(samples[at + stride] >> 6) - i32::from(samples[at - stride] >> 6),
    ]
}

impl crate::Dsp {
    /// Refine a 4×4 affine luma prediction from its padded 6×6 high-precision samples.
    /// Samples retain six fractional bits; integer reference samples supply the border.
    /// # Panics
    /// If a correction is outside the normative ±31 range.
    pub fn vvc_prof_refine(
        &self,
        samples: &[i16; 36],
        corrections: &[[i32; 2]; 16],
        out: &mut [i16; 16],
    ) {
        assert!(
            corrections
                .iter()
                .flatten()
                .all(|&v| (-31..=31).contains(&v))
        );
        // SAFETY: corrections validated; fixed arrays provide complete windows.
        unsafe {
            (self.vvc_refinement_kernels().prof)(samples, corrections, out);
        }
    }
}

pub(crate) fn scalar(samples: &[i16; 36], corrections: &[[i32; 2]; 16], out: &mut [i16; 16]) {
    for (index, value) in out.iter_mut().enumerate() {
        let at = (index / 4 + 1) * 6 + index % 4 + 1;
        let [gx, gy] = gradient(samples, 6, at);
        let correction =
            (gx * corrections[index][0] + gy * corrections[index][1]).clamp(-8192, 8191);
        *value = (i32::from(samples[at]) + correction).clamp(-32768, 32767) as i16;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn captured_prof_blocks_match_reference() {
        for dsp in crate::Isa::ALL.into_iter().filter_map(crate::Dsp::with_isa) {
            for &(bidirectional, samples, corrections, expected) in super::fixtures::CASES {
                let mut actual = [0; 16];
                dsp.vvc_prof_refine(&samples, &corrections, &mut actual);
                if !bidirectional {
                    actual = actual.map(|v| ((i32::from(v) + 32) >> 6).clamp(0, 255) as i16);
                }
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    #[ignore = "requires NC_VVC_OPTICAL_TRACE from independent VVdeC"]
    fn prepared_prof_matches_reference() {
        let trace =
            std::fs::read_to_string(std::env::var("NC_VVC_OPTICAL_TRACE").unwrap()).unwrap();
        let mut lines = trace.lines().filter(|line| line.starts_with("NC_PROF"));
        let dsp = crate::Dsp::new();
        let mut count = 0;
        while let Some(line) = lines.next() {
            let Some(line) = line.strip_prefix("NC_PROF ") else {
                continue;
            };
            let values: Vec<i32> = line
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(values.len(), 69);
            let samples = std::array::from_fn(|i| values[1 + i] as i16);
            let corrections = std::array::from_fn(|i| [values[37 + 2 * i], values[38 + 2 * i]]);
            let expected: Vec<i16> = lines
                .next()
                .unwrap()
                .strip_prefix("NC_PROF_OUT ")
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            let mut actual = [0; 16];
            dsp.vvc_prof_refine(&samples, &corrections, &mut actual);
            if values[0] == 0 {
                actual = actual.map(|v| ((i32::from(v) + 32) >> 6).clamp(0, 255) as i16);
            }
            assert_eq!(actual.as_slice(), expected, "PROF block {count}");
            count += 1;
        }
        assert!(count > 0);
        eprintln!("{count} prepared PROF blocks match VVdeC");
    }
}

#[cfg(test)]
#[path = "vvc_optical_flow_fixtures.rs"]
mod fixtures;
