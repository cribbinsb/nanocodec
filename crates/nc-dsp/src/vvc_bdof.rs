//! H.266 bidirectional optical-flow arithmetic on prepared 14-bit luma predictions.
//! BSD-Clear `VVdeC` `InterPrediction` is the arithmetic reference.
#![allow(unsafe_code)]
use crate::Dsp;

use crate::vvc_optical_flow::gradient;

/// Maximum 16×16 region with its one-sample border.
pub(crate) const MAX_PADDED: usize = 18 * 18;

pub(crate) fn extend<T: Copy>(samples: &mut [T], width: usize, height: usize) {
    let stride = width + 2;
    for y in 1..=height {
        samples[y * stride] = samples[y * stride + 1];
        samples[y * stride + width + 1] = samples[y * stride + width];
    }
    samples.copy_within(stride..2 * stride, 0);
    samples.copy_within(
        height * stride..(height + 1) * stride,
        (height + 1) * stride,
    );
}
pub(crate) fn flow(sums: [i32; 5]) -> [i32; 2] {
    let [gx, gy, ix, iy, cross] = sums;
    let divide = |numerator: i32, denominator: i32| {
        if denominator == 0 {
            0
        } else {
            (numerator >> denominator.ilog2()).clamp(-15, 15)
        }
    };
    let x = divide(4 * ix, gx);
    [x, divide(4 * iy - ((x * cross) >> 1), gy)]
}
impl Dsp {
    /// Apply BDOF to an 8-bit luma region, at most 16×16 samples.
    /// Inputs include a one-sample nearest-integer border and six fractional bits.
    /// # Panics
    /// If dimensions are not positive multiples of four up to 16, or buffers are short.
    pub fn vvc_bdof(&self, size: (usize, usize), inputs: [&[i16]; 2], out: &mut [u8]) {
        let (width, height) = size;
        assert!([4, 8, 12, 16].contains(&width) && [4, 8, 12, 16].contains(&height));
        let stride = width + 2;
        let count = stride * (height + 2);
        assert!(inputs.iter().all(|p| p.len() >= count) && out.len() >= width * height);
        // SAFETY: dimensions and all buffers checked; instance ISA is supported.
        unsafe {
            (self.vvc_refinement_kernels().bdof)(size, inputs, out);
        }
    }
}

pub(crate) fn scalar(size: (usize, usize), inputs: [&[i16]; 2], out: &mut [u8]) {
    let (width, height) = size;
    let stride = width + 2;
    let count = stride * (height + 2);
    let mut samples = [[0i16; MAX_PADDED]; 2];
    for list in 0..2 {
        samples[list][..count].copy_from_slice(&inputs[list][..count]);
    }
    let mut gradients = [[[0i32; 2]; MAX_PADDED]; 2];
    for list in 0..2 {
        for y in 1..=height {
            for x in 1..=width {
                let at = y * stride + x;
                gradients[list][at] = gradient(&samples[list], stride, at);
            }
        }
        // Gradients use the integer border; flow sums instead extend the
        // resulting interior gradients and interpolated prediction values.
        extend(&mut gradients[list], width, height);
        extend(&mut samples[list], width, height);
    }
    for y0 in (0..height).step_by(4) {
        for x0 in (0..width).step_by(4) {
            let mut sums = [0; 5];
            for y in y0..y0 + 6 {
                for x in x0..x0 + 6 {
                    let at = y * stride + x;
                    let [gx, gy] = std::array::from_fn(|axis| {
                        (gradients[0][at][axis] + gradients[1][at][axis]) >> 1
                    });
                    let difference =
                        i32::from(samples[1][at] >> 4) - i32::from(samples[0][at] >> 4);
                    sums[0] += gx.abs();
                    sums[1] += gy.abs();
                    sums[2] += gx.signum() * difference;
                    sums[3] += gy.signum() * difference;
                    sums[4] += gy.signum() * gx;
                }
            }
            let vector = flow(sums);
            for y in y0..y0 + 4 {
                for x in x0..x0 + 4 {
                    let at = (y + 1) * stride + x + 1;
                    let correction: i32 = (0..2)
                        .map(|axis| {
                            vector[axis] * (gradients[0][at][axis] - gradients[1][at][axis])
                        })
                        .sum();
                    let value = i32::from(samples[0][at]) + i32::from(samples[1][at]) + correction;
                    out[y * width + x] = ((value + 64) >> 7).clamp(0, 255) as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn captured_bdof_blocks_match_reference() {
        for dsp in crate::Isa::ALL.into_iter().filter_map(crate::Dsp::with_isa) {
            for &(size, first, second, expected) in super::fixtures::CASES {
                let mut actual = vec![0; size.0 * size.1];
                dsp.vvc_bdof(size, [first, second], &mut actual);
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    #[ignore = "requires NC_VVC_OPTICAL_TRACE from independent VVdeC"]
    fn prepared_bdof_matches_reference() {
        let trace =
            std::fs::read_to_string(std::env::var("NC_VVC_OPTICAL_TRACE").unwrap()).unwrap();
        let mut lines = trace.lines().filter(|line| line.starts_with("NC_BDOF"));
        let dsp = crate::Dsp::new();
        let mut count = 0;
        while let Some(line) = lines.next() {
            let values: Vec<i32> = line
                .strip_prefix("NC_BDOF ")
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            let (width, height) = (values[0] as usize, values[1] as usize);
            let size = (width + 2) * (height + 2);
            assert_eq!(values.len(), 2 + 2 * size);
            let samples: Vec<i16> = values[2..].iter().map(|&v| v as i16).collect();
            let expected: Vec<u8> = lines
                .next()
                .unwrap()
                .strip_prefix("NC_BDOF_OUT ")
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            let mut actual = vec![0; width * height];
            dsp.vvc_bdof(
                (width, height),
                [&samples[..size], &samples[size..]],
                &mut actual,
            );
            assert_eq!(actual, expected, "BDOF block {count} ({width}x{height})");
            count += 1;
        }
        assert!(count > 0);
        eprintln!("{count} prepared BDOF blocks match VVdeC");
    }
}

#[cfg(test)]
#[path = "vvc_bdof_fixtures.rs"]
mod fixtures;
