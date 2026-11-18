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
    /// [`crate::DepthKernels::vvc_bdof`] for deeper samples.
    /// Inputs include a one-sample nearest-integer border and six fractional bits.
    /// # Panics
    /// If dimensions are not positive multiples of four up to 16, or buffers are short.
    pub fn vvc_bdof(&self, size: (usize, usize), inputs: [&[i16]; 2], out: &mut [u8]) {
        check(size, inputs, out.len());
        // SAFETY: dimensions and all buffers checked; instance ISA is supported.
        unsafe {
            (self.vvc_refinement_kernels().bdof)(size, inputs, out);
        }
    }

    /// Assemble the `(width + 2) x (height + 2)` 14-bit BDOF/PROF window from
    /// 8-bit nearest-integer `window` borders (`sample << 6`) and `interior`
    /// (`width x height` high-precision predictions).
    ///
    /// # Panics
    /// If `size` is not in `{4, 8, 12, 16}^2`, or `window`/`interior`/`out` is short.
    pub fn vvc_bdof_pad(
        &self,
        size: (usize, usize),
        window: &nc_common::PlaneView<'_>,
        interior: &[i16],
        out: &mut [i16],
    ) {
        check_pad(size, window, interior, out);
        // SAFETY: dimensions and all buffers checked; instance ISA is supported.
        unsafe {
            (self.vvc_refinement_kernels().bdof_pad)(size, window, interior, out);
        }
    }
}

/// The checks of [`Dsp::vvc_bdof`] (`out_len` samples of any type).
pub(crate) fn check((width, height): (usize, usize), inputs: [&[i16]; 2], out_len: usize) {
    assert!([4, 8, 12, 16].contains(&width) && [4, 8, 12, 16].contains(&height));
    let stride = width + 2;
    let count = stride * (height + 2);
    assert!(inputs.iter().all(|p| p.len() >= count) && out_len >= width * height);
}

/// Precision of high-precision BDOF/PROF border samples (H.266 §8.5.6.3.1).
const PAD_PRECISION: u32 = 14;
/// Minimum left shift of integer border samples (`Max(2, 14 - BitDepth)`).
const MIN_PAD_SHIFT: u32 = 2;

/// The checks of [`Dsp::vvc_bdof_pad`].
pub(crate) fn check_pad<T: nc_common::Sample>(
    (width, height): (usize, usize),
    window: &nc_common::PlaneView<'_, T>,
    interior: &[i16],
    out: &[i16],
) {
    assert!([4, 8, 12, 16].contains(&width) && [4, 8, 12, 16].contains(&height));
    let (pw, ph) = (width + 2, height + 2);
    assert!(window.width() >= pw && window.height() >= ph);
    assert!(interior.len() >= width * height && out.len() >= pw * ph);
}

/// Scalar 8-bit [`Dsp::vvc_bdof_pad`].
pub(crate) fn pad_scalar(
    size: (usize, usize),
    window: &nc_common::PlaneView<'_>,
    interior: &[i16],
    out: &mut [i16],
) {
    pad_scalar_at(size, window, interior, out, nc_common::BitDepth::EIGHT);
}

/// Scalar [`Dsp::vvc_bdof_pad`] at `depth`.
pub(crate) fn pad_scalar_at<T: nc_common::Sample>(
    (width, height): (usize, usize),
    window: &nc_common::PlaneView<'_, T>,
    interior: &[i16],
    out: &mut [i16],
    depth: nc_common::BitDepth,
) {
    let shift = PAD_PRECISION
        .saturating_sub(depth.bits())
        .max(MIN_PAD_SHIFT);
    let stride = width + 2;
    let top = window.row(0);
    for x in 0..stride {
        out[x] = (top[x].to_i32() << shift) as i16;
    }
    for y in 0..height {
        let src = window.row(y + 1);
        let dst = &mut out[(y + 1) * stride..(y + 2) * stride];
        dst[0] = (src[0].to_i32() << shift) as i16;
        dst[1..=width].copy_from_slice(&interior[y * width..(y + 1) * width]);
        dst[width + 1] = (src[width + 1].to_i32() << shift) as i16;
    }
    let bot = window.row(height + 1);
    let bot_dst = &mut out[(height + 1) * stride..(height + 2) * stride];
    for x in 0..stride {
        bot_dst[x] = (bot[x].to_i32() << shift) as i16;
    }
}

/// Fewest bits the final sum drops (H.266 §8.5.6.5: `shift4 = Max(3, 15 −
/// BitDepth)`).
const MIN_FINAL_SHIFT: u32 = 3;
/// Two 14-bit predictions summed: `shift4 = 15 − BitDepth` leaves the
/// sample's bits.
const SUM_PRECISION: u32 = 15;

/// The scalar 8-bit kernel (`Kernels::bdof`).
pub(crate) fn scalar(size: (usize, usize), inputs: [&[i16]; 2], out: &mut [u8]) {
    scalar_at(size, inputs, out, nc_common::BitDepth::EIGHT);
}

/// [`Dsp::vvc_bdof`] at `depth` (H.266 §8.5.6.5). Gradients, sums and the
/// motion refinement work on the 14-bit predictions with fixed shifts, the
/// same at every depth; only the final `(a + b + bdofOffset + offset4) >>
/// shift4` and its clip depend on it.
pub(crate) fn scalar_at<T: nc_common::Sample>(
    size: (usize, usize),
    inputs: [&[i16]; 2],
    out: &mut [T],
    depth: nc_common::BitDepth,
) {
    let shift4 = SUM_PRECISION
        .saturating_sub(depth.bits())
        .max(MIN_FINAL_SHIFT);
    let offset4 = 1 << (shift4 - 1);
    let (width, height) = size;
    let stride = width + 2;
    let count = stride * (height + 2);
    let mut samples = [[0i16; MAX_PADDED]; 2];
    for list in 0..2 {
        samples[list][..count].copy_from_slice(&inputs[list][..count]);
    }
    let mut gradients = [[[0i16; 2]; MAX_PADDED]; 2];
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
                        (i32::from(gradients[0][at][axis]) + i32::from(gradients[1][at][axis])) >> 1
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
                            vector[axis]
                                * (i32::from(gradients[0][at][axis])
                                    - i32::from(gradients[1][at][axis]))
                        })
                        .sum();
                    let value = i32::from(samples[0][at]) + i32::from(samples[1][at]) + correction;
                    out[y * width + x] = depth.clip_to((value + offset4) >> shift4);
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
