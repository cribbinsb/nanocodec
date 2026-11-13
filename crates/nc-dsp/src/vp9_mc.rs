//! VP9 sub-sample motion-compensation kernel (VP9 §8.5.2.4; libvpx
//! `vpx_convolve8_c` with the `vp9_filter.c` kernels, BSD-3-Clause, see
//! `THIRD_PARTY.md`): 8-tap filters at 1/16-sample positions, a horizontal
//! pass rounded and clipped to 8 bits, then a vertical pass likewise.
//! Identity phases (position 0) make the corresponding pass a copy, so
//! always running both passes equals libvpx's copy/horizontal/vertical/2-D
//! dispatch.
//!
//! Scalar reference; SIMD versions belong beside it with a bit-exactness
//! test in nc-checkasm.

/// Filter taps.
pub const VP9_TAPS: usize = 8;
/// Taps before the centre sample: the window starts this far up and left.
pub const VP9_TAPS_BEFORE: usize = 3;
/// Sub-sample positions per sample.
pub const VP9_POSITIONS: usize = 16;
/// Filter output precision (`FILTER_BITS`).
pub(crate) const FILTER_BITS: u32 = 7;

type Kernel = [[i16; VP9_TAPS]; VP9_POSITIONS];

/// `sub_pel_filters_8` (`EIGHTTAP`, "regular").
const REGULAR: Kernel = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 1, -5, 126, 8, -3, 1, 0],
    [-1, 3, -10, 122, 18, -6, 2, 0],
    [-1, 4, -13, 118, 27, -9, 3, -1],
    [-1, 4, -16, 112, 37, -11, 4, -1],
    [-1, 5, -18, 105, 48, -14, 4, -1],
    [-1, 5, -19, 97, 58, -16, 5, -1],
    [-1, 6, -19, 88, 68, -18, 5, -1],
    [-1, 6, -19, 78, 78, -19, 6, -1],
    [-1, 5, -18, 68, 88, -19, 6, -1],
    [-1, 5, -16, 58, 97, -19, 5, -1],
    [-1, 4, -14, 48, 105, -18, 5, -1],
    [-1, 4, -11, 37, 112, -16, 4, -1],
    [-1, 3, -9, 27, 118, -13, 4, -1],
    [0, 2, -6, 18, 122, -10, 3, -1],
    [0, 1, -3, 8, 126, -5, 1, 0],
];

/// `sub_pel_filters_8lp` (`EIGHTTAP_SMOOTH`).
const SMOOTH: Kernel = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-3, -1, 32, 64, 38, 1, -3, 0],
    [-2, -2, 29, 63, 41, 2, -3, 0],
    [-2, -2, 26, 63, 43, 4, -4, 0],
    [-2, -3, 24, 62, 46, 5, -4, 0],
    [-2, -3, 21, 60, 49, 7, -4, 0],
    [-1, -4, 18, 59, 51, 9, -4, 0],
    [-1, -4, 16, 57, 53, 12, -4, -1],
    [-1, -4, 14, 55, 55, 14, -4, -1],
    [-1, -4, 12, 53, 57, 16, -4, -1],
    [0, -4, 9, 51, 59, 18, -4, -1],
    [0, -4, 7, 49, 60, 21, -3, -2],
    [0, -4, 5, 46, 62, 24, -3, -2],
    [0, -4, 4, 43, 63, 26, -2, -2],
    [0, -3, 2, 41, 63, 29, -2, -2],
    [0, -3, 1, 38, 64, 32, -1, -3],
];

/// `sub_pel_filters_8s` (`EIGHTTAP_SHARP`).
const SHARP: Kernel = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-1, 3, -7, 127, 8, -3, 1, 0],
    [-2, 5, -13, 125, 17, -6, 3, -1],
    [-3, 7, -17, 121, 27, -10, 5, -2],
    [-4, 9, -20, 115, 37, -13, 6, -2],
    [-4, 10, -23, 108, 48, -16, 8, -3],
    [-4, 10, -24, 100, 59, -19, 9, -3],
    [-4, 11, -24, 90, 70, -21, 10, -4],
    [-4, 11, -23, 80, 80, -23, 11, -4],
    [-4, 10, -21, 70, 90, -24, 11, -4],
    [-3, 9, -19, 59, 100, -24, 10, -4],
    [-3, 8, -16, 48, 108, -23, 10, -4],
    [-2, 6, -13, 37, 115, -20, 9, -4],
    [-2, 5, -10, 27, 121, -17, 7, -3],
    [-1, 3, -6, 17, 125, -13, 5, -2],
    [0, 1, -3, 8, 127, -7, 3, -1],
];

/// `bilinear_filters` (`BILINEAR`).
const BILINEAR: Kernel = {
    let mut kernel = [[0; VP9_TAPS]; VP9_POSITIONS];
    let mut phase = 0;
    while phase < VP9_POSITIONS {
        kernel[phase][3] = 128 - 8 * phase as i16;
        kernel[phase][4] = 8 * phase as i16;
        phase += 1;
    }
    kernel
};

/// VP9 interpolation filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp9Filter {
    /// `EIGHTTAP` (regular).
    Regular,
    /// `EIGHTTAP_SMOOTH`.
    Smooth,
    /// `EIGHTTAP_SHARP`.
    Sharp,
    /// `BILINEAR`.
    Bilinear,
}

impl Vp9Filter {
    pub(crate) fn kernel(self) -> &'static Kernel {
        match self {
            Self::Regular => &REGULAR,
            Self::Smooth => &SMOOTH,
            Self::Sharp => &SHARP,
            Self::Bilinear => &BILINEAR,
        }
    }
}

pub(crate) fn filter(samples: &[u8], taps: &[i16; VP9_TAPS], step: usize) -> u8 {
    let sum: i32 = taps
        .iter()
        .enumerate()
        .map(|(k, &t)| i32::from(t) * i32::from(samples[k * step]))
        .sum();
    ((sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS).clamp(0, 255) as u8
}

/// Predict a `width`x`height` block (each at most 64) at sub-sample phase
/// `(frac_x, frac_y)` (1/16) from `window`, whose top-left sample lies
/// [`VP9_TAPS_BEFORE`] up and left of the block's integer position and
/// which covers `height + 7` rows of `width + 7` samples at `stride`.
/// Output rows begin `dst_stride` bytes apart.
///
/// # Panics
///
/// If the window or `dst` is too small, a size exceeds 64 or a phase 15.
pub fn vp9_convolve(
    filter_kind: Vp9Filter,
    window: &[u8],
    stride: usize,
    phase: (usize, usize),
    size: (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
) {
    vp9_convolve_into(
        filter_kind,
        window,
        stride,
        phase,
        size,
        dst,
        dst_stride,
        false,
    );
}

/// Scalar oracle for strided prediction and averaging with the first reference.
pub(crate) fn vp9_convolve_into(
    filter_kind: Vp9Filter,
    window: &[u8],
    stride: usize,
    (frac_x, frac_y): (usize, usize),
    (width, height): (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
    average: bool,
) {
    const MAX: usize = 64 + VP9_TAPS - 1;
    assert!(width <= 64 && height <= 64);
    let kernel = filter_kind.kernel();
    let centre = VP9_TAPS_BEFORE * stride + VP9_TAPS_BEFORE;
    // A zero phase is the identity filter: its pass is a copy (exactly
    // libvpx's copy / horizontal-only / vertical-only dispatch).
    match (frac_x, frac_y) {
        (0, 0) => {
            for y in 0..height {
                let row = &window[centre + y * stride..][..width];
                for (out, &pixel) in dst[y * dst_stride..y * dst_stride + width]
                    .iter_mut()
                    .zip(row)
                {
                    store(out, pixel, average);
                }
            }
        }
        (_, 0) => {
            for y in 0..height {
                let line = &window[centre - VP9_TAPS_BEFORE + y * stride..];
                for x in 0..width {
                    store(
                        &mut dst[y * dst_stride + x],
                        filter(&line[x..], &kernel[frac_x], 1),
                        average,
                    );
                }
            }
        }
        (0, _) => {
            for y in 0..height {
                for x in 0..width {
                    let column = &window[VP9_TAPS_BEFORE + x + y * stride..];
                    store(
                        &mut dst[y * dst_stride + x],
                        filter(column, &kernel[frac_y], stride),
                        average,
                    );
                }
            }
        }
        _ => {
            let mut temp = [0u8; MAX * 64];
            // Horizontal pass over every window row the vertical filter reads.
            for row in 0..height + VP9_TAPS - 1 {
                let line = &window[row * stride..];
                for x in 0..width {
                    temp[row * width + x] = filter(&line[x..], &kernel[frac_x], 1);
                }
            }
            for y in 0..height {
                for x in 0..width {
                    store(
                        &mut dst[y * dst_stride + x],
                        filter(&temp[y * width + x..], &kernel[frac_y], width),
                        average,
                    );
                }
            }
        }
    }
}

/// VP9 averages the two already clipped predictions with upward rounding.
#[inline]
fn store(dst: &mut u8, pixel: u8, average: bool) {
    *dst = if average {
        ((u16::from(*dst) + u16::from(pixel) + 1) >> 1) as u8
    } else {
        pixel
    };
}

/// Largest scaled reference step (`x_step_q4`): a reference at most twice
/// the frame size advances at most 2 samples per output sample.
pub const VP9_MAX_SCALED_STEP: usize = 32;

/// Scaled prediction from a reference of a different size (libvpx
/// `vpx_scaled_2d_c` / `convolve`): output sample `x` filters at
/// 1/16-sample position `x0_q4 + x * x_step` (likewise vertically), a
/// horizontal pass rounded to 8 bits over every row the vertical pass
/// reads, then the vertical pass. `window` starts [`VP9_TAPS_BEFORE`] up
/// and left of the integer start and covers
/// `((phase + (size - 1) * step) >> 4) + 8` samples on each axis.
///
/// # Panics
///
/// If a size exceeds 64, a step is zero or exceeds
/// [`VP9_MAX_SCALED_STEP`], a phase is 16 or more, or a buffer is too small.
pub fn vp9_convolve_scaled(
    filter_kind: Vp9Filter,
    window: &[u8],
    stride: usize,
    (x0_q4, x_step): (usize, usize),
    (y0_q4, y_step): (usize, usize),
    (width, height): (usize, usize),
    dst: &mut [u8],
) {
    const SUBPEL_BITS: usize = 4;
    const SUBPEL_MASK: usize = VP9_POSITIONS - 1;
    const MAX_SCALED_ROWS: usize =
        (((64 - 1) * VP9_MAX_SCALED_STEP + SUBPEL_MASK) >> SUBPEL_BITS) + VP9_TAPS;
    assert!(width <= 64 && height <= 64);
    assert!(
        (1..=VP9_MAX_SCALED_STEP).contains(&x_step) && (1..=VP9_MAX_SCALED_STEP).contains(&y_step)
    );
    assert!(x0_q4 < VP9_POSITIONS && y0_q4 < VP9_POSITIONS);
    let kernel = filter_kind.kernel();
    let rows = (((height - 1) * y_step + y0_q4) >> SUBPEL_BITS) + VP9_TAPS;
    let mut temp_buf = [0u8; MAX_SCALED_ROWS * 64];
    let temp = &mut temp_buf[..rows * width];
    for (row, line) in temp.chunks_exact_mut(width).enumerate() {
        let source = &window[row * stride..];
        for (x, out) in line.iter_mut().enumerate() {
            let q4 = x0_q4 + x * x_step;
            *out = filter(&source[q4 >> SUBPEL_BITS..], &kernel[q4 & SUBPEL_MASK], 1);
        }
    }
    for y in 0..height {
        let q4 = y0_q4 + y * y_step;
        let taps = &kernel[q4 & SUBPEL_MASK];
        for x in 0..width {
            dst[y * width + x] = filter(&temp[(q4 >> SUBPEL_BITS) * width + x..], taps, width);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_steps_scale_like_the_plain_convolution() {
        let window: Vec<u8> = (0..23 * 23).map(|i| (i * 37 % 251) as u8).collect();
        for kind in [Vp9Filter::Regular, Vp9Filter::Sharp, Vp9Filter::Bilinear] {
            for phase in [(0, 0), (5, 0), (0, 9), (3, 15)] {
                let (mut plain, mut scaled) = ([0u8; 256], [0u8; 256]);
                vp9_convolve(kind, &window, 23, phase, (16, 16), &mut plain, 16);
                vp9_convolve_scaled(
                    kind,
                    &window,
                    23,
                    (phase.0, 16),
                    (phase.1, 16),
                    (16, 16),
                    &mut scaled,
                );
                assert_eq!(plain, scaled);
            }
        }
    }

    #[test]
    fn every_kernel_phase_sums_to_unity() {
        for kernel in [&REGULAR, &BILINEAR] {
            for taps in kernel {
                assert_eq!(taps.iter().map(|&t| i32::from(t)).sum::<i32>(), 128);
            }
        }
    }

    #[test]
    fn whole_sample_positions_copy_and_passes_round_separately() {
        let window: Vec<u8> = (0..15 * 15).map(|i| (i * 37 % 251) as u8).collect();
        let mut out = [0u8; 64];
        vp9_convolve(Vp9Filter::Regular, &window, 15, (0, 0), (8, 8), &mut out, 8);
        for y in 0..8 {
            assert_eq!(
                &out[y * 8..y * 8 + 8],
                &window[(y + 3) * 15 + 3..(y + 3) * 15 + 11]
            );
        }
        // Bilinear half-sample in both directions on [[0, 1], [1, 2]]:
        // horizontal rounding gives [1, 2], then vertical rounding 2.
        let mut tiny = vec![0u8; 8 * 8];
        tiny[3 * 8 + 3] = 0;
        tiny[3 * 8 + 4] = 1;
        tiny[4 * 8 + 3] = 1;
        tiny[4 * 8 + 4] = 2;
        let mut one = [0u8; 1];
        vp9_convolve(Vp9Filter::Bilinear, &tiny, 8, (8, 8), (1, 1), &mut one, 1);
        assert_eq!(one[0], 2);
    }
}
