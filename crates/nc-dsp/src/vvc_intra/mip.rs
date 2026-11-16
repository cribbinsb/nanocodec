//! H.266 matrix intra prediction, using normative weights and separable upsampling.
//! Algorithms checked against BSD-Clear `VVdeC` `MatrixIntraPrediction`.
//! Also the BDPCM prediction (a copy of one reference edge).
use nc_common::{BitDepth, Sample};

use crate::Dsp;
#[path = "mip_weights.rs"]
mod weights;

impl Dsp {
    /// Predict a matrix-intra block from substituted, unfiltered top/left edges.
    /// # Panics
    /// If dimensions are not powers of two in 4..=64, mode is invalid, an edge
    /// sample is outside 0..=255, or a buffer is too short.
    pub fn vvc_mip_predict(
        &self,
        size: (usize, usize),
        mode: u8,
        transpose: bool,
        top: &[i32],
        left: &[i32],
        out: &mut [u8],
    ) {
        mip_predict(size, mode, transpose, (top, left), out, BitDepth::EIGHT);
    }

    /// Predict block differential PCM from the selected reference edge, without PDPC.
    /// # Panics
    /// If dimensions, direction, edges or output length are invalid.
    pub fn vvc_bdpcm_predict(
        &self,
        size: (usize, usize),
        vertical: bool,
        top: &[i32],
        left: &[i32],
        out: &mut [u8],
    ) {
        bdpcm_predict(size, vertical, (top, left), out, BitDepth::EIGHT);
    }
}

/// [`Dsp::vvc_mip_predict`] at `depth` (H.266 §8.4.5.2.2): the first input
/// of the two smaller size classes is `(1 << (BitDepth − 1)) − pTemp[0]`
/// and the reduced prediction clips to the depth; the upsampling
/// interpolates in-range samples and needs no clip.
///
/// # Panics
/// As [`Dsp::vvc_mip_predict`], edge samples outside `0..=(1 << BitDepth) − 1`.
pub(crate) fn mip_predict<T: Sample>(
    (width, height): (usize, usize),
    mode: u8,
    transpose: bool,
    (top, left): (&[i32], &[i32]),
    out: &mut [T],
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    let valid = |n: usize| n.is_power_of_two() && (4..=64).contains(&n);
    assert!(
        valid(width)
            && valid(height)
            && top.len() >= width
            && left.len() >= height
            && out.len() >= width * height
    );
    assert!(
        top[..width]
            .iter()
            .chain(&left[..height])
            .all(|&v| (0..=depth.max()).contains(&v))
    );
    let class = if width == 4 && height == 4 {
        0
    } else if width == 4 || height == 4 || (width == 8 && height == 8) {
        1
    } else {
        2
    };
    assert!(usize::from(mode) < [16, 8, 6][class]);
    let edge_size = if class == 0 { 2 } else { 4 };
    let side = if class < 2 { 4 } else { 8 };
    let mut input = [0; 8];
    let (first, second) = if transpose {
        (&left[..height], &top[..width])
    } else {
        (&top[..width], &left[..height])
    };
    downsample(first, &mut input[..edge_size]);
    downsample(second, &mut input[edge_size..2 * edge_size]);
    let origin = input[0];
    for value in &mut input[..2 * edge_size] {
        *value -= origin;
    }
    input[0] = if class < 2 { depth.mid() - origin } else { 0 };
    let offset = 32 - 32 * input[..2 * edge_size].iter().sum::<i32>();
    let mut reduced = [T::default(); 64];
    for position in 0..side * side {
        let row: &[u8] = match class {
            0 => &weights::SMALL[usize::from(mode)][position],
            1 => &weights::MEDIUM[usize::from(mode)][position],
            _ => &weights::LARGE[usize::from(mode)][position],
        };
        let dot: i32 = row
            .iter()
            .zip(&input[usize::from(class == 2)..])
            .map(|(&weight, &value)| i32::from(weight) * value)
            .sum();
        let destination = if transpose {
            (position % side) * side + position / side
        } else {
            position
        };
        reduced[destination] = depth.clip_to(((dot + offset) >> 6) + origin);
    }
    upsample((width, height), side, &reduced, top, left, out);
}

/// [`Dsp::vvc_bdpcm_predict`] at `depth`: BDPCM predicts as the pure
/// horizontal or vertical direction (H.266 §8.4.5.2.13, predModeIntra 18
/// or 50 without PDPC), a copy of one reference edge clipped to the depth.
///
/// # Panics
/// As [`Dsp::vvc_bdpcm_predict`].
pub(crate) fn bdpcm_predict<T: Sample>(
    (width, height): (usize, usize),
    vertical: bool,
    (top, left): (&[i32], &[i32]),
    out: &mut [T],
    depth: BitDepth,
) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    let valid = |n: usize| n.is_power_of_two() && (2..=32).contains(&n);
    assert!(
        valid(width)
            && valid(height)
            && top.len() >= width
            && left.len() >= height
            && out.len() >= width * height
    );
    for (y, row) in out[..width * height].chunks_exact_mut(width).enumerate() {
        for (x, destination) in row.iter_mut().enumerate() {
            *destination = depth.clip_to(if vertical { top[x] } else { left[y] });
        }
    }
}

fn downsample(source: &[i32], out: &mut [i32]) {
    let step = source.len() / out.len();
    for (dest, group) in out.iter_mut().zip(source.chunks_exact(step)) {
        *dest = (group.iter().sum::<i32>() + (step / 2) as i32) >> step.ilog2();
    }
}
fn upsample<T: Sample>(
    (width, height): (usize, usize),
    side: usize,
    reduced: &[T],
    top: &[i32],
    left: &[i32],
    out: &mut [T],
) {
    let (horizontal, vertical) = (width / side, height / side);
    let mut rows = [0i32; 64 * 8];
    for y in 0..side {
        let mut before = left[(y + 1) * vertical - 1];
        for x in 0..side {
            let behind = reduced[y * side + x].to_i32();
            for step in 1..=horizontal {
                rows[y * width + x * horizontal + step - 1] = ((horizontal - step) as i32 * before
                    + step as i32 * behind
                    + (horizontal / 2) as i32)
                    >> horizontal.ilog2();
            }
            before = behind;
        }
    }
    for x in 0..width {
        let mut before = top[x];
        for y in 0..side {
            let behind = rows[y * width + x];
            for step in 1..=vertical {
                out[(y * vertical + step - 1) * width + x] = T::from_i32(
                    ((vertical - step) as i32 * before
                        + step as i32 * behind
                        + (vertical / 2) as i32)
                        >> vertical.ilog2(),
                );
            }
            before = behind;
        }
    }
}

#[cfg(test)]
#[path = "mip_tests.rs"]
mod tests;
