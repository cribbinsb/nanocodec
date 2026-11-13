//! H.266 matrix intra prediction, using normative weights and separable upsampling.
//! Algorithms checked against BSD-Clear `VVdeC` `MatrixIntraPrediction`.
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
        (width, height): (usize, usize),
        mode: u8,
        transpose: bool,
        top: &[i32],
        left: &[i32],
        out: &mut [u8],
    ) {
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
                .all(|&v| (0..=255).contains(&v))
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
        input[0] = if class < 2 { 128 - origin } else { 0 };
        let offset = 32 - 32 * input[..2 * edge_size].iter().sum::<i32>();
        let mut reduced = [0u8; 64];
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
            reduced[destination] = (((dot + offset) >> 6) + origin).clamp(0, 255) as u8;
        }
        upsample((width, height), side, &reduced, top, left, out);
    }
    /// Predict block differential PCM from the selected reference edge, without PDPC.
    /// # Panics
    /// If dimensions, direction, edges or output length are invalid.
    pub fn vvc_bdpcm_predict(
        &self,
        (width, height): (usize, usize),
        vertical: bool,
        top: &[i32],
        left: &[i32],
        out: &mut [u8],
    ) {
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
                *destination = if vertical { top[x] } else { left[y] }.clamp(0, 255) as u8;
            }
        }
    }
}
fn downsample(source: &[i32], out: &mut [i32]) {
    let step = source.len() / out.len();
    for (dest, group) in out.iter_mut().zip(source.chunks_exact(step)) {
        *dest = (group.iter().sum::<i32>() + (step / 2) as i32) >> step.ilog2();
    }
}
fn upsample(
    (width, height): (usize, usize),
    side: usize,
    reduced: &[u8],
    top: &[i32],
    left: &[i32],
    out: &mut [u8],
) {
    let (horizontal, vertical) = (width / side, height / side);
    let mut rows = [0i32; 64 * 8];
    for y in 0..side {
        let mut before = left[(y + 1) * vertical - 1];
        for x in 0..side {
            let behind = i32::from(reduced[y * side + x]);
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
                out[(y * vertical + step - 1) * width + x] = (((vertical - step) as i32 * before
                    + step as i32 * behind
                    + (vertical / 2) as i32)
                    >> vertical.ilog2())
                    as u8;
            }
            before = behind;
        }
    }
}

#[cfg(test)]
#[path = "mip_tests.rs"]
mod tests;
