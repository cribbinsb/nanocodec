//! H.266 rectangular planar/DC/angular prediction with wide angles and PDPC.
//! Prepared reference contract matches the square kernels. Normative equations
//! follow BSD-Clear `VVdeC` `IntraPrediction`; see `THIRD_PARTY.md`.
use super::{VvcIntraMode, angular::CUBIC};
const MAX_SIDE: usize = 64;
const ORIGIN: usize = MAX_SIDE;
const ANGLES: [i32; 32] = [
    0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 39, 45, 51, 57, 64, 73, 86,
    102, 128, 171, 256, 341, 512, 1024,
];
const INVERSE: [i32; 32] = [
    0, 16384, 8192, 5461, 4096, 2731, 2048, 1638, 1365, 1170, 1024, 910, 819, 712, 630, 565, 512,
    468, 420, 364, 321, 287, 256, 224, 191, 161, 128, 96, 64, 48, 32, 16,
];
const FILTER_THRESHOLD: [u8; 7] = [24, 24, 24, 14, 2, 0, 0];

/// Whether a luma rectangle uses three-tap filtered reference samples.
/// # Panics
/// If dimensions are not powers of two in 2..=64 or direction exceeds 66.
#[must_use]
pub fn vvc_intra_filter_references((width, height): (usize, usize), direction: u8) -> bool {
    assert!(
        [width, height]
            .iter()
            .all(|n| n.is_power_of_two() && (2..=64).contains(n))
            && direction <= 66
    );
    if direction < 2 {
        return direction == 0 && width * height > 32;
    }
    let mode = wide_angle(width, height, direction);
    let diff = (mode - 18).abs().min((mode - 50).abs());
    let angle = if mode >= 34 { mode - 50 } else { 18 - mode };
    diff > i32::from(FILTER_THRESHOLD[u32::midpoint(width.ilog2(), height.ilog2()) as usize])
        && ANGLES[angle.unsigned_abs() as usize] % 32 == 0
}

pub(crate) fn wide_angle(width: usize, height: usize, direction: u8) -> i32 {
    let shift = [0, 6, 10, 12, 14, 15][width.ilog2().abs_diff(height.ilog2()) as usize];
    let mode = i32::from(direction);
    if direction > 1 && width > height && mode < 2 + shift {
        mode + 65
    } else if direction > 1 && height > width && mode > 66 - shift {
        mode - 65
    } else {
        mode
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn predict(
    width: usize,
    height: usize,
    mode: VvcIntraMode,
    top: &[i32],
    left: &[i32],
    corner: i32,
    out: &mut [u8],
) {
    let direction = match mode {
        VvcIntraMode::Horizontal => Some((18, false)),
        VvcIntraMode::Vertical => Some((50, false)),
        VvcIntraMode::Angular { direction, luma } => Some((direction, luma)),
        _ => None,
    };
    if let Some((direction, luma)) = direction {
        angular(width, height, direction, luma, top, left, corner, 0, out);
        return;
    }
    if width >= 4 && height >= 4 {
        super::isp::smooth(
            (width, height),
            mode == VvcIntraMode::Planar,
            top,
            left,
            out,
        );
    } else {
        smooth(
            (width, height),
            mode == VvcIntraMode::Planar,
            top,
            left,
            out,
        );
    }
}

/// Two-sample chroma edges are below the ISP kernel's minimum dimensions.
fn smooth(
    (width, height): (usize, usize),
    planar: bool,
    top: &[i32],
    left: &[i32],
    out: &mut [u8],
) {
    let pdpc = width >= 4 && height >= 4;
    let scale = (width.ilog2() + height.ilog2() - 2) >> 2;
    let sum = match width.cmp(&height) {
        std::cmp::Ordering::Greater => top[..width].iter().sum::<i32>(),
        std::cmp::Ordering::Less => left[..height].iter().sum::<i32>(),
        std::cmp::Ordering::Equal => top[..width].iter().chain(&left[..height]).sum::<i32>(),
    };
    let divisor = if width == height {
        width + height
    } else {
        width.max(height)
    } as i32;
    let dc = (sum + divisor / 2) / divisor;
    for y in 0..height {
        for x in 0..width {
            let mut value = if planar {
                let horizontal = (width - 1 - x) as i32 * left[y] + (x + 1) as i32 * top[width];
                let vertical = (height - 1 - y) as i32 * top[x] + (y + 1) as i32 * left[height];
                ((horizontal << height.ilog2())
                    + (vertical << width.ilog2())
                    + (width * height) as i32)
                    >> (width.ilog2() + height.ilog2() + 1)
            } else {
                dc
            };
            if pdpc {
                let wl = 32 >> ((2 * x) >> scale).min(31);
                let wt = 32 >> ((2 * y) >> scale).min(31);
                value += (wl * (left[y] - value) + wt * (top[x] - value) + 32) >> 6;
            }
            out[y * width + x] = value.clamp(0, 255) as u8;
        }
    }
}

struct Projection {
    vertical: bool,
    angle: i32,
    inverse: i32,
    gaussian: bool,
    main: [i32; 5 * MAX_SIDE + 8],
    side: [i32; 2 * MAX_SIDE + 4],
}
impl Projection {
    #[allow(clippy::too_many_arguments)]
    fn new(
        width: usize,
        height: usize,
        direction: u8,
        luma: bool,
        top: &[i32],
        left: &[i32],
        corner: i32,
        reference_line: u8,
        prediction_size: Option<(usize, usize)>,
    ) -> Self {
        let (pred_width, pred_height) = prediction_size.unwrap_or((width, height));
        let mode = wide_angle(pred_width, pred_height, direction);
        let vertical = mode >= 34;
        let angle_mode = if vertical { mode - 50 } else { 18 - mode };
        let magnitude = angle_mode.unsigned_abs() as usize;
        let angle = ANGLES[magnitude] * angle_mode.signum();
        let inverse = INVERSE[magnitude];
        let (main_edge, side_edge, side_size) = if vertical {
            (top, left, height)
        } else {
            (left, top, width)
        };
        let diff = (mode - 18).abs().min((mode - 50).abs());
        let gaussian = luma
            && prediction_size.is_none()
            && reference_line == 0
            && diff
                > i32::from(
                    FILTER_THRESHOLD[u32::midpoint(width.ilog2(), height.ilog2()) as usize],
                );
        let mut main = [main_edge[main_edge.len() - 1]; 5 * MAX_SIDE + 8];
        let mut side = [side_edge[side_edge.len() - 1]; 2 * MAX_SIDE + 4];
        main[ORIGIN] = corner;
        side[0] = corner;
        main[ORIGIN + 1..ORIGIN + 1 + main_edge.len()].copy_from_slice(main_edge);
        side[1..=side_edge.len()].copy_from_slice(side_edge);
        if angle < 0 {
            for d in 1..=side_size {
                main[ORIGIN - d] =
                    side[(((d as i32 * inverse + 256) >> 9) as usize).min(side_size)];
            }
        }
        Self {
            vertical,
            angle,
            inverse,
            gaussian,
            main,
            side,
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn angular(
    width: usize,
    height: usize,
    direction: u8,
    luma: bool,
    top: &[i32],
    left: &[i32],
    corner: i32,
    reference_line: u8,
    out: &mut [u8],
) {
    let p = Projection::new(
        width,
        height,
        direction,
        luma,
        top,
        left,
        corner,
        reference_line,
        None,
    );
    render((width, height), luma, corner, reference_line, &p, out);
}

fn render(
    (width, height): (usize, usize),
    luma: bool,
    corner: i32,
    reference_line: u8,
    p: &Projection,
    out: &mut [u8],
) {
    let (columns, rows) = if p.vertical {
        (width, height)
    } else {
        (height, width)
    };
    let enabled = reference_line == 0 && width >= 4 && height >= 4;
    let pdpc = if enabled && p.angle > 0 {
        let scale = (rows.ilog2() as i32 - ((3 * p.inverse - 2).ilog2() as i32 - 8)).min(2);
        (scale >= 0).then_some(scale as u32)
    } else {
        None
    };
    let axis_scale = (width.ilog2() + height.ilog2() - 2) >> 2;
    for y in 0..rows {
        let delta = (y as i32 + 1 + i32::from(reference_line)) * p.angle;
        let offset = delta >> 5;
        let phase = delta & 31;
        let filter = if p.gaussian {
            let half = phase >> 1;
            [16 - half, 32 - half, 16 + half, half]
        } else {
            CUBIC[phase as usize]
        };
        for x in 0..columns {
            let index = (ORIGIN as i32 + i32::from(reference_line) + offset) as usize + x;
            let mut value = if p.angle % 32 == 0 {
                p.main[index + 1]
            } else if luma {
                ((0..4).map(|i| filter[i] * p.main[index + i]).sum::<i32>() + 32) >> 6
            } else {
                ((32 - phase) * p.main[index + 1] + phase * p.main[index + 2] + 16) >> 5
            };
            value = value.clamp(0, 255);
            if enabled && p.angle == 0 {
                let weight = 32 >> ((2 * x) >> axis_scale).min(31);
                value += (weight * (p.side[y + 1] - corner) + 32) >> 6;
            } else if let Some(scale) = pdpc
                && x < 3 << scale
            {
                let side_index = y + 1 + ((256 + (x as i32 + 1) * p.inverse) >> 9) as usize;
                let weight = 32 >> ((2 * x) >> scale);
                value += (weight * (p.side[side_index] - value) + 32) >> 6;
            }
            let index = if p.vertical {
                y * width + x
            } else {
                x * width + y
            };
            out[index] = value.clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dsp, Isa};
    #[test]
    #[ignore = "requires independent VVdeC D_PRED trace (NC_VVC_PRED_TRACE)"]
    fn official_prepared_predictions() {
        let path = std::env::var("NC_VVC_PRED_TRACE").expect("set NC_VVC_PRED_TRACE");
        let trace = std::fs::read_to_string(path).unwrap();
        let mut edge = Vec::new();
        let mut count = 0;
        for line in trace.lines() {
            if let Some(values) = line.strip_prefix("NC_EDGE ") {
                edge = values
                    .split_whitespace()
                    .map(|v| v.parse::<i32>().unwrap())
                    .collect();
            } else if let Some(values) = line.strip_prefix("NC_PRED ") {
                let values: Vec<i32> = values
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
                assert_eq!(&edge[..6], &values[..6]);
                let (width, height) = (edge[3] as usize, edge[4] as usize);
                let direction = edge[5] as u8;
                let mode = match direction {
                    0 => VvcIntraMode::Planar,
                    1 => VvcIntraMode::Dc,
                    _ => VvcIntraMode::Angular {
                        direction,
                        luma: edge[0] == 0,
                    },
                };
                let mut actual = vec![0; width * height];
                predict(
                    width,
                    height,
                    mode,
                    &edge[7..7 + 2 * width],
                    &edge[7 + 2 * width..],
                    edge[6],
                    &mut actual,
                );
                let expected: Vec<u8> = values[6..].iter().map(|&v| v as u8).collect();
                assert_eq!(actual, expected, "prediction {count}: {:?}", &edge[..6]);
                count += 1;
            }
        }
        assert!(count > 2000, "incomplete reference trace: {count}");
        eprintln!("{count} independent prediction blocks matched");
    }
    #[test]
    fn square_limit_matches_existing_scalar_predictors() {
        let dsp = Dsp::with_isa(Isa::Scalar).unwrap();
        for side in [4, 8, 16, 32] {
            for luma in [false, true] {
                for direction in 0..67 {
                    let mode = match direction {
                        0 => VvcIntraMode::Planar,
                        1 => VvcIntraMode::Dc,
                        _ => VvcIntraMode::Angular { direction, luma },
                    };
                    let top: Vec<_> = (0..2 * side)
                        .map(|x| ((x * 47 + 17) % 256) as i32)
                        .collect();
                    let left: Vec<_> = (0..2 * side)
                        .map(|y| ((y * 91 + 39) % 256) as i32)
                        .collect();
                    let mut expected = vec![0; side * side];
                    let mut actual = expected.clone();
                    dsp.vvc_intra_predict(side, mode, &top, &left, 73, &mut expected);
                    // Do not delegate square cases here: verify the new rectangular
                    // equations independently at their square limit.
                    predict(side, side, mode, &top, &left, 73, &mut actual);
                    assert_eq!(actual, expected, "side={side} mode={mode:?}");
                }
            }
        }
    }
    #[test]
    fn rectangular_modes_preserve_constant_references_and_output_guards() {
        for width in [2, 4, 8, 16, 32] {
            for height in [2, 4, 8, 16, 32] {
                for luma in [false, true] {
                    for direction in 0..67 {
                        let mode = match direction {
                            0 => VvcIntraMode::Planar,
                            1 => VvcIntraMode::Dc,
                            _ => VvcIntraMode::Angular { direction, luma },
                        };
                        let mut out = vec![19; width * height + 4];
                        predict(
                            width,
                            height,
                            mode,
                            &vec![173; 2 * width],
                            &vec![173; 2 * height],
                            173,
                            &mut out[..width * height],
                        );
                        assert!(
                            out[..width * height].iter().all(|&v| v == 173),
                            "{width}x{height} {mode:?}"
                        );
                        assert_eq!(&out[width * height..], &[19; 4]);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "rectangular_fixtures.rs"]
mod fixtures;

impl crate::Dsp {
    /// Luma angular/DC prediction from reference line one or two.
    /// Edges start immediately after the displaced top-left corner and contain
    /// `2 * side + reference_line` substituted, unfiltered samples.
    /// # Panics
    /// If dimensions, mode, line or buffer lengths are invalid.
    pub fn vvc_mrl_predict(
        &self,
        (width, height): (usize, usize),
        direction: u8,
        reference_line: u8,
        top: &[i32],
        left: &[i32],
        corner: i32,
        out: &mut [u8],
    ) {
        let valid = |n: usize| n.is_power_of_two() && (4..=64).contains(&n);
        let line = usize::from(reference_line);
        assert!(
            valid(width)
                && valid(height)
                && (1..=66).contains(&direction)
                && matches!(reference_line, 1 | 2)
        );
        assert!(
            top.len() >= 2 * width + line
                && left.len() >= 2 * height + line
                && out.len() >= width * height
        );
        if direction == 1 {
            let top = &top[line..line + width];
            let left = &left[line..line + height];
            let sum = match width.cmp(&height) {
                std::cmp::Ordering::Greater => top.iter().sum::<i32>(),
                std::cmp::Ordering::Less => left.iter().sum::<i32>(),
                std::cmp::Ordering::Equal => top.iter().chain(left).sum::<i32>(),
            };
            let count = if width == height {
                width + height
            } else {
                width.max(height)
            } as i32;
            out[..width * height].fill(((sum + count / 2) / count).clamp(0, 255) as u8);
        } else {
            angular(
                width,
                height,
                direction,
                true,
                &top[..2 * width + line],
                &left[..2 * height + line],
                corner,
                reference_line,
                out,
            );
        }
    }
}

impl crate::Dsp {
    /// Predict an ISP region with CU-based wide angles and unfiltered references.
    /// Vertical one/two-wide transform blocks share a four-wide prediction region.
    /// # Panics
    /// If dimensions, direction or reference/output lengths are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn vvc_intra_predict_isp(
        &self,
        prediction_size: (usize, usize),
        size: (usize, usize),
        direction: u8,
        top: &[i32],
        left: &[i32],
        corner: i32,
        out: &mut [u8],
    ) {
        let (width, height) = size;
        let valid = |n: usize| n.is_power_of_two() && n <= 64;
        assert!(valid(width) && valid(height) && width >= 4 && width * height >= 16);
        assert!(valid(prediction_size.0) && valid(prediction_size.1) && direction <= 66);
        assert!(top.len() > width && left.len() > height && out.len() >= width * height);
        if direction <= 1 {
            if height >= 4 {
                super::isp::smooth(size, direction == 0, top, left, out);
            } else {
                smooth(size, direction == 0, top, left, out);
            }
        } else {
            let projection = Projection::new(
                width,
                height,
                direction,
                true,
                top,
                left,
                corner,
                0,
                Some(prediction_size),
            );
            render(size, true, corner, 0, &projection, out);
        }
    }
}
