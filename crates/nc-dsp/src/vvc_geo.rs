//! H.266 geometric-partition masks, shared by sample blending and motion masks.
//! Equations/tables from BSD-Clear `VVdeC` `Rom` / `UnitTools`; see `THIRD_PARTY.md`.
use crate::Dsp;
const ANGLE_MASK: [i8; 32] = [
    0, -1, 1, 2, 3, 4, -1, -1, 5, -1, -1, 4, 3, 2, 1, -1, 0, -1, 1, 2, 3, 4, -1, -1, 5, -1, -1, 4,
    3, 2, 1, -1,
];
const DISTANCE: [i32; 32] = [
    8, 8, 8, 8, 4, 4, 2, 1, 0, -1, -2, -4, -4, -8, -8, -8, -8, -8, -8, -8, -4, -4, -2, -1, 0, 1, 2,
    4, 4, 8, 8, 8,
];
/// A VVC geometric partition's line in luma-sample coordinates.
#[derive(Debug, Clone, Copy)]
pub struct VvcGeometric {
    direction: [i32; 2],
    offset: [i32; 2],
    flip: bool,
}
impl VvcGeometric {
    /// Construct the mask of an 8..64-sample power-of-two CU.
    /// # Panics
    /// If either dimension or the partition index is invalid.
    #[must_use]
    pub fn new(size: (usize, usize), split: u8) -> Self {
        assert!(
            [size.0, size.1]
                .iter()
                .all(|v| (8..=64).contains(v) && v.is_power_of_two())
        );
        assert!(split < 64);
        let (angle, distance) = parameters(split);
        let (w, h) = (size.0 as i32, size.1 as i32);
        let mut offset = [-w / 2, -h / 2];
        let axis = usize::from(angle % 16 == 8 || (angle % 16 != 0 && h >= w));
        offset[axis] += (if angle < 16 { 1 } else { -1 }) * ((distance * [w, h][axis]) >> 3);
        Self {
            direction: [DISTANCE[angle], DISTANCE[(angle + 8) % 32]],
            offset,
            flip: (13..=27).contains(&angle),
        }
    }
    fn distance(self, x: usize, y: usize) -> i32 {
        (2 * (x as i32 + self.offset[0]) + 1) * self.direction[0]
            + (2 * (y as i32 + self.offset[1]) + 1) * self.direction[1]
    }
    /// First prediction's weight, in eighths, at a luma-sample position.
    #[must_use]
    pub fn weight(self, x: usize, y: usize) -> u8 {
        let distance = self.distance(x, y) * if self.flip { -1 } else { 1 };
        ((36 + distance) >> 3).clamp(0, 8) as u8
    }
    /// Which prediction supplies a 4x4 motion cell: 0, 1, or 2 (both).
    #[must_use]
    pub fn motion_partition(self, x: usize, y: usize) -> usize {
        let distance = self.distance(x + 2, y + 2);
        if distance.abs() < 32 {
            2
        } else if distance <= 0 {
            usize::from(!self.flip)
        } else {
            usize::from(self.flip)
        }
    }
}
fn parameters(split: u8) -> (usize, i32) {
    let mut index = 0;
    for (angle, &mask) in ANGLE_MASK.iter().enumerate() {
        for distance in 0..4 {
            if mask < 0
                || (distance == 0 && angle >= 16)
                || ((distance == 0 || distance == 2) && (mask == 0 || mask == 5))
            {
                continue;
            }
            if index == split {
                return (angle, distance);
            }
            index += 1;
        }
    }
    unreachable!("validated partition index")
}
/// Weights are in eighths: `w` for the first prediction, `8 − w` for the second.
const WEIGHT_DENOMINATOR: i32 = 8;
/// Fewest bits `shift1` drops (H.266 §8.5.7.3: `shift1 = Max(5, 17 − BitDepth)`).
const MIN_BLEND_SHIFT: u32 = 5;
/// The six-fractional-bit precision plus the eighths of the weights, from
/// which `shift1 = 17 − BitDepth` removes all but the sample's bits.
const BLEND_PRECISION: u32 = 17;

impl Dsp {
    /// Blend signed six-fractional-bit predictions using a geometric mask.
    /// `origin` and mask geometry are in luma samples; `chroma` selects 4:2:0
    /// ([`crate::DepthKernels::vvc_geometric_blend`] for deeper samples).
    /// # Panics
    /// If a slice cannot hold the requested rectangle.
    pub fn vvc_geometric_blend(
        &self,
        mask: VvcGeometric,
        origin: (usize, usize),
        size: (usize, usize),
        chroma: bool,
        sources: (&[i16], &[i16]),
        out: &mut [u8],
    ) {
        blend_at(
            mask,
            origin,
            size,
            chroma,
            sources,
            out,
            nc_common::BitDepth::EIGHT,
        );
    }
}

/// [`Dsp::vvc_geometric_blend`] at `depth` (H.266 §8.5.7.3):
/// `Clip3(0, max, (a·w + b·(8 − w) + offset1) >> shift1)`,
/// `shift1 = Max(5, 17 − BitDepth)`, `offset1 = 1 << (shift1 − 1)`.
///
/// # Panics
/// If a slice cannot hold the requested rectangle.
pub(crate) fn blend_at<T: nc_common::Sample>(
    mask: VvcGeometric,
    origin: (usize, usize),
    size: (usize, usize),
    chroma: bool,
    sources: (&[i16], &[i16]),
    out: &mut [T],
    depth: nc_common::BitDepth,
) {
    let shift = BLEND_PRECISION
        .saturating_sub(depth.bits())
        .max(MIN_BLEND_SHIFT);
    let offset = 1 << (shift - 1);
    let count = size.0 * size.1;
    let (a, b) = (&sources.0[..count], &sources.1[..count]);
    for (i, (out, (&a, &b))) in out[..count].iter_mut().zip(a.iter().zip(b)).enumerate() {
        let x = origin.0 + ((i % size.0) << usize::from(chroma));
        let y = origin.1 + ((i / size.0) << usize::from(chroma));
        let weight = i32::from(mask.weight(x, y));
        let sum = weight * i32::from(a) + (WEIGHT_DENOMINATOR - weight) * i32::from(b);
        *out = depth.clip_to((sum + offset) >> shift);
    }
}

#[cfg(test)]
#[path = "vvc_geo_tests.rs"]
mod tests;
