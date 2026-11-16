//! Decoder-side motion refinement of the general AV2 decoder: the
//! mirrored SAD search of `refinemv` ([`sad_refine_mv`]), the optical-flow
//! regression sums ([`opfl_refine_mv`], [`OpflRegressionData`]) and the
//! 8x8 SAD that decides whether a block refines ([`sad8x8`]).
//!
//! Ported from dav2d `src/mc_tmpl.c` (`sad_refine_mv_c`,
//! `opfl_refine_mv_c`, `sad8x8_c`, `struct OpflOffset`,
//! `struct OpflRegressionData`; commit 237229e, Copyright © 2018-2026
//! `VideoLAN` and dav2d authors, BSD-2-Clause, see `docs/licenses/dav2d`).
//! Deeper content is scaled to 8 bits before every sum, so the thresholds
//! and the regression's range are depth-independent.

use nc_common::{BitDepth, Sample};

/// The search range of [`sad_refine_mv`]: offsets -2 to 2 in each
/// direction.
const SEARCH_RANGE: i32 = 2;
/// The SAD window's border on each side of the block: the search range.
const SEARCH_BORDER: usize = SEARCH_RANGE as usize;
/// The zero offset's SAD is scaled by 7/8 (a bias toward no refinement).
const ZERO_BIAS_NUMERATOR: u32 = 7;
/// Bits of the zero offset's scale.
const ZERO_BIAS_BITS: u32 = 3;
/// A SAD below this many times the window area (every other row) ends the
/// search at the zero offset.
const EARLY_EXIT_PER_SAMPLE: u32 = 2;
/// Side of the [`sad8x8`] block.
const SAD_SIDE: usize = 8;
/// Largest optical-flow block width.
const OPFL_MAX_WIDTH: usize = 64;
/// Optical-flow block heights.
const OPFL_HEIGHTS: [usize; 2] = [8, 16];
/// Largest optical-flow block area.
const OPFL_MAX_AREA: usize = OPFL_MAX_WIDTH * 16;
/// Gradients are taken within columns of 16 samples.
const GRADIENT_SPAN: usize = 16;
/// Gradient filter taps on the first and second neighbours.
const GRADIENT_NEAR: i32 = 42;
const GRADIENT_FAR: i32 = -5;
/// Fraction bits of the gradient filter.
const GRADIENT_BITS: u32 = 7;

/// A refinement offset in whole samples (dav2d `struct OpflOffset`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpflOffset {
    /// Vertical offset, -2 to 2.
    pub y: i8,
    /// Horizontal offset, -2 to 2.
    pub x: i8,
}

/// The least-squares sums of one optical-flow sub-block (dav2d `struct
/// OpflRegressionData`): `u`, `v` the horizontal and vertical gradients of
/// the distance-weighted sum, `w` the difference of the two predictions;
/// `su2` and `sv2` start at the sub-block's area (regularisation).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpflRegressionData {
    /// `sum(u * u)` plus the area.
    pub su2: i32,
    /// `sum(u * v)`.
    pub suv: i32,
    /// `sum(v * v)` plus the area.
    pub sv2: i32,
    /// `sum(u * w)`.
    pub suw: i32,
    /// `sum(v * w)`.
    pub svw: i32,
}

/// A view of a block's samples: slice, offset of the top-left sample,
/// stride.
type Samples<'a, T> = (&'a [T], usize, usize);

/// The SAD of every other row of a `w` x `h` window, scaled to 8 bits
/// (dav2d `sad_nxn`).
fn sad_alternate_rows<T: Sample>(
    (p0, o0, s0): Samples<'_, T>,
    (p1, o1, s1): Samples<'_, T>,
    (w, h): (usize, usize),
    depth: BitDepth,
) -> u32 {
    let sad: u32 = (0..h)
        .step_by(2)
        .map(|y| {
            let (a, b) = (&p0[o0 + y * s0..][..w], &p1[o1 + y * s1..][..w]);
            a.iter()
                .zip(b)
                .map(|(a, b)| a.to_i32().abs_diff(b.to_i32()))
                .sum::<u32>()
        })
        .sum();
    sad >> depth.above_eight()
}

/// `view` moved to offset (`dx`, `dy`) from the search window's centre.
fn shifted<T>((data, offset, stride): Samples<'_, T>, dx: i32, dy: i32) -> Samples<'_, T> {
    let x = (SEARCH_RANGE + dx) as usize;
    let y = (SEARCH_RANGE + dy) as usize;
    (data, offset + y * stride + x, stride)
}

/// The whole-sample offset that best matches the two predictions `p0`,
/// `p1` of a `w` x `h` block (8 or 16 each), moving them in opposite
/// directions (dav2d `sad_refine_mv_c`). Each view starts 2 samples above
/// and left of the block and covers `w + 8` x `h + 8` samples. With
/// `is_implicit` the zero offset competes, scaled by 7/8, and wins at
/// once if small.
///
/// # Panics
///
/// If a size is out of range or a slice is too short.
pub fn sad_refine_mv<T: Sample>(
    p0: Samples<'_, T>,
    p1: Samples<'_, T>,
    (w, h): (usize, usize),
    is_implicit: bool,
    depth: BitDepth,
) -> OpflOffset {
    assert!(matches!(w, 8 | 16) && matches!(h, 8 | 16));
    let window = (w + 2 * SEARCH_BORDER, h + 2 * SEARCH_BORDER);
    let mut best = OpflOffset::default();
    let mut best_sad = u32::MAX;
    if is_implicit {
        let sad = sad_alternate_rows(shifted(p0, 0, 0), shifted(p1, 0, 0), window, depth);
        best_sad = (sad * ZERO_BIAS_NUMERATOR + ZERO_BIAS_NUMERATOR) >> ZERO_BIAS_BITS;
        let threshold = (window.0 * window.1) as u32 * EARLY_EXIT_PER_SAMPLE;
        if best_sad < threshold {
            return best;
        }
    }
    for dy in -SEARCH_RANGE..=SEARCH_RANGE {
        for dx in -SEARCH_RANGE..=SEARCH_RANGE {
            if dx == 0 && dy == 0 {
                continue;
            }
            let sad = sad_alternate_rows(shifted(p0, dx, dy), shifted(p1, -dx, -dy), window, depth);
            if sad < best_sad {
                best_sad = sad;
                best = OpflOffset {
                    y: dy as i8,
                    x: dx as i8,
                };
            }
        }
    }
    best
}

/// `value` divided by `2^shift`, rounded to nearest with ties toward zero
/// (dav2d's `(v + rnd - (v < 0)) >> bd_min8`; 8-bit values stay as they
/// are).
fn scale_to_eight(value: i32, shift: u32) -> i32 {
    if shift == 0 {
        return value;
    }
    let rounding = 1 << (shift - 1);
    (value + rounding - i32::from(value < 0)) >> shift
}

/// The 4-tap gradient at `i` of `line(k)` within `[min, max]` (edges
/// repeat; the 2-tap fallback at an edge doubles).
fn gradient(line: impl Fn(usize) -> i32, i: usize, (min, max): (usize, usize)) -> i16 {
    let p0 = line(i.saturating_sub(2).max(min));
    let p1 = line(i.saturating_sub(1).max(min));
    let p2 = line((i + 1).min(max));
    let p3 = line((i + 2).min(max));
    let at_edge = i + 1 > max || i < min + 1;
    let g = ((p2 - p1) * GRADIENT_NEAR + (p3 - p0) * GRADIENT_FAR) * (1 + i32::from(at_edge));
    ((g + (1 << (GRADIENT_BITS - 1)) - 1 + i32::from(g > 0)) >> GRADIENT_BITS) as i16
}

/// The regression sums of the optical-flow sub-blocks (`bs` x `bs`, 4 or
/// 8; 4 only for an 8x8 block) of a `w` x `h` block (`w` a power of two,
/// 8 to 64; `h` 8 or 16) from its two predictions, in raster order into
/// `out` (dav2d `opfl_refine_mv_c`). `distance` holds the weights of the
/// two predictions in the gradient's source (dav2d `union aliasi16 d`).
///
/// # Panics
///
/// If a size is out of range or a slice is too short.
pub fn opfl_refine_mv<T: Sample>(
    out: &mut [OpflRegressionData],
    p0: Samples<'_, T>,
    p1: Samples<'_, T>,
    (w, h): (usize, usize),
    bs: usize,
    distance: [i8; 2],
    depth: BitDepth,
) {
    assert!(matches!(bs, 4 | 8) && (bs == 8 || (w == 8 && h == 8)));
    assert!(w.is_power_of_two() && (8..=OPFL_MAX_WIDTH).contains(&w) && OPFL_HEIGHTS.contains(&h));
    let shift = depth.above_eight();
    let (d0, d1) = (i32::from(distance[0]), i32::from(distance[1]));
    // The distance-weighted sum and the plain difference, at 8 bits.
    let mut weighted = [0_i16; OPFL_MAX_AREA];
    let mut difference = [0_i16; OPFL_MAX_AREA];
    for y in 0..h {
        let a = &p0.0[p0.1 + y * p0.2..][..w];
        let b = &p1.0[p1.1 + y * p1.2..][..w];
        for x in 0..w {
            let (a, b) = (a[x].to_i32(), b[x].to_i32());
            weighted[y * w + x] = scale_to_eight(d0 * a + d1 * b, shift) as i16;
            difference[y * w + x] = scale_to_eight(a - b, shift) as i16;
        }
    }
    let mut gx = [0_i16; OPFL_MAX_AREA];
    let mut gy = [0_i16; OPFL_MAX_AREA];
    for y in 0..h {
        for x in 0..w {
            let span = (
                x / GRADIENT_SPAN * GRADIENT_SPAN,
                (x | (GRADIENT_SPAN - 1)).min(w - 1),
            );
            gx[y * w + x] = gradient(|i| i32::from(weighted[y * w + i]), x, span);
            gy[y * w + x] = gradient(|i| i32::from(weighted[i * w + x]), y, (0, h - 1));
        }
    }
    let area = (bs * bs) as i32;
    let blocks = (0..h)
        .step_by(bs)
        .flat_map(|y| (0..w).step_by(bs).map(move |x| (x, y)));
    for (slot, (x0, y0)) in out.iter_mut().zip(blocks) {
        let mut sums = OpflRegressionData {
            su2: area,
            sv2: area,
            ..OpflRegressionData::default()
        };
        for y in y0..y0 + bs {
            for x in x0..x0 + bs {
                let at = y * w + x;
                let grad_x = i32::from(gx[at]);
                let grad_y = i32::from(gy[at]);
                let diff = i32::from(difference[at]);
                sums.su2 += grad_x * grad_x;
                sums.suv += grad_x * grad_y;
                sums.sv2 += grad_y * grad_y;
                sums.suw += grad_x * diff;
                sums.svw += grad_y * diff;
            }
        }
        *slot = sums;
    }
}

/// The SAD of two 8x8 blocks, scaled to 8 bits (dav2d `sad8x8_c`).
///
/// # Panics
///
/// If a slice is too short.
pub fn sad8x8<T: Sample>(p0: Samples<'_, T>, p1: Samples<'_, T>, depth: BitDepth) -> u32 {
    let sad: u32 = (0..SAD_SIDE)
        .map(|y| {
            let a = &p0.0[p0.1 + y * p0.2..][..SAD_SIDE];
            let b = &p1.0[p1.1 + y * p1.2..][..SAD_SIDE];
            a.iter()
                .zip(b)
                .map(|(a, b)| a.to_i32().abs_diff(b.to_i32()))
                .sum::<u32>()
        })
        .sum();
    sad >> depth.above_eight()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two 24x24 test predictions: `p1` is `p0` moved by (2, -1).
    fn predictions() -> (Vec<u8>, Vec<u8>) {
        let p0: Vec<u8> = (0..24 * 24_usize)
            .map(|i| {
                let (x, y) = (i % 24, i / 24);
                ((x * x * 3 + y * 11 + x * y) % 256) as u8
            })
            .collect();
        let p1 = (0..24 * 24_usize)
            .map(|i| {
                let (x, y) = ((i % 24 + 2).min(23), (i / 24).saturating_sub(1));
                p0[y * 24 + x]
            })
            .collect();
        (p0, p1)
    }

    /// Values captured from dav2d's `sad_refine_mv_c` and
    /// `opfl_refine_mv_c` (8-bit).
    #[test]
    fn matches_dav2d() {
        let (p0, p1) = predictions();
        let eight = BitDepth::EIGHT;
        let offset = sad_refine_mv((&p0, 0, 24), (&p1, 0, 24), (16, 16), false, eight);
        assert_eq!(offset, OpflOffset { y: -1, x: 1 });
        let mut sums = [OpflRegressionData::default(); 4];
        opfl_refine_mv(
            &mut sums,
            (&p0, 0, 24),
            (&p1, 0, 24),
            (8, 8),
            4,
            [3, -3],
            eight,
        );
        let flat: Vec<i32> = sums
            .iter()
            .flat_map(|s| [s.su2, s.suv, s.sv2, s.suw, s.svw])
            .collect();
        assert_eq!(
            flat,
            [
                4922, -1726, 2513, 6204, -2844, 457_136, 65820, 108_673, 138_367, 12091, 4780, 828,
                160, 7292, 1272, 741_364, 328_194, 348_280, 118_550, 25146
            ]
        );
        assert_eq!(sad8x8((&p0, 0, 24), (&p0, 0, 24), eight), 0);
    }
}
