//! The player model: how a decoded picture coded smaller than its stream
//! (an AV1 frame whose `render_size` exceeds its frame size, adaptive
//! coded resolution) is scaled to the render size for display.
//!
//! Players scale decoded video on the GPU, sampling the texture bilinearly
//! with pixel centres aligned (`in = (out + 0.5) * in_size / out_size -
//! 0.5`, clamped at the edges); each plane of a 4:2:0 picture is scaled on
//! its own grid. This is that filter in exact integer arithmetic, so the
//! encoder's reconstruction output (`nanocodec encode --recon`) and the
//! evaluation harness (`eval/nceval/display.py`, its bit-exact twin) score
//! what a viewer sees. Not normative and never in a decoding path: scalar
//! is enough.

use nc_common::{PlaneView, PlaneViewMut};

/// Fractional bits of a sample position.
const POSITION_BITS: u32 = 16;
/// Bits of the interpolation weight kept from a position's fraction.
const WEIGHT_BITS: u32 = 8;
const WEIGHT_ONE: u32 = 1 << WEIGHT_BITS;

/// The two input samples and the weight of the second, for one output
/// sample of a direction scaled from `input` to `output` samples.
#[derive(Debug, Clone, Copy)]
struct Tap {
    first: usize,
    second: usize,
    weight: u32,
}

/// The taps of every output sample of a direction.
fn taps(input: usize, output: usize) -> Vec<Tap> {
    let half = 1i64 << (POSITION_BITS - 1);
    (0..output)
        .map(|out| {
            // ((out + 0.5) * input / output - 0.5) in 1/65536 samples.
            let centre = ((2 * out as i64 + 1) * input as i64) << POSITION_BITS;
            let position = (centre / (2 * output as i64) - half).max(0);
            let first = ((position >> POSITION_BITS) as usize).min(input - 1);
            let fraction = (position >> (POSITION_BITS - WEIGHT_BITS)) as u32 & (WEIGHT_ONE - 1);
            Tap {
                first,
                second: (first + 1).min(input - 1),
                weight: fraction,
            }
        })
        .collect()
}

/// Scale all of `src` into all of `dst` as the player model does.
pub fn display_scale_plane(src: &PlaneView<'_>, dst: &mut PlaneViewMut<'_>) {
    let (in_w, in_h) = (src.width(), src.height());
    let (out_w, out_h) = (dst.width(), dst.height());
    let columns = taps(in_w, out_w);
    // Horizontal pass, unrounded (16-bit products) for exactness.
    let rows: Vec<Vec<u32>> = (0..in_h)
        .map(|y| {
            let row = src.row(y);
            columns
                .iter()
                .map(|t| {
                    u32::from(row[t.first]) * (WEIGHT_ONE - t.weight)
                        + u32::from(row[t.second]) * t.weight
                })
                .collect()
        })
        .collect();
    let round = 1u32 << (2 * WEIGHT_BITS - 1);
    for (y, t) in taps(in_h, out_h).iter().enumerate() {
        let (top, bottom) = (&rows[t.first], &rows[t.second]);
        for ((out, &a), &b) in dst.row_mut(y)[..out_w].iter_mut().zip(top).zip(bottom) {
            *out =
                ((a * (WEIGHT_ONE - t.weight) + b * t.weight + round) >> (2 * WEIGHT_BITS)) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::Plane;

    fn plane(width: usize, height: usize, value: impl Fn(usize, usize) -> u8) -> Plane {
        let mut p = Plane::new(width, height, 0).unwrap();
        let mut view = p.view_mut();
        for y in 0..height {
            for (x, s) in view.row_mut(y).iter_mut().enumerate() {
                *s = value(x, y);
            }
        }
        p
    }

    fn scaled(src: &Plane, width: usize, height: usize) -> Vec<Vec<u8>> {
        let mut dst = Plane::new(width, height, 0).unwrap();
        display_scale_plane(&src.view(), &mut dst.view_mut());
        dst.view().rows().map(<[u8]>::to_vec).collect()
    }

    /// The fixture `eval/tests/test_display.py` checks the Python twin
    /// against: a 3x2 plane doubled.
    #[test]
    fn doubling_matches_the_shared_fixture() {
        let src = plane(3, 2, |x, y| [[10, 50, 200], [90, 0, 255]][y][x]);
        assert_eq!(
            scaled(&src, 6, 4),
            [
                [10, 20, 40, 88, 163, 200],
                [30, 32, 36, 82, 170, 214],
                [70, 56, 27, 70, 184, 241],
                [90, 68, 23, 64, 191, 255],
            ]
        );
    }

    #[test]
    fn same_size_copies_and_flat_stays_flat() {
        let src = plane(7, 5, |x, y| (x * 31 + y * 17) as u8);
        let copy = scaled(&src, 7, 5);
        assert!(src.view().rows().zip(&copy).all(|(a, b)| a == b.as_slice()));
        let flat = plane(5, 3, |_, _| 123);
        assert!(scaled(&flat, 11, 9).iter().flatten().all(|&s| s == 123));
    }
}
