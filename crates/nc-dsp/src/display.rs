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
//! what a viewer sees. Not normative and never in a decoding path.
//!
//! The browser demo also scales each captured camera or video frame to a
//! view's size with it (`nc-web`, every frame): [`DisplayScaler`] keeps the
//! taps and its intermediate row between frames, so a frame allocates
//! nothing.

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
    DisplayScaler::default().scale(src, dst);
}

/// [`display_scale_plane`] with the taps of the last sizes and an
/// intermediate row kept for the next call (exactly the same output).
///
/// Both passes are exact integer sums and only the final value is rounded,
/// so the vertical pass may run first: it is a contiguous blend of two input
/// rows (which vectorises), leaving the gathers of the horizontal pass to
/// the output size only.
#[derive(Debug, Clone, Default)]
pub struct DisplayScaler {
    /// Input and output size the taps were made for.
    sizes: [usize; 4],
    columns: Vec<Tap>,
    rows: Vec<Tap>,
    /// One output row's vertical pass at the input width, unrounded: each
    /// value is at most `255 * 256`, so it fits 16 bits.
    blended: Vec<u16>,
}

impl DisplayScaler {
    /// Scale all of `src` into all of `dst`.
    pub fn scale(&mut self, src: &PlaneView<'_>, dst: &mut PlaneViewMut<'_>) {
        let (in_w, in_h) = (src.width(), src.height());
        let (out_w, out_h) = (dst.width(), dst.height());
        if self.sizes != [in_w, in_h, out_w, out_h] {
            self.sizes = [in_w, in_h, out_w, out_h];
            self.columns = taps(in_w, out_w);
            self.rows = taps(in_h, out_h);
        }
        self.blended.resize(in_w, 0);
        let round = 1u32 << (2 * WEIGHT_BITS - 1);
        for (y, t) in self.rows.iter().enumerate() {
            let (top, bottom) = (&src.row(t.first)[..in_w], &src.row(t.second)[..in_w]);
            let (upper, lower) = (WEIGHT_ONE - t.weight, t.weight);
            for ((o, &a), &b) in self.blended.iter_mut().zip(top).zip(bottom) {
                // At most 255 * 256 (the weights sum to 256).
                *o = (u32::from(a) * upper + u32::from(b) * lower) as u16;
            }
            for (out, c) in dst.row_mut(y)[..out_w].iter_mut().zip(&self.columns) {
                let sum = u32::from(self.blended[c.first]) * (WEIGHT_ONE - c.weight)
                    + u32::from(self.blended[c.second]) * c.weight;
                *out = ((sum + round) >> (2 * WEIGHT_BITS)) as u8;
            }
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

    /// The two-pass definition (horizontal first, as the Python twin
    /// computes it): the vertical-first scaler must equal it exactly.
    fn horizontal_first(src: &Plane, width: usize, height: usize) -> Vec<Vec<u8>> {
        let view = src.view();
        let columns = taps(view.width(), width);
        let rows: Vec<Vec<u32>> = view
            .rows()
            .map(|row| {
                columns
                    .iter()
                    .map(|t| {
                        u32::from(row[t.first]) * (WEIGHT_ONE - t.weight)
                            + u32::from(row[t.second]) * t.weight
                    })
                    .collect()
            })
            .collect();
        taps(view.height(), height)
            .iter()
            .map(|t| {
                rows[t.first]
                    .iter()
                    .zip(&rows[t.second])
                    .map(|(&a, &b)| {
                        ((a * (WEIGHT_ONE - t.weight) + b * t.weight + (1 << 15)) >> 16) as u8
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn vertical_first_equals_horizontal_first() {
        let src = plane(53, 41, |x, y| ((x * 97 + y * 31) ^ (x * y)) as u8);
        for (w, h) in [(35, 27), (106, 82), (53, 20), (20, 41), (1, 1), (200, 3)] {
            assert_eq!(scaled(&src, w, h), horizontal_first(&src, w, h), "{w}x{h}");
        }
    }

    /// A reused scaler gives each call's own result, whatever the sizes
    /// of the call before.
    #[test]
    fn reused_scaler_matches_fresh_calls() {
        let mut scaler = DisplayScaler::default();
        let src = plane(37, 23, |x, y| (x * 7 + y * 29) as u8);
        for (w, h) in [(19, 11), (74, 46), (19, 11), (37, 23)] {
            let mut dst = Plane::new(w, h, 0).unwrap();
            scaler.scale(&src.view(), &mut dst.view_mut());
            let rows: Vec<Vec<u8>> = dst.view().rows().map(<[u8]>::to_vec).collect();
            assert_eq!(rows, scaled(&src, w, h), "{w}x{h}");
        }
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
