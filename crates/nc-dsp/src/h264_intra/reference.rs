//! Scalar H.264 `Intra_8x8` prediction: the oracle, transcribed from the
//! per-sample formulas of §8.3.2.2.2–10 over the filtered samples `p′`.

#![allow(
    clippy::many_single_char_names,
    reason = "x, y, z and the sample accessors follow the standard's formulas"
)]

use super::{H264_INTRA_8X8_SIDE, H264Intra8x8Refs, avg2, avg3, h264_intra_8x8_mode as mode};

/// Predict `mode` into `out` (raster, stride 8).
pub(super) fn predict(refs: &H264Intra8x8Refs, m: u8, out: &mut [u8; 64]) {
    let n = H264_INTRA_8X8_SIDE as isize;
    // p′[x, −1] and p′[−1, y], both reaching the corner at −1.
    let t = |x: isize| {
        if x < 0 {
            refs.corner()
        } else {
            refs.t(x as usize)
        }
    };
    let l = |y: isize| {
        if y < 0 {
            refs.corner()
        } else {
            refs.l(y as usize)
        }
    };
    let dc = refs.dc();
    for (i, sample) in out.iter_mut().enumerate() {
        let (x, y) = (i as isize % n, i as isize / n);
        *sample = match m {
            mode::VERTICAL => t(x),
            mode::HORIZONTAL => l(y),
            mode::DC => dc,
            // §8.3.2.2.5
            mode::DIAGONAL_DOWN_LEFT => {
                if x == 7 && y == 7 {
                    avg3(t(14), t(15), t(15))
                } else {
                    avg3(t(x + y), t(x + y + 1), t(x + y + 2))
                }
            }
            // §8.3.2.2.6
            mode::DIAGONAL_DOWN_RIGHT => match x.cmp(&y) {
                core::cmp::Ordering::Greater => avg3(t(x - y - 2), t(x - y - 1), t(x - y)),
                core::cmp::Ordering::Less => avg3(l(y - x - 2), l(y - x - 1), l(y - x)),
                core::cmp::Ordering::Equal => avg3(t(0), t(-1), l(0)),
            },
            // §8.3.2.2.7
            mode::VERTICAL_RIGHT => {
                let z = 2 * x - y;
                let k = x - (y >> 1);
                if z >= 0 && z % 2 == 0 {
                    avg2(t(k - 1), t(k))
                } else if z > 0 {
                    avg3(t(k - 2), t(k - 1), t(k))
                } else if z == -1 {
                    avg3(l(0), l(-1), t(0))
                } else {
                    avg3(l(y - 2 * x - 1), l(y - 2 * x - 2), l(y - 2 * x - 3))
                }
            }
            // §8.3.2.2.8
            mode::HORIZONTAL_DOWN => {
                let z = 2 * y - x;
                let j = y - (x >> 1);
                if z >= 0 && z % 2 == 0 {
                    avg2(l(j - 1), l(j))
                } else if z > 0 {
                    avg3(l(j - 2), l(j - 1), l(j))
                } else if z == -1 {
                    avg3(l(0), l(-1), t(0))
                } else {
                    avg3(t(x - 2 * y - 1), t(x - 2 * y - 2), t(x - 2 * y - 3))
                }
            }
            // §8.3.2.2.9
            mode::VERTICAL_LEFT => {
                let i = x + (y >> 1);
                if y % 2 == 0 {
                    avg2(t(i), t(i + 1))
                } else {
                    avg3(t(i), t(i + 1), t(i + 2))
                }
            }
            // §8.3.2.2.10
            mode::HORIZONTAL_UP => {
                let z = x + 2 * y;
                let j = y + (x >> 1);
                match z {
                    13 => avg3(l(6), l(7), l(7)),
                    z if z > 13 => l(7),
                    z if z % 2 == 0 => avg2(l(j), l(j + 1)),
                    _ => avg3(l(j), l(j + 1), l(j + 2)),
                }
            }
            _ => unreachable!("Intra8x8PredMode is 0..=8"),
        };
    }
}
