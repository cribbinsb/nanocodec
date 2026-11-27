//! Chroma from luma at every chroma transform size and subsampling (4:2:0,
//! 4:2:2, 4:4:4; AV1 §7.11.5; libaom `av1/common/cfl.c`, BSD-2-Clause, see
//! `THIRD_PARTY.md`). The SIMD kernels (`av1_intra::high`) take 4:2:0, the
//! format of every encoder and of most streams; this scalar reference takes
//! every subsampling.
//!
//! [`CflAc::from_luma`] subsamples the reconstructed luma to chroma
//! resolution in 1/8 units (`Q3`), repeats its last column and row where
//! the luma area decoded inside the picture is smaller than the chroma
//! block (`MaxLumaW` / `MaxLumaH`), and subtracts the rounded average.
//! [`CflAc::predict`] adds `alpha * ac` to the DC prediction already in the
//! destination. Both take samples of any depth.

use nc_common::{BitDepth, PlaneView, PlaneViewMut, Sample};

/// Largest chroma side chroma-from-luma predicts.
pub const CFL_MAX_SIDE: usize = 32;
/// Fractional bits of the subsampled luma (`Q3`): a sum of
/// `1 << (sub_x + sub_y)` luma samples carries that many bits, the rest is
/// a shift (`3 - subX - subY`). At 12 bits a `Q3` value is at most
/// `4095 << 3`, within 16 bits.
const LUMA_FRACTION_BITS: u32 = 3;
/// 4:2:0, the subsampling of the SIMD kernels.
pub const CFL_420: (usize, usize) = (1, 1);
/// `alpha` is in 1/8 units and the AC in 1/8 units: the product is `Q6`.
const SCALED_BITS: u32 = 6;

/// The zero-mean subsampled luma of one chroma block, row-major.
#[derive(Debug, Clone)]
pub struct CflAc {
    ac: [i16; CFL_MAX_SIDE * CFL_MAX_SIDE],
    width: usize,
    height: usize,
}

impl CflAc {
    /// The AC contribution of a `width`x`height` chroma block of a format
    /// subsampled by `(sub_x, sub_y)` (log2). `luma` starts at the
    /// co-located luma origin; its first `luma_width` x `luma_height`
    /// samples were decoded inside the picture (multiples of the
    /// subsampling, at most the chroma size scaled up by it).
    ///
    /// # Panics
    ///
    /// If a chroma side is not 4, 8, 16 or 32, the luma extent is zero,
    /// not a multiple of the subsampling or beyond the scaled chroma size,
    /// a subsampling is above 1, or `luma` is smaller than the extent.
    #[must_use]
    pub fn from_luma<T: Sample>(
        luma: &PlaneView<'_, T>,
        (luma_width, luma_height): (usize, usize),
        (sub_x, sub_y): (usize, usize),
        width: usize,
        height: usize,
    ) -> Self {
        assert!(
            is_side(width) && is_side(height),
            "CfL block {width}x{height}"
        );
        assert!(sub_x <= 1 && sub_y <= 1, "CfL subsampling");
        for (extent, side, sub) in [(luma_width, width, sub_x), (luma_height, height, sub_y)] {
            assert!(
                extent > 0 && extent.is_multiple_of(1 << sub) && extent <= side << sub,
                "CfL luma extent {extent} for chroma side {side}"
            );
        }
        assert!(luma.width() >= luma_width && luma.height() >= luma_height);
        let (stored_w, stored_h) = (luma_width >> sub_x, luma_height >> sub_y);
        let shift = LUMA_FRACTION_BITS - (sub_x + sub_y) as u32;
        let mut ac = [0i16; CFL_MAX_SIDE * CFL_MAX_SIDE];
        for r in 0..height {
            let row = &mut ac[r * width..(r + 1) * width];
            if r < stored_h {
                let rows = (luma.row(r << sub_y), luma.row((r << sub_y) + sub_y));
                for (c, value) in row[..stored_w].iter_mut().enumerate() {
                    let (left, right) = (c << sub_x, (c << sub_x) + sub_x);
                    let mut sum = rows.0[left].to_i32();
                    if sub_x == 1 {
                        sum += rows.0[right].to_i32();
                    }
                    if sub_y == 1 {
                        sum += rows.1[left].to_i32();
                        if sub_x == 1 {
                            sum += rows.1[right].to_i32();
                        }
                    }
                    *value = (sum << shift) as i16;
                }
                let last = row[stored_w - 1];
                row[stored_w..].fill(last);
            } else {
                ac.copy_within((r - 1) * width..r * width, r * width);
            }
        }
        let area = width * height;
        let samples = &mut ac[..area];
        let sum: i32 = samples.iter().map(|&v| i32::from(v)).sum();
        let average = (sum + (area / 2) as i32) >> area.trailing_zeros();
        for value in samples {
            *value -= average as i16;
        }
        Self { ac, width, height }
    }

    /// Add `alpha_q3 * ac` (rounded, §7.11.5) to the DC prediction in `dst`,
    /// clipping to `depth`. `alpha_q3` is `CflAlphaU` / `CflAlphaV`.
    ///
    /// # Panics
    ///
    /// If `dst` is not the block's size.
    pub fn predict<T: Sample>(
        &self,
        alpha_q3: i32,
        dst: &mut PlaneViewMut<'_, T>,
        depth: BitDepth,
    ) {
        let (w, h) = (self.width, self.height);
        assert!(
            dst.width() == w && dst.height() == h,
            "CfL destination size"
        );
        for r in 0..h {
            let ac = &self.ac[r * w..(r + 1) * w];
            for (pixel, &value) in dst.row_mut(r)[..w].iter_mut().zip(ac) {
                let scaled = round_signed(alpha_q3 * i32::from(value), SCALED_BITS);
                *pixel = depth.clip_to(pixel.to_i32() + scaled);
            }
        }
    }
}

/// `Round2Signed(x, bits)`: rounds magnitudes half away from zero.
fn round_signed(x: i32, bits: u32) -> i32 {
    let magnitude = (x.abs() + (1 << (bits - 1))) >> bits;
    if x < 0 { -magnitude } else { magnitude }
}

const fn is_side(n: usize) -> bool {
    n.is_power_of_two() && n >= 4 && n <= CFL_MAX_SIDE
}

/// Scalar one-pass `CfL` prediction (`dc + Round2Signed(alpha_q3 * ac, 6)`)
/// of a format subsampled by `subsampling`, matching [`CflAc::from_luma`]
/// + [`CflAc::predict`].
pub(crate) fn cfl_predict_scalar<T: Sample>(
    luma: &PlaneView<'_, T>,
    luma_extent: (usize, usize),
    subsampling: (usize, usize),
    dc: T,
    alpha_q3: i32,
    dst: &mut PlaneViewMut<'_, T>,
    depth: BitDepth,
) {
    let (width, height) = (dst.width(), dst.height());
    if alpha_q3 == 0 {
        assert!(
            is_side(width) && is_side(height),
            "CfL block {width}x{height}"
        );
        dst.fill(dc);
        return;
    }
    let ac = CflAc::from_luma(luma, luma_extent, subsampling, width, height);
    dst.fill(dc);
    ac.predict(alpha_q3, dst, depth);
}

/// [`cfl_predict_scalar`] of 4:2:0: the scalar entry of the kernel tables.
pub(crate) fn cfl_predict_420_scalar<T: Sample>(
    luma: &PlaneView<'_, T>,
    luma_extent: (usize, usize),
    dc: T,
    alpha_q3: i32,
    dst: &mut PlaneViewMut<'_, T>,
    depth: BitDepth,
) {
    cfl_predict_scalar(luma, luma_extent, CFL_420, dc, alpha_q3, dst, depth);
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::Plane;

    /// §7.11.5 written out: the subsampled luma, its rounded average, and
    /// `dc + Round2Signed(alpha * (L - avg), 6)` clipped.
    fn spec_prediction(
        luma: &PlaneView<'_, u16>,
        (luma_w, luma_h): (usize, usize),
        (sub_x, sub_y): (usize, usize),
        (w, h): (usize, usize),
        (dc, alpha): (i32, i32),
    ) -> Vec<i32> {
        let mut l = vec![0i32; w * h];
        for i in 0..h {
            let luma_y = i.min((luma_h >> sub_y) - 1) << sub_y;
            for j in 0..w {
                let luma_x = j.min((luma_w >> sub_x) - 1) << sub_x;
                let mut t = 0;
                for dy in 0..=sub_y {
                    for dx in 0..=sub_x {
                        t += i32::from(luma.row(luma_y + dy)[luma_x + dx]);
                    }
                }
                l[i * w + j] = t << (3 - sub_x - sub_y);
            }
        }
        let area = (w * h) as i32;
        let avg = (l.iter().sum::<i32>() + area / 2) >> area.trailing_zeros();
        l.iter()
            .map(|&v| {
                BitDepth::new(12)
                    .unwrap()
                    .clip(dc + round_signed(alpha * (v - avg), 6))
            })
            .collect()
    }

    #[test]
    fn every_subsampling_follows_the_specification() {
        let mut plane = Plane::<u16>::new(64, 64, 0).unwrap();
        let mut view = plane.view_mut();
        for y in 0..64 {
            for (x, s) in view.row_mut(y).iter_mut().enumerate() {
                *s = ((x * 97 + y * 61 + x * y * 13) % 4096) as u16;
            }
        }
        let depth = BitDepth::new(12).unwrap();
        for sub in [(0, 0), (1, 0), (1, 1)] {
            for (w, h) in [(4, 4), (8, 4), (16, 8), (32, 32)] {
                let full = (w << sub.0, h << sub.1);
                for extent in [full, (full.0, 4 << sub.1), (4 << sub.0, full.1)] {
                    let luma = plane.region(0, 0, full.0, full.1).unwrap();
                    let mut out = vec![0u16; w * h];
                    let mut dst = PlaneViewMut::new(&mut out, w, w, h).unwrap();
                    cfl_predict_scalar(&luma, extent, sub, 2000, -11, &mut dst, depth);
                    let expected = spec_prediction(&luma, extent, sub, (w, h), (2000, -11));
                    let got: Vec<i32> = out.iter().map(|&s| i32::from(s)).collect();
                    assert_eq!(got, expected, "{sub:?} {w}x{h} {extent:?}");
                }
            }
        }
    }
}
