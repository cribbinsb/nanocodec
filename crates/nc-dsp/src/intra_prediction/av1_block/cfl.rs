//! Chroma from luma for 4:2:0 at every chroma transform size (AV1 §7.11.5;
//! libaom `av1/common/cfl.c`, BSD-2-Clause, see `THIRD_PARTY.md`).
//!
//! [`CflAc::from_luma`] subsamples the reconstructed luma to chroma
//! resolution in 1/8 units (`Q3`), repeats its last column and row where
//! the luma area decoded inside the picture is smaller than the chroma
//! block (`MaxLumaW` / `MaxLumaH`), and subtracts the rounded average.
//! [`CflAc::predict`] adds `alpha * ac` to the DC prediction already in the
//! destination.

use nc_common::{PlaneView, PlaneViewMut};

/// Largest chroma side chroma-from-luma predicts.
pub const CFL_MAX_SIDE: usize = 32;
/// Fractional bits of the subsampled luma (`Q3`).
const LUMA_FRACTION_BITS: u32 = 3;
/// A 2x2 luma sum carries two bits; the rest of `Q3` is a shift.
const SUM_420_SHIFT: u32 = LUMA_FRACTION_BITS - 2;
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
    /// The AC contribution of a `width`x`height` chroma block. `luma` starts
    /// at the co-located luma origin; its first `luma_width` x
    /// `luma_height` samples were decoded inside the picture (even counts,
    /// at most twice the chroma size).
    ///
    /// # Panics
    ///
    /// If a chroma side is not 4, 8, 16 or 32, the luma extent is zero, odd
    /// or beyond twice the chroma size, or `luma` is smaller than it.
    #[must_use]
    pub fn from_luma(
        luma: &PlaneView<'_>,
        luma_width: usize,
        luma_height: usize,
        width: usize,
        height: usize,
    ) -> Self {
        assert!(
            is_side(width) && is_side(height),
            "CfL block {width}x{height}"
        );
        for (extent, side) in [(luma_width, width), (luma_height, height)] {
            assert!(
                extent > 0 && extent.is_multiple_of(2) && extent <= 2 * side,
                "CfL luma extent {extent} for chroma side {side}"
            );
        }
        assert!(luma.width() >= luma_width && luma.height() >= luma_height);
        let (stored_w, stored_h) = (luma_width / 2, luma_height / 2);
        let mut ac = [0i16; CFL_MAX_SIDE * CFL_MAX_SIDE];
        for r in 0..height {
            let row = &mut ac[r * width..(r + 1) * width];
            if r < stored_h {
                let (top, bottom) = (luma.row(2 * r), luma.row(2 * r + 1));
                for (c, value) in row[..stored_w].iter_mut().enumerate() {
                    let sum = u16::from(top[2 * c])
                        + u16::from(top[2 * c + 1])
                        + u16::from(bottom[2 * c])
                        + u16::from(bottom[2 * c + 1]);
                    *value = (sum << SUM_420_SHIFT) as i16;
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

    /// Add `alpha_q3 * ac` (rounded, §7.11.5) to the DC prediction in `dst`
    /// with 8-bit clipping. `alpha_q3` is `CflAlphaU` / `CflAlphaV`.
    ///
    /// # Panics
    ///
    /// If `dst` is not the block's size.
    pub fn predict(&self, alpha_q3: i32, dst: &mut PlaneViewMut<'_>) {
        let (w, h) = (self.width, self.height);
        assert!(
            dst.width() == w && dst.height() == h,
            "CfL destination size"
        );
        for r in 0..h {
            let ac = &self.ac[r * w..(r + 1) * w];
            for (pixel, &value) in dst.row_mut(r)[..w].iter_mut().zip(ac) {
                let scaled = round_signed(alpha_q3 * i32::from(value), SCALED_BITS);
                *pixel = (i32::from(*pixel) + scaled).clamp(0, i32::from(u8::MAX)) as u8;
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
