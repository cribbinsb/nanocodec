//! Squared error of a square block, and how it spreads over the block's
//! quadrants: the distortion terms every backend's RD trials and
//! transform-split gates use (shared `Dsp::variance` kernels).

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::{BlockSize, Dsp};

/// Squared error between `source` and `recon` over the `n`-square block of
/// `plane` at `(x, y)` (inside both pictures).
#[must_use]
pub fn block_sse(
    dsp: &Dsp,
    (source, recon): (&Picture, &Picture),
    plane: PlaneIndex,
    (x, y, side): (usize, usize, usize),
) -> u64 {
    let size = BlockSize::new(side, side).expect("square block sizes are kernel sizes");
    let (source, recon) = (source.plane(plane).view(), recon.plane(plane).view());
    let source = source
        .sub_view(x, y, side, side)
        .expect("block inside the picture");
    let recon = recon
        .sub_view(x, y, side, side)
        .expect("block inside the picture");
    u64::from(dsp.variance(size, &source, &recon).sse)
}

/// Squared error between `a` and `b` (`n`-square, `n` ≥ 8) in each of their
/// four quadrants, in raster order: what a transform-split gate compares
/// (a residual concentrated in one quadrant pays for smaller transforms).
#[must_use]
pub fn quadrant_sse(dsp: &Dsp, a: &PlaneView<'_>, b: &PlaneView<'_>, n: usize) -> [u64; 4] {
    let half = n / 2;
    let size = BlockSize::new(half, half).expect("square quadrant sizes are kernel sizes");
    std::array::from_fn(|i| {
        let (qx, qy) = ((i % 2) * half, (i / 2) * half);
        let a = a
            .sub_view(qx, qy, half, half)
            .expect("quadrant inside the block");
        let b = b
            .sub_view(qx, qy, half, half)
            .expect("quadrant inside the block");
        u64::from(dsp.variance(size, &a, &b).sse)
    })
}
