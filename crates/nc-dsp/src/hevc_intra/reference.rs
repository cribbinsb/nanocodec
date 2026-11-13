//! Scalar HEVC intra kernels: the oracle, transcribed from §8.4.4.2.5
//! (planar, eq. 8-48) and §8.4.4.2.6 (angular, eq. 8-60/8-61).

use super::{ANGLE_FRACTION_BITS, AngularFn, HevcIntraFns, PlanarEdges, PlanarFn};

/// The scalar table.
pub(crate) const KERNELS: HevcIntraFns = HevcIntraFns {
    angular: angular as AngularFn,
    planar: planar as PlanarFn,
};

/// Angular lines (module docs of [`super`]).
pub(crate) fn angular(refs: &[u8], n: usize, angle: i32, out: &mut [u8]) {
    let one = 1 << ANGLE_FRACTION_BITS;
    for (j, line) in out[..n * n].chunks_exact_mut(n).enumerate() {
        let pos = (j as i32 + 1) * angle;
        let (idx, fact) = (pos >> ANGLE_FRACTION_BITS, pos & (one - 1));
        let start = (n as i32 + idx + 1) as usize;
        for (x, o) in line.iter_mut().enumerate() {
            let a = i32::from(refs[start + x]);
            // A whole-sample position reads `a` only: for the steepest angle
            // the sample after the last line's `a` is past `ref[2n]`.
            *o = if fact == 0 {
                a as u8
            } else {
                let b = i32::from(refs[start + x + 1]);
                (((one - fact) * a + fact * b + one / 2) >> ANGLE_FRACTION_BITS) as u8
            };
        }
    }
}

/// Planar (eq. 8-48).
pub(crate) fn planar(edges: &PlanarEdges<'_>, n: usize, out: &mut [u8]) {
    let shift = n.trailing_zeros() + 1;
    let side = n as i32;
    let (top_right, bottom_left) = (i32::from(edges.top_right), i32::from(edges.bottom_left));
    for (row, line) in out[..n * n].chunks_exact_mut(n).enumerate() {
        let left = i32::from(edges.left[row]);
        let row = row as i32;
        for (col, o) in line.iter_mut().enumerate() {
            let col = col as i32;
            let above = i32::from(edges.above[col as usize]);
            let v = (side - 1 - col) * left
                + (col + 1) * top_right
                + (side - 1 - row) * above
                + (row + 1) * bottom_left
                + side;
            *o = (v >> shift) as u8;
        }
    }
}
