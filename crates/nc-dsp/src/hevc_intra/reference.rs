//! Scalar HEVC intra kernels: the oracle, transcribed from §8.4.4.2.4
//! (planar, eq. 8-48) and §8.4.4.2.6 (angular, eq. 8-60/8-61). Generic over
//! the sample type (one source for the 8-bit table and the high-bit-depth
//! one): neither mode can leave the references' range, so no clip and no
//! bit depth is involved.

use nc_common::Sample;

use super::{ANGLE_FRACTION_BITS, AngularFn, HevcIntraFns, PlanarEdges, PlanarFn};

/// The scalar table.
pub(crate) const KERNELS: HevcIntraFns = HevcIntraFns {
    angular: angular::<u8> as AngularFn,
    planar: planar::<u8> as PlanarFn,
};

/// Angular lines (module docs of [`super`]).
pub(crate) fn angular<T: Sample>(refs: &[T], n: usize, angle: i32, out: &mut [T]) {
    let one = 1 << ANGLE_FRACTION_BITS;
    for (j, line) in out[..n * n].chunks_exact_mut(n).enumerate() {
        let pos = (j as i32 + 1) * angle;
        let (idx, fact) = (pos >> ANGLE_FRACTION_BITS, pos & (one - 1));
        let start = (n as i32 + idx + 1) as usize;
        for (x, o) in line.iter_mut().enumerate() {
            let a = refs[start + x];
            // A whole-sample position reads `a` only: for the steepest angle
            // the sample after the last line's `a` is past `ref[2n]`.
            *o = if fact == 0 {
                a
            } else {
                let (a, b) = (a.to_i32(), refs[start + x + 1].to_i32());
                T::from_i32(((one - fact) * a + fact * b + one / 2) >> ANGLE_FRACTION_BITS)
            };
        }
    }
}

/// Planar (eq. 8-48).
pub(crate) fn planar<T: Sample>(edges: &PlanarEdges<'_, T>, n: usize, out: &mut [T]) {
    let shift = n.trailing_zeros() + 1;
    let side = n as i32;
    let (top_right, bottom_left) = (edges.top_right.to_i32(), edges.bottom_left.to_i32());
    for (row, line) in out[..n * n].chunks_exact_mut(n).enumerate() {
        let left = edges.left[row].to_i32();
        let row = row as i32;
        for (col, o) in line.iter_mut().enumerate() {
            let col = col as i32;
            let above = edges.above[col as usize].to_i32();
            let v = (side - 1 - col) * left
                + (col + 1) * top_right
                + (side - 1 - row) * above
                + (row + 1) * bottom_left
                + side;
            *o = T::from_i32(v >> shift);
        }
    }
}
