//! HEVC intra kernels: angular lines through the shared directional run
//! (`crate::intra_runs`), and the [`super::lanes`] planar kernel compiled
//! for AVX2 (4, 8 or 16 lanes, as wide as the block side).

use super::{AngularFn, HevcIntraFns, PlanarEdges, PlanarFn, lanes};
use crate::simd::avx2::{I16x4, I16x8, I16x16};

/// The AVX2 table.
pub(super) const KERNELS: HevcIntraFns = HevcIntraFns {
    angular: angular as AngularFn,
    planar: planar as PlanarFn,
};

#[target_feature(enable = "avx2")]
unsafe fn angular(refs: &[u8], n: usize, angle: i32, out: &mut [u8]) {
    // SAFETY: AVX2 is enabled here; the dispatcher checked the slices, `n` and `angle`.
    unsafe { super::angular_lines::<crate::intra_runs::avx2::Avx2Runs>(refs, n, angle, out) }
}

#[target_feature(enable = "avx2")]
unsafe fn planar(edges: &PlanarEdges<'_>, n: usize, out: &mut [u8]) {
    // SAFETY: as for `angular`.
    unsafe {
        match n {
            4 => lanes::planar::<I16x4, { super::MAX_SIDE }>(edges, n, out),
            8 => lanes::planar::<I16x8, { super::MAX_SIDE }>(edges, n, out),
            _ => lanes::planar::<I16x16, { super::MAX_SIDE }>(edges, n, out),
        }
    }
}
