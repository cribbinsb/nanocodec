//! HEVC intra kernels: angular lines through the shared directional run
//! (`crate::intra_runs`), and the [`super::lanes`] planar kernel compiled for AVX2 (4-lane vectors
//! for 4x4 blocks, 8 lanes otherwise).

use super::{AngularFn, HevcIntraFns, PlanarEdges, PlanarFn, lanes};
use crate::simd::avx2::{I16x4, I16x8};

/// The AVX2 table.
pub(super) const KERNELS: HevcIntraFns = HevcIntraFns {
    angular: angular as AngularFn,
    planar: planar as PlanarFn,
};

#[target_feature(enable = "avx2")]
unsafe fn angular(refs: &[u8], n: usize, angle: i32, out: &mut [u8]) {
    // SAFETY: AVX2 is enabled here; the dispatcher checked the slices, `n` and `angle`.
    super::angular_lines(refs, n, angle, out, |window, weight, line| unsafe {
        crate::intra_runs::avx2::run2(window, weight, line);
    });
}

#[target_feature(enable = "avx2")]
unsafe fn planar(edges: &PlanarEdges<'_>, n: usize, out: &mut [u8]) {
    // SAFETY: as for `angular`.
    unsafe {
        if n == 4 {
            lanes::planar::<I16x4>(edges, n, out);
        } else {
            lanes::planar::<I16x8>(edges, n, out);
        }
    }
}
