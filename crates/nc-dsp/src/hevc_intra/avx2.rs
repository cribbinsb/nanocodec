//! The [`super::lanes`] HEVC intra kernels compiled for AVX2: 4-lane vectors for
//! 4x4 blocks, 8 lanes otherwise.

use super::{AngularFn, HevcIntraFns, PlanarEdges, PlanarFn, lanes};
use crate::simd::avx2::{I16x4, I16x8};

/// The AVX2 table.
pub(super) const KERNELS: HevcIntraFns = HevcIntraFns {
    angular: angular as AngularFn,
    planar: planar as PlanarFn,
};

#[target_feature(enable = "avx2")]
unsafe fn angular(refs: &[u8], n: usize, angle: i32, out: &mut [u8]) {
    // SAFETY: the ISA is enabled here; the dispatcher checked the slices,
    // `n` (4..=32, a power of two) and `angle`.
    unsafe {
        if n == 4 {
            lanes::angular::<I16x4>(refs, n, angle, out);
        } else {
            lanes::angular::<I16x8>(refs, n, angle, out);
        }
    }
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
