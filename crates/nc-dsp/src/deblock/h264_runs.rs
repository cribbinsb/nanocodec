//! H.264's ordered sixteen-line edge descriptors. Validation and the scalar
//! oracle retain the original edge API; x86 shares the grouped byte filters.
use super::runs::{EdgeRun, Plane, ValidatedRuns};
use super::{Edge, EdgeFn, H264EdgeParams};

pub(crate) type RunsFn = unsafe fn(&mut [u8], Plane, &[EdgeRun<H264EdgeParams>]);

#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn ordered(data: &mut [u8], plane: Plane, runs: &[EdgeRun<H264EdgeParams>], kernel: EdgeFn) {
    for run in ValidatedRuns::new(plane, runs, false, 16, |p: H264EdgeParams| {
        assert!(p.bs.iter().all(|&v| v <= 4), "bS above four");
        4
    }) {
        assert_eq!(run.lines, 16, "H.264 luma edge has sixteen lines");
        let (base, across, along) = plane.address(&run);
        let edge = Edge {
            base,
            across,
            along,
            bs: run.filter.bs,
            tc0: run.filter.tc0.map(i16::from),
            alpha: i16::from(run.filter.alpha),
            beta: i16::from(run.filter.beta),
        };
        // SAFETY: dispatch selected a supported ISA; validation covers p3..q3.
        unsafe {
            kernel(data, &edge);
        }
    }
}

pub(crate) fn scalar(data: &mut [u8], plane: Plane, runs: &[EdgeRun<H264EdgeParams>]) {
    // SAFETY: scalar needs no CPU feature; ordered validates each footprint.
    unsafe {
        ordered(data, plane, runs, super::scalar::deblock_fns().luma);
    }
}
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn neon(data: &mut [u8], plane: Plane, runs: &[EdgeRun<H264EdgeParams>]) {
    // SAFETY: NEON dispatch and each validated footprint.
    unsafe {
        ordered(data, plane, runs, super::neon::deblock_fns().luma);
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) unsafe fn simd128(data: &mut [u8], plane: Plane, runs: &[EdgeRun<H264EdgeParams>]) {
    // SAFETY: SIMD128 dispatch and each validated footprint.
    unsafe {
        ordered(data, plane, runs, super::simd128::deblock_fns().luma);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2(data: &mut [u8], plane: Plane, runs: &[EdgeRun<H264EdgeParams>]) {
    // SAFETY: AVX2 enabled; grouped traversal validates each footprint.
    unsafe {
        super::h264_bytes::grouped::runs::<super::h264_bytes::wide::Pair>(data, plane, runs);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl,gfni")]
pub(crate) unsafe fn avx512(data: &mut [u8], plane: Plane, runs: &[EdgeRun<H264EdgeParams>]) {
    // SAFETY: the ICL ISA is enabled; grouped traversal validates each footprint.
    unsafe {
        super::h264_bytes::grouped::runs::<super::h264_bytes::wide::Quad>(data, plane, runs);
    }
}

pub(crate) unsafe fn pair_scalar(data: &mut [u8], edges: &[Edge; 2]) {
    let kernel = super::scalar::deblock_fns().luma;
    // SAFETY: the caller validates both independent footprints and ISA.
    unsafe {
        kernel(data, &edges[0]);
        kernel(data, &edges[1]);
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn pair_neon(data: &mut [u8], edges: &[Edge; 2]) {
    let kernel = super::neon::deblock_fns().luma;
    // SAFETY: the caller validates both independent footprints and ISA.
    unsafe {
        kernel(data, &edges[0]);
        kernel(data, &edges[1]);
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) unsafe fn pair_simd128(data: &mut [u8], edges: &[Edge; 2]) {
    let kernel = super::simd128::deblock_fns().luma;
    // SAFETY: the caller validates both independent footprints and ISA.
    unsafe {
        kernel(data, &edges[0]);
        kernel(data, &edges[1]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pair_avx2(data: &mut [u8], edges: &[Edge; 2]) {
    // SAFETY: caller validates disjoint footprints; the ISA is enabled here.
    unsafe {
        super::h264_bytes::grouped::pair::<super::h264_bytes::wide::Pair>(data, edges);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq,avx512vl,gfni")]
pub(crate) unsafe fn pair_avx512(data: &mut [u8], edges: &[Edge; 2]) {
    // SAFETY: caller validates disjoint footprints; the ISA is enabled here.
    unsafe {
        super::h264_bytes::grouped::pair::<super::h264_bytes::wide::IclPair>(data, edges);
    }
}
