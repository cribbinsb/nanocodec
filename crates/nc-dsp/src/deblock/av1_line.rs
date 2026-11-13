//! AV1 edge-line driver: masked four-line descriptors, using the existing
//! scalar equations or packed SIMD edge kernels. Merging identical adjacent
//! descriptors shares SIMD lanes without changing per-line decisions or the
//! codec's ordering of distinct edges (AV1 §7.14).

use super::av1::{AV1_EDGE_LINES, Av1EdgeFilter};

/// The caller validates every active footprint and enables the ISA.
pub(crate) type LineFn = unsafe fn(&mut [u8], usize, usize, usize, &[Option<Av1EdgeFilter>]);

/// Independent scalar reference: visit each active descriptor separately.
pub(crate) fn scalar(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    segments: &[Option<Av1EdgeFilter>],
) {
    for (i, filter) in segments.iter().enumerate() {
        if let Some(filter) = filter {
            super::av1::filter_edge(
                plane,
                q0 + i * AV1_EDGE_LINES * along,
                across,
                along,
                AV1_EDGE_LINES,
                *filter,
            );
        }
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(clippy::inline_always)] // specialize the driver inside the ISA entry point
#[inline(always)]
unsafe fn packed<B: crate::lpf_bytes::Bytes>(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    segments: &[Option<Av1EdgeFilter>],
) {
    // SAFETY: the line entry validated all complete four-line descriptors.
    unsafe {
        packed_descriptors::<B>(
            plane,
            q0,
            across,
            along,
            segments.len() * AV1_EDGE_LINES,
            super::line::Descriptors {
                entries: segments,
                stride: 1,
                count: segments.len(),
            },
        );
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(clippy::inline_always)]
#[inline(always)]
pub(super) unsafe fn packed_descriptors<B: crate::lpf_bytes::Bytes>(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    lines: usize,
    descriptors: super::line::Descriptors<'_, Av1EdgeFilter>,
) {
    for (first, count, filter) in descriptors.runs() {
        let start = first * AV1_EDGE_LINES;
        let count = (count * AV1_EDGE_LINES).min(lines - start);
        let full = count / AV1_EDGE_LINES * AV1_EDGE_LINES;
        if full != 0 {
            // SAFETY: validated active footprints, complete four-line groups,
            // and the caller enabled B's ISA. Along-edge lines are independent.
            unsafe {
                super::av1_packed::filter_edge::<B>(
                    plane,
                    q0 + start * along,
                    across,
                    along,
                    full,
                    filter,
                );
            }
        }
        if full != count {
            // The last picture unit may have 1–3 lines. The scalar kernel
            // reads only those lines; packed kernels require complete groups.
            super::av1::filter_edge(
                plane,
                q0 + (start + full) * along,
                across,
                along,
                count - full,
                filter,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    segments: &[Option<Av1EdgeFilter>],
) {
    // SAFETY: dispatch validates the footprint and this function enables AVX2.
    unsafe { packed::<crate::lpf_bytes::avx2::B>(plane, q0, across, along, segments) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    segments: &[Option<Av1EdgeFilter>],
) {
    // SAFETY: dispatch validates the footprint and this function enables NEON.
    unsafe { packed::<crate::lpf_bytes::neon::B>(plane, q0, across, along, segments) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) unsafe fn simd128(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    segments: &[Option<Av1EdgeFilter>],
) {
    // SAFETY: dispatch validates the footprint and the build enables SIMD128.
    unsafe { packed::<crate::lpf_bytes::simd128::B>(plane, q0, across, along, segments) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Av1FilterLength, Dsp, Isa};
    use nc_common::PlaneViewMut;

    const SIDE: usize = 16;
    const FILTER: Av1EdgeFilter = Av1EdgeFilter {
        length: Av1FilterLength::Fourteen,
        limit: 32,
        blimit: 64,
        thresh: 8,
    };

    #[test]
    fn rejects_active_footprints_outside_picture() {
        let dsp = Dsp::with_isa(Isa::Scalar).expect("scalar");
        for vertical in [false, true] {
            for pos in [(0, 0), (8, 14), (14, 8), (usize::MAX, 8), (8, usize::MAX)] {
                let result = std::panic::catch_unwind(|| {
                    let mut pixels = [0; SIDE * SIDE];
                    let mut dst = PlaneViewMut::new(&mut pixels, SIDE, SIDE, SIDE).expect("plane");
                    dsp.av1_loop_filter_line(&mut dst, pos, vertical, &[Some(FILTER)]);
                });
                assert!(result.is_err(), "{pos:?}, vertical={vertical}");
            }
        }
    }

    #[test]
    fn trailing_masks_need_no_picture_storage() {
        let dsp = Dsp::with_isa(Isa::Scalar).expect("scalar");
        let mut pixels = [123; SIDE * SIDE];
        let mut dst = PlaneViewMut::new(&mut pixels, SIDE, SIDE, SIDE).expect("plane");
        let mut segments = [None; SIDE];
        segments[0] = Some(FILTER);
        dsp.av1_loop_filter_line(&mut dst, (8, 8), true, &segments);
        dsp.av1_loop_filter_line(&mut dst, (8, 8), false, &segments);
        dsp.av1_loop_filter_line(&mut dst, (usize::MAX, usize::MAX), true, &[]);
        dsp.av1_loop_filter_line(&mut dst, (usize::MAX, usize::MAX), true, &[None]);
        assert_eq!(pixels, [123; SIDE * SIDE]);
    }
}
