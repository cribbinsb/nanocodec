//! Masked VP9 edge lines: shared run grouping with AV1, with VP9's own
//! smoothing equations and scalar partial-line tails.

use super::{VP9_EDGE_LINES, Vp9EdgeFilter};
use crate::Isa;

/// The caller enables the ISA, validates active footprints and clips `lines`
/// to the picture. Only the final active descriptor may be partial.
pub(crate) type LineFn = unsafe fn(&mut [u8], usize, usize, usize, usize, &[Option<Vp9EdgeFilter>]);

pub(crate) fn kernel(isa: Isa) -> LineFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128,
        _ => scalar,
    }
}

/// Independent reference: each active descriptor visits its own lines.
fn scalar(
    data: &mut [u8],
    edge: usize,
    across: usize,
    along: usize,
    lines: usize,
    segments: &[Option<Vp9EdgeFilter>],
) {
    for (i, filter) in segments.iter().enumerate() {
        if let Some(filter) = filter {
            let start = i * VP9_EDGE_LINES;
            crate::vp9_deblock::vp9_filter_edge(
                data,
                edge + start * along,
                across,
                along,
                VP9_EDGE_LINES.min(lines - start),
                filter.width,
                filter.limits,
            );
        }
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(clippy::inline_always)] // expose the constant ISA function to the compiler
#[inline(always)]
unsafe fn merged(
    data: &mut [u8],
    edge: usize,
    across: usize,
    along: usize,
    lines: usize,
    segments: &[Option<Vp9EdgeFilter>],
    kernel: super::FilterFn,
) {
    let descriptors = crate::deblock::line::Descriptors {
        entries: segments,
        stride: 1,
        count: segments.len(),
    };
    for (first, count, filter) in descriptors.runs() {
        let start = first * VP9_EDGE_LINES;
        let count = (count * VP9_EDGE_LINES).min(lines - start);
        // SAFETY: dispatch checked every active footprint, and `count` clips
        // the last group to the picture. The selected kernel enables its ISA.
        unsafe {
            kernel(
                data,
                edge + start * along,
                across,
                along,
                count,
                filter.width,
                filter.limits,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2(
    data: &mut [u8],
    edge: usize,
    across: usize,
    along: usize,
    lines: usize,
    segments: &[Option<Vp9EdgeFilter>],
) {
    // SAFETY: forwarded validated footprint, AVX2 enabled here.
    unsafe {
        merged(
            data,
            edge,
            across,
            along,
            lines,
            segments,
            super::packed::avx2::filter,
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon(
    data: &mut [u8],
    edge: usize,
    across: usize,
    along: usize,
    lines: usize,
    segments: &[Option<Vp9EdgeFilter>],
) {
    // SAFETY: forwarded validated footprint, NEON enabled here.
    unsafe {
        merged(
            data,
            edge,
            across,
            along,
            lines,
            segments,
            super::packed::neon::filter,
        );
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn simd128(
    data: &mut [u8],
    edge: usize,
    across: usize,
    along: usize,
    lines: usize,
    segments: &[Option<Vp9EdgeFilter>],
) {
    // SAFETY: forwarded validated footprint, SIMD128 enabled by the build.
    unsafe {
        merged(
            data,
            edge,
            across,
            along,
            lines,
            segments,
            super::packed::simd128::filter,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dsp, Vp9FilterLimits, Vp9FilterWidth};
    use nc_common::PlaneViewMut;

    const SIDE: usize = 16;

    #[test]
    fn rejects_outside_or_overflowing_active_starts() {
        let dsp = Dsp::with_isa(Isa::Scalar).expect("scalar");
        let filter = Vp9EdgeFilter {
            width: Vp9FilterWidth::Sixteen,
            limits: Vp9FilterLimits::new(32),
        };
        for vertical in [false, true] {
            for (across, along) in [(0, 0), (14, 8), (8, SIDE), (usize::MAX, 8), (8, usize::MAX)] {
                let result = std::panic::catch_unwind(|| {
                    let mut pixels = [0; SIDE * SIDE];
                    let mut dst = PlaneViewMut::new(&mut pixels, SIDE, SIDE, SIDE).expect("plane");
                    let origin = if vertical {
                        (across, along)
                    } else {
                        (along, across)
                    };
                    dsp.vp9_loop_filter_line(&mut dst, origin, vertical, &[Some(filter)]);
                });
                assert!(result.is_err());
            }
        }
    }

    #[test]
    fn clipped_tail_and_masks_need_no_extra_storage() {
        let dsp = Dsp::with_isa(Isa::Scalar).expect("scalar");
        let filter = Vp9EdgeFilter {
            width: Vp9FilterWidth::Sixteen,
            limits: Vp9FilterLimits::new(32),
        };
        let mut pixels = [123; SIDE * SIDE];
        let mut dst = PlaneViewMut::new(&mut pixels, SIDE, SIDE, SIDE).expect("plane");
        let mut segments = [None; SIDE];
        segments[0] = Some(filter);
        dsp.vp9_loop_filter_line(&mut dst, (8, 15), true, &segments);
        dsp.vp9_loop_filter_line(&mut dst, (15, 8), false, &segments);
        dsp.vp9_loop_filter_line(&mut dst, (usize::MAX, usize::MAX), true, &[]);
        dsp.vp9_loop_filter_line(&mut dst, (usize::MAX, usize::MAX), true, &[None]);
        assert_eq!(pixels, [123; SIDE * SIDE]);
    }
}
