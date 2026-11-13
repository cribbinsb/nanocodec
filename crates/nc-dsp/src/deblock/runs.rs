//! Sparse loop-filter runs shared across codecs. Validation travels with each
//! consumed run; there is no dense map or preliminary picture-wide sweep.

use super::av1::Av1EdgeFilter;
use super::{HevcEdge, HevcEdgeFn};

/// A contiguous run of lines along one deblocking edge. Runs are processed
/// in supplied order; callers retain codec-specific pass scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeRun<F> {
    /// Horizontal coordinate of the first q0 sample.
    pub x: usize,
    /// Vertical coordinate of the first q0 sample.
    pub y: usize,
    /// Number of sample lines along the edge; zero skips the run.
    pub lines: usize,
    /// Thresholds and filter shape shared throughout the run.
    pub filter: F,
}

/// HEVC luma thresholds shared by each four-line decision segment in a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HevcLumaFilter {
    /// Luma smoothness threshold β from the caller's QP tables.
    pub beta: u8,
    /// Clipping threshold tC; zero leaves the run unchanged.
    pub tc: u8,
}

/// HEVC chroma threshold shared by every line in a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HevcChromaFilter {
    /// Clipping threshold tC; zero leaves the run unchanged.
    pub tc: u8,
}

#[derive(Clone, Copy)]
pub(crate) struct Plane {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub vertical: bool,
}

impl Plane {
    fn validate<F>(&self, run: &EdgeRun<F>, reach: usize, unit: usize) {
        assert!(run.lines.is_multiple_of(unit), "loop-filter run length");
        let (across, limit, along, end) = if self.vertical {
            (run.x, self.width, run.y, self.height)
        } else {
            (run.y, self.height, run.x, self.width)
        };
        assert!(
            across >= reach && across.checked_add(reach).is_some_and(|v| v <= limit),
            "loop-filter run footprint"
        );
        assert!(
            along.checked_add(run.lines).is_some_and(|v| v <= end),
            "loop-filter run extent"
        );
    }

    fn address<F>(&self, run: &EdgeRun<F>) -> (usize, usize, usize) {
        let base = run
            .y
            .checked_mul(self.stride)
            .and_then(|v| v.checked_add(run.x))
            .expect("loop-filter run address");
        let (across, along) = if self.vertical {
            (1, self.stride)
        } else {
            (self.stride, 1)
        };
        (base, across, along)
    }
}

pub(crate) type Av1RunsFn = unsafe fn(&mut [u8], Plane, &[EdgeRun<Av1EdgeFilter>]);
pub(crate) type HevcLumaRunsFn = unsafe fn(&mut [u8], Plane, &[EdgeRun<HevcLumaFilter>]);
pub(crate) type HevcChromaRunsFn = unsafe fn(&mut [u8], Plane, &[EdgeRun<HevcChromaFilter>]);

/// A sparse traversal whose iterator body contains only scalar validation.
/// Keeping SIMD in the consuming loop preserves the caller's target features;
/// an apply closure previously outlined the intrinsics into scalar functions.
struct ValidatedRuns<'a, F, R> {
    plane: Plane,
    runs: &'a [EdgeRun<F>],
    merge: bool,
    unit: usize,
    reach: R,
    cursor: usize,
}

impl<'a, F, R> ValidatedRuns<'a, F, R> {
    #[inline]
    fn new(plane: Plane, runs: &'a [EdgeRun<F>], merge: bool, unit: usize, reach: R) -> Self {
        Self {
            plane,
            runs,
            merge,
            unit,
            reach,
            cursor: 0,
        }
    }
}

impl<F: Copy + Eq, R: Fn(F) -> usize> Iterator for ValidatedRuns<'_, F, R> {
    type Item = EdgeRun<F>;
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while self.cursor < self.runs.len() {
            let mut run = self.runs[self.cursor];
            self.cursor += 1;
            if run.lines == 0 {
                continue;
            }
            self.plane
                .validate(&run, (self.reach)(run.filter), self.unit);
            if self.merge {
                while let Some(next) = self.runs.get(self.cursor) {
                    let adjacent = if self.plane.vertical {
                        next.x == run.x && run.y.checked_add(run.lines) == Some(next.y)
                    } else {
                        next.y == run.y && run.x.checked_add(run.lines) == Some(next.x)
                    };
                    if next.lines == 0 || next.filter != run.filter || !adjacent {
                        break;
                    }
                    self.plane
                        .validate(next, (self.reach)(next.filter), self.unit);
                    run.lines = run
                        .lines
                        .checked_add(next.lines)
                        .expect("loop-filter merged run");
                    self.cursor += 1;
                }
            }
            return Some(run);
        }
        None
    }
}

pub(crate) fn av1_scalar(data: &mut [u8], plane: Plane, runs: &[EdgeRun<Av1EdgeFilter>]) {
    for run in ValidatedRuns::new(plane, runs, false, 1, |f: Av1EdgeFilter| f.length.reach()) {
        let (base, across, along) = plane.address(&run);
        super::av1::filter_edge(data, base, across, along, run.lines, run.filter);
    }
}

/// Select arithmetic width without moving SIMD into a closure lacking the
/// entry point's target features.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
trait Av1Kernel<B: crate::lpf_bytes::Bytes> {
    unsafe fn filter(
        data: &mut [u8],
        base: usize,
        across: usize,
        along: usize,
        lines: usize,
        filter: Av1EdgeFilter,
        tail: super::Av1EdgeFn,
    );
}
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
struct WideKernel<W>(core::marker::PhantomData<W>);
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
impl<B: crate::lpf_bytes::Bytes, W: crate::lpf_bytes::PairedBytes> Av1Kernel<B> for WideKernel<W> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    unsafe fn filter(
        data: &mut [u8],
        base: usize,
        across: usize,
        along: usize,
        lines: usize,
        filter: Av1EdgeFilter,
        tail: super::Av1EdgeFn,
    ) {
        // SAFETY: caller validates the footprint and enables both lane types' ISA.
        unsafe {
            super::av1_packed::filter_wide::<W, super::av1_tile::PairedTile<W>>(
                data, base, across, along, lines, filter, tail,
            );
        }
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn av1_packed<B: crate::lpf_bytes::Bytes, K: Av1Kernel<B>>(
    data: &mut [u8],
    plane: Plane,
    runs: &[EdgeRun<Av1EdgeFilter>],
    tail: super::Av1EdgeFn,
) {
    for run in ValidatedRuns::new(plane, runs, true, 1, |f: Av1EdgeFilter| f.length.reach()) {
        let (base, across, along) = plane.address(&run);
        let packed = run.lines / 4 * 4;
        if packed != 0 {
            // SAFETY: the iterator validated the complete run and caller enables B's ISA.
            unsafe {
                K::filter(data, base, across, along, packed, run.filter, tail);
            }
        }
        if packed != run.lines {
            super::av1::filter_edge(
                data,
                base + packed * along,
                across,
                along,
                run.lines - packed,
                run.filter,
            );
        }
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn hevc_luma(
    data: &mut [u8],
    plane: Plane,
    runs: &[EdgeRun<HevcLumaFilter>],
    merge: bool,
    kernel: HevcEdgeFn,
) {
    for run in ValidatedRuns::new(plane, runs, merge, 4, |_| 4) {
        let (base, across, along) = plane.address(&run);
        for start in (0..run.lines).step_by(8) {
            let lines = (run.lines - start).min(8);
            let tc = i16::from(run.filter.tc);
            let edge = HevcEdge {
                base: base + start * along,
                across,
                along,
                lines,
                tc: [tc, if lines == 8 { tc } else { 0 }, 0, 0],
                beta: i16::from(run.filter.beta),
            };
            // SAFETY: the iterator validated all four/eight lines; caller enables ISA.
            unsafe {
                kernel(data, &edge);
            }
        }
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn hevc_chroma(
    data: &mut [u8],
    plane: Plane,
    runs: &[EdgeRun<HevcChromaFilter>],
    merge: bool,
    kernel: HevcEdgeFn,
) {
    for run in ValidatedRuns::new(plane, runs, merge, 2, |_| 4) {
        let (base, across, along) = plane.address(&run);
        // Four/eight-line SIMD calls, with the uncommon two-line tail using
        // the independent scalar equation until a measured short kernel exists.
        let mut start = 0;
        while start < run.lines {
            let remaining = run.lines - start;
            let lines = if remaining >= 8 {
                8
            } else if remaining >= 4 {
                4
            } else {
                2
            };
            let edge = HevcEdge {
                base: base + start * along,
                across,
                along,
                lines,
                tc: [i16::from(run.filter.tc); 4],
                beta: 0,
            };
            if lines == 2 {
                super::scalar::hevc_chroma_edge(data, &edge);
            } else {
                // SAFETY: the iterator validated all four/eight lines; caller enables ISA.
                unsafe {
                    kernel(data, &edge);
                }
            }
            start += lines;
        }
    }
}

pub(crate) fn hevc_luma_scalar(data: &mut [u8], plane: Plane, runs: &[EdgeRun<HevcLumaFilter>]) {
    // SAFETY: scalar has no ISA requirement; traversal validates each run.
    unsafe {
        hevc_luma(data, plane, runs, false, super::scalar::hevc_luma_edge);
    }
}
pub(crate) fn hevc_chroma_scalar(
    data: &mut [u8],
    plane: Plane,
    runs: &[EdgeRun<HevcChromaFilter>],
) {
    // SAFETY: scalar has no ISA requirement; traversal validates each run.
    unsafe {
        hevc_chroma(data, plane, runs, false, super::scalar::hevc_chroma_edge);
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! isa_runs {
    ($module:ident, $feature:literal, $bytes:path, $kernel:ty) => {
        pub(crate) mod $module {
            use super::*;
            #[target_feature(enable=$feature)]
            pub(crate) unsafe fn av1(
                data: &mut [u8],
                plane: Plane,
                runs: &[EdgeRun<Av1EdgeFilter>],
            ) {
                // SAFETY: this wrapper enables the ISA; traversal validates each run.
                unsafe {
                    av1_packed::<$bytes, $kernel>(
                        data,
                        plane,
                        runs,
                        super::super::$module::av1_edge,
                    );
                }
            }
            #[target_feature(enable=$feature)]
            pub(crate) unsafe fn luma(
                data: &mut [u8],
                plane: Plane,
                runs: &[EdgeRun<HevcLumaFilter>],
            ) {
                // SAFETY: this wrapper enables the ISA; traversal validates each run.
                unsafe {
                    hevc_luma(data, plane, runs, true, super::super::$module::hevc_luma);
                }
            }
            #[target_feature(enable=$feature)]
            pub(crate) unsafe fn chroma(
                data: &mut [u8],
                plane: Plane,
                runs: &[EdgeRun<HevcChromaFilter>],
            ) {
                // SAFETY: this wrapper enables the ISA; traversal validates each run.
                unsafe {
                    hevc_chroma(data, plane, runs, true, super::super::$module::hevc_chroma);
                }
            }
        }
    };
}
#[cfg(target_arch = "x86_64")]
isa_runs!(
    avx2,
    "avx2",
    crate::lpf_bytes::avx2::B,
    WideKernel<crate::lpf_bytes::avx2_wide::B>
);
#[cfg(target_arch = "aarch64")]
isa_runs!(
    neon,
    "neon",
    crate::lpf_bytes::neon::B,
    WideKernel<crate::lpf_bytes::neon_wide::B>
);
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
isa_runs!(
    simd128,
    "simd128",
    crate::lpf_bytes::simd128::B,
    WideKernel<crate::lpf_bytes::simd128_wide::B>
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Av1FilterLength, Dsp};
    use nc_common::PlaneViewMut;

    #[test]
    fn empty_runs_ignore_coordinates_and_invalid_runs_panic_safely() {
        let filter = Av1EdgeFilter {
            length: Av1FilterLength::Four,
            limit: 32,
            blimit: 64,
            thresh: 0,
        };
        for dsp in [
            Dsp::with_isa(crate::Isa::Scalar).expect("scalar"),
            Dsp::default(),
        ] {
            let mut data = [128u8; 64];
            let mut view = PlaneViewMut::new(&mut data, 8, 8, 8).expect("plane");
            dsp.av1_loop_filter_runs(
                &mut view,
                true,
                &[EdgeRun {
                    x: usize::MAX,
                    y: usize::MAX,
                    lines: 0,
                    filter,
                }],
            );
            for (x, y, lines) in [
                (usize::MAX, 0, 4),
                (4, usize::MAX, 4),
                (4, 0, usize::MAX),
                (0, 0, 4),
                (4, 7, 4),
            ] {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    dsp.av1_loop_filter_runs(
                        &mut view,
                        true,
                        &[EdgeRun {
                            x,
                            y,
                            lines,
                            filter,
                        }],
                    );
                }));
                assert!(result.is_err());
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                dsp.hevc_deblock_luma_runs(
                    &mut view,
                    true,
                    &[EdgeRun {
                        x: 4,
                        y: 0,
                        lines: 3,
                        filter: HevcLumaFilter { beta: 32, tc: 3 },
                    }],
                );
            }));
            assert!(result.is_err());
        }
    }
}
