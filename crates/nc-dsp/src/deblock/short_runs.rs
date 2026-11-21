//! Ordered HEVC/VVC short-luma tile gathering. Both codecs share the same
//! endpoint decisions and arithmetic; their strong clipping limits differ.
//! Only consecutive independent footprints widen. VVC also batches long
//! tiles of the same reach. Partial and narrow filters retain their order.
use super::runs::{
    EdgeRun, HevcLumaFilter, Plane, ValidatedRuns, VvcLumaFilter, vvc_edge, vvc_reach,
};
use super::{HevcEdge, HevcEdgeFn};

/// Split validated runs into eight-line pieces without changing order.
pub(super) struct Tiles<I, F> {
    source: I,
    current: Option<EdgeRun<F>>,
    offset: usize,
    vertical: bool,
}
impl<I, F: Copy> Tiles<I, F> {
    pub(super) fn new(source: I, vertical: bool) -> Self {
        Self {
            source,
            current: None,
            offset: 0,
            vertical,
        }
    }
}
impl<I: Iterator<Item = EdgeRun<F>>, F: Copy> Iterator for Tiles<I, F> {
    type Item = EdgeRun<F>;
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_none() {
            self.current = self.source.next();
            self.offset = 0;
        }
        let run = self.current?;
        let lines = (run.lines - self.offset).min(8);
        let out = EdgeRun {
            x: run.x + if self.vertical { 0 } else { self.offset },
            y: run.y + if self.vertical { self.offset } else { 0 },
            lines,
            filter: run.filter,
        };
        self.offset += lines;
        if self.offset == run.lines {
            self.current = None;
        }
        Some(out)
    }
}

#[cfg(target_arch = "x86_64")]
use crate::simd::avx2::I16x8 as Narrow;
#[cfg(target_arch = "aarch64")]
use crate::simd::neon::I16x8 as Narrow;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::simd::simd128::I16x8 as Narrow;

/// Codec thresholds and the inherited filter for non-short/partial tiles.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
trait ShortFilter: Copy + Eq {
    type Tail: Copy;
    fn reach(self) -> Option<[u8; 2]>;
    fn short(self) -> bool;
    fn thresholds(self) -> (i16, i16);
    unsafe fn tail(data: &mut [u8], plane: Plane, run: &EdgeRun<Self>, kernel: Self::Tail);
}
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(clippy::inline_always)]
impl ShortFilter for HevcLumaFilter {
    type Tail = HevcEdgeFn;
    #[inline(always)]
    fn reach(self) -> Option<[u8; 2]> {
        Some([3; 2])
    }
    #[inline(always)]
    fn short(self) -> bool {
        true
    }
    #[inline(always)]
    fn thresholds(self) -> (i16, i16) {
        (i16::from(self.beta), i16::from(self.tc))
    }
    #[inline(always)]
    unsafe fn tail(data: &mut [u8], plane: Plane, run: &EdgeRun<Self>, kernel: Self::Tail) {
        let (base, across, along) = plane.address(run);
        let tc = i16::from(run.filter.tc);
        let edge = HevcEdge {
            base,
            across,
            along,
            lines: run.lines,
            tc: [tc, if run.lines == 8 { tc } else { 0 }, 0, 0],
            beta: i16::from(run.filter.beta),
        };
        // SAFETY: the driver validated this footprint and enabled the ISA.
        unsafe {
            kernel(data, &edge);
        }
    }
}
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(clippy::inline_always)]
impl ShortFilter for VvcLumaFilter {
    type Tail = crate::vvc_deblock::Kernels;
    #[inline(always)]
    fn reach(self) -> Option<[u8; 2]> {
        (self.lengths[0] >= 3 && self.lengths[1] >= 3).then_some(self.lengths)
    }
    #[inline(always)]
    fn short(self) -> bool {
        self.lengths == [3; 2]
    }
    #[inline(always)]
    fn thresholds(self) -> (i16, i16) {
        (i16::from(self.beta), i16::from(self.tc))
    }
    #[inline(always)]
    unsafe fn tail(data: &mut [u8], plane: Plane, run: &EdgeRun<Self>, kernels: Self::Tail) {
        let edge = vvc_edge(plane, run);
        let kernel = if run.lines == 4 {
            kernels.luma_segment
        } else if run.filter.lengths.iter().all(|n| matches!(n, 3 | 7)) {
            kernels.luma_edge
        } else {
            kernels.luma_extended
        };
        // SAFETY: the driver validated this footprint and enabled the ISA.
        unsafe {
            kernel(data, &edge);
        }
    }
}

/// Batch consecutive independent tiles of the same shape.
/// Narrow, partial and overlapping tiles remain ordering fences.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn short_luma_wide<V: crate::simd::I16x8Tiles, F: ShortFilter, const VVC: bool>(
    data: &mut [u8],
    plane: Plane,
    mut source: impl Iterator<Item = EdgeRun<F>>,
    tail: F::Tail,
) {
    let mut pending = source.next();
    while let Some(first) = pending.take() {
        pending = source.next();
        let Some(reach) = first.filter.reach() else {
            // SAFETY: the driver validated this narrow tile and enabled its ISA.
            unsafe { F::tail(data, plane, &first, tail) };
            continue;
        };
        let short = first.filter.short();
        if first.lines != 8 || (!short && V::LANES == 8) {
            // SAFETY: validation covers the inherited kernel's footprint;
            // dispatch enabled its ISA. Long and partial tiles stay ordered.
            unsafe {
                F::tail(data, plane, &first, tail);
            }
            continue;
        }
        let (base, across, along) = plane.address(&first);
        let initial = super::batch::Geometry {
            base,
            across,
            along,
        };
        if V::LANES == 8 {
            let (beta, tc) = first.filter.thresholds();
            // SAFETY: one validated complete tile; the caller enables V's ISA.
            // Narrow tiers need no multi-tile arrays or repeated threshold shuffles.
            unsafe {
                super::batch::short::<V, VVC>(data, initial, beta, tc);
            }
            continue;
        }
        // Long taps read sixteen positions across the edge. The along
        // extent is eight in either case. No batch may cross a dependency.
        let across_span = if short { 8 } else { 16 };
        let (width, height) = if plane.vertical {
            (across_span, 8)
        } else {
            (8, across_span)
        };
        let mut tiles = [initial; 4];
        let mut positions = [(first.x, first.y); 4];
        let mut count = 1;
        let (beta, tc) = first.filter.thresholds();
        let mut betas = [beta; 8];
        let mut tcs = [tc; 8];
        while count < V::LANES / 8 {
            let Some(next) = pending.as_ref() else {
                break;
            };
            if next.lines != 8
                || next.filter.reach() != Some(reach)
                || positions[..count]
                    .iter()
                    .any(|&(x, y)| x.abs_diff(next.x) < width && y.abs_diff(next.y) < height)
            {
                break;
            }
            let next = pending.take().expect("pending tile");
            pending = source.next();
            let (base, across, along) = plane.address(&next);
            tiles[count] = super::batch::Geometry {
                base,
                across,
                along,
            };
            positions[count] = (next.x, next.y);
            let (beta, tc) = next.filter.thresholds();
            betas[2 * count..2 * count + 2].fill(beta);
            tcs[2 * count..2 * count + 2].fill(tc);
            count += 1;
        }
        // SAFETY: each tile is validated, and the footprint test above
        // proves the tiles are disjoint.
        unsafe {
            if count == V::LANES / 8 {
                filter_tiles::<V, VVC>(data, &tiles, &betas, &tcs, reach);
            } else {
                #[cfg(target_arch = "x86_64")]
                if count >= 2 {
                    filter_tiles::<crate::simd::avx2::I16x16, VVC>(
                        data, &tiles, &betas, &tcs, reach,
                    );
                    if count == 3 {
                        filter_tiles::<Narrow, VVC>(
                            data,
                            &[tiles[2]; 4],
                            &[betas[4]; 8],
                            &[tcs[4]; 8],
                            reach,
                        );
                    }
                    continue;
                }
                if short {
                    super::batch::short::<Narrow, VVC>(data, tiles[0], beta, tc);
                } else {
                    F::tail(data, plane, &first, tail);
                }
            }
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
unsafe fn filter_tiles<V: crate::simd::I16x8Tiles, const VVC: bool>(
    data: &mut [u8],
    tiles: &[super::batch::Geometry; 4],
    beta: &[i16; 8],
    tc: &[i16; 8],
    reach: [u8; 2],
) {
    // SAFETY: the traversal validated each tile, its disjointness and
    // the ISA. Every tile shares the same P/Q reach.
    unsafe {
        if VVC && reach != [3; 2] {
            crate::vvc_deblock::long_luma::tiles::<V>(
                data,
                tiles,
                V::splat_segments(beta),
                V::splat_segments(tc),
                reach.map(usize::from),
            );
        } else {
            super::batch::short_tiles::<V, VVC>(data, tiles, beta, tc);
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
pub(super) unsafe fn vvc_luma_wide<V: crate::simd::I16x8Tiles>(
    data: &mut [u8],
    plane: Plane,
    runs: &[EdgeRun<VvcLumaFilter>],
    tail: crate::vvc_deblock::Kernels,
) {
    let valid = ValidatedRuns::new(plane, runs, true, 4, vvc_reach);
    let source = Tiles::new(valid, plane.vertical);
    // SAFETY: forwarded enabled ISA; iterator validates every footprint.
    unsafe {
        short_luma_wide::<V, _, true>(data, plane, source, tail);
    }
}
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(clippy::inline_always)]
#[inline(always)]
pub(super) unsafe fn hevc_luma_wide<V: crate::simd::I16x8Tiles>(
    data: &mut [u8],
    plane: Plane,
    runs: &[EdgeRun<HevcLumaFilter>],
    tail: HevcEdgeFn,
) {
    let valid = ValidatedRuns::new(plane, runs, false, 4, |_| 4);
    let source = Tiles::new(valid, plane.vertical);
    // SAFETY: forwarded enabled ISA; iterator validates every footprint.
    unsafe {
        short_luma_wide::<V, _, false>(data, plane, source, tail);
    }
}
