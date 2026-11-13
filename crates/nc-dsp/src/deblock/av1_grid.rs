//! AV1 edge grids: validate once, preserve across-edge order, and reuse the
//! line kernels for contiguous horizontal or strided vertical descriptors.

use super::av1::{AV1_EDGE_LINES, Av1EdgeFilter};
use core::ops::Range;

/// Caller supplies the picture storage described by `Grid` and enables the ISA.
pub(crate) type GridFn = unsafe fn(&mut [u8], &Grid<'_>);

/// Validated descriptors and plane geometry. Construction checks every active
/// footprint before a kernel may write, including partial final units.
pub(crate) struct Grid<'a> {
    width: usize,
    height: usize,
    plane_stride: usize,
    vertical: bool,
    rows: Range<usize>,
    columns: usize,
    segments: &'a [Option<Av1EdgeFilter>],
    stride: usize,
}

impl<'a> Grid<'a> {
    pub fn new(
        width: usize,
        height: usize,
        plane_stride: usize,
        vertical: bool,
        rows: Range<usize>,
        segments: &'a [Option<Av1EdgeFilter>],
        stride: usize,
    ) -> Self {
        let columns = width.div_ceil(AV1_EDGE_LINES);
        assert!(
            rows.start <= rows.end && rows.end <= height.div_ceil(AV1_EDGE_LINES),
            "AV1 grid rows outside picture"
        );
        assert!(stride >= columns, "AV1 grid stride shorter than picture");
        if !rows.is_empty() && columns != 0 {
            let end = (rows.end - 1)
                .checked_mul(stride)
                .and_then(|v| v.checked_add(columns))
                .expect("AV1 grid footprint");
            assert!(end <= segments.len(), "AV1 grid descriptors missing");
            for y in rows.clone() {
                for (x, filter) in segments[y * stride..][..columns].iter().enumerate() {
                    if let Some(filter) = filter {
                        let (position, limit) = if vertical {
                            (x * AV1_EDGE_LINES, width)
                        } else {
                            (y * AV1_EDGE_LINES, height)
                        };
                        let reach = filter.length.reach();
                        assert!(
                            position >= reach
                                && position.checked_add(reach).is_some_and(|end| end <= limit),
                            "AV1 grid filter footprint outside picture"
                        );
                    }
                }
            }
        }
        Self {
            width,
            height,
            plane_stride,
            vertical,
            rows,
            columns,
            segments,
            stride,
        }
    }
}

/// Row-major oracle. Each sample line observes increasing across-edge order;
/// changing that order could change overlapping filters on arbitrary grids.
pub(crate) fn scalar(plane: &mut [u8], grid: &Grid<'_>) {
    for y4 in grid.rows.clone() {
        for x4 in 0..grid.columns {
            if let Some(filter) = grid.segments[y4 * grid.stride + x4] {
                let (x, y) = (x4 * AV1_EDGE_LINES, y4 * AV1_EDGE_LINES);
                let (across, along, available) = if grid.vertical {
                    (1, grid.plane_stride, grid.height - y)
                } else {
                    (grid.plane_stride, 1, grid.width - x)
                };
                super::av1::filter_edge(
                    plane,
                    y * grid.plane_stride + x,
                    across,
                    along,
                    available.min(AV1_EDGE_LINES),
                    filter,
                );
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
unsafe fn packed<B: crate::lpf_bytes::Bytes>(plane: &mut [u8], grid: &Grid<'_>) {
    use super::line::Descriptors;
    if grid.rows.is_empty() || grid.columns == 0 {
        return;
    }
    if grid.vertical {
        let start = grid.rows.start * AV1_EDGE_LINES;
        let end = grid
            .rows
            .end
            .saturating_mul(AV1_EDGE_LINES)
            .min(grid.height);
        // Columns increase across the edge; rows within a column do not
        // interact. This is equivalent to the scalar row-major ordering.
        for x4 in 0..grid.columns {
            let descriptors = Descriptors {
                entries: &grid.segments[grid.rows.start * grid.stride + x4..],
                stride: grid.stride,
                count: grid.rows.len(),
            };
            // SAFETY: Grid::new validated every active footprint and the
            // descriptor strides; final picture rows are clipped by `end`.
            unsafe {
                super::av1_line::packed_descriptors::<B>(
                    plane,
                    start * grid.plane_stride + x4 * AV1_EDGE_LINES,
                    1,
                    grid.plane_stride,
                    end - start,
                    descriptors,
                );
            }
        }
    } else {
        for y4 in grid.rows.clone() {
            let descriptors = Descriptors {
                entries: &grid.segments[y4 * grid.stride..][..grid.columns],
                stride: 1,
                count: grid.columns,
            };
            // SAFETY: Grid::new validated all active edges on this row;
            // width clips the final unit, and rows retain increasing order.
            unsafe {
                super::av1_line::packed_descriptors::<B>(
                    plane,
                    y4 * AV1_EDGE_LINES * grid.plane_stride,
                    grid.plane_stride,
                    1,
                    grid.width,
                    descriptors,
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2(plane: &mut [u8], grid: &Grid<'_>) {
    // SAFETY: validated grid; AVX2 is enabled here.
    unsafe { packed::<crate::lpf_bytes::avx2::B>(plane, grid) }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon(plane: &mut [u8], grid: &Grid<'_>) {
    // SAFETY: validated grid; NEON is enabled here.
    unsafe { packed::<crate::lpf_bytes::neon::B>(plane, grid) }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) unsafe fn simd128(plane: &mut [u8], grid: &Grid<'_>) {
    // SAFETY: validated grid; SIMD128 is enabled by the build.
    unsafe { packed::<crate::lpf_bytes::simd128::B>(plane, grid) }
}

#[cfg(test)]
mod tests {
    use crate::{Av1EdgeFilter, Av1FilterLength, Dsp, Isa};
    use nc_common::PlaneViewMut;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    const SIDE: usize = 16;
    const UNITS: usize = SIDE / super::AV1_EDGE_LINES;
    const FILTER: Av1EdgeFilter = Av1EdgeFilter {
        length: Av1FilterLength::Four,
        limit: 255,
        blimit: 254,
        thresh: 255,
    };

    #[test]
    fn validates_all_edges_before_writing() {
        let dsp = Dsp::with_isa(Isa::Scalar).expect("scalar");
        let original: [u8; SIDE * SIDE] = core::array::from_fn(|i| 100 + (i % 7) as u8);
        for vertical in [false, true] {
            let mut pixels = original;
            let mut segments = [None; UNITS * UNITS];
            segments[UNITS + 1] = Some(FILTER);
            // This later entry has an invalid across-edge footprint.
            segments[if vertical {
                2 * UNITS
            } else {
                (UNITS - 1) * UNITS + 1
            }] = Some(Av1EdgeFilter {
                length: Av1FilterLength::Fourteen,
                ..FILTER
            });
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut dst = PlaneViewMut::new(&mut pixels, SIDE, SIDE, SIDE).expect("plane");
                dsp.av1_loop_filter_grid(&mut dst, vertical, 0..UNITS, &segments, UNITS);
            }));
            assert!(result.is_err());
            assert_eq!(pixels, original);
        }
    }

    #[test]
    fn rejects_bad_grid_storage_and_ranges() {
        let dsp = Dsp::with_isa(Isa::Scalar).expect("scalar");
        for (start, end, stride, len) in [
            (0, UNITS, UNITS - 1, 16),
            (0, UNITS, UNITS, 15),
            (0, 2, usize::MAX, 16),
            (2, 1, UNITS, 16),
            (0, usize::MAX, UNITS, 16),
        ] {
            let result = catch_unwind(|| {
                let mut pixels = [0; SIDE * SIDE];
                let mut dst = PlaneViewMut::new(&mut pixels, SIDE, SIDE, SIDE).expect("plane");
                dsp.av1_loop_filter_grid(&mut dst, true, start..end, &vec![None; len], stride);
            });
            assert!(result.is_err());
        }
    }

    #[test]
    fn empty_rows_need_no_descriptors() {
        let dsp = Dsp::with_isa(Isa::Scalar).expect("scalar");
        let mut pixels = [123; SIDE * SIDE];
        let mut dst = PlaneViewMut::new(&mut pixels, SIDE, SIDE, SIDE).expect("plane");
        dsp.av1_loop_filter_grid(&mut dst, true, UNITS..UNITS, &[], UNITS);
        assert_eq!(pixels, [123; SIDE * SIDE]);
    }
}
