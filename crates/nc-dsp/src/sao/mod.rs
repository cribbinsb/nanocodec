//! HEVC sample adaptive offset, edge offset (HEVC §8.7.3): the per-category
//! statistics an encoder needs to choose offsets, and applying them.
//!
//! Each sample `c` is classified against its two neighbours `a`, `b` along
//! the class direction: `edgeIdx = 2 + Sign(c − a) + Sign(c − b)` gives the
//! categories local minimum (1), concave edge (2), convex edge (3) and local
//! maximum (4), or none. Statistics are, per category, the sum of
//! `source − reconstruction` and the sample count; applying adds the
//! category's offset to the reconstruction and clamps to 8 bits.
//!
//! # Kernel contract
//!
//! Kernels are `unsafe fn`s called only from [`crate::Dsp`], which checks the
//! CPU and that the rectangle, and both neighbours of every sample in it,
//! lie inside the planes. Apply kernels use raw vector accesses only after
//! these checks; scalar tails and statistics retain slice-based access.

#[allow(unsafe_code)]
mod bytes;
#[allow(unsafe_code)]
pub(crate) mod generic;
pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx512;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

/// Neighbour displacements `(dx, dy)` of the two neighbours of each
/// edge-offset class (`sao_eo_class` 0..3: horizontal, vertical, 135°, 45°).
pub const SAO_EO_NEIGHBOURS: [[(isize, isize); 2]; 4] = [
    [(-1, 0), (1, 0)],
    [(0, -1), (0, 1)],
    [(-1, -1), (1, 1)],
    [(1, -1), (-1, 1)],
];

/// Per-category (1..4, indexed 0..3) sums of `source − reconstruction` and
/// sample counts of one edge-offset class over a rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SaoStats {
    /// Sum of `source − reconstruction` per category.
    pub sum: [i64; 4],
    /// Number of samples per category.
    pub count: [i64; 4],
}

/// A sample rectangle `x0..x1` x `y0..y1` of a plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rect {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    /// Row increment for statistics; application always processes every row.
    pub row_step: usize,
}

/// Statistics kernel: `rec` and `src` are whole planes; the neighbours of the
/// sample at index `i` of `rec` are at `i + off_a` and `i + off_b`.
pub(crate) type StatsFn = unsafe fn(
    src: &[u8],
    src_stride: usize,
    rec: &[u8],
    rec_stride: usize,
    offs: (isize, isize),
    rect: Rect,
) -> SaoStats;

/// Apply kernel: classify from `input` (the deblocked plane), write
/// `clamp(c + offsets[category])` to `out` over the rectangle (`offsets[0]`
/// for category 1).
pub(crate) type ApplyFn = unsafe fn(
    input: &[u8],
    in_stride: usize,
    out: &mut [u8],
    out_stride: usize,
    offs: (isize, isize),
    offsets: [i8; 4],
    rect: Rect,
);

pub(crate) type BandFn = unsafe fn(&mut [u8], usize, Rect, u8, [i8; 4], &[bool]);

/// The SAO kernels of one ISA.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SaoFns {
    pub stats: StatsFn,
    pub apply: ApplyFn,
    pub band: BandFn,
}

/// `edgeIdx − 2` (−2..2) to category index (0..3), `None` for no category.
pub(crate) fn category(edge: i32) -> Option<usize> {
    match edge {
        -2 => Some(0),
        -1 => Some(1),
        1 => Some(2),
        2 => Some(3),
        _ => None,
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2_bytes;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon_bytes;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128_bytes;

#[cfg(test)]
mod tests {
    #[test]
    fn band_offset_wraps_and_saturates_within_the_rectangle() {
        let mut samples = [99, 0, 7, 8, 15, 16, 23, 24, 248, 255, 99];
        let mut view = nc_common::PlaneViewMut::new(&mut samples, 11, 11, 1).unwrap();
        crate::Dsp::new().sao_band_apply(31, [7, -7, 3, -3], &mut view, (1, 0, 10, 1));
        assert_eq!(samples, [99, 0, 0, 11, 18, 13, 20, 24, 255, 255, 99]);
    }

    use crate::{Dsp, Isa};
    use nc_common::PlaneView;

    #[test]
    fn sampled_rows_match_weighted_scalar_rows() {
        let scalar = Dsp::with_isa(Isa::Scalar).unwrap();
        let best = Dsp::new();
        let w = 67;
        let h = 69;
        let source: Vec<u8> = (0..w * h).map(|i| ((i * 73 + 19) % 256) as u8).collect();
        let rec: Vec<u8> = (0..w * h)
            .map(|i| ((i * 41 + i / 13) % 256) as u8)
            .collect();
        let src = PlaneView::new(&source, w, w, h).unwrap();
        let rec = PlaneView::new(&rec, w, w, h).unwrap();
        for class in 0..4 {
            for step in 1..=4 {
                for end in [1, 2, 3, 4, 5, 33, 68] {
                    let mut want = super::SaoStats::default();
                    for y in (1..end).step_by(step) {
                        let row = scalar.sao_edge_stats(class, &src, &rec, (1, y, 66, y + 1));
                        let weight = step.min(end - y) as i64;
                        for c in 0..4 {
                            want.sum[c] += row.sum[c] * weight;
                            want.count[c] += row.count[c] * weight;
                        }
                    }
                    for dsp in [&scalar, &best] {
                        let got =
                            dsp.sao_edge_stats_sampled(class, &src, &rec, (1, 1, 66, end), step);
                        assert_eq!(got, want, "class{class} step{step} end{end}");
                    }
                }
            }
        }
    }
}
