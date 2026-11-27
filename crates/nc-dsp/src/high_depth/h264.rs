//! The H.264 kernels at any bit depth (High 10): luma and chroma motion
//! compensation (routed by [`super::Pixel::predict`] with the H.264
//! filters), the deblocking filter of a macroblock and the inverse
//! transforms added to the prediction, as the rest of [`DepthKernels`]
//! (module docs of `high_depth`).
//!
//! The deblocking methods take [`H264MbEdges`] with the 8-bit table values
//! of α, β and `tC0` (H.264 Tables 8-16, 8-17): the depth scales them by
//! `2^(BitDepth − 8)` (eqs. 8-463, 8-464, 8-466). The inverse transforms
//! take scaled coefficients in 32 bits: at 10 bits they reach `±2^17`
//! (§8.5.12.1), beyond the 16 bits of the 8-bit kernels.

use nc_common::{BitDepth, PlaneViewMut, Sample};

use super::{Convolve16Fn, DepthKernels, WIDTH_CLASSES};
use crate::deblock::scalar::{Line, chroma_line_at, luma_line_at};
use crate::deblock::{Edge, H264EdgeParams, H264MbEdges, h264_mb};
use crate::dispatch::{deblock_edge, h264_chroma_edge};
use crate::mc::Filter;

/// Luma reach of an edge (p3..q3) and lines per luma / chroma edge.
const LUMA_REACH: usize = 4;
/// Chroma reach (p1..q1).
const CHROMA_REACH: usize = 2;
const LUMA_LINES: usize = 16;
/// Sides of the inverse transforms.
const SIDE_4: usize = 4;
const SIDE_8: usize = 8;

/// An H.264 deblocking edge kernel at a depth.
pub(crate) type H264Edge16Fn = unsafe fn(&mut [u16], &Edge, BitDepth);

/// The `u16` H.264 kernels of one ISA.
#[derive(Debug, Clone, Copy)]
pub(crate) struct H264Fns {
    /// Uni-prediction, luma then chroma, by width class.
    pub predict: [[Convolve16Fn; WIDTH_CLASSES]; 2],
    pub deblock_luma: H264Edge16Fn,
    pub deblock_chroma: H264Edge16Fn,
}

/// The scalar `u16` instances of the MC references.
mod scalar {
    use nc_common::BitDepth;

    use crate::mc::scalar::{h264_chroma_at, h264_luma_at};

    pub(super) fn luma<const W: usize>(
        src: &[u16],
        ss: usize,
        dst: &mut [u16],
        ds: usize,
        h: usize,
        frac: (usize, usize),
        depth: BitDepth,
    ) {
        h264_luma_at::<u16, W>(src, ss, dst, ds, h, frac, depth);
    }

    pub(super) fn chroma<const W: usize>(
        src: &[u16],
        ss: usize,
        dst: &mut [u16],
        ds: usize,
        h: usize,
        frac: (usize, usize),
        depth: BitDepth,
    ) {
        h264_chroma_at::<u16, W>(src, ss, dst, ds, h, frac, depth);
    }
}

impl H264Fns {
    pub(crate) const SCALAR: Self = Self {
        predict: [
            [
                scalar::luma::<4>,
                scalar::luma::<8>,
                scalar::luma::<16>,
                scalar::luma::<32>,
                scalar::luma::<64>,
            ],
            [
                scalar::chroma::<4>,
                scalar::chroma::<8>,
                scalar::chroma::<16>,
                scalar::chroma::<32>,
                scalar::chroma::<64>,
            ],
        ],
        deblock_luma: crate::deblock::scalar::luma_edge_at::<u16>,
        deblock_chroma: crate::deblock::scalar::chroma_edge_at::<u16>,
    };

    /// The prediction kernel of an H.264 filter (`None` for the others).
    pub(crate) fn predict_of(&self, filter: Filter, width_class: usize) -> Option<Convolve16Fn> {
        match filter {
            Filter::H264Luma => Some(self.predict[0][width_class]),
            Filter::H264Chroma => Some(self.predict[1][width_class]),
            _ => None,
        }
    }
}

/// The routing of the H.264 kernels for one sample type: part of
/// [`super::Pixel`] (its supertrait), kept with its family.
#[doc(hidden)]
pub trait H264Pixel: Sample {
    fn h264_deblock_mb_luma(
        kernels: DepthKernels<'_>,
        plane: &mut PlaneViewMut<'_, Self>,
        at: (usize, usize),
        edges: &H264MbEdges,
    );
    fn h264_deblock_mb_chroma(
        kernels: DepthKernels<'_>,
        u: &mut PlaneViewMut<'_, Self>,
        v: &mut PlaneViewMut<'_, Self>,
        at: (usize, usize),
        edges: &H264MbEdges,
    );
    fn h264_deblock_mb_chroma_plane(
        kernels: DepthKernels<'_>,
        plane: &mut PlaneViewMut<'_, Self>,
        at: (usize, usize),
        edges: &H264MbEdges,
    );
}

impl H264Pixel for u8 {
    fn h264_deblock_mb_luma(
        kernels: DepthKernels<'_>,
        plane: &mut PlaneViewMut<'_, u8>,
        at: (usize, usize),
        edges: &H264MbEdges,
    ) {
        kernels.dsp.h264_deblock_mb_luma(plane, at, edges);
    }

    fn h264_deblock_mb_chroma(
        kernels: DepthKernels<'_>,
        u: &mut PlaneViewMut<'_, u8>,
        v: &mut PlaneViewMut<'_, u8>,
        at: (usize, usize),
        edges: &H264MbEdges,
    ) {
        kernels.dsp.h264_deblock_mb_chroma(u, v, at, edges);
    }

    fn h264_deblock_mb_chroma_plane(
        kernels: DepthKernels<'_>,
        plane: &mut PlaneViewMut<'_, u8>,
        at: (usize, usize),
        edges: &H264MbEdges,
    ) {
        kernels.dsp.h264_deblock_mb_chroma_plane(plane, at, edges);
    }
}

/// Scale an 8-bit edge's thresholds to `depth` (eqs. 8-463, 8-464, 8-466).
fn at_depth(mut edge: Edge, depth: BitDepth) -> Edge {
    let shift = depth.above_eight();
    edge.alpha <<= shift;
    edge.beta <<= shift;
    edge.tc0 = edge.tc0.map(|t| t << shift);
    edge
}

/// The position of edge `offset` samples into the macroblock at `(x, y)`.
fn edge_position((x, y): (usize, usize), vertical: bool, offset: usize) -> (usize, usize) {
    if vertical {
        (x + offset, y)
    } else {
        (x, y + offset)
    }
}

impl H264Pixel for u16 {
    fn h264_deblock_mb_luma(
        kernels: DepthKernels<'_>,
        plane: &mut PlaneViewMut<'_, u16>,
        at: (usize, usize),
        edges: &H264MbEdges,
    ) {
        let kernel = kernels.dsp.high.h264.deblock_luma;
        h264_mb::for_each_edge(edges, false, |vertical, offset, e| {
            let pos = edge_position(at, vertical, offset);
            let edge = deblock_edge(
                plane,
                pos,
                vertical,
                LUMA_REACH,
                LUMA_LINES,
                e.bs,
                e.tc0,
                (e.alpha, e.beta),
            );
            // SAFETY: `deblock_edge` checked that every sample the edge
            // reads lies inside `plane`; the table matches a supported ISA.
            unsafe {
                kernel(
                    plane.data_mut(),
                    &at_depth(edge, kernels.depth),
                    kernels.depth,
                );
            }
        });
    }

    fn h264_deblock_mb_chroma(
        kernels: DepthKernels<'_>,
        u: &mut PlaneViewMut<'_, u16>,
        v: &mut PlaneViewMut<'_, u16>,
        at: (usize, usize),
        edges: &H264MbEdges,
    ) {
        Self::h264_deblock_mb_chroma_plane(kernels, u, at, edges);
        Self::h264_deblock_mb_chroma_plane(kernels, v, at, edges);
    }

    fn h264_deblock_mb_chroma_plane(
        kernels: DepthKernels<'_>,
        plane: &mut PlaneViewMut<'_, u16>,
        at: (usize, usize),
        edges: &H264MbEdges,
    ) {
        let kernel = kernels.dsp.high.h264.deblock_chroma;
        h264_mb::for_each_edge(edges, true, |vertical, offset, e| {
            let pos = edge_position(at, vertical, offset);
            let edge = h264_chroma_edge(plane, pos, vertical, e);
            // SAFETY: as for the luma edges.
            unsafe {
                kernel(
                    plane.data_mut(),
                    &at_depth(edge, kernels.depth),
                    kernels.depth,
                );
            }
        });
    }
}

/// Add `residual` (`n`x`n`, row-major) to `dst` with `Clip1` to `depth`.
fn add_clipped<T: Sample>(
    residual: &[i32],
    n: usize,
    dst: &mut PlaneViewMut<'_, T>,
    depth: BitDepth,
) {
    assert!(
        dst.width() >= n && dst.height() >= n,
        "destination smaller than {n}x{n}"
    );
    for (row, line) in residual.chunks_exact(n).enumerate() {
        for (sample, &r) in dst.row_mut(row)[..n].iter_mut().zip(line) {
            *sample = depth.clip_to(sample.to_i32() + r);
        }
    }
}

impl DepthKernels<'_> {
    /// [`crate::Dsp::h264_deblock_mb_luma`] at this depth (`edges` with the
    /// 8-bit thresholds, module docs).
    ///
    /// # Panics
    /// As the 8-bit method.
    pub fn h264_deblock_mb_luma<T: super::Pixel>(
        self,
        plane: &mut PlaneViewMut<'_, T>,
        at: (usize, usize),
        edges: &H264MbEdges,
    ) {
        T::h264_deblock_mb_luma(self, plane, at, edges);
    }

    /// [`crate::Dsp::h264_deblock_mb_chroma`] at this depth.
    ///
    /// # Panics
    /// As the 8-bit method.
    pub fn h264_deblock_mb_chroma<T: super::Pixel>(
        self,
        u: &mut PlaneViewMut<'_, T>,
        v: &mut PlaneViewMut<'_, T>,
        at: (usize, usize),
        edges: &H264MbEdges,
    ) {
        T::h264_deblock_mb_chroma(self, u, v, at, edges);
    }

    /// [`crate::Dsp::h264_deblock_mb_chroma_plane`] at this depth.
    ///
    /// # Panics
    /// As the 8-bit method.
    pub fn h264_deblock_mb_chroma_plane<T: super::Pixel>(
        self,
        plane: &mut PlaneViewMut<'_, T>,
        at: (usize, usize),
        edges: &H264MbEdges,
    ) {
        T::h264_deblock_mb_chroma_plane(self, plane, at, edges);
    }

    /// Filter one line across an H.264 luma edge with the scalar reference
    /// (§8.7.2.3, §8.7.2.4): the line whose q0 sample is (`x`, `y`) of
    /// `plane`, across a `vertical` edge (p samples to its left) or a
    /// horizontal one (p samples above), with the bS and `tC0` of entry 0
    /// of `edge` and its 8-bit thresholds. For edges whose bS and
    /// thresholds change from line to line (H.264 MBAFF edges between frame
    /// and field macroblocks, §8.7.1).
    ///
    /// # Panics
    /// If a sample the line reads (p3..q3) lies outside `plane`, or bS
    /// exceeds 4.
    pub fn h264_deblock_luma_line<T: super::Pixel>(
        self,
        plane: &mut PlaneViewMut<'_, T>,
        at: (usize, usize),
        vertical: bool,
        edge: &H264EdgeParams,
    ) {
        let line = self.line(plane, at, vertical, LUMA_REACH, edge);
        luma_line_at(plane.data_mut(), &line, self.depth);
    }

    /// [`Self::h264_deblock_luma_line`] of a chroma edge (p1..q1).
    ///
    /// # Panics
    /// As the luma method.
    pub fn h264_deblock_chroma_line<T: super::Pixel>(
        self,
        plane: &mut PlaneViewMut<'_, T>,
        at: (usize, usize),
        vertical: bool,
        edge: &H264EdgeParams,
    ) {
        let line = self.line(plane, at, vertical, CHROMA_REACH, edge);
        chroma_line_at(plane.data_mut(), &line, self.depth);
    }

    /// One line of `edge` at `at` of `plane`, checked to lie inside it
    /// with `reach` samples each side, its thresholds at this depth.
    fn line<T: Sample>(
        self,
        plane: &PlaneViewMut<'_, T>,
        at: (usize, usize),
        vertical: bool,
        reach: usize,
        edge: &H264EdgeParams,
    ) -> Line {
        let checked = deblock_edge(
            plane,
            at,
            vertical,
            reach,
            1,
            edge.bs,
            edge.tc0,
            (edge.alpha, edge.beta),
        );
        let scaled = at_depth(checked, self.depth);
        Line {
            q0: scaled.base,
            across: scaled.across,
            bs: scaled.bs[0],
            tc0: scaled.tc0[0],
            alpha: scaled.alpha,
            beta: scaled.beta,
        }
    }

    /// The H.264 inverse 4x4 transform of scaled coefficients (raster
    /// order, §8.5.12.2 with `(x + 32) >> 6`) added to the 4x4 block of
    /// `dst` with `Clip1` to this depth.
    ///
    /// # Panics
    /// If `dst` is smaller than 4x4.
    pub fn h264_inverse_add_4x4<T: super::Pixel>(
        self,
        coeffs: &[i32; 16],
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        let mut residual = [[0; 16]];
        self.dsp
            .h264_inverse_4x4(core::slice::from_ref(coeffs), &mut residual);
        add_clipped(&residual[0], SIDE_4, dst, self.depth);
    }

    /// [`Self::h264_inverse_add_4x4`] of an 8x8 block (§8.5.13.2).
    ///
    /// # Panics
    /// If `dst` is smaller than 8x8.
    pub fn h264_inverse_add_8x8<T: super::Pixel>(
        self,
        coeffs: &[i32; 64],
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        let mut residual = [0; 64];
        self.dsp.h264_inverse_8x8(coeffs, &mut residual);
        add_clipped(&residual, SIDE_8, dst, self.depth);
    }
}
