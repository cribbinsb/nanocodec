//! The VVC in-loop filters at any bit depth: deblocking (H.266 §8.8.3)
//! and the adaptive loop filters (ALF, CC-ALF; §8.8.5), routed as the rest
//! of [`DepthKernels`] (module docs of `high_depth`).
//!
//! Each [`DepthKernels`] method has the name and argument order of the
//! 8-bit [`Dsp`] method, with samples of `T`. Parameters whose range grows
//! with the depth keep their (already wide) types and take the values at
//! the depth:
//!
//! - deblocking `beta`, `tc` (`i32`): `β` and `tC` at the depth, from the
//!   Table 43 values by [`crate::vvc_deblock_beta`] and
//!   [`crate::vvc_deblock_tc`] (H.266 §8.8.3.6.2);
//! - ALF `limits` (`i16`): `AlfClip` at the depth,
//!   [`crate::vvc_alf_clip_value`] (§7.4.3.18);
//! - ALF classification: the activity shift `BitDepth + 4` (§8.8.5.3) is
//!   the kernels' own; CC-ALF clips its correction to `±2^(BitDepth − 1)`
//!   (§8.8.5.7).
//!
//! The ALF kernels take at most [`crate::VVC_ALF_MAX_BIT_DEPTH`] bits.

use nc_common::{BitDepth, PlaneViewMut, Sample};

use super::DepthKernels;
use crate::dispatch::{vvc_chroma_edge_at, vvc_luma_edge_at};
use crate::vvc_alf::{self, VVC_ALF_BLOCK, VVC_ALF_TAPS};
use crate::vvc_deblock::{self, ChromaEdge, VvcLumaEdge};

/// Luma lines of one deblocking segment (one decision).
const LUMA_SEGMENT_LINES: usize = 4;

/// In-place luma deblocking of a [`VvcLumaEdge`] at a depth.
pub(crate) type DeblockLumaEdge16Fn = unsafe fn(&mut [u16], &VvcLumaEdge, BitDepth);
/// Luma deblocking of four gathered lines (`β`, `tC`, reaches, depth).
pub(crate) type DeblockLuma16Fn = unsafe fn(&mut [[u16; 16]; 4], i32, i32, [usize; 2], BitDepth);
/// In-place chroma deblocking of a [`ChromaEdge`] at a depth.
pub(crate) type DeblockChromaEdge16Fn = unsafe fn(&mut [u16], &ChromaEdge, BitDepth);
/// Chroma deblocking of two gathered lines (`β`, `tC`, large, CTU
/// boundary, depth).
pub(crate) type DeblockChroma16Fn = unsafe fn(&mut [[u16; 8]; 2], i32, i32, bool, bool, BitDepth);
/// One block row of the luma ALF (`crate::vvc_alf::FilterBlockRowFn` at a
/// depth).
pub(crate) type AlfBlockRow16Fn = unsafe fn(
    &[[&[u16]; 7]; VVC_ALF_BLOCK],
    &[[i16; VVC_ALF_TAPS]],
    &[u32; VVC_ALF_BLOCK],
    &mut [&mut [u16]; VVC_ALF_BLOCK],
    BitDepth,
);
/// The Laplacian group sums of a grid row (`crate::vvc_alf::LaplacianFn`
/// at `u16`; the sums are exact, so no depth).
pub(crate) type AlfLaplacians16Fn = unsafe fn(&[&[u16]; 4], &mut [[u16; 4]]);
/// ALF classes of a row of 4x4 blocks (`crate::vvc_alf::ClassifyFn` at a
/// depth).
pub(crate) type AlfClassifyFn = unsafe fn(&[[u16; 4]], bool, &mut [[u8; 2]], BitDepth);
/// One clipped luma ALF block row (4 rows, coefficients, `AlfClip` limits,
/// shifts, depth).
pub(crate) type AlfClippedBlockRow16Fn = unsafe fn(
    &[[&[u16]; 7]; VVC_ALF_BLOCK],
    &[[i16; VVC_ALF_TAPS]],
    &[[i16; VVC_ALF_TAPS]],
    &[u32; VVC_ALF_BLOCK],
    &mut [&mut [u16]; VVC_ALF_BLOCK],
    BitDepth,
);
/// One clipped luma ALF row (coefficients, `AlfClip` limits, shift, depth).
pub(crate) type AlfClippedRow16Fn = unsafe fn(
    &[&[u16]; 7],
    &[[i16; VVC_ALF_TAPS]],
    &[[i16; VVC_ALF_TAPS]],
    u32,
    &mut [u16],
    BitDepth,
);
/// One chroma ALF row (coefficients, `AlfClip` limits, shift, depth).
pub(crate) type AlfChromaRow16Fn =
    unsafe fn(&[&[u16]; 7], [i16; 6], [i16; 6], u32, &mut [u16], BitDepth);
/// One CC-ALF chroma row from four luma rows at a depth.
pub(crate) type AlfCrossRow16Fn = unsafe fn(&[&[u16]; 4], [i16; 7], &mut [u16], BitDepth);

/// The `u16` VVC filter kernels of one ISA (part of [`super::HighFns`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct VvcFilterFns {
    pub deblock_luma_edge: DeblockLumaEdge16Fn,
    pub deblock_luma: DeblockLuma16Fn,
    pub deblock_chroma_edge: DeblockChromaEdge16Fn,
    pub deblock_chroma: DeblockChroma16Fn,
    pub alf_filter_block_row: AlfBlockRow16Fn,
    pub alf_laplacians: AlfLaplacians16Fn,
    pub alf_classify: AlfClassifyFn,
    pub alf_clipped_block_row: AlfClippedBlockRow16Fn,
    pub alf_clipped_row: AlfClippedRow16Fn,
    pub alf_chroma_row: AlfChromaRow16Fn,
    pub alf_cross_row: AlfCrossRow16Fn,
}

impl VvcFilterFns {
    /// The scalar references at `u16`.
    pub(crate) const SCALAR: Self = Self {
        deblock_luma_edge: vvc_deblock::luma_edge_at::<u16>,
        deblock_luma: vvc_deblock::luma_at::<u16>,
        deblock_chroma_edge: vvc_deblock::chroma_edge_at::<u16>,
        deblock_chroma: vvc_deblock::chroma_at::<u16>,
        alf_filter_block_row: vvc_alf::filter_block_row_at::<u16>,
        alf_laplacians: vvc_alf::laplacians_at::<u16>,
        alf_classify: vvc_alf::classify_at,
        alf_clipped_block_row: vvc_alf::clipped_block_row_at::<u16>,
        alf_clipped_row: vvc_alf::clipped_row_at::<u16>,
        alf_chroma_row: vvc_alf::chroma_row_at::<u16>,
        alf_cross_row: vvc_alf::cross_row_at::<u16>,
    };

    /// The `u16` VVC filter table for `isa`.
    pub(crate) fn for_isa(isa: crate::Isa) -> Self {
        let mut fns = Self::SCALAR;
        let (deblock_luma_edge, deblock_luma, deblock_chroma_edge, deblock_chroma) =
            vvc_deblock::kernels_u16(isa);
        fns.deblock_luma_edge = deblock_luma_edge;
        fns.deblock_luma = deblock_luma;
        fns.deblock_chroma_edge = deblock_chroma_edge;
        fns.deblock_chroma = deblock_chroma;
        match isa {
            #[cfg(target_arch = "x86_64")]
            crate::Isa::Avx2 => {
                fns.alf_filter_block_row = vvc_alf::high_simd::avx2::filter_block_row;
                fns.alf_laplacians = vvc_alf::high_classify::avx2::laplacians;
                fns.alf_classify = vvc_alf::high_classify::avx2::classify;
                fns.alf_clipped_block_row = vvc_alf::high_simd::avx2::clipped_block_row;
                fns.alf_clipped_row = vvc_alf::high_simd::avx2::clipped_row;
                fns.alf_chroma_row = vvc_alf::high_simd::avx2::chroma_row;
                fns.alf_cross_row = vvc_alf::high_simd::avx2::cross_row;
            }
            #[cfg(target_arch = "x86_64")]
            crate::Isa::Avx512 => {
                fns.alf_filter_block_row = vvc_alf::high_simd::avx512::filter_block_row;
                fns.alf_laplacians = vvc_alf::high_classify::avx512::laplacians;
                fns.alf_classify = vvc_alf::high_classify::avx512::classify;
                fns.alf_clipped_block_row = vvc_alf::high_simd::avx512::clipped_block_row;
                fns.alf_clipped_row = vvc_alf::high_simd::avx512::clipped_row;
                fns.alf_chroma_row = vvc_alf::high_simd::avx512::chroma_row;
                fns.alf_cross_row = vvc_alf::high_simd::avx512::cross_row;
            }
            #[cfg(target_arch = "aarch64")]
            crate::Isa::Neon => {
                fns.alf_filter_block_row = vvc_alf::high_simd::neon::filter_block_row;
                fns.alf_laplacians = vvc_alf::high_classify::neon::laplacians;
                fns.alf_classify = vvc_alf::high_classify::neon::classify;
                fns.alf_clipped_block_row = vvc_alf::high_simd::neon::clipped_block_row;
                fns.alf_clipped_row = vvc_alf::high_simd::neon::clipped_row;
                fns.alf_chroma_row = vvc_alf::high_simd::neon::chroma_row;
                fns.alf_cross_row = vvc_alf::high_simd::neon::cross_row;
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            crate::Isa::Simd128 => {
                fns.alf_filter_block_row = vvc_alf::high_simd::simd128::filter_block_row;
                fns.alf_laplacians = vvc_alf::high_classify::simd128::laplacians;
                fns.alf_classify = vvc_alf::high_classify::simd128::classify;
                fns.alf_clipped_block_row = vvc_alf::high_simd::simd128::clipped_block_row;
                fns.alf_clipped_row = vvc_alf::high_simd::simd128::clipped_row;
                fns.alf_chroma_row = vvc_alf::high_simd::simd128::chroma_row;
                fns.alf_cross_row = vvc_alf::high_simd::simd128::cross_row;
            }
            _ => {}
        }
        fns
    }
}

/// Check `β` and `tC` against their largest values at `depth`.
fn check_thresholds(beta: i32, tc: i32, depth: BitDepth) {
    assert!(
        vvc_deblock::thresholds_fit(beta, tc, depth),
        "VVC deblocking beta {beta} / tc {tc} out of range at {depth:?}"
    );
}

/// The routing of the VVC filter kernels for one sample type: part of
/// [`super::Pixel`] (its supertrait), kept with its family.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub trait VvcFilterPixel: Sample {
    fn vvc_deblock_luma_edge(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, Self>,
        at: (usize, usize),
        vertical: bool,
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    );
    fn vvc_deblock_luma_segment(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, Self>,
        at: (usize, usize),
        vertical: bool,
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    );
    fn vvc_deblock_luma(
        kernels: DepthKernels<'_>,
        edge: &mut [[Self; 16]; 4],
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    );
    fn vvc_deblock_chroma_edge(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, Self>,
        at: (usize, usize),
        vertical: bool,
        lines: usize,
        beta: i32,
        tc: i32,
        large: bool,
        boundary: bool,
    );
    fn vvc_deblock_chroma(
        kernels: DepthKernels<'_>,
        edge: &mut [[Self; 8]; 2],
        beta: i32,
        tc: i32,
        large: bool,
        boundary: bool,
    );
    fn vvc_alf_filter_block_row(
        kernels: DepthKernels<'_>,
        rows: &[[&[Self]; 7]; VVC_ALF_BLOCK],
        coefficients: &[[i16; VVC_ALF_TAPS]],
        shifts: &[u32; VVC_ALF_BLOCK],
        out: &mut [&mut [Self]; VVC_ALF_BLOCK],
    );
    fn vvc_alf_laplacians(kernels: DepthKernels<'_>, rows: &[&[Self]; 4], out: &mut [[u16; 4]]);
    fn vvc_alf_classify(
        kernels: DepthKernels<'_>,
        groups: &[[u16; 4]],
        boundary: bool,
        out: &mut [[u8; 2]],
    );
    fn vvc_alf_filter_clipped_block_row(
        kernels: DepthKernels<'_>,
        rows: &[[&[Self]; 7]; VVC_ALF_BLOCK],
        coefficients: &[[i16; VVC_ALF_TAPS]],
        limits: &[[i16; VVC_ALF_TAPS]],
        shifts: &[u32; VVC_ALF_BLOCK],
        out: &mut [&mut [Self]; VVC_ALF_BLOCK],
    );
    fn vvc_alf_filter_clipped_row(
        kernels: DepthKernels<'_>,
        rows: &[&[Self]; 7],
        coefficients: &[[i16; VVC_ALF_TAPS]],
        limits: &[[i16; VVC_ALF_TAPS]],
        shift: u32,
        out: &mut [Self],
    );
    fn vvc_alf_filter_chroma_row(
        kernels: DepthKernels<'_>,
        rows: &[&[Self]; 7],
        coefficients: [i16; 6],
        limits: [i16; 6],
        shift: u32,
        out: &mut [Self],
    );
    fn vvc_alf_cross_row(
        kernels: DepthKernels<'_>,
        rows: &[&[Self]; 4],
        coefficients: [i16; 7],
        out: &mut [Self],
    );
}

/// `u8`: the 8-bit [`crate::Dsp`] methods with their SIMD tiers.
impl VvcFilterPixel for u8 {
    fn vvc_deblock_luma_edge(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u8>,
        at: (usize, usize),
        vertical: bool,
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    ) {
        kernels
            .dsp
            .vvc_deblock_luma_edge(dst, at, vertical, beta, tc, lengths);
    }

    fn vvc_deblock_luma_segment(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u8>,
        at: (usize, usize),
        vertical: bool,
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    ) {
        kernels
            .dsp
            .vvc_deblock_luma_segment(dst, at, vertical, beta, tc, lengths);
    }

    fn vvc_deblock_luma(
        kernels: DepthKernels<'_>,
        edge: &mut [[u8; 16]; 4],
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    ) {
        kernels.dsp.vvc_deblock_luma(edge, beta, tc, lengths);
    }

    fn vvc_deblock_chroma_edge(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u8>,
        at: (usize, usize),
        vertical: bool,
        lines: usize,
        beta: i32,
        tc: i32,
        large: bool,
        boundary: bool,
    ) {
        kernels
            .dsp
            .vvc_deblock_chroma_edge(dst, at, vertical, lines, beta, tc, large, boundary);
    }

    fn vvc_deblock_chroma(
        kernels: DepthKernels<'_>,
        edge: &mut [[u8; 8]; 2],
        beta: i32,
        tc: i32,
        large: bool,
        boundary: bool,
    ) {
        kernels
            .dsp
            .vvc_deblock_chroma(edge, beta, tc, large, boundary);
    }

    fn vvc_alf_filter_block_row(
        kernels: DepthKernels<'_>,
        rows: &[[&[u8]; 7]; VVC_ALF_BLOCK],
        coefficients: &[[i16; VVC_ALF_TAPS]],
        shifts: &[u32; VVC_ALF_BLOCK],
        out: &mut [&mut [u8]; VVC_ALF_BLOCK],
    ) {
        kernels
            .dsp
            .vvc_alf_filter_block_row(rows, coefficients, shifts, out);
    }

    fn vvc_alf_laplacians(kernels: DepthKernels<'_>, rows: &[&[u8]; 4], out: &mut [[u16; 4]]) {
        kernels.dsp.vvc_alf_laplacians(rows, out);
    }

    fn vvc_alf_classify(
        kernels: DepthKernels<'_>,
        groups: &[[u16; 4]],
        boundary: bool,
        out: &mut [[u8; 2]],
    ) {
        kernels.dsp.vvc_alf_classify(groups, boundary, out);
    }

    fn vvc_alf_filter_clipped_block_row(
        kernels: DepthKernels<'_>,
        rows: &[[&[u8]; 7]; VVC_ALF_BLOCK],
        coefficients: &[[i16; VVC_ALF_TAPS]],
        limits: &[[i16; VVC_ALF_TAPS]],
        shifts: &[u32; VVC_ALF_BLOCK],
        out: &mut [&mut [u8]; VVC_ALF_BLOCK],
    ) {
        for r in 0..VVC_ALF_BLOCK {
            kernels.dsp.vvc_alf_filter_clipped_row(
                &rows[r],
                coefficients,
                limits,
                shifts[r],
                out[r],
            );
        }
    }

    fn vvc_alf_filter_clipped_row(
        kernels: DepthKernels<'_>,
        rows: &[&[u8]; 7],
        coefficients: &[[i16; VVC_ALF_TAPS]],
        limits: &[[i16; VVC_ALF_TAPS]],
        shift: u32,
        out: &mut [u8],
    ) {
        kernels
            .dsp
            .vvc_alf_filter_clipped_row(rows, coefficients, limits, shift, out);
    }

    fn vvc_alf_filter_chroma_row(
        kernels: DepthKernels<'_>,
        rows: &[&[u8]; 7],
        coefficients: [i16; 6],
        limits: [i16; 6],
        shift: u32,
        out: &mut [u8],
    ) {
        kernels
            .dsp
            .vvc_alf_filter_chroma_row(rows, coefficients, limits, shift, out);
    }

    fn vvc_alf_cross_row(
        kernels: DepthKernels<'_>,
        rows: &[&[u8]; 4],
        coefficients: [i16; 7],
        out: &mut [u8],
    ) {
        kernels.dsp.vvc_alf_cross_row(rows, coefficients, out);
    }
}

/// `u16`: validate as the 8-bit methods do (thresholds and limits at the
/// depth), then the [`VvcFilterFns`] table.
impl VvcFilterPixel for u16 {
    fn vvc_deblock_luma_edge(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u16>,
        at: (usize, usize),
        vertical: bool,
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    ) {
        check_thresholds(beta, tc, kernels.depth);
        let edge = vvc_luma_edge_at(
            dst,
            at,
            vertical,
            vvc_deblock::EDGE_LINES,
            beta,
            tc,
            lengths,
        );
        // SAFETY: `vvc_luma_edge_at` checked that every sample the lines
        // read lies inside `dst`; the table matches a supported ISA.
        unsafe {
            (kernels.dsp.high.vvc_filter.deblock_luma_edge)(dst.data_mut(), &edge, kernels.depth);
        }
    }

    fn vvc_deblock_luma_segment(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u16>,
        at: (usize, usize),
        vertical: bool,
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    ) {
        check_thresholds(beta, tc, kernels.depth);
        let edge = vvc_luma_edge_at(dst, at, vertical, LUMA_SEGMENT_LINES, beta, tc, lengths);
        // SAFETY: `vvc_luma_edge_at` checked that every sample the lines
        // read lies inside `dst`; the table matches a supported ISA.
        unsafe {
            (kernels.dsp.high.vvc_filter.deblock_luma_edge)(dst.data_mut(), &edge, kernels.depth);
        }
    }

    fn vvc_deblock_luma(
        kernels: DepthKernels<'_>,
        edge: &mut [[u16; 16]; 4],
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    ) {
        check_thresholds(beta, tc, kernels.depth);
        // SAFETY: fixed-size lines; the kernel checks the reaches; the table
        // matches a supported ISA.
        unsafe {
            (kernels.dsp.high.vvc_filter.deblock_luma)(edge, beta, tc, lengths, kernels.depth);
        }
    }

    fn vvc_deblock_chroma_edge(
        kernels: DepthKernels<'_>,
        dst: &mut PlaneViewMut<'_, u16>,
        at: (usize, usize),
        vertical: bool,
        lines: usize,
        beta: i32,
        tc: i32,
        large: bool,
        boundary: bool,
    ) {
        check_thresholds(beta, tc, kernels.depth);
        let edge = vvc_chroma_edge_at(dst, at, vertical, lines, beta, tc, (large, boundary));
        // Every correction clips to +/-tc, so zero cannot change a sample.
        if tc == 0 {
            return;
        }
        // SAFETY: `vvc_chroma_edge_at` checked that p3..q3 of every line lies
        // inside `dst`; the table matches a supported ISA.
        unsafe {
            (kernels.dsp.high.vvc_filter.deblock_chroma_edge)(dst.data_mut(), &edge, kernels.depth);
        }
    }

    fn vvc_deblock_chroma(
        kernels: DepthKernels<'_>,
        edge: &mut [[u16; 8]; 2],
        beta: i32,
        tc: i32,
        large: bool,
        boundary: bool,
    ) {
        check_thresholds(beta, tc, kernels.depth);
        // SAFETY: fixed-size lines; the table matches a supported ISA.
        unsafe {
            (kernels.dsp.high.vvc_filter.deblock_chroma)(
                edge,
                beta,
                tc,
                large,
                boundary,
                kernels.depth,
            );
        }
    }

    fn vvc_alf_filter_block_row(
        kernels: DepthKernels<'_>,
        rows: &[[&[u16]; 7]; VVC_ALF_BLOCK],
        coefficients: &[[i16; VVC_ALF_TAPS]],
        shifts: &[u32; VVC_ALF_BLOCK],
        out: &mut [&mut [u16]; VVC_ALF_BLOCK],
    ) {
        vvc_alf::check_depth(kernels.depth);
        vvc_alf::check_filter_block_row(rows, coefficients, shifts, out);
        // SAFETY: the lengths are checked above; the table matches a
        // supported ISA.
        unsafe {
            (kernels.dsp.high.vvc_filter.alf_filter_block_row)(
                rows,
                coefficients,
                shifts,
                out,
                kernels.depth,
            );
        }
    }

    fn vvc_alf_laplacians(kernels: DepthKernels<'_>, rows: &[&[u16]; 4], out: &mut [[u16; 4]]) {
        vvc_alf::check_depth(kernels.depth);
        vvc_alf::check_laplacians(rows, out);
        // SAFETY: as for `vvc_alf_filter_block_row`.
        unsafe { (kernels.dsp.high.vvc_filter.alf_laplacians)(rows, out) }
    }

    fn vvc_alf_classify(
        kernels: DepthKernels<'_>,
        groups: &[[u16; 4]],
        boundary: bool,
        out: &mut [[u8; 2]],
    ) {
        vvc_alf::check_depth(kernels.depth);
        vvc_alf::check_classify(groups, out, kernels.depth);
        // SAFETY: as for `vvc_alf_filter_block_row`.
        unsafe { (kernels.dsp.high.vvc_filter.alf_classify)(groups, boundary, out, kernels.depth) }
    }

    fn vvc_alf_filter_clipped_block_row(
        kernels: DepthKernels<'_>,
        rows: &[[&[u16]; 7]; VVC_ALF_BLOCK],
        coefficients: &[[i16; VVC_ALF_TAPS]],
        limits: &[[i16; VVC_ALF_TAPS]],
        shifts: &[u32; VVC_ALF_BLOCK],
        out: &mut [&mut [u16]; VVC_ALF_BLOCK],
    ) {
        vvc_alf::check_depth(kernels.depth);
        vvc_alf::check_clipped_block_row(rows, coefficients, limits, shifts, out, kernels.depth);
        // SAFETY: as for `vvc_alf_filter_block_row`.
        unsafe {
            (kernels.dsp.high.vvc_filter.alf_clipped_block_row)(
                rows,
                coefficients,
                limits,
                shifts,
                out,
                kernels.depth,
            );
        }
    }

    fn vvc_alf_filter_clipped_row(
        kernels: DepthKernels<'_>,
        rows: &[&[u16]; 7],
        coefficients: &[[i16; VVC_ALF_TAPS]],
        limits: &[[i16; VVC_ALF_TAPS]],
        shift: u32,
        out: &mut [u16],
    ) {
        vvc_alf::check_depth(kernels.depth);
        vvc_alf::check_clipped_row(rows, coefficients, limits, shift, out, kernels.depth);
        // SAFETY: as for `vvc_alf_filter_block_row`.
        unsafe {
            (kernels.dsp.high.vvc_filter.alf_clipped_row)(
                rows,
                coefficients,
                limits,
                shift,
                out,
                kernels.depth,
            );
        }
    }

    fn vvc_alf_filter_chroma_row(
        kernels: DepthKernels<'_>,
        rows: &[&[u16]; 7],
        coefficients: [i16; 6],
        limits: [i16; 6],
        shift: u32,
        out: &mut [u16],
    ) {
        vvc_alf::check_depth(kernels.depth);
        vvc_alf::check_chroma_row(rows, coefficients, limits, shift, out, kernels.depth);
        // SAFETY: as for `vvc_alf_filter_block_row`.
        unsafe {
            (kernels.dsp.high.vvc_filter.alf_chroma_row)(
                rows,
                coefficients,
                limits,
                shift,
                out,
                kernels.depth,
            );
        }
    }

    fn vvc_alf_cross_row(
        kernels: DepthKernels<'_>,
        rows: &[&[u16]; 4],
        coefficients: [i16; 7],
        out: &mut [u16],
    ) {
        vvc_alf::check_depth(kernels.depth);
        vvc_alf::check_cross_row(rows, coefficients, out);
        // SAFETY: as for `vvc_alf_filter_block_row`.
        unsafe {
            (kernels.dsp.high.vvc_filter.alf_cross_row)(rows, coefficients, out, kernels.depth);
        }
    }
}

#[allow(clippy::too_many_arguments)]
impl DepthKernels<'_> {
    /// [`crate::Dsp::vvc_deblock_luma_edge`] at this depth (`beta`, `tc` as
    /// for [`Self::vvc_deblock_luma_segment`]).
    ///
    /// # Panics
    /// As the 8-bit method, with the thresholds' range at this depth.
    pub fn vvc_deblock_luma_edge<T: super::Pixel>(
        self,
        dst: &mut PlaneViewMut<'_, T>,
        at: (usize, usize),
        vertical: bool,
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    ) {
        T::vvc_deblock_luma_edge(self, dst, at, vertical, beta, tc, lengths);
    }

    /// [`crate::Dsp::vvc_deblock_luma_segment`] at this depth: `beta` and
    /// `tc` are `β` and `tC` at the depth ([`crate::vvc_deblock_beta`],
    /// [`crate::vvc_deblock_tc`]).
    ///
    /// # Panics
    /// As the 8-bit method, with the thresholds' range at this depth.
    pub fn vvc_deblock_luma_segment<T: super::Pixel>(
        self,
        dst: &mut PlaneViewMut<'_, T>,
        at: (usize, usize),
        vertical: bool,
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    ) {
        T::vvc_deblock_luma_segment(self, dst, at, vertical, beta, tc, lengths);
    }

    /// [`crate::Dsp::vvc_deblock_luma`] at this depth (`beta`, `tc` as for
    /// [`Self::vvc_deblock_luma_segment`]).
    ///
    /// # Panics
    /// As the 8-bit method, with the thresholds' range at this depth.
    pub fn vvc_deblock_luma<T: super::Pixel>(
        self,
        edge: &mut [[T; 16]; 4],
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    ) {
        T::vvc_deblock_luma(self, edge, beta, tc, lengths);
    }

    /// [`crate::Dsp::vvc_deblock_chroma_edge`] at this depth (`beta`, `tc`
    /// as for [`Self::vvc_deblock_luma_segment`]).
    ///
    /// # Panics
    /// As the 8-bit method, with the thresholds' range at this depth.
    pub fn vvc_deblock_chroma_edge<T: super::Pixel>(
        self,
        dst: &mut PlaneViewMut<'_, T>,
        at: (usize, usize),
        vertical: bool,
        lines: usize,
        beta: i32,
        tc: i32,
        large: bool,
        boundary: bool,
    ) {
        T::vvc_deblock_chroma_edge(self, dst, at, vertical, lines, beta, tc, large, boundary);
    }

    /// [`crate::Dsp::vvc_deblock_chroma`] at this depth (`beta`, `tc` as
    /// for [`Self::vvc_deblock_luma_segment`]).
    ///
    /// # Panics
    /// As the 8-bit method, with the thresholds' range at this depth.
    pub fn vvc_deblock_chroma<T: super::Pixel>(
        self,
        edge: &mut [[T; 8]; 2],
        beta: i32,
        tc: i32,
        large: bool,
        boundary: bool,
    ) {
        T::vvc_deblock_chroma(self, edge, beta, tc, large, boundary);
    }

    /// [`crate::Dsp::vvc_alf_filter_block_row`] at this depth (output
    /// clipped to it).
    ///
    /// # Panics
    /// As the 8-bit method, or beyond [`crate::VVC_ALF_MAX_BIT_DEPTH`].
    pub fn vvc_alf_filter_block_row<T: super::Pixel>(
        self,
        rows: &[[&[T]; 7]; VVC_ALF_BLOCK],
        coefficients: &[[i16; VVC_ALF_TAPS]],
        shifts: &[u32; VVC_ALF_BLOCK],
        out: &mut [&mut [T]; VVC_ALF_BLOCK],
    ) {
        T::vvc_alf_filter_block_row(self, rows, coefficients, shifts, out);
    }

    /// [`crate::Dsp::vvc_alf_laplacians`] of samples of `T`; each group
    /// sum is at most [`crate::vvc_alf_max_group_sum`] at this depth.
    ///
    /// # Panics
    /// As the 8-bit method, or beyond [`crate::VVC_ALF_MAX_BIT_DEPTH`].
    pub fn vvc_alf_laplacians<T: super::Pixel>(self, rows: &[&[T]; 4], out: &mut [[u16; 4]]) {
        T::vvc_alf_laplacians(self, rows, out);
    }

    /// [`crate::Dsp::vvc_alf_classify`] at this depth: the activity is
    /// shifted by `BitDepth + 4` (H.266 §8.8.5.3). No argument holds
    /// samples, so generic callers name the sample type:
    /// `kernels.vvc_alf_classify::<T>(…)`.
    ///
    /// # Panics
    /// As the 8-bit method, or beyond [`crate::VVC_ALF_MAX_BIT_DEPTH`].
    pub fn vvc_alf_classify<T: super::Pixel>(
        self,
        groups: &[[u16; 4]],
        boundary: bool,
        out: &mut [[u8; 2]],
    ) {
        T::vvc_alf_classify(self, groups, boundary, out);
    }

    /// Four-row block row of [`Self::vvc_alf_filter_clipped_row`], sharing
    /// each 4x4 block's expanded coefficients and `AlfClip` limits across its
    /// four rows.
    ///
    /// # Panics
    /// As [`Self::vvc_alf_filter_clipped_row`].
    pub fn vvc_alf_filter_clipped_block_row<T: super::Pixel>(
        self,
        rows: &[[&[T]; 7]; VVC_ALF_BLOCK],
        coefficients: &[[i16; VVC_ALF_TAPS]],
        limits: &[[i16; VVC_ALF_TAPS]],
        shifts: &[u32; VVC_ALF_BLOCK],
        out: &mut [&mut [T]; VVC_ALF_BLOCK],
    ) {
        T::vvc_alf_filter_clipped_block_row(self, rows, coefficients, limits, shifts, out);
    }

    /// [`crate::Dsp::vvc_alf_filter_clipped_row`] at this depth: `limits`
    /// are `AlfClip` at the depth ([`crate::vvc_alf_clip_value`], H.266
    /// §7.4.3.18).
    ///
    /// # Panics
    /// As the 8-bit method (a limit that is not `AlfClip` at this depth),
    /// or beyond [`crate::VVC_ALF_MAX_BIT_DEPTH`].
    pub fn vvc_alf_filter_clipped_row<T: super::Pixel>(
        self,
        rows: &[&[T]; 7],
        coefficients: &[[i16; VVC_ALF_TAPS]],
        limits: &[[i16; VVC_ALF_TAPS]],
        shift: u32,
        out: &mut [T],
    ) {
        T::vvc_alf_filter_clipped_row(self, rows, coefficients, limits, shift, out);
    }

    /// [`crate::Dsp::vvc_alf_filter_chroma_row`] at this depth (`limits`
    /// as for [`Self::vvc_alf_filter_clipped_row`]).
    ///
    /// # Panics
    /// As [`Self::vvc_alf_filter_clipped_row`].
    pub fn vvc_alf_filter_chroma_row<T: super::Pixel>(
        self,
        rows: &[&[T]; 7],
        coefficients: [i16; 6],
        limits: [i16; 6],
        shift: u32,
        out: &mut [T],
    ) {
        T::vvc_alf_filter_chroma_row(self, rows, coefficients, limits, shift, out);
    }

    /// [`crate::Dsp::vvc_alf_cross_row`] at this depth: the correction is
    /// clipped to `−2^(BitDepth − 1)..=2^(BitDepth − 1) − 1` and the
    /// result to the depth (H.266 §8.8.5.7).
    ///
    /// # Panics
    /// As the 8-bit method, or beyond [`crate::VVC_ALF_MAX_BIT_DEPTH`].
    pub fn vvc_alf_cross_row<T: super::Pixel>(
        self,
        rows: &[&[T]; 4],
        coefficients: [i16; 7],
        out: &mut [T],
    ) {
        T::vvc_alf_cross_row(self, rows, coefficients, out);
    }
}

#[cfg(test)]
#[path = "vvc_filter_tests.rs"]
mod tests;
