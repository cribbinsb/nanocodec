//! Deblocking edges: H.264 (§8.7.2.3, §8.7.2.4) filters one luma edge (16
//! lines) or one 4:2:0 chroma edge (8 lines) of a macroblock, HEVC (§8.7.2.5)
//! 8 luma lines (two 4-line segments) or 4..8 chroma lines of an edge on the
//! 8x8 grid, given the boundary strengths and thresholds the codec derived.
//!
//! The lines across one edge are independent, so the kernels filter many
//! lines at once (one vector lane per line); the order in which a codec
//! visits *edges* is unchanged. Every decision of the standard (the
//! `filterSamplesFlag` test, `ap < β`, `aq < β`, the strong-filter test) is a
//! per-lane mask, and each output takes its new value only where its mask is
//! set, so lanes that are not filtered are written back unchanged.
//!
//! HEVC decides per 4-line luma segment from its first and last line; the
//! kernels compute the inputs of those decisions in every lane, take the
//! two lines' values out to decide, and spread the segment's decisions back
//! over its lanes (`hevc` module). They hold 8 lines per vector: a vertical
//! edge's lines are rows, transposed into lanes with vector shuffles.
//!
//! # Kernel contract
//!
//! Kernels are `unsafe fn`s called only from [`crate::Dsp`], which checks the
//! CPU and that every sample the edge reads (p3..q3 for luma, p1..q1 for
//! chroma) lies inside the plane. Loads and stores index their slices, so a
//! broken contract panics rather than reads out of bounds.

pub(crate) mod av1;
#[allow(unsafe_code)]
pub(crate) mod av1_grid;
#[allow(unsafe_code)]
pub(crate) mod av1_line;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
pub(crate) mod av1_packed;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
pub(crate) mod av1_tile;
pub(crate) mod av2;
#[allow(unsafe_code)]
pub(crate) mod av2_decide;
#[allow(unsafe_code)]
pub(crate) mod av2_ramp;
#[allow(unsafe_code)]
pub(crate) mod generic;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
pub(crate) mod h264_bytes;
pub(crate) mod h264_mb;
#[allow(unsafe_code)]
pub(crate) mod hevc;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
pub(crate) mod hevc_chroma;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
pub(crate) mod line;
#[allow(unsafe_code)]
pub(crate) mod runs;
#[allow(unsafe_code)]
pub(crate) mod samples;
pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

/// Lines across one luma edge of a macroblock.
pub(crate) const LUMA_LINES: usize = 16;
/// Lines across one chroma edge of a 4:2:0 macroblock.
pub(crate) const CHROMA_LINES: usize = 8;

/// One H.264 edge of a macroblock: bS and `tC0` per 4-sample segment
/// (luma; chroma line `k` takes entry `k / 2`), α and β (§8.7.2.2).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct H264EdgeParams {
    /// bS of each segment (0..=4).
    pub bs: [u8; 4],
    /// `tC0` of each segment (unused where bS is 0 or 4).
    pub tc0: [u8; 4],
    /// α of the edge.
    pub alpha: u8,
    /// β of the edge.
    pub beta: u8,
}

/// Every edge of one H.264 macroblock in one plane type, filtered in the
/// standard's order: the vertical edges left to right, then the
/// horizontal edges top to bottom (§8.7). Edge `e` lies `4 · e` luma
/// samples into the macroblock; chroma (4:2:0) has edges 0 and 2 only, 4
/// chroma samples apart.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct H264MbEdges {
    /// Bit `4 · d + e` set: edge `e` of direction `d` (0 vertical, 1
    /// horizontal) is filtered (its bS is not all zero).
    pub mask: u8,
    /// The edges, by direction then edge index.
    pub edges: [[H264EdgeParams; 4]; 2],
}

/// One edge to filter.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Edge {
    /// Index of q0 of the first line.
    pub base: usize,
    /// Index step from one sample to the next across the edge (from p0 to
    /// q0): 1 for a vertical edge, the stride for a horizontal one.
    pub across: usize,
    /// Index step from one line to the next: the stride for a vertical
    /// edge, 1 for a horizontal one.
    pub along: usize,
    /// bS of each group of lines: 4 lines per entry for luma, 2 for chroma.
    pub bs: [u8; 4],
    /// α of the edge.
    pub alpha: i16,
    /// β of the edge.
    pub beta: i16,
    /// `tC0` per entry of `bs` (unused where bS is 0 or 4).
    pub tc0: [i16; 4],
}

/// An edge kernel.
pub(crate) type EdgeFn = unsafe fn(plane: &mut [u8], edge: &Edge);

/// An H.264 chroma edge kernel for both 4:2:0 planes at once: `u` and `v`
/// share the edge's geometry (stride) and parameters.
pub(crate) type ChromaPairFn = unsafe fn(u: &mut [u8], v: &mut [u8], edge: &Edge);

/// Lines of an HEVC edge kernel call (two luma segments).
pub(crate) const HEVC_LINES: usize = 8;
/// Luma lines per HEVC decision segment.
pub(crate) const HEVC_SEGMENT: usize = 4;
/// Samples on each side of an edge an HEVC kernel reads (p3..q3).
pub(crate) const HEVC_REACH: usize = 4;

/// Lines of an HEVC edge, as [`Edge`] but with HEVC's parameters.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HevcEdge {
    /// Index of q0 of the first line.
    pub base: usize,
    /// Index step across the edge (p0 to q0).
    pub across: usize,
    /// Index step from one line to the next.
    pub along: usize,
    /// Lines: [`HEVC_LINES`] for luma; 4 or 8 for chroma.
    pub lines: usize,
    /// `tC` per group of lines (luma: per 4-line segment, entries 0..2;
    /// chroma: per 2 lines); 0 leaves the group unfiltered.
    pub tc: [i16; 4],
    /// `β` (luma only).
    pub beta: i16,
}

/// An HEVC edge kernel.
pub(crate) type HevcEdgeFn = unsafe fn(plane: &mut [u8], edge: &HevcEdge);

/// The deblocking kernels of one ISA.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DeblockFns {
    /// H.264 luma.
    pub luma: EdgeFn,
    /// H.264 chroma.
    pub chroma: EdgeFn,
    /// H.264 chroma, both planes.
    pub chroma_pair: ChromaPairFn,
    /// HEVC luma.
    pub hevc_luma: HevcEdgeFn,
    /// HEVC chroma.
    pub hevc_chroma: HevcEdgeFn,
    /// AV2 ramps of a decided edge segment.
    pub av2_ramps: av2::Av2RampFn,
    /// AV1 edge segments (see [`Av1EdgeFn`]).
    pub av1_edge: Av1EdgeFn,
    /// AV1 four-line descriptors along one edge.
    pub av1_line: av1_line::LineFn,
    /// Row-major AV1 edge descriptors.
    pub av1_grid: av1_grid::GridFn,
    pub av1_runs: runs::Av1RunsFn,
    pub hevc_luma_runs: runs::HevcLumaRunsFn,
    pub hevc_chroma_runs: runs::HevcChromaRunsFn,
    /// AV2 decisions and ramps of a chunk of segments of one edge line.
    pub av2_chunk: av2::Av2ChunkFn,
}

/// An AV1 edge kernel: `lines` (a multiple of 4) lines across one edge,
/// addressed as [`av1::filter_edge`] (`q0`, `across`, `along`). The caller
/// checked the filter's reach on every line.
pub(crate) type Av1EdgeFn = unsafe fn(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    lines: usize,
    filter: av1::Av1EdgeFilter,
);

/// The deblocking table of `isa`.
///
/// # Safety
///
/// The CPU must support `isa` (`isa.is_supported()`).
#[allow(
    unsafe_code,
    reason = "selects the ISA's table; AVX2's needs the CPU check"
)]
pub(crate) unsafe fn kernels(isa: crate::Isa) -> DeblockFns {
    match isa {
        crate::Isa::Scalar => scalar::deblock_fns(),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the caller checked AVX2 support.
        crate::Isa::Avx2 => unsafe { avx2::deblock_fns() },
        #[cfg(target_arch = "aarch64")]
        crate::Isa::Neon => neon::deblock_fns(),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        crate::Isa::Simd128 => simd128::deblock_fns(),
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_supported() is false for ISAs not compiled in"),
    }
}
