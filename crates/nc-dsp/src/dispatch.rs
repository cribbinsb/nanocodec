//! [`Dsp`]: a per-instance table of kernel implementations behind safe methods.
//!
//! Soundness rests on two checks, both made here and nowhere else:
//!
//! 1. A table for an ISA is only built after [`Isa::is_supported`] confirms
//!    the CPU can run it ([`Dsp::with_isa`]).
//! 2. Every call validates its views against the block size before calling
//!    a kernel, which is what the kernel contract (`pixel` module docs)
//!    requires.

#![allow(unsafe_code)]

use nc_common::{PlaneView, PlaneViewMut};

use crate::av1_inverse::{Av1TxSize, Av1TxType};
use crate::av1_mc::filter::Av1Filters;
use crate::av2_txfm::typed::Av2TxType;
use crate::av2_txfm::typed_lanes::Av2TypedFns;
use crate::av2_txfm::{self, Av2TxSize, Av2TxfmFns};
use crate::block::BlockSize;
use crate::cdef::{self, CdefArgs, CdefBlock, CdefFns, CdefSource};
use crate::codec_txfm::{
    self, CodecTxfmFns, H264Quantizer, HEVC_MAX_LEVEL, HevcForwardMode, HevcQuantizer,
    HevcTransform, LEVEL_SCALE_4X4, LEVEL_SCALE_8X8, MAX_MASK_LEVELS, QUANT_MF_4X4_POS,
    QUANT_MF_8X8_POS,
};
use crate::convert::{self, ConvertFns, DisplayMatrix};
use crate::cpu::Isa;
use crate::deblock::av1::{AV1_EDGE_LINES, Av1EdgeFilter};
use crate::deblock::av2::{
    AV2_DECIDE_REACH, AV2_DECIDE_SEGMENTS, AV2_EDGE_LINES, Av2EdgeFilter, Av2Segments,
};
use crate::deblock::{
    self, DeblockFns, Edge, H264EdgeParams, H264MbEdges, HEVC_LINES, HEVC_REACH, HevcEdge, h264_mb,
};
use crate::diff::{self, DiffFns, SUBPEL_VARIANCE_POSITIONS, Variance, VarianceEnergy};
use crate::mc::{self, ConvolveFn, Filter, McFns, SubBlockMotion};
use crate::pixel::{self, GridBest, GridRows, PixelFns};
use crate::sample::{self, Moments, SampleFns};
use crate::sao::{self, Rect, SAO_EO_NEIGHBOURS, SaoFns, SaoStats};
use crate::txfm::{self, Av1Quantizer, DctSize, TxfmFns};

/// Pixel kernels for one ISA. Cheap to copy; create one per encoder or
/// decoder instance.
///
/// Not every kernel goes through this table: the AV1 symbol-decoder
/// kernels ([`crate::av1_symbol`]) run once per decoded symbol, where an
/// indirect call would cost more than they save, so they use each target's
/// baseline vector unit (SSE2, NEON, SIMD128), chosen at compile time.
#[derive(Debug, Clone, Copy)]
pub struct Dsp {
    isa: Isa,
    pixel: PixelFns,
    sample: SampleFns,
    txfm: TxfmFns,
    codec_txfm: CodecTxfmFns,
    av2_txfm: Av2TxfmFns,
    av2_typed: Av2TypedFns,
    mc: McFns,
    diff: DiffFns,
    reconstruct: crate::reconstruct::Kernel,
    border_store: crate::border_store::BorderStoreFns,
    copy_plane: crate::block_copy::CopyFn,
    cdef: CdefFns,
    sao: SaoFns,
    deblock: DeblockFns,
    convert: ConvertFns,
    vp9_mc: crate::vp9_mc_simd::ConvolveFn,
    vp9_inverse: crate::vp9_inverse::Kernel,
    vp9_inverse_add: crate::vp9_inverse::AddKernel,
    av1_inverse: crate::av1_inverse::Av1InverseFns,
    vp9_filter: crate::vp9_loop_filter::FilterFn,
    vp9_line: crate::vp9_loop_filter::LineFn,
    av2_intra: crate::av2_intra::Av2IntraFns,
    av1_filter_intra: crate::av1_intra::FilterPatchesFn,
    av1_directional: crate::av1_intra::DirectionalFn,
    av1_smooth_paeth: crate::av1_intra::SmoothPaethFn,
    vp9_quantize: crate::vp9_quant::QuantizeFn,
    vp9_largest_magnitude: crate::vp9_quant::MagnitudeFn,
    vp9_intra: crate::vp9_intra::PredictFn,
    h264_intra_8x8: crate::h264_intra::PredictFn,
    hevc_intra: crate::hevc_intra::HevcIntraFns,
    hevc_weight: crate::hevc_weight::WeightFn,
    vvc_intra: crate::vvc_intra::PredictFn,
    vvc_deblock: crate::vvc_deblock::Kernels,
    vvc_alf: crate::vvc_alf::Kernels,
    gram_rows: crate::gram::GramRowsFn,
    vvc_refinement: crate::vvc_refinement::Kernels,
    av1_grain: crate::av1_film_grain::kernels::Kernels,
    av1_scaled: crate::av1_mc::scaled::Kernels,
    resample: crate::resample::kernels::ResampleFn,
    av1_lr: crate::av1_restoration::kernels::Kernels,
    av1_warp: crate::av1_mc::warp_kernels::WarpFn,
    av1_mask_blend: crate::av1_mc::mask_kernels::MaskBlendFn,
    pub(crate) vvc_mts: crate::vvc_mts::Kernels,
    pub(crate) vvc_tx64: crate::vvc_tx64::HalveFn,
}

impl Default for Dsp {
    fn default() -> Self {
        Self::new()
    }
}

impl Dsp {
    /// Kernels for the fastest ISA this CPU and build support.
    /// `NC_DSP_ISA` may force `scalar`, `avx2`, `avx512`, `neon` or `simd128`;
    /// absent or `auto` selects the fastest supported tier. The environment
    /// is read once per table, never in a kernel. [`Self::with_isa`] ignores it.
    ///
    /// # Panics
    ///
    /// If the override is unknown or unsupported; a forced comparison must
    /// never silently measure a different ISA.
    #[must_use]
    pub fn new() -> Self {
        let isa = match std::env::var("NC_DSP_ISA") {
            Ok(value) if value == "auto" => Isa::best(),
            Ok(value) => value
                .parse()
                .expect("NC_DSP_ISA must name a DSP ISA or auto"),
            Err(std::env::VarError::NotPresent) => Isa::best(),
            Err(error) => panic!("invalid NC_DSP_ISA: {error}"),
        };
        Self::with_isa(isa).expect("requested NC_DSP_ISA is not supported by this CPU/build")
    }

    /// Kernels for a specific ISA, or `None` if it cannot run here.
    #[must_use]
    #[allow(clippy::too_many_lines, reason = "one table entry per kernel family")]
    pub fn with_isa(isa: Isa) -> Option<Self> {
        if !isa.is_supported() {
            return None;
        }
        #[cfg(target_arch = "x86_64")]
        if isa == Isa::Avx512 {
            return Self::avx512();
        }
        let (pixel, sample, txfm, mc, diff) = match isa {
            Isa::Scalar => (
                pixel::scalar::pixel_fns(),
                sample::scalar::sample_fns(),
                txfm::scalar::txfm_fns(),
                mc::scalar::mc_fns(),
                diff::scalar::diff_fns(),
            ),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: AVX2 support was checked just above.
            Isa::Avx2 => unsafe {
                (
                    pixel::avx2::pixel_fns(),
                    sample::avx2::sample_fns(),
                    txfm::avx2::txfm_fns(),
                    mc::avx2::mc_fns(),
                    diff::avx2::diff_fns(),
                )
            },
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => (
                pixel::neon::pixel_fns(),
                sample::neon::sample_fns(),
                txfm::neon::txfm_fns(),
                mc::neon::mc_fns(),
                diff::neon::diff_fns(),
            ),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Isa::Simd128 => (
                pixel::simd128::pixel_fns(),
                sample::simd128::sample_fns(),
                txfm::simd128::txfm_fns(),
                mc::simd128::mc_fns(),
                diff::simd128::diff_fns(),
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!("is_supported() is false for ISAs not compiled in"),
        };
        // SAFETY: support for `isa` was checked above.
        let cdef = unsafe { cdef_fns(isa) };
        // SAFETY: support for `isa` was checked above.
        let sao = unsafe { sao_fns(isa) };
        // SAFETY: support for `isa` was checked above.
        let deblock = unsafe { deblock::kernels(isa) };
        // SAFETY: support for `isa` was checked above.
        let convert = unsafe { convert_fns(isa) };
        Some(Self {
            isa,
            pixel,
            sample,
            txfm,
            codec_txfm: codec_txfm_fns(isa),
            av2_txfm: av2_txfm_fns(isa),
            av2_typed: av2_typed_fns(isa),
            mc,
            diff,
            reconstruct: crate::reconstruct::kernel(isa),
            border_store: crate::border_store::border_store_fns(isa),
            copy_plane: crate::block_copy::copy,
            cdef,
            sao,
            deblock,
            convert,
            vp9_filter: crate::vp9_loop_filter::kernel(isa),
            vp9_line: crate::vp9_loop_filter::line_kernel(isa),
            vp9_mc: crate::vp9_mc_simd::kernel(isa),
            vp9_inverse: crate::vp9_inverse::kernel(isa),
            vp9_inverse_add: crate::vp9_inverse::add_kernel(isa),
            av1_inverse: crate::av1_inverse::kernels::for_isa(isa),
            av2_intra: crate::av2_intra::dispatch_fns(isa),
            av1_filter_intra: crate::av1_intra::filter_patches(isa),
            av1_directional: crate::av1_intra::directional(isa),
            av1_smooth_paeth: crate::av1_intra::smooth_paeth(isa),
            vp9_quantize: crate::vp9_quant::kernel(isa),
            vp9_largest_magnitude: crate::vp9_quant::magnitude_kernel(isa),
            vp9_intra: crate::vp9_intra::kernel(isa),
            h264_intra_8x8: crate::h264_intra::kernel(isa),
            hevc_intra: crate::hevc_intra::kernels(isa),
            hevc_weight: crate::hevc_weight::kernel(isa),
            vvc_intra: crate::vvc_intra::kernel(isa),
            vvc_deblock: crate::vvc_deblock::kernels(isa),
            vvc_alf: crate::vvc_alf::kernels(isa),
            gram_rows: crate::gram::kernel(isa),
            vvc_refinement: crate::vvc_refinement::kernels(isa),
            av1_grain: crate::av1_film_grain::kernels::kernels(isa),
            av1_scaled: crate::av1_mc::scaled::kernels(isa),
            resample: crate::resample::kernels::kernel(isa),
            av1_lr: crate::av1_restoration::kernels::kernels(isa),
            av1_warp: crate::av1_mc::warp_kernels::kernel(isa),
            av1_mask_blend: crate::av1_mc::mask_kernels::kernel(isa),
            vvc_mts: crate::vvc_mts::kernels(isa),
            vvc_tx64: crate::vvc_tx64::kernel(isa),
        })
    }

    /// Inherit every fallback before replacing independently tuned kernels.
    #[cfg(target_arch = "x86_64")]
    fn avx512() -> Option<Self> {
        let mut dsp = Self::with_isa(Isa::Avx2)?;
        dsp.isa = Isa::Avx512;
        cdef::avx512::install(&mut dsp.cdef);
        sao::avx512::install(&mut dsp.sao);
        pixel::avx512::install(&mut dsp.pixel);
        sample::avx512::install(&mut dsp.sample);
        dsp.convert.sse_row = crate::convert::avx512::sse_row;
        crate::mc::avx512::install(&mut dsp.mc);
        diff::avx512::install(&mut dsp.diff);
        txfm::avx512::install(&mut dsp.txfm);
        codec_txfm::avx512::install(&mut dsp.codec_txfm);
        crate::av1_inverse::avx512::install(&mut dsp.av1_inverse);
        Some(dsp)
    }

    /// AV2 intra prediction of a `side`×`side` raster `dst` from `edges`,
    /// as [`Av2IntraEdges::predict`] but with this ISA's run and smooth
    /// kernels.
    ///
    /// # Panics
    ///
    /// As [`Av2IntraEdges::predict`].
    ///
    /// [`Av2IntraEdges::predict`]: crate::intra_prediction::av2::Av2IntraEdges::predict
    pub fn av2_intra_predict(
        &self,
        edges: &mut crate::intra_prediction::av2::Av2IntraEdges,
        predictor: crate::intra_prediction::av2::Av2Predictor,
        side: usize,
        dst: &mut [u8],
    ) {
        edges.predict_with(&self.av2_intra, predictor, side, dst);
    }

    /// AV1 filter intra prediction, as
    /// [`IntraEdges::predict_filter`](crate::intra_prediction::IntraEdges::predict_filter)
    /// but with this ISA's patch kernel.
    ///
    /// # Panics
    ///
    /// As `IntraEdges::predict_filter`.
    pub fn av1_intra_predict_filter(
        &self,
        edges: &crate::intra_prediction::IntraEdges,
        mode: crate::intra_prediction::filter::FilterIntraMode,
        dst: &mut nc_common::PlaneViewMut<'_>,
    ) {
        let kernel = self.av1_filter_intra;
        edges.predict_filter_with(
            // SAFETY: the table was built for a supported ISA; the kernel
            // indexes the grid, whose side `predict_filter_with` checked.
            |grid, width, height, mode| unsafe { kernel(grid, width, height, mode) },
            mode,
            dst,
        );
    }

    /// AV1 directional prediction of a square, as
    /// [`IntraEdges::predict_angle`](crate::intra_prediction::IntraEdges::predict_angle)
    /// but with this ISA's kernels.
    ///
    /// # Panics
    ///
    /// As `IntraEdges::predict_angle`.
    pub fn av1_intra_predict_angle(
        &self,
        edges: &crate::intra_prediction::IntraEdges,
        angle: u16,
        dst: &mut nc_common::PlaneViewMut<'_>,
    ) {
        // SAFETY: the table was built for a supported ISA; the kernel
        // checks the angle and the destination as the scalar code does.
        unsafe { (self.av1_directional)(edges, angle, dst) }
    }

    /// AV1 intra prediction of a square in `mode`, as
    /// [`IntraEdges::predict`](crate::intra_prediction::IntraEdges::predict)
    /// but with this ISA's directional, smooth and Paeth kernels.
    ///
    /// # Panics
    ///
    /// As `IntraEdges::predict`.
    pub fn av1_intra_predict(
        &self,
        edges: &crate::intra_prediction::IntraEdges,
        mode: crate::intra_prediction::IntraMode,
        dst: &mut nc_common::PlaneViewMut<'_>,
    ) {
        use crate::intra_prediction::IntraMode;
        match mode {
            IntraMode::Paeth
            | IntraMode::Smooth
            | IntraMode::SmoothVertical
            | IntraMode::SmoothHorizontal => {
                // SAFETY: the table was built for a supported ISA; the
                // kernel checks the destination as the scalar code does.
                unsafe { (self.av1_smooth_paeth)(edges, mode, dst) }
            }
            IntraMode::D45
            | IntraMode::D135
            | IntraMode::D113
            | IntraMode::D157
            | IntraMode::D203
            | IntraMode::D67 => {
                let angle = mode.nominal_angle().expect("oblique modes have an angle");
                self.av1_intra_predict_angle(edges, angle, dst);
            }
            IntraMode::Dc | IntraMode::Vertical | IntraMode::Horizontal => {
                edges.predict(mode, dst);
            }
        }
    }

    /// AV1 intra prediction of any block shape, as
    /// [`BlockEdges::predict`](crate::intra_prediction::av1_block::BlockEdges::predict)
    /// but with this ISA's kernels for square blocks.
    ///
    /// # Panics
    ///
    /// As `BlockEdges::predict`.
    pub fn av1_intra_predict_block(
        &self,
        edges: &crate::intra_prediction::av1_block::BlockEdges,
        kind: crate::intra_prediction::av1_block::IntraKind,
        dst: &mut nc_common::PlaneViewMut<'_>,
    ) {
        use crate::intra_prediction::av1_block::Square;
        edges.predict_with(
            |square, prediction, dst| match prediction {
                Square::Angle(angle) => self.av1_intra_predict_angle(square, angle, dst),
                Square::Mode(mode) => self.av1_intra_predict(square, mode, dst),
            },
            kind,
            dst,
        );
    }

    /// AV1 filter intra of any block shape up to 32x32, as
    /// [`BlockEdges::predict_filter`](crate::intra_prediction::av1_block::BlockEdges::predict_filter)
    /// but with this ISA's patch kernel.
    ///
    /// # Panics
    ///
    /// As `BlockEdges::predict_filter`.
    pub fn av1_intra_predict_filter_block(
        &self,
        edges: &crate::intra_prediction::av1_block::BlockEdges,
        mode: crate::intra_prediction::filter::FilterIntraMode,
        dst: &mut nc_common::PlaneViewMut<'_>,
    ) {
        let kernel = self.av1_filter_intra;
        edges.predict_filter_with(
            // SAFETY: the table was built for a supported ISA; the kernel
            // indexes the grid, whose sides `predict_filter_with` checked.
            |grid, width, height, mode| unsafe { kernel(grid, width, height, mode) },
            mode,
            dst,
        );
    }

    /// VP9 deblocking with the same arguments as [`crate::vp9_filter_edge`].
    /// The full SIMD footprint is checked up front. If it does not fit, use
    /// the scalar function so that its lazy outer-tap access remains unchanged.
    /// Zero lines do nothing. Arbitrary strides and short tails use the oracle.
    pub fn vp9_filter_edge(
        &self,
        data: &mut [u8],
        edge: usize,
        step: usize,
        along: usize,
        lines: usize,
        width: crate::Vp9FilterWidth,
        limits: crate::Vp9FilterLimits,
    ) {
        if lines == 0 {
            return;
        }
        let reach: usize = if width == crate::Vp9FilterWidth::Sixteen {
            8
        } else {
            4
        };
        let before = reach.checked_mul(step).is_some_and(|before| edge >= before);
        let last = (lines - 1)
            .checked_mul(along)
            .and_then(|x| edge.checked_add(x))
            .and_then(|x| (reach - 1).checked_mul(step).and_then(|y| x.checked_add(y)))
            .is_some_and(|last| last < data.len());
        if !before || !last {
            crate::vp9_filter_edge(data, edge, step, along, lines, width, limits);
            return;
        }
        // SAFETY: construction checked the ISA; the checks above cover every tap.
        unsafe {
            (self.vp9_filter)(data, edge, step, along, lines, width, limits);
        }
    }

    /// Filter masked eight-line VP9 segments along one edge (VP9 §8.8).
    /// The final active segment clips to the picture's along-edge extent;
    /// empty and entirely masked lines do nothing. The codec schedules edges.
    ///
    /// # Panics
    /// If an active segment starts outside the picture or its across-edge
    /// footprint leaves the picture.
    pub fn vp9_loop_filter_line(
        &self,
        dst: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        vertical: bool,
        segments: &[Option<crate::Vp9EdgeFilter>],
    ) {
        let mut reach = 0;
        let mut end = 0;
        for (i, filter) in segments.iter().enumerate() {
            if let Some(filter) = filter {
                reach = reach.max(if filter.width == crate::Vp9FilterWidth::Sixteen {
                    8
                } else {
                    4
                });
                end = i + 1;
            }
        }
        if end == 0 {
            return;
        }
        let span = end
            .checked_mul(crate::VP9_EDGE_LINES)
            .expect("VP9 edge length");
        let (across_pos, across_len, along_pos, along_len) = if vertical {
            (x, dst.width(), y, dst.height())
        } else {
            (y, dst.height(), x, dst.width())
        };
        assert!(
            across_pos >= reach
                && across_pos
                    .checked_add(reach)
                    .is_some_and(|v| v <= across_len)
                && along_pos
                    .checked_add(span - crate::VP9_EDGE_LINES)
                    .is_some_and(|v| v < along_len),
            "VP9 loop filter line reads outside the plane"
        );
        let lines = span.min(along_len - along_pos);
        let stride = dst.stride();
        let (across, along) = if vertical { (1, stride) } else { (stride, 1) };
        // SAFETY: all active starts and their across-edge reach are validated;
        // the clipped line count bounds the final tail. The table enables its ISA.
        unsafe {
            (self.vp9_line)(
                dst.data_mut(),
                y * stride + x,
                across,
                along,
                lines,
                &segments[..end],
            );
        }
    }

    /// [`Self::vp9_inverse_transform`] added to the prediction in the
    /// block at the start of `dst` (rows `dst_stride` apart), clipped to 8
    /// bits; `log2` 2..=5 (4x4 to 32x32). The free `vp9_inverse_add` is the
    /// scalar oracle of 8x8 and larger. `nonzero`
    /// is `(rows, columns)`: every coefficient at or past row `rows` or
    /// column `columns` must be zero (the decoder knows where its tokens
    /// went). The kernels pick their zero-region variant from it: DC only,
    /// a top-left 4x4/8x8/16x16 corner, or the whole block (libvpx's eob
    /// classes).
    ///
    /// # Panics
    ///
    /// As [`Self::vp9_inverse_transform`], if a coefficient outside the
    /// extent is nonzero (debug builds), or if `dst` is too small.
    pub fn vp9_inverse_add(
        &self,
        log2: u8,
        (columns, rows): (crate::Vp9Pass, crate::Vp9Pass),
        input: &[i32],
        nonzero: (usize, usize),
        (dst, dst_stride): (&mut [u8], usize),
    ) {
        assert!((2..=5).contains(&log2), "VP9 transform size outside 4..=32");
        assert!(
            log2 != 5 || (columns == crate::Vp9Pass::Dct && rows == crate::Vp9Pass::Dct),
            "VP9 32-point ADST is unsupported"
        );
        let side = 1_usize << log2;
        let input = &input[..side * side];
        debug_assert!(
            input
                .iter()
                .enumerate()
                .all(|(i, &c)| c == 0 || (i / side < nonzero.0 && i % side < nonzero.1))
        );
        assert!(dst_stride >= side && dst.len() > (side - 1) * dst_stride + side - 1);
        if log2 == 2 {
            // 4x4 blocks are scalar on every ISA: fused, with libvpx's DC-only
            // and zero-row shortcuts, they cost a few dozen operations.
            crate::vp9_txfm::vp9_inverse4_add(columns, rows, input, nonzero, dst, dst_stride);
            return;
        }
        // SAFETY: the table matches the checked CPU ISA; the slices cover the
        // validated transform and destination block.
        unsafe {
            (self.vp9_inverse_add)(log2, columns, rows, input, nonzero, dst, dst_stride);
        }
    }

    /// VP9 inverse DCT/ADST with normative i16 stage wrapping. Inputs and
    /// outputs are packed raster blocks; `log2` is 3..=5, with DCT only at 5.
    /// The free `vp9_inverse_transform` function remains the scalar oracle.
    ///
    /// # Panics
    /// If the size/pass combination is invalid or either slice is too short.
    pub fn vp9_inverse_transform(
        &self,
        log2: u8,
        columns: crate::Vp9Pass,
        rows: crate::Vp9Pass,
        input: &[i32],
        output: &mut [i32],
    ) {
        assert!((3..=5).contains(&log2), "VP9 transform size outside 8..=32");
        assert!(
            log2 != 5 || (columns == crate::Vp9Pass::Dct && rows == crate::Vp9Pass::Dct),
            "VP9 32-point ADST is unsupported"
        );
        let samples = 1_usize << (2 * log2);
        let input = &input[..samples];
        let output = &mut output[..samples];
        // SAFETY: the table matches the checked CPU ISA; both slices cover
        // the validated transform and every stage uses its specified wrap.
        unsafe {
            (self.vp9_inverse)(log2, columns, rows, input, output);
        }
    }

    /// The ISA these kernels use.
    #[must_use]
    pub fn isa(&self) -> Isa {
        self.isa
    }

    /// Sum of absolute differences between the top-left `size` blocks of
    /// `src` and `rf`.
    ///
    /// # Panics
    ///
    /// If either view is smaller than `size`.
    #[must_use]
    pub fn sad(&self, size: BlockSize, src: &PlaneView<'_>, rf: &PlaneView<'_>) -> u32 {
        check_covers(size, src, "source");
        check_covers(size, rf, "reference");
        let kernel = self.pixel.sad[size.width_class()];
        // SAFETY: both views cover the block (checked above) and the table
        // matches a supported ISA (checked in `with_isa`).
        unsafe {
            kernel(
                src.data(),
                src.stride(),
                rf.data(),
                rf.stride(),
                size.height(),
            )
        }
    }

    /// SAD of the `src` block against four candidate blocks, which must
    /// share a stride (they come from the same reference plane).
    ///
    /// # Panics
    ///
    /// If any view is smaller than `size`, or the candidates' strides differ.
    #[must_use]
    pub fn sad_x4(
        &self,
        size: BlockSize,
        src: &PlaneView<'_>,
        refs: &[PlaneView<'_>; 4],
    ) -> [u32; 4] {
        check_covers(size, src, "source");
        let ref_stride = refs[0].stride();
        for rf in refs {
            check_covers(size, rf, "reference");
            assert_eq!(
                rf.stride(),
                ref_stride,
                "sad_x4 candidates must share a stride"
            );
        }
        let kernel = self.pixel.sad_x4[size.width_class()];
        let ref_data = refs.map(|rf| rf.data());
        // SAFETY: all views cover the block and share `ref_stride` (checked
        // above); the table matches a supported ISA.
        unsafe {
            kernel(
                src.data(),
                src.stride(),
                ref_data,
                ref_stride,
                size.height(),
            )
        }
    }

    /// [`Dsp::sad_x4`] with the four candidates given as top-left positions
    /// (`x`, `y`) inside one `reference` view, the form motion searches use:
    /// checking a position is two comparisons, much cheaper than building a
    /// view per candidate.
    ///
    /// # Panics
    ///
    /// If `src` is smaller than `size` or a candidate block leaves `reference`.
    #[must_use]
    pub fn sad_x4_at(
        &self,
        size: BlockSize,
        src: &PlaneView<'_>,
        reference: &PlaneView<'_>,
        positions: [(usize, usize); 4],
    ) -> [u32; 4] {
        check_covers(size, src, "source");
        let refs = candidate_slices(size, reference, positions);
        let kernel = self.pixel.sad_x4[size.width_class()];
        // SAFETY: the source view covers the block and every candidate slice
        // covers its block at the reference stride (checked above and in
        // `candidate_slices`); the table matches a supported ISA.
        unsafe {
            kernel(
                src.data(),
                src.stride(),
                refs,
                reference.stride(),
                size.height(),
            )
        }
    }

    /// [`Dsp::sad`] with the reference block given as a top-left position
    /// (`x`, `y`) inside a `reference` view, the form motion searches use (see
    /// [`Dsp::sad_x4_at`]).
    ///
    /// # Panics
    ///
    /// If `src` is smaller than `size` or the reference block leaves
    /// `reference`.
    #[must_use]
    pub fn sad_at(
        &self,
        size: BlockSize,
        src: &PlaneView<'_>,
        reference: &PlaneView<'_>,
        position: (usize, usize),
    ) -> u32 {
        check_covers(size, src, "source");
        let rf = block_slice(size, reference, position);
        let kernel = self.pixel.sad[size.width_class()];
        // SAFETY: the source view covers the block, the reference slice covers
        // its block at the reference stride (checked above and in
        // `block_slice`); the table matches a supported ISA.
        unsafe {
            kernel(
                src.data(),
                src.stride(),
                rf,
                reference.stride(),
                size.height(),
            )
        }
    }

    /// [`Dsp::sad_at`] over every other row only (rows 0, 2, 4, ...): about
    /// half the cost, and an estimate of half the full SAD. For large blocks
    /// whose cost only needs to be compared, not exact (e.g. aggregating
    /// motion costs to 32x32 and 64x64).
    ///
    /// # Panics
    ///
    /// If `size` is less than 8 rows high, `src` is smaller than `size`, or
    /// the reference block leaves `reference`.
    #[must_use]
    pub fn sad_even_rows_at(
        &self,
        size: BlockSize,
        src: &PlaneView<'_>,
        reference: &PlaneView<'_>,
        position: (usize, usize),
    ) -> u32 {
        assert!(size.height() >= 8, "row subsampling needs at least 8 rows");
        check_covers(size, src, "source");
        let rf = block_slice(size, reference, position);
        let kernel = self.pixel.sad[size.width_class()];
        // SAFETY: with doubled strides and half the rows, the kernel reads
        // rows 0, 2, ..., h - 2 of blocks that both slices cover in full
        // (checked above and in `block_slice`). Half of a supported height
        // (>= 8) is still a supported height (>= 4). The table matches a
        // supported ISA.
        unsafe {
            kernel(
                src.data(),
                2 * src.stride(),
                rf,
                2 * reference.stride(),
                size.height() / 2,
            )
        }
    }

    /// SADs of the four 4x4 quadrants of each 8x8 block along a row of
    /// `src` and `rf`: block i covers columns 8i..8i+8 of the top 8 rows.
    /// Each entry is [top-left, top-right, bottom-left, bottom-right] and
    /// sums to that block's SAD.
    ///
    /// # Panics
    ///
    /// If either view is smaller than 8 rows of `8 * out.len()` samples.
    pub fn sad_quads_8x8_row(&self, src: &PlaneView<'_>, rf: &PlaneView<'_>, out: &mut [[u32; 4]]) {
        check_covers_row(out.len(), src, "source");
        check_covers_row(out.len(), rf, "reference");
        // SAFETY: both views cover the row of blocks (checked above); the
        // table matches a supported ISA.
        unsafe {
            (self.pixel.sad_quads_8x8_row)(src.data(), src.stride(), rf.data(), rf.stride(), out);
        }
    }

    /// The 4x4 SADs of a whole 16x16 block, as four 8x8 blocks in raster
    /// order, each `[top-left, top-right, bottom-left, bottom-right]`: the
    /// same values as two [`Dsp::sad_quads_8x8_row`] calls, behind one check.
    ///
    /// # Panics
    ///
    /// If either view is smaller than 16x16.
    #[must_use]
    pub fn sad_quads_16x16(&self, src: &PlaneView<'_>, rf: &PlaneView<'_>) -> [[u32; 4]; 4] {
        let size = BlockSize::new(16, 16).expect("16x16 is a block size");
        check_covers(size, src, "source");
        check_covers(size, rf, "reference");
        let (src_stride, ref_stride) = (src.stride(), rf.stride());
        let mut out = [[0u32; 4]; 4];
        let (top, bottom) = out.split_at_mut(2);
        // SAFETY: both views cover the 16x16 block (checked above), i.e. two
        // rows of two 8x8 blocks; the table matches a supported ISA.
        unsafe {
            (self.pixel.sad_quads_8x8_row)(src.data(), src_stride, rf.data(), ref_stride, top);
            (self.pixel.sad_quads_8x8_row)(
                &src.data()[8 * src_stride..],
                src_stride,
                &rf.data()[8 * ref_stride..],
                ref_stride,
                bottom,
            );
        }
        out
    }

    /// The 4x4 SADs of a 4:2:0 macroblock: [`Dsp::sad_quads_16x16`] of the
    /// luma views and [`Dsp::sad_quads_8x8`] of each chroma pair, behind one
    /// call (separately, the three calls' overhead was about a third of
    /// their cost per H.264 macroblock).
    ///
    /// # Panics
    ///
    /// If a luma view is smaller than 16x16 or a chroma view than 8x8.
    #[must_use]
    pub fn sad_quads_macroblock(
        &self,
        luma: (&PlaneView<'_>, &PlaneView<'_>),
        chroma: [(&PlaneView<'_>, &PlaneView<'_>); 2],
    ) -> ([[u32; 4]; 4], [[u32; 4]; 2]) {
        let size = BlockSize::new(16, 16).expect("16x16 is a block size");
        check_covers(size, luma.0, "source");
        check_covers(size, luma.1, "reference");
        for (src, rf) in chroma {
            check_covers_row(1, src, "source");
            check_covers_row(1, rf, "reference");
        }
        // SAFETY: every view covers its block (checked above); the table
        // matches a supported ISA.
        unsafe { (self.pixel.sad_4x4_macroblock)(pair(luma), [pair(chroma[0]), pair(chroma[1])]) }
    }

    /// The four 4x4 SADs of an 8x8 block, `[top-left, top-right,
    /// bottom-left, bottom-right]`.
    ///
    /// # Panics
    ///
    /// If either view is smaller than 8x8.
    #[must_use]
    pub fn sad_quads_8x8(&self, src: &PlaneView<'_>, rf: &PlaneView<'_>) -> [u32; 4] {
        check_covers_row(1, src, "source");
        check_covers_row(1, rf, "reference");
        let mut out = [[0u32; 4]; 1];
        // SAFETY: both views cover the 8x8 block (checked above); the table
        // matches a supported ISA.
        unsafe {
            (self.pixel.sad_quads_8x8_row)(
                src.data(),
                src.stride(),
                rf.data(),
                rf.stride(),
                &mut out,
            );
        }
        out[0]
    }

    /// SADs of the 16x16 block of `src`, over rows 0, 2, ..., 14 only,
    /// against `out.len()` reference blocks at consecutive x positions
    /// starting at `position` inside `reference`: `out[dx]` is the
    /// candidate at (`x + dx`, `y`). About half the cost of full SADs, and
    /// an estimate of half of them, for wide searches on small images.
    ///
    /// # Panics
    ///
    /// If `src` is smaller than 16x16, `out` is empty, or the scanned
    /// reference area leaves `reference`.
    pub fn sad_scan_16x16_even_rows_at(
        &self,
        src: &PlaneView<'_>,
        reference: &PlaneView<'_>,
        (x, y): (usize, usize),
        out: &mut [u32],
    ) {
        assert!(!out.is_empty(), "empty SAD scan");
        check_covers(
            BlockSize::new(16, 16).expect("16x16 is valid"),
            src,
            "source",
        );
        let width = 16 + out.len() - 1;
        let fits = x
            .checked_add(width)
            .is_some_and(|end| end <= reference.width())
            && y.checked_add(16)
                .is_some_and(|end| end <= reference.height());
        assert!(
            fits,
            "scan of {width}x16 at ({x}, {y}) leaves the {}x{} reference",
            reference.width(),
            reference.height()
        );
        let rf = &reference.data()[y * reference.stride() + x..];
        // SAFETY: the source covers 16x16 and the reference slice covers 16
        // rows of `16 + out.len() - 1` samples at its stride (checked above);
        // the table matches a supported ISA.
        unsafe {
            (self.pixel.sad_scan_16x16_even_rows)(
                src.data(),
                src.stride(),
                rf,
                reference.stride(),
                out,
            );
        }
    }

    /// Fold the SADs of every square block of the 64x64 block of `src`
    /// against the 64x64 block at `position` inside `reference` into
    /// `best` (see [`GridBest`] for the block order): where a block's SAD
    /// is below its best, the best becomes that SAD and its tag `tag`.
    /// 8x8 blocks with `visible[i] == 0` count as SAD zero (pass `u32::MAX`
    /// for blocks that count). `rows` selects exact or row-subsampled SADs.
    ///
    /// # Panics
    ///
    /// If `src` is smaller than 64x64 or the reference block leaves
    /// `reference`.
    pub fn sad_grid_best_64x64_at(
        &self,
        src: &PlaneView<'_>,
        reference: &PlaneView<'_>,
        position: (usize, usize),
        visible: &[u32; 64],
        rows: GridRows,
        best: &mut GridBest,
        tag: u32,
    ) {
        let size = BlockSize::new(64, 64).expect("64x64 is valid");
        check_covers(size, src, "source");
        let rf = block_slice(size, reference, position);
        // SAFETY: as for `sad_at`.
        unsafe {
            (self.pixel.sad_grid_best_64x64[rows as usize])(
                src.data(),
                src.stride(),
                rf,
                reference.stride(),
                visible,
                best,
                tag,
            );
        }
    }

    /// Evaluate four hierarchy-search candidates, retaining input-order ties.
    /// Source loads can be shared across candidates on supported ISAs.
    ///
    /// # Panics
    /// If the source or any reference position does not cover 64x64 pixels.
    pub fn sad_grid_best_x4_at(
        &self,
        src: &PlaneView<'_>,
        reference: &PlaneView<'_>,
        positions: [(usize, usize); 4],
        visible: &[u32; 64],
        rows: GridRows,
        best: &mut GridBest,
        tags: [u32; 4],
    ) {
        if let Some(kernels) = self.pixel.sad_grid_best_x4 {
            let size = BlockSize::new(64, 64).expect("64x64 is valid");
            check_covers(size, src, "source");
            let refs = positions.map(|position| block_slice(size, reference, position));
            // SAFETY: source and all four reference extents were checked above.
            unsafe {
                kernels[rows as usize](
                    src.data(),
                    src.stride(),
                    refs,
                    reference.stride(),
                    visible,
                    best,
                    tags,
                );
            }
        } else {
            for (position, tag) in positions.into_iter().zip(tags) {
                self.sad_grid_best_64x64_at(src, reference, position, visible, rows, best, tag);
            }
        }
    }

    /// SADs of the four 8x8 quadrants of a 16x16 block, with the reference
    /// block given as a position inside `reference` (see [`Dsp::sad_at`]):
    /// [top-left, top-right, bottom-left, bottom-right]. They sum to the
    /// 16x16 SAD.
    ///
    /// # Panics
    ///
    /// If `src` is smaller than 16x16 or the reference block leaves
    /// `reference`.
    #[must_use]
    pub fn sad_quads_16x16_at(
        &self,
        src: &PlaneView<'_>,
        reference: &PlaneView<'_>,
        position: (usize, usize),
    ) -> [u32; 4] {
        let size = BlockSize::new(16, 16).expect("16x16 is valid");
        check_covers(size, src, "source");
        let rf = block_slice(size, reference, position);
        // SAFETY: as for `sad_at`.
        unsafe { (self.pixel.sad_quads_16x16)(src.data(), src.stride(), rf, reference.stride()) }
    }

    /// Sum of absolute Hadamard-transformed differences (see the `pixel`
    /// module docs for tiling and normalisation).
    ///
    /// # Panics
    ///
    /// If either view is smaller than `size`.
    #[must_use]
    #[allow(
        clippy::inline_always,
        reason = "constant block sizes eliminate SATD dispatch and redundant view checks"
    )]
    #[inline(always)]
    // H264 legal two-clip probe: P -0.85%, key -0.38% instructions, exact bytes.
    pub fn satd(&self, size: BlockSize, src: &PlaneView<'_>, rf: &PlaneView<'_>) -> u32 {
        check_covers(size, src, "source");
        check_covers(size, rf, "reference");
        let (w, h) = (size.width(), size.height());
        let kernel = if w >= 8 && h >= 8 {
            self.pixel.satd_8x8
        } else {
            self.pixel.satd_4x4
        };
        // SAFETY: both views cover the block (checked above) and the table
        // matches a supported ISA (checked in `with_isa`).
        unsafe { kernel(src.data(), src.stride(), rf.data(), rf.stride(), w, h) }
    }

    /// Four independently normalized 8x8 SATDs of a 16x16 block, ordered
    /// top-left, top-right, bottom-left, bottom-right.
    ///
    /// # Panics
    ///
    /// If either view is smaller than 16x16.
    #[must_use]
    pub fn satd_8x8_quads(&self, src: &PlaneView<'_>, rf: &PlaneView<'_>) -> [u32; 4] {
        let size = BlockSize::new(16, 16).expect("16x16 is a kernel size");
        check_covers(size, src, "source");
        check_covers(size, rf, "reference");
        if let Some(kernel) = self.pixel.satd_8x8_quads {
            // SAFETY: both views cover 16x16 and dispatch supports this ISA.
            return unsafe { kernel(src.data(), src.stride(), rf.data(), rf.stride()) };
        }
        // Retain each ISA's existing 8x8 kernel until a batch is measured
        // faster there. Scalar dispatch is also the independent reference.
        let mut costs = [0; 4];
        for (q, cost) in costs.iter_mut().enumerate() {
            let (x, y) = (8 * (q % 2), 8 * (q / 2));
            // SAFETY: every quadrant lies within the checked 16x16 views.
            *cost = unsafe {
                (self.pixel.satd_8x8)(
                    &src.data()[y * src.stride() + x..],
                    src.stride(),
                    &rf.data()[y * rf.stride() + x..],
                    rf.stride(),
                    8,
                    8,
                )
            };
        }
        costs
    }

    /// [`Self::satd`] computed on 4x4 tiles whatever the block size (the
    /// cost of coding the block with 4x4 transforms, where `satd` estimates
    /// 8x8 transforms for blocks of 8x8 and larger).
    ///
    /// # Panics
    ///
    /// If either view is smaller than `size`.
    #[must_use]
    #[allow(
        clippy::inline_always,
        reason = "constant block sizes eliminate SATD dispatch and redundant view checks"
    )]
    #[inline(always)]
    pub fn satd_4x4_tiles(&self, size: BlockSize, src: &PlaneView<'_>, rf: &PlaneView<'_>) -> u32 {
        check_covers(size, src, "source");
        check_covers(size, rf, "reference");
        // SAFETY: as for `satd`; the 4x4-tile kernel accepts every size.
        unsafe {
            (self.pixel.satd_4x4)(
                src.data(),
                src.stride(),
                rf.data(),
                rf.stride(),
                size.width(),
                size.height(),
            )
        }
    }
}

impl Dsp {
    /// Sum and sum of squares of each 8x8 block along a row of `src`: block
    /// i covers columns 8i..8i+8 of the top 8 rows.
    ///
    /// # Panics
    ///
    /// If `src` is smaller than 8 rows of `8 * out.len()` samples.
    pub fn moments_8x8_row(&self, src: &PlaneView<'_>, out: &mut [Moments]) {
        check_covers_row(out.len(), src, "source");
        // SAFETY: the view covers the row of blocks (checked above) and the
        // table matches a supported ISA (checked in `with_isa`).
        unsafe { (self.sample.moments_8x8_row)(src.data(), src.stride(), out) }
    }

    /// Sum and sum of squares of each 8-wide, 4-tall block along a row of
    /// `src` (block i covers columns 8i..8i+8 of the top 4 rows): with a
    /// view at twice the plane's stride, the even rows of 8x8 blocks.
    ///
    /// # Panics
    ///
    /// If `src` is smaller than 4 rows of `8 * out.len()` samples.
    pub fn moments_8x4_row(&self, src: &PlaneView<'_>, out: &mut [Moments]) {
        assert!(
            src.width() >= 8 * out.len() && src.height() >= 4,
            "moments_8x4_row source smaller than its blocks"
        );
        // SAFETY: the view covers the row of blocks (checked above) and the
        // table matches a supported ISA (checked in `with_isa`).
        unsafe { (self.sample.moments_8x4_row)(src.data(), src.stride(), out) }
    }

    /// Fill all of `dst` from horizontal pairs of the same row of `src`
    /// (row y of `dst` from row y of `src`): each output the pair's rounded
    /// mean, or with `pick` its left sample. With a view of every other
    /// row, a 2x decimation.
    ///
    /// # Panics
    ///
    /// If `src` is narrower than twice `dst` or shorter than it.
    pub fn pairs_2x(&self, src: &PlaneView<'_>, dst: &mut PlaneViewMut<'_>, pick: bool) {
        let (w, h) = (dst.width(), dst.height());
        assert!(
            src.width() >= 2 * w && src.height() >= h,
            "pairs_2x source {}x{} does not cover the {w}x{h} destination",
            src.width(),
            src.height()
        );
        if w == 0 || h == 0 {
            return;
        }
        let dst_stride = dst.stride();
        let kernel = if pick {
            self.sample.pair_pick_2x
        } else {
            self.sample.pair_mean_2x
        };
        // SAFETY: the views cover the source and destination regions
        // (checked above and by the view constructors) and the table matches
        // a supported ISA.
        unsafe { kernel(src.data(), src.stride(), dst.data_mut(), dst_stride, w, h) };
    }

    /// Fill all of `dst` with the 2x downscale of the top-left
    /// `2 * dst.width()` x `2 * dst.height()` region of `src`: each output is
    /// the rounded mean of a 2x2 box.
    ///
    /// # Panics
    ///
    /// If `src` is smaller than twice `dst` in either dimension.
    pub fn downscale_2x(&self, src: &PlaneView<'_>, dst: &mut PlaneViewMut<'_>) {
        let (w, h) = (dst.width(), dst.height());
        assert!(
            src.width() >= 2 * w && src.height() >= 2 * h,
            "downscale_2x source {}x{} is smaller than twice the {w}x{h} destination",
            src.width(),
            src.height()
        );
        if w == 0 || h == 0 {
            return;
        }
        let dst_stride = dst.stride();
        // SAFETY: the views cover the source and destination regions
        // (checked above and by the view constructors) and the table matches
        // a supported ISA.
        unsafe {
            (self.sample.downscale_2x)(src.data(), src.stride(), dst.data_mut(), dst_stride, w, h);
        }
    }
}

impl Dsp {
    /// Single-reference prediction of a `size` block into `dst` with one
    /// codec's interpolation `filter` (`mc` module docs). `window` starts at
    /// the reference sample under the block's top-left corner, displaced by
    /// the motion vector's whole-sample part, minus [`Filter::before`] in
    /// each direction; `frac_x` and `frac_y` are the vector's sub-sample
    /// parts in the filter's units ([`Filter::positions`]).
    ///
    /// # Panics
    ///
    /// If `window` is smaller than `w + taps - 1` by `h + taps - 1`, `dst` is
    /// smaller than `size`, or an offset is out of range.
    pub fn predict(
        &self,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_>,
        frac_x: usize,
        frac_y: usize,
        dst: &mut PlaneViewMut<'_>,
    ) {
        let (w, h) = (size.width(), size.height());
        let extra = filter.taps() - 1;
        assert!(
            window.width() >= w + extra && window.height() >= h + extra,
            "{filter:?} prediction window {}x{} is smaller than {}x{}",
            window.width(),
            window.height(),
            w + extra,
            h + extra
        );
        assert!(
            frac_x < filter.positions() && frac_y < filter.positions(),
            "{filter:?} sub-sample offsets ({frac_x}, {frac_y}) out of range"
        );
        check_covers(size, &dst.as_view(), "destination");
        let dst_stride = dst.stride();
        // SAFETY: the views cover the window and the block (checked above);
        // the table matches a supported ISA (checked in `with_isa`).
        unsafe {
            (self.mc.table(filter)[size.width_class()])(
                window.data(),
                window.stride(),
                dst.data_mut(),
                dst_stride,
                h,
                frac_x,
                frac_y,
            );
        }
    }

    /// [`Self::predict`] on sample buffers instead of views: `src` starts
    /// at the window's top-left with rows `src_stride` apart, `dst` at the
    /// block's with rows `dst_stride` apart. One footprint check replaces
    /// the views' construction: for a decoder's per-block hot path.
    ///
    /// # Panics
    ///
    /// If `src` does not cover the `w + taps - 1` by `h + taps - 1` window,
    /// `dst` the block, or an offset is out of range.
    #[inline]
    pub fn predict_samples(
        &self,
        filter: Filter,
        size: BlockSize,
        (src, src_stride): (&[u8], usize),
        (frac_x, frac_y): (usize, usize),
        (dst, dst_stride): (&mut [u8], usize),
    ) {
        assert!(
            sample_buffers_cover(
                filter,
                size,
                (src.len(), src_stride),
                (frac_x, frac_y),
                (dst.len(), dst_stride)
            ),
            "AV1 prediction buffers do not cover the block, or offsets out of range"
        );
        // SAFETY: the buffers cover the window and the block (checked
        // above); the table matches a supported ISA (checked in `with_isa`).
        unsafe {
            (self.mc.table(filter)[size.width_class()])(
                src,
                src_stride,
                dst,
                dst_stride,
                size.height(),
                frac_x,
                frac_y,
            );
        }
    }

    /// [`Self::predict_compound`] on sample buffers, as
    /// [`Self::predict_samples`].
    ///
    /// # Panics
    ///
    /// As [`Self::predict_samples`], or if `filter` has no compound
    /// prediction.
    #[inline]
    pub fn predict_compound_samples(
        &self,
        filter: Filter,
        size: BlockSize,
        (src, src_stride): (&[u8], usize),
        (frac_x, frac_y): (usize, usize),
        (dst, dst_stride): (&mut [i16], usize),
    ) {
        let kernels = self
            .mc
            .compound
            .prep(filter)
            .unwrap_or_else(|| panic!("{filter:?} has no compound prediction"));
        assert!(
            sample_buffers_cover(
                filter,
                size,
                (src.len(), src_stride),
                (frac_x, frac_y),
                (dst.len(), dst_stride)
            ),
            "AV1 prediction buffers do not cover the block, or offsets out of range"
        );
        // SAFETY: the buffers cover the window and the block (checked
        // above); the table matches a supported ISA (checked in `with_isa`).
        unsafe {
            (kernels[size.width_class()])(
                src,
                src_stride,
                dst,
                dst_stride,
                size.height(),
                frac_x,
                frac_y,
            );
        }
    }

    /// [`Self::predict`] of a block whose `size` sub-blocks each have their
    /// own vector (H.266 affine motion, §8.5.6.3), in one call: sub-block
    /// `i` of `motion` (raster order, `per_row` to a row) is predicted from
    /// `window` at its [`SubBlockMotion`] into its place in `dst`. The
    /// batch is checked once and each sub-block goes straight to the ISA's
    /// kernel, without the per-call checks and window construction a
    /// [`Self::predict`] per sub-block costs (on 4x4 sub-blocks, as much as
    /// the filtering).
    ///
    /// # Panics
    ///
    /// If a sub-block's window leaves `window`, an offset is out of range,
    /// `per_row` is zero or does not divide `motion`, or `dst` is smaller
    /// than the block.
    pub fn predict_sub_blocks(
        &self,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_>,
        motion: &[SubBlockMotion],
        per_row: usize,
        dst: &mut PlaneViewMut<'_>,
    ) {
        let (w, h) = (size.width(), size.height());
        let extra = filter.taps() - 1;
        assert!(
            per_row > 0 && motion.len().is_multiple_of(per_row),
            "{} sub-blocks do not fill rows of {per_row}",
            motion.len()
        );
        for m in motion {
            assert!(
                usize::from(m.x) + w + extra <= window.width()
                    && usize::from(m.y) + h + extra <= window.height(),
                "{filter:?} sub-block window at ({}, {}) leaves the {}x{} window",
                m.x,
                m.y,
                window.width(),
                window.height()
            );
            assert!(
                usize::from(m.frac_x) < filter.positions()
                    && usize::from(m.frac_y) < filter.positions(),
                "{filter:?} sub-sample offsets ({}, {}) out of range",
                m.frac_x,
                m.frac_y
            );
        }
        let rows = motion.len() / per_row;
        assert!(
            dst.width() >= w * per_row && dst.height() >= h * rows,
            "destination {}x{} is smaller than {per_row}x{rows} sub-blocks of {w}x{h}",
            dst.width(),
            dst.height()
        );
        let kernel = self.mc.table(filter)[size.width_class()];
        let (src, src_stride) = (window.data(), window.stride());
        let dst_stride = dst.stride();
        let out = dst.data_mut();
        for (i, m) in motion.iter().enumerate() {
            let from = usize::from(m.y) * src_stride + usize::from(m.x);
            let to = (i / per_row) * h * dst_stride + (i % per_row) * w;
            // SAFETY: the sub-block's window lies inside `window` and its
            // place inside `dst` (checked above), and the table matches a
            // supported ISA (checked in `with_isa`).
            unsafe {
                kernel(
                    &src[from..],
                    src_stride,
                    &mut out[to..],
                    dst_stride,
                    h,
                    usize::from(m.frac_x),
                    usize::from(m.frac_y),
                );
            }
        }
    }

    /// Compound-precision prediction of a `size` block into `dst` (`h`
    /// rows at `dst_stride`): [`Self::predict`]'s contract with unclipped
    /// `i16` outputs. With an AV1 filter, `InterRound1` = 7 (AV1
    /// §7.11.3.4), one of the two predictions [`Self::average_compound`]
    /// combines; with an HEVC filter, `predSamplesLX - 8192` (HEVC
    /// §8.5.3.3.3), centered to preserve filter overshoot in i16, which
    /// [`Self::hevc_weighted_prediction_centered`] combines.
    ///
    /// # Panics
    ///
    /// If `filter` is not an AV1 or HEVC filter, or as [`Self::predict`].
    pub fn predict_compound(
        &self,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_>,
        (frac_x, frac_y): (usize, usize),
        dst: &mut [i16],
        dst_stride: usize,
    ) {
        let kernels = self
            .mc
            .compound
            .prep(filter)
            .unwrap_or_else(|| panic!("{filter:?} has no compound prediction"));
        let (w, h) = (size.width(), size.height());
        let extra = filter.taps() - 1;
        assert!(
            window.width() >= w + extra && window.height() >= h + extra,
            "{filter:?} compound window {}x{} is smaller than {}x{}",
            window.width(),
            window.height(),
            w + extra,
            h + extra
        );
        assert!(
            frac_x < filter.positions() && frac_y < filter.positions(),
            "{filter:?} sub-sample offsets ({frac_x}, {frac_y}) out of range"
        );
        check_covers_i16(size, dst.len(), dst_stride, "compound prediction");
        // SAFETY: the window and destination cover the block (checked
        // above); the table matches a supported ISA (checked in `with_isa`).
        unsafe {
            (kernels[size.width_class()])(
                window.data(),
                window.stride(),
                dst,
                dst_stride,
                h,
                frac_x,
                frac_y,
            );
        }
    }

    /// Whether the ISA has AV1 kernels for blocks whose directions use
    /// different filter families ([`Self::av1_put_mixed`]).
    pub(crate) fn has_av1_mixed(&self) -> bool {
        self.mc.av1_mixed.is_some()
    }

    /// AV1 prediction of a `size` block whose directions use different
    /// filter families (`filters`, 8-tap families; `dual_filter`) at 1/16
    /// phases `fracs`, final 8-bit samples into `dst`: [`Self::predict`]'s
    /// contract, through the ISA's mixed kernels (`mc::Av1MixedFn`).
    ///
    /// # Panics
    ///
    /// As [`Self::predict`], or if the ISA has no mixed kernels.
    pub(crate) fn av1_put_mixed(
        &self,
        filters: Av1Filters,
        size: BlockSize,
        window: &PlaneView<'_>,
        fracs: (usize, usize),
        dst: &mut PlaneViewMut<'_>,
    ) {
        let kernel = self.av1_mixed().put[size.width_class()];
        check_av1_mixed_window(size, window, fracs);
        check_covers(size, &dst.as_view(), "destination");
        let dst_stride = dst.stride();
        let taps = av1_mixed_taps(filters, size, fracs);
        // SAFETY: the views cover the window and the block (checked above);
        // the table matches a supported ISA (checked in `with_isa`).
        unsafe {
            kernel(
                window.data(),
                window.stride(),
                dst.data_mut(),
                dst_stride,
                size.height(),
                fracs,
                taps,
            );
        }
    }

    /// [`Self::av1_put_mixed`] at compound precision into `dst` (`h` rows
    /// at `dst_stride`), as [`Self::predict_compound`].
    ///
    /// # Panics
    ///
    /// As [`Self::av1_put_mixed`], or if `dst` is too short.
    pub(crate) fn av1_prep_mixed(
        &self,
        filters: Av1Filters,
        size: BlockSize,
        window: &PlaneView<'_>,
        fracs: (usize, usize),
        dst: &mut [i16],
        dst_stride: usize,
    ) {
        let kernel = self.av1_mixed().prep[size.width_class()];
        check_av1_mixed_window(size, window, fracs);
        check_covers_i16(size, dst.len(), dst_stride, "compound prediction");
        let taps = av1_mixed_taps(filters, size, fracs);
        // SAFETY: the window and destination cover the block (checked
        // above); the table matches a supported ISA (checked in `with_isa`).
        unsafe {
            kernel(
                window.data(),
                window.stride(),
                dst,
                dst_stride,
                size.height(),
                fracs,
                taps,
            );
        }
    }

    fn av1_mixed(&self) -> &mc::Av1MixedFns {
        self.mc
            .av1_mixed
            .as_ref()
            .expect("the ISA has mixed-family AV1 kernels")
    }

    /// HEVC default-weight bi-prediction of a `size` block (HEVC eq. 8-251,
    /// 8-bit): the second list's prediction from `window` (as
    /// [`Self::predict_compound`] places and filters it) averaged with the
    /// first list's centred prediction `first` (`size` at `first_stride`,
    /// from [`Self::predict_compound`]) into `dst`. The same samples as
    /// [`Self::hevc_weighted_prediction_centered`] with default weights,
    /// without the second prediction's 16-bit store.
    ///
    /// # Panics
    ///
    /// If `filter` is not an HEVC filter, or a buffer does not cover the
    /// block, as for [`Self::predict_compound`].
    #[allow(clippy::too_many_arguments)]
    pub fn hevc_predict_bi(
        &self,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_>,
        (frac_x, frac_y): (usize, usize),
        (first, first_stride): (&[i16], usize),
        dst: &mut PlaneViewMut<'_>,
    ) {
        let kernels = self
            .mc
            .compound
            .hevc_bi(filter)
            .unwrap_or_else(|| panic!("{filter:?} has no HEVC bi-prediction"));
        let (w, h) = (size.width(), size.height());
        let extra = filter.taps() - 1;
        assert!(
            window.width() >= w + extra && window.height() >= h + extra,
            "{filter:?} bi-prediction window {}x{} is smaller than {}x{}",
            window.width(),
            window.height(),
            w + extra,
            h + extra
        );
        assert!(
            frac_x < filter.positions() && frac_y < filter.positions(),
            "{filter:?} sub-sample offsets ({frac_x}, {frac_y}) out of range"
        );
        check_covers_i16(size, first.len(), first_stride, "first prediction");
        check_covers(size, &dst.as_view(), "destination");
        let dst_stride = dst.stride();
        // SAFETY: the window, first prediction and destination cover the
        // block (checked above); the table matches a supported ISA.
        unsafe {
            (kernels[size.width_class()])(
                window.data(),
                window.stride(),
                first,
                first_stride,
                dst.data_mut(),
                dst_stride,
                h,
                frac_x,
                frac_y,
            );
        }
    }

    /// The equal-weight average of two compound predictions of a `size`
    /// block (`p0`, `p1` at `src_stride`), rounded and clipped into `dst`
    /// (AV1 §7.11.3.15).
    ///
    /// # Panics
    ///
    /// If a buffer does not cover the block.
    pub fn average_compound(
        &self,
        size: BlockSize,
        (p0, p1): (&[i16], &[i16]),
        src_stride: usize,
        dst: &mut PlaneViewMut<'_>,
    ) {
        check_covers_i16(size, p0.len(), src_stride, "first compound prediction");
        check_covers_i16(size, p1.len(), src_stride, "second compound prediction");
        check_covers(size, &dst.as_view(), "destination");
        let dst_stride = dst.stride();
        // SAFETY: every buffer covers the block (checked above); the table
        // matches a supported ISA.
        unsafe {
            (self.mc.compound.average[size.width_class()])(
                p0,
                p1,
                src_stride,
                dst.data_mut(),
                dst_stride,
                size.height(),
            );
        }
    }

    /// A prediction kernel for one H.264 filter and block size, resolved
    /// once (no per-call dispatch, size checks or view construction): for
    /// the per-macroblock hot path.
    ///
    /// # Panics
    ///
    /// If `filter` is not one of the H.264 filters.
    #[must_use]
    pub fn h264_predictor(&self, filter: Filter, size: BlockSize) -> H264Predictor {
        assert!(
            matches!(filter, Filter::H264Luma | Filter::H264Chroma),
            "{filter:?} is not an H.264 filter"
        );
        let shape = (size.width(), size.height());
        let kernels = self
            .mc
            .h264_phases
            .and_then(|phases| phases.slots(filter, shape))
            .unwrap_or([self.mc.table(filter)[size.width_class()]; mc::H264_PHASE_SLOTS]);
        H264Predictor {
            kernels,
            height: size.height(),
            positions: filter.positions(),
            slot_cap: if filter == Filter::H264Luma {
                mc::H264_LUMA_PHASE_CAP
            } else {
                mc::H264_CHROMA_CLASS_CAP
            },
        }
    }

    /// `src - pred` of a `size` block into `out` (row-major at `out_stride`).
    ///
    /// # Panics
    ///
    /// If a view is smaller than `size`, `out_stride < w`, or `out` does not
    /// cover `h` rows.
    pub fn residual(
        &self,
        size: BlockSize,
        src: &PlaneView<'_>,
        pred: &PlaneView<'_>,
        out: &mut [i16],
        out_stride: usize,
    ) {
        check_covers(size, src, "source");
        check_covers(size, pred, "prediction");
        check_covers_i16(size, out.len(), out_stride, "residual");
        // SAFETY: every slice covers the block (checked above); the table
        // matches a supported ISA.
        unsafe {
            (self.diff.residual[size.width_class()])(
                src.data(),
                src.stride(),
                pred.data(),
                pred.stride(),
                out,
                out_stride,
                size.height(),
            );
        }
    }

    /// Write the residual and return its absolute sum without rereading it.
    /// The sum is at most `64 * 64 * 255` for supported 8-bit blocks.
    ///
    /// # Panics
    ///
    /// If a view is smaller than `size`, `out_stride < w`, or `out` does not
    /// cover `h` rows.
    pub fn residual_sad(
        &self,
        size: BlockSize,
        src: &PlaneView<'_>,
        pred: &PlaneView<'_>,
        out: &mut [i16],
        out_stride: usize,
    ) -> u32 {
        check_covers(size, src, "source");
        check_covers(size, pred, "prediction");
        check_covers_i16(size, out.len(), out_stride, "residual");
        // SAFETY: every slice covers the block (checked above); the table
        // matches a supported ISA.
        unsafe {
            (self.diff.residual_sad[size.width_class()])(
                src.data(),
                src.stride(),
                pred.data(),
                pred.stride(),
                out,
                out_stride,
                size.height(),
            )
        }
    }

    /// `dst = clamp(dst + residual, 0, 255)` over a `size` block. Residual
    /// values must lie within `±(2^15 - 256)` (any inverse transform output
    /// does); larger ones give unspecified (but memory-safe) results.
    ///
    /// # Panics
    ///
    /// If `dst` is smaller than `size`, `residual_stride < w`, or `residual`
    /// does not cover `h` rows.
    pub fn add_residual(
        &self,
        size: BlockSize,
        dst: &mut PlaneViewMut<'_>,
        residual: &[i16],
        residual_stride: usize,
    ) {
        check_covers(size, &dst.as_view(), "destination");
        check_covers_i16(size, residual.len(), residual_stride, "residual");
        let dst_stride = dst.stride();
        // SAFETY: as for `residual`.
        unsafe {
            (self.diff.add_residual[size.width_class()])(
                dst.data_mut(),
                dst_stride,
                residual,
                residual_stride,
                size.height(),
            );
        }
    }

    /// Add a packed i32 residual (or DC constant) to prediction and store clipped
    /// pixels directly into the destination plane, without an intermediate block.
    ///
    /// # Panics
    /// If either view or the residual does not cover `size`.
    pub fn reconstruct_i32(
        &self,
        size: BlockSize,
        prediction: &PlaneView<'_>,
        residual: crate::Residual32<'_>,
        dst: &mut PlaneViewMut<'_>,
    ) {
        check_covers(size, prediction, "prediction");
        check_covers(size, &dst.as_view(), "destination");
        if let crate::Residual32::Samples(r) = residual {
            assert!(r.len() >= size.width() * size.height());
        }
        let ds = dst.stride();
        // SAFETY: supported ISA and all row/sample spans checked above.
        unsafe {
            (self.reconstruct)(
                prediction.data(),
                prediction.stride(),
                residual,
                dst.data_mut(),
                ds,
                size.width(),
                size.height(),
            );
        }
    }

    /// AV1 CDEF direction search (§7.15.1, libaom `cdef_find_dir`) on the
    /// 8x8 block of `plane` whose top-left sample is (`x`, `y`): the
    /// block's direction (0..=7) and its directional contrast (`var`, the
    /// input of libaom's `adjust_strength`).
    ///
    /// # Panics
    ///
    /// If the block does not lie inside `plane`.
    #[must_use]
    pub fn cdef_find_dir(&self, plane: &PlaneView<'_>, (x, y): (usize, usize)) -> (u8, i32) {
        const SIDE: usize = 8;
        assert!(
            x + SIDE <= plane.width() && y + SIDE <= plane.height(),
            "CDEF direction block outside the plane"
        );
        let stride = plane.stride();
        // SAFETY: the table matches this CPU (built after `is_supported`);
        // the kernel indexes its slice, which starts at the block.
        unsafe { (self.cdef.find_dir)(&plane.data()[y * stride + x..], stride) }
    }

    /// AV1 CDEF (§7.15.2) on the `size` x `size` block (`size` 8 for luma,
    /// 4 for 4:2:0 chroma) whose top-left sample is (`x`, `y`): read from the
    /// unfiltered `src`, written to the same position of `dst`, which views
    /// the whole plane. `pri` is the (luma-adjusted) primary strength
    /// (0..=15), `sec` the secondary strength (0, 1, 2 or 4), `dir` the
    /// block's direction (0..=7) and `damping` the plane's damping
    /// (`CdefDamping`, one less for chroma). Bit-exact with libaom.
    ///
    /// # Panics
    ///
    /// If `size` is not 4 or 8, a strength is out of range, the block does
    /// not lie inside `src`, or `dst` is smaller than `src`.
    #[allow(clippy::too_many_arguments)]
    pub fn cdef_filter(
        &self,
        size: usize,
        src: &CdefSource,
        (x, y): (usize, usize),
        (pri, sec): (u8, u8),
        dir: u8,
        damping: u8,
        dst: &mut PlaneViewMut<'_>,
    ) {
        let block = CdefBlock { x, y, pri, dir };
        self.cdef_filter_blocks(size, src, &[block], sec, damping, dst);
    }

    /// [`Dsp::cdef_filter`] on every block of `blocks`, which share `size`,
    /// the secondary strength and the damping: one call per plane, so the
    /// argument checks and strength set-up are paid once. Where the ISA has
    /// a two-block 4x4 kernel, interior 4x4 blocks of equal primary strength
    /// are filtered two at a time (each block is independent: it reads only
    /// `src` and writes only its own samples).
    ///
    /// # Panics
    ///
    /// As [`Dsp::cdef_filter`], for any block.
    pub fn cdef_filter_blocks(
        &self,
        size: usize,
        src: &CdefSource,
        blocks: &[CdefBlock],
        sec: u8,
        damping: u8,
        dst: &mut PlaneViewMut<'_>,
    ) {
        match self.cdef.block4_interior_pair {
            Some(pair) if size == 4 => {
                self.cdef_filter_block_pairs(pair, src, blocks, sec, damping, dst);
            }
            _ => self.cdef_filter_each_block(size, src, blocks, sec, damping, dst),
        }
    }

    /// [`Dsp::cdef_filter_blocks`] of 4x4 blocks with the two-block kernel
    /// `pair`: consecutive interior blocks of equal primary strength go
    /// through it; edge blocks and any left without a partner go one at a
    /// time.
    fn cdef_filter_block_pairs(
        &self,
        pair: cdef::CdefPairFn8,
        src: &CdefSource,
        blocks: &[CdefBlock],
        sec: u8,
        damping: u8,
        dst: &mut PlaneViewMut<'_>,
    ) {
        const SIZE: usize = 4;
        assert!(matches!(sec, 0 | 1 | 2 | 4), "CDEF strength out of range");
        assert!(
            dst.width() >= src.width() && dst.height() >= src.height(),
            "CDEF destination smaller than the plane"
        );
        let reach = crate::CDEF_PADDING;
        let (width, height) = (src.width(), src.height());
        let filled_rows = src.filled_rows();
        let dst_stride = dst.stride();
        let mut by_strength: [Option<CdefArgs>; 16] = [None; 16];
        let mut singles = Vec::new();
        let mut waiting: Option<CdefBlock> = None;
        let dst_data = dst.data_mut();
        for &block in blocks {
            let (x, y) = (block.x, block.y);
            let interior =
                x >= reach && y >= reach && x + SIZE + reach <= width && y + SIZE + reach <= height;
            if !interior {
                singles.push(block);
                continue;
            }
            match waiting.take() {
                Some(first) if first.pri == block.pri => {
                    assert!(block.pri <= 15, "CDEF strength out of range");
                    assert!(
                        first.y.max(y) + SIZE + reach <= filled_rows,
                        "CDEF block reads source rows not copied yet"
                    );
                    let args = by_strength[usize::from(block.pri)]
                        .get_or_insert_with(|| CdefArgs::new(block.pri, sec, 0, damping));
                    let base = dst_data.as_mut_ptr();
                    // SAFETY: the table matches this CPU (built after
                    // `is_supported`); both blocks are interior, so every
                    // tap lies inside the plane rows `src.samples()` holds
                    // (checked above, and by the kernel's window checks).
                    // Both blocks lie inside the plane `dst_data` views
                    // (`dst` is at least the source's size) and are
                    // distinct 4x4 blocks of the list, so the two
                    // destinations do not overlap and `dst_data` is not
                    // otherwise used during the call.
                    unsafe {
                        pair(
                            [src.samples(); 2],
                            width,
                            [src.index(first.x, first.y), src.index(x, y)],
                            args,
                            [usize::from(first.dir), usize::from(block.dir)],
                            [
                                base.add(first.y * dst_stride + first.x),
                                base.add(y * dst_stride + x),
                            ],
                            dst_stride,
                        );
                    }
                }
                unpaired => {
                    singles.extend(unpaired);
                    waiting = Some(block);
                }
            }
        }
        singles.extend(waiting);
        self.cdef_filter_each_block(SIZE, src, &singles, sec, damping, dst);
    }

    /// [`Dsp::cdef_filter_blocks`] one block at a time.
    fn cdef_filter_each_block(
        &self,
        size: usize,
        src: &CdefSource,
        blocks: &[CdefBlock],
        sec: u8,
        damping: u8,
        dst: &mut PlaneViewMut<'_>,
    ) {
        assert!(matches!(sec, 0 | 1 | 2 | 4), "CDEF strength out of range");
        // Strength-derived arguments, by primary strength.
        let mut by_strength: [Option<CdefArgs>; 16] = [None; 16];
        for block in blocks {
            assert!(block.pri <= 15, "CDEF strength out of range");
            let args = by_strength[usize::from(block.pri)]
                .get_or_insert_with(|| CdefArgs::new(block.pri, sec, 0, damping))
                .with_dir(block.dir);
            self.cdef_filter_block_at(size, src, (block.x, block.y), &args, dst);
        }
    }

    /// The CDEF kernels of this instance's ISA.
    pub(crate) fn cdef_kernels(&self) -> &CdefFns {
        &self.cdef
    }

    /// CDEF of the `size` x `size` block (8 or 4) at (`x`, `y`) of `dst`
    /// from its unfiltered rows in `src`, with resolved arguments `args`:
    /// the 8-bit kernel when every tap is inside the plane, the padded
    /// 16-bit one otherwise.
    ///
    /// # Panics
    ///
    /// If `size` is not 4 or 8, the block is not inside the plane, `src`
    /// does not hold the rows it reads, or `dst` is smaller than `src`.
    #[inline]
    pub(crate) fn cdef_filter_block_at(
        &self,
        size: usize,
        src: &CdefSource,
        (x, y): (usize, usize),
        args: &CdefArgs,
        dst: &mut PlaneViewMut<'_>,
    ) {
        let (interior_kernel, edge_kernel) = match size {
            8 => (self.cdef.block8_interior, self.cdef.block8),
            4 => (self.cdef.block4_interior, self.cdef.block4),
            _ => panic!("CDEF blocks are 8x8 or 4x4, not {size}"),
        };
        let reach = crate::CDEF_PADDING;
        let (width, height) = (src.width(), src.height());
        assert!(
            x + size <= width
                && y + size <= height
                && dst.width() >= width
                && dst.height() >= height,
            "CDEF block outside the plane"
        );
        assert!(
            (y + size + reach).min(height) <= src.filled_rows(),
            "CDEF block reads source rows not copied yet"
        );
        let dst_stride = dst.stride();
        let out = &mut dst.data_mut()[y * dst_stride + x..];
        if cdef::band::is_interior(src, (x, y), size) {
            // Every tap is inside the plane: the 8-bit kernel reads it
            // directly.
            // SAFETY: the table matches this CPU (built after
            // `is_supported`); every tap lies inside the plane, in rows
            // `src.samples()` holds (checked above; `index` and the kernel's
            // window check catch dropped rows), and `out` starts at the
            // block, which lies inside the plane `dst` views.
            unsafe {
                interior_kernel(src.samples(), width, src.index(x, y), args, out, dst_stride);
            }
        } else {
            // Near an edge: a padded 16-bit copy of the block's window
            // marks the samples outside the plane (`cdef` module docs).
            let side = size + 2 * reach;
            let window = src.padded_window(x, y, size);
            // SAFETY: as above; the window holds the block plus `reach`
            // samples on every side.
            unsafe {
                edge_kernel(
                    &window[..side * side],
                    side,
                    reach * side + reach,
                    args,
                    out,
                    dst_stride,
                );
            }
        }
    }

    /// [`Dsp::cdef_filter`] reading the unfiltered 8-bit plane `src`
    /// directly, for a block whose taps all lie inside it: at least
    /// [`crate::CDEF_PADDING`] samples from every edge of `src`. Bit-exact
    /// with [`Dsp::cdef_filter`] on such blocks.
    ///
    /// # Panics
    ///
    /// If `size` is not 4 or 8, a strength is out of range, the block is
    /// closer than [`crate::CDEF_PADDING`] to an edge of `src`, or `dst` is
    /// smaller than `src`.
    #[allow(clippy::too_many_arguments)]
    pub fn cdef_filter_interior(
        &self,
        size: usize,
        src: &PlaneView<'_>,
        (x, y): (usize, usize),
        (pri, sec): (u8, u8),
        dir: u8,
        damping: u8,
        dst: &mut PlaneViewMut<'_>,
    ) {
        let kernel = match size {
            8 => self.cdef.block8_interior,
            4 => self.cdef.block4_interior,
            _ => panic!("CDEF blocks are 8x8 or 4x4, not {size}"),
        };
        assert!(
            pri <= 15 && matches!(sec, 0 | 1 | 2 | 4),
            "CDEF strength out of range"
        );
        let reach = crate::CDEF_PADDING;
        assert!(
            x >= reach
                && y >= reach
                && x + size + reach <= src.width()
                && y + size + reach <= src.height(),
            "CDEF block within {reach} samples of the plane's edge"
        );
        assert!(
            dst.width() >= src.width() && dst.height() >= src.height(),
            "CDEF destination smaller than the plane"
        );
        let args = CdefArgs::new(pri, sec, dir, damping);
        let dst_stride = dst.stride();
        let dst_data = &mut dst.data_mut()[y * dst_stride + x..];
        // SAFETY: the table matches this CPU (built after `is_supported`);
        // every tap of the block lies inside `src` (checked above), and
        // `dst_data` starts at the block, which lies inside the plane `dst`
        // views.
        unsafe {
            kernel(
                src.data(),
                src.stride(),
                y * src.stride() + x,
                &args,
                dst_data,
                dst_stride,
            );
        }
    }

    /// HEVC SAO edge-offset statistics (`sao` module docs) of class `class`
    /// (0..=3) over the samples `x0..x1` x `y0..y1` of the reconstruction
    /// `rec`, against `src`. Both views are whole planes of equal size.
    ///
    /// # Panics
    ///
    /// If the views differ in size, `class > 3`, or a sample of the
    /// rectangle has a neighbour outside the plane.
    #[must_use]
    pub fn sao_edge_stats(
        &self,
        class: u8,
        src: &PlaneView<'_>,
        rec: &PlaneView<'_>,
        (x0, y0, x1, y1): (usize, usize, usize, usize),
    ) -> SaoStats {
        assert!(
            src.width() == rec.width() && src.height() == rec.height(),
            "SAO planes differ in size"
        );
        let rect = sao_rect(class, rec.width(), rec.height(), (x0, y0, x1, y1));
        let offs = sao_offsets(class, rec.stride());
        // SAFETY: the table matches this CPU; `sao_rect` checked that the
        // rectangle and every neighbour lie inside both planes.
        unsafe {
            (self.sao.stats)(
                src.data(),
                src.stride(),
                rec.data(),
                rec.stride(),
                offs,
                rect,
            )
        }
    }

    /// Estimate SAO statistics from one row per group, weighted by group size.
    /// Neighbours retain their original row positions, so vertical and diagonal
    /// edge classification is unchanged. A step of1 processes every sample.
    ///
    /// # Panics
    /// As [`Self::sao_edge_stats`], or if `row_step` is outside 1..=4.
    #[must_use]
    pub fn sao_edge_stats_sampled(
        &self,
        class: u8,
        src: &PlaneView<'_>,
        rec: &PlaneView<'_>,
        bounds: (usize, usize, usize, usize),
        row_step: usize,
    ) -> SaoStats {
        assert!((1..=4).contains(&row_step));
        assert!(src.width() == rec.width() && src.height() == rec.height());
        let mut rect = sao_rect(class, rec.width(), rec.height(), bounds);
        rect.row_step = row_step;
        let end = rect.y1;
        let remainder = (end - rect.y0) % row_step;
        rect.y1 -= remainder;
        let offs = sao_offsets(class, rec.stride());
        // SAFETY: the CPU table matches this machine and sao_rect validated
        // every neighbour. Sampling visits a subset of that checked rectangle.
        let mut out = unsafe {
            (self.sao.stats)(
                src.data(),
                src.stride(),
                rec.data(),
                rec.stride(),
                offs,
                rect,
            )
        };
        for c in 0..4 {
            out.sum[c] *= row_step as i64;
            out.count[c] *= row_step as i64;
        }
        if remainder != 0 {
            rect.y0 = rect.y1;
            rect.y1 = rect.y0 + 1;
            // SAFETY: this final sampled row is inside the same checked window.
            let last = unsafe {
                (self.sao.stats)(
                    src.data(),
                    src.stride(),
                    rec.data(),
                    rec.stride(),
                    offs,
                    rect,
                )
            };
            for c in 0..4 {
                out.sum[c] += last.sum[c] * remainder as i64;
                out.count[c] += last.count[c] * remainder as i64;
            }
        }
        out
    }

    /// Apply four signed SAO band offsets over an exclusive-end rectangle.
    /// Shared 8-bit HEVC/VVC arithmetic with a wrapping five-bit band index.
    /// # Panics
    /// If the band position or rectangle is out of range.
    pub fn sao_band_apply(
        &self,
        position: u8,
        offsets: [i8; 4],
        plane: &mut PlaneViewMut<'_>,
        bounds: (usize, usize, usize, usize),
    ) {
        self.sao_band_apply_masked(position, offsets, plane, bounds, &[]);
    }

    /// Apply SAO band offsets while leaving PCM-exempt samples unchanged.
    /// `exempt` is empty or tightly packed in rectangle row order; true skips
    /// a sample. The rectangle uses exclusive end points.
    /// # Panics
    /// If the position, rectangle, or exemption mask length is invalid.
    pub fn sao_band_apply_masked(
        &self,
        position: u8,
        offsets: [i8; 4],
        plane: &mut PlaneViewMut<'_>,
        (x0, y0, x1, y1): (usize, usize, usize, usize),
        exempt: &[bool],
    ) {
        assert!(
            position < 32 && x0 <= x1 && x1 <= plane.width() && y0 <= y1 && y1 <= plane.height()
        );
        assert!(exempt.is_empty() || exempt.len() == (x1 - x0) * (y1 - y0));
        let stride = plane.stride();
        let rect = Rect {
            x0,
            y0,
            x1,
            y1,
            row_step: 1,
        };
        // SAFETY: plane construction validates backing storage and stride;
        // the rectangle and packed mask are checked above, ISA by dispatch.
        unsafe {
            (self.sao.band)(plane.data_mut(), stride, rect, position, offsets, exempt);
        }
    }

    /// Apply HEVC SAO edge offsets of class `class` (0..=3) to the samples
    /// `x0..x1` x `y0..y1`: classify from `input` (the deblocked plane) and
    /// write `clamp(sample + offsets[category - 1])` to `out`. Both views are
    /// whole planes of equal size.
    ///
    /// # Panics
    ///
    /// As [`Self::sao_edge_stats`].
    pub fn sao_edge_apply(
        &self,
        class: u8,
        offsets: [i8; 4],
        input: &PlaneView<'_>,
        out: &mut PlaneViewMut<'_>,
        (x0, y0, x1, y1): (usize, usize, usize, usize),
    ) {
        assert!(
            input.width() == out.width() && input.height() == out.height(),
            "SAO planes differ in size"
        );
        let rect = sao_rect(class, input.width(), input.height(), (x0, y0, x1, y1));
        let offs = sao_offsets(class, input.stride());
        let out_stride = out.stride();
        // SAFETY: as for `sao_edge_stats`.
        unsafe {
            (self.sao.apply)(
                input.data(),
                input.stride(),
                out.data_mut(),
                out_stride,
                offs,
                offsets,
                rect,
            );
        }
    }

    /// Filter an AV1 loop-filter edge (§7.14.6): the `lines` lines (a
    /// multiple of [`AV1_EDGE_LINES`]) across the edge whose first q0
    /// sample is (`x`, `y`) of `dst`, all with `filter`. `vertical` edges
    /// separate horizontally adjacent samples. The kernel processes lines
    /// in batches of 16-bit lanes (portable code the compiler vectorises).
    ///
    /// # Panics
    ///
    /// If a sample the filter reads (its length's reach on each side) lies
    /// outside the plane.
    pub fn av1_loop_filter_edge(
        &self,
        dst: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        vertical: bool,
        lines: usize,
        filter: Av1EdgeFilter,
    ) {
        assert!(
            lines.is_multiple_of(AV1_EDGE_LINES),
            "AV1 edges filter 4-line segments"
        );
        let reach = filter.length.reach();
        let stride = dst.stride();
        let (across, along) = if vertical { (1, stride) } else { (stride, 1) };
        let (lines_x, lines_y) = if vertical {
            (x + reach, y + lines)
        } else {
            (x + lines, y + reach)
        };
        assert!(
            (if vertical { x } else { y }) >= reach
                && lines_x <= dst.width()
                && lines_y <= dst.height(),
            "AV1 loop filter edge at ({x}, {y}) reads outside the plane"
        );
        // SAFETY: the table matches a supported ISA; the assertion above
        // covers the filter's reach on every line.
        unsafe {
            (self.deblock.av1_edge)(dst.data_mut(), y * stride + x, across, along, lines, filter);
        }
    }

    /// Filter consecutive four-line segments along one AV1 edge (AV1 §7.14).
    /// `None` leaves its segment untouched. The codec supplies thresholds and
    /// schedules distinct edges; adjacent equal filters may share SIMD lanes.
    /// Empty or entirely masked lines do nothing.
    ///
    /// # Panics
    /// If any active segment's filter footprint leaves the picture.
    pub fn av1_loop_filter_line(
        &self,
        dst: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        vertical: bool,
        segments: &[Option<Av1EdgeFilter>],
    ) {
        let mut reach = 0;
        let mut end = 0;
        for (i, filter) in segments.iter().enumerate() {
            if let Some(filter) = filter {
                reach = reach.max(filter.length.reach());
                end = i + 1;
            }
        }
        if end == 0 {
            return;
        }
        let lines = end.checked_mul(AV1_EDGE_LINES).expect("AV1 edge length");
        let (across_pos, across_len, along_pos, along_len) = if vertical {
            (x, dst.width(), y, dst.height())
        } else {
            (y, dst.height(), x, dst.width())
        };
        assert!(
            across_pos >= reach
                && across_pos
                    .checked_add(reach)
                    .is_some_and(|v| v <= across_len)
                && along_pos.checked_add(lines).is_some_and(|v| v <= along_len),
            "AV1 loop filter line reads outside the plane"
        );
        let stride = dst.stride();
        let (across, along) = if vertical { (1, stride) } else { (stride, 1) };
        // SAFETY: every active descriptor's reach and last line were checked;
        // the table matches the supported ISA. Trailing masked entries are omitted.
        unsafe {
            (self.deblock.av1_line)(
                dst.data_mut(),
                y * stride + x,
                across,
                along,
                &segments[..end],
            );
        }
    }

    /// Filter a row-major grid of AV1 four-line edge descriptors.
    /// Entry `segments[y4 * stride + x4]` describes the left (vertical) or
    /// top (horizontal) edge of the corresponding four-pixel unit. `rows`
    /// selects unit rows; padding entries beyond the plane width are ignored.
    /// Final units clip to the picture. Distinct edges retain increasing
    /// across-edge order; the codec schedules vertical/horizontal passes.
    ///
    /// # Panics
    /// If the selected grid rows, descriptor storage or any active filter
    /// footprint are invalid. Validation finishes before any picture writes.
    pub fn av1_loop_filter_grid(
        &self,
        dst: &mut PlaneViewMut<'_>,
        vertical: bool,
        rows: core::ops::Range<usize>,
        segments: &[Option<Av1EdgeFilter>],
        stride: usize,
    ) {
        let grid = deblock::av1_grid::Grid::new(
            dst.width(),
            dst.height(),
            dst.stride(),
            vertical,
            rows,
            segments,
            stride,
        );
        // SAFETY: Grid::new validates descriptor storage and every active
        // picture footprint, including clipped final units; ISA is supported.
        unsafe { (self.deblock.av1_grid)(dst.data_mut(), &grid) };
    }

    /// Filter sparse AV1 edge runs, including final scalar tails in supplied order.
    /// Empty runs are ignored. Adjacent equal filters may share a kernel call.
    /// The caller schedules codec-specific passes; no dense grid is allocated.
    ///
    /// # Panics
    /// Invalid run lengths or footprints panic when consumed. Earlier runs may
    /// already have modified the picture; there is no separate validation sweep.
    pub fn av1_loop_filter_runs(
        &self,
        dst: &mut PlaneViewMut<'_>,
        vertical: bool,
        runs: &[crate::EdgeRun<Av1EdgeFilter>],
    ) {
        let plane = deblock::runs::Plane {
            width: dst.width(),
            height: dst.height(),
            stride: dst.stride(),
            vertical,
        };
        // SAFETY: the instance selected a supported ISA; the run traversal
        // checks each footprint against the valid view before calling a kernel.
        unsafe {
            (self.deblock.av1_runs)(dst.data_mut(), plane, runs);
        }
    }

    /// Filter sparse HEVC luma runs whose lengths are multiples of four in supplied order.
    /// Empty runs are ignored. Adjacent equal filters may share a kernel call.
    /// The caller schedules codec-specific passes; no dense grid is allocated.
    ///
    /// # Panics
    /// Invalid run lengths or footprints panic when consumed. Earlier runs may
    /// already have modified the picture; there is no separate validation sweep.
    pub fn hevc_deblock_luma_runs(
        &self,
        dst: &mut PlaneViewMut<'_>,
        vertical: bool,
        runs: &[crate::EdgeRun<crate::HevcLumaFilter>],
    ) {
        let plane = deblock::runs::Plane {
            width: dst.width(),
            height: dst.height(),
            stride: dst.stride(),
            vertical,
        };
        // SAFETY: the instance selected a supported ISA; the run traversal
        // checks each footprint against the valid view before calling a kernel.
        unsafe {
            (self.deblock.hevc_luma_runs)(dst.data_mut(), plane, runs);
        }
    }

    /// Filter sparse HEVC chroma runs whose lengths are multiples of two in supplied order.
    /// Empty runs are ignored. Adjacent equal filters may share a kernel call.
    /// The caller schedules codec-specific passes; no dense grid is allocated.
    ///
    /// # Panics
    /// Invalid run lengths or footprints panic when consumed. Earlier runs may
    /// already have modified the picture; there is no separate validation sweep.
    pub fn hevc_deblock_chroma_runs(
        &self,
        dst: &mut PlaneViewMut<'_>,
        vertical: bool,
        runs: &[crate::EdgeRun<crate::HevcChromaFilter>],
    ) {
        let plane = deblock::runs::Plane {
            width: dst.width(),
            height: dst.height(),
            stride: dst.stride(),
            vertical,
        };
        // SAFETY: the instance selected a supported ISA; the run traversal
        // checks each footprint against the valid view before calling a kernel.
        unsafe {
            (self.deblock.hevc_chroma_runs)(dst.data_mut(), plane, runs);
        }
    }

    /// Filter one AV2 deblocking edge segment (AVM
    /// `avm_lpf_{vertical,horizontal}_generic`): the [`AV2_EDGE_LINES`]
    /// lines across the edge whose first q0 sample is (`x`, `y`) of `dst`.
    /// `vertical` edges separate horizontally adjacent samples. Scalar on
    /// every ISA.
    ///
    /// # Panics
    ///
    /// If a sample the filter reads ([`Av2EdgeFilter::reach`]) lies outside
    /// the plane.
    pub fn av2_deblock_edge(
        &self,
        dst: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        vertical: bool,
        filter: Av2EdgeFilter,
    ) {
        let (reach_p, reach_q) = filter.reach();
        let stride = dst.stride();
        let (across, along) = if vertical { (1, stride) } else { (stride, 1) };
        let (edge, extent, lines_end, lines_extent) = if vertical {
            (x, dst.width(), y + AV2_EDGE_LINES, dst.height())
        } else {
            (y, dst.height(), x + AV2_EDGE_LINES, dst.width())
        };
        assert!(
            edge >= reach_p && edge + reach_q <= extent && lines_end <= lines_extent,
            "AV2 deblocking edge at ({x}, {y}) reads outside the plane"
        );
        let plane = dst.data_mut();
        if let Some(ramps) =
            deblock::av2::segment_ramps(plane, y * stride + x, across, along, filter)
        {
            // SAFETY: the table matches a supported ISA (checked in
            // `with_isa`); the kernel indexes the plane, and the reach check
            // above covers every sample the ramps move.
            unsafe { (self.deblock.av2_ramps)(plane, &ramps) };
        }
    }

    /// Filter every segment of one AV2 edge line of `dst`: the vertical edge
    /// with q0 in column `coordinate` (`vertical`), or the horizontal edge
    /// with q0 in row `coordinate`. Segment `k` covers the
    /// [`AV2_EDGE_LINES`] lines from `k * AV2_EDGE_LINES` along the edge and
    /// is filtered with `filters[k]` (`None`: left alone). Equal to
    /// [`Dsp::av2_deblock_edge`] on each filtered segment in turn: segments
    /// of one edge line share no sample, so their decisions run batched.
    ///
    /// # Panics
    ///
    /// If the segments extend past the plane, or a filtered segment reads
    /// outside it (as [`Dsp::av2_deblock_edge`]).
    pub fn av2_deblock_line(
        &self,
        dst: &mut PlaneViewMut<'_>,
        coordinate: usize,
        vertical: bool,
        filters: &[Option<Av2EdgeFilter>],
    ) {
        let stride = dst.stride();
        let (across, along, extent, along_extent) = if vertical {
            (1, stride, dst.width(), dst.height())
        } else {
            (stride, 1, dst.height(), dst.width())
        };
        assert!(
            filters.len() * AV2_EDGE_LINES <= along_extent,
            "AV2 deblocking line at {coordinate} extends past the plane"
        );
        // Batches read AV2_DECIDE_REACH samples on each side of every line;
        // lines nearer the border, and a short last batch, go one by one.
        let batched = coordinate >= AV2_DECIDE_REACH && coordinate + AV2_DECIDE_REACH <= extent;
        let (chunks, remainder) = filters.as_chunks::<AV2_DECIDE_SEGMENTS>();
        for (chunk, first) in chunks.iter().zip((0..).step_by(AV2_DECIDE_SEGMENTS)) {
            if chunk.iter().all(Option::is_none) {
                // Nothing to decide.
            } else if batched
                && let Some(segments) = Av2Segments::new(
                    coordinate * across + first * AV2_EDGE_LINES * along,
                    across,
                    along,
                    chunk,
                )
            {
                // SAFETY: the table matches a supported ISA (checked in
                // `with_isa`); the kernel indexes the plane, and the checks
                // above keep AV2_DECIDE_REACH samples on each side of every
                // line inside it.
                unsafe { (self.deblock.av2_chunk)(dst.data_mut(), &segments) };
            } else {
                self.av2_deblock_segments(dst, coordinate, vertical, first, chunk);
            }
        }
        let first = chunks.len() * AV2_DECIDE_SEGMENTS;
        if remainder.iter().all(Option::is_none) {
            return;
        }
        // A short last chunk goes batched as the last full chunk's span,
        // the segments already filtered masked off (segments of one line
        // share no sample, so the order between them does not matter).
        if batched && let Some(start) = filters.len().checked_sub(AV2_DECIDE_SEGMENTS) {
            let mut tail = [None; AV2_DECIDE_SEGMENTS];
            let fresh = AV2_DECIDE_SEGMENTS - remainder.len();
            tail[fresh..].copy_from_slice(remainder);
            if let Some(segments) = Av2Segments::new(
                coordinate * across + start * AV2_EDGE_LINES * along,
                across,
                along,
                &tail,
            ) {
                // SAFETY: as for the chunks above; the span ends with the
                // line's last segment, inside the plane (checked above).
                unsafe { (self.deblock.av2_chunk)(dst.data_mut(), &segments) };
                return;
            }
        }
        self.av2_deblock_segments(dst, coordinate, vertical, first, remainder);
    }

    /// [`Dsp::av2_deblock_edge`] on segments `first..` of an edge line
    /// (arguments as [`Dsp::av2_deblock_line`]).
    fn av2_deblock_segments(
        &self,
        dst: &mut PlaneViewMut<'_>,
        coordinate: usize,
        vertical: bool,
        first: usize,
        filters: &[Option<Av2EdgeFilter>],
    ) {
        for (segment, filter) in (first..).zip(filters) {
            if let Some(filter) = *filter {
                let along = segment * AV2_EDGE_LINES;
                let position = if vertical {
                    (coordinate, along)
                } else {
                    (along, coordinate)
                };
                self.av2_deblock_edge(dst, position, vertical, filter);
            }
        }
    }

    /// Filter one H.264 luma edge of a macroblock (§8.7.2.3, §8.7.2.4): the
    /// 16 lines across the edge whose first q0 sample is (`x`, `y`) of the
    /// plane `dst`. `vertical` edges separate horizontally adjacent samples
    /// (lines run down the edge). `bs` is the boundary strength of each
    /// group of 4 lines, `tc0` the matching `tC0` (ignored where bS is 0 or
    /// 4), and `alpha` / `beta` the edge thresholds. Samples outside 0..255
    /// are clamped as the standard requires.
    ///
    /// # Panics
    ///
    /// If a sample the edge reads (p3..q3) lies outside the plane, or a bS
    /// exceeds 4.
    pub fn h264_deblock_luma_edge(
        &self,
        dst: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        vertical: bool,
        bs: [u8; 4],
        tc0: [u8; 4],
        (alpha, beta): (u8, u8),
    ) {
        let edge = deblock_edge(dst, (x, y), vertical, 4, 16, bs, tc0, (alpha, beta));
        // SAFETY: the table matches this CPU; `deblock_edge` checked that
        // every sample the edge reads lies inside `dst`.
        unsafe { (self.deblock.luma)(dst.data_mut(), &edge) }
    }

    /// Filter every edge of one H.264 luma macroblock (16x16 at (`x`,
    /// `y`)) that `edges` marks, in the standard's order (§8.7): one call
    /// per macroblock instead of per edge.
    ///
    /// # Panics
    ///
    /// As [`Self::h264_deblock_luma_edge`] for each marked edge.
    pub fn h264_deblock_mb_luma(
        &self,
        plane: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        edges: &H264MbEdges,
    ) {
        h264_mb::for_each_edge(edges, false, |vertical, offset, e| {
            let pos = if vertical {
                (x + offset, y)
            } else {
                (x, y + offset)
            };
            self.h264_deblock_luma_edge(plane, pos, vertical, e.bs, e.tc0, (e.alpha, e.beta));
        });
    }

    /// Filter every edge of one H.264 4:2:0 chroma macroblock (8x8 at
    /// (`x`, `y`) of both chroma planes, which share bS and thresholds)
    /// that `edges` marks, in the standard's order (§8.7).
    ///
    /// # Panics
    ///
    /// As [`Self::h264_deblock_chroma_edge`] for each marked edge.
    pub fn h264_deblock_mb_chroma(
        &self,
        u: &mut PlaneViewMut<'_>,
        v: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        edges: &H264MbEdges,
    ) {
        let position = |vertical: bool, offset: usize| {
            if vertical {
                (x + offset, y)
            } else {
                (x, y + offset)
            }
        };
        let same_geometry =
            (u.width(), u.height(), u.stride()) == (v.width(), v.height(), v.stride());
        if same_geometry {
            // Planes are independent, so filtering each edge in both
            // before the next keeps every plane's own edge order.
            h264_mb::for_each_edge(edges, true, |vertical, offset, e| {
                let pos = position(vertical, offset);
                let edge = h264_chroma_edge(u, pos, vertical, e);
                // SAFETY: the table matches this CPU; `h264_chroma_edge`
                // checked the edge against `u`, whose geometry `v` shares.
                unsafe { (self.deblock.chroma_pair)(u.data_mut(), v.data_mut(), &edge) }
            });
            return;
        }
        self.h264_deblock_mb_chroma_plane(u, (x, y), edges);
        self.h264_deblock_mb_chroma_plane(v, (x, y), edges);
    }

    /// Filter every edge of one H.264 4:2:0 chroma macroblock (8x8 at
    /// (`x`, `y`)) of one plane that `edges` marks, in the standard's
    /// order (§8.7): for planes whose thresholds differ (Cb and Cr with
    /// different QP offsets).
    ///
    /// # Panics
    ///
    /// As [`Self::h264_deblock_chroma_edge`] for each marked edge.
    pub fn h264_deblock_mb_chroma_plane(
        &self,
        plane: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        edges: &H264MbEdges,
    ) {
        h264_mb::for_each_edge(edges, true, |vertical, offset, e| {
            let pos = if vertical {
                (x + offset, y)
            } else {
                (x, y + offset)
            };
            self.h264_deblock_chroma_edge(plane, pos, vertical, e.bs, e.tc0, (e.alpha, e.beta));
        });
    }

    /// Filter one H.264 4:2:0 chroma edge (§8.7.2.3, §8.7.2.4): the 8 lines
    /// across the edge whose first q0 sample is (`x`, `y`). `bs` and `tc0`
    /// apply to pairs of lines (chroma line `k` uses entry `k / 2`). As
    /// [`Self::h264_deblock_luma_edge`] otherwise.
    ///
    /// # Panics
    ///
    /// If a sample the edge reads lies outside the plane (p1..q1, or p3..q3
    /// for a vertical edge, whose lines are transposed 8 samples at a time),
    /// or a bS exceeds 4.
    pub fn h264_deblock_chroma_edge(
        &self,
        dst: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        vertical: bool,
        bs: [u8; 4],
        tc0: [u8; 4],
        (alpha, beta): (u8, u8),
    ) {
        let params = H264EdgeParams {
            bs,
            tc0,
            alpha,
            beta,
        };
        let edge = h264_chroma_edge(dst, (x, y), vertical, &params);
        // SAFETY: as for `h264_deblock_luma_edge`.
        unsafe { (self.deblock.chroma)(dst.data_mut(), &edge) }
    }

    /// Deblock 8 HEVC luma lines (§8.7.2.5.3, .6, .7): the two 4-line
    /// segments of the edge whose first q0 sample is (`x`, `y`) of `dst`.
    /// `vertical` edges separate horizontally adjacent samples. `tc` is each
    /// segment's `tC` (0 leaves the segment unfiltered: bS 0, or a `tC` of 0
    /// which changes nothing), `beta` the edge's `β`.
    ///
    /// # Panics
    ///
    /// If a sample the lines read (p3..q3) lies outside the plane.
    pub fn hevc_deblock_luma_edge(
        &self,
        dst: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        vertical: bool,
        tc: [u8; 2],
        beta: u8,
    ) {
        let edge = hevc_edge(
            dst,
            (x, y),
            vertical,
            HEVC_LINES,
            [tc[0], tc[1], 0, 0],
            beta,
        );
        // SAFETY: the table matches this CPU; `hevc_edge` checked that
        // every sample the lines read lies inside `dst`.
        unsafe { (self.deblock.hevc_luma)(dst.data_mut(), &edge) }
    }

    /// Deblock `lines` (4 or 8) HEVC 4:2:0 chroma lines (§8.7.2.5.5) across
    /// the edge whose first q0 sample is (`x`, `y`); `tc` applies to pairs of
    /// lines (line `k` uses entry `k / 2`; 0 leaves the pair unfiltered).
    ///
    /// # Panics
    ///
    /// If `lines` is not 4 or 8, or a sample within 4 of the edge on a line
    /// lies outside the plane.
    pub fn hevc_deblock_chroma_edge(
        &self,
        dst: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        vertical: bool,
        lines: usize,
        tc: [u8; 4],
    ) {
        assert!(
            lines == HEVC_LINES || lines == HEVC_LINES / 2,
            "HEVC chroma edges are 4 or 8 lines"
        );
        let edge = hevc_edge(dst, (x, y), vertical, lines, tc, 0);
        // SAFETY: as for `hevc_deblock_luma_edge`.
        unsafe { (self.deblock.hevc_chroma)(dst.data_mut(), &edge) }
    }

    /// Variance of `src` against the prediction at whole-sample position
    /// (`x`, `y`) of `reference`, refined by libaom's bilinear filter at
    /// 1/8-sample offsets: [`Self::variance`] when both offsets are zero,
    /// otherwise [`Self::sub_pixel_variance`], with the same results.
    ///
    /// For callers that measure one block at many positions in one large
    /// view (mode decision): one bounds check per call, no per-position view
    /// construction.
    ///
    /// # Panics
    ///
    /// If the `(w + 1)` x `(h + 1)` window at (`x`, `y`) leaves `reference`,
    /// `src` is smaller than `size`, or an offset is 8 or more.
    #[must_use]
    #[allow(
        clippy::inline_always,
        reason = "called per candidate vector in mode decision; the checks fold into the caller's"
    )]
    #[inline(always)]
    pub fn prediction_variance(
        &self,
        size: BlockSize,
        reference: &PlaneView<'_>,
        x: usize,
        y: usize,
        x_offset: usize,
        y_offset: usize,
        src: &PlaneView<'_>,
    ) -> Variance {
        let (w, h) = (size.width(), size.height());
        assert!(
            x + w < reference.width() && y + h < reference.height(),
            "prediction window at ({x}, {y}) leaves the {}x{} reference",
            reference.width(),
            reference.height()
        );
        assert!(
            x_offset < SUBPEL_VARIANCE_POSITIONS && y_offset < SUBPEL_VARIANCE_POSITIONS,
            "sub-pixel offsets ({x_offset}, {y_offset}) out of range"
        );
        check_covers(size, src, "source");
        let stride = reference.stride();
        let window = &reference.data()[y * stride + x..];
        // SAFETY: as for `residual`; the window holds (h + 1) rows of
        // (w + 1) samples, which the kernels check again (once per call).
        unsafe {
            if x_offset == 0 && y_offset == 0 {
                (self.diff.variance[size.width_class()])(
                    src.data(),
                    src.stride(),
                    window,
                    stride,
                    h,
                )
            } else {
                (self.diff.sub_pixel_variance[size.width_class()])(
                    window,
                    stride,
                    x_offset,
                    y_offset,
                    src.data(),
                    src.stride(),
                    h,
                )
            }
        }
    }

    /// [`Self::prediction_variance`] at four positions `(x, y, x_offset,
    /// y_offset)` of one block (a sub-sample cross around one centre),
    /// in one call: the windows are checked up front. The result equals four
    /// separate calls; a shared-load kernel can replace the four kernel
    /// calls behind this entry.
    ///
    /// # Panics
    ///
    /// As [`Self::prediction_variance`] for any of the positions.
    #[must_use]
    pub fn prediction_variance_x4(
        &self,
        size: BlockSize,
        reference: &PlaneView<'_>,
        positions: [(usize, usize, usize, usize); 4],
        src: &PlaneView<'_>,
    ) -> [Variance; 4] {
        positions.map(|(x, y, x_offset, y_offset)| {
            self.prediction_variance(size, reference, x, y, x_offset, y_offset, src)
        })
    }

    /// Sum and sum of squares of `src - pred` over a `size` block.
    ///
    /// # Panics
    ///
    /// If either view is smaller than `size`.
    #[must_use]
    pub fn variance(&self, size: BlockSize, src: &PlaneView<'_>, pred: &PlaneView<'_>) -> Variance {
        check_covers(size, src, "source");
        check_covers(size, pred, "prediction");
        // SAFETY: as for `residual`.
        unsafe {
            (self.diff.variance[size.width_class()])(
                src.data(),
                src.stride(),
                pred.data(),
                pred.stride(),
                size.height(),
            )
        }
    }

    /// The sum and sum of squares of the `size` block `src`'s own samples:
    /// [`Self::variance`] against a block of zeros, in one load per sample.
    /// `n_variance(area)` of the result is the block's AC energy.
    ///
    /// # Panics
    ///
    /// If `src` does not cover a `size` block.
    #[must_use]
    pub fn block_sums(&self, size: BlockSize, src: &PlaneView<'_>) -> Variance {
        check_covers(size, src, "source");
        // SAFETY: as for `residual`.
        unsafe {
            (self.diff.block_sums[size.width_class()])(src.data(), src.stride(), size.height())
        }
    }

    /// [`Self::variance`] of each 8x8 block along a row of `src - pred`:
    /// block i covers columns 8i..8i+8 of the top 8 rows. One dispatch for
    /// the row, so tests that judge a residual block by block (the
    /// encoders' zero-residual test) do not pay the per-call checks of
    /// [`Self::variance`] for every 8x8 block.
    ///
    /// # Panics
    ///
    /// If either view is smaller than 8 rows of `8 * out.len()` samples.
    pub fn variance_8x8_row(
        &self,
        src: &PlaneView<'_>,
        pred: &PlaneView<'_>,
        out: &mut [Variance],
    ) {
        check_covers_row(out.len(), src, "source");
        check_covers_row(out.len(), pred, "prediction");
        let block = BlockSize::new(8, 8).expect("8x8 is a block size");
        let kernel = self.diff.variance[block.width_class()];
        for (i, stats) in out.iter_mut().enumerate() {
            let x = 8 * i;
            // SAFETY: both views cover 8 rows of this block from column `x`
            // (checked above), and the table matches a supported ISA.
            *stats = unsafe {
                kernel(
                    &src.data()[x..],
                    src.stride(),
                    &pred.data()[x..],
                    pred.stride(),
                    8,
                )
            };
        }
    }

    /// Sum of squared differences of each 8x8 block along a row of `src`
    /// and `pred` into `out`: block i covers columns 8i..8i+8 of the top 8
    /// rows (the variance kernels' `sse`, without their sums).
    ///
    /// # Panics
    ///
    /// If either view is smaller than 8 rows of `8 * out.len()` samples.
    pub fn sse_8x8_row(&self, src: &PlaneView<'_>, pred: &PlaneView<'_>, out: &mut [u32]) {
        check_covers_row(out.len(), src, "source");
        check_covers_row(out.len(), pred, "prediction");
        // SAFETY: both views cover 8 rows of `8 * out.len()` samples
        // (checked above), and the table matches a supported ISA.
        unsafe {
            (self.diff.sse_8x8_row)(src.data(), src.stride(), pred.data(), pred.stride(), out);
        }
    }

    /// [`Self::variance`] of `src - pred` and the sums of `pred` itself
    /// ([`VarianceEnergy`]) in one pass.
    ///
    /// # Panics
    ///
    /// If either view is smaller than `size`.
    #[must_use]
    pub fn variance_energy(
        &self,
        size: BlockSize,
        src: &PlaneView<'_>,
        pred: &PlaneView<'_>,
    ) -> VarianceEnergy {
        check_covers(size, src, "source");
        check_covers(size, pred, "prediction");
        // SAFETY: as for `residual`.
        unsafe {
            (self.diff.variance_energy[size.width_class()])(
                src.data(),
                src.stride(),
                pred.data(),
                pred.stride(),
                size.height(),
            )
        }
    }

    /// Variance of `src` against libaom's bilinear prediction from
    /// `reference` at 1/8-sample offsets (`diff` module docs). `reference`
    /// starts at the whole-sample position and must cover `(w + 1)` x
    /// `(h + 1)` samples.
    ///
    /// # Panics
    ///
    /// If a view is too small or an offset is 8 or more.
    #[must_use]
    pub fn sub_pixel_variance(
        &self,
        size: BlockSize,
        reference: &PlaneView<'_>,
        x_offset: usize,
        y_offset: usize,
        src: &PlaneView<'_>,
    ) -> Variance {
        let (w, h) = (size.width(), size.height());
        assert!(
            reference.width() > w && reference.height() > h,
            "sub-pixel reference {}x{} is smaller than {}x{}",
            reference.width(),
            reference.height(),
            w + 1,
            h + 1
        );
        assert!(
            x_offset < SUBPEL_VARIANCE_POSITIONS && y_offset < SUBPEL_VARIANCE_POSITIONS,
            "sub-pixel offsets ({x_offset}, {y_offset}) out of range"
        );
        check_covers(size, src, "source");
        // SAFETY: as for `residual`.
        unsafe {
            (self.diff.sub_pixel_variance[size.width_class()])(
                reference.data(),
                reference.stride(),
                x_offset,
                y_offset,
                src.data(),
                src.stride(),
                h,
            )
        }
    }
}

impl Dsp {
    /// AV1 forward 2-D `DCT_DCT` (libaom `fwd_txfm2d`) of the `n`x`n`
    /// residual at `residual` (row-major, `stride`), into `out` in libaom's
    /// column-major coefficient layout (`txfm` module docs). Residuals must
    /// be differences of 8-bit samples (`|r| <= 255`); larger values give
    /// unspecified (but memory-safe) results.
    ///
    /// # Panics
    ///
    /// If `stride < n`, `residual` does not cover `n` rows, or
    /// `out.len() < n * n`.
    pub fn av1_forward_dct(&self, size: DctSize, residual: &[i16], stride: usize, out: &mut [i32]) {
        let n = size.side();
        assert!(
            stride >= n && residual.len() >= (n - 1) * stride + n,
            "residual of {} values at stride {stride} does not cover {n}x{n}",
            residual.len()
        );
        assert!(
            out.len() >= n * n,
            "{} coefficients for a {n}x{n} block",
            out.len()
        );
        // SAFETY: the table matches a supported ISA (checked in `with_isa`);
        // the kernel bounds-checks its slices.
        unsafe { (self.txfm.forward_dct[size as usize])(residual, stride, out) }
    }

    /// Low-frequency forward DCT of the `side`x`side` residual (16, 32 or
    /// 64; row-major at `stride`) into column-major `out` of the coded
    /// square (`side`², 32² for 64): the [`crate::av1_low_keep`] lowest
    /// frequencies per dimension, as [`Self::av1_forward_dct`] (or
    /// [`Self::av1_forward_dct_64`]) computes them, and zeros elsewhere. An
    /// encoder-side approximation: the decoder inverts what was coded.
    ///
    /// # Panics
    ///
    /// If `side` is not 16, 32 or 64, or the slices do not cover the block.
    pub fn av1_forward_dct_low(
        &self,
        side: usize,
        residual: &[i16],
        stride: usize,
        out: &mut [i32],
    ) {
        let index = match side {
            16 => 0,
            32 => 1,
            64 => 2,
            _ => panic!("low-frequency forward of a {side}x{side} block"),
        };
        let coded = side.min(32);
        assert!(
            stride >= side
                && residual.len() >= (side - 1) * stride + side
                && out.len() >= coded * coded,
            "AV1 forward DCT slices do not cover {side}x{side}"
        );
        if let Some(low) = self.txfm.forward_dct_low[index] {
            // SAFETY: the table matches a supported ISA; the slices cover
            // the block (checked above).
            unsafe { low(residual, stride, out) };
            return;
        }
        match DctSize::from_side(side) {
            Some(size) => self.av1_forward_dct(size, residual, stride, out),
            None => self.av1_forward_dct_64(residual, stride, out),
        }
        zero_high_frequencies(&mut out[..coded * coded], coded, crate::av1_low_keep(side));
    }

    /// [`Self::av1_forward_dct_raster`] of a 16x16 or 32x32 block with a
    /// proven-zero test between the passes: false (leaving `out` untouched)
    /// when every coefficient provably has magnitude at most `zero_limit`
    /// (each row output is at most its input line's L1 norm plus the row
    /// graph's rounding), else true with the exact raster coefficients.
    ///
    /// # Panics
    ///
    /// If `size` is not 16 or 32, or the slices do not cover the block.
    pub fn av1_forward_dct_raster_unless_zero(
        &self,
        size: DctSize,
        residual: &[i16],
        stride: usize,
        zero_limit: u32,
        out: &mut [i32],
    ) -> bool {
        let index = match size {
            DctSize::N16 => 0,
            DctSize::N32 => 1,
            _ => panic!("bounded forward of {size:?}"),
        };
        let n = size.side();
        assert!(
            stride >= n && residual.len() >= (n - 1) * stride + n && out.len() >= n * n,
            "AV1 forward DCT slices do not cover {n}x{n}"
        );
        // SAFETY: the table matches a supported ISA; the slices cover the
        // block (checked above).
        unsafe { (self.txfm.forward_dct_raster_bounded[index])(residual, stride, zero_limit, out) }
    }

    /// [`Self::av1_forward_dct_low`] of a 16x16 or 32x32 block with raster
    /// coefficients (as [`Self::av1_forward_dct_raster`]; VP9).
    ///
    /// # Panics
    ///
    /// If `side` is not 16 or 32, or the slices do not cover the block.
    pub fn av1_forward_dct_raster_low(
        &self,
        side: usize,
        residual: &[i16],
        stride: usize,
        out: &mut [i32],
    ) {
        let (index, size) = match side {
            16 => (0, DctSize::N16),
            32 => (1, DctSize::N32),
            _ => panic!("raster low-frequency forward of a {side}x{side} block"),
        };
        assert!(
            stride >= side
                && residual.len() >= (side - 1) * stride + side
                && out.len() >= side * side,
            "AV1 forward DCT slices do not cover {side}x{side}"
        );
        if let Some(low) = self.txfm.forward_dct_raster_low[index] {
            // SAFETY: the table matches a supported ISA; the slices cover
            // the block (checked above).
            unsafe { low(residual, stride, out) };
            return;
        }
        self.av1_forward_dct_raster(size, residual, stride, out);
        zero_high_frequencies(&mut out[..side * side], side, crate::av1_low_keep(side));
    }

    /// Transpose the `n`x`n` coefficient block `src` into `dst`: converts
    /// [`Self::av1_forward_dct`]'s column-major layout to raster order (VP9)
    /// and back.
    ///
    /// # Panics
    ///
    /// If either slice is shorter than `n * n`.
    pub fn transpose_coefficients(&self, size: DctSize, src: &[i32], dst: &mut [i32]) {
        let n = size.side();
        let (src, dst) = (&src[..n * n], &mut dst[..n * n]);
        // SAFETY: the table matches a supported ISA (checked in `with_isa`);
        // both slices hold exactly `n * n` coefficients.
        unsafe { (self.txfm.transpose[size as usize])(src, dst) }
    }

    /// [`Self::av1_forward_dct`] with raster (row-major) coefficients, the
    /// layout VP9 uses: the forward kernel, then [`Self::transpose_coefficients`].
    ///
    /// # Panics
    ///
    /// As [`Self::av1_forward_dct`].
    pub fn av1_forward_dct_raster(
        &self,
        size: DctSize,
        residual: &[i16],
        stride: usize,
        out: &mut [i32],
    ) {
        if let Some(raster) = self.txfm.forward_dct_raster[size as usize] {
            let n = size.side();
            assert!(
                stride >= n && residual.len() >= (n - 1) * stride + n && out.len() >= n * n,
                "AV1 forward DCT slices do not cover {n}x{n}"
            );
            // SAFETY: the table matches a supported ISA; the slices cover
            // the block (as `av1_forward_dct`).
            unsafe { raster(residual, stride, out) };
            return;
        }
        // Exact-size scratch: clearing a 32x32 buffer for every 8x8 block
        // would cost more than the transpose saves.
        match size {
            DctSize::N4 => self.forward_dct_raster::<16>(size, residual, stride, out),
            DctSize::N8 => self.forward_dct_raster::<64>(size, residual, stride, out),
            DctSize::N16 => self.forward_dct_raster::<256>(size, residual, stride, out),
            DctSize::N32 => self.forward_dct_raster::<1024>(size, residual, stride, out),
        }
    }

    fn forward_dct_raster<const AREA: usize>(
        &self,
        size: DctSize,
        residual: &[i16],
        stride: usize,
        out: &mut [i32],
    ) {
        let mut column_major = [0; AREA];
        self.av1_forward_dct(size, residual, stride, &mut column_major);
        self.transpose_coefficients(size, &column_major, out);
    }

    /// AV1 inverse 2-D `DCT_DCT` of `coeffs` (dequantised, libaom layout),
    /// added to the prediction in the top-left `n`x`n` of `dst` with 8-bit
    /// clipping; bit-exact with AV1 §7.13.3.
    ///
    /// # Panics
    ///
    /// If `coeffs.len() < n * n` or `dst` is smaller than `n`x`n`.
    pub fn av1_inverse_dct_add(&self, size: DctSize, coeffs: &[i32], dst: &mut PlaneViewMut<'_>) {
        let n = size.side();
        assert!(
            coeffs.len() >= n * n,
            "{} coefficients for a {n}x{n} block",
            coeffs.len()
        );
        assert!(
            dst.width() >= n && dst.height() >= n,
            "destination {}x{} is smaller than {n}x{n}",
            dst.width(),
            dst.height()
        );
        let stride = dst.stride();
        // SAFETY: as for `av1_forward_dct`.
        unsafe { (self.txfm.inverse_dct_add[size as usize])(coeffs, dst.data_mut(), stride) }
    }

    /// AV1 inverse transform of any size and type (AV1 §7.13.3), added to
    /// the prediction in the top-left `width`x`height` of `dst` with 8-bit
    /// clipping; bit-exact with libaom `inv_txfm2d_add_c` (`av1_inverse`
    /// module docs).
    ///
    /// `coeffs` holds the coded area column-major (coefficient (row `r`,
    /// column `c`) at `c * tx.coded_height() + r`) and `eob` is the
    /// end-of-block of the scan (0 adds nothing). Square `DCT_DCT` runs the
    /// tuned DCT kernels, using `eob` to pick the EOB-sparse 32/64 graphs.
    ///
    /// # Panics
    ///
    /// If `kind` does not fit `tx` ([`Av1TxType::fits`]), `coeffs` is
    /// shorter than `tx.coded_area()` or `dst` is smaller than the block.
    pub fn av1_inverse_add(
        &self,
        tx: Av1TxSize,
        kind: Av1TxType,
        coeffs: &[i32],
        eob: usize,
        dst: &mut PlaneViewMut<'_>,
    ) {
        assert!(kind.fits(tx), "{kind:?} is not defined for {tx:?}");
        let (width, height) = (tx.width(), tx.height());
        assert!(
            coeffs.len() >= tx.coded_area(),
            "{} coefficients for a {width}x{height} transform",
            coeffs.len()
        );
        assert!(
            dst.width() >= width && dst.height() >= height,
            "destination {}x{} is smaller than {width}x{height}",
            dst.width(),
            dst.height()
        );
        if eob == 0 {
            return;
        }
        if kind == Av1TxType::DctDct && eob == 1 {
            let stride = dst.stride();
            // SAFETY: supported ISA and validated destination; DC is clamped
            // identically to the full graph before either transform pass.
            unsafe { (self.av1_inverse.dc)(tx, coeffs[0], dst.data_mut(), stride) };
            return;
        }
        // Preserve the tuned square kernels, especially the guarded narrow
        // NEON/SIMD128 graphs. Replacing them with the generic sparse graph
        // is not uniformly faster (AVX2 8x8 measured +11% instructions).
        if kind == Av1TxType::DctDct && width == height {
            match DctSize::from_side(width) {
                Some(size) if width <= 16 => self.av1_inverse_dct_add(size, coeffs, dst),
                _ => self.av1_inverse_dct_add_sparse(
                    width,
                    coeffs,
                    crate::av1_inverse::av1_dct_nonzero_side(eob),
                    dst,
                ),
            }
            return;
        }
        // Four-high SIMD transforms cannot omit a lane group. Full-EOB
        // transforms also need no scan-bound calculation.
        let nonzero_rows = if height == 4 || eob >= tx.coded_area() {
            tx.coded_height()
        } else {
            crate::av1_inverse::av1_inverse_nonzero_rows(tx, kind, eob)
        };
        let index = tx.index().expect("valid size (checked by fits)");
        let (vertical, horizontal) = kind.kernels();
        let stride = dst.stride();
        // SAFETY: the table was built for this CPU's ISA (`with_isa`); the
        // lengths were checked above (the kernels also index with checks).
        unsafe {
            (self.av1_inverse.add[index])(
                vertical,
                horizontal,
                coeffs,
                nonzero_rows,
                dst.data_mut(),
                stride,
            );
        }
    }

    /// AV1 lossless inverse 4x4 Walsh–Hadamard transform (§7.13.2.10) of
    /// column-major `coeffs` (as [`Self::av1_inverse_add`]), added to the
    /// top-left 4x4 of `dst` with 8-bit clipping. Coefficients are
    /// dequantised values (16 bits, §7.12.3).
    ///
    /// # Panics
    ///
    /// If `dst` is smaller than 4x4.
    pub fn av1_inverse_wht_add(&self, coeffs: &[i32; 16], dst: &mut PlaneViewMut<'_>) {
        assert!(
            dst.width() >= 4 && dst.height() >= 4,
            "WHT destination below 4x4"
        );
        crate::av1_inverse::wht::inverse_wht_add(coeffs, dst);
    }

    /// AV1 forward 64x64 `DCT_DCT` of `residual` (row-major at `stride`),
    /// keeping the top-left 32x32 coefficients, column-major, in
    /// `out[..1024]` (libaom `fwd_txfm2d_c` with `TX_64X64`).
    ///
    /// # Panics
    ///
    /// If `residual` does not cover 64x64 at `stride` or `out` is shorter
    /// than 1024.
    pub fn av1_forward_dct_64(&self, residual: &[i16], stride: usize, out: &mut [i32]) {
        assert!(
            stride >= 64 && residual.len() >= 63 * stride + 64,
            "residual of {} values at stride {stride} does not cover 64x64",
            residual.len()
        );
        assert!(out.len() >= 32 * 32, "{} coefficients for 32x32", out.len());
        // SAFETY: as for `av1_forward_dct`.
        unsafe { (self.txfm.forward_dct_64)(residual, stride, out) }
    }

    /// AV1 inverse 64x64 `DCT_DCT` of the top-left 32x32 column-major
    /// `coeffs` (the rest zero), added to the prediction in the top-left
    /// 64x64 of `dst` with 8-bit clipping; bit-exact with AV1 §7.13.3.
    ///
    /// # Panics
    ///
    /// If `coeffs` is shorter than 1024 or `dst` smaller than 64x64.
    pub fn av1_inverse_dct_add_64(&self, coeffs: &[i32], dst: &mut PlaneViewMut<'_>) {
        assert!(
            coeffs.len() >= 32 * 32,
            "{} coefficients for 32x32",
            coeffs.len()
        );
        assert!(
            dst.width() >= 64 && dst.height() >= 64,
            "destination {}x{} is smaller than 64x64",
            dst.width(),
            dst.height()
        );
        let stride = dst.stride();
        // SAFETY: as for `av1_forward_dct`.
        unsafe { (self.txfm.inverse_dct_add_64)(coeffs, dst.data_mut(), stride) }
    }

    /// AV1 large inverse with EOB-proven structural zeros. `side` is32 or64;
    /// `nonzero_side` is8,16 or32, and coefficients outside that square must
    /// be zero. This is a semantic promise for matching the full inverse,
    /// not a memory-safety precondition: all coefficient values remain safe.
    /// Full inverse methods remain available without this promise.
    ///
    /// # Panics
    /// If sizes or buffers do not satisfy the stated transform dimensions.
    pub fn av1_inverse_dct_add_sparse(
        &self,
        side: usize,
        coeffs: &[i32],
        nonzero_side: usize,
        dst: &mut PlaneViewMut<'_>,
    ) {
        assert!(matches!(side, 32 | 64));
        assert!(matches!(nonzero_side, 8 | 16 | 32));
        assert!(coeffs.len() >= 1024 && dst.width() >= side && dst.height() >= side);
        let stride = dst.stride();
        // SAFETY: ISA checked at table installation; dimensions validated above.
        unsafe {
            (self.txfm.inverse_dct_add_sparse[usize::from(side == 64)])(
                coeffs,
                dst.data_mut(),
                stride,
                nonzero_side,
            );
        }
    }

    /// Exact transform-domain distortion gain numerator, `sum(2*c*d-d*d)`.
    /// Original coefficient magnitudes are below `2^22`; dequantized values are
    /// signed16-bit. These are semantic arithmetic bounds, not memory-safety
    /// prerequisites. Negative gains and sums up to49 bits are retained.
    ///
    /// # Panics
    /// If slice lengths differ or exceed1024 (or bounds fail in debug builds).
    #[must_use]
    pub fn av1_coefficient_gain(&self, original: &[i32], dequantized: &[i32]) -> i64 {
        assert_eq!(original.len(), dequantized.len());
        assert!(original.len() <= 1024, "AV1 coded coefficient extent");
        debug_assert!(original.iter().all(|&c| c.unsigned_abs() < 1 << 22));
        debug_assert!(dequantized.iter().all(|&d| i16::try_from(d).is_ok()));
        // SAFETY: ISA checked at dispatch installation; buffer lengths checked.
        unsafe { (self.txfm.coefficient_gain)(original, dequantized) }
    }

    /// AV1 quantisation of `coeffs` in place (`txfm` module docs); returns
    /// the number of non-zero levels. Coefficient 0 uses the DC step.
    ///
    /// # Panics
    ///
    /// If a step is not positive.
    pub fn av1_quantize(&self, coeffs: &mut [i32], q: &Av1Quantizer) -> u32 {
        assert!(
            q.dc_step > 0 && q.ac_step > 0,
            "quantiser steps must be positive"
        );
        // SAFETY: as for `av1_forward_dct`.
        unsafe { (self.txfm.quantize)(coeffs, q) }
    }

    /// [`Self::av1_quantize`] of `coeffs` into `out`, keeping `coeffs` (the
    /// RD paths need the unquantised coefficients): no copy first.
    ///
    /// # Panics
    ///
    /// If `out` is shorter than `coeffs` or a step is zero.
    pub fn av1_quantize_from(&self, coeffs: &[i32], q: &Av1Quantizer, out: &mut [i32]) -> u32 {
        assert!(
            q.dc_step > 0 && q.ac_step > 0,
            "quantiser steps must be positive"
        );
        assert!(
            out.len() >= coeffs.len(),
            "quantise output shorter than input"
        );
        // SAFETY: as for `av1_forward_dct`; `out` covers `coeffs`.
        unsafe { (self.txfm.quantize_from)(coeffs, q, out) }
    }

    /// AV1 dequantisation (AV1 §7.12.3) of `levels` into `out`.
    ///
    /// # Panics
    ///
    /// If `out` is shorter than `levels`.
    pub fn av1_dequantize(&self, levels: &[i32], q: &Av1Quantizer, out: &mut [i32]) {
        assert!(
            out.len() >= levels.len(),
            "dequantise output shorter than input"
        );
        // SAFETY: as for `av1_forward_dct`.
        unsafe { (self.txfm.dequantize)(levels, q, out) }
    }
}

/// AV2 (AVM v1.0.0) square transforms (`av2_txfm` module docs for the exact
/// definitions and input ranges).
impl Dsp {
    /// AV2 forward DCT (AVM's encoder scaling) of the `n`x`n` residual at
    /// `residual` (row-major, `stride`) into the coded row-major coefficients
    /// (`size.coded_area()`; a 64x64 transform keeps the top-left 32x32).
    /// Residuals must be differences of 8-bit samples (`|r| <= 255`).
    ///
    /// # Panics
    ///
    /// If `residual` does not cover `n` rows at `stride >= n`, or
    /// `out.len() < size.coded_area()`.
    pub fn av2_forward(&self, size: Av2TxSize, residual: &[i16], stride: usize, out: &mut [i32]) {
        check_square_i16(size.side(), residual.len(), stride);
        check_coeffs(size.coded_side(), out.len());
        // SAFETY: the table matches a supported ISA (checked in `with_isa`);
        // the kernel bounds-checks its slices.
        unsafe { (self.av2_txfm.forward[size as usize])(residual, stride, out) }
    }

    /// AV2 normative inverse DCT (AVM `inv_txfm_c`) of the coded row-major
    /// `coeffs` (clamped to 16 bits on load) into the row-major `n`x`n`
    /// residual `out`, each sample within `-256..=255`.
    ///
    /// # Panics
    ///
    /// If `coeffs.len() < size.coded_area()` or `out.len() < n * n`.
    pub fn av2_inverse(&self, size: Av2TxSize, coeffs: &[i32], out: &mut [i32]) {
        self.av2_inverse_corner(size, coeffs, size.coded_side(), out);
    }

    /// [`Self::av2_inverse`] of coefficients that are zero in every row and
    /// column from `nonzero` on (a low-rate block's end of block bounds
    /// them), which the kernels may skip. Only the top-left square of side
    /// [`Av2TxSize::inverse_read_side`] is read: the rest of `coeffs` may
    /// hold anything.
    ///
    /// # Panics
    ///
    /// As [`Self::av2_inverse`], and if `nonzero` exceeds the coded side.
    pub fn av2_inverse_corner(
        &self,
        size: Av2TxSize,
        coeffs: &[i32],
        nonzero: usize,
        out: &mut [i32],
    ) {
        check_coeffs(size.coded_side(), coeffs.len());
        check_coeffs(size.side(), out.len());
        assert!(
            nonzero <= size.coded_side(),
            "nonzero corner beyond the coded side"
        );
        let read = size.inverse_read_side(nonzero);
        debug_assert!(
            coeffs[..size.coded_side() * read]
                .chunks_exact(size.coded_side())
                .enumerate()
                .all(|(r, row)| row[..read]
                    .iter()
                    .enumerate()
                    .all(|(k, &c)| c == 0 || (r < nonzero && k < nonzero))),
            "coefficients outside the nonzero corner"
        );
        // SAFETY: as for `av2_forward`; the corner bound only lets the
        // kernel skip rows it would compute as zeros.
        unsafe { (self.av2_txfm.inverse[size as usize])(coeffs, nonzero, out) }
    }

    /// [`Self::av2_forward`] of any transform type: `DCT_DCT` takes the DCT
    /// kernels, every other type the scalar matrix products of
    /// `av2_txfm::typed` (AVM `fwd_txfm_c`).
    ///
    /// # Panics
    ///
    /// As [`Self::av2_forward`], and if `tx_type` does not exist at `size`
    /// ([`Av2TxType::supports`]; only `DCT_DCT` exists at 64x64).
    pub fn av2_forward_typed(
        &self,
        size: Av2TxSize,
        tx_type: Av2TxType,
        residual: &[i16],
        stride: usize,
        out: &mut [i32],
    ) {
        if tx_type == Av2TxType::DCT {
            self.av2_forward(size, residual, stride, out);
        } else {
            check_square_i16(size.side(), residual.len(), stride);
            check_coeffs(size.coded_side(), out.len());
            // SAFETY: the table matches a supported ISA; `tx_type` exists at
            // `size` (which excludes 64x64) and the slices were checked above.
            unsafe { (self.av2_typed.forward[size as usize])(tx_type, residual, stride, out) }
        }
    }

    /// [`Self::av2_inverse`] of any transform type (AVM `inv_txfm_c`).
    ///
    /// # Panics
    ///
    /// As [`Self::av2_inverse`] and [`Self::av2_forward_typed`].
    pub fn av2_inverse_typed(
        &self,
        size: Av2TxSize,
        tx_type: Av2TxType,
        coeffs: &[i32],
        out: &mut [i32],
    ) {
        self.av2_inverse_typed_corner(size, tx_type, coeffs, size.coded_side(), out);
    }

    /// [`Self::av2_inverse_typed`] with a nonzero corner bound as for
    /// [`Self::av2_inverse_corner`] (only the DCT kernels use it).
    ///
    /// # Panics
    ///
    /// As [`Self::av2_inverse_typed`] and [`Self::av2_inverse_corner`].
    pub fn av2_inverse_typed_corner(
        &self,
        size: Av2TxSize,
        tx_type: Av2TxType,
        coeffs: &[i32],
        nonzero: usize,
        out: &mut [i32],
    ) {
        if tx_type == Av2TxType::DCT {
            self.av2_inverse_corner(size, coeffs, nonzero, out);
        } else {
            check_coeffs(size.coded_side(), coeffs.len());
            check_coeffs(size.side(), out.len());
            // SAFETY: as for `av2_forward_typed`.
            unsafe { (self.av2_typed.inverse[size as usize])(tx_type, coeffs, out) }
        }
    }
}

impl Dsp {
    /// Convert packed RGBA rows (`4 * luma.width()` bytes each, alpha
    /// ignored) into limited-range BT.601 I420: every luma sample, and the
    /// chroma of each whole 2x2 group (`convert` module docs).
    ///
    /// # Panics
    ///
    /// If `rgba` is not exactly `4 * width * height` bytes, the chroma
    /// windows differ in stride or do not cover `width / 2` by `height / 2`.
    pub fn rgba_to_i420(
        &self,
        rgba: &[u8],
        luma: &mut PlaneViewMut<'_>,
        cb: &mut PlaneViewMut<'_>,
        cr: &mut PlaneViewMut<'_>,
    ) {
        let (w, h) = (luma.width(), luma.height());
        assert_eq!(rgba.len(), 4 * w * h, "rgba_to_i420: RGBA buffer size");
        assert!(
            cb.stride() == cr.stride()
                && cb.width() >= w / 2
                && cr.width() >= w / 2
                && cb.height() >= h / 2
                && cr.height() >= h / 2,
            "rgba_to_i420: chroma windows do not cover {w}x{h} luma"
        );
        if w == 0 || h == 0 {
            return;
        }
        let (luma_stride, chroma_stride) = (luma.stride(), cb.stride());
        // SAFETY: the views cover the regions the kernel contract requires
        // (checked above and by the view constructors) and the table matches
        // a supported ISA.
        unsafe {
            (self.convert.rgba_to_i420)(
                rgba,
                luma.data_mut(),
                luma_stride,
                cb.data_mut(),
                cr.data_mut(),
                chroma_stride,
                w,
                h,
            );
        }
    }
}

/// HEVC and H.264 transforms and quantisers (`codec_txfm` module docs for
/// the exact definitions and input ranges). Every method is bit-exact with
/// the scalar code of `nc-hevc` / `nc-h264`.
impl Dsp {
    /// HEVC forward transform (HM scaling) of the `n`x`n` residual at
    /// `residual` (row-major, `stride`) into row-major coefficients
    /// (`out[v * n + u]`), clamped to 16 bits. Residuals must be
    /// differences of 8-bit samples (`|r| <= 255`).
    ///
    /// # Panics
    ///
    /// If `residual` does not cover `n` rows at `stride >= n`, or
    /// `out.len() < n * n`.
    pub fn hevc_forward(
        &self,
        transform: HevcTransform,
        residual: &[i16],
        stride: usize,
        out: &mut [i32],
    ) {
        let n = transform.side();
        check_square_i16(n, residual.len(), stride);
        check_coeffs(n, out.len());
        // SAFETY: the table matches a supported ISA (checked in `with_isa`);
        // the kernel bounds-checks its slices.
        unsafe { (self.codec_txfm.hevc_forward[transform as usize])(residual, stride, out) }
    }

    /// Forward DCT, omitting the second pass when all coefficients provably
    /// have magnitude <= `zero_limit`. False leaves out untouched; true gives
    /// exactly the usual forward transform, including raw coefficients.
    /// Residuals must be differences of 8-bit samples.
    ///
    /// # Panics
    /// If the transform is DST, input/output do not cover the block, or stride is too short.
    pub fn hevc_forward_unless_zero(
        &self,
        transform: HevcTransform,
        residual: &[i16],
        stride: usize,
        zero_limit: u32,
        out: &mut [i32],
    ) -> bool {
        assert!(
            transform != HevcTransform::Dst4,
            "bounded transform requires a DCT"
        );
        let n = transform.side();
        check_square_i16(n, residual.len(), stride);
        check_coeffs(n, out.len());
        // SAFETY: supported ISA, slices checked above.
        unsafe {
            (self.codec_txfm.hevc_forward_bounded[usize::from(transform.log2() - 2)])(
                residual,
                stride,
                zero_limit,
                out,
                HevcForwardMode::EXACT,
            )
        }
    }

    /// Encoder-only forward in a non-exact [`HevcForwardMode`] (staged rows,
    /// or rows and columns, and/or skipping proven-zero column groups of the
    /// exact column pass) with the usual proven-zero certificate between the
    /// passes. True writes the mode's raw coefficients (staging keeps an
    /// exact DC row and column; skipped groups are zero); false leaves output
    /// untouched.
    ///
    /// # Panics
    /// If the transform is not DCT16/32, `mode` is
    /// [`HevcForwardMode::EXACT`], or slices do not cover the block.
    pub fn hevc_forward_staged_unless_zero(
        &self,
        transform: HevcTransform,
        residual: &[i16],
        stride: usize,
        zero_limit: u32,
        mode: HevcForwardMode,
        out: &mut [i32],
    ) -> bool {
        let n = transform.side();
        assert!(matches!(n, 16 | 32), "staged passes require DCT16/32");
        assert!(
            mode != HevcForwardMode::EXACT,
            "the exact forward is hevc_forward_unless_zero"
        );
        check_square_i16(n, residual.len(), stride);
        check_coeffs(n, out.len());
        // SAFETY: supported ISA and checked slices; only 16/32-point DCTs.
        unsafe {
            (self.codec_txfm.hevc_forward_bounded[usize::from(transform.log2() - 2)])(
                residual, stride, zero_limit, out, mode,
            )
        }
    }

    /// HEVC inverse transform and clipped addition to the prediction.
    /// Coefficients are row-major and clamped to i16 on load, as in
    /// [`Self::hevc_inverse`]. Every nonzero coefficient must lie in the
    /// first `rows` rows and `columns` columns of `extent` (a decoder knows
    /// them from the parse; a larger extent is exact, only slower): the
    /// passes skip what lies outside, all-zero blocks return at once and a
    /// DCT extent of (1, 1) adds the DC constant. Nonzero coefficients
    /// outside the extent give unspecified (but memory-safe) results.
    /// SIMD adapters add directly where profitable; other paths reuse the
    /// bounded inverse.
    ///
    /// # Panics
    /// If coefficients or destination do not cover the transform block, or
    /// the extent exceeds it.
    pub fn hevc_inverse_add(
        &self,
        transform: HevcTransform,
        coeffs: &[i32],
        extent: (usize, usize),
        dst: &mut PlaneViewMut<'_>,
    ) {
        let n = transform.side();
        check_coeffs(n, coeffs.len());
        assert!(
            extent.0 <= n && extent.1 <= n,
            "extent {extent:?} beyond the {n}x{n} transform"
        );
        assert!(
            dst.width() >= n && dst.height() >= n,
            "destination smaller than HEVC transform"
        );
        let stride = dst.stride();
        // SAFETY: supported ISA and validated input/output extents.
        unsafe {
            (self.codec_txfm.hevc_inverse_add[transform as usize])(
                coeffs,
                extent,
                dst.data_mut(),
                stride,
            );
        }
    }

    /// HEVC normative inverse transform (§8.6.4.2) of row-major `coeffs`
    /// (clamped to 16 bits on load) into the row-major residual `out`.
    ///
    /// # Panics
    ///
    /// If `coeffs` or `out` is shorter than `n * n`.
    pub fn hevc_inverse(&self, transform: HevcTransform, coeffs: &[i32], out: &mut [i32]) {
        let n = transform.side();
        check_coeffs(n, coeffs.len());
        check_coeffs(n, out.len());
        // SAFETY: as for `hevc_forward`.
        unsafe { (self.codec_txfm.hevc_inverse[transform as usize])(coeffs, out) }
    }

    /// HEVC (HM) quantisation of `coeffs` in place; returns the number of
    /// non-zero levels. Coefficients must lie within `±32768` (forward
    /// transform output does).
    pub fn hevc_quantize(&self, coeffs: &mut [i32], q: &HevcQuantizer) -> u32 {
        let mf = [q.scale; 16];
        // SAFETY: as for `hevc_forward`.
        unsafe {
            if coeffs.len().is_multiple_of(16) {
                (self.codec_txfm.quantize)(coeffs, &mf, q.offset, q.qbits, HEVC_MAX_LEVEL)
            } else {
                (self.codec_txfm.quantize)(coeffs, &mf[..1], q.offset, q.qbits, HEVC_MAX_LEVEL)
            }
        }
    }

    /// HEVC quantisation into `out`, preserving raw `coeffs` for sign hiding.
    /// Returns the number of non-zero levels. Coefficients must be within
    /// the forward transform's signed 16-bit range.
    ///
    /// # Panics
    ///
    /// If `out` is shorter than `coeffs`.
    pub fn hevc_quantize_from(&self, coeffs: &[i32], q: &HevcQuantizer, out: &mut [i32]) -> u32 {
        assert!(out.len() >= coeffs.len());
        // SAFETY: inputs have the same coefficient bounds as hevc_quantize.
        unsafe { (self.codec_txfm.hevc_quantize_from)(coeffs, q, out) }
    }

    /// HEVC scaling (§8.6.3, eq. 8-309, flat scaling lists) of `levels`
    /// into `out`, clamped to 16 bits.
    ///
    /// # Panics
    ///
    /// If `out` is shorter than `levels`.
    pub fn hevc_dequantize(&self, levels: &[i32], q: &HevcQuantizer, out: &mut [i32]) {
        assert!(
            out.len() >= levels.len(),
            "dequantise output shorter than input"
        );
        // SAFETY: as for `hevc_forward`.
        unsafe { (self.codec_txfm.hevc_dequantize)(levels, q, out) }
    }

    /// VVC independent quantization with nearest rounding. Coefficients must
    /// be in the signed 16-bit forward-transform range. Returns nonzero count.
    pub fn vvc_quantize(&self, coeffs: &mut [i32], q: &crate::VvcQuantizer) -> u32 {
        self.hevc_quantize(coeffs, &q.0)
    }

    /// [`Self::vvc_quantize`] into `out`, keeping `coeffs` (for sign hiding
    /// and the trellis). Returns the nonzero count.
    ///
    /// # Panics
    ///
    /// If `out` is shorter than `coeffs`.
    pub fn vvc_quantize_from(
        &self,
        coeffs: &[i32],
        q: &crate::VvcQuantizer,
        out: &mut [i32],
    ) -> u32 {
        self.hevc_quantize_from(coeffs, &q.0, out)
    }

    /// VVC inverse scaling, clamped to signed 16 bits. The shared HEVC
    /// kernels implement the same scaling, including VVC's extended QP range.
    /// # Panics
    /// If `out` is shorter than `levels`.
    pub fn vvc_dequantize(&self, levels: &[i32], q: &crate::VvcQuantizer, out: &mut [i32]) {
        self.hevc_dequantize(levels, &q.0, out);
    }

    /// VVC 8-bit inverse scaling and rectangular DCT-II, using the shared
    /// HEVC scaling and inverse-transform kernels. Explicit scaling lists and
    /// transform skip are excluded; dependent quantization is supported.
    /// # Panics
    /// If either dimension is not 2/4/8/16/32, QP exceeds 63, or a slice is
    /// shorter than `width * height` samples.
    pub fn vvc_inverse_dct(
        &self,
        width: usize,
        height: usize,
        levels: &[i32],
        qp: u8,
        dependent: bool,
        out: &mut [i32],
    ) {
        crate::codec_txfm::vvc_inverse::inverse(self, width, height, levels, qp, dependent, out);
    }

    /// VVC 8-bit forward DCT-II of a `width` x `height` residual (row-major,
    /// `width` per row) into row-major coefficients, scaled for
    /// [`crate::VvcQuantizer::rect`]. Exact shared transform kernels for the
    /// encoder's rectangular blocks; square blocks equal
    /// [`Self::hevc_forward`] and should use it.
    /// # Panics
    /// If a side is not a power of two up to 32, the area is below 16,
    /// or a slice is shorter than the block.
    pub fn vvc_forward_dct(&self, width: usize, height: usize, residual: &[i16], out: &mut [i32]) {
        crate::codec_txfm::vvc_forward::check(width, height, residual.len(), out.len());
        // SAFETY: the validated block covers each exact transform pass;
        // the table matches this instance's ISA.
        unsafe { (self.codec_txfm.vvc_forward_dct)(width, height, residual, out) };
    }

    /// H.264 forward 4x4 core transform (`Cf · X · Cfᵀ`, unscaled) of every
    /// 4x4 block of a residual region (row-major at `stride`): `out` holds
    /// the blocks in raster order, `blocks_wide` per row of blocks, each in
    /// raster order (`row * 4 + column`).
    ///
    /// # Panics
    ///
    /// If `blocks_wide` is 0 or does not divide `out.len()`, or `residual`
    /// does not cover the region.
    pub fn h264_forward_4x4(
        &self,
        residual: &[i16],
        stride: usize,
        blocks_wide: usize,
        out: &mut [[i32; 16]],
    ) {
        assert!(
            blocks_wide > 0 && out.len().is_multiple_of(blocks_wide),
            "{} blocks are not rows of {blocks_wide}",
            out.len()
        );
        let (w, h) = (4 * blocks_wide, 4 * (out.len() / blocks_wide));
        if h > 0 {
            assert!(
                stride >= w && residual.len() >= (h - 1) * stride + w,
                "residual of {} values at stride {stride} does not cover {w}x{h}",
                residual.len()
            );
        }
        // SAFETY: as for `hevc_forward`.
        unsafe { (self.codec_txfm.h264_forward_4x4)(residual, stride, blocks_wide, out) }
    }

    /// H.264 inverse 4x4 transform (§8.5.12.2, including `(x + 32) >> 6`)
    /// of each scaled block of `coeffs` into residual blocks `out`.
    ///
    /// # Panics
    ///
    /// If `out` is shorter than `coeffs`.
    pub fn h264_inverse_4x4(&self, coeffs: &[[i32; 16]], out: &mut [[i32; 16]]) {
        assert!(out.len() >= coeffs.len(), "fewer output blocks than input");
        // SAFETY: as for `hevc_forward`.
        unsafe { (self.codec_txfm.h264_inverse_4x4)(coeffs, out) }
    }

    /// The 4x4 Hadamard transform of H.264 eq. 8-320 (Intra16x16 luma DC;
    /// its own inverse up to a factor of 16).
    #[must_use]
    pub fn h264_hadamard_4x4(&self, block: &[i32; 16]) -> [i32; 16] {
        let mut out = [[0; 16]];
        // SAFETY: as for `hevc_forward`.
        unsafe { (self.codec_txfm.h264_hadamard_4x4)(core::slice::from_ref(block), &mut out) };
        out[0]
    }

    /// H.264 forward 8x8 transform, `(F · X · Fᵀ + 32) >> 6` with `F` 8
    /// times the inverse basis (`codec_txfm` module docs), of the residual
    /// at `residual` (row-major, `stride`); coefficients in raster order.
    ///
    /// # Panics
    ///
    /// If `residual` does not cover 8 rows at `stride >= 8`.
    pub fn h264_forward_8x8(&self, residual: &[i16], stride: usize, out: &mut [i32; 64]) {
        check_square_i16(8, residual.len(), stride);
        // SAFETY: as for `hevc_forward`.
        unsafe { (self.codec_txfm.h264_forward_8x8)(residual, stride, out) }
    }

    /// H.264 inverse 8x8 transform (§8.5.13.2, including `(x + 32) >> 6`).
    pub fn h264_inverse_8x8(&self, coeffs: &[i32; 64], out: &mut [i32; 64]) {
        // SAFETY: as for `hevc_forward`.
        unsafe { (self.codec_txfm.h264_inverse_8x8)(coeffs, out) }
    }

    /// H.264 quantisation of 4x4 blocks in place (all 16 positions; an
    /// Intra16x16 or chroma block's DC is quantised separately with
    /// [`Self::h264_quantize_dc`] and overwritten by the caller); returns
    /// the number of non-zero levels.
    pub fn h264_quantize_4x4(&self, blocks: &mut [[i32; 16]], q: &H264Quantizer) -> u32 {
        let (qbits, offset) = q.forward_terms(0);
        let mf = &QUANT_MF_4X4_POS[q.rem];
        // SAFETY: as for `hevc_forward`.
        unsafe {
            (self.codec_txfm.quantize)(blocks.as_flattened_mut(), mf, offset, qbits, i32::MAX)
        }
    }

    /// H.264 quantisation of a Hadamard-transformed DC block in place (luma
    /// after the caller's halving, or 2x2 chroma): one more bit than AC.
    pub fn h264_quantize_dc(&self, coeffs: &mut [i32], q: &H264Quantizer) -> u32 {
        let (qbits, offset) = q.forward_terms(1);
        let mf = [QUANT_MF_4X4_POS[q.rem][0]; 16];
        let period = if coeffs.len().is_multiple_of(16) {
            16
        } else {
            1
        };
        // SAFETY: as for `hevc_forward`.
        unsafe { (self.codec_txfm.quantize)(coeffs, &mf[..period], offset, qbits, i32::MAX) }
    }

    /// Bit `i` set where `levels[i]` is non-zero.
    ///
    /// # Panics
    ///
    /// If there are more than 64 levels.
    #[must_use]
    pub fn nonzero_mask(&self, levels: &[i32]) -> u64 {
        assert!(
            levels.len() <= MAX_MASK_LEVELS,
            "nonzero_mask of {} levels",
            levels.len()
        );
        // SAFETY: at most 64 levels (checked above); the table matches a
        // supported ISA (checked in `with_isa`).
        unsafe { (self.codec_txfm.nonzero_mask)(levels) }
    }

    /// H.264 quantisation of an 8x8 block in place (`qbits = 16 + qP / 6`).
    pub fn h264_quantize_8x8(&self, coeffs: &mut [i32; 64], q: &H264Quantizer) -> u32 {
        let (qbits, offset) = q.forward_terms(1);
        let mf = &QUANT_MF_8X8_POS[q.rem];
        // SAFETY: as for `hevc_forward`.
        unsafe { (self.codec_txfm.quantize)(coeffs, mf, offset, qbits, i32::MAX) }
    }

    /// Transform and quantise the four raster-order 8x8 blocks of a 16x16
    /// residual in one dispatch. A set bit in `zero_mask` skips that quadrant
    /// and writes zero levels. Returns each block's non-zero level count.
    ///
    /// # Panics
    ///
    /// If `residual` does not cover 16x16 samples at `stride`.
    pub fn h264_forward_quantize_mb(
        &self,
        residual: &[i16],
        stride: usize,
        zero_mask: u8,
        q: &H264Quantizer,
        out: &mut [[i32; 64]; 4],
    ) -> [u32; 4] {
        check_square_i16(16, residual.len(), stride);
        // SAFETY: the complete macroblock is checked above, and dispatch
        // selects kernels only for an available ISA.
        unsafe { (self.codec_txfm.h264_forward_quantize_mb)(residual, stride, zero_mask, q, out) }
    }

    /// H.264 transform and quantisation of one 4:2:0 chroma component's 8x8
    /// residual (at `stride`): the four 4x4 forward transforms, the 2x2
    /// Hadamard of their DCs quantised as [`Self::h264_quantize_dc`], and the
    /// AC levels in `ac` (raster blocks, position 0 zero) quantised as
    /// [`Self::h264_quantize_4x4`]. Returns the DC levels and the number of
    /// non-zero AC levels.
    ///
    /// # Panics
    ///
    /// If `residual` does not cover 8x8 samples at `stride`.
    pub fn h264_forward_quantize_chroma(
        &self,
        residual: &[i16],
        stride: usize,
        q: &H264Quantizer,
        ac: &mut [[i32; 16]; 4],
    ) -> ([i32; 4], u32) {
        check_square_i16(8, residual.len(), stride);
        // SAFETY: the 8x8 block is checked above; the table matches a
        // supported ISA.
        unsafe { (self.codec_txfm.h264_forward_quantize_chroma)(residual, stride, q, ac) }
    }

    /// H.264 scaling of 4x4 blocks (§8.5.12.1, flat weights), every
    /// position; Intra16x16 and chroma callers replace position 0 with the
    /// separately scaled DC.
    ///
    /// # Panics
    ///
    /// If `out` is shorter than `levels`.
    pub fn h264_dequantize_4x4(
        &self,
        levels: &[[i32; 16]],
        q: &H264Quantizer,
        out: &mut [[i32; 16]],
    ) {
        assert!(out.len() >= levels.len(), "fewer output blocks than input");
        let shift = q.per as i32 - 4;
        // SAFETY: as for `hevc_forward`.
        unsafe {
            (self.codec_txfm.h264_dequantize)(
                levels.as_flattened(),
                &LEVEL_SCALE_4X4[q.rem],
                shift,
                &mut out.as_flattened_mut()[..16 * levels.len()],
            );
        }
    }

    /// Intra16x16 luma DC: inverse Hadamard, then scaling (§8.5.10).
    #[must_use]
    pub fn h264_dequantize_luma_dc(&self, levels: &[i32; 16], q: &H264Quantizer) -> [i32; 16] {
        let f = self.h264_hadamard_4x4(levels);
        let ls = [LEVEL_SCALE_4X4[q.rem][0]; 16];
        let mut out = [0; 16];
        // SAFETY: as for `hevc_forward`.
        unsafe { (self.codec_txfm.h264_dequantize)(&f, &ls, q.per as i32 - 6, &mut out) };
        out
    }

    /// 4:2:0 chroma DC: inverse 2x2 transform, then scaling (§8.5.11.1–2,
    /// eq. 8-330). Four values: plain scalar code on every ISA.
    #[must_use]
    pub fn h264_dequantize_chroma_dc(&self, levels: &[i32; 4], q: &H264Quantizer) -> [i32; 4] {
        let c = levels;
        let f = [
            c[0] + c[1] + c[2] + c[3],
            c[0] - c[1] + c[2] - c[3],
            c[0] + c[1] - c[2] - c[3],
            c[0] - c[1] - c[2] + c[3],
        ];
        let scale = LEVEL_SCALE_4X4[q.rem][0];
        f.map(|v| ((v * scale) << q.per) >> 5)
    }

    /// H.264 reconstruction of the 4x4 blocks of a region in place: each
    /// block of `levels` (raster order, `blocks_wide` per row of blocks) is
    /// scaled (§8.5.12.1, flat weights), its position 0 replaced by `dc[b]`
    /// (Intra16x16 and chroma DCs, scaled by
    /// [`Self::h264_dequantize_luma_dc`] / [`Self::h264_dequantize_chroma_dc`])
    /// when `dc` is given, inverse transformed (§8.5.12.2) and added to
    /// `dst`, which holds the prediction, clamped to 8 bits. Blocks without
    /// levels are left as they are, DC-only blocks get their constant
    /// inverse without a transform: the result is that of
    /// [`Self::h264_dequantize_4x4`], [`Self::h264_inverse_4x4`] and
    /// [`Self::add_residual`].
    ///
    /// # Panics
    ///
    /// If `blocks_wide` is 0 or does not divide `levels.len()`, `dc` is
    /// shorter than `levels`, or `dst` does not cover the region.
    pub fn h264_add_inverse_4x4(
        &self,
        levels: &[[i32; 16]],
        dc: Option<&[i32]>,
        q: &H264Quantizer,
        blocks_wide: usize,
        dst: &mut PlaneViewMut<'_>,
    ) {
        assert!(
            blocks_wide > 0 && levels.len().is_multiple_of(blocks_wide),
            "{} blocks are not rows of {blocks_wide}",
            levels.len()
        );
        let dc = dc.unwrap_or(&[]);
        assert!(
            dc.is_empty() || dc.len() >= levels.len(),
            "fewer DCs than blocks"
        );
        let (w, h) = (4 * blocks_wide, 4 * (levels.len() / blocks_wide));
        assert!(
            dst.width() >= w && dst.height() >= h,
            "destination smaller than {w}x{h}"
        );
        let stride = dst.stride();
        let scale = (&LEVEL_SCALE_4X4[q.rem], q.per as i32 - 4);
        // SAFETY: `dst` covers the region (checked above); the table matches
        // a supported ISA.
        unsafe {
            (self.codec_txfm.h264_add_inverse_4x4)(
                levels,
                dc,
                scale,
                blocks_wide,
                dst.data_mut(),
                stride,
            );
        }
    }

    /// H.264 reconstruction of an 8x8 block in place: `levels` scaled
    /// (§8.5.13.1, flat weights), inverse transformed (§8.5.13.2) and added
    /// to `dst`, which holds the prediction, clamped to 8 bits. As
    /// [`Self::h264_dequantize_8x8`], [`Self::h264_inverse_8x8`] and
    /// [`Self::add_residual`]; zero and DC-only blocks as in
    /// [`Self::h264_add_inverse_4x4`].
    ///
    /// # Panics
    ///
    /// If `dst` is smaller than 8x8.
    pub fn h264_add_inverse_8x8(
        &self,
        levels: &[i32; 64],
        q: &H264Quantizer,
        dst: &mut PlaneViewMut<'_>,
    ) {
        assert!(
            dst.width() >= 8 && dst.height() >= 8,
            "destination smaller than 8x8"
        );
        let stride = dst.stride();
        let scale = (&LEVEL_SCALE_8X8[q.rem], q.per as i32 - 6);
        // SAFETY: as for `h264_add_inverse_4x4`.
        unsafe {
            (self.codec_txfm.h264_add_inverse_8x8)(levels, scale, dst.data_mut(), stride);
        }
    }

    /// H.264 reconstruction of 4x4 blocks whose coefficients are already
    /// scaled (a decoder scales them with the active scaling matrices as it
    /// parses): each block of `coeffs` (raster order, `blocks_wide` per row
    /// of blocks) is inverse transformed (§8.5.12.2) and added to `dst`,
    /// which holds the prediction, clamped to 8 bits. Zero blocks leave
    /// their samples as they are and DC-only blocks add their constant
    /// inverse without a transform: [`Self::h264_add_inverse_4x4`] with unit
    /// scale factors, without its scaling pass.
    ///
    /// # Panics
    ///
    /// If `blocks_wide` is 0 or does not divide `coeffs.len()`, or `dst`
    /// does not cover the region.
    pub fn h264_add_scaled_4x4(
        &self,
        coeffs: &[[i32; 16]],
        blocks_wide: usize,
        dst: &mut PlaneViewMut<'_>,
    ) {
        assert!(
            blocks_wide > 0 && coeffs.len().is_multiple_of(blocks_wide),
            "{} blocks are not rows of {blocks_wide}",
            coeffs.len()
        );
        let (w, h) = (4 * blocks_wide, 4 * (coeffs.len() / blocks_wide));
        assert!(
            dst.width() >= w && dst.height() >= h,
            "destination smaller than {w}x{h}"
        );
        let stride = dst.stride();
        // SAFETY: as for `h264_add_inverse_4x4`.
        unsafe {
            (self.codec_txfm.h264_add_scaled_4x4)(coeffs, blocks_wide, dst.data_mut(), stride);
        }
    }

    /// [`Self::h264_add_scaled_4x4`] for the 16 luma 4x4 blocks of a
    /// macroblock (raster order) whose coded blocks (`coded`, bit `b` for
    /// block `b`) and blocks with AC coefficients (`ac`) the caller knows:
    /// uncoded blocks are skipped and coded ones without AC get their DC's
    /// constant inverse, without scanning the coefficients. A coded block
    /// without its `ac` bit must have zero AC coefficients.
    ///
    /// # Panics
    ///
    /// If `dst` is smaller than 16x16.
    pub fn h264_add_scaled_mb(
        &self,
        coeffs: &[[i32; 16]; 16],
        coded: u16,
        ac: u16,
        dst: &mut PlaneViewMut<'_>,
    ) {
        assert!(
            dst.width() >= 16 && dst.height() >= 16,
            "destination smaller than 16x16"
        );
        let stride = dst.stride();
        // SAFETY: as for `h264_add_inverse_4x4`.
        unsafe {
            (self.codec_txfm.h264_add_scaled_mb)(coeffs, (coded, ac), dst.data_mut(), stride);
        }
    }

    /// [`Self::h264_add_scaled_4x4`] for one 8x8 block (§8.5.13.2).
    ///
    /// # Panics
    ///
    /// If `dst` is smaller than 8x8.
    pub fn h264_add_scaled_8x8(&self, coeffs: &[i32; 64], dst: &mut PlaneViewMut<'_>) {
        assert!(
            dst.width() >= 8 && dst.height() >= 8,
            "destination smaller than 8x8"
        );
        let stride = dst.stride();
        // SAFETY: as for `h264_add_inverse_4x4`.
        unsafe {
            (self.codec_txfm.h264_add_scaled_8x8)(coeffs, dst.data_mut(), stride);
        }
    }

    /// H.264 scaling of an 8x8 block (§8.5.13.1, flat weights).
    pub fn h264_dequantize_8x8(&self, levels: &[i32; 64], q: &H264Quantizer, out: &mut [i32; 64]) {
        // SAFETY: as for `hevc_forward`.
        unsafe {
            (self.codec_txfm.h264_dequantize)(
                levels,
                &LEVEL_SCALE_8X8[q.rem],
                q.per as i32 - 6,
                out,
            );
        }
    }
}

/// Panics unless an `i16` slice of `len` covers `n` rows of `n` at `stride`.
fn check_square_i16(n: usize, len: usize, stride: usize) {
    assert!(
        stride >= n && len >= (n - 1) * stride + n,
        "residual of {len} values at stride {stride} does not cover {n}x{n}"
    );
}

/// Panics unless `len` holds an `n`x`n` block of coefficients.
fn check_coeffs(n: usize, len: usize) {
    assert!(len >= n * n, "{len} coefficients for a {n}x{n} block");
}

/// The data of four `size` blocks at `positions` inside `reference`, each
/// slice covering its block at the reference stride.
fn candidate_slices<'a>(
    size: BlockSize,
    reference: &PlaneView<'a>,
    positions: [(usize, usize); 4],
) -> [&'a [u8]; 4] {
    positions.map(|position| block_slice(size, reference, position))
}

/// The data of the `size` block at `position` inside `reference`, covering
/// the block at the reference stride.
fn block_slice<'a>(size: BlockSize, reference: &PlaneView<'a>, (x, y): (usize, usize)) -> &'a [u8] {
    // Checked additions: a huge position must not wrap around and pass.
    let fits = x
        .checked_add(size.width())
        .is_some_and(|end| end <= reference.width())
        && y.checked_add(size.height())
            .is_some_and(|end| end <= reference.height());
    assert!(
        fits,
        "block at ({x}, {y}) leaves the {}x{} reference",
        reference.width(),
        reference.height()
    );
    &reference.data()[y * reference.stride() + x..]
}

/// A row of `blocks` 8x8 blocks must fit in `view`.
#[allow(
    clippy::inline_always,
    reason = "per-call bounds check; out of line it showed as a separate hot function in callgrind"
)]
#[inline(always)]
fn check_covers_row(blocks: usize, view: &PlaneView<'_>, what: &str) {
    assert!(
        view.width() / 8 >= blocks && view.height() >= 8,
        "{what} view {}x{} is smaller than a row of {blocks} 8x8 blocks",
        view.width(),
        view.height()
    );
}

/// An `i16` buffer of `len` values at `stride` must cover the block.
#[allow(
    clippy::inline_always,
    reason = "per-call bounds check; out of line it showed as a separate hot function in callgrind"
)]
#[inline(always)]
/// The taps of a mixed-family AV1 block: each direction's filter for its
/// block edge, at its phase.
fn av1_mixed_taps(
    filters: Av1Filters,
    size: BlockSize,
    (frac_x, frac_y): (usize, usize),
) -> (&'static [i16; mc::TAPS], &'static [i16; mc::TAPS]) {
    (
        &filters.x.kernel(size.width())[frac_x],
        &filters.y.kernel(size.height())[frac_y],
    )
}

/// Panics unless `window` covers an 8-tap AV1 prediction of `size` and the
/// phases are below 16.
fn check_av1_mixed_window(
    size: BlockSize,
    window: &PlaneView<'_>,
    (frac_x, frac_y): (usize, usize),
) {
    let filter = Filter::Av1;
    let extra = filter.taps() - 1;
    assert!(
        window.width() >= size.width() + extra && window.height() >= size.height() + extra,
        "AV1 mixed prediction window {}x{} is smaller than {}x{}",
        window.width(),
        window.height(),
        size.width() + extra,
        size.height() + extra
    );
    assert!(
        frac_x < filter.positions() && frac_y < filter.positions(),
        "AV1 sub-sample offsets ({frac_x}, {frac_y}) out of range"
    );
}

/// Whether sample buffers (length, stride) of a window and a destination
/// cover a `filter` prediction of `size` (the window and the block), with
/// offsets in range: [`Dsp::predict_samples`]'s one check.
#[inline]
fn sample_buffers_cover(
    filter: Filter,
    size: BlockSize,
    (src_len, src_stride): (usize, usize),
    (frac_x, frac_y): (usize, usize),
    (dst_len, dst_stride): (usize, usize),
) -> bool {
    let (w, h) = (size.width(), size.height());
    let extra = filter.taps() - 1;
    let covers = |len: usize, stride: usize, (w, h): (usize, usize)| {
        stride >= w
            && (h - 1)
                .checked_mul(stride)
                .is_some_and(|rows| rows + w <= len)
    };
    covers(src_len, src_stride, (w + extra, h + extra))
        && covers(dst_len, dst_stride, (w, h))
        && frac_x < filter.positions()
        && frac_y < filter.positions()
}

fn check_covers_i16(size: BlockSize, len: usize, stride: usize, what: &str) {
    let (w, h) = (size.width(), size.height());
    assert!(
        stride >= w && len >= (h - 1) * stride + w,
        "{what} buffer of {len} values at stride {stride} does not cover {w}x{h}"
    );
}

/// A source/reference view pair as a kernel's slices and strides.
#[inline]
fn pair<'a>((src, rf): (&PlaneView<'a>, &PlaneView<'a>)) -> crate::pixel::BlockPair<'a> {
    (src.data(), src.stride(), rf.data(), rf.stride())
}

#[allow(
    clippy::inline_always,
    reason = "per-call bounds check; out of line it showed as a separate hot function in callgrind"
)]
#[inline(always)]
fn check_covers(size: BlockSize, view: &PlaneView<'_>, what: &str) {
    assert!(
        view.width() >= size.width() && view.height() >= size.height(),
        "{what} view {}x{} is smaller than the {}x{} block",
        view.width(),
        view.height(),
        size.width(),
        size.height()
    );
}

/// Check and describe an H.264 deblocking edge: `reach` samples on each
/// Zero every coefficient of the `side`x`side` block `coeffs` outside its
/// top-left `keep`x`keep` square (the same in column-major and raster order).
fn zero_high_frequencies(coeffs: &mut [i32], side: usize, keep: usize) {
    for (line, values) in coeffs.chunks_exact_mut(side).enumerate() {
        let start = if line < keep { keep } else { 0 };
        values[start..].fill(0);
    }
}

/// side of the edge, `lines` lines along it.
#[allow(clippy::too_many_arguments)]
fn deblock_edge(
    dst: &PlaneViewMut<'_>,
    (x, y): (usize, usize),
    vertical: bool,
    reach: usize,
    lines: usize,
    bs: [u8; 4],
    tc0: [u8; 4],
    (alpha, beta): (u8, u8),
) -> Edge {
    assert!(bs.iter().fold(0, |max, &b| max.max(b)) <= 4, "bS above 4");
    // Selects and non-short-circuit tests: callers alternate vertical and
    // horizontal edges, which made the direction a mispredicted branch.
    let (w, h) = (dst.width(), dst.height());
    let pick = |a: usize, b: usize| select_usize(vertical, a, b);
    let (across_at, across_end, along_at, along_end) =
        (pick(x, y), pick(w, h), pick(y, x), pick(h, w));
    #[allow(
        clippy::needless_bitwise_bool,
        reason = "one combined test, no branch per term"
    )]
    let inside = (across_at >= reach)
        & across_at
            .checked_add(reach)
            .is_some_and(|end| end <= across_end)
        & along_at
            .checked_add(lines)
            .is_some_and(|end| end <= along_end);
    assert!(inside, "deblocking edge reads outside the plane");
    let stride = dst.stride();
    let (across, along) = if vertical { (1, stride) } else { (stride, 1) };
    Edge {
        base: y * stride + x,
        across,
        along,
        bs,
        alpha: i16::from(alpha),
        beta: i16::from(beta),
        tc0: tc0.map(i16::from),
    }
}

/// `if cond { a } else { b }` by mask, with no branch: the deblocking
/// edge checks alternate the direction from call to call.
fn select_usize(cond: bool, a: usize, b: usize) -> usize {
    let mask = 0usize.wrapping_sub(usize::from(cond));
    (a & mask) | (b & !mask)
}

/// Check an H.264 4:2:0 chroma edge against `dst` and resolve it to kernel
/// form. A vertical edge's lines may be transposed through 8-sample loads
/// (p3..q3), so its reach is checked as for luma; a horizontal one reads
/// only p1..q1.
fn h264_chroma_edge(
    dst: &PlaneViewMut<'_>,
    (x, y): (usize, usize),
    vertical: bool,
    e: &H264EdgeParams,
) -> Edge {
    let reach = if vertical { 4 } else { 2 };
    deblock_edge(
        dst,
        (x, y),
        vertical,
        reach,
        8,
        e.bs,
        e.tc0,
        (e.alpha, e.beta),
    )
}

/// Check an HEVC edge of `lines` lines (reading p3..q3) against `dst` and
/// resolve it to kernel form.
fn hevc_edge(
    dst: &PlaneViewMut<'_>,
    (x, y): (usize, usize),
    vertical: bool,
    lines: usize,
    tc: [u8; 4],
    beta: u8,
) -> HevcEdge {
    let (w, h) = (dst.width(), dst.height());
    let inside = if vertical {
        x >= HEVC_REACH
            && x.checked_add(HEVC_REACH).is_some_and(|end| end <= w)
            && y.checked_add(lines).is_some_and(|end| end <= h)
    } else {
        y >= HEVC_REACH
            && y.checked_add(HEVC_REACH).is_some_and(|end| end <= h)
            && x.checked_add(lines).is_some_and(|end| end <= w)
    };
    assert!(inside, "deblocking edge reads outside the plane");
    let stride = dst.stride();
    let (across, along) = if vertical { (1, stride) } else { (stride, 1) };
    HevcEdge {
        base: y * stride + x,
        across,
        along,
        lines,
        tc: tc.map(i16::from),
        beta: i16::from(beta),
    }
}

/// Check an SAO rectangle against class `class`'s neighbours in a
/// `width` x `height` plane.
fn sao_rect(
    class: u8,
    width: usize,
    height: usize,
    (x0, y0, x1, y1): (usize, usize, usize, usize),
) -> Rect {
    assert!(class < 4, "SAO edge-offset class {class} out of range");
    let [(ax, ay), (bx, by)] = SAO_EO_NEIGHBOURS[usize::from(class)];
    let reach = |a: isize, b: isize| (usize::from(a < 0 || b < 0), usize::from(a > 0 || b > 0));
    let (left, right) = reach(ax, bx);
    let (top, bottom) = reach(ay, by);
    let empty = x0 >= x1 || y0 >= y1;
    assert!(
        empty || (x0 >= left && x1 + right <= width && y0 >= top && y1 + bottom <= height),
        "SAO rectangle or its neighbours outside the plane"
    );
    Rect {
        x0,
        y0,
        x1: x1.max(x0),
        y1: y1.max(y0),
        row_step: 1,
    }
}

/// Index offsets of class `class`'s two neighbours at row stride `stride`.
fn sao_offsets(class: u8, stride: usize) -> (isize, isize) {
    let [(ax, ay), (bx, by)] = SAO_EO_NEIGHBOURS[usize::from(class)];
    let stride = stride as isize;
    (ay * stride + ax, by * stride + bx)
}

#[cfg(test)]
mod tests {
    use nc_common::PlaneView;

    use super::Dsp;
    use crate::{BlockSize, Isa};

    // Exercise only the checked resolver: invalid coordinates must never
    // reach an unsafe filter, even when release arithmetic would wrap.
    fn resolve_h264_edge(position: (usize, usize), vertical: bool) {
        let mut data = [0u8; 32 * 32];
        let view = nc_common::PlaneViewMut::new(&mut data, 32, 32, 32).unwrap();
        let _ = super::deblock_edge(&view, position, vertical, 4, 16, [1; 4], [1; 4], (20, 10));
    }

    #[test]
    #[should_panic(expected = "deblocking edge reads outside the plane")]
    fn h264_vertical_edge_rejects_overflowing_x() {
        resolve_h264_edge((usize::MAX - 1, 4), true);
    }

    #[test]
    #[should_panic(expected = "deblocking edge reads outside the plane")]
    fn h264_vertical_edge_rejects_overflowing_y() {
        resolve_h264_edge((4, usize::MAX - 7), true);
    }

    #[test]
    #[should_panic(expected = "deblocking edge reads outside the plane")]
    fn h264_horizontal_edge_rejects_overflowing_x() {
        resolve_h264_edge((usize::MAX - 7, 4), false);
    }

    #[test]
    #[should_panic(expected = "deblocking edge reads outside the plane")]
    fn h264_horizontal_edge_rejects_overflowing_y() {
        resolve_h264_edge((4, usize::MAX - 1), false);
    }

    #[test]
    fn h264_edge_accepts_exact_footprint_boundaries() {
        resolve_h264_edge((4, 0), true);
        resolve_h264_edge((28, 16), true);
        resolve_h264_edge((0, 4), false);
        resolve_h264_edge((16, 28), false);
    }

    #[test]
    fn best_isa_matches_scalar_on_a_simple_block() {
        let src: Vec<u8> = (0..64 * 64).map(|i| (i * 7 % 251) as u8).collect();
        let rf: Vec<u8> = (0..64 * 64).map(|i| (i * 13 % 241) as u8).collect();
        let s = PlaneView::new(&src, 64, 64, 64).unwrap();
        let r = PlaneView::new(&rf, 64, 64, 64).unwrap();
        let scalar = Dsp::with_isa(Isa::Scalar).unwrap();
        let best = Dsp::new();
        for size in BlockSize::all() {
            assert_eq!(
                best.sad(size, &s, &r),
                scalar.sad(size, &s, &r),
                "sad {size:?}"
            );
            assert_eq!(
                best.satd(size, &s, &r),
                scalar.satd(size, &s, &r),
                "satd {size:?}"
            );
        }
    }

    #[test]
    fn variance_8x8_row_matches_per_block_variance() {
        let (stride, height) = (70, 9);
        let src: Vec<u8> = (0..stride * height).map(|i| (i * 7 % 251) as u8).collect();
        let pred: Vec<u8> = (0..stride * height).map(|i| (i * 13 % 241) as u8).collect();
        let block = BlockSize::new(8, 8).unwrap();
        let dsp = Dsp::new();
        for blocks in [1, 2, 4, 8] {
            let width = 8 * blocks;
            let s = PlaneView::new(&src[1..], stride, width, 8).unwrap();
            let p = PlaneView::new(&pred[3..], stride, width, 8).unwrap();
            let mut row = vec![crate::Variance::default(); blocks];
            dsp.variance_8x8_row(&s, &p, &mut row);
            for (i, stats) in row.iter().enumerate() {
                let s8 = s.sub_view(8 * i, 0, 8, 8).unwrap();
                let p8 = p.sub_view(8 * i, 0, 8, 8).unwrap();
                assert_eq!(
                    *stats,
                    dsp.variance(block, &s8, &p8),
                    "{blocks} blocks, {i}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "smaller than")]
    fn undersized_view_panics() {
        let data = [0u8; 16];
        let v = PlaneView::new(&data, 4, 4, 4).unwrap();
        let _ = Dsp::new().sad(BlockSize::new(8, 8).unwrap(), &v, &v);
    }
}

/// An H.264 prediction kernel for one filter and block size, from
/// [`Dsp::h264_predictor`].
///
/// The H.264 kernels check their whole window and destination once on entry
/// (`mc::generic::check_h264_window`) and then run on raw pointers, so this
/// handle needs no checks of its own beyond the sub-sample offsets.
///
/// Where the ISA has them, the kernels are resolved per phase
/// (`mc::H264PhaseFns`): a call picks its kernel by table lookup, and the
/// kernel runs without branching on the phase.
#[derive(Clone, Copy)]
pub struct H264Predictor {
    /// By `mc::h264_slot` of the offsets.
    kernels: [ConvolveFn; mc::H264_PHASE_SLOTS],
    height: usize,
    positions: usize,
    /// `mc::h264_slot`'s cap for this filter.
    slot_cap: usize,
}

impl core::fmt::Debug for H264Predictor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("H264Predictor")
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl H264Predictor {
    /// Predict the block from `window` (its top-left reference sample offset
    /// by the filter's `before()`, as for [`Dsp::predict`]) into `dst`.
    ///
    /// # Panics
    ///
    /// If the window or `dst` is too small, or an offset is out of range.
    #[allow(
        clippy::inline_always,
        reason = "called a few times per macroblock; the point of the handle is no call overhead"
    )]
    #[inline(always)]
    pub fn predict(
        &self,
        window: &[u8],
        window_stride: usize,
        frac_x: usize,
        frac_y: usize,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        assert!(
            frac_x < self.positions && frac_y < self.positions,
            "sub-sample offsets ({frac_x}, {frac_y}) out of range"
        );
        let kernel = self.kernels[mc::h264_slot(frac_x, frac_y, self.slot_cap)];
        // SAFETY: the kernel comes from a table built for a supported ISA
        // (`Dsp::with_isa`), and H.264 kernels verify that `window` and
        // `dst` cover the block before touching them.
        unsafe {
            kernel(
                window,
                window_stride,
                dst,
                dst_stride,
                self.height,
                frac_x,
                frac_y,
            );
        }
    }
}

/// The typed AV2 transform table of `isa` (as [`av2_txfm_fns`]).
fn av2_typed_fns(isa: Isa) -> Av2TypedFns {
    match isa {
        Isa::Scalar => av2_txfm::typed_lanes::scalar_fns(),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => av2_txfm::avx2::av2_typed_fns(),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => av2_txfm::neon::av2_typed_fns(),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => av2_txfm::simd128::av2_typed_fns(),
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_supported() is false for ISAs not compiled in"),
    }
}

/// The AV2 transform table of `isa`. Calling its kernels needs `isa`'s
/// support, which [`Dsp::with_isa`] checks before building a table.
fn av2_txfm_fns(isa: Isa) -> Av2TxfmFns {
    match isa {
        Isa::Scalar => av2_txfm::scalar::av2_txfm_fns(),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => av2_txfm::avx2::av2_txfm_fns(),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => av2_txfm::neon::av2_txfm_fns(),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => av2_txfm::simd128::av2_txfm_fns(),
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_supported() is false for ISAs not compiled in"),
    }
}

/// The CDEF kernels of `isa`.
///
/// # Safety
///
/// `isa` must be supported by this CPU ([`Isa::is_supported`]).
unsafe fn cdef_fns(isa: Isa) -> CdefFns {
    match isa {
        Isa::Scalar => cdef::scalar::cdef_fns(),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: guaranteed by the caller.
        Isa::Avx2 => unsafe { cdef::avx2::cdef_fns() },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => cdef::neon::cdef_fns(),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => cdef::simd128::cdef_fns(),
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_supported() is false for ISAs not compiled in"),
    }
}

/// The SAO kernels of `isa`.
///
/// # Safety
///
/// `isa` must be supported by this CPU ([`Isa::is_supported`]).
unsafe fn sao_fns(isa: Isa) -> SaoFns {
    match isa {
        Isa::Scalar => sao::scalar::sao_fns(),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: guaranteed by the caller.
        Isa::Avx2 => unsafe { sao::avx2::sao_fns() },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => sao::neon::sao_fns(),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => sao::simd128::sao_fns(),
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_supported() is false for ISAs not compiled in"),
    }
}

/// The conversion kernel table for `isa`.
///
/// # Safety
///
/// `isa` must be supported by this CPU ([`Isa::is_supported`]).
unsafe fn convert_fns(isa: Isa) -> ConvertFns {
    match isa {
        Isa::Scalar => convert::scalar::convert_fns(),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: guaranteed by the caller.
        Isa::Avx2 => unsafe { convert::avx2::convert_fns() },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => convert::neon::convert_fns(),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => convert::simd128::convert_fns(),
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_supported() is false for ISAs not compiled in"),
    }
}

impl Dsp {
    /// Split a semi-planar (NV12) chroma window of interleaved Cb, Cr pairs
    /// into the Cb window `u` and the Cr window `v` (`convert` module docs).
    ///
    /// # Panics
    ///
    /// If `u` and `v` differ in size or `src` is not exactly `2 * u.width()`
    /// bytes by `u.height()` rows.
    pub fn deinterleave_uv(
        &self,
        src: &PlaneView<'_>,
        u: &mut PlaneViewMut<'_>,
        v: &mut PlaneViewMut<'_>,
    ) {
        let (w, h) = (u.width(), u.height());
        assert!(
            (v.width(), v.height()) == (w, h) && (src.width(), src.height()) == (2 * w, h),
            "deinterleave_uv: {}x{} source for {w}x{h} and {}x{} planes",
            src.width(),
            src.height(),
            v.width(),
            v.height()
        );
        if w == 0 || h == 0 {
            return;
        }
        let (u_stride, v_stride) = (u.stride(), v.stride());
        // SAFETY: the views cover the source and destination regions
        // (checked above and by the view constructors) and the table matches
        // a supported ISA.
        unsafe {
            (self.convert.deinterleave_uv)(
                src.data(),
                src.stride(),
                u.data_mut(),
                u_stride,
                v.data_mut(),
                v_stride,
                w,
                h,
            );
        }
    }
}

impl Dsp {
    /// Convert a decoded 4:2:0 picture (`luma` and its half-size `cb`, `cr`
    /// planes) into packed, opaque RGBA rows of `4 * luma.width()` bytes
    /// with `matrix` (`convert` module docs).
    ///
    /// # Panics
    ///
    /// If the chroma planes do not cover `luma.width().div_ceil(2)` by
    /// `luma.height().div_ceil(2)` samples, or `rgba` is not exactly
    /// `4 * luma.width() * luma.height()` bytes.
    pub fn i420_to_rgba(
        &self,
        matrix: DisplayMatrix,
        luma: &PlaneView<'_>,
        cb: &PlaneView<'_>,
        cr: &PlaneView<'_>,
        rgba: &mut [u8],
    ) {
        let (w, h) = (luma.width(), luma.height());
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        assert!(
            cb.width() >= cw && cr.width() >= cw && cb.height() >= ch && cr.height() >= ch,
            "i420_to_rgba: chroma planes do not cover {w}x{h} luma"
        );
        assert_eq!(rgba.len(), 4 * w * h, "i420_to_rgba: RGBA buffer size");
        if w == 0 {
            return;
        }
        // Row pairs share one chroma row; an odd last row goes alone.
        for (pair, out) in rgba.chunks_mut(8 * w).enumerate() {
            let (top, bottom) = out.split_at_mut(4 * w);
            let bottom_luma = if bottom.is_empty() {
                &[][..]
            } else {
                luma.row(2 * pair + 1)
            };
            // SAFETY: the rows have the lengths the kernel contract requires
            // (checked above) and the table matches a supported ISA.
            unsafe {
                (self.convert.i420_to_rgba_rows)(
                    matrix,
                    [luma.row(2 * pair), bottom_luma],
                    &cb.row(pair)[..cw],
                    &cr.row(pair)[..cw],
                    [top, bottom],
                );
            }
        }
    }
}

impl Dsp {
    /// Sum of squared differences between two equally sized planes (the
    /// browser demo's PSNR).
    ///
    /// # Panics
    ///
    /// If the planes differ in size.
    #[must_use]
    pub fn sse(&self, a: &PlaneView<'_>, b: &PlaneView<'_>) -> u64 {
        assert!(
            (a.width(), a.height()) == (b.width(), b.height()),
            "sse: {}x{} and {}x{} planes",
            a.width(),
            a.height(),
            b.width(),
            b.height()
        );
        let mut sum = 0;
        for y in 0..a.height() {
            for (a, b) in a
                .row(y)
                .chunks(convert::MAX_SSE_ROW)
                .zip(b.row(y).chunks(convert::MAX_SSE_ROW))
            {
                // SAFETY: the chunks are equally long and at most
                // MAX_SSE_ROW samples (kernel contract); the table matches a
                // supported ISA.
                sum += u64::from(unsafe { (self.convert.sse_row)(a, b) });
            }
        }
        sum
    }
}

impl Dsp {
    /// VP9 quantization of `coefficients` into `levels`: index 0 at `dc`,
    /// the rest at `ac` (`vp9_quant` module docs). Returns whether any level
    /// is nonzero. Coefficients must satisfy `|c| < 2^16`; larger values give
    /// unspecified (but memory-safe) levels.
    ///
    /// # Panics
    ///
    /// If the slices differ in length.
    pub fn vp9_quantize(
        &self,
        coefficients: &[i32],
        dc: &crate::Vp9QuantStep,
        ac: &crate::Vp9QuantStep,
        levels: &mut [i32],
    ) -> bool {
        assert_eq!(
            coefficients.len(),
            levels.len(),
            "VP9 quantize lengths differ"
        );
        // SAFETY: the table matches the checked CPU ISA; kernels index only
        // within the equal-length slices.
        unsafe { (self.vp9_quantize)(coefficients, dc, ac, levels) }
    }

    /// The largest magnitude `max |c|` of `coefficients` (0 when empty): the
    /// VP9 encoder's test for blocks that certainly quantize to zero.
    #[must_use]
    pub fn vp9_largest_magnitude(&self, coefficients: &[i32]) -> u32 {
        // SAFETY: the table matches the checked CPU ISA; kernels read only
        // within the slice.
        unsafe { (self.vp9_largest_magnitude)(coefficients) }
    }

    /// VP9 intra prediction of a `side`x`side` transform block (4, 8, 16 or
    /// 32) from its gathered `edges` into `out` (row stride `side`).
    ///
    /// # Panics
    ///
    /// If `side` is not a VP9 transform side or `out` is shorter than
    /// `side * side`.
    pub fn vp9_intra_predict(
        &self,
        mode: crate::Vp9IntraMode,
        side: usize,
        edges: &crate::Vp9IntraEdges,
        out: &mut [u8],
    ) {
        assert!(matches!(side, 4 | 8 | 16 | 32), "VP9 transform side {side}");
        assert!(
            side == 4
                || edges.above[side + 1..=2 * side]
                    .iter()
                    .all(|&s| s == edges.above[side]),
            "VP9 above-right samples must repeat the last above sample past 4x4"
        );
        let out = &mut out[..side * side];
        // SAFETY: the table matches the checked CPU ISA; kernels index only
        // within `out` and the fixed-size edge arrays.
        unsafe { (self.vp9_intra)(mode, side, edges, out) }
    }

    /// H.264 `Intra_8x8` prediction (§8.3.2.2) in `Intra8x8PredMode` `mode`
    /// from the filtered references `refs` into `out` (raster, stride 8).
    ///
    /// # Panics
    ///
    /// If `refs` lacks samples `mode` reads ([`crate::H264Intra8x8Refs::allows`]).
    pub fn h264_intra_8x8(&self, refs: &crate::H264Intra8x8Refs, mode: u8, out: &mut [u8; 64]) {
        assert!(
            refs.allows(mode),
            "Intra_8x8 mode {mode} without its samples"
        );
        // SAFETY: the table matches the checked CPU ISA; kernels index only
        // within the fixed-size border and output.
        unsafe { (self.h264_intra_8x8)(refs, mode, out) }
    }

    /// HEVC angular intra lines (§8.4.4.2.6) of an `n`x`n` block: `refs[n +
    /// i]` is the main reference `ref[i]` for `i` in `−n..=2n`; line `j` of
    /// `out` (a row for vertical modes, a column for horizontal ones) is the
    /// main reference projected at `(j + 1) · angle / 32`.
    ///
    /// # Panics
    ///
    /// If `n` is not 4, 8, 16 or 32, `angle` is outside −32..=32, `refs` is
    /// not `3n + 1` samples long, or `out` is shorter than `n * n`.
    pub fn hevc_intra_angular(&self, refs: &[u8], n: usize, angle: i32, out: &mut [u8]) {
        use crate::hevc_intra::MAX_ANGLE;
        assert!(matches!(n, 4 | 8 | 16 | 32), "HEVC intra side {n}");
        assert!(
            (-MAX_ANGLE..=MAX_ANGLE).contains(&angle),
            "HEVC angle {angle}"
        );
        assert_eq!(refs.len(), 3 * n + 1, "HEVC main reference length");
        let out = &mut out[..n * n];
        // SAFETY: the table matches the checked CPU ISA; kernels index only
        // within the checked slices.
        unsafe { (self.hevc_intra.angular)(refs, n, angle, out) }
    }

    /// HEVC weighted sample prediction (§8.5.3.3.4, see the
    /// `hevc_weight` module docs) of the `dst`-sized block from the 14-bit
    /// predictions `a` and `b`, row-major at the block width:
    /// `dst = clamp((a·w0 + b·w1 + round) >> shift, 0, 255)`. For
    /// uni-prediction pass `b = a` with `w1 = 0`.
    ///
    /// # Panics
    ///
    /// If `a` or `b` is shorter than the block, or `shift` is outside
    /// 1..=14.
    pub fn hevc_weighted_prediction(
        &self,
        weights: crate::HevcWeights,
        a: &[i16],
        b: &[i16],
        dst: &mut PlaneViewMut<'_>,
    ) {
        let (w, h) = (dst.width(), dst.height());
        assert!(
            a.len() >= w * h && b.len() >= w * h,
            "prediction shorter than the block"
        );
        assert!(
            (1..=crate::hevc_weight::MAX_SHIFT).contains(&weights.shift),
            "weighted prediction shift out of range"
        );
        let dst_stride = dst.stride();
        // SAFETY: the kernel matches this CPU (table built for it); the
        // slices cover the block (checked above, and by the view).
        unsafe {
            (self.hevc_weight)(&weights, a, b, w, h, dst.data_mut(), dst_stride);
        }
    }

    /// Combine centered HEVC predictors from [`Self::predict_compound`].
    /// `weights` uses the usual uncentered prediction formula: the 8192 bias
    /// is restored algebraically in its round term, without a sample pass.
    /// The existing [`Self::hevc_weighted_prediction`] keeps its uncentered
    /// contract for VVC and other callers.
    pub fn hevc_weighted_prediction_centered(
        &self,
        mut weights: crate::HevcWeights,
        a: &[i16],
        b: &[i16],
        dst: &mut PlaneViewMut<'_>,
    ) {
        weights.round += i32::from(crate::mc::HEVC_PREP_OFFSET)
            * (i32::from(weights.w0) + i32::from(weights.w1));
        self.hevc_weighted_prediction(weights, a, b, dst);
    }

    /// HEVC planar intra prediction (§8.4.4.2.4) of an `n`x`n` block into
    /// `out` (row stride `n`).
    ///
    /// # Panics
    ///
    /// If `n` is not 4, 8, 16 or 32, an edge is shorter than `n`, or `out`
    /// is shorter than `n * n`.
    pub fn hevc_intra_planar(&self, edges: &crate::HevcPlanarEdges<'_>, n: usize, out: &mut [u8]) {
        assert!(matches!(n, 4 | 8 | 16 | 32), "HEVC intra side {n}");
        let edges = crate::HevcPlanarEdges {
            above: &edges.above[..n],
            left: &edges.left[..n],
            ..*edges
        };
        let out = &mut out[..n * n];
        // SAFETY: as for `hevc_intra_angular`.
        unsafe { (self.hevc_intra.planar)(&edges, n, out) }
    }

    /// VP9 interpolation with byte rounding and clipping after each pass.
    /// `window` begins three samples above and left of the integer position.
    /// Sizes are at most 64, phases are in 0..16, and the source includes seven
    /// extra samples on each axis. Output rows are tightly packed.
    pub fn vp9_convolve(
        &self,
        kind: crate::Vp9Filter,
        window: &[u8],
        stride: usize,
        phase: (usize, usize),
        (width, height): (usize, usize),
        dst: &mut [u8],
    ) {
        self.vp9_convolve_into(kind, window, stride, phase, (width, height), dst, width);
    }

    /// VP9 prediction directly into strided picture rows, with the source-window
    /// and size contract of [`Self::vp9_convolve`]. Padding between rows is preserved.
    pub fn vp9_convolve_into(
        &self,
        kind: crate::Vp9Filter,
        window: &[u8],
        stride: usize,
        phase: (usize, usize),
        size: (usize, usize),
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        self.vp9_convolve_mode(kind, window, stride, phase, size, dst, dst_stride, false);
    }

    /// Predict and average into existing strided picture rows. Combines the
    /// clipped prediction with each destination byte as `(a + b + 1) >> 1`.
    /// Source-window and size contracts are as [`Self::vp9_convolve_into`].
    pub fn vp9_convolve_avg_into(
        &self,
        kind: crate::Vp9Filter,
        window: &[u8],
        stride: usize,
        phase: (usize, usize),
        size: (usize, usize),
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        self.vp9_convolve_mode(kind, window, stride, phase, size, dst, dst_stride, true);
    }

    fn vp9_convolve_mode(
        &self,
        kind: crate::Vp9Filter,
        window: &[u8],
        stride: usize,
        phase: (usize, usize),
        (width, height): (usize, usize),
        dst: &mut [u8],
        dst_stride: usize,
        average: bool,
    ) {
        assert!(
            width <= crate::vp9_mc_simd::MAX_SIDE
                && height <= crate::vp9_mc_simd::MAX_SIDE
                && phase.0 < crate::vp9_mc::VP9_POSITIONS
                && phase.1 < crate::vp9_mc::VP9_POSITIONS
        );
        if width == 0 || height == 0 {
            return;
        }
        crate::mc::generic::check_plane(
            window,
            stride,
            width + crate::VP9_TAPS - 1,
            height + crate::VP9_TAPS - 1,
        );
        crate::mc::generic::check_plane(dst, dst_stride, width, height);
        // SAFETY: the dispatch table matches this CPU. The assertions cover
        // every source tap and destination store, including exact final rows.
        unsafe {
            (self.vp9_mc)(
                kind,
                window,
                stride,
                phase,
                (width, height),
                dst,
                dst_stride,
                average,
            );
        }
    }
}

// The caller has checked ISA support before selecting these kernel tables.
fn codec_txfm_fns(isa: Isa) -> CodecTxfmFns {
    match isa {
        Isa::Scalar => codec_txfm::scalar::codec_txfm_fns(),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => codec_txfm::avx2::codec_txfm_fns(),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => codec_txfm::neon::codec_txfm_fns(),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => codec_txfm::simd128::codec_txfm_fns(),
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_supported() is false for ISAs not compiled in"),
    }
}

impl Dsp {
    /// VVC planar/DC/angular prediction and PDPC using prepared (possibly smoothed)
    /// references. `top` and `left` exclude the corner and include the far
    /// endpoint at index `side`. Angular modes require `2*side` samples per edge;
    /// the codec must apply any required integer-slope reference smoothing.
    /// # Panics
    /// If side/direction is unsupported, buffers are short or references are not bytes.
    pub fn vvc_intra_predict(
        &self,
        side: usize,
        mode: crate::VvcIntraMode,
        top: &[i32],
        left: &[i32],
        corner: i32,
        out: &mut [u8],
    ) {
        self.vvc_intra_predict_checked(side, mode, top, left, corner, false, out);
    }

    /// VVC rectangular prediction from substituted, optionally smoothed references.
    /// Each edge contains twice its block dimension, excluding `corner`.
    /// Square blocks retain the existing dispatched predictor.
    /// # Panics
    /// If dimensions are not powers of two in 2..=64, the direction is invalid,
    /// buffers are short, or reference samples are outside the byte range.
    pub fn vvc_intra_predict_rect(
        &self,
        (width, height): (usize, usize),
        mode: crate::VvcIntraMode,
        top: &[i32],
        left: &[i32],
        corner: i32,
        out: &mut [u8],
    ) {
        let supported = |n: usize| n.is_power_of_two() && (2..=64).contains(&n);
        assert!(supported(width) && supported(height));
        assert!(top.len() >= 2 * width && left.len() >= 2 * height);
        assert!(out.len() >= width * height);
        if let crate::VvcIntraMode::Angular { direction, .. } = mode {
            assert!((2..=66).contains(&direction));
        }
        let (top, left) = (&top[..2 * width], &left[..2 * height]);
        assert!(
            top.iter()
                .chain(left)
                .chain(std::iter::once(&corner))
                .all(|&v| (0..=255).contains(&v))
        );
        if width == height && (4..=32).contains(&width) {
            // SAFETY: dimensions, direction, reference range and all buffer
            // lengths were checked above; the table matches the selected ISA.
            unsafe {
                (self.vvc_intra)(width, mode, top, left, corner, false, out);
            }
        } else {
            crate::vvc_intra::rectangular::predict(width, height, mode, top, left, corner, out);
        }
    }

    /// [`Self::vvc_intra_predict`] in the kernel's natural orientation:
    /// horizontal angular directions (2..=33) may come out transposed,
    /// which the result reports. For costs that are transpose invariant
    /// (SATD of square blocks against the transposed source), this skips
    /// the transposing store.
    /// # Panics
    /// As [`Self::vvc_intra_predict`].
    #[must_use]
    pub fn vvc_intra_predict_oriented(
        &self,
        side: usize,
        mode: crate::VvcIntraMode,
        top: &[i32],
        left: &[i32],
        corner: i32,
        out: &mut [u8],
    ) -> bool {
        self.vvc_intra_predict_checked(side, mode, top, left, corner, true, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn vvc_intra_predict_checked(
        &self,
        side: usize,
        mode: crate::VvcIntraMode,
        top: &[i32],
        left: &[i32],
        corner: i32,
        transposed_ok: bool,
        out: &mut [u8],
    ) -> bool {
        assert!([4, 8, 16, 32].contains(&side));
        let required = match mode {
            crate::VvcIntraMode::Angular { direction, .. } => {
                assert!((2..=66).contains(&direction));
                2 * side
            }
            _ => side + 1,
        };
        assert!(out.len() >= side * side && top.len() >= required && left.len() >= required);
        assert!(
            top[..required]
                .iter()
                .chain(&left[..required])
                .chain(std::iter::once(&corner))
                .all(|&v| (0..=255).contains(&v))
        );
        // SAFETY: the table matches the supported ISA, all slice bounds checked above.
        unsafe { (self.vvc_intra)(side, mode, top, left, corner, transposed_ok, out) }
    }
}

impl Dsp {
    /// VVC luma deblocking across four lines, each p7..p0,q0..q7.
    /// # Panics
    /// If beta/tc are outside 8-bit ranges or a reach is neither 3 nor 7.
    pub fn vvc_deblock_luma(
        &self,
        edge: &mut [[u8; 16]; 4],
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    ) {
        assert!((0..=88).contains(&beta) && (0..=100).contains(&tc));
        if !lengths.iter().all(|&n| n == 3 || n == 7) {
            // Affine and narrow reaches: the scalar reference only.
            crate::vvc_deblock::vvc_deblock_luma(edge, beta, tc, lengths);
            return;
        }
        // SAFETY: per-instance table uses the supported ISA; buffers have fixed sizes.
        unsafe {
            (self.vvc_deblock.luma)(edge, beta, tc, lengths);
        }
    }
    /// Deblock one 4-line VVC luma segment in place (§8.8.3.6) whose first
    /// q0 sample is (`x`, `y`) of `dst`, at any reaches
    /// [`crate::vvc_deblock_luma`] accepts: for segments whose strength or
    /// reaches differ from the other half of their 8-line edge (affine
    /// sub-blocks).
    /// # Panics
    /// If a sample the lines read (p7..q7) lies outside the plane, or
    /// `beta`, `tc` or the reaches are out of range.
    pub fn vvc_deblock_luma_segment(
        &self,
        dst: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        vertical: bool,
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    ) {
        self.vvc_luma_lines(dst, (x, y), vertical, 4, beta, tc, lengths);
    }
    /// Deblock 8 VVC luma lines in place (§8.8.3.6): the two 4-line
    /// segments of the edge whose first q0 sample is (`x`, `y`) of `dst`,
    /// both between the same two coded blocks (one `β`, `tC` and pair of
    /// reaches). `vertical` edges separate horizontally adjacent samples.
    /// Each segment decides its own filter. All legal reaches run the ISA
    /// kernels, including five-sample affine reaches and narrow edges.
    /// # Panics
    /// If a sample the lines read (p7..q7) lies outside the plane, or
    /// `beta`, `tc` or a reach is out of range.
    pub fn vvc_deblock_luma_edge(
        &self,
        dst: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        vertical: bool,
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    ) {
        self.vvc_luma_lines(
            dst,
            (x, y),
            vertical,
            crate::vvc_deblock::EDGE_LINES,
            beta,
            tc,
            lengths,
        );
    }

    fn vvc_luma_lines(
        &self,
        dst: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        vertical: bool,
        lines: usize,
        beta: i32,
        tc: i32,
        lengths: [usize; 2],
    ) {
        use crate::vvc_deblock::LUMA_REACH;
        assert!((0..=88).contains(&beta) && (0..=100).contains(&tc));
        assert!(
            lengths == [1, 1]
                || lengths == [2, 2]
                || lengths.iter().all(|&n| matches!(n, 3 | 5 | 7))
        );
        let (w, h) = (dst.width(), dst.height());
        let inside = if vertical {
            x >= LUMA_REACH
                && x.checked_add(LUMA_REACH).is_some_and(|end| end <= w)
                && y.checked_add(lines).is_some_and(|end| end <= h)
        } else {
            y >= LUMA_REACH
                && y.checked_add(LUMA_REACH).is_some_and(|end| end <= h)
                && x.checked_add(lines).is_some_and(|end| end <= w)
        };
        assert!(inside, "deblocking edge reads outside the plane");
        let stride = dst.stride();
        let (across, along) = if vertical { (1, stride) } else { (stride, 1) };
        let edge = crate::vvc_deblock::VvcLumaEdge {
            base: y * stride + x,
            lines,
            across,
            along,
            beta,
            tc,
            lengths,
        };
        let kernel = if lines == 4 {
            self.vvc_deblock.luma_segment
        } else if lengths.iter().all(|&n| n == 3 || n == 7) {
            self.vvc_deblock.luma_edge
        } else {
            self.vvc_deblock.luma_extended
        };
        // SAFETY: the table matches this CPU; every sample the lines read
        // lies inside `dst` (checked above), and the kernels index slices.
        unsafe { kernel(dst.data_mut(), &edge) }
    }
    /// VVC chroma deblocking in place across 2, 4, 6 or 8 lines.
    /// Each two-line segment decides its own strong/weak filter; the edge
    /// shares `beta`, `tc`, block-size eligibility and CTU-boundary status.
    /// `at` is q0 of the first line; vertical edges cross a picture row.
    /// # Panics
    /// If p3..q3 of any declared line lies outside the plane, the line count
    /// is unsupported, or beta/tc are outside their 8-bit ranges.
    pub fn vvc_deblock_chroma_edge(
        &self,
        dst: &mut PlaneViewMut<'_>,
        (x, y): (usize, usize),
        vertical: bool,
        lines: usize,
        beta: i32,
        tc: i32,
        large: bool,
        boundary: bool,
    ) {
        use crate::deblock::samples::{Edge, LANES, REACH};
        assert!(lines > 0 && lines <= LANES && lines.is_multiple_of(2));
        assert!((0..=88).contains(&beta) && (0..=100).contains(&tc));
        let (width, height) = (dst.width(), dst.height());
        let inside = if vertical {
            x >= REACH
                && x.checked_add(REACH).is_some_and(|end| end <= width)
                && y.checked_add(lines).is_some_and(|end| end <= height)
        } else {
            y >= REACH
                && y.checked_add(REACH).is_some_and(|end| end <= height)
                && x.checked_add(lines).is_some_and(|end| end <= width)
        };
        assert!(inside, "chroma deblocking edge reads outside the plane");
        // Every correction clips to +/-tc, so zero cannot change a sample.
        if tc == 0 {
            return;
        }
        let stride = dst.stride();
        let (across, along) = if vertical { (1, stride) } else { (stride, 1) };
        let edge = crate::vvc_deblock::ChromaEdge {
            samples: Edge {
                base: y * stride + x,
                across,
                along,
                lines,
            },
            beta,
            tc,
            large,
            boundary,
        };
        // SAFETY: the table matches this CPU, and every line's p3..q3
        // samples are inside dst as checked above; kernels index slices.
        unsafe {
            (self.vvc_deblock.chroma_edge)(dst.data_mut(), &edge);
        }
    }
    /// VVC chroma deblocking across two lines, each p3..p0,q0..q3.
    /// # Panics
    /// If beta/tc are outside 8-bit ranges.
    pub fn vvc_deblock_chroma(
        &self,
        edge: &mut [[u8; 8]; 2],
        beta: i32,
        tc: i32,
        large: bool,
        boundary: bool,
    ) {
        assert!((0..=88).contains(&beta) && (0..=100).contains(&tc));
        // SAFETY: per-instance table uses the supported ISA; buffers have fixed sizes.
        unsafe {
            (self.vvc_deblock.chroma)(edge, beta, tc, large, boundary);
        }
    }
}

impl Dsp {
    /// One block row (four rows) of the H.266 luma ALF (`vvc_alf` module
    /// docs): `rows[r]` are the seven rows the taps of `out[r]` read (`dy`
    /// −3..=3), each starting [`VVC_ALF_REACH`](crate::VVC_ALF_REACH)
    /// columns left of `out[r][0]`; `coefficients[i]` filters columns
    /// `4i..4i + 4` of every row; row `r`'s tap sum is rounded by
    /// `shifts[r]`. Coefficients must lie within −128..=127 (§7.4.3.18
    /// `AlfCoeffL`); kernels may narrow them to bytes.
    /// # Panics
    /// If the output rows differ in length, a row is shorter than their
    /// length + 6, the coefficients do not cover them in 4-sample blocks, or
    /// a shift is outside 1..=16.
    pub fn vvc_alf_filter_block_row(
        &self,
        rows: &[[&[u8]; 7]; crate::VVC_ALF_BLOCK],
        coefficients: &[[i16; crate::VVC_ALF_TAPS]],
        shifts: &[u32; crate::VVC_ALF_BLOCK],
        out: &mut [&mut [u8]; crate::VVC_ALF_BLOCK],
    ) {
        use crate::vvc_alf::{VVC_ALF_BLOCK, VVC_ALF_REACH};
        let width = out[0].len();
        assert!(out.iter().all(|row| row.len() == width));
        assert!(shifts.iter().all(|shift| (1..=16).contains(shift)));
        assert_eq!(coefficients.len(), width.div_ceil(VVC_ALF_BLOCK));
        assert!(
            rows.as_flattened()
                .iter()
                .all(|row| row.len() >= width + 2 * VVC_ALF_REACH)
        );
        debug_assert!(
            coefficients
                .as_flattened()
                .iter()
                .all(|c| i8::try_from(*c).is_ok())
        );
        // SAFETY: the table matches this CPU; the lengths are checked above.
        unsafe { (self.vvc_alf.filter_block_row)(rows, coefficients, shifts, out) }
    }

    /// The Laplacian sums of one ALF classification grid row (`vvc_alf`
    /// module docs): `rows` are the rows above, at, below and two below
    /// it, each starting 3 columns left of column 0 (group `q` covers
    /// columns `4q − 2 ..= 4q + 1`);
    /// `out[q]` is group `q`'s `[vertical, horizontal, diagonal 0,
    /// diagonal 1]`.
    /// # Panics
    /// If a row is shorter than `4 * out.len() + 2`.
    pub fn vvc_alf_laplacians(&self, rows: &[&[u8]; 4], out: &mut [[u16; 4]]) {
        assert!(rows.iter().all(|row| row.len() >= 4 * out.len() + 2));
        // SAFETY: the table matches this CPU; the lengths are checked above.
        unsafe { (self.vvc_alf.laplacians)(rows, out) }
    }

    /// The ALF encoder's statistics samples at columns `4i`, `i < count`:
    /// sample `i` holds, at lane `VVC_ALF_TRANSPOSES[transposes[i]][k]`,
    /// the tap sum `a + b − 2c` of tap `k`, at lane 12 the error `c −
    /// source[4i]`, zeros after. `rows` as one row's for [`Self::vvc_alf_filter_block_row`];
    /// `source` the original row from column 0.
    /// # Panics
    /// If `count` exceeds [`VVC_ALF_SAMPLES`](crate::VVC_ALF_SAMPLES), a
    /// row is shorter than `4 * count + 6`, `source` does not reach column
    /// `4 * (count − 1)`, or a transpose index exceeds 3.
    pub fn vvc_alf_samples(
        &self,
        rows: &[&[u8]; 7],
        source: &[u8],
        transposes: &[u8; crate::VVC_ALF_SAMPLES],
        count: usize,
        out: &mut [[i16; crate::VVC_ALF_SAMPLE_LANES]; crate::VVC_ALF_SAMPLES],
    ) {
        use crate::vvc_alf::{VVC_ALF_REACH, VVC_ALF_SAMPLE_STEP, VVC_ALF_SAMPLES};
        assert!(count <= VVC_ALF_SAMPLES);
        assert!(
            rows.iter()
                .all(|row| row.len() >= VVC_ALF_SAMPLE_STEP * count + 2 * VVC_ALF_REACH)
        );
        assert!(count == 0 || source.len() > VVC_ALF_SAMPLE_STEP * (count - 1));
        assert!(transposes.iter().all(|&t| t < 4));
        // SAFETY: the table matches this CPU; the lengths are checked above.
        unsafe { (self.vvc_alf.samples)(rows, source, transposes, count, out) }
    }

    /// Add the Gram matrix of `rows` into `out`: `Σ_x rows[i][x] ·
    /// rows[j][x]` into `out[gram_index(n, i, j)]` for every `i ≤ j < n`
    /// (`crate::gram`). The statistics of least-squares filter design (the
    /// AV1 Wiener restoration search).
    ///
    /// # Panics
    ///
    /// If there are more than [`GRAM_MAX_ROWS`](crate::GRAM_MAX_ROWS) rows,
    /// their lengths differ or are not a multiple of
    /// [`GRAM_ROW_ALIGN`](crate::GRAM_ROW_ALIGN) up to
    /// [`GRAM_ROW_MAX_LEN`](crate::GRAM_ROW_MAX_LEN), or `out` is shorter
    /// than [`gram_terms`](crate::gram_terms)`(n)`. Values beyond
    /// ±[`GRAM_MAX_MAGNITUDE`](crate::GRAM_MAX_MAGNITUDE) may wrap.
    pub fn gram_rows(&self, rows: &[&[i16]], out: &mut [i64]) {
        use crate::gram::{GRAM_MAX_ROWS, GRAM_ROW_ALIGN, GRAM_ROW_MAX_LEN, gram_terms};
        let len = rows.first().map_or(0, |row| row.len());
        assert!(rows.len() <= GRAM_MAX_ROWS && out.len() >= gram_terms(rows.len()));
        assert!(len.is_multiple_of(GRAM_ROW_ALIGN) && len <= GRAM_ROW_MAX_LEN);
        assert!(rows.iter().all(|row| row.len() == len));
        // SAFETY: the table matches this CPU; the lengths are checked above.
        unsafe { (self.gram_rows)(rows, out) }
    }

    /// The packed upper triangle (`out[vvc_alf_gram_index(i, j)]`, `i ≤
    /// j`; the rest of `out` zero) of the Gram matrix `Σ_n samples[n][i] ·
    /// samples[n][j]` of ALF statistics samples (tap sums, then the error)
    /// over their first
    /// [`VVC_ALF_GRAM_COLUMNS`](crate::VVC_ALF_GRAM_COLUMNS) lanes. Values
    /// must lie within
    /// ±[`VVC_ALF_GRAM_MAX_MAGNITUDE`](crate::VVC_ALF_GRAM_MAX_MAGNITUDE);
    /// beyond it the sums may wrap.
    /// # Panics
    /// If the number of samples is odd (pad with a zero sample) or exceeds
    /// [`VVC_ALF_GRAM_MAX_SAMPLES`](crate::VVC_ALF_GRAM_MAX_SAMPLES).
    pub fn vvc_alf_gram(
        &self,
        samples: &[[i16; crate::VVC_ALF_SAMPLE_LANES]],
        out: &mut [i32; crate::VVC_ALF_DOT_LANES],
    ) {
        use crate::vvc_alf::{VVC_ALF_GRAM_ALIGN, VVC_ALF_GRAM_MAX_SAMPLES};
        assert!(
            samples.len().is_multiple_of(VVC_ALF_GRAM_ALIGN)
                && samples.len() <= VVC_ALF_GRAM_MAX_SAMPLES
        );
        // SAFETY: the table matches this CPU; the length is checked above.
        unsafe { (self.vvc_alf.gram)(samples, out) }
    }

    /// `Σ a[k] · b[k]` in 64 bits (wrapping): the ALF decision's
    /// squared-error change of one filter on one class's statistics.
    #[must_use]
    pub fn vvc_alf_dot(
        &self,
        a: &[i32; crate::VVC_ALF_DOT_LANES],
        b: &[i32; crate::VVC_ALF_DOT_LANES],
    ) -> i64 {
        // SAFETY: the table matches this CPU; the arrays have fixed sizes.
        unsafe { (self.vvc_alf.dot)(a, b) }
    }

    /// The ALF classes (§8.8.5.3) of a row of 4x4 blocks: `out[b] =
    /// [class index, transpose index]` from the sums of Laplacian groups
    /// `b` and `b + 1` ([`Self::vvc_alf_laplacians`] summed over the block
    /// row's grid rows); `boundary`: three grid rows next to a virtual
    /// boundary (the activity scales by 96 instead of 64).
    /// Each group sum must be at most
    /// [`VVC_ALF_MAX_GROUP_SUM`](crate::VVC_ALF_MAX_GROUP_SUM).
    /// # Panics
    /// If `groups.len() != out.len() + 1`.
    pub fn vvc_alf_classify(&self, groups: &[[u16; 4]], boundary: bool, out: &mut [[u8; 2]]) {
        assert_eq!(groups.len(), out.len() + 1);
        debug_assert!(
            groups
                .as_flattened()
                .iter()
                .all(|&sum| sum <= crate::VVC_ALF_MAX_GROUP_SUM)
        );
        // SAFETY: the table matches this CPU; the lengths are checked above.
        unsafe { (self.vvc_alf.classify)(groups, boundary, out) }
    }
}

impl Dsp {
    /// One row of the AV1 masked compound blend (the kernel of
    /// [`crate::av1_mc::blend::av1_blend_mask`]): `dst[j]` from `p0[j]`,
    /// `p1[j]` and weight `weights[j]` (0 to 64).
    ///
    /// # Panics
    ///
    /// If an input is shorter than `dst`.
    pub(crate) fn av1_mask_blend_row(
        &self,
        p0: &[i16],
        p1: &[i16],
        weights: &[i16],
        dst: &mut [u8],
    ) {
        let n = dst.len();
        assert!(p0.len() >= n && p1.len() >= n && weights.len() >= n);
        // SAFETY: the table was built for a supported ISA; the kernel indexes
        // its slices (lengths checked).
        unsafe { (self.av1_mask_blend)(p0, p1, weights, dst) }
    }

    /// Warp one 8x8 block from its source window (the kernel of
    /// [`crate::av1_mc::warp::av1_warp`]).
    pub(crate) fn av1_warp8x8(
        &self,
        (window, stride): (&[u8], usize),
        phases: &crate::av1_mc::warp_kernels::WarpPhases,
        round1: u32,
        out: &mut crate::av1_mc::warp_kernels::WarpOut,
    ) {
        assert!((1..=11).contains(&round1), "warp rounding of {round1} bits");
        assert!(
            stride >= crate::av1_mc::warp_kernels::WINDOW_STRIDE
                && window.len() >= crate::av1_mc::warp_kernels::window_span(stride),
            "warp window too short"
        );
        // SAFETY: the table was built for a supported ISA; the kernel indexes
        // the window (its span checked) and writes only `out`.
        unsafe { (self.av1_warp)((window, stride), phases, round1, out) }
    }

    /// The AV1 loop restoration row kernels (driven by
    /// [`crate::av1_restoration::Av1LoopRestoration::apply`]).
    pub(crate) fn av1_lr(&self) -> &crate::av1_restoration::kernels::Kernels {
        &self.av1_lr
    }

    /// Apply AV1 grain to a luma row. Grain is in -128..=127; `scaling` is
    /// the plane's scaling function. Writes exactly `source.len()` pixels.
    ///
    /// # Panics
    /// Panics for short buffers or reversed clipping bounds.
    pub fn av1_grain_luma_row(
        &self,
        source: &[u8],
        grain: &[i16],
        scaling: &crate::av1_film_grain::Av1GrainScaling,
        bounds: [u8; 2],
        out: &mut [u8],
    ) {
        assert!(grain.len() >= source.len() && out.len() >= source.len());
        assert!(bounds[0] <= bounds[1]);
        // SAFETY: table is selected for this CPU; buffers checked, and the
        // scaling's shift and range by its constructor.
        unsafe {
            (self.av1_grain.luma)(source, grain, scaling, bounds, &mut out[..source.len()]);
        }
    }

    /// Apply AV1 grain to a 4:2:0 chroma row, using pairs of colocated luma
    /// samples (an odd final luma sample is duplicated). Coefficients are
    /// centered `[mult, luma_mult, offset]`; `[0, 64, 0]` scales from luma.
    /// Grain and scaling obey the luma-row contract.
    ///
    /// # Panics
    /// Panics for short buffers or reversed clipping bounds.
    #[allow(clippy::too_many_arguments)]
    pub fn av1_grain_chroma_row(
        &self,
        source: &[u8],
        luma: &[u8],
        grain: &[i16],
        scaling: &crate::av1_film_grain::Av1GrainScaling,
        bounds: [u8; 2],
        coefficients: [i16; 3],
        out: &mut [u8],
    ) {
        assert!(grain.len() >= source.len() && out.len() >= source.len());
        assert!(luma.len() >= source.len().saturating_mul(2).saturating_sub(1));
        assert!(bounds[0] <= bounds[1]);
        // SAFETY: as for `av1_grain_luma_row`.
        unsafe {
            (self.av1_grain.chroma)(
                source,
                luma,
                grain,
                scaling,
                bounds,
                coefficients,
                &mut out[..source.len()],
            );
        }
    }

    /// Blend grain overlap samples in -128..=127 with AV1's weights:
    /// `[27, 17]`, `[17, 27]`, or `[23, 22]`.
    ///
    /// # Panics
    /// Panics if `old` is shorter than `new`, or weights are invalid.
    pub fn av1_grain_blend_row(&self, old: &[i16], new: &mut [i16], weights: [i16; 2]) {
        assert!(old.len() >= new.len());
        assert!(matches!(weights, [27, 17] | [17, 27] | [23, 22]));
        // SAFETY: table is selected for this CPU and buffers checked.
        unsafe { (self.av1_grain.blend)(old, new, weights) }
    }
}

impl Dsp {
    pub(crate) fn vvc_refinement_kernels(&self) -> crate::vvc_refinement::Kernels {
        self.vvc_refinement
    }

    /// Scaled AV1 block inter prediction at final 8-bit precision via this
    /// ISA's kernel (`av1_mc::scaled`).
    #[inline]
    pub(crate) fn av1_predict_scaled(
        &self,
        filters: crate::av1_mc::filter::Av1Filters,
        window: &PlaneView<'_>,
        size: (usize, usize),
        axes: (
            crate::av1_mc::convolve::Av1McAxis,
            crate::av1_mc::convolve::Av1McAxis,
        ),
        dst: &mut PlaneViewMut<'_>,
    ) {
        // SAFETY: `self.av1_scaled` matches the validated CPU ISA; the kernel
        // validates window, destination, axes and block dimensions.
        unsafe { (self.av1_scaled.predict)(filters, window, size, axes, dst) }
    }

    /// Scaled AV1 block inter prediction at compound precision via this ISA's
    /// kernel (`av1_mc::scaled`).
    #[inline]
    pub(crate) fn av1_predict_compound_scaled(
        &self,
        filters: crate::av1_mc::filter::Av1Filters,
        window: &PlaneView<'_>,
        size: (usize, usize),
        axes: (
            crate::av1_mc::convolve::Av1McAxis,
            crate::av1_mc::convolve::Av1McAxis,
        ),
        dst: &mut [i16],
        dst_stride: usize,
    ) {
        // SAFETY: `self.av1_scaled` matches the validated CPU ISA; caller
        // checked `dst` and the kernel validates window, axes and dimensions.
        unsafe {
            (self.av1_scaled.predict_compound)(filters, window, size, axes, dst, dst_stride);
        }
    }

    /// Resample all of `src` into all of `dst` with a Lanczos-3 polyphase
    /// filter (`nc_dsp::resample`), using this ISA's SIMD kernels.
    ///
    /// # Panics
    ///
    /// If `src` or `dst` has a zero dimension.
    pub fn resample_plane(&self, src: &PlaneView<'_>, dst: &mut PlaneViewMut<'_>) {
        let mut plan = crate::resample::PlaneResampler::new(
            (src.width(), src.height()),
            (dst.width(), dst.height()),
        );
        self.resample_plane_with(&mut plan, src, dst);
    }

    /// Resample all of `src` into the top-left `plan.out_size()` region of `dst`
    /// with a precomputed [`PlaneResampler`](crate::resample::PlaneResampler),
    /// reusing `plan`'s internal scratch buffers (zero heap allocations).
    ///
    /// # Panics
    ///
    /// If `src` does not match `plan.in_size()` or `dst` is smaller than
    /// `plan.out_size()`.
    pub fn resample_plane_with(
        &self,
        plan: &mut crate::resample::PlaneResampler,
        src: &PlaneView<'_>,
        dst: &mut PlaneViewMut<'_>,
    ) {
        assert_eq!(
            (src.width(), src.height()),
            plan.in_size(),
            "resample_plane_with source {}x{} does not match plan {:?}",
            src.width(),
            src.height(),
            plan.in_size()
        );
        let (out_w, out_h) = plan.out_size();
        assert!(
            dst.width() >= out_w && dst.height() >= out_h,
            "resample_plane_with destination {}x{} is smaller than plan {:?}",
            dst.width(),
            dst.height(),
            plan.out_size()
        );
        // SAFETY: `self.resample` matches the validated CPU ISA, `src` matches
        // `plan.in_size()`, and `dst` covers at least `plan.out_size()`.
        unsafe { (self.resample)(plan, src, dst) }
    }
}

/// Border extension through the ISA's border stores
/// (`crate::border_store`: non-temporal on x86).
impl nc_common::BorderStores for Dsp {
    fn fill(&self, dst: &mut [u8], value: u8) {
        // SAFETY: the table matches a supported ISA (checked in `with_isa`).
        unsafe { (self.border_store.fill)(dst, value) }
    }

    fn copy(&self, dst: &mut [u8], src: &[u8]) {
        assert_eq!(dst.len(), src.len(), "border copy lengths differ");
        // SAFETY: the table matches a supported ISA (checked in `with_isa`).
        unsafe { (self.border_store.copy)(dst, src) }
    }

    fn finish(&self) {
        // SAFETY: the table matches a supported ISA (checked in `with_isa`).
        unsafe { (self.border_store.fence)() }
    }
}

impl Dsp {
    /// Copy the source's top-left rectangle covering the destination.
    ///
    /// # Panics
    /// If the source is smaller than the destination.
    pub fn copy_plane(&self, src: &PlaneView<'_>, dst: &mut PlaneViewMut<'_>) {
        let (w, h, ds) = (dst.width(), dst.height(), dst.stride());
        assert!(
            src.width() >= w && src.height() >= h,
            "copy source too small"
        );
        // SAFETY: both validated views cover the rectangle and the ISA was checked.
        unsafe {
            (self.copy_plane)(src.data(), src.stride(), dst.data_mut(), ds, w, h);
        }
    }
}
