//! Pixel and sample kernels with runtime-selected SIMD implementations.
//!
//! This is the only crate (besides the C ABI) allowed to use `unsafe`, and it
//! is confined to the per-ISA kernel modules under `pixel` and `sample`, the
//! dispatch calls in [`Dsp`], and two small helpers outside the table
//! (`prefetch` hints and `block_copy` row moves). Everything is exposed through safe methods on
//! [`Dsp`] that take [`PlaneView`](nc_common::PlaneView)s.
//!
//! Conventions (see `docs/ARCHITECTURE.md` for the full story):
//!
//! - Every kernel has a scalar reference in `pixel::scalar`. SIMD versions
//!   must return *bit-identical* results; `nc-checkasm` verifies this for
//!   every kernel, block size and ISA.
//! - A [`Dsp`] instance owns its function table; there is no global state.
//!   Pick the best implementation with [`Dsp::new`], or a specific one with
//!   [`Dsp::with_isa`] (used by tests and benchmarks).
//! - SIMD modules contain kernels only, never control logic.

mod analysis_stats;
pub use analysis_stats::{AnalysisBlock, AnalysisChanges, AnalysisSummary};

pub mod av1_film_grain;
#[allow(unsafe_code)]
mod av1_forward;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
mod av1_forward_simd;
mod av1_intra;
mod av1_inverse;
pub mod av1_mc;
pub mod av1_restoration;
pub mod av1_symbol;
pub mod av2_dec;
mod av2_intra;
mod av2_txfm;
mod block;
mod block_copy;
mod cdef;
#[allow(unsafe_code)]
pub mod cfl_sse;
mod codec_txfm;
#[allow(unsafe_code)]
mod constant_add;
mod convert;
mod cpu;
mod deblock;
mod diff;
mod dispatch;
pub mod display;
mod edge;
mod end_of_block;
mod gram;
mod h264_intra;
mod hevc_intra;
mod hevc_weight;
mod high_depth;
mod intra_estimate;
mod intra_runs;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[allow(unsafe_code)]
mod lpf_bytes;
mod mc;
pub mod ms_ssim;
#[allow(unsafe_code)]
mod nn;
mod pixel;
#[allow(unsafe_code)]
mod prefetch;
/// The AV1 / AV2 predefined quantizer matrices.
pub mod quantizer_matrix;
pub mod resample;
pub mod reversible_wht4;
mod sample;
mod sao;
mod simd;
mod transpose;
mod txfm;
mod vp9_deblock;
mod vp9_intra;
#[allow(unsafe_code)]
mod vp9_inverse;
#[allow(unsafe_code)]
mod vp9_loop_filter;
mod vp9_mc;
#[allow(unsafe_code)]
mod vp9_mc_high;
mod vp9_mc_scaled;
#[allow(unsafe_code)]
mod vp9_mc_simd;
#[allow(unsafe_code)]
mod vp9_quant;
mod vp9_txfm;
pub use edge::emulate_edges;
pub use end_of_block::eob_table;
pub use gram::{
    GRAM_MAX_MAGNITUDE, GRAM_MAX_ROWS, GRAM_ROW_ALIGN, GRAM_ROW_MAX_LEN, gram_index, gram_terms,
};
pub use h264_intra::four::{H264Intra4x4Edges, H264Intra4x4Refs};
pub use h264_intra::{
    H264_INTRA_8X8_SIDE, H264Intra8x8Edges, H264Intra8x8Refs, h264_intra_8x8_mode, h264_plane,
};
pub use hevc_intra::PlanarEdges as HevcPlanarEdges;
pub use hevc_weight::HevcWeights;
pub use high_depth::av1::{Av1Coefficient, Av1Pixel};
pub use high_depth::vp9::{Vp9Passes, Vp9Pixel};
pub use high_depth::{DepthKernels, H264Pixel, Pixel};
pub use intra_estimate::{ESTIMATE_EDGE_LEN, ESTIMATE_MAX_SIDE, EstimateStats};
pub use vp9_deblock::{Vp9FilterLimits, Vp9FilterWidth, vp9_filter_edge};
pub use vp9_intra::{Vp9IntraEdges, Vp9IntraMode};
pub use vp9_mc::{
    VP9_MAX_SCALED_STEP, VP9_TAPS, VP9_TAPS_BEFORE, Vp9Filter, vp9_convolve, vp9_convolve_scaled,
};
pub use vp9_mc_scaled::Vp9ScaledScratch;
pub use vp9_quant::Vp9QuantStep;
pub use vp9_txfm::{
    Vp9Pass, vp9_forward_dct8, vp9_forward_transform, vp9_inverse_add, vp9_inverse_dct8,
    vp9_inverse_transform,
};

pub use av1_inverse::{
    AV1_LOW8_MAX_EOB, AV1_LOW16_MAX_EOB, AV1_MAX_CODED_SIDE, Av1TxSize, Av1TxType,
    av1_dct_nonzero_side, av1_inverse_nonzero_rows,
};
pub use av2_txfm::Av2TxSize;
pub use av2_txfm::typed::{Av2Kernel, Av2TxType};
pub use block::BlockSize;
pub use block_copy::{BlockCopy, copy_square, copy_square_samples, copy_square_view};
pub use cdef::band::{
    CDEF_MAX_SETS, CdefLines, CdefSets, CdefStrengths, CdefUnit, cdef_adjust_strength,
};
pub use cdef::{CDEF_PADDING, CDEF_UNAVAILABLE, CdefBlock, CdefSource};
pub use codec_txfm::{
    H264_ZIGZAG_4X4, H264_ZIGZAG_8X8, H264Quantizer, HevcForwardMode, HevcForwardStaging,
    HevcQuantizer, HevcTransform, VvcExtendedBlock, VvcLfnst, VvcQuantizer, VvcTransformKind,
    h264_blocks_of_8x8, h264_coefficient_cost, scale_coefficient,
};
pub use convert::DisplayMatrix;
pub use cpu::Isa;
pub use deblock::av1::{AV1_EDGE_LINES, AV1_EDGE_REACH, Av1EdgeFilter, Av1FilterLength};
pub use deblock::av2::{AV2_EDGE_LINES, Av2EdgeFilter};
pub use deblock::hevc_high::{HevcLumaPiece, MAX_BATCH_PIECES as HEVC_DEBLOCK_BATCH_PIECES};
pub use deblock::lanes::LoopFilterLanes;
pub use deblock::runs::{EdgeRun, HevcChromaFilter, HevcLumaFilter, VvcLumaFilter};
pub use deblock::{H264EdgeParams, H264MbEdges};
pub use diff::{SUBPEL_VARIANCE_POSITIONS, Variance, VarianceEnergy};
pub use dispatch::{
    Dsp, H264Predictor, WindowVariance, av1_inverse_add_i16_dav1d_reference,
    av1_inverse_has_dav1d_model,
};
pub use mc::{
    Filter, SubBlockMotion, VVC_RPR_POSITION_BITS, VvcRprFilter, VvcScaledBlock, vvc_predict_scaled,
};
pub use nn::{
    NN_ACCUMULATOR_ALIGN, NN_ACTIVATION_MAX, NN_DENSE_INPUT_ALIGN, NN_MAX_ACCUMULATOR_SHIFT,
    NN_MAX_DENSE_SHIFT, nn_dense_max_bias,
};
pub use pixel::{
    GRID_BLOCKS, GRID_LARGE_BLOCKS, GRID_SMALL_BLOCKS, GridBest, GridRows, MacroblockSad,
};
pub use prefetch::prefetch_block;
pub use resample::PlaneResampler;
pub use sample::Moments;
pub use sao::{
    MAX_BAND_SAMPLES, SAO_BANDS, SAO_EO_NEIGHBOURS, SaoBandStats, SaoCtb, SaoLimits, SaoStats,
    sao_edge_restricted,
};
pub use transpose::transpose_square;
pub use txfm::low_keep as av1_low_keep;
pub use txfm::{Av1Quantizer, DctSize};
pub use vp9_loop_filter::{VP9_EDGE_LINES, Vp9EdgeFilter};

/// Shared basic intra prediction kernels and reconstructed edges.
pub mod intra_prediction;

mod vvc_deblock;
pub use vvc_deblock::{
    VVC_DEBLOCK_MAX_BETA_PRIME, VVC_DEBLOCK_MAX_TC_PRIME, vvc_deblock_beta, vvc_deblock_chroma,
    vvc_deblock_luma, vvc_deblock_tc,
};

mod vvc_bdof;
mod vvc_dmvr;
mod vvc_lmcs;
mod vvc_optical_flow;
mod vvc_refinement;
pub use vvc_lmcs::VvcLmcs;

mod vvc_recon;
pub use vvc_recon::{vvc_act_inverse, vvc_joint_chroma};
mod vvc_intra;
pub use vvc_intra::rectangular::vvc_intra_filter_references;
pub use vvc_intra::{
    VvcIntraMode, vvc_cclm_downsample, vvc_cclm_downsample_collocated, vvc_cclm_predict,
};

mod vvc_alf;
pub use vvc_alf::{
    VVC_ALF_BLOCK, VVC_ALF_CLIP_INDICES, VVC_ALF_DOT_LANES, VVC_ALF_GRAM_ALIGN,
    VVC_ALF_GRAM_COLUMNS, VVC_ALF_GRAM_MAX_MAGNITUDE, VVC_ALF_GRAM_MAX_SAMPLES, VVC_ALF_GRAM_TERMS,
    VVC_ALF_MAX_BIT_DEPTH, VVC_ALF_MAX_GROUP_SUM, VVC_ALF_MAX_SCALAR_BIT_DEPTH, VVC_ALF_REACH,
    VVC_ALF_SAMPLE_LANES, VVC_ALF_SAMPLE_STEP, VVC_ALF_SAMPLES, VVC_ALF_TAP_OFFSETS, VVC_ALF_TAPS,
    VVC_ALF_TRANSPOSES, vvc_alf_clip_value, vvc_alf_gram_index, vvc_alf_max_group_sum,
};

mod vvc_geo;
mod vvc_mts;
mod vvc_tx64;
pub use vvc_geo::VvcGeometric;

pub use diff::Residual16;
