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

pub mod av1_film_grain;
mod av1_intra;
mod av1_inverse;
pub mod av1_mc;
pub mod av1_restoration;
pub mod av1_symbol;
mod av2_intra;
mod av2_txfm;
mod block;
mod block_copy;
mod border_store;
mod cdef;
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
mod gram;
mod h264_intra;
mod hevc_intra;
mod hevc_weight;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
mod lpf_bytes;
mod mc;
mod pixel;
#[allow(unsafe_code)]
mod prefetch;
pub mod resample;
pub mod reversible_wht4;
mod sample;
mod sao;
mod simd;
mod txfm;
mod vp9_deblock;
mod vp9_intra;
#[allow(unsafe_code)]
mod vp9_inverse;
#[allow(unsafe_code)]
mod vp9_loop_filter;
mod vp9_mc;
#[allow(unsafe_code)]
mod vp9_mc_simd;
#[allow(unsafe_code)]
mod vp9_quant;
mod vp9_txfm;
pub use edge::emulate_edges;
pub use gram::{
    GRAM_MAX_MAGNITUDE, GRAM_MAX_ROWS, GRAM_ROW_ALIGN, GRAM_ROW_MAX_LEN, gram_index, gram_terms,
};
pub use h264_intra::{
    H264_INTRA_8X8_SIDE, H264Intra8x8Edges, H264Intra8x8Refs, h264_intra_8x8_mode,
};
pub use hevc_intra::PlanarEdges as HevcPlanarEdges;
pub use hevc_weight::HevcWeights;
pub use vp9_deblock::{Vp9FilterLimits, Vp9FilterWidth, vp9_filter_edge};
pub use vp9_intra::{Vp9IntraEdges, Vp9IntraMode};
pub use vp9_mc::{
    VP9_MAX_SCALED_STEP, VP9_TAPS, VP9_TAPS_BEFORE, Vp9Filter, vp9_convolve, vp9_convolve_scaled,
};
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
pub use block_copy::{copy_square, copy_square_view};
pub use cdef::band::{
    CDEF_MAX_SETS, CdefLines, CdefSets, CdefStrengths, CdefUnit, cdef_adjust_strength,
};
pub use cdef::{CDEF_PADDING, CDEF_UNAVAILABLE, CdefBlock, CdefSource};
pub use codec_txfm::{
    H264Quantizer, HevcForwardMode, HevcForwardStaging, HevcQuantizer, HevcTransform, VvcLfnst,
    VvcQuantizer, VvcTransformKind,
};
pub use cpu::Isa;
pub use deblock::av1::{AV1_EDGE_LINES, AV1_EDGE_REACH, Av1EdgeFilter, Av1FilterLength};
pub use deblock::av2::{AV2_EDGE_LINES, Av2EdgeFilter};
pub use deblock::runs::{EdgeRun, HevcChromaFilter, HevcLumaFilter};
pub use deblock::{H264EdgeParams, H264MbEdges};
pub use diff::{SUBPEL_VARIANCE_POSITIONS, Variance, VarianceEnergy};
pub use dispatch::{Dsp, H264Predictor};
pub use mc::{Filter, SubBlockMotion};
pub use pixel::{GRID_BLOCKS, GridBest, GridRows};
pub use prefetch::prefetch_block;
pub use resample::PlaneResampler;
pub use sample::Moments;
pub use sao::{SAO_EO_NEIGHBOURS, SaoStats};
pub use txfm::low_keep as av1_low_keep;
pub use txfm::{Av1Quantizer, DctSize};
pub use vp9_loop_filter::{VP9_EDGE_LINES, Vp9EdgeFilter};

/// Shared basic intra prediction kernels and reconstructed edges.
pub mod intra_prediction;

mod vvc_deblock;
pub use vvc_deblock::{vvc_deblock_chroma, vvc_deblock_luma};

mod vvc_bdof;
mod vvc_dmvr;
mod vvc_lmcs;
mod vvc_optical_flow;
mod vvc_refinement;
pub use vvc_lmcs::VvcLmcs;

mod vvc_recon;
pub use vvc_recon::vvc_joint_chroma;
mod vvc_intra;
pub use vvc_intra::rectangular::vvc_intra_filter_references;
pub use vvc_intra::{
    VvcIntraMode, vvc_cclm_downsample, vvc_cclm_downsample_collocated, vvc_cclm_predict,
};

mod vvc_alf;
pub use vvc_alf::{
    VVC_ALF_BLOCK, VVC_ALF_DOT_LANES, VVC_ALF_GRAM_ALIGN, VVC_ALF_GRAM_COLUMNS,
    VVC_ALF_GRAM_MAX_MAGNITUDE, VVC_ALF_GRAM_MAX_SAMPLES, VVC_ALF_GRAM_TERMS,
    VVC_ALF_MAX_GROUP_SUM, VVC_ALF_REACH, VVC_ALF_SAMPLE_LANES, VVC_ALF_SAMPLE_STEP,
    VVC_ALF_SAMPLES, VVC_ALF_TAP_OFFSETS, VVC_ALF_TAPS, VVC_ALF_TRANSPOSES, vvc_alf_gram_index,
};

mod vvc_geo;
mod vvc_mts;
mod vvc_tx64;
pub use vvc_geo::VvcGeometric;

mod reconstruct;
pub use reconstruct::Residual32;
