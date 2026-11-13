//! Kernel verification and benchmarking (the nanocodec "checkasm").
//!
//! Every SIMD kernel must return exactly what its scalar reference returns.
//! [`check`] runs each kernel of an ISA on random and adversarial inputs:
//! every block size, odd and unaligned strides, all-zero / all-255 blocks,
//! and maximum-difference blocks. It reports every mismatch. [`bench`]
//! times each kernel per ISA.
//!
//! Used by the `nc-checkasm` binary (`--bench` for timings) and by this
//! crate's integration test, so `cargo test` covers the SIMD paths of the
//! machine it runs on.

#![forbid(unsafe_code)]

pub mod bench;
mod bench_av1_grid;
mod bench_av1_intra;
mod bench_av1_runs;
mod bench_av2_intra;
mod bench_extra;
mod bench_filters;
mod bench_rows;
mod bench_vp9_filter;
mod bench_vp9_line;
mod bench_vp9_mc;
pub mod check;
mod check_av1_deblock;
mod check_av1_grain;
mod check_av1_grid;
mod check_av1_intra;
mod check_av1_inverse;
mod check_av1_mask;
mod check_av1_restoration;
mod check_av1_scaled;
mod check_av1_symbol;
mod check_av1_warp;
mod check_av2_intra;
mod check_av2_typed;
mod check_border;
mod check_cdef_interior;
mod check_codec;
mod check_codec_txfm;
mod check_display;
mod check_filter_runs;
mod check_filters;
mod check_gram;
mod check_h264_intra;
mod check_hevc_intra;
mod check_hevc_put32;
mod check_hevc_weight;
mod check_sub_blocks;
mod check_transpose;
mod check_vp9_filter;
mod check_vp9_intra;
mod check_vp9_inverse;
mod check_vp9_line;
mod check_vp9_mc;
mod check_vp9_quant;
mod check_vvc;
mod check_vvc_alf;
mod check_vvc_angular;
mod check_vvc_chroma_edge;
mod check_vvc_deblock_edge;
mod check_vvc_inverse;
mod check_vvc_mts;
mod check_vvc_refinement;
mod check_vvc_tx64;
pub mod rng;

pub use bench::{BenchResult, bench, bench_av1_runs};
pub use check::{CheckReport, check};

mod bench_memory;
mod check_memory;
