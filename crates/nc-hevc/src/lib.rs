//! nanocodec's HEVC backend and its Main-profile decoder.
//!
//! Implements [`nc_encoder::CodecBackend`] for HEVC Main profile, 8-bit
//! 4:2:0. Current scope: IDR pictures and single-reference P pictures (in a
//! flat chain or hierarchical-P temporal sub-layers) of
//! 2Nx2N CUs (8x8 to 64x64; 8x8 intra CUs may be `PART_NxN`, intra transform
//! trees reach 4x4 with the DST) — angular intra, skip, merge and AMVP inter with
//! quarter-sample motion — at a fixed QP per picture, with the deblocking
//! filter and SAO (edge offset) on.
//!
//! [`Decoder`] is a complete HEVC Main / Main Still Picture profile decoder
//! (any conforming 8-bit 4:2:0 stream, checked MD5-exact against HM on the
//! 136 Main streams of the JCT-VC conformance set); it shares only normative
//! helpers with the encoder (intra prediction, scans, transforms, DSP kernels).
//!
//! Module map:
//! - `backend`: CTU and CU syntax, reconstruction — [`HevcBackend`];
//! - `grid`: per-picture state of coded blocks (contexts, MV prediction,
//!   deblocking);
//! - `mvpred`, `inter`: merge/AMVP candidate lists and motion compensation;
//! - `collocated`: TMVP collocated motion per reference slot, POC scaling;
//! - `deblock`, `sao`: the in-loop filters;
//! - `syntax`: entropy coding of the recorded slice data;
//! - `headers`: NAL unit headers, VPS/SPS/PPS (temporal sub-layers and
//!   reference picture sets), slice header;
//! - `dec`: the decoder (parameter sets, slice data, motion prediction,
//!   in-loop filters, POC/RPS/DPB and output order; see its module map);
//! - `cabac`, `contexts`: the arithmetic coder and its context variables;
//! - `rate`: CABAC rate estimates for the encoder's coding trials;
//! - `residual`, `scan`: `residual_coding()` and coefficient scans;
//! - `transform`, `quant`, `predict`: the signal path.
//!
//! Ported algorithms and tables keep kvazaar attribution (BSD-3-Clause,
//! Tampere University) in their module docs; see `THIRD_PARTY.md`.

#![forbid(unsafe_code)]

mod backend;
mod cabac;
mod collocated;
mod contexts;
mod deblock;
mod dec;
mod grid;
mod headers;
mod inter;
mod mvpred;
mod predict;
mod quant;
mod quant_group;
mod rate;
mod residual;
mod residual_trellis;
mod sao;
mod scan;
mod syntax;
mod transform;

pub use backend::HevcBackend;

pub use dec::{Decoder, OutputPicture};
pub use headers::access_unit_layer;
