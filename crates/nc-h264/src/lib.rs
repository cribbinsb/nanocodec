//! nanocodec's H.264 backend.
//!
//! Implements [`nc_encoder::CodecBackend`] for H.264 High profile, 8-bit
//! 4:2:0, CABAC, one slice per picture, one active reference picture (from
//! up to two kept for hierarchical-P structures). Key frames
//! are IDR pictures of `I_16x16` DC macroblocks; inter frames are P pictures
//! of `P_L0_16x16` (quarter-sample vectors), `P_Skip` and `I_16x16`
//! macroblocks, at a fixed QP. The deblocking filter is on. Pictures that
//! are not whole macroblocks are padded and cropped (SPS frame cropping).
//!
//! Module map:
//! - `backend`: access units, slice loop, mode signalling costs — [`H264Backend`];
//! - `params`: SPS, PPS, slice headers, level selection;
//! - `references`: temporal layers as `nal_ref_idc`, `frame_num`, sliding
//!   window and reference list modification;
//! - `macroblock`: prediction, transform, quantisation, reconstruction
//!   (transform and quantiser kernels come from nc-dsp);
//! - `syntax`: macroblock-layer CABAC syntax; `residual`: residual blocks;
//!   `residual_rd`: rate-distortion pruning of inter residuals;
//! - `mbinfo`: per-macroblock state and neighbourhoods; `mvpred`: MV prediction;
//! - `mc`: motion-compensated interpolation; `predict`: intra DC prediction;
//! - `deblock`: the in-loop deblocking filter;
//! - `cabac`: the arithmetic encoder and context variables;
//! - `tables`: constants; CABAC and deblocking tables are generated from openh264.
//!
//! The generated tables keep openh264 attribution (BSD-2-Clause, Cisco
//! Systems); see `THIRD_PARTY.md`.

#![forbid(unsafe_code)]

mod backend;
mod cabac;
mod cavlc;
mod chroma_offsets;
mod deblock;
mod decoder;
mod intra4x4;
mod intra8x8;
mod intra_nxn;
mod macroblock;
mod mbinfo;
mod mbqp;
mod mc;
mod mvpred;
mod params;
mod partition;
mod predict;
mod references;
mod residual;
mod residual_rd;
mod syntax;
mod tables;

pub use backend::{H264Backend, H264Params};
pub use decoder::{DecodedFrame, Decoder, OutputQueue};
pub use references::access_unit_layer;
