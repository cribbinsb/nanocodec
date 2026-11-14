//! Native VP9 Profile 0 encoder backend for the shared engine, and a
//! VP9 Profile 0 decoder for any conforming stream (`decoder/`).
//!
//! The engine plans partitions (64x64 superblocks down to 8x8) and picks
//! each block's prediction; this crate writes VP9 syntax: DC intra,
//! single-reference inter prediction, transforms up to 32x32, coefficient
//! tokens and the deblocking filter. See `backend.rs` for the tool set and
//! `header.rs` for the emitted subset.
#![forbid(unsafe_code)]

mod backend;
pub use backend::Vp9Backend;
pub mod bool_coder;
mod decoder;
mod extract;
pub mod stream;
pub mod superframe;
pub use decoder::{DecodedFrame, Decoder};
pub use extract::{
    ExtractedVp9Packet, MAX_VP9_OPERATING_POINT, Vp9ExtractTarget, Vp9StreamExtractor,
};
pub use superframe::{
    escape_single_frame, filter_superframe_layers, pack_coded_layers, pack_spatial_tu,
    pack_superframe, parse_uncompressed_header_info, split_superframe, strip_superframe_to_base,
};

mod coeff;
mod contexts;
mod header;
mod hybrid;
mod inter_pred;
mod intra_pred;
mod loopfilter;
mod motion;
mod mvref;
mod probs;
mod recon;
mod recorder;
mod segmentation;
mod stress;
mod subexp;
mod tables;
mod transform;
mod tree;
mod types;
