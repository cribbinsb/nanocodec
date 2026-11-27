//! The AV2 backend of the shared engine ([`Av2Backend`], the only encode
//! path) and the AV2 decoder ([`Decoder`]).
//!
//! The encoder codes 8-bit 4:2:0 I/P pictures: every intra mode except
//! chroma-from-luma, lossless/lossy residuals, skip and quarter-pel motion.
//! The decoder ([`general`]) decodes other encoders' single-tile 8-bit
//! 4:2:0 AV2 streams as AVM v1.0.0 writes them (30 of 30 foreign streams;
//! no AV2 conformance suite exists); 10-bit, monochrome, several tiles and
//! reference scaling are not decoded yet (docs/handoff/av2.md,
//! docs/FAQ.md#conformance).
#![forbid(unsafe_code)]

pub mod obu;

pub mod ccso;
mod ccso_search;
pub mod cdef;
pub mod cdf_models;
pub mod deblock;
mod delta_q;
pub mod entropy;
pub mod frame_header;
pub mod general;
pub mod sequence;

pub mod transform;
mod tx_partition;
pub mod tx_size;
pub mod tx_type;

pub mod coeff;
mod coeff_tables;
mod coeff_trim;
pub mod interp_filter;
pub mod partition;

mod backend;
mod block_coder;
mod block_state;
mod chroma_weight;
mod packet;
mod physical_slots;
mod pre_tx_skip;
// The tile decoder's analysis records (feature `analysis`).
#[cfg(feature = "analysis")]
mod tile_analysis;
mod tile_decoder;
pub use backend::Av2Backend;
pub use general::decoder::{Decoder, temporal_unit_layer};
mod intra_edges;
pub mod intra_modes;
pub mod motion;
pub mod mv_refs;
pub mod prediction;
