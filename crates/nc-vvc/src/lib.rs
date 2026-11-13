//! H.266/VVC backend using the shared encoder engine.
//!
//! Native 8-bit 4:2:0, quadtree intra/forward inter prediction, full DCT-II
//! residuals, deblocking and SAO.
#![forbid(unsafe_code)]
mod affine;
// The decoders' analysis records (feature `analysis`).
mod affine_candidates;
mod affine_search;
mod alf;
mod alf_aps;
#[cfg(feature = "analysis")]
mod analysis;
mod backend;
mod cabac;
mod chroma;
mod chroma_qp;
mod coding_tree;
mod contexts;
mod headers;
mod recon;
mod syntax;
mod transform_tree;
pub use backend::{VvcBackend, VvcParams};
mod decoder;
mod decoder_blocks;
mod decoder_cabac;
mod decoder_dpb;
mod decoder_headers;
mod decoder_metadata;
pub use decoder::Decoder;
pub use headers::access_unit_layer;
mod decoder_syntax;
mod inter;
mod inter_cost;
mod motion;
mod references;
mod temporal;

mod coeff;
mod coeff_contexts;
mod dep_quant;
mod quant_group;

mod inter_intra;
mod intra;
mod isp;
mod mts;

mod filter;

mod sao;

mod scan;

mod trellis;

#[cfg(test)]
mod angular_conformance;
