//! nanocodec's AV1 backend.
//!
//! Implements [`nc_encoder::CodecBackend`] for AV1 Main profile, 8-bit
//! 4:2:0 (subset: `docs/ARCHITECTURE.md` §7): shown key frames and shown
//! inter frames predicted from one reference slot (the previous frame, the
//! engine's hierarchical-P reference with temporal-layer OBU extensions, or
//! the long-term reference), on square and rectangular partitions of 64x64
//! superblocks; directional, filter and chroma-from-luma intra; sub-pel
//! motion; trellis-quantised residuals with adaptive CDFs; deblocking and
//! CDEF, no loop restoration. The decoder (`decoder/`) aims at full Main
//! profile conformance.
//!
//! Module map:
//! - `backend`: block and frame syntax, reconstruction — [`Av1Backend`];
//! - `obu`: OBU framing (temporal-layer extensions), sequence header
//!   (operating points) and frame headers (reference slots);
//! - `modeinfo`: per-4x4 state of coded blocks (contexts, neighbours);
//! - `mvref`: the reference motion-vector list (`find_mv_stack`);
//! - `mvcoding`: motion-vector difference coding and its bit cost;
//! - `inter`: motion-compensated prediction;
//! - `cdef`: the CDEF loop filter and its strength choice;
//! - `grain`: film grain parameters from the texture coding removed;
//! - `cdf`: adaptive CDFs and the per-frame CDF context;
//! - `coeff`: coefficient coding and its contexts;
//! - `range_coder`: the symbol (range) encoder;
//! - `transform`, `quant`, `predict`: the signal path;
//! - `tables`: constants, mostly generated from libaom.
//!
//! Ported algorithms keep libaom attribution (BSD-2-Clause, Alliance for
//! Open Media) in their module docs; see `THIRD_PARTY.md`.

#![forbid(unsafe_code)]

mod backend;
mod cdef;
mod cdf;
mod cfl;
mod coeff;
mod decoder_range;
mod grain;
mod inter;
mod intra;
#[cfg(test)]
mod inverse_bounds;
mod loopfilter;
mod mds3;
mod modeinfo;
mod mvcoding;
mod mvref;
mod obu;
mod predict;
mod quant;
mod range_coder;
mod reference_store;
mod restoration;
mod scaled_reference;
mod superres;
mod tables;
mod transform;
mod txsize;

pub use backend::decoder::{
    Decoder, MAX_OPERATING_POINT, filter_temporal_unit_layers, temporal_unit_layer,
    temporal_unit_layers,
};
pub use backend::{Av1Backend, FRAME_IDS_PARAM};
pub use grain::enable_film_grain;
