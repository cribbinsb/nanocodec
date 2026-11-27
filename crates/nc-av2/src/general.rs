//! The general AV2 decoder: 8-, 10- and 12-bit 4:2:0 and monochrome
//! (4:0:0) streams of any encoder that follows AVM v1.0.0, not only the
//! subset our own encoder writes (whose 8-bit frames take the subset tile decoder,
//! [`crate::tile_decoder`]). Deeper samples decode in a `u16` instance of
//! the same code ([`sample`], [`instance`]).
//!
//! Not supported yet (each rejected as [`nc_common::decode::DecodeError::Unsupported`]
//! by name): the 4:2:2 and 4:4:4 formats and extended layers
//! (several streams multiplexed). Embedded layers decode,
//! but not yet the temporal motion candidates of a reference of the same
//! display order in a lower layer (docs/handoff/av2.md "Multi-frame headers
//! and layers").
//!
//! Structure and arithmetic follow dav2d (commit 237229e; Copyright ©
//! 2018-2026 `VideoLAN` and dav2d authors, BSD-2-Clause, see
//! `docs/licenses/dav2d`), checked against AVM v1.0.0
//! (BSD-3-Clause-Clear plus the AOM patent licence, `docs/licenses/avm`),
//! reimplemented in Rust. See `THIRD_PARTY.md`.
//!
//! Frames may have any tile layout and come in several tile groups
//! ([`block::decode_tiles`], `decoder`); AVM's
//! `disable_loopfilters_across_tiles` is read but, as in dav2d, not
//! applied.

pub mod bits;
pub mod block;
pub mod cdfs;
pub mod decoder;
pub mod film_grain;
pub mod filters;
pub mod frame_header;
pub mod instance;
pub mod multi_frame_header;
pub mod output;
pub mod references;
pub mod sample;
pub mod sequence;
mod subset;
pub mod tiles;
pub mod warp;
