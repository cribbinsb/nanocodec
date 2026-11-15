//! The general AV2 decoder: 8-bit 4:2:0 streams of any encoder that follows
//! AVM v1.0.0, not only the subset our own encoder writes
//! (its blocks still go through the subset tile decoder,
//! [`crate::tile_decoder`], until the general block layer exists).
//!
//! Structure and arithmetic follow dav2d (commit 237229e; Copyright ©
//! 2018-2026 `VideoLAN` and dav2d authors, BSD-2-Clause, see
//! `docs/licenses/dav2d`), checked against AVM v1.0.0
//! (BSD-3-Clause-Clear plus the AOM patent licence, `docs/licenses/avm`),
//! reimplemented in Rust. See `THIRD_PARTY.md`.

pub mod bits;
pub mod decoder;
pub mod frame_header;
pub mod output;
pub mod references;
pub mod sequence;
mod subset;
pub mod tiles;
