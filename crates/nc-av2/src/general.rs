//! The general AV2 decoder: 8-bit 4:2:0 streams of any encoder that follows
//! AVM v1.0.0, not only the subset our own encoder writes
//! ([`crate::stream`] decodes that subset on its own faster path).
//!
//! Structure and arithmetic follow dav2d (commit 237229e; Copyright ©
//! 2018-2026 `VideoLAN` and dav2d authors, BSD-2-Clause, see
//! `docs/licenses/dav2d`), checked against AVM v1.0.0
//! (BSD-3-Clause-Clear plus the AOM patent licence, `docs/licenses/avm`),
//! reimplemented in Rust. See `THIRD_PARTY.md`.

pub mod bits;
pub mod references;
pub mod sequence;
pub mod tiles;
