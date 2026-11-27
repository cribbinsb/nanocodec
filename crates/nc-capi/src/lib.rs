//! C ABI for embedding the nanocodec encoder and decoder in C/C++ applications.
//!
//! The declarations are in `include/nanocodec.h`; `examples/encode.c` shows
//! the whole life cycle. This crate is a thin layer:
//!
//! - [`session`]: a codec-independent, entirely safe encoder session
//!   (configuration translation, the per-codec engine instance, packet
//!   storage), testable from Rust;
//! - [`receiver`]: the same for decoding: our decoder, the picture on
//!   screen and each packet's outcome (decoded, or frozen with the reason
//!   and the frame to report lost);
//! - [`rtp`]: RTP packetizing of a session's packets and depacketizing for
//!   a decoder (`nanocodec-rtp`);
//! - [`ffi`] and [`ffi_rtp`]: the `extern "C"` functions, which only check
//!   pointers, build borrowed views of caller memory and call the session,
//!   receiver or RTP objects. They are the only modules allowed `unsafe`
//!   (besides nc-dsp, `CLAUDE.md`), and every block carries a `SAFETY:`
//!   comment.
//!
//! Every function reports failure through an `nc_status` code and never
//! unwinds across the ABI: each one that runs codec code catches a panic at
//! the boundary and returns `NC_ERROR_INTERNAL`. That needs unwinding, so the
//! C library is built with the workspace's `capi` profile
//! (`cargo build --profile capi -p nanocodec-capi`); under `--release`
//! (`panic = "abort"`) a panic aborts the process instead.

pub mod ffi;
pub mod ffi_rtp;
pub mod receiver;
pub mod rtp;
pub mod session;
