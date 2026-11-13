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
//! - [`ffi`]: the `extern "C"` functions, which only check pointers, build
//!   borrowed views of caller memory and call the session. It is the one
//!   module allowed `unsafe` (besides nc-dsp, `CLAUDE.md`), and every block
//!   carries a `SAFETY:` comment.
//!
//! Every function reports failure through an `nc_status` code and never
//! unwinds across the ABI: panics are caught at the boundary.

pub mod ffi;
pub mod receiver;
pub mod session;
