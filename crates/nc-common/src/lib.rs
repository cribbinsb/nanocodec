//! Common building blocks shared by every nanocodec crate.
//!
//! This is the bottom layer of the workspace (see `docs/ARCHITECTURE.md`): it
//! depends on nothing else in nanocodec, and everything else may depend on it.
//!
//! - [`Sample`] / [`BitDepth`]: the sample types (`u8` for 8-bit, `u16` for
//!   deeper streams) every buffer and kernel is generic over, and the
//!   runtime depth their clips use (`docs/handoff/high-bit-depth.md`).
//! - [`Plane`] / [`Picture`]: owned sample buffers with aligned rows and
//!   padded borders, the representation of every source and reference frame
//!   (`Plane<u8>` by default; `Plane<u16>` for decoded high bit depth).
//! - [`PlaneView`] / [`PlaneViewMut`]: borrowed, bounds-checked windows into a
//!   plane. Kernels operate on views, never on raw pointers.
//! - [`ShownPicture`]: a decoder's visible output, borrowed until its next
//!   decode.
//! - [`OutputPicture`] / [`DecodedPicture`]: a decoder's output in output
//!   order, shared with its reference buffers, of either sample type.
//! - [`RawFrame`]: borrowed camera frames (I420, YV12, NV12) with any
//!   stride, the encoder's input for frames it does not own.
//! - [`yuv`]: raw I420 file reading and writing (16-bit little-endian
//!   output for deep pictures).
//! - [`BitWriter`] / [`leb128`]: MSB-first header bit writing for all codecs.
//! - [`nal`]: Annex-B NAL unit framing with emulation prevention (H.264/HEVC).
//! - [`cabac`]: the CABAC arithmetic decoding engine (H.264/HEVC, and H.266
//!   with its own probability model).
//! - [`ColourInfo`]: colour signalling (CICP code points, range).
//! - `stream_stats` (feature `analysis`): coding-tool records decoders
//!   can fill.
//! - [`Error`]: the error type for everything above.

#![forbid(unsafe_code)]

mod aligned;
mod bitwriter;
pub mod cabac;
pub mod colour;
pub mod decode;
mod error;
pub mod nal;
pub mod outcome;
mod output;
mod picture;
mod plane;
mod raw_frame;
mod recycle;
mod sample;
mod shown;
#[cfg(feature = "analysis")]
pub mod stream_stats;
mod view;
pub mod yuv;

pub use bitwriter::{BitWriter, leb128};
pub use colour::{ColourDescription, ColourInfo};
pub use error::{Error, Result};
pub use output::{DecodedPicture, OutputPicture};
pub use picture::{ChromaFormat, Picture, PlaneIndex};
pub use plane::{BorderStores, CachedStores, Plane, ROW_ALIGNMENT};
pub use raw_frame::{NV12_CHROMA_BYTES, RawFrame, chroma_len};
pub use recycle::PictureRecycler;
pub use sample::{BitDepth, BitDepths, Sample};
pub use shown::ShownPicture;
pub use view::{PlaneView, PlaneViewMut, copy_row};
