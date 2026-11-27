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
//! - [`cabac_encoder`]: the CABAC arithmetic encoding engine of the HEVC
//!   and H.266 encoders.
//! - [`cell_grid`]: block fills of the codecs' mode-information grids.
//! - [`fnv1a`]: the FNV-1a fingerprint tests compare samples by.
//! - [`propagate_carry`]: the carry step of the 64-bit-window range
//!   encoders (AV1, AV2, VP9).
//! - [`ColourInfo`]: colour signalling (CICP code points, range).
//! - [`SplitMix64`]: the seeded random source of test data and syntax
//!   stress.
//! - `stream_stats` (feature `analysis`): coding-tool records decoders
//!   can fill.
//! - [`Error`]: the error type for everything above.

#![forbid(unsafe_code)]

mod aligned;
mod bitwriter;
pub mod cabac;
pub mod cabac_encoder;
pub mod cell_grid;
pub mod colour;
pub mod decode;
pub mod eob;
mod error;
mod fnv;
pub mod nal;
pub mod outcome;
mod output;
mod picture;
mod plane;
mod range_carry;
mod raw_frame;
mod recycle;
mod rng;
mod sample;
mod shown;
#[cfg(feature = "analysis")]
pub mod stream_stats;
mod view;
pub mod yuv;

pub use bitwriter::{BitWriter, leb128};
pub use colour::{ColourDescription, ColourInfo};
pub use error::{Error, Result};
pub use fnv::{FNV1A_OFFSET, fnv1a, fnv1a_step};
pub use output::{DecodedPicture, OutputPicture, ShownFrame};
pub use picture::{ChromaFormat, ExtendedPicture, Picture, PlaneIndex};
pub use plane::{Plane, ROW_ALIGNMENT};
pub use range_carry::propagate_carry;
pub use raw_frame::{NV12_CHROMA_BYTES, RawFrame, chroma_len};
pub use recycle::PictureRecycler;
pub use rng::SplitMix64;
pub use sample::{BitDepth, BitDepths, Sample};
pub use shown::ShownPicture;
pub use view::{PlaneView, PlaneViewMut, copy_row};
