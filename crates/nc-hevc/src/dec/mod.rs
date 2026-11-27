//! HEVC Main / Main 10 / Main Still Picture profile decoder (H.265 Annex
//! A.3.2, A.3.3, A.3.5): 4:2:0 streams of 8 to 10 bits (luma and chroma
//! may differ) of any structure the profiles allow — I, P
//! and B slices, multiple and dependent slice segments, tiles, WPP, AMP,
//! PCM, lossless, transform skip, scaling lists, weighted prediction, long-
//! term references, temporal sub-layers, CRA/BLA/RASL handling and output
//! in output order through the bumping process of Annex C.5.2.
//!
//! Module map:
//! - `params`, `slice`: parameter sets and slice segment headers;
//! - `cabac`, `ctx`: the arithmetic decoder and context variables;
//! - `ctu`, `residual`: slice data syntax and block reconstruction;
//! - `mvpred`, `mc`: motion vector prediction and inter prediction;
//! - `deblock`, `sao`: the in-loop filters;
//! - `picture`: per-picture block state, CTB layout and stored motion;
//! - `stats`: coding-tool analysis facts (feature `analysis`;
//!   `Decoder::collect_stats`, block records in `ctu`);
//! - `core`: NAL units, POC (§8.3.1), reference picture sets (§8.3.2,
//!   §8.3.3), reference picture lists (§8.3.4) and the DPB (C.5.2);
//! - this module: [`Decoder`], which runs `core` at the stream's sample
//!   type.
//!
//! High bit depth (`docs/handoff/high-bit-depth.md`): everything below this
//! module is written once over the sample type `T: nc_dsp::Pixel` and
//! compiled twice, `u8` for 8-bit streams (exactly the 8-bit decoder) and
//! `u16` for deeper ones, with each plane's depth a runtime
//! [`nc_common::BitDepth`]. [`Decoder`] picks the instance from the SPS
//! that each IRAP picture activates.

mod cabac;
mod core;
mod ctu;
pub(crate) mod ctx;
mod deblock;
mod loop_filter;
mod mc;
mod mvpred;
mod params;
mod picture;
mod residual;
mod sao;
mod slice;
#[cfg(feature = "analysis")]
mod stats;
#[cfg(test)]
mod tests;

use std::collections::VecDeque;

use nc_common::decode::{DecodeError, annex_b_units};
#[cfg(feature = "analysis")]
use nc_common::stream_stats::StreamStats;
use nc_common::{BitDepths, ColourInfo, DecodedPicture, Picture, ShownPicture};

use self::core::Core;

/// The decoding process at the current stream's sample type.
#[derive(Debug)]
enum Instance {
    /// 8-bit luma and chroma.
    Eight(Box<Core<u8>>),
    /// Luma or chroma deeper than 8 bits, in `u16` samples.
    High(Box<Core<u16>>),
}

/// Run `$body` on the current instance's core, bound to `$core`.
macro_rules! with_core {
    ($instance:expr, $core:ident => $body:expr) => {
        match $instance {
            Instance::Eight($core) => $body,
            Instance::High($core) => $body,
        }
    };
}

/// HEVC Main / Main 10 profile decoder. Feed NAL units (or Annex-B data)
/// in decoding order; decoded pictures come out in output order, 8-bit
/// streams as `u8` samples and deeper ones as `u16` ([`DecodedPicture`]).
#[derive(Debug)]
pub struct Decoder {
    instance: Instance,
    /// Pictures output before the sample type changed, ahead of the
    /// current instance's.
    earlier: VecDeque<DecodedPicture>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// The error of 8-bit-only calls on a deeper stream.
const EIGHT_BIT_ONLY: DecodeError = DecodeError::Unsupported(
    "decoding-order output of more than 8 bits (use push and next_output)",
);

impl Decoder {
    /// An empty decoder; the stream must start with parameter sets and an
    /// IRAP picture (pictures before the first IRAP are skipped).
    #[must_use]
    pub fn new() -> Self {
        Self {
            instance: Instance::Eight(Box::new(Core::new())),
            earlier: VecDeque::new(),
        }
    }

    /// Start collecting coding-tool analysis into `stats` (block records,
    /// and per-picture tallies unless [`StreamStats::records_only`]) from
    /// the next picture on. Decoding is unchanged.
    #[cfg(feature = "analysis")]
    pub fn collect_stats(&mut self, stats: StreamStats) {
        with_core!(&mut self.instance, core => core.collect_stats(stats));
    }

    /// The analysis collected so far, if enabled: the block records of
    /// the last completed picture and the tallies of every completed one.
    #[cfg(feature = "analysis")]
    #[must_use]
    pub fn stats(&self) -> Option<&StreamStats> {
        with_core!(&self.instance, core => core.stats())
    }

    /// Complete the analysis of the last picture and hand it over (after
    /// [`Decoder::flush`]); collection stops.
    #[cfg(feature = "analysis")]
    pub fn take_stats(&mut self) -> Option<StreamStats> {
        with_core!(&mut self.instance, core => core.take_stats())
    }

    /// Colour signalling of the last activated SPS (`None` before the first).
    #[must_use]
    pub fn colour(&self) -> Option<ColourInfo> {
        with_core!(&self.instance, core => core.colour())
    }

    /// Bit depths of the last decoded picture (`None` before the first).
    #[must_use]
    pub fn depths(&self) -> Option<BitDepths> {
        with_core!(&self.instance, core => core.depths())
    }

    /// Test hook: overwrite every DPB picture with `value`, as a decoder
    /// that lost a frame would hold wrong references.
    #[cfg(test)]
    pub(crate) fn corrupt_references(&mut self, value: u8) {
        with_core!(&mut self.instance, core => core.corrupt_references(value));
    }

    /// Decode one access unit of Annex-B data and return the first picture
    /// it outputs (C.5.2 bumping; the low-delay use: in a stream without
    /// reordering each packet outputs its own picture). Further pictures it
    /// outputs wait for [`Decoder::next_shown`] or [`Decoder::next_output`].
    /// # Errors
    /// Rejects malformed or truncated data, syntax outside the Main and
    /// Main 10 profiles, streams of more than 8 bits (8-bit output) and an
    /// access unit that outputs no picture.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Picture, DecodeError> {
        self.decode_shown(packet)?;
        match &mut self.instance {
            Instance::Eight(core) => core.shown_copy(),
            Instance::High(_) => Err(EIGHT_BIT_ONLY),
        }
    }

    /// [`Decoder::decode`] without the output copy: the picture is shown
    /// by [`Decoder::shown`] until the next call; an access unit that
    /// outputs none (a skipped RASL picture, `PicOutputFlag` 0, a reserved NAL
    /// type, a picture held for reordering) shows nothing and is no error.
    /// # Errors
    /// As [`Decoder::decode`], except for an access unit without output;
    /// [`DecodeError::MissingKey`] before the first IRAP picture.
    pub fn decode_shown(&mut self, packet: &[u8]) -> Result<(), DecodeError> {
        with_core!(&mut self.instance, core => core.begin_shown());
        self.push(packet)?;
        with_core!(&mut self.instance, core => core.end_shown())?;
        match self.instance {
            Instance::Eight(_) => Ok(()),
            Instance::High(_) => Err(EIGHT_BIT_ONLY),
        }
    }

    /// Show the next picture the last [`Self::decode_shown`] access unit
    /// output, when it output several (a stream that reorders, at an IRAP
    /// picture or a full DPB); `false` (nothing shown) when there is none.
    pub fn next_shown(&mut self) -> bool {
        with_core!(&mut self.instance, core => core.next_shown())
    }

    /// Decode one access unit at either sample depth without an output copy.
    /// The latest picture is borrowed through [`Self::shown_high`] (deep) or [`Self::shown`] (8-bit).
    /// # Errors
    /// As [`Self::decode_shown`], except that supported deep streams are accepted.
    pub fn decode_shown_any_depth(&mut self, packet: &[u8]) -> Result<(), DecodeError> {
        // The original call completes a valid deep picture before reporting
        // its eight-bit return-type restriction. Reuse that exact eight-bit
        // call graph, and lift only this specific API restriction.
        match self.decode_shown(packet) {
            Err(EIGHT_BIT_ONLY) => Ok(()),
            result => result,
        }
    }

    /// Whether the active sequence stores its samples in u16.
    #[must_use]
    pub fn is_high_depth(&self) -> bool {
        matches!(self.instance, Instance::High(_))
    }

    /// Borrow deep output only; the original eight-bit shown path stays separate.
    #[must_use]
    pub fn shown_high(&self) -> Option<(ShownPicture<'_, u16>, nc_common::BitDepths)> {
        match &self.instance {
            Instance::High(core) => Some((core.shown()?, core.depths()?)),
            Instance::Eight(_) => None,
        }
    }

    /// The visible samples of the picture the last successful
    /// [`Decoder::decode_shown`] decoded (8-bit streams).
    #[must_use]
    pub fn shown(&self) -> Option<ShownPicture<'_>> {
        match &self.instance {
            Instance::Eight(core) => core.shown(),
            Instance::High(_) => None,
        }
    }

    /// Feed Annex-B data holding whole NAL units.
    /// # Errors
    /// As [`Decoder::push_nal`].
    pub fn push(&mut self, data: &[u8]) -> Result<(), DecodeError> {
        for nal in annex_b_units(data)? {
            self.push_nal(nal)?;
        }
        Ok(())
    }

    /// Decode one NAL unit (without start code).
    /// # Errors
    /// Rejects malformed or truncated data and syntax outside the Main and
    /// Main 10 profiles; the decoder should then be reset before further
    /// use.
    pub fn push_nal(&mut self, nal: &[u8]) -> Result<(), DecodeError> {
        if with_core!(&self.instance, core => core.needs_other_sample_type(nal)) {
            self.change_sample_type()?;
        }
        with_core!(&mut self.instance, core => core.push_nal(nal))
    }

    /// An IRAP picture activates an SPS of the other sample type: output
    /// every picture of the current one, then continue in the other.
    fn change_sample_type(&mut self) -> Result<(), DecodeError> {
        with_core!(&mut self.instance, core => core.flush())?;
        while let Some(picture) = self.next_current() {
            self.earlier.push_back(picture);
        }
        let placeholder = Instance::Eight(Box::new(Core::new()));
        self.instance = match std::mem::replace(&mut self.instance, placeholder) {
            Instance::Eight(core) => Instance::High(Box::new(Core::succeed(*core))),
            Instance::High(core) => Instance::Eight(Box::new(Core::succeed(*core))),
        };
        Ok(())
    }

    /// Signal the end of the stream: the last picture is finished and every
    /// picture still waiting is queued for output.
    /// # Errors
    /// As [`Decoder::push_nal`] for the last picture.
    pub fn flush(&mut self) -> Result<(), DecodeError> {
        with_core!(&mut self.instance, core => core.flush())
    }

    /// Hand back a picture this decoder returned from [`Decoder::decode`],
    /// once the caller is done with it: later output pictures of its size
    /// reuse the buffer instead of allocating and clearing one.
    pub fn recycle(&mut self, picture: Picture) {
        if let Instance::Eight(core) = &mut self.instance {
            core.recycle(picture);
        }
    }

    /// The next picture in output order, if one is ready.
    pub fn next_output(&mut self) -> Option<DecodedPicture> {
        self.earlier.pop_front().or_else(|| self.next_current())
    }

    /// The current instance's next picture in output order.
    fn next_current(&mut self) -> Option<DecodedPicture> {
        with_core!(&mut self.instance, core => core.next_output().map(DecodedPicture::from))
    }
}
