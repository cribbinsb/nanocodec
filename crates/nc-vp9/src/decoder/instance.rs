//! The public VP9 decoder: the decoding process ([`Core`]) of the stream's
//! sample type, `u8` for profile 0 (8-bit) and `u16` for profile 2 (10- and
//! 12-bit), as `docs/handoff/high-bit-depth.md` describes.
//!
//! A key frame (or an intra-only frame) that signals a depth of the other
//! sample type ends the current core and continues in a new one of that
//! type ([`Core::succeed`]); the frame resets everything a reference could
//! carry, and libvpx rejects references of another depth anyway.
//!
//! The picture-returning and browser calls (`decode`, `decode_packet`,
//! `decode_shown`, ...) are 8-bit: a deeper stream returns
//! [`DecodeError::Unsupported`] there (so the browser player refuses it
//! rather than showing garbage). [`Decoder::decode_kept`] and
//! [`Decoder::shown_vp9_frame`] decode and show either depth.

use nc_common::decode::{DecodeError, Result};
use nc_common::{BitDepth, Picture, ShownPicture};

#[cfg(feature = "analysis")]
use nc_common::stream_stats::StreamStats;

use super::header::signalled_depth;
use super::{Core, DecodedFrame};
use crate::superframe::split_superframe;

/// The decoder of one sample type.
#[derive(Debug)]
enum Instance {
    Eight(Box<Core<u8>>),
    High(Box<Core<u16>>),
}

/// Run `$body` on the current core, whichever its sample type.
macro_rules! with_core {
    ($instance:expr, $core:ident => $body:expr) => {
        match $instance {
            Instance::Eight($core) => $body,
            Instance::High($core) => $body,
        }
    };
}

/// What 8-bit calls return in a 10- or 12-bit stream.
const DEEP_IN_EIGHT_BIT_CALL: &str =
    "VP9 profile 2 (10/12-bit): use decode_kept and shown_vp9_frame";

/// The frame a decoder shows, at either depth: its picture (its planes
/// cover the visible size rounded up to even) and visible size.
#[derive(Debug, Clone, Copy)]
pub enum ShownVp9Frame<'a> {
    /// An 8-bit frame.
    Eight(&'a Picture, (usize, usize)),
    /// A 10- or 12-bit frame in `u16` samples, with its depth.
    High(&'a Picture<u16>, (usize, usize), BitDepth),
}

/// Stateful VP9 decoder of profiles 0 and 2.
#[derive(Debug)]
pub struct Decoder {
    instance: Instance,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            instance: Instance::Eight(Box::default()),
        }
    }
}

impl Decoder {
    /// A decoder awaiting a key frame.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Continue in a core of the sample type that `packet`'s first frame
    /// needs, if it signals a depth of the other type.
    fn prepare(&mut self, packet: &[u8]) {
        let first = split_superframe(packet).map_or(packet, |frames| frames[0]);
        let Some(depth) = signalled_depth(first) else {
            return;
        };
        let held = match &self.instance {
            Instance::Eight(_) => Core::<u8>::holds(depth),
            Instance::High(_) => Core::<u16>::holds(depth),
        };
        if held {
            return;
        }
        let previous = std::mem::replace(&mut self.instance, Instance::Eight(Box::default()));
        self.instance = match previous {
            Instance::Eight(core) => Instance::High(Box::new(Core::succeed(*core))),
            Instance::High(core) => Instance::Eight(Box::new(Core::succeed(*core))),
        };
    }

    /// The 8-bit core for `packet` (after [`Self::prepare`]), or
    /// [`DecodeError::Unsupported`] in a deeper stream.
    fn eight(&mut self, packet: &[u8]) -> Result<&mut Core<u8>> {
        self.prepare(packet);
        match &mut self.instance {
            Instance::Eight(core) => Ok(core),
            Instance::High(_) => Err(DecodeError::Unsupported(DEEP_IN_EIGHT_BIT_CALL)),
        }
    }

    /// Enable or disable base-only mode (`S0`). When enabled, enhancement
    /// frames in Annex B superframes are stripped via
    /// [`crate::Vp9StreamExtractor`] and only the lowest spatial layer (`S0`)
    /// is decoded and displayed.
    pub fn set_base_only(&mut self, base_only: bool) {
        with_core!(&mut self.instance, core => core.set_base_only(base_only));
    }

    /// Whether base-only mode is active.
    #[must_use]
    pub fn base_only(&self) -> bool {
        with_core!(&self.instance, core => core.base_only())
    }

    /// Select the spatial operating point (`0..=2`, default `0` = highest
    /// spatial layer in the stream, `1` = middle of 3 or base of 2, `2` =
    /// lowest of 3).
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::Unsupported`] if `operating_point > 2`.
    pub fn set_operating_point(&mut self, operating_point: u8) -> Result<()> {
        with_core!(&mut self.instance, core => core.set_operating_point(operating_point))
    }

    /// Active spatial operating point (`0` by default).
    #[must_use]
    pub fn operating_point(&self) -> u8 {
        with_core!(&self.instance, core => core.operating_point())
    }

    /// `spatial_id` (`0` on single-layer or base-layer TUs, `1` on enhancement
    /// TUs) of the latest shown frame.
    #[must_use]
    pub fn shown_spatial_id(&self) -> Option<u8> {
        with_core!(&self.instance, core => core.shown_spatial_id())
    }

    /// Maximum visible `(width, height)` seen in the stream so far (including
    /// enhancement frames inspected before stripping when [`Self::base_only`]
    /// is active).
    #[must_use]
    pub fn max_frame_size(&self) -> Option<(usize, usize)> {
        with_core!(&self.instance, core => core.max_frame_size())
    }

    /// The stream's bit depth (8 until a profile 2 key frame).
    #[must_use]
    pub fn bit_depth(&self) -> BitDepth {
        with_core!(&self.instance, core => core.depth())
    }

    /// Decode one 8-bit packet and return the picture it shows: the last
    /// frame of the packet when that frame is shown (as libvpx outputs).
    ///
    /// # Errors
    ///
    /// Truncated or malformed data, a profile other than 0 and 2, a 10- or
    /// 12-bit frame, or a frame whose references are missing.
    pub fn decode_packet(&mut self, packet: &[u8]) -> Result<Option<DecodedFrame>> {
        self.eight(packet)?.decode_packet(packet)
    }

    /// Decode one 8-bit packet and return `(picture, spatial_id)` for the
    /// shown frame (even dimensions), or `None` when the selected operating
    /// point dropped all frames in the packet.
    ///
    /// # Errors
    ///
    /// As [`Self::decode`].
    pub fn decode_with_layer(&mut self, packet: &[u8]) -> Result<Option<(Picture, u8)>> {
        self.eight(packet)?.decode_with_layer(packet)
    }

    /// Decode one 8-bit packet and return its shown picture (even
    /// dimensions), or `None` when the selected operating point dropped all
    /// frames in the packet.
    ///
    /// # Errors
    ///
    /// As [`Self::decode_with_layer`].
    pub fn decode_optional(&mut self, packet: &[u8]) -> Result<Option<Picture>> {
        self.eight(packet)?.decode_optional(packet)
    }

    /// Decode all spatial layers present in an 8-bit `packet` in superframe
    /// order (`S0` .. `S_top`, or the extracted subset when an operating
    /// point is selected) and return each layer's [`DecodedFrame`].
    ///
    /// # Errors
    ///
    /// As [`Self::decode_packet`].
    pub fn decode_all_layers(&mut self, packet: &[u8]) -> Result<Vec<DecodedFrame>> {
        self.eight(packet)?.decode_all_layers(packet)
    }

    /// Decode one 8-bit packet that shows a frame of even size.
    ///
    /// # Errors
    ///
    /// As [`Self::decode_packet`], and a packet that shows nothing or a
    /// frame of odd size.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Picture> {
        self.eight(packet)?.decode(packet)
    }

    /// [`Self::decode`] without the output copy: the frame is shown by
    /// [`Self::shown`] until the next call. When the selected operating point
    /// drops all frames in `packet`, the previous shown frame is retained.
    ///
    /// # Errors
    ///
    /// As [`Self::decode`].
    pub fn decode_shown(&mut self, packet: &[u8]) -> Result<()> {
        self.eight(packet)?.decode_shown(packet)
    }

    /// [`Self::decode_shown`], returning whether `packet` showed a new frame
    /// (`false` when the selected operating point dropped all frames in it).
    ///
    /// # Errors
    ///
    /// As [`Self::decode`].
    pub fn decode_and_show(&mut self, packet: &[u8]) -> Result<bool> {
        self.eight(packet)?.decode_and_show(packet)
    }

    /// Decode one packet of either depth and keep the frame it shows, if
    /// any, until the next call, without copying it
    /// ([`Self::shown_vp9_frame`]); returns whether the packet shows a frame.
    ///
    /// # Errors
    ///
    /// As [`Self::decode_packet`], for streams of any depth.
    pub fn decode_kept(&mut self, packet: &[u8]) -> Result<bool> {
        self.prepare(packet);
        with_core!(&mut self.instance, core => core.decode_kept(packet))
    }

    /// The decoder's own picture of the 8-bit frame the last
    /// [`Self::decode_kept`] showed and its visible size (any parity; the
    /// picture is larger); `None` for a deeper frame.
    #[must_use]
    pub fn shown_frame(&self) -> Option<(&Picture, (usize, usize))> {
        match &self.instance {
            Instance::Eight(core) => core.shown_frame(),
            Instance::High(_) => None,
        }
    }

    /// The frame of either depth the last [`Self::decode_kept`] showed.
    #[must_use]
    pub fn shown_vp9_frame(&self) -> Option<ShownVp9Frame<'_>> {
        match &self.instance {
            Instance::Eight(core) => core
                .shown_frame()
                .map(|(picture, size)| ShownVp9Frame::Eight(picture, size)),
            Instance::High(core) => core
                .shown_frame()
                .map(|(picture, size)| ShownVp9Frame::High(picture, size, core.depth())),
        }
    }

    /// The visible samples of the 8-bit frame the last successful
    /// [`Self::decode_shown`] showed.
    #[must_use]
    pub fn shown(&self) -> Option<ShownPicture<'_>> {
        match &self.instance {
            Instance::Eight(core) => core.shown(),
            Instance::High(_) => None,
        }
    }

    /// Return a picture [`Self::decode`] or [`Self::decode_packet`] produced
    /// once the caller is done with it, so a later frame reuses its buffer
    /// instead of allocating.
    pub fn recycle(&mut self, picture: Picture) {
        if let Instance::Eight(core) = &mut self.instance {
            core.recycle(picture);
        }
    }

    /// Start collecting coding-tool analysis into `stats` (block records,
    /// and per-frame tallies unless [`StreamStats::records_only`]) from the
    /// next frame on. Decoding is unchanged.
    #[cfg(feature = "analysis")]
    pub fn collect_stats(&mut self, stats: StreamStats) {
        with_core!(&mut self.instance, core => core.collect_stats(stats));
    }

    /// The analysis collected so far, if enabled: the block records of the
    /// last decoded frame and the tallies of every completed one.
    #[cfg(feature = "analysis")]
    #[must_use]
    pub fn stats(&self) -> Option<&StreamStats> {
        with_core!(&self.instance, core => core.stats())
    }

    /// Complete the analysis and hand it over; collection stops.
    #[cfg(feature = "analysis")]
    pub fn take_stats(&mut self) -> Option<StreamStats> {
        with_core!(&mut self.instance, core => core.take_stats())
    }
}
