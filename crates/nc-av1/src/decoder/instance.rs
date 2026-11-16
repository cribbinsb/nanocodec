//! The public AV1 decoder: the decoding process ([`Core`]) instantiated
//! for the current sequence's samples, `u8` for 8-bit sequences and `u16`
//! for 10-bit ones (`docs/handoff/high-bit-depth.md`).
//!
//! The instance is chosen per coded video sequence: before a temporal
//! unit whose sequence header needs the other sample type is decoded, the
//! decoder continues in a new instance of that type (a new sequence
//! starts at a key frame, so no reference carries over), keeping the
//! caller's settings. Scanning a unit for its sequence header reads only
//! the OBU headers before its first frame.
//!
//! The picture-returning and browser API ([`Decoder::decode`],
//! [`Decoder::decode_shown`] and friends) is 8-bit and refuses deeper
//! sequences; [`Decoder::decode_any_depth`] with [`Decoder::shown_frame`]
//! decodes and shows a frame of either depth.

use nc_common::decode::DecodeError;
#[cfg(feature = "analysis")]
use nc_common::stream_stats::StreamStats;
use nc_common::{BitDepth, BitDepths, Picture, Sample, ShownFrame, ShownPicture};
use nc_dsp::Pixel;

use super::{Core, obu, sequence};

/// Whether sequences of `depth` decode in the `T` instance.
pub(super) fn fits<T: Pixel>(depth: BitDepth) -> bool {
    (T::MAX_BIT_DEPTH == u8::MAX_BIT_DEPTH) == (depth == BitDepth::EIGHT)
}

/// What the 8-bit-only calls return in a deeper sequence.
const EIGHT_BIT_API: DecodeError =
    DecodeError::Unsupported("AV1 picture of more than 8 bits through an 8-bit call");

/// The decoding process of the current sequence's sample type.
#[derive(Debug)]
enum Instance {
    Eight(Box<Core<u8>>),
    High(Box<Core<u16>>),
}

/// Run `$body` on the current instance's core, bound to `$core`.
macro_rules! with_core {
    ($self:expr, $core:ident => $body:expr) => {
        match &mut $self.instance {
            Instance::Eight($core) => $body,
            Instance::High($core) => $body,
        }
    };
}

/// Stateful AV1 decoder (Main profile: 8- and 10-bit, 4:2:0 and
/// monochrome): one temporal unit (low-overhead OBU format, as in IVF) in,
/// the frame it shows out.
#[derive(Debug)]
pub struct Decoder {
    instance: Instance,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// Start without sequence information or reference pictures.
    #[must_use]
    pub fn new() -> Self {
        Self {
            instance: Instance::Eight(Box::default()),
        }
    }

    /// Continue in the instance `temporal_unit`'s sequence header needs,
    /// if it has one of the other sample type.
    fn prepare(&mut self, temporal_unit: &[u8]) -> Result<(), DecodeError> {
        let Some(payload) = obu::sequence_header(temporal_unit)? else {
            return Ok(());
        };
        let depth = sequence::parse(payload)?.bit_depth;
        self.instance = match &mut self.instance {
            Instance::Eight(core) if !fits::<u8>(depth) => {
                Instance::High(Box::new(core.successor()))
            }
            Instance::High(core) if !fits::<u16>(depth) => {
                Instance::Eight(Box::new(core.successor()))
            }
            _ => return Ok(()),
        };
        Ok(())
    }

    /// Start collecting coding-tool analysis into `stats` (block records,
    /// and per-frame tallies unless [`StreamStats::records_only`]) from the
    /// next frame on. Decoding is unchanged.
    #[cfg(feature = "analysis")]
    pub fn collect_stats(&mut self, stats: StreamStats) {
        with_core!(self, core => core.collect_stats(stats));
    }

    /// The analysis collected so far, if enabled.
    #[cfg(feature = "analysis")]
    #[must_use]
    pub fn stats(&self) -> Option<&StreamStats> {
        match &self.instance {
            Instance::Eight(core) => core.stats(),
            Instance::High(core) => core.stats(),
        }
    }

    /// Complete the analysis and hand it over; collection stops.
    #[cfg(feature = "analysis")]
    pub fn take_stats(&mut self) -> Option<StreamStats> {
        with_core!(self, core => core.take_stats())
    }

    /// Show frames with (`true`, the default) or without the film grain
    /// their parameters signal; references are grain-free either way.
    pub fn set_film_grain(&mut self, apply: bool) {
        with_core!(self, core => core.set_film_grain(apply));
    }

    /// Select the operating point (AV1 §7.18, §6.4.1); see
    /// [`Core::set_operating_point`].
    ///
    /// # Errors
    /// [`DecodeError::Unsupported`] for an operating point out of range.
    pub fn set_operating_point(&mut self, operating_point: u8) -> Result<(), DecodeError> {
        with_core!(self, core => core.set_operating_point(operating_point))
    }

    /// Active operating point index (`0` by default).
    #[must_use]
    pub fn operating_point(&self) -> u8 {
        match &self.instance {
            Instance::Eight(core) => core.operating_point(),
            Instance::High(core) => core.operating_point(),
        }
    }

    /// Return an 8-bit picture [`Decoder::decode`] produced once the caller
    /// is done with it, so a later frame reuses its buffer.
    pub fn recycle(&mut self, picture: Picture) {
        if let Instance::Eight(core) = &mut self.instance {
            core.recycle(picture);
        }
    }

    /// Colour signalling of the active sequence header.
    #[must_use]
    pub fn colour(&self) -> Option<nc_common::ColourInfo> {
        match &self.instance {
            Instance::Eight(core) => core.colour(),
            Instance::High(core) => core.colour(),
        }
    }

    /// Maximum frame dimensions of the active sequence header.
    #[must_use]
    pub fn max_frame_size(&self) -> Option<(usize, usize)> {
        match &self.instance {
            Instance::Eight(core) => core.max_frame_size(),
            Instance::High(core) => core.max_frame_size(),
        }
    }

    /// `spatial_id` of the latest shown frame.
    #[must_use]
    pub fn shown_spatial_id(&self) -> Option<u8> {
        match &self.instance {
            Instance::Eight(core) => core.shown_spatial_id(),
            Instance::High(core) => core.shown_spatial_id(),
        }
    }

    /// Whether the active sequence is monochrome (one plane coded; the
    /// shown frames carry mid-grey chroma).
    #[must_use]
    pub fn is_monochrome(&self) -> bool {
        match &self.instance {
            Instance::Eight(core) => core.is_monochrome(),
            Instance::High(core) => core.is_monochrome(),
        }
    }

    /// Decode one temporal unit and return the 8-bit picture of the highest
    /// spatial layer shown and its `spatial_id`; see
    /// [`Core::decode_with_layer`].
    ///
    /// # Errors
    /// As [`Decoder::decode`]; [`DecodeError::Unsupported`] for a deeper
    /// sequence.
    pub fn decode_with_layer(
        &mut self,
        temporal_unit: &[u8],
    ) -> Result<Option<(Picture, u8)>, DecodeError> {
        self.prepare(temporal_unit)?;
        match &mut self.instance {
            Instance::Eight(core) => core.decode_with_layer(temporal_unit),
            Instance::High(_) => Err(EIGHT_BIT_API),
        }
    }

    /// Decode one temporal unit and return the 8-bit picture it shows, or
    /// `None` when the operating point dropped it all.
    ///
    /// # Errors
    /// As [`Decoder::decode_with_layer`].
    pub fn decode_optional(
        &mut self,
        temporal_unit: &[u8],
    ) -> Result<Option<Picture>, DecodeError> {
        self.decode_with_layer(temporal_unit)
            .map(|opt| opt.map(|(picture, _)| picture))
    }

    /// Decode one temporal unit and return every shown spatial layer as an
    /// 8-bit picture; see [`Core::decode_all_layers`].
    ///
    /// # Errors
    /// As [`Decoder::decode_with_layer`].
    pub fn decode_all_layers(
        &mut self,
        temporal_unit: &[u8],
    ) -> Result<Vec<(Picture, u8)>, DecodeError> {
        self.prepare(temporal_unit)?;
        match &mut self.instance {
            Instance::Eight(core) => core.decode_all_layers(temporal_unit),
            Instance::High(_) => Err(EIGHT_BIT_API),
        }
    }

    /// Decode one temporal unit that must show a frame and return it as an
    /// 8-bit picture.
    ///
    /// # Errors
    /// [`DecodeError`] for malformed, truncated or unsupported syntax, a
    /// temporal unit that shows no frame, missing references, dimensions
    /// beyond the allocation limit, or a sequence deeper than 8 bits.
    pub fn decode(&mut self, temporal_unit: &[u8]) -> Result<Picture, DecodeError> {
        self.decode_with_layer(temporal_unit)?
            .map(|(picture, _)| picture)
            .ok_or(DecodeError::Malformed("AV1 temporal unit shows no frame"))
    }

    /// [`Decoder::decode`] without the output copy: the 8-bit frame is
    /// shown by [`Decoder::shown`] until the next call (the browser's
    /// path).
    ///
    /// # Errors
    /// As [`Decoder::decode`].
    pub fn decode_shown(&mut self, temporal_unit: &[u8]) -> Result<(), DecodeError> {
        self.decode_and_show(temporal_unit).map(|_| ())
    }

    /// [`Decoder::decode_shown`], returning whether the unit showed a new
    /// frame (`false` when the selected operating point dropped all of it).
    ///
    /// # Errors
    /// As [`Decoder::decode`].
    pub fn decode_and_show(&mut self, temporal_unit: &[u8]) -> Result<bool, DecodeError> {
        self.prepare(temporal_unit)?;
        match &mut self.instance {
            Instance::Eight(core) => core.decode_and_show(temporal_unit),
            Instance::High(_) => Err(EIGHT_BIT_API),
        }
    }

    /// [`Decoder::decode_and_show`] at any depth: the frame is shown by
    /// [`Decoder::shown_frame`] until the next call.
    ///
    /// # Errors
    /// As [`Decoder::decode`], except that deeper sequences decode.
    pub fn decode_any_depth(&mut self, temporal_unit: &[u8]) -> Result<bool, DecodeError> {
        self.prepare(temporal_unit)?;
        with_core!(self, core => core.decode_and_show(temporal_unit))
    }

    /// Whether the active sequence stores its samples in u16.
    #[must_use]
    pub fn is_high_depth(&self) -> bool {
        matches!(self.instance, Instance::High(_))
    }

    /// Borrow deep output only; eight-bit callers keep the original shown path.
    #[must_use]
    pub fn shown_high(&self) -> Option<(ShownPicture<'_, u16>, nc_common::BitDepths)> {
        match &self.instance {
            Instance::High(core) => {
                let depth = core.bit_depth();
                Some((
                    core.shown()?,
                    nc_common::BitDepths {
                        luma: depth,
                        chroma: depth,
                    },
                ))
            }
            Instance::Eight(_) => None,
        }
    }

    /// The visible samples of the 8-bit frame the last successful decode
    /// showed (`None` in a deeper sequence: see [`Decoder::shown_frame`]).
    #[must_use]
    pub fn shown(&self) -> Option<ShownPicture<'_>> {
        match &self.instance {
            Instance::Eight(core) => core.shown(),
            Instance::High(_) => None,
        }
    }

    /// The visible samples of the frame the last successful decode showed,
    /// of either depth.
    #[must_use]
    pub fn shown_frame(&self) -> Option<ShownFrame<'_>> {
        match &self.instance {
            Instance::Eight(core) => core.shown().map(ShownFrame::Eight),
            Instance::High(core) => {
                let depth = core.bit_depth();
                core.shown().map(|picture| {
                    ShownFrame::High(
                        picture,
                        BitDepths {
                            luma: depth,
                            chroma: depth,
                        },
                    )
                })
            }
        }
    }

    /// Test hook: [`Core::corrupt_references`] of the 8-bit instance.
    #[cfg(test)]
    pub(super) fn corrupt_references(&mut self, value: u8) {
        with_core!(self, core => core.corrupt_references(value));
    }

    /// Test hook: [`Core::peek_header`].
    #[cfg(test)]
    pub(super) fn peek_header(&self, temporal_unit: &[u8]) -> super::subset::FrameParameters {
        match &self.instance {
            Instance::Eight(core) => core.peek_header(temporal_unit),
            Instance::High(core) => core.peek_header(temporal_unit),
        }
    }
}
