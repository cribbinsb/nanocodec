//! A codec-independent, entirely safe decoder session for the C API: our
//! subset decoder plus the picture on screen. Every packet gets one
//! [`FrameOutcome`] (`nc_common::outcome`): decoded, or frozen on the last
//! good picture because a reference or the key frame is missing or the
//! data is corrupt, with the frame to name in a loss report.

use nc_common::Picture;
use nc_common::decode::DecodeError;
use nc_common::outcome::{FrameOutcome, ReceivedFrames};
use nc_encoder::Codec;

/// Our decoder for one codec.
#[derive(Debug)]
enum AnyDecoder {
    H264(Box<nc_h264::Decoder>),
    Hevc(Box<nc_hevc::Decoder>),
    Av1(Box<nc_av1::Decoder>),
    Vp9(Box<nc_vp9::Decoder>),
}

impl AnyDecoder {
    fn decode(&mut self, packet: &[u8]) -> Result<Option<Picture>, DecodeError> {
        match self {
            Self::H264(decoder) => decoder.decode(packet).map(Some),
            Self::Hevc(decoder) => decoder.decode(packet).map(Some),
            Self::Av1(decoder) => decoder.decode_optional(packet),
            Self::Vp9(decoder) => decoder.decode_optional(packet),
        }
    }

    fn set_operating_point(&mut self, op: u8) -> Result<(), DecodeError> {
        match self {
            Self::Av1(decoder) => decoder.set_operating_point(op),
            Self::Vp9(decoder) if op <= nc_vp9::MAX_VP9_OPERATING_POINT => {
                decoder.set_operating_point(op)
            }
            _ if op == 0 => Ok(()),
            _ => Err(DecodeError::Unsupported(
                "operating points above 0 require AV1 (0..=31) or VP9 (0..=2)",
            )),
        }
    }

    fn spatial_id(&self) -> u8 {
        match self {
            Self::Av1(decoder) => decoder.shown_spatial_id().unwrap_or(0),
            Self::Vp9(decoder) => decoder.shown_spatial_id().unwrap_or(0),
            Self::H264(_) | Self::Hevc(_) => 0,
        }
    }
}

/// One receiver: decoder, the frames it decoded and the picture on screen.
#[derive(Debug)]
pub struct Receiver {
    decoder: AnyDecoder,
    frames: ReceivedFrames,
    /// The last decoded picture: shown until the next one decodes.
    shown: Option<Picture>,
}

impl Receiver {
    /// A receiver for `codec` (H.264, HEVC, AV1 or VP9, the C API's codecs).
    #[must_use]
    pub fn new(codec: Codec) -> Option<Self> {
        let decoder = match codec {
            Codec::H264 => AnyDecoder::H264(Box::default()),
            Codec::Hevc => AnyDecoder::Hevc(Box::default()),
            Codec::Av1 => AnyDecoder::Av1(Box::default()),
            Codec::Vp9 => AnyDecoder::Vp9(Box::default()),
            _ => return None,
        };
        Some(Self {
            decoder,
            frames: ReceivedFrames::default(),
            shown: None,
        })
    }

    /// Decode coded frame `index` (`nc_packet.index`) from `packet`.
    /// `references` is the transport's dependency signal (the frames it
    /// predicts from) when the caller has one, else empty: H.264, HEVC and
    /// AV1 streams with frame ids (`av1.frame_ids`, or a long-term
    /// reference) then reveal a missing reference themselves.
    pub fn decode(&mut self, packet: &[u8], index: u64, references: &[u64]) -> FrameOutcome {
        let decoder = &mut self.decoder;
        let (outcome, picture) = self
            .frames
            .receive(index, references, || decoder.decode(packet));
        if let Some(Some(picture)) = picture {
            self.shown = Some(picture);
        }
        outcome
    }

    /// Select the scalable operating point (`0..=31` on AV1, `0..=2` on VP9).
    ///
    /// # Errors
    ///
    /// Rejects invalid operating points or codecs without scalable operating points.
    pub fn set_operating_point(&mut self, op: u8) -> Result<(), DecodeError> {
        self.decoder.set_operating_point(op)
    }

    /// Spatial layer id (`0..=2`) of the picture currently on screen (`0` on
    /// single-layer streams or before the first decoded picture).
    #[must_use]
    pub fn spatial_id(&self) -> u8 {
        self.decoder.spatial_id()
    }

    /// The picture on screen: the last decoded one (`None` before any).
    #[must_use]
    pub fn picture(&self) -> Option<&Picture> {
        self.shown.as_ref()
    }
}
