//! The shapes every payload format converts to and from, so the packetizer
//! front end and the receiver's assembler are written once.
//!
//! - Sending: a format cuts a unit into [`OutPayload`]s, each tagged with
//!   the layer frame it belongs to (for the Dependency Descriptor and the
//!   VP9 descriptor) and whether it starts or ends that frame.
//! - Receiving: a format turns one RTP payload into [`Piece`]s of
//!   *objects* (NAL units, OBUs, VP9 frames), each possibly a fragment
//!   continued from the previous packet or into the next, plus the
//!   [`PacketInfo`] its headers carry. The assembler concatenates pieces into
//!   objects and the format joins objects into a decoder-ready unit.

/// One RTP payload a format produced for a unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OutPayload {
    /// The payload bytes (payload header included).
    pub bytes: Vec<u8>,
    /// Index of the layer frame in the unit (0 for single-layer units).
    pub frame: usize,
    /// The payload's first byte begins that frame.
    pub start_of_frame: bool,
    /// The payload's last byte ends that frame.
    pub end_of_frame: bool,
}

/// A piece of one object carried in a payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Piece {
    /// The bytes (for a first fragment, starting with the object's header).
    pub data: Vec<u8>,
    /// The piece continues an object begun in an earlier packet.
    pub continues_previous: bool,
    /// The object continues in a later packet.
    pub continues_next: bool,
}

impl Piece {
    /// A complete object.
    pub fn whole(data: Vec<u8>) -> Self {
        Self {
            data,
            continues_previous: false,
            continues_next: false,
        }
    }
}

/// A layer frame's identity and dependencies as one packet describes them
/// (Dependency Descriptor, VP9 flexible-mode descriptor).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FrameTag {
    /// Unwrapped frame identity, unique in the stream.
    pub id: u64,
    /// The packet's first byte begins the frame.
    pub start: bool,
    /// The packet's last byte ends the frame.
    pub end: bool,
    /// Identities of the frames it predicts from (`None`: not described).
    pub references: Option<Vec<u64>>,
    /// Spatial layer.
    pub spatial_id: u8,
    /// Temporal layer.
    pub temporal_id: u8,
}

/// What a payload's headers say beyond its pieces.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PacketInfo {
    /// The packet belongs to a key unit (decoding can start here).
    pub key: bool,
    /// Whether later frames may reference the packet's frame, when its
    /// headers say (H.264 `nal_ref_idc`; `None` when the packet starts no
    /// VCL NAL unit or the format does not say).
    pub reference: Option<bool>,
    /// The layer frame the packet belongs to, when described.
    pub frame: Option<FrameTag>,
    /// The packet's Dependency Descriptor, read in sequence order at
    /// delivery (its template structure may be in an earlier packet that
    /// arrives later).
    pub dependency_descriptor: Option<Vec<u8>>,
}

/// One received payload, parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedPayload {
    /// Pieces in order.
    pub pieces: Vec<Piece>,
    /// Header information.
    pub info: PacketInfo,
}
