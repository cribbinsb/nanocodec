//! Picture structure (§7.4.3 `field_pic_flag`, `bottom_field_flag`): a
//! picture is a frame or one field of a frame. A field is every other row
//! of its frame's buffer (top field: rows 0, 2, ...; bottom field: rows 1,
//! 3, ...), so a field picture is decoded, predicted and filtered as a
//! picture of half the frame's height through views of those rows.
//!
//! Reference identities (`RefPicture::id`, `MbState::ref_pic`) name a frame
//! or one of its fields: the frame's identity times four plus the
//! structure, so two fields of one frame are different pictures and no
//! identity is 0 ("no picture").

use nc_common::PlaneViewMut;

use super::Planes;
use super::sample::H264Sample;

/// One of the two fields of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Parity {
    Top,
    Bottom,
}

impl Parity {
    /// 0 for the top field, 1 for the bottom one: its first row in the
    /// frame, and its index in per-field arrays.
    pub fn index(self) -> usize {
        match self {
            Self::Top => 0,
            Self::Bottom => 1,
        }
    }

    pub fn opposite(self) -> Self {
        match self {
            Self::Top => Self::Bottom,
            Self::Bottom => Self::Top,
        }
    }

    /// Both parities, top first.
    pub const BOTH: [Self; 2] = [Self::Top, Self::Bottom];
}

/// A frame or a field picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Structure {
    Frame,
    Field(Parity),
}

/// Largest frame identity: identities times four must fit a `u32`.
pub(super) const MAX_FRAME_ID: u32 = u32::MAX >> 2;

impl Structure {
    pub fn is_field(self) -> bool {
        matches!(self, Self::Field(_))
    }

    /// The field's parity (`None` for a frame).
    pub fn parity(self) -> Option<Parity> {
        match self {
            Self::Frame => None,
            Self::Field(parity) => Some(parity),
        }
    }

    /// The reference identity of this picture of the frame `frame_id`
    /// (at most [`MAX_FRAME_ID`]).
    pub fn reference_id(self, frame_id: u32) -> u32 {
        let low = match self {
            Self::Field(Parity::Top) => 1,
            Self::Field(Parity::Bottom) => 2,
            Self::Frame => 3,
        };
        (frame_id << 2) | low
    }
}

/// The frame of a reference identity.
pub(super) fn frame_of(reference_id: u32) -> u32 {
    reference_id >> 2
}

/// Writable views of the rows of `planes` that make up picture
/// `structure`: the planes themselves for a frame, every other row for a
/// field.
pub(super) fn picture_planes<T: H264Sample>(
    planes: Planes<'_, T>,
    structure: Structure,
) -> Planes<'_, T> {
    match structure {
        Structure::Frame => planes,
        Structure::Field(parity) => {
            planes.map(|plane: PlaneViewMut<'_, T>| plane.every_other_row(parity.index()))
        }
    }
}
