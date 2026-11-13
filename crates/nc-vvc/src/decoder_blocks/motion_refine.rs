//! MMVD displacement and temporal distance scaling, H.266 §8.5.2.
//! Checked against BSD-Clear `VVdeC` `MergeCtx` / `UnitTools`.
use super::{SliceHeader, motion::Motion};
use crate::{decoder_headers::ReferenceEntry, motion::Mv16};
use nc_common::decode::{DecodeError, Result};
pub(super) fn mmvd(
    mut motion: Motion,
    step: u8,
    direction: u8,
    header: &SliceHeader,
) -> Result<Motion> {
    let magnitude = 4i32
        << (step
            + if header.picture.motion.fullpel_mmvd {
                2
            } else {
                0
            });
    let offset = match direction {
        0 => Mv16 { x: magnitude, y: 0 },
        1 => Mv16 {
            x: -magnitude,
            y: 0,
        },
        2 => Mv16 { x: 0, y: magnitude },
        _ => Mv16 {
            x: 0,
            y: -magnitude,
        },
    };
    let mut offsets = [offset; 2];
    if let [Some(a), Some(b)] = motion.lists {
        let delta = |list: usize, index: u8| match header.references[list]
            .entries
            .get(usize::from(index))
        {
            Some(ReferenceEntry::Short(delta)) => Ok(*delta),
            _ => Err(DecodeError::Unsupported("VVC MMVD long-term reference")),
        };
        let distances = [delta(0, a.reference)?, delta(1, b.reference)?];
        if distances[0] != distances[1] {
            let near = usize::from(distances[0].abs() >= distances[1].abs());
            offsets[near] = scale(offset, distances[near], distances[1 - near])?;
        }
    }
    for (vector, offset) in motion.lists.iter_mut().zip(offsets) {
        if let Some(vector) = vector {
            vector.vector = Mv16 {
                x: vector.vector.x + offset.x,
                y: vector.vector.y + offset.y,
            }
            .clip();
        }
    }
    Ok(motion)
}
pub(super) use crate::temporal::scale;
