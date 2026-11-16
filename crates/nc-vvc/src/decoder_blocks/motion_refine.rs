//! MMVD displacement and temporal distance scaling, H.266 §8.5.2.
//! Checked against BSD-Clear `VVdeC` `MergeCtx` / `UnitTools`.
use super::{SliceHeader, motion::Motion};
use crate::motion::Mv16;
use nc_common::decode::Result;
/// The MMVD motion of merge candidate `motion` (H.266 §8.5.2.7): the
/// offset of `step` and `direction` for the list whose reference is
/// farther, and for the other list that offset scaled by the POC distances
/// (§8.5.2.15), or, when either reference is long-term, mirrored by the
/// distances' signs. `references` are the slice's reference POCs.
pub(super) fn mmvd(
    mut motion: Motion,
    (step, direction): (u8, u8),
    header: &SliceHeader,
    (references, poc): ([&[i32]; 2], i32),
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
        let distance = |list: usize, index: u8| references[list][usize::from(index)] - poc;
        let distances = [distance(0, a.reference), distance(1, b.reference)];
        let long_term = header.long_term(0, a.reference) || header.long_term(1, b.reference);
        if distances[0] != distances[1] {
            let near = usize::from(distances[0].abs() >= distances[1].abs());
            offsets[near] = if !long_term {
                scale(offset, distances[near], distances[1 - near])?
            } else if distances[0].signum() == distances[1].signum() {
                offset
            } else {
                Mv16 {
                    x: -offset.x,
                    y: -offset.y,
                }
            };
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
