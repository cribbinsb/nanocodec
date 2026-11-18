//! End-of-block groups shared by AV1 and AV2 coefficient syntax.
//!
//! AV1's `av1_eob_group_start` and AV2's `av2_eob_group_start` use the
//! same boundaries: after 0 and 1, each group starts one past a power of
//! two. A bit length replaces a reverse scan of the twelve boundaries.
//! The boundary tables come from libaom (BSD-2-Clause) and AVM
//! (BSD-3-Clause-Clear); see `THIRD_PARTY.md`.

/// First position in each group (AV1/AV2 `eob_group_start`).
pub const GROUP_START: [u16; 12] = [0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513];

/// Index of the last boundary at or below `position`.
///
/// Zero maps to group zero, as required by AV2's read-side symbol source.
/// Positions beyond the coded 32x32 coefficient area retain the final
/// group's table-scan behaviour; syntax callers validate the block extent.
#[must_use]
#[inline]
pub const fn group(position: usize) -> usize {
    if position < 2 {
        position
    } else {
        let bits = (usize::BITS - (position - 1).leading_zeros()) as usize;
        let group = bits + 1;
        if group < GROUP_START.len() {
            group
        } else {
            GROUP_START.len() - 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_length_matches_every_boundary_and_the_capped_tail() {
        for position in (0..=4096).chain([usize::MAX]) {
            assert_eq!(
                group(position),
                GROUP_START
                    .iter()
                    .rposition(|&start| usize::from(start) <= position)
                    .unwrap(),
                "position {position}",
            );
        }
    }
}
