//! VP9 reference motion vectors (NEARESTMV/NEARMV) and the inter-mode
//! context (VP9 §6.4.22 / §9.3; libvpx `vp9_mvref_common.c`
//! `find_mv_refs_idx` and `vp9_decodemv.c`, BSD-3-Clause, see `THIRD_PARTY.md`).
//!
//! Streams use one reference frame (`LAST_FRAME`) with no compound
//! prediction, so the sign-bias and second-reference branches never apply.
//! The previous frame's vector at the block's position joins the list after
//! the spatial neighbours when the frame allows it (`use_prev_frame_mvs`).
//! Candidates keep 1/8 precision only when the frame allows it and they
//! are small (`lower_mv_precision`); otherwise they drop to even eighths.

use crate::inter_pred::BlockPosition;
use crate::tables::MV_REF_BLOCKS;
use crate::types::{ModeInfo, ModeInfoGrid, Mv, block_size_index};

/// `MV_BORDER`: candidates may point 16 samples beyond the frame (1/8 units).
const MV_BORDER: i32 = 16 << 3;
/// Neighbours searched (`MVREF_NEIGHBOURS`); the first two also count modes.
const NEIGHBOURS: usize = 8;
const COUNTED_NEIGHBOURS: usize = 2;

/// `counter_to_context`: inter-mode context of the summed neighbour modes
/// (`INVALID_CASE` entries cannot occur and map to 0).
const COUNTER_TO_CONTEXT: [usize; 19] = [2, 3, 4, 1, 3, 0, 0, 0, 0, 5, 5, 0, 5, 0, 0, 0, 0, 0, 6];

/// The block's reference vectors and inter-mode context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MvRefs {
    pub nearest: Mv,
    pub near: Mv,
    pub mode_context: usize,
}

/// `lower_mv_precision`: without 1/8 precision (not allowed, or a large
/// vector) odd components move one eighth towards zero.
fn lower_precision(mv: Mv, allow_hp: bool) -> Mv {
    if allow_hp && crate::motion::uses_high_precision(mv) {
        return mv;
    }
    let lower = |v: i32| if v & 1 != 0 { v - v.signum() } else { v };
    Mv {
        row: lower(mv.row),
        col: lower(mv.col),
    }
}

/// Reference vectors of the block at `position` (size `size_log2`) for
/// `LAST_FRAME` (`secondary_ref == false`) or `GOLDEN_FRAME`
/// (`secondary_ref == true`).
/// `previous` is the previous frame's mode info when `use_prev_frame_mvs`
/// holds (not error resilient, same size, previous frame shown and not a
/// key frame); `allow_hp` is the frame's `allow_high_precision_mv`.
pub(crate) fn find(
    grid: &ModeInfoGrid,
    previous: Option<&ModeInfoGrid>,
    position: &BlockPosition,
    size_log2: u8,
    allow_hp: bool,
    secondary_ref: bool,
) -> MvRefs {
    let (row, col) = (position.row, position.col);
    let blocks = &MV_REF_BLOCKS[block_size_index(size_log2)];
    let candidates = || {
        blocks
            .iter()
            .take(NEIGHBOURS)
            .map(|&[dr, dc]| grid.offset(row, col, isize::from(dr), isize::from(dc)))
    };
    let mode_counter: usize = candidates()
        .take(COUNTED_NEIGHBOURS)
        .flatten()
        .map(ModeInfo::mode_counter)
        .sum();
    let co_located = previous.map(|p| p.at(row, col));
    let mut list: Vec<Mv> = Vec::with_capacity(2);
    // libvpx's `different_ref_found` pass adds only inter candidates of the
    // other reference: with every neighbour on this one (all single-layer
    // frames) it adds nothing, so it is skipped.
    let mut different_ref_found = false;
    for info in candidates().flatten() {
        different_ref_found |= info.inter.is_some() && info.secondary_ref != secondary_ref;
        if let Some((_, mv)) = info.inter
            && info.secondary_ref == secondary_ref
            && list.first() != Some(&mv)
        {
            list.push(mv);
            if list.len() == 2 {
                break;
            }
        }
    }
    if list.len() < 2
        && let Some(info) = co_located
        && let Some((_, mv)) = info.inter
        && info.secondary_ref == secondary_ref
        && list.first() != Some(&mv)
    {
        list.push(mv);
    }
    if list.len() < 2 && different_ref_found {
        for info in candidates().flatten() {
            if let Some((_, mv)) = info.inter
                && info.secondary_ref != secondary_ref
                && list.first() != Some(&mv)
            {
                list.push(mv);
                if list.len() == 2 {
                    break;
                }
            }
        }
    }
    if list.len() < 2
        && let Some(info) = co_located
        && let Some((_, mv)) = info.inter
        && info.secondary_ref != secondary_ref
        && list.first() != Some(&mv)
    {
        list.push(mv);
    }
    let [left, right, top, bottom] = position.edges();
    let clamp = |mv: Mv| {
        lower_precision(
            Mv {
                row: mv.row.clamp(top - MV_BORDER, bottom + MV_BORDER),
                col: mv.col.clamp(left - MV_BORDER, right + MV_BORDER),
            },
            allow_hp,
        )
    };
    MvRefs {
        nearest: clamp(list.first().copied().unwrap_or(Mv::ZERO)),
        near: clamp(list.get(1).copied().unwrap_or(Mv::ZERO)),
        mode_context: COUNTER_TO_CONTEXT[mode_counter],
    }
}
