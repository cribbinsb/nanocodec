//! Candidate reference vectors of a block for any reference frame, with
//! sign-bias scaling, previous-frame vectors and the sub-8x8 rules
//! (VP9 §6.4.22-6.4.24; libvpx `vp9_decodemv.c` `dec_find_mv_refs`,
//! `append_sub8x8_mvs_for_idx` and `vp9_mvref_common.h`, BSD-3-Clause, see
//! `THIRD_PARTY.md`).

use super::block::{BlockInfo, INTRA_FRAME, PredMode, RefFrame};
use super::frame::{BlockPosition, FrameDecoder};
use crate::tables::MV_REF_BLOCKS;
use crate::types::Mv;

/// `MVREF_NEIGHBOURS`: candidate positions searched per block size.
const NEIGHBOURS: usize = 8;
/// The first two candidates may supply sub-block vectors.
const SUB_BLOCK_CANDIDATES: usize = 2;
/// `MV_BORDER`: candidates may point 16 samples beyond the frame (1/8).
const MV_BORDER: i32 = 16 << 3;
/// `idx_n_column_to_subblock`: which 4x4 vector of a sub-8x8 neighbour
/// borders 4x4 block `[block]` (`[0]` left neighbour, `[1]` above).
const NEIGHBOUR_SUB_BLOCK: [[usize; 2]; 4] = [[1, 2], [1, 3], [3, 2], [3, 3]];

/// The candidate list being built (`ADD_MV_REF_LIST_EB`).
struct List {
    mvs: [Mv; 2],
    count: usize,
    /// NEARMV needs two entries; other modes stop at the first.
    early_break: bool,
}

impl List {
    /// Add `mv`; returns whether the search is done.
    fn add(&mut self, mv: Mv) -> bool {
        if self.count == 0 {
            self.mvs[0] = mv;
            self.count = 1;
            self.early_break
        } else if mv != self.mvs[0] {
            self.mvs[1] = mv;
            self.count = 2;
            true
        } else {
            false
        }
    }
}

impl FrameDecoder<'_> {
    /// Block indices at the candidate positions, `None` outside the frame
    /// or the tile's columns (`is_inside`).
    pub(super) fn candidates<'s>(
        &'s self,
        position: &'s BlockPosition,
    ) -> impl Iterator<Item = Option<u32>> + 's {
        MV_REF_BLOCKS[position.size as usize]
            .iter()
            .take(NEIGHBOURS)
            .map(move |&[dr, dc]| {
                let row = position.mi_row.checked_add_signed(isize::from(dr))?;
                let col = position.mi_col.checked_add_signed(isize::from(dc))?;
                let (start, end) = position.tile_cols;
                (row < self.mi_rows && col >= start && col < end)
                    .then(|| self.grid[row * self.mi_cols + col])
            })
    }

    /// `scale_mv`: a neighbour's vector for another reference, negated
    /// when the two references lie on opposite sides in time.
    fn scaled(&self, mv: Mv, from: RefFrame, to: RefFrame) -> Mv {
        let bias = &self.inputs.header.sign_bias;
        if bias[from as usize] == bias[to as usize] {
            mv
        } else {
            Mv {
                row: -mv.row,
                col: -mv.col,
            }
        }
    }

    /// `dec_find_mv_refs`: up to two distinct candidates for `ref_frame`,
    /// clamped near the frame; the count says which entry the mode uses
    /// (`refmv_count - 1`). `block` selects sub-8x8 neighbour vectors.
    pub(super) fn find_mv_refs(
        &self,
        position: &BlockPosition,
        mode: PredMode,
        ref_frame: RefFrame,
        block: Option<usize>,
    ) -> ([Mv; 2], usize) {
        let mut list = List {
            mvs: [Mv::ZERO; 2],
            count: 0,
            early_break: mode != PredMode::NearMv,
        };
        let done = self.search(position, ref_frame, block, &mut list);
        if !done {
            list.count = if mode == PredMode::NearMv { 2 } else { 1 };
        }
        // `clamp_mv_ref`.
        for mv in &mut list.mvs[..list.count] {
            mv.row = mv
                .row
                .clamp(position.to_top - MV_BORDER, position.to_bottom + MV_BORDER);
            mv.col = mv
                .col
                .clamp(position.to_left - MV_BORDER, position.to_right + MV_BORDER);
        }
        (list.mvs, list.count)
    }

    /// The candidate passes; returns whether the list completed early.
    fn search(
        &self,
        position: &BlockPosition,
        ref_frame: RefFrame,
        block: Option<usize>,
        list: &mut List,
    ) -> bool {
        let offsets = &MV_REF_BLOCKS[position.size as usize];
        let mut different_ref_found = false;
        for (i, candidate) in self.candidates(position).enumerate() {
            let Some(index) = candidate else { continue };
            let candidate = &self.blocks[index as usize];
            different_ref_found = true;
            let Some(which) = candidate.ref_frame.iter().position(|&f| f == ref_frame) else {
                continue;
            };
            let mv = match block {
                Some(block) if i < SUB_BLOCK_CANDIDATES && candidate.size.is_sub8x8() => {
                    let from_above = offsets[i][1] == 0;
                    candidate.sub_mvs[NEIGHBOUR_SUB_BLOCK[block][usize::from(from_above)]][which]
                }
                _ => candidate.mv[which],
            };
            if list.add(mv) {
                return true;
            }
        }
        let previous = self
            .inputs
            .previous_mvs
            .map(|mvs| mvs[position.mi_row * self.mi_cols + position.mi_col]);
        if let Some(prev) = previous
            && let Some(which) = prev.ref_frame.iter().position(|&f| f == ref_frame)
            && list.add(prev.mv[which])
        {
            return true;
        }
        if different_ref_found {
            for candidate in self.candidates(position).flatten() {
                let candidate = &self.blocks[candidate as usize];
                if !candidate.is_inter() {
                    continue;
                }
                if candidate.ref_frame[0] != ref_frame
                    && list.add(self.scaled(candidate.mv[0], candidate.ref_frame[0], ref_frame))
                {
                    return true;
                }
                if candidate.is_compound()
                    && candidate.ref_frame[1] != ref_frame
                    && candidate.mv[1] != candidate.mv[0]
                    && list.add(self.scaled(candidate.mv[1], candidate.ref_frame[1], ref_frame))
                {
                    return true;
                }
            }
        }
        if let Some(prev) = previous {
            if prev.ref_frame[0] != ref_frame
                && prev.ref_frame[0] > INTRA_FRAME
                && list.add(self.scaled(prev.mv[0], prev.ref_frame[0], ref_frame))
            {
                return true;
            }
            if prev.ref_frame[1] > INTRA_FRAME
                && prev.ref_frame[1] != ref_frame
                && prev.mv[1] != prev.mv[0]
                && list.add(self.scaled(prev.mv[1], prev.ref_frame[1], ref_frame))
            {
                return true;
            }
        }
        false
    }

    /// `append_sub8x8_mvs_for_idx`: the NEAREST/NEAR vector of 4x4 block
    /// `block` for reference `r`, preferring vectors of this block's
    /// earlier 4x4 blocks.
    pub(super) fn append_sub8x8_mv(
        &self,
        position: &BlockPosition,
        info: &BlockInfo,
        mode: PredMode,
        block: usize,
        r: usize,
    ) -> Mv {
        let ref_frame = info.ref_frame[r];
        let own = |b: usize| info.sub_mvs[b][r];
        let first_other =
            |list: [Mv; 2], than: Mv| list.into_iter().find(|&mv| mv != than).unwrap_or(Mv::ZERO);
        match block {
            0 => {
                let (list, count) = self.find_mv_refs(position, mode, ref_frame, Some(0));
                list[count - 1]
            }
            1 | 2 => {
                if mode == PredMode::NearestMv {
                    own(0)
                } else {
                    let (list, _) = self.find_mv_refs(position, mode, ref_frame, Some(block));
                    first_other(list, own(0))
                }
            }
            _ => {
                if mode == PredMode::NearestMv {
                    own(2)
                } else if own(2) != own(1) {
                    own(1)
                } else if own(2) != own(0) {
                    own(0)
                } else {
                    let (list, _) = self.find_mv_refs(position, mode, ref_frame, Some(block));
                    first_other(list, own(2))
                }
            }
        }
    }
}
