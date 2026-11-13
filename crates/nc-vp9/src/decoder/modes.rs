//! Mode info of one block: segment id, skip, transform size, intra modes,
//! references, inter modes, interpolation filter and vectors, with their
//! neighbour contexts (VP9 §6.4.3-6.4.20, §9.3; libvpx `vp9_decodemv.c`
//! and `vp9_pred_common.[ch]`, BSD-3-Clause, see `THIRD_PARTY.md`).

use nc_common::decode::{DecodeError, Result};

use super::block::{
    ALTREF_FRAME, BlockInfo, BlockSize, GOLDEN_FRAME, INTRA_FRAME, LAST_FRAME, NO_REF,
    NO_SWITCHABLE_FILTER, PredMode, RefFrame,
};
use super::frame::{BlockPosition, FrameDecoder};
use super::header::{ReferenceMode, SegFeature, TxMode};
use crate::bool_coder::BoolReader;
use crate::motion;
use crate::segmentation::SEGMENT_TREE;
use crate::tables;
use crate::transform::TxSize;
use crate::tree;
use crate::types::{IntraMode, Mv};

/// `MV_LOW` / `MV_UPP`: decoded vectors lie strictly inside.
const MV_LIMIT: i32 = 1 << 14;
/// `counter_to_context` of the summed neighbour mode counters
/// (`INVALID_CASE` entries cannot occur).
const COUNTER_TO_CONTEXT: [usize; 19] = [2, 3, 4, 1, 3, 0, 0, 0, 0, 5, 5, 0, 5, 0, 0, 0, 0, 0, 6];

fn read_intra_mode(reader: &mut BoolReader<'_>, probs: &[u8]) -> IntraMode {
    IntraMode::from_index(tree::read(reader, tree::INTRA_MODE, probs))
}

impl FrameDecoder<'_> {
    fn above_info(&self, position: &BlockPosition) -> Option<&BlockInfo> {
        position.above.map(|i| &self.blocks[i as usize])
    }

    fn left_info(&self, position: &BlockPosition) -> Option<&BlockInfo> {
        position.left.map(|i| &self.blocks[i as usize])
    }

    /// `vp9_read_mode_info`.
    pub(super) fn read_mode_info(
        &mut self,
        reader: &mut BoolReader<'_>,
        position: &BlockPosition,
    ) -> Result<BlockInfo> {
        let mut info = BlockInfo {
            size: position.size,
            ..BlockInfo::default()
        };
        if self.inputs.header.is_intra() {
            self.read_intra_frame_mode_info(reader, position, &mut info);
        } else {
            self.read_inter_frame_mode_info(reader, position, &mut info)?;
        }
        Ok(info)
    }

    /// The last frame's segment ids under the block, at their minimum
    /// (`dec_get_segment_id`).
    fn predicted_segment(&self, position: &BlockPosition) -> u8 {
        let map = self.inputs.last_segment_map;
        (0..position.y_mis)
            .flat_map(|r| {
                let start = (position.mi_row + r) * self.mi_cols + position.mi_col;
                map[start..start + position.x_mis].iter().copied()
            })
            .min()
            .unwrap_or(0)
    }

    fn write_segment_map(&mut self, position: &BlockPosition, id: Option<u8>) {
        for r in 0..position.y_mis {
            let start = (position.mi_row + r) * self.mi_cols + position.mi_col;
            let range = start..start + position.x_mis;
            if let Some(id) = id {
                self.segment_map[range].fill(id);
            } else {
                // `copy_segment_id`: the map carries over.
                let last = &self.inputs.last_segment_map[range.clone()];
                self.segment_map[range].copy_from_slice(last);
            }
        }
    }

    /// `read_intra_segment_id` / `read_inter_segment_id`.
    fn read_segment_id(
        &mut self,
        reader: &mut BoolReader<'_>,
        position: &BlockPosition,
        info: &mut BlockInfo,
        inter_frame: bool,
    ) {
        let seg = &self.inputs.header.segmentation;
        if !seg.enabled {
            return;
        }
        let predicted = if inter_frame {
            self.predicted_segment(position)
        } else {
            0
        };
        if !seg.update_map {
            self.write_segment_map(position, None);
            info.segment_id = predicted;
            return;
        }
        let (tree_probs, pred_probs) = (seg.tree_probs, seg.pred_probs);
        let id = if inter_frame && seg.temporal_update {
            let ctx = usize::from(
                self.above_info(position)
                    .is_some_and(|b| b.seg_id_predicted),
            ) + usize::from(self.left_info(position).is_some_and(|b| b.seg_id_predicted));
            info.seg_id_predicted = reader.read(pred_probs[ctx]);
            if info.seg_id_predicted {
                predicted
            } else {
                tree::read(reader, SEGMENT_TREE, &tree_probs) as u8
            }
        } else {
            tree::read(reader, SEGMENT_TREE, &tree_probs) as u8
        };
        self.write_segment_map(position, Some(id));
        info.segment_id = id;
    }

    /// `read_skip`.
    fn read_skip(
        &mut self,
        reader: &mut BoolReader<'_>,
        position: &BlockPosition,
        segment: u8,
    ) -> bool {
        if self
            .inputs
            .header
            .segmentation
            .feature(segment, SegFeature::Skip)
            .is_some()
        {
            return true;
        }
        let ctx = usize::from(self.above_info(position).is_some_and(|b| b.skip))
            + usize::from(self.left_info(position).is_some_and(|b| b.skip));
        let skip = reader.read(self.inputs.context.skip[ctx]);
        self.counts.skip[ctx][usize::from(skip)] += 1;
        skip
    }

    /// `read_tx_size`: coded per block with `TX_MODE_SELECT` (when
    /// `allow_select`), else the largest the mode and block allow.
    fn read_tx_size(
        &mut self,
        reader: &mut BoolReader<'_>,
        position: &BlockPosition,
        allow_select: bool,
    ) -> TxSize {
        let max = position.size.max_tx();
        let tx_mode = self.inputs.compressed.tx_mode;
        if !(allow_select && tx_mode == TxMode::Select && !position.size.is_sub8x8()) {
            return max.min(tx_mode.largest());
        }
        // `get_tx_size_context`.
        let neighbour = |b: Option<&BlockInfo>| {
            b.map(|b| {
                if b.skip {
                    max as usize
                } else {
                    b.tx_size as usize
                }
            })
        };
        let (mut above, mut left) = (
            neighbour(self.above_info(position)),
            neighbour(self.left_info(position)),
        );
        if left.is_none() {
            left = above;
        }
        if above.is_none() {
            above = left;
        }
        let ctx = usize::from(
            above.unwrap_or(max as usize) + left.unwrap_or(max as usize) > max as usize,
        );
        let probs = self.inputs.context.tx_probs(max, ctx);
        let mut size = usize::from(reader.read(probs[0]));
        if size != 0 && max >= TxSize::Tx16 {
            size += usize::from(reader.read(probs[1]));
            if size != 1 && max >= TxSize::Tx32 {
                size += usize::from(reader.read(probs[2]));
            }
        }
        let tx = TxSize::from_log2(size as u8 + 2);
        self.counts.count_tx(max, ctx, tx);
        tx
    }

    /// `read_intra_frame_mode_info` (key and intra-only frames).
    fn read_intra_frame_mode_info(
        &mut self,
        reader: &mut BoolReader<'_>,
        position: &BlockPosition,
        info: &mut BlockInfo,
    ) {
        self.read_segment_id(reader, position, info, false);
        info.skip = self.read_skip(reader, position, info.segment_id);
        info.tx_size = self.read_tx_size(reader, position, true);
        info.ref_frame = [INTRA_FRAME, NO_REF];
        let above = self.above_info(position).copied();
        let left = self.left_info(position).copied();
        // `get_y_mode_probs`: neighbouring 4x4 modes, from this block's
        // own sub-blocks inside it (`vp9_above_block_mode`, `_left_`).
        let read = |reader: &mut BoolReader<'_>, info: &BlockInfo, b: usize| -> IntraMode {
            let above_mode = if b < 2 {
                above
                    .filter(|a| !a.is_inter())
                    .map_or(IntraMode::Dc, |a| a.y_mode(b + 2).intra())
            } else {
                info.sub_modes[b - 2].intra()
            };
            let left_mode = if b.is_multiple_of(2) {
                left.filter(|l| !l.is_inter())
                    .map_or(IntraMode::Dc, |l| l.y_mode(b + 1).intra())
            } else {
                info.sub_modes[b - 1].intra()
            };
            read_intra_mode(
                reader,
                &tables::KF_Y_MODE[above_mode as usize][left_mode as usize],
            )
        };
        let y_mode = match position.size {
            BlockSize::B4x4 => {
                for b in 0..4 {
                    info.sub_modes[b] = PredMode::Intra(read(reader, info, b));
                }
                info.sub_modes[3]
            }
            BlockSize::B4x8 => {
                let first = PredMode::Intra(read(reader, info, 0));
                info.sub_modes[0] = first;
                info.sub_modes[2] = first;
                let second = PredMode::Intra(read(reader, info, 1));
                info.sub_modes[1] = second;
                info.sub_modes[3] = second;
                second
            }
            BlockSize::B8x4 => {
                let first = PredMode::Intra(read(reader, info, 0));
                info.sub_modes[0] = first;
                info.sub_modes[1] = first;
                let second = PredMode::Intra(read(reader, info, 2));
                info.sub_modes[2] = second;
                info.sub_modes[3] = second;
                second
            }
            _ => PredMode::Intra(read(reader, info, 0)),
        };
        info.mode = y_mode;
        info.uv_mode = read_intra_mode(reader, &tables::KF_UV_MODE[y_mode.intra() as usize]);
    }

    /// `read_inter_frame_mode_info`.
    fn read_inter_frame_mode_info(
        &mut self,
        reader: &mut BoolReader<'_>,
        position: &BlockPosition,
        info: &mut BlockInfo,
    ) -> Result<()> {
        self.read_segment_id(reader, position, info, true);
        info.skip = self.read_skip(reader, position, info.segment_id);
        let inter = self.read_is_inter(reader, position, info.segment_id);
        info.tx_size = self.read_tx_size(reader, position, !info.skip || !inter);
        if inter {
            self.read_inter_block_mode_info(reader, position, info)
        } else {
            self.read_intra_block_mode_info(reader, position, info)
        }
    }

    /// `read_is_inter_block`.
    fn read_is_inter(
        &mut self,
        reader: &mut BoolReader<'_>,
        position: &BlockPosition,
        segment: u8,
    ) -> bool {
        if let Some(frame) = self
            .inputs
            .header
            .segmentation
            .feature(segment, SegFeature::RefFrame)
        {
            return frame != i16::from(INTRA_FRAME);
        }
        let (above, left) = (self.above_info(position), self.left_info(position));
        let ctx = match (above, left) {
            (Some(a), Some(l)) => {
                let (ai, li) = (!a.is_inter(), !l.is_inter());
                if ai && li { 3 } else { usize::from(ai || li) }
            }
            (Some(e), None) | (None, Some(e)) => 2 * usize::from(!e.is_inter()),
            (None, None) => 0,
        };
        let inter = reader.read(self.inputs.context.intra_inter[ctx]);
        self.counts.intra_inter[ctx][usize::from(inter)] += 1;
        inter
    }

    /// `read_intra_block_mode_info` (intra blocks of inter frames).
    fn read_intra_block_mode_info(
        &mut self,
        reader: &mut BoolReader<'_>,
        position: &BlockPosition,
        info: &mut BlockInfo,
    ) -> Result<()> {
        let fc = self.inputs.context;
        let mut read_y = |group: usize, counts: &mut crate::probs::Counts| -> Result<PredMode> {
            let mode = read_intra_mode(reader, &fc.y_mode[group]);
            counts.y_mode[group][mode as usize] += 1;
            Ok(PredMode::Intra(mode))
        };
        let counts = &mut self.counts;
        info.mode = match position.size {
            BlockSize::B4x4 => {
                for b in 0..4 {
                    info.sub_modes[b] = read_y(0, counts)?;
                }
                info.sub_modes[3]
            }
            BlockSize::B4x8 => {
                let first = read_y(0, counts)?;
                let second = read_y(0, counts)?;
                info.sub_modes = [first, second, first, second];
                second
            }
            BlockSize::B8x4 => {
                let first = read_y(0, counts)?;
                let second = read_y(0, counts)?;
                info.sub_modes = [first, first, second, second];
                second
            }
            size => read_y(size.size_group(), counts)?,
        };
        let y = info.mode.intra() as usize;
        info.uv_mode = read_intra_mode(reader, &fc.uv_mode[y]);
        self.counts.uv_mode[y][info.uv_mode as usize] += 1;
        info.filter = NO_SWITCHABLE_FILTER;
        info.ref_frame = [INTRA_FRAME, NO_REF];
        Ok(())
    }

    /// `read_ref_frames`.
    fn read_ref_frames(
        &mut self,
        reader: &mut BoolReader<'_>,
        position: &BlockPosition,
        segment: u8,
    ) -> [RefFrame; 2] {
        if let Some(frame) = self
            .inputs
            .header
            .segmentation
            .feature(segment, SegFeature::RefFrame)
        {
            return [frame as RefFrame, NO_REF];
        }
        let compressed = self.inputs.compressed;
        let compound = match compressed.reference_mode {
            ReferenceMode::Single => false,
            ReferenceMode::Compound => true,
            ReferenceMode::Select => {
                let ctx = self.reference_mode_context(position);
                let bit = reader.read(self.inputs.context.comp_inter[ctx]);
                self.counts.comp_inter[ctx][usize::from(bit)] += 1;
                bit
            }
        };
        if compound {
            let fixed = compressed.comp_fixed_ref;
            let index = usize::from(self.inputs.header.sign_bias[fixed as usize]);
            let ctx = self.comp_ref_context(position);
            let bit = reader.read(self.inputs.context.comp_ref[ctx]);
            self.counts.comp_ref[ctx][usize::from(bit)] += 1;
            let mut frames = [NO_REF; 2];
            frames[index] = fixed;
            frames[1 - index] = compressed.comp_var_ref[usize::from(bit)];
            frames
        } else {
            let ctx0 = self.single_ref_p1_context(position);
            let bit0 = reader.read(self.inputs.context.single_ref[ctx0][0]);
            self.counts.single_ref[ctx0][0][usize::from(bit0)] += 1;
            let frame = if bit0 {
                let ctx1 = self.single_ref_p2_context(position);
                let bit1 = reader.read(self.inputs.context.single_ref[ctx1][1]);
                self.counts.single_ref[ctx1][1][usize::from(bit1)] += 1;
                if bit1 { ALTREF_FRAME } else { GOLDEN_FRAME }
            } else {
                LAST_FRAME
            };
            [frame, NO_REF]
        }
    }

    /// `read_inter_block_mode_info`.
    fn read_inter_block_mode_info(
        &mut self,
        reader: &mut BoolReader<'_>,
        position: &BlockPosition,
        info: &mut BlockInfo,
    ) -> Result<()> {
        let allow_hp = self.inputs.header.allow_hp;
        info.ref_frame = self.read_ref_frames(reader, position, info.segment_id);
        let refs = 1 + usize::from(info.is_compound());
        let mode_ctx = self.mode_context(position);
        let size = position.size;
        if self
            .inputs
            .header
            .segmentation
            .feature(info.segment_id, SegFeature::Skip)
            .is_some()
        {
            info.mode = PredMode::ZeroMv;
            if size.is_sub8x8() {
                return Err(DecodeError::Malformed(
                    "VP9 segment skip on a sub-8x8 block",
                ));
            }
        } else if !size.is_sub8x8() {
            info.mode = self.read_inter_mode(reader, mode_ctx);
        }
        info.filter = match self.inputs.header.filter {
            Some(filter) => filter,
            None => self.read_switchable_filter(reader, position),
        };
        if size.is_sub8x8() {
            return self.read_sub8x8_vectors(reader, position, info, mode_ctx);
        }
        let mut best = [Mv::ZERO; 2];
        if info.mode != PredMode::ZeroMv {
            for (r, best) in best.iter_mut().enumerate().take(refs) {
                let (list, count) = self.find_mv_refs(position, info.mode, info.ref_frame[r], None);
                *best = lower_precision(list[count - 1], allow_hp);
            }
        }
        info.mv = self.assign_mv(reader, info.mode, best, best, refs)?;
        Ok(())
    }

    /// Per-4x4 modes and vectors of a sub-8x8 inter block.
    fn read_sub8x8_vectors(
        &mut self,
        reader: &mut BoolReader<'_>,
        position: &BlockPosition,
        info: &mut BlockInfo,
        mode_ctx: usize,
    ) -> Result<()> {
        let allow_hp = self.inputs.header.allow_hp;
        let refs = 1 + usize::from(info.is_compound());
        let (w, h) = (position.size.width_4x4(), position.size.height_4x4());
        let mut best_new = [Mv::ZERO; 2];
        let mut have_new = false;
        let mut mode = PredMode::ZeroMv;
        for idy in (0..2).step_by(h) {
            for idx in (0..2).step_by(w) {
                let block = idy * 2 + idx;
                mode = self.read_inter_mode(reader, mode_ctx);
                let mut best_sub = [Mv::ZERO; 2];
                match mode {
                    PredMode::NearestMv | PredMode::NearMv => {
                        for (r, sub) in best_sub.iter_mut().enumerate().take(refs) {
                            *sub = self.append_sub8x8_mv(position, info, mode, block, r);
                        }
                    }
                    PredMode::NewMv if !have_new => {
                        for (r, best) in best_new.iter_mut().enumerate().take(refs) {
                            let (list, _) = self.find_mv_refs(
                                position,
                                PredMode::NewMv,
                                info.ref_frame[r],
                                None,
                            );
                            *best = lower_precision(list[0], allow_hp);
                        }
                        have_new = true;
                    }
                    _ => {}
                }
                let mvs = self.assign_mv(reader, mode, best_new, best_sub, refs)?;
                info.sub_modes[block] = mode;
                info.sub_mvs[block] = mvs;
                if h == 2 {
                    info.sub_modes[block + 2] = mode;
                    info.sub_mvs[block + 2] = mvs;
                }
                if w == 2 {
                    info.sub_modes[block + 1] = mode;
                    info.sub_mvs[block + 1] = mvs;
                }
            }
        }
        info.mode = mode;
        info.mv = info.sub_mvs[3];
        Ok(())
    }

    /// `assign_mv`: the block's vectors for `mode`; `best` is the NEWMV
    /// base, `near_nearest` the NEAREST/NEAR choice.
    fn assign_mv(
        &mut self,
        reader: &mut BoolReader<'_>,
        mode: PredMode,
        best: [Mv; 2],
        near_nearest: [Mv; 2],
        refs: usize,
    ) -> Result<[Mv; 2]> {
        match mode {
            PredMode::NewMv => {
                let allow_hp = self.inputs.header.allow_hp;
                let mut mvs = [Mv::ZERO; 2];
                for r in 0..refs {
                    let usehp = allow_hp && motion::uses_high_precision(best[r]);
                    let (dr, dc) =
                        motion::decode(reader, self.inputs.context, &mut self.counts, usehp)?;
                    mvs[r] = Mv {
                        row: best[r].row + dr,
                        col: best[r].col + dc,
                    };
                    let valid = |v: i32| v > -MV_LIMIT && v < MV_LIMIT;
                    if !valid(mvs[r].row) || !valid(mvs[r].col) {
                        return Err(DecodeError::Malformed("VP9 motion vector range"));
                    }
                }
                Ok(mvs)
            }
            PredMode::NearestMv | PredMode::NearMv => Ok(near_nearest),
            _ => Ok([Mv::ZERO; 2]),
        }
    }

    fn read_inter_mode(&mut self, reader: &mut BoolReader<'_>, ctx: usize) -> PredMode {
        let index = tree::read(
            reader,
            tree::INTER_MODE,
            &self.inputs.context.inter_mode[ctx],
        );
        self.counts.inter_mode[ctx][index] += 1;
        PredMode::inter(index)
    }

    /// `read_switchable_interp_filter` with
    /// `get_pred_context_switchable_interp`.
    fn read_switchable_filter(
        &mut self,
        reader: &mut BoolReader<'_>,
        position: &BlockPosition,
    ) -> u8 {
        let left = self
            .left_info(position)
            .map_or(NO_SWITCHABLE_FILTER, |b| b.filter);
        let above = self
            .above_info(position)
            .map_or(NO_SWITCHABLE_FILTER, |b| b.filter);
        let ctx = usize::from(if left == above {
            left
        } else if left == NO_SWITCHABLE_FILTER {
            above
        } else if above == NO_SWITCHABLE_FILTER {
            left
        } else {
            NO_SWITCHABLE_FILTER
        });
        let filter = tree::read(
            reader,
            tree::SWITCHABLE_INTERP,
            &self.inputs.context.switchable_interp[ctx],
        );
        self.counts.switchable_interp[ctx][filter] += 1;
        filter as u8
    }

    /// `get_mode_context`: from the modes of the first two candidates.
    fn mode_context(&self, position: &BlockPosition) -> usize {
        let counter: usize = self
            .candidates(position)
            .take(2)
            .flatten()
            .map(|b| self.blocks[b as usize].mode.counter())
            .sum();
        COUNTER_TO_CONTEXT[counter]
    }

    /// `vp9_get_reference_mode_context`.
    fn reference_mode_context(&self, position: &BlockPosition) -> usize {
        let fixed = self.inputs.compressed.comp_fixed_ref;
        match (self.above_info(position), self.left_info(position)) {
            (Some(a), Some(l)) => {
                if !a.is_compound() && !l.is_compound() {
                    usize::from((a.ref_frame[0] == fixed) ^ (l.ref_frame[0] == fixed))
                } else if !a.is_compound() {
                    2 + usize::from(a.ref_frame[0] == fixed || !a.is_inter())
                } else if !l.is_compound() {
                    2 + usize::from(l.ref_frame[0] == fixed || !l.is_inter())
                } else {
                    4
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if e.is_compound() {
                    3
                } else {
                    usize::from(e.ref_frame[0] == fixed)
                }
            }
            (None, None) => 1,
        }
    }

    /// `vp9_get_pred_context_comp_ref_p`.
    fn comp_ref_context(&self, position: &BlockPosition) -> usize {
        let compressed = self.inputs.compressed;
        let fix_ref_idx =
            usize::from(self.inputs.header.sign_bias[compressed.comp_fixed_ref as usize]);
        let var = 1 - fix_ref_idx;
        let (var0, var1) = (compressed.comp_var_ref[0], compressed.comp_var_ref[1]);
        let fixed = compressed.comp_fixed_ref;
        match (self.above_info(position), self.left_info(position)) {
            (Some(a), Some(l)) => {
                let (ai, li) = (!a.is_inter(), !l.is_inter());
                if ai && li {
                    2
                } else if ai || li {
                    let e = if ai { l } else { a };
                    if e.is_compound() {
                        1 + 2 * usize::from(e.ref_frame[var] != var1)
                    } else {
                        1 + 2 * usize::from(e.ref_frame[0] != var1)
                    }
                } else {
                    let (l_sg, a_sg) = (!l.is_compound(), !a.is_compound());
                    let vrfa = if a_sg {
                        a.ref_frame[0]
                    } else {
                        a.ref_frame[var]
                    };
                    let vrfl = if l_sg {
                        l.ref_frame[0]
                    } else {
                        l.ref_frame[var]
                    };
                    if vrfa == vrfl && var1 == vrfa {
                        0
                    } else if l_sg && a_sg {
                        if (vrfa == fixed && vrfl == var0) || (vrfl == fixed && vrfa == var0) {
                            4
                        } else if vrfa == vrfl {
                            3
                        } else {
                            1
                        }
                    } else if l_sg || a_sg {
                        let vrfc = if l_sg { vrfa } else { vrfl };
                        let rfs = if a_sg { vrfa } else { vrfl };
                        if vrfc == var1 && rfs != var1 {
                            1
                        } else if rfs == var1 && vrfc != var1 {
                            2
                        } else {
                            4
                        }
                    } else if vrfa == vrfl {
                        4
                    } else {
                        2
                    }
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if !e.is_inter() {
                    2
                } else if e.is_compound() {
                    4 * usize::from(e.ref_frame[var] != var1)
                } else {
                    3 * usize::from(e.ref_frame[0] != var1)
                }
            }
            (None, None) => 2,
        }
    }

    /// `vp9_get_pred_context_single_ref_p1`.
    fn single_ref_p1_context(&self, position: &BlockPosition) -> usize {
        let has_last = |b: &BlockInfo| b.ref_frame[0] == LAST_FRAME || b.ref_frame[1] == LAST_FRAME;
        let edge = |e: &BlockInfo| {
            if e.is_compound() {
                1 + usize::from(has_last(e))
            } else {
                4 * usize::from(e.ref_frame[0] == LAST_FRAME)
            }
        };
        match (self.above_info(position), self.left_info(position)) {
            (Some(a), Some(l)) => {
                let (ai, li) = (!a.is_inter(), !l.is_inter());
                if ai && li {
                    2
                } else if ai || li {
                    edge(if ai { l } else { a })
                } else {
                    let (a2, l2) = (a.is_compound(), l.is_compound());
                    if a2 && l2 {
                        1 + usize::from(has_last(a) || has_last(l))
                    } else if a2 || l2 {
                        let rfs = if a2 { l.ref_frame[0] } else { a.ref_frame[0] };
                        let comp = if a2 { a } else { l };
                        if rfs == LAST_FRAME {
                            3 + usize::from(has_last(comp))
                        } else {
                            usize::from(has_last(comp))
                        }
                    } else {
                        2 * usize::from(a.ref_frame[0] == LAST_FRAME)
                            + 2 * usize::from(l.ref_frame[0] == LAST_FRAME)
                    }
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if e.is_inter() {
                    edge(e)
                } else {
                    2
                }
            }
            (None, None) => 2,
        }
    }

    /// `vp9_get_pred_context_single_ref_p2`.
    fn single_ref_p2_context(&self, position: &BlockPosition) -> usize {
        let has_golden =
            |b: &BlockInfo| b.ref_frame[0] == GOLDEN_FRAME || b.ref_frame[1] == GOLDEN_FRAME;
        match (self.above_info(position), self.left_info(position)) {
            (Some(a), Some(l)) => {
                let (ai, li) = (!a.is_inter(), !l.is_inter());
                if ai && li {
                    2
                } else if ai || li {
                    let e = if ai { l } else { a };
                    if e.is_compound() {
                        1 + 2 * usize::from(has_golden(e))
                    } else if e.ref_frame[0] == LAST_FRAME {
                        3
                    } else {
                        4 * usize::from(e.ref_frame[0] == GOLDEN_FRAME)
                    }
                } else {
                    let (a2, l2) = (a.is_compound(), l.is_compound());
                    let (above0, above1, left0, left1) = (
                        a.ref_frame[0],
                        a.ref_frame[1],
                        l.ref_frame[0],
                        l.ref_frame[1],
                    );
                    if a2 && l2 {
                        if above0 == left0 && above1 == left1 {
                            3 * usize::from(has_golden(a) || has_golden(l))
                        } else {
                            2
                        }
                    } else if a2 || l2 {
                        let rfs = if a2 { left0 } else { above0 };
                        let comp = if a2 { a } else { l };
                        if rfs == GOLDEN_FRAME {
                            3 + usize::from(has_golden(comp))
                        } else if rfs == ALTREF_FRAME {
                            usize::from(has_golden(comp))
                        } else {
                            1 + 2 * usize::from(has_golden(comp))
                        }
                    } else if above0 == LAST_FRAME && left0 == LAST_FRAME {
                        3
                    } else if above0 == LAST_FRAME || left0 == LAST_FRAME {
                        let edge0 = if above0 == LAST_FRAME { left0 } else { above0 };
                        4 * usize::from(edge0 == GOLDEN_FRAME)
                    } else {
                        2 * usize::from(above0 == GOLDEN_FRAME)
                            + 2 * usize::from(left0 == GOLDEN_FRAME)
                    }
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if !e.is_inter() || (e.ref_frame[0] == LAST_FRAME && !e.is_compound()) {
                    2
                } else if !e.is_compound() {
                    4 * usize::from(e.ref_frame[0] == GOLDEN_FRAME)
                } else {
                    3 * usize::from(has_golden(e))
                }
            }
            (None, None) => 2,
        }
    }
}

/// `lower_mv_precision`: without 1/8 precision (not allowed, or a large
/// vector) odd components move one eighth towards zero.
pub(super) fn lower_precision(mv: Mv, allow_hp: bool) -> Mv {
    if allow_hp && motion::uses_high_precision(mv) {
        return mv;
    }
    let lower = |v: i32| if v & 1 != 0 { v - v.signum() } else { v };
    Mv {
        row: lower(mv.row),
        col: lower(mv.col),
    }
}
