//! An inter block (dav2d `decode_b`'s inter branch through `recon_b`):
//! its syntax ([`super::inter_syntax`]), transform partition, motion,
//! prediction and residual, and what later blocks' contexts read of it.
use super::context::NO_DIRECTION;
use super::inter_pred::InterPrediction;
use super::inter_record::{CompoundType, InterMode, InterRecord, NO_REFERENCE, TIP_FRAME};
use super::inter_syntax::TIP_WEIGHTS;
use super::intra::{IntraModes, mode};
use super::motion::Mv;
use super::record::BlockRecord;
use super::size::BlockSize;
use super::syntax::Neighbours;
use super::transform::{LumaTransforms, TxPartition};
use super::{Tile, context};
use crate::general::filters::edges::NO_SUB_PU;
use crate::general::frame_header::{TIP_FRAME_MODE, filter_mode};
use nc_common::decode::{DecodeError, Result};

impl Tile<'_, '_> {
    /// Decode an inter block; returns its luma transform layout for the
    /// loop filters.
    pub(super) fn inter_block(
        &mut self,
        depth: usize,
        record: &BlockRecord,
        neighbours: Neighbours,
    ) -> Result<LumaTransforms> {
        let size = record
            .luma
            .ok_or(DecodeError::Malformed("AV2 inter block without luma"))?;
        let (mut inter, precision_context) = self.inter_syntax(depth, record, neighbours)?;
        let transforms = self.transform_partition(depth, size, record, false)?;
        let masks = self.masks;
        let wedge = (inter.compound == CompoundType::Wedge).then(|| {
            let (bw4, bh4) = size.units();
            let luma = ((bw4 << 2) as usize, (bh4 << 2) as usize);
            masks.wedge_tmvp(luma, inter.wedge_index as usize)
        });
        let motion = self.resolve_motion(depth, record, &mut inter, wedge)?;
        let prediction = InterPrediction {
            inter,
            mv: motion.mv,
            warp: motion.warp,
            size,
        };
        self.record_covered(&prediction);
        self.sub_pu_level = self.sub_pu_level(&prediction);
        self.inter_prediction = Some(prediction);
        let modes = IntraModes {
            y_mode: inter.mode as u8,
            uv_mode: mode::DC,
            direction: NO_DIRECTION,
            ..IntraModes::default()
        };
        let reconstructed = self.reconstruct(depth, record, &modes, transforms);
        self.inter_prediction = None;
        reconstructed?;
        self.record_intra_contexts(record, &modes, 0);
        self.record_inter_contexts(record, &inter, precision_context);
        Ok(transforms)
    }

    /// Record the block's motion contexts over its extent (dav2d
    /// `decode_b`'s inter `set_ctx`).
    fn record_inter_contexts(
        &mut self,
        record: &BlockRecord,
        inter: &InterRecord,
        precision_context: u8,
    ) {
        let Some(size) = record.luma else {
            return;
        };
        let (bw4, bh4) = size.units();
        let values = [
            inter.compound as u8,
            inter.filter,
            inter.motion_mode as u8,
            u8::from(inter.amvd),
            precision_context,
        ];
        let references = inter.references.map(|r| r as u8);
        for (edge, start, count) in [
            (&mut self.contexts.above, record.column as usize, bw4),
            (
                &mut self.contexts.left,
                context::left_index(record.row),
                bh4,
            ),
        ] {
            let end = (start + count as usize).min(edge.mode.len());
            for (field, value) in [
                (&mut edge.comp_type, values[0]),
                (&mut edge.filter, values[1]),
                (&mut edge.motion_mode, values[2]),
                (&mut edge.amvd, values[3]),
                (&mut edge.mv_precision, values[4]),
            ] {
                field[start..end].fill(value);
            }
            for (side, reference) in edge.reference.iter_mut().zip(references) {
                side[start..end].fill(reference);
            }
        }
    }

    /// The log2 (in 4x4 units) of the sub-blocks whose edges an inter block
    /// deblocks inside itself: TIP, optical-flow and refined blocks predict
    /// per sub-block (dav2d `subpu_flt_lvl`); [`NO_SUB_PU`] for none.
    fn sub_pu_level(&self, prediction: &InterPrediction) -> u8 {
        let inter = &prediction.inter;
        if !self.frame.deblock.sub_pu {
            return NO_SUB_PU;
        }
        let (bw4, bh4) = prediction.size.units();
        if inter.references[0] == TIP_FRAME {
            let refines = self.sequence.inter.tip_refine_mv
                && (self.frame.tip.mode == 1 || self.frame.tip.filter == filter_mode::SHARP);
            let large = if self.frame.tip.mode == TIP_FRAME_MODE {
                !refines
            } else {
                (!refines && bw4.min(bh4) >= 4) || prediction.size == BlockSize::B256x256
            };
            return 1 + u8::from(large);
        }
        if inter.references[1] != NO_REFERENCE {
            if inter.mode >= InterMode::OpflNearNear {
                return 1 - u8::from(prediction.size == BlockSize::B8x8);
            }
            if inter.refine_mv != 0 && inter.compound == CompoundType::Average {
                return 2;
            }
        }
        NO_SUB_PU
    }

    /// A superblock of a TIP frame: one TIP block of the superblock's
    /// size, the frame's TIP vector, no residual (dav2d
    /// `tip_frame_recon_sb`).
    pub(super) fn tip_superblock(&mut self) -> Result<()> {
        let root = self.geometry.root;
        (self.chroma_row, self.chroma_column) = (self.row, self.column);
        let record = BlockRecord {
            luma: Some(root),
            chroma: Some(root),
            row: self.row,
            column: self.column,
            chroma_row: self.row,
            chroma_column: self.column,
            skip_mode: false,
            intra: false,
            intrabc: false,
            skip_txfm: true,
            qindex: self.qindex,
        };
        let tip = &self.frame.tip;
        let inter = InterRecord {
            references: [TIP_FRAME, NO_REFERENCE],
            mode: InterMode::Near,
            filter: tip.filter,
            weight: TIP_WEIGHTS[usize::from(tip.global_weight)],
            ..InterRecord::default()
        };
        let mv = Mv {
            y: i32::from(tip.motion.0),
            x: i32::from(tip.motion.1),
        };
        let prediction = InterPrediction {
            inter,
            mv: [mv, Mv::default()],
            warp: [None, None],
            size: root,
        };
        self.sub_pu_level = self.sub_pu_level(&prediction);
        self.inter_prediction = Some(prediction);
        let modes = IntraModes {
            y_mode: inter.mode as u8,
            uv_mode: mode::DC,
            direction: NO_DIRECTION,
            ..IntraModes::default()
        };
        let transforms = LumaTransforms::Partition(TxPartition::None);
        let reconstructed = self.reconstruct(0, &record, &modes, transforms);
        self.inter_prediction = None;
        reconstructed?;
        if tip.apply_filter {
            self.record_filters(&record, transforms);
            self.record_qindex();
        }
        Ok(())
    }
}
