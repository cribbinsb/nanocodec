//! An inter block (dav2d `decode_b`'s inter branch through `recon_b`):
//! its syntax ([`super::inter_syntax`]), transform partition, motion,
//! prediction and residual, and what later blocks' contexts read of it.
use super::context::NO_DIRECTION;
use super::inter_record::InterRecord;
use super::intra::{IntraModes, mode};
use super::record::BlockRecord;
use super::syntax::Neighbours;
use super::transform::LumaTransforms;
use super::{Tile, context};
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
        let (inter, precision_context) = self.inter_syntax(depth, record, neighbours)?;
        self.has_inter_blocks = true;
        let transforms = self.transform_partition(depth, size, record, false)?;
        let modes = IntraModes {
            y_mode: inter.mode as u8,
            uv_mode: mode::DC,
            direction: NO_DIRECTION,
            ..IntraModes::default()
        };
        self.reconstruct(depth, record, &modes, transforms)?;
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
}
