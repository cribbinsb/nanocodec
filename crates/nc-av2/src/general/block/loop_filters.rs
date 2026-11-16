//! What each decoded block leaves for the loop filters (dav2d `decode_b`'s
//! loop-filter mask updates and `decode_sb`'s per-64x64 quantizer): the
//! deblocking edges of its luma and chroma blocks, the units with a luma
//! residual, and its superblock's quantizer per 64x64 filter block.

use super::record::BlockRecord;
use super::transform::{self, LumaTransforms, TxSize};
use super::{Tile, context};
use crate::general::filters::UNITS64_LOG2;
use crate::general::filters::edges::{self, BlockEdges, NO_SUB_PU};
// Sub-PU edges of TIP frames keep their thresholds (dav2d `ds_subpu_mask`).
use crate::general::frame_header::TIP_FRAME_MODE;

impl Tile<'_, '_> {
    /// Record a decoded block for the loop filters; `transforms` is its
    /// luma transform layout.
    pub(super) fn record_filters(&mut self, record: &BlockRecord, transforms: LumaTransforms) {
        if let Some(size) = record.luma
            && !record.skip_txfm
        {
            let (bw4, bh4) = size.units();
            let columns = bw4.min(self.geometry.columns - record.column);
            let rows = bh4.min(self.geometry.rows - record.row);
            self.filters
                .mark_coded((record.row, record.column), (columns, rows));
        }
        let deblock = &self.frame.deblock;
        if !deblock.level_y.iter().any(|&on| on) {
            return;
        }
        let lossless = self.lossless();
        let frame = (self.geometry.columns, self.geometry.rows);
        let block = |size, chroma, tx, partition| BlockEdges {
            size,
            row: if chroma {
                record.chroma_row
            } else {
                record.row
            },
            column: if chroma {
                record.chroma_column
            } else {
                record.column
            },
            chroma,
            transformed: record.intra || !record.skip_txfm,
            lossless,
            tx,
            partition,
            // Intra and block-copy blocks have no sub-PU edges (dav2d
            // `subpu_flt_lvl`).
            sub_pu: if record.intra {
                NO_SUB_PU
            } else {
                self.sub_pu_level
            },
            scaled_sub_pu: self.frame.tip.mode != TIP_FRAME_MODE,
        };
        if let Some(size) = record.luma {
            let tx = match transforms {
                LumaTransforms::Lossless { large: true } => transform::luma_max(size, true),
                LumaTransforms::Lossless { large: false } => TxSize::TX4,
                LumaTransforms::Partition(partition) => {
                    partition.tx_size(size).unwrap_or(TxSize::TX4)
                }
            };
            let edges = block(size, false, tx, transforms.partition());
            let above = &mut self.contexts.above.deblock_luma[record.column as usize..];
            let left = &mut self.contexts.left.deblock_luma[context::left_index(record.row)..];
            edges::record_block(&mut self.filters.luma_edges, &edges, frame, above, left);
        }
        if let Some(size) = record.chroma
            && (deblock.level_u || deblock.level_v)
        {
            let tx = if lossless {
                TxSize::TX4
            } else {
                transform::chroma_max(size)
            };
            let edges = block(size, true, tx, transform::TxPartition::None);
            let above =
                &mut self.contexts.above.deblock_chroma[(record.chroma_column >> 1) as usize..];
            let left_row = context::left_index(record.chroma_row) >> 1;
            let left = &mut self.contexts.left.deblock_chroma[left_row..];
            edges::record_block(&mut self.filters.chroma_edges, &edges, frame, above, left);
        }
    }

    /// At a superblock's first luma block, after its quantizer delta: the
    /// superblock's quantizer for each of its 64x64 filter blocks.
    pub(super) fn record_qindex(&mut self) {
        let step = self.geometry.superblock_units;
        if (self.row | self.column) & (step - 1) != 0 {
            return;
        }
        let blocks = (step >> UNITS64_LOG2).max(1);
        for y in 0..blocks {
            for x in 0..blocks {
                let (row, column) = (
                    self.row + (y << UNITS64_LOG2),
                    self.column + (x << UNITS64_LOG2),
                );
                if row < self.geometry.rows && column < self.geometry.columns {
                    let index = self.filter_block(row, column);
                    self.filters.qindex[index] = self.qindex;
                }
            }
        }
    }
}
