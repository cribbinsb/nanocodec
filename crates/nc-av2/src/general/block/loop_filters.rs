//! What each decoded block leaves for the loop filters (dav2d `decode_b`'s
//! loop-filter mask updates and `decode_sb`'s per-64x64 quantizer): the
//! deblocking edges of its luma and chroma blocks, the units with a luma
//! residual, and its superblock's quantizer per 64x64 filter block.
//!
//! The deblocking contexts restart at each tile, so the edges along a
//! tile's left and top sides first take the tile's own transform levels;
//! once the tile is decoded they are capped by the levels the neighbouring
//! tile left at its right or bottom side (dav2d `dav2d_deblock_slice_cols`'
//! fix-ups at tile columns and rows from `tx_db_right_edge` and the above
//! contexts of the tile row before; TIP frames fix tile rows only).

use super::record::BlockRecord;
use super::transform::{self, LumaTransforms, TxSize};
use super::{Geometry, Tile, context};
use crate::general::filters::UNITS64_LOG2;
use crate::general::filters::edges::{self, BlockEdges, Direction, NO_SUB_PU};
use crate::general::sample::Av2Sample;
// Sub-PU edges of TIP frames keep their thresholds (dav2d `ds_subpu_mask`).
use crate::general::frame_header::TIP_FRAME_MODE;

/// The deblocking contexts at tile boundaries, per luma and chroma (per
/// unit row): the levels the tile being decoded leaves at its right side
/// and those the tile to its left left; (per unit column) those the tile
/// row above left at its bottom. Empty for a frame of one tile.
#[derive(Debug, Default)]
pub(super) struct TileEdges {
    right: [Vec<u8>; 2],
    left: [Vec<u8>; 2],
    bottom: [Vec<u8>; 2],
}

impl TileEdges {
    /// The boundary contexts of a frame of `geometry` with `tiles` tiles.
    pub(super) fn new(geometry: Geometry, tiles: usize) -> Self {
        if tiles == 1 {
            return Self::default();
        }
        let (rows, columns) = (geometry.rows as usize, geometry.columns as usize);
        let sides = || {
            [
                vec![edges::RESET_LUMA_CONTEXT; rows],
                vec![edges::RESET_CHROMA_CONTEXT; rows.div_ceil(2)],
            ]
        };
        Self {
            right: sides(),
            left: sides(),
            bottom: [
                vec![edges::RESET_LUMA_CONTEXT; columns],
                vec![edges::RESET_CHROMA_CONTEXT; columns.div_ceil(2)],
            ],
        }
    }

    /// At the start of a tile: the levels the tile before it left at its
    /// right side become those at this tile's left.
    pub(super) fn next_tile(&mut self) {
        std::mem::swap(&mut self.left, &mut self.right);
    }
}

impl<T: Av2Sample> Tile<'_, '_, T> {
    /// After superblock row `row` of a tile with another to its right:
    /// keep the levels at its right side (dav2d `tx_db_right_edge`).
    pub(super) fn save_right_edge(&mut self, row: u32) {
        if self.bounds.column_end == self.geometry.columns {
            return;
        }
        let rows = (self.bounds.row_end - row).min(self.geometry.superblock_units) as usize;
        let start = context::left_index(row);
        let luma = row as usize;
        self.boundary_levels.right[0][luma..luma + rows]
            .copy_from_slice(&self.contexts.left.deblock_luma[start..start + rows]);
        let chroma_rows = rows.div_ceil(2);
        self.boundary_levels.right[1][luma / 2..luma / 2 + chroma_rows].copy_from_slice(
            &self.contexts.left.deblock_chroma[start / 2..start / 2 + chroma_rows],
        );
    }

    /// Once a tile is decoded: cap the edge levels along its top and left
    /// sides by its neighbours' (no left fix-up in TIP frames, as dav2d),
    /// and keep the levels at its bottom for the tile row below.
    pub(super) fn finish_tile_edges(&mut self) {
        let bounds = self.bounds;
        let columns = bounds.column_start as usize..bounds.column_end as usize;
        let rows = bounds.row_start as usize..bounds.row_end as usize;
        let halve = |range: &std::ops::Range<usize>| range.start / 2..range.end.div_ceil(2);
        let (chroma_columns, chroma_rows) = (halve(&columns), halve(&rows));
        let maps = &mut self.filters;
        if bounds.row_start > 0 {
            let (luma_row, chroma_row) = (rows.start, chroma_rows.start);
            for column in columns.clone() {
                let limit = self.boundary_levels.bottom[0][column];
                maps.luma_edges
                    .cap(Direction::Horizontal, column, luma_row, limit);
            }
            for column in chroma_columns.clone() {
                let limit = self.boundary_levels.bottom[1][column];
                maps.chroma_edges
                    .cap(Direction::Horizontal, column, chroma_row, limit);
            }
        }
        if bounds.column_start > 0 && self.frame.tip.mode != TIP_FRAME_MODE {
            let (luma_column, chroma_column) = (columns.start, chroma_columns.start);
            for row in rows.clone() {
                let limit = self.boundary_levels.left[0][row];
                maps.luma_edges
                    .cap(Direction::Vertical, luma_column, row, limit);
            }
            for row in chroma_rows {
                let limit = self.boundary_levels.left[1][row];
                maps.chroma_edges
                    .cap(Direction::Vertical, chroma_column, row, limit);
            }
        }
        if bounds.row_end < self.geometry.rows {
            self.boundary_levels.bottom[0][columns.clone()]
                .copy_from_slice(&self.contexts.above.deblock_luma[columns]);
            self.boundary_levels.bottom[1][chroma_columns.clone()]
                .copy_from_slice(&self.contexts.above.deblock_chroma[chroma_columns]);
        }
    }

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
