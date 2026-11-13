//! Transform trees and residuals of inter blocks (AV1 §5.11.17
//! `read_var_tx_size`, §5.11.34–36 `residual()` and `transform_tree()`):
//! each largest transform of the block splits up to twice, its leaves are
//! read 64x64 chunk by chunk, and chroma transforms take the type of the
//! luma transform at their position (§5.11.40 `compute_tx_type`).
//!
//! The traversal follows libaom `read_tx_size_vartx` and
//! `decode_reconstruct_tx` (BSD-2-Clause, see `THIRD_PARTY.md`).
use super::super::block_size::BlockSize;
use super::super::deblock::EdgeBlock;
use super::super::transform::{TxSize, chroma_block};
use super::{BlockPos, FrameDecoder, MI_SIZE};
use crate::decoder_range::SymbolReader;
use nc_common::PlaneIndex;
use nc_common::decode::DecodeError;
use nc_dsp::Av1TxType;

/// `MAX_VARTX_DEPTH`: a largest transform splits at most twice.
const MAX_VARTX_DEPTH: usize = 2;
/// 4x4 units per 64x64 chunk, which large blocks code one at a time.
const CHUNK_MI: usize = 16;
/// The `skip` of every block [`FrameDecoder::inter_residual`] decodes.
const NOT_SKIPPED: bool = false;

/// One luma transform of an inter block: its offset in the block (4x4
/// units), size and decoded type.
#[derive(Debug, Clone, Copy)]
pub(in super::super) struct TxLeaf {
    x4: usize,
    y4: usize,
    tx: TxSize,
    kind: Av1TxType,
}

impl FrameDecoder<'_> {
    /// The luma transforms of an inter block into `maps.tx_leaves`
    /// (§5.11.15 `read_block_tx_size`): a tree per largest transform in
    /// `TX_MODE_SELECT` frames, else largest transforms; and the
    /// transform-size contexts the block leaves.
    pub(super) fn read_inter_tx(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        skip: bool,
    ) -> Result<(), DecodeError> {
        self.maps.tx_leaves.clear();
        let largest = self.largest_tx(block.size);
        let (max_w4, max_h4) = self.visible_units(block);
        let (step_w, step_h) = (largest.width() / MI_SIZE, largest.height() / MI_SIZE);
        let select = self.header.tx_mode_select && block.size != BlockSize::BLOCK_4X4 && !skip;
        for y4 in (0..max_h4).step_by(step_h) {
            for x4 in (0..max_w4).step_by(step_w) {
                if select {
                    self.read_tx_tree(r, block, (max_w4, max_h4), largest, 0, (x4, y4))?;
                } else {
                    self.push_leaf((x4, y4), largest);
                }
            }
        }
        if !select {
            let at = (block.mi_col, block.mi_row);
            let units = block.units();
            // Skipped inter blocks leave their own size (libaom
            // `set_txfm_ctxs`).
            let sides = if skip {
                (block.size.width(), block.size.height())
            } else {
                (largest.width(), largest.height())
            };
            self.txfm.fill(at, units, (sides.0 as u8, sides.1 as u8));
        }
        Ok(())
    }

    /// The block's 4x4 units inside the frame.
    fn visible_units(&self, block: BlockPos) -> (usize, usize) {
        let (mi_cols, mi_rows) = (self.header.width / MI_SIZE, self.header.height / MI_SIZE);
        (
            block.size.width_mi().min(mi_cols - block.mi_col),
            block.size.height_mi().min(mi_rows - block.mi_row),
        )
    }

    fn push_leaf(&mut self, (x4, y4): (usize, usize), tx: TxSize) {
        self.maps.tx_leaves.push(TxLeaf {
            x4,
            y4,
            tx,
            kind: Av1TxType::DctDct,
        });
    }

    /// `read_var_tx_size()` of the `tx` transform at 4x4 offset (`x4`,
    /// `y4`) in the block, `depth` splits down; the block's units inside
    /// the frame are `max_w4` x `max_h4`.
    fn read_tx_tree(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        (max_w4, max_h4): (usize, usize),
        tx: TxSize,
        depth: usize,
        (x4, y4): (usize, usize),
    ) -> Result<(), DecodeError> {
        if x4 >= max_w4 || y4 >= max_h4 {
            return Ok(());
        }
        let at = (block.mi_col + x4, block.mi_row + y4);
        let units = (tx.width() / MI_SIZE, tx.height() / MI_SIZE);
        let split = depth < MAX_VARTX_DEPTH && {
            let side = block.size.width_log2().max(block.size.height_log2());
            let ctx = self
                .txfm
                .split_context(at, side, (tx.width_log2(), tx.height_log2()));
            r.boolean(&mut self.cdfs.txfm_split[ctx])
        };
        let sub = tx.split().filter(|_| split);
        match sub {
            None => {
                self.push_leaf((x4, y4), tx);
                self.txfm
                    .fill(at, units, (tx.width() as u8, tx.height() as u8));
            }
            Some(sub) if sub.width() == MI_SIZE && sub.height() == MI_SIZE => {
                // Splits down to 4x4 stop there.
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    if dx < units.0 && dy < units.1 {
                        self.push_leaf((x4 + dx, y4 + dy), sub);
                    }
                }
                self.txfm
                    .fill(at, units, (sub.width() as u8, sub.height() as u8));
            }
            Some(sub) => {
                let (sw, sh) = (sub.width() / MI_SIZE, sub.height() / MI_SIZE);
                for dy in (0..units.1).step_by(sh) {
                    for dx in (0..units.0).step_by(sw) {
                        let at = (x4 + dx, y4 + dy);
                        self.read_tx_tree(r, block, (max_w4, max_h4), sub, depth + 1, at)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Record the block's luma transforms for deblocking: one area when a
    /// single transform covers it.
    pub(super) fn record_inter_edges(&mut self, block: BlockPos, edge_block: EdgeBlock) {
        let at = (block.mi_col, block.mi_row);
        let leaves = &self.maps.tx_leaves;
        let whole = match leaves[..] {
            [leaf]
                if leaf.tx.width() == block.size.width()
                    && leaf.tx.height() == block.size.height() =>
            {
                Some((leaf.tx.width_log2(), leaf.tx.height_log2()))
            }
            _ => None,
        };
        if whole.is_none() {
            for leaf in leaves {
                let origin = (block.mi_col + leaf.x4, block.mi_row + leaf.y4);
                let tx = (leaf.tx.width_log2(), leaf.tx.height_log2());
                self.maps.edges.leaf(origin, tx, edge_block);
            }
        }
        self.maps
            .edges
            .block(at, edge_block, whole, block.has_chroma());
    }

    /// `residual()` of a non-skipped inter block (§5.11.34): per 64x64
    /// chunk, the luma leaves inside it, then each chroma plane's
    /// transforms. Skipped blocks code no residual and only reset their
    /// contexts ([`Self::reset_block_contexts`]).
    #[inline(never)]
    pub(super) fn inter_residual(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
    ) -> Result<(), DecodeError> {
        let (max_w4, max_h4) = self.visible_units(block);
        let luma_block = (block.size.width(), block.size.height());
        for row in (0..max_h4).step_by(CHUNK_MI) {
            for col in (0..max_w4).step_by(CHUNK_MI) {
                let rows = row..row + CHUNK_MI;
                let cols = col..col + CHUNK_MI;
                for index in 0..self.maps.tx_leaves.len() {
                    let leaf = self.maps.tx_leaves[index];
                    if !rows.contains(&leaf.y4) || !cols.contains(&leaf.x4) {
                        continue;
                    }
                    let x = (block.mi_col + leaf.x4) * MI_SIZE;
                    let y = (block.mi_row + leaf.y4) * MI_SIZE;
                    let source =
                        self.tx_type_source(PlaneIndex::Y, leaf.tx, true, 0, Av1TxType::DctDct);
                    let fills = (leaf.tx.width(), leaf.tx.height()) == luma_block;
                    let kind = self.transform_block(
                        r,
                        PlaneIndex::Y,
                        NOT_SKIPPED,
                        leaf.tx,
                        fills,
                        (x, y),
                        source,
                    )?;
                    self.maps.tx_leaves[index].kind = kind;
                }
                if block.has_chroma() {
                    for plane in [PlaneIndex::U, PlaneIndex::V] {
                        self.inter_chroma_chunk(r, block, plane, (col, row))?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The transforms of one chroma plane inside the chunk at luma 4x4
    /// offset (`col`, `row`).
    fn inter_chroma_chunk(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        plane: PlaneIndex,
        (col, row): (usize, usize),
    ) -> Result<(), DecodeError> {
        let (max_w4, max_h4) = self.visible_units(block);
        let tx = self.chroma_tx(block.size);
        let (chroma_w, chroma_h) = chroma_block(block.size);
        let fills = (tx.width_log2(), tx.height_log2()) == (chroma_w, chroma_h);
        let (step_w, step_h) = (tx.width() / MI_SIZE, tx.height() / MI_SIZE);
        let end_w = ((col + CHUNK_MI).min(max_w4) + 1) >> 1;
        let end_h = ((row + CHUNK_MI).min(max_h4) + 1) >> 1;
        let origin = (block.mi_col >> 1, block.mi_row >> 1);
        for blk_row in ((row >> 1)..end_h).step_by(step_h) {
            for blk_col in ((col >> 1)..end_w).step_by(step_w) {
                let luma = self.luma_tx_type((blk_col << 1, blk_row << 1));
                let source = self.tx_type_source(plane, tx, true, 0, luma);
                let at = (
                    (origin.0 + blk_col) * MI_SIZE,
                    (origin.1 + blk_row) * MI_SIZE,
                );
                self.transform_block(r, plane, NOT_SKIPPED, tx, fills, at, source)?;
            }
        }
        Ok(())
    }

    /// The type of the luma transform covering 4x4 offset (`x4`, `y4`) of
    /// the block.
    fn luma_tx_type(&self, (x4, y4): (usize, usize)) -> Av1TxType {
        self.maps
            .tx_leaves
            .iter()
            .find(|leaf| {
                (leaf.x4..leaf.x4 + leaf.tx.width() / MI_SIZE).contains(&x4)
                    && (leaf.y4..leaf.y4 + leaf.tx.height() / MI_SIZE).contains(&y4)
            })
            .map_or(Av1TxType::DctDct, |leaf| leaf.kind)
    }
}
