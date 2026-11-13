//! Transform sizes and residuals (AV1 §5.11.15–17 transform size syntax,
//! §5.11.34–36 `residual()` and `transform_block()`): the walk over a
//! block's transform blocks in 64x64 chunks, clipped to the frame, the
//! coefficient contexts of each (§8.3.2), and reconstruction.
use super::super::availability::PlaneTransform;
use super::super::block_info::{BlockInfo, Neighbours};
use super::super::block_size::BlockSize;
use super::super::coefficients::{self, Coded, Request, TxTypeSource, TxbNeighbours};
use super::super::transform::{TxSet, TxSize, chroma_block, intra_mode_tx_type};
use super::intra::{IntraModes, IntraTransform};
use super::{BlockPos, FrameDecoder};
use crate::backend::PlaneType;
use crate::decoder_range::SymbolReader;
use nc_common::decode::DecodeError;
use nc_common::{PlaneIndex, PlaneViewMut};
use nc_dsp::intra_prediction::av1_block::{EdgeAvail, EdgeFilterCtx};
use nc_dsp::{Av1TxType, Dsp};

/// 4x4 units per 64x64 chunk, which large blocks code one at a time.
const CHUNK_MI: usize = 16;
/// Coefficients of a 4x4 transform.
const TX_4X4_COEFFICIENTS: usize = 16;
/// `MAX_TX_DEPTH`: intra `tx_depth` splits at most twice.
const MAX_TX_DEPTH: usize = 2;

impl FrameDecoder<'_> {
    /// The largest luma transform of a block (`Max_Tx_Size_Rect`), 4x4 in
    /// lossless frames (§5.11.16 `read_tx_size`, `TX_MODE_ONLY_4X4`).
    pub(super) fn largest_tx(&self, size: BlockSize) -> TxSize {
        if self.header.lossless {
            TxSize::TX_4X4
        } else {
            TxSize::largest(size)
        }
    }
    /// The chroma transform of a block (§5.11.37 `get_tx_size`), 4x4 in
    /// lossless frames.
    pub(super) fn chroma_tx(&self, size: BlockSize) -> TxSize {
        if self.header.lossless {
            TxSize::TX_4X4
        } else {
            TxSize::chroma(size)
        }
    }
    /// `read_tx_size()` of an intra block (§5.11.16): the largest
    /// transform, split `tx_depth` times in `TX_MODE_SELECT` frames.
    pub(super) fn intra_tx_size(
        &mut self,
        reader: &mut SymbolReader<'_>,
        block: BlockPos,
        near: Neighbours,
    ) -> Result<TxSize, DecodeError> {
        let largest = self.largest_tx(block.size);
        if !self.header.tx_mode_select || block.size == BlockSize::BLOCK_4X4 {
            return Ok(largest);
        }
        // Splits down to 4x4 choose the CDF category; at most two are coded.
        let mut splits = 0;
        let mut smaller = largest;
        while let Some(next) = smaller.split() {
            splits += 1;
            smaller = next;
        }
        let ctx = near.tx_depth_context(
            usize::from(self.txfm.above(block.mi_col)),
            usize::from(self.txfm.left(block.mi_row)),
            (largest.width(), largest.height()),
        );
        let depth = reader.symbol(
            &mut self.cdfs.tx_depth[splits - 1][ctx],
            splits.min(MAX_TX_DEPTH) + 1,
        );
        let mut tx = largest;
        for _ in 0..depth {
            tx = tx
                .split()
                .ok_or(DecodeError::Malformed("AV1 transform depth"))?;
        }
        Ok(tx)
    }

    /// `residual()` of an intra block (§5.11.34): each 64x64 chunk, each
    /// plane, its transform blocks in raster order, each predicted then
    /// reconstructed. Transform blocks outside the frame are skipped.
    pub(super) fn intra_residual(
        &mut self,
        reader: &mut SymbolReader<'_>,
        block: BlockPos,
        modes: &IntraModes,
        skip: bool,
        luma_tx: TxSize,
    ) -> Result<(), DecodeError> {
        let (mi_cols, mi_rows) = (self.header.width / 4, self.header.height / 4);
        let max_w4 = block.size.width_mi().min(mi_cols - block.mi_col);
        let max_h4 = block.size.height_mi().min(mi_rows - block.mi_row);
        let planes: &[PlaneIndex] = if block.has_chroma() {
            &PlaneIndex::ALL
        } else {
            &[PlaneIndex::Y]
        };
        let chroma_tx = self.chroma_tx(block.size);
        let (chroma_w, chroma_h) = chroma_block(block.size);
        // The end of the block's decoded luma (`MaxLumaW`, `MaxLumaH`).
        let mut max_luma: (usize, usize) = (0, 0);
        for row in (0..max_h4).step_by(CHUNK_MI) {
            for col in (0..max_w4).step_by(CHUNK_MI) {
                for &plane in planes {
                    let ss = usize::from(plane != PlaneIndex::Y);
                    let (tx, block_units) = if ss == 0 {
                        (luma_tx, (block.size.width_mi(), block.size.height_mi()))
                    } else {
                        (chroma_tx, (1 << (chroma_w - 2), 1 << (chroma_h - 2)))
                    };
                    let (step_w, step_h) = (tx.width() / 4, tx.height() / 4);
                    let end_w = ((col + CHUNK_MI).min(max_w4) + ss) >> ss;
                    let end_h = ((row + CHUNK_MI).min(max_h4) + ss) >> ss;
                    let origin = (block.mi_col >> ss, block.mi_row >> ss);
                    let fills_block = (tx.width() / 4, tx.height() / 4) == block_units;
                    for blk_row in ((row >> ss)..end_h).step_by(step_h) {
                        for blk_col in ((col >> ss)..end_w).step_by(step_w) {
                            let x = (origin.0 + blk_col) * 4;
                            let y = (origin.1 + blk_row) * 4;
                            let geometry = PlaneTransform {
                                block_origin: origin,
                                block_size: block_units,
                                offset: (blk_col, blk_row),
                                size: (step_w, step_h),
                                subsampled: ss != 0,
                            };
                            let edge_filter = self.edge_filter(plane, block);
                            let avail = self.edge_avail(plane, block, &geometry);
                            let luma_extent = (
                                max_luma.0.saturating_sub(x * 2),
                                max_luma.1.saturating_sub(y * 2),
                            );
                            let transform = IntraTransform {
                                plane,
                                origin: (x, y),
                                size: (tx.width(), tx.height()),
                                avail,
                                edge_filter,
                            };
                            self.predict_intra(transform, modes, luma_extent)?;
                            if ss == 0 {
                                max_luma = (x + tx.width(), y + tx.height());
                            }
                            let source = if ss == 0 {
                                self.tx_type_source(
                                    plane,
                                    tx,
                                    false,
                                    modes.luma.intra_dir(),
                                    Av1TxType::DctDct,
                                )
                            } else {
                                let kind = intra_mode_tx_type(modes.uv);
                                self.tx_type_source(plane, tx, false, 0, kind)
                            };
                            self.transform_block(
                                reader,
                                plane,
                                skip,
                                tx,
                                fills_block,
                                (x, y),
                                source,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// `haveAbove`, `haveLeft` and the corners of a transform block of
    /// `plane` (§5.11.35 `transform_block`).
    fn edge_avail(&self, plane: PlaneIndex, block: BlockPos, tx: &PlaneTransform) -> EdgeAvail {
        let (above, left) = block.available(plane);
        let have_above = tx.offset.1 > 0 || above;
        let have_left = tx.offset.0 > 0 || left;
        let (have_above_right, have_below_left) = self.decoded.corners(tx, have_above, have_left);
        EdgeAvail {
            have_above,
            have_left,
            have_above_right,
            have_below_left,
        }
    }

    /// The intra edge filter's context (§7.11.2.8 `get_filter_type`):
    /// whether the above or left block of `plane` is smooth-predicted.
    fn edge_filter(&self, plane: PlaneIndex, block: BlockPos) -> Option<EdgeFilterCtx> {
        if !self.intra_edge_filter {
            return None;
        }
        let (above, left) = block.available(plane);
        let (smooth_above, smooth_left) = if plane == PlaneIndex::Y {
            let near = self
                .maps
                .blocks
                .neighbours(block.mi_col, block.mi_row, (above, left));
            (
                above && near.above.is_some_and(BlockInfo::smooth),
                left && near.left.is_some_and(BlockInfo::smooth),
            )
        } else {
            (
                above && self.uv_smooth.above(block.mi_col | 1),
                left && self.uv_smooth.left(block.mi_row | 1),
            )
        };
        Some(EdgeFilterCtx {
            smooth_above,
            smooth_left,
        })
    }

    /// `reset_block_context()` (§5.11.5) of a skipped inter block: the
    /// coefficient contexts along the block read 0 in every plane it
    /// carries. This is what its transform blocks would leave one by one
    /// (units beyond the frame always hold 0).
    pub(super) fn reset_block_contexts(&mut self, block: BlockPos) {
        let planes = if block.has_chroma() { 3 } else { 1 };
        let (w4, h4) = block.units();
        for index in 0..planes {
            let ss = usize::from(index > 0);
            let span = |at: usize, units: usize| (at >> ss)..((at + units - 1) >> ss) + 1;
            let columns = span(block.mi_col, w4);
            let rows = span(block.mi_row, h4);
            let first = rows.start % (self.superblock.height_mi() >> ss);
            self.above[index][columns].fill(0);
            self.left[index][first..first + rows.len()].fill(0);
        }
    }

    /// Set the coefficient contexts a transform block leaves over `w4` x
    /// `h4` units from plane unit (`x4`, `y4`); units outside the frame
    /// hold 0 (libaom `av1_set_entropy_contexts`).
    fn set_contexts(
        &mut self,
        plane: PlaneIndex,
        (x4, y4): (usize, usize),
        (w4, h4): (usize, usize),
        value: u8,
    ) {
        let index = plane as usize;
        let ss = usize::from(index > 0);
        let (columns, rows) = (
            (self.header.width / 4) >> ss,
            (self.header.height / 4) >> ss,
        );
        let above = &mut self.above[index][x4..x4 + w4];
        above.fill(value);
        if x4 + w4 > columns {
            above[columns.saturating_sub(x4)..].fill(0);
        }
        let row = y4 % (self.superblock.height_mi() >> ss);
        let left = &mut self.left[index][row..row + h4];
        left.fill(value);
        if y4 + h4 > rows {
            left[rows.saturating_sub(y4)..].fill(0);
        }
    }

    /// How the type of a `tx` transform of `plane` is found (§5.11.47,
    /// `compute_tx_type`): luma types are coded unless the set has one
    /// type or `qindex` is 0; chroma takes `chroma_kind` if its set has it;
    /// lossless blocks are `DCT_DCT` (reconstructed by the WHT).
    pub(super) fn tx_type_source(
        &self,
        plane: PlaneIndex,
        tx: TxSize,
        inter: bool,
        intra_dir: usize,
        chroma_kind: Av1TxType,
    ) -> TxTypeSource {
        if self.header.lossless {
            return TxTypeSource::Known(Av1TxType::DctDct);
        }
        let set = TxSet::of(tx, inter, self.header.reduced_tx_set);
        if plane != PlaneIndex::Y {
            let kind = if set.contains(chroma_kind) {
                chroma_kind
            } else {
                Av1TxType::DctDct
            };
            return TxTypeSource::Known(kind);
        }
        if set == TxSet::DctOnly || self.header.base_q_idx == 0 {
            return TxTypeSource::Known(Av1TxType::DctDct);
        }
        TxTypeSource::Coded {
            set,
            intra_dir: (!inter).then_some(intra_dir),
        }
    }
    /// One transform block of `plane` at plane sample (`x`, `y`): its
    /// coefficients (with the neighbour contexts) and inverse transform;
    /// returns its type.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(super) fn transform_block(
        &mut self,
        reader: &mut SymbolReader<'_>,
        plane: PlaneIndex,
        skip: bool,
        tx: TxSize,
        fills_block: bool,
        (x, y): (usize, usize),
        source: TxTypeSource,
    ) -> Result<Av1TxType, DecodeError> {
        let plane_index = plane as usize;
        let shift = usize::from(plane_index > 0);
        let (w4, h4) = (tx.width() / 4, tx.height() / 4);
        let (x4, y4) = (x / 4, y / 4);
        let row = y4 % (self.superblock.height_mi() >> shift);
        let coded = if skip {
            Coded::NONE
        } else {
            let pt = if plane_index == 0 {
                PlaneType::Luma
            } else {
                PlaneType::Chroma
            };
            let request = Request {
                tx,
                plane: pt,
                neighbours: TxbNeighbours {
                    above: &self.above[plane_index][x4..x4 + w4],
                    left: &self.left[plane_index][row..row + h4],
                    fills_block,
                },
                source,
                steps: self.steps[plane_index],
            };
            coefficients::read(
                &mut self.cdfs.coefficients,
                reader,
                &mut self.coefficients,
                request,
            )?
        };
        self.set_contexts(plane, (x4, y4), (w4, h4), coded.context);
        if coded.eob > 0 {
            let mut dst = self
                .recon
                .plane_mut(plane)
                .region_mut(x as isize, y as isize, tx.width(), tx.height())
                .ok_or(DecodeError::Malformed("AV1 inverse rectangle"))?;
            if self.header.lossless {
                inverse_wht(self.dsp, &self.coefficients.coefficients, &mut dst);
            } else {
                self.dsp.av1_inverse_add(
                    tx.dsp(),
                    coded.kind,
                    &self.coefficients.coefficients,
                    coded.eob,
                    &mut dst,
                );
            }
            self.coefficients.clear();
        }
        Ok(coded.kind)
    }
}

/// §7.13.3: lossless blocks invert the 4x4 Walsh-Hadamard of the first
/// `coefficients`.
// Out of line: our own streams are never lossless.
#[cold]
#[inline(never)]
fn inverse_wht(dsp: &Dsp, coefficients: &[i32], dst: &mut PlaneViewMut<'_>) {
    let coefficients = coefficients[..TX_4X4_COEFFICIENTS]
        .try_into()
        .expect("4x4 coefficients");
    dsp.av1_inverse_wht_add(coefficients, dst);
}
