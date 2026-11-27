//! Transform sizes and residuals (AV1 §5.11.15–17 transform size syntax,
//! §5.11.34–36 `residual()` and `transform_block()`): the walk over a
//! block's transform blocks in 64x64 chunks, clipped to the frame, the
//! coefficient contexts of each (§8.3.2), and reconstruction.
use super::super::availability::PlaneTransform;
use super::super::block_info::{BlockInfo, Neighbours, splat_bytes};
use super::super::block_size::BlockSize;
use super::super::coefficients::{self, Coded, Request, TxTypeSource, TxbNeighbours};
use super::super::strided::strided;
use super::super::transform::{TxSet, TxSize, chroma_block};
use super::intra::{IntraModes, IntraTransform};
use super::{BlockPos, FrameDecoder};
use crate::backend::PlaneType;
use crate::decoder_range::SymbolReader;
use crate::tx_type::intra_mode_tx_type;
use nc_common::decode::DecodeError;
use nc_common::{PlaneIndex, PlaneViewMut};
use nc_dsp::intra_prediction::av1_block::{EdgeAvail, EdgeFilterCtx};
use nc_dsp::{Av1Coefficient, Av1TxType, DepthKernels, Pixel};

/// 4x4 units per 64x64 chunk, which large blocks code one at a time.
const CHUNK_MI: usize = 16;
/// Coefficients of a 4x4 transform, and its side (log2).
const TX_4X4_COEFFICIENTS: usize = 16;
const TX_4X4_SIDE_LOG2: u8 = 2;
/// `MAX_TX_DEPTH`: intra `tx_depth` splits at most twice.
const MAX_TX_DEPTH: usize = 2;

impl<T: Pixel> FrameDecoder<'_, T> {
    /// The largest luma transform of a block (`Max_Tx_Size_Rect`), 4x4 in
    /// lossless blocks (§5.11.16 `read_tx_size`).
    pub(super) fn largest_tx(&self, size: BlockSize) -> TxSize {
        if self.block_lossless {
            TxSize::TX_4X4
        } else {
            TxSize::largest(size)
        }
    }
    /// The chroma transform of a block (§5.11.37 `get_tx_size`), 4x4 in
    /// lossless blocks (§5.11.34 `residual`).
    pub(super) fn chroma_tx(&self, size: BlockSize) -> TxSize {
        if self.block_lossless {
            TxSize::TX_4X4
        } else {
            TxSize::chroma(size, self.subsampling(PlaneIndex::U))
        }
    }
    /// The chroma transform the loop filter records for `block`: its
    /// [`Self::chroma_tx`] when the block carries its chroma.
    pub(super) fn recorded_chroma_tx(&self, block: BlockPos) -> Option<TxSize> {
        block.has_chroma().then(|| self.chroma_tx(block.size))
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
        if self.block_lossless || !self.header.tx_mode_select || block.size == BlockSize::BLOCK_4X4
        {
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
        let visible = (
            block.size.width_mi().min(mi_cols - block.mi_col),
            block.size.height_mi().min(mi_rows - block.mi_row),
        );
        let subsampling = self.subsampling(PlaneIndex::U);
        let chroma_tx = self.chroma_tx(block.size);
        let (chroma_w, chroma_h) = chroma_block(block.size, subsampling);
        let chroma_units = (1 << (chroma_w - 2), 1 << (chroma_h - 2));
        let luma_units = (block.size.width_mi(), block.size.height_mi());
        let mut walk = IntraWalk {
            block,
            modes,
            skip,
            visible,
            chunk: (0, 0),
            max_luma: (0, 0),
        };
        for row in strided(0..visible.1, CHUNK_MI) {
            for col in strided(0..visible.0, CHUNK_MI) {
                walk.chunk = (col, row);
                self.intra_plane::<0, 0>(reader, &mut walk, PlaneIndex::Y, (luma_tx, luma_units))?;
                if !block.has_chroma() {
                    continue;
                }
                for plane in [PlaneIndex::U, PlaneIndex::V] {
                    let transforms = (chroma_tx, chroma_units);
                    match subsampling {
                        (1, 1) => self.intra_plane::<1, 1>(reader, &mut walk, plane, transforms),
                        (1, 0) => self.intra_plane::<1, 0>(reader, &mut walk, plane, transforms),
                        _ => self.intra_plane::<0, 0>(reader, &mut walk, plane, transforms),
                    }?;
                }
            }
        }
        Ok(())
    }

    /// The transform blocks of one plane of an intra block inside the
    /// walk's chunk: `tx` transforms over a plane block of `block_units`
    /// 4x4 units, the plane subsampled by (`SX`, `SY`). The subsampling is
    /// a constant so the luma and 4:2:0 walks fold their shifts.
    #[inline]
    fn intra_plane<const SX: usize, const SY: usize>(
        &mut self,
        reader: &mut SymbolReader<'_>,
        walk: &mut IntraWalk<'_>,
        plane: PlaneIndex,
        (tx, block_units): (TxSize, (usize, usize)),
    ) -> Result<(), DecodeError> {
        let IntraWalk {
            block,
            modes,
            skip,
            visible: (max_w4, max_h4),
            chunk: (col, row),
            ..
        } = *walk;
        let (step_w, step_h) = (tx.width() / 4, tx.height() / 4);
        let end_w = ((col + CHUNK_MI).min(max_w4) + SX) >> SX;
        let end_h = ((row + CHUNK_MI).min(max_h4) + SY) >> SY;
        let origin = (block.mi_col >> SX, block.mi_row >> SY);
        let fills_block = (step_w, step_h) == block_units;
        for blk_row in strided((row >> SY)..end_h, step_h) {
            for blk_col in strided((col >> SX)..end_w, step_w) {
                let x = (origin.0 + blk_col) * 4;
                let y = (origin.1 + blk_row) * 4;
                let geometry = PlaneTransform {
                    block_origin: origin,
                    block_size: block_units,
                    offset: (blk_col, blk_row),
                    size: (step_w, step_h),
                    subsampling: (SX, SY),
                };
                let edge_filter = self.edge_filter(plane, block);
                let avail = self.edge_avail(plane, block, &geometry);
                let luma_extent = (
                    walk.max_luma.0.saturating_sub(x << SX),
                    walk.max_luma.1.saturating_sub(y << SY),
                );
                let transform = IntraTransform {
                    plane,
                    origin: (x, y),
                    size: (tx.width(), tx.height()),
                    avail,
                    edge_filter,
                };
                self.predict_intra(transform, modes, luma_extent)?;
                let source = if plane == PlaneIndex::Y {
                    walk.max_luma = (x + tx.width(), y + tx.height());
                    self.tx_type_source(plane, tx, false, modes.luma.intra_dir(), Av1TxType::DctDct)
                } else {
                    let kind = intra_mode_tx_type(modes.uv);
                    self.tx_type_source(plane, tx, false, 0, kind)
                };
                self.transform_block(reader, plane, skip, tx, fills_block, (x, y), source)?;
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
            // The chroma of a sub-8x8 area is the last block's of it
            // (§7.11.2.8: `c++` / `r--` along a subsampled axis).
            let (ss_x, ss_y) = self.subsampling(plane);
            (
                above && self.uv_smooth.above(block.mi_col | ss_x),
                left && self.uv_smooth.left(block.mi_row | ss_y),
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
        let (w4, h4) = block.units();
        let sb_rows = self.superblock.height_mi();
        let first = block.mi_row % sb_rows;
        splat_bytes(&mut self.above[0][block.mi_col..block.mi_col + w4], 0);
        splat_bytes(&mut self.left[0][first..first + h4], 0);
        if !block.has_chroma() {
            return;
        }
        // U and V share their units.
        let (ss_x, ss_y) = self.subsampling(PlaneIndex::U);
        let span = |at: usize, units: usize, ss: usize| (at >> ss)..((at + units - 1) >> ss) + 1;
        let columns = span(block.mi_col, w4, ss_x);
        let rows = span(block.mi_row, h4, ss_y);
        let first = rows.start % (sb_rows >> ss_y);
        for plane in [PlaneIndex::U, PlaneIndex::V] {
            let index = plane as usize;
            splat_bytes(&mut self.above[index][columns.clone()], 0);
            splat_bytes(&mut self.left[index][first..first + rows.len()], 0);
        }
    }

    /// Set the coefficient contexts a transform block leaves over `w4` x
    /// `h4` units from plane unit (`x4`, `y4`); units outside the frame
    /// hold 0 (libaom `av1_set_entropy_contexts`).
    #[inline]
    fn set_contexts(
        &mut self,
        plane: PlaneIndex,
        (x4, y4): (usize, usize),
        (w4, h4): (usize, usize),
        value: u8,
    ) {
        let index = plane as usize;
        let (ss_x, ss_y) = self.subsampling(plane);
        let (columns, rows) = (
            (self.header.width / 4) >> ss_x,
            (self.header.height / 4) >> ss_y,
        );
        let above = &mut self.above[index][x4..x4 + w4];
        splat_bytes(above, value);
        if x4 + w4 > columns {
            splat_bytes(&mut above[columns.saturating_sub(x4)..], 0);
        }
        let row = y4 % (self.superblock.height_mi() >> ss_y);
        let left = &mut self.left[index][row..row + h4];
        splat_bytes(left, value);
        if y4 + h4 > rows {
            splat_bytes(&mut left[rows.saturating_sub(y4)..], 0);
        }
    }

    /// How the type of a `tx` transform of `plane` is found (§5.11.47,
    /// `compute_tx_type`): luma types are coded unless the set has one
    /// type or the segment's `qindex` is 0; chroma takes `chroma_kind` if its set has it;
    /// lossless blocks are `DCT_DCT` (reconstructed by the WHT).
    pub(super) fn tx_type_source(
        &self,
        plane: PlaneIndex,
        tx: TxSize,
        inter: bool,
        intra_dir: usize,
        chroma_kind: Av1TxType,
    ) -> TxTypeSource {
        if self.block_lossless {
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
        if set == TxSet::DctOnly || !self.block_tx_types_coded {
            return TxTypeSource::Known(Av1TxType::DctDct);
        }
        TxTypeSource::Coded {
            set,
            intra_dir: (!inter).then_some(intra_dir),
        }
    }
    /// The quantizer matrix of a `tx` transform of plane `plane` in the
    /// block being decoded (§7.12.3 `SegQMLevel`): none when the frame
    /// uses none, the plane's level is flat or the block is lossless.
    fn quantizer_matrix(&self, plane: usize, tx: TxSize) -> Option<&'static [u8]> {
        let levels = self.header.qm.filter(|_| !self.block_lossless)?;
        let units = |side_log2: u8| side_log2 - TX_4X4_SIDE_LOG2;
        nc_dsp::quantizer_matrix::matrix(
            levels[plane],
            plane > 0,
            (units(tx.width_log2()), units(tx.height_log2())),
        )
    }
    /// One transform block of `plane` at plane sample (`x`, `y`): its
    /// coefficients (with the neighbour contexts) and inverse transform;
    /// returns its type.
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
        let shift = self.subsampling(plane).1;
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
                steps: self.block_steps[plane_index],
                matrix: self.quantizer_matrix(plane_index, tx),
            };
            coefficients::read(
                &mut self.cdfs.coefficients,
                reader,
                &mut self.coefficients,
                request,
            )?
        };
        self.set_contexts(plane, (x4, y4), (w4, h4), coded.context);
        #[cfg(feature = "analysis")]
        if plane == PlaneIndex::Y && self.maps.stats.is_some() {
            self.transform_record((x, y), tx, coded.eob);
        }
        if coded.eob > 0 {
            let kernels = self.kernels();
            let mut dst = self
                .recon
                .plane_mut(plane)
                .region_mut(x as isize, y as isize, tx.width(), tx.height())
                .ok_or(DecodeError::Malformed("AV1 inverse rectangle"))?;
            if self.block_lossless {
                inverse_wht(kernels, &self.coefficients.coefficients, &mut dst);
            } else {
                kernels.av1_inverse_add(
                    tx.dsp(),
                    coded.kind,
                    &mut self.coefficients.coefficients,
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
fn inverse_wht<T: Pixel>(
    kernels: DepthKernels<'_>,
    coefficients: &[T::Coefficient],
    dst: &mut PlaneViewMut<'_, T>,
) {
    let coefficients: [i32; TX_4X4_COEFFICIENTS] =
        core::array::from_fn(|i| coefficients[i].to_i32());
    kernels.av1_inverse_wht_add(&coefficients, dst);
}

/// An intra block's residual walk: the block, its modes and `skip`, its
/// 4x4 units inside the frame, the 64x64 chunk being decoded (luma 4x4
/// offset) and the end of its decoded luma (`MaxLumaW`, `MaxLumaH`).
struct IntraWalk<'m> {
    block: BlockPos,
    modes: &'m IntraModes,
    skip: bool,
    visible: (usize, usize),
    chunk: (usize, usize),
    max_luma: (usize, usize),
}
