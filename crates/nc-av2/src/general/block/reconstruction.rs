//! An intra or block-copy block's reconstruction (dav2d `recon_b`,
//! `recon_b_luma_tx`, `cfl`): luma transform by transform in partition
//! order (intra prediction from the edge, coefficients, secondary and
//! inverse transforms), then chroma (both planes' coefficients first, then
//! intra prediction or chroma from luma, the cross-chroma and inverse
//! transforms), with the coded-unit masks that say which above-right and
//! below-left samples exist. A block copy is predicted before
//! ([`super::intrabc`]) and takes only the residual here.
use super::Tile;
use super::coefficients::{Residual, TransformInput, tx_type};
use super::context::{self, NO_COEFFICIENTS};
use super::intra::{IntraModes, LUMA_MAP_SIDE, mode};
use super::record::BlockRecord;
use super::size::BlockSize;
use super::transform::{LumaTransforms, TxPartition, TxSize, chroma_max, luma_max};
use nc_common::decode::Result;

/// Blocks larger than 64x64 reconstruct in 64x64 (256: 128x128) pieces.
const PIECE_UNITS: u32 = 16;
const LARGE_PIECE_UNITS: u32 = 32;
/// The nominal angle of each directional mode, by mode - 1 (dav2d
/// `dav2d_mode_to_angle_map`), and the angle step of a delta.
const MODE_ANGLES: [i32; 8] = [90, 180, 45, 135, 113, 157, 203, 67];
const ANGLE_STEP: i32 = 3;
/// Wide-angle thresholds by aspect log ratio 1..=4 (dav2d
/// `wide_angle_remap`).
const WIDE_ANGLE_THRESHOLDS: [i32; 4] = [61, 73, 82, 86];
/// dav2d mode numbers the remap produces.
const DIAG_DOWN_LEFT: u8 = 3;
const HOR_UP: u8 = 7;
const VERT_LEFT: u8 = 8;
/// The angles of the mirror directions.
const HALF_TURN: i32 = 180;
const THREE_QUARTER_TURN: i32 = 270;
/// Chroma coefficient buffers hold up to 64x64 samples per plane, indexed
/// in steps of 16 per chroma 4x4 unit (dav2d `cf_uv`).
const CHROMA_COEFFICIENTS: usize = 64 * 64;
const COEFFICIENTS_PER_UNIT: usize = 16;

/// A remapped directional mode and its full angle (dav2d
/// `wide_angle_remap`): rectangular transforms reach past 45 degrees.
fn wide_angle(tx: TxSize, mode: u8, delta: i8, mrl_index: u8) -> (u8, i32) {
    if !(1..=VERT_LEFT).contains(&mode) {
        return (mode, i32::from(delta));
    }
    let adjust = i32::from(mrl_index == 1) - i32::from(mrl_index == 2);
    let angle = MODE_ANGLES[usize::from(mode - 1)] + i32::from(delta) * ANGLE_STEP + adjust;
    let ratio = i32::from(tx.w) - i32::from(tx.h);
    if ratio > 0 && angle > THREE_QUARTER_TURN - WIDE_ANGLE_THRESHOLDS[(ratio - 1) as usize] {
        return (DIAG_DOWN_LEFT, angle - HALF_TURN);
    }
    if ratio < 0 && angle < WIDE_ANGLE_THRESHOLDS[(-1 - ratio) as usize] {
        return (HOR_UP, angle + HALF_TURN);
    }
    (mode, angle)
}

/// The chroma transforms' decoded residuals of one block (dav2d
/// `chroma_txtp`, `chroma_eob`, `cf_uv`).
#[derive(Debug, Clone)]
pub struct ChromaResiduals {
    pub coefficients: [Vec<i16>; 2],
    pub residuals: Vec<[Option<Residual>; 2]>,
}

impl Default for ChromaResiduals {
    fn default() -> Self {
        Self {
            coefficients: [vec![0; CHROMA_COEFFICIENTS], vec![0; CHROMA_COEFFICIENTS]],
            residuals: Vec::new(),
        }
    }
}

impl Tile<'_, '_> {
    /// Reconstruct an intra or block-copy block (its luma and/or chroma
    /// tree parts).
    pub(super) fn reconstruct(
        &mut self,
        depth: usize,
        record: &BlockRecord,
        modes: &IntraModes,
        transforms: LumaTransforms,
    ) -> Result<()> {
        let size = record.luma.or(record.chroma).unwrap_or(BlockSize::B4x4);
        let (bw4, bh4) = size.units();
        if bw4.max(bh4) > PIECE_UNITS {
            self.reconstruct_pieces(depth, record, modes, transforms)?;
            // The pieces move the current position; the block's siblings
            // start from the block's.
            (self.row, self.column) = (record.row, record.column);
            return Ok(());
        }
        let stages = [record.chroma; 2];
        self.reconstruct_piece(depth, (record, size), modes, transforms, stages)
    }

    /// A block above 64x64 in 64x64 pieces (128x128 for 256 sides): luma
    /// piece by piece; chroma coefficients with the first luma piece of
    /// each chroma piece, its reconstruction with the last (dav2d
    /// `recon_b`'s split).
    fn reconstruct_pieces(
        &mut self,
        depth: usize,
        record: &BlockRecord,
        modes: &IntraModes,
        transforms: LumaTransforms,
    ) -> Result<()> {
        let size = record.luma.or(record.chroma).unwrap_or(BlockSize::B4x4);
        let (bw4, bh4) = size.units();
        let large = bw4.max(bh4) == 4 * PIECE_UNITS;
        let step = if large {
            LARGE_PIECE_UNITS
        } else {
            PIECE_UNITS
        };
        let piece = if large {
            BlockSize::B128x128
        } else {
            BlockSize::B64x64
        };
        let chroma_piece = record.chroma.map(|chroma| {
            if large {
                BlockSize::B128x128
            } else if self.lossless() {
                BlockSize::B64x64
            } else {
                // 4:2:0 pieces of 128-wide or -tall blocks (dav2d `csplit`).
                match chroma {
                    BlockSize::B128x64 => BlockSize::B128x64,
                    BlockSize::B64x128 => BlockSize::B64x128,
                    _ => BlockSize::B128x128,
                }
            }
        });
        let interleaved = large || self.lossless();
        let (row0, column0) = (record.row, record.column);
        let row_end = (row0 + bh4).min(self.geometry.rows);
        let column_end = (column0 + bw4).min(self.geometry.columns);
        let (mut y, mut row) = (0, row0);
        let mut chroma_row = row0;
        while row < row_end {
            let (mut x, mut column) = (0, column0);
            let mut chroma_column = column0;
            while column < column_end {
                let stages = if interleaved {
                    [chroma_piece; 2]
                } else {
                    [
                        chroma_piece.filter(|_| x & 1 == 0 && y & 1 == 0),
                        chroma_piece
                            .filter(|_| column + step >= column_end && row + step >= row_end),
                    ]
                };
                let mut piece_record = *record;
                piece_record.luma = record.luma.map(|_| piece);
                piece_record.chroma = chroma_piece;
                (piece_record.row, piece_record.column) = (row, column);
                (piece_record.chroma_row, piece_record.chroma_column) = (chroma_row, chroma_column);
                self.reconstruct_piece(depth, (&piece_record, size), modes, transforms, stages)?;
                if interleaved {
                    chroma_column += step;
                } else if x & 1 == 1 {
                    chroma_column += step << 1;
                }
                column += step;
                x += 1;
            }
            if interleaved {
                chroma_row += step;
            } else if y & 1 == 1 {
                chroma_row += step << 1;
            }
            row += step;
            y += 1;
        }
        Ok(())
    }

    /// One piece of at most 64x64 of a block of size `whole`: luma
    /// transforms, then chroma in its stages (coefficients,
    /// reconstruction).
    fn reconstruct_piece(
        &mut self,
        depth: usize,
        (record, whole): (&BlockRecord, BlockSize),
        modes: &IntraModes,
        transforms: LumaTransforms,
        stages: [Option<BlockSize>; 2],
    ) -> Result<()> {
        if let Some(size) = record.luma {
            if let Some(prediction) = self.inter_prediction {
                self.predict_inter_luma(&prediction, size);
            }
            self.luma_transforms(depth, record, (size, whole), modes, transforms)?;
        }
        if let Some(chroma) = stages[0].or(stages[1]) {
            self.chroma(depth, record, chroma, modes, stages)?;
        }
        Ok(())
    }

    /// Every luma transform of a block in partition order (dav2d
    /// `recon_b`'s `tx_part` switch): parts past the frame's bottom or
    /// right edge are skipped.
    fn luma_transforms(
        &mut self,
        depth: usize,
        record: &BlockRecord,
        (size, whole): (BlockSize, BlockSize),
        modes: &IntraModes,
        transforms: LumaTransforms,
    ) -> Result<()> {
        let (row0, column0) = (record.row, record.column);
        if modes.palette.size != 0 {
            self.predict_palette(size, &modes.palette);
        }
        let (bw4, bh4) = size.units();
        let (w4, h4) = (
            bw4.min(self.geometry.columns - column0),
            bh4.min(self.geometry.rows - row0),
        );
        let mut parts: Vec<(u32, u32, TxSize)> = Vec::new();
        let partition = transforms.partition();
        match transforms {
            LumaTransforms::Lossless { large } => {
                let tx = if large {
                    luma_max(size, true)
                } else {
                    TxSize::TX4
                };
                let (tw4, th4) = tx.units();
                for y in (0..h4).step_by(th4 as usize) {
                    for x in (0..w4).step_by(tw4 as usize) {
                        parts.push((y, x, tx));
                    }
                }
            }
            LumaTransforms::Partition(partition) => {
                let tx = partition.tx_size(size).unwrap_or(TxSize::TX4);
                parts = partition_parts(partition, size, tx, (w4, h4));
            }
        }
        for (y, x, tx) in parts {
            (self.row, self.column) = (row0 + y, column0 + x);
            self.luma_transform(depth, (record, whole), modes, tx, partition)?;
        }
        (self.row, self.column) = (row0, column0);
        Ok(())
    }

    /// One luma transform of a block of size `whole`: prediction,
    /// coefficients, inverse transforms (dav2d `recon_b_luma_tx`).
    fn luma_transform(
        &mut self,
        depth: usize,
        (record, whole): (&BlockRecord, BlockSize),
        modes: &IntraModes,
        tx: TxSize,
        partition: TxPartition,
    ) -> Result<()> {
        let (y_mode, angle) = wide_angle(tx, modes.y_mode, modes.y_angle, modes.mrl_index);
        let column = self.column as usize;
        let left = context::left_index(self.row);
        let (tw4, th4) = tx.units();
        let residual = if record.skip_txfm {
            None
        } else {
            let input = TransformInput {
                plane: 0,
                tx,
                // The whole block's size, not its 64x64 piece's (dav2d
                // `get_skip_ctx` takes the block's).
                block: whole,
                intra: record.intra && !record.intrabc,
                fsc: modes.fsc,
                luma_fsc: modes.fsc,
                y_mode,
                uv_mode: modes.uv_mode,
                u_has_coefficients: false,
                luma_type: 0,
                dequant: self.dequant[0],
            };
            let above = self.contexts.above.luma_coefficients[column..].to_vec();
            let left_edge = self.contexts.left.luma_coefficients[left..].to_vec();
            let mut coefficients = std::mem::take(&mut self.luma_coefficients);
            let residual =
                self.coefficients(depth + 1, &input, (&above, &left_edge), &mut coefficients);
            self.luma_coefficients = coefficients;
            let residual = residual?;
            let range = self.reader.range();
            let (tw, th) = tx.samples();
            trace!(
                self,
                depth + 1,
                "Post-y_cf_blk[tx={tw}x{th},txtp={}/{},eob={}]: r={range}",
                one_d(residual.tx_type & 7),
                one_d((residual.tx_type >> 5) & 7),
                residual.eob.map_or(-1, i32::from)
            );
            Some(residual)
        };
        let context = residual.map_or(NO_COEFFICIENTS, |r| r.context);
        let w = (tw4 as usize).min((self.geometry.columns - self.column) as usize);
        let h = (th4 as usize).min((self.geometry.rows - self.row) as usize);
        if record.skip_txfm {
            self.contexts.above.luma_coefficients[column..column + tw4 as usize].fill(context);
            self.contexts.left.luma_coefficients[left..left + th4 as usize].fill(context);
        } else {
            self.contexts.above.luma_coefficients[column..column + w].fill(context);
            self.contexts.left.luma_coefficients[left..left + h].fill(context);
        }
        // Skipped residuals leave the map as it was (dav2d
        // `recon_b_luma_tx`).
        if let Some(residual) = residual {
            let tx_type = residual.tx_type & tx_type::TYPE_MASK;
            let map_row = self.row as usize & (LUMA_MAP_SIDE - 1);
            let map_column = self.column as usize & (LUMA_MAP_SIDE - 1);
            for y in map_row..(map_row + th4 as usize).min(LUMA_MAP_SIDE) {
                let line = y * LUMA_MAP_SIDE;
                let end = (map_column + tw4 as usize).min(LUMA_MAP_SIDE);
                self.luma_type_map[line + map_column..line + end].fill(tx_type as u8);
            }
        }
        if record.intra && !record.intrabc && modes.palette.size == 0 {
            self.predict_luma(record, modes, tx, (y_mode, angle), partition);
        }
        if let Some(residual) = residual
            && let Some(eob) = residual.eob
        {
            self.add_luma_residual(record, modes, (tx, y_mode), residual, eob);
            self.filters
                .mark_residual((self.row, self.column), (w as u32, h as u32));
        }
        self.mark_coded(
            0,
            (self.row & 63) as usize,
            (self.column & 63) as usize,
            tw4,
            th4,
        );
        Ok(())
    }

    /// Mark `w` x `h` units at (`row`, `column`) of a plane type's
    /// superblock mask as coded.
    fn mark_coded(&mut self, plane: usize, row: usize, column: usize, w: u32, h: u32) {
        let mask = ((1_u64 << w) - 1) << column;
        for line in &mut self.coded[plane][row..(row + h as usize).min(64)] {
            *line |= mask;
        }
    }

    /// A block's chroma: coefficients of both planes (stage 0), then
    /// prediction and inverse transforms (stage 1) (dav2d `recon_b`'s
    /// chroma part).
    #[allow(clippy::too_many_lines, reason = "dav2d's chroma walk")]
    fn chroma(
        &mut self,
        depth: usize,
        record: &BlockRecord,
        chroma: BlockSize,
        modes: &IntraModes,
        stages: [Option<BlockSize>; 2],
    ) -> Result<()> {
        let (cbw4, cbh4) = chroma.units();
        let (cbx, cby) = (record.chroma_column, record.chroma_row);
        let cw4 = (self.geometry.columns - cbx).min(cbw4);
        let ch4 = (self.geometry.rows - cby).min(cbh4);
        let (cbw4ss, cw4ss, ch4ss) = (cbw4.div_ceil(2), cw4.div_ceil(2), ch4.div_ceil(2));
        let uvtx = if self.lossless() {
            TxSize::TX4
        } else {
            chroma_max(chroma)
        };
        let (utw4, uth4) = uvtx.units();
        let ctw4 = utw4.min((self.geometry.columns - cbx).div_ceil(2));
        let cth4 = uth4.min((self.geometry.rows - cby).div_ceil(2));
        let sdp_active = record.luma.is_none();
        let intra = record.intra && (sdp_active || !record.intrabc);
        let skip = !sdp_active && record.skip_txfm;
        let (uv_mode, angle) = if intra {
            wide_angle(uvtx, modes.uv_mode, modes.uv_angle, 0)
        } else {
            (modes.uv_mode, i32::from(modes.uv_angle))
        };
        let (above, left) = Self::chroma_unit_indices_at(cby, cbx);
        if stages[0].is_some() {
            if skip {
                for plane in 0..2 {
                    let end = above + cw4ss as usize;
                    self.contexts.above.chroma_coefficients[plane][above..end]
                        .fill(NO_COEFFICIENTS);
                    let end = left + ch4ss as usize;
                    self.contexts.left.chroma_coefficients[plane][left..end].fill(NO_COEFFICIENTS);
                }
            } else {
                let map = (cby as usize & (LUMA_MAP_SIDE - 1)) * LUMA_MAP_SIDE
                    + (cbx as usize & (LUMA_MAP_SIDE - 1));
                let luma_fsc = if sdp_active {
                    self.luma_fsc_map[map] != 0
                } else {
                    modes.fsc
                };
                let mut store = std::mem::take(&mut self.chroma_residuals);
                store.residuals.clear();
                store
                    .residuals
                    .resize((cbw4ss * cbh4.div_ceil(2)) as usize, [None; 2]);
                let (row0, column0) = (self.row, self.column);
                // The luma type at each chroma transform, or, where the
                // chroma block covers several luma blocks, the current
                // luma block's (dav2d `recon_b`'s `y_txtp`).
                let follows_luma = record.luma == Some(chroma);
                for plane in 0..2 {
                    for y in (0..ch4ss).step_by(uth4 as usize) {
                        for x in (0..cw4ss).step_by(utw4 as usize) {
                            let i = (y * cbw4ss + x) as usize;
                            (self.row, self.column) = (row0 + (y << 1), column0 + (x << 1));
                            let (map_row, map_column) = if follows_luma {
                                (self.row, self.column)
                            } else {
                                (row0, column0)
                            };
                            let luma_map = (map_row as usize & (LUMA_MAP_SIDE - 1)) * LUMA_MAP_SIDE
                                + (map_column as usize & (LUMA_MAP_SIDE - 1));
                            let input = TransformInput {
                                plane: plane + 1,
                                tx: uvtx,
                                block: chroma,
                                intra,
                                fsc: modes.fsc,
                                luma_fsc,
                                y_mode: modes.y_mode,
                                uv_mode,
                                u_has_coefficients: self.u_has_coefficients,
                                luma_type: u16::from(self.luma_type_map[luma_map]),
                                dequant: self.dequant[plane + 1],
                            };
                            let (a, l) = (above + x as usize, left + y as usize);
                            let above_edge =
                                self.contexts.above.chroma_coefficients[plane][a..].to_vec();
                            let left_edge =
                                self.contexts.left.chroma_coefficients[plane][l..].to_vec();
                            let offset = i * COEFFICIENTS_PER_UNIT;
                            let residual = self.coefficients(
                                depth + 1,
                                &input,
                                (&above_edge, &left_edge),
                                &mut store.coefficients[plane][offset..],
                            )?;
                            if plane == 0 {
                                self.u_has_coefficients = residual.eob.is_some();
                            }
                            let range = self.reader.range();
                            let (tw, th) = uvtx.samples();
                            trace!(
                                self,
                                depth + 1,
                                "Post-{}_cf_blk[tx={tw}x{th},txtp={}/{},eob={}]: r={range}",
                                ["u", "v"][plane],
                                one_d(residual.tx_type & 7),
                                one_d((residual.tx_type >> 5) & 7),
                                residual.eob.map_or(-1, i32::from)
                            );
                            store.residuals[i][plane] = Some(residual);
                            self.contexts.above.chroma_coefficients[plane][a..a + ctw4 as usize]
                                .fill(residual.context);
                            self.contexts.left.chroma_coefficients[plane][l..l + cth4 as usize]
                                .fill(residual.context);
                        }
                    }
                }
                (self.row, self.column) = (row0, column0);
                self.chroma_residuals = store;
            }
            if stages[1].is_none() {
                return Ok(());
            }
        }
        let (cbx4, cby4) = ((cbx & 63) as usize >> 1, (cby & 63) as usize >> 1);
        if let Some(prediction) = self.inter_prediction.filter(|_| !sdp_active) {
            let (row, column) = (self.chroma_row, self.chroma_column);
            (self.chroma_row, self.chroma_column) = (cby, cbx);
            self.predict_inter_chroma(&prediction, chroma, record.luma);
            (self.chroma_row, self.chroma_column) = (row, column);
        }
        if skip {
            self.mark_coded(1, cby4, cbx4, cw4ss, ch4ss);
            return Ok(());
        }
        if intra && uv_mode == mode::CFL {
            self.predict_cfl(record, chroma, modes, uvtx);
        }
        for y in (0..ch4ss).step_by(uth4 as usize) {
            for x in (0..cw4ss).step_by(utw4 as usize) {
                let i = (y * cbw4ss + x) as usize;
                if intra && uv_mode != mode::CFL {
                    for plane in 1..=2 {
                        self.predict_chroma(record, plane, (x, y), uvtx, (uv_mode, angle), modes);
                    }
                }
                self.add_chroma_residuals(record, modes, (x, y), uvtx, i, intra);
                let (ctw4, cth4) = (
                    utw4.min((self.geometry.columns - cbx).div_ceil(2)),
                    uth4.min((self.geometry.rows - cby).div_ceil(2)),
                );
                self.mark_coded(1, cby4 + y as usize, cbx4 + x as usize, ctw4, cth4);
            }
        }
        Ok(())
    }
}

/// The transforms of a partition as (row, column, size) offsets in units,
/// in coding order, without those starting past the frame's edge
/// (`visible` units of the block remain inside it).
fn partition_parts(
    partition: TxPartition,
    size: BlockSize,
    tx: TxSize,
    (w4, h4): (u32, u32),
) -> Vec<(u32, u32, TxSize)> {
    let (tw4, th4) = tx.units();
    let mut parts = Vec::with_capacity(4);
    let inside = |y: u32, x: u32| y < h4 && x < w4;
    match partition {
        TxPartition::None => parts.push((0, 0, tx)),
        TxPartition::Split => {
            for (y, x) in [(0, 0), (0, tw4), (th4, 0), (th4, tw4)] {
                if inside(y, x) {
                    parts.push((y, x, tx));
                }
            }
        }
        TxPartition::Horizontal | TxPartition::Horizontal4 => {
            for k in 0..(if partition == TxPartition::Horizontal {
                2
            } else {
                4
            }) {
                if !inside(k * th4, 0) {
                    break;
                }
                parts.push((k * th4, 0, tx));
            }
        }
        TxPartition::Vertical | TxPartition::Vertical4 => {
            for k in 0..(if partition == TxPartition::Vertical {
                2
            } else {
                4
            }) {
                if !inside(0, k * tw4) {
                    break;
                }
                parts.push((0, k * tw4, tx));
            }
        }
        TxPartition::Horizontal5 => {
            let big = TxPartition::Horizontal.tx_size(size).unwrap_or(tx);
            let (_, bh4) = big.units();
            parts.push((0, 0, tx));
            if inside(0, tw4) {
                parts.push((0, tw4, tx));
            }
            if inside(th4, 0) {
                parts.push((th4, 0, big));
                if inside(th4 + bh4, 0) {
                    parts.push((th4 + bh4, 0, tx));
                    if inside(0, tw4) {
                        parts.push((th4 + bh4, tw4, tx));
                    }
                }
            }
        }
        TxPartition::Vertical5 => {
            let big = TxPartition::Vertical.tx_size(size).unwrap_or(tx);
            let (bw4, _) = big.units();
            parts.push((0, 0, tx));
            if inside(th4, 0) {
                parts.push((th4, 0, tx));
            }
            if inside(0, tw4) {
                parts.push((0, tw4, big));
                if inside(0, tw4 + bw4) {
                    parts.push((0, tw4 + bw4, tx));
                    if inside(th4, 0) {
                        parts.push((th4, tw4 + bw4, tx));
                    }
                }
            }
        }
    }
    parts
}

/// dav2d `dav2d_tx1d_names` (for traces).
fn one_d(kind: u16) -> &'static str {
    [
        "dct", "identity", "adst", "flipadst", "ddt", "flipddt", "wht", "?",
    ][usize::from(kind & 7)]
}
