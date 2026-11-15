//! One coding block's syntax up to its tool group (dav2d `decode_b`, AVM
//! `read_intra_frame_mode_info` / `read_inter_frame_mode_info`): skip
//! mode, intra or inter, intra block copy, the residual skip flag, the
//! loop-filter block flags (GDF, the CDEF strength set, CCSO) and the
//! superblock quantizer delta; then the block goes to the intra, intra
//! block copy or inter group, each of which answers `Unsupported` until it
//! is built.
use super::record::BlockRecord;
use super::size::BlockSize;
use super::{Tile, UNITS64_LOG2, context, samples};
use crate::general::frame_header::filters::Cdef;
use nc_common::decode::{DecodeError, Result};

/// Planes of 4:2:0.
const PLANES: usize = 3;
/// Blocks of fewer than this many 4x4 units have no skip mode (4x4, 4x8).
const SKIP_MODE_MIN_UNITS: u32 = 3;
/// Intra block copy needs a side below 16 units (64 samples).
const INTRABC_MAX_SIDE: u32 = 16;
/// Skip-mode blocks' residual skip contexts follow the others'.
const SKIP_MODE_CONTEXT_OFFSET: usize = 3;
/// The superblock quantizer delta's escape symbol, the bits of its escape
/// length (minus one), and the escape values' base.
const DELTA_Q_ESCAPE: usize = 7;
const DELTA_Q_LENGTH_BITS: u32 = 3;
const DELTA_Q_ESCAPE_BASE: i32 = 5;
/// Superblock quantizer range of 8-bit streams.
const MIN_QINDEX: i32 = 1;
const MAX_QINDEX: i32 = 255;

/// A context neighbour: an edge and its index there.
#[derive(Debug, Clone, Copy)]
pub(super) enum Neighbour {
    Above(usize),
    Left(usize),
}

/// Up to two context neighbours and how many count.
#[derive(Debug, Clone, Copy)]
struct Neighbours {
    at: [Option<Neighbour>; 2],
    count: usize,
}

impl Tile<'_, '_> {
    /// Decode one coding block: `luma` and `chroma` are the sizes of the
    /// trees it belongs to (one may be absent: separate trees).
    pub(super) fn block(
        &mut self,
        depth: usize,
        luma: Option<BlockSize>,
        chroma: Option<BlockSize>,
    ) -> Result<()> {
        let size = luma
            .or(chroma)
            .ok_or(DecodeError::Malformed("AV2 block without a tree"))?;
        let (bw4, bh4) = size.units();
        let range = self.reader.range();
        let plane = match (luma, chroma) {
            (Some(_), None) => "y",
            (None, Some(_)) => "uv",
            _ => "yuv",
        };
        trace!(
            self,
            depth - 1,
            "decode_b[y={},x={},bs={}x{},plane={plane}]: r={range}",
            self.row,
            self.column,
            samples(bw4),
            samples(bh4)
        );
        if self.frame.segmentation.is_some() {
            return Err(DecodeError::Unsupported("AV2 segment ids"));
        }
        let has_luma = luma.is_some();
        let neighbours = self.context_neighbours(size);
        let skip_mode = self.skip_mode(depth, size, neighbours)?;
        let intra = if skip_mode {
            false
        } else if !self.frame.is_intra() && !self.intra_region {
            // Mixed regions whose chroma block is offset are inter.
            luma == chroma && {
                let context = self.intra_context(neighbours);
                let inter = self.bool(|m| &mut m.is_inter[context])?;
                let range = self.reader.range();
                trace!(
                    self,
                    depth,
                    "Post-is_inter[ctx={context},{}]: r={range}",
                    u8::from(inter)
                );
                !inter
            }
        } else {
            true
        };
        let intrabc = has_luma && self.intrabc(depth, size, intra)?;
        let skip_txfm = self.skip_txfm(depth, (intra, intrabc, skip_mode), neighbours)?;
        if has_luma {
            self.gdf_flag(depth)?;
        }
        self.cdef_index(depth, size, has_luma && skip_txfm)?;
        if has_luma {
            self.ccso_flags(depth)?;
            self.delta_q(depth, size, skip_txfm)?;
        }
        let record = BlockRecord {
            luma,
            chroma,
            row: self.row,
            column: self.column,
            chroma_row: self.chroma_row,
            chroma_column: self.chroma_column,
            skip_mode,
            intra,
            intrabc,
            skip_txfm,
            qindex: self.qindex,
        };
        if intrabc {
            self.intrabc_block(depth, &record)?;
        } else if intra {
            self.intra_block(depth, &record)?;
        } else {
            self.inter_block(depth, &record)?;
        }
        self.record_contexts(&record);
        Ok(())
    }

    /// An intra block-copy block (the intra group).
    #[allow(clippy::unused_self, reason = "the tool group fills it in")]
    fn intrabc_block(&mut self, _depth: usize, _record: &BlockRecord) -> Result<()> {
        Err(DecodeError::Unsupported("AV2 intra block copy blocks"))
    }

    /// An inter block's references, motion, prediction and residual (the
    /// inter and residual groups).
    #[allow(clippy::unused_self, reason = "the tool group fills it in")]
    fn inter_block(&mut self, _depth: usize, _record: &BlockRecord) -> Result<()> {
        Err(DecodeError::Unsupported(
            "AV2 inter blocks (general block layer)",
        ))
    }

    /// Record the block's common flags in the above and left contexts over
    /// its extent (dav2d `decode_b`'s `case_set` updates).
    fn record_contexts(&mut self, record: &BlockRecord) {
        let Some(size) = record.luma else {
            return;
        };
        let (bw4, bh4) = size.units();
        let column = record.column as usize;
        let row = context::left_index(record.row);
        let flags = [
            u8::from(record.skip_txfm),
            u8::from(record.skip_mode),
            u8::from(record.intra),
            u8::from(record.intrabc),
        ];
        for (edge, start, count) in [
            (&mut self.contexts.above, column, bw4),
            (&mut self.contexts.left, row, bh4),
        ] {
            let end = (start + count as usize).min(edge.intra.len());
            for (field, value) in [
                (&mut edge.skip_txfm, flags[0]),
                (&mut edge.skip_mode, flags[1]),
                (&mut edge.intra, flags[2]),
                (&mut edge.intrabc, flags[3]),
            ] {
                field[start..end].fill(value);
            }
            // Intra blocks record their palette with their modes.
            if !record.intra || record.intrabc {
                edge.palette_size[start..end].fill(0);
            }
        }
    }

    /// The neighbours of skip-mode, intra and residual-skip contexts: left
    /// then above, at the block's far edges when they lie inside the tile,
    /// else at its near edges (dav2d `decode_b` `nx`, `xoff`).
    fn context_neighbours(&self, size: BlockSize) -> Neighbours {
        let (bw4, bh4) = size.units();
        let have_left = self.column > self.bounds.column_start;
        let have_top = self.row > self.bounds.row_start;
        let left = context::left_index(self.row);
        let column = self.column as usize;
        let mut at = [None; 2];
        let mut count = 0;
        if have_left && self.row + bh4 <= self.bounds.row_end {
            at[count] = Some(Neighbour::Left(left + bh4 as usize - 1));
            count += 1;
        }
        if have_top && self.column + bw4 <= self.bounds.column_end {
            at[count] = Some(Neighbour::Above(column + bw4 as usize - 1));
            count += 1;
        }
        if count < 2 && have_left {
            at[count] = Some(Neighbour::Left(left));
            count += 1;
        }
        if count < 2 {
            at[count] = Some(Neighbour::Above(column));
            if count == 0 {
                at[1] = Some(Neighbour::Above(column));
            }
            count += usize::from(have_top);
        }
        Neighbours { at, count }
    }

    /// A context field of a neighbour.
    pub(super) fn neighbour_value(
        &self,
        neighbour: Neighbour,
        field: fn(&context::Edge) -> &Vec<u8>,
    ) -> u8 {
        match neighbour {
            Neighbour::Above(index) => field(&self.contexts.above)[index],
            Neighbour::Left(index) => field(&self.contexts.left)[index],
        }
    }

    /// The sum of a field over the counted neighbours.
    fn neighbour_sum(
        &self,
        neighbours: Neighbours,
        field: fn(&context::Edge) -> &Vec<u8>,
    ) -> usize {
        neighbours.at[..neighbours.count]
            .iter()
            .flatten()
            .map(|&neighbour| usize::from(self.neighbour_value(neighbour, field)))
            .sum()
    }

    /// `skip_mode` of a block in a skip-mode frame.
    fn skip_mode(&mut self, depth: usize, size: BlockSize, neighbours: Neighbours) -> Result<bool> {
        let (bw4, bh4) = size.units();
        if !self.frame.skip_mode || bw4 * bh4 < SKIP_MODE_MIN_UNITS || self.intra_region {
            return Ok(false);
        }
        let context = self.neighbour_sum(neighbours, |edge| &edge.skip_mode);
        let skip_mode = self.bool(|m| &mut m.skip_mode[context])?;
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-skip_mode[ctx={context},{}]: r={range}",
            u8::from(skip_mode)
        );
        Ok(skip_mode)
    }

    /// The intra context: how many neighbours are intra, not block copy
    /// (dav2d `get_intra_ctx`).
    fn intra_context(&self, neighbours: Neighbours) -> usize {
        let count = neighbours.count;
        if count == 0 {
            return 0;
        }
        let intra = |neighbour: Option<Neighbour>| {
            neighbour.is_some_and(|neighbour| {
                self.neighbour_value(neighbour, |edge| &edge.intra) != 0
                    && self.neighbour_value(neighbour, |edge| &edge.intrabc) == 0
            })
        };
        let sum =
            usize::from(intra(neighbours.at[0])) + usize::from(intra(neighbours.at[count - 1]));
        sum + usize::from(sum == count)
    }

    /// `use_intrabc` of an intra luma block of a frame that allows it.
    fn intrabc(&mut self, depth: usize, size: BlockSize, intra: bool) -> Result<bool> {
        let (bw4, bh4) = size.units();
        if self.frame.intra_block_copy.is_none()
            || bw4.min(bh4) >= INTRABC_MAX_SIDE
            || !intra
            || self.intra_region
        {
            return Ok(false);
        }
        let context = self
            .spatial_neighbours(size)
            .iter()
            .flatten()
            .map(|&neighbour| usize::from(self.neighbour_value(neighbour, |edge| &edge.intrabc)))
            .sum::<usize>();
        let intrabc = self.bool(|m| &mut m.intrabc[context])?;
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-intrabc[ctx={context},{}]: r={range}",
            u8::from(intrabc)
        );
        Ok(intrabc)
    }

    /// The spatial neighbours of luma contexts, not across the superblock
    /// top: left then above at the far edges when the block lies inside
    /// the frame, else at the near edges (dav2d `decode_b` `nb`, `boff`).
    pub(super) fn spatial_neighbours(&self, size: BlockSize) -> [Option<Neighbour>; 2] {
        let (bw4, bh4) = size.units();
        let w4 = bw4.min(self.geometry.columns - self.column);
        let h4 = bh4.min(self.geometry.rows - self.row);
        let have_left = self.column > self.bounds.column_start;
        let have_top_in_superblock = self.row & (self.geometry.superblock_units - 1) != 0;
        let left = context::left_index(self.row);
        let column = self.column as usize;
        let mut at = [None; 2];
        let mut count = 0;
        if have_left && bh4 == h4 {
            at[count] = Some(Neighbour::Left(left + bh4 as usize - 1));
            count += 1;
        }
        if have_top_in_superblock && bw4 == w4 {
            at[count] = Some(Neighbour::Above(column + bw4 as usize - 1));
            count += 1;
        }
        if have_left && count < 2 {
            at[count] = Some(Neighbour::Left(left));
            count += 1;
        }
        if have_top_in_superblock && count < 2 {
            at[count] = Some(Neighbour::Above(column));
        }
        at
    }

    /// `skip_txfm`: implied zero for intra blocks (their residual skip is
    /// per transform), read for inter and block-copy blocks.
    fn skip_txfm(
        &mut self,
        depth: usize,
        (intra, intrabc, skip_mode): (bool, bool, bool),
        neighbours: Neighbours,
    ) -> Result<bool> {
        if intra && !intrabc {
            return Ok(false);
        }
        let context = self.neighbour_sum(neighbours, |edge| &edge.skip_txfm)
            + usize::from(skip_mode) * SKIP_MODE_CONTEXT_OFFSET;
        let skip = self.bool(|m| &mut m.skip_txfm[context])?;
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-skip_txfm[ctx={context},{}]: r={range}",
            u8::from(skip)
        );
        Ok(skip)
    }

    /// The GDF flag of the filter unit a block starts (dav2d `decode_b`
    /// "gdf"): read per unit in frames that choose per unit.
    fn gdf_flag(&mut self, depth: usize) -> Result<()> {
        let unit64 = u32::from(self.frame.gdf.block64);
        let unit = unit64 << UNITS64_LOG2;
        if (self.row | self.column) & (unit - 1) != 0 {
            return Ok(());
        }
        let width = self.frame.width.max(self.frame.height);
        let flag = if self.frame.gdf.mode == 2 && width > samples(unit) {
            let flag = self.bool(|m| &mut m.gdf)?;
            let range = self.reader.range();
            trace!(
                self,
                depth,
                "Post-gdf[y={},x={},gdf={}]: r={range}",
                self.row,
                self.column,
                u8::from(flag)
            );
            flag
        } else {
            self.frame.gdf.mode != 0
        };
        self.splat_filter_flag(unit64, |tile, index| tile.gdf_flags[index] = flag);
        Ok(())
    }

    /// Apply `set` to every 64x64 filter block of the `unit64`-sided unit
    /// at the current position.
    fn splat_filter_flag(&mut self, unit64: u32, set: impl Fn(&mut Self, usize)) {
        for y in 0..unit64 {
            for x in 0..unit64 {
                let (row, column) = (
                    self.row + (y << UNITS64_LOG2),
                    self.column + (x << UNITS64_LOG2),
                );
                if row < self.geometry.rows && column < self.geometry.columns {
                    let index = self.filter_block(row, column);
                    set(self, index);
                }
            }
        }
    }

    /// The CDEF strength set of the 64x64 filter block the block starts,
    /// read by its first block with residual (or any block when the frame
    /// filters skipped blocks too).
    fn cdef_index(&mut self, depth: usize, size: BlockSize, skipped: bool) -> Result<()> {
        let Some(cdef) = self.frame.cdef else {
            return Ok(());
        };
        if skipped && !cdef.on_skipped {
            return Ok(());
        }
        let index = self.filter_block(self.row, self.column);
        if self.cdef_index[index].is_some() {
            return Ok(());
        }
        let value = if cdef.strengths == 1 {
            0
        } else {
            self.read_cdef_index(depth, &cdef, index)?
        };
        // Blocks larger than 64x64 cover several filter blocks.
        let (bw4, bh4) = size.units();
        let blocks = |units: u32| (units >> UNITS64_LOG2).max(1);
        for y in 0..blocks(bh4) {
            for x in 0..blocks(bw4) {
                let (row, column) = (
                    self.row + (y << UNITS64_LOG2),
                    self.column + (x << UNITS64_LOG2),
                );
                if row < self.geometry.rows && column < self.geometry.columns {
                    let block = self.filter_block(row, column);
                    self.cdef_index[block] = Some(value);
                }
            }
        }
        Ok(())
    }

    /// Read a CDEF strength set: 0 or not with a neighbour context, then
    /// which of the others.
    fn read_cdef_index(&mut self, depth: usize, cdef: &Cdef, index: usize) -> Result<u8> {
        let unit = 1 << UNITS64_LOG2;
        let stride = (self.geometry.columns as usize).div_ceil(unit as usize);
        let left = if self.column < self.bounds.column_start + unit {
            None
        } else {
            Some(self.cdef_index[index - 1])
        };
        // Above only inside the superblock.
        let top = if (self.row & !(unit - 1)) & (self.geometry.superblock_units - 1) == 0 {
            None
        } else {
            Some(self.cdef_index[index - stride])
        };
        let zero = |value: Option<Option<u8>>| usize::from(value == Some(Some(0)));
        // Coded (Some(Some)) neighbours on both sides, else the one coded.
        let context = match (left, top) {
            (Some(Some(_)), Some(Some(_))) => {
                let context = zero(left) + zero(top);
                context + usize::from(context == 2)
            }
            _ => 2 * usize::from(left.flatten().or(top.flatten()) == Some(0)),
        };
        let value = if self.bool(|m| &mut m.cdef_index0[context])? {
            0
        } else if cdef.strengths == 2 {
            1
        } else {
            1 + match cdef.strengths {
                3 => self.symbol(|m| &mut m.cdef_index3)?,
                4 => self.symbol(|m| &mut m.cdef_index4)?,
                5 => self.symbol(|m| &mut m.cdef_index5)?,
                6 => self.symbol(|m| &mut m.cdef_index6)?,
                7 => self.symbol(|m| &mut m.cdef_index7)?,
                _ => self.symbol(|m| &mut m.cdef_index8)?,
            } as u8
        };
        let range = self.reader.range();
        trace!(
            self,
            depth, "Post-cdef_idx[ctx={context},{value}]: r={range}"
        );
        Ok(value)
    }

    /// The CCSO flags of the filter unit a luma block starts, per plane
    /// the frame filters.
    fn ccso_flags(&mut self, depth: usize) -> Result<()> {
        let unit64 = u32::from(self.frame.ccso.block64);
        let unit = unit64 << UNITS64_LOG2;
        if (self.row | self.column) & (unit - 1) != 0 {
            return Ok(());
        }
        for plane in 0..PLANES {
            let coded = self.frame.ccso.planes[plane];
            if !coded.enabled {
                continue;
            }
            if coded.block_reuse {
                return Err(DecodeError::Unsupported("AV2 CCSO block flags reuse"));
            }
            let left = if self.column == self.bounds.column_start {
                false
            } else {
                let index = self.filter_block(self.row, self.column - (1 << UNITS64_LOG2));
                self.ccso_flags[index][plane]
            };
            let context = usize::from(left) * 2;
            let flag = self.bool(|m| &mut m.ccso[plane][context])?;
            let range = self.reader.range();
            let name = ["y", "u", "v"][plane];
            trace!(
                self,
                depth,
                "Post-ccso[pl={name},ctx={context},{}]: r={range}",
                u8::from(flag)
            );
            self.splat_filter_flag(unit64, |tile, index| tile.ccso_flags[index][plane] = flag);
        }
        Ok(())
    }

    /// The superblock quantizer delta, read by a superblock's first block
    /// (unless it is the whole superblock and skips its residual).
    fn delta_q(&mut self, depth: usize, size: BlockSize, skip_txfm: bool) -> Result<()> {
        let superblock = self.geometry.superblock_units;
        if (self.row | self.column) & (superblock - 1) != 0 {
            return Ok(());
        }
        let Some(res_log2) = self.frame.delta_q else {
            return Ok(());
        };
        if size == self.geometry.root && skip_txfm {
            return Ok(());
        }
        let mut delta = self.symbol(|m| &mut m.delta_q)? as i32;
        if delta == DELTA_Q_ESCAPE as i32 {
            let bits = 1 + self.reader.literal(DELTA_Q_LENGTH_BITS)?;
            delta = self.reader.literal(bits)? as i32 + DELTA_Q_ESCAPE_BASE + (1 << bits);
        }
        if delta != 0 && self.bypass()? {
            delta = -delta;
        }
        let scaled = delta << res_log2;
        let previous = self.qindex;
        self.qindex = (i32::from(self.qindex) + scaled).clamp(MIN_QINDEX, MAX_QINDEX) as u16;
        if self.qindex != previous {
            self.dequant = self.dequantizers(self.qindex);
        }
        let range = self.reader.range();
        let qindex = self.qindex;
        trace!(self, depth, "Post-delta_q[{delta}->{qindex}]: r={range}");
        Ok(())
    }
}
