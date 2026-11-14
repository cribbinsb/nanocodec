//! Tile decoding: the tiles of a frame, the partition walk, block syntax,
//! prediction and residuals of each (AV1 §5.11, §7.11–7.13), and the loop
//! filters run on the result. Intra blocks (`intra`, `residual`) and inter
//! blocks (`inter`, `inter_pred`, `inter_residual`) of every size are
//! decoded. Frame headers outside the
//! subset are rejected earlier by `subset`.
use super::availability::{DecodedUnits, TileBounds};
use super::block_info::{BlockContexts, BlockInfo, ChromaSmooth, Neighbours, TxContexts};
use super::block_size::{BlockSize, MI_SIZE_LOG2};
use super::cdef::{self, CdefMap};
use super::cdf::Cdfs;
use super::coefficients::{CoefficientBuffer, Steps};
use super::deblock::{self, EdgeBlock, EdgeMap, LevelTable};
use super::frame_header::{INTRA_FRAME, TOTAL_REFS, TileInfo};
use super::motion::{MotionCell, MotionGrid};
use super::motion_field::MAX_PROJECTION_DISTANCE;
use super::motion_field::{SavedMotion, TemporalField};
use super::mv_stack::MvFrame;
use super::subset::{FrameParameters, SegmentQuantizers};
use crate::backend::{Dsp, FrameKind, HORZ_ALIKE, Picture, PlaneIndex, VERT_ALIKE, gather};
use crate::decoder_range::SymbolReader;
use crate::segment_id::{MAX_SEGMENTS, SegmentMap, neg_deinterleave};
use crate::tables::Partition;
use crate::tables::{AC_QLOOKUP, DC_QLOOKUP};
use nc_common::decode::DecodeError;
use std::rc::Rc;

// The analysis records (feature `analysis`); every hook below is compiled
// out without it.
#[cfg(feature = "analysis")]
mod analysis;
mod inter;
mod inter_pred;
mod inter_residual;
mod intra;
mod intrabc;
mod lr_units;
mod palette;
mod residual;

pub(super) use crate::scaled_reference::scale_sizes;
pub(super) use inter::InterTools;
pub(super) use lr_units::LrUnitReader;

/// Luma rows of superblocks deblocked together while a frame decodes (two
/// 64x64 superblock rows, one of 128x128). Jetson cycles against deblocking
/// the whole frame at its end (min of 5): 64 lines per group -1.97% /
/// -0.09% (1080p / 720p, our streams), 128 lines -1.84 to -2.70% / -0.61
/// to -0.76% (SVT 1080p -0.43%), 256 lines -0.25% / -0.62%. Shorter groups
/// cut the long vertical-edge runs into more filter calls (+1% instructions
/// at 64 lines on 720p).
const DEBLOCK_GROUP_LINES: usize = 128;

/// The decoder's per-frame block state, reused by every frame.
#[derive(Debug)]
pub(super) struct FrameMaps {
    /// Block state along the above row and left column, which every
    /// context reads.
    blocks: BlockContexts,
    /// The references and vectors the MV stack reads (inter frames
    /// only).
    motion: MotionGrid,
    /// Transform and level state of every 4x4 unit, for deblocking.
    edges: EdgeMap,
    /// The edge runs of the deblocking pass being run.
    edge_runs: Vec<nc_dsp::EdgeRun<nc_dsp::Av1EdgeFilter>>,
    /// CDEF strength indices and coded 8x8 units.
    cdef: CdefMap,
    /// CDEF's line buffers and unit list.
    cdef_scratch: cdef::CdefScratch,
    /// The luma transforms of the inter block being decoded.
    tx_leaves: Vec<inter_residual::TxLeaf>,
    /// Segment ids of the frame's 4x4 units, when it enables segmentation,
    /// and each unit's `seg_id_predicted` (temporal prediction only).
    pub(super) segments: SegmentMap,
    segment_predicted: SegmentMap,
    /// Inter prediction scratch.
    prediction: inter_pred::PredictionScratch,
    /// Where analysis records go, when they are being collected.
    #[cfg(feature = "analysis")]
    pub(super) stats: Option<nc_common::stream_stats::StreamStats>,
    /// What the current block's analysis records collect.
    #[cfg(feature = "analysis")]
    analysis: analysis::BlockScratch,
}

impl FrameMaps {
    pub(super) fn new() -> Self {
        Self {
            blocks: BlockContexts::new(),
            motion: MotionGrid::default(),
            edges: EdgeMap::default(),
            edge_runs: Vec::new(),
            cdef: CdefMap::default(),
            cdef_scratch: cdef::CdefScratch::default(),
            tx_leaves: Vec::new(),
            segments: SegmentMap::default(),
            segment_predicted: SegmentMap::default(),
            prediction: inter_pred::PredictionScratch::default(),
            #[cfg(feature = "analysis")]
            stats: None,
            #[cfg(feature = "analysis")]
            analysis: analysis::BlockScratch::default(),
        }
    }
}

#[allow(clippy::struct_excessive_bools)] // the sequence's tool flags
pub(super) struct FrameDecoder<'a> {
    header: FrameParameters,
    /// What the frame's MV stacks depend on.
    mv_frame: MvFrame,
    /// Boxed so the frame state stays small to move (about 16 KiB).
    pub(super) cdfs: Box<Cdfs>,
    maps: &'a mut FrameMaps,
    /// The frame's deblocking level of each reference and mode type.
    levels: LevelTable,
    /// Coefficient contexts of each plane: per 4x4 unit of the frame
    /// (with a superblock of spare units) above, and of the superblock
    /// row to the left.
    above: [Vec<u8>; 3],
    left: [Vec<u8>; 3],
    /// Dequantiser steps of each plane at `CurrentQIndex`.
    steps: [Steps; 3],
    /// Those of the block being decoded: `steps`, moved by its segment's
    /// quantizer offset (§7.12.2 `get_qindex`).
    block_steps: [Steps; 3],
    /// The block's qindex (`get_qindex(0, segment_id)`).
    block_q: u32,
    /// `CurrentQIndex`, which `delta_q_abs` moves (§5.11.12).
    current_q: u32,
    /// `ReadDeltas`: the superblock's first block codes its deltas.
    read_deltas: bool,
    pub(super) recon: Picture,
    /// The reference pictures by reference name (`LAST_FRAME` ..
    /// `ALTREF_FRAME`; index 0 unused).
    references: [Option<&'a Picture>; TOTAL_REFS],
    /// The frame's inter tools.
    inter_tools: InterTools,
    /// The projected motion field (`use_ref_frame_mvs`), built one
    /// superblock row at a time.
    temporal: TemporalField<'a>,
    /// The vectors the frame keeps for later frames.
    pub(super) saved_motion: Option<SavedMotion>,
    dsp: &'a Dsp,
    /// Transform-size contexts.
    txfm: TxContexts,
    /// The tile being decoded.
    tile: TileBounds,
    /// Palette contexts and the current block's palettes, in frames that
    /// allow screen content tools.
    palette: Option<Box<palette::PaletteState>>,
    /// Which 4x4 units of the current superblock are decoded.
    decoded: DecodedUnits,
    /// The chroma modes intra edge filtering reads.
    uv_smooth: ChromaSmooth,
    /// The sequence enables filter intra (`use_filter_intra` is coded).
    filter_intra: bool,
    /// The sequence's superblock size (64x64 or 128x128).
    superblock: BlockSize,
    /// `enable_intra_edge_filter`: directional prediction filters its edges.
    intra_edge_filter: bool,
    /// `enable_cdef`: non-skipped blocks code `cdef_idx`.
    cdef: bool,
    /// `mono_chrome`: no block carries chroma.
    mono_chrome: bool,
    /// Coefficient scratch, kept zero between transform blocks: a block
    /// clears only the positions it coded (zeroing whole arrays per block
    /// was once the largest single cost of native AV1 decode).
    coefficients: Box<CoefficientBuffer>,
    /// The restoration units superblocks code, when a plane is restored.
    pub(super) restoration: Option<LrUnitReader>,
    /// The primary reference's segment map, when it covers this frame
    /// (`load_previous_segment_ids`, §7.20).
    pub(super) previous_segments: Option<Rc<SegmentMap>>,
}

/// The sequence-header tools the block layer follows.
#[derive(Clone, Copy)]
#[allow(clippy::struct_excessive_bools)] // the specification's flags
pub(super) struct SequenceTools {
    pub superblock: BlockSize,
    pub filter_intra: bool,
    pub intra_edge_filter: bool,
    pub cdef: bool,
    pub mono_chrome: bool,
}

/// Luma samples per mode-info unit (`MI_SIZE`).
const MI_SIZE: usize = 1 << MI_SIZE_LOG2;
/// `DELTA_Q_SMALL`: `delta_q_abs` values coded directly; the largest
/// escapes to `delta_q_rem_bits` (3 bits) and `delta_q_abs_bits`.
const DELTA_Q_SMALL: usize = 3;
const DELTA_Q_REM_BITS: u32 = 3;
/// The range of `CurrentQIndex` (§5.11.12: `Clip3(1, 255, ...)`).
const MIN_DELTA_QINDEX: i32 = 1;
const MAX_QINDEX: i32 = 255;
/// Smallest side of skip-mode blocks (8 samples), 4x4 units.
const SKIP_MODE_MIN_MI: usize = 2;
/// 4x4 units per side of the largest superblock.
const MAX_SUPERBLOCK_MI: usize = 32;
/// Partition contexts per block size (§8.3.2: above and left flags).
const PARTITION_CONTEXTS: usize = 4;
/// `partition` alphabet sizes: 8x8 blocks have no T-shaped or 4-way
/// splits, 128x128 blocks no 4-way splits (§5.11.4).
const PARTITION_SYMBOLS_8X8: usize = 4;
const PARTITION_SYMBOLS_128X128: usize = 8;
const PARTITION_SYMBOLS: usize = 10;
/// Partition types in symbol order.
const PARTITIONS: [Partition; PARTITION_SYMBOLS] = [
    Partition::None,
    Partition::Horz,
    Partition::Vert,
    Partition::Split,
    Partition::HorzA,
    Partition::HorzB,
    Partition::VertA,
    Partition::VertB,
    Partition::Horz4,
    Partition::Vert4,
];

/// A block being decoded: its top-left 4x4 unit and size.
#[derive(Debug, Clone, Copy)]
struct BlockPos {
    mi_col: usize,
    mi_row: usize,
    size: BlockSize,
    /// `HasChroma`, derived once.
    chroma: bool,
    /// 4x4 rows of the tile above the block and columns left of it.
    tile_rows_above: usize,
    tile_cols_left: usize,
}

impl BlockPos {
    /// The `size` block at 4x4 unit (`mi_col`, `mi_row`) of `tile`, in a
    /// frame with chroma unless `mono_chrome`.
    fn new(
        mi_col: usize,
        mi_row: usize,
        size: BlockSize,
        mono_chrome: bool,
        tile: &TileBounds,
    ) -> Self {
        // `HasChroma` (§5.11.5, 4:2:0): a block 4 wide (high) at an even
        // column (row) leaves its chroma to the block after it.
        let thin = |mi4: usize, at: usize| mi4 == 1 && at.is_multiple_of(2);
        let chroma =
            !mono_chrome && !thin(size.height_mi(), mi_row) && !thin(size.width_mi(), mi_col);
        Self {
            mi_col,
            mi_row,
            size,
            chroma,
            tile_rows_above: mi_row - tile.row_start,
            tile_cols_left: mi_col - tile.col_start,
        }
    }
    /// `HasChroma`: the block carries the chroma of its 8x8 area.
    fn has_chroma(self) -> bool {
        self.chroma
    }
    /// (`AvailU`, `AvailL`) of `plane`, inside the tile; the chroma of a
    /// 4-wide (high) block starts a unit further left (up) (§5.11.5
    /// `AvailUChroma`, `AvailLChroma`).
    fn available(self, plane: PlaneIndex) -> (bool, bool) {
        if plane == PlaneIndex::Y {
            return (self.tile_rows_above > 0, self.tile_cols_left > 0);
        }
        let reach = |mi4: usize| if mi4 == 1 { 2 } else { 1 };
        (
            self.tile_rows_above >= reach(self.size.height_mi()),
            self.tile_cols_left >= reach(self.size.width_mi()),
        )
    }
    /// The block's 4x4 units (width, height).
    fn units(self) -> (usize, usize) {
        (self.size.width_mi(), self.size.height_mi())
    }
}

/// Dequantiser steps of `base_q_idx` with the DC and AC deltas of a plane
/// (§7.12.2 `dc_q`, `ac_q`).
fn steps(base_q_idx: u32, (dc_delta, ac_delta): (i32, i32)) -> Steps {
    let q = |delta: i32| (base_q_idx as i32 + delta).clamp(0, DC_QLOOKUP.len() as i32 - 1) as usize;
    Steps {
        dc: i32::from(DC_QLOOKUP[q(dc_delta)]),
        ac: i32::from(AC_QLOOKUP[q(ac_delta)]),
    }
}
impl<'a> FrameDecoder<'a> {
    /// The decoder of one frame of `header`'s coded size.
    pub(super) fn new(
        header: FrameParameters,
        cdfs: Box<Cdfs>,
        dsp: &'a Dsp,
        references: [Option<&'a Picture>; TOTAL_REFS],
        recon: Picture,
        maps: &'a mut FrameMaps,
        sequence_tools: SequenceTools,
        inter_tools: InterTools,
        temporal: TemporalField<'a>,
    ) -> Self {
        let (w, h) = (header.width, header.height);
        let (mi_cols, mi_rows) = (w / MI_SIZE, h / MI_SIZE);
        let screen_content = header.screen_content;
        let margin = MAX_SUPERBLOCK_MI;
        let sb_mi = sequence_tools.superblock.width_mi();
        maps.blocks.start_frame(mi_cols);
        maps.edges
            .start_frame(mi_cols, mi_rows, &header.loop_filter);
        maps.cdef.start_frame(mi_cols, mi_rows);
        if header.kind == FrameKind::Inter || header.allow_intrabc {
            maps.motion.start_frame(mi_cols, mi_rows);
        }
        let mv_frame = MvFrame {
            mi_cols,
            mi_rows,
            allow_high_precision_mv: header.allow_high_precision_mv,
            force_integer_mv: header.force_integer_mv,
            global_motion: header.global_motion,
            sign_bias: header.sign_bias,
            use_ref_frame_mvs: inter_tools.use_ref_frame_mvs,
            projection_distances: inter_tools
                .ref_offsets
                .map(|offset| offset.clamp(-MAX_PROJECTION_DISTANCE, MAX_PROJECTION_DISTANCE)),
        };
        let saved_motion = inter_tools
            .saved_sides
            .filter(|_| header.kind == FrameKind::Inter)
            .map(|_| SavedMotion::new(mi_cols, mi_rows));
        if let Some(segments) = header.segments {
            maps.segments.reset(mi_cols, mi_rows);
            if segments.temporal_update {
                maps.segment_predicted.reset(mi_cols, mi_rows);
            }
        }
        let base_q_idx = header.base_q_idx;
        let plane_steps = header
            .quant_deltas
            .map(|deltas| steps(header.base_q_idx, deltas));
        let restoration = header
            .restoration
            .any()
            .then(|| LrUnitReader::new(header.restoration, header.size));
        Self {
            levels: LevelTable::new(&header.loop_filter),
            header,
            cdfs,
            maps,
            above: [
                vec![0; mi_cols + margin],
                vec![0; mi_cols / 2 + margin / 2],
                vec![0; mi_cols / 2 + margin / 2],
            ],
            left: [vec![0; sb_mi], vec![0; sb_mi / 2], vec![0; sb_mi / 2]],
            steps: plane_steps,
            block_steps: plane_steps,
            block_q: base_q_idx,
            current_q: base_q_idx,
            read_deltas: false,
            recon,
            references,
            inter_tools,
            temporal,
            saved_motion,
            dsp,
            txfm: TxContexts::new(mi_cols),
            tile: TileBounds::frame(mi_cols, mi_rows),
            palette: screen_content.then(|| Box::new(palette::PaletteState::new(mi_cols))),
            decoded: DecodedUnits::new(mi_cols, mi_rows, sequence_tools.superblock),
            uv_smooth: ChromaSmooth::new(mi_cols),
            filter_intra: sequence_tools.filter_intra,
            superblock: sequence_tools.superblock,
            intra_edge_filter: sequence_tools.intra_edge_filter,
            cdef: sequence_tools.cdef,
            mono_chrome: sequence_tools.mono_chrome,
            restoration,
            previous_segments: None,
            mv_frame,
            coefficients: CoefficientBuffer::new(),
        }
    }
    /// Deblocking (§7.14) of the frame's blocks not deblocked yet.
    pub(super) fn deblock(&mut self) {
        self.deblock_until(None);
    }
    /// Deblocking of the blocks decoded before `until` (all when `None`)
    /// not deblocked yet.
    // Out of line: inlined into the frame decode, its edge kernel dispatch
    // stopped inlining (+10M instructions on static 720p).
    #[inline(never)]
    fn deblock_until(&mut self, until: Option<deblock::AreaMark>) {
        deblock::apply(
            self.dsp,
            &mut self.recon,
            &mut self.maps.edges,
            &self.header.loop_filter,
            (self.header.size.width, self.header.size.height),
            &mut self.maps.edge_runs,
            until,
        );
    }
    /// CDEF (§7.15) of the deblocked frame.
    pub(super) fn cdef(&mut self) {
        cdef::apply(
            self.dsp,
            &mut self.recon,
            &self.maps.cdef,
            &self.header.cdef,
            &mut self.maps.cdef_scratch,
        );
    }
    /// The tiles of the frame (§5.11.1 `tile_group_obu`), one coded tile
    /// in `data` per tile of `tiles` in raster order, symbols adapting
    /// unless `disable_cdf_update`. Each tile starts from the frame's
    /// CDFs; the frame keeps those `context_update_tile_id` ends with.
    pub(super) fn tiles(
        &mut self,
        tiles: &TileInfo,
        data: &[&[u8]],
        adapting: bool,
    ) -> Result<(), DecodeError> {
        #[cfg(feature = "analysis")]
        if let Some(s) = &mut self.maps.stats {
            analysis::frame_facts(s, &self.header, data);
        }
        // One tile (our own streams) adapts the frame's set in place.
        let start = (data.len() > 1).then(|| self.cdfs.clone());
        let mut kept = None;
        for (index, bytes) in data.iter().enumerate() {
            if let Some(start) = start.as_ref().filter(|_| index > 0) {
                self.cdfs.clone_from(start);
            }
            if bytes.is_empty() {
                return Err(DecodeError::Truncated);
            }
            let mut reader = SymbolReader::new(bytes).adapting(adapting);
            self.tile(&mut reader, tiles.bounds(index))?;
            if reader.overran() {
                return Err(DecodeError::Truncated);
            }
            if start.is_some() && index == tiles.context_update_tile_id {
                kept = Some(self.cdfs.clone());
            }
        }
        if let Some(kept) = kept {
            self.cdfs = kept;
        }
        #[cfg(feature = "analysis")]
        if let Some(s) = &mut self.maps.stats {
            s.finish();
        }
        Ok(())
    }
    /// `decode_tile()` (§5.11.2) of the tile `tile`, after the state it
    /// starts from: cleared above contexts, the frame's quantizer index
    /// (§5.11.1 `CurrentQIndex = base_q_idx`) and restoration references.
    fn tile(&mut self, r: &mut SymbolReader<'_>, tile: TileBounds) -> Result<(), DecodeError> {
        self.tile = tile;
        self.decoded.start_tile(&tile);
        for above in &mut self.above {
            above.fill(0);
        }
        self.txfm = TxContexts::new(self.header.width / MI_SIZE);
        if let Some(lr) = &mut self.restoration {
            lr.start_tile();
        }
        self.set_q_index(self.header.base_q_idx);
        let side = self.superblock.width_mi();
        // A tile as wide as the frame deblocks its superblock rows in groups
        // (`DEBLOCK_GROUP_LINES`), each once the group below it is decoded,
        // while its samples are still in cache (dav1d filters by superblock
        // row too). Not sooner: the next row's intra prediction reads the
        // unfiltered bottom samples.
        let by_rows = tile.col_start == 0 && tile.col_end >= self.header.width / MI_SIZE;
        let group = (DEBLOCK_GROUP_LINES / (side * MI_SIZE)).max(1);
        let mut group_above = self.maps.edges.mark();
        for mi_row in (tile.row_start..tile.row_end).step_by(side) {
            if (mi_row / side).is_multiple_of(group) {
                group_above = self.maps.edges.mark();
            }
            for left in &mut self.left {
                left.fill(0);
            }
            self.txfm.start_row();
            if self.inter_tools.use_ref_frame_mvs {
                self.temporal.project_rows(
                    tile.col_start / 2..tile.col_end.div_ceil(2),
                    mi_row / 2..(mi_row + side).div_ceil(2),
                );
            }
            for mi_col in (tile.col_start..tile.col_end).step_by(side) {
                self.decoded.start_superblock(mi_col, mi_row);
                self.read_deltas = self.header.delta_q_res.is_some();
                if let Some(lr) = &mut self.restoration {
                    lr.read_superblock(r, &mut self.cdfs, (mi_col, mi_row), side)?;
                }
                self.partition(r, mi_col, mi_row, self.superblock)?;
            }
            if by_rows && (mi_row / side) % group == group - 1 {
                self.deblock_until(Some(group_above));
            }
        }
        Ok(())
    }
    /// `decode_partition()` (AV1 §5.11.4) of the square `size` block at
    /// 4x4 unit (`mi_col`, `mi_row`).
    fn partition(
        &mut self,
        r: &mut SymbolReader<'_>,
        mi_col: usize,
        mi_row: usize,
        size: BlockSize,
    ) -> Result<(), DecodeError> {
        let (mi_cols, mi_rows) = (self.header.width / MI_SIZE, self.header.height / MI_SIZE);
        if mi_row >= mi_rows || mi_col >= mi_cols {
            return Ok(());
        }
        let half = size.width_mi() / 2;
        let quarter = half / 2;
        let has_rows = mi_row + half < mi_rows;
        let has_cols = mi_col + half < mi_cols;
        let partition = self.partition_symbol(r, mi_col, mi_row, size, has_rows, has_cols);
        let sub = size
            .subsize(partition)
            .ok_or(DecodeError::Malformed("AV1 partition of a block size"))?;
        if partition == Partition::Split {
            for (dy, dx) in [(0, 0), (0, half), (half, 0), (half, half)] {
                self.partition(r, mi_col + dx, mi_row + dy, sub)?;
            }
            return Ok(());
        }
        // The square quarters of T-shaped partitions (8x8 and up).
        let split = size.subsize(Partition::Split).unwrap_or(sub);
        // (row offset, column offset, size) of each block, in coding order.
        let blocks: &[(usize, usize, BlockSize)] = match partition {
            Partition::None => &[(0, 0, sub)],
            Partition::Horz => &[(0, 0, sub), (half, 0, sub)],
            Partition::Vert => &[(0, 0, sub), (0, half, sub)],
            Partition::HorzA => &[(0, 0, split), (0, half, split), (half, 0, sub)],
            Partition::HorzB => &[(0, 0, sub), (half, 0, split), (half, half, split)],
            Partition::VertA => &[(0, 0, split), (half, 0, split), (0, half, sub)],
            Partition::VertB => &[(0, 0, sub), (0, half, split), (half, half, split)],
            Partition::Horz4 => &[
                (0, 0, sub),
                (quarter, 0, sub),
                (2 * quarter, 0, sub),
                (3 * quarter, 0, sub),
            ],
            Partition::Vert4 => &[
                (0, 0, sub),
                (0, quarter, sub),
                (0, 2 * quarter, sub),
                (0, 3 * quarter, sub),
            ],
            Partition::Split => unreachable!("handled above"),
        };
        #[cfg(feature = "analysis")]
        {
            self.maps.analysis.shape = analysis::shape(partition);
        }
        for &(dy, dx, block) in blocks {
            // Blocks starting outside the frame are not coded: the second
            // half of a HORZ/VERT at the frame edge (`hasRows`/`hasCols`)
            // and the last strip of a 4-way split (§5.11.4).
            if mi_row + dy < mi_rows && mi_col + dx < mi_cols {
                self.leaf(r, mi_col + dx, mi_row + dy, block)?;
            }
        }
        Ok(())
    }
    /// The partition type: `partition`, `split_or_horz` or `split_or_vert`
    /// as the frame edge allows (§5.11.4), with the §8.3.2 context.
    fn partition_symbol(
        &mut self,
        r: &mut SymbolReader<'_>,
        mi_col: usize,
        mi_row: usize,
        size: BlockSize,
        has_rows: bool,
        has_cols: bool,
    ) -> Partition {
        if size.width_log2() < BlockSize::BLOCK_8X8.width_log2() {
            return Partition::None;
        }
        if !has_rows && !has_cols {
            return Partition::Split;
        }
        let ctx = usize::from(size.width_mi_log2() - 1) * PARTITION_CONTEXTS
            + self
                .maps
                .blocks
                .neighbours(mi_col, mi_row, self.tile.available(mi_col, mi_row))
                .partition_context(size);
        let cdf = &mut self.cdfs.partition[ctx];
        if has_rows && has_cols {
            let symbols = if size == BlockSize::BLOCK_8X8 {
                PARTITION_SYMBOLS_8X8
            } else if size == BlockSize::BLOCK_128X128 {
                PARTITION_SYMBOLS_128X128
            } else {
                PARTITION_SYMBOLS
            };
            return PARTITIONS[r.symbol(cdf, symbols)];
        }
        // 128x128 blocks have no 4-way splits to fold in (§8.3.2).
        let alike = if has_cols {
            &VERT_ALIKE[..]
        } else {
            &HORZ_ALIKE[..]
        };
        let alike = if size == BlockSize::BLOCK_128X128 {
            &alike[..alike.len() - 1]
        } else {
            alike
        };
        let split = r.fixed_boolean(gather(&cdf.icdf, alike)[0]);
        match (split, has_cols) {
            (true, _) => Partition::Split,
            (false, true) => Partition::Horz,
            (false, false) => Partition::Vert,
        }
    }
    /// One coded block (`decode_block()`, §5.11.5): `skip_mode`, `skip`,
    /// `cdef_idx`, `is_inter`, then an intra or inter block.
    fn leaf(
        &mut self,
        r: &mut SymbolReader<'_>,
        mi_col: usize,
        mi_row: usize,
        size: BlockSize,
    ) -> Result<(), DecodeError> {
        let block = BlockPos::new(mi_col, mi_row, size, self.mono_chrome, &self.tile);
        let inter_frame = self.header.kind == FrameKind::Inter;
        let near = self
            .maps
            .blocks
            .neighbours(mi_col, mi_row, block.available(PlaneIndex::Y));
        let skip_mode = self.header.skip_mode_frames.is_some()
            && inter_frame
            && self.read_skip_mode(r, size, near);
        let skip = skip_mode || r.boolean(&mut self.cdfs.skip[near.skip_context()]);
        let segment = match self.header.segments {
            Some(segments) => self.read_segment_id(r, block, skip, segments)?,
            None => 0,
        };
        if !skip {
            self.read_cdef(r, (mi_col, mi_row), block.units());
        }
        if self.read_deltas {
            self.read_delta_qindex(r, size, skip);
            self.read_deltas = false;
        }
        self.set_block_q(segment);
        let inter = skip_mode
            || inter_frame && r.boolean(&mut self.cdfs.intra_inter[near.intra_inter_context()]);
        let intrabc =
            !inter_frame && self.header.allow_intrabc && r.boolean(&mut self.cdfs.intrabc);
        #[cfg(feature = "analysis")]
        if self.maps.stats.is_some() {
            self.start_block_record(r, inter || intrabc);
        }
        if inter || intrabc {
            self.inter_block(r, block, near, skip, skip_mode)
        } else {
            self.intra_block(r, block, near, skip)
        }
    }
    /// `read_skip_mode()` (§5.11.10) in a frame with `skip_mode_present`:
    /// blocks of 8x8 and up code it; skip-mode blocks are skipped compound
    /// blocks.
    // Out of line: our own streams never code skip mode.
    #[cold]
    #[inline(never)]
    fn read_skip_mode(
        &mut self,
        r: &mut SymbolReader<'_>,
        size: BlockSize,
        near: Neighbours,
    ) -> bool {
        size.width_mi().min(size.height_mi()) >= SKIP_MODE_MIN_MI
            && r.boolean(&mut self.cdfs.skip_mode[near.skip_mode_context()])
    }
    /// `read_delta_qindex()` (§5.11.12) of a superblock's first block: a
    /// new `CurrentQIndex`, and the dequantiser steps that follow it
    /// (§7.12.2 `get_qindex`).
    // Out of line: our own streams code no quantizer deltas.
    #[cold]
    #[inline(never)]
    fn read_delta_qindex(&mut self, r: &mut SymbolReader<'_>, size: BlockSize, skip: bool) {
        let Some(res) = self.header.delta_q_res else {
            return;
        };
        if size == self.superblock && skip {
            return;
        }
        let mut abs = r.symbol(&mut self.cdfs.delta_q, DELTA_Q_SMALL + 1) as u32;
        if abs == DELTA_Q_SMALL as u32 {
            let rem_bits = r.literal(DELTA_Q_REM_BITS) + 1;
            abs = r.literal(rem_bits) + (1 << rem_bits) + 1;
        }
        if abs != 0 {
            let delta = if r.literal(1) == 1 {
                -(abs as i32)
            } else {
                abs as i32
            };
            let q = self.current_q as i32 + (delta << res);
            self.set_q_index(q.clamp(MIN_DELTA_QINDEX, MAX_QINDEX) as u32);
        }
    }
    /// `intra_segment_id()` / `inter_segment_id()` without pre-skip ids
    /// (§5.11.8, §5.11.19): a frame that keeps its reference's map copies
    /// it (the block takes the smallest id under it); otherwise a block
    /// predicted from the reference's map (`seg_id_predicted`) takes that,
    /// and the others read their id against the spatial prediction
    /// ([`Self::read_spatial_segment_id`]). The id fills the block's units.
    // Out of line: only streams with segmentation reach it.
    #[cold]
    #[inline(never)]
    fn read_segment_id(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        skip: bool,
        segments: SegmentQuantizers,
    ) -> Result<u8, DecodeError> {
        let (at, units) = ((block.mi_col, block.mi_row), block.units());
        let previous = self.previous_segments.clone();
        let previous = previous.as_deref();
        if !segments.update_map {
            self.maps.segments.copy_from(previous, at, units);
            return Ok(previous.map_or(0, |map| map.min_over(at, units)));
        }
        let id = if segments.temporal_update {
            let predicted = !skip && {
                let (up, left) = block.available(PlaneIndex::Y);
                let flags = &self.maps.segment_predicted;
                let ctx = usize::from(up && flags.get(at.0, at.1 - 1) != 0)
                    + usize::from(left && flags.get(at.0 - 1, at.1) != 0);
                r.boolean(&mut self.cdfs.segment_predicted[ctx])
            };
            self.maps
                .segment_predicted
                .fill(at, units, u8::from(predicted));
            if predicted {
                previous.map_or(0, |map| map.min_over(at, units))
            } else {
                self.read_spatial_segment_id(r, block, skip, segments.last_active)?
            }
        } else {
            self.read_spatial_segment_id(r, block, skip, segments.last_active)?
        };
        self.maps.segments.fill(at, units, id);
        Ok(id)
    }
    /// `read_segment_id()` (§5.11.9): a skipped block takes its spatial
    /// prediction, another codes its id against it.
    fn read_spatial_segment_id(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        skip: bool,
        last_active: u8,
    ) -> Result<u8, DecodeError> {
        let at = (block.mi_col, block.mi_row);
        let prediction = self
            .maps
            .segments
            .predict(at, block.available(PlaneIndex::Y));
        if skip {
            return Ok(prediction.pred);
        }
        let coded = r.symbol(&mut self.cdfs.segment_id[prediction.ctx], MAX_SEGMENTS);
        let count = usize::from(last_active) + 1;
        let id = neg_deinterleave(coded, usize::from(prediction.pred), count);
        if id >= count {
            return Err(DecodeError::Malformed(
                "AV1 segment_id beyond LastActiveSegId",
            ));
        }
        Ok(id as u8)
    }
    /// The block's qindex and dequantiser steps: `CurrentQIndex` moved by
    /// `segment`'s quantizer offset (§7.12.2 `get_qindex(0, segment_id)`).
    fn set_block_q(&mut self, segment: u8) {
        let offset = self.header.segments.map_or(0, |segments| {
            i32::from(segments.alt_q[usize::from(segment)])
        });
        if offset == 0 {
            self.block_q = self.current_q;
            self.block_steps = self.steps;
            return;
        }
        let q = (self.current_q as i32 + offset).clamp(0, MAX_QINDEX) as u32;
        self.block_q = q;
        self.block_steps = self.header.quant_deltas.map(|deltas| steps(q, deltas));
    }
    /// Set `CurrentQIndex` and the dequantiser steps that follow it.
    fn set_q_index(&mut self, q_index: u32) {
        self.current_q = q_index;
        self.steps = self
            .header
            .quant_deltas
            .map(|deltas| steps(q_index, deltas));
    }
    /// An intra block (§5.11.7, §5.11.22): its modes, transform size and
    /// residual, then the state later blocks and the loop filters read.
    fn intra_block(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        near: Neighbours,
        skip: bool,
    ) -> Result<(), DecodeError> {
        let intra_frame = self.header.kind == FrameKind::Key;
        if let Some(palette) = &mut self.palette {
            palette.block = palette::BlockPalette::default();
        }
        let modes = self.intra_modes(r, block, near, intra_frame)?;
        if let Some(palette) = &mut self.palette {
            let frame_mi = (self.header.width / MI_SIZE, self.header.height / MI_SIZE);
            palette.read_tokens(r, &mut self.cdfs, block, frame_mi);
            palette.record(block);
        }
        let tx = self.intra_tx_size(r, block, near)?;
        let units = block.units();
        let at = (block.mi_col, block.mi_row);
        self.txfm
            .fill(at, units, (tx.width() as u8, tx.height() as u8));
        let edge_block = EdgeBlock {
            size: block.size,
            skipped_inter: false,
            levels: self.levels.get(INTRA_FRAME, 0),
        };
        let luma_tx = (tx.width_log2(), tx.height_log2());
        self.maps
            .edges
            .block(at, edge_block, Some(luma_tx), block.has_chroma());
        self.intra_residual(r, block, &modes, skip, tx)?;
        let y_mode = modes.luma.mode.symbol() as u8;
        self.maps.blocks.fill(
            block.mi_col,
            block.mi_row,
            BlockInfo::intra(block.size, skip, y_mode),
        );
        #[cfg(feature = "analysis")]
        if self.maps.stats.is_some() {
            self.intra_records(r, block, &modes, skip);
        }
        if self.intra_edge_filter && block.has_chroma() {
            // The chroma covers whole 8x8 luma areas.
            let origin = (block.mi_col & !1, block.mi_row & !1);
            let span = (units.0.max(2), units.1.max(2));
            self.uv_smooth
                .fill(origin, span, modes.smooth(PlaneIndex::U));
        }
        self.decoded.mark(at, units);
        // Intra frames record blocks only for intra block copy, whose
        // stacks must not take plain intra blocks (one call site keeps the
        // motion store's fill inlined).
        if !intra_frame || self.header.allow_intrabc {
            let cell = if intra_frame {
                MotionCell::no_reference(block.size)
            } else {
                MotionCell::intra(block.size)
            };
            self.record_motion(block, cell);
        }
        Ok(())
    }
    /// An inter block (§5.11.23), or an intra block copy block (§5.11.7,
    /// the only kind decoded here in frames that allow it): its mode info, transform sizes, prediction and
    /// residual, then the state later blocks and the loop filters read.
    // Out of line: inlined into `leaf` (once intra block copy could
    // reach it), its mode reading went out of line instead (+8M
    // instructions on static 720p).
    #[inline(never)]
    fn inter_block(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        near: Neighbours,
        skip: bool,
        skip_mode: bool,
    ) -> Result<(), DecodeError> {
        // Only intra block copy blocks are decoded here in such frames.
        let intrabc = self.header.allow_intrabc;
        let modes = self.inter_modes(r, block, near, skip_mode)?;
        let at = (block.mi_col, block.mi_row);
        // Recorded first: the chroma of 4-sample-wide blocks reads it.
        self.record_motion(block, modes.cell(block.size));
        self.read_inter_tx(r, block, skip)?;
        let reference = modes.refs[0] as usize;
        // `GLOBALMV` and `GLOBAL_GLOBALMV` are mode type 0, the other
        // inter modes 1.
        let levels = self
            .levels
            .get(reference, usize::from(!modes.mode.is_global()));
        let edge_block = EdgeBlock {
            size: block.size,
            skipped_inter: skip,
            levels,
        };
        self.record_inter_edges(block, edge_block);
        if intrabc {
            self.predict_intrabc(block, modes.mv[0])?;
        } else {
            self.predict_inter(block, &modes)?;
        }
        if skip {
            self.reset_block_contexts(block);
        } else {
            self.inter_residual(r, block)?;
        }
        self.maps.blocks.fill(
            block.mi_col,
            block.mi_row,
            if intrabc {
                BlockInfo::intrabc(block.size, skip)
            } else {
                BlockInfo::inter(block.size, skip, modes.refs, modes.flags)
            },
        );
        #[cfg(feature = "analysis")]
        if self.maps.stats.is_some() {
            self.inter_records(r, block, &modes, (skip, skip_mode));
        }
        let units = block.units();
        self.decoded.mark(at, units);
        if self.intra_edge_filter && block.has_chroma() {
            let origin = (block.mi_col & !1, block.mi_row & !1);
            let span = (units.0.max(2), units.1.max(2));
            self.uv_smooth.fill(origin, span, false);
        }
        if let Some(palette) = &mut self.palette {
            palette.record_none(block);
        }
        Ok(())
    }
    /// Record a block of an inter frame in the motion store, and its
    /// vectors for later frames.
    fn record_motion(&mut self, block: BlockPos, cell: MotionCell) {
        let at = (block.mi_col, block.mi_row);
        self.maps.motion.fill(at, cell);
        if let (Some(saved), Some(sides)) = (&mut self.saved_motion, &self.inter_tools.saved_sides)
        {
            let (mi_cols, mi_rows) = (self.header.width / MI_SIZE, self.header.height / MI_SIZE);
            let units = (
                block.size.width_mi().min(mi_cols - block.mi_col),
                block.size.height_mi().min(mi_rows - block.mi_row),
            );
            saved.record(at, units, &cell, *sides);
        }
    }
    /// `read_cdef()` (§5.11.56) of a non-skipped block of `w4` x `h4`
    /// units: the first in its 64x64 filter block codes `cdef_idx`.
    fn read_cdef(
        &mut self,
        reader: &mut SymbolReader<'_>,
        (mi_col, mi_row): (usize, usize),
        (w4, h4): (usize, usize),
    ) {
        if !self.cdef {
            return;
        }
        self.maps.cdef.coded_block((mi_col, mi_row), (w4, h4));
        if self.maps.cdef.needs_index(mi_col, mi_row) {
            let index = reader.literal(u32::from(self.header.cdef.bits)) as i8;
            self.maps.cdef.set_index((mi_col, mi_row), (w4, h4), index);
        }
    }
}
