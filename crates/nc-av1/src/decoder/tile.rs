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
use super::subset::{ForcedSyntax, FrameParameters, Segments};
use crate::backend::{Dsp, FrameKind, HORZ_ALIKE, PlaneIndex, VERT_ALIKE, gather};
use crate::decoder_range::SymbolReader;
use crate::segment_id::{MAX_SEGMENTS, SegmentMap, neg_deinterleave};
use crate::tables::Partition;
use crate::tables::{
    AC_QLOOKUP, AC_QLOOKUP_10, AC_QLOOKUP_12, DC_QLOOKUP, DC_QLOOKUP_10, DC_QLOOKUP_12,
};
use nc_common::decode::DecodeError;
use nc_common::{BitDepth, ChromaFormat, Picture};
use nc_dsp::{DepthKernels, Pixel};
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

/// The decoder's per-frame block state, reused by every frame (`T`: the
/// samples of the prediction scratch).
#[derive(Debug)]
pub(super) struct FrameMaps<T: Pixel> {
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
    cdef_scratch: cdef::CdefScratch<T>,
    /// The luma transforms of the inter block being decoded.
    tx_leaves: Vec<inter_residual::TxLeaf>,
    /// Segment ids of the frame's 4x4 units, when it enables segmentation,
    /// and each unit's `seg_id_predicted` (temporal prediction only).
    pub(super) segments: SegmentMap,
    segment_predicted: SegmentMap,
    /// Inter prediction scratch.
    prediction: inter_pred::PredictionScratch<T>,
    /// The edges of the intra block being predicted, filled in place.
    intra_edges: nc_dsp::intra_prediction::av1_block::BlockEdges<T>,
    /// Where analysis records go, when they are being collected.
    #[cfg(feature = "analysis")]
    pub(super) stats: Option<nc_common::stream_stats::StreamStats>,
    /// What the current block's analysis records collect.
    #[cfg(feature = "analysis")]
    analysis: analysis::BlockScratch,
}

impl<T: Pixel> FrameMaps<T> {
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
            intra_edges: nc_dsp::intra_prediction::av1_block::BlockEdges::empty(
                nc_common::BitDepth::EIGHT,
            ),
            #[cfg(feature = "analysis")]
            stats: None,
            #[cfg(feature = "analysis")]
            analysis: analysis::BlockScratch::default(),
        }
    }
}

#[allow(clippy::struct_excessive_bools)] // the sequence's tool flags
pub(super) struct FrameDecoder<'a, T: Pixel> {
    header: FrameParameters,
    /// What the frame's MV stacks depend on.
    mv_frame: MvFrame,
    /// Boxed so the frame state stays small to move (about 16 KiB).
    pub(super) cdfs: Box<Cdfs>,
    maps: &'a mut FrameMaps<T>,
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
    /// The block's `segment_id`, and whether its segment is lossless
    /// (`Lossless`) and codes luma transform types (§5.11.47).
    block_segment: u8,
    block_lossless: bool,
    block_tx_types_coded: bool,
    /// `DeltaLF[]`, which `delta_lf_abs` moves (§5.11.13): one per
    /// `loop_filter_level[i]`, or entry 0 for all.
    delta_lf: [i8; FRAME_LF_COUNT],
    /// `CurrentQIndex`, which `delta_q_abs` moves (§5.11.12).
    current_q: u32,
    /// `ReadDeltas`: the superblock's first block codes its deltas.
    read_deltas: bool,
    pub(super) recon: Picture<T>,
    /// The reference pictures by reference name (`LAST_FRAME` ..
    /// `ALTREF_FRAME`; index 0 unused).
    references: [Option<&'a Picture<T>>; TOTAL_REFS],
    /// The frame's inter tools.
    inter_tools: InterTools,
    /// The projected motion field (`use_ref_frame_mvs`), built one
    /// superblock row at a time.
    temporal: TemporalField<'a>,
    /// The vectors the frame keeps for later frames.
    pub(super) saved_motion: Option<SavedMotion>,
    dsp: &'a Dsp,
    /// The sequence's `BitDepth` (luma and chroma alike in AV1); read
    /// through [`Self::depth`].
    bit_depth: BitDepth,
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
    /// The sequence's chroma format, and each plane's subsampling
    /// (`(0, 0)`, then `subsampling_x`, `subsampling_y` twice), which block
    /// geometry reads per plane.
    chroma_format: ChromaFormat,
    plane_subsampling: [(usize, usize); 3],
    /// Coefficient scratch, kept zero between transform blocks: a block
    /// clears only the positions it coded (zeroing whole arrays per block
    /// was once the largest single cost of native AV1 decode).
    coefficients: Box<CoefficientBuffer<T::Coefficient>>,
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
    pub chroma_format: ChromaFormat,
    pub bit_depth: BitDepth,
}

/// `depth` as code instantiated for samples `T` sees it: the constant 8
/// bits in the `u8` instance, so that depth arithmetic folds away there
/// (`BitDepths::of`).
#[inline]
pub(super) fn depth_of<T: Pixel>(depth: BitDepth) -> BitDepth {
    nc_common::BitDepths {
        luma: depth,
        chroma: depth,
    }
    .of::<T>()
    .luma
}

/// Luma samples per mode-info unit (`MI_SIZE`).
const MI_SIZE: usize = 1 << MI_SIZE_LOG2;
/// `DELTA_Q_SMALL`: `delta_q_abs` values coded directly; the largest
/// escapes to `delta_q_rem_bits` (3 bits) and `delta_q_abs_bits`.
const DELTA_Q_SMALL: usize = 3;
const DELTA_Q_REM_BITS: u32 = 3;
/// `FRAME_LF_COUNT`: loop filter deltas of `delta_lf_multi` (the two luma
/// directions, U, V); a monochrome frame codes the luma two.
const FRAME_LF_COUNT: usize = 4;
const FRAME_LF_COUNT_MONO: usize = 2;
/// `DELTA_LF_SMALL`: `delta_lf_abs` values coded directly, as `delta_q_abs`.
const DELTA_LF_SMALL: usize = 3;
/// The range of `DeltaLF[]` (`MAX_LOOP_FILTER`).
const MAX_DELTA_LF: i32 = 63;
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
    /// The frame's chroma subsampling (`subsampling_x`, `subsampling_y`).
    subsampling: (u8, u8),
    /// `HasChroma`, derived once.
    chroma: bool,
    /// 4x4 rows of the tile above the block and columns left of it.
    tile_rows_above: usize,
    tile_cols_left: usize,
}

impl BlockPos {
    /// The `size` block at 4x4 unit (`mi_col`, `mi_row`) of `tile`, in a
    /// frame with chroma subsampled by `subsampling` unless `mono_chrome`.
    fn new(
        mi_col: usize,
        mi_row: usize,
        size: BlockSize,
        (mono_chrome, subsampling): (bool, (usize, usize)),
        tile: &TileBounds,
    ) -> Self {
        // `HasChroma` (§5.11.5): along a subsampled axis, a block 4 wide
        // (high) at an even column (row) leaves its chroma to the block
        // after it.
        let thin = |ss: usize, mi4: usize, at: usize| ss == 1 && mi4 == 1 && at.is_multiple_of(2);
        let chroma = !mono_chrome
            && !thin(subsampling.1, size.height_mi(), mi_row)
            && !thin(subsampling.0, size.width_mi(), mi_col);
        Self {
            mi_col,
            mi_row,
            size,
            subsampling: (subsampling.0 as u8, subsampling.1 as u8),
            chroma,
            tile_rows_above: mi_row - tile.row_start,
            tile_cols_left: mi_col - tile.col_start,
        }
    }
    /// `HasChroma`: the block carries the chroma of its 8x8 area.
    fn has_chroma(self) -> bool {
        self.chroma
    }
    /// (`AvailU`, `AvailL`) of `plane`, inside the tile; along a
    /// subsampled axis the chroma of a 4-wide (high) block starts a unit
    /// further left (up) (§5.11.5 `AvailUChroma`, `AvailLChroma`).
    fn available(self, plane: PlaneIndex) -> (bool, bool) {
        if plane == PlaneIndex::Y {
            return (self.tile_rows_above > 0, self.tile_cols_left > 0);
        }
        let reach = |ss: u8, mi4: usize| if ss == 1 && mi4 == 1 { 2 } else { 1 };
        (
            self.tile_rows_above >= reach(self.subsampling.1, self.size.height_mi()),
            self.tile_cols_left >= reach(self.subsampling.0, self.size.width_mi()),
        )
    }
    /// The subsampling of `plane` (`(0, 0)` for luma).
    fn plane_subsampling(self, plane: PlaneIndex) -> (usize, usize) {
        if plane == PlaneIndex::Y {
            (0, 0)
        } else {
            (
                usize::from(self.subsampling.0),
                usize::from(self.subsampling.1),
            )
        }
    }
    /// The block's 4x4 units (width, height).
    fn units(self) -> (usize, usize) {
        (self.size.width_mi(), self.size.height_mi())
    }
}

/// Coefficient contexts of each plane over `units` luma 4x4 units along an
/// axis whose chroma is subsampled by `ss`.
fn plane_contexts(units: usize, ss: usize) -> [Vec<u8>; 3] {
    [vec![0; units], vec![0; units >> ss], vec![0; units >> ss]]
}

/// The luma units a block's chroma covers: its own, widened along a
/// subsampled axis to the whole 8-sample area of a 4-sample side.
fn chroma_area(block: BlockPos) -> ((usize, usize), (usize, usize)) {
    let (ss_x, ss_y) = block.plane_subsampling(PlaneIndex::U);
    let (w4, h4) = block.units();
    (
        (block.mi_col & !ss_x, block.mi_row & !ss_y),
        (w4.max(1 << ss_x), h4.max(1 << ss_y)),
    )
}

/// `BitDepth − 8` of 10-bit samples: the middle `Dc_Qlookup` table.
const TEN_BITS_ABOVE_EIGHT: u32 = 2;

/// Dequantiser steps of `base_q_idx` with the DC and AC deltas of a plane
/// at `depth` (§7.12.2 `dc_q`, `ac_q`: `Dc_Qlookup[(BitDepth − 8) >> 1]`).
fn steps(base_q_idx: u32, (dc_delta, ac_delta): (i32, i32), depth: BitDepth) -> Steps {
    let q = |delta: i32| (base_q_idx as i32 + delta).clamp(0, DC_QLOOKUP.len() as i32 - 1) as usize;
    let (dc, ac) = match depth.above_eight() {
        0 => (&DC_QLOOKUP, &AC_QLOOKUP),
        TEN_BITS_ABOVE_EIGHT => (&DC_QLOOKUP_10, &AC_QLOOKUP_10),
        _ => (&DC_QLOOKUP_12, &AC_QLOOKUP_12),
    };
    Steps {
        dc: i32::from(dc[q(dc_delta)]),
        ac: i32::from(ac[q(ac_delta)]),
        max: Steps::coefficient_max(depth.bits()),
    }
}
impl<'a, T: Pixel> FrameDecoder<'a, T> {
    /// The decoder of one frame of `header`'s coded size.
    pub(super) fn new(
        header: FrameParameters,
        cdfs: Box<Cdfs>,
        dsp: &'a Dsp,
        references: [Option<&'a Picture<T>>; TOTAL_REFS],
        recon: Picture<T>,
        maps: &'a mut FrameMaps<T>,
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
        maps.edges.start_frame(
            mi_cols,
            mi_rows,
            sequence_tools.chroma_format,
            &header.loop_filter,
        );
        maps.cdef.start_frame(mi_cols, mi_rows);
        if header.kind == FrameKind::Inter || header.allow_intrabc {
            maps.motion.start_frame(mi_cols, sb_mi);
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
        let (lossless, tx_types_coded) = (header.lossless[0], header.tx_types_coded[0]);
        let bit_depth = sequence_tools.bit_depth;
        let (ss_x, ss_y) = sequence_tools.chroma_format.subsampling();
        let depth = depth_of::<T>(bit_depth);
        let plane_steps = header
            .quant_deltas
            .map(|deltas| steps(header.base_q_idx, deltas, depth));
        let restoration = header.restoration.any().then(|| {
            LrUnitReader::new(
                header.restoration,
                header.size,
                sequence_tools.chroma_format,
            )
        });
        Self {
            levels: LevelTable::new(&header.loop_filter, header.segments.as_ref()),
            header,
            cdfs,
            maps,
            above: plane_contexts(mi_cols + margin, ss_x),
            left: plane_contexts(sb_mi, ss_y),
            steps: plane_steps,
            block_steps: plane_steps,
            block_q: base_q_idx,
            block_segment: 0,
            block_lossless: lossless,
            block_tx_types_coded: tx_types_coded,
            delta_lf: [0; FRAME_LF_COUNT],
            current_q: base_q_idx,
            read_deltas: false,
            recon,
            references,
            inter_tools,
            temporal,
            saved_motion,
            dsp,
            bit_depth,
            txfm: TxContexts::new(mi_cols),
            tile: TileBounds::frame(mi_cols, mi_rows),
            palette: screen_content.then(|| {
                Box::new(palette::PaletteState::new(
                    mi_cols,
                    depth.bits(),
                    (ss_x, ss_y),
                ))
            }),
            decoded: DecodedUnits::new(mi_cols, mi_rows, sequence_tools.superblock),
            uv_smooth: ChromaSmooth::new(mi_cols),
            filter_intra: sequence_tools.filter_intra,
            superblock: sequence_tools.superblock,
            intra_edge_filter: sequence_tools.intra_edge_filter,
            cdef: sequence_tools.cdef,
            mono_chrome: sequence_tools.mono_chrome,
            chroma_format: sequence_tools.chroma_format,
            plane_subsampling: [(0, 0), (ss_x, ss_y), (ss_x, ss_y)],
            restoration,
            previous_segments: None,
            mv_frame,
            coefficients: CoefficientBuffer::new(),
        }
    }
    /// The samples' bit depth: 8 in the `u8` instance (a constant there).
    #[inline]
    pub(super) fn depth(&self) -> BitDepth {
        depth_of::<T>(self.bit_depth)
    }
    /// The subsampling of `plane` (`(0, 0)` for luma).
    #[inline]
    pub(super) fn subsampling(&self, plane: PlaneIndex) -> (usize, usize) {
        self.plane_subsampling[plane as usize]
    }
    /// The kernels at the samples' depth.
    #[inline]
    pub(super) fn kernels(&self) -> DepthKernels<'a> {
        self.dsp.at(self.depth())
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
            self.kernels(),
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
            self.kernels(),
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
        self.delta_lf = [0; FRAME_LF_COUNT];
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
        // §5.11.4: every block's chroma must have a size
        // (`get_plane_residual_size` valid), which in 4:2:2 rules out the
        // partitions with blocks taller than wide (dav1d refuses them).
        if self.chroma_format == ChromaFormat::Yuv422
            && matches!(
                partition,
                Partition::Vert | Partition::Vert4 | Partition::VertA | Partition::VertB
            )
        {
            return Err(DecodeError::Malformed("AV1 vertical partition in 4:2:2"));
        }
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
    /// One coded block (`decode_block()`, §5.11.5; mode info §5.11.6–7,
    /// §5.11.18): a segment id read before `skip` (`SegIdPreSkip`),
    /// `skip_mode`, `skip`, the segment id read after it, `cdef_idx`, the
    /// superblock's quantizer and loop filter deltas, `is_inter`, then an
    /// intra or inter block. Segment features may infer `skip`,
    /// `is_inter`, the reference and the mode.
    fn leaf(
        &mut self,
        r: &mut SymbolReader<'_>,
        mi_col: usize,
        mi_row: usize,
        size: BlockSize,
    ) -> Result<(), DecodeError> {
        let block = BlockPos::new(
            mi_col,
            mi_row,
            size,
            (
                self.mono_chrome,
                self.plane_subsampling[PlaneIndex::U as usize],
            ),
            &self.tile,
        );
        let inter_frame = self.header.kind == FrameKind::Inter;
        let near = self
            .maps
            .blocks
            .neighbours(mi_col, mi_row, block.available(PlaneIndex::Y));
        let segments = self.header.segments;
        let pre_skip = segments.is_some_and(|s| s.pre_skip);
        let mut segment = 0;
        if let (Some(segments), true) = (segments, pre_skip) {
            segment = self.read_segment_id(r, block, false, segments)?;
        }
        // Features that infer syntax are pre-skip ones: none without it.
        let forced = segments
            .filter(|_| pre_skip)
            .map_or(ForcedSyntax::default(), |s| s.forced(segment));
        let skip_mode = self.header.skip_mode_frames.is_some()
            && inter_frame
            && !forced.any()
            && self.read_skip_mode(r, size, near);
        let skip = skip_mode || forced.skip || r.boolean(&mut self.cdfs.skip[near.skip_context()]);
        if let (Some(segments), false) = (segments, pre_skip) {
            segment = self.read_segment_id(r, block, skip, segments)?;
        }
        if !skip {
            self.read_cdef(r, (mi_col, mi_row), block.units());
        }
        if self.read_deltas {
            self.read_delta_qindex(r, size, skip);
            self.read_delta_lf(r, size, skip);
            self.read_deltas = false;
        }
        self.set_block_segment(segment);
        let inter = skip_mode
            || inter_frame
                && match forced.reference {
                    Some(reference) => reference != INTRA_FRAME as i8,
                    None => {
                        forced.global_motion
                            || r.boolean(&mut self.cdfs.intra_inter[near.intra_inter_context()])
                    }
                };
        let intrabc =
            !inter_frame && self.header.allow_intrabc && r.boolean(&mut self.cdfs.intrabc);
        #[cfg(feature = "analysis")]
        if self.maps.stats.is_some() {
            self.start_block_record(r, inter || intrabc);
        }
        if inter || intrabc {
            self.inter_block(r, block, near, (skip, skip_mode), forced)
        } else {
            self.intra_block(r, block, near, skip)
        }
    }
    /// `read_skip_mode()` (§5.11.10) in a frame with `skip_mode_present`:
    /// blocks of 8x8 and up code it; skip-mode blocks are skipped compound
    /// blocks.
    // Inline: libaom streams read it in most blocks of their compound
    // frames (our own streams never code skip mode, so never reach it).
    #[inline]
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
    /// `read_delta_lf()` (§5.11.13) of a superblock's first block: new
    /// `DeltaLF[]`, one for every `loop_filter_level[i]` or one for all.
    // Out of line: our own streams code no loop filter deltas.
    #[cold]
    #[inline(never)]
    fn read_delta_lf(&mut self, r: &mut SymbolReader<'_>, size: BlockSize, skip: bool) {
        let Some(delta_lf) = self.header.delta_lf else {
            return;
        };
        if size == self.superblock && skip {
            return;
        }
        let count = match (delta_lf.multi, self.mono_chrome) {
            (false, _) => 1,
            (true, false) => FRAME_LF_COUNT,
            (true, true) => FRAME_LF_COUNT_MONO,
        };
        for i in 0..count {
            let cdf = if delta_lf.multi {
                &mut self.cdfs.delta_lf_multi[i]
            } else {
                &mut self.cdfs.delta_lf
            };
            let mut abs = r.symbol(cdf, DELTA_LF_SMALL + 1) as u32;
            if abs == DELTA_LF_SMALL as u32 {
                let rem_bits = r.literal(DELTA_Q_REM_BITS) + 1;
                abs = r.literal(rem_bits) + (1 << rem_bits) + 1;
            }
            if abs != 0 {
                let reduced = if r.literal(1) == 1 {
                    -(abs as i32)
                } else {
                    abs as i32
                };
                let delta = i32::from(self.delta_lf[i]) + (reduced << delta_lf.res);
                self.delta_lf[i] = delta.clamp(-MAX_DELTA_LF, MAX_DELTA_LF) as i8;
            }
        }
    }
    /// `intra_segment_id()` / `inter_segment_id()` (§5.11.8, §5.11.19),
    /// before `skip` (with `skip` false) or after it: a frame that keeps
    /// its reference's map copies
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
        segments: Segments,
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
    /// The block's segment, what follows from it (`Lossless`, whether
    /// luma transform types are coded), and its qindex and dequantiser
    /// steps: `CurrentQIndex` moved by the segment's quantizer offset
    /// (§7.12.2 `get_qindex(0, segment_id)`).
    fn set_block_segment(&mut self, segment: u8) {
        let index = usize::from(segment);
        self.block_segment = segment;
        self.block_lossless = self.header.lossless[index];
        self.block_tx_types_coded = self.header.tx_types_coded[index];
        let offset = self.header.segments.map_or(0, |s| s.alt_q(segment));
        if offset == 0 {
            self.block_q = self.current_q;
            self.block_steps = self.steps;
            return;
        }
        let q = (self.current_q as i32 + offset).clamp(0, MAX_QINDEX) as u32;
        self.block_q = q;
        let depth = self.depth();
        self.block_steps = self
            .header
            .quant_deltas
            .map(|deltas| steps(q, deltas, depth));
    }
    /// The deblocking levels of the block being decoded, predicted from
    /// `reference` with inter mode type `mode_type` (§7.14.4): its
    /// segment's, moved by the superblock's loop filter deltas.
    fn block_levels(&self, reference: usize, mode_type: usize) -> deblock::BlockLevels {
        if self.delta_lf == [0; FRAME_LF_COUNT] {
            return self.levels.get(self.block_segment, reference, mode_type);
        }
        let multi = self.header.delta_lf.is_some_and(|d| d.multi);
        deblock::block_levels(
            &self.header.loop_filter,
            self.header.segments.as_ref(),
            self.block_segment,
            std::array::from_fn(|i| self.delta_lf[if multi { i } else { 0 }]),
            (reference, mode_type),
        )
    }
    /// Set `CurrentQIndex` and the dequantiser steps that follow it.
    fn set_q_index(&mut self, q_index: u32) {
        self.current_q = q_index;
        let depth = self.depth();
        self.steps = self
            .header
            .quant_deltas
            .map(|deltas| steps(q_index, deltas, depth));
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
            levels: self.block_levels(INTRA_FRAME, 0),
        };
        let luma_tx = (tx.width_log2(), tx.height_log2());
        self.maps.edges.block(
            at,
            edge_block,
            Some(luma_tx),
            self.recorded_chroma_tx(block),
        );
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
            let (origin, span) = chroma_area(block);
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
        (skip, skip_mode): (bool, bool),
        forced: ForcedSyntax,
    ) -> Result<(), DecodeError> {
        // Only intra block copy blocks are decoded here in such frames.
        let intrabc = self.header.allow_intrabc;
        let modes = self.inter_modes(r, block, near, skip_mode, forced)?;
        let at = (block.mi_col, block.mi_row);
        // Recorded first: the chroma of 4-sample-wide blocks reads it.
        self.record_motion(block, modes.cell(block.size));
        self.read_inter_tx(r, block, skip)?;
        let reference = modes.refs[0] as usize;
        // `GLOBALMV` and `GLOBAL_GLOBALMV` are mode type 0, the other
        // inter modes 1.
        let levels = self.block_levels(reference, usize::from(!modes.mode.is_global()));
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
            let (origin, span) = chroma_area(block);
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
        self.maps.motion.fill::<T>(at, cell);
        if let (Some(saved), Some(sides)) = (&mut self.saved_motion, &self.inter_tools.saved_sides)
        {
            saved.record(at, &cell, *sides);
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
        self.maps.cdef.coded_block::<T>((mi_col, mi_row), (w4, h4));
        if self.maps.cdef.needs_index(mi_col, mi_row) {
            let index = reader.literal(u32::from(self.header.cdef.bits)) as i8;
            self.maps.cdef.set_index((mi_col, mi_row), (w4, h4), index);
        }
    }
}
