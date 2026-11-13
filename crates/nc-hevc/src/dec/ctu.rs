//! Slice segment data (H.265 §7.3.8): the CTU loop with tiles, WPP and
//! dependent slice segments (§9.3.1, §9.3.2), the coding quadtree, coding
//! units (with PCM and lossless bypass), prediction units, transform trees
//! with QP deltas (§8.6.1), and the reconstruction of each block (§8.4,
//! §8.5, §8.6).

#![allow(
    clippy::many_single_char_names,
    reason = "H.265 sample and block variables keep their specification names (x, y, w, h, n)"
)]

use nc_common::decode::{BitReader, DecodeError};
use nc_common::stream_stats::{
    BlockRecord, IntraClass, Pred, RecordKind, Shape, StreamStats, flags,
};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::Dsp;
use nc_encoder::Mv;

use super::cabac::Engine;
use super::ctx::{self, Contexts};
use super::loop_filter::LoopFilter;
use super::mc::{self, Source};
use super::mvpred::{Collocated, Predictor, Pu};
use super::params::{FLAT_SCALE, Pps, ScalingLists, Sps};
use super::picture::{BlockInfo, Edges, GRID_LOG2, PictureState, PuMotion, flag};
use super::sao;
use super::slice::{SliceHeader, SliceSegment, SliceType};
use super::stats::SaoSyntax;

use crate::predict::{self, Refs, Sides};
use crate::quant::chroma_qp;
use crate::scan::{ScanKind, Scans};

/// `levelScale` (§8.6.3).
const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];
/// Coefficients of the largest transform block.
const MAX_COEFFS: usize = 32 * 32;
/// `cu_qp_delta_abs` prefix length before the Exp-Golomb suffix.
const QP_DELTA_PREFIX: u32 = 5;
/// `CuQpDeltaVal` range for 8-bit video (§7.4.9.14).
const QP_DELTA_RANGE: std::ops::RangeInclusive<i32> = -26..=25;
/// QP values wrap modulo this (equation 8-283).
const QP_WRAP: i32 = 52;
/// Transform-skip residuals are scaled up by 7 bits (§8.6.4.2, v1).
const TRANSFORM_SKIP_SHIFT: u32 = 7;
/// `bdShift = 20 − BitDepth` after the transform (or transform skip).
const RESIDUAL_SHIFT: u32 = 12;

/// `PartMode` (Table 7-10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Part {
    P2Nx2N,
    P2NxN,
    PNx2N,
    PNxN,
    P2NxnU,
    P2NxnD,
    PnLx2N,
    PnRx2N,
}

/// A prediction block `(x, y, w, h)` relative to its coding block.
type PartBlock = (usize, usize, usize, usize);

impl Part {
    /// The prediction blocks `(x, y, w, h)` relative to a CU of side `n`:
    /// the first `count` entries.
    fn blocks(self, n: usize) -> ([PartBlock; 4], usize) {
        let (h, q) = (n / 2, n / 4);
        let none = (0, 0, 0, 0);
        match self {
            Part::P2Nx2N => ([(0, 0, n, n), none, none, none], 1),
            Part::P2NxN => ([(0, 0, n, h), (0, h, n, h), none, none], 2),
            Part::PNx2N => ([(0, 0, h, n), (h, 0, h, n), none, none], 2),
            Part::PNxN => ([(0, 0, h, h), (h, 0, h, h), (0, h, h, h), (h, h, h, h)], 4),
            Part::P2NxnU => ([(0, 0, n, q), (0, q, n, n - q), none, none], 2),
            Part::P2NxnD => ([(0, 0, n, n - q), (0, n - q, n, q), none, none], 2),
            Part::PnLx2N => ([(0, 0, q, n), (q, 0, n - q, n), none, none], 2),
            Part::PnRx2N => ([(0, 0, n - q, n), (n - q, 0, q, n), none, none], 2),
        }
    }

    fn vertical_split(self) -> bool {
        matches!(self, Part::PNx2N | Part::PnLx2N | Part::PnRx2N)
    }

    fn horizontal_split(self) -> bool {
        matches!(self, Part::P2NxN | Part::P2NxnU | Part::P2NxnD)
    }
}

/// Everything about a slice that its segments' data needs.
pub(super) struct SliceContext<'a> {
    pub sps: &'a Sps,
    pub pps: &'a Pps,
    pub header: &'a SliceHeader,
    /// The slice's index in the picture, plus one.
    pub slice: u16,
    pub refs: [Vec<&'a Picture>; 2],
    pub col: Option<Collocated<'a>>,
    pub poc: i32,
    pub no_backward_pred: bool,
    pub dsp: &'a Dsp,
    pub scans: &'a Scans,
}

impl SliceContext<'_> {
    fn scaling(&self) -> Option<&ScalingLists> {
        self.pps.scaling.as_ref().or(self.sps.scaling.as_ref())
    }

    /// `initType` (§9.3.2.2).
    fn init_type(&self) -> usize {
        match (self.header.slice_type, self.header.cabac_init) {
            (SliceType::I, _) => 0,
            (SliceType::P, false) | (SliceType::B, true) => 1,
            (SliceType::P, true) | (SliceType::B, false) => 2,
        }
    }
}

/// CABAC and QP state that crosses slice segment boundaries.
#[derive(Debug, Default)]
pub(super) struct Carry {
    /// Contexts after the second CTB of the previous row (WPP, §9.3.2.4).
    wpp: Option<Contexts>,
    /// Contexts at the end of the previous slice segment (§9.3.2.4).
    dependent: Option<Contexts>,
    /// `QpY` of the last coding unit (`qPY_PREV`, §8.6.1).
    last_qp: i32,
}

/// The current quantization group (§8.6.1).
#[derive(Debug, Clone, Copy, Default)]
struct QuantGroup {
    /// `qPY_PRED`.
    pred: i32,
    /// `CuQpDeltaVal`.
    delta: i32,
    /// `IsCuQpDeltaCoded`.
    coded: bool,
}

/// Why the contexts are (re)initialised at a CTU (§9.3.1): the first CTU
/// of a tile, of a CTB row under WPP, or of a slice segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Start {
    Tile,
    Row,
    Segment { dependent: bool },
}

/// One coding unit being decoded.
#[derive(Debug, Clone, Copy)]
struct Cu {
    x: usize,
    y: usize,
    log2: u8,
    intra: bool,
    part: Part,
    bypass: bool,
    /// `IntraPredModeY` of the four `PART_NxN` partitions (all equal otherwise),
    /// and `IntraPredModeC`.
    luma_modes: [u8; 4],
    chroma_mode: u8,
}

/// A node of the transform tree (§7.3.8.8).
#[derive(Debug, Clone, Copy)]
struct TuNode {
    x: usize,
    y: usize,
    /// `xBase`, `yBase`: the parent's position (for 4x4 chroma).
    base: (usize, usize),
    log2: u8,
    depth: u8,
    blk_idx: usize,
    /// The parent's `cbf_cb`, `cbf_cr`.
    parent_cbf: [bool; 2],
}

/// Working memory of block decoding, kept by the decoder so that no
/// segment or block allocates or clears it. `levels` is all zero between
/// blocks: only the coded positions are set (and scaled in place) and
/// cleared again.
#[derive(Debug)]
pub(super) struct Scratch {
    levels: Vec<i32>,
    positions: Vec<u16>,
    residual: Vec<i32>,
    pred: Vec<u8>,
    /// Intra reference samples, refilled per block.
    refs: Refs,
    buffers: mc::Buffers,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            levels: vec![0; MAX_COEFFS],
            positions: Vec::with_capacity(MAX_COEFFS),
            residual: vec![0; MAX_COEFFS],
            pred: vec![0; MAX_COEFFS],
            refs: Refs::default(),
            buffers: mc::Buffers::default(),
        }
    }
}

/// Decode one slice segment's data into the picture.
pub(super) fn decode_segment(
    cx: &SliceContext<'_>,
    pic: &mut PictureState,
    carry: &mut Carry,
    scratch: &mut Scratch,
    filter: &mut LoopFilter,
    data: &[u8],
    segment: &SliceSegment,
    stats: Option<&mut StreamStats>,
) -> Result<(), DecodeError> {
    let mut d = SegmentDecoder {
        cx,
        pic,
        filter,
        ctx: Contexts::new(cx.init_type(), cx.header.qp),
        qg: QuantGroup::default(),
        last_qp: cx.header.qp,
        qp: cx.header.qp,
        s: scratch,
        stats,
        analysis: Analysis::default(),
    };
    d.run(carry, data, segment)
}

struct SegmentDecoder<'c, 'a> {
    cx: &'c SliceContext<'a>,
    pic: &'c mut PictureState,
    /// In-loop filtering of the CTB rows this segment completes.
    filter: &'c mut LoopFilter,
    ctx: Contexts,
    qg: QuantGroup,
    last_qp: i32,
    /// `QpY` of the current coding unit.
    qp: i32,
    s: &'c mut Scratch,
    /// Where block records go when analysis is enabled.
    stats: Option<&'c mut StreamStats>,
    /// What the current coding and transform unit's records collect.
    analysis: Analysis,
}

/// What the record of the current coding unit collects while it is parsed
/// (analysis only).
#[derive(Debug, Clone, Copy, Default)]
struct Analysis {
    /// Bits consumed when the coding unit started.
    cu_start: usize,
    /// Coded planes of the coding unit ([`flags`]).
    cu_flags: u8,
    /// `CuQpDeltaVal` when coded in this coding unit.
    qp_delta: Option<i8>,
    /// The current transform unit.
    tu: CodedBlock,
}

/// What a transform unit's residual held (analysis): the bits consumed
/// when it started, non-zero levels and transform skip over its blocks.
#[derive(Debug, Clone, Copy, Default)]
struct CodedBlock {
    start: usize,
    nonzero: u16,
    transform_skip: bool,
}

impl Part {
    /// The codec-neutral partition shape.
    fn shape(self) -> Shape {
        match self {
            Part::P2Nx2N => Shape::Whole,
            Part::P2NxN => Shape::HalfHorizontal,
            Part::PNx2N => Shape::HalfVertical,
            Part::PNxN => Shape::Quarter,
            Part::P2NxnU => Shape::QuarterTop,
            Part::P2NxnD => Shape::QuarterBottom,
            Part::PnLx2N => Shape::QuarterLeft,
            Part::PnRx2N => Shape::QuarterRight,
        }
    }
}

/// The class of an `IntraPredModeY` (§8.4.2): 0 planar, 1 DC, angular.
fn intra_class(mode: u8) -> IntraClass {
    match mode {
        predict::PLANAR => IntraClass::Planar,
        predict::DC => IntraClass::Dc,
        _ => IntraClass::Angular,
    }
}

/// A block position or size as a record field (pictures are below 64K).
fn coord(v: usize) -> u16 {
    v as u16
}

impl SegmentDecoder<'_, '_> {
    /// Push a block record when analysis is enabled.
    #[inline]
    fn record(&mut self, record: BlockRecord) {
        if let Some(s) = self.stats.as_deref_mut() {
            s.push(record);
        }
    }

    /// The bits consumed so far, when analysis needs them (else 0).
    #[inline]
    fn bits_mark(&self, e: &Engine<'_>) -> usize {
        if self.stats.is_some() {
            e.bits_consumed()
        } else {
            0
        }
    }

    /// The bits consumed since `start`, a [`Self::bits_mark`].
    #[inline]
    fn bits_since(&self, e: &Engine<'_>, start: usize) -> u32 {
        if self.stats.is_some() {
            (e.bits_consumed() - start) as u32
        } else {
            0
        }
    }

    fn run(
        &mut self,
        carry: &mut Carry,
        data: &[u8],
        segment: &SliceSegment,
    ) -> Result<(), DecodeError> {
        let layout = self.pic.layout.clone();
        let total = layout.ctb_w * layout.ctb_h;
        let wpp = self.cx.pps.entropy_sync;
        let mut ts = layout.rs_to_ts[segment.address];
        let mut e = Engine::new(data, segment.data_offset)?;
        let mut first = true;
        let ctb_size = 1usize << layout.log2_ctb;
        loop {
            let rs = layout.ts_to_rs[ts];
            let (x0, y0) = (
                (rs % layout.ctb_w) * ctb_size,
                (rs / layout.ctb_w) * ctb_size,
            );
            let tile_start =
                ts == 0 || layout.tile_of[rs] != layout.tile_of[layout.ts_to_rs[ts - 1]];
            let row_start =
                rs.is_multiple_of(layout.ctb_w) || layout.tile_of[rs - 1] != layout.tile_of[rs];
            let start = if tile_start {
                Some(Start::Tile)
            } else if wpp && row_start {
                Some(Start::Row)
            } else {
                first.then_some(Start::Segment {
                    dependent: segment.dependent,
                })
            };
            if let Some(start) = start {
                self.start_substream(carry, (x0, y0), start);
            }
            first = false;
            self.ctu(&mut e, rs, (x0, y0))?;
            self.pic.progress.ctb_decoded(rs, rs / layout.ctb_w)?;
            self.filter
                .advance(self.cx.dsp, self.cx.pps, self.pic, false);
            let end = e.terminate();
            if wpp {
                // Store after the second CTB of a row in its tile.
                let second = (rs % layout.ctb_w == 1
                    && layout.tile_of[rs - 1] == layout.tile_of[rs])
                    || (rs % layout.ctb_w >= 2
                        && layout.tile_of[rs - 1] == layout.tile_of[rs]
                        && layout.tile_of[rs - 2] != layout.tile_of[rs]);
                if second {
                    carry.wpp = Some(self.ctx.clone());
                }
            }
            if e.overrun() {
                return Err(DecodeError::Truncated);
            }
            ts += 1;
            if end {
                carry.dependent = Some(self.ctx.clone());
                carry.last_qp = self.last_qp;
                return Ok(());
            }
            if ts >= total {
                return Err(DecodeError::Malformed("slice data beyond the picture"));
            }
            let next = layout.ts_to_rs[ts];
            let new_tile = layout.tile_of[next] != layout.tile_of[rs];
            let new_row = next.is_multiple_of(layout.ctb_w)
                || layout.tile_of[next - 1] != layout.tile_of[next];
            if (self.cx.pps.tiles.is_some() && new_tile) || (wpp && new_row) {
                if !e.terminate() {
                    return Err(DecodeError::Malformed("end_of_subset_one_bit"));
                }
                e = Engine::new(data, e.next_byte())?;
            }
        }
    }

    /// Context and QP initialisation at the start of a substream or segment
    /// (§9.3.1, §9.3.2, §8.6.1).
    fn start_substream(&mut self, carry: &mut Carry, (x0, y0): (usize, usize), start: Start) {
        let fresh = || Contexts::new(self.cx.init_type(), self.cx.header.qp);
        self.ctx = match start {
            Start::Tile | Start::Segment { dependent: false } => fresh(),
            Start::Row => {
                // WPP: inherit from the CTB above-right when it is available.
                let ctb = 1isize << self.pic.layout.log2_ctb;
                let tr = (x0 as isize + ctb, y0 as isize - ctb);
                match &carry.wpp {
                    Some(c) if self.pic.available((x0, y0), tr, self.cx.slice) => c.clone(),
                    _ => fresh(),
                }
            }
            Start::Segment { dependent: true } => carry.dependent.clone().unwrap_or_else(fresh),
        };
        // qPY_PREV restarts at slices, tiles and (with WPP) CTB rows; a
        // dependent segment continues its slice.
        self.last_qp = if start == (Start::Segment { dependent: true }) {
            carry.last_qp
        } else {
            self.cx.header.qp
        };
    }

    /// `coding_tree_unit()` with its SAO syntax (§7.3.8.2).
    fn ctu(
        &mut self,
        e: &mut Engine<'_>,
        rs: usize,
        (x0, y0): (usize, usize),
    ) -> Result<(), DecodeError> {
        let header = self.cx.header;
        if header.sao[0] || header.sao[1] {
            let layout = &self.pic.layout;
            let slice_address = self.pic.slices[usize::from(self.cx.slice - 1)].address;
            let w = layout.ctb_w;
            let same_tile = |n: usize| layout.tile_of[n] == layout.tile_of[rs];
            let left =
                (!rs.is_multiple_of(w) && rs > slice_address && same_tile(rs - 1)).then(|| rs - 1);
            let up = (rs >= w && rs - w >= slice_address && same_tile(rs - w)).then(|| rs - w);
            let start = self.bits_mark(e);
            let syntax = sao::parse(
                e,
                &mut self.ctx,
                self.pic,
                rs,
                sao::Neighbours {
                    left,
                    up,
                    enabled: header.sao,
                },
            );
            if self.stats.is_some() {
                self.sao_record(e, rs, (x0, y0), syntax, start);
            }
        }
        self.quadtree(e, (x0, y0), self.pic.layout.log2_ctb, 0)
    }

    /// `coding_quadtree()` (§7.3.8.4).
    fn quadtree(
        &mut self,
        e: &mut Engine<'_>,
        (x, y): (usize, usize),
        log2: u8,
        depth: u8,
    ) -> Result<(), DecodeError> {
        let sps = self.cx.sps;
        let size = 1usize << log2;
        let split = if x + size <= sps.width && y + size <= sps.height && log2 > sps.log2_min_cb {
            let deeper = |b: Option<&BlockInfo>| b.is_some_and(|b| b.depth > depth);
            let inc =
                usize::from(deeper(self.left_of((x, y)))) + usize::from(deeper(self.above((x, y))));
            e.bin(self.ctx.at(ctx::SPLIT_CU, inc))
        } else {
            log2 > sps.log2_min_cb
        };
        let qg_log2 = sps.log2_ctb - self.cx.pps.cu_qp_delta_depth.unwrap_or(0);
        if log2 >= qg_log2 {
            self.start_quant_group((x, y));
        }
        if split {
            let half = size / 2;
            for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
                if x + dx < sps.width && y + dy < sps.height {
                    self.quadtree(e, (x + dx, y + dy), log2 - 1, depth + 1)?;
                }
            }
            Ok(())
        } else {
            self.coding_unit(e, (x, y), log2, depth)
        }
    }

    /// The block information left of luma `(x, y)` if it is available
    /// (§6.4.1). Inside the current CTB it always is: the left neighbour
    /// comes earlier in z-order, and slices and tiles start at CTBs; only
    /// across a CTB edge does the full availability process run.
    #[inline]
    fn left_of(&self, (x, y): (usize, usize)) -> Option<&BlockInfo> {
        let ctb_mask = (1usize << self.pic.layout.log2_ctb) - 1;
        let inside = x & ctb_mask != 0;
        (inside
            || self
                .pic
                .available((x, y), (x as isize - 1, y as isize), self.cx.slice))
        .then(|| &self.pic.info[self.pic.block(x - 1, y)])
    }

    /// As [`Self::left_of`], for the block above.
    #[inline]
    fn above(&self, (x, y): (usize, usize)) -> Option<&BlockInfo> {
        let ctb_mask = (1usize << self.pic.layout.log2_ctb) - 1;
        let inside = y & ctb_mask != 0;
        (inside
            || self
                .pic
                .available((x, y), (x as isize, y as isize - 1), self.cx.slice))
        .then(|| &self.pic.info[self.pic.block(x, y - 1)])
    }

    /// Set the `size`x`size` area at luma `(x, y)` to `info`: its first row
    /// of 4x4 blocks, then copies of that row (a whole 64x64 coding unit
    /// stores 16 rows of 16 blocks).
    fn store_info(&mut self, (x, y): (usize, usize), size: usize, info: BlockInfo) {
        let blocks = size >> GRID_LOG2;
        let first = self.pic.block(x, y);
        self.pic.info[first..first + blocks].fill(info);
        for row in 1..blocks {
            let start = self.pic.block(x, y + (row << GRID_LOG2));
            self.pic.info.copy_within(first..first + blocks, start);
        }
    }

    /// `qPY_PRED` of the quantization group at `(x, y)` (§8.6.1).
    fn start_quant_group(&mut self, (x, y): (usize, usize)) {
        let ctb_mask = (1usize << self.pic.layout.log2_ctb) - 1;
        let prev = self.last_qp;
        let qp_at = |xn: usize, yn: usize| i32::from(self.pic.info[self.pic.block(xn, yn)].qp);
        // Neighbours count only inside the current CTB.
        let a = if x & ctb_mask != 0 {
            qp_at(x - 1, y)
        } else {
            prev
        };
        let b = if y & ctb_mask != 0 {
            qp_at(x, y - 1)
        } else {
            prev
        };
        self.qg = QuantGroup {
            pred: (a + b + 1) >> 1,
            delta: 0,
            coded: false,
        };
        self.qp = self.qg.pred;
    }

    fn update_qp(&mut self) {
        self.qp = (self.qg.pred + self.qg.delta + QP_WRAP).rem_euclid(QP_WRAP);
    }

    /// `coding_unit()` (§7.3.8.5).
    fn coding_unit(
        &mut self,
        e: &mut Engine<'_>,
        (x, y): (usize, usize),
        log2: u8,
        depth: u8,
    ) -> Result<(), DecodeError> {
        let cx = self.cx;
        let sps = cx.sps;
        let size = 1usize << log2;
        let slice = cx.slice;
        if self.stats.is_some() {
            self.analysis = Analysis {
                cu_start: e.bits_consumed(),
                ..Analysis::default()
            };
        }
        let bypass = cx.pps.transquant_bypass && e.bin(self.ctx.at(ctx::TRANSQUANT_BYPASS, 0));
        let inter_slice = cx.header.slice_type != SliceType::I;
        let skip = inter_slice && {
            let skipped = |b: Option<&BlockInfo>| b.is_some_and(|b| b.flags & flag::SKIP != 0);
            let inc = usize::from(skipped(self.left_of((x, y))))
                + usize::from(skipped(self.above((x, y))));
            e.bin(self.ctx.at(ctx::SKIP, inc))
        };
        self.update_qp();
        let mut cu = Cu {
            x,
            y,
            log2,
            intra: false,
            part: Part::P2Nx2N,
            bypass,
            luma_modes: [predict::DC; 4],
            chroma_mode: predict::DC,
        };
        if !skip {
            cu.intra = !inter_slice || e.bin(self.ctx.at(ctx::PRED_MODE, 0));
            if !cu.intra || log2 == sps.log2_min_cb {
                cu.part = self.part_mode(e, &cu)?;
            }
        }
        let mut flags = 0;
        if cu.intra {
            flags |= flag::INTRA;
        }
        if skip {
            flags |= flag::SKIP;
        }
        if bypass {
            flags |= flag::NO_FILTER;
        }
        let info = BlockInfo {
            slice,
            depth,
            flags,
            intra_mode: predict::DC,
            qp: self.qp as i8,
        };
        self.store_info((x, y), size, info);
        self.pic.mark_edges((x, y), (size, size), Edges::Both);
        if cu.intra {
            // Picture state is not cleared between pictures: intra blocks
            // predict from neither list.
            self.pic.set_intra_motion((x, y), size);
        }
        let predicted_qp = self.qp;
        if skip {
            let pu = (0, 0, size, size);
            self.prediction_unit(e, &cu, pu, 0, depth, true)?;
        } else if cu.intra {
            let pcm = cu.part == Part::P2Nx2N
                && sps
                    .pcm
                    .is_some_and(|p| (p.log2_min..=p.log2_max).contains(&log2))
                && e.terminate();
            if pcm {
                *e = self.pcm(e, &cu)?;
            } else {
                self.intra_modes(e, &mut cu);
                self.transform_root(e, &cu)?;
            }
        } else {
            let mut merge_2nx2n = false;
            let (blocks, count) = cu.part.blocks(size);
            for (i, &pu) in blocks[..count].iter().enumerate() {
                let merged = self.prediction_unit(e, &cu, pu, i, depth, false)?;
                merge_2nx2n = merged && cu.part == Part::P2Nx2N;
            }
            let coded = merge_2nx2n || e.bin(self.ctx.at(ctx::RQT_ROOT_CBF, 0));
            if coded {
                self.transform_root(e, &cu)?;
            }
        }
        if self.qp != predicted_qp {
            // A delta coded in this CU changed its QpY.
            let qp = self.qp as i8;
            self.pic.fill((x, y), (size, size), |b| b.qp = qp);
        }
        self.last_qp = self.qp;
        if self.stats.is_some() {
            self.cu_record(e, &cu, skip, depth);
        }
        Ok(())
    }

    /// Push a CTB's SAO record (`start`: the bits consumed before it).
    #[cold]
    #[inline(never)]
    fn sao_record(
        &mut self,
        e: &Engine<'_>,
        rs: usize,
        (x0, y0): (usize, usize),
        syntax: SaoSyntax,
        start: usize,
    ) {
        let sps = self.cx.sps;
        let size = 1usize << self.pic.layout.log2_ctb;
        let ctb = self.pic.sao[rs];
        self.record(BlockRecord {
            kind: RecordKind::Filter,
            x: coord(x0),
            y: coord(y0),
            w: coord(size.min(sps.width - x0)),
            h: coord(size.min(sps.height - y0)),
            mode: sao::kind_index(ctb[0].kind),
            mode2: sao::kind_index(ctb[1].kind),
            class: syntax as u8,
            bits: self.bits_since(e, start),
            ..BlockRecord::default()
        });
    }

    /// Push the intra prediction blocks' records of a coding unit.
    #[cold]
    #[inline(never)]
    fn intra_records(&mut self, cu: &Cu, parts: usize, pb: usize) {
        for (i, &mode) in cu.luma_modes[..parts].iter().enumerate() {
            self.record(BlockRecord {
                kind: RecordKind::Prediction,
                pred: Pred::Intra,
                x: coord(cu.x + (i & 1) * pb),
                y: coord(cu.y + (i >> 1) * pb),
                w: coord(pb),
                h: coord(pb),
                mode,
                mode2: cu.chroma_mode,
                class: intra_class(mode) as u8,
                ref_idx: [-1; 2],
                ..BlockRecord::default()
            });
        }
    }

    /// Push a prediction unit's record: how its motion was signalled
    /// (skip, merge, and the merge or predictor index) and its vectors.
    #[cold]
    #[inline(never)]
    fn pu_record(
        &mut self,
        (x, y, w, h): (usize, usize, usize, usize),
        (skip, merge, index): (bool, bool, usize),
        motion: &PuMotion,
        mvd: [Mv; 2],
        mvd_bits: u32,
    ) {
        let pred = match (skip, merge) {
            (true, _) => Pred::Skip,
            (false, true) => Pred::Merge,
            (false, false) => Pred::Inter,
        };
        let bi = motion.uses(0) && motion.uses(1);
        let vector = |mv: Mv| [mv.x, mv.y];
        self.record(BlockRecord {
            kind: RecordKind::Prediction,
            pred,
            x: coord(x),
            y: coord(y),
            w: coord(w),
            h: coord(h),
            mode: index as u8,
            flags: if bi { flags::BI } else { 0 },
            ref_idx: [0, 1].map(|l| {
                if motion.uses(l) {
                    motion.ref_idx[l]
                } else {
                    -1
                }
            }),
            mv: motion.mv.map(vector),
            mvd: mvd.map(vector),
            bits: mvd_bits,
            ..BlockRecord::default()
        });
    }

    /// Push a coding unit's record.
    #[cold]
    #[inline(never)]
    fn cu_record(&mut self, e: &Engine<'_>, cu: &Cu, skip: bool, depth: u8) {
        let size = 1usize << cu.log2;
        let pred = match (cu.intra, skip) {
            (true, _) => Pred::Intra,
            (false, true) => Pred::Skip,
            (false, false) => Pred::Inter,
        };
        let a = self.analysis;
        let qp_flag = if a.qp_delta.is_some() {
            flags::QP_DELTA_CODED
        } else {
            0
        };
        self.record(BlockRecord {
            kind: RecordKind::Coding,
            pred,
            shape: cu.part.shape(),
            depth,
            x: coord(cu.x),
            y: coord(cu.y),
            w: coord(size),
            h: coord(size),
            flags: a.cu_flags | qp_flag,
            qp: self.qp as i8,
            qp_delta: a.qp_delta.unwrap_or(0),
            bits: self.bits_since(e, a.cu_start),
            ..BlockRecord::default()
        });
    }

    /// `part_mode` (§9.3.3.7, Table 9-43).
    fn part_mode(&mut self, e: &mut Engine<'_>, cu: &Cu) -> Result<Part, DecodeError> {
        let sps = self.cx.sps;
        if e.bin(self.ctx.at(ctx::PART_MODE, 0)) {
            return Ok(Part::P2Nx2N);
        }
        if cu.intra {
            if cu.log2 == 3 && sps.log2_min_tb > 2 {
                return Err(DecodeError::Malformed("intra NxN without 4x4 transforms"));
            }
            return Ok(Part::PNxN);
        }
        let horizontal = e.bin(self.ctx.at(ctx::PART_MODE, 1));
        if cu.log2 == sps.log2_min_cb {
            if horizontal {
                return Ok(Part::P2NxN);
            }
            if cu.log2 == 3 {
                return Ok(Part::PNx2N);
            }
            return Ok(if e.bin(self.ctx.at(ctx::PART_MODE, 2)) {
                Part::PNx2N
            } else {
                Part::PNxN
            });
        }
        if !sps.amp {
            return Ok(if horizontal { Part::P2NxN } else { Part::PNx2N });
        }
        // AMP: bin 2 (context 3) says symmetric, bin 3 (bypass) which side.
        let symmetric = e.bin(self.ctx.at(ctx::PART_MODE, 3));
        Ok(match (horizontal, symmetric) {
            (true, true) => Part::P2NxN,
            (false, true) => Part::PNx2N,
            (true, false) => {
                if e.bypass() {
                    Part::P2NxnD
                } else {
                    Part::P2NxnU
                }
            }
            (false, false) => {
                if e.bypass() {
                    Part::PnRx2N
                } else {
                    Part::PnLx2N
                }
            }
        })
    }

    /// `pcm_sample()` (§7.3.8.7): raw samples, then the arithmetic decoder
    /// restarts after them (§9.3.2.6).
    fn pcm<'d>(&mut self, e: &Engine<'d>, cu: &Cu) -> Result<Engine<'d>, DecodeError> {
        let pcm = self.cx.sps.pcm.ok_or(DecodeError::Malformed("pcm_flag"))?;
        let data = e.data();
        let start = e.next_byte();
        let mut r = BitReader::new(data.get(start..).ok_or(DecodeError::Truncated)?);
        let size = 1usize << cu.log2;
        for (c, plane) in PlaneIndex::ALL.into_iter().enumerate() {
            let (n, bits) = if c == 0 {
                (size, pcm.bit_depth_luma)
            } else {
                (size / 2, pcm.bit_depth_chroma)
            };
            let (px, py) = if c == 0 {
                (cu.x, cu.y)
            } else {
                (cu.x / 2, cu.y / 2)
            };
            let mut view = self.pic.samples.plane_mut(plane).view_mut();
            for row in 0..n {
                let line = &mut view.row_mut(py + row)[px..px + n];
                for s in line {
                    *s = (r.read_bits(bits)? << (8 - bits)) as u8;
                }
            }
        }
        let size_bits = r.position();
        let mut flags = 0;
        if pcm.loop_filter_disabled {
            flags |= flag::NO_FILTER;
        }
        self.pic
            .fill((cu.x, cu.y), (size, size), |b| b.flags |= flags);
        Engine::new(data, start + size_bits.div_ceil(8))
    }

    /// Luma and chroma intra modes (§7.3.8.5, §8.4.2, §8.4.3).
    fn intra_modes(&mut self, e: &mut Engine<'_>, cu: &mut Cu) {
        let size = 1usize << cu.log2;
        let parts = if cu.part == Part::PNxN { 4 } else { 1 };
        let pb = if parts == 4 { size / 2 } else { size };
        let mut from_list = [false; 4];
        for f in &mut from_list[..parts] {
            *f = e.bin(self.ctx.at(ctx::PREV_INTRA_LUMA, 0));
        }
        let log2_ctb = self.pic.layout.log2_ctb;
        for (i, &listed) in from_list.iter().enumerate().take(parts) {
            let (x, y) = (cu.x + (i & 1) * pb, cu.y + (i >> 1) * pb);
            let left = self.left_of((x, y)).map(|b| b.intra_mode);
            // The above neighbour counts only inside the current CTB row.
            let above = if (y >> log2_ctb) << log2_ctb == y {
                None
            } else {
                self.above((x, y)).map(|b| b.intra_mode)
            };
            let mpm = predict::most_probable(left, above);
            let mode = if listed {
                let index = if e.bypass() {
                    1 + usize::from(e.bypass())
                } else {
                    0
                };
                mpm[index]
            } else {
                let mut mode = e.bypass_bits(5) as u8;
                let mut sorted = mpm;
                sorted.sort_unstable();
                for m in sorted {
                    if mode >= m {
                        mode += 1;
                    }
                }
                mode
            };
            cu.luma_modes[i] = mode;
            self.pic.fill((x, y), (pb, pb), |b| b.intra_mode = mode);
        }
        if parts == 1 {
            cu.luma_modes = [cu.luma_modes[0]; 4];
        }
        let chroma = if e.bin(self.ctx.at(ctx::CHROMA_PRED, 0)) {
            e.bypass_bits(2) as u8
        } else {
            predict::CHROMA_DM
        };
        cu.chroma_mode = predict::chroma_mode(chroma, cu.luma_modes[0]);
        if self.stats.is_some() {
            self.intra_records(cu, parts, pb);
        }
    }

    /// `prediction_unit()` (§7.3.8.6) and its motion compensation. Returns
    /// `merge_flag`.
    fn prediction_unit(
        &mut self,
        e: &mut Engine<'_>,
        cu: &Cu,
        (dx, dy, w, h): (usize, usize, usize, usize),
        part_idx: usize,
        depth: u8,
        skip: bool,
    ) -> Result<bool, DecodeError> {
        let cx = self.cx;
        let header = cx.header;
        let (x, y) = (cu.x + dx, cu.y + dy);
        let merge = skip || e.bin(self.ctx.at(ctx::MERGE_FLAG, 0));
        let pu = Pu {
            cb: (cu.x, cu.y),
            cb_size: 1 << cu.log2,
            x,
            y,
            w,
            h,
            part_idx,
            vertical_split: cu.part.vertical_split(),
            horizontal_split: cu.part.horizontal_split(),
        };
        // The merge index, or the list-0 predictor index (analysis).
        let index;
        let mut mvd = [Mv::default(); 2];
        let mut mvd_bits = 0;
        let motion = if merge {
            index = self.merge_idx(e);
            self.predictor().merge(&pu, index)
        } else {
            let b = header.slice_type == SliceType::B;
            // inter_pred_idc: 0 L0, 1 L1, 2 BI.
            let idc = if !b {
                0
            } else if w + h != 12 && e.bin(self.ctx.at(ctx::INTER_PRED_IDC, usize::from(depth))) {
                2
            } else {
                usize::from(e.bin(self.ctx.at(ctx::INTER_PRED_IDC, ctx::INTER_PRED_IDC_LAST)))
            };
            let mut motion = PuMotion::NONE;
            let mut mvp_flag = [0; 2];
            for list in 0..2 {
                if idc != 2 && idc != list {
                    continue;
                }
                let ref_idx = self.ref_idx(e, header.num_ref_idx[list]);
                motion.ref_idx[list] = ref_idx as i8;
                if !(list == 1 && header.mvd_l1_zero && idc == 2) {
                    let start = self.bits_mark(e);
                    mvd[list] = self.mvd(e)?;
                    mvd_bits += self.bits_since(e, start);
                }
                mvp_flag[list] = usize::from(e.bin(self.ctx.at(ctx::MVP_FLAG, 0)));
            }
            index = mvp_flag[0];
            for list in 0..2 {
                if motion.uses(list) {
                    let predictor = self.predictor();
                    let mvp =
                        predictor.amvp(&pu, list, motion.ref_idx[list] as usize, mvp_flag[list]);
                    motion.mv[list] = Mv {
                        x: mvp.x.wrapping_add(mvd[list].x),
                        y: mvp.y.wrapping_add(mvd[list].y),
                    };
                }
            }
            motion
        };
        self.pic.set_motion((x, y), (w, h), motion);
        if self.stats.is_some() {
            self.pu_record((x, y, w, h), (skip, merge, index), &motion, mvd, mvd_bits);
        }
        if cu.part != Part::P2Nx2N {
            // A 2Nx2N block's edges are its coding block's, already marked.
            self.pic.mark_edges((x, y), (w, h), Edges::Prediction);
        }
        self.motion_compensate((x, y), (w, h), &motion)?;
        Ok(merge)
    }

    fn predictor(&self) -> Predictor<'_> {
        let cx = self.cx;
        let header = cx.header;
        Predictor {
            pic: self.pic,
            slice: cx.slice,
            lists: &self.pic.slices[usize::from(cx.slice - 1)].lists,
            poc: cx.poc,
            col: cx.col,
            collocated_from_l0: header.collocated_from_l0,
            no_backward_pred: cx.no_backward_pred,
            log2_par_mrg_level: cx.pps.log2_parallel_merge_level,
            max_merge: header.max_merge_candidates,
            b_slice: header.slice_type == SliceType::B,
            num_ref_idx: header.num_ref_idx,
        }
    }

    fn merge_idx(&mut self, e: &mut Engine<'_>) -> usize {
        let max = self.cx.header.max_merge_candidates;
        if max <= 1 || !e.bin(self.ctx.at(ctx::MERGE_IDX, 0)) {
            return 0;
        }
        let mut index = 1;
        while index < max - 1 && e.bypass() {
            index += 1;
        }
        index
    }

    /// `ref_idx_lX` (truncated Rice, two context bins).
    fn ref_idx(&mut self, e: &mut Engine<'_>, count: usize) -> usize {
        let max = count.saturating_sub(1);
        let mut index = 0;
        while index < max {
            let bin = if index < 2 {
                e.bin(self.ctx.at(ctx::REF_IDX, index))
            } else {
                e.bypass()
            };
            if !bin {
                break;
            }
            index += 1;
        }
        index
    }

    /// `mvd_coding()` (§7.3.8.9).
    fn mvd(&mut self, e: &mut Engine<'_>) -> Result<Mv, DecodeError> {
        let greater0 = [
            e.bin(self.ctx.at(ctx::ABS_MVD_GT0, 0)),
            e.bin(self.ctx.at(ctx::ABS_MVD_GT0, 0)),
        ];
        let mut greater1 = [false; 2];
        for i in 0..2 {
            if greater0[i] {
                greater1[i] = e.bin(self.ctx.at(ctx::ABS_MVD_GT1, 0));
            }
        }
        let mut v = [0i32; 2];
        for i in 0..2 {
            if greater0[i] {
                let magnitude = if greater1[i] {
                    i64::from(e.exp_golomb(1)?) + 2
                } else {
                    1
                };
                if magnitude > 1 << 15 {
                    return Err(DecodeError::Malformed("motion vector difference"));
                }
                v[i] = if e.bypass() {
                    -(magnitude as i32)
                } else {
                    magnitude as i32
                };
                if v[i] > i32::from(i16::MAX) {
                    return Err(DecodeError::Malformed("motion vector difference"));
                }
            }
        }
        Ok(Mv {
            x: v[0] as i16,
            y: v[1] as i16,
        })
    }

    /// Weighted sample prediction of one prediction block (§8.5.3.3).
    fn motion_compensate(
        &mut self,
        (x, y): (usize, usize),
        (w, h): (usize, usize),
        motion: &PuMotion,
    ) -> Result<(), DecodeError> {
        let header = self.cx.header;
        let weighted = header.weights.as_ref();
        // At most one source per list, kept on the stack.
        let mut sources: [Option<Source<'_>>; 2] = [None, None];
        let mut count = 0;
        for list in 0..2 {
            if !motion.uses(list) {
                continue;
            }
            let index = motion.ref_idx[list] as usize;
            let picture = *self.cx.refs[list]
                .get(index)
                .ok_or(DecodeError::Malformed("reference index"))?;
            sources[count] = Some(Source {
                picture,
                mv: motion.mv[list],
                weight: weighted.and_then(|w| w.lists[list].get(index).copied()),
            });
            count += 1;
        }
        let sources = match sources {
            [Some(a), Some(b)] => [a, b],
            [Some(a), None] => [a, a],
            _ => return Err(DecodeError::Malformed("prediction block without motion")),
        };
        let sources = &sources[..count];
        let denominators = weighted.map_or([0; 2], |w| w.log2_denom);
        mc::predict(
            self.cx.dsp,
            &mut self.s.buffers,
            &mut self.pic.samples,
            (x, y),
            (w, h),
            sources,
            denominators,
        );
        Ok(())
    }

    /// The transform tree of a coding unit (§7.3.8.8).
    fn transform_root(&mut self, e: &mut Engine<'_>, cu: &Cu) -> Result<(), DecodeError> {
        let node = TuNode {
            x: cu.x,
            y: cu.y,
            base: (cu.x, cu.y),
            log2: cu.log2,
            depth: 0,
            blk_idx: 0,
            parent_cbf: [true; 2],
        };
        self.transform_tree(e, cu, node)
    }

    fn transform_tree(
        &mut self,
        e: &mut Engine<'_>,
        cu: &Cu,
        node: TuNode,
    ) -> Result<(), DecodeError> {
        let sps = self.cx.sps;
        let intra_split = cu.intra && cu.part == Part::PNxN;
        let max_depth = if cu.intra {
            sps.max_depth_intra + u8::from(intra_split)
        } else {
            sps.max_depth_inter
        };
        let inter_split =
            sps.max_depth_inter == 0 && !cu.intra && cu.part != Part::P2Nx2N && node.depth == 0;
        let log2 = node.log2;
        let split = if log2 <= sps.log2_max_tb
            && log2 > sps.log2_min_tb
            && node.depth < max_depth
            && !(intra_split && node.depth == 0)
        {
            e.bin(self.ctx.at(ctx::SPLIT_TRANSFORM, usize::from(5 - log2)))
        } else {
            log2 > sps.log2_max_tb || (intra_split && node.depth == 0) || inter_split
        };
        let mut cbf = [false; 2];
        if log2 > 2 {
            for (c, f) in cbf.iter_mut().enumerate() {
                if node.depth == 0 || node.parent_cbf[c] {
                    *f = e.bin(self.ctx.at(ctx::CBF_CHROMA, usize::from(node.depth)));
                }
            }
        } else {
            cbf = node.parent_cbf;
        }
        if split {
            let half = 1usize << (log2 - 1);
            for (i, (dx, dy)) in [(0, 0), (half, 0), (0, half), (half, half)]
                .into_iter()
                .enumerate()
            {
                let child = TuNode {
                    x: node.x + dx,
                    y: node.y + dy,
                    base: (node.x, node.y),
                    log2: log2 - 1,
                    depth: node.depth + 1,
                    blk_idx: i,
                    parent_cbf: cbf,
                };
                self.transform_tree(e, cu, child)?;
            }
            return Ok(());
        }
        let luma_cbf = if cu.intra || node.depth != 0 || cbf[0] || cbf[1] {
            e.bin(self.ctx.at(ctx::CBF_LUMA, usize::from(node.depth == 0)))
        } else {
            true
        };
        self.transform_unit(e, cu, node, luma_cbf, cbf)
    }

    /// `transform_unit()` (§7.3.8.10) with its reconstruction.
    fn transform_unit(
        &mut self,
        e: &mut Engine<'_>,
        cu: &Cu,
        node: TuNode,
        luma_cbf: bool,
        cbf: [bool; 2],
    ) -> Result<(), DecodeError> {
        let size = 1usize << node.log2;
        if (luma_cbf || cbf[0] || cbf[1])
            && self.cx.pps.cu_qp_delta_depth.is_some()
            && !self.qg.coded
        {
            self.cu_qp_delta(e)?;
        }
        if node.depth > 0 {
            // The root's edges are its coding block's, already marked.
            self.pic
                .mark_edges((node.x, node.y), (size, size), Edges::Transform);
        }
        if luma_cbf {
            self.pic
                .fill((node.x, node.y), (size, size), |b| b.flags |= flag::CODED);
        }
        let quadrant = usize::from(node.x - cu.x >= (1 << cu.log2) / 2)
            + 2 * usize::from(node.y - cu.y >= (1 << cu.log2) / 2);
        let luma_mode = cu
            .intra
            .then(|| cu.luma_modes[if cu.part == Part::PNxN { quadrant } else { 0 }]);
        if self.stats.is_some() {
            // Nothing stays live across the blocks when analysis is off.
            self.start_tu(e);
        }
        self.block(e, cu, (node.x, node.y), node.log2, 0, luma_mode, luma_cbf)?;
        let chroma_mode = cu.intra.then_some(cu.chroma_mode);
        if node.log2 > 2 {
            for c in 1..3 {
                self.block(
                    e,
                    cu,
                    (node.x / 2, node.y / 2),
                    node.log2 - 1,
                    c,
                    chroma_mode,
                    cbf[c - 1],
                )?;
            }
        } else if node.blk_idx == 3 {
            for c in 1..3 {
                self.block(
                    e,
                    cu,
                    (node.base.0 / 2, node.base.1 / 2),
                    2,
                    c,
                    chroma_mode,
                    cbf[c - 1],
                )?;
            }
        }
        if self.stats.is_some() {
            let bits = self.bits_since(e, self.analysis.tu.start);
            self.tu_record(cu, node, (luma_cbf, cbf), bits);
        }
        Ok(())
    }

    /// Start collecting a transform unit's residual (analysis).
    #[cold]
    #[inline(never)]
    fn start_tu(&mut self, e: &Engine<'_>) {
        self.analysis.tu = CodedBlock {
            start: e.bits_consumed(),
            ..CodedBlock::default()
        };
    }

    /// Add a coded block's residual to its transform unit's (analysis).
    #[cold]
    #[inline(never)]
    fn note_block(&mut self, transform_skip: bool) {
        let tu = &mut self.analysis.tu;
        tu.nonzero = tu.nonzero.saturating_add(self.s.positions.len() as u16);
        tu.transform_skip |= transform_skip;
    }

    /// Push a transform unit's record. A 4x4 luma block's chroma is coded
    /// with the fourth of its quadrant (§7.3.8.10), so only that one
    /// carries the chroma flags.
    #[cold]
    #[inline(never)]
    fn tu_record(&mut self, cu: &Cu, node: TuNode, (luma_cbf, cbf): (bool, [bool; 2]), bits: u32) {
        let residual = self.analysis.tu;
        let chroma = node.log2 > 2 || node.blk_idx == 3;
        let mut coded = if luma_cbf { flags::CODED_Y } else { 0 };
        if chroma && cbf[0] {
            coded |= flags::CODED_CB;
        }
        if chroma && cbf[1] {
            coded |= flags::CODED_CR;
        }
        self.analysis.cu_flags |= coded;
        let size = coord(1 << node.log2);
        self.record(BlockRecord {
            kind: RecordKind::Transform,
            pred: if cu.intra { Pred::Intra } else { Pred::Inter },
            depth: node.depth,
            x: coord(node.x),
            y: coord(node.y),
            w: size,
            h: size,
            flags: coded
                | if residual.transform_skip {
                    flags::TRANSFORM_SKIP
                } else {
                    0
                },
            qp: self.qp as i8,
            nonzero: residual.nonzero,
            bits,
            ..BlockRecord::default()
        });
    }

    /// `cu_qp_delta_abs` and `cu_qp_delta_sign_flag` (§7.3.8.14).
    fn cu_qp_delta(&mut self, e: &mut Engine<'_>) -> Result<(), DecodeError> {
        let mut magnitude = 0u32;
        while magnitude < QP_DELTA_PREFIX
            && e.bin(self.ctx.at(ctx::CU_QP_DELTA, usize::from(magnitude > 0)))
        {
            magnitude += 1;
        }
        if magnitude == QP_DELTA_PREFIX {
            magnitude += e.exp_golomb(0)?;
        }
        let mut delta =
            i32::try_from(magnitude).map_err(|_| DecodeError::Malformed("cu_qp_delta_abs"))?;
        if delta != 0 && e.bypass() {
            delta = -delta;
        }
        if !QP_DELTA_RANGE.contains(&delta) {
            return Err(DecodeError::Malformed("CuQpDeltaVal"));
        }
        self.qg.delta = delta;
        self.qg.coded = true;
        self.analysis.qp_delta = Some(delta as i8);
        self.update_qp();
        Ok(())
    }

    /// Predict (intra), parse and reconstruct one transform block of
    /// component `c` at plane position `(x, y)`.
    #[allow(clippy::too_many_arguments)]
    fn block(
        &mut self,
        e: &mut Engine<'_>,
        cu: &Cu,
        (x, y): (usize, usize),
        log2: u8,
        c: usize,
        intra_mode: Option<u8>,
        coded: bool,
    ) -> Result<(), DecodeError> {
        let n = 1usize << log2;
        if let Some(mode) = intra_mode {
            self.intra_predict((x, y), n, c, mode);
        }
        if !coded {
            return Ok(());
        }
        let scan_kind =
            intra_mode.map_or(ScanKind::Diagonal, |m| ScanKind::for_intra(m, log2, c == 0));
        let scan = self.cx.scans.get(log2, scan_kind);
        let pps = self.cx.pps;
        let options = super::residual::Options {
            luma: c == 0,
            sign_hiding: pps.sign_hiding && !cu.bypass,
            transform_skip: pps.transform_skip && !cu.bypass,
        };
        self.s.positions.clear();
        let count = n * n;
        let skip = super::residual::read(
            e,
            &mut self.ctx,
            scan,
            options,
            &mut self.s.levels[..count],
            &mut self.s.positions,
        );
        let Ok(skip) = skip else {
            // Keep the all-zero invariant for the next block.
            for &p in &self.s.positions {
                self.s.levels[usize::from(p)] = 0;
            }
            return skip.map(|_| ());
        };
        if self.stats.is_some() {
            self.note_block(skip);
        }
        self.reconstruct(cu, (x, y), log2, c, skip, intra_mode.is_some());
        for &p in &self.s.positions {
            self.s.levels[usize::from(p)] = 0;
        }
        Ok(())
    }

    /// Scaling and transformation of the parsed levels (§8.6.2–§8.6.4),
    /// added to the prediction in the picture. DCT and DST blocks go
    /// through the DSP's fused inverse-and-add (all-zero, DC-only and
    /// zero-region shortcuts inside); transform skip and lossless bypass,
    /// both rare, build the residual here.
    fn reconstruct(
        &mut self,
        cu: &Cu,
        (x, y): (usize, usize),
        log2: u8,
        c: usize,
        transform_skip: bool,
        intra: bool,
    ) {
        let n = 1usize << log2;
        let count = n * n;
        let extent = if cu.bypass {
            (n, n)
        } else {
            self.scale_levels(log2, c, intra)
        };
        let mut dst = self
            .pic
            .samples
            .plane_mut(PlaneIndex::ALL[c])
            .region_mut(x as isize, y as isize, n, n)
            .expect("transform blocks lie inside the picture");
        if !cu.bypass && !transform_skip {
            let kind =
                crate::transform::kind(log2, intra && c == 0).expect("transform sizes 4..32");
            self.cx
                .dsp
                .hevc_inverse_add(kind, &self.s.levels[..count], extent, &mut dst);
            return;
        }
        let residual = &mut self.s.residual[..count];
        if cu.bypass {
            residual.copy_from_slice(&self.s.levels[..count]);
        } else {
            for (r, &level) in residual.iter_mut().zip(&self.s.levels[..count]) {
                *r = ((level << TRANSFORM_SKIP_SHIFT) + (1 << (RESIDUAL_SHIFT - 1)))
                    >> RESIDUAL_SHIFT;
            }
        }
        for (row, line) in residual.chunks_exact(n).enumerate() {
            for (s, &r) in dst.row_mut(row).iter_mut().zip(line) {
                *s = (i32::from(*s) + r).clamp(0, 255) as u8;
            }
        }
    }

    /// Scale the parsed levels in place (§8.6.3, `scaling_list_enabled`
    /// factors or the flat 16), visiting only the coded positions. Returns
    /// the rows and columns that hold them, for the inverse transform.
    fn scale_levels(&mut self, log2: u8, c: usize, intra: bool) -> (usize, usize) {
        let qp = if c == 0 {
            self.qp
        } else {
            let offset = if c == 1 {
                self.cx.pps.cb_qp_offset + self.cx.header.cb_qp_offset
            } else {
                self.cx.pps.cr_qp_offset + self.cx.header.cr_qp_offset
            };
            chroma_qp(self.qp + offset)
        };
        let matrix = if intra { 0 } else { 3 } + c;
        let factors = self.cx.scaling().map(|s| s.factors(log2, matrix));
        // bdShift = BitDepth + log2(nTbS) + 10 − 15 (§8.6.3).
        let shift = u32::from(log2) + 3;
        let scale = LEVEL_SCALE[(qp % 6) as usize] << (qp / 6);
        let (mut rows, mut columns) = (0, 0);
        for &p in &self.s.positions {
            let p = usize::from(p);
            rows = rows.max((p >> log2) + 1);
            columns = columns.max((p & ((1 << log2) - 1)) + 1);
            let m = factors.map_or(i64::from(FLAT_SCALE), |f| i64::from(f[p]));
            let v = (i64::from(self.s.levels[p]) * m * scale + (1 << (shift - 1))) >> shift;
            self.s.levels[p] = v.clamp(-32768, 32767) as i32;
        }
        (rows, columns)
    }

    /// Intra sample prediction of an `n`x`n` block of component `c`
    /// (§8.4.4.2) written into the picture.
    fn intra_predict(&mut self, (x, y): (usize, usize), n: usize, c: usize, mode: u8) {
        let shift = usize::from(c > 0);
        let plane = PlaneIndex::ALL[c];
        let constrained = self.cx.pps.constrained_intra_pred;
        let pic = &*self.pic;
        let cur = (x << shift, y << shift);
        // Without constrained intra prediction, availability is uniform
        // along each side of the block (see `Refs::gather`).
        let sides = if constrained {
            Sides::PerUnit
        } else {
            Sides::Uniform
        };
        self.s.refs.gather_into(
            pic.samples.plane(plane),
            (x, y),
            n,
            4 >> shift,
            sides,
            |xn, yn| {
                let nb = ((xn << shift) as isize, (yn << shift) as isize);
                pic.available(cur, nb, self.cx.slice)
                    && (!constrained
                        || pic.info[pic.block(nb.0 as usize, nb.1 as usize)].flags & flag::INTRA
                            != 0)
            },
        );
        let strong = self.cx.sps.strong_intra_smoothing;
        let mut view = self
            .pic
            .samples
            .plane_mut(plane)
            .region_mut(x as isize, y as isize, n, n)
            .expect("intra block inside the picture");
        predict::predict_smoothed_into(
            self.cx.dsp,
            &self.s.refs,
            (mode, c == 0, strong),
            &mut self.s.pred,
            &mut view,
        );
    }
}
