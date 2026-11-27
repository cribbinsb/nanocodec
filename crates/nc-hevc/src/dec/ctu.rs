//! Slice segment data (H.265 §7.3.8): the CTU loop with tiles, WPP and
//! dependent slice segments (§9.3.1, §9.3.2), the coding quadtree, coding
//! units (with PCM and lossless bypass), prediction units, transform trees
//! with QP deltas (§8.6.1), and the reconstruction of each block (§8.4,
//! §8.5, §8.6).
//!
//! Chroma follows `ChromaArrayType` (§7.4.3.2.1): none in 4:0:0; in 4:2:2
//! each chroma transform block is two square blocks stacked vertically,
//! each with its own `cbf_cb` / `cbf_cr`, and chroma intra modes map
//! through Table 8-3; in 4:4:4 chroma transform blocks are the luma size
//! (4x4 included) and an `NxN` intra coding unit codes four chroma modes.

#![allow(
    clippy::many_single_char_names,
    reason = "H.265 sample and block variables keep their specification names (x, y, w, h, n)"
)]

use nc_common::decode::{BitReader, DecodeError};
#[cfg(feature = "analysis")]
use nc_common::stream_stats::StreamStats;
use nc_common::{BitDepths, ChromaFormat, Picture, PlaneIndex};
use nc_dsp::{Dsp, Pixel};
use nc_encoder::Mv;

use super::cabac::Engine;
use super::ctx::{self, Contexts};
use super::loop_filter::LoopFilter;
use super::mc::{self, Source};
use super::mvpred::{Collocated, Predictor, Pu};
use super::params::{FLAT_SCALE, MAX_QP, Pps, ScalingLists, Sps, chroma_qp, qp_bd_offset};
use super::picture::{BlockInfo, Edges, GRID_LOG2, PictureState, PuMotion, flag};
use super::residual::{Coded, LOG2_TRANSFORM_RANGE, RangeOptions};
use super::sao;
use super::slice::{SliceHeader, SliceSegment, SliceType};

// The analysis records (feature `analysis`); every hook below is compiled
// out without it.
#[cfg(feature = "analysis")]
mod analysis;
use crate::inter::Component;
use crate::predict::{self, IntraFilters, Refs, Sides};
use crate::quant::MAX_CHROMA_QPI;
use crate::scan::{ScanKind, Scans};

/// `levelScale` (§8.6.3).
const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];
/// Coefficients of the largest transform block.
const MAX_COEFFS: usize = 32 * 32;
/// `cu_qp_delta_abs` prefix length before the Exp-Golomb suffix.
const QP_DELTA_PREFIX: u32 = 5;
/// `CuQpDeltaVal` lies in `−(26 + QpBdOffsetY / 2)..=25 + QpBdOffsetY / 2`
/// (§7.4.9.14).
const QP_DELTA_BELOW: i32 = 26;
const QP_DELTA_ABOVE: i32 = 25;
/// QP values wrap modulo this plus `QpBdOffsetY` (equation 8-283).
const QP_WRAP: i32 = MAX_QP + 1;
/// `tsShift = 5 + Log2(nTbS)`: transform-skip residuals are scaled up to
/// the inverse transform's scale (§8.6.4.2; 7 for 4x4 as in version 1).
const TRANSFORM_SKIP_SHIFT_BELOW: u32 = 5;
/// The smallest transform, the only one scaling lists apply to when
/// transform-skipped and the one `transform_skip_rotation_enabled_flag`
/// rotates.
const MIN_TRANSFORM_LOG2: u8 = 2;
const ROTATED_SIZE: usize = 1 << MIN_TRANSFORM_LOG2;
/// Cross-component prediction scales by `ResScaleVal / 8` (§7.3.8.12).
const CROSS_COMPONENT_SHIFT: u32 = 3;
/// `log_res_scale_abs_plus1` is truncated unary with `cMax` 4 (§9.3.3).
const MAX_LOG_RES_SCALE: u32 = 4;
/// `bdShift = RESIDUAL_BITS − BitDepth` after the transform (or transform
/// skip, §8.6.2).
const RESIDUAL_BITS: u32 = 20;
/// `bdShift = BitDepth + Log2(nTbS) − SCALING_SHIFT_BELOW` of the scaling
/// process (§8.6.3: `+ 10 − 15`), and its `+ 10` before `−
/// log2TransformRange` with extended precision.
const SCALING_SHIFT_BELOW: u32 = 5;
const SCALING_SHIFT_ABOVE: u32 = 10;

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
pub(super) struct SliceContext<'a, T: Pixel> {
    pub sps: &'a Sps,
    pub pps: &'a Pps,
    pub header: &'a SliceHeader,
    /// The slice's index in the picture, plus one.
    pub slice: u16,
    pub refs: [Vec<&'a Picture<T>>; 2],
    pub col: Option<Collocated<'a>>,
    pub poc: i32,
    pub no_backward_pred: bool,
    pub dsp: &'a Dsp,
    pub scans: &'a Scans,
    /// The SPS's depths as the instance sees them (`BitDepths::of`).
    pub depths: BitDepths,
    /// Which instance of the residual parser blocks take.
    pub residual: ResidualPath,
    /// `Log2MaxTransformSkipSize` where `transform_skip_enabled_flag`, else
    /// 0 (no block codes `transform_skip_flag`).
    pub transform_skip_log2: u8,
    /// Residual arithmetic beyond 16 bits, `None` for the 16-bit path.
    pub wide: Option<WideRange>,
    /// The chroma geometry of the SPS.
    pub chroma: ChromaLayout,
}

/// The residual parser instance of a slice's blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResidualPath {
    /// Version 1: no range extension tool.
    Main,
    /// Range extension tools, 16-bit levels.
    Range,
    /// [`WideRange`] levels.
    Wide,
}

impl ResidualPath {
    pub fn of(sps: &Sps, pps: &Pps) -> Self {
        if WideRange::of(sps).is_some() {
            Self::Wide
        } else if sps.range.residual_tools()
            || pps.range.log2_max_transform_skip > MIN_TRANSFORM_LOG2
        {
            Self::Range
        } else {
            Self::Main
        }
    }
}

/// The chroma geometry of a sequence (`ChromaArrayType`, §7.4.3.2.1),
/// resolved once per slice for the block loops.
#[derive(Debug, Clone, Copy)]
pub(super) struct ChromaLayout {
    /// The chroma planes' format, `None` for 4:0:0.
    format: Option<ChromaFormat>,
    /// Log2 `SubWidthC`, `SubHeightC`.
    sx: usize,
    sy: usize,
    /// The smallest luma transform block coded with its own chroma blocks
    /// (8x8; 4x4 in 4:4:4; none in 4:0:0): smaller ones share their
    /// parent's (§7.3.8.8, §7.3.8.10).
    own_min_log2: u8,
    /// Chroma blocks per component of a transform unit: two in 4:2:2.
    blocks: usize,
    /// 4:4:4: chroma takes luma's mode-dependent scans and reference
    /// filtering.
    full: bool,
    /// How inter prediction addresses the chroma planes, `None` in 4:0:0.
    component: Option<Component>,
    /// The intra filters of luma and chroma blocks (§8.4.4.2), before
    /// `disableIntraBoundaryFilter`.
    intra_filters: [IntraFilters; 2],
}

/// `own_min_log2` of 4:0:0: no transform block has chroma.
const NO_CHROMA_LOG2: u8 = u8::MAX;
/// `own_min_log2` of the subsampled formats and of 4:4:4.
const SUBSAMPLED_CHROMA_MIN_LOG2: u8 = 3;
const FULL_CHROMA_MIN_LOG2: u8 = 2;

impl ChromaLayout {
    pub fn of(sps: &Sps) -> Self {
        let format = sps.chroma;
        let (sx, sy) = format.map_or((0, 0), ChromaFormat::subsampling);
        let full = format == Some(ChromaFormat::Yuv444);
        let smoothing = !sps.range.intra_smoothing_disabled;
        let filters = |luma: bool| IntraFilters {
            smooth: (luma || full) && smoothing,
            strong: luma && sps.strong_intra_smoothing,
            dc_edges: luma,
            angular_edges: luma,
        };
        Self {
            format,
            sx,
            sy,
            own_min_log2: match format {
                None => NO_CHROMA_LOG2,
                Some(_) if full => FULL_CHROMA_MIN_LOG2,
                Some(_) => SUBSAMPLED_CHROMA_MIN_LOG2,
            },
            blocks: if format == Some(ChromaFormat::Yuv422) {
                2
            } else {
                1
            },
            full,
            component: format.map(Component::chroma),
            intra_filters: [filters(true), filters(false)],
        }
    }
}

impl<T: Pixel> SliceContext<'_, T> {
    fn scaling(&self) -> Option<&ScalingLists> {
        self.pps.scaling.as_ref().or(self.sps.scaling.as_ref())
    }

    /// `ScalingFactor` of a block (`kind`: intra, transform skip), `None`
    /// for the flat 16: without scaling lists, and for transform-skip
    /// blocks above 4x4 (§8.6.4.2: m = 16).
    fn scaling_factors(
        &self,
        log2: u8,
        c: usize,
        (intra, transform_skip): (bool, bool),
    ) -> Option<&[u8]> {
        let matrix = if intra { 0 } else { 3 } + c;
        self.scaling()
            .filter(|_| !transform_skip || log2 == MIN_TRANSFORM_LOG2)
            .map(|s| s.factors(log2, matrix))
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
    /// `IntraPredModeY` and `IntraPredModeC` of the four `PART_NxN`
    /// partitions (all equal otherwise; chroma differs only in 4:4:4).
    luma_modes: [u8; 4],
    chroma_modes: [u8; 4],
    /// `intra_chroma_pred_mode` is 4 (DM) per partition: cross-component
    /// prediction applies (§7.3.8.12).
    chroma_dm: [bool; 4],
}

/// What a block's residual has to do with cross-component prediction
/// (§7.3.8.12): nothing, kept for the chroma blocks of its transform unit
/// (luma, when the PPS enables the tool), or scaled into a chroma
/// residual (`ResScaleVal`, not 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrossComponent {
    None,
    KeepLuma,
    Scale(i32),
}

/// Where a block's residual comes from in [`SegmentDecoder::reconstruct_general`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidualSource {
    /// No coded residual (a chroma block of cross-component prediction).
    Zero,
    /// `cu_transquant_bypass_flag`: the levels.
    Bypass,
    TransformSkip,
    Transform(nc_dsp::HevcTransform),
}

/// The direction RDPCM accumulates an untransformed residual in (§8.6.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rdpcm {
    Horizontal,
    Vertical,
}

/// Residual arithmetic beyond the 16-bit kernels (§7.4.9.11, §8.6.2 to
/// §8.6.4): with `extended_precision_processing_flag`, or samples deeper
/// than 12 bits. Levels are `i32` and the inverse transform is the 64-bit
/// normative one ([`crate::transform::inverse_extended`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct WideRange {
    /// `extended_precision_processing_flag`.
    extended: bool,
    /// `log2TransformRange` of luma and chroma: `CoeffMin = −2^range`.
    log2_range: [u32; 2],
    /// The inverse transform's second shift of luma and chroma: `Max(20 −
    /// BitDepth, extended ? 11 : 0)`.
    second_shift: [u32; 2],
}

/// The deepest samples the 16-bit residual path takes (the depth its
/// kernels and transform-skip shifts are sized for).
const NARROW_MAX_BIT_DEPTH: u32 = 12;
/// `log2TransformRange = Max(15, BitDepth + EXTENDED_RANGE_ABOVE_DEPTH)`
/// with extended precision (§7.4.9.11).
const EXTENDED_RANGE_ABOVE_DEPTH: u32 = 6;
/// The smallest second inverse-transform shift with extended precision
/// (§8.6.4.2).
const EXTENDED_MIN_SECOND_SHIFT: u32 = 11;

impl WideRange {
    /// The wide arithmetic of `sps`, `None` when the 16-bit path holds it.
    pub fn of(sps: &Sps) -> Option<Self> {
        let depths = [sps.depths.luma, sps.depths.chroma];
        let extended = sps.range.extended_precision;
        if !extended && depths.iter().all(|d| d.bits() <= NARROW_MAX_BIT_DEPTH) {
            return None;
        }
        Some(Self {
            extended,
            log2_range: depths.map(|d| {
                if extended {
                    LOG2_TRANSFORM_RANGE.max(d.bits() + EXTENDED_RANGE_ABOVE_DEPTH)
                } else {
                    LOG2_TRANSFORM_RANGE
                }
            }),
            second_shift: depths.map(|d| {
                let min = if extended {
                    EXTENDED_MIN_SECOND_SHIFT
                } else {
                    0
                };
                (RESIDUAL_BITS - d.bits()).max(min)
            }),
        })
    }

    /// `CoeffMin`, `CoeffMax` of component `c`.
    fn coeff_range(self, c: usize) -> (i32, i32) {
        let range = self.log2_range[usize::from(c > 0)];
        (-(1 << range), (1 << range) - 1)
    }

    /// `tsShift` and `bdShift` of a transform-skip block of side `2^log2`
    /// in component `c` (§8.6.4.2).
    fn transform_skip_shifts(self, c: usize, log2: u8) -> (u32, u32) {
        let second = self.second_shift[usize::from(c > 0)];
        let base = if self.extended {
            TRANSFORM_SKIP_SHIFT_BELOW.min(second - 2)
        } else {
            TRANSFORM_SKIP_SHIFT_BELOW
        };
        (base + u32::from(log2), second)
    }
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
    parent_cbf: ChromaCbf,
}

/// `cbf_cb` and `cbf_cr` of a transform node: per component, its top and
/// (4:2:2 only) bottom chroma block.
type ChromaCbf = [[bool; 2]; 2];

/// Whether any chroma block of a node is coded.
#[inline]
fn any_chroma(cbf: ChromaCbf) -> bool {
    cbf[0][0] || cbf[0][1] || cbf[1][0] || cbf[1][1]
}

/// Working memory of block decoding, kept by the decoder so that no
/// segment or block allocates or clears it. `levels` is all zero between
/// blocks: only the coded positions are set (and scaled in place) and
/// cleared again.
#[derive(Debug)]
pub(super) struct Scratch<T: Pixel> {
    /// Parsed, then scaled, levels: 16 bits by §7.4.9.11 (`CoeffMinY` to
    /// `CoeffMaxY`) and §8.6.3 (the scaled values' clamp).
    levels: Vec<i16>,
    /// `levels` of [`WideRange`] streams.
    wide_levels: Vec<i32>,
    positions: Vec<u16>,
    /// Residuals built in full (`reconstruct_general`), and the luma one
    /// cross-component prediction reads.
    residual: Vec<i32>,
    luma_residual: Vec<i32>,
    /// An inverse transform's output before it is widened into `residual`.
    transformed: Vec<i16>,
    pred: Vec<T>,
    /// Intra reference samples, refilled per block.
    refs: Refs<T>,
    buffers: mc::Buffers,
    /// Where analysis records go, when they are being collected.
    #[cfg(feature = "analysis")]
    pub stats: Option<StreamStats>,
}

impl<T: Pixel> Default for Scratch<T> {
    fn default() -> Self {
        Self {
            levels: vec![0; MAX_COEFFS],
            wide_levels: vec![0; MAX_COEFFS],
            positions: Vec::with_capacity(MAX_COEFFS),
            residual: vec![0; MAX_COEFFS],
            luma_residual: vec![0; MAX_COEFFS],
            transformed: vec![0; MAX_COEFFS],
            pred: vec![T::default(); MAX_COEFFS],
            refs: Refs::default(),
            buffers: mc::Buffers::default(),
            #[cfg(feature = "analysis")]
            stats: None,
        }
    }
}

/// Decode one slice segment's data into the picture.
pub(super) fn decode_segment<T: Pixel>(
    cx: &SliceContext<'_, T>,
    pic: &mut PictureState<T>,
    carry: &mut Carry,
    scratch: &mut Scratch<T>,
    filter: &mut LoopFilter<T>,
    data: &[u8],
    segment: &SliceSegment,
) -> Result<(), DecodeError> {
    let mut d = SegmentDecoder {
        cx,
        pic,
        filter,
        ctx: Contexts::new(cx.init_type(), cx.header.qp),
        qg: QuantGroup::default(),
        last_qp: cx.header.qp,
        qp: cx.header.qp,
        chroma_qp_offset: [0; 2],
        chroma_qp_offset_coded: false,
        s: scratch,
        #[cfg(feature = "analysis")]
        analysis: analysis::Analysis::default(),
    };
    d.run(carry, data, segment)
}

struct SegmentDecoder<'c, 'a, T: Pixel> {
    cx: &'c SliceContext<'a, T>,
    pic: &'c mut PictureState<T>,
    /// In-loop filtering of the CTB rows this segment completes.
    filter: &'c mut LoopFilter<T>,
    ctx: Contexts,
    qg: QuantGroup,
    last_qp: i32,
    /// `QpY` of the current coding unit.
    qp: i32,
    /// `CuQpOffsetCb`, `CuQpOffsetCr` (§7.4.9.14), and whether the current
    /// chroma QP offset group coded them (`IsCuChromaQpOffsetCoded`).
    chroma_qp_offset: [i32; 2],
    chroma_qp_offset_coded: bool,
    s: &'c mut Scratch<T>,
    /// What the current units' analysis records collect.
    #[cfg(feature = "analysis")]
    analysis: analysis::Analysis,
}

impl<T: Pixel> SegmentDecoder<'_, '_, T> {
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
    fn start_substream(&mut self, carry: &Carry, (x0, y0): (usize, usize), start: Start) {
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
            #[cfg(feature = "analysis")]
            let start = e.bits_consumed();
            #[cfg_attr(
                not(feature = "analysis"),
                allow(unused_variables, clippy::let_unit_value)
            )]
            let signalled = sao::parse(
                e,
                &mut self.ctx,
                self.pic,
                rs,
                sao::Neighbours {
                    left,
                    up,
                    enabled: header.sao,
                    log2_offset_scale: self.cx.pps.range.log2_sao_offset_scale,
                },
            );
            #[cfg(feature = "analysis")]
            if self.s.stats.is_some() {
                self.sao_record(e, rs, (x0, y0), signalled, start);
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
        if self.cx.header.chroma_qp_offsets
            && let Some(offsets) = &self.cx.pps.range.chroma_qp_offsets
            && log2 >= sps.log2_ctb - offsets.depth
        {
            // A chroma QP offset group starts (§7.3.8.4).
            self.chroma_qp_offset_coded = false;
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

    /// `QpY` (eq. 8-283): `((qPY_PRED + CuQpDeltaVal + 52 + 2 ·
    /// QpBdOffsetY) % (52 + QpBdOffsetY)) − QpBdOffsetY`.
    fn update_qp(&mut self) {
        let offset = qp_bd_offset(self.cx.depths.luma);
        let wrap = QP_WRAP + offset;
        self.qp = (self.qg.pred + self.qg.delta + offset).rem_euclid(wrap) - offset;
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
        #[cfg(feature = "analysis")]
        if self.s.stats.is_some() {
            self.start_cu(e);
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
            chroma_modes: [predict::DC; 4],
            chroma_dm: [false; 4],
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
            self.pic.filter_exempt = true;
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
        #[cfg(feature = "analysis")]
        if self.s.stats.is_some() {
            self.cu_record(e, &cu, skip, depth);
        }
        Ok(())
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
        let planes = if self.cx.chroma.format.is_some() {
            3
        } else {
            1
        };
        for (c, plane) in PlaneIndex::ALL.into_iter().enumerate().take(planes) {
            let bits = if c == 0 {
                pcm.bit_depth_luma
            } else {
                pcm.bit_depth_chroma
            };
            let (sx, sy) = self.subsampling(c);
            let (px, py, w, h) = (cu.x >> sx, cu.y >> sy, size >> sx, size >> sy);
            // `recSamples = pcm_sample << (BitDepth − PcmBitDepth)` (eq. 8-7).
            let depth = self.cx.depths.plane(c).bits();
            let mut view = self.pic.samples.plane_mut(plane).view_mut();
            for row in 0..h {
                let line = &mut view.row_mut(py + row)[px..px + w];
                for s in line {
                    *s = T::from_i32((r.read_bits(bits)? << (depth - bits)) as i32);
                }
            }
        }
        let size_bits = r.position();
        let mut flags = 0;
        if pcm.loop_filter_disabled {
            flags |= flag::NO_FILTER;
            self.pic.filter_exempt = true;
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
        if let Some(format) = self.cx.sps.chroma {
            // 4:4:4 codes a chroma mode per partition, the others one.
            let chroma_parts = if format == ChromaFormat::Yuv444 {
                parts
            } else {
                1
            };
            for i in 0..chroma_parts {
                let coded = if e.bin(self.ctx.at(ctx::CHROMA_PRED, 0)) {
                    e.bypass_bits(2) as u8
                } else {
                    predict::CHROMA_DM
                };
                cu.chroma_dm[i] = coded == predict::CHROMA_DM;
                let mode = predict::chroma_mode(coded, cu.luma_modes[i]);
                cu.chroma_modes[i] = if format == ChromaFormat::Yuv422 {
                    predict::chroma_mode_422(mode)
                } else {
                    mode
                };
            }
            if chroma_parts == 1 {
                cu.chroma_modes = [cu.chroma_modes[0]; 4];
                cu.chroma_dm = [cu.chroma_dm[0]; 4];
            }
        }
        #[cfg(feature = "analysis")]
        if self.s.stats.is_some() {
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
        let motion = if merge {
            let index = self.merge_idx(e);
            #[cfg(feature = "analysis")]
            {
                self.analysis.pu.index = index;
            }
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
            let mut mvd = [Mv::default(); 2];
            let mut mvp_flag = [0; 2];
            for list in 0..2 {
                if idc != 2 && idc != list {
                    continue;
                }
                let ref_idx = self.ref_idx(e, header.num_ref_idx[list]);
                motion.ref_idx[list] = ref_idx as i8;
                if !(list == 1 && header.mvd_l1_zero && idc == 2) {
                    #[cfg(feature = "analysis")]
                    let start = e.bits_consumed();
                    mvd[list] = self.mvd(e)?;
                    #[cfg(feature = "analysis")]
                    {
                        self.analysis.pu.mvd_bits += (e.bits_consumed() - start) as u32;
                    }
                }
                mvp_flag[list] = usize::from(e.bin(self.ctx.at(ctx::MVP_FLAG, 0)));
            }
            #[cfg(feature = "analysis")]
            {
                self.analysis.pu.index = mvp_flag[0];
                self.analysis.pu.mvd = mvd;
            }
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
        #[cfg(feature = "analysis")]
        if self.s.stats.is_some() {
            self.pu_record((x, y, w, h), (skip, merge), &motion);
        }
        if cu.part != Part::P2Nx2N {
            // A 2Nx2N block's edges are its coding block's, already marked.
            self.pic.mark_edges((x, y), (w, h), Edges::Prediction);
        }
        self.motion_compensate((x, y), (w, h), &motion)?;
        Ok(merge)
    }

    fn predictor(&self) -> Predictor<'_, T> {
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
        let mut sources: [Option<Source<'_, T>>; 2] = [None, None];
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
            self.cx.depths,
            self.cx.chroma.component,
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
            parent_cbf: [[true, false]; 2],
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
        let mut cbf: ChromaCbf = [[false; 2]; 2];
        let layout = &self.cx.chroma;
        if log2 >= layout.own_min_log2 {
            // 4:2:2 codes the bottom block's flag where its blocks are
            // coded: unsplit, or split into 4x4 luma blocks (§7.3.8.8).
            let two = layout.blocks == 2 && (!split || log2 == SUBSAMPLED_CHROMA_MIN_LOG2);
            let inc = usize::from(node.depth);
            for (f, parent) in cbf.iter_mut().zip(node.parent_cbf) {
                if node.depth == 0 || parent[0] || parent[1] {
                    f[0] = e.bin(self.ctx.at(ctx::CBF_CHROMA, inc));
                    f[1] = two && e.bin(self.ctx.at(ctx::CBF_CHROMA, inc));
                }
            }
        } else if layout.format.is_some() {
            // 4x4 luma blocks: their chroma is their parent's.
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
        let luma_cbf = if cu.intra || node.depth != 0 || any_chroma(cbf) {
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
        cbf: ChromaCbf,
    ) -> Result<(), DecodeError> {
        let size = 1usize << node.log2;
        let chroma_cbf = any_chroma(cbf);
        if (luma_cbf || chroma_cbf) && self.cx.pps.cu_qp_delta_depth.is_some() && !self.qg.coded {
            self.cu_qp_delta(e)?;
        }
        if self.cx.header.chroma_qp_offsets
            && chroma_cbf
            && !cu.bypass
            && !self.chroma_qp_offset_coded
        {
            self.cu_chroma_qp_offset(e);
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
        #[cfg(feature = "analysis")]
        if self.s.stats.is_some() {
            self.start_tu(e);
        }
        let cross_component = self.cx.pps.range.cross_component_prediction;
        let keep_luma = if cross_component {
            CrossComponent::KeepLuma
        } else {
            CrossComponent::None
        };
        self.block(
            e,
            cu,
            (node.x, node.y),
            node.log2,
            0,
            luma_mode,
            (luma_cbf, keep_luma),
        )?;
        let layout = &self.cx.chroma;
        if layout.format.is_some() {
            let chroma_mode = cu
                .intra
                .then(|| cu.chroma_modes[if cu.part == Part::PNxN { quadrant } else { 0 }]);
            let (sx, sy) = (layout.sx, layout.sy);
            // Chroma blocks of 4x4 luma blocks (4:2:0, 4:2:2) are coded
            // with the last of the four, at their parent's position.
            let at = if node.log2 >= layout.own_min_log2 {
                Some(((node.x >> sx, node.y >> sy), node.log2 - sx as u8))
            } else {
                (node.blk_idx == 3).then_some(((node.base.0 >> sx, node.base.1 >> sy), 2))
            };
            if let Some(((xc, yc), log2c)) = at {
                let blocks = layout.blocks;
                // §7.3.8.10: luma coded, and inter or DM chroma.
                let predicted = cross_component
                    && luma_cbf
                    && (!cu.intra
                        || cu.chroma_dm[if cu.part == Part::PNxN { quadrant } else { 0 }]);
                for c in 1..3 {
                    let cross = if predicted {
                        match self.cross_component_scale(e, c - 1) {
                            0 => CrossComponent::None,
                            scale => CrossComponent::Scale(scale),
                        }
                    } else {
                        CrossComponent::None
                    };
                    for (t, &flag) in cbf[c - 1][..blocks].iter().enumerate() {
                        let y = yc + (t << log2c);
                        let coded = (flag, cross);
                        self.block(e, cu, (xc, y), log2c, c, chroma_mode, coded)?;
                    }
                }
            }
        }
        #[cfg(feature = "analysis")]
        if self.s.stats.is_some() {
            let coded = cbf.map(|f| f.contains(&true));
            self.tu_record(e, cu, node, (luma_cbf, coded));
        }
        Ok(())
    }

    /// `cu_chroma_qp_offset_flag` and `cu_chroma_qp_offset_idx`
    /// (§7.3.8.10, semantics §7.4.9.14): `CuQpOffsetCb`, `CuQpOffsetCr`.
    fn cu_chroma_qp_offset(&mut self, e: &mut Engine<'_>) {
        let list = &self
            .cx
            .pps
            .range
            .chroma_qp_offsets
            .as_ref()
            .expect("the slice enables chroma QP offsets with the PPS lists")
            .list;
        self.chroma_qp_offset = if e.bin(self.ctx.at(ctx::CU_CHROMA_QP_OFFSET_FLAG, 0)) {
            // Truncated unary, every bin on one context.
            let mut index = 0;
            while index + 1 < list.len() && e.bin(self.ctx.at(ctx::CU_CHROMA_QP_OFFSET_IDX, 0)) {
                index += 1;
            }
            list[index]
        } else {
            [0; 2]
        };
        self.chroma_qp_offset_coded = true;
    }

    /// `cross_comp_pred(x0, y0, c)` (§7.3.8.12): `ResScaleVal` of chroma
    /// component `c` (0: Cb, 1: Cr).
    fn cross_component_scale(&mut self, e: &mut Engine<'_>, c: usize) -> i32 {
        let mut log_plus1 = 0;
        while log_plus1 < MAX_LOG_RES_SCALE
            && e.bin(
                self.ctx
                    .at(ctx::LOG_RES_SCALE_ABS, 4 * c + log_plus1 as usize),
            )
        {
            log_plus1 += 1;
        }
        if log_plus1 == 0 {
            return 0;
        }
        let magnitude = 1 << (log_plus1 - 1);
        if e.bin(self.ctx.at(ctx::RES_SCALE_SIGN, c)) {
            -magnitude
        } else {
            magnitude
        }
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
        let half_offset = qp_bd_offset(self.cx.depths.luma) / 2;
        if !(-(QP_DELTA_BELOW + half_offset)..=QP_DELTA_ABOVE + half_offset).contains(&delta) {
            return Err(DecodeError::Malformed("CuQpDeltaVal"));
        }
        self.qg.delta = delta;
        self.qg.coded = true;
        #[cfg(feature = "analysis")]
        {
            self.analysis.qp_delta = Some(delta as i8);
        }
        self.update_qp();
        Ok(())
    }

    /// Predict (intra), parse and reconstruct one transform block of
    /// component `c` at plane position `(x, y)`; `cross`: its part in
    /// cross-component prediction.
    #[allow(clippy::too_many_arguments)]
    fn block(
        &mut self,
        e: &mut Engine<'_>,
        cu: &Cu,
        (x, y): (usize, usize),
        log2: u8,
        c: usize,
        intra_mode: Option<u8>,
        (coded, cross): (bool, CrossComponent),
    ) -> Result<(), DecodeError> {
        let n = 1usize << log2;
        if let Some(mode) = intra_mode {
            self.intra_predict(cu, (x, y), n, c, mode);
        }
        if !coded {
            if let CrossComponent::Scale(_) = cross {
                // The chroma residual is the luma one's scaled share.
                let source = ResidualSource::Zero;
                self.reconstruct_general(cu, (x, y), log2, c, source, None, cross);
            }
            return Ok(());
        }
        // 4:4:4 chroma takes luma's mode-dependent scans (4x4 and 8x8,
        // §7.4.9.11).
        let luma_scans = c == 0 || self.cx.chroma.full;
        let scan_kind = intra_mode.map_or(ScanKind::Diagonal, |m| {
            ScanKind::for_intra(m, log2, luma_scans)
        });
        let scan = self.cx.scans.get(log2, scan_kind);
        let pps = self.cx.pps;
        let options = super::residual::Options {
            luma: c == 0,
            sign_hiding: pps.sign_hiding && !cu.bypass,
            transform_skip: !cu.bypass && log2 <= self.cx.transform_skip_log2,
        };
        self.s.positions.clear();
        let count = n * n;
        let parsed = if self.cx.residual == ResidualPath::Wide {
            let range = self.range_options(cu, c, intra_mode);
            super::residual::read::<T, true, i32>(
                e,
                &mut self.ctx,
                scan,
                (options, &range),
                &mut self.s.wide_levels[..count],
                &mut self.s.positions,
            )
        } else if self.cx.residual == ResidualPath::Range {
            let range = self.range_options(cu, c, intra_mode);
            super::residual::read::<T, true, i16>(
                e,
                &mut self.ctx,
                scan,
                (options, &range),
                &mut self.s.levels[..count],
                &mut self.s.positions,
            )
        } else {
            super::residual::read::<T, false, i16>(
                e,
                &mut self.ctx,
                scan,
                (options, &RangeOptions::NONE),
                &mut self.s.levels[..count],
                &mut self.s.positions,
            )
        };
        let Ok(parsed) = parsed else {
            self.clear_levels();
            return parsed.map(|_| ());
        };
        #[cfg(feature = "analysis")]
        if self.s.stats.is_some() {
            self.note_block(parsed.transform_skip);
        }
        self.reconstruct(cu, (x, y), log2, c, parsed, intra_mode, cross);
        self.clear_levels();
        Ok(())
    }

    /// The range extension tools of a block of component `c` (intra with
    /// `intra_mode`, else inter).
    fn range_options(&self, cu: &Cu, c: usize, intra_mode: Option<u8>) -> RangeOptions {
        let range = &self.cx.sps.range;
        RangeOptions {
            bypass: cu.bypass,
            explicit_rdpcm: !cu.intra && range.explicit_rdpcm,
            implicit_rdpcm: range.implicit_rdpcm
                && matches!(intra_mode, Some(predict::HORIZONTAL | predict::VERTICAL)),
            skip_context: range.transform_skip_context,
            persistent_rice: range.persistent_rice_adaptation,
            align: range.cabac_bypass_alignment,
            limited_prefix: self.cx.wide.is_some_and(|w| w.extended),
            log2_range: self
                .cx
                .wide
                .map_or(LOG2_TRANSFORM_RANGE, |w| w.log2_range[usize::from(c > 0)]),
        }
    }

    /// Keep the all-zero invariant of the level buffers for the next block.
    fn clear_levels(&mut self) {
        if self.cx.residual == ResidualPath::Wide {
            for &p in &self.s.positions {
                self.s.wide_levels[usize::from(p)] = 0;
            }
        } else {
            for &p in &self.s.positions {
                self.s.levels[usize::from(p)] = 0;
            }
        }
    }

    /// Scaling and transformation of the parsed levels (§8.6.2–§8.6.4),
    /// added to the prediction in the picture. DCT and DST blocks go
    /// through the DSP's fused inverse-and-add (all-zero, DC-only and
    /// zero-region shortcuts inside); transform skip, lossless bypass and
    /// residuals that cross-component prediction reads or changes, all
    /// rare, build the residual first ([`Self::reconstruct_general`]).
    #[allow(clippy::too_many_arguments)]
    fn reconstruct(
        &mut self,
        cu: &Cu,
        (x, y): (usize, usize),
        log2: u8,
        c: usize,
        parsed: Coded,
        intra_mode: Option<u8>,
        cross: CrossComponent,
    ) {
        let n = 1usize << log2;
        let count = n * n;
        let intra = intra_mode.is_some();
        let transform_skip = parsed.transform_skip;
        let wide = self.cx.wide;
        let extent = if cu.bypass {
            (n, n)
        } else if let Some(wide) = wide {
            self.scale_wide_levels(log2, c, (intra, transform_skip), wide);
            (n, n)
        } else {
            self.scale_levels(log2, c, intra, transform_skip)
        };
        let source = if cu.bypass {
            ResidualSource::Bypass
        } else if transform_skip {
            ResidualSource::TransformSkip
        } else {
            let kind =
                crate::transform::kind(log2, intra && c == 0).expect("transform sizes 4..32");
            if cross == CrossComponent::None && wide.is_none() {
                let depth = self.cx.depths.plane(c);
                let mut dst = self
                    .pic
                    .samples
                    .plane_mut(PlaneIndex::ALL[c])
                    .region_mut(x as isize, y as isize, n, n)
                    .expect("transform blocks lie inside the picture");
                self.cx.dsp.at(depth).hevc_inverse_add(
                    kind,
                    &self.s.levels[..count],
                    extent,
                    &mut dst,
                );
                return;
            }
            ResidualSource::Transform(kind)
        };
        // RDPCM of untransformed residuals (§8.6.2): implicit for intra
        // modes 10 and 26, explicit for inter blocks.
        let range = &self.cx.sps.range;
        let rdpcm = if source == ResidualSource::Bypass || source == ResidualSource::TransformSkip {
            match intra_mode {
                Some(predict::HORIZONTAL) if range.implicit_rdpcm => Some(Rdpcm::Horizontal),
                Some(predict::VERTICAL) if range.implicit_rdpcm => Some(Rdpcm::Vertical),
                Some(_) => None,
                None => parsed.explicit_rdpcm.map(|vertical| {
                    if vertical {
                        Rdpcm::Vertical
                    } else {
                        Rdpcm::Horizontal
                    }
                }),
            }
        } else {
            None
        };
        self.reconstruct_general(cu, (x, y), log2, c, source, rdpcm, cross);
    }

    /// The residual of a block built in full, then added (`Clip1`):
    /// transform skip (`tsShift = 5 + Log2(nTbS)`, `bdShift = 20 −
    /// BitDepth`, §8.6.4.2), lossless bypass, or the inverse transform of
    /// a block whose residual cross-component prediction reads; then the
    /// 180° rotation of 4x4 intra residuals
    /// (`transform_skip_rotation_enabled_flag`), RDPCM (§8.6.2) and
    /// cross-component prediction (§8.6.6). Kept out of line: inlined
    /// into the block decoder it pushed the residual parser out of it (+1 %
    /// decode instructions, callgrind on a 720p stream of ours,
    /// 2026-11-15).
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_general(
        &mut self,
        cu: &Cu,
        (x, y): (usize, usize),
        log2: u8,
        c: usize,
        source: ResidualSource,
        rdpcm: Option<Rdpcm>,
        cross: CrossComponent,
    ) {
        let n = 1usize << log2;
        let count = n * n;
        let depth = self.cx.depths.plane(c);
        let wide = self.cx.wide;
        let levels = &self.s.levels[..count];
        let wide_levels = &self.s.wide_levels[..count];
        let residual = &mut self.s.residual[..count];
        match source {
            ResidualSource::Zero => residual.fill(0),
            ResidualSource::Bypass | ResidualSource::TransformSkip => {
                let rotate =
                    self.cx.sps.range.transform_skip_rotation && n == ROTATED_SIZE && cu.intra;
                let shifts = (source == ResidualSource::TransformSkip).then(|| match wide {
                    Some(wide) => wide.transform_skip_shifts(c, log2),
                    None => (
                        TRANSFORM_SKIP_SHIFT_BELOW + u32::from(log2),
                        RESIDUAL_BITS - depth.bits(),
                    ),
                });
                if wide.is_some() {
                    untransformed(wide_levels, residual, rotate, shifts);
                } else {
                    untransformed(levels, residual, rotate, shifts);
                }
            }
            ResidualSource::Transform(kind) => {
                if let Some(wide) = wide {
                    let shift = wide.second_shift[usize::from(c > 0)];
                    let range = wide.coeff_range(c);
                    crate::transform::inverse_extended(kind, wide_levels, residual, range, shift);
                } else {
                    let transformed = &mut self.s.transformed[..count];
                    self.cx
                        .dsp
                        .at(depth)
                        .hevc_inverse(kind, levels, transformed);
                    for (r, &t) in residual.iter_mut().zip(transformed.iter()) {
                        *r = i32::from(t);
                    }
                }
            }
        }
        match rdpcm {
            Some(Rdpcm::Horizontal) => {
                for row in residual.chunks_exact_mut(n) {
                    for i in 1..n {
                        row[i] += row[i - 1];
                    }
                }
            }
            Some(Rdpcm::Vertical) => {
                for i in n..count {
                    residual[i] += residual[i - n];
                }
            }
            None => {}
        }
        match cross {
            CrossComponent::None => {}
            CrossComponent::KeepLuma => self.s.luma_residual[..count].copy_from_slice(residual),
            CrossComponent::Scale(scale) => {
                // rC += (ResScaleVal · ((rY << BitDepthC) >> BitDepthY)) >> 3.
                let (luma_bits, chroma_bits) =
                    (self.cx.depths.luma.bits(), self.cx.depths.chroma.bits());
                // In 64 bits: a 16-bit residual shifted by 16 leaves i32.
                for (r, &luma) in residual.iter_mut().zip(&self.s.luma_residual[..count]) {
                    let aligned = (i64::from(luma) << chroma_bits) >> luma_bits;
                    *r += ((i64::from(scale) * aligned) >> CROSS_COMPONENT_SHIFT) as i32;
                }
            }
        }
        let mut dst = self
            .pic
            .samples
            .plane_mut(PlaneIndex::ALL[c])
            .region_mut(x as isize, y as isize, n, n)
            .expect("transform blocks lie inside the picture");
        for (row, line) in residual.chunks_exact(n).enumerate() {
            for (s, &r) in dst.row_mut(row).iter_mut().zip(line) {
                *s = depth.clip_to(s.to_i32() + r);
            }
        }
    }

    /// Scale the parsed levels in place (§8.6.3, `scaling_list_enabled`
    /// factors or the flat 16), visiting only the coded positions. Returns
    /// the rows and columns that hold them, for the inverse transform.
    fn scale_levels(
        &mut self,
        log2: u8,
        c: usize,
        intra: bool,
        transform_skip: bool,
    ) -> (usize, usize) {
        let depths = self.cx.depths;
        let qp = self.component_qp(c);
        let factors = self.cx.scaling_factors(log2, c, (intra, transform_skip));
        // bdShift = BitDepth + log2(nTbS) + 10 − 15 (§8.6.3).
        let shift = depths.plane(c).bits() + u32::from(log2) - SCALING_SHIFT_BELOW;
        let (level_scale, per) = (LEVEL_SCALE[(qp % 6) as usize], (qp / 6) as u32);
        let (mut rows, mut columns) = (0, 0);
        for &p in &self.s.positions {
            let p = usize::from(p);
            rows = rows.max((p >> log2) + 1);
            columns = columns.max((p & ((1 << log2) - 1)) + 1);
            let m = factors.map_or(i32::from(FLAT_SCALE), |f| i32::from(f[p]));
            let level = i32::from(self.s.levels[p]);
            self.s.levels[p] = nc_dsp::scale_coefficient(level, m, level_scale, per, shift);
        }
        (rows, columns)
    }

    /// [`Self::scale_levels`] of the wide levels (§8.6.3 with
    /// `log2TransformRange`): `bdShift = BitDepth + Log2(nTbS) + 10 −
    /// log2TransformRange`, clipped to `CoeffMin..=CoeffMax`, in 64 bits.
    fn scale_wide_levels(&mut self, log2: u8, c: usize, kind: (bool, bool), wide: WideRange) {
        let qp = self.component_qp(c);
        let factors = self.cx.scaling_factors(log2, c, kind);
        let range = wide.log2_range[usize::from(c > 0)];
        let shift = self.cx.depths.plane(c).bits() + u32::from(log2) + SCALING_SHIFT_ABOVE - range;
        let level_scale = i64::from(LEVEL_SCALE[(qp % 6) as usize]);
        let per = (qp / 6) as u32;
        let (min, max) = wide.coeff_range(c);
        for &p in &self.s.positions {
            let p = usize::from(p);
            let m = factors.map_or(i64::from(FLAT_SCALE), |f| i64::from(f[p]));
            let level = i64::from(self.s.wide_levels[p]);
            let scaled = (((level * m * level_scale) << per) + (1 << (shift - 1))) >> shift;
            self.s.wide_levels[p] = scaled.clamp(i64::from(min), i64::from(max)) as i32;
        }
    }

    /// `qP` of component `c` (§8.6.1): `Qp′Y`, `Qp′Cb` or `Qp′Cr`, the QP
    /// plus the component's `QpBdOffset`.
    fn component_qp(&self, c: usize) -> i32 {
        let depths = self.cx.depths;
        if c == 0 {
            self.qp + qp_bd_offset(depths.luma)
        } else {
            let offset = if c == 1 {
                self.cx.pps.cb_qp_offset + self.cx.header.cb_qp_offset
            } else {
                self.cx.pps.cr_qp_offset + self.cx.header.cr_qp_offset
            } + self.chroma_qp_offset[c - 1];
            let chroma_offset = qp_bd_offset(depths.chroma);
            let qpi = (self.qp + offset).clamp(-chroma_offset, MAX_CHROMA_QPI);
            let format = self.cx.sps.chroma.expect("chroma blocks only with chroma");
            chroma_qp(format, qpi) + chroma_offset
        }
    }

    /// The subsampling `(sx, sy)` of component `c` (log2 `SubWidthC`,
    /// `SubHeightC`; 0 for luma).
    fn subsampling(&self, c: usize) -> (usize, usize) {
        if c > 0 {
            (self.cx.chroma.sx, self.cx.chroma.sy)
        } else {
            (0, 0)
        }
    }

    /// Intra sample prediction of an `n`x`n` block of component `c`
    /// (§8.4.4.2) written into the picture.
    fn intra_predict(&mut self, cu: &Cu, (x, y): (usize, usize), n: usize, c: usize, mode: u8) {
        let (sx, sy) = self.subsampling(c);
        let plane = PlaneIndex::ALL[c];
        let constrained = self.cx.pps.constrained_intra_pred;
        let pic = &*self.pic;
        let cur = (x << sx, y << sy);
        // Without constrained intra prediction, availability is uniform
        // along each side of the block (see `Refs::gather`).
        let sides = if constrained {
            Sides::PerUnit
        } else {
            Sides::Uniform
        };
        let depth = self.cx.depths.plane(c);
        self.s.refs.gather_into(
            pic.samples.plane(plane),
            (x, y),
            n,
            // Availability is uniform over 4x4 luma blocks: the finer
            // subsampled axis sets the run.
            4 >> sx.max(sy),
            sides,
            depth,
            |xn, yn| {
                let nb = ((xn << sx) as isize, (yn << sy) as isize);
                pic.available(cur, nb, self.cx.slice)
                    && (!constrained
                        || pic.info[pic.block(nb.0 as usize, nb.1 as usize)].flags & flag::INTRA
                            != 0)
            },
        );
        let mut filters = self.cx.chroma.intra_filters[usize::from(c > 0)];
        // disableIntraBoundaryFilter (§8.4.4.2.6).
        filters.angular_edges &= !(cu.bypass && self.cx.sps.range.implicit_rdpcm);
        let mut view = self
            .pic
            .samples
            .plane_mut(plane)
            .region_mut(x as isize, y as isize, n, n)
            .expect("intra block inside the picture");
        predict::predict_smoothed_into(
            self.cx.dsp.at(depth),
            &self.s.refs,
            (mode, filters),
            &mut self.s.pred,
            &mut view,
        );
    }
}

/// The untransformed residual of `levels` into `residual`: lossless
/// (`shifts` `None`) or transform skip (`(tsShift, bdShift)`: `r = (d <<
/// tsShift + 2^(bdShift − 1)) >> bdShift`, §8.6.4.2), read rotated by 180°
/// when `rotate` (§8.6.2). Computed in 64 bits; the results fit `i32`:
/// `|d| <= 2^22` and `tsShift - bdShift <= 2`.
fn untransformed<L: Copy + Into<i64>>(
    levels: &[L],
    residual: &mut [i32],
    rotate: bool,
    shifts: Option<(u32, u32)>,
) {
    let last = levels.len() - 1;
    for (i, r) in residual.iter_mut().enumerate() {
        let level: i64 = levels[if rotate { last - i } else { i }].into();
        *r = match shifts {
            None => level,
            Some((ts_shift, bd_shift)) => ((level << ts_shift) + (1 << (bd_shift - 1))) >> bd_shift,
        } as i32;
    }
}
