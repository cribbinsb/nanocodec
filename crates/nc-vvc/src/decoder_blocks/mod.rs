//! H.266 intra and predicted coding trees, using the shared CABAC and intra-mode syntax.
//! General-tree syntax follows BSD-Clear `VVdeC`; see `THIRD_PARTY.md`.
#[cfg(feature = "analysis")]
mod analysis;
#[cfg(feature = "analysis")]
pub(crate) use analysis::record as record_analysis;
mod cclm;
mod chroma_alf;
mod ciip;
mod coeff;
mod deblock;
mod filter_syntax;
mod filters;
mod ibc;
mod palette;
pub(crate) use filter_syntax::{CtuFilters, counts as filter_counts};
#[cfg(test)]
pub(crate) use filters::apply as apply_filters;
pub(crate) use filters::{FilterScratch, apply_with_scratch as apply_filters_with_scratch};
mod sbt;
mod scaling_lists;
pub(crate) use scaling_lists::of_slice as scaling_lists;
mod skip_coeff;
#[cfg(test)]
pub(super) use deblock::apply as deblock;
pub(crate) use deblock::apply_with_scratch as deblock_with_scratch;
mod reconstruct;
mod reshape;
pub(crate) use reshape::model as reshape_model;
mod references;
pub(super) use inter_prediction::Sources as InterSources;
pub(super) use reconstruct::reconstruct;
#[cfg(test)]
use reconstruct::reconstructed;
pub(crate) use reconstruct::{ReconScratch, ResidualContext};
mod affine_motion;
mod boundaries;
pub(crate) use boundaries::Closures;
mod amvp_syntax;
mod initialization;
mod inter_models;
mod inter_prediction;
mod prof;
mod quant_groups;
mod refinement;
mod resampling;
pub(crate) use refinement::{Edges, RefinementScratch, derive as derive_refinement};
pub(crate) use resampling::{Geometry, Scaling};
mod inter_syntax;
mod intra_syntax;
mod isp;
mod layout;
mod merge_syntax;
mod motion;
mod motion_candidates;
mod motion_refine;
mod temporal;
#[cfg(test)]
pub(super) use motion::derive as derive_motion;
pub(crate) use motion::{MotionScratch, derive_with_scratch as derive_motion_with_scratch};
pub(crate) use temporal::{Field as MotionField, Temporal};
mod transform;
mod transform_syntax;
mod transform_units;
use transform_units::Transform;
mod tree;
mod unit;
mod wavefront;
use inter_syntax::Inter;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Constraint {
    Any,
    Intra,
    Inter,
}
use crate::{
    contexts::{Context, Contexts},
    decoder_cabac::Reader,
    decoder_headers::{PictureParameters, Sequence, SliceHeader},
    intra,
};
use nc_common::{
    ChromaFormat,
    decode::{DecodeError, Result},
};
#[cfg(test)]
pub(super) use transform::inverse_residuals;
use tree::{Node, Split};

/// Persistent scratch buffers reused across slices and pictures of a general
/// decoding instance (`General<T>`), avoiding per-slice and per-picture heap
/// allocation, zeroing, and deallocation.
#[derive(Debug, Default)]
pub(crate) struct DecodeScratch<T: nc_dsp::Pixel> {
    pub(crate) parser_grid: [Vec<u32>; 2],
    pub(crate) motion: MotionScratch,
    pub(crate) refinement: RefinementScratch<T>,
    pub(crate) recon: ReconScratch<T>,
    pub(crate) deblock_grid: [Vec<u32>; 2],
    pub(crate) filters: FilterScratch<T>,
}

/// Slice-reusable compact storage of non-zero `(pos, level)` pairs and
/// coefficient-parsing grids across all transform units, avoiding per-TU
/// `Vec` allocations and stack `memset`s. Levels are 16-bit
/// (`log2TransformRange` 15, §7.4.11.11), or 32-bit in `wide` with extended
/// precision (up to 22 bits); a span into `wide` has [`WIDE_SPAN`] set.
#[derive(Debug)]
pub(crate) struct CoeffArena {
    pub(super) entries: Vec<(u16, i16)>,
    pub(super) wide: Vec<(u16, i32)>,
    pub(super) grid: Box<[u32; coeff::PADDED_AREA]>,
    pub(super) pass1: Box<[u8; coeff::PADDED_AREA]>,
    pub(super) skip_vals: Box<[i32; 64 * 64]>,
}

impl Default for CoeffArena {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            wide: Vec::new(),
            grid: Box::new([0; coeff::PADDED_AREA]),
            pass1: Box::new([0; coeff::PADDED_AREA]),
            skip_vals: Box::new([0; 64 * 64]),
        }
    }
}

/// The flag of a span into [`CoeffArena::wide`].
pub(super) const WIDE_SPAN: u32 = 1 << 31;

impl CoeffArena {
    /// The levels of `span`.
    #[inline]
    pub(super) fn slice(&self, (start, len): (u32, u16)) -> Levels<'_> {
        let s = (start & !WIDE_SPAN) as usize;
        let end = s + usize::from(len);
        if start & WIDE_SPAN == 0 {
            Levels::Narrow(&self.entries[s..end])
        } else {
            Levels::Wide(&self.wide[s..end])
        }
    }

    /// The span start of the next level pushed to `entries`, or to `wide`.
    pub(super) fn span_start(&self, wide: bool) -> Result<u32> {
        let len = if wide {
            self.wide.len()
        } else {
            self.entries.len()
        };
        u32::try_from(len)
            .ok()
            .filter(|&start| start & WIDE_SPAN == 0)
            .map(|start| if wide { start | WIDE_SPAN } else { start })
            .ok_or(DecodeError::Malformed("VVC coeff arena"))
    }

    /// The span from `start` (from [`Self::span_start`]) to the end.
    pub(super) fn span_from(&self, start: u32) -> Result<(u32, u16)> {
        let len = if start & WIDE_SPAN == 0 {
            self.entries.len()
        } else {
            self.wide.len()
        };
        u16::try_from(len - (start & !WIDE_SPAN) as usize)
            .map(|count| (start, count))
            .map_err(|_| DecodeError::Malformed("VVC coeff count"))
    }
}

/// The levels of one coded block: 16-bit, or 32-bit with extended precision.
#[derive(Clone, Copy)]
pub(super) enum Levels<'a> {
    Narrow(&'a [(u16, i16)]),
    Wide(&'a [(u16, i32)]),
}

impl Levels<'_> {
    /// Whether no level is coded.
    #[inline]
    pub(super) fn is_empty(self) -> bool {
        match self {
            Self::Narrow(levels) => levels.is_empty(),
            Self::Wide(levels) => levels.is_empty(),
        }
    }

    /// Write each level into `buffer` at its raster position.
    #[inline]
    pub(super) fn scatter(self, buffer: &mut [i32]) {
        match self {
            Self::Narrow(levels) => {
                for &(pos, value) in levels {
                    buffer[usize::from(pos)] = i32::from(value);
                }
            }
            Self::Wide(levels) => {
                for &(pos, value) in levels {
                    buffer[usize::from(pos)] = value;
                }
            }
        }
    }

    /// Zero the positions [`Self::scatter`] wrote.
    #[inline]
    pub(super) fn clear(self, buffer: &mut [i32]) {
        match self {
            Self::Narrow(levels) => levels
                .iter()
                .for_each(|&(pos, _)| buffer[usize::from(pos)] = 0),
            Self::Wide(levels) => levels
                .iter()
                .for_each(|&(pos, _)| buffer[usize::from(pos)] = 0),
        }
    }
}

#[derive(Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent normative CU and refinement flags"
)]
pub(super) struct Unit {
    node: Node,
    channel: usize,
    combined: bool,
    inter: Option<Inter>,
    motion: Option<motion::Motion>,
    affine_models: [Option<crate::affine::Model>; 2],
    affine_fields: [Option<crate::affine::Field>; 2],
    geometric: Option<[motion::Motion; 2]>,
    /// The resolved block vector of an IBC CU (whole samples, 1/16 units).
    block_vector: Option<crate::motion::Mv16>,
    subblocks: Vec<motion::Motion>,
    skip: bool,
    ciip_weight: u8,
    bdpcm: [u8; 2],
    mip: Option<(u8, bool)>,
    reference_line: u8,
    isp: Option<crate::isp::Split>,
    sbt: Option<sbt::Sbt>,
    /// The palette of a palette-coded CU (`pred_mode_plt_flag`).
    palette: Option<Box<palette::Palette>>,
    prof: bool,
    smvd: bool,
    bdof: bool,
    dmvr: Vec<refinement::Region>,
    lfnst: u8,
    mts: u8,
    luma_mode: u8,
    chroma_mode: u8,
    /// Whether 4:4:4 chroma of a single-tree MIP CU takes the derived mode,
    /// and so is predicted by MIP too (H.266 §8.4.3, §8.4.5.2.2).
    chroma_mip: bool,
    /// `cu_act_enabled_flag` (H.266 §7.3.11.5): the residuals are coded in
    /// YCgCo-R and converted back before reconstruction (§8.7.4.6).
    act: bool,
    /// Luma QP of the CU (`QpY`, `−QpBdOffset ..= 63`; a chroma-tree CU
    /// holds the luma QP its chroma QPs derive from).
    qp: i8,
    /// CU chroma QP offsets (Cb, Cr, joint) from the PPS list.
    chroma_offsets: [i8; 3],
    /// Index of the CU's tile: neighbours in another tile are unavailable.
    tile: u16,
    /// Index of the CU's slice in its picture (in decoding order): what
    /// its slice header is, and loop filtering across slice edges.
    slice: u16,
    transforms: Vec<Transform>,
}
impl Unit {
    /// The CU's luma QP and chroma QP offsets, as `transform::qp` takes them.
    fn quant(&self) -> (i8, [i8; 3]) {
        (self.qp, self.chroma_offsets)
    }
    /// Whether the CU is intra predicted (`MODE_INTRA`; palette CUs are
    /// `MODE_PLT`, H.266 §7.4.12.2).
    fn intra(&self) -> bool {
        self.inter.is_none() && self.palette.is_none()
    }
    /// Whether this is an IBC CU (known from its syntax, before its vector
    /// is resolved).
    fn block_vector_signalled(&self) -> bool {
        matches!(self.inter, Some(Inter::Ibc(_)))
    }
}
const EMPTY_CELL: u32 = u32::MAX;

struct Parser<'a> {
    sequence: &'a Sequence,
    header: &'a SliceHeader,
    arena: &'a mut CoeffArena,
    dimensions: [usize; 2],
    stride: usize,
    grid: &'a mut [Vec<u32>; 2],
    units: &'a mut Vec<Unit>,
    contexts: Contexts,
    trees: tree::Models,
    coefficients: coeff::Models,
    skip_coefficients: skip_coeff::Models,
    cclm: [Context; 2],
    /// The `cu_act_enabled_flag` context (§9.3.2.2, Table 51).
    act: Context,
    intra_models: intra_syntax::Models,
    joint: [Context; 3],
    constraint: [Context; 2],
    ref_index: [Context; 2],
    inter_models: inter_models::Models,
    transform_models: transform_syntax::Models,
    sbt_models: sbt::Models,
    ibc_models: ibc::Models,
    palette_models: palette::Models,
    /// The palette predictor (§7.4.12.13), kept across the slice's CUs.
    palette_predictor: palette::Predictor,
    groups: quant_groups::QuantGroups,
    /// The range extension's residual coding state (persistent Rice
    /// statistics and the slice's Rice and range parameters).
    rice: coeff::RiceContext,
    /// The CTU being parsed.
    ctu: layout::Ctu,
    /// The index of the slice being parsed in its picture.
    slice: u16,
    /// `NoBackwardPredFlag` (H.266 §8.5.1): no active reference follows
    /// the current picture in output order.
    no_backward_prediction: bool,
}

/// Parse one slice of the picture: its CUs, and the SAO/ALF syntax of its
/// CTUs into `filters` (the picture's, by CTU address). Unsupported tools
/// are rejected before their conditionally present CABAC bins can be misread.
#[cfg(test)]
pub(super) fn parse(
    bytes: &[u8],
    s: &Sequence,
    p: &PictureParameters,
    h: &SliceHeader,
) -> Result<Vec<Unit>> {
    let mut filters = vec![filter_syntax::CtuFilters::default(); h.ctus.len()];
    let mut arena = CoeffArena::default();
    let mut units = parse_filtered(
        bytes,
        s,
        p,
        h,
        filter_syntax::Counts::default(),
        (0, no_backward_prediction(h)),
        &mut filters,
        &mut arena,
    )?;
    populate_test_levels(&mut units, &arena, s);
    Ok(units)
}

#[cfg(test)]
pub(super) fn populate_test_levels(units: &mut [Unit], arena: &CoeffArena, s: &Sequence) {
    for unit in units {
        for transform in &mut unit.transforms {
            for plane in 0..3 {
                if !transform.coded[plane] {
                    continue;
                }
                let block = if plane == 0 {
                    transform.block
                } else {
                    transform.chroma_block.unwrap()
                };
                let (width, height) = s.scaled(plane != 0, (block.width(), block.height()));
                let area = width * height;
                let mut full = vec![0i32; area];
                arena.slice(transform.spans[plane]).scatter(&mut full);
                transform.levels[plane] = full;
            }
        }
    }
}

/// `NoBackwardPredFlag` of a slice whose references are all short-term
/// (the tests' streams): no reference follows the current picture.
#[cfg(test)]
fn no_backward_prediction(h: &SliceHeader) -> bool {
    h.references.iter().zip(h.active).all(|(list, count)| {
        list.entries
            .iter()
            .take(usize::from(count))
            .all(|r| matches!(r, crate::decoder_headers::ReferenceEntry::Short(d) if *d <= 0))
    })
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn parse_filtered(
    bytes: &[u8],
    s: &Sequence,
    p: &PictureParameters,
    h: &SliceHeader,
    filter_counts: filter_syntax::Counts,
    (slice, no_backward): (u16, bool),
    filters: &mut [filter_syntax::CtuFilters],
    arena: &mut CoeffArena,
) -> Result<Vec<Unit>> {
    let mut grid = [Vec::new(), Vec::new()];
    let mut units = Vec::new();
    parse_filtered_into(
        bytes,
        s,
        p,
        h,
        filter_counts,
        (slice, no_backward),
        filters,
        arena,
        &mut grid,
        &mut units,
    )?;
    Ok(units)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_filtered_into(
    bytes: &[u8],
    s: &Sequence,
    p: &PictureParameters,
    h: &SliceHeader,
    filter_counts: filter_syntax::Counts,
    (slice, no_backward): (u16, bool),
    filters: &mut [filter_syntax::CtuFilters],
    arena: &mut CoeffArena,
    grid_scratch: &mut [Vec<u32>; 2],
    units_out: &mut Vec<Unit>,
) -> Result<()> {
    validate_tools(s)?;
    arena.entries.clear();
    arena.wide.clear();
    if arena.entries.capacity() < 16_384 {
        arena.entries.reserve(16_384 - arena.entries.capacity());
    }
    let dimensions = p.dimensions.map(|v| v as usize);
    let inter = h.kind != 2;
    let scan = layout::scan(&h.ctus, &p.partitioning.tiles, s.ctu_log2);
    let mut parser = Parser::new(s, p, h, scan[0], arena, grid_scratch, units_out);
    parser.slice = slice;
    parser.no_backward_prediction = no_backward;
    let (qp, init_kind) = parser.initialization();
    let mut reader = Reader::new(
        bytes
            .get(h.payload_offset..)
            .ok_or(DecodeError::Truncated)?,
    )?;
    let ctu = 1 << s.ctu_log2;
    let columns = dimensions[0].div_ceil(ctu);
    let mut in_slice = vec![false; filters.len()];
    for position in &scan {
        in_slice[position.address] = true;
    }
    let mut filter_models = filter_syntax::Models::new(qp, init_kind);
    let initial = parser.save_contexts(&filter_models);
    let mut row_start = None;
    for (index, &position) in scan.iter().enumerate() {
        let saved = if position.starts_tile() {
            Some(&initial)
        } else {
            row_start.as_ref().filter(|_| position.starts_row())
        };
        if let Some(saved) = saved.filter(|_| index != 0) {
            if !reader.terminate()? {
                return Err(DecodeError::Malformed("VVC CABAC substream termination"));
            }
            reader.restart()?;
            parser.start_row(saved, &mut filter_models);
        }
        parser.ctu = position;
        let neighbours = [
            (position.x > position.tile_origin.0).then(|| position.address - 1),
            (position.y > position.tile_origin.1).then(|| position.address - columns),
        ]
        .map(|address| address.filter(|&a| in_slice[a]).map(|a| filters[a]));
        filters[position.address] = filter_models.read(
            &mut reader,
            &mut parser.contexts,
            h,
            neighbours,
            (filter_counts, s.depth),
        )?;
        if inter {
            parser.tree(
                &mut reader,
                Node::root(position.x, position.y, ctu),
                0,
                s.chroma,
                Constraint::Any,
            )?;
        } else {
            parser.intra_ctu(&mut reader, Node::root(position.x, position.y, ctu))?;
        }
        if s.entropy_sync && position.starts_row() {
            row_start = Some(parser.save_contexts(&filter_models));
        }
    }
    if !reader.terminate()? {
        return Err(DecodeError::Malformed("VVC slice CABAC termination"));
    }
    Ok(())
}
impl<'a> Parser<'a> {
    fn new(
        s: &'a Sequence,
        p: &PictureParameters,
        h: &'a SliceHeader,
        first: layout::Ctu,
        arena: &'a mut CoeffArena,
        grid: &'a mut [Vec<u32>; 2],
        units: &'a mut Vec<Unit>,
    ) -> Self {
        let dimensions = p.dimensions.map(|v| v as usize);
        let stride = dimensions[0].div_ceil(4);
        let cells = stride * dimensions[1].div_ceil(4);
        for g in grid.iter_mut() {
            g.clear();
            g.resize(cells, EMPTY_CELL);
        }
        units.clear();
        let inter = h.kind != 2;
        let (qp, init_kind) = initialization_of(h);
        let mut rice = coeff::RiceContext {
            log2_range: s.log2_transform_range(),
            base_level: rice_base_level(s, h),
            extension: s.range.rice_extension,
            statistics: s.range.persistent_rice.then_some([0; 3]),
            reverse_last: h.reverse_last,
        };
        rice.reset_statistics(s.depth.bits());
        Self {
            rice,
            ctu: first,
            slice: 0,
            no_backward_prediction: false,
            sequence: s,
            header: h,
            arena,
            dimensions,
            stride,
            grid,
            units,
            contexts: initialization::contexts(qp, init_kind),
            inter_models: inter_models::Models::new(qp, init_kind),
            trees: tree::Models::new(qp, init_kind),
            coefficients: coeff::Models::new(qp, init_kind),
            skip_coefficients: skip_coeff::Models::new(qp, init_kind),
            intra_models: intra_syntax::Models::new(qp, init_kind),
            transform_models: transform_syntax::Models::new(qp, init_kind),
            sbt_models: sbt::Models::new(qp, init_kind),
            ibc_models: ibc::Models::new(qp, init_kind),
            palette_models: palette::Models::new(qp, init_kind),
            palette_predictor: palette::Predictor::default(),
            groups: quant_groups::QuantGroups::new(
                [p.cu_qp_delta, h.cu_chroma_offsets],
                [
                    h.picture.qp_subdiv[usize::from(inter)],
                    h.picture.chroma_subdiv[usize::from(inter)],
                ],
                &p.cu_chroma_offsets,
                (h.qp as i8, s.qp_bd_offset() as i8),
                init_kind,
            ),
            constraint: initialization::bank([[25, 20], [25, 12], [35, 35]], [1, 0], qp, init_kind),
            ref_index: initialization::bank([[5, 35], [20, 35], [35, 35]], [0, 4], qp, init_kind),
            cclm: initialization::bank([[26, 27], [34, 27], [59, 27]], [4, 9], qp, init_kind),
            act: initialization::bank::<1, 1>([[46], [46], [52]], [1], qp, init_kind)[0],
            joint: initialization::bank(
                [[42, 43, 52], [27, 36, 45], [12, 21, 35]],
                [1, 1, 0],
                qp,
                init_kind,
            ),
        }
    }

    /// Slice QP and initialisation type of the context variables.
    fn initialization(&self) -> (u32, u8) {
        initialization_of(self.header)
    }
}

/// Slice QP and initialisation type of the context variables (§9.3.2.2:
/// the QP is `Clip3(0, 63, SliceQpY)`, as `SliceQpY` may be negative above
/// 8 bits).
fn initialization_of(h: &SliceHeader) -> (u32, u8) {
    let kind = if h.cabac_init && h.kind != 2 {
        1 - h.kind
    } else {
        h.kind
    };
    (h.qp.clamp(0, MAX_CONTEXT_QP) as u32, kind)
}

/// The deepest samples decoded (Main 12, and 12-bit streams of Main 16
/// 4:4:4): residuals and high-precision inter samples fit 16 bits.
const MAX_DECODED_BIT_DEPTH: u32 = 12;

/// `baseLevel` of the remainders after the context-coded flags (H.266
/// §9.3.3.11, VTM `setRiceBaseLevel`): 4, or with the extended Rice
/// derivation 5, 10 or 15 by depth and slice type.
fn rice_base_level(s: &Sequence, h: &SliceHeader) -> u32 {
    const STEP: u32 = 5;
    const DEEP_BITS: u32 = 12;
    if !s.range.rice_extension {
        return coeff::RiceContext::PLAIN.base_level;
    }
    let intra = u32::from(h.kind == 2);
    if s.depth.bits() > DEEP_BITS {
        STEP * (2 - intra)
    } else {
        STEP * (3 - intra)
    }
}

/// The largest QP of context initialisation (§9.3.2.2).
const MAX_CONTEXT_QP: i32 = 63;

fn validate_tools(s: &Sequence) -> Result<()> {
    // Each unsupported tool is named, so a corpus run reports what is missing.
    let unsupported = [
        (s.ctu_log2 > 7, "VVC CTU size"),
        // 16-bit samples need wider residuals and inter intermediates than
        // the 16-bit buffers the reconstruction uses.
        (
            s.depth.bits() > MAX_DECODED_BIT_DEPTH,
            "VVC bit depth above 12",
        ),
    ];
    match unsupported.iter().find(|(present, _)| *present) {
        Some(&(_, tool)) => Err(DecodeError::Unsupported(tool)),
        None => Ok(()),
    }
}

impl Parser<'_> {
    fn intra_ctu(&mut self, reader: &mut Reader<'_>, node: Node) -> Result<()> {
        if node.x >= self.dimensions[0] || node.y >= self.dimensions[1] {
            return Ok(());
        }
        if !self.sequence.chroma {
            return self.tree(reader, node, 0, false, Constraint::Intra);
        }
        if !self.sequence.dual_tree {
            // A single tree starts unconstrained so that small luma blocks
            // open a local dual tree (H.266 §7.4.12.4 modeTypeCondition).
            return self.tree(reader, node, 0, true, Constraint::Any);
        }
        // Dual-tree chroma follows luma independently within each 64x64 region.
        if node.width > 64 {
            // The joint 128-sample level starts groups for both trees.
            for channel in 0..2 {
                self.enter_group(node, channel);
            }
            for child in node.group_children(Split::Quad, true, self.max_subdiv()) {
                self.intra_ctu(reader, child)?;
            }
        } else {
            for channel in 0..2 {
                self.tree(reader, node, channel, false, Constraint::Intra)?;
            }
        }
        Ok(())
    }
    fn neighbour(&self, x: usize, y: usize, channel: usize) -> Option<&Unit> {
        if x >= self.dimensions[0] || y >= self.dimensions[1] {
            return None;
        }
        // Another tile's CUs are unavailable (H.266 §6.4.4).
        let cell = self.grid[channel][y / 4 * self.stride + x / 4];
        (cell != EMPTY_CELL)
            .then(|| &self.units[cell as usize])
            .filter(|unit| unit.tile == self.ctu.tile)
    }
    fn tree(
        &mut self,
        r: &mut Reader<'_>,
        node: Node,
        channel: usize,
        combined: bool,
        constraint: Constraint,
    ) -> Result<()> {
        if node.x >= self.dimensions[0] || node.y >= self.dimensions[1] {
            return Ok(());
        }
        if node.width < 4 || node.height < 4 || node.qt_depth + node.mtt_depth > 16 {
            return Err(DecodeError::Malformed("VVC coding tree depth/size"));
        }
        let dual = self.header.kind == 2 && self.sequence.chroma && self.sequence.dual_tree;
        self.enter_group(node, channel);
        // A local chroma tree contains the unsplit parent of constrained luma
        // children (H.266 §6.4.1-6.4.3: no split of an intra-only chroma tree),
        // in single-tree I slices as in P and B slices.
        if channel == 1 && !dual {
            return self.unit(r, node, channel, false, Constraint::Intra);
        }
        let partition = &self.header.picture.partitions[if self.header.kind != 2 {
            1
        } else if channel == 0 {
            0
        } else {
            2
        }];
        let limits = tree::Limits {
            min_qt: 1 << partition.min_qt_log2,
            min_cb: 1 << self.sequence.min_cb_log2,
            max_bt: 1 << partition.max_bt_log2,
            max_tt: 1 << partition.max_tt_log2,
            max_depth: partition.mtt_depth,
            chroma: (channel != 0).then(|| self.sequence.subsampling()),
            dual,
            inter_only: constraint == Constraint::Inter,
        };
        let neighbours = [
            self.neighbour(node.x.wrapping_sub(1), node.y, channel),
            self.neighbour(node.x, node.y.wrapping_sub(1), channel),
        ]
        .map(|u| u.map(|u| u.node));
        let split = self
            .trees
            .read(r, node, limits, self.dimensions, neighbours)?;
        if split == Split::None {
            return self.unit(r, node, channel, combined, constraint);
        }
        let child_constraint = self.mode_constraint(r, node, split, combined, constraint)?;
        let separate_chroma =
            combined && constraint == Constraint::Any && child_constraint == Constraint::Intra;
        let children = node.group_children(split, node.outside(self.dimensions), self.max_subdiv());
        for child in children {
            self.tree(
                r,
                child,
                channel,
                combined && !separate_chroma,
                child_constraint,
            )?;
        }
        if separate_chroma {
            self.tree(r, node, 1, false, Constraint::Intra)?;
        }
        Ok(())
    }

    fn mode_constraint(
        &mut self,
        r: &mut Reader<'_>,
        node: Node,
        split: Split,
        combined: bool,
        parent: Constraint,
    ) -> Result<Constraint> {
        // H.266 §7.4.12.4 modeTypeCondition: 0 in 4:4:4 (and monochrome,
        // which has no combined tree).
        if !combined || parent != Constraint::Any || self.sequence.format == ChromaFormat::Yuv444 {
            return Ok(parent);
        }
        let divisor = if matches!(
            split,
            Split::Quad | Split::TripleHorizontal | Split::TripleVertical
        ) {
            4
        } else {
            2
        };
        let area = node.width * node.height / divisor;
        let narrow = (node.width == 8 && split == Split::Vertical)
            || (node.width == 16 && split == Split::TripleVertical);
        // Children of 32 luma samples (BT of 64, TT of 128) constrain only
        // 4:2:0, where their chroma blocks would be 2 samples wide.
        let small = area < 64 && (area < 32 || self.sequence.format == ChromaFormat::Yuv420);
        if !small && !narrow {
            return Ok(parent);
        }
        // H.266 §7.4.12.4 modeTypeCondition: an I slice infers intra without a flag.
        if area < 32 || self.header.kind == 2 {
            return Ok(Constraint::Intra);
        }
        let context = usize::from(
            [
                self.neighbour(node.x.wrapping_sub(1), node.y, 0),
                self.neighbour(node.x, node.y.wrapping_sub(1), 0),
            ]
            .iter()
            .any(|u| u.is_some_and(Unit::intra)),
        );
        Ok(if r.bin(&mut self.constraint[context])? {
            Constraint::Intra
        } else {
            Constraint::Inter
        })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod inter_tests;

#[cfg(test)]
mod extended_tests;

#[cfg(test)]
mod b_reconstruction_tests;

#[cfg(test)]
mod general_inter_tests;
#[cfg(test)]
mod general_tools_tests;

#[cfg(test)]
pub(crate) use general_tools_tests::compare as compare_trace;
