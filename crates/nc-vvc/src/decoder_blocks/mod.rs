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
pub(crate) use filter_syntax::{CtuFilters, counts as filter_counts};
pub(crate) use filters::apply as apply_filters;
mod sbt;
mod scaling_lists;
pub(crate) use scaling_lists::of_slice as scaling_lists;
mod skip_coeff;
pub(super) use deblock::apply as deblock;
mod reconstruct;
mod reshape;
pub(crate) use reshape::model as reshape_model;
mod references;
pub(super) use inter_prediction::Sources as InterSources;
pub(super) use reconstruct::reconstruct;
#[cfg(test)]
use reconstruct::reconstructed;
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
pub(crate) use refinement::{Edges, derive as derive_refinement};
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
pub(super) use motion::derive as derive_motion;
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
use nc_common::decode::{DecodeError, Result};
pub(super) use transform::inverse_residuals;
use tree::{Node, Split};

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
    prof: bool,
    smvd: bool,
    bdof: bool,
    dmvr: Vec<refinement::Region>,
    lfnst: u8,
    mts: u8,
    luma_mode: u8,
    chroma_mode: u8,
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
    /// Whether this is an IBC CU (known from its syntax, before its vector
    /// is resolved).
    fn block_vector_signalled(&self) -> bool {
        matches!(self.inter, Some(Inter::Ibc(_)))
    }
}
struct Parser<'a> {
    sequence: &'a Sequence,
    header: &'a SliceHeader,
    dimensions: [usize; 2],
    stride: usize,
    grid: [Vec<Option<usize>>; 2],
    units: Vec<Unit>,
    contexts: Contexts,
    trees: tree::Models,
    coefficients: coeff::Models,
    skip_coefficients: skip_coeff::Models,
    cclm: [Context; 2],
    intra_models: intra_syntax::Models,
    joint: [Context; 3],
    constraint: [Context; 2],
    ref_index: [Context; 2],
    inter_models: inter_models::Models,
    transform_models: transform_syntax::Models,
    sbt_models: sbt::Models,
    ibc_models: ibc::Models,
    groups: quant_groups::QuantGroups,
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
    parse_filtered(
        bytes,
        s,
        p,
        h,
        filter_syntax::Counts::default(),
        (0, no_backward_prediction(h)),
        &mut filters,
    )
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

pub(super) fn parse_filtered(
    bytes: &[u8],
    s: &Sequence,
    p: &PictureParameters,
    h: &SliceHeader,
    filter_counts: filter_syntax::Counts,
    (slice, no_backward): (u16, bool),
    filters: &mut [filter_syntax::CtuFilters],
) -> Result<Vec<Unit>> {
    validate_tools(s)?;
    let dimensions = p.dimensions.map(|v| v as usize);
    let inter = h.kind != 2;
    let scan = layout::scan(&h.ctus, &p.partitioning.tiles, s.ctu_log2);
    let mut parser = Parser::new(s, p, h, scan[0]);
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
    // CTUs of other slices are unavailable (§6.4.4), as are other tiles'.
    let mut in_slice = vec![false; filters.len()];
    for position in &scan {
        in_slice[position.address] = true;
    }
    let mut filter_models = filter_syntax::Models::new(qp, init_kind);
    let initial = parser.save_contexts(&filter_models);
    // Wavefront: the context variables after the first CTU of the row above.
    let mut row_start = None;
    for (index, &position) in scan.iter().enumerate() {
        // A tile, and with entropy synchronisation each CTU row of a tile,
        // is a CABAC substream: a terminating bin and byte alignment end the
        // one before (H.266 §7.3.11.1).
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
        // The CTUs to the left and above, when they are in the same tile
        // and slice.
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
    Ok(parser.units)
}
impl<'a> Parser<'a> {
    fn new(s: &'a Sequence, p: &PictureParameters, h: &'a SliceHeader, first: layout::Ctu) -> Self {
        let dimensions = p.dimensions.map(|v| v as usize);
        let stride = dimensions[0].div_ceil(4);
        let cells = stride * dimensions[1].div_ceil(4);
        let inter = h.kind != 2;
        let (qp, init_kind) = initialization_of(h);
        Self {
            ctu: first,
            slice: 0,
            no_backward_prediction: false,
            sequence: s,
            header: h,
            dimensions,
            stride,
            grid: [vec![None; cells], vec![None; cells]],
            units: Vec::new(),
            contexts: initialization::contexts(qp, init_kind),
            inter_models: inter_models::Models::new(qp, init_kind),
            trees: tree::Models::new(qp, init_kind),
            coefficients: coeff::Models::new(qp, init_kind),
            skip_coefficients: skip_coeff::Models::new(qp, init_kind),
            intra_models: intra_syntax::Models::new(qp, init_kind),
            transform_models: transform_syntax::Models::new(qp, init_kind),
            sbt_models: sbt::Models::new(qp, init_kind),
            ibc_models: ibc::Models::new(qp, init_kind),
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

/// The largest QP of context initialisation (§9.3.2.2).
const MAX_CONTEXT_QP: i32 = 63;

fn validate_tools(s: &Sequence) -> Result<()> {
    // Each unsupported tool is named, so a corpus run reports what is missing.
    let unsupported = [
        (s.ctu_log2 > 7, "VVC CTU size"),
        (s.tools.palette, "VVC palette mode"),
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
        self.grid[channel][y / 4 * self.stride + x / 4]
            .map(|index| &self.units[index])
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
            chroma: channel != 0,
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
        if !combined || parent != Constraint::Any {
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
        if area >= 64 && !narrow {
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
            .any(|u| u.is_some_and(|u| u.inter.is_none())),
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
