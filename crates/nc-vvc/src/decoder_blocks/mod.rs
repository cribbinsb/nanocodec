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
pub(crate) use filter_syntax::counts as filter_counts;
pub(crate) use filters::apply as apply_filters;
pub(crate) struct Parsed {
    pub units: Vec<Unit>,
    pub filters: Vec<filter_syntax::CtuFilters>,
}
mod sbt;
mod skip_coeff;
pub(super) use deblock::apply as deblock;
mod reconstruct;
mod reshape;
pub(crate) use reshape::model as reshape_model;
mod references;
pub(super) use reconstruct::reconstruct;
mod affine_motion;
mod amvp_syntax;
mod initialization;
mod inter_models;
mod inter_prediction;
mod prof;
mod refinement;
pub(crate) use refinement::derive as derive_refinement;
mod inter_syntax;
mod intra_syntax;
mod isp;
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
    transforms: Vec<Transform>,
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
}

/// Decode supported general I/P syntax before reconstruction. Unsupported tools
/// are rejected before their conditionally present CABAC bins can be misread.
#[cfg(test)]
pub(super) fn parse(
    bytes: &[u8],
    s: &Sequence,
    p: &PictureParameters,
    h: &SliceHeader,
) -> Result<Vec<Unit>> {
    parse_filtered(bytes, s, p, h, filter_syntax::Counts::default()).map(|p| p.units)
}

pub(super) fn parse_filtered(
    bytes: &[u8],
    s: &Sequence,
    p: &PictureParameters,
    h: &SliceHeader,
    filter_counts: filter_syntax::Counts,
) -> Result<Parsed> {
    validate_tools(s, p, h)?;
    let dimensions = p.dimensions.map(|v| v as usize);
    let stride = dimensions[0].div_ceil(4);
    let cells = stride * dimensions[1].div_ceil(4);
    let qp = h.qp as u32;
    let inter = h.kind != 2;
    let init_kind = if h.cabac_init && inter {
        1 - h.kind
    } else {
        h.kind
    };
    let mut parser = Parser {
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
        constraint: initialization::bank([[25, 20], [25, 12], [35, 35]], [1, 0], qp, init_kind),
        ref_index: initialization::bank([[5, 35], [20, 35], [35, 35]], [0, 4], qp, init_kind),
        cclm: initialization::bank([[26, 27], [34, 27], [59, 27]], [4, 9], qp, init_kind),
        joint: initialization::bank(
            [[42, 43, 52], [27, 36, 45], [12, 21, 35]],
            [1, 1, 0],
            qp,
            init_kind,
        ),
    };
    let mut reader = Reader::new(
        bytes
            .get(h.payload_offset..)
            .ok_or(DecodeError::Truncated)?,
    )?;
    let ctu = 1 << s.ctu_log2;
    let columns = dimensions[0].div_ceil(ctu);
    let mut filters = Vec::new();
    let mut filter_models = filter_syntax::Models::new(qp, init_kind);
    for y in (0..dimensions[1]).step_by(ctu) {
        for x in (0..dimensions[0]).step_by(ctu) {
            let index = filters.len();
            let neighbours = [
                if x > 0 {
                    Some(filters[index - 1])
                } else {
                    None
                },
                if y > 0 {
                    Some(filters[index - columns])
                } else {
                    None
                },
            ];
            filters.push(filter_models.read(
                &mut reader,
                &mut parser.contexts,
                h,
                neighbours,
                filter_counts,
            )?);
            if inter {
                parser.tree(
                    &mut reader,
                    Node::root(x, y, ctu),
                    0,
                    s.chroma,
                    Constraint::Any,
                )?;
            } else {
                parser.intra_ctu(&mut reader, Node::root(x, y, ctu))?;
            }
        }
    }
    if !reader.terminate()? {
        return Err(DecodeError::Malformed("VVC slice CABAC termination"));
    }
    Ok(Parsed {
        units: parser.units,
        filters,
    })
}
fn validate_tools(s: &Sequence, p: &PictureParameters, h: &SliceHeader) -> Result<()> {
    if h.kind != 2 && (s.tools.weighted.contains(&true) || s.tools.wraparound)
        || !h.picture.output
        || h.picture.gdr
        || !s.tools.ladf.is_empty()
        || h.picture.boundaries.iter().any(|v| !v.is_empty())
        || s.ctu_log2 > 7
        || s.tools.palette
        || s.tools.ibc
        || p.cu_qp_delta
        || h.cu_chroma_offsets
        || h.scaling
        || s.entropy_sync
        || !h.entry_offsets.is_empty()
        || p.partitioning.tiles.iter().any(|v| v.len() != 1)
    {
        return Err(DecodeError::Unsupported("VVC general coding tools"));
    }
    Ok(())
}

impl Parser<'_> {
    fn intra_ctu(&mut self, reader: &mut Reader<'_>, node: Node) -> Result<()> {
        if node.x >= self.dimensions[0] || node.y >= self.dimensions[1] {
            return Ok(());
        }
        if !self.sequence.chroma || !self.sequence.dual_tree {
            return self.tree(reader, node, 0, self.sequence.chroma, Constraint::Intra);
        }
        // Dual-tree chroma follows luma independently within each 64x64 region.
        if node.width > 64 {
            for child in node.children(Split::Quad, true) {
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
        self.grid[channel][y / 4 * self.stride + x / 4].map(|index| &self.units[index])
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
        // A local chroma tree contains the unsplit parent of constrained luma children.
        if channel == 1 && self.header.kind != 2 {
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
            dual: self.header.kind == 2 && self.sequence.chroma && self.sequence.dual_tree,
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
        for child in node.children(split, node.outside(self.dimensions)) {
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
        if area < 32 {
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
