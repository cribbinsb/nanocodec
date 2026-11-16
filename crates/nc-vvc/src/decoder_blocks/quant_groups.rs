//! Quantisation groups: CU QP deltas and CU chroma QP offsets (H.266
//! §7.3.11.10, §7.4.12.10, §8.7.1). Group starts, QP prediction and the
//! binarisations follow the VTM reference decoder (`CABACReader::coding_tree`,
//! `cu_qp_delta`, `cu_chroma_qp_offset`, `CU::predictQP`; BSD 3-clause, see
//! `THIRD_PARTY.md`).
use super::{Context, Node, Parser, Reader, Result, Split, Unit, initialization::bank};
use nc_common::decode::DecodeError;

/// `slice_type` of I slices (H.266 §7.4.8.1).
const I_SLICE: u8 = 2;
/// `cu_qp_delta_abs` prefix: truncated unary up to this value, then EG0.
const DELTA_PREFIX_MAX: u32 = 5;
/// Luma QP values wrap modulo this plus `QpBdOffset` (H.266 §8.7.1).
const QP_RANGE: i32 = 64;
/// Longest Exp-Golomb prefix accepted for a QP delta: any conforming value
/// (at most 32 after the prefix) fits well inside it.
const MAX_GOLOMB_PREFIX: u32 = 8;

pub(super) struct QuantGroups {
    /// Whether the PPS allows CU QP deltas and the slice CU chroma offsets.
    enabled: [bool; 2],
    /// Deepest subdivision that still starts a group, per kind.
    max_subdiv: [u8; 2],
    /// The CU chroma QP offset list of the PPS (Cb, Cr, joint).
    offsets: Vec<[i8; 3]>,
    /// QP of the latest luma CU in decoding order (`qPY_PREV`), then of the
    /// current one.
    qp: i8,
    /// A group started and its QP has not been predicted yet.
    start: bool,
    delta_coded: bool,
    /// `IsCuChromaQpOffsetCoded`, per coding tree of an I slice's dual
    /// tree; a single tree and its local dual trees share entry 0 (H.266
    /// §7.3.11.4: one variable, which a local chroma tree inherits).
    offset_coded: [bool; 2],
    /// Index + 1 into `offsets` in force (0: no CU offset).
    offset_index: u8,
    slice_qp: i8,
    /// `QpBdOffset`: luma QPs range from its negative to 63.
    qp_bd_offset: i8,
    models: Models,
}

/// Context variables of `cu_qp_delta_abs`, `cu_chroma_qp_offset_flag` and
/// `cu_chroma_qp_offset_idx`.
#[derive(Clone, Copy)]
pub(super) struct Models {
    delta: [Context; 2],
    offset_flag: [Context; 1],
    offset_idc: [Context; 1],
}

impl QuantGroups {
    pub fn new(
        enabled: [bool; 2],
        max_subdiv: [u8; 2],
        offsets: &[[i8; 3]],
        (slice_qp, qp_bd_offset): (i8, i8),
        init_kind: u8,
    ) -> Self {
        let qp = slice_qp.max(0) as u32;
        Self {
            enabled,
            max_subdiv,
            offsets: offsets.to_vec(),
            qp: slice_qp,
            start: false,
            delta_coded: false,
            offset_coded: [false; 2],
            offset_index: 0,
            slice_qp,
            qp_bd_offset,
            models: Models {
                delta: bank([[35, 35]; 3], [8, 8], qp, init_kind),
                offset_flag: bank([[35]; 3], [8], qp, init_kind),
                offset_idc: bank([[35]; 3], [8], qp, init_kind),
            },
        }
    }

    pub fn models(&self) -> Models {
        self.models
    }

    /// A new CABAC substream that resets QP prediction (a wavefront row):
    /// its context variables, and the slice QP as `qPY_PREV` (§8.7.1).
    pub fn restart(&mut self, models: Models) {
        self.models = models;
        self.qp = self.slice_qp;
    }
}

impl Node {
    /// This node's children with their subdivision depth and group flags
    /// (§7.3.11.4 `cbSubdiv`, `qgOnY`, `qgOnC`).
    pub(super) fn group_children(self, split: Split, implicit: bool, max: [u8; 2]) -> Vec<Self> {
        let triple = matches!(split, Split::TripleHorizontal | Split::TripleVertical);
        let mut children = self.children(split, implicit);
        for (index, child) in children.iter_mut().enumerate() {
            let step = match split {
                Split::Quad => 2,
                _ if triple && index != 1 => 2,
                _ => 1,
            };
            child.subdiv = self.subdiv + step;
            // All three parts of a ternary split share the outer parts' test.
            let tested = self.subdiv + if triple { 2 } else { step };
            child.groups = [0, 1].map(|kind| self.groups[kind] && tested <= max[kind]);
        }
        children
    }
}

impl Parser<'_> {
    pub(super) fn max_subdiv(&self) -> [u8; 2] {
        self.groups.max_subdiv
    }

    /// A coding-tree node that may start a quantisation group.
    pub(super) fn enter_group(&mut self, node: Node, channel: usize) {
        let tree = self.offset_tree(channel);
        let g = &mut self.groups;
        if g.enabled[0] && node.groups[0] && channel == 0 {
            g.start = true;
            g.delta_coded = false;
        }
        if g.enabled[1] && node.groups[1] {
            g.offset_coded[tree] = false;
            g.offset_index = 0;
        }
    }

    /// The `offset_coded` entry of `channel`'s tree: its own in an I
    /// slice's dual tree, else the shared one.
    fn offset_tree(&self, channel: usize) -> usize {
        let dual = self.header.kind == I_SLICE && self.sequence.chroma && self.sequence.dual_tree;
        if dual { channel } else { 0 }
    }

    /// QP state at the start of a CU: predicts the group's QP at its first
    /// CU, and gives a chroma-tree CU the QP of the luma CU at its centre.
    /// Returns the luma QP to restore after a chroma-tree CU.
    pub(super) fn begin_unit_qp(&mut self, unit: &mut Unit) -> Option<i8> {
        if self.groups.start {
            self.groups.start = false;
            self.groups.qp = self.predicted_qp(unit.node);
        }
        let mut saved = None;
        if self.groups.enabled[0] && unit.channel == 1 {
            let n = unit.node;
            if let Some(luma) = self.neighbour(n.x + n.width / 2, n.y + n.height / 2, 0) {
                saved = Some(self.groups.qp);
                self.groups.qp = luma.qp;
            }
        }
        unit.qp = self.groups.qp;
        saved
    }

    /// Record the CU's final QP and chroma offsets once its residual syntax
    /// is read (a delta or offset coded in a transform unit applies to the
    /// whole CU).
    pub(super) fn end_unit_qp(&mut self, unit: &mut Unit, saved: Option<i8>) {
        unit.qp = self.groups.qp;
        unit.chroma_offsets = match self.groups.offset_index {
            0 => [0; 3],
            index => self.groups.offsets[usize::from(index) - 1],
        };
        if let Some(luma) = saved {
            self.groups.qp = luma;
        }
    }

    /// §8.7.1 `qPY_PRED` for the group whose first CU is `node`.
    fn predicted_qp(&self, node: Node) -> i8 {
        let mask = (1usize << self.sequence.ctu_log2) - 1;
        let previous = self.groups.qp;
        let at = |x: usize, y: usize| self.neighbour(x, y, 0).map(|unit| unit.qp);
        let above = at(node.x, node.y.wrapping_sub(1));
        // The first group of a CTB row in a tile continues from the CTB above.
        if self.ctu.starts_row()
            && (node.x, node.y) == (self.ctu.x, self.ctu.y)
            && let Some(above) = above
        {
            return above;
        }
        let a = if node.y & mask == 0 { None } else { above }.unwrap_or(previous);
        let b = if node.x & mask == 0 {
            None
        } else {
            at(node.x.wrapping_sub(1), node.y)
        }
        .unwrap_or(previous);
        ((i16::from(a) + i16::from(b) + 1) >> 1) as i8
    }

    /// The QP delta and chroma offset syntax of one transform unit, after its
    /// coded flags (§7.3.11.10).
    pub(super) fn unit_qp_syntax(
        &mut self,
        r: &mut Reader<'_>,
        unit: &Unit,
        luma_coded: bool,
        chroma_coded: bool,
    ) -> Result<()> {
        let n = unit.node;
        let large = n.width > 64 || n.height > 64;
        let separate = !unit.combined && self.sequence.chroma;
        if (large || luma_coded || chroma_coded)
            && (!separate || unit.channel == 0)
            && self.groups.enabled[0]
            && !self.groups.delta_coded
        {
            let delta = self.qp_delta(r)?;
            let g = &mut self.groups;
            // QpY = ((qPY_PRED + CuQpDeltaVal + 64 + 2 · QpBdOffset) %
            // (64 + QpBdOffset)) − QpBdOffset (§8.7.1).
            let offset = i32::from(g.qp_bd_offset);
            g.qp =
                ((i32::from(g.qp) + delta + offset).rem_euclid(QP_RANGE + offset) - offset) as i8;
            g.delta_coded = true;
        }
        if (!separate || unit.channel == 1)
            && self.groups.enabled[1]
            && !self.groups.offset_coded[self.offset_tree(unit.channel)]
        {
            // A chroma tree measures its own (half-size) block.
            let shift = usize::from(separate);
            if n.width >> shift > 64 || n.height >> shift > 64 || chroma_coded {
                self.groups.offset_index = self.chroma_offset_index(r)?;
                let tree = self.offset_tree(unit.channel);
                self.groups.offset_coded[tree] = true;
            }
        }
        Ok(())
    }

    fn qp_delta(&mut self, r: &mut Reader<'_>) -> Result<i32> {
        let [first, rest] = &mut self.groups.models.delta;
        let mut magnitude = u32::from(r.bin(first)?);
        while magnitude != 0 && magnitude < DELTA_PREFIX_MAX && r.bin(rest)? {
            magnitude += 1;
        }
        if magnitude == DELTA_PREFIX_MAX {
            let mut prefix = 0;
            while r.bypass()? {
                prefix += 1;
                if prefix > MAX_GOLOMB_PREFIX {
                    return Err(DecodeError::Malformed("VVC CU QP delta"));
                }
            }
            magnitude += ((1 << prefix) - 1) + r.bypass_bits(prefix)?;
        }
        let negative = magnitude != 0 && r.bypass()?;
        let magnitude = magnitude as i32;
        Ok(if negative { -magnitude } else { magnitude })
    }

    fn chroma_offset_index(&mut self, r: &mut Reader<'_>) -> Result<u8> {
        let g = &mut self.groups;
        if !r.bin(&mut g.models.offset_flag[0])? {
            return Ok(0);
        }
        let mut index = 1;
        while usize::from(index) < g.offsets.len() && r.bin(&mut g.models.offset_idc[0])? {
            index += 1;
        }
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subdivision_counts_two_per_quad_and_one_per_binary_split() {
        let root = Node::root(0, 0, 128);
        let quad = root.group_children(Split::Quad, false, [2, 0]);
        assert!(
            quad.iter()
                .all(|n| n.subdiv == 2 && n.groups == [true, false])
        );
        let binary = quad[0].group_children(Split::Horizontal, false, [2, 0]);
        assert!(
            binary
                .iter()
                .all(|n| n.subdiv == 3 && n.groups == [false, false])
        );
    }

    #[test]
    fn ternary_parts_share_the_outer_parts_group_test() {
        let root = Node::root(0, 0, 64);
        let parts = root.group_children(Split::TripleVertical, false, [1, 2]);
        let subdivs: Vec<_> = parts.iter().map(|n| n.subdiv).collect();
        assert_eq!(subdivs, [2, 1, 2]);
        // The middle part has subdivision 1 but is tested as 2, like the others.
        assert!(parts.iter().all(|n| n.groups == [false, true]));
    }
}
