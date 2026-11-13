//! Entropy coding of a picture's slice data from recorded decisions.
//!
//! The backend decides and reconstructs every CU during the engine's
//! partition walk, but writes nothing yet: SAO parameters (HEVC §7.3.8.3)
//! come first in each CTU and depend on the deblocked picture, which only
//! exists once every CU is reconstructed. So the walk records its decisions
//! as [`Item`]s, and [`write_slice_data`] codes them, with the SAO syntax in
//! place, after the in-loop filters have run.
//!
//! Context selection reads the finished [`Grid`]; neighbour availability
//! follows z-scan order, so it sees exactly the CUs a decoder has decoded.

use nc_common::BitWriter;
use nc_encoder::{Block, Mv};

use crate::cabac::CabacWriter;
use crate::contexts::{Contexts, SliceKind};
use crate::grid::Grid;
use crate::headers::{LOG2_CTB, LOG2_MIN_CB, LOG2_MIN_TB, SIGN_DATA_HIDING};
use crate::mvpred::MAX_MERGE_CANDIDATES;
use crate::predict::{CHROMA_DM, ModeCode};
use crate::quant_group::QuantGroup;
use crate::residual::write_residual;
use crate::sao::{self, PictureSao};
use crate::scan::{ScanKind, Scans};

/// `cu_qp_delta_abs` prefix: truncated unary bins before the Exp-Golomb
/// suffix (§9.3.3.10).
const QP_DELTA_PREFIX: u32 = 5;
/// Exp-Golomb order of the `cu_qp_delta_abs` suffix.
const QP_DELTA_EG_ORDER: u32 = 0;
/// Bits of `rem_intra_luma_pred_mode` (fixed length, bypass).
const REM_INTRA_MODE_BITS: u32 = 5;
/// Bits of `intra_chroma_pred_mode` 0..=3 after its first bin (bypass).
const CHROMA_PRED_MODE_BITS: u32 = 2;
/// Exp-Golomb order of `abs_mvd_minus2` (§9.3.3.x, EG1).
pub const MVD_EG_ORDER: u32 = 1;

/// One plane's quantised residual for a transform block.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlaneResidual {
    /// log2 of the transform side.
    pub log2: u8,
    /// Coefficient scan order (`scanIdx`).
    pub scan: ScanKind,
    /// Start of the block's row-major levels in the picture's level store
    /// (`side²` entries; meaningful only when coded).
    pub levels: u32,
    /// Any level is non-zero (`cbf`).
    pub coded: bool,
}

impl PlaneResidual {
    /// The block's levels in `store`.
    #[must_use]
    pub fn levels(self, store: &[i16]) -> &[i16] {
        let start = self.levels as usize;
        &store[start..start + (1 << (2 * self.log2))]
    }
}

/// The three residual blocks of one transform unit.
#[derive(Debug, Clone, Copy, Default)]
pub struct TransformUnit {
    /// Luma position of the unit.
    pub x: usize,
    /// Luma position of the unit.
    pub y: usize,
    /// log2 of the luma side.
    pub log2: u8,
    /// Luma residual.
    pub luma: PlaneResidual,
    /// Cb residual.
    pub cb: PlaneResidual,
    /// Cr residual.
    pub cr: PlaneResidual,
}

impl TransformUnit {
    /// Any of the three blocks has coefficients.
    #[must_use]
    pub fn coded(&self) -> bool {
        self.luma.coded || self.cb.coded || self.cr.coded
    }
}

/// The luma intra modes of an intra CU, coded against each prediction
/// unit's candidate list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraLuma {
    /// `PART_2Nx2N`: one mode.
    Whole(ModeCode),
    /// `PART_NxN` (8x8 CUs only, §7.3.8.5): one mode per 4x4 prediction
    /// unit, z order. `IntraSplitFlag` forces the first transform split.
    Quarters([ModeCode; 4]),
}

impl IntraLuma {
    /// The codes in syntax order.
    #[must_use]
    pub fn codes(&self) -> &[ModeCode] {
        match self {
            Self::Whole(code) => std::slice::from_ref(code),
            Self::Quarters(codes) => codes,
        }
    }

    /// `IntraSplitFlag`.
    #[must_use]
    pub fn split(self) -> bool {
        matches!(self, Self::Quarters(_))
    }
}

/// How a CU is coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CuKind {
    /// Intra, with the luma mode(s) coded as `luma`.
    Intra {
        /// The luma modes against their candidate lists.
        luma: IntraLuma,
        /// `intra_chroma_pred_mode` (`predict::CHROMA_DM`: chroma follows
        /// luma).
        chroma: u8,
    },
    /// `cu_skip_flag`: merge candidate `merge_idx`, no residual.
    Skip {
        /// Merge candidate index.
        merge_idx: usize,
    },
    /// Merge candidate `merge_idx` with a residual.
    Merge {
        /// Merge candidate index.
        merge_idx: usize,
    },
    /// AMVP: `mv` coded as a difference to AMVP candidate `mvp_idx` (`pred`).
    Amvp {
        /// The vector.
        mv: Mv,
        /// The predictor it is coded against.
        pred: Mv,
        /// `mvp_l0_flag`.
        mvp_idx: usize,
    },
}

/// Most units: a 64x64 CU holds sixteen 16x16 transforms, an 8x8 or 16x16
/// CU sixteen 4x4 / 8x8 ones at most at the depths the backend uses.
pub const MAX_TUS: usize = 16;
/// Default TU32 needs at most four inline entries per CU.
const INLINE_TUS: usize = 4;

/// Keep the default record footprint small. Only the optional16x16 maximum
/// with a64x64 CU spills to one contiguous allocation.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum TuStorage {
    Inline([TransformUnit; INLINE_TUS]),
    Overflow(Box<[TransformUnit; MAX_TUS]>),
}

/// The transform units of one CU in coding order. Default TU32 stays inline.
#[derive(Debug)]
pub struct TransformUnits {
    storage: TuStorage,
    len: usize,
}

impl Default for TransformUnits {
    fn default() -> Self {
        Self {
            storage: TuStorage::Inline([TransformUnit::default(); INLINE_TUS]),
            len: 0,
        }
    }
}

impl TransformUnits {
    /// Append a unit, spilling once only when a fifth unit is needed.
    pub fn push(&mut self, tu: TransformUnit) {
        assert!(self.len < MAX_TUS, "too many transform units in one CU");
        if self.len == INLINE_TUS
            && let TuStorage::Inline(inline) = &self.storage
        {
            let mut units = Box::new([TransformUnit::default(); MAX_TUS]);
            units[..INLINE_TUS].copy_from_slice(inline);
            self.storage = TuStorage::Overflow(units);
        }
        match &mut self.storage {
            TuStorage::Inline(units) => units[self.len] = tu,
            TuStorage::Overflow(units) => units[self.len] = tu,
        }
        self.len += 1;
    }

    /// Keep only the first `len` units (a losing coding trial's are
    /// dropped).
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// The units, contiguous for both storage forms.
    #[must_use]
    pub fn as_slice(&self) -> &[TransformUnit] {
        match &self.storage {
            TuStorage::Inline(units) => &units[..self.len],
            TuStorage::Overflow(units) => &units[..self.len],
        }
    }

    /// The units, writable (chroma coded after a luma-decided tree).
    pub fn as_mut_slice(&mut self) -> &mut [TransformUnit] {
        match &mut self.storage {
            TuStorage::Inline(units) => &mut units[..self.len],
            TuStorage::Overflow(units) => &mut units[..self.len],
        }
    }
}

/// A CU's transform units in the picture's unit store: the recorded
/// units live in one per-picture list (as the levels do in the level
/// store), so an item stays small whatever its CU holds.
#[derive(Debug, Clone, Copy, Default)]
pub struct TuRange {
    start: u32,
    len: u32,
}

impl TuRange {
    /// The `len` units from `start` of the store.
    #[must_use]
    pub fn new(start: usize, len: usize) -> Self {
        Self {
            start: u32::try_from(start).expect("unit store under 4 Gi entries"),
            len: u32::try_from(len).expect("at most MAX_TUS units"),
        }
    }

    /// The units in `store`.
    #[must_use]
    pub fn units(self, store: &[TransformUnit]) -> &[TransformUnit] {
        let start = self.start as usize;
        &store[start..start + self.len as usize]
    }
}

/// One recorded CU.
#[derive(Debug)]
pub struct CuRecord {
    /// The CU.
    pub block: Block,
    /// Its coding mode.
    pub kind: CuKind,
    /// Its transform units in coding order.
    pub tus: TuRange,
}

/// One element of the slice data, in coding order. Items are written per
/// CU and read back by the slice writer at the end of the frame, so their
/// size is memory traffic: the transform units live in their own store.
#[derive(Debug)]
pub enum Item {
    /// Start of the CTU with top-left luma sample `(x, y)`.
    Ctu {
        /// Luma position.
        x: usize,
        /// Luma position.
        y: usize,
    },
    /// A partition decision of the coding quadtree.
    Split {
        /// The node.
        node: Block,
        /// Split into four.
        split: bool,
    },
    /// A coding unit.
    Cu(CuRecord),
}

/// Write `slice_segment_data()` for `items` and return its bits, including
/// the trailing bits. `sao` holds the SAO parameters and the components they
/// cover when SAO is on. With `block_qp` (`cu_qp_delta_enabled_flag`) each
/// CU's `QpY` is read from `grid` and coded as `cu_qp_delta`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn write_slice_data(
    items: &[Item],
    (levels, units): (&[i16], &[TransformUnit]),
    slice: SliceKind,
    qp: i32,
    block_qp: bool,
    grid: &Grid,
    transform: TransformSyntax<'_>,
    sao: Option<(&PictureSao, sao::Components)>,
) -> BitWriter {
    let tables = Tables {
        transform,
        levels,
        units,
    };
    let mut w = CabacWriter::new();
    // Contexts are initialised at the start of every slice (§9.3.2.2).
    let mut ctx = Contexts::for_slice(slice, qp);
    let mut group = block_qp.then(|| QuantGroup::new(qp));
    let mut first_ctu = true;
    for item in items {
        match item {
            Item::Ctu { x, y } => {
                if let Some(group) = group.as_mut() {
                    group.begin_group();
                }
                if !first_ctu {
                    // end_of_slice_segment_flag of the previous CTU.
                    w.encode_terminate(false);
                }
                first_ctu = false;
                if let Some((sao, components)) = sao {
                    sao::write(
                        &mut w,
                        &mut ctx,
                        sao,
                        components,
                        (x >> LOG2_CTB, y >> LOG2_CTB),
                    );
                }
            }
            Item::Split { node, split } => {
                if split_flag_present(grid, node) {
                    let ctx_inc = split_ctx_inc(grid, node);
                    w.encode_bin(&mut ctx.split_cu_flag[ctx_inc], *split);
                }
            }
            Item::Cu(cu) => {
                let mut qp = group.as_mut().map(|group| CuQp {
                    group,
                    qp: i32::from(grid.at(cu.block.x, cu.block.y).qp),
                });
                write_cu(&mut w, &mut ctx, grid, &tables, slice, cu, &mut qp);
                if let Some(CuQp { group, qp }) = qp {
                    let residual = cu.tus.units(tables.units).iter().any(TransformUnit::coded);
                    let coded = group.finish_cu(qp, residual);
                    debug_assert_eq!(coded, qp, "recorded QpY disagrees with the deltas");
                }
            }
        }
    }
    // end_of_slice_segment_flag = 1, then rbsp_slice_segment_trailing_bits().
    w.encode_terminate(true);
    let mut data = w.finish();
    data.trailing_bits();
    data
}

/// `split_cu_flag` is present only for nodes inside the picture and above
/// the minimum size (§7.3.8.4); otherwise it is inferred.
#[must_use]
pub fn split_flag_present(grid: &Grid, node: &Block) -> bool {
    node.log2_size() > LOG2_MIN_CB as u8 && node.is_inside(grid.width(), grid.height())
}

/// `ctxInc` of `split_cu_flag` (§9.3.4.2.2): one for each available left /
/// above neighbour that is deeper than `node`.
#[must_use]
pub fn split_ctx_inc(grid: &Grid, node: &Block) -> usize {
    let depth = LOG2_CTB as u8 - node.log2_size();
    let deeper = |xn: isize, yn: isize| grid.left_or_above(xn, yn).is_some_and(|c| c.depth > depth);
    let (x, y) = (node.x as isize, node.y as isize);
    usize::from(deeper(x - 1, y)) + usize::from(deeper(x, y - 1))
}

/// `ctxInc` of `cu_skip_flag` (§9.3.4.2.2): skipped available neighbours.
fn skip_ctx_inc(grid: &Grid, block: &Block) -> usize {
    let skipped = |xn: isize, yn: isize| grid.left_or_above(xn, yn).is_some_and(|c| c.skip);
    let (x, y) = (block.x as isize, block.y as isize);
    usize::from(skipped(x - 1, y)) + usize::from(skipped(x, y - 1))
}

/// SPS transform-tree constraints and coefficient scan tables.
pub struct TransformSyntax<'a> {
    /// Normative coefficient scan tables.
    pub scans: &'a Scans,
    /// SPS maximum transform log2.
    pub max_log2: u8,
    /// SPS `max_transform_hierarchy_depth_inter`.
    pub inter_depth: u8,
    /// SPS `max_transform_hierarchy_depth_intra`.
    pub intra_depth: u8,
}

/// The quantization group of the CU being written and the CU's `QpY`
/// (with `cu_qp_delta_enabled_flag`).
struct CuQp<'g> {
    group: &'g mut QuantGroup,
    qp: i32,
}

struct Tables<'a> {
    transform: TransformSyntax<'a>,
    /// The picture's level store.
    levels: &'a [i16],
    /// The picture's transform unit store.
    units: &'a [TransformUnit],
}

fn write_cu(
    w: &mut CabacWriter,
    ctx: &mut Contexts,
    grid: &Grid,
    tables: &Tables<'_>,
    slice: SliceKind,
    cu: &CuRecord,
    qp: &mut Option<CuQp<'_>>,
) {
    let block = &cu.block;
    if slice == SliceKind::Predicted {
        let skip = matches!(cu.kind, CuKind::Skip { .. });
        w.encode_bin(&mut ctx.cu_skip_flag[skip_ctx_inc(grid, block)], skip);
    }
    match cu.kind {
        CuKind::Skip { merge_idx } => write_merge_idx(w, ctx, merge_idx),
        CuKind::Intra { luma, chroma } => {
            if slice == SliceKind::Predicted {
                w.encode_bin(&mut ctx.pred_mode_flag, true); // MODE_INTRA
            }
            if block.log2_size() == LOG2_MIN_CB as u8 {
                // part_mode: 1 for PART_2Nx2N, 0 for PART_NxN.
                w.encode_bin(&mut ctx.part_mode, !luma.split());
            }
            write_intra_luma_modes(w, ctx, luma.codes());
            // intra_chroma_pred_mode: "0" for DM, else "1" and two bypass bits.
            w.encode_bin(&mut ctx.intra_chroma_pred_mode, chroma != CHROMA_DM);
            if chroma != CHROMA_DM {
                w.encode_bypass(u32::from(chroma), CHROMA_PRED_MODE_BITS);
            }
            let tree = TreeKind::Intra {
                split: luma.split(),
            };
            write_transform_tree(w, ctx, tables, block, cu.tus.units(tables.units), tree, qp);
        }
        CuKind::Merge { merge_idx } => {
            w.encode_bin(&mut ctx.pred_mode_flag, false); // MODE_INTER
            w.encode_bin(&mut ctx.part_mode, true); // PART_2Nx2N
            w.encode_bin(&mut ctx.merge_flag, true);
            write_merge_idx(w, ctx, merge_idx);
            // rqt_root_cbf is inferred 1 for a merged 2Nx2N CU.
            write_transform_tree(
                w,
                ctx,
                tables,
                block,
                cu.tus.units(tables.units),
                TreeKind::Inter,
                qp,
            );
        }
        CuKind::Amvp { mv, pred, mvp_idx } => {
            w.encode_bin(&mut ctx.pred_mode_flag, false); // MODE_INTER
            w.encode_bin(&mut ctx.part_mode, true); // PART_2Nx2N
            w.encode_bin(&mut ctx.merge_flag, false);
            write_mvd(w, ctx, mv, pred);
            w.encode_bin(&mut ctx.mvp_flag, mvp_idx == 1);
            let residual = cu.tus.units(tables.units).iter().any(TransformUnit::coded);
            w.encode_bin(&mut ctx.rqt_root_cbf, residual);
            if residual {
                write_transform_tree(
                    w,
                    ctx,
                    tables,
                    block,
                    cu.tus.units(tables.units),
                    TreeKind::Inter,
                    qp,
                );
            }
        }
    }
}

/// Which transform hierarchy limit a CU's tree follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeKind {
    /// `max_transform_hierarchy_depth_inter`.
    Inter,
    /// `max_transform_hierarchy_depth_intra`, plus one level forced by
    /// `IntraSplitFlag` (`split`).
    Intra {
        /// `IntraSplitFlag` (`PART_NxN`).
        split: bool,
    },
}

impl TreeKind {
    /// `MaxTrafoDepth` (§7.4.9.8) given the SPS depths.
    #[must_use]
    pub fn max_depth(self, transform: &TransformSyntax<'_>) -> u8 {
        match self {
            Self::Inter => transform.inter_depth,
            Self::Intra { split } => transform.intra_depth + u8::from(split),
        }
    }

    /// Whether `split_transform_flag` is inferred 1 at depth 0
    /// (`IntraSplitFlag`).
    fn forced_first_split(self) -> bool {
        matches!(self, Self::Intra { split: true })
    }
}

/// The smallest transform log2 a CU of `cu_log2` may split to with
/// `max_depth` levels (`MaxTrafoDepth`), clamped to the SPS sizes.
#[must_use]
pub fn min_transform_log2(cu_log2: u8, max_depth: u8, max_log2: u8) -> u8 {
    cu_log2
        .saturating_sub(max_depth)
        .clamp(LOG2_MIN_TB as u8, max_log2)
}

/// `prev_intra_luma_pred_flag` of every prediction unit, then each unit's
/// `mpm_idx` or `rem_intra_luma_pred_mode` (§7.3.8.5).
fn write_intra_luma_modes(w: &mut CabacWriter, ctx: &mut Contexts, codes: &[ModeCode]) {
    for code in codes {
        w.encode_bin(
            &mut ctx.prev_intra_luma_pred_flag,
            matches!(code, ModeCode::Mpm(_)),
        );
    }
    for code in codes {
        match *code {
            // mpm_idx: truncated rice, cMax 2, bypass bins.
            ModeCode::Mpm(idx) => match idx {
                0 => w.encode_bypass(0, 1),
                1 => w.encode_bypass(0b10, 2),
                _ => w.encode_bypass(0b11, 2),
            },
            ModeCode::Rem(rem) => w.encode_bypass(u32::from(rem), REM_INTRA_MODE_BITS),
        }
    }
}

/// `transform_tree()` (§7.3.8.8) with maximum-size implied splits and
/// optional splits.
fn write_transform_tree(
    w: &mut CabacWriter,
    ctx: &mut Contexts,
    tables: &Tables<'_>,
    block: &Block,
    tus: &[TransformUnit],
    tree: TreeKind,
    qp: &mut Option<CuQp<'_>>,
) {
    debug_assert!(!tus.is_empty());
    let min_log2 = min_transform_log2(
        block.log2_size(),
        tree.max_depth(&tables.transform),
        tables.transform.max_log2,
    );
    let node = TransformNode {
        block: *block,
        depth: 0,
        parent_chroma: (true, true),
    };
    write_transform_node(w, ctx, tables, node, tus, tree, min_log2, qp);
}

/// A node of the transform tree being written.
#[derive(Debug, Clone, Copy)]
struct TransformNode {
    block: Block,
    /// `trafoDepth`.
    depth: usize,
    /// The parent's `cbf_cb` / `cbf_cr`, which gate this node's flags.
    parent_chroma: (bool, bool),
}

/// Whether `split_transform_flag` is coded at `node` (§7.3.8.8): inside
/// the SPS size range, above the CU's smallest transform, and not forced
/// by `IntraSplitFlag`.
#[must_use]
pub fn split_transform_flag_present(
    node_log2: u8,
    depth: usize,
    min_log2: u8,
    max_log2: u8,
    forced_first_split: bool,
) -> bool {
    node_log2 <= max_log2 && node_log2 > min_log2 && !(forced_first_split && depth == 0)
}

/// Chroma flags aggregate each subtree and gate its children's flags. A
/// 4x4 luma block's chroma belongs to its 8x8 parent: no flags of its own
/// (they are the parent's), and the residual follows the fourth block
/// (§7.3.8.8, §7.3.8.10), which is where the backend records it.
#[allow(clippy::too_many_arguments)]
fn write_transform_node(
    w: &mut CabacWriter,
    ctx: &mut Contexts,
    tables: &Tables<'_>,
    node: TransformNode,
    tus: &[TransformUnit],
    tree: TreeKind,
    min_log2: u8,
    qp: &mut Option<CuQp<'_>>,
) {
    let TransformNode {
        block,
        depth,
        parent_chroma,
    } = node;
    let split = tus.len() != 1 || tus[0].log2 != block.log2_size();
    if split_transform_flag_present(
        block.log2_size(),
        depth,
        min_log2,
        tables.transform.max_log2,
        tree.forced_first_split(),
    ) {
        w.encode_bin(
            &mut ctx.transform_split[usize::from(5 - block.log2_size())],
            split,
        );
    }
    let (cb, cr) = if block.log2_size() > LOG2_MIN_TB as u8 {
        let cb = tus.iter().any(|tu| tu.cb.coded);
        let cr = tus.iter().any(|tu| tu.cr.coded);
        if depth == 0 || parent_chroma.0 {
            w.encode_bin(&mut ctx.cbf_chroma[depth], cb);
        }
        if depth == 0 || parent_chroma.1 {
            w.encode_bin(&mut ctx.cbf_chroma[depth], cr);
        }
        (cb, cr)
    } else {
        parent_chroma
    };
    if split {
        let mut remaining = tus;
        for child in block.quadrants() {
            let count = remaining
                .iter()
                .take_while(|tu| {
                    tu.x >= child.x
                        && tu.x < child.x + child.size()
                        && tu.y >= child.y
                        && tu.y < child.y + child.size()
                })
                .count();
            debug_assert!(count > 0);
            let (leaves, rest) = remaining.split_at(count);
            let child = TransformNode {
                block: child,
                depth: depth + 1,
                parent_chroma: (cb, cr),
            };
            write_transform_node(w, ctx, tables, child, leaves, tree, min_log2, qp);
            remaining = rest;
        }
        debug_assert!(remaining.is_empty());
    } else {
        let tu = &tus[0];
        let intra = matches!(tree, TreeKind::Intra { .. });
        if intra || depth != 0 || cb || cr {
            w.encode_bin(&mut ctx.cbf_luma[usize::from(depth == 0)], tu.luma.coded);
        } else {
            debug_assert!(tu.luma.coded);
        }
        // transform_unit() (§7.3.8.10): the group's delta comes with its
        // first unit that has a coded block (cbfChroma of a 4x4 luma unit
        // is its 8x8 parent's).
        if let Some(CuQp { group, qp }) = qp
            && group.delta_pending()
            && (tu.luma.coded || cb || cr)
        {
            write_cu_qp_delta(w, ctx, group.code_delta(*qp));
        }
        write_tu_residuals(w, ctx, tables, tu);
    }
}

/// `transform_unit()` residuals in Y, Cb, Cr order (§7.3.8.10).
fn write_tu_residuals(
    w: &mut CabacWriter,
    ctx: &mut Contexts,
    tables: &Tables<'_>,
    tu: &TransformUnit,
) {
    for (residual, luma) in [(&tu.luma, true), (&tu.cb, false), (&tu.cr, false)] {
        if residual.coded {
            write_residual(
                w,
                ctx,
                tables.transform.scans.get(residual.log2, residual.scan),
                residual.levels(tables.levels),
                luma,
                SIGN_DATA_HIDING,
            );
        }
    }
}

/// `cu_qp_delta_abs` (prefix TR cMax 5, first bin context 0, the others
/// context 1; EG0 suffix, §9.3.3.10) and `cu_qp_delta_sign_flag` (§7.3.8.14).
fn write_cu_qp_delta(w: &mut CabacWriter, ctx: &mut Contexts, delta: i32) {
    let magnitude = delta.unsigned_abs();
    for bin in 0..magnitude.min(QP_DELTA_PREFIX) {
        w.encode_bin(&mut ctx.cu_qp_delta_abs[usize::from(bin > 0)], true);
    }
    if magnitude < QP_DELTA_PREFIX {
        w.encode_bin(&mut ctx.cu_qp_delta_abs[usize::from(magnitude > 0)], false);
    } else {
        write_exp_golomb(w, magnitude - QP_DELTA_PREFIX, QP_DELTA_EG_ORDER);
    }
    if delta != 0 {
        w.encode_bypass(u32::from(delta < 0), 1);
    }
}

/// `merge_idx` (§9.3.3.x): truncated rice with cMax `MaxNumMergeCand − 1`,
/// first bin context coded, the rest bypass.
fn write_merge_idx(w: &mut CabacWriter, ctx: &mut Contexts, idx: usize) {
    if MAX_MERGE_CANDIDATES < 2 {
        return;
    }
    w.encode_bin(&mut ctx.merge_idx, idx > 0);
    let c_max = MAX_MERGE_CANDIDATES - 1;
    let mut bin = 1;
    while bin < c_max && idx >= bin {
        w.encode_bypass(u32::from(idx > bin), 1);
        bin += 1;
    }
}

/// `mvd_coding()` (§7.3.8.9) of `mv − pred`.
fn write_mvd(w: &mut CabacWriter, ctx: &mut Contexts, mv: Mv, pred: Mv) {
    let d = [
        i32::from(mv.x) - i32::from(pred.x),
        i32::from(mv.y) - i32::from(pred.y),
    ];
    for v in d {
        w.encode_bin(&mut ctx.abs_mvd_greater0, v != 0);
    }
    for v in d {
        if v != 0 {
            w.encode_bin(&mut ctx.abs_mvd_greater1, v.abs() > 1);
        }
    }
    for v in d {
        if v != 0 {
            if v.abs() > 1 {
                write_exp_golomb(w, v.unsigned_abs() - 2, MVD_EG_ORDER);
            }
            w.encode_bypass(u32::from(v < 0), 1);
        }
    }
}

/// k-th order Exp-Golomb bypass bins (§9.3.3.3).
fn write_exp_golomb(w: &mut CabacWriter, mut value: u32, mut k: u32) {
    while value >= (1 << k) {
        w.encode_bypass(1, 1);
        value -= 1 << k;
        k += 1;
    }
    w.encode_bypass(0, 1);
    w.encode_bypass(value, k);
}

#[cfg(test)]
mod transform_storage_tests {
    use super::{INLINE_TUS, MAX_TUS, TransformUnit, TransformUnits, TuStorage};

    #[test]
    fn default_capacity_stays_inline_and_overflow_is_contiguous() {
        let mut units = TransformUnits::default();
        for i in 0..INLINE_TUS {
            units.push(TransformUnit {
                x: i,
                ..TransformUnit::default()
            });
        }
        assert!(matches!(&units.storage, TuStorage::Inline(_)));
        units.push(TransformUnit {
            x: INLINE_TUS,
            ..TransformUnit::default()
        });
        assert!(matches!(&units.storage, TuStorage::Overflow(_)));
        let allocation = units.as_slice().as_ptr();
        for i in INLINE_TUS + 1..MAX_TUS {
            units.push(TransformUnit {
                x: i,
                ..TransformUnit::default()
            });
            assert_eq!(units.as_slice().as_ptr(), allocation);
        }
        assert_eq!(units.as_slice().len(), MAX_TUS);
        assert!(units.as_slice().iter().enumerate().all(|(i, tu)| tu.x == i));
        // Adding TU16 support must not inline sixteen entries into every CU.
        assert!(
            std::mem::size_of::<TransformUnits>()
                <= INLINE_TUS * std::mem::size_of::<TransformUnit>()
                    + 3 * std::mem::size_of::<usize>()
        );
    }

    #[test]
    #[should_panic(expected = "too many transform units")]
    fn overflow_is_bounded() {
        let mut units = TransformUnits::default();
        for _ in 0..=MAX_TUS {
            units.push(TransformUnit::default());
        }
    }
}
