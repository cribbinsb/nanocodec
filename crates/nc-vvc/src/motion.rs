//! H.266 §8.5.2 merge/AMVP derivation for one-reference CUs (spatial,
//! temporal from `temporal`, history, pairwise average and zero
//! candidates), and the per-8x8 record of coded CUs (with the vectors of
//! their 4x4 sub-blocks) that neighbour derivations, split contexts and
//! deblocking read. Affine candidate lists are in `affine_candidates`.
//!
//! Derived from `VVenC` `UnitTools` (BSD-3-Clause-Clear; `THIRD_PARTY.md`).
//! No multiple reference indices.
use crate::temporal::MotionField;
use nc_encoder::Block;
pub(crate) const GRID_SIZE: usize = 8;
/// Motion is stored per 4x4 sub-block (§8.5.5.9: the affine sub-block).
pub(crate) const SUB_BLOCK: usize = 4;
/// Sub-blocks per grid cell side.
const SUBS_PER_CELL: usize = GRID_SIZE / SUB_BLOCK;
use nc_encoder::Mv;

/// The largest `MaxNumMergeCand` (§7.4.3.4: `sps_six_minus_max_num_merge_cand`
/// codes 6 minus it).
pub(crate) const MAX_MERGE_CANDIDATES: usize = 6;
/// §8.5.2.4: lists of this many candidates or more may end in a pairwise
/// average (an eighth-sample vector where the two components' sum is odd).
pub(crate) const PAIRWISE_LIST_SIZE: usize = 3;
/// §8.5.2.6: history candidates fill the list up to one short of its size;
/// they are compared with A1 and B1 only for the two most recent entries.
const HISTORY_PRUNED_ENTRIES: usize = 2;
/// §8.5.2.3: B2 is considered only while fewer spatial candidates than this
/// have entered the list.
const SPATIAL_BEFORE_B2: usize = 4;
// The collocated field has the grid's cell granularity: a cell's first
// sub-block vector is the field's entry (`MotionGrid::commit_motion_field`).
const _: () = assert!(GRID_SIZE == 1 << crate::temporal::FIELD_LOG2);
/// §8.8.3.5: a vector difference of half a luma sample (in sixteenths) or
/// more across an edge gives it boundary strength 1.
const DEBLOCK_MOTION_STEP: i32 = 8;
/// §7.4.? `clipToStorageBitDepth`: vectors are stored in 18 bits.
const MV_STORAGE_BITS: u32 = 18;

/// A motion vector in sixteenth luma samples, the precision every derived
/// and stored vector has (§8.5.2: `MvL0` after `AmvrShift`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Mv16 {
    pub x: i32,
    pub y: i32,
}

impl Mv16 {
    pub(crate) const ZERO: Self = Self { x: 0, y: 0 };

    /// The engine's quarter-sample vector `mv`.
    pub(crate) fn from_quarter(mv: Mv) -> Self {
        Self {
            x: i32::from(mv.x) << 2,
            y: i32::from(mv.y) << 2,
        }
    }

    /// This vector in quarter samples, if it lies on that grid.
    pub(crate) fn quarter(self) -> Option<Mv> {
        if self.x & 3 != 0 || self.y & 3 != 0 {
            return None;
        }
        Some(Mv {
            x: i16::try_from(self.x >> 2).ok()?,
            y: i16::try_from(self.y >> 2).ok()?,
        })
    }

    /// This vector rounded to a multiple of `1 << shift` (§8.5.2.14, ties
    /// towards zero).
    pub(crate) fn round(self, shift: u32) -> Self {
        if shift == 0 {
            return self;
        }
        let round = |c: i32| ((c + (1 << (shift - 1)) - i32::from(c >= 0)) >> shift) << shift;
        Self {
            x: round(self.x),
            y: round(self.y),
        }
    }

    /// This vector clipped to the stored range (`clipToStorageBitDepth`).
    pub(crate) fn clip(self) -> Self {
        let limit = 1 << (MV_STORAGE_BITS - 1);
        Self {
            x: self.x.clamp(-limit, limit - 1),
            y: self.y.clamp(-limit, limit - 1),
        }
    }

    /// Whether this vector differs from `other` by half a luma sample or
    /// more in either component (deblocking strength, §8.8.3.5).
    fn far_from(self, other: Self) -> bool {
        (self.x - other.x).abs() >= DEBLOCK_MOTION_STEP
            || (self.y - other.y).abs() >= DEBLOCK_MOTION_STEP
    }
}

/// A coded inter CU's motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Motion {
    /// One vector for the whole CU.
    Vector(Mv16),
    /// An affine model, one vector per 4x4 sub-block.
    Affine(crate::affine::Model),
}

/// The neighbour-dependent contexts and intra MPMs of a CU's mode syntax
/// (what the slice writer keeps of [`Predictors`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct CuContexts {
    pub skip_ctx: usize,
    pub intra_ctx: usize,
    pub affine_ctx: usize,
    pub mpm: [u8; 6],
}

/// A regular merge list (§8.5.2.2) of `MaxNumMergeCand` entries, in
/// `merge_idx` order.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MergeList {
    mvs: [Mv16; MAX_MERGE_CANDIDATES],
    len: u8,
}

impl MergeList {
    /// The list's candidates.
    pub(crate) fn as_slice(&self) -> &[Mv16] {
        &self.mvs[..usize::from(self.len)]
    }

    /// The `merge_idx` of the first candidate equal to `mv`.
    pub(crate) fn position(&self, mv: Mv16) -> Option<usize> {
        self.as_slice()
            .iter()
            .position(|&candidate| candidate == mv)
    }
}

impl std::ops::Index<usize> for MergeList {
    type Output = Mv16;
    fn index(&self, index: usize) -> &Mv16 {
        &self.as_slice()[index]
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Predictors {
    /// The regular merge list (§8.5.2.2), `merge_idx` order.
    pub merge: MergeList,
    /// The AMVP predictors at each [`Amvr`] precision.
    pub amvp: [[Mv16; 2]; Amvr::ALL.len()],
    pub skip_ctx: usize,
    pub intra_ctx: usize,
    /// `merge_subblock_flag` and `inter_affine_flag` context: affine
    /// left and above CUs (§9.3.4.2.2).
    pub affine_ctx: usize,
    pub mpm: [u8; 6],
}
/// The adaptive motion vector resolution of an AMVP CU's predictors and
/// difference (§7.4.11.5 `AmvrShift`; the half-sample precision, which
/// switches the interpolation filter, is not used).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Amvr {
    /// Quarter samples: `amvr_flag` 0 (or not coded with a zero MVD).
    #[default]
    Quarter,
    /// Whole samples: `amvr_precision_idx` 1.
    Integer,
    /// Four samples: `amvr_precision_idx` 2.
    FourSamples,
}

impl Predictors {
    pub(crate) fn contexts(&self) -> CuContexts {
        CuContexts {
            skip_ctx: self.skip_ctx,
            intra_ctx: self.intra_ctx,
            affine_ctx: self.affine_ctx,
            mpm: self.mpm,
        }
    }
}

impl Amvr {
    pub(crate) const ALL: [Self; 3] = [Self::Quarter, Self::Integer, Self::FourSamples];

    /// `AmvrShift`: log2 of the precision in sixteenth samples.
    pub(crate) fn shift(self) -> u32 {
        match self {
            Self::Quarter => 2,
            Self::Integer => 4,
            Self::FourSamples => 6,
        }
    }

    /// `v` rounded to the precision (§8.5.2.14, ties towards zero).
    fn round(self, v: Mv16) -> Mv16 {
        v.round(self.shift())
    }

    /// Whether `v` lies on the precision's grid.
    pub(crate) fn holds(self, v: Mv16) -> bool {
        let mask = (1 << self.shift()) - 1;
        v.x & mask == 0 && v.y & mask == 0
    }
}

/// A two-entry candidate list (merge and AMVP both hold two), filled in
/// order and padded with zero vectors, without allocating.
#[derive(Debug, Default, Clone, Copy)]
struct Candidates {
    mvs: [Mv16; 2],
    len: usize,
}
impl Candidates {
    fn push(&mut self, mv: Mv16) {
        self.mvs[self.len] = mv;
        self.len += 1;
    }
    fn full(&self) -> bool {
        self.len == self.mvs.len()
    }
    /// The list, its unfilled entries the zero vector.
    fn padded(self) -> [Mv16; 2] {
        self.mvs
    }
}

/// A block as a cell stores it: every coded CU writes a cell per 8x8, so
/// cells are kept small (u16 positions: pictures are narrower than 2^16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CellBlock {
    x: u16,
    y: u16,
    log2_width: u8,
    log2_height: u8,
}

impl CellBlock {
    fn of(block: &Block) -> Self {
        Self {
            x: block.x as u16,
            y: block.y as u16,
            log2_width: block.log2_width,
            log2_height: block.log2_height,
        }
    }

    fn block(self) -> Block {
        Block {
            x: usize::from(self.x),
            y: usize::from(self.y),
            log2_width: self.log2_width,
            log2_height: self.log2_height,
        }
    }
}

/// One 8x8 cell of a coded CU, kept small (40 bytes): the grid is cleared
/// for every picture and rewritten for every CU, and every neighbour lookup
/// reads it.
#[derive(Debug, Clone, Copy)]
struct Cell {
    /// The vector of the cell's top-left 4x4 sub-block (every sub-block's,
    /// unless the CU is affine); `None` for intra.
    mv: Option<Mv16>,
    /// The CU's model, if it is an affine CU: its index in
    /// `MotionGrid::affine_models`, from which its other sub-block vectors
    /// are derived.
    affine: Option<u32>,
    skip: bool,
    /// The CU (split contexts; its position is unused).
    cu: CellBlock,
    /// The transform unit holding this cell (the deblocking walk's edges,
    /// §8.8.3: `transform_tree`); a CU whose intra sub-partitions are
    /// narrower than a cell is one unit here.
    tu: CellBlock,
    /// log2 of the width and height of the luma transform blocks in this
    /// cell (deblocking filter lengths): `tu`'s, or a sub-partition's.
    transform_log2: [u8; 2],
    /// The CU's intra sub-partitions: its chroma is one transform block
    /// (`cu`), and the edges between them are luma only.
    isp: Option<crate::isp::Split>,
    intra_mode: u8,
    /// Coded coefficients of this cell's transform unit, by plane.
    nonzero: [bool; 3],
}

#[derive(Debug)]
pub(crate) struct MotionGrid {
    ctu_log2: u8,
    cols: usize,
    cells: Vec<Option<Cell>>,
    /// The models of this picture's affine CUs (`Cell::affine`).
    affine_models: Vec<crate::affine::Model>,
    // Normative five-entry history, oldest first. AMVP examines up to four.
    history: Vec<Mv16>,
    /// `MaxNumMergeCand`.
    merge_candidates: usize,
    /// One stored motion field per shared reference slot. Flat streams keep one.
    collocated: Vec<MotionField>,
    collocated_slot: usize,
    temporal_distances: [u32; 2],
    /// Whether this picture uses temporal candidates
    /// (`ph_temporal_mvp_enabled_flag`).
    temporal: bool,
}
impl MotionGrid {
    /// A grid for a `width` x `height` coded picture of `1 << ctu_log2`
    /// CTUs, with merge lists of `merge_candidates` (`MaxNumMergeCand`).
    pub(crate) fn new(width: usize, height: usize, ctu_log2: u8, merge_candidates: usize) -> Self {
        debug_assert!((1..=MAX_MERGE_CANDIDATES).contains(&merge_candidates));
        let cols = width.div_ceil(GRID_SIZE);
        Self {
            ctu_log2,
            cols,
            cells: vec![None; cols * height.div_ceil(GRID_SIZE)],
            affine_models: Vec::new(),
            history: Vec::with_capacity(5),
            merge_candidates,
            collocated: vec![MotionField::new(width, height)],
            collocated_slot: 0,
            temporal_distances: [1, 1],
            temporal: false,
        }
    }
    pub(crate) fn ctu_size(&self) -> usize {
        1 << self.ctu_log2
    }
    /// Start a picture, with (`temporal`) or without temporal candidates
    /// from the last committed picture.
    pub(crate) fn reset(&mut self, temporal: bool) {
        self.cells.fill(None);
        self.affine_models.clear();
        self.history.clear();
        self.temporal = temporal;
    }

    /// Keep the coded picture's motion as the next picture's collocated
    /// motion (§8.5.2.11), once the picture is final.
    pub(crate) fn commit_motion_field(&mut self) {
        self.store_motion_field(0);
    }

    /// Allocate only the reference slots the shared temporal structure needs.
    pub(crate) fn with_slots(mut self, slots: usize) -> Self {
        self.collocated.resize(slots, self.collocated[0].clone());
        self
    }

    pub(crate) fn select_collocated(&mut self, slot: usize, target: u32, source: u32) {
        self.collocated_slot = slot;
        self.temporal_distances = [target, source];
    }

    pub(crate) fn store_motion_field(&mut self, slot: usize) {
        self.collocated[slot].store(self.cells.iter().map(|cell| cell.and_then(|c| c.mv)));
    }
    pub(crate) fn begin_ctu(&mut self, x: usize) {
        // §8.5.2.16: history is reset at the start of each tile CTU row.
        if x == 0 {
            self.history.clear();
        }
    }
    /// The coded cell covering luma sample `(x, y)`, if any. A reference:
    /// a cell is about 140 bytes, and every neighbour lookup (predictors,
    /// contexts, the deblocking walk) reads only a field or two of it.
    fn get(&self, x: isize, y: isize) -> Option<&Cell> {
        if x < 0 || y < 0 {
            return None;
        }
        let (col, row) = (x / GRID_SIZE as isize, y / GRID_SIZE as isize);
        if col as usize >= self.cols {
            return None;
        }
        self.cells
            .get(row as usize * self.cols + col as usize)
            .and_then(Option::as_ref)
    }
    pub(crate) fn available(&self, x: isize, y: isize) -> bool {
        self.get(x, y).is_some()
    }

    /// The vector of the coded inter sub-block covering luma sample
    /// `(x, y)`; `None` if it is not coded yet or intra.
    pub(crate) fn mv_at(&self, x: isize, y: isize) -> Option<Mv16> {
        let cell = self.get(x, y)?;
        let mv = cell.mv?;
        let Some(index) = cell.affine else {
            return Some(mv);
        };
        // An affine CU's sub-block vectors follow from its model (as they
        // were derived when it was recorded).
        let cu = cell.cu.block();
        let field = self.affine_models[index as usize].field(cu.width(), cu.height());
        let sub = |c: isize, origin: usize| (c as usize - origin) / SUB_BLOCK;
        Some(field.at(sub(x, cu.x), sub(y, cu.y)))
    }

    /// The coded CU covering luma sample `(x, y)` and its affine model, if
    /// it is an affine CU (inherited affine candidates, §8.5.5.5).
    pub(crate) fn affine_at(&self, x: isize, y: isize) -> Option<(Block, crate::affine::Model)> {
        let cell = self.get(x, y)?;
        cell.affine
            .map(|index| (cell.cu.block(), self.affine_models[index as usize]))
    }

    /// Whether the CU covering luma sample `(x, y)` is a coded affine CU.
    fn is_affine(&self, x: isize, y: isize) -> bool {
        self.get(x, y).is_some_and(|c| c.affine.is_some())
    }

    /// For each CTU in raster order, whether every CU coded in it is a skip
    /// CU (merge, no residual: `vvc.sao_skip_unchanged`).
    pub(crate) fn skipped_ctus(&self) -> Vec<bool> {
        self.ctus_where(|c| c.skip)
    }

    /// For each CTU in raster order, whether every CU coded in it is an inter
    /// CU without luma residual (`vvc.alf_skip_unchanged` 2).
    pub(crate) fn luma_predicted_ctus(&self) -> Vec<bool> {
        self.ctus_where(|c| c.mv.is_some() && !c.nonzero[0])
    }

    /// For each CTU in raster order, whether every coded cell in it meets
    /// `test`.
    fn ctus_where(&self, test: impl Fn(&Cell) -> bool) -> Vec<bool> {
        let side = self.ctu_size() / GRID_SIZE;
        let rows = self.cells.len() / self.cols;
        let ctu_cols = self.cols.div_ceil(side);
        (0..rows.div_ceil(side) * ctu_cols)
            .map(|index| {
                let (col0, row0) = ((index % ctu_cols) * side, (index / ctu_cols) * side);
                (row0..(row0 + side).min(rows)).all(|row| {
                    let start = row * self.cols + col0;
                    let end = row * self.cols + (col0 + side).min(self.cols);
                    self.cells[start..end].iter().flatten().all(&test)
                })
            })
            .collect()
    }
    pub(crate) fn intra_mpm(&self, block: &Block) -> [u8; 6] {
        let (x, y) = (block.x as isize, block.y as isize);
        let (w, h) = (block.width() as isize, block.height() as isize);
        let mode = |c: Option<&Cell>| {
            c.filter(|c| c.mv.is_none())
                .map_or(crate::intra::PLANAR, |c| c.intra_mode)
        };
        let left = mode(self.get(x - 1, y + h - 1));
        // §8.4.2: the above candidate is not taken from the CTU row above.
        let above = if block.y.is_multiple_of(self.ctu_size()) {
            crate::intra::PLANAR
        } else {
            mode(self.get(x + w - 1, y - 1))
        };
        crate::intra::mpm(left, above)
    }
    /// The transform unit covering luma sample `(x, y)`, if coded
    /// (deblocking edges are TU edges, §8.8.3).
    pub(crate) fn transform_unit(&self, x: usize, y: usize) -> Option<Block> {
        self.get(x as isize, y as isize).map(|c| c.tu.block())
    }

    /// The intra sub-partitions of the CU at luma sample `(x, y)`, if it
    /// is coded with sub-partitions narrower than a cell (the deblocking
    /// walk filters the edges between them itself).
    pub(crate) fn narrow_sub_partitions(&self, x: usize, y: usize) -> Option<crate::isp::Split> {
        self.get(x as isize, y as isize)
            .filter(|c| c.tu == c.cu)
            .and_then(|c| c.isp)
    }
    /// The coded block edge at `(x, y)` (8 lines along it), if a transform
    /// unit starts there: the transform sizes across it and its strengths.
    pub(crate) fn deblocking_edge(&self, x: usize, y: usize, horizontal: bool) -> Option<Edge> {
        let q = self.get(x as isize, y as isize)?;
        if if horizontal {
            y != usize::from(q.tu.y)
        } else {
            x != usize::from(q.tu.x)
        } {
            return None;
        }
        let before = |at: usize| at as isize - 1;
        let (px, py) = if horizontal {
            (x as isize, before(y))
        } else {
            (before(x), y as isize)
        };
        let p = self.get(px, py)?;
        let (luma, chroma) = if p.mv.is_some() && q.mv.is_some() {
            // A coded luma residual already sets both strengths to one.
            // As in HEVC, only derive motion when it can change the answer.
            let luma = if p.nonzero[0] || q.nonzero[0] {
                [1; 2]
            } else {
                self.segment_motion((x, y), horizontal).map(u8::from)
            };
            (
                luma,
                [1, 2].map(|ch| u8::from(p.nonzero[ch] || q.nonzero[ch])),
            )
        } else if q.isp.is_some() && p.cu == q.cu {
            // Between sub-partitions: no chroma edge (§8.8.3.3).
            ([2; 2], [0; 2])
        } else {
            ([2; 2], [2; 2])
        };
        // The transform blocks across the edge set the filter lengths:
        // luma's, and chroma's (a sub-partitioned CU's chroma is the CU's).
        let across = |(w, h): (usize, usize)| if horizontal { h } else { w };
        let chroma_size = |c: &Cell| {
            let block = if c.isp.is_some() { c.cu } else { c.tu }.block();
            across((block.width(), block.height()))
        };
        Some(Edge {
            sizes: [
                [p, q].map(|c| across(c.transform_log2.map(|l| 1usize << l).into())),
                [chroma_size(p), chroma_size(q)],
            ],
            luma,
            chroma,
            affine: [p.affine.is_some(), q.affine.is_some()],
        })
    }

    /// For each 4-line segment of the 8 lines from `(x, y)` along an edge,
    /// whether the inter sub-blocks on its two sides move half a sample or
    /// more apart (§8.8.3.5).
    pub(crate) fn segment_motion(&self, (x, y): (usize, usize), horizontal: bool) -> [bool; 2] {
        std::array::from_fn(|segment| {
            let (x, y) = (x as isize, y as isize);
            let along = (SUB_BLOCK * segment) as isize;
            let (p, q) = if horizontal {
                ((x + along, y - 1), (x + along, y))
            } else {
                ((x - 1, y + along), (x, y + along))
            };
            match (self.mv_at(p.0, p.1), self.mv_at(q.0, q.1)) {
                (Some(p), Some(q)) => p.far_from(q),
                _ => false,
            }
        })
    }

    /// The affine CU covering luma sample `(x, y)`, if it is one (its
    /// sub-block edges inside are deblocked too, §8.8.3.2).
    pub(crate) fn affine_cu(&self, x: usize, y: usize) -> Option<Block> {
        self.get(x as isize, y as isize)
            .filter(|c| c.affine.is_some())
            .map(|c| c.cu.block())
    }
    /// The neighbour-dependent contexts of coding-tree `node`'s split bins
    /// (§9.3.4.2.2): its left (x − 1, y) and above (x, y − 1) CUs.
    pub(crate) fn split_contexts(
        &self,
        node: &crate::coding_tree::Node,
        allowed: crate::coding_tree::Allowed,
    ) -> crate::coding_tree::SplitContexts {
        let (x, y) = (node.block.x as isize, node.block.y as isize);
        let neighbour = |cell: Option<&Cell>| {
            cell.map(|c| {
                let cu = c.cu.block();
                crate::coding_tree::NeighbourCu {
                    width: cu.width(),
                    height: cu.height(),
                    qt_depth: self.qt_depth(&cu),
                }
            })
        };
        crate::coding_tree::SplitContexts::new(
            node,
            self.qt_depth(&node.block),
            allowed,
            [neighbour(self.get(x - 1, y)), neighbour(self.get(x, y - 1))],
        )
    }
    /// `CqtDepth` of a CU or node: quadtree nodes are square, and a
    /// multi-type tree leaf's longer side is its quadtree parent's.
    fn qt_depth(&self, block: &Block) -> u8 {
        self.ctu_log2 - block.log2_width.max(block.log2_height)
    }
    pub(crate) fn predictors(&self, block: &Block) -> Predictors {
        let (x, y) = (block.x as isize, block.y as isize);
        let (w, h) = (block.width() as isize, block.height() as isize);
        // §8.5.2.3: A1 (left), B1 (above), B0 (above right), A0 (below
        // left), B2 (above left).
        let left = self.mv_at(x - 1, y + h - 1);
        let above = self.mv_at(x + w - 1, y - 1);
        let right = self.mv_at(x + w, y - 1);
        let below = self.mv_at(x - 1, y + h);
        let corner = self.mv_at(x - 1, y - 1);
        let temporal = self.temporal_candidate(block);
        let merge = self.merge_list([above, left, right, below, corner], temporal);
        let spatial = [below.or(left), right.or(above).or(corner)];
        let amvp = Amvr::ALL.map(|amvr| self.amvp_list(spatial, temporal, amvr));
        let left = self.get(x - 1, y);
        let above = self.get(x, y - 1);
        Predictors {
            mpm: self.intra_mpm(block),
            merge,
            amvp,
            skip_ctx: usize::from(left.is_some_and(|c| c.skip))
                + usize::from(above.is_some_and(|c| c.skip)),
            intra_ctx: usize::from(
                left.is_some_and(|c| c.mv.is_none()) || above.is_some_and(|c| c.mv.is_none()),
            ),
            affine_ctx: usize::from(self.is_affine(x - 1, y))
                + usize::from(self.is_affine(x, y - 1)),
        }
    }
    /// The collocated vector of `block` (§8.5.2.11), if this picture uses
    /// temporal candidates.
    pub(crate) fn temporal_candidate(&self, block: &Block) -> Option<Mv16> {
        self.temporal
            .then(|| self.collocated[self.collocated_slot].candidate(block, self.ctu_log2))
            .flatten()
            .and_then(|mv| self.scale_temporal(mv))
    }

    /// The collocated vector below and right of `block` alone, if this
    /// picture uses temporal candidates (§8.5.5.6's bottom-right control
    /// point).
    pub(crate) fn temporal_corner(&self, block: &Block) -> Option<Mv16> {
        self.temporal
            .then(|| self.collocated[self.collocated_slot].bottom_right(block, self.ctu_log2))
            .flatten()
            .and_then(|mv| self.scale_temporal(mv))
    }

    fn scale_temporal(&self, mv: Mv16) -> Option<Mv16> {
        let [target, source] = self.temporal_distances;
        crate::temporal::scale(mv, target as i32, source as i32).ok()
    }

    /// §8.5.2.8-8.5.2.9: the AMVP list from the left (A) and above (B)
    /// spatial predictors at precision `amvr`: both rounded to it, B dropped
    /// when equal to A, then the collocated vector `temporal` (rounded),
    /// then history (rounded, not deduplicated), then zero.
    fn amvp_list(
        &self,
        [a, b]: [Option<Mv16>; 2],
        temporal: Option<Mv16>,
        amvr: Amvr,
    ) -> [Mv16; 2] {
        let mut candidates = Candidates::default();
        let (a, b) = (a.map(|v| amvr.round(v)), b.map(|v| amvr.round(v)));
        if let Some(v) = a {
            candidates.push(v);
        }
        if let Some(v) = b
            && a != Some(v)
        {
            candidates.push(v);
        }
        if let Some(v) = temporal
            && !candidates.full()
        {
            candidates.push(amvr.round(v));
        }
        // Only already coded spatial cells are available.
        for &v in self.history.iter().take(4) {
            if candidates.full() {
                break;
            }
            candidates.push(amvr.round(v));
        }
        candidates.padded()
    }

    /// §8.5.2.2-8.5.2.6: the regular merge list from the spatial neighbours'
    /// vectors `[B1, A1, B0, A0, B2]` (None: unavailable or intra), then the
    /// collocated vector `temporal`, history, the pairwise average of the
    /// first two and zero vectors. One reference picture: equal vectors are
    /// equal motion.
    fn merge_list(
        &self,
        [b1, a1, b0, a0, b2]: [Option<Mv16>; 5],
        temporal: Option<Mv16>,
    ) -> MergeList {
        let size = self.merge_candidates;
        let differs =
            |candidate: Option<Mv16>, other: Option<Mv16>| candidate.filter(|&v| other != Some(v));
        let spatial = [b1, differs(a1, b1), differs(b0, b1), differs(a0, a1)];
        let b2 = (spatial.iter().flatten().count() < SPATIAL_BEFORE_B2)
            .then(|| differs(differs(b2, a1), b1))
            .flatten();
        // Unfilled entries are the zero candidates (one reference index).
        let mut list = MergeList {
            mvs: [Mv16::ZERO; MAX_MERGE_CANDIDATES],
            len: size as u8,
        };
        let mut len = 0;
        for v in spatial.into_iter().chain([b2, temporal]).flatten() {
            if len == size {
                return list;
            }
            list.mvs[len] = v;
            len += 1;
        }
        for (age, &v) in self.history.iter().rev().enumerate() {
            if len + 1 >= size {
                break;
            }
            if age >= HISTORY_PRUNED_ENTRIES || (a1 != Some(v) && b1 != Some(v)) {
                list.mvs[len] = v;
                len += 1;
            }
        }
        if len >= 2 && len < size {
            list.mvs[len] = pairwise_average(list.mvs[0], list.mvs[1]);
        }
        list
    }

    /// Record coded CU `block` (`motion` `None` for intra) and its
    /// transform units `units`.
    pub(crate) fn record(
        &mut self,
        block: &Block,
        motion: Option<&Motion>,
        skip: bool,
        intra_mode: u8,
        units: &crate::transform_tree::CodedUnits,
    ) {
        let model = match motion {
            Some(Motion::Affine(model)) => Some(*model),
            _ => None,
        };
        let field = model.map(|model| model.field(block.width(), block.height()));
        let affine = model.map(|model| {
            self.affine_models.push(model);
            u32::try_from(self.affine_models.len() - 1).expect("affine CUs fit a u32")
        });
        // The vector of the top-left sub-block of the cell at CU offset
        // (`col`, `row`) in cells.
        let cell_mv = |col: usize, row: usize| {
            motion.map(|m| match (m, &field) {
                (_, Some(field)) => field.at(SUBS_PER_CELL * col, SUBS_PER_CELL * row),
                (Motion::Vector(mv), None) => *mv,
                (Motion::Affine(_), None) => unreachable!("affine CUs have a field"),
            })
        };
        // Sub-partitions narrower than a cell are recorded as their CU.
        let narrow = units
            .as_slice()
            .iter()
            .any(|(tu, _)| tu.width().min(tu.height()) < GRID_SIZE);
        let whole = [(
            *block,
            [units.as_slice().iter().any(|(_, f)| f[0]), false, false],
        )];
        let cells = if narrow { &whole[..] } else { units.as_slice() };
        let transform = units.as_slice()[0].0;
        for &(tu, nonzero) in cells {
            let transform = if narrow { transform } else { tu };
            let cell = |mv| {
                Some(Cell {
                    mv,
                    affine,
                    skip,
                    cu: CellBlock::of(block),
                    tu: CellBlock::of(&tu),
                    transform_log2: [transform.log2_width, transform.log2_height],
                    isp: units.isp(),
                    intra_mode,
                    nonzero,
                })
            };
            // Without an affine field every cell of the unit is the same:
            // one value filled along each row (building one per cell was
            // most of this function).
            let uniform = field.is_none().then(|| cell(cell_mv(0, 0)));
            let (x0, x1) = (tu.x / GRID_SIZE, (tu.x + tu.width()) / GRID_SIZE);
            for y in tu.y / GRID_SIZE..(tu.y + tu.height()) / GRID_SIZE {
                let row = &mut self.cells[y * self.cols + x0..y * self.cols + x1];
                match uniform {
                    Some(uniform) => row.fill(uniform),
                    None => {
                        for (x, target) in (x0..).zip(row) {
                            *target =
                                cell(cell_mv(x - block.x / GRID_SIZE, y - block.y / GRID_SIZE));
                        }
                    }
                }
            }
        }
        // §8.5.2.16: affine CUs leave the history unchanged.
        if let Some(&Motion::Vector(mv)) = motion {
            self.history.retain(|&old| old != mv);
            if self.history.len() == 5 {
                self.history.remove(0);
            }
            self.history.push(mv);
        }
    }
}

/// A transform-unit edge's deblocking inputs (`MotionGrid::deblocking_edge`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Edge {
    /// The luma, then the chroma, transform block sizes across the edge on
    /// its P and Q sides.
    pub sizes: [[usize; 2]; 2],
    /// Luma boundary strength of each 4-line segment.
    pub luma: [u8; 2],
    /// Cb and Cr boundary strength (8 luma lines: one chroma segment).
    pub chroma: [u8; 2],
    /// Whether the P and Q sides are affine CUs (their long filters reach
    /// at most 5 samples, §8.8.3.3).
    pub affine: [bool; 2],
}

/// §8.5.2.4 with §8.5.2.14 rounding (right shift 1, ties towards zero):
/// the average of two candidates' vectors.
fn pairwise_average(a: Mv16, b: Mv16) -> Mv16 {
    let half = |sum: i32| (sum + 1 - i32::from(sum >= 0)) >> 1;
    Mv16 {
        x: half(a.x + b.x),
        y: half(a.y + b.y),
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn cells_stay_small() {
        // Every coded CU writes one per 8x8 (14400 per 720p picture); about
        // 140 bytes with full `Block`s, 84 with every sub-block's vector and
        // the affine model in the cell.
        assert!(std::mem::size_of::<Option<super::Cell>>() <= 40);
    }
    use super::*;

    #[test]
    fn pairwise_average_rounds_towards_zero() {
        let mv = |x: i32, y: i32| Mv16 { x, y };
        assert_eq!(pairwise_average(mv(16, -24), mv(8, 8)), mv(12, -8));
        assert_eq!(pairwise_average(mv(4, -4), mv(0, 0)), mv(2, -2));
        assert_eq!(pairwise_average(mv(3, -3), mv(0, 0)), mv(1, -1));
    }

    #[test]
    fn coded_residual_determines_both_deblocking_segments() {
        for horizontal in [false, true] {
            for coded_side in [None, Some(0), Some(1)] {
                for difference in [0, 7, 8, 32] {
                    let mut grid = MotionGrid::new(16, 16, 4, 2);
                    for y in 0..2 {
                        for x in 0..2 {
                            let side = if horizontal { y } else { x };
                            let block = CellBlock {
                                x: (x * 8) as u16,
                                y: (y * 8) as u16,
                                log2_width: 3,
                                log2_height: 3,
                            };
                            // The Q side moves across the edge.
                            let vector = Mv16 {
                                x: if side == 1 { difference } else { 0 },
                                y: 0,
                            };
                            grid.cells[y * 2 + x] = Some(Cell {
                                mv: Some(vector),
                                affine: None,
                                skip: false,
                                cu: block,
                                tu: block,
                                transform_log2: [3; 2],
                                isp: None,
                                intra_mode: 0,
                                nonzero: [coded_side == Some(side), false, true],
                            });
                        }
                    }
                    let (x, y) = if horizontal { (0, 8) } else { (8, 0) };
                    let edge = grid.deblocking_edge(x, y, horizontal).unwrap();
                    assert_eq!(
                        edge.luma,
                        if coded_side.is_some() {
                            [1; 2]
                        } else {
                            [u8::from(difference >= 8); 2]
                        }
                    );
                    assert_eq!(edge.chroma, [0, 1]);
                }
            }
        }
    }

    #[test]
    fn affine_sub_block_vectors_follow_the_recorded_model() {
        let mut grid = MotionGrid::new(32, 32, 5, 2);
        let cu = Block {
            x: 0,
            y: 0,
            log2_width: 4,
            log2_height: 4,
        };
        let model = crate::affine::Model {
            cp: [Mv16 { x: 16, y: -8 }, Mv16 { x: 48, y: 24 }, Mv16::ZERO],
            six: false,
        };
        let units = crate::transform_tree::coded_units(&cu, None, &Default::default(), 5);
        grid.record(&cu, Some(&Motion::Affine(model)), false, 0, &units);
        let field = model.field(16, 16);
        for row in 0..4 {
            for col in 0..4 {
                let at = ((SUB_BLOCK * col) as isize, (SUB_BLOCK * row) as isize);
                assert_eq!(grid.mv_at(at.0, at.1), Some(field.at(col, row)));
            }
        }
        assert_eq!(grid.affine_at(4, 12), Some((cu, model)));
    }

    #[test]
    fn amvr_rounding_ties_towards_zero() {
        // Quarter-sample vectors, in sixteenths.
        let mv = |x: i32, y: i32| Mv16 { x: 4 * x, y: 4 * y };
        assert_eq!(Amvr::Integer.round(mv(2, -2)), mv(0, 0));
        assert_eq!(Amvr::Integer.round(mv(3, -3)), mv(4, -4));
        assert_eq!(Amvr::Integer.round(mv(6, -6)), mv(4, -4));
        assert_eq!(Amvr::FourSamples.round(mv(8, -9)), mv(0, -16));
        assert_eq!(Amvr::Quarter.round(mv(5, -7)), mv(5, -7));
        assert!(Amvr::Integer.holds(mv(-8, 12)) && !Amvr::Integer.holds(mv(-8, 13)));
    }
}
