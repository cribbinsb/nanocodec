//! H.266 §8.5.2 merge/AMVP derivation for P-slice CUs (spatial, temporal
//! from `temporal`, history, pairwise average and zero candidates), and the
//! per-8x8 record of coded CUs (with the vectors of their 4x4 sub-blocks
//! and their reference index) that neighbour derivations, split contexts
//! and deblocking read. Affine candidate lists are in `affine_candidates`.
//!
//! A picture has one active reference, or two with the second reference
//! (`refs.second`, `references.rs`): candidates then carry `refIdxL0`
//! (§8.5.2.2), AMVP keeps only neighbours predicting from the target
//! picture (§8.5.2.10, no spatial scaling in H.266), the collocated vector
//! is scaled to the target's POC distance (§8.5.2.12) and zero merge
//! candidates step through the reference indices (§8.5.2.5).
//!
//! Derived from `VVenC` `UnitTools` (BSD-3-Clause-Clear; `THIRD_PARTY.md`).
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
/// Most active references of a P slice this encoder signals
/// (`NumRefIdxActive[0]`): the previous picture and, with the second
/// reference, the last high-quality one.
pub(crate) const MAX_ACTIVE_REFERENCES: usize = 2;

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

/// A list-0 motion vector and the reference it points into
/// (`MvL0`, `RefIdxL0`): two candidates are the same motion only when both
/// agree (§8.5.2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct RefMv {
    pub mv: Mv16,
    pub ref_idx: u8,
}

impl RefMv {
    /// `mv` into reference `ref_idx`.
    pub(crate) const fn new(mv: Mv16, ref_idx: u8) -> Self {
        Self { mv, ref_idx }
    }
}

/// A neighbour's or candidate's motion as one picture's merge and AMVP
/// derivation (§8.5.2) carries it: a bare [`Mv16`] in a picture of one
/// active reference, where every inter block predicts from reference 0, or
/// a [`RefMv`] with more. The derivation is written once over this type and
/// compiled for each: a single-reference picture's carries, stores and
/// compares no reference index.
trait Candidate: Copy + PartialEq {
    /// Whether candidates may point into different references.
    const MULTI: bool;
    /// `motion` as a candidate (its index dropped without `MULTI`).
    fn of(motion: RefMv) -> Self;
    /// The candidate's vector and reference.
    fn motion(self) -> RefMv;
    /// The motion of the coded inter sub-block of `grid` covering luma
    /// sample `(x, y)`.
    fn at(grid: &MotionGrid, x: isize, y: isize) -> Option<Self>;
}

impl Candidate for Mv16 {
    const MULTI: bool = false;
    fn of(motion: RefMv) -> Self {
        motion.mv
    }
    fn motion(self) -> RefMv {
        RefMv::new(self, 0)
    }
    fn at(grid: &MotionGrid, x: isize, y: isize) -> Option<Self> {
        grid.mv_at(x, y)
    }
}

impl Candidate for RefMv {
    const MULTI: bool = true;
    fn of(motion: RefMv) -> Self {
        motion
    }
    fn motion(self) -> RefMv {
        self
    }
    fn at(grid: &MotionGrid, x: isize, y: isize) -> Option<Self> {
        grid.motion_at(x, y)
    }
}

/// The active references of the current picture: their POC distances
/// (current minus reference, `DiffPicOrderCnt`), by reference index, and
/// `NumRefIdxActive[0]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveReferences {
    pub distances: [u32; MAX_ACTIVE_REFERENCES],
    pub count: usize,
}

impl ActiveReferences {
    /// One reference, the previous picture.
    pub(crate) const SINGLE: Self = Self {
        distances: [1; MAX_ACTIVE_REFERENCES],
        count: 1,
    };
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
    /// Each candidate's `refIdxL0` (all 0 with one active reference).
    refs: [u8; MAX_MERGE_CANDIDATES],
    len: u8,
    /// Whether the picture has more than one active reference; without,
    /// every index is 0 and the vectors alone are the motion.
    multi: bool,
}

impl MergeList {
    /// The list's candidates' vectors.
    pub(crate) fn as_slice(&self) -> &[Mv16] {
        &self.mvs[..usize::from(self.len)]
    }

    /// Candidate `index`: its vector and reference.
    pub(crate) fn candidate(&self, index: usize) -> RefMv {
        RefMv::new(self.as_slice()[index], self.refs[index])
    }

    /// The vectors of the candidates into reference `ref_idx`, with their
    /// `merge_idx`.
    pub(crate) fn of_reference(&self, ref_idx: u8) -> impl Iterator<Item = (usize, Mv16)> + '_ {
        self.as_slice()
            .iter()
            .zip(&self.refs)
            .enumerate()
            .filter(move |&(_, (_, &candidate_ref))| candidate_ref == ref_idx)
            .map(|(index, (&mv, _))| (index, mv))
    }

    /// The `merge_idx` of the first candidate equal to `mv` into reference
    /// `ref_idx`. With one active reference the vectors alone are the
    /// motion.
    #[expect(
        clippy::inline_always,
        reason = "every rate estimate of a vector asks it: outlined it cost 0.17% of a VVC Default encode (x86 callgrind)"
    )]
    #[inline(always)]
    pub(crate) fn position(&self, mv: Mv16, ref_idx: u8) -> Option<usize> {
        if self.multi {
            (0..usize::from(self.len))
                .find(|&index| self.mvs[index] == mv && self.refs[index] == ref_idx)
        } else {
            debug_assert_eq!(ref_idx, 0);
            self.as_slice()
                .iter()
                .position(|&candidate| candidate == mv)
        }
    }

    /// The vectors of the candidates into reference `ref_idx`.
    pub(crate) fn vectors_into(&self, ref_idx: u8) -> impl Iterator<Item = Mv16> + '_ {
        self.as_slice()
            .iter()
            .zip(&self.refs)
            .filter_map(move |(&mv, &candidate_ref)| (candidate_ref == ref_idx).then_some(mv))
    }

    /// The first candidate's vector, if it is into reference `ref_idx`.
    pub(crate) fn first_into(&self, ref_idx: u8) -> Option<Mv16> {
        // A list has at least one candidate (`MaxNumMergeCand` >= 1).
        (self.refs[0] == ref_idx).then_some(self.mvs[0])
    }
}

impl std::ops::Index<usize> for MergeList {
    type Output = Mv16;
    fn index(&self, index: usize) -> &Mv16 {
        &self.as_slice()[index]
    }
}

/// A CU's AMVP predictors into one reference at each [`Amvr`] precision.
pub(crate) type AmvpLists = [[Mv16; 2]; Amvr::ALL.len()];

#[derive(Debug, Clone, Copy)]
pub(crate) struct Predictors {
    /// The regular merge list (§8.5.2.2), `merge_idx` order.
    pub merge: MergeList,
    /// The AMVP predictors into reference `amvp_ref`.
    pub amvp: AmvpLists,
    /// The reference index `amvp` predicts into.
    pub amvp_ref: u8,
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
    /// unless the CU is affine); `None` for intra. Its reference index is
    /// in `MotionGrid::ref_indices`.
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

/// A CU's grid entries, and the picture-wide state coding a CU changes
/// (the history list, the affine models), noted before a whole-CU coding
/// trial (`backend/cu_trial.rs`) to be put back when it is undone.
#[derive(Debug, Default)]
pub(crate) struct CuSnapshot {
    cells: Vec<Option<Cell>>,
    ref_indices: Vec<u8>,
    affine_models: usize,
    history: Vec<RefMv>,
}

#[derive(Debug)]
pub(crate) struct MotionGrid {
    ctu_log2: u8,
    cols: usize,
    cells: Vec<Option<Cell>>,
    /// Each cell's reference index (`RefIdxL0`), written only in pictures
    /// of more than one active reference (empty until the first): a
    /// single-reference picture's CUs all predict from reference 0, so its
    /// data path carries no index.
    ref_indices: Vec<u8>,
    /// The models of this picture's affine CUs (`Cell::affine`).
    affine_models: Vec<crate::affine::Model>,
    // Normative five-entry history, oldest first. AMVP examines up to four.
    history: Vec<RefMv>,
    /// `MaxNumMergeCand`.
    merge_candidates: usize,
    /// One stored motion field per shared reference slot. Flat streams keep one.
    collocated: Vec<MotionField>,
    collocated_slot: usize,
    /// The current picture's active references.
    references: ActiveReferences,
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
            ref_indices: Vec::new(),
            affine_models: Vec::new(),
            history: Vec::with_capacity(5),
            merge_candidates,
            collocated: vec![MotionField::new(width, height)],
            collocated_slot: 0,
            references: ActiveReferences::SINGLE,
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

    /// The current picture's active `references` and the slot of its
    /// collocated picture (`RefPicList[0][0]`: `sh_collocated_ref_idx` 0).
    pub(crate) fn select_references(
        &mut self,
        collocated_slot: usize,
        references: ActiveReferences,
    ) {
        self.collocated_slot = collocated_slot;
        self.references = references;
        if references.count > 1 {
            self.ref_indices.resize(self.cells.len(), 0);
        }
    }

    /// Whether the current picture has more than one active reference.
    fn multi_reference(&self) -> bool {
        self.references.count > 1
    }

    /// Keep the coded picture's motion as a later picture's collocated
    /// motion in `slot`: its vectors and its references' distances, and
    /// each vector's reference index only when it had more than one.
    pub(crate) fn store_motion_field(&mut self, slot: usize) {
        let multi = self.multi_reference();
        let field = &mut self.collocated[slot];
        for (entry, cell) in field
            .vectors_to_store(&self.references)
            .iter_mut()
            .zip(&self.cells)
        {
            *entry = cell
                .and_then(|c| c.mv)
                .map(crate::temporal::at_storage_precision);
        }
        if multi {
            field.store_ref_indices(self.ref_indices.iter().copied());
        }
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
        Some(self.sub_block_vector(cell, mv, x, y))
    }

    /// The motion of the coded inter sub-block covering luma sample
    /// `(x, y)` in a picture of more than one active reference: its vector
    /// and reference.
    pub(crate) fn motion_at(&self, x: isize, y: isize) -> Option<RefMv> {
        debug_assert!(self.multi_reference());
        let mv = self.mv_at(x, y)?;
        // `mv_at` found a coded cell, so the position is in the grid.
        let index = (y as usize / GRID_SIZE) * self.cols + x as usize / GRID_SIZE;
        Some(RefMv::new(mv, self.ref_indices[index]))
    }

    /// The vector of inter `cell`'s sub-block covering luma sample
    /// `(x, y)`, given the cell's own (top-left sub-block) vector `mv`.
    #[inline]
    fn sub_block_vector(&self, cell: &Cell, mv: Mv16, x: isize, y: isize) -> Mv16 {
        let Some(index) = cell.affine else {
            return mv;
        };
        // An affine CU's sub-block vectors follow from its model (as they
        // were derived when it was recorded).
        let cu = cell.cu.block();
        let field = self.affine_models[index as usize].field(cu.width(), cu.height());
        let sub = |c: isize, origin: usize| (c as usize - origin) / SUB_BLOCK;
        field.at(sub(x, cu.x), sub(y, cu.y))
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
    pub(crate) fn segment_motion(&self, at: (usize, usize), horizontal: bool) -> [bool; 2] {
        if self.multi_reference() {
            self.segment_motion_for::<RefMv>(at, horizontal)
        } else {
            self.segment_motion_for::<Mv16>(at, horizontal)
        }
    }

    /// [`Self::segment_motion`], the sides' motion carried as `C`:
    /// different reference pictures give strength 1 too (with one active
    /// reference every inter block shares it).
    fn segment_motion_for<C: Candidate>(
        &self,
        (x, y): (usize, usize),
        horizontal: bool,
    ) -> [bool; 2] {
        std::array::from_fn(|segment| {
            let (x, y) = (x as isize, y as isize);
            let along = (SUB_BLOCK * segment) as isize;
            let (p, q) = if horizontal {
                ((x + along, y - 1), (x + along, y))
            } else {
                ((x - 1, y + along), (x, y + along))
            };
            match (C::at(self, p.0, p.1), C::at(self, q.0, q.1)) {
                (Some(p), Some(q)) => {
                    let (p, q) = (p.motion(), q.motion());
                    p.ref_idx != q.ref_idx || p.mv.far_from(q.mv)
                }
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
    /// The predictors of `block`, its AMVP lists into reference `ref_idx`.
    pub(crate) fn predictors(&self, block: &Block, ref_idx: u8) -> Predictors {
        // Every picture without the second reference has one active
        // reference: its derivation, compiled apart, compares no indices.
        if self.multi_reference() {
            self.predictors_for::<RefMv>(block, ref_idx)
        } else {
            self.predictors_for::<Mv16>(block, ref_idx)
        }
    }

    /// [`Self::predictors`], the neighbours' motion carried as `C`.
    fn predictors_for<C: Candidate>(&self, block: &Block, ref_idx: u8) -> Predictors {
        let ref_idx = if C::MULTI { ref_idx } else { 0 };
        let (x, y) = (block.x as isize, block.y as isize);
        let (w, h) = (block.width() as isize, block.height() as isize);
        // §8.5.2.3: A1 (left), B1 (above), B0 (above right), A0 (below
        // left), B2 (above left).
        let left = C::at(self, x - 1, y + h - 1);
        let above = C::at(self, x + w - 1, y - 1);
        let right = C::at(self, x + w, y - 1);
        let below = C::at(self, x - 1, y + h);
        let corner = C::at(self, x - 1, y - 1);
        let collocated = self.collocated_motion(block);
        // The collocated vector into reference 0 serves merge and AMVP.
        let temporal = collocated.and_then(|c| self.scale_temporal(c, 0));
        let merge = self.merge_list([above, left, right, below, corner], temporal);
        // §8.5.2.10: the first neighbour of each group into the target
        // picture.
        let spatial = [
            first_into(&[below, left], ref_idx),
            first_into(&[right, above, corner], ref_idx),
        ];
        let temporal = if ref_idx == 0 {
            temporal
        } else {
            collocated.and_then(|c| self.scale_temporal(c, ref_idx))
        };
        let amvp = Amvr::ALL.map(|amvr| self.amvp_list::<C>(spatial, temporal, ref_idx, amvr));
        let left = self.get(x - 1, y);
        let above = self.get(x, y - 1);
        Predictors {
            mpm: self.intra_mpm(block),
            merge,
            amvp,
            amvp_ref: ref_idx,
            skip_ctx: usize::from(left.is_some_and(|c| c.skip))
                + usize::from(above.is_some_and(|c| c.skip)),
            intra_ctx: usize::from(
                left.is_some_and(|c| c.mv.is_none()) || above.is_some_and(|c| c.mv.is_none()),
            ),
            affine_ctx: usize::from(self.is_affine(x - 1, y))
                + usize::from(self.is_affine(x, y - 1)),
        }
    }
    /// The collocated vector of `block` (§8.5.2.11) and the POC distance
    /// it spans, if this picture uses temporal candidates.
    #[inline]
    fn collocated_motion(&self, block: &Block) -> Option<(Mv16, u32)> {
        self.temporal
            .then(|| self.collocated[self.collocated_slot].candidate(block, self.ctu_log2))
            .flatten()
    }

    /// The collocated vector of `block` (§8.5.2.11) scaled to reference
    /// index 0, if this picture uses temporal candidates.
    pub(crate) fn temporal_candidate(&self, block: &Block) -> Option<Mv16> {
        self.collocated_motion(block)
            .and_then(|c| self.scale_temporal(c, 0))
    }

    /// The collocated vector below and right of `block` alone, if this
    /// picture uses temporal candidates (§8.5.5.6's bottom-right control
    /// point).
    pub(crate) fn temporal_corner(&self, block: &Block) -> Option<Mv16> {
        self.temporal
            .then(|| self.collocated[self.collocated_slot].bottom_right(block, self.ctu_log2))
            .flatten()
            .and_then(|c| self.scale_temporal(c, 0))
    }

    /// Collocated vector `mv`, spanning POC distance `source`, scaled to
    /// reference `ref_idx`'s distance (§8.5.2.12).
    fn scale_temporal(&self, (mv, source): (Mv16, u32), ref_idx: u8) -> Option<Mv16> {
        let target = self.references.distances[usize::from(ref_idx)];
        crate::temporal::scale(mv, target as i32, source as i32).ok()
    }

    /// §8.5.2.8-8.5.2.9: the AMVP list into reference `ref_idx` from the
    /// left (A) and above (B) spatial predictors at precision `amvr`: both
    /// rounded to it, B dropped when equal to A, then the collocated vector
    /// `temporal` (rounded), then history into the same reference (rounded,
    /// not deduplicated), then zero.
    #[inline]
    fn amvp_list<C: Candidate>(
        &self,
        [a, b]: [Option<Mv16>; 2],
        temporal: Option<Mv16>,
        ref_idx: u8,
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
        for v in self.history.iter().take(4) {
            if candidates.full() {
                break;
            }
            if !C::MULTI || v.ref_idx == ref_idx {
                candidates.push(amvr.round(v.mv));
            }
        }
        candidates.padded()
    }

    /// §8.5.2.2-8.5.2.6: the regular merge list from the spatial neighbours'
    /// motion `[B1, A1, B0, A0, B2]` (None: unavailable or intra), then the
    /// collocated vector `temporal` (into reference 0), history, the
    /// pairwise average of the first two (into the first one's reference)
    /// and zero vectors into each active reference in turn.
    fn merge_list<C: Candidate>(
        &self,
        [b1, a1, b0, a0, b2]: [Option<C>; 5],
        temporal: Option<Mv16>,
    ) -> MergeList {
        let size = self.merge_candidates;
        let differs =
            |candidate: Option<C>, other: Option<C>| candidate.filter(|&v| other != Some(v));
        let spatial = [b1, differs(a1, b1), differs(b0, b1), differs(a0, a1)];
        let b2 = (spatial.iter().flatten().count() < SPATIAL_BEFORE_B2)
            .then(|| differs(differs(b2, a1), b1))
            .flatten();
        let temporal = temporal.map(|mv| C::of(RefMv::new(mv, 0)));
        let mut list = MergeList {
            mvs: [Mv16::ZERO; MAX_MERGE_CANDIDATES],
            refs: [0; MAX_MERGE_CANDIDATES],
            len: size as u8,
            multi: C::MULTI,
        };
        let mut len = 0;
        // With one active reference the indices stay 0.
        let push = |list: &mut MergeList, len: &mut usize, v: C| {
            let motion = v.motion();
            list.mvs[*len] = motion.mv;
            if C::MULTI {
                list.refs[*len] = motion.ref_idx;
            }
            *len += 1;
        };
        for v in spatial.into_iter().chain([b2, temporal]).flatten() {
            if len == size {
                return list;
            }
            push(&mut list, &mut len, v);
        }
        let history = self.history.iter().map(|&v| C::of(v));
        for (age, v) in history.rev().enumerate() {
            if len + 1 >= size {
                break;
            }
            if age >= HISTORY_PRUNED_ENTRIES || (a1 != Some(v) && b1 != Some(v)) {
                push(&mut list, &mut len, v);
            }
        }
        if len >= 2 && len < size {
            let average = pairwise_average(list.mvs[0], list.mvs[1]);
            let first = list.refs[0];
            push(&mut list, &mut len, C::of(RefMv::new(average, first)));
        }
        // §8.5.2.5: the zero candidates' `zeroIdx` steps through the active
        // references, then stays at 0 (their vectors and, with one active
        // reference, their indices are already zero).
        if C::MULTI {
            let references = self.references.count;
            for (zero, index) in (len..size).enumerate() {
                let ref_idx = if zero < references { zero } else { 0 };
                list.refs[index] = ref_idx as u8;
            }
        }
        list
    }

    /// Record coded CU `block` (`motion` `None` for intra) and its
    /// transform units `units`.
    pub(crate) fn record(
        &mut self,
        block: &Block,
        motion: Option<&Motion>,
        ref_idx: u8,
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
            let cell = |mv: Option<Mv16>| {
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
        if self.multi_reference() {
            self.record_ref_idx(block, ref_idx);
        }
        // §8.5.2.16: affine CUs leave the history unchanged.
        if let Some(&Motion::Vector(mv)) = motion {
            let mv = RefMv::new(mv, ref_idx);
            self.history.retain(|&old| old != mv);
            if self.history.len() == 5 {
                self.history.remove(0);
            }
            self.history.push(mv);
        }
    }

    /// The index ranges of `block`'s cells, row by row.
    fn cell_rows(&self, block: &Block) -> impl Iterator<Item = std::ops::Range<usize>> + use<> {
        let (x0, x1) = (block.x / GRID_SIZE, (block.x + block.width()) / GRID_SIZE);
        let cols = self.cols;
        (block.y / GRID_SIZE..(block.y + block.height()) / GRID_SIZE)
            .map(move |y| y * cols + x0..y * cols + x1)
    }

    /// Note in `snapshot` what coding CU `block` changes here.
    pub(crate) fn save_cu(&self, block: &Block, snapshot: &mut CuSnapshot) {
        snapshot.cells.clear();
        snapshot.ref_indices.clear();
        for row in self.cell_rows(block) {
            snapshot.cells.extend_from_slice(&self.cells[row.clone()]);
            if !self.ref_indices.is_empty() {
                snapshot
                    .ref_indices
                    .extend_from_slice(&self.ref_indices[row]);
            }
        }
        snapshot.affine_models = self.affine_models.len();
        snapshot.history.clone_from(&self.history);
    }

    /// Put back what [`Self::save_cu`] noted of CU `block`.
    pub(crate) fn restore_cu(&mut self, block: &Block, snapshot: &CuSnapshot) {
        let mut cells = snapshot.cells.chunks_exact(block.width() / GRID_SIZE);
        let mut ref_indices = snapshot.ref_indices.chunks_exact(block.width() / GRID_SIZE);
        for row in self.cell_rows(block) {
            self.cells[row.clone()].copy_from_slice(cells.next().expect("noted"));
            if let Some(saved) = ref_indices.next() {
                self.ref_indices[row].copy_from_slice(saved);
            }
        }
        self.affine_models.truncate(snapshot.affine_models);
        self.history.clone_from(&snapshot.history);
    }

    /// Set the reference index of CU `block`'s cells (pictures of more
    /// than one active reference).
    fn record_ref_idx(&mut self, block: &Block, ref_idx: u8) {
        let (x0, x1) = (block.x / GRID_SIZE, (block.x + block.width()) / GRID_SIZE);
        for y in block.y / GRID_SIZE..(block.y + block.height()) / GRID_SIZE {
            self.ref_indices[y * self.cols + x0..y * self.cols + x1].fill(ref_idx);
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

/// The vector of the first available neighbour in `group` that predicts
/// from reference `ref_idx` (§8.5.2.10's spatial AMVP candidates; with one
/// active reference, every neighbour does).
#[inline]
fn first_into<C: Candidate>(group: &[Option<C>], ref_idx: u8) -> Option<Mv16> {
    group
        .iter()
        .flatten()
        .map(|c| c.motion())
        .find(|motion| !C::MULTI || motion.ref_idx == ref_idx)
        .map(|motion| motion.mv)
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
        grid.record(&cu, Some(&Motion::Affine(model)), 0, false, 0, &units);
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
