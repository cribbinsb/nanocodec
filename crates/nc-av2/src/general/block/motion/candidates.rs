//! A block's candidate vectors for a reference pair, and its candidate
//! warp models (dav2d `refmvs.c` `dav2d_refmvs_find` with
//! `add_spatial_candidate`, `add_candidate_sngl`, `add_candidate_comp`,
//! `add_candidate_c2s`, `add_derived`, `add_temporal_candidate`,
//! `add_tmvp`, `model_from_corners`, `get_warpmv_proj`; dav2d commit
//! 237229e, BSD-2-Clause, see `docs/licenses/dav2d`).
//!
//! The order: high-priority temporal candidates, the spatial neighbours
//! (bottom-most left, right-most above, top-most left, left-most above,
//! below-left, above-right), low-priority temporal candidates, above-left,
//! two further left; the heaviest nearest one first; derived candidates
//! (projected, trajectory, cross-reference), the bank, clipping, the
//! global vector, extra combinations of the first three; for warps the
//! corner model, the neighbours', the warp bank's, global and identity
//! models; for block copies the default vectors.
use super::MotionStore;
use super::MotionUnit;
use super::bank::MvBank;
use super::frame::DRL_REORDER_ALWAYS;
use super::vector::{
    MAX_COMPONENT, Mv, RefPair, TIP_REFERENCE, TemporalMv, project_temporal, scale, scale_temporal,
    whole,
};
use crate::general::block::size::BlockSize;
use crate::general::frame_header::filter_mode;
use crate::general::warp::{BlockArea, IDENTITY, UNITY, global_vector};

/// The list holds up to six candidates.
pub const MAX_CANDIDATES: usize = 6;
/// The warp list holds up to four models.
pub const MAX_WARPS: usize = 4;
/// Derived candidates and the compound pairs' single-reference cache hold
/// four.
const MAX_DERIVED: usize = 4;
/// Duplicate checks stop once this many comparisons were made (dav2d
/// `iter_cntr` against `max_iter`): 16 for the list, 2 for the derived
/// and single caches.
const MAX_COMPARISONS: u32 = 16;
const MAX_CACHE_COMPARISONS: u32 = 2;
/// DRL reordering when only the nearest candidates need it (dav2d
/// `enable_drl_reorder` 1) and how many nearest each mode needs.
const DRL_REORDER_CONSTRAINED: u8 = 1;
const REORDER_MIN_ALWAYS: usize = 2;
const REORDER_MIN_CONSTRAINED: usize = 4;
/// Blocks this many units or less on a side read their above-right and
/// below-left neighbours.
const CORNER_MAX_UNITS: u32 = 16;
/// The block copy's default vectors look this far left past a
/// superblock's width, in samples.
pub const DEFAULT_LEFT_EXTRA: i32 = 256;
/// The superblock side the block copy defaults use, in samples, for
/// superblock log2 6.
const BLOCK_COPY_SUPERBLOCK: i32 = 64;
/// Eighth-sample vector units per 4x4 unit, and luma samples per unit.
const VECTOR_UNITS_PER_UNIT: i32 = 32;
const SAMPLES_PER_UNIT: i32 = 4;
/// The equal compound weight (dav2d `cwp_idx` 8).
pub const EQUAL_WEIGHT: i8 = 8;
/// Blocks larger than 8 units on both sides get extra candidates mixing
/// the first three's components.
const EXTRA_MIN_UNITS: u32 = 8;
/// The extra candidates' (row source, column source) pairs.
const EXTRA_PAIRS: [(usize, usize); 6] = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)];
/// Identity warp models close the warp list, at most two.
const DEFAULT_WARPS: usize = 2;
/// Corner-model rounding: entries in Q16, to multiples of 64.
const MODEL_STEP_MASK: i32 = 0x3f;
const MODEL_STEP_HALF: i32 = 0x20;
const MODEL_ENTRY_MAX: i32 = 0x7fc0;
const MODEL_TRANSLATION_MIN: i64 = -0x800_0000;
const MODEL_TRANSLATION_MAX: i64 = 0x7ff_ffc0;
/// A corner vector difference to a Q16 matrix entry: `<< 11` and divided
/// by the side in units (`<< 13` per sample over 4 samples per unit).
const CORNER_SCALE_LOG2: u32 = 11;
const TRANSLATION_LOG2: u32 = 13;
/// TIP blocks of this smallest side (units) and up, without MV
/// refinement, keep 16x16 motion.
const TIP_COARSE_MIN_UNITS: u32 = 4;

/// One candidate (dav2d `refmvs_candidate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    /// The vector, and the second reference's for compound pairs.
    pub mv: [Mv; 2],
    /// How strongly the neighbours suggest it (the reorder key).
    pub weight: u16,
    /// The compound weight index it brings.
    pub cwp: i8,
    /// Where the spatial neighbour it came from lies, in units from the
    /// block's top-left (0, 0 for others).
    pub y_off: i8,
    /// See `y_off`.
    pub x_off: i8,
}

impl Default for Candidate {
    fn default() -> Self {
        Self {
            mv: [Mv::default(); 2],
            weight: 0,
            cwp: EQUAL_WEIGHT,
            y_off: 0,
            x_off: 0,
        }
    }
}

/// A block's candidates (dav2d `mvstack`, `cnt`) and warp models.
#[derive(Debug, Clone, Copy)]
pub struct Candidates {
    /// The candidates in order, defaults beyond `count`.
    pub list: [Candidate; MAX_CANDIDATES],
    /// How many were found.
    pub count: usize,
    /// The warp models (when asked for).
    pub warps: [[i32; 6]; MAX_WARPS],
    /// How many.
    pub warp_count: usize,
}

/// A list being built, with dav2d's count of duplicate comparisons.
#[derive(Debug, Clone, Copy, Default)]
struct List {
    entries: [Candidate; MAX_CANDIDATES],
    count: usize,
    comparisons: u32,
}

impl List {
    /// Swap the heaviest of the first `nearest` (the first of equals) to
    /// the front.
    fn promote_heaviest(&mut self, nearest: usize) {
        let mut heaviest = 0;
        for n in 1..nearest {
            if self.entries[n].weight > self.entries[heaviest].weight {
                heaviest = n;
            }
        }
        self.entries.swap(0, heaviest);
    }

    /// The index of an equal entry, while comparisons remain (each
    /// search costs the positions it looked at).
    fn find(&mut self, limit: u32, equal: impl Fn(&Candidate) -> bool) -> Option<usize> {
        if self.comparisons >= limit {
            return None;
        }
        let found = self.entries[..self.count].iter().position(equal);
        self.comparisons += found.map_or(self.count, |m| m + 1) as u32;
        found
    }

    /// Add a one-reference vector or weigh an equal one (dav2d
    /// `add_candidate_sngl`).
    fn add_single(
        &mut self,
        (limit, comparisons): (usize, u32),
        weight: u16,
        mv: Mv,
        offsets: (i8, i8),
    ) {
        if let Some(m) = self.find(comparisons, |c| c.mv[0] == mv) {
            self.entries[m].weight = self.entries[m].weight.wrapping_add(weight);
            return;
        }
        if self.count >= limit {
            return;
        }
        let entry = &mut self.entries[self.count];
        entry.mv[0] = mv;
        entry.weight = weight;
        (entry.y_off, entry.x_off) = offsets;
        self.count += 1;
    }

    /// Add a vector pair or weigh an equal one (dav2d
    /// `add_candidate_comp`).
    fn add_pair(&mut self, (limit, comparisons): (usize, u32), weight: u16, cwp: i8, mv: [Mv; 2]) {
        if let Some(m) = self.find(comparisons, |c| c.mv == mv) {
            self.entries[m].weight = self.entries[m].weight.wrapping_add(weight);
            return;
        }
        if self.count >= limit {
            return;
        }
        let entry = &mut self.entries[self.count];
        entry.mv = mv;
        entry.weight = weight;
        entry.cwp = cwp;
        self.count += 1;
    }
}

/// The one-reference vectors compound neighbours brought, with their
/// references (dav2d `refmvs_sngl_mv_cache`).
#[derive(Debug, Clone, Copy, Default)]
struct SingleCache {
    entries: [(Mv, i8); MAX_DERIVED],
    count: usize,
    comparisons: u32,
}

impl SingleCache {
    /// dav2d `add_candidate_c2s`.
    fn add(&mut self, reference: i8, mv: Mv) {
        if self.comparisons < MAX_CACHE_COMPARISONS {
            let found = self.entries[..self.count]
                .iter()
                .position(|&(m, r)| m == mv && r == reference);
            self.comparisons += found.map_or(self.count, |m| m + 1) as u32;
            if found.is_some() {
                return;
            }
        }
        if self.count < MAX_DERIVED {
            self.entries[self.count] = (mv, reference);
            self.count += 1;
        }
    }
}

/// The search state (dav2d `refmvs_state`).
struct Search<'s> {
    store: &'s MotionStore,
    refs: RefPair,
    /// The global vectors of the pair.
    global: [Mv; 2],
    list: List,
    derived: List,
    single: SingleCache,
    /// The block's 8x8 unit in the superblock row's trajectory buffers.
    trajectory_unit: (usize, usize),
    /// The superblock row's first 8x8 row.
    row_start8: u32,
}

/// The candidate list of an empty search.
const DERIVED_LIMITS: (usize, u32) = (MAX_DERIVED, MAX_CACHE_COMPARISONS);
const LIST_LIMITS: (usize, u32) = (MAX_CANDIDATES, MAX_COMPARISONS);

impl Search<'_> {
    fn trajectory(&self, reference: i8) -> TemporalMv {
        let (row, column) = self.trajectory_unit;
        self.store
            .frame
            .temporal
            .trajectory(reference as usize, row, column)
    }

    /// The vector a TIP neighbour gives reference `n` of the TIP pair at
    /// 8x8 unit (`row8`, `column8`): its offset plus the scaled projected
    /// field.
    fn tip_vector(&self, b: &MotionUnit, n: usize, (row8, column8): (u32, u32)) -> Mv {
        let frame = &self.store.frame;
        let field = frame.projected(row8 as usize, column8 as usize).0.or_zero();
        scale_temporal(field, frame.tip.factors[n]).clipped_add(b.mv[0])
    }

    /// Add a spatial neighbour `b` lying at unit offsets (`y_off`,
    /// `x_off`), at 8x8 unit (`row8`, `column8`) of the frame (dav2d
    /// `add_spatial_candidate`).
    fn add_spatial(
        &mut self,
        (y_off, x_off): (i32, i32),
        weight: u16,
        b: &MotionUnit,
        (row8, column8): (i32, i32),
    ) {
        if self.list.count >= MAX_CANDIDATES || b.is_intra() {
            return;
        }
        let frame = &self.store.frame;
        let params = &frame.params;
        let (mut row8, mut column8) = (row8, column8);
        if b.refs.get(0) == TIP_REFERENCE {
            // TIP blocks keep 16x16 motion unless refined finely.
            let (bw4, bh4) = b.size.units();
            let coarse = if params.tip_mode == super::frame::TIP_FRAME_MODE {
                !params.tip_refine_mv || params.tip_filter != filter_mode::SHARP
            } else {
                (!params.tip_refine_mv && bw4.min(bh4) >= TIP_COARSE_MIN_UNITS)
                    || b.size == BlockSize::B256x256
            };
            if coarse {
                let (top8, left8) = ((b.row >> 1) as i32, (b.column >> 1) as i32);
                row8 = top8 + ((row8 - top8) & !1);
                column8 = left8 + ((column8 - left8) & !1);
            }
        }
        let at = (row8 as u32, column8 as u32);
        let offsets = (y_off as i8, x_off as i8);
        let refs = self.refs;
        let tip = frame.tip.refs;
        let b_tip = b.refs.get(0) == TIP_REFERENCE;
        let global = |n: usize, mv: Mv| {
            if b.global && self.global[n] != Mv::INVALID {
                self.global[n]
            } else {
                mv
            }
        };
        if !refs.is_compound() {
            self.add_spatial_single(weight, b, at, offsets);
        } else if b_tip && refs == tip {
            let field = frame.projected(at.0 as usize, at.1 as usize).0.or_zero();
            let mv = frame
                .tip
                .factors
                .map(|f| scale_temporal(field, f).clipped_add(b.mv[0]));
            self.list.add_pair(LIST_LIMITS, weight, EQUAL_WEIGHT, mv);
        } else if b.refs == refs {
            let mv = [global(0, b.mv[0]), global(1, b.mv[1])];
            self.list.add_pair(LIST_LIMITS, weight, b.cwp, mv);
        } else {
            self.add_cross(weight, b);
        }
    }

    /// A spatial neighbour for a one-reference list (dav2d
    /// `add_spatial_candidate`'s `ref.ref[1] == -1` branch): its matching
    /// vector, a TIP neighbour's projected one, or derived candidates
    /// (from the TIP pair, along trajectories, or projected by distance).
    fn add_spatial_single(
        &mut self,
        weight: u16,
        b: &MotionUnit,
        at: (u32, u32),
        offsets: (i8, i8),
    ) {
        let frame = &self.store.frame;
        let params = &frame.params;
        let target = self.refs.get(0);
        let tip = frame.tip.refs;
        let b_tip = b.refs.get(0) == TIP_REFERENCE;
        let global = if b.global && self.global[0] != Mv::INVALID {
            Some(self.global[0])
        } else {
            None
        };
        let count = 1 + usize::from(target >= 0);
        for n in 0..count {
            // The rank whose vector `b` brings for its reference `n`.
            let source = if b_tip { tip.get(n) } else { b.refs.get(n) };
            if b.refs.get(n) == target {
                self.list
                    .add_single(LIST_LIMITS, weight, global.unwrap_or(b.mv[n]), offsets);
            } else if b_tip && tip.get(n) == target {
                let mv = self.tip_vector(b, n, at);
                self.list.add_single(LIST_LIMITS, weight, mv, offsets);
            } else if target == TIP_REFERENCE && b.refs == tip {
                let delta = Mv {
                    y: b.mv[0].y.wrapping_sub(b.mv[1].y),
                    x: b.mv[0].x.wrapping_sub(b.mv[1].x),
                };
                let out = scale(delta, frame.tip.factors[0]);
                let mv = Mv {
                    y: b.mv[0]
                        .y
                        .wrapping_sub(out.y)
                        .clamp(-MAX_COMPONENT, MAX_COMPONENT),
                    x: b.mv[0]
                        .x
                        .wrapping_sub(out.x)
                        .clamp(-MAX_COMPONENT, MAX_COMPONENT),
                };
                self.derived.add_single(DERIVED_LIMITS, weight, mv, (0, 0));
                break;
            } else if params.trajectories
                && params.use_motion_fields
                && RefPair::rank(target).is_some()
                && (b_tip || RefPair::rank(b.refs.get(n)).is_some())
                && self.trajectory(target).is_valid()
                && self.trajectory(source).is_valid()
            {
                let (a, from) = if b_tip {
                    (self.trajectory(tip.get(n)), self.tip_vector(b, n, at))
                } else {
                    (self.trajectory(b.refs.get(n)), b.mv[n])
                };
                let c = self.trajectory(target);
                let along = |v: i32, c: i16, a: i16| {
                    (v + i32::from(c) - i32::from(a)).clamp(-MAX_COMPONENT, MAX_COMPONENT)
                };
                let mv = Mv {
                    y: along(from.y, c.y, a.y),
                    x: along(from.x, c.x, a.x),
                };
                self.derived.add_single(DERIVED_LIMITS, weight, mv, (0, 0));
            } else if RefPair::rank(target).is_some()
                && b.refs.get(0) >= 0
                && frame.future[target as usize] == frame.future[source as usize]
            {
                let (mv, den) = if b_tip {
                    (
                        self.tip_vector(b, n, at),
                        frame.abs_distance[tip.get(n) as usize],
                    )
                } else {
                    (b.mv[n], frame.abs_distance[b.refs.get(n) as usize])
                };
                let mv = mv.project(
                    i32::from(frame.abs_distance[target as usize]),
                    i32::from(den),
                );
                self.derived.add_single(DERIVED_LIMITS, weight, mv, (0, 0));
            }
            if !b.refs.is_compound() && !b_tip {
                break;
            }
        }
    }

    /// A compound pair from a neighbour with other references: through
    /// the trajectories, and by pairing its matching vector with a cached
    /// one of the other reference (dav2d `add_spatial_candidate`'s last
    /// branch).
    fn add_cross(&mut self, weight: u16, b: &MotionUnit) {
        let refs = self.refs;
        let params = &self.store.frame.params;
        let [first, second] = refs.0;
        if params.trajectories
            && params.use_motion_fields
            && b.refs.get(0) != TIP_REFERENCE
            && first != second
            && self.trajectory(first).is_valid()
            && self.trajectory(second).is_valid()
        {
            let (b1, b2) = (self.trajectory(first), self.trajectory(second));
            for n in 0..2 {
                if b.refs.get(n) < 0 {
                    break;
                }
                let a = self.trajectory(b.refs.get(n));
                if !a.is_valid() {
                    continue;
                }
                let along = |c: TemporalMv| Mv {
                    y: (b.mv[n].y + i32::from(c.y) - i32::from(a.y))
                        .clamp(-MAX_COMPONENT, MAX_COMPONENT),
                    x: (b.mv[n].x + i32::from(c.x) - i32::from(a.x))
                        .clamp(-MAX_COMPONENT, MAX_COMPONENT),
                };
                self.derived
                    .add_pair(DERIVED_LIMITS, weight, EQUAL_WEIGHT, [along(b1), along(b2)]);
            }
        }
        let ns = if first == b.refs.get(0) || first == b.refs.get(1) {
            0
        } else if second != b.refs.get(0) && second != b.refs.get(1) {
            return;
        } else {
            1
        };
        let nc = usize::from(refs.get(ns) != b.refs.get(0));
        let mv = if b.global && self.global[ns] != Mv::INVALID {
            self.global[ns]
        } else {
            b.mv[nc]
        };
        let other = refs.get(1 - ns);
        if let Some(&(cached, _)) = self.single.entries[..self.single.count]
            .iter()
            .find(|&&(_, r)| r == other)
        {
            let mut pair = [Mv::default(); 2];
            pair[ns] = mv;
            pair[1 - ns] = cached;
            self.derived
                .add_pair(DERIVED_LIMITS, weight, EQUAL_WEIGHT, pair);
        }
        self.single.add(b.refs.get(nc), mv);
    }

    /// Move the derived candidates into the list (dav2d `add_derived`).
    fn add_derived(&mut self, limit: usize) {
        for n in 0..self.derived.count {
            if self.list.count >= MAX_CANDIDATES {
                break;
            }
            let entry = self.derived.entries[n];
            if self.refs.is_compound() {
                self.list
                    .add_pair((limit, MAX_COMPARISONS), 0, EQUAL_WEIGHT, entry.mv);
            } else {
                self.list
                    .add_single((limit, MAX_COMPARISONS), 0, entry.mv[0], (0, 0));
            }
        }
    }

    /// One temporal candidate at 8x8 unit (`row8` within the superblock
    /// row, `column8`) (dav2d `add_temporal_candidate`); whether it was
    /// added.
    fn add_temporal(&mut self, (row8, column8): (u32, u32)) -> bool {
        let frame = &self.store.frame;
        let refs = self.refs;
        let Some(first) = RefPair::rank(refs.get(0)) else {
            return false;
        };
        let vector = |reference: usize| {
            let t = frame
                .temporal
                .trajectory(reference, row8 as usize, column8 as usize);
            if frame.params.trajectories && t.is_valid() {
                return Some(t.widen());
            }
            let (mv, distance) =
                frame.projected((self.row_start8 + row8) as usize, column8 as usize);
            mv.is_valid()
                .then(|| project_temporal(mv, i32::from(frame.distance[reference]), distance))
        };
        let Some(mv) = vector(first) else {
            return false;
        };
        if !refs.is_compound() {
            let weight = 1 + u16::from(frame.abs_distance[first] <= 2);
            let before = self.list.count;
            self.list.add_single(LIST_LIMITS, weight, mv, (0, 0));
            return self.list.count > before;
        }
        let Some(second) = vector(refs.get(1) as usize) else {
            return false;
        };
        let before = self.list.count;
        self.list
            .add_pair(LIST_LIMITS, 1, EQUAL_WEIGHT, [mv, second]);
        self.list.count > before
    }

    /// The temporal candidates of a block (dav2d `add_tmvp`): near its
    /// bottom-right inside, else just past it.
    fn add_temporal_candidates(
        &mut self,
        (row, column): (u32, u32),
        (bw4, bh4): (u32, u32),
        (w4, h4): (u32, u32),
    ) {
        if !self.store.frame.projecting {
            return;
        }
        let side = self.store.geometry.superblock_units();
        let (bw8, bh8) = ((bw4 >> 1).min(8), (bh4 >> 1).min(8));
        let step_h = if bw4 >= 16 { 2 } else { 1 };
        let step_v = if bh4 >= 16 { 2 } else { 1 };
        let x_off = 2 * bw8 as i32 - 2 * step_h;
        let y_off = 2 * bh8 as i32 - 2 * step_v;
        let unit = |dy: u32, dx: u32| (((row + dy) & (side - 1)) >> 1, u32::midpoint(column, dx));
        let first = (0..w4 as i32).contains(&x_off)
            && (0..h4 as i32).contains(&y_off)
            && self.add_temporal(unit(y_off as u32, x_off as u32));
        if !first && (bw4 > 4 || bh4 > 4) {
            self.add_temporal(unit(bh8, bw8));
        }
    }
}

/// The block a search is for.
struct Block {
    size: BlockSize,
    area: BlockArea,
    /// Its extent inside the tile.
    extent: (u32, u32),
    /// The first reference.
    target: i8,
    comp: bool,
    /// The pair's global vectors.
    global: [Mv; 2],
    bounds: Bounds,
    superblock_top: bool,
    have_left: bool,
    /// The bottom-most and top-most left neighbours.
    bottom_left: Option<MotionUnit>,
    top_left: Option<MotionUnit>,
    above: Above,
}

/// The warp models being listed.
struct WarpList {
    models: [[i32; 6]; MAX_WARPS],
    count: usize,
    /// Models are listed at all, for this reference.
    wanted: bool,
    reference: i8,
}

impl WarpList {
    /// Add a model while there is room.
    fn push(&mut self, model: [i32; 6]) {
        if self.count < MAX_WARPS {
            self.models[self.count] = model;
            self.count += 1;
        }
    }

    /// A neighbour's model joins, while there is room, when it warps from
    /// the same reference (dav2d checks the room only after the first
    /// three neighbours, which with the corner model cannot fill it).
    fn offer(&mut self, b: &MotionUnit) {
        if self.wanted && b.warped && b.refs.get(0) == self.reference {
            self.push(b.matrix);
        }
    }
}

impl Search<'_> {
    /// The neighbours in dav2d's order with the temporal candidates
    /// (`temporal`: whether they come first) and the neighbours' warp
    /// models; returns how many candidates are nearest.
    fn spatial(&mut self, block: &Block, temporal: Option<bool>, warps: &mut WarpList) -> usize {
        let g = self.store.geometry;
        let side = g.superblock_units();
        let BlockArea {
            row,
            column,
            bw4,
            bh4,
        } = block.area;
        let size = (bw4, bh4);
        if temporal == Some(true) {
            self.add_temporal_candidates((row, column), size, block.extent);
        }
        let row8 = |r: u32| (r >> 1) as i32;
        let left8 = (column as i32 - 1) >> 1;
        let top8 = (row as i32 - 1) >> 1;
        let above = &block.above;
        let (offset, width) = (above.offset as i32, above.width as i32);
        let column8 = |dx: i32| (column as i32 + dx) >> 1;
        let boundary = 1 << i32::from(block.superblock_top);
        if let Some(b) = &block.bottom_left {
            warps.offer(b);
            self.add_spatial((bh4 as i32 - 1, -1), 1, b, (row8(row + bh4 - 1), left8));
        }
        if let Some(b) = &above.right_most {
            warps.offer(b);
            let x = width - boundary - offset;
            self.add_spatial((-1, x), u16::from(x >= 0), b, (top8, column8(x)));
        }
        if let Some(b) = &block.top_left {
            warps.offer(b);
            self.add_spatial((0, -1), 1, b, (row8(row), left8));
        }
        if let Some(b) = &above.left_most {
            warps.offer(b);
            self.add_spatial(
                (-1, -offset),
                u16::from(offset == 0),
                b,
                (top8, column8(-offset)),
            );
        }
        let below = row + bh4;
        if block.have_left
            && bh4 <= CORNER_MAX_UNITS
            && below & (side - 1) != 0
            && below < g.row_end
        {
            let b = *self.store.unit(below, column - 1);
            warps.offer(&b);
            self.add_spatial((bh4 as i32, -1), 1, &b, (row8(below), left8));
        }
        if let Some(b) = &above.top_right {
            warps.offer(b);
            let x = width - offset;
            self.add_spatial((-1, x), 1, b, (top8, column8(x)));
        }
        if temporal == Some(false) && self.list.count < MAX_CANDIDATES {
            self.add_temporal_candidates((row, column), size, block.extent);
        }
        if let Some(b) = &above.top_left {
            warps.offer(b);
            let x = -boundary - offset;
            self.add_spatial((-1, x), 0, b, (top8, column8(x)));
        }
        let nearest = self.list.count;
        if block.have_left {
            self.further_left(block, warps);
        }
        nearest
    }

    /// Two or three units further left, where a different block lies
    /// there than beside the block.
    fn further_left(&mut self, block: &Block, warps: &mut WarpList) {
        let BlockArea {
            row,
            column,
            bw4,
            bh4,
        } = block.area;
        let adj = 3 - (column & u32::from(bw4 == 1));
        if column < self.store.geometry.column_start + adj {
            return;
        }
        let further = column - adj;
        for (near, unit_row, offset_y) in [
            (block.bottom_left, row + bh4 - 1, bh4 as i32 - 1),
            (block.top_left, row, 0),
        ] {
            let Some(near) = near else {
                continue;
            };
            let b = *self.store.unit(unit_row, further);
            if b.size.units().0 < adj || near.size != b.size {
                warps.offer(&b);
                self.add_spatial(
                    (offset_y, -(adj as i32)),
                    0,
                    &b,
                    ((unit_row >> 1) as i32, (further >> 1) as i32),
                );
            }
        }
    }

    /// Clip the listed vectors to the block's bounds.
    fn clip(&mut self, block: &Block) {
        let bounds = &block.bounds;
        for entry in &mut self.list.entries[..self.list.count] {
            for mv in &mut entry.mv[..=usize::from(block.comp)] {
                mv.y = mv.y.clamp(bounds.min.y, bounds.max.y);
                mv.x = mv.x.clamp(bounds.min.x, bounds.max.x);
            }
        }
    }

    /// The global vectors, then for large blocks combinations of the first
    /// three candidates' components.
    fn add_global_and_extra(&mut self, block: &Block) {
        let c = usize::from(block.comp);
        let global = block.global;
        let duplicate = self.list.find(MAX_COMPARISONS, |e| {
            e.mv[0] == global[0] && e.mv[c] == global[c]
        });
        if duplicate.is_none() {
            self.list.entries[self.list.count] = Candidate {
                mv: global,
                ..Candidate::default()
            };
            self.list.count += 1;
        }
        let (bw4, bh4) = (block.area.bw4, block.area.bh4);
        if bw4.min(bh4) <= EXTRA_MIN_UNITS || !(2..MAX_CANDIDATES).contains(&self.list.count) {
            return;
        }
        let list = self.list.entries;
        let pairs = if self.list.count == 2 {
            &EXTRA_PAIRS[..2]
        } else {
            &EXTRA_PAIRS[..]
        };
        self.derived = List::default();
        for (entry, &(y, x)) in self.derived.entries.iter_mut().zip(pairs) {
            for (i, mv) in entry.mv.iter_mut().enumerate().take(c + 1) {
                *mv = Mv {
                    y: list[y].mv[i].y,
                    x: list[x].mv[i].x,
                };
            }
        }
        self.derived.count = pairs.len();
        self.add_derived(MAX_CANDIDATES);
    }

    /// A block copy's default vectors: a superblock up, a superblock (and
    /// 256 samples) left, the block's height up, its width left.
    fn add_block_copy_defaults(&mut self, block: &Block, superblock_log2: u8, limit: usize) {
        let superblock = BLOCK_COPY_SUPERBLOCK << (superblock_log2 - 6);
        let (bw4, bh4) = (block.area.bw4 as i32, block.area.bh4 as i32);
        let defaults = [
            Mv {
                y: -(superblock * 8),
                x: 0,
            },
            Mv {
                y: 0,
                x: -(8 * (superblock + DEFAULT_LEFT_EXTRA)),
            },
            Mv {
                y: -(bh4 * VECTOR_UNITS_PER_UNIT),
                x: 0,
            },
            Mv {
                y: 0,
                x: -(bw4 * VECTOR_UNITS_PER_UNIT),
            },
        ];
        for mv in defaults {
            if self.list.count >= limit {
                break;
            }
            let entry = &mut self.list.entries[self.list.count];
            entry.mv[0] = mv;
            entry.weight = 0;
            self.list.count += 1;
        }
    }
}

/// Whether a neighbour serves the warp corner model of reference
/// `reference`, and which of its vectors (dav2d's `*_ref_idx` tests).
fn corner_index(b: &MotionUnit, reference: i8) -> Option<usize> {
    if b.refs.get(0) == reference {
        Some(0)
    } else if b.refs.get(1) == reference && !b.warped {
        Some(1)
    } else {
        None
    }
}

/// A corner's vector: the neighbour's, or its warp model's at sample
/// (`x`, `y`) clipped to the bounds (dav2d `get_warpmv_proj`).
fn corner_vector(b: &MotionUnit, index: usize, (x, y): (i32, i32), bounds: &Bounds) -> Mv {
    if !b.warped {
        return b.mv[index];
    }
    let m = &b.matrix;
    let xc = (m[2] - UNITY)
        .wrapping_mul(x)
        .wrapping_add(m[3].wrapping_mul(y))
        .wrapping_add(m[0]);
    let yc = (m[5] - UNITY)
        .wrapping_mul(y)
        .wrapping_add(m[4].wrapping_mul(x))
        .wrapping_add(m[1]);
    let round =
        |v: i32| ((v + 0x1000 - i32::from(v < 0)) >> 13).clamp(-MAX_COMPONENT, MAX_COMPONENT);
    Mv {
        y: round(yc).clamp(bounds.min.y, bounds.max.y),
        x: round(xc).clamp(bounds.min.x, bounds.max.x),
    }
}

/// A block's vector bounds (dav2d `minx` .. `maxy`).
struct Bounds {
    min: Mv,
    max: Mv,
}

/// The affine model through three corner vectors (dav2d
/// `model_from_corners`), if they differ and stay in the frame.
fn corner_model(
    [top_left, top_right, bottom_left]: [Mv; 3],
    (x, y): (i32, i32),
    size: BlockSize,
) -> Option<[i32; 6]> {
    if top_right == top_left && bottom_left == top_left {
        return None;
    }
    let (bw4, bh4) = size.units();
    let (w_log2, h_log2) = size.log2_units();
    let reach = |a: i32, b: i32, c: i32, extent: u32| {
        a.min(b).min(c + extent as i32 * VECTOR_UNITS_PER_UNIT)
    };
    if reach(top_left.x, bottom_left.x, top_right.x, bw4) < -x * 8
        || reach(top_left.y, top_right.y, bottom_left.y, bh4) < -y * 8
    {
        return None;
    }
    let slope = |d: i32, log2: u8| {
        ((i64::from(d) << CORNER_SCALE_LOG2) >> log2)
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    };
    let mut m = [0_i32; 6];
    m[2] = slope(top_right.x.wrapping_sub(top_left.x), w_log2);
    m[4] = slope(top_right.y.wrapping_sub(top_left.y), w_log2);
    m[3] = slope(bottom_left.x.wrapping_sub(top_left.x), h_log2);
    m[5] = slope(bottom_left.y.wrapping_sub(top_left.y), h_log2);
    let translation = |v: i32, a: i32, b: i32| {
        ((i64::from(v) << TRANSLATION_LOG2)
            - i64::from(x) * i64::from(a)
            - i64::from(y) * i64::from(b))
        .clamp(MODEL_TRANSLATION_MIN, MODEL_TRANSLATION_MAX) as i32
    };
    m[0] = translation(top_left.x, m[2], m[3]);
    m[1] = translation(top_left.y, m[4], m[5]);
    for entry in &mut m[2..] {
        let v = (*entry).clamp(-MODEL_ENTRY_MAX, MODEL_ENTRY_MAX);
        *entry = (v + MODEL_STEP_HALF - i32::from(v < 0)) & !MODEL_STEP_MASK;
    }
    m[2] += UNITY;
    m[5] += UNITY;
    Some(m)
}

/// The above neighbours of a block (dav2d `tl`, `lmt`, `rmt`, `tr`) and
/// the column offset at a superblock's top (`x_off`) and the rounded
/// width (`abw4`).
#[derive(Default)]
struct Above {
    top_left: Option<MotionUnit>,
    left_most: Option<MotionUnit>,
    right_most: Option<MotionUnit>,
    top_right: Option<MotionUnit>,
    offset: u32,
    width: u32,
}

impl MotionStore {
    /// The above neighbours of a block `bw4` units wide (`w4` inside the
    /// tile): in the row above within the superblock, or at a superblock's
    /// top from the 8x8 backup row.
    fn above_neighbours(&self, (row, column): (u32, u32), (bw4, w4): (u32, u32)) -> Above {
        let g = self.geometry;
        let side = g.superblock_units();
        let mut above = Above::default();
        if row <= g.row_start {
            return above;
        }
        if row & (side - 1) == 0 {
            above.offset = column & 1;
            above.width = (bw4 + 1) & !1;
            let at = |x8: u32| *self.above(x8);
            if column as i32 - above.offset as i32 - 2 >= g.column_start as i32 {
                above.top_left = Some(if column & (side - 2) != 0 {
                    at((column >> 1) - 1)
                } else {
                    *self.above_left()
                });
            }
            if bw4 > 2 {
                above.left_most = Some(at(column >> 1));
            }
            if bw4 == w4 {
                above.right_most = Some(at((column >> 1) + (above.width >> 1) - 1));
            }
            if column - above.offset + above.width < g.column_end && bw4 <= CORNER_MAX_UNITS {
                above.top_right = Some(at((column >> 1) + (above.width >> 1)));
            }
        } else {
            above.width = bw4;
            if column > g.column_start {
                above.top_left = Some(*self.unit(row - 1, column - 1));
            }
            if bw4 > 1 {
                above.left_most = Some(*self.unit(row - 1, column));
            }
            if bw4 == w4 {
                above.right_most = Some(*self.unit(row - 1, column + bw4 - 1));
            }
            let right = column + bw4;
            if right & (side - 1) != 0 && right < g.column_end && bw4 <= CORNER_MAX_UNITS {
                // Not decoded yet when still marked intra-or-undecoded.
                above.top_right = Some(*self.unit(row - 1, right)).filter(|unit| !unit.is_intra());
            }
        }
        above
    }

    /// The candidate list of the block at (`row`, `column`) for `refs`
    /// (dav2d `dav2d_refmvs_find`); `skip_mode` for skip-mode blocks,
    /// `warps` to list warp models too (one reference, not TIP).
    #[must_use]
    pub fn find(
        &self,
        refs: RefPair,
        size: BlockSize,
        skip_mode: bool,
        (row, column): (u32, u32),
        warps: bool,
    ) -> Candidates {
        let block = self.block(refs, size, (row, column));
        let side = self.geometry.superblock_units();
        let mut warp_list = WarpList {
            models: [[0; 6]; MAX_WARPS],
            count: 0,
            wanted: warps,
            reference: refs.get(0),
        };
        if warps && let Some(model) = Self::corner_warp(&block) {
            warp_list.push(model);
        }
        let mut search = Search {
            store: self,
            refs,
            global: block.global,
            list: List::default(),
            derived: List::default(),
            single: SingleCache::default(),
            trajectory_unit: (((row & (side - 1)) >> 1) as usize, (column >> 1) as usize),
            row_start8: (row & !(side - 1)) >> 1,
        };
        let params = &self.frame.params;
        let use_temporal = params.use_motion_fields && (refs.get(0) != refs.get(1) || skip_mode);
        let high_priority = !block.comp
            && block.target >= 0
            && u32::from(self.frame.high_priority) & (1 << (block.target as u32 + 1)) != 0;
        let nearest = search.spatial(
            &block,
            use_temporal.then_some(high_priority),
            &mut warp_list,
        );
        let reorder = match params.drl_reorder {
            DRL_REORDER_ALWAYS => nearest >= REORDER_MIN_ALWAYS,
            DRL_REORDER_CONSTRAINED => !high_priority && nearest >= REORDER_MIN_CONSTRAINED,
            _ => false,
        };
        if reorder {
            search.list.promote_heaviest(nearest);
        }
        let limit = 1 + usize::from(if block.target >= 0 {
            params.max_drl_bits
        } else {
            params.max_block_copy_drl_bits
        });
        if block.comp && search.list.count < limit {
            search.add_derived(limit);
        }
        if let Some(bank) = &self.bank {
            self.bank_candidates(&mut search, bank, block.area, limit);
        }
        if !block.comp && search.list.count < limit {
            search.add_derived(limit);
        }
        search.clip(&block);
        if search.list.count < MAX_CANDIDATES && block.target >= 0 {
            search.add_global_and_extra(&block);
        }
        if warps && warp_list.count < MAX_WARPS {
            let rank = block.target as usize;
            for matrix in self.warps.recent(rank) {
                warp_list.push(*matrix);
            }
            warp_list.push(self.frame.global[rank].matrix);
            for _ in 0..DEFAULT_WARPS {
                warp_list.push(IDENTITY);
            }
        }
        if block.target < 0 {
            search.add_block_copy_defaults(&block, self.geometry.superblock_log2, limit);
        }
        let mut result = Candidates {
            list: [Candidate::default(); MAX_CANDIDATES],
            count: search.list.count,
            warps: warp_list.models,
            warp_count: warp_list.count,
        };
        result.list[..result.count].copy_from_slice(&search.list.entries[..result.count]);
        result
    }

    /// What the search needs of the block: its extent, global vectors,
    /// bounds and neighbours.
    fn block(&self, refs: RefPair, size: BlockSize, (row, column): (u32, u32)) -> Block {
        let g = self.geometry;
        let frame = &self.frame;
        let params = &frame.params;
        let (bw4, bh4) = size.units();
        let w4 = bw4.min(g.column_end - column);
        let h4 = bh4.min(g.row_end - row);
        let comp = refs.is_compound();
        let area = BlockArea {
            row,
            column,
            bw4,
            bh4,
        };
        let global_of = |reference: i8| {
            frame
                .global_model(reference)
                .map_or(Mv::default(), |model| {
                    global_vector(
                        model,
                        area,
                        (frame.columns, frame.rows),
                        params.mv_precision,
                        params.force_integer_mv,
                    )
                })
        };
        let have_left = column > g.column_start;
        Block {
            size,
            area,
            extent: (w4, h4),
            target: refs.get(0),
            comp,
            global: [
                global_of(refs.get(0)),
                if comp {
                    global_of(refs.get(1))
                } else {
                    Mv::default()
                },
            ],
            bounds: Bounds {
                min: Mv {
                    y: -((row + bh4) as i32 + 4) * VECTOR_UNITS_PER_UNIT,
                    x: -((column + bw4) as i32 + 4) * VECTOR_UNITS_PER_UNIT,
                },
                max: Mv {
                    y: (frame.rows as i32 - row as i32 + 4) * VECTOR_UNITS_PER_UNIT,
                    x: (frame.columns as i32 - column as i32 + 4) * VECTOR_UNITS_PER_UNIT,
                },
            },
            superblock_top: row & (g.superblock_units() - 1) == 0,
            have_left,
            bottom_left: (have_left && bh4 == h4).then(|| *self.unit(row + bh4 - 1, column - 1)),
            top_left: (have_left && bh4 > 1).then(|| *self.unit(row, column - 1)),
            above: self.above_neighbours((row, column), (bw4, w4)),
        }
    }

    /// The warp model through the block's corner neighbours' vectors
    /// (dav2d's "Warp corners"): bottom-left with above-left and
    /// right-most above, else with left-most above and above-right.
    fn corner_warp(block: &Block) -> Option<[i32; 6]> {
        let BlockArea {
            row,
            column,
            bw4,
            bh4,
        } = block.area;
        let target = block.target;
        let corner = |b: &MotionUnit, index: usize, at: (u32, u32)| {
            corner_vector(
                b,
                index,
                ((at.1 * 4) as i32, (at.0 * 4) as i32),
                &block.bounds,
            )
        };
        let bl = block.bottom_left?;
        let bl_mv = corner(&bl, corner_index(&bl, target)?, (row + bh4, column));
        let position = ((column * 4) as i32, (row * 4) as i32);
        let above = &block.above;
        let through = |left: Option<MotionUnit>, right: Option<MotionUnit>| {
            let (left, right) = (left?, right?);
            let (i, j) = (corner_index(&left, target)?, corner_index(&right, target)?);
            let tl_mv = corner(&left, i, (row, column));
            let tr_mv = corner(&right, j, (row, column + bw4));
            corner_model([tl_mv, tr_mv, bl_mv], position, block.size)
        };
        through(above.top_left, above.right_most)
            .or_else(|| through(above.left_most, above.top_right))
    }

    /// The bank's entries of the pair's class, most recent first, that are
    /// not listed and point into the frame, up to `limit` candidates.
    fn bank_candidates(
        &self,
        search: &mut Search<'_>,
        bank: &MvBank,
        area: BlockArea,
        limit: usize,
    ) {
        let refs = search.refs;
        let comp = usize::from(refs.is_compound());
        let shared = MvBank::shared(refs);
        let frame = &self.frame;
        let (width, height) = (
            (frame.columns as i32) * SAMPLES_PER_UNIT,
            (frame.rows as i32) * SAMPLES_PER_UNIT,
        );
        for entry in bank.recent(refs) {
            if search.list.count >= limit {
                break;
            }
            if shared && entry.refs != refs {
                continue;
            }
            let mv = entry.mv;
            if search
                .list
                .find(MAX_COMPARISONS, |c| {
                    c.mv[0] == mv[0] && c.mv[comp] == mv[comp]
                })
                .is_some()
            {
                continue;
            }
            let inside = mv[..=comp].iter().all(|v| {
                let x = area.column as i32 * SAMPLES_PER_UNIT + whole(v.x, 3);
                let y = area.row as i32 * SAMPLES_PER_UNIT + whole(v.y, 3);
                x > -(area.bw4 as i32) * SAMPLES_PER_UNIT
                    && y > -(area.bh4 as i32) * SAMPLES_PER_UNIT
                    && x < width
                    && y < height
            });
            if !inside {
                continue;
            }
            let at = search.list.count;
            let candidate = &mut search.list.entries[at];
            candidate.mv = mv;
            candidate.weight = 0;
            if comp == 1 {
                candidate.cwp = entry.cwp;
            }
            (candidate.y_off, candidate.x_off) = (0, 0);
            search.list.count += 1;
        }
    }
}
