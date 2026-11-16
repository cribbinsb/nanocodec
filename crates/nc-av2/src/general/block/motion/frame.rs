//! The frame's part of motion prediction (dav2d `refmvs_frame`,
//! `dav2d_refmvs_init_frame` with `abs_closest_ref` and `topo_insert`;
//! dav2d commit 237229e, BSD-2-Clause, see `docs/licenses/dav2d`): what
//! the candidate lists need of the headers, each reference's distance and
//! side, the TIP references' scale factors, which references' motion
//! fields are projected onto this frame and how (the "mfmv" list), the
//! projected field ([`super::temporal`]) and the frame's own temporal
//! field, kept with the picture for later frames.
use std::rc::Rc;

use super::vector::{
    DIVISION_MULTIPLIERS, QuantizedMv, REFERENCES, RefPair, TIP_REFERENCE, TemporalMv,
};
use crate::general::frame_header::FrameHeader;
use crate::general::references::order_difference;
use crate::general::sequence::SequenceHeader;
use crate::general::warp::{WarpModel, kind};

/// Distances are kept within this many frames.
pub const MAX_DISTANCE: i32 = 31;
/// DRL reordering of the nearest candidates: always (`enable_drl_reorder`
/// 2), or only without both-side references and temporal priority (1).
pub const DRL_REORDER_ALWAYS: u8 = 2;
/// References this close are temporal candidates of high priority.
const HIGH_PRIORITY_DISTANCE: u8 = 2;
/// At most this many references are projected (one in the reduced
/// mode): three among the nearest, four in all.
const MAX_PROJECTIONS: usize = 4;
const MAX_NEAREST_PROJECTIONS: usize = 3;
/// A projection distance that is out of range (dav2d `INVALID_REF2CUR`).
pub const INVALID_DISTANCE: i8 = -32;
/// Luma samples per temporal-field unit (8x8), as a shift, and the
/// field's rows and stride round up to this many samples.
pub const FIELD_UNIT_LOG2: u32 = 3;
const FIELD_ALIGN_LOG2: u32 = 8;
/// The frame's TIP prediction (TIP OBUs).
pub const TIP_FRAME_MODE: u8 = 2;

/// The temporal field of a frame (dav2d `refmvs_temporal_block` array
/// `rp`): per 8x8 unit, the two quantized vectors and their references,
/// for later frames' projection.
#[derive(Debug, Clone)]
pub struct TemporalField {
    /// Units per row (the width rounded up to 256 samples).
    pub stride: usize,
    /// The vectors (the same twice for one reference).
    pub mvs: Vec<[QuantizedMv; 2]>,
    /// Their references (none where no vector is kept).
    pub refs: Vec<RefPair>,
}

impl TemporalField {
    fn new(stride: usize, rows: usize) -> Self {
        Self {
            stride,
            mvs: vec![[QuantizedMv::INVALID; 2]; stride * rows],
            refs: vec![RefPair::NONE; stride * rows],
        }
    }

    /// Record a block's vectors over its 8x8 units (dav2d
    /// `update_temporal`, TIP blocks): `refs` and `mv` in the order the
    /// field keeps them (swapped by the caller per [`FrameMotion::flip`]),
    /// a missing one replaced by the other.
    pub fn record(
        &mut self,
        (row8, column8): (usize, usize),
        (w8, h8): (usize, usize),
        refs: RefPair,
        mv: [super::vector::Mv; 2],
    ) {
        let (mvs, refs) = merge_pair(
            [QuantizedMv::quantize(mv[0]), QuantizedMv::quantize(mv[1])],
            refs,
        );
        for y in row8..row8 + h8 {
            let at = y * self.stride + column8;
            self.mvs[at..at + w8].fill(mvs);
            self.refs[at..at + w8].fill(refs);
        }
    }
}

/// A pair of field vectors with a missing one replaced by the other, the
/// pair "none" when both are missing (dav2d's `INVALID_TRAJ` cases).
#[must_use]
pub fn merge_pair(mut mvs: [QuantizedMv; 2], mut refs: RefPair) -> ([QuantizedMv; 2], RefPair) {
    match (mvs[0].is_valid(), mvs[1].is_valid()) {
        (false, false) => refs = RefPair::NONE,
        (false, true) => {
            mvs[0] = mvs[1];
            refs.0[0] = refs.0[1];
        }
        (true, false) => {
            mvs[1] = mvs[0];
            refs.0[1] = refs.0[0];
        }
        (true, true) => {}
    }
    (mvs, refs)
}

/// What a reference brings to the frame's motion prediction.
#[derive(Debug, Clone, Default)]
pub struct ReferenceMotion {
    /// Its display order.
    pub order: u32,
    /// The display order of each of its own references (all seven ranks).
    pub reference_orders: [u32; REFERENCES],
    /// How many references it had (0: intra).
    pub reference_count: u8,
    /// Its temporal field, when it kept one and has the frame's size.
    pub field: Option<Rc<TemporalField>>,
}

/// One reference whose motion field is projected (dav2d `MfmvRef` with
/// its `mfmv_ref2*` rows).
#[derive(Debug, Clone, Copy)]
pub struct Projection {
    /// The reference rank.
    pub reference: usize,
    /// The TIP reference it pairs with, if any (`tgt`).
    pub target: Option<usize>,
    /// Which of its vectors: 0 the first (toward the past), 1 the second.
    pub side: usize,
    /// Its distance to the current frame, or [`INVALID_DISTANCE`].
    pub to_current: i8,
    /// Per reference of it: its distance (0 when out of range).
    pub to_reference: [i8; REFERENCES],
    /// Per reference of it: that frame's rank in the current frame, if any.
    pub reference_rank: [Option<usize>; REFERENCES],
    /// Per reference of it: Q14 factors projecting its vectors to the
    /// current frame and to the other end of the trajectory.
    pub factors: [[i32; 2]; REFERENCES],
}

/// The TIP references and the Q14 factors of their vectors from the
/// projected field (dav2d `rf->tip`).
#[derive(Debug, Clone, Copy)]
pub struct TipMotion {
    /// The two references.
    pub refs: RefPair,
    /// Each one's factor.
    pub factors: [i32; 2],
    /// Their distance apart (at most 31).
    pub distance: u8,
}

/// What the motion code reads of the sequence and frame headers.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools, reason = "header flags")]
pub struct MotionParams {
    /// The superblock side in 4x4 units (dav2d `sbsz`).
    pub superblock_units: u32,
    /// `enable_drl_reorder`.
    pub drl_reorder: u8,
    /// The MV bank (`enable_refmvbank`).
    pub bank: bool,
    /// Motion trajectories (`enable_mv_traj`).
    pub trajectories: bool,
    /// The sequence keeps temporal fields (`enable_ref_frame_mvs`).
    pub keep_fields: bool,
    /// `enable_tip_refine_mv`.
    pub tip_refine_mv: bool,
    /// `max_drl_bits`, and block copies' `max_bvp_drl_bits`.
    pub max_drl_bits: u8,
    /// See `max_drl_bits`.
    pub max_block_copy_drl_bits: u8,
    /// `mv_precision` (0..3).
    pub mv_precision: u8,
    /// `force_integer_mv`.
    pub force_integer_mv: bool,
    /// `use_ref_frame_mvs`.
    pub use_motion_fields: bool,
    /// The temporal sample step (1 or 2 8x8 units).
    pub sample_step: u32,
    /// TIP: 0 off, 1 a reference, 2 the frame.
    pub tip_mode: u8,
    /// The TIP interpolation filter.
    pub tip_filter: u8,
    /// `tip_frame_hole_fill`.
    pub tip_hole_fill: bool,
    /// Key or intra-only.
    pub intra_frame: bool,
    /// `num_total_refs`.
    pub reference_count: usize,
}

/// The frame's motion prediction state (dav2d `refmvs_frame`).
#[derive(Debug, Clone)]
pub struct FrameMotion {
    /// Header values.
    pub params: MotionParams,
    /// The frame in 4x4 units (8x8-rounded) and 8x8 units.
    pub columns: u32,
    /// See `columns`.
    pub rows: u32,
    /// Units per row of the fields (dav2d `rp_stride`).
    pub stride: usize,
    /// Each rank's global motion, shear derived.
    pub global: [WarpModel; REFERENCES],
    /// Blocks may warp by each rank's global motion (dav2d
    /// `gmv_warp_allowed`).
    pub global_warp_allowed: [bool; REFERENCES],
    /// Each reference is in the future (dav2d `ref_sign`).
    pub future: [bool; REFERENCES],
    /// Current minus reference display order, within 31 (dav2d
    /// `pocdiff`), and its magnitude.
    pub distance: [i8; REFERENCES],
    /// See `distance`.
    pub abs_distance: [u8; REFERENCES],
    /// Ranks (bit `rank + 1`) whose temporal candidates come first.
    pub high_priority: u8,
    /// Per ordered pair of ranks (bit `8 * first + second`): the field
    /// keeps the pair's vectors swapped (dav2d `ref_flip`).
    pub flip: u64,
    /// The TIP references.
    pub tip: TipMotion,
    /// The references projected onto this frame, in order.
    pub projections: Vec<Projection>,
    /// Some reference is projected (dav2d `rf->use_ref_frame_mvs`).
    pub projecting: bool,
    /// Each rank's temporal field.
    pub references: [Option<Rc<TemporalField>>; REFERENCES],
    /// The projected field and the trajectories ([`super::temporal`]).
    pub temporal: super::temporal::Projected,
    /// This frame's temporal field, when the sequence keeps them.
    pub field: Option<TemporalField>,
}

/// `a - b` in display order as the sequence counts it (dav2d
/// `get_poc_diff`: zero without order hints).
fn difference(order_hint_bits: u8, a: u32, b: u32) -> i32 {
    if order_hint_bits == 0 {
        0
    } else {
        order_difference(a, b)
    }
}

/// The smallest distance from a reference to one of its own references
/// on side `future` of it that is on the same side of the current frame
/// (dav2d `abs_closest_ref`), or 255.
fn closest(to_reference: &[i32; REFERENCES], to_current: &[i32; REFERENCES], future: bool) -> u32 {
    (0..REFERENCES)
        .filter(|&n| {
            if future {
                to_current[n] > 0 && to_reference[n] > 0
            } else {
                to_current[n] < 0 && to_reference[n] < 0
            }
        })
        .map(|n| to_reference[n].unsigned_abs())
        .fold(u8::MAX.into(), u32::min)
}

/// Depth-first order of the references by "is a reference of" (dav2d
/// `topo_insert`).
fn topological(
    rank: usize,
    order: &mut Vec<usize>,
    position: &mut [Option<usize>; REFERENCES],
    edges: &[[Option<usize>; REFERENCES]; REFERENCES],
    counts: [u8; REFERENCES],
) {
    if position[rank].is_some() {
        return;
    }
    position[rank] = Some(0);
    if counts[rank] != 0 {
        for edge in edges[rank].iter().flatten() {
            topological(*edge, order, position, edges, counts);
        }
    }
    position[rank] = Some(order.len());
    order.push(rank);
}

impl FrameMotion {
    /// The state of a frame (dav2d `dav2d_refmvs_init_frame`), from the
    /// headers and what each rank's reference brings (unused for intra
    /// frames).
    #[must_use]
    pub fn new(
        sequence: &SequenceHeader,
        header: &FrameHeader,
        references: &[ReferenceMotion; REFERENCES],
    ) -> Self {
        let params = MotionParams {
            superblock_units: 1 << (header.superblock_log2 - 2),
            drl_reorder: sequence.references.drl_reorder,
            bank: sequence.references.mv_bank,
            trajectories: sequence.inter.mv_trajectories,
            keep_fields: sequence.inter.motion_fields,
            tip_refine_mv: sequence.inter.tip_refine_mv,
            max_drl_bits: header.max_drl_bits,
            max_block_copy_drl_bits: header.intra_block_copy.map_or(0, |copy| copy.max_drl_bits),
            mv_precision: header.mv_precision,
            force_integer_mv: header.force_integer_mv,
            use_motion_fields: header.use_motion_fields,
            sample_step: u32::from(header.motion_field_step.max(1)),
            tip_mode: header.tip.mode,
            tip_filter: header.tip.filter,
            tip_hole_fill: header.tip.hole_fill,
            intra_frame: header.is_intra(),
            reference_count: usize::from(header.reference_count),
        };
        let columns8 = header.width.div_ceil(8);
        let rows8 = header.height.div_ceil(8);
        let align = |size: u32| {
            (size + (1 << FIELD_ALIGN_LOG2) - 1) >> FIELD_ALIGN_LOG2
                << (FIELD_ALIGN_LOG2 - FIELD_UNIT_LOG2)
        };
        let stride = align(header.width) as usize;
        let mut global = [WarpModel::default(); REFERENCES];
        let mut global_warp_allowed = [false; REFERENCES];
        for rank in 0..REFERENCES {
            let warp = header.global_motion.warps[rank];
            let mut model = WarpModel::of_matrix(warp.kind, warp.matrix);
            let sheared = model.shear();
            // References of the frame's size only (no scaled prediction).
            global_warp_allowed[rank] =
                warp.kind > kind::TRANSLATION && !header.force_integer_mv && sheared;
            global[rank] = model;
        }
        let mut motion = Self {
            params,
            columns: columns8 * 2,
            rows: rows8 * 2,
            stride,
            global,
            global_warp_allowed,
            future: [false; REFERENCES],
            distance: [0; REFERENCES],
            abs_distance: [0; REFERENCES],
            high_priority: 0,
            flip: 0,
            tip: TipMotion {
                refs: RefPair([
                    header.tip.references[0] as i8,
                    header.tip.references[1] as i8,
                ]),
                factors: [0; 2],
                distance: 0,
            },
            projections: Vec::new(),
            projecting: false,
            references: Default::default(),
            temporal: super::temporal::Projected::new(&params, stride, rows8 as usize),
            field: params
                .keep_fields
                .then(|| TemporalField::new(stride, align(header.height) as usize)),
        };
        if !header.is_intra() {
            motion.distances(sequence, header, references);
        }
        motion
    }

    /// The per-reference values and the projection list.
    fn distances(
        &mut self,
        sequence: &SequenceHeader,
        header: &FrameHeader,
        references: &[ReferenceMotion; REFERENCES],
    ) {
        let bits = sequence.inter.order_hint_bits;
        let diff = |a: u32, b: u32| difference(bits, a, b);
        let count = self.params.reference_count;
        let current = header.order;
        let mut to_reference = [[0_i32; REFERENCES]; REFERENCES];
        let mut to_current = [[0_i32; REFERENCES]; REFERENCES];
        let mut edges = [[None; REFERENCES]; REFERENCES];
        let mut has_side = [[false; 2]; REFERENCES];
        let counts: [u8; REFERENCES] = std::array::from_fn(|i| references[i].reference_count);
        for i in 0..count {
            let reference = &references[i];
            self.future[i] = diff(reference.order, current) < 0;
            self.distance[i] =
                diff(current, reference.order).clamp(-MAX_DISTANCE, MAX_DISTANCE) as i8;
            self.abs_distance[i] = self.distance[i].unsigned_abs();
            if counts[i] == 0 {
                continue;
            }
            for n in 0..REFERENCES {
                let order = reference.reference_orders[n];
                to_reference[i][n] = diff(reference.order, order);
                has_side[i][0] |= to_reference[i][n] > 0;
                has_side[i][1] |= to_reference[i][n] < 0;
                to_current[i][n] = diff(current, order);
                edges[i][n] = (0..count).find(|&m| references[m].order == order);
            }
        }
        let both_sides = header.has_future_references && header.has_past_references;
        if sequence.references.drl_reorder != DRL_REORDER_ALWAYS && !both_sides {
            for i in 0..count {
                if self.abs_distance[i] <= HIGH_PRIORITY_DISTANCE {
                    self.high_priority |= 2 << i;
                }
            }
        }
        for i in 0..count {
            for n in 0..count {
                let flip = if self.future[i] == self.future[n] {
                    diff(references[i].order, references[n].order) < 0
                } else {
                    self.future[n]
                };
                self.flip |= u64::from(flip) << (i * 8 + n);
            }
        }
        if header.tip.mode != 0 {
            self.tip_factors(bits, header, references);
        }
        if header.use_motion_fields && bits != 0 {
            for (rank, reference) in references.iter().enumerate() {
                self.references[rank].clone_from(&reference.field);
            }
            self.projection_list(
                sequence,
                header,
                references,
                (&to_reference, &to_current, &edges, &has_side),
                counts,
            );
        }
        self.projecting = !self.projections.is_empty();
    }

    /// The TIP references' factors (dav2d's "tip setup").
    fn tip_factors(
        &mut self,
        bits: u8,
        header: &FrameHeader,
        references: &[ReferenceMotion; REFERENCES],
    ) {
        let [first, second] = self.tip.refs.0.map(|r| r as usize);
        let both_sides = header.has_future_references && header.has_past_references;
        let (a, b) = (references[first].order, references[second].order);
        let apart = if both_sides {
            difference(bits, b, a)
        } else {
            difference(bits, a, b)
        };
        self.tip.distance = apart.unsigned_abs().min(MAX_DISTANCE as u32) as u8;
        let multiplier = i32::from(DIVISION_MULTIPLIERS[usize::from(self.tip.distance)]);
        let factor = |d: i8| {
            let f = i32::from(d.unsigned_abs()).min(MAX_DISTANCE) * multiplier;
            if (d < 0) ^ (apart < 0) { -f } else { f }
        };
        self.tip.factors = [factor(self.distance[first]), factor(self.distance[second])];
    }

    /// Which references' fields are projected, in order (dav2d's
    /// "temporal MV setup"): the TIP pair's, the nearest on each side,
    /// the past ones again the other way, then the rest in dependency
    /// order; with each one's distances and factors.
    #[allow(clippy::type_complexity, reason = "the setup's tables")]
    fn projection_list(
        &mut self,
        sequence: &SequenceHeader,
        header: &FrameHeader,
        references: &[ReferenceMotion; REFERENCES],
        (to_reference, to_current, edges, has_side): (
            &[[i32; REFERENCES]; REFERENCES],
            &[[i32; REFERENCES]; REFERENCES],
            &[[Option<usize>; REFERENCES]; REFERENCES],
            &[[bool; 2]; REFERENCES],
        ),
        counts: [u8; REFERENCES],
    ) {
        let bits = sequence.inter.order_hint_bits;
        let diff = |a: u32, b: u32| difference(bits, a, b);
        let count = self.params.reference_count;
        // Nearest past first (largest positive distance first, stable).
        let mut order: Vec<usize> = Vec::with_capacity(count);
        for n in 0..count {
            let at = order
                .iter()
                .position(|&m| self.distance[n] > self.distance[m])
                .unwrap_or(order.len());
            order.insert(at, n);
        }
        let first_future = order.iter().take_while(|&&n| self.future[n]).count();
        let mut topo = Vec::with_capacity(REFERENCES);
        let mut position = [None; REFERENCES];
        for n in 0..count {
            topological(n, &mut topo, &mut position, edges, counts);
        }
        if topo.len() <= 1 {
            return;
        }
        let has_field = |n: usize| self.references[n].is_some();
        let mut done = [[false; 2]; REFERENCES];
        for (n, done) in done.iter_mut().enumerate().take(count) {
            if !has_field(n) {
                *done = [true, true];
            }
        }
        let mut list: Vec<(usize, Option<usize>, usize)> = Vec::with_capacity(MAX_PROJECTIONS);
        let [tip0, tip1] = self.tip.refs.0.map(|r| r as usize);
        if sequence.inter.tip != 0 && (self.future[tip0] || self.future[tip1]) {
            let o = usize::from(position[tip0] > position[tip1]);
            let tips = [tip0, tip1];
            let (from, to) = (tips[1 - o], tips[o]);
            let side = usize::from(diff(references[from].order, references[to].order) < 0);
            // References without a usable field are not projected.
            if has_field(from) {
                list.push((from, Some(to), side));
                done[from][side] = true;
            }
        }
        let reduced = sequence.inter.reduced_motion_fields;
        let limit = if reduced { 1 } else { MAX_NEAREST_PROJECTIONS };
        let push = |list: &mut Vec<_>,
                    reference: usize,
                    side: usize,
                    done: &mut [[bool; 2]; REFERENCES]| {
            list.push((reference, None, side));
            done[reference][side] = true;
            list.len() == limit
        };
        if list.len() < limit {
            'nearest: for n in 0..2 {
                let past = (first_future > n)
                    .then(|| order[first_future - n - 1])
                    .filter(|&r| has_side[r][1]);
                let future = (first_future + n < count)
                    .then(|| order[first_future + n])
                    .filter(|&r| has_side[r][0]);
                let past_first = match (past, future) {
                    (Some(p), Some(f)) => {
                        let reach = |r: usize, side: bool| {
                            if has_field(r) {
                                closest(&to_reference[r], &to_current[r], side)
                            } else {
                                u8::MAX.into()
                            }
                        };
                        reach(p, false) < reach(f, true)
                    }
                    _ => false,
                };
                if past_first
                    && let Some(p) = past.filter(|&p| !done[p][1])
                    && push(&mut list, p, 1, &mut done)
                {
                    break 'nearest;
                }
                if let Some(f) = future.filter(|&f| !done[f][0])
                    && push(&mut list, f, 0, &mut done)
                {
                    break 'nearest;
                }
                if !past_first
                    && let Some(p) = past.filter(|&p| !done[p][1])
                    && push(&mut list, p, 1, &mut done)
                {
                    break 'nearest;
                }
            }
        }
        self.push_backward_and_rest(&mut list, &mut done, (&order, first_future), &topo, limit);
        self.projections = list
            .into_iter()
            .map(|(reference, target, side)| {
                self.projection(bits, header, references, (reference, target, side))
            })
            .collect();
    }

    /// The projection list's tail: the nearest past references again in
    /// the other direction (within the nearest `limit`), then every
    /// reference in reverse dependency order, each side once.
    fn push_backward_and_rest(
        &self,
        list: &mut Vec<(usize, Option<usize>, usize)>,
        done: &mut [[bool; 2]; REFERENCES],
        (order, first_future): (&[usize], usize),
        topo: &[usize],
        limit: usize,
    ) {
        if list.len() < limit && first_future > 0 {
            let reference = order[first_future - 1];
            if !done[reference][0] {
                list.push((reference, None, 0));
                done[reference][0] = true;
            }
            if list.len() < limit && first_future > 1 {
                let reference = order[first_future - 2];
                if !done[reference][0] {
                    list.push((reference, None, 0));
                    done[reference][0] = true;
                }
            }
        }
        let limit = if limit == 1 { 1 } else { MAX_PROJECTIONS };
        if list.len() < limit {
            'rest: for &reference in topo.iter().rev() {
                let side = usize::from(self.distance[reference] >= 0);
                for side in [side, 1 - side] {
                    if !done[reference][side] {
                        list.push((reference, None, side));
                        done[reference][side] = true;
                        if list.len() == limit {
                            break 'rest;
                        }
                    }
                }
            }
        }
    }

    /// One projected reference's distances and factors.
    fn projection(
        &self,
        bits: u8,
        header: &FrameHeader,
        references: &[ReferenceMotion; REFERENCES],
        (reference, target, side): (usize, Option<usize>, usize),
    ) -> Projection {
        let count = self.params.reference_count;
        let mut projection = Projection {
            reference,
            target,
            side,
            to_current: INVALID_DISTANCE,
            to_reference: [0; REFERENCES],
            reference_rank: [None; REFERENCES],
            factors: [[0; 2]; REFERENCES],
        };
        let order = references[reference].order;
        let to_current = difference(bits, order, header.order);
        if to_current.abs() > MAX_DISTANCE {
            return projection;
        }
        projection.to_current = to_current as i8;
        for m in 0..REFERENCES {
            let far = references[reference].reference_orders[m];
            let to_reference = difference(bits, order, far);
            let d2 = if to_reference.abs() <= MAX_DISTANCE {
                to_reference
            } else {
                0
            };
            projection.to_reference[m] = d2 as i8;
            // Frames at the projected reference's own order (overlays,
            // other layers) end no trajectory.
            if to_reference != 0 {
                projection.reference_rank[m] = (0..count).find(|&l| far == references[l].order);
            }
            let d1 = to_current;
            let multiplier = i32::from(
                DIVISION_MULTIPLIERS[d2.unsigned_abs().min(MAX_DISTANCE as u32) as usize],
            );
            let mut f0 = d1.abs().min(MAX_DISTANCE) * multiplier;
            if (d1 < 0) ^ (d2 < 0) {
                f0 = -f0;
            }
            let d3 = d1 - d2;
            let mut f1 = d3.abs().min(MAX_DISTANCE) * multiplier;
            if (d3 < 0) ^ (d2 > 0) {
                f1 = -f1;
            }
            projection.factors[m] = [f0, f1];
        }
        projection
    }

    /// The global motion vector source of a rank for candidate lists.
    #[must_use]
    pub fn global_model(&self, reference: i8) -> Option<&WarpModel> {
        RefPair::rank(reference).map(|rank| &self.global[rank])
    }

    /// Whether the pair `refs` is kept swapped in the temporal field.
    #[must_use]
    pub fn flipped(&self, refs: RefPair) -> bool {
        let [a, b] = refs.0;
        a >= 0
            && b >= 0
            && a < TIP_REFERENCE
            && b < TIP_REFERENCE
            && self.flip >> (a as usize * 8 + b as usize) & 1 != 0
    }

    /// The frame's temporal field, to keep with the picture.
    #[must_use]
    pub fn into_field(self) -> Option<TemporalField> {
        self.field
    }

    /// A projected vector and its reference distance (for TIP prediction,
    /// dav2d's `rp_proj` reads): row and column in 8x8 units of the frame.
    #[must_use]
    pub fn projected(&self, row8: usize, column8: usize) -> (TemporalMv, u8) {
        self.temporal.projected(row8, column8)
    }
}
