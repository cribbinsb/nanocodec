//! Predictions that refine their vectors per sub-block: TIP blocks (the
//! projected motion field plus the block's offset, optionally refined),
//! optical flow and decoder-side MV refinement of compound blocks, and
//! chroma predicted from those refined vectors (dav2d `tip_pred`,
//! `opfl_pred`, `rmv_uvpred`, `opfl_mv_adj`).
use super::blend::{DEFERRED_MASK, Implicit};
use super::kernels::{self, Regression};
use super::mc::{Bounds, Target};
use super::{EQUAL_WEIGHT, InterPrediction, Plane, TIP_SINGLE_WEIGHT, UNIT_LOG2};
use crate::general::block::Tile;
use crate::general::block::inter_record::{CompoundType, InterMode, Reference};
use crate::general::block::motion::Mv;
use crate::general::block::size::BlockSize;
use crate::general::frame_header::filter_mode;
use crate::general::warp;

/// Vectors in sixteenth samples are rounded to eighths with the
/// refinement's own rounding (dav2d's `(v + (d > 0)) >> 1`).
const SIXTEENTH_SHIFT: u32 = 1;
/// The refinement windows' search margin: half a sample... two 4x4 units
/// in total (dav2d `step + 2`), read 4 samples up and left (32 eighths).
const REFINE_MARGIN_UNITS: u32 = 2;
const REFINE_OFFSET_EIGHTHS: i32 = 32;
const REFINE_OFFSET_SAMPLES: usize = 4;
/// SAD thresholds below which TIP skips optical flow, by TIP mode.
const TIP_REFERENCE_SAD: u32 = 6;
const TIP_FRAME_SAD: u32 = 15;
/// Optical-flow corrections are clipped to a sample (sixteenths).
const FLOW_LIMIT: i32 = 16;
/// The regression's products are reduced to this many bits.
const FLOW_PRODUCT_BITS: i32 = 23;
const FLOW_SOLUTION_BITS: i32 = 22;
const FLOW_SOLUTION_SHIFT: i32 = 3;
/// Rows of the refined-vector grid (8x8 units of a 128x128 area).
const REFINED_STRIDE: usize = 16;
const REFINED_MASK: u32 = 31;
/// The 8-tap filters reach this far beyond a sub-block on either side
/// (`3` before, `4` after: the window adds 7).
const TAPS: i32 = 7;
const TAPS_BEFORE: i32 = 3;

/// One sub-block's vector corrections (dav2d `OpflMvDeltaBlock`).
#[derive(Debug, Clone, Copy, Default)]
struct Correction {
    deltas: [Mv; 2],
}

/// The reference distances of an optical-flow pair as its weights
/// (dav2d `union aliasi16 d`): zero when either distance is.
fn flow_weights(tile: &Tile<'_, '_>, references: [Reference; 2]) -> [i8; 2] {
    let refs = &tile.references;
    let [r0, r1] = references.map(|r| r as usize);
    let (d0, d1) = (refs.abs_distance[r0], refs.abs_distance[r1]);
    if d0 == 0 || d1 == 0 {
        return [0, 0];
    }
    let signed = |magnitude: i8, sign: i8| if sign < 0 { -magnitude } else { magnitude };
    [
        signed(1 + i8::from(d0 > d1), -refs.distance[r0]),
        signed(1 + i8::from(d1 > d0), refs.distance[r1]),
    ]
}

/// The corrections of a sub-block from its gradient regression (dav2d
/// `opfl_mv_adj`).
fn flow_correction(r: &Regression, d: [i8; 2]) -> Correction {
    let bits = |v: i32| 1 + (v.unsigned_abs() + u32::from(v == 0)).ilog2() as i32;
    let (mut su2, mut suv, mut sv2, mut suw, mut svw) = (r.su2, r.suv, r.sv2, r.suw, r.svw);
    let nbits_max = (bits(su2) + bits(sv2))
        .max(bits(sv2) + bits(suw))
        .max(bits(suv) + bits(svw))
        .max(bits(su2) + bits(svw))
        .max(bits(suv) + bits(suw));
    let rbits = (nbits_max - FLOW_PRODUCT_BITS).max(0) >> 1;
    if rbits != 0 {
        let rnd = (1 << rbits) >> 1;
        su2 = (su2 + rnd) >> rbits;
        sv2 = (sv2 + rnd) >> rbits;
        suv = (suv + rnd - i32::from(suv < 0)) >> rbits;
        suw = (suw + rnd - i32::from(suw < 0)) >> rbits;
        svw = (svw + rnd - i32::from(svw < 0)) >> rbits;
    }
    let det = su2 * sv2 - suv * suv;
    if det <= 0 {
        return Correction::default();
    }
    let mut s = [sv2 * suw - suv * svw, su2 * svw - suv * suw];
    let (idet, shift) = warp::resolve_divisor(det as u64);
    let shift = shift as i32;
    let idet_bits = (idet as u32).ilog2() as i32;
    for value in &mut s {
        if *value == 0 {
            continue;
        }
        let mut magnitude = value.unsigned_abs() as i32;
        let rbits = (magnitude.ilog2() as i32 + idet_bits - FLOW_SOLUTION_BITS).max(0);
        if rbits > 0 {
            magnitude = (magnitude + ((1 << rbits) >> 1)) >> rbits;
        }
        let ibits = FLOW_SOLUTION_SHIFT + rbits - shift;
        magnitude = if ibits >= 0 {
            magnitude * idet * (1 << ibits)
        } else {
            (magnitude * idet + ((1 << -ibits) >> 1)) >> -ibits
        };
        *value = if *value < 0 { -magnitude } else { magnitude };
    }
    let clip = |v: i32| v.clamp(-FLOW_LIMIT, FLOW_LIMIT);
    let (d0, d1) = (i32::from(d[0]), i32::from(d[1]));
    Correction {
        deltas: [
            Mv {
                x: -clip(d0 * s[0]),
                y: -clip(d0 * s[1]),
            },
            Mv {
                x: clip(d1 * s[0]),
                y: clip(d1 * s[1]),
            },
        ],
    }
}

/// Halve a sixteenth-sample vector for 4:2:0 chroma (dav2d
/// `scaledown_16pel_mv_for_chroma`).
fn chroma_sixteenths(mv: Mv) -> Mv {
    let half = |v: i32| (v + i32::from(v > 0)) >> 1;
    Mv {
        x: half(mv.x),
        y: half(mv.y),
    }
}

/// Luma units as 4:2:0 chroma units.
fn chroma_units(luma: u32) -> u32 {
    luma >> 1
}

/// A refined sixteenth-sample vector back to eighths, rounding with the
/// correction's sign (dav2d's `dmv`).
fn to_eighths(mv: Mv, delta: Mv) -> Mv {
    Mv {
        y: (mv.y + i32::from(delta.y > 0)) >> SIXTEENTH_SHIFT,
        x: (mv.x + i32::from(delta.x > 0)) >> SIXTEENTH_SHIFT,
    }
}

impl Tile<'_, '_> {
    /// The index of the refined-vector grid at luma unit (`row`,
    /// `column`) (dav2d `t->rmv`).
    fn refined_index(row: u32, column: u32) -> usize {
        ((row & REFINED_MASK) >> 1) as usize * REFINED_STRIDE
            + ((column & REFINED_MASK) >> 1) as usize
    }

    /// Bounds of a refinement around a sub-block starting at
    /// (`left`, `top`) of `w` x `h` samples in a frame of
    /// (`width`, `height`).
    fn refine_bounds(
        (left, top): (i32, i32),
        (w, h): (i32, i32),
        (width, height): (i32, i32),
    ) -> Bounds {
        Bounds {
            left: left.clamp(0, width - 1),
            right: (left + w + TAPS).clamp(1, width),
            top: top.clamp(0, height - 1),
            bottom: (top + h + TAPS).clamp(1, height),
        }
    }

    /// A TIP block's luma: per sub-block of the projected motion field
    /// plus the block's vector, refined where the frame allows it; into
    /// the picture (one TIP reference) or the intermediates. Returns the
    /// implicit mask's state.
    #[allow(clippy::too_many_lines, reason = "dav2d tip_pred")]
    pub(super) fn tip_prediction(
        &mut self,
        block: &InterPrediction,
        target: Target,
        (bw4, bh4): (u32, u32),
        (w4, h4): (u32, u32),
    ) -> Implicit {
        let frame_refines = self.tip_refines();
        let bidirectional = block.inter.weight != TIP_SINGLE_WEIGHT;
        let tip = self.frame.tip.references.map(|r| r as Reference);
        let references = if bidirectional { tip } else { [tip[0], tip[0]] };
        let [r0, r1] = references.map(|r| r as usize);
        let refine = frame_refines
            && self.frame.tip.mode == 1
            && self.sequence.inter.refine_mv
            && self.references.distance[r0] == -self.references.distance[r1];
        let step = self.tip_step(block);
        let flow = frame_refines
            && self.sequence.inter.optical_flow != 0
            && self.frame.has_future_references
            && self.frame.has_past_references;
        let implicit =
            self.sequence.inter.implicit_masked_blend && block.inter.weight == EQUAL_WEIGHT;
        if implicit {
            self.inter_scratch.mask[0] = DEFERRED_MASK;
        }
        let mut have_mask = false;
        let d = if flow {
            flow_weights(self, references)
        } else {
            [0, 0]
        };
        let threshold = if self.frame.tip.mode == 1 {
            TIP_REFERENCE_SAD
        } else {
            TIP_FRAME_SAD
        };
        let (width, height) = (
            (self.geometry.columns << UNIT_LOG2) as i32,
            (self.geometry.rows << UNIT_LOG2) as i32,
        );
        let (row0, column0) = (self.row, self.column);
        let swap = self.motion_store().reference_flip(r0, r1);
        let intermediate_stride = (bw4 << UNIT_LOG2) as usize;
        for y in (0..h4).step_by(step as usize) {
            for x in (0..w4).step_by(step as usize) {
                let (row, column) = (row0 + y, column0 + x);
                let field = self.motion_store().tip_vector(row >> 1, column >> 1);
                let mut cmv = [Mv::default(); 2];
                let mut corner = [(0, 0); 2];
                for i in 0..=usize::from(bidirectional) {
                    let projected = self.motion_store().tip_scaled(field, i);
                    let limit = 0xffff;
                    cmv[i] = Mv {
                        y: (projected.y + block.mv[0].y).clamp(-limit, limit),
                        x: (projected.x + block.mv[0].x).clamp(-limit, limit),
                    };
                    corner[i] = (
                        ((column << UNIT_LOG2) as i32) + (cmv[i].x >> 3) - TAPS_BEFORE,
                        ((row << UNIT_LOG2) as i32) + (cmv[i].y >> 3) - TAPS_BEFORE,
                    );
                }
                if !bidirectional {
                    cmv[1] = cmv[0];
                    corner[1] = corner[0];
                }
                let index = Self::refined_index(row, column);
                self.inter_scratch.refined[index][1] = cmv;
                let sub = (step << UNIT_LOG2) as i32;
                if flow || refine {
                    let refined = self.refine_sub_block(
                        (row, column),
                        cmv,
                        references,
                        (step, corner, (width, height)),
                        (refine, true),
                        (
                            flow && !(block.size == BlockSize::B256x256
                                && self.frame.tip.mode == 1),
                            threshold,
                            d,
                        ),
                    );
                    let mut cmv = refined.0;
                    let correction = refined.1;
                    for i in 0..2 {
                        let offset = (y << UNIT_LOG2) as usize * intermediate_stride
                            + (x << UNIT_LOG2) as usize;
                        let bounds = Self::refine_bounds(corner[i], (sub, sub), (width, height));
                        self.mc_sixteenth(
                            Target::Compound {
                                index: i,
                                offset,
                                stride: intermediate_stride,
                            },
                            Plane::Luma,
                            (step, step),
                            (column, row),
                            cmv[i],
                            references[i] as usize,
                            block.inter.filter,
                            bounds,
                        );
                    }
                    let temporal = [
                        to_eighths(cmv[0], correction.deltas[0]),
                        to_eighths(cmv[1], correction.deltas[1]),
                    ];
                    self.motion_store_mut().update_temporal(
                        (row >> 1, column >> 1),
                        (step >> 1, step >> 1),
                        references,
                        temporal,
                        swap,
                    );
                    if implicit {
                        let deferred = (bw4 * bh4 * 16) as usize;
                        have_mask |= self.implicit_mask(
                            (intermediate_stride, (column0, x), (row0, y)),
                            cmv,
                            (4, 4),
                            (step, step),
                            deferred,
                            Plane::Luma,
                        );
                    }
                    cmv = cmv.map(chroma_sixteenths);
                    self.inter_scratch.refined[index][0] = cmv;
                } else {
                    for i in 0..=usize::from(bidirectional) {
                        let target = match target {
                            Target::Picture(offset) => {
                                let stride = self.planes.stride[0];
                                Target::Picture(
                                    offset
                                        + (y << UNIT_LOG2) as usize * stride
                                        + (x << UNIT_LOG2) as usize,
                                )
                            }
                            Target::Compound { .. } | Target::Refine { .. } => Target::Compound {
                                index: i,
                                offset: (y << UNIT_LOG2) as usize * intermediate_stride
                                    + (x << UNIT_LOG2) as usize,
                                stride: intermediate_stride,
                            },
                        };
                        let bounds = self.frame_bounds(Plane::Luma);
                        self.mc(
                            target,
                            Plane::Luma,
                            (step, step),
                            (column, row),
                            cmv[i],
                            references[i] as usize,
                            block.inter.filter,
                            bounds,
                        );
                    }
                    self.motion_store_mut().update_temporal(
                        (row >> 1, column >> 1),
                        (step >> 1, step >> 1),
                        references,
                        cmv,
                        swap,
                    );
                    if implicit {
                        let deferred = (bw4 * bh4 * 16) as usize;
                        have_mask |= self.implicit_mask(
                            (intermediate_stride, (column0, x), (row0, y)),
                            cmv,
                            (3, 3),
                            (step, step),
                            deferred,
                            Plane::Luma,
                        );
                    }
                    self.inter_scratch.refined[index][0] = cmv;
                }
            }
        }
        if implicit && have_mask {
            Implicit::Ready
        } else {
            Implicit::Off
        }
    }

    /// Bilinear predictions of a sub-block's two references around their
    /// vectors, the refinement's search (sample offset) and, where the
    /// frame allows it, its optical flow; returns the refined vectors in
    /// sixteenth samples and the flow corrections.
    #[allow(clippy::type_complexity, reason = "the refinement's inputs")]
    fn refine_sub_block(
        &mut self,
        (row, column): (u32, u32),
        mut cmv: [Mv; 2],
        references: [Reference; 2],
        (step, corner, frame): (u32, [(i32, i32); 2], (i32, i32)),
        (refine, implicit_search): (bool, bool),
        (flow, threshold, d): (bool, u32, [i8; 2]),
    ) -> ([Mv; 2], Correction) {
        let side = step + REFINE_MARGIN_UNITS;
        let stride = (((side << UNIT_LOG2) as usize) + 63) & !63;
        let sub = (step << UNIT_LOG2) as i32;
        for i in 0..2 {
            let bounds = Self::refine_bounds(corner[i], (sub, sub), frame);
            let offset_mv = Mv {
                y: cmv[i].y - REFINE_OFFSET_EIGHTHS,
                x: cmv[i].x - REFINE_OFFSET_EIGHTHS,
            };
            self.mc_refine_window(
                i,
                stride,
                (side, side),
                (column, row),
                offset_mv,
                references[i] as usize,
                bounds,
            );
        }
        let (mut dy, mut dx) = (0, 0);
        if refine {
            (dy, dx) = kernels::sad_refine(
                (
                    &self.inter_scratch.refine[0],
                    &self.inter_scratch.refine[1],
                    stride,
                ),
                ((step << UNIT_LOG2) as usize, (step << UNIT_LOG2) as usize),
                implicit_search,
            );
            cmv[0].y += 8 * dy;
            cmv[1].y -= 8 * dy;
            cmv[0].x += 8 * dx;
            cmv[1].x -= 8 * dx;
        }
        let at = |dy: i32, dx: i32| {
            (REFINE_OFFSET_SAMPLES as i32 + dy) as usize * stride
                + (REFINE_OFFSET_SAMPLES as i32 + dx) as usize
        };
        let (first, second) = (at(dy, dx), at(-dy, -dx));
        let sad = if flow {
            kernels::sad8x8(
                (&self.inter_scratch.refine[0], first),
                (&self.inter_scratch.refine[1], second),
                stride,
            )
        } else {
            0
        };
        let correction = if flow && sad >= threshold {
            let mut regression = [Regression::default(); 4];
            kernels::flow_regression(
                &mut regression,
                (&self.inter_scratch.refine[0], first),
                (&self.inter_scratch.refine[1], second),
                stride,
                ((step << UNIT_LOG2) as usize, (step << UNIT_LOG2) as usize),
                8,
                d,
            );
            flow_correction(&regression[0], d)
        } else {
            Correction::default()
        };
        for (mv, delta) in cmv.iter_mut().zip(correction.deltas) {
            *mv = Mv {
                x: mv.x * 2 + delta.x,
                y: mv.y * 2 + delta.y,
            };
        }
        (cmv, correction)
    }

    /// A bilinear prediction into refinement buffer `index`.
    #[allow(clippy::too_many_arguments, reason = "the mc call's parts")]
    fn mc_refine_window(
        &mut self,
        index: usize,
        stride: usize,
        (w4, h4): (u32, u32),
        position: (u32, u32),
        mv: Mv,
        reference: usize,
        bounds: Bounds,
    ) {
        self.mc(
            Target::Refine { index, stride },
            Plane::Luma,
            (w4, h4),
            position,
            mv,
            reference,
            filter_mode::BILINEAR,
            bounds,
        );
    }

    /// A compound block refined by optical flow or MV refinement: luma
    /// into the intermediates per sub-block (dav2d `opfl_pred`). Returns
    /// the implicit mask's state.
    #[allow(clippy::too_many_lines, reason = "dav2d opfl_pred")]
    pub(super) fn optical_flow_prediction(
        &mut self,
        block: &InterPrediction,
        (bw4, bh4): (u32, u32),
        (w4, h4): (u32, u32),
    ) -> Implicit {
        let inter = &block.inter;
        let refine = inter.compound == CompoundType::Average && inter.refine_mv != 0;
        let flow = inter.mode >= InterMode::OpflNearNear;
        let (width, height) = (
            (self.geometry.columns << UNIT_LOG2) as i32,
            (self.geometry.rows << UNIT_LOG2) as i32,
        );
        let references = inter.references;
        let [r0, r1] = references.map(|r| r as usize);
        let implicit = self.sequence.inter.implicit_masked_blend && inter.weight == EQUAL_WEIGHT;
        if implicit {
            self.inter_scratch.mask[0] = DEFERRED_MASK;
        }
        let mut have_mask = false;
        let d = flow_weights(self, references);
        let flow_block = 2 - u32::from(block.size == BlockSize::B8x8);
        let swap = self.motion_store().reference_flip(r0, r1);
        let (row0, column0) = (self.row, self.column);
        let stride = (bw4 << UNIT_LOG2) as usize;
        let mv = block.mv;
        let (sh4, sw4) = (bh4.min(4), bw4.min(4));
        let mut top = [0, 1].map(|i| ((row0 << UNIT_LOG2) as i32) + (mv[i].y >> 3) - TAPS_BEFORE);
        let deferred = (bw4 * bh4 * 16) as usize;
        for y in (0..h4).step_by(sh4 as usize) {
            let mut left =
                [0, 1].map(|i| ((column0 << UNIT_LOG2) as i32) + (mv[i].x >> 3) - TAPS_BEFORE);
            if refine {
                for x in (0..w4).step_by(sw4 as usize) {
                    let (row, column) = (row0 + y, column0 + x);
                    let side_w = sw4 + REFINE_MARGIN_UNITS;
                    let side_h = sh4 + REFINE_MARGIN_UNITS;
                    let window_stride = (((side_w << UNIT_LOG2) as usize) + 63) & !63;
                    for i in 0..2 {
                        let bounds = Self::refine_bounds(
                            (left[i], top[i]),
                            ((sw4 << UNIT_LOG2) as i32, (sh4 << UNIT_LOG2) as i32),
                            (width, height),
                        );
                        let offset_mv = Mv {
                            y: mv[i].y - REFINE_OFFSET_EIGHTHS,
                            x: mv[i].x - REFINE_OFFSET_EIGHTHS,
                        };
                        self.mc_refine_window(
                            i,
                            window_stride,
                            (side_w, side_h),
                            (column, row),
                            offset_mv,
                            references[i] as usize,
                            bounds,
                        );
                    }
                    let (dy, dx) = kernels::sad_refine(
                        (
                            &self.inter_scratch.refine[0],
                            &self.inter_scratch.refine[1],
                            window_stride,
                        ),
                        ((sw4 << UNIT_LOG2) as usize, (sh4 << UNIT_LOG2) as usize),
                        inter.refine_mv == 2,
                    );
                    if flow {
                        let at = |dy: i32, dx: i32| {
                            (REFINE_OFFSET_SAMPLES as i32 + dy) as usize * window_stride
                                + (REFINE_OFFSET_SAMPLES as i32 + dx) as usize
                        };
                        let mut regression = [Regression::default(); 4];
                        kernels::flow_regression(
                            &mut regression,
                            (&self.inter_scratch.refine[0], at(dy, dx)),
                            (&self.inter_scratch.refine[1], at(-dy, -dx)),
                            window_stride,
                            ((sw4 << UNIT_LOG2) as usize, (sh4 << UNIT_LOG2) as usize),
                            (flow_block << UNIT_LOG2) as usize,
                            d,
                        );
                        let mut r = 0;
                        for by in (0..sh4).step_by(2) {
                            for bx in (0..sw4).step_by(2) {
                                let correction = flow_correction(&regression[r], d);
                                r += 1;
                                let mut sub_mv = [
                                    Mv {
                                        y: mv[0].y * 2 + correction.deltas[0].y + dy * 16,
                                        x: mv[0].x * 2 + correction.deltas[0].x + dx * 16,
                                    },
                                    Mv {
                                        y: mv[1].y * 2 + correction.deltas[1].y - dy * 16,
                                        x: mv[1].x * 2 + correction.deltas[1].x - dx * 16,
                                    },
                                ];
                                for i in 0..2 {
                                    let offset = ((y + by) << UNIT_LOG2) as usize * stride
                                        + ((x + bx) << UNIT_LOG2) as usize;
                                    let bounds = Self::refine_bounds(
                                        (left[i], top[i]),
                                        ((sw4 << UNIT_LOG2) as i32, (sh4 << UNIT_LOG2) as i32),
                                        (width, height),
                                    );
                                    self.mc_sixteenth(
                                        Target::Compound {
                                            index: i,
                                            offset,
                                            stride,
                                        },
                                        Plane::Luma,
                                        (flow_block, flow_block),
                                        (column0 + x + bx, row0 + y + by),
                                        sub_mv[i],
                                        references[i] as usize,
                                        inter.filter,
                                        bounds,
                                    );
                                }
                                let temporal = [
                                    to_eighths(sub_mv[0], correction.deltas[0]),
                                    to_eighths(sub_mv[1], correction.deltas[1]),
                                ];
                                let (r8, c8) = ((row0 + y + by) >> 1, (column0 + x + bx) >> 1);
                                self.motion_store_mut().update_temporal(
                                    (r8, c8),
                                    (1, 1),
                                    references,
                                    temporal,
                                    swap,
                                );
                                if implicit {
                                    have_mask |= self.implicit_mask(
                                        (stride, (column0, x + bx), (row0, y + by)),
                                        sub_mv,
                                        (4, 4),
                                        (2, 2),
                                        deferred,
                                        Plane::Luma,
                                    );
                                }
                                sub_mv = sub_mv.map(chroma_sixteenths);
                                let index = Self::refined_index(row0 + y + by, column0 + x + bx);
                                self.inter_scratch.refined[index][0] = sub_mv;
                            }
                        }
                    } else {
                        let sub_mv = [
                            Mv {
                                y: mv[0].y + dy * 8,
                                x: mv[0].x + dx * 8,
                            },
                            Mv {
                                y: mv[1].y - dy * 8,
                                x: mv[1].x - dx * 8,
                            },
                        ];
                        for i in 0..2 {
                            let offset =
                                (y << UNIT_LOG2) as usize * stride + (x << UNIT_LOG2) as usize;
                            let bounds = Self::refine_bounds(
                                (left[i], top[i]),
                                ((sw4 << UNIT_LOG2) as i32, (sh4 << UNIT_LOG2) as i32),
                                (width, height),
                            );
                            self.mc(
                                Target::Compound {
                                    index: i,
                                    offset,
                                    stride,
                                },
                                Plane::Luma,
                                (sw4, sh4),
                                (column, row),
                                sub_mv[i],
                                references[i] as usize,
                                inter.filter,
                                bounds,
                            );
                        }
                        self.motion_store_mut().update_temporal(
                            (row >> 1, column >> 1),
                            (sw4 >> 1, sh4 >> 1),
                            references,
                            sub_mv,
                            swap,
                        );
                        if implicit {
                            have_mask |= self.implicit_mask(
                                (stride, (column0, x), (row0, y)),
                                sub_mv,
                                (3, 3),
                                (sw4, sh4),
                                deferred,
                                Plane::Luma,
                            );
                        }
                        let index = Self::refined_index(row, column);
                        self.inter_scratch.refined[index][0] = sub_mv;
                    }
                    for value in &mut left {
                        *value += 16;
                    }
                }
            } else {
                have_mask |= self.flow_rows(
                    block,
                    (y, sh4),
                    (w4, stride),
                    (left, top),
                    (references, swap, d, flow_block),
                    implicit,
                );
            }
            for value in &mut top {
                *value += (sh4 << UNIT_LOG2) as i32;
            }
        }
        if implicit && have_mask {
            Implicit::Ready
        } else {
            Implicit::Off
        }
    }

    /// One row of `sh4` units of an optical-flow block without MV
    /// refinement: bilinear predictions of the whole row, the regression
    /// per flow block, the corrected predictions (dav2d `opfl_pred`'s
    /// `else` branch). Returns whether an implicit mask was built.
    #[allow(clippy::too_many_lines, reason = "dav2d opfl_pred's row")]
    fn flow_rows(
        &mut self,
        block: &InterPrediction,
        (y, sh4): (u32, u32),
        (w4, stride): (u32, usize),
        (left, top): ([i32; 2], [i32; 2]),
        (references, swap, d, flow_block): ([Reference; 2], bool, [i8; 2], u32),
        implicit: bool,
    ) -> bool {
        let (bw4, bh4) = block
            .size
            .units()
            .min(((stride >> UNIT_LOG2) as u32, u32::MAX));
        let _ = bh4;
        let (row0, column0) = (self.row, self.column);
        let (width, height) = (
            (self.geometry.columns << UNIT_LOG2) as i32,
            (self.geometry.rows << UNIT_LOG2) as i32,
        );
        let mv = block.mv;
        let window_stride = (((bw4 << UNIT_LOG2) as usize) + 63) & !63;
        for i in 0..2 {
            let bounds = self.frame_bounds(Plane::Luma);
            self.mc(
                Target::Refine {
                    index: i,
                    stride: window_stride,
                },
                Plane::Luma,
                (bw4, sh4),
                (column0, row0 + y),
                mv[i],
                references[i] as usize,
                filter_mode::BILINEAR,
                bounds,
            );
        }
        let blocks = ((bw4 * sh4) / (flow_block * flow_block)) as usize;
        let mut regression = vec![Regression::default(); blocks.max(1)];
        kernels::flow_regression(
            &mut regression,
            (&self.inter_scratch.refine[0], 0),
            (&self.inter_scratch.refine[1], 0),
            window_stride,
            ((bw4 << UNIT_LOG2) as usize, (sh4 << UNIT_LOG2) as usize),
            (flow_block << UNIT_LOG2) as usize,
            d,
        );
        let mut have_mask = false;
        let mut corrections = [Correction::default(); 4];
        let mut used = 0;
        let row_blocks = (bw4 >> u32::from(flow_block == 2)) as usize;
        let deferred = (bw4 * block.size.units().1 * 16) as usize;
        let mut line = 0;
        for by in (0..sh4).step_by(flow_block as usize) {
            for (bx, sums) in (0..w4)
                .step_by(flow_block as usize)
                .zip(&regression[line..])
            {
                let correction = flow_correction(sums, d);
                let first_only = flow_block == 1 && (bx != 0 || by != 0);
                let sub_mv = [
                    Mv {
                        y: mv[0].y * 2 + correction.deltas[0].y,
                        x: mv[0].x * 2 + correction.deltas[0].x,
                    },
                    Mv {
                        y: mv[1].y * 2 + correction.deltas[1].y,
                        x: mv[1].x * 2 + correction.deltas[1].x,
                    },
                ];
                for i in 0..2 {
                    let offset =
                        ((y + by) << UNIT_LOG2) as usize * stride + (bx << UNIT_LOG2) as usize;
                    let bounds = Self::refine_bounds(
                        (
                            left[i] + (bx << UNIT_LOG2) as i32,
                            top[i] + (by << UNIT_LOG2) as i32,
                        ),
                        (8, 8),
                        (width, height),
                    );
                    self.mc_sixteenth(
                        Target::Compound {
                            index: i,
                            offset,
                            stride,
                        },
                        Plane::Luma,
                        (flow_block, flow_block),
                        (column0 + bx, row0 + y + by),
                        sub_mv[i],
                        references[i] as usize,
                        block.inter.filter,
                        bounds,
                    );
                }
                if flow_block > 1 {
                    let temporal = [
                        to_eighths(sub_mv[0], correction.deltas[0]),
                        to_eighths(sub_mv[1], correction.deltas[1]),
                    ];
                    let (row, column) = (row0 + y + by, column0 + bx);
                    let (r8, c8) = (row >> 1, column >> 1);
                    self.motion_store_mut().update_temporal(
                        (r8, c8),
                        (1, 1),
                        references,
                        temporal,
                        swap,
                    );
                } else {
                    corrections[used] = correction;
                    used += 1;
                }
                if implicit {
                    have_mask |= self.implicit_mask(
                        (stride, (column0, bx), (row0, y + by)),
                        sub_mv,
                        (4, 4),
                        (flow_block, flow_block),
                        deferred,
                        Plane::Luma,
                    );
                }
                if !first_only {
                    let index = Self::refined_index(row0 + y + by, column0 + bx);
                    self.inter_scratch.refined[index][0] = sub_mv.map(chroma_sixteenths);
                }
            }
            line += row_blocks;
        }
        if flow_block == 1 {
            // An 8x8 block's four 4x4 flows store one averaged vector.
            let mut temporal = [Mv::default(); 2];
            for i in 0..2 {
                let sum =
                    |f: fn(Mv) -> i32| corrections.iter().map(|c| f(c.deltas[i])).sum::<i32>();
                let (sx, sy) = (sum(|m| m.x), sum(|m| m.y));
                temporal[i] = Mv {
                    x: (mv[i].x * 8 + sx + 3 + i32::from(sx > 0)) >> 3,
                    y: (mv[i].y * 8 + sy + 3 + i32::from(sy > 0)) >> 3,
                };
            }
            self.motion_store_mut().update_temporal(
                (row0 >> 1, column0 >> 1),
                (1, 1),
                references,
                temporal,
                swap,
            );
        }
        have_mask
    }

    /// A refined block's chroma plane `plane` from its refined vectors,
    /// per `refine_step` units, each in `flow_step` parts (dav2d
    /// `rmv_uvpred`); into the picture at `dst` for single-reference TIP,
    /// else the intermediates. Returns the implicit mask's state (U only).
    #[allow(clippy::too_many_lines, reason = "dav2d rmv_uvpred")]
    pub(super) fn refined_chroma(
        &mut self,
        block: &InterPrediction,
        plane: usize,
        dst: Option<usize>,
        (refine_step, flow_step): (u32, u32),
        (bw4, bh4): (u32, u32),
    ) -> Implicit {
        let tip = block.inter.references[0] == super::super::inter_record::TIP_FRAME;
        let references = if tip {
            self.frame.tip.references.map(|r| r as Reference)
        } else {
            block.inter.references
        };
        let bidirectional = block.inter.weight != TIP_SINGLE_WEIGHT;
        let implicit = plane == 1
            && self.sequence.inter.implicit_masked_blend
            && block.inter.weight == EQUAL_WEIGHT;
        if implicit {
            self.inter_scratch.mask[0] = DEFERRED_MASK;
        }
        let mut have_mask = false;
        let kind = Plane::Chroma(plane);
        let (width, height) = (
            ((self.geometry.columns << UNIT_LOG2) >> 1) as i32,
            ((self.geometry.rows << UNIT_LOG2) >> 1) as i32,
        );
        let (rw4, rh4) = (bw4.min(refine_step), bh4.min(refine_step));
        let (ow4, oh4) = (bw4.min(flow_step), bh4.min(flow_step));
        let h_taps = 2 + 2 * i32::from(rw4 > 2);
        let v_taps = 2 + 2 * i32::from(rh4 > 2);
        let (cbx, cby) = (self.chroma_column, self.chroma_row);
        let h4 = bh4.min(self.geometry.rows - cby);
        let w4 = bw4.min(self.geometry.columns - cbx);
        let stride = ((bw4 << UNIT_LOG2) >> 1) as usize;
        let picture_stride = self.planes.stride[plane];
        let deferred = ((bw4 * bh4 * 16) >> 2) as usize;
        for y in (0..h4).step_by(rh4 as usize) {
            for x in (0..w4).step_by(rw4 as usize) {
                let index = Self::refined_index(cby + y, cbx + x);
                let refined = self.inter_scratch.refined[index];
                if !bidirectional {
                    let offset = dst.unwrap_or(0)
                        + (((y << UNIT_LOG2) >> 1) as usize) * picture_stride
                        + ((x << UNIT_LOG2) >> 1) as usize;
                    let bounds = self.frame_bounds(kind);
                    self.mc(
                        Target::Picture(offset),
                        kind,
                        (rw4, rh4),
                        (cbx + x, cby + y),
                        refined[0][0],
                        references[0] as usize,
                        block.inter.filter,
                        bounds,
                    );
                    continue;
                }
                let mut bounds = [self.frame_bounds(kind); 2];
                for i in 0..2 {
                    let source = if tip { refined[1][i] } else { block.mv[i] };
                    let top = (((cby + y) << UNIT_LOG2) >> 1) as i32 + (source.y >> 4);
                    let left = (((cbx + x) << UNIT_LOG2) >> 1) as i32 + (source.x >> 4);
                    bounds[i] = Bounds {
                        bottom: (top + ((rh4 << UNIT_LOG2) >> 1) as i32 + v_taps).clamp(1, height),
                        right: (left + ((rw4 << UNIT_LOG2) >> 1) as i32 + h_taps).clamp(1, width),
                        top: (top + 1 - v_taps).clamp(0, height - 1),
                        left: (left + 1 - h_taps).clamp(0, width - 1),
                    };
                }
                for by in (0..rh4).step_by(oh4 as usize) {
                    for bx in (0..rw4).step_by(ow4 as usize) {
                        let index = Self::refined_index(cby + y + by, cbx + x + bx);
                        let sub = self.inter_scratch.refined[index][0];
                        let offset = (((y + by) << UNIT_LOG2) >> 1) as usize * stride
                            + (((x + bx) << UNIT_LOG2) >> 1) as usize;
                        for i in 0..2 {
                            self.mc_sixteenth(
                                Target::Compound {
                                    index: i,
                                    offset,
                                    stride,
                                },
                                kind,
                                (ow4 >> 1, oh4 >> 1),
                                ((cbx + x + bx) >> 1, (cby + y + by) >> 1),
                                sub[i],
                                references[i] as usize,
                                block.inter.filter,
                                bounds[i],
                            );
                        }
                        if implicit {
                            have_mask |= self.implicit_mask(
                                (
                                    stride,
                                    (cbx >> 1, chroma_units(x + bx)),
                                    (cby >> 1, chroma_units(y + by)),
                                ),
                                sub,
                                (4, 4),
                                (ow4 >> 1, oh4 >> 1),
                                deferred,
                                kind,
                            );
                        }
                    }
                }
            }
        }
        if implicit && have_mask {
            Implicit::Ready
        } else {
            Implicit::Off
        }
    }
}
