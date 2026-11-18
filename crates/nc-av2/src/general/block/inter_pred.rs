//! Inter prediction of a block (dav2d `bytefn(dav2d_recon_b)`'s inter
//! paths): single references (translation, global and local warps, TIP)
//! with block-adaptive weights or inter-intra, compound references
//! (averaged, weighted, wedge and difference masks, implicit masked
//! blending, optical flow and MV refinement), luma per 64x64 piece and
//! chroma per chroma block, including 4:2:0 chroma blocks that cover
//! several luma blocks.
//!
//! Ported from dav2d commit 237229e `src/recon_tmpl.c` (`recon_b`'s inter
//! branches); BSD-2-Clause, see `docs/licenses/dav2d`.
mod blend;
mod kernels;
mod mc;
mod refine;

use super::Tile;
use super::inter_record::{CompoundType, InterMode, InterRecord, MotionMode, Reference, TIP_FRAME};
use super::motion::{MotionStore, Mv};
use super::size::BlockSize;
use crate::general::warp::WarpModel;
use blend::Implicit;
use mc::{Bounds, Target};

/// TIP blocks predicting from one of the TIP frame's references take this
/// weight (dav2d `cwp_idx` 16).
const TIP_SINGLE_WEIGHT: i8 = 16;
/// The unweighted average (dav2d `cwp_idx` 8).
const EQUAL_WEIGHT: i8 = 8;
/// The largest prediction side.
const MAX_SIDE: usize = 64;
/// Luma samples per 4x4 unit, as a shift.
const UNIT_LOG2: u32 = 2;

/// The resolved motion of an inter block, as the prediction reads it.
#[derive(Debug, Clone, Copy)]
pub struct InterPrediction {
    /// The block's syntax.
    pub inter: InterRecord,
    /// Its final vectors.
    pub mv: [Mv; 2],
    /// Its warp models, where valid (local warps).
    pub warp: [Option<WarpModel>; 2],
    /// The whole block's size (the prediction runs per 64x64 piece).
    pub size: BlockSize,
}

impl InterPrediction {
    /// A single reference (not counting a TIP block of both TIP
    /// references).
    fn single(&self) -> bool {
        self.inter.references[1] < 0
            && (self.inter.references[0] != TIP_FRAME || self.inter.weight == TIP_SINGLE_WEIGHT)
    }
}

/// The intermediates of compound prediction, the masks and the scratch
/// of refinement (dav2d `scratch.compinter`, `seg_mask`, `p`,
/// `interintra`; `t->rmv`).
#[derive(Debug, Clone)]
pub struct InterScratch {
    /// Copy dispatch is chosen once per decoder workspace.
    pub(super) copies: nc_dsp::BlockCopy,
    /// The two predictions before blending.
    pub(super) compound: [Vec<i16>; 2],
    /// A difference or implicit mask, per luma piece and chroma.
    pub(super) mask: Vec<u8>,
    /// Bilinear predictions refinement searches over.
    pub(super) refine: [Vec<u8>; 2],
    /// An inter-intra block's intra prediction.
    pub(super) intra: Vec<u8>,
    /// Reference windows of predictions reaching past the frame.
    pub(super) window: Vec<u8>,
    /// Refined vectors per 8x8 of a 128x128 area, luma and chroma (dav2d
    /// `t->rmv`).
    pub(super) refined: Vec<[[Mv; 2]; 2]>,
    /// The block-adaptive weights of a block's first piece, per plane
    /// (dav2d `t->pb.bawp`).
    pub(super) bawp: [(i32, i32); 3],
    /// Each inter block's first reference, vector, size and filter at
    /// its top-left unit of a 16x16-unit area, for 4:2:0 chroma blocks
    /// covering several luma blocks (dav2d `f->frame_thread.b`).
    pub(super) covered: Vec<Covered>,
}

/// What a chroma block covering several luma blocks predicts each of
/// them from.
#[derive(Debug, Clone, Copy)]
pub struct Covered {
    reference: Reference,
    mv: Mv,
    size: BlockSize,
    filter: u8,
}

/// Units a side of the covered-block area.
const COVERED_SIDE: u32 = 16;

/// Refined vectors are kept per 8x8 over this many 8x8 units a side.
const REFINED_SIDE: usize = 16;
/// Refinement searches cover a block and two units beyond it.
const REFINE_SIDE: usize = (MAX_SIDE + 16) * (MAX_SIDE + 16);
/// A window of a prediction's reference: the block plus the filter taps.
const WINDOW_SIDE: usize = MAX_SIDE * 2 + 16;

impl InterScratch {
    /// Allocate tile inter-prediction scratch using pre-resolved `copies`.
    pub(super) fn new(copies: nc_dsp::BlockCopy) -> Self {
        Self {
            copies,
            compound: [vec![0; MAX_SIDE * MAX_SIDE], vec![0; MAX_SIDE * MAX_SIDE]],
            mask: vec![0; MAX_SIDE * MAX_SIDE],
            refine: [vec![0; REFINE_SIDE], vec![0; REFINE_SIDE]],
            intra: vec![0; MAX_SIDE * MAX_SIDE],
            window: vec![0; WINDOW_SIDE * WINDOW_SIDE],
            refined: vec![[[Mv::default(); 2]; 2]; REFINED_SIDE * REFINED_SIDE],
            bawp: [(0, 0); 3],
            covered: vec![
                Covered {
                    reference: 0,
                    mv: Mv::default(),
                    size: BlockSize::B4x4,
                    filter: 0,
                };
                (COVERED_SIDE * COVERED_SIDE) as usize
            ],
        }
    }
}

impl Default for InterScratch {
    fn default() -> Self {
        Self::new(nc_dsp::Dsp::new().block_copies())
    }
}

/// Which plane type a prediction is of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plane {
    Luma,
    Chroma(usize),
}

impl Plane {
    fn index(self) -> usize {
        match self {
            Self::Luma => 0,
            Self::Chroma(plane) => plane,
        }
    }

    /// The subsampling shift (4:2:0: both directions).
    fn shift(self) -> u32 {
        u32::from(self != Self::Luma)
    }
}

impl Tile<'_, '_> {
    /// Predict a luma piece of `size` at the current position.
    pub(super) fn predict_inter_luma(&mut self, block: &InterPrediction, size: BlockSize) {
        let (bw4, bh4) = size.units();
        let w4 = bw4.min(self.geometry.columns - self.column);
        let h4 = bh4.min(self.geometry.rows - self.row);
        let inter = &block.inter;
        let position = (self.column, self.row);
        let target = Target::Picture(self.planes.offset(0, self.row, self.column));
        if block.single() {
            let reference = inter.references[0];
            if reference == TIP_FRAME {
                self.tip_prediction(block, target, (bw4, bh4), (w4, h4));
            } else if let Some(model) = self.warp_model(block, 0, (bw4, bh4)) {
                self.warp_affine(
                    target,
                    Plane::Luma,
                    size,
                    position,
                    reference as usize,
                    &model,
                );
            } else {
                let bounds = self.frame_bounds(Plane::Luma);
                self.mc(
                    target,
                    Plane::Luma,
                    (bw4, bh4),
                    position,
                    block.mv[0],
                    reference as usize,
                    inter.filter,
                    bounds,
                );
            }
            if inter.bawp[0] != 0 {
                self.bawp(block, Plane::Luma, (bw4, bh4), (w4, h4), inter.bawp[0]);
            } else if inter.motion_mode == MotionMode::InterIntra || inter.warp_inter_intra {
                self.blend_inter_intra(block, Plane::Luma, (bw4, bh4), position);
            }
            return;
        }
        let stride = (bw4 << UNIT_LOG2) as usize;
        let implicit = if inter.references[0] == TIP_FRAME {
            self.tip_prediction(
                block,
                Target::Compound {
                    index: 0,
                    offset: 0,
                    stride,
                },
                (bw4, bh4),
                (w4, h4),
            )
        } else if inter.mode >= InterMode::OpflNearNear
            || (inter.refine_mv != 0 && inter.compound == CompoundType::Average)
        {
            self.optical_flow_prediction(block, (bw4, bh4), (w4, h4))
        } else {
            let implicit = self.implicit_state(block);
            for i in 0..2 {
                let target = Target::Compound {
                    index: i,
                    offset: 0,
                    stride,
                };
                let reference = inter.references[i] as usize;
                if let Some(model) = self.compound_warp_model(block, i, (bw4, bh4)) {
                    self.warp_affine(target, Plane::Luma, size, position, reference, &model);
                } else {
                    let bounds = self.frame_bounds(Plane::Luma);
                    self.mc(
                        target,
                        Plane::Luma,
                        (bw4, bh4),
                        position,
                        block.mv[i],
                        reference,
                        inter.filter,
                        bounds,
                    );
                }
            }
            implicit
        };
        let pixels = (stride, (bh4 << UNIT_LOG2) as usize);
        let offset = self.planes.offset(0, self.row, self.column);
        self.blend_compound(block, Plane::Luma, offset, pixels, size, implicit);
    }

    /// Predict a chroma block of `chroma` (chroma units of the luma
    /// grid) at the current chroma position.
    pub(super) fn predict_inter_chroma(
        &mut self,
        block: &InterPrediction,
        chroma: BlockSize,
        luma: Option<BlockSize>,
    ) {
        let (cbw4, cbh4) = chroma.units();
        let position = (self.chroma_column, self.chroma_row);
        let (bw4, bh4) = block.size.units();
        if luma != Some(chroma) && bw4.min(bh4) < 16 {
            self.predict_covered_chroma(chroma);
            return;
        }
        let inter = &block.inter;
        // V blends with U's implicit mask (dav2d `bacpu`).
        let mut u_implicit = Implicit::Off;
        for plane in 1..=2 {
            let offset = self
                .planes
                .offset(plane, self.chroma_row >> 1, self.chroma_column >> 1);
            let target = Target::Picture(offset);
            let kind = Plane::Chroma(plane);
            if block.single() {
                let reference = inter.references[0];
                if reference == TIP_FRAME {
                    let step = self.tip_step(block);
                    self.refined_chroma(block, plane, Some(offset), (step, step), (cbw4, cbh4));
                } else if let Some(model) = self.warp_model(block, 0, (bw4, bh4)) {
                    self.warp_affine(target, kind, chroma, position, reference as usize, &model);
                } else {
                    let bounds = self.frame_bounds(kind);
                    self.mc(
                        target,
                        kind,
                        (cbw4, cbh4),
                        position,
                        block.mv[0],
                        reference as usize,
                        inter.filter,
                        bounds,
                    );
                }
                let (cw4, ch4) = (
                    cbw4.min(self.geometry.columns - self.chroma_column),
                    cbh4.min(self.geometry.rows - self.chroma_row),
                );
                if inter.bawp[1] != 0 {
                    self.bawp(block, kind, (cbw4, cbh4), (cw4, ch4), 1);
                } else if inter.motion_mode == MotionMode::InterIntra || inter.warp_inter_intra {
                    self.blend_inter_intra(block, kind, (cbw4, cbh4), position);
                }
                continue;
            }
            let stride = ((cbw4 << UNIT_LOG2) >> 1) as usize;
            let implicit = if inter.references[0] == TIP_FRAME {
                let step = self.tip_step(block);
                self.refined_chroma(block, plane, None, (step, step), (cbw4, cbh4))
            } else if inter.mode >= InterMode::OpflNearNear
                || (inter.refine_mv != 0 && inter.compound == CompoundType::Average)
            {
                let refine =
                    u32::from(inter.compound == CompoundType::Average && inter.refine_mv != 0);
                let opfl = u32::from(inter.mode >= InterMode::OpflNearNear);
                self.refined_chroma(block, plane, None, (2 << refine, 4 >> opfl), (cbw4, cbh4))
            } else {
                let implicit = if plane == 1 {
                    self.implicit_state(block)
                } else {
                    Implicit::Off
                };
                for i in 0..2 {
                    let target = Target::Compound {
                        index: i,
                        offset: 0,
                        stride,
                    };
                    let reference = inter.references[i] as usize;
                    if let Some(model) = self.compound_warp_model(block, i, (bw4, bh4)) {
                        self.warp_affine(target, kind, chroma, position, reference, &model);
                    } else {
                        let bounds = self.frame_bounds(kind);
                        self.mc(
                            target,
                            kind,
                            (cbw4, cbh4),
                            position,
                            block.mv[i],
                            reference,
                            inter.filter,
                            bounds,
                        );
                    }
                }
                implicit
            };
            let implicit = if plane == 1 {
                u_implicit = implicit;
                implicit
            } else {
                u_implicit
            };
            let pixels = (stride, ((cbh4 << UNIT_LOG2) >> 1) as usize);
            self.blend_compound(block, kind, offset, pixels, chroma, implicit);
        }
    }

    /// Keep an inter block's motion for a chroma block covering it.
    pub(super) fn record_covered(&mut self, block: &InterPrediction) {
        let index = Self::covered_index(self.row, self.column);
        self.inter_scratch.covered[index] = Covered {
            reference: block.inter.references[0],
            mv: block.mv[0],
            size: block.size,
            filter: block.inter.filter,
        };
    }

    fn covered_index(row: u32, column: u32) -> usize {
        ((row & (COVERED_SIDE - 1)) * COVERED_SIDE + (column & (COVERED_SIDE - 1))) as usize
    }

    /// A chroma block covering several luma blocks: each of them predicts
    /// its part of both chroma planes (dav2d `recon_b`'s sub8x8 coding).
    fn predict_covered_chroma(&mut self, chroma: BlockSize) {
        let (cbw4, cbh4) = chroma.units();
        let (cbx, cby) = (self.chroma_column, self.chroma_row);
        let cw4 = cbw4.min(self.geometry.columns - cbx);
        let ch4 = cbh4.min(self.geometry.rows - cby);
        let mut done = [0_u32; COVERED_SIDE as usize];
        for y in 0..ch4 {
            for x in 0..cw4 {
                if done[y as usize] & (1 << x) != 0 {
                    continue;
                }
                let covered = self.inter_scratch.covered[Self::covered_index(cby + y, cbx + x)];
                let (w4, h4) = covered.size.units();
                for plane in 1..=2 {
                    let kind = Plane::Chroma(plane);
                    let stride = self.planes.stride[plane];
                    let offset = self.planes.offset(plane, cby >> 1, cbx >> 1)
                        + ((y << UNIT_LOG2) >> 1) as usize * stride
                        + ((x << UNIT_LOG2) >> 1) as usize;
                    let bounds = self.frame_bounds(kind);
                    self.mc(
                        Target::Picture(offset),
                        kind,
                        (w4, h4),
                        (cbx + x, cby + y),
                        covered.mv,
                        Self::reference_rank(covered.reference),
                        covered.filter,
                        bounds,
                    );
                }
                for row in &mut done[y as usize..((y + h4) as usize).min(COVERED_SIDE as usize)] {
                    *row |= ((1 << w4) - 1) << x;
                }
            }
        }
    }

    /// The motion store (inter frames always have one).
    fn motion_store(&self) -> &MotionStore {
        self.motion.as_ref().expect("inter frames keep motion")
    }

    /// See [`Tile::motion_store`].
    fn motion_store_mut(&mut self) -> &mut MotionStore {
        self.motion.as_mut().expect("inter frames keep motion")
    }

    /// The global warp of reference `rank`, where usable.
    fn global_warp(&self, rank: usize) -> Option<WarpModel> {
        self.motion_store().global_warp(rank)
    }

    /// The warp model a single-reference block predicts with: its local
    /// warp, or the global motion of a GLOBALMV block (dav2d
    /// `gmv_warp_allowed`); none for translation or whole-sample frames.
    fn warp_model(
        &self,
        block: &InterPrediction,
        index: usize,
        (bw4, bh4): (u32, u32),
    ) -> Option<WarpModel> {
        if self.frame.force_integer_mv {
            return None;
        }
        let inter = &block.inter;
        let reference = inter.references[index] as usize;
        if inter.mode == InterMode::Global
            && bw4.min(bh4) > 1
            && self.global_warp(reference).is_some()
        {
            return self.global_warp(reference);
        }
        if inter.motion_mode >= MotionMode::WarpCausal {
            return block.warp[index];
        }
        None
    }

    /// The warp model of one reference of a compound block.
    fn compound_warp_model(
        &self,
        block: &InterPrediction,
        index: usize,
        (bw4, bh4): (u32, u32),
    ) -> Option<WarpModel> {
        let inter = &block.inter;
        let reference = inter.references[index] as usize;
        if inter.mode == InterMode::GlobalGlobal
            && bw4.min(bh4) > 1
            && let Some(model) = self.global_warp(reference)
        {
            return Some(model);
        }
        if inter.motion_mode == MotionMode::WarpCausal {
            return block.warp[index];
        }
        None
    }

    /// Whether a compound block may blend by the implicit mask of its
    /// predictions' positions (dav2d `imp_msk_bld`, `bacp = 2`).
    fn implicit_state(&self, block: &InterPrediction) -> Implicit {
        if self.sequence.inter.implicit_masked_blend
            && block.inter.motion_mode != MotionMode::WarpCausal
            && block.inter.mode != InterMode::GlobalGlobal
        {
            Implicit::Pending
        } else {
            Implicit::Off
        }
    }

    /// A plane's prediction bounds: the 8-rounded frame.
    fn frame_bounds(&self, plane: Plane) -> Bounds {
        let shift = plane.shift();
        Bounds {
            left: 0,
            right: ((self.geometry.columns << UNIT_LOG2) >> shift) as i32,
            top: 0,
            bottom: ((self.geometry.rows << UNIT_LOG2) >> shift) as i32,
        }
    }

    /// The step (in 4x4 units) of a TIP block's motion field (dav2d
    /// `tip_pred`'s and `recon_b`'s `step`).
    fn tip_step(&self, block: &InterPrediction) -> u32 {
        let (bw4, bh4) = block.size.units();
        let refine = self.tip_refines();
        let large = if self.frame.tip.mode == crate::general::frame_header::TIP_FRAME_MODE {
            !refine
        } else {
            (!refine && bw4.min(bh4) >= 4) || block.size == BlockSize::B256x256
        };
        2 << u32::from(large)
    }

    /// Whether TIP predictions are refined (dav2d `tip_pred`'s first
    /// `opfl`).
    fn tip_refines(&self) -> bool {
        self.sequence.inter.tip_refine_mv
            && (self.frame.tip.mode == 1
                || self.frame.tip.filter == crate::general::frame_header::filter_mode::SHARP)
    }

    /// A reference rank as a slot of the frame's references.
    fn reference_rank(reference: Reference) -> usize {
        reference as usize
    }
}
