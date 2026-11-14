//! HEVC's random syntax choices for the syntax-coverage stress test
//! (`debug.random_decisions`, see `nc_encoder::DebugRng`).
//!
//! The engine already randomises the coding quadtree, prediction modes and
//! vectors. This module randomises what the HEVC backend decides itself,
//! always among legal options:
//!
//! - per slice: the deblocking override (off, or random beta/tc offsets)
//!   and `slice_sao_luma_flag` / `slice_sao_chroma_flag`;
//! - per CTB: SAO merge left/up, or explicit off / edge / band parameters
//!   with random classes, offsets and band positions;
//! - intra CUs: any luma mode, any `intra_chroma_pred_mode`, `PART_NxN`
//!   with four luma modes on 8x8 CUs, and a random transform tree down to
//!   4x4 luma blocks (DST);
//! - inter CUs: merge (any matching `merge_idx`) or AMVP (either `mvp_idx`)
//!   for the engine's vector, a random transform tree, and residual dropped
//!   per plane and transform unit (an all-dropped merge CU becomes
//!   `cu_skip`, an AMVP one codes `rqt_root_cbf` 0).
//!
//! To reach those trees and overrides, [`HevcBackend::enable_random_syntax`]
//! also widens the parameter sets: the largest transform hierarchy depths
//! the subset decodes and `deblocking_filter_override_enabled_flag`. Real
//! encodes never construct [`RandomSyntax`].

use nc_common::{Picture, PlaneIndex};
use nc_encoder::{Block, DebugRng, Mv, PredictionMode};

use super::{
    HevcBackend, Predicted, Quantisation, TuPositions, copy_square, plane_rect, transform_units,
};
use crate::grid::{Cell, quadrant};
use crate::headers::{LOG2_CTB, LOG2_MIN_CB, LOG2_MIN_TB, SliceDeblocking};
use crate::predict::{self, CHROMA_DM, ModeCode};
use crate::sao::{self, BANDS, ComponentSao, CtbSao, MAX_OFFSET, PictureSao, SaoKind, Signal};
use crate::scan::ScanKind;
use crate::syntax::{CuKind, IntraLuma, MAX_TUS, PlaneResidual, TransformUnit, TransformUnits};

/// `max_transform_hierarchy_depth_inter` / `_intra` of stress streams: the
/// largest the subset decodes (TUs down to 8x8 in a 32x32 CU, 16 per CU at
/// most; intra one deeper, down to 8x8 in a 64x64 CU or 4x4 in a 32x32).
const INTER_TRANSFORM_DEPTH: u8 = 2;
const INTRA_TRANSFORM_DEPTH: u8 = 3;
/// Smallest random transforms: 4x4 intra luma (chroma coded with the 8x8
/// parent after the fourth block); inter stays at 8x8, the encoder's
/// smallest (`inter_tree.rs`).
const MIN_INTRA_TU_LOG2: u8 = 2;
const MIN_INTER_TU_LOG2: u8 = 3;
/// z-order index of the 4x4 block carrying its parent's chroma.
const LAST_QUADRANT: usize = 3;
/// Chance (in 1/256) of an 8x8 intra CU being `PART_NxN`.
const NXN_CHANCE: u64 = 96;
/// Edge-offset classes and SAO categories.
const EDGE_CLASSES: u64 = 4;
const CATEGORIES: usize = 4;
/// `slice_deblocking_filter` offset range (`_div2`, §7.4.7.1).
const DEBLOCK_OFFSET_DIV2_MAX: i32 = 6;

/// Chance (in 1/256) of a slice overriding the PPS deblocking control, and
/// of an override switching the filter off.
const DEBLOCK_OVERRIDE_CHANCE: u64 = 160;
const DEBLOCK_DISABLE_CHANCE: u64 = 64;
/// Chance (in 1/256) of each slice SAO flag being on.
const SAO_COMPONENT_CHANCE: u64 = 192;
/// Chance (in 1/256) of a CTB merging its SAO parameters from the left or
/// above neighbour, and of a component's SAO being band rather than edge.
const SAO_MERGE_CHANCE: u64 = 64;
const SAO_BAND_CHANCE: u64 = 128;
/// Chance (in 1/256) of a component's SAO being off.
const SAO_OFF_CHANCE: u64 = 48;
/// Chance (in 1/256) of an intra CU taking one of its most probable modes
/// (`mpm_idx`) rather than any mode.
const MPM_CHANCE: u64 = 96;
/// Chance (in 1/256) of an explicit `intra_chroma_pred_mode` (0..=3)
/// rather than DM.
const EXPLICIT_CHROMA_CHANCE: u64 = 160;
/// Chance (in 1/256) of a transform node that may split doing so.
const TU_SPLIT_CHANCE: u64 = 96;
/// Chance (in 1/256) of one plane's residual of a transform unit being
/// dropped.
const DROP_CHANCE: u64 = 64;
/// Chance (in 1/256) of an inter CU whose vector is a merge candidate
/// being coded with AMVP anyway.
const AMVP_ANYWAY_CHANCE: u64 = 80;

/// Where random transform splits stop.
#[derive(Debug, Clone, Copy)]
struct SplitLimits {
    /// Largest transform: larger nodes split.
    max_log2: u8,
    /// Levels below the CU that may split.
    depth: u8,
    /// Smallest transform.
    min_log2: u8,
}

/// The random source of the syntax stress test.
#[derive(Debug, Clone)]
pub(super) struct RandomSyntax {
    rng: DebugRng,
}

impl RandomSyntax {
    /// The deblocking control of a slice.
    pub(super) fn slice_deblocking(&mut self) -> SliceDeblocking {
        if !self.rng.chance(DEBLOCK_OVERRIDE_CHANCE) {
            return SliceDeblocking::Inherited;
        }
        SliceDeblocking::Override {
            disabled: self.rng.chance(DEBLOCK_DISABLE_CHANCE),
            beta_offset_div2: self.rng.symmetric(DEBLOCK_OFFSET_DIV2_MAX),
            tc_offset_div2: self.rng.symmetric(DEBLOCK_OFFSET_DIV2_MAX),
        }
    }

    /// Random SAO parameters of one component with `planes` planes (1 for
    /// luma, 2 for chroma), or `None` for off.
    fn component_sao(&mut self, planes: usize) -> Option<ComponentSao> {
        if self.rng.chance(SAO_OFF_CHANCE) {
            return None;
        }
        let mut offsets = [[0i8; CATEGORIES]; 2];
        let kind = if self.rng.chance(SAO_BAND_CHANCE) {
            for plane in &mut offsets[..planes] {
                for offset in plane.iter_mut() {
                    *offset = self.rng.symmetric(MAX_OFFSET) as i8;
                }
            }
            SaoKind::Band([0, 1].map(|_| self.rng.below(u64::from(BANDS)) as u8))
        } else {
            for plane in &mut offsets[..planes] {
                // Categories 1 and 2 add, 3 and 4 subtract (§7.4.9.3).
                for (cat, offset) in plane.iter_mut().enumerate() {
                    let magnitude = self.rng.below(MAX_OFFSET as u64 + 1) as i8;
                    *offset = if cat < 2 { magnitude } else { -magnitude };
                }
            }
            SaoKind::Edge(self.rng.below(EDGE_CLASSES) as u8)
        };
        Some(ComponentSao { kind, offsets })
    }

    /// Random transform units of `block`: forced splits above `max_log2`,
    /// then random splits within `depth` levels of the CU, never below
    /// `min_log2` nor beyond [`MAX_TUS`] units.
    fn transform_units(
        &mut self,
        block: &Block,
        max_log2: u8,
        depth: u8,
        min_log2: u8,
    ) -> TuPositions {
        // The units the forced splits alone make, reserved up front so a
        // random split never takes the room a later forced split needs.
        let forced = transform_units(block, max_log2).as_slice().len();
        let mut extra = 0;
        let mut out = TuPositions::default();
        let limits = SplitLimits {
            max_log2,
            depth,
            min_log2,
        };
        self.split_node(block, 0, limits, forced, &mut extra, &mut out);
        out
    }

    fn split_node(
        &mut self,
        node: &Block,
        level: u8,
        limits: SplitLimits,
        forced: usize,
        extra: &mut usize,
        out: &mut TuPositions,
    ) {
        // A split adds three units.
        const ADDED: usize = 3;
        let split = node.log2_size() > limits.max_log2
            || (level < limits.depth
                && node.log2_size() > limits.min_log2
                && forced + *extra + ADDED <= MAX_TUS
                && self.rng.chance(TU_SPLIT_CHANCE)
                && {
                    *extra += ADDED;
                    true
                });
        if split {
            for child in node.quadrants() {
                self.split_node(&child, level + 1, limits, forced, extra, out);
            }
        } else {
            out.push((node.x, node.y, node.log2_size()));
        }
    }
}

impl HevcBackend {
    /// Turn the random syntax choices on, widening the parameter sets so
    /// they can be signalled.
    pub(super) fn enable_random_syntax(&mut self, rng: DebugRng) {
        self.info.inter_transform_depth = INTER_TRANSFORM_DEPTH;
        self.info.intra_transform_depth = INTRA_TRANSFORM_DEPTH;
        self.info.deblock_override = true;
        self.random = Some(RandomSyntax { rng });
    }

    fn random_syntax(&mut self) -> &mut RandomSyntax {
        self.random.as_mut().expect("random decisions enabled")
    }

    fn rng(&mut self) -> &mut DebugRng {
        &mut self.random_syntax().rng
    }

    /// The random replacement of `encode_intra_cu` / `encode_inter_cu`.
    pub(super) fn encode_random_cu(
        &mut self,
        block: &Block,
        mode: PredictionMode,
        source: &Picture,
        reference: Option<&Picture>,
        recon: &mut Picture,
    ) {
        match mode {
            PredictionMode::IntraDc => self.random_intra_cu(block, source, recon),
            PredictionMode::Inter(mv) => {
                let reference = reference.expect("inter blocks need a reference picture");
                self.random_inter_cu(block, mv, source, reference, recon);
            }
        }
    }

    /// A random luma mode for a prediction block with candidates `mpm`.
    fn random_luma_mode(&mut self, mpm: [u8; 3]) -> u8 {
        let rng = self.rng();
        if rng.chance(MPM_CHANCE) {
            mpm[rng.below(mpm.len() as u64) as usize]
        } else {
            rng.below(u64::from(predict::MODES)) as u8
        }
    }

    /// An intra CU with random luma and chroma modes, partition (`PART_NxN`
    /// for 8x8 CUs) and transform tree.
    fn random_intra_cu(&mut self, block: &Block, source: &Picture, recon: &mut Picture) {
        // Only DC avoids the above-right samples, dirty on intra refresh's
        // dirty edge (see `choose_intra_luma`); such CUs stay unsplit.
        let dirty_edge = self
            .refresh
            .as_ref()
            .is_some_and(|w| w.on_dirty_edge(block));
        let quarters =
            !dirty_edge && block.log2_size() == LOG2_MIN_CB as u8 && self.rng().chance(NXN_CHANCE);
        let mut modes = [predict::DC; 4];
        let mut codes = [ModeCode::Mpm(0); 4];
        let units = if quarters { 4 } else { 1 };
        let half = block.size() / 2;
        for unit in 0..units {
            let (x, y) = (block.x + (unit & 1) * half, block.y + (unit >> 1) * half);
            let mpm = self.most_probable_modes(x, y);
            let mode = if dirty_edge {
                predict::DC
            } else {
                self.random_luma_mode(mpm)
            };
            codes[unit] = ModeCode::of(mode, mpm);
            if quarters {
                modes[unit] = mode;
                self.grid.set_intra_cell(block.x, block.y, modes);
            } else {
                modes = [mode; 4];
            }
        }
        let chroma = if !dirty_edge && self.rng().chance(EXPLICIT_CHROMA_CHANCE) {
            self.rng().below(u64::from(CHROMA_DM)) as u8
        } else {
            CHROMA_DM
        };
        let chroma_mode = predict::chroma_mode(chroma, modes[0]);
        let (max_log2, depth) = (
            self.info.max_transform_log2,
            self.info.intra_transform_depth + u8::from(quarters),
        );
        let positions = if quarters {
            transform_units(block, LOG2_MIN_TB as u8)
        } else {
            self.random_syntax()
                .transform_units(block, max_log2, depth, MIN_INTRA_TU_LOG2)
        };
        let mut tus = TransformUnits::default();
        for &(x, y, log2) in positions.as_slice() {
            let luma_area = plane_rect(PlaneIndex::Y, x, y, log2);
            let luma_res = self.random_intra_block(
                PlaneIndex::Y,
                luma_area,
                modes[quadrant(x, y)],
                source,
                recon,
            );
            // 4x4 luma blocks: the 8x8 parent's chroma follows the fourth.
            let chroma_area = if log2 > LOG2_MIN_TB as u8 {
                Some((x, y, log2))
            } else {
                (quadrant(x, y) == LAST_QUADRANT).then(|| {
                    let unit = 1 << LOG2_MIN_TB;
                    (x - unit, y - unit, LOG2_MIN_TB as u8 + 1)
                })
            };
            let [cb, cr] = [PlaneIndex::U, PlaneIndex::V].map(|plane| {
                chroma_area.map_or(PlaneResidual::default(), |(cx, cy, clog2)| {
                    let area = plane_rect(plane, cx, cy, clog2);
                    self.random_intra_block(plane, area, chroma_mode, source, recon)
                })
            });
            tus.push(TransformUnit {
                x,
                y,
                log2,
                luma: luma_res,
                cb,
                cr,
            });
        }
        self.record_cu(
            block,
            Cell {
                intra: true,
                intra_modes: modes,
                ..Cell::default()
            },
            tus.as_slice(),
        );
        let luma = if quarters {
            IntraLuma::Quarters(codes)
        } else {
            IntraLuma::Whole(codes[0])
        };
        self.push_cu(block, CuKind::Intra { luma, chroma }, tus.as_slice());
    }

    /// Predict and code one intra block of `plane` (`(x, y, n)` in plane
    /// samples), dropping its residual at random.
    fn random_intra_block(
        &mut self,
        plane: PlaneIndex,
        (x, y, n): (usize, usize, usize),
        mode: u8,
        source: &Picture,
        recon: &mut Picture,
    ) -> PlaneResidual {
        let is_luma = plane == PlaneIndex::Y;
        let refs = self.intra_refs(recon, plane, x, y, n);
        let mut pred = [0u8; crate::transform::MAX_COEFFS];
        predict::predict(&self.dsp, &refs, mode, is_luma, &mut pred);
        let scan = ScanKind::for_intra(mode, n.trailing_zeros() as u8, is_luma);
        let residual = self.code_residual(
            plane,
            x,
            y,
            n,
            (&pred, n),
            source,
            recon,
            Predicted::Intra,
            scan,
            Quantisation::Final,
        );
        self.maybe_drop(residual, plane, (x, y, n), (&pred, n), recon)
    }

    /// An inter CU with vector `mv`, signalled by merge or AMVP at random,
    /// with a random transform tree and dropped residual.
    fn random_inter_cu(
        &mut self,
        block: &Block,
        mv: Mv,
        source: &Picture,
        reference: &Picture,
        recon: &mut Picture,
    ) {
        let predictors = self
            .current
            .filter(|c| (c.x, c.y, c.log2) == (block.x, block.y, block.log2_size()))
            .unwrap_or_else(|| self.predictors(block));
        self.predict_inter_cu(block, mv, reference);
        let (max_log2, depth) = (
            self.info.max_transform_log2,
            self.info.inter_transform_depth,
        );
        let positions =
            self.random_syntax()
                .transform_units(block, max_log2, depth, MIN_INTER_TU_LOG2);
        let mut tus = TransformUnits::default();
        for &(x, y, log2) in positions.as_slice() {
            let [luma, cb, cr] = PlaneIndex::ALL.map(|plane| {
                let (px, py, size) = plane_rect(plane, x, y, log2);
                let (cu_x, cu_y, cu_n) = plane_rect(plane, block.x, block.y, block.log2_size());
                let start = (py - cu_y) * cu_n + px - cu_x;
                // The prediction is copied out: `code_residual` and the drop
                // both read it while `self` is borrowed.
                let pred = self.inter_pred[plane as usize][start..].to_vec();
                let residual = self.code_residual(
                    plane,
                    px,
                    py,
                    size,
                    (&pred, cu_n),
                    source,
                    recon,
                    Predicted::Inter,
                    ScanKind::Diagonal,
                    Quantisation::Final,
                );
                self.maybe_drop(residual, plane, (px, py, size), (&pred, cu_n), recon)
            });
            tus.push(TransformUnit {
                x,
                y,
                log2,
                luma,
                cb,
                cr,
            });
        }
        let residual = tus.as_slice().iter().any(TransformUnit::coded);
        let merge = predictors.merge.as_slice();
        let merge_indices: Vec<usize> = (0..merge.len()).filter(|&i| merge[i] == mv).collect();
        let rng = self.rng();
        let kind = if !merge_indices.is_empty() && !rng.chance(AMVP_ANYWAY_CHANCE) {
            let merge_idx = merge_indices[rng.below(merge_indices.len() as u64) as usize];
            if residual {
                CuKind::Merge { merge_idx }
            } else {
                CuKind::Skip { merge_idx }
            }
        } else {
            let mvp_idx = rng.below(predictors.amvp.len() as u64) as usize;
            CuKind::Amvp {
                mv,
                pred: predictors.amvp[mvp_idx],
                mvp_idx,
            }
        };
        let skip = matches!(kind, CuKind::Skip { .. });
        self.record_cu(
            block,
            Cell {
                skip,
                mv,
                ..Cell::default()
            },
            tus.as_slice(),
        );
        self.push_cu(block, kind, tus.as_slice());
    }

    /// Drop a coded block's residual at random: the reconstruction becomes
    /// the prediction and the block is coded with `cbf` 0.
    fn maybe_drop(
        &mut self,
        residual: PlaneResidual,
        plane: PlaneIndex,
        (x, y, n): (usize, usize, usize),
        (pred, stride): (&[u8], usize),
        recon: &mut Picture,
    ) -> PlaneResidual {
        if !residual.coded || !self.rng().chance(DROP_CHANCE) {
            return residual;
        }
        copy_square(
            &mut recon.plane_mut(plane).view_mut(),
            (x, y),
            n,
            pred,
            stride,
        );
        PlaneResidual {
            coded: false,
            ..residual
        }
    }

    /// Random slice SAO flags and per-CTB parameters, applied to `recon`.
    pub(super) fn random_sao(&mut self, recon: &mut Picture) {
        let ctb = 1usize << LOG2_CTB;
        let (cols, rows) = (recon.width().div_ceil(ctb), recon.height().div_ceil(ctb));
        let rng = self.rng();
        let components = (
            rng.chance(SAO_COMPONENT_CHANCE),
            rng.chance(SAO_COMPONENT_CHANCE),
        );
        self.sao_components.luma = components.0;
        self.sao_components.chroma = components.1;
        if !(components.0 || components.1) {
            self.sao = None;
            return;
        }
        let random = self.random.as_mut().expect("random decisions enabled");
        let mut decision = PictureSao {
            ctbs: Vec::with_capacity(cols * rows),
            cols,
        };
        for index in 0..cols * rows {
            let (rx, ry) = (index % cols, index / cols);
            let entry = if rx > 0 && random.rng.chance(SAO_MERGE_CHANCE) {
                (decision.ctbs[index - 1].0, Signal::MergeLeft)
            } else if ry > 0 && random.rng.chance(SAO_MERGE_CHANCE) {
                (decision.ctbs[index - cols].0, Signal::MergeUp)
            } else {
                let params = CtbSao {
                    luma: components.0.then(|| random.component_sao(1)).flatten(),
                    chroma: components.1.then(|| random.component_sao(2)).flatten(),
                };
                (params, Signal::Explicit)
            };
            decision.ctbs.push(entry);
        }
        sao::apply(recon, &decision);
        self.sao = Some(decision);
    }
}
