//! An inter block's final vectors and warp models from its syntax and the
//! candidate lists, recorded for later blocks and frames (dav2d
//! `decode.c` `recon_b`'s motion resolution with `derive_warpmv`,
//! `extend_warpmv` and `debug_warp_matrix`; `env.h` `mv_reduce_prec`,
//! `get_warpmv_2d`, `get_gmv_2d`; dav2d commit 237229e, BSD-2-Clause, see
//! `docs/licenses/dav2d`).
use super::Tile;
use super::inter_record::{CompoundType, InterMode, InterRecord, MotionMode, WARP_DELTA_ABSENT};
use super::motion::candidates::Candidates;
use super::motion::vector::{RefPair, TIP_REFERENCE};
use super::motion::{InterMotion, Mv, WedgeSides};
use super::record::BlockRecord;
use super::size::BlockSize;
use crate::general::warp::{
    BlockArea, IDENTITY, OFF_DIAGONAL_MAX, UNITY, WarpModel, global_vector, kind, model_vector,
};
use nc_common::decode::{DecodeError, Result};

/// The finest vector precision (eighth samples, dav2d `mv_prec` 6).
const EIGHTH_PRECISION: u8 = 6;
/// Residuals at this precision or coarser round their predictor first
/// (whole samples and coarser).
const ROUNDED_PRECISION_MAX: u8 = 3;
/// A warp delta parameter's step (Q16 / 64).
const WARP_DELTA_STEP_LOG2: u32 = 10;
/// Optical-flow refinement of every block (`opfl_refine_type` 2): the
/// refined compound modes list as their unrefined ones.
const OPTICAL_FLOW_ALL: u8 = 2;
/// dav2d numbers the modes; the list kind changes after `NEARMV_NEWMV`.
const OPTICAL_FLOW_MODE_SHIFT: i32 = 6;

/// An inter block's resolved motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedMotion {
    /// The final vectors (the second for compound blocks).
    pub mv: [Mv; 2],
    /// The warp model per reference of a warped block (`motion_mode`
    /// above inter-intra), when usable (its kind not [`kind::INVALID`]).
    pub warp: [Option<WarpModel>; 2],
}

impl Tile<'_, '_> {
    /// Resolve an inter block's vectors and warp models, trace them as
    /// dav2d does, and record the block in the banks and the motion store
    /// (dav2d `recon_b`'s inter motion resolution and splats). A skip-mode
    /// block takes its candidate's compound weight into `inter.weight`.
    /// `wedge_sides` is a wedge compound's per-8x8 split (dav2d
    /// `WEDGE_TMVP`), needed when the frame keeps a temporal field.
    pub(super) fn resolve_motion(
        &mut self,
        depth: usize,
        record: &BlockRecord,
        inter: &mut InterRecord,
        wedge_sides: Option<&[u8]>,
    ) -> Result<ResolvedMotion> {
        let size = record
            .luma
            .ok_or(DecodeError::Malformed("AV2 inter block without luma"))?;
        let refs = RefPair(inter.references);
        let mut resolved = ResolvedMotion {
            mv: inter.residual,
            warp: [None; 2],
        };
        if !refs.is_compound() {
            self.resolve_single(depth, size, inter, &mut resolved)?;
        } else if record.skip_mode {
            let candidates = self.candidates(refs, size, true, false)?;
            trace!(self, depth, "find_mv_refs({},{})", refs.get(0), refs.get(1));
            for (n, c) in candidates.list[..candidates.count].iter().enumerate() {
                trace!(
                    self,
                    depth + 1,
                    "mv[{n}/{}]: y={},x={},y2={},x2={},w={},cwp={}",
                    candidates.count,
                    c.mv[0].y,
                    c.mv[0].x,
                    c.mv[1].y,
                    c.mv[1].x,
                    c.weight,
                    c.cwp
                );
            }
            let candidate =
                candidates.list[usize::from(inter.drl[0]).min(candidates.list.len() - 1)];
            resolved.mv = candidate.mv;
            inter.weight = candidate.cwp;
        } else {
            self.resolve_compound(depth, size, inter, &mut resolved)?;
        }
        self.trace_final(depth, inter, &resolved);
        self.record_motion(record, size, inter, &resolved, wedge_sides)?;
        // Prediction uses only usable models.
        resolved.warp = resolved.warp.map(|w| w.filter(WarpModel::is_valid));
        Ok(resolved)
    }

    /// The motion store's candidate list for the current block.
    fn candidates(
        &self,
        refs: RefPair,
        size: BlockSize,
        skip_mode: bool,
        warps: bool,
    ) -> Result<Candidates> {
        let store = self.motion.as_ref().ok_or(DecodeError::Malformed(
            "AV2 inter block without a motion store",
        ))?;
        Ok(store.find(refs, size, skip_mode, (self.row, self.column), warps))
    }

    /// The vector a reference's global motion gives the block.
    fn global_motion_vector(&self, reference: i8, size: BlockSize) -> Mv {
        let Some(store) = &self.motion else {
            return Mv::default();
        };
        let frame = &store.frame;
        let (bw4, bh4) = size.units();
        let area = BlockArea {
            row: self.row,
            column: self.column,
            bw4,
            bh4,
        };
        frame
            .global_model(reference)
            .map_or(Mv::default(), |model| {
                global_vector(
                    model,
                    area,
                    (frame.columns, frame.rows),
                    frame.params.mv_precision,
                    frame.params.force_integer_mv,
                )
            })
    }

    /// A predictor plus a coded residual, the predictor first rounded to
    /// the residual's precision when coarse (dav2d `mv_reduce_prec`).
    fn add_residual(predictor: Mv, residual: Mv, precision: u8, adaptive: bool) -> Mv {
        let predictor = if !adaptive && precision <= ROUNDED_PRECISION_MAX {
            predictor.reduce_precision(precision)
        } else {
            predictor
        };
        Mv {
            y: predictor.y.wrapping_add(residual.y),
            x: predictor.x.wrapping_add(residual.x),
        }
    }

    /// A one-reference block's vector and warp model (dav2d `recon_b`'s
    /// `ref.ref[1] == -1` branch).
    fn resolve_single(
        &mut self,
        depth: usize,
        size: BlockSize,
        inter: &InterRecord,
        resolved: &mut ResolvedMotion,
    ) -> Result<()> {
        let reference = inter.references[0];
        if inter.mode == InterMode::Global {
            resolved.mv[0] = self.global_motion_vector(reference, size);
            return Ok(());
        }
        let want_warps = reference != TIP_REFERENCE && inter.mode > InterMode::New;
        let candidates = self.candidates(RefPair::single(reference), size, false, want_warps)?;
        self.trace_single_list(depth, reference, &candidates, want_warps);
        let warp_index = usize::from(inter.warp_reference);
        let warped = inter.motion_mode > MotionMode::InterIntra;
        if warped && want_warps && warp_index >= candidates.warp_count {
            return Err(DecodeError::Malformed("AV2 warp reference index"));
        }
        let base_warp = candidates
            .warps
            .get(warp_index)
            .copied()
            .unwrap_or(IDENTITY);
        let (bw4, bh4) = size.units();
        let drl = usize::from(inter.drl[0]).min(candidates.list.len() - 1);
        let mut mv = if inter.mode == InterMode::Warp {
            let store = self
                .motion
                .as_ref()
                .ok_or(DecodeError::Malformed("AV2 motion store"))?;
            let area = BlockArea {
                row: self.row,
                column: self.column,
                bw4,
                bh4,
            };
            let precision = if inter.warp_with_residual {
                inter.precision[0]
            } else {
                EIGHTH_PRECISION
            };
            model_vector(
                &base_warp,
                area,
                (store.frame.columns, store.frame.rows),
                precision,
            )
        } else {
            candidates.list[drl].mv[0]
        };
        if matches!(inter.mode, InterMode::New | InterMode::WarpNew)
            || (inter.mode == InterMode::Warp && inter.warp_with_residual)
        {
            mv = Self::add_residual(mv, inter.residual[0], inter.precision[0], inter.amvd);
        }
        resolved.mv[0] = mv;
        let position = (self.row, self.column);
        resolved.warp[0] = match inter.motion_mode {
            MotionMode::WarpDelta => Some(Self::warp_delta_model(
                &base_warp,
                inter.warp_delta,
                mv,
                size,
                position,
            )),
            MotionMode::WarpCausal => Some(self.derive_warp(size, reference, mv)),
            MotionMode::WarpExtend => {
                Some(self.extend_warp(size, reference, mv, &candidates.list[drl]))
            }
            _ => None,
        };
        Ok(())
    }
    /// The candidate list's trace lines (dav2d `recon_b`'s
    /// `find_mv_refs(%d,-1)` block).
    fn trace_single_list(
        &mut self,
        depth: usize,
        reference: i8,
        candidates: &Candidates,
        warps: bool,
    ) {
        trace!(self, depth, "find_mv_refs({reference},-1)");
        for (n, c) in candidates.list[..candidates.count].iter().enumerate() {
            trace!(
                self,
                depth + 1,
                "mv[{n}/{}]: y={},x={},w={},y_off={},x_off={}",
                candidates.count,
                c.mv[0].y,
                c.mv[0].x,
                c.weight,
                c.y_off,
                c.x_off
            );
        }
        if warps {
            for (n, w) in candidates.warps[..candidates.warp_count].iter().enumerate() {
                trace!(
                    self,
                    depth + 1,
                    "warp[{n}/{}]: {}, {}, {}, {}, {}, {}",
                    candidates.warp_count,
                    w[0],
                    w[1],
                    w[2],
                    w[3],
                    w[4],
                    w[5]
                );
            }
        }
    }

    /// A listed warp model with the coded deltas added to its 2x2 part
    /// (two-parameter deltas mirror a rotation and zoom), its translation
    /// through the block's vector (dav2d's `MM_WARP_DELTA`).
    fn warp_delta_model(
        base_warp: &[i32; 6],
        deltas: [i8; 4],
        mv: Mv,
        size: BlockSize,
        position: (u32, u32),
    ) -> WarpModel {
        let (bw4, bh4) = size.units();
        let mut model = WarpModel::of_matrix(kind::AFFINE, *base_warp);
        let m = &mut model.matrix;
        for (n, &delta) in deltas.iter().enumerate() {
            if delta == WARP_DELTA_ABSENT {
                break;
            }
            let base = if n == 0 || n == 3 { UNITY } else { 0 };
            m[2 + n] = if delta == 0 {
                base_warp[n + 2]
            } else {
                (base_warp[n + 2] + (i32::from(delta) << WARP_DELTA_STEP_LOG2))
                    .clamp(base - OFF_DIAGONAL_MAX, base + OFF_DIAGONAL_MAX)
            };
        }
        if deltas[2] == WARP_DELTA_ABSENT {
            m[5] = m[2];
            m[4] = -m[3];
        }
        model.set_translation((bw4, bh4), mv, position);
        model.finish();
        model
    }

    /// A compound block's vectors: from a pair list, or from each
    /// reference's own list (dav2d `recon_b`'s compound branch).
    fn resolve_compound(
        &mut self,
        depth: usize,
        size: BlockSize,
        inter: &InterRecord,
        resolved: &mut ResolvedMotion,
    ) -> Result<()> {
        let refs = RefPair(inter.references);
        if inter.mode == InterMode::GlobalGlobal {
            for n in 0..2 {
                resolved.mv[n] = self.global_motion_vector(refs.get(n), size);
            }
            return Ok(());
        }
        let refined_all =
            self.frame.optical_flow == OPTICAL_FLOW_ALL && inter.mode >= InterMode::OpflNearNear;
        let shift = if refined_all {
            OPTICAL_FLOW_MODE_SHIFT
        } else {
            0
        };
        let pair_list = inter.mode as i32 - shift > InterMode::NearNew as i32;
        let (mut candidates, counts) = if pair_list {
            let candidates = self.candidates(refs, size, false, false)?;
            let count = candidates.count;
            (candidates, [count; 2])
        } else if refs.get(0) == refs.get(1) {
            let mut candidates =
                self.candidates(RefPair::single(refs.get(0)), size, false, false)?;
            for c in &mut candidates.list {
                c.mv[1] = c.mv[0];
                c.weight = c.weight.wrapping_mul(0x101);
            }
            let count = candidates.count;
            (candidates, [count; 2])
        } else {
            let mut candidates =
                self.candidates(RefPair::single(refs.get(0)), size, false, false)?;
            let second = self.candidates(RefPair::single(refs.get(1)), size, false, false)?;
            for (c, s) in candidates.list.iter_mut().zip(&second.list) {
                c.mv[1] = s.mv[0];
                c.weight = (c.weight & 0xff) | s.weight.wrapping_shl(8);
            }
            let counts = [candidates.count, second.count];
            (candidates, counts)
        };
        candidates.count = counts[0];
        trace!(self, depth, "find_mv_refs({},{})", refs.get(0), refs.get(1));
        if pair_list {
            for (n, c) in candidates.list[..counts[0]].iter().enumerate() {
                trace!(
                    self,
                    depth + 1,
                    "mv[{n}/{}]: y={},x={},y2={},x2={},w={}",
                    counts[0],
                    c.mv[0].y,
                    c.mv[0].x,
                    c.mv[1].y,
                    c.mv[1].x,
                    c.weight
                );
            }
        } else {
            for (drl, &count) in counts.iter().enumerate() {
                for (n, c) in candidates.list[..count].iter().enumerate() {
                    trace!(
                        self,
                        depth + 1,
                        "mv[{drl}:{n}/{count}]: y={},x={},w={}",
                        c.mv[drl].y,
                        c.mv[drl].x,
                        (c.weight >> (8 * drl)) & 0xff
                    );
                }
            }
        }
        for (n, mode) in inter.mode.per_reference().into_iter().enumerate() {
            let drl = usize::from(inter.drl[n]).min(candidates.list.len() - 1);
            let predictor = candidates.list[drl].mv[n];
            resolved.mv[n] = if mode == InterMode::New {
                Self::add_residual(predictor, inter.residual[n], inter.precision[n], inter.amvd)
            } else {
                predictor
            };
        }
        if inter.motion_mode == MotionMode::WarpCausal {
            for n in 0..2 {
                resolved.warp[n] = Some(self.derive_warp(size, refs.get(n), resolved.mv[n]));
            }
        }
        Ok(())
    }

    /// The final vectors' trace lines (dav2d `debug_warp_matrix` and
    /// "final 2dmv").
    fn trace_final(&mut self, depth: usize, inter: &InterRecord, resolved: &ResolvedMotion) {
        let compound = RefPair(inter.references).is_compound();
        if inter.motion_mode > MotionMode::InterIntra {
            for n in 0..=usize::from(compound) {
                let m = resolved.warp[n].map_or(IDENTITY, |w| w.matrix);
                let h = |v: i32| format!("{}{:x}", if v < 0 { '-' } else { ' ' }, v.unsigned_abs());
                trace!(
                    self,
                    depth,
                    "[ {}, {} | {}, {}, {}, {} ] mv=y:{},x:{}",
                    h(m[0]),
                    h(m[1]),
                    h(m[2]),
                    h(m[3]),
                    h(m[4]),
                    h(m[5]),
                    resolved.mv[n].y,
                    resolved.mv[n].x
                );
            }
        } else if compound {
            let [a, b] = resolved.mv;
            trace!(
                self,
                depth, "final 2dmv: y={},x={} | y={},x={}", a.y, a.x, b.y, b.x
            );
        } else {
            let a = resolved.mv[0];
            trace!(self, depth, "final 2dmv: y={},x={}", a.y, a.x);
        }
    }

    /// Record the block: the MV bank, the warp bank, the store (dav2d
    /// `dav2d_refmvs_bank_add`, `dav2d_refmvs_warp_add`, the splats).
    fn record_motion(
        &mut self,
        record: &BlockRecord,
        size: BlockSize,
        inter: &InterRecord,
        resolved: &ResolvedMotion,
        wedge_sides: Option<&[u8]>,
    ) -> Result<()> {
        let refs = RefPair(inter.references);
        let compound = refs.is_compound();
        let Some(store) = &mut self.motion else {
            return Err(DecodeError::Malformed(
                "AV2 inter block without a motion store",
            ));
        };
        let cwp = if compound { inter.weight } else { 0 };
        store.add_to_bank(record.row, record.column, size, refs, resolved.mv, cwp);
        let warped = inter.motion_mode > MotionMode::InterIntra;
        if warped {
            let mut added = true;
            if let Some(model) = resolved.warp[0].filter(WarpModel::is_valid) {
                added = store.add_warp(&model.matrix, refs.get(0) as usize);
            }
            if added
                && compound
                && let Some(model) = resolved.warp[1].filter(WarpModel::is_valid)
            {
                store.add_warp(&model.matrix, refs.get(1) as usize);
            }
        }
        let wedge = if inter.compound == CompoundType::Wedge {
            match wedge_sides {
                Some(sides) => Some(WedgeSides {
                    sides,
                    sign: inter.wedge_sign,
                }),
                None if store.frame.field.is_some() => {
                    return Err(DecodeError::Unsupported("AV2 wedge compound motion field"));
                }
                None => None,
            }
        } else {
            None
        };
        let refined = inter.mode >= InterMode::OpflNearNear;
        let refine_mv = inter.refine_mv != 0 && inter.compound == CompoundType::Average;
        let matrix = |n: usize| resolved.warp[n].map_or(IDENTITY, |w| w.matrix);
        store.record_inter(&InterMotion {
            row: record.row,
            column: record.column,
            size,
            refs,
            mv: resolved.mv,
            global_mode: matches!(inter.mode, InterMode::Global | InterMode::GlobalGlobal),
            warps: warped.then(|| [matrix(0), matrix(1)]),
            cwp: inter.weight,
            wedge,
            keep_temporal: !refined || !refine_mv,
        });
        Ok(())
    }
}
