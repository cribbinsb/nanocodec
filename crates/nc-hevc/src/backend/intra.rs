//! Intra CUs: the luma mode search, the intra transform tree and the
//! `PART_NxN` partition of 8x8 CUs (HEVC §7.3.8.5, §7.3.8.8, §8.4.2).
//!
//! Luma is decided first, then chroma is coded along the chosen transform
//! tree: HEVC predicts each plane only from its own reconstruction, so
//! coding all luma blocks before the chroma blocks reconstructs exactly what
//! the decoder's interleaved order does.
//!
//! - **Mode search**: SATD + √λ · mode bits over a coarse-to-fine shortlist
//!   ([`HevcBackend::choose_intra_mode`]); the shortlist is HEVC's own
//!   pruning of its 35 modes (the shared `IntraModeSet` lists AV1's smooth
//!   and Paeth modes, which HEVC lacks).
//! - **Transform tree** (`hevc.intra_tu_depth`): each node is coded whole,
//!   and, where its residual is unevenly spread
//!   (`hevc.intra_tu_split_spread`), also split into four; the cheaper by
//!   `SSE + λ · bits` (CABAC estimates, [`crate::rate`]) is kept. Smaller
//!   blocks predict from nearer reconstructed samples.
//! - **`PART_NxN`** (`hevc.intra_nxn`): an 8x8 CU whose 2Nx2N cost suggests
//!   detail (`hevc.intra_nxn_min_cost`) is also coded as four 4x4
//!   prediction units with their own modes (4x4 DST, §8.6.4.2), compared
//!   by the same cost.
//!
//! With every tool off this codes exactly what the single-mode, unsplit
//! intra CU always did.

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::BlockSize;
use nc_dsp::intra_prediction::IntraMode;
use nc_encoder::IntraModeSet;
use nc_encoder::{Block, CodecBackend, Mv, PredictionMode};

use super::rd::SavedSquare;
use super::{HevcBackend, InterIntraSearch, Predicted, Quantisation, plane_rect};
use crate::contexts::SliceKind;
use crate::grid::{Cell, quadrant};
use crate::headers::{LOG2_MIN_CB, LOG2_MIN_TB};
use crate::predict::{self, ModeCode};
use crate::scan::ScanKind;
use crate::syntax::{
    CuKind, IntraLuma, MAX_TUS, PlaneResidual, TransformUnit, TransformUnits, min_transform_log2,
    split_transform_flag_present,
};
use crate::transform::MAX_COEFFS;

/// z-order index of the fourth 4x4 block, which carries the chroma of its
/// 8x8 parent (§7.3.8.10, `blkIdx == 3`).
const LAST_QUADRANT: usize = 3;
/// Luma samples of an 8x8 CU.
const MIN_CU_SAMPLES: f32 = 64.0;
/// Smallest luma transform block whose mode choice the transform-cost
/// pricing (`hevc.intra_rd_candidates`) covers: 4x4 blocks take the DST,
/// which [`HevcBackend::luma_transform_cost`] does not model.
const LOG2_MIN_RD_TB: u8 = 3;

/// One luma transform block of an intra trial.
#[derive(Debug, Clone, Copy, Default)]
struct LumaLeaf {
    x: usize,
    y: usize,
    log2: u8,
    residual: PlaneResidual,
}

/// The luma blocks of a CU (no heap allocation): the first `len` of
/// `leaves`.
#[derive(Debug, Clone, Copy, Default)]
struct LumaLeaves {
    leaves: [LumaLeaf; MAX_TUS],
    len: usize,
}

impl LumaLeaves {
    fn push(&mut self, leaf: LumaLeaf) {
        self.leaves[self.len] = leaf;
        self.len += 1;
    }

    fn truncate(&mut self, len: usize) {
        self.len = len;
    }

    fn iter(&self) -> impl Iterator<Item = LumaLeaf> + '_ {
        self.leaves[..self.len].iter().copied()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut LumaLeaf> + '_ {
        self.leaves[..self.len].iter_mut()
    }
}

/// One way of coding an intra CU's luma, with its estimated cost.
#[derive(Debug, Clone, Copy)]
struct LumaChoice {
    luma: IntraLuma,
    /// Mode of each 4x4 quadrant (z order) for the grid.
    modes: [u8; 4],
    leaves: LumaLeaves,
    /// `SSE + λ · bits`; 0 when no alternative needed it.
    cost: f32,
    /// The leaves were coded in a trial, with the plain quantiser.
    trial: bool,
}

/// What a luma transform tree shares while it is coded.
#[derive(Debug, Clone, Copy)]
struct LumaTree {
    cu: Block,
    mode: u8,
    min_log2: u8,
    /// Optional splits may be tried.
    try_splits: bool,
    /// Costs are needed (some alternative is compared).
    costed: bool,
}

impl HevcBackend {
    /// Choose the luma mode of the `2^log2` prediction block at `(x, y)` by
    /// SATD + √λ·(mode bits): planar, DC, every fourth angle and the
    /// candidate list, then ±2 and ±1 around the best angle (a coarse-to-fine
    /// search in the manner of kvazaar's rough intra search). Larger blocks
    /// are searched on their first transform block. With a `seed` (the
    /// enclosing CU's mode, for `PART_NxN` units) the angle sweep is
    /// replaced by the seed: a 4x4 unit's best mode is mostly near its CU's,
    /// and the refinement still moves ±2/±1 from the best. Returns the mode
    /// and its cost. `ranking`, if given, receives the modes the full search
    /// scored: the chosen one, then the others by cost
    /// ([`CodecBackend::intra_alternatives`]); nothing from the coarse
    /// search.
    pub(super) fn choose_intra_mode(
        &self,
        (x, y, log2): (usize, usize, u8),
        source: &Picture,
        recon: &Picture,
        mpm: [u8; 3],
        seed: Option<u8>,
        ranking: Option<&mut Vec<(u8, f32)>>,
    ) -> (u8, f32) {
        let log2 = log2.min(self.info.max_transform_log2);
        let n = 1usize << log2;
        let refs = self.intra_refs(recon, PlaneIndex::Y, x, y, n);
        let size = BlockSize::new(n, n).expect("intra blocks are 4..32");
        let src = source.plane(PlaneIndex::Y).view();
        let src = src.sub_view(x, y, n, n).expect("block inside the picture");
        // Horizontal modes are predicted transposed; compare them against
        // the transposed source (SATD is transpose invariant). It is built
        // on the first horizontal mode only: the smooth P-frame set (planar,
        // DC) never needs it, and the scalar transpose of a 32x32 block was
        // the largest part of that path's cost (callgrind, HEVC Fast CBR,
        // docs/analysis/rd-mode-audit-2026-11-19.md).
        let mut src_t_buffer = [0u8; MAX_COEFFS];
        let mut src_t_built = false;
        let sqrt_lambda = self.frame_quantization.sqrt_lambda;
        let mut pred = [0u8; MAX_COEFFS];
        let search_refs = predict::SearchRefs::new(&refs);
        let mut cost_of = |mode: u8| {
            let transposed = search_refs.predict_oriented(&self.dsp, mode, &mut pred);
            let view = PlaneView::new(&pred, n, n, n).expect("prediction covers the block");
            if transposed && !src_t_built {
                nc_dsp::transpose_square(&src, &mut src_t_buffer);
                src_t_built = true;
            }
            let src_t = PlaneView::new(&src_t_buffer, n, n, n).expect("block buffer");
            let reference = if transposed { &src_t } else { &src };
            self.dsp.satd(size, reference, &view) as f32
                + sqrt_lambda * ModeCode::of(mode, mpm).bits()
        };
        let search = match self.slice {
            SliceKind::Predicted | SliceKind::Bi => self.inter_intra,
            SliceKind::Intra => self.key_intra,
        };
        if search != InterIntraSearch::Full {
            // P-frame intra mostly repairs brightness changes and occlusions,
            // where a smooth prediction does most of the work; the angular
            // sweep (about 16 SATD evaluations) is kept for key frames
            // unless `hevc.key_intra_coarse` (a Fast speed lever). The
            // engine's shared ranking keeps the cheapest.
            let coarse = [
                predict::PLANAR,
                predict::DC,
                predict::HORIZONTAL,
                predict::VERTICAL,
            ];
            let (mode, cost) = nc_encoder::rank_intra(coarse.into_iter().chain(mpm), 1, cost_of)[0];
            return (mode, cost);
        }
        let mut tried = [f32::INFINITY; predict::MODES as usize];
        let mut cost = |mode: u8, best: &mut (f32, u8)| {
            if tried[usize::from(mode)].is_finite() {
                return;
            }
            let c = cost_of(mode);
            tried[usize::from(mode)] = c;
            if c < best.0 {
                *best = (c, mode);
            }
        };
        let mut best = (f32::MAX, predict::DC);
        let sweep = (2..predict::MODES)
            .step_by(4)
            .filter(|_| seed.is_none())
            .chain(seed);
        for mode in [predict::PLANAR, predict::DC]
            .into_iter()
            .chain(sweep)
            .chain(mpm)
        {
            cost(mode, &mut best);
        }
        for step in [2, 1] {
            let centre = best.1;
            if centre >= 2 {
                for mode in [centre.saturating_sub(step).max(2), (centre + step).min(34)] {
                    cost(mode, &mut best);
                }
            }
        }
        let priced = self.intra_rd_candidates > 1 || self.intra_cut().is_active();
        let chosen = if priced && seed.is_none() && log2 >= LOG2_MIN_RD_TB {
            self.priced_intra_mode(&refs, (&src, n), mpm, &tried)
        } else {
            (best.1, best.0)
        };
        if let Some(ranking) = ranking {
            let mut others: Vec<(u8, f32)> = (0..predict::MODES)
                .map(|m| (m, tried[usize::from(m)]))
                .filter(|&(m, cost)| m != chosen.0 && cost.is_finite())
                .collect();
            others.sort_by(|a, b| a.1.total_cmp(&b.1));
            ranking.clear();
            ranking.push((chosen.0, tried[usize::from(chosen.0)]));
            ranking.extend(others);
        }
        chosen
    }

    /// The staged intra cut of this slice's intra CUs.
    fn intra_cut(&self) -> nc_encoder::IntraCut {
        self.intra_stage.cut(self.slice == SliceKind::Intra)
    }

    /// The luma mode the transform-cost pricing chooses among the modes the
    /// search `tried` (their SATD + √λ · mode bits; infinite where not
    /// tried): the `hevc.intra_rd_candidates` cheapest, or the staged
    /// decision's survivors where its cut is on ([`nc_encoder::IntraStage`];
    /// a single survivor is taken unpriced). Returns the mode and its SATD
    /// cost.
    fn priced_intra_mode(
        &self,
        refs: &predict::Refs,
        (src, n): (&PlaneView<'_>, usize),
        mpm: [u8; 3],
        tried: &[f32; predict::MODES as usize],
    ) -> (u8, f32) {
        let searched = (0..predict::MODES).filter(|&m| tried[usize::from(m)].is_finite());
        let cost = |m: u8| tried[usize::from(m)];
        let cut = self.intra_cut();
        let shortlist = if cut.is_active() {
            let all = nc_encoder::rank_intra(searched, usize::from(predict::MODES), cost);
            cut.survivors(all)
        } else {
            nc_encoder::rank_intra(searched, self.intra_rd_candidates, cost)
        };
        match shortlist.as_slice() {
            [only] if cut.is_active() => *only,
            _ => self.rd_intra_mode(refs, (src, n), mpm, &shortlist),
        }
    }

    /// The luma mode among the SATD-ranked `shortlist` (`hevc.intra_rd_candidates`)
    /// whose first transform block (`src`, side `n`) codes cheapest: each is
    /// predicted as coded, transformed, plainly quantised and priced by
    /// coefficient-domain error plus λ · (CABAC level bits + mode bits)
    /// ([`Self::luma_transform_cost`]). Why: SATD scores residual energy the
    /// quantiser removes; on VP9 keys pricing the best three this way gained
    /// 3 % all-intra BD-rate (docs/analysis/rd-mode-audit-2026-11-19.md).
    /// Returns the mode and its SATD cost (what the inter fallback compares).
    fn rd_intra_mode(
        &self,
        refs: &predict::Refs,
        (src, n): (&PlaneView<'_>, usize),
        mpm: [u8; 3],
        shortlist: &[(u8, f32)],
    ) -> (u8, f32) {
        let lambda = self.rd_lambda();
        let mut pred = [0u8; MAX_COEFFS];
        let mut best = (f32::INFINITY, shortlist[0]);
        for &(mode, satd) in shortlist {
            predict::predict(&self.dsp, refs, mode, true, &mut pred);
            let view = PlaneView::new(&pred, n, n, n).expect("prediction covers the block");
            let (distortion, bits) = self.luma_transform_cost(src, &view, mode);
            let cost =
                distortion as f32 + lambda * (bits + self.intra_mode_bits(ModeCode::of(mode, mpm)));
            if cost < best.0 {
                best = (cost, (mode, satd));
            }
        }
        best.1
    }

    /// Whether P-frame intra CU `block` keeps the DC prediction without a
    /// mode search (`hevc.inter_intra_dc_only`).
    fn inter_intra_dc_only(&self, block: &Block) -> bool {
        self.slice.is_inter()
            && matches!(self.inter_intra, InterIntraSearch::DcOnly { max_log2, .. } if block.log2_size() <= max_log2)
    }

    /// The inter vector, and its `reference`, that intra CU `block` beat
    /// only through the engine's directional estimate
    /// (`BlockDecision::intra_fallback`), where it costs less than intra on
    /// the reconstruction: the best luma mode's SATD + √λ·bits (as the mode
    /// search measures it, on the CU's first transform block) against the
    /// vector's on the same samples, as x264 compares them. A CU that would
    /// keep DC without a search falls back unconditionally: the estimate
    /// promised a direction it cannot use. `Err` codes intra, with the luma
    /// mode the comparison already searched (the search
    /// [`Self::encode_intra_cu`] would repeat on the same samples).
    pub(super) fn inter_fallback<'r>(
        &mut self,
        block: &Block,
        fallback: Option<Mv>,
        source: &Picture,
        reference: Option<&'r Picture>,
        recon: &Picture,
    ) -> Result<(Mv, &'r Picture), Option<u8>> {
        let (Some(mv), Some(reference)) = (fallback, reference) else {
            return Err(None);
        };
        if self.inter_intra_dc_only(block) {
            return Ok((mv, reference));
        }
        let mpm = self.most_probable_modes(block.x, block.y);
        let (intra_mode, intra_cost) = self.choose_intra_mode(
            (block.x, block.y, block.log2_size()),
            source,
            recon,
            mpm,
            None,
            None,
        );
        self.predict_inter_cu(block, mv, reference);
        let n = block.size().min(1 << self.info.max_transform_log2);
        let src = source.plane(PlaneIndex::Y).view();
        let src = src
            .sub_view(block.x, block.y, n, n)
            .expect("block inside the picture");
        let pred = PlaneView::new(&self.inter_pred[PlaneIndex::Y as usize], n, n, block.size())
            .expect("CU prediction");
        let size = BlockSize::new(n, n).expect("intra blocks are 4..32");
        let inter_cost = self.dsp.satd(size, &src, &pred) as f32
            + self.frame_quantization.sqrt_lambda
                * self.mode_bits(block, &PredictionMode::Inter(mv));
        if inter_cost < intra_cost {
            Ok((mv, reference))
        } else {
            Err(Some(intra_mode))
        }
    }

    /// An intra CU: luma (mode, transform tree, partition) is decided and
    /// reconstructed, then chroma (DM: the first unit's luma mode) is coded
    /// along the same tree. `searched` is the luma mode
    /// [`Self::choose_intra_mode`] already chose for the CU, if any;
    /// `chroma` its `intra_chroma_pred_mode` (`predict::CHROMA_DM` unless
    /// the leaf search tries another, `leaf_alternatives.rs`).
    pub(super) fn encode_intra_cu(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
        (searched, chroma): (Option<u8>, u8),
    ) {
        let mut choice = self.choose_intra_luma(block, source, recon, searched);
        if choice.trial && self.trellis_for(PlaneIndex::Y).is_some() {
            self.finalise_intra_luma(&mut choice, source, recon);
        }
        let chroma_mode = predict::chroma_mode(chroma, choice.modes[0]);
        let mut tus = TransformUnits::default();
        for leaf in choice.leaves.iter() {
            // 4x4 luma blocks: the 8x8 parent's chroma follows the fourth.
            let chroma_area = if leaf.log2 > LOG2_MIN_TB as u8 {
                Some((leaf.x, leaf.y, leaf.log2))
            } else {
                (quadrant(leaf.x, leaf.y) == LAST_QUADRANT).then(|| {
                    let parent = 1 << LOG2_MIN_TB;
                    (leaf.x - parent, leaf.y - parent, LOG2_MIN_TB as u8 + 1)
                })
            };
            let [cb, cr] = [PlaneIndex::U, PlaneIndex::V].map(|plane| {
                chroma_area.map_or(PlaneResidual::default(), |(x, y, log2)| {
                    self.code_intra_block(
                        plane,
                        plane_rect(plane, x, y, log2),
                        chroma_mode,
                        source,
                        recon,
                        Quantisation::Final,
                    )
                })
            });
            tus.push(TransformUnit {
                x: leaf.x,
                y: leaf.y,
                log2: leaf.log2,
                luma: leaf.residual,
                cb,
                cr,
            });
        }
        self.record_cu(
            block,
            Cell {
                intra: true,
                intra_modes: choice.modes,
                ..Cell::default()
            },
            tus.as_slice(),
        );
        self.push_cu(
            block,
            CuKind::Intra {
                luma: choice.luma,
                chroma,
            },
            tus.as_slice(),
        );
    }

    /// Predict (mode `mode`) and code one intra block of `plane`, `(x, y, n)`
    /// in plane samples.
    fn code_intra_block(
        &self,
        plane: PlaneIndex,
        (x, y, n): (usize, usize, usize),
        mode: u8,
        source: &Picture,
        recon: &mut Picture,
        quantisation: Quantisation,
    ) -> PlaneResidual {
        let luma = plane == PlaneIndex::Y;
        let mut pred = [0u8; MAX_COEFFS];
        self.intra_prediction(plane, (x, y, n), mode, recon, &mut pred);
        let scan = ScanKind::for_intra(mode, n.trailing_zeros() as u8, luma);
        self.code_residual(
            plane,
            x,
            y,
            n,
            (&pred, n),
            source,
            recon,
            Predicted::Intra,
            scan,
            quantisation,
        )
    }

    /// The intra prediction (mode `mode`) of the `n`-square block of `plane`
    /// at `(x, y)` into `pred`: the one the smooth choice handed over
    /// ([`Self::smooth_intra_mode`]) when it is that luma block's, else
    /// predicted from the gathered references.
    fn intra_prediction(
        &self,
        plane: PlaneIndex,
        (x, y, n): (usize, usize, usize),
        mode: u8,
        recon: &Picture,
        pred: &mut [u8; MAX_COEFFS],
    ) {
        let luma = plane == PlaneIndex::Y;
        {
            let mut scratch = self.scratch.borrow_mut();
            let handed = luma && scratch.smooth_prediction == Some(((x, y, n), mode));
            scratch.smooth_prediction = None;
            if handed {
                pred[..n * n].copy_from_slice(&scratch.smooth_samples[..n * n]);
                return;
            }
        }
        let refs = self.intra_refs(recon, plane, x, y, n);
        predict::predict(&self.dsp, &refs, mode, luma, pred);
    }

    /// Re-code the luma leaves of `choice`, decided in a trial with the plain
    /// quantiser, as final blocks (the trellis), in coding order: each
    /// leaf predicts from the reconstruction of the ones before it.
    fn finalise_intra_luma(&self, choice: &mut LumaChoice, source: &Picture, recon: &mut Picture) {
        let (luma, modes) = (choice.luma, choice.modes);
        for leaf in choice.leaves.iter_mut() {
            let mode = match luma {
                IntraLuma::Quarters(_) => modes[quadrant(leaf.x, leaf.y)],
                IntraLuma::Whole(_) => modes[0],
            };
            leaf.residual = self.code_intra_block(
                PlaneIndex::Y,
                (leaf.x, leaf.y, 1 << leaf.log2),
                mode,
                source,
                recon,
                Quantisation::Final,
            );
        }
        choice.trial = false;
    }

    /// Planar or DC for P-frame intra CU `block` (`hevc.inter_intra_smooth`),
    /// the cheaper by SATD + √λ · mode bits on its first transform block,
    /// as [`Self::choose_intra_mode`] ranks them; the winner's prediction is
    /// handed to that block's coding ([`Self::code_intra_block`]), so the
    /// choice costs one reference gather and two predictions in all.
    /// Measured in docs/analysis/rd-mode-audit-2026-11-19.md.
    fn smooth_intra_mode(
        &self,
        block: &Block,
        source: &Picture,
        recon: &Picture,
        mpm: [u8; 3],
    ) -> u8 {
        let n = block.size().min(1 << self.info.max_transform_log2);
        let refs = self.intra_refs(recon, PlaneIndex::Y, block.x, block.y, n);
        let size = BlockSize::new(n, n).expect("intra blocks are 4..32");
        let src = source.plane(PlaneIndex::Y).view();
        let src = src
            .sub_view(block.x, block.y, n, n)
            .expect("block inside the picture");
        let sqrt_lambda = self.frame_quantization.sqrt_lambda;
        let mut predictions = [[0u8; MAX_COEFFS]; 2];
        let mut costs = [0f32; 2];
        // The engine's smooth set, planar for its smooth mode.
        let set = IntraModeSet::Smooth.modes();
        let modes: [u8; 2] = std::array::from_fn(|i| match set[i] {
            IntraMode::Smooth => predict::PLANAR,
            _ => predict::DC,
        });
        for ((&mode, pred), cost) in modes.iter().zip(&mut predictions).zip(&mut costs) {
            predict::predict(&self.dsp, &refs, mode, true, pred);
            let view = PlaneView::new(&pred[..], n, n, n).expect("prediction covers the block");
            *cost = self.dsp.satd(size, &src, &view) as f32
                + sqrt_lambda * ModeCode::of(mode, mpm).bits();
        }
        // Ties keep DC, the smooth set's first mode.
        let best = usize::from(costs[1] < costs[0]);
        let mut scratch = self.scratch.borrow_mut();
        scratch.smooth_prediction = Some(((block.x, block.y, n), modes[best]));
        scratch.smooth_samples[..n * n].copy_from_slice(&predictions[best][..n * n]);
        modes[best]
    }

    /// Decide and reconstruct the luma of intra `block`.
    fn choose_intra_luma(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
        searched: Option<u8>,
    ) -> LumaChoice {
        // Planar and angular prediction (and reference smoothing) read the
        // above-right samples (2N wide in all, §8.4.4.2.2), dirty where they
        // reach past intra refresh's range; DC reads only the left and above
        // ones (§8.4.4.2.5). Such CUs keep one unsplit DC prediction.
        let dirty_edge = self
            .refresh
            .as_ref()
            .is_some_and(|w| w.reads_dirty_above_right(block));
        // With the coarse P-frame intra search, P-frame intra CUs (mostly
        // brightness repairs and occlusions) keep one unsplit 2Nx2N
        // prediction too: the trials there cost P frames more than the
        // detail they recover (crowd09 QP 42, intra depth 2: P frames
        // +10.9% instructions with trials, +2.0% without).
        let trials =
            !dirty_edge && (self.inter_intra == InterIntraSearch::Full || !self.slice.is_inter());
        let try_quarters = trials && self.tools.intra_nxn && block.log2_size() == LOG2_MIN_CB as u8;
        let try_splits = trials && self.info.intra_transform_depth > 0;
        let costed = try_quarters || try_splits;
        let mpm = self.most_probable_modes(block.x, block.y);
        let smooth_only = matches!(
            self.inter_intra,
            InterIntraSearch::DcOnly { smooth: true, .. }
        );
        let mode = if dirty_edge || (self.inter_intra_dc_only(block) && !smooth_only) {
            predict::DC
        } else if smooth_only && searched.is_none() && self.inter_intra_dc_only(block) {
            self.smooth_intra_mode(block, source, recon, mpm)
        } else if let Some(mode) = searched {
            mode
        } else {
            self.choose_intra_mode(
                (block.x, block.y, block.log2_size()),
                source,
                recon,
                mpm,
                None,
                None,
            )
            .0
        };
        let code = ModeCode::of(mode, mpm);
        let tree = LumaTree {
            cu: *block,
            mode,
            min_log2: min_transform_log2(
                block.log2_size(),
                self.info.intra_transform_depth,
                self.info.max_transform_log2,
            ),
            try_splits,
            costed,
        };
        let mut whole = LumaChoice {
            luma: IntraLuma::Whole(code),
            modes: [mode; 4],
            leaves: LumaLeaves::default(),
            cost: 0.0,
            trial: costed,
        };
        let tree_cost = self.code_luma_tree(&tree, *block, source, recon, &mut whole.leaves);
        if !costed {
            return whole;
        }
        let part_bits = if block.log2_size() == LOG2_MIN_CB as u8 {
            self.bin_bits(self.contexts.part_mode, true)
        } else {
            0.0
        };
        whole.cost = tree_cost + self.rd_lambda() * (part_bits + self.intra_mode_bits(code));
        // NxN gate: a 2Nx2N cost below `intra_nxn_min_cost` λ per sample
        // means the block is already cheap to code; four 4x4 units would
        // add mode bits for little gain.
        if !try_quarters
            || whole.cost < self.tools.intra_nxn_min_cost * self.rd_lambda() * MIN_CU_SAMPLES
        {
            return whole;
        }
        let saved = SavedSquare::save(recon, PlaneIndex::Y, (block.x, block.y, block.size()));
        let quarters = self.code_quarters(block, mode, source, recon);
        if quarters.cost < whole.cost {
            quarters
        } else {
            saved.restore(recon);
            whole
        }
    }

    /// Code the luma of 8x8 `block` as `PART_NxN`: four 4x4 prediction
    /// units in z order, each with its own mode (the later units' candidate
    /// lists see the earlier ones) and one 4x4 transform block.
    fn code_quarters(
        &mut self,
        block: &Block,
        seed: u8,
        source: &Picture,
        recon: &mut Picture,
    ) -> LumaChoice {
        let unit = 1usize << LOG2_MIN_TB;
        let mut modes = [0u8; 4];
        let mut codes = [ModeCode::Mpm(0); 4];
        let mut leaves = LumaLeaves::default();
        let mut bits = self.bin_bits(self.contexts.part_mode, false);
        let mut distortion = 0.0;
        // The grid's cell for this CU carries the decided units' modes for
        // the candidate lists; `record_cu` overwrites it with the final CU.
        for index in 0..4 {
            let (x, y) = (block.x + (index & 1) * unit, block.y + (index >> 1) * unit);
            let (left, above) = self.grid.neighbour_intra_modes(x, y);
            let mpm = predict::most_probable(left, above);
            let mode = self
                .choose_intra_mode(
                    (x, y, LOG2_MIN_TB as u8),
                    source,
                    recon,
                    mpm,
                    Some(seed),
                    None,
                )
                .0;
            modes[index] = mode;
            codes[index] = ModeCode::of(mode, mpm);
            self.grid.set_intra_cell(block.x, block.y, modes);
            let residual = self.code_intra_block(
                PlaneIndex::Y,
                (x, y, unit),
                mode,
                source,
                recon,
                Quantisation::Plain,
            );
            distortion +=
                self.reconstruction_sse(source, recon, PlaneIndex::Y, (x, y, unit)) as f32;
            bits += self.intra_mode_bits(codes[index])
                + self.bin_bits(self.contexts.cbf_luma[0], residual.coded)
                + self.residual_bits(residual, true);
            leaves.push(LumaLeaf {
                x,
                y,
                log2: LOG2_MIN_TB as u8,
                residual,
            });
        }
        LumaChoice {
            luma: IntraLuma::Quarters(codes),
            modes,
            leaves,
            cost: distortion + self.rd_lambda() * bits,
            trial: true,
        }
    }

    /// Code the luma transform tree of `tree.cu` below `node`, returning its
    /// cost (`SSE + λ · bits` when `tree.costed`, else 0). A node above the
    /// largest transform splits; a node that may split and whose residual
    /// is unevenly spread is coded both ways and the cheaper kept.
    fn code_luma_tree(
        &self,
        tree: &LumaTree,
        node: Block,
        source: &Picture,
        recon: &mut Picture,
        leaves: &mut LumaLeaves,
    ) -> f32 {
        let max_log2 = self.info.max_transform_log2;
        if node.log2_size() > max_log2 {
            return node
                .quadrants()
                .into_iter()
                .map(|child| self.code_luma_tree(tree, child, source, recon, leaves))
                .sum();
        }
        let depth = usize::from(tree.cu.log2_size() - node.log2_size());
        let n = node.size();
        let mut pred = [0u8; MAX_COEFFS];
        self.intra_prediction(
            PlaneIndex::Y,
            (node.x, node.y, n),
            tree.mode,
            recon,
            &mut pred,
        );
        let scan = ScanKind::for_intra(tree.mode, node.log2_size(), true);
        let flag_present =
            split_transform_flag_present(node.log2_size(), depth, tree.min_log2, max_log2, false);
        // Split gate: an evenly spread residual gains little from four
        // smaller transforms predicted from nearer samples.
        let try_split = tree.try_splits
            && flag_present
            && self
                .residual_spread(
                    &source
                        .plane(PlaneIndex::Y)
                        .view()
                        .sub_view(node.x, node.y, n, n)
                        .expect("block inside the picture"),
                    &PlaneView::new(&pred, n, n, n).expect("prediction covers the block"),
                    n,
                )
                .0
                >= self.tools.intra_tu_split_spread;
        let residual = self.code_residual(
            PlaneIndex::Y,
            node.x,
            node.y,
            n,
            (&pred, n),
            source,
            recon,
            Predicted::Intra,
            scan,
            if tree.costed {
                Quantisation::Plain
            } else {
                Quantisation::Final
            },
        );
        let leaf = LumaLeaf {
            x: node.x,
            y: node.y,
            log2: node.log2_size(),
            residual,
        };
        let leaf_cost = if tree.costed {
            let bits = self.bin_bits(
                self.contexts.cbf_luma[usize::from(depth == 0)],
                residual.coded,
            ) + self.residual_bits(residual, true)
                + self.split_transform_bits(node.log2_size(), flag_present, false);
            self.reconstruction_sse(source, recon, PlaneIndex::Y, (node.x, node.y, n)) as f32
                + self.rd_lambda() * bits
        } else {
            0.0
        };
        // Detail gate: a leaf already cheap per sample (below
        // `intra_tu_split_min_cost` λ) gains little from four smaller
        // predictions.
        let detailed =
            leaf_cost >= self.tools.intra_tu_split_min_cost * self.rd_lambda() * (n * n) as f32;
        if !(try_split && residual.coded && detailed) {
            leaves.push(leaf);
            return leaf_cost;
        }
        let saved = SavedSquare::save(recon, PlaneIndex::Y, (node.x, node.y, n));
        let start = leaves.len;
        let split_cost = self.rd_lambda() * self.split_transform_bits(node.log2_size(), true, true)
            + node
                .quadrants()
                .into_iter()
                .map(|child| self.code_luma_tree(tree, child, source, recon, leaves))
                .sum::<f32>();
        if split_cost < leaf_cost {
            return split_cost;
        }
        saved.restore(recon);
        leaves.truncate(start);
        leaves.push(leaf);
        leaf_cost
    }
}
