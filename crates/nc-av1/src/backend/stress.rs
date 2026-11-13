//! AV1's random syntax choices for the syntax-coverage stress test
//! (`debug.random_decisions`, see `nc_encoder::DebugRng`).
//!
//! The engine already randomises partitions, prediction modes and vectors.
//! This module randomises what the AV1 backend decides itself, always among
//! legal options:
//!
//! - per frame: the chroma `delta_q`, the deblocking levels and sharpness
//!   (when `av1.loop_filter` is on), and the CDEF damping and strengths with
//!   a random per-superblock `cdef_idx`;
//! - intra blocks (key and inter frames): any luma mode with any angle
//!   delta, filter intra where the sequence enables it, and `CfL` with
//!   random slopes where allowed;
//! - inter blocks: `NEARESTMV`, `NEARMV` (any matching list position),
//!   `GLOBALMV` or `NEWMV` from any reachable list position, whichever
//!   denote the engine's vector;
//! - inter transform trees (`TX_MODE_SELECT` in every inter frame);
//! - residual dropped per plane (all planes: `skip`).
//!
//! Real encodes never construct [`RandomSyntax`].

use nc_common::{Picture, PlaneIndex};
use nc_encoder::{Block, DebugRng};

use super::{
    Av1Backend, CompoundInterMode, DELTA_Q_MAX, DELTA_Q_MIN, InterMode, MAX_NEAR_INDEX,
    PlaneResidual, Prediction, uv_mode_alphabet,
};
use crate::cdef::{self, CdefParams};
use crate::cfl::ChromaFromLuma;
use crate::intra::{LumaIntra, MAX_ANGLE_DELTA, filter_intra_allowed};
use crate::loopfilter::{LoopFilterParams, MAX_LOOP_FILTER};
use crate::modeinfo::Av1Mv;
use crate::mvref::{CompoundMvStack, MvStack};
use crate::predict::{FilterIntraMode, IntraMode};
use crate::tables::CFL_ALLOWED;
use crate::txsize::TxSplit;

/// Largest `RefMvIdx` a `NEWMV` block can name (two `drl_mode` symbols).
const MAX_NEW_INDEX: usize = 2;
/// Largest `loop_filter_sharpness`, `cdef_damping_minus_3`, CDEF primary
/// strength and coded secondary strength (AV1 §5.9.11, §5.9.19).
const MAX_SHARPNESS: u64 = 7;
const MAX_DAMPING_MINUS_3: u64 = 3;
const MAX_CDEF_PRIMARY: u64 = 15;
const MAX_CDEF_SECONDARY: u64 = 3;
/// Largest `CfL` slope magnitude (`cfl_alpha_u/v` + 1).
const MAX_CFL_ALPHA: i32 = 16;

/// Chance (in 1/256) of a frame-level value taking a random non-default
/// value (chroma `delta_q`) or of a loop-filter level being zero.
const DELTA_Q_CHANCE: u64 = 128;
const ZERO_LEVEL_CHANCE: u64 = 64;
/// Chance (in 1/256) of a superblock taking the second CDEF strength.
const CDEF_ALTERNATE_CHANCE: u64 = 128;
/// Chance (in 1/256) of an inter transform (or quarter) splitting.
const TX_SPLIT_CHANCE: u64 = 112;
/// Chance (in 1/256) of an intra block that allows `CfL` using it.
const CFL_CHANCE: u64 = 128;
/// Chance (in 1/256) of all of a block's residual being dropped (`skip`),
/// and otherwise of one plane's.
const DROP_ALL_CHANCE: u64 = 40;
const DROP_PLANE_CHANCE: u64 = 56;

/// The random source of the syntax stress test.
#[derive(Debug, Clone)]
pub(super) struct RandomSyntax {
    rng: DebugRng,
}

impl RandomSyntax {
    pub(super) fn new(rng: DebugRng) -> Self {
        Self { rng }
    }

    /// The frame's chroma `delta_q` (`DeltaQUDc` = `DeltaQUAc`).
    pub(super) fn chroma_delta_q(&mut self) -> i32 {
        if self.rng.chance(DELTA_Q_CHANCE) {
            DELTA_Q_MIN + self.rng.below((DELTA_Q_MAX - DELTA_Q_MIN + 1) as u64) as i32
        } else {
            0
        }
    }

    /// The frame's deblocking levels and sharpness. Zero levels are common
    /// so every combination of filtered and unfiltered directions occurs.
    pub(super) fn loop_filter(&mut self) -> LoopFilterParams {
        let mut level = || {
            if self.rng.chance(ZERO_LEVEL_CHANCE) {
                0
            } else {
                self.rng.below(u64::from(MAX_LOOP_FILTER) + 1) as u8
            }
        };
        let level = [level(), level(), level(), level()];
        LoopFilterParams {
            level,
            sharpness: self.rng.below(MAX_SHARPNESS + 1) as u8,
        }
    }

    /// Any luma intra prediction the subset codes: every mode, any angle
    /// delta of a directional one, and (where the sequence enables it)
    /// filter intra on half the DC blocks it is allowed on.
    pub(super) fn luma_intra(&mut self, log2_size: u8, filter_intra: bool) -> LumaIntra {
        let modes = IntraMode::ALL;
        let mode = modes[self.rng.below(modes.len() as u64) as usize];
        let mut luma = LumaIntra::plain(mode);
        if mode.nominal_angle().is_some() {
            let deltas = 2 * MAX_ANGLE_DELTA as u64 + 1;
            luma.angle_delta = self.rng.below(deltas) as i8 - MAX_ANGLE_DELTA;
        }
        if filter_intra
            && mode == IntraMode::Dc
            && filter_intra_allowed(log2_size)
            && self.rng.below(2) == 0
        {
            let filters = FilterIntraMode::ALL;
            luma.filter = Some(filters[self.rng.below(filters.len() as u64) as usize]);
        }
        luma
    }

    /// A random one of the inter modes that denote `mv` with `stack`.
    pub(super) fn inter_mode(&mut self, stack: &MvStack, mv: Av1Mv) -> InterMode {
        // NEWMV reaches list position k when the stack holds more than k
        // vectors; NEARMV position 1 always, positions 2 and 3 likewise.
        let mut options: Vec<InterMode> = (0..=MAX_NEW_INDEX)
            .filter(|&k| k == 0 || stack.count > k)
            .map(InterMode::New)
            .collect();
        if mv == stack.vector(0) {
            options.push(InterMode::Nearest);
        }
        options.extend(
            (1..=MAX_NEAR_INDEX)
                .filter(|&k| (k == 1 || k < stack.count) && stack.vector(k) == mv)
                .map(InterMode::Near),
        );
        if mv == Av1Mv::ZERO {
            options.push(InterMode::Global);
        }
        options[self.rng.below(options.len() as u64) as usize]
    }

    /// A random one of the compound inter modes that denote `mvs` with `stack`.
    pub(super) fn compound_mode(
        &mut self,
        stack: &CompoundMvStack,
        mvs: [Av1Mv; 2],
    ) -> CompoundInterMode {
        let mut options: Vec<CompoundInterMode> = (0..=MAX_NEW_INDEX)
            .filter(|&k| k == 0 || stack.count > k)
            .map(CompoundInterMode::NewNew)
            .collect();
        let v0 = stack.vector(0);
        if mvs == v0 {
            options.push(CompoundInterMode::NearestNearest);
        }
        if mvs[0] == v0[0] {
            options.push(CompoundInterMode::NearestNew);
        }
        if mvs[1] == v0[1] {
            options.push(CompoundInterMode::NewNearest);
        }
        for k in (1..=MAX_NEAR_INDEX).filter(|&k| k == 1 || k < stack.count) {
            let vk = stack.vector(k);
            if mvs == vk {
                options.push(CompoundInterMode::NearNear(k));
            }
            if mvs[0] == vk[0] {
                options.push(CompoundInterMode::NearNew(k));
            }
            if mvs[1] == vk[1] {
                options.push(CompoundInterMode::NewNear(k));
            }
        }
        if mvs == [Av1Mv::ZERO, Av1Mv::ZERO] {
            options.push(CompoundInterMode::GlobalGlobal);
        }
        options[self.rng.below(options.len() as u64) as usize]
    }

    /// A random inter transform tree (split, then each quarter again).
    pub(super) fn tx_split(&mut self) -> TxSplit {
        if !self.rng.chance(TX_SPLIT_CHANCE) {
            return TxSplit::NONE;
        }
        TxSplit {
            split: true,
            quadrants: [0; 4].map(|_| self.rng.chance(TX_SPLIT_CHANCE)),
        }
    }

    /// CDEF strengths of a frame.
    fn cdef_params(&mut self) -> CdefParams {
        CdefParams {
            damping_minus_3: self.rng.below(MAX_DAMPING_MINUS_3 + 1) as u8,
            y_pri: self.rng.below(MAX_CDEF_PRIMARY + 1) as u8,
            y_sec: self.rng.below(MAX_CDEF_SECONDARY + 1) as u8,
            uv_pri: self.rng.below(MAX_CDEF_PRIMARY + 1) as u8,
            uv_sec: self.rng.below(MAX_CDEF_SECONDARY + 1) as u8,
        }
    }
}

impl Av1Backend {
    fn random_syntax(&mut self) -> &mut RandomSyntax {
        self.random.as_mut().expect("random decisions enabled")
    }

    /// `CfL` with random slopes for an intra block that allows it (luma
    /// already reconstructed), or `None`.
    pub(super) fn random_cfl(
        &mut self,
        block: &Block,
        intra: bool,
        source: &Picture,
        recon: &Picture,
    ) -> Option<ChromaFromLuma> {
        if !intra || uv_mode_alphabet(block).0 != CFL_ALLOWED {
            return None;
        }
        let rng = &mut self.random_syntax().rng;
        if !rng.chance(CFL_CHANCE) {
            return None;
        }
        let mut cfl = ChromaFromLuma::search(block, source, recon);
        // At least one slope is non-zero: the joint sign has no zero/zero.
        cfl.alpha = loop {
            let alpha = [0, 1].map(|_| rng.symmetric(MAX_CFL_ALPHA) as i8);
            if alpha != [0, 0] {
                break alpha;
            }
        };
        Some(cfl)
    }

    /// Drop residual planes at random: the plane is predicted again and
    /// coded with no coefficients. `keep_luma` keeps luma (chroma predicted
    /// from it by `CfL`, or a transform tree).
    pub(super) fn random_residual_drops(
        &mut self,
        block: &Block,
        predictions: [&Prediction<'_>; 3],
        residuals: &mut [PlaneResidual; 3],
        keep_luma: bool,
        recon: &mut Picture,
    ) {
        let rng = &mut self.random_syntax().rng;
        let all = rng.chance(DROP_ALL_CHANCE);
        let drops = [0, 1, 2].map(|_| all || rng.chance(DROP_PLANE_CHANCE));
        for (index, plane) in PlaneIndex::ALL.into_iter().enumerate() {
            let residual = &mut residuals[index];
            if !drops[index] || residual.eob == 0 || (index == 0 && keep_luma) {
                continue;
            }
            self.predict_plane(block, plane, predictions[index], recon);
            residual.eob = 0;
            residual.distortion = None;
        }
    }

    /// Random CDEF strengths and per-superblock strength choices, applied
    /// to `recon`.
    pub(super) fn random_cdef(&mut self, recon: &mut Picture) {
        let random = self.random.as_mut().expect("random decisions enabled");
        let params = random.cdef_params();
        let selected: Vec<bool> = (0..self.grids.cdef_slots.len())
            .map(|_| random.rng.chance(CDEF_ALTERNATE_CHANCE))
            .collect();
        cdef::apply_selected_with_source(
            recon,
            &self.grids.mode_info,
            params,
            &selected,
            &mut self.cdef_input,
        );
        self.finish_cdef_indices(params, &selected);
    }
}
