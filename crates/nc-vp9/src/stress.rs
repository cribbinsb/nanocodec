//! VP9's random syntax choices for the syntax-coverage stress test
//! (`debug.random_decisions`, see `nc_encoder::DebugRng`).
//!
//! The engine already randomises partitions, intra/inter choice and
//! vectors. This module randomises what the VP9 backend decides itself,
//! always among legal options:
//!
//! - per frame: the loop-filter level (sometimes zero) and sharpness, the
//!   interpolation filter (fixed, or switchable with a random filter per
//!   inter block);
//! - inter blocks: `NEARESTMV`, `NEARMV`, `ZEROMV` or `NEWMV` (including a
//!   zero difference), whichever denote the engine's vector, moved by a
//!   random 1/8 step when the frame allows high precision;
//! - intra blocks: any coded luma and chroma mode (DC, V, H, TM);
//! - every block: its residual dropped (`skip`) by chance;
//! - inter frames: segmentation on by chance, with random tree
//!   probabilities, per-segment filter data and segment ids.
//!
//! Real encodes never construct [`RandomSyntax`].

use nc_encoder::DebugRng;

use crate::loopfilter::MAX_LEVEL;
use crate::mvref::MvRefs;
use crate::segmentation::{
    MAX_SEGMENTS, PREDICTION_PROBS, SegmentFeatures, Segmentation, TREE_PROBS,
};
use crate::types::{InterMode, IntraMode, Mv};

/// Chance (in 1/256) of a frame's filter level being zero.
const ZERO_LEVEL_CHANCE: u64 = 64;
/// Chance (in 1/256) of an inter frame enabling segmentation, of it
/// sending new features, and of a segment having filter data.
const SEGMENTATION_CHANCE: u64 = 128;
/// Largest `sharpness_level`.
const MAX_SHARPNESS: u8 = 7;
/// Chance (in 1/256) of an inter frame allowing 1/8-sample vectors.
const HIGH_PRECISION_CHANCE: u64 = 128;
/// Chance (in 1/256) of a block's residual being dropped.
const DROP_RESIDUAL_CHANCE: u64 = 40;

/// The backend's random source while the stress test runs.
#[derive(Debug, Clone)]
pub(crate) struct RandomSyntax {
    rng: DebugRng,
}

impl RandomSyntax {
    pub(crate) fn new(rng: DebugRng) -> Self {
        Self { rng }
    }

    /// A legal frame filter level.
    pub(crate) fn filter_level(&mut self) -> u8 {
        if self.rng.chance(ZERO_LEVEL_CHANCE) {
            0
        } else {
            1 + self.rng.below(u64::from(MAX_LEVEL)) as u8
        }
    }

    /// Any of the four frame interpolation filters.
    /// The frame filter, or `None` for switchable (one in five frames).
    pub(crate) fn interp_filter(&mut self) -> Option<nc_dsp::Vp9Filter> {
        [
            Some(nc_dsp::Vp9Filter::Smooth),
            Some(nc_dsp::Vp9Filter::Regular),
            Some(nc_dsp::Vp9Filter::Sharp),
            Some(nc_dsp::Vp9Filter::Bilinear),
            None,
        ][self.rng.below(5) as usize]
    }

    /// Any loop-filter sharpness.
    pub(crate) fn sharpness(&mut self) -> u8 {
        self.rng.below(u64::from(MAX_SHARPNESS) + 1) as u8
    }

    /// Whether the frame allows 1/8-sample vectors.
    pub(crate) fn allow_hp(&mut self) -> bool {
        self.rng.chance(HIGH_PRECISION_CHANCE)
    }

    /// `mv` moved by a random 1/8 step on each axis.
    pub(crate) fn high_precision_vector(&mut self, mv: Mv) -> Mv {
        let step = |rng: &mut DebugRng| rng.below(3) as i32 - 1;
        Mv {
            row: mv.row + step(&mut self.rng),
            col: mv.col + step(&mut self.rng),
        }
    }

    /// Any switchable filter for a block.
    pub(crate) fn block_filter(&mut self) -> nc_dsp::Vp9Filter {
        crate::contexts::symbol_filter(self.rng.below(3) as usize)
    }

    /// Any inter mode that denotes `mv`.
    pub(crate) fn inter_mode(&mut self, refs: &MvRefs, mv: Mv) -> InterMode {
        let mut modes = vec![InterMode::New];
        for (mode, vector) in [
            (InterMode::Zero, Mv::ZERO),
            (InterMode::Nearest, refs.nearest),
            (InterMode::Near, refs.near),
        ] {
            if vector == mv {
                modes.push(mode);
            }
        }
        modes[self.rng.below(modes.len() as u64) as usize]
    }

    /// Any of `modes` for luma and for chroma.
    pub(crate) fn intra_modes(&mut self, modes: &[IntraMode]) -> (IntraMode, IntraMode) {
        let len = modes.len() as u64;
        (
            modes[self.rng.below(len) as usize],
            modes[self.rng.below(len) as usize],
        )
    }

    /// A legal segmentation for the frame, given the features in force.
    pub(crate) fn segmentation(&mut self, state: SegmentFeatures, key: bool) -> Segmentation {
        if key || !self.rng.chance(SEGMENTATION_CHANCE) {
            return Segmentation::disabled(state);
        }
        let mut tree_probs = [0; TREE_PROBS];
        for prob in &mut tree_probs {
            *prob = 1 + self.rng.below(255) as u8;
        }
        let temporal = self.rng.chance(SEGMENTATION_CHANCE).then(|| {
            let mut probs = [0; PREDICTION_PROBS];
            for prob in &mut probs {
                *prob = 1 + self.rng.below(255) as u8;
            }
            probs
        });
        let update_data = self.rng.chance(SEGMENTATION_CHANCE);
        let mut features = state;
        if update_data {
            features.absolute = self.rng.chance(SEGMENTATION_CHANCE);
            let range = 2 * u64::from(MAX_LEVEL) + 1;
            for filter in &mut features.filter {
                *filter = self
                    .rng
                    .chance(SEGMENTATION_CHANCE)
                    .then(|| (self.rng.below(range) as i64 - i64::from(MAX_LEVEL)) as i8);
            }
        }
        Segmentation {
            enabled: true,
            tree_probs,
            temporal,
            update_data,
            features,
        }
    }

    /// Any segment id.
    pub(crate) fn segment_id(&mut self) -> u8 {
        self.rng.below(MAX_SEGMENTS as u64) as u8
    }

    /// Whether to drop the block's whole residual.
    pub(crate) fn drop_residual(&mut self) -> bool {
        self.rng.chance(DROP_RESIDUAL_CHANCE)
    }
}
