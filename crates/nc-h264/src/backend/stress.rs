//! H.264's random syntax choices for the syntax-coverage stress test
//! (`debug.random_decisions`, see `nc_encoder::DebugRng`).
//!
//! The engine already randomises partitions, prediction modes, vectors and
//! block QPs. This module randomises what the H.264 backend decides on its
//! own, always among legal options:
//!
//! - per slice: `cabac_init_idc` and the deblocking control
//!   (`disable_deblocking_filter_idc`, alpha/beta offsets);
//! - intra macroblocks: `I_16x16` or `I_NxN` (`Intra_4x4` or `Intra_8x8`),
//!   every `Intra16x16PredMode`, `Intra4x4PredMode`, `Intra8x8PredMode` and
//!   `intra_chroma_pred_mode` the neighbours allow;
//! - inter macroblocks: the macroblock partition and each partition's
//!   vector, the transform size, dropped residual (per 8x8 luma block, per
//!   chroma component, or all of it, which makes `P_Skip` likely) and coding
//!   a skippable macroblock as `P_L0_16x16` instead.
//!
//! Real encodes never construct [`RandomSyntax`]; the backend's normal
//! decision code only checks whether it exists.

use nc_common::{Picture, PlaneIndex};
use nc_encoder::{Block, DebugRng, Mv, PredictionMode};

use super::{H264Backend, MbChoice};
use crate::cabac::CABAC_INIT_IDC_COUNT;
use crate::deblock::{
    DEBLOCK_IDC_NO_SLICE_EDGES, DEBLOCK_IDC_OFF, DEBLOCK_IDC_ON, DEBLOCK_OFFSET_DIV2_MAX,
    DeblockControl,
};
use crate::intra_nxn::MODE_COUNT;
use crate::intra4x4::code_intra_4x4_with;
use crate::intra8x8::code_intra_8x8_with;
use crate::macroblock::{
    ALL_QUADRANTS, IntraModes, Prediction, ResidualKind, ResidualSkip, predict_inter,
};
use crate::mbinfo::{MbPartition, Neighbours};
use crate::predict::{Availability, ChromaMode, Edges, Intra16x16Mode, chroma_8x8, luma_16x16};

/// Chance (in 1/256) of each slice-level option leaving its default, so
/// default and non-default values both stay common.
const SLICE_OPTION_CHANCE: u64 = 160;
/// Chance (in 1/256) that an intra macroblock is `I_NxN` rather than
/// `I_16x16`.
const INTRA_4X4_CHANCE: u64 = 128;
/// Chance (in 1/256) that an `I_NxN` macroblock uses 8x8 rather than 4x4
/// transforms (`Intra_8x8` prediction).
const INTRA_8X8_CHANCE: u64 = 128;
/// Chance (in 1/256) that an `I_NxN` block takes its predicted mode (one
/// flag) rather than a random one (`rem_intra*_pred_mode`).
const PREDICTED_MODE_CHANCE: u64 = 80;
/// Chance (in 1/256) that an inter macroblock keeps the engine's 16x16
/// partition and vector.
const UNSPLIT_CHANCE: u64 = 96;
/// Chance (in 1/256) that a partition takes the engine's 16x16 vector
/// rather than its sub-block vector or a perturbed one.
const SAME_VECTOR_CHANCE: u64 = 64;
/// Largest perturbation of a partition vector, quarter samples.
const VECTOR_JITTER: i32 = 6;
/// Chance (in 1/256) that all of an inter macroblock's residual is dropped.
const DROP_ALL_CHANCE: u64 = 48;
/// Chance (in 1/256) that one 8x8 luma block's, or one chroma component's,
/// residual is dropped.
const DROP_PART_CHANCE: u64 = 64;
/// Chance (in 1/256) that an inter macroblock uses 4x4 rather than 8x8
/// luma transforms.
const TRANSFORM_4X4_CHANCE: u64 = 128;
/// Chance (in 1/256) that a macroblock which could be `P_Skip` is coded as
/// `P_L0_16x16` with zero `mvd` and no residual instead.
const KEEP_CODED_CHANCE: u64 = 64;

/// The random source and the choices made for the macroblock being coded.
#[derive(Debug, Clone)]
pub(super) struct RandomSyntax {
    rng: DebugRng,
    /// Luma transform of the current inter macroblock.
    inter_kind: ResidualKind,
    /// Whether the current macroblock must not become `P_Skip`.
    keep_coded: bool,
}

impl RandomSyntax {
    pub(super) fn new(rng: DebugRng) -> Self {
        Self {
            rng,
            inter_kind: ResidualKind::Inter8x8,
            keep_coded: false,
        }
    }

    /// `cabac_init_idc` of a P slice.
    pub(super) fn cabac_init_idc(&mut self) -> u8 {
        if self.rng.chance(SLICE_OPTION_CHANCE) {
            self.rng.below(u64::from(CABAC_INIT_IDC_COUNT)) as u8
        } else {
            0
        }
    }

    /// The deblocking control of a slice.
    pub(super) fn deblock_control(&mut self) -> DeblockControl {
        if !self.rng.chance(SLICE_OPTION_CHANCE) {
            return DeblockControl::DEFAULT;
        }
        let disable_idc = [DEBLOCK_IDC_ON, DEBLOCK_IDC_OFF, DEBLOCK_IDC_NO_SLICE_EDGES]
            [self.rng.below(3) as usize];
        let mut offset = || self.rng.symmetric(i32::from(DEBLOCK_OFFSET_DIV2_MAX)) as i8;
        let control = DeblockControl {
            disable_idc,
            alpha_offset_div2: offset(),
            beta_offset_div2: offset(),
        };
        if control.offsets_coded() {
            control
        } else {
            // Absent offsets are inferred as 0 (§7.4.3).
            DeblockControl {
                disable_idc,
                ..DeblockControl::DEFAULT
            }
        }
    }

    /// The luma transform of the current inter macroblock.
    pub(super) fn inter_kind(&self) -> ResidualKind {
        self.inter_kind
    }

    /// Whether the current macroblock must not become `P_Skip`.
    pub(super) fn keeps_coded(&self) -> bool {
        self.keep_coded
    }

    /// A random element of `options` (not empty).
    fn pick<T: Copy>(&mut self, options: impl Iterator<Item = T>) -> T {
        let options: Vec<T> = options.collect();
        options[self.rng.below(options.len() as u64) as usize]
    }
}

impl H264Backend {
    /// The random replacement of `choose_intra` / `choose_inter` for the
    /// macroblock `block`, whose engine-chosen mode is `mode`.
    pub(super) fn random_choice(
        &mut self,
        block: &Block,
        mode: PredictionMode,
        source: &Picture,
        reference: Option<&Picture>,
        recon: &mut Picture,
        sub_vectors: Option<[Mv; 4]>,
        pred: &mut Prediction,
    ) -> (MbChoice, ResidualSkip) {
        match mode {
            PredictionMode::IntraDc => (
                self.random_intra(block, source, recon, pred),
                ResidualSkip::NONE,
            ),
            PredictionMode::Inter(mv) => {
                let reference = reference.expect("inter blocks need a reference picture");
                self.random_inter(block, reference, mv, sub_vectors, pred)
            }
        }
    }

    /// Random `I_16x16` or `I_NxN` modes, each among those the neighbours
    /// allow (§8.3.1.2, §8.3.3, §8.3.4). `I_NxN` luma is reconstructed into
    /// `recon` block by block.
    fn random_intra(
        &mut self,
        block: &Block,
        source: &Picture,
        recon: &mut Picture,
        pred: &mut Prediction,
    ) -> MbChoice {
        let (x, y) = (block.x, block.y);
        let (col, row) = Self::mb_index(block);
        let avail = Availability::at(x, y);
        let random = self.random.as_mut().expect("random decisions enabled");
        let luma = random.pick(Intra16x16Mode::usable(avail));
        let chroma = random.pick(ChromaMode::usable(avail));
        // Intra_4x4 may read the dirty above-right macroblock on intra
        // refresh's dirty edge (see `choose_intra`).
        let refresh_edge = self
            .refresh
            .as_ref()
            .is_some_and(|w| w.on_dirty_edge(block));
        let intra_4x4 = !refresh_edge && random.rng.chance(INTRA_4X4_CHANCE);
        pred.luma = luma_16x16(
            &Edges::gather(&recon.plane(PlaneIndex::Y).view(), x, y),
            luma,
        );
        pred.chroma = [PlaneIndex::U, PlaneIndex::V]
            .map(|p| chroma_8x8(&Edges::gather(&recon.plane(p).view(), x / 2, y / 2), chroma));
        let modes = IntraModes {
            luma,
            chroma,
            luma_satd: 0,
        };
        if !intra_4x4 {
            return MbChoice::Intra16x16(modes);
        }
        let nb = Neighbours::of(&self.mbs, self.params.mb_cols, col, row);
        let intra_8x8 = random.rng.chance(INTRA_8X8_CHANCE);
        let rng = &mut random.rng;
        let top_right_mb = row > 0 && col + 1 < self.params.mb_cols;
        // The predicted mode, or a random one the neighbours allow.
        let mut pick_mode = |allows: &dyn Fn(u8) -> bool, predicted: u8| {
            if allows(predicted) && rng.chance(PREDICTED_MODE_CHANCE) {
                predicted
            } else {
                let allowed: Vec<u8> = (0..MODE_COUNT).filter(|&m| allows(m)).collect();
                allowed[rng.below(allowed.len() as u64) as usize]
            }
        };
        if intra_8x8 {
            let i8 = code_intra_8x8_with(
                &self.dsp,
                source,
                recon,
                (x, y),
                top_right_mb,
                self.quantizers.luma_intra(),
                |bx, by, own| nb.predicted_intra_4x4_mode(bx, by, own),
                |_, refs, predicted| {
                    let mode = pick_mode(&|m| refs.allows(m), predicted);
                    let mut pred = [0; 64];
                    self.dsp.h264_intra_8x8(refs, mode, &mut pred);
                    (mode, pred, 0.0)
                },
                f32::INFINITY,
            )
            .expect("never aborted");
            return MbChoice::Intra8x8 {
                modes,
                i8: Box::new(i8),
            };
        }
        let i4 = code_intra_4x4_with(
            &self.dsp,
            source,
            recon,
            (x, y),
            top_right_mb,
            self.quantizers.luma_intra(),
            |bx, by, own| nb.predicted_intra_4x4_mode(bx, by, own),
            |_, neighbours, predicted| {
                let mode = pick_mode(&|m| neighbours.allows(m), predicted);
                (mode, neighbours.predict(mode), 0.0)
            },
            None,
            f32::INFINITY,
        )
        .expect("never aborted");
        MbChoice::Intra4x4 {
            modes,
            i4: Box::new(i4),
        }
    }

    /// A random partitioning of the inter macroblock whose engine vector is
    /// `mv`: each partition takes `mv`, its first quadrant's `sub_vectors`
    /// entry, or one of those perturbed. Then random residual drops.
    fn random_inter(
        &mut self,
        block: &Block,
        reference: &Picture,
        mv: Mv,
        sub_vectors: Option<[Mv; 4]>,
        pred: &mut Prediction,
    ) -> (MbChoice, ResidualSkip) {
        let (x, y) = (block.x, block.y);
        let sub_vectors = sub_vectors.unwrap_or([mv; 4]);
        let random = self.random.as_mut().expect("random decisions enabled");
        let rng = &mut random.rng;
        let (mut partition, mut mvs) = (MbPartition::P16x16, [mv; 4]);
        if !rng.chance(UNSPLIT_CHANCE) {
            partition =
                [MbPartition::P16x8, MbPartition::P8x16, MbPartition::P8x8][rng.below(3) as usize];
            for part in 0..partition.count() {
                let (x8, y8, w8, h8) = partition.rect(part);
                let mut vector = if rng.chance(SAME_VECTOR_CHANCE) {
                    mv
                } else {
                    sub_vectors[y8 * 2 + x8]
                };
                if rng.chance(SAME_VECTOR_CHANCE) {
                    vector.x += rng.symmetric(VECTOR_JITTER) as i16;
                    vector.y += rng.symmetric(VECTOR_JITTER) as i16;
                }
                for qy in y8..y8 + h8 {
                    for qx in x8..x8 + w8 {
                        mvs[qy * 2 + qx] = vector;
                    }
                }
            }
        }
        let mut zero = ResidualSkip::NONE;
        if rng.chance(DROP_ALL_CHANCE) {
            zero.luma_quadrants = ALL_QUADRANTS;
        } else {
            for b8 in 0..4 {
                if rng.chance(DROP_PART_CHANCE) {
                    zero.luma_quadrants |= 1 << b8;
                }
            }
            zero.chroma = [rng.chance(DROP_PART_CHANCE), rng.chance(DROP_PART_CHANCE)];
        }
        if zero.luma_quadrants == ALL_QUADRANTS {
            zero.luma = true;
            zero.chroma = [true; 2];
        }
        random.inter_kind = if random.rng.chance(TRANSFORM_4X4_CHANCE) {
            ResidualKind::Inter
        } else {
            ResidualKind::Inter8x8
        };
        random.keep_coded = random.rng.chance(KEEP_CODED_CHANCE);
        if !self.refresh_allows_quadrants(x, &mvs) {
            (partition, mvs) = (MbPartition::P16x16, [mv; 4]);
        }
        predict_inter(&self.mc, reference, x, y, &mvs, pred);
        (MbChoice::Inter { partition, mvs }, zero)
    }
}
