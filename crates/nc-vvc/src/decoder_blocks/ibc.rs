//! Intra block copy (H.266 §7.3.11.5, §8.6): prediction-mode syntax with IBC
//! enabled, block-vector candidates and history, and the copy from the
//! current picture's reconstruction. Follows the VTM reference decoder
//! (`CABACReader::cu_skip_flag`, `pred_mode`, `PU::getIBCMergeCandidates`,
//! `PU::fillIBCMvpCand`; BSD 3-clause, see `THIRD_PARTY.md`).
use super::{
    Constraint, Context, Inter, Parser, Reader, Result, Unit, initialization::bank,
    inter_syntax::index,
};
use crate::motion::Mv16;
use nc_common::{Picture, PlaneIndex};

/// Largest CU side that may use IBC.
const MAX_SIDE: usize = 64;
/// Block-vector history length (`MAX_NUM_HMVP_CANDS`).
const HISTORY: usize = 5;
/// Fractional bits of a stored block vector (whole samples in 1/16 units).
const FRACTION_BITS: u32 = 4;
/// AMVR shifts of an IBC vector difference: whole and four samples.
const PRECISION_SHIFTS: [u32; 2] = [4, 6];

/// How an IBC CU signals its block vector.
#[derive(Debug, Clone, Copy)]
pub(super) enum BlockVector {
    Merge(u8),
    Predicted {
        difference: [i32; 2],
        predictor: u8,
        /// Index into `PRECISION_SHIFTS`.
        coarse: bool,
    },
}

#[derive(Clone)]
pub(super) struct Models {
    flag: [Context; 3],
}
impl Models {
    pub fn new(qp: u32, kind: u8) -> Self {
        Self {
            flag: bank(
                [[0, 43, 45], [0, 57, 44], [17, 42, 36]],
                [1, 5, 8],
                qp,
                kind,
            ),
        }
    }
}

impl Parser<'_> {
    /// `cu_skip_flag`, `pred_mode_flag` and `pred_mode_ibc_flag` of a luma or
    /// single-tree CU when the SPS enables IBC; `true` for an intra CU.
    pub(super) fn prediction_with_ibc(
        &mut self,
        r: &mut Reader<'_>,
        unit: &mut Unit,
        constraint: Constraint,
    ) -> Result<bool> {
        let n = unit.node;
        let allowed = n.width <= MAX_SIDE && n.height <= MAX_SIDE;
        let small = n.width * n.height == 16;
        let neighbours = [
            self.neighbour(n.x.wrapping_sub(1), n.y, 0),
            self.neighbour(n.x, n.y.wrapping_sub(1), 0),
        ];
        let count =
            |test: fn(&Unit) -> bool| neighbours.iter().filter(|u| u.is_some_and(test)).count();
        let skip_context = count(|u| u.skip);
        let ibc_context = count(|u| matches!(u.inter, Some(Inter::Ibc(_))));
        let intra_context = usize::from(count(|u| u.inter.is_none()) != 0);
        let ibc = if self.header.kind == 2 || constraint == Constraint::Intra {
            // Only intra and IBC are possible; a skipped CU is IBC.
            if !allowed {
                return Ok(true);
            }
            unit.skip = r.bin(&mut self.contexts.skip[skip_context])?;
            unit.skip || r.bin(&mut self.ibc_models.flag[ibc_context])?
        } else {
            unit.skip = r.bin(&mut self.contexts.skip[skip_context])?;
            let may = allowed && constraint != Constraint::Inter;
            if unit.skip {
                may && (small || r.bin(&mut self.ibc_models.flag[ibc_context])?)
            } else if constraint == Constraint::Inter {
                false
            } else if small {
                // A 4x4 CU is intra unless it is IBC.
                if !r.bin(&mut self.ibc_models.flag[ibc_context])? {
                    return Ok(true);
                }
                true
            } else if r.bin(&mut self.contexts.intra[intra_context])? {
                return Ok(true);
            } else {
                may && r.bin(&mut self.ibc_models.flag[ibc_context])?
            }
        };
        if !ibc {
            if self.header.kind == 2 || constraint == Constraint::Intra {
                return Ok(true);
            }
            self.inter_prediction(r, unit)?;
            return Ok(false);
        }
        unit.inter = Some(Inter::Ibc(self.block_vector_syntax(r, unit.skip)?));
        Ok(false)
    }

    fn block_vector_syntax(&mut self, r: &mut Reader<'_>, skip: bool) -> Result<BlockVector> {
        let candidates = self.sequence.tools.max_ibc_merge;
        if skip || r.bin(&mut self.contexts.merge)? {
            return Ok(BlockVector::Merge(index(
                r,
                &mut self.contexts.merge_idx,
                candidates,
            )?));
        }
        let difference = crate::decoder_syntax::read_mvd(r, &mut self.contexts)?;
        let predictor = u8::from(candidates > 1 && r.bin(&mut self.contexts.mvp)?);
        let coarse = self.sequence.tools.amvr
            && difference != [0; 2]
            && r.bin(&mut self.inter_models.precision[1])?;
        Ok(BlockVector::Predicted {
            difference,
            predictor,
            coarse,
        })
    }
}

/// Block-vector prediction state of a picture: the history list, reset at
/// each CTU row (§8.6.2.6).
#[derive(Default)]
pub(super) struct History(Vec<Mv16>);

impl History {
    pub fn clear(&mut self) {
        self.0.clear();
    }

    #[cfg(test)]
    fn clone_for_test(&self) -> Self {
        Self(self.0.clone())
    }

    /// Resolve an IBC CU's block vector from its left (`A1`) and above (`B1`)
    /// IBC neighbours and the history, then record it (§8.6.2.2).
    pub fn resolve(
        &mut self,
        signalled: BlockVector,
        (width, height): (usize, usize),
        [left, above]: [Option<Mv16>; 2],
        maximum: u8,
    ) -> Mv16 {
        let maximum = usize::from(maximum);
        let large = width * height > 16;
        let (left, above) = (left.filter(|_| large), above.filter(|_| large));
        let mut candidates = Vec::with_capacity(maximum);
        candidates.extend(left);
        if candidates.len() < maximum {
            candidates.extend(above.filter(|&v| left != Some(v)));
        }
        // Only the latest history entry is compared with the neighbours.
        for (age, &entry) in self.0.iter().rev().enumerate() {
            if candidates.len() >= maximum {
                break;
            }
            if age > 0 || !large || (left != Some(entry) && above != Some(entry)) {
                candidates.push(entry);
            }
        }
        candidates.resize(maximum.max(2), Mv16::ZERO);
        let vector = match signalled {
            BlockVector::Merge(index) => candidates[usize::from(index)],
            BlockVector::Predicted {
                difference,
                predictor,
                coarse,
            } => {
                let shift = PRECISION_SHIFTS[usize::from(coarse)];
                let predicted = candidates[usize::from(predictor)].round(shift);
                Mv16 {
                    x: super::motion::wrap_component(predicted.x + (difference[0] << shift)),
                    y: super::motion::wrap_component(predicted.y + (difference[1] << shift)),
                }
            }
        };
        if large {
            self.0.retain(|&old| old != vector);
            if self.0.len() == HISTORY {
                self.0.remove(0);
            }
            self.0.push(vector);
        }
        vector
    }
}

/// Whether the block a vector refers to lies inside the picture. A
/// conforming stream refers only to reconstructed samples of the current and
/// left CTUs (§8.6.2.1); this bounds every later sample read.
pub(super) fn inside(
    vector: Mv16,
    (x, y, width, height): (usize, usize, usize, usize),
    dimensions: [usize; 2],
) -> bool {
    let (dx, dy) = (vector.x >> FRACTION_BITS, vector.y >> FRACTION_BITS);
    let (left, top) = (x as i64 + i64::from(dx), y as i64 + i64::from(dy));
    left >= 0
        && top >= 0
        && left + width as i64 <= dimensions[0] as i64
        && top + height as i64 <= dimensions[1] as i64
}

/// Copy a block's prediction from the picture's reconstruction so far
/// (before the loop filters), displaced by the block vector; chroma uses the
/// halved vector (§8.6.3).
pub(super) fn predict<T: nc_common::Sample>(
    picture: &Picture<T>,
    plane: PlaneIndex,
    vector: Mv16,
    (x, y, width, height): (usize, usize, usize, usize),
    out: &mut [T],
) {
    let shift = FRACTION_BITS + u32::from(plane != PlaneIndex::Y);
    // Motion derivation accepted the vector (`inside`), so the region exists.
    if let Some(source) = picture.plane(plane).region(
        x as isize + (vector.x >> shift) as isize,
        y as isize + (vector.y >> shift) as isize,
        width,
        height,
    ) {
        for (row, target) in out.chunks_exact_mut(width).enumerate() {
            target.copy_from_slice(&source.row(row)[..width]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(x: i32, y: i32) -> Mv16 {
        Mv16 {
            x: x << 4,
            y: y << 4,
        }
    }

    #[test]
    fn candidates_take_neighbours_then_history_then_zero() {
        let mut history = History::default();
        // Three earlier CUs fill the history: (-8, 0), (-16, 0), (0, -8).
        for (x, y) in [(-8, 0), (-16, 0), (0, -8)] {
            let merged =
                history.resolve(BlockVector::Merge(0), (8, 8), [Some(vector(x, y)), None], 6);
            assert_eq!(merged, vector(x, y));
        }
        let (left, above) = (vector(0, -8), vector(-4, -4));
        let pick = |history: &mut History, index| {
            history.clone_for_test().resolve(
                BlockVector::Merge(index),
                (8, 8),
                [Some(left), Some(above)],
                6,
            )
        };
        // Left, above, then the history newest first; its newest entry equals
        // the left neighbour and is pruned.
        assert_eq!(pick(&mut history, 0), left);
        assert_eq!(pick(&mut history, 1), above);
        assert_eq!(pick(&mut history, 2), vector(-16, 0));
        assert_eq!(pick(&mut history, 3), vector(-8, 0));
        assert_eq!(pick(&mut history, 4), Mv16::ZERO);
    }

    #[test]
    fn small_blocks_use_only_the_history_and_do_not_enter_it() {
        let mut history = History::default();
        history.resolve(
            BlockVector::Merge(0),
            (8, 8),
            [Some(vector(-8, 0)), None],
            6,
        );
        let neighbours = [Some(vector(-4, 0)), Some(vector(0, -4))];
        // A 4x4 CU ignores its neighbours and keeps the unpruned history.
        let small = history.resolve(BlockVector::Merge(0), (4, 4), neighbours, 6);
        assert_eq!(small, vector(-8, 0));
        assert_eq!(history.0, [vector(-8, 0)]);
    }

    #[test]
    fn coarse_difference_rounds_the_predictor_to_four_samples() {
        let mut history = History::default();
        let signalled = BlockVector::Predicted {
            difference: [-1, 2],
            predictor: 0,
            coarse: true,
        };
        // Predictor (-6, 5) rounds to (-4, 4), a tie going towards zero; the
        // difference is in units of 4.
        let resolved = history.resolve(signalled, (16, 16), [Some(vector(-6, 5)), None], 6);
        assert_eq!(resolved, vector(-8, 12));
    }

    #[test]
    fn vectors_leaving_the_picture_are_rejected() {
        let block = (16, 8, 8, 8);
        assert!(inside(vector(-16, -8), block, [64, 64]));
        assert!(!inside(vector(-17, 0), block, [64, 64]));
        assert!(!inside(vector(0, -9), block, [64, 64]));
        assert!(!inside(vector(41, 0), block, [64, 64]));
    }
}
