//! H.266 multi-reference spatial/history motion derivation for general P pictures.
//! Uses the existing sixteenth-sample vector type; BSD-Clear `VVdeC` `UnitTools` reference.
use super::{Inter, Sequence, SliceHeader, Unit};
use crate::motion::Mv16;
use nc_common::decode::{DecodeError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Vector {
    pub reference: u8,
    pub vector: Mv16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Motion {
    pub lists: [Option<Vector>; 2],
    pub weight: u8,
    pub half: bool,
}

pub(crate) fn derive(
    units: &mut [Unit],
    dimensions: [usize; 2],
    sequence: &Sequence,
    header: &SliceHeader,
    references: [&[i32]; 2],
    temporal: Option<&super::Temporal<'_>>,
) -> Result<()> {
    let ctu_log2 = sequence.ctu_log2;
    let merge_log2 = sequence.tools.parallel_merge_log2;
    let stride = dimensions[0].div_ceil(4);
    let mut grid = vec![None; stride * dimensions[1].div_ceil(4)];
    let mut owners = vec![None; grid.len()];
    let mut history = Vec::with_capacity(5);
    let mut row = usize::MAX;
    for unit_index in 0..units.len() {
        let (previous, remaining) = units.split_at_mut(unit_index);
        let unit = &mut remaining[0];
        if unit.channel != 0 {
            continue;
        }
        let n = unit.node;
        if n.y >> ctu_log2 != row {
            row = n.y >> ctu_log2;
            history.clear();
        }
        let at = |x: usize, y: usize| {
            if x < dimensions[0] && y < dimensions[1] {
                grid[(y >> 2) * stride + (x >> 2)]
            } else {
                None
            }
        };
        let neighbours = |x: usize, y: usize| {
            let motion = at(x, y)?;
            let owner: usize = owners[(y >> 2) * stride + (x >> 2)]?;
            let previous: &Unit = &previous[owner];
            Some(super::affine_motion::Neighbour {
                node: previous.node,
                motion,
                models: previous.affine_models,
            })
        };
        let positions = [
            (n.x + n.width - 1, n.y.wrapping_sub(1)),
            (n.x.wrapping_sub(1), n.y + n.height - 1),
            (n.x + n.width, n.y.wrapping_sub(1)),
            (n.x.wrapping_sub(1), n.y + n.height),
            (n.x.wrapping_sub(1), n.y.wrapping_sub(1)),
        ];
        let spatial = positions.map(|(x, y)| at(x, y));
        let eligible = std::array::from_fn(|i| {
            let (x, y) = positions[i];
            spatial[i].filter(|_| {
                x >> merge_log2 != n.x >> merge_log2 || y >> merge_log2 != n.y >> merge_log2
            })
        });
        let affine = super::affine_motion::Candidates {
            node: n,
            references,
            neighbours: &neighbours,
            ctu_log2,
            merge_log2,
            temporal,
        };
        let mut resolved = Selection {
            affine,
            spatial,
            eligible,
            history: &history,
            sequence,
            header,
        }
        .resolve(unit)?;
        restrict_biprediction(unit.node, &mut resolved);
        if let Some(motion) = resolved {
            if motion.lists.iter().zip(references).any(|(vector, refs)| {
                vector.is_some_and(|v| usize::from(v.reference) >= refs.len())
            }) {
                return Err(DecodeError::Malformed("VVC unavailable motion reference"));
            }
            if unit.subblocks.is_empty()
                && unit.geometric.is_none()
                && unit.affine_models.iter().all(Option::is_none)
                && (n.x + n.width) >> merge_log2 > n.x >> merge_log2
                && (n.y + n.height) >> merge_log2 > n.y >> merge_log2
            {
                history.retain(|old| old.lists != motion.lists);
                if history.len() == 5 {
                    history.remove(0);
                }
                history.push(motion);
            }
        }
        unit.motion = resolved;
        for y in (n.y..n.y + n.height).step_by(4) {
            for x in (n.x..n.x + n.width).step_by(4) {
                let index = (y >> 2) * stride + (x >> 2);
                grid[index] = self::at(unit, x, y);
                owners[index] = Some(unit_index);
            }
        }
    }
    Ok(())
}

fn restrict_biprediction(n: super::Node, resolved: &mut Option<Motion>) {
    if n.width + n.height <= 12
        && let Some(motion) = resolved
        && motion.lists.iter().all(Option::is_some)
    {
        motion.lists[1] = None;
        motion.weight = 0;
    }
}

struct Selection<'a, F> {
    affine: super::affine_motion::Candidates<'a, F>,
    spatial: [Option<Motion>; 5],
    eligible: [Option<Motion>; 5],
    history: &'a [Motion],
    sequence: &'a Sequence,
    header: &'a SliceHeader,
}
impl<F: Fn(usize, usize) -> Option<super::affine_motion::Neighbour>> Selection<'_, F> {
    fn resolve(&self, unit: &mut Unit) -> Result<Option<Motion>> {
        let (n, references, temporal) = (unit.node, self.affine.references, self.affine.temporal);
        let (affine, eligible, history, sequence, header) = (
            &self.affine,
            self.eligible,
            self.history,
            self.sequence,
            self.header,
        );
        let maximum = sequence.tools.max_merge;
        let candidates = || {
            merge(
                eligible,
                history,
                references.map(<[i32]>::len),
                maximum,
                temporal.and_then(|t| t.merge(n)),
            )
        };
        let resolved = match unit.inter {
            None => None,
            Some(Inter::Merge(index) | Inter::Ciip(index)) => {
                Some(candidates()[usize::from(index)])
            }
            Some(Inter::Mmvd {
                base,
                step,
                direction,
            }) => Some(super::motion_refine::mmvd(
                candidates()[usize::from(base)],
                step,
                direction,
                header,
            )?),
            Some(Inter::Subblock(index)) => {
                let subblocks = temporal
                    .filter(|_| sequence.tools.subblock_mvp)
                    .and_then(|t| t.subblocks(n, eligible[1]));
                let available = subblocks.is_some();
                if index == 0
                    && let Some(cells) = subblocks
                {
                    let motion = cells[0];
                    unit.subblocks = cells;
                    Some(motion)
                } else {
                    let list =
                        affine.merge(sequence.tools.affine_six, sequence.tools.max_subblock_merge);
                    let (motion, models) = list[usize::from(index) - usize::from(available)];
                    unit.affine_models = models;
                    Some(motion)
                }
            }
            Some(Inter::Amvp {
                lists,
                points,
                precision,
                weight,
            }) => Some(self.amvp(unit, lists, points, precision, weight)),
            Some(Inter::Geometric { indices, .. }) => {
                let candidates = candidates();
                let selected = indices.map(|index| {
                    let source = candidates[usize::from(index)];
                    let list = usize::from(index & 1);
                    let list = if source.lists[list].is_some() {
                        list
                    } else {
                        1 - list
                    };
                    let mut motion = Motion::default();
                    motion.lists[list] = source.lists[list];
                    motion
                });
                unit.geometric = Some(selected);
                Some(selected[0])
            }
        };
        Ok(resolved)
    }
    fn amvp(
        &self,
        unit: &mut Unit,
        lists: [Option<super::inter_syntax::Prediction>; 2],
        points: u8,
        precision: u8,
        weight: u8,
    ) -> Motion {
        let (n, references, temporal) = (unit.node, self.affine.references, self.affine.temporal);
        let (affine, spatial, history) = (&self.affine, self.spatial, self.history);
        let shift = [2, 4, 6, 3][usize::from(precision)];
        let mut motion = Motion {
            weight,
            half: precision == 3,
            ..Motion::default()
        };
        for (list, prediction) in lists.into_iter().enumerate() {
            if let Some(prediction) = prediction {
                if points > 1 {
                    let model = affine.amvp(points == 3, precision, prediction, list);
                    unit.affine_models[list] = Some(model);
                    motion.lists[list] = Some(Vector {
                        reference: prediction.reference,
                        vector: model.cp[0],
                    });
                    continue;
                }
                let predictor = amvp(
                    spatial,
                    history,
                    list,
                    prediction.reference,
                    references,
                    shift,
                    temporal.and_then(|t| t.candidate(n, list, prediction.reference)),
                )[usize::from(prediction.mvp)];
                motion.lists[list] = Some(Vector {
                    reference: prediction.reference,
                    vector: Mv16 {
                        x: wrap_component(predictor.x + (prediction.mvd[0][0] << shift)),
                        y: wrap_component(predictor.y + (prediction.mvd[0][1] << shift)),
                    },
                });
            }
        }
        motion
    }
}

use super::motion_candidates::{amvp, merge};

// H.266 §8.5.2.14 uses signed 18-bit modular reduction for decoded MVD sums,
// unlike the saturating clip used when deriving an affine model.
pub(super) fn wrap_component(value: i32) -> i32 {
    const STORAGE_BITS: u32 = 18;
    (value << (32 - STORAGE_BITS)) >> (32 - STORAGE_BITS)
}

#[cfg(test)]
#[path = "motion_tests.rs"]
mod tests;

/// Motion seen by a neighbouring CU or the deblocker at a luma position.
pub(super) fn at(unit: &Unit, x: usize, y: usize) -> Option<Motion> {
    let n = unit.node;
    let mut motion = unit.motion?;
    if !unit.subblocks.is_empty() {
        return Some(unit.subblocks[(y - n.y) / 8 * (n.width / 8) + (x - n.x) / 8]);
    }
    if let (Some(pair), Some(Inter::Geometric { split, .. })) = (unit.geometric, unit.inter) {
        let mask = nc_dsp::VvcGeometric::new((n.width, n.height), split);
        motion = match mask.motion_partition(((x - n.x) / 4) * 4, ((y - n.y) / 4) * 4) {
            0 => pair[0],
            1 => pair[1],
            _ => Motion {
                lists: std::array::from_fn(|list| pair[1].lists[list].or(pair[0].lists[list])),
                ..Motion::default()
            },
        };
    }
    let bidirectional = motion.lists.iter().all(Option::is_some);
    for (vector, model) in motion.lists.iter_mut().zip(unit.affine_models) {
        if let (Some(vector), Some(model)) = (vector, model) {
            vector.vector = model
                .field_for_prediction(n.width, n.height, bidirectional)
                .at((x - n.x) / 4, (y - n.y) / 4);
        }
    }
    Some(motion)
}
