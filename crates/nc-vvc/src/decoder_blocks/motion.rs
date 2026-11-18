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

/// Entries of the motion history list (`MaxNumMergeCand - 1` at most).
const HISTORY: usize = 5;
const NONE_OWNER: u32 = u32::MAX;
const SUBBLOCK_FLAG: u32 = 1 << 31;

#[derive(Debug, Default)]
pub(crate) struct MotionScratch {
    owners: Vec<u32>,
    subblock_motions: Vec<(u32, Option<Motion>)>,
}

struct MotionGrid<'a> {
    owners: &'a mut Vec<u32>,
    subblock_motions: &'a mut Vec<(u32, Option<Motion>)>,
    stride: usize,
    dimensions: [usize; 2],
}

impl<'a> MotionGrid<'a> {
    fn from_scratch(scratch: &'a mut MotionScratch, dimensions: [usize; 2]) -> Self {
        let stride = dimensions[0].div_ceil(4);
        let cells = stride * dimensions[1].div_ceil(4);
        scratch.owners.clear();
        scratch.owners.resize(cells, NONE_OWNER);
        scratch.subblock_motions.clear();
        Self {
            owners: &mut scratch.owners,
            subblock_motions: &mut scratch.subblock_motions,
            stride,
            dimensions,
        }
    }

    #[inline]
    fn lookup(
        &self,
        previous: &[Unit],
        (x, y): (usize, usize),
        x_limit: usize,
        tile: u16,
    ) -> Option<(usize, Motion)> {
        if x >= x_limit || y >= self.dimensions[1] {
            return None;
        }
        let raw = self.owners[(y >> 2) * self.stride + (x >> 2)];
        if raw == NONE_OWNER {
            return None;
        }
        let (owner_idx, motion) = if raw & SUBBLOCK_FLAG != 0 {
            let (u_idx, m) = self.subblock_motions[(raw & !SUBBLOCK_FLAG) as usize];
            (u_idx as usize, m?)
        } else {
            let u_idx = raw as usize;
            (u_idx, previous[u_idx].motion?)
        };
        // Motion of another tile is unavailable (H.266 §6.4.4).
        (previous[owner_idx].tile == tile).then_some((owner_idx, motion))
    }
}

#[cfg(test)]
pub(crate) fn derive(
    units: &mut [Unit],
    dimensions: [usize; 2],
    sequence: &Sequence,
    header: &SliceHeader,
    (references, poc): ([&[i32]; 2], i32),
    temporal: Option<&super::Temporal<'_>>,
) -> Result<()> {
    let mut scratch = MotionScratch::default();
    derive_with_scratch(
        units,
        dimensions,
        sequence,
        header,
        (references, poc),
        temporal,
        &mut scratch,
    )
}

pub(crate) fn derive_with_scratch(
    units: &mut [Unit],
    dimensions: [usize; 2],
    sequence: &Sequence,
    header: &SliceHeader,
    (references, poc): ([&[i32]; 2], i32),
    temporal: Option<&super::Temporal<'_>>,
    scratch: &mut MotionScratch,
) -> Result<()> {
    if units.iter().all(|u| u.inter.is_none()) {
        return Ok(());
    }
    let ctu_log2 = sequence.ctu_log2;
    let merge_log2 = sequence.tools.parallel_merge_log2;
    let mut grid = MotionGrid::from_scratch(scratch, dimensions);
    let mut history = Vec::with_capacity(HISTORY);
    let mut vectors = super::ibc::History::default();
    let mut row = (usize::MAX, 0);
    for unit_index in 0..units.len() {
        let (previous, remaining) = units.split_at_mut(unit_index);
        let unit = &mut remaining[0];
        if unit.channel != 0 {
            continue;
        }
        let (n, tile) = (unit.node, unit.tile);
        if (n.y >> ctu_log2, tile) != row {
            row = (n.y >> ctu_log2, tile);
            history.clear();
            vectors.clear();
        }
        if unit.inter.is_none() {
            continue;
        }
        if let Some(Inter::Ibc(signalled)) = unit.inter {
            let maximum = sequence.tools.max_ibc_merge;
            unit.block_vector = Some(block_vector(
                unit,
                signalled,
                previous,
                (grid.owners.as_slice(), dimensions),
                &mut vectors,
                maximum,
            )?);
            claim(grid.owners, n, grid.stride, unit_index as u32);
            continue;
        }
        // With entropy synchronisation nothing in a CTU column to the right
        // of the CU's own is available (H.266 §6.4.4).
        let x_limit = dimensions[0].min(if sequence.entropy_sync {
            ((n.x >> ctu_log2) + 1) << ctu_log2
        } else {
            usize::MAX
        });
        let neighbours = |x: usize, y: usize| {
            let (owner, motion) = grid.lookup(previous, (x, y), x_limit, tile)?;
            let prev = &previous[owner];
            Some(super::affine_motion::Neighbour {
                node: prev.node,
                motion,
                models: prev.affine_models,
            })
        };
        let positions = [
            (n.x + n.width - 1, n.y.wrapping_sub(1)),
            (n.x.wrapping_sub(1), n.y + n.height - 1),
            (n.x + n.width, n.y.wrapping_sub(1)),
            (n.x.wrapping_sub(1), n.y + n.height),
            (n.x.wrapping_sub(1), n.y.wrapping_sub(1)),
        ];
        let spatial = positions.map(|p| grid.lookup(previous, p, x_limit, tile).map(|(_, m)| m));
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
            poc,
        }
        .resolve(unit)?;
        restrict_biprediction(n, &mut resolved);
        if let Some(motion) = resolved {
            record(unit, motion, references, merge_log2, &mut history)?;
            if unit.affine_models.iter().any(Option::is_some) {
                let bidir = motion.lists.iter().all(Option::is_some);
                unit.affine_fields = unit
                    .affine_models
                    .map(|m| m.map(|model| model.field_for_prediction(n.width, n.height, bidir)));
            }
        }
        unit.motion = resolved;
        fill_motion_owners(&mut grid, unit, unit_index as u32);
    }
    Ok(())
}

fn fill_motion_owners(grid: &mut MotionGrid<'_>, unit: &Unit, unit_index: u32) {
    if unit.motion.is_none() {
        return;
    }
    let n = unit.node;
    if unit.subblocks.is_empty()
        && unit.geometric.is_none()
        && unit.affine_fields[0].is_none()
        && unit.affine_fields[1].is_none()
    {
        claim(grid.owners, n, grid.stride, unit_index);
        return;
    }
    for y in (n.y..n.y + n.height).step_by(4) {
        let row_base = (y >> 2) * grid.stride;
        for x in (n.x..n.x + n.width).step_by(4) {
            let sub_idx = grid.subblock_motions.len() as u32;
            grid.subblock_motions
                .push((unit_index, at_with_fields(unit, x, y, &unit.affine_fields)));
            grid.owners[row_base + (x >> 2)] = SUBBLOCK_FLAG | sub_idx;
        }
    }
}

/// Resolve an IBC CU's block vector from the IBC CUs to its left (A1) and
/// above (B1) in its tile, and the vector history (H.266 §8.6.2).
fn block_vector(
    unit: &Unit,
    signalled: super::ibc::BlockVector,
    previous: &[Unit],
    (owners, dimensions): (&[u32], [usize; 2]),
    vectors: &mut super::ibc::History,
    maximum: u8,
) -> Result<Mv16> {
    let n = unit.node;
    let stride = dimensions[0].div_ceil(4);
    let neighbour = |x: usize, y: usize| {
        if x >= dimensions[0] || y >= dimensions[1] {
            return None;
        }
        let raw = owners[(y >> 2) * stride + (x >> 2)];
        if raw == NONE_OWNER || raw & SUBBLOCK_FLAG != 0 {
            return None;
        }
        let owner = raw as usize;
        previous[owner]
            .block_vector
            .filter(|_| previous[owner].tile == unit.tile)
    };
    let neighbours = [
        neighbour(n.x.wrapping_sub(1), n.y + n.height - 1),
        neighbour(n.x + n.width - 1, n.y.wrapping_sub(1)),
    ];
    let vector = vectors.resolve(signalled, (n.width, n.height), neighbours, maximum);
    if super::ibc::inside(vector, (n.x, n.y, n.width, n.height), dimensions) {
        Ok(vector)
    } else {
        Err(DecodeError::Malformed("VVC IBC block vector"))
    }
}

/// Check a CU's resolved motion against the reference lists and enter it in
/// the history list when the CU is eligible (§8.5.2.16).
fn record(
    unit: &Unit,
    motion: Motion,
    references: [&[i32]; 2],
    merge_log2: u8,
    history: &mut Vec<Motion>,
) -> Result<()> {
    let n = unit.node;
    if motion
        .lists
        .iter()
        .zip(references)
        .any(|(vector, refs)| vector.is_some_and(|v| usize::from(v.reference) >= refs.len()))
    {
        return Err(DecodeError::Malformed("VVC unavailable motion reference"));
    }
    if unit.subblocks.is_empty()
        && unit.geometric.is_none()
        && unit.affine_models.iter().all(Option::is_none)
        && (n.x + n.width) >> merge_log2 > n.x >> merge_log2
        && (n.y + n.height) >> merge_log2 > n.y >> merge_log2
    {
        history.retain(|old| old.lists != motion.lists);
        if history.len() == HISTORY {
            history.remove(0);
        }
        history.push(motion);
    }
    Ok(())
}

/// Record a CU as the owner of its 4x4 cells.
fn claim(owners: &mut [u32], n: super::Node, stride: usize, unit: u32) {
    let cols = n.width.div_ceil(4);
    for y in (n.y..n.y + n.height).step_by(4) {
        let base = (y >> 2) * stride + (n.x >> 2);
        owners[base..base + cols].fill(unit);
    }
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
    /// The current picture's POC.
    poc: i32,
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
            None | Some(Inter::Ibc(_)) => None,
            Some(Inter::Merge(index) | Inter::Ciip(index)) => {
                Some(candidates()[usize::from(index)])
            }
            Some(Inter::Mmvd {
                base,
                step,
                direction,
            }) => Some(super::motion_refine::mmvd(
                candidates()[usize::from(base)],
                (step, direction),
                header,
                (references, self.poc),
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
                    let tools = &sequence.tools;
                    let list =
                        affine.merge(tools.affine, tools.affine_six, tools.max_subblock_merge);
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
#[inline]
pub(super) fn at(unit: &Unit, x: usize, y: usize) -> Option<Motion> {
    let motion = unit.motion?;
    if unit.subblocks.is_empty()
        && unit.geometric.is_none()
        && unit.affine_fields[0].is_none()
        && unit.affine_fields[1].is_none()
    {
        return Some(motion);
    }
    at_with_fields(unit, x, y, &unit.affine_fields)
}

#[inline]
pub(super) fn at_with_fields(
    unit: &Unit,
    x: usize,
    y: usize,
    fields: &[Option<crate::affine::Field>; 2],
) -> Option<Motion> {
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
    for (vector, field) in motion.lists.iter_mut().zip(fields) {
        if let (Some(vector), Some(field)) = (vector, field) {
            vector.vector = field.at((x - n.x) / 4, (y - n.y) / 4);
        }
    }
    Some(motion)
}
