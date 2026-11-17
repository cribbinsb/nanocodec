//! Collocated motion retention and distance-scaled predictors, H.266 §8.5.2.
//! General reference metadata complements the shared sixteenth-sample motion type.
//! Candidate rules and compact MV precision follow BSD-Clear `VVdeC` `UnitTools`.
use super::{
    Node, Unit,
    motion::{Motion, Vector},
};
use crate::{decoder_headers::SliceHeader, motion::Mv16};
#[derive(Debug)]
pub(crate) struct Field {
    pub poc: i32,
    dimensions: [usize; 2],
    /// The reference POCs of each slice's two lists, by slice index.
    references: Vec<[Vec<i32>; 2]>,
    /// Whether each of those references is long-term, by slice index.
    long_term: Vec<[Vec<bool>; 2]>,
    /// The motion of each 8x8 cell and the index of its slice.
    cells: Vec<Option<(Motion, u16)>>,
}
impl Field {
    /// The motion of a decoded picture whose slices' reference lists are
    /// `references` (by slice index).
    pub fn new(
        units: &[Unit],
        dimensions: [usize; 2],
        poc: i32,
        (references, headers): (Vec<[Vec<i32>; 2]>, &[SliceHeader]),
    ) -> Self {
        let long_term = headers
            .iter()
            .zip(&references)
            .map(|(header, lists)| {
                std::array::from_fn(|list| {
                    (0..lists[list].len())
                        .map(|index| header.long_term(list, index as u8))
                        .collect()
                })
            })
            .collect();
        let stride = dimensions[0].div_ceil(8);
        let mut cells = vec![None; stride * dimensions[1].div_ceil(8)];
        for unit in units.iter().filter(|u| u.channel == 0) {
            let Some(base_motion) = unit.motion else {
                continue;
            };
            let n = unit.node;
            if unit.dmvr.is_empty()
                && unit.subblocks.is_empty()
                && unit.geometric.is_none()
                && unit.affine_models[0].is_none()
                && unit.affine_models[1].is_none()
            {
                let (x0, x1) = (n.x.div_ceil(8), (n.x + n.width).div_ceil(8));
                let (y0, y1) = (n.y.div_ceil(8), (n.y + n.height).div_ceil(8));
                if x0 < x1 && y0 < y1 {
                    let entry = Some((base_motion, unit.slice));
                    for cy in y0..y1 {
                        cells[cy * stride + x0..cy * stride + x1].fill(entry);
                    }
                }
                continue;
            }
            let bidirectional = base_motion.lists.iter().all(Option::is_some);
            let fields = unit.affine_models.map(|m| {
                m.map(|model| model.field_for_prediction(n.width, n.height, bidirectional))
            });
            for y in (n.y.div_ceil(8) * 8..n.y + n.height).step_by(8) {
                for x in (n.x.div_ceil(8) * 8..n.x + n.width).step_by(8) {
                    cells[y / 8 * stride + x / 8] =
                        super::refinement::temporal_with_fields(unit, x, y, fields)
                            .map(|m| (m, unit.slice));
                }
            }
        }
        Self {
            poc,
            dimensions,
            references,
            long_term,
            cells,
        }
    }
}
pub(crate) struct Temporal<'a> {
    field: &'a Field,
    poc: i32,
    references: [&'a [i32]; 2],
    /// Whether each of the current slice's references is long-term.
    long_term: [Vec<bool>; 2],
    selected_list: usize,
    low_delay: bool,
    ctu_log2: u8,
    /// The first column and row past the collocated positions the slice may
    /// use: the picture's, or its subpicture's when that is treated as a
    /// picture (§8.5.2.12, §8.5.5.4).
    limits: [usize; 2],
}
impl<'a> Temporal<'a> {
    /// Keep collocated positions inside the subpicture whose last luma
    /// column and row are `area[2]` and `area[3]`, if any.
    pub fn within(mut self, area: Option<[i32; 4]>) -> Self {
        if let Some([_, _, right, bottom]) = area {
            self.limits = [right as usize + 1, bottom as usize + 1];
        }
        self
    }
    pub fn new(
        field: &'a Field,
        poc: i32,
        references: [&'a [i32]; 2],
        header: &SliceHeader,
        ctu_log2: u8,
    ) -> Self {
        Self {
            field,
            poc,
            references,
            long_term: std::array::from_fn(|list| {
                (0..references[list].len())
                    .map(|index| header.long_term(list, index as u8))
                    .collect()
            }),
            selected_list: usize::from(header.collocated[0]),
            low_delay: references.iter().flat_map(|r| r.iter()).all(|&p| p <= poc),
            ctu_log2,
            limits: field.dimensions,
        }
    }
    fn at(&self, x: usize, y: usize, list: usize, reference: u8, subblock: bool) -> Option<Mv16> {
        if x >= self.limits[0] || y >= self.limits[1] {
            return None;
        }
        let width = self.field.dimensions[0];
        let (cell, slice) = self.field.cells[y / 8 * width.div_ceil(8) + x / 8]?;
        let source_list = if self.low_delay || subblock {
            list
        } else {
            1 - self.selected_list
        };
        let (source_list, vector) = if let Some(v) = cell.lists[source_list] {
            (source_list, v)
        } else if subblock && !self.low_delay {
            return None;
        } else {
            (1 - source_list, cell.lists[1 - source_list]?)
        };
        let (slice, index) = (usize::from(slice), usize::from(vector.reference));
        let source = self.field.references[slice][source_list][index];
        let target = *self.references[list].get(usize::from(reference))?;
        // A long-term reference only predicts from a long-term one, unscaled
        // (H.266 §8.5.2.12).
        let long_term = self.long_term[list][usize::from(reference)];
        if long_term != self.field.long_term[slice][source_list][index] {
            return None;
        }
        let vector = Mv16 {
            x: storage_precision(vector.vector.x),
            y: storage_precision(vector.vector.y),
        };
        if long_term {
            return Some(vector.clip());
        }
        super::motion_refine::scale(vector, self.poc - target, self.field.poc - source).ok()
    }
    pub(super) fn candidate(&self, n: Node, list: usize, reference: u8) -> Option<Mv16> {
        if n.width + n.height <= 12 {
            return None;
        }
        self.bottom_right(n, list, reference).or_else(|| {
            self.at(
                n.x + n.width / 2,
                n.y + n.height / 2,
                list,
                reference,
                false,
            )
        })
    }
    pub(super) fn bottom_right(&self, n: Node, list: usize, reference: u8) -> Option<Mv16> {
        let (x, y) = (n.x + n.width, n.y + n.height);
        if y >> self.ctu_log2 != n.y >> self.ctu_log2 {
            return None;
        }
        self.at(x, y, list, reference, false)
    }
    pub(super) fn subblocks(&self, n: Node, left: Option<Motion>) -> Option<Vec<Motion>> {
        let offset = left
            .and_then(|m| {
                (0..2).find_map(|list| {
                    let v = m.lists[list]?;
                    (self.references[list][usize::from(v.reference)] == self.field.poc)
                        .then_some(v.vector)
                })
            })
            .unwrap_or_default()
            .round(4);
        let origin = [
            n.x >> self.ctu_log2 << self.ctu_log2,
            n.y >> self.ctu_log2 << self.ctu_log2,
        ];
        let maximum = [
            (origin[0] + (1 << self.ctu_log2) + 3).min(self.limits[0] - 1),
            (origin[1] + (1 << self.ctu_log2) - 1).min(self.limits[1] - 1),
        ];
        let at = |x: usize, y: usize| {
            let x =
                (x as i32 + (offset.x >> 4)).clamp(origin[0] as i32, maximum[0] as i32) as usize;
            let y =
                (y as i32 + (offset.y >> 4)).clamp(origin[1] as i32, maximum[1] as i32) as usize;
            let motion = Motion {
                lists: std::array::from_fn(|list| {
                    self.at(x, y, list, 0, true).map(|vector| Vector {
                        reference: 0,
                        vector,
                    })
                }),
                ..Motion::default()
            };
            motion.lists.iter().any(Option::is_some).then_some(motion)
        };
        let default = at(n.x + n.width / 2, n.y + n.height / 2)?;
        let mut cells = Vec::with_capacity(n.width * n.height / 64);
        for y in (n.y..n.y + n.height).step_by(8) {
            for x in (n.x..n.x + n.width).step_by(8) {
                cells.push(at(x + 4, y + 4).unwrap_or(default));
            }
        }
        Some(cells)
    }
    pub(super) fn merge(&self, n: Node) -> Option<Motion> {
        let motion = Motion {
            lists: std::array::from_fn(|list| {
                self.candidate(n, list, 0).map(|vector| Vector {
                    reference: 0,
                    vector,
                })
            }),
            ..Motion::default()
        };
        motion.lists.iter().any(Option::is_some).then_some(motion)
    }
}
fn storage_precision(value: i32) -> i32 {
    let magnitude = (value ^ (value >> 31)) as u32;
    let shift = ((magnitude | 31).ilog2() + 1).saturating_sub(6);
    ((value + ((1 << shift) >> 1)) >> shift) << shift
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stored_precision_preserves_small_vectors_and_rounds_large_signed_vectors() {
        for value in -32..=31 {
            assert_eq!(storage_precision(value), value);
        }
        assert_eq!(storage_precision(127), 128);
        assert_eq!(storage_precision(-127), -126);
        assert_eq!(storage_precision(131_071), 131_072);
        assert_eq!(
            super::super::motion_refine::scale(
                Mv16 {
                    x: 131_072,
                    y: -131_072
                },
                1,
                1
            )
            .unwrap(),
            Mv16 {
                x: 131_071,
                y: -131_072
            }
        );
    }
    #[test]
    fn reference_distance_scaling_handles_future_references() {
        let field = Field {
            poc: 2,
            dimensions: [8, 8],
            references: vec![[vec![0], vec![]]],
            long_term: vec![[vec![false], vec![]]],
            cells: vec![Some((
                Motion {
                    lists: [
                        Some(Vector {
                            reference: 0,
                            vector: Mv16 { x: 64, y: -32 },
                        }),
                        None,
                    ],
                    ..Motion::default()
                },
                0,
            ))],
        };
        let temporal = Temporal {
            field: &field,
            poc: 1,
            references: [&[0], &[2]],
            long_term: [vec![false], vec![false]],
            selected_list: 1,
            low_delay: false,
            ctu_log2: 7,
            limits: [8, 8],
        };
        assert_eq!(temporal.at(0, 0, 0, 0, false), Some(Mv16 { x: 32, y: -16 }));
        assert_eq!(temporal.at(0, 0, 1, 0, false), Some(Mv16 { x: -32, y: 16 }));
        assert_eq!(temporal.at(0, 0, 1, 0, true), None);
        assert_eq!(temporal.at(8, 0, 0, 0, false), None);
    }
}
