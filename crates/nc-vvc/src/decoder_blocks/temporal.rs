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
    references: [Vec<i32>; 2],
    cells: Vec<Option<Motion>>,
}
impl Field {
    pub fn new(
        units: &[Unit],
        dimensions: [usize; 2],
        poc: i32,
        references: [Vec<i32>; 2],
    ) -> Self {
        let stride = dimensions[0].div_ceil(8);
        let mut cells = vec![None; stride * dimensions[1].div_ceil(8)];
        for unit in units.iter().filter(|u| u.channel == 0) {
            let n = unit.node;
            for y in (n.y.div_ceil(8) * 8..n.y + n.height).step_by(8) {
                for x in (n.x.div_ceil(8) * 8..n.x + n.width).step_by(8) {
                    cells[y / 8 * stride + x / 8] = super::refinement::temporal(unit, x, y);
                }
            }
        }
        Self {
            poc,
            dimensions,
            references,
            cells,
        }
    }
}
pub(crate) struct Temporal<'a> {
    field: &'a Field,
    poc: i32,
    references: [&'a [i32]; 2],
    selected_list: usize,
    low_delay: bool,
    ctu_log2: u8,
}
impl<'a> Temporal<'a> {
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
            selected_list: usize::from(header.collocated[0]),
            low_delay: references.iter().flat_map(|r| r.iter()).all(|&p| p <= poc),
            ctu_log2,
        }
    }
    fn at(&self, x: usize, y: usize, list: usize, reference: u8, subblock: bool) -> Option<Mv16> {
        let [width, height] = self.field.dimensions;
        if x >= width || y >= height {
            return None;
        }
        let cell = self.field.cells[y / 8 * width.div_ceil(8) + x / 8]?;
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
        let source = self.field.references[source_list][usize::from(vector.reference)];
        let target = *self.references[list].get(usize::from(reference))?;
        let vector = Mv16 {
            x: storage_precision(vector.vector.x),
            y: storage_precision(vector.vector.y),
        };
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
            (origin[0] + (1 << self.ctu_log2) + 3).min(self.field.dimensions[0] - 1),
            (origin[1] + (1 << self.ctu_log2) - 1).min(self.field.dimensions[1] - 1),
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
            references: [vec![0], vec![]],
            cells: vec![Some(Motion {
                lists: [
                    Some(Vector {
                        reference: 0,
                        vector: Mv16 { x: 64, y: -32 },
                    }),
                    None,
                ],
                ..Motion::default()
            })],
        };
        let temporal = Temporal {
            field: &field,
            poc: 1,
            references: [&[0], &[2]],
            selected_list: 1,
            low_delay: false,
            ctu_log2: 7,
        };
        assert_eq!(temporal.at(0, 0, 0, 0, false), Some(Mv16 { x: 32, y: -16 }));
        assert_eq!(temporal.at(0, 0, 1, 0, false), Some(Mv16 { x: -32, y: 16 }));
        assert_eq!(temporal.at(0, 0, 1, 0, true), None);
        assert_eq!(temporal.at(8, 0, 0, 0, false), None);
    }
}
