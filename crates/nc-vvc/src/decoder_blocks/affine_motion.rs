//! Reference-aware affine candidate control for general decoded pictures.
//! Uses the shared affine model, sub-block field and normative rounding helper.
//! Candidate conditions follow BSD-Clear `VVdeC` `UnitTools` (§8.5.5).
use super::{
    Node,
    inter_syntax::Prediction,
    motion::{Motion, Vector},
};
use crate::{
    affine::{Model, round_shift},
    motion::Mv16,
};
#[derive(Clone, Copy)]
pub(super) struct Neighbour {
    pub node: Node,
    pub motion: Motion,
    pub models: [Option<Model>; 2],
}
struct Corners {
    left: [(usize, usize); 2],
    above: [(usize, usize); 3],
    points: [Vec<(usize, usize)>; 3],
}
impl Corners {
    fn new(n: Node) -> Self {
        let (a0, a1, a2) = (
            (n.x.wrapping_sub(1), n.y + n.height),
            (n.x.wrapping_sub(1), n.y + n.height - 1),
            (n.x.wrapping_sub(1), n.y),
        );
        let (b0, b1, b2, b3) = (
            (n.x + n.width, n.y.wrapping_sub(1)),
            (n.x + n.width - 1, n.y.wrapping_sub(1)),
            (n.x.wrapping_sub(1), n.y.wrapping_sub(1)),
            (n.x, n.y.wrapping_sub(1)),
        );
        Self {
            left: [a0, a1],
            above: [b0, b1, b2],
            points: [vec![b2, b3, a2], vec![b1, b0], vec![a1, a0]],
        }
    }
}
pub(super) struct Candidates<'a, F> {
    pub node: Node,
    pub references: [&'a [i32]; 2],
    pub neighbours: &'a F,
    pub ctu_log2: u8,
    pub merge_log2: u8,
    pub temporal: Option<&'a super::Temporal<'a>>,
}
impl<F: Fn(usize, usize) -> Option<Neighbour>> Candidates<'_, F> {
    pub fn amvp(&self, six: bool, precision: u8, prediction: Prediction, list: usize) -> Model {
        let (node, references, neighbours, ctu_log2) =
            (self.node, self.references, self.neighbours, self.ctu_log2);
        let shift = [2, 0, 4][usize::from(precision)];
        let corners = Corners::new(node);
        let same = |n: Neighbour, source: usize| {
            n.motion.lists[source].is_some_and(|v| {
                references[source][usize::from(v.reference)]
                    == references[list][usize::from(prediction.reference)]
            })
        };
        let mut candidates = Vec::with_capacity(2);
        for positions in [&corners.left[..], &corners.above[..]] {
            let model = positions.iter().find_map(|&(x, y)| {
                let n = neighbours(x, y)?;
                [list, 1 - list].into_iter().find_map(|source| {
                    n.models[source].filter(|_| same(n, source)).map(|model| {
                        inherit(node, n.node, model, six, source, neighbours, ctu_log2)
                    })
                })
            });
            if let Some(model) = model {
                candidates.push(round(model, shift));
            }
        }
        if candidates.len() < 2 {
            let points = corners.points.map(|positions| {
                positions.into_iter().find_map(|(x, y)| {
                    let n = neighbours(x, y)?;
                    [list, 1 - list].into_iter().find_map(|source| {
                        n.motion.lists[source]
                            .filter(|_| same(n, source))
                            .map(|v| v.vector.round(shift))
                    })
                })
            });
            if points[..if six { 3 } else { 2 }]
                .iter()
                .all(Option::is_some)
            {
                candidates.push(Model {
                    cp: points.map(Option::unwrap_or_default),
                    six,
                });
            }
            for vector in points.into_iter().rev().flatten() {
                if candidates.len() == 2 {
                    break;
                }
                candidates.push(Model {
                    cp: [vector; 3],
                    six,
                });
            }
        }
        if candidates.len() < 2
            && let Some(vector) = self
                .temporal
                .and_then(|t| t.candidate(node, list, prediction.reference))
        {
            candidates.push(Model {
                cp: [vector.round(shift); 3],
                six,
            });
        }
        candidates.resize(
            2,
            Model {
                cp: [Mv16::ZERO; 3],
                six,
            },
        );
        let mut model = candidates[usize::from(prediction.mvp)];
        for point in 0..model.points() {
            let delta = prediction.mvd[point];
            let first = if point == 0 {
                [0; 2]
            } else {
                prediction.mvd[0]
            };
            model.cp[point] = Mv16 {
                x: super::motion::wrap_component(
                    model.cp[point].x + ((delta[0] + first[0]) << shift),
                ),
                y: super::motion::wrap_component(
                    model.cp[point].y + ((delta[1] + first[1]) << shift),
                ),
            };
        }
        model
    }
}
fn round(mut model: Model, shift: u32) -> Model {
    if shift != 0 {
        model.cp = model.cp.map(|v| v.round(shift));
    }
    model
}
fn inherit(
    node: Node,
    previous: Node,
    model: Model,
    six: bool,
    list: usize,
    neighbours: &impl Fn(usize, usize) -> Option<Neighbour>,
    ctu_log2: u8,
) -> Model {
    let [mut lt, mut rt, lb] = model.cp;
    let mut origin_y = previous.y;
    let bottom = previous.y + previous.height;
    let boundary = bottom.is_multiple_of(1 << ctu_log2) && bottom == node.y;
    if boundary {
        lt = neighbours(previous.x, bottom - 1).unwrap().motion.lists[list]
            .unwrap()
            .vector;
        rt = neighbours(previous.x + previous.width - 1, bottom - 1)
            .unwrap()
            .motion
            .lists[list]
            .unwrap()
            .vector;
        origin_y = bottom;
    }
    let scale = |v: i32, size: usize| i64::from(v) << (7 - size.ilog2());
    let hor = (
        scale(rt.x - lt.x, previous.width),
        scale(rt.y - lt.y, previous.width),
    );
    let ver = if model.six && !boundary {
        (
            scale(lb.x - lt.x, previous.height),
            scale(lb.y - lt.y, previous.height),
        )
    } else {
        (-hor.1, hor.0)
    };
    let at = |x: usize, y: usize| {
        let (dx, dy) = (x as i64 - previous.x as i64, y as i64 - origin_y as i64);
        Mv16 {
            x: round_shift((i64::from(lt.x) << 7) + hor.0 * dx + ver.0 * dy, 7),
            y: round_shift((i64::from(lt.y) << 7) + hor.1 * dx + ver.1 * dy, 7),
        }
        .clip()
    };
    Model {
        cp: [
            at(node.x, node.y),
            at(node.x + node.width, node.y),
            if six {
                at(node.x, node.y + node.height)
            } else {
                Mv16::ZERO
            },
        ],
        six,
    }
}

impl<F: Fn(usize, usize) -> Option<Neighbour>> Candidates<'_, F> {
    pub fn merge(&self, six: bool, maximum: u8) -> Vec<(Motion, [Option<Model>; 2])> {
        let (node, neighbours, ctu_log2, merge_log2) =
            (self.node, self.neighbours, self.ctu_log2, self.merge_log2);
        let references = self.references.map(<[i32]>::len);
        let corners = Corners::new(node);
        let eligible = |x: usize, y: usize| {
            if x >> merge_log2 == node.x >> merge_log2 && y >> merge_log2 == node.y >> merge_log2 {
                None
            } else {
                neighbours(x, y)
            }
        };
        let mut candidates = Vec::new();
        // Inherited affine candidates keep both directional slots even when
        // they refer to the same CU (H.266 §8.5.5.3).
        for positions in [&corners.left[..], &corners.above[..]] {
            if let Some(n) = positions
                .iter()
                .find_map(|&(x, y)| eligible(x, y).filter(|n| n.models.iter().any(Option::is_some)))
            {
                let models = std::array::from_fn(|list| {
                    n.models[list]
                        .map(|m| inherit(node, n.node, m, m.six, list, neighbours, ctu_log2))
                });
                let mut motion = n.motion;
                for (vector, model) in motion.lists.iter_mut().zip(models) {
                    if let (Some(vector), Some(model)) = (vector, model) {
                        vector.vector = model.cp[0];
                    }
                }
                motion.half = false;
                candidates.push((motion, models));
            }
        }
        let spatial = corners.points.map(|positions| {
            positions
                .into_iter()
                .find_map(|(x, y)| eligible(x, y))
                .map(|n| n.motion)
        });
        let temporal = self.temporal.map(|t| Motion {
            lists: std::array::from_fn(|list| {
                t.bottom_right(node, list, 0).map(|vector| Vector {
                    reference: 0,
                    vector,
                })
            }),
            ..Motion::default()
        });
        let points = [spatial[0], spatial[1], spatial[2], temporal];
        for indices in [
            &[0, 1, 2][..],
            &[0, 1, 3][..],
            &[0, 2, 3][..],
            &[1, 2, 3][..],
            &[0, 1][..],
            &[0, 2][..],
        ] {
            if indices.len() == 3 && !six {
                continue;
            }
            candidates.extend(constructed(node, points, indices));
        }

        candidates.truncate(usize::from(maximum));
        let zero = Model {
            cp: [Mv16::ZERO; 3],
            six: false,
        };
        candidates.resize(
            usize::from(maximum),
            (
                Motion {
                    lists: references.map(|count| {
                        (count != 0).then_some(Vector {
                            reference: 0,
                            vector: Mv16::ZERO,
                        })
                    }),
                    ..Motion::default()
                },
                references.map(|count| (count != 0).then_some(zero)),
            ),
        );
        candidates
    }
}

fn constructed(
    node: Node,
    points: [Option<Motion>; 4],
    indices: &[usize],
) -> Option<(Motion, [Option<Model>; 2])> {
    if indices.iter().any(|&i| points[i].is_none()) {
        return None;
    }
    let mut motion = Motion::default();
    let mut models = [None; 2];
    for (list, model) in models.iter_mut().enumerate() {
        let first = points[indices[0]].unwrap().lists[list];
        let Some(first) = first else {
            continue;
        };
        if !indices.iter().all(|&i| {
            points[i].unwrap().lists[list].is_some_and(|v| v.reference == first.reference)
        }) {
            continue;
        }
        let mut cp = points.map(|p| {
            p.and_then(|p| p.lists[list])
                .map_or(Mv16::ZERO, |v| v.vector)
        });
        if indices.len() == 3 {
            let missing = (0..4).find(|i| !indices.contains(i)).unwrap();
            let (a, b, c) = match missing {
                0 => (1, 2, 3),
                1 => (3, 0, 2),
                2 => (3, 0, 1),
                _ => (0, 0, 0),
            };
            if missing < 3 {
                cp[missing] = Mv16 {
                    x: cp[a].x + cp[b].x - cp[c].x,
                    y: cp[a].y + cp[b].y - cp[c].y,
                }
                .clip();
            }
        }
        if indices == [0, 2] {
            let factor = 7 + node.width.ilog2() - node.height.ilog2();
            cp[1] = Mv16 {
                x: round_shift(
                    (i64::from(cp[0].x) << 7) + (i64::from(cp[2].y - cp[0].y) << factor),
                    7,
                ),
                y: round_shift(
                    (i64::from(cp[0].y) << 7) - (i64::from(cp[2].x - cp[0].x) << factor),
                    7,
                ),
            }
            .clip();
        }
        motion.lists[list] = Some(Vector {
            reference: first.reference,
            vector: cp[0],
        });
        *model = Some(Model {
            cp: [cp[0], cp[1], cp[2]],
            six: indices.len() == 3,
        });
    }
    if motion.lists.iter().any(Option::is_some) {
        if motion.lists.iter().all(Option::is_some) {
            motion.weight = points[usize::from(indices == [1, 2, 3])].unwrap().weight;
        }
        Some((motion, models))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn same_cu_keeps_both_directional_inherited_merge_slots() {
        let vector = Mv16 { x: 5, y: 9 };
        let neighbour = Neighbour {
            node: Node::root(0, 0, 32),
            motion: Motion {
                lists: [
                    Some(Vector {
                        reference: 0,
                        vector,
                    }),
                    None,
                ],
                ..Motion::default()
            },
            models: [
                Some(Model {
                    cp: [vector; 3],
                    six: false,
                }),
                None,
            ],
        };
        let neighbours = |x, y| (x < 32 && y < 32).then_some(neighbour);
        let candidates = Candidates {
            node: Node::root(32, 8, 8),
            references: [&[0], &[]],
            neighbours: &neighbours,
            ctu_log2: 7,
            merge_log2: 2,
            temporal: None,
        };
        let list = candidates.merge(true, 5);
        // Two inherited slots followed by the LT/LB constructed model. Removing
        // the repeated inherited CU shifts the zero padding into slot two.
        for candidate in &list[..3] {
            assert_eq!(candidate.0.lists[0].unwrap().vector, vector);
        }
        assert_eq!(list[3].0.lists[0].unwrap().vector, Mv16::ZERO);
    }
}
