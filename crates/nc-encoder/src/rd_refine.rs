//! Rate-distortion refinement of a final motion vector (x264's `subme` 8/9
//! "qpel RD"): the vector a cheap search chose is moved to whichever of its
//! quarter-sample neighbours codes with the lowest true cost, distortion
//! after quantisation plus λ times the entropy coder's bits.
//!
//! x264 walks until the centre stays; here the walk's size is the caller's
//! [`RdWalk`]. Only the walk lives here; the cost of a vector is codec syntax (its
//! residual, mode and vector bits), so the backend supplies it. Every codec
//! can call this with its own coded-cost estimate.

use crate::mv::Mv;

/// The four quarter-sample neighbours of a vector that share a row or a
/// column with it, in quarter samples.
const CROSS: [(i16, i16); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// The eight quarter-sample neighbours of a vector, in quarter samples.
const SQUARE: [(i16, i16); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Most steps any walk takes.
const MAX_STEPS: usize = 3;

/// Vectors one walk evaluates at most: the start and `MAX_STEPS` squares.
const MAX_VISITED: usize = 1 + MAX_STEPS * SQUARE.len();

/// How far a refinement walks. Every evaluation is a coded-cost estimate
/// (a transform, quantisation, reconstruction and entropy count), so the
/// walk's size is the stage's cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RdWalk {
    /// The eight neighbours of each centre rather than the four sharing a
    /// row or column with it.
    pub square: bool,
    /// Moves at most (1 to 3); the walk stops early where the centre stays.
    pub steps: usize,
}

impl RdWalk {
    /// The vectors one move from `centre` can reach.
    pub fn neighbours(self, centre: Mv) -> impl Iterator<Item = Mv> {
        let offsets: &[(i16, i16)] = if self.square { &SQUARE } else { &CROSS };
        offsets.iter().map(move |&(dx, dy)| Mv {
            x: centre.x.saturating_add(dx),
            y: centre.y.saturating_add(dy),
        })
    }
}

/// The vector of lowest `cost` reached from `start` (whose cost is
/// `start_cost`) by at most `walk.steps` moves to a quarter-sample
/// neighbour, and its cost. `cost` returns `None` for a vector the block
/// may not use (an intra refresh window, a picture-range limit); such a
/// vector is never chosen. No vector is evaluated twice.
pub fn rd_refine_vector(
    start: Mv,
    start_cost: f32,
    walk: RdWalk,
    mut cost: impl FnMut(Mv) -> Option<f32>,
) -> (Mv, f32) {
    let mut visited = [start; MAX_VISITED];
    let mut visited_len = 1;
    let mut best = (start, start_cost);
    for _ in 0..walk.steps.clamp(1, MAX_STEPS) {
        let centre = best.0;
        for mv in walk.neighbours(centre) {
            if visited[..visited_len].contains(&mv) {
                continue;
            }
            visited[visited_len] = mv;
            visited_len += 1;
            if let Some(c) = cost(mv).filter(|&c| c < best.1) {
                best = (mv, c);
            }
        }
        if best.0 == centre {
            break;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walks_to_the_minimum_within_two_steps() {
        let target = Mv { x: 2, y: -1 };
        let cost = |mv: Mv| {
            let (dx, dy) = (i32::from(mv.x - target.x), i32::from(mv.y - target.y));
            Some((dx * dx + dy * dy) as f32)
        };
        let start = Mv::ZERO;
        let walk = RdWalk {
            square: true,
            steps: 2,
        };
        let (mv, c) = rd_refine_vector(start, cost(start).unwrap(), walk, cost);
        assert_eq!((mv, c), (target, 0.0));
    }

    #[test]
    fn never_chooses_a_forbidden_vector_and_keeps_a_local_minimum() {
        let start = Mv { x: 4, y: 4 };
        let walk = RdWalk {
            square: false,
            steps: 1,
        };
        let (mv, c) = rd_refine_vector(start, 1.0, walk, |mv| (mv.x != 5).then_some(2.0));
        assert_eq!((mv, c), (start, 1.0));
        let (mv, _) = rd_refine_vector(start, 1.0, walk, |mv| (mv.x != 5).then_some(0.5));
        assert_ne!(mv.x, 5);
    }
}
