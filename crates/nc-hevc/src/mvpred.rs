//! Motion vector prediction for P slices with one or two reference
//! pictures (the second is `refs.second`'s high-quality frame): the merge
//! candidate list of `MaxNumMergeCand` entries (HEVC §8.5.3.2.2–8.5.3.2.5)
//! and the AMVP predictor list (§8.5.3.2.6–8.5.3.2.7) of a 2Nx2N prediction
//! unit, with temporal (TMVP) candidates from the collocated picture, the
//! slice's first reference (§8.5.3.2.8).
//!
//! Every reference is a short-term picture, so a neighbour pointing at the
//! other reference gives an AMVP candidate scaled by POC distance, and two
//! merge candidates are the same motion when their vectors and reference
//! indices are equal. Temporal vectors are scaled by POC distance when the
//! collocated picture's own reference is at a different distance
//! (`collocated.rs`). The derivation follows our decoder's
//! (`dec/mvpred.rs`), which is checked against HM.

use nc_encoder::Mv;

use crate::collocated::{Collocated, MAX_REF_IDX, scale_mv};
use crate::grid::Grid;
use crate::headers::LOG2_CTB;

/// Collocated motion is stored once per 16x16 block (§8.5.3.2.8: the
/// position is rounded down to a multiple of 16).
const COL_LOG2: usize = 4;

/// The largest `MaxNumMergeCand` (§7.4.7.1: `five_minus_max_num_merge_cand`
/// is 0..=4).
pub const MAX_MERGE_CANDIDATES: usize = 5;
/// `MaxNumMergeCand` unless `hevc.merge_candidates` says otherwise.
pub const DEFAULT_MERGE_CANDIDATES: usize = 2;
/// AMVP lists always have two entries.
pub const AMVP_CANDIDATES: usize = 2;

/// The motion of a P-slice prediction unit: its vector and `ref_idx_l0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Motion {
    pub mv: Mv,
    pub ref_idx: u8,
}

/// The slice's reference list: the POC distance from the current picture
/// to each `ref_idx_l0` entry (all short-term pictures).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceDistances {
    distances: [i32; MAX_REF_IDX],
    count: usize,
}

impl ReferenceDistances {
    /// A list of one reference at `distance`.
    #[must_use]
    pub fn one(distance: i32) -> Self {
        Self {
            distances: [distance; MAX_REF_IDX],
            count: 1,
        }
    }

    /// A list of two references: `primary` (`ref_idx_l0` 0) and the older
    /// `secondary`.
    #[must_use]
    pub fn two(primary: i32, secondary: i32) -> Self {
        Self {
            distances: [primary, secondary],
            count: MAX_REF_IDX,
        }
    }

    /// `num_ref_idx_l0_active`.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// The distance to `ref_idx`.
    #[must_use]
    pub fn distance(&self, ref_idx: u8) -> i32 {
        self.distances[usize::from(ref_idx)]
    }

    /// The distances by `ref_idx_l0` (unused entries repeat the first).
    #[must_use]
    pub fn by_index(&self) -> [i32; MAX_REF_IDX] {
        self.distances
    }
}

/// The five spatial neighbours of a PU (§8.5.3.2.3, Figure 8-3): their
/// motion where they are available (§6.4.2) and inter coded.
struct Spatial {
    a0: Option<Motion>,
    a1: Option<Motion>,
    b0: Option<Motion>,
    b1: Option<Motion>,
    b2: Option<Motion>,
}

impl Spatial {
    fn of(grid: &Grid, x: usize, y: usize, size: usize) -> Self {
        // A1, B1 and B2 border the PU and are decoded before it; only A0
        // (below-left) and B0 (above-right) need a z-scan test.
        let availability = grid.reference_availability(x, y, size);
        let motion = |xn: isize, yn: isize| {
            grid.reference_cell(&availability, xn, yn)
                .filter(|c| !c.intra)
                .map(|c| Motion {
                    mv: c.mv,
                    ref_idx: c.ref_idx,
                })
        };
        let (xi, yi, s) = (x as isize, y as isize, size as isize);
        Self {
            a0: motion(xi - 1, yi + s),
            a1: motion(xi - 1, yi + s - 1),
            b0: motion(xi + s, yi - 1),
            b1: motion(xi + s - 1, yi - 1),
            b2: motion(xi - 1, yi - 1),
        }
    }
}

/// The temporal candidate of the PU at `(x, y)` for a reference at
/// `cur_distance` (§8.5.3.2.8): the collocated picture's vector at the
/// bottom-right of the PU (if it lies in the picture and in the same CTB
/// row), otherwise at its centre. `None` if that block is intra or there is
/// no collocated picture.
#[must_use]
pub fn temporal_candidate(
    collocated: Option<&Collocated<'_>>,
    (x, y, size): (usize, usize, usize),
    cur_distance: i32,
) -> Option<Mv> {
    let collocated = collocated?;
    let col = collocated.grid;
    let at = |cx: usize, cy: usize| {
        let cell = col.at((cx >> COL_LOG2) << COL_LOG2, (cy >> COL_LOG2) << COL_LOG2);
        (!cell.intra).then(|| collocated.scale(cell.mv, cell.ref_idx, cur_distance))
    };
    let (br_x, br_y) = (x + size, y + size);
    let same_ctb_row = (y >> LOG2_CTB) == (br_y >> LOG2_CTB);
    let bottom_right = (same_ctb_row && br_x < col.width() && br_y < col.height())
        .then(|| at(br_x, br_y))
        .flatten();
    bottom_right.or_else(|| at(x + size / 2, y + size / 2))
}

/// A PU's merge candidate list: the first `len` entries (`MaxNumMergeCand`,
/// the slice header's value), vectors and reference indices kept apart so
/// that the vectors alone are a slice (the engine's candidate vectors).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeList {
    vectors: [Mv; MAX_MERGE_CANDIDATES],
    ref_indices: [u8; MAX_MERGE_CANDIDATES],
    len: usize,
}

impl MergeList {
    /// The candidates' vectors, in `merge_idx` order.
    #[must_use]
    pub fn vectors(&self) -> &[Mv] {
        &self.vectors[..self.len]
    }

    /// Candidate `index`.
    #[must_use]
    pub fn get(&self, index: usize) -> Motion {
        Motion {
            mv: self.vectors[index],
            ref_idx: self.ref_indices[index],
        }
    }

    /// The first index whose candidate is `motion`.
    #[must_use]
    #[inline]
    pub fn index_of(&self, motion: Motion) -> Option<usize> {
        (0..self.len).find(|&i| self.get(i) == motion)
    }

    /// The candidates, in `merge_idx` order.
    pub fn motions(&self) -> impl Iterator<Item = Motion> + '_ {
        (0..self.len).map(|i| self.get(i))
    }
}

/// The merge list and the AMVP list for reference `amvp_ref` of the 2Nx2N
/// PU at `(x, y)`, sharing one derivation of the spatial neighbours.
/// Exact; with region availability in `Spatial::of`, HEVC Fast P frames
/// fell 1.2% in instructions (callgrind, crowd09 QP 47).
#[must_use]
pub fn candidate_lists(
    grid: &Grid,
    col: Option<&Collocated<'_>>,
    (x, y, size): (usize, usize, usize),
    (merge_len, references, amvp_ref): (usize, &ReferenceDistances, u8),
) -> (MergeList, [Mv; AMVP_CANDIDATES]) {
    let spatial = Spatial::of(grid, x, y, size);
    let pu = (x, y, size);
    // The merge list's temporal candidate is for `ref_idx` 0; so is the AMVP
    // list's when it predicts the first reference: derive it once.
    let first_once = std::cell::OnceCell::new();
    let first = || *first_once.get_or_init(|| temporal_candidate(col, pu, references.distance(0)));
    let amvp_temporal = || {
        if amvp_ref == 0 {
            first()
        } else {
            temporal_candidate(col, pu, references.distance(amvp_ref))
        }
    };
    (
        merge_list(&spatial, first, (merge_len, references.count())),
        amvp_list(&spatial, amvp_temporal, references, amvp_ref),
    )
}

/// The merge candidate list: spatial candidates A1, B1, B0, A0, B2 with
/// the specification's pruning, the temporal candidate (`ref_idx` 0), then
/// zero candidates (§8.5.3.2.5; the combined bi-predictive candidates are
/// B-slice only).
fn merge_list(
    n: &Spatial,
    temporal: impl FnOnce() -> Option<Mv>,
    (len, references): (usize, usize),
) -> MergeList {
    debug_assert!((1..=MAX_MERGE_CANDIDATES).contains(&len));
    // §8.5.3.2.3: B1 is pruned against A1, B0 against B1, A0 against A1,
    // and B2 against A1 and B1 (and dropped when the other four exist).
    let differs = |c: Option<Motion>, other: Option<Motion>| c.filter(|&v| Some(v) != other);
    let a1 = n.a1;
    let b1 = differs(n.b1, n.a1);
    let b0 = differs(n.b0, n.b1);
    let a0 = differs(n.a0, n.a1);
    let four = [a1, b1, b0, a0].iter().filter(|c| c.is_some()).count() == 4;
    let b2 = if four {
        None
    } else {
        differs(differs(n.b2, n.a1), n.b1)
    };
    // The temporal candidate matters only when the spatial ones leave room
    // in the list (§8.5.3.2.2: it follows them and is not pruned).
    let spatial = [a1, b1, b0, a0, b2];
    let room = spatial.iter().flatten().count() < len;
    let temporal = if room {
        temporal().map(|mv| Motion { mv, ref_idx: 0 })
    } else {
        None
    };
    // Entries past the spatial and temporal candidates are zero candidates
    // (§8.5.3.2.5), whose reference index steps through the list
    // (`zeroIdx`) and then stays 0: zero vectors on `ref_idx` 0 when the
    // slice has one reference.
    let mut list = MergeList {
        vectors: [Mv::ZERO; MAX_MERGE_CANDIDATES],
        ref_indices: [0; MAX_MERGE_CANDIDATES],
        len,
    };
    let mut filled = 0;
    for ((mv, ref_idx), motion) in list.vectors[..len]
        .iter_mut()
        .zip(&mut list.ref_indices)
        .zip(spatial.into_iter().chain([temporal]).flatten())
    {
        (*mv, *ref_idx) = (motion.mv, motion.ref_idx);
        filled += 1;
    }
    for (zero, ref_idx) in list.ref_indices[filled..len]
        .iter_mut()
        .enumerate()
        .take(references)
    {
        *ref_idx = zero as u8;
    }
    list
}

/// The AMVP list of reference `target` (§8.5.3.2.6, §8.5.3.2.7).
fn amvp_list(
    nb: &Spatial,
    temporal: impl FnOnce() -> Option<Mv>,
    references: &ReferenceDistances,
    target: u8,
) -> [Mv; AMVP_CANDIDATES] {
    // A neighbour pointing at the target picture gives its vector as it
    // is; one pointing at the other (short-term) reference, its vector
    // scaled by the two POC distances (eq. 8-179..8-183).
    let same = |m: &Motion| (m.ref_idx == target).then_some(m.mv);
    let scaled = |m: &Motion| {
        Some(scale_mv(
            m.mv,
            references.distance(m.ref_idx),
            references.distance(target),
        ))
    };
    let a = [nb.a0, nb.a1];
    let b = [nb.b0, nb.b1, nb.b2];
    // §8.5.3.2.7: A is the first of A0, A1 pointing at the target, else
    // the first available scaled; B the first of B0, B1, B2 pointing at the
    // target. When neither A0 nor A1 is available (isScaledFlag 0), A takes
    // B's vector and B is re-derived allowing scaled candidates.
    let (mv_a, mv_b) = if references.count() == 1 {
        // Every neighbour points at the one reference: A and B are the first
        // available of each group, and B re-derived equals B. (The general
        // path gives the same; this one keeps single-reference slices as
        // cheap as before the second reference existed.)
        let mv_b = nb.b0.or(nb.b1).or(nb.b2).map(|m| m.mv);
        (nb.a0.or(nb.a1).map(|m| m.mv).or(mv_b), mv_b)
    } else {
        let is_scaled = a.iter().any(Option::is_some);
        let mut mv_a = a
            .iter()
            .flatten()
            .find_map(same)
            .or_else(|| a.iter().flatten().find_map(scaled));
        let mut mv_b = b.iter().flatten().find_map(same);
        if !is_scaled {
            if mv_a.is_none() {
                mv_a = mv_b;
            }
            mv_b = b.iter().flatten().find_map(scaled);
        }
        (mv_a, mv_b)
    };
    // A, then B unless equal to A; the temporal vector only when that
    // leaves room; zero vectors after.
    let mut list = [Mv::ZERO; AMVP_CANDIDATES];
    let spatial = match (mv_a, mv_b) {
        (Some(a), Some(b)) if a != b => [Some(a), Some(b)],
        (Some(only), _) | (None, Some(only)) => [Some(only), None],
        (None, None) => [None, None],
    };
    let temporal = if spatial[1].is_some() {
        None
    } else {
        temporal()
    };
    for (slot, mv) in list
        .iter_mut()
        .zip(spatial.into_iter().chain([temporal]).flatten())
    {
        *slot = mv;
    }
    list
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Cell;

    fn inter(mv: Mv) -> Cell {
        inter_ref(mv, 0)
    }

    fn inter_ref(mv: Mv, ref_idx: u8) -> Cell {
        Cell {
            mv,
            ref_idx,
            ..Cell::default()
        }
    }

    fn on_first(mv: Mv) -> Motion {
        Motion { mv, ref_idx: 0 }
    }

    fn lists(g: &Grid, at: (usize, usize, usize), len: usize) -> (MergeList, [Mv; 2]) {
        candidate_lists(g, None, at, (len, &ReferenceDistances::one(1), 0))
    }

    #[test]
    fn merge_list_prunes_and_pads_with_zero() {
        let mut g = Grid::new(64, 64);
        let v = Mv { x: 4, y: -8 };
        // Left and above neighbours with the same vector: only one entry.
        g.fill(0, 8, 8, inter(v));
        g.fill(8, 0, 8, inter(v));
        let lists = lists(&g, (8, 8, 8), DEFAULT_MERGE_CANDIDATES);
        assert_eq!(
            lists.0.motions().collect::<Vec<_>>(),
            [on_first(v), on_first(Mv::ZERO)]
        );
    }

    #[test]
    fn longer_merge_lists_take_more_spatial_candidates() {
        let mut g = Grid::new(64, 64);
        let (left, above, corner) = (Mv { x: 4, y: 0 }, Mv { x: 0, y: 4 }, Mv { x: 8, y: 8 });
        g.fill(0, 8, 8, inter(left));
        g.fill(8, 0, 8, inter(above));
        g.fill(0, 0, 8, inter(corner));
        // A1, B1, then B2 (A0 and B0 are not yet decoded or are outside),
        // then a zero candidate.
        let long = lists(&g, (8, 8, 8), MAX_MERGE_CANDIDATES - 1);
        let expected = [left, above, corner, Mv::ZERO].map(on_first);
        assert_eq!(long.0.motions().collect::<Vec<_>>(), expected);
        assert_eq!(long.0.index_of(on_first(corner)), Some(2));
        // Two candidates keep the first two.
        let short = lists(&g, (8, 8, 8), DEFAULT_MERGE_CANDIDATES);
        assert_eq!(
            short.0.motions().collect::<Vec<_>>(),
            [on_first(left), on_first(above)]
        );
    }

    #[test]
    fn amvp_uses_above_when_left_is_missing() {
        let mut g = Grid::new(64, 64);
        let v = Mv { x: 1, y: 2 };
        g.fill(0, 0, 16, inter(v));
        // Block (0, 8): no left neighbours, B0 = B1 = v; A takes B's vector
        // and the duplicate is removed.
        let lists = lists(&g, (0, 8, 8), DEFAULT_MERGE_CANDIDATES);
        assert_eq!(lists.1, [v, Mv::ZERO]);
    }

    #[test]
    fn intra_neighbours_are_unavailable() {
        let mut g = Grid::new(64, 64);
        g.fill(
            0,
            8,
            8,
            Cell {
                intra: true,
                ..Cell::default()
            },
        );
        let lists = lists(&g, (8, 8, 8), DEFAULT_MERGE_CANDIDATES);
        assert_eq!(
            lists.0.motions().collect::<Vec<_>>(),
            [on_first(Mv::ZERO); 2]
        );
    }

    #[test]
    fn merge_candidates_carry_their_reference_and_zeros_step_through_both() {
        let mut g = Grid::new(64, 64);
        let v = Mv { x: 4, y: -8 };
        // The same vector on different references is different motion.
        g.fill(0, 8, 8, inter_ref(v, 1));
        g.fill(8, 0, 8, inter_ref(v, 0));
        let refs = ReferenceDistances::two(1, 8);
        let (merge, _) = candidate_lists(&g, None, (8, 8, 8), (5, &refs, 0));
        let zero = |ref_idx| Motion {
            mv: Mv::ZERO,
            ref_idx,
        };
        // A1, B1, then B2 (the decoded zero-motion cell above-left), then
        // the zero candidates with zeroIdx 0 and 1.
        assert_eq!(
            merge.motions().collect::<Vec<_>>(),
            [
                Motion { mv: v, ref_idx: 1 },
                Motion { mv: v, ref_idx: 0 },
                zero(0),
                zero(0),
                zero(1)
            ]
        );
    }

    #[test]
    fn amvp_scales_a_neighbour_on_the_other_reference() {
        let mut g = Grid::new(64, 64);
        // Left neighbour points at the second reference, 4 pictures back.
        g.fill(0, 8, 8, inter_ref(Mv { x: 16, y: -8 }, 1));
        let refs = ReferenceDistances::two(1, 4);
        // For ref_idx 0 (1 picture back) its vector is scaled by 1/4; for
        // ref_idx 1 it is used as it is.
        let (_, first) = candidate_lists(&g, None, (8, 8, 8), (2, &refs, 0));
        assert_eq!(first, [Mv { x: 4, y: -2 }, Mv::ZERO]);
        let (_, second) = candidate_lists(&g, None, (8, 8, 8), (2, &refs, 1));
        assert_eq!(second, [Mv { x: 16, y: -8 }, Mv::ZERO]);
    }
}
