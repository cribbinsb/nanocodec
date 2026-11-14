//! Motion vector prediction for P slices with one reference picture (which
//! is also the collocated picture): the merge candidate list of
//! `MaxNumMergeCand` entries (HEVC §8.5.3.2.2–8.5.3.2.5) and the AMVP predictor list (§8.5.3.2.6–8.5.3.2.7)
//! of a 2Nx2N prediction unit, with temporal (TMVP) candidates from the
//! reference picture's motion (§8.5.3.2.8).
//!
//! With a single reference, every available inter neighbour uses the same
//! reference picture: no AMVP scaling is ever needed, and two candidates
//! have the same motion exactly when their vectors are equal. Temporal
//! vectors are scaled by POC distance when the collocated picture's own
//! reference is at a different distance (`collocated.rs`).

use nc_encoder::Mv;

use crate::collocated::Collocated;
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

/// The five spatial neighbours of a PU (§8.5.3.2.3, Figure 8-3): their
/// vectors where they are available (§6.4.2) and inter coded.
struct Spatial {
    a0: Option<Mv>,
    a1: Option<Mv>,
    b0: Option<Mv>,
    b1: Option<Mv>,
    b2: Option<Mv>,
}

impl Spatial {
    fn of(grid: &Grid, x: usize, y: usize, size: usize) -> Self {
        // A1, B1 and B2 border the PU and are decoded before it; only A0
        // (below-left) and B0 (above-right) need a z-scan test.
        let availability = grid.reference_availability(x, y, size);
        let mv = |xn: isize, yn: isize| {
            grid.reference_cell(&availability, xn, yn)
                .filter(|c| !c.intra)
                .map(|c| c.mv)
        };
        let (xi, yi, s) = (x as isize, y as isize, size as isize);
        Self {
            a0: mv(xi - 1, yi + s),
            a1: mv(xi - 1, yi + s - 1),
            b0: mv(xi + s, yi - 1),
            b1: mv(xi + s - 1, yi - 1),
            b2: mv(xi - 1, yi - 1),
        }
    }
}

/// The temporal candidate of the PU at `(x, y)` (§8.5.3.2.8): the
/// collocated picture's vector at the bottom-right of the PU (if it lies in
/// the picture and in the same CTB row), otherwise at its centre. `None`
/// if that block is intra or there is no collocated picture.
#[must_use]
pub fn temporal_candidate(
    collocated: Option<&Collocated<'_>>,
    x: usize,
    y: usize,
    size: usize,
) -> Option<Mv> {
    let collocated = collocated?;
    let col = collocated.grid;
    let at = |cx: usize, cy: usize| {
        let cell = col.at((cx >> COL_LOG2) << COL_LOG2, (cy >> COL_LOG2) << COL_LOG2);
        (!cell.intra).then(|| collocated.scale(cell.mv))
    };
    let (br_x, br_y) = (x + size, y + size);
    let same_ctb_row = (y >> LOG2_CTB) == (br_y >> LOG2_CTB);
    let bottom_right = (same_ctb_row && br_x < col.width() && br_y < col.height())
        .then(|| at(br_x, br_y))
        .flatten();
    bottom_right.or_else(|| at(x + size / 2, y + size / 2))
}

/// A PU's merge candidate list: the first `len` entries of `candidates`
/// (`MaxNumMergeCand`, the slice header's value).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeList {
    candidates: [Mv; MAX_MERGE_CANDIDATES],
    len: usize,
}

impl MergeList {
    /// The list's candidates, in `merge_idx` order.
    #[must_use]
    pub fn as_slice(&self) -> &[Mv] {
        &self.candidates[..self.len]
    }

    /// The first index whose candidate is `mv`.
    #[must_use]
    pub fn index_of(&self, mv: Mv) -> Option<usize> {
        self.as_slice().iter().position(|&m| m == mv)
    }
}

/// The merge and AMVP lists of the 2Nx2N PU at `(x, y)`, sharing one
/// derivation of the spatial neighbours and of the temporal candidate.
/// Exact; with region availability in `Spatial::of`, HEVC Fast P frames
/// fell 1.2% in instructions (callgrind, crowd09 QP 47).
#[must_use]
pub fn candidate_lists(
    grid: &Grid,
    col: Option<&Collocated<'_>>,
    (x, y, size): (usize, usize, usize),
    merge_len: usize,
) -> (MergeList, [Mv; AMVP_CANDIDATES]) {
    let spatial = Spatial::of(grid, x, y, size);
    let temporal_once = std::cell::OnceCell::new();
    let temporal = || *temporal_once.get_or_init(|| temporal_candidate(col, x, y, size));
    (
        merge_list(&spatial, temporal, merge_len),
        amvp_list(&spatial, temporal),
    )
}

/// The merge candidate list: spatial candidates A1, B1, B0, A0, B2 with
/// the specification's pruning, the temporal candidate, then zero vectors
/// (§8.5.3.2.5; the combined bi-predictive candidates are B-slice only).
fn merge_list(n: &Spatial, temporal: impl FnOnce() -> Option<Mv>, len: usize) -> MergeList {
    debug_assert!((1..=MAX_MERGE_CANDIDATES).contains(&len));
    // §8.5.3.2.3: B1 is pruned against A1, B0 against B1, A0 against A1,
    // and B2 against A1 and B1 (and dropped when the other four exist).
    let differs = |c: Option<Mv>, other: Option<Mv>| c.filter(|&v| Some(v) != other);
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
    // Entries beyond the spatial and temporal candidates are zero
    // candidates (§8.5.3.2.5): with one reference every one has refIdx 0,
    // so they are all the zero vector.
    let mut candidates = [Mv::ZERO; MAX_MERGE_CANDIDATES];
    // The temporal candidate matters only when the spatial ones leave room
    // in the list (§8.5.3.2.2: it follows them and is not pruned).
    let spatial = [a1, b1, b0, a0, b2];
    let room = spatial.iter().flatten().count() < len;
    let temporal = if room { temporal() } else { None };
    for (slot, mv) in candidates[..len]
        .iter_mut()
        .zip(spatial.into_iter().chain([temporal]).flatten())
    {
        *slot = mv;
    }
    MergeList { candidates, len }
}

/// The AMVP list (§8.5.3.2.6).
fn amvp_list(nb: &Spatial, temporal: impl FnOnce() -> Option<Mv>) -> [Mv; AMVP_CANDIDATES] {
    // §8.5.3.2.7: A is the first available of A0, A1; B the first of B0,
    // B1, B2. When neither A0 nor A1 is available (isScaledFlag 0), A takes
    // B's vector and B is re-derived — to the same value, as every
    // neighbour shares the one reference picture.
    let is_scaled = nb.a0.is_some() || nb.a1.is_some();
    let mv_b = nb.b0.or(nb.b1).or(nb.b2);
    let mv_a = if is_scaled { nb.a0.or(nb.a1) } else { mv_b };
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
        Cell {
            mv,
            ..Cell::default()
        }
    }

    #[test]
    fn merge_list_prunes_and_pads_with_zero() {
        let mut g = Grid::new(64, 64);
        let v = Mv { x: 4, y: -8 };
        // Left and above neighbours with the same vector: only one entry.
        g.fill(0, 8, 8, inter(v));
        g.fill(8, 0, 8, inter(v));
        let lists = candidate_lists(&g, None, (8, 8, 8), DEFAULT_MERGE_CANDIDATES);
        assert_eq!(lists.0.as_slice(), [v, Mv::ZERO]);
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
        let lists = candidate_lists(&g, None, (8, 8, 8), MAX_MERGE_CANDIDATES - 1);
        assert_eq!(lists.0.as_slice(), [left, above, corner, Mv::ZERO]);
        assert_eq!(lists.0.index_of(corner), Some(2));
        // Two candidates keep the first two.
        let short = candidate_lists(&g, None, (8, 8, 8), DEFAULT_MERGE_CANDIDATES);
        assert_eq!(short.0.as_slice(), [left, above]);
    }

    #[test]
    fn amvp_uses_above_when_left_is_missing() {
        let mut g = Grid::new(64, 64);
        let v = Mv { x: 1, y: 2 };
        g.fill(0, 0, 16, inter(v));
        // Block (0, 8): no left neighbours, B0 = B1 = v; A takes B's vector
        // and the duplicate is removed.
        let lists = candidate_lists(&g, None, (0, 8, 8), DEFAULT_MERGE_CANDIDATES);
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
        let lists = candidate_lists(&g, None, (8, 8, 8), DEFAULT_MERGE_CANDIDATES);
        assert_eq!(lists.0.as_slice(), [Mv::ZERO; 2]);
    }
}
