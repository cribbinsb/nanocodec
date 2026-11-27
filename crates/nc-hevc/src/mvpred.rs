//! Motion vector prediction for P slices and generalised B slices (both
//! reference lists hold the same past pictures, in the same order) with one
//! or two reference pictures (the second is `refs.second`'s high-quality
//! frame): the merge candidate list of `MaxNumMergeCand` entries (HEVC
//! §8.5.3.2.2–8.5.3.2.5, with B slices' combined bi-predictive and
//! two-list zero candidates) and the AMVP predictor list of either list
//! (§8.5.3.2.6–8.5.3.2.7) of a 2Nx2N prediction unit, with temporal (TMVP)
//! candidates from the collocated picture, list 0's first reference
//! (§8.5.3.2.8).
//!
//! A neighbour pointing at the other reference gives an AMVP candidate
//! scaled by POC distance when both are short-term pictures, and two merge
//! candidates are the same motion when their vectors and reference indices
//! are equal. Both lists holding the same pictures, a reference index names
//! the same picture in either list. Temporal vectors are scaled by POC
//! distance when the collocated picture's own reference is at a different
//! distance (`collocated.rs`). A long-term reference (`ltr.interval`'s, the
//! only reference of a recovery picture) is never scaled, and a candidate
//! pointing at a long-term picture is unavailable for a short-term target
//! and the reverse (`LongTermRefPic`, §8.5.3.2.7, §8.5.3.2.8). The
//! derivation follows our decoder's (`dec/mvpred.rs`), which is checked
//! against HM.

use nc_encoder::Mv;

use crate::collocated::{Collocated, ListKinds, MAX_REF_IDX, scale_mv};
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
/// `l0CandIdx` / `l1CandIdx` of combined bi-predictive merge candidates
/// (§8.5.3.2.4, Table 8-6), as many as `numOrigMergeCand` allows.
const COMBINATIONS: [(usize, usize); 12] = [
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 2),
    (2, 1),
    (0, 3),
    (3, 0),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
];

/// Reference list 0 or 1 (`X` of `mvLX`, `refIdxLX`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum List {
    /// `L0`.
    L0,
    /// `L1` (B slices).
    L1,
}

/// One list's motion: a vector and its reference index.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ListMotion {
    pub mv: Mv,
    pub ref_idx: u8,
}

/// The motion of a prediction unit: list 0's vector and `ref_idx_l0` (every
/// inter unit coded here predicts from list 0), and list 1's of a
/// bi-predicted unit in a B slice.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Motion {
    pub mv: Mv,
    pub ref_idx: u8,
    /// List 1's motion (`PRED_BI`); `None` for `PRED_L0`.
    pub l1: Option<ListMotion>,
}

impl Motion {
    /// Uni-prediction from list 0 at `mv` into reference `ref_idx`.
    #[must_use]
    pub fn uni(mv: Mv, ref_idx: u8) -> Self {
        Self {
            mv,
            ref_idx,
            l1: None,
        }
    }

    /// This motion as uni-prediction from list 0, when it predicts the same
    /// samples: itself, or a bi-predicted unit whose two halves are the same
    /// picture and vector (default bi-prediction of two equal predictions,
    /// `(2a + 64) >> 7`, is uni-prediction's `(a + 32) >> 6`, HEVC eq.
    /// 8-250 and 8-251 at 8 bits).
    #[must_use]
    pub fn as_uni(self) -> Option<Self> {
        let uni = Self::uni(self.mv, self.ref_idx);
        match self.l1 {
            None => Some(self),
            Some(l1) if l1.mv == self.mv && l1.ref_idx == self.ref_idx => Some(uni),
            Some(_) => None,
        }
    }

    /// The motion of `list`, if the unit predicts from it.
    #[must_use]
    pub fn of(&self, list: List) -> Option<ListMotion> {
        match list {
            List::L0 => Some(ListMotion {
                mv: self.mv,
                ref_idx: self.ref_idx,
            }),
            List::L1 => self.l1,
        }
    }
}

/// The slice's reference list: the POC distance from the current picture
/// to each `ref_idx_l0` entry, and which entries are long-term pictures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceDistances {
    kinds: ListKinds,
    count: usize,
}

impl ReferenceDistances {
    /// A list of one short-term reference at `distance`.
    #[must_use]
    pub fn one(distance: i32) -> Self {
        Self {
            kinds: ListKinds::short_term([distance; MAX_REF_IDX]),
            count: 1,
        }
    }

    /// A list of two short-term references: `primary` (`ref_idx_l0` 0) and
    /// the older `secondary`.
    #[must_use]
    pub fn two(primary: i32, secondary: i32) -> Self {
        Self {
            kinds: ListKinds::short_term([primary, secondary]),
            count: MAX_REF_IDX,
        }
    }

    /// This list with entry `ref_idx` a long-term picture.
    #[must_use]
    pub fn with_long_term(mut self, ref_idx: u8) -> Self {
        self.kinds.long_term[usize::from(ref_idx)] = true;
        if self.count == 1 {
            // Unused entries repeat the first.
            self.kinds.long_term = [true; MAX_REF_IDX];
        }
        self
    }

    /// `num_ref_idx_l0_active`.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// The distance to `ref_idx`.
    #[must_use]
    pub fn distance(&self, ref_idx: u8) -> i32 {
        self.kinds.distances[usize::from(ref_idx)]
    }

    /// Whether `ref_idx` is a long-term picture.
    #[must_use]
    pub fn long_term(&self, ref_idx: u8) -> bool {
        self.kinds.long_term[usize::from(ref_idx)]
    }

    /// `ref_idx` as a target of temporal prediction: its distance, and
    /// whether it is long-term.
    #[must_use]
    pub fn target(&self, ref_idx: u8) -> (i32, bool) {
        (self.distance(ref_idx), self.long_term(ref_idx))
    }

    /// The entries by `ref_idx_l0` (unused entries repeat the first).
    #[must_use]
    pub fn by_index(&self) -> ListKinds {
        self.kinds
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
                .map(crate::grid::Cell::motion)
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

/// The temporal candidate of list `list` of the PU at `(x, y)` for a
/// reference `target` (its POC distance, and whether it is long-term;
/// §8.5.3.2.8): the collocated picture's
/// vector at the bottom-right of the PU (if it lies in the picture and in
/// the same CTB row), otherwise at its centre. `None` if that block is intra
/// or there is no collocated picture, or exactly one of its reference and
/// `target` is long-term. Every reference precedes the current
/// picture (`NoBackwardPredFlag` 1, §8.5.3.2.9), so a bi-predicted
/// collocated block gives list `list`'s vector and a uni-predicted one its
/// list 0 vector.
#[must_use]
pub fn temporal_candidate(
    collocated: Option<&Collocated<'_>>,
    (x, y, size): (usize, usize, usize),
    (target, list): ((i32, bool), List),
) -> Option<Mv> {
    let collocated = collocated?;
    let col = collocated.grid;
    let at = |cx: usize, cy: usize| {
        let cell = col.at((cx >> COL_LOG2) << COL_LOG2, (cy >> COL_LOG2) << COL_LOG2);
        if cell.intra {
            return None;
        }
        let source = cell.motion().of(list).unwrap_or(ListMotion {
            mv: cell.mv,
            ref_idx: cell.ref_idx,
        });
        collocated.scale(source.mv, source.ref_idx, target)
    };
    let (br_x, br_y) = (x + size, y + size);
    let same_ctb_row = (y >> LOG2_CTB) == (br_y >> LOG2_CTB);
    let bottom_right = (same_ctb_row && br_x < col.width() && br_y < col.height())
        .then(|| at(br_x, br_y))
        .flatten();
    bottom_right.or_else(|| at(x + size / 2, y + size / 2))
}

/// A PU's merge candidate list: the first `len` entries (`MaxNumMergeCand`,
/// the slice header's value), list 0's vectors kept apart so that they
/// alone are a slice (the engine's candidate vectors).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeList {
    vectors: [Mv; MAX_MERGE_CANDIDATES],
    ref_indices: [u8; MAX_MERGE_CANDIDATES],
    second: [Option<ListMotion>; MAX_MERGE_CANDIDATES],
    len: usize,
}

impl MergeList {
    /// The candidates' list 0 vectors, in `merge_idx` order.
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
            l1: self.second[index],
        }
    }

    /// The first index whose candidate predicts the same samples as
    /// `motion` ([`Motion::as_uni`] for uni-prediction: B slices' zero and
    /// temporal candidates predict from both lists), with that candidate.
    #[must_use]
    #[inline]
    pub fn predicting(&self, motion: Motion) -> Option<(usize, Motion)> {
        (0..self.len)
            .map(|i| (i, self.get(i)))
            .find(|&(_, candidate)| {
                candidate == motion || (motion.l1.is_none() && candidate.as_uni() == Some(motion))
            })
    }

    /// The candidates, in `merge_idx` order.
    pub fn motions(&self) -> impl Iterator<Item = Motion> + '_ {
        (0..self.len).map(|i| self.get(i))
    }

    fn set(&mut self, index: usize, motion: Motion) {
        self.vectors[index] = motion.mv;
        self.ref_indices[index] = motion.ref_idx;
        self.second[index] = motion.l1;
    }
}

/// What the candidate lists of one slice depend on beyond the picture's
/// motion: `MaxNumMergeCand`, the reference list and the slice type.
#[derive(Debug, Clone, Copy)]
pub struct SliceLists<'a> {
    /// `MaxNumMergeCand`.
    pub merge_len: usize,
    /// The POC distances of the (shared) reference list.
    pub references: &'a ReferenceDistances,
    /// A generalised B slice (combined bi-predictive and two-list
    /// candidates).
    pub bi: bool,
}

/// The merge list and the AMVP list for reference `amvp_ref` of list 0 of
/// the 2Nx2N PU at `(x, y)`, sharing one derivation of the spatial
/// neighbours. Exact; with region availability in `Spatial::of`, HEVC Fast
/// P frames fell 1.2% in instructions (callgrind, crowd09 QP 47).
#[must_use]
pub fn candidate_lists(
    grid: &Grid,
    col: Option<&Collocated<'_>>,
    (x, y, size): (usize, usize, usize),
    (slice, amvp_ref): (SliceLists<'_>, u8),
) -> (MergeList, [Mv; AMVP_CANDIDATES]) {
    let spatial = Spatial::of(grid, x, y, size);
    let pu = (x, y, size);
    let references = slice.references;
    // The merge list's list 0 temporal candidate is for `ref_idx` 0; so is
    // the AMVP list's when it predicts the first reference: derive it once.
    let first_once = std::cell::OnceCell::new();
    let first = || {
        *first_once.get_or_init(|| temporal_candidate(col, pu, (references.target(0), List::L0)))
    };
    let amvp_temporal = || {
        if amvp_ref == 0 {
            first()
        } else {
            temporal_candidate(col, pu, (references.target(amvp_ref), List::L0))
        }
    };
    let temporal = || {
        let l1 = || temporal_candidate(col, pu, (references.target(0), List::L1));
        first().map(|mv| Motion {
            mv,
            ref_idx: 0,
            l1: slice
                .bi
                .then(l1)
                .flatten()
                .map(|mv| ListMotion { mv, ref_idx: 0 }),
        })
    };
    (
        merge_list(&spatial, temporal, slice),
        amvp_list(&spatial, amvp_temporal, references, (List::L0, amvp_ref)),
    )
}

/// The AMVP list of reference `target` of `list` for the 2Nx2N PU at
/// `(x, y)` (list 1's half of a bi-predicted unit).
#[must_use]
pub fn amvp_candidates(
    grid: &Grid,
    col: Option<&Collocated<'_>>,
    (x, y, size): (usize, usize, usize),
    references: &ReferenceDistances,
    (list, target): (List, u8),
) -> [Mv; AMVP_CANDIDATES] {
    let spatial = Spatial::of(grid, x, y, size);
    let temporal = || temporal_candidate(col, (x, y, size), (references.target(target), list));
    amvp_list(&spatial, temporal, references, (list, target))
}

/// The merge candidate list: spatial candidates A1, B1, B0, A0, B2 with
/// the specification's pruning, the temporal candidate (`ref_idx` 0), in a
/// B slice the combined bi-predictive candidates, then zero candidates
/// (§8.5.3.2.2–8.5.3.2.5).
fn merge_list(
    n: &Spatial,
    temporal: impl FnOnce() -> Option<Motion>,
    slice: SliceLists<'_>,
) -> MergeList {
    let len = slice.merge_len;
    debug_assert!((1..=MAX_MERGE_CANDIDATES).contains(&len));
    let mut list = MergeList {
        vectors: [Mv::ZERO; MAX_MERGE_CANDIDATES],
        ref_indices: [0; MAX_MERGE_CANDIDATES],
        second: [None; MAX_MERGE_CANDIDATES],
        len,
    };
    let mut filled = 0;
    let mut push = |candidate| {
        if let Some(motion) = candidate {
            list.set(filled, motion);
            filled += 1;
        }
        filled == len
    };

    // §8.5.3.2.3: B1 is pruned against A1, B0 against B1, A0 against A1,
    // and B2 against A1 and B1. Once the list is full, candidates beyond
    // that point cannot change its contents.
    let differs = |c: Option<Motion>, other: Option<Motion>| c.filter(|&v| Some(v) != other);
    if push(n.a1)
        || push(differs(n.b1, n.a1))
        || push(differs(n.b0, n.b1))
        || push(differs(n.a0, n.a1))
    {
        return list;
    }
    if filled < 4
        && let Some(b2) = differs(differs(n.b2, n.a1), n.b1)
    {
        list.set(filled, b2);
        filled += 1;
    }
    // The temporal candidate follows the spatial ones, without pruning.
    if filled < len
        && let Some(motion) = temporal()
    {
        list.set(filled, motion);
        filled += 1;
    }
    if slice.bi {
        filled = combined(&mut list, filled);
    }
    // Entries past these are zero candidates (§8.5.3.2.5), whose reference
    // index steps through the list (`zeroIdx`) and then stays 0: zero
    // vectors on `ref_idx` 0 when the slice has one reference. In a B
    // slice they predict from both lists.
    if !slice.bi && slice.references.count() == 1 {
        // A one-reference P slice's zero candidates are already in the list.
        return list;
    }
    for (zero, index) in (filled..len).enumerate() {
        let ref_idx = if zero < slice.references.count() {
            zero as u8
        } else {
            0
        };
        list.set(
            index,
            Motion {
                mv: Mv::ZERO,
                ref_idx,
                l1: slice.bi.then_some(ListMotion {
                    mv: Mv::ZERO,
                    ref_idx,
                }),
            },
        );
    }
    list
}

/// Append B slices' combined bi-predictive candidates (§8.5.3.2.4) to the
/// `original` candidates of `list`: list 0 of one candidate with list 1 of
/// another, unless both name the same picture with the same vector (one
/// reference index names the same picture in either list). Returns the new
/// candidate count.
fn combined(list: &mut MergeList, original: usize) -> usize {
    let mut n = original;
    if original < 2 || original >= list.len {
        return n;
    }
    for &(i0, i1) in COMBINATIONS.iter().take(original * (original - 1)) {
        let (c0, c1) = (list.get(i0), list.get(i1));
        if let Some(l1) = c1.l1
            && (c0.ref_idx != l1.ref_idx || c0.mv != l1.mv)
        {
            list.set(
                n,
                Motion {
                    mv: c0.mv,
                    ref_idx: c0.ref_idx,
                    l1: Some(l1),
                },
            );
            n += 1;
        }
        if n == list.len {
            break;
        }
    }
    n
}

/// The AMVP list of reference `target` of `list` (§8.5.3.2.6, §8.5.3.2.7).
fn amvp_list(
    nb: &Spatial,
    temporal: impl FnOnce() -> Option<Mv>,
    references: &ReferenceDistances,
    (list, target): (List, u8),
) -> [Mv; AMVP_CANDIDATES] {
    // A neighbour's motion in list X, then in list Y (eq. 8-170..8-172).
    let other = match list {
        List::L0 => List::L1,
        List::L1 => List::L0,
    };
    let lists = |m: &Motion| [m.of(list), m.of(other)].into_iter().flatten();
    // A neighbour pointing at the target picture gives its vector as it
    // is; one pointing at the other reference, its vector scaled by the two
    // POC distances when both are short-term (eq. 8-179..8-183), as it is
    // when both are long-term; none when only one is long-term.
    let same = |m: &Motion| lists(m).find(|l| l.ref_idx == target).map(|l| l.mv);
    let target_long_term = references.long_term(target);
    let scaled = |m: &Motion| {
        lists(m).find_map(|l| {
            if references.long_term(l.ref_idx) != target_long_term {
                None
            } else if target_long_term {
                Some(l.mv)
            } else {
                Some(scale_mv(
                    l.mv,
                    references.distance(l.ref_idx),
                    references.distance(target),
                ))
            }
        })
    };
    let a = [nb.a0, nb.a1];
    let b = [nb.b0, nb.b1, nb.b2];
    // §8.5.3.2.7: A is the first of A0, A1 pointing at the target, else
    // the first available scaled; B the first of B0, B1, B2 pointing at the
    // target. When neither A0 nor A1 is available (isScaledFlag 0), A takes
    // B's vector and B is re-derived allowing scaled candidates.
    let (mv_a, mv_b) = if references.count() == 1 && list == List::L0 {
        // Every neighbour predicts from list 0 at the one reference: A and B
        // are the first available of each group, and B re-derived equals B.
        // (The general path gives the same; this one keeps single-reference
        // slices as cheap as before the second reference existed.)
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

    fn p_slice(merge_len: usize, references: &ReferenceDistances) -> SliceLists<'_> {
        SliceLists {
            merge_len,
            references,
            bi: false,
        }
    }

    fn on_first(mv: Mv) -> Motion {
        Motion::uni(mv, 0)
    }

    fn lists(g: &Grid, at: (usize, usize, usize), len: usize) -> (MergeList, [Mv; 2]) {
        candidate_lists(g, None, at, (p_slice(len, &ReferenceDistances::one(1)), 0))
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
        assert_eq!(long.0.predicting(on_first(corner)).map(|(i, _)| i), Some(2));
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
        let (merge, _) = candidate_lists(&g, None, (8, 8, 8), (p_slice(5, &refs), 0));
        let zero = |ref_idx| Motion::uni(Mv::ZERO, ref_idx);
        // A1, B1, then B2 (the decoded zero-motion cell above-left), then
        // the zero candidates with zeroIdx 0 and 1.
        assert_eq!(
            merge.motions().collect::<Vec<_>>(),
            [
                Motion::uni(v, 1),
                Motion::uni(v, 0),
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
        let (_, first) = candidate_lists(&g, None, (8, 8, 8), (p_slice(2, &refs), 0));
        assert_eq!(first, [Mv { x: 4, y: -2 }, Mv::ZERO]);
        let (_, second) = candidate_lists(&g, None, (8, 8, 8), (p_slice(2, &refs), 1));
        assert_eq!(second, [Mv { x: 16, y: -8 }, Mv::ZERO]);
    }

    fn bi(l0: (Mv, u8), l1: (Mv, u8)) -> Motion {
        Motion {
            mv: l0.0,
            ref_idx: l0.1,
            l1: Some(ListMotion {
                mv: l1.0,
                ref_idx: l1.1,
            }),
        }
    }

    #[test]
    fn b_slices_combine_candidates_and_pad_with_two_list_zeros() {
        let mut g = Grid::new(64, 64);
        let (left, above) = (Mv { x: 4, y: 0 }, Mv { x: 0, y: 4 });
        g.fill(0, 8, 8, inter(left));
        let mut bi_above = inter(above);
        bi_above.l1 = Some(ListMotion {
            mv: left,
            ref_idx: 0,
        });
        g.fill(8, 0, 8, bi_above);
        let refs = ReferenceDistances::one(1);
        let slice = SliceLists {
            merge_len: 5,
            references: &refs,
            bi: true,
        };
        let (merge, _) = candidate_lists(&g, None, (8, 8, 8), (slice, 0));
        // A1 (uni), B1 (bi), then B2 (the zero-motion cell above-left, uni):
        // combination (0, 1) takes A1's list 0 and B1's list 1, which are the
        // same picture and vector, so it is skipped; (1, 0) and (0, 2) need
        // a list 1 on A1 / B2 and are skipped too; (2, 0) likewise; (1, 2)
        // and (2, 1): only B1 has list 1, so (2, 1) gives B2's list 0 with
        // B1's list 1. Then two-list zero candidates.
        assert_eq!(
            merge.motions().collect::<Vec<_>>(),
            [
                Motion::uni(left, 0),
                bi((above, 0), (left, 0)),
                Motion::uni(Mv::ZERO, 0),
                bi((Mv::ZERO, 0), (left, 0)),
                bi((Mv::ZERO, 0), (Mv::ZERO, 0)),
            ]
        );
    }

    #[test]
    fn list_one_amvp_prefers_neighbours_list_one() {
        let mut g = Grid::new(64, 64);
        let mut left = inter(Mv { x: 8, y: 0 });
        left.l1 = Some(ListMotion {
            mv: Mv { x: -4, y: 4 },
            ref_idx: 0,
        });
        g.fill(0, 8, 8, left);
        let refs = ReferenceDistances::one(1);
        let l1 = amvp_candidates(&g, None, (8, 8, 8), &refs, (List::L1, 0));
        assert_eq!(l1, [Mv { x: -4, y: 4 }, Mv::ZERO]);
        let l0 = amvp_candidates(&g, None, (8, 8, 8), &refs, (List::L0, 0));
        assert_eq!(l0, [Mv { x: 8, y: 0 }, Mv::ZERO]);
    }

    #[test]
    fn uni_motion_merges_with_a_candidate_of_two_equal_halves() {
        let g = Grid::new(64, 64);
        let refs = ReferenceDistances::one(1);
        let slice = SliceLists {
            merge_len: 2,
            references: &refs,
            bi: true,
        };
        // No neighbours: the zero candidates predict from both lists.
        let (merge, _) = candidate_lists(&g, None, (0, 0, 8), (slice, 0));
        let zero = bi((Mv::ZERO, 0), (Mv::ZERO, 0));
        assert_eq!(merge.get(0), zero);
        assert_eq!(merge.predicting(Motion::uni(Mv::ZERO, 0)), Some((0, zero)));
        assert_eq!(merge.predicting(Motion::uni(Mv { x: 4, y: 0 }, 0)), None);
    }
}
