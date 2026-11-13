//! Intra mode candidates shared by the backends' intra searches.
//!
//! Every codec costs a short list of intra modes for a block and codes the
//! cheapest: the list is the codec-independent part of that search, chosen
//! by one registered parameter per codec (AV1 `av1.keyframe_intra_modes`,
//! AV2 `av2.intra_modes`). Codecs map these modes onto their own (AV2's
//! vertical and horizontal are its zero-delta directional modes) and may add
//! the modes of already coded neighbours ([`with_neighbours`]), which are
//! likely and cheap to signal in mode-list syntax. [`shortlist_intra`] also
//! provides codec-independent proxy ranking and a bounded full-RD candidate list
//! for directional searches; VVC supplies its prediction and syntax costs.
//!
//! [`directions_by_orientation`] orders the eight directional modes by how
//! much of the block's gradient energy runs along each (a histogram of
//! oriented gradients, as libaom prunes directional modes), so a search can
//! predict only the few likeliest directions.

use nc_common::PlaneView;
use nc_dsp::intra_prediction::IntraMode;

/// The intra modes a block's search costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraModeSet {
    /// DC only: no search.
    DcOnly,
    /// DC, the three smooth modes and Paeth.
    NonDirectional,
    /// The non-directional modes plus vertical and horizontal.
    Extended,
}

impl IntraModeSet {
    /// The set selected by a registered parameter value: 0 DC only,
    /// 1 non-directional, 2 extended (rounded; values above 2 select the
    /// extended set).
    #[must_use]
    pub fn from_param(value: f32) -> Self {
        match value.round() as i32 {
            i32::MIN..=0 => Self::DcOnly,
            1 => Self::NonDirectional,
            _ => Self::Extended,
        }
    }

    /// The set's modes, in the order searches visit them (ties keep the
    /// earlier mode).
    #[must_use]
    pub fn modes(self) -> &'static [IntraMode] {
        match self {
            Self::DcOnly => &[IntraMode::Dc],
            Self::NonDirectional => &IntraMode::NON_DIRECTIONAL,
            Self::Extended => &IntraMode::EXTENDED,
        }
    }
}

/// Line orientations told apart: 0, 22.5, ..., 157.5 degrees.
const ORIENTATIONS: usize = 8;
/// Unit vectors (×1024) of the boundaries between orientation bins, at
/// 11.25 + 22.5k degrees (y up).
const BIN_BOUNDARIES: [(i32, i32); ORIENTATIONS] = [
    (1004, 200),
    (851, 569),
    (569, 851),
    (200, 1004),
    (-200, 1004),
    (-569, 851),
    (-851, 569),
    (-1004, 200),
];
/// The directional mode whose prediction lines run at each orientation
/// (prediction angle modulo 180: H 180, D203 23, D45, D67, V 90, D113, D135,
/// D157).
const MODE_OF_ORIENTATION: [IntraMode; ORIENTATIONS] = [
    IntraMode::Horizontal,
    IntraMode::D203,
    IntraMode::D45,
    IntraMode::D67,
    IntraMode::Vertical,
    IntraMode::D113,
    IntraMode::D135,
    IntraMode::D157,
];

/// Gradient energy (|gx| + |gy|, central differences inside the block) per
/// line orientation, the line running perpendicular to the gradient.
#[must_use]
pub fn orientation_histogram(block: &PlaneView<'_>) -> [u64; ORIENTATIONS] {
    let mut histogram = [0u64; ORIENTATIONS];
    let (w, h) = (block.width(), block.height());
    for y in 1..h.saturating_sub(1) {
        let (above, row, below) = (block.row(y - 1), block.row(y), block.row(y + 1));
        for x in 1..w - 1 {
            let gx = i32::from(row[x + 1]) - i32::from(row[x - 1]);
            // y up, so that angles count as AV1's prediction angles do.
            let gy = i32::from(above[x]) - i32::from(below[x]);
            if gx == 0 && gy == 0 {
                continue;
            }
            // The line direction, turned into the upper half-plane.
            let (mut u, mut v) = (-gy, gx);
            if v < 0 || (v == 0 && u < 0) {
                (u, v) = (-u, -v);
            }
            let passed = BIN_BOUNDARIES
                .iter()
                .filter(|&&(bx, by)| bx * v - by * u > 0)
                .count();
            histogram[passed % ORIENTATIONS] += u64::from(gx.unsigned_abs() + gy.unsigned_abs());
        }
    }
    histogram
}

/// The eight directional modes, the one most of `block`'s gradient energy
/// runs along first (ties keep the orientation order).
#[must_use]
pub fn directions_by_orientation(block: &PlaneView<'_>) -> [IntraMode; ORIENTATIONS] {
    let histogram = orientation_histogram(block);
    let mut order: [usize; ORIENTATIONS] = std::array::from_fn(|i| i);
    order.sort_by_key(|&i| std::cmp::Reverse(histogram[i]));
    order.map(|i| MODE_OF_ORIENTATION[i])
}

/// `set` followed by the `neighbours` it does not already contain, each once.
#[must_use]
pub fn with_neighbours<T: Copy + PartialEq>(
    set: impl IntoIterator<Item = T>,
    neighbours: impl IntoIterator<Item = T>,
) -> Vec<T> {
    let mut candidates: Vec<T> = set.into_iter().collect();
    for mode in neighbours {
        if !candidates.contains(&mode) {
            candidates.push(mode);
        }
    }
    candidates
}

/// Retain the mandatory intra modes and add the lowest proxy-cost alternatives.
///
/// The codec supplies its prediction/signalling proxy through `rough_cost`;
/// the shared engine owns deduplication, budgeting and deterministic ranking.
/// Every alternative is scored once, ties keep input order, and mandatory
/// modes cannot be displaced. The caller performs its full RD evaluation on
/// the returned list. A budget covering all alternatives avoids proxy work.
#[must_use]
pub fn shortlist_intra<T: Copy + PartialEq>(
    mandatory: impl IntoIterator<Item = T>,
    alternatives: impl IntoIterator<Item = T>,
    extra_budget: usize,
    mut rough_cost: impl FnMut(T) -> f32,
) -> Vec<T> {
    let mut selected = with_neighbours(std::iter::empty(), mandatory);
    if extra_budget == 0 {
        return selected;
    }
    let mut rest = with_neighbours(std::iter::empty(), alternatives);
    rest.retain(|m| !selected.contains(m));
    if extra_budget >= rest.len() {
        selected.extend(rest);
        return selected;
    }
    let mut ranked: Vec<_> = rest.into_iter().map(|m| (m, rough_cost(m))).collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    selected.extend(ranked.into_iter().take(extra_budget).map(|(m, _)| m));
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortlist_preserves_required_modes_and_stable_ties() {
        let mut evaluated = Vec::new();
        let selected = shortlist_intra([0, 1, 0], [1, 2, 3, 4, 2], 2, |m| {
            evaluated.push(m);
            if m == 3 { 0. } else { 1. }
        });
        assert_eq!(selected, [0, 1, 3, 2]);
        assert_eq!(evaluated, [2, 3, 4]);
        assert_eq!(
            shortlist_intra([0, 1], [2, 3], 0, |_| panic!("no proxy needed")),
            [0, 1]
        );
        assert_eq!(
            shortlist_intra([0, 1], [1, 2, 3], 2, |_| panic!("exhaustive")),
            [0, 1, 2, 3]
        );
    }

    #[test]
    fn parameter_values_select_growing_sets() {
        assert_eq!(IntraModeSet::from_param(0.0).modes(), &[IntraMode::Dc]);
        assert_eq!(
            IntraModeSet::from_param(1.0).modes(),
            &IntraMode::NON_DIRECTIONAL
        );
        assert_eq!(IntraModeSet::from_param(2.0).modes(), &IntraMode::EXTENDED);
        assert_eq!(IntraModeSet::from_param(3.0), IntraModeSet::Extended);
    }

    #[test]
    fn stripes_rank_their_own_direction_first() {
        let n = 16;
        let pattern = |f: &dyn Fn(usize, usize) -> usize| {
            let mut samples = vec![0u8; n * n];
            for y in 0..n {
                for x in 0..n {
                    samples[y * n + x] = if f(x, y) % 4 < 2 { 40 } else { 200 };
                }
            }
            samples
        };
        // Vertical stripes vary along x only; "/" stripes are constant along
        // x + y (y down), i.e. lines at 45 degrees; "\\" at 135.
        for (f, mode) in [
            (
                &(|x: usize, _y: usize| x) as &dyn Fn(usize, usize) -> usize,
                IntraMode::Vertical,
            ),
            (&|_x: usize, y: usize| y, IntraMode::Horizontal),
            (&|x: usize, y: usize| x + y, IntraMode::D45),
            (&|x: usize, y: usize| x + n - y, IntraMode::D135),
        ] {
            let samples = pattern(f);
            let view = PlaneView::new(&samples, n, n, n).unwrap();
            assert_eq!(directions_by_orientation(&view)[0], mode);
        }
    }

    #[test]
    fn neighbours_extend_the_set_once_each() {
        assert_eq!(with_neighbours([1, 2], [2, 5, 5, 7]), vec![1, 2, 5, 7]);
    }
}
