//! Luma motion vector prediction (§8.4.1.3) of one reference list, the
//! `P_Skip` motion vector (§8.4.1.1) and the neighbours' reference indices
//! of spatial direct prediction (§8.4.1.2.2), from the neighbouring
//! partitions A, B and C (or D) of §8.4.1.3.2.
//!
//! The neighbours' vectors and reference indices are gathered once per
//! macroblock into a 6x5 cache around its 4x4 blocks (openh264's
//! `WelsFillCacheInter` layout, BSD-2-Clause): row 0 holds D, the bottom
//! row of B and C; column 0 the right column of A; column 5 is never
//! available. An entry's reference index also tells its availability:
//! [`REF_UNAVAILABLE`] for no partition (outside the picture or the slice,
//! or a block of the current macroblock not decoded yet), -1 for intra or a
//! partition that does not predict from the cache's list.
//! The current macroblock's blocks become available as their partitions are
//! recorded, in decoding order, which is the availability rule of
//! §6.4.11.7. Prediction then reads fixed cache positions instead of
//! branching on where each neighbouring block lies.

use nc_encoder::Mv;

use super::mbstate::{MbState, Neighbours, PairNeighbours, Partition, partition_blocks};

/// Entries per cache row: A's column, four blocks, the never-available
/// column right of the macroblock.
const CACHE_STRIDE: usize = 6;
/// The neighbour row and four block rows.
const CACHE_LEN: usize = 5 * CACHE_STRIDE;
/// Beyond the grid: the up-left neighbours of the left column's blocks
/// 1..=3 where they are not the A entry of the row above (MBAFF frames,
/// [`PairNeighbours::corner`]).
const CORNERS: usize = CACHE_LEN;
/// Entries with the corner slots.
const CACHE_SLOTS: usize = CACHE_LEN + 3;
/// Cache index of 4x4 block (0, 0).
const CACHE_ORIGIN: usize = CACHE_STRIDE + 1;
/// Reference index of an entry without a partition.
const REF_UNAVAILABLE: i8 = -2;

/// A neighbouring macroblock that is not available: every block reads as
/// [`REF_UNAVAILABLE`] with a zero vector. (Intra macroblocks keep
/// [`MbState::EMPTY`]'s reference index -1 and zero vectors, as the
/// prediction treats them.)
const UNAVAILABLE: MbState = MbState {
    ref_idx: [[REF_UNAVAILABLE; 4]; 2],
    ..MbState::EMPTY
};

/// `n`, or [`UNAVAILABLE`] standing in for it.
fn or_unavailable(n: Option<&MbState>) -> &MbState {
    n.unwrap_or(&UNAVAILABLE)
}

/// The cache index of 4x4 block (`x4`, `y4`) of the current macroblock.
fn cache_index(x4: usize, y4: usize) -> usize {
    CACHE_ORIGIN + y4 * CACHE_STRIDE + x4
}

/// Vectors and reference indices of one list around and inside the
/// current macroblock.
pub(super) struct MotionCache {
    mvs: [Mv; CACHE_SLOTS],
    refs: [i8; CACHE_SLOTS],
    /// The left column's up-left neighbours are the corner slots.
    split_corners: bool,
}

impl MotionCache {
    /// The list `list` cache of a macroblock with neighbours `nb`, before
    /// any of its own partitions is decoded.
    pub fn new(nb: &Neighbours<'_>, list: usize) -> Self {
        if nb.pair.is_some() {
            return Self::new_in_pair(nb, list);
        }
        let mut cache = Self {
            mvs: [Mv::ZERO; CACHE_SLOTS],
            refs: [REF_UNAVAILABLE; CACHE_SLOTS],
            split_corners: false,
        };
        let (a, b) = (or_unavailable(nb.a), or_unavailable(nb.b));
        let (c, d) = (or_unavailable(nb.c), or_unavailable(nb.d));
        cache.put(0, d.motion(list, 3, 3));
        for x4 in 0..4 {
            cache.put(1 + x4, b.motion(list, x4, 3));
        }
        cache.put(CACHE_STRIDE - 1, c.motion(list, 0, 3));
        for y4 in 0..4 {
            cache.put(cache_index(0, y4) - 1, a.motion(list, 3, y4));
        }
        cache
    }

    /// [`Self::new`] in an MBAFF frame: A's blocks from the macroblock
    /// and row of each block row, D's corner row, every neighbour scaled to
    /// the current macroblock's frame or field coding.
    #[inline(never)]
    fn new_in_pair(nb: &Neighbours<'_>, list: usize) -> Self {
        let mut cache = Self {
            mvs: [Mv::ZERO; CACHE_SLOTS],
            refs: [REF_UNAVAILABLE; CACHE_SLOTS],
            split_corners: false,
        };
        let d_row = nb.pair.map_or(3, |p| p.d_row());
        let read = |n: Option<&MbState>, x4: usize, y4: usize| match n {
            Some(n) => scaled_motion(nb, n, list, x4, y4),
            None => UNAVAILABLE.motion(list, x4, y4),
        };
        cache.put(0, read(nb.d, 3, d_row));
        for x4 in 0..4 {
            cache.put(1 + x4, read(nb.b, x4, 3));
        }
        cache.put(CACHE_STRIDE - 1, read(nb.c, 0, 3));
        for y4 in 0..4 {
            let (n, row) = nb
                .left_row(y4)
                .map_or((None, y4), |(m, row)| (Some(m), row));
            cache.put(cache_index(0, y4) - 1, read(n, 3, row));
        }
        if let Some(pair) = nb.pair.filter(PairNeighbours::split_left) {
            cache.split_corners = true;
            for y4 in 1..4 {
                let (n, row) = pair.corner(y4).map_or((None, 3), |(m, row)| (Some(m), row));
                cache.put(CORNERS + y4 - 1, read(n, 3, row));
            }
        }
        cache
    }

    fn put(&mut self, index: usize, (mv, ref_idx): (Mv, i8)) {
        self.mvs[index] = mv;
        self.refs[index] = ref_idx;
    }

    /// Record the decoded partition (`x4`, `y4`, `w4`, `h4`): its blocks
    /// become available to the partitions after it.
    pub fn record(&mut self, (x4, y4, w4, h4): (usize, usize, usize, usize), mv: Mv, ref_idx: i8) {
        let covered = partition_blocks(x4, y4, w4, h4, 4);
        // Every block, kept or replaced by a select: a fixed trip count.
        for block in 0..16 {
            let index = cache_index(block % 4, block / 4);
            let inside = (covered >> block) & 1 != 0;
            self.mvs[index] = core::hint::select_unpredictable(inside, mv, self.mvs[index]);
            self.refs[index] = core::hint::select_unpredictable(inside, ref_idx, self.refs[index]);
        }
    }

    /// The predictor of the partition (`x4`, `y4`, `w4`, `h4`) with
    /// reference index `ref_idx` of a macroblock with `partition`, `part`
    /// its `mbPartIdx`.
    #[inline]
    pub fn predict(
        &self,
        (x4, y4, w4, _h4): (usize, usize, usize, usize),
        partition: Partition,
        part: usize,
        ref_idx: i8,
    ) -> Mv {
        let at = cache_index(x4, y4);
        let [a, b, mut c] = self.neighbours(at, w4);
        // A left-column block below the first row in an MBAFF frame whose
        // left pair is coded the other way: its D is a corner slot.
        if self.split_corners && x4 == 0 && y4 > 0 && c == b - 1 {
            c = CORNERS + y4 - 1;
        }
        // Directional prediction of 16x8 and 8x16 partitions (eqs.
        // 8-200..8-203).
        let directional = match (partition, part) {
            (Partition::P16x8, 0) => Some(b),
            (Partition::P16x8, _) | (Partition::P8x16, 0) => Some(a),
            (Partition::P8x16, _) => Some(c),
            _ => None,
        };
        if let Some(n) = directional
            && self.refs[n] == ref_idx
        {
            return self.mvs[n];
        }
        self.median([a, b, c], ref_idx)
    }

    /// The `P_Skip` vector of a macroblock with neighbours `nb`: a cache
    /// holding only the four blocks [`MotionCache::skip_mv`] reads (A's
    /// (3, 0), B's (0, 3), C's (0, 3), D's (3, 3)). Filling the whole
    /// cache for it was a third of the cost of a skipped macroblock.
    pub fn skip_mv_of(nb: &Neighbours<'_>) -> Mv {
        if nb.pair.is_some() {
            return Self::new_in_pair(nb, 0).skip_mv();
        }
        let mut cache = Self {
            mvs: [Mv::ZERO; CACHE_SLOTS],
            refs: [REF_UNAVAILABLE; CACHE_SLOTS],
            split_corners: false,
        };
        cache.put(0, or_unavailable(nb.d).motion(0, 3, 3));
        cache.put(1, or_unavailable(nb.b).motion(0, 0, 3));
        cache.put(CACHE_STRIDE - 1, or_unavailable(nb.c).motion(0, 0, 3));
        cache.put(CACHE_ORIGIN - 1, or_unavailable(nb.a).motion(0, 3, 0));
        cache.skip_mv()
    }

    /// The `P_Skip` vector (§8.4.1.1): zero when A or B is unavailable or
    /// either is a zero vector on reference 0, else the 16x16 prediction.
    fn skip_mv(&self) -> Mv {
        let [a, b, c] = self.neighbours(CACHE_ORIGIN, 4);
        let zero_on_first = |n: usize| self.refs[n] == 0 && self.mvs[n] == Mv::ZERO;
        let unavailable = self.refs[a] == REF_UNAVAILABLE || self.refs[b] == REF_UNAVAILABLE;
        if unavailable || zero_on_first(a) || zero_on_first(b) {
            return Mv::ZERO;
        }
        self.median([a, b, c], 0)
    }

    /// The reference index of spatial direct prediction (§8.4.1.2.2,
    /// eqs. 8-184..8-186): the smallest non-negative one of the
    /// macroblock's neighbours A, B and C (or D), else -1. Read before any
    /// partition of the macroblock is recorded.
    pub fn direct_ref_idx(&self) -> i8 {
        let [a, b, c] = self.neighbours(CACHE_ORIGIN, 4);
        let min_positive = |x: i8, y: i8| if x >= 0 && y >= 0 { x.min(y) } else { x.max(y) };
        min_positive(self.refs[a], min_positive(self.refs[b], self.refs[c])).max(-1)
    }

    /// The 16x16 predictor with reference index `ref_idx` (spatial direct
    /// prediction's vector, §8.4.1.2.2).
    pub fn predict_16x16(&self, ref_idx: i8) -> Mv {
        self.median(self.neighbours(CACHE_ORIGIN, 4), ref_idx)
    }

    /// Cache indices of A, B and C (D where C is unavailable) of the
    /// partition whose top-left block is at cache index `at`, `w4` blocks
    /// wide.
    fn neighbours(&self, at: usize, w4: usize) -> [usize; 3] {
        let (a, b) = (at - 1, at - CACHE_STRIDE);
        let c = b + w4;
        let c_or_d = core::hint::select_unpredictable(self.refs[c] == REF_UNAVAILABLE, b - 1, c);
        [a, b, c_or_d]
    }

    /// §8.4.1.3.1: B and C fall back to A when both are unavailable; a
    /// single neighbour with the same reference index gives its vector,
    /// else the component-wise median. Every case is computed and the
    /// result selected: which applies varies from partition to partition.
    fn median(&self, [a, b, c]: [usize; 3], ref_idx: i8) -> Mv {
        let only_a = self.refs[b] == REF_UNAVAILABLE
            && self.refs[c] == REF_UNAVAILABLE
            && self.refs[a] != REF_UNAVAILABLE;
        let (b, c) = core::hint::select_unpredictable(only_a, (a, a), (b, c));
        let same = [a, b, c].map(|n| self.refs[n] == ref_idx);
        let first_same = core::hint::select_unpredictable(
            same[0],
            a,
            core::hint::select_unpredictable(same[1], b, c),
        );
        let med = |p: i16, q: i16, r: i16| p.max(q).min(p.min(q).max(r));
        let (ma, mb, mc) = (self.mvs[a], self.mvs[b], self.mvs[c]);
        let median = Mv {
            x: med(ma.x, mb.x, mc.x),
            y: med(ma.y, mb.y, mc.y),
        };
        let single = same.iter().filter(|&&s| s).count() == 1;
        core::hint::select_unpredictable(single, self.mvs[first_same], median)
    }
}

/// `n`'s motion of list `list` at (`x4`, `y4`) as the current macroblock
/// reads it: in an MBAFF frame a neighbour coded the other way has its
/// vertical vector and reference index scaled (§8.4.1.3.1: a field
/// macroblock halves a frame neighbour's vector, `/` truncating, and doubles
/// its index; a frame macroblock the reverse).
pub(super) fn scaled_motion(
    nb: &Neighbours<'_>,
    n: &MbState,
    list: usize,
    x4: usize,
    y4: usize,
) -> (Mv, i8) {
    let (mv, ref_idx) = n.motion(list, x4, y4);
    match nb.coded_otherwise(n) {
        None => (mv, ref_idx),
        Some(true) => (
            Mv {
                x: mv.x,
                y: mv.y / 2,
            },
            if ref_idx >= 0 { ref_idx * 2 } else { ref_idx },
        ),
        Some(false) => (
            Mv {
                x: mv.x,
                y: mv.y.saturating_mul(2),
            },
            if ref_idx >= 0 { ref_idx >> 1 } else { ref_idx },
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_NEIGHBOURS: Neighbours<'static> = Neighbours {
        a: None,
        b: None,
        c: None,
        d: None,
        pair: None,
    };

    fn mv(x: i16) -> Mv {
        Mv { x, y: -x }
    }

    #[test]
    fn blocks_not_yet_decoded_are_unavailable() {
        // 4x4 blocks 0, 1, 2 of the first 8x8 block decoded: the
        // above-right of block 3 (1, 1) is block 4 (2, 0), not decoded yet,
        // so C falls back to D, block 0. All on reference 0: the median of
        // A (0, 1), B (1, 0) and D (0, 0).
        let mut cache = MotionCache::new(&NO_NEIGHBOURS, 0);
        cache.record((0, 0, 1, 1), mv(1), 0);
        cache.record((1, 0, 1, 1), mv(5), 0);
        cache.record((0, 1, 1, 1), mv(9), 0);
        let pred = cache.predict((1, 1, 1, 1), Partition::P8x8, 0, 0);
        assert_eq!(pred, mv(5));
        // Once block 4 is decoded it is C of a later partition at (1, 1).
        cache.record((2, 0, 1, 1), mv(100), 0);
        let pred = cache.predict((1, 1, 1, 1), Partition::P8x8, 0, 0);
        assert_eq!(pred, mv(9));
    }

    #[test]
    fn skip_vector_from_the_four_neighbours_matches_the_full_cache() {
        let inter = |x: i16, ref_idx: i8| {
            let mut m = MbState::EMPTY;
            m.mvs[0] = core::array::from_fn(|i| mv(x + i as i16));
            m.ref_idx[0] = [ref_idx; 4];
            m
        };
        let states = [inter(3, 0), inter(-7, 1), inter(0, 0), MbState::EMPTY];
        for (ai, bi, ci, di) in (0..5usize).flat_map(|a| {
            (0..5usize).flat_map(move |b| {
                (0..5usize).flat_map(move |c| (0..5usize).map(move |d| (a, b, c, d)))
            })
        }) {
            let pick = |i: usize| states.get(i);
            let nb = Neighbours {
                a: pick(ai),
                b: pick(bi),
                c: pick(ci),
                d: pick(di),
                pair: None,
            };
            assert_eq!(
                MotionCache::skip_mv_of(&nb),
                MotionCache::new(&nb, 0).skip_mv(),
                "{ai} {bi} {ci} {di}"
            );
        }
    }

    #[test]
    fn only_a_available_gives_its_vector() {
        let mut cache = MotionCache::new(&NO_NEIGHBOURS, 0);
        cache.record((0, 0, 2, 2), mv(7), 1);
        // The second 8x8 block: A is the first (reference 1), B and C lie
        // outside the picture.
        assert_eq!(cache.predict((2, 0, 2, 2), Partition::P8x8, 1, 0), mv(7));
        assert_eq!(cache.skip_mv(), Mv::ZERO);
    }
}
