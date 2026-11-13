//! Luma motion vector prediction of P macroblocks (§8.4.1.3) and the
//! `P_Skip` motion vector (§8.4.1.1), from the neighbouring partitions A, B
//! and C (or D) of §8.4.1.3.2.
//!
//! The neighbours' vectors and reference indices are gathered once per
//! macroblock into a 6x5 cache around its 4x4 blocks (openh264's
//! `WelsFillCacheInter` layout, BSD-2-Clause): row 0 holds D, the bottom
//! row of B and C; column 0 the right column of A; column 5 is never
//! available. An entry's reference index also tells its availability:
//! [`REF_UNAVAILABLE`] for no partition (outside the picture or the slice,
//! or a block of the current macroblock not decoded yet), -1 for intra.
//! The current macroblock's blocks become available as their partitions are
//! recorded, in decoding order, which is the availability rule of
//! §6.4.11.7. Prediction then reads fixed cache positions instead of
//! branching on where each neighbouring block lies.

use nc_encoder::Mv;

use super::mbstate::{MbState, Neighbours, Partition, partition_blocks};

/// Entries per cache row: A's column, four blocks, the never-available
/// column right of the macroblock.
const CACHE_STRIDE: usize = 6;
/// The neighbour row and four block rows.
const CACHE_LEN: usize = 5 * CACHE_STRIDE;
/// Cache index of 4x4 block (0, 0).
const CACHE_ORIGIN: usize = CACHE_STRIDE + 1;
/// Reference index of an entry without a partition.
const REF_UNAVAILABLE: i8 = -2;

/// A neighbouring macroblock that is not available: every block reads as
/// [`REF_UNAVAILABLE`] with a zero vector. (Intra macroblocks keep
/// [`MbState::EMPTY`]'s reference index -1 and zero vectors, as the
/// prediction treats them.)
const UNAVAILABLE: MbState = MbState {
    ref_idx: [REF_UNAVAILABLE; 4],
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

/// Vectors and reference indices around and inside the current
/// macroblock.
pub(super) struct MotionCache {
    mvs: [Mv; CACHE_LEN],
    refs: [i8; CACHE_LEN],
}

impl MotionCache {
    /// The cache of a macroblock with neighbours `nb`, before any of its
    /// own partitions is decoded.
    pub fn new(nb: &Neighbours<'_>) -> Self {
        let mut cache = Self {
            mvs: [Mv::ZERO; CACHE_LEN],
            refs: [REF_UNAVAILABLE; CACHE_LEN],
        };
        let (a, b) = (or_unavailable(nb.a), or_unavailable(nb.b));
        let (c, d) = (or_unavailable(nb.c), or_unavailable(nb.d));
        cache.put(0, d.motion(3, 3));
        for x4 in 0..4 {
            cache.put(1 + x4, b.motion(x4, 3));
        }
        cache.put(CACHE_STRIDE - 1, c.motion(0, 3));
        for y4 in 0..4 {
            cache.put(cache_index(0, y4) - 1, a.motion(3, y4));
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
    pub fn predict(
        &self,
        (x4, y4, w4, _h4): (usize, usize, usize, usize),
        partition: Partition,
        part: usize,
        ref_idx: i8,
    ) -> Mv {
        let [a, b, c] = self.neighbours(cache_index(x4, y4), w4);
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

    /// The `P_Skip` vector (§8.4.1.1): zero when A or B is unavailable or
    /// either is a zero vector on reference 0, else the 16x16 prediction.
    pub fn skip_mv(&self) -> Mv {
        let [a, b, c] = self.neighbours(CACHE_ORIGIN, 4);
        let zero_on_first = |n: usize| self.refs[n] == 0 && self.mvs[n] == Mv::ZERO;
        let unavailable = self.refs[a] == REF_UNAVAILABLE || self.refs[b] == REF_UNAVAILABLE;
        if unavailable || zero_on_first(a) || zero_on_first(b) {
            return Mv::ZERO;
        }
        self.median([a, b, c], 0)
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

#[cfg(test)]
mod tests {
    use super::*;

    const NO_NEIGHBOURS: Neighbours<'static> = Neighbours {
        a: None,
        b: None,
        c: None,
        d: None,
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
        let mut cache = MotionCache::new(&NO_NEIGHBOURS);
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
    fn only_a_available_gives_its_vector() {
        let mut cache = MotionCache::new(&NO_NEIGHBOURS);
        cache.record((0, 0, 2, 2), mv(7), 1);
        // The second 8x8 block: A is the first (reference 1), B and C lie
        // outside the picture.
        assert_eq!(cache.predict((2, 0, 2, 2), Partition::P8x8, 1, 0), mv(7));
        assert_eq!(cache.skip_mv(), Mv::ZERO);
    }
}
