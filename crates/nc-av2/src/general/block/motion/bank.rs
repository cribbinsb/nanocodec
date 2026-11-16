//! The reference MV bank (recent vectors per reference class, reused as
//! candidates when the neighbours give too few) and the warp bank (recent
//! warp models per reference) of a tile (dav2d `refmvs_tile` `bank`,
//! `warp`; `refmvs.c` `refmvs_bank_add`, `dav2d_refmvs_bank_update`,
//! `dav2d_refmvs_bank_add`, `dav2d_refmvs_warp_add`; dav2d commit 237229e,
//! BSD-2-Clause, see `docs/licenses/dav2d`).
use super::vector::{Mv, REFERENCES, RefPair};

/// Entries per class, a ring with the most recent last.
pub const ENTRIES: usize = 4;
const MASK: usize = ENTRIES - 1;
/// Classes: single references 0..5, the pairs (0, 0) and (0, 1), then
/// every other pair or reference (6, TIP, none) with its pair kept.
const CLASSES: usize = 9;
const FIRST_PAIR_CLASS: usize = 6;
const OTHER_CLASS: usize = 8;
/// Single references with a class of their own.
const LAST_SINGLE_CLASS: i8 = 5;
/// The bank stops taking vectors after this many per superblock, or per
/// bank unit, and starts each superblock with room for at least this
/// many.
const MAX_SUPERBLOCK_HITS: u8 = 64;
const MAX_UNIT_HITS: u8 = 16;
const MIN_AVAILABLE: u8 = 4;
/// The warp bank stops taking models after this many per superblock.
const MAX_WARP_HITS: u8 = 64;
/// The compound weight of an entry that has none.
pub const NO_WEIGHT: i8 = 0;

/// The class of a reference pair (dav2d's `c`).
#[must_use]
pub fn class(refs: RefPair) -> usize {
    let [first, second] = refs.0;
    if !refs.is_compound() {
        if (0..=LAST_SINGLE_CLASS).contains(&first) {
            first as usize
        } else {
            OTHER_CLASS
        }
    } else if first == 0 && (0..=1).contains(&second) {
        FIRST_PAIR_CLASS + second as usize
    } else {
        OTHER_CLASS
    }
}

/// One bank entry as a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankEntry {
    /// The vectors (the second for compound pairs).
    pub mv: [Mv; 2],
    /// Its reference pair (meaningful in the shared class only).
    pub refs: RefPair,
    /// Its compound weight index (pair classes only).
    pub cwp: i8,
}

/// The MV bank.
#[derive(Debug, Clone)]
pub struct MvBank {
    mv: [[[Mv; 2]; ENTRIES]; CLASSES],
    /// The compound weights of the pair classes (6, 7, 8).
    cwp: [[i8; ENTRIES]; CLASSES - FIRST_PAIR_CLASS],
    /// The pairs of the shared class.
    refs: [RefPair; ENTRIES],
    size: [u8; CLASSES],
    start: [u8; CLASSES],
    /// Vectors added in this superblock, and in this bank unit.
    superblock_hits: u8,
    unit_hits: u8,
    /// How many more vectors the superblock may add.
    available: u8,
}

impl Default for MvBank {
    fn default() -> Self {
        Self {
            mv: [[[Mv::default(); 2]; ENTRIES]; CLASSES],
            cwp: [[NO_WEIGHT; ENTRIES]; CLASSES - FIRST_PAIR_CLASS],
            refs: [RefPair::NONE; ENTRIES],
            size: [0; CLASSES],
            start: [0; CLASSES],
            superblock_hits: 0,
            unit_hits: 0,
            available: 0,
        }
    }
}

/// Move entry `from` of a ring of `ENTRIES` to the most recent place
/// `to`, shifting those between back by one (dav2d's reorder loop).
fn move_to_back<T: Copy>(ring: &mut [T; ENTRIES], from: usize, to: usize) {
    let moved = ring[from];
    let mut at = from;
    while at != to {
        let next = (at + 1) & MASK;
        ring[at] = ring[next];
        at = next;
    }
    ring[to] = moved;
}

/// Where a ring of `size` entries starting at `start` puts a new entry,
/// advancing it (dav2d's `sz == 4 ? idx++ & 3 : size++`).
fn next_slot(size: &mut u8, start: &mut u8) -> usize {
    if usize::from(*size) == ENTRIES {
        let slot = usize::from(*start) & MASK;
        *start = start.wrapping_add(1);
        slot
    } else {
        *size += 1;
        usize::from(*size) - 1
    }
}

/// The most recent place of a ring (dav2d's `to`).
fn last_slot(size: u8, start: u8) -> usize {
    if usize::from(size) == ENTRIES {
        usize::from(start.wrapping_sub(1)) & MASK
    } else {
        usize::from(size) - 1
    }
}

impl MvBank {
    /// At each superblock row of a tile: empty (dav2d
    /// `dav2d_refmvs_tile_sbrow_init`).
    pub fn clear(&mut self) {
        self.size = [0; CLASSES];
        self.start = [0; CLASSES];
    }

    /// At each superblock: the limits restart (dav2d `reset_sb`).
    pub fn start_superblock(&mut self) {
        (self.superblock_hits, self.unit_hits, self.available) = (0, 0, 0);
    }

    /// The limits at a block (dav2d `dav2d_refmvs_bank_update`): each
    /// superblock, and each bank unit of it (`unit_log2` units square),
    /// makes room for a vector per bank unit the block covers.
    pub fn update(
        &mut self,
        (row, column): (u32, u32),
        (bw4, bh4): (u32, u32),
        superblock_units: u32,
        unit_log2: u32,
    ) {
        let room = ((bw4 >> unit_log2).max(1) * (bh4 >> unit_log2).max(1)) as u8;
        if (row | column) & (superblock_units - 1) == 0 {
            self.unit_hits = 0;
            self.available = room.max(MIN_AVAILABLE);
        } else if (row | column) & ((1 << unit_log2) - 1) == 0 {
            self.unit_hits = 0;
            self.available = self.available.wrapping_add(room);
        }
    }

    /// Offer a block's vectors within the limits (dav2d
    /// `dav2d_refmvs_bank_add` after its update).
    pub fn offer(&mut self, refs: RefPair, mv: [Mv; 2], cwp: i8) {
        if self.superblock_hits >= MAX_SUPERBLOCK_HITS
            || self.unit_hits >= MAX_UNIT_HITS
            || self.available == 0
        {
            return;
        }
        self.unit_hits += 1;
        self.available -= 1;
        self.insert(refs, mv, if refs.is_compound() { cwp } else { NO_WEIGHT });
    }

    /// Add vectors, or move an equal entry to the most recent place
    /// (dav2d `refmvs_bank_add`).
    pub fn insert(&mut self, refs: RefPair, mv: [Mv; 2], cwp: i8) {
        self.superblock_hits = self.superblock_hits.wrapping_add(1);
        let c = class(refs);
        let comp = usize::from(refs.is_compound());
        let (size, start) = (self.size[c], self.start[c]);
        let found = (0..usize::from(size)).find(|&n| {
            let i = (usize::from(start) + n) & MASK;
            (c < OTHER_CLASS || refs == self.refs[i])
                && mv[0] == self.mv[c][i][0]
                && mv[comp] == self.mv[c][i][comp]
        });
        if let Some(n) = found {
            let from = (usize::from(start) + n) & MASK;
            let to = last_slot(size, start);
            move_to_back(&mut self.mv[c], from, to);
            if c == OTHER_CLASS {
                move_to_back(&mut self.refs, from, to);
            }
            if c >= FIRST_PAIR_CLASS {
                move_to_back(&mut self.cwp[c - FIRST_PAIR_CLASS], from, to);
            }
            return;
        }
        let slot = next_slot(&mut self.size[c], &mut self.start[c]);
        self.mv[c][slot] = mv;
        if c == OTHER_CLASS {
            self.refs[slot] = refs;
        }
        if refs.is_compound() {
            self.cwp[c - FIRST_PAIR_CLASS][slot] = cwp;
        }
    }

    /// The entries of `refs`' class, most recent first (in the shared
    /// class, all of them: the caller matches the pair).
    pub fn recent(&self, refs: RefPair) -> impl Iterator<Item = BankEntry> + '_ {
        let c = class(refs);
        let (size, start) = (self.size[c], self.start[c]);
        let last = usize::from(size) + usize::from(start);
        (0..usize::from(size)).map(move |n| {
            let i = last.wrapping_sub(1 + n) & MASK;
            BankEntry {
                mv: self.mv[c][i],
                refs: self.refs[i],
                cwp: if c >= FIRST_PAIR_CLASS {
                    self.cwp[c - FIRST_PAIR_CLASS][i]
                } else {
                    NO_WEIGHT
                },
            }
        })
    }

    /// Whether `class(refs)` is the shared class, whose entries carry
    /// their pair.
    #[must_use]
    pub fn shared(refs: RefPair) -> bool {
        class(refs) == OTHER_CLASS
    }
}

/// The warp bank: per reference rank, the last four warp matrices.
#[derive(Debug, Clone, Default)]
pub struct WarpBank {
    matrices: [[[i32; 6]; ENTRIES]; REFERENCES],
    size: [u8; REFERENCES],
    start: [u8; REFERENCES],
    hits: u8,
}

impl WarpBank {
    /// At each superblock row of a tile: empty.
    pub fn clear(&mut self) {
        self.size = [0; REFERENCES];
        self.start = [0; REFERENCES];
    }

    /// At each superblock: the limit restarts.
    pub fn start_superblock(&mut self) {
        self.hits = 0;
    }

    /// Add a model of reference `rank`, or move one with the same 2x2
    /// part to the most recent place (dav2d `dav2d_refmvs_warp_add`);
    /// false once the superblock's limit is reached.
    pub fn insert(&mut self, matrix: &[i32; 6], rank: usize) -> bool {
        if self.hits >= MAX_WARP_HITS {
            return false;
        }
        self.hits += 1;
        let (size, start) = (self.size[rank], self.start[rank]);
        let ring = &mut self.matrices[rank];
        let found = (0..usize::from(size))
            .find(|&n| ring[(usize::from(start) + n) & MASK][2..] == matrix[2..]);
        if let Some(n) = found {
            move_to_back(
                ring,
                (usize::from(start) + n) & MASK,
                last_slot(size, start),
            );
            return true;
        }
        let slot = next_slot(&mut self.size[rank], &mut self.start[rank]);
        self.matrices[rank][slot] = *matrix;
        true
    }

    /// The models of `rank`, most recent first.
    pub fn recent(&self, rank: usize) -> impl Iterator<Item = &[i32; 6]> + '_ {
        let last = usize::from(self.size[rank]) + usize::from(self.start[rank]);
        (0..usize::from(self.size[rank]))
            .map(move |n| &self.matrices[rank][last.wrapping_sub(1 + n) & MASK])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(y: i32, x: i32) -> [Mv; 2] {
        [Mv { y, x }, Mv::default()]
    }

    #[test]
    fn classes() {
        assert_eq!(class(RefPair::single(3)), 3);
        assert_eq!(class(RefPair::single(6)), 8);
        assert_eq!(class(RefPair::NONE), 8);
        assert_eq!(class(RefPair([0, 1])), 7);
        assert_eq!(class(RefPair([1, 2])), 8);
    }

    /// The ring keeps four, most recent first; a repeat moves to the
    /// front.
    #[test]
    fn ring_order() {
        let mut bank = MvBank::default();
        let refs = RefPair::single(0);
        for v in 1..=5 {
            bank.insert(refs, mv(v, 0), 0);
        }
        bank.insert(refs, mv(3, 0), 0);
        let order: Vec<i32> = bank.recent(refs).map(|e| e.mv[0].y).collect();
        assert_eq!(order, [3, 5, 4, 2]);
    }

    #[test]
    fn warp_ring() {
        let mut bank = WarpBank::default();
        let m = |a: i32| [a, 0, 1 << 16, a, 0, 1 << 16];
        for a in 0..3 {
            assert!(bank.insert(&m(a), 2));
        }
        // Same 2x2 part as the first: moves it, keeping its translation.
        assert!(bank.insert(&[9, 9, 1 << 16, 0, 0, 1 << 16], 2));
        let firsts: Vec<i32> = bank.recent(2).map(|m| m[0]).collect();
        assert_eq!(firsts, [0, 2, 1]);
    }
}
