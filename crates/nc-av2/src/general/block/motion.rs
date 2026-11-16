//! The motion of the blocks decoded so far, per 4x4 unit, the reference
//! MV bank, and the candidate list of an intra block copy vector (dav2d
//! `refmvs.h` `refmvs_tile`; `refmvs.c` `dav2d_refmvs_find`,
//! `dav2d_refmvs_tile_sbrow_init`, `dav2d_refmvs_reset_sb`, the above-row
//! backup of `dav2d_refmvs_save_tmvs`, `dav2d_refmvs_bank_update`,
//! `dav2d_refmvs_bank_add`, `splat_mv_c`; dav2d commit 237229e, BSD-2-Clause,
//! see `docs/licenses/dav2d`).
//!
//! Minimal for intra block copy, and built to be extended by the inter
//! group: a unit is intra or a block copy ([`Motion`]: add the inter
//! variant with its references and vectors); the bank holds only its
//! class of block-copy vectors (dav2d class 8 with the reference pair
//! "none": add the other classes); the list ([`MotionStore::block_copy_candidates`])
//! is `dav2d_refmvs_find` for the "none" pair, whose spatial candidates
//! are block copies only (add the reference matching, temporal, derived
//! and global candidates). The temporal store (`rp`) is not kept: general
//! frames keep no motion field for later frames yet.
use super::size::BlockSize;

/// The units kept: 64 rows (a 256x256 superblock row) by 128 columns
/// (two of its superblocks: the current one and the one to its left),
/// indexed modulo their extent (dav2d `rt->r`).
const RING_ROWS: u32 = 64;
const RING_COLUMNS: u32 = 128;
/// Eighth-sample vector units per sample, as a shift.
const VECTOR_FRACTION_BITS: u32 = 3;
/// Candidate positions per 4x4 unit in vector units: the clip margin of a
/// candidate is 4 units past the block (dav2d `minx` .. `maxy`).
const CLIP_MARGIN_UNITS: i32 = 4;
const VECTOR_UNITS_PER_UNIT: i32 = 32;
/// The list holds up to six candidates.
const MAX_CANDIDATES: usize = 6;
/// Duplicate checks stop once this many comparisons were made (dav2d
/// `iter_cntr` against `max_iter`).
const MAX_COMPARISONS: u32 = 16;
/// DRL reordering of the nearest candidates: always, or when no temporal
/// candidate is high priority and four or more are nearest
/// (`enable_drl_reorder`).
const DRL_REORDER_ALWAYS: u8 = 2;
const DRL_REORDER_CONSTRAINED: u8 = 1;
const REORDER_MIN_ALWAYS: usize = 2;
const REORDER_MIN_CONSTRAINED: usize = 4;
/// Blocks this many units or less on a side read their above-right and
/// below-left neighbours.
const CORNER_MAX_UNITS: u32 = 16;
/// The block copy's default vectors look this far left past a
/// superblock's width, in samples.
pub const DEFAULT_LEFT_EXTRA: i32 = 256;
/// Bank entries per class.
const BANK_ENTRIES: u8 = 4;
const BANK_MASK: u8 = BANK_ENTRIES - 1;
/// The bank stops taking vectors after this many per superblock, or per
/// bank unit (dav2d `hits`), and starts each superblock with room for at
/// least this many.
const BANK_MAX_SUPERBLOCK_HITS: u8 = 64;
const BANK_MAX_UNIT_HITS: u8 = 16;
const BANK_MIN_AVAILABLE: u8 = 4;
/// At a superblock's start the bank takes at most this many vectors from
/// the row above.
const BANK_REFILL_MAX: u32 = 4;

/// A motion vector in eighth samples (dav2d `mv`). 32-bit components:
/// projected and predicted vectors are clamped to +-0xffff, 17 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mv {
    /// Down, in eighth samples.
    pub y: i32,
    /// Right, in eighth samples.
    pub x: i32,
}

impl Mv {
    /// Scaled by `num / den` display-order distances (dav2d
    /// `dav2d_mv_projection`): `den` below 32, `num` within +-31.
    #[must_use]
    pub fn project(self, num: i32, den: i32) -> Self {
        /// `2^14 / den` (dav2d `div_mult`).
        const RECIPROCALS: [i32; 32] = [
            0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170,
            1092, 1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
        ];
        const SHIFT: u32 = 14;
        const LIMIT: i64 = 0xffff;
        let factor = i64::from(num * RECIPROCALS[den as usize]);
        let scale = |v: i32| {
            let product = i64::from(v) * factor;
            ((product + (1 << (SHIFT - 1)) - i64::from(product < 0)) >> SHIFT).clamp(-LIMIT, LIMIT)
                as i32
        };
        Self {
            y: scale(self.y),
            x: scale(self.x),
        }
    }

    /// Rounded to whole samples (dav2d `fix_int_mv_precision`): to the
    /// nearest, halves toward zero.
    #[must_use]
    pub fn to_integer(self) -> Self {
        let round = |v: i32| (v + i32::from(v < 0) + 3) & !7;
        Self {
            y: round(self.y),
            x: round(self.x),
        }
    }
}

/// What a 4x4 unit was predicted from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Motion {
    /// Intra prediction, or not decoded yet (dav2d `INVALID_MV` with no
    /// reference).
    Intra,
    /// Intra block copy with its vector (no reference).
    BlockCopy(Mv),
}

/// One 4x4 unit's record (dav2d `refmvs_block`'s part block copies read).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionUnit {
    /// What it was predicted from.
    pub motion: Motion,
    /// The size of the block the unit belongs to.
    pub size: BlockSize,
}

/// A unit nothing has written yet (dav2d's zeroed allocation: its size
/// is never read before a block writes it).
const EMPTY: MotionUnit = MotionUnit {
    motion: Motion::Intra,
    size: BlockSize::B256x256,
};

/// One candidate of the list: its vector and weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Candidate {
    /// The vector.
    pub mv: Mv,
    /// How strongly the neighbours suggest it (the list's reorder key).
    pub weight: u16,
}

/// The candidate list of a block (dav2d `mvstack`, zero beyond `count`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidates {
    /// The candidates in order.
    pub list: [Candidate; MAX_CANDIDATES],
    /// How many were found.
    pub count: usize,
}

/// The reference MV bank's class of block-copy vectors: a ring of four
/// with the most recent last, and the limits on how many a superblock
/// adds (dav2d `rt->bank`, class 8).
#[derive(Debug, Clone, Copy, Default)]
struct Bank {
    vectors: [Mv; BANK_ENTRIES as usize],
    size: u8,
    index: u8,
    /// Vectors added in this superblock, and in this bank unit.
    superblock_hits: u8,
    unit_hits: u8,
    /// How many more vectors the superblock may add.
    available: u8,
}

impl Bank {
    /// Add a vector, or move an equal one to the most recent place (dav2d
    /// `refmvs_bank_add`).
    fn insert(&mut self, mv: Mv) {
        self.superblock_hits += 1;
        let found = (0..self.size)
            .find(|&n| self.vectors[usize::from(self.index.wrapping_add(n) & BANK_MASK)] == mv);
        if let Some(n) = found {
            let to = if self.size == BANK_ENTRIES {
                self.index.wrapping_sub(1) & BANK_MASK
            } else {
                self.size - 1
            };
            let mut from = self.index.wrapping_add(n) & BANK_MASK;
            let moved = self.vectors[usize::from(from)];
            while from != to {
                let next = (from + 1) & BANK_MASK;
                self.vectors[usize::from(from)] = self.vectors[usize::from(next)];
                from = next;
            }
            self.vectors[usize::from(to)] = moved;
            return;
        }
        let target = if self.size == BANK_ENTRIES {
            let target = self.index & BANK_MASK;
            self.index = self.index.wrapping_add(1);
            target
        } else {
            self.size += 1;
            self.size - 1
        };
        self.vectors[usize::from(target)] = mv;
    }

    /// The vectors from the most recent back.
    fn recent(&self) -> impl Iterator<Item = Mv> + '_ {
        let start = self.size.wrapping_add(self.index).wrapping_sub(1);
        (0..self.size).map(move |n| self.vectors[usize::from(start.wrapping_sub(n) & BANK_MASK)])
    }
}

/// Where the store's frame and tile lie, in 4x4 units.
#[derive(Debug, Clone, Copy)]
pub struct StoreGeometry {
    /// The frame's width and height (8x8-rounded: dav2d `iw4`, `ih4`).
    pub columns: u32,
    /// See `columns`.
    pub rows: u32,
    /// The superblock side, as a log2 of luma samples (6, 7 or 8).
    pub superblock_log2: u8,
    /// The tile's first column.
    pub column_start: u32,
    /// The tile's column end (the frame's, at most).
    pub column_end: u32,
    /// The tile's first row.
    pub row_start: u32,
    /// The tile's row end (the frame's, at most).
    pub row_end: u32,
}

impl StoreGeometry {
    fn superblock_units(&self) -> u32 {
        1 << (self.superblock_log2 - 2)
    }
}

/// The motion of one tile (dav2d `refmvs_tile` with the frame's above row
/// `rf->ra`).
#[derive(Debug, Clone)]
pub struct MotionStore {
    geometry: StoreGeometry,
    units: Vec<MotionUnit>,
    /// Per 8x8 column of the frame, the unit above the current superblock
    /// row (its odd row, even column), and the one that was above the
    /// current superblock's top-left before its left neighbour replaced it.
    above: Vec<MotionUnit>,
    above_left: MotionUnit,
    /// The bank, when the sequence has one (`enable_refmvbank`).
    bank: Option<Bank>,
}

impl MotionStore {
    /// A store for a tile; `bank` when the sequence uses the MV bank.
    #[must_use]
    pub fn new(geometry: StoreGeometry, bank: bool) -> Self {
        Self {
            geometry,
            units: vec![EMPTY; (RING_ROWS * RING_COLUMNS) as usize],
            above: vec![EMPTY; geometry.columns.div_ceil(2) as usize],
            above_left: EMPTY,
            bank: bank.then(Bank::default),
        }
    }

    fn index(row: u32, column: u32) -> usize {
        ((row % RING_ROWS) * RING_COLUMNS + column % RING_COLUMNS) as usize
    }

    fn unit(&self, row: u32, column: u32) -> MotionUnit {
        self.units[Self::index(row, column)]
    }

    /// At each superblock row of the tile: the bank starts empty (dav2d
    /// `dav2d_refmvs_tile_sbrow_init`).
    pub fn start_row(&mut self) {
        if let Some(bank) = &mut self.bank {
            *bank = Bank::default();
        }
    }

    /// At each superblock: its units are not decoded yet, the bank's
    /// limits restart, and in inter frames below the tile's first row
    /// (`refill`) the bank takes up to four vectors from the row above
    /// (dav2d `dav2d_refmvs_reset_sb`).
    pub fn start_superblock(&mut self, row: u32, column: u32, refill: bool) {
        let side = self.geometry.superblock_units();
        for y in row..row + side {
            for x in column..column + side {
                self.units[Self::index(y, x)].motion = Motion::Intra;
            }
        }
        let Some(bank) = &mut self.bank else {
            return;
        };
        (bank.superblock_hits, bank.unit_hits, bank.available) = (0, 0, 0);
        if !refill {
            return;
        }
        let end = (column + side).min(self.geometry.column_end);
        let (mut x, mut hits) = (column, 0);
        while x < end {
            let above = self.above[(x >> 1) as usize];
            x += above.size.units().0;
            if let Motion::BlockCopy(mv) = above.motion {
                bank.insert(mv);
                hits += 1;
                if hits == BANK_REFILL_MAX {
                    break;
                }
            }
        }
    }

    /// After each superblock: keep its bottom row (at 8x8 resolution) for
    /// the superblock row below (dav2d `dav2d_refmvs_save_tmvs`'s backup).
    pub fn end_superblock(&mut self, row: u32, column: u32) {
        let side = self.geometry.superblock_units();
        // Superblock positions and sides are even in 4x4 units.
        let (row8, column8, side8) = (row >> 1, column >> 1, side >> 1);
        let row_end8 = (row8 + side8).min(self.geometry.rows >> 1);
        let column_end8 = (column8 + side8).min(self.geometry.columns.div_ceil(2));
        let source_row = (row_end8 - 1) * 2 + 1;
        self.above_left = self.above[(column_end8 - 1) as usize];
        for x8 in column8..column_end8 {
            self.above[x8 as usize] = self.unit(source_row, x8 * 2);
        }
    }

    /// Record a block's motion over its units (dav2d `splat_mv_c`).
    pub fn splat(&mut self, row: u32, column: u32, size: BlockSize, motion: Motion) {
        let (bw4, bh4) = size.units();
        let unit = MotionUnit { motion, size };
        for y in row..row + bh4 {
            let start = Self::index(y, column);
            self.units[start..start + bw4 as usize].fill(unit);
        }
    }

    /// The bank's limits at a block (dav2d `dav2d_refmvs_bank_update`):
    /// each superblock, and each bank unit of it (a 2, 4 or 8 unit square
    /// by superblock size), makes room for a vector per bank unit the
    /// block covers.
    pub fn update_bank(&mut self, row: u32, column: u32, size: BlockSize) {
        let side = self.geometry.superblock_units();
        let shift = u32::from(self.geometry.superblock_log2) - 5;
        let Some(bank) = &mut self.bank else {
            return;
        };
        let (bw4, bh4) = size.units();
        let room = ((bw4 >> shift).max(1) * (bh4 >> shift).max(1)) as u8;
        if (row | column) & (side - 1) == 0 {
            bank.unit_hits = 0;
            bank.available = room.max(BANK_MIN_AVAILABLE);
        } else if (row | column) & ((1 << shift) - 1) == 0 {
            bank.unit_hits = 0;
            bank.available += room;
        }
    }

    /// Offer a block copy's vector to the bank (dav2d
    /// `dav2d_refmvs_bank_add`).
    pub fn add_to_bank(&mut self, row: u32, column: u32, size: BlockSize, mv: Mv) {
        self.update_bank(row, column, size);
        let Some(bank) = &mut self.bank else {
            return;
        };
        if bank.superblock_hits >= BANK_MAX_SUPERBLOCK_HITS
            || bank.unit_hits >= BANK_MAX_UNIT_HITS
            || bank.available == 0
        {
            return;
        }
        bank.unit_hits += 1;
        bank.available -= 1;
        bank.insert(mv);
    }

    /// The candidate list of a block copy at (`row`, `column`): spatial
    /// block-copy neighbours, reordered by weight, then the bank's
    /// vectors, clipped near the frame, then the default vectors (dav2d
    /// `dav2d_refmvs_find` with the reference pair "none"); up to
    /// `1 + max_drl_bits` are listed, and `drl_reorder` is the
    /// sequence's `enable_drl_reorder`.
    #[must_use]
    pub fn block_copy_candidates(
        &self,
        (row, column): (u32, u32),
        size: BlockSize,
        drl_reorder: u8,
        max_drl_bits: u8,
    ) -> Candidates {
        let mut list = CandidateList::default();
        let nearest = self.spatial_candidates(&mut list, (row, column), size);
        if (drl_reorder == DRL_REORDER_ALWAYS && nearest >= REORDER_MIN_ALWAYS)
            || (drl_reorder == DRL_REORDER_CONSTRAINED && nearest >= REORDER_MIN_CONSTRAINED)
        {
            list.promote_heaviest(nearest);
        }
        let limit = 1 + usize::from(max_drl_bits);
        self.bank_candidates(&mut list, (row, column), size, limit);
        let (bw4, bh4) = size.units();
        let g = &self.geometry;
        let bound = |position: u32, extent: u32, frame: u32| {
            let position = position as i32;
            (
                -(position + extent as i32 + CLIP_MARGIN_UNITS) * VECTOR_UNITS_PER_UNIT,
                (frame as i32 - position + CLIP_MARGIN_UNITS) * VECTOR_UNITS_PER_UNIT,
            )
        };
        let (min_x, max_x) = bound(column, bw4, g.columns);
        let (min_y, max_y) = bound(row, bh4, g.rows);
        for candidate in &mut list.list[..list.count] {
            candidate.mv.y = candidate.mv.y.clamp(min_y, max_y);
            candidate.mv.x = candidate.mv.x.clamp(min_x, max_x);
        }
        // The default vectors: a superblock up, a superblock (and 256
        // samples) left, the block's height up, its width left.
        let superblock = 1_i32 << g.superblock_log2;
        let unit = |units: u32| -(units as i32 * VECTOR_UNITS_PER_UNIT);
        let defaults = [
            Mv {
                y: -(superblock << VECTOR_FRACTION_BITS),
                x: 0,
            },
            Mv {
                y: 0,
                x: -((superblock + DEFAULT_LEFT_EXTRA) << VECTOR_FRACTION_BITS),
            },
            Mv { y: unit(bh4), x: 0 },
            Mv { y: 0, x: unit(bw4) },
        ];
        for mv in defaults {
            if list.count >= limit {
                break;
            }
            list.list[list.count] = Candidate { mv, weight: 0 };
            list.count += 1;
        }
        Candidates {
            list: list.list,
            count: list.count,
        }
    }

    /// The spatial candidates in dav2d's order: bottom-most left,
    /// right-most above, top-most left, left-most above, below-left,
    /// above-right, above-left, then two further left; returns how many
    /// came before the further ones (the nearest).
    fn spatial_candidates(
        &self,
        list: &mut CandidateList,
        (row, column): (u32, u32),
        size: BlockSize,
    ) -> usize {
        let g = &self.geometry;
        let side = g.superblock_units();
        let (bw4, bh4) = size.units();
        let w4 = bw4.min(g.column_end - column);
        let h4 = bh4.min(g.row_end - row);
        let have_left = column > g.column_start;
        let bottom_left_unit =
            (have_left && bh4 == h4).then(|| self.unit(row + bh4 - 1, column - 1));
        let above = self.above_neighbours((row, column), (bw4, w4));
        if let Some(unit) = bottom_left_unit {
            list.add_spatial(unit, 1);
        }
        if let Some((unit, position)) = above.right_most {
            list.add_spatial(unit, u16::from(position >= 0));
        }
        let top_left_unit = (have_left && bh4 > 1).then(|| self.unit(row, column - 1));
        if let Some(unit) = top_left_unit {
            list.add_spatial(unit, 1);
        }
        if let Some(unit) = above.left_most {
            list.add_spatial(unit, u16::from(above.offset == 0));
        }
        let below = row + bh4;
        if have_left && bh4 <= CORNER_MAX_UNITS && below & (side - 1) != 0 && below < g.row_end {
            list.add_spatial(self.unit(below, column - 1), 1);
        }
        if let Some(unit) = above.top_right {
            list.add_spatial(unit, 1);
        }
        if let Some(unit) = above.top_left {
            list.add_spatial(unit, 0);
        }
        let nearest = list.count;
        if have_left {
            // Two and three units further left, where a different block
            // lies there.
            let step = 3 - (column & u32::from(bw4 == 1));
            if column >= g.column_start + step {
                let further = column - step;
                let differs = |unit: MotionUnit, near: MotionUnit| {
                    unit.size.units().0 < step || near.size != unit.size
                };
                for (near, unit_row) in [(bottom_left_unit, row + bh4 - 1), (top_left_unit, row)] {
                    if let Some(near) = near {
                        let unit = self.unit(unit_row, further);
                        if differs(unit, near) {
                            list.add_spatial(unit, 0);
                        }
                    }
                }
            }
        }
        nearest
    }

    /// The above neighbours of a block `bw4` units wide (`w4` inside the
    /// tile): in the row above within the superblock, or at a superblock's
    /// top from the 8x8 backup row.
    fn above_neighbours(&self, (row, column): (u32, u32), (bw4, w4): (u32, u32)) -> Above {
        let g = &self.geometry;
        let side = g.superblock_units();
        let mut above = Above::default();
        if row <= g.row_start {
            return above;
        }
        let have_left = column > g.column_start;
        if row & (side - 1) == 0 {
            let offset = column & 1;
            let width = (bw4 + 1) & !1;
            let at = |x8: u32| self.above[x8 as usize];
            above.offset = offset;
            if (column - offset) as i32 - 2 >= g.column_start as i32 {
                let unit = if column & (side - 2) != 0 {
                    at((column >> 1) - 1)
                } else {
                    self.above_left
                };
                above.top_left = Some(unit);
            }
            if bw4 > 2 {
                above.left_most = Some(at(column >> 1));
            }
            if bw4 == w4 {
                let position = width as i32 - 2 - offset as i32;
                above.right_most = Some((at((column >> 1) + (width >> 1) - 1), position));
            }
            if column - offset + width < g.column_end && bw4 <= CORNER_MAX_UNITS {
                above.top_right = Some(at((column >> 1) + (width >> 1)));
            }
        } else {
            if have_left {
                above.top_left = Some(self.unit(row - 1, column - 1));
            }
            if bw4 > 1 {
                above.left_most = Some(self.unit(row - 1, column));
            }
            if bw4 == w4 {
                above.right_most = Some((self.unit(row - 1, column + bw4 - 1), bw4 as i32 - 1));
            }
            let right = column + bw4;
            if right & (side - 1) != 0 && right < g.column_end && bw4 <= CORNER_MAX_UNITS {
                // Not decoded yet when still marked intra-or-undecoded:
                // only block copies count anyway.
                above.top_right =
                    Some(self.unit(row - 1, right)).filter(|unit| unit.motion != Motion::Intra);
            }
        }
        above
    }

    /// The bank's vectors, most recent first, that are not listed and
    /// point into the frame, up to `limit` candidates.
    fn bank_candidates(
        &self,
        list: &mut CandidateList,
        (row, column): (u32, u32),
        size: BlockSize,
        limit: usize,
    ) {
        let Some(bank) = &self.bank else {
            return;
        };
        let (bw4, bh4) = size.units();
        let g = &self.geometry;
        let samples = |units: u32| (units << 2) as i32;
        for mv in bank.recent() {
            if list.count >= limit {
                break;
            }
            if list.find(mv).is_some() {
                continue;
            }
            // The whole-sample position, rounded toward zero.
            let whole = |v: i32| {
                let magnitude = (v.unsigned_abs() >> VECTOR_FRACTION_BITS) as i32;
                if v < 0 { -magnitude } else { magnitude }
            };
            let x = samples(column) + whole(mv.x);
            let y = samples(row) + whole(mv.y);
            if x <= -samples(bw4)
                || y <= -samples(bh4)
                || x >= samples(g.columns)
                || y >= samples(g.rows)
            {
                continue;
            }
            list.push(Candidate { mv, weight: 0 });
        }
    }
}

/// The above neighbours of a block (dav2d `tl`, `lmt`, `rmt`, `tr`) and
/// the column offset at a superblock's top (`x_off`); the right-most one
/// with its position relative to the block (its weight is 1 when not
/// negative).
#[derive(Debug, Clone, Copy, Default)]
struct Above {
    top_left: Option<MotionUnit>,
    left_most: Option<MotionUnit>,
    right_most: Option<(MotionUnit, i32)>,
    top_right: Option<MotionUnit>,
    offset: u32,
}

/// The list being built, with dav2d's count of duplicate comparisons.
#[derive(Debug, Clone, Copy, Default)]
struct CandidateList {
    list: [Candidate; MAX_CANDIDATES],
    count: usize,
    comparisons: u32,
}

impl CandidateList {
    /// The index of an equal vector, while comparisons remain (dav2d
    /// `iter_cntr`: each search costs the positions it looked at).
    fn find(&mut self, mv: Mv) -> Option<usize> {
        if self.comparisons >= MAX_COMPARISONS {
            return None;
        }
        let found = self.list[..self.count].iter().position(|c| c.mv == mv);
        self.comparisons += found.map_or(self.count, |m| m + 1) as u32;
        found
    }

    fn push(&mut self, candidate: Candidate) {
        if self.count < MAX_CANDIDATES {
            self.list[self.count] = candidate;
            self.count += 1;
        }
    }

    /// A spatial neighbour: a block copy's vector adds its weight to an
    /// equal candidate or joins the list (dav2d `add_spatial_candidate`,
    /// `add_candidate_sngl`).
    fn add_spatial(&mut self, unit: MotionUnit, weight: u16) {
        if self.count >= MAX_CANDIDATES {
            return;
        }
        let Motion::BlockCopy(mv) = unit.motion else {
            return;
        };
        match self.find(mv) {
            Some(m) => self.list[m].weight += weight,
            None => self.push(Candidate { mv, weight }),
        }
    }

    /// Swap the heaviest of the first `nearest` (the first of equals) to
    /// the front.
    fn promote_heaviest(&mut self, nearest: usize) {
        let mut heaviest = 0;
        for n in 1..nearest {
            if self.list[n].weight > self.list[heaviest].weight {
                heaviest = n;
            }
        }
        self.list.swap(0, heaviest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> StoreGeometry {
        StoreGeometry {
            columns: 32,
            rows: 32,
            superblock_log2: 6,
            column_start: 0,
            column_end: 32,
            row_start: 0,
            row_end: 32,
        }
    }

    /// A block with nothing coded beside it gets the four defaults.
    #[test]
    fn defaults_alone() {
        let store = MotionStore::new(geometry(), true);
        let list = store.block_copy_candidates((0, 0), BlockSize::B16x16, 2, 3);
        let mvs: Vec<_> = list.list[..list.count]
            .iter()
            .map(|c| (c.mv.y, c.mv.x))
            .collect();
        assert_eq!(mvs, [(-512, 0), (0, -2560), (-128, 0), (0, -128)]);
    }

    /// The left neighbour's copy comes first; the bank adds a different
    /// one; equal vectors are not repeated.
    #[test]
    fn neighbour_then_bank() {
        let mut store = MotionStore::new(geometry(), true);
        let left = Mv { y: -64, x: -8 };
        store.splat(4, 0, BlockSize::B16x16, Motion::BlockCopy(left));
        store.add_to_bank(4, 0, BlockSize::B16x16, left);
        store.add_to_bank(0, 0, BlockSize::B16x16, Mv { y: -96, x: 0 });
        let list = store.block_copy_candidates((4, 4), BlockSize::B16x16, 2, 3);
        let mvs: Vec<_> = list.list[..list.count]
            .iter()
            .map(|c| (c.mv.y, c.mv.x))
            .collect();
        assert_eq!(mvs, [(-64, -8), (-96, 0), (-512, 0), (0, -2560)]);
    }

    #[test]
    fn integer_rounding() {
        let mv = Mv { y: -12, x: 13 }.to_integer();
        assert_eq!((mv.y, mv.x), (-8, 16));
    }
}
