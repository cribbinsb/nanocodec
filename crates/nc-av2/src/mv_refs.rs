//! Motion-vector reference lists (DRL) for single-reference AV2 blocks of
//! any square size from 8×8 to 64×64.
//!
//! AVM v1.0.0 `setup_ref_mv_list`, `scan_blk_mbmi`, `add_ref_mv_candidate`,
//! `get_row_smvp_states`, `has_top_right`, `has_bottom_left`,
//! `is_valid_candidate`, `add_ref_mv_bank_candidates`, `check_rmb_cand`,
//! `insert_mvp_candidate`, `clamp_mv_ref`, `update_ref_mv_bank`,
//! `av2_reset_refmv_bank` and `av2_find_mode_ctx`
//! (`av2/common/mvref_common.c`, `av2_common_int.h`) define the list, in
//! this order:
//!
//! 1. the adjacent spatial neighbours and the non-adjacent left column;
//! 2. the reference MV bank (recent vectors of the superblock row), when
//!    the sequence enables it, up to `max_drl_bits + 1` entries;
//! 3. clamping, then the global vector (zero: identity global motion);
//! 4. for blocks larger than 32×32 on both sides, row/column combinations
//!    of the first three entries.
//!
//! Duplicates are pruned only while the list's comparison budget lasts
//! (AVM `MAX_PR_NUM`). The sequence disables DRL reordering, temporal
//! (TMVP) candidates and warp, and every block uses the one reference, so
//! derived (other-reference) candidates never appear and the weights AVM
//! keeps never matter.
//!
//! Motion is stored per 8×8 cell. AVM reads 4×4 units, but every coded block
//! here covers whole cells, and the superblock-boundary rule of
//! `get_row_smvp_states` (8×8-aligned above positions) moves a position
//! only within its cell, except that it drops an 8-wide block's second
//! above position; so the cell grid sees exactly AVM's candidates and
//! duplicate comparisons.
//! `has_top_right` and `has_bottom_left` reduce to "already coded" for
//! 64×64 superblocks, which the grid records directly. See `THIRD_PARTY.md`
//! for reference terms.
mod compound;

use crate::entropy::{Reader, Writer};
pub use compound::{SKIP_MODE, SkipCandidates, read_skip_drl, skip_drl_bits, write_skip_drl};
use nc_common::decode::{DecodeError, Result};
use nc_encoder::symbol_cost::symbol_bits;

type Vector = (i32, i32);
/// The two vectors of a same-reference compound candidate (AVM `this_mv`,
/// `comp_mv`).
pub type Pair = (Vector, Vector);

/// Motion grid granularity, in luma samples.
const CELL: usize = 8;
/// Largest block side the scan supports (64×64 superblocks).
const MAX_BLOCK: usize = 64;
/// Superblock side, in luma samples.
const SUPERBLOCK: usize = 64;
/// Largest picture side of the subset.
const MAX_DIMENSION: usize = 4096;
/// Column of the non-adjacent left candidates, in luma samples left of the
/// block (AVM `MVREF_COLS = 2`: column offset `-(2 << 1) + 1` 4×4 units).
const NON_ADJACENT_COLUMN: i32 = 12;
/// AVM `MV_BORDER` (16 samples in eighth-sample units) used by `clamp_mv_ref`.
const MV_BORDER: i32 = 128;
/// Eighth-sample units per luma sample.
const EIGHTHS: i32 = 8;
/// Largest stored vector component magnitude.
const MAX_COMPONENT: i32 = 32766;
/// AVM `MAX_REF_MV_STACK_SIZE`: entries of a reference list.
pub const MAX_STACK: usize = 6;
/// AVM `MAX_PR_NUM`: duplicate comparisons a list may spend; later
/// candidates are appended unpruned.
const MAX_PRUNE_COMPARISONS: u32 = 16;
/// Blocks get row/column combination candidates only when both sides
/// exceed 8 4×4 units, 32 samples (AVM `max_ext_stack_size`, on
/// `xd->width` / `xd->height` in 4×4 units).
const MIN_EXTENSION_SIDE: usize = 64;
/// The combination candidates in AVM order: (row source, column source)
/// list indices (`setup_ref_mv_list`, "Add (`0_y`, `1_x`)" and so on).
const EXTENSION_PAIRS: [(usize, usize); 6] = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)];
/// Spatial positions a list scans ([`ReferenceGrid::scan`]).
const SCAN_POSITIONS: usize = 9;
/// AVM `REF_MV_BANK_SIZE`.
const BANK_SIZE: usize = 4;
/// AVM `MAX_RMB_SB_HITS`: bank updates per superblock.
const BANK_SUPERBLOCK_UPDATES: u32 = 64;
/// AVM `BANK_SB_ABOVE_ROW_MAX_HITS`: blocks of the row above loaded into
/// the bank at a superblock's start.
const BANK_ABOVE_ROW_UPDATES: usize = 4;
/// DRL symbols beyond the third share the third's models
/// (AVM `av2_get_drl_cdf`: `drl_cdf[AVMMIN(idx, 2)]`).
const DRL_MODEL_TYPES: usize = 3;
/// AVM `MIN_MAX_DRL_BITS` and `MAX_MAX_DRL_BITS` (`MAX_REF_MV_STACK_SIZE - 1`).
const DRL_BITS: std::ops::RangeInclusive<u8> = 1..=(MAX_STACK as u8 - 1);

/// The sequence's reference-list configuration (AVM `def_max_drl_bits`,
/// `enable_refmvbank`; frames never override the DRL size).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListConfig {
    /// DRL symbols at most: `max_drl_bits + 1` selectable entries, 1..=5.
    pub max_drl_bits: u8,
    /// The reference MV bank adds recent vectors of the superblock row.
    pub ref_mv_bank: bool,
}

impl ListConfig {
    /// Two entries, no bank: the smallest list.
    pub const BASIC: Self = Self {
        max_drl_bits: 1,
        ref_mv_bank: false,
    };

    /// Reject a DRL size outside AVM's range.
    /// # Errors
    /// Rejects `max_drl_bits` outside 1..=5.
    pub fn validate(self) -> Result<Self> {
        if DRL_BITS.contains(&self.max_drl_bits) {
            Ok(self)
        } else {
            Err(DecodeError::Malformed("AV2 max_drl_bits"))
        }
    }

    /// Selectable list entries.
    fn entries(self) -> usize {
        usize::from(self.max_drl_bits) + 1
    }
}

/// A stored vector. Components fit `i16`: [`ReferenceGrid::commit`]
/// rejects magnitudes above [`MAX_COMPONENT`]. The narrow fields keep a
/// [`Cell`] small (the grid is read for every candidate list, and on the
/// Jetson its cache footprint shows).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Narrow {
    row: i16,
    col: i16,
}

impl Narrow {
    fn new((row, col): Vector) -> Self {
        let narrow = |v: i32| i16::try_from(v).expect("components are checked on commit");
        Self {
            row: narrow(row),
            col: narrow(col),
        }
    }

    fn vector(self) -> Vector {
        (i32::from(self.row), i32::from(self.col))
    }
}

/// The motion of an inter block: one vector, or (`second`) the two vectors
/// of a same-reference compound block (AVM `ref_frame` `{LAST, LAST}`,
/// `mv[0]` and `mv[1]`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Entry {
    first: Narrow,
    second: Option<Narrow>,
    /// The (first) vector was coded as NEWMV (counted by mode contexts).
    is_new: bool,
}

impl Entry {
    fn vector(self) -> Vector {
        self.first.vector()
    }

    fn pair(self) -> Option<Pair> {
        self.second
            .map(|second| (self.first.vector(), second.vector()))
    }
}

/// The motion an inter block commits to the grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockMotion {
    /// A single-reference vector.
    Single {
        /// The block's vector.
        vector: Vector,
        /// Coded as NEWMV.
        is_new: bool,
    },
    /// A same-reference compound block (skip mode: `NEAR_NEARMV`).
    Compound(Pair),
}

/// Coding state of one 8×8 cell. Intra blocks are coded but never match
/// the single reference frame, so they add no candidates or context.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Cell {
    Uncoded,
    /// A coded block, identified by its first column (AVM `mi_col_start`,
    /// enough to tell blocks apart within one row; below
    /// [`MAX_DIMENSION`], so `u16`), with its motion when inter.
    Coded {
        column: u16,
        motion: Option<Entry>,
    },
}

/// How an inter block's vector is signalled (AVM `inter_single_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterMode {
    /// A DRL candidate vector, unchanged.
    Near,
    /// Identity global motion: the zero vector.
    Global,
    /// A DRL candidate plus a coded difference.
    New,
}

impl InterMode {
    const ALL: [Self; 3] = [Self::Near, Self::Global, Self::New];

    fn symbol(self) -> usize {
        self as usize
    }

    /// Whether the mode codes a DRL index.
    #[must_use]
    pub const fn has_drl(self) -> bool {
        !matches!(self, Self::Global)
    }
}

/// A block's selectable DRL references and their common mode context.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidates {
    /// Eighth-pel row/column predictors. Entries from `count` on are zero:
    /// AVM initialises the list to zero vectors, and a DRL index may name
    /// any of the `max_drl_bits + 1` entries.
    pub vectors: [Vector; MAX_STACK],
    /// Derived selectable entries, 1..=`max_drl_bits + 1`.
    pub count: u8,
    /// The sequence's DRL size.
    pub max_drl_bits: u8,
    /// AV2 single-inter-mode and DRL context, in 0..=4.
    pub mode_context: u8,
    /// Approximate bits of each inter mode symbol and of each DRL index in
    /// this context, priced once per list: the engine asks for every
    /// candidate vector of a block ([`mode_bits`]).
    mode_symbol_bits: [f32; InterMode::ALL.len()],
    drl_index_bits: [f32; MAX_STACK],
}

impl Candidates {
    fn new(vectors: [Vector; MAX_STACK], count: usize, max_drl_bits: u8, mode_context: u8) -> Self {
        let context = usize::from(mode_context);
        let drl_index_bits = std::array::from_fn(|index| {
            drl_symbols(max_drl_bits, index)
                .map(|(kind, symbol)| symbol_bits(&[DRL[kind][context], 0], symbol))
                .sum()
        });
        Self {
            vectors,
            count: count as u8,
            max_drl_bits,
            mode_context,
            mode_symbol_bits: InterMode::ALL.map(|mode| symbol_bits(&MODE[context], mode.symbol())),
            drl_index_bits,
        }
    }

    /// The predictor DRL index `index` names.
    /// # Errors
    /// Rejects an index beyond the DRL size.
    pub fn vector(&self, index: usize) -> Result<Vector> {
        if index > usize::from(self.max_drl_bits) {
            return Err(DecodeError::Malformed("AV2 DRL index"));
        }
        Ok(self.vectors[index])
    }

    /// The derived (selectable) predictors.
    #[must_use]
    pub fn derived(&self) -> &[Vector] {
        &self.vectors[..usize::from(self.count)]
    }
}

/// A reference list under construction (AVM `ref_mv_stack`, `refmv_count`,
/// `drl_pr_count`).
#[derive(Debug, Default)]
struct Stack {
    vectors: [Vector; MAX_STACK],
    count: usize,
    comparisons: u32,
}

impl Stack {
    /// Whether `vector` is not yet listed. While the comparison budget
    /// lasts, every comparison made counts against it; after that nothing
    /// is pruned.
    fn is_new(&mut self, vector: Vector) -> bool {
        if self.comparisons >= MAX_PRUNE_COMPARISONS {
            return true;
        }
        for &listed in &self.vectors[..self.count] {
            self.comparisons += 1;
            if listed == vector {
                return false;
            }
        }
        true
    }

    /// Append `vector` unless the list is full or (budget permitting) it is
    /// listed already.
    fn add(&mut self, vector: Vector) {
        if self.count < MAX_STACK && self.is_new(vector) {
            self.push(vector);
        }
    }

    fn push(&mut self, vector: Vector) {
        self.vectors[self.count] = vector;
        self.count += 1;
    }
}

/// Recent motion of one reference type in the superblock row (one AVM
/// `rmb_buffer` list), oldest first.
#[derive(Debug, Default, Clone)]
struct BankList<T> {
    entries: [T; BANK_SIZE],
    count: usize,
}

impl<T: Copy + PartialEq> BankList<T> {
    /// Make `entry` the newest (AVM `update_ref_mv_bank`): a listed entry
    /// moves to the end, a new one drops the oldest when full.
    fn update(&mut self, entry: T) {
        let listed = &mut self.entries[..self.count];
        if let Some(index) = listed.iter().position(|&e| e == entry) {
            listed[index..].rotate_left(1);
        } else if self.count < BANK_SIZE {
            self.entries[self.count] = entry;
            self.count += 1;
        } else {
            self.entries.rotate_left(1);
            self.entries[BANK_SIZE - 1] = entry;
        }
    }

    /// Newest first.
    fn newest_first(&self) -> impl Iterator<Item = T> + '_ {
        self.entries[..self.count].iter().rev().copied()
    }
}

/// The reference MV bank (AVM `REF_MV_BANK`): the single-reference list
/// and the same-reference compound list (`{LAST, LAST}`), the only types
/// the subset codes.
#[derive(Debug, Default, Clone)]
struct Bank {
    single: BankList<Vector>,
    compound: BankList<Pair>,
    /// Updates in the current superblock, over both lists (AVM
    /// `rmb_sb_hits`). AVM also limits updates per 8×8 unit
    /// (`remain_hits`, `rmb_unit_hits`), but every block here starts on an
    /// 8×8 unit and adds at least one unit's allowance before its own
    /// update, so only this limit can bind.
    updates: u32,
}

impl Bank {
    fn update(&mut self, entry: Entry) {
        if self.updates >= BANK_SUPERBLOCK_UPDATES {
            return;
        }
        self.updates += 1;
        match entry.pair() {
            Some(pair) => self.compound.update(pair),
            None => self.single.update(entry.vector()),
        }
    }
}

/// A block's position and side, `(x, y, size)` in luma samples.
type BlockKey = (usize, usize, usize);

/// Coding-order motion state for one independently decoded tile.
/// Call [`Self::begin_superblock`] before each superblock and commit each
/// block immediately after coding it: the availability of diagonal
/// neighbours depends on the actual partition traversal, and the bank on
/// the coding order.
#[derive(Debug)]
pub struct ReferenceGrid {
    width: usize,
    height: usize,
    config: ListConfig,
    cells: Vec<Cell>,
    bank: Bank,
    /// The most recent block's candidate list, valid until the next commit.
    last: std::cell::Cell<Option<(BlockKey, Candidates)>>,
    /// The most recent block's skip-mode list, likewise.
    last_skip: std::cell::Cell<Option<(BlockKey, SkipCandidates)>>,
}

impl ReferenceGrid {
    /// Allocate one cell per 8×8 block in a positive multiple-of-eight picture.
    /// # Errors
    /// Rejects unsupported dimensions, an invalid configuration and
    /// excessive allocation.
    pub fn new(width: usize, height: usize, config: ListConfig) -> Result<Self> {
        if width == 0
            || height == 0
            || !width.is_multiple_of(CELL)
            || !height.is_multiple_of(CELL)
            || width > MAX_DIMENSION
            || height > MAX_DIMENSION
        {
            return Err(DecodeError::Unsupported("AV2 MV grid dimensions"));
        }
        Ok(Self {
            width,
            height,
            config: config.validate()?,
            cells: vec![Cell::Uncoded; width / CELL * (height / CELL)],
            bank: Bank::default(),
            last: std::cell::Cell::new(None),
            last_skip: std::cell::Cell::new(None),
        })
    }

    /// Validate a square block of side `size` at `(x, y)`, inside the picture.
    fn check_block(&self, x: usize, y: usize, size: usize) -> Result<()> {
        if !size.is_power_of_two()
            || !(CELL..=MAX_BLOCK).contains(&size)
            || !x.is_multiple_of(size)
            || !y.is_multiple_of(size)
            || x + size > self.width
            || y + size > self.height
        {
            return Err(DecodeError::Malformed("AV2 MV grid block position"));
        }
        Ok(())
    }

    fn cell(&self, x: i32, y: i32) -> Cell {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return Cell::Uncoded;
        }
        self.cells[y as usize / CELL * (self.width / CELL) + x as usize / CELL]
    }

    fn get(&self, x: i32, y: i32) -> Option<Entry> {
        match self.cell(x, y) {
            Cell::Coded { motion, .. } => motion,
            Cell::Uncoded => None,
        }
    }

    /// The non-adjacent left candidate in `row`, if it belongs to a block
    /// other than the left neighbour's (AVM `is_valid_candidate`).
    fn non_adjacent(&self, x: i32, row: i32) -> Option<Entry> {
        match (
            self.cell(x - NON_ADJACENT_COLUMN, row),
            self.cell(x - 1, row),
        ) {
            (Cell::Coded { column, motion }, Cell::Coded { column: inner, .. })
                if column != inner =>
            {
                motion
            }
            _ => None,
        }
    }

    /// Start the superblock whose top-left sample is `(x, y)`: the bank
    /// empties at each superblock row and then takes up to
    /// [`BANK_ABOVE_ROW_UPDATES`] inter blocks of the row above, left to
    /// right (AVM `decode_tile`, `av2_reset_refmv_bank`).
    /// # Errors
    /// Rejects a position off the superblock grid or outside the picture.
    pub fn begin_superblock(&mut self, x: usize, y: usize) -> Result<()> {
        if !x.is_multiple_of(SUPERBLOCK)
            || !y.is_multiple_of(SUPERBLOCK)
            || x >= self.width
            || y >= self.height
        {
            return Err(DecodeError::Malformed("AV2 superblock position"));
        }
        self.last.set(None);
        self.last_skip.set(None);
        if !self.config.ref_mv_bank {
            return Ok(());
        }
        if x == 0 {
            self.bank = Bank::default();
        }
        self.bank.updates = 0;
        if y == 0 {
            return Ok(());
        }
        let (above, count) = self.above_row_entries(x, y);
        for &entry in &above[..count] {
            self.bank.update(entry);
        }
        Ok(())
    }

    /// The motion of the first [`BANK_ABOVE_ROW_UPDATES`] inter blocks on
    /// the row above the superblock at `(x, y)`, left to right: each block
    /// once, at its first column (AVM steps by the block's width), intra
    /// blocks passed over.
    fn above_row_entries(&self, x: usize, y: usize) -> ([Entry; BANK_ABOVE_ROW_UPDATES], usize) {
        let row = y as i32 - 1;
        let end = (x + SUPERBLOCK).min(self.width);
        let unused = Entry {
            first: Narrow::new((0, 0)),
            second: None,
            is_new: false,
        };
        let mut entries = [unused; BANK_ABOVE_ROW_UPDATES];
        let mut count = 0;
        for column in (x..end).step_by(CELL) {
            if let Cell::Coded {
                column: start,
                motion: Some(entry),
            } = self.cell(column as i32, row)
                && usize::from(start) == column
            {
                entries[count] = entry;
                count += 1;
                if count == BANK_ABOVE_ROW_UPDATES {
                    break;
                }
            }
        }
        (entries, count)
    }

    /// Derive the selectable references before coding the inter block of
    /// side `size` at `(x, y)`.
    /// # Errors
    /// Rejects invalid geometry or missing mandatory above/left neighbours.
    pub fn candidates(&self, x: usize, y: usize, size: usize) -> Result<Candidates> {
        // The engine prices every candidate vector of a block (mode_bits),
        // each asking for the same list: derive it once until the grid
        // changes ([`Self::fill`] clears the memo).
        if let Some((key, candidates)) = self.last.get()
            && key == (x, y, size)
        {
            return Ok(candidates);
        }
        let candidates = self.derive_candidates(x, y, size)?;
        self.last.set(Some(((x, y, size), candidates)));
        Ok(candidates)
    }

    /// Reject a block whose left or above neighbour is not yet coded.
    #[inline]
    fn check_neighbours_coded(&self, px: i32, py: i32) -> Result<()> {
        if px > 0 && self.cell(px - 1, py) == Cell::Uncoded
            || py > 0 && self.cell(px, py - 1) == Cell::Uncoded
        {
            return Err(DecodeError::Malformed("AV2 motion grid coding order"));
        }
        Ok(())
    }

    /// The spatial candidates of the block of side `size` at `(x, y)` in
    /// AVM scan order (`setup_ref_mv_list`): the adjacent positions (left
    /// at the bottom row, above at the last column, left, above,
    /// bottom-left, top-right, top-left), then the non-adjacent column at
    /// the bottom and top rows; `None` for intra, unavailable or unscanned
    /// positions. (An array, not an iterator chain: the scan runs for every
    /// list the engine prices, and the chain cost ~0.5% of P frames.)
    #[inline]
    fn scan(&self, x: usize, y: usize, size: usize) -> [Option<Entry>; SCAN_POSITIONS] {
        let (px, py, last) = (x as i32, y as i32, size as i32 - 1);
        // On a superblock's top row an 8-wide block has no second above
        // position (`get_row_smvp_states`, `BLOCK_WIDTH_8`): its first one
        // already stands for the whole 8×8 cell above. Inside the superblock
        // both are scanned, and the second costs a duplicate comparison.
        let above_first_scanned = size > CELL || !y.is_multiple_of(SUPERBLOCK);
        [
            self.get(px - 1, py + last),
            self.get(px + last, py - 1),
            self.get(px - 1, py),
            above_first_scanned.then(|| self.get(px, py - 1)).flatten(),
            self.get(px - 1, py + last + 1),
            self.get(px + last + 1, py - 1),
            self.get(px - 1, py - 1),
            self.non_adjacent(px, py + last),
            self.non_adjacent(px, py),
        ]
    }

    fn derive_candidates(&self, x: usize, y: usize, size: usize) -> Result<Candidates> {
        self.check_block(x, y, size)?;
        let (px, py, last) = (x as i32, y as i32, size as i32 - 1);
        self.check_neighbours_coded(px, py)?;
        let left = [self.get(px - 1, py + last), self.get(px - 1, py)];
        let above = [self.get(px + last, py - 1), self.get(px, py - 1)];
        let matched = |side: &[Option<Entry>; 2]| u8::from(side.iter().any(Option::is_some));
        let new = left.iter().chain(&above).flatten().any(|e| e.is_new);
        let mode_context = matched(&left) + matched(&above) + 2 * u8::from(new);
        let mut stack = Stack::default();
        // Both vectors of a same-reference compound neighbour match the
        // reference (AVM `add_ref_mv_candidate`: `candidate->ref_frame[ref]
        // == rf[0]` for `ref` 0 and 1).
        for entry in self.scan(x, y, size).into_iter().flatten() {
            stack.add(entry.first.vector());
            if let Some(second) = entry.second {
                stack.add(second.vector());
            }
        }
        if self.config.ref_mv_bank {
            self.add_bank_candidates(&mut stack, (px, py), size as i32);
        }
        // AVM clamps the derived entries before inserting global motion.
        for vector in &mut stack.vectors[..stack.count] {
            *vector = self.clamp_reference((px, py), size, *vector);
        }
        stack.add((0, 0));
        if size >= MIN_EXTENSION_SIDE {
            let derived = stack.count;
            for (row, col) in EXTENSION_PAIRS {
                if row.max(col) < derived {
                    stack.add((stack.vectors[row].0, stack.vectors[col].1));
                }
            }
        }
        Ok(Candidates::new(
            stack.vectors,
            stack.count.min(self.config.entries()),
            self.config.max_drl_bits,
            mode_context,
        ))
    }

    /// `vector` clamped to reach at most [`MV_BORDER`] beyond the picture
    /// from the block of side `size` at `(px, py)` (AVM `clamp_mv_ref`).
    #[inline]
    fn clamp_reference(&self, (px, py): (i32, i32), size: usize, (row, col): Vector) -> Vector {
        let border = (size as i32) * EIGHTHS + MV_BORDER;
        (
            row.clamp(
                -py * EIGHTHS - border,
                (self.height as i32 - py) * EIGHTHS + MV_BORDER,
            ),
            col.clamp(
                -px * EIGHTHS - border,
                (self.width as i32 - px) * EIGHTHS + MV_BORDER,
            ),
        )
    }

    /// Whether a bank vector's reference block lies at least partly inside
    /// the picture (AVM `check_rmb_cand`; it divides with C truncation, as
    /// Rust's `/` does).
    fn reaches_picture(&self, (px, py): (i32, i32), size: i32, (row, col): Vector) -> bool {
        let (reference_x, reference_y) = (px + col / EIGHTHS, py + row / EIGHTHS);
        reference_x > -size
            && reference_y > -size
            && reference_x < self.width as i32
            && reference_y < self.height as i32
    }

    /// Fill open selectable entries from the bank, newest first, skipping
    /// vectors listed already (budget permitting) and vectors whose
    /// reference block would lie wholly outside the picture (AVM
    /// `add_ref_mv_bank_candidates`, `check_rmb_cand`).
    fn add_bank_candidates(&self, stack: &mut Stack, (px, py): (i32, i32), size: i32) {
        let limit = self.config.entries();
        for vector in self.bank.single.newest_first() {
            if stack.count >= limit {
                break;
            }
            if stack.is_new(vector) && self.reaches_picture((px, py), size, vector) {
                stack.push(vector);
            }
        }
    }

    /// Store the final absolute vector of the inter block of side `size` at
    /// `(x, y)` after it has been coded. `is_new`: coded as NEWMV (counted by
    /// later mode contexts).
    /// # Errors
    /// Rejects repeated blocks, invalid geometry, or non-quarter-pel vectors.
    pub fn commit(
        &mut self,
        x: usize,
        y: usize,
        size: usize,
        vector: Vector,
        is_new: bool,
    ) -> Result<()> {
        self.commit_motion(x, y, size, BlockMotion::Single { vector, is_new })
    }

    /// Store the final motion of the inter block of side `size` at `(x, y)`
    /// after it has been coded.
    /// # Errors
    /// Rejects repeated blocks, invalid geometry, or non-quarter-pel vectors.
    pub fn commit_motion(
        &mut self,
        x: usize,
        y: usize,
        size: usize,
        motion: BlockMotion,
    ) -> Result<()> {
        let (vectors, is_new) = match motion {
            BlockMotion::Single { vector, is_new } => ([vector, vector], is_new),
            BlockMotion::Compound((first, second)) => ([first, second], false),
        };
        if vectors
            .iter()
            .flat_map(|&(row, col)| [row, col])
            .any(|v| !(-MAX_COMPONENT..=MAX_COMPONENT).contains(&v) || v & 1 != 0)
        {
            return Err(DecodeError::Malformed("AV2 MV grid entry"));
        }
        let entry = Entry {
            first: Narrow::new(vectors[0]),
            second: matches!(motion, BlockMotion::Compound(_)).then(|| Narrow::new(vectors[1])),
            is_new,
        };
        self.fill(x, y, size, Some(entry))?;
        if self.config.ref_mv_bank {
            self.bank.update(entry);
        }
        Ok(())
    }

    /// Record a coded intra block, which provides no motion candidates.
    /// # Errors
    /// Rejects repeated blocks and invalid geometry.
    pub fn commit_intra(&mut self, x: usize, y: usize, size: usize) -> Result<()> {
        self.fill(x, y, size, None)
    }

    fn fill(&mut self, x: usize, y: usize, size: usize, motion: Option<Entry>) -> Result<()> {
        self.last.set(None);
        self.last_skip.set(None);
        self.check_block(x, y, size)?;
        let stride = self.width / CELL;
        let cells = (y / CELL..(y + size) / CELL)
            .flat_map(|row| (x / CELL..(x + size) / CELL).map(move |col| row * stride + col));
        if cells
            .clone()
            .any(|index| self.cells[index] != Cell::Uncoded)
        {
            return Err(DecodeError::Malformed("AV2 MV grid entry"));
        }
        let column = u16::try_from(x).expect("checked against the picture width");
        for index in cells {
            self.cells[index] = Cell::Coded { column, motion };
        }
        Ok(())
    }
}

const MODE: [[u16; 3]; 5] = [
    [22725, 21668, 0],
    [11207, 11010, 0],
    [7357, 7054, 0],
    [18651, 18427, 0],
    [14480, 14191, 0],
];
/// AVM `default_drl_cdf[type][context]` (inverse), for ranking and for
/// fixed-probability coding.
const DRL: [[u16; 5]; DRL_MODEL_TYPES] = [
    [17047, 11653, 13201, 15166, 19449],
    [14076, 13425, 14561, 14860, 14464],
    [10611, 9535, 9986, 10415, 10311],
];

/// The DRL symbols of `index`: (model type, symbol) for each coded
/// decision, ending with a 0 unless the index is the last entry
/// (AVM `read_drl_idx`).
fn drl_symbols(max_drl_bits: u8, index: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..usize::from(max_drl_bits).min(index + 1))
        .map(move |step| (step.min(DRL_MODEL_TYPES - 1), usize::from(index > step)))
}

/// Bits of the cheapest way to signal an existing vector under the
/// writer's current probabilities: `GLOBALMV`, or `NEARMV` with DRL index 0
/// (one DRL decision, as `max_drl_bits` is at least 1), at the cheapest
/// mode context. For the planner's per-leaf cost.
#[must_use]
pub fn cheapest_reuse_bits(writer: &Writer) -> f32 {
    const FIRST_DRL_TYPE: usize = 0;
    (0..MODE.len())
        .map(|context| {
            let mode = writer.current(&MODE[context], |m| &m.inter_mode[context]);
            let drl = writer.current(&[DRL[FIRST_DRL_TYPE][context], 0], |m| {
                &m.drl[FIRST_DRL_TYPE][context]
            });
            let global = symbol_bits(&mode, InterMode::Global.symbol());
            let near = symbol_bits(&mode, InterMode::Near.symbol()) + symbol_bits(&drl, 0);
            global.min(near)
        })
        .fold(f32::INFINITY, f32::min)
}

/// Code the single-reference inter mode with its neighbor-derived context.
/// # Errors
/// Rejects invalid contexts and propagates entropy errors.
pub fn write_inter_mode(writer: &mut Writer, context: u8, mode: InterMode) -> Result<()> {
    let cdf = MODE
        .get(usize::from(context))
        .ok_or(DecodeError::Malformed("AV2 MV mode context"))?;
    writer.model(mode.symbol(), cdf, |m| {
        &mut m.inter_mode[usize::from(context)]
    })
}

/// Decode the single-reference inter mode.
/// # Errors
/// Rejects invalid contexts and entropy errors.
pub fn read_inter_mode(reader: &mut Reader<'_>, context: u8) -> Result<InterMode> {
    let cdf = MODE
        .get(usize::from(context))
        .ok_or(DecodeError::Malformed("AV2 MV mode context"))?;
    let symbol = reader.model(cdf, |m| &mut m.inter_mode[usize::from(context)])?;
    Ok(InterMode::ALL[symbol])
}

/// Code GLOBALMV (false) or NEWMV (true) with neighbor-derived context.
/// # Errors
/// Rejects invalid contexts and propagates entropy errors.
pub fn write_mode(writer: &mut Writer, context: u8, is_new: bool) -> Result<()> {
    write_inter_mode(
        writer,
        context,
        if is_new {
            InterMode::New
        } else {
            InterMode::Global
        },
    )
}

/// Decode GLOBALMV or NEWMV, rejecting NEARMV (not emitted by the picture encoder).
/// # Errors
/// Rejects invalid contexts, unsupported modes and entropy errors.
pub fn read_mode(reader: &mut Reader<'_>, context: u8) -> Result<bool> {
    match read_inter_mode(reader, context)? {
        InterMode::Global => Ok(false),
        InterMode::New => Ok(true),
        InterMode::Near => Err(DecodeError::Unsupported("AV2 near motion mode")),
    }
}

/// Approximate bits of the inter mode and DRL syntax, for mode ranking.
#[must_use]
pub fn mode_bits(candidates: Candidates, mode: InterMode, index: usize) -> f32 {
    let drl = if mode.has_drl() {
        candidates.drl_index_bits[index]
    } else {
        0.0
    };
    candidates.mode_symbol_bits[mode.symbol()] + drl
}

/// Write DRL index `index` (`max_drl_bits` decisions at most).
/// # Errors
/// Rejects an underived index or invalid context and propagates entropy
/// errors.
pub fn write_drl(writer: &mut Writer, candidates: Candidates, index: usize) -> Result<()> {
    let context = usize::from(candidates.mode_context);
    if index >= usize::from(candidates.count) || context >= MODE.len() {
        return Err(DecodeError::Malformed("AV2 DRL index"));
    }
    for (kind, symbol) in drl_symbols(candidates.max_drl_bits, index) {
        writer.model(symbol, &[DRL[kind][context], 0], |m| {
            &mut m.drl[kind][context]
        })?;
    }
    Ok(())
}

/// Read a DRL index (AVM `read_drl_idx`). Indices past the derived entries
/// are legal and name zero vectors.
/// # Errors
/// Rejects an invalid context and entropy errors.
pub fn read_drl(reader: &mut Reader<'_>, candidates: Candidates) -> Result<usize> {
    let context = usize::from(candidates.mode_context);
    if context >= MODE.len() {
        return Err(DecodeError::Malformed("AV2 DRL context"));
    }
    let mut index = 0;
    while index < usize::from(candidates.max_drl_bits) {
        let kind = index.min(DRL_MODEL_TYPES - 1);
        if reader.model(&[DRL[kind][context], 0], |m| &mut m.drl[kind][context])? == 0 {
            break;
        }
        index += 1;
    }
    Ok(index)
}
#[cfg(test)]
mod tests {
    use super::*;
    const REFERENCE_HASHES: [u64; 20] = [
        18_108_579_133_170_363_140,
        18_108_579_133_170_363_140,
        18_108_579_133_170_363_140,
        18_108_579_133_170_363_140,
        1_347_904_059_069_687_898,
        15_521_649_158_962_192_692,
        6_452_403_573_578_437_995,
        16_277_340_703_219_229_390,
        6_600_393_617_970_150_283,
        6_539_078_507_857_285_089,
        730_523_105_399_700_258,
        17_786_036_967_685_468_393,
        13_695_811_693_275_090_668,
        17_861_732_359_718_822_651,
        5_355_437_261_173_124_684,
        656_526_331_992_512_881,
        6_489_671_396_705_318_174,
        7_725_099_750_821_090_629,
        942_966_438_731_786_527,
        4_975_158_960_919_833_534,
    ];

    /// AVM traces of the six-entry list with the reference MV bank (the
    /// whole list), from `tools/codegen/check_av2_mv_refs.py`.
    const FULL_LIST_HASHES: [u64; 20] = [
        10_124_604_356_780_897_156,
        10_124_604_356_780_897_156,
        10_124_604_356_780_897_156,
        10_124_604_356_780_897_156,
        14_872_152_748_070_535_610,
        6_348_782_961_930_729_898,
        17_457_308_889_268_577_101,
        8_069_586_520_036_873_239,
        14_571_438_519_859_541_643,
        12_682_055_044_823_212_259,
        2_038_814_211_512_361_940,
        4_576_082_905_058_028_793,
        10_871_935_114_642_004_332,
        9_911_531_742_396_180_873,
        207_988_313_197_210_709,
        12_461_661_955_524_228_393,
        2_583_631_729_601_320_958,
        786_343_296_595_360_182,
        5_409_526_391_379_836_242,
        12_329_721_540_583_931_873,
    ];

    /// The vector a trace commits for its `step`-th block under `pattern`
    /// (as the oracle's `leaf`): `(vector, is_new)`.
    fn trace_vector(pattern: i32, step: i32) -> ((i32, i32), bool) {
        match pattern {
            1 if step % 3 != 0 => ((((step / 3) % 3 - 1) * 16, ((step / 5) % 3 - 1) * 16), true),
            2 => (
                (((step * 37) % 129 - 64) * 16, ((step * 53) % 129 - 64) * 16),
                true,
            ),
            3 if step % 2 != 0 => ((((step * 17) % 5 - 2) * 2, ((step * 11) % 5 - 2) * 2), true),
            _ => ((0, 0), false),
        }
    }

    /// Hashes of the candidate lists of every 8×8 block (coding order) over
    /// the oracle's pictures and vector patterns, `record` choosing the
    /// values each list contributes.
    fn trace_hashes(config: ListConfig, record: impl Fn(&Candidates) -> Vec<i32>) -> Vec<u64> {
        let mut hashes = Vec::new();
        for (width, height) in [(8, 8), (24, 16), (64, 64), (72, 40), (128, 136)] {
            for pattern in 0..4 {
                let mut grid = ReferenceGrid::new(width, height, config).unwrap();
                let mut hash = 14_695_981_039_346_656_037_u64;
                let mut step = 0_i32;
                crate::partition::write_8x8(&mut Writer::new(), width, height, |_, x, y| {
                    if x.is_multiple_of(SUPERBLOCK) && y.is_multiple_of(SUPERBLOCK) {
                        grid.begin_superblock(x, y)?;
                    }
                    let c = grid.candidates(x, y, 8)?;
                    for value in record(&c) {
                        hash = (hash ^ u64::from(value as u32)).wrapping_mul(1_099_511_628_211);
                    }
                    let (vector, is_new) = trace_vector(pattern, step);
                    grid.commit(x, y, 8, vector, is_new)?;
                    step += 1;
                    Ok(())
                })
                .unwrap();
                hashes.push(hash);
            }
        }
        hashes
    }

    #[test]
    fn matches_original_avm_mv_reference_and_context_traces() {
        let basic = trace_hashes(ListConfig::BASIC, |c| {
            vec![
                i32::from(c.count),
                c.vectors[0].0,
                c.vectors[0].1,
                c.vectors[1].0,
                c.vectors[1].1,
                i32::from(c.mode_context),
            ]
        });
        assert_eq!(basic, REFERENCE_HASHES);
        let full_config = ListConfig {
            max_drl_bits: 5,
            ref_mv_bank: true,
        };
        let full = trace_hashes(full_config, |c| {
            let vectors = c.vectors.iter().flat_map(|&(row, col)| [row, col]);
            std::iter::once(i32::from(c.count))
                .chain(vectors)
                .chain([i32::from(c.mode_context)])
                .collect()
        });
        assert_eq!(full, FULL_LIST_HASHES);
    }

    #[test]
    fn cheapest_reuse_is_the_cheapest_mode_context() {
        let writer = Writer::new();
        let cheapest = cheapest_reuse_bits(&writer);
        for context in 0..MODE.len() {
            let c = Candidates::new([(0, 0); MAX_STACK], 1, 1, context as u8);
            let near = mode_bits(c, InterMode::Near, 0);
            let global = mode_bits(c, InterMode::Global, 0);
            assert!(cheapest <= near.min(global) + 1e-4, "context {context}");
        }
        assert!(cheapest > 0.0);
    }

    #[test]
    fn mode_drl_entropy_and_invalid_state() {
        let mut writer = Writer::new();
        for mode_context in 0..5 {
            for is_new in [false, true] {
                let c = Candidates::new(
                    [(0, 0), (16, -16), (0, 0), (0, 0), (0, 0), (0, 0)],
                    2,
                    1,
                    mode_context,
                );
                write_mode(&mut writer, mode_context, is_new).unwrap();
                for index in 0..2 {
                    write_drl(&mut writer, c, index).unwrap();
                }
            }
        }
        let packet = writer.finish();
        let mut reader = Reader::new(&packet).unwrap();
        for mode_context in 0..5 {
            for is_new in [false, true] {
                let c = Candidates::new(
                    [(0, 0), (16, -16), (0, 0), (0, 0), (0, 0), (0, 0)],
                    2,
                    1,
                    mode_context,
                );
                assert_eq!(read_mode(&mut reader, mode_context).unwrap(), is_new);
                for index in 0..2 {
                    assert_eq!(read_drl(&mut reader, c).unwrap(), index);
                }
            }
        }
        let mut grid = ReferenceGrid::new(16, 16, ListConfig::BASIC).unwrap();
        assert!(grid.candidates(8, 8, 8).is_err());
        assert!(grid.candidates(0, 0, 32).is_err());
        assert!(grid.commit(0, 0, 8, (1, 0), false).is_err());
        grid.commit(0, 0, 8, (0, 0), false).unwrap();
        assert!(grid.commit(0, 0, 8, (0, 0), false).is_err());
        assert!(grid.commit(0, 0, 16, (0, 0), false).is_err());
    }
}
