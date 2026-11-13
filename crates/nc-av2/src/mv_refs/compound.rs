//! The same-reference compound (`{LAST, LAST}`) reference list that skip
//! mode selects from (AVM v1.0.0 `av2_find_mv_refs` with `skip_mode`, the
//! compound paths of `setup_ref_mv_list`, `add_ref_mv_candidate`,
//! `fill_mvp_from_derived_smvp`, `add_ref_mv_bank_candidates`,
//! `check_rmb_cand` and `insert_mvp_candidate`; `av2/common/mvref_common.c`).
//!
//! The list is built from the same spatial scan as the single-reference
//! list ([`ReferenceGrid::scan`]), in this order:
//!
//! 1. compound neighbours add their vector pair;
//! 2. single-reference neighbours instead derive pairs: each one after the
//!    first is paired with the first single vector seen (AVM `single_mv`
//!    and `derived_mv_stack`, with their own small capacity and comparison
//!    budgets); the derived pairs fill open selectable entries;
//! 3. the compound bank (recent pairs of the superblock row), when enabled;
//! 4. clamping of both vectors, then the global pair (zero, zero), then for
//!    blocks larger than 32×32 the row/column combinations of the first
//!    three entries, taken for both vectors alike.
//!
//! Temporal, trajectory, TIP and warp candidates are disabled by the
//! sequence, and every block uses the one reference, so a compound
//! neighbour always matches the pair. See `THIRD_PARTY.md` for reference
//! terms.
use super::{
    BlockKey, EXTENSION_PAIRS, Entry, MAX_PRUNE_COMPARISONS, MAX_STACK, MIN_EXTENSION_SIDE, Pair,
    ReferenceGrid, Vector,
};
use crate::entropy::{Reader, Writer};
use nc_common::decode::Result;
use nc_encoder::symbol_cost::symbol_bits;

/// AVM `MAX_DR_STACK_SIZE`: derived pairs and single vectors kept.
const DERIVED_CAPACITY: usize = 4;
/// AVM `MAX_DR_PR_NUM`: duplicate comparisons the derived-pair and
/// single-vector lists may each spend; later entries are added unpruned.
const DERIVED_PRUNE_COMPARISONS: u32 = 2;

/// AVM `default_skip_drl_cdf[idx]` (inverse): the skip-mode DRL
/// decisions, by position (the third and later share the last model).
const SKIP_DRL: [u16; 3] = [11134, 15392, 14336];
/// AVM `default_skip_mode_cdfs[ctx]` (inverse), by the number of skip-mode
/// neighbours.
pub const SKIP_MODE: [[u16; 2]; 3] = [[1804, 0], [10999, 0], [20284, 0]];

/// The skip-mode DRL decisions of `index`: (model, symbol) for each coded
/// decision, ending with a 0 unless the index is the last entry (AVM
/// `read_drl_idx` with `skip_drl_cdf`; dav2d `skip_mode_drl_idx`).
fn skip_drl_symbols(max_drl_bits: u8, index: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..usize::from(max_drl_bits).min(index + 1))
        .map(move |step| (step.min(SKIP_DRL.len() - 1), usize::from(index > step)))
}

/// Write skip-mode DRL index `index`.
/// # Errors
/// Rejects an index beyond the DRL size and propagates entropy errors.
pub fn write_skip_drl(writer: &mut Writer, max_drl_bits: u8, index: usize) -> Result<()> {
    if index > usize::from(max_drl_bits) {
        return Err(nc_common::decode::DecodeError::Malformed(
            "AV2 skip-mode DRL index",
        ));
    }
    for (kind, symbol) in skip_drl_symbols(max_drl_bits, index) {
        writer.model(symbol, &[SKIP_DRL[kind], 0], |m| &mut m.skip_drl[kind])?;
    }
    Ok(())
}

/// Read a skip-mode DRL index.
/// # Errors
/// Propagates entropy errors.
pub fn read_skip_drl(reader: &mut Reader<'_>, max_drl_bits: u8) -> Result<usize> {
    let mut index = 0;
    while index < usize::from(max_drl_bits) {
        let kind = index.min(SKIP_DRL.len() - 1);
        if reader.model(&[SKIP_DRL[kind], 0], |m| &mut m.skip_drl[kind])? == 0 {
            break;
        }
        index += 1;
    }
    Ok(index)
}

/// Approximate bits of skip-mode DRL index `index`, for ranking.
#[must_use]
pub fn skip_drl_bits(max_drl_bits: u8, index: usize) -> f32 {
    skip_drl_symbols(max_drl_bits, index)
        .map(|(kind, symbol)| symbol_bits(&[SKIP_DRL[kind], 0], symbol))
        .sum()
}

/// A block's skip-mode references.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkipCandidates {
    /// Eighth-pel `(this_mv, comp_mv)` pairs. Entries from `count` on are
    /// zero: AVM initialises the list to zero vectors, and a DRL index may
    /// name any of the `max_drl_bits + 1` entries.
    pub pairs: [Pair; MAX_STACK],
    /// Derived selectable entries, 1..=`max_drl_bits + 1`.
    pub count: u8,
    /// The sequence's DRL size.
    pub max_drl_bits: u8,
}

impl SkipCandidates {
    /// The pair DRL index `index` names (zero vectors past the derived
    /// entries).
    #[must_use]
    pub fn pair(&self, index: usize) -> Pair {
        self.pairs[index.min(MAX_STACK - 1)]
    }
}

/// A list of `N` entries under construction whose duplicate check spends
/// at most `budget` comparisons (AVM `ref_mv_stack` with `drl_pr_count`,
/// `derived_mv_stack` with `drl_dr_pr_count`, `single_mv` with
/// `drl_dr_single_pr_count`).
#[derive(Debug)]
struct Pruned<T, const N: usize> {
    entries: [T; N],
    count: usize,
    comparisons: u32,
    budget: u32,
}

impl<T: Copy + PartialEq + Default, const N: usize> Pruned<T, N> {
    fn new(budget: u32) -> Self {
        Self {
            entries: [T::default(); N],
            count: 0,
            comparisons: 0,
            budget,
        }
    }

    /// Whether `entry` is not yet listed; while the budget lasts every
    /// comparison counts against it, after that nothing is pruned.
    fn is_new(&mut self, entry: T) -> bool {
        if self.comparisons >= self.budget {
            return true;
        }
        for &listed in &self.entries[..self.count] {
            self.comparisons += 1;
            if listed == entry {
                return false;
            }
        }
        true
    }

    /// Append `entry` unless it is listed (budget permitting) or the list
    /// holds `limit` entries already. AVM compares before it checks for
    /// room.
    fn add(&mut self, entry: T, limit: usize) {
        if self.is_new(entry) && self.count < limit.min(N) {
            self.push(entry);
        }
    }

    fn push(&mut self, entry: T) {
        self.entries[self.count] = entry;
        self.count += 1;
    }

    fn listed(&self) -> &[T] {
        &self.entries[..self.count]
    }
}

impl ReferenceGrid {
    /// Derive the skip-mode references before coding the block of side
    /// `size` at `(x, y)`.
    /// # Errors
    /// Rejects invalid geometry or missing mandatory above/left neighbours.
    pub fn skip_candidates(&self, x: usize, y: usize, size: usize) -> Result<SkipCandidates> {
        let key: BlockKey = (x, y, size);
        if let Some((cached, candidates)) = self.last_skip.get()
            && cached == key
        {
            return Ok(candidates);
        }
        let candidates = self.derive_skip_candidates(x, y, size)?;
        self.last_skip.set(Some((key, candidates)));
        Ok(candidates)
    }

    fn derive_skip_candidates(&self, x: usize, y: usize, size: usize) -> Result<SkipCandidates> {
        self.check_block(x, y, size)?;
        let (px, py) = (x as i32, y as i32);
        self.check_neighbours_coded(px, py)?;
        let limit = self.config.entries();
        let mut stack = Pruned::<Pair, MAX_STACK>::new(MAX_PRUNE_COMPARISONS);
        let mut derived = Pruned::<Pair, DERIVED_CAPACITY>::new(DERIVED_PRUNE_COMPARISONS);
        let mut singles = Pruned::<Vector, DERIVED_CAPACITY>::new(DERIVED_PRUNE_COMPARISONS);
        for entry in self.scan(x, y, size).into_iter().flatten() {
            // AVM `scan_blk_mbmi` stops looking once the list is full.
            if stack.count >= MAX_STACK {
                continue;
            }
            scan_entry(entry, &mut stack, &mut derived, &mut singles);
        }
        if stack.count < limit {
            for &pair in derived.listed() {
                stack.add(pair, limit);
            }
        }
        if self.config.ref_mv_bank {
            for pair in self.bank.compound.newest_first() {
                if stack.count >= limit {
                    break;
                }
                if stack.is_new(pair)
                    && self.reaches_picture((px, py), size as i32, pair.0)
                    && self.reaches_picture((px, py), size as i32, pair.1)
                {
                    stack.push(pair);
                }
            }
        }
        for pair in &mut stack.entries[..stack.count] {
            *pair = (
                self.clamp_reference((px, py), size, pair.0),
                self.clamp_reference((px, py), size, pair.1),
            );
        }
        stack.add(((0, 0), (0, 0)), MAX_STACK);
        if size >= MIN_EXTENSION_SIDE {
            let original = stack.count;
            for (row, col) in EXTENSION_PAIRS {
                if row.max(col) < original {
                    let (a, b) = (stack.entries[row], stack.entries[col]);
                    stack.add(((a.0.0, b.0.1), (a.1.0, b.1.1)), MAX_STACK);
                }
            }
        }
        Ok(SkipCandidates {
            pairs: stack.entries,
            count: stack.count.min(limit) as u8,
            max_drl_bits: self.config.max_drl_bits,
        })
    }
}

/// One spatial neighbour's contribution to the compound list (AVM
/// `add_ref_mv_candidate` for `rf = {LAST, LAST}`): a compound neighbour's
/// pair, or a single neighbour's vector paired with the first single vector
/// seen before it.
fn scan_entry(
    entry: Entry,
    stack: &mut Pruned<Pair, MAX_STACK>,
    derived: &mut Pruned<Pair, DERIVED_CAPACITY>,
    singles: &mut Pruned<Vector, DERIVED_CAPACITY>,
) {
    if let Some(pair) = entry.pair() {
        stack.add(pair, MAX_STACK);
        return;
    }
    let vector = entry.vector();
    if let Some(&first) = singles.listed().first() {
        derived.add((vector, first), DERIVED_CAPACITY);
    }
    singles.add(vector, DERIVED_CAPACITY);
}
