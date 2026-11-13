//! VP9 tree-coded symbols (`vpx_tree_index` trees, VP9 §9.3) and their bit
//! costs, plus the trees this subset codes. Trees follow libvpx
//! `vp9_entropymode.c` (BSD-3-Clause; see `THIRD_PARTY.md`).
//!
//! A tree is a flat array of pairs: entry `i` is the branch taken on bit
//! `i & 1` from node `i >> 1`; a positive entry is the index of the next
//! pair, zero or negative is the leaf `-entry`. Node `n` reads probability
//! `probs[n]`.

#[cfg(test)]
use crate::bool_coder::BoolWriter;
use crate::bool_coder::{BitSink, BoolReader};

/// A coding tree: its entries, and each symbol's path from the root,
/// worked out when the constant is built (writing and costing a symbol
/// then reads its path instead of searching the tree for it).
#[derive(Debug)]
pub(crate) struct Tree {
    entries: &'static [i8],
    paths: [Path; MAX_SYMBOLS],
}

/// The `(node, bit)` decisions coding one symbol, root first.
#[derive(Debug, Clone, Copy)]
struct Path {
    steps: [(u8, bool); MAX_DEPTH],
    len: u8,
}

impl Path {
    const NONE: Self = Self {
        steps: [(0, false); MAX_DEPTH],
        len: 0,
    };
}

impl Tree {
    /// The tree of `entries` (pairs of branches; see the module comment).
    pub(crate) const fn new(entries: &'static [i8]) -> Self {
        let mut paths = [Path::NONE; MAX_SYMBOLS];
        let mut leaf = 0;
        while leaf < entries.len() {
            if entries[leaf] <= 0 {
                paths[(-entries[leaf]) as usize] = path_to(entries, leaf);
            }
            leaf += 1;
        }
        Self { entries, paths }
    }

    /// The flat entries, for walks from the root (reading, merging counts).
    pub(crate) fn entries(&self) -> &[i8] {
        self.entries
    }

    /// The `(node, bit)` decisions of `value`, root first.
    pub(crate) fn path(&self, value: usize) -> &[(u8, bool)] {
        let path = &self.paths[value];
        assert!(path.len > 0, "symbol {value} not in tree");
        &path.steps[..usize::from(path.len)]
    }
}

/// The path from the root to entry `leaf`: each step up finds the entry
/// pointing at the current pair.
const fn path_to(entries: &[i8], leaf: usize) -> Path {
    let mut up = [(0, false); MAX_DEPTH];
    let (mut len, mut entry) = (0, leaf);
    loop {
        up[len] = ((entry >> 1) as u8, entry & 1 == 1);
        len += 1;
        let pair = entry & !1;
        if pair == 0 {
            break;
        }
        entry = 0;
        while !(entries[entry] > 0 && entries[entry] as usize == pair) {
            entry += 1;
        }
    }
    let mut path = Path::NONE;
    while (path.len as usize) < len {
        path.steps[path.len as usize] = up[len - 1 - path.len as usize];
        path.len += 1;
    }
    path
}

/// `vp9_intra_mode_tree`: DC, TM, V, H, D135, D117, D45, D63, D153, D207.
pub(crate) const INTRA_MODE: &Tree = &Tree::new(&[
    0, 2, -9, 4, -1, 6, 8, 12, -2, 10, -4, -5, -3, 14, -8, 16, -6, -7,
]);
/// `vp9_partition_tree`: NONE, HORZ, VERT, SPLIT.
pub(crate) const PARTITION: &Tree = &Tree::new(&[0, 2, -1, 4, -2, -3]);
/// `vp9_inter_mode_tree` over `INTER_OFFSET`: ZERO, NEAREST, NEAR, NEW.
pub(crate) const INTER_MODE: &Tree = &Tree::new(&[-2, 2, 0, 4, -1, -3]);
/// `vp9_switchable_interp_tree`: `EIGHTTAP` (0), `EIGHTTAP_SMOOTH` (1),
/// `EIGHTTAP_SHARP` (2).
pub(crate) const SWITCHABLE_INTERP: &Tree = &Tree::new(&[0, 2, -1, -2]);

/// Partition symbols.
pub(crate) const PARTITION_NONE: usize = 0;
pub(crate) const PARTITION_SPLIT: usize = 3;

/// Deepest path in any VP9 tree (the 11-class vector tree).
const MAX_DEPTH: usize = 10;
/// Most symbols in any VP9 tree (the vector classes).
const MAX_SYMBOLS: usize = 11;

/// Write `value` with node probabilities `probs`.
pub(crate) fn write(writer: &mut impl BitSink, tree: &Tree, probs: &[u8], value: usize) {
    for &(node, bit) in tree.path(value) {
        writer.write(bit, probs[usize::from(node)]);
    }
}

/// Read a symbol. Inlined into each call site, which knows its tree: the
/// walk then runs over constant entries (x86 -0.2 to -0.3 % of a decode).
#[allow(
    clippy::inline_always,
    reason = "a call per symbol; the plain hint was not taken"
)]
#[inline(always)]
pub(crate) fn read(reader: &mut BoolReader<'_>, tree: &Tree, probs: &[u8]) -> usize {
    let mut index = 0;
    loop {
        // The trees are the specification's constants: every entry is a
        // leaf or a node inside the tree.
        let entry = tree.entries[index + usize::from(reader.read(probs[index >> 1]))];
        if entry <= 0 {
            return (-entry) as usize;
        }
        index = entry as usize;
    }
}

/// Estimated bits of coding `bit` with probability-of-zero `prob` (/256).
pub(crate) fn bit_cost(bit: bool, prob: u8) -> f32 {
    let p = if bit {
        256 - usize::from(prob)
    } else {
        usize::from(prob)
    };
    crate::tables::BIT_COST[p]
}

/// Estimated bits of `value` in `tree`.
pub(crate) fn cost(tree: &Tree, probs: &[u8], value: usize) -> f32 {
    // Summed root first, as the bits are coded (float addition order).
    tree.path(value)
        .iter()
        .map(|&(node, bit)| bit_cost(bit, probs[usize::from(node)]))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_symbol_round_trips_and_costs_sum_to_one() {
        for (tree, symbols) in [(INTRA_MODE, 10), (PARTITION, 4), (INTER_MODE, 4)] {
            let probs: Vec<u8> = (0..symbols).map(|i| (37 + 50 * i) as u8).collect();
            let mut writer = BoolWriter::new();
            for value in 0..symbols {
                write(&mut writer, tree, &probs, value);
            }
            let packet = writer.finish();
            let mut reader = BoolReader::new(&packet).unwrap();
            for value in 0..symbols {
                assert_eq!(read(&mut reader, tree, &probs), value);
            }
            let total: f32 = (0..symbols)
                .map(|v| 2f32.powf(-cost(tree, &probs, v)))
                .sum();
            assert!((total - 1.0).abs() < 1e-4, "{total}");
        }
    }
}
