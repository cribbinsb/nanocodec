//! Codec-independent choice of a frame's CDEF strength sets: given each
//! filter block's distortion under every candidate strength, pick the few
//! candidates the frame signals and the one each block uses, trading
//! distortion against the bits of the per-block index.
//!
//! The greedy forward selection with a refinement pass follows libaom's
//! `joint_strength_search` idea (BSD-2-Clause, Alliance for Open Media),
//! reimplemented: the backend measures distortions with its own filter and
//! supplies the rate of each set count.

/// The selected strength sets and each block's choice among them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrengthSelection {
    /// Candidate index of each signalled set, in signalled order.
    pub sets: Vec<usize>,
    /// Per block, the position in `sets` of the set it uses.
    pub choices: Vec<usize>,
}

/// Choose at most `max_sets` (at least one) of `candidates` strengths.
///
/// `errors` holds, block by block, each candidate's distortion (block-major,
/// `candidates` per block). `rate_bits(count)` is the frame's signalling
/// cost in bits with `count` sets (header and every block's index), and
/// `lambda` converts bits to distortion. Ties keep the earlier candidate, so
/// listing "off" first prefers it.
///
/// # Panics
///
/// Panics if `candidates` is zero or `errors` is not a whole number of
/// blocks.
#[must_use]
pub fn select_strength_sets(
    errors: &[u64],
    candidates: usize,
    max_sets: usize,
    lambda: f64,
    rate_bits: impl Fn(usize) -> f64,
) -> StrengthSelection {
    assert!(candidates > 0 && errors.len().is_multiple_of(candidates));
    let max_sets = max_sets.clamp(1, candidates);
    let mut sets: Vec<usize> = Vec::with_capacity(max_sets);
    let mut best: Option<(f64, Vec<usize>)> = None;
    while sets.len() < max_sets {
        let Some(added) = best_addition(errors, candidates, &sets) else {
            break;
        };
        sets.push(added);
        refine(errors, candidates, &mut sets);
        let cost = total_error(errors, candidates, &sets) as f64 + lambda * rate_bits(sets.len());
        if best.as_ref().is_none_or(|(best_cost, _)| cost < *best_cost) {
            best = Some((cost, sets.clone()));
        }
    }
    let sets = best.map_or_else(|| vec![0], |(_, sets)| sets);
    let choices = errors
        .chunks_exact(candidates)
        .map(|block| best_position(block, &sets))
        .collect();
    StrengthSelection { sets, choices }
}

/// The position in `sets` of the set with the least error in `block`
/// (the earliest on ties).
fn best_position(block: &[u64], sets: &[usize]) -> usize {
    let mut position = 0;
    for (index, &set) in sets.iter().enumerate() {
        if block[set] < block[sets[position]] {
            position = index;
        }
    }
    position
}

/// Total error when every block takes its best set of `sets`.
fn total_error(errors: &[u64], candidates: usize, sets: &[usize]) -> u64 {
    errors
        .chunks_exact(candidates)
        .map(|block| sets.iter().map(|&set| block[set]).min().unwrap_or(0))
        .sum()
}

/// The candidate whose addition to `sets` lowers the total error most, if
/// any lowers it (any candidate when `sets` is empty).
fn best_addition(errors: &[u64], candidates: usize, sets: &[usize]) -> Option<usize> {
    let current = (!sets.is_empty()).then(|| total_error(errors, candidates, sets));
    let mut trial = sets.to_vec();
    trial.push(0);
    let mut best: Option<(u64, usize)> = None;
    for candidate in (0..candidates).filter(|c| !sets.contains(c)) {
        *trial.last_mut().expect("trial has the added slot") = candidate;
        let error = total_error(errors, candidates, &trial);
        if best.is_none_or(|(best_error, _)| error < best_error) {
            best = Some((error, candidate));
        }
    }
    best.filter(|&(error, _)| current.is_none_or(|current| error < current))
        .map(|(_, candidate)| candidate)
}

/// One pass replacing each set in turn by the candidate that lowers the
/// total error most with the others fixed (greedy choices made early can
/// be improved once later sets cover some blocks).
fn refine(errors: &[u64], candidates: usize, sets: &mut [usize]) {
    for position in 0..sets.len() {
        let mut best = (total_error(errors, candidates, sets), sets[position]);
        for candidate in 0..candidates {
            if sets.contains(&candidate) {
                continue;
            }
            let kept = sets[position];
            sets[position] = candidate;
            let error = total_error(errors, candidates, sets);
            sets[position] = kept;
            if error < best.0 {
                best = (error, candidate);
            }
        }
        sets[position] = best.1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_signalling_gives_each_block_its_best_candidate() {
        // Three blocks, three candidates; each block prefers a different one.
        let errors = [1, 9, 9, 9, 1, 9, 9, 9, 1];
        let selection = select_strength_sets(&errors, 3, 8, 0.0, |_| 0.0);
        assert_eq!(selection.sets.len(), 3);
        for (block, &choice) in selection.choices.iter().enumerate() {
            assert_eq!(selection.sets[choice], block);
        }
    }

    #[test]
    fn expensive_indices_keep_one_set() {
        let errors = [1, 9, 9, 9, 1, 9, 9, 9, 1];
        let selection = select_strength_sets(&errors, 3, 8, 1.0, |count| 100.0 * count as f64);
        assert_eq!(selection.sets.len(), 1);
        assert_eq!(selection.choices, vec![0; 3]);
    }

    #[test]
    fn ties_prefer_the_first_candidate() {
        let errors = [5, 5, 5, 5];
        let selection = select_strength_sets(&errors, 2, 2, 1.0, |count| count as f64);
        assert_eq!(selection.sets, vec![0]);
        assert_eq!(selection.choices, vec![0, 0]);
    }

    #[test]
    fn refinement_replaces_an_early_greedy_choice() {
        // Candidate 2 is the best single set, but with two sets {0, 1} is
        // exact: refinement must drop 2.
        let errors = [0, 10, 4, 10, 0, 4];
        let selection = select_strength_sets(&errors, 3, 2, 0.0, |_| 0.0);
        let mut sets = selection.sets.clone();
        sets.sort_unstable();
        assert_eq!(sets, vec![0, 1]);
    }
}
