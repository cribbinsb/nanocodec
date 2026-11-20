//! Counted walks over units: `start, start + step, ..` below `end`, for the
//! block layer's transform and chunk walks.
//!
//! `Iterator::step_by` over a range sets up with a division (the step count)
//! and keeps that count live through the loop; this iterator is a compare
//! and an add per step. In the residual walks it saved 0.08M instructions a
//! frame on `aom720_good4_1000k` (callgrind, 2026-11-19).
use std::ops::Range;

/// The walk returned by [`strided`].
#[derive(Debug, Clone)]
pub(crate) struct Strided {
    at: usize,
    end: usize,
    step: usize,
}

impl Iterator for Strided {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.at >= self.end {
            return None;
        }
        let at = self.at;
        self.at += self.step;
        Some(at)
    }
}

/// `range.start`, then every `step` (nonzero) after it, below `range.end`.
#[inline]
pub(crate) fn strided(range: Range<usize>, step: usize) -> Strided {
    debug_assert!(step > 0, "a strided walk needs a step");
    Strided {
        at: range.start,
        end: range.end,
        step,
    }
}

#[cfg(test)]
mod tests {
    use super::strided;

    #[test]
    fn matches_step_by() {
        for start in 0..6 {
            for end in 0..20 {
                for step in 1..9 {
                    let ours: Vec<_> = strided(start..end, step).collect();
                    let std: Vec<_> = (start..end).step_by(step).collect();
                    assert_eq!(ours, std, "{start}..{end} by {step}");
                }
            }
        }
    }
}
