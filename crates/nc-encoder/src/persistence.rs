//! Temporal persistence of blocks, from past frames only: the low-delay
//! signal behind the persistence term of block AQ (`aq.rs`).
//!
//! A block whose content the next frames will predict well (still
//! background, static vegetation) is worth coding finely once: later frames
//! copy it for free. A block nothing predicts (water, smoke, wind-blown
//! foliage) is coded again every frame, and finer steps there mostly buy
//! noise. x264's mb-tree and libaom's TPL measure this with a lookahead;
//! with forward prediction and no lookahead this tracker measures it
//! backwards, on the assumption that the recent past predicts the future:
//!
//! every 8x8 cell keeps an exponential moving average of its motion-search
//! error per sample (`MotionAnalysis::fine`), the predictability of its
//! content (a moving but rigid object predicts well, water does not). Still
//! blocks are the reuse term's (`reuse.rs`), which measures their reuse.
//!
//! The moving average also damps flicker: a cell's offset changes by at
//! most `1 - persist.memory` of the change in its log error per frame.

use nc_analysis::MotionAnalysis;

use crate::params::PersistParams;

/// Samples per 8x8 cell, the divisor of the motion-search SAD.
const CELL_SAMPLES: f64 = 64.;
/// Error floor per sample (sensor noise level): below it all cells are
/// equally predictable, so clean static regions do not dominate the mean.
const MIN_ERROR: f64 = 1.;

/// Per-cell persistence history of one stream.
#[derive(Debug)]
pub(crate) struct PersistenceTracker {
    strength: f64,
    memory: f64,
    cols: usize,
    /// Moving average of the motion-search error per sample, per cell;
    /// empty until the first inter frame.
    error: Vec<f64>,
}

impl PersistenceTracker {
    /// `None` when the term is off.
    pub fn new(params: &PersistParams) -> Option<Self> {
        (params.strength > 0.).then(|| Self {
            strength: f64::from(params.strength),
            memory: f64::from(params.memory),
            cols: 0,
            error: Vec::new(),
        })
    }

    /// Fold in one frame's analysis. Frames without motion (key frames)
    /// keep the history, so a key frame refines the regions the previous
    /// frames found persistent.
    pub fn observe(&mut self, motion: Option<&MotionAnalysis>) {
        if let Some(motion) = motion {
            let fine = &motion.fine;
            let fresh = self.cols != fine.cols() || self.error.len() != fine.cells().len();
            self.cols = fine.cols();
            let sample_error = |sad: u32| (f64::from(sad) / CELL_SAMPLES).max(MIN_ERROR);
            if fresh {
                self.error.clear();
                self.error
                    .extend(fine.cells().iter().map(|b| sample_error(b.sad)));
            } else {
                for (average, block) in self.error.iter_mut().zip(fine.cells()) {
                    *average =
                        self.memory * *average + (1. - self.memory) * sample_error(block.sad);
                }
            }
        }
    }

    /// Add every cell's persistence offset, in QP units (negative is
    /// finer), to `offsets`, a `cols`-wide grid of the same cells. Cells
    /// without history get none.
    pub fn add_offsets(&self, cols: usize, offsets: &mut [f64]) {
        if cols != self.cols || offsets.len() != self.error.len() {
            return;
        }
        let count = self.error.len().max(1) as f64;
        let mean = self.error.iter().map(|e| e.log2()).sum::<f64>() / count;
        for (offset, error) in offsets.iter_mut().zip(&self.error) {
            *offset += self.strength * (error.log2() - mean);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ParamOverrides;

    fn tracker(strength: f32) -> PersistenceTracker {
        let mut o = ParamOverrides::default();
        o.set("persist.strength", strength);
        PersistenceTracker::new(&PersistParams::from_overrides(&o)).unwrap()
    }

    #[test]
    fn predictable_cells_get_finer() {
        let mut t = tracker(1.);
        // Two cells: error 2 and 8 per sample (mean log2 = 2).
        t.cols = 2;
        t.error = vec![2., 8.];
        let mut offsets = [0.; 2];
        t.add_offsets(2, &mut offsets);
        assert!((offsets[0] + 1.).abs() < 1e-9);
        assert!((offsets[1] - 1.).abs() < 1e-9);
        // A grid of another shape gets nothing.
        let mut other = [0.; 3];
        t.add_offsets(3, &mut other);
        assert_eq!(other, [0.; 3]);
        assert!(PersistenceTracker::new(&PersistParams::default()).is_none());
    }
}
