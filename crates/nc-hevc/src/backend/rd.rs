//! Rate-distortion helpers of the backend's coding trials (intra `PART_NxN`,
//! intra and inter transform splits): estimated residual bits from the
//! slice's context states, reconstruction distortion, how unevenly a
//! residual's energy is spread, and saving a trial's reconstruction so the
//! losing alternative can be undone.
//!
//! Costs are `SSE + λ · bits` in the units of the engine's mode decision
//! (`HevcBackend::lambda`).

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::BlockSize;

use super::HevcBackend;
use crate::cabac::Context;
use crate::headers::{LOG2_MAX_TB, SIGN_DATA_HIDING};
use crate::predict::ModeCode;
use crate::rate::BitEstimator;
use crate::residual::write_residual;
use crate::scan::{Level, ScanKind};
use crate::syntax::PlaneResidual;
use crate::transform::{MAX_COEFFS, MAX_SIZE};

/// Bypass bins of `mpm_idx` 0 and of the other two indices (truncated
/// rice, cMax 2), and of `rem_intra_luma_pred_mode` (§9.3.3).
const MPM_FIRST_BINS: f32 = 1.0;
const MPM_OTHER_BINS: f32 = 2.0;
const REM_MODE_BINS: f32 = 5.0;

impl HevcBackend {
    /// λ of the trials' costs: the mode decision's.
    pub(super) fn rd_lambda(&self) -> f32 {
        self.frame_quantization.lambda
    }

    /// Estimated bits of `bin` in `ctx` (slice-initial state).
    pub(super) fn bin_bits(&self, ctx: Context, bin: bool) -> f32 {
        self.bin_costs.bin(ctx, bin)
    }

    /// Estimated bits of `split_transform_flag` at a node of `log2` (0 when
    /// the flag is not `present`).
    pub(super) fn split_transform_bits(&self, log2: u8, present: bool, split: bool) -> f32 {
        if present {
            let ctx = self.contexts.transform_split[LOG2_MAX_TB as usize - usize::from(log2)];
            self.bin_bits(ctx, split)
        } else {
            0.0
        }
    }

    /// Estimated bits of a coded block's `residual_coding()`, from the
    /// slice's initial contexts (adapting within the block only); 0 for an
    /// uncoded block.
    pub(super) fn residual_bits(&self, residual: PlaneResidual, luma: bool) -> f32 {
        if !residual.coded {
            return 0.0;
        }
        let store = self.level_store.borrow();
        self.levels_bits(residual.levels(&store), residual.log2, residual.scan, luma)
    }

    /// Estimated bits of `residual_coding()` of the row-major `levels` (some
    /// non-zero) of a `2^log2` block in `scan` order.
    pub(super) fn levels_bits<L: Level>(
        &self,
        levels: &[L],
        log2: u8,
        scan: ScanKind,
        luma: bool,
    ) -> f32 {
        let mut contexts = self.contexts.clone();
        let mut estimate = BitEstimator::new(&self.bin_costs);
        write_residual(
            &mut estimate,
            &mut contexts,
            self.scans.get(log2, scan),
            levels,
            luma,
            SIGN_DATA_HIDING,
        );
        estimate.bits
    }

    /// Estimated bits of an intra luma mode's syntax: its
    /// `prev_intra_luma_pred_flag` and `mpm_idx` / `rem_intra_luma_pred_mode`.
    pub(super) fn intra_mode_bits(&self, code: ModeCode) -> f32 {
        let from_list = matches!(code, ModeCode::Mpm(_));
        let bypass = match code {
            ModeCode::Mpm(0) => MPM_FIRST_BINS,
            ModeCode::Mpm(_) => MPM_OTHER_BINS,
            ModeCode::Rem(_) => REM_MODE_BINS,
        };
        self.bin_bits(self.contexts.prev_intra_luma_pred_flag, from_list) + bypass
    }

    /// Squared error of the `n`-square block of `plane` at `(x, y)` between
    /// the reconstruction and the source.
    pub(super) fn reconstruction_sse(
        &self,
        source: &Picture,
        recon: &Picture,
        plane: PlaneIndex,
        (x, y, n): (usize, usize, usize),
    ) -> u64 {
        let size = BlockSize::new(n, n).expect("transform sizes 4..32");
        let src = source.plane(plane).view();
        let rec = recon.plane(plane).view();
        let src = src.sub_view(x, y, n, n).expect("block inside the picture");
        let rec = rec.sub_view(x, y, n, n).expect("block inside the picture");
        u64::from(self.dsp.variance(size, &src, &rec).sse)
    }

    /// How unevenly the residual `src − pred` of an `n`-square block (n ≥ 8)
    /// spreads over its four quadrants: 4 · (largest quadrant energy) /
    /// (total energy), from 1 (even) to 4 (all in one quadrant); 1 when
    /// there is no residual.
    pub(super) fn residual_spread(
        &self,
        src: &PlaneView<'_>,
        pred: &PlaneView<'_>,
        n: usize,
    ) -> f32 {
        let half = n / 2;
        let size = BlockSize::new(half, half).expect("quadrants 4..16");
        let mut largest = 0u64;
        let mut total = 0u64;
        for (qx, qy) in [(0, 0), (half, 0), (0, half), (half, half)] {
            let a = src.sub_view(qx, qy, half, half).expect("quadrant");
            let b = pred.sub_view(qx, qy, half, half).expect("quadrant");
            let sse = u64::from(self.dsp.variance(size, &a, &b).sse);
            largest = largest.max(sse);
            total += sse;
        }
        if total == 0 {
            1.0
        } else {
            (4 * largest) as f32 / total as f32
        }
    }
}

/// A square of one plane's reconstruction, saved before a trial overwrites
/// it so the losing alternative can be undone.
pub(super) struct SavedSquare {
    samples: [u8; MAX_COEFFS],
    plane: PlaneIndex,
    x: usize,
    y: usize,
    n: usize,
}

impl SavedSquare {
    /// Save the `n`-square (n ≤ 32) of `plane` at `(x, y)`.
    pub(super) fn save(
        recon: &Picture,
        plane: PlaneIndex,
        (x, y, n): (usize, usize, usize),
    ) -> Self {
        debug_assert!(n <= MAX_SIZE);
        let mut samples = [0u8; MAX_COEFFS];
        let view = recon.plane(plane).view();
        for (row, out) in samples[..n * n].chunks_exact_mut(n).enumerate() {
            out.copy_from_slice(&view.row(y + row)[x..x + n]);
        }
        Self {
            samples,
            plane,
            x,
            y,
            n,
        }
    }

    /// Write the saved samples back.
    pub(super) fn restore(&self, recon: &mut Picture) {
        let n = self.n;
        let mut view = recon.plane_mut(self.plane).view_mut();
        for (row, saved) in self.samples[..n * n].chunks_exact(n).enumerate() {
            view.row_mut(self.y + row)[self.x..self.x + n].copy_from_slice(saved);
        }
    }
}
