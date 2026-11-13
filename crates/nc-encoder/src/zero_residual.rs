//! Pre-transform detection of residuals that would quantise to zero, shared
//! by the codec backends.
//!
//! A backend that skips the forward transform for an inter block must guess
//! whether its residual would have coded anything. The test here compares the
//! residual with the quantiser in the units of an orthonormal 2-D DCT, where
//! energy is preserved (Parseval), so no transform is run:
//!
//! - the block's DC coefficient is exactly `sum / n` for an `n`-square block;
//! - the energy of the remaining (AC) coefficients is the mean-removed
//!   residual energy, which bounds every one of them.
//!
//! A residual that is mostly a DC offset is therefore judged by its DC alone
//! (cheap to code DC-only), and texture by its mean-removed energy. That
//! energy is split as in Cisco Thor's early-skip check
//! (`enc/encode_block.c`: `check_early_skip_sub_block`, which transforms 2x2
//! averages of each sub-block and compares every coefficient with a fraction
//! of the first quantiser level): the mean-removed energy of each 8x8
//! sub-block (local texture, which concentrates in few coefficients) and the
//! energy of the sub-block means around the block mean (smooth gradients
//! across the block, the low frequencies). Each part must stay below the
//! squared AC zero limit times `texture_scale`. With `texture_scale` = 1 and
//! one 8x8 sub-block the test is a proof up to the transform's integer
//! rounding; larger blocks and scales trade that guarantee for more skips
//! (energy spread over many coefficients rarely puts one above the limit).
//! libaom's `predict_skip_txfm` (`av1/encoder/tx_search.c`) uses the same
//! quantiser-relative idea with a per-pixel MSE gate and a transform pass.
//!
//! Thor: BSD-2-Clause, Cisco Systems; libaom: BSD-2-Clause, Alliance for Open
//! Media. Ideas only, reimplemented (see `THIRD_PARTY.md`).

use nc_common::PlaneView;
use nc_dsp::{Dsp, Variance};

/// Side of the sub-blocks whose texture is judged separately.
const SUB_BLOCK: usize = 8;
/// Sub-blocks along a row of the largest block (64 samples).
const MAX_SUB_BLOCKS_PER_ROW: usize = 64 / SUB_BLOCK;
/// Samples in a sub-block.
const SUB_BLOCK_AREA: i64 = (SUB_BLOCK * SUB_BLOCK) as i64;

/// The largest orthonormal-DCT coefficient magnitudes that still quantise to
/// zero, in residual sample units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZeroLimits {
    /// For the DC coefficient.
    pub dc: f32,
    /// For every AC coefficient.
    pub ac: f32,
}

/// True when the residual `src − pred` of an `n`-square block (`n` a
/// multiple of 8) would quantise to all zeros by the test in the module
/// docs, with the AC energy allowance scaled by `texture_scale` (≥ 1).
///
/// # Panics
///
/// If `n` is not a multiple of 8 or the views are smaller than `n`.
#[must_use]
pub fn residual_quantises_to_zero(
    dsp: &Dsp,
    n: usize,
    src: &PlaneView<'_>,
    pred: &PlaneView<'_>,
    limits: ZeroLimits,
    texture_scale: f32,
) -> bool {
    assert!(
        n.is_multiple_of(SUB_BLOCK),
        "block side {n} is not a multiple of 8"
    );
    let ac_energy_limit = f64::from(texture_scale) * f64::from(limits.ac).powi(2);
    let mut block_sum = 0_i64;
    // Sum over sub-blocks of sum_i^2 / 64: the energy of the sub-block means.
    let mut mean_energy = 0.0_f64;
    let mut row_stats = [Variance::default(); MAX_SUB_BLOCKS_PER_ROW];
    let row_stats = &mut row_stats[..n / SUB_BLOCK];
    for y in (0..n).step_by(SUB_BLOCK) {
        let src_row = src
            .sub_view(0, y, n, SUB_BLOCK)
            .expect("row inside the block");
        let pred_row = pred
            .sub_view(0, y, n, SUB_BLOCK)
            .expect("row inside the block");
        dsp.variance_8x8_row(&src_row, &pred_row, row_stats);
        for stats in row_stats.iter() {
            let sum = i64::from(stats.sum);
            let sum_sq = (sum * sum) as f64 / SUB_BLOCK_AREA as f64;
            if f64::from(stats.sse) - sum_sq >= ac_energy_limit {
                return false;
            }
            block_sum += sum;
            mean_energy += sum_sq;
        }
    }
    let area = (n * n) as f64;
    let dc = block_sum as f64 / n as f64;
    let gradient_energy = mean_energy - (block_sum * block_sum) as f64 / area;
    dc.abs() < f64::from(limits.dc) && gradient_energy < ac_energy_limit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn views(n: usize, residual: impl Fn(usize, usize) -> i32) -> (Vec<u8>, Vec<u8>) {
        let pred = vec![128_u8; n * n];
        let src = (0..n * n)
            .map(|i| (128 + residual(i % n, i / n)).clamp(0, 255) as u8)
            .collect();
        (src, pred)
    }

    fn check(n: usize, residual: impl Fn(usize, usize) -> i32, limits: ZeroLimits) -> bool {
        let (src, pred) = views(n, residual);
        let src = PlaneView::new(&src, n, n, n).unwrap();
        let pred = PlaneView::new(&pred, n, n, n).unwrap();
        residual_quantises_to_zero(&Dsp::new(), n, &src, &pred, limits, 1.0)
    }

    #[test]
    fn dc_offset_is_judged_by_the_dc_limit_alone() {
        // A constant 2 over 16x16: DC = 512 / 16 = 32, no AC energy.
        let limits = |dc| ZeroLimits { dc, ac: 1.0 };
        assert!(check(16, |_, _| 2, limits(33.0)));
        assert!(!check(16, |_, _| 2, limits(31.0)));
    }

    #[test]
    fn local_texture_and_gradients_count_as_ac() {
        let limits = ZeroLimits {
            dc: 1000.0,
            ac: 6.0,
        };
        // Zero-mean checkerboard of ±1 in one 8x8 sub-block: energy 64 > 36.
        let texture = |x: usize, y: usize| {
            if x < 8 && y < 8 {
                if (x + y).is_multiple_of(2) { 1 } else { -1 }
            } else {
                0
            }
        };
        assert!(!check(16, texture, limits));
        // Left half +1, right half −1: flat inside every sub-block, but the
        // sub-block means carry 256 of gradient energy.
        assert!(!check(16, |x, _| if x < 8 { 1 } else { -1 }, limits));
        assert!(check(16, |_, _| 0, limits));
    }
}
