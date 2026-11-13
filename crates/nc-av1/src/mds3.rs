//! SVT-AV1's light-PD1 "MDS3" shortcuts for inter blocks
//! (`docs/reference/svt-av1-m12-rtc.md` §3, `full_loop_core_light_pd1`):
//!
//! - **Coefficient shaving** (`shave_coeff`): trailing ±1 levels separated
//!   from the rest by at least 8 zeros (in scan order) are dropped.
//! - **Chroma energy skip** (`lpd1_chroma_energy_skip`): chroma residuals
//!   with tiny energy are not transformed.
//!
//! Historical measurements (quick clips, 10 s intra period, vs the same
//! encoder without them):
//! - SVT's transform-skip score (`lpd1_should_perform_tx`, `lpd1_tx` level 3
//!   thresholds): +2.8% BD-rate (PSNR-YUV) overall, +12% on bodycam.
//!   Now available with explicit controls behind `av1.pre_tx_skip` (score in
//!   `nc_encoder::pre_tx_skip_score`).
//! - SVT's block-skip check (`lpd1_blk_skip_luma_rd`, 75% margin) with the
//!   linear EOB rate (`coeff_rate_est_lvl 0`): −21 points on the static
//!   clip but unstable on high motion at coarse quantisers (crowd curves
//!   degenerated: residuals dropped, quality decaying, intra refresh
//!   costs). It needs SVT's own λ scale to be safe; revisit with the
//!   rate–distortion λ calibration.

/// Chroma sampled-SAD denominator; the tunable numerator defaults to SVT's 5.
const CHROMA_SKIP_SAD_DEN: u32 = 8;

/// Drop trailing ±1 levels preceded by at least `min_zero_run` zeros in scan
/// order. Zero also removes contiguous trailing ones, including a lone DC one.
/// Returns the new end of block.
#[must_use]
pub fn shave_trailing_ones(
    scan: &[u16],
    levels: &mut [i32],
    mut eob: usize,
    min_zero_run: usize,
) -> usize {
    while eob > 0 {
        let last = usize::from(scan[eob - 1]);
        if levels[last].abs() != 1 {
            break;
        }
        let previous = scan[..eob - 1]
            .iter()
            .rposition(|&p| levels[usize::from(p)] != 0);
        let zeros = previous.map_or(eob - 1, |p| eob - 2 - p);
        if zeros < min_zero_run {
            break;
        }
        levels[last] = 0;
        eob = previous.map_or(0, |p| p + 1);
    }
    eob
}

/// Whether a chroma residual (`n`x`n`, row-major) is too weak to code.
#[must_use]
pub fn skip_chroma_residual(residual: &[i16], n: usize, numerator: u32) -> bool {
    if numerator == 0 {
        return false;
    }
    let sad: u32 = (0..n)
        .step_by(2)
        .map(|r| {
            residual[r * n..(r + 1) * n]
                .iter()
                .map(|&v| u32::from(v.unsigned_abs()))
                .sum::<u32>()
        })
        .sum();
    sad * CHROMA_SKIP_SAD_DEN < numerator * (n * n) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaving_threshold_has_exact_gap_boundary_and_zero_is_aggressive() {
        let scan: Vec<u16> = (0..16).collect();
        let mut levels = vec![0; 16];
        levels[0] = 5;
        levels[9] = 1; // Exactly eight zeros between the nonzero coefficients.
        let mut conservative = levels.clone();
        assert_eq!(shave_trailing_ones(&scan, &mut conservative, 10, 9), 10);
        assert_eq!(shave_trailing_ones(&scan, &mut levels, 10, 8), 1);
        levels[1] = -1;
        levels[2] = 1;
        assert_eq!(shave_trailing_ones(&scan, &mut levels, 3, 0), 1);
        assert_eq!(&levels[..3], &[5, 0, 0]);
        levels[0] = 1;
        assert_eq!(shave_trailing_ones(&scan, &mut levels, 1, 0), 0);
    }

    #[test]
    fn chroma_threshold_samples_alternate_rows_and_zero_disables() {
        let mut residual = vec![0i16; 16];
        residual[0] = 9;
        residual[4..8].fill(255); // This row is not sampled.
        assert!(skip_chroma_residual(&residual, 4, 5)); // 9*8 < 5*16.
        residual[0] = 10;
        assert!(!skip_chroma_residual(&residual, 4, 5)); // Strict boundary.
        assert!(skip_chroma_residual(&residual, 4, 6));
        residual.fill(0);
        assert!(skip_chroma_residual(&residual, 4, 5));
        assert!(!skip_chroma_residual(&residual, 4, 0));
    }

    #[test]
    fn shaving_drops_isolated_trailing_ones() {
        let scan: Vec<u16> = (0..16).collect();
        let mut levels = vec![0i32; 16];
        levels[0] = 5;
        levels[12] = 1;
        let eob = shave_trailing_ones(&scan, &mut levels, 13, 8);
        assert_eq!((eob, levels[12]), (1, 0));
        // A ±1 close to other levels stays.
        levels[3] = -1;
        assert_eq!(shave_trailing_ones(&scan, &mut levels, 4, 8), 4);
    }
}
