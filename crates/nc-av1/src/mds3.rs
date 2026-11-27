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

use nc_common::PlaneView;
use nc_dsp::{BlockSize, Dsp};

/// Chroma sampled-SAD denominator; the tunable numerator defaults to SVT's 5.
const CHROMA_SKIP_SAD_DEN: u32 = 8;
/// Fewest rows the even-row SAD kernel takes ([`Dsp::sad_even_rows`]).
const SAMPLED_SAD_MIN_ROWS: usize = 8;

/// Drop trailing ±1 levels preceded by at least `min_zero_run` zeros in scan
/// order. Zero also removes contiguous trailing ones, including a lone DC one.
/// Returns the new end of block.
#[must_use]
pub fn shave_trailing_ones(
    scan: &[u16],
    levels: &mut [i16],
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

/// Whether an `n`x`n` chroma residual is too weak to code, from the source
/// block `src` and its prediction `pred` (SVT samples every other row).
/// Blocks of 8 rows or more take the SAD kernel over the even rows, before
/// any residual is formed; a 4x4 block's two sampled rows are summed here.
#[must_use]
pub fn skip_chroma_residual(
    dsp: &Dsp,
    src: &PlaneView<'_>,
    pred: &PlaneView<'_>,
    n: usize,
    numerator: u32,
) -> bool {
    if numerator == 0 {
        return false;
    }
    let sad = match BlockSize::new(n, n).filter(|size| size.height() >= SAMPLED_SAD_MIN_ROWS) {
        Some(size) => dsp.sad_even_rows(size, src, pred),
        None => (0..n)
            .step_by(2)
            .map(|y| {
                src.row(y)[..n]
                    .iter()
                    .zip(&pred.row(y)[..n])
                    .map(|(&s, &p)| u32::from(s.abs_diff(p)))
                    .sum::<u32>()
            })
            .sum(),
    };
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

    /// `skip_chroma_residual` on an `n`x`n` source over a flat prediction.
    fn skips(source: &[u8], n: usize, numerator: u32) -> bool {
        let pred = vec![100u8; n * n];
        skip_chroma_residual(
            &Dsp::new(),
            &PlaneView::new(source, n, n, n).expect("source covers the block"),
            &PlaneView::new(&pred, n, n, n).expect("prediction covers the block"),
            n,
            numerator,
        )
    }

    #[test]
    fn chroma_threshold_samples_alternate_rows_and_zero_disables() {
        let mut source = vec![100u8; 16];
        source[0] = 109;
        source[4..8].fill(255); // This row is not sampled.
        assert!(skips(&source, 4, 5)); // 9*8 < 5*16.
        source[0] = 90;
        assert!(!skips(&source, 4, 5)); // Strict boundary.
        assert!(skips(&source, 4, 6));
        source.fill(100);
        assert!(skips(&source, 4, 5));
        assert!(!skips(&source, 4, 0));
    }

    #[test]
    fn chroma_threshold_kernel_rows_match_the_sampled_sum() {
        for n in [8, 16, 32] {
            let mut source = vec![100u8; n * n];
            // Odd rows are not sampled; even rows carry a SAD of exactly
            // the bound (n * n * 5 / 8), which is not below it.
            for y in (1..n).step_by(2) {
                source[y * n..][..n].fill(0);
            }
            let mut remaining = n * n * 5 / 8;
            for v in source.chunks_exact_mut(n).step_by(2).flatten() {
                let step = remaining.min(100);
                *v = 100 + step as u8;
                remaining -= step;
            }
            assert_eq!(remaining, 0);
            assert!(!skips(&source, n, 5), "{n}x{n} at the bound");
            source[0] -= 1;
            assert!(skips(&source, n, 5), "{n}x{n} one below the bound");
        }
    }

    #[test]
    fn shaving_drops_isolated_trailing_ones() {
        let scan: Vec<u16> = (0..16).collect();
        let mut levels = vec![0i16; 16];
        levels[0] = 5;
        levels[12] = 1;
        let eob = shave_trailing_ones(&scan, &mut levels, 13, 8);
        assert_eq!((eob, levels[12]), (1, 0));
        // A ±1 close to other levels stays.
        levels[3] = -1;
        assert_eq!(shave_trailing_ones(&scan, &mut levels, 4, 8), 4);
    }
}
