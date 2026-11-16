//! Finish pre-analysis statistics across independent 8x8 blocks. Pixel moments
//! and quadrant SADs are already measured by the shared sample kernels; this
//! family computes rounded means, exact variance, SAD/static classification
//! and frame totals on 8 AVX2 / 16 ICL / 4 NEON or SIMD128 blocks at a time.
//! All arithmetic fits i32: sum <= 16320, sum^2 + 63 < 2^28, quadrant
//! SAD <= 4080 and block SAD <= 16320, including doubled even-row estimates.

use crate::{Isa, Moments};

#[allow(unsafe_code)]
mod generic;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod x86;

/// Statistics of one 8x8 luma block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AnalysisBlock {
    /// Mean sample value.
    pub mean: u8,
    /// Sum of the block's samples, exact (`mean` is rounded). With
    /// `sum_sq_dev` it gives the block's exact moments ([`Self::sum_sq`]).
    pub sum: u16,
    /// Sum of squared deviations from the mean (64 x the variance).
    pub sum_sq_dev: u32,
    /// SAD against the co-located block of the previous frame (`None` for
    /// the first frame); an 8x8 SAD, at most `64 * 255`.
    pub sad_prev: Option<u16>,
    /// Unchanged since the previous frame (see module docs).
    pub is_static: bool,
}

impl AnalysisBlock {
    /// Sum of the block's squared samples. `sum_sq_dev` is
    /// `floor((64 * sum_sq - sum^2) / 64)`, which is `sum_sq - ceil(sum^2 / 64)`
    /// since `64 * sum_sq` is a multiple of 64, so this inverts it exactly.
    #[must_use]
    pub fn sum_sq(&self) -> u32 {
        let sum = u32::from(self.sum);
        self.sum_sq_dev + (sum * sum).div_ceil(Moments::SAMPLES_8X8)
    }
}

/// Already measured changes for one luma block row. Chroma 8x8 quadrants
/// cover pairs of luma blocks: top or bottom two quadrants, one per block.
#[derive(Debug, Clone, Copy)]
pub struct AnalysisChanges<'a> {
    /// Four luma quadrant SADs per block.
    pub luma: &'a [[u16; 4]],
    /// U and V quadrant SADs per pair of luma blocks.
    pub chroma: [&'a [[u16; 4]]; 2],
    /// Select the bottom pair of chroma quadrants.
    pub bottom: bool,
    /// Inclusive maximum whole-block SAD.
    pub block_limit: u32,
    /// Inclusive maximum luma/chroma quadrant SAD.
    pub quarter_limit: u32,
}

impl AnalysisChanges<'_> {
    fn tail(&self, blocks: usize) -> AnalysisChanges<'_> {
        debug_assert!(blocks.is_multiple_of(2));
        AnalysisChanges {
            luma: &self.luma[blocks..],
            chroma: [&self.chroma[0][blocks / 2..], &self.chroma[1][blocks / 2..]],
            ..*self
        }
    }
}

/// Totals of a block row, accumulated without another pass over its output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnalysisSummary {
    /// Number of static blocks.
    pub static_blocks: usize,
    /// Sum of rounded per-block means (the frame's existing luma measure).
    pub mean_sum: u64,
}

pub(crate) type StatsFn =
    unsafe fn(&[Moments], Option<AnalysisChanges<'_>>, &mut [AnalysisBlock]) -> AnalysisSummary;

pub(crate) fn scalar(
    moments: &[Moments],
    changes: Option<AnalysisChanges<'_>>,
    out: &mut [AnalysisBlock],
) -> AnalysisSummary {
    let mut total = AnalysisSummary::default();
    for (i, (out, &m)) in out.iter_mut().zip(moments).enumerate() {
        let (sad_prev, is_static) = changes.map_or((None, false), |c| {
            let quad = c.luma[i].map(u32::from);
            let sad = quad.iter().sum::<u32>();
            let q = 2 * usize::from(c.bottom) + i % 2;
            let chroma = u32::from(c.chroma[0][i / 2][q].max(c.chroma[1][i / 2][q]));
            let busiest = quad.into_iter().max().unwrap_or(0).max(chroma);
            (
                Some(sad as u16),
                (sad <= c.block_limit) && (busiest <= c.quarter_limit),
            )
        });
        *out = AnalysisBlock {
            mean: m.mean_8x8(),
            sum: m.sum as u16,
            sum_sq_dev: m.sum_sq_dev_8x8(),
            sad_prev,
            is_static,
        };
        total.mean_sum += u64::from(out.mean);
        total.static_blocks += usize::from(is_static);
    }
    total
}

pub(crate) fn kernel(isa: Isa) -> StatsFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => x86::avx2,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => x86::icl,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::stats,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::stats,
        _ => scalar,
    }
}

#[cfg(test)]
mod tests {
    use crate::{AnalysisBlock, AnalysisChanges, Dsp, Isa, Moments};

    #[test]
    #[should_panic(expected = "one moment per output block")]
    fn output_extent_is_checked() {
        let dsp = Dsp::with_isa(Isa::Scalar).expect("scalar");
        let _ = dsp.analysis_stats_row(&[], None, &mut [AnalysisBlock::default()]);
    }

    #[test]
    #[should_panic(expected = "luma changes cover the row")]
    fn luma_extent_is_checked() {
        let dsp = Dsp::with_isa(Isa::Scalar).expect("scalar");
        let changes = AnalysisChanges {
            luma: &[],
            chroma: [&[[0; 4]], &[[0; 4]]],
            bottom: false,
            block_limit: 0,
            quarter_limit: 0,
        };
        let _ = dsp.analysis_stats_row(
            &[Moments::default()],
            Some(changes),
            &mut [AnalysisBlock::default()],
        );
    }

    #[test]
    #[should_panic(expected = "chroma changes cover block pairs")]
    fn odd_chroma_extent_is_checked() {
        let dsp = Dsp::with_isa(Isa::Scalar).expect("scalar");
        let changes = AnalysisChanges {
            luma: &[[0; 4]; 3],
            chroma: [&[[0; 4]], &[[0; 4]; 2]],
            bottom: true,
            block_limit: 0,
            quarter_limit: 0,
        };
        let _ = dsp.analysis_stats_row(
            &[Moments::default(); 3],
            Some(changes),
            &mut [AnalysisBlock::default(); 3],
        );
    }
}
