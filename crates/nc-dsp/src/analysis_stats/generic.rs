//! One block-parallel arithmetic graph; ISA modules only load/deinterleave
//! existing row statistics and supply the small vector operations.

#![allow(
    clippy::inline_always,
    reason = "ISA primitives and the shared graph must inline into their target-feature wrapper"
)]

use super::{AnalysisBlock, AnalysisChanges, AnalysisSummary, Moments, scalar};

pub(super) const MAX_LANES: usize = 16;
const MEAN_ROUND: i32 = (Moments::SAMPLES_8X8 / 2) as i32;
const SQUARE_ROUND: i32 = (Moments::SAMPLES_8X8 - 1) as i32;
const MAX_BLOCK_SAD: u32 = Moments::SAMPLES_8X8 * u8::MAX as u32;
const MAX_QUARTER_SAD: u32 = MAX_BLOCK_SAD / 4;

/// Methods execute their ISA, only inside its target-feature wrapper.
/// Raw loads read exactly LANES moments/quads and LANES/2 chroma tuples;
/// `run` enters only for complete batches, after Dsp's extent validation.
pub(super) trait Lanes: Copy {
    const LANES: usize;
    fn splat(value: i32) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn shr6(self) -> Self;
    fn store(self, out: &mut [i32; MAX_LANES]);
    unsafe fn moments(src: *const Moments) -> (Self, Self);
    unsafe fn changes(c: AnalysisChanges<'_>, start: usize) -> (Self, Self);
}

/// ISA and validated extents are inherited from the wrapper/Dsp table.
#[inline(always)]
pub(super) unsafe fn run<V: Lanes>(
    moments: &[Moments],
    changes: Option<AnalysisChanges<'_>>,
    out: &mut [AnalysisBlock],
) -> AnalysisSummary {
    // Specialise once per row: key frames never load SADs or classify.
    if let Some(c) = changes {
        // SAFETY: the caller guarantees this ISA and all row extents.
        unsafe { batches::<V, true>(moments, c, out) }
    } else {
        let empty = AnalysisChanges {
            luma: &[],
            chroma: [&[], &[]],
            bottom: false,
            block_limit: 0,
            quarter_limit: 0,
        };
        // SAFETY: unchanged batches never read `empty`; moments/out cover the row.
        unsafe { batches::<V, false>(moments, empty, out) }
    }
}

#[inline(always)]
unsafe fn batches<V: Lanes, const CHANGED: bool>(
    moments: &[Moments],
    c: AnalysisChanges<'_>,
    out: &mut [AnalysisBlock],
) -> AnalysisSummary {
    let full = out.len() / V::LANES * V::LANES;
    let mut total = AnalysisSummary::default();
    let mut sums = [0; MAX_LANES];
    let mut means = [0; MAX_LANES];
    let mut variance = [0; MAX_LANES];
    let mut sads = [0; MAX_LANES];
    let mut busy = [0; MAX_LANES];
    // Inputs are bounded 8x8/4x4 statistics, so a larger limit accepts all.
    let block_limit = V::splat(c.block_limit.min(MAX_BLOCK_SAD) as i32);
    let quarter_limit = V::splat(c.quarter_limit.min(MAX_QUARTER_SAD) as i32);
    for start in (0..full).step_by(V::LANES) {
        // SAFETY: a complete batch fits the validated moments slice.
        let (sum, squares) = unsafe { V::moments(moments.as_ptr().add(start)) };
        let mean = sum.add(V::splat(MEAN_ROUND)).shr6();
        let spread = squares.sub(sum.mul(sum).add(V::splat(SQUARE_ROUND)).shr6());
        sum.store(&mut sums);
        mean.store(&mut means);
        spread.store(&mut variance);
        if CHANGED {
            // SAFETY: all change slices cover this complete batch.
            let (sad, busiest) = unsafe { V::changes(c, start) };
            sad.store(&mut sads);
            // Signed distances are within +/-16320. A positive result fails
            // an inclusive limit; no lane-dependent branch is needed.
            sad.sub(block_limit)
                .max(busiest.sub(quarter_limit))
                .store(&mut busy);
        }
        for (i, dst) in out[start..start + V::LANES].iter_mut().enumerate() {
            let is_static = CHANGED && busy[i] <= 0;
            *dst = AnalysisBlock {
                mean: means[i] as u8,
                sum: sums[i] as u16,
                sum_sq_dev: variance[i] as u32,
                sad_prev: CHANGED.then_some(sads[i] as u16),
                is_static,
            };
            total.mean_sum += u64::from(dst.mean);
            total.static_blocks += usize::from(is_static);
        }
    }
    let tail = scalar(
        &moments[full..],
        CHANGED.then(|| c.tail(full)),
        &mut out[full..],
    );
    total.mean_sum += tail.mean_sum;
    total.static_blocks += tail.static_blocks;
    total
}
