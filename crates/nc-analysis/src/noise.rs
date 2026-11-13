//! Sensor-noise level estimation.
//!
//! The noise level sets how much frame-to-frame difference still counts as
//! "unchanged" (static detection) and, later, how the encoder treats small
//! coefficients. It is estimated from **the flattest blocks' best-match
//! SADs**:
//!
//! - on a flat block, content cannot explain a difference between frames
//!   (whatever the motion, the block matches its surroundings), so its
//!   best-match SAD is set by noise alone;
//! - textured blocks are excluded: on them, motion the integer-pel search
//!   cannot follow exactly (sub-pel shifts, deformation) adds to the SAD
//!   and would read as noise;
//! - the estimate never depends on how many blocks are still, so neither a
//!   moving camera nor a static-detection threshold can feed back into it.
//!
//! Two independent samples with noise sigma s differ by `2s / sqrt(pi)`
//! (about 1.128 s) on average, which converts a SAD per sample to sigma.
//! The flattest tenth of the frame's blocks (by variance; noise itself adds
//! variance, so a fixed variance limit would find no "flat" blocks in very
//! noisy video) gives a median SAD; smoothed over frames, that is the
//! estimate.
//!
//! The first frame has no previous frame to match against; it uses the
//! variance of its flattest blocks instead (an over-estimate where the
//! flattest blocks still have texture).

use core::cmp::Ordering;

use crate::grid::Grid;
use crate::motion::MotionBlock;
use crate::preanalysis::BlockStats;

/// Mean |difference| of two independent samples per unit of noise sigma.
pub const MEAN_ABS_DIFF_PER_SIGMA: f32 = 1.128;

/// Samples in an 8x8 analysis block.
const BLOCK_PIXELS: f32 = 64.0;

/// Fraction of the frame's blocks, flattest first, that the estimate uses.
const FLAT_FRACTION: f32 = 0.1;

/// Blocks above this variance (per sample; noise of sigma 10) never count as
/// flat, so a frame without flat areas keeps the previous estimate rather than
/// reading texture as noise. Provisional.
const FLAT_MAX_VARIANCE: u32 = 100;
/// [`FLAT_MAX_VARIANCE`] as an 8x8 block's sum of squared deviations.
const FLAT_MAX_SUM_SQ_DEV: u32 = FLAT_MAX_VARIANCE * BLOCK_PIXELS as u32;
/// Variance buckets that locate the flattest blocks.
const VARIANCE_BUCKETS: usize = 256;
/// Width (log2, in sums of squared deviations) of a variance bucket: every
/// flat block falls in one below [`TEXTURED_BUCKET`].
const VARIANCE_BUCKET_SHIFT: u32 = 5;
/// The bucket of the blocks too textured to count as flat: past every flat
/// block's bucket.
const TEXTURED_BUCKET: usize = VARIANCE_BUCKETS - 1;
const _: () = assert!((FLAT_MAX_SUM_SQ_DEV >> VARIANCE_BUCKET_SHIFT) < TEXTURED_BUCKET as u32);

/// Fewer flat blocks than this and the frame's estimate is not trusted (the
/// previous estimate is kept).
const MIN_FLAT_BLOCKS: usize = 32;

/// Weight of the new frame's estimate in the running estimate: noise changes
/// slowly (gain, temperature, light level), single frames are noisy.
const SMOOTHING: f32 = 0.25;

/// Fraction of flattest blocks whose variance gives the first frame's
/// estimate.
const FIRST_FRAME_FLAT_FRACTION: f32 = 0.1;

/// The running noise estimate after a frame with motion analysis: `previous`
/// updated with this frame's flat blocks (`stats`) and their best matches
/// (`matches`, the 8x8 motion field on the same grid).
#[must_use]
pub fn update(previous: f32, stats: &Grid<BlockStats>, matches: &Grid<MotionBlock>) -> f32 {
    // The flattest blocks are those with the smallest (variance, SAD)
    // pairs among the flat ones. Count the blocks per variance bucket
    // (textured blocks into one bucket of their own, no branch), then gather
    // the SADs of the buckets below the one holding the last wanted block
    // and select only within that one: exactly the flattest `count`. One full selection over all flat blocks measured about 30
    // instructions per block.
    let bucket = |s: &BlockStats| {
        if s.sum_sq_dev <= FLAT_MAX_SUM_SQ_DEV {
            (s.sum_sq_dev >> VARIANCE_BUCKET_SHIFT) as usize
        } else {
            TEXTURED_BUCKET
        }
    };
    let mut counts = [0u32; VARIANCE_BUCKETS];
    for s in stats.cells() {
        counts[bucket(s)] += 1;
    }
    let flat_count = stats.cells().len() - counts[TEXTURED_BUCKET] as usize;
    let wanted = (stats.cells().len() as f32 * FLAT_FRACTION) as usize;
    let count = wanted.min(flat_count);
    if count < MIN_FLAT_BLOCKS {
        return previous;
    }
    let (edge, rank) = bucket_of_rank(&counts, count - 1);
    let mut flat_sads: Vec<u32> = Vec::with_capacity(count);
    // (variance, SAD) packed into one integer that orders exactly as the
    // pair does.
    let mut edge_keys: Vec<u64> = Vec::new();
    for (s, m) in stats.cells().iter().zip(matches.cells()) {
        match bucket(s).cmp(&edge) {
            Ordering::Less => flat_sads.push(m.sad),
            Ordering::Equal => edge_keys.push((u64::from(s.sum_sq_dev) << 32) | u64::from(m.sad)),
            Ordering::Greater => {}
        }
    }
    edge_keys.select_nth_unstable(rank);
    flat_sads.extend(edge_keys[..=rank].iter().map(|&key| key as u32));
    let middle = flat_sads.len() / 2;
    let (_, &mut median, _) = flat_sads.select_nth_unstable(middle);
    let sigma = median as f32 / BLOCK_PIXELS / MEAN_ABS_DIFF_PER_SIGMA;
    previous + SMOOTHING * (sigma - previous)
}

/// The bucket holding rank `k` (0-based) of the values counted in
/// `counts`, and the rank within that bucket.
fn bucket_of_rank(counts: &[u32], k: usize) -> (usize, usize) {
    let mut below = 0usize;
    for (bucket, &count) in counts.iter().enumerate() {
        let count = count as usize;
        if k < below + count {
            return (bucket, k - below);
        }
        below += count;
    }
    unreachable!("rank {k} beyond the {below} counted values");
}

/// The first frame's estimate, from the variance of its flattest blocks.
#[must_use]
pub fn first_frame(stats: &Grid<BlockStats>) -> f32 {
    let mut variances: Vec<u32> = stats.cells().iter().map(|b| b.sum_sq_dev).collect();
    let flat = ((variances.len() as f32 * FIRST_FRAME_FLAT_FRACTION) as usize).max(1);
    variances.select_nth_unstable(flat - 1);
    let mean_sum_sq_dev =
        variances[..flat].iter().map(|&v| f64::from(v)).sum::<f64>() / flat as f64;
    (mean_sum_sq_dev / f64::from(BLOCK_PIXELS)).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCK_PIXELS, FLAT_FRACTION, FLAT_MAX_SUM_SQ_DEV, MEAN_ABS_DIFF_PER_SIGMA, MIN_FLAT_BLOCKS,
        SMOOTHING, first_frame, update,
    };
    use crate::grid::Grid;
    use crate::motion::MotionBlock;
    use crate::preanalysis::BlockStats;

    fn stats(sum_sq_dev: u32) -> BlockStats {
        BlockStats {
            sum_sq_dev,
            ..BlockStats::default()
        }
    }

    fn matched(sad: u32) -> MotionBlock {
        MotionBlock {
            sad,
            ..MotionBlock::default()
        }
    }

    #[test]
    fn flat_blocks_set_the_estimate_and_textured_blocks_are_ignored() {
        // 40 flat blocks matching with the SAD noise of sigma 2 gives, and
        // 360 textured blocks with large SADs.
        let sad = (2.0 * MEAN_ABS_DIFF_PER_SIGMA * BLOCK_PIXELS) as u32;
        let mut cells = vec![stats(64); 40];
        cells.extend(vec![stats(64 * 400); 360]);
        let mut sads = vec![matched(sad); 40];
        sads.extend(vec![matched(sad * 10); 360]);
        let stats = Grid::from_cells(20, 20, cells);
        let matches = Grid::from_cells(20, 20, sads);
        // Converges towards 2 from 0.
        let mut sigma = 0.0;
        for _ in 0..40 {
            sigma = update(sigma, &stats, &matches);
        }
        assert!((sigma - 2.0).abs() < 0.05, "{sigma}");
    }

    #[test]
    fn too_few_flat_blocks_keep_the_previous_estimate() {
        let stats = Grid::from_cells(4, 1, vec![stats(64 * 400); 4]);
        let matches = Grid::from_cells(4, 1, vec![matched(1000); 4]);
        assert!((update(1.5, &stats, &matches) - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn first_frame_reads_flat_block_variance() {
        // Variance 4 (sigma 2) in the flattest blocks.
        let mut cells = vec![stats(4 * 64); 10];
        cells.extend(vec![stats(64 * 1000); 90]);
        let grid = Grid::from_cells(10, 10, cells);
        assert!((first_frame(&grid) - 2.0).abs() < 0.01);
    }

    /// The estimate by one full selection over every flat block's
    /// (variance, SAD) pair: what `update` computes, more slowly.
    fn update_by_full_selection(
        previous: f32,
        stats: &Grid<BlockStats>,
        matches: &Grid<MotionBlock>,
    ) -> f32 {
        let mut keys: Vec<u64> = stats
            .cells()
            .iter()
            .zip(matches.cells())
            .filter(|(s, _)| s.sum_sq_dev <= FLAT_MAX_SUM_SQ_DEV)
            .map(|(s, m)| (u64::from(s.sum_sq_dev) << 32) | u64::from(m.sad))
            .collect();
        let count = ((stats.cells().len() as f32 * FLAT_FRACTION) as usize).min(keys.len());
        if count < MIN_FLAT_BLOCKS {
            return previous;
        }
        keys.sort_unstable();
        let mut sads: Vec<u32> = keys[..count].iter().map(|&k| k as u32).collect();
        sads.sort_unstable();
        let sigma = sads[sads.len() / 2] as f32 / BLOCK_PIXELS / MEAN_ABS_DIFF_PER_SIGMA;
        previous + SMOOTHING * (sigma - previous)
    }

    #[test]
    fn bucketed_selection_matches_a_full_selection() {
        // Pseudo-random variances (many ties, some textured) and SADs.
        let mut state = 0x9e37_79b9_u32;
        let mut next = |modulus: u32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state % modulus
        };
        for (cols, rows, max_variance) in [(40, 30, 7000), (90, 45, 300), (16, 9, 70_000)] {
            let cells = (0..cols * rows)
                .map(|_| stats(next(max_variance)))
                .collect();
            let sads = (0..cols * rows).map(|_| matched(next(2000))).collect();
            let (stats, matches) = (
                Grid::from_cells(cols, rows, cells),
                Grid::from_cells(cols, rows, sads),
            );
            let (got, want) = (
                update(1.25, &stats, &matches),
                update_by_full_selection(1.25, &stats, &matches),
            );
            assert!(got.to_bits() == want.to_bits(), "{got} vs {want}");
        }
    }
}
