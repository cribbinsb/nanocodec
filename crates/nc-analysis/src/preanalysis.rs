//! Per-block source statistics against the previous source frame.
//!
//! For every 8x8 luma block: its mean and variance, its SAD against the
//! co-located block of the previous frame, and whether it is *static*.
//! Static blocks are the fast path of the encoder (`docs/ARCHITECTURE.md` §5): they
//! are coded as skip unless a later, codec-level quality check objects.
//!
//! A block is static when, against the previous frame at zero motion:
//!
//! 1. the 8x8 luma SAD is small,
//! 2. no 4x4 quarter of it changed much on its own (catches a small object
//!    moving inside an otherwise still block), and
//! 3. neither 4x4 chroma block changed much (catches colour-only changes).
//!
//! "Small" scales with the measured noise level, so sensor noise on a still
//! scene does not read as motion. All thresholds are provisional: they are
//! named here so tuning (`docs/PROCEDURES.md` "Tunable parameters and
//! tuning") has one place to change them.
//!
//! The noise level comes from the caller (see `noise`).
//!
//! **Even rows.** With [`StatRows::Even`] every statistic reads even rows
//! only: the pass is memory-bound on the Cortex-A78 (each frame stream
//! ~55 cycles per macroblock), and rows are separate cache lines. Moments
//! come from each block's four even rows, doubled (so mean and variance keep
//! their scale). The luma change is one quadrant-SAD call on the even rows
//! of a two-block-row strip, which gives each block's left and right
//! 4-column halves over its four even rows; doubled, their sum estimates
//! the block SAD and their larger one a fully changed 4x4 quarter (8 of its
//! 16 samples, so twice the half's SAD). Chroma likewise, a call per two
//! chroma strips: each chroma column half over both quadrants' even rows
//! (16 samples, doubled) stands for the quadrant. The estimates are
//! unbiased, but noisier, so the quarter limit is widened
//! ([`EVEN_ROW_SUB_BLOCK_RATIO`]) to keep the static share.
//!
//! **Mean luma.** The frame's mean luma is reported so the encoder can spot
//! global lighting or exposure jumps (IR day/night switching, auto-exposure)
//! by comparing consecutive frames. Live camera input has no scene cuts.

use nc_common::{Picture, Plane, PlaneIndex, PlaneView};
use nc_dsp::{Dsp, Moments};

use crate::pyramid::Downsample;

use crate::grid::Grid;
use crate::noise::MEAN_ABS_DIFF_PER_SIGMA;

/// Edge of an analysis block, in luma samples.
pub const BLOCK: usize = 8;
/// `BLOCK` as a power of two.
const BLOCK_LOG2: u8 = 3;
const BLOCK_PIXELS: u32 = (BLOCK * BLOCK) as u32;
const SUB_BLOCK: usize = 4;
const SUB_BLOCK_PIXELS: u32 = (SUB_BLOCK * SUB_BLOCK) as u32;

/// Minimum static SAD per pixel, for noise-free (e.g. already compressed)
/// sources. Provisional.
const STATIC_MIN_SAD_PER_PIXEL: f32 = 1.0;
/// Static blocks may differ from the previous frame by this multiple of the
/// mean difference noise alone produces. Provisional.
const STATIC_NOISE_MARGIN: f32 = 1.5;
/// Maximum static SAD per pixel, however noisy the estimate says the source
/// is (noise of sigma ~6): beyond this, a textured block moving by a sample
/// would also pass. The estimate comes from flat blocks only (`noise`), so
/// texture does not inflate it. Provisional.
const STATIC_MAX_SAD_PER_PIXEL: f32 = 10.0;
/// A 4x4 quarter (or chroma block) may be this much busier per pixel than
/// the 8x8 limit before the block stops being static. Provisional.
const STATIC_SUB_BLOCK_FACTOR: f32 = 2.0;

/// Which rows pre-analysis' statistics read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StatRows {
    /// Every row.
    #[default]
    All,
    /// Even rows only, scaled to every row's (module docs).
    Even,
}

/// Statistics read on every other row count each sample twice.
const EVEN_ROW_SCALE: u32 = 2;
/// The quarter (4x4 luma, chroma quadrant) static limit on even-row
/// statistics relative to every row's. Calibrated so the static share
/// matches every-row detection (60 frames each of six TRAIN clips: crowd,
/// static, lowlight, bodycam, pan, noisy crowd: within -0.7..+0.4 points;
/// 1.0 made 1-15 points fewer blocks static, the noisy crowd most). SADs
/// and variances are unbiased (ratio 0.98-1.00).
const EVEN_ROW_SUB_BLOCK_RATIO: f32 = 1.5;

/// Statistics of one 8x8 luma block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockStats {
    /// Mean sample value.
    pub mean: u8,
    /// Sum of the block's samples, exact (`mean` is rounded). With
    /// `sum_sq_dev` it gives the block's exact moments ([`Self::sum_sq`]).
    pub sum: u16,
    /// Sum of squared deviations from the mean (64 x the variance).
    pub sum_sq_dev: u32,
    /// SAD against the co-located block of the previous frame (`None` for
    /// the first frame).
    pub sad_prev: Option<u32>,
    /// Unchanged since the previous frame (see module docs).
    pub is_static: bool,
}

/// Frame-level results of pre-analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct PreAnalysis {
    /// Per-block statistics, on the 8x8 grid.
    pub blocks: Grid<BlockStats>,
    /// Fraction of blocks that are static.
    pub static_fraction: f32,
    /// Mean luma of the frame (for lighting/exposure-jump detection).
    pub mean_luma: f32,
}

impl BlockStats {
    /// Sum of the block's squared samples. `sum_sq_dev` is
    /// `floor((64 * sum_sq - sum^2) / 64)`, which is `sum_sq - ceil(sum^2 / 64)`
    /// since `64 * sum_sq` is a multiple of 64, so this inverts it exactly.
    #[must_use]
    pub fn sum_sq(&self) -> u32 {
        let sum = u32::from(self.sum);
        self.sum_sq_dev + (sum * sum).div_ceil(BLOCK_PIXELS)
    }
}

impl PreAnalysis {
    /// `n · variance` of the source luma of the `2^log2_size`-square block
    /// at (`x`, `y`) (multiples of 8), from the 8x8 moments: the same value
    /// as measuring the block with the variance kernel against zero. `None`
    /// for blocks below 8x8 or reaching past the grid.
    /// Cells overhanging the picture edge hold replicated border samples, so
    /// callers use it only for blocks inside the analysed picture.
    #[must_use]
    // Always inlined into mode decision: as a call it cost H.264 ~0.1% of
    // its instructions (callgrind, h264-fast crowd 720p).
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "measured: the call costs more than the body"
    )]
    pub fn luma_energy(&self, x: usize, y: usize, log2_size: u8) -> Option<u64> {
        let cells = 1usize << log2_size.checked_sub(BLOCK_LOG2)?;
        let (col0, row0) = (x / BLOCK, y / BLOCK);
        let (mut sum, mut sum_sq) = (0u64, 0u64);
        for r in 0..cells {
            let run = self.blocks.row_run(col0, row0 + r, cells);
            if run.len() != cells {
                return None;
            }
            for b in run {
                sum += u64::from(b.sum);
                sum_sq += u64::from(b.sum_sq());
            }
        }
        // `Variance::n_variance`: the squared-sum term divides rounding down.
        let area = 1u64 << (2 * log2_size);
        Some(sum_sq - sum * sum / area)
    }

    /// Mean 8x8 luma variance of the frame (SVT's `pic_avg_variance`).
    #[must_use]
    pub fn mean_variance(&self) -> f32 {
        let cells = self.blocks.cells();
        let total: u64 = cells.iter().map(|b| u64::from(b.sum_sq_dev)).sum();
        (total as f64 / (cells.len() * BLOCK * BLOCK) as f64) as f32
    }
}

/// Thresholds for one frame, as SAD totals.
#[derive(Debug, Clone, Copy)]
struct StaticLimits {
    block: u32,
    sub_block: u32,
}

impl StaticLimits {
    /// The limits for statistics read on even rows (module docs): the
    /// quarter estimates sum half the samples, so noise spreads them wider
    /// and the quarter limit widens to keep the static share.
    fn for_even_rows(self) -> Self {
        Self {
            block: self.block,
            sub_block: (self.sub_block as f32 * EVEN_ROW_SUB_BLOCK_RATIO) as u32,
        }
    }

    fn for_noise(noise_sigma: f32) -> Self {
        let per_pixel = (STATIC_NOISE_MARGIN * MEAN_ABS_DIFF_PER_SIGMA * noise_sigma)
            .clamp(STATIC_MIN_SAD_PER_PIXEL, STATIC_MAX_SAD_PER_PIXEL);
        Self {
            block: (per_pixel * BLOCK_PIXELS as f32) as u32,
            sub_block: (STATIC_SUB_BLOCK_FACTOR * per_pixel * SUB_BLOCK_PIXELS as f32) as u32,
        }
    }
}

/// Share of 8x8 blocks whose exact moments must equal the previous frame's
/// for the frame to count as a repeat. Sensor noise makes equal moments
/// rare elsewhere; the slack admits a repeat whose source refreshed a few
/// blocks.
pub const REPEAT_BLOCK_SHARE: f32 = 0.98;

/// Whether `cur` repeats `prev` (the previous input, not the reference): a
/// source that duplicates frames (a camera at half the stream rate, a
/// stalled capture) shows almost every block's sum and spread unchanged.
#[must_use]
pub fn repeats(cur: &PreAnalysis, prev: &PreAnalysis) -> bool {
    let (a, b) = (cur.blocks.cells(), prev.blocks.cells());
    if a.len() != b.len() || a.is_empty() {
        return false;
    }
    let same = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.sum == y.sum && x.sum_sq_dev == y.sum_sq_dev)
        .count();
    same as f32 >= REPEAT_BLOCK_SHARE * a.len() as f32
}

/// Analyse `cur` against `prev` (the previous source frame, if any).
/// `noise_sigma` (the running estimate from `noise`) sets the static
/// thresholds; `stat_rows` which rows the statistics read. With
/// `half_level` (the frame's half-resolution pyramid level and how it is
/// made), each luma block row is also downscaled into it right after its
/// moments, while it is in cache: on the Cortex-A78 a separate pass re-read the
/// whole luma plane from memory (Jetson bench, crowd 720p: moments plus
/// downscale 74 -> 47 ns per macroblock).
///
/// Both pictures must have borders of at least one block, so blocks that
/// overhang the right or bottom edge read replicated samples.
///
/// One pass in strips of two block rows: each 8-row chroma strip holds the
/// chroma of two luma block rows, so its 4x4 SADs are measured once and read
/// straight into both rows' statistics (no per-frame chroma grid), while the
/// rows' luma is still in cache. The output grid is filled in place, row by
/// row.
#[must_use]
pub fn analyze(
    dsp: &Dsp,
    cur: &Picture,
    prev: Option<&Picture>,
    noise_sigma: f32,
    mut half_level: Option<(&mut Plane, Downsample)>,
    stat_rows: StatRows,
) -> PreAnalysis {
    let cols = cur.width().div_ceil(BLOCK);
    let rows = cur.height().div_ceil(BLOCK);
    let limits = match stat_rows {
        StatRows::All => StaticLimits::for_noise(noise_sigma),
        StatRows::Even => StaticLimits::for_noise(noise_sigma).for_even_rows(),
    };
    let mut blocks = vec![BlockStats::default(); cols * rows];
    // Kernels work on a whole row of blocks per call.
    let mut moments = vec![Moments::default(); cols];
    let mut quads = vec![[0u32; 4]; cols];
    let mut chroma = [
        vec![[0u32; 4]; cols.div_ceil(CHROMA_PAIR)],
        vec![[0u32; 4]; cols.div_ceil(CHROMA_PAIR)],
    ];
    // Even rows: a strip's luma halves, and the chroma halves of a strip
    // pair (U, V), measured on its first strip.
    let mut luma_halves = vec![[0u32; 4]; cols];
    let mut chroma_halves = [
        vec![[0u32; 4]; cols.div_ceil(CHROMA_PAIR)],
        vec![[0u32; 4]; cols.div_ceil(CHROMA_PAIR)],
    ];
    let (mut static_count, mut luma_sum) = (0usize, 0u64);
    let strips = rows.div_ceil(CHROMA_PAIR);
    for (strip, out_rows) in blocks.chunks_mut(CHROMA_PAIR * cols).enumerate() {
        if let Some(prev) = prev {
            match stat_rows {
                StatRows::All => chroma_strip_sads(dsp, cur, prev, strip, &mut chroma),
                StatRows::Even => {
                    if strip + 1 == strips && strip % CHROMA_PAIR == 0 {
                        // A last strip without a partner: its even rows
                        // would read past the chroma border, so all rows.
                        chroma_strip_sads(dsp, cur, prev, strip, &mut chroma);
                    } else {
                        if strip % CHROMA_PAIR == 0 {
                            chroma_pair_halves(dsp, cur, prev, strip, &mut chroma_halves);
                        }
                        spread_chroma_halves(&chroma_halves, strip % CHROMA_PAIR, &mut chroma);
                    }
                    dsp.sad_quads_8x8_row(
                        &even_rows(cur, PlaneIndex::Y, strip, cols * BLOCK),
                        &even_rows(prev, PlaneIndex::Y, strip, cols * BLOCK),
                        &mut luma_halves,
                    );
                }
            }
        }
        for (half, out) in out_rows.chunks_exact_mut(cols).enumerate() {
            let row = CHROMA_PAIR * strip + half;
            let luma = luma_row(cur, row, cols);
            match stat_rows {
                StatRows::All => dsp.moments_8x8_row(&luma, &mut moments),
                StatRows::Even => {
                    let even =
                        PlaneView::new(luma.data(), 2 * luma.stride(), luma.width(), BLOCK / 2)
                            .expect("a block row holds its even rows");
                    dsp.moments_8x4_row(&even, &mut moments);
                    for m in &mut moments {
                        m.sum *= EVEN_ROW_SCALE;
                        m.sum_sq *= EVEN_ROW_SCALE;
                    }
                }
            }
            if let Some((level, downsample)) = half_level.as_mut() {
                downscale_block_row(dsp, &luma, row, level, *downsample);
            }
            let Some(prev) = prev else {
                for (stats, m) in out.iter_mut().zip(&moments) {
                    *stats = block_stats(*m, None, false);
                    luma_sum += u64::from(stats.mean);
                }
                continue;
            };
            match stat_rows {
                StatRows::All => {
                    dsp.sad_quads_8x8_row(&luma, &luma_row(prev, row, cols), &mut quads);
                }
                StatRows::Even => spread_luma_halves(&luma_halves, half, &mut quads),
            }
            // Each chroma 8x8 block covers a pair of luma blocks per row: its
            // top quadrant pair belongs to the strip's first luma row, the
            // bottom pair to the second, one quadrant per luma block.
            let pairs = out
                .chunks_mut(CHROMA_PAIR)
                .zip(moments.chunks(CHROMA_PAIR))
                .zip(quads.chunks(CHROMA_PAIR))
                .zip(chroma[0].iter().zip(&chroma[1]));
            for (((stats, moments), quads), (u, v)) in pairs {
                for (i, ((stats, m), quad)) in stats.iter_mut().zip(moments).zip(quads).enumerate()
                {
                    let q = CHROMA_PAIR * half + i;
                    let sad = quad[0] + quad[1] + quad[2] + quad[3];
                    let busiest = quad[0].max(quad[1]).max(quad[2]).max(quad[3]);
                    // Both tests, without a branch: the first alone
                    // mispredicted on about one block in four (callgrind
                    // branch simulation, crowd 720p).
                    let is_static =
                        (sad <= limits.block) & (busiest.max(u[q]).max(v[q]) <= limits.sub_block);
                    static_count += usize::from(is_static);
                    *stats = block_stats(*m, Some(sad), is_static);
                    luma_sum += u64::from(stats.mean);
                }
            }
        }
    }
    // Integer sums convert to `f64` exactly, as the per-block means did.
    let mean_luma = luma_sum as f64 / blocks.len() as f64;
    PreAnalysis {
        static_fraction: static_count as f32 / blocks.len() as f32,
        blocks: Grid::from_cells(cols, rows, blocks),
        mean_luma: mean_luma as f32,
    }
}

/// The statistics of a block with luma `moments`.
#[inline]
fn block_stats(moments: Moments, sad_prev: Option<u32>, is_static: bool) -> BlockStats {
    BlockStats {
        mean: moments.mean_8x8(),
        sum: moments.sum as u16,
        sum_sq_dev: moments.sum_sq_dev_8x8(),
        sad_prev,
        is_static,
    }
}

/// Luma block rows (and columns) per 8x8 chroma block in 4:2:0.
const CHROMA_PAIR: usize = 2;

/// The 8-row luma strip holding block row `row`.
fn luma_row(picture: &Picture, row: usize, cols: usize) -> PlaneView<'_> {
    picture
        .plane(PlaneIndex::Y)
        .region(0, (row * BLOCK) as isize, cols * BLOCK, BLOCK)
        .expect("block row inside padded plane")
}

/// Downscale luma block row `row` (`luma`, its 8 rows) into its 4 rows of
/// the half-resolution level `half` (fewer at the level's bottom edge).
fn downscale_block_row(
    dsp: &Dsp,
    luma: &PlaneView<'_>,
    row: usize,
    half: &mut Plane,
    downsample: Downsample,
) {
    let top = row * BLOCK / 2;
    let rows = (BLOCK / 2).min(half.height().saturating_sub(top));
    if rows == 0 {
        return;
    }
    let width = half.width();
    let mut view = half.view_mut();
    let mut out = view
        .sub_view_mut(0, top, width, rows)
        .expect("rows inside the half level");
    downsample.fill(dsp, luma, &mut out);
}

/// The even rows of `plane`'s rows `16 * strip ..` (luma: a two-block-row
/// strip; chroma: a pair of chroma strips) as an 8-row view of `width`.
fn even_rows(picture: &Picture, plane: PlaneIndex, strip: usize, width: usize) -> PlaneView<'_> {
    let rows = picture
        .plane(plane)
        .region(0, (strip * 2 * BLOCK) as isize, width, 2 * BLOCK)
        .expect("strip inside padded plane");
    PlaneView::new(rows.data(), 2 * rows.stride(), width, BLOCK).expect("even rows of the strip")
}

/// Luma row `half` of a strip as quadrant inputs from its even-row halves
/// (`[left, right]` of the first block row, then of the second): each half
/// doubled as two quadrants, the other two empty, so their sum is the block
/// SAD estimate and their maximum the quarter estimate (module docs).
fn spread_luma_halves(halves: &[[u32; 4]], half: usize, quads: &mut [[u32; 4]]) {
    for (out, h) in quads.iter_mut().zip(halves) {
        let (left, right) = (h[CHROMA_PAIR * half], h[CHROMA_PAIR * half + 1]);
        *out = [EVEN_ROW_SCALE * left, EVEN_ROW_SCALE * right, 0, 0];
    }
}

/// The even-row chroma halves of chroma strips `strip` and `strip + 1`
/// (U into `out[0]`, V into `out[1]`): per 8x8 chroma block `[left, right]`
/// of the first strip, then of the second.
fn chroma_pair_halves(
    dsp: &Dsp,
    cur: &Picture,
    prev: &Picture,
    strip: usize,
    out: &mut [Vec<[u32; 4]>; 2],
) {
    // The pair's first strip index in units of strip pairs.
    let pair = strip / CHROMA_PAIR;
    for (plane, halves) in [PlaneIndex::U, PlaneIndex::V].into_iter().zip(out) {
        let width = halves.len() * BLOCK;
        dsp.sad_quads_8x8_row(
            &even_rows(cur, plane, pair, width),
            &even_rows(prev, plane, pair, width),
            halves,
        );
    }
}

/// Chroma strip `which` (0 or 1) of a pair as quadrant SADs from its
/// even-row halves: each column half, doubled, stands for both quadrants
/// it covers (module docs).
fn spread_chroma_halves(halves: &[Vec<[u32; 4]>; 2], which: usize, out: &mut [Vec<[u32; 4]>; 2]) {
    for (halves, quads) in halves.iter().zip(out.iter_mut()) {
        for (h, q) in halves.iter().zip(quads.iter_mut()) {
            let (left, right) = (
                EVEN_ROW_SCALE * h[CHROMA_PAIR * which],
                EVEN_ROW_SCALE * h[CHROMA_PAIR * which + 1],
            );
            *q = [left, right, left, right];
        }
    }
}

/// The 4x4 quadrant SADs (U into `out[0]`, V into `out[1]`) of every 8x8
/// chroma block of chroma strip `strip` against the previous frame: each
/// chroma 8x8 block's four quadrants are the chroma of a 2x2 group of luma
/// blocks. Measuring the whole strip with the quadrant kernel is cheaper
/// than checking each block's chroma on its own, even though blocks whose
/// luma already moved do not need it.
fn chroma_strip_sads(
    dsp: &Dsp,
    cur: &Picture,
    prev: &Picture,
    strip: usize,
    out: &mut [Vec<[u32; 4]>; 2],
) {
    let y = (strip * BLOCK) as isize;
    for (plane, quads) in [PlaneIndex::U, PlaneIndex::V].into_iter().zip(out) {
        let width = quads.len() * BLOCK;
        let (a, b) = (
            cur.plane(plane).region(0, y, width, BLOCK),
            prev.plane(plane).region(0, y, width, BLOCK),
        );
        let (a, b) = (
            a.expect("chroma block row inside padded plane"),
            b.expect("chroma block row inside padded plane"),
        );
        dsp.sad_quads_8x8_row(&a, &b, quads);
    }
}

#[cfg(test)]
mod tests {
    use nc_common::{Picture, PlaneIndex, PlaneView};
    use nc_dsp::{BlockSize, Dsp};

    use super::{StatRows, analyze, repeats};

    fn textured(width: usize, height: usize, phase: usize) -> Picture {
        let mut p = Picture::new(width, height, 16).unwrap();
        for plane in PlaneIndex::ALL {
            let mut v = p.plane_mut(plane).view_mut();
            for y in 0..v.height() {
                for (x, s) in v.row_mut(y).iter_mut().enumerate() {
                    *s = ((x * 37 + y * 11 + phase) % 251) as u8;
                }
            }
        }
        p.extend_borders();
        p
    }

    #[test]
    fn identical_frames_are_static_and_changes_are_not() {
        let dsp = Dsp::new();
        let a = textured(64, 32, 0);
        let mut b = a.clone();
        // Change one 4x4 corner of block (2, 1) only.
        {
            let mut v = b.plane_mut(PlaneIndex::Y).view_mut();
            for y in 8..12 {
                v.row_mut(y)[16..20].fill(255);
            }
        }
        let first = analyze(&dsp, &a, None, 0.0, None, StatRows::All);
        assert!(
            first
                .blocks
                .cells()
                .iter()
                .all(|s| s.sad_prev.is_none() && !s.is_static)
        );

        let same = analyze(&dsp, &a, Some(&a), 0.0, None, StatRows::All);
        assert!(same.blocks.cells().iter().all(|s| s.is_static));
        assert!((same.static_fraction - 1.0).abs() < f32::EPSILON);

        let changed = analyze(&dsp, &b, Some(&a), 0.0, None, StatRows::All);
        assert!(!changed.blocks.get(2, 1).is_static);
        assert!(changed.blocks.get(3, 1).is_static);
    }

    #[test]
    fn even_rows_keep_static_decisions_and_scale() {
        let dsp = Dsp::new();
        let a = textured(64, 32, 0);
        let mut b = a.clone();
        // Change one 4x4 corner of block (2, 1) only.
        {
            let mut v = b.plane_mut(PlaneIndex::Y).view_mut();
            for y in 8..12 {
                v.row_mut(y)[16..20].fill(255);
            }
        }
        let same = analyze(&dsp, &a, Some(&a), 0.0, None, StatRows::Even);
        assert!(same.blocks.cells().iter().all(|s| s.is_static));
        // An odd number of chroma strips (the last has no partner).
        let odd = textured(64, 40, 0);
        let odd_same = analyze(&dsp, &odd, Some(&odd), 0.0, None, StatRows::Even);
        assert!(odd_same.blocks.cells().iter().all(|s| s.is_static));
        let changed = analyze(&dsp, &b, Some(&a), 0.0, None, StatRows::Even);
        assert!(!changed.blocks.get(2, 1).is_static);
        assert!(changed.blocks.get(3, 1).is_static);
        // A vertically constant picture has the same moments either way.
        let mut flat_rows = a.clone();
        {
            let mut v = flat_rows.plane_mut(PlaneIndex::Y).view_mut();
            let first = v.row_mut(0).to_vec();
            for y in 1..32 {
                v.row_mut(y).copy_from_slice(&first);
            }
        }
        flat_rows.extend_borders();
        let all = analyze(&dsp, &flat_rows, None, 0.0, None, StatRows::All);
        let even = analyze(&dsp, &flat_rows, None, 0.0, None, StatRows::Even);
        assert_eq!(all.blocks, even.blocks);
    }

    #[test]
    fn luma_energy_matches_the_variance_kernel() {
        let dsp = Dsp::new();
        let picture = textured(64, 32, 5);
        let pre = analyze(&dsp, &picture, None, 0.0, None, StatRows::All);
        let zeros = [0u8; 32 * 32];
        for (x, y, log2_size) in [(0, 0, 3), (8, 16, 3), (16, 0, 4), (32, 0, 5)] {
            let n = 1usize << log2_size;
            let size = BlockSize::new(n, n).unwrap();
            let block = picture
                .plane(PlaneIndex::Y)
                .region(x as isize, y as isize, n, n)
                .unwrap();
            let zeros = PlaneView::new(&zeros, n, n, n).unwrap();
            let expected = dsp
                .variance(size, &block, &zeros)
                .n_variance((n * n) as u32);
            assert_eq!(pre.luma_energy(x, y, log2_size), Some(u64::from(expected)));
        }
        assert_eq!(pre.luma_energy(48, 16, 5), None, "past the grid");
        assert_eq!(pre.luma_energy(0, 0, 2), None, "below 8x8");
    }

    #[test]
    fn a_duplicated_frame_repeats_and_a_moved_one_does_not() {
        let dsp = Dsp::new();
        let first = analyze(&dsp, &textured(64, 48, 0), None, 0.0, None, StatRows::All);
        let again = analyze(&dsp, &textured(64, 48, 0), None, 0.0, None, StatRows::All);
        let moved = analyze(&dsp, &textured(64, 48, 1), None, 0.0, None, StatRows::All);
        assert!(repeats(&again, &first));
        assert!(!repeats(&moved, &first));
    }
}
