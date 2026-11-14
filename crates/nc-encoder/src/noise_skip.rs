//! The noise skip (`decision.noise_skip`): a still block whose
//! zero-vector prediction is already as close to the source as the
//! camera's own noise allows is coded as a zero-vector skip, whatever the
//! frame quantizer.
//!
//! What it skips: coding sensor noise on still content. A live camera on a
//! still scene changes every frame by noise alone; the quantizer-relative
//! zero test (`still_skip`) passes only while the step is coarse, so a rate
//! controller with bits to spare lowers the step until the noise is
//! codable and fills any target (docs/analysis/static-noise-2026-11-14.md:
//! AV1 Default at 1 Mbps on a still webcam, 65 kbit per frame with a flat
//! PSNR against the clean scene). Skips taken here do not depend on the
//! quantizer, so rate control cannot re-spend them: the stream undershoots.
//!
//! Why it is safe:
//! - The test is against the *current* source: the zero-vector residual's
//!   mean squared error must be within `scale` times the cell's noise
//!   variance in every 8x8 luma cell and every 8x8 chroma block. Its
//!   expectation is the noise variance plus the reconstruction's own error
//!   against the scene, so with `scale` near 2 a block skips only once its
//!   reconstruction is within about one noise sigma of the scene. A poor
//!   picture (a starved key) keeps being refined; slow drift and light
//!   changes accumulate in the residual until they exceed the bound and are
//!   coded.
//! - Only blocks whose every cell the zero vector has predicted about as
//!   well as the best match ([`zero_explains`]) for [`MIN_RUN`] frames in a
//!   row are tested. Pre-analysis' one-frame static
//!   test cannot be the gate: its threshold follows the analyzer's noise
//!   estimate from the flattest blocks, which on a webcam are clipped
//!   highlights with no noise at all, so noisy mid-tones read as changed
//!   (a still webcam at 854x480: 7% of the picture static).
//! - The noise map learns a cell's variance only from such still frames,
//!   from its zero-motion SAD against the previous source (two noisy
//!   samples differ by `MEAN_ABS_DIFF_PER_SIGMA` sigma on average), so
//!   detail never reads as noise; a cell that moves forgets its estimate,
//!   and an
//!   estimate is capped at [`CAP_TO_MEDIAN`] times the frame's median, so
//!   slow motion or flicker the search does not follow in one cell cannot
//!   teach it a large noise.
//! - Chroma has no temporal estimate of its own: its bound is the luma
//!   variance times `decision.noise_skip_chroma_ratio` squared.

use nc_analysis::Grid;
use nc_analysis::MotionAnalysis;
use nc_analysis::motion::MotionBlock;
use nc_analysis::noise::MEAN_ABS_DIFF_PER_SIGMA;
use nc_analysis::preanalysis::PreAnalysis;
use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::{Dsp, Variance};

use crate::partition::Block;

/// Side of a motion-field cell, in luma samples.
const CELL: usize = 8;
/// Samples in a cell.
const CELL_SAMPLES: f32 = (CELL * CELL) as f32;
/// Smallest block side tested: its 4:2:0 chroma must hold whole 8x8 blocks.
const MIN_BLOCK: usize = 2 * CELL;
/// Most cells across a tested block (a 128-sample AV1 superblock).
const MAX_COLS: usize = 128 / CELL;
/// Weight of a frame's sample in a cell's running variance: about four
/// frames' memory, as the analyzer smooths its sensor-noise estimate.
const SMOOTHING: f32 = 0.25;
/// Still frames in a row ([`zero_explains`]) before a cell's estimate is
/// trusted: the noise skip's own stillness test.
const MIN_RUN: u8 = 4;
/// A cell is still when its zero-motion SAD is at most this multiple of
/// its best match's SAD (plus [`ZERO_MATCH_SLACK`]): on flat noisy areas
/// the motion search picks whichever vector matches the noise best (the
/// "fictional" vectors of a still webcam), gaining only a fraction of the
/// SAD, while real motion of texture multiplies it.
const ZERO_MATCH_RATIO: u32 = 5;
/// [`ZERO_MATCH_RATIO`] as a fraction: 5/4.
const ZERO_MATCH_DIVISOR: u32 = 4;
/// Absolute slack of the zero-match test: one code value per sample of an
/// 8x8 cell.
const ZERO_MATCH_SLACK: u32 = 64;
/// Largest camera displacement per frame (`GlobalMotion::peak_displacement`,
/// luma samples) at which the noise map still learns. On a moving camera,
/// flat areas move with nearly equal SADs at every vector, so the zero
/// vector "explains" them and their motion error read as noise: the noise
/// skip then kept stale content on a pan (`pool_pan_jaad_video_0194`,
/// 150 kbps, AV1, with the skip-aware budget: 1.4 dB lost, 14-49% of the
/// picture skipped). Still cameras read at most 0.5 samples (the TRAIN
/// and VAL clips, `rate_control.rs` `STARTUP_CAMERA_MOTION`).
const MAX_STILL_CAMERA_SAMPLES: f32 = 0.5;
/// The noise skip acts only where the frame's quantizer step is at most
/// this many cell noise sigmas. At a coarser step the codec's own
/// zero-residual test already skips what noise alone changes, and what is
/// left within "noise" is the coarse reconstruction's own error: skipping
/// it froze a poor picture. On `pool_pan_jaad_video_0194` at 150 kbps
/// (AV1 qindex 255, a 228-sample step) a 4% budget change flipped which
/// blocks the noise test passed and lost 1.4 dB from one base frame on.
/// A still webcam at 1 Mbps codes at steps of 7-20 samples with cell sigmas
/// of 0.5-3.5, inside the bound.
const CODABLE_STEPS_PER_SIGMA: f32 = 16.;
/// Smallest cell sigma: half a code value, the rounding the
/// reconstruction carries even where the source does not change at all
/// (clipped highlights read sigma 0).
const MIN_SIGMA: f32 = 0.5;
/// Largest cell sigma as a multiple of the frame's median zero-vector
/// sigma. The camera's noise varies with brightness (a still webcam:
/// sigma 0.3 in bright areas, 2-3.5 in mid-tones, median 0.7), so the cap
/// is wide; it only stops one cell from learning motion as noise.
const CAP_TO_MEDIAN: f32 = 6.;

/// One cell's running noise estimate.
#[derive(Debug, Clone, Copy, Default)]
struct Cell {
    /// Running luma noise variance.
    variance: f32,
    /// Still frames in a row, saturating at [`MIN_RUN`].
    run: u8,
}

/// Per-8x8-cell temporal noise variance of the still parts of the picture.
#[derive(Debug, Clone)]
pub(crate) struct NoiseMap {
    cells: Grid<Cell>,
    /// Scratch for the frame median.
    samples: Vec<f32>,
}

impl NoiseMap {
    pub(crate) fn new() -> Self {
        Self {
            cells: Grid::filled(0, 0, Cell::default()),
            samples: Vec::new(),
        }
    }

    /// Learn from a frame's pre-analysis (each cell's zero-motion SAD
    /// against the previous source) and its motion analysis: cells the zero
    /// vector predicts about as well as the best match ([`zero_explains`])
    /// update their variance from the zero-motion SAD, other cells forget
    /// theirs. A frame whose camera moves (`camera_still` false,
    /// [`camera_is_still`]) forgets the whole map.
    pub(crate) fn update(
        &mut self,
        pre: &PreAnalysis,
        fine: &Grid<MotionBlock>,
        camera_still: bool,
    ) {
        let blocks = &pre.blocks;
        if !camera_still {
            self.cells = Grid::filled(blocks.cols(), blocks.rows(), Cell::default());
            return;
        }
        if self.cells.cols() != blocks.cols() || self.cells.rows() != blocks.rows() {
            self.cells = Grid::filled(blocks.cols(), blocks.rows(), Cell::default());
        }
        let sigma = |sad: u32| sad as f32 / CELL_SAMPLES / MEAN_ABS_DIFF_PER_SIGMA;
        let still_sad = |col: usize, row: usize| {
            let zero = blocks.get(col, row).sad_prev?;
            let best = fine.try_get(col as isize, row as isize)?;
            zero_explains(zero, best.sad).then_some(zero)
        };
        self.samples.clear();
        self.samples.extend(
            blocks
                .positions()
                .filter_map(|(col, row)| still_sad(col, row))
                .map(sigma),
        );
        if self.samples.is_empty() {
            self.cells = Grid::filled(blocks.cols(), blocks.rows(), Cell::default());
            return;
        }
        let middle = self.samples.len() / 2;
        let (_, &mut median, _) = self.samples.select_nth_unstable_by(middle, f32::total_cmp);
        let cap = CAP_TO_MEDIAN * median.max(MIN_SIGMA);
        for (col, row) in blocks.positions() {
            let zero = still_sad(col, row);
            let cell = self.cells.get_mut(col, row);
            match zero {
                Some(sad) => {
                    let s = sigma(sad).clamp(MIN_SIGMA, cap);
                    let sample = s * s;
                    cell.variance = if cell.run == 0 {
                        sample
                    } else {
                        cell.variance + SMOOTHING * (sample - cell.variance)
                    };
                    cell.run = cell.run.saturating_add(1).min(MIN_RUN);
                }
                None => *cell = Cell::default(),
            }
        }
    }

    /// Whether every cell `block` covers has a trusted estimate (still for
    /// [`MIN_RUN`] frames): only such blocks are worth testing.
    pub(crate) fn trusts(&self, block: &Block) -> bool {
        let (col0, row0) = (block.x / CELL, block.y / CELL);
        let (cols, rows) = (block.width() / CELL, block.height() / CELL);
        block.width() >= MIN_BLOCK
            && block.height() >= MIN_BLOCK
            && col0 + cols <= self.cells.cols()
            && row0 + rows <= self.cells.rows()
            && (0..rows).all(|r| {
                self.cells
                    .row_run(col0, row0 + r, cols)
                    .iter()
                    .all(|c| c.run >= MIN_RUN)
            })
    }

    /// Whether the zero-vector residual of `block` (`source` against the
    /// co-located `reference`) is within `scale` times its cells' noise
    /// variance in every 8x8 luma cell, and within `scale` times that
    /// variance times `chroma_ratio` squared in every 8x8 chroma block.
    /// Blocks [`Self::trusts`] does not cover never pass.
    pub(crate) fn explains(
        &self,
        dsp: &Dsp,
        block: &Block,
        (source, reference): (&Picture, &Picture),
        scale: f32,
        chroma_ratio: f32,
        step: f32,
    ) -> bool {
        let (width, height) = (block.width(), block.height());
        let codable = codable_variance(step);
        let (col0, row0) = (block.x / CELL, block.y / CELL);
        let (cols, rows) = (width / CELL, height / CELL);
        if cols > MAX_COLS || !self.trusts(block) {
            return false;
        }
        let mut buffer = [Variance::default(); MAX_COLS];
        let sse = &mut buffer[..cols];
        let luma_ok = (0..rows).all(|r| {
            let run = self.cells.row_run(col0, row0 + r, cols);
            let y = block.y + r * CELL;
            let (src, rf) = (
                row_view(source, PlaneIndex::Y, block.x, y, width),
                row_view(reference, PlaneIndex::Y, block.x, y, width),
            );
            dsp.variance_8x8_row(&src, &rf, sse);
            run.iter().zip(sse.iter()).all(|(c, v)| {
                c.variance >= codable && v.sse as f32 <= scale * c.variance * CELL_SAMPLES
            })
        });
        if !luma_ok {
            return false;
        }
        // Each 8x8 chroma block covers 2x2 luma cells; its bound is their
        // mean variance.
        let chroma_scale = scale * chroma_ratio * chroma_ratio;
        let sse = &mut buffer[..cols / 2];
        [PlaneIndex::U, PlaneIndex::V].into_iter().all(|plane| {
            (0..rows / 2).all(|r| {
                let y = block.y / 2 + r * CELL;
                let (src, rf) = (
                    row_view(source, plane, block.x / 2, y, width / 2),
                    row_view(reference, plane, block.x / 2, y, width / 2),
                );
                dsp.variance_8x8_row(&src, &rf, sse);
                let above = self.cells.row_run(col0, row0 + 2 * r, cols);
                let below = self.cells.row_run(col0, row0 + 2 * r + 1, cols);
                sse.iter().enumerate().all(|(i, v)| {
                    let variance = (above[2 * i].variance
                        + above[2 * i + 1].variance
                        + below[2 * i].variance
                        + below[2 * i + 1].variance)
                        / 4.;
                    v.sse as f32 <= chroma_scale * variance * CELL_SAMPLES
                })
            })
        })
    }
}

/// The smallest cell noise variance the noise skip acts on at a frame
/// quantizer of sample step `step`: noise of sigma below
/// `step / CODABLE_STEPS_PER_SIGMA` is not what a step that coarse codes.
fn codable_variance(step: f32) -> f32 {
    let sigma = step / CODABLE_STEPS_PER_SIGMA;
    sigma * sigma
}

/// Whether the camera is still in a frame's motion analysis (pictures of
/// `width` x `height`): its peak displacement below
/// [`MAX_STILL_CAMERA_SAMPLES`].
pub(crate) fn camera_is_still(motion: &MotionAnalysis, (width, height): (usize, usize)) -> bool {
    motion.global.peak_displacement(width, height) < MAX_STILL_CAMERA_SAMPLES
}

/// Whether a cell's zero-motion SAD `zero` is within what noise explains
/// of its best match's SAD `best`.
pub(crate) fn zero_explains(zero: u32, best: u32) -> bool {
    zero * ZERO_MATCH_DIVISOR <= best * ZERO_MATCH_RATIO + ZERO_MATCH_SLACK * ZERO_MATCH_DIVISOR
}

/// An 8-row strip of `plane` of `picture`, `width` wide from (`x`, `y`).
fn row_view(
    picture: &Picture,
    plane: PlaneIndex,
    x: usize,
    y: usize,
    width: usize,
) -> PlaneView<'_> {
    picture
        .plane(plane)
        .view()
        .sub_view(x, y, width, CELL)
        .expect("a block inside the picture")
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_analysis::Mv;
    use nc_analysis::preanalysis::BlockStats;

    fn frame(cols: usize, rows: usize, zero: u32, best: u32) -> (PreAnalysis, Grid<MotionBlock>) {
        let stats = BlockStats {
            mean: 128,
            sum: 128 * 64,
            sum_sq_dev: 0,
            sad_prev: Some(zero),
            is_static: false,
        };
        let pre = PreAnalysis {
            blocks: Grid::filled(cols, rows, stats),
            static_fraction: 0.,
            mean_luma: 128.,
        };
        let mv = Mv { x: 1, y: 0 };
        (pre, Grid::filled(cols, rows, MotionBlock { mv, sad: best }))
    }

    #[test]
    fn a_cell_is_trusted_after_a_still_run_and_forgets_when_it_moves() {
        let mut map = NoiseMap::new();
        let block = Block::square(0, 0, 4);
        // Zero SAD 72 per 64 samples (sigma about 1), a noise match at 64.
        let (pre, fine) = frame(4, 4, 72, 64);
        for _ in 1..MIN_RUN {
            map.update(&pre, &fine, true);
        }
        assert!(!map.trusts(&block));
        map.update(&pre, &fine, true);
        assert!(map.trusts(&block));
        let cell = *map.cells.get(1, 1);
        assert!((cell.variance - (72. / 64. / MEAN_ABS_DIFF_PER_SIGMA).powi(2)).abs() < 1e-4);
        // Texture moving: the zero vector is far worse than the best match.
        let (pre, fine) = frame(4, 4, 2000, 300);
        map.update(&pre, &fine, true);
        assert!(!map.trusts(&block));
        // A moving camera forgets everything, still-looking cells included.
        let (pre, fine) = frame(4, 4, 72, 64);
        for _ in 0..MIN_RUN {
            map.update(&pre, &fine, true);
        }
        map.update(&pre, &fine, false);
        assert!(!map.trusts(&block));
    }

    #[test]
    fn one_cell_cannot_learn_more_than_the_cap_nor_less_than_rounding() {
        let (mut pre, mut fine) = frame(4, 4, 64, 64);
        pre.blocks.get_mut(2, 2).sad_prev = Some(64 * 64);
        fine.get_mut(2, 2).sad = 64 * 64;
        pre.blocks.get_mut(3, 3).sad_prev = Some(0);
        let mut map = NoiseMap::new();
        map.update(&pre, &fine, true);
        let median = 1. / MEAN_ABS_DIFF_PER_SIGMA;
        let capped = map.cells.get(2, 2).variance;
        assert!((capped - (CAP_TO_MEDIAN * median).powi(2)).abs() < 1e-3);
        assert!((map.cells.get(3, 3).variance - MIN_SIGMA * MIN_SIGMA).abs() < 1e-6);
    }
}
