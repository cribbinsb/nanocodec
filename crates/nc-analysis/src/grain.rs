//! Film grain noise model: the source's own noise (sensor noise, film
//! grain), described as an auto-regressive (AR) filter and a strength per
//! intensity, for codecs whose decoders re-synthesize grain (AV1 §7.18.3,
//! AV2, the VVC film grain characteristics SEI).
//!
//! The noise is measured in the source alone, so its strength does not
//! depend on the bitrate: it is what a 3x3 binomial blur (the cheapest
//! real-time denoiser) removes from the source. It is read only on flat
//! blocks, where the blur removes noise and nothing else: blocks whose
//! blurred content, once a plane is fitted to remove its shading, is quiet.
//! Edges, foliage and spray keep structure through the blur, so they are
//! never counted as noise: grain stands in for noise the coder removed,
//! not for lost detail. Being a high-pass, the measure keeps the fine
//! grain that reads as noise and leaves slow, blotchy variation out of the
//! model.
//!
//! On the same blocks the model also measures how much of that noise coding
//! removed (the source's noise less the coded picture's), so a codec can
//! cap the grain at it: at high rates the coded picture keeps most of the
//! noise, and full-strength grain on top would double it.
//!
//! The model follows the design of libaom's noise model
//! (`aom_dsp/noise_model.c`, BSD-2-Clause, see `THIRD_PARTY.md`): source
//! noise against a denoised source on flat blocks (found after a plane fit
//! removes their shading), an AR fit of that noise, chroma with one more
//! term on the co-located luma noise, and a noise strength per intensity
//! bin (chroma strength as a function of luma, as libaom's). It departs
//! where real time needs it: the denoiser is a fixed 3x3 blur on the
//! measured blocks (not a Wiener filter over the frame), and the AR filter
//! solves Yule-Walker equations built from the noise's autocorrelation (one
//! dot product per offset, no per-sample matrix update).

use nc_common::{Picture, PlaneIndex, PlaneView};

/// Largest AR lag (AV1 `ar_coeff_lag` is 0 to 3).
pub const MAX_LAG: usize = 3;

/// Luma intensity bins of the strength function, over 8-bit samples.
pub const INTENSITY_BINS: usize = 8;

/// 8-bit levels per intensity bin.
pub const BIN_WIDTH: usize = 256 / INTENSITY_BINS;

/// Side of the luma analysis block (chroma: half).
const BLOCK: usize = 16;

/// Fewer measured blocks than this in a bin and the bin's strength is not
/// trusted.
const MIN_BIN_BLOCKS: f64 = 4.0;

/// Fewer measured luma samples than this and the frame has no model
/// (four 16x16 blocks).
const MIN_SAMPLES: f64 = 1024.0;

/// Ridge added to the Yule-Walker diagonal, relative to the variance: keeps
/// the solve stable on nearly white or nearly degenerate noise.
const RIDGE: f64 = 1e-4;

/// Fraction bits of the stored noise samples: the plane-fit residual is not
/// an integer, and whole levels would add rounding noise (1/12 level²)
/// comparable to the quiet noise being measured.
const NOISE_FRACTION_BITS: u32 = 2;

/// Stored noise samples are clamped to ± this (128 levels): a row's
/// products then fit `i32` for widths up to 8192 (512² · 8192 = 2³¹),
/// and measured blocks are far quieter than the clamp.
const NOISE_CLAMP: f64 = 512.0;

/// The AR neighbourhood of lag `lag`: every (row, column) offset above and
/// left of a sample, rows top to bottom and columns left to right (the
/// coefficient order of AV1 `ar_coeffs_*`, §7.18.3.3).
pub fn ar_offsets(lag: usize) -> impl Iterator<Item = (isize, isize)> {
    let lag = lag as isize;
    (-lag..=0)
        .flat_map(move |dr| (-lag..=lag).map(move |dc| (dr, dc)))
        .take_while(|&(dr, dc)| !(dr == 0 && dc == 0))
}

/// A fitted noise model.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseModel {
    /// AR lag.
    pub lag: usize,
    /// Luma AR coefficients, in [`ar_offsets`] order.
    pub luma_ar: Vec<f64>,
    /// Cb and Cr AR coefficients in [`ar_offsets`] order, then the
    /// coefficient of the co-located luma noise (mean of its 2x2 samples).
    pub chroma_ar: [Vec<f64>; 2],
    /// Source noise standard deviation (8-bit levels) of Y, Cb and Cr per
    /// luma intensity bin; `None` where too few blocks were measured.
    pub noise: [[Option<f64>; INTENSITY_BINS]; 3],
    /// Standard deviation of the noise coding removed (the source's noise
    /// less the coded picture's) on the same blocks, as `noise`.
    pub loss: [[Option<f64>; INTENSITY_BINS]; 3],
}

impl NoiseModel {
    /// The grain to synthesize per plane and bin: the source noise, capped
    /// at `loss_cap` times the noise coding removed, so that grain never
    /// adds more than coding took away.
    #[must_use]
    pub fn grain_strength(&self, loss_cap: f64) -> [[Option<f64>; INTENSITY_BINS]; 3] {
        std::array::from_fn(|plane| {
            std::array::from_fn(|bin| {
                let (noise, loss) = (self.noise[plane][bin]?, self.loss[plane][bin]?);
                Some(noise.min(loss_cap * loss))
            })
        })
    }
}

/// What [`NoiseStats::measure`] measures, and which blocks count as flat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureSettings {
    /// AR lag (at most [`MAX_LAG`]).
    pub lag: usize,
    /// A block is flat when its blurred content, less its shading plane,
    /// deviates by at most this many 8-bit levels (plus the share loud
    /// noise leaves after the blur): texture and edges do more.
    pub flat_structure: f64,
    /// Largest kurtosis of what the blur removed (3 for Gaussian noise):
    /// sparse lines and specks are heavier-tailed.
    pub max_kurtosis: f64,
    /// Whether chroma noise is measured too (its correlations cost more
    /// than luma's); without it the chroma model is empty.
    pub chroma: bool,
}

/// The autocorrelation offsets the Yule-Walker equations of lag `lag` need:
/// every difference of two neighbourhood positions (sample included), one of
/// each `±d` pair: rows `0..=lag`, columns `-2 lag..=2 lag`, and on row 0
/// only columns `>= 0`.
fn correlation_offsets(lag: usize) -> Vec<(usize, isize)> {
    let reach = 2 * lag as isize;
    (0..=lag)
        .flat_map(|dr| (-reach..=reach).map(move |dc| (dr, dc)))
        .filter(|&(dr, dc)| dr > 0 || dc >= 0)
        .collect()
}

/// One frame's measured blocks of one plane in one luma intensity bin:
/// each block's variance of the source noise and of the noise coding
/// removed.
#[derive(Debug, Clone, Default)]
struct BinSamples {
    noise: Vec<f64>,
    loss: Vec<f64>,
}

impl BinSamples {
    fn add(&mut self, noise: f64, loss: f64) {
        self.noise.push(noise);
        self.loss.push(loss);
    }

    /// The bin's statistics: median variances, so that a few blocks of
    /// texture that passed the flatness tests do not decide the bin.
    fn stats(mut self) -> BinStats {
        BinStats {
            noise: median(&mut self.noise),
            loss: median(&mut self.loss),
            blocks: self.noise.len() as f64,
        }
    }
}

/// The median of `values` (0 when empty); reorders them.
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let (count, middle) = (values.len(), values.len() / 2);
    let (below, &mut upper, _) = values.select_nth_unstable_by(middle, f64::total_cmp);
    if count.is_multiple_of(2) {
        let lower = below.iter().copied().fold(f64::MIN, f64::max);
        f64::midpoint(lower, upper)
    } else {
        upper
    }
}

/// One plane's measured blocks in one luma intensity bin: typical (median)
/// variances per block of the source noise and of the noise coding
/// removed, and the block count.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct BinStats {
    noise: f64,
    loss: f64,
    blocks: f64,
}

impl BinStats {
    /// Standard deviations of noise and loss, when enough blocks were
    /// measured.
    fn deviations(&self) -> (Option<f64>, Option<f64>) {
        if self.blocks >= MIN_BIN_BLOCKS {
            (Some(self.noise.sqrt()), Some(self.loss.sqrt()))
        } else {
            (None, None)
        }
    }
}

/// Second-order statistics of the noise, as means per measured sample, so
/// that frames combine by weighted averaging. Correlations are in stored
/// noise units (see [`NOISE_FRACTION_BITS`]): only their ratios are used.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseStats {
    lag: usize,
    /// Measured luma samples of the frame (0 for a blend of frames that had
    /// some).
    luma_samples: f64,
    /// Luma `E[e(p) e(p + d)]` for each [`correlation_offsets`] `d`.
    luma: Vec<f64>,
    /// Per chroma plane: `E[c(p) c(p + d)]`, as `luma`.
    chroma: [Vec<f64>; 2],
    /// Per chroma plane: `E[l(p) c(p + a)]` for each [`ar_offsets`] `a`,
    /// then `a = 0`, where `l` is the 2x2 mean of the luma noise.
    cross: [Vec<f64>; 2],
    /// `E[l(p)²]` at chroma positions.
    luma_at_chroma: f64,
    /// Per plane and luma intensity bin: noise and loss of its blocks.
    bins: [[BinStats; INTENSITY_BINS]; 3],
}

/// The masked noise of one plane. Stored samples fit `i16` (see
/// [`NOISE_CLAMP`]), so rows correlate in vectorisable 16-bit lanes.
struct NoisePlane {
    width: usize,
    height: usize,
    values: Vec<i16>,
    /// Rows holding any measured block (all others are zero).
    active: Vec<bool>,
}

impl NoisePlane {
    fn row(&self, y: usize) -> &[i16] {
        &self.values[y * self.width..(y + 1) * self.width]
    }

    /// `Σ e(p) f(p + (dr, dc))` over every `p` with `p + (dr, dc)` inside.
    fn correlate(&self, other: &Self, dr: usize, dc: isize) -> f64 {
        let shift = dc.unsigned_abs();
        if shift >= self.width || dr >= self.height {
            return 0.0;
        }
        let span = self.width - shift;
        let mut total = 0i64;
        for y in (0..self.height - dr).filter(|&y| self.active[y] && other.active[y + dr]) {
            let (a, b) = (self.row(y), other.row(y + dr));
            let (a, b) = if dc >= 0 {
                (&a[..span], &b[shift..])
            } else {
                (&a[shift..], &b[..span])
            };
            total += i64::from(dot(a, b));
        }
        total as f64
    }
}

/// Lanes of [`dot`]'s accumulator: a multiple of every SIMD width the
/// compiler targets, so the lane loop vectorises to multiply-add.
const DOT_LANES: usize = 16;

/// `Σ a[i] b[i]` of two equal-length rows (the sum fits `i32`, see
/// [`NOISE_CLAMP`]).
fn dot(a: &[i16], b: &[i16]) -> i32 {
    let mut lanes = [0i32; DOT_LANES];
    let ((a_chunks, a_tail), (b_chunks, b_tail)) =
        (a.as_chunks::<DOT_LANES>(), b.as_chunks::<DOT_LANES>());
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for ((lane, &x), &y) in lanes.iter_mut().zip(x).zip(y) {
            *lane += i32::from(x) * i32::from(y);
        }
    }
    let tail: i32 = a_tail
        .iter()
        .zip(b_tail)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum();
    lanes.iter().sum::<i32>() + tail
}

/// The visible-plane view of `p`'s plane `index`.
fn plane(p: &Picture, index: PlaneIndex) -> PlaneView<'_> {
    p.plane(index).view()
}

/// Weights of the binomial blur `[1 2 1] / 4` along each axis; the 3x3
/// kernel (their outer product) sums to [`BLUR_SUM`].
const BLUR_TAPS: [i32; 3] = [1, 2, 1];
const BLUR_SUM: i32 = 16;

/// The blurred part and the noise (what the blur removed) of a square block,
/// row-major, in levels.
struct BlockSplit {
    side: usize,
    blurred: Vec<f64>,
    noise: Vec<f64>,
}

impl BlockSplit {
    /// The split of `plane`'s `side` x `side` block at `(x0, y0)`; the blur
    /// reads one neighbour beyond the block, clamped to the plane.
    fn new(plane: &PlaneView<'_>, (x0, y0): (usize, usize), side: usize) -> Self {
        let (width, height) = (plane.width(), plane.height());
        let span = side + 2;
        let mut window = Vec::with_capacity(span * span);
        let interior = x0 >= 1 && x0 + side < width;
        for dy in 0..span {
            let row = plane.row((y0 + dy).saturating_sub(1).min(height - 1));
            if interior {
                window.extend(row[x0 - 1..=x0 + side].iter().map(|&v| i32::from(v)));
            } else {
                let clamped = |dx: usize| row[(x0 + dx).saturating_sub(1).min(width - 1)];
                window.extend((0..span).map(|dx| i32::from(clamped(dx))));
            }
        }
        // Separable integer passes: rows, then columns.
        let tap = |t: &[i32]| BLUR_TAPS[0] * t[0] + BLUR_TAPS[1] * t[1] + BLUR_TAPS[2] * t[2];
        let across: Vec<i32> = window
            .chunks_exact(span)
            .flat_map(|row| row.windows(3).map(tap))
            .collect();
        let unit = f64::from(BLUR_SUM);
        let mut blurred = Vec::with_capacity(side * side);
        let mut noise = Vec::with_capacity(side * side);
        for (y, rows) in across.windows(3 * side).step_by(side).enumerate() {
            let (above, rest) = rows.split_at(side);
            let (middle, below) = rest.split_at(side);
            let centre = &window[(y + 1) * span + 1..][..side];
            for (((&sample, &a), &m), &b) in centre.iter().zip(above).zip(middle).zip(below) {
                let sum = tap(&[a, m, b]);
                blurred.push(f64::from(sum) / unit);
                noise.push(f64::from(BLUR_SUM * sample - sum) / unit);
            }
        }
        Self {
            side,
            blurred,
            noise,
        }
    }

    /// Whether the block is flat under `settings`: its blurred part, less
    /// its shading plane, is quiet, and what the blur removed is not
    /// heavy-tailed. Thin lines (twigs, wires against the sky) pass the
    /// blur as "noise", but as a few large values among many small ones;
    /// noise is near Gaussian.
    fn is_flat(&self, settings: &MeasureSettings) -> bool {
        let structure = shading_free_variance(&self.blurred, self.side).sqrt();
        structure <= settings.flat_structure + BLURRED_NOISE_RATIO * rms(&self.noise)
            && kurtosis(&self.noise) <= settings.max_kurtosis
    }
}

/// Below this variance (levels²) a block has no tails to measure: it is
/// noiseless or clipped.
const NOISELESS_VARIANCE: f64 = 1e-3;

/// `E[(v − mean)⁴] / variance²` of `values`: 3 for Gaussian noise, larger
/// when a few samples carry most of the energy.
fn kurtosis(values: &[f64]) -> f64 {
    let count = values.len().max(1) as f64;
    let mean = values.iter().sum::<f64>() / count;
    let (mut second, mut fourth) = (0.0, 0.0);
    for v in values {
        let d2 = (v - mean).powi(2);
        second += d2;
        fourth += d2 * d2;
    }
    let variance = second / count;
    if variance < NOISELESS_VARIANCE {
        return 0.0;
    }
    fourth / count / (variance * variance)
}

/// How much structure pure noise leaves after the blur, relative to the
/// noise the blur removes: white noise keeps 0.375 of its deviation through
/// the 3x3 binomial blur and loses 0.8 to it (ratio 0.47), correlated
/// noise somewhat more. The flatness test allows this much so that loud
/// noise is still measured.
const BLURRED_NOISE_RATIO: f64 = 0.5;

/// Root mean square of `values` about their mean.
fn rms(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

/// Variance of `values` about their mean.
fn variance(values: &[f64]) -> f64 {
    let count = values.len().max(1) as f64;
    let mean = values.iter().sum::<f64>() / count;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count
}

/// The variance of the `side` x `side` block `values` (row-major) about its
/// least-squares plane `a + b x + c y`: what is left once the block's
/// shading is removed. Centred coordinates are orthogonal, so the plane's
/// two slopes take `(Σ x v)² / Σ x²` and `(Σ y v)² / Σ y²` out of the
/// variance in closed form (`Σ (x − centre)²` over the block is
/// `n² (n² − 1) / 12`).
fn shading_free_variance(values: &[f64], side: usize) -> f64 {
    let centre = (side as f64 - 1.0) / 2.0;
    let (mut sum, mut squares, mut sum_x, mut sum_y) = (0.0, 0.0, 0.0, 0.0);
    for (y, row) in values.chunks_exact(side).enumerate() {
        let dy = y as f64 - centre;
        for (x, &v) in row.iter().enumerate() {
            sum += v;
            squares += v * v;
            sum_x += (x as f64 - centre) * v;
            sum_y += dy * v;
        }
    }
    let area = (side * side) as f64;
    let spread = area * (area - 1.0) / 12.0;
    let centred = squares - sum * sum / area;
    ((centred - (sum_x * sum_x + sum_y * sum_y) / spread) / area).max(0.0)
}

/// Which luma blocks are measured, and each block's mean luma intensity bin.
struct BlockMask {
    columns: usize,
    rows: usize,
    /// Per block: the intensity bin when measured.
    selected: Vec<Option<usize>>,
}

impl BlockMask {
    /// The flat blocks of `source` (its luma plane).
    fn new(
        source: &PlaneView<'_>,
        (width, height): (usize, usize),
        settings: &MeasureSettings,
    ) -> Self {
        let (columns, rows) = (width / BLOCK, height / BLOCK);
        let selected = (0..rows * columns)
            .map(|i| {
                let origin = ((i % columns) * BLOCK, (i / columns) * BLOCK);
                let block = source.sub_view(origin.0, origin.1, BLOCK, BLOCK)?;
                if !BlockSplit::new(source, origin, BLOCK).is_flat(settings) {
                    return None;
                }
                let sum: u32 = block.rows().flatten().map(|&v| u32::from(v)).sum();
                let mean = sum as usize / (BLOCK * BLOCK);
                Some(mean / BIN_WIDTH)
            })
            .collect();
        Self {
            columns,
            rows,
            selected,
        }
    }
}

/// The noise of `plane` on the masked blocks (`block` the plane's block
/// side): what the blur removes from the source, block mean removed, zero
/// elsewhere. Each masked block's noise variance, and the variance of the
/// noise coding removed (the source's noise less the coded picture's), go
/// into `bins`.
fn noise_plane(
    source: &PlaneView<'_>,
    coded: &PlaneView<'_>,
    mask: &BlockMask,
    block: usize,
    bins: &mut [BinSamples; INTENSITY_BINS],
) -> NoisePlane {
    let (width, height) = (mask.columns * block, mask.rows * block);
    let mut values = vec![0i16; width * height];
    let mut active = vec![false; height];
    let unit = f64::from(1u32 << NOISE_FRACTION_BITS);
    for (i, bin) in mask.selected.iter().enumerate() {
        let Some(bin) = *bin else { continue };
        let origin = ((i % mask.columns) * block, (i / mask.columns) * block);
        active[origin.1..origin.1 + block].fill(true);
        let noise = BlockSplit::new(source, origin, block).noise;
        let mean = noise.iter().sum::<f64>() / noise.len() as f64;
        for (y, rows) in noise.chunks_exact(block).enumerate() {
            let out = &mut values[(origin.1 + y) * width + origin.0..][..block];
            for (o, &n) in out.iter_mut().zip(rows) {
                *o = ((n - mean) * unit).round().clamp(-NOISE_CLAMP, NOISE_CLAMP) as i16;
            }
        }
        let kept = BlockSplit::new(coded, origin, block).noise;
        let removed: Vec<f64> = noise.iter().zip(&kept).map(|(n, k)| n - k).collect();
        bins[bin].add(variance(&noise), variance(&removed));
    }
    NoisePlane {
        width,
        height,
        values,
        active,
    }
}

/// The luma noise averaged over each 2x2, at chroma resolution.
fn downsampled(luma: &NoisePlane) -> NoisePlane {
    let (width, height) = (luma.width / 2, luma.height / 2);
    let mut values = vec![0i16; width * height];
    let active: Vec<bool> = (0..height).map(|y| luma.active[2 * y]).collect();
    for y in (0..height).filter(|&y| active[y]) {
        let (top, bottom) = (luma.row(2 * y), luma.row(2 * y + 1));
        for (x, out) in values[y * width..(y + 1) * width].iter_mut().enumerate() {
            let sum: i32 = [top[2 * x], top[2 * x + 1], bottom[2 * x], bottom[2 * x + 1]]
                .into_iter()
                .map(i32::from)
                .sum();
            *out = ((sum + 2) >> 2) as i16;
        }
    }
    NoisePlane {
        width,
        height,
        values,
        active,
    }
}

impl NoiseStats {
    /// Measure the noise of `source`, and how much of it `coded` (its
    /// reconstruction) lost, on the same blocks, over the visible `width` x
    /// `height` (4:2:0).
    #[must_use]
    pub fn measure(
        source: &Picture,
        coded: &Picture,
        size: (usize, usize),
        settings: &MeasureSettings,
    ) -> Self {
        let lag = settings.lag.min(MAX_LAG);
        let mask = BlockMask::new(&plane(source, PlaneIndex::Y), size, settings);
        let mut samples: [[BinSamples; INTENSITY_BINS]; 3] = Default::default();
        let luma = noise_plane(
            &plane(source, PlaneIndex::Y),
            &plane(coded, PlaneIndex::Y),
            &mask,
            BLOCK,
            &mut samples[0],
        );
        let luma_samples = mask.selected.iter().flatten().count() as f64 * (BLOCK * BLOCK) as f64;
        let offsets = correlation_offsets(lag);
        let autocorrelation = |noise: &NoisePlane, samples: f64| -> Vec<f64> {
            offsets
                .iter()
                .map(|&(dr, dc)| noise.correlate(noise, dr, dc) / samples.max(1.0))
                .collect()
        };
        let luma_corr = autocorrelation(&luma, luma_samples);
        let chroma_samples = luma_samples / 4.0;
        let mut chroma = [0, 1].map(|_| vec![0.0; offsets.len()]);
        let mut cross = [0, 1].map(|_| vec![0.0; ar_offsets(lag).count() + 1]);
        let mut luma_at_chroma = 0.0;
        if settings.chroma {
            let luma_half = downsampled(&luma);
            for (c, index) in [PlaneIndex::U, PlaneIndex::V].into_iter().enumerate() {
                let noise = noise_plane(
                    &plane(source, index),
                    &plane(coded, index),
                    &mask,
                    BLOCK / 2,
                    &mut samples[c + 1],
                );
                chroma[c] = autocorrelation(&noise, chroma_samples);
                // `l(p) c(p + a)` with `a` above or left: correlate `c` at
                // `p + a` against `l` at `p`, i.e. `c(q) l(q − a)`.
                cross[c] = ar_offsets(lag)
                    .chain(std::iter::once((0, 0)))
                    .map(|(dr, dc)| {
                        noise.correlate(&luma_half, dr.unsigned_abs(), -dc)
                            / chroma_samples.max(1.0)
                    })
                    .collect();
            }
            luma_at_chroma = luma_half.correlate(&luma_half, 0, 0) / chroma_samples.max(1.0);
        }
        let bins = samples.map(|plane| plane.map(BinSamples::stats));
        Self {
            lag,
            luma_samples,
            luma: luma_corr,
            chroma,
            cross,
            luma_at_chroma,
            bins,
        }
    }

    /// Fold `newer` into these statistics with weight `weight` (0..=1);
    /// a frame without measured samples leaves them unchanged.
    pub fn blend(&mut self, newer: &Self, weight: f64) {
        if newer.luma_samples < MIN_SAMPLES || newer.lag != self.lag {
            return;
        }
        let mix = |old: &mut f64, new: f64| *old += weight * (new - *old);
        let mix_all = |old: &mut [f64], new: &[f64]| {
            for (o, &n) in old.iter_mut().zip(new) {
                mix(o, n);
            }
        };
        mix_all(&mut self.luma, &newer.luma);
        for c in 0..2 {
            mix_all(&mut self.chroma[c], &newer.chroma[c]);
            mix_all(&mut self.cross[c], &newer.cross[c]);
        }
        mix(&mut self.luma_at_chroma, newer.luma_at_chroma);
        for (old, new) in self.bins.iter_mut().zip(&newer.bins) {
            for (o, n) in old.iter_mut().zip(new) {
                // Bins blend by block count, so a bin this frame did not
                // measure keeps its strength.
                let total = o.blocks + weight * n.blocks;
                if total > 0.0 {
                    o.noise = (o.noise * o.blocks + weight * n.noise * n.blocks) / total;
                    o.loss = (o.loss * o.blocks + weight * n.loss * n.blocks) / total;
                }
                o.blocks = (1.0 - weight) * o.blocks + weight * n.blocks;
            }
        }
        self.luma_samples = self.luma_samples.max(newer.luma_samples);
    }

    /// Whether these statistics have enough measured samples for a model.
    #[must_use]
    pub fn is_measured(&self) -> bool {
        self.luma_samples >= MIN_SAMPLES
    }

    /// The autocorrelation `r(d)` of `corr` at any offset (`r(−d) = r(d)`).
    fn at(&self, corr: &[f64], (dr, dc): (isize, isize)) -> f64 {
        let (dr, dc) = if dr < 0 || (dr == 0 && dc < 0) {
            (-dr, -dc)
        } else {
            (dr, dc)
        };
        // Row-major over rows 0..=lag and columns -2 lag..=2 lag, minus
        // row 0's 2 lag negative columns (`correlation_offsets`).
        let row = 4 * self.lag as isize + 1;
        corr[(dr * row + dc) as usize]
    }

    /// The fitted model, or `None` without enough measured noise.
    #[must_use]
    pub fn model(&self) -> Option<NoiseModel> {
        if !self.is_measured() || self.luma[0] <= 0.0 {
            return None;
        }
        let offsets: Vec<(isize, isize)> = ar_offsets(self.lag).collect();
        let n = offsets.len();
        let yule_walker = |corr: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let mut matrix = vec![0.0; n * n];
            for (i, &a) in offsets.iter().enumerate() {
                for (j, &b) in offsets.iter().enumerate() {
                    matrix[i * n + j] = self.at(corr, (b.0 - a.0, b.1 - a.1));
                }
            }
            let rhs = offsets.iter().map(|&a| self.at(corr, a)).collect();
            (matrix, rhs)
        };
        let (matrix, rhs) = yule_walker(&self.luma);
        let luma_ar = solve(matrix, rhs, n)?;
        let chroma_ar = [0, 1].map(|c| {
            // The chroma system adds the luma term as a last unknown.
            let (square, rhs) = yule_walker(&self.chroma[c]);
            let m = n + 1;
            let mut matrix = vec![0.0; m * m];
            for i in 0..n {
                matrix[i * m..i * m + n].copy_from_slice(&square[i * n..(i + 1) * n]);
                matrix[i * m + n] = self.cross[c][i];
                matrix[n * m + i] = self.cross[c][i];
            }
            matrix[n * m + n] = self.luma_at_chroma;
            let mut rhs = rhs;
            rhs.push(self.cross[c][n]);
            if self.chroma[c][0] > 0.0 {
                solve(matrix, rhs, m).unwrap_or_else(|| vec![0.0; m])
            } else {
                vec![0.0; m]
            }
        });
        let deviations = self.bins.map(|plane| plane.map(|bin| bin.deviations()));
        Some(NoiseModel {
            lag: self.lag,
            luma_ar,
            chroma_ar,
            noise: deviations.map(|plane| plane.map(|d| d.0)),
            loss: deviations.map(|plane| plane.map(|d| d.1)),
        })
    }
}

/// Solve the symmetric `n` x `n` system `matrix · x = rhs` (Gaussian
/// elimination with partial pivoting, after a small ridge on the diagonal);
/// `None` when it is singular.
fn solve(mut matrix: Vec<f64>, mut rhs: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    let scale = (0..n).map(|i| matrix[i * n + i]).fold(0.0, f64::max);
    if scale <= 0.0 {
        return None;
    }
    for i in 0..n {
        matrix[i * n + i] += RIDGE * scale;
    }
    for col in 0..n {
        let pivot = (col..n).max_by(|&a, &b| {
            matrix[a * n + col]
                .abs()
                .total_cmp(&matrix[b * n + col].abs())
        })?;
        if matrix[pivot * n + col].abs() < f64::EPSILON * scale {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                matrix.swap(col * n + k, pivot * n + k);
            }
            rhs.swap(col, pivot);
        }
        for row in col + 1..n {
            let factor = matrix[row * n + col] / matrix[col * n + col];
            for k in col..n {
                matrix[row * n + k] -= factor * matrix[col * n + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n).map(|k| matrix[row * n + k] * x[k]).sum();
        x[row] = (rhs[row] - tail) / matrix[row * n + row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2-D AR(1)-like field: each sample is white noise plus 0.5 of its
    /// left neighbour.
    fn correlated_noise(width: usize, height: usize) -> Vec<i16> {
        let mut state = 12345u32;
        let mut white = || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            ((state >> 16) % 21) as i16 - 10
        };
        let mut values = vec![0i16; width * height];
        for y in 0..height {
            for x in 0..width {
                let left = if x > 0 { values[y * width + x - 1] } else { 0 };
                values[y * width + x] = white() + left / 2;
            }
        }
        values
    }

    const SETTINGS: MeasureSettings = MeasureSettings {
        lag: 2,
        flat_structure: 2.0,
        max_kurtosis: 5.0,
        chroma: false,
    };

    /// A picture whose luma is `luma(x, y)` and chroma mid-grey.
    fn picture(width: usize, height: usize, luma: impl Fn(usize, usize) -> u8) -> Picture {
        let mut p = Picture::new(width, height, 16).expect("picture");
        for plane in PlaneIndex::ALL {
            let mut v = p.plane_mut(plane).view_mut();
            for y in 0..v.height() {
                for (x, s) in v.row_mut(y).iter_mut().enumerate() {
                    *s = if plane == PlaneIndex::Y {
                        luma(x, y)
                    } else {
                        128
                    };
                }
            }
        }
        p
    }

    /// Mid-grey (bin 3) plus near-Gaussian noise of deviation 2.83 levels
    /// (a sum of four uniforms over -2..=2, variance 2 each).
    fn noisy_grey(width: usize, height: usize) -> Picture {
        let mut noise = vec![0u8; width * height];
        let mut state = 99u32;
        for n in &mut noise {
            let mut sum = 0i32;
            for _ in 0..4 {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
                sum += ((state >> 16) % 5) as i32 - 2;
            }
            *n = (100 + sum) as u8;
        }
        picture(width, height, |x, y| noise[y * width + x])
    }

    #[test]
    fn source_noise_does_not_depend_on_the_coded_picture() {
        let (width, height) = (128, 64);
        let source = noisy_grey(width, height);
        let smooth = picture(width, height, |_, _| 100);
        let bin = 100 / BIN_WIDTH;
        let measure = |coded: &Picture| {
            NoiseStats::measure(&source, coded, (width, height), &SETTINGS)
                .model()
                .expect("flat noisy blocks are measured")
        };
        let (lossy, lossless) = (measure(&smooth), measure(&source));
        let noise = lossy.noise[0][bin].expect("bin measured");
        assert_eq!(lossless.noise[0][bin], Some(noise));
        // The blur removes 0.8 of white noise's deviation (2.26 here).
        assert!((2.0..3.0).contains(&noise), "noise {noise}");
        // Coding that smoothed everything removed all of it; coding that
        // kept the source removed none.
        assert!((lossy.loss[0][bin].expect("loss") - noise).abs() < 1e-9);
        assert_eq!(lossless.loss[0][bin], Some(0.0));
        assert_eq!(lossless.grain_strength(1.0)[0][bin], Some(0.0));
    }

    #[test]
    fn shading_planes_carry_no_variance() {
        let side = 8;
        let ramp: Vec<f64> = (0..side * side)
            .map(|i| 3.0 + 0.5 * (i % side) as f64 - 1.5 * (i / side) as f64)
            .collect();
        assert!(shading_free_variance(&ramp, side) < 1e-9);
        // A ±1 checkerboard is orthogonal to every plane: its variance stays.
        let checker: Vec<f64> = ramp
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v + if (i % side + i / side) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        assert!((shading_free_variance(&checker, side) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn edges_and_thin_lines_are_not_noise() {
        let (width, height) = (128, 64);
        let ramp_edges = picture(width, height, |x, _| if x % 16 < 8 { 60 } else { 140 });
        let twigs = picture(
            width,
            height,
            |x, y| if (x + y) % 16 == 0 { 40 } else { 200 },
        );
        for source in [ramp_edges, twigs] {
            let stats = NoiseStats::measure(&source, &source, (width, height), &SETTINGS);
            assert!(!stats.is_measured());
        }
    }

    #[test]
    fn correlation_offset_index_matches_lookup() {
        for lag in 0..=MAX_LAG {
            let stats = NoiseStats {
                lag,
                luma_samples: 0.0,
                luma: (0..correlation_offsets(lag).len())
                    .map(|i| i as f64)
                    .collect(),
                chroma: [Vec::new(), Vec::new()],
                cross: [Vec::new(), Vec::new()],
                luma_at_chroma: 0.0,
                bins: [[BinStats::default(); INTENSITY_BINS]; 3],
            };
            for (i, &(dr, dc)) in correlation_offsets(lag).iter().enumerate() {
                assert_eq!(stats.at(&stats.luma, (dr as isize, dc)), i as f64);
                assert_eq!(stats.at(&stats.luma, (-(dr as isize), -dc)), i as f64);
            }
        }
    }

    #[test]
    fn fits_a_left_neighbour_filter() {
        let (width, height) = (256, 128);
        let noise = NoisePlane {
            width,
            height,
            values: correlated_noise(width, height),
            active: vec![true; height],
        };
        let lag = 1;
        let samples = (width * height) as f64;
        let luma = correlation_offsets(lag)
            .iter()
            .map(|&(dr, dc)| noise.correlate(&noise, dr, dc) / samples)
            .collect();
        let stats = NoiseStats {
            lag,
            luma_samples: samples,
            luma,
            chroma: [0, 1].map(|_| vec![0.0; correlation_offsets(lag).len()]),
            cross: [vec![0.0; 5], vec![0.0; 5]],
            luma_at_chroma: 0.0,
            bins: [[BinStats::default(); INTENSITY_BINS]; 3],
        };
        let model = stats.model().expect("measured");
        // Offsets (-1,-1), (-1,0), (-1,1), (0,-1): only the left one is 0.5.
        let left = model.luma_ar[3];
        assert!((left - 0.5).abs() < 0.05, "left coefficient {left}");
        for &c in &model.luma_ar[..3] {
            assert!(c.abs() < 0.05, "coefficient {c}");
        }
    }
}
