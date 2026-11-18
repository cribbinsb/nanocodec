//! AV1 film grain synthesis (AV1 §7.18.3), 4:2:0, 8-bit with the ISA's
//! row kernels and deeper samples with the scalar ones
//! ([`crate::Dsp::av1_add_film_grain_at`]): pseudo-random
//! grain templates shaped by an auto-regressive filter, laid over the
//! output frame in 32x32 luma blocks at random offsets (with optional
//! overlap blending), scaled by a piecewise-linear function of the
//! underlying samples. Applied to the shown frame only, never to
//! references.
//!
//! AV2 keeps the templates, scaling and blending and changes only how the
//! blocks are laid ([`GrainBlocks`], [`crate::Dsp::av2_add_film_grain`]):
//! 16x16 or 32x32 blocks, each placed by two scaled 9-bit random offsets,
//! and stripe seeds counted in 16-row units (dav2d 237229e
//! `src/filmgrain_tmpl.c` `fgy_32x32xn_c`, `fguv_32x32xn_c`; BSD-2-Clause,
//! see `THIRD_PARTY.md`).
//!
//! Written from the spec's structure (§7.18.3.3–5: grain generation,
//! scaling lookup, noise stripes, noise image, blending); the arithmetic
//! is checked against libaom v3.15.3 `av1/decoder/grain_synthesis.c`
//! (`av1_add_film_grain`; BSD-2-Clause, see `THIRD_PARTY.md`).

mod gaussian;
pub(crate) mod high_simd;
pub(crate) mod kernels;
mod synthesis;

pub use kernels::Av1GrainScaling;

use nc_common::{BitDepth, PlaneView, PlaneViewMut};

use gaussian::GAUSSIAN_SEQUENCE;

/// The `film_grain_params()` syntax values (AV1 §5.9.30) synthesis needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools, reason = "the specification's flags")]
pub struct Av1FilmGrainParams {
    /// `grain_seed`.
    pub grain_seed: u16,
    /// `point_y_value` / `point_y_scaling`, `num_y_points` of them.
    pub y_points: Av1ScalingPoints<14>,
    /// `chroma_scaling_from_luma`.
    pub chroma_scaling_from_luma: bool,
    /// `point_cb_value` / `point_cb_scaling` (at most 10 in AV1, 14 in AV2).
    pub cb_points: Av1ScalingPoints<14>,
    /// `point_cr_value` / `point_cr_scaling` (as `cb_points`).
    pub cr_points: Av1ScalingPoints<14>,
    /// `grain_scaling_minus_8`.
    pub grain_scaling_minus_8: u8,
    /// `ar_coeff_lag` (0 to 3).
    pub ar_coeff_lag: u8,
    /// `ar_coeffs_y_plus_128`.
    pub ar_coeffs_y_plus_128: [u8; 24],
    /// `ar_coeffs_cb_plus_128`.
    pub ar_coeffs_cb_plus_128: [u8; 25],
    /// `ar_coeffs_cr_plus_128`.
    pub ar_coeffs_cr_plus_128: [u8; 25],
    /// `ar_coeff_shift_minus_6`.
    pub ar_coeff_shift_minus_6: u8,
    /// `grain_scale_shift`.
    pub grain_scale_shift: u8,
    /// `cb_mult`, `cb_luma_mult`, `cb_offset`.
    pub cb_mult: (u8, u8, u16),
    /// `cr_mult`, `cr_luma_mult`, `cr_offset`.
    pub cr_mult: (u8, u8, u16),
    /// `overlap_flag`.
    pub overlap: bool,
    /// `clip_to_restricted_range`.
    pub clip_to_restricted_range: bool,
    /// The identity colour matrix: restricted-range chroma clips to luma's
    /// 235 rather than 240 (AV1: `matrix_coefficients == MC_IDENTITY`, which
    /// needs 4:4:4, so never in 4:2:0; AV2: the FGM's `mc_identity`).
    pub mc_identity: bool,
}

/// Up to `N` scaling points: (`value`, `scaling`) pairs in increasing value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1ScalingPoints<const N: usize> {
    /// Point count (`num_*_points`).
    pub count: usize,
    /// `point_*_value`.
    pub value: [u8; N],
    /// `point_*_scaling`.
    pub scaling: [u8; N],
}

impl<const N: usize> Default for Av1ScalingPoints<N> {
    fn default() -> Self {
        Self {
            count: 0,
            value: [0; N],
            scaling: [0; N],
        }
    }
}

/// Bits of the random numbers that index [`GAUSSIAN_SEQUENCE`].
const GAUSSIAN_BITS: u32 = 11;
/// Precision of [`GAUSSIAN_SEQUENCE`] samples.
const GAUSSIAN_PRECISION: u32 = 12;
/// Smallest and largest 8-bit grain value (`GrainMin`, `GrainMax`; both
/// scale by `1 << (BitDepth − 8)`, [`grain_range`]).
const GRAIN_MIN: i32 = -128;
const GRAIN_MAX: i32 = 127;
/// Luma grain template size (`73` rows of `82`).
const LUMA_TEMPLATE: (usize, usize) = (73, 82);
/// 4:2:0 chroma grain template size (`38` rows of `44`).
const CHROMA_TEMPLATE: (usize, usize) = (38, 44);
/// Template rows and columns before the auto-regressive filter starts
/// (its largest lag).
const AR_PADDING: usize = 3;
/// Seeds of the chroma templates (`grain_seed ^ 0xb524`, `^ 0x49d8`).
const CB_SEED_XOR: u16 = 0xb524;
const CR_SEED_XOR: u16 = 0x49d8;
/// Luma rows per noise stripe and side of AV1's grain blocks.
const BLOCK: usize = 32;
/// Side of AV2's small grain blocks (`block_size = 0`); its large ones
/// are [`BLOCK`].
const AV2_SMALL_BLOCK: usize = 16;
/// Bits of AV2's random block offsets, and of the draws discarded after
/// each (dav2d `fgy_32x32xn_c`).
const AV2_OFFSET_BITS: u32 = 9;
const AV2_DISCARD_BITS: u32 = 16;
/// Random numbers AV2 discards after each block offset.
const AV2_DISCARDS: usize = 3;
/// AV2's offset scale: `(3 − block_size) * random >> 6` spans 0..=23 in
/// 16x16 blocks and 0..=15 in 32x32 ones.
const AV2_OFFSET_SCALE: usize = 3;
const AV2_OFFSET_SHIFT: u32 = 6;
/// Luma rows and columns a block's grain reaches past the block, for
/// overlap with the next one.
const OVERLAP: usize = 2;
/// Restricted (studio) range bounds, 8-bit.
const RESTRICTED_MIN: i32 = 16;
const RESTRICTED_MAX_LUMA: i32 = 235;
const RESTRICTED_MAX_CHROMA: i32 = 240;

/// The film grain shift register (`RandomRegister`, AV1 §7.18.3.2).
struct Random(u16);

impl Random {
    /// `get_random_number(bits)`: a 16-bit LFSR with taps 0, 1, 3, 12.
    fn next(&mut self, bits: u32) -> usize {
        let r = self.0;
        let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
        self.0 = (r >> 1) | (bit << 15);
        usize::from(self.0 >> (16 - bits)) & ((1 << bits) - 1)
    }
}

/// `Round2` of a signed value.
fn round2(value: i32, bits: u32) -> i32 {
    if bits == 0 {
        value
    } else {
        (value + (1 << (bits - 1))) >> bits
    }
}

/// How the grain blocks are laid over the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrainBlocks {
    /// AV1 §7.18.3.5: 32x32 blocks, each placed by one 8-bit random number
    /// (4 bits per axis).
    Av1,
    /// AV2 (dav2d `fgy_32x32xn_c`): 32x32 blocks when `large` (FGM
    /// `block_size = 1`), else 16x16, each placed by two 9-bit random
    /// numbers, row then column.
    Av2 {
        /// `block_size`.
        large: bool,
    },
}

impl GrainBlocks {
    /// Side of a luma block, and the luma rows of a noise stripe.
    fn luma_side(self) -> usize {
        match self {
            Self::Av2 { large: false } => AV2_SMALL_BLOCK,
            Self::Av1 | Self::Av2 { large: true } => BLOCK,
        }
    }

    /// `RandomRegister` of stripe `index`: AV2 numbers stripes in 16-row
    /// units (`rel_row_num = row_num << block_size`).
    fn stripe_seed(self, seed: u16, index: usize) -> u16 {
        let unit = match self {
            Self::Av1 | Self::Av2 { large: false } => index,
            Self::Av2 { large: true } => index << 1,
        };
        stripe_seed(seed, unit)
    }

    /// The next block's (column, row) offset into the template, in units
    /// of two luma samples.
    fn offsets(self, random: &mut Random) -> (usize, usize) {
        match self {
            Self::Av1 => {
                let offset = random.next(8);
                (offset >> 4, offset & 15)
            }
            Self::Av2 { large } => {
                let scale = AV2_OFFSET_SCALE - usize::from(large);
                let mut next = || {
                    let offset = (scale * random.next(AV2_OFFSET_BITS)) >> AV2_OFFSET_SHIFT;
                    for _ in 0..AV2_DISCARDS {
                        random.next(AV2_DISCARD_BITS);
                    }
                    offset
                };
                let row = next();
                (next(), row)
            }
        }
    }
}

/// `GrainMin..=GrainMax` at `depth`.
fn grain_range(depth: BitDepth) -> (i32, i32) {
    (
        GRAIN_MIN << depth.above_eight(),
        ((GRAIN_MAX + 1) << depth.above_eight()) - 1,
    )
}

/// A stored grain value: `i8` at 8 bits (dav1d's `entry` at 8bpc), `i16`
/// deeper (`GrainMin..=GrainMax` grows with the depth: ±512 at 10 bits).
pub(crate) trait Grain: Copy + Default + Into<i32> {
    /// `value`, within the depth's grain range.
    fn from_i32(value: i32) -> Self;
}

impl Grain for i8 {
    fn from_i32(value: i32) -> Self {
        value as i8
    }
}

impl Grain for i16 {
    fn from_i32(value: i32) -> Self {
        value as i16
    }
}

/// Largest template area (`LUMA_TEMPLATE`: `73 * 82 = 5986` samples),
/// stack-allocated so synthesis allocates no per-frame template or AR vectors.
const MAX_TEMPLATE_SAMPLES: usize = LUMA_TEMPLATE.0 * LUMA_TEMPLATE.1;
const MAX_AR_ABOVE: usize = 24;
const MAX_AR_LEFT: usize = AR_PADDING;
const MAX_SHAPED_COLS: usize = LUMA_TEMPLATE.1 - 2 * AR_PADDING;
const CHROMA_SHAPED_ROWS: usize = CHROMA_TEMPLATE.0 - AR_PADDING;
const CHROMA_SHAPED_COLS: usize = CHROMA_TEMPLATE.1 - 2 * AR_PADDING;

/// A grain template: `rows` x `columns` grain values. The white noise is
/// within `round2(±1752, 12 − BitDepth)` (the extremes of
/// [`GAUSSIAN_SEQUENCE`] at the smallest shift) and the auto-regressive
/// filter clamps to `GrainMin..=GrainMax`, so `G` of the depth is exact.
struct Template<G> {
    rows: usize,
    columns: usize,
    values: [G; MAX_TEMPLATE_SAMPLES],
}

/// White Gaussian grain (AV1 §7.18.3.3), or zeros when the plane has no
/// grain (no random numbers drawn then).
fn white_noise<G: Grain>(
    random: &mut Random,
    (rows, columns): (usize, usize),
    shift: u32,
    enabled: bool,
) -> Template<G> {
    let mut values = [G::default(); MAX_TEMPLATE_SAMPLES];
    if enabled {
        let round = if shift == 0 { 0 } else { 1 << (shift - 1) };
        for out in &mut values[..rows * columns] {
            let g = i32::from(GAUSSIAN_SEQUENCE[random.next(GAUSSIAN_BITS)]);
            *out = G::from_i32((g + round) >> shift);
        }
    }
    Template {
        rows,
        columns,
        values,
    }
}

/// The auto-regressive neighbourhood of lag `lag`: every (row, column)
/// offset above and left of the sample, in coefficient order.
fn ar_positions(lag: usize) -> impl Iterator<Item = (isize, isize)> {
    let lag = lag as isize;
    (-lag..=0)
        .flat_map(move |dr| (-lag..=lag).map(move |dc| (dr, dc)))
        .take_while(|&(dr, dc)| !(dr == 0 && dc == 0))
}

/// The auto-regressive filter (AV1 §7.18.3.3) over rows `AR_PADDING..` of
/// `template` with `coefficients` (lag `lag`, shift `shift`), plus `extra(y,
/// sums)`'s term per sample (chroma's co-located luma grain). Row by row:
/// the rows above contribute to every sample of a row at once (one pass
/// over the row per tap, which vectorises; the integer sum's order does
/// not matter), then the taps to the left, which depend on the samples just
/// filtered, run along the row in order. Zero coefficients are skipped.
fn auto_regress<G: Grain>(
    template: &mut Template<G>,
    lag: usize,
    coefficients: &[u8],
    (shift, (min, max)): (u32, (i32, i32)),
    extra: impl Fn(usize, &mut [i32]),
) {
    let (rows, columns) = (template.rows, template.columns);
    let mut above = [(0isize, 0isize, 0i32); MAX_AR_ABOVE];
    let mut above_len = 0;
    let mut left = [(0isize, 0i32); MAX_AR_LEFT];
    let mut left_len = 0;
    for ((row, column), &value) in ar_positions(lag).zip(coefficients) {
        let coefficient = i32::from(value) - 128;
        if coefficient == 0 {
            continue;
        }
        if row < 0 {
            above[above_len] = (row, column, coefficient);
            above_len += 1;
        } else {
            left[left_len] = (column, coefficient);
            left_len += 1;
        }
    }
    let above = &above[..above_len];
    let left = &left[..left_len];
    let (first, last) = (AR_PADDING, columns - AR_PADDING);
    let shaped = last - first;
    let round = if shift == 0 { 0 } else { 1 << (shift - 1) };
    let mut sums_buf = [0i32; MAX_SHAPED_COLS];
    let sums = &mut sums_buf[..shaped];
    for y in AR_PADDING..rows {
        sums.fill(0);
        for &(row, column, coefficient) in above {
            let start = (y as isize + row) as usize * columns + (first as isize + column) as usize;
            let source = &template.values[start..start + shaped];
            for (sum, &value) in sums.iter_mut().zip(source) {
                *sum += coefficient * value.into();
            }
        }
        extra(y, sums);
        let row = &mut template.values[y * columns..(y + 1) * columns];
        for (x, &partial) in (first..last).zip(sums.iter()) {
            let mut sum = partial;
            for &(column, coefficient) in left {
                sum += coefficient * row[(x as isize + column) as usize].into();
            }
            row[x] = G::from_i32((row[x].into() + ((sum + round) >> shift)).clamp(min, max));
        }
    }
}

/// The 8-bit grain templates of the three planes (AV1 §7.18.3.3).
fn templates(p: &Av1FilmGrainParams) -> [Template<i8>; 3] {
    templates_at(p, BitDepth::EIGHT)
}

/// The grain templates of the three planes at `depth` (AV1 §7.18.3.3:
/// `shift = 12 − BitDepth + grain_scale_shift`).
fn templates_at<G: Grain>(p: &Av1FilmGrainParams, depth: BitDepth) -> [Template<G>; 3] {
    let shift = GAUSSIAN_PRECISION - depth.bits() + u32::from(p.grain_scale_shift);
    let ar = (u32::from(p.ar_coeff_shift_minus_6) + 6, grain_range(depth));
    let lag = usize::from(p.ar_coeff_lag);
    let mut luma = white_noise::<G>(
        &mut Random(p.grain_seed),
        LUMA_TEMPLATE,
        shift,
        p.y_points.count > 0,
    );
    if p.y_points.count > 0 {
        auto_regress(&mut luma, lag, &p.ar_coeffs_y_plus_128, ar, |_, _| {});
    }
    let cb_enabled = p.cb_points.count > 0 || p.chroma_scaling_from_luma;
    let cr_enabled = p.cr_points.count > 0 || p.chroma_scaling_from_luma;
    let with_luma = p.y_points.count > 0 && (cb_enabled || cr_enabled);
    let mut luma_sub = [[0i32; CHROMA_SHAPED_COLS]; CHROMA_SHAPED_ROWS];
    if with_luma {
        let lcols = luma.columns;
        for (cy, dst_row) in luma_sub.iter_mut().enumerate() {
            let ly = (cy << 1) + AR_PADDING;
            let r0 = &luma.values[ly * lcols + AR_PADDING..][..2 * CHROMA_SHAPED_COLS];
            let r1 = &luma.values[(ly + 1) * lcols + AR_PADDING..][..2 * CHROMA_SHAPED_COLS];
            for (col, avg) in dst_row.iter_mut().enumerate() {
                let (tl, tr): (i32, i32) = (r0[2 * col].into(), r0[2 * col + 1].into());
                let (bl, br): (i32, i32) = (r1[2 * col].into(), r1[2 * col + 1].into());
                *avg = (tl + tr + bl + br + 2) >> 2;
            }
        }
    }
    let luma_coeff_idx = ar_positions(lag).count();
    let chroma = |seed_xor: u16, enabled: bool, coefficients: &[u8; 25]| -> Template<G> {
        let mut grain = white_noise::<G>(
            &mut Random(p.grain_seed ^ seed_xor),
            CHROMA_TEMPLATE,
            shift,
            enabled,
        );
        if !enabled {
            return grain;
        }
        let luma_coefficient = i32::from(coefficients[luma_coeff_idx]) - 128;
        let use_luma = with_luma && luma_coefficient != 0;
        auto_regress(&mut grain, lag, coefficients, ar, |y, sums| {
            if !use_luma {
                return;
            }
            for (sum, &avg) in sums.iter_mut().zip(&luma_sub[y - AR_PADDING]) {
                *sum += luma_coefficient * avg;
            }
        });
        grain
    };
    let cb = chroma(CB_SEED_XOR, cb_enabled, &p.ar_coeffs_cb_plus_128);
    let cr = chroma(CR_SEED_XOR, cr_enabled, &p.ar_coeffs_cr_plus_128);
    [luma, cb, cr]
}

/// Root-mean-square grain value of each plane's template (Y, Cb, Cr) for
/// `p`'s seed, AR filter and shifts, over the part the AR filter shaped:
/// the noise a unit scaling adds, which an encoder divides out to choose
/// scaling points for a target noise level. Zero for a plane without grain.
#[must_use]
pub fn av1_grain_template_rms(p: &Av1FilmGrainParams) -> [f64; 3] {
    templates(p).map(|t| {
        let cols = t.columns - 2 * AR_PADDING;
        let count = (t.rows - AR_PADDING) * cols;
        let mut energy = 0.0f64;
        for y in AR_PADDING..t.rows {
            for &v in &t.values[y * t.columns + AR_PADDING..][..cols] {
                let v = f64::from(i32::from(v));
                energy += v * v;
            }
        }
        (energy / count as f64).sqrt()
    })
}

/// The scaling lookup of a plane (AV1 §7.18.3.4): piecewise-linear
/// through the points, flat outside them.
///
/// Every value is a point's scaling or between two neighbours' (the
/// rounded interpolation never overshoots its end: checked over every
/// step in `scaling_interpolation_stays_between_points`), so `u8` is exact.
fn scaling_lut<const N: usize>(points: &Av1ScalingPoints<N>) -> [u8; 256] {
    let mut lut = [0u8; 256];
    if points.count == 0 {
        return lut;
    }
    let value = |i: usize| usize::from(points.value[i]);
    let scaling = |i: usize| i32::from(points.scaling[i]);
    lut[..value(0)].fill(points.scaling[0]);
    for i in 0..points.count - 1 {
        let delta_y = scaling(i + 1) - scaling(i);
        let delta_x = (value(i + 1) - value(i)) as i32;
        let delta = i64::from(delta_y) * i64::from((65536 + (delta_x >> 1)) / delta_x);
        for x in 0..delta_x {
            lut[value(i) + x as usize] = interpolate(scaling(i), i64::from(x) * delta) as u8;
        }
    }
    let last = points.count - 1;
    lut[value(last)..].fill(points.scaling[last]);
    lut
}

/// `scaling + Round2(offset, 16)`: a step of the scaling interpolation.
fn interpolate(scaling: i32, offset: i64) -> i32 {
    scaling + ((offset + 32768) >> 16) as i32
}

/// `RandomRegister` of luma stripe `index` (AV1 §7.18.3.5).
fn stripe_seed(seed: u16, index: usize) -> u16 {
    let n = index as u16;
    seed ^ ((n.wrapping_mul(37).wrapping_add(178) & 255) << 8)
        ^ (n.wrapping_mul(173).wrapping_add(105) & 255)
}

/// Blend grain `new` over the previous block's `old` at distance
/// `index` into the overlap (27/17 then 17/27 over two samples, 23/22
/// over one when subsampled).
fn blend_overlap(old: i32, new: i32, (index, sub): (usize, usize), (min, max): (i32, i32)) -> i32 {
    let mixed = match (sub, index) {
        (1, _) => old * 23 + new * 22,
        (_, 0) => old * 27 + new * 17,
        _ => old * 17 + new * 27,
    };
    round2(mixed, 5).clamp(min, max)
}

/// Add film grain (AV1 §7.18.3) to the `width` x `height` frame `input`
/// (Y, U, V views) into `output`. 4:2:0, 8-bit; colour matrices other
/// than identity (identity needs 4:4:4).
///
/// # Panics
///
/// If a plane view is smaller than its plane of the frame.
pub fn av1_add_film_grain(
    p: &Av1FilmGrainParams,
    input: [&PlaneView<'_>; 3],
    (width, height): (usize, usize),
    output: [&mut PlaneViewMut<'_>; 3],
) {
    crate::Dsp::new().av1_add_film_grain(p, input, (width, height), output);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`Template`]'s bound: white noise at the smallest shift fits `i8`.
    #[test]
    fn white_noise_fits_i8() {
        let shift = GAUSSIAN_PRECISION - 8;
        for &g in &GAUSSIAN_SEQUENCE {
            let v = round2(i32::from(g), shift);
            assert!((GRAIN_MIN..=GRAIN_MAX).contains(&v), "{g} -> {v}");
        }
    }

    /// [`scaling_lut`]'s bound: every interpolation step between two
    /// scalings stays between them, so the lookup fits `u8`.
    #[test]
    fn scaling_interpolation_stays_between_points() {
        for delta_x in 1..=255_i32 {
            let reciprocal = i64::from((65536 + (delta_x >> 1)) / delta_x);
            for delta_y in -255..=255_i32 {
                let delta = i64::from(delta_y) * reciprocal;
                for x in 0..delta_x {
                    let step = interpolate(0, i64::from(x) * delta);
                    assert!((delta_y.min(0)..=delta_y.max(0)).contains(&step));
                }
            }
        }
    }
}
