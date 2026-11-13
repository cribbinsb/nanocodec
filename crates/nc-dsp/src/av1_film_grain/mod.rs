//! AV1 film grain synthesis (AV1 §7.18.3), 8-bit 4:2:0: pseudo-random
//! grain templates shaped by an auto-regressive filter, laid over the
//! output frame in 32x32 luma blocks at random offsets (with optional
//! overlap blending), scaled by a piecewise-linear function of the
//! underlying samples. Applied to the shown frame only, never to
//! references.
//!
//! Written from the spec's structure (§7.18.3.3–5: grain generation,
//! scaling lookup, noise stripes, noise image, blending); the arithmetic
//! is checked against libaom v3.15.3 `av1/decoder/grain_synthesis.c`
//! (`av1_add_film_grain`; BSD-2-Clause, see `THIRD_PARTY.md`).

mod gaussian;
pub(crate) mod kernels;
mod synthesis;

use nc_common::{PlaneView, PlaneViewMut};

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
    /// `point_cb_value` / `point_cb_scaling`.
    pub cb_points: Av1ScalingPoints<10>,
    /// `point_cr_value` / `point_cr_scaling`.
    pub cr_points: Av1ScalingPoints<10>,
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
/// Smallest and largest 8-bit grain value (`GrainMin`, `GrainMax`).
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
/// Luma rows per noise stripe and side of the grain blocks.
const BLOCK: usize = 32;
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

/// A grain template: `rows` x `columns` values.
struct Template {
    columns: usize,
    values: Vec<i16>,
}

impl Template {
    fn at(&self, row: usize, column: usize) -> i32 {
        i32::from(self.values[row * self.columns + column])
    }
}

/// White Gaussian grain (AV1 §7.18.3.3), or zeros when the plane has no
/// grain (no random numbers drawn then).
fn white_noise(
    random: &mut Random,
    (rows, columns): (usize, usize),
    shift: u32,
    enabled: bool,
) -> Template {
    let values = (0..rows * columns)
        .map(|_| {
            if enabled {
                round2(
                    i32::from(GAUSSIAN_SEQUENCE[random.next(GAUSSIAN_BITS)]),
                    shift,
                ) as i16
            } else {
                0
            }
        })
        .collect();
    Template { columns, values }
}

/// The auto-regressive neighbourhood of lag `lag`: every (row, column)
/// offset above and left of the sample, in coefficient order.
fn ar_positions(lag: usize) -> impl Iterator<Item = (isize, isize)> {
    let lag = lag as isize;
    (-lag..=0)
        .flat_map(move |dr| (-lag..=lag).map(move |dc| (dr, dc)))
        .take_while(|&(dr, dc)| !(dr == 0 && dc == 0))
}

/// Resolve the AR neighbourhood once per template. Zero coefficients contribute
/// nothing; removing them preserves the recurrence and its rounding exactly.
fn ar_taps(lag: usize, columns: usize, coefficients: &[u8]) -> Vec<(isize, i32)> {
    ar_positions(lag)
        .zip(coefficients)
        .filter_map(|((row, column), &value)| {
            let coefficient = i32::from(value) - 128;
            (coefficient != 0).then_some((row * columns as isize + column, coefficient))
        })
        .collect()
}

fn ar_sum(template: &Template, index: usize, taps: &[(isize, i32)]) -> i32 {
    taps.iter()
        .map(|&(offset, coefficient)| {
            coefficient * i32::from(template.values[(index as isize + offset) as usize])
        })
        .sum()
}

/// The grain templates of the three planes (AV1 §7.18.3.3).
fn templates(p: &Av1FilmGrainParams) -> [Template; 3] {
    let shift = GAUSSIAN_PRECISION - 8 + u32::from(p.grain_scale_shift);
    let ar_shift = u32::from(p.ar_coeff_shift_minus_6) + 6;
    let lag = usize::from(p.ar_coeff_lag);
    let mut luma = white_noise(
        &mut Random(p.grain_seed),
        LUMA_TEMPLATE,
        shift,
        p.y_points.count > 0,
    );
    if p.y_points.count > 0 {
        let taps = ar_taps(lag, luma.columns, &p.ar_coeffs_y_plus_128);
        for y in AR_PADDING..LUMA_TEMPLATE.0 {
            for x in AR_PADDING..LUMA_TEMPLATE.1 - AR_PADDING {
                let index = y * luma.columns + x;
                let sum = ar_sum(&luma, index, &taps);
                luma.values[index] = (i32::from(luma.values[index]) + round2(sum, ar_shift))
                    .clamp(GRAIN_MIN, GRAIN_MAX) as i16;
            }
        }
    }
    let chroma = |seed_xor: u16, enabled: bool, coefficients: &[u8; 25]| {
        let mut grain = white_noise(
            &mut Random(p.grain_seed ^ seed_xor),
            CHROMA_TEMPLATE,
            shift,
            enabled,
        );
        if !enabled {
            return grain;
        }
        let luma_coefficient = coefficients[ar_positions(lag).count()];
        let taps = ar_taps(lag, grain.columns, coefficients);
        for y in AR_PADDING..CHROMA_TEMPLATE.0 {
            for x in AR_PADDING..CHROMA_TEMPLATE.1 - AR_PADDING {
                let index = y * grain.columns + x;
                let mut sum = ar_sum(&grain, index, &taps);
                if p.y_points.count > 0 {
                    // The co-located 2x2 luma grain, averaged.
                    let (ly, lx) = (
                        ((y - AR_PADDING) << 1) + AR_PADDING,
                        ((x - AR_PADDING) << 1) + AR_PADDING,
                    );
                    let luma_sum = luma.at(ly, lx)
                        + luma.at(ly, lx + 1)
                        + luma.at(ly + 1, lx)
                        + luma.at(ly + 1, lx + 1);
                    sum += (i32::from(luma_coefficient) - 128) * round2(luma_sum, 2);
                }
                grain.values[index] = (i32::from(grain.values[index]) + round2(sum, ar_shift))
                    .clamp(GRAIN_MIN, GRAIN_MAX) as i16;
            }
        }
        grain
    };
    let cb_enabled = p.cb_points.count > 0 || p.chroma_scaling_from_luma;
    let cr_enabled = p.cr_points.count > 0 || p.chroma_scaling_from_luma;
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
        let rows = t.values.len() / t.columns;
        let shaped: Vec<i32> = (AR_PADDING..rows)
            .flat_map(|y| (AR_PADDING..t.columns - AR_PADDING).map(move |x| (y, x)))
            .map(|(y, x)| t.at(y, x))
            .collect();
        let energy: f64 = shaped.iter().map(|&v| f64::from(v * v)).sum();
        (energy / shaped.len() as f64).sqrt()
    })
}

/// The scaling lookup of a plane (AV1 §7.18.3.4): piecewise-linear
/// through the points, flat outside them.
fn scaling_lut<const N: usize>(points: &Av1ScalingPoints<N>) -> [i32; 256] {
    let mut lut = [0i32; 256];
    if points.count == 0 {
        return lut;
    }
    let value = |i: usize| usize::from(points.value[i]);
    let scaling = |i: usize| i32::from(points.scaling[i]);
    lut[..value(0)].fill(scaling(0));
    for i in 0..points.count - 1 {
        let delta_y = scaling(i + 1) - scaling(i);
        let delta_x = (value(i + 1) - value(i)) as i32;
        let delta = i64::from(delta_y) * i64::from((65536 + (delta_x >> 1)) / delta_x);
        for x in 0..delta_x {
            lut[value(i) + x as usize] = scaling(i) + ((i64::from(x) * delta + 32768) >> 16) as i32;
        }
    }
    let last = points.count - 1;
    lut[value(last)..].fill(scaling(last));
    lut
}

/// `RandomRegister` of luma stripe `index`.
fn stripe_seed(seed: u16, index: usize) -> u16 {
    let n = index as u16;
    seed ^ ((n.wrapping_mul(37).wrapping_add(178) & 255) << 8)
        ^ (n.wrapping_mul(173).wrapping_add(105) & 255)
}

/// Blend grain `new` over the previous block's `old` at distance
/// `index` into the overlap (27/17 then 17/27 over two samples, 23/22
/// over one when subsampled).
fn blend_overlap(old: i32, new: i32, index: usize, sub: usize) -> i32 {
    let mixed = match (sub, index) {
        (1, _) => old * 23 + new * 22,
        (_, 0) => old * 27 + new * 17,
        _ => old * 17 + new * 27,
    };
    round2(mixed, 5).clamp(GRAIN_MIN, GRAIN_MAX)
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
