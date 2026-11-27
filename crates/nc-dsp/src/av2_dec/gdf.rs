//! The guided detail filter (GDF) of AV2 luma (dav2d `gdf_prep_c`,
//! `gdf_add_c` in `looprestoration_tmpl.c`; AVM v1.0.0 `gdf.c`): each
//! sample's correction is looked up from a trained table, indexed by three
//! quantised filter responses of the samples around it; which response is
//! which depends on the local gradient class of its 2x2 block.
//!
//! [`gdf_error`] computes a block's corrections from a window of the
//! frame normalised to 10 bits, with [`GDF_BORDER`] samples of context on
//! every side (the caller decides what fills them: dav2d's stripe and frame
//! edge rules live in `nc_av2::general::filters::gdf`); [`gdf_add`] adds
//! them, scaled by the frame's strength.

use nc_common::{BitDepth, Sample};

use crate::Dsp;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod vector8;

/// Rows and columns of context the window holds on each side of the
/// block.
pub const GDF_BORDER: usize = 6;
/// Largest block side (dav2d's 64x64 stripe blocks).
pub const GDF_MAX_SIDE: usize = 64;
/// Bits the window's samples carry.
pub const GDF_INPUT_BITS: u32 = 10;
/// Row stride of the corrections [`gdf_error`] writes.
pub const GDF_ERROR_STRIDE: usize = GDF_MAX_SIDE;

/// The three filter responses round from Q15 (dav2d `(abs(v) + (1 << 14)) >> 15`).
const RESPONSE_SHIFT: u32 = 15;
/// Pairwise sums are clipped to 10 signed bits.
const PAIR_MIN: i32 = -512;
const PAIR_MAX: i32 = 511;
/// Gradient sums are scaled down by 16 before their cap.
const GRADIENT_SHIFT: u32 = 4;
/// The correction's precision beyond the sample's: `12 - bit depth` bits.
const CORRECTION_BITS: u32 = 12;
/// Directions of the gradient: (dy, dx) of the two outer samples.
const GRADIENT_DIRECTIONS: [(isize, isize); 4] = [(1, 0), (0, 1), (1, 1), (-1, 1)];

/// One trained parameter set (dav2d `Dav2dGdfParams`).
#[derive(Debug, Clone, Copy)]
pub struct GdfParams {
    /// Quantisation levels per response on each side of zero (5 or 8).
    pub scale: u16,
    /// Each response's bias (the third's grows with the gradients).
    pub bias: [i16; 3],
    /// Cap of each direction's gradient in the third bias.
    pub gradient_alpha: [u16; 4],
    /// Weight of each direction's capped gradient.
    pub gradient_weight: [i16; 4],
    /// Clip of each neighbour pair's differences (vertical and horizontal
    /// at distances 1..6, the diagonals, the knight moves).
    pub alpha: [[u16; 2]; 9],
    /// Weights of the distance-1, -2 and -3 pairs in each response.
    pub weight1: [[i16; 2]; 7],
    /// Weights of the distance-3..6 pairs in the first two responses.
    pub weight2: [[i16; 4]; 2],
    /// Weights of the diagonal pairs in the first two responses.
    pub weight3: [[i16; 2]; 2],
    /// Weights of the knight-move pairs.
    pub weight4: [[i16; 2]; 2],
    /// The correction by combined response index (`(2 · scale)^3`).
    pub error_lut: &'static [i8],
}

/// A window of the frame around one block: `samples[(y + GDF_BORDER) *
/// stride + x + GDF_BORDER]` is the block's sample (x, y), in 10 bits.
#[derive(Debug, Clone, Copy)]
pub struct GdfWindow<'a> {
    /// The samples.
    pub samples: &'a [u16],
    /// Row stride.
    pub stride: usize,
    /// The block's width (a multiple of 8, at most 64).
    pub width: usize,
    /// The block's height (the same).
    pub height: usize,
}

impl GdfWindow<'_> {
    /// The sample at (`x`, `y`) of the block (up to [`GDF_BORDER`] outside).
    #[inline]
    fn at(&self, x: isize, y: isize) -> i32 {
        let border = GDF_BORDER as isize;
        let index = (y + border) as usize * self.stride + (x + border) as usize;
        i32::from(self.samples[index])
    }

    /// The gradient of direction `d` at (`x`, `y`): `|2b - a - c|`.
    fn gradient(&self, d: usize, x: isize, y: isize) -> i32 {
        let (dy, dx) = GRADIENT_DIRECTIONS[d];
        (2 * self.at(x, y) - self.at(x - dx, y - dy) - self.at(x + dx, y + dy)).abs()
    }
}

/// The clipped sum of a neighbour pair's differences from the centre
/// (dav2d `gdf_sum`).
fn pair(a: i32, b: i32, alpha: u16, centre: i32) -> i32 {
    let alpha = i32::from(alpha);
    let a = (a - centre).clamp(-alpha, alpha);
    let b = (b - centre).clamp(-alpha, alpha);
    (a + b).clamp(PAIR_MIN, PAIR_MAX)
}

/// A 2x2 block's gradient class and the third response's bias, from the
/// gradients of the 4x4 samples around it.
fn block_class(window: &GdfWindow<'_>, params: &GdfParams, x: isize, y: isize) -> (u8, i32) {
    let mut sums = [0_i32; 4];
    for (d, sum) in sums.iter_mut().enumerate() {
        for yy in y - 1..y + 3 {
            for xx in x - 1..x + 3 {
                *sum += window.gradient(d, xx, yy);
            }
        }
    }
    let mut bias = i32::from(params.bias[2]);
    for ((&sum, &alpha), &weight) in sums
        .iter()
        .zip(&params.gradient_alpha)
        .zip(&params.gradient_weight)
    {
        bias += (sum >> GRADIENT_SHIFT).min(i32::from(alpha)) * i32::from(weight);
    }
    let class = u8::from(sums[0] <= sums[1]) | u8::from(sums[2] <= sums[3]) << 1;
    (class, bias)
}

/// The three responses of the sample at (`x`, `y`) in its block of class
/// `class` (bit 0 swaps vertical and horizontal pairs, bit 1 the two
/// diagonals).
fn responses(
    window: &GdfWindow<'_>,
    params: &GdfParams,
    (x, y): (isize, isize),
    (class, bias): (u8, i32),
) -> [i32; 3] {
    let swap = |on: bool, a: (i32, i32), b: (i32, i32)| if on { (b, a) } else { (a, b) };
    let transpose = class & 1 != 0;
    let mirror = class & 2 != 0;
    let m = window.at(x, y);
    let mut values = [i32::from(params.bias[0]), i32::from(params.bias[1]), bias];
    let w1 = |i: usize, j: usize| i32::from(params.weight1[i][j]);
    for k in 0..6_isize {
        let ku = k as usize;
        let ((above, below), (left, right)) = swap(
            transpose,
            (window.at(x, y - 1 - k), window.at(x, y + 1 + k)),
            (window.at(x - k - 1, y), window.at(x + k + 1, y)),
        );
        let vertical = pair(above, below, params.alpha[ku][0], m);
        let horizontal = pair(left, right, params.alpha[ku][1], m);
        if k < 2 {
            for (index, value) in values.iter_mut().enumerate() {
                *value += vertical * w1(ku * 3 + index, 0) + horizontal * w1(ku * 3 + index, 1);
            }
        } else {
            values[0] += vertical * i32::from(params.weight2[0][ku - 2]);
            values[1] += horizontal * i32::from(params.weight2[1][ku - 2]);
            if k == 2 {
                values[2] += vertical * w1(6, 0) + horizontal * w1(6, 1);
            }
        }
    }
    // The diagonals at distance one.
    let ((tl, br), (tr, bl)) = swap(
        mirror,
        (window.at(x - 1, y - 1), window.at(x + 1, y + 1)),
        (window.at(x + 1, y - 1), window.at(x - 1, y + 1)),
    );
    let down = pair(tl, br, params.alpha[6][0], m);
    let up = pair(tr, bl, params.alpha[6][1], m);
    for (r, value) in values.iter_mut().take(2).enumerate() {
        *value += down * i32::from(params.weight3[r][0]) + up * i32::from(params.weight3[r][1]);
    }
    // The knight moves: two rows and one column, or one row and two columns.
    let mut ttl = window.at(x - 1, y - 2);
    let mut bbr = window.at(x + 1, y + 2);
    let mut tll = window.at(x - 2, y - 1);
    let mut brr = window.at(x + 2, y + 1);
    let mut ttr = window.at(x + 1, y - 2);
    let mut bbl = window.at(x - 1, y + 2);
    let mut trr = window.at(x + 2, y - 1);
    let mut bll = window.at(x - 2, y + 1);
    if transpose {
        (ttl, bbr, tll, brr) = (tll, brr, ttl, bbr);
        (ttr, bbl, trr, bll) = (trr, bll, ttr, bbl);
    }
    if mirror {
        (ttl, bbr, ttr, bbl) = (ttr, bbl, ttl, bbr);
        (tll, brr, trr, bll) = (trr, bll, tll, brr);
    }
    let tall_down = pair(ttl, bbr, params.alpha[7][0], m);
    let tall_up = pair(ttr, bbl, params.alpha[7][1], m);
    let wide_down = pair(tll, brr, params.alpha[8][0], m);
    let wide_up = pair(trr, bll, params.alpha[8][1], m);
    values[0] +=
        tall_down * i32::from(params.weight4[0][0]) + tall_up * i32::from(params.weight4[0][1]);
    values[1] +=
        wide_down * i32::from(params.weight4[1][0]) + wide_up * i32::from(params.weight4[1][1]);
    values
}

/// The table index of three responses: each quantised to `2 · scale`
/// levels.
fn table_index(values: [i32; 3], scale: i32) -> usize {
    let mut index = 0;
    for value in values {
        let v = value * scale;
        let rounded = ((v.abs() + (1 << (RESPONSE_SHIFT - 1))) >> RESPONSE_SHIFT) * v.signum();
        let level = rounded.clamp(-scale, scale - 1) + scale;
        index = index * 2 * scale + level;
    }
    index as usize
}

/// The corrections of a block (dav2d `gdf_prep_c`): `error[y *
/// GDF_ERROR_STRIDE + x]` for each of its samples.
///
/// # Panics
///
/// If the window is smaller than the block and its border, the block is
/// larger than [`GDF_MAX_SIDE`] or not a multiple of 8, or `error` is too
/// short.
pub fn gdf_error(window: &GdfWindow<'_>, params: &GdfParams, error: &mut [i8]) {
    validate_error_window(window, error);
    gdf_error_scalar(window, params, error);
}

/// Compute the block corrections using the selected DSP tier.
///
/// # Panics
///
/// If the window or output falls outside the requirements of [`gdf_error`].
#[allow(unsafe_code)]
pub fn gdf_error_with_dsp(dsp: &Dsp, window: &GdfWindow<'_>, params: &GdfParams, error: &mut [i8]) {
    validate_error_window(window, error);
    #[cfg(target_arch = "x86_64")]
    if matches!(dsp.isa(), crate::Isa::Avx2 | crate::Isa::Avx512)
        && window.width != 0
        && window.height != 0
        && vector_params(params)
    {
        // SAFETY: the DSP tier enables AVX2. The window has the documented
        // 10-bit samples, and its geometry, the output and LUT are checked.
        unsafe {
            if !window.width.is_multiple_of(16) {
                vector8::error::<crate::simd::avx2::I16x8>(window, params, error);
            } else if dsp.isa() == crate::Isa::Avx512 {
                avx2::error::<true>(window, params, error);
            } else {
                avx2::error::<false>(window, params, error);
            }
        }
        return;
    }
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    if dsp.isa() != crate::Isa::Scalar
        && window.width != 0
        && window.height != 0
        && vector_params(params)
    {
        #[cfg(target_arch = "aarch64")]
        type Lanes = crate::simd::neon::I16x8;
        #[cfg(target_arch = "wasm32")]
        type Lanes = crate::simd::simd128::I16x8;
        // SAFETY: the DSP tier enables the architecture's baseline SIMD.
        // The stripe and LUT are checked and its samples are in 10 bits.
        unsafe { vector8::error::<Lanes>(window, params, error) };
        return;
    }
    let _ = dsp;
    gdf_error_scalar(window, params, error);
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
fn vector_params(params: &GdfParams) -> bool {
    matches!(params.scale, 5 | 8)
        && params.error_lut.len() >= (2 * usize::from(params.scale)).pow(3)
        && params.alpha.iter().flatten().all(|&cap| cap <= 511)
        && params.gradient_alpha.iter().all(|&cap| cap <= 511)
}

fn validate_error_window(window: &GdfWindow<'_>, error: &[i8]) {
    let (width, height) = (window.width, window.height);
    assert!(width <= GDF_MAX_SIDE && height <= GDF_MAX_SIDE);
    assert!(width.is_multiple_of(8) && height.is_multiple_of(8));
    assert!(window.stride >= width + 2 * GDF_BORDER);
    assert!(window.samples.len() / window.stride >= height + 2 * GDF_BORDER);
    if width != 0 && height != 0 {
        assert!(error.len() >= (height - 1) * GDF_ERROR_STRIDE + width);
    }
}

fn gdf_error_scalar(window: &GdfWindow<'_>, params: &GdfParams, error: &mut [i8]) {
    let (width, height) = (window.width, window.height);
    let scale = i32::from(params.scale);
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            let block = block_class(window, params, x as isize, y as isize);
            for (yy, xx) in [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)] {
                let values = responses(window, params, (xx as isize, yy as isize), block);
                error[yy * GDF_ERROR_STRIDE + xx] = params.error_lut[table_index(values, scale)];
            }
        }
    }
}

/// Add a block's corrections, times the frame's strength `scale` (1..4),
/// to `width` x `height` samples of `dst` (row stride `stride`; dav2d
/// `gdf_add_c`).
///
/// # Panics
///
/// If `dst` or `error` is too short for the block.
pub fn gdf_add<T: Sample>(
    dst: &mut [T],
    stride: usize,
    error: &[i8],
    (width, height): (usize, usize),
    scale: i32,
    depth: BitDepth,
) {
    let shift = CORRECTION_BITS - depth.bits();
    let round = (1 << shift) >> 1;
    for y in 0..height {
        let row = &mut dst[y * stride..y * stride + width];
        let errors = &error[y * GDF_ERROR_STRIDE..y * GDF_ERROR_STRIDE + width];
        for (sample, &e) in row.iter_mut().zip(errors) {
            let diff = i32::from(e) * scale;
            let correction = ((diff.abs() + round) >> shift) * diff.signum();
            *sample = depth.clip_to::<T>(sample.to_i32() + correction);
        }
    }
}

/// Apply guided detail corrections to 8-bit pixels using the selected tier.
///
/// # Panics
///
/// As [`gdf_add`].
#[allow(unsafe_code)]
pub fn gdf_add_with_dsp(
    dsp: &Dsp,
    dst: &mut [u8],
    stride: usize,
    error: &[i8],
    (width, height): (usize, usize),
    scale: i32,
) {
    let size = (width, height);
    if width != 0
        && height != 0
        && width <= GDF_MAX_SIDE
        && width.is_multiple_of(8)
        && (1..=4).contains(&scale)
    {
        #[cfg(target_arch = "x86_64")]
        if matches!(dsp.isa(), crate::Isa::Avx2 | crate::Isa::Avx512) {
            // SAFETY: the selected tier enables AVX2. Rows check their spans.
            unsafe {
                if width == 32 {
                    add_avx2_bytes::<32>(dst, stride, error, height, scale);
                } else if width == 64 {
                    add_avx2_bytes::<64>(dst, stride, error, height, scale);
                } else if width.is_multiple_of(16) {
                    add_lanes::<crate::simd::avx2::I16x16>(dst, stride, error, size, scale);
                } else {
                    add_lanes::<crate::simd::avx2::I16x8>(dst, stride, error, size, scale);
                }
            }
            return;
        }
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
        if dsp.isa() != crate::Isa::Scalar {
            #[cfg(target_arch = "aarch64")]
            type Lanes = crate::simd::neon::I16x8;
            #[cfg(target_arch = "wasm32")]
            type Lanes = crate::simd::simd128::I16x8;
            // SAFETY: the selected tier enables baseline SIMD and rows
            // check their spans.
            unsafe { add_lanes::<Lanes>(dst, stride, error, size, scale) };
            return;
        }
    }
    let _ = dsp;
    gdf_add(dst, stride, error, size, scale, BitDepth::EIGHT);
}

/// Apply one vector of corrections at a time. Unlike dav2d's lossless
/// mask in the SIMD loop, the caller supplies corrections for a whole
/// non-lossless block. This uses the same rounded result with signed
/// words, for any i8 correction and scale 1..4.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
unsafe fn add_lanes<V: crate::simd::I16Lanes>(
    dst: &mut [u8],
    stride: usize,
    error: &[i8],
    (width, height): (usize, usize),
    scale: i32,
) {
    const SHIFT: u32 = CORRECTION_BITS - 8;
    let round = V::splat(1 << (SHIFT - 1));
    let scale = V::splat(scale as i16);
    for y in 0..height {
        let row = &mut dst[y * stride..y * stride + width];
        let err = &error[y * GDF_ERROR_STRIDE..y * GDF_ERROR_STRIDE + width];
        for (pixels, errors) in row
            .chunks_exact_mut(V::LANES)
            .zip(err.chunks_exact(V::LANES))
        {
            let diff = V::load_i8(errors).mul(scale);
            // Negative values subtract one before half-up rounding, giving
            // round(abs(diff)) with its original sign, including ties.
            let correction = diff.add(round).add(diff.sra(15)).sra(SHIFT);
            V::load_u8(pixels).add(correction).store_u8_sat(pixels);
        }
    }
}

/// The byte correction schedule of dav2d `gdf_add_8bpc_avx2`
/// (BSD-2-Clause, see `THIRD_PARTY.md`). The non-lossless block needs no
/// lossless-mask blend. The Q15 factors keep the magnitude of -128
/// representable and give exactly round(abs(error) * scale / 16).
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
unsafe fn add_avx2_bytes<const WIDTH: usize>(
    dst: &mut [u8],
    stride: usize,
    error: &[i8],
    height: usize,
    scale: i32,
) {
    use core::arch::x86_64::{
        _mm256_abs_epi8, _mm256_adds_epu8, _mm256_blendv_epi8, _mm256_loadu_si256,
        _mm256_mulhrs_epi16, _mm256_packus_epi16, _mm256_set1_epi16, _mm256_setzero_si256,
        _mm256_slli_epi16, _mm256_storeu_si256, _mm256_subs_epu8, _mm256_unpackhi_epi8,
        _mm256_unpacklo_epi8,
    };
    let dst_end = stride
        .checked_mul(height - 1)
        .and_then(|n| n.checked_add(WIDTH));
    let error_end = GDF_ERROR_STRIDE
        .checked_mul(height - 1)
        .and_then(|n| n.checked_add(WIDTH));
    assert!(dst_end.is_some_and(|n| n <= dst.len()));
    assert!(error_end.is_some_and(|n| n <= error.len()));
    let multiplier = _mm256_set1_epi16((scale as i16) << 4);
    let zero = _mm256_setzero_si256();
    // SAFETY: the caller selected AVX2 and a multiple-of-32 width. Each
    // indexed load and store stays within one of the checked row spans.
    unsafe {
        for y in 0..height {
            let row = dst.as_mut_ptr().add(y * stride);
            let err = error.as_ptr().add(y * GDF_ERROR_STRIDE);
            for x in (0..WIDTH).step_by(32) {
                let signed = _mm256_loadu_si256(err.add(x).cast());
                let samples = _mm256_loadu_si256(row.add(x).cast());
                let absolute = _mm256_abs_epi8(signed);
                let lo = _mm256_slli_epi16::<7>(_mm256_unpacklo_epi8(absolute, zero));
                let hi = _mm256_slli_epi16::<7>(_mm256_unpackhi_epi8(absolute, zero));
                let correction = _mm256_packus_epi16(
                    _mm256_mulhrs_epi16(lo, multiplier),
                    _mm256_mulhrs_epi16(hi, multiplier),
                );
                let result = _mm256_blendv_epi8(
                    _mm256_adds_epu8(samples, correction),
                    _mm256_subs_epu8(samples, correction),
                    signed,
                );
                _mm256_storeu_si256(row.add(x).cast(), result);
            }
        }
    }
}
