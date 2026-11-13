//! VP9 scalar 8/16/32-point transforms: the DCT on the existing AV1
//! butterfly graphs (only cosine precision, pass scaling and inverse
//! wrapping differ), and the 8/16-point ADST generated from libvpx (`adst`).
//! Checked against libvpx v1.16.0 forward/inverse C (BSD-3-Clause) by nc-vp9's
//! independent golden vectors. AV1 retains its clamp policy and SIMD dispatch.
use crate::txfm::{dct, inverse_lanes::InverseLanes};

mod adst;

/// VP9 `cospi_k_64` (`vpx_dsp/txfm_common.h`), the 14-bit cosines of
/// k*pi/64. The shared graphs index cos(j*pi/128), hence doubled indices;
/// the 8/16/32-point graphs read only even indices.
pub(crate) const VP9_COSPI_64: [i32; 32] = [
    16384, 16364, 16305, 16207, 16069, 15893, 15679, 15426, 15137, 14811, 14449, 14053, 13623,
    13160, 12665, 12140, 11585, 11003, 10394, 9760, 9102, 8423, 7723, 7005, 6270, 5520, 4756, 3981,
    3196, 2404, 1606, 804,
];
const COSINE: [i32; 64] = {
    let mut values = [0; 64];
    let mut k = 0;
    while k < 32 {
        values[2 * k] = VP9_COSPI_64[k];
        k += 1;
    }
    values
};
/// Cosine precision of every VP9 rotation.
const COS_BIT: u32 = 14;

#[derive(Clone, Copy)]
struct Wrapped(i32);
impl InverseLanes for Wrapped {
    fn splat(value: i32) -> Self {
        Self(value)
    }
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        let value =
            (i64::from(w0) * i64::from(a.0) + i64::from(w1) * i64::from(b.0) + (1 << (bit - 1)))
                >> bit;
        Self(i32::from(value as i16))
    }
    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        Self::half_btf(w, a, 0, Self(0), bit)
    }
    fn clamp16(self) -> Self {
        Self(i32::from(self.0 as i16))
    }
}

/// The 1-D transform of one pass of a VP9 hybrid transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp9Pass {
    /// The DCT.
    Dct,
    /// The ADST (8- and 16-point; the 4-point ADST lives in nc-vp9).
    Adst,
}

/// One 1-D forward pass of `N` points (full precision).
fn forward_line<const N: usize>(pass: Vp9Pass, line: &[i32; N]) -> [i32; N] {
    let mut out = [0; N];
    match (pass, N) {
        (Vp9Pass::Dct, 8) => dct::fdct8(
            line.first_chunk().unwrap(),
            out.first_chunk_mut().unwrap(),
            &COSINE,
            COS_BIT,
        ),
        (Vp9Pass::Dct, 16) => dct::fdct16(
            line.first_chunk().unwrap(),
            out.first_chunk_mut().unwrap(),
            &COSINE,
            COS_BIT,
        ),
        (Vp9Pass::Dct, 32) => dct::fdct32(
            line.first_chunk().unwrap(),
            out.first_chunk_mut().unwrap(),
            &COSINE,
            COS_BIT,
        ),
        (Vp9Pass::Adst, 8) => {
            adst::fadst8(line.first_chunk().unwrap(), out.first_chunk_mut().unwrap());
        }
        (Vp9Pass::Adst, 16) => {
            adst::fadst16(line.first_chunk().unwrap(), out.first_chunk_mut().unwrap());
        }
        _ => unreachable!("VP9 has no {N}-point {pass:?} here"),
    }
    out
}

/// One 1-D inverse pass of `N` points with VP9's 16-bit stage wrap.
fn inverse_line<const N: usize>(pass: Vp9Pass, line: &[i32; N]) -> [i32; N] {
    let mut out = [0; N];
    match pass {
        Vp9Pass::Dct => {
            let wrapped: [Wrapped; N] = std::array::from_fn(|i| Wrapped(line[i]));
            let mut result = [Wrapped(0); N];
            match N {
                8 => dct::idct8(
                    wrapped.first_chunk().unwrap(),
                    result.first_chunk_mut().unwrap(),
                    &COSINE,
                    COS_BIT,
                ),
                16 => dct::idct16(
                    wrapped.first_chunk().unwrap(),
                    result.first_chunk_mut().unwrap(),
                    &COSINE,
                    COS_BIT,
                ),
                32 => dct::idct32(
                    wrapped.first_chunk().unwrap(),
                    result.first_chunk_mut().unwrap(),
                    &COSINE,
                    COS_BIT,
                ),
                _ => unreachable!("VP9 has no {N}-point DCT here"),
            }
            out = std::array::from_fn(|i| result[i].0);
        }
        Vp9Pass::Adst => match N {
            8 => adst::iadst8(line.first_chunk().unwrap(), out.first_chunk_mut().unwrap()),
            16 => adst::iadst16(line.first_chunk().unwrap(), out.first_chunk_mut().unwrap()),
            _ => unreachable!("VP9 has no {N}-point ADST"),
        },
    }
    out
}

/// Forward transform: columns of the input times `PRE`, then rows; `post`
/// maps the result to VP9's coefficient scale.
fn forward<const N: usize>(
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i16],
    output: &mut [i32],
    post: impl Fn(i32) -> i32,
) {
    /// Input gain for precision; every pass gains sqrt(N/2).
    const PRE: i32 = 4;
    let mut temp = [[0i32; N]; N];
    for x in 0..N {
        let line: [i32; N] = std::array::from_fn(|y| i32::from(input[y * N + x]) * PRE);
        let transformed = forward_line(columns, &line);
        for y in 0..N {
            temp[y][x] = transformed[y];
        }
    }
    for (y, row) in temp.iter().enumerate() {
        let transformed = forward_line(rows, row);
        for x in 0..N {
            output[y * N + x] = post(transformed[x]);
        }
    }
}

/// Inverse transform: rows (inputs wrapped to 16 bits) then columns, then
/// the normative rounding shift.
fn inverse<const N: usize>(
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i32],
    output: &mut [i32],
    shift: u32,
) {
    let mut temp = [[0i32; N]; N];
    for (y, row) in temp.iter_mut().enumerate() {
        let line: [i32; N] = std::array::from_fn(|x| i32::from(input[y * N + x] as i16));
        *row = inverse_line(rows, &line);
    }
    for x in 0..N {
        let line: [i32; N] = std::array::from_fn(|y| temp[y][x]);
        let transformed = inverse_line(columns, &line);
        for y in 0..N {
            output[y * N + x] = (transformed[y] + (1 << (shift - 1))) >> shift;
        }
    }
}

/// Divide by `d`, rounding half away from zero (encoder-side scaling only).
const fn divide_round(value: i32, d: i32) -> i32 {
    if value < 0 {
        -((-value + d / 2) / d)
    } else {
        (value + d / 2) / d
    }
}

/// VP9 forward transform of an `N`x`N` raster residual (`N = 1 << log2` in
/// 8..=32; 32 is DCT only), returning raster coefficients at VP9's scale:
/// 8x the orthonormal transform up to 16x16, 4x for 32x32. Encoder-side
/// only: any accurate transform at this scale is a valid encoder choice.
///
/// # Panics
///
/// If `log2` is outside 3..=5, a 32-point ADST is asked for, or a slice is
/// shorter than `N * N`.
pub fn vp9_forward_transform(
    log2: u8,
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i16],
    output: &mut [i32],
) {
    match log2 {
        // VP9 truncates the final 8x8 divide toward zero, not arithmetic shift.
        3 => forward::<8>(columns, rows, input, output, |v| v / 2),
        4 => forward::<16>(columns, rows, input, output, |v| divide_round(v, 4)),
        5 => forward::<32>(columns, rows, input, output, |v| divide_round(v, 16)),
        _ => panic!("VP9 transform size 1 << {log2} outside 8..=32"),
    }
}

/// VP9 inverse transform of `N`x`N` dequantized raster coefficients
/// (`N = 1 << log2` in 8..=32), returning residuals before addition and
/// clipping, with VP9's normative arithmetic (`vp9_ihtNxN_add_c`,
/// `vpx_idct32x32_1024_add_c`).
///
/// # Panics
///
/// As [`vp9_forward_transform`].
pub fn vp9_inverse_transform(
    log2: u8,
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i32],
    output: &mut [i32],
) {
    match log2 {
        3 => inverse::<8>(columns, rows, input, output, 5),
        4 => inverse::<16>(columns, rows, input, output, 6),
        5 => inverse::<32>(columns, rows, input, output, 6),
        _ => panic!("VP9 transform size 1 << {log2} outside 8..=32"),
    }
}

/// [`vp9_inverse_transform`] added to the prediction in the `N`x`N` block
/// at the start of `dst` (rows `dst_stride` apart) and clipped to 8 bits
/// (`vp9_ihtNxN_add_c`).
///
/// # Panics
///
/// As [`vp9_inverse_transform`], or if `dst` is too small.
pub fn vp9_inverse_add(
    log2: u8,
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i32],
    dst: &mut [u8],
    dst_stride: usize,
) {
    let side = 1 << log2;
    let mut residual = [0; 32 * 32];
    let residual = &mut residual[..side * side];
    vp9_inverse_transform(log2, columns, rows, input, residual);
    for (row, values) in dst.chunks_mut(dst_stride).zip(residual.chunks_exact(side)) {
        for (sample, &value) in row[..side].iter_mut().zip(values.iter()) {
            *sample = (i32::from(*sample) + value).clamp(0, 255) as u8;
        }
    }
}

/// VP9 8×8 forward DCT (see [`vp9_forward_transform`]).
#[must_use]
pub fn vp9_forward_dct8(input: &[i16; 64]) -> [i32; 64] {
    let mut output = [0; 64];
    vp9_forward_transform(3, Vp9Pass::Dct, Vp9Pass::Dct, input, &mut output);
    output
}

/// VP9 8×8 inverse DCT of 16-bit coefficients (see [`vp9_inverse_transform`]).
#[must_use]
pub fn vp9_inverse_dct8(input: &[i16; 64]) -> [i32; 64] {
    let mut output = [0; 64];
    vp9_inverse_transform(
        3,
        Vp9Pass::Dct,
        Vp9Pass::Dct,
        &input.map(i32::from),
        &mut output,
    );
    output
}
