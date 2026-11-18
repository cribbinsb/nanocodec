//! VP9 scalar 8/16/32-point transforms: the DCT on the existing AV1
//! butterfly graphs (only cosine precision, pass scaling and inverse
//! wrapping differ), and the 8/16-point ADST generated from libvpx (`adst`).
//! Checked against libvpx v1.16.0 forward/inverse C (BSD-3-Clause) by nc-vp9's
//! independent golden vectors. AV1 retains its clamp policy and SIMD dispatch.
//!
//! The inverse runs at either depth class ([`InverseDepth`]): libvpx's 8-bit
//! decoder wraps every stage to 16 bits (`WRAPLOW`), its high-bit-depth
//! one (`vpx_highbd_idct*`, `vpx_highbd_iadst*`, the same statements) keeps
//! 32-bit stages with 64-bit products (`HIGHBD_WRAPLOW`).
use nc_common::{BitDepth, Sample};

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

/// The stage arithmetic of the inverse at one depth class.
pub(crate) trait InverseDepth {
    /// DCT lanes on the shared butterfly graphs.
    type Lane: InverseLanes;
    /// `WRAPLOW` (16 bits) or `HIGHBD_WRAPLOW` (32 bits) of a stage value.
    fn wrap(value: i64) -> i64;
    /// A coefficient or stage value as a lane.
    fn lane(value: i32) -> Self::Lane;
    /// A lane's value.
    fn value(lane: Self::Lane) -> i32;
}

/// libvpx's 8-bit decoder: 16-bit stages.
pub(crate) enum Eight {}

/// libvpx's high-bit-depth decoder: 32-bit stages, 64-bit products.
pub(crate) enum High {}

impl InverseDepth for Eight {
    type Lane = Wrapped;
    #[inline]
    fn wrap(value: i64) -> i64 {
        i64::from(value as i16)
    }
    #[inline]
    fn lane(value: i32) -> Wrapped {
        Wrapped(value)
    }
    #[inline]
    fn value(lane: Wrapped) -> i32 {
        lane.0
    }
}

impl InverseDepth for High {
    type Lane = Wide;
    #[inline]
    fn wrap(value: i64) -> i64 {
        i64::from(value as i32)
    }
    #[inline]
    fn lane(value: i32) -> Wide {
        Wide(value)
    }
    #[inline]
    fn value(lane: Wide) -> i32 {
        lane.0
    }
}

/// A DCT stage of the 8-bit decoder: products rounded and wrapped to 16 bits.
#[derive(Clone, Copy)]
pub(crate) struct Wrapped(i32);
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

/// A DCT stage of the high-bit-depth decoder: 64-bit products, 32-bit
/// stages (`HIGHBD_WRAPLOW` truncates; valid streams never reach it).
#[derive(Clone, Copy)]
pub(crate) struct Wide(i32);
impl InverseLanes for Wide {
    fn splat(value: i32) -> Self {
        Self(value)
    }
    fn add(self, other: Self) -> Self {
        Self(self.0.wrapping_add(other.0))
    }
    fn sub(self, other: Self) -> Self {
        Self(self.0.wrapping_sub(other.0))
    }
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        let value =
            (i64::from(w0) * i64::from(a.0) + i64::from(w1) * i64::from(b.0) + (1 << (bit - 1)))
                >> bit;
        Self(value as i32)
    }
    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        Self::half_btf(w, a, 0, Self(0), bit)
    }
    /// The 8-bit stage wrap does not apply above 8 bits.
    fn clamp16(self) -> Self {
        self
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

/// One 1-D inverse pass of `N` points with the depth's stage wrap.
fn inverse_line<const N: usize, D: InverseDepth>(pass: Vp9Pass, line: &[i32; N]) -> [i32; N] {
    let mut out = [0; N];
    match pass {
        Vp9Pass::Dct => {
            let lanes: [D::Lane; N] = std::array::from_fn(|i| D::lane(line[i]));
            let mut result = [D::lane(0); N];
            match N {
                8 => dct::idct8(
                    lanes.first_chunk().unwrap(),
                    result.first_chunk_mut().unwrap(),
                    &COSINE,
                    COS_BIT,
                ),
                16 => dct::idct16(
                    lanes.first_chunk().unwrap(),
                    result.first_chunk_mut().unwrap(),
                    &COSINE,
                    COS_BIT,
                ),
                32 => dct::idct32(
                    lanes.first_chunk().unwrap(),
                    result.first_chunk_mut().unwrap(),
                    &COSINE,
                    COS_BIT,
                ),
                _ => unreachable!("VP9 has no {N}-point DCT here"),
            }
            // Every DCT output leaves the graph through `half_btf` or
            // `clamp16`, so it is already wrapped.
            out = std::array::from_fn(|i| D::value(result[i]));
        }
        Vp9Pass::Adst => match N {
            8 => adst::iadst8::<D>(line.first_chunk().unwrap(), out.first_chunk_mut().unwrap()),
            16 => adst::iadst16::<D>(line.first_chunk().unwrap(), out.first_chunk_mut().unwrap()),
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
    output: &mut [i16],
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
            output[y * N + x] = crate::simd::saturate_i16(post(transformed[x]));
        }
    }
}

/// Inverse transform: rows then columns, then the normative rounding shift
/// (at 8 bits the result, at most 2^15 >> 5 in magnitude, fits 16 bits).
fn inverse<const N: usize, D: InverseDepth, C: Copy + Into<i32>>(
    (columns, rows): (Vp9Pass, Vp9Pass),
    input: &[C],
    output: &mut [i32],
    shift: u32,
) {
    let mut temp = [[0i32; N]; N];
    for (y, row) in temp.iter_mut().enumerate() {
        let line: [i32; N] = std::array::from_fn(|x| input[y * N + x].into());
        *row = inverse_line::<N, D>(rows, &line);
    }
    for x in 0..N {
        let line: [i32; N] = std::array::from_fn(|y| temp[y][x]);
        let transformed = inverse_line::<N, D>(columns, &line);
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
/// 8x the orthonormal transform up to 16x16, 4x for 32x32, saturated to 16
/// bits. Encoder-side only: any accurate transform at this scale is a valid
/// encoder choice. An orthonormal basis vector's L1 norm is at most
/// `sqrt(N)`, so 8-bit residuals give coefficients of at most
/// `8 * 255 * N` (16320 at 8x8, 32640 at 16x16 and, at 4x, 32x32) plus
/// rounding: the saturation does not trigger.
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
    output: &mut [i16],
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
/// `vpx_idct32x32_1024_add_c`). Coefficients are 16-bit: §8.6.2 requires
/// every dequantized value to fit 8 + `BitDepth` bits, libvpx's 8-bit
/// `tran_low_t`.
///
/// # Panics
///
/// As [`vp9_forward_transform`].
pub fn vp9_inverse_transform(
    log2: u8,
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i16],
    output: &mut [i16],
) {
    let side = 1 << log2;
    let mut residual = [0; 32 * 32];
    let residual = &mut residual[..side * side];
    inverse_at::<Eight, i16>(log2, (columns, rows), input, residual);
    for (out, &value) in output.iter_mut().zip(residual.iter()) {
        *out = value as i16;
    }
}

/// The `N`x`N` inverse (`N = 1 << log2` in 8..=32) at the depth class
/// `D`, into `output` as 32-bit residuals.
fn inverse_at<D: InverseDepth, C: Copy + Into<i32>>(
    log2: u8,
    passes: (Vp9Pass, Vp9Pass),
    input: &[C],
    output: &mut [i32],
) {
    match log2 {
        3 => inverse::<8, D, C>(passes, input, output, 5),
        4 => inverse::<16, D, C>(passes, input, output, 6),
        5 => inverse::<32, D, C>(passes, input, output, 6),
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
    input: &[i16],
    dst: &mut [u8],
    dst_stride: usize,
) {
    let side = 1 << log2;
    let mut residual = [0; 32 * 32];
    let residual = &mut residual[..side * side];
    inverse_at::<Eight, i16>(log2, (columns, rows), input, residual);
    add_clipped((dst, dst_stride), side, residual, BitDepth::EIGHT);
}

/// The inverse of any size (4x4 to 32x32) of samples deeper than 8 bits:
/// 32-bit coefficients, libvpx's high-bit-depth stages
/// (`vp9_highbd_iht{4x4,8x8,16x16}_*_add_c`, `vpx_highbd_idct32x32_*_add_c`),
/// added to the block at the start of `dst` and clipped to `depth`
/// (`highbd_clip_pixel_add`). libvpx zeroes a 1-D input of magnitude 2^25
/// or more (`detect_invalid_highbd_input`); valid streams never have one,
/// and here such input only wraps.
pub(crate) fn vp9_inverse_add_high<T: Sample>(
    log2: u8,
    (columns, rows): (Vp9Pass, Vp9Pass),
    input: &[i32],
    (dst, dst_stride): (&mut [T], usize),
    depth: BitDepth,
) {
    let passes = (columns, rows);
    let dst = (dst, dst_stride);
    // A residual of the block's own size: zeroing a 32x32 one per block
    // was a sixth of a 10-bit decode.
    match log2 {
        2 => {
            let mut residual = [[0; 4]; 4];
            inverse4::<High, i32>(passes, input, 4, residual.as_flattened_mut());
            add_clipped(dst, 4, residual.as_flattened(), depth);
        }
        3 => add_high::<8, T>(passes, input, dst, 5, depth),
        4 => add_high::<16, T>(passes, input, dst, 6, depth),
        _ => add_high::<32, T>(passes, input, dst, 6, depth),
    }
}

/// The `N`x`N` high-bit-depth inverse with final shift `shift`, added.
fn add_high<const N: usize, T: Sample>(
    passes: (Vp9Pass, Vp9Pass),
    input: &[i32],
    dst: (&mut [T], usize),
    shift: u32,
    depth: BitDepth,
) {
    let mut residual = [[0; N]; N];
    inverse::<N, High, i32>(passes, input, residual.as_flattened_mut(), shift);
    add_clipped(dst, N, residual.as_flattened(), depth);
}

/// Add `residual` (`side` per row) to the block at the start of `dst`,
/// clipped to `depth`.
fn add_clipped<T: Sample>(
    (dst, dst_stride): (&mut [T], usize),
    side: usize,
    residual: &[i32],
    depth: BitDepth,
) {
    for (row, values) in dst.chunks_mut(dst_stride).zip(residual.chunks_exact(side)) {
        for (sample, &value) in row[..side].iter_mut().zip(values.iter()) {
            *sample = depth.clip_to(sample.to_i32() + value);
        }
    }
}

/// `cospi_8_64`, `cospi_16_64`, `cospi_24_64` and `sinpi_k_9` (14 bits).
const COS8: i32 = 15137;
const COS16: i32 = 11585;
const COS24: i32 = 6270;
const SIN1: i32 = 5283;
const SIN2: i32 = 9929;
const SIN3: i32 = 13377;
const SIN4: i32 = 15212;
/// The 4x4 inverse's final rounding shift.
const SHIFT4: u32 = 4;

/// `dct_const_round_shift`.
fn round14(value: i64) -> i64 {
    (value + (1 << 13)) >> 14
}

/// `idct4_c` / `vpx_highbd_idct4_c`, stages wrapped by `D`.
fn idct4<D: InverseDepth>(line: [i32; 4]) -> [i32; 4] {
    let [a, b, c, d] = line.map(i64::from);
    let wrap = D::wrap;
    let even0 = wrap(round14((a + c) * i64::from(COS16)));
    let even1 = wrap(round14((a - c) * i64::from(COS16)));
    let odd0 = wrap(round14(b * i64::from(COS24) - d * i64::from(COS8)));
    let odd1 = wrap(round14(b * i64::from(COS8) + d * i64::from(COS24)));
    [
        wrap(even0 + odd1),
        wrap(even1 + odd0),
        wrap(even1 - odd0),
        wrap(even0 - odd1),
    ]
    .map(|v| v as i32)
}

/// `iadst4_c` / `vpx_highbd_iadst4_c`, stages wrapped by `D`.
fn iadst4<D: InverseDepth>(line: [i32; 4]) -> [i32; 4] {
    let [a, b, c, d] = line.map(i64::from);
    let wrap = D::wrap;
    let (sin1, sin2, sin3, sin4) = (
        i64::from(SIN1),
        i64::from(SIN2),
        i64::from(SIN3),
        i64::from(SIN4),
    );
    let difference = wrap(a - c + d);
    let first = sin1 * a + sin4 * c + sin2 * d;
    let second = sin2 * a - sin1 * c - sin4 * d;
    let middle = sin3 * b;
    [
        wrap(round14(first + middle)),
        wrap(round14(second + middle)),
        wrap(round14(sin3 * difference)),
        wrap(round14(first + second - middle)),
    ]
    .map(|v| v as i32)
}

fn line4<D: InverseDepth>(pass: Vp9Pass, line: [i32; 4]) -> [i32; 4] {
    match pass {
        Vp9Pass::Dct => idct4::<D>(line),
        Vp9Pass::Adst => iadst4::<D>(line),
    }
}

/// The 4x4 inverse of raster `input` into residuals, rows from
/// `nonzero_rows` on being zero (`vp9_iht4x4_16_add_c` before the add).
fn inverse4<D: InverseDepth, C: Copy + Into<i32>>(
    (columns, rows): (Vp9Pass, Vp9Pass),
    input: &[C],
    nonzero_rows: usize,
    residual: &mut [i32],
) {
    const SIDE: usize = 4;
    let round = |value: i32| (value + (1 << (SHIFT4 - 1))) >> SHIFT4;
    let mut temp = [[0i32; SIDE]; SIDE];
    for (y, row) in temp.iter_mut().enumerate().take(nonzero_rows.min(SIDE)) {
        *row = line4::<D>(rows, std::array::from_fn(|x| input[y * SIDE + x].into()));
    }
    for x in 0..SIDE {
        let column = line4::<D>(columns, std::array::from_fn(|y| temp[y][x]));
        for (y, value) in column.into_iter().enumerate() {
            residual[y * SIDE + x] = round(value);
        }
    }
}

/// The 4x4 inverse transform of raster `input` added to the 4x4 block at
/// the start of `dst` (rows `dst_stride` apart), clipped to 8 bits
/// (`vp9_iht4x4_16_add_c`, `vpx_idct4x4_16_add_c`). Rows from
/// `nonzero_rows` on must be zero; a DCT block whose only coefficient is
/// the DC adds one constant (`vpx_idct4x4_1_add_c`).
pub(crate) fn vp9_inverse4_add(
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i16],
    (nonzero_rows, nonzero_columns): (usize, usize),
    dst: &mut [u8],
    dst_stride: usize,
) {
    const SIDE: usize = 4;
    if (nonzero_rows, nonzero_columns) == (1, 1) && (columns, rows) == (Vp9Pass::Dct, Vp9Pass::Dct)
    {
        let round = |value: i64| (value + (1 << (SHIFT4 - 1))) >> SHIFT4;
        let wrap = Eight::wrap;
        let row = wrap(round14(i64::from(input[0]) * i64::from(COS16)));
        let value = round(wrap(round14(row * i64::from(COS16)))) as i32;
        for line in dst.chunks_mut(dst_stride).take(SIDE) {
            for sample in &mut line[..SIDE] {
                *sample = (i32::from(*sample) + value).clamp(0, 255) as u8;
            }
        }
        return;
    }
    let mut residual = [0; SIDE * SIDE];
    inverse4::<Eight, i16>((columns, rows), input, nonzero_rows, &mut residual);
    add_clipped((dst, dst_stride), SIDE, &residual, BitDepth::EIGHT);
}

/// VP9 8×8 forward DCT (see [`vp9_forward_transform`]).
#[must_use]
pub fn vp9_forward_dct8(input: &[i16; 64]) -> [i16; 64] {
    let mut output = [0; 64];
    vp9_forward_transform(3, Vp9Pass::Dct, Vp9Pass::Dct, input, &mut output);
    output
}

/// VP9 8×8 inverse DCT (see [`vp9_inverse_transform`]).
#[must_use]
pub fn vp9_inverse_dct8(input: &[i16; 64]) -> [i16; 64] {
    let mut output = [0; 64];
    vp9_inverse_transform(3, Vp9Pass::Dct, Vp9Pass::Dct, input, &mut output);
    output
}
