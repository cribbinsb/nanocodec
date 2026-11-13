//! VP9 inverse transforms with eight wrapping i16 lanes per butterfly.
//! The unchanged `vp9_txfm` functions are the oracle. Like libvpx's NEON
//! and SSE kernels (BSD-3-Clause; `THIRD_PARTY.md`), rotations widen before
//! rounding and the separable passes transpose eight transforms together.
//! Unlike a saturating pack, narrowing preserves VP9's stage wrapping even
//! for arbitrary i32 input coefficients after their initial i16 truncation.

#![allow(clippy::inline_always)]

use crate::Vp9Pass;
use crate::txfm::{dct, inverse_lanes::InverseLanes, sparse_lanes::Sparse};

mod adst;
#[cfg(target_arch = "x86_64")]
pub(super) mod avx2;
#[cfg(target_arch = "aarch64")]
pub(super) mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) mod simd128;

const LANES: usize = 8;
const COS_BITS: u32 = 14;
const COSINE: [i32; 64] = {
    let mut values = [0; 64];
    let mut i = 0;
    while i < 32 {
        values[2 * i] = crate::vp9_txfm::VP9_COSPI_64[i];
        i += 1;
    }
    values
};

/// `2 * w` when it fits `i16`. Then `round_shift(w * a, 14)` is one
/// rounding high-half multiply by it, `(a * 2w + 2^14) >> 15` (libvpx's
/// partial inverses multiply by doubled cosines with `_mm_mulhrs_epi16` /
/// `vqrdmulhq_n_s16`): exact for every `i16` input, and the result fits
/// `i16`, so it equals the wrapped wide product. Weights are constants, so
/// the choice folds at compile time.
const fn doubled_weight(w: i32) -> Option<i16> {
    if 2 * w > i16::MAX as i32 || 2 * w < -(i16::MAX as i32) {
        None
    } else {
        Some((2 * w) as i16)
    }
}

/// Wide products may wrap modulo 2^32: after rounding by 14 and truncating
/// to i16, the discarded multiple is 2^18, itself a multiple of 2^16.
/// This is exact for the ADST's four-product rotations even at extremes.
trait Products: Copy {
    type Narrow: Lanes<Wide = Self>;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn narrow(self) -> Self::Narrow;
    fn round14(self) -> Self::Narrow;
}

trait Lanes: InverseLanes {
    type Wide: Products<Narrow = Self>;
    fn wide(self, weight: i32) -> Self::Wide;
    fn products(w0: i32, a: Self, w1: i32, b: Self) -> Self::Wide;
    fn load_i32_wrapping(src: &[i32]) -> Self;
    fn load_i16(src: &[i16]) -> Self;
    fn store_i16(self, dst: &mut [i16]);
    fn store_shifted(self, dst: &mut [i32], shift: u32);
    /// Round by `shift` and add to the eight samples at `dst`, clipping to
    /// 8 bits.
    ///
    /// # Safety
    ///
    /// `dst` must point to eight readable and writable bytes.
    unsafe fn add_shifted(self, dst: *mut u8, shift: u32);
    fn zip_lo(self, other: Self) -> Self;
    fn zip_hi(self, other: Self) -> Self;
}

#[inline(always)]
fn transpose<V: Lanes>(mut rows: [V; LANES]) -> [V; LANES] {
    for _ in 0..3 {
        let mut next = rows;
        for i in 0..LANES / 2 {
            next[2 * i] = rows[i].zip_lo(rows[i + LANES / 2]);
            next[2 * i + 1] = rows[i].zip_hi(rows[i + LANES / 2]);
        }
        rows = next;
    }
    rows
}

#[inline(always)]
fn line<V: Lanes, const N: usize>(kind: Vp9Pass, input: &[V; N], output: &mut [V; N]) {
    match (kind, N) {
        (Vp9Pass::Dct, 8) => dct::idct8(
            input.first_chunk().unwrap(),
            output.first_chunk_mut().unwrap(),
            &COSINE,
            COS_BITS,
        ),
        (Vp9Pass::Dct, 16) => dct::idct16(
            input.first_chunk().unwrap(),
            output.first_chunk_mut().unwrap(),
            &COSINE,
            COS_BITS,
        ),
        (Vp9Pass::Dct, 32) => dct::idct32(
            input.first_chunk().unwrap(),
            output.first_chunk_mut().unwrap(),
            &COSINE,
            COS_BITS,
        ),
        (Vp9Pass::Adst, 8) => adst::iadst8(
            input.first_chunk().unwrap(),
            output.first_chunk_mut().unwrap(),
        ),
        (Vp9Pass::Adst, 16) => adst::iadst16(
            input.first_chunk().unwrap(),
            output.first_chunk_mut().unwrap(),
        ),
        _ => unreachable!("validated VP9 transform size/pass"),
    }
}

/// The row pass into `temp` (i16, row-major). Rows from `nonzero_rows` on
/// hold only zero coefficients and transform to zero, so their groups of
/// eight are cleared instead of transformed.
#[inline(always)]
fn first_pass<V: Lanes, const N: usize>(
    rows: Vp9Pass,
    input: &[i32],
    nonzero_rows: usize,
    temp: &mut [i16],
) {
    for first in (0..N).step_by(LANES) {
        if first >= nonzero_rows {
            temp[first * N..(first + LANES) * N].fill(0);
            continue;
        }
        let mut x = [V::splat(0); N];
        for col in (0..N).step_by(LANES) {
            x[col..col + LANES].copy_from_slice(&load_tile::<V, N>(input, first, col));
        }
        let mut y = x;
        line(rows, &x, &mut y);
        for col in (0..N).step_by(LANES) {
            let mut tile = [V::splat(0); LANES];
            tile.copy_from_slice(&y[col..col + LANES]);
            for (row, value) in transpose(tile).into_iter().enumerate() {
                value.store_i16(&mut temp[(first + row) * N + col..]);
            }
        }
    }
}

/// The column pass of eight columns from `col`: rows of `N` results.
#[inline(always)]
fn column_pass<V: Lanes, const N: usize>(columns: Vp9Pass, temp: &[i16], col: usize) -> [V; N] {
    let mut x = [V::splat(0); N];
    for (row, lanes) in x.iter_mut().enumerate() {
        *lanes = V::load_i16(&temp[row * N + col..]);
    }
    let mut y = x;
    line(columns, &x, &mut y);
    y
}

/// Final rounding shift of the inverse transforms (`ROUND_POWER_OF_TWO`).
const fn output_shift(n: usize) -> u32 {
    if n == 8 { 5 } else { 6 }
}

#[inline(always)]
fn transform<V: Lanes, const N: usize>(
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i32],
    output: &mut [i32],
) {
    let input = &input[..N * N];
    let output = &mut output[..N * N];
    // Keep scratch proportional to this transform: 16×16 must not clear a
    // 32×32 buffer. Every entry is written by the first pass before reuse.
    let mut temp = [[0_i16; N]; N];
    let temp = temp.as_flattened_mut();
    first_pass::<V, N>(rows, input, N, temp);
    for col in (0..N).step_by(LANES) {
        let y = column_pass::<V, N>(columns, temp, col);
        for (row, value) in y.into_iter().enumerate() {
            value.store_shifted(&mut output[row * N + col..], output_shift(N));
        }
    }
}

/// The validated `N`x`N` destination block at the start of `dst` as a row
/// pointer: one up-front check instead of one per eight-sample add.
#[inline(always)]
fn block_rows<const N: usize>(dst: &mut [u8], dst_stride: usize) -> *mut u8 {
    assert!(dst_stride >= N && dst.len() >= (N - 1) * dst_stride + N);
    dst.as_mut_ptr()
}

/// Add rows `y` (eight columns from `col`) of a column pass into the block.
#[inline(always)]
fn add_rows<V: Lanes, const N: usize>(
    y: [V; N],
    (block, dst_stride): (*mut u8, usize),
    col: usize,
) {
    for (row, value) in y.into_iter().enumerate() {
        // SAFETY: `block` covers the N x N block checked by `block_rows`;
        // `col + 8 <= N`.
        unsafe {
            value.add_shifted(block.add(row * dst_stride + col), output_shift(N));
        }
    }
}

/// [`transform`] added into the picture block at the start of `dst`, for
/// the ADST combinations (rows from `nonzero_rows` on are skipped).
#[inline(always)]
fn transform_add<V: Lanes, const N: usize>(
    (columns, rows): (Vp9Pass, Vp9Pass),
    input: &[i32],
    nonzero_rows: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    let input = &input[..N * N];
    let block = block_rows::<N>(dst, dst_stride);
    let mut temp = [[0_i16; N]; N];
    let temp = temp.as_flattened_mut();
    first_pass::<V, N>(rows, input, nonzero_rows, temp);
    for col in (0..N).step_by(LANES) {
        add_rows(
            column_pass::<V, N>(columns, temp, col),
            (block, dst_stride),
            col,
        );
    }
}

/// The eight rows from `first` of an `N`-wide raster block, columns
/// `col..col + 8`, transposed: element `i` holds column `col + i`.
#[inline(always)]
fn load_tile<V: Lanes, const N: usize>(input: &[i32], first: usize, col: usize) -> [V; LANES] {
    let mut tile = [V::splat(0); LANES];
    for (row, lanes) in tile.iter_mut().enumerate() {
        *lanes = V::load_i32_wrapping(&input[(first + row) * N + col..]);
    }
    transpose(tile)
}

/// The 2-D inverse DCT of an `N`x`N` block whose nonzero coefficients lie
/// in its top-left `K`x`K` corner, added into `dst` (libvpx's eob classes:
/// `vpx_idct8x8_12_add`, `vpx_idct16x16_10_add`/`_38_add`,
/// `vpx_idct32x32_34_add`/`_135_add` and the full transforms). `G` is the
/// number of eight-row groups the corner spans (`ceil(K / 8)`).
///
/// The row pass transforms only those groups, with the inputs past `K`
/// structural zeros ([`corner_dct`]), and keeps its outputs in vectors
/// (lanes are rows); the column pass transposes them back and runs each
/// column transform with the rows past `K` zero, adding straight into `dst`.
#[inline(always)]
fn dct_add<V: Lanes, const N: usize, const K: usize, const G: usize>(
    input: &[i32],
    dst: &mut [u8],
    dst_stride: usize,
) {
    let input = &input[..N * N];
    let block = block_rows::<N>(dst, dst_stride);
    let mut row_outputs = [[V::splat(0); N]; G];
    for (group, outputs) in row_outputs.iter_mut().enumerate() {
        let mut x = [V::splat(0); N];
        for col in (0..K).step_by(LANES) {
            x[col..col + LANES].copy_from_slice(&load_tile::<V, N>(input, group * LANES, col));
        }
        *outputs = corner_dct::<V, N, K>(&x);
    }
    for col in (0..N).step_by(LANES) {
        let mut x = [V::splat(0); N];
        for (group, outputs) in row_outputs.iter().enumerate() {
            let mut tile = [V::splat(0); LANES];
            tile.copy_from_slice(&outputs[col..col + LANES]);
            x[group * LANES..(group + 1) * LANES].copy_from_slice(&transpose(tile));
        }
        add_rows(corner_dct::<V, N, K>(&x), (block, dst_stride), col);
    }
}

/// The `N`-point inverse DCT of `input`, whose elements from `K` on are
/// zero: they enter the graph as structural zeros ([`Sparse`]), so the
/// operations they feed fold away. The conversions are tiny constant loops,
/// kept apart from the loads so that they unroll and every discriminant is
/// a compile-time constant.
#[inline(always)]
fn corner_dct<V: Lanes, const N: usize, const K: usize>(input: &[V; N]) -> [V; N] {
    let mut x = [Sparse::Zero; N];
    for (sparse, &lanes) in x.iter_mut().zip(input).take(K) {
        *sparse = Sparse::Lanes(lanes);
    }
    let mut y = [Sparse::Zero; N];
    match N {
        8 => dct::idct8(
            x.first_chunk().unwrap(),
            y.first_chunk_mut().unwrap(),
            &COSINE,
            COS_BITS,
        ),
        16 => dct::idct16(
            x.first_chunk().unwrap(),
            y.first_chunk_mut().unwrap(),
            &COSINE,
            COS_BITS,
        ),
        32 => dct::idct32(
            x.first_chunk().unwrap(),
            y.first_chunk_mut().unwrap(),
            &COSINE,
            COS_BITS,
        ),
        _ => unreachable!("validated VP9 transform size"),
    }
    y.map(Sparse::get)
}

/// A DC-only inverse DCT: every residual is the same constant, two
/// rotations of the DC coefficient, added with byte saturation (libvpx
/// `vpx_idctNxN_1_add`: `clip(p + r)` is `p` saturating-plus `min(r, 255)`,
/// or saturating-minus for a negative `r`). Fixed-width rows vectorise.
#[inline(always)]
fn dc_add<const N: usize>(dc: i32, dst: &mut [u8], dst_stride: usize) {
    let wrap = |value: i32| i32::from(value as i16);
    let rotate = |value: i32| wrap((value * COSINE[32] + (1 << (COS_BITS - 1))) >> COS_BITS);
    let shift = output_shift(N);
    let residual = (rotate(rotate(wrap(dc))) + (1 << (shift - 1))) >> shift;
    let magnitude = residual.unsigned_abs().min(u32::from(u8::MAX)) as u8;
    for row in dst.chunks_mut(dst_stride).take(N) {
        let row: &mut [u8; N] = (&mut row[..N]).try_into().unwrap();
        if residual >= 0 {
            for sample in row {
                *sample = sample.saturating_add(magnitude);
            }
        } else {
            for sample in row {
                *sample = sample.saturating_sub(magnitude);
            }
        }
    }
}

/// The inverse added into `dst`, choosing the smallest zero region that
/// holds the `(rows, columns)` extent of the nonzero coefficients.
#[inline(always)]
fn inverse_add<V: Lanes>(
    log2: u8,
    passes: (Vp9Pass, Vp9Pass),
    input: &[i32],
    (rows, columns): (usize, usize),
    (dst, dst_stride): (&mut [u8], usize),
) {
    let corner = rows.max(columns);
    if passes != (Vp9Pass::Dct, Vp9Pass::Dct) {
        match log2 {
            3 => transform_add::<V, 8>(passes, input, rows, dst, dst_stride),
            4 => transform_add::<V, 16>(passes, input, rows, dst, dst_stride),
            _ => unreachable!("validated VP9 ADST size"),
        }
        return;
    }
    match (log2, corner) {
        (3, ..=1) => dc_add::<8>(input[0], dst, dst_stride),
        (4, ..=1) => dc_add::<16>(input[0], dst, dst_stride),
        (5, ..=1) => dc_add::<32>(input[0], dst, dst_stride),
        (3, ..=4) => dct_add::<V, 8, 4, 1>(input, dst, dst_stride),
        (3, _) => dct_add::<V, 8, 8, 1>(input, dst, dst_stride),
        (4, ..=4) => dct_add::<V, 16, 4, 1>(input, dst, dst_stride),
        (4, ..=8) => dct_add::<V, 16, 8, 1>(input, dst, dst_stride),
        (4, _) => dct_add::<V, 16, 16, 2>(input, dst, dst_stride),
        (5, ..=8) => dct_add::<V, 32, 8, 1>(input, dst, dst_stride),
        (5, ..=16) => dct_add::<V, 32, 16, 2>(input, dst, dst_stride),
        (5, _) => dct_add::<V, 32, 32, 4>(input, dst, dst_stride),
        _ => unreachable!("validated VP9 transform size"),
    }
}

#[inline(always)]
fn inverse<V: Lanes>(log2: u8, columns: Vp9Pass, rows: Vp9Pass, input: &[i32], output: &mut [i32]) {
    match log2 {
        3 => transform::<V, 8>(columns, rows, input, output),
        4 => transform::<V, 16>(columns, rows, input, output),
        5 => transform::<V, 32>(columns, rows, input, output),
        _ => unreachable!("validated VP9 transform size"),
    }
}
