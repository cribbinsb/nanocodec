//! The self-guided row kernels written once over [`SgrLanes`], eight
//! 32-bit lanes, which each ISA module implements with its vector type and
//! instantiates inside its `#[target_feature]` functions
//! ([`sgr_kernels`]). The arithmetic is the scalar references' in
//! [`super::self_guided`], term for term; the only change of form is the
//! `A` lookup (`av1_x_by_xplus1`), computed as the spec's quotient
//! `((z << 8) + z / 2) / (z + 1)` in single precision: both operands are
//! exact integers below 2^17 and a fractional part is at least
//! `1 / (z + 1) ≥ 2^-8`, far above the quotient's rounding error
//! (`2^-16`), so truncating the float quotient is the integer division.
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the [`super::kernels`] contract); every access indexes a slice.

#![allow(
    clippy::many_single_char_names,
    reason = "the spec's and dav1d's sample, tap and statistic names"
)]
#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use super::self_guided::{
    BLEND_SHIFT, MTABLE_BITS, ONE_ROW_SHIFT, PRJ_ONE, RECIP_BITS, SGR_BITS, SOURCE_SHIFT,
    TWO_ROW_SHIFT, Z_MAX, one_by_x,
};

/// Lanes of every [`SgrLanes`] vector.
pub(super) const LANES: usize = 8;

/// Eight signed 32-bit lanes and the operations the self-guided kernels
/// need. Arithmetic wraps; shifts take their count as a constant.
pub(super) trait SgrLanes: Copy {
    /// Every lane `value`.
    fn splat(value: i32) -> Self;
    /// `src[..8]`.
    fn load(src: &[i32]) -> Self;
    /// `src[..8]`, zero-extended.
    fn load_u8(src: &[u8]) -> Self;
    /// Store into `dst[..8]`.
    fn store(self, dst: &mut [i32]);
    /// Store `clamp(lane, 0, 255)` into `dst[..8]`.
    fn store_u8_sat(self, dst: &mut [u8]);
    /// Lane-wise wrapping add.
    fn add(self, other: Self) -> Self;
    /// Lane-wise wrapping subtract.
    fn sub(self, other: Self) -> Self;
    /// Lane-wise wrapping multiply (low 32 bits).
    fn mul(self, other: Self) -> Self;
    /// Lane-wise left shift.
    fn shl<const BITS: i32>(self) -> Self;
    /// Lane-wise logical right shift (the lanes as `u32`).
    fn shr<const BITS: i32>(self) -> Self;
    /// Lane-wise arithmetic right shift.
    fn sra<const BITS: i32>(self) -> Self;
    /// Lane-wise signed minimum.
    fn min(self, other: Self) -> Self;
    /// Lane-wise signed maximum.
    fn max(self, other: Self) -> Self;
    /// All ones (-1) where the lanes are equal, else 0.
    fn eq(self, other: Self) -> Self;
    /// `floor(self / den)` for `0 <= self < 2^24` and `0 < den < 2^24`,
    /// through single precision.
    fn div_floor(self, den: Self) -> Self;
}

/// `5 · (x + y) + x`: the radius-2 weights 6 (`x`) and 5 (`y`) without a
/// multiply.
#[inline(always)]
fn six_five<V: SgrLanes>(x: V, y: V) -> V {
    let sum = x.add(y);
    sum.shl::<2>().add(sum).add(x)
}

/// `4 · (x + y) − y`: the radius-1 weights 4 (`x`) and 3 (`y`).
#[inline(always)]
fn four_three<V: SgrLanes>(x: V, y: V) -> V {
    x.add(y).shl::<2>().sub(y)
}

/// Columns `j..j + 8` of the filtered row: `(a · src + b + round) >> shift`.
#[inline(always)]
fn filtered<V: SgrLanes, const SHIFT: i32>(a: V, b: V, src: &[u8], j: usize) -> V {
    a.mul(V::load_u8(&src[j..]))
        .add(b)
        .add(V::splat(1 << (SHIFT - 1)))
        .sra::<SHIFT>()
}

/// Columns `j..j + 8` of the radius-2 weights between two box rows
/// (written as functions, not closures, so that they inline into the
/// kernels' `#[target_feature]` bodies).
#[inline(always)]
fn weigh_between<V: SgrLanes>(v: [&[i32]; 2], j: usize) -> V {
    let straight = V::load(&v[0][j + 1..]).add(V::load(&v[1][j + 1..]));
    let diagonal = V::load(&v[0][j..])
        .add(V::load(&v[0][j + 2..]))
        .add(V::load(&v[1][j..]))
        .add(V::load(&v[1][j + 2..]));
    six_five(straight, diagonal)
}

/// Columns `j..j + 8` of the radius-2 weights on a box row.
#[inline(always)]
fn weigh_on<V: SgrLanes>(v: &[i32], j: usize) -> V {
    six_five(
        V::load(&v[j + 1..]),
        V::load(&v[j..]).add(V::load(&v[j + 2..])),
    )
}

/// Columns `j..j + 8` of the radius-1 weights.
#[inline(always)]
fn weigh_around<V: SgrLanes>(v: [&[i32]; 3], j: usize) -> V {
    let cross = V::load(&v[1][j..])
        .add(V::load(&v[1][j + 1..]))
        .add(V::load(&v[1][j + 2..]))
        .add(V::load(&v[0][j + 1..]))
        .add(V::load(&v[2][j + 1..]));
    let diagonal = V::load(&v[0][j..])
        .add(V::load(&v[0][j + 2..]))
        .add(V::load(&v[2][j..]))
        .add(V::load(&v[2][j + 2..]));
    four_three(cross, diagonal)
}

/// Horizontal box sums of radius `R` (as `self_guided::box_sums`).
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(super) unsafe fn box_sums<V: SgrLanes, const R: usize>(
    src: &[u8],
    sum: &mut [i32],
    squares: &mut [i32],
    n: usize,
) {
    for j in (0..n).step_by(LANES) {
        let mut s = V::splat(0);
        let mut q = V::splat(0);
        for k in 0..=2 * R {
            let p = V::load_u8(&src[j + k..]);
            s = s.add(p);
            q = q.add(p.mul(p));
        }
        s.store(&mut sum[j..]);
        q.store(&mut squares[j..]);
    }
}

/// Box statistics from `N` rows of horizontal sums (as
/// `self_guided::box_ab`).
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(super) unsafe fn box_ab<V: SgrLanes, const N: usize>(
    sums: [&[i32]; N],
    squares: [&[i32]; N],
    scale: u32,
    (a, b): (&mut [i32], &mut [i32]),
    n: usize,
) {
    let count = (N * N) as u32;
    let (count_v, recip) = (V::splat(count as i32), V::splat(one_by_x(count) as i32));
    let scale = V::splat(scale as i32);
    let (zero, one, z_max) = (V::splat(0), V::splat(1), V::splat(Z_MAX));
    let unit = V::splat(1 << SGR_BITS);
    for j in (0..n).step_by(LANES) {
        let mut sum = V::load(&sums[0][j..]);
        let mut square = V::load(&squares[0][j..]);
        for row in 1..N {
            sum = sum.add(V::load(&sums[row][j..]));
            square = square.add(V::load(&squares[row][j..]));
        }
        let p = square.mul(count_v).sub(sum.mul(sum)).max(zero);
        // p · s < 2^32 for every set (the scales are chosen so), so the
        // logical shift of the wrapped product is exact.
        let z = p
            .mul(scale)
            .add(V::splat(1 << (MTABLE_BITS - 1)))
            .shr::<{ MTABLE_BITS as i32 }>()
            .min(z_max);
        // x_by_xplus1: the quotient, 0 -> 1, and 255 -> 256.
        let factor = z
            .shl::<{ SGR_BITS as i32 }>()
            .add(z.shr::<1>())
            .div_floor(z.add(one))
            .max(one)
            .sub(z.eq(z_max));
        let mean = unit
            .sub(factor)
            .mul(sum)
            .mul(recip)
            .add(V::splat(1 << (RECIP_BITS - 1)))
            .shr::<{ RECIP_BITS as i32 }>();
        factor.store(&mut a[j..]);
        mean.store(&mut b[j..]);
    }
}

/// A radius-2 output row between two box rows (as
/// `self_guided::filter_between`).
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(super) unsafe fn filter_between<V: SgrLanes>(
    a: [&[i32]; 2],
    b: [&[i32]; 2],
    src: &[u8],
    out: &mut [i32],
    n: usize,
) {
    for j in (0..n).step_by(LANES) {
        filtered::<V, { TWO_ROW_SHIFT as i32 }>(weigh_between(a, j), weigh_between(b, j), src, j)
            .store(&mut out[j..]);
    }
}

/// A radius-2 output row on a box row (as `self_guided::filter_on`).
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(super) unsafe fn filter_on<V: SgrLanes>(
    a: &[i32],
    b: &[i32],
    src: &[u8],
    out: &mut [i32],
    n: usize,
) {
    for j in (0..n).step_by(LANES) {
        filtered::<V, { ONE_ROW_SHIFT as i32 }>(weigh_on(a, j), weigh_on(b, j), src, j)
            .store(&mut out[j..]);
    }
}

/// A radius-1 output row (as `self_guided::filter_around`).
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(super) unsafe fn filter_around<V: SgrLanes>(
    a: [&[i32]; 3],
    b: [&[i32]; 3],
    src: &[u8],
    out: &mut [i32],
    n: usize,
) {
    for j in (0..n).step_by(LANES) {
        filtered::<V, { TWO_ROW_SHIFT as i32 }>(weigh_around(a, j), weigh_around(b, j), src, j)
            .store(&mut out[j..]);
    }
}

/// The projection onto all of `dst` (as `self_guided::blend`).
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(super) unsafe fn blend<V: SgrLanes>(
    src: &[u8],
    filtered: [&[i32]; 2],
    weights: [i32; 2],
    dst: &mut [u8],
) {
    let n = dst.len();
    let (w0, w1) = (V::splat(weights[0]), V::splat(weights[1]));
    let round = V::splat(1 << (BLEND_SHIFT - 1));
    for j in (0..n).step_by(LANES) {
        let u = V::load_u8(&src[j..]).shl::<{ SOURCE_SHIFT as i32 }>();
        let v = u
            .mul(V::splat(PRJ_ONE))
            .add(w0.mul(V::load(&filtered[0][j..]).sub(u)))
            .add(w1.mul(V::load(&filtered[1][j..]).sub(u)))
            .add(round)
            .sra::<{ BLEND_SHIFT as i32 }>();
        if j + LANES <= n {
            v.store_u8_sat(&mut dst[j..j + LANES]);
        } else {
            let mut tail = [0u8; LANES];
            v.store_u8_sat(&mut tail);
            dst[j..].copy_from_slice(&tail[..n - j]);
        }
    }
}

/// The self-guided entries of an ISA's kernel table, for its
/// [`SgrLanes`] type `$lanes`, each a `$feature` function.
macro_rules! sgr_kernels {
    ($(#[$feature:meta])*; $lanes:ty) => {
        $(#[$feature])*
        unsafe fn box_sums3(src: &[u8], sum: &mut [i32], squares: &mut [i32], n: usize) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::box_sums::<$lanes, 1>(src, sum, squares, n) }
        }

        $(#[$feature])*
        unsafe fn box_sums5(src: &[u8], sum: &mut [i32], squares: &mut [i32], n: usize) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::box_sums::<$lanes, 2>(src, sum, squares, n) }
        }

        $(#[$feature])*
        unsafe fn box_ab3(
            sums: [&[i32]; 3],
            squares: [&[i32]; 3],
            scale: u32,
            ab: (&mut [i32], &mut [i32]),
            n: usize,
        ) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::box_ab::<$lanes, 3>(sums, squares, scale, ab, n) }
        }

        $(#[$feature])*
        unsafe fn box_ab5(
            sums: [&[i32]; 5],
            squares: [&[i32]; 5],
            scale: u32,
            ab: (&mut [i32], &mut [i32]),
            n: usize,
        ) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::box_ab::<$lanes, 5>(sums, squares, scale, ab, n) }
        }

        $(#[$feature])*
        unsafe fn between(a: [&[i32]; 2], b: [&[i32]; 2], src: &[u8], out: &mut [i32], n: usize) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::filter_between::<$lanes>(a, b, src, out, n) }
        }

        $(#[$feature])*
        unsafe fn on(a: &[i32], b: &[i32], src: &[u8], out: &mut [i32], n: usize) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::filter_on::<$lanes>(a, b, src, out, n) }
        }

        $(#[$feature])*
        unsafe fn around(a: [&[i32]; 3], b: [&[i32]; 3], src: &[u8], out: &mut [i32], n: usize) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::filter_around::<$lanes>(a, b, src, out, n) }
        }

        $(#[$feature])*
        unsafe fn blend(src: &[u8], filtered: [&[i32]; 2], weights: [i32; 2], dst: &mut [u8]) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::blend::<$lanes>(src, filtered, weights, dst) }
        }
    };
}

pub(super) use sgr_kernels;
