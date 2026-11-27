//! The self-guided row kernels written once over [`SgrLanes`] (eight or
//! sixteen 32-bit lanes), shared across `u8` (8-bit) and `u16` (10- and
//! 12-bit) samples.
//!
//! Horizontal box sums (`<= 5 * 4095 = 20475`) and `A` (`1..=256`) fit `i16`
//! at every AV1 bit depth; filtered rows fit `i16` at 8 bits (`<= 4084`) and
//! use `i32` (`LaneRow`) at high bit depth (`<= 65584` at 12 bits, avoiding
//! intermediate pack/unpack instructions).
//!
//! Attribution: reciprocal `A` factor and ring schedule follow dav1d 1.5.4
//! `src/looprestoration_tmpl.c` and `src/x86/looprestoration*_avx2.asm`
//! (BSD-2-Clause; see `THIRD_PARTY.md`).

#![allow(
    clippy::many_single_char_names,
    reason = "the spec's and dav1d's sample, tap and statistic names"
)]
#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use nc_common::{BitDepth, Sample};

use crate::simd::I16Lanes;

use super::AV1_LR_OVERREAD;
use super::self_guided::{
    BLEND_SHIFT, BoxRows, MTABLE_BITS, ONE_ROW_SHIFT, PRJ_ONE, ProjectionRows, RECIP_BITS,
    RadiusTwo, SGR_BITS, SOURCE_SHIFT, TWO_ROW_SHIFT, Z_MAX, one_by_x,
};

/// Default lane count of [`SgrLanes`] vectors (AVX2, NEON, SIMD128).
pub(super) const LANES: usize = 8;
/// Largest lane count of any [`SgrLanes`] vector (AVX-512).
const MAX_LANES: usize = 16;

/// Signed 32-bit lanes (`LANES` = 8 or 16) and the operations the self-guided
/// kernels need. Arithmetic wraps; shifts take their count as a constant.
pub(crate) trait SgrLanes: Copy {
    const LANES: usize = LANES;

    /// Every lane `value`.
    fn splat(value: i32) -> Self;
    /// `src[..Self::LANES]`.
    fn load(src: &[i32]) -> Self;
    /// `src[..Self::LANES]`, sign-extended.
    fn load_i16(src: &[i16]) -> Self;
    /// `src[..Self::LANES]`, zero-extended.
    fn load_u8(src: &[u8]) -> Self;
    /// `src[..Self::LANES]`, zero-extended (`u16` samples).
    #[inline(always)]
    fn load_u16(src: &[u16]) -> Self {
        let mut wide = [0i32; MAX_LANES];
        for (o, &s) in wide[..Self::LANES].iter_mut().zip(&src[..Self::LANES]) {
            *o = i32::from(s);
        }
        Self::load(&wide)
    }
    /// Store into `dst[..Self::LANES]`.
    fn store(self, dst: &mut [i32]);
    /// Store into `dst[..Self::LANES]` lanes that fit `i16` (saturating otherwise).
    fn store_i16(self, dst: &mut [i16]);
    /// Store `clamp(lane, 0, 255)` into `dst[..Self::LANES]`.
    fn store_u8_sat(self, dst: &mut [u8]);
    /// Store `clamp(lane, 0, max_val)` into `dst[..Self::LANES]`.
    #[inline(always)]
    fn store_u16_clip(self, max_val: i32, dst: &mut [u16]) {
        let mut wide = [0i32; MAX_LANES];
        self.store(&mut wide[..Self::LANES]);
        for (o, &v) in dst[..Self::LANES].iter_mut().zip(&wide[..Self::LANES]) {
            *o = v.clamp(0, max_val) as u16;
        }
    }
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
    /// Exact AV1 `x_by_xplus1` factor for `0 <= self <= Z_MAX`.
    #[inline(always)]
    fn sgr_factor(self) -> Self {
        let one = Self::splat(1);
        self.shl::<{ SGR_BITS as i32 }>()
            .add(self.shr::<1>())
            .div_floor(self.add(one))
            .max(one)
            .sub(self.eq(Self::splat(Z_MAX)))
    }
}

/// Reusable filtered-row storage: `i16` for 8-bit, `i32` for high bit depth.
#[derive(Debug, Default)]
pub(crate) struct FilteredScratch {
    narrow: [Vec<i16>; 2],
    wide: [Vec<i32>; 2],
}

/// A row element that loads into and stores from [`SgrLanes`].
pub(crate) trait LaneRow: Copy + Default {
    fn load<V: SgrLanes>(src: &[Self]) -> V;
    fn store<V: SgrLanes>(v: V, dst: &mut [Self]);
    fn to_i32(self) -> i32;
    fn from_i32(value: i32) -> Self;
    fn scratch_vecs(scratch: &mut FilteredScratch) -> &mut [Vec<Self>; 2];
}

impl LaneRow for i16 {
    #[inline(always)]
    fn load<V: SgrLanes>(src: &[Self]) -> V {
        V::load_i16(src)
    }
    #[inline(always)]
    fn store<V: SgrLanes>(v: V, dst: &mut [Self]) {
        v.store_i16(dst);
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }
    #[inline(always)]
    fn from_i32(value: i32) -> Self {
        value as i16
    }
    #[inline(always)]
    fn scratch_vecs(scratch: &mut FilteredScratch) -> &mut [Vec<Self>; 2] {
        &mut scratch.narrow
    }
}

impl LaneRow for i32 {
    #[inline(always)]
    fn load<V: SgrLanes>(src: &[Self]) -> V {
        V::load(src)
    }
    #[inline(always)]
    fn store<V: SgrLanes>(v: V, dst: &mut [Self]) {
        v.store(dst);
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self
    }
    #[inline(always)]
    fn from_i32(value: i32) -> Self {
        value
    }
    #[inline(always)]
    fn scratch_vecs(scratch: &mut FilteredScratch) -> &mut [Vec<Self>; 2] {
        &mut scratch.wide
    }
}

/// Sample adapter so `u8` and `u16` share one set of SGR lane kernels.
pub(super) trait SgrSample: Sample {
    type Filtered: LaneRow;
    fn load_sample<V: SgrLanes>(src: &[Self]) -> V;
    fn store_clipped<V: SgrLanes>(v: V, depth: BitDepth, dst: &mut [Self]);
}

impl SgrSample for u8 {
    type Filtered = i16;
    #[inline(always)]
    fn load_sample<V: SgrLanes>(src: &[u8]) -> V {
        V::load_u8(src)
    }
    #[inline(always)]
    fn store_clipped<V: SgrLanes>(v: V, _depth: BitDepth, dst: &mut [u8]) {
        v.store_u8_sat(dst);
    }
}

impl SgrSample for u16 {
    type Filtered = i32;
    #[inline(always)]
    fn load_sample<V: SgrLanes>(src: &[u16]) -> V {
        V::load_u16(src)
    }
    #[inline(always)]
    fn store_clipped<V: SgrLanes>(v: V, depth: BitDepth, dst: &mut [u16]) {
        v.store_u16_clip(depth.max(), dst);
    }
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

/// Columns `j..j + V::LANES` of the filtered row: `(a · src + b + round) >> shift`.
#[inline(always)]
fn filtered<V: SgrLanes, S: SgrSample, const SHIFT: i32>(a: V, b: V, src: &[S], j: usize) -> V {
    filter_sample::<V, SHIFT>(a, b, S::load_sample::<V>(&src[j..]))
}

#[inline(always)]
fn filter_sample<V: SgrLanes, const SHIFT: i32>(a: V, b: V, sample: V) -> V {
    a.mul(sample)
        .add(b)
        .add(V::splat(1 << (SHIFT - 1)))
        .sra::<SHIFT>()
}

#[inline(always)]
fn weigh_between<V: SgrLanes, T: LaneRow>(v: [&[T]; 2], j: usize) -> V {
    let straight = T::load::<V>(&v[0][j + 1..]).add(T::load(&v[1][j + 1..]));
    let diagonal = T::load::<V>(&v[0][j..])
        .add(T::load(&v[0][j + 2..]))
        .add(T::load(&v[1][j..]))
        .add(T::load(&v[1][j + 2..]));
    six_five(straight, diagonal)
}

#[inline(always)]
fn weigh_on<V: SgrLanes, T: LaneRow>(v: &[T], j: usize) -> V {
    six_five(
        T::load(&v[j + 1..]),
        T::load::<V>(&v[j..]).add(T::load(&v[j + 2..])),
    )
}

/// Keep each radius-two box row already weighted 5/6/5 in the high-depth
/// ring. Later output rows reuse it individually or add two weighted rows.
#[inline(always)]
pub(super) fn weight_radius_two<V: SgrLanes, W: I16Lanes>(a: &mut [i16], b: &mut [i32], n: usize) {
    let width = n.saturating_sub(2);
    // A is at most 256. Multiplying by the sum of the three weights
    // still fits a signed word.
    for col in (0..width).step_by(W::LANES) {
        let outer = W::load(&a[col..]).add(W::load(&a[col + 2..]));
        let centre = W::load(&a[col + 1..]);
        let factors = outer.mul(W::splat(5)).add(centre.mul(W::splat(6)));
        factors.store(&mut a[col..]);
    }
    // B retains dword precision.
    for col in (0..width).step_by(V::LANES) {
        let means = weigh_on::<V, _>(b, col);
        means.store(&mut b[col..]);
    }
}

/// Radius-one projections use weights 343 above and below, and 444
/// on the middle box row. Save both for each newly calculated row.
#[inline(always)]
pub(super) fn weight_radius_one<V: SgrLanes, W: I16Lanes>(
    (outer_a, centre_a): (&mut [i16], &mut [i16]),
    (outer_b, centre_b): (&mut [i32], &mut [i32]),
    n: usize,
) {
    let width = n.saturating_sub(2);
    for col in (0..width).step_by(W::LANES) {
        let side = W::load(&outer_a[col..]).add(W::load(&outer_a[col + 2..]));
        let middle = W::load(&outer_a[col + 1..]);
        let uniform = side.add(middle).mul(W::splat(4));
        uniform.store(&mut centre_a[col..]);
        uniform.sub(side).store(&mut outer_a[col..]);
    }
    for col in (0..width).step_by(V::LANES) {
        let side = V::load(&outer_b[col..]).add(V::load(&outer_b[col + 2..]));
        let middle = V::load(&outer_b[col + 1..]);
        let uniform = side.add(middle).shl::<2>();
        uniform.store(&mut centre_b[col..]);
        uniform.sub(side).store(&mut outer_b[col..]);
    }
}

#[inline(always)]
fn weigh_around<V: SgrLanes, T: LaneRow>(v: [&[T]; 3], j: usize) -> V {
    let cross = T::load::<V>(&v[1][j..])
        .add(T::load(&v[1][j + 1..]))
        .add(T::load(&v[1][j + 2..]))
        .add(T::load(&v[0][j + 1..]))
        .add(T::load(&v[2][j + 1..]));
    let diagonal = T::load::<V>(&v[0][j..])
        .add(T::load(&v[0][j + 2..]))
        .add(T::load(&v[2][j..]))
        .add(T::load(&v[2][j + 2..]));
    four_three(cross, diagonal)
}

/// Project the high-depth output from the A/B rings. Both radii
/// share one source load and keep their filtered samples in registers.
#[inline(always)]
pub(super) fn project<V: SgrLanes>(
    rows: ProjectionRows<'_>,
    src: &[u16],
    weights: [i32; 2],
    dst: &mut [u16],
    depth: BitDepth,
) {
    match rows.radius2 {
        RadiusTwo::Off => {
            project_with::<V, 0>(src, weights, dst, depth, ([&[]; 2], [&[]; 2]), rows.radius1);
        }
        RadiusTwo::On(a, b) => {
            project_with::<V, 1>(src, weights, dst, depth, ([a, a], [b, b]), rows.radius1);
        }
        RadiusTwo::Between(five) => {
            project_with::<V, 2>(src, weights, dst, depth, five, rows.radius1);
        }
    }
}

#[inline(always)]
fn project_with<V: SgrLanes, const FIVE: u8>(
    src: &[u16],
    weights: [i32; 2],
    dst: &mut [u16],
    depth: BitDepth,
    five: BoxRows<'_, 2>,
    three: Option<BoxRows<'_, 3>>,
) {
    if let Some(three) = three {
        project_lanes::<V, FIVE, true>(src, weights, dst, depth, five, three);
    } else {
        project_lanes::<V, FIVE, false>(src, weights, dst, depth, five, ([&[]; 3], [&[]; 3]));
    }
}

#[inline(always)]
fn project_lanes<V: SgrLanes, const FIVE: u8, const THREE: bool>(
    src: &[u16],
    weights: [i32; 2],
    dst: &mut [u16],
    depth: BitDepth,
    (five_a, five_b): BoxRows<'_, 2>,
    (three_a, three_b): BoxRows<'_, 3>,
) {
    let [w0, w1] = weights.map(V::splat);
    let round = V::splat(1 << (BLEND_SHIFT - 1));
    let centre = dst.len() + 2 + AV1_LR_OVERREAD;
    for j in (0..dst.len()).step_by(V::LANES) {
        let p = V::load_u16(&src[j..]);
        let u = p.shl::<{ SOURCE_SHIFT as i32 }>();
        let f0 = match FIVE {
            2 => filter_sample::<V, { TWO_ROW_SHIFT as i32 }>(
                V::load_i16(&five_a[0][j..]).add(V::load_i16(&five_a[1][j..])),
                V::load(&five_b[0][j..]).add(V::load(&five_b[1][j..])),
                p,
            ),
            1 => filter_sample::<V, { ONE_ROW_SHIFT as i32 }>(
                V::load_i16(&five_a[0][j..]),
                V::load(&five_b[0][j..]),
                p,
            ),
            _ => u,
        };
        let f1 = if THREE {
            filter_sample::<V, { TWO_ROW_SHIFT as i32 }>(
                V::load_i16(&three_a[0][j..])
                    .add(V::load_i16(&three_a[1][centre + j..]))
                    .add(V::load_i16(&three_a[2][j..])),
                V::load(&three_b[0][j..])
                    .add(V::load(&three_b[1][centre + j..]))
                    .add(V::load(&three_b[2][j..])),
                p,
            )
        } else {
            u
        };
        let result = p.add(
            w0.mul(f0.sub(u))
                .add(w1.mul(f1.sub(u)))
                .add(round)
                .sra::<{ BLEND_SHIFT as i32 }>(),
        );
        let tail = &mut dst[j..];
        if tail.len() >= V::LANES {
            result.store_u16_clip(depth.max(), &mut tail[..V::LANES]);
        } else {
            let mut pad = [0; MAX_LANES];
            result.store_u16_clip(depth.max(), &mut pad[..V::LANES]);
            tail.copy_from_slice(&pad[..tail.len()]);
        }
    }
}

/// Horizontal box sums of radius `R` for any [`SgrSample`] (`u8` or `u16`).
#[inline(always)]
pub(super) unsafe fn box_sums<V: SgrLanes, S: SgrSample, const R: usize>(
    src: &[S],
    sum: &mut [i16],
    squares: &mut [i32],
    n: usize,
) {
    for j in (0..n).step_by(V::LANES) {
        let mut s = V::splat(0);
        let mut q = V::splat(0);
        for k in 0..=2 * R {
            let p = S::load_sample::<V>(&src[j + k..]);
            s = s.add(p);
            q = q.add(p.mul(p));
        }
        s.store_i16(&mut sum[j..]);
        q.store(&mut squares[j..]);
    }
}

/// Box statistics from `N` rows of horizontal sums at `BITS_ABOVE` (`0` for
/// 8-bit, `2` for 10-bit, `4` for 12-bit).
#[inline(always)]
pub(super) unsafe fn box_ab_at<V: SgrLanes, const N: usize, const BITS_ABOVE: i32>(
    sums: [&[i16]; N],
    squares: [&[i32]; N],
    scale: u32,
    (a, b): (&mut [i16], &mut [i32]),
    n: usize,
) {
    let count = (N * N) as u32;
    let (count_v, recip) = (V::splat(count as i32), V::splat(one_by_x(count) as i32));
    let scale = V::splat(scale as i32);
    let (zero, z_max) = (V::splat(0), V::splat(Z_MAX));
    let unit = V::splat(1 << SGR_BITS);
    for j in (0..n).step_by(V::LANES) {
        let mut sum = V::load_i16(&sums[0][j..]);
        let mut square = V::load(&squares[0][j..]);
        for row in 1..N {
            sum = sum.add(V::load_i16(&sums[row][j..]));
            square = square.add(V::load(&squares[row][j..]));
        }
        let (sum_8, square_8) = match BITS_ABOVE {
            0 => (sum, square),
            2 => (
                sum.add(V::splat(2)).shr::<2>(),
                square.add(V::splat(8)).shr::<4>(),
            ),
            4 => (
                sum.add(V::splat(8)).shr::<4>(),
                square.add(V::splat(128)).shr::<8>(),
            ),
            _ => unreachable!("AV1 bit depths are 8, 10, and 12"),
        };
        let p = square_8.mul(count_v).sub(sum_8.mul(sum_8)).max(zero);
        let z = p
            .mul(scale)
            .add(V::splat(1 << (MTABLE_BITS - 1)))
            .shr::<{ MTABLE_BITS as i32 }>()
            .min(z_max);
        let factor = z.sgr_factor();
        let mean = unit
            .sub(factor)
            .mul(sum)
            .mul(recip)
            .add(V::splat(1 << (RECIP_BITS - 1)))
            .shr::<{ RECIP_BITS as i32 }>();
        factor.store_i16(&mut a[j..]);
        mean.store(&mut b[j..]);
    }
}

#[inline(always)]
pub(super) unsafe fn filter_between<V: SgrLanes, S: SgrSample>(
    a: [&[i16]; 2],
    b: [&[i32]; 2],
    src: &[S],
    out: &mut [S::Filtered],
    n: usize,
) {
    for j in (0..n).step_by(V::LANES) {
        let v = filtered::<V, S, { TWO_ROW_SHIFT as i32 }>(
            weigh_between(a, j),
            weigh_between(b, j),
            src,
            j,
        );
        S::Filtered::store(v, &mut out[j..]);
    }
}

#[inline(always)]
pub(super) unsafe fn filter_on<V: SgrLanes, S: SgrSample>(
    a: &[i16],
    b: &[i32],
    src: &[S],
    out: &mut [S::Filtered],
    n: usize,
) {
    for j in (0..n).step_by(V::LANES) {
        let v = filtered::<V, S, { ONE_ROW_SHIFT as i32 }>(weigh_on(a, j), weigh_on(b, j), src, j);
        S::Filtered::store(v, &mut out[j..]);
    }
}

#[inline(always)]
pub(super) unsafe fn filter_around<V: SgrLanes, S: SgrSample>(
    a: [&[i16]; 3],
    b: [&[i32]; 3],
    src: &[S],
    out: &mut [S::Filtered],
    n: usize,
) {
    if n == 0 {
        return;
    }
    let lanes = V::LANES;
    let padded = n.checked_add(lanes - 1).expect("SGR row extent") / lanes * lanes;
    let neighbors = padded.checked_add(2).expect("SGR neighbor extent");
    assert!(
        a.iter().all(|row| row.len() >= neighbors) && b.iter().all(|row| row.len() >= neighbors)
    );
    assert!(src.len() >= padded && out.len() >= padded);
    for j in (0..padded).step_by(lanes) {
        // SAFETY: j <= padded - lanes; the six input rows have >= padded + 2 elements.
        let a = core::array::from_fn(|row| unsafe {
            core::slice::from_raw_parts(a[row].as_ptr().add(j), lanes + 2)
        });
        // SAFETY: the same six-row extent check covers the B rows.
        let b = core::array::from_fn(|row| unsafe {
            core::slice::from_raw_parts(b[row].as_ptr().add(j), lanes + 2)
        });
        // SAFETY: source and destination hold `padded` samples.
        let (src, out) = unsafe {
            (
                core::slice::from_raw_parts(src.as_ptr().add(j), lanes),
                core::slice::from_raw_parts_mut(out.as_mut_ptr().add(j), lanes),
            )
        };
        let v = filtered::<V, S, { TWO_ROW_SHIFT as i32 }>(
            weigh_around(a, 0),
            weigh_around(b, 0),
            src,
            0,
        );
        S::Filtered::store(v, out);
    }
}

#[inline(always)]
pub(super) unsafe fn blend<V: SgrLanes, S: SgrSample>(
    src: &[S],
    filtered: [&[S::Filtered]; 2],
    weights: [i32; 2],
    dst: &mut [S],
    depth: BitDepth,
) {
    let n = dst.len();
    let lanes = V::LANES;
    let (w0, w1) = (V::splat(weights[0]), V::splat(weights[1]));
    let round = V::splat(1 << (BLEND_SHIFT - 1));
    for j in (0..n).step_by(lanes) {
        let u = S::load_sample::<V>(&src[j..]).shl::<{ SOURCE_SHIFT as i32 }>();
        let f0 = S::Filtered::load::<V>(&filtered[0][j..]);
        let f1 = S::Filtered::load::<V>(&filtered[1][j..]);
        let v = u
            .mul(V::splat(PRJ_ONE))
            .add(w0.mul(f0.sub(u)))
            .add(w1.mul(f1.sub(u)))
            .add(round)
            .sra::<{ BLEND_SHIFT as i32 }>();
        if j + lanes <= n {
            S::store_clipped(v, depth, &mut dst[j..j + lanes]);
        } else {
            let mut tail = [S::default(); MAX_LANES];
            S::store_clipped(v, depth, &mut tail[..lanes]);
            dst[j..].copy_from_slice(&tail[..n - j]);
        }
    }
}

/// The 8-bit self-guided entries of an ISA's kernel table, for its
/// [`SgrLanes`] type `$lanes`, each a `$feature` function.
macro_rules! sgr_kernels {
    ($(#[$feature:meta])*; $lanes:ty) => {
        $(#[$feature])*
        unsafe fn box_sums3(src: &[u8], sum: &mut [i16], squares: &mut [i32], n: usize) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::box_sums::<$lanes, u8, 1>(src, sum, squares, n) }
        }

        $(#[$feature])*
        unsafe fn box_sums5(src: &[u8], sum: &mut [i16], squares: &mut [i32], n: usize) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::box_sums::<$lanes, u8, 2>(src, sum, squares, n) }
        }

        $(#[$feature])*
        unsafe fn box_ab3(
            sums: [&[i16]; 3],
            squares: [&[i32]; 3],
            scale: u32,
            ab: (&mut [i16], &mut [i32]),
            n: usize,
        ) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::box_ab_at::<$lanes, 3, 0>(sums, squares, scale, ab, n) }
        }

        $(#[$feature])*
        unsafe fn box_ab5(
            sums: [&[i16]; 5],
            squares: [&[i32]; 5],
            scale: u32,
            ab: (&mut [i16], &mut [i32]),
            n: usize,
        ) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::box_ab_at::<$lanes, 5, 0>(sums, squares, scale, ab, n) }
        }

        $(#[$feature])*
        unsafe fn between(a: [&[i16]; 2], b: [&[i32]; 2], src: &[u8], out: &mut [i16], n: usize) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::filter_between::<$lanes, u8>(a, b, src, out, n) }
        }

        $(#[$feature])*
        unsafe fn on(a: &[i16], b: &[i32], src: &[u8], out: &mut [i16], n: usize) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::filter_on::<$lanes, u8>(a, b, src, out, n) }
        }

        $(#[$feature])*
        unsafe fn around(a: [&[i16]; 3], b: [&[i32]; 3], src: &[u8], out: &mut [i16], n: usize) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe { super::sgr_lanes::filter_around::<$lanes, u8>(a, b, src, out, n) }
        }

        $(#[$feature])*
        unsafe fn blend(src: &[u8], filtered: [&[i16]; 2], weights: [i32; 2], dst: &mut [u8]) {
            // SAFETY: the table holding this kernel was built for its ISA.
            unsafe {
                super::sgr_lanes::blend::<$lanes, u8>(
                    src,
                    filtered,
                    weights,
                    dst,
                    nc_common::BitDepth::EIGHT,
                )
            }
        }
    };
}

pub(super) use sgr_kernels;

#[cfg(test)]
mod tests {
    use super::LANES;
    use crate::Isa;
    use crate::av1_restoration::{kernels, self_guided};

    #[test]
    fn around_exact_extents_match_scalar_and_preserve_canaries() {
        for isa in Isa::ALL
            .into_iter()
            .filter(|&isa| isa != Isa::Scalar && isa.is_supported())
        {
            let step = if isa == Isa::Avx512 { 16 } else { LANES };
            let kernel = kernels::kernels(isa).around;
            for n in [1_usize, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
                let padded = n.div_ceil(step) * step;
                let neighbors = padded + 2;
                let a: [Vec<i16>; 3] = core::array::from_fn(|row| {
                    (0..neighbors + 2)
                        .map(|j| ((j * 37 + row * 71) % 257) as i16)
                        .collect()
                });
                let b: [Vec<i32>; 3] = core::array::from_fn(|row| {
                    (0..neighbors + 2)
                        .map(|j| ((j * 2749 + row * 113) % 65090) as i32)
                        .collect()
                });
                let a = a.each_ref().map(|row| &row[1..=neighbors]);
                let b = b.each_ref().map(|row| &row[1..=neighbors]);
                let src: Vec<u8> = (0..padded).map(|j| (j * 37) as u8).collect();
                let mut expected = vec![0i16; padded];
                self_guided::filter_around(a, b, &src, &mut expected, padded);
                let mut guarded = vec![i16::MIN; padded + 2];
                // SAFETY: the ISA is available; each input has the exact
                // asserted padded extent and the output is exclusively borrowed.
                unsafe {
                    kernel(a, b, &src, &mut guarded[1..=padded], n);
                }
                assert_eq!(&guarded[1..=padded], expected, "{isa:?}, n={n}");
                assert_eq!(guarded[0], i16::MIN);
                assert_eq!(guarded[padded + 1], i16::MIN);
            }
            // SAFETY: this ISA is available and zero output must not touch buffers.
            unsafe {
                kernel([&[]; 3], [&[]; 3], &[], &mut [], 0);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn around_rejects_short_extents_before_writing() {
        for isa in Isa::ALL
            .into_iter()
            .filter(|&isa| isa != Isa::Scalar && isa.is_supported())
        {
            let step = if isa == Isa::Avx512 { 16 } else { LANES };
            let kernel = kernels::kernels(isa).around;
            for short in 0..8 {
                let a_rows = [
                    vec![1_i16; step + 2],
                    vec![1_i16; step + 2],
                    vec![1_i16; step + 2],
                ];
                let b_rows = [
                    vec![1_i32; step + 2],
                    vec![1_i32; step + 2],
                    vec![1_i32; step + 2],
                ];
                let mut a = a_rows.each_ref().map(Vec::as_slice);
                let mut b = b_rows.each_ref().map(Vec::as_slice);
                match short {
                    0..3 => a[short] = &a[short][..=step],
                    3..6 => b[short - 3] = &b[short - 3][..=step],
                    _ => {}
                }
                let src = vec![17u8; step];
                let mut out = vec![i16::MIN; step];
                let expected_out = out.clone();
                let src_len = if short == 6 { step - 1 } else { step };
                let out_len = if short == 7 { step - 1 } else { step };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: ISA is available. The kernel's checked-slice
                    // contract must reject these deliberately short buffers.
                    unsafe {
                        kernel(a, b, &src[..src_len], &mut out[..out_len], 1);
                    }
                }));
                assert!(result.is_err(), "{isa:?}, short buffer {short}");
                assert_eq!(out, expected_out);
            }
        }
    }
}
