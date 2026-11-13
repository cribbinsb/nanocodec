//! WebAssembly SIMD128 vector types.
//!
//! See the `simd` module docs for why the methods are safe functions: they
//! are only ever called from kernels compiled with SIMD128 enabled (the
//! module only exists in such builds). Loads and stores index their slices
//! first; each pointer access is one `unsafe` block.
//!
//! Builds with relaxed SIMD (`+relaxed-simd`, the browser's second module)
//! route lane selects and Q15 multiplies through [`relaxed`], whose relaxed
//! instructions are fully specified for the inputs these methods accept, so
//! both builds give identical results on every engine and CPU.

#![allow(clippy::inline_always)]

use core::arch::wasm32::{
    i8x16_shuffle, i8x16_swizzle, i16x8_abs, i16x8_add, i16x8_eq, i16x8_gt, i16x8_lt, i16x8_max,
    i16x8_min, i16x8_mul, i16x8_narrow_i32x4, i16x8_neg, i16x8_shr, i16x8_shuffle, i16x8_splat,
    i16x8_sub, i32x4_abs, i32x4_add, i32x4_bitmask, i32x4_dot_i16x8, i32x4_eq,
    i32x4_extadd_pairwise_i16x8, i32x4_extend_low_i16x8, i32x4_extmul_high_i16x8,
    i32x4_extmul_low_i16x8, i32x4_extract_lane, i32x4_gt, i32x4_max, i32x4_min, i32x4_mul,
    i32x4_shl, i32x4_shr, i32x4_shuffle, i32x4_splat, i32x4_sub, i64x2_shuffle, u8x16_narrow_i16x8,
    u16x8_extend_low_u8x16, u16x8_load_extend_u8x8, u16x8_min, u16x8_sub_sat,
    u32x4_extend_low_u16x8, u32x4_shr, v128, v128_and, v128_load, v128_load32_zero,
    v128_load64_zero, v128_or, v128_store, v128_store32_lane, v128_store64_lane, v128_xor,
};

use core::mem::MaybeUninit;

use super::{I16FilterLanes, I16Lanes, I16Pairs, I16x8Shuffles, I32Lanes, packed_pair};
use relaxed::{q15_mul_round, select_i16, select_i32};

/// Lane selects, bounded byte lookups and the rounding Q15 multiply, as relaxed-SIMD instructions
/// when the build enables them and plain SIMD128 otherwise. Each relaxed
/// form is used only where its result is fully specified:
///
/// - `relaxed_laneselect` may decide a lane by its top bit alone, or by
///   every bit; the two agree when each mask lane is all ones or all zeros,
///   which every caller guarantees (masks come from lane compares of the
///   same width: [`I32Lanes::select`], [`I16FilterLanes::blend`], and the
///   sign test in `with_sign_of`).
/// - `relaxed_q15mulr` may differ from the saturating form only for
///   `i16::MIN * i16::MIN`, which [`I16Lanes::mul_q15_round`] excludes.
/// - `relaxed_swizzle` differs only for out-of-range byte indices; callers
///   of `swizzle_in_bounds` restrict them to 0..=15.
pub(crate) mod relaxed {
    use core::arch::wasm32::v128;
    #[cfg(target_feature = "relaxed-simd")]
    use core::arch::wasm32::{
        i8x16_relaxed_swizzle, i16x8_relaxed_laneselect, i16x8_relaxed_q15mulr,
        i32x4_relaxed_laneselect,
    };
    #[cfg(not(target_feature = "relaxed-simd"))]
    use core::arch::wasm32::{i8x16_swizzle, i16x8_q15mulr_sat, v128_bitselect};

    /// Table lookup with every byte index in 0..=15. Relaxed swizzle differs
    /// only for out-of-range indices, so this restricted form is bit-exact.
    #[inline(always)]
    pub(crate) fn swizzle_in_bounds(table: v128, index: v128) -> v128 {
        #[cfg(target_feature = "relaxed-simd")]
        return i8x16_relaxed_swizzle(table, index);
        #[cfg(not(target_feature = "relaxed-simd"))]
        return i8x16_swizzle(table, index);
    }

    /// `if_set` where the 16-bit mask lane is all ones, `if_clear` where zero.
    #[inline(always)]
    pub(crate) fn select_i16(mask: v128, if_set: v128, if_clear: v128) -> v128 {
        #[cfg(target_feature = "relaxed-simd")]
        return i16x8_relaxed_laneselect(if_set, if_clear, mask);
        #[cfg(not(target_feature = "relaxed-simd"))]
        return v128_bitselect(if_set, if_clear, mask);
    }

    /// `if_set` where the 32-bit mask lane is all ones, `if_clear` where zero.
    #[inline(always)]
    pub(super) fn select_i32(mask: v128, if_set: v128, if_clear: v128) -> v128 {
        #[cfg(target_feature = "relaxed-simd")]
        return i32x4_relaxed_laneselect(if_set, if_clear, mask);
        #[cfg(not(target_feature = "relaxed-simd"))]
        return v128_bitselect(if_set, if_clear, mask);
    }

    /// `(a * b + 2^14) >> 15` per 16-bit lane; not both lanes `i16::MIN`.
    #[inline(always)]
    pub(crate) fn q15_mul_round(a: v128, b: v128) -> v128 {
        #[cfg(target_feature = "relaxed-simd")]
        return i16x8_relaxed_q15mulr(a, b);
        #[cfg(not(target_feature = "relaxed-simd"))]
        return i16x8_q15mulr_sat(a, b);
    }
}

/// Four `i32` lanes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct I32x4(v128);

/// Eight `i16` lanes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct I16x8(v128);

/// Four `i16` lanes in the low half of a register; the high half stays zero.
#[derive(Debug, Clone, Copy)]
pub(crate) struct I16x4(v128);

/// Sum of the four `i32` lanes of `v`.
#[inline(always)]
fn sum_lanes(v: v128) -> i64 {
    i64::from(i32x4_extract_lane::<0>(v))
        + i64::from(i32x4_extract_lane::<1>(v))
        + i64::from(i32x4_extract_lane::<2>(v))
        + i64::from(i32x4_extract_lane::<3>(v))
}

/// `(acc + round) >> shift` on four `i32` lanes.
#[inline(always)]
fn round_shift_x4(acc: v128, round: i32, shift: u32) -> v128 {
    i32x4_shr(i32x4_add(acc, i32x4_splat(round)), shift)
}

/// The low four bytes of `v` stored into `dst[..4]`.
#[inline(always)]
fn store_low4(v: v128, dst: &mut [u8]) {
    let dst = &mut dst[..4];
    // SAFETY: `dst` holds 4 bytes.
    unsafe { v128_store32_lane::<0>(v, dst.as_mut_ptr().cast()) }
}

/// The low eight bytes of `v` stored into `dst[..8]`.
#[inline(always)]
fn store_low8(v: v128, dst: &mut [u8]) {
    let dst = &mut dst[..8];
    // SAFETY: `dst` holds 8 bytes.
    unsafe { v128_store64_lane::<0>(v, dst.as_mut_ptr().cast()) }
}

// ---------------------------------------------------------------- I32x4 ----

impl I32Lanes for I32x4 {
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(value: i32) -> Self {
        Self(i32x4_splat(value))
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self(i32x4_add(self.0, other.0))
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self(i32x4_sub(self.0, other.0))
    }

    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        let sum = i32x4_add(
            i32x4_mul(a.0, i32x4_splat(w0)),
            i32x4_mul(b.0, i32x4_splat(w1)),
        );
        Self(round_shift_x4(sum, 1 << (bit - 1), bit))
    }

    #[inline(always)]
    fn clamp16(self) -> Self {
        Self(i32x4_max(
            i32x4_min(self.0, i32x4_splat(i32::from(i16::MAX))),
            i32x4_splat(i32::from(i16::MIN)),
        ))
    }

    #[inline(always)]
    fn round_shift(self, bits: u32) -> Self {
        if bits == 0 {
            return self;
        }
        Self(round_shift_x4(self.0, 1 << (bits - 1), bits))
    }

    #[inline(always)]
    fn shl(self, bits: u32) -> Self {
        Self(i32x4_shl(self.0, bits))
    }

    #[inline(always)]
    fn mul_scalar(self, c: i32) -> Self {
        Self(i32x4_mul(self.0, i32x4_splat(c)))
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self(i32x4_mul(self.0, other.0))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(i32x4_abs(self.0))
    }

    #[inline(always)]
    fn shr_logical(self, bits: u32) -> Self {
        Self(u32x4_shr(self.0, bits))
    }

    #[inline(always)]
    fn sra(self, bits: u32) -> Self {
        Self(i32x4_shr(self.0, bits))
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self(i32x4_min(self.0, other.0))
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self(i32x4_max(self.0, other.0))
    }

    #[inline(always)]
    fn with_sign_of(self, like: Self) -> Self {
        // (x ^ m) - m with m = like >> 31.
        let m = i32x4_shr(like.0, 31);
        Self(i32x4_sub(v128_xor(self.0, m), m))
    }

    #[inline(always)]
    fn count_nonzero(self) -> u32 {
        4 - u32::from(i32x4_bitmask(i32x4_eq(self.0, i32x4_splat(0)))).count_ones()
    }

    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        let src = &src[..4];
        // SAFETY: `src` holds 4 values (16 bytes).
        Self(unsafe { v128_load(src.as_ptr().cast()) })
    }

    #[inline(always)]
    fn store(self, dst: &mut [i32]) {
        let dst = &mut dst[..4];
        // SAFETY: `dst` holds 4 values (16 bytes).
        unsafe { v128_store(dst.as_mut_ptr().cast(), self.0) }
    }

    #[inline(always)]
    fn load_i16(src: &[i16]) -> Self {
        let src = &src[..4];
        // SAFETY: `src` holds 4 values (8 bytes).
        Self(i32x4_extend_low_i16x8(unsafe {
            v128_load64_zero(src.as_ptr().cast())
        }))
    }

    #[inline(always)]
    fn add_to_u8(self, dst: &mut [u8]) {
        let bytes = &dst[..4];
        // SAFETY: `bytes` holds 4 bytes.
        let packed = unsafe { v128_load32_zero(bytes.as_ptr().cast()) };
        let pred = u32x4_extend_low_u16x8(u16x8_extend_low_u8x16(packed));
        let sum = i32x4_add(self.0, pred);
        let words = i16x8_narrow_i32x4(sum, sum);
        store_low4(u8x16_narrow_i16x8(words, words), dst);
    }

    #[inline(always)]
    fn store_transposed(rows: &[Self], dst: &mut [MaybeUninit<i32>], dst_stride: usize) {
        let r = [rows[0].0, rows[1].0, rows[2].0, rows[3].0];
        let t0 = i32x4_shuffle::<0, 4, 1, 5>(r[0], r[1]);
        let t1 = i32x4_shuffle::<2, 6, 3, 7>(r[0], r[1]);
        let t2 = i32x4_shuffle::<0, 4, 1, 5>(r[2], r[3]);
        let t3 = i32x4_shuffle::<2, 6, 3, 7>(r[2], r[3]);
        let out = [
            i64x2_shuffle::<0, 2>(t0, t2),
            i64x2_shuffle::<1, 3>(t0, t2),
            i64x2_shuffle::<0, 2>(t1, t3),
            i64x2_shuffle::<1, 3>(t1, t3),
        ];
        for (c, &row) in out.iter().enumerate() {
            let dst = &mut dst[c * dst_stride..][..Self::LANES];
            // SAFETY: `dst` holds 4 values (indexed above); uninitialised
            // memory may be written.
            unsafe { v128_store(dst.as_mut_ptr().cast(), row) };
        }
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Self {
        Self(i32x4_gt(self.0, other.0))
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Self {
        Self(i32x4_eq(self.0, other.0))
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self(v128_or(self.0, other.0))
    }

    #[inline(always)]
    fn select(self, if_set: Self, if_clear: Self) -> Self {
        Self(select_i32(self.0, if_set.0, if_clear.0))
    }
}

// ---------------------------------------------------------------- I16x8 ----

impl I16Lanes for I16x8 {
    const LANES: usize = 8;
    type Acc = [v128; 2];

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self(i16x8_min(self.0, other.0))
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self(i16x8_max(self.0, other.0))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(i16x8_abs(self.0))
    }

    #[inline(always)]
    fn with_sign_of(self, like: Self) -> Self {
        Self(select_i16(
            i16x8_lt(like.0, i16x8_splat(0)),
            i16x8_neg(self.0),
            self.0,
        ))
    }

    #[inline(always)]
    fn splat(value: i16) -> Self {
        Self(i16x8_splat(value))
    }

    #[inline(always)]
    fn load_u8(src: &[u8]) -> Self {
        let src = &src[..8];
        // SAFETY: `src` holds 8 bytes.
        Self(unsafe { u16x8_load_extend_u8x8(src.as_ptr()) })
    }

    #[inline(always)]
    fn load(src: &[i16]) -> Self {
        let src = &src[..8];
        // SAFETY: `src` holds 8 values (16 bytes).
        Self(unsafe { v128_load(src.as_ptr().cast()) })
    }

    #[inline(always)]
    fn store(self, dst: &mut [i16]) {
        let dst = &mut dst[..8];
        // SAFETY: `dst` holds 8 values (16 bytes).
        unsafe { v128_store(dst.as_mut_ptr().cast(), self.0) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self(i16x8_add(self.0, other.0))
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self(i16x8_sub(self.0, other.0))
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self(i16x8_mul(self.0, other.0))
    }

    #[inline(always)]
    fn mul_q15_round(self, other: Self) -> Self {
        Self(q15_mul_round(self.0, other.0))
    }

    #[inline(always)]
    fn sra(self, bits: u32) -> Self {
        Self(i16x8_shr(self.0, bits))
    }

    #[inline(always)]
    fn clamp_u8(self) -> Self {
        Self(i16x8_max(
            i16x8_min(self.0, i16x8_splat(255)),
            i16x8_splat(0),
        ))
    }

    #[inline(always)]
    fn store_u8_sat(self, dst: &mut [u8]) {
        store_low8(u8x16_narrow_i16x8(self.0, self.0), dst);
    }

    #[inline(always)]
    fn acc_zero() -> Self::Acc {
        [i32x4_splat(0); 2]
    }

    #[inline(always)]
    fn acc_mul_add(acc: Self::Acc, a: Self, coeff: i16) -> Self::Acc {
        let c = i16x8_splat(coeff);
        [
            i32x4_add(acc[0], i32x4_extmul_low_i16x8(a.0, c)),
            i32x4_add(acc[1], i32x4_extmul_high_i16x8(a.0, c)),
        ]
    }

    #[inline(always)]
    fn acc_add(acc: Self::Acc, a: Self) -> Self::Acc {
        [i32x4_add(acc[0], i32x4_extadd_pairwise_i16x8(a.0)), acc[1]]
    }

    #[inline(always)]
    fn acc_add_sq(acc: Self::Acc, a: Self) -> Self::Acc {
        [i32x4_add(acc[0], i32x4_dot_i16x8(a.0, a.0)), acc[1]]
    }

    #[inline(always)]
    fn acc_round_shift_store_u8(acc: Self::Acc, round: i32, shift: u32, dst: &mut [u8]) {
        let words = i16x8_narrow_i32x4(
            round_shift_x4(acc[0], round, shift),
            round_shift_x4(acc[1], round, shift),
        );
        store_low8(u8x16_narrow_i16x8(words, words), dst);
    }

    #[inline(always)]
    fn acc_round_shift_narrow(acc: Self::Acc, round: i32, shift: u32) -> Self {
        Self(i16x8_narrow_i32x4(
            round_shift_x4(acc[0], round, shift),
            round_shift_x4(acc[1], round, shift),
        ))
    }

    #[inline(always)]
    fn acc_sum(acc: Self::Acc) -> i64 {
        sum_lanes(acc[0]) + sum_lanes(acc[1])
    }
}

// ---------------------------------------------------------------- I16x4 ----

impl I16Lanes for I16x4 {
    const LANES: usize = 4;
    type Acc = v128;

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self(i16x8_min(self.0, other.0))
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self(i16x8_max(self.0, other.0))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(i16x8_abs(self.0))
    }

    #[inline(always)]
    fn with_sign_of(self, like: Self) -> Self {
        Self(select_i16(
            i16x8_lt(like.0, i16x8_splat(0)),
            i16x8_neg(self.0),
            self.0,
        ))
    }

    #[inline(always)]
    fn splat(value: i16) -> Self {
        let v = i16x8_splat(value);
        Self(i64x2_shuffle::<0, 2>(v, i16x8_splat(0)))
    }

    #[inline(always)]
    fn load_u8(src: &[u8]) -> Self {
        let src = &src[..4];
        // SAFETY: `src` holds 4 bytes.
        Self(u16x8_extend_low_u8x16(unsafe {
            v128_load32_zero(src.as_ptr().cast())
        }))
    }

    #[inline(always)]
    fn load(src: &[i16]) -> Self {
        let src = &src[..4];
        // SAFETY: `src` holds 4 values (8 bytes).
        Self(unsafe { v128_load64_zero(src.as_ptr().cast()) })
    }

    #[inline(always)]
    fn store(self, dst: &mut [i16]) {
        let dst = &mut dst[..4];
        // SAFETY: `dst` holds 4 values (8 bytes).
        unsafe { v128_store64_lane::<0>(self.0, dst.as_mut_ptr().cast()) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self(i16x8_add(self.0, other.0))
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self(i16x8_sub(self.0, other.0))
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self(i16x8_mul(self.0, other.0))
    }

    #[inline(always)]
    fn mul_q15_round(self, other: Self) -> Self {
        Self(q15_mul_round(self.0, other.0))
    }

    #[inline(always)]
    fn sra(self, bits: u32) -> Self {
        Self(i16x8_shr(self.0, bits))
    }

    #[inline(always)]
    fn clamp_u8(self) -> Self {
        Self(i16x8_max(
            i16x8_min(self.0, i16x8_splat(255)),
            i16x8_splat(0),
        ))
    }

    #[inline(always)]
    fn store_u8_sat(self, dst: &mut [u8]) {
        store_low4(u8x16_narrow_i16x8(self.0, self.0), dst);
    }

    #[inline(always)]
    fn acc_zero() -> Self::Acc {
        i32x4_splat(0)
    }

    #[inline(always)]
    fn acc_mul_add(acc: Self::Acc, a: Self, coeff: i16) -> Self::Acc {
        i32x4_add(acc, i32x4_extmul_low_i16x8(a.0, i16x8_splat(coeff)))
    }

    #[inline(always)]
    fn acc_add(acc: Self::Acc, a: Self) -> Self::Acc {
        i32x4_add(acc, i32x4_extadd_pairwise_i16x8(a.0))
    }

    #[inline(always)]
    fn acc_add_sq(acc: Self::Acc, a: Self) -> Self::Acc {
        i32x4_add(acc, i32x4_dot_i16x8(a.0, a.0))
    }

    #[inline(always)]
    fn acc_round_shift_store_u8(acc: Self::Acc, round: i32, shift: u32, dst: &mut [u8]) {
        let v = round_shift_x4(acc, round, shift);
        let words = i16x8_narrow_i32x4(v, v);
        store_low4(u8x16_narrow_i16x8(words, words), dst);
    }

    #[inline(always)]
    fn acc_round_shift_narrow(acc: Self::Acc, round: i32, shift: u32) -> Self {
        Self(i16x8_narrow_i32x4(
            round_shift_x4(acc, round, shift),
            i32x4_splat(0),
        ))
    }

    #[inline(always)]
    fn acc_sum(acc: Self::Acc) -> i64 {
        sum_lanes(acc)
    }
}

// ------------------------------------------------------- filter lanes ----

impl I16FilterLanes for I16x8 {
    #[inline(always)]
    fn load_halves(lo: &[i16], hi: &[i16]) -> Self {
        let (lo, hi) = (&lo[..4], &hi[..4]);
        // SAFETY: `lo` and `hi` hold 4 values (8 bytes) each.
        unsafe {
            Self(i64x2_shuffle::<0, 2>(
                v128_load64_zero(lo.as_ptr().cast()),
                v128_load64_zero(hi.as_ptr().cast()),
            ))
        }
    }

    #[inline(always)]
    fn store_u8_sat_halves(self, lo: &mut [u8], hi: &mut [u8]) {
        let packed = u8x16_narrow_i16x8(self.0, self.0);
        lo[..4].copy_from_slice(&i32x4_extract_lane::<0>(packed).to_le_bytes());
        hi[..4].copy_from_slice(&i32x4_extract_lane::<1>(packed).to_le_bytes());
    }

    #[inline(always)]
    fn min_u16(self, other: Self) -> Self {
        Self(u16x8_min(self.0, other.0))
    }

    #[inline(always)]
    fn sub_sat_u16(self, other: Self) -> Self {
        Self(u16x8_sub_sat(self.0, other.0))
    }

    #[inline(always)]
    fn eq_mask(self, other: Self) -> Self {
        Self(i16x8_eq(self.0, other.0))
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(v128_and(self.0, other.0))
    }

    #[inline(always)]
    fn gt_mask(self, other: Self) -> Self {
        Self(i16x8_gt(self.0, other.0))
    }

    #[inline(always)]
    fn blend(self, if_set: Self, if_clear: Self) -> Self {
        Self(select_i16(self.0, if_set.0, if_clear.0))
    }
}

/// Bytes `K..K + 16` of the 32-byte concatenation `lo ‖ hi`: a constant
/// `i8x16.shuffle` with consecutive indices, which engines lower to one
/// byte-align instruction (`palignr`, `ext`) instead of a table shuffle.
macro_rules! concat_bytes {
    ($lo:expr, $hi:expr, $k:literal) => {
        i8x16_shuffle::<
            { $k },
            { $k + 1 },
            { $k + 2 },
            { $k + 3 },
            { $k + 4 },
            { $k + 5 },
            { $k + 6 },
            { $k + 7 },
            { $k + 8 },
            { $k + 9 },
            { $k + 10 },
            { $k + 11 },
            { $k + 12 },
            { $k + 13 },
            { $k + 14 },
            { $k + 15 },
        >($lo, $hi)
    };
}

impl I16Pairs for I16x8 {
    type Pairs = I32x4;

    #[inline(always)]
    fn splat_pair(first: i16, second: i16) -> Self {
        Self(i32x4_splat(packed_pair(first, second)))
    }

    #[inline(always)]
    fn pair_dot(self, other: Self) -> I32x4 {
        I32x4(i32x4_dot_i16x8(self.0, other.0))
    }

    #[inline(always)]
    fn narrow_pairs(low: I32x4, high: I32x4) -> Self {
        Self(i16x8_narrow_i32x4(low.0, high.0))
    }
}

impl I16x8Shuffles for I16x8 {
    type Wide = I32x4;

    #[inline(always)]
    fn lanes_up(self, n: usize) -> Self {
        // `n` is a constant wherever the CDEF direction search's unrolled
        // loops inline this, so the match folds to one shuffle.
        let zero = i16x8_splat(0);
        Self(match n {
            0 => self.0,
            1 => concat_bytes!(zero, self.0, 14),
            2 => concat_bytes!(zero, self.0, 12),
            3 => concat_bytes!(zero, self.0, 10),
            4 => concat_bytes!(zero, self.0, 8),
            5 => concat_bytes!(zero, self.0, 6),
            6 => concat_bytes!(zero, self.0, 4),
            7 => concat_bytes!(zero, self.0, 2),
            _ => zero,
        })
    }

    #[inline(always)]
    fn lanes_down(self, n: usize) -> Self {
        let zero = i16x8_splat(0);
        Self(match n {
            0 => self.0,
            1 => concat_bytes!(self.0, zero, 2),
            2 => concat_bytes!(self.0, zero, 4),
            3 => concat_bytes!(self.0, zero, 6),
            4 => concat_bytes!(self.0, zero, 8),
            5 => concat_bytes!(self.0, zero, 10),
            6 => concat_bytes!(self.0, zero, 12),
            7 => concat_bytes!(self.0, zero, 14),
            _ => zero,
        })
    }

    #[inline(always)]
    fn shuffle_bytes(self, pattern: &[u8; 16]) -> Self {
        // SAFETY: `pattern` holds 16 bytes. `swizzle` zeroes the bytes whose
        // index is out of range.
        Self(i8x16_swizzle(self.0, unsafe {
            v128_load(pattern.as_ptr().cast())
        }))
    }

    #[inline(always)]
    fn zip_lo(self, other: Self) -> Self {
        Self(i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(self.0, other.0))
    }

    #[inline(always)]
    fn zip_hi(self, other: Self) -> Self {
        Self(i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(self.0, other.0))
    }

    #[inline(always)]
    fn madd(self, other: Self) -> I32x4 {
        I32x4(i32x4_dot_i16x8(self.0, other.0))
    }

    #[inline(always)]
    fn sum_lanes4(x: [I32x4; 4]) -> I32x4 {
        let pairs = |a: v128, b: v128| {
            i32x4_add(
                i32x4_shuffle::<0, 4, 2, 6>(a, b),
                i32x4_shuffle::<1, 5, 3, 7>(a, b),
            )
        };
        let (lo, hi) = (pairs(x[0].0, x[1].0), pairs(x[2].0, x[3].0));
        I32x4(i32x4_add(
            i32x4_shuffle::<0, 1, 4, 5>(lo, hi),
            i32x4_shuffle::<2, 3, 6, 7>(lo, hi),
        ))
    }
}
