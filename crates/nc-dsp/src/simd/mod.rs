//! Lane abstractions: one vector type per ISA and width, so that kernels
//! whose arithmetic is the same in every lane are written once.
//!
//! Prediction, residual and transform kernels apply identical arithmetic to
//! every column (or row) of a block. Instead of writing each one four times,
//! the arithmetic is written once in a module's `generic.rs`, over these
//! traits, and each ISA module instantiates it with its vector types inside a
//! `#[target_feature]` function. The traits are deliberately small: loads and
//! stores, lane-wise integer operations and the few widening steps the
//! kernels need.
//!
//! # Soundness
//!
//! Loads and stores take slices and index them first (`&src[..LANES]`), so
//! they never read or write outside their arguments, whatever the caller
//! passes. The remaining precondition is the CPU: a vector type's methods
//! execute its ISA's instructions. They are `#[inline(always)]` and only
//! ever called from generic kernels, which are `unsafe fn`s that callers may
//! only use inside a `#[target_feature]` function of that ISA; those wrapper
//! functions are reachable only through a [`crate::Dsp`] table built after
//! [`crate::Isa::is_supported`]. Vector values cannot be created any other
//! way: the types are crate-private, with private fields.
//!
//! `i32` implements [`I32Lanes`] with one lane: the transform reference is the
//! same butterfly code on that type, computing in 64 bits where libaom does.
//! Prediction and residual kernels have separate, plainly written scalar
//! references instead.

use core::mem::MaybeUninit;

pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx512;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx512_i16;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon_dot;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon_u8;

/// A vector of `LANES` signed 32-bit lanes (transform arithmetic).
pub(crate) trait I32Lanes: Copy {
    /// Lanes per vector.
    const LANES: usize;

    /// Every lane `value`.
    fn splat(value: i32) -> Self;
    /// Lane-wise wrapping add.
    fn add(self, other: Self) -> Self;
    /// Lane-wise wrapping subtract.
    fn sub(self, other: Self) -> Self;
    /// libaom `half_btf`: `round_shift(w0 * a + w1 * b, bit)` per lane, with
    /// `1 <= bit <= 13`. Implementations may compute in 32 bits: callers keep
    /// `|w0 * a| + |w1 * b| < 2^30` (see the `txfm` module's range notes).
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self;
    /// Clamp every lane to the signed 16-bit range (the 8-bit stage range).
    fn clamp16(self) -> Self;
    /// libaom `round_shift`: `(x + (1 << (bits - 1))) >> bits`; `bits == 0`
    /// is the identity. `bits <= 16`.
    fn round_shift(self, bits: u32) -> Self;
    /// Lane-wise left shift by `bits <= 10`.
    fn shl(self, bits: u32) -> Self;
    /// Lane-wise wrapping multiply by a constant.
    fn mul_scalar(self, c: i32) -> Self;
    /// `self + x[0] · c[0] + .. + x[3] · c[3]`, lane-wise wrapping: four
    /// lines times four constants of one matrix row (the integer transforms'
    /// dot products). NEON overrides it with multiply-accumulates by element
    /// of one loaded constant vector, saving a broadcast load per product.
    #[allow(clippy::inline_always)] // a kernel building block, as the impls
    #[inline(always)]
    fn mul_add4(self, x: [Self; 4], c: &[i32; 4]) -> Self {
        let mut acc = self;
        for (x, &c) in x.iter().zip(c) {
            acc = acc.add(x.mul_scalar(c));
        }
        acc
    }
    /// Lane-wise wrapping multiply.
    fn mul(self, other: Self) -> Self;
    /// Lane-wise absolute value (`i32::MIN` stays as is).
    fn abs(self) -> Self;
    /// Lane-wise logical right shift by `bits < 32`.
    fn shr_logical(self, bits: u32) -> Self;
    /// Lane-wise signed arithmetic right shift by `bits < 32`.
    fn sra(self, bits: u32) -> Self;
    /// Lane-wise signed minimum.
    fn min(self, other: Self) -> Self;
    /// Lane-wise signed maximum.
    fn max(self, other: Self) -> Self;
    /// `self` negated in the lanes where `like` is negative (`self` is a
    /// non-negative magnitude).
    fn with_sign_of(self, like: Self) -> Self;
    /// Number of non-zero lanes.
    fn count_nonzero(self) -> u32;
    /// The first `LANES` values of `src`.
    fn load(src: &[i32]) -> Self;
    /// Store into the first `LANES` values of `dst`.
    fn store(self, dst: &mut [i32]);
    /// The first `LANES` values of `src`, sign-extended.
    fn load_i16(src: &[i16]) -> Self;
    /// `dst[i] = clamp(dst[i] + lane i, 0, 255)` for the first `LANES` bytes.
    fn add_to_u8(self, dst: &mut [u8]);
    /// Store the `LANES` x `LANES` tile formed by `rows` (`rows.len() ==
    /// LANES`), transposed: lane `c` of `rows[r]` goes to
    /// `dst[c * dst_stride + r]`. `dst` may be uninitialised; every element
    /// of the tile is written.
    fn store_transposed(rows: &[Self], dst: &mut [MaybeUninit<i32>], dst_stride: usize);
    /// Transpose a square lane tile and add its rows to an 8-bit picture.
    /// Like `store_transposed`, but clips after adding the prediction.
    /// ISA implementations can keep the transpose in registers.
    #[inline]
    #[allow(unsafe_code)]
    fn add_transposed_u8(rows: &[Self], dst: &mut [u8], dst_stride: usize) {
        const MAX_LANES: usize = 16;
        assert!(Self::LANES <= MAX_LANES);
        let mut tile = [MaybeUninit::<i32>::uninit(); MAX_LANES * MAX_LANES];
        Self::store_transposed(rows, &mut tile, Self::LANES);
        // SAFETY: store_transposed wrote every cell of this complete tile.
        let tile = unsafe {
            core::slice::from_raw_parts(tile.as_ptr().cast::<i32>(), Self::LANES * Self::LANES)
        };
        for row in 0..Self::LANES {
            Self::load(&tile[row * Self::LANES..]).add_to_u8(&mut dst[row * dst_stride..]);
        }
    }
    /// All ones in the lanes where `self > other` (signed), zero elsewhere.
    fn gt(self, other: Self) -> Self;
    /// All ones in the lanes where `self == other`, zero elsewhere.
    fn eq(self, other: Self) -> Self;
    /// Lane-wise bitwise or.
    fn or(self, other: Self) -> Self;
    /// `if_set` in the lanes where the mask `self` is all ones, `if_clear`
    /// where it is zero (`self` comes from [`I32Lanes::gt`] and friends).
    fn select(self, if_set: Self, if_clear: Self) -> Self;
}

/// A vector of `LANES` signed 16-bit lanes (prediction and residual
/// arithmetic), with a matching 32-bit accumulator.
/// Entries of an [`I16Lanes::lookup`] table: 256 and one of padding.
pub(crate) const LOOKUP_TABLE: usize = 257;

pub(crate) trait I16Lanes: Copy {
    /// Lanes per vector.
    const LANES: usize;
    /// Signed 32-bit accumulators, one per lane (or per pair of lanes for
    /// the order-independent sums [`Self::acc_add`] / [`Self::acc_add_sq`]).
    type Acc: Copy;

    /// Every lane `value`.
    fn splat(value: i16) -> Self;
    /// The first `LANES` bytes of `src`, zero-extended.
    fn load_u8(src: &[u8]) -> Self;
    /// The first `LANES` values of `src`.
    fn load(src: &[i16]) -> Self;
    /// Store into the first `LANES` values of `dst`.
    fn store(self, dst: &mut [i16]);
    /// Lane-wise wrapping add.
    fn add(self, other: Self) -> Self;
    /// Lane-wise wrapping subtract.
    fn sub(self, other: Self) -> Self;
    /// Lane-wise wrapping multiply (low 16 bits).
    fn mul(self, other: Self) -> Self;
    /// Lane-wise rounding Q15 multiply-high: `(a * b + 2^14) >> 15` (x86
    /// `pmulhrsw`, NEON `sqrdmulh`, wasm `q15mulr`). Exact as long as not
    /// both lanes are `i16::MIN`.
    fn mul_q15_round(self, other: Self) -> Self;
    /// Lane-wise arithmetic right shift by `bits <= 15`.
    fn sra(self, bits: u32) -> Self;
    /// Clamp every lane to `0..=255`.
    fn clamp_u8(self) -> Self;
    /// Store `clamp(lane, 0, 255)` into the first `LANES` bytes of `dst`.
    fn store_u8_sat(self, dst: &mut [u8]);
    /// [`Self::load_u8`] from a pointer, for kernels whose whole window was
    /// validated once at the API boundary.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reading `LANES` bytes.
    #[allow(
        unsafe_code,
        clippy::inline_always,
        reason = "unchecked load for validated windows"
    )]
    #[inline(always)]
    unsafe fn load_u8_at(src: *const u8) -> Self {
        // SAFETY: the caller guarantees `LANES` readable bytes; the slice's
        // own bounds check then folds away.
        Self::load_u8(unsafe { core::slice::from_raw_parts(src, Self::LANES) })
    }
    /// Lane `i` is `table[indices[i]]` (a 256-entry lookup by byte, padded
    /// by one entry so that a 32-bit gather of the last stays inside), for
    /// the first `LANES` indices. The default builds the vector in memory;
    /// ISAs override it with lane inserts or gathers.
    #[inline(always)]
    #[allow(clippy::inline_always, reason = "per vector in the grain kernels")]
    #[cfg_attr(
        target_arch = "aarch64",
        allow(dead_code, reason = "NEON grain uses `tbl`")
    )]
    fn lookup(table: &[i16; LOOKUP_TABLE], indices: &[u8]) -> Self {
        let mut lanes = [0i16; 32];
        for (lane, &index) in lanes.iter_mut().zip(&indices[..Self::LANES]) {
            *lane = table[usize::from(index)];
        }
        Self::load(&lanes)
    }
    /// [`Self::store_u8_sat`] to a pointer (see [`Self::load_u8_at`]).
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writing `LANES` bytes.
    #[allow(
        unsafe_code,
        clippy::inline_always,
        reason = "unchecked store for validated windows"
    )]
    #[inline(always)]
    unsafe fn store_u8_sat_at(self, dst: *mut u8) {
        // SAFETY: the caller guarantees `LANES` writable bytes.
        self.store_u8_sat(unsafe { core::slice::from_raw_parts_mut(dst, Self::LANES) });
    }
    /// [`Self::mul_add_u8_pair`] from pointers (see [`Self::load_u8_at`]).
    ///
    /// # Safety
    ///
    /// `a` and `b` must each be valid for reading `LANES` bytes.
    #[allow(
        unsafe_code,
        clippy::inline_always,
        reason = "unchecked loads for validated windows"
    )]
    #[inline(always)]
    unsafe fn mul_add_u8_pair_at(a: *const u8, b: *const u8, wa: u8, wb: u8) -> Self {
        // SAFETY: the caller guarantees `LANES` readable bytes at each.
        unsafe {
            Self::mul_add_u8_pair(
                core::slice::from_raw_parts(a, Self::LANES),
                core::slice::from_raw_parts(b, Self::LANES),
                wa,
                wb,
            )
        }
    }
    /// `a[i]·wa + b[i]·wb` over the first `LANES` bytes of `a` and `b`, for
    /// unsigned weights with `wa + wb <= 128` (so no lane exceeds 32640 and
    /// the result is exact in 16 bits). x86 computes it with one
    /// `pmaddubsw` on the interleaved bytes, NEON with a widening
    /// multiply-accumulate; the default is the plain widened form.
    #[allow(
        clippy::inline_always,
        reason = "lane operations must inline into the kernels"
    )]
    #[inline(always)]
    fn mul_add_u8_pair(a: &[u8], b: &[u8], wa: u8, wb: u8) -> Self {
        Self::load_u8(a)
            .mul(Self::splat(i16::from(wa)))
            .add(Self::load_u8(b).mul(Self::splat(i16::from(wb))))
    }
    /// Lane-wise signed minimum.
    fn min(self, other: Self) -> Self;
    /// Lane-wise signed maximum.
    fn max(self, other: Self) -> Self;
    /// Lane-wise absolute value (`i16::MIN` stays as is).
    fn abs(self) -> Self;
    /// `self` negated in the lanes where `like` is negative. `self` is a
    /// non-negative magnitude that is zero wherever `like` is zero (x86
    /// `psignw` semantics agree with a plain negate-if-negative there).
    fn with_sign_of(self, like: Self) -> Self;

    /// All-zero accumulators.
    fn acc_zero() -> Self::Acc;
    /// `acc + widen(a) * coeff`, per lane.
    fn acc_mul_add(acc: Self::Acc, a: Self, coeff: i16) -> Self::Acc;
    /// `acc + widen(a) * ca + widen(b) * cb`, per lane: two filter taps at
    /// once. ISAs with a paired multiply-add (x86 `pmaddwd`) override it.
    #[allow(clippy::inline_always)] // must inline into `#[target_feature]` kernels
    #[inline(always)]
    fn acc_mul_add2(acc: Self::Acc, a: Self, ca: i16, b: Self, cb: i16) -> Self::Acc {
        Self::acc_mul_add(Self::acc_mul_add(acc, a, ca), b, cb)
    }
    /// `acc + widen(a)`; only the total ([`Self::acc_sum`]) is meaningful.
    fn acc_add(acc: Self::Acc, a: Self) -> Self::Acc;
    /// `acc + widen(a)^2`; only the total ([`Self::acc_sum`]) is meaningful.
    fn acc_add_sq(acc: Self::Acc, a: Self) -> Self::Acc;
    /// Store `clamp((acc + round) >> shift, 0, 255)` per lane into the first
    /// `LANES` bytes of `dst` (accumulators from [`Self::acc_mul_add`]).
    fn acc_round_shift_store_u8(acc: Self::Acc, round: i32, shift: u32, dst: &mut [u8]);
    /// `(acc + round) >> shift` per lane, saturated to 16 bits
    /// (accumulators from [`Self::acc_mul_add`]).
    fn acc_round_shift_narrow(acc: Self::Acc, round: i32, shift: u32) -> Self;
    /// Sum of every accumulator.
    fn acc_sum(acc: Self::Acc) -> i64;
}

/// Further 16-bit operations of the loop-filter kernels, on the vector
/// types those kernels use: half-vector row access, so one vector can hold
/// two short block rows, and unsigned lane arithmetic.
pub(crate) trait I16FilterLanes: I16Lanes {
    /// Lanes `0..LANES / 2` from the start of `lo`, the others from the
    /// start of `hi`.
    fn load_halves(lo: &[i16], hi: &[i16]) -> Self;
    /// Store `clamp(lane, 0, 255)` of lanes `0..LANES / 2` into the start of
    /// `lo`, and of the others into the start of `hi`.
    fn store_u8_sat_halves(self, lo: &mut [u8], hi: &mut [u8]);
    /// Lane-wise minimum, comparing the lanes as unsigned 16-bit values.
    fn min_u16(self, other: Self) -> Self;
    /// Lane-wise `max(self - other, 0)`, on the lanes as unsigned 16-bit
    /// values.
    fn sub_sat_u16(self, other: Self) -> Self;
    /// −1 (all bits set) in the lanes where `self == other`, else 0.
    fn eq_mask(self, other: Self) -> Self;
    /// Lane-wise bitwise and.
    fn and(self, other: Self) -> Self;
    /// −1 (all bits set) in the lanes where `self > other` (signed), else 0.
    fn gt_mask(self, other: Self) -> Self;
    /// `if_set` in the lanes where the mask `self` is −1, `if_clear` where
    /// it is 0 (`self` from [`Self::gt_mask`], [`Self::eq_mask`], [`Self::and`]).
    fn blend(self, if_set: Self, if_clear: Self) -> Self;
}

/// Byte-shuffle pattern entry that yields a zero byte (the CDEF direction
/// search's patterns; AVX2 has its own search).
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
pub(crate) const ZERO_BYTE: u8 = 0x80;

/// Lane rearrangements on eight 16-bit lanes, and a pairwise multiply-add
/// into four 32-bit lanes: what the CDEF direction search needs.
pub(crate) trait I16x8Shuffles: I16Lanes {
    /// Four 32-bit lanes.
    type Wide: I32Lanes;

    /// Byte `i` of the result is byte `pattern[i]` of `self`, or zero where
    /// `pattern[i]` is 0x80 (every entry is below 16 or 0x80).
    fn shuffle_bytes(self, pattern: &[u8; 16]) -> Self;
    /// Lane `i + n` of the result is lane `i` of `self`; lanes below `n`
    /// are zero (`n <= 8`). For the 128-bit CDEF direction search.
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn lanes_up(self, n: usize) -> Self;
    /// Lane `i` of the result is lane `i + n` of `self`; lanes from `8 − n`
    /// are zero (`n <= 8`).
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn lanes_down(self, n: usize) -> Self;
    /// Lanes `0..4` of `self` and `other`, interleaved: `s0 o0 s1 o1 ..`.
    fn zip_lo(self, other: Self) -> Self;
    /// Lanes `4..8` of `self` and `other`, interleaved.
    fn zip_hi(self, other: Self) -> Self;
    /// `self[2k] * other[2k] + self[2k + 1] * other[2k + 1]` in 32-bit lane
    /// `k` (the products must not overflow their sum).
    fn madd(self, other: Self) -> Self::Wide;
    /// Lane `k` of the result is the sum of the lanes of `x[k]` (wrapping).
    fn sum_lanes4(x: [Self::Wide; 4]) -> Self::Wide;
}

/// Signed 16-bit lanes taken as pairs, with one 32-bit lane per pair: the
/// pairwise multiply-add of matrix products on 16-bit operands (`vvc_mts`).
pub(crate) trait I16Pairs: I16Lanes {
    /// One 32-bit lane per pair of 16-bit lanes.
    type Pairs: I32Lanes;

    /// Every pair `(first, second)`: even lanes `first`, odd lanes `second`.
    fn splat_pair(first: i16, second: i16) -> Self;
    /// `self[2k] * other[2k] + self[2k + 1] * other[2k + 1]` in 32-bit lane
    /// `k` (the sum of two 16-bit products cannot overflow unless both are
    /// `-32768 * -32768`).
    fn pair_dot(self, other: Self) -> Self::Pairs;
    /// The lanes of `low` then those of `high`, each saturated to 16 bits:
    /// the inverse of taking a vector as pairs.
    fn narrow_pairs(low: Self::Pairs, high: Self::Pairs) -> Self;
}

/// The two 16-bit halves of a 32-bit lane holding `first` in its low half.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn packed_pair(first: i16, second: i16) -> i32 {
    (i32::from(second) << 16) | i32::from(first as u16)
}

/// Sixteen unsigned 8-bit lanes and two eight-lane 16-bit accumulators: the
/// operations of the 8-bit CDEF kernel (`cdef::narrow`), which holds two
/// 8-sample rows or four 4-sample rows of a block per vector.
// Implemented by ISAs with an 8-bit CDEF kernel (NEON so far).
#[cfg(target_arch = "aarch64")]
#[allow(
    unsafe_code,
    reason = "the loads take pointers the kernel checked once per block"
)]
pub(crate) trait U8FilterLanes: Copy {
    /// Sixteen 16-bit sums, one per byte lane.
    type Acc: Copy;

    /// Bytes `0..8` from `top`, bytes `8..16` from `bottom`.
    ///
    /// # Safety
    ///
    /// Both point to eight readable bytes; the ISA is enabled.
    unsafe fn load_8x2(top: *const u8, bottom: *const u8) -> Self;
    /// Bytes `4r..4r + 4` from `rows[r]`.
    ///
    /// # Safety
    ///
    /// Each points to four readable bytes; the ISA is enabled.
    unsafe fn load_4x4(rows: [*const u8; 4]) -> Self;
    /// Inverse of [`Self::load_8x2`].
    fn store_8x2(self, top: &mut [u8], bottom: &mut [u8]);
    /// Inverse of [`Self::load_4x4`], into rows `stride` apart in `dst`.
    fn store_4x4(self, dst: &mut [u8], stride: usize);
    /// Every lane `value`.
    fn splat(value: u8) -> Self;
    /// `|self - other|`.
    fn abs_diff(self, other: Self) -> Self;
    /// All bits set in the lanes where `self > other`, else 0.
    fn gt(self, other: Self) -> Self;
    /// The count [`U8FilterLanes::shr`] takes for a right shift by `shift`
    /// (below 8), made once per block.
    fn shift_count(shift: u32) -> Self;
    /// Logical shift right by the shift `count` stands for.
    fn shr(self, count: Self) -> Self;
    /// `max(self - other, 0)`.
    fn sub_sat(self, other: Self) -> Self;
    /// Lane-wise minimum.
    fn min(self, other: Self) -> Self;
    /// Lane-wise maximum.
    fn max(self, other: Self) -> Self;
    /// Two's-complement negation of the lanes (as `i8`) where `mask` is all
    /// ones; the others unchanged.
    fn negate_where(self, mask: Self) -> Self;
    /// Lane-wise wrapping sum.
    fn add_wrapping(self, other: Self) -> Self;
    /// All sums zero.
    fn acc_zero() -> Self::Acc;
    /// `acc + weight * self`, the lanes read as `i8`.
    fn acc_mla(acc: Self::Acc, values: Self, weight: i8) -> Self::Acc;
    /// `clamp(self + ((8 + sum - (sum < 0)) >> 4), 0, 255)` per lane: libaom's
    /// rounding of the CDEF sum onto the centre sample.
    fn add_rounded(self, acc: Self::Acc) -> Self;
}
