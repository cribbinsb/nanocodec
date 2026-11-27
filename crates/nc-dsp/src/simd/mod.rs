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

#[cfg(target_arch = "x86_64")]
pub(crate) mod gfni;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon_dot;

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
#[allow(unsafe_code)]
pub(crate) mod sve2;

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
    /// Store into the first `LANES` values of `dst`, each lane saturated to
    /// the signed 16-bit range: [`I32Lanes::clamp16`] then a narrowing
    /// store (16-bit transform stages). ISAs override it with one
    /// saturating narrow.
    #[inline]
    fn store_sat16(self, dst: &mut [i16]) {
        const MAX_LANES: usize = 16;
        assert!(Self::LANES <= MAX_LANES);
        let mut wide = [0; MAX_LANES];
        self.store(&mut wide);
        for (narrow, &value) in dst[..Self::LANES].iter_mut().zip(&wide) {
            *narrow = saturate_i16(value);
        }
    }
    /// The first `2 * LANES` values of `src`, sign-extended, as two vectors.
    /// ISAs override it with one full 128-bit load (NEON, SIMD128).
    #[inline]
    fn load_i16_pair(src: &[i16]) -> (Self, Self) {
        (Self::load_i16(src), Self::load_i16(&src[Self::LANES..]))
    }
    /// [`I32Lanes::store_sat16`] of `low` then `high` into the first
    /// `2 * LANES` values of `dst`. ISAs override it to narrow the pair into
    /// one store (one full 128-bit store on NEON and SIMD128).
    #[inline]
    fn store_sat16_pair(low: Self, high: Self, dst: &mut [i16]) {
        low.store_sat16(dst);
        high.store_sat16(&mut dst[Self::LANES..]);
    }
    /// `dst[i] = clamp(dst[i] + lane i, 0, 255)` for the first `LANES` bytes.
    fn add_to_u8(self, dst: &mut [u8]);
    /// Add one constant `residual` to an `n`x`n` block of `u8` samples at
    /// `stride`, clipping to `0..=255`. Vector implementations route through
    /// `constant_add` (16- or 32-byte saturating ops).
    #[inline]
    fn add_constant_u8(residual: i32, n: usize, dst: &mut [u8], stride: usize) {
        let lanes = Self::splat(residual);
        for r in 0..n {
            for c in (0..n).step_by(Self::LANES) {
                lanes.add_to_u8(&mut dst[r * stride + c..]);
            }
        }
    }
    /// `dst[i] = clamp(dst[i] + lane i, 0, max_val)` for the first `LANES`
    /// `u16` samples (`0 <= max_val <= 65535`).
    #[inline]
    #[cfg_attr(
        not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )),
        allow(dead_code, reason = "u16 SIMD kernels target vector builds")
    )]
    fn add_to_u16(self, dst: &mut [u16], max_val: i32) {
        const MAX_LANES: usize = 16;
        assert!(Self::LANES <= MAX_LANES);
        let mut wide = [0i32; MAX_LANES];
        self.store(&mut wide);
        for (sample, &delta) in dst[..Self::LANES].iter_mut().zip(&wide) {
            *sample = (i32::from(*sample) + delta).clamp(0, max_val) as u16;
        }
    }
    /// Store the `LANES` x `LANES` tile formed by `rows` (`rows.len() ==
    /// LANES`), transposed: lane `c` of `rows[r]` goes to
    /// `dst[c * dst_stride + r]`. `dst` may be uninitialised; every element
    /// of the tile is written.
    fn store_transposed(rows: &[Self], dst: &mut [MaybeUninit<i32>], dst_stride: usize);
    /// [`I32Lanes::store_transposed`] into 16-bit storage, each lane
    /// saturated as by [`I32Lanes::store_sat16`]. ISAs override it to
    /// narrow the transposed registers directly.
    #[inline]
    #[allow(unsafe_code)]
    fn store_transposed_sat16(rows: &[Self], dst: &mut [MaybeUninit<i16>], dst_stride: usize) {
        const MAX_LANES: usize = 16;
        assert!(Self::LANES <= MAX_LANES);
        let mut tile = [MaybeUninit::<i32>::uninit(); MAX_LANES * MAX_LANES];
        Self::store_transposed(rows, &mut tile, Self::LANES);
        // SAFETY: store_transposed wrote every cell of this complete tile.
        let tile = unsafe {
            core::slice::from_raw_parts(tile.as_ptr().cast::<i32>(), Self::LANES * Self::LANES)
        };
        for row in 0..Self::LANES {
            let values = &tile[row * Self::LANES..][..Self::LANES];
            let dst = &mut dst[row * dst_stride..][..Self::LANES];
            for (narrow, &value) in dst.iter_mut().zip(values) {
                *narrow = MaybeUninit::new(saturate_i16(value));
            }
        }
    }
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
    /// Transpose a square lane tile and add its rows to a `u16` picture,
    /// clipping each sample to `0..=max_val`.
    #[inline]
    #[allow(unsafe_code)]
    #[cfg_attr(
        not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )),
        allow(dead_code, reason = "u16 SIMD kernels target vector ISAs")
    )]
    fn add_transposed_u16(rows: &[Self], dst: &mut [u16], dst_stride: usize, max_val: i32) {
        const MAX_LANES: usize = 16;
        assert!(Self::LANES <= MAX_LANES);
        let mut tile = [MaybeUninit::<i32>::uninit(); MAX_LANES * MAX_LANES];
        Self::store_transposed(rows, &mut tile, Self::LANES);
        // SAFETY: store_transposed wrote every cell of this complete tile.
        let tile = unsafe {
            core::slice::from_raw_parts(tile.as_ptr().cast::<i32>(), Self::LANES * Self::LANES)
        };
        for row in 0..Self::LANES {
            Self::load(&tile[row * Self::LANES..])
                .add_to_u16(&mut dst[row * dst_stride..], max_val);
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
/// Entries of an [`I16Lanes::lookup_u12`] table: 4096 and one of padding.
pub(crate) const HIGH_LOOKUP_TABLE: usize = 4097;
/// Lanes of the widest [`I16Lanes`] vector (AVX-512).
pub(crate) const MAX_I16_LANES: usize = 32;

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
    /// The first `LANES` unsigned 16-bit samples of `src` (reinterpreted as
    /// `i16`; non-negative for 8..=12-bit samples).
    #[inline(always)]
    #[allow(unsafe_code, clippy::inline_always)]
    fn load_u16(src: &[u16]) -> Self {
        let src = &src[..Self::LANES];
        // SAFETY: `u16` and `i16` have identical size, alignment, and bit validity.
        Self::load(unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<i16>(), Self::LANES) })
    }
    /// Store into the first `LANES` unsigned 16-bit samples of `dst`.
    #[inline(always)]
    #[allow(unsafe_code, clippy::inline_always)]
    fn store_u16(self, dst: &mut [u16]) {
        let dst = &mut dst[..Self::LANES];
        // SAFETY: `u16` and `i16` have identical size, alignment, and bit validity.
        self.store(unsafe {
            core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<i16>(), Self::LANES)
        });
    }
    /// [`Self::load_u16`] from a pointer, for kernels whose window was
    /// validated once at entry.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reading `LANES` `u16` elements.
    #[inline(always)]
    #[allow(unsafe_code, clippy::inline_always)]
    unsafe fn load_u16_at(src: *const u16) -> Self {
        // SAFETY: caller guarantees `LANES` readable `u16`s.
        Self::load(unsafe { core::slice::from_raw_parts(src.cast::<i16>(), Self::LANES) })
    }
    /// [`Self::store_u16`] to a pointer, for kernels whose output slice was
    /// validated once at entry.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writing `LANES` `u16` elements.
    #[inline(always)]
    #[allow(unsafe_code, clippy::inline_always)]
    unsafe fn store_u16_at(self, dst: *mut u16) {
        // SAFETY: caller guarantees `LANES` writable `u16`s.
        self.store(unsafe { core::slice::from_raw_parts_mut(dst.cast::<i16>(), Self::LANES) });
    }
    /// The first `LANES` values of `src`, sign-extended (8-bit film grain).
    /// The default builds the vector in memory; the grain kernels' ISAs
    /// override it with a widening load.
    #[inline(always)]
    #[allow(clippy::inline_always, reason = "per vector in the grain kernels")]
    fn load_i8(src: &[i8]) -> Self {
        let mut lanes = [0i16; MAX_I16_LANES];
        for (lane, &value) in lanes.iter_mut().zip(&src[..Self::LANES]) {
            *lane = i16::from(value);
        }
        Self::load(&lanes)
    }
    /// Store `clamp(lane, -128, 127)` into the first `LANES` values of
    /// `dst`. The default narrows in memory; the grain kernels' ISAs
    /// override it with a saturating narrow.
    #[inline(always)]
    #[allow(clippy::inline_always, reason = "per vector in the grain kernels")]
    fn store_i8(self, dst: &mut [i8]) {
        let mut lanes = [0i16; MAX_I16_LANES];
        self.store(&mut lanes);
        for (out, &lane) in dst[..Self::LANES].iter_mut().zip(&lanes) {
            *out = lane.clamp(i16::from(i8::MIN), i16::from(i8::MAX)) as i8;
        }
    }
    /// Lane-wise wrapping add.
    fn add(self, other: Self) -> Self;
    /// Lane-wise saturating add (`clamp(a + b, i16::MIN, i16::MAX)`).
    fn add_sat(self, other: Self) -> Self;
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
    /// Lane-wise logical right shift of unsigned 16-bit lanes by `bits <= 15`.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    #[cfg_attr(
        not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )),
        allow(dead_code, reason = "u16 SIMD kernels target vector builds")
    )]
    fn srl(self, bits: u32) -> Self {
        let mut lanes = [0i16; MAX_I16_LANES];
        self.store(&mut lanes);
        for lane in &mut lanes[..Self::LANES] {
            *lane = ((*lane as u16) >> bits) as i16;
        }
        Self::load(&lanes)
    }
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
    /// Lane `i` is `table[clamp(self[i], 0, 4095)]` (a 4096-entry lookup for
    /// 10- and 12-bit film grain, padded by one entry so 32-bit gathers stay
    /// inside).
    #[inline(always)]
    #[allow(clippy::inline_always)]
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(dead_code, reason = "u16 SIMD kernels target x86_64 and aarch64")
    )]
    fn lookup_u12(self, table: &[i16; HIGH_LOOKUP_TABLE]) -> Self {
        let mut lanes = [0i16; MAX_I16_LANES];
        self.max(Self::splat(0))
            .min(Self::splat(4095))
            .store(&mut lanes);
        for lane in &mut lanes[..Self::LANES] {
            *lane = table[(*lane as usize) & 4095];
        }
        Self::load(&lanes)
    }
    /// Lane `i` is `(pairs[2 * i] + pairs[2 * i + 1] + 1) >> 1` for `LANES`
    /// `u16` pairs whose samples fit in `0..=16383`.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(dead_code, reason = "u16 SIMD kernels target x86_64 and aarch64")
    )]
    fn pair_avg_u16(pairs: &[u16]) -> Self {
        let mut lanes = [0i16; MAX_I16_LANES];
        let (chunks, _) = pairs[..2 * Self::LANES].as_chunks::<2>();
        for (lane, &[a, b]) in lanes.iter_mut().zip(chunks) {
            *lane = ((u32::from(a) + u32::from(b) + 1) >> 1) as i16;
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
    /// `src[2i] + src[2i + 1]` over the first `2 · LANES` bytes of `src`:
    /// the horizontal pair sums that halve a row of a mask or plane. x86
    /// computes it with one `pmaddubsw` against ones, NEON with `uaddlp`,
    /// SIMD128 with a pairwise extending add; the default widens each pair.
    #[allow(
        clippy::inline_always,
        reason = "lane operations must inline into the kernels"
    )]
    #[inline(always)]
    fn pair_sum_u8(src: &[u8]) -> Self {
        let mut lanes = [0i16; MAX_I16_LANES];
        let (pairs, _) = src[..2 * Self::LANES].as_chunks::<2>();
        for (lane, &[a, b]) in lanes.iter_mut().zip(pairs) {
            *lane = i16::from(a) + i16::from(b);
        }
        Self::load(&lanes)
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
    /// `widen(a) * widen(wa) + widen(b) * widen(wb)` per lane in [`Self::Acc`]
    /// layout, ready for [`Self::acc_round_shift_narrow`].
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(dead_code, reason = "u16 SIMD kernels target x86_64 and aarch64")
    )]
    fn acc_vec_mul_add2(a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        let (mut ab, mut wab, mut bb, mut wbb) = (
            [0i16; MAX_I16_LANES],
            [0i16; MAX_I16_LANES],
            [0i16; MAX_I16_LANES],
            [0i16; MAX_I16_LANES],
        );
        a.store(&mut ab);
        wa.store(&mut wab);
        b.store(&mut bb);
        wb.store(&mut wbb);
        let mut acc = Self::acc_zero();
        for i in 0..Self::LANES {
            let mut unit = [0i16; MAX_I16_LANES];
            unit[i] = 1;
            let basis = Self::load(&unit);
            let p = i32::from(ab[i]) * i32::from(wab[i]) + i32::from(bb[i]) * i32::from(wbb[i]);
            let hi = (p >> 15) as i16;
            let lo = (p - (i32::from(hi) << 15)) as i16;
            acc = Self::acc_mul_add(acc, basis.mul(Self::splat(lo)), 1);
            acc = Self::acc_mul_add(acc, basis, hi);
            acc = Self::acc_mul_add(acc, basis.mul(Self::splat(hi)), i16::MAX);
        }
        acc
    }
    /// `acc + widen(a) * widen(wa) + widen(b) * widen(wb)` per lane in
    /// [`Self::Acc`] layout, ready for [`Self::acc_round_shift_narrow`].
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn acc_vec_fma2(mut acc: Self::Acc, a: Self, wa: Self, b: Self, wb: Self) -> Self::Acc {
        let (mut ab, mut wab, mut bb, mut wbb) = (
            [0i16; MAX_I16_LANES],
            [0i16; MAX_I16_LANES],
            [0i16; MAX_I16_LANES],
            [0i16; MAX_I16_LANES],
        );
        a.store(&mut ab);
        wa.store(&mut wab);
        b.store(&mut bb);
        wb.store(&mut wbb);
        for i in 0..Self::LANES {
            let mut unit = [0i16; MAX_I16_LANES];
            unit[i] = 1;
            let basis = Self::load(&unit);
            let p = i32::from(ab[i]) * i32::from(wab[i]) + i32::from(bb[i]) * i32::from(wbb[i]);
            let hi = (p >> 15) as i16;
            let lo = (p - (i32::from(hi) << 15)) as i16;
            acc = Self::acc_mul_add(acc, basis.mul(Self::splat(lo)), 1);
            acc = Self::acc_mul_add(acc, basis, hi);
            acc = Self::acc_mul_add(acc, basis.mul(Self::splat(hi)), i16::MAX);
        }
        acc
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
    /// Eight-tap horizontal interpolation when this ISA can do four
    /// halfword products per dot-product lane.
    ///
    /// # Safety
    /// The caller enables this type's ISA. `src` covers `Self::LANES + 7`
    /// signed samples and the rounded dot sums fit in i32.
    #[allow(
        unsafe_code,
        reason = "this optional SIMD leaf requires a checked row pointer"
    )]
    #[inline]
    unsafe fn horizontal_dot_8tap(
        _src: *const i16,
        _taps: [i16; 8],
        _round: i32,
        _shift: u32,
    ) -> Option<Self> {
        None
    }

    /// True when [`Self::horiz_taps_acc`] provides an ISA-specific fast path
    /// that extracts shifted windows from two loads (`p[0..LANES]` and
    /// `p[N - 1..LANES + N - 1]`) and multiplies by indexed tap lanes.
    const FAST_HORIZ_EXT: bool = false;
    /// Compute `Σ_k cx[k] * p[x + k]` from `s0 = p[0..LANES]` and
    /// `s_last = p[N - 1..LANES + N - 1]`. Only called when
    /// [`Self::FAST_HORIZ_EXT`] is true and `N` is 4 or 8.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn horiz_taps_acc<const N: usize>(
        _s0: Self,
        _s_last: Self,
        _six_x: bool,
        _cx: &[i16; N],
    ) -> Self::Acc {
        unreachable!("horiz_taps_acc requires FAST_HORIZ_EXT")
    }
}

/// Further 16-bit operations of the loop-filter kernels, on the vector
/// types those kernels use: half-vector row access, so one vector can hold
/// two short block rows, and unsigned lane arithmetic.
pub(crate) trait I16FilterLanes: I16Lanes {
    /// Lanes `0..LANES / 2` from the start of `lo`, the others from the
    /// start of `hi`.
    fn load_halves(lo: &[i16], hi: &[i16]) -> Self;
    /// Lanes `0..LANES / 2` zero-extended from `lo`, and `LANES / 2..LANES`
    /// zero-extended from `hi`.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn load_u8_halves(lo: &[u8], hi: &[u8]) -> Self {
        let half = Self::LANES / 2;
        let mut buf = [0u8; MAX_I16_LANES];
        buf[..half].copy_from_slice(&lo[..half]);
        buf[half..Self::LANES].copy_from_slice(&hi[..half]);
        Self::load_u8(&buf)
    }
    /// Store `clamp(lane, 0, 255)` of lanes `0..LANES / 2` into the start of
    /// `lo`, and of the others into the start of `hi`.
    fn store_u8_sat_halves(self, lo: &mut [u8], hi: &mut [u8]);
    /// Store lanes `0..LANES / 2` into `lo` and `LANES / 2..LANES` into `hi`.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn store_halves(self, lo: &mut [i16], hi: &mut [i16]) {
        let half = Self::LANES / 2;
        let mut buf = [0i16; MAX_I16_LANES];
        self.store(&mut buf);
        lo[..half].copy_from_slice(&buf[..half]);
        hi[..half].copy_from_slice(&buf[half..Self::LANES]);
    }
    /// Lanes `0..LANES / 2` from the high half of `self` (`LANES / 2..LANES`)
    /// and lanes `LANES / 2..LANES` from the low half of `next` (`0..LANES / 2`).
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn combine_hi_lo(self, next: Self) -> Self {
        let half = Self::LANES / 2;
        let mut a = [0i16; MAX_I16_LANES];
        let mut b = [0i16; MAX_I16_LANES];
        self.store(&mut a);
        next.store(&mut b);
        a.copy_within(half..Self::LANES, 0);
        a[half..Self::LANES].copy_from_slice(&b[..half]);
        Self::load(&a)
    }
    /// Transpose each 8-lane (128-bit) group of `rows` independently (`LANES`
    /// must be a multiple of 8).
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn transpose_groups8(rows: [Self; 8]) -> [Self; 8] {
        let mut src = [[0i16; MAX_I16_LANES]; 8];
        for (r, v) in rows.into_iter().enumerate() {
            v.store(&mut src[r]);
        }
        let mut dst = [[0i16; MAX_I16_LANES]; 8];
        let groups = Self::LANES / 8;
        for g in 0..groups {
            let base = 8 * g;
            for r in 0..8 {
                for c in 0..8 {
                    dst[c][base + r] = src[r][base + c];
                }
            }
        }
        [
            Self::load(&dst[0]),
            Self::load(&dst[1]),
            Self::load(&dst[2]),
            Self::load(&dst[3]),
            Self::load(&dst[4]),
            Self::load(&dst[5]),
            Self::load(&dst[6]),
            Self::load(&dst[7]),
        ]
    }
    /// Quarter-vector row load (`LANES / 4` lanes from each of four rows).
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn load_quarters(rows: [&[i16]; 4]) -> Self {
        let q = Self::LANES / 4;
        let mut buf = [0i16; MAX_I16_LANES];
        for (i, row) in rows.into_iter().enumerate() {
            buf[i * q..(i + 1) * q].copy_from_slice(&row[..q]);
        }
        Self::load(&buf)
    }
    /// Quarter-vector `u8` row load (`LANES / 4` bytes zero-extended from each
    /// of four rows).
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn load_u8_quarters(rows: [&[u8]; 4]) -> Self {
        let q = Self::LANES / 4;
        let mut buf = [0u8; MAX_I16_LANES];
        for (i, row) in rows.into_iter().enumerate() {
            buf[i * q..(i + 1) * q].copy_from_slice(&row[..q]);
        }
        Self::load_u8(&buf)
    }
    /// Quarter-vector row store (`LANES / 4` lanes into each of four rows).
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn store_quarters(self, rows: [&mut [i16]; 4]) {
        let q = Self::LANES / 4;
        let mut buf = [0i16; MAX_I16_LANES];
        self.store(&mut buf);
        for (i, row) in rows.into_iter().enumerate() {
            row[..q].copy_from_slice(&buf[i * q..(i + 1) * q]);
        }
    }
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
    /// Whether any lane in `self` is non-zero.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    #[cfg_attr(
        not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )),
        allow(dead_code, reason = "u16 SIMD kernels target vector builds")
    )]
    fn any_nonzero(self) -> bool {
        let mut buf = [0i16; MAX_I16_LANES];
        self.store(&mut buf);
        buf[..Self::LANES].iter().any(|&v| v != 0)
    }
    /// Deinterleave the first `2 * LANES` `u8` samples of `src` into `(even,
    /// odd)`, zero-extended to `i16`.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn load_u8_deinterleaved(src: &[u8]) -> (Self, Self) {
        let pairs = &src[..2 * Self::LANES];
        let mut even = [0i16; MAX_I16_LANES];
        let mut odd = [0i16; MAX_I16_LANES];
        for i in 0..Self::LANES {
            even[i] = i16::from(pairs[2 * i]);
            odd[i] = i16::from(pairs[2 * i + 1]);
        }
        (Self::load(&even), Self::load(&odd))
    }
    /// Deinterleave the first `2 * LANES` `u16` samples of `src` into `(even,
    /// odd)`, where lane `i` of `even` is `src[2 * i]` and lane `i` of `odd`
    /// is `src[2 * i + 1]`.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn load_u16_deinterleaved(src: &[u16]) -> (Self, Self) {
        let pairs = &src[..2 * Self::LANES];
        let mut even = [0i16; MAX_I16_LANES];
        let mut odd = [0i16; MAX_I16_LANES];
        for i in 0..Self::LANES {
            even[i] = pairs[2 * i] as i16;
            odd[i] = pairs[2 * i + 1] as i16;
        }
        (Self::load(&even), Self::load(&odd))
    }
    /// Combine the upper `LANES / 2` lanes of `self` with the lower `LANES / 2`
    /// lanes of `next`: `[self[LANES/2..], next[..LANES/2]]`.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn combine_mid_halves(self, next: Self) -> Self {
        let half = Self::LANES / 2;
        let (mut a, mut b) = ([0i16; MAX_I16_LANES], [0i16; MAX_I16_LANES]);
        self.store(&mut a);
        next.store(&mut b);
        let mut out = [0i16; MAX_I16_LANES];
        out[..half].copy_from_slice(&a[half..Self::LANES]);
        out[half..Self::LANES].copy_from_slice(&b[..half]);
        Self::load(&out)
    }
    /// Shift by one quarter (`LANES / 4` lanes): `[self[q..LANES], next[..q]]`.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn shift_quarter(self, next: Self) -> Self {
        let q = Self::LANES / 4;
        let rem = Self::LANES - q;
        let (mut a, mut b) = ([0i16; MAX_I16_LANES], [0i16; MAX_I16_LANES]);
        self.store(&mut a);
        next.store(&mut b);
        let mut out = [0i16; MAX_I16_LANES];
        out[..rem].copy_from_slice(&a[q..Self::LANES]);
        out[rem..Self::LANES].copy_from_slice(&b[..q]);
        Self::load(&out)
    }
    /// Pre-expand all 6 ALF tap pairs across `LANES / 4` 4-wide blocks in the
    /// ISA's native pair layout (interleaved `[t_{2p}, t_{2p+1}]` on dot-product
    /// ISAs, block-replicated `[t_{2p}, t_{2p+1}]` on widening-MLA ISAs).
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn expand_alf_params(blocks: &[[i16; 12]]) -> [[Self; 2]; 6] {
        [
            Self::expand_alf_pair(blocks, 0),
            Self::expand_alf_pair(blocks, 1),
            Self::expand_alf_pair(blocks, 2),
            Self::expand_alf_pair(blocks, 3),
            Self::expand_alf_pair(blocks, 4),
            Self::expand_alf_pair(blocks, 5),
        ]
    }
    /// Broadcast a uniform ALF tap pair `(t0, t1)` across all lanes in the
    /// ISA's native [`Self::acc_alf_clipped_pair`] layout (chroma 5x5 ALF).
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn splat_alf_pair(t0: i16, t1: i16) -> [Self; 2] {
        [Self::splat(t0), Self::splat(t1)]
    }
    /// Accumulate one symmetric ALF tap pair `(2p, 2p + 1)` with clipping:
    /// `acc + c0 * (clip(a0 - c, lim0) + clip(b0 - c, lim0)) + c1 * (clip(a1 - c, lim1) + clip(b1 - c, lim1))`.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn acc_alf_clipped_pair(
        acc: Self::Acc,
        (a0, b0): (Self, Self),
        (a1, b1): (Self, Self),
        centre: Self,
        lim: [Self; 2],
        coeff: [Self; 2],
    ) -> Self::Acc {
        let zero = Self::splat(0);
        let (neg0, neg1) = (zero.sub(lim[0]), zero.sub(lim[1]));
        let d0 = a0
            .sub(centre)
            .max(neg0)
            .min(lim[0])
            .add(b0.sub(centre).max(neg0).min(lim[0]));
        let d1 = a1
            .sub(centre)
            .max(neg1)
            .min(lim[1])
            .add(b1.sub(centre).max(neg1).min(lim[1]));
        Self::acc_vec_fma2(acc, d0, coeff[0], d1, coeff[1])
    }
    /// Accumulate one linear ALF tap pair `(2p, 2p + 1)` (`d0 * c0 + d1 * c1`)
    /// using [`Self::expand_alf_params`] weights.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn acc_alf_linear_pair(acc: Self::Acc, d0: Self, d1: Self, coeff: [Self; 2]) -> Self::Acc {
        Self::acc_vec_fma2(acc, d0, coeff[0], d1, coeff[1])
    }
    /// Expand tap pair `(2 * p, 2 * p + 1)` across `LANES / 4` 4-wide ALF
    /// blocks (`blocks[0..LANES / 4]`) into two vectors, keeping live register
    /// pressure bounded inside the tap loop.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn expand_alf_pair(blocks: &[[i16; 12]], p: usize) -> [Self; 2] {
        let groups = &blocks[..Self::LANES / 4];
        let (mut l0, mut l1) = ([0i16; MAX_I16_LANES], [0i16; MAX_I16_LANES]);
        for (b, block) in groups.iter().enumerate() {
            l0[4 * b..4 * b + 4].fill(block[2 * p]);
            l1[4 * b..4 * b + 4].fill(block[2 * p + 1]);
        }
        [Self::load(&l0), Self::load(&l1)]
    }
}

/// Independent 8-word tiles in 128-bit lanes. The 16/32-word adapters
/// transpose two/four deblocking tiles together without crossing tiles.
pub(crate) trait I16x8Tiles: I16FilterLanes {
    /// Repeat each value over one independent four-line decision segment.
    fn splat_segments(values: &[i16; 8]) -> Self;
    fn zip_tiles_lo(self, other: Self) -> Self;
    fn zip_tiles_hi(self, other: Self) -> Self;
    fn shuffle_tiles(self, pattern: &[u8; 16]) -> Self;
    /// Eight bytes from each of the first `LANES / 8` rows, widened.
    fn load_tiles8(rows: [&[u8]; 4]) -> Self;
    /// Read a vertical tile of eight rows of sixteen bytes, then widen.
    /// # Safety
    /// The tile at `origin` with this row stride lies inside `data`.
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_code)]
    unsafe fn load_u8_vertical16(data: &[u8], origin: usize, stride: usize) -> [Self; 16];
    /// Narrow before writing the same tile.
    /// # Safety
    /// The tile at `origin` with this row stride lies inside `data`.
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_code)]
    unsafe fn store_u8_vertical16(data: &mut [u8], origin: usize, stride: usize, taps: &[Self; 16]);
    /// Narrow each 8-word tile into the corresponding eight-byte row.
    fn store_tiles8(self, rows: [&mut [u8]; 4]);
    /// Eight `u16` samples (`0..=32767`) from each of the first `LANES / 8` rows.
    #[cfg_attr(
        not(target_arch = "x86_64"),
        allow(
            dead_code,
            reason = "multi-piece u16 tile batching requires >= 16-lane vectors"
        )
    )]
    fn load_u16_tiles8(rows: [&[u16]; 4]) -> Self;
    /// Clamp each 8-word tile to `0..=max_val` and store into the corresponding row.
    #[cfg_attr(
        not(target_arch = "x86_64"),
        allow(
            dead_code,
            reason = "multi-piece u16 tile batching requires >= 16-lane vectors"
        )
    )]
    fn store_u16_tiles8(self, rows: [&mut [u16]; 4], max_val: i16);
}

macro_rules! eight_word_tiles {
    ($ty:ty) => {
        #[allow(clippy::inline_always)]
        impl I16x8Tiles for $ty {
            #[inline(always)]
            fn splat_segments(values: &[i16; 8]) -> Self {
                const SELECT: [u8; 16] = [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3];
                I16x8Shuffles::shuffle_bytes(Self::load(values), &SELECT)
            }
            #[inline(always)]
            fn zip_tiles_lo(self, other: Self) -> Self {
                I16x8Shuffles::zip_lo(self, other)
            }
            #[inline(always)]
            fn zip_tiles_hi(self, other: Self) -> Self {
                I16x8Shuffles::zip_hi(self, other)
            }
            #[inline(always)]
            fn shuffle_tiles(self, pattern: &[u8; 16]) -> Self {
                I16x8Shuffles::shuffle_bytes(self, pattern)
            }
            #[inline(always)]
            fn load_tiles8(rows: [&[u8]; 4]) -> Self {
                Self::load_u8(rows[0])
            }
            #[cfg(target_arch = "aarch64")]
            #[inline(always)]
            #[allow(unsafe_code)]
            unsafe fn load_u8_vertical16(data: &[u8], origin: usize, stride: usize) -> [Self; 16] {
                // SAFETY: the caller checked the entire tile.
                unsafe { Self::load_u8_vertical16(data, origin, stride) }
            }
            #[cfg(target_arch = "aarch64")]
            #[inline(always)]
            #[allow(unsafe_code)]
            unsafe fn store_u8_vertical16(
                data: &mut [u8],
                origin: usize,
                stride: usize,
                taps: &[Self; 16],
            ) {
                // SAFETY: the caller checked the entire tile.
                unsafe { Self::store_u8_vertical16(data, origin, stride, taps) }
            }
            #[inline(always)]
            fn store_tiles8(self, rows: [&mut [u8]; 4]) {
                self.store_u8_sat(rows.into_iter().next().expect("first tile"));
            }
            #[inline(always)]
            fn load_u16_tiles8(rows: [&[u16]; 4]) -> Self {
                Self::load_u16(rows[0])
            }
            #[inline(always)]
            fn store_u16_tiles8(self, rows: [&mut [u16]; 4], max_val: i16) {
                self.max(Self::splat(0))
                    .min(Self::splat(max_val))
                    .store_u16(rows.into_iter().next().expect("first tile"));
            }
        }
    };
}
#[cfg(target_arch = "x86_64")]
eight_word_tiles!(avx2::I16x8);
#[cfg(target_arch = "aarch64")]
eight_word_tiles!(neon::I16x8);
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
eight_word_tiles!(simd128::I16x8);

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

    /// Signed rounding right shift, with `SHIFT` in 1..=15. Q15 rounding
    /// by the corresponding power of two is exact for every signed word.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn rounded_shr<const SHIFT: i32>(self) -> Self {
        self.mul_q15_round(Self::splat(1 << (15 - SHIFT)))
    }

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
    /// Per-lane `self[i]^2 + other[i]^2`, split into low/high four lanes.
    /// Products and their sum must fit i32. Native widening MACs can avoid
    /// interleaving the inputs merely to use a pairwise dot product.
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn paired_squares(self, other: Self) -> [Self::Wide; 2] {
        let low = self.zip_lo(other);
        let high = self.zip_hi(other);
        [low.madd(low), high.madd(high)]
    }
    /// Lane `k` of the result is the sum of the lanes of `x[k]` (wrapping).
    fn sum_lanes4(x: [Self::Wide; 4]) -> Self::Wide;
}

/// 32-bit lanes whose values fit `i16`, multiplied two vectors at a time:
/// each lane packs a value of two vectors as a 16-bit pair, and one pairwise
/// multiply-add by two constants (`vpmaddwd`, `i32x4.dot_i16x8_s`) does two
/// products per lane (the AV2 inverse DCT's butterflies, `av2_txfm::paired`).
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
pub(crate) trait WordPairLanes: I32Lanes {
    /// Transpose a square of `LANES` input vectors into `LANES` outputs.
    fn transpose_registers(rows: &[Self], columns: &mut [Self]);

    /// One 16-bit pair per lane.
    type Pairs: Copy;
    /// Lane `i` holds (`self[i]`, `high[i]`) truncated to 16 bits each:
    /// exact when every value fits `i16`.
    fn word_pairs(self, high: Self) -> Self::Pairs;
    /// Pack adjacent vectors into the first `N / 2` slots after verifying
    /// that every packed lane fits `i16`. The caller can keep its wide path when
    /// the range check is unavailable or fails.
    fn checked_word_pairs<const N: usize>(_values: &[Self; N]) -> Option<[Self::Pairs; N]> {
        None
    }
    /// `low * w0 + high * w1` per lane, exact in `i32`, with the weights
    /// packed as `weights = (w1 << 16) | (w0 as u16)`.
    fn pair_dot(pairs: Self::Pairs, weights: i32) -> Self;
}

/// Signed 16-bit lanes taken as pairs, with one 32-bit lane per pair: the
/// pairwise multiply-add of matrix products on 16-bit operands (`vvc_mts`).
pub(crate) trait I16Pairs: I16Lanes {
    /// Whether [`Self::pair_dot`] executes as a single native dot-product
    /// instruction (`vpmaddwd`, `i32x4.dot_i16x8_s`).
    const FAST_PAIR_DOT: bool = true;
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
    /// Interleave the low 16 bits of `even` and `odd`:
    /// `[even[0], odd[0], even[1], odd[1], ...]` (exact when every lane of
    /// `even` and `odd` fits signed `i16`).
    fn interleave_pairs(even: Self::Pairs, odd: Self::Pairs) -> Self;
}

/// `value` saturated to the signed 16-bit range (one lane of
/// [`I32Lanes::store_sat16`]).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn saturate_i16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

/// The two 16-bit halves of a 32-bit lane holding `first` in its low half.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn packed_pair(first: i16, second: i16) -> i32 {
    (i32::from(second) << 16) | i32::from(first as u16)
}

/// Sample load/store adapter so `i16`-lane SIMD kernels (`vvc_alf::high_simd`,
/// `mc::precision_simd`) instantiate once at `u8` and once at `u16` with zero
/// runtime overhead on either path.
#[allow(unsafe_code)]
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code, reason = "SIMD sample adapters target vector builds")
)]
pub(crate) trait SimdSample: nc_common::Sample {
    /// Load `V::LANES` samples from `src`, zero-extended to `i16` lanes.
    fn load_slice<V: I16Lanes>(src: &[Self]) -> V;
    /// Load `V::LANES` samples from a validated pointer, zero-extended.
    ///
    /// # Safety
    /// `src` must be valid for reading `V::LANES` elements of `Self`.
    unsafe fn load_lanes<V: I16Lanes>(src: *const Self) -> V;
    /// Clamp `v` to `0..=max_val` (`0..=255` for `u8`) and store `V::LANES`
    /// samples into `dst`.
    fn store_clamped_slice<V: I16Lanes>(v: V, zero: V, max_val: V, dst: &mut [Self]);
    /// Clamp `v` to `0..=max_val` (`0..=255` for `u8`) and store `V::LANES`
    /// samples at `dst`.
    ///
    /// # Safety
    /// `dst` must be valid for writing `V::LANES` elements of `Self`.
    unsafe fn store_clamped<V: I16Lanes>(v: V, zero: V, max_val: V, dst: *mut Self);
    /// Deinterleave `2 * V::LANES` samples of `src` into `(even, odd)`.
    fn load_deinterleaved<V: I16FilterLanes>(src: &[Self]) -> (V, V);
    /// Load `V::LANES / 2` samples at `x` from each of `p0` and `p1`.
    ///
    /// # Safety
    /// `p0.add(x)` and `p1.add(x)` must each cover `V::LANES / 2` readable elements.
    unsafe fn load_2rows<V: I16FilterLanes>(p0: *const Self, p1: *const Self, x: usize) -> V;
    /// Load `V::LANES / 4` samples at `x` from each of `p[0..4]`.
    ///
    /// # Safety
    /// Each `p[k].add(x)` must cover `V::LANES / 4` readable elements.
    unsafe fn load_4rows<V: I16FilterLanes>(p: [*const Self; 4], x: usize) -> V;
    /// Load two adjacent samples at `k` from each of four rows (`8` lanes).
    fn load_w2x4<V: I16Lanes>(rows: [&[Self]; 4], k: usize) -> V;
}

#[allow(unsafe_code, clippy::inline_always)]
impl SimdSample for u8 {
    #[inline(always)]
    fn load_slice<V: I16Lanes>(src: &[u8]) -> V {
        V::load_u8(src)
    }
    #[inline(always)]
    unsafe fn load_lanes<V: I16Lanes>(src: *const u8) -> V {
        // SAFETY: caller guarantees `V::LANES` readable bytes at `src`.
        unsafe { V::load_u8_at(src) }
    }
    #[inline(always)]
    fn store_clamped_slice<V: I16Lanes>(v: V, _zero: V, _max_val: V, dst: &mut [u8]) {
        v.store_u8_sat(dst);
    }
    #[inline(always)]
    unsafe fn store_clamped<V: I16Lanes>(v: V, _zero: V, _max_val: V, dst: *mut u8) {
        // SAFETY: caller guarantees `V::LANES` writable bytes at `dst`.
        unsafe { v.store_u8_sat_at(dst) };
    }
    #[inline(always)]
    fn load_deinterleaved<V: I16FilterLanes>(src: &[u8]) -> (V, V) {
        V::load_u8_deinterleaved(src)
    }
    #[inline(always)]
    unsafe fn load_2rows<V: I16FilterLanes>(p0: *const u8, p1: *const u8, x: usize) -> V {
        let half = V::LANES / 2;
        // SAFETY: caller validated `p0.add(x)` and `p1.add(x)` for `half` bytes.
        unsafe {
            V::load_u8_halves(
                core::slice::from_raw_parts(p0.add(x), half),
                core::slice::from_raw_parts(p1.add(x), half),
            )
        }
    }
    #[inline(always)]
    unsafe fn load_4rows<V: I16FilterLanes>(p: [*const u8; 4], x: usize) -> V {
        let q = V::LANES / 4;
        // SAFETY: caller validated each `p[k].add(x)` for `q` bytes.
        unsafe {
            V::load_u8_quarters([
                core::slice::from_raw_parts(p[0].add(x), q),
                core::slice::from_raw_parts(p[1].add(x), q),
                core::slice::from_raw_parts(p[2].add(x), q),
                core::slice::from_raw_parts(p[3].add(x), q),
            ])
        }
    }
    #[inline(always)]
    fn load_w2x4<V: I16Lanes>(rows: [&[u8]; 4], k: usize) -> V {
        let lanes = [
            i16::from(rows[0][k]),
            i16::from(rows[0][k + 1]),
            i16::from(rows[1][k]),
            i16::from(rows[1][k + 1]),
            i16::from(rows[2][k]),
            i16::from(rows[2][k + 1]),
            i16::from(rows[3][k]),
            i16::from(rows[3][k + 1]),
        ];
        V::load(&lanes)
    }
}

#[allow(unsafe_code, clippy::inline_always)]
impl SimdSample for u16 {
    #[inline(always)]
    fn load_slice<V: I16Lanes>(src: &[u16]) -> V {
        V::load_u16(src)
    }
    #[inline(always)]
    unsafe fn load_lanes<V: I16Lanes>(src: *const u16) -> V {
        // SAFETY: caller guarantees `V::LANES` readable `u16`s at `src`.
        unsafe { V::load_u16_at(src) }
    }
    #[inline(always)]
    fn store_clamped_slice<V: I16Lanes>(v: V, zero: V, max_val: V, dst: &mut [u16]) {
        v.max(zero).min(max_val).store_u16(dst);
    }
    #[inline(always)]
    unsafe fn store_clamped<V: I16Lanes>(v: V, zero: V, max_val: V, dst: *mut u16) {
        // SAFETY: caller guarantees `V::LANES` writable `u16`s at `dst`.
        unsafe { v.max(zero).min(max_val).store_u16_at(dst) };
    }
    #[inline(always)]
    fn load_deinterleaved<V: I16FilterLanes>(src: &[u16]) -> (V, V) {
        V::load_u16_deinterleaved(src)
    }
    #[inline(always)]
    unsafe fn load_2rows<V: I16FilterLanes>(p0: *const u16, p1: *const u16, x: usize) -> V {
        let half = V::LANES / 2;
        // SAFETY: caller validated `p0.add(x)` and `p1.add(x)` for `half` aligned `u16`s.
        unsafe {
            V::load_halves(
                core::slice::from_raw_parts(p0.add(x).cast::<i16>(), half),
                core::slice::from_raw_parts(p1.add(x).cast::<i16>(), half),
            )
        }
    }
    #[inline(always)]
    unsafe fn load_4rows<V: I16FilterLanes>(p: [*const u16; 4], x: usize) -> V {
        let q = V::LANES / 4;
        // SAFETY: caller validated each `p[k].add(x)` for `q` aligned `u16`s.
        unsafe {
            V::load_quarters([
                core::slice::from_raw_parts(p[0].add(x).cast::<i16>(), q),
                core::slice::from_raw_parts(p[1].add(x).cast::<i16>(), q),
                core::slice::from_raw_parts(p[2].add(x).cast::<i16>(), q),
                core::slice::from_raw_parts(p[3].add(x).cast::<i16>(), q),
            ])
        }
    }
    #[inline(always)]
    fn load_w2x4<V: I16Lanes>(rows: [&[u16]; 4], k: usize) -> V {
        let lanes = [
            rows[0][k] as i16,
            rows[0][k + 1] as i16,
            rows[1][k] as i16,
            rows[1][k + 1] as i16,
            rows[2][k] as i16,
            rows[2][k + 1] as i16,
            rows[3][k] as i16,
            rows[3][k + 1] as i16,
        ];
        V::load(&lanes)
    }
}
