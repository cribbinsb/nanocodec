//! WebAssembly SIMD128 loop restoration kernels, the NEON kernels'
//! arithmetic (dav1d 1.5.4 `src/arm/64/looprestoration.S`, BSD-2-Clause,
//! see `THIRD_PARTY.md`) in SIMD128 terms: the horizontal Wiener pass in
//! wrapping 16-bit lanes with a saturating add for the clamp (exact: see
//! `avx2.rs`), the shifted rows taken with byte shuffles; the vertical
//! pass with `i32x4.dot_i16x8_s` on interleaved pair sums, as `pmaddwd`.
//! The self-guided rows are [`super::sgr_lanes`] on two `v128`.

#![allow(
    clippy::many_single_char_names,
    reason = "the spec's and dav1d's sample, tap and statistic names"
)]
#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use core::arch::wasm32::{
    f32x4_convert_i32x4, f32x4_div, i8x16_shuffle, i16x8_add, i16x8_add_sat, i16x8_mul,
    i16x8_narrow_i32x4, i16x8_shl, i16x8_shr, i16x8_shuffle, i16x8_splat, i32x4_add,
    i32x4_dot_i16x8, i32x4_eq, i32x4_max, i32x4_min, i32x4_mul, i32x4_shl, i32x4_shr, i32x4_splat,
    i32x4_sub, i32x4_trunc_sat_f32x4, i64x2_extract_lane, u8x16_narrow_i16x8,
    u16x8_extend_high_u8x16, u16x8_extend_low_u8x16, u32x4_extend_high_u16x8,
    u32x4_extend_low_u16x8, u32x4_shr, u64x2_splat, v128, v128_load, v128_store,
};

use super::kernels::{Kernels, WIENER_ROWS};
use super::sgr_lanes::{SgrLanes, sgr_kernels};
use super::wiener::{ROUND1, SATURATION_BIAS, SATURATION_OFFSET, WienerTaps};
use crate::simd::packed_pair;

/// This ISA's table.
pub(super) const KERNELS: Kernels = Kernels {
    wiener_stripe: None,
    wiener_h,
    wiener_v,
    box_sums3,
    box_sums5,
    box_ab3,
    box_ab5,
    between,
    on,
    around,
    blend,
};

/// Outputs per Wiener step.
const STEP: usize = 8;

/// The eight 16-bit lanes from lane `K` on of `low` then `high`.
macro_rules! ext {
    ($low:expr, $high:expr, $k:literal) => {
        i8x16_shuffle::<
            { 2 * $k },
            { 2 * $k + 1 },
            { 2 * $k + 2 },
            { 2 * $k + 3 },
            { 2 * $k + 4 },
            { 2 * $k + 5 },
            { 2 * $k + 6 },
            { 2 * $k + 7 },
            { 2 * $k + 8 },
            { 2 * $k + 9 },
            { 2 * $k + 10 },
            { 2 * $k + 11 },
            { 2 * $k + 12 },
            { 2 * $k + 13 },
            { 2 * $k + 14 },
            { 2 * $k + 15 },
        >($low, $high)
    };
}

#[target_feature(enable = "simd128")]
fn wiener_h(src: &[u8], taps: WienerTaps, out: &mut [i16], n: usize) {
    let blocks = n.div_ceil(STEP);
    // Every load and store below stays within these bounds.
    assert!(src.len() >= blocks * STEP + STEP && out.len() >= blocks * STEP);
    let [c0, c1, c2, c3] = taps.map(|t| i16x8_splat(t));
    let offset = i16x8_splat(SATURATION_OFFSET);
    let bias = i16x8_splat(SATURATION_BIAS);
    for block in 0..blocks {
        let j = block * STEP;
        // SAFETY: bytes j .. j + 16 lie in `src` (checked above).
        let x = unsafe { v128_load(src.as_ptr().add(j).cast()) };
        let (low, high) = (u16x8_extend_low_u8x16(x), u16x8_extend_high_u8x16(x));
        let p3 = ext!(low, high, 3);
        let a = i16x8_add(low, ext!(low, high, 6));
        let b = i16x8_add(ext!(low, high, 1), ext!(low, high, 5));
        let c = i16x8_add(ext!(low, high, 2), ext!(low, high, 4));
        let sum = i16x8_add(
            i16x8_add(i16x8_mul(p3, c3), i16x8_mul(a, c0)),
            i16x8_add(i16x8_mul(b, c1), i16x8_mul(c, c2)),
        );
        let saturated = i16x8_add_sat(sum, i16x8_add(i16x8_shl(p3, 7), offset));
        let result = i16x8_add(i16x8_shr(saturated, 3), bias);
        // SAFETY: values j .. j + 8 lie in `out` (checked above).
        unsafe { v128_store(out.as_mut_ptr().add(j).cast(), result) };
    }
}

/// Outputs per vertical Wiener step: two 8-lane halves.
const V_STEP: usize = 2 * STEP;

/// Eight vertical Wiener outputs at `rows[k] + j`, saturated to 16 bits
/// (`Σ >> 11`).
///
/// # Safety
///
/// Every `rows[k]` must be readable for 8 values at `j`.
#[inline(always)]
unsafe fn vertical8(
    rows: &[*const i16; WIENER_ROWS],
    j: usize,
    (outer, inner): (v128, v128),
) -> v128 {
    // SAFETY: the caller's contract.
    let row = |k: usize| unsafe { v128_load(rows[k].add(j).cast()) };
    let a = i16x8_add(row(0), row(6));
    let b = i16x8_add(row(1), row(5));
    let c = i16x8_add(row(2), row(4));
    let d = row(3);
    let low = i32x4_add(
        i32x4_dot_i16x8(i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b), outer),
        i32x4_dot_i16x8(i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(c, d), inner),
    );
    let high = i32x4_add(
        i32x4_dot_i16x8(i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b), outer),
        i32x4_dot_i16x8(i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(c, d), inner),
    );
    i16x8_narrow_i32x4(i32x4_shr(low, ROUND1), i32x4_shr(high, ROUND1))
}

#[target_feature(enable = "simd128")]
fn wiener_v(rows: &[&[i16]; WIENER_ROWS], taps: WienerTaps, dst: &mut [u8]) {
    let n = dst.len();
    let blocks = n.div_ceil(V_STEP);
    // Every load below stays within these bounds.
    assert!(rows.iter().all(|row| row.len() >= blocks * V_STEP));
    let pointers = rows.map(<[i16]>::as_ptr);
    let [c0, c1, c2, c3] = taps;
    let taps = (
        i32x4_splat(packed_pair(c0, c1)),
        i32x4_splat(packed_pair(c2, c3)),
    );
    for block in 0..blocks {
        let j = block * V_STEP;
        // SAFETY: values j .. j + 16 of every row lie in it (checked
        // above).
        let bytes = unsafe {
            u8x16_narrow_i16x8(
                vertical8(&pointers, j, taps),
                vertical8(&pointers, j + STEP, taps),
            )
        };
        if j + V_STEP <= n {
            // SAFETY: bytes j .. j + 16 lie in `dst`.
            unsafe { v128_store(dst.as_mut_ptr().add(j).cast(), bytes) };
        } else {
            let mut tail = [0u8; V_STEP];
            // SAFETY: the store covers `tail`.
            unsafe { v128_store(tail.as_mut_ptr().cast(), bytes) };
            dst[j..].copy_from_slice(&tail[..n - j]);
        }
    }
}

/// Eight `i32` lanes for the self-guided kernels.
#[derive(Clone, Copy)]
struct Lanes(v128, v128);

impl Lanes {
    /// `f` on both halves.
    #[inline(always)]
    fn map(self, f: impl Fn(v128) -> v128) -> Self {
        Self(f(self.0), f(self.1))
    }

    /// `f` on both halves of `self` and `other`.
    #[inline(always)]
    fn zip(self, other: Self, f: impl Fn(v128, v128) -> v128) -> Self {
        Self(f(self.0, other.0), f(self.1, other.1))
    }
}

/// Shift count of a constant-shift method (always below 32).
#[inline(always)]
fn count(bits: i32) -> u32 {
    bits as u32
}

impl SgrLanes for Lanes {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        let v = i32x4_splat(value);
        Self(v, v)
    }
    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        let src = &src[..8];
        // SAFETY: both loads cover `src`.
        unsafe {
            Self(
                v128_load(src.as_ptr().cast()),
                v128_load(src[4..].as_ptr().cast()),
            )
        }
    }
    #[inline(always)]
    fn load_u8(src: &[u8]) -> Self {
        let bytes = u64::from_le_bytes(src[..8].try_into().expect("eight bytes"));
        let words = u16x8_extend_low_u8x16(u64x2_splat(bytes));
        Self(
            u32x4_extend_low_u16x8(words),
            u32x4_extend_high_u16x8(words),
        )
    }
    #[inline(always)]
    fn store(self, dst: &mut [i32]) {
        let dst = &mut dst[..8];
        // SAFETY: both stores cover `dst`.
        unsafe {
            v128_store(dst.as_mut_ptr().cast(), self.0);
            v128_store(dst[4..].as_mut_ptr().cast(), self.1);
        }
    }
    #[inline(always)]
    fn store_u8_sat(self, dst: &mut [u8]) {
        let words = i16x8_narrow_i32x4(self.0, self.1);
        let bytes = i64x2_extract_lane::<0>(u8x16_narrow_i16x8(words, words));
        dst[..8].copy_from_slice(&bytes.to_le_bytes());
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self.zip(other, |a, b| i32x4_add(a, b))
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        self.zip(other, |a, b| i32x4_sub(a, b))
    }
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        self.zip(other, |a, b| i32x4_mul(a, b))
    }
    #[inline(always)]
    fn shl<const BITS: i32>(self) -> Self {
        self.map(|a| i32x4_shl(a, count(BITS)))
    }
    #[inline(always)]
    fn shr<const BITS: i32>(self) -> Self {
        self.map(|a| u32x4_shr(a, count(BITS)))
    }
    #[inline(always)]
    fn sra<const BITS: i32>(self) -> Self {
        self.map(|a| i32x4_shr(a, count(BITS)))
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.zip(other, |a, b| i32x4_min(a, b))
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.zip(other, |a, b| i32x4_max(a, b))
    }
    #[inline(always)]
    fn eq(self, other: Self) -> Self {
        self.zip(other, |a, b| i32x4_eq(a, b))
    }
    #[inline(always)]
    fn div_floor(self, den: Self) -> Self {
        self.zip(den, |a, b| {
            i32x4_trunc_sat_f32x4(f32x4_div(f32x4_convert_i32x4(a), f32x4_convert_i32x4(b)))
        })
    }
}

sgr_kernels!(#[target_feature(enable = "simd128")]; Lanes);
