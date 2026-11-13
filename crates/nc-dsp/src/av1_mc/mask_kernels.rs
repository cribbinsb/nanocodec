//! The masked compound blend of one row (AV1 §7.11.3.14, wedge and
//! difference-weighted compound) as a kernel per [`crate::Dsp`]: a scalar
//! reference and a SIMD version over [`I16x8Shuffles`] and [`I16Pairs`]
//! (AVX2's 128-bit lanes, NEON, SIMD128), after dav1d 1.5.4 `mask_c`
//! (`src/mc_tmpl.c`, BSD-2-Clause, see `THIRD_PARTY.md`): the two
//! compound-precision predictions interleaved with the weight pairs
//! `(m, 64 − m)`, one `pmaddwd` per four outputs, rounded and saturated to
//! 8 bits.
//!
//! The products stay exact: compound samples are within 16 bits and the
//! weights sum to 64, so each pair sums to at most `64 · 2^15`.

#![allow(unsafe_code, clippy::inline_always)]

use super::blend::{AV1_MASK_BITS, AV1_MASK_MAX};
use super::{INTER_POST_ROUND, clip_pixel, round2};
use crate::Isa;
use crate::simd::{I16Pairs, I16x8Shuffles, I32Lanes};

/// Rounding of the blended sum.
const BLEND_SHIFT: u32 = AV1_MASK_BITS + INTER_POST_ROUND;
/// Outputs per SIMD step.
const LANES: usize = 8;

/// `dst[j] = Clip1(Round2(m · p0 + (64 − m) · p1, 6 + InterPostRound))`
/// for every `j < dst.len()`, `m = weights[j]` (0 to 64).
pub(crate) type MaskBlendFn = unsafe fn(p0: &[i16], p1: &[i16], weights: &[i16], dst: &mut [u8]);

/// The kernel of `isa` (AVX-512 tables inherit AVX2's).
pub(crate) fn kernel(isa: Isa) -> MaskBlendFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => blend_row_avx2,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => blend_row_neon,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => blend_row_simd128,
        _ => blend_row_scalar,
    }
}

/// Scalar reference.
pub(crate) fn blend_row_scalar(p0: &[i16], p1: &[i16], weights: &[i16], dst: &mut [u8]) {
    let max = i32::from(AV1_MASK_MAX);
    for (j, out) in dst.iter_mut().enumerate() {
        let m = i32::from(weights[j]);
        let sum = m * i32::from(p0[j]) + (max - m) * i32::from(p1[j]);
        *out = clip_pixel(round2(sum, BLEND_SHIFT));
    }
}

/// The SIMD kernel over 8-lane `V`: whole vectors, then the scalar tail.
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
unsafe fn blend_row_lanes<V>(p0: &[i16], p1: &[i16], weights: &[i16], dst: &mut [u8])
where
    V: I16x8Shuffles + I16Pairs<Pairs = <V as I16x8Shuffles>::Wide>,
{
    let n = dst.len();
    let whole = n - n % LANES;
    let max = V::splat(AV1_MASK_MAX.into());
    for j in (0..whole).step_by(LANES) {
        let (a, b) = (V::load(&p0[j..]), V::load(&p1[j..]));
        let m = V::load(&weights[j..]);
        let rest = max.sub(m);
        let low = a.zip_lo(b).madd(m.zip_lo(rest)).round_shift(BLEND_SHIFT);
        let high = a.zip_hi(b).madd(m.zip_hi(rest)).round_shift(BLEND_SHIFT);
        V::narrow_pairs(low, high).store_u8_sat(&mut dst[j..j + LANES]);
    }
    blend_row_scalar(
        &p0[whole..],
        &p1[whole..],
        &weights[whole..],
        &mut dst[whole..],
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn blend_row_avx2(p0: &[i16], p1: &[i16], weights: &[i16], dst: &mut [u8]) {
    // SAFETY: AVX2 is enabled by this wrapper.
    unsafe { blend_row_lanes::<crate::simd::avx2::I16x8>(p0, p1, weights, dst) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn blend_row_neon(p0: &[i16], p1: &[i16], weights: &[i16], dst: &mut [u8]) {
    // SAFETY: NEON is enabled by this wrapper.
    unsafe { blend_row_lanes::<crate::simd::neon::I16x8>(p0, p1, weights, dst) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn blend_row_simd128(p0: &[i16], p1: &[i16], weights: &[i16], dst: &mut [u8]) {
    // SAFETY: SIMD128 is enabled for the whole module in this build.
    unsafe { blend_row_lanes::<crate::simd::simd128::I16x8>(p0, p1, weights, dst) }
}
