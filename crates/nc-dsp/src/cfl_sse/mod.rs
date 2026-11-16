//! Squared error of an AV1 chroma-from-luma prediction against the source:
//! the encoder's `CfL` slope search (`nc-av1` `cfl.rs`) prices up to 33
//! slopes per chroma block and plane with it.
//!
//! The prediction at each sample is `clamp(dc + scaled, 0, 255)`, where
//! `scaled = sign(p) * ((|p| + 32) >> 6)` and `p = ac * alpha` (AV1 §7.11.5
//! `predict_chroma_from_luma`, `Round2Signed(alpha * L, 6)`). With
//! `|ac| <= 8 * 255` (the subsampled luma AC, twice the 2x2 sum less its
//! mean) and `|alpha| <= 16`, `p` fits an i16 lane, and the symmetric
//! rounding is the floor `(p + 32 + (p >> 15)) >> 6` (for `p < 0`,
//! `-floor((-p + 32) / 64) = floor((p + 31) / 64)`), so the SIMD kernels
//! are the shared 16-bit lanes ([`crate::simd::I16Lanes`]) a row at a time,
//! summing squares in their 32-bit accumulators.
//!
//! Blocks are square with sides 4, 8 or 16.

/// Largest luma AC magnitude the lanes take: twice a 2x2 sum of 8-bit
/// samples (the AC is that less the block mean, so within it).
pub const MAX_AC: i16 = 8 * 255;
/// Largest slope magnitude (`CflAlphaU` / `CflAlphaV`, AV1 §5.11.45).
pub const MAX_ALPHA: i8 = 16;
/// Block sides the kernels take.
pub const SIDES: [usize; 3] = [4, 8, 16];

use crate::simd::I16Lanes;

/// Rounding of the scaled AC (AV1 `Round2Signed(.., 6)`).
const SCALE_SHIFT: u32 = 6;
const SCALE_HALF: i32 = 1 << (SCALE_SHIFT - 1);

/// A `CfL` SSE (the scalar reference's contract): the `side` x `side` source
/// block at the start of `src` (rows `stride` apart) against the prediction
/// from `ac` (`side * side`, row-major), `dc` and `alpha`. `unsafe`: it may
/// run its ISA's instructions, so only a table built for a supported ISA
/// calls it.
pub(crate) type SseFn =
    unsafe fn(src: &[u8], stride: usize, ac: &[i16], side: usize, dc: u8, alpha: i8) -> u32;

/// The `CfL` SSE of `isa` (`Dsp::cfl_sse`).
pub(crate) fn kernel(isa: crate::Isa) -> SseFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        crate::Isa::Avx2 | crate::Isa::Avx512 => avx2,
        #[cfg(target_arch = "aarch64")]
        crate::Isa::Neon => neon,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        crate::Isa::Simd128 => simd128,
        _ => scalar,
    }
}

/// One predicted sample (AV1 §7.11.5).
#[must_use]
pub fn predict_sample(dc: u8, ac: i16, alpha: i8) -> u8 {
    let product = i32::from(ac) * i32::from(alpha);
    let magnitude = (product.abs() + SCALE_HALF) >> SCALE_SHIFT;
    let scaled = if product < 0 { -magnitude } else { magnitude };
    (i32::from(dc) + scaled).clamp(0, 255) as u8
}

/// The scalar reference.
pub(crate) fn scalar(src: &[u8], stride: usize, ac: &[i16], side: usize, dc: u8, alpha: i8) -> u32 {
    src.chunks(stride)
        .zip(ac.chunks_exact(side))
        .take(side)
        .map(|(row, ac)| {
            row[..side]
                .iter()
                .zip(ac)
                .map(|(&s, &a)| {
                    let error = i32::from(s) - i32::from(predict_sample(dc, a, alpha));
                    (error * error) as u32
                })
                .sum::<u32>()
        })
        .sum()
}

/// The kernels' arithmetic on one vector of `V::LANES` samples: the
/// source `src` against the prediction from `ac`.
#[allow(clippy::inline_always, reason = "per vector in the kernels")]
#[inline(always)]
fn add_vector<V: I16Lanes>(
    acc: V::Acc,
    src: V,
    ac: V,
    (dc, alpha, round): (i16, i16, i16),
) -> V::Acc {
    let product = ac.mul(V::splat(alpha));
    // (p + 32 + (p >> 15)) >> 6: Round2Signed (module docs).
    let scaled = product
        .add(V::splat(round))
        .add(product.sra(15))
        .sra(SCALE_SHIFT);
    let predicted = V::splat(dc).add(scaled).clamp_u8();
    V::acc_add_sq(acc, src.sub(predicted))
}

/// The SSE of a block whose rows are `V::LANES` wide, or a multiple of it.
#[allow(clippy::inline_always, reason = "instantiated per ISA and width")]
#[inline(always)]
fn rows<V: I16Lanes>(src: &[u8], stride: usize, ac: &[i16], side: usize, dc: u8, alpha: i8) -> u32 {
    let constants = (i16::from(dc), i16::from(alpha), SCALE_HALF as i16);
    let mut acc = V::acc_zero();
    for (row, ac) in src.chunks(stride).zip(ac.chunks_exact(side)).take(side) {
        for (samples, ac) in row[..side]
            .chunks_exact(V::LANES)
            .zip(ac.chunks_exact(V::LANES))
        {
            acc = add_vector::<V>(acc, V::load_u8(samples), V::load(ac), constants);
        }
    }
    V::acc_sum(acc) as u32
}

/// The kernels of one ISA: `rows` with the vector type that fits the side.
macro_rules! isa_kernel {
    ($(#[$attr:meta])* $name:ident, $v4:ty, $v8:ty, $v16:ty) => {
        $(#[$attr])*
        pub(crate) unsafe fn $name(
            src: &[u8],
            stride: usize,
            ac: &[i16],
            side: usize,
            dc: u8,
            alpha: i8,
        ) -> u32 {
            assert!(SIDES.contains(&side), "CfL side {side}");
            match side {
                4 => rows::<$v4>(src, stride, ac, side, dc, alpha),
                8 => rows::<$v8>(src, stride, ac, side, dc, alpha),
                _ => rows::<$v16>(src, stride, ac, side, dc, alpha),
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
isa_kernel!(
    /// [`scalar`] on AVX2 (and AVX-512 CPUs: 16 lanes cover the widest row).
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    avx2,
    crate::simd::avx2::I16x4,
    crate::simd::avx2::I16x8,
    crate::simd::avx2::I16x16
);
#[cfg(target_arch = "aarch64")]
isa_kernel!(
    /// [`scalar`] on NEON.
    ///
    /// # Safety
    ///
    /// None beyond the table's contract: NEON is always present on `AArch64`.
    neon,
    crate::simd::neon::I16x4,
    crate::simd::neon::I16x8,
    crate::simd::neon::I16x8
);
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
isa_kernel!(
    /// [`scalar`] on SIMD128.
    ///
    /// # Safety
    ///
    /// None beyond the table's contract: SIMD128 is enabled at build time.
    simd128,
    crate::simd::simd128::I16x4,
    crate::simd::simd128::I16x8,
    crate::simd::simd128::I16x8
);
