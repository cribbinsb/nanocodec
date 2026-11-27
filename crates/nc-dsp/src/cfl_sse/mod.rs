//! Squared error and alpha search of an AV1 chroma-from-luma prediction
//! against the source: the encoder's `CfL` slope search (`nc-av1` `cfl.rs`)
//! prices up to 33 slopes (`0, +1, -1, ..., +16, -16`) per chroma block and
//! plane.
//!
//! The prediction at each sample is `clamp(dc + scaled, 0, 255)`, where
//! `scaled = sign(p) * ((|p| + 32) >> 6)` and `p = ac * alpha` (AV1 §7.11.5
//! `predict_chroma_from_luma`, `Round2Signed(alpha * L, 6)`). With
//! `|ac| <= 8 * 255` (the subsampled luma AC, twice the 2x2 sum less its
//! mean) and `|alpha| <= 16`, `|p| <= 32640 < i16::MAX` fits an i16 lane
//! without overflow (including `-p` for the negative slope `-alpha`), and the
//! symmetric rounding is `(p + 32 + (p >> 15)) >> 6` (for `p < 0`,
//! `-floor((-p + 32) / 64) = floor((p + 31) / 64)`), so the SIMD kernels
//! use the shared 16-bit lanes ([`crate::simd::I16Lanes`]) a row at a time,
//! summing squares in their 32-bit accumulators.
//!
//! Supported chroma shapes are every square and 2:1 rectangle with sides in
//! `{4, 8, 16}` ([`SHAPES`]), so rectangular AV1 key partitions run directly
//! without tiling into squares. [`BestAlphaFn`] evaluates four `±alpha` pairs
//! (eight slopes) per pass over the block, sharing the `src`, `ac` and
//! `p = ac * alpha` values (`-p` needs no multiply) while preserving the
//! exact `0, +1, -1, ..., +16, -16` strict-`<` tie-breaking order.

/// Largest luma AC magnitude the lanes take: twice a 2x2 sum of 8-bit
/// samples (the AC is that less the block mean, so within it).
pub const MAX_AC: i16 = 8 * 255;
/// Largest slope magnitude (`CflAlphaU` / `CflAlphaV`, AV1 §5.11.45).
pub const MAX_ALPHA: i8 = 16;
/// Block sides the kernels take.
pub const SIDES: [usize; 3] = [4, 8, 16];
/// All supported `(width, height)` chroma shapes (squares and 2:1 rectangles).
pub const SHAPES: [(usize, usize); 7] =
    [(4, 4), (8, 8), (16, 16), (4, 8), (8, 4), (8, 16), (16, 8)];

use crate::simd::I16Lanes;

/// Rounding of the scaled AC (AV1 `Round2Signed(.., 6)`).
const SCALE_SHIFT: u32 = 6;
const SCALE_HALF: i32 = 1 << (SCALE_SHIFT - 1);

/// A `CfL` SSE (the scalar reference's contract): the `width` x `height`
/// source block at the start of `src` (rows `stride` apart) against the
/// prediction from `ac` (`width * height`, row-major), `dc` and `alpha`.
/// `unsafe`: it may run its ISA's instructions, so only a table built for a
/// supported ISA calls it.
pub(crate) type SseFn = unsafe fn(
    src: &[u8],
    stride: usize,
    ac: &[i16],
    width: usize,
    height: usize,
    dc: u8,
    alpha: i8,
) -> u32;

/// Fused `CfL` slope search returning `(dc_sse, best_sse, best_alpha)` over
/// `0, +1, -1, ..., +MAX_ALPHA, -MAX_ALPHA` with strict `<` tie-breaking.
pub(crate) type BestAlphaFn = unsafe fn(
    src: &[u8],
    stride: usize,
    ac: &[i16],
    width: usize,
    height: usize,
    dc: u8,
) -> (u32, u32, i8);

/// One ISA's `CfL` SSE and fused alpha-search kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CflSseFns {
    pub sse: SseFn,
    pub best_alpha: BestAlphaFn,
}

/// The `CfL` kernels of `isa` (`Dsp::cfl_sse`, `Dsp::cfl_best_alpha`).
pub(crate) fn kernels(isa: crate::Isa) -> CflSseFns {
    match isa {
        #[cfg(target_arch = "x86_64")]
        crate::Isa::Avx512 => CflSseFns {
            sse: avx512_sse,
            best_alpha: avx512_best_alpha,
        },
        #[cfg(target_arch = "x86_64")]
        crate::Isa::Avx2 => CflSseFns {
            sse: avx2_sse,
            best_alpha: avx2_best_alpha,
        },
        #[cfg(target_arch = "aarch64")]
        crate::Isa::Neon => CflSseFns {
            sse: neon_sse,
            best_alpha: neon_best_alpha,
        },
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        crate::Isa::Simd128 => CflSseFns {
            sse: simd128_sse,
            best_alpha: simd128_best_alpha,
        },
        _ => CflSseFns {
            sse: scalar,
            best_alpha: scalar_best_alpha,
        },
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

/// The scalar reference for a single slope.
pub(crate) fn scalar(
    src: &[u8],
    stride: usize,
    ac: &[i16],
    width: usize,
    height: usize,
    dc: u8,
    alpha: i8,
) -> u32 {
    src.chunks(stride)
        .zip(ac.chunks_exact(width))
        .take(height)
        .map(|(row, ac)| {
            row[..width]
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

/// The scalar reference for the full slope search.
pub(crate) fn scalar_best_alpha(
    src: &[u8],
    stride: usize,
    ac: &[i16],
    width: usize,
    height: usize,
    dc: u8,
) -> (u32, u32, i8) {
    let dc_sse = scalar(src, stride, ac, width, height, dc, 0);
    let mut best_sse = dc_sse;
    let mut best_alpha = 0_i8;
    for alpha in (1..=MAX_ALPHA).flat_map(|a| [a, -a]) {
        if best_sse == 0 {
            break;
        }
        let sse = scalar(src, stride, ac, width, height, dc, alpha);
        if sse < best_sse {
            best_sse = sse;
            best_alpha = alpha;
        }
    }
    (dc_sse, best_sse, best_alpha)
}

/// The kernels' arithmetic on one vector of `V::LANES` samples for a
/// precomputed `product = ac * alpha`.
#[allow(clippy::inline_always, reason = "per vector in the kernels")]
#[inline(always)]
fn add_product<V: I16Lanes>(acc: V::Acc, src: V, product: V, v_dc: V, v_round: V) -> V::Acc {
    // (p + 32 + (p >> 15)) >> 6: Round2Signed (module docs).
    let scaled = product.add(v_round).add(product.sra(15)).sra(SCALE_SHIFT);
    let predicted = v_dc.add(scaled).clamp_u8();
    V::acc_add_sq(acc, src.sub(predicted))
}

/// The SSE of a `width` x `height` block whose rows are a multiple of
/// `V::LANES` wide.
#[allow(clippy::inline_always, reason = "instantiated per ISA and width")]
#[inline(always)]
fn rows<V: I16Lanes>(
    src: &[u8],
    stride: usize,
    ac: &[i16],
    width: usize,
    height: usize,
    dc: u8,
    alpha: i8,
) -> u32 {
    let v_dc = V::splat(i16::from(dc));
    let v_alpha = V::splat(i16::from(alpha));
    let v_round = V::splat(SCALE_HALF as i16);
    let mut acc = V::acc_zero();
    for (row, ac_row) in src.chunks(stride).zip(ac.chunks_exact(width)).take(height) {
        for (samples, ac_chunk) in row[..width]
            .chunks_exact(V::LANES)
            .zip(ac_row.chunks_exact(V::LANES))
        {
            let product = V::load(ac_chunk).mul(v_alpha);
            acc = add_product::<V>(acc, V::load_u8(samples), product, v_dc, v_round);
        }
    }
    V::acc_sum(acc) as u32
}

/// Fused slope search over a `width` x `height` block: computes `dc_sse`,
/// then evaluates four `±alpha` pairs (8 slopes) per pass so `src` and `ac`
/// are loaded once per 8 slopes and `-p = 0 - p` avoids a second multiply.
#[allow(clippy::inline_always, reason = "instantiated per ISA and width")]
#[inline(always)]
fn best_alpha_rows<V: I16Lanes>(
    src: &[u8],
    stride: usize,
    ac: &[i16],
    width: usize,
    height: usize,
    dc: u8,
) -> (u32, u32, i8) {
    let v_dc = V::splat(i16::from(dc));
    let mut dc_acc = V::acc_zero();
    for row in src.chunks(stride).take(height) {
        for samples in row[..width].chunks_exact(V::LANES) {
            dc_acc = V::acc_add_sq(dc_acc, V::load_u8(samples).sub(v_dc));
        }
    }
    let dc_sse = V::acc_sum(dc_acc) as u32;
    if dc_sse == 0 {
        return (0, 0, 0);
    }

    let v_round = V::splat(SCALE_HALF as i16);
    let v_zero = V::splat(0);
    let mut best_sse = dc_sse;
    let mut best_alpha = 0_i8;

    for base in [1_i8, 5, 9, 13] {
        let alphas = [base, base + 1, base + 2, base + 3];
        let v_a0 = V::splat(i16::from(alphas[0]));
        let v_a1 = V::splat(i16::from(alphas[1]));
        let v_a2 = V::splat(i16::from(alphas[2]));
        let v_a3 = V::splat(i16::from(alphas[3]));
        let mut p0 = V::acc_zero();
        let mut n0 = V::acc_zero();
        let mut p1 = V::acc_zero();
        let mut n1 = V::acc_zero();
        let mut p2 = V::acc_zero();
        let mut n2 = V::acc_zero();
        let mut p3 = V::acc_zero();
        let mut n3 = V::acc_zero();

        for (row, ac_row) in src.chunks(stride).zip(ac.chunks_exact(width)).take(height) {
            for (samples, ac_chunk) in row[..width]
                .chunks_exact(V::LANES)
                .zip(ac_row.chunks_exact(V::LANES))
            {
                let s = V::load_u8(samples);
                let a = V::load(ac_chunk);

                let prod0 = a.mul(v_a0);
                p0 = add_product::<V>(p0, s, prod0, v_dc, v_round);
                n0 = add_product::<V>(n0, s, v_zero.sub(prod0), v_dc, v_round);

                let prod1 = a.mul(v_a1);
                p1 = add_product::<V>(p1, s, prod1, v_dc, v_round);
                n1 = add_product::<V>(n1, s, v_zero.sub(prod1), v_dc, v_round);

                let prod2 = a.mul(v_a2);
                p2 = add_product::<V>(p2, s, prod2, v_dc, v_round);
                n2 = add_product::<V>(n2, s, v_zero.sub(prod2), v_dc, v_round);

                let prod3 = a.mul(v_a3);
                p3 = add_product::<V>(p3, s, prod3, v_dc, v_round);
                n3 = add_product::<V>(n3, s, v_zero.sub(prod3), v_dc, v_round);
            }
        }

        let pairs = [
            (alphas[0], V::acc_sum(p0) as u32, V::acc_sum(n0) as u32),
            (alphas[1], V::acc_sum(p1) as u32, V::acc_sum(n1) as u32),
            (alphas[2], V::acc_sum(p2) as u32, V::acc_sum(n2) as u32),
            (alphas[3], V::acc_sum(p3) as u32, V::acc_sum(n3) as u32),
        ];
        for (a, pos_sse, neg_sse) in pairs {
            if pos_sse < best_sse {
                best_sse = pos_sse;
                best_alpha = a;
            }
            if neg_sse < best_sse {
                best_sse = neg_sse;
                best_alpha = -a;
            }
        }
        if best_sse == 0 {
            break;
        }
    }

    (dc_sse, best_sse, best_alpha)
}

/// The kernels of one ISA: `rows` and `best_alpha_rows` with the vector type
/// that fits the block width.
macro_rules! isa_kernels {
    ($(#[$attr:meta])* $sse_name:ident, $best_name:ident, $v4:ty, $v8:ty, $v16:ty) => {
        $(#[$attr])*
        pub(crate) unsafe fn $sse_name(
            src: &[u8],
            stride: usize,
            ac: &[i16],
            width: usize,
            height: usize,
            dc: u8,
            alpha: i8,
        ) -> u32 {
            debug_assert!(SHAPES.contains(&(width, height)), "CfL shape {width}x{height}");
            match width {
                4 => rows::<$v4>(src, stride, ac, width, height, dc, alpha),
                8 => rows::<$v8>(src, stride, ac, width, height, dc, alpha),
                _ => rows::<$v16>(src, stride, ac, width, height, dc, alpha),
            }
        }

        $(#[$attr])*
        pub(crate) unsafe fn $best_name(
            src: &[u8],
            stride: usize,
            ac: &[i16],
            width: usize,
            height: usize,
            dc: u8,
        ) -> (u32, u32, i8) {
            debug_assert!(SHAPES.contains(&(width, height)), "CfL shape {width}x{height}");
            match width {
                4 => best_alpha_rows::<$v4>(src, stride, ac, width, height, dc),
                8 => best_alpha_rows::<$v8>(src, stride, ac, width, height, dc),
                _ => best_alpha_rows::<$v16>(src, stride, ac, width, height, dc),
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
isa_kernels!(
    /// [`scalar`] / [`scalar_best_alpha`] on AVX2.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    avx2_sse,
    avx2_best_alpha,
    crate::simd::avx2::I16x4,
    crate::simd::avx2::I16x8,
    crate::simd::avx2::I16x16
);

#[cfg(target_arch = "x86_64")]
isa_kernels!(
    /// [`scalar`] / [`scalar_best_alpha`] on AVX-512 (AVX-512VL gives 32 YMM
    /// registers so all eight accumulators stay in registers with zero spills).
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512F/BW/DQ/VL and AVX2.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2")]
    avx512_sse,
    avx512_best_alpha,
    crate::simd::avx2::I16x4,
    crate::simd::avx2::I16x8,
    crate::simd::avx2::I16x16
);

#[cfg(target_arch = "aarch64")]
isa_kernels!(
    /// [`scalar`] / [`scalar_best_alpha`] on NEON.
    ///
    /// # Safety
    ///
    /// None beyond the table's contract: NEON is always present on `AArch64`.
    #[target_feature(enable = "neon")]
    neon_sse,
    neon_best_alpha,
    crate::simd::neon::I16x4,
    crate::simd::neon::I16x8,
    crate::simd::neon::I16x8
);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
isa_kernels!(
    /// [`scalar`] / [`scalar_best_alpha`] on WebAssembly `simd128`.
    ///
    /// # Safety
    ///
    /// None beyond the table's contract: `simd128` is enabled at build time.
    #[target_feature(enable = "simd128")]
    simd128_sse,
    simd128_best_alpha,
    crate::simd::simd128::I16x4,
    crate::simd::simd128::I16x8,
    crate::simd::simd128::I16x8
);
