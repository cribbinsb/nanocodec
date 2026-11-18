//! High-bit-depth (`u16`, 10 and 12 bpc) AV1 inverse transform SIMD tables
//! across AVX2, AVX-512, NEON, and SIMD128, built on [`super::generic::inverse_add_lanes`].
//!
//! Attribution: DC-only shortcut, EOB scan-bound row/column pruning, and
//! bit-depth stage clipping follow dav1d 1.5.4 (`src/itx_tmpl.c`,
//! `src/x86/itx16_avx2.asm`, `src/x86/itx16_avx512.asm`, `src/arm/64/itx16.S`,
//! BSD-2-Clause; see `THIRD_PARTY.md`) and libaom's normative stage ranges.

#![allow(unsafe_code, clippy::inline_always)]
#![cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code, unused_imports)
)]

use nc_common::BitDepth;

use super::generic::inverse_add_lanes;
use super::high::inverse_add_kernels;
use super::{Av1TxSize, Kernel1d};
use crate::Isa;
use crate::simd::{I16Lanes, I32Lanes};

pub(crate) type HighInverseAddFn = unsafe fn(
    vertical: Kernel1d,
    horizontal: Kernel1d,
    coeffs: &[i32],
    nonzero_rows: usize,
    dst: &mut [u16],
    dst_stride: usize,
    depth: BitDepth,
);

pub(crate) type HighConstantAddFn = unsafe fn(
    residual: i32,
    width: usize,
    height: usize,
    dst: &mut [u16],
    stride: usize,
    depth: BitDepth,
);

#[derive(Debug, Clone, Copy)]
pub(crate) struct HighAv1InverseFns {
    pub add_10: [HighInverseAddFn; 19],
    pub add_12: [HighInverseAddFn; 19],
    pub dc: HighConstantAddFn,
}

const fn sides<const W: usize, const H: usize>() -> Av1TxSize {
    Av1TxSize {
        width_log2: W.trailing_zeros() as u8,
        height_log2: H.trailing_zeros() as u8,
    }
}

macro_rules! all_sizes_split {
    ($small_fn:ident, $wide_fn:ident $(, $small:ty $(, $wide:ty)?)?) => {
        [
            $small_fn::<$($small, )? 4, 4>,
            $wide_fn::<$($($wide, )?)? 8, 8>,
            $wide_fn::<$($($wide, )?)? 16, 16>,
            $wide_fn::<$($($wide, )?)? 32, 32>,
            $wide_fn::<$($($wide, )?)? 64, 64>,
            $small_fn::<$($small, )? 4, 8>,
            $small_fn::<$($small, )? 8, 4>,
            $wide_fn::<$($($wide, )?)? 8, 16>,
            $wide_fn::<$($($wide, )?)? 16, 8>,
            $wide_fn::<$($($wide, )?)? 16, 32>,
            $wide_fn::<$($($wide, )?)? 32, 16>,
            $wide_fn::<$($($wide, )?)? 32, 64>,
            $wide_fn::<$($($wide, )?)? 64, 32>,
            $small_fn::<$($small, )? 4, 16>,
            $small_fn::<$($small, )? 16, 4>,
            $wide_fn::<$($($wide, )?)? 8, 32>,
            $wide_fn::<$($($wide, )?)? 32, 8>,
            $wide_fn::<$($($wide, )?)? 16, 64>,
            $wide_fn::<$($($wide, )?)? 64, 16>,
        ]
    };
}

macro_rules! all_sizes {
    ($f:ident $(, $small:ty, $wide:ty)?) => {
        all_sizes_split!($f, $f $(, $small, $wide)?)
    };
}

fn scalar_add_10<const W: usize, const H: usize>(
    vertical: Kernel1d,
    horizontal: Kernel1d,
    coeffs: &[i32],
    _nonzero_rows: usize,
    dst: &mut [u16],
    dst_stride: usize,
    depth: BitDepth,
) {
    inverse_add_kernels::<u16, 18, 16, W, H>(
        sides::<W, H>(),
        (vertical, horizontal),
        coeffs,
        (dst, dst_stride),
        depth,
    );
}

fn scalar_add_12<const W: usize, const H: usize>(
    vertical: Kernel1d,
    horizontal: Kernel1d,
    coeffs: &[i32],
    _nonzero_rows: usize,
    dst: &mut [u16],
    dst_stride: usize,
    depth: BitDepth,
) {
    inverse_add_kernels::<u16, 20, 18, W, H>(
        sides::<W, H>(),
        (vertical, horizontal),
        coeffs,
        (dst, dst_stride),
        depth,
    );
}

fn scalar_constant_add(
    residual: i32,
    width: usize,
    height: usize,
    dst: &mut [u16],
    stride: usize,
    depth: BitDepth,
) {
    let max_val = depth.max();
    let residual = residual.clamp(-max_val, max_val);
    for row in dst.chunks_mut(stride).take(height) {
        for sample in &mut row[..width] {
            *sample = (i32::from(*sample) + residual).clamp(0, max_val) as u16;
        }
    }
}

impl HighAv1InverseFns {
    pub const SCALAR: Self = Self {
        add_10: all_sizes!(scalar_add_10),
        add_12: all_sizes!(scalar_add_12),
        dc: scalar_constant_add,
    };

    pub fn for_isa(isa: Isa) -> Self {
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => x86::avx2_fns(),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => x86::avx512_fns(),
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => neon::neon_fns(),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Isa::Simd128 => simd128::simd128_fns(),
            _ => Self::SCALAR,
        }
    }
}

/// Check whether every active 12-bit coefficient has `|c| < 2048`: within
/// that bound every stage of a 64-point butterfly stays `< 2^17`, so 32-bit
/// `I32Lanes` arithmetic with 20-bit row clamps is overflow-free.
#[inline(always)]
fn coeffs_fit_12bit_i32<V: I32Lanes>(
    coeffs: &[i32],
    coded_w: usize,
    coded_h: usize,
    active_h: usize,
) -> bool {
    let mut acc = V::splat(0);
    for c in 0..coded_w {
        for r0 in (0..active_h).step_by(V::LANES) {
            acc = acc.or(V::load(&coeffs[c * coded_h + r0..]).abs());
        }
    }
    acc.shr_logical(11).count_nonzero() == 0
}

#[inline(always)]
unsafe fn simd_add_12<V: I32Lanes, const W: usize, const H: usize>(
    vertical: Kernel1d,
    horizontal: Kernel1d,
    coeffs: &[i32],
    nonzero_rows: usize,
    dst: &mut [u16],
    dst_stride: usize,
    depth: BitDepth,
) {
    let size = sides::<W, H>();
    let (coded_w, coded_h) = (size.coded_width(), size.coded_height());
    let rows_bound = nonzero_rows.clamp(1, coded_h);
    let active_h = (rows_bound.div_ceil(V::LANES) * V::LANES).min(coded_h);
    let coeffs = &coeffs[..coded_w * coded_h];
    if coeffs_fit_12bit_i32::<V>(coeffs, coded_w, coded_h, active_h) {
        // SAFETY: forwarded ISA contract; `|c| < 2048` keeps 12-bit row stages `< 2^17`.
        unsafe {
            inverse_add_lanes::<V, i32, u16, 20, 18, W, H>(
                vertical,
                horizontal,
                coeffs,
                nonzero_rows,
                dst,
                dst_stride,
                depth.max(),
            );
        }
    } else {
        inverse_add_kernels::<u16, 20, 18, W, H>(
            size,
            (vertical, horizontal),
            coeffs,
            (dst, dst_stride),
            depth,
        );
    }
}

#[inline(always)]
unsafe fn constant_add_u16_lanes<V4: I16Lanes, V8: I16Lanes, VWide: I16Lanes>(
    residual: i32,
    width: usize,
    height: usize,
    dst: &mut [u16],
    stride: usize,
    depth: BitDepth,
) {
    assert!(width == 4 || width == 8 || width.is_multiple_of(16));
    assert!(height == 0 || (stride >= width && dst.len() >= (height - 1) * stride + width));
    let max_val = depth.max();
    let r16 = residual.clamp(-max_val, max_val) as i16;
    let hi16 = max_val as i16;
    let (res_w, zero_w, hi_w) = (VWide::splat(r16), VWide::splat(0), VWide::splat(hi16));
    let (res_8, zero_8, hi_8) = (V8::splat(r16), V8::splat(0), V8::splat(hi16));
    let (res_4, zero_4, hi_4) = (V4::splat(r16), V4::splat(0), V4::splat(hi16));
    for y in 0..height {
        let row = &mut dst[y * stride..y * stride + width];
        let mut x = 0usize;
        while x + VWide::LANES <= width {
            // SAFETY: `row[x..]` holds `VWide::LANES` `u16` words in `0..=4095`.
            unsafe {
                let s = core::slice::from_raw_parts_mut(
                    row[x..].as_mut_ptr().cast::<i16>(),
                    VWide::LANES,
                );
                VWide::load(s).add(res_w).max(zero_w).min(hi_w).store(s);
            }
            x += VWide::LANES;
        }
        while V8::LANES < VWide::LANES && x + V8::LANES <= width {
            // SAFETY: `row[x..]` holds `V8::LANES` `u16` words in `0..=4095`.
            unsafe {
                let s =
                    core::slice::from_raw_parts_mut(row[x..].as_mut_ptr().cast::<i16>(), V8::LANES);
                V8::load(s).add(res_8).max(zero_8).min(hi_8).store(s);
            }
            x += V8::LANES;
        }
        if x + V4::LANES <= width {
            // SAFETY: `row[x..]` holds `V4::LANES` `u16` words in `0..=4095`.
            unsafe {
                let s =
                    core::slice::from_raw_parts_mut(row[x..].as_mut_ptr().cast::<i16>(), V4::LANES);
                V4::load(s).add(res_4).max(zero_4).min(hi_4).store(s);
            }
        }
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! isa_high_inverse {
    ($mod_name:ident, $fn_name:ident, $feat:literal, $narrow_mod:path, $small:ty, $wide:ty, $v4:ty, $v8:ty, $vw:ty) => {
        mod $mod_name {
            use super::super::{
                BitDepth, HighAv1InverseFns, Kernel1d, constant_add_u16_lanes, inverse_add_lanes,
                simd_add_12,
            };
            use $narrow_mod as narrow_isa;

            #[target_feature(enable = $feat)]
            pub(super) unsafe fn add_10_small<
                V: crate::simd::I32Lanes,
                const W: usize,
                const H: usize,
            >(
                vertical: Kernel1d,
                horizontal: Kernel1d,
                coeffs: &[i32],
                nonzero_rows: usize,
                dst: &mut [u16],
                dst_stride: usize,
                depth: BitDepth,
            ) {
                // SAFETY: target feature enabled by this wrapper; `V::LANES` divides `W` and `H`.
                unsafe {
                    inverse_add_lanes::<V, i32, u16, 18, 16, W, H>(
                        vertical,
                        horizontal,
                        coeffs,
                        nonzero_rows,
                        dst,
                        dst_stride,
                        depth.max(),
                    );
                }
            }

            #[target_feature(enable = $feat)]
            pub(super) unsafe fn add_10_narrow<const W: usize, const H: usize>(
                vertical: Kernel1d,
                horizontal: Kernel1d,
                coeffs: &[i32],
                nonzero_rows: usize,
                dst: &mut [u16],
                dst_stride: usize,
                depth: BitDepth,
            ) {
                // SAFETY: target feature enabled by this wrapper; `W >= 8 && H >= 8`.
                unsafe {
                    narrow_isa::av1_inverse_add_u16::<i32, W, H>(
                        (vertical, horizontal),
                        coeffs,
                        nonzero_rows,
                        dst,
                        dst_stride,
                        depth.max(),
                    );
                }
            }

            #[target_feature(enable = $feat)]
            pub(super) unsafe fn add_12<
                V: crate::simd::I32Lanes,
                const W: usize,
                const H: usize,
            >(
                vertical: Kernel1d,
                horizontal: Kernel1d,
                coeffs: &[i32],
                nonzero_rows: usize,
                dst: &mut [u16],
                dst_stride: usize,
                depth: BitDepth,
            ) {
                // SAFETY: target feature enabled by this wrapper; `V::LANES` divides `W` and `H`.
                unsafe {
                    simd_add_12::<V, W, H>(
                        vertical,
                        horizontal,
                        coeffs,
                        nonzero_rows,
                        dst,
                        dst_stride,
                        depth,
                    );
                }
            }

            #[target_feature(enable = $feat)]
            pub(super) unsafe fn dc(
                residual: i32,
                width: usize,
                height: usize,
                dst: &mut [u16],
                stride: usize,
                depth: BitDepth,
            ) {
                // SAFETY: target feature enabled by this wrapper.
                unsafe {
                    constant_add_u16_lanes::<$v4, $v8, $vw>(
                        residual, width, height, dst, stride, depth,
                    );
                }
            }

            pub(crate) fn $fn_name() -> HighAv1InverseFns {
                HighAv1InverseFns {
                    add_10: all_sizes_split!(add_10_small, add_10_narrow, $small),
                    add_12: all_sizes!(add_12, $small, $wide),
                    dc,
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{
        BitDepth, HighAv1InverseFns, Kernel1d, constant_add_u16_lanes, inverse_add_lanes,
        simd_add_12,
    };
    use crate::simd::avx512::I32x16;
    use crate::txfm::avx2_inverse;

    isa_high_inverse!(
        avx2_impl,
        avx2_fns,
        "avx2",
        crate::txfm::avx2_inverse,
        crate::simd::avx2::I32x4,
        crate::simd::avx2::I32x8,
        crate::simd::avx2::I16x4,
        crate::simd::avx2::I16x8,
        crate::simd::avx2::I16x16
    );
    pub(super) use avx2_impl::avx2_fns;

    const fn is_adst(kernel: Kernel1d) -> bool {
        matches!(kernel, Kernel1d::Adst | Kernel1d::FlipAdst)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn add_10_16x16_512(
        vertical: Kernel1d,
        horizontal: Kernel1d,
        coeffs: &[i32],
        nonzero_rows: usize,
        dst: &mut [u16],
        dst_stride: usize,
        depth: BitDepth,
    ) {
        // SAFETY: AVX-512F/BW/VL (and AVX2) enabled; 16 lanes divide 16x16.
        unsafe {
            if is_adst(vertical) && is_adst(horizontal) {
                inverse_add_lanes::<I32x16, i32, u16, 18, 16, 16, 16>(
                    vertical,
                    horizontal,
                    coeffs,
                    nonzero_rows,
                    dst,
                    dst_stride,
                    depth.max(),
                );
            } else {
                avx2_inverse::av1_inverse_add_u16::<i32, 16, 16>(
                    (vertical, horizontal),
                    coeffs,
                    nonzero_rows,
                    dst,
                    dst_stride,
                    depth.max(),
                );
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn add_12_512<const W: usize, const H: usize>(
        vertical: Kernel1d,
        horizontal: Kernel1d,
        coeffs: &[i32],
        nonzero_rows: usize,
        dst: &mut [u16],
        dst_stride: usize,
        depth: BitDepth,
    ) {
        // SAFETY: AVX-512F/BW/VL enabled; 16 lanes divide every installed `W >= 16, H >= 16`.
        unsafe {
            simd_add_12::<I32x16, W, H>(
                vertical,
                horizontal,
                coeffs,
                nonzero_rows,
                dst,
                dst_stride,
                depth,
            );
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn dc_512(
        residual: i32,
        width: usize,
        height: usize,
        dst: &mut [u16],
        stride: usize,
        depth: BitDepth,
    ) {
        // SAFETY: AVX-512F/BW/VL enabled by this wrapper.
        unsafe {
            constant_add_u16_lanes::<
                crate::simd::avx2::I16x4,
                crate::simd::avx2::I16x8,
                crate::simd::avx512_i16::I16x32,
            >(residual, width, height, dst, stride, depth);
        }
    }

    pub(super) fn avx512_fns() -> HighAv1InverseFns {
        let mut table = avx2_fns();
        macro_rules! install_16 {
            ($arr:expr, $f:ident) => {
                $arr[2] = $f::<16, 16>;
                $arr[3] = $f::<32, 32>;
                $arr[4] = $f::<64, 64>;
                $arr[9] = $f::<16, 32>;
                $arr[10] = $f::<32, 16>;
                $arr[11] = $f::<32, 64>;
                $arr[12] = $f::<64, 32>;
                $arr[17] = $f::<16, 64>;
                $arr[18] = $f::<64, 16>;
            };
        }
        table.add_10[2] = add_10_16x16_512;
        install_16!(table.add_12, add_12_512);
        table.dc = dc_512;
        table
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    isa_high_inverse!(
        neon_impl,
        neon_fns,
        "neon",
        crate::txfm::neon_inverse,
        crate::simd::neon::I32x4,
        crate::simd::neon::I32x4,
        crate::simd::neon::I16x4,
        crate::simd::neon::I16x8,
        crate::simd::neon::I16x8
    );
    pub(super) use neon_impl::neon_fns;
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128 {
    isa_high_inverse!(
        simd128_impl,
        simd128_fns,
        "simd128",
        crate::txfm::simd128_inverse,
        crate::simd::simd128::I32x4,
        crate::simd::simd128::I32x4,
        crate::simd::simd128::I16x4,
        crate::simd::simd128::I16x8,
        crate::simd::simd128::I16x8
    );
    pub(super) use simd128_impl::simd128_fns;
}
