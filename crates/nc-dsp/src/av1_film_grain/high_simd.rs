//! High-bit-depth (`u16`, 10- and 12-bit) AV1 film grain row SIMD kernels
//! across AVX2, AVX-512, and NEON.
//!
//! Attribution: Q15 `pmulhrsw`/`sqrdmulh` grain scaling, interpolated `u12`
//! lookup table expansion, and paired chroma-from-luma index arithmetic follow
//! dav1d 1.5.4 (`src/fg_apply_tmpl.c`, `src/x86/filmgrain16_avx2.asm`,
//! `src/x86/filmgrain16_avx512.asm`, `src/arm/64/filmgrain16.S`, BSD-2-Clause;
//! see `THIRD_PARTY.md`).

#![allow(unsafe_code, clippy::inline_always)]
#![cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code, unused_imports)
)]

use nc_common::BitDepth;

use super::Av1GrainScaling;
use super::kernels::{blend_at, chroma_at, luma_at, scale_lut};
use crate::Isa;
use crate::av1_mc::high_blend::BlendSample;
use crate::simd::{HIGH_LOOKUP_TABLE, I16Lanes};

/// Pre-interpolated Q15 grain scaling lookup for one plane at `depth` (`<= 12` bpc).
#[derive(Debug, Clone)]
pub(crate) struct HighGrainScaling {
    pub base: Av1GrainScaling,
    pub q15: [i16; HIGH_LOOKUP_TABLE],
    pub depth: BitDepth,
}

impl HighGrainScaling {
    #[must_use]
    pub fn new(lut: &[u8; 256], shift: u8, depth: BitDepth) -> Box<Self> {
        let base = Av1GrainScaling::new(lut, shift);
        let mut q15 = [0i16; HIGH_LOOKUP_TABLE];
        if depth.bits() <= 12 {
            let bits = depth.above_eight();
            let max_index = (256usize << bits) - 1;
            let up = 15 - shift;
            for (i, slot) in q15[..=max_index].iter_mut().enumerate() {
                *slot = (scale_lut(lut, i, bits) as i16) << up;
            }
        }
        Box::new(Self { base, q15, depth })
    }
}

pub(crate) type HighLumaRowFn = unsafe fn(
    source: &[u16],
    grain: &[i16],
    scaling: &HighGrainScaling,
    bounds: [i32; 2],
    out: &mut [u16],
);

pub(crate) type HighChromaRowFn = unsafe fn(
    source: &[u16],
    luma: &[u16],
    grain: &[i16],
    scaling: &HighGrainScaling,
    bounds: [i32; 2],
    coefficients: [i16; 3],
    out: &mut [u16],
);

pub(crate) type HighBlendRowFn =
    unsafe fn(old: &[i16], new: &mut [i16], weights: [i16; 2], range: (i32, i32));

/// High-bit-depth (`u16`) AV1 film grain row kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HighAv1GrainFns {
    pub luma: HighLumaRowFn,
    pub chroma: HighChromaRowFn,
    pub blend: HighBlendRowFn,
}

fn scalar_luma(
    source: &[u16],
    grain: &[i16],
    scaling: &HighGrainScaling,
    bounds: [i32; 2],
    out: &mut [u16],
) {
    luma_at(
        source,
        grain,
        &scaling.base,
        (bounds, scaling.depth.above_eight()),
        out,
    );
}

#[allow(clippy::too_many_arguments)]
fn scalar_chroma(
    source: &[u16],
    luma: &[u16],
    grain: &[i16],
    scaling: &HighGrainScaling,
    bounds: [i32; 2],
    coefficients: [i16; 3],
    out: &mut [u16],
) {
    chroma_at(
        source,
        luma,
        grain,
        &scaling.base,
        (bounds, scaling.depth.above_eight()),
        coefficients,
        out,
    );
}

fn scalar_blend(old: &[i16], new: &mut [i16], weights: [i16; 2], range: (i32, i32)) {
    blend_at(old, new, weights, range);
}

impl HighAv1GrainFns {
    pub const SCALAR: Self = Self {
        luma: scalar_luma,
        chroma: scalar_chroma,
        blend: scalar_blend,
    };

    pub fn for_isa(isa: Isa) -> Self {
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => x86_avx2::FNS,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => x86_avx512::FNS,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => neon_high::FNS,
            _ => Self::SCALAR,
        }
    }
}

impl crate::Dsp {
    pub(super) fn av1_grain_high_luma_row(
        &self,
        source: &[u16],
        grain: &[i16],
        scaling: &HighGrainScaling,
        bounds: [i32; 2],
        out: &mut [u16],
    ) {
        // SAFETY: `self.high` matches this CPU's supported ISA.
        unsafe { (self.high.av1_grain.luma)(source, grain, scaling, bounds, out) };
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn av1_grain_high_chroma_row(
        &self,
        source: &[u16],
        luma: &[u16],
        grain: &[i16],
        scaling: &HighGrainScaling,
        bounds: [i32; 2],
        coefficients: [i16; 3],
        out: &mut [u16],
    ) {
        // SAFETY: `self.high` matches this CPU's supported ISA.
        unsafe {
            (self.high.av1_grain.chroma)(source, luma, grain, scaling, bounds, coefficients, out);
        }
    }

    pub(super) fn av1_grain_high_blend_row(
        &self,
        old: &[i16],
        new: &mut [i16],
        weights: [i16; 2],
        range: (i32, i32),
    ) {
        // SAFETY: `self.high` matches this CPU's supported ISA.
        unsafe { (self.high.av1_grain.blend)(old, new, weights, range) };
    }
}

#[inline(always)]
unsafe fn luma_row_lanes<V: I16Lanes>(
    source: &[u16],
    grain: &[i16],
    scaling: &HighGrainScaling,
    bounds: [i32; 2],
    out: &mut [u16],
) {
    let len = source.len().min(grain.len()).min(out.len());
    let bits = scaling.depth.above_eight();
    if scaling.depth.bits() > 12 {
        luma_at(
            &source[..len],
            &grain[..len],
            &scaling.base,
            (bounds, bits),
            &mut out[..len],
        );
        return;
    }
    let (lo, hi) = (V::splat(bounds[0] as i16), V::splat(bounds[1] as i16));
    let end = len / V::LANES * V::LANES;
    for x in (0..end).step_by(V::LANES) {
        // SAFETY: `x + V::LANES <= end <= len`.
        unsafe {
            let orig = u16::load_i16::<V>(&source[x..]);
            let scales = orig.lookup_u12(&scaling.q15);
            let noise = V::load(&grain[x..]).mul_q15_round(scales);
            u16::store_i16(orig.add(noise).max(lo).min(hi), &mut out[x..]);
        }
    }
    if end < len {
        luma_at(
            &source[end..len],
            &grain[end..len],
            &scaling.base,
            (bounds, bits),
            &mut out[end..len],
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn chroma_row_lanes<V: I16Lanes>(
    source: &[u16],
    luma: &[u16],
    grain: &[i16],
    scaling: &HighGrainScaling,
    bounds: [i32; 2],
    coefficients: [i16; 3],
    out: &mut [u16],
) {
    let len = source.len().min(grain.len()).min(out.len());
    let bits = scaling.depth.above_eight();
    if scaling.depth.bits() > 12 || luma.is_empty() {
        chroma_at(
            &source[..len],
            luma,
            &grain[..len],
            &scaling.base,
            (bounds, bits),
            coefficients,
            &mut out[..len],
        );
        return;
    }
    let (lo, hi) = (V::splat(bounds[0] as i16), V::splat(bounds[1] as i16));
    let (zero, max_idx) = (V::splat(0), V::splat(((256 << bits) - 1) as i16));
    let [mult, luma_mult, offset] = coefficients;
    let from_luma = coefficients == [0, 64, 0];
    let bias = i32::from(offset) << (bits + 6);
    let end = len.min(luma.len() / 2) / V::LANES * V::LANES;
    for x in (0..end).step_by(V::LANES) {
        // SAFETY: `x + V::LANES <= end <= len` and `2 * (x + V::LANES) <= luma.len()`.
        unsafe {
            let orig = u16::load_i16::<V>(&source[x..]);
            let avg = V::pair_avg_u16(&luma[2 * x..2 * (x + V::LANES)]);
            let idx = if from_luma {
                avg.max(zero).min(max_idx)
            } else {
                let acc = V::acc_mul_add2(V::acc_zero(), avg, luma_mult, orig, mult);
                V::acc_round_shift_narrow(acc, bias, 6)
                    .max(zero)
                    .min(max_idx)
            };
            let scales = idx.lookup_u12(&scaling.q15);
            let noise = V::load(&grain[x..]).mul_q15_round(scales);
            u16::store_i16(orig.add(noise).max(lo).min(hi), &mut out[x..]);
        }
    }
    if end < len {
        chroma_at(
            &source[end..len],
            &luma[2 * end..],
            &grain[end..len],
            &scaling.base,
            (bounds, bits),
            coefficients,
            &mut out[end..len],
        );
    }
}

#[inline(always)]
unsafe fn blend_row_lanes<V: I16Lanes>(
    old: &[i16],
    new: &mut [i16],
    weights: [i16; 2],
    (min, max): (i32, i32),
) {
    let len = old.len().min(new.len());
    if min < i32::from(i16::MIN) || max > i32::from(i16::MAX) {
        blend_at(&old[..len], &mut new[..len], weights, (min, max));
        return;
    }
    let (lo, hi) = (V::splat(min as i16), V::splat(max as i16));
    let end = len / V::LANES * V::LANES;
    for x in (0..end).step_by(V::LANES) {
        let acc = V::acc_mul_add2(
            V::acc_zero(),
            V::load(&old[x..]),
            weights[0],
            V::load(&new[x..]),
            weights[1],
        );
        V::acc_round_shift_narrow(acc, 16, 5)
            .max(lo)
            .min(hi)
            .store(&mut new[x..]);
    }
    if end < len {
        blend_at(&old[end..len], &mut new[end..len], weights, (min, max));
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! high_grain_isa {
    ($mod_name:ident, $(#[$feature:meta])*, $v:ty) => {
        mod $mod_name {
            use super::{
                HighAv1GrainFns, HighGrainScaling, blend_row_lanes, chroma_row_lanes,
                luma_row_lanes,
            };

            $(#[$feature])*
            unsafe fn luma(
                source: &[u16],
                grain: &[i16],
                scaling: &HighGrainScaling,
                bounds: [i32; 2],
                out: &mut [u16],
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { luma_row_lanes::<$v>(source, grain, scaling, bounds, out) };
            }

            #[allow(clippy::too_many_arguments)]
            $(#[$feature])*
            unsafe fn chroma(
                source: &[u16],
                luma_src: &[u16],
                grain: &[i16],
                scaling: &HighGrainScaling,
                bounds: [i32; 2],
                coefficients: [i16; 3],
                out: &mut [u16],
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe {
                    chroma_row_lanes::<$v>(
                        source,
                        luma_src,
                        grain,
                        scaling,
                        bounds,
                        coefficients,
                        out,
                    );
                }
            }

            $(#[$feature])*
            unsafe fn blend(old: &[i16], new: &mut [i16], weights: [i16; 2], range: (i32, i32)) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { blend_row_lanes::<$v>(old, new, weights, range) };
            }

            pub(super) const FNS: HighAv1GrainFns = HighAv1GrainFns {
                luma,
                chroma,
                blend,
            };
        }
    };
}

#[cfg(target_arch = "x86_64")]
high_grain_isa!(
    x86_avx2,
    #[target_feature(enable = "avx2")],
    crate::simd::avx2::I16x16
);

#[cfg(target_arch = "x86_64")]
high_grain_isa!(
    x86_avx512,
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")],
    crate::simd::avx512_i16::I16x32
);

#[cfg(target_arch = "aarch64")]
high_grain_isa!(
    neon_high,
    #[target_feature(enable = "neon")],
    crate::simd::neon::I16x8
);
