//! AV1 grain row arithmetic, dispatched per decoder through `Dsp`.
//!
//! Independently reimplemented from AV1 film grain synthesis and the BSD-2
//! dav1d 1.5.4 `filmgrain_tmpl.c` arithmetic; see `THIRD_PARTY.md`.

#![allow(unsafe_code)]
use crate::Isa;
mod generic;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;
#[cfg(target_arch = "x86_64")]
mod x86;

type LumaFn = unsafe fn(&[u8], &[i16], &[i32; 256], u8, [u8; 2], &mut [u8]);
type ChromaFn = unsafe fn(&[u8], &[u8], &[i16], &[i32; 256], u8, [u8; 2], [i16; 3], &mut [u8]);
type BlendFn = unsafe fn(&[i16], &mut [i16], [i16; 2]);

#[derive(Debug, Clone, Copy)]
pub(crate) struct Kernels {
    pub luma: LumaFn,
    pub chroma: ChromaFn,
    pub blend: BlendFn,
}

pub(crate) fn kernels(isa: Isa) -> Kernels {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => x86::KERNELS,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::KERNELS,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => wasm::KERNELS,
        _ => Kernels {
            luma,
            chroma,
            blend,
        },
    }
}

fn apply(original: u8, grain: i16, scale: i32, shift: u8, bounds: [u8; 2]) -> u8 {
    let noise = (scale * i32::from(grain) + (1 << (shift - 1))) >> shift;
    (i32::from(original) + noise).clamp(i32::from(bounds[0]), i32::from(bounds[1])) as u8
}

pub(super) fn luma(
    source: &[u8],
    grain: &[i16],
    scaling: &[i32; 256],
    shift: u8,
    bounds: [u8; 2],
    out: &mut [u8],
) {
    for ((&original, &noise), dst) in source.iter().zip(grain).zip(out) {
        *dst = apply(
            original,
            noise,
            scaling[usize::from(original)],
            shift,
            bounds,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn chroma(
    source: &[u8],
    luma: &[u8],
    grain: &[i16],
    scaling: &[i32; 256],
    shift: u8,
    bounds: [u8; 2],
    coefficients: [i16; 3],
    out: &mut [u8],
) {
    let [mult, luma_mult, offset] = coefficients.map(i32::from);
    for (x, ((&original, &noise), dst)) in source.iter().zip(grain).zip(out).enumerate() {
        let average =
            (i32::from(luma[2 * x]) + i32::from(luma[(2 * x + 1).min(luma.len() - 1)]) + 1) >> 1;
        let index = (((average * luma_mult + mult * i32::from(original)) >> 6) + offset)
            .clamp(0, 255) as usize;
        *dst = apply(original, noise, scaling[index], shift, bounds);
    }
}

pub(super) fn blend(old: &[i16], new: &mut [i16], weights: [i16; 2]) {
    for (&previous, next) in old.iter().zip(new) {
        *next = ((i32::from(previous) * i32::from(weights[0])
            + i32::from(*next) * i32::from(weights[1])
            + 16)
            >> 5)
            .clamp(-128, 127) as i16;
    }
}
