//! AV1 grain row arithmetic, dispatched per decoder through `Dsp`.
//!
//! Independently reimplemented from AV1 film grain synthesis and the BSD-2
//! dav1d 1.5.4 `filmgrain_tmpl.c` arithmetic; see `THIRD_PARTY.md`.

#![allow(unsafe_code)]
use crate::Isa;
use crate::simd::LOOKUP_TABLE;
mod generic;

/// A plane's scaling function for one frame (AV1 §7.18.3.4): the lookup
/// (0..=255 per sample value, as dav1d's `uint8_t` table; NEON's `tbl`
/// reads it directly) and `ScalingShift` (8..=11), with the same lookup
/// pre-scaled to the SIMD kernels' Q15 multiply, `scaling << (15 - shift)`
/// (at most `255 << 7`, so it fits `i16`), built once per frame rather than
/// shifted per sample.
#[derive(Debug, Clone)]
pub struct Av1GrainScaling {
    lut: [u8; 256],
    #[cfg_attr(
        target_arch = "aarch64",
        allow(dead_code, reason = "NEON looks up `lut`")
    )]
    q15: [i16; LOOKUP_TABLE],
    shift: u8,
}

/// The Q15 multiply's precision.
const Q15: u8 = 15;

impl Av1GrainScaling {
    /// The scaling of lookup `lut` with `ScalingShift` `shift`.
    ///
    /// # Panics
    ///
    /// If `shift` is not 8..=11.
    #[must_use]
    pub fn new(lut: &[u8; 256], shift: u8) -> Self {
        assert!((8..=11).contains(&shift), "grain scaling shift {shift}");
        Self {
            lut: *lut,
            q15: core::array::from_fn(|i| lut.get(i).map_or(0, |&v| i16::from(v) << (Q15 - shift))),
            shift,
        }
    }
}
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;
#[cfg(target_arch = "x86_64")]
mod x86;

type LumaFn = unsafe fn(&[u8], &[i8], &Av1GrainScaling, [u8; 2], &mut [u8]);
type ChromaFn = unsafe fn(&[u8], &[u8], &[i8], &Av1GrainScaling, [u8; 2], [i16; 3], &mut [u8]);
type BlendFn = unsafe fn(&[i8], &mut [i8], [i16; 2]);

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

fn apply(original: u8, grain: i8, scale: u8, shift: u8, bounds: [u8; 2]) -> u8 {
    let noise = (i32::from(scale) * i32::from(grain) + (1 << (shift - 1))) >> shift;
    (i32::from(original) + noise).clamp(i32::from(bounds[0]), i32::from(bounds[1])) as u8
}

pub(super) fn luma(
    source: &[u8],
    grain: &[i8],
    scaling: &Av1GrainScaling,
    bounds: [u8; 2],
    out: &mut [u8],
) {
    for ((&original, &noise), dst) in source.iter().zip(grain).zip(out) {
        *dst = apply(
            original,
            noise,
            scaling.lut[usize::from(original)],
            scaling.shift,
            bounds,
        );
    }
}

pub(super) fn chroma(
    source: &[u8],
    luma: &[u8],
    grain: &[i8],
    scaling: &Av1GrainScaling,
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
        *dst = apply(original, noise, scaling.lut[index], scaling.shift, bounds);
    }
}

pub(super) fn blend(old: &[i8], new: &mut [i8], weights: [i16; 2]) {
    for (&previous, next) in old.iter().zip(new) {
        *next = ((i32::from(previous) * i32::from(weights[0])
            + i32::from(*next) * i32::from(weights[1])
            + 16)
            >> 5)
            .clamp(i32::from(i8::MIN), i32::from(i8::MAX)) as i8;
    }
}
