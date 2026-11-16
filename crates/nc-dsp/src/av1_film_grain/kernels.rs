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

/// The scaling function of 8-bit lookup `lut` at sample `index` of
/// `bits_above` bits over 8 (AV1 §7.18.3.5 `scale_lut`): deeper samples
/// interpolate between the two entries their top 8 bits select.
pub(crate) fn scale_lut(lut: &[u8; 256], index: usize, bits_above: u32) -> i32 {
    let x = index >> bits_above;
    let start = i32::from(lut[x]);
    if bits_above == 0 || x == lut.len() - 1 {
        return start;
    }
    let rem = (index - (x << bits_above)) as i32;
    let end = i32::from(lut[x + 1]);
    start + (((end - start) * rem + (1 << (bits_above - 1))) >> bits_above)
}

/// Luma grain at any depth (the scalar reference of [`luma`] with
/// [`scale_lut`]): `bounds` are the depth's clip.
pub(crate) fn luma_at<T: nc_common::Sample>(
    source: &[T],
    grain: &[i16],
    scaling: &Av1GrainScaling,
    (bounds, bits_above): ([i32; 2], u32),
    out: &mut [T],
) {
    for ((&original, &noise), dst) in source.iter().zip(grain).zip(out) {
        let original = original.to_i32();
        let scale = scale_lut(&scaling.lut, original as usize, bits_above);
        *dst = T::from_i32(add_noise(
            original,
            i32::from(noise),
            scale,
            scaling.shift,
            bounds,
        ));
    }
}

/// Chroma grain at any depth (the scalar reference of [`chroma`]): the
/// merged index is `Clip1((combined >> 6) + (offset << (BitDepth − 8)))`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_at<T: nc_common::Sample>(
    source: &[T],
    luma: &[T],
    grain: &[i16],
    scaling: &Av1GrainScaling,
    (bounds, bits_above): ([i32; 2], u32),
    coefficients: [i16; 3],
    out: &mut [T],
) {
    let [mult, luma_mult, offset] = coefficients.map(i32::from);
    let max_index = (256 << bits_above) - 1;
    for (x, ((&original, &noise), dst)) in source.iter().zip(grain).zip(out).enumerate() {
        let original = original.to_i32();
        let average =
            (luma[2 * x].to_i32() + luma[(2 * x + 1).min(luma.len() - 1)].to_i32() + 1) >> 1;
        let index = (((average * luma_mult + mult * original) >> 6) + (offset << bits_above))
            .clamp(0, max_index) as usize;
        let scale = scale_lut(&scaling.lut, index, bits_above);
        *dst = T::from_i32(add_noise(
            original,
            i32::from(noise),
            scale,
            scaling.shift,
            bounds,
        ));
    }
}

/// Grain overlap blend at any depth, clamped to `range`.
pub(crate) fn blend_at(old: &[i16], new: &mut [i16], weights: [i16; 2], (min, max): (i32, i32)) {
    for (&previous, next) in old.iter().zip(new) {
        let mixed =
            i32::from(previous) * i32::from(weights[0]) + i32::from(*next) * i32::from(weights[1]);
        *next = ((mixed + 16) >> 5).clamp(min, max) as i16;
    }
}

/// `Clip3(bounds, original + Round2(scale * grain, shift))`.
fn add_noise(original: i32, grain: i32, scale: i32, shift: u8, bounds: [i32; 2]) -> i32 {
    let noise = (scale * grain + (1 << (shift - 1))) >> shift;
    (original + noise).clamp(bounds[0], bounds[1])
}

fn apply(original: u8, grain: i8, scale: u8, shift: u8, bounds: [u8; 2]) -> u8 {
    // Within `bounds`, so a byte.
    add_noise(
        i32::from(original),
        i32::from(grain),
        i32::from(scale),
        shift,
        bounds.map(i32::from),
    ) as u8
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
