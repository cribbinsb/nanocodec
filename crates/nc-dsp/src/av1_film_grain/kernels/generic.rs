//! Shared AV1 grain lane arithmetic; ISA wrappers supply the vector widths.
#![allow(unsafe_code, clippy::inline_always)]
use super::Av1GrainScaling;
use crate::simd::I16Lanes;

/// Add the scaled grain `noise` (lanes of `grain`) to `source` and clip.
#[inline(always)]
unsafe fn add_noise<V: I16Lanes>(
    source: &[u8],
    grain: &[i16],
    scales: &[i16; 16],
    bounds: [u8; 2],
    out: &mut [u8],
) {
    // The Q15 scale reproduces AV1's signed round-shift exactly
    // (`Av1GrainScaling`) without overflowing a signed 16-bit product.
    let noise = V::load(grain).mul_q15_round(V::load(scales));
    V::load_u8(source)
        .add(noise)
        .max(V::splat(i16::from(bounds[0])))
        .min(V::splat(i16::from(bounds[1])))
        .store_u8_sat(out);
}

#[inline(always)]
pub(super) unsafe fn luma<V: I16Lanes>(
    source: &[u8],
    grain: &[i16],
    scaling: &Av1GrainScaling,
    bounds: [u8; 2],
    out: &mut [u8],
) {
    let q15 = &scaling.q15;
    let end = source.len() / V::LANES * V::LANES;
    for x in (0..end).step_by(V::LANES) {
        let mut scales = [0i16; 16];
        for (dst, &original) in scales.iter_mut().zip(&source[x..x + V::LANES]) {
            *dst = q15[usize::from(original)];
        }
        // SAFETY: forwarded ISA contract; the lanes lie in the rows.
        unsafe { add_noise::<V>(&source[x..], &grain[x..], &scales, bounds, &mut out[x..]) };
    }
    super::luma(
        &source[end..],
        &grain[end..],
        scaling,
        bounds,
        &mut out[end..],
    );
}

#[inline(always)]
pub(super) unsafe fn chroma<V: I16Lanes>(
    source: &[u8],
    luma: &[u8],
    grain: &[i16],
    scaling: &Av1GrainScaling,
    bounds: [u8; 2],
    coefficients: [i16; 3],
    out: &mut [u8],
) {
    let q15 = &scaling.q15;
    let [mult, luma_mult, offset] = coefficients.map(i32::from);
    // Scaling from luma (`[0, 64, 0]`) indexes by the luma average itself.
    let from_luma = coefficients == [0, 64, 0];
    // Lanes whose luma pairs are both inside the row; the rest (an odd
    // final luma sample) take the reference.
    let end = source.len().min(luma.len() / 2) / V::LANES * V::LANES;
    for x in (0..end).step_by(V::LANES) {
        let mut scales = [0i16; 16];
        let (pairs, _) = luma[2 * x..2 * (x + V::LANES)].as_chunks::<2>();
        for ((dst, &original), pair) in scales.iter_mut().zip(&source[x..x + V::LANES]).zip(pairs) {
            let average = (i32::from(pair[0]) + i32::from(pair[1]) + 1) >> 1;
            let index = if from_luma {
                average
            } else {
                (((average * luma_mult + mult * i32::from(original)) >> 6) + offset).clamp(0, 255)
            };
            *dst = q15[(index as usize) & 0xff];
        }
        // SAFETY: forwarded ISA contract; the lanes lie in the rows.
        unsafe { add_noise::<V>(&source[x..], &grain[x..], &scales, bounds, &mut out[x..]) };
    }
    if end < source.len() {
        super::chroma(
            &source[end..],
            &luma[2 * end..],
            &grain[end..],
            scaling,
            bounds,
            coefficients,
            &mut out[end..],
        );
    }
}

#[inline(always)]
pub(super) unsafe fn blend<V: I16Lanes>(old: &[i16], new: &mut [i16], weights: [i16; 2]) {
    let end = new.len() / V::LANES * V::LANES;
    for x in (0..end).step_by(V::LANES) {
        V::load(&old[x..])
            .mul(V::splat(weights[0]))
            .add(V::load(&new[x..]).mul(V::splat(weights[1])))
            .add(V::splat(16))
            .sra(5)
            .max(V::splat(-128))
            .min(V::splat(127))
            .store(&mut new[x..]);
    }
    super::blend(&old[end..], &mut new[end..], weights);
}
