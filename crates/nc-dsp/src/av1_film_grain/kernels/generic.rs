//! Shared AV1 grain lane arithmetic; ISA wrappers supply the vector widths.
#![allow(unsafe_code, clippy::inline_always)]
use crate::simd::{I16Lanes, I32Lanes};

#[inline(always)]
pub(super) unsafe fn luma<V: I16Lanes>(
    source: &[u8],
    grain: &[i16],
    scaling: &[i32; 256],
    shift: u8,
    bounds: [u8; 2],
    out: &mut [u8],
) {
    let end = source.len() / V::LANES * V::LANES;
    for x in (0..end).step_by(V::LANES) {
        let mut scales = [0i16; 16];
        for (dst, &original) in scales[..V::LANES].iter_mut().zip(&source[x..]) {
            *dst = (scaling[usize::from(original)] << (15 - shift)) as i16;
        }
        // Scale is at most 255 << 7: Q15 rounding exactly reproduces AV1's
        // signed round-shift without overflowing a signed 16-bit product.
        let noise = V::load(&grain[x..]).mul_q15_round(V::load(&scales));
        V::load_u8(&source[x..])
            .add(noise)
            .max(V::splat(i16::from(bounds[0])))
            .min(V::splat(i16::from(bounds[1])))
            .store_u8_sat(&mut out[x..]);
    }
    super::luma(
        &source[end..],
        &grain[end..],
        scaling,
        shift,
        bounds,
        &mut out[end..],
    );
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn chroma<V: I16Lanes, W: I32Lanes>(
    source: &[u8],
    luma: &[u8],
    grain: &[i16],
    scaling: &[i32; 256],
    shift: u8,
    bounds: [u8; 2],
    coefficients: [i16; 3],
    out: &mut [u8],
) {
    let end = source.len() / V::LANES * V::LANES;
    for x in (0..end).step_by(V::LANES) {
        let mut originals = [0i32; 16];
        let mut averages = [0i32; 16];
        let mut indices = [0i32; 16];
        for i in 0..V::LANES {
            originals[i] = i32::from(source[x + i]);
            averages[i] = (i32::from(luma[2 * (x + i)])
                + i32::from(luma[(2 * (x + i) + 1).min(luma.len() - 1)])
                + 1)
                >> 1;
        }
        for i in (0..V::LANES).step_by(W::LANES) {
            W::load(&originals[i..])
                .mul_scalar(i32::from(coefficients[0]))
                .add(W::load(&averages[i..]).mul_scalar(i32::from(coefficients[1])))
                .sra(6)
                .add(W::splat(i32::from(coefficients[2])))
                .max(W::splat(0))
                .min(W::splat(255))
                .store(&mut indices[i..]);
        }
        let mut scales = [0i16; 16];
        for (dst, index) in scales[..V::LANES].iter_mut().zip(indices) {
            *dst = (scaling[index as usize] << (15 - shift)) as i16;
        }
        let noise = V::load(&grain[x..]).mul_q15_round(V::load(&scales));
        V::load_u8(&source[x..])
            .add(noise)
            .max(V::splat(i16::from(bounds[0])))
            .min(V::splat(i16::from(bounds[1])))
            .store_u8_sat(&mut out[x..]);
    }
    if end < source.len() {
        super::chroma(
            &source[end..],
            &luma[2 * end..],
            &grain[end..],
            scaling,
            shift,
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
