//! Sixteen-column byte classifier and CCSO lookup for `AArch64`.

#![allow(unsafe_code)]

use core::arch::aarch64::{
    uint8x16_t, uint8x16x2_t, vabsq_s8, vaddq_u8, vandq_u8, vbslq_u8, vcleq_u8, vcltzq_s8,
    vdupq_n_s8, vdupq_n_u8, vld1q_u8, vld1q_u8_x4, vld2q_u8, vorrq_u8, vqaddq_u8, vqsubq_u8,
    vqtbl1q_u8, vqtbl2q_u8, vqtbl4q_u8, vreinterpretq_s8_u8, vreinterpretq_u8_s8, vshlq_n_u8,
    vshlq_u8, vshrq_n_u8, vst1q_u8, vsubq_u8, vtstq_u8,
};

use super::{CcsoFilter, CcsoMode};

#[target_feature(enable = "neon")]
unsafe fn load_luma<const CHROMA: bool>(luma: *const u8) -> uint8x16_t {
    // SAFETY: the caller validates the complete 16- or 32-byte source span.
    unsafe {
        if CHROMA {
            vld2q_u8(luma).0
        } else {
            vld1q_u8(luma)
        }
    }
}

#[target_feature(enable = "neon")]
fn classify<const THREE: bool>(
    center: uint8x16_t,
    neighbour: uint8x16_t,
    threshold: uint8x16_t,
) -> uint8x16_t {
    let not_low = vcleq_u8(vqsubq_u8(center, neighbour), threshold);
    if THREE {
        let not_high = vcleq_u8(vqsubq_u8(neighbour, center), threshold);
        vsubq_u8(vaddq_u8(vdupq_n_u8(1), not_high), not_low)
    } else {
        vandq_u8(not_low, vdupq_n_u8(1))
    }
}

#[target_feature(enable = "neon")]
unsafe fn span<const CHROMA: bool, const LEVELS: u8>(
    filter: &CcsoFilter,
    dst: &mut [u8],
    source: [&[u8]; 3],
    threshold: u8,
) {
    const LANES: usize = 16;
    let [center, first, second] = source.map(<[u8]>::as_ptr);
    let pitch = if CHROMA { LANES * 2 } else { LANES };
    let threshold = vdupq_n_u8(threshold);
    let shift = vdupq_n_s8(-(filter.band_shift as i8));
    let one = vdupq_n_u8(1);
    let low_nibble = vdupq_n_u8(15);

    // SAFETY: all tables include their full vector footprint, and vector
    // loads in the loop are limited to complete destination chunks.
    unsafe {
        let codes = vld1q_u8_x4(filter.codes.as_ptr());
        let offsets = vld1q_u8(filter.offsets.as_ptr().cast());
        for (chunk, dst) in dst.as_chunks_mut::<LANES>().0.iter_mut().enumerate() {
            let at = chunk * pitch;
            let center = load_luma::<CHROMA>(center.add(at));
            let band = vshlq_u8(center, shift);
            let index = if LEVELS == 1 {
                band
            } else {
                let first = load_luma::<CHROMA>(first.add(at));
                let second = load_luma::<CHROMA>(second.add(at));
                let (a, b) = if LEVELS == 2 {
                    (
                        classify::<false>(center, first, threshold),
                        classify::<false>(center, second, threshold),
                    )
                } else {
                    (
                        classify::<true>(center, first, threshold),
                        classify::<true>(center, second, threshold),
                    )
                };
                vorrq_u8(vshlq_n_u8::<4>(band), vorrq_u8(vshlq_n_u8::<2>(a), b))
            };
            let byte = vshrq_n_u8::<1>(index);
            let packed = if LEVELS == 1 {
                vqtbl2q_u8(uint8x16x2_t(codes.0, codes.1), byte)
            } else {
                vqtbl4q_u8(codes, byte)
            };
            let code = vbslq_u8(vtstq_u8(index, one), vshrq_n_u8::<4>(packed), packed);
            let offset = vqtbl1q_u8(offsets, vandq_u8(code, low_nibble));
            let sign = vcltzq_s8(vreinterpretq_s8_u8(offset));
            let dst = dst.as_mut_ptr();
            let pixels = vld1q_u8(dst);
            let added = vqaddq_u8(pixels, offset);
            let subtracted = vqsubq_u8(
                pixels,
                vreinterpretq_u8_s8(vabsq_s8(vreinterpretq_s8_u8(offset))),
            );
            vst1q_u8(dst, vbslq_u8(sign, subtracted, added));
        }
    }
}

/// # Safety
/// NEON must be enabled and the source slices must contain the destination's
/// complete luma footprint.
#[target_feature(enable = "neon")]
pub(super) unsafe fn apply(
    filter: &CcsoFilter,
    dst: &mut [u8],
    source: [&[u8]; 3],
    chroma: bool,
) -> usize {
    // SAFETY: the full SIMD chunks stay within the validated spans.
    unsafe {
        match (chroma, filter.mode) {
            (false, CcsoMode::Bands) => span::<false, 1>(filter, dst, source, 0),
            (true, CcsoMode::Bands) => span::<true, 1>(filter, dst, source, 0),
            (false, CcsoMode::Two(t)) => span::<false, 2>(filter, dst, source, t),
            (true, CcsoMode::Two(t)) => span::<true, 2>(filter, dst, source, t),
            (false, CcsoMode::Three(t)) => span::<false, 3>(filter, dst, source, t),
            (true, CcsoMode::Three(t)) => span::<true, 3>(filter, dst, source, t),
        }
    }
    dst.len() & !15
}
