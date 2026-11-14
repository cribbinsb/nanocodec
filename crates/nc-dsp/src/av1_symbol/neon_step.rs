//! Fused NEON symbol step, guided by dav1d 1.5.4 arm/64/msac.S
//! `decode_update` (BSD-2-Clause, Copyright 2019 VideoLAN/dav1d authors and
//! Martin Storsjo; `THIRD_PARTY.md`). Loaded CDFs and comparison masks serve
//! both interval search and adaptation. Rust owns refill and counters.
use super::{Av1SymbolStep, padded, step::normalize};
use core::arch::aarch64::{
    uint16x8_t, vaddq_u16, vandq_u16, vcgeq_u16, vcltq_u16, vdupq_n_s16, vdupq_n_u16, vextq_u16,
    vget_lane_u16, vget_low_u8, vgetq_lane_u16, vld1q_u16, vmvnq_u16, vpaddq_u16, vqdmulhq_s16,
    vqtbl1_u8, vreinterpret_u16_u8, vreinterpretq_s16_u16, vreinterpretq_u8_u16,
    vreinterpretq_u16_s16, vshlq_n_u16, vshlq_s16, vst1q_u16, vsubq_u16,
};

/// Signed biased difference is exact even for a zero CDF before the
/// decoded symbol: p - ((p - 32768 + 2^rate - 1) >> rate).
#[inline]
#[target_feature(enable = "neon")]
fn adapt8(p: uint16x8_t, before: uint16x8_t, rate: u32) -> uint16x8_t {
    let bias = vdupq_n_u16((32769 - (1u32 << rate)) as u16);
    let delta = vreinterpretq_s16_u16(vsubq_u16(p, vandq_u16(before, bias)));
    vsubq_u16(
        p,
        vreinterpretq_u16_s16(vshlq_s16(delta, vdupq_n_s16(-(rate as i16)))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn eight<const N: usize>(
    icdf: &mut [u16; N],
    symbols: usize,
    window: u64,
    range: u32,
    rate: Option<u32>,
) -> Av1SymbolStep {
    let mut lanes: [u16; 8] = padded(icdf);
    // SAFETY: the local array provides eight initialized readable lanes.
    let p = unsafe { vld1q_u16(lanes.as_ptr()) };
    let indices = [0u16, 1, 2, 3, 4, 5, 6, 7];
    // SAFETY: indices has eight initialized lanes.
    let index = unsafe { vld1q_u16(indices.as_ptr()) };
    let last = vdupq_n_u16((symbols - 1) as u16);
    let prob = vandq_u16(p, vdupq_n_u16(0xffc0));
    let product = vreinterpretq_u16_s16(vqdmulhq_s16(
        vreinterpretq_s16_u16(prob),
        vdupq_n_s16((range & 0x7f00) as i16),
    ));
    let bounds = vandq_u16(
        vaddq_u16(
            vaddq_u16(prob, product),
            vshlq_n_u16::<2>(vsubq_u16(last, index)),
        ),
        vcltq_u16(index, last),
    );
    let selected = vcgeq_u16(vdupq_n_u16((window >> 48) as u16), bounds);
    // Summing the selected suffix creates the byte pair [2*s, 2*s+1]
    // directly, avoiding a scalar round trip to form table indices.
    let lookup = [
        0xfdfeu16, 0xfdfe, 0xfdfe, 0xfdfe, 0xfdfe, 0xfdfe, 0xfdfe, 0x0f0e,
    ];
    // SAFETY: lookup contains eight initialized lanes.
    let lookup = unsafe { vld1q_u16(lookup.as_ptr()) };
    let key = vandq_u16(selected, lookup);
    let key = vpaddq_u16(key, key);
    let key = vpaddq_u16(key, key);
    let key = vpaddq_u16(key, key);
    let table_index = vget_low_u8(vreinterpretq_u8_u16(key));
    let symbol = usize::from(vgetq_lane_u16::<0>(key) as u8) / 2;
    let upper = vextq_u16::<7>(vdupq_n_u16(range as u16), bounds);
    let lower = u32::from(vget_lane_u16::<0>(vreinterpret_u16_u8(vqtbl1_u8(
        vreinterpretq_u8_u16(bounds),
        table_index,
    ))));
    let interval = u32::from(vget_lane_u16::<0>(vreinterpret_u16_u8(vqtbl1_u8(
        vreinterpretq_u8_u16(vsubq_u16(upper, bounds)),
        table_index,
    ))));
    if let Some(rate) = rate {
        // SAFETY: lanes is an eight-element writable local array; copy
        // back only N entries, never overwriting the codec's CDF counter.
        unsafe { vst1q_u16(lanes.as_mut_ptr(), adapt8(p, vmvnq_u16(selected), rate)) };
        icdf.copy_from_slice(&lanes[..N]);
    }
    normalize(symbol, window, lower, interval)
}

#[inline]
pub(super) fn decode<const N: usize>(
    icdf: &mut [u16; N],
    symbols: usize,
    window: u64,
    range: u32,
    rate: Option<u32>,
) -> Av1SymbolStep {
    // SAFETY: NEON is baseline on aarch64; the parent restricts N to 5..8.
    unsafe { eight(icdf, symbols, window, range, rate) }
}
