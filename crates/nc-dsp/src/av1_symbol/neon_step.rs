//! Fused NEON symbol step, guided by dav1d 1.5.4 `src/arm/64/msac.S`
//! `decode_update` (BSD-2-Clause, Copyright 2019 VideoLAN/dav1d authors and
//! Martin Storsjo; `THIRD_PARTY.md`). Loaded CDFs and comparison masks serve
//! both interval search and adaptation. Rust owns refill and counters.
use super::{
    Av1SymbolStep, LANES, padded,
    step::{interval_16, normalize},
};
use core::arch::aarch64::{
    uint16x8_t, vaddq_u16, vaddvq_u16, vandq_u16, vcgeq_u16, vcltq_u16, vcombine_u16, vdup_n_u16,
    vdupq_n_s16, vdupq_n_u16, vextq_u16, vget_lane_u16, vget_low_u8, vget_low_u16, vgetq_lane_u16,
    vgetq_lane_u64, vld1_u16, vld1q_u16, vmvnq_u16, vpaddq_u16, vqdmulhq_s16, vqtbl1_u8,
    vreinterpret_u16_u8, vreinterpretq_s16_u16, vreinterpretq_u8_u16, vreinterpretq_u16_s16,
    vreinterpretq_u64_u16, vshlq_n_u16, vshlq_s16, vshrq_n_u16, vst1_u16, vst1q_u16, vsubq_u16,
};

/// Signed biased difference is exact even for a zero CDF before the
/// decoded symbol: `p - ((p - 32768 + 2^rate - 1) >> rate)`.
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
fn bounds8(p: uint16x8_t, base: u16, symbols: usize, range: u32) -> uint16x8_t {
    let indices = [
        base,
        base + 1,
        base + 2,
        base + 3,
        base + 4,
        base + 5,
        base + 6,
        base + 7,
    ];
    // SAFETY: `indices` has eight initialized lanes.
    let index = unsafe { vld1q_u16(indices.as_ptr()) };
    let last = vdupq_n_u16((symbols - 1) as u16);
    let prob = vandq_u16(p, vdupq_n_u16(0xffc0));
    let product = vreinterpretq_u16_s16(vqdmulhq_s16(
        vreinterpretq_s16_u16(prob),
        vdupq_n_s16((range & 0x7f00) as i16),
    ));
    vandq_u16(
        vaddq_u16(
            vaddq_u16(prob, product),
            vshlq_n_u16::<2>(vsubq_u16(last, index)),
        ),
        vcltq_u16(index, last),
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
    let p = if N == 4 {
        // SAFETY: `N == 4`, so `icdf` has four readable `u16`s.
        unsafe { vcombine_u16(vld1_u16(icdf.as_ptr()), vdup_n_u16(0)) }
    } else if N == LANES {
        // SAFETY: `N == 8`, so `icdf` has eight readable `u16`s.
        unsafe { vld1q_u16(icdf.as_ptr()) }
    } else {
        let lanes: [u16; LANES] = padded(icdf);
        // SAFETY: `lanes` provides eight initialized readable lanes.
        unsafe { vld1q_u16(lanes.as_ptr()) }
    };
    let bounds = bounds8(p, 0, symbols, range);
    let selected = vcgeq_u16(vdupq_n_u16((window >> 48) as u16), bounds);
    // Summing the selected suffix creates the byte pair [2*s, 2*s+1]
    // directly, avoiding a scalar round trip to form table indices.
    let lookup = [
        0xfdfeu16, 0xfdfe, 0xfdfe, 0xfdfe, 0xfdfe, 0xfdfe, 0xfdfe, 0x0f0e,
    ];
    // SAFETY: `lookup` contains eight initialized lanes.
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
        let adapted = adapt8(p, vmvnq_u16(selected), rate);
        if N == 4 {
            // SAFETY: `N == 4`, so `icdf` has four writable `u16`s.
            unsafe { vst1_u16(icdf.as_mut_ptr(), vget_low_u16(adapted)) };
        } else if N == LANES {
            // SAFETY: `N == 8`, so `icdf` has eight writable `u16`s.
            unsafe { vst1q_u16(icdf.as_mut_ptr(), adapted) };
        } else {
            let mut lanes = [0u16; LANES];
            // SAFETY: `lanes` is an eight-element writable local array.
            unsafe { vst1q_u16(lanes.as_mut_ptr(), adapted) };
            icdf.copy_from_slice(&lanes[..N]);
        }
    }
    normalize(symbol, window, lower, interval)
}

#[inline]
#[target_feature(enable = "neon")]
fn sixteen<const N: usize>(
    icdf: &mut [u16; N],
    symbols: usize,
    window: u64,
    range: u32,
    rate: Option<u32>,
) -> Av1SymbolStep {
    let (p0, p1) = if N == 2 * LANES {
        // SAFETY: `N == 16`, so `icdf` has 16 readable `u16`s.
        unsafe {
            (
                vld1q_u16(icdf.as_ptr()),
                vld1q_u16(icdf.as_ptr().add(LANES)),
            )
        }
    } else {
        let lanes: [u16; 2 * LANES] = padded(icdf);
        // SAFETY: `lanes` has 16 readable `u16`s.
        unsafe {
            (
                vld1q_u16(lanes.as_ptr()),
                vld1q_u16(lanes.as_ptr().add(LANES)),
            )
        }
    };
    let b0 = bounds8(p0, 0, symbols, range);
    let b1 = bounds8(p1, LANES as u16, symbols, range);
    let val = vdupq_n_u16((window >> 48) as u16);
    let bf0 = vmvnq_u16(vcgeq_u16(val, b0));
    let bf1 = vmvnq_u16(vcgeq_u16(val, b1));
    let symbol =
        usize::from(vaddvq_u16(vshrq_n_u16::<15>(bf0)) + vaddvq_u16(vshrq_n_u16::<15>(bf1)));
    if let Some(rate) = rate {
        let (a0, a1) = (adapt8(p0, bf0, rate), adapt8(p1, bf1, rate));
        if N == 2 * LANES {
            // SAFETY: `N == 16`, so `icdf` has 16 writable `u16`s.
            unsafe {
                vst1q_u16(icdf.as_mut_ptr(), a0);
                vst1q_u16(icdf.as_mut_ptr().add(LANES), a1);
            }
        } else {
            let mut lanes = [0u16; 2 * LANES];
            // SAFETY: `lanes` has 16 writable `u16`s.
            unsafe {
                vst1q_u16(lanes.as_mut_ptr(), a0);
                vst1q_u16(lanes.as_mut_ptr().add(LANES), a1);
            }
            icdf.copy_from_slice(&lanes[..N]);
        }
    }
    let u0 = vreinterpretq_u64_u16(b0);
    let u1 = vreinterpretq_u64_u16(b1);
    let bounds64 = (
        vgetq_lane_u64::<0>(u0),
        vgetq_lane_u64::<1>(u0),
        vgetq_lane_u64::<0>(u1),
        vgetq_lane_u64::<1>(u1),
    );
    let (lower, upper) = interval_16(range, bounds64, symbol);
    normalize(symbol, window, lower, upper - lower)
}

#[inline]
pub(super) fn decode<const N: usize>(
    icdf: &mut [u16; N],
    symbols: usize,
    window: u64,
    range: u32,
    rate: Option<u32>,
) -> Av1SymbolStep {
    // SAFETY: NEON is baseline on aarch64; the parent restricts `N` to `4..=16`.
    unsafe {
        if N <= LANES {
            eight(icdf, symbols, window, range, rate)
        } else {
            sixteen(icdf, symbols, window, range, rate)
        }
    }
}
