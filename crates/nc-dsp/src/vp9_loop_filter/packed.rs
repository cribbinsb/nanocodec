//! Eight-line VP9 edge-filter schedule over the shared packed-byte lanes
//! (`crate::lpf_bytes`): masks, flat decisions and the 8/16 smoothing.
#![allow(clippy::inline_always)]
use crate::lpf_bytes::{
    Bytes, INNER, LANES, Words, difference, load, narrow, select, store, within,
};
use crate::{Vp9FilterLimits, Vp9FilterWidth};
#[cfg(target_arch = "x86_64")]
pub(super) mod avx2;
#[cfg(target_arch = "aarch64")]
pub(super) mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) mod simd128;

#[inline(always)]
fn masks<B: Bytes>(p: &[B; LANES], q: &[B; LANES], limits: Vp9FilterLimits) -> (B, B, B) {
    let mut adjacent = difference(p[1], p[0]).max(difference(q[1], q[0]));
    let smooth = within(adjacent, limits.variance as u8);
    let mut flat = adjacent;
    for i in 2..INNER {
        adjacent = adjacent
            .max(difference(p[i], p[i - 1]))
            .max(difference(q[i], q[i - 1]));
        flat = flat.max(difference(p[i], p[0])).max(difference(q[i], q[0]));
    }
    let across = difference(p[0], q[0]);
    let boundary = across
        .add_sat(across)
        .add_sat(difference(p[1], q[1]).shr::<1>());
    // Saturating the weighted boundary difference is exact for thresholds <255.
    let mask = within(adjacent, limits.inside as u8).and(within(boundary, limits.boundary as u8));
    (mask, smooth.xor(B::splat(255)), within(flat, 1).and(mask))
}

/// Sliding box sum plus one extra centre sample; positive weights sum to
/// 8 or 16, so every intermediate, including rounding, is <=4088 in i16.
#[inline(always)]
fn wide<B: Bytes, const N: usize, const SHIFT: i32>(input: &[B; N], output: &mut [B; N], mask: B) {
    let taps = input.map(Bytes::widen);
    let radius = N / 2 - 1;
    let mut sum = B::W::splat(1 << (SHIFT - 1));
    // First written position is 1, with replicated endpoints.
    for offset in 0..=(2 * radius) {
        let at = (1 + offset).saturating_sub(radius).min(N - 1);
        sum = sum.add(taps[at]);
    }
    for i in 1..N - 1 {
        output[i] = select(output[i], B::narrow(sum.add(taps[i]).shr::<SHIFT>()), mask);
        let leaving = i.saturating_sub(radius);
        let entering = (i + radius + 1).min(N - 1);
        sum = sum.sub(taps[leaving]).add(taps[entering]);
    }
}

#[inline(always)]
unsafe fn group<B: Bytes, const WIDTH: usize>(
    data: &mut [u8],
    edge: usize,
    step: usize,
    along: usize,
    limits: Vp9FilterLimits,
) {
    // SAFETY: the checked caller provides all eight lines and filter taps.
    let inner = unsafe { load::<B>(data.as_ptr(), edge, step, along, INNER) };
    let mut p = [B::splat(0); LANES];
    let mut q = p;
    for i in 0..INNER {
        p[i] = inner[INNER - 1 - i];
        q[i] = inner[INNER + i];
    }
    let (mask, hev, flat) = masks(&p, &q, limits);
    if !mask.any() {
        return;
    }
    let (old_p, old_q) = (p, q);
    narrow(&mut p, &mut q, mask, hev);
    let mut result = core::array::from_fn(|i| {
        if i < INNER {
            p[INNER - 1 - i]
        } else {
            q[i - INNER]
        }
    });
    let mut written = 2;
    if WIDTH != 4 && flat.any() {
        written = 3;
        wide::<B, 8, 3>(&inner, &mut result, flat);
        if WIDTH == 16 {
            // SAFETY: full-width footprint was checked; outer data is needed only now.
            let (left, right) = unsafe {
                (
                    load::<B>(data.as_ptr(), edge, step, along, LANES),
                    load::<B>(data.as_ptr(), edge, step, along, 0),
                )
            };
            let mut outer_flat = B::splat(0);
            for i in 0..INNER {
                outer_flat = outer_flat
                    .max(difference(left[i], old_p[0]))
                    .max(difference(right[i + INNER], old_q[0]));
            }
            let flat2 = within(outer_flat, 1).and(flat);
            if flat2.any() {
                let all =
                    core::array::from_fn(|i| if i < LANES { left[i] } else { right[i - LANES] });
                let mut out = all;
                out[INNER..INNER + LANES].copy_from_slice(&result);
                wide::<B, 16, 4>(&all, &mut out, flat2);
                // SAFETY: same validated full-width tiles as the loads above.
                unsafe {
                    store::<B>(
                        data.as_mut_ptr(),
                        edge,
                        step,
                        along,
                        LANES,
                        out[..LANES].try_into().expect("tile"),
                        1..LANES,
                    );
                    store::<B>(
                        data.as_mut_ptr(),
                        edge,
                        step,
                        along,
                        0,
                        out[LANES..].try_into().expect("tile"),
                        0..LANES - 1,
                    );
                }
                return;
            }
        }
    }
    // SAFETY: same validated inner tile as the initial load.
    unsafe {
        store::<B>(
            data.as_mut_ptr(),
            edge,
            step,
            along,
            INNER,
            result,
            INNER - written..INNER + written,
        );
    }
}

// Separate width-specialized entry points keep the wide-filter register saves
// and stack frame off the common filter4 path.
#[inline(never)]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn run<B: Bytes, const WIDTH: usize>(
    data: &mut [u8],
    edge: usize,
    step: usize,
    along: usize,
    lines: usize,
    width: Vp9FilterWidth,
    limits: Vp9FilterLimits,
) {
    // Unusual thresholds/geometry retain the oracle's complete public semantics.
    let reach = if width == Vp9FilterWidth::Sixteen {
        8
    } else {
        4
    };
    let independent = (along == 1 && step >= lines) || (step == 1 && along >= 2 * reach);
    if limits.boundary >= 255 || !independent {
        crate::vp9_deblock::vp9_filter_edge(data, edge, step, along, lines, width, limits);
        return;
    }
    let full = lines / LANES * LANES;
    for first in (0..full).step_by(LANES) {
        // SAFETY: dispatch validated every group and the ISA entry point enables B.
        unsafe {
            group::<B, WIDTH>(data, edge + first * along, step, along, limits);
        }
    }
    if full != lines {
        crate::vp9_deblock::vp9_filter_edge(
            data,
            edge + full * along,
            step,
            along,
            lines - full,
            width,
            limits,
        );
    }
}
