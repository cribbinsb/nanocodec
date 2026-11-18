//! High-bit-depth (`u16`, 10- and 12-bit) VP9 deblocking SIMD kernels across
//! AVX2, AVX-512, NEON, and SIMD128.
//!
//! Shares the 16-bit lane primitives (`narrow_lanes`, `smooth_lanes`,
//! `load_h`/`store_h`, `load_v_tile`/`store_v_tile`) with [`super::av1_high`]
//! while keeping VP9's normative 4-tap `filter_mask`, 8-sample outer flatness
//! (`p7..p4`, `q4..q7`), and 15-tap smoothing (`N = 16`, centre weighted
//! twice: `16 * 4095 + 8 = 65528 <= u16::MAX` at 12 bits). Vertical edges
//! transpose 8-word tiles in 128-bit groups ([`I16x8Tiles`]: 16 lines per step
//! on AVX2, 32 on AVX-512, 8 on NEON and SIMD128).
//!
//! Attribution: transposed vertical-edge layout and flat-mask gating follow
//! libvpx (`vpx_dsp/x86/highbd_loopfilter_sse2.c`, `vpx_highbd_lpf_vertical_*`,
//! BSD-3-Clause; see `THIRD_PARTY.md`).

#![allow(clippy::inline_always)]
#![cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code, unused_imports)
)]

use nc_common::BitDepth;

use super::av1_high::{
    abs_diff, le_mask, load_h, load_v_tile, narrow_lanes, smooth_lanes, store_h, store_v_tile,
};
use crate::high_depth::Vp9Filter16Fn;
use crate::simd::{I16FilterLanes, I16x8Shuffles, I16x8Tiles};
use crate::vp9_deblock::{filter_lines_at, vp9_filter_edge_at};
use crate::{Isa, Vp9FilterLimits, Vp9FilterWidth};

const TILE: usize = 8;
const INNER_REACH: usize = 4;
const WIDE_REACH: usize = 8;
const MAX_SIMD_BITS: u32 = 12;

pub(crate) fn vp9_filter(isa: Isa) -> Vp9Filter16Fn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => x86_avx2::filter,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => x86_avx512::filter,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon_high::filter,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128_high::filter,
        _ => filter_lines_at::<u16>,
    }
}

/// [`narrow_lanes`] at a runtime depth shift (`depth - 8`): VP9 carries the
/// shift as a value, while AV1 specialises its whole kernel per depth.
#[inline(always)]
fn narrow_at<V: I16FilterLanes>(taps: (V, V, V, V), mask: V, hev: V, shift: u32) -> [V; 4] {
    // `depth.bits() <= MAX_SIMD_BITS` (12) keeps the shift at most 4.
    debug_assert!(shift <= MAX_SIMD_BITS - 8);
    match shift {
        0 => narrow_lanes::<V, 0>(taps, mask, hev),
        1 => narrow_lanes::<V, 1>(taps, mask, hev),
        2 => narrow_lanes::<V, 2>(taps, mask, hev),
        3 => narrow_lanes::<V, 3>(taps, mask, hev),
        _ => narrow_lanes::<V, 4>(taps, mask, hev),
    }
}

#[inline(always)]
fn vp9_masks<V: I16FilterLanes>(
    p: &[V; INNER_REACH],
    q: &[V; INNER_REACH],
    width: Vp9FilterWidth,
    scaled: Vp9FilterLimits,
    shift: u32,
    active: V,
) -> (V, V, V) {
    let limit_p1 = V::splat(scaled.inside as i16 + 1);
    let blimit_p1 = V::splat(scaled.boundary as i16 + 1);
    let thresh = V::splat(scaled.variance as i16);
    let flat_p1 = V::splat((1_i16 << shift) + 1);

    let mut adj = abs_diff(p[1], p[0]).max(abs_diff(q[1], q[0]));
    let hev = adj.gt_mask(thresh);
    let mut flat_spread = adj;
    for i in 2..INNER_REACH {
        adj = adj
            .max(abs_diff(p[i], p[i - 1]))
            .max(abs_diff(q[i], q[i - 1]));
        flat_spread = flat_spread
            .max(abs_diff(p[i], p[0]))
            .max(abs_diff(q[i], q[0]));
    }
    let d0 = abs_diff(p[0], q[0]);
    let across = d0.add(d0).add(abs_diff(p[1], q[1]).sra(1));
    let mask = le_mask(adj, limit_p1)
        .and(le_mask(across, blimit_p1))
        .and(active);
    let flat = if width == Vp9FilterWidth::Four {
        V::splat(0)
    } else {
        le_mask(flat_spread, flat_p1).and(mask)
    };
    (mask, hev, flat)
}

#[inline(always)]
fn vp9_flat2_mask<V: I16FilterLanes>(
    (p7, p6, p5, p4): (V, V, V, V),
    (p0, q0): (V, V),
    (q4, q5, q6, q7): (V, V, V, V),
    flat: V,
    shift: u32,
) -> V {
    let flat_p1 = V::splat((1_i16 << shift) + 1);
    let spread = abs_diff(p4, p0)
        .max(abs_diff(q4, q0))
        .max(abs_diff(p5, p0))
        .max(abs_diff(q5, q0))
        .max(abs_diff(p6, p0))
        .max(abs_diff(q6, q0))
        .max(abs_diff(p7, p0))
        .max(abs_diff(q7, q0));
    le_mask(spread, flat_p1).and(flat)
}

#[inline(always)]
fn transpose_tiles<V: I16x8Tiles>(mut rows: [V; TILE]) -> [V; TILE] {
    for _ in 0..3 {
        let mut next = rows;
        for k in 0..TILE / 2 {
            next[2 * k] = rows[k].zip_tiles_lo(rows[k + TILE / 2]);
            next[2 * k + 1] = rows[k].zip_tiles_hi(rows[k + TILE / 2]);
        }
        rows = next;
    }
    rows
}

#[inline(always)]
unsafe fn load_v_wide<V: I16x8Tiles>(data: *const u16, origin: usize, along: usize) -> [V; TILE] {
    let mut rows = [V::splat(0); TILE];
    for (k, row) in rows.iter_mut().enumerate() {
        *row = if V::LANES == 8 {
            // SAFETY: caller checked 8 `u16`s at `origin + k * along`.
            V::load(unsafe {
                core::slice::from_raw_parts(data.add(origin + k * along).cast::<i16>(), TILE)
            })
        } else if V::LANES == 16 {
            // SAFETY: caller checked two 8-line tiles separated by `8 * along`.
            unsafe {
                let lo =
                    core::slice::from_raw_parts(data.add(origin + k * along).cast::<i16>(), TILE);
                let hi = core::slice::from_raw_parts(
                    data.add(origin + (TILE + k) * along).cast::<i16>(),
                    TILE,
                );
                V::load_halves(lo, hi)
            }
        } else {
            // SAFETY: caller checked four 8-line tiles separated by `8 * along`.
            unsafe {
                let q = core::array::from_fn(|g| {
                    core::slice::from_raw_parts(
                        data.add(origin + (g * TILE + k) * along).cast::<i16>(),
                        TILE,
                    )
                });
                V::load_quarters(q)
            }
        };
    }
    transpose_tiles(rows)
}

#[inline(always)]
unsafe fn store_v_wide<V: I16x8Tiles>(
    positions: [V; TILE],
    data: *mut u16,
    origin: usize,
    along: usize,
) {
    let rows = transpose_tiles(positions);
    for (k, row) in rows.into_iter().enumerate() {
        if V::LANES == 8 {
            // SAFETY: caller checked 8 writable `u16`s at `origin + k * along`.
            row.store(unsafe {
                core::slice::from_raw_parts_mut(data.add(origin + k * along).cast::<i16>(), TILE)
            });
        } else if V::LANES == 16 {
            // SAFETY: `along >= 8`, so the two 8-word row slices are disjoint.
            unsafe {
                let lo = core::slice::from_raw_parts_mut(
                    data.add(origin + k * along).cast::<i16>(),
                    TILE,
                );
                let hi = core::slice::from_raw_parts_mut(
                    data.add(origin + (TILE + k) * along).cast::<i16>(),
                    TILE,
                );
                row.store_halves(lo, hi);
            }
        } else {
            // SAFETY: `along >= 8`, so all four 8-word row slices are disjoint.
            unsafe {
                let [r0, r1, r2, r3] = [0, 1, 2, 3].map(|g| {
                    core::slice::from_raw_parts_mut(
                        data.add(origin + (g * TILE + k) * along).cast::<i16>(),
                        TILE,
                    )
                });
                row.store_quarters([r0, r1, r2, r3]);
            }
        }
    }
}

#[inline(always)]
unsafe fn h_group<V: I16FilterLanes, const LINES: usize>(
    data: &mut [u16],
    edge: usize,
    across: usize,
    width: Vp9FilterWidth,
    scaled: Vp9FilterLimits,
    shift: u32,
) {
    let active = if LINES == 4 {
        V::load_halves(&[-1; 4], &[0; 4])
    } else {
        V::splat(-1)
    };
    let ptr = data.as_ptr();
    let mptr = data.as_mut_ptr();
    // SAFETY: caller validated the edge reach across all lines in this group.
    unsafe {
        let p3 = load_h::<V, LINES>(ptr, edge, across, -4);
        let p2 = load_h::<V, LINES>(ptr, edge, across, -3);
        let p1 = load_h::<V, LINES>(ptr, edge, across, -2);
        let p0 = load_h::<V, LINES>(ptr, edge, across, -1);
        let q0 = load_h::<V, LINES>(ptr, edge, across, 0);
        let q1 = load_h::<V, LINES>(ptr, edge, across, 1);
        let q2 = load_h::<V, LINES>(ptr, edge, across, 2);
        let q3 = load_h::<V, LINES>(ptr, edge, across, 3);
        let (mask, hev, flat) = vp9_masks(
            &[p0, p1, p2, p3],
            &[q0, q1, q2, q3],
            width,
            scaled,
            shift,
            active,
        );
        if !mask.any_nonzero() {
            return;
        }
        let [np1, np0, nq0, nq1] = narrow_at((p1, p0, q0, q1), mask, hev, shift);
        if !flat.any_nonzero() {
            store_h::<V, LINES>(np1, mptr, edge, across, -2);
            store_h::<V, LINES>(np0, mptr, edge, across, -1);
            store_h::<V, LINES>(nq0, mptr, edge, across, 0);
            store_h::<V, LINES>(nq1, mptr, edge, across, 1);
            return;
        }
        let mut out8 = [p3, p2, np1, np0, nq0, nq1, q2, q3];
        smooth_lanes::<V, 8, 3, 0, 3>(&[p3, p2, p1, p0, q0, q1, q2, q3], &mut out8, flat);
        if width == Vp9FilterWidth::Sixteen {
            let p7 = load_h::<V, LINES>(ptr, edge, across, -8);
            let p6 = load_h::<V, LINES>(ptr, edge, across, -7);
            let p5 = load_h::<V, LINES>(ptr, edge, across, -6);
            let p4 = load_h::<V, LINES>(ptr, edge, across, -5);
            let q4 = load_h::<V, LINES>(ptr, edge, across, 4);
            let q5 = load_h::<V, LINES>(ptr, edge, across, 5);
            let q6 = load_h::<V, LINES>(ptr, edge, across, 6);
            let q7 = load_h::<V, LINES>(ptr, edge, across, 7);
            let flat2 = vp9_flat2_mask((p7, p6, p5, p4), (p0, q0), (q4, q5, q6, q7), flat, shift);
            if flat2.any_nonzero() {
                let line16 = [
                    p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7,
                ];
                let mut out16 = [
                    p7, p6, p5, p4, out8[0], out8[1], out8[2], out8[3], out8[4], out8[5], out8[6],
                    out8[7], q4, q5, q6, q7,
                ];
                smooth_lanes::<V, 16, 7, 0, 4>(&line16, &mut out16, flat2);
                for (k, &v) in out16[1..15].iter().enumerate() {
                    store_h::<V, LINES>(v, mptr, edge, across, k as isize - 7);
                }
                return;
            }
        }
        for (k, &v) in out8[1..7].iter().enumerate() {
            store_h::<V, LINES>(v, mptr, edge, across, k as isize - 3);
        }
    }
}

#[inline(always)]
unsafe fn v_wide_group<V: I16x8Tiles>(
    data: &mut [u16],
    edge: usize,
    along: usize,
    width: Vp9FilterWidth,
    scaled: Vp9FilterLimits,
    shift: u32,
) {
    // SAFETY: caller validated `V::LANES` vertical rows and `width` reach.
    let inner = unsafe { load_v_wide::<V>(data.as_ptr(), edge - INNER_REACH, along) };
    let p = [inner[3], inner[2], inner[1], inner[0]];
    let q = [inner[4], inner[5], inner[6], inner[7]];
    let (mask, hev, flat) = vp9_masks(&p, &q, width, scaled, shift, V::splat(-1));
    if !mask.any_nonzero() {
        return;
    }
    let [np1, np0, nq0, nq1] = narrow_at((p[1], p[0], q[0], q[1]), mask, hev, shift);
    let mut result = [inner[0], inner[1], np1, np0, nq0, nq1, inner[6], inner[7]];
    if flat.any_nonzero() {
        smooth_lanes::<V, 8, 3, 0, 3>(&inner, &mut result, flat);
        if width == Vp9FilterWidth::Sixteen {
            // SAFETY: caller validated `WIDE_REACH` each side of `edge`.
            let (mut left, mut right) = unsafe {
                (
                    load_v_wide::<V>(data.as_ptr(), edge - WIDE_REACH, along),
                    load_v_wide::<V>(data.as_ptr(), edge, along),
                )
            };
            let flat2 = vp9_flat2_mask(
                (left[0], left[1], left[2], left[3]),
                (p[0], q[0]),
                (right[4], right[5], right[6], right[7]),
                flat,
                shift,
            );
            if flat2.any_nonzero() {
                let mut line16 = [V::splat(0); 16];
                line16[..8].copy_from_slice(&left);
                line16[8..16].copy_from_slice(&right);
                let mut out16 = line16;
                out16[4..12].copy_from_slice(&result);
                smooth_lanes::<V, 16, 7, 0, 4>(&line16, &mut out16, flat2);
                left[1..8].copy_from_slice(&out16[1..8]);
                right[..7].copy_from_slice(&out16[8..15]);
                // SAFETY: same validated 16-sample footprint as the loads above.
                unsafe {
                    store_v_wide::<V>(left, data.as_mut_ptr(), edge - WIDE_REACH, along);
                    store_v_wide::<V>(right, data.as_mut_ptr(), edge, along);
                }
                return;
            }
        }
    }
    // SAFETY: same validated `p3..q3` footprint as the load above.
    unsafe { store_v_wide::<V>(result, data.as_mut_ptr(), edge - INNER_REACH, along) };
}

#[inline(always)]
unsafe fn v_short_four<V: I16x8Shuffles + I16FilterLanes>(
    data: &mut [u16],
    edge: usize,
    along: usize,
    width: Vp9FilterWidth,
    scaled: Vp9FilterLimits,
    shift: u32,
) {
    let active = V::load_halves(&[-1; 4], &[0; 4]);
    // SAFETY: caller validated 4 vertical rows and `width` reach.
    let inner = unsafe { load_v_tile::<V, 4>(data.as_ptr(), edge - INNER_REACH, along) };
    let p = [inner[3], inner[2], inner[1], inner[0]];
    let q = [inner[4], inner[5], inner[6], inner[7]];
    let (mask, hev, flat) = vp9_masks(&p, &q, width, scaled, shift, active);
    if !mask.any_nonzero() {
        return;
    }
    let [np1, np0, nq0, nq1] = narrow_at((p[1], p[0], q[0], q[1]), mask, hev, shift);
    let mut result = [inner[0], inner[1], np1, np0, nq0, nq1, inner[6], inner[7]];
    if flat.any_nonzero() {
        smooth_lanes::<V, 8, 3, 0, 3>(&inner, &mut result, flat);
        if width == Vp9FilterWidth::Sixteen {
            // SAFETY: caller validated `WIDE_REACH` each side of `edge`.
            let (mut left, mut right) = unsafe {
                (
                    load_v_tile::<V, 4>(data.as_ptr(), edge - WIDE_REACH, along),
                    load_v_tile::<V, 4>(data.as_ptr(), edge, along),
                )
            };
            let flat2 = vp9_flat2_mask(
                (left[0], left[1], left[2], left[3]),
                (p[0], q[0]),
                (right[4], right[5], right[6], right[7]),
                flat,
                shift,
            );
            if flat2.any_nonzero() {
                let mut line16 = [V::splat(0); 16];
                line16[..8].copy_from_slice(&left);
                line16[8..16].copy_from_slice(&right);
                let mut out16 = line16;
                out16[4..12].copy_from_slice(&result);
                smooth_lanes::<V, 16, 7, 0, 4>(&line16, &mut out16, flat2);
                left[1..8].copy_from_slice(&out16[1..8]);
                right[..7].copy_from_slice(&out16[8..15]);
                // SAFETY: same validated 4-row 16-sample footprint.
                unsafe {
                    store_v_tile::<V, 4>(left, data.as_mut_ptr(), edge - WIDE_REACH, along);
                    store_v_tile::<V, 4>(right, data.as_mut_ptr(), edge, along);
                }
                return;
            }
        }
    }
    // SAFETY: same validated 4-row `p3..q3` footprint.
    unsafe { store_v_tile::<V, 4>(result, data.as_mut_ptr(), edge - INNER_REACH, along) };
}

#[inline(always)]
unsafe fn vp9_filter_lanes<V8: I16x8Shuffles + I16x8Tiles, VWide: I16x8Tiles>(
    data: &mut [u16],
    (edge, step, along): (usize, usize, usize),
    lines: usize,
    width: Vp9FilterWidth,
    limits: Vp9FilterLimits,
    depth: BitDepth,
) {
    debug_assert_eq!(V8::LANES, 8);
    if lines == 0 {
        return;
    }
    let reach = if width == Vp9FilterWidth::Sixteen {
        WIDE_REACH
    } else {
        INNER_REACH
    };
    let independent = (along == 1 && step >= lines) || (step == 1 && along >= 2 * reach);
    let scaled = limits.at(depth);
    let thresholds_ok = scaled.inside >= 0
        && scaled.boundary >= 0
        && scaled.variance >= 0
        && scaled.boundary < i32::from(i16::MAX);
    if depth.bits() > MAX_SIMD_BITS
        || !independent
        || !thresholds_ok
        || !crate::vp9_deblock::footprint_fits(data.len(), (edge, step, along), lines, width)
    {
        vp9_filter_edge_at(data, (edge, step, along), lines, width, limits, depth);
        return;
    }
    let shift = depth.above_eight();
    let mut done = 0;
    if along == 1 {
        while done + VWide::LANES <= lines {
            // SAFETY: footprint covers `edge + done .. edge + done + VWide::LANES`.
            unsafe { h_group::<VWide, 0>(data, edge + done, step, width, scaled, shift) };
            done += VWide::LANES;
        }
        while V8::LANES < VWide::LANES && done + TILE <= lines {
            // SAFETY: footprint covers `edge + done .. edge + done + 8`.
            unsafe { h_group::<V8, 0>(data, edge + done, step, width, scaled, shift) };
            done += TILE;
        }
        if done + 4 <= lines {
            // SAFETY: footprint covers `edge + done .. edge + done + 4`.
            unsafe { h_group::<V8, 4>(data, edge + done, step, width, scaled, shift) };
            done += 4;
        }
    } else {
        while done + VWide::LANES <= lines {
            // SAFETY: footprint covers `VWide::LANES` vertical rows from `done`.
            unsafe {
                v_wide_group::<VWide>(data, edge + done * along, along, width, scaled, shift);
            }
            done += VWide::LANES;
        }
        while V8::LANES < VWide::LANES && done + TILE <= lines {
            // SAFETY: footprint covers 8 vertical rows from `done`.
            unsafe {
                v_wide_group::<V8>(data, edge + done * along, along, width, scaled, shift);
            }
            done += TILE;
        }
        if done + 4 <= lines {
            // SAFETY: footprint covers 4 vertical rows from `done`.
            unsafe { v_short_four::<V8>(data, edge + done * along, along, width, scaled, shift) };
            done += 4;
        }
    }
    if done < lines {
        vp9_filter_edge_at(
            data,
            (edge + done * along, step, along),
            lines - done,
            width,
            limits,
            depth,
        );
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! vp9_deblock_isa {
    ($mod_name:ident, $(#[$feature:meta])*, $v8:ty, $vwide:ty) => {
        mod $mod_name {
            use nc_common::BitDepth;

            use super::vp9_filter_lanes;
            use crate::{Vp9FilterLimits, Vp9FilterWidth};

            $(#[$feature])*
            pub(super) unsafe fn filter(
                data: &mut [u16],
                at: (usize, usize, usize),
                lines: usize,
                width: Vp9FilterWidth,
                limits: Vp9FilterLimits,
                depth: BitDepth,
            ) {
                // SAFETY: wrapper enables target_feature; caller checked footprint.
                unsafe { vp9_filter_lanes::<$v8, $vwide>(data, at, lines, width, limits, depth) }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
vp9_deblock_isa!(
    x86_avx2,
    #[target_feature(enable = "avx2")],
    crate::simd::avx2::I16x8,
    crate::simd::avx2::I16x16
);

#[cfg(target_arch = "x86_64")]
vp9_deblock_isa!(
    x86_avx512,
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")],
    crate::simd::avx2::I16x8,
    crate::simd::avx512_i16::I16x32
);

#[cfg(target_arch = "aarch64")]
vp9_deblock_isa!(
    neon_high,
    #[target_feature(enable = "neon")],
    crate::simd::neon::I16x8,
    crate::simd::neon::I16x8
);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
vp9_deblock_isa!(
    simd128_high,
    #[target_feature(enable = "simd128")],
    crate::simd::simd128::I16x8,
    crate::simd::simd128::I16x8
);
