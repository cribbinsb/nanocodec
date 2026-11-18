//! High-bit-depth (`u16`, 10- and 12-bit) AV1 deblocking SIMD kernels across
//! AVX2, AVX-512, and NEON.
//!
//! Decisions (`filterMask`, `hevMask`, `flat`, `flat2`), the 4-tap narrow
//! filter, and the 6/8/14-tap wide smoothing filters run in 16-bit lanes
//! ([`I16FilterLanes`]): horizontal edges process `VWide::LANES` contiguous
//! columns per step (32 on AVX-512, 16 on AVX2, 8 on NEON) loading only the
//! active reach; vertical edges transpose 8x8 `u16` tiles in registers
//! ([`I16x8Shuffles`]). In 14-tap smoothing at 12 bits, `16 * 4095 + 8 = 65528`
//! fits unsigned `u16`, so wrapping `add`/`sub` followed by logical `srl(4)`
//! is exact without widening to 32 bits.
//!
//! Attribution: transposed 8x8 tile organization and flat-mask short-circuiting
//! follow dav1d 1.5.4 (`src/loopfilter_tmpl.c`, `src/x86/loopfilter16_avx2.asm`,
//! `src/x86/loopfilter16_avx512.asm`, `src/arm/64/loopfilter16.S`,
//! BSD-2-Clause; see `THIRD_PARTY.md`).

#![allow(clippy::inline_always)]
#![cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code, unused_imports)
)]

use nc_common::BitDepth;

use super::av1::{Av1EdgeFilter, Av1FilterLength, filter_edge_at};
use super::runs::{EdgeRun, Plane, ValidatedRuns, av1_scalar_at};
use super::samples::transpose;
use crate::Isa;
use crate::simd::{I16FilterLanes, I16Lanes, I16x8Shuffles};

/// Sparse `u16` AV1 loop-filter run dispatcher for one ISA.
pub(crate) type Av1Runs16Fn =
    unsafe fn(data: &mut [u16], plane: Plane, runs: &[EdgeRun<Av1EdgeFilter>], depth: BitDepth);

pub(crate) fn av1_runs(isa: Isa) -> Av1Runs16Fn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => x86_avx2::runs,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => x86_avx512::runs,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon_high::runs,
        _ => av1_scalar_at::<u16>,
    }
}

#[inline(always)]
fn abs_diff<V: I16Lanes>(a: V, b: V) -> V {
    a.sub(b).abs()
}

#[inline(always)]
fn le_mask<V: I16FilterLanes>(x: V, limit_plus_one: V) -> V {
    limit_plus_one.gt_mask(x)
}

/// Compute `(mask, hev, flat)` for `TAPS` (`2`, `3`, or `4`) decision samples
/// per side at `shift = BitDepth - 8` (`2` or `4`).
#[inline(always)]
fn masks<V: I16FilterLanes, const TAPS: usize>(
    p: &[V; 4],
    q: &[V; 4],
    filter: Av1EdgeFilter,
    shift: u32,
    active: V,
) -> (V, V, V) {
    let limit_p1 = (i16::from(filter.limit) << shift) + 1;
    let blimit_p1 = (i16::from(filter.blimit) << shift) + 1;
    let thresh = i16::from(filter.thresh) << shift;
    let flat_p1 = (1_i16 << shift) + 1;

    let mut adj = abs_diff(p[1], p[0]).max(abs_diff(q[1], q[0]));
    let hev = adj.gt_mask(V::splat(thresh));
    let mut flat_spread = adj;
    for i in 2..TAPS {
        adj = adj
            .max(abs_diff(p[i], p[i - 1]))
            .max(abs_diff(q[i], q[i - 1]));
        flat_spread = flat_spread
            .max(abs_diff(p[i], p[0]))
            .max(abs_diff(q[i], q[0]));
    }
    let d0 = abs_diff(p[0], q[0]);
    let across = d0.add(d0).add(abs_diff(p[1], q[1]).sra(1));
    let mask = le_mask(adj, V::splat(limit_p1))
        .and(le_mask(across, V::splat(blimit_p1)))
        .and(active);
    let flat = le_mask(flat_spread, V::splat(flat_p1)).and(mask);
    (mask, hev, flat)
}

/// Narrow 4-tap filter (`[np1, np0, nq0, nq1]`) in signed range `[-offset, offset - 1]`.
#[inline(always)]
fn narrow_lanes<V: I16FilterLanes>(
    (p1, p0, q0, q1): (V, V, V, V),
    mask: V,
    hev: V,
    shift: u32,
) -> [V; 4] {
    let off = 128_i16 << shift;
    let (lo, hi) = (V::splat(-off), V::splat(off - 1));
    let offset = V::splat(off);
    let (ps1, ps0, qs0, qs1) = (
        p1.sub(offset),
        p0.sub(offset),
        q0.sub(offset),
        q1.sub(offset),
    );
    let outer = hev.and(ps1.sub(qs1).max(lo).min(hi));
    let diff0 = qs0.sub(ps0);
    let base = outer.add(diff0.add(diff0).add(diff0)).max(lo).min(hi);
    let f1 = base.add(V::splat(4)).max(lo).min(hi).sra(3);
    let f2 = base.add(V::splat(3)).max(lo).min(hi).sra(3);
    let nq0 = mask.blend(qs0.sub(f1).max(lo).min(hi).add(offset), q0);
    let np0 = mask.blend(ps0.add(f2).max(lo).min(hi).add(offset), p0);
    let tap = f1.add(V::splat(1)).sra(1);
    let mask_outer = hev.blend(V::splat(0), mask);
    let nq1 = mask_outer.blend(qs1.sub(tap).max(lo).min(hi).add(offset), q1);
    let np1 = mask_outer.blend(ps1.add(tap).max(lo).min(hi).add(offset), p1);
    [np1, np0, nq0, nq1]
}

/// Sliding-box wide filter (`6`, `8`, or `14` taps) using modular 16-bit sums
/// and unsigned `srl(SHIFT)` (`<= 65528` at 12-bit 14-tap).
#[inline(always)]
fn smooth_lanes<
    V: I16FilterLanes,
    const N: usize,
    const OUTER: usize,
    const CENTRE: usize,
    const SHIFT: u32,
>(
    input: &[V; N],
    output: &mut [V; N],
    mask: V,
) {
    let mut sum = V::splat(1_i16 << (SHIFT - 1));
    for offset in 0..=(2 * OUTER) {
        sum = sum.add(input[(1 + offset).saturating_sub(OUTER).min(N - 1)]);
    }
    for i in 1..N - 1 {
        let mut total = sum;
        for &tap in &input[i - CENTRE..=i + CENTRE] {
            total = total.add(tap);
        }
        output[i] = mask.blend(total.srl(SHIFT), output[i]);
        sum = sum
            .sub(input[i.saturating_sub(OUTER)])
            .add(input[(i + OUTER + 1).min(N - 1)]);
    }
}

#[inline(always)]
fn flat2_mask<V: I16FilterLanes>(
    (p6, p5, p4, p0): (V, V, V, V),
    (q0, q4, q5, q6): (V, V, V, V),
    flat: V,
    shift: u32,
) -> V {
    let flat_p1 = V::splat((1_i16 << shift) + 1);
    let spread = abs_diff(p4, p0)
        .max(abs_diff(q4, q0))
        .max(abs_diff(p5, p0))
        .max(abs_diff(q5, q0))
        .max(abs_diff(p6, p0))
        .max(abs_diff(q6, q0));
    le_mask(spread, flat_p1).and(flat)
}

// ----------------------------------------- horizontal edges (along == 1) ----

#[inline(always)]
unsafe fn load_h<V: I16FilterLanes, const LINES: usize>(
    data: *const u16,
    edge: usize,
    across: usize,
    pos: isize,
) -> V {
    // SAFETY: caller validated `pos` rows from `edge` lie inside `data`.
    let ptr = unsafe { data.add(edge.wrapping_add_signed(pos * across as isize)) };
    if LINES == 4 {
        // SAFETY: caller validated 4 readable `u16`s on an 8-lane vector.
        let four = unsafe { core::slice::from_raw_parts(ptr.cast::<i16>(), 4) };
        V::load_halves(four, &[0; 4])
    } else {
        // SAFETY: caller validated `V::LANES` readable `u16`s.
        V::load(unsafe { core::slice::from_raw_parts(ptr.cast::<i16>(), V::LANES) })
    }
}

#[inline(always)]
unsafe fn store_h<V: I16FilterLanes, const LINES: usize>(
    v: V,
    data: *mut u16,
    edge: usize,
    across: usize,
    pos: isize,
) {
    // SAFETY: caller validated `pos` rows from `edge` lie inside `data`.
    let ptr = unsafe { data.add(edge.wrapping_add_signed(pos * across as isize)) };
    if LINES == 4 {
        let mut scratch = [0i16; 4];
        // SAFETY: caller validated 4 writable `u16`s on an 8-lane vector.
        let four = unsafe { core::slice::from_raw_parts_mut(ptr.cast::<i16>(), 4) };
        v.store_halves(four, &mut scratch);
    } else {
        // SAFETY: caller validated `V::LANES` writable `u16`s.
        v.store(unsafe { core::slice::from_raw_parts_mut(ptr.cast::<i16>(), V::LANES) });
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn h_eight_or_fourteen<V: I16FilterLanes, const LINES: usize>(
    ptr: *const u16,
    mptr: *mut u16,
    edge: usize,
    across: usize,
    filter: Av1EdgeFilter,
    shift: u32,
    active: V,
    (p1, p0, q0, q1): (V, V, V, V),
) {
    // SAFETY: caller validated `filter.length.reach()` rows each side of `edge`.
    unsafe {
        let p2 = load_h::<V, LINES>(ptr, edge, across, -3);
        let p3 = load_h::<V, LINES>(ptr, edge, across, -4);
        let q2 = load_h::<V, LINES>(ptr, edge, across, 2);
        let q3 = load_h::<V, LINES>(ptr, edge, across, 3);
        let (mask, hev, flat) =
            masks::<V, 4>(&[p0, p1, p2, p3], &[q0, q1, q2, q3], filter, shift, active);
        if !mask.any_nonzero() {
            return;
        }
        let [np1, np0, nq0, nq1] = narrow_lanes((p1, p0, q0, q1), mask, hev, shift);
        if !flat.any_nonzero() {
            store_h::<V, LINES>(np1, mptr, edge, across, -2);
            store_h::<V, LINES>(np0, mptr, edge, across, -1);
            store_h::<V, LINES>(nq0, mptr, edge, across, 0);
            store_h::<V, LINES>(nq1, mptr, edge, across, 1);
            return;
        }
        let mut out8 = [p3, p2, np1, np0, nq0, nq1, q2, q3];
        smooth_lanes::<V, 8, 3, 0, 3>(&[p3, p2, p1, p0, q0, q1, q2, q3], &mut out8, flat);
        if filter.length == Av1FilterLength::Fourteen {
            let p4 = load_h::<V, LINES>(ptr, edge, across, -5);
            let p5 = load_h::<V, LINES>(ptr, edge, across, -6);
            let p6 = load_h::<V, LINES>(ptr, edge, across, -7);
            let q4 = load_h::<V, LINES>(ptr, edge, across, 4);
            let q5 = load_h::<V, LINES>(ptr, edge, across, 5);
            let q6 = load_h::<V, LINES>(ptr, edge, across, 6);
            let all = flat2_mask((p6, p5, p4, p0), (q0, q4, q5, q6), flat, shift);
            if all.any_nonzero() {
                let line14 = [p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6];
                let mut out14 = [
                    p6, p5, p4, out8[0], out8[1], out8[2], out8[3], out8[4], out8[5], out8[6],
                    out8[7], q4, q5, q6,
                ];
                smooth_lanes::<V, 14, 6, 1, 4>(&line14, &mut out14, all);
                for (k, &v) in out14[1..13].iter().enumerate() {
                    store_h::<V, LINES>(v, mptr, edge, across, k as isize - 6);
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
unsafe fn h_group<V: I16FilterLanes, const LINES: usize>(
    data: &mut [u16],
    edge: usize,
    across: usize,
    filter: Av1EdgeFilter,
    shift: u32,
) {
    let active = if LINES == 4 {
        V::load_halves(&[-1; 4], &[0; 4])
    } else {
        V::splat(-1)
    };
    let ptr = data.as_ptr();
    let mptr = data.as_mut_ptr();
    // SAFETY: caller validated `filter.length.reach()` rows each side of `edge`.
    unsafe {
        let p0 = load_h::<V, LINES>(ptr, edge, across, -1);
        let p1 = load_h::<V, LINES>(ptr, edge, across, -2);
        let q0 = load_h::<V, LINES>(ptr, edge, across, 0);
        let q1 = load_h::<V, LINES>(ptr, edge, across, 1);
        match filter.length {
            Av1FilterLength::Four => {
                let z = V::splat(0);
                let (mask, hev, _) =
                    masks::<V, 2>(&[p0, p1, z, z], &[q0, q1, z, z], filter, shift, active);
                if !mask.any_nonzero() {
                    return;
                }
                let [np1, np0, nq0, nq1] = narrow_lanes((p1, p0, q0, q1), mask, hev, shift);
                store_h::<V, LINES>(np1, mptr, edge, across, -2);
                store_h::<V, LINES>(np0, mptr, edge, across, -1);
                store_h::<V, LINES>(nq0, mptr, edge, across, 0);
                store_h::<V, LINES>(nq1, mptr, edge, across, 1);
            }
            Av1FilterLength::Six => {
                let p2 = load_h::<V, LINES>(ptr, edge, across, -3);
                let q2 = load_h::<V, LINES>(ptr, edge, across, 2);
                let z = V::splat(0);
                let (mask, hev, flat) =
                    masks::<V, 3>(&[p0, p1, p2, z], &[q0, q1, q2, z], filter, shift, active);
                if !mask.any_nonzero() {
                    return;
                }
                let [np1, np0, nq0, nq1] = narrow_lanes((p1, p0, q0, q1), mask, hev, shift);
                let mut out = [p2, np1, np0, nq0, nq1, q2];
                if flat.any_nonzero() {
                    let line = [p2, p1, p0, q0, q1, q2];
                    smooth_lanes::<V, 6, 2, 1, 3>(&line, &mut out, flat);
                }
                for (k, &v) in out[1..5].iter().enumerate() {
                    store_h::<V, LINES>(v, mptr, edge, across, k as isize - 2);
                }
            }
            Av1FilterLength::Eight | Av1FilterLength::Fourteen => {
                h_eight_or_fourteen::<V, LINES>(
                    ptr,
                    mptr,
                    edge,
                    across,
                    filter,
                    shift,
                    active,
                    (p1, p0, q0, q1),
                );
            }
        }
    }
}

// ------------------------------------------ vertical edges (across == 1) ----

#[inline(always)]
unsafe fn load_v_tile<V: I16x8Shuffles, const LINES: usize>(
    data: *const u16,
    origin: usize,
    along: usize,
) -> [V; 8] {
    let mut rows = [V::splat(0); 8];
    for (i, row) in rows.iter_mut().enumerate().take(LINES) {
        // SAFETY: caller checked 8 `u16`s at `origin + i * along` for `i < LINES`.
        *row = V::load(unsafe {
            core::slice::from_raw_parts(data.add(origin + i * along).cast::<i16>(), 8)
        });
    }
    transpose(rows)
}

#[inline(always)]
unsafe fn store_v_tile<V: I16x8Shuffles, const LINES: usize>(
    positions: [V; 8],
    data: *mut u16,
    origin: usize,
    along: usize,
) {
    let rows = transpose(positions);
    for (i, row) in rows.into_iter().enumerate().take(LINES) {
        // SAFETY: caller checked 8 writable `u16`s at `origin + i * along` for `i < LINES`.
        row.store(unsafe {
            core::slice::from_raw_parts_mut(data.add(origin + i * along).cast::<i16>(), 8)
        });
    }
}

#[inline(always)]
unsafe fn v_fourteen<V: I16x8Shuffles + I16FilterLanes, const LINES: usize>(
    data: &mut [u16],
    edge: usize,
    along: usize,
    flat: V,
    shift: u32,
    (old_p0, old_q0): (V, V),
    inner_out: &[V; 8],
) -> bool {
    // SAFETY: caller checked `edge >= 8` and `edge + (LINES - 1) * along + 7 < data.len()`.
    let (mut left, mut right) = unsafe {
        (
            load_v_tile::<V, LINES>(data.as_ptr(), edge - 8, along),
            load_v_tile::<V, LINES>(data.as_ptr(), edge, along),
        )
    };
    let all = flat2_mask(
        (left[1], left[2], left[3], old_p0),
        (old_q0, right[4], right[5], right[6]),
        flat,
        shift,
    );
    if !all.any_nonzero() {
        return false;
    }
    let mut line = [V::splat(0); 14];
    line[..7].copy_from_slice(&left[1..8]);
    line[7..14].copy_from_slice(&right[..7]);
    let mut out = line;
    out[3..7].copy_from_slice(&inner_out[..4]);
    out[7..11].copy_from_slice(&inner_out[4..8]);
    smooth_lanes::<V, 14, 6, 1, 4>(&line, &mut out, all);
    left[2..8].copy_from_slice(&out[1..7]);
    right[..6].copy_from_slice(&out[7..13]);
    // SAFETY: same checked 16-sample footprint as the loads above.
    unsafe {
        store_v_tile::<V, LINES>(left, data.as_mut_ptr(), edge - 8, along);
        store_v_tile::<V, LINES>(right, data.as_mut_ptr(), edge, along);
    }
    true
}

#[inline(always)]
unsafe fn v_group<V: I16x8Shuffles + I16FilterLanes, const LINES: usize>(
    data: &mut [u16],
    edge: usize,
    along: usize,
    filter: Av1EdgeFilter,
    shift: u32,
) {
    let active = if LINES == 4 {
        V::load_halves(&[-1; 4], &[0; 4])
    } else {
        V::splat(-1)
    };
    // SAFETY: caller validated `p3..q3` across all `LINES` rows.
    let inner = unsafe { load_v_tile::<V, LINES>(data.as_ptr(), edge - 4, along) };
    let p = [inner[3], inner[2], inner[1], inner[0]];
    let q = [inner[4], inner[5], inner[6], inner[7]];
    let (mask, hev, flat) = match filter.length {
        Av1FilterLength::Four => masks::<V, 2>(&p, &q, filter, shift, active),
        Av1FilterLength::Six => masks::<V, 3>(&p, &q, filter, shift, active),
        Av1FilterLength::Eight | Av1FilterLength::Fourteen => {
            masks::<V, 4>(&p, &q, filter, shift, active)
        }
    };
    if !mask.any_nonzero() {
        return;
    }
    let [np1, np0, nq0, nq1] = narrow_lanes((p[1], p[0], q[0], q[1]), mask, hev, shift);
    let mut result = [inner[0], inner[1], np1, np0, nq0, nq1, inner[6], inner[7]];
    if filter.length == Av1FilterLength::Six && flat.any_nonzero() {
        let line: [V; 6] = [inner[1], inner[2], inner[3], inner[4], inner[5], inner[6]];
        let mut out: [V; 6] = [
            result[1], result[2], result[3], result[4], result[5], result[6],
        ];
        smooth_lanes::<V, 6, 2, 1, 3>(&line, &mut out, flat);
        result[1..7].copy_from_slice(&out);
    } else if matches!(
        filter.length,
        Av1FilterLength::Eight | Av1FilterLength::Fourteen
    ) && flat.any_nonzero()
    {
        smooth_lanes::<V, 8, 3, 0, 3>(&inner, &mut result, flat);
        if filter.length == Av1FilterLength::Fourteen {
            // SAFETY: caller checked `tile_reach == 8` for `Fourteen`.
            if unsafe {
                v_fourteen::<V, LINES>(data, edge, along, flat, shift, (p[0], q[0]), &result)
            } {
                return;
            }
        }
    }
    // SAFETY: same checked `p3..q3` footprint as the load above.
    unsafe { store_v_tile::<V, LINES>(result, data.as_mut_ptr(), edge - 4, along) };
}

// ------------------------------------------------ run dispatcher ----

#[inline(always)]
unsafe fn filter_edge_lanes<V8: I16x8Shuffles + I16FilterLanes, VWide: I16FilterLanes>(
    data: &mut [u16],
    q0: usize,
    across: usize,
    along: usize,
    lines: usize,
    filter: Av1EdgeFilter,
    depth: BitDepth,
) {
    debug_assert_eq!(V8::LANES, 8);
    let shift = depth.above_eight();
    if along == 1 {
        let r = filter.length.reach();
        let fits = q0 >= r * across
            && lines
                .checked_sub(1)
                .is_some_and(|last| q0 + last + (r - 1) * across < data.len());
        if !fits {
            filter_edge_at(data, q0, across, along, lines, filter, depth);
            return;
        }
        let mut done = 0;
        while done + VWide::LANES <= lines {
            // SAFETY: `fits` covers `q0 + done .. q0 + done + VWide::LANES`.
            unsafe { h_group::<VWide, 0>(data, q0 + done, across, filter, shift) };
            done += VWide::LANES;
        }
        while V8::LANES < VWide::LANES && done + 8 <= lines {
            // SAFETY: `fits` covers `q0 + done .. q0 + done + 8`.
            unsafe { h_group::<V8, 0>(data, q0 + done, across, filter, shift) };
            done += 8;
        }
        if done + 4 <= lines {
            // SAFETY: `fits` covers `q0 + done .. q0 + done + 4`.
            unsafe { h_group::<V8, 4>(data, q0 + done, across, filter, shift) };
            done += 4;
        }
        if done < lines {
            filter_edge_at(data, q0 + done, across, along, lines - done, filter, depth);
        }
        return;
    }
    let tile_reach = if filter.length == Av1FilterLength::Fourteen {
        8
    } else {
        4
    };
    let fits = across == 1
        && q0 >= tile_reach
        && lines
            .checked_sub(1)
            .is_some_and(|last| q0 + last * along + (tile_reach - 1) < data.len());
    if !fits {
        filter_edge_at(data, q0, across, along, lines, filter, depth);
        return;
    }
    let mut done = 0;
    while done + 8 <= lines {
        // SAFETY: `fits` covers 8 vertical rows starting at `q0 + done * along`.
        unsafe { v_group::<V8, 8>(data, q0 + done * along, along, filter, shift) };
        done += 8;
    }
    if done + 4 <= lines {
        // SAFETY: `fits` covers 4 vertical rows starting at `q0 + done * along`.
        unsafe { v_group::<V8, 4>(data, q0 + done * along, along, filter, shift) };
        done += 4;
    }
    if done < lines {
        filter_edge_at(
            data,
            q0 + done * along,
            across,
            along,
            lines - done,
            filter,
            depth,
        );
    }
}

#[inline(always)]
unsafe fn av1_runs_lanes<V8: I16x8Shuffles + I16FilterLanes, VWide: I16FilterLanes>(
    data: &mut [u16],
    plane: Plane,
    runs: &[EdgeRun<Av1EdgeFilter>],
    depth: BitDepth,
) {
    let bits = depth.bits();
    if bits != 10 && bits != 12 {
        av1_scalar_at(data, plane, runs, depth);
        return;
    }
    for run in ValidatedRuns::new(plane, runs, true, 1, |f: Av1EdgeFilter| f.length.reach()) {
        let (base, across, along) = plane.address(&run);
        // SAFETY: `ValidatedRuns` checked the run footprint; caller enabled ISA.
        unsafe {
            filter_edge_lanes::<V8, VWide>(data, base, across, along, run.lines, run.filter, depth);
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! high_deblock_isa {
    ($mod_name:ident, $(#[$feature:meta])*, $v8:ty, $vwide:ty) => {
        mod $mod_name {
            use nc_common::BitDepth;

            use super::super::av1::Av1EdgeFilter;
            use super::super::runs::{EdgeRun, Plane};
            use super::av1_runs_lanes;

            $(#[$feature])*
            pub(super) unsafe fn runs(
                data: &mut [u16],
                plane: Plane,
                runs: &[EdgeRun<Av1EdgeFilter>],
                depth: BitDepth,
            ) {
                // SAFETY: wrapper enables target_feature.
                unsafe { av1_runs_lanes::<$v8, $vwide>(data, plane, runs, depth) }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
high_deblock_isa!(
    x86_avx2,
    #[target_feature(enable = "avx2")],
    crate::simd::avx2::I16x8,
    crate::simd::avx2::I16x16
);

#[cfg(target_arch = "x86_64")]
high_deblock_isa!(
    x86_avx512,
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")],
    crate::simd::avx2::I16x8,
    crate::simd::avx512_i16::I16x32
);

#[cfg(target_arch = "aarch64")]
high_deblock_isa!(
    neon_high,
    #[target_feature(enable = "neon")],
    crate::simd::neon::I16x8,
    crate::simd::neon::I16x8
);
