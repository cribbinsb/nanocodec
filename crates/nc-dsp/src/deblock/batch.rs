//! Independent short-luma segments for HEVC and VVC, on 8/16/32 word lanes.
//! Each four-line segment retains its endpoint decision. Eight-line tiles
//! transpose independently in 128-bit groups, following dav1d 1.5.4
//! `loopfilter16_avx512.asm` (VideoLAN/dav1d authors and Two Orioles, LLC,
//! BSD-2-Clause; `THIRD_PARTY.md`). Filter arithmetic stays in `short_luma`.
//! Inputs are bytes; curvature sums, thresholds and filter sums fit i16.
#![allow(clippy::inline_always)]

use super::short_luma;
use crate::simd::I16x8Tiles;

const POSITIONS: usize = 8;
const REACH: usize = POSITIONS / 2;
const TILE: usize = 8;
const FIRST: [u8; 16] = [0, 1, 0, 1, 0, 1, 0, 1, 8, 9, 8, 9, 8, 9, 8, 9];
const LAST: [u8; 16] = [6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15];

#[derive(Clone, Copy)]
pub(super) struct Geometry {
    pub base: usize,
    pub across: usize,
    pub along: usize,
}

#[inline(always)]
fn transpose<V: I16x8Tiles>(rows: [V; TILE]) -> [V; TILE] {
    let mut rows = rows;
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
fn load<V: I16x8Tiles>(data: &[u8], e: Geometry) -> [V; POSITIONS] {
    let origin = e.base.checked_sub(REACH * e.across).expect("edge reach");
    let last = origin + (V::LANES - 1) * e.along + (POSITIONS - 1) * e.across;
    assert!(last < data.len(), "complete deblocking batch");
    let mut values = [V::splat(0); POSITIONS];
    if e.along == 1 {
        for (k, value) in values.iter_mut().enumerate() {
            let at = origin + k * e.across;
            // SAFETY: the last-byte check covers this complete position row.
            *value = V::load_u8(unsafe { data.get_unchecked(at..at + V::LANES) });
        }
    } else {
        for (k, value) in values.iter_mut().enumerate() {
            let mut rows = [&[][..]; 4];
            for (group, row) in rows.iter_mut().enumerate().take(V::LANES / TILE) {
                let at = origin + (group * TILE + k) * e.along;
                // SAFETY: every tile row lies inside the checked footprint.
                *row = unsafe { data.get_unchecked(at..at + TILE) };
            }
            *value = V::load_tiles8(rows);
        }
        values = transpose(values);
    }
    values
}

#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "fixed tap indices avoid outlining an iterator over SIMD values"
)]
fn store<V: I16x8Tiles>(data: &mut [u8], e: Geometry, values: [V; POSITIONS]) {
    let origin = e.base - REACH * e.across;
    if e.along == 1 {
        for k in 1..POSITIONS - 1 {
            let at = origin + k * e.across;
            // SAFETY: load checked the same complete footprint before filtering.
            values[k].store_u8_sat(unsafe { data.get_unchecked_mut(at..at + V::LANES) });
        }
    } else {
        // Each tile row is separate: a vertical footprint has stride >= 8;
        // within a vector, the groups' row indices differ by eight.
        let rows = transpose(values);
        let ptr = data.as_mut_ptr();
        for (k, value) in rows.into_iter().enumerate() {
            let mut tiles: [&mut [u8]; 4] = [&mut [], &mut [], &mut [], &mut []];
            for (group, tile) in tiles.iter_mut().enumerate().take(V::LANES / TILE) {
                let at = origin + (group * TILE + k) * e.along;
                // SAFETY: load checked each eight-byte span; these tile rows
                // are disjoint. Unchanged p3/q3 are written back verbatim.
                *tile = unsafe { core::slice::from_raw_parts_mut(ptr.add(at), TILE) };
            }
            value.store_tiles8(tiles);
        }
    }
}

#[inline(always)]
fn less<V: I16x8Tiles>(a: V, b: V) -> V {
    b.gt_mask(a)
}
#[inline(always)]
fn endpoints<V: I16x8Tiles>(v: V) -> (V, V) {
    (v.shuffle_tiles(&FIRST), v.shuffle_tiles(&LAST))
}

/// Filter a contiguous batch after its complete footprint was validated.
/// # Safety
/// V's ISA is enabled; all sample lines are inside the plane view.
#[inline(always)]
pub(super) unsafe fn short<V: I16x8Tiles, const VVC: bool>(
    data: &mut [u8],
    e: Geometry,
    beta: i16,
    tc: i16,
) {
    if let Some(values) = filter::<V, VVC>(load::<V>(data, e), V::splat(beta), V::splat(tc)) {
        store(data, e, values);
    }
}

/// Four/eight-line segments packed without crossing their endpoint decisions.
/// HEVC clips strong outputs by [2,2,2] tC, VVC by [3,2,1].
#[inline(always)]
#[allow(clippy::similar_names)]
fn filter<V: I16x8Tiles, const VVC: bool>(
    values: [V; POSITIONS],
    beta: V,
    tc: V,
) -> Option<[V; POSITIONS]> {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = values;
    let zero = V::splat(0);
    let two = V::splat(2);
    let dp = p2.sub(p1.mul(two)).add(p0).abs();
    let dq = q2.sub(q1.mul(two)).add(q0).abs();
    let (d0, d3) = endpoints(dp.add(dq));
    let active = tc.gt_mask(zero).and(less(d0.add(d3), beta));
    if !active.any_nonzero() {
        return None;
    }
    let (spread0, spread3) = endpoints(p3.sub(p0).abs().add(q0.sub(q3).abs()));
    let (step0, step3) = endpoints(p0.sub(q0).abs());
    let maximum = d0.max(d3);
    let strong = active
        .and(less(maximum.add(maximum), beta.sra(2)))
        .and(less(spread0.max(spread3), beta.sra(3)))
        .and(less(
            step0.max(step3),
            tc.mul(V::splat(5)).add(V::splat(1)).sra(1),
        ));
    let normal = active.and(strong.eq_mask(zero));
    let mut out = values;
    if normal.any_nonzero() {
        let n = short_luma::normal([p2, p1, p0], [q0, q1, q2], tc);
        let normal = normal.and(n.enabled);
        let threshold = beta.add(beta.sra(1)).sra(3);
        let (dp0, dp3) = endpoints(dp);
        let (dq0, dq3) = endpoints(dq);
        out[2] = normal.and(less(dp0.add(dp3), threshold)).blend(n.p1, p1);
        out[3] = normal.blend(n.p0, p0);
        out[4] = normal.blend(n.q0, q0);
        out[5] = normal.and(less(dq0.add(dq3), threshold)).blend(n.q1, q1);
    }
    if strong.any_nonzero() {
        let twice = tc.add(tc);
        let limits = if VVC {
            [twice.add(tc), twice, tc]
        } else {
            [twice; 3]
        };
        let [sp0, sp1, sp2] = short_luma::strong_side([p3, p2, p1, p0], [q0, q1], limits);
        let [sq0, sq1, sq2] = short_luma::strong_side([q3, q2, q1, q0], [p0, p1], limits);
        for (k, value) in [(1, sp2), (2, sp1), (3, sp0), (4, sq0), (5, sq1), (6, sq2)] {
            out[k] = strong.blend(value, out[k]);
        }
    }
    Some(out)
}

/// Filter independent eight-line tiles with per-segment thresholds.
/// Geometry may differ between tiles; the driver proves their read/write
/// footprints disjoint before combining them. Orientation is shared.
/// # Safety
/// V's ISA is enabled. Each of the first LANES/8 tiles is valid, and no
/// two of those tile footprints overlap.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "fixed tap indices avoid outlining an iterator over SIMD values"
)]
pub(super) unsafe fn short_tiles<V: I16x8Tiles, const VVC: bool>(
    data: &mut [u8],
    tiles: &[Geometry; 4],
    beta: &[i16; 8],
    tc: &[i16; 8],
) {
    let groups = V::LANES / TILE;
    let horizontal = tiles[0].along == 1;
    let mut origins = [0; 4];
    for (g, e) in tiles.iter().take(groups).enumerate() {
        let origin = e.base.checked_sub(REACH * e.across).expect("edge reach");
        let last = origin + (TILE - 1) * (e.along + e.across);
        assert!(last < data.len(), "complete deblocking tile");
        origins[g] = origin;
    }
    let mut values = [V::splat(0); POSITIONS];
    for (k, value) in values.iter_mut().enumerate() {
        let mut sources = [&[][..]; 4];
        for (g, source) in sources.iter_mut().enumerate().take(groups) {
            let e = tiles[g];
            let at = origins[g] + k * if horizontal { e.across } else { e.along };
            // SAFETY: the exact tile bounds were checked above.
            *source = unsafe { data.get_unchecked(at..at + TILE) };
        }
        *value = V::load_tiles8(sources);
    }
    if !horizontal {
        values = transpose(values);
    }
    let Some(mut out) = filter::<V, VVC>(values, V::splat_segments(beta), V::splat_segments(tc))
    else {
        return;
    };
    if !horizontal {
        out = transpose(out);
    }
    let ptr = data.as_mut_ptr();
    let changed = if horizontal {
        1..POSITIONS - 1
    } else {
        0..POSITIONS
    };
    for k in changed {
        let mut destinations: [&mut [u8]; 4] = [&mut [], &mut [], &mut [], &mut []];
        for (g, destination) in destinations.iter_mut().enumerate().take(groups) {
            let e = tiles[g];
            let at = origins[g] + k * if horizontal { e.across } else { e.along };
            // SAFETY: exact bounds checked above, and the driver establishes
            // that different tiles are disjoint. Within a tile rows are disjoint.
            *destination = unsafe { core::slice::from_raw_parts_mut(ptr.add(at), TILE) };
        }
        out[k].store_tiles8(destinations);
    }
}
