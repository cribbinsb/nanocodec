//! AV1 edge filters (AV1 §7.14.6) on the packed-byte lanes of
//! [`crate::lpf_bytes`]: eight lines per vector, byte masks and the narrow
//! filter in bytes (AV1's 4-tap filter is libvpx's), and the 6/8/14-tap
//! smoothing in widened words, computed only when some lane is flat. Same
//! results as [`super::av1::filter_edge`] (the scalar kernel).
//! Four-line memory operations follow dav1d 1.5.4 `loopfilter_avx2.asm`
//! (VideoLAN/dav1d authors and Two Orioles, BSD-2) as a design reference;
//! the shared filter arithmetic is unchanged, with no assembly copied.
//!
//! Every smoothing filter is a clamped box sum plus a short centre box:
//! output `k` of the line `x` is `(Σ_{|j−k|≤R} x[clamp(j)] + Σ_{|j−k|≤c}
//! x[j] + round) >> shift`, with (`R`, `c`, `shift`) = (2, 1, 3) for the
//! 6-tap [1, 2, 2, 2, 1] on p2..q2 (the end sample repeats: 3·p2), (3, 0, 3)
//! for the 8-tap on p3..q3, and (6, 1, 4) for the 14-tap
//! [1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1] on p6..q6. Sums of at most 16
//! samples plus rounding stay below 4096, well inside `i16`.

#![allow(clippy::inline_always)]

use super::av1::{Av1EdgeFilter, Av1FilterLength};
use crate::lpf_bytes::{
    Bytes, INNER, LANES, Words, difference, load, narrow, select, store, within,
};

/// `flat`/`flat2` threshold (as the scalar kernel).
const FLAT_THRESHOLD: u8 = 1;
/// Saturating byte sums give the exact `blimit` test only below 255.
const MAX_EXACT_BLIMIT: u8 = 254;
/// Samples each side the 14-tap tiles span (two 8x8 tiles: p7..p0, q0..q7).
const WIDE_TILE: usize = 8;
/// Samples of the 14-tap line p6..q6.
const FOURTEEN_SAMPLES: usize = 14;

/// Tile memory layout, separate from the shared AV1 filter arithmetic. The
/// ordinary implementation addresses one eight-line tile; wide backends may
/// keep independent tiles together in a wider register.
pub(crate) trait Tile<B: Bytes> {
    /// Caller validates every sample and enables B's ISA.
    unsafe fn load<const LINES: usize>(
        data: *const u8,
        edge: usize,
        across: usize,
        along: usize,
        before: usize,
    ) -> [B; LANES];
    /// Caller validates every sample and enables B's ISA.
    unsafe fn store<const LINES: usize>(
        data: *mut u8,
        edge: usize,
        across: usize,
        along: usize,
        before: usize,
        values: [B; LANES],
        changed: core::ops::Range<usize>,
    );
}

struct BasicTile;
impl<B: Bytes> Tile<B> for BasicTile {
    #[inline(always)]
    unsafe fn load<const LINES: usize>(
        data: *const u8,
        edge: usize,
        across: usize,
        along: usize,
        before: usize,
    ) -> [B; LANES] {
        // SAFETY: forwarded validated tile and ISA contract.
        unsafe { load_lines::<B, LINES>(data, edge, across, along, before) }
    }
    #[inline(always)]
    unsafe fn store<const LINES: usize>(
        data: *mut u8,
        edge: usize,
        across: usize,
        along: usize,
        before: usize,
        values: [B; LANES],
        changed: core::ops::Range<usize>,
    ) {
        // SAFETY: forwarded validated tile and ISA contract.
        unsafe {
            store_lines::<B, LINES>(data, edge, across, along, before, values, changed);
        }
    }
}

/// The `filterMask`, `hevMask` and `flat` lane masks of a filter reading
/// `TAPS` samples each side for its decisions (2, 3 or 4).
#[inline(always)]
fn masks<B: Bytes, const TAPS: usize>(
    p: &[B; LANES],
    q: &[B; LANES],
    filter: Av1EdgeFilter,
) -> (B, B, B) {
    let mut adjacent = difference(p[1], p[0]).max(difference(q[1], q[0]));
    let hev = within(adjacent, filter.thresh).xor(B::splat(u8::MAX));
    let mut flat = adjacent;
    for i in 2..TAPS {
        adjacent = adjacent
            .max(difference(p[i], p[i - 1]))
            .max(difference(q[i], q[i - 1]));
        flat = flat.max(difference(p[i], p[0])).max(difference(q[i], q[0]));
    }
    let across = difference(p[0], q[0]);
    let boundary = across
        .add_sat(across)
        .add_sat(difference(p[1], q[1]).shr::<1>());
    let mask = within(adjacent, filter.limit).and(within(boundary, filter.blimit));
    (mask, hev, within(flat, FLAT_THRESHOLD).and(mask))
}

/// The smoothing filter of the module docs on the `N`-sample line `input`
/// (outputs 1..N−1), written where `mask` is set.
#[inline(always)]
fn smooth<B: Bytes, const N: usize, const OUTER: usize, const CENTRE: usize, const SHIFT: i32>(
    input: &[B; N],
    output: &mut [B; N],
    mask: B,
) {
    // Keep widening in the target-feature caller. Array::map outlined on
    // x86 and called an ISA helper once per tap instead of emitting widening.
    let mut taps = [B::W::splat(0); N];
    for (tap, sample) in taps.iter_mut().zip(input) {
        *tap = sample.widen();
    }
    let mut sum = B::W::splat(1 << (SHIFT - 1));
    // The outer box of output 1, with the first sample repeated.
    for offset in 0..=(2 * OUTER) {
        sum = sum.add(taps[(1 + offset).saturating_sub(OUTER).min(N - 1)]);
    }
    for i in 1..N - 1 {
        let mut total = sum;
        for tap in &taps[i - CENTRE..=i + CENTRE] {
            total = total.add(*tap);
        }
        output[i] = select(output[i], B::narrow(total.shr::<SHIFT>()), mask);
        sum = sum
            .sub(taps[i.saturating_sub(OUTER)])
            .add(taps[(i + OUTER + 1).min(N - 1)]);
    }
}

/// Filter four or eight lines of one edge (q0 of the first at `edge`).
///
/// # Safety
///
/// The caller checked the footprint: [`INNER`] samples each side of all
/// `LINES` lines ([`WIDE_TILE`] for the 14-tap filter), with `across` or
/// `along` equal to 1.
#[inline(always)]
unsafe fn group<B: Bytes, T: Tile<B>, const REACH: usize, const LINES: usize>(
    data: &mut [u8],
    edge: usize,
    across: usize,
    along: usize,
    filter: Av1EdgeFilter,
) {
    // SAFETY: the inner tile (p3..q3) lies in the checked footprint.
    let inner = unsafe { T::load::<LINES>(data.as_ptr(), edge, across, along, INNER) };
    let mut p = [B::splat(0); LANES];
    let mut q = p;
    for i in 0..INNER {
        p[i] = inner[INNER - 1 - i];
        q[i] = inner[INNER + i];
    }
    let decision_taps = match REACH {
        2 => 2,
        3 => 3,
        _ => INNER,
    };
    let (mask, hev, flat) = match decision_taps {
        2 => masks::<B, 2>(&p, &q, filter),
        3 => masks::<B, 3>(&p, &q, filter),
        _ => masks::<B, INNER>(&p, &q, filter),
    };
    // Zero padding must not make inactive lanes pass flat/mask tests.
    let active = if LINES == 4 {
        // SAFETY: four constant readable bytes; the caller enabled B's ISA.
        unsafe { B::load4([u8::MAX; 4].as_ptr()) }
    } else {
        B::splat(u8::MAX)
    };
    let (mask, flat) = (mask.and(active), flat.and(active));
    if !mask.any() {
        return;
    }
    let old_p0_q0 = (p[0], q[0]);
    narrow(&mut p, &mut q, mask, hev);
    let mut result: [B; LANES] = core::array::from_fn(|i| {
        if i < INNER {
            p[INNER - 1 - i]
        } else {
            q[i - INNER]
        }
    });
    let mut written = 2;
    if REACH == 3 && flat.any() {
        let line: [B; 6] = inner[1..7].try_into().expect("p2..q2");
        let mut out: [B; 6] = result[1..7].try_into().expect("p2..q2");
        smooth::<B, 6, 2, 1, 3>(&line, &mut out, flat);
        result[1..7].copy_from_slice(&out);
    } else if REACH >= INNER && flat.any() {
        written = 3;
        smooth::<B, 8, 3, 0, 3>(&inner, &mut result, flat);
        if REACH == 7 {
            // SAFETY: the 14-tap footprint (p7..q7) was checked.
            let stored = unsafe {
                fourteen::<B, T, LINES>(data, edge, across, along, flat, old_p0_q0, &result)
            };
            if stored {
                return;
            }
        }
    }
    // SAFETY: the same checked inner tile as the initial load.
    unsafe {
        T::store::<LINES>(
            data.as_mut_ptr(),
            edge,
            across,
            along,
            INNER,
            result,
            INNER - written..INNER + written,
        );
    }
}

/// The 14-tap stage: where `flat` and `flat2` hold, smooth p5..q5 over the
/// two 8x8 tiles and store them (with the narrow/8-tap `result` inside).
/// Returns whether it stored; otherwise the caller stores `result`.
///
/// # Safety
///
/// The caller checked [`WIDE_TILE`] samples each side of all `LINES` lines.
#[inline(always)]
unsafe fn fourteen<B: Bytes, T: Tile<B>, const LINES: usize>(
    data: &mut [u8],
    edge: usize,
    across: usize,
    along: usize,
    flat: B,
    (old_p0, old_q0): (B, B),
    result: &[B; LANES],
) -> bool {
    const REACH: usize = 7;
    // SAFETY: the 14-tap footprint (p7..q7) was checked.
    let (left, right) = unsafe {
        (
            T::load::<LINES>(data.as_ptr(), edge, across, along, WIDE_TILE),
            T::load::<LINES>(data.as_ptr(), edge, across, along, 0),
        )
    };
    let mut outer = B::splat(0);
    for i in INNER..REACH {
        outer = outer
            .max(difference(left[WIDE_TILE - 1 - i], old_p0))
            .max(difference(right[i], old_q0));
    }
    let flat2 = within(outer, FLAT_THRESHOLD).and(flat);
    if !flat2.any() {
        return false;
    }
    // p6..q6: the left tile without p7, the right without q7.
    let line: [B; FOURTEEN_SAMPLES] = core::array::from_fn(|i| {
        if i < REACH {
            left[i + 1]
        } else {
            right[i - REACH]
        }
    });
    let mut out = line;
    out[REACH - INNER..REACH + INNER].copy_from_slice(result);
    smooth::<B, FOURTEEN_SAMPLES, 6, 1, 4>(&line, &mut out, flat2);
    let new_left: [B; LANES] = core::array::from_fn(|i| if i == 0 { left[0] } else { out[i - 1] });
    let new_right: [B; LANES] =
        core::array::from_fn(|i| if i < REACH { out[REACH + i] } else { right[i] });
    // SAFETY: the same checked 14-tap tiles as the loads above.
    unsafe {
        T::store::<LINES>(
            data.as_mut_ptr(),
            edge,
            across,
            along,
            WIDE_TILE,
            new_left,
            2..LANES,
        );
        T::store::<LINES>(
            data.as_mut_ptr(),
            edge,
            across,
            along,
            0,
            new_right,
            0..LANES - 2,
        );
    }
    true
}

/// Samples each side the tile loads of a filter span.
const fn tile_reach(length: Av1FilterLength) -> usize {
    match length {
        Av1FilterLength::Fourteen => WIDE_TILE,
        _ => INNER,
    }
}

/// [`super::av1::filter_edge`] on packed lanes: whole eight-line groups in
/// place, a four-line remainder through direct short loads/stores, and the scalar kernel
/// whenever the tiles would leave the plane or `blimit` exceeds the exact
/// byte range.
///
/// # Safety
///
/// The ISA of `B` must be enabled (the entry points below do so), `lines`
/// must be a multiple of four, and the normative filter footprint must be
/// valid (the public dispatcher checks it; wider tile loads are checked here).
#[inline(always)]
pub(crate) unsafe fn filter_edge<B: Bytes>(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    lines: usize,
    filter: Av1EdgeFilter,
) {
    let reach = tile_reach(filter.length);
    let fits = q0 >= reach * across
        && lines
            .checked_sub(1)
            .is_some_and(|last| q0 + last * along + (reach - 1) * across < plane.len());
    let independent = across == 1 || along == 1;
    if !fits || !independent || filter.blimit > MAX_EXACT_BLIMIT {
        super::av1::filter_edge(plane, q0, across, along, lines, filter);
        return;
    }
    let full = lines / LANES * LANES;
    for first in (0..full).step_by(LANES) {
        // SAFETY: `fits` covers every line; the caller enabled B's ISA.
        unsafe {
            by_length::<B, BasicTile, LANES>(plane, q0 + first * along, across, along, filter);
        };
    }
    if full < lines {
        debug_assert_eq!(lines - full, super::av1::AV1_EDGE_LINES);
        // SAFETY: fits covers these four remaining lines; dispatch requires
        // a multiple of four. Only those lines are loaded and stored.
        unsafe { by_length::<B, BasicTile, 4>(plane, q0 + full * along, across, along, filter) };
    }
}

/// Sixteen independent lines share the same arithmetic as the ordinary
/// eight-line kernel. Short remainders retain the established narrow path.
///
/// # Safety
/// Both lane implementations' ISA must be enabled; normative footprints must
/// be valid, and `lines` must be a multiple of four.
#[inline(always)]
pub(crate) unsafe fn filter_wide<B: Bytes, T: Tile<B>>(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    lines: usize,
    filter: Av1EdgeFilter,
    tail: super::Av1EdgeFn,
) {
    const WIDE_LINES: usize = 16;
    let reach = tile_reach(filter.length);
    let fits = q0 >= reach * across
        && lines
            .checked_sub(1)
            .is_some_and(|last| q0 + last * along + (reach - 1) * across < plane.len());
    if !fits || (across != 1 && along != 1) || filter.blimit > MAX_EXACT_BLIMIT {
        super::av1::filter_edge(plane, q0, across, along, lines, filter);
        return;
    }
    let full = lines / WIDE_LINES * WIDE_LINES;
    for first in (0..full).step_by(WIDE_LINES) {
        // SAFETY: fits covers all sixteen lines; caller enables the ISA.
        unsafe {
            by_length::<B, T, WIDE_LINES>(plane, q0 + first * along, across, along, filter);
        }
    }
    if full < lines {
        // SAFETY: the validated remainder is a multiple of four.
        unsafe {
            tail(
                plane,
                q0 + full * along,
                across,
                along,
                lines - full,
                filter,
            );
        }
    }
}

/// One group, specialised by filter length.
///
/// # Safety
///
/// As [`group`].
#[inline(always)]
unsafe fn by_length<B: Bytes, T: Tile<B>, const LINES: usize>(
    data: &mut [u8],
    edge: usize,
    across: usize,
    along: usize,
    filter: Av1EdgeFilter,
) {
    // SAFETY: forwarded contract.
    unsafe {
        match filter.length {
            Av1FilterLength::Four => group::<B, T, 2, LINES>(data, edge, across, along, filter),
            Av1FilterLength::Six => group::<B, T, 3, LINES>(data, edge, across, along, filter),
            Av1FilterLength::Eight => group::<B, T, 4, LINES>(data, edge, across, along, filter),
            Av1FilterLength::Fourteen => group::<B, T, 7, LINES>(data, edge, across, along, filter),
        }
    }
}

/// A four-line tile stays in registers; eight-line groups use the existing
/// shared tile path. In particular, horizontal loads never read line 5.
#[inline(always)]
unsafe fn load_lines<B: Bytes, const LINES: usize>(
    data: *const u8,
    edge: usize,
    across: usize,
    along: usize,
    before: usize,
) -> [B; LANES] {
    if LINES == LANES {
        // SAFETY: caller checked the complete eight-line tile.
        return unsafe { load::<B>(data, edge, across, along, before) };
    }
    // SAFETY: caller checked the four-line tile. Horizontal edges load four
    // samples per position; vertical edges load eight positions in four rows.
    unsafe {
        let rows = core::array::from_fn(|i| {
            if along == 1 {
                B::load4(data.add(edge - before * across + i * across))
            } else if i < LINES {
                B::load(data.add(edge - before + i * along))
            } else {
                B::splat(0)
            }
        });
        if along == 1 { rows } else { B::transpose(rows) }
    }
}

#[inline(always)]
unsafe fn store_lines<B: Bytes, const LINES: usize>(
    data: *mut u8,
    edge: usize,
    across: usize,
    along: usize,
    before: usize,
    values: [B; LANES],
    changed: core::ops::Range<usize>,
) {
    if LINES == LANES {
        // SAFETY: caller checked the complete eight-line tile.
        unsafe {
            store::<B>(data, edge, across, along, before, values, changed);
        }
        return;
    }
    // SAFETY: same checked four-line tile as the loads. Do not store padding lanes.
    unsafe {
        if along == 1 {
            for i in changed {
                values[i].store4(data.add(edge - before * across + i * across));
            }
        } else {
            for (i, value) in B::transpose(values).into_iter().take(LINES).enumerate() {
                value.store(data.add(edge - before + i * along));
            }
        }
    }
}
