//! Shared p3..q3 plane access for HEVC and VVC deblocking kernels.
//! Horizontal edges load contiguous lines; vertical edges transpose eight
//! positions into lanes. Partial calls touch only their declared lines.

use crate::simd::I16x8Shuffles;

pub(crate) const LANES: usize = 8;
pub(crate) const POSITIONS: usize = 8;
const HALF_LANES: usize = LANES / 2;
pub(crate) const REACH: usize = POSITIONS / 2;

/// Geometry of independent lines crossing an edge, starting at q0.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Edge {
    pub base: usize,
    pub across: usize,
    pub along: usize,
    pub lines: usize,
}

/// The first line's p3, after one check that every sample a call touches
/// (p3..q3 of each of the edge's lines) lies in `plane`: the per-sample
/// index arithmetic and slice checks cost more than the filter itself
/// (HEVC Faster x86, callgrind).
#[allow(clippy::inline_always)]
#[inline(always)]
fn checked_origin<T>(plane: &[T], e: &Edge) -> usize {
    assert!((1..=LANES).contains(&e.lines), "1 to 8 lines per edge");
    assert!(e.along == 1 || e.across == 1, "lines are rows or columns");
    let origin = e
        .base
        .checked_sub(REACH * e.across)
        .expect("deblocking sample inside the plane");
    let last = origin + (e.lines - 1) * e.along + (POSITIONS - 1) * e.across;
    assert!(last < plane.len(), "deblocking sample inside the plane");
    origin
}

/// The `len` samples of `plane` from `at`, within the range
/// [`checked_origin`] checked.
///
/// # Safety
///
/// `at..at + len` lies in the range checked by [`checked_origin`].
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn run<T>(plane: &[T], at: usize, len: usize) -> &[T] {
    // SAFETY: caller contract.
    unsafe { plane.get_unchecked(at..at + len) }
}

/// [`run`], mutable.
///
/// # Safety
///
/// As [`run`].
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn run_mut<T>(plane: &mut [T], at: usize, len: usize) -> &mut [T] {
    // SAFETY: caller contract.
    unsafe { plane.get_unchecked_mut(at..at + len) }
}

/// The `lines` (fewer than [`LANES`]) samples of `plane` from `at`, zero
/// padded to a lane row. Half rows (chroma edges) copy a fixed four samples
/// one by one: a copy of runtime length was a `memcpy` call per position
/// (VVC Faster x86, 2.2 million calls, 0.3 % of the encode's instructions).
///
/// # Safety
///
/// As [`run`] for `at..at + lines`.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn short_run<T: Copy + Default>(plane: &[T], at: usize, lines: usize) -> [T; LANES] {
    let mut row = [T::default(); LANES];
    // SAFETY: caller contract.
    let samples = unsafe { run(plane, at, lines) };
    if lines == HALF_LANES {
        for (to, &from) in row.iter_mut().zip(samples).take(HALF_LANES) {
            *to = from;
        }
    } else {
        row[..lines].copy_from_slice(samples);
    }
    row
}

/// Store the first `lines` (fewer than [`LANES`]) samples of `row` to
/// `plane` from `at`: [`short_run`]'s inverse, half rows likewise copied
/// one by one.
///
/// # Safety
///
/// As [`run_mut`] for `at..at + lines`.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn store_short_run<T: Copy>(plane: &mut [T], at: usize, row: &[T; LANES], lines: usize) {
    // SAFETY: caller contract.
    let samples = unsafe { run_mut(plane, at, lines) };
    if lines == HALF_LANES {
        for (to, &from) in samples.iter_mut().zip(row).take(HALF_LANES) {
            *to = from;
        }
    } else {
        samples.copy_from_slice(&row[..lines]);
    }
}

/// The 8x8 transpose of `rows` (three rounds of interleaving row `k` with
/// row `k + 4`).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn transpose<V: I16x8Shuffles>(rows: [V; LANES]) -> [V; LANES] {
    let mut m = rows;
    for _ in 0..3 {
        let mut next = m;
        for k in 0..LANES / 2 {
            next[2 * k] = m[k].zip_lo(m[k + 4]);
            next[2 * k + 1] = m[k].zip_hi(m[k + 4]);
        }
        m = next;
    }
    m
}

/// Positions p3..q3 of the edge's lines, one vector per position (lanes
/// past `e.lines` are unspecified).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn load<V: I16x8Shuffles>(plane: &[u8], e: &Edge) -> [V; POSITIONS] {
    let origin = checked_origin(plane, e);
    let mut out = [V::splat(0); POSITIONS];
    if e.along == 1 {
        for (k, v) in out.iter_mut().enumerate() {
            let at = origin + k * e.across;
            *v = if e.lines == LANES {
                // SAFETY: position k of the edge's 8 lines (`run`).
                V::load_u8(unsafe { run(plane, at, LANES) })
            } else {
                // SAFETY: position k of the edge's lines (`run`).
                V::load_u8(&unsafe { short_run(plane, at, e.lines) })
            };
        }
    } else {
        for (line, v) in out.iter_mut().enumerate().take(e.lines) {
            // SAFETY: p3..q3 of one line (`across` is 1 for these edges).
            *v = V::load_u8(unsafe { run(plane, origin + line * e.along, POSITIONS) });
        }
        out = transpose(out);
    }
    out
}

/// Store positions p3..q3 (clamped to 8 bits) of the edge's lines; for a
/// horizontal edge only positions `from..to` are written.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn store<V: I16x8Shuffles>(
    plane: &mut [u8],
    e: &Edge,
    values: [V; POSITIONS],
    (from, to): (usize, usize),
) {
    let origin = checked_origin(plane, e);
    if e.along == 1 {
        for (offset, v) in values[from..to].iter().enumerate() {
            let at = origin + (from + offset) * e.across;
            if e.lines == LANES {
                // SAFETY: tap `from + offset` of the checked eight-line footprint.
                v.store_u8_sat(unsafe { run_mut(plane, at, LANES) });
            } else {
                let mut bytes = [0u8; LANES];
                v.store_u8_sat(&mut bytes);
                // SAFETY: tap `from + offset` of the checked short-line footprint.
                unsafe { store_short_run(plane, at, &bytes, e.lines) };
            }
        }
    } else {
        let rows = transpose(values);
        for (line, v) in rows.iter().enumerate().take(e.lines) {
            // SAFETY: p3..q3 of one line (`across` is 1 for these edges).
            v.store_u8_sat(unsafe { run_mut(plane, origin + line * e.along, POSITIONS) });
        }
    }
}

/// [`load`] for `u16` samples (`0..=32767`).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn load_u16<V: I16x8Shuffles>(plane: &[u16], e: &Edge) -> [V; POSITIONS] {
    let origin = checked_origin(plane, e);
    let mut out = [V::splat(0); POSITIONS];
    if e.along == 1 {
        for (k, v) in out.iter_mut().enumerate() {
            let at = origin + k * e.across;
            *v = if e.lines == LANES {
                // SAFETY: position k of the edge's 8 lines (`run`).
                V::load_u16(unsafe { run(plane, at, LANES) })
            } else {
                // SAFETY: position k of the edge's lines (`run`).
                V::load_u16(&unsafe { short_run(plane, at, e.lines) })
            };
        }
    } else {
        for (line, v) in out.iter_mut().enumerate().take(e.lines) {
            // SAFETY: p3..q3 of one line (`across` is 1 for these edges).
            *v = V::load_u16(unsafe { run(plane, origin + line * e.along, POSITIONS) });
        }
        out = transpose(out);
    }
    out
}

/// [`store`] for `u16` samples, clamped to `0..=max_val`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn store_u16<V: I16x8Shuffles>(
    plane: &mut [u16],
    e: &Edge,
    values: [V; POSITIONS],
    (from, to): (usize, usize),
    max_val: i16,
) {
    let origin = checked_origin(plane, e);
    let (zero, hi) = (V::splat(0), V::splat(max_val));
    if e.along == 1 {
        for (offset, v) in values[from..to].iter().enumerate() {
            let at = origin + (from + offset) * e.across;
            let clamped = v.max(zero).min(hi);
            if e.lines == LANES {
                // SAFETY: tap `from + offset` of the checked eight-line footprint.
                clamped.store_u16(unsafe { run_mut(plane, at, LANES) });
            } else {
                let mut words = [0u16; LANES];
                clamped.store_u16(&mut words);
                // SAFETY: tap `from + offset` of the checked short-line footprint.
                unsafe { store_short_run(plane, at, &words, e.lines) };
            }
        }
    } else {
        let rows = transpose(values);
        for (line, v) in rows.iter().enumerate().take(e.lines) {
            // SAFETY: p3..q3 of one line (`across` is 1 for these edges).
            v.max(zero)
                .min(hi)
                .store_u16(unsafe { run_mut(plane, origin + line * e.along, POSITIONS) });
        }
    }
}
