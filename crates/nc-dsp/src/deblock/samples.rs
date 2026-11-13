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
fn checked_origin(plane: &[u8], e: &Edge) -> usize {
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
/// [`checked_origin`] checked: `at..at + len` is either one line's p3..q3
/// (`len` = 8, `across` = 1) or one position of the edge's lines (`len` =
/// lines, `along` = 1).
///
/// # Safety
///
/// `at..at + len` lies in the range checked by [`checked_origin`].
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn run(plane: &[u8], at: usize, len: usize) -> &[u8] {
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
unsafe fn run_mut(plane: &mut [u8], at: usize, len: usize) -> &mut [u8] {
    // SAFETY: caller contract.
    unsafe { plane.get_unchecked_mut(at..at + len) }
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
                let mut bytes = [0u8; LANES];
                // A four-line chroma batch can use a word load instead of a
                // runtime memcpy when its copy size is constant.
                let lines = if e.lines == HALF_LANES {
                    HALF_LANES
                } else {
                    e.lines
                };
                // SAFETY: position k of the edge's lines (`run`).
                bytes[..lines].copy_from_slice(unsafe { run(plane, at, lines) });
                V::load_u8(&bytes)
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
        // Slice the tap range once: a take/skip iterator retains an outlined
        // nth call and iterator state in the SIMD128 horizontal store.
        for (offset, v) in values[from..to].iter().enumerate() {
            let at = origin + (from + offset) * e.across;
            if e.lines == LANES {
                // SAFETY: tap `from + offset` of the checked eight-line footprint.
                v.store_u8_sat(unsafe { run_mut(plane, at, LANES) });
            } else {
                let mut bytes = [0u8; LANES];
                v.store_u8_sat(&mut bytes);
                let lines = if e.lines == HALF_LANES {
                    HALF_LANES
                } else {
                    e.lines
                };
                // SAFETY: tap `from + offset` of the checked short-line footprint.
                unsafe { run_mut(plane, at, lines) }.copy_from_slice(&bytes[..lines]);
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
