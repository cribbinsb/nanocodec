//! H.264 single-plane chroma deblocking written once over
//! [`I16FilterLanes`]: one lane per line across the edge. Every value fits
//! 16 bits (at most `4 · 255` before a shift). Decisions are all-ones lane
//! masks combined with `and` and applied with `blend` (no multiplies).
//! Luma, and chroma of both planes at once, use the byte lanes of
//! `h264_bytes`.

use super::{CHROMA_LINES, Edge};
use crate::simd::{I16FilterLanes, I16Lanes, I16x8Shuffles};

/// Largest lane count of any vector type (AVX2 `I16x16`).
const MAX_LANES: usize = 16;

/// −1 in the lanes where `a < b`, else 0.
#[allow(clippy::inline_always)] // must inline into `#[target_feature]` kernels
#[inline(always)]
fn less<V: I16FilterLanes>(a: V, b: V) -> V {
    b.gt_mask(a)
}

/// The bS value that selects the strong filter (§8.7.2.4).
const STRONG_BS: u8 = 4;

/// `Clip3(−t, t, x)` per lane.
#[allow(clippy::inline_always)]
#[inline(always)]
fn clip<V: I16Lanes>(x: V, t: V) -> V {
    x.max(V::splat(0).sub(t)).min(t)
}

/// Index of sample `pos` across the edge (p3 = −4 … q3 = 3) of line `line`.
/// Unchecked: [`crate::Dsp`] verified that every sample of every line of the
/// edge lies inside the plane (`deblock_edge`) before calling a kernel, so
/// the kernels load and store through pointers at these indices.
#[allow(clippy::inline_always)]
#[inline(always)]
fn index(e: &Edge, line: usize, pos: isize) -> usize {
    (e.base + line * e.along).wrapping_add_signed(pos * e.across as isize)
}

/// The samples at positions `-(N / 2)..N / 2` across the edge (p side
/// first) of the lines `first..first + V::LANES`. Lines of a horizontal edge
/// are contiguous in memory: one vector load per position. A vertical
/// edge's lines are rows, so each line's `N` samples are one slice,
/// transposed into the lanes.
#[allow(clippy::inline_always)]
#[inline(always)]
#[allow(clippy::needless_range_loop)] // a transpose: lanes index the inner arrays
fn load_lines<V: I16Lanes, const N: usize>(plane: &[u8], e: &Edge, first: usize) -> [V; N] {
    let half = (N / 2) as isize;
    let mut out = [V::splat(0); N];
    if e.along == 1 {
        for (k, v) in out.iter_mut().enumerate() {
            // SAFETY: inside the plane, as checked by the dispatcher (`index`).
            *v = unsafe { V::load_u8_at(plane.as_ptr().add(index(e, first, k as isize - half))) };
        }
    } else {
        let mut lanes = [[0i16; MAX_LANES]; N];
        for lane in 0..V::LANES {
            let start = index(e, first + lane, -half);
            for (k, &sample) in plane[start..start + N].iter().enumerate() {
                lanes[k][lane] = i16::from(sample);
            }
        }
        for (v, lane_values) in out.iter_mut().zip(&lanes) {
            *v = V::load(lane_values);
        }
    }
    out
}

/// Store `values` (clamped to 8 bits) at positions `from..from + N` across
/// the edge of the lines `first..first + V::LANES`.
#[allow(clippy::inline_always)]
#[inline(always)]
#[allow(clippy::needless_range_loop)] // a transpose, as in `load_lines`
fn store_lines<V: I16Lanes, const N: usize>(
    plane: &mut [u8],
    e: &Edge,
    first: usize,
    from: isize,
    values: [V; N],
) {
    if e.along == 1 {
        for (k, v) in values.iter().enumerate() {
            // SAFETY: inside the plane, as checked by the dispatcher (`index`).
            unsafe {
                v.store_u8_sat_at(plane.as_mut_ptr().add(index(e, first, from + k as isize)));
            }
        }
    } else {
        let mut lanes = [[0i16; MAX_LANES]; N];
        for (v, lane_values) in values.iter().zip(&mut lanes) {
            v.store(lane_values);
        }
        for lane in 0..V::LANES {
            let start = index(e, first + lane, from);
            for (k, sample) in plane[start..start + N].iter_mut().enumerate() {
                *sample = lanes[k][lane].clamp(0, 255) as u8;
            }
        }
    }
}

/// Per-lane bS and tC0 of the lines `first..`, `lines_per_entry` lines per
/// entry of `e.bs`; `None` if no line of the group is filtered.
#[allow(clippy::inline_always)]
#[inline(always)]
fn strengths<V: I16Lanes>(e: &Edge, first: usize, lines_per_entry: usize) -> Option<(V, V)> {
    let (mut bs, mut tc0) = ([0i16; MAX_LANES], [0i16; MAX_LANES]);
    for lane in 0..V::LANES {
        let entry = (first + lane) / lines_per_entry;
        bs[lane] = i16::from(e.bs[entry]);
        tc0[lane] = e.tc0[entry];
    }
    if bs[..V::LANES].iter().all(|&b| b == 0) {
        return None;
    }
    // Not `bool::then(|| …)`: closures do not inherit target features.
    Some((V::load(&bs), V::load(&tc0)))
}

/// The common filterSamplesFlag test (§8.7.2.2): −1 where the line is
/// filtered.
#[allow(clippy::inline_always)]
#[inline(always)]
fn filter_flag<V: I16FilterLanes>(bs: V, [p1, p0, q0, q1]: [V; 4], alpha: V, beta: V) -> V {
    less(V::splat(0), bs)
        .and(less(p0.sub(q0).abs(), alpha))
        .and(less(p1.sub(p0).abs(), beta))
        .and(less(q1.sub(q0).abs(), beta))
}

/// The bS < 4 `delta` of p0/q0, clipped to `tc`.
#[allow(clippy::inline_always)]
#[inline(always)]
fn normal_delta<V: I16Lanes>([p1, p0, q0, q1]: [V; 4], tc: V) -> V {
    let delta = q0
        .sub(p0)
        .mul(V::splat(4))
        .add(p1.sub(q1))
        .add(V::splat(4))
        .sra(3);
    clip(delta, tc)
}

/// Lines per transposed group (an 8x8 tile of samples p3..q3).
const TRANSPOSED_LINES: usize = 8;

/// Positions p3..q3 of the vertical edge lines `first..first + 8`: eight
/// row loads and an 8x8 transpose.
#[allow(clippy::inline_always)]
#[inline(always)]
fn load_transposed<T: I16x8Shuffles>(plane: &[u8], e: &Edge, first: usize) -> [T; 8] {
    // A loop, not `array::from_fn`: closures do not inherit target features.
    let mut rows = [T::splat(0); 8];
    for (line, row) in rows.iter_mut().enumerate() {
        // SAFETY: inside the plane, as checked by the dispatcher (`index`).
        *row = unsafe { T::load_u8_at(plane.as_ptr().add(index(e, first + line, -4))) };
    }
    super::hevc::transpose(rows)
}

/// Store positions p3..q3 (clamped to 8 bits) of the vertical edge lines
/// `first..first + 8`: transpose back, one 8-byte store per row.
#[allow(clippy::inline_always)]
#[inline(always)]
fn store_transposed<T: I16x8Shuffles>(plane: &mut [u8], e: &Edge, first: usize, positions: [T; 8]) {
    let rows = super::hevc::transpose(positions);
    for (line, row) in rows.iter().enumerate() {
        // SAFETY: inside the plane, as checked by the dispatcher (`index`).
        unsafe { row.store_u8_sat_at(plane.as_mut_ptr().add(index(e, first + line, -4))) };
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn chroma<V: I16FilterLanes, T: I16x8Shuffles + I16FilterLanes>(
    plane: &mut [u8],
    e: &Edge,
) {
    if e.along == 1 {
        let mut first = 0;
        while first < CHROMA_LINES {
            chroma_lines::<V>(plane, e, first);
            first += V::LANES;
        }
    } else {
        let mut first = 0;
        while first < CHROMA_LINES {
            chroma_lines_transposed::<T>(plane, e, first);
            first += TRANSPOSED_LINES;
        }
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn chroma_lines<V: I16FilterLanes>(plane: &mut [u8], e: &Edge, first: usize) {
    let Some((bs, tc0)) = strengths::<V>(e, first, 2) else {
        return;
    };
    let positions = load_lines::<V, 4>(plane, e, first);
    store_lines(plane, e, first, -1, chroma_filter(e, bs, tc0, positions));
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn chroma_lines_transposed<T: I16x8Shuffles + I16FilterLanes>(
    plane: &mut [u8],
    e: &Edge,
    first: usize,
) {
    let Some((bs, tc0)) = strengths::<T>(e, first, 2) else {
        return;
    };
    let mut positions = load_transposed::<T>(plane, e, first);
    let [p0, q0] = chroma_filter(
        e,
        bs,
        tc0,
        [positions[2], positions[3], positions[4], positions[5]],
    );
    positions[3] = p0;
    positions[4] = q0;
    store_transposed(plane, e, first, positions);
}

/// The filtered p0 and q0 of chroma lines whose samples p1..q1 are
/// `positions` (§8.7.2.3, §8.7.2.4 with chromaStyleFilteringFlag).
#[allow(clippy::inline_always)]
#[inline(always)]
fn chroma_filter<V: I16FilterLanes>(e: &Edge, bs: V, tc0: V, [p1, p0, q0, q1]: [V; 4]) -> [V; 2] {
    let (alpha, beta) = (V::splat(e.alpha), V::splat(e.beta));
    let flag = filter_flag(bs, [p1, p0, q0, q1], alpha, beta);
    let two = V::splat(2);
    let delta = normal_delta([p1, p0, q0, q1], tc0.add(V::splat(1)));
    let is_four = less(V::splat(i16::from(STRONG_BS) - 1), bs);
    let new_p0 = is_four.blend(p1.mul(two).add(p0).add(q1).add(two).sra(2), p0.add(delta));
    let new_q0 = is_four.blend(q1.mul(two).add(q0).add(p1).add(two).sra(2), q0.sub(delta));
    [flag.blend(new_p0, p0), flag.blend(new_q0, q0)]
}
