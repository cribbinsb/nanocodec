//! AV1 loop filter (deblocking) of one 4-line edge segment (AV1 §7.14.6):
//! the narrow 4-tap filter, the 6-tap chroma, 8-tap and 14-tap luma
//! filters, and the per-line `filterMask`/`hevMask`/`flat`/`flat2`
//! decisions that choose between them.
//!
//! Scalar reference; the codec (nc-av1) decides which edges to filter, with
//! which length and level. Reimplemented from the normative filter
//! description (structure follows libaom `aom_dsp/loopfilter.c`,
//! BSD-2-Clause, see `THIRD_PARTY.md`).

/// Lines one call filters (an edge of one 4x4 unit).
pub const AV1_EDGE_LINES: usize = 4;
/// Largest number of samples a filter reads on each side of an edge (14-tap).
pub const AV1_EDGE_REACH: usize = 7;
/// `flat`/`flat2` threshold: samples within this of p0 (q0) count as flat.
const FLAT_THRESHOLD: i16 = 1;

/// Filter length of an edge, from the transform sizes on its two sides
/// (AV1 §7.14.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av1FilterLength {
    /// 4-tap: 4x4 transforms.
    Four,
    /// 6-tap: chroma edges between transforms of 8x8 and up.
    Six,
    /// 8-tap: luma 8x8 transforms.
    Eight,
    /// 14-tap: luma 16x16 transforms and up.
    Fourteen,
}

impl Av1FilterLength {
    /// Samples the filter reads on each side of the edge.
    #[must_use]
    pub fn reach(self) -> usize {
        match self {
            Self::Four => 2,
            Self::Six => 3,
            Self::Eight => 4,
            Self::Fourteen => AV1_EDGE_REACH,
        }
    }
}

/// One edge segment's filter: its length and the level's thresholds
/// (AV1 §7.14.4: `limit`, `blimit`, `thresh`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1EdgeFilter {
    /// Filter length (taps).
    pub length: Av1FilterLength,
    /// Largest step between neighbouring samples on one side (`limit`).
    pub limit: u8,
    /// Largest weighted step across the edge (`blimit`).
    pub blimit: u8,
    /// High-edge-variance threshold (`thresh`).
    pub thresh: u8,
}

/// Reference: filter the [`AV1_EDGE_LINES`] lines across one edge in
/// `plane`, one line at a time. `q0` is the index of the first line's q0
/// sample, `across` the index step from p0 to q0 (1 for a vertical edge,
/// the stride for a horizontal one) and `along` the step from one line to
/// the next.
#[cfg(test)]
pub(crate) fn filter_edge_reference(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    filter: Av1EdgeFilter,
) {
    let reach = filter.length.reach();
    for line in 0..AV1_EDGE_LINES {
        let base = q0 + line * along;
        // p[i] is the i-th sample before the edge, q[i] the i-th after.
        let mut p = [0_i32; AV1_EDGE_REACH];
        let mut q = [0_i32; AV1_EDGE_REACH];
        for i in 0..reach {
            p[i] = i32::from(plane[base - (i + 1) * across]);
            q[i] = i32::from(plane[base + i * across]);
        }
        let (new_p, new_q) = filter_line(&p, &q, filter);
        for i in 0..reach {
            plane[base - (i + 1) * across] = new_p[i] as u8;
            plane[base + i * across] = new_q[i] as u8;
        }
    }
}

/// One line: the filtered `p` and `q` samples.
#[cfg(test)]
fn filter_line(
    p: &[i32; AV1_EDGE_REACH],
    q: &[i32; AV1_EDGE_REACH],
    filter: Av1EdgeFilter,
) -> ([i32; AV1_EDGE_REACH], [i32; AV1_EDGE_REACH]) {
    let (limit, blimit, thresh) = (
        i32::from(filter.limit),
        i32::from(filter.blimit),
        i32::from(filter.thresh),
    );
    let (mut np, mut nq) = (*p, *q);
    // Steps between neighbouring samples on each side, out to `taps`.
    let smooth = |taps: usize| {
        (1..taps).all(|i| (p[i] - p[i - 1]).abs() <= limit && (q[i] - q[i - 1]).abs() <= limit)
    };
    let across_edge = (p[0] - q[0]).abs() * 2 + (p[1] - q[1]).abs() / 2 <= blimit;
    // Samples `indices` away from the edge all within FLAT_THRESHOLD of p0/q0.
    let flat = |indices: &[usize]| {
        indices.iter().all(|&i| {
            (p[i] - p[0]).abs() <= i32::from(FLAT_THRESHOLD)
                && (q[i] - q[0]).abs() <= i32::from(FLAT_THRESHOLD)
        })
    };
    match filter.length {
        Av1FilterLength::Four => {
            if smooth(2) && across_edge {
                narrow(p, q, thresh, &mut np, &mut nq);
            }
        }
        Av1FilterLength::Six => {
            if smooth(3) && across_edge {
                if flat(&[1, 2]) {
                    wide(p, q, &SIX_TAP, &mut np, &mut nq);
                } else {
                    narrow(p, q, thresh, &mut np, &mut nq);
                }
            }
        }
        Av1FilterLength::Eight | Av1FilterLength::Fourteen => {
            if smooth(4) && across_edge {
                let flat_inner = flat(&[1, 2, 3]);
                if filter.length == Av1FilterLength::Fourteen && flat_inner && flat(&[4, 5, 6]) {
                    wide(p, q, &FOURTEEN_TAP, &mut np, &mut nq);
                } else if flat_inner {
                    wide(p, q, &EIGHT_TAP, &mut np, &mut nq);
                } else {
                    narrow(p, q, thresh, &mut np, &mut nq);
                }
            }
        }
    }
    (np, nq)
}

#[cfg(test)]
/// Clamp to the signed 8-bit range the narrow filter works in.
fn clamp_signed(value: i32) -> i32 {
    value.clamp(-128, 127)
}

#[cfg(test)]
/// The narrow filter (AV1 §7.14.6.3): adjusts p1..q1, or only p0/q0 when
/// the edge has high variance.
fn narrow(
    p: &[i32; AV1_EDGE_REACH],
    q: &[i32; AV1_EDGE_REACH],
    thresh: i32,
    np: &mut [i32; AV1_EDGE_REACH],
    nq: &mut [i32; AV1_EDGE_REACH],
) {
    const OFFSET: i32 = 128;
    let (ps1, ps0, qs0, qs1) = (p[1] - OFFSET, p[0] - OFFSET, q[0] - OFFSET, q[1] - OFFSET);
    let hev = (p[1] - p[0]).abs() > thresh || (q[1] - q[0]).abs() > thresh;
    let outer = if hev { clamp_signed(ps1 - qs1) } else { 0 };
    let base = clamp_signed(outer + 3 * (qs0 - ps0));
    // Round one side +4 and the other +3, so a base of 4 moves q0 by 1 and
    // p0 by 0 rather than both rounding the same way.
    let filter1 = clamp_signed(base + 4) >> 3;
    let filter2 = clamp_signed(base + 3) >> 3;
    nq[0] = clamp_signed(qs0 - filter1) + OFFSET;
    np[0] = clamp_signed(ps0 + filter2) + OFFSET;
    if !hev {
        let tap = (filter1 + 1) >> 1;
        nq[1] = clamp_signed(qs1 - tap) + OFFSET;
        np[1] = clamp_signed(ps1 + tap) + OFFSET;
    }
}

#[cfg(test)]
/// A wide (smoothing) filter: output `k` on the p side is
/// `Σ weights[k][j] · line[j] >> shift`, rounded, over the line
/// `p[n-1]..p[0], q[0]..q[n-1]`; the q side mirrors it.
struct WideFilter<const TAPS: usize, const OUTPUTS: usize> {
    /// Samples read on each side.
    reach: usize,
    /// Weights of p[OUTPUTS-1]..p[0] (rows), over p[reach-1]..q[reach-1].
    weights: [[i32; TAPS]; OUTPUTS],
    shift: u32,
}

#[cfg(test)]
/// 6-tap chroma: 5-tap [1, 2, 2, 2, 1] smoothing of p1..q1 (AV1 §7.14.6.4).
const SIX_TAP: WideFilter<6, 2> = WideFilter {
    reach: 3,
    weights: [[3, 2, 2, 1, 0, 0], [1, 2, 2, 2, 1, 0]],
    shift: 3,
};

#[cfg(test)]
/// 8-tap luma: 7-tap [1, 1, 1, 2, 1, 1, 1] smoothing of p2..q2.
const EIGHT_TAP: WideFilter<8, 3> = WideFilter {
    reach: 4,
    weights: [
        [3, 2, 1, 1, 1, 0, 0, 0],
        [2, 1, 2, 1, 1, 1, 0, 0],
        [1, 1, 1, 2, 1, 1, 1, 0],
    ],
    shift: 3,
};

#[cfg(test)]
/// 14-tap luma: 13-tap [1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1] smoothing of
/// p5..q5.
const FOURTEEN_TAP: WideFilter<14, 6> = WideFilter {
    reach: 7,
    weights: [
        [7, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [5, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [4, 1, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [3, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0],
        [2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 0],
    ],
    shift: 4,
};

#[cfg(test)]
/// Apply `filter`: rows are the outputs p[OUTPUTS-1]..p[0]; the q side uses
/// the same rows on the mirrored line.
fn wide<const TAPS: usize, const OUTPUTS: usize>(
    p: &[i32; AV1_EDGE_REACH],
    q: &[i32; AV1_EDGE_REACH],
    filter: &WideFilter<TAPS, OUTPUTS>,
    np: &mut [i32; AV1_EDGE_REACH],
    nq: &mut [i32; AV1_EDGE_REACH],
) {
    let n = filter.reach;
    // line[j]: p[n-1-j] for j < n, else q[j-n]; mirrored swaps p and q.
    let line = |j: usize, first: &[i32; AV1_EDGE_REACH], second: &[i32; AV1_EDGE_REACH]| {
        if j < n {
            first[n - 1 - j]
        } else {
            second[j - n]
        }
    };
    let round = 1 << (filter.shift - 1);
    for (row, weights) in filter.weights.iter().enumerate() {
        // Row `row` produces p[OUTPUTS-1-row] (and mirrored q).
        let target = OUTPUTS - 1 - row;
        let sum_p: i32 = (0..TAPS).map(|j| weights[j] * line(j, p, q)).sum();
        let sum_q: i32 = (0..TAPS).map(|j| weights[j] * line(j, q, p)).sum();
        np[target] = (sum_p + round) >> filter.shift;
        nq[target] = (sum_q + round) >> filter.shift;
    }
}

/// Lines the batched kernel filters together: one vector of 16-bit lanes
/// per tap, so the compiler can process the lines in parallel.
const BATCH: usize = 16;
/// Signed offset of the narrow filter's 8-bit arithmetic.
const SIGNED_OFFSET: i16 = 128;

/// One tap (sample position) across all lines of a batch.
type Lanes = [i16; BATCH];

/// Filter `lines` (a multiple of [`AV1_EDGE_LINES`]) lines across one edge
/// in `plane`, `BATCH` lines at a time; addressing as
/// [`filter_edge_reference`]. Same output as filtering each line alone: the
/// lines are independent.
pub(crate) fn filter_edge(
    plane: &mut [u8],
    q0: usize,
    across: usize,
    along: usize,
    lines: usize,
    filter: Av1EdgeFilter,
) {
    let reach = filter.length.reach();
    let mut done = 0;
    while done < lines {
        let n = (lines - done).min(BATCH);
        let base = q0 + done * along;
        let mut p = [[0_i16; BATCH]; AV1_EDGE_REACH];
        let mut q = [[0_i16; BATCH]; AV1_EDGE_REACH];
        for i in 0..reach {
            for l in 0..n {
                p[i][l] = i16::from(plane[base + l * along - (i + 1) * across]);
                q[i][l] = i16::from(plane[base + l * along + i * across]);
            }
        }
        let (np, nq) = filter_batch(&p, &q, filter);
        for i in 0..reach {
            for l in 0..n {
                plane[base + l * along - (i + 1) * across] = np[i][l] as u8;
                plane[base + l * along + i * across] = nq[i][l] as u8;
            }
        }
        done += n;
    }
}

/// All decisions and filters of one batch. Every lane computes every
/// candidate and a 0/1 lane mask selects (`old + mask · (new − old)`), so
/// the loops have no branches and the compiler vectorises them.
fn filter_batch(
    p: &[Lanes; AV1_EDGE_REACH],
    q: &[Lanes; AV1_EDGE_REACH],
    filter: Av1EdgeFilter,
) -> ([Lanes; AV1_EDGE_REACH], [Lanes; AV1_EDGE_REACH]) {
    let (limit, blimit, thresh) = (
        i16::from(filter.limit),
        i16::from(filter.blimit),
        i16::from(filter.thresh),
    );
    let taps = match filter.length {
        Av1FilterLength::Four => 2,
        Av1FilterLength::Six => 3,
        Av1FilterLength::Eight | Av1FilterLength::Fourteen => 4,
    };
    // mask: filter at all; flat: smooth taps 1..taps; flat2: taps 4..7.
    let mut mask = [1_i16; BATCH];
    let mut flat = [1_i16; BATCH];
    let mut flat2 = [1_i16; BATCH];
    for i in 1..taps {
        for l in 0..BATCH {
            let steps = (p[i][l] - p[i - 1][l])
                .abs()
                .max((q[i][l] - q[i - 1][l]).abs());
            mask[l] &= i16::from(steps <= limit);
            let spread = (p[i][l] - p[0][l]).abs().max((q[i][l] - q[0][l]).abs());
            flat[l] &= i16::from(spread <= FLAT_THRESHOLD);
        }
    }
    for l in 0..BATCH {
        let across = (p[0][l] - q[0][l]).abs() * 2 + ((p[1][l] - q[1][l]).abs() >> 1);
        mask[l] &= i16::from(across <= blimit);
    }
    if filter.length == Av1FilterLength::Fourteen {
        for i in 4..AV1_EDGE_REACH {
            for l in 0..BATCH {
                let spread = (p[i][l] - p[0][l]).abs().max((q[i][l] - q[0][l]).abs());
                flat2[l] &= i16::from(spread <= FLAT_THRESHOLD);
            }
        }
    }
    let (mut np, mut nq) = (*p, *q);
    narrow_batch(p, q, thresh, &mask, &mut np, &mut nq);
    let both: Lanes = std::array::from_fn(|l| mask[l] & flat[l]);
    match filter.length {
        Av1FilterLength::Four => {}
        Av1FilterLength::Six => six_batch(p, q, &both, &mut np, &mut nq),
        Av1FilterLength::Eight => eight_batch(p, q, &both, &mut np, &mut nq),
        Av1FilterLength::Fourteen => {
            eight_batch(p, q, &both, &mut np, &mut nq);
            let all: Lanes = std::array::from_fn(|l| both[l] & flat2[l]);
            fourteen_batch(p, q, &all, &mut np, &mut nq);
        }
    }
    (np, nq)
}

/// `old + select · (new − old)` with `select` 0 or 1.
fn blend(old: i16, new: i16, select: i16) -> i16 {
    old + select * (new - old)
}

/// The narrow filter (§7.14.6.3) where `mask` is 1.
fn narrow_batch(
    p: &[Lanes; AV1_EDGE_REACH],
    q: &[Lanes; AV1_EDGE_REACH],
    thresh: i16,
    mask: &Lanes,
    np: &mut [Lanes; AV1_EDGE_REACH],
    nq: &mut [Lanes; AV1_EDGE_REACH],
) {
    let clamp = |v: i16| v.clamp(-128, 127);
    for l in 0..BATCH {
        let (ps1, ps0) = (p[1][l] - SIGNED_OFFSET, p[0][l] - SIGNED_OFFSET);
        let (qs0, qs1) = (q[0][l] - SIGNED_OFFSET, q[1][l] - SIGNED_OFFSET);
        let hev = i16::from((p[1][l] - p[0][l]).abs().max((q[1][l] - q[0][l]).abs()) > thresh);
        let base = clamp(hev * clamp(ps1 - qs1) + 3 * (qs0 - ps0));
        let filter1 = clamp(base + 4) >> 3;
        let filter2 = clamp(base + 3) >> 3;
        let tap = (1 - hev) * ((filter1 + 1) >> 1);
        let m = mask[l];
        nq[0][l] = blend(q[0][l], clamp(qs0 - filter1) + SIGNED_OFFSET, m);
        np[0][l] = blend(p[0][l], clamp(ps0 + filter2) + SIGNED_OFFSET, m);
        nq[1][l] = blend(q[1][l], clamp(qs1 - tap) + SIGNED_OFFSET, m);
        np[1][l] = blend(p[1][l], clamp(ps1 + tap) + SIGNED_OFFSET, m);
    }
}

/// 6-tap chroma [1, 2, 2, 2, 1] where `select` is 1.
fn six_batch(
    p: &[Lanes; AV1_EDGE_REACH],
    q: &[Lanes; AV1_EDGE_REACH],
    select: &Lanes,
    np: &mut [Lanes; AV1_EDGE_REACH],
    nq: &mut [Lanes; AV1_EDGE_REACH],
) {
    for l in 0..BATCH {
        let (p2, p1, p0) = (p[2][l], p[1][l], p[0][l]);
        let (q0, q1, q2) = (q[0][l], q[1][l], q[2][l]);
        let s = select[l];
        np[1][l] = blend(np[1][l], (p2 * 3 + p1 * 2 + p0 * 2 + q0 + 4) >> 3, s);
        np[0][l] = blend(np[0][l], (p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + 4) >> 3, s);
        nq[0][l] = blend(nq[0][l], (p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + 4) >> 3, s);
        nq[1][l] = blend(nq[1][l], (p0 + q0 * 2 + q1 * 2 + q2 * 3 + 4) >> 3, s);
    }
}

/// 8-tap luma [1, 1, 1, 2, 1, 1, 1] where `select` is 1.
fn eight_batch(
    p: &[Lanes; AV1_EDGE_REACH],
    q: &[Lanes; AV1_EDGE_REACH],
    select: &Lanes,
    np: &mut [Lanes; AV1_EDGE_REACH],
    nq: &mut [Lanes; AV1_EDGE_REACH],
) {
    for l in 0..BATCH {
        let (p3, p2, p1, p0) = (p[3][l], p[2][l], p[1][l], p[0][l]);
        let (q0, q1, q2, q3) = (q[0][l], q[1][l], q[2][l], q[3][l]);
        let s = select[l];
        np[2][l] = blend(np[2][l], (3 * p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3, s);
        np[1][l] = blend(np[1][l], (2 * p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3, s);
        np[0][l] = blend(np[0][l], (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3, s);
        nq[0][l] = blend(nq[0][l], (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3, s);
        nq[1][l] = blend(nq[1][l], (p1 + p0 + q0 + 2 * q1 + q2 + 2 * q3 + 4) >> 3, s);
        nq[2][l] = blend(nq[2][l], (p0 + q0 + q1 + 2 * q2 + 3 * q3 + 4) >> 3, s);
    }
}

/// 14-tap luma [1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1] where `select` is 1
/// (overriding the 8-tap result).
fn fourteen_batch(
    p: &[Lanes; AV1_EDGE_REACH],
    q: &[Lanes; AV1_EDGE_REACH],
    select: &Lanes,
    np: &mut [Lanes; AV1_EDGE_REACH],
    nq: &mut [Lanes; AV1_EDGE_REACH],
) {
    for side in 0..2 {
        let (near, far) = if side == 0 { (p, q) } else { (q, p) };
        let out = if side == 0 { &mut *np } else { &mut *nq };
        for l in 0..BATCH {
            let (p6, p5, p4, p3, p2, p1, p0) = (
                near[6][l], near[5][l], near[4][l], near[3][l], near[2][l], near[1][l], near[0][l],
            );
            let (q0, q1, q2, q3, q4, q5) = (
                far[0][l], far[1][l], far[2][l], far[3][l], far[4][l], far[5][l],
            );
            let s = select[l];
            let new = [
                (p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8)
                    >> 4,
                (p6 * 2 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8)
                    >> 4,
                (p6 * 3 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8) >> 4,
                (p6 * 4 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4,
                (p6 * 5 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8) >> 4,
                (p6 * 7 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8) >> 4,
            ];
            for (i, &v) in new.iter().enumerate() {
                out[i][l] = blend(out[i][l], v, s);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// libaom's explicit 13-tap formulas for p5..p0 (`filter14`), as the
    /// independent oracle of the weight table.
    fn libaom_fourteen(p: &[i32; 7], q: &[i32; 7]) -> [i32; 6] {
        let r = |v: i32| (v + 8) >> 4;
        let (p6, p5, p4, p3, p2, p1, p0) = (p[6], p[5], p[4], p[3], p[2], p[1], p[0]);
        let (q0, q1, q2, q3, q4, q5) = (q[0], q[1], q[2], q[3], q[4], q[5]);
        [
            r(p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5),
            r(p6 * 2 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4),
            r(p6 * 3 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3),
            r(p6 * 4 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2),
            r(p6 * 5 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1),
            r(p6 * 7 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0),
        ]
    }

    #[test]
    fn wide_tables_match_the_explicit_formulas() {
        let p = [100, 101, 99, 102, 98, 100, 101];
        let q = [104, 103, 105, 102, 106, 104, 103];
        let (mut np, mut nq) = (p, q);
        wide(&p, &q, &FOURTEEN_TAP, &mut np, &mut nq);
        let want = libaom_fourteen(&p, &q);
        for (i, &w) in want.iter().enumerate() {
            assert_eq!(np[i], w, "p{i}");
        }
        // 8-tap op2..op0 per libaom filter8.
        let r = |v: i32| (v + 4) >> 3;
        let (mut np, mut nq) = (p, q);
        wide(&p, &q, &EIGHT_TAP, &mut np, &mut nq);
        let (p3, p2, p1, p0, q0, q1, q2, q3) = (p[3], p[2], p[1], p[0], q[0], q[1], q[2], q[3]);
        assert_eq!(np[2], r(3 * p3 + 2 * p2 + p1 + p0 + q0));
        assert_eq!(np[1], r(2 * p3 + p2 + 2 * p1 + p0 + q0 + q1));
        assert_eq!(np[0], r(p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2));
        assert_eq!(nq[0], r(p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3));
        // 6-tap op1, op0.
        let (mut np, mut nq) = (p, q);
        wide(&p, &q, &SIX_TAP, &mut np, &mut nq);
        assert_eq!(np[1], r(p2 * 3 + p1 * 2 + p0 * 2 + q0));
        assert_eq!(np[0], r(p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1));
        assert_eq!(nq[1], r(p0 + q0 * 2 + q1 * 2 + q[2] * 3));
    }

    #[test]
    fn batched_kernel_matches_the_line_reference() {
        let mut state = 0x1234_5678_u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        for length in [
            Av1FilterLength::Four,
            Av1FilterLength::Six,
            Av1FilterLength::Eight,
            Av1FilterLength::Fourteen,
        ] {
            for level in [1_u8, 8, 20, 40, 63] {
                let filter = Av1EdgeFilter {
                    length,
                    limit: (level >> 1).max(1),
                    blimit: 2 * (level + 2) + (level >> 1).max(1),
                    thresh: level >> 4,
                };
                for vertical in [true, false] {
                    // 24 lines of a 32x32 plane; smooth-ish data with steps.
                    let stride = 32;
                    let base: Vec<u8> = (0..stride * stride)
                        .map(|i| 100 + (next() % 9) + if i % stride >= 16 { 6 } else { 0 })
                        .collect();
                    let (q0, across, along) = if vertical {
                        (16 + stride * 4, 1, stride)
                    } else {
                        (4 + stride * 16, stride, 1)
                    };
                    let mut fast = base.clone();
                    filter_edge(&mut fast, q0, across, along, 24, filter);
                    let mut slow = base.clone();
                    for seg in 0..6 {
                        filter_edge_reference(
                            &mut slow,
                            q0 + seg * 4 * along,
                            across,
                            along,
                            filter,
                        );
                    }
                    assert_eq!(fast, slow, "{length:?} level {level} vertical {vertical}");
                }
            }
        }
    }

    #[test]
    fn narrow_filter_moves_a_small_step_and_keeps_a_strong_edge() {
        let filter = |length| Av1EdgeFilter {
            length,
            limit: 4,
            blimit: 20,
            thresh: 1,
        };
        // A small step (100 | 104) on flat sides is smoothed.
        let mut plane = [100, 100, 100, 100, 104, 104, 104, 104];
        let mut lines = [0u8; 8 * AV1_EDGE_LINES];
        for l in 0..AV1_EDGE_LINES {
            lines[l * 8..l * 8 + 8].copy_from_slice(&plane);
        }
        filter_edge(
            &mut lines,
            4,
            1,
            8,
            AV1_EDGE_LINES,
            filter(Av1FilterLength::Eight),
        );
        assert!(lines[3] > 100 && lines[4] < 104, "{:?}", &lines[..8]);
        // A strong edge (100 | 160) exceeds blimit and is left alone.
        plane[4..].fill(160);
        for l in 0..AV1_EDGE_LINES {
            lines[l * 8..l * 8 + 8].copy_from_slice(&plane);
        }
        filter_edge(
            &mut lines,
            4,
            1,
            8,
            AV1_EDGE_LINES,
            filter(Av1FilterLength::Four),
        );
        assert_eq!(&lines[..8], &plane);
    }
}
