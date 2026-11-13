//! Scalar deblocking: the per-line references (and the kernels of builds
//! without SIMD), written directly from the standards: H.264 §8.7.2.3 /
//! §8.7.2.4, HEVC §8.7.2.5.3, .6, .7 and §8.7.2.5.5.

use super::{CHROMA_LINES, DeblockFns, Edge, HEVC_LINES, HEVC_SEGMENT, HevcEdge, LUMA_LINES};

/// The scalar table.
pub(crate) fn deblock_fns() -> DeblockFns {
    DeblockFns {
        luma: luma_edge,
        chroma: chroma_edge,
        chroma_pair,
        hevc_luma: hevc_luma_edge,
        hevc_chroma: hevc_chroma_edge,
        av2_ramps: super::av2::apply_ramps,
        av1_edge: super::av1::filter_edge,
        av1_line: super::av1_line::scalar,
        av1_grid: super::av1_grid::scalar,
        av1_runs: super::runs::av1_scalar,
        hevc_luma_runs: super::runs::hevc_luma_scalar,
        hevc_chroma_runs: super::runs::hevc_chroma_scalar,
        av2_chunk: super::av2::filter_chunk,
    }
}

/// Index of the sample `offset` (p3 = −4 … q3 = 3) across the edge from
/// q0 of the line starting at `q0`.
fn at(q0: usize, across: usize, offset: isize) -> usize {
    q0.checked_add_signed(offset * across as isize)
        .expect("deblocking sample inside the plane")
}

fn luma_edge(plane: &mut [u8], e: &Edge) {
    let (alpha, beta) = (i32::from(e.alpha), i32::from(e.beta));
    for line in 0..LUMA_LINES {
        let bs = e.bs[line / 4];
        if bs == 0 {
            continue;
        }
        let q0i = e.base + line * e.along;
        let s: [i32; 8] =
            core::array::from_fn(|i| i32::from(plane[at(q0i, e.across, i as isize - 4)]));
        let [p3, p2, p1, p0, q0, q1, q2, q3] = s;
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        let (ap, aq) = ((p2 - p0).abs(), (q2 - q0).abs());
        let mut out = s;
        if bs < 4 {
            let tc0 = i32::from(e.tc0[line / 4]);
            let tc = tc0 + i32::from(ap < beta) + i32::from(aq < beta);
            let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            out[3] = p0 + delta;
            out[4] = q0 - delta;
            if ap < beta {
                out[2] = p1 + ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0);
            }
            if aq < beta {
                out[5] = q1 + ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0);
            }
        } else {
            let strong = (p0 - q0).abs() < ((alpha >> 2) + 2);
            if ap < beta && strong {
                out[3] = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3;
                out[2] = (p2 + p1 + p0 + q0 + 2) >> 2;
                out[1] = (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3;
            } else {
                out[3] = (2 * p1 + p0 + q1 + 2) >> 2;
            }
            if aq < beta && strong {
                out[4] = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3;
                out[5] = (p0 + q0 + q1 + q2 + 2) >> 2;
                out[6] = (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3;
            } else {
                out[4] = (2 * q1 + q0 + p1 + 2) >> 2;
            }
        }
        // p2..q2: the only samples the luma filters change.
        for i in 1..7 {
            plane[at(q0i, e.across, i as isize - 4)] = out[i].clamp(0, 255) as u8;
        }
    }
}

/// [`chroma_edge`] in each plane.
fn chroma_pair(u: &mut [u8], v: &mut [u8], e: &Edge) {
    chroma_edge(u, e);
    chroma_edge(v, e);
}

fn chroma_edge(plane: &mut [u8], e: &Edge) {
    let (alpha, beta) = (i32::from(e.alpha), i32::from(e.beta));
    for line in 0..CHROMA_LINES {
        let bs = e.bs[line / 2];
        if bs == 0 {
            continue;
        }
        let q0i = e.base + line * e.along;
        let s: [i32; 4] =
            core::array::from_fn(|i| i32::from(plane[at(q0i, e.across, i as isize - 2)]));
        let [p1, p0, q0, q1] = s;
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        let (p0n, q0n) = if bs < 4 {
            let tc = i32::from(e.tc0[line / 2]) + 1;
            let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            (p0 + delta, q0 - delta)
        } else {
            ((2 * p1 + p0 + q1 + 2) >> 2, (2 * q1 + q0 + p1 + 2) >> 2)
        };
        plane[at(q0i, e.across, -1)] = p0n.clamp(0, 255) as u8;
        plane[at(q0i, e.across, 0)] = q0n.clamp(0, 255) as u8;
    }
}

/// The 8 samples p3..q3 of the HEVC line whose q0 is at `q0`.
fn hevc_line(plane: &[u8], q0: usize, across: usize) -> [i32; 8] {
    core::array::from_fn(|i| i32::from(plane[at(q0, across, i as isize - 4)]))
}

/// HEVC luma (§8.7.2.5.3 decisions, §8.7.2.5.6 / .7 filtering), one
/// 4-line segment at a time.
pub(super) fn hevc_luma_edge(plane: &mut [u8], e: &HevcEdge) {
    let beta = i32::from(e.beta);
    for segment in 0..HEVC_LINES / HEVC_SEGMENT {
        let tc = i32::from(e.tc[segment]);
        if tc == 0 {
            continue;
        }
        let first = e.base + segment * HEVC_SEGMENT * e.along;
        let last = first + (HEVC_SEGMENT - 1) * e.along;
        let [p30, p20, p10, p00, q00, q10, q20, q30] = hevc_line(plane, first, e.across);
        let [p33, p23, p13, p03, q03, q13, q23, q33] = hevc_line(plane, last, e.across);
        let dp0 = (p20 - 2 * p10 + p00).abs();
        let dp3 = (p23 - 2 * p13 + p03).abs();
        let dq0 = (q20 - 2 * q10 + q00).abs();
        let dq3 = (q23 - 2 * q13 + q03).abs();
        let (dpq0, dpq3) = (dp0 + dq0, dp3 + dq3);
        if dpq0 + dpq3 >= beta {
            continue;
        }
        let strong_line = |dpq: i32, p0: i32, p3: i32, q0: i32, q3: i32| {
            2 * dpq < (beta >> 2)
                && (p3 - p0).abs() + (q0 - q3).abs() < (beta >> 3)
                && (p0 - q0).abs() < ((5 * tc + 1) >> 1)
        };
        let strong = strong_line(dpq0, p00, p30, q00, q30) && strong_line(dpq3, p03, p33, q03, q33);
        let side_threshold = (beta + (beta >> 1)) >> 3;
        let filter_p1 = dp0 + dp3 < side_threshold;
        let filter_q1 = dq0 + dq3 < side_threshold;
        for k in 0..HEVC_SEGMENT {
            let q0i = first + k * e.along;
            let s = hevc_line(plane, q0i, e.across);
            let [p3, p2, p1, p0, q0, q1, q2, q3] = s;
            let mut out = s;
            if strong {
                let c = |v: i32, orig: i32| v.clamp(orig - 2 * tc, orig + 2 * tc);
                out[3] = c((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3, p0);
                out[2] = c((p2 + p1 + p0 + q0 + 2) >> 2, p1);
                out[1] = c((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3, p2);
                out[4] = c((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3, q0);
                out[5] = c((p0 + q0 + q1 + q2 + 2) >> 2, q1);
                out[6] = c((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3, q2);
            } else {
                let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
                if delta.abs() >= tc * 10 {
                    continue;
                }
                let delta = delta.clamp(-tc, tc);
                out[3] = p0 + delta;
                out[4] = q0 - delta;
                let half = tc >> 1;
                if filter_p1 {
                    out[2] = p1 + ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-half, half);
                }
                if filter_q1 {
                    out[5] = q1 + ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-half, half);
                }
            }
            for (i, &v) in out.iter().enumerate() {
                plane[at(q0i, e.across, i as isize - 4)] = v.clamp(0, 255) as u8;
            }
        }
    }
}

/// HEVC chroma (§8.7.2.5.5): p0 and q0 of each line with a non-zero `tC`.
pub(super) fn hevc_chroma_edge(plane: &mut [u8], e: &HevcEdge) {
    for line in 0..e.lines {
        let tc = i32::from(e.tc[line / 2]);
        if tc == 0 {
            continue;
        }
        let q0i = e.base + line * e.along;
        let [p1, p0, q0, q1] =
            core::array::from_fn(|i| i32::from(plane[at(q0i, e.across, i as isize - 2)]));
        let delta = ((((q0 - p0) << 2) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
        plane[at(q0i, e.across, -1)] = (p0 + delta).clamp(0, 255) as u8;
        plane[at(q0i, e.across, 0)] = (q0 - delta).clamp(0, 255) as u8;
    }
}
