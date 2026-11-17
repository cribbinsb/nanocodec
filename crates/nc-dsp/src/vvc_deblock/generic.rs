//! VVC filtering across four edge lines at once. The normative edge decision
//! is shared with scalar; arithmetic and clipping operate on independent lanes.
#![allow(unsafe_code, clippy::inline_always)]
use nc_common::BitDepth;

use super::LumaFilter;
use crate::simd::I32Lanes;

#[inline(always)]
fn clip<V: I32Lanes>(value: V, original: V, limit: i32, max_val: i32) -> V {
    value
        .max(original.sub(V::splat(limit)))
        .min(original.add(V::splat(limit)))
        .max(V::splat(0))
        .min(V::splat(max_val))
}
#[inline(always)]
fn round<V: I32Lanes>(value: V, bits: u32) -> V {
    value.add(V::splat(1 << (bits - 1))).sra(bits)
}
#[inline(always)]
fn sum<V: I32Lanes>(values: &[V]) -> V {
    let mut s = V::splat(0);
    for &v in values {
        s = s.add(v);
    }
    s
}

#[inline(always)]
fn luma_core<V: I32Lanes>(
    columns: [V; 16],
    decision: LumaFilter,
    tc: i32,
    lengths: [usize; 2],
    max_val: i32,
) -> [V; 16] {
    let p = [
        columns[7], columns[6], columns[5], columns[4], columns[3], columns[2], columns[1],
        columns[0],
    ];
    let q = [
        columns[8],
        columns[9],
        columns[10],
        columns[11],
        columns[12],
        columns[13],
        columns[14],
        columns[15],
    ];
    let mut out = columns;
    match decision {
        LumaFilter::Long => long(&p, &q, tc, lengths, max_val, &mut out),
        LumaFilter::Short { strong: true, .. } => strong(&p, &q, tc, max_val, &mut out),
        LumaFilter::Short { second, .. } => weak(&p, &q, tc, second, max_val, &mut out),
        LumaFilter::None => {}
    }
    out
}

/// # Safety
/// V's ISA must be enabled; V has four lanes.
#[inline(always)]
pub(super) unsafe fn luma<V: I32Lanes>(
    edge: &mut [[u8; 16]; 4],
    beta: i32,
    tc: i32,
    lengths: [usize; 2],
) {
    let decision = super::decision(edge, beta, tc, lengths);
    if matches!(decision, LumaFilter::None) {
        return;
    }
    let mut columns = [V::splat(0); 16];
    for (i, col) in columns.iter_mut().enumerate() {
        *col = V::load(&[
            i32::from(edge[0][i]),
            i32::from(edge[1][i]),
            i32::from(edge[2][i]),
            i32::from(edge[3][i]),
        ]);
    }
    let out = luma_core(columns, decision, tc, lengths, 255);
    for (i, value) in out.iter().enumerate() {
        let mut values = [0; 4];
        value.store(&mut values);
        for (row, v) in edge.iter_mut().zip(values) {
            row[i] = v as u8;
        }
    }
}

/// # Safety
/// V's ISA must be enabled; V has four lanes.
#[inline(always)]
pub(super) unsafe fn luma_u16<V: I32Lanes>(
    edge: &mut [[u16; 16]; 4],
    beta: i32,
    tc: i32,
    lengths: [usize; 2],
    depth: BitDepth,
) {
    let decision = super::decision(edge, beta, tc, lengths);
    if matches!(decision, LumaFilter::None) {
        return;
    }
    let mut columns = [V::splat(0); 16];
    for (i, col) in columns.iter_mut().enumerate() {
        *col = V::load(&[
            i32::from(edge[0][i]),
            i32::from(edge[1][i]),
            i32::from(edge[2][i]),
            i32::from(edge[3][i]),
        ]);
    }
    let out = luma_core(columns, decision, tc, lengths, depth.max());
    for (i, value) in out.iter().enumerate() {
        let mut values = [0; 4];
        value.store(&mut values);
        for (row, v) in edge.iter_mut().zip(values) {
            row[i] = v as u16;
        }
    }
}

#[inline(always)]
fn long<V: I32Lanes>(
    p: &[V; 8],
    q: &[V; 8],
    tc: i32,
    lengths: [usize; 2],
    max_val: i32,
    out: &mut [V; 16],
) {
    let middle = match (lengths[0].max(lengths[1]), lengths[0].min(lengths[1])) {
        (7, 7) => round(
            p[0].add(q[0])
                .mul_scalar(2)
                .add(sum(&p[1..7]))
                .add(sum(&q[1..7])),
            4,
        ),
        (5, 5) => round(
            sum(&p[..3])
                .add(sum(&q[..3]))
                .mul_scalar(2)
                .add(p[3])
                .add(q[3])
                .add(p[4])
                .add(q[4]),
            4,
        ),
        (7, 5) => round(
            sum(&p[..2])
                .add(sum(&q[..2]))
                .mul_scalar(2)
                .add(sum(&p[2..6]))
                .add(sum(&q[2..6])),
            4,
        ),
        (5, 3) => round(sum(&p[..4]).add(sum(&q[..4])), 3),
        _ => {
            let (large, small) = if lengths[0] == 7 { (p, q) } else { (q, p) };
            round(
                large[0]
                    .add(small[0])
                    .mul_scalar(2)
                    .add(small[0])
                    .add(small[1].add(small[2]).mul_scalar(2))
                    .add(large[1])
                    .add(small[1])
                    .add(sum(&large[2..7])),
                4,
            )
        }
    };
    for (which, side) in [p, q].into_iter().enumerate() {
        let length = lengths[which];
        let far = round(side[length - 1].add(side[length]), 1);
        let weights: &[i32] = match length {
            7 => &[59, 50, 41, 32, 23, 14, 5],
            5 => &[58, 45, 32, 19, 6],
            _ => &[53, 32, 11],
        };
        let limits: &[i32] = if length == 3 {
            &[6, 4, 2]
        } else {
            &[6, 5, 4, 3, 2, 1, 1]
        };
        for i in 0..length {
            let value = round(
                middle
                    .mul_scalar(weights[i])
                    .add(far.mul_scalar(64 - weights[i])),
                6,
            );
            out[if which == 0 { 7 - i } else { 8 + i }] =
                clip(value, side[i], (tc * limits[i]) >> 1, max_val);
        }
    }
}

#[inline(always)]
fn strong<V: I32Lanes>(p: &[V; 8], q: &[V; 8], tc: i32, max_val: i32, out: &mut [V; 16]) {
    for (which, (near, far)) in [(p, q), (q, p)].into_iter().enumerate() {
        let values = [
            round(
                near[2]
                    .add(near[1].add(near[0]).add(far[0]).mul_scalar(2))
                    .add(far[1]),
                3,
            ),
            round(near[2].add(near[1]).add(near[0]).add(far[0]), 2),
            round(
                near[3]
                    .mul_scalar(2)
                    .add(near[2].mul_scalar(3))
                    .add(near[1])
                    .add(near[0])
                    .add(far[0]),
                3,
            ),
        ];
        for i in 0..3 {
            out[if which == 0 { 7 - i } else { 8 + i }] =
                clip(values[i], near[i], (3 - i as i32) * tc, max_val);
        }
    }
}

#[inline(always)]
fn weak<V: I32Lanes>(
    p: &[V; 8],
    q: &[V; 8],
    tc: i32,
    second: [bool; 2],
    max_val: i32,
    out: &mut [V; 16],
) {
    let delta = round(
        q[0].sub(p[0])
            .mul_scalar(9)
            .sub(q[1].sub(p[1]).mul_scalar(3)),
        4,
    );
    let enabled = V::splat(10 * tc).gt(delta.abs());
    let delta = delta.max(V::splat(-tc)).min(V::splat(tc));
    out[7] = enabled.select(clip(p[0].add(delta), p[0], tc, max_val), p[0]);
    out[8] = enabled.select(clip(q[0].sub(delta), q[0], tc, max_val), q[0]);
    for (which, side) in [p, q].into_iter().enumerate() {
        if second[which] {
            let signed = if which == 0 {
                delta
            } else {
                V::splat(0).sub(delta)
            };
            let correction = round(side[2].add(side[0]), 1)
                .sub(side[1])
                .add(signed)
                .sra(1)
                .max(V::splat(-(tc >> 1)))
                .min(V::splat(tc >> 1));
            out[if which == 0 { 6 } else { 9 }] = enabled.select(
                clip(side[1].add(correction), side[1], tc >> 1, max_val),
                side[1],
            );
        }
    }
}

#[inline(always)]
fn chroma_core<V: I32Lanes>(
    cols: [V; 8],
    strong_mode: bool,
    boundary: bool,
    tc: i32,
    max_val: i32,
) -> [V; 8] {
    let p = [cols[3], cols[2], cols[1], cols[0]];
    let q = [cols[4], cols[5], cols[6], cols[7]];
    let mut out = cols;
    if !strong_mode {
        let delta = round(q[0].sub(p[0]).mul_scalar(4).add(p[1]).sub(q[1]), 3)
            .max(V::splat(-tc))
            .min(V::splat(tc));
        out[3] = clip(p[0].add(delta), p[0], tc, max_val);
        out[4] = clip(q[0].sub(delta), q[0], tc, max_val);
    } else if boundary {
        let values = [
            p[1].mul_scalar(3)
                .add(p[0].mul_scalar(2))
                .add(q[0])
                .add(q[1])
                .add(q[2]),
            p[1].mul_scalar(2)
                .add(p[0])
                .add(q[0].mul_scalar(2))
                .add(q[1])
                .add(q[2])
                .add(q[3]),
            p[1].add(p[0])
                .add(q[0])
                .add(q[1].mul_scalar(2))
                .add(q[2])
                .add(q[3].mul_scalar(2)),
            p[0].add(q[0])
                .add(q[1])
                .add(q[2].mul_scalar(2))
                .add(q[3].mul_scalar(3)),
        ];
        for i in 0..4 {
            out[3 + i] = clip(round(values[i], 3), cols[3 + i], tc, max_val);
        }
    } else {
        for (which, (near, far)) in [(&p, &q), (&q, &p)].into_iter().enumerate() {
            let values = [
                sum(&near[1..4])
                    .add(near[0].mul_scalar(2))
                    .add(sum(&far[..3])),
                near[3]
                    .mul_scalar(2)
                    .add(near[2])
                    .add(near[1].mul_scalar(2))
                    .add(near[0])
                    .add(far[0])
                    .add(far[1]),
                near[3]
                    .mul_scalar(3)
                    .add(near[2].mul_scalar(2))
                    .add(near[1])
                    .add(near[0])
                    .add(far[0]),
            ];
            for i in 0..3 {
                out[if which == 0 { 3 - i } else { 4 + i }] =
                    clip(round(values[i], 3), near[i], tc, max_val);
            }
        }
    }
    out
}

/// # Safety
/// V's ISA must be enabled; V has four lanes (two chroma lines plus padding).
#[inline(always)]
pub(super) unsafe fn chroma<V: I32Lanes>(
    edge: &mut [[u8; 8]; 2],
    beta: i32,
    tc: i32,
    large: bool,
    boundary: bool,
) {
    let strong_mode = large && super::chroma_strong(edge, beta, tc, boundary);
    let mut cols = [V::splat(0); 8];
    for (i, col) in cols.iter_mut().enumerate() {
        *col = V::load(&[i32::from(edge[0][i]), i32::from(edge[1][i]), 0, 0]);
    }
    let out = chroma_core(cols, strong_mode, boundary, tc, 255);
    for (i, value) in out.iter().enumerate() {
        let mut values = [0; 4];
        value.store(&mut values);
        for (row, v) in edge.iter_mut().zip(values) {
            row[i] = v as u8;
        }
    }
}

/// # Safety
/// V's ISA must be enabled; V has four lanes (two chroma lines plus padding).
#[inline(always)]
pub(super) unsafe fn chroma_u16<V: I32Lanes>(
    edge: &mut [[u16; 8]; 2],
    beta: i32,
    tc: i32,
    large: bool,
    boundary: bool,
    depth: BitDepth,
) {
    let strong_mode = large && super::chroma_strong(edge, beta, tc, boundary);
    let mut cols = [V::splat(0); 8];
    for (i, col) in cols.iter_mut().enumerate() {
        *col = V::load(&[i32::from(edge[0][i]), i32::from(edge[1][i]), 0, 0]);
    }
    let out = chroma_core(cols, strong_mode, boundary, tc, depth.max());
    for (i, value) in out.iter().enumerate() {
        let mut values = [0; 4];
        value.store(&mut values);
        for (row, v) in edge.iter_mut().zip(values) {
            row[i] = v as u16;
        }
    }
}

macro_rules! kernels {
    ($(#[$attr:meta])*; $lanes:ty, $edge_lanes:ty) => {
        pub(super) fn kernels() -> super::Kernels {
            super::Kernels { luma, luma_edge, luma_segment, luma_extended, chroma, chroma_edge }
        }
        pub(super) fn kernels_u16() -> super::Kernels16 {
            (luma_edge_u16, luma_u16, chroma_edge_u16, chroma_u16)
        }
        $(#[$attr])*
        unsafe fn luma_edge(plane: &mut [u8], edge: &super::VvcLumaEdge) {
            // SAFETY: target feature enabled; the kernel bounds-checks every index.
            unsafe { super::inplace::luma::<$edge_lanes, 8, false>(plane, edge); }
        }
        $(#[$attr])*
        unsafe fn luma_segment(plane: &mut [u8], edge: &super::VvcLumaEdge) {
            // SAFETY: target feature enabled; the API checked four complete lines.
            unsafe { super::inplace::luma::<$edge_lanes, 4, true>(plane, edge); }
        }
        $(#[$attr])*
        unsafe fn luma_extended(plane: &mut [u8], edge: &super::VvcLumaEdge) {
            // SAFETY: target feature enabled; the API checked eight complete lines.
            unsafe { super::inplace::luma::<$edge_lanes, 8, true>(plane, edge); }
        }
        $(#[$attr])*
        unsafe fn luma(edge: &mut [[u8; 16]; 4], beta: i32, tc: i32, lengths: [usize; 2]) {
            // SAFETY: target feature enabled and the chosen lane type has four lanes.
            unsafe { super::generic::luma::<$lanes>(edge, beta, tc, lengths); }
        }
        $(#[$attr])*
        unsafe fn chroma_edge(plane: &mut [u8], edge: &super::ChromaEdge) {
            // SAFETY: target feature enabled; the API validated all samples.
            unsafe { super::chroma_edge::simd::<$edge_lanes>(plane, edge); }
        }
        $(#[$attr])*
        unsafe fn chroma(edge: &mut [[u8; 8]; 2], beta: i32, tc: i32, large: bool, boundary: bool) {
            // SAFETY: target feature enabled and the chosen lane type has four lanes.
            unsafe { super::generic::chroma::<$lanes>(edge, beta, tc, large, boundary); }
        }
        $(#[$attr])*
        unsafe fn luma_edge_u16(
            plane: &mut [u16],
            edge: &super::VvcLumaEdge,
            depth: nc_common::BitDepth,
        ) {
            // SAFETY: target feature enabled; the API checked `edge.lines` complete lines.
            unsafe {
                if edge.lines == super::EDGE_LINES {
                    super::inplace::luma_u16::<$edge_lanes, 8, true>(plane, edge, depth);
                } else {
                    super::inplace::luma_u16::<$edge_lanes, 4, true>(plane, edge, depth);
                }
            }
        }
        $(#[$attr])*
        unsafe fn luma_u16(
            edge: &mut [[u16; 16]; 4],
            beta: i32,
            tc: i32,
            lengths: [usize; 2],
            depth: nc_common::BitDepth,
        ) {
            // SAFETY: target feature enabled and the chosen lane type has four lanes.
            unsafe { super::generic::luma_u16::<$lanes>(edge, beta, tc, lengths, depth); }
        }
        $(#[$attr])*
        unsafe fn chroma_edge_u16(
            plane: &mut [u16],
            edge: &super::ChromaEdge,
            depth: nc_common::BitDepth,
        ) {
            // SAFETY: target feature enabled; the API validated all samples.
            unsafe { super::chroma_edge::simd_u16::<$edge_lanes>(plane, edge, depth); }
        }
        $(#[$attr])*
        unsafe fn chroma_u16(
            edge: &mut [[u16; 8]; 2],
            beta: i32,
            tc: i32,
            large: bool,
            boundary: bool,
            depth: nc_common::BitDepth,
        ) {
            // SAFETY: target feature enabled and the chosen lane type has four lanes.
            unsafe { super::generic::chroma_u16::<$lanes>(edge, beta, tc, large, boundary, depth); }
        }
    };
}
pub(crate) use kernels;
