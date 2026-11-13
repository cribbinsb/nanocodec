//! VVC filtering across four edge lines at once. The normative edge decision
//! is shared with scalar; arithmetic and clipping operate on independent lanes.
#![allow(unsafe_code, clippy::inline_always)]
use super::LumaFilter;
use crate::simd::I32Lanes;

#[inline(always)]
fn clip<V: I32Lanes>(value: V, original: V, limit: i32) -> V {
    value
        .max(original.sub(V::splat(limit)))
        .min(original.add(V::splat(limit)))
        .max(V::splat(0))
        .min(V::splat(255))
}
#[inline(always)]
fn round<V: I32Lanes>(value: V, bits: u32) -> V {
    value.add(V::splat(1 << (bits - 1))).sra(bits)
}
#[inline(always)]
fn sum<V: I32Lanes>(values: &[V]) -> V {
    values.iter().fold(V::splat(0), |s, &v| s.add(v))
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
    let columns: [V; 16] = std::array::from_fn(|i| V::load(&edge.map(|row| i32::from(row[i]))));
    let p: [V; 8] = std::array::from_fn(|i| columns[7 - i]);
    let q: [V; 8] = std::array::from_fn(|i| columns[8 + i]);
    let mut out = columns;
    match decision {
        LumaFilter::Long => long(&p, &q, tc, lengths, &mut out),
        LumaFilter::Short { strong: true, .. } => strong(&p, &q, tc, &mut out),
        LumaFilter::Short { second, .. } => weak(&p, &q, tc, second, &mut out),
        LumaFilter::None => unreachable!(),
    }
    for (i, value) in out.iter().enumerate() {
        let mut values = [0; 4];
        value.store(&mut values);
        for (row, value) in edge.iter_mut().zip(values) {
            row[i] = value as u8;
        }
    }
}
#[inline(always)]
fn long<V: I32Lanes>(p: &[V; 8], q: &[V; 8], tc: i32, lengths: [usize; 2], out: &mut [V; 16]) {
    let middle = if lengths == [7, 7] {
        round(
            p[0].add(q[0])
                .mul_scalar(2)
                .add(sum(&p[1..7]))
                .add(sum(&q[1..7])),
            4,
        )
    } else {
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
    };
    for (which, side) in [p, q].into_iter().enumerate() {
        let length = lengths[which];
        let far = round(side[length - 1].add(side[length]), 1);
        let weights: &[i32] = if length == 7 {
            &[59, 50, 41, 32, 23, 14, 5]
        } else {
            &[53, 32, 11]
        };
        let limits: &[i32] = if length == 7 {
            &[6, 5, 4, 3, 2, 1, 1]
        } else {
            &[6, 4, 2]
        };
        for i in 0..length {
            let value = round(
                middle
                    .mul_scalar(weights[i])
                    .add(far.mul_scalar(64 - weights[i])),
                6,
            );
            out[if which == 0 { 7 - i } else { 8 + i }] =
                clip(value, side[i], (tc * limits[i]) >> 1);
        }
    }
}
#[inline(always)]
fn strong<V: I32Lanes>(p: &[V; 8], q: &[V; 8], tc: i32, out: &mut [V; 16]) {
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
                clip(values[i], near[i], (3 - i as i32) * tc);
        }
    }
}
#[inline(always)]
fn weak<V: I32Lanes>(p: &[V; 8], q: &[V; 8], tc: i32, second: [bool; 2], out: &mut [V; 16]) {
    let delta = round(
        q[0].sub(p[0])
            .mul_scalar(9)
            .sub(q[1].sub(p[1]).mul_scalar(3)),
        4,
    );
    let enabled = V::splat(10 * tc).gt(delta.abs());
    let delta = delta.max(V::splat(-tc)).min(V::splat(tc));
    out[7] = enabled.select(clip(p[0].add(delta), p[0], tc), p[0]);
    out[8] = enabled.select(clip(q[0].sub(delta), q[0], tc), q[0]);
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
            out[if which == 0 { 6 } else { 9 }] =
                enabled.select(clip(side[1].add(correction), side[1], tc >> 1), side[1]);
        }
    }
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
    let strong = large && super::chroma_strong(edge, beta, tc, boundary);
    let cols: [V; 8] =
        std::array::from_fn(|i| V::load(&[i32::from(edge[0][i]), i32::from(edge[1][i]), 0, 0]));
    let p: [V; 4] = std::array::from_fn(|i| cols[3 - i]);
    let q: [V; 4] = std::array::from_fn(|i| cols[4 + i]);
    let mut out = cols;
    if !strong {
        let delta = round(q[0].sub(p[0]).mul_scalar(4).add(p[1]).sub(q[1]), 3)
            .max(V::splat(-tc))
            .min(V::splat(tc));
        out[3] = clip(p[0].add(delta), p[0], tc);
        out[4] = clip(q[0].sub(delta), q[0], tc);
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
            out[3 + i] = clip(round(values[i], 3), cols[3 + i], tc);
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
                    clip(round(values[i], 3), near[i], tc);
            }
        }
    }
    for (i, value) in out.iter().enumerate() {
        let mut values = [0; 4];
        value.store(&mut values);
        for (row, value) in edge.iter_mut().zip(values) {
            row[i] = value as u8;
        }
    }
}
macro_rules! kernels {
    ($(#[$attr:meta])*; $lanes:ty, $edge_lanes:ty) => {
        pub(super) fn kernels() -> super::Kernels { super::Kernels { luma, luma_edge, luma_segment, luma_extended, chroma, chroma_edge } }
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
        unsafe fn luma(edge: &mut [[u8;16];4], beta: i32, tc: i32, lengths: [usize;2]) {
            // SAFETY: target feature enabled and the chosen lane type has four lanes.
            unsafe { super::generic::luma::<$lanes>(edge,beta,tc,lengths); }
        }
        $(#[$attr])*
        unsafe fn chroma_edge(plane: &mut [u8], edge: &super::ChromaEdge) {
            // SAFETY: target feature enabled; the API validated all samples.
            unsafe { super::chroma_edge::simd::<$edge_lanes>(plane, edge); }
        }
        $(#[$attr])*
        unsafe fn chroma(edge: &mut [[u8;8];2], beta: i32, tc: i32, large: bool, boundary: bool) {
            // SAFETY: target feature enabled and the chosen lane type has four lanes.
            unsafe { super::generic::chroma::<$lanes>(edge,beta,tc,large,boundary); }
        }
    };
}
pub(crate) use kernels;
