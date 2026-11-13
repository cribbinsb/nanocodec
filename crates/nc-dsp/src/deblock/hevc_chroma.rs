//! HEVC chroma (§8.7.2.5.5) on the shared packed loop-filter lanes. Only
//! p1/p0/q0/q1 are read and p0/q0 written: vertical edges transpose bytes
//! before widening, avoiding the luma kernel's full p3..q3 word tiles.

use super::HevcEdge;
use crate::lpf_bytes::{Bytes, LANES, MaskWords, Words};

/// Four or eight lines, with tC shared by each pair of chroma lines.
///
/// # Safety
/// The caller enables B's ISA and validates p1..q1 on every declared line.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) unsafe fn filter<B: Bytes<W: MaskWords>>(plane: &mut [u8], edge: &HevcEdge) {
    // SAFETY: forwarded validated footprint and ISA contract.
    unsafe {
        match edge.lines {
            4 => filter_lines::<B, 4>(plane, edge),
            8 => filter_lines::<B, 8>(plane, edge),
            _ => unreachable!("HEVC chroma has four or eight lines"),
        }
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn filter_lines<B: Bytes<W: MaskWords>, const LINES: usize>(
    plane: &mut [u8],
    edge: &HevcEdge,
) {
    let base = edge.base - 2 * edge.across;
    let mut samples = [B::splat(0); LANES];
    // SAFETY: the caller checked every p1..q1 row/column. Exact-width byte
    // loads read only the four taps, and only LINES rows on vertical edges.
    unsafe {
        if edge.along == 1 {
            for (i, sample) in samples[..4].iter_mut().enumerate() {
                let p = plane.as_ptr().add(base + i * edge.across);
                *sample = if LINES == 4 { B::load4(p) } else { B::load(p) };
            }
        } else {
            for (i, sample) in samples[..LINES].iter_mut().enumerate() {
                *sample = B::load4(plane.as_ptr().add(base + i * edge.along));
            }
            samples = B::transpose(samples);
        }
    }
    let [p1, p0, q0, q1] = [
        samples[0].widen(),
        samples[1].widen(),
        samples[2].widen(),
        samples[3].widen(),
    ];
    let tc = B::W::load(&core::array::from_fn(|i| edge.tc[i / 2]));
    let delta = q0
        .sub(p0)
        .mul(B::W::splat(4))
        .add(p1)
        .sub(q1)
        .add(B::W::splat(4))
        .shr::<3>();
    let delta = delta.max(B::W::splat(0).sub(tc)).min(tc);
    let (p0, q0) = (B::narrow(p0.add(delta)), B::narrow(q0.sub(delta)));
    // SAFETY: p0/q0 are inside the footprint. Stores touch neither outer
    // taps nor row padding, and never write inactive lines of a short call.
    unsafe {
        if edge.along == 1 {
            let p = plane.as_mut_ptr().add(edge.base - edge.across);
            let q = plane.as_mut_ptr().add(edge.base);
            if LINES == 4 {
                p0.store4(p);
                q0.store4(q);
            } else {
                p0.store(p);
                q0.store(q);
            }
        } else {
            let mut columns = [B::splat(0); LANES];
            columns[0] = p0;
            columns[1] = q0;
            let rows = B::transpose(columns);
            for (i, row) in rows.into_iter().take(LINES).enumerate() {
                row.store2(plane.as_mut_ptr().add(edge.base - 1 + i * edge.along));
            }
        }
    }
}
