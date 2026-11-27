//! Ordered H.264 luma batching on full byte lanes. The same scalar equations
//! filter two AVX2 or four AVX-512 sixteen-line edges, with the established
//! per-edge transposes. Adjacent dependent edges remain sequential (§8.7).
#![allow(clippy::inline_always)]
use super::wide::BatchBytes;
use super::x86::B as Narrow;
use super::{Edge, H264Bytes, H264Math, LumaParams, luma_filter};
use crate::deblock::H264EdgeParams;
use crate::deblock::runs::{EdgeRun, Plane, ValidatedRuns};

struct Params([Edge; 4]);
impl<B: BatchBytes> LumaParams<B> for Params {
    #[inline(always)]
    fn bs(&self) -> B {
        B::assemble([
            super::luma_lanes(self.0[0].bs.map(i16::from)),
            super::luma_lanes(self.0[1].bs.map(i16::from)),
            super::luma_lanes(self.0[2].bs.map(i16::from)),
            super::luma_lanes(self.0[3].bs.map(i16::from)),
        ])
    }
    #[inline(always)]
    fn tc0(&self) -> B {
        B::assemble([
            super::luma_lanes(self.0[0].tc0),
            super::luma_lanes(self.0[1].tc0),
            super::luma_lanes(self.0[2].tc0),
            super::luma_lanes(self.0[3].tc0),
        ])
    }
    #[inline(always)]
    fn alpha(&self) -> B {
        B::assemble([
            Narrow::splat(self.0[0].alpha as u8),
            Narrow::splat(self.0[1].alpha as u8),
            Narrow::splat(self.0[2].alpha as u8),
            Narrow::splat(self.0[3].alpha as u8),
        ])
    }
    #[inline(always)]
    fn beta(&self) -> B {
        B::assemble([
            Narrow::splat(self.0[0].beta as u8),
            Narrow::splat(self.0[1].beta as u8),
            Narrow::splat(self.0[2].beta as u8),
            Narrow::splat(self.0[3].beta as u8),
        ])
    }
    #[inline(always)]
    fn alpha_quarter(&self) -> B {
        B::assemble([
            Narrow::splat(((self.0[0].alpha >> 2) + 2) as u8),
            Narrow::splat(((self.0[1].alpha >> 2) + 2) as u8),
            Narrow::splat(((self.0[2].alpha >> 2) + 2) as u8),
            Narrow::splat(((self.0[3].alpha >> 2) + 2) as u8),
        ])
    }
}
#[inline(always)]
fn edge(plane: Plane, run: &EdgeRun<H264EdgeParams>) -> Edge {
    assert_eq!(run.lines, 16, "H.264 luma edge has sixteen lines");
    let (base, across, along) = plane.address(run);
    Edge {
        base,
        across,
        along,
        bs: run.filter.bs,
        tc0: run.filter.tc0.map(i16::from),
        alpha: i16::from(run.filter.alpha),
        beta: i16::from(run.filter.beta),
    }
}

/// # Safety
/// Enabled ISA, `B::EDGES` validated, pairwise disjoint sixteen-line footprints.
#[inline(always)]
unsafe fn filter<B: BatchBytes, const STRONG: bool>(data: &mut [u8], p: &Params) {
    let vertical = p.0[0].across == 1;
    let ptr = data.as_mut_ptr();
    let mut narrow = [[Narrow::splat(0); 8]; 4];
    for (i, positions) in narrow.iter_mut().enumerate().take(B::EDGES) {
        let e = &p.0[i];
        if vertical {
            let rows = core::array::from_fn(|line| {
                ptr.wrapping_add(super::index(e, line, -4)).cast_const()
            });
            // SAFETY: each edge's sixteen p3..q3 spans were validated.
            *positions = unsafe { Narrow::load_rows_8(rows) };
        } else {
            let used = if STRONG { 0..8 } else { 1..7 };
            for k in used {
                // SAFETY: complete sixteen-byte position row is validated.
                positions[k] = unsafe { Narrow::load(ptr.add(super::index(e, 0, k as isize - 4))) };
            }
        }
    }
    let mut values = [B::splat(0); 8];
    for (k, value) in values.iter_mut().enumerate() {
        *value = B::assemble([narrow[0][k], narrow[1][k], narrow[2][k], narrow[3][k]]);
    }
    let Some([p2, p1, p0, q0, q1, q2]) = luma_filter::<B, STRONG, true>(p, values) else {
        return;
    };
    let outputs = [
        p2.split(),
        p1.split(),
        p0.split(),
        q0.split(),
        q1.split(),
        q2.split(),
    ];
    for (i, positions) in narrow.iter().enumerate().take(B::EDGES) {
        let e = &p.0[i];
        if vertical {
            let rows = core::array::from_fn(|line| ptr.wrapping_add(super::index(e, line, -4)));
            let out = [
                positions[0],
                outputs[0][i],
                outputs[1][i],
                outputs[2][i],
                outputs[3][i],
                outputs[4][i],
                outputs[5][i],
                positions[7],
            ];
            // SAFETY: validated independent footprint; p3/q3 stay unchanged.
            unsafe {
                Narrow::store_rows_8(out, rows);
            }
        } else {
            let changed = if STRONG { 0..6 } else { 1..5 };
            for k in changed {
                // SAFETY: complete position rows are within the footprint.
                unsafe {
                    outputs[k][i].store(ptr.add(super::index(e, 0, k as isize - 3)));
                }
            }
        }
    }
}

/// # Safety
/// B's ISA is enabled. Geometry is validated before any pixel access.
#[inline(always)]
pub(crate) unsafe fn runs<B: BatchBytes>(
    data: &mut [u8],
    plane: Plane,
    runs: &[EdgeRun<H264EdgeParams>],
) {
    let mut source = ValidatedRuns::new(plane, runs, false, 16, |p: H264EdgeParams| {
        assert!(p.bs.iter().all(|&v| v <= 4), "bS above four");
        4
    });
    let mut pending = source.next();
    while let Some(first) = pending.take() {
        pending = source.next();
        let initial = edge(plane, &first);
        let mut p = Params([initial; 4]);
        let mut positions = [(first.x, first.y); 4];
        let mut count = 1;
        while count < B::EDGES {
            let Some(next) = pending.as_ref() else {
                break;
            };
            let dependent = positions[..count].iter().any(|&(x, y)| {
                if plane.vertical {
                    x.abs_diff(next.x) < 8 && y.abs_diff(next.y) < 16
                } else {
                    y.abs_diff(next.y) < 8 && x.abs_diff(next.x) < 16
                }
            });
            if dependent {
                break;
            }
            let next = pending.take().expect("independent edge");
            pending = source.next();
            p.0[count] = edge(plane, &next);
            positions[count] = (next.x, next.y);
            count += 1;
        }
        // SAFETY: every admitted footprint is validated and pairwise disjoint.
        // Partial groups use the matching complete adapter, retaining edge order.
        unsafe {
            if count == B::EDGES {
                apply::<B>(data, &p);
            } else {
                if count >= 2 {
                    apply::<B::Pair>(data, &p);
                }
                for e in &p.0[if count >= 2 { 2 } else { 0 }..count] {
                    super::luma::<Narrow>(data, e);
                }
            }
        }
    }
}
#[inline(always)]
unsafe fn apply<B: BatchBytes>(data: &mut [u8], p: &Params) {
    let strong = p.0[..B::EDGES].iter().any(|e| e.bs.contains(&4));
    // SAFETY: caller validates independent complete footprints and enabled ISA.
    unsafe {
        if strong {
            filter::<B, true>(data, p);
        } else {
            filter::<B, false>(data, p);
        }
    }
}

/// # Safety
/// B has two edges, its ISA is enabled, and both footprints are validated
/// and disjoint. This entry avoids revalidating/repacking run descriptors.
#[inline(always)]
pub(crate) unsafe fn pair<B: BatchBytes>(data: &mut [u8], edges: &[Edge; 2]) {
    let p = Params([edges[0], edges[1], edges[0], edges[1]]);
    // SAFETY: forwarded from the validated macroblock pair contract.
    unsafe {
        apply::<B>(data, &p);
    }
}
