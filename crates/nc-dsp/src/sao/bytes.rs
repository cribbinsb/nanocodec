//! Byte-wide SAO apply schedule shared by all SIMD instruction sets.
//! Classification selects one signed offset; separate unsigned saturating
//! add/subtract applies it without widening samples to 16-bit lanes.
#![allow(clippy::inline_always)]
use super::Rect;

pub(super) trait Bytes: Copy {
    const LANES: usize;
    unsafe fn load(p: *const u8) -> Self;
    unsafe fn store(self, p: *mut u8);
    fn splat(v: u8) -> Self;
    fn eq(self, b: Self) -> Self;
    fn gt(self, b: Self) -> Self;
    fn and(self, b: Self) -> Self;
    fn add(self, b: Self) -> Self;
    fn sub(self, b: Self) -> Self;
    fn add_sat(self, b: Self) -> Self;
    fn sub_sat(self, b: Self) -> Self;
    fn band(self) -> Self;
}

#[inline(always)]
fn offsets<V: Bytes>(offsets: [i8; 4]) -> ([V; 4], [V; 4]) {
    // Only scalar extraction uses closures; vector splats stay in this body
    // because outlined closures do not inherit target features.
    let pos = |i: usize| offsets[i].max(0) as u8;
    let neg = |i: usize| offsets[i].min(0).unsigned_abs();
    (
        [
            V::splat(pos(0)),
            V::splat(pos(1)),
            V::splat(pos(2)),
            V::splat(pos(3)),
        ],
        [
            V::splat(neg(0)),
            V::splat(neg(1)),
            V::splat(neg(2)),
            V::splat(neg(3)),
        ],
    )
}

#[inline(always)]
fn adjust<V: Bytes>(c: V, index: V, cats: [V; 4], pos: [V; 4], neg: [V; 4], enabled: V) -> V {
    let mut plus = V::splat(0);
    let mut minus = V::splat(0);
    for k in 0..4 {
        let mask = index.eq(cats[k]).and(enabled);
        plus = plus.add(pos[k].and(mask));
        minus = minus.add(neg[k].and(mask));
    }
    c.add_sat(plus).sub_sat(minus)
}

#[inline(always)]
pub(super) unsafe fn band<V: Bytes>(
    plane: &mut [u8],
    stride: usize,
    rect: Rect,
    position: u8,
    off: [i8; 4],
    exempt: &[bool],
) {
    let (pos, neg) = offsets::<V>(off);
    let cats = [
        V::splat(position),
        V::splat((position + 1) & 31),
        V::splat((position + 2) & 31),
        V::splat((position + 3) & 31),
    ];
    let width = rect.x1 - rect.x0;
    // SAFETY: dispatch checked the rectangle and packed mask footprint. Each
    // vector stays inside a row; bool has a one-byte 0/1 representation.
    unsafe {
        for y in rect.y0..rect.y1 {
            let row = plane.as_mut_ptr().add(y * stride + rect.x0);
            let mask_start = (y - rect.y0) * width;
            let mut x = 0;
            while x + V::LANES <= width {
                let c = V::load(row.add(x));
                let enabled = if exempt.is_empty() {
                    V::splat(255)
                } else {
                    V::load(exempt.as_ptr().cast::<u8>().add(mask_start + x)).eq(V::splat(0))
                };
                adjust(c, c.band(), cats, pos, neg, enabled).store(row.add(x));
                x += V::LANES;
            }
            for x in x..width {
                if exempt.is_empty() || !exempt[mask_start + x] {
                    let p = row.add(x);
                    *p = super::scalar::band_sample(*p, position, off);
                }
            }
        }
    }
}

#[inline(always)]
pub(super) unsafe fn edge<V: Bytes, Half: crate::simd::I16FilterLanes>(
    input: &[u8],
    in_stride: usize,
    out: &mut [u8],
    out_stride: usize,
    (a, b): (isize, isize),
    off: [i8; 4],
    rect: Rect,
) {
    let width = rect.x1 - rect.x0;
    if width < V::LANES {
        // SAFETY: identical checked footprints and ISA as the byte path.
        unsafe {
            super::generic::apply::<Half>(input, in_stride, out, out_stride, (a, b), off, rect);
        }
        return;
    }
    let (pos, neg) = offsets::<V>(off);
    let cats = [V::splat(254), V::splat(255), V::splat(1), V::splat(2)];
    // SAFETY: dispatch validates both neighbouring samples of every point in
    // the rectangle and the destination. Vector runs never cross its rows.
    unsafe {
        for y in rect.y0..rect.y1 {
            let src = input.as_ptr().add(y * in_stride + rect.x0);
            let dst = out.as_mut_ptr().add(y * out_stride + rect.x0);
            let mut x = 0;
            while x + V::LANES <= width {
                let c = V::load(src.add(x));
                let n0 = V::load(src.add(x).offset(a));
                let n1 = V::load(src.add(x).offset(b));
                // Comparisons are 0xff, so reversed subtraction gives sign(c-n).
                let index = n0.gt(c).sub(c.gt(n0)).add(n1.gt(c).sub(c.gt(n1)));
                adjust(c, index, cats, pos, neg, V::splat(255)).store(dst.add(x));
                x += V::LANES;
            }
        }
        // A half-width vector before the scalar tail prevents cropped CTBs
        // (e.g. widths31/63) from losing their existing SIMD coverage. This
        // second strip is independent because classification reads input.
        let tail = width / V::LANES * V::LANES;
        if tail != width {
            super::generic::apply::<Half>(
                input,
                in_stride,
                out,
                out_stride,
                (a, b),
                off,
                Rect {
                    x0: rect.x0 + tail,
                    ..rect
                },
            );
        }
    }
}
