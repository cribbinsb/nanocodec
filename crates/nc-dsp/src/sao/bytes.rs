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
    #[cfg(not(target_arch = "wasm32"))]
    fn eq(self, b: Self) -> Self;
    fn gt(self, b: Self) -> Self;
    fn table(values: [u8; 16]) -> Self;
    fn lookup(self, table: Self) -> Self;
    #[cfg(not(target_arch = "wasm32"))]
    fn and(self, b: Self) -> Self;
    fn add(self, b: Self) -> Self;
    fn sub(self, b: Self) -> Self;
    fn add_sat(self, b: Self) -> Self;
    fn sub_sat(self, b: Self) -> Self;
    #[cfg(not(target_arch = "wasm32"))]
    fn band(self) -> Self;
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
pub(super) unsafe fn band<V: Bytes>(
    plane: &mut [u8],
    stride: usize,
    rect: Rect,
    position: u8,
    off: [i8; 4],
    exempt: &[bool],
) {
    let table = |values: [u8; 4]| {
        let [a, b, c, d] = values;
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, d, c, b, a]
    };
    let plus = V::table(table(off.map(|v| v.max(0) as u8)));
    let minus = V::table(table(off.map(|v| v.min(0).unsigned_abs())));
    let position_vec = V::splat(position);
    let mask = V::splat(31);
    let end = V::splat(15);
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
                // Only relative bands 0..3 reach the four populated entries.
                let index = end.sub_sat(c.band().sub(position_vec).and(mask));
                c.add_sat(index.lookup(plus).and(enabled))
                    .sub_sat(index.lookup(minus).and(enabled))
                    .store(row.add(x));
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
    let table = |values: [u8; 4]| {
        let [a, b, c, d] = values;
        [a, b, 0, c, d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    };
    let plus = V::table(table(off.map(|v| v.max(0) as u8)));
    let minus = V::table(table(off.map(|v| v.min(0).unsigned_abs())));
    let bias = V::splat(2);
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
                let index = n0.gt(c).sub(c.gt(n0)).add(n1.gt(c).sub(c.gt(n1))).add(bias);
                c.add_sat(index.lookup(plus))
                    .sub_sat(index.lookup(minus))
                    .store(dst.add(x));
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
