//! Lane arithmetic for bilinear DMVR, search SAD and affine PROF.
#![allow(unsafe_code, clippy::inline_always)]
use core::mem::MaybeUninit;

use nc_common::{BitDepth, PlaneView};

use crate::simd::{I16Lanes, I32Lanes};
use crate::vvc_dmvr::{SEARCH_RADIUS, SEARCH_WIDTH, fractional};

#[inline(always)]
fn bilinear_h_u8<V: I16Lanes>(row: &[u8], x: usize, fx: u8) -> V {
    if fx == 0 {
        V::load_u8(&row[x..]).mul(V::splat(4))
    } else {
        V::mul_add_u8_pair(&row[x..], &row[x + 1..], 16 - fx, fx)
            .add(V::splat(2))
            .sra(2)
    }
}

#[inline(always)]
fn bilinear_h_row_u8<W: I16Lanes, H: I16Lanes>(row: &[u8], width: usize, fx: u8, dst: &mut [i16]) {
    let mut x = 0;
    while x + W::LANES <= width {
        bilinear_h_u8::<W>(row, x, fx).store(&mut dst[x..]);
        x += W::LANES;
    }
    while x + H::LANES <= width {
        bilinear_h_u8::<H>(row, x, fx).store(&mut dst[x..]);
        x += H::LANES;
    }
    if x < width && width >= H::LANES {
        let tx = width - H::LANES;
        bilinear_h_u8::<H>(row, tx, fx).store(&mut dst[tx..]);
        return;
    }
    while x < width {
        dst[x] =
            ((i32::from(row[x]) * i32::from(16 - fx) + i32::from(row[x + 1]) * i32::from(fx) + 2)
                >> 2) as i16;
        x += 1;
    }
}

#[inline(always)]
fn bilinear_v_i16<L: I16Lanes>(top: L, bottom: L, fy: i16) -> L {
    top.mul(L::splat(16 - fy))
        .add(bottom.mul(L::splat(fy)))
        .add(L::splat(8))
        .sra(4)
}

#[inline(always)]
fn bilinear_v_row<W: I16Lanes, H: I16Lanes>(
    top: &[i16],
    bottom: &[i16],
    width: usize,
    fy: i16,
    dst: &mut [i16],
) {
    let mut x = 0;
    while x + W::LANES <= width {
        bilinear_v_i16(W::load(&top[x..]), W::load(&bottom[x..]), fy).store(&mut dst[x..]);
        x += W::LANES;
    }
    while x + H::LANES <= width {
        bilinear_v_i16(H::load(&top[x..]), H::load(&bottom[x..]), fy).store(&mut dst[x..]);
        x += H::LANES;
    }
    if x < width && width >= H::LANES {
        let tx = width - H::LANES;
        bilinear_v_i16(H::load(&top[tx..]), H::load(&bottom[tx..]), fy).store(&mut dst[tx..]);
        return;
    }
    while x < width {
        dst[x] =
            ((i32::from(top[x]) * i32::from(16 - fy) + i32::from(bottom[x]) * i32::from(fy) + 8)
                >> 4) as i16;
        x += 1;
    }
}

#[inline(always)]
fn bilinear_u8_2vec<W: I16Lanes, H: I16Lanes>(
    (width, height): (usize, usize),
    input: &PlaneView<'_>,
    (fx, fy): (u8, i16),
    out: &mut [i16],
) {
    let tx = width - H::LANES;
    let r0 = input.row(0);
    let (mut top0, mut top1) = (
        bilinear_h_u8::<W>(r0, 0, fx),
        bilinear_h_u8::<H>(r0, tx, fx),
    );
    for (y, dst) in out.chunks_exact_mut(width).take(height).enumerate() {
        let rn = input.row(y + 1);
        let (bot0, bot1) = (
            bilinear_h_u8::<W>(rn, 0, fx),
            bilinear_h_u8::<H>(rn, tx, fx),
        );
        bilinear_v_i16(top0, bot0, fy).store(dst);
        bilinear_v_i16(top1, bot1, fy).store(&mut dst[tx..]);
        (top0, top1) = (bot0, bot1);
    }
}

#[inline(always)]
pub(super) unsafe fn bilinear<W: I16Lanes, H: I16Lanes>(
    size: (usize, usize),
    input: &PlaneView<'_>,
    phases: (usize, usize),
    out: &mut [i16],
) {
    let (width, height) = size;
    if width == 0 || height == 0 {
        return;
    }
    let (fx, fy) = (phases.0 as u8, phases.1 as i16);
    if fy == 0 {
        for (y, dst) in out.chunks_exact_mut(width).take(height).enumerate() {
            bilinear_h_row_u8::<W, H>(input.row(y), width, fx, dst);
        }
        return;
    }
    if W::LANES < width && width <= W::LANES + H::LANES {
        bilinear_u8_2vec::<W, H>(size, input, (fx, fy), out);
        return;
    }
    if width <= 64 {
        let mut hbuf = [[MaybeUninit::<i16>::uninit(); 64]; 2];
        // SAFETY: `bilinear_h_row_u8` writes all `0..width` elements before any read.
        let (b0, b1) = unsafe {
            (
                core::slice::from_raw_parts_mut(hbuf[0].as_mut_ptr().cast::<i16>(), width),
                core::slice::from_raw_parts_mut(hbuf[1].as_mut_ptr().cast::<i16>(), width),
            )
        };
        bilinear_h_row_u8::<W, H>(input.row(0), width, fx, b0);
        for (y, dst) in out.chunks_exact_mut(width).take(height).enumerate() {
            if y & 1 == 0 {
                bilinear_h_row_u8::<W, H>(input.row(y + 1), width, fx, b1);
                bilinear_v_row::<W, H>(b0, b1, width, fy, dst);
            } else {
                bilinear_h_row_u8::<W, H>(input.row(y + 1), width, fx, b0);
                bilinear_v_row::<W, H>(b1, b0, width, fy, dst);
            }
        }
        return;
    }
    for (y, row) in out.chunks_exact_mut(width).take(height).enumerate() {
        let (top, bottom) = (input.row(y), input.row(y + 1));
        let mut x = 0;
        while x + W::LANES <= width {
            bilinear_v_i16(
                bilinear_h_u8::<W>(top, x, fx),
                bilinear_h_u8::<W>(bottom, x, fx),
                fy,
            )
            .store(&mut row[x..]);
            x += W::LANES;
        }
        while x < width {
            let ht = (i32::from(top[x]) * i32::from(16 - fx)
                + i32::from(top[x + 1]) * i32::from(fx)
                + 2)
                >> 2;
            let hb = (i32::from(bottom[x]) * i32::from(16 - fx)
                + i32::from(bottom[x + 1]) * i32::from(fx)
                + 2)
                >> 2;
            row[x] = ((ht * i32::from(16 - fy) + hb * i32::from(fy) + 8) >> 4) as i16;
            x += 1;
        }
    }
}

#[inline(always)]
fn bilinear_h_u16<L: I16Lanes>(row: &[u16], x: usize, fx: i16, offset1: i32, shift1: u32) -> L {
    if fx == 0 && shift1 == 4 {
        return L::load_u16(&row[x..]);
    }
    if fx == 0 && shift1 < 4 {
        return L::load_u16(&row[x..]).mul(L::splat(1i16 << (4 - shift1)));
    }
    let p = row[x..=x + L::LANES].as_ptr();
    // SAFETY: `row[x..x + L::LANES + 1]` was bounds-checked once above.
    let acc = unsafe {
        L::acc_mul_add2(
            L::acc_zero(),
            L::load_u16_at(p),
            16 - fx,
            L::load_u16_at(p.add(1)),
            fx,
        )
    };
    L::acc_round_shift_narrow(acc, offset1, shift1)
}

#[inline(always)]
fn bilinear_h_row_u16<W: I16Lanes, H: I16Lanes>(
    row: &[u16],
    width: usize,
    fx: i16,
    offset1: i32,
    shift1: u32,
    dst: &mut [i16],
) {
    let mut x = 0;
    while x + W::LANES <= width {
        bilinear_h_u16::<W>(row, x, fx, offset1, shift1).store(&mut dst[x..]);
        x += W::LANES;
    }
    while x + H::LANES <= width {
        bilinear_h_u16::<H>(row, x, fx, offset1, shift1).store(&mut dst[x..]);
        x += H::LANES;
    }
    if x < width && width >= H::LANES {
        let tx = width - H::LANES;
        bilinear_h_u16::<H>(row, tx, fx, offset1, shift1).store(&mut dst[tx..]);
        return;
    }
    while x < width {
        dst[x] = ((i32::from(row[x]) * i32::from(16 - fx)
            + i32::from(row[x + 1]) * i32::from(fx)
            + offset1)
            >> shift1) as i16;
        x += 1;
    }
}

#[inline(always)]
fn bilinear_u16_2vec<W: I16Lanes, H: I16Lanes>(
    (width, height): (usize, usize),
    input: &PlaneView<'_, u16>,
    (fx, fy): (i16, i16),
    (offset1, shift1): (i32, u32),
    out: &mut [i16],
) {
    let tx = width - H::LANES;
    let r0 = input.row(0);
    let mut top0 = bilinear_h_u16::<W>(r0, 0, fx, offset1, shift1);
    let mut top1 = bilinear_h_u16::<H>(r0, tx, fx, offset1, shift1);
    for (y, dst) in out.chunks_exact_mut(width).take(height).enumerate() {
        let rn = input.row(y + 1);
        let bot0 = bilinear_h_u16::<W>(rn, 0, fx, offset1, shift1);
        let bot1 = bilinear_h_u16::<H>(rn, tx, fx, offset1, shift1);
        bilinear_v_i16(top0, bot0, fy).store(dst);
        bilinear_v_i16(top1, bot1, fy).store(&mut dst[tx..]);
        (top0, top1) = (bot0, bot1);
    }
}

#[inline(always)]
fn bilinear_u16_3vec<W: I16Lanes, H: I16Lanes>(
    (width, height): (usize, usize),
    input: &PlaneView<'_, u16>,
    (fx, fy): (i16, i16),
    (offset1, shift1): (i32, u32),
    out: &mut [i16],
) {
    let tx = width - H::LANES;
    let r0 = input.row(0);
    let mut top0 = bilinear_h_u16::<W>(r0, 0, fx, offset1, shift1);
    let mut top1 = bilinear_h_u16::<W>(r0, W::LANES, fx, offset1, shift1);
    let mut top2 = bilinear_h_u16::<H>(r0, tx, fx, offset1, shift1);
    for (y, dst) in out.chunks_exact_mut(width).take(height).enumerate() {
        let rn = input.row(y + 1);
        let bot0 = bilinear_h_u16::<W>(rn, 0, fx, offset1, shift1);
        let bot1 = bilinear_h_u16::<W>(rn, W::LANES, fx, offset1, shift1);
        let bot2 = bilinear_h_u16::<H>(rn, tx, fx, offset1, shift1);
        bilinear_v_i16(top0, bot0, fy).store(dst);
        bilinear_v_i16(top1, bot1, fy).store(&mut dst[W::LANES..]);
        bilinear_v_i16(top2, bot2, fy).store(&mut dst[tx..]);
        (top0, top1, top2) = (bot0, bot1, bot2);
    }
}

#[inline(always)]
pub(super) unsafe fn bilinear_u16<W: I16Lanes, H: I16Lanes>(
    size: (usize, usize),
    input: &PlaneView<'_, u16>,
    phases: (usize, usize),
    out: &mut [i16],
    depth: BitDepth,
) {
    let (width, height) = size;
    if width == 0 || height == 0 {
        return;
    }
    let (fx, fy) = (phases.0 as i16, phases.1 as i16);
    let shift1 = depth.bits() - 6;
    let offset1 = 1i32 << (shift1 - 1);
    if fy == 0 {
        for (y, dst) in out.chunks_exact_mut(width).take(height).enumerate() {
            bilinear_h_row_u16::<W, H>(input.row(y), width, fx, offset1, shift1, dst);
        }
        return;
    }
    if W::LANES < width && width <= W::LANES + H::LANES {
        bilinear_u16_2vec::<W, H>(size, input, (fx, fy), (offset1, shift1), out);
        return;
    }
    if 2 * W::LANES < width && width <= 2 * W::LANES + H::LANES {
        bilinear_u16_3vec::<W, H>(size, input, (fx, fy), (offset1, shift1), out);
        return;
    }
    if width <= 64 {
        let mut hbuf = [[MaybeUninit::<i16>::uninit(); 64]; 2];
        // SAFETY: `bilinear_h_row_u16` writes all `0..width` elements before any read.
        let (b0, b1) = unsafe {
            (
                core::slice::from_raw_parts_mut(hbuf[0].as_mut_ptr().cast::<i16>(), width),
                core::slice::from_raw_parts_mut(hbuf[1].as_mut_ptr().cast::<i16>(), width),
            )
        };
        bilinear_h_row_u16::<W, H>(input.row(0), width, fx, offset1, shift1, b0);
        for (y, dst) in out.chunks_exact_mut(width).take(height).enumerate() {
            if y & 1 == 0 {
                bilinear_h_row_u16::<W, H>(input.row(y + 1), width, fx, offset1, shift1, b1);
                bilinear_v_row::<W, H>(b0, b1, width, fy, dst);
            } else {
                bilinear_h_row_u16::<W, H>(input.row(y + 1), width, fx, offset1, shift1, b0);
                bilinear_v_row::<W, H>(b1, b0, width, fy, dst);
            }
        }
        return;
    }
    for (y, row) in out.chunks_exact_mut(width).take(height).enumerate() {
        let (top, bottom) = (input.row(y), input.row(y + 1));
        let mut x = 0;
        while x + W::LANES <= width {
            let ht = bilinear_h_u16::<W>(top, x, fx, offset1, shift1);
            let hb = bilinear_h_u16::<W>(bottom, x, fx, offset1, shift1);
            bilinear_v_i16(ht, hb, fy).store(&mut row[x..]);
            x += W::LANES;
        }
        while x < width {
            let ht = (i32::from(top[x]) * i32::from(16 - fx)
                + i32::from(top[x + 1]) * i32::from(fx)
                + offset1)
                >> shift1;
            let hb = (i32::from(bottom[x]) * i32::from(16 - fx)
                + i32::from(bottom[x + 1]) * i32::from(fx)
                + offset1)
                >> shift1;
            row[x] = ((ht * i32::from(16 - fy) + hb * i32::from(fy) + 8) >> 4) as i16;
            x += 1;
        }
    }
}

#[inline(always)]
unsafe fn dmvr_sad<V: I16Lanes>(
    (width, height): (usize, usize),
    p0: *const u16,
    p1: *const u16,
    dx: isize,
    dy: isize,
) -> i32 {
    debug_assert!(width.is_multiple_of(V::LANES));
    let stride = width + 2 * SEARCH_RADIUS;
    let offset = dy * stride as isize + dx;
    let bias = V::splat(i16::MIN);
    let mut acc = V::acc_zero();
    for y in (0..height).step_by(2) {
        let row_base = ((y + SEARCH_RADIUS) * stride + SEARCH_RADIUS) as isize;
        let mut x = 0;
        while x < width {
            let at = row_base + x as isize;
            // SAFETY: caller validated `p0`/`p1` cover `stride * (height + 4)`
            // and `dx, dy` lie within `[-2, 2]`.
            let (a, b) = unsafe {
                (
                    V::load_u16_at(p0.offset(at + offset)),
                    V::load_u16_at(p1.offset(at - offset)),
                )
            };
            acc = V::acc_add(acc, a.max(b).sub(a.min(b)).sub(bias));
            x += V::LANES;
        }
    }
    let samples = (width * (height / 2)) as i64;
    (V::acc_sum(acc) + 32_768 * samples) as i32
}

#[inline(always)]
pub(super) unsafe fn dmvr<V: I16Lanes>(
    size: (usize, usize),
    inputs: [&[i16]; 2],
) -> ([i32; 2], bool) {
    let (width, height) = size;
    let stride = width + 2 * SEARCH_RADIUS;
    let needed = stride * (height + 2 * SEARCH_RADIUS);
    let p0 = inputs[0][..needed].as_ptr().cast::<u16>();
    let p1 = inputs[1][..needed].as_ptr().cast::<u16>();
    // SAFETY: caller enables `V`'s ISA and verified padded search windows.
    let center = unsafe { dmvr_sad::<V>(size, p0, p1, 0, 0) };
    let biased_center = center - (center >> 2);
    let mut best = biased_center;
    let area = (width * height) as i32;
    if best < area {
        return ([0; 2], false);
    }
    let mut costs = [0i32; SEARCH_WIDTH * SEARCH_WIDTH];
    let mut displacement = [0i32; 2];
    for y in -2..=2_i32 {
        for x in -2..=2_i32 {
            let cost = if x == 0 && y == 0 {
                biased_center
            } else {
                // SAFETY: `x, y` lie within `[-2, 2]`, inside the 2-sample border.
                unsafe { dmvr_sad::<V>(size, p0, p1, x as isize, y as isize) }
            };
            costs[((y + 2) * SEARCH_WIDTH as i32 + x + 2) as usize] = cost;
            if cost < best {
                best = cost;
                displacement = [x, y];
            }
        }
    }
    let [x, y] = displacement;
    let mut vector = displacement.map(|v| v * 16);
    if x.abs() < 2 && y.abs() < 2 {
        let at = ((y + 2) * SEARCH_WIDTH as i32 + x + 2) as usize;
        vector[0] += fractional(costs[at], costs[at - 1], costs[at + 1]);
        vector[1] += fractional(
            costs[at],
            costs[at - SEARCH_WIDTH],
            costs[at + SEARCH_WIDTH],
        );
    }
    (vector, best >= 2 * area)
}

/// The gradients of [`crate::vvc_optical_flow::gradient`] for `W::LANES`
/// samples from `at` (`ptr` validated once for `stride * (height + 2)`).
#[inline(always)]
pub(super) unsafe fn gradient16_ptr<W: I16Lanes>(
    ptr: *const u16,
    stride: usize,
    at: usize,
) -> [W; 2] {
    // SAFETY: caller validated `ptr` covers `(height + 2) * stride` samples.
    unsafe {
        [
            W::load_u16_at(ptr.add(at + 1))
                .sra(6)
                .sub(W::load_u16_at(ptr.add(at - 1)).sra(6)),
            W::load_u16_at(ptr.add(at + stride))
                .sra(6)
                .sub(W::load_u16_at(ptr.add(at - stride)).sra(6)),
        ]
    }
}

#[inline(always)]
pub(super) unsafe fn gradient<V: I32Lanes>(samples: &[i16], stride: usize, at: usize) -> [V; 2] {
    [
        V::load_i16(&samples[at + 1..])
            .sra(6)
            .sub(V::load_i16(&samples[at - 1..]).sra(6)),
        V::load_i16(&samples[at + stride..])
            .sra(6)
            .sub(V::load_i16(&samples[at - stride..]).sra(6)),
    ]
}

#[inline(always)]
pub(super) unsafe fn prof<V: I32Lanes>(
    samples: &[i16; 36],
    corrections: &[[i8; 2]; 16],
    out: &mut [i16; 16],
) {
    debug_assert_eq!(V::LANES, 4);
    for y in 0..4 {
        let at = (y + 1) * 6 + 1;
        // SAFETY: caller supplies a supported four-lane ISA and padded 6×6 block.
        let [gx, gy] = unsafe { gradient::<V>(samples, 6, at) };
        let cx = core::array::from_fn::<_, 4, _>(|x| i32::from(corrections[y * 4 + x][0]));
        let cy = core::array::from_fn::<_, 4, _>(|x| i32::from(corrections[y * 4 + x][1]));
        let delta = gx
            .mul(V::load(&cx))
            .add(gy.mul(V::load(&cy)))
            .max(V::splat(-8192))
            .min(V::splat(8191));
        let value = V::load_i16(&samples[at..]).add(delta).clamp16();
        value.store_sat16(&mut out[y * 4..][..4]);
    }
}
