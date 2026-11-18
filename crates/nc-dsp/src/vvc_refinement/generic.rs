//! Lane arithmetic for bilinear DMVR, search SAD and affine PROF.
#![allow(unsafe_code, clippy::inline_always)]
use nc_common::{BitDepth, PlaneView};

use crate::simd::{I16Lanes, I32Lanes};
use crate::vvc_dmvr::{SEARCH_RADIUS, SEARCH_WIDTH, fractional};

#[inline(always)]
unsafe fn bilinear_h_u8_ptr<V: I16Lanes>(p: *const u8, fx: u8, two: V, four: V) -> V {
    // SAFETY: caller validated `p` covers `V::LANES + usize::from(fx != 0)` bytes.
    unsafe {
        if fx == 0 {
            V::load_u8_at(p).mul(four)
        } else {
            V::mul_add_u8_pair_at(p, p.add(1), 16 - fx, fx)
                .add(two)
                .sra(2)
        }
    }
}

#[inline(always)]
fn bilinear_v_splats<L: I16Lanes>(fy: i16) -> (L, L, L) {
    (L::splat(16 - fy), L::splat(fy), L::splat(8))
}

#[inline(always)]
fn bilinear_v_pre<L: I16Lanes>(top: L, bottom: L, (wy0, wy1, off2): (L, L, L)) -> L {
    top.mul(wy0).add(bottom.mul(wy1)).add(off2).sra(4)
}

#[inline(always)]
unsafe fn bilinear_u8_strip<L: I16Lanes>(
    (width, height, x_off): (usize, usize, usize),
    input: &PlaneView<'_>,
    (fx, fy): (u8, i16),
    out: &mut [i16],
) {
    let stride = input.stride();
    let extra = usize::from(fx != 0);
    let req_src = height * stride + x_off + L::LANES + extra;
    let src_data = &input.data()[..req_src];
    let dst_data = &mut out[..width * height];
    let (two, four) = (L::splat(2), L::splat(4));
    let v_sp = bilinear_v_splats::<L>(fy);
    // SAFETY: `src_data` and `dst_data` validated once above.
    unsafe {
        let mut src_p = src_data.as_ptr().add(x_off);
        let mut dst_p = dst_data.as_mut_ptr().add(x_off).cast::<u16>();
        let mut top = bilinear_h_u8_ptr::<L>(src_p, fx, two, four);
        for _ in 0..height {
            src_p = src_p.add(stride);
            let bot = bilinear_h_u8_ptr::<L>(src_p, fx, two, four);
            bilinear_v_pre(top, bot, v_sp).store_u16_at(dst_p);
            dst_p = dst_p.add(width);
            top = bot;
        }
    }
}

#[inline(always)]
fn bilinear_h_only_u8<W: I16Lanes, H: I16Lanes>(
    (width, height): (usize, usize),
    input: &PlaneView<'_>,
    fx: u8,
    out: &mut [i16],
) {
    let stride = input.stride();
    let extra = usize::from(fx != 0);
    let src_data = &input.data()[..(height - 1) * stride + width + extra];
    let (two_w, four_w) = (W::splat(2), W::splat(4));
    let (two_h, four_h) = (H::splat(2), H::splat(4));
    // SAFETY: `src_data` validated across `height` rows of `width + extra` bytes.
    unsafe {
        let mut row_p = src_data.as_ptr();
        for dst in out.chunks_exact_mut(width).take(height) {
            let mut x = 0;
            while x + W::LANES <= width {
                bilinear_h_u8_ptr::<W>(row_p.add(x), fx, two_w, four_w).store(&mut dst[x..]);
                x += W::LANES;
            }
            while x + H::LANES <= width {
                bilinear_h_u8_ptr::<H>(row_p.add(x), fx, two_h, four_h).store(&mut dst[x..]);
                x += H::LANES;
            }
            if x < width && width >= H::LANES {
                let tx = width - H::LANES;
                bilinear_h_u8_ptr::<H>(row_p.add(tx), fx, two_h, four_h).store(&mut dst[tx..]);
            } else {
                while x < width {
                    dst[x] = ((i32::from(*row_p.add(x)) * i32::from(16 - fx)
                        + i32::from(*row_p.add(x + extra)) * i32::from(fx)
                        + 2)
                        >> 2) as i16;
                    x += 1;
                }
            }
            row_p = row_p.add(stride);
        }
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
        bilinear_h_only_u8::<W, H>(size, input, fx, out);
        return;
    }
    if width >= H::LANES {
        let mut sx = 0;
        // SAFETY: `bilinear_u8_strip` validates `input` and `out` bounds.
        unsafe {
            while sx + W::LANES <= width {
                bilinear_u8_strip::<W>((width, height, sx), input, (fx, fy), out);
                sx += W::LANES;
            }
            while sx + H::LANES <= width {
                bilinear_u8_strip::<H>((width, height, sx), input, (fx, fy), out);
                sx += H::LANES;
            }
            if sx < width {
                bilinear_u8_strip::<H>((width, height, width - H::LANES), input, (fx, fy), out);
            }
        }
        return;
    }
    for (y, row) in out.chunks_exact_mut(width).take(height).enumerate() {
        let (top, bottom) = (input.row(y), input.row(y + 1));
        for (x, dst_x) in row.iter_mut().enumerate() {
            let ht = (i32::from(top[x]) * i32::from(16 - fx)
                + i32::from(top[x + 1]) * i32::from(fx)
                + 2)
                >> 2;
            let hb = (i32::from(bottom[x]) * i32::from(16 - fx)
                + i32::from(bottom[x + 1]) * i32::from(fx)
                + 2)
                >> 2;
            *dst_x = ((ht * i32::from(16 - fy) + hb * i32::from(fy) + 8) >> 4) as i16;
        }
    }
}

#[inline(always)]
unsafe fn bilinear_h_u16_ptr<L: I16Lanes>(
    p: *const u16,
    fx: i16,
    (wx0, wx1, off1): (L, L, L),
    (offset1, shift1): (i32, u32),
) -> L {
    // SAFETY: caller validated `p` covers `L::LANES + usize::from(fx != 0)` `u16`s.
    unsafe {
        let s0 = L::load_u16_at(p);
        if fx == 0 {
            if shift1 == 4 {
                return s0;
            }
            if shift1 < 4 {
                return s0.mul(wx0);
            }
            return s0.add(off1).sra(shift1 - 4);
        }
        let s1 = L::load_u16_at(p.add(1));
        if shift1 <= 4 {
            return s0.mul(wx0).add(s1.mul(wx1)).add(off1).sra(shift1);
        }
        let acc = L::acc_mul_add2(L::acc_zero(), s0, 16 - fx, s1, fx);
        L::acc_round_shift_narrow(acc, offset1, shift1)
    }
}

#[inline(always)]
fn bilinear_h_u16_splats<L: I16Lanes>(fx: i16, offset1: i32, shift1: u32) -> (L, L, L) {
    if fx == 0 {
        let mul = if shift1 < 4 { 1i16 << (4 - shift1) } else { 1 };
        let off = if shift1 > 4 { 1i16 << (shift1 - 5) } else { 0 };
        (L::splat(mul), L::splat(0), L::splat(off))
    } else {
        (
            L::splat(16 - fx),
            L::splat(fx),
            L::splat(if shift1 <= 4 { offset1 as i16 } else { 0 }),
        )
    }
}

#[inline(always)]
unsafe fn bilinear_u16_strip<L: I16Lanes>(
    (width, height, x_off): (usize, usize, usize),
    input: &PlaneView<'_, u16>,
    (fx, fy): (i16, i16),
    (offset1, shift1): (i32, u32),
    out: &mut [i16],
) {
    let stride = input.stride();
    let extra = usize::from(fx != 0);
    let req_src = height * stride + x_off + L::LANES + extra;
    let src_data = &input.data()[..req_src];
    let dst_data = &mut out[..width * height];
    let h_sp = bilinear_h_u16_splats::<L>(fx, offset1, shift1);
    let v_sp = bilinear_v_splats::<L>(fy);
    // SAFETY: `src_data` and `dst_data` validated once above.
    unsafe {
        let mut src_p = src_data.as_ptr().add(x_off);
        let mut dst_p = dst_data.as_mut_ptr().add(x_off).cast::<u16>();
        let mut top = bilinear_h_u16_ptr::<L>(src_p, fx, h_sp, (offset1, shift1));
        for _ in 0..height {
            src_p = src_p.add(stride);
            let bot = bilinear_h_u16_ptr::<L>(src_p, fx, h_sp, (offset1, shift1));
            bilinear_v_pre(top, bot, v_sp).store_u16_at(dst_p);
            dst_p = dst_p.add(width);
            top = bot;
        }
    }
}

#[inline(always)]
fn bilinear_h_only_u16<W: I16Lanes, H: I16Lanes>(
    (width, height): (usize, usize),
    input: &PlaneView<'_, u16>,
    fx: i16,
    (offset1, shift1): (i32, u32),
    out: &mut [i16],
) {
    let stride = input.stride();
    let extra = usize::from(fx != 0);
    let src_data = &input.data()[..(height - 1) * stride + width + extra];
    let h_w = bilinear_h_u16_splats::<W>(fx, offset1, shift1);
    let h_h = bilinear_h_u16_splats::<H>(fx, offset1, shift1);
    // SAFETY: `src_data` validated across `height` rows of `width + extra` `u16`s.
    unsafe {
        let mut row_p = src_data.as_ptr();
        for dst in out.chunks_exact_mut(width).take(height) {
            let mut x = 0;
            while x + W::LANES <= width {
                bilinear_h_u16_ptr::<W>(row_p.add(x), fx, h_w, (offset1, shift1))
                    .store(&mut dst[x..]);
                x += W::LANES;
            }
            while x + H::LANES <= width {
                bilinear_h_u16_ptr::<H>(row_p.add(x), fx, h_h, (offset1, shift1))
                    .store(&mut dst[x..]);
                x += H::LANES;
            }
            if x < width && width >= H::LANES {
                let tx = width - H::LANES;
                bilinear_h_u16_ptr::<H>(row_p.add(tx), fx, h_h, (offset1, shift1))
                    .store(&mut dst[tx..]);
            } else {
                while x < width {
                    dst[x] = ((i32::from(*row_p.add(x)) * i32::from(16 - fx)
                        + i32::from(*row_p.add(x + extra)) * i32::from(fx)
                        + offset1)
                        >> shift1) as i16;
                    x += 1;
                }
            }
            row_p = row_p.add(stride);
        }
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
        bilinear_h_only_u16::<W, H>(size, input, fx, (offset1, shift1), out);
        return;
    }
    if width >= H::LANES {
        let mut sx = 0;
        // SAFETY: `bilinear_u16_strip` validates `input` and `out` bounds.
        unsafe {
            while sx + W::LANES <= width {
                bilinear_u16_strip::<W>(
                    (width, height, sx),
                    input,
                    (fx, fy),
                    (offset1, shift1),
                    out,
                );
                sx += W::LANES;
            }
            while sx + H::LANES <= width {
                bilinear_u16_strip::<H>(
                    (width, height, sx),
                    input,
                    (fx, fy),
                    (offset1, shift1),
                    out,
                );
                sx += H::LANES;
            }
            if sx < width {
                bilinear_u16_strip::<H>(
                    (width, height, width - H::LANES),
                    input,
                    (fx, fy),
                    (offset1, shift1),
                    out,
                );
            }
        }
        return;
    }
    for (y, row) in out.chunks_exact_mut(width).take(height).enumerate() {
        let (top, bottom) = (input.row(y), input.row(y + 1));
        for (x, dst_x) in row.iter_mut().enumerate() {
            let ht = (i32::from(top[x]) * i32::from(16 - fx)
                + i32::from(top[x + 1]) * i32::from(fx)
                + offset1)
                >> shift1;
            let hb = (i32::from(bottom[x]) * i32::from(16 - fx)
                + i32::from(bottom[x + 1]) * i32::from(fx)
                + offset1)
                >> shift1;
            *dst_x = ((ht * i32::from(16 - fy) + hb * i32::from(fy) + 8) >> 4) as i16;
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
unsafe fn dmvr_sad_x5<V: I16Lanes>(
    (width, height): (usize, usize),
    p0: *const u16,
    p1: *const u16,
    dy: isize,
    include_center: bool,
) -> [i32; SEARCH_WIDTH] {
    debug_assert!(width.is_multiple_of(V::LANES));
    let stride = width + 2 * SEARCH_RADIUS;
    let dy_off = dy * stride as isize;
    let bias = V::splat(i16::MIN);
    let mut acc = [
        V::acc_zero(),
        V::acc_zero(),
        V::acc_zero(),
        V::acc_zero(),
        V::acc_zero(),
    ];
    for y in (0..height).step_by(2) {
        let row_base = ((y + SEARCH_RADIUS) * stride + SEARCH_RADIUS) as isize;
        let mut x = 0;
        while x < width {
            let at0 = row_base + x as isize + dy_off;
            let at1 = row_base + x as isize - dy_off;
            // SAFETY: caller validated `p0`/`p1` cover `stride * (height + 4)`
            // and `dx, dy` lie within `[-2, 2]`.
            unsafe {
                let a0 = V::load_u16_at(p0.offset(at0 - 2));
                let b0 = V::load_u16_at(p1.offset(at1 + 2));
                acc[0] = V::acc_add(acc[0], a0.max(b0).sub(a0.min(b0)).sub(bias));
                let a1 = V::load_u16_at(p0.offset(at0 - 1));
                let b1 = V::load_u16_at(p1.offset(at1 + 1));
                acc[1] = V::acc_add(acc[1], a1.max(b1).sub(a1.min(b1)).sub(bias));
                if include_center {
                    let a2 = V::load_u16_at(p0.offset(at0));
                    let b2 = V::load_u16_at(p1.offset(at1));
                    acc[2] = V::acc_add(acc[2], a2.max(b2).sub(a2.min(b2)).sub(bias));
                }
                let a3 = V::load_u16_at(p0.offset(at0 + 1));
                let b3 = V::load_u16_at(p1.offset(at1 - 1));
                acc[3] = V::acc_add(acc[3], a3.max(b3).sub(a3.min(b3)).sub(bias));
                let a4 = V::load_u16_at(p0.offset(at0 + 2));
                let b4 = V::load_u16_at(p1.offset(at1 - 2));
                acc[4] = V::acc_add(acc[4], a4.max(b4).sub(a4.min(b4)).sub(bias));
            }
            x += V::LANES;
        }
    }
    let adjust = 32_768 * (width * (height / 2)) as i64;
    [
        (V::acc_sum(acc[0]) + adjust) as i32,
        (V::acc_sum(acc[1]) + adjust) as i32,
        if include_center {
            (V::acc_sum(acc[2]) + adjust) as i32
        } else {
            0
        },
        (V::acc_sum(acc[3]) + adjust) as i32,
        (V::acc_sum(acc[4]) + adjust) as i32,
    ]
}

/// Whether every element in both `needed`-sample windows fits `-2047..=2047`
/// (true for all normative 8..=12-bit bilinear inputs, whose samples lie in
/// `0..=1023`), allowing 8-row SAD accumulation directly in signed 16-bit lanes
/// (`8 * 4094 = 32_752 <= i16::MAX`).
#[inline(always)]
fn inputs_fit_i16_sad<V: I16Lanes>(in0: &[i16], in1: &[i16], needed: usize) -> bool {
    let mut min_v = V::splat(2047);
    let mut max_v = V::splat(-2047);
    let mut i = 0;
    while i + V::LANES <= needed {
        let (a, b) = (V::load(&in0[i..]), V::load(&in1[i..]));
        min_v = min_v.min(a).min(b);
        max_v = max_v.max(a).max(b);
        i += V::LANES;
    }
    if i < needed && needed >= V::LANES {
        let tail = needed - V::LANES;
        let (a, b) = (V::load(&in0[tail..]), V::load(&in1[tail..]));
        min_v = min_v.min(a).min(b);
        max_v = max_v.max(a).max(b);
    } else {
        while i < needed {
            if !(-2047..=2047).contains(&in0[i]) || !(-2047..=2047).contains(&in1[i]) {
                return false;
            }
            i += 1;
        }
    }
    let (mut mins, mut maxs) = ([0i16; 32], [0i16; 32]);
    min_v.store(&mut mins[..V::LANES]);
    max_v.store(&mut maxs[..V::LANES]);
    mins[..V::LANES].iter().all(|&v| v >= -2047) && maxs[..V::LANES].iter().all(|&v| v <= 2047)
}

#[inline(always)]
unsafe fn dmvr_sad_fast<V: I16Lanes>(
    (width, height): (usize, usize),
    p0: *const u16,
    p1: *const u16,
    dx: isize,
    dy: isize,
) -> i32 {
    let stride = width + 2 * SEARCH_RADIUS;
    let offset = dy * stride as isize + dx;
    let mut sum16 = V::splat(0);
    for y in (0..height).step_by(2) {
        let row_base = ((y + SEARCH_RADIUS) * stride + SEARCH_RADIUS) as isize;
        let mut x = 0;
        while x < width {
            let at = row_base + x as isize;
            // SAFETY: caller validated `p0`/`p1` cover `stride * (height + 4)`.
            let (a, b) = unsafe {
                (
                    V::load_u16_at(p0.offset(at + offset)),
                    V::load_u16_at(p1.offset(at - offset)),
                )
            };
            sum16 = sum16.add(a.sub(b).abs());
            x += V::LANES;
        }
    }
    V::acc_sum(V::acc_add(V::acc_zero(), sum16)) as i32
}

#[inline(always)]
unsafe fn dmvr_sad_x5_fast<V: I16Lanes>(
    (width, height): (usize, usize),
    p0: *const u16,
    p1: *const u16,
    dy: isize,
    include_center: bool,
) -> [i32; SEARCH_WIDTH] {
    let stride = width + 2 * SEARCH_RADIUS;
    let dy_off = dy * stride as isize;
    let mut sum16 = [V::splat(0); SEARCH_WIDTH];
    for y in (0..height).step_by(2) {
        let row_base = ((y + SEARCH_RADIUS) * stride + SEARCH_RADIUS) as isize;
        let mut x = 0;
        while x < width {
            let at0 = row_base + x as isize + dy_off;
            let at1 = row_base + x as isize - dy_off;
            // SAFETY: caller validated `p0`/`p1` cover `stride * (height + 4)`.
            unsafe {
                let a0 = V::load_u16_at(p0.offset(at0 - 2));
                let b0 = V::load_u16_at(p1.offset(at1 + 2));
                sum16[0] = sum16[0].add(a0.sub(b0).abs());
                let a1 = V::load_u16_at(p0.offset(at0 - 1));
                let b1 = V::load_u16_at(p1.offset(at1 + 1));
                sum16[1] = sum16[1].add(a1.sub(b1).abs());
                if include_center {
                    let a2 = V::load_u16_at(p0.offset(at0));
                    let b2 = V::load_u16_at(p1.offset(at1));
                    sum16[2] = sum16[2].add(a2.sub(b2).abs());
                }
                let a3 = V::load_u16_at(p0.offset(at0 + 1));
                let b3 = V::load_u16_at(p1.offset(at1 - 1));
                sum16[3] = sum16[3].add(a3.sub(b3).abs());
                let a4 = V::load_u16_at(p0.offset(at0 + 2));
                let b4 = V::load_u16_at(p1.offset(at1 - 2));
                sum16[4] = sum16[4].add(a4.sub(b4).abs());
            }
            x += V::LANES;
        }
    }
    let lane_sum = |v: V| V::acc_sum(V::acc_add(V::acc_zero(), v)) as i32;
    [
        lane_sum(sum16[0]),
        lane_sum(sum16[1]),
        if include_center {
            lane_sum(sum16[2])
        } else {
            0
        },
        lane_sum(sum16[3]),
        lane_sum(sum16[4]),
    ]
}

#[inline(always)]
pub(super) unsafe fn dmvr<V: I16Lanes>(
    size: (usize, usize),
    inputs: [&[i16]; 2],
) -> ([i32; 2], bool) {
    let (width, height) = size;
    let stride = width + 2 * SEARCH_RADIUS;
    let needed = stride * (height + 2 * SEARCH_RADIUS);
    let in0 = &inputs[0][..needed];
    let in1 = &inputs[1][..needed];
    let fast_i16 = height <= 16 && inputs_fit_i16_sad::<V>(in0, in1, needed);
    let p0 = in0.as_ptr().cast::<u16>();
    let p1 = in1.as_ptr().cast::<u16>();
    // SAFETY: caller enables `V`'s ISA and verified padded search windows.
    let center = unsafe {
        if fast_i16 {
            dmvr_sad_fast::<V>(size, p0, p1, 0, 0)
        } else {
            dmvr_sad::<V>(size, p0, p1, 0, 0)
        }
    };
    let biased_center = center - (center >> 2);
    let mut best = biased_center;
    let area = (width * height) as i32;
    if best < area {
        return ([0; 2], false);
    }
    let mut costs = [0i32; SEARCH_WIDTH * SEARCH_WIDTH];
    let mut displacement = [0i32; 2];
    for y in -2..=2_i32 {
        // SAFETY: `y` lies within `[-2, 2]`, inside the 2-sample border.
        let mut row_costs = unsafe {
            if fast_i16 {
                dmvr_sad_x5_fast::<V>(size, p0, p1, y as isize, y != 0)
            } else {
                dmvr_sad_x5::<V>(size, p0, p1, y as isize, y != 0)
            }
        };
        if y == 0 {
            row_costs[SEARCH_RADIUS] = biased_center;
        }
        let row_base = (y + 2) as usize * SEARCH_WIDTH;
        for (xi, &cost) in row_costs.iter().enumerate() {
            costs[row_base + xi] = cost;
            if cost < best {
                best = cost;
                displacement = [xi as i32 - 2, y];
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

/// Precision of high-precision BDOF/PROF border samples (H.266 §8.5.6.3.1).
const BDOF_PAD_PRECISION: u32 = 14;
/// Minimum left shift of integer border samples (`Max(2, 14 - BitDepth)`).
const MIN_BDOF_PAD_SHIFT: u32 = 2;

#[inline(always)]
fn scale_border_u8<W: I16Lanes, V: I16Lanes, H: I16Lanes>(
    src: &[u8],
    dst: &mut [i16],
    len: usize,
    scale: i16,
) {
    let (sw, sv, sh) = (W::splat(scale), V::splat(scale), H::splat(scale));
    let mut x = 0;
    while x + W::LANES <= len {
        W::load_u8(&src[x..]).mul(sw).store(&mut dst[x..]);
        x += W::LANES;
    }
    while x + V::LANES <= len {
        V::load_u8(&src[x..]).mul(sv).store(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= len {
        H::load_u8(&src[x..]).mul(sh).store(&mut dst[x..]);
        x += H::LANES;
    }
    if x < len {
        let tail = len - H::LANES;
        H::load_u8(&src[tail..]).mul(sh).store(&mut dst[tail..]);
    }
}

#[inline(always)]
fn scale_border_u16<W: I16Lanes, V: I16Lanes, H: I16Lanes>(
    src: &[u16],
    dst: &mut [i16],
    len: usize,
    scale: i16,
) {
    let (sw, sv, sh) = (W::splat(scale), V::splat(scale), H::splat(scale));
    let mut x = 0;
    while x + W::LANES <= len {
        W::load_u16(&src[x..]).mul(sw).store(&mut dst[x..]);
        x += W::LANES;
    }
    while x + V::LANES <= len {
        V::load_u16(&src[x..]).mul(sv).store(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= len {
        H::load_u16(&src[x..]).mul(sh).store(&mut dst[x..]);
        x += H::LANES;
    }
    if x < len {
        let tail = len - H::LANES;
        H::load_u16(&src[tail..]).mul(sh).store(&mut dst[tail..]);
    }
}

#[inline(always)]
fn copy_interior_row<W: I16Lanes, V: I16Lanes, H: I16Lanes>(
    src: &[i16],
    dst: &mut [i16],
    width: usize,
) {
    let mut x = 0;
    while x + W::LANES <= width {
        W::load(&src[x..]).store(&mut dst[x..]);
        x += W::LANES;
    }
    while x + V::LANES <= width {
        V::load(&src[x..]).store(&mut dst[x..]);
        x += V::LANES;
    }
    if x + H::LANES <= width {
        H::load(&src[x..]).store(&mut dst[x..]);
    }
}

#[inline(always)]
pub(super) unsafe fn bdof_pad<W: I16Lanes, V: I16Lanes, H: I16Lanes>(
    (width, height): (usize, usize),
    window: &PlaneView<'_>,
    interior: &[i16],
    out: &mut [i16],
) {
    const SHIFT: u32 = BDOF_PAD_PRECISION - 8;
    const SCALE: i16 = 1 << SHIFT;
    let stride = width + 2;
    let ref_stride = window.stride();
    let data = &window.data()[..(height + 1) * ref_stride + stride];
    let interior = &interior[..width * height];
    let out = &mut out[..stride * (height + 2)];
    scale_border_u8::<W, V, H>(&data[..stride], &mut out[..stride], stride, SCALE);
    let (mut sp, mut ip, mut dp) = (data.as_ptr(), interior.as_ptr(), out.as_mut_ptr());
    // SAFETY: `data`, `interior`, and `out` were validated above for `height + 2` rows.
    unsafe {
        for _ in 0..height {
            sp = sp.add(ref_stride);
            dp = dp.add(stride);
            *dp = (i16::from(*sp)) << SHIFT;
            copy_interior_row::<W, V, H>(
                core::slice::from_raw_parts(ip, width),
                core::slice::from_raw_parts_mut(dp.add(1), width),
                width,
            );
            *dp.add(width + 1) = (i16::from(*sp.add(width + 1))) << SHIFT;
            ip = ip.add(width);
        }
        sp = sp.add(ref_stride);
        dp = dp.add(stride);
        scale_border_u8::<W, V, H>(
            core::slice::from_raw_parts(sp, stride),
            core::slice::from_raw_parts_mut(dp, stride),
            stride,
            SCALE,
        );
    }
}

#[inline(always)]
unsafe fn bdof_pad_u16_w16<W: I16Lanes, V: I16Lanes, H: I16Lanes>(
    height: usize,
    (mut sp, ref_stride): (*const u16, usize),
    mut ip: *const i16,
    mut dp: *mut i16,
    (shift, scale): (u32, i16),
) {
    let (sw, sv, sh) = (W::splat(scale), V::splat(scale), H::splat(scale));
    // SAFETY: caller validated `sp` (`18` cols x `height + 2` rows), `ip`, and `dp`.
    unsafe {
        if W::LANES == 16 {
            W::load_u16_at(sp).mul(sw).store_u16_at(dp.cast());
        } else {
            V::load_u16_at(sp).mul(sv).store_u16_at(dp.cast());
            V::load_u16_at(sp.add(8))
                .mul(sv)
                .store_u16_at(dp.add(8).cast());
        }
        H::load_u16_at(sp.add(14))
            .mul(sh)
            .store_u16_at(dp.add(14).cast());
        for _ in 0..height {
            sp = sp.add(ref_stride);
            dp = dp.add(18);
            *dp = (i32::from(*sp) << shift) as i16;
            if W::LANES == 16 {
                W::load_u16_at(ip.cast()).store_u16_at(dp.add(1).cast());
            } else {
                V::load_u16_at(ip.cast()).store_u16_at(dp.add(1).cast());
                V::load_u16_at(ip.add(8).cast()).store_u16_at(dp.add(9).cast());
            }
            *dp.add(17) = (i32::from(*sp.add(17)) << shift) as i16;
            ip = ip.add(16);
        }
        sp = sp.add(ref_stride);
        dp = dp.add(18);
        if W::LANES == 16 {
            W::load_u16_at(sp).mul(sw).store_u16_at(dp.cast());
        } else {
            V::load_u16_at(sp).mul(sv).store_u16_at(dp.cast());
            V::load_u16_at(sp.add(8))
                .mul(sv)
                .store_u16_at(dp.add(8).cast());
        }
        H::load_u16_at(sp.add(14))
            .mul(sh)
            .store_u16_at(dp.add(14).cast());
    }
}

#[inline(always)]
unsafe fn bdof_pad_u16_w8<V: I16Lanes, H: I16Lanes>(
    height: usize,
    (mut sp, ref_stride): (*const u16, usize),
    mut ip: *const i16,
    mut dp: *mut i16,
    (shift, scale): (u32, i16),
) {
    let (sv, sh) = (V::splat(scale), H::splat(scale));
    // SAFETY: caller validated `sp` (`10` cols x `height + 2` rows), `ip`, and `dp`.
    unsafe {
        V::load_u16_at(sp).mul(sv).store_u16_at(dp.cast());
        H::load_u16_at(sp.add(6))
            .mul(sh)
            .store_u16_at(dp.add(6).cast());
        for _ in 0..height {
            sp = sp.add(ref_stride);
            dp = dp.add(10);
            *dp = (i32::from(*sp) << shift) as i16;
            V::load_u16_at(ip.cast()).store_u16_at(dp.add(1).cast());
            *dp.add(9) = (i32::from(*sp.add(9)) << shift) as i16;
            ip = ip.add(8);
        }
        sp = sp.add(ref_stride);
        dp = dp.add(10);
        V::load_u16_at(sp).mul(sv).store_u16_at(dp.cast());
        H::load_u16_at(sp.add(6))
            .mul(sh)
            .store_u16_at(dp.add(6).cast());
    }
}

#[inline(always)]
pub(super) unsafe fn bdof_pad_u16<W: I16Lanes, V: I16Lanes, H: I16Lanes>(
    (width, height): (usize, usize),
    window: &PlaneView<'_, u16>,
    interior: &[i16],
    out: &mut [i16],
    depth: BitDepth,
) {
    let shift = BDOF_PAD_PRECISION
        .saturating_sub(depth.bits())
        .max(MIN_BDOF_PAD_SHIFT);
    let scale = 1i16 << shift;
    let stride = width + 2;
    let ref_stride = window.stride();
    let data = &window.data()[..(height + 1) * ref_stride + stride];
    let interior = &interior[..width * height];
    let out = &mut out[..stride * (height + 2)];
    let (mut sp, mut ip, mut dp) = (data.as_ptr(), interior.as_ptr(), out.as_mut_ptr());
    if width == 16 {
        // SAFETY: `data`, `interior`, and `out` validated above for `width == 16`.
        unsafe {
            bdof_pad_u16_w16::<W, V, H>(height, (sp, ref_stride), ip, dp, (shift, scale));
        }
        return;
    }
    if width == 8 && V::LANES == 8 {
        // SAFETY: `data`, `interior`, and `out` validated above for `width == 8`.
        unsafe {
            bdof_pad_u16_w8::<V, H>(height, (sp, ref_stride), ip, dp, (shift, scale));
        }
        return;
    }
    scale_border_u16::<W, V, H>(&data[..stride], &mut out[..stride], stride, scale);
    // SAFETY: `data`, `interior`, and `out` were validated above for `height + 2` rows.
    unsafe {
        for _ in 0..height {
            sp = sp.add(ref_stride);
            dp = dp.add(stride);
            *dp = (i32::from(*sp) << shift) as i16;
            copy_interior_row::<W, V, H>(
                core::slice::from_raw_parts(ip, width),
                core::slice::from_raw_parts_mut(dp.add(1), width),
                width,
            );
            *dp.add(width + 1) = (i32::from(*sp.add(width + 1)) << shift) as i16;
            ip = ip.add(width);
        }
        sp = sp.add(ref_stride);
        dp = dp.add(stride);
        scale_border_u16::<W, V, H>(
            core::slice::from_raw_parts(sp, stride),
            core::slice::from_raw_parts_mut(dp, stride),
            stride,
            scale,
        );
    }
}
