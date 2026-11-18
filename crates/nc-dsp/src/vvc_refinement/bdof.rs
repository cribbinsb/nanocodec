//! Optical-flow gradients, six-by-six flow sums and BDOF correction across
//! bit depths (`u8` and `u16`).
#![allow(unsafe_code, clippy::inline_always)]
use core::mem::MaybeUninit;

use nc_common::{BitDepth, Sample};

use crate::simd::{I16FilterLanes, I16Lanes, I32Lanes};
use crate::vvc_bdof::flow;

#[inline(always)]
unsafe fn row_step<L: I16Lanes>(
    p0: *const u16,
    p1: *const u16,
    stride: usize,
    at: usize,
    dx_ptr: *mut u16,
    dy_ptr: *mut u16,
    t_rows: &mut [[i16; 22]; 5],
    x_dst: usize,
) {
    // SAFETY: caller enables `L`'s ISA and validated `(height + 2) * stride`
    // input samples plus `dx_ptr`/`dy_ptr` for `L::LANES` elements.
    unsafe {
        let [g0x, g0y] = super::generic::gradient16_ptr::<L>(p0, stride, at);
        let [g1x, g1y] = super::generic::gradient16_ptr::<L>(p1, stride, at);
        g0x.sub(g1x).store_u16_at(dx_ptr);
        g0y.sub(g1y).store_u16_at(dy_ptr);
        let gx = g0x.add(g1x).sra(1);
        let gy = g0y.add(g1y).sra(1);
        let diff = L::load_u16_at(p1.add(at))
            .sra(4)
            .sub(L::load_u16_at(p0.add(at)).sra(4));
        let sx = gx.max(L::splat(-1)).min(L::splat(1));
        let sy = gy.max(L::splat(-1)).min(L::splat(1));
        gx.abs().store(&mut t_rows[0][x_dst..]);
        gy.abs().store(&mut t_rows[1][x_dst..]);
        sx.mul(diff).store(&mut t_rows[2][x_dst..]);
        sy.mul(diff).store(&mut t_rows[3][x_dst..]);
        sy.mul(gx).store(&mut t_rows[4][x_dst..]);
    }
}

#[inline(always)]
unsafe fn sum6_vec<G: I16Lanes>(p: *const u16, dst: *mut u16) {
    // SAFETY: caller ensures `p` has `G::LANES + 5` readable `i16`s (`<= 22`).
    unsafe {
        let s01 = G::load_u16_at(p).add(G::load_u16_at(p.add(1)));
        let s23 = G::load_u16_at(p.add(2)).add(G::load_u16_at(p.add(3)));
        let s45 = G::load_u16_at(p.add(4)).add(G::load_u16_at(p.add(5)));
        s01.add(s23).add(s45).store_u16_at(dst);
    }
}

#[inline(always)]
unsafe fn prepare_terms<G: I16Lanes, W4: I16Lanes>(
    p0: *const u16,
    p1: *const u16,
    (width, height): (usize, usize),
    dx_buf: &mut [[MaybeUninit<i16>; 16]; 16],
    dy_buf: &mut [[MaybeUninit<i16>; 16]; 16],
    row_sums: &mut [[[MaybeUninit<i16>; 5]; 4]; 18],
) {
    let stride = width + 2;
    let nbx = width / 4;
    let mut t_rows = [[0i16; 22]; 5];
    let mut sum_scratch = [MaybeUninit::<i16>::uninit(); 16];
    for iy in 0..height {
        let y_base = (iy + 1) * stride;
        let dx_row = dx_buf[iy].as_mut_ptr().cast::<u16>();
        let dy_row = dy_buf[iy].as_mut_ptr().cast::<u16>();
        let mut x = 0;
        while G::LANES > 4 && x + G::LANES <= width {
            // SAFETY: `x + G::LANES <= width <= 16`.
            unsafe {
                row_step::<G>(
                    p0,
                    p1,
                    stride,
                    y_base + x + 1,
                    dx_row.add(x),
                    dy_row.add(x),
                    &mut t_rows,
                    x + 1,
                );
            }
            x += G::LANES;
        }
        while x < width {
            // SAFETY: `W4::LANES == 4` and `width` is a multiple of 4 (`<= 16`).
            unsafe {
                row_step::<W4>(
                    p0,
                    p1,
                    stride,
                    y_base + x + 1,
                    dx_row.add(x),
                    dy_row.add(x),
                    &mut t_rows,
                    x + 1,
                );
            }
            x += 4;
        }
        for (k, tr) in t_rows.iter_mut().enumerate() {
            tr[0] = tr[1];
            tr[width + 1] = tr[width];
            let tp = tr.as_ptr().cast::<u16>();
            let sp = sum_scratch.as_mut_ptr().cast::<u16>();
            // SAFETY: `G::LANES <= 16`, `tp` has 22 initialized `i16`s, and
            // `sum6_vec` writes `width.max(G::LANES)` lanes into `sum_scratch`.
            unsafe {
                sum6_vec::<G>(tp, sp);
                if width > G::LANES {
                    sum6_vec::<G>(tp.add(8), sp.add(8));
                }
                for (bx, slot) in row_sums[iy + 1].iter_mut().enumerate().take(nbx) {
                    slot[k].write(*sp.add(bx * 4).cast::<i16>());
                }
            }
        }
    }
    let first = row_sums[1];
    row_sums[0][..nbx].copy_from_slice(&first[..nbx]);
    let last = row_sums[height];
    row_sums[height + 1][..nbx].copy_from_slice(&last[..nbx]);
}

type GradGrid = [[MaybeUninit<i16>; 16]; 16];

#[inline(always)]
unsafe fn apply_subblock_u16<G: I16FilterLanes>(
    (in0, in1): (&[i16], &[i16]),
    (dx_buf, dy_buf): (&GradGrid, &GradGrid),
    (x0, y0, width, stride): (usize, usize, usize, usize),
    vector: [i32; 2],
    (offset4, shift4, max_val): (i32, u32, i16),
    out: &mut [u16],
) {
    let (vx, vy) = (vector[0] as i16, vector[1] as i16);
    let zero = G::splat(0);
    let max_vec = G::splat(max_val);
    let at0 = (y0 + 1) * stride + x0 + 1;
    // SAFETY: `prepare_terms` initialized `dx_buf`/`dy_buf` for `0..height` x `0..width`,
    // `in0`/`in1` cover `(height + 2) * stride`, and `out` covers `width * height`.
    unsafe {
        if G::LANES == 16 {
            let s0 = G::load_quarters([
                &in0[at0..],
                &in0[at0 + stride..],
                &in0[at0 + 2 * stride..],
                &in0[at0 + 3 * stride..],
            ]);
            let s1 = G::load_quarters([
                &in1[at0..],
                &in1[at0 + stride..],
                &in1[at0 + 2 * stride..],
                &in1[at0 + 3 * stride..],
            ]);
            let dx = G::load_quarters([
                core::slice::from_raw_parts(dx_buf[y0].as_ptr().cast::<i16>().add(x0), 4),
                core::slice::from_raw_parts(dx_buf[y0 + 1].as_ptr().cast::<i16>().add(x0), 4),
                core::slice::from_raw_parts(dx_buf[y0 + 2].as_ptr().cast::<i16>().add(x0), 4),
                core::slice::from_raw_parts(dx_buf[y0 + 3].as_ptr().cast::<i16>().add(x0), 4),
            ]);
            let dy = G::load_quarters([
                core::slice::from_raw_parts(dy_buf[y0].as_ptr().cast::<i16>().add(x0), 4),
                core::slice::from_raw_parts(dy_buf[y0 + 1].as_ptr().cast::<i16>().add(x0), 4),
                core::slice::from_raw_parts(dy_buf[y0 + 2].as_ptr().cast::<i16>().add(x0), 4),
                core::slice::from_raw_parts(dy_buf[y0 + 3].as_ptr().cast::<i16>().add(x0), 4),
            ]);
            let mut acc = G::acc_mul_add2(G::acc_zero(), s0, 1, s1, 1);
            acc = G::acc_mul_add2(acc, dx, vx, dy, vy);
            let res = G::acc_round_shift_narrow(acc, offset4, shift4)
                .max(zero)
                .min(max_vec);
            let dst_ptr = out.as_mut_ptr().cast::<i16>();
            res.store_quarters([
                core::slice::from_raw_parts_mut(dst_ptr.add(y0 * width + x0), 4),
                core::slice::from_raw_parts_mut(dst_ptr.add((y0 + 1) * width + x0), 4),
                core::slice::from_raw_parts_mut(dst_ptr.add((y0 + 2) * width + x0), 4),
                core::slice::from_raw_parts_mut(dst_ptr.add((y0 + 3) * width + x0), 4),
            ]);
        } else {
            debug_assert_eq!(G::LANES, 8);
            for pair in [0, 2] {
                let y = y0 + pair;
                let at = (y + 1) * stride + x0 + 1;
                let s0 = G::load_halves(&in0[at..], &in0[at + stride..]);
                let s1 = G::load_halves(&in1[at..], &in1[at + stride..]);
                let dx = G::load_halves(
                    core::slice::from_raw_parts(dx_buf[y].as_ptr().cast::<i16>().add(x0), 4),
                    core::slice::from_raw_parts(dx_buf[y + 1].as_ptr().cast::<i16>().add(x0), 4),
                );
                let dy = G::load_halves(
                    core::slice::from_raw_parts(dy_buf[y].as_ptr().cast::<i16>().add(x0), 4),
                    core::slice::from_raw_parts(dy_buf[y + 1].as_ptr().cast::<i16>().add(x0), 4),
                );
                let mut acc = G::acc_mul_add2(G::acc_zero(), s0, 1, s1, 1);
                acc = G::acc_mul_add2(acc, dx, vx, dy, vy);
                let res = G::acc_round_shift_narrow(acc, offset4, shift4)
                    .max(zero)
                    .min(max_vec);
                let dst_ptr = out.as_mut_ptr().cast::<i16>();
                res.store_halves(
                    core::slice::from_raw_parts_mut(dst_ptr.add(y * width + x0), 4),
                    core::slice::from_raw_parts_mut(dst_ptr.add((y + 1) * width + x0), 4),
                );
            }
        }
    }
}

#[inline(always)]
pub(super) unsafe fn bdof_at<T: Sample, V4: I32Lanes, G: I16FilterLanes, W4: I16Lanes>(
    size: (usize, usize),
    inputs: [&[i16]; 2],
    out: &mut [T],
    depth: BitDepth,
) {
    debug_assert_eq!(V4::LANES, 4);
    debug_assert_eq!(W4::LANES, 4);
    let depth = depth.of::<T>();
    let shift4 = 15u32.saturating_sub(depth.bits()).max(3);
    let offset4 = 1i32 << (shift4 - 1);
    let max_val = depth.max();
    let (width, height) = size;
    let stride = width + 2;
    let count = stride * (height + 2);
    let in0 = &inputs[0][..count];
    let in1 = &inputs[1][..count];
    let out = &mut out[..width * height];
    let mut dx_buf = [[MaybeUninit::<i16>::uninit(); 16]; 16];
    let mut dy_buf = [[MaybeUninit::<i16>::uninit(); 16]; 16];
    let mut row_sums = [[[MaybeUninit::<i16>::uninit(); 5]; 4]; 18];
    // SAFETY: caller enables the SIMD ISA and checked buffer bounds (`width, height <= 16`).
    unsafe {
        prepare_terms::<G, W4>(
            in0.as_ptr().cast::<u16>(),
            in1.as_ptr().cast::<u16>(),
            size,
            &mut dx_buf,
            &mut dy_buf,
            &mut row_sums,
        );
    }
    for by in 0..height / 4 {
        let y0 = by * 4;
        for bx in 0..width / 4 {
            let x0 = bx * 4;
            let mut sums = [0i32; 5];
            for rs in &row_sums[y0..y0 + 6] {
                for (k, s) in sums.iter_mut().enumerate() {
                    // SAFETY: `prepare_terms` initialized `0..height + 2` for `0..width / 4`.
                    *s += i32::from(unsafe { rs[bx][k].assume_init() });
                }
            }
            let vector = flow(sums);
            if core::mem::size_of::<T>() == 2 {
                // SAFETY: `Sample` is `u8` or `u16`, so `size_of::<T>() == 2` is `u16`.
                unsafe {
                    apply_subblock_u16::<G>(
                        (in0, in1),
                        (&dx_buf, &dy_buf),
                        (x0, y0, width, stride),
                        vector,
                        (offset4, shift4, max_val as i16),
                        core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u16>(), out.len()),
                    );
                }
                continue;
            }
            for y in y0..y0 + 4 {
                let at = (y + 1) * stride + x0 + 1;
                // SAFETY: `prepare_terms` initialized `dx_buf[y][0..width]` and `dy_buf[y][0..width]`.
                let (dx, dy) = unsafe {
                    (
                        V4::load_i16(core::slice::from_raw_parts(
                            dx_buf[y].as_ptr().cast::<i16>().add(x0),
                            4,
                        )),
                        V4::load_i16(core::slice::from_raw_parts(
                            dy_buf[y].as_ptr().cast::<i16>().add(x0),
                            4,
                        )),
                    )
                };
                let value = V4::load_i16(&in0[at..])
                    .add(V4::load_i16(&in1[at..]))
                    .add(dx.mul_scalar(vector[0]))
                    .add(dy.mul_scalar(vector[1]))
                    .add(V4::splat(offset4))
                    .sra(shift4)
                    .max(V4::splat(0))
                    .min(V4::splat(max_val));
                let dst = &mut out[y * width + x0..][..4];
                let mut tmp = [0i16; 4];
                value.store_sat16(&mut tmp);
                // SAFETY: `size_of::<T>() == 1` means `T` is `u8`.
                W4::load(&tmp).store_u8_sat(unsafe {
                    core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), 4)
                });
            }
        }
    }
}

#[inline(always)]
pub(super) unsafe fn bdof<V4: I32Lanes, G: I16FilterLanes, W4: I16Lanes>(
    size: (usize, usize),
    inputs: [&[i16]; 2],
    out: &mut [u8],
) {
    // SAFETY: forwarded to `bdof_at` at 8 bits under the same contract.
    unsafe {
        bdof_at::<u8, V4, G, W4>(size, inputs, out, BitDepth::EIGHT);
    }
}
