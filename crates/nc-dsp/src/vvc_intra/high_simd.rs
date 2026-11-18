//! Multi-tier `u16` SIMD VVC intra prediction (square and rectangular Planar,
//! DC, Horizontal, Vertical, Angular 2..=66, and PDPC) over [`I16Lanes`].
//!
//! Shares [`crate::hevc_intra::lanes::planar_u16`] and
//! [`crate::intra_runs::high`] (`run2_u16`, `run4_u16`) with HEVC Main 10.
//! PDPC two-term sums (`wL * (left - v) + wT * (top - v)`) reach `65,504` at
//! 10 bits, so all PDPC terms accumulate in exact 32-bit lanes
//! (`acc_vec_mul_add2` / `acc_mul_add`) before rounding back to `u16`.

#![allow(unsafe_code, clippy::inline_always)]

use nc_common::BitDepth;

use crate::hevc_intra::PlanarEdges;
use crate::intra_runs::high::{run2_u16, run4_u16};
use crate::simd::I16Lanes;

use super::VvcIntraMode;
use super::angular::{Angular, CUBIC, MAIN_LEN, MAX_SIDE};
use super::rectangular::{self, ORIGIN, Projection};

const PDPC_REACH: usize = 3;
const PDPC_SHIFT: u32 = 6;
const PDPC_ROUND: i32 = 1 << (PDPC_SHIFT - 1);
const PDPC_SPAN: usize = 32;
const PDPC_Q15_SHIFT: u32 = 15 - 6;
const RECT_MAX: usize = rectangular::MAX_SIDE;

#[inline(always)]
unsafe fn square_non_angular<V: I16Lanes>(
    n: usize,
    mode: VvcIntraMode,
    top: &[i32],
    left: &[i32],
    corner: i32,
    out: &mut [u16],
    max_val: i16,
) {
    let mut above = [0u16; MAX_SIDE + 1];
    let mut side = [0u16; MAX_SIDE + 1];
    for (dst, &v) in above.iter_mut().zip(&top[..=n]) {
        *dst = v as u16;
    }
    for (dst, &v) in side.iter_mut().zip(&left[..=n]) {
        *dst = v as u16;
    }
    let out = &mut out[..n * n];
    match mode {
        VvcIntraMode::Angular { .. } => unreachable!(),
        VvcIntraMode::Planar => {
            let edges = PlanarEdges {
                above: &above[..n],
                left: &side[..n],
                top_right: above[n],
                bottom_left: side[n],
            };
            // SAFETY: forwarded ISA and `n` multiple-of-`V::LANES` contract.
            unsafe { crate::hevc_intra::lanes::planar_u16::<V>(&edges, n, out) };
        }
        VvcIntraMode::Dc => {
            let mut sum = 0u32;
            for &v in &above[..n] {
                sum += u32::from(v);
            }
            for &v in &side[..n] {
                sum += u32::from(v);
            }
            out.fill(((sum + n as u32) / (2 * n as u32)) as u16);
        }
        VvcIntraMode::Vertical => {
            for line in out.chunks_exact_mut(n) {
                line.copy_from_slice(&above[..n]);
            }
        }
        VvcIntraMode::Horizontal => {
            for (line, &v) in out.chunks_exact_mut(n).zip(&side[..n]) {
                line.fill(v);
            }
        }
    }
    pdpc_u16::<V>(
        (n, n),
        mode,
        &above[..n],
        &side[..n],
        corner as i16,
        out,
        max_val,
    );
}

#[inline(always)]
fn pdpc_u16<V: I16Lanes>(
    (width, height): (usize, usize),
    mode: VvcIntraMode,
    above: &[u16],
    side: &[u16],
    corner: i16,
    out: &mut [u16],
    max_val: i16,
) {
    let scale = (width.ilog2() + height.ilog2() - 2) >> 2;
    let reach_x = (PDPC_REACH << scale).min(width);
    let reach_y = (PDPC_REACH << scale).min(height);
    let mut weights = [0i16; RECT_MAX];
    for (d, weight) in weights[..reach_x].iter_mut().enumerate() {
        *weight = 32 >> ((2 * d) >> scale);
    }
    let (zero, hi) = (V::splat(0), V::splat(max_val));
    let corner_v = V::splat(corner);
    let reached_cols = reach_x.div_ceil(V::LANES) * V::LANES;
    for (y, line) in out[..width * height].chunks_exact_mut(width).enumerate() {
        let wt_scalar = if y < reach_y {
            32i16 >> ((2 * y) >> scale)
        } else {
            0
        };
        let wt = V::splat(wt_scalar);
        let left = V::splat(side[y] as i16);
        let columns = match mode {
            VvcIntraMode::Horizontal if y >= reach_y => 0,
            _ if y >= reach_y => reached_cols.min(width),
            _ => width,
        };
        for x in (0..columns).step_by(V::LANES) {
            let value = V::load_u16(&line[x..]);
            let wl = V::load(&weights[x..]);
            let top_v = V::load_u16(&above[x..]);
            let acc = match mode {
                VvcIntraMode::Angular { .. } => unreachable!(),
                VvcIntraMode::Planar | VvcIntraMode::Dc => {
                    V::acc_vec_mul_add2(left.sub(value), wl, top_v.sub(value), wt)
                }
                VvcIntraMode::Horizontal => {
                    V::acc_mul_add(V::acc_zero(), top_v.sub(corner_v), wt_scalar)
                }
                VvcIntraMode::Vertical => V::acc_vec_mul_add2(left.sub(corner_v), wl, zero, zero),
            };
            let corr = V::acc_round_shift_narrow(acc, PDPC_ROUND, PDPC_SHIFT);
            value
                .add_sat(corr)
                .max(zero)
                .min(hi)
                .store_u16(&mut line[x..]);
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn square_angular<V: I16Lanes>(
    n: usize,
    direction: u8,
    luma: bool,
    top: &[i32],
    left: &[i32],
    corner: i32,
    transposed_ok: bool,
    out: &mut [u16],
    max_val: i16,
) -> bool {
    let angular = Angular::new(n, direction, luma, top, left, corner);
    let mut main = [0u16; MAIN_LEN];
    for (dst, &v) in main.iter_mut().zip(angular.main()) {
        *dst = v as u16;
    }
    let pdpc_width = angular
        .pdpc_scale
        .map_or(0, |scale| (3usize << scale).min(n));
    let pdpc_span = (pdpc_width.div_ceil(V::LANES) * V::LANES).min(n);
    let mut weights = [0i16; PDPC_SPAN];
    let mut offsets = [0usize; PDPC_SPAN];
    if let Some(scale) = angular.pdpc_scale {
        for x in 0..pdpc_width {
            weights[x] = (32 >> ((2 * x) >> scale)) << PDPC_Q15_SHIFT;
            offsets[x] = angular.side_offset(x);
        }
    }
    let exact = angular.angle.abs() == 32;
    let in_place = angular.vertical || transposed_ok;
    let mut predicted = [0u16; MAX_SIDE];
    for y in 0..n {
        let start = angular.row_start(y);
        let phase = angular.phase(y);
        let side_row = angular.side_row(y);
        let mut side = [0i16; PDPC_SPAN];
        for x in 0..pdpc_width {
            side[x] = side_row[offsets[x]] as i16;
        }
        let line = if in_place {
            &mut out[y * n..(y + 1) * n]
        } else {
            &mut predicted[..n]
        };
        // SAFETY: forwarded ISA contract; `main` holds the extended window.
        unsafe {
            if exact {
                line.copy_from_slice(&main[start + 1..=start + n]);
            } else if luma {
                let c = angular.coefficients(phase);
                let taps = [c[0] as i16, c[1] as i16, c[2] as i16, c[3] as i16];
                run4_u16::<V>(&main[start..start + n + 3], taps, line, max_val);
            } else {
                run2_u16::<V>(&main[start + 1..=start + 1 + n], phase as usize, line);
            }
        }
        for x in (0..pdpc_span).step_by(V::LANES) {
            let value = V::load_u16(&line[x..]);
            value
                .add(
                    V::load(&side[x..])
                        .sub(value)
                        .mul_q15_round(V::load(&weights[x..])),
                )
                .store_u16(&mut line[x..]);
        }
        if !in_place {
            for (x, &v) in predicted[..n].iter().enumerate() {
                out[x * n + y] = v;
            }
        }
    }
    !angular.vertical && transposed_ok
}

#[inline(always)]
unsafe fn rect_smooth<V: I16Lanes>(
    (width, height): (usize, usize),
    planar: bool,
    top: &[i32],
    left: &[i32],
    out: &mut [u16],
    max_val: i16,
) {
    let (log2_w, log2_h) = (width.ilog2(), height.ilog2());
    let mut above = [0u16; RECT_MAX];
    let mut side = [0u16; RECT_MAX];
    for (d, &v) in above[..width].iter_mut().zip(&top[..width]) {
        *d = v as u16;
    }
    for (d, &v) in side[..height].iter_mut().zip(&left[..height]) {
        *d = v as u16;
    }
    let out = &mut out[..width * height];
    if planar {
        let shift = log2_w + log2_h + 1;
        let mut lw = [0i16; RECT_MAX];
        let mut rw = [0i16; RECT_MAX];
        for x in 0..width {
            lw[x] = (((width - 1 - x) as i32) << log2_h) as i16;
            rw[x] = (((x + 1) as i32) << log2_h) as i16;
        }
        let top_right = top[width] as i16;
        let bottom_left = left[height];
        for (y, line) in out.chunks_exact_mut(width).enumerate() {
            let left_y = side[y] as i16;
            let above_w = (((height - 1 - y) as i32) << log2_w) as i16;
            let constant = (((y as i32 + 1) * bottom_left) << log2_w) + (width * height) as i32;
            for x in (0..width).step_by(V::LANES) {
                let mut acc = V::acc_mul_add2(
                    V::acc_zero(),
                    V::load(&lw[x..]),
                    left_y,
                    V::load(&rw[x..]),
                    top_right,
                );
                acc = V::acc_mul_add(acc, V::load_u16(&above[x..]), above_w);
                V::acc_round_shift_narrow(acc, constant, shift).store_u16(&mut line[x..]);
            }
        }
    } else {
        let dc = match width.cmp(&height) {
            std::cmp::Ordering::Equal => {
                let mut s = 0i32;
                for &v in &top[..width] {
                    s += v;
                }
                for &v in &left[..height] {
                    s += v;
                }
                (s + width as i32) >> (log2_w + 1)
            }
            std::cmp::Ordering::Greater => {
                let mut s = 0i32;
                for &v in &top[..width] {
                    s += v;
                }
                (s + (width as i32 >> 1)) >> log2_w
            }
            std::cmp::Ordering::Less => {
                let mut s = 0i32;
                for &v in &left[..height] {
                    s += v;
                }
                (s + (height as i32 >> 1)) >> log2_h
            }
        };
        out.fill(dc as u16);
    }
    let mode = if planar {
        VvcIntraMode::Planar
    } else {
        VvcIntraMode::Dc
    };
    pdpc_u16::<V>(
        (width, height),
        mode,
        &above[..width],
        &side[..height],
        0,
        out,
        max_val,
    );
}

#[inline(always)]
fn rect_pdpc_setup(
    (width, height): (usize, usize),
    columns: usize,
    rows: usize,
    p: &Projection,
    weights: &mut [i16; RECT_MAX],
    side_offs: &mut [usize; RECT_MAX],
) -> usize {
    let axis_scale = (width.ilog2() + height.ilog2() - 2) >> 2;
    if p.angle == 0 {
        let reach = (3usize << axis_scale).min(columns);
        for (x, w) in weights[..reach].iter_mut().enumerate() {
            *w = 32 >> ((2 * x) >> axis_scale);
        }
        return reach;
    }
    if p.angle > 0 {
        let scale = (rows.ilog2() as i32 - ((3 * p.inverse - 2).ilog2() as i32 - 8)).min(2);
        if scale >= 0 {
            let scale = scale as u32;
            let reach = (3usize << scale).min(columns);
            for x in 0..reach {
                weights[x] = (32 >> ((2 * x) >> scale)) << PDPC_Q15_SHIFT;
                side_offs[x] = ((256 + (x as i32 + 1) * p.inverse) >> 9) as usize;
            }
            return reach;
        }
    }
    0
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn rect_angular<V: I16Lanes>(
    (width, height): (usize, usize),
    direction: u8,
    luma: bool,
    (top, left, corner): (&[i32], &[i32], i32),
    out: &mut [u16],
    max_val: i16,
) {
    let p = Projection::new(width, height, direction, luma, top, left, corner, 0, None);
    let mut main = [0u16; 5 * RECT_MAX + 8];
    for (dst, &v) in main.iter_mut().zip(&p.main) {
        *dst = v as u16;
    }
    let (columns, rows) = if p.vertical {
        (width, height)
    } else {
        (height, width)
    };
    let mut weights = [0i16; RECT_MAX];
    let mut side_offs = [0usize; RECT_MAX];
    let pdpc_cols = rect_pdpc_setup(
        (width, height),
        columns,
        rows,
        &p,
        &mut weights,
        &mut side_offs,
    );
    let pdpc_span = (pdpc_cols.div_ceil(V::LANES) * V::LANES).min(columns);
    let (zero, hi) = (V::splat(0), V::splat(max_val));
    let mut predicted = [0u16; RECT_MAX];
    for y in 0..rows {
        let delta = (y as i32 + 1) * p.angle;
        let start = (ORIGIN as i32 + (delta >> 5)) as usize;
        let phase = delta & 31;
        let line = if p.vertical {
            &mut out[y * width..(y + 1) * width]
        } else {
            &mut predicted[..columns]
        };
        // SAFETY: forwarded ISA contract; `main` holds the extended projection.
        unsafe {
            if p.angle % 32 == 0 {
                line.copy_from_slice(&main[start + 1..=start + columns]);
            } else if luma {
                let f = if p.gaussian {
                    let half = phase >> 1;
                    [16 - half, 32 - half, 16 + half, half]
                } else {
                    CUBIC[phase as usize]
                };
                run4_u16::<V>(
                    &main[start..start + columns + 3],
                    [f[0] as i16, f[1] as i16, f[2] as i16, f[3] as i16],
                    line,
                    max_val,
                );
            } else {
                run2_u16::<V>(&main[start + 1..=start + 1 + columns], phase as usize, line);
            }
        }
        if p.angle == 0 {
            let diff = V::splat((p.side[y + 1] - corner) as i16);
            for x in (0..pdpc_span).step_by(V::LANES) {
                let value = V::load_u16(&line[x..]);
                let acc = V::acc_vec_mul_add2(diff, V::load(&weights[x..]), zero, zero);
                let corr = V::acc_round_shift_narrow(acc, PDPC_ROUND, PDPC_SHIFT);
                value
                    .add_sat(corr)
                    .max(zero)
                    .min(hi)
                    .store_u16(&mut line[x..]);
            }
        } else if pdpc_span > 0 {
            let mut side = [0i16; RECT_MAX];
            for x in 0..pdpc_cols {
                side[x] = p.side[y + 1 + side_offs[x]] as i16;
            }
            for x in (0..pdpc_span).step_by(V::LANES) {
                let value = V::load_u16(&line[x..]);
                value
                    .add_sat(
                        V::load(&side[x..])
                            .sub(value)
                            .mul_q15_round(V::load(&weights[x..])),
                    )
                    .max(zero)
                    .min(hi)
                    .store_u16(&mut line[x..]);
            }
        }
        if !p.vertical {
            for (x, &v) in predicted[..columns].iter().enumerate() {
                out[x * width + y] = v;
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn predict_square_lanes<A4: I16Lanes, A8: I16Lanes, A16: I16Lanes, A32: I16Lanes>(
    n: usize,
    mode: VvcIntraMode,
    top: &[i32],
    left: &[i32],
    corner: i32,
    transposed_ok: bool,
    out: &mut [u16],
    depth: BitDepth,
) -> bool {
    if depth.bits() > 10 {
        return super::scalar_at(n, mode, top, left, corner, transposed_ok, out, depth);
    }
    let max_val = depth.max() as i16;
    let mode = match mode {
        VvcIntraMode::Angular { direction: 18, .. } => VvcIntraMode::Horizontal,
        VvcIntraMode::Angular { direction: 50, .. } => VvcIntraMode::Vertical,
        mode => mode,
    };
    // SAFETY: caller enables lane ISAs; `n` is 4, 8, 16, or 32.
    unsafe {
        if let VvcIntraMode::Angular { direction, luma } = mode {
            match n {
                4 => square_angular::<A4>(
                    n,
                    direction,
                    luma,
                    top,
                    left,
                    corner,
                    transposed_ok,
                    out,
                    max_val,
                ),
                8 => square_angular::<A8>(
                    n,
                    direction,
                    luma,
                    top,
                    left,
                    corner,
                    transposed_ok,
                    out,
                    max_val,
                ),
                16 => square_angular::<A16>(
                    n,
                    direction,
                    luma,
                    top,
                    left,
                    corner,
                    transposed_ok,
                    out,
                    max_val,
                ),
                _ => square_angular::<A32>(
                    n,
                    direction,
                    luma,
                    top,
                    left,
                    corner,
                    transposed_ok,
                    out,
                    max_val,
                ),
            }
        } else {
            match n {
                4 => square_non_angular::<A4>(n, mode, top, left, corner, out, max_val),
                8 => square_non_angular::<A8>(n, mode, top, left, corner, out, max_val),
                16 => square_non_angular::<A16>(n, mode, top, left, corner, out, max_val),
                _ => square_non_angular::<A32>(n, mode, top, left, corner, out, max_val),
            }
            false
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn predict_rect_lanes<A4: I16Lanes, A8: I16Lanes, A16: I16Lanes, A32: I16Lanes>(
    width: usize,
    height: usize,
    mode: VvcIntraMode,
    top: &[i32],
    left: &[i32],
    corner: i32,
    out: &mut [u16],
    depth: BitDepth,
) {
    if width < 4 || height < 4 || depth.bits() > 10 {
        rectangular::predict(width, height, mode, top, left, corner, out, depth);
        return;
    }
    let max_val = depth.max() as i16;
    let dir = match mode {
        VvcIntraMode::Horizontal => Some((18, false)),
        VvcIntraMode::Vertical => Some((50, false)),
        VvcIntraMode::Angular { direction, luma } => Some((direction, luma)),
        _ => None,
    };
    // SAFETY: caller enables lane ISAs; `width` and `height` are powers of two >= 4.
    unsafe {
        if let Some((direction, luma)) = dir {
            let proj_mode = rectangular::wide_angle(width, height, direction);
            let cols = if proj_mode >= 34 { width } else { height };
            match cols {
                4 => rect_angular::<A4>(
                    (width, height),
                    direction,
                    luma,
                    (top, left, corner),
                    out,
                    max_val,
                ),
                8 => rect_angular::<A8>(
                    (width, height),
                    direction,
                    luma,
                    (top, left, corner),
                    out,
                    max_val,
                ),
                16 => rect_angular::<A16>(
                    (width, height),
                    direction,
                    luma,
                    (top, left, corner),
                    out,
                    max_val,
                ),
                _ => rect_angular::<A32>(
                    (width, height),
                    direction,
                    luma,
                    (top, left, corner),
                    out,
                    max_val,
                ),
            }
        } else {
            let planar = mode == VvcIntraMode::Planar;
            match width {
                4 => rect_smooth::<A4>((width, height), planar, top, left, out, max_val),
                8 => rect_smooth::<A8>((width, height), planar, top, left, out, max_val),
                16 => rect_smooth::<A16>((width, height), planar, top, left, out, max_val),
                _ => rect_smooth::<A32>((width, height), planar, top, left, out, max_val),
            }
        }
    }
}

macro_rules! isa_vvc_intra_u16 {
    ($mod_name:ident, $a4:ty, $a8:ty, $a16:ty, $a32:ty, $feature:literal) => {
        pub(crate) mod $mod_name {
            use nc_common::BitDepth;

            use super::super::VvcIntraMode;
            use super::{predict_rect_lanes, predict_square_lanes};

            #[target_feature(enable = $feature)]
            #[allow(clippy::too_many_arguments)]
            pub(crate) unsafe fn predict(
                n: usize,
                mode: VvcIntraMode,
                top: &[i32],
                left: &[i32],
                corner: i32,
                transposed_ok: bool,
                out: &mut [u16],
                depth: BitDepth,
            ) -> bool {
                // SAFETY: `$feature` enabled; caller validated square inputs.
                unsafe {
                    predict_square_lanes::<$a4, $a8, $a16, $a32>(
                        n,
                        mode,
                        top,
                        left,
                        corner,
                        transposed_ok,
                        out,
                        depth,
                    )
                }
            }

            #[target_feature(enable = $feature)]
            #[allow(clippy::too_many_arguments)]
            pub(crate) unsafe fn predict_rect(
                width: usize,
                height: usize,
                mode: VvcIntraMode,
                top: &[i32],
                left: &[i32],
                corner: i32,
                out: &mut [u16],
                depth: BitDepth,
            ) {
                // SAFETY: `$feature` enabled; caller validated rectangular inputs.
                unsafe {
                    predict_rect_lanes::<$a4, $a8, $a16, $a32>(
                        width, height, mode, top, left, corner, out, depth,
                    );
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
isa_vvc_intra_u16!(
    avx2,
    crate::simd::avx2::I16x4,
    crate::simd::avx2::I16x8,
    crate::simd::avx2::I16x16,
    crate::simd::avx2::I16x16,
    "avx2"
);

#[cfg(target_arch = "x86_64")]
isa_vvc_intra_u16!(
    avx512,
    crate::simd::avx2::I16x4,
    crate::simd::avx2::I16x8,
    crate::simd::avx2::I16x16,
    crate::simd::avx512_i16::I16x32,
    "avx512f,avx512bw,avx512vl"
);

#[cfg(target_arch = "aarch64")]
isa_vvc_intra_u16!(
    neon,
    crate::simd::neon::I16x4,
    crate::simd::neon::I16x8,
    crate::simd::neon::I16x8,
    crate::simd::neon::I16x8,
    "neon"
);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
isa_vvc_intra_u16!(
    simd128,
    crate::simd::simd128::I16x4,
    crate::simd::simd128::I16x8,
    crate::simd::simd128::I16x8,
    crate::simd::simd128::I16x8,
    "simd128"
);
