//! High-bit-depth (`u16`, 10- and 12-bit) AV1 loop restoration SIMD kernels
//! (Wiener over [`I16Lanes`] and Self-Guided over [`SgrLanes`]), plus 16-lane
//! AVX-512 Self-Guided kernels shared by both `u8` and `u16`.
//!
//! Attribution: stripe ring schedule, 5-tap outer-zero specialization, and
//! reciprocal `A` factor follow dav1d 1.5.4 (`src/looprestoration_tmpl.c`,
//! `src/x86/looprestoration16_avx2.asm`, `src/x86/looprestoration16_avx512.asm`,
//! `src/arm/64/looprestoration16.S`, BSD-2-Clause; see `THIRD_PARTY.md`).

#![allow(
    clippy::many_single_char_names,
    reason = "the spec's and dav1d's sample, tap and statistic names"
)]
#![allow(unsafe_code, clippy::inline_always)]
#![cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code, unused_imports)
)]

use nc_common::{BitDepth, PlaneViewMut};

use super::kernels::{RowKernels, WIENER_ROWS};
use super::self_guided::{
    blend_at, box_ab_at as scalar_box_ab_at, box_sums as scalar_box_sums,
    filter_around as scalar_around, filter_between as scalar_between, filter_on as scalar_on,
};
use super::sgr_lanes::{self, SgrLanes};
use super::wiener::{WienerTaps, h_row_at, v_row_at};
use super::{AV1_LR_BORDER, AV1_LR_OVERREAD};
use crate::Isa;
use crate::simd::I16Lanes;

/// Smallest horizontal Wiener intermediate at 10 and 12 bits (`-2^13`).
const H_CLAMP_MIN: i16 = -8192;
/// Largest horizontal Wiener intermediate at 10 and 12 bits (`3 * 2^13 - 1`).
const H_CLAMP_MAX: i16 = 24575;

pub(crate) type WienerStripe16Fn = unsafe fn(
    horizontal: WienerTaps,
    vertical: WienerTaps,
    window: (&[u16], usize),
    size: (usize, usize),
    ring: &mut [i16],
    dst: (&mut PlaneViewMut<'_, u16>, usize, usize),
    depth: BitDepth,
);
pub(crate) type WienerH16Fn =
    unsafe fn(src: &[u16], taps: WienerTaps, out: &mut [i16], n: usize, depth: BitDepth);
pub(crate) type WienerV16Fn =
    unsafe fn(rows: &[&[i16]; WIENER_ROWS], taps: WienerTaps, dst: &mut [u16], depth: BitDepth);
pub(crate) type BoxSums16Fn =
    unsafe fn(src: &[u16], sum: &mut [i16], squares: &mut [i32], n: usize);
pub(crate) type BoxAb16Fn<const N: usize> = unsafe fn(
    sums: [&[i16]; N],
    squares: [&[i32]; N],
    scale: u32,
    ab: (&mut [i16], &mut [i32]),
    n: usize,
    depth: BitDepth,
);
pub(crate) type Between16Fn =
    unsafe fn(a: [&[i16]; 2], b: [&[i32]; 2], src: &[u16], out: &mut [i32], n: usize);
pub(crate) type On16Fn = unsafe fn(a: &[i16], b: &[i32], src: &[u16], out: &mut [i32], n: usize);
pub(crate) type Around16Fn =
    unsafe fn(a: [&[i16]; 3], b: [&[i32]; 3], src: &[u16], out: &mut [i32], n: usize);
pub(crate) type Blend16Fn =
    unsafe fn(src: &[u16], filtered: [&[i32]; 2], weights: [i32; 2], dst: &mut [u16], BitDepth);

/// High-bit-depth (`u16`) loop restoration kernels for one ISA.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HighLrFns {
    pub wiener_stripe: Option<WienerStripe16Fn>,
    pub wiener_h: WienerH16Fn,
    pub wiener_v: WienerV16Fn,
    pub box_sums3: BoxSums16Fn,
    pub box_sums5: BoxSums16Fn,
    pub box_ab3: BoxAb16Fn<3>,
    pub box_ab5: BoxAb16Fn<5>,
    pub between: Between16Fn,
    pub on: On16Fn,
    pub around: Around16Fn,
    pub blend: Blend16Fn,
}

fn scalar_box_ab16<const N: usize>(
    sums: [&[i16]; N],
    squares: [&[i32]; N],
    scale: u32,
    ab: (&mut [i16], &mut [i32]),
    n: usize,
    depth: BitDepth,
) {
    scalar_box_ab_at(sums, squares, scale, ab, (n, depth.above_eight()));
}

impl HighLrFns {
    pub const SCALAR: Self = Self {
        wiener_stripe: None,
        wiener_h: h_row_at::<u16>,
        wiener_v: v_row_at::<u16>,
        box_sums3: scalar_box_sums::<u16, 1>,
        box_sums5: scalar_box_sums::<u16, 2>,
        box_ab3: scalar_box_ab16::<3>,
        box_ab5: scalar_box_ab16::<5>,
        between: scalar_between::<u16, i32>,
        on: scalar_on::<u16, i32>,
        around: scalar_around::<u16, i32>,
        blend: blend_at::<u16, i32>,
    };

    pub fn for_isa(isa: Isa) -> Self {
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => x86_avx2::FNS,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => x86_avx512::FNS,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => neon_high::FNS,
            // As for 8 bits ([`super::kernels::kernels`]): SVE2 runs the NEON rows.
            #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
            Isa::Sve2 => neon_high::FNS,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Isa::Simd128 => simd128_high::FNS,
            _ => Self::SCALAR,
        }
    }

    pub fn at(&self, depth: BitDepth) -> HighLrBound<'_> {
        HighLrBound { fns: self, depth }
    }
}

/// [`HighLrFns`] bound to a runtime [`BitDepth`] (`10` or `12`).
pub(crate) struct HighLrBound<'a> {
    fns: &'a HighLrFns,
    depth: BitDepth,
}

fn rows_cover<T>(rows: &[&[T]], n: usize) -> bool {
    rows.iter().all(|row| row.len() >= n + AV1_LR_OVERREAD)
}

impl RowKernels<u16> for HighLrBound<'_> {
    type Filtered = i32;

    fn wiener_stripe(
        &self,
        taps: (WienerTaps, WienerTaps),
        window: (&[u16], usize),
        size: (usize, usize),
        ring: &mut [i16],
        dst: (&mut PlaneViewMut<'_, u16>, usize, usize),
    ) -> bool {
        let Some(kernel) = self.fns.wiener_stripe else {
            return false;
        };
        // SAFETY: `fns` was built for a supported ISA; `kernel` validates extents.
        unsafe { kernel(taps.0, taps.1, window, size, ring, dst, self.depth) };
        true
    }

    fn wiener_h(&self, src: &[u16], taps: WienerTaps, out: &mut [i16], n: usize) {
        assert!(src.len() >= n + 6 + AV1_LR_OVERREAD && out.len() >= n + AV1_LR_OVERREAD);
        // SAFETY: supported ISA; lengths checked above.
        unsafe { (self.fns.wiener_h)(src, taps, out, n, self.depth) }
    }

    fn wiener_v(&self, rows: &[&[i16]; WIENER_ROWS], taps: WienerTaps, dst: &mut [u16]) {
        assert!(rows.iter().all(|r| r.len() >= dst.len() + AV1_LR_OVERREAD));
        // SAFETY: supported ISA; lengths checked above.
        unsafe { (self.fns.wiener_v)(rows, taps, dst, self.depth) }
    }

    fn sgr_box_sums(
        &self,
        radius: usize,
        src: &[u16],
        sum: &mut [i16],
        squares: &mut [i32],
        n: usize,
    ) {
        assert!(src.len() >= n + 2 * radius + AV1_LR_OVERREAD);
        assert!(sum.len() >= n + AV1_LR_OVERREAD && squares.len() >= n + AV1_LR_OVERREAD);
        let kernel = match radius {
            1 => self.fns.box_sums3,
            2 => self.fns.box_sums5,
            _ => unreachable!("self-guided radii are 1 and 2"),
        };
        // SAFETY: supported ISA; lengths checked above.
        unsafe { kernel(src, sum, squares, n) }
    }

    fn sgr_box_ab3(
        &self,
        sums: [&[i16]; 3],
        squares: [&[i32]; 3],
        scale: u32,
        (a, b): (&mut [i16], &mut [i32]),
        n: usize,
    ) {
        assert!(rows_cover(&sums, n) && rows_cover(&squares, n));
        assert!(a.len() >= n + AV1_LR_OVERREAD && b.len() >= n + AV1_LR_OVERREAD);
        // SAFETY: supported ISA; lengths checked above.
        unsafe { (self.fns.box_ab3)(sums, squares, scale, (a, b), n, self.depth) }
    }

    fn sgr_box_ab5(
        &self,
        sums: [&[i16]; 5],
        squares: [&[i32]; 5],
        scale: u32,
        (a, b): (&mut [i16], &mut [i32]),
        n: usize,
    ) {
        assert!(rows_cover(&sums, n) && rows_cover(&squares, n));
        assert!(a.len() >= n + AV1_LR_OVERREAD && b.len() >= n + AV1_LR_OVERREAD);
        // SAFETY: supported ISA; lengths checked above.
        unsafe { (self.fns.box_ab5)(sums, squares, scale, (a, b), n, self.depth) }
    }

    fn sgr_filter_between(
        &self,
        a: [&[i16]; 2],
        b: [&[i32]; 2],
        src: &[u16],
        out: &mut [i32],
        n: usize,
    ) {
        assert!(rows_cover(&a, n + 2) && rows_cover(&b, n + 2));
        assert!(src.len() >= n + AV1_LR_OVERREAD && out.len() >= n + AV1_LR_OVERREAD);
        // SAFETY: supported ISA; lengths checked above.
        unsafe { (self.fns.between)(a, b, src, out, n) }
    }

    fn sgr_filter_on(&self, a: &[i16], b: &[i32], src: &[u16], out: &mut [i32], n: usize) {
        assert!(rows_cover(&[a], n + 2) && rows_cover(&[b], n + 2));
        assert!(src.len() >= n + AV1_LR_OVERREAD && out.len() >= n + AV1_LR_OVERREAD);
        // SAFETY: supported ISA; lengths checked above.
        unsafe { (self.fns.on)(a, b, src, out, n) }
    }

    fn sgr_filter_around(
        &self,
        a: [&[i16]; 3],
        b: [&[i32]; 3],
        src: &[u16],
        out: &mut [i32],
        n: usize,
    ) {
        assert!(rows_cover(&a, n + 2) && rows_cover(&b, n + 2));
        assert!(src.len() >= n + AV1_LR_OVERREAD && out.len() >= n + AV1_LR_OVERREAD);
        // SAFETY: supported ISA; lengths checked above.
        unsafe { (self.fns.around)(a, b, src, out, n) }
    }

    fn sgr_blend(&self, src: &[u16], filtered: [&[i32]; 2], weights: [i32; 2], dst: &mut [u16]) {
        let n = dst.len();
        assert!(src.len() >= n + AV1_LR_OVERREAD && rows_cover(&filtered, n));
        // SAFETY: supported ISA; lengths checked above.
        unsafe { (self.fns.blend)(src, filtered, weights, dst, self.depth) }
    }
}

// ----------------------------------------- lane-generic u16 Wiener ----

#[inline(always)]
unsafe fn load_u16_as_i16<V: I16Lanes>(ptr: *const u16) -> V {
    // SAFETY: caller validated `V::LANES` readable `u16` samples (`0..=4095`).
    V::load(unsafe { core::slice::from_raw_parts(ptr.cast::<i16>(), V::LANES) })
}

#[inline(always)]
unsafe fn load_i16_at<V: I16Lanes>(ptr: *const i16) -> V {
    // SAFETY: caller validated `V::LANES` readable `i16` values.
    V::load(unsafe { core::slice::from_raw_parts(ptr, V::LANES) })
}

#[inline(always)]
unsafe fn wiener_h_vec<V: I16Lanes, const FIVE: bool, const ROUND0: u32, const BIAS: i16>(
    at: *const u16,
    [c0, c1, c2, c3]: WienerTaps,
) -> V {
    let centre = c3 + 128;
    let round = 1_i32 << (ROUND0 - 1);
    let low = V::splat(H_CLAMP_MIN);
    let high = V::splat(H_CLAMP_MAX);
    let bias = V::splat(BIAS);
    // SAFETY: caller checked `V::LANES + 6` readable `u16`s starting at `at`.
    unsafe {
        let s1 = load_u16_as_i16::<V>(at.add(1)).add(load_u16_as_i16::<V>(at.add(5)));
        let s2 = load_u16_as_i16::<V>(at.add(2)).add(load_u16_as_i16::<V>(at.add(4)));
        let s3 = load_u16_as_i16::<V>(at.add(3));
        let acc = if FIVE {
            let a = V::acc_mul_add2(V::acc_zero(), s1, c1, s2, c2);
            V::acc_mul_add(a, s3, centre)
        } else {
            let s0 = load_u16_as_i16::<V>(at).add(load_u16_as_i16::<V>(at.add(6)));
            let a = V::acc_mul_add2(V::acc_zero(), s0, c0, s1, c1);
            V::acc_mul_add2(a, s2, c2, s3, centre)
        };
        V::acc_round_shift_narrow(acc, round, ROUND0)
            .max(low)
            .min(high)
            .add(bias)
    }
}

#[inline(always)]
fn wiener_v5_vec<V: I16Lanes, const ROUND1: u32>(
    [r1, r2, r3, r4, r5]: [V; 5],
    [_c0, c1, c2, c3]: WienerTaps,
    zero: V,
    cap: V,
) -> V {
    let mut acc = V::acc_mul_add2(V::acc_zero(), r1, c1, r5, c1);
    acc = V::acc_mul_add2(acc, r2, c2, r4, c2);
    acc = V::acc_mul_add(acc, r3, c3);
    V::acc_round_shift_narrow(acc, 0, ROUND1).max(zero).min(cap)
}

#[inline(always)]
fn wiener_v7_vec<V: I16Lanes, const ROUND1: u32>(
    [r0, r1, r2, r3, r4, r5, r6]: [V; 7],
    [c0, c1, c2, c3]: WienerTaps,
    zero: V,
    cap: V,
) -> V {
    let mut acc = V::acc_mul_add2(V::acc_zero(), r0, c0, r6, c0);
    acc = V::acc_mul_add2(acc, r1, c1, r5, c1);
    acc = V::acc_mul_add2(acc, r2, c2, r4, c2);
    acc = V::acc_mul_add(acc, r3, c3);
    V::acc_round_shift_narrow(acc, 0, ROUND1).max(zero).min(cap)
}

#[inline(always)]
unsafe fn store_tail_u16<V: I16Lanes>(v: V, dst: *mut u16, rem: usize) {
    let mut tail = [0i16; 32];
    v.store(&mut tail[..V::LANES]);
    // SAFETY: caller checked `rem < V::LANES` writable `u16`s at `dst`.
    unsafe { core::ptr::copy_nonoverlapping(tail.as_ptr().cast::<u16>(), dst, rem) };
}

#[inline(always)]
unsafe fn wiener_h_ptr<V: I16Lanes, const FIVE: bool, const ROUND0: u32, const BIAS: i16>(
    src: *const u16,
    taps: WienerTaps,
    out: *mut i16,
    blocks: usize,
) {
    // SAFETY: caller checked `blocks * V::LANES + 6` readable `u16`s and
    // `blocks * V::LANES` writable `i16`s.
    unsafe {
        for block in 0..blocks {
            let j = block * V::LANES;
            let v = wiener_h_vec::<V, FIVE, ROUND0, BIAS>(src.add(j), taps);
            v.store(core::slice::from_raw_parts_mut(out.add(j), V::LANES));
        }
    }
}

#[inline(always)]
unsafe fn wiener_v_ptr<V: I16Lanes, const FIVE: bool, const ROUND1: u32>(
    rows: &[*const i16; WIENER_ROWS],
    taps: WienerTaps,
    dst: *mut u16,
    n: usize,
    max_val: i16,
) {
    let zero = V::splat(0);
    let cap = V::splat(max_val);
    let full = n / V::LANES;
    // SAFETY: caller checked `n.div_ceil(V::LANES) * V::LANES` readable `i16`s
    // per row and `n` writable `u16`s in `dst`.
    unsafe {
        for block in 0..full {
            let j = block * V::LANES;
            let v = if FIVE {
                let r = [
                    load_i16_at::<V>(rows[1].add(j)),
                    load_i16_at::<V>(rows[2].add(j)),
                    load_i16_at::<V>(rows[3].add(j)),
                    load_i16_at::<V>(rows[4].add(j)),
                    load_i16_at::<V>(rows[5].add(j)),
                ];
                wiener_v5_vec::<V, ROUND1>(r, taps, zero, cap)
            } else {
                let r = [
                    load_i16_at::<V>(rows[0].add(j)),
                    load_i16_at::<V>(rows[1].add(j)),
                    load_i16_at::<V>(rows[2].add(j)),
                    load_i16_at::<V>(rows[3].add(j)),
                    load_i16_at::<V>(rows[4].add(j)),
                    load_i16_at::<V>(rows[5].add(j)),
                    load_i16_at::<V>(rows[6].add(j)),
                ];
                wiener_v7_vec::<V, ROUND1>(r, taps, zero, cap)
            };
            v.store_u16_at(dst.add(j));
        }
        let j = full * V::LANES;
        if j < n {
            let v = if FIVE {
                let r = [
                    load_i16_at::<V>(rows[1].add(j)),
                    load_i16_at::<V>(rows[2].add(j)),
                    load_i16_at::<V>(rows[3].add(j)),
                    load_i16_at::<V>(rows[4].add(j)),
                    load_i16_at::<V>(rows[5].add(j)),
                ];
                wiener_v5_vec::<V, ROUND1>(r, taps, zero, cap)
            } else {
                let r = [
                    load_i16_at::<V>(rows[0].add(j)),
                    load_i16_at::<V>(rows[1].add(j)),
                    load_i16_at::<V>(rows[2].add(j)),
                    load_i16_at::<V>(rows[3].add(j)),
                    load_i16_at::<V>(rows[4].add(j)),
                    load_i16_at::<V>(rows[5].add(j)),
                    load_i16_at::<V>(rows[6].add(j)),
                ];
                wiener_v7_vec::<V, ROUND1>(r, taps, zero, cap)
            };
            store_tail_u16(v, dst.add(j), n - j);
        }
    }
}

#[inline(always)]
unsafe fn wiener_h_lanes<V: I16Lanes>(
    src: &[u16],
    taps: WienerTaps,
    out: &mut [i16],
    n: usize,
    depth: BitDepth,
) {
    let blocks = n.div_ceil(V::LANES);
    assert!(src.len() >= blocks * V::LANES + 6 && out.len() >= blocks * V::LANES);
    // SAFETY: caller enabled ISA; bounds checked above.
    unsafe {
        match depth.bits() {
            10 => wiener_h_ptr::<V, false, 3, 8>(src.as_ptr(), taps, out.as_mut_ptr(), blocks),
            12 => wiener_h_ptr::<V, false, 5, 2>(src.as_ptr(), taps, out.as_mut_ptr(), blocks),
            _ => h_row_at(src, taps, out, n, depth),
        }
    }
}

#[inline(always)]
unsafe fn wiener_v_lanes<V: I16Lanes>(
    rows: &[&[i16]; WIENER_ROWS],
    taps: WienerTaps,
    dst: &mut [u16],
    depth: BitDepth,
) {
    let n = dst.len();
    let blocks = n.div_ceil(V::LANES);
    assert!(rows.iter().all(|row| row.len() >= blocks * V::LANES));
    let ptrs = rows.map(<[i16]>::as_ptr);
    let max_val = depth.max() as i16;
    // SAFETY: caller enabled ISA; bounds checked above.
    unsafe {
        match depth.bits() {
            10 => wiener_v_ptr::<V, false, 11>(&ptrs, taps, dst.as_mut_ptr(), n, max_val),
            12 => wiener_v_ptr::<V, false, 9>(&ptrs, taps, dst.as_mut_ptr(), n, max_val),
            _ => v_row_at(rows, taps, dst, depth),
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn wiener_strip_col<
    V: I16Lanes,
    const H5: bool,
    const V5: bool,
    const ROUND0: u32,
    const ROUND1: u32,
    const BIAS: i16,
    const TAIL: bool,
>(
    horizontal: WienerTaps,
    vertical: WienerTaps,
    win_col: *const u16,
    stride: usize,
    out_col: *mut u16,
    dst_stride: usize,
    h: usize,
    rem: usize,
    (zero, cap): (V, V),
) {
    // SAFETY: `wiener_stripe_lanes` validated `(h + 6)` window rows with
    // `V::LANES + 6` readable `u16`s per strip and `h` output rows.
    unsafe {
        if V5 {
            let mut r1 = wiener_h_vec::<V, H5, ROUND0, BIAS>(win_col.add(stride), horizontal);
            let mut r2 = wiener_h_vec::<V, H5, ROUND0, BIAS>(win_col.add(2 * stride), horizontal);
            let mut r3 = wiener_h_vec::<V, H5, ROUND0, BIAS>(win_col.add(3 * stride), horizontal);
            let mut r4 = wiener_h_vec::<V, H5, ROUND0, BIAS>(win_col.add(4 * stride), horizontal);
            for r in 0..h {
                let r5 =
                    wiener_h_vec::<V, H5, ROUND0, BIAS>(win_col.add((r + 5) * stride), horizontal);
                let v = wiener_v5_vec::<V, ROUND1>([r1, r2, r3, r4, r5], vertical, zero, cap);
                if TAIL {
                    store_tail_u16(v, out_col.add(r * dst_stride), rem);
                } else {
                    v.store_u16_at(out_col.add(r * dst_stride));
                }
                (r1, r2, r3, r4) = (r2, r3, r4, r5);
            }
        } else {
            let mut r0 = wiener_h_vec::<V, H5, ROUND0, BIAS>(win_col, horizontal);
            let mut r1 = wiener_h_vec::<V, H5, ROUND0, BIAS>(win_col.add(stride), horizontal);
            let mut r2 = wiener_h_vec::<V, H5, ROUND0, BIAS>(win_col.add(2 * stride), horizontal);
            let mut r3 = wiener_h_vec::<V, H5, ROUND0, BIAS>(win_col.add(3 * stride), horizontal);
            let mut r4 = wiener_h_vec::<V, H5, ROUND0, BIAS>(win_col.add(4 * stride), horizontal);
            let mut r5 = wiener_h_vec::<V, H5, ROUND0, BIAS>(win_col.add(5 * stride), horizontal);
            for r in 0..h {
                let r6 =
                    wiener_h_vec::<V, H5, ROUND0, BIAS>(win_col.add((r + 6) * stride), horizontal);
                let v =
                    wiener_v7_vec::<V, ROUND1>([r0, r1, r2, r3, r4, r5, r6], vertical, zero, cap);
                if TAIL {
                    store_tail_u16(v, out_col.add(r * dst_stride), rem);
                } else {
                    v.store_u16_at(out_col.add(r * dst_stride));
                }
                (r0, r1, r2, r3, r4, r5) = (r1, r2, r3, r4, r5, r6);
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn wiener_stripe_depth<
    V: I16Lanes,
    const H5: bool,
    const V5: bool,
    const ROUND0: u32,
    const ROUND1: u32,
    const BIAS: i16,
>(
    horizontal: WienerTaps,
    vertical: WienerTaps,
    (window, stride): (&[u16], usize),
    (w, h): (usize, usize),
    _ring: &mut [i16],
    (dst, x, y): (&mut PlaneViewMut<'_, u16>, usize, usize),
    max_val: i16,
) {
    let full = w / V::LANES;
    let dst_stride = dst.stride();
    let bounds = (V::splat(0), V::splat(max_val));
    // SAFETY: `wiener_stripe_lanes` validated all window and destination extents.
    unsafe {
        let win_base = window.as_ptr();
        let out_base = dst.data_mut().as_mut_ptr().add(y * dst_stride + x);
        for block in 0..full {
            let j = block * V::LANES;
            wiener_strip_col::<V, H5, V5, ROUND0, ROUND1, BIAS, false>(
                horizontal,
                vertical,
                win_base.add(j),
                stride,
                out_base.add(j),
                dst_stride,
                h,
                V::LANES,
                bounds,
            );
        }
        let j = full * V::LANES;
        if j < w {
            wiener_strip_col::<V, H5, V5, ROUND0, ROUND1, BIAS, true>(
                horizontal,
                vertical,
                win_base.add(j),
                stride,
                out_base.add(j),
                dst_stride,
                h,
                w - j,
                bounds,
            );
        }
    }
}

#[inline(always)]
unsafe fn wiener_stripe_axes<V: I16Lanes, const H5: bool, const V5: bool>(
    horizontal: WienerTaps,
    vertical: WienerTaps,
    window: (&[u16], usize),
    size: (usize, usize),
    ring: &mut [i16],
    dst: (&mut PlaneViewMut<'_, u16>, usize, usize),
    depth: BitDepth,
) {
    let max_val = depth.max() as i16;
    // SAFETY: caller enabled ISA and `wiener_stripe_lanes` validated extents.
    unsafe {
        match depth.bits() {
            10 => wiener_stripe_depth::<V, H5, V5, 3, 11, 8>(
                horizontal, vertical, window, size, ring, dst, max_val,
            ),
            12 => wiener_stripe_depth::<V, H5, V5, 5, 9, 2>(
                horizontal, vertical, window, size, ring, dst, max_val,
            ),
            _ => unreachable!("validated 10- or 12-bit Wiener stripe"),
        }
    }
}

#[inline(always)]
unsafe fn wiener_stripe_lanes<V: I16Lanes>(
    horizontal: WienerTaps,
    vertical: WienerTaps,
    (window, stride): (&[u16], usize),
    (w, h): (usize, usize),
    ring: &mut [i16],
    (dst, x, y): (&mut PlaneViewMut<'_, u16>, usize, usize),
    depth: BitDepth,
) {
    assert!(x <= dst.width() && w <= dst.width() - x);
    assert!(y <= dst.height() && h <= dst.height() - y);
    if w == 0 || h == 0 {
        return;
    }
    let row_len = w.checked_add(AV1_LR_OVERREAD).expect("Wiener row extent");
    let source_len = row_len
        .checked_add(2 * AV1_LR_BORDER)
        .expect("Wiener source row");
    let source_rows = h
        .checked_add(2 * AV1_LR_BORDER)
        .expect("Wiener source rows");
    let source_end = (source_rows - 1)
        .checked_mul(stride)
        .and_then(|off| off.checked_add(source_len))
        .expect("Wiener source extent");
    let blocks = w.div_ceil(V::LANES);
    assert!(blocks * V::LANES <= row_len && blocks * V::LANES + 6 <= source_len);
    assert!(window.len() >= source_end);
    assert!(
        ring.len()
            >= WIENER_ROWS
                .checked_mul(row_len)
                .expect("Wiener ring extent")
    );
    if depth.bits() != 10 && depth.bits() != 12 {
        for t in 0..h + 2 * AV1_LR_BORDER {
            let slot = t % WIENER_ROWS;
            h_row_at(
                &window[t * stride..t * stride + source_len],
                horizontal,
                &mut ring[slot * row_len..(slot + 1) * row_len],
                w,
                depth,
            );
            if let Some(r) = t.checked_sub(WIENER_ROWS - 1) {
                let rows: [&[i16]; WIENER_ROWS] = core::array::from_fn(|k| {
                    let s = (r + k) % WIENER_ROWS;
                    &ring[s * row_len..(s + 1) * row_len]
                });
                v_row_at(&rows, vertical, &mut dst.row_mut(y + r)[x..x + w], depth);
            }
        }
        return;
    }
    // SAFETY: caller enabled ISA and all spans are checked above.
    unsafe {
        match (horizontal[0] == 0, vertical[0] == 0) {
            (false, false) => wiener_stripe_axes::<V, false, false>(
                horizontal,
                vertical,
                (window, stride),
                (w, h),
                ring,
                (dst, x, y),
                depth,
            ),
            (true, false) => wiener_stripe_axes::<V, true, false>(
                horizontal,
                vertical,
                (window, stride),
                (w, h),
                ring,
                (dst, x, y),
                depth,
            ),
            (false, true) => wiener_stripe_axes::<V, false, true>(
                horizontal,
                vertical,
                (window, stride),
                (w, h),
                ring,
                (dst, x, y),
                depth,
            ),
            (true, true) => wiener_stripe_axes::<V, true, true>(
                horizontal,
                vertical,
                (window, stride),
                (w, h),
                ring,
                (dst, x, y),
                depth,
            ),
        }
    }
}

#[inline(always)]
unsafe fn box_ab_u16<V: SgrLanes, const N: usize>(
    sums: [&[i16]; N],
    squares: [&[i32]; N],
    scale: u32,
    ab: (&mut [i16], &mut [i32]),
    n: usize,
    depth: BitDepth,
) {
    // SAFETY: caller enabled ISA and checked all extents.
    unsafe {
        match depth.above_eight() {
            2 => sgr_lanes::box_ab_at::<V, N, 2>(sums, squares, scale, ab, n),
            4 => sgr_lanes::box_ab_at::<V, N, 4>(sums, squares, scale, ab, n),
            bits => scalar_box_ab_at(sums, squares, scale, ab, (n, bits)),
        }
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! high_lr_isa {
    ($mod_name:ident, $(#[$feature:meta])*, $w_lanes:ty, $sgr_lanes:ty) => {
        mod $mod_name {
            use nc_common::{BitDepth, PlaneViewMut};

            use super::super::kernels::WIENER_ROWS;
            use super::super::sgr_lanes;
            use super::super::wiener::WienerTaps;
            use super::{
                HighLrFns, box_ab_u16, wiener_h_lanes, wiener_stripe_lanes, wiener_v_lanes,
            };

            $(#[$feature])*
            unsafe fn wiener_stripe(
                horizontal: WienerTaps,
                vertical: WienerTaps,
                window: (&[u16], usize),
                size: (usize, usize),
                ring: &mut [i16],
                dst: (&mut PlaneViewMut<'_, u16>, usize, usize),
                depth: BitDepth,
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe {
                    wiener_stripe_lanes::<$w_lanes>(
                        horizontal, vertical, window, size, ring, dst, depth,
                    );
                }
            }

            $(#[$feature])*
            unsafe fn wiener_h(
                src: &[u16],
                taps: WienerTaps,
                out: &mut [i16],
                n: usize,
                depth: BitDepth,
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { wiener_h_lanes::<$w_lanes>(src, taps, out, n, depth) }
            }

            $(#[$feature])*
            unsafe fn wiener_v(
                rows: &[&[i16]; WIENER_ROWS],
                taps: WienerTaps,
                dst: &mut [u16],
                depth: BitDepth,
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { wiener_v_lanes::<$w_lanes>(rows, taps, dst, depth) }
            }

            $(#[$feature])*
            unsafe fn box_sums3(src: &[u16], sum: &mut [i16], squares: &mut [i32], n: usize) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { sgr_lanes::box_sums::<$sgr_lanes, u16, 1>(src, sum, squares, n) }
            }

            $(#[$feature])*
            unsafe fn box_sums5(src: &[u16], sum: &mut [i16], squares: &mut [i32], n: usize) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { sgr_lanes::box_sums::<$sgr_lanes, u16, 2>(src, sum, squares, n) }
            }

            $(#[$feature])*
            unsafe fn box_ab3(
                sums: [&[i16]; 3],
                squares: [&[i32]; 3],
                scale: u32,
                ab: (&mut [i16], &mut [i32]),
                n: usize,
                depth: BitDepth,
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { box_ab_u16::<$sgr_lanes, 3>(sums, squares, scale, ab, n, depth) }
            }

            $(#[$feature])*
            unsafe fn box_ab5(
                sums: [&[i16]; 5],
                squares: [&[i32]; 5],
                scale: u32,
                ab: (&mut [i16], &mut [i32]),
                n: usize,
                depth: BitDepth,
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { box_ab_u16::<$sgr_lanes, 5>(sums, squares, scale, ab, n, depth) }
            }

            $(#[$feature])*
            unsafe fn between(
                a: [&[i16]; 2],
                b: [&[i32]; 2],
                src: &[u16],
                out: &mut [i32],
                n: usize,
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { sgr_lanes::filter_between::<$sgr_lanes, u16>(a, b, src, out, n) }
            }

            $(#[$feature])*
            unsafe fn on(a: &[i16], b: &[i32], src: &[u16], out: &mut [i32], n: usize) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { sgr_lanes::filter_on::<$sgr_lanes, u16>(a, b, src, out, n) }
            }

            $(#[$feature])*
            unsafe fn around(
                a: [&[i16]; 3],
                b: [&[i32]; 3],
                src: &[u16],
                out: &mut [i32],
                n: usize,
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { sgr_lanes::filter_around::<$sgr_lanes, u16>(a, b, src, out, n) }
            }

            $(#[$feature])*
            unsafe fn blend(
                src: &[u16],
                filtered: [&[i32]; 2],
                weights: [i32; 2],
                dst: &mut [u16],
                depth: BitDepth,
            ) {
                // SAFETY: target_feature enabled by wrapper.
                unsafe { sgr_lanes::blend::<$sgr_lanes, u16>(src, filtered, weights, dst, depth) }
            }

            pub(super) const FNS: HighLrFns = HighLrFns {
                wiener_stripe: Some(wiener_stripe),
                wiener_h,
                wiener_v,
                box_sums3,
                box_sums5,
                box_ab3,
                box_ab5,
                between,
                on,
                around,
                blend,
            };
        }
    };
}

#[cfg(target_arch = "x86_64")]
high_lr_isa!(
    x86_avx2,
    #[target_feature(enable = "avx2")],
    crate::simd::avx2::I16x16,
    super::super::avx2::Lanes
);

#[cfg(target_arch = "x86_64")]
high_lr_isa!(
    x86_avx512,
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")],
    crate::simd::avx512_i16::I16x32,
    super::avx512_sgr::Lanes16
);

#[cfg(target_arch = "aarch64")]
high_lr_isa!(
    neon_high,
    #[target_feature(enable = "neon")],
    crate::simd::neon::I16x8,
    super::super::neon::Lanes
);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
high_lr_isa!(
    simd128_high,
    #[target_feature(enable = "simd128")],
    crate::simd::simd128::I16x8,
    super::super::simd128::Lanes
);

// ----------------------------------------- 16-lane AVX-512 SGR ----

#[cfg(target_arch = "x86_64")]
mod avx512_sgr {
    use core::arch::x86_64::{
        __m512i, _mm_cvtsi32_si128, _mm_loadu_si128, _mm_storeu_si128, _mm256_loadu_si256,
        _mm256_storeu_si256, _mm512_add_epi32, _mm512_add_ps, _mm512_cmpeq_epi32_mask,
        _mm512_cmpneq_epi32_mask, _mm512_cvtepi16_epi32, _mm512_cvtepi32_ps, _mm512_cvtepu8_epi32,
        _mm512_cvtepu16_epi32, _mm512_cvtsepi32_epi16, _mm512_cvttps_epi32, _mm512_cvtusepi32_epi8,
        _mm512_cvtusepi32_epi16, _mm512_div_ps, _mm512_loadu_si512, _mm512_maskz_min_epi32,
        _mm512_maskz_set1_epi32, _mm512_max_epi32, _mm512_min_epi32, _mm512_mul_ps,
        _mm512_mullo_epi32, _mm512_rcp14_ps, _mm512_set1_epi32, _mm512_set1_ps,
        _mm512_setzero_si512, _mm512_sll_epi32, _mm512_sra_epi32, _mm512_srl_epi32,
        _mm512_storeu_si512, _mm512_sub_epi32,
    };

    use super::super::kernels::Kernels;
    use super::super::self_guided::{SGR_BITS, Z_MAX};
    use super::super::sgr_lanes::{SgrLanes, sgr_kernels};

    /// Sixteen `i32` lanes for AVX-512 self-guided kernels (`u8` and `u16`).
    #[derive(Clone, Copy)]
    pub(super) struct Lanes16(__m512i);

    // SAFETY (every method): `Lanes16` exists only inside `avx512f,avx512bw,avx512vl`
    // kernels; slices are indexed before raw loads and stores.
    impl SgrLanes for Lanes16 {
        const LANES: usize = 16;

        #[inline(always)]
        fn splat(value: i32) -> Self {
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_set1_epi32(value) })
        }
        #[inline(always)]
        fn load(src: &[i32]) -> Self {
            let src = &src[..16];
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_loadu_si512(src.as_ptr().cast()) })
        }
        #[inline(always)]
        fn load_i16(src: &[i16]) -> Self {
            let src = &src[..16];
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_cvtepi16_epi32(_mm256_loadu_si256(src.as_ptr().cast())) })
        }
        #[inline(always)]
        fn load_u8(src: &[u8]) -> Self {
            let src = &src[..16];
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_cvtepu8_epi32(_mm_loadu_si128(src.as_ptr().cast())) })
        }
        #[inline(always)]
        fn load_u16(src: &[u16]) -> Self {
            let src = &src[..16];
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_cvtepu16_epi32(_mm256_loadu_si256(src.as_ptr().cast())) })
        }
        #[inline(always)]
        fn store(self, dst: &mut [i32]) {
            let dst = &mut dst[..16];
            // SAFETY: see impl comment.
            unsafe { _mm512_storeu_si512(dst.as_mut_ptr().cast(), self.0) }
        }
        #[inline(always)]
        fn store_i16(self, dst: &mut [i16]) {
            let dst = &mut dst[..16];
            // SAFETY: see impl comment.
            unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), _mm512_cvtsepi32_epi16(self.0)) }
        }
        #[inline(always)]
        fn store_u8_sat(self, dst: &mut [u8]) {
            let dst = &mut dst[..16];
            // SAFETY: see impl comment.
            unsafe {
                let pos = _mm512_max_epi32(self.0, _mm512_setzero_si512());
                _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm512_cvtusepi32_epi8(pos));
            }
        }
        #[inline(always)]
        fn store_u16_clip(self, max_val: i32, dst: &mut [u16]) {
            let dst = &mut dst[..16];
            // SAFETY: see impl comment.
            unsafe {
                let clamped = _mm512_min_epi32(
                    _mm512_max_epi32(self.0, _mm512_setzero_si512()),
                    _mm512_set1_epi32(max_val),
                );
                _mm256_storeu_si256(dst.as_mut_ptr().cast(), _mm512_cvtusepi32_epi16(clamped));
            }
        }
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_add_epi32(self.0, other.0) })
        }
        #[inline(always)]
        fn sub(self, other: Self) -> Self {
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_sub_epi32(self.0, other.0) })
        }
        #[inline(always)]
        fn mul(self, other: Self) -> Self {
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_mullo_epi32(self.0, other.0) })
        }
        #[inline(always)]
        fn shl<const BITS: i32>(self) -> Self {
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_sll_epi32(self.0, _mm_cvtsi32_si128(BITS)) })
        }
        #[inline(always)]
        fn shr<const BITS: i32>(self) -> Self {
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_srl_epi32(self.0, _mm_cvtsi32_si128(BITS)) })
        }
        #[inline(always)]
        fn sra<const BITS: i32>(self) -> Self {
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_sra_epi32(self.0, _mm_cvtsi32_si128(BITS)) })
        }
        #[inline(always)]
        fn min(self, other: Self) -> Self {
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_min_epi32(self.0, other.0) })
        }
        #[inline(always)]
        fn max(self, other: Self) -> Self {
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_max_epi32(self.0, other.0) })
        }
        #[inline(always)]
        fn eq(self, other: Self) -> Self {
            // SAFETY: see impl comment.
            Self(unsafe { _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(self.0, other.0), -1) })
        }
        #[inline(always)]
        fn sgr_factor(self) -> Self {
            // SAFETY: see `avx2::Lanes::sgr_factor` error bound; `_mm512_rcp14_ps`
            // has `< 2^-14` relative error, 4x tighter than `_mm256_rcp_ps`.
            unsafe {
                let unit = _mm512_set1_epi32(1 << SGR_BITS);
                let den = _mm512_cvtepi32_ps(_mm512_add_epi32(self.0, _mm512_set1_epi32(1)));
                let rounded = _mm512_cvttps_epi32(_mm512_add_ps(
                    _mm512_mul_ps(
                        _mm512_rcp14_ps(den),
                        _mm512_set1_ps(f32::from(1_u16 << SGR_BITS)),
                    ),
                    _mm512_set1_ps(0.5),
                ));
                let max = _mm512_set1_epi32(Z_MAX);
                let not_max = _mm512_cmpneq_epi32_mask(self.0, max);
                let complement = _mm512_maskz_min_epi32(not_max, rounded, max);
                Self(_mm512_sub_epi32(unit, complement))
            }
        }
        #[inline(always)]
        fn div_floor(self, den: Self) -> Self {
            // SAFETY: see impl comment.
            Self(unsafe {
                _mm512_cvttps_epi32(_mm512_div_ps(
                    _mm512_cvtepi32_ps(self.0),
                    _mm512_cvtepi32_ps(den.0),
                ))
            })
        }
    }

    sgr_kernels!(#[target_feature(enable = "avx512f,avx512bw,avx512vl")]; Lanes16);

    pub(crate) const fn install_u8(base: Kernels) -> Kernels {
        Kernels {
            box_sums3,
            box_sums5,
            box_ab3,
            box_ab5,
            between,
            on,
            around,
            blend,
            sgr_stripe: Some(super::super::sgr_avx2::sgr_stripe_icl),
            ..base
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{Lanes16, SgrLanes, Z_MAX};
        use crate::Isa;
        use crate::av1_restoration::self_guided::x_by_xplus1;

        #[test]
        fn avx512_reciprocal_factor_matches_entire_legal_domain() {
            if !Isa::Avx512.is_supported() {
                return;
            }
            // SAFETY: AVX-512F/BW/VL detected above.
            unsafe { check_reciprocal_factors() };
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
        unsafe fn check_reciprocal_factors() {
            for first in (0..=Z_MAX).step_by(16) {
                let z: [i32; 16] = core::array::from_fn(|i| first + i as i32);
                let mut got = [0; 16];
                Lanes16::load(&z).sgr_factor().store(&mut got);
                for (&z, &factor) in z.iter().zip(&got) {
                    assert_eq!(factor as u32, x_by_xplus1(z as u32), "z={z}");
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) const fn avx512_u8_kernels(base: super::kernels::Kernels) -> super::kernels::Kernels {
    avx512_sgr::install_u8(base)
}
