//! High-bit-depth (`u16`, 10- and 12-bit) VP9 8-tap / bilinear motion
//! compensation across AVX2, AVX-512, NEON, and SIMD128.
//!
//! Shares [`tap_pairs`], [`h_dot_lanes`] and [`v_dot_lanes`] (`I16Lanes::acc_mul_add2`:
//! `_mm256_madd_epi16` / `_mm512_madd_epi16` / `i32x4_dot_i16x8` / NEON
//! widening MACs on tap pairs) with [`crate::av1_mc::high_mc`]. VP9 differs
//! from AV1 in rounding and clipping its horizontal pass to sample depth
//! (`FILTER_BITS = 7`) before the vertical pass, and in upward-rounded sample
//! averaging (`(dst + pred + 1) >> 1`) for compound blocks.
//!
//! Attribution: zero-phase copy/average, 1-D horizontal/vertical shortcuts,
//! and paired-tap `madd` follow libvpx (`vpx_dsp/x86/highbd_convolve_avx2.c`,
//! BSD-3-Clause; see `THIRD_PARTY.md`).

#![allow(clippy::inline_always)]
#![cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code, unused_imports)
)]

use core::mem::MaybeUninit;

use nc_common::BitDepth;

use crate::Isa;
use crate::av1_mc::high_mc::{
    h_dot_lanes, load_i16, store_u16, tap_pair_count, tap_pairs, v_dot_lanes,
};
use crate::high_depth::{Vp9Mc16Fn, av1_widths};
use crate::simd::I16Lanes;
use crate::vp9_mc::{FILTER_BITS, VP9_TAPS, VP9_TAPS_BEFORE, rows_at};

const IDENTITY_TAPS: [i16; VP9_TAPS] = [0, 0, 0, 128, 0, 0, 0, 0];
/// Index of the tap that multiplies the predicted sample itself.
const FILTER_CENTER: usize = VP9_TAPS_BEFORE + 1;
const MAX_BLOCK_HEIGHT: usize = 64;
const MAX_INTER_ROWS: usize = MAX_BLOCK_HEIGHT + VP9_TAPS - 1;
const ROUND: i32 = 1 << (FILTER_BITS - 1);
const MAX_SIMD_BITS: u32 = 12;

fn scalar_mc<const W: usize>(
    window: (&[u16], usize),
    dst: (&mut [u16], usize),
    h: usize,
    taps: (&[i16; VP9_TAPS], &[i16; VP9_TAPS]),
    average: bool,
    depth: BitDepth,
) {
    rows_at::<u16, W>(window, dst, h, taps, average, depth);
}

pub(crate) fn kernels(isa: Isa) -> [Vp9Mc16Fn; 5] {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => x86::AVX2,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => x86::AVX512,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::NEON,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::SIMD128,
        _ => av1_widths!(scalar_mc),
    }
}

/// Runs `$body` with `$p` bound to the const tap-pair count `$count` (1..=4).
macro_rules! by_pairs {
    ($count:expr, $p:ident => $body:expr) => {
        match $count {
            1 => {
                const $p: usize = 1;
                $body
            }
            2 => {
                const $p: usize = 2;
                $body
            }
            3 => {
                const $p: usize = 3;
                $body
            }
            _ => {
                const $p: usize = 4;
                $body
            }
        }
    };
}

#[inline(always)]
unsafe fn write_or_avg<V: I16Lanes, const AVG: bool>(pred: V, dst: *mut u16) {
    if AVG {
        // SAFETY: caller validated `V::LANES` readable and writable `u16`s (`0..=4095`).
        unsafe {
            let cur = load_i16::<V>(dst);
            store_u16(cur.add(pred).add(V::splat(1)).sra(1), dst);
        }
    } else {
        // SAFETY: caller validated `V::LANES` writable `u16`s.
        unsafe { store_u16(pred, dst) };
    }
}

#[inline(always)]
unsafe fn convolve_lanes<V: I16Lanes, const W: usize, const AVG: bool>(
    (src, ss): (&[u16], usize),
    (dst, ds): (&mut [u16], usize),
    h: usize,
    (fx, fy): (&[i16; VP9_TAPS], &[i16; VP9_TAPS]),
    depth: BitDepth,
) {
    assert!(
        h > 0
            && h <= MAX_BLOCK_HEIGHT
            && src.len() >= (h + VP9_TAPS - 2) * ss + W + VP9_TAPS - 1
            && dst.len() >= (h - 1) * ds + W,
        "VP9 u16 MC window or destination too small"
    );
    debug_assert!(W.is_multiple_of(V::LANES));
    let (src_p, dst_p) = (src.as_ptr(), dst.as_mut_ptr());
    let (hx_id, vy_id) = (*fx == IDENTITY_TAPS, *fy == IDENTITY_TAPS);
    if hx_id && vy_id {
        for r in 0..h {
            // SAFETY: validated window and destination cover `W` words per row.
            unsafe {
                let row = src_p.add((r + VP9_TAPS_BEFORE) * ss + VP9_TAPS_BEFORE);
                let out = dst_p.add(r * ds);
                if AVG {
                    for c0 in (0..W).step_by(V::LANES) {
                        write_or_avg::<V, true>(load_i16::<V>(row.add(c0)), out.add(c0));
                    }
                } else {
                    core::ptr::copy_nonoverlapping(row, out, W);
                }
            }
        }
        return;
    }
    let clamp = (V::splat(0), V::splat(depth.max() as i16));
    let fy_pairs = tap_pair_count(fy);
    // SAFETY: every arm forwards the window and destination validated above.
    unsafe {
        if vy_id {
            let dst = (dst_p, ds);
            by_pairs!(tap_pair_count(fx), P => {
                h_only::<V, W, AVG, P>((src_p, ss), dst, h, fx, clamp);
            });
        } else if hx_id {
            let column = (src_p.add(VP9_TAPS_BEFORE), ss);
            by_pairs!(fy_pairs, P => {
                v_only::<V, W, AVG, P>(column, (dst_p, ds), h, fy, clamp);
            });
        } else {
            let mut inter = [MaybeUninit::<[u16; W]>::uninit(); MAX_INTER_ROWS];
            let first_row = FILTER_CENTER - fy_pairs;
            let rows = first_row..h + first_row + 2 * fy_pairs - 1;
            by_pairs!(tap_pair_count(fx), P => {
                h_rows::<V, W, P>((src_p, ss), &mut inter, rows, fx, clamp);
            });
            let inter_rows = (inter.as_ptr().cast::<u16>(), W);
            by_pairs!(fy_pairs, P => {
                v_only::<V, W, AVG, P>(inter_rows, (dst_p, ds), h, fy, clamp);
            });
        }
    }
}

/// Rounds a tap sum by `FILTER_BITS` and clips it to the sample range: VP9
/// clips after each pass (AV1 keeps the horizontal pass at higher precision).
#[inline(always)]
fn round_clip<V: I16Lanes>(acc: V::Acc, (zero, max_v): (V, V)) -> V {
    V::acc_round_shift_narrow(acc, ROUND, FILTER_BITS)
        .max(zero)
        .min(max_v)
}

/// Horizontal-only prediction (`fy` is the identity).
#[inline(always)]
unsafe fn h_only<V: I16Lanes, const W: usize, const AVG: bool, const PAIRS: usize>(
    (src_p, ss): (*const u16, usize),
    (dst_p, ds): (*mut u16, usize),
    h: usize,
    fx: &[i16; VP9_TAPS],
    clamp: (V, V),
) {
    let fx = tap_pairs::<PAIRS>(fx);
    for r in 0..h {
        // SAFETY: row `r + 3` lies inside the validated `h + 7` window rows.
        let row = unsafe { src_p.add((r + VP9_TAPS_BEFORE) * ss) };
        for c0 in (0..W).step_by(V::LANES) {
            // SAFETY: `c0 + V::LANES + 7` lies inside `W + 7`, and `dst` covers `W`.
            unsafe {
                let pred = round_clip(h_dot_lanes::<V, PAIRS>(row.add(c0), fx), clamp);
                write_or_avg::<V, AVG>(pred, dst_p.add(r * ds + c0));
            }
        }
    }
}

/// Vertical prediction from `src_p`, which points at the block's first column
/// in the window's first tap row: the source window when `fx` is the identity,
/// else the clipped horizontal pass.
#[inline(always)]
unsafe fn v_only<V: I16Lanes, const W: usize, const AVG: bool, const PAIRS: usize>(
    (src_p, ss): (*const u16, usize),
    (dst_p, ds): (*mut u16, usize),
    h: usize,
    fy: &[i16; VP9_TAPS],
    clamp: (V, V),
) {
    let fy = tap_pairs::<PAIRS>(fy);
    for r in 0..h {
        for c0 in (0..W).step_by(V::LANES) {
            // SAFETY: rows `r..r + 8` and `c0..c0 + V::LANES` lie inside the window.
            unsafe {
                let pred = round_clip(
                    v_dot_lanes::<V, PAIRS>(src_p.add(r * ss + c0), ss, fy),
                    clamp,
                );
                write_or_avg::<V, AVG>(pred, dst_p.add(r * ds + c0));
            }
        }
    }
}

/// The clipped horizontal pass over window rows `rows` into `inter`.
#[inline(always)]
unsafe fn h_rows<V: I16Lanes, const W: usize, const PAIRS: usize>(
    (src_p, ss): (*const u16, usize),
    inter: &mut [MaybeUninit<[u16; W]>; MAX_INTER_ROWS],
    rows: core::ops::Range<usize>,
    fx: &[i16; VP9_TAPS],
    clamp: (V, V),
) {
    let fx = tap_pairs::<PAIRS>(fx);
    for r in rows {
        // SAFETY: `r < h + 7`, inside the validated window.
        let row = unsafe { src_p.add(r * ss) };
        let line = inter[r].as_mut_ptr().cast::<u16>();
        for c0 in (0..W).step_by(V::LANES) {
            // SAFETY: writes `V::LANES` of the `W` words of `inter[r]`.
            unsafe {
                let pred = round_clip(h_dot_lanes::<V, PAIRS>(row.add(c0), fx), clamp);
                store_u16(pred, line.add(c0));
            }
        }
    }
}

#[inline(always)]
unsafe fn dispatch_avg<V: I16Lanes, const W: usize>(
    src: (&[u16], usize),
    dst: (&mut [u16], usize),
    h: usize,
    taps: (&[i16; VP9_TAPS], &[i16; VP9_TAPS]),
    average: bool,
    depth: BitDepth,
) {
    if depth.bits() > MAX_SIMD_BITS {
        rows_at::<u16, W>(src, dst, h, taps, average, depth);
        return;
    }
    // SAFETY: forwarded ISA and slice contracts.
    unsafe {
        if average {
            convolve_lanes::<V, W, true>(src, dst, h, taps, depth);
        } else {
            convolve_lanes::<V, W, false>(src, dst, h, taps, depth);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use nc_common::BitDepth;

    use super::{VP9_TAPS, Vp9Mc16Fn, dispatch_avg};
    use crate::simd::avx2::{I16x4, I16x8, I16x16};
    use crate::simd::avx512_i16::I16x32;

    macro_rules! x86_width {
        ($name:ident, $feat:literal, $v:ty, $w:literal) => {
            #[target_feature(enable = $feat)]
            unsafe fn $name(
                src: (&[u16], usize),
                dst: (&mut [u16], usize),
                h: usize,
                taps: (&[i16; VP9_TAPS], &[i16; VP9_TAPS]),
                average: bool,
                depth: BitDepth,
            ) {
                // SAFETY: target feature enabled by this wrapper.
                unsafe { dispatch_avg::<$v, $w>(src, dst, h, taps, average, depth) }
            }
        };
    }

    x86_width!(mc_w4, "avx2", I16x4, 4);
    x86_width!(mc_w8, "avx2", I16x8, 8);
    x86_width!(mc_w16, "avx2", I16x16, 16);
    x86_width!(mc_w32, "avx2", I16x16, 32);
    x86_width!(mc_w64, "avx2", I16x16, 64);
    x86_width!(mc_w32_512, "avx512f,avx512bw,avx512vl", I16x32, 32);
    x86_width!(mc_w64_512, "avx512f,avx512bw,avx512vl", I16x32, 64);

    pub(super) const AVX2: [Vp9Mc16Fn; 5] = [mc_w4, mc_w8, mc_w16, mc_w32, mc_w64];
    pub(super) const AVX512: [Vp9Mc16Fn; 5] = [mc_w4, mc_w8, mc_w16, mc_w32_512, mc_w64_512];
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use nc_common::BitDepth;

    use super::{VP9_TAPS, Vp9Mc16Fn, dispatch_avg};
    use crate::simd::neon::{I16x4, I16x8};

    macro_rules! neon_width {
        ($name:ident, $v:ty, $w:literal) => {
            #[target_feature(enable = "neon")]
            unsafe fn $name(
                src: (&[u16], usize),
                dst: (&mut [u16], usize),
                h: usize,
                taps: (&[i16; VP9_TAPS], &[i16; VP9_TAPS]),
                average: bool,
                depth: BitDepth,
            ) {
                // SAFETY: NEON is enabled by this wrapper.
                unsafe { dispatch_avg::<$v, $w>(src, dst, h, taps, average, depth) }
            }
        };
    }

    neon_width!(mc_w4, I16x4, 4);
    neon_width!(mc_w8, I16x8, 8);
    neon_width!(mc_w16, I16x8, 16);
    neon_width!(mc_w32, I16x8, 32);
    neon_width!(mc_w64, I16x8, 64);

    pub(super) const NEON: [Vp9Mc16Fn; 5] = [mc_w4, mc_w8, mc_w16, mc_w32, mc_w64];
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128 {
    use nc_common::BitDepth;

    use super::{VP9_TAPS, Vp9Mc16Fn, dispatch_avg};
    use crate::simd::simd128::{I16x4, I16x8};

    macro_rules! simd128_width {
        ($name:ident, $v:ty, $w:literal) => {
            #[target_feature(enable = "simd128")]
            unsafe fn $name(
                src: (&[u16], usize),
                dst: (&mut [u16], usize),
                h: usize,
                taps: (&[i16; VP9_TAPS], &[i16; VP9_TAPS]),
                average: bool,
                depth: BitDepth,
            ) {
                // SAFETY: SIMD128 is enabled by this wrapper.
                unsafe { dispatch_avg::<$v, $w>(src, dst, h, taps, average, depth) }
            }
        };
    }

    simd128_width!(mc_w4, I16x4, 4);
    simd128_width!(mc_w8, I16x8, 8);
    simd128_width!(mc_w16, I16x8, 16);
    simd128_width!(mc_w32, I16x8, 32);
    simd128_width!(mc_w64, I16x8, 64);

    pub(super) const SIMD128: [Vp9Mc16Fn; 5] = [mc_w4, mc_w8, mc_w16, mc_w32, mc_w64];
}
