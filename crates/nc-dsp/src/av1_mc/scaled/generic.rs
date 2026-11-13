//! Generic SIMD scaled AV1 inter prediction over [`I16Lanes`],
//! [`I16Pairs`] and [`I16x8Shuffles`] (AV1 §7.11.3.4).
//!
//! Horizontal filtering preloads each output column's 8-tap filter vector
//! once per block, evaluates four 8-tap dot products at a time with pairwise
//! multiply-adds (`madd` + `sum_lanes4`), rounds by `InterRound0 = 3` and
//! narrows to `i16` (`[-1785, 5865]`, exact). Vertical filtering runs
//! contiguous `V::LANES`-wide `acc_mul_add2` passes across the eight
//! intermediate rows of each output row's `y` phase, rounded by
//! `InterRound1 = 11` (`Clip1`) or `InterRound1 = 7` (compound).

#![allow(unsafe_code, clippy::inline_always)]

use core::mem::MaybeUninit;

use nc_common::{PlaneView, PlaneViewMut};

use crate::av1_mc::convolve::{AV1_MAX_PREDICTION_EDGE, Av1McAxis, check_prediction};
use crate::av1_mc::filter::{AV1_FILTER_TAPS, Av1Filters, Kernel};
use crate::av1_mc::{INTER_ROUND0, INTER_ROUND1_COMPOUND, INTER_ROUND1_FINAL, clip_pixel, round2};
use crate::simd::{I16Lanes, I16Pairs, I16x8Shuffles, I32Lanes};

/// Inline stack scratch capacity (`i16` samples, 16 KiB): covers every
/// upscaled and unscaled block up to 64x64 (`rows <= 71`, `needed <= 4544`)
/// without heap allocation, while staying at clippy's 16 KiB stack limit.
const INLINE_INTERMEDIATE_SAMPLES: usize = 128 * 64;

/// The first `len` samples of `buf`, zeroed. Only the span a block needs is
/// written: zeroing the whole 16 KiB array on every call was 7% of a
/// `480p30 + 720p5` encode's instructions (callgrind `memset`), since mode
/// decision predicts every scaled candidate through here.
#[inline(always)]
fn zeroed_prefix(buf: &mut [MaybeUninit<i16>], len: usize) -> &mut [i16] {
    let prefix = &mut buf[..len];
    prefix.fill(MaybeUninit::new(0));
    // SAFETY: every element of `prefix` was initialised by the fill above,
    // and `MaybeUninit<i16>` has the same layout as `i16`.
    unsafe { &mut *(core::ptr::from_mut::<[MaybeUninit<i16>]>(prefix) as *mut [i16]) }
}

/// Final 8-bit scaled prediction (`InterRound1 = 11`, then `Clip1`).
///
/// # Safety
///
/// The caller must enable the target ISA of `V4`, `V8`, and `VWide`.
#[inline(always)]
pub(super) unsafe fn predict<V4, V8, VWide>(
    filters: Av1Filters,
    window: &PlaneView<'_>,
    (w, h): (usize, usize),
    (x, y): (Av1McAxis, Av1McAxis),
    dst: &mut PlaneViewMut<'_>,
) where
    V4: I16Lanes,
    V8: I16x8Shuffles + I16Pairs<Pairs = <V8 as I16x8Shuffles>::Wide>,
    VWide: I16Lanes,
{
    assert!(
        dst.width() >= w && dst.height() >= h,
        "AV1 prediction destination {}x{} is smaller than {w}x{h}",
        dst.width(),
        dst.height()
    );
    check_prediction(window, (w, h), (x, y));
    let kernel_x = filters.x.kernel(w);
    let kernel_y = filters.y.kernel(h);
    let rows = y.span(h);
    let needed = rows * w;
    let mut stack_buf = [MaybeUninit::<i16>::uninit(); INLINE_INTERMEDIATE_SAMPLES];
    let mut heap_buf = Vec::new();
    let inter: &mut [i16] = if needed <= INLINE_INTERMEDIATE_SAMPLES {
        zeroed_prefix(&mut stack_buf, needed)
    } else {
        heap_buf.resize(needed, 0);
        &mut heap_buf
    };
    horizontal_pass::<V8>(kernel_x, window, w, rows, x, inter);

    let round = 1i32 << (INTER_ROUND1_FINAL - 1);
    for r in 0..h {
        let (start, phase) = y.sample(r);
        let fy = &kernel_y[phase];
        let base = start * w;
        let out_row = &mut dst.row_mut(r)[..w];
        let mut c0 = 0usize;
        while c0 + VWide::LANES <= w {
            let acc = vertical_acc::<VWide>(inter, base + c0, w, fy);
            VWide::acc_round_shift_store_u8(acc, round, INTER_ROUND1_FINAL, &mut out_row[c0..]);
            c0 += VWide::LANES;
        }
        if V8::LANES < VWide::LANES && c0 + V8::LANES <= w {
            let acc = vertical_acc::<V8>(inter, base + c0, w, fy);
            V8::acc_round_shift_store_u8(acc, round, INTER_ROUND1_FINAL, &mut out_row[c0..]);
            c0 += V8::LANES;
        }
        if c0 + V4::LANES <= w {
            let acc = vertical_acc::<V4>(inter, base + c0, w, fy);
            V4::acc_round_shift_store_u8(acc, round, INTER_ROUND1_FINAL, &mut out_row[c0..]);
            c0 += V4::LANES;
        }
        for (c, out) in out_row.iter_mut().enumerate().skip(c0) {
            let sum: i32 = fy
                .iter()
                .enumerate()
                .map(|(t, &tap)| i32::from(tap) * i32::from(inter[(start + t) * w + c]))
                .sum();
            *out = clip_pixel(round2(sum, INTER_ROUND1_FINAL));
        }
    }
}

/// Compound-precision scaled prediction (`InterRound1 = 7`, `i16` output).
///
/// # Safety
///
/// The caller must enable the target ISA of `V4`, `V8`, and `VWide`.
#[inline(always)]
pub(super) unsafe fn predict_compound<V4, V8, VWide>(
    filters: Av1Filters,
    window: &PlaneView<'_>,
    (w, h): (usize, usize),
    (x, y): (Av1McAxis, Av1McAxis),
    dst: &mut [i16],
    dst_stride: usize,
) where
    V4: I16Lanes,
    V8: I16x8Shuffles + I16Pairs<Pairs = <V8 as I16x8Shuffles>::Wide>,
    VWide: I16Lanes,
{
    check_prediction(window, (w, h), (x, y));
    let kernel_x = filters.x.kernel(w);
    let kernel_y = filters.y.kernel(h);
    let rows = y.span(h);
    let needed = rows * w;
    let mut stack_buf = [MaybeUninit::<i16>::uninit(); INLINE_INTERMEDIATE_SAMPLES];
    let mut heap_buf = Vec::new();
    let inter: &mut [i16] = if needed <= INLINE_INTERMEDIATE_SAMPLES {
        zeroed_prefix(&mut stack_buf, needed)
    } else {
        heap_buf.resize(needed, 0);
        &mut heap_buf
    };
    horizontal_pass::<V8>(kernel_x, window, w, rows, x, inter);

    let round = 1i32 << (INTER_ROUND1_COMPOUND - 1);
    for r in 0..h {
        let (start, phase) = y.sample(r);
        let fy = &kernel_y[phase];
        let base = start * w;
        let out_row = &mut dst[r * dst_stride..r * dst_stride + w];
        let mut c0 = 0usize;
        while c0 + VWide::LANES <= w {
            let acc = vertical_acc::<VWide>(inter, base + c0, w, fy);
            VWide::acc_round_shift_narrow(acc, round, INTER_ROUND1_COMPOUND)
                .store(&mut out_row[c0..]);
            c0 += VWide::LANES;
        }
        if V8::LANES < VWide::LANES && c0 + V8::LANES <= w {
            let acc = vertical_acc::<V8>(inter, base + c0, w, fy);
            V8::acc_round_shift_narrow(acc, round, INTER_ROUND1_COMPOUND).store(&mut out_row[c0..]);
            c0 += V8::LANES;
        }
        if c0 + V4::LANES <= w {
            let acc = vertical_acc::<V4>(inter, base + c0, w, fy);
            V4::acc_round_shift_narrow(acc, round, INTER_ROUND1_COMPOUND).store(&mut out_row[c0..]);
            c0 += V4::LANES;
        }
        for (c, out) in out_row.iter_mut().enumerate().skip(c0) {
            let sum: i32 = fy
                .iter()
                .enumerate()
                .map(|(t, &tap)| i32::from(tap) * i32::from(inter[(start + t) * w + c]))
                .sum();
            *out = round2(sum, INTER_ROUND1_COMPOUND) as i16;
        }
    }
}

#[inline(always)]
fn horizontal_pass<V8>(
    kernel_x: &Kernel,
    window: &PlaneView<'_>,
    w: usize,
    rows: usize,
    x: Av1McAxis,
    inter: &mut [i16],
) where
    V8: I16x8Shuffles + I16Pairs<Pairs = <V8 as I16x8Shuffles>::Wide>,
{
    let mut starts_uninit = [MaybeUninit::<usize>::uninit(); AV1_MAX_PREDICTION_EDGE];
    let mut phases_uninit = [MaybeUninit::<u8>::uninit(); AV1_MAX_PREDICTION_EDGE];
    let mut taps_uninit = [MaybeUninit::<V8>::uninit(); AV1_MAX_PREDICTION_EDGE];
    for c in 0..w {
        let (start, phase) = x.sample(c);
        starts_uninit[c].write(start);
        phases_uninit[c].write(phase as u8);
        taps_uninit[c].write(V8::load(&kernel_x[phase]));
    }
    // SAFETY: indices `0..w` were initialised in the loop above.
    let (starts, phases, taps_x) = unsafe {
        (
            &*(core::ptr::from_ref::<[MaybeUninit<usize>]>(&starts_uninit[..w])
                as *const [usize]),
            &*(core::ptr::from_ref::<[MaybeUninit<u8>]>(&phases_uninit[..w]) as *const [u8]),
            &*(core::ptr::from_ref::<[MaybeUninit<V8>]>(&taps_uninit[..w]) as *const [V8]),
        )
    };
    let w8 = w & !7;
    for (r, line) in inter.chunks_exact_mut(w).take(rows).enumerate() {
        let source = window.row(r);
        let mut c0 = 0usize;
        while c0 < w8 {
            let lo = dot4::<V8>(source, &starts[c0..], &taps_x[c0..]);
            let hi = dot4::<V8>(source, &starts[c0 + 4..], &taps_x[c0 + 4..]);
            V8::narrow_pairs(lo, hi).store(&mut line[c0..]);
            c0 += 8;
        }
        if c0 + 4 <= w {
            let lo = dot4::<V8>(source, &starts[c0..], &taps_x[c0..]);
            let mut tmp = [0i16; 8];
            V8::narrow_pairs(lo, lo).store(&mut tmp);
            line[c0..c0 + 4].copy_from_slice(&tmp[..4]);
            c0 += 4;
        }
        for c in c0..w {
            let start = starts[c];
            let phase = usize::from(phases[c]);
            let sum: i32 = kernel_x[phase]
                .iter()
                .zip(&source[start..start + AV1_FILTER_TAPS])
                .map(|(&t, &s)| i32::from(t) * i32::from(s))
                .sum();
            line[c] = round2(sum, INTER_ROUND0) as i16;
        }
    }
}

#[inline(always)]
fn dot4<V8: I16x8Shuffles>(source: &[u8], starts: &[usize], taps: &[V8]) -> V8::Wide {
    let p0 = V8::load_u8(&source[starts[0]..]).madd(taps[0]);
    let p1 = V8::load_u8(&source[starts[1]..]).madd(taps[1]);
    let p2 = V8::load_u8(&source[starts[2]..]).madd(taps[2]);
    let p3 = V8::load_u8(&source[starts[3]..]).madd(taps[3]);
    V8::sum_lanes4([p0, p1, p2, p3]).round_shift(INTER_ROUND0)
}

#[inline(always)]
fn vertical_acc<V: I16Lanes>(
    inter: &[i16],
    offset: usize,
    stride: usize,
    fy: &[i16; AV1_FILTER_TAPS],
) -> V::Acc {
    let mut acc = V::acc_zero();
    for t in (0..AV1_FILTER_TAPS).step_by(2) {
        let a = V::load(&inter[offset + t * stride..]);
        let b = V::load(&inter[offset + (t + 1) * stride..]);
        acc = V::acc_mul_add2(acc, a, fy[t], b, fy[t + 1]);
    }
    acc
}
