//! Prediction generic over [`I16Lanes`]: `V::LANES` output samples of a row
//! per step.
//!
//! - AV1 taps are halved (module docs), so the rounding shifts are one less:
//!   horizontal `(sum + 2) >> 2`, vertical `(sum + 512) >> 10`, and the
//!   single-pass forms `((x + 2) >> 2 + 8) >> 4` and `(y + 32) >> 6`.
//! - HEVC sums stay below `2^15` horizontally (at most `112 * 255`); the
//!   2-D form `((v >> 6) + 32) >> 6` equals `(v + 2048) >> 12`.
//! - H.264 half-sample sums stay below `2^15` (at most `52 * 255`); the
//!   centre sample `j` filters them again in 32 bits.
//!
//! AV1 and the shared 2-D driver validate full footprints once, then use
//! the same pointer-lane access pattern as our H.264 driver. dav1d 1.5.4
//! `mc_tmpl.c` (VideoLAN/dav1d and Two Orioles, BSD-2-Clause) was reviewed
//! for separate copy/h/v paths; the filter graphs and rounding remain ours.
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs).

#![allow(clippy::inline_always)]

#[cfg(target_arch = "aarch64")]
use super::{Av1Family, TAP_CENTRE, TAPS};
use super::{
    H264_CHROMA_CLASS_CAP, H264_CHROMA_COPY, H264_CHROMA_HORIZONTAL, H264_CHROMA_VERTICAL,
    h264_slot,
};
use crate::simd::I16Lanes;

/// Validate a rectangular footprint once before pointer-based lane accesses.
/// Pixel and compound outputs use the same element-count contract.
#[inline(always)]
pub(crate) fn check_plane<T>(data: &[T], stride: usize, w: usize, h: usize) {
    let end = h
        .checked_sub(1)
        .and_then(|row| row.checked_mul(stride))
        .and_then(|at| at.checked_add(w));
    assert!(
        w > 0 && stride >= w && end.is_some_and(|end| end <= data.len()),
        "prediction plane footprint too small"
    );
}

/// Horizontal filter at a pointer within a validated source footprint.
///
/// # Safety
/// `row` covers `V::LANES + taps.len() - 1` bytes and V's ISA is enabled.
#[inline(always)]
unsafe fn filter_row_at<V: I16Lanes>(row: *const u8, taps: &[i16]) -> V {
    let mut sum = V::splat(0);
    for (t, &tap) in taps.iter().enumerate() {
        // SAFETY: the caller validates every tap's complete vector load.
        sum = sum.add(unsafe { V::load_u8_at(row.add(t)) }.mul(V::splat(tap)));
    }
    sum
}

/// Store compound lanes to a validated destination footprint.
#[inline(always)]
unsafe fn store_words<V: I16Lanes>(value: V, dst: *mut i16) {
    // SAFETY: the caller provides V::LANES writable words.
    value.store(unsafe { core::slice::from_raw_parts_mut(dst, V::LANES) });
}

/// AV1 prediction of a `w`x`h` block from its window (module contract)
/// with filter family `F`.
///
/// # Safety
///
/// `V`'s ISA must be available.
// x86-64 and wasm32 run `av1_avx2` and `av1_simd128` instead.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn av1_convolve<V: I16Lanes, F: Av1Family>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    // Families whose outer taps are identically zero (regular, smooth) keep
    // the original window origin but never load or multiply those samples.
    if F::OUTER_ZERO {
        av1_taps::<V, F, 6>(src, src_stride, dst, dst_stride, (w, h), (frac_x, frac_y));
    } else {
        av1_taps::<V, F, TAPS>(src, src_stride, dst, dst_stride, (w, h), (frac_x, frac_y));
    }
}

/// [`av1_convolve`] over the central `N` taps of each filter (`N` is 6 or
/// 8): the window's first `(8 - N) / 2` rows and columns are skipped.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn av1_taps<V: I16Lanes, F: Av1Family, const N: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    (w, h): (usize, usize),
    (frac_x, frac_y): (usize, usize),
) {
    check_plane(src, src_stride, w + TAPS - 1, h + TAPS - 1);
    check_plane(dst, dst_stride, w, h);
    assert!(w.is_multiple_of(V::LANES), "partial prediction vector");
    let (input, output) = (src.as_ptr(), dst.as_mut_ptr());

    // Whole-sample vectors (most blocks of static content) copy the window
    // centre; the filter set-up below would be wasted on them.
    if (frac_x, frac_y) == (0, 0) {
        for r in 0..h {
            let from = (r + TAP_CENTRE) * src_stride + TAP_CENTRE;
            // SAFETY: both complete row spans were checked above; slices are disjoint.
            unsafe {
                core::ptr::copy_nonoverlapping(input.add(from), output.add(r * dst_stride), w);
            }
        }
        return;
    }
    let lead = (TAPS - N) / 2;
    let (taps_x, taps_y) = (F::taps(w, frac_x), F::taps(h, frac_y));
    let mut fx = [0_i16; N];
    let mut fy = [0_i16; N];
    for ((x, y), (&tx, &ty)) in fx
        .iter_mut()
        .zip(&mut fy)
        .zip(taps_x[lead..].iter().zip(&taps_y[lead..]))
    {
        (*x, *y) = (tx / 2, ty / 2);
    }
    match (frac_x, frac_y) {
        (_, 0) => {
            for r in 0..h {
                // SAFETY: this complete tap row lies in the checked source footprint.
                let row = unsafe { input.add((r + TAP_CENTRE) * src_stride + lead) };
                for c0 in (0..w).step_by(V::LANES) {
                    // SAFETY: c0..c0+LANES and all N taps fit the checked row.
                    let sum = unsafe { filter_row_at::<V>(row.add(c0), &fx) };
                    let first = sum.add(V::splat(2)).sra(2);
                    // SAFETY: the checked output footprint covers this complete vector.
                    unsafe {
                        first
                            .add(V::splat(8))
                            .sra(4)
                            .store_u8_sat_at(output.add(r * dst_stride + c0));
                    }
                }
            }
        }
        (0, _) => {
            for r in 0..h {
                for c0 in (0..w).step_by(V::LANES) {
                    let mut sum = V::splat(0);
                    for (t, &tap) in fy.iter().enumerate() {
                        // SAFETY: the checked footprint includes every vertical tap row.
                        let s = unsafe {
                            V::load_u8_at(input.add((r + t + lead) * src_stride + c0 + TAP_CENTRE))
                        };
                        sum = sum.add(s.mul(V::splat(tap)));
                    }
                    // SAFETY: the checked output footprint covers this complete vector.
                    unsafe {
                        sum.add(V::splat(32))
                            .sra(6)
                            .store_u8_sat_at(output.add(r * dst_stride + c0));
                    }
                }
            }
        }
        _ => separable_2d::<V, N>(
            &src[lead * src_stride + lead..],
            src_stride,
            dst,
            dst_stride,
            w,
            h,
            &fx,
            &fy,
            (2, 2),
            (512, 10),
        ),
    }
}

/// AV1 prediction of a `w`x`h` block from its window (module contract)
/// with filter family `F` at compound precision (`InterRound1` = 7, AV1
/// §7.11.3.4): one of a compound block's two predictions, unclipped, for
/// [`av1_average`]. With halved taps the passes round `(sum + 2) >> 2` and
/// `(sum + 32) >> 6`; a pass whose offset is zero is the identity, scaling
/// by 16 (`<< 4`) when both are.
///
/// # Safety
///
/// `V`'s ISA must be available.
#[cfg(target_arch = "aarch64")] // NEON without dot products only
#[inline(always)]
pub(crate) unsafe fn av1_prep<V: I16Lanes, F: Av1Family>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    w: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    if F::OUTER_ZERO {
        av1_prep_taps::<V, F, 6>(src, src_stride, dst, dst_stride, (w, h), (frac_x, frac_y));
    } else {
        av1_prep_taps::<V, F, TAPS>(src, src_stride, dst, dst_stride, (w, h), (frac_x, frac_y));
    }
}

/// [`av1_prep`] over the central `N` taps of each filter, as
/// [`av1_taps`].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn av1_prep_taps<V: I16Lanes, F: Av1Family, const N: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    (w, h): (usize, usize),
    (frac_x, frac_y): (usize, usize),
) {
    check_plane(src, src_stride, w + TAPS - 1, h + TAPS - 1);
    check_plane(dst, dst_stride, w, h);
    assert!(w.is_multiple_of(V::LANES), "partial prediction vector");
    let (input, output) = (src.as_ptr(), dst.as_mut_ptr());

    if (frac_x, frac_y) == (0, 0) {
        for r in 0..h {
            // SAFETY: the identity row lies inside the validated tap window.
            let row = unsafe { input.add((r + TAP_CENTRE) * src_stride + TAP_CENTRE) };
            for c0 in (0..w).step_by(V::LANES) {
                // SAFETY: the checked source/output rows cover this complete vector.
                unsafe {
                    store_words(
                        V::load_u8_at(row.add(c0)).mul(V::splat(COMPOUND_SCALE)),
                        output.add(r * dst_stride + c0),
                    );
                }
            }
        }
        return;
    }
    let lead = (TAPS - N) / 2;
    let (taps_x, taps_y) = (F::taps(w, frac_x), F::taps(h, frac_y));
    let mut fx = [0_i16; N];
    let mut fy = [0_i16; N];
    for ((x, y), (&tx, &ty)) in fx
        .iter_mut()
        .zip(&mut fy)
        .zip(taps_x[lead..].iter().zip(&taps_y[lead..]))
    {
        (*x, *y) = (tx / 2, ty / 2);
    }
    match (frac_x, frac_y) {
        (_, 0) => {
            for r in 0..h {
                // SAFETY: this complete tap row lies in the checked source footprint.
                let row = unsafe { input.add((r + TAP_CENTRE) * src_stride + lead) };
                for c0 in (0..w).step_by(V::LANES) {
                    // SAFETY: all horizontal taps and this output vector fit the checked footprints.
                    unsafe {
                        store_words(
                            filter_row_at::<V>(row.add(c0), &fx).add(V::splat(2)).sra(2),
                            output.add(r * dst_stride + c0),
                        );
                    }
                }
            }
        }
        (0, _) => {
            for r in 0..h {
                for c0 in (0..w).step_by(V::LANES) {
                    let mut sum = V::splat(0);
                    for (t, &tap) in fy.iter().enumerate() {
                        // SAFETY: every vertical tap fits the checked source footprint.
                        let s = unsafe {
                            V::load_u8_at(input.add((r + t + lead) * src_stride + c0 + TAP_CENTRE))
                        };
                        sum = sum.add(s.mul(V::splat(tap)));
                    }
                    // SAFETY: the checked output footprint covers this complete vector.
                    unsafe {
                        store_words(sum.add(V::splat(2)).sra(2), output.add(r * dst_stride + c0));
                    }
                }
            }
        }
        _ => separable_2d_with::<V, N, _>(
            &src[lead * src_stride + lead..],
            src_stride,
            (w, h),
            &fx,
            &fy,
            (2, 2),
            &mut CompoundOutput {
                dst,
                stride: dst_stride,
            },
        ),
    }
}

/// Scale of a whole-sample compound prediction (`1 << (14 - 8 - 2)`
/// halved-tap bits: the sample in the compound domain).
#[cfg(target_arch = "aarch64")]
const COMPOUND_SCALE: i16 = 16;

/// The rounded average of two compound predictions of a `w`x`h` block
/// (AV1 §7.11.3.15 at equal weights: `Round2(p0 + p1, 5)`, clipped):
/// `p0` and `p1` at `src_stride`, `dst` at `dst_stride`. Two predictions
/// sum within 16 bits (each below `2^14` in magnitude).
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(crate) unsafe fn av1_average<V: I16Lanes>(
    p0: &[i16],
    p1: &[i16],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    check_plane(p0, src_stride, w, h);
    check_plane(p1, src_stride, w, h);
    check_plane(dst, dst_stride, w, h);
    assert!(w.is_multiple_of(V::LANES), "partial prediction vector");
    for r in 0..h {
        for c0 in (0..w).step_by(V::LANES) {
            let at = r * src_stride + c0;
            // SAFETY: all three footprints cover these complete vectors;
            // exact-length slices fold the lane helpers' bounds checks away.
            unsafe {
                let a = core::slice::from_raw_parts(p0.as_ptr().add(at), V::LANES);
                let b = core::slice::from_raw_parts(p1.as_ptr().add(at), V::LANES);
                V::load(a)
                    .add(V::load(b))
                    .add(V::splat(16))
                    .sra(5)
                    .store_u8_sat_at(dst.as_mut_ptr().add(r * dst_stride + c0));
            }
        }
    }
}

/// Two-pass filter of a `w`x`h` block: horizontal sums rounded by
/// `first = (round, shift)` (no rounding when `shift` is 0), then the
/// vertical filter in 32 bits, rounded by `last` and clipped to 8 bits. Each
/// column of `V::LANES` outputs keeps a rolling window of its last
/// `fy.len()` horizontally filtered rows in registers, so no scratch buffer
/// is cleared or written per call.
///
/// `N` (the tap count) is a constant so that the window is a fixed set of
/// registers and every tap loop unrolls.
#[cfg(not(target_arch = "wasm32"))] // SIMD128 has `separable_simd128`
#[inline(always)]
fn separable_2d<V: I16Lanes, const N: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
    fx: &[i16; N],
    fy: &[i16; N],
    first: (i16, u32),
    (last_round, last_shift): (i32, u32),
) {
    let mut output = ByteOutput {
        dst,
        stride: dst_stride,
        round: last_round,
        shift: last_shift,
    };
    separable_2d_with::<V, N, _>(src, src_stride, (w, h), fx, fy, first, &mut output);
}

/// Where [`separable_2d_with`] puts each vertical accumulator of
/// `V::LANES` outputs (a trait rather than a closure so that the store
/// always inlines: as a call it cost ~4% of AV2 P-frame instructions).
trait VerticalOutput<V: I16Lanes> {
    fn check(&self, w: usize, h: usize);
    /// Safety: `check` covered this row and `V::LANES` outputs at c0.
    unsafe fn store(&mut self, acc: V::Acc, row: usize, c0: usize);
}

#[cfg(not(target_arch = "wasm32"))] // SIMD128 has `separable_simd128`
/// Final 8-bit samples: `clamp((acc + round) >> shift)`.
struct ByteOutput<'a> {
    dst: &'a mut [u8],
    stride: usize,
    round: i32,
    shift: u32,
}

#[cfg(not(target_arch = "wasm32"))] // SIMD128 has `separable_simd128`
impl<V: I16Lanes> VerticalOutput<V> for ByteOutput<'_> {
    #[inline(always)]
    fn check(&self, w: usize, h: usize) {
        check_plane(self.dst, self.stride, w, h);
    }
    #[inline(always)]
    unsafe fn store(&mut self, acc: V::Acc, row: usize, c0: usize) {
        // SAFETY: caller checked this complete vector in the output footprint.
        let dst = unsafe {
            core::slice::from_raw_parts_mut(
                self.dst.as_mut_ptr().add(row * self.stride + c0),
                V::LANES,
            )
        };
        V::acc_round_shift_store_u8(acc, self.round, self.shift, dst);
    }
}

/// Compound-precision samples: `(acc + 32) >> 6` in 16 bits.
#[cfg(target_arch = "aarch64")]
struct CompoundOutput<'a> {
    dst: &'a mut [i16],
    stride: usize,
}

#[cfg(target_arch = "aarch64")]
impl<V: I16Lanes> VerticalOutput<V> for CompoundOutput<'_> {
    #[inline(always)]
    fn check(&self, w: usize, h: usize) {
        check_plane(self.dst, self.stride, w, h);
    }
    #[inline(always)]
    unsafe fn store(&mut self, acc: V::Acc, row: usize, c0: usize) {
        // SAFETY: caller checked this complete vector in the output footprint.
        unsafe {
            store_words(
                V::acc_round_shift_narrow(acc, 32, 6),
                self.dst.as_mut_ptr().add(row * self.stride + c0),
            );
        }
    }
}

/// [`separable_2d`]'s passes, handing each vertical accumulator of
/// `V::LANES` outputs, with its row and first column, to `output`.
#[inline(always)]
fn separable_2d_with<V: I16Lanes, const N: usize, O: VerticalOutput<V>>(
    src: &[u8],
    src_stride: usize,
    (w, h): (usize, usize),
    fx: &[i16; N],
    fy: &[i16; N],
    (first_round, first_shift): (i16, u32),
    output: &mut O,
) {
    check_plane(src, src_stride, w + N - 1, h + N - 1);
    output.check(w, h);
    assert!(w.is_multiple_of(V::LANES), "partial prediction vector");
    let source = src.as_ptr();
    for c0 in (0..w).step_by(V::LANES) {
        let mut window = [V::splat(0); N];
        for (t, row) in window.iter_mut().enumerate().take(N - 1) {
            // SAFETY: every tap lies inside the checked source footprint.
            *row = unsafe {
                first_pass_at::<V>(
                    source.add(t * src_stride + c0),
                    fx,
                    first_round,
                    first_shift,
                )
            };
        }
        for r in 0..h {
            // SAFETY: the final rolling-window row lies in the checked footprint.
            window[N - 1] = unsafe {
                first_pass_at::<V>(
                    source.add((r + N - 1) * src_stride + c0),
                    fx,
                    first_round,
                    first_shift,
                )
            };
            let mut acc = V::acc_zero();
            for t in (0..N).step_by(2) {
                acc = V::acc_mul_add2(acc, window[t], fy[t], window[t + 1], fy[t + 1]);
            }
            // SAFETY: output.check validated every row and w is a multiple of LANES.
            unsafe {
                output.store(acc, r, c0);
            }
            window.copy_within(1.., 0);
        }
    }
}

/// Keep the horizontal graph in the ISA-enabled caller. An outlined closure
/// otherwise calls individual vector intrinsics across a target-feature boundary.
#[inline(always)]
unsafe fn first_pass_at<V: I16Lanes>(row: *const u8, taps: &[i16], round: i16, shift: u32) -> V {
    // SAFETY: the caller checked the complete tap window for these lanes.
    let sum = unsafe { filter_row_at::<V>(row, taps) };
    if shift == 0 {
        sum
    } else {
        sum.add(V::splat(round)).sra(shift)
    }
}

/// HEVC vertical sums retain their full range by centering before the
/// shared accumulator narrows to signed sample lanes. Unlike AV1 compound
/// output, HEVC has no rounding term before its six-bit precision shift.
struct HevcOutput<'a> {
    dst: &'a mut [i16],
    stride: usize,
}

impl<V: I16Lanes> VerticalOutput<V> for HevcOutput<'_> {
    #[inline(always)]
    fn check(&self, w: usize, h: usize) {
        check_plane(self.dst, self.stride, w, h);
    }
    #[inline(always)]
    unsafe fn store(&mut self, acc: V::Acc, row: usize, c0: usize) {
        let value = V::acc_round_shift_narrow(acc, -(i32::from(super::HEVC_PREP_OFFSET) << 6), 6);
        // SAFETY: the shared driver checked every destination lane.
        unsafe {
            store_words(value, self.dst.as_mut_ptr().add(row * self.stride + c0));
        }
    }
}

/// HEVC 14-bit prediction for weighted/bi-prediction (HEVC §8.5.3.3):
/// copy pixels scaled by 64, unrounded one-dimensional tap sums, or the
/// two-dimensional sum shifted by six without AV1's compound rounding.
/// Every result is centered by 8192 before narrowing to preserve overshoot.
///
/// # Safety
/// `V`'s ISA must be enabled. Whole footprints are checked before accesses.
#[inline(always)]
pub(crate) unsafe fn hevc_prep<V: I16Lanes, const N: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    w: usize,
    h: usize,
    fx: &[i16; N],
    fy: &[i16; N],
    (frac_x, frac_y): (usize, usize),
) {
    check_plane(src, src_stride, w + N - 1, h + N - 1);
    check_plane(dst, dst_stride, w, h);
    assert!(w.is_multiple_of(V::LANES), "partial prediction vector");
    let centre = N / 2 - 1;
    let (input, output) = (src.as_ptr(), dst.as_mut_ptr());
    match (frac_x, frac_y) {
        (0, 0) => {
            for r in 0..h {
                for c in (0..w).step_by(V::LANES) {
                    // SAFETY: the checked source and destination contain both vectors.
                    unsafe {
                        store_words(
                            V::load_u8_at(input.add((r + centre) * src_stride + c + centre))
                                .mul(V::splat(64))
                                .sub(V::splat(super::HEVC_PREP_OFFSET)),
                            output.add(r * dst_stride + c),
                        );
                    }
                }
            }
        }
        (_, 0) => {
            for r in 0..h {
                for c in (0..w).step_by(V::LANES) {
                    // SAFETY: every horizontal tap and output lane lies in the checked footprint.
                    unsafe {
                        store_words(
                            filter_row_at::<V>(input.add((r + centre) * src_stride + c), fx)
                                .sub(V::splat(super::HEVC_PREP_OFFSET)),
                            output.add(r * dst_stride + c),
                        );
                    }
                }
            }
        }
        (0, _) => {
            for r in 0..h {
                for c in (0..w).step_by(V::LANES) {
                    let mut sum = V::splat(0);
                    for (t, &tap) in fy.iter().enumerate() {
                        // SAFETY: every vertical tap row covers these complete lanes.
                        let value =
                            unsafe { V::load_u8_at(input.add((r + t) * src_stride + c + centre)) };
                        sum = sum.add(value.mul(V::splat(tap)));
                    }
                    // SAFETY: the checked destination includes this complete vector.
                    unsafe {
                        store_words(
                            sum.sub(V::splat(super::HEVC_PREP_OFFSET)),
                            output.add(r * dst_stride + c),
                        );
                    }
                }
            }
        }
        _ => separable_2d_with::<V, N, _>(
            src,
            src_stride,
            (w, h),
            fx,
            fy,
            (0, 0),
            &mut HevcOutput {
                dst,
                stride: dst_stride,
            },
        ),
    }
}

/// `first + 8192 + 64`, with the first prediction centred: the bias of
/// HEVC default bi-prediction (HEVC eq. 8-251 at 8-bit depth: `shift2 = 7`,
/// `offset2 = 64`) once the centring offset is restored.
const HEVC_BI_BIAS: i16 = super::HEVC_PREP_OFFSET + 64;
/// `shift2` of default bi-prediction.
const HEVC_BI_SHIFT: u32 = 7;

/// Default-weight HEVC bi-prediction of the 2-D case: the second list's
/// vertical sum (before its `>> 6`) and the first list's centred
/// prediction, averaged into 8-bit samples. Rounding
/// `((acc >> 6) + first + bias) >> 7` is one floor shift:
/// `(acc + 64·(first + bias)) >> 13`.
struct HevcBiOutput<'a> {
    first: &'a [i16],
    first_stride: usize,
    dst: &'a mut [u8],
    stride: usize,
}

impl<V: I16Lanes> VerticalOutput<V> for HevcBiOutput<'_> {
    #[inline(always)]
    fn check(&self, w: usize, h: usize) {
        check_plane(self.first, self.first_stride, w, h);
        check_plane(self.dst, self.stride, w, h);
    }
    #[inline(always)]
    unsafe fn store(&mut self, acc: V::Acc, row: usize, c0: usize) {
        let first = V::load(&self.first[row * self.first_stride + c0..]);
        let scale = 1 << 6;
        let sum = V::acc_mul_add2(acc, first, scale, V::splat(HEVC_BI_BIAS), scale);
        V::acc_round_shift_store_u8(
            sum,
            0,
            HEVC_BI_SHIFT + 6,
            &mut self.dst[row * self.stride + c0..],
        );
    }
}

/// One vector of default-weight bi-prediction from an uncentred
/// one-dimensional or whole-sample prediction `second` (which fits i16) and
/// the first list's centred prediction at row `r`, column `c`, averaged in
/// 32-bit lanes. A function, not a closure: an outlined closure would call
/// the vector intrinsics across the target-feature boundary.
#[inline(always)]
fn hevc_bi_average<V: I16Lanes>(
    second: V,
    (first, first_stride): (&[i16], usize),
    (dst, dst_stride): (&mut [u8], usize),
    r: usize,
    c: usize,
) {
    let first = V::load(&first[r * first_stride + c..]);
    let sum = V::acc_mul_add2(V::acc_zero(), second, 1, first, 1);
    V::acc_round_shift_store_u8(
        sum,
        i32::from(HEVC_BI_BIAS),
        HEVC_BI_SHIFT,
        &mut dst[r * dst_stride + c..],
    );
}

/// Default-weight HEVC bi-prediction (HEVC §8.5.3.3.4.2, eq. 8-251) of a
/// `w`x`h` block: the second list's prediction from `src`, computed as
/// [`hevc_prep`] does but kept in registers, averaged with the first
/// list's centred prediction `first` straight into 8-bit samples (no
/// 16-bit store and reload of the second prediction). The average takes
/// the paired 32-bit sum of `VVdeC`'s `addAvg` (BSD-3-Clause-Clear).
///
/// # Safety
/// `V`'s ISA must be enabled. Whole footprints are checked before accesses.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn hevc_bi<V: I16Lanes, const N: usize>(
    src: &[u8],
    src_stride: usize,
    (first, first_stride): (&[i16], usize),
    dst: &mut [u8],
    dst_stride: usize,
    (w, h): (usize, usize),
    (fx, fy): (&[i16; N], &[i16; N]),
    (frac_x, frac_y): (usize, usize),
) {
    check_plane(src, src_stride, w + N - 1, h + N - 1);
    check_plane(first, first_stride, w, h);
    check_plane(dst, dst_stride, w, h);
    assert!(w.is_multiple_of(V::LANES), "partial prediction vector");
    let centre = N / 2 - 1;
    let input = src.as_ptr();
    match (frac_x, frac_y) {
        (0, 0) => {
            for r in 0..h {
                for c in (0..w).step_by(V::LANES) {
                    // SAFETY: the checked source contains this vector.
                    let s =
                        unsafe { V::load_u8_at(input.add((r + centre) * src_stride + c + centre)) };
                    hevc_bi_average::<V>(
                        s.mul(V::splat(64)),
                        (first, first_stride),
                        (dst, dst_stride),
                        r,
                        c,
                    );
                }
            }
        }
        (_, 0) => {
            for r in 0..h {
                for c in (0..w).step_by(V::LANES) {
                    // SAFETY: every horizontal tap lies in the checked footprint.
                    let s =
                        unsafe { filter_row_at::<V>(input.add((r + centre) * src_stride + c), fx) };
                    hevc_bi_average::<V>(s, (first, first_stride), (dst, dst_stride), r, c);
                }
            }
        }
        (0, _) => {
            for r in 0..h {
                for c in (0..w).step_by(V::LANES) {
                    let mut sum = V::splat(0);
                    for (t, &tap) in fy.iter().enumerate() {
                        // SAFETY: every vertical tap row covers these complete lanes.
                        let value =
                            unsafe { V::load_u8_at(input.add((r + t) * src_stride + c + centre)) };
                        sum = sum.add(value.mul(V::splat(tap)));
                    }
                    hevc_bi_average::<V>(sum, (first, first_stride), (dst, dst_stride), r, c);
                }
            }
        }
        _ => separable_2d_with::<V, N, _>(
            src,
            src_stride,
            (w, h),
            fx,
            fy,
            (0, 0),
            &mut HevcBiOutput {
                first,
                first_stride,
                dst,
                stride: dst_stride,
            },
        ),
    }
}

/// HEVC prediction (luma `fL` or chroma `fC`, `taps.len()` = 8 or 4) of a
/// `w`x`h` block.
///
/// # Safety
///
/// `V`'s ISA must be available.
#[cfg(not(target_arch = "wasm32"))] // SIMD128 has `separable_simd128`
#[inline(always)]
pub(crate) unsafe fn hevc<V: I16Lanes, const N: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
    fx: &[i16; N],
    fy: &[i16; N],
    (frac_x, frac_y): (usize, usize),
) {
    let centre = N / 2 - 1;
    match (frac_x, frac_y) {
        (0, 0) => {
            for r in 0..h {
                let from = (r + centre) * src_stride + centre;
                dst[r * dst_stride..r * dst_stride + w].copy_from_slice(&src[from..from + w]);
            }
        }
        (_, 0) => {
            for r in 0..h {
                let row = &src[(r + centre) * src_stride..];
                for c0 in (0..w).step_by(V::LANES) {
                    filter_row::<V>(row, c0, fx)
                        .add(V::splat(32))
                        .sra(6)
                        .store_u8_sat(&mut dst[r * dst_stride + c0..]);
                }
            }
        }
        (0, _) => {
            for r in 0..h {
                for c0 in (0..w).step_by(V::LANES) {
                    let mut sum = V::splat(0);
                    for (t, &tap) in fy.iter().enumerate() {
                        let s = V::load_u8(&src[(r + t) * src_stride + c0 + centre..]);
                        sum = sum.add(s.mul(V::splat(tap)));
                    }
                    sum.add(V::splat(32))
                        .sra(6)
                        .store_u8_sat(&mut dst[r * dst_stride + c0..]);
                }
            }
        }
        _ => separable_2d::<V, N>(
            src,
            src_stride,
            dst,
            dst_stride,
            w,
            h,
            fx,
            fy,
            (0, 0),
            (2048, 12),
        ),
    }
}

/// Unrounded filter sum of `V::LANES` outputs from column `c0` of a row.
#[cfg(not(target_arch = "wasm32"))] // SIMD128 has `separable_simd128`
#[inline(always)]
fn filter_row<V: I16Lanes>(row: &[u8], c0: usize, taps: &[i16]) -> V {
    let mut sum = V::splat(0);
    for (t, &tap) in taps.iter().enumerate() {
        sum = sum.add(V::load_u8(&row[c0 + t..]).mul(V::splat(tap)));
    }
    sum
}

/// H.264's six-tap `[1, -5, 20, 20, -5, 1]` filter, unrounded, of six
/// vectors, using the filter's symmetry: `(x0 + x5) - 5·(x1 + x4) +
/// 20·(x2 + x3)` (spec eq. 8-241). Three adds and two multiplies instead of
/// six multiplies and five adds; the integer result is identical, and no
/// intermediate leaves the 16-bit range for 8-bit samples (|sum| < 2^14).
#[inline(always)]
fn six_tap<V: I16Lanes>(x: [V; 6]) -> V {
    let outer = x[0].add(x[5]);
    let inner = x[1].add(x[4]);
    let centre = x[2].add(x[3]);
    outer
        .sub(inner.mul(V::splat(5)))
        .add(centre.mul(V::splat(20)))
}

/// The six-tap sum of the `V::LANES` outputs starting at `p` (which points
/// two samples left of the first output's integer position).
///
/// # Safety
///
/// `p` must be valid for reading `V::LANES + 5` bytes.
#[inline(always)]
unsafe fn six_tap_at<V: I16Lanes>(p: *const u8) -> V {
    // SAFETY: the caller guarantees `LANES + 5` readable bytes from `p`.
    unsafe {
        six_tap([
            V::load_u8_at(p),
            V::load_u8_at(p.add(1)),
            V::load_u8_at(p.add(2)),
            V::load_u8_at(p.add(3)),
            V::load_u8_at(p.add(4)),
            V::load_u8_at(p.add(5)),
        ])
    }
}

/// `(sum + 16) >> 5` without the clip: for values stored straight to bytes,
/// whose saturating store performs `Clip1` itself.
#[inline(always)]
fn half_unclipped<V: I16Lanes>(sum: V) -> V {
    sum.add(V::splat(16)).sra(5)
}

/// Checks the module's kernel contract for an H.264 kernel once, so the
/// row loops can run on raw pointers: the window covers `h + extra` rows of
/// `w + extra` samples at `src_stride`, and `dst` covers `h` rows of `w`.
/// [`crate::Dsp::predict`] guarantees this; the assertion makes a broken
/// caller panic instead of reading out of bounds.
#[inline(always)]
pub(crate) fn check_h264_window(
    src: &[u8],
    src_stride: usize,
    dst: &[u8],
    dst_stride: usize,
    (w, h, extra): (usize, usize, usize),
) {
    assert!(
        h > 0
            && src_stride >= w + extra
            && src.len() >= (h + extra - 1) * src_stride + w + extra
            && dst_stride >= w
            && dst.len() >= (h - 1) * dst_stride + w,
        "H.264 prediction window or destination too small"
    );
}

/// H.264 luma prediction of a `w`x`h` block (window 2 up, 2 left). The
/// centre sample `j` filters a rolling window of six unrounded horizontal
/// sums per column of outputs, kept in registers.
///
/// The window and destination are checked once ([`check_h264_window`]); the
/// row loops then address them through raw pointers, with no per-row or
/// per-load bounds checks (`w` is a constant in each instantiation).
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(crate) unsafe fn h264_luma<V: I16Lanes>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    check_h264_window(src, src_stride, dst, dst_stride, (w, h, 5));
    let (src_ptr, dst_ptr) = (src.as_ptr(), dst.as_mut_ptr());
    // SAFETY: all pointer arithmetic below stays inside the checked area: every source access is in rows
    // `0..h + 5` at columns `0..w + 5` of the window and every store in rows
    // `0..h` at columns `0..w` of `dst`, which `check_h264_window` verified.
    unsafe {
        match (frac_x, frac_y) {
            (0, 0) => {
                for r in 0..h {
                    core::ptr::copy_nonoverlapping(
                        src_ptr.add((r + 2) * src_stride + 2),
                        dst_ptr.add(r * dst_stride),
                        w,
                    );
                }
            }
            (_, 0) => {
                // `b`, averaged with `G` (1) or its right neighbour (3).
                for r in 0..h {
                    let row = src_ptr.add((r + 2) * src_stride);
                    let out = dst_ptr.add(r * dst_stride);
                    for c0 in (0..w).step_by(V::LANES) {
                        let sum = six_tap_at::<V>(row.add(c0));
                        let value = match frac_x {
                            1 => avg(V::load_u8_at(row.add(c0 + 2)), half(sum)),
                            3 => avg(V::load_u8_at(row.add(c0 + 3)), half(sum)),
                            _ => half_unclipped(sum),
                        };
                        value.store_u8_sat_at(out.add(c0));
                    }
                }
            }
            (0, _) => {
                // `h`, averaged with `G` (1) or the sample below it (3): a
                // rolling window of six rows per column of outputs.
                for c0 in (0..w).step_by(V::LANES) {
                    let col = src_ptr.add(c0 + 2);
                    let mut rows: [V; 6] = [V::splat(0); 6];
                    for (t, row) in rows.iter_mut().enumerate() {
                        *row = V::load_u8_at(col.add(t * src_stride));
                    }
                    for r in 0..h {
                        let sum = six_tap(rows);
                        let value = match frac_y {
                            1 => avg(rows[2], half(sum)),
                            3 => avg(rows[3], half(sum)),
                            _ => half_unclipped(sum),
                        };
                        value.store_u8_sat_at(dst_ptr.add(r * dst_stride + c0));
                        if r + 1 < h {
                            rows.copy_within(1..6, 0);
                            rows[5] = V::load_u8_at(col.add((r + 6) * src_stride));
                        }
                    }
                }
            }
            _ => h264_luma_2d::<V>(
                src_ptr, src_stride, dst_ptr, dst_stride, w, h, frac_x, frac_y,
            ),
        }
    }
}

/// The two-dimensional positions of [`h264_luma`]. Per output column, two
/// rolling windows of six rows, each row computed once: the unrounded
/// horizontal 6-tap sums (they give the centre `j` and, rounded, the
/// horizontal halves `b` = row 2 and `s` = row 3), and the raw samples of
/// the column the vertical half `h` (or `m`, one to the right) needs.
///
/// # Safety
///
/// `V`'s ISA must be available, and `s`/`d` must satisfy
/// [`check_h264_window`] for `(w, h, 5)` (as `src_ptr`/`dst_ptr`).
#[inline(always)]
unsafe fn h264_luma_2d<V: I16Lanes>(
    src_ptr: *const u8,
    src_stride: usize,
    dst_ptr: *mut u8,
    dst_stride: usize,
    w: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    let vertical_dx = match frac_x {
        1 => Some(0),
        3 => Some(1),
        _ => None, // frac_x == 2: no vertical half
    };
    // No closures in here: they would not inherit the kernel's target
    // features (docs/ARCHITECTURE.md).
    // SAFETY: rows `0..h + 5`, columns `0..w + 5` of the window and the
    // `h`x`w` destination, as the caller guarantees.
    unsafe {
        for c0 in (0..w).step_by(V::LANES) {
            let col = src_ptr.add(c0 + 2 + vertical_dx.unwrap_or(0));
            let mut sums: [V; 6] = [V::splat(0); 6];
            let mut samples: [V; 6] = [V::splat(0); 6];
            for t in 0..6 {
                sums[t] = six_tap_at::<V>(src_ptr.add(t * src_stride + c0));
                if vertical_dx.is_some() {
                    samples[t] = V::load_u8_at(col.add(t * src_stride));
                }
            }
            for r in 0..h {
                let (b, s3) = (half(sums[2]), half(sums[3]));
                let value = match (frac_x, frac_y) {
                    (1 | 3, 1) => avg(b, vertical_half(&samples)),
                    (1 | 3, 3) => avg(vertical_half(&samples), s3),
                    (2, 1) => avg(b, centre::<V>(&sums)),
                    (2, 3) => avg(centre::<V>(&sums), s3),
                    (1 | 3, 2) => avg(vertical_half(&samples), centre::<V>(&sums)),
                    _ => centre::<V>(&sums),
                };
                value.store_u8_sat_at(dst_ptr.add(r * dst_stride + c0));
                if r + 1 < h {
                    sums.copy_within(1..6, 0);
                    sums[5] = six_tap_at::<V>(src_ptr.add((r + 6) * src_stride + c0));
                    if vertical_dx.is_some() {
                        samples.copy_within(1..6, 0);
                        samples[5] = V::load_u8_at(col.add((r + 6) * src_stride));
                    }
                }
            }
        }
    }
}

/// The vertical half sample (rounded, clipped) of six rows of samples.
#[inline(always)]
fn vertical_half<V: I16Lanes>(rows: &[V; 6]) -> V {
    half(six_tap(*rows))
}

/// H.264 rounded average of two samples.
#[inline(always)]
fn avg<V: I16Lanes>(p: V, q: V) -> V {
    p.add(q).add(V::splat(1)).sra(1)
}

/// `Clip1((sum + 16) >> 5)`: a half sample from its 6-tap sum.
#[inline(always)]
fn half<V: I16Lanes>(sum: V) -> V {
    sum.add(V::splat(16)).sra(5).clamp_u8()
}

/// Centre sample `j` from six unrounded horizontal sums, top to bottom.
#[inline(always)]
fn centre<V: I16Lanes>(sums: &[V; 6]) -> V {
    // Symmetric form, as in `six_tap`. The pair sums stay inside 16 bits
    // (each horizontal sum is in [-2550, 10710], so a pair is in
    // [-5100, 21420]); the weighted total needs 32 bits.
    let outer = sums[0].add(sums[5]);
    let inner = sums[1].add(sums[4]);
    let middle = sums[2].add(sums[3]);
    let acc = V::acc_mul_add2(V::acc_zero(), outer, 1, inner, -5);
    let acc = V::acc_mul_add2(acc, middle, 20, V::splat(0), 0);
    V::acc_round_shift_narrow(acc, 512, 10).clamp_u8()
}

/// H.264 chroma prediction: bilinear at eighth-sample offsets.
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(crate) unsafe fn h264_chroma<V: I16Lanes>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    // Separable form of the standard's 2-D weights (eq. 8-266): a
    // horizontal pass `(8 - dx)·A + dx·B` per row (at most 2040, one
    // byte-pair multiply-add), then `((8 - dy)·top + dy·bottom + 32) >> 6`
    // between consecutive rows. The products distribute exactly, so the
    // result is identical, and each source row is filtered once instead of
    // twice. With one offset zero the weights of the other direction are
    // 8 and 0, so the sum is 8 times a one-dimensional one and
    // `(8·s + 32) >> 6 = (s + 4) >> 3`: one pass, no second multiply; with
    // both zero the block is a copy. The window is checked once; the loops
    // run on raw pointers.
    let class = h264_slot(frac_x, frac_y, H264_CHROMA_CLASS_CAP);
    // SAFETY: forwarded.
    unsafe {
        h264_chroma_of_class::<V>(
            src, src_stride, dst, dst_stride, w, h, frac_x, frac_y, class,
        );
    }
}

/// [`h264_chroma`] of the offsets' class (`H264_CHROMA_*`, which
/// [`h264_slot`] gives): a constant `class` compiles to one path.
///
/// # Safety
///
/// `V`'s ISA must be available.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn h264_chroma_of_class<V: I16Lanes>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
    class: usize,
) {
    debug_assert_eq!(class, h264_slot(frac_x, frac_y, H264_CHROMA_CLASS_CAP));
    check_h264_window(src, src_stride, dst, dst_stride, (w, h, 1));
    let (src_ptr, dst_ptr) = (src.as_ptr(), dst.as_mut_ptr());
    let (dx, dy) = (frac_x as u8, frac_y as u8);
    // SAFETY: rows `0..h + 1` at columns `0..w + 1` of the window and the
    // `h`x`w` destination, verified by `check_h264_window`.
    unsafe {
        match class {
            H264_CHROMA_COPY => {
                for r in 0..h {
                    core::ptr::copy_nonoverlapping(
                        src_ptr.add(r * src_stride),
                        dst_ptr.add(r * dst_stride),
                        w,
                    );
                }
            }
            H264_CHROMA_HORIZONTAL | H264_CHROMA_VERTICAL => {
                // The second tap is the next column, or the next row.
                let (step, weight) = if class == H264_CHROMA_HORIZONTAL {
                    (1, dx)
                } else {
                    (src_stride, dy)
                };
                for r in 0..h {
                    let row = src_ptr.add(r * src_stride);
                    for c0 in (0..w).step_by(V::LANES) {
                        let a = row.add(c0);
                        V::mul_add_u8_pair_at(a, a.add(step), 8 - weight, weight)
                            .add(V::splat(4))
                            .sra(3)
                            .store_u8_sat_at(dst_ptr.add(r * dst_stride + c0));
                    }
                }
            }
            _ => {
                let (wy0, wy1) = (V::splat(8 - i16::from(dy)), V::splat(i16::from(dy)));
                for c0 in (0..w).step_by(V::LANES) {
                    let mut top =
                        V::mul_add_u8_pair_at(src_ptr.add(c0), src_ptr.add(c0 + 1), 8 - dx, dx);
                    for r in 0..h {
                        let row = src_ptr.add((r + 1) * src_stride + c0);
                        let bottom = V::mul_add_u8_pair_at(row, row.add(1), 8 - dx, dx);
                        top.mul(wy0)
                            .add(bottom.mul(wy1))
                            .add(V::splat(32))
                            .sra(6)
                            .store_u8_sat_at(dst_ptr.add(r * dst_stride + c0));
                        top = bottom;
                    }
                }
            }
        }
    }
}

/// The 16 kernel slots ([`super::h264_slot`] order: `frac_x + 4 · frac_y`)
/// of one H.264 luma shape, `H264_LUMA_SHAPES[$shape]`, from `$kernel::<
/// lanes, width, height, frac_x, frac_y>`.
// Used by the ISAs whose H.264 kernels are resolved per shape and phase
// (AVX2, SIMD128); NEON keeps its width kernels.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! h264_luma_phases {
    ($kernel:ident, $lanes:ty, $shape:literal) => {
        h264_luma_phases!(@row $kernel, $lanes, $shape, [0, 1, 2, 3])
    };
    (@row $kernel:ident, $lanes:ty, $shape:literal, [$($fy:literal),*]) => {
        [$(
            $kernel::<$lanes, { H264_LUMA_SHAPES[$shape].0 }, { H264_LUMA_SHAPES[$shape].1 }, 0, $fy>,
            $kernel::<$lanes, { H264_LUMA_SHAPES[$shape].0 }, { H264_LUMA_SHAPES[$shape].1 }, 1, $fy>,
            $kernel::<$lanes, { H264_LUMA_SHAPES[$shape].0 }, { H264_LUMA_SHAPES[$shape].1 }, 2, $fy>,
            $kernel::<$lanes, { H264_LUMA_SHAPES[$shape].0 }, { H264_LUMA_SHAPES[$shape].1 }, 3, $fy>,
        )*]
    };
}
// Used by the ISAs whose H.264 kernels are resolved per shape and phase
// (AVX2, SIMD128); NEON keeps its width kernels.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
pub(crate) use h264_luma_phases;

/// The four class kernels (`H264_CHROMA_*` order) of one H.264 chroma
/// shape, `H264_CHROMA_SHAPES[$shape]`, from `$kernel::<lanes, width,
/// height, class>`.
// Used by the ISAs whose H.264 kernels are resolved per shape and phase
// (AVX2, SIMD128); NEON keeps its width kernels.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! h264_chroma_classes {
    ($kernel:ident, $lanes:ty, $shape:literal) => {
        [
            $kernel::<
                $lanes,
                { H264_CHROMA_SHAPES[$shape].0 },
                { H264_CHROMA_SHAPES[$shape].1 },
                { H264_CHROMA_COPY },
            >,
            $kernel::<
                $lanes,
                { H264_CHROMA_SHAPES[$shape].0 },
                { H264_CHROMA_SHAPES[$shape].1 },
                { H264_CHROMA_HORIZONTAL },
            >,
            $kernel::<
                $lanes,
                { H264_CHROMA_SHAPES[$shape].0 },
                { H264_CHROMA_SHAPES[$shape].1 },
                { H264_CHROMA_VERTICAL },
            >,
            $kernel::<
                $lanes,
                { H264_CHROMA_SHAPES[$shape].0 },
                { H264_CHROMA_SHAPES[$shape].1 },
                { H264_CHROMA_BOTH },
            >,
        ]
    };
}
// Used by the ISAs whose H.264 kernels are resolved per shape and phase
// (AVX2, SIMD128); NEON keeps its width kernels.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
pub(crate) use h264_chroma_classes;
