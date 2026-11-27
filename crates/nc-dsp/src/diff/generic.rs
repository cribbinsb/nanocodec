//! Residual and variance kernels generic over [`I16Lanes`]: `V::LANES`
//! samples of a row per step.
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs).

#![allow(clippy::inline_always)]

use core::ops::Range;

use super::{Prediction, ResidualRows, Variance, VarianceEnergy, bilinear_taps};
use crate::simd::I16Lanes;

/// Most column chunks of any block: 64 samples in vectors of at least 8
/// lanes (4-lane vectors only serve 4-wide blocks).
const MAX_CHUNKS: usize = 8;

/// `src - pred` into `out`, over a `W`-wide block of `h` rows (width a
/// constant and the buffers checked once, as for [`variance`]).
///
/// # Safety
///
/// `V`'s ISA must be available.
///
/// # Panics
///
/// If a buffer does not hold `h` rows of `W` values at its stride.
#[inline(always)]
pub(crate) unsafe fn residual<V: I16Lanes, const W: usize, const SUM_ABS: bool>(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [i16],
    out_stride: usize,
    h: usize,
) -> u32 {
    assert!(
        covers(src.len(), src_stride, h, W)
            && covers(pred.len(), pred_stride, h, W)
            && covers(out.len(), out_stride, h, W)
    );
    // |src - pred| sums stay in i16 lanes for `span` rows and widen once per
    // span instead of once per vector: one span covers every block up to
    // 32 wide (`rows_per_widen`), so only wider blocks loop over spans.
    let span = rows_per_widen(W, V::LANES);
    if !SUM_ABS || span >= MAX_BLOCK_HEIGHT {
        // SAFETY: the assertion above covers rows `0..h` of `W` values.
        let narrow = unsafe {
            residual_rows::<V, W, SUM_ABS>(
                src,
                src_stride,
                pred,
                pred_stride,
                out,
                out_stride,
                0..h,
            )
        };
        return if SUM_ABS {
            V::acc_sum(V::acc_add(V::acc_zero(), narrow)) as u32
        } else {
            0
        };
    }
    let mut sum = V::acc_zero();
    for top in (0..h).step_by(span) {
        // SAFETY: as above.
        let narrow = unsafe {
            residual_rows::<V, W, SUM_ABS>(
                src,
                src_stride,
                pred,
                pred_stride,
                out,
                out_stride,
                top..h.min(top + span),
            )
        };
        sum = V::acc_add(sum, narrow);
    }
    V::acc_sum(sum) as u32
}

/// Largest block height (the kernels' contract).
const MAX_BLOCK_HEIGHT: usize = 64;

/// [`residual`] over `rows`, returning the rows' |src - pred| sums in i16
/// lanes (zero unless `SUM_ABS`).
///
/// # Safety
///
/// `V`'s ISA must be available and the buffers must hold `rows` of `W`
/// values at their strides.
#[inline(always)]
unsafe fn residual_rows<V: I16Lanes, const W: usize, const SUM_ABS: bool>(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [i16],
    out_stride: usize,
    rows: core::ops::Range<usize>,
) -> V {
    let mut narrow = V::splat(0);
    for r in rows {
        // SAFETY: guaranteed by the caller.
        let (source, prediction, difference) = unsafe {
            (
                row::<W>(src, src_stride, r),
                row::<W>(pred, pred_stride, r),
                row_mut_i16::<W>(out, out_stride, r),
            )
        };
        for c0 in (0..W).step_by(V::LANES) {
            let delta = V::load_u8(&source[c0..]).sub(V::load_u8(&prediction[c0..]));
            delta.store(&mut difference[c0..]);
            if SUM_ABS {
                narrow = narrow.add(delta.abs());
            }
        }
    }
    narrow
}

/// Rows of a `width`-wide block whose sums of values of magnitude at most
/// 255 (8-bit samples, their differences and absolute differences) fit an
/// i16 lane: a row adds at most 255 per vector, `width.div_ceil(lanes)`
/// vectors, so 128 rows for one vector per row, 16 for 64 samples in
/// 8-lane vectors (16 * 8 * 255 = 32640 <= 32767), 32 in 16-lane vectors.
const fn rows_per_widen(width: usize, lanes: usize) -> usize {
    i16::MAX as usize / (width.div_ceil(lanes) * u8::MAX as usize)
}

/// Rows `0..h` of a `W`-wide block in spans of [`rows_per_widen`]: per span
/// the sums stay in i16 lanes (one wrapping add per vector, exact by the
/// bound) and widen once into the 32-bit accumulator, as libaom's AVX2
/// variance keeps its sum in 16 bits. Blocks up to 32 wide take one span.
#[inline(always)]
fn widen_spans<V: I16Lanes, const W: usize>(h: usize) -> WidenSpans {
    WidenSpans {
        top: 0,
        h,
        span: rows_per_widen(W, V::LANES),
    }
}

/// The iterator of [`widen_spans`]. Its span is a constant of each kernel,
/// so where it covers the tallest block the whole loop folds to `0..h`
/// (with `step_by` the span bookkeeping cost about 40 instructions per
/// call, more than it saved on a 16x16 block on x86).
struct WidenSpans {
    top: usize,
    h: usize,
    span: usize,
}

impl Iterator for WidenSpans {
    type Item = Range<usize>;

    #[inline(always)]
    fn next(&mut self) -> Option<Range<usize>> {
        if self.top >= self.h {
            return None;
        }
        let end = if self.span >= MAX_BLOCK_HEIGHT {
            self.h
        } else {
            self.h.min(self.top + self.span)
        };
        let rows = self.top..end;
        self.top = end;
        Some(rows)
    }
}

/// `dst = clamp(prediction + residual, 0, 255)` over a `W`-wide block of
/// `h` rows (`ReconstructFn`): the prediction is zero-extended to 16 bits,
/// the residual added with saturation, and the store saturates to bytes.
/// An in-place prediction reads each vector of `dst` before writing it.
///
/// # Safety
///
/// `V`'s ISA must be available.
///
/// # Panics
///
/// If a buffer does not hold `h` rows of `W` values at its stride.
#[inline(always)]
pub(crate) unsafe fn reconstruct<V: I16Lanes, const W: usize>(
    prediction: Prediction<'_>,
    residual: ResidualRows<'_>,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
) {
    assert!(covers(dst.len(), dst_stride, h, W));
    let out = dst.as_mut_ptr();
    let (from, from_stride) = match prediction {
        Prediction::InPlace => (out.cast_const(), dst_stride),
        Prediction::Block(block, stride) => {
            assert!(covers(block.len(), stride, h, W));
            (block.as_ptr(), stride)
        }
    };
    match residual {
        ResidualRows::Samples(samples, stride) => {
            assert!(covers(samples.len(), stride, h, W));
            for r in 0..h {
                // SAFETY: the assertion above covers row `r` of `W` values.
                let row = unsafe { row_i16::<W>(samples, stride, r) };
                for c0 in (0..W).step_by(V::LANES) {
                    // SAFETY: see `add_to_prediction`.
                    unsafe {
                        add_to_prediction(
                            from.add(r * from_stride + c0),
                            V::load(&row[c0..]),
                            out.add(r * dst_stride + c0),
                        );
                    };
                }
            }
        }
        ResidualRows::Constant(dc) => {
            let dc = V::splat(dc);
            for r in 0..h {
                for c0 in (0..W).step_by(V::LANES) {
                    // SAFETY: see `add_to_prediction`.
                    unsafe {
                        add_to_prediction(
                            from.add(r * from_stride + c0),
                            dc,
                            out.add(r * dst_stride + c0),
                        );
                    };
                }
            }
        }
    }
}

/// `V::LANES` reconstructed samples (an `#[inline(always)]` function, not
/// a closure, so it inlines into the kernel with its target features).
///
/// # Safety
///
/// The assertions of [`reconstruct`] cover `V::LANES` samples at `from`
/// and `to` (the same samples for an in-place prediction, read before the
/// store); `V`'s ISA is available.
#[inline(always)]
unsafe fn add_to_prediction<V: I16Lanes>(from: *const u8, residual: V, to: *mut u8) {
    // SAFETY: forwarded from the caller.
    unsafe { V::load_u8_at(from).add_sat(residual).store_u8_sat_at(to) };
}

/// Variance of `src - pred` over a `W`-wide block of `h` rows.
///
/// The width is a constant so that, once the rows are known to lie inside
/// the slices (checked once, up front), every per-lane load is provably in
/// bounds and the loop carries no per-row checks.
///
/// # Safety
///
/// `V`'s ISA must be available.
///
/// # Panics
///
/// If `src` or `pred` does not hold `h` rows of `W` samples at its stride.
#[inline(always)]
pub(crate) unsafe fn variance<V: I16Lanes, const W: usize>(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    h: usize,
) -> Variance {
    assert!(covers(src.len(), src_stride, h, W) && covers(pred.len(), pred_stride, h, W));
    let (mut sum, mut sse) = (V::acc_zero(), V::acc_zero());
    for rows in widen_spans::<V, W>(h) {
        let mut narrow = V::splat(0);
        for r in rows {
            // SAFETY: the assertion above covers rows `0..h` of `W` samples.
            let (source, prediction) =
                unsafe { (row::<W>(src, src_stride, r), row::<W>(pred, pred_stride, r)) };
            for c0 in (0..W).step_by(V::LANES) {
                let difference = V::load_u8(&source[c0..]).sub(V::load_u8(&prediction[c0..]));
                narrow = narrow.add(difference);
                sse = V::acc_add_sq(sse, difference);
            }
        }
        sum = V::acc_add(sum, narrow);
    }
    Variance {
        sse: V::acc_sum(sse) as u32,
        sum: V::acc_sum(sum) as i32,
    }
}

/// The sums of `src`'s own samples: [`variance`] against zero without the
/// second load and the subtraction.
///
/// # Safety
///
/// `V`'s ISA must be available.
///
/// # Panics
///
/// If `src` does not hold `h` rows of `W` samples at its stride.
#[inline(always)]
pub(crate) unsafe fn block_sums<V: I16Lanes, const W: usize>(
    src: &[u8],
    src_stride: usize,
    h: usize,
) -> Variance {
    assert!(covers(src.len(), src_stride, h, W));
    let (mut sum, mut sse) = (V::acc_zero(), V::acc_zero());
    for rows in widen_spans::<V, W>(h) {
        let mut narrow = V::splat(0);
        for r in rows {
            // SAFETY: the assertion above covers rows `0..h` of `W` samples.
            let source = unsafe { row::<W>(src, src_stride, r) };
            for c0 in (0..W).step_by(V::LANES) {
                let x = V::load_u8(&source[c0..]);
                narrow = narrow.add(x);
                sse = V::acc_add_sq(sse, x);
            }
        }
        sum = V::acc_add(sum, narrow);
    }
    Variance {
        sse: V::acc_sum(sse) as u32,
        sum: V::acc_sum(sum) as i32,
    }
}

/// [`variance`] of `src - pred` and the sums of `pred` itself, in one pass.
///
/// # Safety
///
/// `V`'s ISA must be available.
///
/// # Panics
///
/// If either buffer does not hold `h` rows of `W` samples at its stride.
#[inline(always)]
pub(crate) unsafe fn variance_energy<V: I16Lanes, const W: usize>(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    h: usize,
) -> VarianceEnergy {
    assert!(covers(src.len(), src_stride, h, W) && covers(pred.len(), pred_stride, h, W));
    let (mut sum, mut sse) = (V::acc_zero(), V::acc_zero());
    let (mut pred_sum, mut pred_sq) = (V::acc_zero(), V::acc_zero());
    for rows in widen_spans::<V, W>(h) {
        let (mut narrow, mut pred_narrow) = (V::splat(0), V::splat(0));
        for r in rows {
            // SAFETY: the assertion above covers rows `0..h` of `W` samples.
            let (source, prediction) =
                unsafe { (row::<W>(src, src_stride, r), row::<W>(pred, pred_stride, r)) };
            for c0 in (0..W).step_by(V::LANES) {
                let p = V::load_u8(&prediction[c0..]);
                let difference = V::load_u8(&source[c0..]).sub(p);
                narrow = narrow.add(difference);
                sse = V::acc_add_sq(sse, difference);
                pred_narrow = pred_narrow.add(p);
                pred_sq = V::acc_add_sq(pred_sq, p);
            }
        }
        sum = V::acc_add(sum, narrow);
        pred_sum = V::acc_add(pred_sum, pred_narrow);
    }
    VarianceEnergy {
        residual: Variance {
            sse: V::acc_sum(sse) as u32,
            sum: V::acc_sum(sum) as i32,
        },
        prediction: Variance {
            sse: V::acc_sum(pred_sq) as u32,
            sum: V::acc_sum(pred_sum) as i32,
        },
    }
}

/// Whether a strided buffer of `len` samples holds `rows` rows of `width`.
#[inline(always)]
pub(super) fn covers(len: usize, stride: usize, rows: usize, width: usize) -> bool {
    rows == 0
        || (rows - 1)
            .checked_mul(stride)
            .and_then(|o| o.checked_add(width))
            .is_some_and(|end| end <= len)
}

/// Row `r` of a strided buffer, `N` samples long, as a fixed-size slice.
///
/// # Safety
///
/// `r * stride + N <= data.len()`.
#[inline(always)]
unsafe fn row<const N: usize>(data: &[u8], stride: usize, r: usize) -> &[u8; N] {
    debug_assert!(r * stride + N <= data.len());
    // SAFETY: the caller guarantees the row lies inside `data`.
    unsafe { &*data.as_ptr().add(r * stride).cast::<[u8; N]>() }
}

/// `i16` row `r` of a strided buffer, `N` values long.
///
/// # Safety
///
/// `r * stride + N <= data.len()`.
#[inline(always)]
unsafe fn row_i16<const N: usize>(data: &[i16], stride: usize, r: usize) -> &[i16; N] {
    debug_assert!(r * stride + N <= data.len());
    // SAFETY: the caller guarantees the row lies inside `data`.
    unsafe { &*data.as_ptr().add(r * stride).cast::<[i16; N]>() }
}

/// Writable `i16` row `r` of a strided buffer, `N` values long.
///
/// # Safety
///
/// `r * stride + N <= data.len()`.
#[inline(always)]
unsafe fn row_mut_i16<const N: usize>(data: &mut [i16], stride: usize, r: usize) -> &mut [i16; N] {
    debug_assert!(r * stride + N <= data.len());
    // SAFETY: the caller guarantees the row lies inside `data`.
    unsafe { &mut *data.as_mut_ptr().add(r * stride).cast::<[i16; N]>() }
}

/// Row `r` of a strided buffer, `n` samples long (`n` a constant after
/// inlining, e.g. `W + 1`).
///
/// # Safety
///
/// `r * stride + n <= data.len()`.
#[inline(always)]
unsafe fn row_n(data: &[u8], stride: usize, r: usize, n: usize) -> &[u8] {
    debug_assert!(r * stride + n <= data.len());
    // SAFETY: the caller guarantees the row lies inside `data`.
    unsafe { core::slice::from_raw_parts(data.as_ptr().add(r * stride), n) }
}

/// Variance of `src` against libaom's two-pass bilinear prediction from
/// `rf`. Each column of `V::LANES` samples carries its previous first-pass
/// row in a register, so the two passes need no scratch buffer.
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(crate) unsafe fn sub_pixel_variance<V: I16Lanes, const W: usize>(
    rf: &[u8],
    ref_stride: usize,
    x_offset: usize,
    y_offset: usize,
    src: &[u8],
    src_stride: usize,
    h: usize,
) -> Variance {
    // The reference window is (W + 1) x (h + 1) samples; checked once here
    // so the row loop needs no per-row bounds checks.
    assert!(covers(rf.len(), ref_stride, h + 1, W + 1) && covers(src.len(), src_stride, h, W));
    // A zero offset's taps are (128, 0), so its pass returns its input
    // exactly: skipping it gives the same result. The fixed-stage sub-sample
    // search mostly tries positions with one zero offset.
    let block = Block {
        rf,
        ref_stride,
        src,
        src_stride,
        h,
    };
    // SAFETY: the assertion above covers the windows every pass reads.
    unsafe {
        match (x_offset, y_offset) {
            (_, 0) => block.variance::<V, W, true, false>(x_offset, y_offset),
            (0, _) => block.variance::<V, W, false, true>(x_offset, y_offset),
            _ => block.variance::<V, W, true, true>(x_offset, y_offset),
        }
    }
}

/// The reference window and source block of one sub-pixel variance.
struct Block<'a> {
    rf: &'a [u8],
    ref_stride: usize,
    src: &'a [u8],
    src_stride: usize,
    h: usize,
}

impl Block<'_> {
    /// [`sub_pixel_variance`] of a `W`-wide block with the horizontal and
    /// vertical passes enabled or skipped (`H`, `V_PASS`); a skipped pass
    /// must have offset 0.
    ///
    /// # Safety
    ///
    /// `rf` holds `h + 1` rows of `W + 1` samples at `ref_stride` and `src`
    /// `h` rows of `W` at `src_stride`.
    #[inline(always)]
    unsafe fn variance<V: I16Lanes, const W: usize, const H: bool, const V_PASS: bool>(
        &self,
        x_offset: usize,
        y_offset: usize,
    ) -> Variance {
        let h = self.h;
        let fx = q15_weight::<V>(bilinear_taps(x_offset).1);
        let fy1_q15 = q15_weight::<V>(bilinear_taps(y_offset).1);
        let chunks = W / V::LANES;
        // First-pass output of the previous row, per column chunk.
        let mut upper = [V::splat(0); MAX_CHUNKS];
        if V_PASS {
            // SAFETY: the caller guarantees the window (row 0 of W + 1).
            let top = unsafe { row_n(self.rf, self.ref_stride, 0, W + 1) };
            for (i, row) in upper.iter_mut().enumerate().take(chunks) {
                *row = first_pass::<V, H>(top, i * V::LANES, fx);
            }
        }
        let (mut sum, mut sse) = (V::acc_zero(), V::acc_zero());
        for rows in widen_spans::<V, W>(h) {
            let mut narrow = V::splat(0);
            for r in rows {
                let next_row = if V_PASS { r + 1 } else { r };
                // SAFETY: rows up to `h` of W + 1 (reference) and `h - 1` of
                // W (source) are inside the windows, as the caller guarantees.
                let (next, source) = unsafe {
                    (
                        row_n(self.rf, self.ref_stride, next_row, W + 1),
                        row::<W>(self.src, self.src_stride, r),
                    )
                };
                for (i, previous) in upper.iter_mut().enumerate().take(chunks) {
                    let c0 = i * V::LANES;
                    let lower = first_pass::<V, H>(next, c0, fx);
                    let pred = if V_PASS {
                        bilinear_blend(*previous, lower, fy1_q15)
                    } else {
                        lower
                    };
                    // A prediction of 8-bit samples, so |diff| <= 255.
                    let diff = V::load_u8(&source[c0..]).sub(pred);
                    narrow = narrow.add(diff);
                    sse = V::acc_add_sq(sse, diff);
                    *previous = lower;
                }
            }
            sum = V::acc_add(sum, narrow);
        }
        Variance {
            sse: V::acc_sum(sse) as u32,
            sum: V::acc_sum(sum) as i32,
        }
    }
}

/// The first (horizontal) pass, or the samples themselves when it is
/// skipped (an `#[inline(always)]` function, not a closure: closures do not
/// inherit the kernel's target features).
#[inline(always)]
fn first_pass<V: I16Lanes, const H: bool>(row: &[u8], c0: usize, fx: V) -> V {
    if H {
        bilinear_blend(V::load_u8(&row[c0..]), V::load_u8(&row[c0 + 1..]), fx)
    } else {
        V::load_u8(&row[c0..])
    }
}

/// The second tap `f1` of a bilinear pair `(128 - f1, f1)` as a Q15 weight
/// for [`bilinear_blend`].
#[inline(always)]
fn q15_weight<V: I16Lanes>(f1: i16) -> V {
    // f1 <= 112, so f1 << 8 <= 28672 fits an i16.
    V::splat(f1 << 8)
}

/// libaom's bilinear tap `(a * (128 - f1) + b * f1 + 64) >> 7`, computed as
/// `a + (((b - a) * f1 + 64) >> 7)`: since `128a` is a multiple of 128 the
/// two are equal, and with `w = f1 << 8` the rounding multiply-high
/// `((b - a) * w + 2^14) >> 15` is exactly `((b - a) * f1 + 64) >> 7`.
/// `|b - a| <= 255` and `w <= 28672`, so no lane saturates. Three
/// instructions instead of five.
#[inline(always)]
fn bilinear_blend<V: I16Lanes>(a: V, b: V, w: V) -> V {
    a.add(b.sub(a).mul_q15_round(w))
}

/// Total squared error. Accumulate across a row before reducing, since
/// callers of this entry need only the plane sum.
///
/// # Safety
///
/// The planes have the same dimensions and the CPU supports `V`.
#[inline(always)]
pub(crate) unsafe fn plane_sse<V: I16Lanes>(
    a: &nc_common::PlaneView<'_>,
    b: &nc_common::PlaneView<'_>,
) -> u64 {
    // Even the complete sum of this many largest differences is within i32.
    const MAX_SAMPLES: usize = 32 * 1024;
    let mut sum = 0;
    for (a, b) in a.rows().zip(b.rows()) {
        for (a, b) in a.chunks(MAX_SAMPLES).zip(b.chunks(MAX_SAMPLES)) {
            let pairs = a.len() / (2 * V::LANES) * (2 * V::LANES);
            let mut lo = V::acc_zero();
            let mut hi = V::acc_zero();
            for x in (0..pairs).step_by(2 * V::LANES) {
                // SAFETY: the loop covers complete pairs of vectors in both rows.
                let (a0, b0, a1, b1) = unsafe {
                    (
                        V::load_u8_at(a.as_ptr().add(x)),
                        V::load_u8_at(b.as_ptr().add(x)),
                        V::load_u8_at(a.as_ptr().add(x + V::LANES)),
                        V::load_u8_at(b.as_ptr().add(x + V::LANES)),
                    )
                };
                lo = V::acc_add_sq(lo, a0.sub(b0));
                hi = V::acc_add_sq(hi, a1.sub(b1));
            }
            let mut tail = pairs;
            if a.len() - tail >= V::LANES {
                let difference = V::load_u8(&a[tail..]).sub(V::load_u8(&b[tail..]));
                lo = V::acc_add_sq(lo, difference);
                tail += V::LANES;
            }
            sum += (V::acc_sum(lo) + V::acc_sum(hi)) as u64;
            sum += a[tail..]
                .iter()
                .zip(&b[tail..])
                .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
                .sum::<u64>();
        }
    }
    sum
}
