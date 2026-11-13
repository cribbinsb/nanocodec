//! Block inter prediction (AV1 §7.11.3.4): a horizontal filter pass into an
//! intermediate array rounded by `InterRound0`, then a vertical pass rounded
//! by `InterRound1`, stepping `xStep`/`yStep` 1/1024 samples per output
//! sample (1024 unless the reference is scaled).
//!
//! The output precision depends on how the prediction is used:
//! [`Dsp::av1_predict`] gives final 8-bit samples (`InterRound1` = 11, then
//! `Clip1`), [`Dsp::av1_predict_compound`] the compound precision
//! (`InterRound1` = 7, [`INTER_POST_ROUND`](super::INTER_POST_ROUND) bits
//! above pixels) that the compound blends and masks take.

use nc_common::{PlaneView, PlaneViewMut};

use super::filter::{AV1_FILTER_TAPS, AV1_SUBPEL_POSITIONS, Av1Filter, Av1Filters};
use super::{INTER_ROUND0, INTER_ROUND1_COMPOUND, INTER_ROUND1_FINAL, clip_pixel, round2};
use crate::{BlockSize, Dsp, Filter, Isa};

/// `SCALE_SUBPEL_BITS`: positions and steps are in 1/1024 samples.
pub const AV1_SCALE_SUBPEL_BITS: u32 = 10;
/// The step of an unscaled reference: one sample.
pub const AV1_UNSCALED_STEP: u32 = 1 << AV1_SCALE_SUBPEL_BITS;
/// Bits of a 1/1024 position below the 1/16 filter phase
/// (`SCALE_EXTRA_BITS`).
const SCALE_EXTRA_BITS: u32 = AV1_SCALE_SUBPEL_BITS - AV1_SUBPEL_POSITIONS.trailing_zeros();
/// Mask of the 1/1024 fraction of a position.
const SCALE_FRACTION_MASK: u32 = AV1_UNSCALED_STEP - 1;
/// Largest step a conformant stream needs: a reference at most twice the
/// frame size (AV1 §7.11.3.3, `2 * FrameWidth >= RefUpscaledWidth`).
pub const AV1_MAX_STEP: u32 = 2 * AV1_UNSCALED_STEP;
/// Widest and tallest prediction block (`MAX_SB_SIZE`).
pub const AV1_MAX_PREDICTION_EDGE: usize = 128;

/// Where one direction of a prediction samples its window: the start
/// position's fraction of a sample (`startX & 1023`) and the step per
/// output sample, both in 1/1024 samples.
///
/// The window starts [`AV1_FILTER_BEFORE`](super::filter::AV1_FILTER_BEFORE)
/// samples before the start position's integer sample (`startX >> 10`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1McAxis {
    /// Fraction of a sample at the first output sample, below 1024.
    pub position: u32,
    /// Step per output sample: 1024 unscaled, otherwise `xStep`/`yStep`
    /// (at most [`AV1_MAX_STEP`]).
    pub step: u32,
}

impl Av1McAxis {
    /// An unscaled direction at 1/16-sample phase `frac` (below 16).
    #[must_use]
    pub const fn unscaled(frac: usize) -> Self {
        Self {
            position: (frac as u32) << SCALE_EXTRA_BITS,
            step: AV1_UNSCALED_STEP,
        }
    }

    /// Window samples this direction reads for `length` output samples:
    /// up to the last output sample's integer position plus the filter
    /// taps (`length + 7` unscaled).
    #[must_use]
    pub const fn span(self, length: usize) -> usize {
        let last = self.position as usize + self.step as usize * (length - 1);
        (last >> AV1_SCALE_SUBPEL_BITS) + AV1_FILTER_TAPS
    }

    /// The constant 1/16 phase of an unscaled direction.
    const fn unscaled_phase(self) -> Option<usize> {
        if self.step == AV1_UNSCALED_STEP {
            Some((self.position >> SCALE_EXTRA_BITS) as usize)
        } else {
            None
        }
    }

    /// The window sample (before the taps) and the 1/16 filter phase of
    /// output sample `index`.
    pub(super) fn sample(self, index: usize) -> (usize, usize) {
        let position = self.position as usize + self.step as usize * index;
        (
            position >> AV1_SCALE_SUBPEL_BITS,
            (position & SCALE_FRACTION_MASK as usize) >> SCALE_EXTRA_BITS,
        )
    }
}

/// The §7.11.3.4 filter passes over `window` for a `w`x`h` block; `store`
/// receives each output sample `(row, column, Round2(sum, round1))`.
fn block_inter_prediction(
    filters: Av1Filters,
    window: &PlaneView<'_>,
    (w, h): (usize, usize),
    (x, y): (Av1McAxis, Av1McAxis),
    round1: u32,
    mut store: impl FnMut(usize, usize, i32),
) {
    check_prediction(window, (w, h), (x, y));
    let kernel_x = filters.x.kernel(w);
    let kernel_y = filters.y.kernel(h);
    let rows = y.span(h);
    let mut intermediate = vec![0i32; rows * w];
    for (r, line) in intermediate.chunks_exact_mut(w).enumerate() {
        let source = window.row(r);
        for (c, out) in line.iter_mut().enumerate() {
            let (start, phase) = x.sample(c);
            let sum: i32 = kernel_x[phase]
                .iter()
                .zip(&source[start..start + AV1_FILTER_TAPS])
                .map(|(&t, &s)| i32::from(t) * i32::from(s))
                .sum();
            *out = round2(sum, INTER_ROUND0);
        }
    }
    for r in 0..h {
        let (start, phase) = y.sample(r);
        for c in 0..w {
            let sum: i32 = kernel_y[phase]
                .iter()
                .enumerate()
                .map(|(t, &tap)| i32::from(tap) * intermediate[(start + t) * w + c])
                .sum();
            store(r, c, round2(sum, round1));
        }
    }
}

/// [`Dsp::av1_predict`] by the scalar reference passes: shapes and
/// filter combinations the SIMD kernels do not cover (kept out of line,
/// away from the common case).
#[inline(never)]
pub(super) fn predict_scalar(
    filters: Av1Filters,
    window: &PlaneView<'_>,
    (w, h): (usize, usize),
    (x, y): (Av1McAxis, Av1McAxis),
    dst: &mut PlaneViewMut<'_>,
) {
    assert!(
        dst.width() >= w && dst.height() >= h,
        "AV1 prediction destination {}x{} is smaller than {w}x{h}",
        dst.width(),
        dst.height()
    );
    block_inter_prediction(
        filters,
        window,
        (w, h),
        (x, y),
        INTER_ROUND1_FINAL,
        |r, c, v| {
            dst.row_mut(r)[c] = clip_pixel(v);
        },
    );
}

/// [`Dsp::av1_predict_compound`] by the scalar reference passes.
#[inline(never)]
pub(super) fn predict_compound_scalar(
    filters: Av1Filters,
    window: &PlaneView<'_>,
    (w, h): (usize, usize),
    (x, y): (Av1McAxis, Av1McAxis),
    dst: &mut [i16],
    dst_stride: usize,
) {
    block_inter_prediction(
        filters,
        window,
        (w, h),
        (x, y),
        INTER_ROUND1_COMPOUND,
        |r, c, v| {
            dst[r * dst_stride + c] = v as i16;
        },
    );
}

/// Panics unless the block size, positions and window are in range.
pub(super) fn check_prediction(
    window: &PlaneView<'_>,
    (w, h): (usize, usize),
    (x, y): (Av1McAxis, Av1McAxis),
) {
    assert!(
        (1..=AV1_MAX_PREDICTION_EDGE).contains(&w) && (1..=AV1_MAX_PREDICTION_EDGE).contains(&h),
        "AV1 prediction block {w}x{h} out of range"
    );
    for axis in [x, y] {
        assert!(
            axis.position < AV1_UNSCALED_STEP && (1..=AV1_MAX_STEP).contains(&axis.step),
            "AV1 prediction position {axis:?} out of range"
        );
    }
    assert!(
        window.width() >= x.span(w) && window.height() >= y.span(h),
        "AV1 prediction window {}x{} is smaller than {}x{}",
        window.width(),
        window.height(),
        x.span(w),
        y.span(h)
    );
}

/// Largest sub-block edge of [`Dsp::predict_half_step`] (blocks of up to
/// 128 samples).
const HALF_STEP_MAX_EDGE: usize = AV1_MAX_PREDICTION_EDGE / 2;
/// The step of a reference half the frame's size: half a sample.
const HALF_STEP: u32 = AV1_UNSCALED_STEP / 2;
/// Smallest block edge whose half-size sub-blocks still use the 8-tap
/// kernels (blocks of 4 or fewer samples filter with 4 taps).
const HALF_STEP_MIN_EDGE: usize = 16;

/// The window offset and 1/16 phase of the output samples of `parity` (0
/// even, 1 odd) along a direction stepping half a sample.
fn parity_start(axis: Av1McAxis, parity: usize) -> (usize, usize) {
    let position = axis.position as usize + parity * HALF_STEP as usize;
    (
        position >> AV1_SCALE_SUBPEL_BITS,
        (position & SCALE_FRACTION_MASK as usize) >> SCALE_EXTRA_BITS,
    )
}

/// The unscaled kernel and sub-block size of a half-step prediction
/// ([`Dsp::predict_half_step`]), if this case is one.
fn half_step_case(
    filters: Av1Filters,
    (w, h): (usize, usize),
    (x, y): (Av1McAxis, Av1McAxis),
) -> Option<(Filter, BlockSize)> {
    if x.step != HALF_STEP || y.step != HALF_STEP || w.min(h) < HALF_STEP_MIN_EDGE {
        return None;
    }
    Some((family(filters)?, BlockSize::new(w / 2, h / 2)?))
}

/// The prediction kernel family of `filters`: one of the three 8-tap
/// families in both directions.
#[inline]
fn family(filters: Av1Filters) -> Option<Filter> {
    match (filters.x, filters.y) {
        (Av1Filter::Regular, Av1Filter::Regular) => Some(Filter::Av1),
        (Av1Filter::Smooth, Av1Filter::Smooth) => Some(Filter::Av1Smooth),
        (Av1Filter::Sharp, Av1Filter::Sharp) => Some(Filter::Av1Sharp),
        _ => None,
    }
}

/// The SIMD kernels an unscaled block of a kernel size runs.
#[derive(Debug, Clone, Copy)]
enum SimdKernels {
    /// One family in both directions: the shared prediction kernels
    /// ([`Dsp::predict`], [`Dsp::predict_compound`]).
    Family(Filter),
    /// Two different 8-tap families (`dual_filter`): the mixed kernels with
    /// each direction's taps ([`Dsp::av1_put_mixed`], [`Dsp::av1_prep_mixed`]).
    Mixed(Av1Filters),
}

/// One prediction kernel call: which kernels, the block size they run at
/// and the 1/16 phases.
type SimdCase = (SimdKernels, BlockSize, (usize, usize));

/// Blocks with a 128-sample edge run the 64-sample kernels on tiles of the
/// block. The narrow edge is kept: four samples select the 4-tap filters.
const KERNEL_EDGE: usize = 64;

/// Blocks with a 2-sample edge (the pieces of sub-8x8 chroma, §7.11.3.1)
/// run the 4-sample kernels on a padded copy of their window: their own
/// outputs read only samples inside it. Kept for blocks whose padded edges
/// are at most this.
const NARROW_PADDED_MAX: usize = 16;
/// The smallest kernel edge.
const KERNEL_MIN_EDGE: usize = 4;
/// Samples of a padded narrow block's window and of its prediction.
const NARROW_WINDOW: usize = (NARROW_PADDED_MAX + AV1_FILTER_TAPS - 1).pow(2);
const NARROW_BLOCK: usize = NARROW_PADDED_MAX * NARROW_PADDED_MAX;

impl Dsp {
    /// The SIMD kernels for this case, if any: unscaled, each direction one
    /// of the three 8-tap families (both the same, or different where the
    /// ISA has mixed kernels), and a block both of whose edges are 4 to 64
    /// samples (powers of two).
    #[inline]
    fn simd_case(
        &self,
        filters: Av1Filters,
        (w, h): (usize, usize),
        (x, y): (Av1McAxis, Av1McAxis),
    ) -> Option<SimdCase> {
        let kernels = match family(filters) {
            Some(filter) => SimdKernels::Family(filter),
            None if self.has_av1_mixed()
                && filters.x != Av1Filter::Bilinear
                && filters.y != Av1Filter::Bilinear =>
            {
                SimdKernels::Mixed(filters)
            }
            None => return None,
        };
        let size = BlockSize::new(w, h)?;
        Some((kernels, size, (x.unscaled_phase()?, y.unscaled_phase()?)))
    }

    /// [`Self::simd_case`] for blocks with a 128-sample edge, tiled by the
    /// 64-sample kernels ([`tiles`]); SIMD ISAs only, so the scalar ISA
    /// keeps the full-block equation as an independent oracle.
    fn wide_simd_case(
        &self,
        filters: Av1Filters,
        (w, h): (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
    ) -> Option<SimdCase> {
        if self.isa() == Isa::Scalar
            || (w != AV1_MAX_PREDICTION_EDGE && h != AV1_MAX_PREDICTION_EDGE)
        {
            return None;
        }
        let case = self.simd_case(filters, (w.min(KERNEL_EDGE), h.min(KERNEL_EDGE)), axes)?;
        (w.is_multiple_of(case.1.width()) && h.is_multiple_of(case.1.height())).then_some(case)
    }

    /// [`Self::simd_case`] for blocks with a 2-sample edge, at the 4-sample
    /// kernel size ([`NARROW_PADDED_MAX`]); SIMD ISAs only, as
    /// [`Self::wide_simd_case`].
    fn narrow_simd_case(
        &self,
        filters: Av1Filters,
        (w, h): (usize, usize),
        axes: (Av1McAxis, Av1McAxis),
    ) -> Option<SimdCase> {
        let padded = (w.max(KERNEL_MIN_EDGE), h.max(KERNEL_MIN_EDGE));
        if self.isa() == Isa::Scalar
            || w.min(h) >= KERNEL_MIN_EDGE
            || padded.0.max(padded.1) > NARROW_PADDED_MAX
        {
            return None;
        }
        self.simd_case(filters, padded, axes)
    }

    /// Run the padded `case` of a `w`x`h` block (one edge 2 samples): its
    /// window copied into a padded one (the extra samples, zero, reach only
    /// outputs that are dropped), the prediction made there, and the block's
    /// own `w`x`h` part copied out.
    fn put_narrow(
        &self,
        case: SimdCase,
        window: &PlaneView<'_>,
        (w, h): (usize, usize),
        dst: &mut PlaneViewMut<'_>,
    ) {
        let size = case.1;
        let (span_w, span_h) = (
            size.width() + AV1_FILTER_TAPS - 1,
            size.height() + AV1_FILTER_TAPS - 1,
        );
        let mut samples = [0u8; NARROW_WINDOW];
        for (r, row) in samples
            .chunks_exact_mut(span_w)
            .take(h + AV1_FILTER_TAPS - 1)
            .enumerate()
        {
            let source = window.row(r);
            row[..w + AV1_FILTER_TAPS - 1].copy_from_slice(&source[..w + AV1_FILTER_TAPS - 1]);
        }
        let padded = PlaneView::new(&samples, span_w, span_w, span_h)
            .expect("the padded window fits its buffer");
        let mut block = [0u8; NARROW_BLOCK];
        let mut prediction =
            PlaneViewMut::new(&mut block, size.width(), size.width(), size.height())
                .expect("the padded block fits its buffer");
        self.put_case(case, &padded, &mut prediction);
        for r in 0..h {
            dst.row_mut(r)[..w].copy_from_slice(&prediction.row_mut(r)[..w]);
        }
    }

    /// Run `case` on `window` into `dst` at final 8-bit precision.
    #[inline]
    fn put_case(
        &self,
        (kernels, size, fracs): SimdCase,
        window: &PlaneView<'_>,
        dst: &mut PlaneViewMut<'_>,
    ) {
        match kernels {
            SimdKernels::Family(filter) => {
                self.predict(filter, size, window, fracs.0, fracs.1, dst);
            }
            SimdKernels::Mixed(filters) => {
                self.av1_put_mixed(filters, size, window, fracs, dst);
            }
        }
    }

    /// Run `case` on `window` into `dst` (at `dst_stride`) at compound
    /// precision.
    #[inline]
    fn prep_case(
        &self,
        (kernels, size, fracs): SimdCase,
        window: &PlaneView<'_>,
        dst: &mut [i16],
        dst_stride: usize,
    ) {
        match kernels {
            SimdKernels::Family(filter) => {
                self.predict_compound(filter, size, window, fracs, dst, dst_stride);
            }
            SimdKernels::Mixed(filters) => {
                self.av1_prep_mixed(filters, size, window, fracs, dst, dst_stride);
            }
        }
    }

    /// [`Self::av1_predict`] of an unscaled block at sub-sample offsets
    /// `frac_x`, `frac_y` (1/16 sample, below 16): a decoder's common case,
    /// which goes straight to the prediction kernels when both directions
    /// use one family and the block is a kernel size (4 to 64 samples),
    /// and otherwise takes [`Self::av1_predict`]'s general path.
    ///
    /// # Panics
    ///
    /// As [`Self::av1_predict`].
    #[inline]
    pub fn av1_put(
        &self,
        filters: Av1Filters,
        window: &PlaneView<'_>,
        (w, h): (usize, usize),
        (frac_x, frac_y): (usize, usize),
        dst: &mut PlaneViewMut<'_>,
    ) {
        match (family(filters), BlockSize::new(w, h)) {
            (Some(filter), Some(size)) => self.predict(filter, size, window, frac_x, frac_y, dst),
            _ => self.av1_predict(
                filters,
                window,
                (w, h),
                (Av1McAxis::unscaled(frac_x), Av1McAxis::unscaled(frac_y)),
                dst,
            ),
        }
    }

    /// [`Self::av1_put`] on sample buffers (as [`Self::predict_samples`]):
    /// `src` from the window's top-left at `src_stride`, `dst` from the
    /// block's at `dst_stride`.
    ///
    /// # Panics
    ///
    /// As [`Self::av1_predict`], or if a buffer does not cover its window.
    #[inline]
    pub fn av1_put_samples(
        &self,
        filters: Av1Filters,
        (src, src_stride): (&[u8], usize),
        (w, h): (usize, usize),
        fracs: (usize, usize),
        (dst, dst_stride): (&mut [u8], usize),
    ) {
        if let (Some(filter), Some(size)) = (family(filters), BlockSize::new(w, h)) {
            self.predict_samples(filter, size, (src, src_stride), fracs, (dst, dst_stride));
        } else {
            self.av1_put_samples_general(
                filters,
                (src, src_stride),
                (w, h),
                fracs,
                (dst, dst_stride),
            );
        }
    }

    /// [`Self::av1_put_samples`] of the other cases, through views (kept
    /// out of line, away from the common case).
    #[cold]
    #[inline(never)]
    fn av1_put_samples_general(
        &self,
        filters: Av1Filters,
        (src, src_stride): (&[u8], usize),
        (w, h): (usize, usize),
        fracs: (usize, usize),
        (dst, dst_stride): (&mut [u8], usize),
    ) {
        let span = (w + AV1_FILTER_TAPS - 1, h + AV1_FILTER_TAPS - 1);
        let window =
            PlaneView::new(src, src_stride, span.0, span.1).expect("window covers its samples");
        let mut dst =
            PlaneViewMut::new(dst, dst_stride, w, h).expect("destination covers the block");
        self.av1_put(filters, &window, (w, h), fracs, &mut dst);
    }

    /// [`Self::av1_put_samples`] at compound precision
    /// ([`Self::av1_predict_compound`] of an unscaled block): one of a
    /// compound block's two predictions, into `dst` at `dst_stride`.
    ///
    /// # Panics
    ///
    /// As [`Self::av1_put_samples`].
    #[inline]
    pub fn av1_prep_samples(
        &self,
        filters: Av1Filters,
        (src, src_stride): (&[u8], usize),
        (w, h): (usize, usize),
        fracs: (usize, usize),
        (dst, dst_stride): (&mut [i16], usize),
    ) {
        if let (Some(filter), Some(size)) = (family(filters), BlockSize::new(w, h)) {
            self.predict_compound_samples(
                filter,
                size,
                (src, src_stride),
                fracs,
                (dst, dst_stride),
            );
        } else {
            self.av1_prep_samples_general(
                filters,
                (src, src_stride),
                (w, h),
                fracs,
                (dst, dst_stride),
            );
        }
    }

    /// [`Self::av1_prep_samples`] of the other cases, through a view (kept
    /// out of line, away from the common case).
    #[cold]
    #[inline(never)]
    fn av1_prep_samples_general(
        &self,
        filters: Av1Filters,
        (src, src_stride): (&[u8], usize),
        (w, h): (usize, usize),
        fracs: (usize, usize),
        (dst, dst_stride): (&mut [i16], usize),
    ) {
        let span = (w + AV1_FILTER_TAPS - 1, h + AV1_FILTER_TAPS - 1);
        let window =
            PlaneView::new(src, src_stride, span.0, span.1).expect("window covers its samples");
        self.av1_predict_compound(
            filters,
            &window,
            (w, h),
            (Av1McAxis::unscaled(fracs.0), Av1McAxis::unscaled(fracs.1)),
            dst,
            dst_stride,
        );
    }

    /// Predict a `w`x`h` block from `window` into `dst` at final 8-bit
    /// precision (AV1 §7.11.3.4 with `InterRound1` = 11, then `Clip1`):
    /// the prediction of a single-reference block.
    ///
    /// `window` starts 3 samples up and left of the block's integer
    /// reference position and covers [`Av1McAxis::span`] samples in each
    /// direction. Unscaled blocks of 4 to 128 samples whose directions use
    /// 8-tap families run the SIMD prediction kernels (128 edges tile the
    /// 64 kernels); everything else uses the scaled kernels.
    ///
    /// # Panics
    ///
    /// If a block edge is not 1 to 128, a position or step is out of range,
    /// the window is too small or `dst` does not cover the block.
    #[inline]
    pub fn av1_predict(
        &self,
        filters: Av1Filters,
        window: &PlaneView<'_>,
        (w, h): (usize, usize),
        (x, y): (Av1McAxis, Av1McAxis),
        dst: &mut PlaneViewMut<'_>,
    ) {
        if let Some(case) = self.simd_case(filters, (w, h), (x, y)) {
            self.put_case(case, window, dst);
        } else if let Some(case) = self.wide_simd_case(filters, (w, h), (x, y)) {
            check_prediction(window, (w, h), (x, y));
            assert!(
                dst.width() >= w && dst.height() >= h,
                "AV1 prediction destination too small"
            );
            for (left, top) in tiles((w, h), case.1) {
                let source = tile_window(window, (left, top), case.1);
                let mut target = dst
                    .sub_view_mut(left, top, case.1.width(), case.1.height())
                    .expect("checked output covers every tile");
                self.put_case(case, &source, &mut target);
            }
        } else if let Some(case) = self.narrow_simd_case(filters, (w, h), (x, y)) {
            check_prediction(window, (w, h), (x, y));
            assert!(
                dst.width() >= w && dst.height() >= h,
                "AV1 prediction destination too small"
            );
            self.put_narrow(case, window, (w, h), dst);
        } else if let Some((filter, size)) = half_step_case(filters, (w, h), (x, y)) {
            self.predict_half_step(filter, size, window, (x, y), dst);
        } else {
            self.av1_predict_scaled(filters, window, (w, h), (x, y), dst);
        }
    }

    /// [`Self::av1_predict`] from a reference half the frame's size in
    /// both directions (steps of half a sample: the first frame after an
    /// adaptive-resolution switch up). Each output sample of one of the
    /// four parities (even/odd column, even/odd row) sits a whole number of
    /// samples from the next of its parity at one fixed 1/16 phase, so the
    /// block is four unscaled `size` predictions, interleaved. Exact: the
    /// same taps and roundings as the scaled passes, as the sub-blocks keep
    /// the 8-tap kernels (`half_step_case` requires edges of 16 or more).
    fn predict_half_step(
        &self,
        filter: Filter,
        size: BlockSize,
        window: &PlaneView<'_>,
        (x, y): (Av1McAxis, Av1McAxis),
        dst: &mut PlaneViewMut<'_>,
    ) {
        // `size` is a sub-block: half the block.
        let (w, h) = (size.width(), size.height());
        let mut samples = [0u8; HALF_STEP_MAX_EDGE * HALF_STEP_MAX_EDGE];
        for parity_y in 0..2 {
            let (top, frac_y) = parity_start(y, parity_y);
            for parity_x in 0..2 {
                let (left, frac_x) = parity_start(x, parity_x);
                let sub = window
                    .sub_view(left, top, w + AV1_FILTER_TAPS - 1, h + AV1_FILTER_TAPS - 1)
                    .expect("the scaled window covers every parity's window");
                let mut out = PlaneViewMut::new(&mut samples[..w * h], w, w, h)
                    .expect("the scratch covers a sub-block");
                self.predict(filter, size, &sub, frac_x, frac_y, &mut out);
                for (r, row) in samples[..w * h].chunks_exact(w).enumerate() {
                    let line = &mut dst.row_mut(2 * r + parity_y)[parity_x..];
                    for (c, &v) in row.iter().enumerate() {
                        line[2 * c] = v;
                    }
                }
            }
        }
    }

    /// Predict a `w`x`h` block from `window` at compound precision (AV1
    /// §7.11.3.4 with `InterRound1` = 7) into `dst` (`h` rows of `w` at
    /// `dst_stride`): one of the two predictions a compound blend or mask
    /// combines. `window` as for [`Self::av1_predict`], and the same blocks
    /// run the SIMD kernels (at compound precision).
    ///
    /// # Panics
    ///
    /// As [`Self::av1_predict`], or if `dst` is too short.
    pub fn av1_predict_compound(
        &self,
        filters: Av1Filters,
        window: &PlaneView<'_>,
        (w, h): (usize, usize),
        (x, y): (Av1McAxis, Av1McAxis),
        dst: &mut [i16],
        dst_stride: usize,
    ) {
        assert!(
            dst_stride >= w && dst.len() >= (h - 1) * dst_stride + w,
            "AV1 compound prediction buffer too small for {w}x{h}"
        );
        if let Some(case) = self.simd_case(filters, (w, h), (x, y)) {
            self.prep_case(case, window, dst, dst_stride);
            return;
        }
        if let Some(case) = self.wide_simd_case(filters, (w, h), (x, y)) {
            check_prediction(window, (w, h), (x, y));
            for (left, top) in tiles((w, h), case.1) {
                let source = tile_window(window, (left, top), case.1);
                self.prep_case(
                    case,
                    &source,
                    &mut dst[top * dst_stride + left..],
                    dst_stride,
                );
            }
            return;
        }
        self.av1_predict_compound_scaled(filters, window, (w, h), (x, y), dst, dst_stride);
    }
}

/// The top-left corners of the `size` tiles covering a `w`x`h` block.
fn tiles((w, h): (usize, usize), size: BlockSize) -> impl Iterator<Item = (usize, usize)> {
    (0..h)
        .step_by(size.height())
        .flat_map(move |top| (0..w).step_by(size.width()).map(move |left| (left, top)))
}

/// The window of the `size` tile at (`left`, `top`) of a block's window.
fn tile_window<'w>(
    window: &PlaneView<'w>,
    (left, top): (usize, usize),
    size: BlockSize,
) -> PlaneView<'w> {
    window
        .sub_view(
            left,
            top,
            size.width() + AV1_FILTER_TAPS - 1,
            size.height() + AV1_FILTER_TAPS - 1,
        )
        .expect("checked full window covers every tile")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Half-sample steps (a reference half the frame's size) through the
    /// interleaved unscaled kernels equal the scaled reference passes, at
    /// every start phase, for every family and the block sizes that take
    /// the shortcut (and those that do not).
    #[test]
    fn half_steps_match_the_reference_passes() {
        let dsp = Dsp::new();
        for (w, h) in [(16, 16), (32, 16), (16, 64), (128, 128), (8, 16), (4, 4)] {
            for position in [0, 64, 448, 512, 576, 1023] {
                let x = Av1McAxis {
                    position,
                    step: HALF_STEP,
                };
                let y = Av1McAxis {
                    position: (position * 7) % 1024,
                    step: HALF_STEP,
                };
                let (span_x, span_y) = (x.span(w), y.span(h));
                let samples: Vec<u8> = (0..span_x * span_y)
                    .map(|i| ((i * 37 + i / span_x * 11) % 251) as u8)
                    .collect();
                let window = PlaneView::new(&samples, span_x, span_x, span_y).unwrap();
                for filter in [Av1Filter::Regular, Av1Filter::Smooth, Av1Filter::Sharp] {
                    let filters = Av1Filters::both(filter);
                    let mut fast = vec![0u8; w * h];
                    let mut view = PlaneViewMut::new(&mut fast, w, w, h).unwrap();
                    dsp.av1_predict(filters, &window, (w, h), (x, y), &mut view);
                    let mut reference = vec![0u8; w * h];
                    block_inter_prediction(
                        filters,
                        &window,
                        (w, h),
                        (x, y),
                        INTER_ROUND1_FINAL,
                        |r, c, v| {
                            reference[r * w + c] = clip_pixel(v);
                        },
                    );
                    assert_eq!(fast, reference, "{w}x{h} position {position} {filter:?}");
                }
            }
        }
    }

    /// The SIMD shortcut equals the scalar §7.11.3.4 passes it replaces.
    #[test]
    fn simd_shortcut_matches_the_reference_passes() {
        let dsp = Dsp::new();
        let (w, h) = (16, 8);
        let stride = w + AV1_FILTER_TAPS;
        let samples: Vec<u8> = (0..stride * (h + AV1_FILTER_TAPS))
            .map(|i| (i * 37 % 251) as u8)
            .collect();
        let window = PlaneView::new(&samples, stride, stride, h + AV1_FILTER_TAPS).unwrap();
        for filter in [Av1Filter::Regular, Av1Filter::Smooth, Av1Filter::Sharp] {
            for (fx, fy) in [(0, 0), (5, 0), (0, 9), (3, 15)] {
                let axes = (Av1McAxis::unscaled(fx), Av1McAxis::unscaled(fy));
                let mut fast = vec![0u8; w * h];
                let mut view = PlaneViewMut::new(&mut fast, w, w, h).unwrap();
                dsp.av1_predict(Av1Filters::both(filter), &window, (w, h), axes, &mut view);
                let mut reference = vec![0u8; w * h];
                block_inter_prediction(
                    Av1Filters::both(filter),
                    &window,
                    (w, h),
                    axes,
                    INTER_ROUND1_FINAL,
                    |r, c, v| {
                        reference[r * w + c] = clip_pixel(v);
                    },
                );
                assert_eq!(
                    fast,
                    reference,
                    "{filter:?} phase ({fx}, {fy}) on {:?}",
                    dsp.isa()
                );
            }
        }
    }
}
