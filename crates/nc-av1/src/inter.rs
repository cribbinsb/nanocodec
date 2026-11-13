//! AV1 inter prediction for one single-reference block: unscaled ([`predict`],
//! AV1 §7.11.3.3 motion vector scaling with scale factor 1) or scaled
//! ([`predict_scaled`], §7.11.3.3 arbitrary reference scaling), followed by
//! §7.11.3.4 block inter prediction.
//!
//! The frame header fixes `interpolation_filter = EIGHTTAP` (regular); the
//! filtering itself is nc-dsp's `Filter::Av1` kernel (8-tap, or the 4-tap
//! variant for 4-sample dimensions; 8-bit non-compound rounding). This
//! module positions its reference window. Reference samples outside the
//! picture repeat the nearest edge sample (the spec's `Clip3` of reference
//! coordinates): reconstructions carry edge-extended borders, and vectors
//! that reach past them use a clamped copy of the window.
//!
//! A reference of another size than the frame (superres frames, whose
//! references keep the upscaled width, and frame size switches) goes
//! through [`predict_scaled`]: §7.11.3.3 scaled positions and steps.

use nc_common::{Plane, PlaneView, PlaneViewMut};
use nc_dsp::av1_mc::filter::{Av1Filter, Av1Filters};
use nc_dsp::av1_mc::scale::Av1ScaleSizes;
use nc_dsp::{BlockSize, Dsp, Filter};

use crate::modeinfo::Av1Mv;
use crate::scaled_reference::scaled_window;

/// Sub-sample positions per sample (`1 << SUBPEL_BITS`).
const SUBPEL_SHIFTS: i32 = 16;

/// Predict the `size`x`size` block at (`x`, `y`) of `reference` (a plane of
/// the previous reconstruction, borders extended), displaced by `mv`, into
/// `dst`. `subsampled` is true for the 4:2:0 chroma planes, whose vector is
/// the luma vector at half the sample density.
pub fn predict(
    dsp: &Dsp,
    reference: &Plane,
    x: usize,
    y: usize,
    size: usize,
    mv: Av1Mv,
    subsampled: bool,
    dst: &mut PlaneViewMut<'_>,
) {
    // Position in 1/16 samples of this plane (`(2 * mv) >> subX`).
    let shift = u32::from(subsampled);
    let pos_x = ((x as i32) << 4) + ((2 * i32::from(mv.col)) >> shift);
    let pos_y = ((y as i32) << 4) + ((2 * i32::from(mv.row)) >> shift);
    let frac_x = (pos_x & (SUBPEL_SHIFTS - 1)) as usize;
    let frac_y = (pos_y & (SUBPEL_SHIFTS - 1)) as usize;
    let before = Filter::Av1.before() as isize;
    let span = size + Filter::Av1.taps() - 1;
    let x0 = (pos_x >> 4) as isize - before;
    let y0 = (pos_y >> 4) as isize - before;
    let block = BlockSize::new(size, size).expect("square AV1 block size");

    if let Some(window) = reference.region(x0, y0, span, span) {
        dsp.predict(Filter::Av1, block, &window, frac_x, frac_y, dst);
        return;
    }
    // The window leaves the extended border: gather it with clamped
    // coordinates (as decoders do).
    let view = reference.view();
    let (last_x, last_y) = (view.width() as isize - 1, view.height() as isize - 1);
    let mut gathered = vec![0u8; span * span];
    for (r, row) in gathered.chunks_exact_mut(span).enumerate() {
        let src = view.row((y0 + r as isize).clamp(0, last_y) as usize);
        for (c, v) in row.iter_mut().enumerate() {
            *v = src[(x0 + c as isize).clamp(0, last_x) as usize];
        }
    }
    let window = PlaneView::new(&gathered, span, span, span).expect("window covers its samples");
    dsp.predict(Filter::Av1, block, &window, frac_x, frac_y, dst);
}

/// Small stack buffer (`1 KiB`) for boundary-clamped scaled windows up to
/// `32x32` samples (covering `16x16` blocks at any `S0 -> S1` upscale ratio).
const SMALL_STACK_SPAN: usize = 32 * 32;
/// Maximum stack buffer (`16 KiB`) for boundary-clamped scaled windows up to
/// `128x128` samples (covering every `S0 -> S1` upscale window up to `64x64`
/// blocks — `span <= 71`, `total <= 5041` — and `64x64` downscales up to
/// `step <= 1792`, `1.75x`, without heap allocation). Larger downscale windows
/// fall back to `heap`.
const MAX_STACK_SPAN: usize = 128 * 128;

/// [`predict`] from a reference scaled by `sizes` (luma sizes: the
/// reference's upscaled size and the frame's coded size; AV1 §7.11.3.3).
/// `reference_size` is the size of this plane of the reference: positions
/// are clamped to it, not to the coded area beyond it.
#[allow(clippy::too_many_arguments)] // One block: plane, position, size, motion, scale.
pub fn predict_scaled(
    dsp: &Dsp,
    reference: &Plane,
    reference_size: (usize, usize),
    sizes: Av1ScaleSizes,
    (x, y): (usize, usize),
    size: usize,
    mv: Av1Mv,
    subsampled: bool,
    dst: &mut PlaneViewMut<'_>,
) {
    let (origin, axes) = scaled_window(sizes, (x, y), mv, (subsampled, subsampled));
    let span = (axes.0.span(size), axes.1.span(size));
    let filters = Av1Filters::both(Av1Filter::Regular);
    let (last_x, last_y) = (reference_size.0 as isize - 1, reference_size.1 as isize - 1);
    let inside = origin.0 >= 0
        && origin.1 >= 0
        && origin.0 + span.0 as isize - 1 <= last_x
        && origin.1 + span.1 as isize - 1 <= last_y;
    if inside && let Some(window) = reference.region(origin.0, origin.1, span.0, span.1) {
        dsp.av1_predict(filters, &window, (size, size), axes, dst);
        return;
    }
    let view = reference.view();
    let total = span.0 * span.1;
    let mut small = [0u8; SMALL_STACK_SPAN];
    let mut large = [0u8; MAX_STACK_SPAN];
    let mut heap = Vec::new();
    let gathered: &mut [u8] = if total <= SMALL_STACK_SPAN {
        &mut small[..total]
    } else if total <= MAX_STACK_SPAN {
        &mut large[..total]
    } else {
        heap.resize(total, 0);
        &mut heap
    };
    for (r, row) in gathered.chunks_exact_mut(span.0).enumerate() {
        let src = view.row((origin.1 + r as isize).clamp(0, last_y) as usize);
        for (c, v) in row.iter_mut().enumerate() {
            *v = src[(origin.0 + c as isize).clamp(0, last_x) as usize];
        }
    }
    let window =
        PlaneView::new(gathered, span.0, span.0, span.1).expect("window covers its samples");
    dsp.av1_predict(filters, &window, (size, size), axes, dst);
}

/// Compound `InterRound1 = 7` unscaled prediction of the `size`x`size` block at
/// (`x`, `y`) of `reference` displaced by `mv` into `dst` (`size * size`
/// unclipped `i16` samples, stride `size`; `AV1 §7.11.3.4`).
pub fn predict_compound(
    dsp: &Dsp,
    reference: &Plane,
    x: usize,
    y: usize,
    size: usize,
    mv: Av1Mv,
    subsampled: bool,
    dst: &mut [i16],
) {
    let shift = u32::from(subsampled);
    let pos_x = ((x as i32) << 4) + ((2 * i32::from(mv.col)) >> shift);
    let pos_y = ((y as i32) << 4) + ((2 * i32::from(mv.row)) >> shift);
    let frac_x = (pos_x & (SUBPEL_SHIFTS - 1)) as usize;
    let frac_y = (pos_y & (SUBPEL_SHIFTS - 1)) as usize;
    let before = Filter::Av1.before() as isize;
    let span = size + Filter::Av1.taps() - 1;
    let x0 = (pos_x >> 4) as isize - before;
    let y0 = (pos_y >> 4) as isize - before;
    let block = BlockSize::new(size, size).expect("square AV1 block size");

    if let Some(window) = reference.region(x0, y0, span, span) {
        dsp.predict_compound(Filter::Av1, block, &window, (frac_x, frac_y), dst, size);
        return;
    }
    let view = reference.view();
    let (last_x, last_y) = (view.width() as isize - 1, view.height() as isize - 1);
    let mut gathered = vec![0u8; span * span];
    for (r, row) in gathered.chunks_exact_mut(span).enumerate() {
        let src = view.row((y0 + r as isize).clamp(0, last_y) as usize);
        for (c, v) in row.iter_mut().enumerate() {
            *v = src[(x0 + c as isize).clamp(0, last_x) as usize];
        }
    }
    let window = PlaneView::new(&gathered, span, span, span).expect("window covers its samples");
    dsp.predict_compound(Filter::Av1, block, &window, (frac_x, frac_y), dst, size);
}

/// Compound `InterRound1 = 7` scaled prediction of the `size`x`size` block at
/// (`x`, `y`) of `reference` displaced by `mv` into `dst` (`size * size`
/// unclipped `i16` samples, stride `size`; `AV1 §7.11.3.3`, `§7.11.3.4`).
#[allow(clippy::too_many_arguments)]
pub fn predict_scaled_compound(
    dsp: &Dsp,
    reference: &Plane,
    reference_size: (usize, usize),
    sizes: Av1ScaleSizes,
    (x, y): (usize, usize),
    size: usize,
    mv: Av1Mv,
    subsampled: bool,
    dst: &mut [i16],
) {
    let (origin, axes) = scaled_window(sizes, (x, y), mv, (subsampled, subsampled));
    let span = (axes.0.span(size), axes.1.span(size));
    let filters = Av1Filters::both(Av1Filter::Regular);
    let (last_x, last_y) = (reference_size.0 as isize - 1, reference_size.1 as isize - 1);
    let inside = origin.0 >= 0
        && origin.1 >= 0
        && origin.0 + span.0 as isize - 1 <= last_x
        && origin.1 + span.1 as isize - 1 <= last_y;
    if inside && let Some(window) = reference.region(origin.0, origin.1, span.0, span.1) {
        dsp.av1_predict_compound(filters, &window, (size, size), axes, dst, size);
        return;
    }
    let view = reference.view();
    let total = span.0 * span.1;
    let mut small = [0u8; SMALL_STACK_SPAN];
    let mut large = [0u8; MAX_STACK_SPAN];
    let mut heap = Vec::new();
    let gathered: &mut [u8] = if total <= SMALL_STACK_SPAN {
        &mut small[..total]
    } else if total <= MAX_STACK_SPAN {
        &mut large[..total]
    } else {
        heap.resize(total, 0);
        &mut heap
    };
    for (r, row) in gathered.chunks_exact_mut(span.0).enumerate() {
        let src = view.row((origin.1 + r as isize).clamp(0, last_y) as usize);
        for (c, v) in row.iter_mut().enumerate() {
            *v = src[(origin.0 + c as isize).clamp(0, last_x) as usize];
        }
    }
    let window =
        PlaneView::new(gathered, span.0, span.0, span.1).expect("window covers its samples");
    dsp.av1_predict_compound(filters, &window, (size, size), axes, dst, size);
}
