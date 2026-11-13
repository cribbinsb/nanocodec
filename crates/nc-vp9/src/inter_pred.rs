//! VP9 inter prediction of a whole block, one plane at a time (VP9 §8.5.2;
//! libvpx `vp9_decodeframe.c` `dec_build_inter_predictors`, BSD-3-Clause,
//! see `THIRD_PARTY.md`).
//!
//! The vector is first clamped so that the block stays within a margin of
//! the frame (`clamp_mv_to_umv_border_sb`); the reference is read with edge
//! replication beyond the visible frame size (which need not be a
//! multiple of 8; the planes cover whole 8x8 units). Interpolation uses the
//! frame's or block's filter, either unscaled (`vp9_convolve`) when the
//! reference matches the frame size or scaled (`vp9_convolve_scaled`,
//! `VP9 §8.5.2.3`–`§8.5.2.4`) when predicting across spatial layers.

use nc_common::{Picture, PlaneIndex, PlaneView};

use crate::types::{MI_SIZE, Mv};

/// Largest block side.
const MAX_SIDE: usize = 64;
/// Window side for the largest block at the largest reference scale step
/// (`134`): `(((64 - 1) * 32 + 15) >> 4) + 8`.
const MAX_WINDOW: usize =
    (((MAX_SIDE - 1) * nc_dsp::VP9_MAX_SCALED_STEP + 15) >> 4) + nc_dsp::VP9_TAPS;
/// Samples of one block's edge-replicated reference window (the
/// `window` scratch of [`predict`] / [`predict_scaled`]).
pub(crate) const WINDOW_SAMPLES: usize = MAX_WINDOW * MAX_WINDOW;
/// Samples of filter support kept around the block (`VP9_INTERP_EXTEND`).
const INTERP_EXTEND: i32 = 4;
/// Sub-sample bits of the 1/16-sample plane positions (`SUBPEL_BITS`).
const SUBPEL_BITS: u32 = 4;
const SUBPEL_MASK: i32 = (1 << SUBPEL_BITS) - 1;
/// `REF_SCALE_SHIFT`: fixed-point bits of a reference scale factor (`VP9 §8.5.2.3`).
const REF_SCALE_SHIFT: u32 = 14;
const REF_NO_SCALE: i64 = 1 << REF_SCALE_SHIFT;

/// Position of a block inside the frame, in 8x8 units, for the clamps.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockPosition {
    /// Top-left 8x8 unit of the block.
    pub row: usize,
    pub col: usize,
    /// Block side in 8x8 units.
    pub size: usize,
    /// Frame size in 8x8 units.
    pub rows: usize,
    pub cols: usize,
    /// Visible frame size in luma samples (`frame_width`, `frame_height`):
    /// reference samples beyond it replicate its edge.
    pub width: usize,
    pub height: usize,
}

impl BlockPosition {
    /// Distances from the block to the frame edges, in 1/8 luma samples
    /// (`mb_to_left_edge` .. `mb_to_bottom_edge`; right/bottom are negative
    /// once the block passes the edge).
    pub(crate) fn edges(&self) -> [i32; 4] {
        let eighths = |units: isize| (units * MI_SIZE as isize * 8) as i32;
        [
            -eighths(self.col as isize),
            eighths(self.cols as isize - self.size as isize - self.col as isize),
            -eighths(self.row as isize),
            eighths(self.rows as isize - self.size as isize - self.row as isize),
        ]
    }
}

/// The vector of one plane in 1/16 plane samples, clamped as the decoder does.
#[inline]
fn clamped_plane_mv(mv: Mv, position: &BlockPosition, subsampling: u32, side: i32) -> (i32, i32) {
    let [left, right, top, bottom] = position.edges();
    let scale = 1 << (1 - subsampling);
    let spel_left = (INTERP_EXTEND + side) << SUBPEL_BITS;
    let spel_right = spel_left - (1 << SUBPEL_BITS);
    let row = (mv.row * scale).clamp(top * scale - spel_left, bottom * scale + spel_right);
    let col = (mv.col * scale).clamp(left * scale - spel_left, right * scale + spel_right);
    (row, col)
}

/// Predict `plane` of the block at `position` from `reference` into `out`
/// (row stride = plane block side); `window` (at least
/// [`WINDOW_SAMPLES`]) is scratch for blocks near the frame edge.
pub(crate) fn predict(
    dsp: &nc_dsp::Dsp,
    reference: &Picture,
    plane: PlaneIndex,
    position: &BlockPosition,
    mv: Mv,
    filter: nc_dsp::Vp9Filter,
    window: &mut [u8],
    out: &mut [u8],
) {
    let subsampling = u32::from(plane != PlaneIndex::Y);
    let side = (position.size * MI_SIZE) >> subsampling;
    let (row, col) = clamped_plane_mv(mv, position, subsampling, side as i32);
    let x = ((position.col * MI_SIZE) >> subsampling) as i32;
    let y = ((position.row * MI_SIZE) >> subsampling) as i32;
    let (x16, y16) = (x * 16 + col, y * 16 + row);
    let (ix, iy) = (x16.div_euclid(16), y16.div_euclid(16));
    let frac = (x16.rem_euclid(16) as usize, y16.rem_euclid(16) as usize);
    // The window, with samples beyond the plane replicating its edges.
    let plane = reference.plane(plane);
    let view = plane.view();
    // The reference as VP9 decoders read it: the visible picture with its
    // edges replicated (libvpx extends borders from the crop size).
    let visible = |size: usize| ((size + subsampling as usize) >> subsampling) as i32;
    let (width, height) = (visible(position.width), visible(position.height));
    let before = nc_dsp::VP9_TAPS_BEFORE as i32;
    let span = side + nc_dsp::VP9_TAPS - 1;
    let (left, top) = (ix - before, iy - before);
    if left >= 0 && top >= 0 && left + span as i32 <= width && top + span as i32 <= height {
        // The whole window is inside the plane: read it in place.
        let start = top as usize * view.stride() + left as usize;
        dsp.vp9_convolve(
            filter,
            &view.data()[start..],
            view.stride(),
            frac,
            (side, side),
            out,
        );
        return;
    }
    // Otherwise build it (in the caller's scratch, not zeroed per block)
    // with the reference's edge replication.
    fill_window(&view, (width, height), (left, top), (span, span), window);
    dsp.vp9_convolve(
        filter,
        &window[..span * span],
        span,
        frac,
        (side, side),
        out,
    );
}

/// Predict `plane` of the block at `position` from `reference` of visible
/// size `ref_size` (`(ref_width, ref_height)`) into `out` (row stride = plane
/// block side), using `VP9 §8.5.2.3`–`§8.5.2.4` reference scaling when
/// `ref_size != (position.width, position.height)`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn predict_with_ref_size(
    dsp: &nc_dsp::Dsp,
    reference: &Picture,
    ref_size: (usize, usize),
    plane: PlaneIndex,
    position: &BlockPosition,
    mv: Mv,
    filter: nc_dsp::Vp9Filter,
    window: &mut [u8],
    out: &mut [u8],
) {
    // A same-size reference (every single-layer frame) is not scaled: the
    // check is inlined into the callers, which then call `predict` directly.
    if ref_size == (position.width, position.height) {
        predict(dsp, reference, plane, position, mv, filter, window, out);
    } else {
        predict_scaled(
            dsp, reference, ref_size, plane, position, mv, filter, window, out,
        );
    }
}

/// [`predict_with_ref_size`] from a reference of another size.
#[allow(clippy::too_many_arguments)]
fn predict_scaled(
    dsp: &nc_dsp::Dsp,
    reference: &Picture,
    ref_size: (usize, usize),
    plane: PlaneIndex,
    position: &BlockPosition,
    mv: Mv,
    filter: nc_dsp::Vp9Filter,
    window: &mut [u8],
    out: &mut [u8],
) {
    let x_fp = ((ref_size.0 as i64) << REF_SCALE_SHIFT) / position.width as i64;
    let y_fp = ((ref_size.1 as i64) << REF_SCALE_SHIFT) / position.height as i64;
    if x_fp == REF_NO_SCALE && y_fp == REF_NO_SCALE {
        predict(dsp, reference, plane, position, mv, filter, window, out);
        return;
    }
    let scale_x = |val: i32| ((i64::from(val) * x_fp) >> REF_SCALE_SHIFT) as i32;
    let scale_y = |val: i32| ((i64::from(val) * y_fp) >> REF_SCALE_SHIFT) as i32;
    let x_step = scale_x(16) as usize;
    let y_step = scale_y(16) as usize;

    let subsampling = u32::from(plane != PlaneIndex::Y);
    let side = (position.size * MI_SIZE) >> subsampling;
    let (mv_row, mv_col) = clamped_plane_mv(mv, position, subsampling, side as i32);
    let mi_x = (position.col * MI_SIZE) as i32;
    let mi_y = (position.row * MI_SIZE) as i32;
    let x_start = mi_x >> subsampling;
    let y_start = mi_y >> subsampling;
    let x0 = scale_x(x_start);
    let y0 = scale_y(y_start);
    let col = scale_x(mv_col) + (scale_x(mi_x << SUBPEL_BITS) & SUBPEL_MASK);
    let row = scale_y(mv_row) + (scale_y(mi_y << SUBPEL_BITS) & SUBPEL_MASK);
    let x16 = ((x0 + (col >> SUBPEL_BITS)) << SUBPEL_BITS) + (col & SUBPEL_MASK);
    let y16 = ((y0 + (row >> SUBPEL_BITS)) << SUBPEL_BITS) + (row & SUBPEL_MASK);
    let (x_phase, y_phase) = ((x16 & SUBPEL_MASK) as usize, (y16 & SUBPEL_MASK) as usize);
    let span_w = ((x_phase + (side - 1) * x_step) >> SUBPEL_BITS) + nc_dsp::VP9_TAPS;
    let span_h = ((y_phase + (side - 1) * y_step) >> SUBPEL_BITS) + nc_dsp::VP9_TAPS;
    let before = nc_dsp::VP9_TAPS_BEFORE as i32;
    let (left, top) = ((x16 >> SUBPEL_BITS) - before, (y16 >> SUBPEL_BITS) - before);

    let plane = reference.plane(plane);
    let view = plane.view();
    let visible = |size: usize| ((size + subsampling as usize) >> subsampling) as i32;
    let (ref_w, ref_h) = (visible(ref_size.0), visible(ref_size.1));
    if left >= 0 && top >= 0 && left + span_w as i32 <= ref_w && top + span_h as i32 <= ref_h {
        let start = top as usize * view.stride() + left as usize;
        nc_dsp::vp9_convolve_scaled(
            filter,
            &view.data()[start..],
            view.stride(),
            (x_phase, x_step),
            (y_phase, y_step),
            (side, side),
            out,
        );
        return;
    }
    fill_window(&view, (ref_w, ref_h), (left, top), (span_w, span_h), window);
    nc_dsp::vp9_convolve_scaled(
        filter,
        &window[..span_w * span_h],
        span_w,
        (x_phase, x_step),
        (y_phase, y_step),
        (side, side),
        out,
    );
}

fn fill_window(
    view: &PlaneView<'_>,
    (width, height): (i32, i32),
    (left, top): (i32, i32),
    (span_w, span_h): (usize, usize),
    window: &mut [u8],
) {
    let window = &mut window[..span_w * span_h];
    let first = left.clamp(0, width) as usize;
    let end = (left + span_w as i32).clamp(0, width) as usize;
    let before_plane = (-left).clamp(0, span_w as i32) as usize;
    let inside = end - first;
    for (wy, line) in window.chunks_exact_mut(span_w).enumerate() {
        let source = &view.row((top + wy as i32).clamp(0, height - 1) as usize)[..width as usize];
        let (head, rest) = line.split_at_mut(before_plane);
        head.fill(source[0]);
        let (middle, tail) = rest.split_at_mut(inside);
        middle.copy_from_slice(&source[first..end]);
        tail.fill(source[width as usize - 1]);
    }
}
