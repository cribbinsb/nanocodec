//! Prediction from a reference of another size than the frame (dav2d
//! `mc`'s scaled branch and `ext_warp_scaled`): block positions map into
//! the reference by its scale ([`ReferenceScale`]), the filters step
//! through it ([`nc_dsp::av2_dec::mc_scaled`]), and windows reaching past
//! its 8-rounded size repeat its edge samples.
use nc_common::PlaneIndex;
use nc_dsp::av2_dec::mc_8tap::InterpFilter;
use nc_dsp::av2_dec::mc_scaled::{self, ScaledSteps, WarpSteps};

use super::mc::Target;
use super::{Plane, kernels};
use crate::general::block::Tile;
use crate::general::block::frame_refs::ReferenceScale;
use crate::general::block::motion::Mv;
use crate::general::sample::Av2Sample;
use crate::general::warp::WarpModel;

/// One in a scale factor (1/16384).
const SCALE_ONE: i64 = 1 << 14;
/// A scaled position: the product's rounding and shift to 1/1024 sample,
/// and the half-sample centring it adds (dav2d `scale_mv`).
const SCALE_POSITION_ROUND: i64 = 128;
const SCALE_POSITION_SHIFT: u32 = 8;
const SCALE_POSITION_CENTRE: i32 = 32;
/// The centre offset of the scale factor in a position (dav2d's
/// `(scale - 0x4000) * 8`).
const SCALE_OFFSET_FACTOR: i64 = 8;
/// Fraction bits of a scaled position.
const POSITION_BITS: u32 = 10;
const POSITION_MASK: i32 = (1 << POSITION_BITS) - 1;
/// Taps of the 8-tap filters before and after a sample.
const TAPS_BEFORE: i32 = 3;
const TAPS_AFTER: i32 = 4;
/// Rows of the scaled window scratch (dav2d's 320-wide `emu_edge` buffer):
/// a 64-sample block at twice the size plus the taps fits.
const WINDOW_STRIDE: usize = 320;
/// Warps of a scaled reference run in sub-blocks of at most 8 samples a
/// side, each in 4x4 filter blocks.
const WARP_BLOCK: i32 = 8;
const EXT_WARP_BLOCK: i32 = 4;
/// Fraction bits of warp model positions.
const WARP_MODEL_BITS: u32 = 16;
/// A warp position's offset of half a model unit (dav2d `0x10000`) and
/// the vector's `>> 13`, from model to eighth-sample units.
const WARP_MODEL_ONE: i64 = 1 << WARP_MODEL_BITS;
const WARP_MV_SHIFT: u32 = 13;
/// The 4x4 block's top-left from its centre (dav2d `- 0x20000`, two
/// samples), and the scaled warp position's rounding (dav2d `0x2000 >> 14`).
const WARP_CENTRE_OFFSET: i64 = 2 << WARP_MODEL_BITS;
const WARP_SCALE_ROUND: i64 = 1 << 13;
const WARP_SCALE_SHIFT: u32 = 14;
/// Samples of a scaled warp's filter window beyond its steps (dav2d
/// `src_w = ((mx + 3 * dx) >> 16) + 9`).
const WARP_WINDOW_EXTRA: i32 = 9;
/// Steps of a warp in 1/65536 from steps in 1/1024.
const WARP_STEP_SHIFT: u32 = 6;

/// `value * scale` around the scale's centre in 1/1024 sample (dav2d
/// `scale_mv`, `scale_pos`).
fn scale_position(value: i32, scale: i32) -> i32 {
    let product =
        i64::from(value) * i64::from(scale) + (i64::from(scale) - SCALE_ONE) * SCALE_OFFSET_FACTOR;
    let magnitude = (product.abs() + SCALE_POSITION_ROUND) >> SCALE_POSITION_SHIFT;
    (if product < 0 { -magnitude } else { magnitude }) as i32 + SCALE_POSITION_CENTRE
}

/// A warp position scaled to the reference (dav2d `scale_warp_pos`).
fn scale_warp_position(value: i64, scale: i32) -> i64 {
    let magnitude = ((value * i64::from(scale)).abs() + WARP_SCALE_ROUND) >> WARP_SCALE_SHIFT;
    if value < 0 { -magnitude } else { magnitude }
}

impl<T: Av2Sample> Tile<'_, '_, T> {
    /// The width and height of a reference plane, 8-rounded in luma (dav2d
    /// `((p.w + 7) & ~7) >> ss`): references are kept at that size.
    fn reference_extent(&self, reference: usize, plane: Plane) -> (i32, i32) {
        let picture = self.reference_picture(reference);
        let plane = picture.plane(PlaneIndex::ALL[plane.index()]);
        (plane.width() as i32, plane.height() as i32)
    }

    /// The source of a scaled read of `size` samples whose top-left is at
    /// (`left`, `top`) of the reference: the reference itself, or the
    /// scaled window with its edges repeated (offset by the taps before).
    fn scaled_source(
        &mut self,
        reference: usize,
        plane: Plane,
        (left, top): (i32, i32),
        (width, height): (i32, i32),
        (before, after): (i32, i32),
    ) -> (bool, usize, usize) {
        let (w, h) = self.reference_extent(reference, plane);
        let picture = self.reference_picture(reference);
        let (data, stride) = kernels::plane(picture, plane.index());
        if left < before || top < before || left + width + after > w || top + height + after > h {
            let window = &mut self.inter_scratch.scaled_window;
            let rows = (height + before + after) as usize;
            if window.len() < rows * WINDOW_STRIDE {
                window.resize(rows * WINDOW_STRIDE, T::default());
            }
            nc_dsp::emulate_edges(
                (data, stride),
                (w as usize, h as usize),
                ((left - before) as isize, (top - before) as isize),
                ((width + before + after) as usize, rows),
                (window, WINDOW_STRIDE),
            );
            (
                true,
                before as usize * WINDOW_STRIDE + before as usize,
                WINDOW_STRIDE,
            )
        } else {
            (false, top as usize * stride + left as usize, stride)
        }
    }

    /// [`Tile::mc`] from a reference of another size (dav2d `mc`'s
    /// scaled branch).
    #[allow(clippy::too_many_arguments, reason = "dav2d's mc contract")]
    pub(super) fn mc_scaled(
        &mut self,
        target: Target,
        plane: Plane,
        (w, h): (usize, usize),
        (bx, by): (u32, u32),
        mv: Mv,
        (reference, scale): (usize, ReferenceScale),
        filter: u8,
    ) {
        let shift = plane.shift();
        let mul = 4 >> shift;
        // Sixteenth samples of the plane.
        let origin_x = ((bx as i32 * mul) << 4) + mv.x * (1 << (1 - shift));
        let origin_y = ((by as i32 * mul) << 4) + mv.y * (1 << (1 - shift));
        let (x, y) = (
            scale_position(origin_x, scale.scale[0]),
            scale_position(origin_y, scale.scale[1]),
        );
        let (left, top) = (x >> POSITION_BITS, y >> POSITION_BITS);
        let right = ((x + (w as i32 - 1) * scale.step[0]) >> POSITION_BITS) + 1;
        let bottom = ((y + (h as i32 - 1) * scale.step[1]) >> POSITION_BITS) + 1;
        let (window, offset, stride) = self.scaled_source(
            reference,
            plane,
            (left, top),
            (right - left, bottom - top),
            (TAPS_BEFORE, TAPS_AFTER),
        );
        let steps = ScaledSteps {
            fraction: (x & POSITION_MASK, y & POSITION_MASK),
            step: (scale.step[0], scale.step[1]),
        };
        let filter = InterpFilter::from_index(filter).unwrap_or(InterpFilter::Regular);
        let depth = self.depth().of::<T>();
        let picture = self.reference_picture(reference);
        let src = if window {
            &self.inter_scratch.scaled_window[..]
        } else {
            kernels::plane(picture, plane.index()).0
        };
        match target {
            Target::Picture(dst_offset) => {
                let dst_stride = self.planes.stride[plane.index()];
                let dst = &mut self.planes.data[plane.index()][..];
                mc_scaled::put_scaled(
                    (dst, dst_offset, dst_stride),
                    (src, offset, stride),
                    (w, h),
                    steps,
                    filter,
                    depth,
                );
            }
            Target::Compound {
                index,
                offset: tmp_offset,
                stride: tmp_stride,
            } => {
                let tmp = &mut self.inter_scratch.compound[index][tmp_offset..];
                mc_scaled::prep_scaled(
                    (tmp, tmp_stride),
                    (src, offset, stride),
                    (w, h),
                    steps,
                    filter,
                    depth,
                );
            }
            Target::Refine { .. } => {
                unreachable!("refinement is off for references of another size")
            }
        }
    }

    /// The extended warp of a `w` x `h` block at unit (`bx`, `by`) of
    /// `plane` by `model` from a reference of another size (dav2d
    /// `ext_warp_scaled`): per sub-block of at most 8x8, a window around
    /// its centre's vector, then 4x4 filter blocks.
    #[allow(
        clippy::too_many_arguments,
        reason = "dav2d's ext_warp_scaled contract"
    )]
    pub(super) fn ext_warp_scaled(
        &mut self,
        target: Target,
        plane: Plane,
        (w, h): (i32, i32),
        (bx, by): (u32, u32),
        (reference, scale): (usize, ReferenceScale),
        model: &WarpModel,
    ) {
        let shift = plane.shift() as i32;
        let m = model.matrix.map(i64::from);
        let (width, height) = self.reference_extent(reference, plane);
        let (sw, sh) = (w.min(WARP_BLOCK), h.min(WARP_BLOCK));
        let (px, py) = ((bx as i32 * 4) >> shift, (by as i32 * 4) >> shift);
        let (step_x, step_y) = (scale.step[0], scale.step[1]);
        let (dx, dy) = (step_x << WARP_STEP_SHIFT, step_y << WARP_STEP_SHIFT);
        let unshifted = 1 - shift;
        for y in (0..h).step_by(sh as usize) {
            for x in (0..w).step_by(sw as usize) {
                // The window around the eighth-sample vector at the
                // sub-block's centre, in luma samples.
                let cx = i64::from(((px + x) << shift) + ((sw << shift) >> 1));
                let cy = i64::from(((py + y) << shift) + ((sh << shift) >> 1));
                let clamp = |value: i64| value.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
                let mvx = (clamp((m[2] - WARP_MODEL_ONE) * cx + m[3] * cy + m[0]) >> WARP_MV_SHIFT)
                    as i32;
                let mvy = (clamp(m[4] * cx + (m[5] - WARP_MODEL_ONE) * cy + m[1]) >> WARP_MV_SHIFT)
                    as i32;
                let pos_x =
                    scale_position(((px + x) << 4) + mvx * (1 << unshifted), scale.scale[0]);
                let pos_y =
                    scale_position(((py + y) << 4) + mvy * (1 << unshifted), scale.scale[1]);
                let left = ((pos_x >> POSITION_BITS) - TAPS_BEFORE).clamp(0, width - 1);
                let right = (((pos_x + (sw - 1) * step_x) >> POSITION_BITS) + 1 + TAPS_AFTER)
                    .clamp(1, width);
                let top = ((pos_y >> POSITION_BITS) - TAPS_BEFORE).clamp(0, height - 1);
                let bottom = (((pos_y + (sh - 1) * step_y) >> POSITION_BITS) + 1 + TAPS_AFTER)
                    .clamp(1, height);
                for yy in (y..y + sh).step_by(EXT_WARP_BLOCK as usize) {
                    let src_y = i64::from((py + yy + 2) << shift);
                    let (m3, m5) = (m[3] * src_y + m[0], m[5] * src_y + m[1]);
                    for xx in (x..x + sw).step_by(EXT_WARP_BLOCK as usize) {
                        let src_x = i64::from((px + xx + 2) << shift);
                        let pos4_x = scale_warp_position(
                            ((m[2] * src_x + m3) >> shift) - WARP_CENTRE_OFFSET,
                            scale.scale[0],
                        );
                        let pos4_y = scale_warp_position(
                            ((m[4] * src_x + m5) >> shift) - WARP_CENTRE_OFFSET,
                            scale.scale[1],
                        );
                        let (ix, mx) = (
                            (pos4_x >> WARP_MODEL_BITS) as i32,
                            (pos4_x & (WARP_MODEL_ONE - 1)) as i32,
                        );
                        let (iy, my) = (
                            (pos4_y >> WARP_MODEL_BITS) as i32,
                            (pos4_y & (WARP_MODEL_ONE - 1)) as i32,
                        );
                        let src_w = ((mx + 3 * dx) >> WARP_MODEL_BITS) + WARP_WINDOW_EXTRA;
                        let src_h = ((my + 3 * dy) >> WARP_MODEL_BITS) + WARP_WINDOW_EXTRA;
                        let source = self.warp_source(
                            (reference, plane),
                            (ix, iy),
                            (src_w, src_h),
                            (left, right, top, bottom),
                        );
                        let steps = WarpSteps {
                            fraction: (mx, my),
                            step: (dx, dy),
                        };
                        self.ext_warp_scaled_block(
                            target,
                            plane,
                            (reference, source),
                            (xx as usize, yy as usize),
                            steps,
                        );
                    }
                }
            }
        }
    }

    /// The source of a scaled 4x4 warp block reading `size` samples from 3
    /// before (`ix`, `iy`): the reference, or the window when the read
    /// leaves the sub-block's area (dav2d `ext_warp_scaled`'s `emu_edge`).
    fn warp_source(
        &mut self,
        (reference, plane): (usize, Plane),
        (ix, iy): (i32, i32),
        (src_w, src_h): (i32, i32),
        (left, right, top, bottom): (i32, i32, i32, i32),
    ) -> (bool, usize, usize) {
        let picture = self.reference_picture(reference);
        let (data, stride) = kernels::plane(picture, plane.index());
        if ix - TAPS_BEFORE < left
            || ix - TAPS_BEFORE + src_w > right
            || iy - TAPS_BEFORE < top
            || iy - TAPS_BEFORE + src_h > bottom
        {
            let window = &mut self.inter_scratch.scaled_window;
            let rows = src_h as usize;
            if window.len() < rows * WINDOW_STRIDE {
                window.resize(rows * WINDOW_STRIDE, T::default());
            }
            let origin = top as usize * stride + left as usize;
            nc_dsp::emulate_edges(
                (&data[origin..], stride),
                ((right - left) as usize, (bottom - top) as usize),
                (
                    (ix - TAPS_BEFORE - left) as isize,
                    (iy - TAPS_BEFORE - top) as isize,
                ),
                (src_w as usize, rows),
                (window, WINDOW_STRIDE),
            );
            (
                true,
                TAPS_BEFORE as usize * WINDOW_STRIDE + TAPS_BEFORE as usize,
                WINDOW_STRIDE,
            )
        } else {
            (false, iy as usize * stride + ix as usize, stride)
        }
    }

    /// Run one scaled 4x4 warp block into the target at (`x`, `y`) of the
    /// block.
    fn ext_warp_scaled_block(
        &mut self,
        target: Target,
        plane: Plane,
        (reference, (window, offset, stride)): (usize, (bool, usize, usize)),
        (x, y): (usize, usize),
        steps: WarpSteps,
    ) {
        let depth = self.depth().of::<T>();
        let picture = self.reference_picture(reference);
        let src = if window {
            &self.inter_scratch.scaled_window[..]
        } else {
            kernels::plane(picture, plane.index()).0
        };
        let src = (src, offset, stride);
        match target {
            Target::Picture(dst_offset) => {
                let dst_stride = self.planes.stride[plane.index()];
                let dst = &mut self.planes.data[plane.index()][..];
                mc_scaled::ext_warp4x4_scaled(
                    (dst, dst_offset + y * dst_stride + x, dst_stride),
                    src,
                    steps,
                    depth,
                );
            }
            Target::Compound {
                index,
                offset: tmp_offset,
                stride: tmp_stride,
            } => {
                let tmp = &mut self.inter_scratch.compound[index][..];
                mc_scaled::ext_warp4x4t_scaled(
                    (tmp, tmp_offset + y * tmp_stride + x, tmp_stride),
                    src,
                    steps,
                    depth,
                );
            }
            Target::Refine { .. } => unreachable!("refinement windows are not warped"),
        }
    }
}
