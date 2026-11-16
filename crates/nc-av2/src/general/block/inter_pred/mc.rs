//! Motion compensation of one block from a reference picture: the
//! translational filters with their reference window (dav2d `mc`,
//! `mc_opfl`), and warps by 8x8 or 4x4 sub-blocks (dav2d `warp_affine`,
//! `ext_warp`); windows reaching past the frame repeat its edge samples
//! (dav2d `emu_edge`).
use super::kernels;
use super::{Plane, UNIT_LOG2};
use crate::general::block::Tile;
use crate::general::block::motion::Mv;
use crate::general::block::size::BlockSize;
use crate::general::warp::WarpModel;

/// Taps of the 8-tap filters before and after a sample.
const TAPS_BEFORE: i32 = 3;
const TAPS_AFTER: i32 = 4;
/// Warps filter 8x8 sub-blocks (4x4 for blocks under 8 samples a side or
/// non-affine models).
const WARP_BLOCK: i32 = 8;
const EXT_WARP_BLOCK: i32 = 4;
/// Fraction bits of warp positions and of a vector's sub-sample phase.
const WARP_POSITION_BITS: u32 = 16;
const WARP_PHASE_BITS: u32 = 10;
const WARP_PHASE_MASK: i64 = 63;
/// Warp sub-block positions are rounded by half a phase step.
const EXT_WARP_ROUND: i64 = 0x200;

/// The columns of a refinement window past its power-of-two width.
const REFINE_TAIL: usize = 8;

/// Where a prediction goes.
#[derive(Debug, Clone, Copy)]
pub(super) enum Target {
    /// The plane being reconstructed, at this offset.
    Picture(usize),
    /// A compound intermediate, at this offset, rows `stride` apart.
    Compound {
        index: usize,
        offset: usize,
        stride: usize,
    },
    /// A refinement search's 8-bit window, rows `stride` apart.
    Refine { index: usize, stride: usize },
}

/// The area of a reference a prediction may read, in plane samples
/// (dav2d `mc`'s `left`, `right`, `top`, `bottom`).
#[derive(Debug, Clone, Copy)]
pub(super) struct Bounds {
    pub left: i32,
    pub right: i32,
    pub top: i32,
    pub bottom: i32,
}

/// Where a filter reads its source: a reference plane at an offset, or
/// the window scratch at an offset.
#[derive(Debug, Clone, Copy)]
enum Source {
    Reference(usize),
    Window(usize),
}

impl Tile<'_, '_> {
    /// Predict a `bw4` x `bh4` block (luma 4x4 units; chroma of the luma
    /// grid) at unit (`bx`, `by`) of `plane` by `mv` (eighth samples of
    /// luma) from reference `reference` with `filter` (dav2d `mc`).
    #[allow(clippy::too_many_arguments, reason = "dav2d's mc contract")]
    pub(super) fn mc(
        &mut self,
        target: Target,
        plane: Plane,
        (bw4, bh4): (u32, u32),
        (bx, by): (u32, u32),
        mv: Mv,
        reference: usize,
        filter: u8,
        bounds: Bounds,
    ) {
        let shift = plane.shift();
        let mul = 4 >> shift;
        let phase_mask = 15 >> (1 - shift);
        let (mx, my) = (mv.x & phase_mask, mv.y & phase_mask);
        let dx = bx as i32 * mul + (mv.x >> (3 + shift));
        let dy = by as i32 * mul + (mv.y >> (3 + shift));
        let (w, h) = ((bw4 as i32 * mul) as usize, (bh4 as i32 * mul) as usize);
        let source = self.window_or_reference(
            plane,
            reference,
            (dx, dy),
            (w, h),
            (mx != 0, my != 0),
            bounds,
        );
        // Phases in sixteenths: luma's eighth-sample phase doubles.
        let phase = (mx << (1 - shift), my << (1 - shift));
        self.filter_into(target, plane, source, reference, (w, h), phase, filter);
    }

    /// Like [`Tile::mc`] with sixteenth-sample vectors in the plane's own
    /// samples and an intermediate target (dav2d `mc_opfl`).
    #[allow(clippy::too_many_arguments, reason = "dav2d's mc_opfl contract")]
    pub(super) fn mc_sixteenth(
        &mut self,
        target: Target,
        plane: Plane,
        (bw4, bh4): (u32, u32),
        (bx4, by4): (u32, u32),
        mv: Mv,
        reference: usize,
        filter: u8,
        bounds: Bounds,
    ) {
        let (mx, my) = (mv.x & 15, mv.y & 15);
        let dx = (bx4 << UNIT_LOG2) as i32 + (mv.x >> 4);
        let dy = (by4 << UNIT_LOG2) as i32 + (mv.y >> 4);
        let (w, h) = ((bw4 << UNIT_LOG2) as usize, (bh4 << UNIT_LOG2) as usize);
        let source = self.window_or_reference(
            plane,
            reference,
            (dx, dy),
            (w, h),
            (mx != 0, my != 0),
            bounds,
        );
        self.filter_into(target, plane, source, reference, (w, h), (mx, my), filter);
    }

    /// The source of a `w` x `h` prediction at (`dx`, `dy`): the reference
    /// itself, or a window of it with repeated edges where the filter taps
    /// (when the phase is not zero) leave `bounds`.
    fn window_or_reference(
        &mut self,
        plane: Plane,
        reference: usize,
        (dx, dy): (i32, i32),
        (w, h): (usize, usize),
        (filtered_x, filtered_y): (bool, bool),
        bounds: Bounds,
    ) -> Source {
        let before = |filtered: bool| if filtered { TAPS_BEFORE } else { 0 };
        let after = |filtered: bool| if filtered { TAPS_AFTER } else { 0 };
        let (bx, by) = (before(filtered_x), before(filtered_y));
        let picture = self.reference_picture(reference);
        let (data, stride) = kernels::plane(picture, plane.index());
        if dx - bx < bounds.left
            || dy - by < bounds.top
            || dx + w as i32 + after(filtered_x) > bounds.right
            || dy + h as i32 + after(filtered_y) > bounds.bottom
        {
            let width = w + (bx + after(filtered_x)) as usize;
            let height = h + (by + after(filtered_y)) as usize;
            let origin = bounds.top as usize * stride + bounds.left as usize;
            nc_dsp::emulate_edges(
                (&data[origin..], stride),
                (
                    (bounds.right - bounds.left) as usize,
                    (bounds.bottom - bounds.top) as usize,
                ),
                (
                    (dx - bx - bounds.left) as isize,
                    (dy - by - bounds.top) as isize,
                ),
                (width, height),
                (&mut self.inter_scratch.window, kernels::WINDOW_STRIDE),
            );
            Source::Window(by as usize * kernels::WINDOW_STRIDE + bx as usize)
        } else {
            Source::Reference(dy as usize * stride + dx as usize)
        }
    }

    /// Run the translational filter into a target.
    #[allow(clippy::too_many_arguments, reason = "the filter call's parts")]
    fn filter_into(
        &mut self,
        target: Target,
        plane: Plane,
        source: Source,
        reference: usize,
        (w, h): (usize, usize),
        (mx, my): (i32, i32),
        filter: u8,
    ) {
        let picture = self.reference_picture(reference);
        let (reference_data, reference_stride) = kernels::plane(picture, plane.index());
        let (src, src_offset, src_stride) = match source {
            Source::Reference(offset) => (reference_data, offset, reference_stride),
            Source::Window(offset) => (
                &self.inter_scratch.window[..],
                offset,
                kernels::WINDOW_STRIDE,
            ),
        };
        match target {
            Target::Picture(offset) => {
                let stride = self.planes.stride[plane.index()];
                kernels::put(
                    filter,
                    (&mut self.planes.data[plane.index()], offset, stride),
                    (src, src_offset, src_stride),
                    (w, h),
                    (mx, my),
                );
            }
            Target::Compound {
                index,
                offset,
                stride,
            } => {
                kernels::prep(
                    filter,
                    (&mut self.inter_scratch.compound[index], offset, stride),
                    (src, src_offset, src_stride),
                    (w, h),
                    (mx, my),
                );
            }
            Target::Refine { index, stride } => {
                // Refinement windows are two units wider than a power of
                // two: the kernel runs on the power of two and then the
                // last two units (dav2d `mc`'s `bw4 & (bw4 - 1)` case).
                let dst = &mut self.inter_scratch.refine[index];
                let head = if w.is_power_of_two() {
                    w
                } else {
                    w - REFINE_TAIL
                };
                kernels::put(
                    filter,
                    (dst, 0, stride),
                    (src, src_offset, src_stride),
                    (head, h),
                    (mx, my),
                );
                if head != w {
                    kernels::put(
                        filter,
                        (dst, head, stride),
                        (src, src_offset + head, src_stride),
                        (REFINE_TAIL, h),
                        (mx, my),
                    );
                }
            }
        }
    }

    /// Warp a block of `size` (plane units of the luma grid) at unit
    /// (`bx`, `by`) by `model` (dav2d `warp_affine`): 8x8 sub-blocks of an
    /// affine model, else 4x4 ones.
    pub(super) fn warp_affine(
        &mut self,
        target: Target,
        plane: Plane,
        size: BlockSize,
        (bx, by): (u32, u32),
        reference: usize,
        model: &WarpModel,
    ) {
        let shift = plane.shift();
        let (bw4, bh4) = size.units();
        let (w, h) = ((bw4 << UNIT_LOG2) >> shift, (bh4 << UNIT_LOG2) >> shift);
        if !model.affine || w.min(h) < WARP_BLOCK as u32 {
            self.ext_warp(
                target,
                plane,
                (w as i32, h as i32),
                (bx, by),
                reference,
                model,
            );
            return;
        }
        let m = model.matrix.map(i64::from);
        let width = ((self.geometry.columns << UNIT_LOG2) >> shift) as i32;
        let height = ((self.geometry.rows << UNIT_LOG2) >> shift) as i32;
        let shear = model.shear;
        let window = (WARP_BLOCK + TAPS_BEFORE + TAPS_AFTER) as usize;
        for y in (0..h as i32).step_by(WARP_BLOCK as usize) {
            let src_y = i64::from(by as i32 * 4 + ((y + 4) << shift));
            let (m3, m5) = (m[3] * src_y + m[0], m[5] * src_y + m[1]);
            for x in (0..w as i32).step_by(WARP_BLOCK as usize) {
                let src_x = i64::from(bx as i32 * 4 + ((x + 4) << shift));
                let mvx = (m[2] * src_x + m3) >> shift;
                let mvy = (m[4] * src_x + m5) >> shift;
                let dx = (mvx >> WARP_POSITION_BITS) as i32 - 4;
                let mx = ((mvx as i32 & 0xffff)
                    - i32::from(shear.alpha) * 4
                    - i32::from(shear.beta) * 7)
                    & !0x3f;
                let dy = (mvy >> WARP_POSITION_BITS) as i32 - 4;
                let my = ((mvy as i32 & 0xffff)
                    - i32::from(shear.gamma) * 4
                    - i32::from(shear.delta) * 4)
                    & !0x3f;
                let picture = self.reference_picture(reference);
                let (data, stride) = kernels::plane(picture, plane.index());
                let source = if dx < TAPS_BEFORE
                    || dx + WARP_BLOCK + TAPS_AFTER > width
                    || dy < TAPS_BEFORE
                    || dy + WARP_BLOCK + TAPS_AFTER > height
                {
                    nc_dsp::emulate_edges(
                        (data, stride),
                        (width as usize, height as usize),
                        ((dx - TAPS_BEFORE) as isize, (dy - TAPS_BEFORE) as isize),
                        (window, window),
                        (&mut self.inter_scratch.window, kernels::WINDOW_STRIDE),
                    );
                    Source::Window(
                        TAPS_BEFORE as usize * kernels::WINDOW_STRIDE + TAPS_BEFORE as usize,
                    )
                } else {
                    Source::Reference(dy as usize * stride + dx as usize)
                };
                self.warp_into(
                    target,
                    plane,
                    source,
                    reference,
                    (x as usize, y as usize),
                    shear,
                    (mx, my),
                    false,
                );
            }
        }
    }

    /// Warp by 4x4 sub-blocks, each positioned by the model at its centre,
    /// reading within a window around its 8x8 group (dav2d `ext_warp`).
    fn ext_warp(
        &mut self,
        target: Target,
        plane: Plane,
        (w, h): (i32, i32),
        (bx, by): (u32, u32),
        reference: usize,
        model: &WarpModel,
    ) {
        let shift = plane.shift();
        let m = model.matrix.map(i64::from);
        let width = ((self.geometry.columns << UNIT_LOG2) >> shift) as i32;
        let height = ((self.geometry.rows << UNIT_LOG2) >> shift) as i32;
        let (sw, sh) = (w.min(WARP_BLOCK), h.min(WARP_BLOCK));
        let (hsw, hsh) = (sw >> 1, sh >> 1);
        let (bx, by) = (bx as i32 * 4, by as i32 * 4);
        let window = (EXT_WARP_BLOCK + TAPS_BEFORE + TAPS_AFTER) as usize;
        for y in (0..h).step_by(sh as usize) {
            let src_y = i64::from(by + ((y + hsh) << shift));
            let (m3, m5) = (m[3] * src_y + m[0], m[5] * src_y + m[1]);
            for x in (0..w).step_by(sw as usize) {
                let src_x = i64::from(bx + ((x + hsw) << shift));
                let mvx = (m[2] * src_x + m3) >> shift;
                let mvy = (m[4] * src_x + m5) >> shift;
                let left_window = (mvx >> WARP_POSITION_BITS) as i32 - hsw - TAPS_BEFORE;
                let top_window = (mvy >> WARP_POSITION_BITS) as i32 - hsh - TAPS_BEFORE;
                let bounds = Bounds {
                    left: left_window.clamp(0, width - 1),
                    right: (left_window + sw + 7).clamp(1, width),
                    top: top_window.clamp(0, height - 1),
                    bottom: (top_window + sh + 7).clamp(1, height),
                };
                for yy in (y..y + sh).step_by(EXT_WARP_BLOCK as usize) {
                    let src_y = i64::from(by + ((yy + 2) << shift));
                    let (m3, m5) = (m[3] * src_y + m[0], m[5] * src_y + m[1]);
                    for xx in (x..x + sw).step_by(EXT_WARP_BLOCK as usize) {
                        let src_x = i64::from(bx + ((xx + 2) << shift));
                        let mvx = ((m[2] * src_x + m3) >> shift) + EXT_WARP_ROUND;
                        let mvy = ((m[4] * src_x + m5) >> shift) + EXT_WARP_ROUND;
                        let dx = (mvx >> WARP_POSITION_BITS) as i32 - 2;
                        let mx = ((mvx >> WARP_PHASE_BITS) & WARP_PHASE_MASK) as i32;
                        let dy = (mvy >> WARP_POSITION_BITS) as i32 - 2;
                        let my = ((mvy >> WARP_PHASE_BITS) & WARP_PHASE_MASK) as i32;
                        let picture = self.reference_picture(reference);
                        let (data, stride) = kernels::plane(picture, plane.index());
                        let source = if dx - TAPS_BEFORE < bounds.left
                            || dx + EXT_WARP_BLOCK + TAPS_AFTER > bounds.right
                            || dy - TAPS_BEFORE < bounds.top
                            || dy + EXT_WARP_BLOCK + TAPS_AFTER > bounds.bottom
                        {
                            let origin = bounds.top as usize * stride + bounds.left as usize;
                            nc_dsp::emulate_edges(
                                (&data[origin..], stride),
                                (
                                    (bounds.right - bounds.left) as usize,
                                    (bounds.bottom - bounds.top) as usize,
                                ),
                                (
                                    (dx - TAPS_BEFORE - bounds.left) as isize,
                                    (dy - TAPS_BEFORE - bounds.top) as isize,
                                ),
                                (window, window),
                                (&mut self.inter_scratch.window, kernels::WINDOW_STRIDE),
                            );
                            Source::Window(
                                TAPS_BEFORE as usize * kernels::WINDOW_STRIDE
                                    + TAPS_BEFORE as usize,
                            )
                        } else {
                            Source::Reference(dy as usize * stride + dx as usize)
                        };
                        self.warp_into(
                            target,
                            plane,
                            source,
                            reference,
                            (xx as usize, yy as usize),
                            model.shear,
                            (mx, my),
                            true,
                        );
                    }
                }
            }
        }
    }

    /// Run one warp sub-block (8x8, or a 4x4 `ext` one) into the target at
    /// (`x`, `y`) of the block.
    #[allow(clippy::too_many_arguments, reason = "the warp call's parts")]
    fn warp_into(
        &mut self,
        target: Target,
        plane: Plane,
        source: Source,
        reference: usize,
        (x, y): (usize, usize),
        shear: crate::general::warp::Shear,
        (mx, my): (i32, i32),
        ext: bool,
    ) {
        let picture = self.reference_picture(reference);
        let (reference_data, reference_stride) = kernels::plane(picture, plane.index());
        let (src, src_offset, src_stride) = match source {
            Source::Reference(offset) => (reference_data, offset, reference_stride),
            Source::Window(offset) => (
                &self.inter_scratch.window[..],
                offset,
                kernels::WINDOW_STRIDE,
            ),
        };
        let abcd = [shear.alpha, shear.beta, shear.gamma, shear.delta];
        match target {
            Target::Picture(offset) => {
                let stride = self.planes.stride[plane.index()];
                let dst = (
                    &mut self.planes.data[plane.index()][..],
                    offset + y * stride + x,
                    stride,
                );
                if ext {
                    kernels::ext_warp(dst, (src, src_offset, src_stride), (mx, my));
                } else {
                    kernels::warp(dst, (src, src_offset, src_stride), abcd, (mx, my));
                }
            }
            Target::Compound {
                index,
                offset,
                stride,
            } => {
                let dst = (
                    &mut self.inter_scratch.compound[index][..],
                    offset + y * stride + x,
                    stride,
                );
                if ext {
                    kernels::ext_warp_prep(dst, (src, src_offset, src_stride), (mx, my));
                } else {
                    kernels::warp_prep(dst, (src, src_offset, src_stride), abcd, (mx, my));
                }
            }
            Target::Refine { .. } => unreachable!("refinement windows are not warped"),
        }
    }
}
