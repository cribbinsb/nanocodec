//! Combining predictions: the compound blends (average, weighted average,
//! wedge and difference masks, the implicit mask of predictions reaching
//! past the frame), block-adaptive weights and inter-intra (dav2d
//! `recon_b`'s compound switch, `get_mask`, `bawp`, `iiblend`).
use super::kernels;
use super::{EQUAL_WEIGHT, InterPrediction, Plane, UNIT_LOG2};
use crate::general::block::Tile;
use crate::general::block::inter_record::CompoundType;
use crate::general::block::motion::Mv;
use crate::general::block::size::BlockSize;
use crate::general::sample::Av2Sample;
use nc_dsp::av2_dec::ipred::SampleRef;
use nc_dsp::av2_dec::ipred::{
    self, ANGLE_HAS_LEFT_FLAG, ANGLE_HAS_TOP_FLAG, ANGLE_IBP_FLAG, EdgeRequest, IntraPredMode,
};
use nc_dsp::av2_dec::mc;

/// The implicit mask's state for a compound prediction (dav2d `bacp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Implicit {
    /// Not used.
    Off,
    /// Allowed: the blend decides from the whole block's vectors.
    Pending,
    /// Built per sub-block by the prediction: blend with it.
    Ready,
}

/// A mask marker for one not yet initialised (dav2d
/// `DEFERRED_MASK_MARKER`), and the neutral weight it starts from.
pub(super) const DEFERRED_MASK: u8 = 0xff;
const NEUTRAL_MASK: u8 = 0x20;
/// The unscaled block-adaptive weight (8 fraction bits) and the offset
/// without edges.
const BAWP_UNIT: i32 = 256;
const BAWP_NO_EDGE_BETA: i32 = -128;
const BAWP_SHIFT: u32 = 8;
/// Explicit weights step by this, further for distant references.
const BAWP_EXPLICIT_STEP: i32 = 16;
const BAWP_FAR_DISTANCE: u8 = 4;
/// Edge samples of the weights' fit, as log2 counts, by whether both
/// edges exist, height and width class, edge (dav2d `n_edge_samples`).
const BAWP_EDGE_SAMPLES: [[[[u32; 2]; 3]; 3]; 2] = [
    [
        [[2, 2], [3, 2], [4, 2]],
        [[2, 3], [3, 3], [4, 3]],
        [[2, 4], [3, 4], [4, 4]],
    ],
    [
        [[2, 2], [2, 2], [4, 0]],
        [[2, 2], [3, 3], [3, 3]],
        [[0, 4], [3, 3], [4, 4]],
    ],
];
const BAWP_MAX_CLASS: u32 = 2;
/// Masks of luma pieces of large blocks are kept per 16x16 units; blocks
/// larger than this many units a side reuse their first piece's
/// block-adaptive weights (twice that on a chroma side outside lossless
/// frames).
const MASK_PIECE_UNITS: u32 = 16;
/// Inter-intra modes (dav2d `InterIntraPredMode`): DC, vertical,
/// horizontal, smooth, and the angle each predicts at.
const INTER_INTRA_SMOOTH: u8 = 3;
const INTER_INTRA_ANGLES: [i32; 4] = [0, 90, 180, 0];
/// Edge buffer positions of the intra prediction (as `pixels.rs`).
const EDGE_CORNER: usize = 128;
/// The largest mask row.
const MASK_MAX_STRIDE: usize = 64;

impl<T: Av2Sample> Tile<'_, '_, T> {
    /// Blend a compound block's two intermediates into the picture at
    /// `offset` of a `w` x `h` (plane samples) prediction of `size`.
    #[allow(
        clippy::many_single_char_names,
        reason = "dav2d's w, h, the plane p and the two predictions"
    )]
    pub(super) fn blend_compound(
        &mut self,
        block: &InterPrediction,
        plane: Plane,
        offset: usize,
        (w, h): (usize, usize),
        size: BlockSize,
        implicit: Implicit,
    ) {
        let inter = &block.inter;
        let p = plane.index();
        let stride = self.planes.stride[p];
        let shift = plane.shift();
        let mask_offset = self.difference_mask_offset(block, plane);
        let mask_stride = Self::difference_mask_stride(block);
        let depth = self.depth();
        let [first, second] = &self.inter_scratch.compound;
        let dst = (&mut self.planes.data[p][..], offset, stride);
        match inter.compound {
            CompoundType::Wedge => {
                let (bw4, bh4) = size.units();
                let luma = ((bw4 << UNIT_LOG2) as usize, (bh4 << UNIT_LOG2) as usize);
                let mask = kernels::wedge_mask(self.masks, luma, inter.wedge_index, shift == 1);
                let (a, b) = if inter.wedge_sign {
                    (second, first)
                } else {
                    (first, second)
                };
                kernels::mask(dst, (a, b), ((w, h), mask), depth);
            }
            CompoundType::Difference => {
                let (a, b) = if inter.mask_sign {
                    (second, first)
                } else {
                    (first, second)
                };
                if plane == Plane::Luma {
                    kernels::w_mask_420(
                        dst,
                        (a, b),
                        (w, h),
                        (&mut self.inter_scratch.mask[mask_offset..], mask_stride),
                        (inter.mask_sign, depth),
                    );
                } else {
                    let weights = &self.inter_scratch.mask[mask_offset..];
                    kernels::mask(dst, (a, b), ((w, h), weights), depth);
                }
            }
            CompoundType::None | CompoundType::Average => {
                if inter.weight != EQUAL_WEIGHT {
                    kernels::w_avg(dst, (first, second), ((w, h), inter.weight), depth);
                    return;
                }
                let use_mask = match implicit {
                    Implicit::Off => false,
                    Implicit::Ready => true,
                    Implicit::Pending => self.implicit_mask_pending(block, plane, (w, h)),
                };
                let [first, second] = &self.inter_scratch.compound;
                let dst = (&mut self.planes.data[p][..], offset, stride);
                if use_mask {
                    let weights = &self.inter_scratch.mask;
                    kernels::mask(dst, (first, second), ((w, h), weights), depth);
                } else {
                    kernels::avg(dst, (first, second), (w, h), depth);
                }
            }
        }
    }

    /// Build the implicit mask of a whole compound block from its two
    /// vectors when either prediction reaches past the frame (dav2d
    /// `recon_b`'s `bacp == 2`).
    fn implicit_mask_pending(
        &mut self,
        block: &InterPrediction,
        plane: Plane,
        (w, h): (usize, usize),
    ) -> bool {
        let shift = plane.shift();
        let (x4, y4) = if plane == Plane::Luma {
            (self.column, self.row)
        } else {
            (self.chroma_column >> 1, self.chroma_row >> 1)
        };
        self.implicit_mask(
            (w, (x4, 0), (y4, 0)),
            block.mv,
            (3 + shift, 3 + shift),
            ((w >> UNIT_LOG2) as u32, (h >> UNIT_LOG2) as u32),
            0,
            plane,
        )
    }

    /// The implicit mask of a `bw4` x `bh4` part at (`x4`, `y4`) of a
    /// block at (`bx4`, `by4`) (plane 4x4 units) with vectors `mv` of
    /// `bits` fraction bits: built where either prediction leaves the
    /// frame, initialising the whole mask (`deferred` samples) first
    /// (dav2d `get_mask`). Returns whether it was built.
    pub(super) fn implicit_mask(
        &mut self,
        (stride, (bx4, x4), (by4, y4)): (usize, (u32, u32), (u32, u32)),
        mv: [Mv; 2],
        (h_bits, v_bits): (u32, u32),
        (bw4, bh4): (u32, u32),
        deferred: usize,
        plane: Plane,
    ) -> bool {
        let shift = plane.shift();
        let frame_w = ((self.geometry.columns << UNIT_LOG2) >> shift) as i32;
        let frame_h = ((self.geometry.rows << UNIT_LOG2) >> shift) as i32;
        let x = ((bx4 + x4) << UNIT_LOG2) as i32;
        let y = ((by4 + y4) << UNIT_LOG2) as i32;
        let (x0, y0) = (x + (mv[0].x >> h_bits), y + (mv[0].y >> v_bits));
        let (x1, y1) = (x + (mv[1].x >> h_bits), y + (mv[1].y >> v_bits));
        let (w, h) = ((bw4 << UNIT_LOG2) as i32, (bh4 << UNIT_LOG2) as i32);
        if x0 < 0
            || x1 < 0
            || y0 < 0
            || y1 < 0
            || x0 + w >= frame_w
            || x1 + w >= frame_w
            || y0 + h >= frame_h
            || y1 + h >= frame_h
        {
            if deferred != 0 && self.inter_scratch.mask[0] == DEFERRED_MASK {
                self.inter_scratch.mask[..deferred].fill(NEUTRAL_MASK);
            }
            let start = ((y4 as usize * stride) + x4 as usize) << UNIT_LOG2;
            kernels::bacp_mask(
                (&mut self.inter_scratch.mask[start..], stride),
                (w as usize, h as usize),
                ((x0, y0), (x1, y1)),
                (frame_w, frame_h),
            );
            return true;
        }
        false
    }

    /// The row length of a block's difference mask (luma writes it at
    /// chroma resolution, at most 64).
    fn difference_mask_stride(block: &InterPrediction) -> usize {
        let (bw4, _) = block.size.units();
        (((bw4 << UNIT_LOG2) >> 1) as usize).min(MASK_MAX_STRIDE)
    }

    /// Where the current piece's part of the difference mask starts.
    fn difference_mask_offset(&self, block: &InterPrediction, plane: Plane) -> usize {
        let (bw4, bh4) = block.size.units();
        if bw4.min(bh4) < MASK_PIECE_UNITS {
            return 0;
        }
        let stride = Self::difference_mask_stride(block);
        let (x, y) = if plane == Plane::Luma {
            (self.column >> 1, self.row >> 1)
        } else {
            (self.chroma_column >> 1, self.chroma_row >> 1)
        };
        let piece = |v: u32| ((v & (MASK_PIECE_UNITS - 1)) << UNIT_LOG2) as usize;
        piece(y) * stride + piece(x)
    }

    /// Scale a prediction by a linear model of its reference's edges
    /// against the block's own (dav2d `bawp`), at the current position of
    /// `plane`; `kind` is the luma weight's kind (1 for chroma).
    #[allow(clippy::too_many_lines, reason = "dav2d bawp")]
    pub(super) fn bawp(
        &mut self,
        block: &InterPrediction,
        plane: Plane,
        (bw4, bh4): (u32, u32),
        (w4, h4): (u32, u32),
        kind: u8,
    ) {
        let depth = self.depth();
        let p = plane.index();
        let shift = plane.shift();
        let mul = 4 >> shift;
        let lossless = self.lossless();
        let (bx, by) = if plane == Plane::Luma {
            (self.column, self.row)
        } else {
            (self.chroma_column, self.chroma_row)
        };
        let (sb_w4, sb_h4) = block.size.units();
        let piece_limit = MASK_PIECE_UNITS << u32::from(shift == 1 && !lossless);
        let offset = self.planes.offset(p, by >> shift, bx >> shift);
        let stride = self.planes.stride[p];
        let size = ((bw4 * mul) as usize, (bh4 * mul) as usize);
        if (sb_w4 > piece_limit && bx & (sb_w4 - 1) != 0)
            || (sb_h4 > piece_limit && by & (sb_h4 - 1) != 0)
        {
            let (alpha, beta) = self.inter_scratch.bawp[p];
            if alpha != BAWP_UNIT || beta != 0 {
                mc::morph(
                    &mut self.planes.data[p],
                    offset,
                    stride,
                    (alpha, beta),
                    size,
                    depth,
                );
            }
            return;
        }
        self.inter_scratch.bawp[p] = (BAWP_UNIT, 0);
        let reference = block.inter.references[0] as usize;
        let mv = block.mv[0];
        let whole = |v: i32, s: u32| (v + 3 + i32::from(v >= 0)) >> (3 + s);
        let ref_x = (bx * mul) as i32 + whole(mv.x, shift);
        let ref_y = (by * mul) as i32 + whole(mv.y, shift);
        let sb_w = sb_w4.min(self.geometry.columns - bx) * mul;
        let sb_h = sb_h4.min(self.geometry.rows - by) * mul;
        let (right, bottom) = (
            (self.geometry.columns * mul) as i32,
            (self.geometry.rows * mul) as i32,
        );
        if ref_y + sb_h as i32 > bottom
            || ref_x + sb_w as i32 > right
            || ref_y - 1 < 0
            || ref_x - 1 < 0
        {
            return;
        }
        let have_left = bx > self.bounds.column_start;
        let have_above = by > self.bounds.row_start;
        let lw4 = w4.ilog2().min(BAWP_MAX_CLASS) - shift;
        let lh4 = h4.ilog2().min(BAWP_MAX_CLASS) - shift;
        let counts =
            BAWP_EDGE_SAMPLES[usize::from(have_above && have_left)][lh4 as usize][lw4 as usize];
        let above_log2 = if have_above { counts[0] } else { 0 };
        let left_log2 = if have_left { counts[1] } else { 0 };
        let count_log2 = above_log2
            + if above_log2 == left_log2 {
                u32::from(above_log2 != 0)
            } else {
                left_log2
            };
        let picture = self.reference_picture(reference);
        let (ref_data, ref_stride) = kernels::plane(picture, p);
        let at_ref = |x: i32, y: i32| {
            ref_data[(ref_y + y) as usize * ref_stride + (ref_x + x) as usize].to_i32()
        };
        let own = &self.planes.data[p];
        let at_own =
            |x: isize, y: isize| own[(offset as isize + y * stride as isize + x) as usize].to_i32();
        let (mut sum_x, mut sum_y, mut sum_xy, mut sum_xx) = (0, 0, 0, 0);
        let mut add = |source: i32, target: i32| {
            sum_x += source;
            sum_y += target;
            sum_xy += source * target;
            sum_xx += source * source;
        };
        if above_log2 != 0 {
            let width = 4 << lw4;
            let step = width >> above_log2;
            for i in (step >> 1..width).step_by(step as usize) {
                add(at_ref(i, -1), at_own(i as isize, -1));
            }
        }
        if left_log2 != 0 {
            let height = 4 << lh4;
            let step = height >> left_log2;
            for i in (step >> 1..height).step_by(step as usize) {
                add(at_ref(-1, i), at_own(-1, i as isize));
            }
        }
        let alpha = if plane != Plane::Luma {
            if have_left || have_above {
                self.inter_scratch.bawp[0].0
            } else {
                BAWP_UNIT
            }
        } else if kind != 1 {
            let far = self.references.abs_distance[reference] > BAWP_FAR_DISTANCE;
            let magnitude = 1 + i32::from(kind >> 2) + i32::from(far);
            let sign = if kind & 1 != 0 { 1 } else { -1 };
            BAWP_UNIT + BAWP_EXPLICIT_STEP * magnitude * sign
        } else if count_log2 != 0 {
            let product = |a: i32, b: i32| ((i64::from(a) * i64::from(b)) >> count_log2) as i32;
            let num = sum_xy - product(sum_x, sum_y);
            let den = sum_xx - product(sum_x, sum_x);
            ipred::derive_alpha(num, den, BAWP_UNIT)
        } else {
            BAWP_UNIT
        };
        let beta = if count_log2 != 0 {
            let diff = (sum_y << BAWP_SHIFT) - sum_x * alpha;
            (diff.abs() >> count_log2) * diff.signum()
        } else {
            BAWP_NO_EDGE_BETA
        };
        self.inter_scratch.bawp[p] = (alpha, beta);
        mc::morph(
            &mut self.planes.data[p],
            offset,
            stride,
            (alpha, beta),
            size,
            depth,
        );
    }

    /// Blend an intra prediction from the block's own edges into its
    /// inter prediction (dav2d `iiblend`) by a smooth-weighted or wedge
    /// mask; (`bw4`, `bh4`) the block in luma units.
    #[allow(clippy::too_many_lines, reason = "dav2d iiblend")]
    pub(super) fn blend_inter_intra(
        &mut self,
        block: &InterPrediction,
        plane: Plane,
        (bw4, bh4): (u32, u32),
        (bx, by): (u32, u32),
    ) {
        let depth = self.depth();
        let inter = &block.inter;
        let p = plane.index();
        let shift = plane.shift();
        let chroma = shift as usize;
        let mode = if inter.inter_intra_mode == INTER_INTRA_SMOOTH {
            IntraPredMode::Smooth
        } else {
            IntraPredMode::from_index(inter.inter_intra_mode).unwrap_or(IntraPredMode::Dc)
        };
        let angle = INTER_INTRA_ANGLES[usize::from(inter.inter_intra_mode)];
        let (mut n_tr, mut n_bl) = (0, 0);
        let superblock = self.geometry.superblock_units;
        let b = self.bounds;
        if mode == IntraPredMode::Smooth {
            let (bx4, by4) = (bx & 63, by & 63);
            if by > b.row_start {
                let w = bw4.min(b.column_end.saturating_sub(bx + bw4)) as i32;
                if by & (superblock - 1) == 0 {
                    n_tr = w;
                } else {
                    let end = ((bx + superblock) & !(superblock - 1)).min(b.column_end) as i32;
                    let w = w.min(end - self.column as i32 - bw4 as i32);
                    if w > 0 {
                        // Smooth prediction reads one sample past the block.
                        let line = self.coded[chroma][((by4 >> shift) - 1) as usize];
                        n_tr = ((line >> ((bx4 + bw4) >> shift)) & 1) as i32;
                    }
                }
            }
            if bx > b.column_start {
                let end = ((by + superblock) & !(superblock - 1)).min(b.row_end) as i32;
                let h = (bh4 as i32).min(end - by as i32 - bh4 as i32);
                if h > 0 {
                    n_bl = if bx & (superblock - 1) == 0 {
                        h
                    } else {
                        let line = self.coded[chroma][((by4 + bh4) >> shift) as usize];
                        ((line >> ((bx4 - 1) >> shift)) & 1) as i32
                    };
                }
            }
        }
        let (ssbw4, ssbh4) = (bw4 >> shift, bh4 >> shift);
        let ibp = self.sequence.intra.ibp && ssbw4.max(ssbh4) > 1;
        let flags = if ibp { ANGLE_IBP_FLAG } else { 0 }
            | if bx > b.column_start {
                ANGLE_HAS_LEFT_FLAG
            } else {
                0
            }
            | if by > b.row_start {
                ANGLE_HAS_TOP_FLAG
            } else {
                0
            };
        let request = EdgeRequest {
            x: (bx >> shift) as i32,
            y: (by >> shift) as i32,
            end_x: (b.column_end >> shift) as i32,
            end_y: (b.row_end >> shift) as i32,
            n_tr,
            n_bl,
            mode,
            tw4: ssbw4 as i32,
            th4: ssbh4 as i32,
            intra_flags: angle | flags,
        };
        let offset = self.planes.offset(p, by >> shift, bx >> shift);
        let stride = self.planes.stride[p];
        let top_sb_edge = by & (superblock - 1) == 0 && by > 0;
        let data = &self.planes.data[p];
        // The row above a superblock is the picture's own before the loop
        // filters (they run after the tile).
        let row_above = top_sb_edge.then(|| {
            SampleRef::new(
                data,
                offset - stride - ((bx >> shift) << UNIT_LOG2) as usize,
            )
        });
        let resolved = ipred::prepare_intra_edges(
            &request,
            data,
            offset,
            stride,
            row_above,
            &mut self.edge,
            EDGE_CORNER,
            depth,
        );
        let (w, h) = ((ssbw4 << UNIT_LOG2) as usize, (ssbh4 << UNIT_LOG2) as usize);
        ipred::predict(
            resolved,
            &mut self.inter_scratch.intra,
            0,
            w,
            &self.edge,
            EDGE_CORNER,
            w,
            h,
            flags,
            0,
            0,
            depth,
        );
        let mask = if inter.wedge_index < 0 {
            kernels::inter_intra_mask(self.masks, (w, h), inter.inter_intra_mode)
        } else {
            let luma = ((bw4 << UNIT_LOG2) as usize, (bh4 << UNIT_LOG2) as usize);
            kernels::wedge_mask(self.masks, luma, inter.wedge_index, shift == 1)
        };
        kernels::blend(
            (&mut self.planes.data[p], offset, stride),
            &self.inter_scratch.intra,
            (w, h),
            mask,
        );
    }
}
