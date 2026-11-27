//! Inter prediction of P and B macroblocks (§8.4.2): fractional sample
//! interpolation through the nc-dsp H.264 kernels, bi-prediction and
//! default, explicit and implicit weighted prediction (§8.4.2.3), for every
//! chroma format: 4:2:0 chroma at eighth samples on both axes (the hot
//! path), 4:2:2 at eighth samples across and quarter samples down (eq.
//! 8-229: `mvCLX[1]` in units of `1 / (4 · SubHeightC)`), 4:4:4 chroma
//! interpolated as luma (§8.4.2.2).
//!
//! A reference field is every other row of its frame buffer. Its padded
//! border is the frame's (an edge extension of frame rows), which is right
//! beside each row but not above or below a field, so a field's windows
//! that leave its rows vertically are copied clamped to the field
//! (§8.4.2.2.1).
//!
//! Interpolation is a per-sample process, so a partition is predicted as
//! any tiling of square blocks with its vector: the whole macroblock when
//! all its vectors and references agree, else each 8x8 block whole or as
//! four 4x4 blocks. The prediction is written straight into the picture,
//! where the residual is then added.

use nc_common::{BitDepth, BitDepths, Plane, PlaneIndex, PlaneView, PlaneViewMut, Sample};
use nc_dsp::{BlockSize, Dsp, Filter, H264Predictor};
use nc_encoder::Mv;

use super::chroma::ChromaArray;
use super::dpb::RefPicture;
use super::mbstate::{LISTS, MbKind, MbState, Partition};
use super::sample::{H264Sample, McKernel};
use super::slice::{Weight, WeightTable};
use super::structure::{Parity, Structure};
use super::{DecodeError, Planes, Result};

/// Fractional bits of a luma vector (quarter samples) and of the same
/// vector read as a 4:2:0 chroma vector (eighth samples, eq. 8-229).
const LUMA_FRAC_BITS: i32 = 2;
const CHROMA_FRAC_BITS: i32 = 3;
/// Luma block shapes (width, height) predicted with one kernel call: the
/// macroblock, its 16x8 and 8x16 partitions, 8x8 and 4x4 blocks. Chroma
/// blocks are half as wide and high (a 2x2 chroma block is predicted as
/// the corner of a 4x4 one).
const SHAPES: [(usize, usize); 5] = [(16, 16), (16, 8), (8, 16), (8, 8), (4, 4)];
const SHAPE_16X16: usize = 0;
const SHAPE_16X8: usize = 1;
const SHAPE_8X16: usize = 2;
const SHAPE_8X8: usize = 3;
const SHAPE_4X4: usize = 4;
/// Largest interpolation window: 16 + 6 - 1 samples square.
const MAX_WINDOW: usize = 21;
/// Smallest kernel block side.
const MIN_KERNEL_SIDE: usize = 4;

/// The interpolation kernels of every block shape, per component.
#[derive(Debug, Clone)]
pub(super) struct InterKernels {
    luma: [H264Predictor; SHAPES.len()],
    chroma: [H264Predictor; SHAPES.len()],
    /// 4:2:2 chroma: half the width, the full height.
    chroma_422: [H264Predictor; SHAPES.len()],
    /// The kernels' block sizes.
    luma_sizes: [BlockSize; SHAPES.len()],
    chroma_sizes: [BlockSize; SHAPES.len()],
    chroma_422_sizes: [BlockSize; SHAPES.len()],
}

impl InterKernels {
    pub fn new(dsp: &Dsp) -> Self {
        let size = |(w, h): (usize, usize)| BlockSize::new(w, h).expect("kernel size");
        let chroma_side = |side: usize| (side / 2).max(MIN_KERNEL_SIDE);
        let luma_sizes = SHAPES.map(size);
        let chroma_sizes = SHAPES.map(|(w, h)| size((chroma_side(w), chroma_side(h))));
        let chroma_422_sizes = SHAPES.map(|(w, h)| size((chroma_side(w), h)));
        Self {
            luma: luma_sizes.map(|s| dsp.h264_predictor(Filter::H264Luma, s)),
            chroma: chroma_sizes.map(|s| dsp.h264_predictor(Filter::H264Chroma, s)),
            chroma_422: chroma_422_sizes.map(|s| dsp.h264_predictor(Filter::H264Chroma, s)),
            luma_sizes,
            chroma_sizes,
            chroma_422_sizes,
        }
    }

    fn chroma_422(&self, shape: usize) -> McKernel<'_> {
        McKernel {
            predictor: &self.chroma_422[shape],
            filter: Filter::H264Chroma,
            size: self.chroma_422_sizes[shape],
        }
    }

    fn luma(&self, shape: usize) -> McKernel<'_> {
        McKernel {
            predictor: &self.luma[shape],
            filter: Filter::H264Luma,
            size: self.luma_sizes[shape],
        }
    }

    fn chroma(&self, shape: usize) -> McKernel<'_> {
        McKernel {
            predictor: &self.chroma[shape],
            filter: Filter::H264Chroma,
            size: self.chroma_sizes[shape],
        }
    }
}

/// One reference plane as the kernels read it: its padded samples and
/// the position of visible sample (0, 0) in them.
#[derive(Clone, Copy)]
struct Source<'a, T: Sample> {
    padded: PlaneView<'a, T>,
    border: (usize, usize),
    /// The visible samples, for windows beyond the padding.
    visible: PlaneView<'a, T>,
}

impl<'a, T: Sample> Source<'a, T> {
    /// The picture `structure` of a frame's `plane`: the plane, or one
    /// field's rows with the padding beside them only.
    fn new(plane: &'a Plane<T>, structure: Structure) -> Self {
        let Structure::Field(parity) = structure else {
            return Self {
                padded: plane.padded_view(),
                border: (plane.border_x(), plane.border_y()),
                visible: plane.view(),
            };
        };
        let border_x = plane.border_x();
        let rows = plane
            .region(
                -(border_x as isize),
                0,
                plane.width() + 2 * border_x,
                plane.height(),
            )
            .expect("the rows with their side padding lie inside the plane");
        Self {
            padded: rows.every_other_row(parity.index()),
            border: (border_x, 0),
            visible: plane.view().every_other_row(parity.index()),
        }
    }
}

/// One list entry's picture as prediction reads it.
#[derive(Clone, Copy)]
struct Reference<'a, T: Sample> {
    /// Y, Cb and Cr.
    planes: [Source<'a, T>; 3],
    /// Vertical offset of chroma vectors in eighth samples (Table 8-10):
    /// between fields of opposite parity, the half-row shift of their
    /// chroma sample positions.
    chroma_dy: i16,
}

/// Table 8-10: the chroma vector offset of a `current` picture predicting
/// from a `reference` one.
fn chroma_offset(current: Structure, reference: Structure) -> i16 {
    match (current.parity(), reference.parity()) {
        (Some(Parity::Top), Some(Parity::Bottom)) => -2,
        (Some(Parity::Bottom), Some(Parity::Top)) => 2,
        _ => 0,
    }
}

/// `mv` with its vertical component moved by `dy` (chroma of fields).
#[inline]
fn chroma_mv(mv: Mv, dy: i16) -> Mv {
    Mv {
        x: mv.x,
        y: mv.y.wrapping_add(dy),
    }
}

/// The weighting of a slice's predictions (§8.4.2.3).
pub(super) enum Weighting<'a> {
    /// No weights: single predictions as they are, bi-predictions averaged.
    Default,
    /// `pred_weight_table()` (P slices with `weighted_pred_flag`, B slices
    /// with `weighted_bipred_idc` 1).
    Explicit(&'a WeightTable),
    /// `weighted_bipred_idc` 2: bi-prediction weights `(w0, w1)` of each
    /// pair of list entries (`[ref_idx_l0][ref_idx_l1]`, §8.4.2.3.1);
    /// single predictions unweighted.
    Implicit(Vec<Vec<[i32; 2]>>),
}

/// Weight denominator (`logWD`) of implicit weights (eq. 8-305).
pub(super) const IMPLICIT_LOG2_DENOM: u32 = 5;

/// The motion of one prediction block: per list, its vector and reference
/// index (`None`: the list is not used).
type BlockMotion = [Option<(Mv, i8)>; LISTS];

/// The reference lists of a slice and its weights.
pub(super) struct InterContext<'a, T: Sample> {
    kernels: &'a InterKernels,
    dsp: &'a Dsp,
    /// Each list entry's picture (a separately coded colour plane's own
    /// plane first).
    refs: [Vec<Option<Reference<'a, T>>>; LISTS],
    weighting: Weighting<'a>,
    /// `ChromaArrayType`.
    chroma: ChromaArray,
    depths: BitDepths,
}

/// The second prediction of a bi-predicted block (luma 16x16, chroma 8x8
/// at most), before it is combined with the first in the picture.
struct Scratch<T> {
    luma: [T; LUMA_SCRATCH * LUMA_SCRATCH],
    chroma: [[T; CHROMA_SCRATCH * CHROMA_SCRATCH]; 2],
}

/// Strides of [`Scratch`]: the largest blocks.
const LUMA_SCRATCH: usize = 16;
const CHROMA_SCRATCH: usize = 8;

/// A block of one component to predict.
#[derive(Clone, Copy)]
struct Target<'k> {
    kernel: McKernel<'k>,
    dsp: &'k Dsp,
    /// The component's bit depth.
    depth: BitDepth,
    /// Position and size in the component's samples.
    pos: (usize, usize),
    size: (usize, usize),
}

/// The sample grid of one component's interpolation: fractional bits of
/// its vectors, filter taps and the taps before the centre sample (as
/// constants, so the window arithmetic folds in each instantiation).
trait Grid {
    const FRAC_BITS: i32;
    /// Fractional bits of the vertical component, if fewer, and the shift
    /// that brings its fraction to the kernel's units.
    const FRAC_BITS_Y: i32 = Self::FRAC_BITS;
    const FRAC_SHIFT_Y: u32 = 0;
    const TAPS: usize;
    const BEFORE: usize;
}

/// Luma: quarter samples, the 6-tap filter (§8.4.2.2.1).
struct LumaGrid;
impl Grid for LumaGrid {
    const FRAC_BITS: i32 = LUMA_FRAC_BITS;
    const TAPS: usize = 6;
    const BEFORE: usize = 2;
}

/// 4:2:0 chroma: eighth samples, bilinear (§8.4.2.2.2).
struct ChromaGrid;
impl Grid for ChromaGrid {
    const FRAC_BITS: i32 = CHROMA_FRAC_BITS;
    const TAPS: usize = 2;
    const BEFORE: usize = 0;
}

/// 4:2:2 chroma: eighth samples across, quarter samples down (the full
/// vertical resolution: eq. 8-229 with `SubHeightC` 1), each read by the
/// bilinear kernel in eighths (§8.4.2.2.2: `yFracC` is `(mvCLX[1] & 3) << 1`).
struct Chroma422Grid;
impl Grid for Chroma422Grid {
    const FRAC_BITS: i32 = CHROMA_FRAC_BITS;
    const FRAC_BITS_Y: i32 = LUMA_FRAC_BITS;
    const FRAC_SHIFT_Y: u32 = (CHROMA_FRAC_BITS - LUMA_FRAC_BITS) as u32;
    const TAPS: usize = 2;
    const BEFORE: usize = 0;
}

impl<'a, T: H264Sample> InterContext<'a, T> {
    /// The prediction of a slice of picture `current` from `refs`, of
    /// chroma format `chroma`; `colour_plane` is the `colour_plane_id` of
    /// separately coded planes (its reference plane predicts the decoded
    /// one), else 0.
    pub fn new(
        (kernels, dsp): (&'a InterKernels, &'a Dsp),
        (refs, current): ([&'a [Option<RefPicture<T>>]; LISTS], Structure),
        weighting: Weighting<'a>,
        (chroma, colour_plane): (ChromaArray, usize),
        depths: BitDepths,
    ) -> Self {
        let planes = |list: &'a [Option<RefPicture<T>>]| {
            list.iter()
                .map(|r| {
                    r.as_ref().map(|r| {
                        let mut planes =
                            PlaneIndex::ALL.map(|p| Source::new(r.picture.plane(p), r.structure));
                        planes.rotate_left(colour_plane);
                        Reference {
                            planes,
                            chroma_dy: chroma_offset(current, r.structure),
                        }
                    })
                })
                .collect()
        };
        Self {
            kernels,
            dsp,
            refs: refs.map(planes),
            weighting,
            chroma,
            // Folds to the 8-bit constants in the 8-bit instance.
            depths: depths.of::<T>(),
        }
    }

    /// Predict the inter macroblock `mb` at macroblock position (`mx`,
    /// `my`).
    #[allow(
        clippy::inline_always,
        reason = "measured: out of line from `reconstruct` (gate build) it cost \
                  about 2 % of the instructions of a 720p P stream (callgrind)"
    )]
    #[inline(always)]
    pub fn predict(
        &self,
        dst: &mut Planes<'_, T>,
        mb: &MbState,
        mx: usize,
        my: usize,
    ) -> Result<()> {
        // One instance for list-0-only macroblocks (all of a P slice), with
        // list 1 folded away, one for the others.
        if mb.uses_l1() {
            self.predict_mb::<true>(dst, mb, mx, my)
        } else {
            self.predict_mb::<false>(dst, mb, mx, my)
        }
    }

    /// [`InterContext::predict`] of a macroblock that predicts from list 1
    /// (`L1`) or not.
    #[allow(
        clippy::inline_always,
        reason = "measured: left out of `reconstruct` by the inliner (gate \
                  build), the P instance cost about 1.8 % of the instructions of \
                  a 720p P stream (callgrind)"
    )]
    #[inline(always)]
    fn predict_mb<const L1: bool>(
        &self,
        dst: &mut Planes<'_, T>,
        mb: &MbState,
        mx: usize,
        my: usize,
    ) -> Result<()> {
        let origin = (mx * 16, my * 16);
        let lists = if L1 { LISTS } else { 1 };
        if mb.single_motion {
            return self.mb_block::<L1>(dst, mb, (0, 0), origin, SHAPE_16X16);
        }
        // 16x8 and 8x16 partitions have one motion each: one call per
        // partition (4x4 block 8 starts the lower half, 2 the right half).
        match mb.kind {
            MbKind::Inter(Partition::P16x8) => {
                self.mb_block::<L1>(dst, mb, (0, 0), origin, SHAPE_16X8)?;
                let lower = (origin.0, origin.1 + 8);
                return self.mb_block::<L1>(dst, mb, (8, 2), lower, SHAPE_16X8);
            }
            MbKind::Inter(Partition::P8x16) => {
                self.mb_block::<L1>(dst, mb, (0, 0), origin, SHAPE_8X16)?;
                let right = (origin.0 + 8, origin.1);
                return self.mb_block::<L1>(dst, mb, (2, 1), right, SHAPE_8X16);
            }
            _ => {}
        }
        for b8 in 0..4 {
            let (x4, y4) = ((b8 % 2) * 2, (b8 / 2) * 2);
            let corner = y4 * 4 + x4;
            let same = [(1, 0), (0, 1), (1, 1)].iter().all(|&(dx, dy)| {
                let block = (y4 + dy) * 4 + x4 + dx;
                (0..lists).all(|list| mb.mvs[list][block] == mb.mvs[list][corner])
            });
            if same {
                let pos = (mx * 16 + x4 * 4, my * 16 + y4 * 4);
                self.mb_block::<L1>(dst, mb, (corner, b8), pos, SHAPE_8X8)?;
                continue;
            }
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let block = (y4 + dy) * 4 + x4 + dx;
                let pos = (mx * 16 + (x4 + dx) * 4, my * 16 + (y4 + dy) * 4);
                self.mb_block::<L1>(dst, mb, (block, b8), pos, SHAPE_4X4)?;
            }
        }
        Ok(())
    }

    /// The block of `shape` at `pos` of `mb`, with the motion of its 4x4
    /// block `block` in 8x8 block `b8`.
    #[allow(
        clippy::inline_always,
        reason = "measured: keeps the P path at the base instruction count (callgrind)"
    )]
    #[inline(always)]
    fn mb_block<const L1: bool>(
        &self,
        dst: &mut Planes<'_, T>,
        mb: &MbState,
        (block, b8): (usize, usize),
        pos: (usize, usize),
        shape: usize,
    ) -> Result<()> {
        if !L1 {
            // Every block of a macroblock without list 1 uses list 0.
            let motion = (mb.mvs[0][block], mb.ref_idx[0][b8]);
            return self.single::<0>(dst, motion, pos, shape);
        }
        let motion = core::array::from_fn(|list| {
            let ref_idx = mb.ref_idx[list][b8];
            (ref_idx >= 0).then(|| (mb.mvs[list][block], ref_idx))
        });
        self.block(dst, motion, pos, shape)
    }

    /// The picture of entry `ref_idx` of list `list`.
    fn reference(&self, list: usize, ref_idx: i8) -> Result<&Reference<'a, T>> {
        usize::try_from(ref_idx)
            .ok()
            .and_then(|r| self.refs[list].get(r).and_then(Option::as_ref))
            .ok_or(DecodeError::Malformed("inter prediction from no picture"))
    }

    /// One luma block of shape `SHAPES[shape]` at (`x`, `y`) and its
    /// chroma, predicted with `motion` into the picture planes `dst`.
    #[allow(
        clippy::inline_always,
        reason = "measured: keeps the P path at the base instruction count (callgrind)"
    )]
    #[inline(always)]
    fn block(
        &self,
        dst: &mut Planes<'_, T>,
        motion: BlockMotion,
        pos: (usize, usize),
        shape: usize,
    ) -> Result<()> {
        match motion {
            [Some(m), None] => self.single::<0>(dst, m, pos, shape),
            [None, Some(m)] => self.single::<1>(dst, m, pos, shape),
            [Some(m0), Some(m1)] => self.bi(dst, m0, m1, pos, shape),
            [None, None] => Err(DecodeError::Malformed("inter block without a list")),
        }
    }

    /// The luma and chroma targets of a block.
    #[allow(
        clippy::inline_always,
        reason = "measured: keeps the P path at the base instruction count (callgrind)"
    )]
    #[inline(always)]
    fn targets(&self, pos: (usize, usize), shape: usize) -> (Target<'_>, Target<'_>) {
        (self.luma_target(pos, shape), self.chroma_target(pos, shape))
    }

    /// The luma block of `shape` at `pos`.
    #[allow(
        clippy::inline_always,
        reason = "measured: keeps the P path at the base instruction count (callgrind)"
    )]
    #[inline(always)]
    fn luma_target(&self, pos: (usize, usize), shape: usize) -> Target<'_> {
        Target {
            kernel: self.kernels.luma(shape),
            dsp: self.dsp,
            depth: self.depths.luma,
            pos,
            size: SHAPES[shape],
        }
    }

    /// The 4:2:0 chroma block of the luma block of `shape` at `pos`.
    #[allow(
        clippy::inline_always,
        reason = "measured: keeps the P path at the base instruction count (callgrind)"
    )]
    #[inline(always)]
    fn chroma_target(&self, (x, y): (usize, usize), shape: usize) -> Target<'_> {
        let (w, h) = SHAPES[shape];
        Target {
            kernel: self.kernels.chroma(shape),
            dsp: self.dsp,
            depth: self.depths.chroma,
            pos: (x / 2, y / 2),
            size: (w / 2, h / 2),
        }
    }

    /// A block predicted from one list (`LIST`, its vector and reference
    /// index), explicitly weighted where the slice says so (eq. 8-300).
    fn single<const LIST: usize>(
        &self,
        dst: &mut Planes<'_, T>,
        (mv, ref_idx): (Mv, i8),
        pos: (usize, usize),
        shape: usize,
    ) -> Result<()> {
        let reference = self.reference(LIST, ref_idx)?;
        let sources = &reference.planes;
        let weight = match &self.weighting {
            Weighting::Explicit(table) if !table.is_identity(LIST, ref_idx as usize) => {
                Some((*table, table.weights[LIST][ref_idx as usize]))
            }
            _ => None,
        };
        let luma = self.luma_target(pos, shape);
        let [luma_dst, cb_dst, cr_dst] = dst;
        interpolate::<LumaGrid, T>(&luma, &sources[0], luma_dst, mv);
        if let Some((table, weights)) = weight {
            apply_weight(luma_dst, &luma, weights[0], table.luma_log2_denom);
        }
        match self.chroma {
            ChromaArray::Yuv420 => {}
            ChromaArray::Mono => return Ok(()),
            ChromaArray::Yuv422 | ChromaArray::Yuv444 => {
                self.single_chroma([cb_dst, cr_dst], sources, (mv, pos, shape), weight);
                return Ok(());
            }
        }
        let chroma = self.chroma_target(pos, shape);
        let mv = chroma_mv(mv, reference.chroma_dy);
        if shape == SHAPE_4X4 {
            for (source, plane_dst) in [(&sources[1], &mut *cb_dst), (&sources[2], &mut *cr_dst)] {
                corner_block::<ChromaGrid, T>(&chroma, source, plane_dst, mv);
            }
        } else {
            interpolate_pair(&chroma, [&sources[1], &sources[2]], [cb_dst, cr_dst], mv);
        }
        if let Some((table, weights)) = weight {
            for (comp, plane_dst) in [cb_dst, cr_dst].into_iter().enumerate() {
                apply_weight(
                    plane_dst,
                    &chroma,
                    weights[1 + comp],
                    table.chroma_log2_denom,
                );
            }
        }
        Ok(())
    }

    /// A bi-predicted block: list 0 into the picture, list 1 into a
    /// scratch, combined by the slice's weighting (eqs. 8-273, 8-301).
    fn bi(
        &self,
        dst: &mut Planes<'_, T>,
        (mv0, ref0): (Mv, i8),
        (mv1, ref1): (Mv, i8),
        pos: (usize, usize),
        shape: usize,
    ) -> Result<()> {
        let references = [self.reference(0, ref0)?, self.reference(1, ref1)?];
        let sources = references.map(|r| &r.planes);
        let chroma_mvs = [
            chroma_mv(mv0, references[0].chroma_dy),
            chroma_mv(mv1, references[1].chroma_dy),
        ];
        let weights = self.bi_weights(ref0 as usize, ref1 as usize);
        let (luma, chroma) = self.targets(pos, shape);
        let mut scratch = Scratch {
            luma: [T::default(); LUMA_SCRATCH * LUMA_SCRATCH],
            chroma: [[T::default(); CHROMA_SCRATCH * CHROMA_SCRATCH]; 2],
        };
        let [luma_dst, cb_dst, cr_dst] = dst;
        interpolate::<LumaGrid, T>(&luma, &sources[0][0], luma_dst, mv0);
        predict::<LumaGrid, T>(&luma, &sources[1][0], mv1, &mut scratch.luma, LUMA_SCRATCH);
        combine(luma_dst, &luma, &scratch.luma, LUMA_SCRATCH, weights[0]);
        match self.chroma {
            ChromaArray::Yuv420 => {}
            ChromaArray::Mono => return Ok(()),
            ChromaArray::Yuv422 | ChromaArray::Yuv444 => {
                self.bi_chroma([cb_dst, cr_dst], sources, ([mv0, mv1], pos, shape), weights);
                return Ok(());
            }
        }
        // The smallest kernel predicts 4x4 chroma; 2x2 blocks use its corner.
        let kernel_target = Target {
            size: (
                chroma.size.0.max(MIN_KERNEL_SIDE),
                chroma.size.1.max(MIN_KERNEL_SIDE),
            ),
            ..chroma
        };
        for (comp, plane_dst) in [cb_dst, cr_dst].into_iter().enumerate() {
            if shape == SHAPE_4X4 {
                corner_block::<ChromaGrid, T>(
                    &chroma,
                    &sources[0][1 + comp],
                    plane_dst,
                    chroma_mvs[0],
                );
            } else {
                interpolate::<ChromaGrid, T>(
                    &chroma,
                    &sources[0][1 + comp],
                    plane_dst,
                    chroma_mvs[0],
                );
            }
            let out = &mut scratch.chroma[comp];
            predict::<ChromaGrid, T>(
                &kernel_target,
                &sources[1][1 + comp],
                chroma_mvs[1],
                out,
                CHROMA_SCRATCH,
            );
            combine(plane_dst, &chroma, out, CHROMA_SCRATCH, weights[1 + comp]);
        }
        Ok(())
    }

    /// The chroma target of a 4:2:2 block (half the luma width, all its
    /// height) or a 4:4:4 one (the luma block at the chroma depth).
    fn other_chroma_target(&self, (x, y): (usize, usize), shape: usize) -> Target<'_> {
        let (w, h) = SHAPES[shape];
        if self.chroma == ChromaArray::Yuv444 {
            return Target {
                depth: self.depths.chroma,
                ..self.luma_target((x, y), shape)
            };
        }
        Target {
            kernel: self.kernels.chroma_422(shape),
            dsp: self.dsp,
            depth: self.depths.chroma,
            pos: (x / 2, y),
            size: (w / 2, h),
        }
    }

    /// Predict a 4:2:2 or 4:4:4 chroma `target` from `source` with `mv`
    /// (the luma vector: Table 8-10's field offset is 4:2:0's only)
    /// straight into `dst`.
    fn other_chroma(
        &self,
        target: &Target<'_>,
        source: &Source<'_, T>,
        dst: &mut PlaneViewMut<'_, T>,
        (mv, shape): (Mv, usize),
    ) {
        if self.chroma == ChromaArray::Yuv444 {
            interpolate::<LumaGrid, T>(target, source, dst, mv);
        } else if shape == SHAPE_4X4 {
            corner_block::<Chroma422Grid, T>(target, source, dst, mv);
        } else {
            interpolate::<Chroma422Grid, T>(target, source, dst, mv);
        }
    }

    /// The Cb and Cr of a 4:2:2 or 4:4:4 block predicted from one list,
    /// out of line: the 4:2:0 path stays as it was.
    #[inline(never)]
    fn single_chroma(
        &self,
        dst: [&mut PlaneViewMut<'_, T>; 2],
        sources: &[Source<'a, T>; 3],
        (mv, pos, shape): (Mv, (usize, usize), usize),
        weight: Option<(&WeightTable, [Weight; 3])>,
    ) {
        let target = self.other_chroma_target(pos, shape);
        for (comp, plane_dst) in dst.into_iter().enumerate() {
            self.other_chroma(&target, &sources[1 + comp], plane_dst, (mv, shape));
            if let Some((table, weights)) = weight {
                apply_weight(
                    plane_dst,
                    &target,
                    weights[1 + comp],
                    table.chroma_log2_denom,
                );
            }
        }
    }

    /// The Cb and Cr of a bi-predicted 4:2:2 or 4:4:4 block.
    #[inline(never)]
    fn bi_chroma(
        &self,
        dst: [&mut PlaneViewMut<'_, T>; 2],
        sources: [&[Source<'a, T>; 3]; 2],
        ([mv0, mv1], pos, shape): ([Mv; 2], (usize, usize), usize),
        weights: [BiWeight; 3],
    ) {
        let target = self.other_chroma_target(pos, shape);
        // The kernel's block: a 2x4 4:2:2 block is the corner of a 4x4 one.
        let kernel_target = Target {
            size: (
                target.size.0.max(MIN_KERNEL_SIDE),
                target.size.1.max(MIN_KERNEL_SIDE),
            ),
            ..target
        };
        let mut scratch = [T::default(); LUMA_SCRATCH * LUMA_SCRATCH];
        for (comp, plane_dst) in dst.into_iter().enumerate() {
            self.other_chroma(&target, &sources[0][1 + comp], plane_dst, (mv0, shape));
            if self.chroma == ChromaArray::Yuv444 {
                predict::<LumaGrid, T>(
                    &kernel_target,
                    &sources[1][1 + comp],
                    mv1,
                    &mut scratch,
                    LUMA_SCRATCH,
                );
            } else {
                predict::<Chroma422Grid, T>(
                    &kernel_target,
                    &sources[1][1 + comp],
                    mv1,
                    &mut scratch,
                    LUMA_SCRATCH,
                );
            }
            combine(
                plane_dst,
                &target,
                &scratch,
                LUMA_SCRATCH,
                weights[1 + comp],
            );
        }
    }

    /// The bi-prediction weights of entries `ref0` and `ref1`, per
    /// component (offsets at the component's depth, eqs. 8-297, 8-299).
    fn bi_weights(&self, ref0: usize, ref1: usize) -> [BiWeight; 3] {
        match &self.weighting {
            Weighting::Default => [BiWeight::AVERAGE; 3],
            Weighting::Explicit(table) => core::array::from_fn(|comp| {
                let denom = if comp == 0 {
                    table.luma_log2_denom
                } else {
                    table.chroma_log2_denom
                };
                let shift = self.depths.plane(comp).above_eight();
                let (w0, w1) = (table.weights[0][ref0][comp], table.weights[1][ref1][comp]);
                BiWeight {
                    weights: [w0.weight, w1.weight],
                    offset: ((w0.offset << shift) + (w1.offset << shift) + 1) >> 1,
                    log2_denom: denom,
                }
            }),
            Weighting::Implicit(table) => {
                let weights = table[ref0][ref1];
                [BiWeight {
                    weights,
                    offset: 0,
                    log2_denom: IMPLICIT_LOG2_DENOM,
                }; 3]
            }
        }
    }
}

/// The weighting of one component of a bi-prediction (eq. 8-301):
/// `((p0 · w0 + p1 · w1 + 2^logWD) >> (logWD + 1)) + offset`; averaging is
/// weights 1, 1 with `logWD` 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BiWeight {
    weights: [i32; 2],
    /// `(o0 + o1 + 1) >> 1`.
    offset: i32,
    log2_denom: u32,
}

impl BiWeight {
    /// Default bi-prediction (eq. 8-273): `(p0 + p1 + 1) >> 1`.
    const AVERAGE: Self = Self {
        weights: [1, 1],
        offset: 0,
        log2_denom: 0,
    };
}

/// Combine the first prediction (in `dst` at `target`'s position) with the
/// second (`second`, `stride`) by `weight`, in place.
fn combine<T: Sample>(
    dst: &mut PlaneViewMut<'_, T>,
    target: &Target<'_>,
    second: &[T],
    stride: usize,
    weight: BiWeight,
) {
    let depth = target.depth.of::<T>();
    let (w, h) = target.size;
    let mut block = dst
        .sub_view_mut(target.pos.0, target.pos.1, w, h)
        .expect("block inside the picture");
    let round = 1 << weight.log2_denom;
    let shift = weight.log2_denom + 1;
    for row in 0..h {
        let out = &mut block.row_mut(row)[..w];
        let other = &second[row * stride..row * stride + w];
        for (sample, &p1) in out.iter_mut().zip(other) {
            let sum = sample.to_i32() * weight.weights[0] + p1.to_i32() * weight.weights[1] + round;
            *sample = depth.clip_to((sum >> shift) + weight.offset);
        }
    }
}

/// Predict `target` from `source` straight into the picture plane `dst`.
#[allow(
    clippy::inline_always,
    reason = "measured: keeps the P path at the base instruction count (callgrind)"
)]
#[inline(always)]
fn interpolate<G: Grid, T: H264Sample>(
    target: &Target<'_>,
    source: &Source<'_, T>,
    dst: &mut PlaneViewMut<'_, T>,
    mv: Mv,
) {
    let (x, y) = target.pos;
    let (w, h) = target.size;
    assert!(
        x + w <= dst.width() && y + h <= dst.height(),
        "block inside the picture"
    );
    let stride = dst.stride();
    predict::<G, T>(
        target,
        source,
        mv,
        &mut dst.data_mut()[y * stride + x..],
        stride,
    );
}

/// [`interpolate`] of the same chroma block in Cb and Cr. Both planes have
/// one geometry, so the window is placed and checked once for the two.
#[allow(
    clippy::inline_always,
    reason = "measured: keeps the P path at the base instruction count (callgrind)"
)]
#[inline(always)]
fn interpolate_pair<T: H264Sample>(
    target: &Target<'_>,
    sources: [&Source<'_, T>; 2],
    dst: [&mut PlaneViewMut<'_, T>; 2],
    mv: Mv,
) {
    let window = Window::<ChromaGrid>::new(target, sources[0], mv);
    let same_geometry = sources[1].border == sources[0].border
        && sources[1].padded.stride() == sources[0].padded.stride()
        && (sources[1].padded.width(), sources[1].padded.height())
            == (sources[0].padded.width(), sources[0].padded.height());
    let (x, y) = target.pos;
    let (w, h) = target.size;
    for (source, dst) in sources.into_iter().zip(dst) {
        assert!(
            x + w <= dst.width() && y + h <= dst.height(),
            "block inside the picture"
        );
        let stride = dst.stride();
        let out = &mut dst.data_mut()[y * stride + x..];
        match window.start.filter(|_| same_geometry) {
            Some(start) => T::interpolate(
                target.dsp,
                target.depth,
                &target.kernel,
                &source.padded.data()[start..],
                source.padded.stride(),
                window.frac,
                out,
                stride,
            ),
            None => predict::<ChromaGrid, T>(target, source, mv, out, stride),
        }
    }
}

/// A chroma block smaller than the smallest kernel (2x2 in 4:2:0, 2x4 in
/// 4:2:2): the top-left corner of a 4x4 prediction.
fn corner_block<G: Grid, T: H264Sample>(
    target: &Target<'_>,
    source: &Source<'_, T>,
    dst: &mut PlaneViewMut<'_, T>,
    mv: Mv,
) {
    let corner = Target {
        size: (MIN_KERNEL_SIDE, MIN_KERNEL_SIDE),
        ..*target
    };
    let mut block = [T::default(); MIN_KERNEL_SIDE * MIN_KERNEL_SIDE];
    predict::<G, T>(&corner, source, mv, &mut block, MIN_KERNEL_SIDE);
    let (w, h) = target.size;
    let mut out = dst
        .sub_view_mut(target.pos.0, target.pos.1, w, h)
        .expect("block inside the picture");
    for (row, samples) in block
        .as_chunks::<MIN_KERNEL_SIDE>()
        .0
        .iter()
        .take(h)
        .enumerate()
    {
        out.row_mut(row).copy_from_slice(&samples[..w]);
    }
}

/// Where a block's interpolation window lies in a reference plane.
struct Window<G> {
    /// Window corner relative to the visible (0, 0).
    corner: (isize, isize),
    /// Offset of the corner in the padded samples, when the whole window
    /// lies inside them.
    start: Option<usize>,
    /// Sub-sample offsets.
    frac: (usize, usize),
    grid: core::marker::PhantomData<G>,
}

impl<G: Grid> Window<G> {
    fn new<T: Sample>(target: &Target<'_>, source: &Source<'_, T>, mv: Mv) -> Self {
        let (mvx, mvy) = (i32::from(mv.x), i32::from(mv.y));
        let mask = (1 << G::FRAC_BITS) - 1;
        let mask_y = (1 << G::FRAC_BITS_Y) - 1;
        let (w, h) = (target.size.0 + G::TAPS - 1, target.size.1 + G::TAPS - 1);
        let left = target.pos.0 as isize + (mvx >> G::FRAC_BITS) as isize - G::BEFORE as isize;
        let top = target.pos.1 as isize + (mvy >> G::FRAC_BITS_Y) as isize - G::BEFORE as isize;
        let padded = &source.padded;
        let (col, row) = (
            left + source.border.0 as isize,
            top + source.border.1 as isize,
        );
        let inside = col >= 0
            && row >= 0
            && col as usize + w <= padded.width()
            && row as usize + h <= padded.height();
        Self {
            corner: (left, top),
            start: inside.then(|| row as usize * padded.stride() + col as usize),
            frac: (
                (mvx & mask) as usize,
                ((mvy & mask_y) << G::FRAC_SHIFT_Y) as usize,
            ),
            grid: core::marker::PhantomData,
        }
    }
}

/// Predict `target` displaced by `mv` into `out` (`out_stride`). Reference
/// samples outside the picture are clamped to its edge (eqs. 8-228..8-231):
/// the padded border is an edge extension, and windows beyond it are
/// copied clamped. Phase 0 of every filter is the identity (eqs. 8-241,
/// 8-266): the kernels copy whole-sample blocks themselves.
#[allow(
    clippy::inline_always,
    reason = "measured: keeps the P path at the base instruction count (callgrind)"
)]
#[inline(always)]
fn predict<G: Grid, T: H264Sample>(
    target: &Target<'_>,
    source: &Source<'_, T>,
    mv: Mv,
    out: &mut [T],
    out_stride: usize,
) {
    let window = Window::<G>::new(target, source, mv);
    let call = |window_samples: &[T], stride: usize, out: &mut [T]| {
        T::interpolate(
            target.dsp,
            target.depth,
            &target.kernel,
            window_samples,
            stride,
            window.frac,
            out,
            out_stride,
        );
    };
    if let Some(start) = window.start {
        let padded = &source.padded;
        call(&padded.data()[start..], padded.stride(), out);
    } else {
        let size = (target.size.0 + G::TAPS - 1, target.size.1 + G::TAPS - 1);
        let mut buf = [T::default(); MAX_WINDOW * MAX_WINDOW];
        let (left, top) = window.corner;
        let view = clamped_window(&source.visible, left, top, size, &mut buf);
        call(view.data(), view.stride(), out);
    }
}

fn clamped_window<'a, T: Sample>(
    picture: &PlaneView<'_, T>,
    left: isize,
    top: isize,
    (width, height): (usize, usize),
    buf: &'a mut [T],
) -> PlaneView<'a, T> {
    let (w, h) = (picture.width() as isize, picture.height() as isize);
    for (r, row) in buf.chunks_exact_mut(width).take(height).enumerate() {
        let src = picture.row((top + r as isize).clamp(0, h - 1) as usize);
        for (c, v) in row.iter_mut().enumerate() {
            *v = src[(left + c as isize).clamp(0, w - 1) as usize];
        }
    }
    PlaneView::new(&buf[..width * height], width, width, height).expect("window fits the buffer")
}

/// Explicit weighted sample prediction of one list (eqs. 8-270, 8-271)
/// over `target`'s block of `plane`, in place; the offset is scaled to the
/// depth (eqs. 8-297, 8-299).
fn apply_weight<T: Sample>(
    plane: &mut PlaneViewMut<'_, T>,
    target: &Target<'_>,
    w: Weight,
    log2_denom: u32,
) {
    let ((x, y), (width, height)) = (target.pos, target.size);
    let depth = target.depth.of::<T>();
    let mut block = plane
        .sub_view_mut(x, y, width, height)
        .expect("block inside the picture");
    let round = if log2_denom >= 1 {
        1 << (log2_denom - 1)
    } else {
        0
    };
    let weigh = WeighRow {
        weight: w.weight,
        offset: w.offset << depth.above_eight(),
        round,
        log2_denom,
        depth,
    };
    // Rows of a constant width: each compiles to straight-line vector code
    // (no per-sample loop exits, which mispredict as block widths vary).
    match width {
        16 => weigh.rows::<16, T>(&mut block),
        8 => weigh.rows::<8, T>(&mut block),
        4 => weigh.rows::<4, T>(&mut block),
        2 => weigh.rows::<2, T>(&mut block),
        _ => unreachable!("prediction block width {width}"),
    }
}

/// One weight's arithmetic (eq. 8-270): `((s · weight + round) >>
/// log2_denom) + offset`, clipped.
struct WeighRow {
    weight: i32,
    offset: i32,
    round: i32,
    log2_denom: u32,
    depth: BitDepth,
}

impl WeighRow {
    /// Weigh every row of `block`, `W` samples wide.
    fn rows<const W: usize, T: Sample>(&self, block: &mut PlaneViewMut<'_, T>) {
        for row in 0..block.height() {
            let samples: &mut [T; W] = (&mut block.row_mut(row)[..W])
                .try_into()
                .expect("block row");
            for sample in samples {
                let weighted =
                    ((sample.to_i32() * self.weight + self.round) >> self.log2_denom) + self.offset;
                *sample = self.depth.clip_to(weighted);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grids_match_the_dsp_filters() {
        for (filter, taps, before) in [
            (Filter::H264Luma, LumaGrid::TAPS, LumaGrid::BEFORE),
            (Filter::H264Chroma, ChromaGrid::TAPS, ChromaGrid::BEFORE),
        ] {
            assert_eq!(
                (filter.taps(), filter.before()),
                (taps, before),
                "{filter:?}"
            );
        }
    }
}
