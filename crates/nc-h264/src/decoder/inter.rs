//! Inter prediction of P macroblocks (§8.4.2): fractional sample
//! interpolation through the nc-dsp H.264 kernels and explicit weighted
//! prediction (§8.4.2.3.2).
//!
//! Interpolation is a per-sample process, so a partition is predicted as
//! any tiling of square blocks with its vector: the whole macroblock when
//! all its vectors and references agree, else each 8x8 block whole or as
//! four 4x4 blocks. The prediction is written straight into the picture,
//! where the residual is then added.

use nc_common::{Plane, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, Filter, H264Predictor};
use nc_encoder::Mv;

use super::dpb::RefPicture;
use super::mbstate::{MbKind, MbState, Partition};
use super::slice::{Weight, WeightTable};
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
}

impl InterKernels {
    pub fn new(dsp: &Dsp) -> Self {
        let kernel = |filter, (w, h): (usize, usize)| {
            dsp.h264_predictor(filter, BlockSize::new(w, h).expect("kernel size"))
        };
        let chroma_side = |side: usize| (side / 2).max(MIN_KERNEL_SIDE);
        Self {
            luma: SHAPES.map(|shape| kernel(Filter::H264Luma, shape)),
            chroma: SHAPES
                .map(|(w, h)| kernel(Filter::H264Chroma, (chroma_side(w), chroma_side(h)))),
        }
    }
}

/// One reference plane as the kernels read it: its padded samples and
/// the position of visible sample (0, 0) in them.
#[derive(Clone, Copy)]
struct Source<'a> {
    padded: PlaneView<'a>,
    border: (usize, usize),
    /// The plane itself, for windows beyond the padding.
    plane: &'a Plane,
}

impl<'a> Source<'a> {
    fn new(plane: &'a Plane) -> Self {
        Self {
            padded: plane.padded_view(),
            border: (plane.border_x(), plane.border_y()),
            plane,
        }
    }
}

/// The reference list of a slice and its weights.
pub(super) struct InterContext<'a> {
    kernels: &'a InterKernels,
    /// Planes (Y, U, V) of each list entry's picture.
    refs: Vec<Option<[Source<'a>; 3]>>,
    weights: Option<&'a WeightTable>,
    monochrome: bool,
}

/// A block of one component to predict.
#[derive(Clone, Copy)]
struct Target<'k> {
    kernel: &'k H264Predictor,
    /// Position and size in the component's samples.
    pos: (usize, usize),
    size: (usize, usize),
}

/// The sample grid of one component's interpolation: fractional bits of
/// its vectors, filter taps and the taps before the centre sample (as
/// constants, so the window arithmetic folds in each instantiation).
trait Grid {
    const FRAC_BITS: i32;
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

impl<'a> InterContext<'a> {
    pub fn new(
        kernels: &'a InterKernels,
        refs: &'a [Option<RefPicture>],
        weights: Option<&'a WeightTable>,
        monochrome: bool,
    ) -> Self {
        let refs = refs
            .iter()
            .map(|r| {
                r.as_ref()
                    .map(|r| PlaneIndex::ALL.map(|p| Source::new(r.picture.plane(p))))
            })
            .collect();
        Self {
            kernels,
            refs,
            weights,
            monochrome,
        }
    }

    /// Predict the P macroblock `mb` at macroblock position (`mx`, `my`).
    pub fn predict(&self, dst: &mut Planes<'_>, mb: &MbState, mx: usize, my: usize) -> Result<()> {
        let origin = (mx * 16, my * 16);
        if mb.single_motion {
            return self.block(dst, mb.mvs[0], mb.ref_idx[0], origin, SHAPE_16X16);
        }
        // 16x8 and 8x16 partitions have one vector each: one call per
        // partition (4x4 block 8 starts the lower half, 2 the right half).
        match mb.kind {
            MbKind::Inter(Partition::P16x8) => {
                self.block(dst, mb.mvs[0], mb.ref_idx[0], origin, SHAPE_16X8)?;
                let lower = (origin.0, origin.1 + 8);
                return self.block(dst, mb.mvs[8], mb.ref_idx[2], lower, SHAPE_16X8);
            }
            MbKind::Inter(Partition::P8x16) => {
                self.block(dst, mb.mvs[0], mb.ref_idx[0], origin, SHAPE_8X16)?;
                let right = (origin.0 + 8, origin.1);
                return self.block(dst, mb.mvs[2], mb.ref_idx[1], right, SHAPE_8X16);
            }
            _ => {}
        }
        for b8 in 0..4 {
            let (x4, y4) = ((b8 % 2) * 2, (b8 / 2) * 2);
            let ref_idx = mb.ref_idx[b8];
            let corner = mb.mvs[y4 * 4 + x4];
            let same = [(1, 0), (0, 1), (1, 1)]
                .iter()
                .all(|&(dx, dy)| mb.mvs[(y4 + dy) * 4 + x4 + dx] == corner);
            if same {
                let pos = (mx * 16 + x4 * 4, my * 16 + y4 * 4);
                self.block(dst, corner, ref_idx, pos, SHAPE_8X8)?;
                continue;
            }
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let mv = mb.mvs[(y4 + dy) * 4 + x4 + dx];
                let pos = (mx * 16 + (x4 + dx) * 4, my * 16 + (y4 + dy) * 4);
                self.block(dst, mv, ref_idx, pos, SHAPE_4X4)?;
            }
        }
        Ok(())
    }

    /// One luma block of shape `SHAPES[shape]` at (`x`, `y`) and its
    /// chroma, from reference `ref_idx` displaced by `mv`, into the picture
    /// planes `dst`.
    fn block(
        &self,
        dst: &mut Planes<'_>,
        mv: Mv,
        ref_idx: i8,
        (x, y): (usize, usize),
        shape: usize,
    ) -> Result<()> {
        let sources = usize::try_from(ref_idx)
            .ok()
            .and_then(|r| self.refs.get(r).and_then(Option::as_ref))
            .ok_or(DecodeError::Malformed("inter prediction from no picture"))?;
        let (w, h) = SHAPES[shape];
        let weight = self
            .weights
            .filter(|w| !w.is_identity(ref_idx as usize))
            .map(|w| (w, w.weights[ref_idx as usize]));
        let [luma_dst, cb_dst, cr_dst] = dst;
        let luma = Target {
            kernel: &self.kernels.luma[shape],
            pos: (x, y),
            size: (w, h),
        };
        interpolate::<LumaGrid>(&luma, &sources[0], luma_dst, mv);
        if let Some((table, weights)) = weight {
            apply_weight(luma_dst, (x, y), (w, h), weights[0], table.luma_log2_denom);
        }
        if self.monochrome {
            return Ok(());
        }
        let chroma = Target {
            kernel: &self.kernels.chroma[shape],
            pos: (x / 2, y / 2),
            size: (w / 2, h / 2),
        };
        if shape == SHAPE_4X4 {
            for (source, plane_dst) in [(&sources[1], &mut *cb_dst), (&sources[2], &mut *cr_dst)] {
                corner_2x2(&chroma, source, plane_dst, mv);
            }
        } else {
            interpolate_pair(&chroma, [&sources[1], &sources[2]], [cb_dst, cr_dst], mv);
        }
        if let Some((table, weights)) = weight {
            for (comp, plane_dst) in [cb_dst, cr_dst].into_iter().enumerate() {
                apply_weight(
                    plane_dst,
                    chroma.pos,
                    chroma.size,
                    weights[1 + comp],
                    table.chroma_log2_denom,
                );
            }
        }
        Ok(())
    }
}

/// Predict `target` from `source` straight into the picture plane `dst`.
fn interpolate<G: Grid>(
    target: &Target<'_>,
    source: &Source<'_>,
    dst: &mut PlaneViewMut<'_>,
    mv: Mv,
) {
    let (x, y) = target.pos;
    let (w, h) = target.size;
    assert!(
        x + w <= dst.width() && y + h <= dst.height(),
        "block inside the picture"
    );
    let stride = dst.stride();
    predict::<G>(
        target,
        source,
        mv,
        &mut dst.data_mut()[y * stride + x..],
        stride,
    );
}

/// [`interpolate`] of the same chroma block in Cb and Cr. Both planes have
/// one geometry, so the window is placed and checked once for the two.
fn interpolate_pair(
    target: &Target<'_>,
    sources: [&Source<'_>; 2],
    dst: [&mut PlaneViewMut<'_>; 2],
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
            Some(start) => target.kernel.predict(
                &source.padded.data()[start..],
                source.padded.stride(),
                window.frac.0,
                window.frac.1,
                out,
                stride,
            ),
            None => predict::<ChromaGrid>(target, source, mv, out, stride),
        }
    }
}

/// A 2x2 chroma block: the top-left corner of a 4x4 prediction (the
/// smallest kernel).
fn corner_2x2(target: &Target<'_>, source: &Source<'_>, dst: &mut PlaneViewMut<'_>, mv: Mv) {
    let corner = Target {
        size: (MIN_KERNEL_SIDE, MIN_KERNEL_SIDE),
        ..*target
    };
    let mut block = [0u8; MIN_KERNEL_SIDE * MIN_KERNEL_SIDE];
    predict::<ChromaGrid>(&corner, source, mv, &mut block, MIN_KERNEL_SIDE);
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
    fn new(target: &Target<'_>, source: &Source<'_>, mv: Mv) -> Self {
        let (mvx, mvy) = (i32::from(mv.x), i32::from(mv.y));
        let mask = (1 << G::FRAC_BITS) - 1;
        let (w, h) = (target.size.0 + G::TAPS - 1, target.size.1 + G::TAPS - 1);
        let left = target.pos.0 as isize + (mvx >> G::FRAC_BITS) as isize - G::BEFORE as isize;
        let top = target.pos.1 as isize + (mvy >> G::FRAC_BITS) as isize - G::BEFORE as isize;
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
            frac: ((mvx & mask) as usize, (mvy & mask) as usize),
            grid: core::marker::PhantomData,
        }
    }
}

/// Predict `target` displaced by `mv` into `out` (`out_stride`). Reference
/// samples outside the picture are clamped to its edge (eqs. 8-228..8-231):
/// the padded border is an edge extension, and windows beyond it are
/// copied clamped. Phase 0 of every filter is the identity (eqs. 8-241,
/// 8-266): the kernels copy whole-sample blocks themselves.
fn predict<G: Grid>(
    target: &Target<'_>,
    source: &Source<'_>,
    mv: Mv,
    out: &mut [u8],
    out_stride: usize,
) {
    let window = Window::<G>::new(target, source, mv);
    let (fx, fy) = window.frac;
    if let Some(start) = window.start {
        let padded = &source.padded;
        target.kernel.predict(
            &padded.data()[start..],
            padded.stride(),
            fx,
            fy,
            out,
            out_stride,
        );
    } else {
        let size = (target.size.0 + G::TAPS - 1, target.size.1 + G::TAPS - 1);
        let mut buf = [0u8; MAX_WINDOW * MAX_WINDOW];
        let (left, top) = window.corner;
        let view = clamped_window(source.plane, left, top, size, &mut buf);
        target
            .kernel
            .predict(view.data(), view.stride(), fx, fy, out, out_stride);
    }
}

fn clamped_window<'a>(
    source: &Plane,
    left: isize,
    top: isize,
    (width, height): (usize, usize),
    buf: &'a mut [u8],
) -> PlaneView<'a> {
    let picture = source.view();
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
/// over the `width` x `height` block at (`x`, `y`) of `plane`, in place.
fn apply_weight(
    plane: &mut PlaneViewMut<'_>,
    (x, y): (usize, usize),
    (width, height): (usize, usize),
    w: Weight,
    log2_denom: u32,
) {
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
        offset: w.offset,
        round,
        log2_denom,
    };
    // Rows of a constant width: each compiles to straight-line vector code
    // (no per-sample loop exits, which mispredict as block widths vary).
    match width {
        16 => weigh.rows::<16>(&mut block),
        8 => weigh.rows::<8>(&mut block),
        4 => weigh.rows::<4>(&mut block),
        2 => weigh.rows::<2>(&mut block),
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
}

impl WeighRow {
    /// Weigh every row of `block`, `W` samples wide.
    fn rows<const W: usize>(&self, block: &mut PlaneViewMut<'_>) {
        for row in 0..block.height() {
            let samples: &mut [u8; W] = (&mut block.row_mut(row)[..W])
                .try_into()
                .expect("block row");
            for sample in samples {
                let weighted = ((i32::from(*sample) * self.weight + self.round) >> self.log2_denom)
                    + self.offset;
                *sample = weighted.clamp(0, 255) as u8;
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
