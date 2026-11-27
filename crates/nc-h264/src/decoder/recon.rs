//! Reconstruction of one decoded macroblock into the picture: `I_PCM`
//! samples, intra prediction block by block with its residual (§8.3), or
//! inter prediction (`inter.rs`) followed by the residual (§8.5), for each
//! chroma format: 4:2:0 / 4:2:2 chroma with its own intra modes and blocks,
//! 4:4:4 Cb and Cr reconstructed like luma. Lossless macroblocks add their
//! residual untransformed (`lossless.rs`); the P macroblocks of SP slices
//! and SI macroblocks requantise their prediction instead (§8.6,
//! `switching.rs`).

use nc_common::{BitDepth, BitDepths, PlaneIndex, PlaneViewMut, Sample};
use nc_dsp::Dsp;

use super::chroma::{self, ChromaArray};
use super::inter::InterContext;
use super::intra::{self, Avail};
use super::lossless::{self, Dpcm};
use super::mb::{MbExtra, PCM_LUMA_SAMPLES};
use super::mbstate::{self, MbKind, MbState, Neighbours, PairNeighbours};
use super::sample::H264Sample;
use super::switching::{Switching, SwitchingLevels};
use super::transform::Residual;
use super::{Planes, Result};

const MB_SIZE: usize = 16;
const CHROMA_MB_SIZE: usize = 8;
/// Side of a 4x4 block.
const SIDE_4: usize = 4;

/// Which neighbouring samples intra prediction may read (§8.3.1.2,
/// §8.3.3, §8.3.4: with `constrained_intra_pred_flag`, inter macroblocks
/// count as unavailable, and `SI` ones unless the current one is `SI`): the
/// left column per 4x4 block row (in an MBAFF frame it may come from both
/// macroblocks of the left pair), and the macroblocks B, C and D.
#[derive(Debug, Clone, Copy)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "one availability per neighbouring macroblock"
)]
struct IntraNeighbours {
    /// Bit `y4`: the left samples of 4x4 block row `y4` are available.
    left: u8,
    b: bool,
    c: bool,
    d: bool,
}

impl IntraNeighbours {
    /// Nothing available (an inter macroblock reads no neighbouring samples).
    const NONE: Self = Self {
        left: 0,
        b: false,
        c: false,
        d: false,
    };

    #[inline]
    fn of(nb: &Neighbours<'_>, constrained: bool, si: bool) -> Self {
        let usable = |m: &MbState| !constrained || (m.is_intra() && (si || m.kind != MbKind::Si));
        let left = match &nb.pair {
            None => {
                let a = nb.a.is_some_and(usable);
                rows(a, a)
            }
            Some(pair) => left_in_pair(pair, usable),
        };
        Self {
            left,
            b: nb.b.is_some_and(usable),
            c: nb.c.is_some_and(usable),
            d: nb.d.is_some_and(usable),
        }
    }

    /// Whether the left samples of the upper and the lower half are
    /// available (4:2:0 chroma blocks).
    fn left_halves(self) -> [bool; 2] {
        [self.left & 0b0011 == 0b0011, self.left & 0b1100 == 0b1100]
    }

    /// Availability for the block of `size4` 4x4 blocks at (`x4`, `y4`).
    fn block(self, x4: usize, y4: usize, size4: usize) -> Avail {
        let right = x4 + size4;
        let top_right = if y4 == 0 {
            if right < 4 { self.b } else { self.c }
        } else {
            right < 4 && mbstate::decode_index(right, y4 - 1) < mbstate::decode_index(x4, y4)
        };
        Avail {
            left: x4 > 0 || {
                let block_rows = ((1u8 << size4) - 1) << y4;
                self.left & block_rows == block_rows
            },
            top: y4 > 0 || self.b,
            top_right,
            top_left: match (x4 > 0, y4 > 0) {
                (true, true) => true,
                (false, true) => (self.left >> (y4 - 1)) & 1 != 0,
                (true, false) => self.b,
                (false, false) => self.d,
            },
        }
    }
}

/// Left availability bits of the upper and the lower two 4x4 rows.
fn rows(upper: bool, lower: bool) -> u8 {
    (u8::from(upper) * 0b0011) | (u8::from(lower) * 0b1100)
}

/// The left availability bits of an MBAFF macroblock (`usable` says which
/// macroblocks intra prediction may read). Its left samples lie in the
/// left pair row for row (Table 6-4 maps them to the samples beside them):
/// a pair coded alike gives the macroblock in the same position; a field
/// pair beside a frame macroblock every row from its two macroblocks in
/// turn; a frame pair beside a field macroblock its top macroblock to the
/// upper half.
#[inline(never)]
fn left_in_pair(pair: &PairNeighbours<'_>, usable: impl Fn(&MbState) -> bool) -> u8 {
    let Some((top, bottom)) = pair.left_pair else {
        return 0;
    };
    match (pair.field, top.field) {
        (false, false) | (true, true) => {
            let a = usable(if pair.bottom { bottom } else { top });
            rows(a, a)
        }
        (false, true) => {
            let both = usable(top) && usable(bottom);
            rows(both, both)
        }
        (true, false) => rows(usable(top), usable(bottom)),
    }
}

/// Everything reconstruction needs besides the macroblock.
pub(super) struct ReconContext<'a, T: Sample> {
    pub dsp: &'a Dsp,
    pub inter: InterContext<'a, T>,
    pub constrained_intra_pred: bool,
    /// `ChromaArrayType`.
    pub chroma: ChromaArray,
    /// The QS fields of an SP or SI slice.
    pub switching: Option<Switching>,
    pub depths: BitDepths,
}

/// Reconstruct macroblock `mb` at macroblock position (`mx`, `my`).
pub(super) fn reconstruct<T: H264Sample>(
    ctx: &ReconContext<'_, T>,
    planes: &mut Planes<'_, T>,
    (mx, my): (usize, usize),
    mb: &MbState,
    nb: &Neighbours<'_>,
    residual: &mut Residual<T::Coeff>,
    extra: &mut MbExtra,
) -> Result<()> {
    let (x, y) = (mx * MB_SIZE, my * MB_SIZE);
    // Only intra macroblocks read their neighbours' samples.
    let intra = if mb.is_intra() || mb.kind == MbKind::Si {
        IntraNeighbours::of(nb, ctx.constrained_intra_pred, mb.kind == MbKind::Si)
    } else {
        IntraNeighbours::NONE
    };
    // Folds to the 8-bit constants in the 8-bit instance.
    let depths = ctx.depths.of::<T>();
    if mb.kind == MbKind::IPcm {
        pcm(planes, (x, y), extra, ctx.chroma);
        return Ok(());
    }
    if let Some(switching) = ctx.switching
        && (mb.kind == MbKind::Si || !mb.is_intra())
    {
        let levels = &mut extra.switching;
        return requantised(ctx, switching, planes, (mx, my), mb, intra, levels);
    }
    if !mb.is_intra() {
        ctx.inter.predict(planes, mb, mx, my)?;
    }
    let [luma, cb, cr] = planes;
    let at = ((x, y), intra);
    luma_component::<T, 0>((ctx.dsp, depths.luma), luma, at, mb, residual)?;
    match ctx.chroma {
        ChromaArray::Yuv420 if !residual.bypass() => chroma_420(ctx, [cb, cr], at, mb, residual),
        ChromaArray::Mono => Ok(()),
        _ => other_chroma(ctx, [cb, cr], at, mb, residual),
    }
}

/// A P macroblock of an SP slice or an `SI` macroblock: its inter or
/// `Intra_4x4` prediction (block by block, in decoding order) replaced by
/// the §8.6 reconstruction from `levels`. Cold and out of line, and its
/// intra prediction is the generic one: a second caller of the 8-bit
/// `intra_4x4` kernel took it out of line in every intra macroblock
/// (+0.5 % decode instructions on the High suite streams).
#[cold]
#[inline(never)]
fn requantised<T: H264Sample>(
    ctx: &ReconContext<'_, T>,
    switching: Switching,
    planes: &mut Planes<'_, T>,
    (mx, my): (usize, usize),
    mb: &MbState,
    intra: IntraNeighbours,
    levels: &mut SwitchingLevels,
) -> Result<()> {
    let (x, y) = (mx * MB_SIZE, my * MB_SIZE);
    let depths = ctx.depths.of::<T>();
    let si = mb.kind == MbKind::Si;
    if !si {
        ctx.inter.predict(planes, mb, mx, my)?;
    }
    let qp = i32::from(mb.qp);
    let [luma, cb, cr] = planes;
    for index in 0..16 {
        let (x4, y4) = mbstate::block_position(index);
        let (px, py) = (x + x4 * 4, y + y4 * 4);
        if si {
            let mode = mb.intra_modes[y4 * 4 + x4];
            let avail = (intra.block(x4, y4, 1), depths.luma);
            intra::predict_nxn(luma, (px, py), 4, mode, avail)?;
        }
        let block = (y4 * 4 + x4, qp);
        let dsp = (ctx.dsp, depths.luma);
        switching.luma_4x4(dsp, levels, block, &mut block_view(luma, (px, py), 4));
    }
    // SP and SI slices are Extended profile: 4:2:0 or 4:0:0.
    if ctx.chroma.is_mono() {
        return Ok(());
    }
    let (cx, cy) = (x / 2, y / 2);
    for (comp, plane) in [cb, cr].into_iter().enumerate() {
        if si {
            let avail = (intra.block(0, 0, 4), intra.left_halves(), depths.chroma);
            let mode = mb.chroma_pred_mode;
            intra::predict_chroma::<T, CHROMA_MB_SIZE>(plane, (cx, cy), mode, avail)?;
        }
        let dsp = (ctx.dsp, depths.chroma);
        let mut block = block_view(plane, (cx, cy), CHROMA_MB_SIZE);
        switching.chroma(dsp, levels, (comp, qp), &mut block);
    }
    Ok(())
}

/// Predict (intra) and add the residual of component `PLANE`'s luma-like
/// blocks: luma, or Cb and Cr in 4:4:4.
#[allow(
    clippy::inline_always,
    reason = "the luma of every macroblock: out of line (gate build) it took \
              the inter prediction out of `reconstruct` as well (about +2 % \
              instructions on a 720p P stream, callgrind)"
)]
#[inline(always)]
fn luma_component<T: H264Sample, const PLANE: usize>(
    (dsp, depth): (&Dsp, BitDepth),
    view: &mut PlaneViewMut<'_, T>,
    ((x, y), intra): ((usize, usize), IntraNeighbours),
    mb: &MbState,
    residual: &mut Residual<T::Coeff>,
) -> Result<()> {
    if residual.bypass() {
        return lossless_component(depth, (view, PLANE), ((x, y), intra), mb, residual);
    }
    let dsp = (dsp, depth);
    match mb.kind {
        // `SI` outside an SI slice is not decoded (`mb.rs`).
        MbKind::IntraNxN | MbKind::Si => {
            luma_nxn::<T, PLANE>(dsp, view, (x, y), mb, intra, residual)?;
        }
        MbKind::Intra16x16(mode) => {
            intra::predict_16x16(view, (x, y), mode, (intra.block(0, 0, 4), depth))?;
            add_luma::<T, PLANE>(dsp, view, (x, y), mb, residual);
        }
        MbKind::Inter(_) | MbKind::PSkip | MbKind::BSkip | MbKind::BDirect16x16 | MbKind::IPcm => {
            add_luma::<T, PLANE>(dsp, view, (x, y), mb, residual);
        }
    }
    Ok(())
}

/// [`luma_component`] of a lossless macroblock (transform bypass): the
/// same prediction, its residual added untransformed with the DPCM of its
/// horizontal and vertical intra modes (§8.5.15). Out of line: lossless
/// macroblocks are rare outside lossless streams; and with the generic
/// intra prediction, as `requantised`: a second caller of the 8-bit
/// `Intra_4x4` kernel takes it out of line in every intra macroblock.
#[cold]
#[inline(never)]
fn lossless_component<T: H264Sample>(
    depth: BitDepth,
    (view, plane): (&mut PlaneViewMut<'_, T>, usize),
    ((x, y), intra): ((usize, usize), IntraNeighbours),
    mb: &MbState,
    residual: &mut Residual<T::Coeff>,
) -> Result<()> {
    let dpcm = match mb.kind {
        MbKind::IntraNxN | MbKind::Si if mb.transform_8x8 => {
            for b8 in 0..4 {
                let (x4, y4) = ((b8 % 2) * 2, (b8 / 2) * 2);
                let (px, py) = (x + x4 * 4, y + y4 * 4);
                let mode = mb.intra_modes[y4 * 4 + x4];
                let avail = (intra.block(x4, y4, 2), depth);
                intra::predict_nxn(view, (px, py), 8, mode, avail)?;
                let mut dst = block_view(view, (px, py), 8);
                lossless::add_8x8(
                    residual,
                    (plane, b8),
                    Dpcm::of_luma_mode(mode),
                    &mut dst,
                    depth,
                );
            }
            return Ok(());
        }
        MbKind::IntraNxN | MbKind::Si => {
            for index in 0..16 {
                let (x4, y4) = mbstate::block_position(index);
                let (px, py) = (x + x4 * 4, y + y4 * 4);
                let mode = mb.intra_modes[y4 * 4 + x4];
                let avail = (intra.block(x4, y4, 1), depth);
                intra::predict_nxn(view, (px, py), 4, mode, avail)?;
                let block = (plane, y4 * 4 + x4);
                let mut dst = block_view(view, (px, py), 4);
                lossless::add_4x4(residual, block, Dpcm::of_luma_mode(mode), &mut dst, depth);
            }
            return Ok(());
        }
        MbKind::Intra16x16(mode) => {
            intra::predict_16x16(view, (x, y), mode, (intra.block(0, 0, 4), depth))?;
            Dpcm::of_luma_mode(mode)
        }
        MbKind::Inter(_) | MbKind::PSkip | MbKind::BSkip | MbKind::BDirect16x16 | MbKind::IPcm => {
            Dpcm::None
        }
    };
    if residual.luma(plane).has_any() {
        let mut dst = block_view(view, (x, y), MB_SIZE);
        lossless::add_macroblock(residual, (plane, mb.transform_8x8), dpcm, &mut dst, depth);
    }
    Ok(())
}

/// 4:2:0 chroma with a transformed residual (the hot path): intra
/// prediction by `intra_chroma_pred_mode`, then the residual's 4x4 blocks.
fn chroma_420<T: H264Sample>(
    ctx: &ReconContext<'_, T>,
    planes: [&mut PlaneViewMut<'_, T>; 2],
    ((x, y), intra): ((usize, usize), IntraNeighbours),
    mb: &MbState,
    residual: &mut Residual<T::Coeff>,
) -> Result<()> {
    // Inter chroma without coefficients is the prediction already.
    if !mb.is_intra() && !residual.has_chroma_any() {
        return Ok(());
    }
    let depth = ctx.depths.of::<T>().chroma;
    let (cx, cy) = (x / 2, y / 2);
    for (comp, plane) in planes.into_iter().enumerate() {
        if mb.is_intra() {
            let avail = (intra.block(0, 0, 4), intra.left_halves(), depth);
            let mode = mb.chroma_pred_mode;
            intra::predict_chroma::<T, { MB_SIZE / 2 }>(plane, (cx, cy), mode, avail)?;
        }
        for b in 0..4 {
            if !residual.has_chroma(comp, b) {
                continue;
            }
            let (bx, by) = chroma::chroma_block_position(b);
            let mut dst = block_view(plane, (cx + bx, cy + by), SIDE_4);
            residual.add_chroma_4x4(ctx.dsp, depth, (comp, b), &mut dst);
        }
    }
    Ok(())
}

/// The chroma of 4:2:2 and 4:4:4 macroblocks, and of lossless 4:2:0 ones,
/// out of line (the 4:2:0 path stays as it was): 4:4:4 Cb and Cr like
/// luma; 4:2:0 / 4:2:2 chroma intra prediction and the residual's 4x4
/// blocks, or (lossless) the untransformed residual of the whole block.
#[inline(never)]
fn other_chroma<T: H264Sample>(
    ctx: &ReconContext<'_, T>,
    planes: [&mut PlaneViewMut<'_, T>; 2],
    at: ((usize, usize), IntraNeighbours),
    mb: &MbState,
    residual: &mut Residual<T::Coeff>,
) -> Result<()> {
    let depth = ctx.depths.of::<T>().chroma;
    if ctx.chroma == ChromaArray::Yuv444 {
        let [cb, cr] = planes;
        luma_component::<T, 1>((ctx.dsp, depth), cb, at, mb, residual)?;
        return luma_component::<T, 2>((ctx.dsp, depth), cr, at, mb, residual);
    }
    let ((x, y), intra) = at;
    if !mb.is_intra() && !residual.has_chroma_any() {
        return Ok(());
    }
    let (sx, sy) = ctx.chroma.subsampling();
    let (cx, cy) = (x >> sx, y >> sy);
    let size = ctx.chroma.mb_size();
    for (comp, plane) in planes.into_iter().enumerate() {
        if mb.is_intra() {
            let avail = (intra.block(0, 0, 4), intra.left_halves(), depth);
            let mode = mb.chroma_pred_mode;
            if ctx.chroma == ChromaArray::Yuv422 {
                intra::predict_chroma::<T, MB_SIZE>(plane, (cx, cy), mode, avail)?;
            } else {
                intra::predict_chroma::<T, { MB_SIZE / 2 }>(plane, (cx, cy), mode, avail)?;
            }
        }
        if residual.bypass() {
            let dpcm = if mb.is_intra() {
                Dpcm::of_chroma_mode(mb.chroma_pred_mode)
            } else {
                Dpcm::None
            };
            let mut dst = rect_view(plane, (cx, cy), size);
            lossless::add_chroma(residual, (comp, size), dpcm, &mut dst, depth);
            continue;
        }
        for b in 0..ctx.chroma.chroma_blocks() {
            if !residual.has_chroma(comp, b) {
                continue;
            }
            let (bx, by) = chroma::chroma_block_position(b);
            let mut dst = block_view(plane, (cx + bx, cy + by), SIDE_4);
            residual.add_chroma_4x4(ctx.dsp, depth, (comp, b), &mut dst);
        }
    }
    Ok(())
}

/// The `side` x `side` block at `(x, y)` of `plane`.
fn block_view<'p, T: Sample>(
    plane: &'p mut PlaneViewMut<'_, T>,
    at: (usize, usize),
    side: usize,
) -> PlaneViewMut<'p, T> {
    rect_view(plane, at, (side, side))
}

/// The `w` x `h` block at `(x, y)` of `plane`.
fn rect_view<'p, T: Sample>(
    plane: &'p mut PlaneViewMut<'_, T>,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
) -> PlaneViewMut<'p, T> {
    plane
        .sub_view_mut(x, y, w, h)
        .expect("block inside the picture")
}

/// `Intra_4x4` or `Intra_8x8` of component `PLANE`: predict each block,
/// then add its residual, in decoding order. Luma with the 8-bit kernels;
/// Cb and Cr (4:4:4) with the generic prediction, so the 8-bit `Intra_4x4`
/// kernel keeps its one caller and stays inlined (`requantised`).
fn luma_nxn<T: H264Sample, const PLANE: usize>(
    (dsp, depth): (&Dsp, BitDepth),
    view: &mut PlaneViewMut<'_, T>,
    (x, y): (usize, usize),
    mb: &MbState,
    intra: IntraNeighbours,
    residual: &mut Residual<T::Coeff>,
) -> Result<()> {
    if mb.transform_8x8 {
        for b8 in 0..4 {
            let (x4, y4) = ((b8 % 2) * 2, (b8 / 2) * 2);
            let (px, py) = (x + x4 * 4, y + y4 * 4);
            let mode = mb.intra_modes[y4 * 4 + x4];
            let avail = (intra.block(x4, y4, 2), depth);
            if PLANE == 0 {
                T::intra_8x8(dsp, view, (px, py), mode, avail)?;
            } else {
                intra::predict_nxn(view, (px, py), 8, mode, avail)?;
            }
            let mut dst = block_view(view, (px, py), 8);
            residual.luma(PLANE).add_8x8((dsp, depth), b8, &mut dst);
        }
        return Ok(());
    }
    for index in 0..16 {
        let (x4, y4) = mbstate::block_position(index);
        let (px, py) = (x + x4 * 4, y + y4 * 4);
        let mode = mb.intra_modes[y4 * 4 + x4];
        let avail = (intra.block(x4, y4, 1), depth);
        if PLANE == 0 {
            T::intra_4x4(dsp, view, (px, py), mode, avail)?;
        } else {
            intra::predict_nxn(view, (px, py), 4, mode, avail)?;
        }
        let mut dst = block_view(view, (px, py), 4);
        residual
            .luma(PLANE)
            .add_4x4((dsp, depth), y4 * 4 + x4, &mut dst);
    }
    Ok(())
}

/// Add the residual of component `PLANE` of a predicted macroblock (4x4 or
/// 8x8 blocks).
#[inline]
fn add_luma<T: H264Sample, const PLANE: usize>(
    (dsp, depth): (&Dsp, BitDepth),
    view: &mut PlaneViewMut<'_, T>,
    (x, y): (usize, usize),
    mb: &MbState,
    residual: &mut Residual<T::Coeff>,
) {
    let blocks = residual.luma(PLANE);
    if !blocks.has_any() {
        return;
    }
    if mb.transform_8x8 {
        for b8 in 0..4 {
            if !blocks.has_8x8(b8) {
                continue;
            }
            let (px, py) = (x + (b8 % 2) * 8, y + (b8 / 2) * 8);
            blocks.add_8x8((dsp, depth), b8, &mut block_view(view, (px, py), 8));
        }
        return;
    }
    blocks.add_4x4_all((dsp, depth), &mut block_view(view, (x, y), MB_SIZE));
}

/// Copy `I_PCM` samples (luma raster, then Cb and Cr raster, each
/// `MbWidthC` x `MbHeightC`).
fn pcm<T: Sample>(
    planes: &mut Planes<'_, T>,
    (x, y): (usize, usize),
    extra: &MbExtra,
    chroma: ChromaArray,
) {
    let mut copy =
        |index: PlaneIndex, offset: usize, at: (usize, usize), (w, h): (usize, usize)| {
            let mut dst = rect_view(&mut planes[index as usize], at, (w, h));
            for row in 0..h {
                let start = offset + row * w;
                let samples = &extra.pcm[start..start + w];
                for (out, &value) in dst.row_mut(row).iter_mut().zip(samples) {
                    *out = T::from_i32(i32::from(value));
                }
            }
        };
    copy(PlaneIndex::Y, 0, (x, y), (MB_SIZE, MB_SIZE));
    if chroma.is_mono() {
        return;
    }
    let (sx, sy) = chroma.subsampling();
    let samples = chroma.pcm_chroma_samples();
    let (at, size) = ((x >> sx, y >> sy), chroma.mb_size());
    copy(PlaneIndex::U, PCM_LUMA_SAMPLES, at, size);
    copy(PlaneIndex::V, PCM_LUMA_SAMPLES + samples, at, size);
}
