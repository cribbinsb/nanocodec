//! Reconstruction of one decoded macroblock into the picture: `I_PCM`
//! samples, intra prediction block by block with its residual (§8.3), or
//! inter prediction (`inter.rs`) followed by the residual (§8.5); the P
//! macroblocks of SP slices and SI macroblocks requantise their prediction
//! instead (§8.6, `switching.rs`).

use nc_common::{BitDepth, BitDepths, PlaneIndex, PlaneViewMut, Sample};
use nc_dsp::Dsp;

use super::inter::InterContext;
use super::intra::{self, Avail};
use super::mb::{MbExtra, PCM_LUMA_SAMPLES};
use super::mbstate::{self, MbKind, MbState, Neighbours};
use super::sample::H264Sample;
use super::switching::{Switching, SwitchingLevels};
use super::transform::Residual;
use super::{Planes, Result};

const MB_SIZE: usize = 16;
const CHROMA_MB_SIZE: usize = 8;

/// Which neighbouring macroblocks intra prediction may read (§8.3.1.2,
/// §8.3.3, §8.3.4: with `constrained_intra_pred_flag`, inter macroblocks
/// count as unavailable, and `SI` ones unless the current one is `SI`).
#[derive(Debug, Clone, Copy)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "one availability per neighbouring macroblock"
)]
struct IntraNeighbours {
    a: bool,
    b: bool,
    c: bool,
    d: bool,
}

impl IntraNeighbours {
    fn of(nb: &Neighbours<'_>, constrained: bool, si: bool) -> Self {
        let usable = |m: Option<&MbState>| {
            m.is_some_and(|m| !constrained || (m.is_intra() && (si || m.kind != MbKind::Si)))
        };
        Self {
            a: usable(nb.a),
            b: usable(nb.b),
            c: usable(nb.c),
            d: usable(nb.d),
        }
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
            left: x4 > 0 || self.a,
            top: y4 > 0 || self.b,
            top_right,
            top_left: match (x4 > 0, y4 > 0) {
                (true, true) => true,
                (false, true) => self.a,
                (true, false) => self.b,
                (false, false) => self.d,
            },
        }
    }
}

/// Everything reconstruction needs besides the macroblock.
pub(super) struct ReconContext<'a, T: Sample> {
    pub dsp: &'a Dsp,
    pub inter: InterContext<'a, T>,
    pub constrained_intra_pred: bool,
    pub monochrome: bool,
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
    let intra = IntraNeighbours::of(nb, ctx.constrained_intra_pred, mb.kind == MbKind::Si);
    // Folds to the 8-bit constants in the 8-bit instance.
    let depths = ctx.depths.of::<T>();
    if mb.kind == MbKind::IPcm {
        pcm(planes, (x, y), extra, ctx.monochrome);
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
    let dsp = (ctx.dsp, depths.luma);
    match mb.kind {
        // `SI` outside an SI slice is not decoded (`mb.rs`).
        MbKind::IntraNxN | MbKind::Si => luma_nxn(dsp, luma, (x, y), mb, intra, residual)?,
        MbKind::Intra16x16(mode) => {
            intra::predict_16x16(luma, (x, y), mode, (intra.block(0, 0, 4), depths.luma))?;
            add_luma(dsp, luma, (x, y), mb, residual);
        }
        MbKind::Inter(_) | MbKind::PSkip | MbKind::BSkip | MbKind::BDirect16x16 | MbKind::IPcm => {
            add_luma(dsp, luma, (x, y), mb, residual);
        }
    }
    // Inter chroma without coefficients is the prediction already.
    if ctx.monochrome || (!mb.is_intra() && !residual.has_chroma_any()) {
        return Ok(());
    }
    let (cx, cy) = (x / 2, y / 2);
    for (comp, plane) in [cb, cr].into_iter().enumerate() {
        if mb.is_intra() {
            let avail = (intra.block(0, 0, 4), depths.chroma);
            intra::predict_chroma(plane, (cx, cy), mb.chroma_pred_mode, avail)?;
        }
        for b in 0..4 {
            if !residual.has_chroma(comp, b) {
                continue;
            }
            let mut dst = block_view(plane, (cx + (b % 2) * 4, cy + (b / 2) * 4), 4);
            residual.add_chroma_4x4(ctx.dsp, depths.chroma, (comp, b), &mut dst);
        }
    }
    Ok(())
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
    if ctx.monochrome {
        return Ok(());
    }
    let (cx, cy) = (x / 2, y / 2);
    for (comp, plane) in [cb, cr].into_iter().enumerate() {
        if si {
            let avail = (intra.block(0, 0, 4), depths.chroma);
            intra::predict_chroma(plane, (cx, cy), mb.chroma_pred_mode, avail)?;
        }
        let dsp = (ctx.dsp, depths.chroma);
        let mut block = block_view(plane, (cx, cy), CHROMA_MB_SIZE);
        switching.chroma(dsp, levels, (comp, qp), &mut block);
    }
    Ok(())
}

/// The `side` x `side` block at `(x, y)` of `plane`.
fn block_view<'p, T: Sample>(
    plane: &'p mut PlaneViewMut<'_, T>,
    (x, y): (usize, usize),
    side: usize,
) -> PlaneViewMut<'p, T> {
    plane
        .sub_view_mut(x, y, side, side)
        .expect("block inside the picture")
}

/// `Intra_4x4` or `Intra_8x8`: predict each block, then add its residual,
/// in decoding order.
fn luma_nxn<T: H264Sample>(
    (dsp, depth): (&Dsp, BitDepth),
    plane: &mut PlaneViewMut<'_, T>,
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
            T::intra_8x8(dsp, plane, (px, py), mode, (intra.block(x4, y4, 2), depth))?;
            residual.add_luma_8x8(dsp, depth, b8, &mut block_view(plane, (px, py), 8));
        }
        return Ok(());
    }
    for index in 0..16 {
        let (x4, y4) = mbstate::block_position(index);
        let (px, py) = (x + x4 * 4, y + y4 * 4);
        let mode = mb.intra_modes[y4 * 4 + x4];
        T::intra_4x4(dsp, plane, (px, py), mode, (intra.block(x4, y4, 1), depth))?;
        residual.add_luma_4x4(dsp, depth, y4 * 4 + x4, &mut block_view(plane, (px, py), 4));
    }
    Ok(())
}

/// Add the luma residual of a predicted macroblock (4x4 or 8x8 blocks).
#[inline]
fn add_luma<T: H264Sample>(
    (dsp, depth): (&Dsp, BitDepth),
    plane: &mut PlaneViewMut<'_, T>,
    (x, y): (usize, usize),
    mb: &MbState,
    residual: &mut Residual<T::Coeff>,
) {
    if !residual.has_luma() {
        return;
    }
    if mb.transform_8x8 {
        for b8 in 0..4 {
            if !residual.has_luma_8x8(b8) {
                continue;
            }
            let (px, py) = (x + (b8 % 2) * 8, y + (b8 / 2) * 8);
            residual.add_luma_8x8(dsp, depth, b8, &mut block_view(plane, (px, py), 8));
        }
        return;
    }
    residual.add_luma_4x4_all(dsp, depth, &mut block_view(plane, (x, y), MB_SIZE));
}

/// Copy `I_PCM` samples (luma raster, then Cb and Cr raster).
fn pcm<T: Sample>(
    planes: &mut Planes<'_, T>,
    (x, y): (usize, usize),
    extra: &MbExtra,
    monochrome: bool,
) {
    let mut copy = |index: PlaneIndex, offset: usize, (px, py): (usize, usize), side: usize| {
        let mut dst = block_view(&mut planes[index as usize], (px, py), side);
        for row in 0..side {
            let start = offset + row * side;
            let samples = &extra.pcm[start..start + side];
            for (out, &value) in dst.row_mut(row).iter_mut().zip(samples) {
                *out = T::from_i32(i32::from(value));
            }
        }
    };
    copy(PlaneIndex::Y, 0, (x, y), MB_SIZE);
    if monochrome {
        return;
    }
    let chroma = CHROMA_MB_SIZE * CHROMA_MB_SIZE;
    copy(
        PlaneIndex::U,
        PCM_LUMA_SAMPLES,
        (x / 2, y / 2),
        CHROMA_MB_SIZE,
    );
    copy(
        PlaneIndex::V,
        PCM_LUMA_SAMPLES + chroma,
        (x / 2, y / 2),
        CHROMA_MB_SIZE,
    );
}
