//! Reconstruction of one decoded macroblock into the picture: `I_PCM`
//! samples, intra prediction block by block with its residual (§8.3), or
//! inter prediction (`inter.rs`) followed by the residual (§8.5).

use nc_common::{PlaneIndex, PlaneViewMut};
use nc_dsp::Dsp;

use super::inter::InterContext;
use super::intra::{self, Avail};
use super::mb::{MbExtra, PCM_LUMA_SAMPLES};
use super::mbstate::{self, MbKind, MbState, Neighbours};
use super::transform::Residual;
use super::{Planes, Result};

const MB_SIZE: usize = 16;
const CHROMA_MB_SIZE: usize = 8;

/// Which neighbouring macroblocks intra prediction may read (§8.3.1.2:
/// with `constrained_intra_pred_flag`, inter macroblocks count as
/// unavailable).
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
    fn of(nb: &Neighbours<'_>, constrained: bool) -> Self {
        let usable = |m: Option<&MbState>| m.is_some_and(|m| !constrained || m.is_intra());
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
pub(super) struct ReconContext<'a> {
    pub dsp: &'a Dsp,
    pub inter: InterContext<'a>,
    pub constrained_intra_pred: bool,
    pub monochrome: bool,
}

/// Reconstruct macroblock `mb` at macroblock position (`mx`, `my`).
pub(super) fn reconstruct(
    ctx: &ReconContext<'_>,
    planes: &mut Planes<'_>,
    (mx, my): (usize, usize),
    mb: &MbState,
    nb: &Neighbours<'_>,
    residual: &mut Residual,
    extra: &MbExtra,
) -> Result<()> {
    let (x, y) = (mx * MB_SIZE, my * MB_SIZE);
    let intra = IntraNeighbours::of(nb, ctx.constrained_intra_pred);
    if mb.kind == MbKind::IPcm {
        pcm(planes, (x, y), extra, ctx.monochrome);
        return Ok(());
    }
    if !mb.is_intra() {
        ctx.inter.predict(planes, mb, mx, my)?;
    }
    let [luma, cb, cr] = planes;
    match mb.kind {
        MbKind::IntraNxN => luma_nxn(ctx.dsp, luma, (x, y), mb, intra, residual)?,
        MbKind::Intra16x16(mode) => {
            intra::predict_16x16(luma, x, y, mode, intra.block(0, 0, 4))?;
            add_luma(ctx.dsp, luma, (x, y), mb, residual);
        }
        MbKind::Inter(_) | MbKind::PSkip | MbKind::IPcm => {
            add_luma(ctx.dsp, luma, (x, y), mb, residual);
        }
    }
    // Inter chroma without coefficients is the prediction already.
    if ctx.monochrome || (!mb.is_intra() && !residual.has_chroma_any()) {
        return Ok(());
    }
    let (cx, cy) = (x / 2, y / 2);
    for (comp, plane) in [cb, cr].into_iter().enumerate() {
        if mb.is_intra() {
            intra::predict_chroma(plane, cx, cy, mb.chroma_pred_mode, intra.block(0, 0, 4))?;
        }
        for b in 0..4 {
            if !residual.has_chroma(comp, b) {
                continue;
            }
            let mut dst = block_view(plane, (cx + (b % 2) * 4, cy + (b / 2) * 4), 4);
            residual.add_chroma_4x4(ctx.dsp, comp, b, &mut dst);
        }
    }
    Ok(())
}

/// The `side` x `side` block at `(x, y)` of `plane`.
fn block_view<'p>(
    plane: &'p mut PlaneViewMut<'_>,
    (x, y): (usize, usize),
    side: usize,
) -> PlaneViewMut<'p> {
    plane
        .sub_view_mut(x, y, side, side)
        .expect("block inside the picture")
}

/// `Intra_4x4` or `Intra_8x8`: predict each block, then add its residual,
/// in decoding order.
fn luma_nxn(
    dsp: &Dsp,
    plane: &mut PlaneViewMut<'_>,
    (x, y): (usize, usize),
    mb: &MbState,
    intra: IntraNeighbours,
    residual: &mut Residual,
) -> Result<()> {
    if mb.transform_8x8 {
        for b8 in 0..4 {
            let (x4, y4) = ((b8 % 2) * 2, (b8 / 2) * 2);
            let (px, py) = (x + x4 * 4, y + y4 * 4);
            let mode = mb.intra_modes[y4 * 4 + x4];
            intra::predict_nxn(plane, px, py, 8, mode, intra.block(x4, y4, 2))?;
            residual.add_luma_8x8(dsp, b8, &mut block_view(plane, (px, py), 8));
        }
        return Ok(());
    }
    for index in 0..16 {
        let (x4, y4) = mbstate::block_position(index);
        let (px, py) = (x + x4 * 4, y + y4 * 4);
        intra::predict_nxn(
            plane,
            px,
            py,
            4,
            mb.intra_modes[y4 * 4 + x4],
            intra.block(x4, y4, 1),
        )?;
        residual.add_luma_4x4(dsp, y4 * 4 + x4, &mut block_view(plane, (px, py), 4));
    }
    Ok(())
}

/// Add the luma residual of a predicted macroblock (4x4 or 8x8 blocks).
fn add_luma(
    dsp: &Dsp,
    plane: &mut PlaneViewMut<'_>,
    (x, y): (usize, usize),
    mb: &MbState,
    residual: &mut Residual,
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
            residual.add_luma_8x8(dsp, b8, &mut block_view(plane, (px, py), 8));
        }
        return;
    }
    residual.add_luma_4x4_all(dsp, &mut block_view(plane, (x, y), MB_SIZE));
}

/// Copy `I_PCM` samples (luma raster, then Cb and Cr raster).
fn pcm(planes: &mut Planes<'_>, (x, y): (usize, usize), extra: &MbExtra, monochrome: bool) {
    let mut copy = |index: PlaneIndex, offset: usize, (px, py): (usize, usize), side: usize| {
        let mut dst = block_view(&mut planes[index as usize], (px, py), side);
        for row in 0..side {
            let start = offset + row * side;
            dst.row_mut(row)
                .copy_from_slice(&extra.pcm[start..start + side]);
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
