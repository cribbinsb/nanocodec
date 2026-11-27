//! The nc-dsp inter kernels as the prediction calls them: 8-bit, with
//! a picture or intermediate as (slice, offset, stride) and dav2d's
//! filter numbers (`nc_dsp::av2_dec`: `mc_8tap`, `compound`, `warp`,
//! `opfl`, `bacp`, `wedge`).
use nc_common::{BitDepth, Picture, PlaneIndex};
use nc_dsp::av2_dec::compound::{self, MaskLayout};
use nc_dsp::av2_dec::mc_8tap::{self, InterpFilter};
use nc_dsp::av2_dec::wedge::InterMasks;
use nc_dsp::av2_dec::{bacp, opfl, warp};
use nc_dsp::{BlockSize, Dsp, Filter};

pub use nc_dsp::av2_dec::opfl::OpflRegressionData as Regression;

/// Rows of a reference window built by repeating edge samples.
pub(super) const WINDOW_STRIDE: usize = 192;
/// The decoder codes 8-bit samples only.
const DEPTH: BitDepth = BitDepth::EIGHT;

/// A plane of a reference picture: its samples from (0, 0) and stride.
pub(super) fn plane(picture: &Picture, index: usize) -> (&[u8], usize) {
    let plane = picture.plane(PlaneIndex::ALL[index]);
    let samples = plane
        .samples_at(0, 0, plane.width(), plane.height())
        .expect("the visible area is inside the plane");
    (samples, plane.stride())
}

/// dav2d's filter number as the kernels' filter.
fn interp(filter: u8) -> InterpFilter {
    InterpFilter::from_index(filter).unwrap_or(InterpFilter::Regular)
}

/// The AV1-family entry points take the full eight-tap rectangle even
/// for a single fractional axis. AV2's edge scratch may contain only the
/// requested axes. Keep those smaller windows on the reference path.
#[inline]
fn shared_translational(
    filter: InterpFilter,
    (src, offset, stride): (&[u8], usize, usize),
    (w, h): (usize, usize),
) -> Option<(Filter, BlockSize, &[u8])> {
    let filter = match filter {
        InterpFilter::Regular => Filter::Av1,
        InterpFilter::Smooth => Filter::Av1Smooth,
        InterpFilter::Sharp => Filter::Av1Sharp,
        InterpFilter::Bilinear => return None,
    };
    let size = BlockSize::new(w, h)?;
    let before = filter.before();
    let extra = filter.taps() - 1;
    let col = offset.checked_rem(stride)?;
    if col < before || col + w + (extra - before) > stride {
        return None;
    }
    let start = offset.checked_sub(before * stride + before)?;
    let end = start + (h + extra - 1) * stride + w + extra;
    Some((filter, size, src.get(start..end)?))
}

/// A translational prediction into 8-bit samples (dav2d `mc[filter]`),
/// phases in sixteenths.
pub(super) fn put(
    dsp: &Dsp,
    filter: u8,
    (dst, dst_offset, dst_stride): (&mut [u8], usize, usize),
    (src, src_offset, src_stride): (&[u8], usize, usize),
    size: (usize, usize),
    phase: (i32, i32),
) {
    if phase == (0, 0) {
        dsp.block_copies().copy_rect(
            &src[src_offset..],
            src_stride,
            &mut dst[dst_offset..],
            dst_stride,
            size,
            false,
        );
        return;
    }
    let filter = interp(filter);
    if let Some((family, block, window)) =
        shared_translational(filter, (src, src_offset, src_stride), size)
    {
        dsp.predict_samples(
            family,
            block,
            (window, src_stride),
            (phase.0 as usize, phase.1 as usize),
            (&mut dst[dst_offset..], dst_stride),
        );
        return;
    }
    mc_8tap::put_8tap(
        dst, dst_offset, dst_stride, src, src_offset, src_stride, size, phase, filter, DEPTH,
    );
}

/// A translational prediction into an intermediate (dav2d `mct[filter]`).
pub(super) fn prep(
    dsp: &Dsp,
    filter: u8,
    (tmp, tmp_offset, tmp_stride): (&mut [i16], usize, usize),
    (src, src_offset, src_stride): (&[u8], usize, usize),
    size: (usize, usize),
    phase: (i32, i32),
) {
    if phase == (0, 0) {
        dsp.block_copies().prep_rect(
            &src[src_offset..],
            src_stride,
            &mut tmp[tmp_offset..],
            tmp_stride,
            size,
            (4, 0),
        );
        return;
    }
    let filter = interp(filter);
    if let Some((family, block, window)) =
        shared_translational(filter, (src, src_offset, src_stride), size)
    {
        dsp.predict_compound_samples(
            family,
            block,
            (window, src_stride),
            (phase.0 as usize, phase.1 as usize),
            (&mut tmp[tmp_offset..], tmp_stride),
        );
        return;
    }
    mc_8tap::prep_8tap(
        &mut tmp[tmp_offset..],
        tmp_stride,
        src,
        src_offset,
        src_stride,
        size,
        phase,
        filter,
        DEPTH,
    );
}

/// An 8x8 warp into samples (dav2d `warp8x8`).
pub(super) fn warp(
    dsp: &Dsp,
    dst: (&mut [u8], usize, usize),
    src: (&[u8], usize, usize),
    shear: [i16; 4],
    phase: (i32, i32),
) {
    warp::warp_affine_8x8_with_dsp(dsp, dst, src, shear, phase);
}

/// An 8x8 warp into an intermediate (dav2d `warp8x8t`).
pub(super) fn warp_prep(
    dsp: &Dsp,
    (tmp, tmp_offset, tmp_stride): (&mut [i16], usize, usize),
    src: (&[u8], usize, usize),
    shear: [i16; 4],
    phase: (i32, i32),
) {
    warp::warp_affine_8x8t_with_dsp(dsp, (&mut tmp[tmp_offset..], tmp_stride), src, shear, phase);
}

/// A 4x4 warp sub-block into samples (dav2d `ext_warp4x4`).
pub(super) fn ext_warp(
    (dst, dst_offset, dst_stride): (&mut [u8], usize, usize),
    (src, src_offset, src_stride): (&[u8], usize, usize),
    phase: (i32, i32),
) {
    mc_8tap::ext_warp4x4(
        dst, dst_offset, dst_stride, src, src_offset, src_stride, phase, DEPTH,
    );
}

/// A 4x4 warp sub-block into an intermediate (dav2d `ext_warp4x4t`).
pub(super) fn ext_warp_prep(
    (tmp, tmp_offset, tmp_stride): (&mut [i16], usize, usize),
    (src, src_offset, src_stride): (&[u8], usize, usize),
    phase: (i32, i32),
) {
    mc_8tap::ext_warp4x4t(
        &mut tmp[tmp_offset..],
        tmp_stride,
        src,
        src_offset,
        src_stride,
        phase,
        DEPTH,
    );
}

/// The average of two intermediates (dav2d `avg`).
pub(super) fn avg(
    (dst, offset, stride): (&mut [u8], usize, usize),
    (a, b): (&[i16], &[i16]),
    size: (usize, usize),
) {
    compound::avg(dst, offset, stride, a, b, size, DEPTH);
}

/// Their weighted average, the first weighing `weight` sixteenths
/// (dav2d `w_avg`).
pub(super) fn w_avg(
    (dst, offset, stride): (&mut [u8], usize, usize),
    (a, b): (&[i16], &[i16]),
    size: (usize, usize),
    weight: i8,
) {
    compound::w_avg(dst, offset, stride, a, b, size, i32::from(weight), DEPTH);
}

/// Their blend by a per-sample mask of the first (dav2d `mask`).
pub(super) fn mask(
    (dst, offset, stride): (&mut [u8], usize, usize),
    (a, b): (&[i16], &[i16]),
    size: (usize, usize),
    weights: &[u8],
) {
    compound::mask(dst, offset, stride, a, b, size, weights, DEPTH);
}

/// Their blend by a mask of their difference, writing the mask at 4:2:0
/// chroma resolution (dav2d `w_mask[2]`).
pub(super) fn w_mask_420(
    (dst, offset, stride): (&mut [u8], usize, usize),
    (a, b): (&[i16], &[i16]),
    size: (usize, usize),
    (weights, weights_stride): (&mut [u8], usize),
    sign: bool,
) {
    compound::w_mask(
        dst,
        offset,
        stride,
        a,
        b,
        size,
        (weights, weights_stride),
        MaskLayout::I420,
        sign,
        DEPTH,
    );
}

/// An intra prediction blended into samples by a mask (dav2d `blend`).
pub(super) fn blend(
    (dst, offset, stride): (&mut [u8], usize, usize),
    intra: &[u8],
    size: (usize, usize),
    weights: &[u8],
) {
    compound::blend(dst, offset, stride, intra, size, weights);
}

/// The implicit mask of predictions starting at `first` and `second` in
/// a `frame`-sized reference (dav2d `bacp_mask`).
pub(super) fn bacp_mask(
    (weights, stride): (&mut [u8], usize),
    size: (usize, usize),
    (first, second): ((i32, i32), (i32, i32)),
    (w, h): (i32, i32),
) {
    bacp::bacp_mask(weights, stride, size, first, second, (w as u32, h as u32));
}

/// The SAD search of MV refinement over two bilinear windows (dav2d
/// `sad_refine_mv`): the offset of the first (the second moves
/// opposite).
pub(super) fn sad_refine(
    (a, b, stride): (&[u8], &[u8], usize),
    size: (usize, usize),
    implicit: bool,
) -> (i32, i32) {
    let offset = opfl::sad_refine_mv((a, 0, stride), (b, 0, stride), size, implicit, DEPTH);
    (i32::from(offset.y), i32::from(offset.x))
}

/// The 8x8 SAD of two windows (dav2d `sad8x8`).
pub(super) fn sad8x8(
    (a, a_offset): (&[u8], usize),
    (b, b_offset): (&[u8], usize),
    stride: usize,
) -> u32 {
    opfl::sad8x8((a, a_offset, stride), (b, b_offset, stride), DEPTH)
}

/// The optical-flow regressions of `block`-sided sub-blocks (dav2d
/// `opfl_refine_mv`).
pub(super) fn flow_regression(
    out: &mut [Regression],
    (a, a_offset): (&[u8], usize),
    (b, b_offset): (&[u8], usize),
    stride: usize,
    size: (usize, usize),
    block: usize,
    distances: [i8; 2],
) {
    opfl::opfl_refine_mv(
        out,
        (a, a_offset, stride),
        (b, b_offset, stride),
        size,
        block,
        distances,
        DEPTH,
    );
}

/// The wedge mask `index` of a `w` x `h` (luma samples) block, at 4:2:0
/// chroma resolution when `chroma`.
pub(super) fn wedge_mask(
    masks: &InterMasks,
    size: (usize, usize),
    index: i8,
    chroma: bool,
) -> &[u8] {
    let layout = if chroma {
        MaskLayout::I420
    } else {
        MaskLayout::I444
    };
    masks.wedge(size, index as usize, layout)
}

/// The inter-intra smooth mask of `mode` (dav2d `InterIntraPredMode`)
/// for a `w` x `h` plane block.
pub(super) fn inter_intra_mask(masks: &InterMasks, size: (usize, usize), mode: u8) -> &[u8] {
    use nc_dsp::av1_mc::mask::Av1InterIntraMode;
    let mode = match mode {
        0 => Av1InterIntraMode::Dc,
        1 => Av1InterIntraMode::Vertical,
        2 => Av1InterIntraMode::Horizontal,
        _ => Av1InterIntraMode::Smooth,
    };
    masks.interintra(size, mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_translational_matches_av2_reference() {
        let dsp = Dsp::new();
        for (w, h) in [(4, 4), (4, 8), (8, 4), (8, 8), (16, 32), (64, 64)] {
            let stride = w + 7;
            let origin = 3 * (stride + 1);
            let src: Vec<_> = (0..stride * (h + 7))
                .map(|i| ((i * 43) ^ (i * i * 71)) as u8)
                .collect();
            let mut pixels = vec![97_u8; (w + 4) * h];
            let mut intermediates = vec![-97_i16; pixels.len()];
            let out_stride = w + 4;
            for filter in 0..3 {
                for mx in 0..16 {
                    for my in 0..16 {
                        let mut ref_pixels = pixels.clone();
                        let mut ref_intermediates = intermediates.clone();
                        let (size, phase) = ((w, h), (mx, my));
                        put(
                            &dsp,
                            filter,
                            (&mut pixels, 2, out_stride),
                            (&src, origin, stride),
                            size,
                            phase,
                        );
                        prep(
                            &dsp,
                            filter,
                            (&mut intermediates, 2, out_stride),
                            (&src, origin, stride),
                            size,
                            phase,
                        );
                        mc_8tap::put_8tap(
                            &mut ref_pixels,
                            2,
                            out_stride,
                            &src,
                            origin,
                            stride,
                            size,
                            phase,
                            interp(filter),
                            DEPTH,
                        );
                        mc_8tap::prep_8tap(
                            &mut ref_intermediates[2..],
                            out_stride,
                            &src,
                            origin,
                            stride,
                            size,
                            phase,
                            interp(filter),
                            DEPTH,
                        );
                        assert_eq!(pixels, ref_pixels, "{size:?} {filter} {phase:?}");
                        assert_eq!(
                            intermediates, ref_intermediates,
                            "{size:?} {filter} {phase:?}"
                        );
                    }
                }
            }
        }
    }
}
