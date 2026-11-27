//! The nc-dsp inter kernels as the prediction calls them: samples of the
//! decoder's type at the sequence's depth, with a picture or intermediate
//! as (slice, offset, stride) and dav2d's filter numbers (`nc_dsp::av2_dec`:
//! `mc_8tap`, `compound`, `warp`, `opfl`, `bacp`, `wedge`). 8-bit samples
//! take the shared SIMD kernels where they exist ([`Av2Sample::eight`]).
use nc_common::{BitDepth, Picture, PlaneIndex, Sample};
use nc_dsp::av2_dec::compound::{self, MaskLayout};
use nc_dsp::av2_dec::mc_8tap::{self, InterpFilter};
use nc_dsp::av2_dec::wedge::InterMasks;
use nc_dsp::av2_dec::{bacp, opfl, warp};
use nc_dsp::{BlockSize, Dsp, Filter};

use crate::general::sample::Av2Sample;

pub use nc_dsp::av2_dec::opfl::OpflRegressionData as Regression;

/// Rows of a reference window built by repeating edge samples.
pub(super) const WINDOW_STRIDE: usize = 192;

/// A plane of a reference picture: its samples from (0, 0) and stride.
pub(super) fn plane<T: Sample>(picture: &Picture<T>, index: usize) -> (&[T], usize) {
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

/// The compound shift of 8-bit samples (dav2d `intermediate_bits` 4; no
/// `PREP_BIAS`).
const EIGHT_BIT_PREP_SHIFT: u32 = 4;

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

/// A translational prediction into samples (dav2d `mc[filter]`), phases
/// in sixteenths.
pub(super) fn put<T: Av2Sample>(
    dsp: &Dsp,
    filter: u8,
    (dst, dst_offset, dst_stride): (&mut [T], usize, usize),
    (src, src_offset, src_stride): (&[T], usize, usize),
    (size, phase): ((usize, usize), (i32, i32)),
    depth: BitDepth,
) {
    let depth = depth.of::<T>();
    let filter = interp(filter);
    if let (Some(dst), Some(src)) = (T::eight_mut(dst), T::eight(src)) {
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
    }
    mc_8tap::put_8tap(
        dst, dst_offset, dst_stride, src, src_offset, src_stride, size, phase, filter, depth,
    );
}

/// A translational prediction into an intermediate (dav2d `mct[filter]`).
pub(super) fn prep<T: Av2Sample>(
    dsp: &Dsp,
    filter: u8,
    (tmp, tmp_offset, tmp_stride): (&mut [i16], usize, usize),
    (src, src_offset, src_stride): (&[T], usize, usize),
    (size, phase): ((usize, usize), (i32, i32)),
    depth: BitDepth,
) {
    let depth = depth.of::<T>();
    let filter = interp(filter);
    if let Some(src) = T::eight(src) {
        if phase == (0, 0) {
            dsp.block_copies().prep_rect(
                &src[src_offset..],
                src_stride,
                &mut tmp[tmp_offset..],
                tmp_stride,
                size,
                (EIGHT_BIT_PREP_SHIFT, 0),
            );
            return;
        }
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
        depth,
    );
}

/// An 8x8 warp into samples (dav2d `warp8x8`).
pub(super) fn warp<T: Av2Sample>(
    dsp: &Dsp,
    (dst, offset, stride): (&mut [T], usize, usize),
    (src, src_offset, src_stride): (&[T], usize, usize),
    (shear, phase): ([i16; 4], (i32, i32)),
    depth: BitDepth,
) {
    if let (Some(dst), Some(src)) = (T::eight_mut(dst), T::eight(src)) {
        let src = (src, src_offset, src_stride);
        warp::warp_affine_8x8_with_dsp(dsp, (dst, offset, stride), src, shear, phase);
    } else {
        warp::warp_affine_8x8(
            dst, offset, stride, src, src_offset, src_stride, shear, phase, depth,
        );
    }
}

/// An 8x8 warp into an intermediate (dav2d `warp8x8t`).
pub(super) fn warp_prep<T: Av2Sample>(
    dsp: &Dsp,
    (tmp, tmp_offset, tmp_stride): (&mut [i16], usize, usize),
    (src, src_offset, src_stride): (&[T], usize, usize),
    (shear, phase): ([i16; 4], (i32, i32)),
    depth: BitDepth,
) {
    let tmp = &mut tmp[tmp_offset..];
    if let Some(src) = T::eight(src) {
        let src = (src, src_offset, src_stride);
        warp::warp_affine_8x8t_with_dsp(dsp, (tmp, tmp_stride), src, shear, phase);
    } else {
        warp::warp_affine_8x8t(
            tmp, tmp_stride, src, src_offset, src_stride, shear, phase, depth,
        );
    }
}

/// A 4x4 warp sub-block into samples (dav2d `ext_warp4x4`).
pub(super) fn ext_warp<T: Av2Sample>(
    dsp: &Dsp,
    (dst, dst_offset, dst_stride): (&mut [T], usize, usize),
    (src, src_offset, src_stride): (&[T], usize, usize),
    phase: (i32, i32),
    depth: BitDepth,
) {
    if let (Some(dst), Some(src)) = (T::eight_mut(dst), T::eight(src)) {
        mc_8tap::ext_warp4x4_with_dsp(
            dsp,
            (dst, dst_offset, dst_stride),
            (src, src_offset, src_stride),
            phase,
        );
    } else {
        mc_8tap::ext_warp4x4(
            dst, dst_offset, dst_stride, src, src_offset, src_stride, phase, depth,
        );
    }
}

/// A 4x4 warp sub-block into an intermediate (dav2d `ext_warp4x4t`).
pub(super) fn ext_warp_prep<T: Av2Sample>(
    dsp: &Dsp,
    (tmp, tmp_offset, tmp_stride): (&mut [i16], usize, usize),
    (src, src_offset, src_stride): (&[T], usize, usize),
    phase: (i32, i32),
    depth: BitDepth,
) {
    let tmp = &mut tmp[tmp_offset..];
    if let Some(src) = T::eight(src) {
        let src = (src, src_offset, src_stride);
        mc_8tap::ext_warp4x4t_with_dsp(dsp, (tmp, tmp_stride), src, phase);
    } else {
        mc_8tap::ext_warp4x4t(tmp, tmp_stride, src, src_offset, src_stride, phase, depth);
    }
}

/// The average of two intermediates (dav2d `avg`).
pub(super) fn avg<T: Av2Sample>(
    (dst, offset, stride): (&mut [T], usize, usize),
    (a, b): (&[i16], &[i16]),
    size: (usize, usize),
    depth: BitDepth,
) {
    compound::avg(dst, offset, stride, a, b, size, depth.of::<T>());
}

/// Their weighted average, the first weighing `weight` sixteenths
/// (dav2d `w_avg`).
pub(super) fn w_avg<T: Av2Sample>(
    (dst, offset, stride): (&mut [T], usize, usize),
    (a, b): (&[i16], &[i16]),
    (size, weight): ((usize, usize), i8),
    depth: BitDepth,
) {
    let depth = depth.of::<T>();
    compound::w_avg(dst, offset, stride, a, b, size, i32::from(weight), depth);
}

/// Their blend by a per-sample mask of the first (dav2d `mask`).
pub(super) fn mask<T: Av2Sample>(
    (dst, offset, stride): (&mut [T], usize, usize),
    (a, b): (&[i16], &[i16]),
    (size, weights): ((usize, usize), &[u8]),
    depth: BitDepth,
) {
    compound::mask(dst, offset, stride, a, b, size, weights, depth.of::<T>());
}

/// Their blend by a mask of their difference, writing the mask at 4:2:0
/// chroma resolution (dav2d `w_mask[2]`).
pub(super) fn w_mask_420<T: Av2Sample>(
    (dst, offset, stride): (&mut [T], usize, usize),
    (a, b): (&[i16], &[i16]),
    size: (usize, usize),
    (weights, weights_stride): (&mut [u8], usize),
    (sign, depth): (bool, BitDepth),
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
        depth.of::<T>(),
    );
}

/// An intra prediction blended into samples by a mask (dav2d `blend`).
pub(super) fn blend<T: Sample>(
    (dst, offset, stride): (&mut [T], usize, usize),
    intra: &[T],
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
pub(super) fn sad_refine<T: Av2Sample>(
    (a, b, stride): (&[T], &[T], usize),
    size: (usize, usize),
    implicit: bool,
    depth: BitDepth,
) -> (i32, i32) {
    let depth = depth.of::<T>();
    let offset = opfl::sad_refine_mv((a, 0, stride), (b, 0, stride), size, implicit, depth);
    (i32::from(offset.y), i32::from(offset.x))
}

/// The 8x8 SAD of two windows (dav2d `sad8x8`).
pub(super) fn sad8x8<T: Av2Sample>(
    (a, a_offset): (&[T], usize),
    (b, b_offset): (&[T], usize),
    stride: usize,
    depth: BitDepth,
) -> u32 {
    opfl::sad8x8(
        (a, a_offset, stride),
        (b, b_offset, stride),
        depth.of::<T>(),
    )
}

/// The optical-flow regressions of `block`-sided sub-blocks (dav2d
/// `opfl_refine_mv`).
pub(super) fn flow_regression<T: Av2Sample>(
    out: &mut [Regression],
    (a, a_offset): (&[T], usize),
    (b, b_offset): (&[T], usize),
    (stride, size, block): (usize, (usize, usize), usize),
    distances: [i8; 2],
    depth: BitDepth,
) {
    opfl::opfl_refine_mv(
        out,
        (a, a_offset, stride),
        (b, b_offset, stride),
        size,
        block,
        distances,
        depth.of::<T>(),
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

    const DEPTH: BitDepth = BitDepth::EIGHT;

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
                            (size, phase),
                            DEPTH,
                        );
                        prep(
                            &dsp,
                            filter,
                            (&mut intermediates, 2, out_stride),
                            (&src, origin, stride),
                            (size, phase),
                            DEPTH,
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
