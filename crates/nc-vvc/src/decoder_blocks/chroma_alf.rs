//! Chroma ALF and CC-ALF boundary preparation (which samples each output
//! reads: the ALF virtual boundary, picture edges and closed edges through
//! the shared ALF `Region`); sample arithmetic lives in shared DSP.
use super::{boundaries::Closures, filter_syntax::CtuFilters, filters::alf_bank};
use crate::{alf::Region, decoder_headers::Adaptation};
use nc_common::{
    Picture, PlaneIndex,
    decode::{DecodeError, Result},
};
use nc_dsp::{DepthKernels, Pixel, vvc_alf_clip_value};

/// The luma regions CTB `index` is filtered in: the parts `closures` names,
/// or the whole CTB.
fn regions(closures: &Closures, index: usize, ctu: usize, size: (usize, usize)) -> Vec<Region> {
    closures.alf_regions(index).unwrap_or_else(|| {
        let columns = size.0.div_ceil(ctu);
        let (x, y) = (index % columns * ctu, index / columns * ctu);
        vec![Region {
            rect: (x, y, ctu.min(size.0 - x), ctu.min(size.1 - y)),
            closed: [false; 4],
            corners: [false; 2],
        }]
    })
}

pub(super) fn apply<T: Pixel>(
    picture: &mut Picture<T>,
    filters: &[CtuFilters],
    (ctu, closures): (usize, &Closures),
    banks: &[Option<Box<Adaptation>>; 8],
    dsp: DepthKernels<'_>,
) -> Result<()> {
    let size = (picture.width(), picture.height());
    let side = ctu / 2;
    for plane in 1..3 {
        let source = picture.plane(PlaneIndex::ALL[plane]).clone();
        let input = source.view();
        let mut output = picture.plane_mut(PlaneIndex::ALL[plane]).view_mut();
        let mut padded: [Vec<T>; 7] = std::array::from_fn(|_| vec![T::default(); side + 6]);
        for (index, filter) in filters.iter().enumerate() {
            let (Some(alternative), Some(id)) = (filter.alf[plane], filter.banks.chroma) else {
                continue;
            };
            let selected = alf_bank(banks, id)?
                .chroma
                .get(usize::from(alternative))
                .ok_or(DecodeError::Malformed("VVC chroma ALF alternative"))?;
            let limits = selected
                .clip
                .map(|c| vvc_alf_clip_value(usize::from(c), dsp.depth()));
            for region in regions(closures, index, ctu, size) {
                let region = region.halved();
                let (x0, y0, width, height) = region.rect;
                let plane_size = (input.width(), input.height());
                for y in y0..y0 + height {
                    let position = y % side;
                    let boundary = side - 2;
                    let distance = if position < boundary {
                        boundary - 1 - position
                    } else {
                        position - boundary
                    };
                    let reach = distance.min(2) as isize;
                    for (row, samples) in padded.iter_mut().enumerate() {
                        let dy = (row as isize - 3).clamp(-reach, reach);
                        for (x, sample) in samples[..width + 6].iter_mut().enumerate() {
                            let at = (x0 as isize + x as isize - 3, y as isize + dy);
                            let (sx, sy) = region.source(at, plane_size);
                            *sample = input.row(sy)[sx];
                        }
                    }
                    dsp.vvc_alf_filter_chroma_row(
                        &padded.each_ref().map(Vec::as_slice),
                        selected.coefficients,
                        limits,
                        if distance == 0 { 10 } else { 7 },
                        &mut output.row_mut(y)[x0..x0 + width],
                    );
                }
            }
        }
    }
    cross(picture, filters, (ctu, closures), banks, dsp)
}

fn cross<T: Pixel>(
    picture: &mut Picture<T>,
    filters: &[CtuFilters],
    (ctu, closures): (usize, &Closures),
    banks: &[Option<Box<Adaptation>>; 8],
    dsp: DepthKernels<'_>,
) -> Result<()> {
    if !filters.iter().any(|f| f.cross != [0; 2]) {
        return Ok(());
    }
    // Called before luma ALF, retaining the pre-ALF luma required by §8.8.5.
    let luma = picture.plane(PlaneIndex::Y).clone();
    let input = luma.view();
    let size = (input.width(), input.height());
    let mut padded: [Vec<T>; 4] = std::array::from_fn(|_| vec![T::default(); ctu + 2]);
    for plane in 0..2 {
        let mut output = picture.plane_mut(PlaneIndex::ALL[plane + 1]).view_mut();
        for (index, filter) in filters.iter().enumerate() {
            let selection = filter.cross[plane];
            let Some(id) = filter.banks.cross[plane].filter(|_| selection != 0) else {
                continue;
            };
            let coefficients = *alf_bank(banks, id)?.cross[plane]
                .get(usize::from(selection - 1))
                .ok_or(DecodeError::Malformed("VVC cross-component ALF filter"))?;
            // The luma regions, read around each chroma sample's luma.
            for region in regions(closures, index, ctu, size) {
                let (x0, y0, width, height) = region.halved().rect;
                for y in y0..y0 + height {
                    let position = (2 * y) % ctu;
                    let boundary = ctu - 4;
                    let offsets = if position == boundary {
                        [0, 0, 0, 0]
                    } else if position == boundary - 2 {
                        [-1, 0, 1, 1]
                    } else {
                        [-1, 0, 1, 2]
                    };
                    for (samples, dy) in padded.iter_mut().zip(offsets) {
                        for (x, sample) in samples[..2 * width + 2].iter_mut().enumerate() {
                            let at = (2 * x0 as isize + x as isize - 1, 2 * y as isize + dy);
                            let (sx, sy) = region.source(at, size);
                            *sample = input.row(sy)[sx];
                        }
                    }
                    dsp.vvc_alf_cross_row(
                        &padded.each_ref().map(Vec::as_slice),
                        coefficients,
                        &mut output.row_mut(y)[x0..x0 + width],
                    );
                }
            }
        }
    }
    Ok(())
}
