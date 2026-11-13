//! Chroma ALF boundary preparation; sample arithmetic lives in shared DSP.
use super::filter_syntax::CtuFilters;
use crate::decoder_headers::{Adaptation, AdaptationData, SliceHeader};
use nc_common::{
    Picture, PlaneIndex,
    decode::{DecodeError, Result},
};
use nc_dsp::Dsp;

pub(super) fn apply(
    picture: &mut Picture,
    filters: &[CtuFilters],
    ctu: usize,
    header: &SliceHeader,
    banks: &[Option<Box<Adaptation>>; 8],
    dsp: &Dsp,
) -> Result<()> {
    let Some(id) = header.alf.chroma else {
        return cross(picture, filters, ctu, header, banks, dsp);
    };
    let Some(AdaptationData::Alf(bank)) = banks[usize::from(id)].as_ref().map(|a| &a.data) else {
        return Err(DecodeError::Malformed("VVC missing chroma ALF bank"));
    };
    let columns = picture.width().div_ceil(ctu);
    let side = ctu / 2;
    for plane in 1..3 {
        let source = picture.plane(PlaneIndex::ALL[plane]).clone();
        let input = source.view();
        let mut output = picture.plane_mut(PlaneIndex::ALL[plane]).view_mut();
        let mut padded: [Vec<u8>; 7] = std::array::from_fn(|_| vec![0; side + 6]);
        for (index, filter) in filters.iter().enumerate() {
            let Some(alternative) = filter.alf[plane] else {
                continue;
            };
            let selected = &bank.chroma[usize::from(alternative)];
            let limits = selected.clip.map(|c| [256, 32, 8, 2][usize::from(c)]);
            let (x0, y0) = (index % columns * side, index / columns * side);
            let end = (x0 + side).min(input.width());
            for y in y0..(y0 + side).min(input.height()) {
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
                    let sy = (y as isize + dy).clamp(0, input.height() as isize - 1) as usize;
                    for (x, sample) in samples[..end - x0 + 6].iter_mut().enumerate() {
                        let sx = (x0 as isize + x as isize - 3).clamp(0, input.width() as isize - 1)
                            as usize;
                        *sample = input.row(sy)[sx];
                    }
                }
                dsp.vvc_alf_filter_chroma_row(
                    &padded.each_ref().map(Vec::as_slice),
                    selected.coefficients,
                    limits,
                    if distance == 0 { 10 } else { 7 },
                    &mut output.row_mut(y)[x0..end],
                );
            }
        }
    }
    cross(picture, filters, ctu, header, banks, dsp)
}

fn cross(
    picture: &mut Picture,
    filters: &[CtuFilters],
    ctu: usize,
    header: &SliceHeader,
    banks: &[Option<Box<Adaptation>>; 8],
    dsp: &Dsp,
) -> Result<()> {
    if !filters.iter().any(|f| f.cross != [0; 2]) {
        return Ok(());
    }
    // Called before luma ALF, retaining the pre-ALF luma required by §8.8.5.
    let luma = picture.plane(PlaneIndex::Y).clone();
    let input = luma.view();
    let columns = picture.width().div_ceil(ctu);
    let side = ctu / 2;
    let mut padded: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0; ctu + 2]);
    for plane in 0..2 {
        let Some(id) = header.alf.cross[plane] else {
            continue;
        };
        let Some(AdaptationData::Alf(bank)) = banks[usize::from(id)].as_ref().map(|a| &a.data)
        else {
            return Err(DecodeError::Malformed(
                "VVC missing cross-component ALF bank",
            ));
        };
        let mut output = picture.plane_mut(PlaneIndex::ALL[plane + 1]).view_mut();
        for (index, filter) in filters.iter().enumerate() {
            let selection = filter.cross[plane];
            if selection == 0 {
                continue;
            }
            let coefficients = bank.cross[plane][usize::from(selection - 1)];
            let (x0, y0) = (index % columns * side, index / columns * side);
            let end = (x0 + side).min(output.width());
            for y in y0..(y0 + side).min(output.height()) {
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
                    let sy = (2 * y as isize + dy).clamp(0, input.height() as isize - 1) as usize;
                    for (x, sample) in samples[..2 * (end - x0) + 2].iter_mut().enumerate() {
                        let sx = (2 * x0 as isize + x as isize - 1)
                            .clamp(0, input.width() as isize - 1)
                            as usize;
                        *sample = input.row(sy)[sx];
                    }
                }
                dsp.vvc_alf_cross_row(
                    &padded.each_ref().map(Vec::as_slice),
                    coefficients,
                    &mut output.row_mut(y)[x0..end],
                );
            }
        }
    }
    Ok(())
}
