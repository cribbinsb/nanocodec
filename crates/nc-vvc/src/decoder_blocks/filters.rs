//! General picture filter application through the codec's existing SAO/ALF engines.
use super::{Sequence, SliceHeader, filter_syntax::CtuFilters};
use crate::{
    alf::{Analysis, ClippedFilters, CtbFilter},
    alf_aps::SignalledFilters,
    decoder_headers::{Adaptation, AdaptationData},
};
use nc_common::{
    Picture, PlaneIndex,
    decode::{DecodeError, Result},
};
use nc_dsp::Dsp;

pub(crate) fn apply(
    picture: &mut Picture,
    filters: &[CtuFilters],
    sequence: &Sequence,
    header: &SliceHeader,
    banks: &[Option<Box<Adaptation>>; 8],
    dsp: &Dsp,
) -> Result<()> {
    let ctu = 1 << sequence.ctu_log2;
    if filters
        .iter()
        .any(|f| f.edge.classes.iter().any(Option::is_some))
    {
        let edge: Vec<_> = filters.iter().map(|f| f.edge).collect();
        crate::sao::apply(dsp, picture, &edge, ctu);
    }
    let columns = picture.width().div_ceil(ctu);
    for (index, filter) in filters.iter().enumerate() {
        for (plane, band) in filter.bands.iter().enumerate() {
            if let Some(band) = band {
                let shift = usize::from(plane != 0);
                let mut view = picture.plane_mut(PlaneIndex::ALL[plane]).view_mut();
                let x = (index % columns * ctu) >> shift;
                let y = (index / columns * ctu) >> shift;
                let area = (
                    x,
                    y,
                    (x + (ctu >> shift)).min(view.width()),
                    (y + (ctu >> shift)).min(view.height()),
                );
                dsp.sao_band_apply(*band, filter.edge.offsets[plane], &mut view, area);
            }
        }
    }
    if filters
        .iter()
        .any(|f| f.alf[1..].iter().any(Option::is_some) || f.cross != [0; 2])
    {
        super::chroma_alf::apply(picture, filters, ctu, header, banks, dsp)?;
    }
    if !filters.iter().any(|f| f.alf[0].is_some()) {
        return Ok(());
    }
    let analysis = Analysis::new(dsp, picture, ctu, |i| filters[i].alf[0].is_some());
    let fixed: Vec<_> = filters
        .iter()
        .map(|f| f.alf[0].filter(|&i| i < 16).map(CtbFilter::Fixed))
        .collect();
    analysis.apply(picture, &fixed, None);
    // Every bank sees the pre-ALF samples retained by the shared analysis.
    for (selection, &id) in header.alf.luma.iter().enumerate() {
        let Some(AdaptationData::Alf(alf)) = banks[usize::from(id)].as_ref().map(|a| &a.data)
        else {
            return Err(DecodeError::Malformed("VVC missing luma ALF bank"));
        };
        let decisions: Vec<_> = filters
            .iter()
            .map(|f| (f.alf[0] == Some(16 + selection as u8)).then_some(CtbFilter::Signalled))
            .collect();
        if alf.luma.iter().any(|f| f.clip.iter().any(|&c| c != 0)) {
            let bank = ClippedFilters {
                classes: alf.classes,
                coefficients: alf.luma.iter().map(|f| f.coefficients).collect(),
                limits: alf
                    .luma
                    .iter()
                    .map(|f| f.clip.map(|c| [256, 32, 8, 2][usize::from(c)]))
                    .collect(),
            };
            analysis.apply_clipped(picture, &decisions, &bank);
        } else {
            let bank = SignalledFilters {
                classes: alf.classes,
                filters: alf
                    .luma
                    .iter()
                    .map(|f| f.coefficients.map(|c| c as i8))
                    .collect(),
            };
            analysis.apply(picture, &decisions, Some(&bank));
        }
    }
    Ok(())
}
