//! General picture filter application through the codec's existing SAO/ALF engines.
use super::{Sequence, boundaries::Closures, filter_syntax::CtuFilters};
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

/// SAO, then ALF, of a decoded picture: each CTB as its syntax says, not
/// reading across what `closures` closes.
pub(crate) fn apply(
    picture: &mut Picture,
    filters: &[CtuFilters],
    (sequence, closures): (&Sequence, &Closures),
    banks: &[Option<Box<Adaptation>>; 8],
    dsp: &Dsp,
) -> Result<()> {
    let ctu = 1 << sequence.ctu_log2;
    if filters
        .iter()
        .any(|f| f.edge.classes.iter().any(Option::is_some))
    {
        let edge: Vec<_> = filters.iter().map(|f| f.edge).collect();
        let limits = crate::sao::Limits {
            neighbours: (0..filters.len())
                .map(|ctb| Some(closures.neighbours(ctb)).filter(|cells| *cells != [[true; 3]; 3]))
                .collect(),
            cuts: closures.cuts(0),
        };
        crate::sao::apply_limited(dsp, picture, &edge, ctu, &limits);
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
        super::chroma_alf::apply(picture, filters, (ctu, closures), banks, dsp)?;
    }
    if !filters.iter().any(|f| f.alf[0].is_some()) {
        return Ok(());
    }
    // CTBs at closed edges or virtual boundaries are analysed region by
    // region; the others as part of the picture. Every analysis copies the
    // pre-ALF samples before any is filtered.
    let regions: Vec<_> = filters
        .iter()
        .enumerate()
        .filter(|(_, f)| f.alf[0].is_some())
        .filter_map(|(ctb, f)| Some((*f, closures.alf_regions(ctb)?)))
        .flat_map(|(f, regions)| regions.into_iter().map(move |region| (f, region)))
        .map(|(f, region)| (f, Analysis::region(dsp, picture, ctu, &region)))
        .collect();
    let whole: Vec<CtuFilters> = filters
        .iter()
        .enumerate()
        .map(|(ctb, f)| {
            let mut f = *f;
            if f.alf[0].is_some() && closures.alf_regions(ctb).is_some() {
                f.alf[0] = None;
                f.banks.luma = None;
            }
            f
        })
        .collect();
    let analysis = Analysis::new(dsp, picture, ctu, |i| whole[i].alf[0].is_some());
    filter_luma(&analysis, picture, &whole, banks)?;
    for (filter, analysis) in &regions {
        filter_luma(analysis, picture, std::slice::from_ref(filter), banks)?;
    }
    Ok(())
}

/// Luma ALF with `analysis` of the CTBs `filters` names (the picture's in
/// raster order, or one region's).
fn filter_luma(
    analysis: &Analysis<'_>,
    picture: &mut Picture,
    filters: &[CtuFilters],
    banks: &[Option<Box<Adaptation>>; 8],
) -> Result<()> {
    let fixed: Vec<_> = filters
        .iter()
        .map(|f| f.alf[0].filter(|&i| i < 16).map(CtbFilter::Fixed))
        .collect();
    analysis.apply(picture, &fixed, None);
    // Every bank sees the pre-ALF samples retained by the analysis.
    let mut ids: Vec<u8> = filters.iter().filter_map(|f| f.banks.luma).collect();
    ids.sort_unstable();
    ids.dedup();
    for id in ids {
        let alf = alf_bank(banks, id)?;
        let decisions: Vec<_> = filters
            .iter()
            .map(|f| (f.banks.luma == Some(id)).then_some(CtbFilter::Signalled))
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

/// The ALF parameters of APS `id`.
pub(super) fn alf_bank(
    banks: &[Option<Box<Adaptation>>; 8],
    id: u8,
) -> Result<&crate::decoder_headers::AlfData> {
    match banks[usize::from(id)].as_ref().map(|a| &a.data) {
        Some(AdaptationData::Alf(alf)) => Ok(alf),
        _ => Err(DecodeError::Malformed("VVC missing ALF APS")),
    }
}
