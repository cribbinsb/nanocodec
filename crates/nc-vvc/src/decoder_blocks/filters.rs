//! General picture filter application through the codec's existing SAO/ALF engines.
use super::{Sequence, boundaries::Closures, filter_syntax::CtuFilters};
use crate::{
    alf::{Analysis, ClippedFilters, CtbFilter},
    alf_aps::SignalledFilters,
    decoder_headers::{Adaptation, AdaptationData},
};
use nc_common::{
    BitDepth, Picture, PlaneIndex,
    decode::{DecodeError, Result},
};
use nc_dsp::{Dsp, Pixel, vvc_alf_clip_value};

#[derive(Debug, Default)]
pub(crate) struct FilterScratch<T> {
    alf_samples: Vec<T>,
    alf_classes: Vec<[u8; 2]>,
    chroma_buf: Vec<T>,
}

/// SAO, then ALF, of a decoded picture: each CTB as its syntax says, not
/// reading across what `closures` closes.
#[cfg(test)]
pub(crate) fn apply<T: Pixel>(
    picture: &mut Picture<T>,
    filters: &[CtuFilters],
    (sequence, closures): (&Sequence, &Closures),
    banks: &[Option<Box<Adaptation>>; 8],
    (dsp, depth): (&Dsp, BitDepth),
) -> Result<()> {
    let mut scratch = FilterScratch::default();
    apply_with_scratch(
        picture,
        filters,
        (sequence, closures),
        banks,
        (dsp, depth),
        &mut scratch,
    )
}

/// [`apply`] reusing `scratch` buffers across decoded pictures.
pub(crate) fn apply_with_scratch<T: Pixel>(
    picture: &mut Picture<T>,
    filters: &[CtuFilters],
    (sequence, closures): (&Sequence, &Closures),
    banks: &[Option<Box<Adaptation>>; 8],
    (dsp, depth): (&Dsp, BitDepth),
    scratch: &mut FilterScratch<T>,
) -> Result<()> {
    let kernels = dsp.at(depth.of::<T>());
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
        crate::sao::apply_limited(kernels, picture, &edge, ctu, &limits);
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
                let offsets = filter.edge.offsets[plane].map(i16::from);
                kernels.sao_band_apply_masked(*band, offsets, &mut view, area, &[]);
            }
        }
    }
    if filters
        .iter()
        .any(|f| f.alf[1..].iter().any(Option::is_some) || f.cross != [0; 2])
    {
        super::chroma_alf::apply(
            picture,
            filters,
            (ctu, closures),
            banks,
            kernels,
            &mut scratch.chroma_buf,
        )?;
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
        .map(|(f, region)| (f, Analysis::region((dsp, depth), picture, ctu, &region)))
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
    let depth_t = depth.of::<T>();
    if whole.iter().any(|f| f.alf[0].is_some()) {
        let samples = std::mem::take(&mut scratch.alf_samples);
        let classes = std::mem::take(&mut scratch.alf_classes);
        let analysis = Analysis::new_with_scratch(
            (dsp, depth),
            picture,
            ctu,
            |i| whole[i].alf[0].is_some(),
            samples,
            classes,
        );
        filter_luma(&analysis, picture, &whole, (banks, depth_t))?;
        let (samples, classes) = analysis.into_scratch();
        scratch.alf_samples = samples;
        scratch.alf_classes = classes;
    }
    for (filter, analysis) in &regions {
        filter_luma(
            analysis,
            picture,
            std::slice::from_ref(filter),
            (banks, depth_t),
        )?;
    }
    Ok(())
}

/// Luma ALF with `analysis` of the CTBs `filters` names (the picture's in
/// raster order, or one region's).
fn filter_luma<T: Pixel>(
    analysis: &Analysis<'_, T>,
    picture: &mut Picture<T>,
    filters: &[CtuFilters],
    (banks, depth): (&[Option<Box<Adaptation>>; 8], BitDepth),
) -> Result<()> {
    if filters.iter().any(|f| matches!(f.alf[0], Some(0..16))) {
        let fixed: Vec<_> = filters
            .iter()
            .map(|f| f.alf[0].filter(|&i| i < 16).map(CtbFilter::Fixed))
            .collect();
        analysis.apply(picture, &fixed, None);
    }
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
                    .map(|f| f.clip.map(|c| vvc_alf_clip_value(usize::from(c), depth)))
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
