//! H.266 §8.8.4 sample adaptive offset: the encoder's decision (the
//! engine's shared `nc_encoder::sao`, priced with VVC's two SAO contexts),
//! applying it, and its syntax (§7.3.11.3), which follows HEVC's. Syntax
//! follows `VVenC` (BSD-3-Clause-Clear; `THIRD_PARTY.md`).
use crate::{coeff::Bins, contexts::Context, contexts::Contexts};
use nc_common::{BitDepth, Picture, PlaneIndex, PlaneView, Sample, decode::Result};
use nc_dsp::{DepthKernels, Dsp, Pixel};
use nc_encoder::sao::{ComponentSao, MAX_OFFSET, SaoContexts, SaoKind};
pub(crate) use nc_encoder::sao::{CtbSao, CtbTypes, Signal};

/// Edge-offset parameters as applied: each channel's class (`None`: no
/// edge offset) and each plane's offsets. The general decoder's form, and
/// the edge part of a [`CtbSao`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CtuSao {
    pub classes: [Option<u8>; 2],
    pub offsets: [[i8; 4]; 3],
}

impl CtuSao {
    /// The edge-offset part of `params`.
    fn edges_of(params: &CtbSao) -> Self {
        let mut edge = Self::default();
        for (channel, component) in [params.luma, params.chroma].into_iter().enumerate() {
            let Some(component) = component else { continue };
            if let SaoKind::Edge(class) = component.kind {
                edge.classes[channel] = Some(class);
                for (i, &plane) in channel_planes(channel).iter().enumerate() {
                    edge.offsets[plane as usize] = component.offsets[i];
                }
            }
        }
        edge
    }
}

/// The planes of channel 0 (luma) and channel 1 (chroma).
fn channel_planes(channel: usize) -> &'static [PlaneIndex] {
    if channel == 0 {
        &[PlaneIndex::Y]
    } else {
        &[PlaneIndex::U, PlaneIndex::V]
    }
}

/// The samples of the CTB at luma `(x, y)` (`ctu` luma samples square,
/// clipped to the `width` x `height` plane, subsampled by `shift`) whose
/// `class` neighbours lie in the plane.
fn bounds(
    (width, height): (usize, usize),
    shift: usize,
    (x, y): (usize, usize),
    ctu: usize,
    class: u8,
) -> (usize, usize, usize, usize) {
    let horizontal = usize::from(class != 1);
    let vertical = usize::from(class != 0);
    (
        (x >> shift).max(horizontal),
        (y >> shift).max(vertical),
        ((x + ctu) >> shift).min(width - horizontal),
        ((y + ctu) >> shift).min(height - vertical),
    )
}
/// The width and height of `plane` of `picture`.
fn plane_size<T: Sample>(picture: &Picture<T>, plane: PlaneIndex) -> (usize, usize) {
    let plane = picture.plane(plane);
    (plane.width(), plane.height())
}

/// VVC's SAO context models (§9.3.2.2) for the shared decision's rate
/// estimates.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SaoModels {
    merge: Context,
    sao_type: Context,
}

impl SaoModels {
    /// The models of a slice's freshly initialised contexts.
    pub(crate) fn of(contexts: &Contexts) -> Self {
        Self {
            merge: contexts.sao_merge,
            sao_type: contexts.sao_type,
        }
    }
}

impl SaoContexts for SaoModels {
    fn merge_bits(&self, bin: bool) -> f32 {
        self.merge.bits(bin)
    }
    fn type_bits(&self, bin: bool) -> f32 {
        self.sao_type.bits(bin)
    }
    fn code_merge(&mut self, bin: bool) {
        self.merge.update(bin);
    }
    fn code_type(&mut self, bin: bool) {
        self.sao_type.update(bin);
    }
}

/// Apply `params`, one per `ctu`-sample CTU in raster order: edge offsets
/// (classified from the deblocked samples), then band offsets, which read
/// only the sample itself and run in place.
pub(crate) fn apply(dsp: &Dsp, picture: &mut Picture, params: &[CtbSao], ctu: usize) {
    let kernels = dsp.at(BitDepth::EIGHT);
    let edges: Vec<CtuSao> = params.iter().map(CtuSao::edges_of).collect();
    if edges.iter().any(|e| e.classes.iter().any(Option::is_some)) {
        apply_limited(kernels, picture, &edges, ctu, &Limits::default());
    }
    for (index, ctb) in params.iter().enumerate() {
        for (channel, component) in [ctb.luma, ctb.chroma].into_iter().enumerate() {
            let Some(ComponentSao {
                kind: SaoKind::Band(positions),
                offsets,
            }) = component
            else {
                continue;
            };
            for (i, &plane) in channel_planes(channel).iter().enumerate() {
                let offsets = offsets[i].map(i16::from);
                apply_band(kernels, picture, (plane, index, ctu), positions[i], offsets);
            }
        }
    }
}

/// Band offset (§8.8.4.3) of `plane` in the CTB at raster `index` of
/// `ctu`-sample CTUs: the four bands from `position` move by `offsets`.
pub(crate) fn apply_band<T: Pixel>(
    kernels: DepthKernels<'_>,
    picture: &mut Picture<T>,
    (plane, index, ctu): (PlaneIndex, usize, usize),
    position: u8,
    offsets: [i16; 4],
) {
    let columns = picture.width().div_ceil(ctu);
    let shift = usize::from(plane != PlaneIndex::Y);
    let mut view = picture.plane_mut(plane).view_mut();
    let x = (index % columns * ctu) >> shift;
    let y = (index / columns * ctu) >> shift;
    let area = (
        x,
        y,
        (x + (ctu >> shift)).min(view.width()),
        (y + (ctu >> shift)).min(view.height()),
    );
    kernels.sao_band_apply_masked(position, offsets, &mut view, area, &[]);
}

/// What a decoded picture's edge offsets may not compare across (H.266
/// §8.8.4.2): CTBs at closed slice, tile or subpicture edges, and virtual
/// boundaries.
#[derive(Default)]
pub(crate) struct Limits {
    /// Per CTB (raster order), which of the 3x3 CTBs around it may be
    /// read; empty or `None` when all may.
    pub neighbours: Vec<Option<[[bool; 3]; 3]>>,
    /// The virtual boundaries' luma columns, then rows.
    pub cuts: [Vec<usize>; 2],
}

/// [`apply`] within `limits`, to samples of any depth (the general
/// decoder's): up to 10 bits `SaoOffsetVal` is the coded offset (H.266
/// §7.4.9.3.2: its shift `BitDepth − Min(BitDepth, 10)` is zero).
pub(crate) fn apply_limited<T: Pixel>(
    dsp: DepthKernels<'_>,
    picture: &mut Picture<T>,
    params: &[CtuSao],
    ctu: usize,
    limits: &Limits,
) {
    for plane in PlaneIndex::ALL {
        apply_plane(dsp, picture, plane, params, ctu, limits);
    }
}

/// [`apply`] to one plane, CTU row by CTU row. Edge offsets classify from
/// the deblocked samples, so each row's are copied (with the row below)
/// into a band buffer before it is filtered, and the row above comes from
/// the band before: no copy of the whole picture, and none at all for rows
/// whose CTUs leave the plane alone.
fn apply_plane<T: Pixel>(
    dsp: DepthKernels<'_>,
    picture: &mut Picture<T>,
    plane: PlaneIndex,
    params: &[CtuSao],
    ctu: usize,
    limits: &Limits,
) {
    let channel = usize::from(plane != PlaneIndex::Y);
    let cuts = limits
        .cuts
        .each_ref()
        .map(|positions| positions.iter().map(|&p| p >> channel).collect::<Vec<_>>());
    let (width, height) = plane_size(picture, plane);
    let band_rows = ctu >> channel;
    let columns = picture.width().div_ceil(ctu);
    // The band's rows with the one above and the one below it.
    let mut band = vec![T::default(); width * (band_rows + 2)];
    // The deblocked last row of the band before, if that band was filtered.
    let mut above = vec![T::default(); width];
    let mut above_filtered = false;
    for (row, ctus) in params.chunks(columns).enumerate() {
        if ctus.iter().all(|sao| sao.classes[channel].is_none()) {
            above_filtered = false;
            continue;
        }
        let top = row * band_rows;
        let bottom = (top + band_rows).min(height);
        let (first, last) = (top.saturating_sub(1), (bottom + 1).min(height));
        let rows = last - first;
        let source = picture.plane(plane).view();
        for (y, line) in (first..last).zip(band.chunks_exact_mut(width)) {
            let deblocked = if y + 1 == top && above_filtered {
                &above[..]
            } else {
                source.row(y)
            };
            line.copy_from_slice(deblocked);
        }
        above.copy_from_slice(source.row(bottom - 1));
        above_filtered = true;
        let input = PlaneView::new(&band, width, width, rows).expect("the band holds its rows");
        let mut output = picture
            .plane_mut(plane)
            .region_mut(0, first as isize, width, rows)
            .expect("the band lies inside the plane");
        // The virtual boundary rows in the band's coordinates.
        let band_cuts: Vec<usize> = cuts[1]
            .iter()
            .filter(|&&p| p > first)
            .map(|&p| p - first)
            .collect();
        for (column, sao) in ctus.iter().enumerate() {
            let Some(class) = sao.classes[channel] else {
                continue;
            };
            let side = ctu >> channel;
            let (x, y) = (column * side, row * side);
            let near = |positions: &[usize], start: usize, end: usize| {
                positions.iter().any(|&p| start <= p && p <= end)
            };
            let restricted = limits
                .neighbours
                .get(row * columns + column)
                .copied()
                .flatten()
                .or_else(|| {
                    (near(&cuts[0], x, x + side) || near(&cuts[1], y, y + side))
                        .then_some([[true; 3]; 3])
                });
            if let Some(ctbs) = restricted {
                let area = nc_dsp::SaoCtb {
                    x0: x,
                    y0: y - first,
                    x1: (x + side).min(width),
                    y1: (y + side).min(height) - first,
                    size: side,
                    width,
                    height: rows,
                };
                let limits = nc_dsp::SaoLimits {
                    ctbs,
                    cuts: [&cuts[0], &band_cuts],
                    exempt: &[],
                };
                let offsets = sao.offsets[plane as usize].map(i16::from);
                let parameters = (class, offsets, dsp.depth());
                nc_dsp::sao_edge_restricted(&input, &mut output, area, parameters, &limits);
                continue;
            }
            let (x0, y0, x1, y1) = bounds(
                (width, height),
                channel,
                (column * ctu, row * ctu),
                ctu,
                class,
            );
            dsp.sao_edge_apply(
                class,
                sao.offsets[plane as usize].map(i16::from),
                &input,
                &mut output,
                (x0, y0 - first, x1, y1 - first),
            );
        }
    }
}
/// One CTU's SAO syntax (§7.3.11.3), both channels on in the slice.
/// Writing codes `source` as `signal` says (a merge copies `left` or `up`,
/// which must then equal it); reading ignores both and returns what the
/// stream says (merged parameters copied from `left` or `up`).
pub(crate) fn code(
    io: &mut impl Bins,
    c: &mut Contexts,
    (source, signal): (CtbSao, Signal),
    left: Option<CtbSao>,
    up: Option<CtbSao>,
) -> Result<CtbSao> {
    for (neighbour, merge) in [(left, Signal::MergeLeft), (up, Signal::MergeUp)] {
        let Some(neighbour) = neighbour else { continue };
        if io.bin(&mut c.sao_merge, signal == merge)? {
            return Ok(neighbour);
        }
    }
    let mut decoded = CtbSao::default();
    for channel in 0..2 {
        let wanted = source.channel(channel);
        if !io.bin(&mut c.sao_type, wanted.is_some())? {
            continue;
        }
        let edge = wanted.is_none_or(|w| matches!(w.kind, SaoKind::Edge(_)));
        // sao_type_idx second bin: 1 edge, 0 band.
        let edge = io.bypass(u32::from(edge), 1)? != 0;
        let mut component = ComponentSao {
            kind: if edge {
                SaoKind::Edge(0)
            } else {
                SaoKind::Band([0; 2])
            },
            offsets: [[0; 4]; 2],
        };
        for i in 0..channel_planes(channel).len() {
            let want = wanted.map_or([0; 4], |w| w.offsets[i]);
            let mut magnitudes = [0i8; 4];
            for (k, magnitude) in magnitudes.iter_mut().enumerate() {
                // sao_offset_abs: truncated unary, cMax 7, bypass.
                let target = want[k].unsigned_abs();
                let mut abs = 0;
                while abs < MAX_OFFSET as u8 && io.bypass(u32::from(target > abs), 1)? != 0 {
                    abs += 1;
                }
                *magnitude = abs as i8;
            }
            if edge {
                // Categories 1 and 2 add, 3 and 4 subtract (§7.4.10.3).
                for (k, &m) in magnitudes.iter().enumerate() {
                    component.offsets[i][k] = if k < 2 { m } else { -m };
                }
                if i == 0 {
                    let class = wanted.and_then(ComponentSao::edge_class).unwrap_or(0);
                    component.kind = SaoKind::Edge(io.bypass(u32::from(class), 2)? as u8);
                }
            } else {
                for (k, &m) in magnitudes.iter().enumerate() {
                    let negative = m != 0 && io.bypass(u32::from(want[k] < 0), 1)? != 0;
                    component.offsets[i][k] = if negative { -m } else { m };
                }
                let position = match wanted.map(|w| w.kind) {
                    Some(SaoKind::Band(positions)) => positions[i],
                    _ => 0,
                };
                let position = io.bypass(u32::from(position), BAND_POSITION_BITS)? as u8;
                if let SaoKind::Band(positions) = &mut component.kind {
                    positions[i] = position;
                }
            }
        }
        if channel == 0 {
            decoded.luma = Some(component);
        } else {
            decoded.chroma = Some(component);
        }
    }
    Ok(decoded)
}

/// Bits of `sao_band_position` (fixed length, bypass).
const BAND_POSITION_BITS: u32 = 5;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacWriter;
    use crate::coeff::BitCost;
    use crate::decoder_cabac::Reader;

    fn samples() -> [Option<ComponentSao>; 3] {
        [
            None,
            Some(ComponentSao {
                kind: SaoKind::Band([30, 4]),
                offsets: [[3, 0, -7, 1], [-1, 2, 0, 7]],
            }),
            Some(ComponentSao {
                kind: SaoKind::Edge(2),
                offsets: [[7, 1, 0, -2], [0, 0, -1, -7]],
            }),
        ]
    }

    /// Band and edge parameters survive writing and reading (§7.3.11.3),
    /// and the shared decision prices them in the bits the writer codes.
    #[test]
    fn parameters_round_trip_and_shared_rates_match() {
        for luma in samples() {
            for chroma in samples() {
                for signal in [Signal::Explicit, Signal::MergeLeft, Signal::MergeUp] {
                    // Luma has one plane: its second entries are not coded.
                    let luma = luma.map(|mut l| {
                        l.offsets[1] = [0; 4];
                        if let SaoKind::Band(positions) = &mut l.kind {
                            positions[1] = 0;
                        }
                        l
                    });
                    let params = CtbSao { luma, chroma };
                    let mut writer = CabacWriter::new();
                    let mut contexts = Contexts::new(32, true);
                    let mut models = SaoModels::of(&contexts);
                    let mut price = BitCost::default();
                    code(
                        &mut price,
                        &mut contexts.clone(),
                        (params, signal),
                        Some(params),
                        Some(params),
                    )
                    .unwrap();
                    let shared = nc_encoder::sao::rate::commit(
                        &mut models,
                        &params,
                        signal,
                        (true, true),
                        [true, true],
                    );
                    assert!(
                        (shared - f64::from(price.0)).abs() < 1e-3,
                        "{params:?} {signal:?}"
                    );
                    code(
                        &mut writer,
                        &mut contexts,
                        (params, signal),
                        Some(params),
                        Some(params),
                    )
                    .unwrap();
                    writer.encode_terminate(true);
                    let mut bits = writer.finish();
                    bits.trailing_bits();
                    let bytes = bits.into_bytes();
                    let mut reader = Reader::new(&bytes).unwrap();
                    let mut contexts = Contexts::new(32, true);
                    let neighbour = Some(params);
                    let decoded = code(
                        &mut reader,
                        &mut contexts,
                        (CtbSao::default(), Signal::Explicit),
                        neighbour,
                        neighbour,
                    )
                    .unwrap();
                    assert_eq!(decoded, params, "{signal:?}");
                }
            }
        }
    }
}
