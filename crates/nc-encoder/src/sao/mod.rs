//! The encoder's sample adaptive offset decision, shared by HEVC (§7.3.8.3,
//! §8.7.3) and VVC (H.266 §7.3.11.3, §8.8.4), whose SAO tools are the same:
//! per CTB, each of luma and chroma is off, an edge offset of one of four
//! classes, or a band offset (four bands from a position, each chroma plane
//! its own); Cb and Cr share the type and edge class; a CTB may instead copy
//! its left or upper neighbour's parameters (merge).
//!
//! The structure is HM's `TEncSampleAdaptiveOffset` (BSD-3-Clause, ITU/ISO/
//! IEC; `THIRD_PARTY.md`), reimplemented:
//! - statistics per CTB, plane and type ([`stats`]): per edge category or
//!   band, the sum of `source − deblocked` and the count;
//! - offsets per type from them ([`offsets`]): the rounded mean, then the
//!   magnitude of least `distortion + λ·bits` towards zero, and for band
//!   offset the four consecutive bands of least cost;
//! - per CTB, luma then chroma: off or the cheapest type, then the CTB's own
//!   parameters against merging left or up, each priced in bits by the
//!   codec's context states ([`rate`]) and in distortion normalised by each
//!   channel's λ (`Σ D_c / λ_c + bits`, HM's `modeNormCost`).
//!
//! The codec supplies the planes, λs, CTB size and its two context models;
//! it applies and writes the parameters with its own syntax code.
//!
//! **Cheap mode** ([`Candidates::Reuse`]): fast presets search the types only
//! on some pictures and reuse each CTB's types in between, gathering only
//! their statistics and re-deriving offsets (SVT-AV1's CDEF strength reuse
//! idea, `docs/reference/svt-av1-m12-rtc.md` §4). Merge candidates of other
//! types gather their statistics on demand, so a merge is never priced
//! against missing statistics (the HEVC bug fixed in `336527d`).

mod offsets;
pub mod rate;
mod stats;

use nc_common::{Picture, PlaneIndex};
use nc_dsp::Dsp;

pub use rate::SaoContexts;
use stats::CtbStats;
pub use stats::class_rect;

/// Edge-offset classes (`sao_eo_class`: horizontal, vertical, 135°, 45°).
pub const EDGE_CLASSES: u8 = 4;
/// Largest `sao_offset_abs` at 8 bits: `(1 << (Min(bitDepth, 10) − 5)) − 1`
/// (HEVC §7.4.9.3.2, H.266 §7.4.10.3).
pub const MAX_OFFSET: i32 = 7;
/// Number of bands of band offset (`bandTable`, HEVC §8.7.3.2).
pub const BANDS: u8 = nc_dsp::SAO_BANDS as u8;

/// A component's SAO type: one edge-offset class, or band offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaoType {
    /// Edge offset of this `sao_eo_class`.
    Edge(u8),
    /// Band offset.
    Band,
}

/// How a component's samples are classified (`sao_type_idx`) and, for band
/// offset, each plane's `sao_band_position` (Y, or Cb and Cr).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaoKind {
    /// Edge offset with this `sao_eo_class` (shared by Cb and Cr).
    Edge(u8),
    /// Band offset with each plane's band position.
    Band([u8; 2]),
}

/// SAO parameters of luma, or of chroma (Cb and Cr share the type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentSao {
    /// Edge or band offset.
    pub kind: SaoKind,
    /// Per plane (Y, or Cb and Cr), the offsets of edge categories 1..4 or
    /// of the four bands from the band position, signed as applied. Edge
    /// categories 1 and 2 are ≥ 0, 3 and 4 are ≤ 0 (HEVC §7.4.9.3.2).
    pub offsets: [[i8; 4]; 2],
}

impl ComponentSao {
    /// The type of these parameters.
    #[must_use]
    pub fn sao_type(self) -> SaoType {
        match self.kind {
            SaoKind::Edge(class) => SaoType::Edge(class),
            SaoKind::Band(_) => SaoType::Band,
        }
    }

    /// The edge-offset class, `None` for a band offset.
    #[must_use]
    pub fn edge_class(self) -> Option<u8> {
        match self.kind {
            SaoKind::Edge(class) => Some(class),
            SaoKind::Band(_) => None,
        }
    }
}

/// SAO parameters of one CTB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CtbSao {
    /// Luma parameters, `None` for off.
    pub luma: Option<ComponentSao>,
    /// Chroma parameters (Cb and Cr), `None` for off.
    pub chroma: Option<ComponentSao>,
}

impl CtbSao {
    /// The parameters of channel 0 (luma) or 1 (chroma).
    #[must_use]
    pub fn channel(&self, channel: usize) -> Option<ComponentSao> {
        if channel == 0 { self.luma } else { self.chroma }
    }

    fn channel_mut(&mut self, channel: usize) -> &mut Option<ComponentSao> {
        if channel == 0 {
            &mut self.luma
        } else {
            &mut self.chroma
        }
    }
}

/// How a CTB's parameters are signalled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// `sao_merge_left_flag`: copy the left CTB.
    MergeLeft,
    /// `sao_merge_up_flag`: copy the CTB above.
    MergeUp,
    /// Coded explicitly.
    Explicit,
}

/// The SAO decision of a picture.
#[derive(Debug, Clone, Default)]
pub struct PictureSao {
    /// Per CTB in raster order: parameters and how they are signalled.
    pub ctbs: Vec<(CtbSao, Signal)>,
    /// CTB columns.
    pub cols: usize,
}

/// The types of a CTB's luma and chroma (`None`: off), kept for reuse.
pub type CtbTypes = [Option<SaoType>; 2];

/// The types of each CTB of a decision, for [`Candidates::Reuse`].
#[must_use]
pub fn types(sao: &PictureSao) -> Vec<CtbTypes> {
    sao.ctbs
        .iter()
        .map(|(p, _)| [0, 1].map(|channel| p.channel(channel).map(ComponentSao::sao_type)))
        .collect()
}

/// The types a picture's CTBs evaluate.
#[derive(Debug, Clone, Copy)]
pub enum Candidates<'a> {
    /// Every edge class, and band offset when [`Settings::band`] is set.
    Search,
    /// Per CTB, only the types it chose on the last searched picture (its
    /// offsets re-derived); a channel that was off stays off.
    Reuse(&'a [CtbTypes]),
}

/// How a picture's SAO is decided.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// `slice_sao_luma_flag` and `slice_sao_chroma_flag`.
    pub channels: [bool; 2],
    /// λ of luma and of chroma decisions (squared error per bit).
    pub lambdas: [f64; 2],
    /// Luma CTB side in samples (4:2:0: chroma CTBs are half).
    pub ctb: usize,
    /// Statistics read every `row_step`-th row (1..=4), weighted back to
    /// the full count; the filter still runs on every sample.
    pub row_step: usize,
    /// Evaluate band offset on searched pictures.
    pub band: bool,
}

/// The planes of channel 0 (luma) and 1 (chroma).
fn channel_planes(channel: usize) -> &'static [PlaneIndex] {
    if channel == 0 {
        &[PlaneIndex::Y]
    } else {
        &[PlaneIndex::U, PlaneIndex::V]
    }
}

/// Decide the SAO parameters of a picture from its deblocked samples.
///
/// `contexts` are the codec's SAO context models at the start of the slice;
/// they follow the decided syntax CTB by CTB, so every CTB is priced with
/// the states the coder will have. CTBs marked in `skipped` (all CUs
/// skipped: they copy already filtered samples) are off without statistics,
/// merged from a neighbour that is off too.
pub fn decide<C: SaoContexts>(
    dsp: &Dsp,
    (source, deblocked): (&Picture, &Picture),
    settings: &Settings,
    candidates: Candidates<'_>,
    skipped: Option<&[bool]>,
    mut contexts: C,
) -> PictureSao {
    let ctb = settings.ctb;
    let cols = deblocked.width().div_ceil(ctb);
    let rows = deblocked.height().div_ceil(ctb);
    let mut ctbs: Vec<(CtbSao, Signal)> = Vec::with_capacity(cols * rows);
    for ry in 0..rows {
        for rx in 0..cols {
            let index = ry * cols + rx;
            let neighbours = (rx > 0, ry > 0);
            let choice = if skipped.is_some_and(|s| s[index]) {
                unfiltered(&ctbs, index, cols)
            } else {
                let mut stats = CtbStats::new(dsp, (source, deblocked), (rx, ry), settings);
                let types = match candidates {
                    Candidates::Search => None,
                    Candidates::Reuse(types) => Some(types[index]),
                };
                let neighbour = |offset: usize| ctbs[index - offset].0;
                let merges = [
                    neighbours.0.then(|| (neighbour(1), Signal::MergeLeft)),
                    neighbours.1.then(|| (neighbour(cols), Signal::MergeUp)),
                ];
                choose(&mut stats, settings, types, merges, &contexts, neighbours)
            };
            rate::commit(
                &mut contexts,
                &choice.0,
                choice.1,
                neighbours,
                settings.channels,
            );
            ctbs.push(choice);
        }
    }
    PictureSao { ctbs, cols }
}

/// SAO off for the CTB at `index` (`ctbs` holds the CTBs before it):
/// merged from the left or upper neighbour when that is off too, else
/// signalled.
fn unfiltered(ctbs: &[(CtbSao, Signal)], index: usize, cols: usize) -> (CtbSao, Signal) {
    let off = CtbSao::default();
    let is_off = |i: usize| ctbs[i].0 == off;
    let signal = if !index.is_multiple_of(cols) && is_off(index - 1) {
        Signal::MergeLeft
    } else if index >= cols && is_off(index - cols) {
        Signal::MergeUp
    } else {
        Signal::Explicit
    };
    (off, signal)
}

/// One CTB's parameters of least normalised cost (HM `decideBlkParams`):
/// its own (each channel off or its best type, [`best_channel`]) or a
/// merge candidate's. Ties keep the own parameters.
fn choose<C: SaoContexts>(
    stats: &mut CtbStats<'_>,
    settings: &Settings,
    types: Option<CtbTypes>,
    merges: [Option<(CtbSao, Signal)>; 2],
    contexts: &C,
    neighbours: (bool, bool),
) -> (CtbSao, Signal) {
    let mut ctx = contexts.clone();
    let mut cost = rate::code_merges(&mut ctx, Signal::Explicit, neighbours);
    let mut own = CtbSao::default();
    for channel in 0..2 {
        if !settings.channels[channel] {
            continue;
        }
        let allowed: Vec<SaoType> = match types {
            None => search_types(settings.band),
            Some(types) => types[channel].into_iter().collect(),
        };
        let (params, channel_cost) = best_channel(
            stats,
            channel,
            &allowed,
            settings.lambdas[channel],
            &mut ctx,
        );
        *own.channel_mut(channel) = params;
        cost += channel_cost;
    }
    let mut best = (own, Signal::Explicit, cost);
    for (candidate, signal) in merges.into_iter().flatten() {
        let mut ctx = contexts.clone();
        let bits = rate::code_merges(&mut ctx, signal, neighbours);
        let cost = bits + normalised_distortion(stats, &candidate, settings);
        if cost < best.2 {
            best = (candidate, signal, cost);
        }
    }
    (best.0, best.1)
}

/// Every edge class, then band offset when enabled.
fn search_types(band: bool) -> Vec<SaoType> {
    (0..EDGE_CLASSES)
        .map(SaoType::Edge)
        .chain(band.then_some(SaoType::Band))
        .collect()
}

/// A channel's best own parameters among `allowed` types, or off, and
/// their normalised cost (`D / λ` plus bits of its type and offsets at
/// `ctx`, which then codes the chosen type's first bin; HM
/// `deriveModeNewRDO`).
fn best_channel<C: SaoContexts>(
    stats: &mut CtbStats<'_>,
    channel: usize,
    allowed: &[SaoType],
    lambda: f64,
    ctx: &mut C,
) -> (Option<ComponentSao>, f64) {
    let mut best = (None, f64::from(ctx.type_bits(false)));
    let on_bits = f64::from(ctx.type_bits(true));
    for &sao_type in allowed {
        let mut params = ComponentSao {
            kind: match sao_type {
                SaoType::Edge(class) => SaoKind::Edge(class),
                SaoType::Band => SaoKind::Band([0; 2]),
            },
            offsets: [[0; 4]; 2],
        };
        let mut distortion = 0;
        for (i, &plane) in channel_planes(channel).iter().enumerate() {
            let derived = match sao_type {
                SaoType::Edge(class) => offsets::edge(&stats.edge(plane, class), lambda),
                SaoType::Band => {
                    let band = offsets::band(&stats.band(plane), lambda);
                    if let SaoKind::Band(positions) = &mut params.kind {
                        positions[i] = band.position;
                    }
                    band.offsets
                }
            };
            params.offsets[i] = derived.offsets;
            distortion += derived.distortion;
        }
        let bits = on_bits + rate::offset_bits(&params, channel_planes(channel).len());
        let cost = distortion as f64 / lambda + bits;
        if cost < best.1 {
            best = (Some(params), cost);
        }
    }
    ctx.code_type(best.0.is_some());
    best
}

/// `Σ D_c / λ_c` of applying `params` to the CTB of `stats`.
fn normalised_distortion(stats: &mut CtbStats<'_>, params: &CtbSao, settings: &Settings) -> f64 {
    let mut total = 0.0;
    for channel in 0..2 {
        let Some(component) = params.channel(channel) else {
            continue;
        };
        if !settings.channels[channel] {
            continue;
        }
        let mut distortion = 0;
        for (i, &plane) in channel_planes(channel).iter().enumerate() {
            let offsets = component.offsets[i];
            distortion += match component.kind {
                SaoKind::Edge(class) => {
                    offsets::edge_distortion(&stats.edge(plane, class), offsets)
                }
                SaoKind::Band(positions) => {
                    offsets::band_distortion(&stats.band(plane), positions[i], offsets)
                }
            };
        }
        total += distortion as f64 / settings.lambdas[channel];
    }
    total
}

#[cfg(test)]
mod tests;
