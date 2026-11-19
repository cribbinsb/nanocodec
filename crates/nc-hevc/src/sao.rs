//! Sample adaptive offset (HEVC §7.3.8.3, §8.7.3): the encoder's decision
//! (the engine's shared `nc_encoder::sao`, priced with HEVC's two SAO
//! contexts), applying it, and its syntax.
//!
//! Per CTB and component, SAO either classifies every deblocked sample by
//! its two neighbours along one of four directions (edge offset: local
//! minimum, concave edge, convex edge, local maximum, each with an offset)
//! or by its value into 32 bands (band offset: four consecutive bands from
//! a position, each with an offset). Cb and Cr share the type and edge
//! class; each has its own offsets and band position. A CTB may copy its
//! left or upper neighbour's parameters (merge).

use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::Dsp;
pub use nc_encoder::sao::{
    BANDS, Candidates, ComponentSao, CtbSao, CtbTypes, MAX_OFFSET, PictureSao, SaoKind, Signal,
    types,
};
use nc_encoder::sao::{SaoContexts, Settings, class_rect};

use crate::cabac::{BinSink, Context};
use crate::contexts::{Contexts, SliceKind};
use crate::headers::LOG2_CTB;

/// Offsets per component and plane (edge categories 1..4, or four bands).
const CATEGORIES: usize = 4;
/// Bits of `sao_band_position` (fixed length, bypass).
const BAND_POSITION_BITS: u32 = 5;

/// Which components SAO is applied to (`slice_sao_luma_flag`,
/// `slice_sao_chroma_flag`) and how the encoder decides it (`hevc.sao_*`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Components {
    /// Luma SAO is on.
    pub luma: bool,
    /// Chroma SAO is on.
    pub chroma: bool,
    /// Luma decisions use λ times this (higher: fewer luma offsets).
    pub luma_lambda_scale: f64,
    /// Sample one row per group for encoder statistics (1 is full sampling).
    pub stats_row_step: usize,
    /// CTBs whose CUs are all skipped get SAO off without statistics
    /// (`hevc.sao_skip_unchanged`).
    pub skip_unchanged: bool,
    /// Which types are searched, and on which pictures.
    pub search: TypeSearch,
}

/// The SAO types the encoder searches (`hevc.sao_band`,
/// `hevc.sao_search_interval`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeSearch {
    /// Band offset is evaluated beside the four edge classes.
    pub band: bool,
    /// Types are searched on key pictures and every this many pictures,
    /// reused in between (1: every picture).
    pub interval: u64,
}

/// HEVC's SAO context models (§9.3.2.2, Table 9-5..9-6) for the shared
/// decision's rate estimates.
#[derive(Debug, Clone, Copy)]
pub struct SaoModels {
    merge: Context,
    sao_type: Context,
}

impl SaoModels {
    /// The models at the start of a slice of `slice` kind at `qp`, as the
    /// slice data writer initialises them.
    #[must_use]
    pub fn for_slice(slice: SliceKind, qp: i32) -> Self {
        let contexts = Contexts::for_slice(slice, qp);
        Self {
            merge: contexts.sao_merge_flag,
            sao_type: contexts.sao_type_idx,
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

/// Decide the SAO parameters of a picture from the deblocked
/// reconstruction (`nc_encoder::sao::decide`). `lambdas` are luma's (before
/// the luma scale) and chroma's; `models` are the slice's initial SAO
/// contexts. CTBs marked in `skipped` (all CUs skipped) get SAO off.
#[must_use]
pub fn decide(
    (source, deblocked): (&Picture, &Picture),
    lambdas: [f64; 2],
    components: Components,
    candidates: Candidates<'_>,
    skipped: Option<&[bool]>,
    models: SaoModels,
) -> PictureSao {
    let settings = Settings {
        channels: [components.luma, components.chroma],
        lambdas: [lambdas[0] * components.luma_lambda_scale, lambdas[1]],
        ctb: 1 << LOG2_CTB,
        row_step: components.stats_row_step,
        band: components.search.band,
    };
    nc_encoder::sao::decide(
        &Dsp::new(),
        (source, deblocked),
        &settings,
        candidates,
        skipped,
        models,
    )
}

/// Apply `sao` to `recon` (deblocked on entry, SAO output on return).
///
/// Classification reads deblocked samples (§8.7.3.1), so a CTB row with an
/// edge-offset CTB is filtered in place from a band holding its deblocked
/// rows plus one row above and below: the row below is not filtered yet,
/// and the row above was kept before that row was filtered (only when this
/// row needs it). A band (a CTB row) stays in cache, where a copy of the
/// whole plane did not. Rows with only band offsets (or none) read only the
/// sample itself and run in place without a copy.
pub fn apply(recon: &mut Picture, sao: &PictureSao) {
    let dsp = Dsp::new();
    let mut band = Vec::new();
    let mut above = Vec::new();
    for plane in [PlaneIndex::Y, PlaneIndex::U, PlaneIndex::V] {
        let shift = usize::from(plane != PlaneIndex::Y);
        let side = (1usize << LOG2_CTB) >> shift;
        let mut view = recon.plane_mut(plane).view_mut();
        let (width, height, stride) = (view.width(), view.height(), view.stride());
        let rows: Vec<&[(CtbSao, Signal)]> = sao.ctbs.chunks(sao.cols).collect();
        for (row, ctbs) in rows.iter().enumerate() {
            let (y0, y1) = (row * side, ((row + 1) * side).min(height));
            // The next row's edge classification reads this row's last
            // deblocked row: keep it before this row is filtered.
            let next_has_edge = rows.get(row + 1).is_some_and(|next| has_edge(next, plane));
            if !has_edge(ctbs, plane) {
                above.clear();
                if next_has_edge {
                    above.extend_from_slice(&view.as_view().row(y1 - 1)[..width]);
                }
                let rows_view = &mut view.data_mut()[y0 * stride..];
                let mut out = PlaneViewMut::new(rows_view, stride, width, y1 - y0)
                    .expect("the CTB row of the plane");
                for (col, (params, _)) in ctbs.iter().enumerate() {
                    if let Some((positions, offsets)) = band_params(params, plane) {
                        apply_band(&dsp, positions, offsets, &mut out, (col * side, 0, side));
                    }
                }
                continue;
            }
            // Band rows `top..bottom`: one deblocked row each side, where the
            // plane has one.
            let (top, bottom) = (y0.saturating_sub(1), (y1 + 1).min(height));
            band.clear();
            band.extend_from_slice(&above);
            for y in y0..bottom {
                band.extend_from_slice(&view.as_view().row(y)[..width]);
            }
            above.clear();
            if next_has_edge {
                above.extend_from_slice(&band[(y1 - 1 - top) * width..][..width]);
            }
            let input = PlaneView::new(&band, width, width, bottom - top)
                .expect("a band of the plane's width");
            let rows_view = &mut view.data_mut()[top * stride..];
            let mut out = PlaneViewMut::new(rows_view, stride, width, bottom - top)
                .expect("the band's rows of the plane");
            for (col, (params, _)) in ctbs.iter().enumerate() {
                let Some(e) = component(params, plane) else {
                    continue;
                };
                let cr = usize::from(plane == PlaneIndex::V);
                let offsets = e.offsets[cr];
                let origin = (col * side, y0 - top, side);
                match e.kind {
                    SaoKind::Edge(class) => {
                        let rect = class_rect(class, (input.width(), input.height()), origin);
                        dsp.sao_edge_apply(class, offsets, &input, &mut out, rect);
                    }
                    SaoKind::Band(positions) => {
                        apply_band(&dsp, positions[cr], offsets, &mut out, origin);
                    }
                }
            }
        }
        above.clear();
    }
}

/// The SAO parameters of `plane`'s component of one CTB.
fn component(params: &CtbSao, plane: PlaneIndex) -> Option<&ComponentSao> {
    if plane == PlaneIndex::Y {
        params.luma.as_ref()
    } else {
        params.chroma.as_ref()
    }
}

/// Whether any CTB of a row uses edge offset in `plane`.
fn has_edge(ctbs: &[(CtbSao, Signal)], plane: PlaneIndex) -> bool {
    ctbs.iter().any(|(params, _)| {
        component(params, plane).is_some_and(|e| matches!(e.kind, SaoKind::Edge(_)))
    })
}

/// `plane`'s band position and offsets of one CTB, if it uses band offset.
fn band_params(params: &CtbSao, plane: PlaneIndex) -> Option<(u8, [i8; CATEGORIES])> {
    let e = component(params, plane)?;
    let cr = usize::from(plane == PlaneIndex::V);
    match e.kind {
        SaoKind::Band(positions) => Some((positions[cr], e.offsets[cr])),
        SaoKind::Edge(_) => None,
    }
}

/// Band offset of the CTB `(x, y, side)` (plane samples) of `plane`
/// (§8.7.3.2): samples in the four bands from `position` (modulo 32) move by
/// that band's offset. Only the sample itself is read, so it runs in place.
fn apply_band(
    dsp: &Dsp,
    position: u8,
    offsets: [i8; CATEGORIES],
    plane: &mut PlaneViewMut<'_>,
    (x, y, side): (usize, usize, usize),
) {
    let bounds = (
        x,
        y,
        (x + side).min(plane.width()),
        (y + side).min(plane.height()),
    );
    dsp.sao_band_apply(position, offsets, plane, bounds);
}

/// `sao()` syntax of the CTB at CTB position `(rx, ry)` (§7.3.8.3) for the
/// components switched on in the slice header.
pub fn write(
    w: &mut impl BinSink,
    ctx: &mut Contexts,
    sao: &PictureSao,
    components: Components,
    (rx, ry): (usize, usize),
) {
    let (params, signal) = sao.ctbs[ry * sao.cols + rx];
    if rx > 0 {
        w.encode_bin(&mut ctx.sao_merge_flag, signal == Signal::MergeLeft);
        if signal == Signal::MergeLeft {
            return;
        }
    }
    if ry > 0 {
        w.encode_bin(&mut ctx.sao_merge_flag, signal == Signal::MergeUp);
        if signal == Signal::MergeUp {
            return;
        }
    }
    if components.luma {
        write_component(w, ctx, params.luma, 1);
    }
    if components.chroma {
        write_component(w, ctx, params.chroma, 2);
    }
}

/// Type, offsets and class or band position of luma (`planes` 1) or chroma
/// (Cb then Cr; Cr inherits the type and edge class).
fn write_component(
    w: &mut impl BinSink,
    ctx: &mut Contexts,
    params: Option<ComponentSao>,
    planes: usize,
) {
    // sao_type_idx: truncated rice cMax 2, first bin context coded.
    w.encode_bin(&mut ctx.sao_type_idx, params.is_some());
    let Some(e) = params else { return };
    // Second bin: edge (2) or band (1).
    w.encode_bypass(u32::from(matches!(e.kind, SaoKind::Edge(_))), 1);
    for (component, offsets) in e.offsets[..planes].iter().enumerate() {
        for &o in offsets {
            // sao_offset_abs: truncated unary, cMax 7, bypass.
            let a = i32::from(o).unsigned_abs();
            for _ in 0..a {
                w.encode_bypass(1, 1);
            }
            if a < MAX_OFFSET as u32 {
                w.encode_bypass(0, 1);
            }
        }
        match e.kind {
            SaoKind::Band(positions) => {
                for &o in offsets {
                    if o != 0 {
                        w.encode_bypass(u32::from(o < 0), 1); // sao_offset_sign
                    }
                }
                w.encode_bypass(u32::from(positions[component]), BAND_POSITION_BITS);
            }
            // sao_eo_class_luma / sao_eo_class_chroma (FL, 2 bits).
            SaoKind::Edge(class) if component == 0 => w.encode_bypass(u32::from(class), 2),
            SaoKind::Edge(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Band offset (§8.7.3.2): the four bands from the position, modulo 32,
    /// move by their offsets; other samples stay.
    #[test]
    fn band_offset_moves_four_bands_with_wraparound() {
        // Bands are 8 samples wide; position 30 covers bands 30, 31, 0, 1.
        let mut samples = [240u8, 255, 3, 15, 16, 100];
        let mut plane = PlaneViewMut::new(&mut samples, 6, 6, 1).expect("plane");
        apply_band(&Dsp::new(), 30, [-2, 3, 4, -5], &mut plane, (0, 0, 64));
        assert_eq!(samples, [238, 255, 7, 10, 16, 100]);
    }

    /// The categories of §8.7.3.2 on a horizontal triple `a c b`: one sample
    /// classified per plane, counted by the statistics kernel.
    #[test]
    fn categories_follow_the_specification() {
        let category = |c: u8, a: u8, b: u8| -> Option<usize> {
            let rec = [a, c, b];
            let src = [0u8; 3];
            let view = PlaneView::new(&rec, 3, 3, 1).expect("plane");
            let src = PlaneView::new(&src, 3, 3, 1).expect("plane");
            let s = Dsp::new().sao_edge_stats(0, &src, &view, (1, 0, 2, 1));
            s.count.iter().position(|&n| n == 1).map(|i| i + 1)
        };
        assert_eq!(category(10, 20, 20), Some(1)); // local minimum
        assert_eq!(category(10, 10, 20), Some(2)); // concave edge
        assert_eq!(category(10, 10, 10), None); // flat
        assert_eq!(category(20, 20, 10), Some(3)); // convex edge
        assert_eq!(category(20, 10, 10), Some(4)); // local maximum
    }

    /// The shared decision prices a CTB's SAO syntax in exactly the bits
    /// HEVC's writer codes (bypass bins and the two adapting contexts).
    #[test]
    fn shared_rates_match_the_writer() {
        use crate::rate::{BinCosts, BitEstimator};
        let band = ComponentSao {
            kind: SaoKind::Band([30, 4]),
            offsets: [[3, 0, -7, 1], [-1, 2, 0, 7]],
        };
        let edge = ComponentSao {
            kind: SaoKind::Edge(2),
            offsets: [[7, 1, 0, -2], [0, 0, -1, -7]],
        };
        let all = [None, Some(band), Some(edge)];
        let costs = BinCosts::new();
        for (luma, chroma) in all.iter().flat_map(|&l| all.iter().map(move |&c| (l, c))) {
            for signal in [Signal::Explicit, Signal::MergeLeft, Signal::MergeUp] {
                for channels in [[true, true], [true, false], [false, true]] {
                    let params = CtbSao { luma, chroma };
                    let sao = PictureSao {
                        ctbs: vec![(params, signal); 4],
                        cols: 2,
                    };
                    let components = Components {
                        luma: channels[0],
                        chroma: channels[1],
                        luma_lambda_scale: 1.0,
                        stats_row_step: 1,
                        skip_unchanged: false,
                        search: TypeSearch {
                            band: true,
                            interval: 1,
                        },
                    };
                    let mut ctx = Contexts::for_slice(SliceKind::Predicted, 30);
                    let mut models = SaoModels::for_slice(SliceKind::Predicted, 30);
                    let mut estimate = BitEstimator::new(&costs);
                    write(&mut estimate, &mut ctx, &sao, components, (1, 1));
                    let shared = nc_encoder::sao::rate::commit(
                        &mut models,
                        &params,
                        signal,
                        (true, true),
                        channels,
                    );
                    assert!(
                        (shared - f64::from(estimate.bits)).abs() < 1e-3,
                        "{params:?} {signal:?} {channels:?}: {shared} vs {}",
                        estimate.bits
                    );
                    assert_eq!(models.merge, ctx.sao_merge_flag);
                    assert_eq!(models.sao_type, ctx.sao_type_idx);
                }
            }
        }
    }
}
