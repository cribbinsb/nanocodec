//! Sample adaptive offset (HEVC §7.3.8.3, §8.7.3). The encoder decides edge
//! offsets only; band offsets are coded and applied for the syntax stress
//! test (`backend/stress.rs`) and decoded.
//!
//! Per CTB and component, SAO classifies every deblocked sample by its two
//! neighbours along one of four directions (the edge-offset class) into
//! "local minimum, concave edge, convex edge, local maximum" (categories 1
//! to 4) and adds a per-category offset — positive for 1 and 2, negative
//! for 3 and 4. Cb and Cr share their class; each has its own offsets.
//!
//! The encoder's choice follows SVT-AV1's cheap CDEF strength reuse
//! (`docs/reference/svt-av1-m12-rtc.md` §4): on *search* frames (key frames
//! and every [`SEARCH_INTERVAL`]-th frame) every class is evaluated; other
//! frames reuse the class chosen for the co-located CTB on the last search
//! frame and only derive its offsets. Offsets are the rounded mean
//! difference to the source per category; a CTB uses SAO, merges with its
//! left or upper neighbour's parameters, or turns SAO off, whichever costs
//! least in squared error plus λ times the signalling bits.

use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::Dsp;

use crate::cabac::CabacWriter;
use crate::contexts::Contexts;
use crate::headers::LOG2_CTB;

/// Number of edge-offset classes.
const CLASSES: usize = 4;
/// Categories with an offset (1..4; 0 means no change).
const CATEGORIES: usize = 4;
/// Largest `sao_offset_abs` for 8-bit video: `(1 << (8 − 5)) − 1`.
pub const MAX_OFFSET: i32 = 7;
/// Search all classes every this many frames (and on key frames).
pub const SEARCH_INTERVAL: u64 = 4;

/// Bits of `sao_band_position` (fixed length, bypass).
const BAND_POSITION_BITS: u32 = 5;
/// `bandShift` for 8-bit video (`bitDepth − 5`, §8.7.3.2): 32 bands.
const BAND_SHIFT: u32 = 3;
/// Number of bands.
pub const BANDS: u8 = 32;

/// How a component's samples are classified (`sao_type_idx`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaoKind {
    /// Edge offset with this `sao_eo_class` (shared by Cb and Cr).
    Edge(u8),
    /// Band offset with each plane's `sao_band_position` (Y, or Cb and Cr).
    Band([u8; 2]),
}

/// SAO parameters of one component (luma), or of chroma (shared type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentSao {
    /// Edge or band offset.
    pub kind: SaoKind,
    /// Offsets of categories (edge) or of the four bands from the band
    /// position (band) per plane (Y, or Cb and Cr), signed as applied. Edge
    /// categories 1 and 2 are ≥ 0, 3 and 4 are ≤ 0.
    pub offsets: [[i8; CATEGORIES]; 2],
}

impl ComponentSao {
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

/// Sum of (source − deblocked) and sample count per category, per class.
#[derive(Debug, Clone, Copy, Default)]
struct ClassStats {
    sum: [i64; CATEGORIES],
    count: [i64; CATEGORIES],
}

impl ClassStats {
    /// Best offsets and the squared-error change they give.
    fn best_offsets(&self) -> ([i8; CATEGORIES], i64) {
        let mut offsets = [0i8; CATEGORIES];
        let mut delta = 0;
        for (cat, offset) in offsets.iter_mut().enumerate() {
            let n = self.count[cat];
            if n == 0 {
                continue;
            }
            let mean = (self.sum[cat] as f64 / n as f64).round() as i32;
            // Categories 1, 2 only brighten; 3, 4 only darken (§7.4.9.3).
            let o = if cat < 2 {
                mean.clamp(0, MAX_OFFSET)
            } else {
                mean.clamp(-MAX_OFFSET, 0)
            };
            *offset = o as i8;
            delta += self.delta(cat, o);
        }
        (offsets, delta)
    }

    /// Squared-error change of applying offset `o` to category `cat`.
    fn delta(&self, cat: usize, o: i32) -> i64 {
        let o = i64::from(o);
        self.count[cat] * o * o - 2 * o * self.sum[cat]
    }

    fn delta_of(&self, offsets: [i8; CATEGORIES]) -> i64 {
        (0..CATEGORIES)
            .map(|c| self.delta(c, i32::from(offsets[c])))
            .sum()
    }
}

/// The samples of the CTB `(cx, cy)` (in plane samples, side `ctb`) of a
/// `width` x `height` plane that have both neighbours of `class` inside the
/// picture, as `(x0, y0, x1, y1)` (possibly empty).
fn class_rect(
    class: usize,
    (width, height): (usize, usize),
    (cx, cy, ctb): (usize, usize, usize),
) -> (usize, usize, usize, usize) {
    let [(ax, ay), (bx, by)] = nc_dsp::SAO_EO_NEIGHBOURS[class];
    let (left, right) = (usize::from(ax < 0 || bx < 0), usize::from(ax > 0 || bx > 0));
    let (top, bottom) = (usize::from(ay < 0 || by < 0), usize::from(ay > 0 || by > 0));
    (
        cx.max(left),
        cy.max(top),
        (cx + ctb).min(width.saturating_sub(right)),
        (cy + ctb).min(height.saturating_sub(bottom)),
    )
}

/// Edge-offset statistics of one CTB of one plane for `class`.
///
/// A row step above one approximates parameter selection by sampling rows;
/// the selected normative filter still runs on every pixel. Step two saves
/// 3.26% GOP300 instructions, costing 0.34% PSNR / 0.50% VMAF-NEG BD-rate
/// on supported VAL. Noise best PSNR loses up to 0.033dB on VAL (0.198dB
/// on TRAIN): this optional tradeoff does not pass the strict no-loss guard.
fn stats(
    dsp: &Dsp,
    source: &nc_common::Plane,
    deblocked: &nc_common::Plane,
    (cx, cy, ctb): (usize, usize, usize),
    class: usize,
    row_step: usize,
) -> ClassStats {
    let rec = deblocked.view();
    let rect = class_rect(class, (rec.width(), rec.height()), (cx, cy, ctb));
    let s = dsp.sao_edge_stats_sampled(class as u8, &source.view(), &rec, rect, row_step);
    ClassStats {
        sum: s.sum,
        count: s.count,
    }
}

/// Approximate bits of explicit edge-offset parameters for one component
/// (type, class, four truncated-unary offsets).
fn edge_bits(offsets: &[[i8; CATEGORIES]], with_class: bool) -> f64 {
    let type_and_class = 2.0 + if with_class { 2.0 } else { 0.0 };
    let offset_bits: f64 = offsets
        .iter()
        .flatten()
        .map(|&o| f64::from(i32::from(o).abs().min(MAX_OFFSET - 1) + 1))
        .sum();
    type_and_class + offset_bits
}

/// Which components SAO is applied to (`slice_sao_luma_flag`,
/// `slice_sao_chroma_flag`) and the λ multiple for the luma decision.
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
}

/// Decide the SAO parameters of a picture from the deblocked
/// reconstruction. `reuse` is the per-CTB classes of the last search frame
/// (`None` on a search frame). CTBs marked in `skipped` (all CUs skipped)
/// get SAO off, merged from a neighbour where it is off too.
pub fn decide(
    source: &Picture,
    deblocked: &Picture,
    lambda: f64,
    components: Components,
    reuse: Option<&[(Option<u8>, Option<u8>)]>,
    skipped: Option<&[bool]>,
) -> PictureSao {
    let dsp = Dsp::new();
    let ctb = 1usize << LOG2_CTB;
    let (w, h) = (deblocked.width(), deblocked.height());
    let cols = w.div_ceil(ctb);
    let rows = h.div_ceil(ctb);
    let mut ctbs: Vec<(CtbSao, Signal)> = Vec::with_capacity(cols * rows);
    for ry in 0..rows {
        for rx in 0..cols {
            let index = ry * cols + rx;
            if skipped.is_some_and(|s| s[index]) {
                ctbs.push(unfiltered(&ctbs, index, cols));
                continue;
            }
            let (reuse_luma, reuse_chroma) = reuse.map_or((None, None), |r| r[index]);
            let luma_component = Component {
                dsp: &dsp,
                stats_row_step: components.stats_row_step,
                source,
                deblocked,
                planes: &[PlaneIndex::Y],
                rect: (rx * ctb, ry * ctb, ctb),
            };
            let chroma_component = Component {
                dsp: &dsp,
                stats_row_step: components.stats_row_step,
                source,
                deblocked,
                planes: &[PlaneIndex::U, PlaneIndex::V],
                rect: (rx * ctb / 2, ry * ctb / 2, ctb / 2),
            };
            let classes = |on: bool, reused: Option<u8>| match (on, reuse) {
                (false, _) => Classes::Off,
                (true, None) => Classes::Search,
                (true, Some(_)) => Classes::reuse(reused),
            };
            let (luma, mut luma_stats) = decide_component(
                &luma_component,
                lambda * components.luma_lambda_scale,
                classes(components.luma, reuse_luma),
            );
            let (chroma, mut chroma_stats) = decide_component(
                &chroma_component,
                lambda,
                classes(components.chroma, reuse_chroma),
            );
            let own = CtbSao { luma, chroma };
            // Cost of the CTB's own parameters vs copying a neighbour's.
            let mut cost_of = |p: &CtbSao| -> f64 {
                cost_with(p.luma, &mut luma_stats, &luma_component)
                    + cost_with(p.chroma, &mut chroma_stats, &chroma_component)
            };
            let own_bits = edge_bits_of(&own, components);
            let own_cost = cost_of(&own)
                + lambda * (own_bits + f64::from(u8::from(rx > 0)) + f64::from(u8::from(ry > 0)));
            let mut choice = (own, Signal::Explicit, own_cost);
            if rx > 0 {
                let left = ctbs[index - 1].0;
                let c = cost_of(&left) + lambda;
                if c < choice.2 {
                    choice = (left, Signal::MergeLeft, c);
                }
            }
            if ry > 0 {
                let up = ctbs[index - cols].0;
                let c = cost_of(&up) + lambda * f64::from(1 + u8::from(rx > 0));
                if c < choice.2 {
                    choice = (up, Signal::MergeUp, c);
                }
            }
            ctbs.push((choice.0, choice.1));
        }
    }
    PictureSao { ctbs, cols }
}

/// SAO off for the CTB at `index` (`ctbs` holds the CTBs before it):
/// merged from the left or above neighbour when that is off too (one bin),
/// else signalled. A skipped CTB copies already filtered samples, so it
/// gets no further filtering and needs no statistics.
fn unfiltered(ctbs: &[(CtbSao, Signal)], index: usize, cols: usize) -> (CtbSao, Signal) {
    let off = CtbSao {
        luma: None,
        chroma: None,
    };
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

/// Which edge-offset classes to evaluate for a component of a CTB.
#[derive(Debug, Clone, Copy)]
enum Classes {
    /// All four (a search frame).
    Search,
    /// Only the class reused from the last search frame.
    Only(u8),
    /// None: SAO was off there on the last search frame.
    Off,
}

impl Classes {
    fn reuse(class: Option<u8>) -> Self {
        class.map_or(Classes::Off, Classes::Only)
    }
}

/// Per-class statistics of a component's planes (one or two); `None` for
/// a class not evaluated (a reuse frame evaluates only the reused class).
type ComponentStats = [Option<[ClassStats; 2]>; CLASSES];

/// One component of one CTB: the planes and their CTB rectangle.
struct Component<'a> {
    dsp: &'a Dsp,
    stats_row_step: usize,
    source: &'a Picture,
    deblocked: &'a Picture,
    planes: &'a [PlaneIndex],
    rect: (usize, usize, usize),
}

impl Component<'_> {
    fn stats(&self, class: usize) -> [ClassStats; 2] {
        let (cx, cy, ctb) = self.rect;
        let mut out = [ClassStats::default(); 2];
        for (o, &plane) in out.iter_mut().zip(self.planes) {
            *o = stats(
                self.dsp,
                self.source.plane(plane),
                self.deblocked.plane(plane),
                (cx, cy, ctb),
                class,
                self.stats_row_step,
            );
        }
        out
    }
}

/// Squared-error change of applying `p` (a neighbour's parameters when
/// costing a merge); statistics of a class not yet evaluated are gathered
/// here, so a merge is never costed against missing statistics.
fn cost_with(p: Option<ComponentSao>, stats: &mut ComponentStats, component: &Component) -> f64 {
    p.map_or(0.0, |e| {
        let class = usize::from(
            e.edge_class()
                .expect("the encoder's decisions use edge offsets only"),
        );
        let s = *stats[class].get_or_insert_with(|| component.stats(class));
        (s[0].delta_of(e.offsets[0]) + s[1].delta_of(e.offsets[1])) as f64
    })
}

fn edge_bits_of(p: &CtbSao, components: Components) -> f64 {
    let luma = p.luma.map_or(1.0, |e| edge_bits(&e.offsets[..1], true));
    let chroma = p.chroma.map_or(1.0, |e| edge_bits(&e.offsets, true));
    f64::from(u8::from(components.luma)) * luma + f64::from(u8::from(components.chroma)) * chroma
}

/// The best edge offset (or off) for one component of one CTB, and the
/// statistics of every class computed (for merge costing).
fn decide_component(
    component: &Component,
    lambda: f64,
    classes: Classes,
) -> (Option<ComponentSao>, ComponentStats) {
    let mut all = ComponentStats::default();
    let classes: Vec<usize> = match classes {
        Classes::Search => (0..CLASSES).collect(),
        Classes::Only(c) => vec![usize::from(c)],
        Classes::Off => vec![],
    };
    // Off costs the one-bin type.
    let mut best: (Option<ComponentSao>, f64) = (None, lambda);
    for class in classes {
        let mut offsets = [[0i8; CATEGORIES]; 2];
        let mut delta = 0;
        let s = component.stats(class);
        for (i, s) in s.iter().enumerate().take(component.planes.len()) {
            let (o, d) = s.best_offsets();
            offsets[i] = o;
            delta += d;
        }
        all[class] = Some(s);
        let bits = edge_bits(&offsets[..component.planes.len()], true);
        let cost = delta as f64 + lambda * bits;
        if cost < best.1 {
            best = (
                Some(ComponentSao {
                    kind: SaoKind::Edge(class as u8),
                    offsets,
                }),
                cost,
            );
        }
    }
    (best.0, all)
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
                        apply_band(positions, offsets, &mut out, (col * side, 0, side));
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
                        let rect =
                            class_rect(usize::from(class), (input.width(), input.height()), origin);
                        dsp.sao_edge_apply(class, offsets, &input, &mut out, rect);
                    }
                    SaoKind::Band(positions) => {
                        apply_band(positions[cr], offsets, &mut out, origin);
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
    position: u8,
    offsets: [i8; CATEGORIES],
    plane: &mut PlaneViewMut<'_>,
    origin: (usize, usize, usize),
) {
    let (x, y, side) = origin;
    let mut band_offset = [0i32; BANDS as usize];
    for (k, &offset) in offsets.iter().enumerate() {
        band_offset[(usize::from(position) + k) % usize::from(BANDS)] = i32::from(offset);
    }
    let x_end = (x + side).min(plane.width());
    for row in y..(y + side).min(plane.height()) {
        for sample in &mut plane.row_mut(row)[x..x_end] {
            let shifted = i32::from(*sample) + band_offset[usize::from(*sample >> BAND_SHIFT)];
            *sample = shifted.clamp(0, i32::from(u8::MAX)) as u8;
        }
    }
}

/// The per-CTB edge classes `(luma, chroma)` of a decision, for reuse.
#[must_use]
pub fn classes(sao: &PictureSao) -> Vec<(Option<u8>, Option<u8>)> {
    sao.ctbs
        .iter()
        .map(|(p, _)| {
            (
                p.luma.and_then(ComponentSao::edge_class),
                p.chroma.and_then(ComponentSao::edge_class),
            )
        })
        .collect()
}

/// `sao()` syntax of the CTB at CTB position `(rx, ry)` (§7.3.8.3) for the
/// components switched on in the slice header.
pub fn write(
    w: &mut CabacWriter,
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
    w: &mut CabacWriter,
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
        apply_band(30, [-2, 3, 4, -5], &mut plane, (0, 0, 64));
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
}
