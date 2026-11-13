//! H.266 §8.8 edge-offset SAO using the existing shared DSP statistics/kernels.
//! Syntax follows `VVenC` (BSD-3-Clause-Clear; `THIRD_PARTY.md`). Four EO classes,
//! independent luma/chroma choices, CTU merges left/up (§7.3.11.3); band
//! offsets are not emitted.
use crate::{coeff::Bins, contexts::Contexts};
use nc_common::{
    Picture, PlaneIndex, PlaneView,
    decode::{DecodeError, Result},
};
use nc_dsp::{Dsp, SaoStats};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CtuSao {
    pub classes: [Option<u8>; 2],
    pub offsets: [[i8; 4]; 3],
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
fn plane_size(picture: &Picture, plane: PlaneIndex) -> (usize, usize) {
    let plane = picture.plane(plane);
    (plane.width(), plane.height())
}

/// Largest edge-offset magnitude (`sao_offset_abs`, 8-bit).
const MAX_OFFSET: i32 = 7;

/// Each category's offset of least `error + lambda * bits` over magnitudes
/// 0..=7 (signs fixed by category), and the summed cost.
///
/// Only magnitudes up to the ceiling of the error's unconstrained minimum
/// (`sum / count` towards the category's sign) are priced: above it the
/// error grows strictly with the magnitude and the bits never fall, so no
/// larger magnitude costs less, and the first of equal costs is kept as in
/// a full scan.
fn offsets(stats: SaoStats, lambda: f32) -> ([i8; 4], f32) {
    let mut chosen = [0; 4];
    let mut cost = 0.;
    for (i, offset) in chosen.iter_mut().enumerate() {
        let sign: i32 = if i < 2 { 1 } else { -1 };
        let (count, toward) = (stats.count[i], i64::from(sign) * stats.sum[i]);
        let largest = if count <= 0 || toward <= 0 {
            0
        } else {
            (toward as u64)
                .div_ceil(count as u64)
                .min(MAX_OFFSET as u64) as i32
        };
        let (value, rd) = (0..=largest)
            .map(|abs| {
                let value = abs * sign;
                let error =
                    stats.count[i] * i64::from(value * value) - 2 * stats.sum[i] * i64::from(value);
                (
                    value,
                    error as f32 + lambda * ((abs + 1).min(MAX_OFFSET) as f32),
                )
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        *offset = value as i8;
        cost += rd;
    }
    (chosen, cost)
}
/// Edge-offset classes (§8.8.4.2: horizontal, vertical, 135°, 45°).
const CLASSES: usize = 4;
/// Bits of a CTU's own parameters beyond its offsets, per enabled channel:
/// the type (edge offset) and the two class bits.
const CLASS_BITS: f32 = 3.;
/// Approximate bits of `sao_merge_left_flag` = 1, and of
/// `sao_merge_left_flag` = 0 then `sao_merge_up_flag` = 1.
const MERGE_LEFT_BITS: f32 = 1.;
const MERGE_UP_BITS: f32 = 2.;

/// One CTU's edge statistics: every plane (Y, U, V) under every class.
type CtuStats = [[SaoStats; CLASSES]; 3];

/// Each CTU's luma and chroma edge-offset class (None: off).
pub(crate) type Classes = [Option<u8>; 2];

/// How [`decide`] searches.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Search<'a> {
    /// Statistics read every `row_step`-th row (1..=4, weighted back to the
    /// full count: HEVC's `hevc.sao_stats_row_step`), which leaves the
    /// classification unchanged.
    pub row_step: usize,
    /// CTUs whose CUs are all skip CUs (`vvc.sao_skip_unchanged`): they
    /// copy already filtered reference samples, so SAO stays off there,
    /// without statistics (HEVC's `hevc.sao_skip_unchanged`).
    pub skipped: Option<&'a [bool]>,
    /// Each CTU's classes from the last searched picture
    /// (`vvc.sao_search_interval`): only their statistics are gathered and
    /// their offsets re-derived, a quarter of the work (HEVC's class reuse).
    pub classes: Option<&'a [Classes]>,
}

/// Choose each CTU's edge-offset classes and offsets: its own best, or its
/// left or upper neighbour's parameters where merging them (§7.3.11.3,
/// coded by [`code`] whenever they are equal) costs less.
/// `lambdas` are the luma and chroma channels' (`Quantizers::chroma_lambda`);
/// CTUs are `ctu` luma samples square, in raster order.
pub(crate) fn decide(
    dsp: &Dsp,
    source: &Picture,
    recon: &Picture,
    lambdas: [f32; 2],
    ctu: usize,
    search: &Search,
) -> Vec<CtuSao> {
    let columns = recon.width().div_ceil(ctu);
    let mut all: Vec<CtuSao> = Vec::new();
    for y in (0..recon.height()).step_by(ctu) {
        for x in (0..recon.width()).step_by(ctu) {
            let index = all.len();
            if search.skipped.is_some_and(|s| s[index]) {
                all.push(CtuSao::default());
                continue;
            }
            let allowed = search.classes.map(|c| c[index]);
            let stats = ctu_statistics(dsp, source, recon, (x, y), ctu, search.row_step, allowed);
            let (mut chosen, mut best) = own_parameters(&stats, lambdas, allowed);
            let neighbours = [
                (x > 0).then(|| (all[index - 1], MERGE_LEFT_BITS)),
                (y > 0).then(|| (all[index - columns], MERGE_UP_BITS)),
            ];
            for (neighbour, bits) in neighbours.into_iter().flatten() {
                // Reused classes leave the other classes without statistics.
                if allowed.is_some_and(|a| !within(&neighbour, a)) {
                    continue;
                }
                let cost = change(&stats, &neighbour) + lambdas[0] * bits;
                if cost < best {
                    (chosen, best) = (neighbour, cost);
                }
            }
            all.push(chosen);
        }
    }
    all
}

/// The classes `params` chose, for reuse by later pictures.
pub(crate) fn classes(params: &[CtuSao]) -> Vec<Classes> {
    params.iter().map(|p| p.classes).collect()
}

/// Whether every channel `params` turns on uses the class in `allowed`.
fn within(params: &CtuSao, allowed: Classes) -> bool {
    params
        .classes
        .iter()
        .zip(allowed)
        .all(|(class, allowed)| class.is_none() || *class == allowed)
}

/// The classes a channel may take: all four, or only the reused one.
fn candidate_classes(allowed: Option<Classes>, channel: usize) -> Vec<u8> {
    match allowed {
        None => (0..CLASSES as u8).collect(),
        Some(classes) => classes[channel].into_iter().collect(),
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

/// The edge statistics of the `ctu`-sample CTU at luma `position`, of the
/// `allowed` classes only when given.
fn ctu_statistics(
    dsp: &Dsp,
    source: &Picture,
    recon: &Picture,
    position: (usize, usize),
    ctu: usize,
    row_step: usize,
    allowed: Option<Classes>,
) -> CtuStats {
    let mut stats = CtuStats::default();
    for channel in 0..2 {
        for class in candidate_classes(allowed, channel) {
            for &plane in channel_planes(channel) {
                stats[plane as usize][usize::from(class)] = dsp.sao_edge_stats_sampled(
                    class,
                    &source.plane(plane).view(),
                    &recon.plane(plane).view(),
                    bounds(plane_size(recon, plane), channel, position, ctu, class),
                    row_step,
                );
            }
        }
    }
    stats
}

/// A CTU's best own parameters (each channel off or its cheapest class
/// among the `allowed` ones) and their cost: the squared-error change plus
/// λ times their bits.
fn own_parameters(stats: &CtuStats, lambdas: [f32; 2], allowed: Option<Classes>) -> (CtuSao, f32) {
    let mut ctu = CtuSao::default();
    let mut total = 0.;
    for (channel, lambda) in lambdas.into_iter().enumerate() {
        let planes = channel_planes(channel);
        let mut best = 0.;
        for class in candidate_classes(allowed, channel) {
            let mut cost = lambda * CLASS_BITS;
            let mut values = [[0; 4]; 3];
            for &plane in planes {
                let (offsets, rd) = offsets(stats[plane as usize][usize::from(class)], lambda);
                values[plane as usize] = offsets;
                cost += rd;
            }
            if cost < best {
                best = cost;
                ctu.classes[channel] = Some(class);
                for &plane in planes {
                    ctu.offsets[plane as usize] = values[plane as usize];
                }
            }
        }
        total += best;
    }
    (ctu, total)
}

/// The squared-error change of applying `params` to a CTU with `stats`.
fn change(stats: &CtuStats, params: &CtuSao) -> f32 {
    let mut error = 0;
    for plane in PlaneIndex::ALL {
        let Some(class) = params.classes[usize::from(plane != PlaneIndex::Y)] else {
            continue;
        };
        let stats = stats[plane as usize][usize::from(class)];
        for (i, &value) in params.offsets[plane as usize].iter().enumerate() {
            let value = i64::from(value);
            error += stats.count[i] * value * value - 2 * stats.sum[i] * value;
        }
    }
    error as f32
}
/// Apply `params`, one per `ctu`-sample CTU in raster order.
pub(crate) fn apply(dsp: &Dsp, picture: &mut Picture, params: &[CtuSao], ctu: usize) {
    for plane in PlaneIndex::ALL {
        apply_plane(dsp, picture, plane, params, ctu);
    }
}

/// [`apply`] to one plane, CTU row by CTU row. Edge offsets classify from
/// the deblocked samples, so each row's are copied (with the row below)
/// into a band buffer before it is filtered, and the row above comes from
/// the band before: no copy of the whole picture, and none at all for rows
/// whose CTUs leave the plane alone.
fn apply_plane(dsp: &Dsp, picture: &mut Picture, plane: PlaneIndex, params: &[CtuSao], ctu: usize) {
    let channel = usize::from(plane != PlaneIndex::Y);
    let (width, height) = plane_size(picture, plane);
    let band_rows = ctu >> channel;
    let columns = picture.width().div_ceil(ctu);
    // The band's rows with the one above and the one below it.
    let mut band = vec![0; width * (band_rows + 2)];
    // The deblocked last row of the band before, if that band was filtered.
    let mut above = vec![0; width];
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
        for (column, sao) in ctus.iter().enumerate() {
            let Some(class) = sao.classes[channel] else {
                continue;
            };
            let (x0, y0, x1, y1) = bounds(
                (width, height),
                channel,
                (column * ctu, row * ctu),
                ctu,
                class,
            );
            dsp.sao_edge_apply(
                class,
                sao.offsets[plane as usize],
                &input,
                &mut output,
                (x0, y0 - first, x1, y1 - first),
            );
        }
    }
}
/// One CTU's SAO syntax (§7.3.11.3). `left` and `up` are the neighbours'
/// parameters where they exist; `source` merges with the first that equals
/// it. Returns the parameters as coded.
pub(crate) fn code(
    io: &mut impl Bins,
    c: &mut Contexts,
    source: CtuSao,
    left: Option<CtuSao>,
    up: Option<CtuSao>,
) -> Result<CtuSao> {
    for neighbour in [left, up].into_iter().flatten() {
        if io.bin(&mut c.sao_merge, source == neighbour)? {
            return Ok(neighbour);
        }
    }
    let mut decoded = CtuSao::default();
    for plane in 0..3 {
        let channel = usize::from(plane != 0);
        let enabled = if plane == 2 {
            decoded.classes[1].is_some()
        } else {
            io.bin(&mut c.sao_type, source.classes[channel].is_some())?
        };
        if !enabled {
            continue;
        }
        if plane != 2 && io.bypass(1, 1)? == 0 {
            return Err(DecodeError::Unsupported("VVC SAO band offset"));
        }
        for i in 0..4 {
            let mut abs = 0;
            while abs < 7
                && io.bypass(u32::from(source.offsets[plane][i].unsigned_abs() > abs), 1)? != 0
            {
                abs += 1;
            }
            decoded.offsets[plane][i] = abs as i8 * if i < 2 { 1 } else { -1 };
        }
        if plane != 2 {
            decoded.classes[channel] =
                Some(io.bypass(u32::from(source.classes[channel].unwrap_or(0)), 2)? as u8);
        }
    }
    Ok(decoded)
}
