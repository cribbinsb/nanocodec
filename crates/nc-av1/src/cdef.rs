//! CDEF, the constrained directional enhancement filter (AV1 §7.15), and the
//! encoder's choice of its strengths.
//!
//! The filter must match decoders bit-exactly. The strength adjustment here
//! is ported from libaom (`av1/common/cdef_block.c`: `adjust_strength`;
//! `av1/common/cdef.c`; BSD-2-Clause, Alliance for Open Media); the
//! direction search and the block filter are `nc_dsp::Dsp::cdef_find_dir`
//! and `nc_dsp::Dsp::cdef_filter` (SIMD, same libaom semantics), the filter
//! reading a padded 16-bit copy of the unfiltered plane. Directions come
//! from the unfiltered luma and are found once per call, for every plane,
//! and only when a primary strength is in use (they are unused otherwise).
//!
//! Frames default to one strength set (`cdef_bits = 0`); the optional
//! two-set path selects luma off/model per 64x64 filter block.
//! 8x8 blocks whose four 4x4 units are all `skip` are left unfiltered, as
//! decoders do; the non-skipped ones are listed once per call, so a 64x64
//! filter block with none (libaom `cdef_count == 0`) costs nothing.
//!
//! Strength choice follows SVT-AV1's real-time level 9
//! (`docs/reference/svt-av1-m12-rtc.md` §4): chroma strength from the
//! quantiser model `svt_pick_cdef_from_qp`; luma strength searched over
//! {off, the quantiser model's choice} on "base" frames only, measuring
//! distortion on every 4th row, and reused on the other frames; damping
//! `3 + (base_q_idx >> 6)`.

use nc_common::{Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::{CDEF_PADDING, CdefBlock, CdefSource, Dsp, Moments};

use crate::modeinfo::ModeInfoGrid;

/// Normative CDEF filter-block size, independent of coding partitions.
const FILTER_BLOCK_SIZE: usize = 64;
/// Luma samples covered by one normatively filtered unit.
const FILTER_UNIT_SIZE: usize = 8;

/// One frame's CDEF parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CdefParams {
    /// `CdefDamping - 3` (0..3).
    pub damping_minus_3: u8,
    /// Luma primary strength (0..15).
    pub y_pri: u8,
    /// Luma secondary strength as coded (0..3; 3 means 4).
    pub y_sec: u8,
    /// Chroma primary strength (0..15).
    pub uv_pri: u8,
    /// Chroma secondary strength as coded (0..3; 3 means 4).
    pub uv_sec: u8,
}

/// Secondary strength from its coded value (AV1: 3 means 4).
fn secondary(coded: u8) -> i32 {
    i32::from(coded) + i32::from(coded == 3)
}

impl CdefParams {
    /// Write one or two strength sets (AV1 §5.9.19).
    pub fn write(self, w: &mut nc_common::BitWriter, alternate: Option<Self>) {
        w.put_bits(u32::from(self.damping_minus_3), 2); // cdef_damping_minus_3
        w.put_bits(u32::from(alternate.is_some()), 2); // cdef_bits
        for strength in std::iter::once(self).chain(alternate) {
            debug_assert_eq!(strength.damping_minus_3, self.damping_minus_3);
            w.put_bits(u32::from(strength.y_pri), 4);
            w.put_bits(u32::from(strength.y_sec), 2);
            w.put_bits(u32::from(strength.uv_pri), 4);
            w.put_bits(u32::from(strength.uv_sec), 2);
        }
    }

    fn luma_off(self) -> bool {
        self.y_pri == 0 && self.y_sec == 0
    }

    fn chroma_off(self) -> bool {
        self.uv_pri == 0 && self.uv_sec == 0
    }
}

/// The quantiser model of SVT-AV1, shared with AV2 through the engine.
pub use nc_encoder::cdef_strengths_from_q as strengths_from_q;

/// What [`choose_luma`] found on a frame, for [`apply`] on the same frame to
/// reuse instead of repeating it.
#[derive(Debug, Clone)]
pub struct LumaSearch {
    /// The chosen luma strength: (primary, coded secondary).
    pub strength: (u8, u8),
    /// The units the search listed, if it listed them.
    units: Option<Units>,
    /// The luma filtered with `strength`, when that is not off.
    filtered: Option<Vec<u8>>,
}

/// Choose the luma strength of a base frame: off or `candidate`, whichever
/// leaves less squared error against `source` (registered row stride, default4).
/// `recon` is the unfiltered frame; it is not modified.
#[must_use]
pub fn choose_luma(
    source: &Picture,
    recon: &Picture,
    skips: &ModeInfoGrid,
    params: CdefParams,
    candidate: (u8, u8),
    row_step: usize,
    storage: &mut CdefSource,
) -> LumaSearch {
    if candidate == (0, 0) {
        return LumaSearch {
            strength: candidate,
            units: None,
            filtered: None,
        };
    }
    let dsp = Dsp::new();
    let src = source.plane(PlaneIndex::Y).view();
    let unfiltered = recon.plane(PlaneIndex::Y).view();
    let (width, height) = (unfiltered.width(), unfiltered.height());
    let units = Units::of(&dsp, &unfiltered, skips, candidate.0 != 0);
    let input = &mut *storage;
    input.fill_from(&unfiltered);
    let mut filtered = Vec::with_capacity(width * height);
    for row in unfiltered.rows() {
        filtered.extend_from_slice(row);
    }
    let trial = CdefParams {
        y_pri: candidate.0,
        y_sec: candidate.1,
        ..params
    };
    let mut out = PlaneViewMut::new(&mut filtered, width, width, height)
        .expect("a buffer of the plane's size");
    filter_plane(&dsp, input, &mut out, &units.list, trial, PlaneIndex::Y);
    let sse = |plane: &PlaneView<'_>| -> u64 {
        (0..height)
            .step_by(row_step)
            .map(|r| {
                plane
                    .row(r)
                    .iter()
                    .zip(src.row(r))
                    .map(|(&a, &b)| {
                        let d = i64::from(a) - i64::from(b);
                        (d * d) as u64
                    })
                    .sum::<u64>()
            })
            .sum()
    };
    let filtered_view = PlaneView::new(&filtered, width, width, height).expect("same buffer");
    let better = sse(&filtered_view) < sse(&unfiltered);
    LumaSearch {
        strength: if better { candidate } else { (0, 0) },
        units: Some(units),
        filtered: better.then_some(filtered),
    }
}

/// Choose the frame's chroma strength among `candidates` (plus off), whichever
/// leaves the least squared U+V error against `source` on every `row_step`-th
/// chroma row. SVT-AV1's level-7 CDEF search (flat-IPP P frames of low
/// picture variance at M12) searches UV strengths instead of taking the QP
/// model, which at high quality is off. `recon` is not modified; ties keep
/// the earlier candidate, with off first.
#[must_use]
pub fn choose_chroma(
    source: &Picture,
    recon: &Picture,
    skips: &ModeInfoGrid,
    params: CdefParams,
    candidates: &[(u8, u8)],
    row_step: usize,
    storage: &mut CdefSource,
) -> (u8, u8) {
    let dsp = Dsp::new();
    let with_directions = candidates.iter().any(|&(pri, _)| pri != 0);
    let units = Units::of(
        &dsp,
        &recon.plane(PlaneIndex::Y).view(),
        skips,
        with_directions,
    );
    let mut best = ((0, 0), u64::MAX);
    for strength in std::iter::once((0, 0)).chain(candidates.iter().copied()) {
        let trial = CdefParams {
            uv_pri: strength.0,
            uv_sec: strength.1,
            ..params
        };
        let mut error = 0;
        for plane in [PlaneIndex::U, PlaneIndex::V] {
            let unfiltered = recon.plane(plane).view();
            let (width, height) = (unfiltered.width(), unfiltered.height());
            let mut filtered = Vec::with_capacity(width * height);
            for row in unfiltered.rows() {
                filtered.extend_from_slice(row);
            }
            if strength != (0, 0) {
                storage.fill_from(&unfiltered);
                let mut out = PlaneViewMut::new(&mut filtered, width, width, height)
                    .expect("a buffer of the plane's size");
                filter_plane(&dsp, storage, &mut out, &units.list, trial, plane);
            }
            let view = PlaneView::new(&filtered, width, width, height).expect("same buffer");
            error += sampled_sse(&view, &source.plane(plane).view(), row_step);
        }
        if error < best.1 {
            best = (strength, error);
        }
    }
    best.0
}

/// Squared error of `plane` against `reference` on every `row_step`-th row.
fn sampled_sse(plane: &PlaneView<'_>, reference: &PlaneView<'_>, row_step: usize) -> u64 {
    (0..plane.height())
        .step_by(row_step.max(1))
        .map(|r| {
            plane
                .row(r)
                .iter()
                .zip(reference.row(r))
                .map(|(&a, &b)| {
                    let d = i64::from(a) - i64::from(b);
                    (d * d) as u64
                })
                .sum::<u64>()
        })
        .sum()
}

/// Choose luma off/model independently in each 64x64 filter block and apply
/// the result. Both strength sets share the modeled chroma strengths.
/// `row_step` is the registered distortion-sampling stride (default4).
/// Directions and all filter inputs come from the unfiltered reconstruction.
#[must_use]
pub fn choose_superblocks(
    source: &Picture,
    recon: &mut Picture,
    skips: &ModeInfoGrid,
    params: CdefParams,
    row_step: usize,
    storage: &mut CdefSource,
) -> Vec<bool> {
    let dsp = Dsp::new();
    let unfiltered = recon.plane(PlaneIndex::Y).view();
    let (width, height) = (unfiltered.width(), unfiltered.height());
    let units = Units::of(
        &dsp,
        &unfiltered,
        skips,
        params.y_pri != 0 || params.uv_pri != 0,
    );
    if units.list.is_empty() {
        return vec![false; width.div_ceil(FILTER_BLOCK_SIZE) * height.div_ceil(FILTER_BLOCK_SIZE)];
    }
    // The luma is filtered with the model in place; `storage` keeps the
    // unfiltered samples, which the choice compares against and which are
    // put back in the filter blocks that keep luma off. That costs no
    // whole-plane copy beyond the filter's own input.
    let selected = {
        let mut luma = recon.plane_mut(PlaneIndex::Y).view_mut();
        storage.fill_from(&luma.as_view());
        filter_plane(&dsp, storage, &mut luma, &units.list, params, PlaneIndex::Y);
        let selected = local_luma_choices(
            &source.plane(PlaneIndex::Y).view(),
            &storage.view(),
            &luma.as_view(),
            &units.list,
            row_step,
        );
        restore_unselected(&storage.view(), &mut luma, &units.list, &selected);
        selected
    };
    // Chroma shares one modeled strength, and uses the same unfiltered-luma
    // directions irrespective of which luma option won in a filter block.
    if !params.chroma_off() {
        for plane in [PlaneIndex::U, PlaneIndex::V] {
            let mut out = recon.plane_mut(plane).view_mut();
            filter_plane_in_place(&dsp, storage, &mut out, &units.list, params, plane);
        }
    }
    selected
}

/// Reapply a cached local choice using the current reconstruction directions.
/// Only selected luma units are filtered; chroma retains its current QP model.
/// (Tests; the encoder reuses its storage through the `_with_source` form.)
#[cfg(test)]
pub fn apply_selected(
    recon: &mut Picture,
    skips: &ModeInfoGrid,
    params: CdefParams,
    selected: &[bool],
) {
    apply_selected_with_source(recon, skips, params, selected, &mut CdefSource::new());
}

pub fn apply_selected_with_source(
    recon: &mut Picture,
    skips: &ModeInfoGrid,
    params: CdefParams,
    selected: &[bool],
    storage: &mut CdefSource,
) {
    let dsp = Dsp::new();
    let luma = recon.plane(PlaneIndex::Y).view();
    let columns = luma.width().div_ceil(FILTER_BLOCK_SIZE);
    let units = Units::of(&dsp, &luma, skips, params.y_pri != 0 || params.uv_pri != 0);
    let luma_units: Vec<Unit> = units
        .list
        .iter()
        .copied()
        .filter(|unit| selected[unit.y / FILTER_BLOCK_SIZE * columns + unit.x / FILTER_BLOCK_SIZE])
        .collect();
    for plane in PlaneIndex::ALL {
        let mut out = recon.plane_mut(plane).view_mut();
        let list = if plane == PlaneIndex::Y {
            &luma_units
        } else {
            &units.list
        };
        filter_plane_in_place(&dsp, storage, &mut out, list, params, plane);
    }
}

/// Outside the active 8x8 filter units, model and off samples are identical.
/// Their SSE terms cancel exactly. Accumulate only those units, retaining the
/// original sampling phase relative to each 64x64 filter block.
fn local_luma_choices(
    source: &PlaneView<'_>,
    unfiltered: &PlaneView<'_>,
    filtered: &PlaneView<'_>,
    units: &[Unit],
    row_step: usize,
) -> Vec<bool> {
    let (width, height) = (unfiltered.width(), unfiltered.height());
    let columns = width.div_ceil(FILTER_BLOCK_SIZE);
    let mut improvements = vec![0_i64; columns * height.div_ceil(FILTER_BLOCK_SIZE)];
    for unit in units {
        let phase = (unit.y % FILTER_BLOCK_SIZE) % row_step;
        let first = unit.y + (row_step - phase) % row_step;
        let right = (unit.x + FILTER_UNIT_SIZE).min(width);
        // One unit's gain is at most 64 * 255^2 in magnitude: exact in i32.
        let mut gain = 0_i32;
        for y in (first..(unit.y + FILTER_UNIT_SIZE).min(height)).step_by(row_step) {
            gain += source.row(y)[unit.x..right]
                .iter()
                .zip(&unfiltered.row(y)[unit.x..right])
                .zip(&filtered.row(y)[unit.x..right])
                .map(|((&original, &off), &model)| {
                    let off_error = i32::from(original) - i32::from(off);
                    let model_error = i32::from(original) - i32::from(model);
                    off_error * off_error - model_error * model_error
                })
                .sum::<i32>();
        }
        improvements[unit.y / FILTER_BLOCK_SIZE * columns + unit.x / FILTER_BLOCK_SIZE] +=
            i64::from(gain);
    }
    improvements.into_iter().map(|gain| gain > 0).collect()
}

/// Put the `unfiltered` samples back into the units of `luma` whose 64x64
/// filter block keeps luma off (`selected` false). Only listed units were
/// filtered, so the rest of the plane already holds them.
fn restore_unselected(
    unfiltered: &PlaneView<'_>,
    luma: &mut PlaneViewMut<'_>,
    units: &[Unit],
    selected: &[bool],
) {
    let (width, height) = (luma.width(), luma.height());
    let columns = width.div_ceil(FILTER_BLOCK_SIZE);
    for unit in units {
        if selected[unit.y / FILTER_BLOCK_SIZE * columns + unit.x / FILTER_BLOCK_SIZE] {
            continue;
        }
        let right = (unit.x + FILTER_UNIT_SIZE).min(width);
        for y in unit.y..(unit.y + FILTER_UNIT_SIZE).min(height) {
            luma.row_mut(y)[unit.x..right].copy_from_slice(&unfiltered.row(y)[unit.x..right]);
        }
    }
}

/// Apply CDEF to `recon` in place (AV1 §7.15). `search` is this frame's
/// [`choose_luma`] result, if it ran: its units and filtered luma are
/// reused where they match `params`. (Tests; the encoder reuses its storage
/// through [`apply_with_source`].)
#[cfg(test)]
pub fn apply(
    recon: &mut Picture,
    skips: &ModeInfoGrid,
    params: CdefParams,
    search: Option<&LumaSearch>,
) {
    apply_with_source(recon, skips, params, search, &mut CdefSource::new());
}

pub fn apply_with_source(
    recon: &mut Picture,
    skips: &ModeInfoGrid,
    params: CdefParams,
    search: Option<&LumaSearch>,
    storage: &mut CdefSource,
) {
    if params.luma_off() && params.chroma_off() {
        return;
    }
    let dsp = Dsp::new();
    // Directions come from the unfiltered luma, even when only chroma is
    // filtered, so they are found before any plane changes.
    let primary =
        (!params.luma_off() && params.y_pri != 0) || (!params.chroma_off() && params.uv_pri != 0);
    let found = search
        .and_then(|s| s.units.as_ref())
        .filter(|units| units.with_directions || !primary);
    let computed;
    let units = if let Some(units) = found {
        units
    } else {
        computed = Units::of(&dsp, &recon.plane(PlaneIndex::Y).view(), skips, primary);
        &computed
    };
    // The search filtered the luma with exactly these parameters when it
    // chose its candidate.
    let filtered_luma = search
        .filter(|s| s.strength == (params.y_pri, params.y_sec))
        .and_then(|s| s.filtered.as_deref());
    for plane in PlaneIndex::ALL {
        let off = if plane == PlaneIndex::Y {
            params.luma_off()
        } else {
            params.chroma_off()
        };
        if off {
            continue;
        }
        let mut out = recon.plane_mut(plane).view_mut();
        if let (PlaneIndex::Y, Some(luma)) = (plane, filtered_luma) {
            let (width, height) = (out.width(), out.height());
            out.copy_from(
                &PlaneView::new(luma, width, width, height).expect("a luma-sized buffer"),
            );
            continue;
        }
        filter_plane_in_place(&dsp, storage, &mut out, &units.list, params, plane);
    }
}

/// The units of a frame that CDEF filters, in raster order, and whether
/// their directions were found.
#[derive(Debug, Clone)]
struct Units {
    list: Vec<Unit>,
    with_directions: bool,
}

impl Units {
    /// List the units; directions are found in the unfiltered `luma` when
    /// `with_directions`, otherwise left 0, which is what a zero primary
    /// strength uses.
    fn of(dsp: &Dsp, luma: &PlaneView<'_>, skips: &ModeInfoGrid, with_directions: bool) -> Self {
        Self {
            list: filtered_units(dsp, luma, skips, with_directions),
            with_directions,
        }
    }
}

/// An 8x8 luma unit CDEF filters (not all `skip`), at luma sample
/// (`x`, `y`), with its direction and directional contrast.
#[derive(Debug, Clone, Copy)]
struct Unit {
    x: usize,
    y: usize,
    dir: u8,
    var: i32,
}

/// The units of a frame that CDEF filters, in raster order (see
/// [`Units::of`]).
fn filtered_units(
    dsp: &Dsp,
    luma: &PlaneView<'_>,
    skips: &ModeInfoGrid,
    with_directions: bool,
) -> Vec<Unit> {
    let mut units = Vec::new();
    for mi_row in (0..skips.mi_rows).step_by(2) {
        let top = skips.row(mi_row);
        let bottom = skips.row((mi_row + 1).min(skips.mi_rows - 1));
        for mi_col in (0..skips.mi_cols).step_by(2) {
            let right = (mi_col + 1).min(skips.mi_cols - 1);
            // libaom `is_8x8_block_skip`: all four 4x4 units skip (repeating
            // the last row or column at the frame's edge changes nothing).
            if top[mi_col].skip && top[right].skip && bottom[mi_col].skip && bottom[right].skip {
                continue;
            }
            let (x, y) = (mi_col * 4, mi_row * 4);
            let (dir, var) = if with_directions {
                dsp.cdef_find_dir(luma, (x, y))
            } else {
                (0, 0)
            };
            units.push(Unit { x, y, dir, var });
        }
    }
    units
}

/// Filter every listed unit of one plane from `input` into `output`.
fn filter_plane(
    dsp: &Dsp,
    input: &CdefSource,
    output: &mut PlaneViewMut<'_>,
    units: &[Unit],
    params: CdefParams,
    plane: PlaneIndex,
) {
    let (filter, blocks) = plane_blocks(units, params, plane);
    dsp.cdef_filter_blocks(
        filter.size,
        input,
        &blocks,
        filter.sec,
        filter.damping,
        output,
    );
}

/// [`filter_plane`] in place: `plane` is both the unfiltered input and the
/// output, with `storage` holding the unfiltered samples. The rows are
/// copied into `storage` one 64x64 filter-block row at a time, just before
/// that band is filtered, instead of the whole plane up front: each band is
/// copied while it is in cache and read back soon after.
fn filter_plane_in_place(
    dsp: &Dsp,
    storage: &mut CdefSource,
    plane_view: &mut PlaneViewMut<'_>,
    units: &[Unit],
    params: CdefParams,
    plane: PlaneIndex,
) {
    let (filter, blocks) = plane_blocks(units, params, plane);
    // One filter-block row of this plane (64 luma rows, 32 chroma rows).
    let band = FILTER_BLOCK_SIZE * filter.size / FILTER_UNIT_SIZE;
    storage.start(plane_view.width(), plane_view.height());
    let mut rest = blocks.as_slice();
    while let Some(first) = rest.first() {
        let band_end = (first.y / band + 1) * band;
        let (now, later) = rest.split_at(rest.partition_point(|block| block.y < band_end));
        // The band's blocks read up to CDEF_PADDING rows below it; rows
        // above it were copied, still unfiltered, with the previous band.
        storage.fill_rows_to(&plane_view.as_view(), band_end + CDEF_PADDING);
        dsp.cdef_filter_blocks(
            filter.size,
            storage,
            now,
            filter.sec,
            filter.damping,
            plane_view,
        );
        rest = later;
    }
}

/// Block size, secondary strength and damping shared by a plane's blocks.
#[derive(Debug, Clone, Copy)]
struct PlaneFilter {
    size: usize,
    sec: u8,
    damping: u8,
}

/// The blocks of one plane that `params` changes, in raster order (as
/// `units`), with their adjusted strengths and directions.
fn plane_blocks(
    units: &[Unit],
    params: CdefParams,
    plane: PlaneIndex,
) -> (PlaneFilter, Vec<CdefBlock>) {
    let chroma = plane != PlaneIndex::Y;
    let shift = usize::from(chroma);
    let size = 8 >> shift;
    let damping = 3 + params.damping_minus_3 - u8::from(chroma);
    let (pri, sec) = if chroma {
        (i32::from(params.uv_pri), secondary(params.uv_sec))
    } else {
        (i32::from(params.y_pri), secondary(params.y_sec))
    };
    let blocks = units
        .iter()
        .filter_map(|unit| {
            let adjusted = if chroma {
                pri
            } else {
                adjust_strength(pri, unit.var)
            };
            // Skipped blocks: the output already holds the unfiltered samples.
            (adjusted != 0 || sec != 0).then(|| CdefBlock {
                x: unit.x >> shift,
                y: unit.y >> shift,
                pri: adjusted as u8,
                dir: if pri == 0 { 0 } else { unit.dir },
            })
        })
        .collect();
    let filter = PlaneFilter {
        size,
        sec: sec as u8,
        damping,
    };
    (filter, blocks)
}

/// libaom `adjust_strength`: scale the luma primary strength by the block's
/// directional contrast.
pub(crate) fn adjust_strength(strength: i32, var: i32) -> i32 {
    if var == 0 {
        return 0;
    }
    let i = if var >> 6 != 0 {
        (31 - (var >> 6).leading_zeros() as i32).min(12)
    } else {
        0
    };
    (strength * (4 + i) + 8) >> 4
}

/// Superblock side of the picture-variance estimate.
const VARIANCE_BLOCK: usize = 64;
/// Side of the quadrants whose means the estimate averages.
const VARIANCE_QUADRANT: usize = 32;
/// Side of the blocks `Dsp::moments_8x8_row` sums.
const MOMENTS_BLOCK: usize = 8;

/// Sum and sum of squares of one quadrant's sampled (even-row) samples. A
/// quadrant's 16 x 32 samples sum to at most 2^17 and their squares to at
/// most 2^25, so 32 bits are exact.
#[derive(Debug, Clone, Copy, Default)]
struct QuadrantSums {
    sum: u32,
    square_sum: u32,
}

/// Exact RTC `BLOCK_MEAN_PREC_SUB` picture variance, in 8-bit sample squared units.
/// Preserve all three levels of fixed-point truncation and edge replication.
///
/// Only even rows are sampled. Superblocks inside the picture sum them with
/// `Dsp::moments_8x8_row` over a view of the even rows; superblocks that
/// cross the right or bottom edge replicate edge samples and use
/// [`edge_quadrant_sums`].
pub fn picture_variance(dsp: &Dsp, luma: &PlaneView<'_>) -> u32 {
    let (width, height) = (luma.width(), luma.height());
    let even_rows = PlaneView::new(luma.data(), 2 * luma.stride(), width, height.div_ceil(2))
        .expect("even rows of a valid view");
    let inner_columns = width / VARIANCE_BLOCK * VARIANCE_BLOCK;
    let mut moments = vec![Moments::default(); inner_columns / MOMENTS_BLOCK];
    let mut row_sums = vec![[QuadrantSums::default(); 4]; inner_columns / VARIANCE_BLOCK];
    let mut total = 0_u64;
    let mut count = 0;
    for top in (0..height).step_by(VARIANCE_BLOCK) {
        let inner_row = top + VARIANCE_BLOCK <= height;
        if inner_row {
            inner_row_sums(dsp, &even_rows, top, &mut moments, &mut row_sums);
        }
        for left in (0..width).step_by(VARIANCE_BLOCK) {
            let quadrants = if inner_row && left + VARIANCE_BLOCK <= width {
                row_sums[left / VARIANCE_BLOCK]
            } else {
                std::array::from_fn(|q| edge_quadrant_sums(luma, left, top, q))
            };
            let mut mean = 0_u64;
            let mut square_sum = 0_u64;
            for quadrant in quadrants {
                square_sum += u64::from(quadrant.square_sum);
                // 8x8 Q8 means are sum<<3; 16x16 averaging is exact.
                // The 32x32 truncation leaves floor(quadrant_sum/2).
                mean += u64::from(quadrant.sum >> 1);
            }
            mean >>= 2;
            // All three Q16 square-mean averages are exact: sum<<11 /64.
            total += ((square_sum << 5) - mean * mean) >> 16;
            count += 1;
        }
    }
    (total / count) as u32
}

/// Quadrant sums of every inner superblock in the superblock row at `top`:
/// four rows of 8x8 moments over the even rows (16 picture rows each pair).
fn inner_row_sums(
    dsp: &Dsp,
    even_rows: &PlaneView<'_>,
    top: usize,
    moments: &mut [Moments],
    row_sums: &mut [[QuadrantSums; 4]],
) {
    const BLOCKS_PER_QUADRANT: usize = VARIANCE_QUADRANT / MOMENTS_BLOCK;
    row_sums.fill([QuadrantSums::default(); 4]);
    let block_rows = VARIANCE_BLOCK / 2 / MOMENTS_BLOCK;
    for block_row in 0..block_rows {
        let rows = even_rows
            .sub_view(
                0,
                top / 2 + block_row * MOMENTS_BLOCK,
                moments.len() * MOMENTS_BLOCK,
                MOMENTS_BLOCK,
            )
            .expect("inner superblock rows lie inside the picture");
        dsp.moments_8x8_row(&rows, moments);
        let lower = block_row / (block_rows / 2);
        for (i, m) in moments.iter().enumerate() {
            let column = i % (2 * BLOCKS_PER_QUADRANT) / BLOCKS_PER_QUADRANT;
            let sums = &mut row_sums[i / (2 * BLOCKS_PER_QUADRANT)][2 * lower + column];
            sums.sum += m.sum;
            sums.square_sum += m.sum_sq;
        }
    }
}

/// Quadrant `quadrant` (raster order) of the superblock at `left`, `top`,
/// replicating the last column and row past the picture's edge.
fn edge_quadrant_sums(
    luma: &PlaneView<'_>,
    left: usize,
    top: usize,
    quadrant: usize,
) -> QuadrantSums {
    let qx = left + (quadrant % 2) * VARIANCE_QUADRANT;
    let qy = top + (quadrant / 2) * VARIANCE_QUADRANT;
    let mut sums = QuadrantSums::default();
    for dy in (0..VARIANCE_QUADRANT).step_by(2) {
        let row = luma.row((qy + dy).min(luma.height() - 1));
        for dx in 0..VARIANCE_QUADRANT {
            let value = u32::from(row[(qx + dx).min(luma.width() - 1)]);
            sums.sum += value;
            sums.square_sum += value * value;
        }
    }
    sums
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modeinfo::ModeInfo;
    use nc_common::BitWriter;

    fn choose_fresh(
        source: &Picture,
        recon: &mut Picture,
        skips: &ModeInfoGrid,
        params: CdefParams,
        row_step: usize,
    ) -> Vec<bool> {
        choose_superblocks(
            source,
            recon,
            skips,
            params,
            row_step,
            &mut CdefSource::new(),
        )
    }

    fn same_pixels(a: &Picture, b: &Picture) {
        for plane in PlaneIndex::ALL {
            assert!(
                a.plane(plane)
                    .view()
                    .rows()
                    .zip(b.plane(plane).view().rows())
                    .all(|(left, right)| left == right)
            );
        }
    }

    #[test]
    fn reusable_workspace_matches_fresh_inputs_across_frames_sizes_and_modes() {
        let mut reused = CdefSource::new();
        for (frame, (width, height)) in [(72, 72), (72, 72), (128, 64), (8, 8), (64, 128), (72, 72)]
            .into_iter()
            .enumerate()
        {
            let mut recon = Picture::new(width, height, 0).unwrap();
            for plane in PlaneIndex::ALL {
                let mut view = recon.plane_mut(plane).view_mut();
                for y in 0..view.height() {
                    for (x, value) in view.row_mut(y).iter_mut().enumerate() {
                        *value = (x * 17 + y * 31 + frame * 47 + plane as usize * 59) as u8;
                    }
                }
            }
            let mut source = recon.clone();
            for value in source.plane_mut(PlaneIndex::Y).view_mut().row_mut(0) {
                *value = value.wrapping_add(7);
            }
            let grid = ModeInfoGrid::new(width / 4, height / 4);
            let params = CdefParams {
                damping_minus_3: 2,
                y_pri: 5,
                y_sec: 1,
                uv_pri: 3,
                uv_sec: 2,
            };
            let mut actual = recon.clone();
            let mut expected = recon.clone();
            apply_with_source(&mut actual, &grid, params, None, &mut reused);
            apply(&mut expected, &grid, params, None);
            same_pixels(&actual, &expected);
            assert_eq!((reused.width(), reused.height()), (width / 2, height / 2));
            let found = choose_luma(&source, &recon, &grid, params, (5, 1), 4, &mut reused);
            let fresh = choose_luma(
                &source,
                &recon,
                &grid,
                params,
                (5, 1),
                4,
                &mut CdefSource::new(),
            );
            assert_eq!(found.strength, fresh.strength);
            assert_eq!(found.filtered, fresh.filtered);
            let mut actual = recon.clone();
            let mut expected = recon.clone();
            let selected = choose_superblocks(&source, &mut actual, &grid, params, 3, &mut reused);
            let selected_fresh = choose_superblocks(
                &source,
                &mut expected,
                &grid,
                params,
                3,
                &mut CdefSource::new(),
            );
            assert_eq!(selected, selected_fresh);
            same_pixels(&actual, &expected);
            let mut actual = recon.clone();
            let mut expected = recon.clone();
            apply_selected_with_source(&mut actual, &grid, params, &selected, &mut reused);
            apply_selected(&mut expected, &grid, params, &selected);
            same_pixels(&actual, &expected);
        }
    }

    fn hierarchical_variance_oracle(luma: &PlaneView<'_>) -> u32 {
        let mut total = 0_u64;
        let mut count = 0;
        for top in (0..luma.height()).step_by(64) {
            for left in (0..luma.width()).step_by(64) {
                let mut means = [0_u64; 64];
                let mut squares = [0_u64; 64];
                for by in 0..8 {
                    for bx in 0..8 {
                        let index = by * 8 + bx;
                        for dy in (0..8).step_by(2) {
                            let row = luma.row((top + by * 8 + dy).min(luma.height() - 1));
                            for dx in 0..8 {
                                let value =
                                    u64::from(row[(left + bx * 8 + dx).min(luma.width() - 1)]);
                                means[index] += value;
                                squares[index] += value * value;
                            }
                        }
                        means[index] <<= 3;
                        squares[index] <<= 11;
                    }
                }
                for side in [8, 4, 2] {
                    for y in 0..side / 2 {
                        for x in 0..side / 2 {
                            let indices = [
                                y * 2 * side + x * 2,
                                y * 2 * side + x * 2 + 1,
                                (y * 2 + 1) * side + x * 2,
                                (y * 2 + 1) * side + x * 2 + 1,
                            ];
                            means[y * side / 2 + x] =
                                indices.iter().map(|&i| means[i]).sum::<u64>() >> 2;
                            squares[y * side / 2 + x] =
                                indices.iter().map(|&i| squares[i]).sum::<u64>() >> 2;
                        }
                    }
                }
                total += (squares[0] - means[0] * means[0]) >> 16;
                count += 1;
            }
        }
        (total / count) as u32
    }

    #[test]
    fn quadrant_variance_equals_fixed_point_hierarchy() {
        for (width, height) in [(8, 8), (64, 64), (72, 72), (200, 120), (128, 192)] {
            for seed in 0..8_u32 {
                let mut state = seed + 1;
                let pixels: Vec<u8> = (0..width * height)
                    .map(|_| {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        ((state >> 24) % (32 + seed * 31).min(256)) as u8
                    })
                    .collect();
                let view = PlaneView::new(&pixels, width, width, height).unwrap();
                assert_eq!(
                    picture_variance(&Dsp::new(), &view),
                    hierarchical_variance_oracle(&view),
                    "{width}x{height}, seed{seed}"
                );
            }
        }
    }

    #[test]
    fn reference_variance_scale_and_classification_boundary() {
        for (amplitude, expected) in [(37, 342), (38, 361)] {
            let pixels: Vec<u8> = (0..4096)
                .map(|i| if i % 2 == 0 { 0 } else { amplitude })
                .collect();
            let view = PlaneView::new(&pixels, 64, 64, 64).unwrap();
            assert_eq!(picture_variance(&Dsp::new(), &view), expected);
        }
        let edge: Vec<u8> = (0..72 * 72)
            .map(|i| if i % 72 % 2 == 0 { 0 } else { 40 })
            .collect();
        assert_eq!(
            picture_variance(&Dsp::new(), &PlaneView::new(&edge, 72, 72, 72).unwrap()),
            246
        );
    }

    #[test]
    fn strength_table_has_exact_legacy_and_two_set_syntax() {
        let params = CdefParams {
            damping_minus_3: 2,
            y_pri: 1,
            y_sec: 2,
            uv_pri: 3,
            uv_sec: 1,
        };
        let mut one = BitWriter::new();
        params.write(&mut one, None);
        assert_eq!(one.into_bytes(), [0x81, 0x8d]);
        let mut two = BitWriter::new();
        params.write(
            &mut two,
            Some(CdefParams {
                y_pri: 0,
                y_sec: 0,
                ..params
            }),
        );
        two.byte_align_zero();
        assert_eq!(two.into_bytes(), [0x91, 0x8d, 0x00, 0xd0]);
    }

    #[test]
    fn sparse_selection_matches_dense_sse_for_every_sampling_stride() {
        let (width, height) = (200, 120);
        let mut recon = Picture::new(width, height, 0).unwrap();
        let mut source = recon.clone();
        for y in 0..height {
            for x in 0..width {
                let value = (80 + (x * 3 + y * 7 + (x ^ y) % 5) % 80) as u8;
                recon.plane_mut(PlaneIndex::Y).view_mut().row_mut(y)[x] = value;
                let noise = ((x * 13 + y * 19) % 11) as i32 - 5;
                source.plane_mut(PlaneIndex::Y).view_mut().row_mut(y)[x] =
                    (i32::from(value) + noise).clamp(0, 255) as u8;
            }
        }
        let mut grid = ModeInfoGrid::new(width / 4, height / 4);
        for row in 0..grid.mi_rows {
            for col in 0..grid.mi_cols {
                if (row / 2 + col / 2).is_multiple_of(5) {
                    grid.fill(
                        row,
                        col,
                        ModeInfo {
                            skip: true,
                            ..ModeInfo::default()
                        },
                    );
                }
            }
        }
        let params = CdefParams {
            damping_minus_3: 1,
            y_pri: 8,
            y_sec: 2,
            uv_pri: 0,
            uv_sec: 0,
        };
        let mut model = recon.clone();
        apply(&mut model, &grid, params, None);
        let original = source.plane(PlaneIndex::Y).view();
        let unfiltered = recon.plane(PlaneIndex::Y).view();
        let filtered: Vec<u8> = model
            .plane(PlaneIndex::Y)
            .view()
            .rows()
            .flatten()
            .copied()
            .collect();
        let units = Units::of(&Dsp::new(), &unfiltered, &grid, true);
        let filtered_view = PlaneView::new(&filtered, width, width, height).unwrap();
        for row_step in 1..=8 {
            let sparse = local_luma_choices(
                &original,
                &unfiltered,
                &filtered_view,
                &units.list,
                row_step,
            );
            let mut dense = Vec::new();
            for top in (0..height).step_by(FILTER_BLOCK_SIZE) {
                for left in (0..width).step_by(FILTER_BLOCK_SIZE) {
                    let mut improvement = 0_i64;
                    for y in (top..(top + FILTER_BLOCK_SIZE).min(height)).step_by(row_step) {
                        for x in left..(left + FILTER_BLOCK_SIZE).min(width) {
                            let value = i32::from(original.row(y)[x]);
                            let off = value - i32::from(unfiltered.row(y)[x]);
                            let model = value - i32::from(filtered[y * width + x]);
                            improvement += i64::from(off * off - model * model);
                        }
                    }
                    dense.push(improvement > 0);
                }
            }
            assert_eq!(sparse, dense, "row stride{row_step}");
        }
    }

    #[test]
    fn local_choices_equal_independent_full_frame_filters_including_edges() {
        for (width, height, row_step) in [(192, 128, 1), (200, 120, 4), (200, 120, 8)] {
            let mut recon = Picture::new(width, height, 0).unwrap();
            for plane in PlaneIndex::ALL {
                let mut out = recon.plane_mut(plane).view_mut();
                for y in 0..out.height() {
                    for (x, value) in out.row_mut(y).iter_mut().enumerate() {
                        *value = (80 + (x * 3 + y * 7 + (x ^ y) % 5) % 80) as u8;
                    }
                }
            }
            let grid = ModeInfoGrid::new(width / 4, height / 4);
            let params = CdefParams {
                damping_minus_3: 1,
                y_pri: 8,
                y_sec: 2,
                uv_pri: 3,
                uv_sec: 1,
            };
            let mut model = recon.clone();
            apply(&mut model, &grid, params, None);
            let mut off = recon.clone();
            apply(
                &mut off,
                &grid,
                CdefParams {
                    y_pri: 0,
                    y_sec: 0,
                    ..params
                },
                None,
            );
            let mut source = off.clone();
            for y in 0..height {
                for x in 0..width {
                    if (x / FILTER_BLOCK_SIZE + y / FILTER_BLOCK_SIZE).is_multiple_of(2) {
                        source.plane_mut(PlaneIndex::Y).view_mut().row_mut(y)[x] =
                            model.plane(PlaneIndex::Y).view().row(y)[x];
                    }
                }
            }
            let mut actual = recon.clone();
            let selected = choose_fresh(&source, &mut actual, &grid, params, row_step);
            let mut cached = recon.clone();
            apply_selected(&mut cached, &grid, params, &selected);
            same_pixels(&cached, &actual);
            assert!(selected.iter().any(|&choice| choice));
            assert!(selected.iter().any(|&choice| !choice));
            let columns = width.div_ceil(FILTER_BLOCK_SIZE);
            for plane in PlaneIndex::ALL {
                let out = actual.plane(plane).view();
                for y in 0..out.height() {
                    for (x, &value) in out.row(y).iter().enumerate() {
                        let model_chosen = plane != PlaneIndex::Y
                            || selected[y / FILTER_BLOCK_SIZE * columns + x / FILTER_BLOCK_SIZE];
                        let expected = if model_chosen { &model } else { &off };
                        assert_eq!(value, expected.plane(plane).view().row(y)[x]);
                    }
                }
            }
            let mut skipped = grid;
            for row in 0..skipped.mi_rows {
                for col in 0..skipped.mi_cols {
                    skipped.fill(
                        row,
                        col,
                        ModeInfo {
                            skip: true,
                            ..ModeInfo::default()
                        },
                    );
                }
            }
            let mut all_skipped = recon.clone();
            assert!(
                choose_fresh(&source, &mut all_skipped, &skipped, params, row_step)
                    .iter()
                    .all(|&v| !v)
            );
            for plane in PlaneIndex::ALL {
                for (a, b) in all_skipped
                    .plane(plane)
                    .view()
                    .rows()
                    .zip(recon.plane(plane).view().rows())
                {
                    assert_eq!(a, b);
                }
            }
        }
    }
}
