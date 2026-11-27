//! Loop restoration of the general decoder (dav2d `lr_apply_tmpl.c`:
//! `dav2d_lr_slice`, `lr_sbrow`, `lr_stripe`): per stripe and 64-wide
//! column, the restoration unit covering it chooses the filter: luma's
//! non-separable Wiener (the frame's filter, a filter per class, or the
//! unit's own) or PC-Wiener (a pretrained filter per class), chroma's
//! non-separable Wiener with its part on the co-located luma. The kernels
//! are nc-dsp's [`nc_dsp::av2_dec::wiener`].
//!
//! Every filter reads the frame after CDEF and CCSO (`source`) inside its
//! stripe and the deblocked frame next to it ([`super::stripes`]); its
//! output goes to `picture`, which GDF then corrects
//! ([`super::gdf`], also from `source`).

use nc_common::{BitDepth, Picture, PlaneIndex, PlaneView, PlaneViewMut, Sample};
use nc_dsp::av2_dec::wiener::{self, CHROMA_TAPS, ClassModel, LumaFilter, WINDOW_BORDER, Window};
use nc_dsp::av2_dec::wiener_tables::{
    FEATURE_CLASS, FRAME_FILTER_OF_CLASS, PRETRAINED_FILTER_OF_CLASS, PRETRAINED_FILTERS,
};

use super::stripes::{Boundary, STRIPE_CONTEXT_ROWS, Stripe, TileSpans, stripes};
use super::{FilterMaps, RestorationUnit};
use crate::general::frame_header::FrameHeader;
use crate::general::frame_header::restoration::{NsWiener, kind};
use crate::general::sequence::SequenceHeader;
use crate::transform::dequantizer_step;

/// Width of a restoration column (dav2d `lr_sbrow`'s 64-wide steps).
const COLUMN: usize = 64;
/// Quantizer index limits of the class tables' four quantizer classes
/// (dav2d `init_wiener`).
const QUANTIZER_CLASS_LIMITS: [u16; 3] = [130, 190, 220];
/// Luma samples per 4x4 unit, as a shift.
const UNIT_LOG2: usize = 2;
/// Samples of a 4x4 block's side.
const BLOCK: usize = 4;

/// What the frame's restoration needs besides the maps.
struct Frame<'a> {
    header: &'a FrameHeader,
    /// The quantizer class of the class tables.
    quantizer_class: usize,
    /// The class model at the frame's quantizer step.
    model: ClassModel,
    /// `cfl_ds_filter_index`: how chroma's luma part is downsampled.
    downsample: u8,
    /// The tiles' extents: restoration units restart at each tile.
    tiles: TileSpans,
    /// The sample depth.
    depth: BitDepth,
}

/// Restore `picture` from `source` (the frame after CDEF and CCSO) and
/// `deblocked`, per the block layer's restoration units in `maps`.
pub fn filter<T: Sample>(
    picture: &mut Picture<T>,
    (source, deblocked): (&Picture<T>, &Picture<T>),
    maps: &FilterMaps,
    (header, sequence, depth): (&FrameHeader, &SequenceHeader, BitDepth),
) {
    let qindex = header.quantizer.base.min(sequence.max_qindex());
    // The classes count raw indices at every depth (dav2d `init_wiener`).
    let quantizer_class = QUANTIZER_CLASS_LIMITS
        .iter()
        .take_while(|&&limit| qindex >= limit)
        .count();
    // The model's step is on the 8-bit scale, rounded (dav2d `qstep`).
    let extra = depth.above_eight();
    let step = dequantizer_step(i32::from(qindex)) as i32;
    let luma_source = source.plane(PlaneIndex::Y).view();
    let frame = Frame {
        header,
        quantizer_class,
        model: ClassModel::new((step + ((1 << extra) >> 1)) >> extra),
        downsample: sequence.intra.cfl_downsample_filter,
        depth: depth.of::<T>(),
        tiles: TileSpans::new(
            header,
            (luma_source.width(), luma_source.height()),
            sequence.filters.disable_across_tiles,
        ),
    };
    let luma_deblocked = deblocked.plane(PlaneIndex::Y).view();
    if header.restoration.planes[0].kind != kind::NONE {
        let mut out = picture.plane_mut(PlaneIndex::Y).view_mut();
        restore_luma(&mut out, (&luma_source, &luma_deblocked), maps, &frame);
    }
    for (index, plane) in [(1, PlaneIndex::U), (2, PlaneIndex::V)] {
        if header.restoration.planes[index].kind == kind::NONE {
            continue;
        }
        let chroma = (source.plane(plane).view(), deblocked.plane(plane).view());
        let mut out = picture.plane_mut(plane).view_mut();
        restore_chroma(
            &mut out,
            (&chroma.0, &chroma.1),
            (&luma_source, &luma_deblocked),
            (maps, index),
            &frame,
        );
    }
}

/// The luma column of the unit each 64-wide restoration column of a
/// plane subsampled by `ss` takes, in a frame `width` luma samples wide,
/// units `1 << unit_log2` samples of that plane (dav2d `lr_sbrow`'s walk of
/// `next_iter_start_x`): the unit at the column's unit-aligned position,
/// or the one before it where that is the last of the frame's row, starts
/// less than half a unit from the frame's right edge and is not where the
/// tile the walk has reached starts. The walk moves on one tile at most per
/// column.
fn unit_columns(tiles: &TileSpans, width: usize, unit_log2: u8, ss: usize) -> Vec<usize> {
    let unit = 1_usize << unit_log2;
    let plane_width = width >> ss;
    let starts: Vec<usize> = tiles.columns.iter().map(|&start| start >> ss).collect();
    let mut tile = 0;
    let mut columns = vec![0];
    let mut x = 0;
    while x + COLUMN < width {
        x += COLUMN;
        let mut aligned = (x >> ss) & !(unit - 1);
        if tile + 1 < starts.len() && aligned >= starts[tile + 1] {
            tile += 1;
        }
        if aligned > starts[tile] && plane_width - aligned < unit / 2 {
            aligned -= unit;
        }
        columns.push(aligned << ss);
    }
    columns
}

/// The luma row of the unit covering a 64-row band starting at luma row
/// `band_top` of a plane subsampled by `ss` (dav2d `lr_sbrow`'s
/// `aligned_unit_pos_*`): units are laid out from the top of the band's
/// tile row (`tile_rows`: the tile row starts, then the frame height), and
/// the tile row's last unit absorbs a remainder below half a unit (AVM
/// v1.0.0 places units per tile, `av2_get_rutile_rect`;
/// dav2d 7e899beb).
fn unit_row(band_top: usize, tile_rows: &[usize], unit_log2: u8, ss: usize) -> usize {
    let unit = 1_usize << unit_log2;
    // The band of a frame's last stripe can start at its bottom.
    let starts = &tile_rows[..tile_rows.len() - 1];
    let tile = starts.partition_point(|&start| start <= band_top) - 1;
    let (start, end) = (tile_rows[tile] >> ss, tile_rows[tile + 1] >> ss);
    let aligned = start + (((band_top >> ss) - start) & !(unit - 1));
    let aligned = if aligned > start && aligned + unit / 2 > end {
        aligned - unit
    } else {
        aligned
    };
    aligned << ss
}

/// The unit of plane `plane` at luma column `column` (from
/// [`unit_columns`]) covering a stripe's band.
fn unit_at<'a>(
    maps: &'a FilterMaps,
    plane: usize,
    (stripe, column, tiles): (Stripe, usize, &TileSpans),
    unit_log2: u8,
) -> Option<&'a RestorationUnit> {
    let ss = usize::from(plane > 0);
    let row = unit_row(stripe.band_top, &tiles.rows, unit_log2, ss);
    let block = maps.filter_block((row >> UNIT_LOG2) as u32, (column >> UNIT_LOG2) as u32);
    maps.restoration[plane][block].as_ref()
}

/// Fill a window around the block at columns `left..left + width` of
/// `stripe` from `source` (in the stripe) and `deblocked` (next to it);
/// columns past `span_start..span_end` (the plane, or its tile with loop
/// filters kept within tiles: AVM v1.0.0
/// `av2_loop_restoration_filter_unit`'s `tile_boundary_left`/`_right`)
/// repeat its edge columns.
fn fill_window<T: Sample>(
    window: &mut [T],
    stride: usize,
    (source, deblocked): (&PlaneView<'_, T>, &PlaneView<'_, T>),
    stripe: Stripe,
    (left, width): (usize, usize),
    (span_start, span_end): (usize, usize),
) {
    let rows = stripe.height() + 2 * WINDOW_BORDER;
    for (j, out) in window.chunks_exact_mut(stride).take(rows).enumerate() {
        let offset = j as isize - WINDOW_BORDER as isize;
        let row = match stripe.context_row(offset, STRIPE_CONTEXT_ROWS) {
            (y, false) => source.row(y),
            (y, true) => deblocked.row(y),
        };
        for (i, sample) in out[..width + 2 * WINDOW_BORDER].iter_mut().enumerate() {
            let x = (left + i)
                .saturating_sub(WINDOW_BORDER)
                .clamp(span_start, span_end - 1);
            *sample = row[x];
        }
    }
}

/// Luma's restoration, stripe by stripe and column by column.
fn restore_luma<T: Sample>(
    out: &mut PlaneViewMut<'_, T>,
    (source, deblocked): (&PlaneView<'_, T>, &PlaneView<'_, T>),
    maps: &FilterMaps,
    frame: &Frame<'_>,
) {
    let depth = frame.depth.of::<T>();
    let (width, height) = (source.width(), source.height());
    let restoration = &frame.header.restoration;
    let wiener = &restoration.planes[0].wiener;
    let stride = COLUMN + 2 * WINDOW_BORDER;
    let mut window = vec![T::default(); stride * stride];
    let unit_log2 = restoration.unit_log2[0];
    let columns = unit_columns(&frame.tiles, width, unit_log2, 0);
    for stripe in stripes(height, frame.tiles.row_starts(), frame.tiles.isolated) {
        for left in (0..width).step_by(COLUMN) {
            let column = columns[left / COLUMN];
            let Some(unit) = unit_at(maps, 0, (stripe, column, &frame.tiles), unit_log2) else {
                continue;
            };
            if unit.kind == kind::NONE {
                continue;
            }
            let w = (width - left).min(COLUMN);
            let span = frame.tiles.column_span(left);
            fill_window(
                &mut window,
                stride,
                (source, deblocked),
                stripe,
                (left, w),
                span,
            );
            let view = Window {
                samples: &window,
                stride,
            };
            let classified =
                unit.kind == kind::PC_WIENER || (wiener.frame_filters && wiener.classes > 1);
            let skips = if classified {
                stripe_skip_counts(maps, stripe, (left, w))
            } else {
                Vec::new()
            };
            let blocks_across = w / BLOCK;
            for by in 0..stripe.height() / BLOCK {
                for bx in 0..blocks_across {
                    let (x, y) = (bx * BLOCK, by * BLOCK);
                    let class = classified.then(|| {
                        let skip = skips[by * blocks_across + bx];
                        let index = frame.model.feature_index(&view, (x, y), skip, depth);
                        usize::from(FEATURE_CLASS[index])
                    });
                    let filter = luma_filter(unit, wiener, class, frame);
                    let out_stride = out.stride();
                    let start = (stripe.top + y) * out_stride + left + x;
                    wiener::luma_block(
                        &view,
                        filter,
                        (x, y),
                        &mut out.data_mut()[start..],
                        out_stride,
                        depth,
                    );
                }
            }
        }
    }
}

/// The luma filter of a block of class `class` (when classified) in a
/// unit of type `unit.kind`.
fn luma_filter<'a>(
    unit: &'a RestorationUnit,
    wiener: &'a NsWiener,
    class: Option<usize>,
    frame: &Frame<'_>,
) -> LumaFilter<'a> {
    let q = frame.quantizer_class;
    match (unit.kind, class) {
        (kind::PC_WIENER, Some(class)) => {
            let filter = usize::from(PRETRAINED_FILTER_OF_CLASS[q][class]);
            LumaFilter::Pretrained(&PRETRAINED_FILTERS[q][filter])
        }
        (_, Some(class)) => {
            let classes = usize::from(wiener.class_index) - 1;
            let filter = usize::from(FRAME_FILTER_OF_CLASS[q][classes][class]);
            LumaFilter::NonSeparable(&wiener.filters[filter])
        }
        _ if wiener.frame_filters => LumaFilter::NonSeparable(&wiener.filters[0]),
        _ => LumaFilter::NonSeparable(&unit.filters[0]),
    }
}

/// Each 4x4 block's skip count in the stripe's column (rows above the
/// stripe's first and below its last repeat them; dav2d `lr_stripe`'s
/// `noskip_mask` rows).
fn stripe_skip_counts(
    maps: &FilterMaps,
    stripe: Stripe,
    (left, width): (usize, usize),
) -> Vec<usize> {
    let stride = maps.luma_edges.columns;
    let rows = stripe.height() / BLOCK;
    let row = |r: usize| {
        let start = ((stripe.top >> UNIT_LOG2) + r) * stride + (left >> UNIT_LOG2);
        &maps.residual[start..start + width / BLOCK]
    };
    let mut counts = Vec::with_capacity(rows * width / BLOCK);
    for r in 0..rows {
        let above = row(r.saturating_sub(1));
        let below = row((r + 1).min(rows - 1));
        counts.extend(wiener::skip_counts([above, row(r), below]));
    }
    counts
}

/// One chroma plane's restoration (4:2:0), stripe by stripe and column by
/// column.
fn restore_chroma<T: Sample>(
    out: &mut PlaneViewMut<'_, T>,
    (source, deblocked): (&PlaneView<'_, T>, &PlaneView<'_, T>),
    luma: (&PlaneView<'_, T>, &PlaneView<'_, T>),
    (maps, plane): (&FilterMaps, usize),
    frame: &Frame<'_>,
) {
    let depth = frame.depth.of::<T>();
    let (width, height) = (source.width(), source.height());
    let (luma_width, luma_height) = (luma.0.width(), luma.0.height());
    let restoration = &frame.header.restoration;
    let wiener = &restoration.planes[plane].wiener;
    let column = COLUMN / 2;
    let stride = column + 2 * WINDOW_BORDER;
    let mut window = vec![T::default(); stride * (COLUMN / 2 + 2 * WINDOW_BORDER)];
    let mut luma_window = vec![T::default(); stride * (COLUMN / 2 + 2 * WINDOW_BORDER)];
    let unit_log2 = restoration.unit_log2[1];
    let unit_columns = unit_columns(&frame.tiles, luma_width, unit_log2, 1);
    for luma_stripe in stripes(luma_height, frame.tiles.row_starts(), frame.tiles.isolated) {
        let stripe = Stripe {
            top: luma_stripe.top / 2,
            bottom: luma_stripe.bottom.div_ceil(2).min(height),
            frame_height: height,
            band_top: luma_stripe.band_top / 2,
            ..luma_stripe
        };
        for left in (0..width).step_by(column) {
            let unit = unit_at(
                maps,
                plane,
                (luma_stripe, unit_columns[left / column], &frame.tiles),
                unit_log2,
            );
            let Some(unit) = unit.filter(|unit| unit.kind == kind::NS_WIENER) else {
                continue;
            };
            let taps: &[i8; CHROMA_TAPS] = if wiener.frame_filters {
                &wiener.filters[0]
            } else {
                &unit.filters[0]
            };
            let w = (width - left).min(column);
            // The chroma columns of the tile (4:2:0).
            let (start, end) = frame.tiles.column_span(left * 2);
            let span = (start / 2, end.div_ceil(2).min(width));
            fill_window(
                &mut window,
                stride,
                (source, deblocked),
                stripe,
                (left, w),
                span,
            );
            fill_luma_window(
                &mut luma_window,
                stride,
                luma,
                stripe,
                (left, w, span),
                frame.downsample,
            );
            let view = Window {
                samples: &window,
                stride,
            };
            let luma_view = Window {
                samples: &luma_window,
                stride,
            };
            let out_stride = out.stride();
            let data = out.data_mut();
            for y in 0..stripe.height() {
                for x in 0..w {
                    data[(stripe.top + y) * out_stride + left + x] =
                        wiener::chroma_sample(&view, &luma_view, taps, (x, y), depth);
                }
            }
        }
    }
}

/// The co-located luma of a chroma stripe's column, downsampled to chroma
/// resolution (dav2d `backup_row_luma`, `wiener_uv_luma_c`'s rows): in
/// the stripe from `source`; next to it from the deblocked rows, the
/// second row out pairing the nearest deblocked row with itself; at the
/// frame's top and bottom (and isolated tiles') the stripe's edge row
/// repeats. Chroma columns past `span` (the plane, or its tile) repeat the
/// downsampled edge column.
fn fill_luma_window<T: Sample>(
    window: &mut [T],
    stride: usize,
    (source, deblocked): (&PlaneView<'_, T>, &PlaneView<'_, T>),
    stripe: Stripe,
    (left, width, (span_start, span_end)): (usize, usize, (usize, usize)),
    downsample: u8,
) {
    let rows = stripe.height() + 2 * WINDOW_BORDER;
    let height = stripe.height() as isize;
    for (j, out) in window.chunks_exact_mut(stride).take(rows).enumerate() {
        let offset = j as isize - WINDOW_BORDER as isize;
        // The two luma rows of this chroma row, and which frame they
        // come from.
        let (pair, plane) = if (0..height).contains(&offset)
            || (offset < 0 && stripe.above == Boundary::TileRow)
            || (offset >= height && stripe.below == Boundary::TileRow)
        {
            let y = 2 * stripe.top.wrapping_add_signed(offset);
            ((y, y + 1), source)
        } else if offset < 0 {
            if stripe.above == Boundary::Edge {
                ((2 * stripe.top, 2 * stripe.top + 1), source)
            } else {
                let top = 2 * stripe.top;
                // Row -1 pairs deblocked rows -2 and -1; further rows
                // repeat row -2.
                let pair = if offset == -1 {
                    (top - 2, top - 1)
                } else {
                    (top - 2, top - 2)
                };
                (pair, deblocked)
            }
        } else if stripe.below == Boundary::Edge {
            let y = 2 * (stripe.bottom - 1);
            ((y, y + 1), source)
        } else {
            let bottom = 2 * stripe.bottom;
            let pair = if offset == height {
                (bottom, bottom + 1)
            } else {
                (bottom + 1, bottom + 1)
            };
            (pair, deblocked)
        };
        let (upper, lower) = (plane.row(pair.0), plane.row(pair.1.min(plane.height() - 1)));
        for (i, sample) in out[..width + 2 * WINDOW_BORDER].iter_mut().enumerate() {
            let x = (left + i)
                .saturating_sub(WINDOW_BORDER)
                .clamp(span_start, span_end - 1);
            *sample = downsampled(upper, lower, 2 * x, downsample);
        }
    }
}

/// One downsampled luma sample from columns `x`, `x + 1` of two rows
/// (dav2d `backup_row_luma`'s `cfl_ds_flt` cases).
fn downsampled<T: Sample>(upper: &[T], lower: &[T], x: usize, filter: u8) -> T {
    let (a, b) = (upper[x].to_i32(), lower[x].to_i32());
    let right = (x + 1).min(upper.len() - 1);
    match filter {
        0 => T::from_i32((a + b + upper[right].to_i32() + lower[right].to_i32()) >> 2),
        1 => T::from_i32((a + b) >> 1),
        _ => upper[x],
    }
}
