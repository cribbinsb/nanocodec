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

use nc_common::{BitDepth, Picture, PlaneIndex, PlaneView, PlaneViewMut};
use nc_dsp::av2_dec::wiener::{self, CHROMA_TAPS, ClassModel, LumaFilter, WINDOW_BORDER, Window};
use nc_dsp::av2_dec::wiener_tables::{
    FEATURE_CLASS, FRAME_FILTER_OF_CLASS, PRETRAINED_FILTER_OF_CLASS, PRETRAINED_FILTERS,
};

use super::stripes::{STRIPE_CONTEXT_ROWS, Stripe, stripes};
use super::{FilterMaps, RestorationUnit};
use crate::general::block::dequantizer_step;
use crate::general::frame_header::FrameHeader;
use crate::general::frame_header::restoration::{NsWiener, kind};

/// Width of a restoration column (dav2d `lr_sbrow`'s 64-wide steps).
const COLUMN: usize = 64;
/// A stripe's band starts this many rows below the stripe (all but the
/// first stripe start 8 rows above their 64-row band).
const BAND_OFFSET: usize = 8;
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
}

/// Restore `picture` from `source` (the frame after CDEF and CCSO) and
/// `deblocked`, per the block layer's restoration units in `maps`.
pub fn filter(
    picture: &mut Picture,
    (source, deblocked): (&Picture, &Picture),
    maps: &FilterMaps,
    (header, downsample): (&FrameHeader, u8),
) {
    let qindex = header.quantizer.base.min(255);
    let quantizer_class = QUANTIZER_CLASS_LIMITS
        .iter()
        .take_while(|&&limit| qindex >= limit)
        .count();
    let frame = Frame {
        header,
        quantizer_class,
        model: ClassModel::new(dequantizer_step(i32::from(qindex)) as i32),
        downsample,
    };
    let luma_source = source.plane(PlaneIndex::Y).view();
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

/// The unit's luma position covering a 64-row band starting at luma row
/// `band_top` and a column starting at luma column `left` of a plane
/// subsampled by `ss`, units `1 << unit_log2` samples of that plane (dav2d
/// `lr_sbrow`'s `aligned_unit_pos_*` and `next_iter_start_x`): the
/// last unit of a row or column absorbs a remainder below half a unit.
fn unit_position(
    (band_top, left): (usize, usize),
    (width, height): (usize, usize),
    unit_log2: u8,
    ss: usize,
) -> (usize, usize) {
    let unit = 1_usize << unit_log2;
    let align = |position: usize, extent: usize| {
        let aligned = position & !(unit - 1);
        if aligned != 0 && aligned + unit / 2 > extent {
            aligned - unit
        } else {
            aligned
        }
    };
    let y = align(band_top >> ss, height >> ss) << ss;
    let x = align(left >> ss, width >> ss) << ss;
    (x, y)
}

/// The unit of plane `plane` covering a stripe's column.
fn unit_at(
    maps: &FilterMaps,
    plane: usize,
    (stripe, left): (Stripe, usize),
    (width, unit_log2): (usize, u8),
) -> Option<&RestorationUnit> {
    let band_top = if stripe.top == 0 {
        0
    } else {
        stripe.top + BAND_OFFSET
    };
    let ss = usize::from(plane > 0);
    let (x, y) = unit_position(
        (band_top, left),
        (width, stripe.frame_height),
        unit_log2,
        ss,
    );
    let block = maps.filter_block((y >> UNIT_LOG2) as u32, (x >> UNIT_LOG2) as u32);
    maps.restoration[plane][block].as_ref()
}

/// Fill a window around the block at columns `left..left + width` of
/// `stripe` from `source` (in the stripe) and `deblocked` (next to it);
/// columns past the plane repeat its edge columns.
fn fill_window(
    window: &mut [u8],
    stride: usize,
    (source, deblocked): (&PlaneView<'_>, &PlaneView<'_>),
    stripe: Stripe,
    (left, width): (usize, usize),
) {
    let plane_width = source.width();
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
                .min(plane_width - 1);
            *sample = row[x];
        }
    }
}

/// Luma's restoration, stripe by stripe and column by column.
fn restore_luma(
    out: &mut PlaneViewMut<'_>,
    (source, deblocked): (&PlaneView<'_>, &PlaneView<'_>),
    maps: &FilterMaps,
    frame: &Frame<'_>,
) {
    let (width, height) = (source.width(), source.height());
    let restoration = &frame.header.restoration;
    let wiener = &restoration.planes[0].wiener;
    let stride = COLUMN + 2 * WINDOW_BORDER;
    let mut window = vec![0_u8; stride * stride];
    for stripe in stripes(height) {
        for left in (0..width).step_by(COLUMN) {
            let unit_log2 = restoration.unit_log2[0];
            let Some(unit) = unit_at(maps, 0, (stripe, left), (width, unit_log2)) else {
                continue;
            };
            if unit.kind == kind::NONE {
                continue;
            }
            let w = (width - left).min(COLUMN);
            fill_window(&mut window, stride, (source, deblocked), stripe, (left, w));
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
                        let index = frame
                            .model
                            .feature_index(&view, (x, y), skip, BitDepth::EIGHT);
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
                        BitDepth::EIGHT,
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
fn restore_chroma(
    out: &mut PlaneViewMut<'_>,
    (source, deblocked): (&PlaneView<'_>, &PlaneView<'_>),
    luma: (&PlaneView<'_>, &PlaneView<'_>),
    (maps, plane): (&FilterMaps, usize),
    frame: &Frame<'_>,
) {
    let (width, height) = (source.width(), source.height());
    let (luma_width, luma_height) = (luma.0.width(), luma.0.height());
    let restoration = &frame.header.restoration;
    let wiener = &restoration.planes[plane].wiener;
    let column = COLUMN / 2;
    let stride = column + 2 * WINDOW_BORDER;
    let mut window = vec![0_u8; stride * (COLUMN / 2 + 2 * WINDOW_BORDER)];
    let mut luma_window = vec![0_u8; stride * (COLUMN / 2 + 2 * WINDOW_BORDER)];
    for luma_stripe in stripes(luma_height) {
        let stripe = Stripe {
            top: luma_stripe.top / 2,
            bottom: luma_stripe.bottom.div_ceil(2).min(height),
            frame_height: height,
        };
        for left in (0..width).step_by(column) {
            let unit_log2 = restoration.unit_log2[1];
            let unit = unit_at(
                maps,
                plane,
                (luma_stripe, left * 2),
                (luma_width, unit_log2),
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
            fill_window(&mut window, stride, (source, deblocked), stripe, (left, w));
            fill_luma_window(
                &mut luma_window,
                stride,
                luma,
                stripe,
                (left, w),
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
                        wiener::chroma_sample(&view, &luma_view, taps, (x, y), BitDepth::EIGHT);
                }
            }
        }
    }
}

/// The co-located luma of a chroma stripe's column, downsampled to chroma
/// resolution (dav2d `backup_row_luma`, `wiener_uv_luma_c`'s rows): in
/// the stripe from `source`; next to it from the deblocked rows, the
/// second row out pairing the nearest deblocked row with itself; at the
/// frame's top and bottom the stripe's edge row repeats. Columns past the
/// plane repeat the downsampled edge column.
fn fill_luma_window(
    window: &mut [u8],
    stride: usize,
    (source, deblocked): (&PlaneView<'_>, &PlaneView<'_>),
    stripe: Stripe,
    (left, width): (usize, usize),
    downsample: u8,
) {
    let chroma_width = source.width().div_ceil(2);
    let rows = stripe.height() + 2 * WINDOW_BORDER;
    let height = stripe.height() as isize;
    for (j, out) in window.chunks_exact_mut(stride).take(rows).enumerate() {
        let offset = j as isize - WINDOW_BORDER as isize;
        // The two luma rows of this chroma row, and which frame they
        // come from.
        let (pair, plane) = if (0..height).contains(&offset) {
            let y = 2 * (stripe.top + offset as usize);
            ((y, y + 1), source)
        } else if offset < 0 {
            if stripe.top == 0 {
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
        } else if stripe.bottom == stripe.frame_height {
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
                .min(chroma_width - 1);
            *sample = downsampled(upper, lower, 2 * x, downsample);
        }
    }
}

/// One downsampled luma sample from columns `x`, `x + 1` of two rows
/// (dav2d `backup_row_luma`'s `cfl_ds_flt` cases).
fn downsampled(upper: &[u8], lower: &[u8], x: usize, filter: u8) -> u8 {
    let (a, b) = (u16::from(upper[x]), u16::from(lower[x]));
    let right = (x + 1).min(upper.len() - 1);
    match filter {
        0 => ((a + b + u16::from(upper[right]) + u16::from(lower[right])) >> 2) as u8,
        1 => u16::midpoint(a, b) as u8,
        _ => upper[x],
    }
}
