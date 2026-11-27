//! The guided detail filter of the general decoder (dav2d `lr_apply_tmpl.c`
//! `lr_stripe`'s GDF part, `gdf_prep_c`'s input rules): luma only, in the
//! loop-restoration stripes, block by 64x64 block where the block layer's
//! flag is set, with nc-dsp's [`gdf_error_with_dsp`] and [`gdf_add`].
//!
//! Each block's window is the frame after CDEF and CCSO, except outside
//! its stripe: there the two rows next to the stripe come from the
//! deblocked frame (dav2d's `lr_db_line`), further rows repeat the second
//! of them, and at the frame's top and bottom the stripe's own edge row
//! repeats. Columns outside the frame repeat its edge columns.

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::Dsp;
use nc_dsp::av2_dec::gdf::{
    GDF_BORDER, GDF_ERROR_STRIDE, GDF_INPUT_BITS, GDF_MAX_SIDE, GdfParams, GdfWindow,
    gdf_add_with_dsp, gdf_error_with_dsp,
};
use nc_dsp::av2_dec::gdf_tables::{INTER, INTRA};

use super::FilterMaps;
use super::stripes::{STRIPE_CONTEXT_ROWS, Stripe, stripes};
use crate::general::frame_header::FrameHeader;

/// Rows below a stripe's top whose filter block decides its flag (dav2d
/// `inc`: the stripe starts 8 rows above its 64-row band).
const FLAG_ROW_OFFSET: usize = 8;
/// Reference distance class by the larger distance of the first two
/// references (dav2d `ref_dst_idx_tbl`).
const DISTANCE_CLASS: [usize; 12] = [4, 0, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4];
/// References whose distances choose the inter parameters.
const DISTANCE_REFERENCES: usize = 2;

/// The frame's parameter set: intra frames' by qp class, inter frames' by
/// the farther of their first two references (`distances`, display
/// order differences of each reference rank) too.
#[must_use]
pub fn params(header: &FrameHeader, distances: &[u32]) -> &'static GdfParams {
    let qp = usize::from(header.gdf.qp_index);
    if header.is_intra() {
        &INTRA[qp]
    } else {
        let farthest = distances
            .iter()
            .take(DISTANCE_REFERENCES)
            .copied()
            .max()
            .unwrap_or(0);
        let class = DISTANCE_CLASS[(farthest as usize).min(DISTANCE_CLASS.len() - 1)];
        &INTER[class][qp]
    }
}

/// Correct the luma of `picture` (after the Wiener filters, if any) where
/// the block layer set GDF flags, from `source` (the frame after CDEF and
/// CCSO, before restoration) and `deblocked` (the frame before CDEF).
pub fn filter(
    dsp: &Dsp,
    picture: &mut Picture,
    (source, deblocked): (&Picture, &Picture),
    maps: &FilterMaps,
    (header, params): (&FrameHeader, &GdfParams),
) {
    let source = source.plane(PlaneIndex::Y).view();
    let deblocked = deblocked.plane(PlaneIndex::Y).view();
    let (width, height) = (source.width(), source.height());
    let scale = i32::from(header.gdf.scale);
    let stride = GDF_MAX_SIDE + 2 * GDF_BORDER;
    let mut window = vec![0_u16; stride * stride];
    let mut error = vec![0_i8; GDF_ERROR_STRIDE * GDF_MAX_SIDE];
    let mut output = picture.plane_mut(PlaneIndex::Y).view_mut();
    for stripe in stripes(height) {
        let flag_row = if stripe.top != 0 && stripe.top + FLAG_ROW_OFFSET < height {
            stripe.top + FLAG_ROW_OFFSET
        } else {
            stripe.top
        };
        for left in (0..width).step_by(GDF_MAX_SIDE) {
            let block = maps.filter_block((flag_row >> 2) as u32, (left >> 2) as u32);
            if !maps.gdf_flags[block] {
                continue;
            }
            let w = (width - left).min(GDF_MAX_SIDE);
            fill_window(
                &mut window,
                stride,
                (&source, &deblocked),
                stripe,
                (left, w),
            );
            let view = GdfWindow {
                samples: &window,
                stride,
                width: w,
                height: stripe.height(),
            };
            gdf_error_with_dsp(dsp, &view, params, &mut error);
            let out_stride = output.stride();
            let start = stripe.top * out_stride + left;
            gdf_add_with_dsp(
                dsp,
                &mut output.data_mut()[start..],
                out_stride,
                &error,
                (w, stripe.height()),
                scale,
            );
        }
    }
}

/// The window of the block of `stripe` at columns `left..left + width`,
/// normalised to 10 bits.
fn fill_window(
    window: &mut [u16],
    stride: usize,
    (source, deblocked): (&PlaneView<'_>, &PlaneView<'_>),
    stripe: Stripe,
    (left, width): (usize, usize),
) {
    let up = GDF_INPUT_BITS - 8;
    let frame_width = source.width();
    let first = left.saturating_sub(GDF_BORDER);
    let pre = GDF_BORDER.saturating_sub(left);
    let end = (left + width + GDF_BORDER).min(frame_width);
    let middle = end - first;
    let row_width = width + 2 * GDF_BORDER;
    let rows = stripe.height() + 2 * GDF_BORDER;
    for (j, out) in window.chunks_exact_mut(stride).take(rows).enumerate() {
        let offset = j as isize - GDF_BORDER as isize;
        let row = match stripe.context_row(offset, STRIPE_CONTEXT_ROWS) {
            (y, false) => source.row(y),
            (y, true) => deblocked.row(y),
        };
        let (prefix, out) = out[..row_width].split_at_mut(pre);
        let (pixels, suffix) = out.split_at_mut(middle);
        prefix.fill(u16::from(row[0]) << up);
        for (dst, &src) in pixels.iter_mut().zip(&row[first..end]) {
            *dst = u16::from(src) << up;
        }
        suffix.fill(u16::from(row[frame_width - 1]) << up);
    }
}
