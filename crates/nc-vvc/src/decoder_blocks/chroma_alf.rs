//! Chroma ALF and CC-ALF boundary preparation (which samples each output
//! reads: the ALF virtual boundary, picture edges and closed edges through
//! the shared ALF `Region`); sample arithmetic lives in shared DSP.
use super::{boundaries::Closures, filter_syntax::CtuFilters, filters::alf_bank};
use crate::{alf::Region, decoder_headers::Adaptation};
use nc_common::{
    Picture, PlaneIndex,
    decode::{DecodeError, Result},
};
use nc_dsp::{DepthKernels, Pixel, vvc_alf_clip_value};

/// Whole-CTB fallback region when `closures` does not split CTB `index`.
fn whole_region(index: usize, ctu: usize, size: (usize, usize)) -> Region {
    let columns = size.0.div_ceil(ctu);
    let (x, y) = (index % columns * ctu, index / columns * ctu);
    Region {
        rect: (x, y, ctu.min(size.0 - x), ctu.min(size.1 - y)),
        closed: [false; 4],
        corners: [false; 2],
    }
}

pub(super) fn apply<T: Pixel>(
    picture: &mut Picture<T>,
    filters: &[CtuFilters],
    (ctu, closures): (usize, &Closures),
    banks: &[Option<Box<Adaptation>>; 8],
    dsp: DepthKernels<'_>,
    chroma_buf: &mut Vec<T>,
) -> Result<()> {
    let size = (picture.width(), picture.height());
    let side = ctu / 2;
    for plane in 1..3 {
        if !filters
            .iter()
            .any(|f| f.alf[plane].is_some() && f.banks.chroma.is_some())
        {
            continue;
        }
        let (cw, ch) = {
            let view = picture.plane(PlaneIndex::ALL[plane]).view();
            let (cw, ch) = (view.width(), view.height());
            let total = cw * ch;
            if chroma_buf.len() < total {
                chroma_buf.resize(total, T::default());
            }
            for (dst_row, y) in chroma_buf[..total].chunks_exact_mut(cw).zip(0..ch) {
                dst_row.copy_from_slice(view.row(y));
            }
            (cw, ch)
        };
        let input = nc_common::PlaneView::new(&chroma_buf[..cw * ch], cw, cw, ch)
            .expect("copied chroma plane");
        let mut output = picture.plane_mut(PlaneIndex::ALL[plane]).view_mut();
        let mut padded: [Vec<T>; 5] = std::array::from_fn(|_| vec![T::default(); side + 6]);
        for (index, filter) in filters.iter().enumerate() {
            let (Some(alternative), Some(id)) = (filter.alf[plane], filter.banks.chroma) else {
                continue;
            };
            let selected = alf_bank(banks, id)?
                .chroma
                .get(usize::from(alternative))
                .ok_or(DecodeError::Malformed("VVC chroma ALF alternative"))?;
            let limits = selected
                .clip
                .map(|c| vvc_alf_clip_value(usize::from(c), dsp.depth()));
            let sub;
            let single;
            let list: &[Region] = if let Some(parts) = closures.alf_regions(index) {
                sub = parts;
                &sub
            } else {
                single = [whole_region(index, ctu, size)];
                &single
            };
            for &region in list {
                apply_chroma_region(
                    &input,
                    &mut output,
                    region.halved(),
                    side,
                    (&selected.coefficients, limits),
                    &mut padded,
                    dsp,
                );
            }
        }
    }
    cross(picture, filters, (ctu, closures), banks, dsp)
}

#[allow(clippy::too_many_arguments)]
fn apply_chroma_region<T: Pixel>(
    input: &nc_common::PlaneView<'_, T>,
    output: &mut nc_common::PlaneViewMut<'_, T>,
    region: Region,
    side: usize,
    (coefficients, limits): (&[i16; 6], [i16; 6]),
    padded: &mut [Vec<T>; 5],
    dsp: DepthKernels<'_>,
) {
    let (x0, y0, width, height) = region.rect;
    let pad_len = width + 6;
    let x_start = x0 as isize - 3;
    let (y_min, y_max, direct_x) =
        region.direct_span_bounds((input.width(), input.height()), x_start, pad_len);
    let (in_data, in_stride) = (input.data(), input.stride());
    for y in y0..y0 + height {
        let position = y % side;
        let boundary = side - 2;
        let distance = if position < boundary {
            boundary - 1 - position
        } else {
            position - boundary
        };
        let reach = distance.min(2) as isize;
        let shift = if distance == 0 { 10 } else { 7 };
        let yi = y as isize;
        if let Some(ux) = direct_x {
            let row_at = |dy: isize| {
                let sy = (yi + dy.clamp(-reach, reach)).clamp(y_min, y_max) as usize;
                &in_data[sy * in_stride + ux..][..pad_len]
            };
            let (r_m2, r_m1, r_0, r_p1, r_p2) =
                (row_at(-2), row_at(-1), row_at(0), row_at(1), row_at(2));
            dsp.vvc_alf_filter_chroma_row(
                &[r_m2, r_m2, r_m1, r_0, r_p1, r_p2, r_p2],
                *coefficients,
                limits,
                shift,
                &mut output.row_mut(y)[x0..x0 + width],
            );
            continue;
        }
        let [p0, p1, p2, p3, p4] = padded;
        let r_m2 = region.row_span(
            input,
            x_start,
            yi + (-2isize).clamp(-reach, reach),
            &mut p0[..pad_len],
        );
        let r_m1 = region.row_span(
            input,
            x_start,
            yi + (-1isize).clamp(-reach, reach),
            &mut p1[..pad_len],
        );
        let r_0 = region.row_span(input, x_start, yi, &mut p2[..pad_len]);
        let r_p1 = region.row_span(
            input,
            x_start,
            yi + (1isize).clamp(-reach, reach),
            &mut p3[..pad_len],
        );
        let r_p2 = region.row_span(
            input,
            x_start,
            yi + (2isize).clamp(-reach, reach),
            &mut p4[..pad_len],
        );
        dsp.vvc_alf_filter_chroma_row(
            &[r_m2, r_m2, r_m1, r_0, r_p1, r_p2, r_p2],
            *coefficients,
            limits,
            shift,
            &mut output.row_mut(y)[x0..x0 + width],
        );
    }
}

fn cross<T: Pixel>(
    picture: &mut Picture<T>,
    filters: &[CtuFilters],
    (ctu, closures): (usize, &Closures),
    banks: &[Option<Box<Adaptation>>; 8],
    dsp: DepthKernels<'_>,
) -> Result<()> {
    if !filters.iter().any(|f| f.cross != [0; 2]) {
        return Ok(());
    }
    // Called before luma ALF, borrowing the pre-ALF luma required by §8.8.5
    // while mutating U and V directly without cloning the luma plane.
    let [luma, u_plane, v_plane] = picture.planes_mut();
    let input = luma.view();
    let size = (input.width(), input.height());
    let mut padded: [Vec<T>; 4] = std::array::from_fn(|_| vec![T::default(); ctu + 2]);
    for (plane, chroma_plane) in [u_plane, v_plane].into_iter().enumerate() {
        let mut output = chroma_plane.view_mut();
        for (index, filter) in filters.iter().enumerate() {
            let selection = filter.cross[plane];
            let Some(id) = filter.banks.cross[plane].filter(|_| selection != 0) else {
                continue;
            };
            let coefficients = *alf_bank(banks, id)?.cross[plane]
                .get(usize::from(selection - 1))
                .ok_or(DecodeError::Malformed("VVC cross-component ALF filter"))?;
            // The luma regions, read around each chroma sample's luma.
            let sub;
            let single;
            let list: &[Region] = if let Some(parts) = closures.alf_regions(index) {
                sub = parts;
                &sub
            } else {
                single = [whole_region(index, ctu, size)];
                &single
            };
            for &region in list {
                apply_cross_region(
                    &input,
                    &mut output,
                    region,
                    ctu,
                    coefficients,
                    &mut padded,
                    dsp,
                );
            }
        }
    }
    Ok(())
}

fn apply_cross_region<T: Pixel>(
    input: &nc_common::PlaneView<'_, T>,
    output: &mut nc_common::PlaneViewMut<'_, T>,
    region: Region,
    ctu: usize,
    coefficients: [i16; 7],
    padded: &mut [Vec<T>; 4],
    dsp: DepthKernels<'_>,
) {
    let size = (input.width(), input.height());
    let (x0, y0, width, height) = region.halved().rect;
    let pad_len = 2 * width + 2;
    let x_start = 2 * x0 as isize - 1;
    let (y_min, y_max, direct_x) = region.direct_span_bounds(size, x_start, pad_len);
    let (in_data, in_stride) = (input.data(), input.stride());
    for y in y0..y0 + height {
        let position = (2 * y) % ctu;
        let boundary = ctu - 4;
        let offsets = if position == boundary {
            [0, 0, 0, 0]
        } else if position == boundary - 2 {
            [-1, 0, 1, 1]
        } else {
            [-1, 0, 1, 2]
        };
        let base_y = 2 * y as isize;
        if let Some(ux) = direct_x {
            let rows = offsets.map(|dy| {
                let sy = (base_y + dy).clamp(y_min, y_max) as usize;
                &in_data[sy * in_stride + ux..][..pad_len]
            });
            dsp.vvc_alf_cross_row(&rows, coefficients, &mut output.row_mut(y)[x0..x0 + width]);
            continue;
        }
        let [p0, p1, p2, p3] = padded;
        let rows = [
            region.row_span(input, x_start, base_y + offsets[0], &mut p0[..pad_len]),
            region.row_span(input, x_start, base_y + offsets[1], &mut p1[..pad_len]),
            region.row_span(input, x_start, base_y + offsets[2], &mut p2[..pad_len]),
            region.row_span(input, x_start, base_y + offsets[3], &mut p3[..pad_len]),
        ];
        dsp.vvc_alf_cross_row(&rows, coefficients, &mut output.row_mut(y)[x0..x0 + width]);
    }
}
