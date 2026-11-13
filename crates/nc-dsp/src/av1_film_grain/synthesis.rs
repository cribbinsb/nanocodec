//! Film-grain stripe scheduling over shared row kernels, without full-frame noise buffers.
use super::{
    AR_PADDING, Av1FilmGrainParams, Av1GrainScaling, BLOCK, OVERLAP, RESTRICTED_MAX_CHROMA,
    RESTRICTED_MAX_LUMA, RESTRICTED_MIN, Random, Template, blend_overlap, scaling_lut, stripe_seed,
    templates,
};
use crate::Dsp;
use nc_common::{PlaneView, PlaneViewMut};

impl Dsp {
    /// Apply normative AV1 8-bit 4:2:0 grain to shown output, leaving references untouched.
    /// Templates and intensity tables are prepared once for the frame's parameters/seed.
    /// # Panics
    /// If any plane is smaller than the requested frame dimensions.
    pub fn av1_add_film_grain(
        &self,
        params: &Av1FilmGrainParams,
        input: [&PlaneView<'_>; 3],
        size: (usize, usize),
        output: [&mut PlaneViewMut<'_>; 3],
    ) {
        let templates = templates(params);
        let shift = params.grain_scaling_minus_8 + 8;
        let luma_lut = scaling_lut(&params.y_points);
        let luma = Av1GrainScaling::new(&luma_lut, shift);
        let chroma = if params.chroma_scaling_from_luma {
            [luma.clone(), luma.clone()]
        } else {
            [
                Av1GrainScaling::new(&scaling_lut(&params.cb_points), shift),
                Av1GrainScaling::new(&scaling_lut(&params.cr_points), shift),
            ]
        };
        for (plane, destination) in output.into_iter().enumerate() {
            let table = if plane == 0 {
                &luma
            } else {
                &chroma[plane - 1]
            };
            synthesize_plane(
                self,
                params,
                &templates[plane],
                plane,
                (input, destination),
                table,
                size,
            );
        }
    }
}

fn synthesize_plane(
    dsp: &Dsp,
    params: &Av1FilmGrainParams,
    template: &Template,
    plane: usize,
    (input, output): ([&PlaneView<'_>; 3], &mut PlaneViewMut<'_>),
    scaling: &Av1GrainScaling,
    (width, height): (usize, usize),
) {
    let sub = usize::from(plane != 0);
    let (plane_w, plane_h) = ((width + sub) >> sub, (height + sub) >> sub);
    assert!(input[plane].width() >= plane_w && input[plane].height() >= plane_h);
    assert!(output.width() >= plane_w && output.height() >= plane_h);
    let points = [
        params.y_points.count,
        params.cb_points.count,
        params.cr_points.count,
    ][plane];
    if points == 0 && !(plane != 0 && params.chroma_scaling_from_luma) {
        for y in 0..plane_h {
            output.row_mut(y)[..plane_w].copy_from_slice(&input[plane].row(y)[..plane_w]);
        }
        return;
    }
    let block = BLOCK >> sub;
    let reach = OVERLAP >> sub;
    let stride = plane_w.div_ceil(block) * block + reach;
    let mut current = vec![0i16; (block + reach) * stride];
    let mut previous = vec![0i16; current.len()];
    let bounds = if params.clip_to_restricted_range {
        [
            RESTRICTED_MIN as u8,
            if plane == 0 {
                RESTRICTED_MAX_LUMA
            } else {
                RESTRICTED_MAX_CHROMA
            } as u8,
        ]
    } else {
        [0, 255]
    };
    let coefficients = if params.chroma_scaling_from_luma {
        [0, 64, 0]
    } else {
        let (mult, luma_mult, offset) = if plane == 1 {
            params.cb_mult
        } else {
            params.cr_mult
        };
        [
            i16::from(mult) - 128,
            i16::from(luma_mult) - 128,
            offset as i16 - 256,
        ]
    };
    for (stripe_index, first_row) in (0..plane_h).step_by(block).enumerate() {
        fill_stripe(params, template, sub, stripe_index, stride, &mut current);
        for y in 0..block.min(plane_h - first_row) {
            let noise = &mut current[y * stride..y * stride + plane_w];
            if params.overlap && stripe_index != 0 && y < reach {
                let old = &previous[(block + y) * stride..(block + y) * stride + plane_w];
                dsp.av1_grain_blend_row(old, noise, weights(sub, y));
            }
            let source = &input[plane].row(first_row + y)[..plane_w];
            let destination = &mut output.row_mut(first_row + y)[..plane_w];
            if plane == 0 {
                dsp.av1_grain_luma_row(source, noise, scaling, bounds, destination);
            } else {
                let luma = &input[0].row((first_row + y) * 2)[..width];
                dsp.av1_grain_chroma_row(
                    source,
                    luma,
                    noise,
                    scaling,
                    bounds,
                    coefficients,
                    destination,
                );
            }
        }
        // Only the final overlap rows survive into the next stripe. Reusing both
        // buffers avoids frame-sized scratch and allocation per block row.
        std::mem::swap(&mut current, &mut previous);
    }
}

fn weights(sub: usize, index: usize) -> [i16; 2] {
    match (sub, index) {
        (1, _) => [23, 22],
        (_, 0) => [27, 17],
        _ => [17, 27],
    }
}

fn fill_stripe(
    params: &Av1FilmGrainParams,
    template: &Template,
    sub: usize,
    index: usize,
    stride: usize,
    stripe: &mut [i16],
) {
    let block = BLOCK >> sub;
    let reach = OVERLAP >> sub;
    let mut random = Random(stripe_seed(params.grain_seed, index));
    for x in (0..stride - reach).step_by(block) {
        let offset = random.next(8);
        let base_x = AR_PADDING + (AR_PADDING << (1 - sub)) + ((offset >> 4) << (1 - sub));
        let base_y = AR_PADDING + (AR_PADDING << (1 - sub)) + ((offset & 15) << (1 - sub));
        for y in 0..block + reach {
            let source =
                &template.values[(base_y + y) * template.columns + base_x..][..block + reach];
            let destination = &mut stripe[y * stride + x..][..block + reach];
            let overlap = if params.overlap && x != 0 { reach } else { 0 };
            for j in 0..overlap {
                destination[j] =
                    blend_overlap(i32::from(destination[j]), i32::from(source[j]), j, sub) as i16;
            }
            destination[overlap..].copy_from_slice(&source[overlap..]);
        }
    }
}
