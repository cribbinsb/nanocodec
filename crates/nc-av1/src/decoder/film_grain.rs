//! Film grain synthesis of a shown frame (AV1 §7.18.3) with the
//! parameters `film_grain_params()` parsed (§5.9.30), on top of
//! `nc_dsp::av1_film_grain`. Output only: the grain goes into a separate
//! picture, never into a reference.

use nc_common::decode::DecodeError;
use nc_common::{Picture, Plane, PlaneIndex, PlaneViewMut};
use nc_dsp::av1_film_grain::{Av1FilmGrainParams, Av1ScalingPoints};

use super::frame_header::FilmGrainParams;

/// The synthesis parameters of the parsed syntax.
fn synthesis_params(g: &FilmGrainParams) -> Av1FilmGrainParams {
    Av1FilmGrainParams {
        grain_seed: g.grain_seed,
        y_points: Av1ScalingPoints {
            count: g.num_y_points,
            value: g.point_y_value,
            scaling: g.point_y_scaling,
        },
        chroma_scaling_from_luma: g.chroma_scaling_from_luma,
        cb_points: Av1ScalingPoints {
            count: g.num_cb_points,
            value: g.point_cb_value,
            scaling: g.point_cb_scaling,
        },
        cr_points: Av1ScalingPoints {
            count: g.num_cr_points,
            value: g.point_cr_value,
            scaling: g.point_cr_scaling,
        },
        grain_scaling_minus_8: g.grain_scaling_minus_8,
        ar_coeff_lag: g.ar_coeff_lag,
        ar_coeffs_y_plus_128: g.ar_coeffs_y_plus_128,
        ar_coeffs_cb_plus_128: g.ar_coeffs_cb_plus_128,
        ar_coeffs_cr_plus_128: g.ar_coeffs_cr_plus_128,
        ar_coeff_shift_minus_6: g.ar_coeff_shift_minus_6,
        grain_scale_shift: g.grain_scale_shift,
        cb_mult: (g.cb_mult, g.cb_luma_mult, g.cb_offset),
        cr_mult: (g.cr_mult, g.cr_luma_mult, g.cr_offset),
        overlap: g.overlap_flag,
        clip_to_restricted_range: g.clip_to_restricted_range,
    }
}

/// Width and height of `plane` of a `width` x `height` 4:2:0 frame.
fn plane_size(plane: PlaneIndex, (width, height): (usize, usize)) -> (usize, usize) {
    if plane == PlaneIndex::Y {
        (width, height)
    } else {
        (width.div_ceil(2), height.div_ceil(2))
    }
}

/// The writable top-left `width` x `height` samples of `plane`.
fn writable(plane: &mut Plane, (width, height): (usize, usize)) -> Option<PlaneViewMut<'_>> {
    plane.region_mut(0, 0, width, height)
}

/// Add film grain to the `width` x `height` (upscaled) frame `input`,
/// writing the shown frame to `output`.
pub(crate) fn apply(
    dsp: &nc_dsp::Dsp,
    params: &FilmGrainParams,
    input: &Picture,
    (width, height): (usize, usize),
    output: &mut Picture,
) -> Result<(), DecodeError> {
    let size = |plane: PlaneIndex| plane_size(plane, (width, height));
    let views = PlaneIndex::ALL.map(|plane| {
        let (plane_w, plane_h) = size(plane);
        input.plane(plane).region(0, 0, plane_w, plane_h)
    });
    let [Some(luma), Some(cb), Some(cr)] = views else {
        return Err(DecodeError::Malformed(
            "AV1 film grain input smaller than its frame",
        ));
    };
    let [out_y, out_u, out_v] = output.planes_mut();
    let (Some(mut grain_y), Some(mut grain_u), Some(mut grain_v)) = (
        writable(out_y, size(PlaneIndex::Y)),
        writable(out_u, size(PlaneIndex::U)),
        writable(out_v, size(PlaneIndex::V)),
    ) else {
        return Err(DecodeError::Malformed(
            "AV1 film grain output smaller than its frame",
        ));
    };
    dsp.av1_add_film_grain(
        &synthesis_params(params),
        [&luma, &cb, &cr],
        (width, height),
        [&mut grain_y, &mut grain_u, &mut grain_v],
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without scaling points the frame is copied unchanged.
    #[test]
    fn no_points_copies_the_frame() {
        let mut input = Picture::new(64, 48, 16).unwrap();
        for plane in PlaneIndex::ALL {
            let mut view = input.plane_mut(plane).view_mut();
            for y in 0..view.height() {
                for (x, s) in view.row_mut(y).iter_mut().enumerate() {
                    *s = (x * 5 + y * 3) as u8;
                }
            }
        }
        let mut output = Picture::new(64, 48, 16).unwrap();
        let params = FilmGrainParams {
            apply_grain: true,
            grain_seed: 1234,
            ..FilmGrainParams::default()
        };
        apply(&nc_dsp::Dsp::new(), &params, &input, (64, 48), &mut output).unwrap();
        for plane in PlaneIndex::ALL {
            assert_eq!(
                output.plane(plane).view().data(),
                input.plane(plane).view().data()
            );
        }
    }
}
