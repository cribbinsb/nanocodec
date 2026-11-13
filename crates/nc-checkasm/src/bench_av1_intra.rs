//! Benchmarks of AV1 intra prediction: directional, smooth, Paeth and the
//! filter intra patch kernels.

use nc_dsp::{BlockSize, Dsp};

use crate::bench::time;

/// AV1 directional intra, one angle per zone (zone 2 at 129 degrees mixes
/// both edges, at 157 it is mostly the left column), and filter intra.
pub(crate) fn bench_av1_intra(
    dsp: &Dsp,
    src: &[u8],
    push: &mut impl FnMut(&'static str, BlockSize, f64),
) {
    use nc_dsp::intra_prediction::{EdgeExtension, IntraEdges, IntraMode};
    const PICTURE: usize = 3 * 64;
    const ORIGIN: usize = 64;
    let mut plane = nc_common::Plane::new(PICTURE, PICTURE, 0).expect("plane");
    let mut view = plane.view_mut();
    for y in 0..PICTURE {
        for (x, sample) in view.row_mut(y)[..PICTURE].iter_mut().enumerate() {
            *sample = src[(y * PICTURE + x) % src.len()];
        }
    }
    let extension = EdgeExtension {
        above_right: true,
        below_left: true,
    };
    for n in [4, 8, 16, 32, 64] {
        let size = BlockSize::new(n, n).expect("intra sizes are block sizes");
        let edges = IntraEdges::extended(&plane, ORIGIN, ORIGIN, n, extension);
        let mut out = nc_common::Plane::new(n, n, 0).expect("plane");
        let angles = [
            ("av1_intra_angle_z1_51", 51),
            ("av1_intra_angle_z2_129", 129),
            ("av1_intra_angle_z2_157", 157),
            ("av1_intra_angle_z3_197", 197),
        ];
        for (name, angle) in angles {
            push(
                name,
                size,
                time(|| {
                    dsp.av1_intra_predict_angle(&edges, angle, &mut out.view_mut());
                    u32::from(out.view().row(n - 1)[n - 1])
                }),
            );
        }
        let modes = [
            ("av1_intra_smooth", IntraMode::Smooth),
            ("av1_intra_smooth_v", IntraMode::SmoothVertical),
            ("av1_intra_smooth_h", IntraMode::SmoothHorizontal),
            ("av1_intra_paeth", IntraMode::Paeth),
        ];
        for (name, mode) in modes {
            push(
                name,
                size,
                time(|| {
                    dsp.av1_intra_predict(&edges, mode, &mut out.view_mut());
                    u32::from(out.view().row(n - 1)[n - 1])
                }),
            );
        }
        if n <= nc_dsp::intra_prediction::filter::FILTER_INTRA_MAX_SIZE {
            let mode = nc_dsp::intra_prediction::filter::FilterIntraMode::Paeth;
            push(
                "av1_filter_intra",
                size,
                time(|| {
                    dsp.av1_intra_predict_filter(&edges, mode, &mut out.view_mut());
                    u32::from(out.view().row(n - 1)[n - 1])
                }),
            );
        }
    }
}
