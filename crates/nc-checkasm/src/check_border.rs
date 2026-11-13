//! Border stores (`Plane::extend_borders_with` through a [`Dsp`]) against
//! the scalar ones: whole planes of odd sizes and borders, so every span
//! has unaligned ends around its streamed body.
use crate::check::CheckReport;
use crate::rng::Rng;
use nc_common::Plane;
use nc_dsp::Dsp;

/// Largest plane side tried.
const MAX_SIDE: usize = 70;
/// Largest border tried (more than one `ROW_ALIGNMENT` step).
const MAX_BORDER: usize = 140;

pub(crate) fn check(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for _ in 0..iterations {
        let (width, height) = (1 + rng.below(MAX_SIDE), 1 + rng.below(MAX_SIDE));
        let border = rng.below(MAX_BORDER + 1);
        let mut plane = Plane::new(width, height, border).expect("valid plane");
        {
            let mut view = plane.view_mut();
            for y in 0..height {
                view.row_mut(y).iter_mut().for_each(|s| *s = rng.byte());
            }
        }
        let extend = |dsp: &Dsp| {
            let mut extended = plane.clone();
            extended.extend_borders_with(dsp);
            extended.padded_view().data().to_vec()
        };
        report.cases += 1;
        if extend(candidate) != extend(reference) {
            report.failures.push(format!(
                "{} extend_borders {width}x{height} border {border}: planes differ",
                candidate.isa()
            ));
        }
    }
}
