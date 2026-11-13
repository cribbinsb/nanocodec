//! AV1 warped motion exactness: whole blocks warped by the candidate's and
//! the scalar reference's 8x8 warp kernel (`av1_warp`, `av1_warp_compound`),
//! over random valid models near the shear limits, blocks at and past the
//! plane edges, and extreme sample patterns.

use crate::{check::CheckReport, rng::Rng};
use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::Dsp;
use nc_dsp::av1_mc::warp::{av1_setup_shear, av1_warp, av1_warp_compound};

/// Models drawn per plane pattern.
const MODELS: usize = 40;
/// `1 << WARPEDMODEL_PREC_BITS`.
const MODEL_ONE: i32 = 1 << 16;

/// A random affine model whose shear terms reach the validity limits.
fn draw_model(rng: &mut Rng, (width, height): (usize, usize)) -> [i32; 6] {
    let mut term = |range: i32| rng.below(2 * range as usize + 1) as i32 - range;
    let shear = MODEL_ONE / 8;
    [
        term(width as i32 * MODEL_ONE),
        term(height as i32 * MODEL_ONE),
        MODEL_ONE + term(shear),
        term(shear),
        term(shear),
        MODEL_ONE + term(shear),
    ]
}

fn sample(pattern: usize, i: usize, rng: &mut Rng) -> u8 {
    match pattern {
        0 => rng.byte(),
        1 => (i % 2 * 255) as u8,
        2 => [0, 255, 255, 0, 255, 0, 0, 255][i % 8],
        _ => (i * 7) as u8,
    }
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let (width, height) = (72, 56);
    for pattern in 0..4 {
        let samples: Vec<u8> = (0..width * height)
            .map(|i| sample(pattern, i, rng))
            .collect();
        let plane = PlaneView::new(&samples, width, width, height).unwrap();
        for _ in 0..MODELS {
            let model = draw_model(rng, (width, height));
            let Some(shear) = av1_setup_shear(&model) else {
                continue;
            };
            let sub = rng.below(2) == 1;
            let size = (8 << rng.below(3), 8 << rng.below(3));
            let position = (rng.below(width) & !7, rng.below(height) & !7);
            let warp = |dsp: &Dsp| {
                let mut pixels = vec![0u8; size.0 * size.1];
                let mut view = PlaneViewMut::new(&mut pixels, size.0, size.0, size.1).unwrap();
                av1_warp(
                    dsp,
                    &plane,
                    &model,
                    shear,
                    position,
                    size,
                    (sub, sub),
                    &mut view,
                );
                let mut compound = vec![0i16; size.0 * size.1];
                av1_warp_compound(
                    dsp,
                    &plane,
                    &model,
                    shear,
                    position,
                    size,
                    (sub, sub),
                    &mut compound,
                    size.0,
                );
                (pixels, compound)
            };
            report.cases += 1;
            if warp(candidate) != warp(reference) {
                report.failures.push(format!(
                    "AV1 warp mismatch: model {model:?}, block {size:?} at {position:?}, \
                     pattern {pattern}, sub {sub}"
                ));
            }
        }
    }
}
