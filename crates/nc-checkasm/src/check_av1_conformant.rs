//! AV1 inverse transforms on conformant blocks, in the style of dav1d's
//! checkasm (`tests/checkasm/itx.c`): random residuals in [-255, 255],
//! transformed by a floating-point forward transform of the type, rounded,
//! optionally quantised and dequantised, then cut to a random end of block
//! in the type's scan order (`nc_av1::scan_positions`). Such blocks keep
//! every inverse stage inside §7.13.3's ranges, where every ISA's
//! `Dsp::av1_inverse_add_i16` (dav1d's 16-bit saturating semantics or the
//! guarded libaom-exact kernels) and dav1d's scalar model must all equal
//! libaom's 32-bit scalar arithmetic: across ISAs, every result is the same.
//!
//! The forward transforms follow dav1d 1.5.4 `tests/checkasm/itx.c`
//! (`fdct_1d`, `fadst_1d`, its per-size `scaling_factors`; Copyright (c)
//! 2018, `VideoLAN` and dav1d authors, Two Orioles, LLC; BSD-2-Clause, see
//! `THIRD_PARTY.md`), reimplemented. Unlike dav1d's, the flipped ADST
//! reverses its input, each direction uses its own kernel, and identity
//! carries the 1-D gain of the others (`sqrt(n / 2)`), so every type's
//! coefficients land in the range real streams use.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use crate::{CheckReport, rng::Rng};
use nc_common::PlaneViewMut;
use nc_dsp::{
    Av1TxSize, Av1TxType, Dsp, av1_inverse_add_i16_dav1d_reference, av1_inverse_has_dav1d_model,
};

/// Largest residual magnitude: 8-bit prediction differences.
const RESIDUAL_MAX: usize = 255;
/// Blocks per (size, type), plus one per [`ITERATIONS_PER_BLOCK`]
/// iterations.
const BLOCKS: usize = 32;
const ITERATIONS_PER_BLOCK: usize = 4;
/// Quantiser steps tried: 1 is the unquantised transform.
const STEPS: [i32; 4] = [1, 4, 24, 96];
/// Extra columns right of the block and rows below it, which must not
/// change.
const MARGIN: usize = 3;
/// dav1d's forward scale by `log2(w * h) - 4`: it makes the inverse
/// (with the sizes' row and column shifts) return roughly the residual.
const SCALING_FACTORS: [f64; 9] = [
    4.0,
    4.0 * FRAC_1_SQRT_2,
    2.0,
    2.0 * FRAC_1_SQRT_2,
    1.0,
    0.5 * FRAC_1_SQRT_2,
    0.25,
    0.125 * FRAC_1_SQRT_2,
    0.0625,
];

/// One 1-D forward kernel.
#[derive(Clone, Copy, PartialEq)]
enum Forward {
    Dct,
    Adst,
    FlipAdst,
    Identity,
}

/// The (vertical, horizontal) forward kernels of `kind`.
fn kernels(kind: Av1TxType) -> (Forward, Forward) {
    use Av1TxType::{
        AdstAdst, AdstDct, AdstFlipadst, DctAdst, DctDct, DctFlipadst, FlipadstAdst, FlipadstDct,
        FlipadstFlipadst, HAdst, HDct, HFlipadst, Idtx, VAdst, VDct, VFlipadst,
    };
    use Forward::{Adst, Dct, FlipAdst, Identity};
    match kind {
        DctDct => (Dct, Dct),
        AdstDct => (Adst, Dct),
        DctAdst => (Dct, Adst),
        AdstAdst => (Adst, Adst),
        FlipadstDct => (FlipAdst, Dct),
        DctFlipadst => (Dct, FlipAdst),
        FlipadstFlipadst => (FlipAdst, FlipAdst),
        AdstFlipadst => (Adst, FlipAdst),
        FlipadstAdst => (FlipAdst, Adst),
        Idtx => (Identity, Identity),
        VDct => (Dct, Identity),
        HDct => (Identity, Dct),
        VAdst => (Adst, Identity),
        HAdst => (Identity, Adst),
        VFlipadst => (FlipAdst, Identity),
        HFlipadst => (Identity, FlipAdst),
    }
}

/// The matrix of `kernel` at `n` samples, row-major (`out[i] = sum over j
/// of m[i * n + j] * in[j]`): dav1d's DCT-II with a `1/sqrt(2)` DC row, its
/// ADST (the sine of AV1's ADST4, and of ADST8/16 otherwise), the ADST of
/// the reversed input, or the identity at the DCT's gain.
fn basis(kernel: Forward, n: usize) -> Vec<f64> {
    let entry = |i: usize, j: usize| -> f64 {
        let adst = |j: usize| {
            let phase = if n == 4 {
                ((j + 1) * (2 * i + 1)) as f64 / 9.0
            } else {
                ((2 * j + 1) * (2 * i + 1)) as f64 / (4 * n) as f64
            };
            (PI * phase).sin()
        };
        match kernel {
            Forward::Dct => {
                let dc = if i == 0 { FRAC_1_SQRT_2 } else { 1.0 };
                dc * (PI * ((2 * j + 1) * i) as f64 / (2 * n) as f64).cos()
            }
            Forward::Adst => adst(j),
            Forward::FlipAdst => adst(n - 1 - j),
            Forward::Identity if i == j => (n as f64 / 2.0).sqrt(),
            Forward::Identity => 0.0,
        }
    };
    (0..n * n).map(|k| entry(k / n, k % n)).collect()
}

/// `matrix` (from [`basis`]) applied to `input`.
fn apply(matrix: &[f64], input: &[f64]) -> Vec<f64> {
    matrix
        .chunks_exact(input.len())
        .map(|row| row.iter().zip(input).map(|(m, x)| m * x).sum())
        .collect()
}

/// A conformant block of `kind` at `size`: column-major coefficients of the
/// coded area and its end of block in the type's scan.
fn conformant_block(
    size: Av1TxSize,
    kind: Av1TxType,
    (vertical, horizontal): (&[f64], &[f64]),
    rng: &mut Rng,
) -> (Vec<i32>, usize) {
    let (w, h) = (size.width(), size.height());
    let scale = SCALING_FACTORS[(size.width_log2 + size.height_log2) as usize - 4];
    // Rows, then columns; `spectrum[r * w + c]`.
    let mut spectrum = vec![0.0; w * h];
    for row in spectrum.chunks_exact_mut(w) {
        let residual: Vec<f64> = (0..w)
            .map(|_| rng.below(2 * RESIDUAL_MAX + 1) as f64 - RESIDUAL_MAX as f64)
            .collect();
        for (out, value) in row.iter_mut().zip(apply(horizontal, &residual)) {
            *out = value * scale;
        }
    }
    for c in 0..w {
        let column: Vec<f64> = (0..h).map(|r| spectrum[r * w + c]).collect();
        for (r, value) in apply(vertical, &column).into_iter().enumerate() {
            spectrum[r * w + c] = value;
        }
    }

    let step = STEPS[rng.below(STEPS.len())];
    let (cw, ch) = (size.coded_width(), size.coded_height());
    let mut coeffs = vec![0i32; cw * ch];
    for c in 0..cw {
        for r in 0..ch {
            let level = (spectrum[r * w + c] / f64::from(step)).round() as i32;
            // §7.12.3 clamps dequantised values to 16 bits at 8-bit depth.
            coeffs[c * ch + r] = (level * step).clamp(i32::from(i16::MIN), i32::from(i16::MAX));
        }
    }

    // A random end of block, often small (the drivers' sparse paths):
    // everything from it on in scan order is zero, the last coded one not.
    let scan = nc_av1::scan_positions(size, kind);
    let eob = match rng.below(4) {
        0 => 1,
        1 => 1 + rng.below(scan.len().min(64)),
        _ => 1 + rng.below(scan.len()),
    };
    for &position in &scan[eob..] {
        coeffs[position] = 0;
    }
    let last = &mut coeffs[scan[eob - 1]];
    if *last == 0 {
        *last = step;
    }
    (coeffs, eob)
}

pub(crate) fn check(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    for size in Av1TxSize::ALL {
        let (w, h) = (size.width(), size.height());
        let stride = w + MARGIN;
        for kind in Av1TxType::ALL.into_iter().filter(|kind| kind.fits(size)) {
            let (vertical, horizontal) = kernels(kind);
            let matrices = (&basis(vertical, h)[..], &basis(horizontal, w)[..]);
            for _ in 0..BLOCKS + iterations / ITERATIONS_PER_BLOCK {
                let (coeffs, eob) = conformant_block(size, kind, matrices, rng);
                let mut prediction = vec![0u8; stride * (h + MARGIN)];
                rng.fill(&mut prediction);
                // libaom: the 32-bit scalar reference, over the whole area.
                let mut expected = prediction.clone();
                let mut dst = PlaneViewMut::new(&mut expected, stride, w, h).expect("plane");
                reference.av1_inverse_add(size, kind, &coeffs, size.coded_area(), &mut dst);

                let narrow: Vec<i16> = coeffs.iter().map(|&c| c as i16).collect();
                let mut actual = prediction.clone();
                let mut dst = PlaneViewMut::new(&mut actual, stride, w, h).expect("plane");
                candidate.av1_inverse_add_i16(size, kind, &mut narrow.clone(), eob, &mut dst);
                report.cases += 1;
                if actual != expected {
                    report.failures.push(format!(
                        "av1_inverse_add_i16 conformant {size:?} {kind:?} eob={eob}"
                    ));
                }

                if av1_inverse_has_dav1d_model(size, kind) {
                    let mut model = prediction;
                    let mut dst = PlaneViewMut::new(&mut model, stride, w, h).expect("plane");
                    av1_inverse_add_i16_dav1d_reference(
                        size,
                        kind,
                        &mut narrow.clone(),
                        eob,
                        &mut dst,
                    );
                    report.cases += 1;
                    if model != expected {
                        report.failures.push(format!(
                            "dav1d model conformant {size:?} {kind:?} eob={eob}"
                        ));
                    }
                }
            }
        }
    }
}
