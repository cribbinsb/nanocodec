//! Typed AV2 transforms (every kernel pair at every size it exists) against
//! the scalar reference, on random, extreme and alternating inputs, with
//! guarded outputs.
use crate::{check::CheckReport, rng::Rng};
use nc_common::BitDepth;
use nc_dsp::av2_dec::itx::{self, RectTxSize, TxType};
use nc_dsp::{Av2Kernel, Av2TxSize, Av2TxType, Dsp};

/// Guard value around outputs, to catch out-of-range writes.
const GUARD: i32 = 0x5A5A_5A5A;
/// [`GUARD`] of the 16-bit inverse outputs.
const RESIDUAL_GUARD: i16 = 0x5A5A;
/// Random blocks per (size, type).
const RANDOM_BLOCKS: usize = 4;
/// Largest residual magnitude of 8-bit video (the forward contract).
const MAX_RESIDUAL: i16 = 255;
const KERNELS: [Av2Kernel; 4] = [
    Av2Kernel::Dct,
    Av2Kernel::Identity,
    Av2Kernel::Adst,
    Av2Kernel::FlipAdst,
];

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for size in Av2TxSize::ALL {
        for vertical in KERNELS {
            for horizontal in KERNELS {
                let tx_type = Av2TxType {
                    vertical,
                    horizontal,
                };
                if tx_type.supports(size) && size != Av2TxSize::Tx64 {
                    check_type(candidate, reference, size, tx_type, rng, report);
                }
            }
        }
    }
    check_decoder_square(candidate, reference, rng, report);
}

fn check_type(
    candidate: &Dsp,
    reference: &Dsp,
    size: Av2TxSize,
    tx_type: Av2TxType,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let n = size.side();
    let mut residuals: Vec<Vec<i16>> = (0..RANDOM_BLOCKS)
        .map(|_| {
            (0..n * n)
                .map(|_| i16::from(rng.byte()) - i16::from(rng.byte()))
                .collect()
        })
        .collect();
    residuals.push(vec![MAX_RESIDUAL; n * n]);
    residuals.push(vec![-MAX_RESIDUAL; n * n]);
    residuals.push(
        (0..n * n)
            .map(|i| {
                if (i + i / n).is_multiple_of(2) {
                    MAX_RESIDUAL
                } else {
                    -MAX_RESIDUAL
                }
            })
            .collect(),
    );
    for residual in &residuals {
        let run = |dsp: &Dsp| {
            let mut out = vec![GUARD; n * n + 2];
            dsp.av2_forward_typed(size, tx_type, residual, n, &mut out[1..=n * n]);
            out
        };
        report.cases += 1;
        if run(candidate) != run(reference) {
            report.failures.push(format!(
                "{} av2_forward_typed {n}x{n} {tx_type:?}",
                candidate.isa()
            ));
        }
    }
    let mut coefficients: Vec<Vec<i16>> = (0..RANDOM_BLOCKS)
        .map(|_| (0..n * n).map(|_| rng.next_u64() as i16).collect())
        .collect();
    coefficients.push(vec![i16::MAX; n * n]);
    coefficients.push(vec![i16::MIN; n * n]);
    coefficients.push(
        (0..n * n)
            .map(|i| if i % 2 == 0 { i16::MAX } else { i16::MIN })
            .collect(),
    );
    for coeffs in &coefficients {
        let run = |dsp: &Dsp| {
            let mut out = vec![RESIDUAL_GUARD; n * n + 2];
            dsp.av2_inverse_typed(size, tx_type, coeffs, &mut out[1..=n * n]);
            out
        };
        report.cases += 1;
        if run(candidate) != run(reference) {
            report.failures.push(format!(
                "{} av2_inverse_typed {n}x{n} {tx_type:?}",
                candidate.isa()
            ));
        }
    }
}

/// The decoder uses column order, keeps unused coefficients outside the
/// transform, and clears the input as part of reconstruction.
fn check_decoder_square(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for size in [
        RectTxSize::Tx4x4,
        RectTxSize::Tx8x8,
        RectTxSize::Tx16x16,
        RectTxSize::Tx32x32,
        RectTxSize::Tx64x64,
    ] {
        let side = size.width();
        let coded = size.coded_width();
        let area = coded * coded;
        let stride = side + 7;
        let offset = stride + 1;
        for eob in [0, 1, (area - 1) as u16] {
            let coeffs: Vec<i16> = (0..area + 2).map(|_| rng.next_u64() as i16).collect();
            let pixels: Vec<u8> = (0..(side + 2) * stride).map(|_| rng.byte()).collect();
            let run = |dsp: &Dsp| {
                let mut pixels = pixels.clone();
                let mut coeffs = coeffs.clone();
                itx::inverse_transform_add_with_dsp(
                    dsp,
                    &mut pixels,
                    (offset, stride),
                    &mut coeffs[1..],
                    (size, TxType::DCT_DCT),
                    eob,
                    BitDepth::EIGHT,
                );
                (pixels, coeffs)
            };
            report.cases += 1;
            if run(candidate) != run(reference) {
                report.failures.push(format!(
                    "{} AV2 decoder square {size:?}, eob {eob} and stride {stride}",
                    candidate.isa()
                ));
            }
        }
    }
}
