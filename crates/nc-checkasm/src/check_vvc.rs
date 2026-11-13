//! VVC DSP checks: independent quant formulas, prepared intra edges and all
//! short/long deblocking shapes, including clipping and CTU chroma restrictions.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{Dsp, VvcIntraMode, VvcQuantizer};

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    quant(candidate, rng, report);
    rectangular_forward(candidate, reference, report);
    crate::check_vvc_refinement::check(candidate, reference, rng, report);
    crate::check_vvc_inverse::check(candidate, reference, rng, report);
    crate::check_vvc_mts::check(candidate, reference, rng, report);
    crate::check_vvc_tx64::check(candidate, reference, rng, report);
    intra(candidate, reference, rng, report);
    deblock(candidate, reference, rng, report);
    crate::check_vvc_deblock_edge::check(candidate, reference, rng, report);
    crate::check_vvc_chroma_edge::check(candidate, reference, rng, report);
    crate::check_vvc_alf::check(candidate, reference, rng, report);
    crate::check_sub_blocks::check(candidate, reference, rng, report);
}
fn quant(candidate: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for log2 in 2..=5 {
        for qp in 0..=63u8 {
            let q = VvcQuantizer::new(qp, log2);
            for len in [1, 7, 16, 65, 1024] {
                let raw: Vec<i32> = (0..len)
                    .map(|i| match i % 5 {
                        0 => -32768,
                        1 => 32767,
                        2 => 0,
                        _ => rng.below(65536) as i32 - 32768,
                    })
                    .collect();
                let shift = 21 + u32::from(qp / 6) - u32::from(log2);
                let scale = [26214i64, 23302, 20560, 18396, 16384, 14564][usize::from(qp % 6)];
                let expected: Vec<i32> = raw
                    .iter()
                    .map(|&v| {
                        let a = ((i64::from(v.abs()) * scale + (1 << (shift - 1))) >> shift)
                            .min(32767) as i32;
                        if v < 0 { -a } else { a }
                    })
                    .collect();
                let mut actual = raw.clone();
                let nonzero = candidate.vvc_quantize(&mut actual, &q);
                report.cases += 1;
                if actual != expected
                    || nonzero as usize != expected.iter().filter(|&&v| v != 0).count()
                {
                    report.failures.push(format!(
                        "{} vvc_quant qp={qp} log2={log2} len={len}",
                        candidate.isa()
                    ));
                }
                // Use the entire signed level range, not only quantizer output:
                // decoder inputs must saturate correctly even at QP 63.
                let shift = i32::from(log2) - 1 - i32::from(qp / 6);
                let scale = [40i64, 45, 51, 57, 64, 72][usize::from(qp % 6)];
                let expected: Vec<i32> = raw
                    .iter()
                    .map(|&v| {
                        let product = i64::from(v) * scale;
                        let v = if shift > 0 {
                            (product + (1 << (shift - 1))) >> shift
                        } else {
                            product << -shift
                        };
                        v.clamp(-32768, 32767) as i32
                    })
                    .collect();
                let mut actual = vec![0; len];
                candidate.vvc_dequantize(&raw, &q, &mut actual);
                report.cases += 1;
                if actual != expected {
                    report.failures.push(format!(
                        "{} vvc_dequant qp={qp} log2={log2} len={len}",
                        candidate.isa()
                    ));
                }
            }
        }
    }
}
fn intra(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for side in [4, 8, 16, 32] {
        for pattern in 0..32 {
            let mut sample = || match pattern {
                0 => 0,
                1 => 255,
                2 => {
                    if rng.below(2) == 0 {
                        0
                    } else {
                        255
                    }
                }
                _ => i32::from(rng.byte()),
            };
            let top: Vec<_> = (0..=side).map(|_| sample()).collect();
            let left: Vec<_> = (0..=side).map(|_| sample()).collect();
            let corner = sample();
            for mode in [
                VvcIntraMode::Planar,
                VvcIntraMode::Dc,
                VvcIntraMode::Horizontal,
                VvcIntraMode::Vertical,
            ] {
                // Guard bytes catch writes past the requested block.
                let mut expected = vec![93; side * side + 3];
                let mut actual = expected.clone();
                reference.vvc_intra_predict(side, mode, &top, &left, corner, &mut expected[1..]);
                candidate.vvc_intra_predict(side, mode, &top, &left, corner, &mut actual[1..]);
                report.cases += 1;
                if actual != expected {
                    report.failures.push(format!(
                        "{} vvc_intra {side} {mode:?} pattern={pattern}",
                        candidate.isa()
                    ));
                }
            }
        }
    }
}
fn deblock(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for beta in [0, 6, 16, 32, 64, 88] {
        for tc in [0, 1, 2, 4, 16, 50, 100] {
            for pattern in 0..48 {
                let mut sample = |x: usize, y: usize, half: usize| match pattern {
                    0 => 0,
                    1 => 255,
                    2..=15 => (32 + pattern + y + usize::from(x >= half) * (pattern - 2)) as u8,
                    16..=31 => {
                        (128i32 + (x as i32 - half as i32) * ((pattern - 16) as i32) + y as i32)
                            .clamp(0, 255) as u8
                    }
                    // Smooth decision endpoints with unrelated middle lines
                    // exercise per-line weak-filter masks and strong clipping.
                    32..=39 if y == 0 || y == 3 => (100 + usize::from(x >= half)) as u8,
                    _ => rng.byte(),
                };
                let edge: [[u8; 16]; 4] =
                    std::array::from_fn(|y| std::array::from_fn(|x| sample(x, y, 8)));
                let chroma: [[u8; 8]; 2] =
                    std::array::from_fn(|y| std::array::from_fn(|x| sample(x, y, 4)));
                for lengths in [[3, 3], [3, 7], [7, 3], [7, 7]] {
                    let (mut expected, mut actual) = (edge, edge);
                    reference.vvc_deblock_luma(&mut expected, beta, tc, lengths);
                    candidate.vvc_deblock_luma(&mut actual, beta, tc, lengths);
                    report.cases += 1;
                    if actual != expected {
                        report.failures.push(format!("{} vvc_deblock_luma beta={beta} tc={tc} lengths={lengths:?} pattern={pattern}",candidate.isa()));
                    }
                }
                for large in [false, true] {
                    for boundary in [false, true] {
                        let (mut expected, mut actual) = (chroma, chroma);
                        reference.vvc_deblock_chroma(&mut expected, beta, tc, large, boundary);
                        candidate.vvc_deblock_chroma(&mut actual, beta, tc, large, boundary);
                        report.cases += 1;
                        if actual != expected {
                            report.failures.push(format!("{} vvc_deblock_chroma beta={beta} tc={tc} large={large} boundary={boundary} pattern={pattern}",candidate.isa()));
                        }
                    }
                }
            }
        }
    }
}

/// Every supported shape, including narrow scalar fallbacks, exact-length
/// input and guarded output. Only the public 8-bit residual range is used.
fn rectangular_forward(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    const GUARD: i32 = 0x1357_2468;
    for width in [1, 2, 4, 8, 16, 32] {
        for height in [1, 2, 4, 8, 16, 32] {
            let area = width * height;
            if area < 16 {
                continue;
            }
            for pattern in 0..11 {
                let mut input = vec![0i16; area + 1];
                for (i, sample) in input[1..].iter_mut().enumerate() {
                    *sample = match pattern {
                        0 => 0,
                        1 => 255,
                        2 => -255,
                        3 => {
                            if (i + i / width) % 2 == 0 {
                                255
                            } else {
                                -255
                            }
                        }
                        4 => ((i * 37 + i / width * 101) % 511) as i16 - 255,
                        5 => {
                            if i == 0 {
                                255
                            } else {
                                0
                            }
                        }
                        6 => {
                            if i + 1 == area {
                                -255
                            } else {
                                0
                            }
                        }
                        // Unit impulses and odd signed sums expose both
                        // rounding signs in the short second transform.
                        7 => i16::from(i == 0),
                        8 => -i16::from(i + 1 == area),
                        9 => (i % 3) as i16 - 1,
                        _ => {
                            if (i / width).is_multiple_of(2) {
                                1
                            } else {
                                -1
                            }
                        }
                    };
                }
                let run = |dsp: &Dsp| {
                    let mut out = vec![GUARD; area + 2];
                    dsp.vvc_forward_dct(width, height, &input[1..], &mut out[1..=area]);
                    out
                };
                crate::check_codec::record(
                    report,
                    candidate.isa(),
                    run(candidate) == run(reference),
                    || format!("VVC rectangular forward {width}x{height} pattern {pattern}"),
                );
            }
        }
    }
}
