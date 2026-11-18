//! VVC ALF kernels against the scalar reference: short outputs, vector
//! boundaries and mixed scalar/AVX2 tails; boundary/interior shifts plus
//! contract extrema; extreme samples and coefficients (output saturation);
//! and statistics at their documented value bounds.
use crate::{check::CheckReport, rng::Rng};
use nc_common::BitDepth;
use nc_dsp::{
    Dsp, VVC_ALF_BLOCK, VVC_ALF_DOT_LANES, VVC_ALF_GRAM_ALIGN, VVC_ALF_GRAM_MAX_MAGNITUDE,
    VVC_ALF_GRAM_MAX_SAMPLES, VVC_ALF_MAX_GROUP_SUM, VVC_ALF_REACH, VVC_ALF_SAMPLE_LANES,
    VVC_ALF_SAMPLE_STEP, VVC_ALF_SAMPLES, VVC_ALF_TAPS, vvc_alf_clip_value,
};

/// Widths up to two CTBs, including both sides of the 16/32-lane boundaries.
const WIDTHS: [usize; 21] = [
    1, 3, 4, 7, 8, 12, 16, 20, 31, 32, 33, 47, 48, 63, 64, 65, 68, 79, 95, 96, 128,
];

/// A row of `len` samples: noise, extremes, or a flat area with an edge.
fn row(rng: &mut Rng, pattern: usize, len: usize) -> Vec<u8> {
    (0..len)
        .map(|x| match pattern {
            0 => 0,
            1 => 255,
            2 => {
                if rng.below(2) == 0 {
                    0
                } else {
                    255
                }
            }
            3 => 100 + u8::from(x >= len / 2) * 60,
            _ => rng.byte(),
        })
        .collect()
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    filter(candidate, reference, rng, report);
    clipped_and_cross(candidate, reference, rng, report);
    laplacians(candidate, reference, rng, report);
    classify(candidate, reference, rng, report);
    samples(candidate, reference, rng, report);
    gram(candidate, reference, rng, report);
    dot(candidate, reference, rng, report);
}

#[allow(clippy::too_many_lines)]
fn clipped_and_cross(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let clips: [i16; 4] = std::array::from_fn(|i| vvc_alf_clip_value(i, BitDepth::EIGHT));
    for width in WIDTHS {
        for pattern in 0..6 {
            let slack = if pattern >= 3 { 16 } else { 0 };
            let data: Vec<Vec<u8>> = (0..10)
                .map(|_| row(rng, pattern, width + 2 * VVC_ALF_REACH + slack))
                .collect();
            let block_rows: [[&[u8]; 7]; VVC_ALF_BLOCK] = std::array::from_fn(|r| {
                std::array::from_fn(|i| {
                    let dy = i as isize - 3;
                    let dy = if pattern % 2 == 0 {
                        dy
                    } else {
                        dy.clamp(-(r as isize), r as isize)
                    };
                    data[(r as isize + 3 + dy) as usize].as_slice()
                })
            });
            let blocks = width.div_ceil(VVC_ALF_BLOCK);
            let coefficients: Vec<[i16; VVC_ALF_TAPS]> = (0..blocks)
                .map(|_| std::array::from_fn(|_| rng.below(256) as i16 - 128))
                .collect();
            let limits: Vec<[i16; VVC_ALF_TAPS]> = (0..blocks)
                .map(|_| std::array::from_fn(|_| clips[rng.below(4)]))
                .collect();
            for shifts in [[7; VVC_ALF_BLOCK], [10, 7, 7, 10]] {
                let run = |dsp: &Dsp| {
                    let mut out: [Vec<u8>; VVC_ALF_BLOCK] = std::array::from_fn(|_| vec![0; width]);
                    dsp.vvc_alf_filter_clipped_block_row(
                        &block_rows,
                        &coefficients,
                        &limits,
                        &shifts,
                        &mut out.each_mut().map(Vec::as_mut_slice),
                    );
                    out
                };
                report.cases += 1;
                if run(candidate) != run(reference) {
                    report.failures.push(format!(
                        "{} u8 vvc_alf_filter_clipped_block_row width={width} pattern={pattern} shifts={shifts:?}",
                        candidate.isa()
                    ));
                }
            }
            let single = block_rows[0];
            let chroma_c: [i16; 6] = std::array::from_fn(|_| rng.below(256) as i16 - 128);
            let chroma_l: [i16; 6] = std::array::from_fn(|_| clips[rng.below(4)]);
            for shift in [7, 10] {
                let mut clipped = [vec![0u8; width], vec![0u8; width]];
                let mut chroma = [vec![0u8; width], vec![0u8; width]];
                for (k, dsp) in [candidate, reference].into_iter().enumerate() {
                    dsp.vvc_alf_filter_clipped_row(
                        &single,
                        &coefficients,
                        &limits,
                        shift,
                        &mut clipped[k],
                    );
                    dsp.vvc_alf_filter_chroma_row(
                        &single,
                        chroma_c,
                        chroma_l,
                        shift,
                        &mut chroma[k],
                    );
                }
                report.cases += 2;
                if clipped[0] != clipped[1] {
                    report.failures.push(format!(
                        "{} u8 vvc_alf_filter_clipped_row width={width} pattern={pattern} shift={shift}",
                        candidate.isa()
                    ));
                }
                if chroma[0] != chroma[1] {
                    report.failures.push(format!(
                        "{} u8 vvc_alf_filter_chroma_row width={width} pattern={pattern} shift={shift}",
                        candidate.isa()
                    ));
                }
            }
            // Exercise both even and odd byte alignments of luma rows (`misalign` 0 or 1).
            let misalign = pattern & 1;
            let luma_data: Vec<Vec<u8>> = (0..4)
                .map(|_| row(rng, pattern, 2 * width + 2 + slack + misalign))
                .collect();
            let cross_rows: [&[u8]; 4] = std::array::from_fn(|r| &luma_data[r][misalign..]);
            let cross_c: [i16; 7] = std::array::from_fn(|k| match pattern {
                2 if k % 2 == 0 => -64,
                1 | 2 => 64,
                _ => rng.below(129) as i16 - 64,
            });
            let base = row(rng, (pattern + 2) % 6, width);
            let mut cross_out = [base.clone(), base];
            candidate.vvc_alf_cross_row(&cross_rows, cross_c, &mut cross_out[0]);
            reference.vvc_alf_cross_row(&cross_rows, cross_c, &mut cross_out[1]);
            report.cases += 1;
            if cross_out[0] != cross_out[1] {
                report.failures.push(format!(
                    "{} u8 vvc_alf_cross_row width={width} pattern={pattern}",
                    candidate.isa()
                ));
            }
        }
    }
}

fn dot(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    // Decision-sized values (weights to 2^14, terms to 2^30) and full-range
    // ones (the sums wrap identically).
    for (weight_bits, term_bits) in [(14, 30), (31, 31)] {
        for case in 0..64 {
            let value = |rng: &mut Rng, bits: u32| {
                let magnitude = (rng.next_u64() >> (64 - bits)) as i64;
                (if rng.below(2) == 0 {
                    magnitude
                } else {
                    -magnitude
                }) as i32
            };
            let a: [i32; VVC_ALF_DOT_LANES] = std::array::from_fn(|_| value(rng, weight_bits));
            let b: [i32; VVC_ALF_DOT_LANES] = std::array::from_fn(|_| value(rng, term_bits));
            report.cases += 1;
            if candidate.vvc_alf_dot(&a, &b) != reference.vvc_alf_dot(&a, &b) {
                report.failures.push(format!(
                    "{} vvc_alf_dot bits={weight_bits}/{term_bits} case={case}",
                    candidate.isa()
                ));
            }
        }
    }
}

fn classify(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for blocks in [1, 3, 4, 7, 8, 9, 16, 17, 32] {
        for pattern in 0..8 {
            // Uniform, flat, one direction dominant, or the largest sums.
            let groups: Vec<[u16; 4]> = (0..=blocks)
                .map(|_| {
                    std::array::from_fn(|d| match pattern {
                        0 => 0,
                        1 => VVC_ALF_MAX_GROUP_SUM,
                        2..=5 if d == pattern - 2 => {
                            rng.below(usize::from(VVC_ALF_MAX_GROUP_SUM)) as u16
                        }
                        2..=5 => rng.below(64) as u16,
                        _ => rng.below(usize::from(VVC_ALF_MAX_GROUP_SUM) + 1) as u16,
                    })
                })
                .collect();
            for boundary in [false, true] {
                let run = |dsp: &Dsp| {
                    let mut out = vec![[0; 2]; blocks];
                    dsp.vvc_alf_classify(&groups, boundary, &mut out);
                    out
                };
                report.cases += 1;
                if run(candidate) != run(reference) {
                    report.failures.push(format!(
                        "{} vvc_alf_classify blocks={blocks} pattern={pattern} boundary={boundary}",
                        candidate.isa()
                    ));
                }
            }
        }
    }
}

fn samples(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for count in 0..=VVC_ALF_SAMPLES {
        for pattern in 0..8 {
            let len = VVC_ALF_SAMPLE_STEP * count + 2 * VVC_ALF_REACH;
            let data: Vec<Vec<u8>> = (0..7).map(|_| row(rng, pattern, len)).collect();
            let rows: [&[u8]; 7] = std::array::from_fn(|r| data[r].as_slice());
            let source = row(rng, 7 - pattern, VVC_ALF_SAMPLE_STEP * count);
            let transposes = std::array::from_fn(|_| rng.below(4) as u8);
            let run = |dsp: &Dsp| {
                // Stale values must be overwritten for every computed sample.
                let mut out = [[7; VVC_ALF_SAMPLE_LANES]; VVC_ALF_SAMPLES];
                dsp.vvc_alf_samples(&rows, &source, &transposes, count, &mut out);
                out
            };
            report.cases += 1;
            if run(candidate) != run(reference) {
                report.failures.push(format!(
                    "{} vvc_alf_samples count={count} pattern={pattern}",
                    candidate.isa()
                ));
            }
        }
    }
}

fn filter(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for width in WIDTHS {
        for pattern in 0..8 {
            // Ten rows, as a block row away from virtual boundaries reads;
            // odd patterns mirror them per output row, as next to one. Noise
            // rows carry slack past the filter's reach, as padded pictures
            // do (kernels may load whole vectors there).
            let slack = if pattern >= 4 { 16 } else { 0 };
            let data: Vec<Vec<u8>> = (0..10)
                .map(|_| row(rng, pattern, width + 2 * VVC_ALF_REACH + slack))
                .collect();
            let rows: [[&[u8]; 7]; VVC_ALF_BLOCK] = std::array::from_fn(|r| {
                std::array::from_fn(|i| {
                    let dy = i as isize - 3;
                    let dy = if pattern % 2 == 0 {
                        dy
                    } else {
                        dy.clamp(-(r as isize), r as isize)
                    };
                    data[(r as isize + 3 + dy) as usize].as_slice()
                })
            });
            // Fixed-set range (|c| ≤ 64) and the signalled range's extremes.
            let limit = if pattern % 2 == 0 { 64 } else { 128 };
            let coefficients: Vec<[i16; VVC_ALF_TAPS]> = (0..width.div_ceil(4))
                .map(|_| std::array::from_fn(|_| rng.below(2 * limit) as i16 - limit as i16))
                .collect();
            for shifts in [
                [7; VVC_ALF_BLOCK],
                [10, 7, 7, 10],
                [1, 16, 7, 10],
                [16, 1, 15, 2],
            ] {
                let run = |dsp: &Dsp| {
                    let mut out: [Vec<u8>; VVC_ALF_BLOCK] = std::array::from_fn(|_| vec![0; width]);
                    dsp.vvc_alf_filter_block_row(
                        &rows,
                        &coefficients,
                        &shifts,
                        &mut out.each_mut().map(Vec::as_mut_slice),
                    );
                    out
                };
                report.cases += 1;
                if run(candidate) != run(reference) {
                    report.failures.push(format!(
                        "{} vvc_alf_filter_block_row width={width} pattern={pattern} shifts={shifts:?}",
                        candidate.isa()
                    ));
                }
            }
        }
    }
}

fn laplacians(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for groups in [1, 2, 3, 4, 5, 8, 17, 33] {
        for pattern in 0..8 {
            let data: Vec<Vec<u8>> = (0..4).map(|_| row(rng, pattern, 4 * groups + 2)).collect();
            let rows: [&[u8]; 4] = std::array::from_fn(|r| data[r].as_slice());
            let run = |dsp: &Dsp| {
                let mut out = vec![[0; 4]; groups];
                dsp.vvc_alf_laplacians(&rows, &mut out);
                out
            };
            report.cases += 1;
            if run(candidate) != run(reference) {
                report.failures.push(format!(
                    "{} vvc_alf_laplacians groups={groups} pattern={pattern}",
                    candidate.isa()
                ));
            }
        }
    }
}

fn gram(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    // Every even tail of an eight-sample step, a CTB's sampling and the cap.
    let lengths = (0..=8).map(|n| n * VVC_ALF_GRAM_ALIGN);
    for len in lengths.chain([512, VVC_ALF_GRAM_MAX_SAMPLES]) {
        for extreme in [false, true] {
            let bound = VVC_ALF_GRAM_MAX_MAGNITUDE;
            let samples: Vec<[i16; VVC_ALF_SAMPLE_LANES]> = (0..len)
                .map(|_| {
                    std::array::from_fn(|_| {
                        // All at the bound: the largest sums.
                        if extreme {
                            bound
                        } else {
                            rng.below(2 * bound as usize + 1) as i16 - bound
                        }
                    })
                })
                .collect();
            let run = |dsp: &Dsp| {
                let mut out = [7; VVC_ALF_DOT_LANES];
                dsp.vvc_alf_gram(&samples, &mut out);
                out
            };
            report.cases += 1;
            if run(candidate) != run(reference) {
                report.failures.push(format!(
                    "{} vvc_alf_gram len={len} extreme={extreme}",
                    candidate.isa()
                ));
            }
        }
    }
}
