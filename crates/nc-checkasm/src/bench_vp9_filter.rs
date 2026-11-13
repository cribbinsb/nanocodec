//! VP9 edge filtering: rejected, narrow, flat and mixed lanes in both orientations.

use crate::bench::{BenchResult, time};
use nc_dsp::{Dsp, Vp9FilterLimits, Vp9FilterWidth};

const SIDE: usize = 32;
const ORIGIN: usize = 12 * SIDE + 12;
const LINES: usize = 8;

pub(crate) fn bench(dsp: &Dsp, results: &mut Vec<BenchResult>) {
    let names = [
        [
            [
                "vp9_filter4_h_reject",
                "vp9_filter4_h_narrow",
                "vp9_filter4_h_flat",
                "vp9_filter4_h_mixed",
            ],
            [
                "vp9_filter4_v_reject",
                "vp9_filter4_v_narrow",
                "vp9_filter4_v_flat",
                "vp9_filter4_v_mixed",
            ],
        ],
        [
            [
                "vp9_filter8_h_reject",
                "vp9_filter8_h_narrow",
                "vp9_filter8_h_flat",
                "vp9_filter8_h_mixed",
            ],
            [
                "vp9_filter8_v_reject",
                "vp9_filter8_v_narrow",
                "vp9_filter8_v_flat",
                "vp9_filter8_v_mixed",
            ],
        ],
        [
            [
                "vp9_filter16_h_reject",
                "vp9_filter16_h_narrow",
                "vp9_filter16_h_flat",
                "vp9_filter16_h_mixed",
            ],
            [
                "vp9_filter16_v_reject",
                "vp9_filter16_v_narrow",
                "vp9_filter16_v_flat",
                "vp9_filter16_v_mixed",
            ],
        ],
    ];
    for (index, width) in [
        Vp9FilterWidth::Four,
        Vp9FilterWidth::Eight,
        Vp9FilterWidth::Sixteen,
    ]
    .into_iter()
    .enumerate()
    {
        for (vertical, patterns) in names[index].iter().enumerate() {
            let (step, along) = if vertical == 0 { (SIDE, 1) } else { (1, SIDE) };
            for (pattern, &name) in patterns.iter().enumerate() {
                let input: [u8; SIDE * SIDE] = core::array::from_fn(|i| {
                    let across = if vertical == 0 { i / SIDE } else { i % SIDE };
                    let line = if vertical == 0 { i % SIDE } else { i / SIDE };
                    match pattern {
                        0 => (i * 37 + i / SIDE * 11) as u8,
                        1 => 120 + (across % 3) as u8 * 2,
                        2 => 120 + u8::from(across >= 12) * 4,
                        _ => {
                            if line.is_multiple_of(2) {
                                120 + u8::from(across >= 12) * 4
                            } else {
                                (i * 37) as u8
                            }
                        }
                    }
                });
                let mut output = input;
                let ns_per_call = time(|| {
                    output.copy_from_slice(&input);
                    dsp.vp9_filter_edge(
                        &mut output,
                        ORIGIN,
                        step,
                        along,
                        LINES,
                        width,
                        Vp9FilterLimits::new(32),
                    );
                    u32::from(output[ORIGIN])
                });
                results.push(BenchResult {
                    kernel: name,
                    size: (LINES, [4, 8, 16][index]),
                    isa: dsp.isa(),
                    ns_per_call,
                });
            }
        }
    }
}
