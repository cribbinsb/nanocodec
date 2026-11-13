//! AV1 contiguous run widths by tap count, orientation and active filter path.

use crate::bench::{BenchResult, time};
use nc_common::PlaneViewMut;
use nc_dsp::{Av1EdgeFilter, Av1FilterLength, Dsp, EdgeRun};

const SIDE: usize = 80;
const EDGE: usize = 8;

pub(crate) fn bench(dsp: &Dsp, results: &mut Vec<BenchResult>) {
    for vertical in [false, true] {
        for flat in [false, true] {
            let names = match (vertical, flat) {
                (false, false) => ["av1_run_narrow_h", "av1_edge_narrow_h"],
                (false, true) => ["av1_run_flat_h", "av1_edge_flat_h"],
                (true, false) => ["av1_run_narrow_v", "av1_edge_narrow_v"],
                (true, true) => ["av1_run_flat_v", "av1_edge_flat_v"],
            };
            let input: [u8; SIDE * SIDE] = core::array::from_fn(|i| {
                let across = if vertical { i % SIDE } else { i / SIDE };
                // Flat halves differ by one: all smoothing paths remain active.
                // The narrow pattern has a larger step and rejects flat masks.
                120 + u8::from(across >= EDGE) * if flat { 1 } else { 4 }
                    + if flat { 0 } else { (across % 3) as u8 }
            });
            for length in [
                Av1FilterLength::Four,
                Av1FilterLength::Six,
                Av1FilterLength::Eight,
                Av1FilterLength::Fourteen,
            ] {
                let taps = match length {
                    Av1FilterLength::Four => 4,
                    Av1FilterLength::Six => 6,
                    Av1FilterLength::Eight => 8,
                    Av1FilterLength::Fourteen => 14,
                };
                for lines in [16, 64] {
                    let run = EdgeRun {
                        x: EDGE,
                        y: EDGE,
                        lines,
                        filter: Av1EdgeFilter {
                            length,
                            limit: 32,
                            blimit: 64,
                            thresh: 8,
                        },
                    };
                    let mut output = input;
                    for (single, name) in names.into_iter().enumerate() {
                        let ns_per_call = time(|| {
                            output.copy_from_slice(&input);
                            let mut dst = PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE)
                                .expect("benchmark plane");
                            if single == 0 {
                                dsp.av1_loop_filter_runs(&mut dst, vertical, &[run]);
                            } else {
                                dsp.av1_loop_filter_edge(
                                    &mut dst,
                                    (EDGE, EDGE),
                                    vertical,
                                    lines,
                                    run.filter,
                                );
                            }
                            u32::from(output[EDGE * SIDE + EDGE])
                        });
                        results.push(BenchResult {
                            kernel: name,
                            size: (taps, lines),
                            isa: dsp.isa(),
                            ns_per_call,
                        });
                    }
                }
            }
        }
    }
}
