//! The engine-statistics kernels against the scalar reference: the shared
//! end of block (random scans, sparse and dense levels, block lengths with
//! and without a vector tail) and the directional intra estimate
//! statistics (every side, random and flat content, random edges).
use crate::{check::CheckReport, rng::Rng};
use nc_common::PlaneView;
use nc_dsp::{Dsp, ESTIMATE_EDGE_LEN};

/// Random cases per shape.
const CASES: usize = 24;
/// Intra estimate block sides.
const SIDES: [usize; 5] = [4, 8, 16, 32, 64];

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    check_end_of_block(candidate, reference, rng, report);
    check_intra_estimate(candidate, reference, rng, report);
}

fn check_end_of_block(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for len in [1, 7, 16, 17, 33, 64, 100, 256, 1024] {
        for case in 0..CASES {
            // A random permutation as the scan, its end-of-block table.
            let mut scan: Vec<usize> = (0..len).collect();
            for i in (1..len).rev() {
                scan.swap(i, rng.below(i + 1));
            }
            let mut table = vec![0u16; len];
            for (i, &pos) in scan.iter().enumerate() {
                table[pos] = (i + 1) as u16;
            }
            let levels: Vec<i16> = (0..len)
                .map(|_| match case % 3 {
                    0 => i16::from(rng.below(64) == 0) * (rng.next_u64() as i16 | 1),
                    1 => rng.next_u64() as i16,
                    _ => 0,
                })
                .collect();
            report.cases += 1;
            if candidate.end_of_block(&levels, &table) != reference.end_of_block(&levels, &table) {
                report.failures.push(format!(
                    "{} end_of_block len={len} case={case}",
                    candidate.isa()
                ));
            }
            // The same levels at 32 bits, some beyond 16 bits.
            let wide: Vec<i32> = levels
                .iter()
                .map(|&l| {
                    if case % 4 == 3 {
                        i32::from(l) << 15
                    } else {
                        i32::from(l)
                    }
                })
                .collect();
            report.cases += 1;
            if candidate.end_of_block_wide(&wide, &table)
                != reference.end_of_block_wide(&wide, &table)
            {
                report.failures.push(format!(
                    "{} end_of_block_wide len={len} case={case}",
                    candidate.isa()
                ));
            }
            report.cases += 1;
            if candidate.last_nonzero_wide(&wide, &table)
                != reference.end_of_block_wide(&wide, &table).0
            {
                report.failures.push(format!(
                    "{} last_nonzero_wide len={len} case={case}",
                    candidate.isa()
                ));
            }
        }
    }
}

fn check_intra_estimate(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for n in SIDES {
        let stride = n + 3;
        for case in 0..CASES {
            let flat = case % 4 == 0;
            let samples: Vec<u8> = (0..n * stride)
                .map(|_| if flat { 128 } else { rng.next_u64() as u8 })
                .collect();
            let edge: Vec<u8> = (0..ESTIMATE_EDGE_LEN)
                .map(|_| rng.next_u64() as u8)
                .collect();
            let view = PlaneView::new(&samples, stride, n, n).expect("block view");
            report.cases += 1;
            if candidate.intra_estimate_stats(&view, n, &edge)
                != reference.intra_estimate_stats(&view, n, &edge)
            {
                report.failures.push(format!(
                    "{} intra_estimate_stats n={n} case={case}",
                    candidate.isa()
                ));
            }
        }
    }
}
