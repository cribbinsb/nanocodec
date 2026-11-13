//! AV1 inverse transforms of every size and type (`Dsp::av1_inverse_add`)
//! against scalar: random, sparse, DC-only and extreme coefficients on a
//! random prediction inside a larger plane (stride and bounds), plus
//! per-size benchmarks of the kinds the decoder meets beyond square DCT.

use crate::{BenchResult, CheckReport, rng::Rng};
use nc_common::PlaneViewMut;
use nc_dsp::{Av1TxSize, Av1TxType, Dsp};

/// Coefficient patterns per (size, type).
const PATTERNS: usize = 5;
/// Extra columns right of the block and rows below it, which must not
/// change.
const MARGIN: usize = 3;

fn coefficient(pattern: usize, index: usize, rng: &mut Rng) -> i32 {
    match pattern {
        0 => i32::from(rng.next_u64() as u16 as i16),
        1 => rng.below(512) as i32 - 256,
        2 if rng.below(8) == 0 => rng.below(8192) as i32 - 4096,
        3 if index == 0 => i32::from(rng.next_u64() as u16 as i16),
        4 if index.is_multiple_of(2) => i32::from(i16::MAX),
        4 => i32::from(i16::MIN),
        _ => 0,
    }
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
            check_scan_prefixes(candidate, reference, size, kind, rng, report);
            for pattern in 0..PATTERNS {
                for _ in 0..iterations.max(1) {
                    let coeffs: Vec<i32> = (0..size.coded_area())
                        .map(|i| coefficient(pattern, i, rng))
                        .collect();
                    let eob = if pattern == 3 { 1 } else { size.coded_area() };
                    let mut expected = vec![0u8; stride * (h + MARGIN)];
                    rng.fill(&mut expected);
                    let mut actual = expected.clone();
                    for (dsp, pixels, used_eob) in [
                        (reference, &mut expected, size.coded_area()),
                        (candidate, &mut actual, eob),
                    ] {
                        let mut dst = PlaneViewMut::new(pixels, stride, w, h).expect("plane");
                        dsp.av1_inverse_add(size, kind, &coeffs, used_eob, &mut dst);
                    }
                    report.cases += 1;
                    if actual != expected {
                        report.failures.push(format!(
                            "av1_inverse_add {w}x{h} {kind:?} pattern={pattern}"
                        ));
                    }
                }
            }
        }
    }
}

/// A scan prefix's rectangle/triangle bounds, including each 4/8/16-row
/// cutoff. The reference deliberately runs with a full EOB so it cannot
/// share the candidate's sparse shortcut. Within a completed diagonal the
/// ordering is immaterial to the conservative row bound.
fn check_scan_prefixes(
    candidate: &Dsp,
    reference: &Dsp,
    size: Av1TxSize,
    kind: Av1TxType,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    use Av1TxType::{HAdst, HDct, HFlipadst, VAdst, VDct, VFlipadst};
    let (cw, ch) = (size.coded_width(), size.coded_height());
    let mut positions: Vec<_> = (0..cw * ch).collect();
    match kind {
        HDct | HAdst | HFlipadst => {}
        VDct | VAdst | VFlipadst => positions.sort_by_key(|&p| (p % ch, p / ch)),
        _ => positions.sort_by_key(|&p| (p / ch + p % ch, p % ch)),
    }
    for eob in [
        1, 2, 9, 10, 11, 25, 26, 27, 35, 36, 37, 99, 100, 101, 135, 136, 137,
    ] {
        if eob > positions.len() {
            continue;
        }
        let mut coeffs = vec![0; cw * ch];
        for &p in &positions[..eob] {
            coeffs[p] = i32::from(rng.next_u64() as i16);
        }
        let stride = size.width() + MARGIN;
        let mut expected = vec![0; stride * (size.height() + MARGIN)];
        rng.fill(&mut expected);
        let mut actual = expected.clone();
        reference.av1_inverse_add(
            size,
            kind,
            &coeffs,
            size.coded_area(),
            &mut PlaneViewMut::new(&mut expected, stride, size.width(), size.height()).unwrap(),
        );
        candidate.av1_inverse_add(
            size,
            kind,
            &coeffs,
            eob,
            &mut PlaneViewMut::new(&mut actual, stride, size.width(), size.height()).unwrap(),
        );
        report.cases += 1;
        if actual != expected {
            report
                .failures
                .push(format!("av1 inverse prefix {size:?} {kind:?} eob={eob}"));
        }
    }
}

/// Benchmarked kinds with their kernel names: both ADSTs, identity, and
/// the DCT (only rectangles run the new driver for it).
const BENCH_KINDS: [(Av1TxType, &str); 3] = [
    (Av1TxType::AdstAdst, "av1_inverse_adst_adst"),
    (Av1TxType::Idtx, "av1_inverse_idtx"),
    (Av1TxType::DctDct, "av1_inverse_dct_dct"),
];

pub(crate) fn bench(dsp: &Dsp, results: &mut Vec<BenchResult>) {
    let mut rng = Rng::new(0x1A7);
    let coeffs: Vec<i32> = (0..32 * 32).map(|_| rng.below(512) as i32 - 256).collect();
    let mut pixels = vec![128u8; 64 * 64];
    for size in Av1TxSize::ALL {
        let (w, h) = (size.width(), size.height());
        for (eob, full_eob, name) in [
            (1, false, "av1_inverse_dc"),
            (1, true, "av1_inverse_dc_full"),
            (10, false, "av1_inverse_eob10"),
            (10, true, "av1_inverse_eob10_full"),
        ] {
            let mut sparse = vec![0; size.coded_area()];
            // All diagonals r+c < 4 form the first ten scan entries.
            for c in 0..size.coded_width().min(4) {
                for r in 0..size.coded_height().min(4 - c) {
                    if eob != 1 || (r == 0 && c == 0) {
                        sparse[c * size.coded_height() + r] = 31;
                    }
                }
            }
            results.push(BenchResult {
                kernel: name,
                size: (w, h),
                isa: dsp.isa(),
                ns_per_call: crate::bench::time(|| {
                    let mut dst = PlaneViewMut::new(&mut pixels, w, w, h).unwrap();
                    dsp.av1_inverse_add(
                        size,
                        Av1TxType::DctDct,
                        &sparse,
                        if full_eob { size.coded_area() } else { eob },
                        &mut dst,
                    );
                    u32::from(pixels[0])
                }),
            });
        }
        for (kind, name) in BENCH_KINDS {
            if !kind.fits(size) {
                continue;
            }
            results.push(BenchResult {
                kernel: name,
                size: (w, h),
                isa: dsp.isa(),
                ns_per_call: crate::bench::time(|| {
                    let mut dst = PlaneViewMut::new(&mut pixels, w, w, h).expect("plane");
                    dsp.av1_inverse_add(size, kind, &coeffs, size.coded_area(), &mut dst);
                    u32::from(pixels[0])
                }),
            });
        }
    }
}
