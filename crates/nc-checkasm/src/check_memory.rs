//! Memory-kernel boundaries: every cache-line alignment and masked tail,
//! untouched destination padding, and SSE totals crossing signed/u32 limits.
use crate::check::CheckReport;
use nc_common::{BorderStores, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp};

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    check_sample_rows(candidate, reference, report);
    check_satd_quads(candidate, reference, report);
    for offset in 0..64 {
        for len in [0, 1, 31, 32, 63, 64, 65, 127, 128, 129, 255] {
            let src: Vec<_> = (0..len).map(|i| (i * 37 + 17) as u8).collect();
            for copy in [false, true] {
                let run = |dsp: &Dsp| {
                    let mut out = vec![0xa5; offset + len + 65];
                    let span = &mut out[offset..offset + len];
                    if copy {
                        dsp.copy(span, &src);
                    } else {
                        dsp.fill(span, 91);
                    }
                    dsp.finish();
                    out
                };
                record(
                    report,
                    candidate,
                    run(candidate) == run(reference),
                    "border alignment/tail",
                );
            }
        }
    }
    for w in 0..=129 {
        let (ss, ds, h) = (w + 3, w + 7, 3);
        let src: Vec<_> = (0..1 + (h - 1) * ss + w)
            .map(|i| (i * 17 + 31) as u8)
            .collect();
        let src = PlaneView::new(&src[1..], ss, w, h).unwrap();
        let run = |dsp: &Dsp| {
            let mut out = vec![0xa5; 3 + h * ds];
            let mut dst = PlaneViewMut::new(&mut out[3..], ds, w, h).unwrap();
            dsp.copy_plane(&src, &mut dst);
            out
        };
        record(
            report,
            candidate,
            run(candidate) == run(reference),
            "plane copy tail/padding",
        );
    }
    for w in [32767, 32768, 65535, 65536, 65537, 70000] {
        let a = vec![0; w];
        let b = vec![255; w];
        let a = PlaneView::new(&a, w, w, 1).unwrap();
        let b = PlaneView::new(&b, w, w, 1).unwrap();
        record(
            report,
            candidate,
            candidate.sse(&a, &b) == w as u64 * 255 * 255,
            "SSE wide total",
        );
    }
}

/// Frame-width rows straddle full-loop/tail boundaries with unaligned starts
/// and changing row alignment. Compare guards too, including the final row.
fn check_sample_rows(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    for w in [31, 32, 33, 639, 640, 641, 1279, 1280, 1281, 1884] {
        for offset in [0, 1, 31, 63] {
            let (ss, ds, h) = (2 * w + 3, w + 7, 4);
            let src: Vec<_> = (0..offset + (2 * h - 1) * ss + 2 * w)
                .map(|i| (i * 37 + i / 13) as u8)
                .collect();
            let src = PlaneView::new(&src[offset..], ss, 2 * w, 2 * h).unwrap();
            for mode in 0..3 {
                let run = |dsp: &Dsp| {
                    let mut out = vec![0xa5; offset + h * ds + 1];
                    let mut dst = PlaneViewMut::new(&mut out[offset..], ds, w, h).unwrap();
                    if mode == 0 {
                        dsp.downscale_2x(&src, &mut dst);
                    } else {
                        dsp.pairs_2x(&src, &mut dst, mode == 2);
                    }
                    out
                };
                record(
                    report,
                    candidate,
                    run(candidate) == run(reference),
                    "sample row alignment/tail",
                );
            }
        }
    }
}
fn record(report: &mut CheckReport, dsp: &Dsp, same: bool, what: &str) {
    report.cases += 1;
    if !same {
        report.failures.push(format!("{} {what}", dsp.isa()));
    }
}

/// Exact final-row extents and different source/prediction strides catch
/// accidental padding reads and quadrant interchange in paired-row loads.
fn check_satd_quads(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    let eight = BlockSize::new(8, 8).unwrap();
    for (ss, rs) in [(16, 16), (17, 31), (1280, 16)] {
        for offset in [0, 1, 15, 31] {
            let src: Vec<_> = (0..offset + 15 * ss + 16)
                .map(|i| (i * 37 + i / ss * 13) as u8)
                .collect();
            let rf: Vec<_> = (0..offset + 15 * rs + 16)
                .map(|i| (i * 11 + i / rs * 29) as u8)
                .collect();
            let src = PlaneView::new(&src[offset..], ss, 16, 16).unwrap();
            let rf = PlaneView::new(&rf[offset..], rs, 16, 16).unwrap();
            let expected: [u32; 4] = core::array::from_fn(|q| {
                let (x, y) = (8 * (q % 2), 8 * (q / 2));
                reference.satd(
                    eight,
                    &src.sub_view(x, y, 8, 8).unwrap(),
                    &rf.sub_view(x, y, 8, 8).unwrap(),
                )
            });
            record(
                report,
                candidate,
                candidate.satd_8x8_quads(&src, &rf) == expected,
                "SATD quadrant exact extent/stride",
            );
        }
    }
}
