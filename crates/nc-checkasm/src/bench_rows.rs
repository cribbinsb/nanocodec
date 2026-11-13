//! Row-tail benchmarks at real frame widths, including a non-vector multiple.

use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::Dsp;

use crate::bench::{BenchResult, time};

const WIDTHS: [usize; 3] = [640, 1280, 1884];
const ROWS: usize = 4;

pub(crate) fn bench_rows(dsp: &Dsp, results: &mut Vec<BenchResult>) {
    for width in WIDTHS {
        let input: Vec<u8> = (0..width * ROWS)
            .map(|i| (i * 37 + i / width * 11) as u8)
            .collect();
        let src = PlaneView::new(&input, width, width, ROWS).expect("frame rows");
        let half = width / 2;
        let mut output = vec![0; half * (ROWS / 2)];
        results.push(BenchResult {
            kernel: "downscale_2x_rows",
            size: (width, ROWS),
            isa: dsp.isa(),
            ns_per_call: time(|| {
                let mut dst =
                    PlaneViewMut::new(&mut output, half, half, ROWS / 2).expect("frame rows");
                dsp.downscale_2x(&src, &mut dst);
                u32::from(output[0])
            }),
        });
        let (mut cb, mut cr) = (vec![0; half * ROWS], vec![0; half * ROWS]);
        results.push(BenchResult {
            kernel: "deinterleave_uv_rows",
            size: (width, ROWS),
            isa: dsp.isa(),
            ns_per_call: time(|| {
                let mut u = PlaneViewMut::new(&mut cb, half, half, ROWS).expect("frame rows");
                let mut v = PlaneViewMut::new(&mut cr, half, half, ROWS).expect("frame rows");
                dsp.deinterleave_uv(&src, &mut u, &mut v);
                u32::from(cb[0]) + u32::from(cr[0])
            }),
        });
    }
}
