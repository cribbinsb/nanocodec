//! Cached copies, border stores and row statistics at cache-line and frame widths.
use crate::bench::{BenchResult, time};
use nc_common::{BorderStores, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, Filter, Moments};

pub(crate) fn bench(dsp: &Dsp, results: &mut Vec<BenchResult>) {
    for width in [64, 1280, 1884] {
        rows(dsp, results, width);
    }
    for width in [16, 32, 64] {
        prediction(dsp, results, width);
    }
}
fn rows(dsp: &Dsp, results: &mut Vec<BenchResult>, w: usize) {
    let input: Vec<_> = (0..w * 8).map(|i| (i * 37 + 11) as u8).collect();
    let src = PlaneView::new(&input, w, w, 8).unwrap();
    let mut output = vec![0; w * 8];
    let mut push = |kernel, ns_per_call| {
        results.push(BenchResult {
            kernel,
            size: (w, 8),
            isa: dsp.isa(),
            ns_per_call,
        });
    };
    push(
        "copy_plane",
        time(|| {
            let mut dst = PlaneViewMut::new(&mut output, w, w, 8).unwrap();
            dsp.copy_plane(&src, &mut dst);
            u32::from(output[0])
        }),
    );
    push(
        "plane_sse",
        time(|| dsp.sse(&src, &PlaneView::new(&output, w, w, 8).unwrap()) as u32),
    );
    let mut moments = vec![Moments::default(); w / 8];
    push(
        "moments_row8",
        time(|| {
            dsp.moments_8x8_row(&src, &mut moments);
            moments[0].sum_sq
        }),
    );
    push(
        "moments_row4",
        time(|| {
            dsp.moments_8x4_row(&src, &mut moments);
            moments[0].sum_sq
        }),
    );
    for (pick, name) in [(false, "pair_mean_rows"), (true, "pair_pick_rows")] {
        push(
            name,
            time(|| {
                let mut dst = PlaneViewMut::new(&mut output, w, w / 2, 8).unwrap();
                dsp.pairs_2x(&src, &mut dst, pick);
                u32::from(output[0])
            }),
        );
    }
    push(
        "border_fill",
        time(|| {
            dsp.fill(&mut output, 97);
            dsp.finish();
            u32::from(output[0])
        }),
    );
    push(
        "border_copy",
        time(|| {
            dsp.copy(&mut output, &input);
            dsp.finish();
            u32::from(output[0])
        }),
    );
}
fn prediction(dsp: &Dsp, results: &mut Vec<BenchResult>, w: usize) {
    let size = BlockSize::new(w, w).unwrap();
    let input = vec![97; (w + 7) * (w + 7)];
    let src = PlaneView::new(&input, w + 7, w + 7, w + 7).unwrap();
    let mut output = vec![0; w * w];
    let mut push = |kernel, ns_per_call| {
        results.push(BenchResult {
            kernel,
            size: (w, w),
            isa: dsp.isa(),
            ns_per_call,
        });
    };
    push(
        "fullpel_put",
        time(|| {
            let mut dst = PlaneViewMut::new(&mut output, w, w, w).unwrap();
            dsp.predict(Filter::Av1, size, &src, 0, 0, &mut dst);
            u32::from(output[0])
        }),
    );
    let (a, b) = (vec![97 * 16; w * w], vec![129 * 16; w * w]);
    push(
        "compound_average",
        time(|| {
            let mut dst = PlaneViewMut::new(&mut output, w, w, w).unwrap();
            dsp.average_compound(size, (&a, &b), w, &mut dst);
            u32::from(output[0])
        }),
    );
}
