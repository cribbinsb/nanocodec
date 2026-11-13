//! Bounded input/SAD kernel probe for callgrind and locked native/WASM timing.
use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, GridBest, GridRows};
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).map_or("rgba", String::as_str);
    let rounds: usize = args.get(2).map_or(Ok(100), |s| s.parse()).unwrap();
    let dsp = if args.get(3).is_some_and(|arg| arg == "scalar") {
        Dsp::with_isa(nc_dsp::Isa::Scalar).unwrap()
    } else {
        Dsp::new()
    };
    let (width, height) = (1280, 720);
    let src: Vec<_> = (0..4 * width * height)
        .map(|i| (i * 17 + i / 37) as u8)
        .collect();
    let mut luma = vec![0; width * height];
    let mut cb = vec![0; width * height / 4];
    let mut cr = vec![0; width * height / 4];
    let start = Instant::now();
    let mut sum = 0u64;
    for _ in 0..rounds {
        if mode == "rgba" {
            dsp.rgba_to_i420(
                black_box(&src),
                &mut PlaneViewMut::new(&mut luma, width, width, height).unwrap(),
                &mut PlaneViewMut::new(&mut cb, width / 2, width / 2, height / 2).unwrap(),
                &mut PlaneViewMut::new(&mut cr, width / 2, width / 2, height / 2).unwrap(),
            );
            sum += u64::from(black_box(luma[123]));
        } else if mode == "grid" || mode == "grid-all" || mode == "scan" || mode == "satd" {
            let input = PlaneView::new(&src, 256, 64, 64).unwrap();
            let reference = PlaneView::new(&src[19..], 256, 96, 64).unwrap();
            if mode == "scan" {
                let mut costs = [0; 33];
                dsp.sad_scan_16x16_even_rows_at(black_box(&input), &reference, (0, 0), &mut costs);
                sum += u64::from(black_box(costs[0]));
            } else if mode == "satd" {
                sum += u64::from(dsp.satd(
                    BlockSize::new(64, 64).unwrap(),
                    black_box(&input),
                    &reference,
                ));
            } else {
                let rows = if mode == "grid-all" {
                    GridRows::All
                } else {
                    GridRows::Even
                };
                let mut best = GridBest::default();
                dsp.sad_grid_best_x4_at(
                    black_box(&input),
                    &reference,
                    [(0, 0), (1, 0), (2, 0), (3, 0)],
                    &[u32::MAX; 64],
                    rows,
                    &mut best,
                    [0, 1, 2, 3],
                );
                sum += u64::from(black_box(best.sad[84]));
            }
        } else {
            for n in [8, 16, 32, 64] {
                let size = BlockSize::new(n, n).unwrap();
                let input = PlaneView::new(&src, 256, n, n).unwrap();
                let refs =
                    [1, 3, 5, 7].map(|offset| PlaneView::new(&src[offset..], 256, n, n).unwrap());
                sum += if mode == "sad" {
                    u64::from(dsp.sad(size, black_box(&input), black_box(&refs[0])))
                } else {
                    dsp.sad_x4(size, black_box(&input), &refs)
                        .into_iter()
                        .map(u64::from)
                        .sum()
                };
            }
        }
    }
    println!(
        "{mode} {rounds} calls {:.3} ms checksum {sum}",
        start.elapsed().as_secs_f64() * 1000.
    );
}
