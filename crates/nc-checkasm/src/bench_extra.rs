//! Dispatch entries omitted by the original transform/pixel benchmarks.

use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, GridBest, GridRows, HevcQuantizer, HevcTransform};

use crate::bench::time;

const SIDE: usize = 64;
const AREA: usize = SIDE * SIDE;
const QP: u8 = 32;

pub(crate) fn bench_extra(dsp: &Dsp, push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    let source: [u8; AREA] = core::array::from_fn(|i| (i * 37 + i / SIDE * 13) as u8);
    let prediction: [u8; AREA] = core::array::from_fn(|i| (i * 17 + i / SIDE * 11) as u8);
    let src = PlaneView::new(&source, SIDE, SIDE, SIDE).expect("plane");
    let pred = PlaneView::new(&prediction, SIDE, SIDE, SIDE).expect("plane");
    for side in [4, 8, 16, 32, 64] {
        let block = BlockSize::new(side, side).expect("block");
        let mut residual = vec![0; side * side];
        push(
            "residual_sad",
            block,
            time(|| dsp.residual_sad(block, &src, &pred, &mut residual, side)),
        );
        let mut output = prediction;
        push(
            "add_residual",
            block,
            time(|| {
                output.copy_from_slice(&prediction);
                let mut dst = PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE).expect("plane");
                dsp.add_residual(block, &mut dst, &residual, side);
                u32::from(output[0])
            }),
        );
        push(
            "satd_4x4_tiles",
            block,
            time(|| dsp.satd_4x4_tiles(block, &src, &pred)),
        );
    }
    let mb = BlockSize::new(16, 16).expect("block");
    push(
        "sad_4x4_macroblock",
        mb,
        time(|| {
            dsp.sad_quads_macroblock((&src, &pred), [(&src, &pred); 2])
                .0[0][0]
        }),
    );
    let block = BlockSize::new(SIDE, SIDE).expect("block");
    let mut best = GridBest::default();
    let visible = [u32::MAX; SIDE];
    push(
        "sad_grid_best_even",
        block,
        time(|| {
            dsp.sad_grid_best_64x64_at(&src, &pred, (0, 0), &visible, GridRows::Even, &mut best, 1);
            best.sad[0]
        }),
    );
    // Four adjacent full-pel positions, including every hierarchy level.
    let wide_data: [u8; 67 * SIDE] = core::array::from_fn(|i| (i * 17 + i / 67 * 11) as u8);
    let wide = PlaneView::new(&wide_data, 67, 67, SIDE).expect("four grid positions");
    for (name, rows) in [
        ("sad_grid_x4", GridRows::All),
        ("sad_grid_x4_even", GridRows::Even),
    ] {
        push(
            name,
            block,
            time(|| {
                dsp.sad_grid_best_x4_at(
                    &src,
                    &wide,
                    [(0, 0), (1, 0), (2, 0), (3, 0)],
                    &visible,
                    rows,
                    &mut best,
                    [0, 1, 2, 3],
                );
                best.sad[84]
            }),
        );
    }
    bench_bounded(dsp, &source, push);
}

fn bench_bounded(dsp: &Dsp, source: &[u8], push: &mut impl FnMut(&'static str, BlockSize, f64)) {
    for transform in HevcTransform::ALL {
        if transform == HevcTransform::Dst4 {
            continue;
        }
        let side = transform.side();
        let block = BlockSize::new(side, side).expect("block");
        let residual: Vec<i16> = source[..side * side]
            .iter()
            .map(|&x| i16::from(x) - 128)
            .collect();
        let mut coefficients = vec![0; side * side];
        push(
            "hevc_forward_bounded",
            block,
            time(|| {
                u32::from(dsp.hevc_forward_unless_zero(
                    transform,
                    &residual,
                    side,
                    0,
                    &mut coefficients,
                ))
            }),
        );
        let q = HevcQuantizer::new(QP, transform.log2(), false);
        let mut levels = coefficients.clone();
        push(
            "hevc_quantize_from",
            block,
            time(|| dsp.hevc_quantize_from(&coefficients, &q, &mut levels)),
        );
    }
}
