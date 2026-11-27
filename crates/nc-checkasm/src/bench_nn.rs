//! Network inference benchmarks (`Dsp::nn_*`): the planned dense layers and
//! accumulator updates. `size` is (inputs, outputs) for a dense layer and
//! (width, rows added) for the accumulator.
use crate::bench::{BenchResult, time};
use nc_dsp::{Dsp, NN_ACTIVATION_MAX};

/// Clipped layers, `(inputs, outputs)`.
const CLIPPED_LAYERS: [(usize, usize); 3] = [(32, 64), (64, 16), (32, 16)];
/// The output layer: 16 inputs padded to one 32-input vector, one output.
const LINEAR_LAYER: (usize, usize) = (32, 1);
/// Accumulator widths.
const WIDTHS: [usize; 2] = [64, 256];
/// Rows one update adds.
const ROW_COUNTS: [usize; 2] = [8, 64];
/// Rows of the column table (the row lists walk it out of order).
const TABLE_ROWS: usize = 256;
/// A typical activation scale.
const SHIFT: u32 = 6;

pub(crate) fn bench(dsp: &Dsp, results: &mut Vec<BenchResult>) {
    let mut push = |kernel, size, ns_per_call| {
        results.push(BenchResult {
            kernel,
            size,
            isa: dsp.isa(),
            ns_per_call,
        });
    };
    for (inputs, outputs) in CLIPPED_LAYERS {
        let (x, w, bias) = layer(inputs, outputs);
        let mut out = vec![0u8; outputs];
        push(
            "nn_dense_clipped",
            (inputs, outputs),
            time(|| {
                dsp.nn_dense_clipped(&x, &w, &bias, SHIFT, &mut out);
                u32::from(out[0])
            }),
        );
    }
    let (inputs, outputs) = LINEAR_LAYER;
    let (x, w, bias) = layer(inputs, outputs);
    let mut out = vec![0i32; outputs];
    push(
        "nn_dense_linear",
        LINEAR_LAYER,
        time(|| {
            dsp.nn_dense_linear(&x, &w, &bias, &mut out);
            out[0] as u32
        }),
    );
    for width in WIDTHS {
        // The timed calls keep adding to one accumulator, whose sums wrap:
        // the same instructions and time as sums that do not.
        let columns: Vec<i16> = (0..TABLE_ROWS * width)
            .map(|i| (i * 29 % 61) as i16 - 30)
            .collect();
        let mut acc = vec![0i16; width];
        for count in ROW_COUNTS {
            let rows: Vec<u16> = (0..count).map(|i| (i * 37 % TABLE_ROWS) as u16).collect();
            push(
                "nn_add_columns_i16",
                (width, count),
                time(|| {
                    dsp.nn_add_columns_i16(&mut acc, &columns, &rows);
                    acc[0] as u32
                }),
            );
        }
        let mut out = vec![0u8; width];
        push(
            "nn_clip_i16_to_u8",
            (width, 1),
            time(|| {
                dsp.nn_clip_i16_to_u8(&acc, SHIFT, &mut out);
                u32::from(out[0])
            }),
        );
    }
}

/// A dense layer of fixed pseudo-random values within the contract.
fn layer(inputs: usize, outputs: usize) -> (Vec<u8>, Vec<i8>, Vec<i32>) {
    let x = (0..inputs)
        .map(|i| (i * 53 % (usize::from(NN_ACTIVATION_MAX) + 1)) as u8)
        .collect();
    let w = (0..inputs * outputs).map(|i| (i * 37 + 11) as i8).collect();
    let bias = (0..outputs).map(|i| i as i32 * 1000 - 3000).collect();
    (x, w, bias)
}
