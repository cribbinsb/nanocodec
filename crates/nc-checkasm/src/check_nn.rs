//! The network inference kernels (`Dsp::nn_*`, `nc-dsp` module `nn`)
//! against the scalar reference over their whole contracts: activations in
//! 0..=127, every `i8` weight (−128 included), biases over all of `i32`
//! (the sums wrap identically on every ISA) and within the no-overflow
//! bound, every shift, and the edges: all-maximum inputs against
//! all-extreme weights of one sign (the largest 16-bit pair sums), zero
//! inputs, a single output, and widths and output counts that leave partial
//! vector steps.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::{
    Dsp, NN_ACCUMULATOR_ALIGN, NN_ACTIVATION_MAX, NN_DENSE_INPUT_ALIGN, NN_MAX_ACCUMULATOR_SHIFT,
    NN_MAX_DENSE_SHIFT, nn_dense_max_bias,
};

/// Dense layers `(inputs, outputs)`: the planned layers, a single output,
/// odd output counts, an odd number of 32-input vectors, and more outputs
/// than one block of the clipped layer.
const LAYERS: [(usize, usize); 9] = [
    (32, 64),
    (64, 16),
    (32, 16),
    (32, 1),
    (32, 3),
    (96, 7),
    (128, 13),
    (160, 2),
    (64, 70),
];

/// Accumulator widths in units of [`NN_ACCUMULATOR_ALIGN`]: one vector,
/// partial and whole register blocks.
const WIDTH_UNITS: [usize; 8] = [1, 2, 3, 4, 5, 7, 8, 16];

/// Row-list lengths of an accumulator update.
const ROW_COUNTS: [usize; 5] = [0, 1, 2, 8, 64];

/// Rows of the test column table.
const TABLE_ROWS: usize = 96;

/// Random rounds per dense layer and value pattern.
const DENSE_ROUNDS: usize = 4;

/// How a dense layer's values are drawn.
#[derive(Debug, Clone, Copy)]
enum Values {
    /// Uniform over the contract; biases over all of `i32`.
    Random,
    /// Uniform over the contract; biases within the no-overflow bound.
    Bounded,
    /// Inputs and weights at their extremes at random: 0 or 127, −128 or 127.
    Extremes,
    /// Every input 127 and every weight this value.
    Saturating(i8),
    /// Every input zero.
    ZeroInputs,
}

const VALUES: [Values; 7] = [
    Values::Random,
    Values::Bounded,
    Values::Extremes,
    Values::Saturating(i8::MAX),
    Values::Saturating(-i8::MAX),
    Values::Saturating(i8::MIN),
    Values::ZeroInputs,
];

/// A uniform value in `-limit..=limit`.
fn within(rng: &mut Rng, limit: i32) -> i32 {
    (rng.next_u64() % (2 * u64::from(limit.unsigned_abs()) + 1)) as i32 - limit
}

/// One dense layer's inputs, weights and biases.
struct Layer {
    x: Vec<u8>,
    w: Vec<i8>,
    bias: Vec<i32>,
}

impl Layer {
    fn draw(rng: &mut Rng, inputs: usize, outputs: usize, values: Values) -> Self {
        assert!(inputs.is_multiple_of(NN_DENSE_INPUT_ALIGN));
        let limit = nn_dense_max_bias(inputs);
        let activation = |rng: &mut Rng| rng.below(usize::from(NN_ACTIVATION_MAX) + 1) as u8;
        let pick = |rng: &mut Rng| match values {
            Values::Random => (activation(rng), rng.byte() as i8, rng.next_u64() as i32),
            Values::Bounded => (activation(rng), rng.byte() as i8, within(rng, limit)),
            Values::Extremes => (
                NN_ACTIVATION_MAX * rng.below(2) as u8,
                [i8::MIN, i8::MAX][rng.below(2)],
                within(rng, limit),
            ),
            Values::Saturating(weight) => (
                NN_ACTIVATION_MAX,
                weight,
                if weight < 0 { -limit } else { limit },
            ),
            Values::ZeroInputs => (0, rng.byte() as i8, rng.next_u64() as i32),
        };
        Self {
            x: (0..inputs).map(|_| pick(rng).0).collect(),
            w: (0..inputs * outputs).map(|_| pick(rng).1).collect(),
            bias: (0..outputs).map(|_| pick(rng).2).collect(),
        }
    }
}

fn check_dense(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for (inputs, outputs) in LAYERS {
        for values in VALUES {
            for _ in 0..DENSE_ROUNDS {
                let Layer { x, w, bias } = Layer::draw(rng, inputs, outputs, values);
                let (mut got, mut want) = (vec![0i32; outputs], vec![0i32; outputs]);
                candidate.nn_dense_linear(&x, &w, &bias, &mut got);
                reference.nn_dense_linear(&x, &w, &bias, &mut want);
                report.cases += 1;
                if got != want {
                    report.failures.push(format!(
                        "{} nn_dense_linear {inputs}->{outputs} {values:?}",
                        candidate.isa()
                    ));
                }
                for shift in 0..=NN_MAX_DENSE_SHIFT {
                    let (mut got, mut want) = (vec![0u8; outputs], vec![0u8; outputs]);
                    candidate.nn_dense_clipped(&x, &w, &bias, shift, &mut got);
                    reference.nn_dense_clipped(&x, &w, &bias, shift, &mut want);
                    report.cases += 1;
                    if got != want {
                        report.failures.push(format!(
                            "{} nn_dense_clipped {inputs}->{outputs} shift={shift} {values:?}",
                            candidate.isa()
                        ));
                    }
                }
            }
        }
    }
}

/// A 16-bit test value: random, or (`extreme`) one of the limits and the
/// values either side of the clip.
fn word(rng: &mut Rng, extreme: bool) -> i16 {
    const EDGES: [i16; 7] = [i16::MIN, -1, 0, 127, 128, 255, i16::MAX];
    if extreme {
        EDGES[rng.below(EDGES.len())]
    } else {
        rng.next_u64() as i16
    }
}

fn check_accumulator(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for units in WIDTH_UNITS {
        let width = units * NN_ACCUMULATOR_ALIGN;
        for extreme in [false, true] {
            let columns: Vec<i16> = (0..TABLE_ROWS * width)
                .map(|_| word(rng, extreme))
                .collect();
            let start: Vec<i16> = (0..width).map(|_| word(rng, extreme)).collect();
            for count in ROW_COUNTS {
                let rows: Vec<u16> = (0..count).map(|_| rng.below(TABLE_ROWS) as u16).collect();
                let (mut got, mut want) = (start.clone(), start.clone());
                candidate.nn_add_columns_i16(&mut got, &columns, &rows);
                reference.nn_add_columns_i16(&mut want, &columns, &rows);
                let added = got == want;
                // Subtract a different list, so the result is not the start.
                let rows: Vec<u16> = (0..count).map(|_| rng.below(TABLE_ROWS) as u16).collect();
                candidate.nn_sub_columns_i16(&mut got, &columns, &rows);
                reference.nn_sub_columns_i16(&mut want, &columns, &rows);
                report.cases += 2;
                if !added || got != want {
                    report.failures.push(format!(
                        "{} nn_{}_columns_i16 width={width} rows={count} extreme={extreme}",
                        candidate.isa(),
                        if added { "sub" } else { "add" }
                    ));
                }
            }
            for shift in 0..=NN_MAX_ACCUMULATOR_SHIFT {
                let (mut got, mut want) = (vec![0u8; width], vec![0u8; width]);
                candidate.nn_clip_i16_to_u8(&start, shift, &mut got);
                reference.nn_clip_i16_to_u8(&start, shift, &mut want);
                report.cases += 1;
                if got != want {
                    report.failures.push(format!(
                        "{} nn_clip_i16_to_u8 width={width} shift={shift} extreme={extreme}",
                        candidate.isa()
                    ));
                }
            }
        }
    }
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    check_dense(candidate, reference, rng, report);
    check_accumulator(candidate, reference, rng, report);
}
