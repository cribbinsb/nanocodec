//! The scalar network kernels against plain 64-bit formulas, and every
//! supported ISA against them at the edges of the contracts.

use super::*;
use nc_common::SplitMix64;

/// Dense layer shapes `(inputs, outputs)`: the planned layers, a single
/// output, and counts that leave partial output groups and clip steps.
const SHAPES: [(usize, usize); 6] = [(32, 64), (64, 16), (32, 16), (32, 1), (96, 7), (64, 70)];

/// The dense sum in 64 bits, where nothing can wrap.
fn wide_sum(x: &[u8], row: &[i8], bias: i32) -> i64 {
    row.iter().zip(x).fold(i64::from(bias), |sum, (&w, &x)| {
        sum + i64::from(w) * i64::from(x)
    })
}

/// A dense layer whose values `pick` draws: `(x, w, bias)`.
fn layer(
    inputs: usize,
    outputs: usize,
    mut pick: impl FnMut() -> (u8, i8, i32),
) -> (Vec<u8>, Vec<i8>, Vec<i32>) {
    let x = (0..inputs).map(|_| pick().0).collect();
    let w = (0..inputs * outputs).map(|_| pick().1).collect();
    let bias = (0..outputs).map(|_| pick().2).collect();
    (x, w, bias)
}

/// Every ISA this machine runs.
fn tables() -> impl Iterator<Item = Dsp> {
    Isa::ALL.into_iter().filter_map(Dsp::with_isa)
}

#[test]
fn dense_matches_the_wide_formula_within_the_bias_bound() {
    let mut rng = SplitMix64::new(7);
    for (inputs, outputs) in SHAPES {
        let limit = i64::from(nn_dense_max_bias(inputs));
        let (x, w, bias) = layer(inputs, outputs, || {
            let bias = rng.below(2 * limit as u64 + 1) as i64 - limit;
            (
                rng.below(u64::from(NN_ACTIVATION_MAX) + 1) as u8,
                rng.next_u64() as i8,
                bias as i32,
            )
        });
        for dsp in tables() {
            let mut linear = vec![0i32; outputs];
            dsp.nn_dense_linear(&x, &w, &bias, &mut linear);
            for shift in [0, 6, NN_MAX_DENSE_SHIFT] {
                let mut clipped = vec![0u8; outputs];
                dsp.nn_dense_clipped(&x, &w, &bias, shift, &mut clipped);
                for (o, row) in w.chunks_exact(inputs).enumerate() {
                    let want = wide_sum(&x, row, bias[o]);
                    assert_eq!(i64::from(linear[o]), want, "{} {inputs}->{o}", dsp.isa());
                    let activation = (want >> shift).clamp(0, i64::from(NN_ACTIVATION_MAX));
                    assert_eq!(
                        i64::from(clipped[o]),
                        activation,
                        "{} shift {shift}",
                        dsp.isa()
                    );
                }
            }
        }
    }
}

#[test]
fn dense_is_exact_at_the_pair_sum_edges() {
    // Every input at the maximum against every weight at one extreme: the
    // largest 16-bit pair sums of either sign, with the bias at its bound.
    for (inputs, outputs) in SHAPES {
        for weight in [i8::MIN, -i8::MAX, i8::MAX] {
            let limit = nn_dense_max_bias(inputs);
            let bias = if weight < 0 { -limit } else { limit };
            let (x, w, bias) = layer(inputs, outputs, || (NN_ACTIVATION_MAX, weight, bias));
            let want = wide_sum(&x, &w[..inputs], bias[0]);
            assert!(i32::try_from(want).is_ok(), "the bound holds the sum");
            for dsp in tables() {
                let mut out = vec![0i32; outputs];
                dsp.nn_dense_linear(&x, &w, &bias, &mut out);
                assert!(
                    out.iter().all(|&sum| i64::from(sum) == want),
                    "{}",
                    dsp.isa()
                );
            }
        }
    }
}

#[test]
fn zero_inputs_leave_the_bias() {
    let (x, w, bias) = layer(64, 5, || (0, i8::MIN, -77));
    for dsp in tables() {
        let (mut linear, mut clipped) = ([0i32; 5], [1u8; 5]);
        dsp.nn_dense_linear(&x, &w, &bias, &mut linear);
        dsp.nn_dense_clipped(&x, &w, &bias, 0, &mut clipped);
        assert_eq!((linear, clipped), ([-77; 5], [0; 5]), "{}", dsp.isa());
    }
}

#[test]
fn activation_floors_and_clips() {
    let acc: Vec<i16> = (0..256)
        .map(|i| match i % 8 {
            0 => i16::MIN,
            1 => i16::MAX,
            2 => -1,
            3 => 127,
            4 => 128,
            _ => (i * 257 - 30_000) as i16,
        })
        .collect();
    for shift in 0..=NN_MAX_ACCUMULATOR_SHIFT {
        let want: Vec<u8> = acc
            .iter()
            .map(|&v| (i64::from(v) >> shift).clamp(0, 127) as u8)
            .collect();
        for dsp in tables() {
            let mut out = vec![0u8; acc.len()];
            dsp.nn_clip_i16_to_u8(&acc, shift, &mut out);
            assert_eq!(out, want, "{} shift {shift}", dsp.isa());
        }
    }
}

#[test]
fn columns_add_then_subtract_back() {
    const ROWS: usize = 40;
    let mut rng = SplitMix64::new(11);
    for width in [16, 48, 64, 80, 256] {
        let columns: Vec<i16> = (0..ROWS * width).map(|_| rng.next_u64() as i16).collect();
        let rows: Vec<u16> = (0..23).map(|_| rng.below(ROWS as u64) as u16).collect();
        let start: Vec<i16> = (0..width).map(|_| rng.next_u64() as i16).collect();
        let want: Vec<i16> = (0..width)
            .map(|j| {
                let sum: i64 = rows
                    .iter()
                    .map(|&row| i64::from(columns[usize::from(row) * width + j]))
                    .sum();
                (i64::from(start[j]) + sum) as i16
            })
            .collect();
        for dsp in tables() {
            let mut acc = start.clone();
            dsp.nn_add_columns_i16(&mut acc, &columns, &rows);
            assert_eq!(acc, want, "{} width {width}", dsp.isa());
            dsp.nn_sub_columns_i16(&mut acc, &columns, &rows);
            assert_eq!(acc, start, "{} width {width}", dsp.isa());
        }
    }
}

#[test]
#[should_panic(expected = "a row of the column table")]
fn a_row_outside_the_table_panics() {
    let mut acc = [0i16; 64];
    Dsp::new().nn_add_columns_i16(&mut acc, &[0; 128], &[1, 2]);
}
