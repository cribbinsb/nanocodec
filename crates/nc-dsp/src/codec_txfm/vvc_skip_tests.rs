//! Independent wide-integer inverse-skip equation and differential saturation tests.
use crate::{Dsp, Isa};
#[test]
fn inverse_skip_matches_normative_scaling() {
    for width in [2, 4, 8, 16, 32] {
        for height in [2, 4, 8, 16, 32] {
            let levels: Vec<i32> = (0..width * height)
                .map(|i| [-32768, 32767, 20000, -10000, 1, -1, 0][i % 7])
                .collect();
            for direction in 0..=2 {
                let mut integrated = levels.clone();
                for y in 0..height {
                    for x in 0..width {
                        let previous = match direction {
                            1 if x != 0 => integrated[y * width + x - 1],
                            2 if y != 0 => integrated[(y - 1) * width + x],
                            _ => 0,
                        };
                        integrated[y * width + x] =
                            (levels[y * width + x] + previous).clamp(-32768, 32767);
                    }
                }
                for qp in [0, 4, 5, 12, 37, 63] {
                    let effective = qp.max(4);
                    let scale = [40i64, 45, 51, 57, 64, 72][usize::from(effective % 6)];
                    let shift = 6 - i32::from(effective / 6);
                    let expected: Vec<i32> = integrated
                        .iter()
                        .map(|&level| {
                            let product = i64::from(level) * scale;
                            (if shift > 0 {
                                (product + (1 << (shift - 1))) >> shift
                            } else {
                                product << -shift
                            })
                            .clamp(-32768, 32767) as i32
                        })
                        .collect();
                    for dsp in [Dsp::with_isa(Isa::Scalar).unwrap(), Dsp::new()] {
                        let mut actual = vec![0; width * height];
                        dsp.vvc_inverse_skip((width, height), &levels, qp, direction, &mut actual);
                        assert_eq!(
                            actual, expected,
                            "{width}x{height} QP{qp} direction{direction}"
                        );
                    }
                }
            }
        }
    }
}
