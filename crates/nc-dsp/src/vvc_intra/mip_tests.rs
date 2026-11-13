//! Independent prepared-reference oracle cases from BSD-Clear `VVdeC` d2614690.
use crate::Dsp;
#[test]
fn matrix_predictions_match_reference_decoder() {
    let dsp = Dsp::new();
    let mut cases = 0;
    for line in include_str!("mip_fixtures.txt").lines() {
        let values: Vec<i32> = line
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let (width, height) = (values[0] as usize, values[1] as usize);
        let top = &values[4..4 + width];
        let left = &values[4 + width..4 + width + height];
        let expected: Vec<u8> = values[4 + width + height..]
            .iter()
            .map(|&v| v as u8)
            .collect();
        let mut actual = vec![0; width * height];
        dsp.vvc_mip_predict(
            (width, height),
            values[2] as u8,
            values[3] != 0,
            top,
            left,
            &mut actual,
        );
        assert_eq!(
            actual, expected,
            "shape {width}x{height} mode {} transpose {}",
            values[2], values[3]
        );
        cases += 1;
    }
    assert_eq!(cases, 72);
}
#[test]
fn neutral_boundary_stays_neutral_for_every_matrix_mode() {
    let dsp = Dsp::new();
    for width in [4, 8, 16, 32, 64] {
        for height in [4, 8, 16, 32, 64] {
            let count = if width == 4 && height == 4 {
                16
            } else if width == 4 || height == 4 || (width == 8 && height == 8) {
                8
            } else {
                6
            };
            for mode in 0..count {
                for transpose in [false, true] {
                    let mut actual = vec![0; width * height];
                    dsp.vvc_mip_predict(
                        (width, height),
                        mode,
                        transpose,
                        &[128; 64],
                        &[128; 64],
                        &mut actual,
                    );
                    assert!(actual.iter().all(|&v| v == 128));
                }
            }
        }
    }
}

#[test]
fn displaced_reference_prediction_matches_oracle() {
    let mut actual = [0; 16];
    Dsp::new().vvc_mrl_predict(
        (4, 4),
        4,
        2,
        &[108, 105, 107, 108, 105, 106, 106, 96, 101, 114],
        &[109, 112, 107, 105, 102, 97, 88, 77, 75, 75],
        109,
        &mut actual,
    );
    assert_eq!(
        actual,
        [
            100, 95, 87, 78, 94, 85, 77, 75, 83, 76, 75, 75, 76, 75, 75, 75
        ]
    );
}
