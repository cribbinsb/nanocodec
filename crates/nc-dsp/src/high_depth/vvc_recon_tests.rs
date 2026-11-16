//! The VVC reconstruction kernels at 8 bits through [`DepthKernels`] equal
//! the 8-bit [`Dsp`] methods (one generic source), and stay inside the
//! depth, with the expected scale, at 10 bits.

use nc_common::{BitDepth, PlaneViewMut};

use crate::{Dsp, VvcIntraMode, VvcLfnst, VvcLmcs, VvcTransformKind};

/// A deterministic pattern in `0..=max`, with runs at the top.
fn pattern(len: usize, seed: usize, max: i32) -> Vec<i32> {
    (0..len)
        .map(|i| {
            if (i / 3 + seed).is_multiple_of(5) {
                max
            } else {
                (((i * 2_654_435_761 + seed * 40_503) >> 7) % (max as usize + 1)) as i32
            }
        })
        .collect()
}

/// Every mode the predictors take, angular ones for luma and chroma.
fn modes() -> impl Iterator<Item = VvcIntraMode> {
    [VvcIntraMode::Planar, VvcIntraMode::Dc].into_iter().chain(
        (2..=66).flat_map(|direction| {
            [true, false].map(|luma| VvcIntraMode::Angular { direction, luma })
        }),
    )
}

fn wide(samples: &[u8]) -> Vec<u16> {
    samples.iter().map(|&v| u16::from(v)).collect()
}

const SIDES: [usize; 6] = [2, 4, 8, 16, 32, 64];

#[test]
fn square_and_rectangular_prediction_at_eight_bits_match() {
    let dsp = Dsp::new();
    let eight = dsp.at(BitDepth::EIGHT);
    for (seed, mode) in modes().enumerate() {
        for side in [4, 8, 16, 32] {
            let top = pattern(2 * side, seed, 255);
            let left = pattern(2 * side, seed + 1, 255);
            let mut want = vec![0u8; side * side];
            let mut got = vec![0u16; side * side];
            dsp.vvc_intra_predict(side, mode, &top, &left, 77, &mut want);
            eight.vvc_intra_predict(side, mode, &top, &left, 77, &mut got);
            assert_eq!(wide(&want), got, "{side} {mode:?}");
            let transposed = dsp.vvc_intra_predict_oriented(side, mode, &top, &left, 9, &mut want);
            let got_transposed =
                eight.vvc_intra_predict_oriented(side, mode, &top, &left, 9, &mut got);
            assert_eq!((wide(&want), transposed), (got, got_transposed));
        }
        for (width, height) in SIDES.iter().flat_map(|&w| SIDES.map(|h| (w, h))) {
            let top = pattern(2 * width, seed + 2, 255);
            let left = pattern(2 * height, seed + 3, 255);
            let mut want = vec![0u8; width * height];
            let mut got = vec![0u16; width * height];
            dsp.vvc_intra_predict_rect((width, height), mode, &top, &left, 200, &mut want);
            eight.vvc_intra_predict_rect((width, height), mode, &top, &left, 200, &mut got);
            assert_eq!(wide(&want), got, "{width}x{height} {mode:?}");
        }
    }
}

#[test]
fn isp_mrl_mip_and_bdpcm_at_eight_bits_match() {
    let dsp = Dsp::new();
    let eight = dsp.at(BitDepth::EIGHT);
    let valid = |n: usize| (4..=64).contains(&n);
    for (width, height) in SIDES.iter().flat_map(|&w| SIDES.map(|h| (w, h))) {
        let area = width * height;
        let top = pattern(2 * width + 2, width, 255);
        let left = pattern(2 * height + 2, height, 255);
        let (mut want, mut got) = (vec![0u8; area], vec![0u16; area]);
        for direction in 0..=66 {
            if width >= 4 && area >= 16 {
                let cu = (width.max(8), height.max(8));
                dsp.vvc_intra_predict_isp(
                    cu,
                    (width, height),
                    direction,
                    &top,
                    &left,
                    31,
                    &mut want,
                );
                eight.vvc_intra_predict_isp(
                    cu,
                    (width, height),
                    direction,
                    &top,
                    &left,
                    31,
                    &mut got,
                );
                assert_eq!(wide(&want), got, "ISP {width}x{height} {direction}");
            }
            if valid(width) && valid(height) && direction >= 1 {
                for line in [1, 2] {
                    dsp.vvc_mrl_predict(
                        (width, height),
                        direction,
                        line,
                        &top,
                        &left,
                        5,
                        &mut want,
                    );
                    eight.vvc_mrl_predict(
                        (width, height),
                        direction,
                        line,
                        &top,
                        &left,
                        5,
                        &mut got,
                    );
                    assert_eq!(wide(&want), got, "MRL {width}x{height} {direction} {line}");
                }
            }
        }
        if valid(width) && valid(height) {
            let modes = if area == 16 {
                16
            } else if width == 4 || height == 4 || area == 64 {
                8
            } else {
                6
            };
            for mode in 0..modes {
                for transpose in [false, true] {
                    dsp.vvc_mip_predict((width, height), mode, transpose, &top, &left, &mut want);
                    eight.vvc_mip_predict((width, height), mode, transpose, &top, &left, &mut got);
                    assert_eq!(wide(&want), got, "MIP {width}x{height} {mode} {transpose}");
                }
            }
        }
        if width <= 32 && height <= 32 {
            for vertical in [false, true] {
                dsp.vvc_bdpcm_predict((width, height), vertical, &top, &left, &mut want);
                eight.vvc_bdpcm_predict((width, height), vertical, &top, &left, &mut got);
                assert_eq!(wide(&want), got, "BDPCM {width}x{height} {vertical}");
            }
        }
    }
}

#[test]
fn residual_add_at_eight_bits_matches() {
    let dsp = Dsp::new();
    let eight = dsp.at(BitDepth::EIGHT);
    for (width, height) in [(1, 16), (4, 4), (8, 2), (16, 32), (64, 64)] {
        let area = width * height;
        let prediction: Vec<u8> = pattern(area, width, 255).iter().map(|&v| v as u8).collect();
        for limit in [300, 32767] {
            let residual: Vec<i16> = pattern(area, height, 2 * limit)
                .iter()
                .map(|&v| (v - limit) as i16)
                .collect();
            let mut want = prediction.clone();
            let mut got = wide(&prediction);
            dsp.vvc_add_residual((width, height), &mut want, &residual);
            eight.vvc_add_residual((width, height), &mut got, &residual);
            assert_eq!(wide(&want), got, "{width}x{height} {limit}");
        }
    }
}

/// Levels with a few large values near the low frequencies.
fn levels(area: usize, seed: usize) -> Vec<i32> {
    (0..area)
        .map(|i| {
            if i % 7 < 2 && i < area / 2 {
                (pattern(1, i + seed, 4000)[0]) - 2000
            } else {
                0
            }
        })
        .collect()
}

#[test]
fn inverse_transforms_at_eight_bits_match() {
    let dsp = Dsp::new();
    let eight = dsp.at(BitDepth::EIGHT);
    let kinds = [
        VvcTransformKind::Dct2,
        VvcTransformKind::Dst7,
        VvcTransformKind::Dct8,
    ];
    for (width, height) in [
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (4, 16),
        (32, 8),
        (1, 16),
        (2, 8),
    ] {
        let area = width * height;
        let input = levels(area, width + height);
        for qp in [4, 27, 63] {
            for dependent in [false, true] {
                for horizontal in kinds {
                    for vertical in kinds {
                        let pair = [horizontal, vertical];
                        if !pair[0].valid(width) || !pair[1].valid(height) {
                            continue;
                        }
                        let (mut want, mut got) = (vec![0i16; area], vec![0i16; area]);
                        dsp.vvc_inverse_primary(
                            (width, height),
                            pair,
                            &input,
                            qp,
                            dependent,
                            &mut want,
                        );
                        eight.vvc_inverse_primary(
                            (width, height),
                            pair,
                            &input,
                            qp,
                            dependent,
                            &mut got,
                        );
                        assert_eq!(want, got, "{width}x{height} {pair:?} {qp}");
                        let matrix = vec![16u8; area];
                        eight.vvc_inverse_with_matrix(
                            (width, height),
                            pair,
                            None,
                            &input,
                            (qp, dependent),
                            &matrix,
                            &mut got,
                        );
                        assert_eq!(want, got, "matrix {width}x{height} {pair:?} {qp}");
                    }
                }
                if width >= 4 && height >= 4 {
                    let lfnst = VvcLfnst {
                        index: 1 + u8::from(dependent),
                        direction: 18,
                        prediction_size: (width, height),
                    };
                    let (mut want, mut got) = (vec![0i16; area], vec![0i16; area]);
                    dsp.vvc_inverse_lfnst((width, height), lfnst, &input, qp, dependent, &mut want);
                    eight.vvc_inverse_lfnst(
                        (width, height),
                        lfnst,
                        &input,
                        qp,
                        dependent,
                        &mut got,
                    );
                    assert_eq!(want, got, "LFNST {width}x{height} {qp}");
                }
            }
            if (2..=32).contains(&width) && (2..=32).contains(&height) {
                for bdpcm in 0..=2 {
                    let (mut want, mut got) = (vec![0i16; area], vec![0i16; area]);
                    dsp.vvc_inverse_skip((width, height), &input, qp, bdpcm, &mut want);
                    eight.vvc_inverse_skip((width, height), &input, qp, bdpcm, &mut got);
                    assert_eq!(want, got, "skip {width}x{height} {qp} {bdpcm}");
                }
            }
        }
    }
}

#[test]
fn lmcs_at_eight_bits_matches() {
    let dsp = Dsp::new();
    let eight = dsp.at(BitDepth::EIGHT);
    let mut deltas = [0i16; 16];
    for (i, delta) in deltas.iter_mut().enumerate() {
        *delta = [3, -5, 7, 0, -2, 9, -8, 1][i % 8];
    }
    let model8 = VvcLmcs::new([2, 13], deltas, 2).unwrap();
    let model16 = VvcLmcs::<u16>::new_at([2, 13], deltas, 2, BitDepth::EIGHT).unwrap();
    let mut want: Vec<u8> = (0..=255).collect();
    let mut got: Vec<u16> = (0..=255).collect();
    dsp.vvc_forward_lmcs(&model8, &mut want);
    eight.vvc_forward_lmcs(&model16, &mut got);
    assert_eq!(wide(&want), got);
    dsp.vvc_inverse_lmcs(
        &model8,
        &mut PlaneViewMut::new(&mut want, 16, 16, 16).unwrap(),
    );
    eight.vvc_inverse_lmcs(
        &model16,
        &mut PlaneViewMut::new(&mut got, 16, 16, 16).unwrap(),
    );
    assert_eq!(wide(&want), got);
    assert!((0..=255u8).all(|v| model8.chroma_scale(v) == model16.chroma_scale(u16::from(v))));
    let mut want: Vec<i16> = pattern(64, 1, 2000)
        .iter()
        .map(|&v| (v - 1000) as i16)
        .collect();
    let mut got = want.clone();
    dsp.vvc_scale_chroma_residual(3000, &mut want);
    eight.vvc_scale_chroma_residual(3000, &mut got);
    assert_eq!(want, got);
}

#[test]
fn ten_bit_predictions_of_flat_references_are_flat() {
    let dsp = Dsp::new();
    let ten = dsp.at(BitDepth::TEN);
    let value = 1000;
    let flat = vec![value; 2 * 64 + 2];
    let expected = |out: &[u16]| out.iter().all(|&v| i32::from(v) == value);
    for mode in modes() {
        for side in [4, 8, 16, 32] {
            let mut out = vec![0u16; side * side];
            ten.vvc_intra_predict(side, mode, &flat, &flat, value, &mut out);
            assert!(expected(&out), "{side} {mode:?}");
        }
        for (width, height) in SIDES.iter().flat_map(|&w| SIDES.map(|h| (w, h))) {
            let mut out = vec![0u16; width * height];
            ten.vvc_intra_predict_rect((width, height), mode, &flat, &flat, value, &mut out);
            assert!(expected(&out), "{width}x{height} {mode:?}");
        }
    }
    for direction in 1..=66 {
        let mut out = vec![0u16; 16 * 8];
        ten.vvc_mrl_predict((16, 8), direction, 2, &flat, &flat, value, &mut out);
        assert!(expected(&out), "MRL {direction}");
        ten.vvc_intra_predict_isp((16, 32), (16, 8), direction, &flat, &flat, value, &mut out);
        assert!(expected(&out), "ISP {direction}");
    }
    let mut out = vec![0u16; 8 * 4];
    ten.vvc_bdpcm_predict((8, 4), true, &flat, &flat, &mut out);
    assert!(expected(&out));
}

#[test]
fn ten_bit_predictions_stay_inside_the_depth() {
    let dsp = Dsp::new();
    let ten = dsp.at(BitDepth::TEN);
    let max = BitDepth::TEN.max();
    let inside = |out: &[u16]| out.iter().all(|&v| i32::from(v) <= max);
    for (seed, mode) in modes().enumerate() {
        let top = pattern(128, seed, max);
        let left = pattern(128, seed + 7, max);
        for (width, height) in [(4, 4), (8, 32), (64, 16), (32, 32)] {
            let mut out = vec![0u16; width * height];
            ten.vvc_intra_predict_rect((width, height), mode, &top, &left, max, &mut out);
            assert!(inside(&out), "{width}x{height} {mode:?}");
        }
    }
    for mode in 0..6 {
        let mut out = vec![0u16; 32 * 16];
        let (top, left) = (pattern(32, mode, max), pattern(16, mode + 1, max));
        ten.vvc_mip_predict((32, 16), mode as u8, mode % 2 == 0, &top, &left, &mut out);
        assert!(inside(&out), "MIP {mode}");
    }
    // Edges at the 10-bit mid value predict it exactly in every class
    // and mode (the first MIP input is `(1 << (BitDepth − 1)) − pTemp[0]`).
    let mid = BitDepth::TEN.mid();
    let neutral = [mid; 64];
    for (size, modes) in [((4, 4), 16), ((8, 4), 8), ((16, 32), 6)] {
        for mode in 0..modes {
            let mut out = vec![0u16; size.0 * size.1];
            ten.vvc_mip_predict(size, mode, false, &neutral, &neutral, &mut out);
            assert!(out.iter().all(|&v| i32::from(v) == mid), "{size:?} {mode}");
        }
    }
    let mut prediction = vec![1020u16; 16];
    ten.vvc_add_residual((4, 4), &mut prediction, &[100; 16]);
    assert!(prediction.iter().all(|&v| i32::from(v) == max));
}

#[test]
fn ten_bit_inverse_transforms_keep_the_eight_bit_scale() {
    // Two more bits of depth add 12 to `Qp′` for the same quantiser step:
    // the scaling is then exactly the 8-bit one, and the second stage
    // shifts two bits less, so the residual is four times the 8-bit one
    // up to rounding.
    let dsp = Dsp::new();
    let ten = dsp.at(BitDepth::TEN);
    let kinds = [
        [VvcTransformKind::Dct2; 2],
        [VvcTransformKind::Dst7, VvcTransformKind::Dct8],
    ];
    for (width, height) in [
        (4, 4),
        (8, 8),
        (32, 32),
        (16, 4),
        (8, 32),
        (64, 64),
        (1, 16),
    ] {
        let area = width * height;
        let input = levels(area, width);
        for pair in kinds {
            if !pair[0].valid(width) || !pair[1].valid(height) {
                continue;
            }
            for (qp, dependent) in [(10, false), (30, true)] {
                let (mut eight, mut deep) = (vec![0i16; area], vec![0i16; area]);
                dsp.vvc_inverse_primary((width, height), pair, &input, qp, dependent, &mut eight);
                ten.vvc_inverse_primary(
                    (width, height),
                    pair,
                    &input,
                    qp + 12,
                    dependent,
                    &mut deep,
                );
                // Away from the 16-bit clip of the residual.
                let unclipped = |e: i16| 4 * i32::from(e).abs() < i32::from(i16::MAX) - 4;
                for (&e, &d) in eight.iter().zip(&deep).filter(|(e, _)| unclipped(**e)) {
                    assert!(
                        (i32::from(d) - 4 * i32::from(e)).abs() <= 2,
                        "{width}x{height} {pair:?}: {d} vs 4 x {e}"
                    );
                }
            }
        }
    }
    // Transform skip does not depend on the depth, beyond its QP range.
    let input = levels(64, 3);
    let (mut eight, mut deep) = (vec![0i16; 64], vec![0i16; 64]);
    dsp.vvc_inverse_skip((8, 8), &input, 40, 1, &mut eight);
    ten.vvc_inverse_skip((8, 8), &input, 40, 1, &mut deep);
    assert_eq!(eight, deep);
    ten.vvc_inverse_skip((8, 8), &input, 75, 0, &mut deep);
}

#[test]
fn ten_bit_square_dct_matches_the_scalar_reference() {
    // Square DCT-II blocks take the high-depth HEVC kernels; the other
    // shapes the generic scalar reference. Both must agree.
    let dsp = Dsp::new();
    let ten = dsp.at(BitDepth::TEN);
    for side in [4, 8, 16, 32] {
        let area = side * side;
        let input = levels(area, side);
        let mut kernel = vec![0i16; area];
        let dct = [VvcTransformKind::Dct2; 2];
        ten.vvc_inverse_primary((side, side), dct, &input, 40, false, &mut kernel);
        let scaling =
            crate::codec_txfm::vvc_inverse::scaling((side, side), 40, false, BitDepth::TEN);
        let mut scaled = vec![0i16; area];
        dsp.vvc_scale(&input, &scaling, &mut scaled);
        let mut reference = vec![0i16; area];
        crate::codec_txfm::vvc_primary::inverse(
            (side, side),
            dct,
            &scaled,
            &mut reference,
            BitDepth::TEN,
        );
        assert_eq!(kernel, reference, "{side}");
    }
}

#[test]
fn ten_bit_lmcs_scales_the_eight_bit_model() {
    let dsp = Dsp::new();
    let ten = dsp.at(BitDepth::TEN);
    let depth = BitDepth::TEN;
    let model = VvcLmcs::<u16>::new_at([1, 15], [0; 16], 0, depth).unwrap();
    let mut samples: Vec<u16> = (0..=1023).collect();
    ten.vvc_forward_lmcs(&model, &mut samples);
    assert_eq!(
        samples,
        (0..=1023u16)
            .map(|v| v.saturating_sub(64))
            .collect::<Vec<_>>()
    );
    ten.vvc_inverse_lmcs(
        &model,
        &mut PlaneViewMut::new(&mut samples, 1024, 1024, 1).unwrap(),
    );
    assert_eq!(
        samples,
        (0..=1023u16).map(|v| v.max(64)).collect::<Vec<_>>()
    );
    assert!((0..=1023).all(|v| model.chroma_scale(v) == 2048));
    // 10-bit bin widths: 8..=511 (OrgCW 64).
    for (delta, valid) in [(-56, true), (-57, false), (447, true), (448, false)] {
        let model = VvcLmcs::<u16>::new_at([5, 5], [delta; 16], 0, depth);
        assert_eq!(model.is_some(), valid, "{delta}");
    }
    let mut residual = [-32768, -1025, -3, 3, 1023, 32767];
    ten.vvc_scale_chroma_residual(3072, &mut residual);
    assert_eq!(residual, [-1536, -1536, -5, 5, 1535, 1535]);
}
