//! The `u16` kernels at 8 bits equal the 8-bit kernels (the generic
//! references are one source), and stay inside the depth at 10 bits.

use nc_common::{BitDepth, PlaneView, PlaneViewMut};

use crate::{BlockSize, Dsp, Filter, HevcTransform, HevcWeights, Isa};

/// A deterministic sample pattern with edges and flat runs, `max` its top.
fn pattern(len: usize, seed: usize, max: u32) -> Vec<u32> {
    (0..len)
        .map(|i| {
            let v = (i * 2_654_435_761 + seed * 40_503) >> 7;
            if (i / 5 + seed).is_multiple_of(7) {
                max
            } else {
                (v as u32) % (max + 1)
            }
        })
        .collect()
}

fn wide(values: &[u32]) -> Vec<u16> {
    values.iter().map(|&v| v as u16).collect()
}

fn narrow(values: &[u32]) -> Vec<u8> {
    values.iter().map(|&v| v as u8).collect()
}

#[test]
fn prediction_at_eight_bits_matches_the_eight_bit_kernels() {
    let dsp = Dsp::new();
    let eight = dsp.at(BitDepth::EIGHT);
    for filter in [Filter::HevcLuma, Filter::HevcChroma] {
        for side in [4, 8, 16, 32, 64] {
            let size = BlockSize::new(side, side).unwrap();
            let span = side + filter.taps() - 1;
            let source = pattern(span * span, side, 255);
            let (bytes, words) = (narrow(&source), wide(&source));
            let window8 = PlaneView::new(&bytes, span, span, span).unwrap();
            let window16 = PlaneView::new(&words, span, span, span).unwrap();
            for frac in [(0, 0), (1, 0), (0, 3), (2, 1), (filter.positions() - 1, 2)] {
                let mut out8 = vec![0u8; side * side];
                let mut out16 = vec![0u16; side * side];
                eight.predict(
                    filter,
                    size,
                    &window8,
                    frac,
                    &mut PlaneViewMut::new(&mut out8, side, side, side).unwrap(),
                );
                eight.predict(
                    filter,
                    size,
                    &window16,
                    frac,
                    &mut PlaneViewMut::new(&mut out16, side, side, side).unwrap(),
                );
                assert_eq!(
                    wide(&out8.iter().map(|&v| u32::from(v)).collect::<Vec<_>>()),
                    out16
                );
                let mut prep8 = vec![0i16; side * side];
                let mut prep16 = vec![0i16; side * side];
                eight.predict_compound(filter, size, &window8, frac, &mut prep8, side);
                eight.predict_compound(filter, size, &window16, frac, &mut prep16, side);
                assert_eq!(prep8, prep16, "{filter:?} {side} {frac:?}");
                let mut bi8 = vec![0u8; side * side];
                let mut bi16 = vec![0u16; side * side];
                eight.hevc_predict_bi(
                    filter,
                    size,
                    &window8,
                    frac,
                    (&prep8, side),
                    &mut PlaneViewMut::new(&mut bi8, side, side, side).unwrap(),
                );
                eight.hevc_predict_bi(
                    filter,
                    size,
                    &window16,
                    frac,
                    (&prep16, side),
                    &mut PlaneViewMut::new(&mut bi16, side, side, side).unwrap(),
                );
                assert_eq!(bi8.iter().map(|&v| u16::from(v)).collect::<Vec<_>>(), bi16);
            }
        }
    }
}

#[test]
fn ten_bit_prediction_keeps_the_fourteen_bit_scale() {
    // A flat 10-bit block at value v predicts v << 4 at every position,
    // and uni-prediction returns v.
    let dsp = Dsp::with_isa(Isa::Scalar).unwrap();
    let ten = dsp.at(BitDepth::TEN);
    let side = 8;
    let size = BlockSize::new(side, side).unwrap();
    let span = side + 7;
    let words = vec![1000u16; span * span];
    let window = PlaneView::new(&words, span, span, span).unwrap();
    for frac in [(0, 0), (1, 0), (0, 2), (3, 3)] {
        let mut prep = vec![0i16; side * side];
        ten.predict_compound(Filter::HevcLuma, size, &window, frac, &mut prep, side);
        assert!(
            prep.iter().all(|&v| i32::from(v) + 8192 == 1000 << 4),
            "{frac:?}"
        );
        let mut out = vec![0u16; side * side];
        ten.predict(
            Filter::HevcLuma,
            size,
            &window,
            frac,
            &mut PlaneViewMut::new(&mut out, side, side, side).unwrap(),
        );
        assert!(out.iter().all(|&v| v == 1000));
    }
}

#[test]
fn weighting_intra_and_inverse_at_eight_bits_match() {
    let dsp = Dsp::new();
    let eight = dsp.at(BitDepth::EIGHT);
    let a: Vec<i16> = pattern(64, 1, 16_000)
        .iter()
        .map(|&v| v as i16 - 8192)
        .collect();
    let b: Vec<i16> = pattern(64, 2, 16_000)
        .iter()
        .map(|&v| v as i16 - 8192)
        .collect();
    let weights = HevcWeights {
        w0: 1,
        w1: 1,
        round: 64,
        shift: 7,
    };
    let mut out8 = vec![0u8; 64];
    let mut out16 = vec![0u16; 64];
    eight.hevc_weighted_prediction_centered(
        weights,
        &a,
        &b,
        &mut PlaneViewMut::new(&mut out8, 8, 8, 8).unwrap(),
    );
    eight.hevc_weighted_prediction_centered(
        weights,
        &a,
        &b,
        &mut PlaneViewMut::new(&mut out16, 8, 8, 8).unwrap(),
    );
    assert_eq!(
        out8.iter().map(|&v| u16::from(v)).collect::<Vec<_>>(),
        out16
    );

    let refs = pattern(3 * 16 + 1, 3, 255);
    let (refs8, refs16) = (narrow(&refs), wide(&refs));
    for angle in [-32, -13, 0, 5, 32] {
        let mut o8 = vec![0u8; 256];
        let mut o16 = vec![0u16; 256];
        eight.hevc_intra_angular(&refs8, 16, angle, &mut o8);
        eight.hevc_intra_angular(&refs16, 16, angle, &mut o16);
        assert_eq!(o8.iter().map(|&v| u16::from(v)).collect::<Vec<_>>(), o16);
    }

    for transform in HevcTransform::ALL {
        let n = transform.side();
        let coeffs: Vec<i16> = (0..n * n)
            .map(|i| {
                if i % 5 == 0 {
                    (i as i16 % 300) - 150
                } else {
                    0
                }
            })
            .collect();
        let base = pattern(n * n, n, 255);
        let mut d8 = narrow(&base);
        let mut d16 = wide(&base);
        eight.hevc_inverse_add(
            transform,
            &coeffs,
            (n, n),
            &mut PlaneViewMut::new(&mut d8, n, n, n).unwrap(),
        );
        eight.hevc_inverse_add(
            transform,
            &coeffs,
            (n, n),
            &mut PlaneViewMut::new(&mut d16, n, n, n).unwrap(),
        );
        assert_eq!(
            d8.iter().map(|&v| u16::from(v)).collect::<Vec<_>>(),
            d16,
            "{transform:?}"
        );
    }
}

#[test]
fn loop_filters_at_eight_bits_match() {
    let dsp = Dsp::new();
    let eight = dsp.at(BitDepth::EIGHT);
    let side = 16;
    for seed in 0..8 {
        let base = pattern(side * side, seed, 255);
        for vertical in [false, true] {
            let mut d8 = narrow(&base);
            let mut d16 = wide(&base);
            for (tc, beta) in [([3, 9], 40), ([1, 0], 64), ([12, 24], 20)] {
                eight.hevc_deblock_luma_edge(
                    &mut PlaneViewMut::new(&mut d8, side, side, side).unwrap(),
                    (8, 4),
                    vertical,
                    tc,
                    beta,
                );
                eight.hevc_deblock_luma_edge(
                    &mut PlaneViewMut::new(&mut d16, side, side, side).unwrap(),
                    (8, 4),
                    vertical,
                    tc,
                    beta,
                );
                eight.hevc_deblock_chroma_edge(
                    &mut PlaneViewMut::new(&mut d8, side, side, side).unwrap(),
                    (4, 8),
                    !vertical,
                    8,
                    [2, 0, 5, 9],
                );
                eight.hevc_deblock_chroma_edge(
                    &mut PlaneViewMut::new(&mut d16, side, side, side).unwrap(),
                    (4, 8),
                    !vertical,
                    8,
                    [2, 0, 5, 9],
                );
            }
            assert_eq!(d8.iter().map(|&v| u16::from(v)).collect::<Vec<_>>(), d16);
        }
        let mut s8 = narrow(&base);
        let mut s16 = wide(&base);
        let input8 = s8.clone();
        let input16 = s16.clone();
        for class in 0..4 {
            eight.sao_edge_apply(
                class,
                [3, 1, -1, -4],
                &PlaneView::new(&input8, side, side, side).unwrap(),
                &mut PlaneViewMut::new(&mut s8, side, side, side).unwrap(),
                (1, 1, 15, 15),
            );
            eight.sao_edge_apply(
                class,
                [3, 1, -1, -4],
                &PlaneView::new(&input16, side, side, side).unwrap(),
                &mut PlaneViewMut::new(&mut s16, side, side, side).unwrap(),
                (1, 1, 15, 15),
            );
        }
        eight.sao_band_apply_masked(
            seed as u8 * 3,
            [7, -7, 2, -2],
            &mut PlaneViewMut::new(&mut s8, side, side, side).unwrap(),
            (0, 0, side, side),
            &[],
        );
        eight.sao_band_apply_masked(
            seed as u8 * 3,
            [7, -7, 2, -2],
            &mut PlaneViewMut::new(&mut s16, side, side, side).unwrap(),
            (0, 0, side, side),
            &[],
        );
        assert_eq!(s8.iter().map(|&v| u16::from(v)).collect::<Vec<_>>(), s16);
    }
}

#[test]
fn ten_bit_outputs_stay_inside_the_depth() {
    let dsp = Dsp::new();
    let ten = dsp.at(BitDepth::TEN);
    let side = 16;
    let base = wide(&pattern(side * side, 5, 1023));
    let mut d = base.clone();
    ten.hevc_deblock_luma_edge(
        &mut PlaneViewMut::new(&mut d, side, side, side).unwrap(),
        (8, 4),
        true,
        [96, 40],
        256,
    );
    ten.sao_band_apply_masked(
        31,
        [31, -31, 31, 31],
        &mut PlaneViewMut::new(&mut d, side, side, side).unwrap(),
        (0, 0, side, side),
        &[],
    );
    assert!(d.iter().all(|&v| v <= 1023));
    let coeffs = vec![i16::MAX; 32 * 32];
    ten.hevc_inverse_add(
        HevcTransform::Dct32,
        &coeffs,
        (32, 32),
        &mut PlaneViewMut::new(&mut vec![1023u16; 1024], 32, 32, 32).unwrap(),
    );
}

#[test]
fn h264_deblocking_and_inverse_at_eight_bits_match() {
    use crate::{H264EdgeParams, H264MbEdges};
    let dsp = Dsp::new();
    let eight = dsp.at(BitDepth::EIGHT);
    let (side, stride) = (24, 24);
    let params = |seed: u8| H264EdgeParams {
        bs: [seed % 5, 4, (seed + 2) % 4, 1],
        tc0: [seed % 13, 2, 9, 0],
        alpha: 40 + seed * 9,
        beta: 6 + seed,
    };
    for seed in 0..8u8 {
        let edges = H264MbEdges {
            mask: 0xff,
            edges: [[params(seed); 4], [params(seed + 1); 4]],
        };
        let base = pattern(side * side, usize::from(seed), 255);
        let (mut d8, mut d16) = (narrow(&base), wide(&base));
        eight.h264_deblock_mb_luma(
            &mut PlaneViewMut::new(&mut d8, side, side, stride).unwrap(),
            (4, 4),
            &edges,
        );
        eight.h264_deblock_mb_luma(
            &mut PlaneViewMut::new(&mut d16, side, side, stride).unwrap(),
            (4, 4),
            &edges,
        );
        eight.h264_deblock_mb_chroma_plane(
            &mut PlaneViewMut::new(&mut d8, side, side, stride).unwrap(),
            (8, 8),
            &edges,
        );
        eight.h264_deblock_mb_chroma_plane(
            &mut PlaneViewMut::new(&mut d16, side, side, stride).unwrap(),
            (8, 8),
            &edges,
        );
        assert_eq!(d8.iter().map(|&v| u16::from(v)).collect::<Vec<_>>(), d16);
        let coeffs: [i32; 64] =
            core::array::from_fn(|i| (pattern(64, usize::from(seed), 600)[i] as i32) - 300);
        let coeffs4: [i16; 16] = core::array::from_fn(|i| coeffs[i] as i16);
        let (mut r8, mut r16) = (narrow(&base), wide(&base));
        dsp.h264_add_scaled_4x4(
            &[coeffs4],
            1,
            &mut PlaneViewMut::new(&mut r8, side, side, stride).unwrap(),
        );
        eight.h264_inverse_add_4x4(
            &core::array::from_fn(|i| coeffs[i]),
            &mut PlaneViewMut::new(&mut r16, side, side, stride).unwrap(),
        );
        let coeffs8: [i16; 64] = core::array::from_fn(|i| coeffs[i] as i16);
        dsp.h264_add_scaled_8x8(
            &coeffs8,
            &mut PlaneViewMut::new(&mut r8[8..], stride, 8, 8).unwrap(),
        );
        eight.h264_inverse_add_8x8(
            &coeffs,
            &mut PlaneViewMut::new(&mut r16[8..], stride, 8, 8).unwrap(),
        );
        assert_eq!(r8.iter().map(|&v| u16::from(v)).collect::<Vec<_>>(), r16);
    }
}

#[test]
fn h264_prediction_at_eight_bits_matches_the_eight_bit_kernels() {
    let dsp = Dsp::new();
    let eight = dsp.at(BitDepth::EIGHT);
    for filter in [Filter::H264Luma, Filter::H264Chroma] {
        for side in [4, 8, 16] {
            let size = BlockSize::new(side, side).unwrap();
            let span = side + filter.taps() - 1;
            let source = pattern(span * span, side, 255);
            let (bytes, words) = (narrow(&source), wide(&source));
            let window8 = PlaneView::new(&bytes, span, span, span).unwrap();
            let window16 = PlaneView::new(&words, span, span, span).unwrap();
            for fy in 0..filter.positions() {
                for fx in 0..filter.positions() {
                    let mut out8 = vec![0u8; side * side];
                    let mut out16 = vec![0u16; side * side];
                    eight.predict(
                        filter,
                        size,
                        &window8,
                        (fx, fy),
                        &mut PlaneViewMut::new(&mut out8, side, side, side).unwrap(),
                    );
                    eight.predict(
                        filter,
                        size,
                        &window16,
                        (fx, fy),
                        &mut PlaneViewMut::new(&mut out16, side, side, side).unwrap(),
                    );
                    let out8: Vec<u16> = out8.iter().map(|&v| u16::from(v)).collect();
                    assert_eq!(out8, out16, "{filter:?} {side} ({fx}, {fy})");
                }
            }
        }
    }
}
