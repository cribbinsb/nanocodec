//! The VVC inter kernels at any depth: at 8 bits the `u16` instances equal
//! the 8-bit [`Dsp`] methods (every ISA's, so the generic references stay
//! one source with the SIMD kernels), and at 10 bits flat inputs keep their
//! value and outputs stay inside the depth.

use nc_common::{BitDepth, PlaneView, PlaneViewMut};

use crate::{BlockSize, Dsp, Filter, HevcWeights, Isa, SubBlockMotion, VvcGeometric};

/// A deterministic pseudo-random sequence in `0..=max`.
fn pattern(len: usize, seed: usize, max: u32) -> Vec<u32> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ seed as u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % (u64::from(max) + 1)) as u32
        })
        .collect()
}

fn narrow(values: &[u32]) -> Vec<u8> {
    values.iter().map(|&v| v as u8).collect()
}

fn wide(values: &[u32]) -> Vec<u16> {
    values.iter().map(|&v| v as u16).collect()
}

fn widened(values: &[u8]) -> Vec<u16> {
    values.iter().map(|&v| u16::from(v)).collect()
}

/// 14-bit predictions of 8-bit samples with filter overshoot: the inputs of
/// the blends and BDOF.
fn predictions(len: usize, seed: usize) -> Vec<i16> {
    pattern(len, seed, 17_000)
        .iter()
        .map(|&v| v as i16 - 300)
        .collect()
}

/// Every ISA this CPU runs, so the 8-bit side covers each SIMD tier.
fn tables() -> impl Iterator<Item = Dsp> {
    Isa::ALL.into_iter().filter_map(Dsp::with_isa)
}

const FILTERS: [Filter; 5] = [
    Filter::HevcLuma,
    Filter::HevcChroma,
    Filter::VvcLuma,
    Filter::VvcAffineLuma,
    Filter::VvcChroma,
];

#[test]
fn high_precision_at_eight_bits_matches() {
    let dsp = Dsp::new();
    let eight = dsp.at(BitDepth::EIGHT);
    for filter in FILTERS {
        for (w, h) in [(2, 4), (4, 2), (12, 8), (16, 16), (64, 64)] {
            let (span_w, span_h) = (w + filter.taps() - 1, h + filter.taps() - 1);
            let source = pattern(span_w * span_h, w * h, 255);
            let (bytes, words) = (narrow(&source), wide(&source));
            let window8 = PlaneView::new(&bytes, span_w, span_w, span_h).unwrap();
            let window16 = PlaneView::new(&words, span_w, span_w, span_h).unwrap();
            let last = filter.positions() - 1;
            for phases in [(0, 0), (1, 0), (0, 3), (2, 1), (last, last)] {
                for alternate in [false, filter == Filter::VvcLuma] {
                    let phases = if alternate { (8, phases.1) } else { phases };
                    let mut out8 = vec![0i16; w * h];
                    let mut out16 = vec![0i16; w * h];
                    eight.predict_high_precision(
                        filter,
                        (w, h),
                        &window8,
                        phases,
                        alternate,
                        &mut out8,
                    );
                    eight.predict_high_precision(
                        filter,
                        (w, h),
                        &window16,
                        phases,
                        alternate,
                        &mut out16,
                    );
                    assert_eq!(out8, out16, "{filter:?} {w}x{h} {phases:?} {alternate}");
                }
            }
        }
    }
}

#[test]
fn sub_blocks_at_eight_bits_match() {
    let (side, per_row, rows) = (4, 4, 2);
    let size = BlockSize::new(side, side).unwrap();
    for dsp in tables() {
        let eight = dsp.at(BitDepth::EIGHT);
        for filter in [Filter::VvcAffineLuma, Filter::VvcChroma] {
            let span = 24;
            let source = pattern(span * span, 7, 255);
            let (bytes, words) = (narrow(&source), wide(&source));
            let motion: Vec<SubBlockMotion> = pattern(4 * per_row * rows, 3, 1000)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| SubBlockMotion {
                    x: (v[0] % 12) as u16,
                    y: (v[1] % 12) as u16,
                    frac_x: (v[2] as usize % filter.positions()) as u8,
                    frac_y: (v[3] as usize % filter.positions()) as u8,
                })
                .collect();
            let (w, h) = (side * per_row, side * rows);
            let mut out8 = vec![0u8; w * h];
            let mut out16 = vec![0u16; w * h];
            eight.predict_sub_blocks(
                filter,
                size,
                &PlaneView::new(&bytes, span, span, span).unwrap(),
                &motion,
                per_row,
                &mut PlaneViewMut::new(&mut out8, w, w, h).unwrap(),
            );
            eight.predict_sub_blocks(
                filter,
                size,
                &PlaneView::new(&words, span, span, span).unwrap(),
                &motion,
                per_row,
                &mut PlaneViewMut::new(&mut out16, w, w, h).unwrap(),
            );
            assert_eq!(widened(&out8), out16, "{filter:?}");
        }
    }
}

#[test]
fn weighting_and_geometric_blend_at_eight_bits_match() {
    let (w, h) = (16, 8);
    let a = predictions(w * h, 1);
    let b = predictions(w * h, 2);
    let cases = [
        // Default bi-prediction (H.266 §8.5.6.6.2 at 8 bits).
        HevcWeights {
            w0: 1,
            w1: 1,
            round: 64,
            shift: 7,
        },
        // BCW weights 5/8, 3/8.
        HevcWeights {
            w0: 5,
            w1: 3,
            round: 1 << 9,
            shift: 10,
        },
        // Explicit uni-prediction with an offset.
        HevcWeights {
            w0: 37,
            w1: 0,
            round: (1 << 6) + (-9 << 7),
            shift: 7,
        },
    ];
    for dsp in tables() {
        let eight = dsp.at(BitDepth::EIGHT);
        for weights in cases {
            let mut out8 = vec![0u8; w * h];
            let mut out16 = vec![0u16; w * h];
            eight.hevc_weighted_prediction(
                weights,
                &a,
                &b,
                &mut PlaneViewMut::new(&mut out8, w, w, h).unwrap(),
            );
            eight.hevc_weighted_prediction(
                weights,
                &a,
                &b,
                &mut PlaneViewMut::new(&mut out16, w, w, h).unwrap(),
            );
            assert_eq!(widened(&out8), out16, "{weights:?}");
        }
        for split in [0, 7, 31, 63] {
            let mask = VvcGeometric::new((32, 16), split);
            for (origin, chroma) in [((0, 0), false), ((16, 8), false), ((0, 0), true)] {
                let mut out8 = vec![0u8; w * h];
                let mut out16 = vec![0u16; w * h];
                eight.vvc_geometric_blend(mask, origin, (w, h), chroma, (&a, &b), &mut out8);
                eight.vvc_geometric_blend(mask, origin, (w, h), chroma, (&a, &b), &mut out16);
                assert_eq!(widened(&out8), out16, "{split} {origin:?} {chroma}");
            }
        }
    }
}

#[test]
fn dmvr_bilinear_and_bdof_at_eight_bits_match() {
    for dsp in tables() {
        let eight = dsp.at(BitDepth::EIGHT);
        for (w, h) in [(8, 8), (16, 8), (20, 20)] {
            let source = pattern((w + 1) * (h + 1), w + h, 255);
            let (bytes, words) = (narrow(&source), wide(&source));
            let window8 = PlaneView::new(&bytes, w + 1, w + 1, h + 1).unwrap();
            let window16 = PlaneView::new(&words, w + 1, w + 1, h + 1).unwrap();
            for phases in [(0, 0), (8, 0), (0, 3), (5, 11), (15, 15)] {
                let mut out8 = vec![0i16; w * h];
                let mut out16 = vec![0i16; w * h];
                eight.vvc_dmvr_bilinear((w, h), &window8, phases, &mut out8);
                eight.vvc_dmvr_bilinear((w, h), &window16, phases, &mut out16);
                assert_eq!(out8, out16, "{w}x{h} {phases:?}");
            }
        }
        for (w, h) in [(4, 4), (8, 16), (12, 8), (16, 16)] {
            let count = (w + 2) * (h + 2);
            for seed in 0..4 {
                let inputs = [predictions(count, seed), predictions(count, seed + 9)];
                let mut out8 = vec![0u8; w * h];
                let mut out16 = vec![0u16; w * h];
                eight.vvc_bdof((w, h), [&inputs[0], &inputs[1]], &mut out8);
                eight.vvc_bdof((w, h), [&inputs[0], &inputs[1]], &mut out16);
                assert_eq!(widened(&out8), out16, "{w}x{h} seed {seed}");
            }
        }
    }
}

/// A flat 10-bit sample value, and the same at the 14-bit precision
/// (`<< (14 − BitDepth)`).
const FLAT: u16 = 1000;
const FLAT_14: i16 = (FLAT as i16) << 4;
const TEN_MAX: u16 = 1023;

#[test]
fn ten_bit_prediction_keeps_flat_blocks() {
    let dsp = Dsp::new();
    let ten = dsp.at(BitDepth::TEN);
    let span = 16 + 7;
    let words = vec![FLAT; span * span];
    let window = PlaneView::new(&words, span, span, span).unwrap();
    for filter in FILTERS {
        for phases in [(0, 0), (1, 0), (0, 3), (3, 2)] {
            for alternate in [false, filter == Filter::VvcLuma] {
                let phases = if alternate { (8, phases.1) } else { phases };
                let mut out = vec![0i16; 16 * 12];
                ten.predict_high_precision(filter, (16, 12), &window, phases, alternate, &mut out);
                assert!(out.iter().all(|&v| v == FLAT_14), "{filter:?} {phases:?}");
            }
        }
    }
    let motion = [
        SubBlockMotion {
            x: 0,
            y: 3,
            frac_x: 5,
            frac_y: 0,
        },
        SubBlockMotion {
            x: 7,
            y: 1,
            frac_x: 9,
            frac_y: 13,
        },
    ];
    let mut out = vec![0u16; 8 * 4];
    ten.predict_sub_blocks(
        Filter::VvcAffineLuma,
        BlockSize::new(4, 4).unwrap(),
        &window,
        &motion,
        2,
        &mut PlaneViewMut::new(&mut out, 8, 8, 4).unwrap(),
    );
    assert!(out.iter().all(|&v| v == FLAT));

    let mut bilinear = vec![0i16; 16 * 16];
    ten.vvc_dmvr_bilinear((16, 16), &window, (7, 3), &mut bilinear);
    // DMVR's 10-bit precision is the sample itself at 10 bits.
    assert!(bilinear.iter().all(|&v| v == FLAT as i16));
}

#[test]
fn ten_bit_blends_keep_flat_blocks_and_clip() {
    let dsp = Dsp::new();
    let ten = dsp.at(BitDepth::TEN);
    let (w, h) = (16, 16);
    let flat = vec![FLAT_14; w * h];
    let high = vec![i16::MAX; w * h];
    // Default bi-prediction at 10 bits: shift2 = 15 − 10, offset2 = 1 << 4.
    let default = HevcWeights {
        w0: 1,
        w1: 1,
        round: 1 << 4,
        shift: 5,
    };
    for (input, expected) in [(&flat, FLAT), (&high, TEN_MAX)] {
        let mut out = vec![0u16; w * h];
        let mut view = PlaneViewMut::new(&mut out, w, w, h).unwrap();
        ten.hevc_weighted_prediction(default, input, input, &mut view);
        assert!(out.iter().all(|&v| v == expected));

        let mut out = vec![0u16; w * h];
        let mask = VvcGeometric::new((w, h), 11);
        ten.vvc_geometric_blend(mask, (0, 0), (w, h), false, (input, input), &mut out);
        assert!(out.iter().all(|&v| v == expected));
    }
    let count = (w + 2) * (h + 2);
    let padded = vec![FLAT_14; count];
    let mut out = vec![0u16; w * h];
    ten.vvc_bdof((w, h), [&padded, &padded], &mut out);
    assert!(out.iter().all(|&v| v == FLAT));
    let noisy = [predictions(count, 4), predictions(count, 5)];
    let noisy = noisy.map(|p| p.iter().map(|&v| v.saturating_mul(2)).collect::<Vec<_>>());
    ten.vvc_bdof((w, h), [&noisy[0], &noisy[1]], &mut out);
    assert!(out.iter().all(|&v| v <= TEN_MAX));
}
