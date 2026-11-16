//! The VP9 `u16` routes at 8 bits equal the 8-bit kernels (one source for
//! both depths), family by family, and stay inside the depth at 10 and 12
//! bits.

use nc_common::{BitDepth, Sample};

use crate::{
    Dsp, VP9_MAX_SCALED_STEP, VP9_TAPS, Vp9Filter, Vp9FilterLimits, Vp9FilterWidth, Vp9IntraEdges,
    Vp9IntraMode, Vp9Pass, Vp9ScaledScratch,
};

/// A deterministic pattern of samples up to `max`, with flat runs and
/// edges (so the filters' decisions take both ways).
fn pattern(len: usize, seed: usize, max: u32) -> Vec<u32> {
    (0..len)
        .map(|i| {
            let v = (i.wrapping_mul(2_654_435_761) + seed * 40_503) >> 7;
            match (i / 11 + seed) % 9 {
                0 => max,
                1 | 2 => max / 2 + (seed as u32 % 3),
                _ => (v as u32) % (max + 1),
            }
        })
        .collect()
}

fn samples<T: Sample>(values: &[u32]) -> Vec<T> {
    values.iter().map(|&v| T::from_i32(v as i32)).collect()
}

fn widened(values: &[u8]) -> Vec<u16> {
    values.iter().map(|&v| u16::from(v)).collect()
}

const TWELVE: BitDepth = BitDepth::new(12).unwrap();
const DEEP: [BitDepth; 2] = [BitDepth::TEN, TWELVE];

fn edges<T: Sample>(seed: usize, side: usize, max: u32) -> Vp9IntraEdges<T> {
    let values = pattern(3 * 32 + 1, seed, max);
    let mut edges = Vp9IntraEdges {
        above: std::array::from_fn(|i| T::from_i32(values[i] as i32)),
        left: std::array::from_fn(|i| T::from_i32(values[65 + i] as i32)),
        has_above: seed % 4 != 1,
        has_left: seed % 4 != 2,
    };
    if side > 4 {
        // The above-right contract past 4x4.
        let last = edges.above[side];
        edges.above[side + 1..=2 * side].fill(last);
    }
    edges
}

#[test]
fn intra_at_eight_bits_matches_the_eight_bit_kernel() {
    let dsp = Dsp::new();
    let kernels = dsp.at(BitDepth::EIGHT);
    for seed in 0..12 {
        for side in [4, 8, 16, 32] {
            for mode in Vp9IntraMode::ALL {
                let narrow: Vp9IntraEdges<u8> = edges(seed, side, 255);
                let wide = Vp9IntraEdges {
                    above: narrow.above.map(u16::from),
                    left: narrow.left.map(u16::from),
                    has_above: narrow.has_above,
                    has_left: narrow.has_left,
                };
                let (mut want, mut got) = ([0u8; 1024], [0u16; 1024]);
                kernels.vp9_intra_predict(mode, side, &narrow, &mut want);
                kernels.vp9_intra_predict(mode, side, &wide, &mut got);
                assert_eq!(widened(&want), got, "{mode:?} {side} seed {seed}");
            }
        }
    }
}

#[test]
fn deep_intra_stays_inside_the_depth() {
    let dsp = Dsp::new();
    for depth in DEEP {
        let max = depth.max() as u32;
        for side in [4, 8, 16, 32] {
            for mode in Vp9IntraMode::ALL {
                let edges: Vp9IntraEdges<u16> = edges(side, side, max);
                let mut out = [0u16; 1024];
                dsp.at(depth)
                    .vp9_intra_predict(mode, side, &edges, &mut out);
                assert!(out.iter().all(|&s| u32::from(s) <= max), "{mode:?} {side}");
            }
        }
    }
}

const FILTERS: [Vp9Filter; 4] = [
    Vp9Filter::Regular,
    Vp9Filter::Smooth,
    Vp9Filter::Sharp,
    Vp9Filter::Bilinear,
];

#[test]
fn convolution_at_eight_bits_matches_the_eight_bit_kernels() {
    let dsp = Dsp::new();
    let kernels = dsp.at(BitDepth::EIGHT);
    let stride = 64 + VP9_TAPS;
    let window = pattern(stride * stride, 3, 255);
    let narrow = samples::<u8>(&window);
    let wide = samples::<u16>(&window);
    for (i, kind) in FILTERS.into_iter().enumerate() {
        for size in [(4, 4), (8, 4), (16, 16), (32, 64), (64, 64)] {
            for phase in [(0, 0), (5, 0), (0, 9), (3 + i, 15 - i)] {
                for average in [false, true] {
                    let base = pattern(64 * 64, i, 255);
                    let mut want = samples::<u8>(&base);
                    let mut got = samples::<u16>(&base);
                    let source = (&narrow[..], stride);
                    kernels.vp9_convolve(kind, source, phase, size, (&mut want, 64), average);
                    kernels.vp9_convolve(
                        kind,
                        (&wide[..], stride),
                        phase,
                        size,
                        (&mut got, 64),
                        average,
                    );
                    assert_eq!(widened(&want), got, "{kind:?} {size:?} {phase:?}");
                }
            }
        }
    }
}

#[test]
fn scaled_convolution_at_eight_bits_matches_the_eight_bit_driver() {
    let dsp = Dsp::new();
    let kernels = dsp.at(BitDepth::EIGHT);
    let stride = 140;
    let window = pattern(stride * 150, 5, 255);
    let narrow = samples::<u8>(&window);
    let wide = samples::<u16>(&window);
    let mut scratch = Vp9ScaledScratch::default();
    for kind in FILTERS {
        for (width, height) in [(4, 4), (8, 16), (64, 64)] {
            for steps in [((0, 16), (7, 20)), ((3, 32), (0, 32)), ((15, 24), (9, 8))] {
                assert!(steps.0.1 <= VP9_MAX_SCALED_STEP);
                for average in [false, true] {
                    let base = pattern(64 * 64, width, 255);
                    let mut want = samples::<u8>(&base);
                    let mut got = samples::<u16>(&base);
                    kernels.vp9_convolve_scaled(
                        kind,
                        (&narrow[..], stride),
                        steps,
                        (width, height),
                        (&mut want, 64),
                        average,
                        &mut scratch,
                    );
                    kernels.vp9_convolve_scaled(
                        kind,
                        (&wide[..], stride),
                        steps,
                        (width, height),
                        (&mut got, 64),
                        average,
                        &mut scratch,
                    );
                    assert_eq!(widened(&want), got, "{kind:?} {width}x{height} {steps:?}");
                }
            }
        }
    }
}

#[test]
fn deep_convolution_stays_inside_the_depth() {
    let dsp = Dsp::new();
    let stride = 64 + VP9_TAPS;
    for depth in DEEP {
        let max = depth.max() as u32;
        let window = samples::<u16>(&pattern(stride * stride, 9, max));
        for kind in FILTERS {
            let mut out = vec![0u16; 64 * 64];
            dsp.at(depth).vp9_convolve(
                kind,
                (&window, stride),
                (7, 9),
                (64, 64),
                (&mut out, 64),
                false,
            );
            assert!(out.iter().all(|&s| u32::from(s) <= max));
        }
    }
}

/// Sparse coefficients small enough that no 8-bit stage wraps, so the
/// 8-bit (16-bit stages) and deep (32-bit stages) inverses agree.
fn coefficients(side: usize, seed: usize, magnitude: i32) -> Vec<i32> {
    pattern(side * side, seed, 2 * magnitude as u32)
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            if i % 7 == seed % 7 || i < 3 {
                v as i32 - magnitude
            } else {
                0
            }
        })
        .collect()
}

const PASSES: [(Vp9Pass, Vp9Pass); 4] = [
    (Vp9Pass::Dct, Vp9Pass::Dct),
    (Vp9Pass::Adst, Vp9Pass::Dct),
    (Vp9Pass::Dct, Vp9Pass::Adst),
    (Vp9Pass::Adst, Vp9Pass::Adst),
];

#[test]
fn deep_inverse_at_eight_bits_matches_the_eight_bit_kernel() {
    let dsp = Dsp::new();
    let kernels = dsp.at(BitDepth::EIGHT);
    for log2 in 2..=5u8 {
        let side = 1 << log2;
        for passes in PASSES {
            if log2 == 5 && passes != (Vp9Pass::Dct, Vp9Pass::Dct) {
                continue;
            }
            for seed in 0..6 {
                let input = coefficients(side, seed, 300);
                let narrow: Vec<i16> = input.iter().map(|&c| c as i16).collect();
                let base = pattern(side * side, seed, 255);
                let mut want = samples::<u8>(&base);
                let mut got = samples::<u16>(&base);
                kernels.vp9_inverse_add(log2, passes, &narrow, (side, side), (&mut want, side));
                kernels.vp9_inverse_add(log2, passes, &input, (side, side), (&mut got, side));
                assert_eq!(widened(&want), got, "{side} {passes:?} seed {seed}");
            }
        }
    }
}

#[test]
fn deep_inverse_stays_inside_the_depth() {
    let dsp = Dsp::new();
    for depth in DEEP {
        let max = depth.max() as u32;
        for log2 in 2..=5u8 {
            let side = 1 << log2;
            // Coefficients of the depth's range (8 + BitDepth bits).
            let input = coefficients(side, log2.into(), 1 << (depth.bits() + 6));
            let mut out = samples::<u16>(&pattern(side * side, 1, max));
            dsp.at(depth).vp9_inverse_add(
                log2,
                (Vp9Pass::Dct, Vp9Pass::Dct),
                &input,
                (side, side),
                (&mut out, side),
            );
            assert!(out.iter().all(|&s| u32::from(s) <= max), "{depth:?} {side}");
        }
    }
}

const WIDTHS: [Vp9FilterWidth; 3] = [
    Vp9FilterWidth::Four,
    Vp9FilterWidth::Eight,
    Vp9FilterWidth::Sixteen,
];

#[test]
fn deblocking_at_eight_bits_matches_the_eight_bit_kernel() {
    let dsp = Dsp::new();
    let kernels = dsp.at(BitDepth::EIGHT);
    let stride = 32;
    for seed in 0..16 {
        for width in WIDTHS {
            for vertical in [false, true] {
                let limits = Vp9FilterLimits::with_sharpness(1 + (seed as u8 * 4) % 63, 0);
                // Gentle slopes so that the flat filters engage too.
                let base: Vec<u32> = pattern(stride * stride, seed, 6)
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| 100 + v + (i % stride) as u32 / 4)
                    .collect();
                let mut want = samples::<u8>(&base);
                let mut got = samples::<u16>(&base);
                let (step, along) = if vertical { (1, stride) } else { (stride, 1) };
                let at = (8 * stride + 8, step, along);
                kernels.vp9_filter_edge(&mut want, at, 16, width, limits);
                kernels.vp9_filter_edge(&mut got, at, 16, width, limits);
                assert_eq!(widened(&want), got, "{width:?} seed {seed}");
            }
        }
    }
}

#[test]
fn deep_deblocking_stays_inside_the_depth() {
    let dsp = Dsp::new();
    let stride = 32;
    for depth in DEEP {
        let max = depth.max() as u32;
        for width in WIDTHS {
            let mut data = samples::<u16>(&pattern(stride * stride, 2, max));
            let limits = Vp9FilterLimits::new(63);
            dsp.at(depth).vp9_filter_edge(
                &mut data,
                (8 * stride + 8, 1, stride),
                16,
                width,
                limits,
            );
            assert!(data.iter().all(|&s| u32::from(s) <= max));
        }
    }
}

#[test]
fn deep_deblocking_in_line_groups_matches_the_per_line_form() {
    let stride = 48;
    for depth in DEEP {
        let max = depth.max() as u32;
        for seed in 0..24 {
            for width in WIDTHS {
                for vertical in [false, true] {
                    let level = 1 + (seed as u8 * 5) % 63;
                    let limits = Vp9FilterLimits::with_sharpness(level, seed as u8 % 8);
                    // Gentle slopes (flat filters) and noise (narrow or none).
                    let spread = if seed % 3 == 0 {
                        max
                    } else {
                        6 << depth.above_eight()
                    };
                    let base: Vec<u32> = pattern(stride * stride, seed, spread)
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| (v + (i % stride) as u32).min(max))
                        .collect();
                    let mut want = samples::<u16>(&base);
                    let mut got = want.clone();
                    let (step, along) = if vertical { (1, stride) } else { (stride, 1) };
                    let at = (8 * stride + 8, step, along);
                    let lines = 1 + seed % 20;
                    crate::vp9_deblock::vp9_filter_edge_at(
                        &mut want, at, lines, width, limits, depth,
                    );
                    crate::vp9_deblock::filter_lines_at(&mut got, at, lines, width, limits, depth);
                    assert_eq!(want, got, "{depth:?} {width:?} seed {seed}");
                }
            }
        }
    }
}
