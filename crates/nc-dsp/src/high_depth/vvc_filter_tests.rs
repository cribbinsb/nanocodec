//! The VVC filter kernels of `Dsp::at`: at 8 bits the generic references
//! (through the `u16` table) equal the 8-bit `Dsp` methods with their SIMD
//! tiers; at 10 bits they keep the depth's thresholds, clips and scale.

use nc_common::{BitDepth, PlaneViewMut};

use crate::{
    Dsp, VVC_ALF_BLOCK, VVC_ALF_TAPS, vvc_alf_clip_value, vvc_deblock_beta, vvc_deblock_tc,
};

/// A deterministic pseudo-random sequence in `0..=max`, with runs of
/// near-equal values (so that deblocking decisions take every branch).
fn noise(seed: u32, len: usize, max: u32) -> Vec<u32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut level = max / 2;
    (0..len)
        .map(|i| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let r = state >> 8;
            if i % 13 == 0 {
                level = r % (max + 1);
            }
            match r % 7 {
                0 => r % (max + 1),
                1 => max,
                _ => (level + r % 5).min(max),
            }
        })
        .collect()
}

fn bytes(values: &[u32]) -> Vec<u8> {
    values.iter().map(|&v| v as u8).collect()
}

fn words(values: &[u32]) -> Vec<u16> {
    values.iter().map(|&v| v as u16).collect()
}

fn widen(values: &[u8]) -> Vec<u16> {
    values.iter().map(|&v| u16::from(v)).collect()
}

/// The legal 8-bit `β`, `tC` pairs exercised: Table 43 extremes and middles.
const THRESHOLDS: [(i32, i32); 5] = [(0, 0), (6, 1), (38, 7), (64, 25), (88, 99)];

/// Every legal pair of luma reaches.
const REACHES: [[usize; 2]; 11] = [
    [1, 1],
    [2, 2],
    [3, 3],
    [3, 5],
    [3, 7],
    [5, 3],
    [5, 5],
    [5, 7],
    [7, 3],
    [7, 5],
    [7, 7],
];

#[test]
fn deblocking_at_eight_bits_matches_the_eight_bit_kernels() {
    let dsp = Dsp::new();
    let eight = dsp.at(BitDepth::EIGHT);
    let side = 32;
    for seed in 0..12 {
        let source = noise(seed, side * side, 255);
        for (beta, tc) in THRESHOLDS {
            for lengths in REACHES {
                for vertical in [false, true] {
                    let mut plane8 = bytes(&source);
                    let mut plane16 = words(&source);
                    let at = (8, 12);
                    dsp.vvc_deblock_luma_segment(
                        &mut PlaneViewMut::new(&mut plane8, side, side, side).unwrap(),
                        at,
                        vertical,
                        beta,
                        tc,
                        lengths,
                    );
                    eight.vvc_deblock_luma_segment(
                        &mut PlaneViewMut::new(&mut plane16, side, side, side).unwrap(),
                        at,
                        vertical,
                        beta,
                        tc,
                        lengths,
                    );
                    assert_eq!(
                        widen(&plane8),
                        plane16,
                        "luma segment {beta} {tc} {lengths:?}"
                    );
                }
                let rows = &source[..64];
                let mut edge8: [[u8; 16]; 4] =
                    std::array::from_fn(|r| std::array::from_fn(|i| rows[16 * r + i] as u8));
                let mut edge16 = edge8.map(|row| row.map(u16::from));
                dsp.vvc_deblock_luma(&mut edge8, beta, tc, lengths);
                eight.vvc_deblock_luma(&mut edge16, beta, tc, lengths);
                assert_eq!(
                    edge8.map(|row| row.map(u16::from)),
                    edge16,
                    "luma {lengths:?}"
                );
            }
            for (large, boundary) in [(false, false), (true, false), (true, true)] {
                for (lines, vertical) in [(2, false), (4, true), (8, false), (6, true)] {
                    let mut plane8 = bytes(&source);
                    let mut plane16 = words(&source);
                    let at = (4, 8);
                    dsp.vvc_deblock_chroma_edge(
                        &mut PlaneViewMut::new(&mut plane8, side, side, side).unwrap(),
                        at,
                        vertical,
                        lines,
                        beta,
                        tc,
                        large,
                        boundary,
                    );
                    eight.vvc_deblock_chroma_edge(
                        &mut PlaneViewMut::new(&mut plane16, side, side, side).unwrap(),
                        at,
                        vertical,
                        lines,
                        beta,
                        tc,
                        large,
                        boundary,
                    );
                    assert_eq!(widen(&plane8), plane16, "chroma edge {beta} {tc} {lines}");
                }
                let mut edge8: [[u8; 8]; 2] =
                    std::array::from_fn(|r| std::array::from_fn(|i| source[8 * r + i] as u8));
                let mut edge16 = edge8.map(|row| row.map(u16::from));
                dsp.vvc_deblock_chroma(&mut edge8, beta, tc, large, boundary);
                eight.vvc_deblock_chroma(&mut edge16, beta, tc, large, boundary);
                assert_eq!(edge8.map(|row| row.map(u16::from)), edge16, "chroma");
            }
        }
    }
}

#[test]
fn deblocking_thresholds_scale_with_the_depth() {
    let ten = BitDepth::TEN;
    let nine = BitDepth::new(9).unwrap();
    // β′ = 88, tC′ = 395 (Table 43 at Q = 63 and 65).
    assert_eq!(vvc_deblock_beta(88, BitDepth::EIGHT), 88);
    assert_eq!(vvc_deblock_beta(88, ten), 352);
    assert_eq!(vvc_deblock_tc(395, BitDepth::EIGHT), 99);
    assert_eq!(vvc_deblock_tc(395, nine), 198);
    assert_eq!(vvc_deblock_tc(395, ten), 395);
    assert_eq!(vvc_deblock_tc(395, BitDepth::new(12).unwrap()), 1580);
}

#[test]
fn ten_bit_deblocking_smooths_inside_the_depth() {
    let dsp = Dsp::new();
    let ten = dsp.at(BitDepth::TEN);
    let (beta, tc) = (
        vvc_deblock_beta(88, BitDepth::TEN),
        vvc_deblock_tc(395, BitDepth::TEN),
    );
    // A flat field stays flat.
    let mut edge = [[700u16; 16]; 4];
    ten.vvc_deblock_luma(&mut edge, beta, tc, [7, 7]);
    assert_eq!(edge, [[700; 16]; 4]);
    // A small step between two flat sides is smoothed across, monotonically,
    // with values beyond 8 bits.
    let mut edge: [[u16; 16]; 4] = [std::array::from_fn(|i| if i < 8 { 600 } else { 640 }); 4];
    ten.vvc_deblock_luma(&mut edge, beta, tc, [7, 7]);
    for row in edge {
        assert!(row.windows(2).all(|w| w[0] <= w[1]), "{row:?}");
        assert!(row[7] > 600 && row[8] < 640, "{row:?}");
    }
    let mut chroma: [[u16; 8]; 2] = [std::array::from_fn(|i| if i < 4 { 900 } else { 960 }); 2];
    ten.vvc_deblock_chroma(&mut chroma, beta, tc, true, false);
    for row in chroma {
        assert!(row.windows(2).all(|w| w[0] <= w[1]) && row[3] > 900 && row[4] < 960);
    }
    // Extreme input stays inside 0..=1023, in place as gathered.
    let side = 32;
    let source = noise(5, side * side, 1023);
    for lengths in REACHES {
        let mut plane = words(&source);
        ten.vvc_deblock_luma_segment(
            &mut PlaneViewMut::new(&mut plane, side, side, side).unwrap(),
            (8, 8),
            true,
            beta,
            tc,
            lengths,
        );
        assert!(plane.iter().all(|&v| v <= 1023));
    }
    let mut plane = words(&source);
    ten.vvc_deblock_chroma_edge(
        &mut PlaneViewMut::new(&mut plane, side, side, side).unwrap(),
        (4, 8),
        false,
        8,
        beta,
        tc,
        true,
        true,
    );
    assert!(plane.iter().all(|&v| v <= 1023));
}

#[test]
#[should_panic(expected = "out of range")]
fn ten_bit_deblocking_rejects_thresholds_beyond_the_depth() {
    let dsp = Dsp::new();
    let mut edge = [[0u16; 16]; 4];
    dsp.at(BitDepth::TEN).vvc_deblock_luma(
        &mut edge,
        vvc_deblock_beta(88, BitDepth::TEN) + 1,
        0,
        [3, 3],
    );
}

/// Seven rows of `width + 6` samples from `seed`, at most `max`.
fn alf_rows(seed: u32, width: usize, max: u32) -> Vec<Vec<u32>> {
    (0..7)
        .map(|r| noise(seed * 7 + r, width + 6, max))
        .collect()
}

/// Coefficients within the signalled range, one set per 4-sample block.
fn alf_coefficients(width: usize, seed: usize) -> Vec<[i16; VVC_ALF_TAPS]> {
    (0..width.div_ceil(VVC_ALF_BLOCK))
        .map(|b| std::array::from_fn(|k| ((b * 37 + k * 13 + seed * 5) % 97) as i16 - 48))
        .collect()
}

#[test]
fn alf_at_eight_bits_matches_the_eight_bit_kernels() {
    let dsp = Dsp::new();
    let eight = dsp.at(BitDepth::EIGHT);
    let clip8: [i16; 4] = std::array::from_fn(|i| vvc_alf_clip_value(i, BitDepth::EIGHT));
    for (seed, width) in [4, 8, 12, 16, 28, 64, 68].into_iter().enumerate() {
        let seed = seed as u32;
        let data = alf_rows(seed, width, 255);
        let (data8, data16): (Vec<_>, Vec<_>) = data.iter().map(|r| (bytes(r), words(r))).unzip();
        let rows8: [&[u8]; 7] = std::array::from_fn(|r| data8[r].as_slice());
        let rows16: [&[u16]; 7] = std::array::from_fn(|r| data16[r].as_slice());
        let coefficients = alf_coefficients(width, seed as usize);
        let limits: Vec<[i16; VVC_ALF_TAPS]> = (0..coefficients.len())
            .map(|b| std::array::from_fn(|k| clip8[(b + k) % 4]))
            .collect();

        let block8: [[&[u8]; 7]; VVC_ALF_BLOCK] =
            std::array::from_fn(|r| std::array::from_fn(|i| rows8[(i + r) % 7]));
        let block16: [[&[u16]; 7]; VVC_ALF_BLOCK] =
            std::array::from_fn(|r| std::array::from_fn(|i| rows16[(i + r) % 7]));
        let shifts = [7, 10, 7, 9];
        let mut out8: [Vec<u8>; VVC_ALF_BLOCK] = std::array::from_fn(|_| vec![0; width]);
        let mut out16: [Vec<u16>; VVC_ALF_BLOCK] = std::array::from_fn(|_| vec![0; width]);
        dsp.vvc_alf_filter_block_row(
            &block8,
            &coefficients,
            &shifts,
            &mut out8.each_mut().map(Vec::as_mut_slice),
        );
        eight.vvc_alf_filter_block_row(
            &block16,
            &coefficients,
            &shifts,
            &mut out16.each_mut().map(Vec::as_mut_slice),
        );
        for (a, b) in out8.iter().zip(&out16) {
            assert_eq!(&widen(a), b, "block row width {width}");
        }

        for shift in [7, 10] {
            let mut out8 = vec![0u8; width];
            let mut out16 = vec![0u16; width];
            dsp.vvc_alf_filter_clipped_row(&rows8, &coefficients, &limits, shift, &mut out8);
            eight.vvc_alf_filter_clipped_row(&rows16, &coefficients, &limits, shift, &mut out16);
            assert_eq!(widen(&out8), out16, "clipped width {width}");
            let chroma = std::array::from_fn(|k| coefficients[0][k]);
            let chroma_limits = std::array::from_fn(|k| limits[0][k]);
            let mut out8 = vec![0u8; width];
            let mut out16 = vec![0u16; width];
            dsp.vvc_alf_filter_chroma_row(&rows8, chroma, chroma_limits, shift, &mut out8);
            eight.vvc_alf_filter_chroma_row(&rows16, chroma, chroma_limits, shift, &mut out16);
            assert_eq!(widen(&out8), out16, "chroma width {width}");
        }

        let groups = width / 4 + 1;
        let lrows8: [&[u8]; 4] = std::array::from_fn(|r| rows8[r]);
        let lrows16: [&[u16]; 4] = std::array::from_fn(|r| rows16[r]);
        let mut sums8 = vec![[0; 4]; groups];
        let mut sums16 = vec![[0; 4]; groups];
        dsp.vvc_alf_laplacians(&lrows8, &mut sums8);
        eight.vvc_alf_laplacians(&lrows16, &mut sums16);
        assert_eq!(sums8, sums16, "laplacians width {width}");
        // Four grid rows' sums, as the decoder adds them up.
        let block_sums: Vec<[u16; 4]> = sums8.iter().map(|g| g.map(|v| 4 * v)).collect();
        for boundary in [false, true] {
            let mut classes8 = vec![[0; 2]; groups - 1];
            let mut classes16 = vec![[0; 2]; groups - 1];
            dsp.vvc_alf_classify(&block_sums, boundary, &mut classes8);
            eight.vvc_alf_classify::<u16>(&block_sums, boundary, &mut classes16);
            assert_eq!(classes8, classes16, "classify width {width}");
        }

        let cross: [i16; 7] =
            std::array::from_fn(|k| (k as i16 * 29 + seed as i16 * 11) % 129 - 64);
        let half = width / 2;
        let base = noise(seed + 100, half, 255);
        let mut out8 = bytes(&base);
        let mut out16 = words(&base);
        dsp.vvc_alf_cross_row(&lrows8, cross, &mut out8);
        eight.vvc_alf_cross_row(&lrows16, cross, &mut out16);
        assert_eq!(widen(&out8), out16, "cross width {width}");
    }
}

#[test]
fn alf_clipping_values_follow_the_depth() {
    let at = |depth| std::array::from_fn::<_, 4, _>(|i| vvc_alf_clip_value(i, depth));
    assert_eq!(at(BitDepth::EIGHT), [256, 32, 8, 2]);
    assert_eq!(at(BitDepth::TEN), [1024, 128, 32, 8]);
}

#[test]
fn ten_bit_alf_scales_with_the_samples() {
    // Samples four times an 8-bit picture's: the Laplacians scale by four,
    // the classes (activity shifted by BitDepth + 4) stay the same, and the
    // unclipped (AlfClip[0]) nonlinear filter equals the linear one.
    let dsp = Dsp::new();
    let (eight, ten) = (dsp.at(BitDepth::EIGHT), dsp.at(BitDepth::TEN));
    let width = 64;
    let data = alf_rows(3, width, 255);
    let data8: Vec<Vec<u8>> = data.iter().map(|r| bytes(r)).collect();
    let data10: Vec<Vec<u16>> = data
        .iter()
        .map(|r| r.iter().map(|&v| (4 * v) as u16).collect())
        .collect();
    let lrows8: [&[u8]; 4] = std::array::from_fn(|r| data8[r].as_slice());
    let lrows10: [&[u16]; 4] = std::array::from_fn(|r| data10[r].as_slice());
    let groups = width / 4 + 1;
    let mut sums8 = vec![[0; 4]; groups];
    let mut sums10 = vec![[0; 4]; groups];
    eight.vvc_alf_laplacians(&lrows8, &mut sums8);
    ten.vvc_alf_laplacians(&lrows10, &mut sums10);
    assert_eq!(
        sums8.iter().map(|g| g.map(|v| 4 * v)).collect::<Vec<_>>(),
        sums10
    );
    let block = |sums: &[[u16; 4]]| sums.iter().map(|g| g.map(|v| 4 * v)).collect::<Vec<_>>();
    for boundary in [false, true] {
        let mut classes8 = vec![[0; 2]; groups - 1];
        let mut classes10 = vec![[0; 2]; groups - 1];
        eight.vvc_alf_classify::<u8>(&block(&sums8), boundary, &mut classes8);
        ten.vvc_alf_classify::<u16>(&block(&sums10), boundary, &mut classes10);
        assert_eq!(classes8, classes10);
    }

    let rows10: [&[u16]; 7] = std::array::from_fn(|r| data10[r].as_slice());
    let coefficients = alf_coefficients(width, 1);
    let unclipped = vec![[vvc_alf_clip_value(0, BitDepth::TEN); VVC_ALF_TAPS]; coefficients.len()];
    let block_rows: [[&[u16]; 7]; VVC_ALF_BLOCK] = [rows10; VVC_ALF_BLOCK];
    let mut linear: [Vec<u16>; VVC_ALF_BLOCK] = std::array::from_fn(|_| vec![0; width]);
    ten.vvc_alf_filter_block_row(
        &block_rows,
        &coefficients,
        &[7; VVC_ALF_BLOCK],
        &mut linear.each_mut().map(Vec::as_mut_slice),
    );
    let mut clipped = vec![0u16; width];
    ten.vvc_alf_filter_clipped_row(&rows10, &coefficients, &unclipped, 7, &mut clipped);
    assert_eq!(linear[0], clipped);
    assert!(clipped.iter().all(|&v| v <= 1023) && clipped.iter().any(|&v| v > 255));
}

#[test]
fn ten_bit_alf_keeps_flat_areas_and_the_depth() {
    let dsp = Dsp::new();
    let ten = dsp.at(BitDepth::TEN);
    let flat = vec![777u16; 70];
    let rows: [&[u16]; 7] = [flat.as_slice(); 7];
    let coefficients = alf_coefficients(64, 2);
    let limits = vec![[vvc_alf_clip_value(2, BitDepth::TEN); VVC_ALF_TAPS]; 16];
    let mut out = vec![0u16; 64];
    ten.vvc_alf_filter_clipped_row(&rows, &coefficients, &limits, 10, &mut out);
    assert!(out.iter().all(|&v| v == 777));
    let mut out = vec![0u16; 64];
    ten.vvc_alf_filter_chroma_row(
        &rows,
        [9, -3, 20, -7, 11, 4],
        [1024, 128, 32, 8, 1024, 128],
        7,
        &mut out,
    );
    assert!(out.iter().all(|&v| v == 777));
    // Extreme samples stay inside 0..=1023.
    let data: Vec<Vec<u16>> = alf_rows(9, 64, 1023).iter().map(|r| words(r)).collect();
    let rows: [&[u16]; 7] = std::array::from_fn(|r| data[r].as_slice());
    let limits = vec![[vvc_alf_clip_value(0, BitDepth::TEN); VVC_ALF_TAPS]; 16];
    ten.vvc_alf_filter_clipped_row(&rows, &coefficients, &limits, 7, &mut out);
    assert!(out.iter().all(|&v| v <= 1023));
}

#[test]
#[should_panic(expected = "assertion failed")]
fn ten_bit_alf_rejects_eight_bit_clipping_values() {
    let dsp = Dsp::new();
    let flat = vec![0u16; 10];
    let rows: [&[u16]; 7] = [flat.as_slice(); 7];
    let mut out = [0u16; 4];
    dsp.at(BitDepth::TEN)
        .vvc_alf_filter_chroma_row(&rows, [0; 6], [256; 6], 7, &mut out);
}

#[test]
fn ten_bit_cross_component_clips_the_correction_to_the_depth() {
    let dsp = Dsp::new();
    let ten = dsp.at(BitDepth::TEN);
    // Chroma sample 0 reads luma column 1: centre 0 below a 1023 above.
    let upper = [0, 1023, 0, 0];
    let middle = [1023, 0, 1023, 1023];
    let lower = [1023; 4];
    let rows = [&upper[..], &middle[..], &lower[..], &lower[..]];
    // 64 · 1023 rounds to 512, clipped to 511 (2^(10 − 1) − 1).
    let mut out = [300u16];
    ten.vvc_alf_cross_row(&rows, [64, 0, 0, 0, 0, 0, 0], &mut out);
    assert_eq!(out, [811]);
    // All taps positive: far past +511, the result clips to 1023.
    let mut out = [600u16];
    ten.vvc_alf_cross_row(&rows, [64; 7], &mut out);
    assert_eq!(out, [1023]);
    // Centre 1023, neighbours 0: the correction clips to −512.
    let zeros = [0u16; 4];
    let centre = [0, 1023, 0, 0];
    let rows = [&zeros[..], &centre[..], &zeros[..], &zeros[..]];
    let mut out = [1000u16];
    ten.vvc_alf_cross_row(&rows, [64; 7], &mut out);
    assert_eq!(out, [488]);
}
