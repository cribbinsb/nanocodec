use super::*;
use crate::{VvcQuantizer, VvcTransformKind};

/// A deterministic xorshift stream.
struct Stream(u32);

impl Stream {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

/// The exact low half of the 64-point forward with the HEVC / `VVenC`
/// scaling (rows shifted by 5, columns by 12, clamped to 16 bits): the
/// reference the approximate forward is measured against.
fn exact_forward(residual: &[i16], out: &mut [i32]) {
    let round = |sum: i64, shift: u32| (sum + (1 << (shift - 1))) >> shift;
    let mut rows = [[0i64; CODED_SIDE]; BLOCK_SIDE];
    for (y, samples) in residual.as_chunks::<BLOCK_SIDE>().0.iter().enumerate() {
        for (k, basis) in DCT64.iter().enumerate() {
            let sum = samples
                .iter()
                .zip(basis)
                .map(|(&r, &b)| i64::from(r) * i64::from(b))
                .sum();
            rows[y][k] = round(sum, 5);
        }
    }
    for (k, basis) in DCT64.iter().enumerate() {
        for x in 0..CODED_SIDE {
            let sum = (0..BLOCK_SIDE)
                .map(|y| rows[y][x] * i64::from(basis[y]))
                .sum();
            out[k * CODED_SIDE + x] = round(sum, 12).clamp(-32768, 32767) as i32;
        }
    }
}

/// A 64x64 source/prediction pair: a smooth gradient plus `noise` (peak)
/// against a flat prediction.
fn blocks(stream: &mut Stream, noise: u32) -> (Vec<u8>, Vec<u8>) {
    let source = (0..BLOCK_AREA)
        .map(|i| {
            let (x, y) = ((i % BLOCK_SIDE) as i32, (i / BLOCK_SIDE) as i32);
            let smooth = 128 + (x - 32) * 2 - (y - 32) + ((x * y) >> 6);
            let jitter = (stream.next() % (2 * noise + 1)) as i32 - noise as i32;
            (smooth + jitter).clamp(0, 255) as u8
        })
        .collect();
    let prediction = (0..BLOCK_AREA)
        .map(|_| 124 + (stream.next() % (noise / 4 + 1)) as u8)
        .collect();
    (source, prediction)
}

fn view(samples: &[u8]) -> PlaneView<'_> {
    PlaneView::new(samples, BLOCK_SIDE, BLOCK_SIDE, BLOCK_SIDE).unwrap()
}

/// The approximate and exact coefficients of a block pair.
fn both(dsp: &Dsp, source: &[u8], prediction: &[u8]) -> (Vec<i32>, Vec<i32>) {
    let residual: Vec<i16> = source
        .iter()
        .zip(prediction)
        .map(|(&s, &p)| i16::from(s) - i16::from(p))
        .collect();
    let mut exact = vec![0; CODED_AREA];
    exact_forward(&residual, &mut exact);
    let mut approximate = vec![0; CODED_AREA];
    assert!(dsp.vvc_forward64(
        &view(source),
        &view(prediction),
        0,
        HevcForwardMode::EXACT,
        &mut approximate
    ));
    (approximate, exact)
}

#[test]
fn the_halved_residual_matches_its_definition_on_every_isa() {
    let mut stream = Stream(7);
    let (source, prediction) = blocks(&mut stream, 120);
    let mut expected = [0i16; CODED_AREA];
    // SAFETY: the scalar reference needs no ISA.
    unsafe { halve(&source, BLOCK_SIDE, &prediction, BLOCK_SIDE, &mut expected) };
    for (i, &value) in expected.iter().enumerate() {
        let (x, y) = (2 * (i % CODED_SIDE), 2 * (i / CODED_SIDE));
        let sum: i32 = [(0, 0), (1, 0), (0, 1), (1, 1)]
            .iter()
            .map(|&(dx, dy)| {
                let p = (y + dy) * BLOCK_SIDE + x + dx;
                i32::from(source[p]) - i32::from(prediction[p])
            })
            .sum();
        assert_eq!(i32::from(value), (sum + 2).div_euclid(4), "{i}");
    }
    let dsp = Dsp::new();
    let mut got = [0i16; CODED_AREA];
    // SAFETY: the table matches this CPU.
    unsafe { (dsp.vvc_tx64)(&source, BLOCK_SIDE, &prediction, BLOCK_SIDE, &mut got) };
    assert_eq!(got, expected);
}

/// Mean absolute error of the approximate low half against the exact one,
/// as a fraction of the exact coefficients' mean magnitude, below and from
/// diagonal 16 (`u + v`), of the approximate forward with and without
/// the repair, over eight blocks of `noise`.
fn relative_errors(dsp: &Dsp, noise: u32) -> [[f64; 2]; 2] {
    let mut stream = Stream(0x9e37_79b9);
    let mut error = [[0f64; 2]; 2];
    let mut magnitude = [0f64; 2];
    for _ in 0..8 {
        let (source, prediction) = blocks(&mut stream, noise);
        let (approximate, exact) = both(dsp, &source, &prediction);
        let mut halved = [0i16; CODED_AREA];
        // SAFETY: the scalar reference needs no ISA.
        unsafe { halve(&source, BLOCK_SIDE, &prediction, BLOCK_SIDE, &mut halved) };
        let mut raw = vec![0; CODED_AREA];
        dsp.hevc_forward(
            HevcTransform::dct(5).unwrap(),
            &halved,
            CODED_SIDE,
            &mut raw,
        );
        for (i, &e) in exact.iter().enumerate() {
            let band = usize::from(i % CODED_SIDE + i / CODED_SIDE >= CODED_SIDE / 2);
            error[0][band] += f64::from((approximate[i] - e).abs());
            error[1][band] += f64::from((raw[i] - e).abs());
            magnitude[band] += f64::from(e.abs());
        }
    }
    error.map(|e| [e[0] / magnitude[0], e[1] / magnitude[1]])
}

#[test]
fn the_forward_approximates_the_exact_low_half() {
    // Measured (fractions of the mean magnitude, below / from diagonal 16):
    // smooth (noise 2) 0.019 / 0.68, noisy (60) 0.094 / 0.58. The neighbour
    // differences alias in as tan(πk/128) per dimension, so the upper band
    // is only roughly right; blocks coded with 64-point transforms at our
    // rates keep a few levels near DC.
    let dsp = Dsp::new();
    let [[smooth_low, smooth_high], _] = relative_errors(&dsp, 2);
    assert!(smooth_low < 0.03, "smooth low band {smooth_low}");
    assert!(smooth_high < 0.8, "smooth high band {smooth_high}");
    let [[noisy_low, noisy_high], _] = relative_errors(&dsp, 60);
    assert!(noisy_low < 0.12, "noisy low band {noisy_low}");
    assert!(noisy_high < 0.8, "noisy high band {noisy_high}");
}

#[test]
fn the_repair_brings_the_forward_closer() {
    // Little near DC, where cos(πk/128) is close to 1 (smooth low band
    // 0.0187 against 0.0194 unrepaired), 5-7% from diagonal 16 (0.68 / 0.74).
    let dsp = Dsp::new();
    for noise in [2, 60] {
        let [repaired, raw] = relative_errors(&dsp, noise);
        assert!(repaired[0] < raw[0] && repaired[1] < raw[1], "{noise}");
    }
}

#[test]
fn a_proven_zero_block_codes_nothing() {
    let dsp = Dsp::new();
    let flat = vec![100u8; BLOCK_AREA];
    let mut nearly = flat.clone();
    nearly[BLOCK_AREA / 2] = 120;
    let mut out = vec![0; CODED_AREA];
    let quantizer = VvcQuantizer::rect(37, 6, 6);
    let limit = quantizer.largest_zero_coefficient();
    let (s, p) = (view(&nearly), view(&flat));
    assert!(!dsp.vvc_forward64(&s, &p, limit, HevcForwardMode::EXACT, &mut out));
}

#[test]
fn the_inverse_matches_the_general_inverse() {
    let dsp = Dsp::new();
    let mut stream = Stream(11);
    for pattern in 0..6 {
        for (qp, dependent) in [(4, false), (22, true), (37, false), (51, true), (63, false)] {
            let mut compact = vec![0i32; CODED_AREA];
            for (i, level) in compact.iter_mut().enumerate() {
                let (u, v) = (i % CODED_SIDE, i / CODED_SIDE);
                let r = stream.next();
                *level = match pattern {
                    0 if u + v < 4 => (r % 41) as i32 - 20,
                    1 if r.is_multiple_of(7) => (r % 2001) as i32 - 1000,
                    2 => (r % 32767) as i32 - 16383,
                    3 if i == 0 => 1,
                    4 if v == 31 || u == 31 => (r % 9) as i32 - 4,
                    5 if r.is_multiple_of(3) => (r % 5) as i32 - 2,
                    _ => 0,
                };
            }
            let mut full = vec![0i32; BLOCK_AREA];
            for (i, &level) in compact.iter().enumerate() {
                full[(i / CODED_SIDE) * BLOCK_SIDE + i % CODED_SIDE] = level;
            }
            let mut expected = vec![0; BLOCK_AREA];
            dsp.vvc_inverse_primary(
                (BLOCK_SIDE, BLOCK_SIDE),
                [VvcTransformKind::Dct2; 2],
                &full,
                qp,
                dependent,
                &mut expected,
            );
            let mut got = vec![0i16; BLOCK_AREA];
            dsp.vvc_inverse_dct64(&compact, qp, dependent, &mut got);
            let got: Vec<i32> = got.into_iter().map(i32::from).collect();
            assert_eq!(got, expected, "pattern {pattern} qp {qp} {dependent}");
        }
    }
}

#[test]
fn forward_then_inverse_returns_a_smooth_residual() {
    // At QP 4 (unit step) a smooth residual comes back within a few levels:
    // the approximation is close enough to be a usable transform.
    let dsp = Dsp::new();
    let mut stream = Stream(5);
    let (source, prediction) = blocks(&mut stream, 0);
    let mut coefficients = vec![0; CODED_AREA];
    assert!(dsp.vvc_forward64(
        &view(&source),
        &view(&prediction),
        0,
        HevcForwardMode::EXACT,
        &mut coefficients
    ));
    let quantizer = VvcQuantizer::rect(4, 6, 6);
    let mut levels = vec![0; CODED_AREA];
    dsp.vvc_quantize_from(&coefficients, &quantizer, &mut levels);
    let mut back = vec![0i16; BLOCK_AREA];
    dsp.vvc_inverse_dct64(&levels, 4, false, &mut back);
    let error: i64 = source
        .iter()
        .zip(&prediction)
        .zip(&back)
        .map(|((&s, &p), &r)| i64::from((i32::from(s) - i32::from(p) - i32::from(r)).abs()))
        .sum();
    // The prediction's 0..15 noise is high frequency the 64-point
    // transform drops; the smooth part comes back.
    assert!(
        error < 6 * BLOCK_AREA as i64,
        "mean error {}",
        error / BLOCK_AREA as i64
    );
}
