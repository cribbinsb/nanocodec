//! HEVC weighted sample prediction against the scalar reference: default
//! bi-prediction and explicit uni/bi weights, every block width the codec
//! uses (including 2, 6, 12, 24 and 48 wide chroma and AMP blocks), random
//! and extreme 14-bit predictions, guarded output rows at a wider stride.
use crate::{check::CheckReport, rng::Rng};
use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, Filter, HevcWeights};

/// Guard byte around the block, to catch out-of-range writes.
const GUARD: u8 = 0xA5;
/// Extra samples right of each output row.
const PAD: usize = 3;
/// Block widths and heights tried.
const WIDTHS: [usize; 11] = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 3];
const HEIGHTS: [usize; 4] = [2, 4, 8, 16];
/// Random weight sets per block shape.
const WEIGHT_SETS: usize = 12;
/// Range of 14-bit intermediate predictions (8-tap filters overshoot the
/// 0..=255·64 range by about a quarter on each side).
const PRED_MIN: i32 = -10_000;
const PRED_MAX: i32 = 26_000;

fn predictions(len: usize, pattern: usize, rng: &mut Rng) -> Vec<i16> {
    (0..len)
        .map(|i| match pattern {
            0 => (PRED_MIN + rng.below((PRED_MAX - PRED_MIN) as usize) as i32) as i16,
            1 => i16::MAX,
            2 => i16::MIN + 1,
            _ => {
                if i.is_multiple_of(2) {
                    PRED_MAX as i16
                } else {
                    PRED_MIN as i16
                }
            }
        })
        .collect()
}

/// Weights as the decoder forms them: default bi-prediction, and explicit
/// uni and bi with denominators 0..=7, weights −128..=255 and offsets
/// −128..=127.
fn weights(set: usize, rng: &mut Rng) -> (HevcWeights, bool) {
    if set == 0 {
        return (
            HevcWeights {
                w0: 1,
                w1: 1,
                round: 64,
                shift: 7,
            },
            true,
        );
    }
    let log2wd = 6 + rng.below(8) as u32;
    let weight = |rng: &mut Rng| rng.below(384) as i16 - 128;
    let offset = |rng: &mut Rng| rng.below(256) as i32 - 128;
    if set % 2 == 1 {
        let o = offset(rng);
        let w = HevcWeights {
            w0: weight(rng),
            w1: 0,
            round: (1 << (log2wd - 1)) + (o << log2wd),
            shift: log2wd,
        };
        (w, false)
    } else {
        let (o0, o1) = (offset(rng), offset(rng));
        let w = HevcWeights {
            w0: weight(rng),
            w1: weight(rng),
            round: (o0 + o1 + 1) << log2wd,
            shift: log2wd + 1,
        };
        (w, true)
    }
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    check_centered_extremes(candidate, reference, report);
    for w in WIDTHS {
        for h in HEIGHTS {
            for pattern in 0..4 {
                for set in 0..WEIGHT_SETS {
                    let a = predictions(w * h, pattern, rng);
                    let b = predictions(w * h, (pattern + set) % 4, rng);
                    let (weights, bi) = weights(set, rng);
                    let second = if bi { &b } else { &a };
                    let stride = w + PAD;
                    let run = |dsp: &Dsp| {
                        let mut out = vec![GUARD; stride * h + 2];
                        let mut view =
                            PlaneViewMut::new(&mut out[1..], stride, w, h).expect("output view");
                        dsp.hevc_weighted_prediction(weights, &a, second, &mut view);
                        out
                    };
                    report.cases += 1;
                    if run(candidate) != run(reference) {
                        report.failures.push(format!(
                            "{} hevc_weighted_prediction {w}x{h} pattern={pattern} {weights:?}",
                            candidate.isa()
                        ));
                    }
                }
            }
        }
    }
}

/// Default uni/bi and signed explicit weights for the overshoot regression.
const EXTREME_WEIGHTS: [HevcWeights; 5] = [
    HevcWeights {
        w0: 1,
        w1: 1,
        round: 64,
        shift: 7,
    },
    HevcWeights {
        w0: 1,
        w1: 0,
        round: 32,
        shift: 6,
    },
    HevcWeights {
        w0: 0,
        w1: 1,
        round: 32,
        shift: 6,
    },
    HevcWeights {
        w0: 5,
        w1: -3,
        round: 256,
        shift: 9,
    },
    HevcWeights {
        w0: 127,
        w1: -128,
        round: 16384,
        shift: 14,
    },
];

/// An independent i32 oracle covers the sign-product pattern that exceeds
/// i16 before centering: 33150 and its complementary predictor -16830.
fn check_centered_extremes(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    const TAPS: [i32; 8] = [-1, 4, -11, 40, 40, -11, 4, -1];
    const OFFSET: i32 = 1 << 13;
    const H: usize = 4;
    for w in [4, 8, 16, 32, 64] {
        let stride = w + 9;
        let len = (H + 6) * stride + w + 7;
        let source: Vec<u8> = (0..len)
            .map(|i| {
                if (TAPS[(i / stride) % 8] > 0) == (TAPS[(i % stride) % 8] > 0) {
                    255
                } else {
                    0
                }
            })
            .collect();
        let complement: Vec<u8> = source.iter().map(|&p| 255 - p).collect();
        let (gold_a, gold_b) = (
            extreme_oracle(&source, stride, w, H),
            extreme_oracle(&complement, stride, w, H),
        );
        assert_eq!((gold_a[0], gold_b[0]), (33150, -16830));
        let size = BlockSize::new(w, H).expect("shape");
        for dsp in [candidate, reference] {
            let prep = |src: &[u8]| {
                let window = PlaneView::new(src, stride, w + 7, H + 7).expect("window");
                let mut out = vec![0i16; w * H];
                dsp.predict_compound(Filter::HevcLuma, size, &window, (2, 2), &mut out, w);
                out
            };
            let (a, b) = (prep(&source), prep(&complement));
            let exact = a
                .iter()
                .zip(&gold_a)
                .chain(b.iter().zip(&gold_b))
                .all(|(&got, &gold)| i32::from(got) == gold - OFFSET);
            report.cases += 1;
            if !exact {
                report
                    .failures
                    .push(format!("{} HEVC centered extreme prep {w}x{H}", dsp.isa()));
            }
            for weights in EXTREME_WEIGHTS {
                let ds = w + PAD;
                let mut got = vec![GUARD; (H - 1) * ds + w];
                let mut want = got.clone();
                for y in 0..H {
                    for x in 0..w {
                        let i = y * w + x;
                        want[y * ds + x] = ((gold_a[i] * i32::from(weights.w0)
                            + gold_b[i] * i32::from(weights.w1)
                            + weights.round)
                            >> weights.shift)
                            .clamp(0, 255) as u8;
                    }
                }
                if weights.w0 == 1 && weights.w1 == 1 {
                    assert_eq!(want[0], 128);
                }
                let mut view = PlaneViewMut::new(&mut got, ds, w, H).expect("output");
                dsp.hevc_weighted_prediction_centered(weights, &a, &b, &mut view);
                report.cases += 1;
                if got != want {
                    report.failures.push(format!(
                        "{} HEVC centered extreme weighted {w}x{H} {weights:?}",
                        dsp.isa()
                    ));
                }
            }
        }
    }
}

/// Direct 32-bit two-pass sample equation, independently of DSP prep/storage.
fn extreme_oracle(src: &[u8], stride: usize, w: usize, h: usize) -> Vec<i32> {
    const TAPS: [i32; 8] = [-1, 4, -11, 40, 40, -11, 4, -1];
    (0..w * h)
        .map(|i| {
            let (y, x) = (i / w, i % w);
            let mut sum = 0;
            for (dy, &ty) in TAPS.iter().enumerate() {
                for (dx, &tx) in TAPS.iter().enumerate() {
                    sum += ty * tx * i32::from(src[(y + dy) * stride + x + dx]);
                }
            }
            sum >> 6
        })
        .collect()
}
