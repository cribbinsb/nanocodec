//! Tests of the shared SAO decision on synthetic pictures.

use nc_common::{Picture, PlaneIndex};
use nc_dsp::{Dsp, SaoBandStats, SaoStats};

use super::offsets::{self, delta};
use super::rate::{self, SaoContexts};
use super::{Candidates, ComponentSao, CtbSao, SaoKind, SaoType, Settings, Signal, decide};

/// A context model where every bin costs one bit and nothing adapts.
#[derive(Clone)]
struct Flat;

impl SaoContexts for Flat {
    fn merge_bits(&self, _: bool) -> f32 {
        1.0
    }
    fn type_bits(&self, _: bool) -> f32 {
        1.0
    }
    fn code_merge(&mut self, _: bool) {}
    fn code_type(&mut self, _: bool) {}
}

fn settings(lambda: f64) -> Settings {
    Settings {
        channels: [true, true],
        lambdas: [lambda, lambda],
        ctb: 16,
        row_step: 1,
        band: true,
    }
}

/// A `w` x `h` picture whose every sample of every plane is `value(x, y)`.
fn picture(w: usize, h: usize, value: impl Fn(usize, usize) -> u8) -> Picture {
    let mut p = Picture::new(w, h, 0).expect("picture");
    for plane in PlaneIndex::ALL {
        let mut view = p.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            for (x, s) in view.row_mut(y).iter_mut().enumerate() {
                *s = value(x, y);
            }
        }
    }
    p
}

#[test]
fn edge_offsets_are_the_rounded_mean_at_small_lambda_and_zero_at_large() {
    let stats = SaoStats {
        sum: [100, 21, -30, 50],
        count: [20, 10, 10, 10],
    };
    // Category 4 (local maximum) only darkens: its positive mean gives 0.
    assert_eq!(offsets::edge(&stats, 1e-6).offsets, [5, 2, -3, 0]);
    assert_eq!(offsets::edge(&stats, 1e9).offsets, [0; 4]);
}

#[test]
fn edge_offset_shrinks_while_bits_cost_more_than_distortion_saves() {
    // The rounded mean is 7; the chosen magnitude must be the cheapest of
    // all in distortion plus λ times its bits.
    let stats = SaoStats {
        sum: [70, 0, 0, 0],
        count: [10, 0, 0, 0],
    };
    let lambda = 30.0;
    let chosen = offsets::edge(&stats, lambda).offsets[0];
    let cost =
        |o: i32| delta(10, 70, o) as f64 + lambda * f64::from(offsets::magnitude_bits(o, false));
    assert!((0..=7).all(|o| cost(i32::from(chosen)) <= cost(o)));
}

#[test]
fn band_position_covers_the_biased_bands_and_wraps() {
    let mut stats = SaoBandStats::default();
    for (band, mean) in [(30, 4), (31, 3), (0, 5), (1, 2)] {
        stats.count[band] = 100;
        stats.sum[band] = 100 * mean;
    }
    stats.count[10] = 100; // unbiased band: no offset wanted
    let chosen = offsets::band(&stats, 1.0);
    assert_eq!(chosen.position, 30);
    assert_eq!(chosen.offsets.offsets, [4, 3, 5, 2]);
    assert!(chosen.offsets.distortion < 0);
}

#[test]
fn committed_bits_follow_the_binarisation() {
    let band = CtbSao {
        luma: Some(ComponentSao {
            kind: SaoKind::Band([3, 0]),
            offsets: [[2, 0, -1, 7], [0; 4]],
        }),
        chroma: None,
    };
    // Merge left 0, merge up 0; luma: type 1 + 1 bypass, |2|+1, 1, |1|+1, 7
    // (cMax), 3 signs, 5 position bits; chroma: type off 1.
    let expected = 2.0 + 2.0 + (3.0 + 1.0 + 2.0 + 7.0) + 3.0 + 5.0 + 1.0;
    let bits = rate::commit(
        &mut Flat,
        &band,
        Signal::Explicit,
        (true, true),
        [true, true],
    );
    assert!((bits - expected).abs() < 1e-9, "{bits} vs {expected}");
    let merged = rate::commit(
        &mut Flat,
        &band,
        Signal::MergeUp,
        (true, true),
        [true, true],
    );
    assert!((merged - 2.0).abs() < 1e-9);
}

#[test]
fn a_dark_offset_picture_takes_band_offset() {
    // Source is the deblocked picture plus 3 in a smooth dark ramp: edge
    // categories are empty on a ramp, so only band offset can correct it.
    let source = picture(32, 32, |x, _| 40 + (x / 4) as u8 + 3);
    let deblocked = picture(32, 32, |x, _| 40 + (x / 4) as u8);
    let sao = decide(
        &Dsp::new(),
        (&source, &deblocked),
        &settings(1.0),
        Candidates::Search,
        None,
        Flat,
    );
    assert_eq!(sao.cols, 2);
    let (first, signal) = sao.ctbs[0];
    assert_eq!(signal, Signal::Explicit);
    let luma = first.luma.expect("luma SAO on");
    assert_eq!(luma.sao_type(), SaoType::Band);
    let SaoKind::Band([position, _]) = luma.kind else {
        unreachable!()
    };
    // Samples 40..47 are band 5, the only band with samples: any window
    // holding it costs the same, and the first is kept.
    assert_eq!(position, 2);
    assert_eq!(luma.offsets[0], [0, 0, 0, 3]);
    // The CTB beside has the same samples and errors: it merges left.
    assert_eq!(sao.ctbs[1].1, Signal::MergeLeft);
}

#[test]
fn reuse_keeps_each_ctb_to_its_types() {
    let source = picture(32, 32, |x, _| 40 + (x / 4) as u8 + 3);
    let deblocked = picture(32, 32, |x, _| 40 + (x / 4) as u8);
    let types = vec![[None, None]; 4];
    let sao = decide(
        &Dsp::new(),
        (&source, &deblocked),
        &settings(1.0),
        Candidates::Reuse(&types),
        None,
        Flat,
    );
    assert!(sao.ctbs.iter().all(|(p, _)| *p == CtbSao::default()));
}
