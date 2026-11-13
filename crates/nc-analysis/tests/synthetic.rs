//! End-to-end checks of the analysis pipeline on synthetic sequences whose
//! true motion is known exactly.

use nc_analysis::motion::SearchConfig;
use nc_analysis::{Analyzer, FrameAnalysis, Mv};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::Dsp;

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

/// A picture whose luma at (x, y) is `f(x, y)` and whose chroma is flat.
fn picture(f: impl Fn(f64, f64) -> f64) -> Picture {
    let mut p = Picture::new(WIDTH, HEIGHT, 0).unwrap();
    let mut luma = p.plane_mut(PlaneIndex::Y).view_mut();
    for y in 0..HEIGHT {
        for (x, s) in luma.row_mut(y).iter_mut().enumerate() {
            *s = f(x as f64, y as f64).round().clamp(0.0, 255.0) as u8;
        }
    }
    for plane in [PlaneIndex::U, PlaneIndex::V] {
        p.plane_mut(plane).view_mut().fill(128);
    }
    p
}

/// Deterministic white noise on integer positions.
fn noise(x: f64, y: f64) -> f64 {
    let (x, y) = (x as i64 as u64, y as i64 as u64);
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ y.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h % 256) as f64
}

/// Natural-image-like texture: low-frequency structure (which the coarse,
/// quarter-resolution search can follow) plus fine detail (which makes the
/// full-resolution match unique).
fn detailed(x: f64, y: f64) -> f64 {
    0.7 * smooth(x, y) + 0.3 * noise(x, y)
}

/// Smooth, non-repeating texture that survives sub-sample resampling.
fn smooth(x: f64, y: f64) -> f64 {
    128.0
        + 40.0 * (0.131 * x + 0.071 * y).sin()
        + 30.0 * (0.053 * x - 0.113 * y + 1.0).sin()
        + 25.0 * (0.217 * x + 0.171 * y + 2.0).sin()
        + 15.0 * (0.019 * x + 0.029 * y + 0.5).sin()
}

fn analyze_pair(first: &Picture, second: &Picture) -> FrameAnalysis {
    let mut analyzer = Analyzer::new(WIDTH, HEIGHT, Dsp::new()).unwrap();
    analyzer.analyze(first);
    analyzer.analyze(second).clone()
}

/// Fraction of interior 8x8 blocks with exactly `expected` as vector and a
/// perfect match. Interior: the block's whole 64x64 superblock (its part
/// inside the picture) has its match inside the picture. The search works
/// per superblock (SVT-AV1 style), so where part of a superblock's match
/// leaves the picture, its other blocks are not guaranteed the exact vector.
fn exact_fraction(analysis: &FrameAnalysis, expected: Mv) -> f64 {
    let fine = &analysis.motion.as_ref().unwrap().fine;
    let matched_inside = |x: i64, y: i64, w: i64, h: i64| {
        let (rx, ry) = (x + i64::from(expected.x), y + i64::from(expected.y));
        rx >= 0 && ry >= 0 && rx + w <= WIDTH as i64 && ry + h <= HEIGHT as i64
    };
    let (mut interior, mut exact) = (0, 0);
    for (col, row) in fine.positions() {
        let (sb_x, sb_y) = ((col / 8 * 64) as i64, (row / 8 * 64) as i64);
        let sb_h = 64.min(HEIGHT as i64 - sb_y);
        let sb_w = 64.min(WIDTH as i64 - sb_x);
        if !matched_inside(sb_x, sb_y, sb_w, sb_h) {
            continue;
        }
        interior += 1;
        let b = fine.get(col, row);
        if b.mv == expected && b.sad == 0 {
            exact += 1;
        }
    }
    f64::from(exact) / f64::from(interior)
}

#[test]
fn finds_exact_integer_translations() {
    // Small, medium and large (the hierarchical window reaches ±64 x ±16
    // luma samples from a standing start).
    for (dx, dy) in [(5, -3), (37, 12), (-60, 15)] {
        let first = picture(detailed);
        let second = picture(|x, y| detailed(x + f64::from(dx), y + f64::from(dy)));
        let analysis = analyze_pair(&first, &second);
        let fraction = exact_fraction(&analysis, Mv::new(dx, dy));
        assert!(
            fraction > 0.99,
            "translation ({dx}, {dy}): only {fraction:.3} exact"
        );
    }
}

#[test]
fn lighter_search_still_finds_small_translations_with_fewer_positions() {
    let first = picture(detailed);
    let second = picture(|x, y| detailed(x + 5.0, y - 3.0));
    let full = analyze_pair(&first, &second);
    let mut analyzer = Analyzer::new(WIDTH, HEIGHT, Dsp::new()).unwrap();
    // A narrower window still bridges the half-resolution refinement's
    // last sample; without level 1 a quarter-resolution vector (steps of
    // four) would need the full window.
    analyzer.set_search_config(SearchConfig {
        window_half_width: 2,
        window_half_height: 1,
        mixed_motion: false,
        ..SearchConfig::default()
    });
    analyzer.analyze(&first);
    let light = analyzer.analyze(&second).clone();
    let fraction = exact_fraction(&light, Mv::new(5, -3));
    assert!(fraction > 0.99, "light search: only {fraction:.3} exact");
    let positions = |a: &FrameAnalysis| a.motion.as_ref().unwrap().effort.full_pel.positions;
    assert!(positions(&light) < positions(&full));
}

/// Analyse `frames` in order and return the last frame's analysis.
fn analyze_sequence(frames: &[Picture]) -> FrameAnalysis {
    let mut analyzer = Analyzer::new(WIDTH, HEIGHT, Dsp::new()).unwrap();
    for frame in &frames[..frames.len() - 1] {
        analyzer.analyze(frame);
    }
    analyzer.analyze(&frames[frames.len() - 1]).clone()
}

#[test]
fn large_motion_after_a_still_period_is_found() {
    // After several identical frames every temporal candidate is zero and
    // the zero-vector gates have bypassed the search; then the scene jumps
    // by (-50, 14). The jump fails the gates, so the full search runs.
    let (dx, dy) = (-50, 14);
    let still = picture(detailed);
    let moved = picture(|x, y| detailed(x + f64::from(dx), y + f64::from(dy)));
    let mut frames = vec![still; 5];
    frames.push(moved);
    let analysis = analyze_sequence(&frames);
    let fraction = exact_fraction(&analysis, Mv::new(dx, dy));
    assert!(
        fraction > 0.99,
        "only {fraction:.3} exact after the still period"
    );
}

#[test]
fn accelerating_motion_is_tracked_beyond_the_window() {
    // Steadily accelerating pan, 50 to 80 samples per frame: the last two
    // steps are beyond the hierarchical window around zero (±64), reachable
    // only because last frame's vector seeds the window centre.
    let offsets = [0.0, 50.0, 110.0, 180.0, 260.0];
    let frames: Vec<Picture> = offsets
        .iter()
        .map(|&o| picture(move |x, y| detailed(x + o, y)))
        .collect();
    let analysis = analyze_sequence(&frames);
    let fraction = exact_fraction(&analysis, Mv::new(80, 0));
    assert!(fraction > 0.99, "only {fraction:.3} exact");
}

#[test]
fn static_sequence_is_static_with_zero_motion() {
    let frame = picture(smooth);
    let mut analyzer = Analyzer::new(WIDTH, HEIGHT, Dsp::new()).unwrap();
    for i in 0..3 {
        let a = analyzer.analyze(&frame);
        if i == 0 {
            assert!(a.motion.is_none());
            continue;
        }
        assert!((a.pre.static_fraction - 1.0).abs() < f32::EPSILON);
        let motion = a.motion.as_ref().unwrap();
        assert!(
            motion
                .fine
                .cells()
                .iter()
                .all(|b| b.mv == Mv::ZERO && b.sad == 0)
        );
    }
}

#[test]
fn global_motion_recovers_a_pan() {
    let (tx, ty) = (-20.0, 8.0);
    let first = picture(smooth);
    let second = picture(|x, y| smooth(x + tx, y + ty));
    let global = analyze_pair(&first, &second).motion.unwrap().global;
    assert!((f64::from(global.tx) - tx).abs() < 1.0, "{global:?}");
    assert!((f64::from(global.ty) - ty).abs() < 1.0, "{global:?}");
    assert!(global.zoom.abs() < 0.005, "{global:?}");
    assert!(global.inlier_fraction > 0.5, "{global:?}");
}

#[test]
fn global_motion_recovers_pan_and_zoom() {
    let (tx, ty, zoom) = (6.0, -4.0, 0.03);
    let (cx, cy) = (WIDTH as f64 / 2.0, HEIGHT as f64 / 2.0);
    let first = picture(smooth);
    let second = picture(|x, y| smooth(x + tx + zoom * (x - cx), y + ty + zoom * (y - cy)));
    let global = analyze_pair(&first, &second).motion.unwrap().global;
    assert!((f64::from(global.tx) - tx).abs() < 1.0, "{global:?}");
    assert!((f64::from(global.ty) - ty).abs() < 1.0, "{global:?}");
    assert!((f64::from(global.zoom) - zoom).abs() < 0.005, "{global:?}");
    assert!(global.inlier_fraction > 0.5, "{global:?}");
}

#[test]
fn layered_links_search_against_the_named_slot() {
    // A steady pan of (3, -2) per frame, analysed as L1T3 positions 0..=4:
    // each frame must find the motion over its own reference distance.
    use nc_analysis::{FrameLink, ReferenceLink, SlotMask};
    let (dx, dy) = (3.0, -2.0);
    let frames: Vec<Picture> = (0..5)
        .map(|i| picture(|x, y| detailed(x + dx * f64::from(i), y + dy * f64::from(i))))
        .collect();
    let link = |slot, distance, refresh| FrameLink {
        reference: ReferenceLink { slot, distance },
        refresh,
    };
    let structure = [
        link(0, 1, SlotMask::ALL),
        link(0, 1, SlotMask::NONE),
        link(0, 2, SlotMask::only(1)),
        link(1, 1, SlotMask::NONE),
        link(0, 4, SlotMask::only(0)),
    ];
    let mut analyzer = Analyzer::with_reference_slots(WIDTH, HEIGHT, Dsp::new(), 2).unwrap();
    for (i, (frame, link)) in frames.iter().zip(structure).enumerate() {
        let analysis = analyzer.analyze_linked(frame, link);
        if i == 0 {
            assert!(analysis.motion.is_none());
            continue;
        }
        let d = link.reference.distance as i16;
        let fraction = exact_fraction(analysis, Mv::new(3 * d, -2 * d));
        assert!(fraction > 0.99, "frame {i}: only {fraction:.3} exact");
    }
}

#[test]
fn seeded_disposable_frames_track_a_pan_and_can_become_references() {
    // L1T2 with `hme_from_seed`: the disposable T1 frames centre their
    // window on the T0 frame's HME vectors, halved, and build no pyramid.
    // Frame 3 is then refreshed as a reference (a scene cut coded as a key),
    // and frame 4 turns sharply: only an HME search against frame 3's own
    // levels finds that.
    use nc_analysis::{FrameLink, ReferenceLink, SlotMask};
    let positions = [(0, 0), (12, 4), (24, 8), (36, 12), (-4, 24)];
    let frames: Vec<Picture> = positions
        .iter()
        .map(|&(px, py)| picture(|x, y| detailed(x + f64::from(px), y + f64::from(py))))
        .collect();
    let link = |distance, refresh| FrameLink {
        reference: ReferenceLink { slot: 0, distance },
        refresh,
    };
    let structure = [
        link(1, SlotMask::ALL),
        link(1, SlotMask::NONE),
        link(2, SlotMask::ALL),
        link(1, SlotMask::NONE),
        link(1, SlotMask::ALL),
    ];
    let references = [0, 0, 0, 2, 3];
    let mut analyzer = Analyzer::new(WIDTH, HEIGHT, Dsp::new()).unwrap();
    for (i, (frame, link)) in frames.iter().zip(structure).enumerate() {
        analyzer.set_search_config(SearchConfig {
            hme_from_seed: link.refresh == SlotMask::NONE,
            ..SearchConfig::default()
        });
        let analysis = analyzer.analyze_linked(frame, link);
        if i > 0 {
            let (r, c) = (positions[references[i]], positions[i]);
            let expected = Mv::new((c.0 - r.0) as i16, (c.1 - r.1) as i16);
            let fraction = exact_fraction(analysis, expected);
            assert!(fraction > 0.99, "frame {i}: only {fraction:.3} exact");
        }
        if i == 3 {
            analyzer.refresh_current(SlotMask::ALL);
        }
    }
}
