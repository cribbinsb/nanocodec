//! Bit-exactness of the AV1 inter prediction kernels (`nc_dsp::av1_mc`)
//! against libaom v3.15.3.
//!
//! The golden files in `golden/av1_mc/` were produced by a throwaway C
//! harness (kept out of the repository) that ran libaom's C kernels —
//! `av1_convolve_2d_sr_c`, `av1_dist_wtd_convolve_2d_c`,
//! `av1_convolve_2d_scale_c`, `av1_scaled_x/y`,
//! `aom_lowbd_blend_a64_d16_mask_c`, `av1_build_compound_diffwtd_mask_d16_c`,
//! `av1_get_contiguous_soft_mask`, `av1_combine_interintra`,
//! `aom_blend_a64_[hv]mask_c`, `av1_get_shear_params`,
//! `av1_find_projection` and `av1_warp_affine_c` — on inputs drawn from a
//! seeded `SplitMix64` stream. Each test here draws the same inputs in the
//! same order, runs our kernel and compares one line per case: FNV-1a
//! hashes of the outputs (compound samples as little-endian `i16`, without
//! libaom's internal offset of 6144), or the values themselves for the
//! position, shear and estimation cases.

use nc_checkasm::rng::Rng;
use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::Dsp;
use nc_dsp::av1_mc::blend::{
    Av1CompoundPred, Av1Mask, Av1ObmcEdge, av1_blend_average, av1_blend_distance,
    av1_blend_interintra, av1_blend_mask, av1_distance_weights, av1_obmc_blend,
};
use nc_dsp::av1_mc::convolve::Av1McAxis;
use nc_dsp::av1_mc::filter::{Av1Filter, Av1Filters};
use nc_dsp::av1_mc::mask::{
    AV1_WEDGE_SIZES, AV1_WEDGE_TYPES, Av1InterIntraMode, Av1WedgeMasks, av1_difference_weight_mask,
    av1_interintra_smooth_mask,
};
use nc_dsp::av1_mc::scale::{Av1ScaleSizes, av1_scale_position};
use nc_dsp::av1_mc::warp::{
    Av1WarpSample, av1_setup_shear, av1_warp, av1_warp_compound, av1_warp_estimation,
};

/// AV1 block edges.
const SIZES: [usize; 7] = [2, 4, 8, 16, 32, 64, 128];
/// Rows and columns a window needs beyond the block (8-tap filters).
const TAPS_EXTRA: usize = 7;

/// The harness's `SplitMix64` seed of each case group.
const SEEDS: [(&str, u64); 12] = [
    ("convolve", 1),
    ("scaled", 2),
    ("scale_position", 3),
    ("blend", 4),
    ("mask_blend", 5),
    ("diffwtd", 6),
    ("wedge", 7),
    ("interintra", 8),
    ("obmc", 9),
    ("shear", 10),
    ("warp_estimation", 11),
    ("warp", 12),
];

/// The golden lines of one case group.
fn golden(name: &str) -> Vec<String> {
    let path = format!(
        "{}/tests/golden/av1_mc/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .map(str::to_owned)
        .collect()
}

/// The harness's draws.
struct Draw(Rng);

impl Draw {
    fn below(&mut self, bound: usize) -> usize {
        self.0.below(bound)
    }

    fn signed(&mut self, bound: usize, offset: i32) -> i32 {
        self.below(bound) as i32 - offset
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }

    /// `n` samples of one of four patterns: random, extremes, near-flat.
    fn pixels(&mut self, n: usize, pattern: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                let b = self.0.byte();
                match pattern {
                    1 => (b & 1) * 255,
                    2 => 128 + b % 16,
                    _ => b,
                }
            })
            .collect()
    }

    /// A compound-precision sample value.
    fn compound_value(&mut self) -> i16 {
        self.signed(8000, 2000) as i16
    }
}

fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn hex(hash: u64) -> String {
    format!("{hash:016x}")
}

fn hash_i16(values: &[i16]) -> u64 {
    fnv(values.iter().flat_map(|v| v.to_le_bytes()))
}

/// Compare every case of a group; report all mismatching case numbers.
fn check_group(name: &str, mut case: impl FnMut(&mut Draw) -> String) {
    let lines = golden(name);
    let seed = SEEDS
        .iter()
        .find(|&&(group, _)| group == name)
        .map_or_else(|| panic!("no seed for group {name}"), |&(_, seed)| seed);
    let mut draw = Draw(Rng::new(seed));
    let failures: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, want)| (case(&mut draw) != *want).then_some(i))
        .collect();
    assert!(
        failures.is_empty(),
        "{name}: {} of {} cases differ from libaom, first {:?}",
        failures.len(),
        lines.len(),
        &failures[..failures.len().min(10)]
    );
}

/// One unscaled prediction's draws: filters, phases, pattern and window.
struct Pred {
    filters: Av1Filters,
    axes: (Av1McAxis, Av1McAxis),
    stride: usize,
    window: Vec<u8>,
}

impl Pred {
    fn draw(d: &mut Draw, (w, h): (usize, usize)) -> Self {
        let filters = Av1Filters {
            x: d.pick(&Av1Filter::ALL),
            y: d.pick(&Av1Filter::ALL),
        };
        let axes = (
            Av1McAxis::unscaled(d.below(16)),
            Av1McAxis::unscaled(d.below(16)),
        );
        let pattern = d.below(4);
        let stride = w + TAPS_EXTRA + d.below(5);
        let window = d.pixels(stride * (h + TAPS_EXTRA), pattern);
        Self {
            filters,
            axes,
            stride,
            window,
        }
    }

    fn view(&self) -> PlaneView<'_> {
        let rows = self.window.len() / self.stride;
        PlaneView::new(&self.window, self.stride, self.stride, rows).unwrap()
    }

    fn compound(&self, dsp: &Dsp, (w, h): (usize, usize)) -> Vec<i16> {
        let mut out = vec![0i16; w * h];
        dsp.av1_predict_compound(self.filters, &self.view(), (w, h), self.axes, &mut out, w);
        out
    }
}

/// The final and compound predictions' hashes from one window.
fn both_precisions(
    dsp: &Dsp,
    filters: Av1Filters,
    window: &PlaneView<'_>,
    (w, h): (usize, usize),
    axes: (Av1McAxis, Av1McAxis),
) -> String {
    let mut pixels = vec![0u8; w * h];
    dsp.av1_predict(
        filters,
        window,
        (w, h),
        axes,
        &mut PlaneViewMut::new(&mut pixels, w, w, h).unwrap(),
    );
    let mut compound = vec![0i16; w * h];
    dsp.av1_predict_compound(filters, window, (w, h), axes, &mut compound, w);
    format!("{} {}", hex(fnv(pixels)), hex(hash_i16(&compound)))
}

#[test]
fn convolve_matches_libaom_for_every_filter_pair_and_size() {
    for dsp in [Dsp::new(), Dsp::with_isa(nc_dsp::Isa::Scalar).unwrap()] {
        check_group("convolve", |d| {
            let size = (d.pick(&SIZES), d.pick(&SIZES));
            let pred = Pred::draw(d, size);
            both_precisions(&dsp, pred.filters, &pred.view(), size, pred.axes)
        });
    }
}

#[test]
fn scaled_convolve_matches_libaom() {
    let dsp = Dsp::new();
    check_group("scaled", |d| {
        let (w, h) = (d.pick(&SIZES), d.pick(&SIZES));
        let filters = Av1Filters {
            x: d.pick(&Av1Filter::ALL),
            y: d.pick(&Av1Filter::ALL),
        };
        let x = Av1McAxis {
            position: d.below(1024) as u32,
            step: 64 + d.below(1985) as u32,
        };
        let y = Av1McAxis {
            position: d.below(1024) as u32,
            step: 64 + d.below(1985) as u32,
        };
        let pattern = d.below(4);
        let stride = x.span(w) + d.below(5);
        let rows = y.span(h);
        let window = d.pixels(stride * rows, pattern);
        let view = PlaneView::new(&window, stride, stride, rows).unwrap();
        both_precisions(&dsp, filters, &view, (w, h), (x, y))
    });
}

#[test]
fn scale_positions_match_libaom() {
    check_group("scale_position", |d| {
        let frame = (16 + d.below(2000) as u32, 16 + d.below(2000) as u32);
        let low = (frame.0.div_ceil(16), frame.1.div_ceil(16));
        let reference = (
            low.0 + d.below((2 * frame.0 - low.0 + 1) as usize) as u32,
            low.1 + d.below((2 * frame.1 - low.1 + 1) as usize) as u32,
        );
        let position = (
            d.below(frame.0 as usize) as u32,
            d.below(frame.1 as usize) as u32,
        );
        let mv = (d.signed(4096, 2048), d.signed(4096, 2048));
        let sub = (d.below(2) == 1, d.below(2) == 1);
        let p = av1_scale_position(position, mv, sub, Av1ScaleSizes { reference, frame });
        let start = |sample: i64, axis: Av1McAxis| (sample << 10) + i64::from(axis.position);
        format!(
            "{} {} {} {}",
            start(p.sample.0, p.x),
            start(p.sample.1, p.y),
            p.x.step,
            p.y.step
        )
    });
}

#[test]
fn compound_average_and_distance_blends_match_libaom() {
    let dsp = Dsp::new();
    check_group("blend", |d| {
        let size @ (w, h) = (d.pick(&SIZES), d.pick(&SIZES));
        let (a, b) = (Pred::draw(d, size), Pred::draw(d, size));
        let distance = d.below(2) == 1;
        let weights = av1_distance_weights(d.signed(64, 32), d.signed(64, 32));
        let (p0, p1) = (a.compound(&dsp, size), b.compound(&dsp, size));
        let (p0, p1) = (
            Av1CompoundPred {
                data: &p0,
                stride: w,
            },
            Av1CompoundPred {
                data: &p1,
                stride: w,
            },
        );
        let mut out = vec![0u8; w * h];
        let mut dst = PlaneViewMut::new(&mut out, w, w, h).unwrap();
        if distance {
            av1_blend_distance(p0, p1, weights, size, &mut dst);
        } else {
            av1_blend_average(p0, p1, size, &mut dst);
        }
        format!("{} {} {}", weights[0], weights[1], hex(fnv(out)))
    });
}

#[test]
fn compound_mask_blend_matches_libaom() {
    for dsp in [Dsp::new(), Dsp::with_isa(nc_dsp::Isa::Scalar).unwrap()] {
        compound_mask_blend_matches_libaom_with(&dsp);
    }
}

fn compound_mask_blend_matches_libaom_with(dsp: &Dsp) {
    check_group("mask_blend", |d| {
        let size @ (w, h) = (d.pick(&SIZES[..5]), d.pick(&SIZES[..5]));
        let (sub_x, sub_y) = (d.below(2) == 1, d.below(2) == 1);
        let p0: Vec<i16> = (0..w * h).map(|_| d.compound_value()).collect();
        let p1: Vec<i16> = (0..w * h).map(|_| d.compound_value()).collect();
        let (mw, mh) = (w << u32::from(sub_x), h << u32::from(sub_y));
        let weights: Vec<u8> = (0..mw * mh).map(|_| d.below(65) as u8).collect();
        let mut out = vec![0u8; w * h];
        av1_blend_mask(
            dsp,
            Av1CompoundPred {
                data: &p0,
                stride: w,
            },
            Av1CompoundPred {
                data: &p1,
                stride: w,
            },
            Av1Mask {
                data: &weights,
                stride: mw,
                sub_x,
                sub_y,
            },
            size,
            &mut PlaneViewMut::new(&mut out, w, w, h).unwrap(),
        );
        hex(fnv(out))
    });
}

#[test]
fn difference_weight_masks_match_libaom() {
    check_group("diffwtd", |d| {
        let size @ (w, h) = (d.pick(&SIZES[..6]), d.pick(&SIZES[..6]));
        let inverse = d.below(2) == 1;
        let (mut p0, mut p1) = (Vec::new(), Vec::new());
        for _ in 0..w * h {
            let a = d.compound_value();
            let close = d.below(2) == 1;
            let b = if close {
                a + d.signed(512, 256) as i16
            } else {
                d.compound_value()
            };
            p0.push(a);
            p1.push(b);
        }
        let mut mask = vec![0u8; w * h];
        av1_difference_weight_mask(
            Av1CompoundPred {
                data: &p0,
                stride: w,
            },
            Av1CompoundPred {
                data: &p1,
                stride: w,
            },
            inverse,
            size,
            &mut mask,
        );
        hex(fnv(mask))
    });
}

#[test]
fn wedge_masks_match_libaom() {
    let masks = Av1WedgeMasks::new();
    let mut cases = AV1_WEDGE_SIZES.iter().flat_map(|&size| {
        [false, true]
            .into_iter()
            .flat_map(move |sign| (0..AV1_WEDGE_TYPES).map(move |index| (size, sign, index)))
    });
    check_group("wedge", |_| {
        let (size, sign, index) = cases.next().unwrap();
        hex(fnv(masks.mask(size, sign, index).unwrap().iter().copied()))
    });
}

#[test]
fn interintra_blend_matches_libaom() {
    let masks = Av1WedgeMasks::new();
    let modes = [
        Av1InterIntraMode::Dc,
        Av1InterIntraMode::Vertical,
        Av1InterIntraMode::Horizontal,
        Av1InterIntraMode::Smooth,
    ];
    check_group("interintra", |d| {
        let luma = d.pick(&AV1_WEDGE_SIZES[..7]);
        let sub = d.below(2) == 1;
        let wedge = d.below(2) == 1;
        let mode = d.pick(&modes);
        let index = d.below(AV1_WEDGE_TYPES);
        let pattern = d.below(4);
        let size @ (w, h) = (luma.0 >> u32::from(sub), luma.1 >> u32::from(sub));
        let intra = d.pixels(w * h, pattern);
        let mut inter = d.pixels(w * h, pattern);
        let mut smooth = vec![0u8; w * h];
        let mask = if wedge {
            Av1Mask {
                data: masks.mask(luma, false, index).unwrap(),
                stride: luma.0,
                sub_x: sub,
                sub_y: sub,
            }
        } else {
            av1_interintra_smooth_mask(mode, size, &mut smooth);
            Av1Mask {
                data: &smooth,
                stride: w,
                sub_x: false,
                sub_y: false,
            }
        };
        av1_blend_interintra(
            &PlaneView::new(&intra, w, w, h).unwrap(),
            mask,
            size,
            &mut PlaneViewMut::new(&mut inter, w, w, h).unwrap(),
        );
        hex(fnv(inter))
    });
}

#[test]
fn obmc_blend_matches_libaom() {
    check_group("obmc", |d| {
        let left = d.below(2) == 1;
        let overlap = d.pick(&[1, 2, 4, 8, 16, 32]);
        let other = d.pick(&SIZES);
        let size @ (w, h) = if left {
            (overlap, other)
        } else {
            (other, overlap)
        };
        let pattern = d.below(4);
        let mut own = d.pixels(w * h, pattern);
        let neighbour = d.pixels(w * h, pattern);
        let edge = if left {
            Av1ObmcEdge::Left
        } else {
            Av1ObmcEdge::Above
        };
        av1_obmc_blend(
            &PlaneView::new(&neighbour, w, w, h).unwrap(),
            edge,
            size,
            &mut PlaneViewMut::new(&mut own, w, w, h).unwrap(),
        );
        hex(fnv(own))
    });
}

/// A random affine warp model with translation up to `translation`.
fn draw_model(d: &mut Draw, translation: usize) -> [i32; 6] {
    let dev = d.pick(&[256, 2048, 8192, 40000]);
    let mut around = |centre: i32| centre + d.signed(2 * dev, dev as i32);
    let (m2, m3, m4, m5) = (around(65536), around(0), around(0), around(65536));
    let t = translation as i32;
    [
        d.signed(2 * translation, t),
        d.signed(2 * translation, t),
        m2,
        m3,
        m4,
        m5,
    ]
}

#[test]
fn shear_setup_matches_libaom() {
    check_group("shear", |d| {
        match av1_setup_shear(&draw_model(d, 1 << 20)) {
            Some(s) => format!("1 {} {} {} {}", s.alpha, s.beta, s.gamma, s.delta),
            None => "0".to_owned(),
        }
    });
}

#[test]
fn warp_estimation_matches_libaom() {
    const BLOCKS: [(i32, i32); 17] = [
        (8, 8),
        (8, 16),
        (16, 8),
        (16, 16),
        (16, 32),
        (32, 16),
        (32, 32),
        (32, 64),
        (64, 32),
        (64, 64),
        (64, 128),
        (128, 64),
        (128, 128),
        (8, 32),
        (32, 8),
        (16, 64),
        (64, 16),
    ];
    check_group("warp_estimation", |d| {
        let (bw, bh) = d.pick(&BLOCKS);
        let (mi_row, mi_col) = (d.below(256) as u32, d.below(256) as u32);
        let mv = (d.signed(1024, 512), d.signed(1024, 512));
        let count = 1 + d.below(8);
        let samples: Vec<Av1WarpSample> = (0..count)
            .map(|_| {
                let sx = d.signed((2 * bw * 8 + 128) as usize, 64);
                let sy = d.signed((2 * bh * 8 + 128) as usize, 64);
                let jitter = if d.below(4) == 0 { 600 } else { 40 };
                let dx = sx + mv.1 + d.signed(2 * jitter, jitter as i32);
                let dy = sy + mv.0 + d.signed(2 * jitter, jitter as i32);
                Av1WarpSample {
                    source: (sx, sy),
                    destination: (dx, dy),
                }
            })
            .collect();
        let estimate = av1_warp_estimation(
            &samples,
            mv,
            (mi_col * 4, mi_row * 4),
            (bw as u32, bh as u32),
        );
        let (invalid, m) = match estimate {
            None => (1, [0; 6]),
            Some(m) => (i32::from(av1_setup_shear(&m).is_none()), m),
        };
        format!(
            "{invalid} {} {} {} {} {} {}",
            m[0], m[1], m[2], m[3], m[4], m[5]
        )
    });
}

#[test]
fn warp_filter_matches_libaom() {
    for dsp in [Dsp::new(), Dsp::with_isa(nc_dsp::Isa::Scalar).unwrap()] {
        warp_filter_matches_libaom_with(&dsp);
    }
}

fn warp_filter_matches_libaom_with(dsp: &Dsp) {
    check_group("warp", |d| {
        let (width, height) = (16 + d.below(200), 16 + d.below(200));
        let stride = width + d.below(8);
        let pattern = d.below(4);
        let samples = d.pixels(stride * height, pattern);
        let sub = d.below(2) == 1;
        let size @ (w, h) = (d.pick(&[4, 8, 16, 32, 64]), d.pick(&[4, 8, 16, 32, 64]));
        let position = (d.below(width) & !3, d.below(height) & !3);
        let (model, shear) = loop {
            let model = draw_model(d, width * 65536);
            if let Some(shear) = av1_setup_shear(&model) {
                break (model, shear);
            }
        };
        let reference = PlaneView::new(&samples, stride, width, height).unwrap();
        let mut pixels = vec![0u8; w * h];
        av1_warp(
            dsp,
            &reference,
            &model,
            shear,
            position,
            size,
            (sub, sub),
            &mut PlaneViewMut::new(&mut pixels, w, w, h).unwrap(),
        );
        let mut compound = vec![0i16; w * h];
        av1_warp_compound(
            dsp,
            &reference,
            &model,
            shear,
            position,
            size,
            (sub, sub),
            &mut compound,
            w,
        );
        format!("{} {}", hex(fnv(pixels)), hex(hash_i16(&compound)))
    });
}
