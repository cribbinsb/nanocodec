//! Bit-exactness of AV1 film grain synthesis (`nc_dsp::av1_film_grain`)
//! against libaom v3.15.3.
//!
//! `golden/film_grain.txt` comes from a throwaway C harness (kept out of
//! the repository) that ran libaom's `av1_add_film_grain` on frames and
//! parameters drawn from a seeded `SplitMix64` stream (odd and even sizes,
//! every lag, overlap, restricted range, chroma scaling from luma). Per
//! frame it lists FNV-1a hashes of the three output planes; this test
//! draws the same frames and parameters in the same order.

use nc_checkasm::rng::Rng;
use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::av1_film_grain::{Av1FilmGrainParams, Av1ScalingPoints};
use nc_dsp::{Dsp, Isa};

/// The harness's seed.
const SEED: u64 = 14;

struct Draw(Rng);

impl Draw {
    fn below(&mut self, bound: usize) -> usize {
        self.0.below(bound)
    }

    fn plane(&mut self, w: usize, h: usize, pattern: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(w * h);
        for r in 0..h {
            for c in 0..w {
                let b = self.0.byte();
                out.push(match pattern {
                    1 => (b & 1) * 255,
                    2 => 128 + b % 16,
                    3 => (c * 3 + r * 5 + usize::from(b % 4)) as u8,
                    _ => b,
                });
            }
        }
        out
    }

    /// Up to `N` increasing scaling points.
    fn points<const N: usize>(&mut self) -> Av1ScalingPoints<N> {
        let mut points = Av1ScalingPoints::<N> {
            count: self.below(N + 1),
            ..Av1ScalingPoints::default()
        };
        let mut previous = -1i32;
        for k in 0..points.count {
            let value = previous + 1 + self.below(40) as i32;
            if value > 255 {
                points.count = k;
                break;
            }
            points.value[k] = value as u8;
            points.scaling[k] = self.below(256) as u8;
            previous = value;
        }
        points
    }

    /// An auto-regressive coefficient plus 128.
    fn coefficient(&mut self) -> u8 {
        (128 + self.below(129) as i32 - 64) as u8
    }
}

fn fnv(view: &PlaneView<'_>) -> String {
    let hash = view
        .rows()
        .flat_map(|r| r[..view.width()].iter())
        .fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
    format!("{hash:016x}")
}

#[test]
fn film_grain_matches_libaom() {
    let golden = include_str!("golden/film_grain.txt");
    let mut d = Draw(Rng::new(SEED));
    let mut failures = Vec::new();
    for (case, want) in golden.lines().enumerate() {
        let (width, height) = (2 + d.below(300), 2 + d.below(300));
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let mut p = Av1FilmGrainParams {
            grain_seed: d.below(65536) as u16,
            ..Av1FilmGrainParams::default()
        };
        p.y_points = d.points();
        p.chroma_scaling_from_luma = d.below(4) == 0;
        if !p.chroma_scaling_from_luma && p.y_points.count > 0 {
            p.cb_points = d.points();
            p.cr_points = d.points();
        }
        p.grain_scaling_minus_8 = d.below(4) as u8;
        p.ar_coeff_lag = d.below(4) as u8;
        let positions = 2 * usize::from(p.ar_coeff_lag) * (usize::from(p.ar_coeff_lag) + 1);
        let chroma_positions = positions + usize::from(p.y_points.count > 0);
        if p.y_points.count > 0 {
            for k in 0..positions {
                p.ar_coeffs_y_plus_128[k] = d.coefficient();
            }
        }
        if p.chroma_scaling_from_luma || p.cb_points.count > 0 {
            for k in 0..chroma_positions {
                p.ar_coeffs_cb_plus_128[k] = d.coefficient();
            }
        }
        if p.chroma_scaling_from_luma || p.cr_points.count > 0 {
            for k in 0..chroma_positions {
                p.ar_coeffs_cr_plus_128[k] = d.coefficient();
            }
        }
        p.ar_coeff_shift_minus_6 = d.below(4) as u8;
        p.grain_scale_shift = d.below(4) as u8;
        if p.cb_points.count > 0 {
            p.cb_mult = (d.below(256) as u8, d.below(256) as u8, d.below(512) as u16);
        }
        if p.cr_points.count > 0 {
            p.cr_mult = (d.below(256) as u8, d.below(256) as u8, d.below(512) as u16);
        }
        p.overlap = d.below(2) == 1;
        p.clip_to_restricted_range = d.below(2) == 1;
        let pattern = d.below(4);
        let luma = d.plane(width, height, pattern);
        let cb = d.plane(cw, ch, pattern);
        let cr = d.plane(cw, ch, pattern);
        let (mut oy, mut ou, mut ov) = (
            vec![0u8; width * height],
            vec![0u8; cw * ch],
            vec![0u8; cw * ch],
        );
        let input = [
            &PlaneView::new(&luma, width, width, height).unwrap(),
            &PlaneView::new(&cb, cw, cw, ch).unwrap(),
            &PlaneView::new(&cr, cw, cw, ch).unwrap(),
        ];
        for isa in Isa::ALL {
            let Some(dsp) = Dsp::with_isa(isa) else {
                continue;
            };
            dsp.av1_add_film_grain(
                &p,
                input,
                (width, height),
                [
                    &mut PlaneViewMut::new(&mut oy, width, width, height).unwrap(),
                    &mut PlaneViewMut::new(&mut ou, cw, cw, ch).unwrap(),
                    &mut PlaneViewMut::new(&mut ov, cw, cw, ch).unwrap(),
                ],
            );
            let got = format!(
                "{} {} {}",
                fnv(&PlaneView::new(&oy, width, width, height).unwrap()),
                fnv(&PlaneView::new(&ou, cw, cw, ch).unwrap()),
                fnv(&PlaneView::new(&ov, cw, cw, ch).unwrap())
            );
            if got != want {
                failures.push((case, isa));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} frames differ from libaom, first {:?}",
        failures.len(),
        &failures[..failures.len().min(10)]
    );
}
