//! Bit-exactness of AV1 loop restoration and super-resolution
//! (`nc_dsp::av1_restoration`) against libaom v3.15.3.
//!
//! `golden/restoration.txt` comes from a throwaway C harness (kept out of
//! the repository) that ran libaom's `av1_loop_restoration_save_boundary_lines`,
//! `av1_upscale_normative_rows` (with 1 to 3 tile columns) and
//! `av1_loop_restoration_filter_unit` on frames drawn from a seeded
//! `SplitMix64` stream. Per frame it lists FNV-1a hashes of the three
//! upscaled CDEF planes and of the three restored planes; this test draws
//! the same frames and parameters in the same order.

use nc_checkasm::rng::Rng;
use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::av1_restoration::{
    Av1LoopRestoration, Av1LrFilter, Av1LrScratch, Av1SgrProj, Av1StripeBoundaries, Av1Superres,
    Av1Wiener, av1_lr_unit_count,
};
use nc_dsp::{Dsp, Isa};

/// The harness's seed.
const SEED: u64 = 13;
/// Frames in the golden file.
const FRAMES: usize = 60;

fn fnv(rows: impl Iterator<Item = u8>) -> u64 {
    rows.fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn hash(plane: &PlaneView<'_>) -> String {
    format!(
        "{:016x}",
        fnv(plane
            .rows()
            .flat_map(|r| r[..plane.width()].iter().copied()))
    )
}

/// `Round2(value, shift)` of a dimension.
fn round2(value: usize, shift: bool) -> usize {
    (value + usize::from(shift)) >> u32::from(shift)
}

struct Draw(Rng);

impl Draw {
    fn below(&mut self, bound: usize) -> usize {
        self.0.below(bound)
    }

    fn signed(&mut self, low: i32, count: usize) -> i8 {
        (low + self.below(count) as i32) as i8
    }

    /// A `w`x`h` plane of one of four patterns.
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

    fn unit_filter(&mut self, kind: usize, chroma: bool) -> Av1LrFilter {
        let kind = if kind == 3 { self.below(3) } else { kind };
        match kind {
            1 => {
                let mut taps = || {
                    [
                        if chroma { 0 } else { self.signed(-5, 16) },
                        self.signed(-23, 32),
                        self.signed(-17, 64),
                    ]
                };
                let vertical = taps();
                let horizontal = taps();
                Av1LrFilter::Wiener(Av1Wiener {
                    vertical,
                    horizontal,
                })
            }
            2 => {
                let set = self.below(16) as u8;
                let xqd = [self.signed(-96, 128), self.signed(-32, 128)];
                Av1LrFilter::SgrProj(Av1SgrProj { set, xqd })
            }
            _ => Av1LrFilter::None,
        }
    }
}

/// One frame plane's draws.
struct Plane {
    deblocked: Vec<u8>,
    cdef: Vec<u8>,
}

/// The hash of `plane` (`width` x `rows`) restored in place by `lr` with
/// each of `dsps`.
fn restore_on_every_isa(
    dsps: &[Dsp],
    lr: &Av1LoopRestoration<'_>,
    (plane, width, rows): (&[u8], usize, usize),
    boundaries: &Av1StripeBoundaries,
    sub: bool,
) -> Vec<String> {
    dsps.iter()
        .map(|dsp| {
            let mut restored = plane.to_vec();
            let mut view = PlaneViewMut::new(&mut restored, width, width, rows).unwrap();
            lr.apply(
                dsp,
                &mut view,
                boundaries,
                sub,
                &mut Av1LrScratch::default(),
            );
            hash(&PlaneView::new(&restored, width, width, rows).unwrap())
        })
        .collect()
}

#[test]
fn restoration_and_superres_match_libaom() {
    let golden = include_str!("golden/restoration.txt");
    let mut d = Draw(Rng::new(SEED));
    let mut failures = Vec::new();
    let dsps: Vec<Dsp> = Isa::ALL.into_iter().filter_map(Dsp::with_isa).collect();
    for (case, want) in golden.lines().enumerate().take(FRAMES) {
        let width = 40 + d.below(280);
        let height = 16 + d.below(300);
        let superres = d.below(3) == 0;
        let denom = if superres { 9 + d.below(8) } else { 8 };
        let coded = if superres {
            (width * 8 + denom / 2) / denom
        } else {
            width
        };
        let mi_cols = 2 * ((coded + 7) >> 3);
        let luma_unit = 64 << d.below(3);
        let uv_shift = d.below(2);
        let _tile_columns = d.below(3);
        let mut planes = Vec::new();
        for p in 0..3 {
            let sub = p > 0;
            let readable = (mi_cols * 4) >> u32::from(sub);
            let rows = round2(height, sub);
            let pattern = d.below(4);
            let deblocked = d.plane(readable, rows, pattern);
            let cdef = d.plane(readable, rows, pattern);
            planes.push(Plane { deblocked, cdef });
        }
        let mut upscaled_hashes = Vec::new();
        let mut restored_hashes = Vec::new();
        for (p, plane) in planes.iter().enumerate() {
            let sub = p > 0;
            let readable = (mi_cols * 4) >> u32::from(sub);
            let (coded_w, up_w, rows) =
                (round2(coded, sub), round2(width, sub), round2(height, sub));
            let scaler = superres.then(|| Av1Superres::new(coded_w, up_w));
            let source_width = if superres { readable } else { coded_w };
            let deblocked = PlaneView::new(&plane.deblocked, readable, source_width, rows).unwrap();
            let cdef_source = PlaneView::new(&plane.cdef, readable, source_width, rows).unwrap();
            let mut upscaled = vec![0u8; up_w * rows];
            let mut up_view = PlaneViewMut::new(&mut upscaled, up_w, up_w, rows).unwrap();
            match &scaler {
                Some(s) => s.upscale_plane(&cdef_source, &mut up_view),
                None => up_view.copy_from(&cdef_source),
            }
            let cdef = PlaneView::new(&upscaled, up_w, up_w, rows).unwrap();
            upscaled_hashes.push(hash(&cdef));
            let boundaries = Av1StripeBoundaries::save(&deblocked, sub, scaler.as_ref());
            let unit_size = if sub {
                luma_unit >> uv_shift
            } else {
                luma_unit
            };
            let kind = d.below(4);
            let (columns, unit_rows) = (
                av1_lr_unit_count(unit_size, up_w),
                av1_lr_unit_count(unit_size, rows),
            );
            let filters: Vec<Av1LrFilter> = (0..columns * unit_rows)
                .map(|_| d.unit_filter(kind, sub))
                .collect();
            let lr = Av1LoopRestoration {
                unit_size,
                columns,
                rows: unit_rows,
                filters: &filters,
            };
            // Every ISA this machine runs must restore the same plane.
            let plane_hashes =
                restore_on_every_isa(&dsps, &lr, (&upscaled, up_w, rows), &boundaries, sub);
            assert!(
                plane_hashes.iter().all(|h| *h == plane_hashes[0]),
                "frame {case} plane {p}: ISAs differ: {plane_hashes:?}"
            );
            restored_hashes.push(plane_hashes[0].clone());
        }
        let got = format!(
            "{} {}",
            upscaled_hashes.join(" "),
            restored_hashes.join(" ")
        );
        if got != want {
            failures.push((case, got, want.to_owned()));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {FRAMES} frames differ, first {:?}",
        failures.len(),
        failures.first()
    );
}
