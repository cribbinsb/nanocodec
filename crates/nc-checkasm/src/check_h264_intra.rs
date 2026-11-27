//! H.264 `Intra_8x8` prediction against the scalar reference: every mode
//! the neighbours allow, every availability combination, random, extreme
//! and alternating neighbours.
use crate::{check::CheckReport, rng::Rng};
use nc_common::{BitDepth, Sample};
use nc_dsp::{Dsp, H264Intra8x8Edges, h264_intra_8x8_mode, h264_plane};

/// Random neighbour sets per availability combination.
const RANDOM_EDGES: usize = 8;
/// Neighbour patterns: random sets, then all 255, all 0, alternating.
const PATTERNS: usize = RANDOM_EDGES + 3;
/// Availability flags: top, top-right, left, corner.
const AVAILABILITY_COMBINATIONS: u8 = 16;

fn sample(pattern: usize, i: usize, rng: &mut Rng) -> u8 {
    match pattern {
        p if p < RANDOM_EDGES => rng.byte(),
        p if p == RANDOM_EDGES => u8::MAX,
        p if p == RANDOM_EDGES + 1 => 0,
        _ => {
            if i.is_multiple_of(2) {
                u8::MAX
            } else {
                0
            }
        }
    }
}

pub(crate) fn check(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    for flags in 0..AVAILABILITY_COMBINATIONS {
        for pattern in 0..PATTERNS {
            let mut i = 0;
            let mut next = || {
                i += 1;
                sample(pattern, i, rng)
            };
            let edges = H264Intra8x8Edges {
                top: core::array::from_fn(|_| next()),
                left: core::array::from_fn(|_| next()),
                corner: next(),
                has_top: flags & 1 != 0,
                has_top_right: flags & 2 != 0,
                has_left: flags & 4 != 0,
                has_corner: flags & 8 != 0,
            };
            let refs = edges.filtered();
            for mode in (0..h264_intra_8x8_mode::COUNT).filter(|&m| refs.allows(m)) {
                let run = |dsp: &Dsp| {
                    let mut out = [0u8; 64];
                    dsp.h264_intra_8x8(&refs, mode, &mut out);
                    out
                };
                report.cases += 1;
                if run(candidate) != run(reference) {
                    report.failures.push(format!(
                        "{} h264_intra_8x8 mode={mode} flags={flags:#06b} pattern={pattern}",
                        candidate.isa()
                    ));
                }
            }
        }
    }
    let rng = &mut rng.clone();
    check_plane::<u8, 8>(BitDepth::EIGHT, rng, report);
    check_plane::<u8, 16>(BitDepth::EIGHT, rng, report);
    check_plane::<u16, 8>(BitDepth::TEN, rng, report);
    check_plane::<u16, 16>(BitDepth::TEN, rng, report);
    check_plane::<u16, 8>(BitDepth::new(12).unwrap(), rng, report);
    check_plane::<u16, 16>(BitDepth::new(12).unwrap(), rng, report);
}

fn check_plane<T: Sample, const N: usize>(
    depth: BitDepth,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let stride = N + 5;
    for pattern in 0..16 {
        let mut edge = |index: usize| {
            T::from_i32(match pattern {
                0 => {
                    if index < N / 2 {
                        0
                    } else {
                        depth.max()
                    }
                }
                1 => {
                    if index < N / 2 {
                        depth.max()
                    } else {
                        0
                    }
                }
                2 => {
                    if index.is_multiple_of(2) {
                        0
                    } else {
                        depth.max()
                    }
                }
                _ => rng.below(depth.max() as usize + 1) as i32,
            })
        };
        let above: [T; N] = core::array::from_fn(&mut edge);
        let left: [T; N] = core::array::from_fn(|i| edge(N - i));
        let corner = edge(N);
        let half = N / 2;
        let k = if N == 16 { 5 } else { 34 };
        let slope = |values: &[T; N]| {
            let at = |i| if i == 0 { corner } else { values[i - 1] };
            let sum: i32 = (1..=half)
                .map(|i| i as i32 * (at(half + i).to_i32() - at(half - i).to_i32()))
                .sum();
            (k * sum + 32) >> 6
        };
        let (dx, dy) = (slope(&above), slope(&left));
        let base = 16 * (above[N - 1].to_i32() + left[N - 1].to_i32());
        let centre = half as i32 - 1;
        let mut expected = vec![T::from_i32(123); stride * (N + 1)];
        let mut actual = expected.clone();
        for y in 0..N {
            for x in 0..N {
                let value = base + dx * (x as i32 - centre) + dy * (y as i32 - centre);
                expected[y * stride + x + 1] = depth.clip_to((value + 16) >> 5);
            }
        }
        h264_plane(&mut actual[1..], stride, (&above, &left, corner), depth);
        report.cases += 1;
        if actual != expected {
            report.failures.push(format!(
                "H.264 plane {N}x{N} depth {} pattern {pattern}",
                depth.bits()
            ));
        }
    }
}
