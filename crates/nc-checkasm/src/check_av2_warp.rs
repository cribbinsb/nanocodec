//! AV2 shares the eight-bit warp schedule with AV1 and uses its own phase table.

use nc_common::BitDepth;
use nc_dsp::Dsp;
use nc_dsp::av2_dec::mc_8tap::{
    ext_warp4x4, ext_warp4x4_with_dsp, ext_warp4x4t, ext_warp4x4t_with_dsp,
};
use nc_dsp::av2_dec::warp::{
    warp_affine_8x8, warp_affine_8x8_with_dsp, warp_affine_8x8t, warp_affine_8x8t_with_dsp,
};

use crate::{CheckReport, rng::Rng};

pub(crate) fn check(dsp: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    const DEST: usize = 16;
    const START: usize = 2 * DEST + 3;
    const DEPTH: BitDepth = BitDepth::EIGHT;
    for stride in [15, 16, 32] {
        let origin = 3 * stride + 3;
        for pattern in 0..4 {
            let src: Vec<_> = (0..stride * 15)
                .map(|i| match pattern {
                    0 => rng.below(256) as u8,
                    1 => (((i % stride + i / stride) & 1) * 255) as u8,
                    2 => 0,
                    _ => 255,
                })
                .collect();
            for (shear, phase) in [
                ([0; 4], (0, 0)),
                ([100, -200, 300, -50], (5632, 42944)),
                ([-1200, 300, 40, 200], (-781, 10243)),
                ([5120, 1200, -3199, -800], (4352, -45328)),
                ([0; 4], (-197_120, 262_655)),
                ([0; 4], (262_655, -197_120)),
            ] {
                let mut expected = [173u8; DEST * DEST];
                let mut actual = expected;
                let mut expected_prep = [2222i16; DEST * DEST];
                let mut actual_prep = expected_prep;
                warp_affine_8x8(
                    &mut expected,
                    START,
                    DEST,
                    &src,
                    origin,
                    stride,
                    shear,
                    phase,
                    DEPTH,
                );
                warp_affine_8x8t(
                    &mut expected_prep[START..],
                    DEST,
                    &src,
                    origin,
                    stride,
                    shear,
                    phase,
                    DEPTH,
                );
                let window = (&src[..], origin, stride);
                warp_affine_8x8_with_dsp(dsp, (&mut actual, START, DEST), window, shear, phase);
                warp_affine_8x8t_with_dsp(
                    dsp,
                    (&mut actual_prep[START..], DEST),
                    window,
                    shear,
                    phase,
                );
                report.cases += 2;
                if actual != expected || actual_prep != expected_prep {
                    report.failures.push(format!(
                        "AV2 warp mismatch: stride {stride}, pattern {pattern}, shear {shear:?}, phase {phase:?}, put {}, prep {}",
                        actual == expected,
                        actual_prep == expected_prep
                    ));
                }
            }
        }
    }
    check_extended(dsp, rng, report);
}

fn check_extended(dsp: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    const DST_STRIDE: usize = 11;
    const DST_OFFSET: usize = 5;
    const DST_LEN: usize = DST_STRIDE * 4 + 7;
    for x in [0, 1, 7, 17, 31, 32, 47, 63] {
        for y in [0, 1, 7, 17, 31, 32, 47, 63] {
            let width: usize = if x == 0 { 4 } else { 11 };
            let height: usize = if y == 0 { 4 } else { 11 };
            for padding in [0, 9] {
                let stride = width + padding;
                let offset = 3 * usize::from(x != 0) + 3 * stride * usize::from(y != 0);
                let len = stride * height;
                for pattern in 0..3 {
                    let src: Vec<_> = (0..len)
                        .map(|i| match pattern {
                            0 => rng.below(256) as u8,
                            1 => 255 * (rng.below(2) as u8),
                            _ => ((i * 13 + i / stride * 29) & 255) as u8,
                        })
                        .collect();
                    let src = (&src[..], offset, stride);
                    let phase = (x, y);
                    let mut expected = [173u8; DST_LEN];
                    let mut actual = expected;
                    let mut expected_prep = [2222i16; DST_LEN];
                    let mut actual_prep = expected_prep;
                    ext_warp4x4(
                        &mut expected,
                        DST_OFFSET,
                        DST_STRIDE,
                        src.0,
                        src.1,
                        src.2,
                        phase,
                        BitDepth::EIGHT,
                    );
                    ext_warp4x4t(
                        &mut expected_prep[DST_OFFSET..],
                        DST_STRIDE,
                        src.0,
                        src.1,
                        src.2,
                        phase,
                        BitDepth::EIGHT,
                    );
                    ext_warp4x4_with_dsp(dsp, (&mut actual, DST_OFFSET, DST_STRIDE), src, phase);
                    ext_warp4x4t_with_dsp(
                        dsp,
                        (&mut actual_prep[DST_OFFSET..], DST_STRIDE),
                        src,
                        phase,
                    );
                    report.cases += 2;
                    if actual != expected || actual_prep != expected_prep {
                        report.failures.push(format!(
                            "AV2 extended warp mismatch: phase {phase:?}, stride {stride}, pattern {pattern}, put {}, prep {}",
                            actual == expected,
                            actual_prep == expected_prep,
                        ));
                    }
                }
            }
        }
    }
}
