//! The relaxed-SIMD byte-pair dot product the relaxed browser module's VP9
//! interpolation relies on (`nc_dsp` `vp9_mc_simd::simd128_dot`):
//! `i16x8.relaxed_dot_i8x16_i7x16_s` with a signed byte and a 7-bit operand
//! in 0..=127 must give the exact pair sum on this engine. Relaxed SIMD lets
//! engines differ (x86 `pmaddubsw` saturates and reads the second operand as
//! unsigned; Arm does neither) exactly where that precondition fails, so this
//! runs on the engine itself: every (signed byte, 7-bit byte) product, paired
//! with each extreme partner, under node on x86 and on an Arm host.
use crate::check::CheckReport;
use core::arch::wasm32::{
    i8x16_replace_lane, i16x8_extract_lane, i16x8_relaxed_dot_i8x16_i7x16, u8x16_splat, v128,
};

/// Partners for the second product of each pair: the extremes of both
/// signs (the largest sums in magnitude) and zero.
const PARTNERS: [(i8, u8); 4] = [(-128, 127), (127, 127), (0, 0), (-1, 127)];

pub(crate) fn check(report: &mut CheckReport) {
    let mut failures = 0usize;
    for a in i8::MIN..=i8::MAX {
        for b in 0..=127u8 {
            let (mut pixels, mut taps): (v128, v128) = (u8x16_splat(0), u8x16_splat(0));
            for (lane, &(a2, b2)) in PARTNERS.iter().enumerate() {
                // Lanes 2k, 2k+1 form i16 lane k; lanes 4-7 repeat 0-3 swapped.
                pixels = set(set(pixels, 2 * lane, a as u8), 2 * lane + 1, a2 as u8);
                taps = set(set(taps, 2 * lane, b), 2 * lane + 1, b2);
                pixels = set(set(pixels, 2 * lane + 8, a2 as u8), 2 * lane + 9, a as u8);
                taps = set(set(taps, 2 * lane + 8, b2), 2 * lane + 9, b);
            }
            let dot = i16x8_relaxed_dot_i8x16_i7x16(pixels, taps);
            for (lane, &(a2, b2)) in PARTNERS.iter().enumerate() {
                let want = i32::from(a) * i32::from(b) + i32::from(a2) * i32::from(b2);
                for got in [lane_of(dot, lane), lane_of(dot, lane + 4)] {
                    report.cases += 1;
                    if i32::from(got) != want {
                        failures += 1;
                    }
                }
            }
        }
    }
    if failures > 0 {
        report.failures.push(format!(
            "relaxed_dot_i8x16_i7x16: {failures} inexact pair sums"
        ));
    }
}

fn set(v: v128, lane: usize, value: u8) -> v128 {
    // Lane indices are constants in the intrinsic: dispatch on the runtime one.
    macro_rules! at {
        ($($n:literal)*) => {
            match lane { $($n => i8x16_replace_lane::<$n>(v, value as i8),)* _ => unreachable!("16 lanes") }
        };
    }
    at!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
}

fn lane_of(v: v128, lane: usize) -> i16 {
    macro_rules! at {
        ($($n:literal)*) => { match lane { $($n => i16x8_extract_lane::<$n>(v),)* _ => unreachable!("8 lanes") } };
    }
    at!(0 1 2 3 4 5 6 7)
}
