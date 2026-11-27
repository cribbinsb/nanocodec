//! Shared 10-bit HEVC and VVC luma interpolation with 16-bit partial products.
//! The four-row vertical and paired horizontal schedules follow
//! libhevc's `ihevc_hbd_inter_pred_filters_neon_intr.c` (Apache-2.0).

#![allow(unsafe_code)]

use nc_common::BitDepth;

use crate::high_depth::Convolve16Fn;
use crate::high_depth::vector;
use crate::mc::{HEVC_LUMA, HevcLumaTaps, SeparableTaps, VVC_LUMA, VvcLumaTaps};
use core::arch::aarch64::{
    int16x8_t, uint16x4_t, uint16x8_t, vdup_n_u16, vdupq_n_u16, veorq_u16, vext_u16, vextq_u16,
    vld1_u16, vld1q_s16, vld1q_u16, vmin_u16, vminq_u16, vmlal_laneq_s16, vmlaq_laneq_u16,
    vmull_laneq_s16, vqaddq_s16, vqrshrun_n_s32, vreinterpret_s16_u16, vreinterpretq_s16_u16,
    vreinterpretq_u16_s16, vrshrq_n_u16, vst1_u16, vst1q_u16,
};

const HEVC_PHASES: u16 = narrow_phases(&HEVC_LUMA);
const VVC_PHASES: u16 = narrow_phases(&VVC_LUMA);

pub(super) const HEVC: [Convolve16Fn; 5] = [
    predict::<HevcLumaTaps, 4, HEVC_PHASES>,
    predict::<HevcLumaTaps, 8, HEVC_PHASES>,
    predict::<HevcLumaTaps, 16, HEVC_PHASES>,
    predict::<HevcLumaTaps, 32, HEVC_PHASES>,
    predict::<HevcLumaTaps, 64, HEVC_PHASES>,
];

pub(super) const VVC: [Convolve16Fn; 5] = [
    predict::<VvcLumaTaps, 4, VVC_PHASES>,
    predict::<VvcLumaTaps, 8, VVC_PHASES>,
    predict::<VvcLumaTaps, 16, VVC_PHASES>,
    predict::<VvcLumaTaps, 32, VVC_PHASES>,
    predict::<VvcLumaTaps, 64, VVC_PHASES>,
];

/// A group whose absolute coefficients sum to at most 64 can
/// accumulate 10-bit samples centered at 512 in signed halfwords.
const fn narrow_phases(table: &[[i16; 8]]) -> u16 {
    let mut mask = 0;
    let mut phase = 1;
    while phase < table.len() {
        let t = table[phase];
        let a = t[0].abs() + t[1].abs() + t[2].abs() + t[4].abs();
        let b = t[3].abs() + t[5].abs() + t[6].abs() + t[7].abs();
        if a <= 64 && b <= 64 {
            mask |= 1 << phase;
        }
        phase += 1;
    }
    mask
}

#[target_feature(enable = "neon")]
unsafe fn predict<T: SeparableTaps<8>, const W: usize, const PHASES: u16>(
    src: &[u16],
    ss: usize,
    dst: &mut [u16],
    ds: usize,
    height: usize,
    (fx, fy): (usize, usize),
    depth: BitDepth,
) {
    // SAFETY: the high-depth dispatcher checked the eight-tap source window
    // and destination. The narrow partial products require Main 10 samples.
    unsafe {
        let phase = fx | fy;
        if depth == BitDepth::TEN
            && (fx == 0) != (fy == 0)
            && height.is_multiple_of(4)
            && (W == 4 || PHASES == HEVC_PHASES || (PHASES >> phase) & 1 != 0)
        {
            if W == 4 {
                narrow4(
                    (src.as_ptr(), ss),
                    (dst.as_mut_ptr(), ds),
                    height,
                    vld1q_s16(T::TABLE[phase].as_ptr()),
                    fx != 0,
                );
                return;
            }
            let taps = &T::TABLE[phase];
            let first = taps[0] + taps[1] + taps[2] + taps[4];
            let filter = LumaFilter {
                coeffs: vld1q_u16(taps.as_ptr().cast()),
                first_bias: vdupq_n_u16((-i32::from(first) * 512) as u16),
                second_bias: vdupq_n_u16((i32::from(first - 64) * 512) as u16),
            };
            if fx == 0 {
                vertical::<W>(
                    (src.as_ptr().add(3), ss),
                    (dst.as_mut_ptr(), ds),
                    height,
                    filter,
                );
            } else {
                let source = (src.as_ptr().add(3 * ss), ss);
                let output = (dst.as_mut_ptr(), ds);
                if W == 8 || W == 64 {
                    horizontal::<W, 2>(source, output, height, filter);
                } else {
                    horizontal::<W, 1>(source, output, height, filter);
                }
            }
        } else {
            vector::neon::predict::<T, 8, W>(src, ss, dst, ds, height, (fx, fy), depth);
        }
    }
}

#[derive(Clone, Copy)]
struct LumaFilter {
    coeffs: uint16x8_t,
    first_bias: uint16x8_t,
    second_bias: uint16x8_t,
}

#[target_feature(enable = "neon")]
#[inline]
fn filter8(a: [uint16x8_t; 8], filter: LumaFilter) -> uint16x8_t {
    // The absolute tap sums of the two groups are at most 64 in every
    // fractional phase. Bias each group as if its input were in -512..=511.
    // Both centered partials fit a signed word, even when individual MACs
    // wrap. Saturating their sum implements the final clip about 512.
    let c = filter.coeffs;
    let left = vmlaq_laneq_u16::<0>(filter.first_bias, a[0], c);
    let left = vmlaq_laneq_u16::<1>(left, a[1], c);
    let left = vmlaq_laneq_u16::<2>(left, a[2], c);
    let left = vmlaq_laneq_u16::<4>(left, a[4], c);
    let right = vmlaq_laneq_u16::<3>(filter.second_bias, a[3], c);
    let right = vmlaq_laneq_u16::<5>(right, a[5], c);
    let right = vmlaq_laneq_u16::<6>(right, a[6], c);
    let right = vmlaq_laneq_u16::<7>(right, a[7], c);
    let sum = vqaddq_s16(vreinterpretq_s16_u16(left), vreinterpretq_s16_u16(right));
    let unsigned = veorq_u16(vreinterpretq_u16_s16(sum), vdupq_n_u16(0x8000));
    vminq_u16(vrshrq_n_u16::<6>(unsigned), vdupq_n_u16(1023))
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn vertical<const W: usize>(
    (source, ss): (*const u16, usize),
    (output, ds): (*mut u16, usize),
    height: usize,
    filter: LumaFilter,
) {
    // SAFETY: height is a multiple of four, width a multiple of eight, and
    // the checked source includes all seven padding rows.
    unsafe {
        for x in (0..W).step_by(8) {
            let source = source.add(x);
            let output = output.add(x);
            let mut kept: [uint16x8_t; 7] = core::array::from_fn(|k| vld1q_u16(source.add(k * ss)));
            for y in (0..height).step_by(4) {
                let next = source.add((y + 7) * ss);
                let a = [
                    kept[0],
                    kept[1],
                    kept[2],
                    kept[3],
                    kept[4],
                    kept[5],
                    kept[6],
                    vld1q_u16(next),
                    vld1q_u16(next.add(ss)),
                    vld1q_u16(next.add(2 * ss)),
                    vld1q_u16(next.add(3 * ss)),
                ];
                let out = output.add(y * ds);
                for j in 0..4 {
                    let taps = [
                        a[j],
                        a[j + 1],
                        a[j + 2],
                        a[j + 3],
                        a[j + 4],
                        a[j + 5],
                        a[j + 6],
                        a[j + 7],
                    ];
                    vst1q_u16(out.add(j * ds), filter8(taps, filter));
                }
                kept = [a[4], a[5], a[6], a[7], a[8], a[9], a[10]];
            }
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn align8(a: uint16x8_t, b: uint16x8_t) -> [uint16x8_t; 8] {
    [
        a,
        vextq_u16::<1>(a, b),
        vextq_u16::<2>(a, b),
        vextq_u16::<3>(a, b),
        vextq_u16::<4>(a, b),
        vextq_u16::<5>(a, b),
        vextq_u16::<6>(a, b),
        vextq_u16::<7>(a, b),
    ]
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn horizontal<const W: usize, const ROWS: usize>(
    (source, ss): (*const u16, usize),
    (output, ds): (*mut u16, usize),
    height: usize,
    filter: LumaFilter,
) {
    // SAFETY: each vector uses the W+7 checked words of its row. The last
    // unused word of the second load may reach into the next checked
    // padding row. Horizontal output stops four rows before the end.
    unsafe {
        for y in (0..height).step_by(ROWS) {
            let p = source.add(y * ss);
            let q = p.add(ss);
            let out = output.add(y * ds);
            let mut top = vld1q_u16(p);
            let mut bottom = if ROWS == 2 { vld1q_u16(q) } else { top };
            for x in (0..W).step_by(8) {
                let next_top = vld1q_u16(p.add(x + 8));
                vst1q_u16(out.add(x), filter8(align8(top, next_top), filter));
                top = next_top;
                if ROWS == 2 {
                    let next_bottom = vld1q_u16(q.add(x + 8));
                    vst1q_u16(
                        out.add(x + ds),
                        filter8(align8(bottom, next_bottom), filter),
                    );
                    bottom = next_bottom;
                }
            }
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn filter4(a: [uint16x4_t; 8], coeffs: int16x8_t) -> uint16x4_t {
    let a = a.map(|v| vreinterpret_s16_u16(v));
    let sum = vmull_laneq_s16::<0>(a[0], coeffs);
    let sum = vmlal_laneq_s16::<1>(sum, a[1], coeffs);
    let sum = vmlal_laneq_s16::<2>(sum, a[2], coeffs);
    let sum = vmlal_laneq_s16::<3>(sum, a[3], coeffs);
    let sum = vmlal_laneq_s16::<4>(sum, a[4], coeffs);
    let sum = vmlal_laneq_s16::<5>(sum, a[5], coeffs);
    let sum = vmlal_laneq_s16::<6>(sum, a[6], coeffs);
    let sum = vmlal_laneq_s16::<7>(sum, a[7], coeffs);
    vmin_u16(vqrshrun_n_s32::<6>(sum), vdup_n_u16(1023))
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn narrow4(
    (source, ss): (*const u16, usize),
    (output, ds): (*mut u16, usize),
    height: usize,
    coeffs: int16x8_t,
    along_rows: bool,
) {
    // SAFETY: all four output samples and their seven neighbours are
    // inside the checked source and output footprints.
    unsafe {
        if along_rows {
            let source = source.add(3 * ss);
            for y in 0..height {
                let p = source.add(y * ss);
                let a = vld1_u16(p);
                let b = vld1_u16(p.add(4));
                let c = vld1_u16(p.add(8));
                let taps = [
                    a,
                    vext_u16::<1>(a, b),
                    vext_u16::<2>(a, b),
                    vext_u16::<3>(a, b),
                    b,
                    vext_u16::<1>(b, c),
                    vext_u16::<2>(b, c),
                    vext_u16::<3>(b, c),
                ];
                vst1_u16(output.add(y * ds), filter4(taps, coeffs));
            }
        } else {
            let source = source.add(3);
            for y in (0..height).step_by(4) {
                let p = source.add(y * ss);
                let a: [uint16x4_t; 11] = core::array::from_fn(|k| vld1_u16(p.add(k * ss)));
                for j in 0..4 {
                    let taps = [
                        a[j],
                        a[j + 1],
                        a[j + 2],
                        a[j + 3],
                        a[j + 4],
                        a[j + 5],
                        a[j + 6],
                        a[j + 7],
                    ];
                    vst1_u16(output.add((y + j) * ds), filter4(taps, coeffs));
                }
            }
        }
    }
}
