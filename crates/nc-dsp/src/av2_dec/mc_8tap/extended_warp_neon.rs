//! Four-column widened multiply-accumulate schedule from the dav2d NEON
//! extended warp (BSD-2-Clause, `THIRD_PARTY.md`).

use core::arch::aarch64::{
    int8x8_t, int16x4_t, int16x8_t, int32x4_t, uint8x16_t, vaddq_s32, vcombine_s16, vcombine_u8,
    vcreate_u8, vdupq_n_s32, vext_u8, vextq_s16, vget_high_s16, vget_lane_u32, vget_low_s16,
    vhaddq_s16, vld1_s8, vld1_u8, vld1q_u8, vmlal_lane_s16, vmlaq_laneq_s16, vmovl_s8, vmovl_u8,
    vmull_lane_s16, vmulq_laneq_s16, vqmovun_s16, vqtbl1q_u8, vreinterpret_u32_u8,
    vreinterpretq_s16_u16, vrshrn_n_s32, vrshrq_n_s16, vst1_s16,
};

use crate::mc::neon_dotprod::{GATHER, unsigned_dot_lane};

const SIDE: usize = 4;
const TAPS: usize = 8;
const ACTIVE_TAPS: usize = 6;

#[target_feature(enable = "neon")]
#[inline]
fn coefficients(taps: &[i8; TAPS]) -> [int16x4_t; 2] {
    // SAFETY: taps has eight readable bytes.
    let vector = vmovl_s8(unsafe { vld1_s8(taps.as_ptr()) });
    [vget_low_s16(vector), vget_high_s16(vector)]
}

#[target_feature(enable = "neon")]
#[inline]
fn dot(a: [int16x4_t; ACTIVE_TAPS], [lo, hi]: [int16x4_t; 2]) -> int32x4_t {
    let mut sum = vmull_lane_s16::<1>(a[0], lo);
    sum = vmlal_lane_s16::<2>(sum, a[1], lo);
    sum = vmlal_lane_s16::<3>(sum, a[2], lo);
    sum = vmlal_lane_s16::<0>(sum, a[3], hi);
    sum = vmlal_lane_s16::<1>(sum, a[4], hi);
    vmlal_lane_s16::<2>(sum, a[5], hi)
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn horizontal(src: *const u8, coeff: [int16x4_t; 2]) -> int32x4_t {
    // SAFETY: the two eight-byte loads cover samples 1 through 9.
    let (a, b) = unsafe { (vld1_u8(src.add(1)), vld1_u8(src.add(2))) };
    let a = vreinterpretq_s16_u16(vmovl_u8(a));
    let b = vreinterpretq_s16_u16(vmovl_u8(b));
    dot(
        [
            vget_low_s16(a),
            vget_low_s16(vextq_s16::<1>(a, a)),
            vget_low_s16(vextq_s16::<2>(a, a)),
            vget_low_s16(vextq_s16::<3>(a, a)),
            vget_high_s16(a),
            vget_high_s16(b),
        ],
        coeff,
    )
}

#[target_feature(enable = "neon")]
#[inline]
fn dot_coefficients(taps: &[i8; TAPS]) -> (int8x8_t, [uint8x16_t; 2]) {
    // SAFETY: the tables contain the full vectors read here.
    unsafe {
        (
            vld1_s8(taps.as_ptr()),
            [vld1q_u8(GATHER[0].as_ptr()), vld1q_u8(GATHER[1].as_ptr())],
        )
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn horizontal_i8mm<const PADDED: bool>(
    src: *const u8,
    (coeff, [first, last]): (int8x8_t, [uint8x16_t; 2]),
) -> int32x4_t {
    // SAFETY: the selected path guarantees 16 or 11 readable bytes.
    // The I8MM caller checked the feature. Each table uses only bytes 0–10.
    unsafe {
        let pixels = if PADDED {
            vld1q_u8(src)
        } else {
            let last = vld1_u8(src.add(3));
            vcombine_u8(vld1_u8(src), vext_u8::<5>(last, last))
        };
        let low = unsigned_dot_lane::<0>(vdupq_n_s32(0), vqtbl1q_u8(pixels, first), coeff);
        unsigned_dot_lane::<1>(low, vqtbl1q_u8(pixels, last), coeff)
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn vertical_pair(a: [int16x8_t; ACTIVE_TAPS], w: int16x8_t) -> int16x8_t {
    let left = vmulq_laneq_s16::<1>(a[0], w);
    let left = vmlaq_laneq_s16::<2>(left, a[1], w);
    let left = vmlaq_laneq_s16::<3>(left, a[2], w);
    let right = vmulq_laneq_s16::<4>(a[3], w);
    let right = vmlaq_laneq_s16::<5>(right, a[4], w);
    let right = vmlaq_laneq_s16::<6>(right, a[5], w);
    // Each three-tap sum fits signed words. Halving the total before the
    // final rounding is exact because both output shifts are positive.
    vhaddq_s16(left, right)
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn vertical<const PREP: bool>(
    src: *const u8,
    stride: usize,
    taps: &[i8; TAPS],
    mut store: impl FnMut(usize, int16x4_t),
) {
    // The outer tap of each side is zero at every phase. Pair rows 0/2
    // and 1/3, as in dav2d, so each multiply updates two output rows.
    // SAFETY: each of the nine active rows has four readable bytes.
    let rows: [u32; 9] = core::array::from_fn(|i| unsafe {
        src.add((i + 1) * stride).cast::<u32>().read_unaligned()
    });
    let [a, b, c, d, e, f, g] = core::array::from_fn(|i| {
        let bytes = u64::from(rows[i]) | (u64::from(rows[i + 2]) << 32);
        vreinterpretq_s16_u16(vmovl_u8(vcreate_u8(bytes)))
    });
    // SAFETY: there are eight readable coefficients.
    let weights = vmovl_s8(unsafe { vld1_s8(taps.as_ptr()) });
    let round = |x| {
        if PREP {
            vrshrq_n_s16::<2>(x)
        } else {
            vrshrq_n_s16::<6>(x)
        }
    };
    let even = round(vertical_pair([a, b, c, d, e, f], weights));
    let odd = round(vertical_pair([b, c, d, e, f, g], weights));
    store(0, vget_low_s16(even));
    store(1, vget_low_s16(odd));
    store(2, vget_high_s16(even));
    store(3, vget_high_s16(odd));
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn filter<const PREP: bool, H: Copy>(
    src: *const u8,
    stride: usize,
    taps: (Option<&[i8; TAPS]>, Option<&[i8; TAPS]>),
    prepare: impl Fn(&[i8; TAPS]) -> H,
    horizontal: impl Fn(*const u8, H) -> int32x4_t,
    mut store: impl FnMut(usize, int16x4_t),
) {
    // SAFETY: the public wrapper validates every row of the active axes.
    unsafe {
        match taps {
            (Some(x), None) => {
                let coeff = prepare(x);
                for row in 0..SIDE {
                    let sum = horizontal(src.add(row * stride), coeff);
                    let pixels = if PREP {
                        vrshrn_n_s32::<3>(sum)
                    } else {
                        // The first stage's half-up bias survives final rounding.
                        vrshrn_n_s32::<7>(vaddq_s32(sum, vdupq_n_s32(4)))
                    };
                    store(row, pixels);
                }
            }
            (Some(x), Some(y)) => {
                let coeff = coefficients(y);
                let horizontal_coeff = prepare(x);
                let [a, b, c, d, e, f, g, h, i] = core::array::from_fn(|row| {
                    let sum = horizontal(src.add((row + 1) * stride), horizontal_coeff);
                    vrshrn_n_s32::<3>(sum)
                });
                let rows = [
                    [a, b, c, d, e, f],
                    [b, c, d, e, f, g],
                    [c, d, e, f, g, h],
                    [d, e, f, g, h, i],
                ];
                for (row, samples) in rows.into_iter().enumerate() {
                    let sum = dot(samples, coeff);
                    let pixels = if PREP {
                        vrshrn_n_s32::<7>(sum)
                    } else {
                        vrshrn_n_s32::<11>(sum)
                    };
                    store(row, pixels);
                }
            }
            (None, Some(y)) => vertical::<PREP>(src, stride, y, store),
            (None, None) => unreachable!("zero-phase extended warp is a copy"),
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn dispatch<const PREP: bool>(
    src: *const u8,
    stride: usize,
    taps: (Option<&[i8; TAPS]>, Option<&[i8; TAPS]>),
    padded: bool,
    store: impl FnMut(usize, int16x4_t),
) {
    // SAFETY: the spans are checked and I8MM is used only after detection.
    unsafe {
        if taps.0.is_some() && std::arch::is_aarch64_feature_detected!("i8mm") {
            if padded {
                filter::<PREP, _>(
                    src,
                    stride,
                    taps,
                    |taps| dot_coefficients(taps),
                    |p, c| horizontal_i8mm::<true>(p, c),
                    store,
                );
            } else {
                filter::<PREP, _>(
                    src,
                    stride,
                    taps,
                    |taps| dot_coefficients(taps),
                    |p, c| horizontal_i8mm::<false>(p, c),
                    store,
                );
            }
        } else {
            filter::<PREP, _>(
                src,
                stride,
                taps,
                |taps| coefficients(taps),
                |p, c| horizontal(p, c),
                store,
            );
        }
    }
}

/// # Safety
/// NEON is enabled and the source and output spans were checked.
#[target_feature(enable = "neon")]
pub(super) unsafe fn put(
    dst: *mut u8,
    stride: usize,
    src: *const u8,
    src_stride: usize,
    h: Option<&[i8; TAPS]>,
    v: Option<&[i8; TAPS]>,
    padded: bool,
) {
    // SAFETY: the filter and each four-byte store keep to checked spans.
    unsafe {
        dispatch::<false>(src, src_stride, (h, v), padded, |row, words| {
            let bytes = vqmovun_s16(vcombine_s16(words, words));
            dst.add(row * stride)
                .cast::<u32>()
                .write_unaligned(vget_lane_u32::<0>(vreinterpret_u32_u8(bytes)));
        });
    }
}

/// # Safety
/// As `put`, with a word output.
#[target_feature(enable = "neon")]
pub(super) unsafe fn prep(
    dst: *mut i16,
    stride: usize,
    src: *const u8,
    src_stride: usize,
    h: Option<&[i8; TAPS]>,
    v: Option<&[i8; TAPS]>,
    padded: bool,
) {
    // SAFETY: the filter and each four-word store keep to checked spans.
    unsafe {
        dispatch::<true>(src, src_stride, (h, v), padded, |row, words| {
            vst1_s16(dst.add(row * stride), words)
        });
    }
}
