//! H.264 §8.3.3.4 and §8.3.4.4 plane prediction, shared by encode and decode.
//! The Arm 16-wide schedule uses libavc's weighted half-edge differences and
//! one pair of horizontal ramps for the full block. Reference:
//! `ih264_intra_pred_luma_16x16_mode_plane_av8`, Apache-2.0.
//! Copyright (C) 2015 The Android Open Source Project. Originally developed
//! and contributed by Ittiam Systems Pvt. Ltd. See `THIRD_PARTY.md`.

use nc_common::{BitDepth, Sample};

/// Predict a `W`×`H` plane block: 16×16 luma, 8×8 4:2:0 chroma or 8×16
/// 4:2:2 chroma (§8.3.4.4: `xCF`, `yCF` and the gradient scales 34 and 5
/// follow each side's length, 8 or 16). The top and left edges are in
/// picture order, with the shared top-left sample passed separately.
///
/// # Panics
///
/// If the destination does not contain the complete block.
#[inline]
#[allow(unsafe_code)]
pub fn h264_plane<T: Sample, const W: usize, const H: usize>(
    dst: &mut [T],
    stride: usize,
    (above, left, corner): (&[T; W], &[T; H], T),
    depth: BitDepth,
) {
    const { assert!((W == 8 || W == 16) && (H == 8 || H == 16)) };
    let span = stride.checked_mul(H - 1).and_then(|n| n.checked_add(W));
    assert!(stride >= W && span.is_some_and(|n| n <= dst.len()));
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if T::BYTES == 1 && W == 16 && H == 16 {
        // SAFETY: the sealed one-byte Sample is u8. Each complete,
        // 16-sample edge is readable and the destination span is checked.
        unsafe {
            plane16_neon(
                dst.as_mut_ptr().cast(),
                stride,
                above.as_ptr().cast(),
                left.as_ptr().cast(),
                corner.to_i32() as u8,
            );
        }
        return;
    }
    // H (or V) of one side of `n` samples: `n / 2` weighted differences
    // about its centre, the last one against the corner.
    let gradient = |samples: &[T], n: usize| {
        let half = n / 2;
        let mut sum = half as i32 * (samples[n - 1].to_i32() - corner.to_i32());
        for k in 0..half - 1 {
            sum += (k as i32 + 1) * (samples[half + k].to_i32() - samples[half - 2 - k].to_i32());
        }
        sum
    };
    let scale = |n: usize| if n == 16 { 5 } else { 34 };
    let x_step = (scale(W) * gradient(above, W) + 32) >> 6;
    let y_step = (scale(H) * gradient(left, H) + 32) >> 6;
    // `3 + xCF` and `3 + yCF`.
    let (x_centre, y_centre) = (W as i32 / 2 - 1, H as i32 / 2 - 1);
    let base = 16 * (above[W - 1].to_i32() + left[H - 1].to_i32()) + 16;
    let mut start = base - x_centre * x_step - y_centre * y_step;
    // Use word arithmetic for byte pixels. At every block shape the extrema
    // of the full 8-bit ramp are bounded by 16 * 510 + 16 + 16 * 1355.
    if T::BYTES == 1 {
        let ramp: [i16; W] = core::array::from_fn(|x| (x_step * x as i32) as i16);
        for y in 0..H {
            let offset = y * stride;
            for (sample, step) in dst[offset..offset + W].iter_mut().zip(ramp) {
                *sample = depth.clip_to(i32::from(((start as i16) + step) >> 5));
            }
            start += y_step;
        }
    } else {
        let ramp: [i32; W] = core::array::from_fn(|x| x_step * x as i32);
        for y in 0..H {
            let offset = y * stride;
            for (sample, step) in dst[offset..offset + W].iter_mut().zip(ramp) {
                *sample = depth.clip_to((start + step) >> 5);
            }
            start += y_step;
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[allow(
    unsafe_code,
    clippy::inline_always,
    reason = "fold full-block stores into the output buffer"
)]
#[inline(always)]
unsafe fn plane16_neon(dst: *mut u8, stride: usize, above: *const u8, left: *const u8, corner: u8) {
    use core::arch::aarch64::{
        vaddlvq_s16, vaddq_s16, vcombine_u8, vdup_n_u8, vdupq_n_s16, vext_u8, vget_high_u8,
        vget_low_u8, vgetq_lane_u8, vld1q_s16, vld1q_u8, vmulq_n_s16, vmulq_s16, vqshrun_n_s16,
        vreinterpretq_s16_u16, vrev64_u8, vst1q_u8, vsubl_u8,
    };
    const WEIGHTS: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    const LOW_RAMP: [i16; 8] = [-7, -6, -5, -4, -3, -2, -1, 0];
    // SAFETY: the caller checked both 16-byte edges and all output rows.
    unsafe {
        let weights = vld1q_s16(WEIGHTS.as_ptr());
        let top = vld1q_u8(above);
        let side = vld1q_u8(left);
        let gradient = |edge| {
            let near = vrev64_u8(vext_u8::<7>(vdup_n_u8(corner), vget_low_u8(edge)));
            let differences = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(edge), near));
            vaddlvq_s16(vmulq_s16(differences, weights))
        };
        let x_step = ((gradient(top) * 5 + 32) >> 6) as i16;
        let y_step = ((gradient(side) * 5 + 32) >> 6) as i16;
        let lo = vmulq_n_s16(vld1q_s16(LOW_RAMP.as_ptr()), x_step);
        let hi = vmulq_n_s16(weights, x_step);
        let tail = i16::from(vgetq_lane_u8::<15>(top)) + i16::from(vgetq_lane_u8::<15>(side));
        let mut origin = vdupq_n_s16(16 * (tail + 1) - 7 * y_step);
        let step = vdupq_n_s16(y_step);
        for y in 0..16 {
            let first = vqshrun_n_s16::<5>(vaddq_s16(origin, lo));
            let second = vqshrun_n_s16::<5>(vaddq_s16(origin, hi));
            vst1q_u8(dst.add(y * stride), vcombine_u8(first, second));
            origin = vaddq_s16(origin, step);
        }
    }
}
