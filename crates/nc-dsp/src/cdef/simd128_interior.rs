//! Interior CDEF with packed-byte constraints and widening paired sums.
//! Like SVT-AV1's BSD-licensed native-byte CDEF, opposite signed constraints
//! are added before widening. Their magnitude is at most 30; four secondary
//! constraints sum to at most 16. Only weighted accumulation needs i16 lanes.
//! Equations and signed rounding follow AV1 §7.15.2 and the scalar oracle.

use super::{CDEF_PADDING, CdefArgs, DIRECTIONS, SEC_TAPS};
use core::arch::wasm32::{
    i8x16_add, i8x16_splat, i8x16_sub, i16x8_add, i16x8_extmul_high_i8x16, i16x8_extmul_low_i8x16,
    i16x8_shr, i16x8_splat, u8x16_max, u8x16_min, u8x16_narrow_i16x8, u8x16_splat, u8x16_sub_sat,
    u16x8_extend_high_u8x16, u16x8_extend_low_u8x16, u16x8_shr, u32x4_extract_lane,
    u64x2_extract_lane, v128, v128_and, v128_load32_lane, v128_load32_zero, v128_load64_lane,
    v128_load64_zero, v128_or,
};

const VECTOR_BYTES: usize = 16;
const ROUND: i16 = 8;
const ROUND_SHIFT: u32 = 4;

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn load<const N: usize>(base: *const u8, stride: usize) -> v128 {
    // SAFETY: the caller checked 16/N complete rows of N bytes in the window.
    unsafe {
        if N == 8 {
            let first = v128_load64_zero(base.cast());
            v128_load64_lane::<1>(first, base.add(stride).cast())
        } else {
            let first = v128_load32_zero(base.cast());
            let second = v128_load32_lane::<1>(first, base.add(stride).cast());
            let third = v128_load32_lane::<2>(second, base.add(2 * stride).cast());
            v128_load32_lane::<3>(third, base.add(3 * stride).cast())
        }
    }
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn store<const N: usize>(value: v128, dst: *mut u8, stride: usize) {
    // SAFETY: exact-size unaligned stores cover only the checked output rows.
    unsafe {
        if N == 8 {
            dst.cast::<u64>()
                .write_unaligned(u64x2_extract_lane::<0>(value));
            dst.add(stride)
                .cast::<u64>()
                .write_unaligned(u64x2_extract_lane::<1>(value));
        } else {
            dst.cast::<u32>()
                .write_unaligned(u32x4_extract_lane::<0>(value));
            dst.add(stride)
                .cast::<u32>()
                .write_unaligned(u32x4_extract_lane::<1>(value));
            dst.add(2 * stride)
                .cast::<u32>()
                .write_unaligned(u32x4_extract_lane::<2>(value));
            dst.add(3 * stride)
                .cast::<u32>()
                .write_unaligned(u32x4_extract_lane::<3>(value));
        }
    }
}

struct State {
    centre: v128,
    low: v128,
    high: v128,
    sum: [v128; 2],
}

#[target_feature(enable = "simd128")]
#[inline]
fn constrain(pixel: v128, centre: v128, strength: i16, shift: u32) -> v128 {
    let positive = u8x16_sub_sat(pixel, centre);
    let negative = u8x16_sub_sat(centre, pixel);
    let magnitude = v128_or(positive, negative);
    // Engines have no byte shift on x86 or ARM and widen, shift and narrow
    // for `u8x16_shr`. One 16-bit shift spills high-byte bits into the low
    // byte; the byte mask removes them, giving exact logical byte shifts,
    // including shifts >= 8 (the mask is then zero).
    let shift = shift.min(8);
    let mask = u8x16_splat((u16::from(u8::MAX) >> shift) as u8);
    let shifted = v128_and(u16x8_shr(magnitude, shift), mask);
    let bound = u8x16_sub_sat(u8x16_splat(strength as u8), shifted);
    // Only one difference is nonzero; subtracting their constrained unsigned
    // magnitudes applies the sign without a compare/negate/bitselect sequence.
    i8x16_sub(u8x16_min(positive, bound), u8x16_min(negative, bound))
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn pair<const N: usize>(
    state: &mut State,
    base: *const u8,
    stride: usize,
    offset: (isize, isize),
    strength: i16,
    shift: u32,
    clip: bool,
) -> v128 {
    let delta = offset.0 * stride as isize + offset.1;
    // SAFETY: both taps and the packed rows lie within the checked window.
    let (a, b) = unsafe {
        (
            load::<N>(base.offset(delta), stride),
            load::<N>(base.offset(-delta), stride),
        )
    };
    if clip {
        state.low = u8x16_min(state.low, u8x16_min(a, b));
        state.high = u8x16_max(state.high, u8x16_max(a, b));
    }
    i8x16_add(
        constrain(a, state.centre, strength, shift),
        constrain(b, state.centre, strength, shift),
    )
}

#[target_feature(enable = "simd128")]
#[inline]
fn accumulate(state: &mut State, pair: v128, weight: i16) {
    let weight = i8x16_splat(weight as i8);
    state.sum[0] = i16x8_add(state.sum[0], i16x8_extmul_low_i8x16(pair, weight));
    state.sum[1] = i16x8_add(state.sum[1], i16x8_extmul_high_i8x16(pair, weight));
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn filter<const N: usize, const DIR: usize>(
    src: &[u8],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    let top = origin
        .checked_sub(CDEF_PADDING * stride + CDEF_PADDING)
        .expect("interior source");
    let side = N + 2 * CDEF_PADDING;
    let window = &src[top..][..(side - 1) * stride + side];
    let dst = &mut dst[..(N - 1) * dst_stride + N];
    let clip = args.pri != 0 && args.sec != 0;
    for row in (0..N).step_by(VECTOR_BYTES / N) {
        // SAFETY: the checked source window includes every tap in the packed
        // rows, and the destination includes every row stored by this group.
        unsafe {
            let base = window
                .as_ptr()
                .add((row + CDEF_PADDING) * stride + CDEF_PADDING);
            let centre = load::<N>(base, stride);
            let mut state = State {
                centre,
                low: centre,
                high: centre,
                sum: [i16x8_splat(0); 2],
            };
            for (k, &weight) in SEC_TAPS.iter().enumerate() {
                if args.pri != 0 {
                    let value = pair::<N>(
                        &mut state,
                        base,
                        stride,
                        DIRECTIONS[DIR][k],
                        args.pri,
                        args.pri_shift,
                        clip,
                    );
                    accumulate(&mut state, value, args.pri_taps[k]);
                }
                if args.sec != 0 {
                    let a = pair::<N>(
                        &mut state,
                        base,
                        stride,
                        DIRECTIONS[(DIR + 2) & 7][k],
                        args.sec,
                        args.sec_shift,
                        clip,
                    );
                    let b = pair::<N>(
                        &mut state,
                        base,
                        stride,
                        DIRECTIONS[(DIR + 6) & 7][k],
                        args.sec,
                        args.sec_shift,
                        clip,
                    );
                    accumulate(&mut state, i8x16_add(a, b), weight);
                }
            }
            let centres = [
                u16x8_extend_low_u8x16(centre),
                u16x8_extend_high_u8x16(centre),
            ];
            for (sum, centre) in state.sum.iter_mut().zip(centres) {
                let rounded = i16x8_add(i16x8_add(*sum, i16x8_shr(*sum, 15)), i16x8_splat(ROUND));
                *sum = i16x8_add(centre, i16x8_shr(rounded, ROUND_SHIFT));
            }
            let mut result = u8x16_narrow_i16x8(state.sum[0], state.sum[1]);
            if clip {
                result = u8x16_max(state.low, u8x16_min(state.high, result));
            }
            store::<N>(result, dst.as_mut_ptr().add(row * dst_stride), dst_stride);
        }
    }
}

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn block<const N: usize>(
    src: &[u8],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: each specialization receives the checked interior contract.
    unsafe {
        match args.dir {
            0 => filter::<N, 0>(src, stride, origin, args, dst, dst_stride),
            1 => filter::<N, 1>(src, stride, origin, args, dst, dst_stride),
            2 => filter::<N, 2>(src, stride, origin, args, dst, dst_stride),
            3 => filter::<N, 3>(src, stride, origin, args, dst, dst_stride),
            4 => filter::<N, 4>(src, stride, origin, args, dst, dst_stride),
            5 => filter::<N, 5>(src, stride, origin, args, dst, dst_stride),
            6 => filter::<N, 6>(src, stride, origin, args, dst, dst_stride),
            _ => filter::<N, 7>(src, stride, origin, args, dst, dst_stride),
        }
    }
}

/// Two interior 4x4 blocks ([`super::CdefPairFn8`] contract), one after the
/// other in one call: a vector holds one 4x4 block, so nothing is shared
/// in the lanes, but the call, the strength set-up and the dispatch on the
/// direction are paid once per unit's U and V instead of twice.
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn block4_pair(
    src: [&[u8]; 2],
    stride: usize,
    origins: [usize; 2],
    args: &CdefArgs,
    dirs: [usize; 2],
    dst: [*mut u8; 2],
    dst_stride: usize,
) {
    const N: usize = 4;
    for i in 0..2 {
        // SAFETY: the caller guarantees four writable rows of four samples
        // at `dst[i]` that nothing else references (`CdefPairFn8`).
        let out = unsafe { core::slice::from_raw_parts_mut(dst[i], (N - 1) * dst_stride + N) };
        let args = args.with_dir(dirs[i] as u8);
        // SAFETY: the kernel checks the block's source window against
        // `src[i]`.
        unsafe { block::<N>(src[i], stride, origins[i], &args, out, dst_stride) };
    }
}
