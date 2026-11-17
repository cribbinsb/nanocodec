//! Full dav1d NEON CDEF filters for AV1/AV2 eight-bit samples. Fully edged
//! blocks use byte constraints and two bounded byte accumulators; frame
//! edges use word constraints so the unavailable-sample sentinel survives.
//! Interiors borrow their unfiltered source; edges borrow the caller's
//! 12/8-stride padded windows. Other edge pitches use dav1d's compact
//! 16/8-stride scratch layout. Reimplemented from
//! dav1d 1.5.4 `src/arm/64/cdef.S` and `cdef_tmpl.S`, BSD-2-Clause,
//! Copyright 2018 VideoLAN/dav1d authors, 2019/2020 Martin Storsjo;
//! see `THIRD_PARTY.md`. High-depth filtering stays in the shared word family.

use core::arch::aarch64::{
    int8x16_t, int16x8_t, uint8x16_t, uint16x8_t, vabdq_u8, vabdq_u16, vaddq_s16, vbslq_s8,
    vbslq_u8, vcgtq_u8, vcltzq_s8, vcltzq_s16, vcombine_s16, vcombine_u8, vcreate_u8, vdupq_n_s8,
    vdupq_n_s16, vdupq_n_u8, vdupq_n_u16, vhaddq_s8, vld1_s16, vld1_u8, vld1q_s16, vmaxq_s16,
    vmaxq_u8, vminq_s16, vminq_u8, vminq_u16, vmlaq_s8, vmlaq_s16, vmovn_s16, vnegq_s8, vnegq_s16,
    vqsubq_u8, vqsubq_u16, vreinterpret_u8_s8, vreinterpret_u32_u8, vreinterpretq_s8_u8,
    vreinterpretq_s16_u16, vreinterpretq_u8_s8, vreinterpretq_u16_s16, vreinterpretq_u32_u8,
    vreinterpretq_u64_u8, vrhaddq_s8, vrshrq_n_s8, vrshrq_n_s16, vshlq_u8, vshlq_u16, vsqaddq_u8,
    vst1_lane_u32, vst1_u8, vst1q_lane_u32, vst1q_lane_u64, vsubq_s16,
};

use super::{CDEF_PADDING, CdefArgs, DIRECTIONS, SEC_TAPS, neon_padding};

#[derive(Clone, Copy)]
struct ByteAccum {
    first: int8x16_t,
    second: int8x16_t,
    min: uint8x16_t,
    max: uint8x16_t,
}

#[derive(Clone, Copy)]
struct WordAccum {
    sum: int16x8_t,
    min: uint16x8_t,
    max: int16x8_t,
}

#[inline]
fn check_output<const N: usize>(dst: &[u8], stride: usize) {
    // An arbitrary stride can overflow usize before it is rejected.
    // One widened bound preserves the complete footprint check.
    let end = (N - 1) as u128 * stride as u128 + N as u128;
    assert!(
        stride >= N && end <= dst.len() as u128,
        "CDEF output too small"
    );
}

/// dav1d repeats six directions so dir+2/dir+6 need no wrap. Compact
/// sample offsets fit i8 for both16/8 strides, rather than machine words.
const fn direction_table<const S: usize>() -> [[i8; 2]; 14] {
    let mut table = [[0; 2]; 14];
    let mut dir = 0;
    while dir < table.len() {
        let mut tap = 0;
        while tap < 2 {
            let (row, col) = DIRECTIONS[dir & 7][tap];
            table[dir][tap] = (row * S as isize + col) as i8;
            tap += 1;
        }
        dir += 1;
    }
    table
}

const LUMA_DIRECTIONS: [[i8; 2]; 14] = direction_table::<16>();
const PADDED_LUMA_DIRECTIONS: [[i8; 2]; 14] = direction_table::<12>();
const CHROMA_DIRECTIONS: [[i8; 2]; 14] = direction_table::<8>();

#[allow(clippy::inline_always)]
#[inline(always)]
fn offsets<const S: usize>(dir: usize) -> [[i8; 2]; 3] {
    let dir = dir & 7;
    let table = if S == 16 {
        &LUMA_DIRECTIONS
    } else if S == 12 {
        &PADDED_LUMA_DIRECTIONS
    } else {
        &CHROMA_DIRECTIONS
    };
    [table[dir], table[dir + 2], table[dir + 6]]
}

/// Two eight-pixel rows or four four-pixel rows. Load rows independently
/// before combining them: a chain of lane loads serializes on Cortex-A78.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_bytes<const N: usize>(p: *const u8, stride: usize) -> uint8x16_t {
    // SAFETY: each row has N initialized bytes in the checked source window.
    unsafe {
        if N == 8 {
            vcombine_u8(vld1_u8(p), vld1_u8(p.add(stride)))
        } else {
            let first = u64::from(u32::from_le(p.cast::<u32>().read_unaligned()));
            let second = u64::from(u32::from_le(p.add(stride).cast::<u32>().read_unaligned()));
            let third = u64::from(u32::from_le(
                p.add(2 * stride).cast::<u32>().read_unaligned(),
            ));
            let fourth = u64::from(u32::from_le(
                p.add(3 * stride).cast::<u32>().read_unaligned(),
            ));
            vcombine_u8(
                vcreate_u8(first | (second << 32)),
                vcreate_u8(third | (fourth << 32)),
            )
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_bytes<const N: usize>(p: *mut u8, stride: usize, value: uint8x16_t) {
    // SAFETY: the entry checked all N output rows and their N samples.
    unsafe {
        if N == 8 {
            let words = vreinterpretq_u64_u8(value);
            vst1q_lane_u64::<0>(p.cast(), words);
            vst1q_lane_u64::<1>(p.add(stride).cast(), words);
        } else {
            let words = vreinterpretq_u32_u8(value);
            vst1q_lane_u32::<0>(p.cast(), words);
            vst1q_lane_u32::<1>(p.add(stride).cast(), words);
            vst1q_lane_u32::<2>(p.add(2 * stride).cast(), words);
            vst1q_lane_u32::<3>(p.add(3 * stride).cast(), words);
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn constrain_byte(
    p: uint8x16_t,
    x: uint8x16_t,
    strength: uint8x16_t,
    shift: int8x16_t,
) -> int8x16_t {
    let magnitude = vabdq_u8(p, x);
    let clip = vqsubq_u8(strength, vshlq_u8(magnitude, shift));
    let value = vminq_u8(magnitude, clip);
    let negative = vnegq_s8(vreinterpretq_s8_u8(value));
    vreinterpretq_s8_u8(vbslq_u8(
        vcgtq_u8(x, p),
        vreinterpretq_u8_s8(negative),
        value,
    ))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn byte_taps<const N: usize, const CLIP: bool>(
    p: *const u8,
    stride: usize,
    off: isize,
    x: uint8x16_t,
    strength: uint8x16_t,
    shift: int8x16_t,
    weight: i8,
    acc: &mut ByteAccum,
) {
    // SAFETY: the checked source window contains the two-row/four-row
    // group and every +/-off neighbor, within its two-sample halo.
    unsafe {
        let (a, b) = (
            load_bytes::<N>(p.offset(off), stride),
            load_bytes::<N>(p.offset(-off), stride),
        );
        if CLIP {
            acc.min = vminq_u8(acc.min, vminq_u8(a, b));
            acc.max = vmaxq_u8(acc.max, vmaxq_u8(a, b));
        }
        let weight = vdupq_n_s8(weight);
        acc.first = vmlaq_s8(acc.first, constrain_byte(a, x, strength, shift), weight);
        acc.second = vmlaq_s8(acc.second, constrain_byte(b, x, strength, shift), weight);
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn bytes<const N: usize, const DIR: usize, const PRI: bool, const SEC: bool>(
    src: *const u8,
    stride: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    let mut directions = [[0isize; 2]; 3];
    for (axis, delta) in [0, 2, 6].into_iter().enumerate() {
        for tap in 0..2 {
            let (row, col) = DIRECTIONS[(DIR + delta) & 7][tap];
            directions[axis][tap] = row * stride as isize + col;
        }
    }
    let pri = vdupq_n_u8(args.pri as u8);
    let sec = vdupq_n_u8(args.sec as u8);
    let pri_shift = vdupq_n_s8(-(args.pri_shift.min(8) as i8));
    let sec_shift = vdupq_n_s8(-(args.sec_shift.min(8) as i8));
    for row in (0..N).step_by(16 / N) {
        // SAFETY: src starts at the checked block center; the full group
        // and all tap offsets stay inside initialized source rows. Output
        // was checked once by the entry before pointer-based stores.
        unsafe {
            let p = src.add(row * stride);
            let x = load_bytes::<N>(p, stride);
            // Each side contributes at most 15*(4+2)+4*2*(2+1)=114.
            // Starting the first at -1 keeps it in [-115,113], the second
            // in [-114,114]. Every intermediate therefore fits signed bytes.
            let mut acc = ByteAccum {
                first: vdupq_n_s8(-1),
                second: vdupq_n_s8(0),
                min: x,
                max: x,
            };
            for k in 0..2 {
                if PRI {
                    byte_taps::<N, SEC>(
                        p,
                        stride,
                        directions[0][k],
                        x,
                        pri,
                        pri_shift,
                        args.pri_taps[k] as i8,
                        &mut acc,
                    );
                }
                if SEC {
                    byte_taps::<N, PRI>(
                        p,
                        stride,
                        directions[1][k],
                        x,
                        sec,
                        sec_shift,
                        SEC_TAPS[k] as i8,
                        &mut acc,
                    );
                    byte_taps::<N, PRI>(
                        p,
                        stride,
                        directions[2][k],
                        x,
                        sec,
                        sec_shift,
                        SEC_TAPS[k] as i8,
                        &mut acc,
                    );
                }
            }
            // Rounded and unrounded halves recover the negative correction
            // from the -1 bias without ever widening the nine-bit total.
            let rounded = vrhaddq_s8(acc.first, acc.second);
            let truncated = vhaddq_s8(acc.first, acc.second);
            let half = vbslq_s8(vcltzq_s8(rounded), truncated, rounded);
            let mut result = vsqaddq_u8(x, vrshrq_n_s8::<3>(half));
            if PRI && SEC {
                result = vmaxq_u8(acc.min, vminq_u8(acc.max, result));
            }
            store_bytes::<N>(dst.as_mut_ptr().add(row * dst_stride), dst_stride, result);
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_words<const N: usize, const S: usize>(p: *const i16) -> int16x8_t {
    // SAFETY: an eight-word row or two four-word rows are initialized.
    unsafe {
        if N == 8 {
            vld1q_s16(p)
        } else {
            vcombine_s16(vld1_s16(p), vld1_s16(p.add(S)))
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn constrain_word(
    p: int16x8_t,
    x: int16x8_t,
    strength: uint16x8_t,
    shift: int16x8_t,
) -> int16x8_t {
    let magnitude = vabdq_u16(vreinterpretq_u16_s16(p), vreinterpretq_u16_s16(x));
    let clip = vreinterpretq_s16_u16(vqsubq_u16(strength, vshlq_u16(magnitude, shift)));
    vmaxq_s16(vminq_s16(vsubq_s16(p, x), clip), vnegq_s16(clip))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn word_taps<const N: usize, const S: usize, const CLIP: bool>(
    p: *const i16,
    off: isize,
    x: int16x8_t,
    strength: uint16x8_t,
    shift: int16x8_t,
    weight: i16,
    acc: &mut WordAccum,
) {
    // SAFETY: all tap rows/payloads fit the compact padded word window.
    unsafe {
        let (a, b) = (
            load_words::<N, S>(p.offset(off)),
            load_words::<N, S>(p.offset(-off)),
        );
        if CLIP {
            acc.min = vminq_u16(
                acc.min,
                vminq_u16(vreinterpretq_u16_s16(a), vreinterpretq_u16_s16(b)),
            );
            acc.max = vmaxq_s16(acc.max, vmaxq_s16(a, b));
        }
        let weight = vdupq_n_s16(weight);
        acc.sum = vmlaq_s16(acc.sum, constrain_word(a, x, strength, shift), weight);
        acc.sum = vmlaq_s16(acc.sum, constrain_word(b, x, strength, shift), weight);
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn words<const N: usize, const S: usize, const PRI: bool, const SEC: bool>(
    src: *const i16,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    let directions = offsets::<S>(args.dir);
    let pri = vdupq_n_u16(args.pri as u16);
    let sec = vdupq_n_u16(args.sec as u16);
    let pri_shift = vdupq_n_s16(-(args.pri_shift.min(16) as i16));
    let sec_shift = vdupq_n_s16(-(args.sec_shift.min(16) as i16));
    for row in (0..N).step_by(8 / N) {
        // SAFETY: the checked compact word window contains the group and
        // every neighbor; the entry checked the destination footprint.
        unsafe {
            let p = src.add(row * S);
            let x = load_words::<N, S>(p);
            let mut acc = WordAccum {
                sum: vdupq_n_s16(0),
                min: vreinterpretq_u16_s16(x),
                max: x,
            };
            for k in 0..2 {
                if PRI {
                    word_taps::<N, S, SEC>(
                        p,
                        isize::from(directions[0][k]),
                        x,
                        pri,
                        pri_shift,
                        args.pri_taps[k],
                        &mut acc,
                    );
                }
                if SEC {
                    word_taps::<N, S, PRI>(
                        p,
                        isize::from(directions[1][k]),
                        x,
                        sec,
                        sec_shift,
                        SEC_TAPS[k],
                        &mut acc,
                    );
                    word_taps::<N, S, PRI>(
                        p,
                        isize::from(directions[2][k]),
                        x,
                        sec,
                        sec_shift,
                        SEC_TAPS[k],
                        &mut acc,
                    );
                }
            }
            let correction = vreinterpretq_s16_u16(vcltzq_s16(acc.sum));
            let mut result = vaddq_s16(x, vrshrq_n_s16::<4>(vaddq_s16(acc.sum, correction)));
            if PRI && SEC {
                result = vmaxq_s16(vreinterpretq_s16_u16(acc.min), vminq_s16(acc.max, result));
            }
            // With both strengths the range clamp bounds the result. With
            // only one, its positive weights sum to 12<16, so it cannot
            // overshoot the sample range. XTN is therefore exact, as dav1d.
            let result = vreinterpret_u8_s8(vmovn_s16(result));
            let p = dst.as_mut_ptr().add(row * dst_stride);
            if N == 8 {
                vst1_u8(p, result);
            } else {
                let result = vreinterpret_u32_u8(result);
                vst1_lane_u32::<0>(p.cast(), result);
                vst1_lane_u32::<1>(p.add(dst_stride).cast(), result);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_words<const N: usize, const S: usize>(
    src: *const i16,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: caller supplies the checked N+4 payload in S-stride rows.
    unsafe {
        match (args.pri != 0, args.sec != 0) {
            (true, true) => words::<N, S, true, true>(src, args, dst, dst_stride),
            (true, false) => words::<N, S, true, false>(src, args, dst, dst_stride),
            _ => words::<N, S, false, true>(src, args, dst, dst_stride),
        }
    }
}

// Keep strength classes separate so primary-only blocks do not pay the
// callee-save frame needed by the twelve-neighbor both-strength filter.
#[inline(never)]
#[target_feature(enable = "neon")]
unsafe fn byte_mode<const N: usize, const PRI: bool, const SEC: bool>(
    p: *const u8,
    stride: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: entry enabled NEON and checked all source/output footprints.
    unsafe {
        macro_rules! direction {
            ($dir:literal) => {
                bytes::<N, $dir, PRI, SEC>(p, stride, args, dst, dst_stride)
            };
        }
        match args.dir & 7 {
            0 => direction!(0),
            1 => direction!(1),
            2 => direction!(2),
            3 => direction!(3),
            4 => direction!(4),
            5 => direction!(5),
            6 => direction!(6),
            _ => direction!(7),
        }
    }
}

/// Interior byte path. `CdefSource` already preserves every unfiltered
/// neighbor, so borrow its padded rows instead of recopying the same halo.
/// The lane layout and arithmetic remain dav1d's byte path.
#[target_feature(enable = "neon")]
pub(super) unsafe fn interior<const N: usize>(
    src: &[u8],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    check_output::<N>(dst, dst_stride);
    if args.pri == 0 && args.sec == 0 {
        neon_padding::window_start::<u8, N>(src, stride, origin);
        // SAFETY: both complete footprints were checked. Copy the same
        // packed row groups as filtering, without repeated slice checks.
        unsafe {
            for row in (0..N).step_by(16 / N) {
                let x = load_bytes::<N>(src.as_ptr().add(origin + row * stride), stride);
                store_bytes::<N>(dst.as_mut_ptr().add(row * dst_stride), dst_stride, x);
            }
        }
        return;
    }
    neon_padding::window_start::<u8, N>(src, stride, origin);
    // SAFETY: the checked source contains the block plus all two-sample
    // neighbors. Borrowing it preserves unfiltered input; NEON is enabled.
    unsafe {
        let p = src.as_ptr().add(origin);
        match (args.pri != 0, args.sec != 0) {
            (true, true) => byte_mode::<N, true, true>(p, stride, args, dst, dst_stride),
            (true, false) => byte_mode::<N, true, false>(p, stride, args, dst, dst_stride),
            _ => byte_mode::<N, false, true>(p, stride, args, dst, dst_stride),
        }
    }
}

/// Frame-edge path, preserving dav1d's signed 0x8000 unavailable sentinel.
#[target_feature(enable = "neon")]
pub(super) unsafe fn edge<const N: usize, const S: usize, const A: usize>(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    check_output::<N>(dst, dst_stride);
    if args.pri == 0 && args.sec == 0 {
        neon_padding::window_start::<i16, N>(src, stride, origin);
        for row in 0..N {
            let p = &src[origin + row * stride..origin + row * stride + N];
            for (&sample, output) in p
                .iter()
                .zip(&mut dst[row * dst_stride..row * dst_stride + N])
            {
                *output = sample as u8;
            }
        }
        return;
    }
    if stride == N + 2 * CDEF_PADDING {
        neon_padding::window_start::<i16, N>(src, stride, origin);
        // SAFETY: CdefSource already built every sentinel payload. Borrow
        // its 12/8-stride window rather than copying it a second time.
        unsafe {
            let p = src.as_ptr().add(origin);
            if N == 8 {
                filter_words::<N, 12>(p, args, dst, dst_stride);
            } else {
                filter_words::<N, 8>(p, args, dst, dst_stride);
            }
        }
        return;
    }
    let padded = neon_padding::compact::<i16, N, S, A>(src, stride, origin);
    // SAFETY: all word payloads are initialized, including unavailable
    // neighbors; filter loads stay within them and NEON is enabled.
    unsafe {
        let p = padded.as_ptr().add(CDEF_PADDING * S + CDEF_PADDING);
        match (args.pri != 0, args.sec != 0) {
            (true, true) => words::<N, S, true, true>(p, args, dst, dst_stride),
            (true, false) => words::<N, S, true, false>(p, args, dst, dst_stride),
            _ => words::<N, S, false, true>(p, args, dst, dst_stride),
        }
    }
}
