//! Whole-block AVX-512 byte CDEF for an interior 8x8 block (AV1 §7.15.2).
//! Extends the existing dav1d/SVT-AV1-inspired AVX2 packed-byte schedule:
//! eight independent rows fill one register instead of two four-row passes.
//! Loads and stores still touch exactly eight bytes per row. Direction search,
//! boundary handling, chroma and the scalar oracle keep their existing paths.
//! Primary pair differences fit ±30, secondary four-tap differences fit ±16,
//! and weighted sums fit ±228; byte pairing and signed-word MACs are exact.

use super::{CDEF_PADDING, CdefArgs, CdefFns, DIRECTIONS, SEC_TAPS};
use core::arch::x86_64::{
    __m512i, _mm_cvtsi32_si128, _mm_cvtsi128_si64, _mm_extract_epi64, _mm512_add_epi8,
    _mm512_add_epi16, _mm512_and_si512, _mm512_castsi512_si128, _mm512_extracti32x4_epi32,
    _mm512_maddubs_epi16, _mm512_max_epu8, _mm512_min_epu8, _mm512_or_si512, _mm512_packus_epi16,
    _mm512_set_epi64, _mm512_set1_epi8, _mm512_set1_epi16, _mm512_setzero_si512, _mm512_srai_epi16,
    _mm512_srl_epi16, _mm512_sub_epi8, _mm512_subs_epu8, _mm512_unpackhi_epi8,
    _mm512_unpacklo_epi8,
};

const SIDE: usize = 8;
const ROUND: i16 = 8;
const ROUND_SHIFT: u32 = 4;

/// Replace only the full interior luma block; all other entries stay AVX2.
pub(crate) fn install(fns: &mut CdefFns) {
    fns.block8_interior = block;
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn load(base: *const u8, stride: usize) -> __m512i {
    // SAFETY: the caller checked eight bytes in each of eight rows, including
    // the direction-specific halo. Unaligned scalar loads never cross a row.
    unsafe {
        _mm512_set_epi64(
            base.add(7 * stride).cast::<i64>().read_unaligned(),
            base.add(6 * stride).cast::<i64>().read_unaligned(),
            base.add(5 * stride).cast::<i64>().read_unaligned(),
            base.add(4 * stride).cast::<i64>().read_unaligned(),
            base.add(3 * stride).cast::<i64>().read_unaligned(),
            base.add(2 * stride).cast::<i64>().read_unaligned(),
            base.add(stride).cast::<i64>().read_unaligned(),
            base.cast::<i64>().read_unaligned(),
        )
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn store(value: __m512i, dst: *mut u8, stride: usize) {
    let pairs = [
        _mm512_castsi512_si128(value),
        _mm512_extracti32x4_epi32::<1>(value),
        _mm512_extracti32x4_epi32::<2>(value),
        _mm512_extracti32x4_epi32::<3>(value),
    ];
    for (pair, rows) in pairs.into_iter().enumerate() {
        // SAFETY: the caller checked all eight complete destination rows.
        // Each store touches only the eight output samples, preserving padding.
        unsafe {
            dst.add(2 * pair * stride)
                .cast::<i64>()
                .write_unaligned(_mm_cvtsi128_si64(rows));
            dst.add((2 * pair + 1) * stride)
                .cast::<i64>()
                .write_unaligned(_mm_extract_epi64::<1>(rows));
        }
    }
}

struct State {
    centre: __m512i,
    low: __m512i,
    high: __m512i,
    sum: [__m512i; 2],
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn constrain(pixel: __m512i, centre: __m512i, strength: i16, shift: u32) -> __m512i {
    let positive = _mm512_subs_epu8(pixel, centre);
    let negative = _mm512_subs_epu8(centre, pixel);
    let magnitude = _mm512_or_si512(positive, negative);
    // Word shifts spill high-byte bits into the low byte. The byte mask
    // removes them, giving exact logical byte shifts, including shifts >= 8.
    let shift = shift.min(8);
    let shifted = _mm512_and_si512(
        _mm512_srl_epi16(magnitude, _mm_cvtsi32_si128(shift as i32)),
        _mm512_set1_epi8((u16::from(u8::MAX) >> shift) as i8),
    );
    let bound = _mm512_subs_epu8(_mm512_set1_epi8(strength as i8), shifted);
    // At most one difference is nonzero, so this subtraction applies the
    // sign directly without a compare or signed-weight vector per tap.
    _mm512_sub_epi8(
        _mm512_min_epu8(positive, bound),
        _mm512_min_epu8(negative, bound),
    )
}

/// Combine the two taps `a` and `b` of one offset pair: their constrained
/// differences from the centre, summed, and (with `clip`) the running
/// minimum and maximum.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn pair(
    state: &mut State,
    (a, b): (__m512i, __m512i),
    strength: i16,
    shift: u32,
    clip: bool,
) -> __m512i {
    if clip {
        state.low = _mm512_min_epu8(state.low, _mm512_min_epu8(a, b));
        state.high = _mm512_max_epu8(state.high, _mm512_max_epu8(a, b));
    }
    _mm512_add_epi8(
        constrain(a, state.centre, strength, shift),
        constrain(b, state.centre, strength, shift),
    )
}

/// Eight rows whose direction is fixed for the whole block.
struct BlockRows<const DIR: usize> {
    base: *const u8,
    stride: usize,
}

impl<const DIR: usize> BlockRows<DIR> {
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    #[inline]
    unsafe fn centre(&self) -> __m512i {
        // SAFETY: `filter` checked the entire block window.
        unsafe { load(self.base, self.stride) }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    #[inline]
    unsafe fn taps(&self, turn: usize, k: usize) -> (__m512i, __m512i) {
        let (dy, dx) = DIRECTIONS[(DIR + turn) & 7][k];
        let delta = dy * self.stride as isize + dx;
        // SAFETY: both offsets stay inside the two-sample halo checked by
        // `filter`, including the final row's exact eight-byte load.
        unsafe {
            (
                load(self.base.offset(delta), self.stride),
                load(self.base.offset(-delta), self.stride),
            )
        }
    }
}

/// Primary taps lie along the block's direction; secondary taps two
/// directions either side of it (AV1 §7.15.2).
const PRIMARY_TURN: usize = 0;
const SECONDARY_TURNS: [usize; 2] = [2, 6];

/// Filter the rows of one vector: the CDEF output for every lane.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn filter_rows<const DIR: usize>(window: &BlockRows<DIR>, args: &CdefArgs) -> __m512i {
    let clip = args.pri != 0 && args.sec != 0;
    // SAFETY: the window's loads lie in the caller's checked windows.
    unsafe {
        let centre = window.centre();
        let mut state = State {
            centre,
            low: centre,
            high: centre,
            sum: [_mm512_setzero_si512(); 2],
        };
        if args.pri != 0 {
            let first = pair(
                &mut state,
                window.taps(PRIMARY_TURN, 0),
                args.pri,
                args.pri_shift,
                clip,
            );
            let second = pair(
                &mut state,
                window.taps(PRIMARY_TURN, 1),
                args.pri,
                args.pri_shift,
                clip,
            );
            accumulate(&mut state, [first, second], args.pri_taps);
        }
        if args.sec != 0 {
            let mut values = [_mm512_setzero_si512(); 2];
            for (k, value) in values.iter_mut().enumerate() {
                // Keep both tap graphs inside this AVX-512-enabled function.
                // An array-map closure can be outlined without its caller's
                // target features, introducing calls across the ISA boundary.
                let first = pair(
                    &mut state,
                    window.taps(SECONDARY_TURNS[0], k),
                    args.sec,
                    args.sec_shift,
                    clip,
                );
                let second = pair(
                    &mut state,
                    window.taps(SECONDARY_TURNS[1], k),
                    args.sec,
                    args.sec_shift,
                    clip,
                );
                *value = _mm512_add_epi8(first, second);
            }
            accumulate(&mut state, values, SEC_TAPS);
        }
        let zero = _mm512_setzero_si512();
        let centres = [
            _mm512_unpacklo_epi8(centre, zero),
            _mm512_unpackhi_epi8(centre, zero),
        ];
        for (sum, centre) in state.sum.iter_mut().zip(centres) {
            let rounded = _mm512_add_epi16(
                _mm512_add_epi16(*sum, _mm512_srai_epi16::<15>(*sum)),
                _mm512_set1_epi16(ROUND),
            );
            *sum = _mm512_add_epi16(centre, _mm512_srai_epi16::<ROUND_SHIFT>(rounded));
        }
        let result = _mm512_packus_epi16(state.sum[0], state.sum[1]);
        if clip {
            _mm512_max_epu8(state.low, _mm512_min_epu8(state.high, result))
        } else {
            result
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn accumulate(state: &mut State, values: [__m512i; 2], weights: [i16; 2]) {
    // Unsigned byte weights multiply signed byte pairs. Primary pairs are
    // bounded by 30; the four secondary differences by 16. Pairing the two
    // tap distances halves the widening multiply-adds versus widening each.
    let taps = _mm512_set1_epi16(weights[0] | (weights[1] << 8));
    state.sum[0] = _mm512_add_epi16(
        state.sum[0],
        _mm512_maddubs_epi16(taps, _mm512_unpacklo_epi8(values[0], values[1])),
    );
    state.sum[1] = _mm512_add_epi16(
        state.sum[1],
        _mm512_maddubs_epi16(taps, _mm512_unpackhi_epi8(values[0], values[1])),
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn filter<const DIR: usize>(
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
    let extent = SIDE + 2 * CDEF_PADDING;
    let window = &src[top..][..(extent - 1) * stride + extent];
    let dst = &mut dst[..(SIDE - 1) * dst_stride + SIDE];
    // SAFETY: this window covers the block and all taps up to two rows and
    // columns away. The destination covers exactly eight complete rows.
    unsafe {
        let rows = BlockRows::<DIR> {
            base: window.as_ptr().add(CDEF_PADDING * stride + CDEF_PADDING),
            stride,
        };
        store(filter_rows(&rows, args), dst.as_mut_ptr(), dst_stride);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn block(
    src: &[u8],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: each direction specialization receives the checked interior contract.
    unsafe {
        match args.dir {
            0 => filter::<0>(src, stride, origin, args, dst, dst_stride),
            1 => filter::<1>(src, stride, origin, args, dst, dst_stride),
            2 => filter::<2>(src, stride, origin, args, dst, dst_stride),
            3 => filter::<3>(src, stride, origin, args, dst, dst_stride),
            4 => filter::<4>(src, stride, origin, args, dst, dst_stride),
            5 => filter::<5>(src, stride, origin, args, dst, dst_stride),
            6 => filter::<6>(src, stride, origin, args, dst, dst_stride),
            _ => filter::<7>(src, stride, origin, args, dst, dst_stride),
        }
    }
}
