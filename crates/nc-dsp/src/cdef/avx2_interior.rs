//! Packed-byte interior CDEF using paired widening multiply-adds (AV1 §7.15.2).
//! The instruction strategy follows dav1d/SVT-AV1's BSD-licensed byte CDEF;
//! the equations and oracle remain nanocodec's scalar CDEF. Opposite signed
//! constraints are combined in bytes, then `pmaddubsw` applies the tap weights.
//! Each contribution is at most 15*4, and the total magnitude is at most
//! 12*15 + 12*4 = 228, so all signed-word accumulators are exact.
//!
//! A vector holds four rows of an 8x8 block, or four rows of each of two
//! 4x4 blocks ([`block4_pair`]): every lane's arithmetic depends only on its
//! own samples and the shared strengths, so two chroma blocks with different
//! directions fill the vector that one would leave half idle.

use super::{CDEF_PADDING, CdefArgs, DIRECTIONS, SEC_TAPS};
use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_cvtsi128_si64, _mm_extract_epi32,
    _mm_extract_epi64, _mm_set_epi32, _mm256_add_epi8, _mm256_add_epi16, _mm256_and_si256,
    _mm256_broadcastsi128_si256, _mm256_castsi256_si128, _mm256_extracti128_si256,
    _mm256_maddubs_epi16, _mm256_max_epu8, _mm256_min_epu8, _mm256_or_si256, _mm256_packus_epi16,
    _mm256_set_epi64x, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_setr_epi32,
    _mm256_setzero_si256, _mm256_srai_epi16, _mm256_srl_epi16, _mm256_sub_epi8, _mm256_subs_epu8,
    _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
};

const VECTOR_BYTES: usize = 32;
const ROUND: i16 = 8;
const ROUND_SHIFT: i32 = 4;

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load<const N: usize>(base: *const u8, stride: usize) -> __m256i {
    // SAFETY: the caller validated N bytes in each of these four rows. Four
    // wide rows fill the vector; four narrow rows fill its low half, duplicated
    // in the high half so the same paired arithmetic serves chroma.
    unsafe {
        if N == 8 {
            _mm256_set_epi64x(
                base.add(3 * stride).cast::<i64>().read_unaligned(),
                base.add(2 * stride).cast::<i64>().read_unaligned(),
                base.add(stride).cast::<i64>().read_unaligned(),
                base.cast::<i64>().read_unaligned(),
            )
        } else {
            _mm256_broadcastsi128_si256(_mm_set_epi32(
                base.add(3 * stride).cast::<i32>().read_unaligned(),
                base.add(2 * stride).cast::<i32>().read_unaligned(),
                base.add(stride).cast::<i32>().read_unaligned(),
                base.cast::<i32>().read_unaligned(),
            ))
        }
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store<const N: usize>(value: __m256i, dst: *mut u8, stride: usize) {
    let lo = _mm256_castsi256_si128(value);
    let hi = _mm256_extracti128_si256::<1>(value);
    // SAFETY: the caller checked four complete destination rows. Exact-size
    // unaligned stores never overwrite row padding or the following row.
    unsafe {
        if N == 8 {
            dst.cast::<i64>().write_unaligned(_mm_cvtsi128_si64(lo));
            dst.add(stride)
                .cast::<i64>()
                .write_unaligned(_mm_extract_epi64::<1>(lo));
            dst.add(2 * stride)
                .cast::<i64>()
                .write_unaligned(_mm_cvtsi128_si64(hi));
            dst.add(3 * stride)
                .cast::<i64>()
                .write_unaligned(_mm_extract_epi64::<1>(hi));
        } else {
            store_4x4(lo, dst, stride);
        }
    }
}

/// Store the four 4-sample rows of `rows` (row 0 in the low lane).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_4x4(rows: __m128i, dst: *mut u8, stride: usize) {
    // SAFETY: the caller checked four complete 4-sample destination rows.
    unsafe {
        dst.cast::<i32>().write_unaligned(_mm_cvtsi128_si32(rows));
        dst.add(stride)
            .cast::<i32>()
            .write_unaligned(_mm_extract_epi32::<1>(rows));
        dst.add(2 * stride)
            .cast::<i32>()
            .write_unaligned(_mm_extract_epi32::<2>(rows));
        dst.add(3 * stride)
            .cast::<i32>()
            .write_unaligned(_mm_extract_epi32::<3>(rows));
    }
}

struct State {
    centre: __m256i,
    low: __m256i,
    high: __m256i,
    sum: [__m256i; 2],
}

#[target_feature(enable = "avx2")]
#[inline]
fn constrain(pixel: __m256i, centre: __m256i, strength: i16, shift: u32) -> __m256i {
    let positive = _mm256_subs_epu8(pixel, centre);
    let negative = _mm256_subs_epu8(centre, pixel);
    let magnitude = _mm256_or_si256(positive, negative);
    // Word shifts spill high-byte bits into the low byte. The byte mask
    // removes them, giving exact logical byte shifts, including shifts >= 8.
    let shift = shift.min(8);
    let shifted = _mm256_and_si256(
        _mm256_srl_epi16(magnitude, _mm_cvtsi32_si128(shift as i32)),
        _mm256_set1_epi8((u16::from(u8::MAX) >> shift) as i8),
    );
    let bound = _mm256_subs_epu8(_mm256_set1_epi8(strength as i8), shifted);
    // At most one difference is nonzero, so this subtraction applies the
    // sign directly without a compare or signed-weight vector per tap.
    _mm256_sub_epi8(
        _mm256_min_epu8(positive, bound),
        _mm256_min_epu8(negative, bound),
    )
}

/// Combine the two taps `a` and `b` of one offset pair: their constrained
/// differences from the centre, summed, and (with `clip`) the running
/// minimum and maximum.
#[target_feature(enable = "avx2")]
#[inline]
fn pair(
    state: &mut State,
    (a, b): (__m256i, __m256i),
    strength: i16,
    shift: u32,
    clip: bool,
) -> __m256i {
    if clip {
        state.low = _mm256_min_epu8(state.low, _mm256_min_epu8(a, b));
        state.high = _mm256_max_epu8(state.high, _mm256_max_epu8(a, b));
    }
    _mm256_add_epi8(
        constrain(a, state.centre, strength, shift),
        constrain(b, state.centre, strength, shift),
    )
}

/// The samples one vector of output rows reads: its centre rows and, for
/// each tap, the rows at the tap's offset and at its negation.
trait Window {
    /// The output rows' own samples.
    unsafe fn centre(&self) -> __m256i;
    /// The rows at offset `k` (0 or 1) along the direction `turn` steps
    /// (0, 2 or 6) past the block's own, and at its negation.
    unsafe fn taps(&self, turn: usize, k: usize) -> (__m256i, __m256i);
}

/// Four rows of one block whose direction is the constant `DIR`.
struct BlockRows<const N: usize, const DIR: usize> {
    base: *const u8,
    stride: usize,
}

impl<const N: usize, const DIR: usize> Window for BlockRows<N, DIR> {
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn centre(&self) -> __m256i {
        // SAFETY: the caller checked the block's window (`filter`).
        unsafe { load::<N>(self.base, self.stride) }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn taps(&self, turn: usize, k: usize) -> (__m256i, __m256i) {
        let (dy, dx) = DIRECTIONS[(DIR + turn) & 7][k];
        let delta = dy * self.stride as isize + dx;
        // SAFETY: both taps lie in the checked window.
        unsafe {
            (
                load::<N>(self.base.offset(delta), self.stride),
                load::<N>(self.base.offset(-delta), self.stride),
            )
        }
    }
}

/// The four rows of each of two 4x4 blocks, the first in the low lanes,
/// each with its own direction.
struct PairRows {
    bases: [*const u8; 2],
    dirs: [usize; 2],
    stride: usize,
}

impl PairRows {
    /// Rows 0..4 of each block, `deltas` samples from its top-left sample.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load(&self, deltas: [isize; 2]) -> __m256i {
        let row = |block: usize, r: usize| -> i32 {
            // SAFETY: the caller checked both blocks' windows (`block4_pair`).
            unsafe {
                self.bases[block]
                    .offset(deltas[block] + (r * self.stride) as isize)
                    .cast::<i32>()
                    .read_unaligned()
            }
        };
        _mm256_setr_epi32(
            row(0, 0),
            row(0, 1),
            row(0, 2),
            row(0, 3),
            row(1, 0),
            row(1, 1),
            row(1, 2),
            row(1, 3),
        )
    }
}

impl Window for PairRows {
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn centre(&self) -> __m256i {
        // SAFETY: as `load`.
        unsafe { self.load([0, 0]) }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn taps(&self, turn: usize, k: usize) -> (__m256i, __m256i) {
        let deltas = self.dirs.map(|dir| {
            let (dy, dx) = DIRECTIONS[(dir + turn) & 7][k];
            dy * self.stride as isize + dx
        });
        // SAFETY: as `load`; both taps lie in each block's checked window.
        unsafe { (self.load(deltas), self.load(deltas.map(|d| -d))) }
    }
}

/// Primary taps lie along the block's direction; secondary taps two
/// directions either side of it (AV1 §7.15.2).
const PRIMARY_TURN: usize = 0;
const SECONDARY_TURNS: [usize; 2] = [2, 6];

/// Filter the rows of one vector: the CDEF output for every lane.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn filter_rows<W: Window>(window: &W, args: &CdefArgs) -> __m256i {
    let clip = args.pri != 0 && args.sec != 0;
    // SAFETY: the window's loads lie in the caller's checked windows.
    unsafe {
        let centre = window.centre();
        let mut state = State {
            centre,
            low: centre,
            high: centre,
            sum: [_mm256_setzero_si256(); 2],
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
            let mut values = [_mm256_setzero_si256(); 2];
            for (k, value) in values.iter_mut().enumerate() {
                // Keep both tap graphs inside this AVX2-enabled function.
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
                *value = _mm256_add_epi8(first, second);
            }
            accumulate(&mut state, values, SEC_TAPS);
        }
        let zero = _mm256_setzero_si256();
        let centres = [
            _mm256_unpacklo_epi8(centre, zero),
            _mm256_unpackhi_epi8(centre, zero),
        ];
        for (sum, centre) in state.sum.iter_mut().zip(centres) {
            let rounded = _mm256_add_epi16(
                _mm256_add_epi16(*sum, _mm256_srai_epi16::<15>(*sum)),
                _mm256_set1_epi16(ROUND),
            );
            *sum = _mm256_add_epi16(centre, _mm256_srai_epi16::<ROUND_SHIFT>(rounded));
        }
        let result = _mm256_packus_epi16(state.sum[0], state.sum[1]);
        if clip {
            _mm256_max_epu8(state.low, _mm256_min_epu8(state.high, result))
        } else {
            result
        }
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn accumulate(state: &mut State, values: [__m256i; 2], weights: [i16; 2]) {
    // Unsigned byte weights multiply signed byte pairs. Primary pairs are
    // bounded by 30; the four secondary differences by 16. Pairing the two
    // tap distances halves the widening multiply-adds versus widening each.
    let taps = _mm256_set1_epi16(weights[0] | (weights[1] << 8));
    state.sum[0] = _mm256_add_epi16(
        state.sum[0],
        _mm256_maddubs_epi16(taps, _mm256_unpacklo_epi8(values[0], values[1])),
    );
    state.sum[1] = _mm256_add_epi16(
        state.sum[1],
        _mm256_maddubs_epi16(taps, _mm256_unpackhi_epi8(values[0], values[1])),
    );
}

#[target_feature(enable = "avx2")]
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
    let rows = if N == 8 { VECTOR_BYTES / N } else { N };
    for row in (0..N).step_by(rows) {
        // SAFETY: the checked window includes both padding rows/columns, and
        // the destination includes every row stored by this four-row group.
        unsafe {
            let base = window
                .as_ptr()
                .add((row + CDEF_PADDING) * stride + CDEF_PADDING);
            let rows = BlockRows::<N, DIR> { base, stride };
            let result = filter_rows(&rows, args);
            store::<N>(result, dst.as_mut_ptr().add(row * dst_stride), dst_stride);
        }
    }
}

/// The top-left sample of the 4x4 block at `origin` of `src`, after checking
/// that its window (the block plus [`CDEF_PADDING`] on every side) lies in
/// `src`.
fn checked_block4(src: &[u8], stride: usize, origin: usize) -> *const u8 {
    const SIDE: usize = 4 + 2 * CDEF_PADDING;
    let top = origin
        .checked_sub(CDEF_PADDING * stride + CDEF_PADDING)
        .expect("interior source");
    let window = &src[top..][..(SIDE - 1) * stride + SIDE];
    window[CDEF_PADDING * stride + CDEF_PADDING..].as_ptr()
}

/// Two interior 4x4 blocks with shared strengths and damping (`args`; its
/// direction is unused) and their own directions `dirs`: block `i` reads
/// `src[i]` at `origins[i]` and writes the block at `dst[i]`, exactly as
/// [`block::<4>`] would ([`super::CdefPairFn8`] contract).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn block4_pair(
    src: [&[u8]; 2],
    stride: usize,
    origins: [usize; 2],
    args: &CdefArgs,
    dirs: [usize; 2],
    dst: [*mut u8; 2],
    dst_stride: usize,
) {
    let rows = PairRows {
        bases: [
            checked_block4(src[0], stride, origins[0]),
            checked_block4(src[1], stride, origins[1]),
        ],
        dirs: dirs.map(|dir| dir & 7),
        stride,
    };
    // SAFETY: both source windows were checked above; the caller guarantees
    // both destination blocks (`CdefPairFn8`).
    unsafe {
        let result = filter_rows(&rows, args);
        store_4x4(_mm256_castsi256_si128(result), dst[0], dst_stride);
        store_4x4(_mm256_extracti128_si256::<1>(result), dst[1], dst_stride);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn block<const N: usize>(
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
