//! Ice Lake 4×4 high-depth CDEF, following dav1d's `cdef16_avx512.asm`.
//! A dword holds the two tap distances of one output. Two register
//! tables cover the complete source halo.

use super::{CDEF_PADDING, CDEF_UNAVAILABLE, CdefArgs, CdefSource, DIRECTIONS};
use core::arch::x86_64::{
    __m128i, __m512i, _mm_cvtsi32_si128, _mm_extract_epi64, _mm_loadu_si128, _mm_mask_loadu_epi16,
    _mm_set1_epi16, _mm_storel_epi64, _mm256_castsi128_si256, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_inserti128_si256, _mm512_abs_epi16, _mm512_add_epi16,
    _mm512_add_epi32, _mm512_castsi256_si512, _mm512_cmplt_epi16_mask, _mm512_cvtepi32_epi16,
    _mm512_dpwssd_epi32, _mm512_inserti64x4, _mm512_loadu_si512, _mm512_mask_sub_epi16,
    _mm512_max_epi16, _mm512_min_epi16, _mm512_min_epu16, _mm512_permutex2var_epi16,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_srai_epi32,
    _mm512_srl_epi16, _mm512_srli_epi32, _mm512_sub_epi16, _mm512_subs_epu16,
};

const SIDE: usize = 4;
const HALO: usize = SIDE + 2 * CDEF_PADDING;

const fn centers() -> [i16; 32] {
    let mut out = [0; 32];
    let mut i = 0;
    while i < 16 {
        let center = (i / SIDE + CDEF_PADDING) * HALO + i % SIDE + CDEF_PADDING;
        out[2 * i] = center as i16;
        out[2 * i + 1] = center as i16;
        i += 1;
    }
    out
}

const fn offsets() -> [i32; 8] {
    let mut out = [0; 8];
    let mut i = 0;
    while i < 8 {
        let [(r0, c0), (r1, c1)] = DIRECTIONS[i];
        out[i] = pack(
            (r0 * HALO as isize + c0) as i16,
            (r1 * HALO as isize + c1) as i16,
        );
        i += 1;
    }
    out
}

const fn pack(a: i16, b: i16) -> i32 {
    i32::from_le_bytes([
        a.to_le_bytes()[0],
        a.to_le_bytes()[1],
        b.to_le_bytes()[0],
        b.to_le_bytes()[1],
    ])
}

const CENTERS: [i16; 32] = centers();
const OFFSETS: [i32; 8] = offsets();

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
fn pack_rows([a, b, c, d]: [__m128i; 4]) -> __m512i {
    let low = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(a), b);
    let high = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(c), d);
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(low), high)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
fn constrain(p: __m512i, center: __m512i, strength: __m512i, shift: __m128i) -> __m512i {
    let mag = _mm512_abs_epi16(_mm512_sub_epi16(p, center));
    let limit = _mm512_subs_epu16(strength, _mm512_srl_epi16(mag, shift));
    let clipped = _mm512_min_epu16(mag, limit);
    let negative = _mm512_cmplt_epi16_mask(p, center);
    _mm512_mask_sub_epi16(clipped, negative, _mm512_setzero_si512(), clipped)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn filter<const PRI: bool, const SEC: bool>(
    [a, b]: [__m512i; 2],
    args: &CdefArgs,
    dst: &mut [u16],
    stride: usize,
) {
    let dir = args.dir & 7;
    let dst = &mut dst[..3 * stride + SIDE];
    // SAFETY: CENTERS is 64 readable bytes. Each destination store
    // covers one checked four-word row.
    unsafe {
        let indices = _mm512_loadu_si512(CENTERS.as_ptr().cast());
        let center = _mm512_permutex2var_epi16(a, indices, b);
        let (mut min, mut max) = (center, center);
        let mut sum = _mm512_setzero_si512();
        let mut accumulate = |direction: usize, strength: i16, shift: u32, weights: [i16; 2]| {
            let offset = _mm512_set1_epi32(OFFSETS[direction]);
            let strength = _mm512_set1_epi16(strength);
            let weights = _mm512_set1_epi32(pack(weights[0], weights[1]));
            let shift = _mm_cvtsi32_si128(shift as i32);
            let near = _mm512_permutex2var_epi16(a, _mm512_add_epi16(indices, offset), b);
            let far = _mm512_permutex2var_epi16(a, _mm512_sub_epi16(indices, offset), b);
            if PRI && SEC {
                min = _mm512_min_epu16(min, _mm512_min_epu16(near, far));
                max = _mm512_max_epi16(max, _mm512_max_epi16(near, far));
            }
            let pairs = _mm512_add_epi16(
                constrain(near, center, strength, shift),
                constrain(far, center, strength, shift),
            );
            sum = _mm512_dpwssd_epi32(sum, pairs, weights);
        };
        if PRI {
            accumulate(dir, args.pri, args.pri_shift, args.pri_taps);
        }
        if SEC {
            accumulate((dir + 2) & 7, args.sec, args.sec_shift, [2, 1]);
            accumulate((dir + 6) & 7, args.sec, args.sec_shift, [2, 1]);
        }
        let rounded = _mm512_srai_epi32::<4>(_mm512_add_epi32(
            _mm512_add_epi32(sum, _mm512_set1_epi32(8)),
            _mm512_srai_epi32::<31>(sum),
        ));
        // Only the low word of each dword is stored. The adjacent copy
        // participates in the signed word clamp without affecting it.
        let mut output = _mm512_add_epi16(center, rounded);
        if PRI && SEC {
            let min = _mm512_min_epu16(min, _mm512_srli_epi32::<16>(min));
            let max = _mm512_max_epi16(max, _mm512_srli_epi32::<16>(max));
            output = _mm512_max_epi16(output, min);
            output = _mm512_min_epi16(output, max);
        }
        let rows = _mm512_cvtepi32_epi16(output);
        let (lo, hi) = (
            _mm256_castsi256_si128(rows),
            _mm256_extracti128_si256::<1>(rows),
        );
        let dst = dst.as_mut_ptr();
        _mm_storel_epi64(dst.cast(), lo);
        dst.add(stride)
            .cast::<i64>()
            .write_unaligned(_mm_extract_epi64::<1>(lo));
        _mm_storel_epi64(dst.add(2 * stride).cast(), hi);
        dst.add(3 * stride)
            .cast::<i64>()
            .write_unaligned(_mm_extract_epi64::<1>(hi));
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn apply(table: [__m512i; 2], args: &CdefArgs, dst: &mut [u16], stride: usize) {
    // SAFETY: the selected mode passes the same table and destination.
    unsafe {
        match (args.pri != 0, args.sec != 0) {
            (true, true) => filter::<true, true>(table, args, dst, stride),
            (true, false) => filter::<true, false>(table, args, dst, stride),
            (false, true) => filter::<false, true>(table, args, dst, stride),
            (false, false) => filter::<false, false>(table, args, dst, stride),
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub(super) unsafe fn border(
    src: &CdefSource<u16>,
    (x, y): (usize, usize),
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
) {
    let left = CDEF_PADDING.saturating_sub(x);
    let top = CDEF_PADDING.saturating_sub(y);
    let first_x = x.saturating_sub(CDEF_PADDING);
    let first_y = y.saturating_sub(CDEF_PADDING);
    let width = src.width();
    let columns = (HALO - left).min(width - first_x);
    let rows = (HALO - top).min(src.height() - first_y);
    let start = src.index(first_x, first_y);
    let input = &src.samples()[start..][..(rows - 1) * width + columns];
    let valid = (((1u16 << columns) - 1) << left) as u8;
    let end = top + rows;
    // SAFETY: every unmasked source lane lies in the checked source span.
    unsafe {
        let missing = _mm_set1_epi16(CDEF_UNAVAILABLE);
        let load = |i: usize| {
            // At the left edge the base lies before the row. The mask
            // suppresses exactly those columns, so only the valid part
            // of the row is accessed even at the start of the allocation.
            let ptr = input.as_ptr().add(i * width).wrapping_sub(left);
            _mm_mask_loadu_epi16(missing, valid, ptr.cast())
        };
        let [a, b] = match (top, rows) {
            (2, 6) => [
                pack_rows([missing, missing, load(0), load(1)]),
                pack_rows([load(2), load(3), load(4), load(5)]),
            ],
            (0, 6) => [
                pack_rows([load(0), load(1), load(2), load(3)]),
                pack_rows([load(4), load(5), missing, missing]),
            ],
            (0, 8) => [
                pack_rows([load(0), load(1), load(2), load(3)]),
                pack_rows([load(4), load(5), load(6), load(7)]),
            ],
            _ => {
                let row = |i: usize| {
                    if (top..end).contains(&i) {
                        load(i - top)
                    } else {
                        missing
                    }
                };
                [
                    pack_rows([row(0), row(1), row(2), row(3)]),
                    pack_rows([row(4), row(5), row(6), row(7)]),
                ]
            }
        };
        apply([a, b], args, dst, dst_stride);
    }
}

/// The same distance-pair filter on a complete eight-by-eight word halo.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn interior_mixed(
    source: &[u16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
) {
    let start = origin
        .checked_sub(CDEF_PADDING * (stride + 1))
        .expect("interior chroma halo");
    let input = &source[start..][..(HALO - 1) * stride + HALO];
    // SAFETY: eight samples fit in each row of the checked input.
    unsafe {
        let row = |i: usize| _mm_loadu_si128(input.as_ptr().add(i * stride).cast());
        filter::<true, true>(
            [
                pack_rows([row(0), row(1), row(2), row(3)]),
                pack_rows([row(4), row(5), row(6), row(7)]),
            ],
            args,
            dst,
            dst_stride,
        );
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub(super) unsafe fn interior_pair_mixed(
    src: [&[u16]; 2],
    stride: usize,
    origins: [usize; 2],
    args: &CdefArgs,
    dst: [*mut u16; 2],
    dst_stride: usize,
) {
    for i in 0..2 {
        // SAFETY: each destination is an independently checked four-row block.
        unsafe {
            let out = core::slice::from_raw_parts_mut(dst[i], 3 * dst_stride + SIDE);
            interior_mixed(src[i], stride, origins[i], args, out, dst_stride);
        }
    }
}
