//! Byte CDEF after dav1d 1.5.4 `src/x86/cdef_avx512.asm`
//! (`cdef_filter_*_8bpc`), Copyright © 2020 VideoLAN/dav1d authors and
//! Two Orioles, LLC, BSD-2-Clause (see `THIRD_PARTY.md` and docs/licenses/dav1d).
//!
//! One 4x4 output quarter reads an 8x8 byte table, including its halo.
//! `vpermb` selects four taps per pixel; GFNI shifts their unsigned absolute
//! differences and VNNI sums constrained magnitudes with signed tap weights.
//! AV1 and AV2 share this graph. The scalar reference, padded/high-depth
//! filters and direction search are independent of this interior schedule.
//!
//! Unlike dav1d's in-place ABI, our unfiltered source includes the halo, so
//! the table needs no left/top merge or unavailable-edge mask. Exact eight-
//! byte row loads also work at a minimally sized plane's final row.
//!
//! The 10/12-bit (`u16`) blocks ([`install_high`]) use the shared 16-bit
//! lane graph (`cdef::generic`) at 32 lanes instead.

use super::{
    CDEF_PADDING, CdefArgs, CdefFns, CdefSource, DIRECTIONS, HighCdefFns, SEC_TAPS, generic,
};
use crate::simd::I16Lanes;
use crate::simd::avx512_i16::I16x32;
use crate::simd::gfni::shift_matrix;
use core::arch::x86_64::{
    __m128i, __m512i, _MM_CMPINT_NLE, _mm_cvtsi128_si64, _mm_maskz_loadu_epi8, _mm_slli_si128,
    _mm_srli_si128, _mm_storel_epi64, _mm_unpackhi_epi32, _mm_unpacklo_epi32, _mm512_add_epi8,
    _mm512_add_epi32, _mm512_and_si512, _mm512_bitshuffle_epi64_mask, _mm512_castsi128_si512,
    _mm512_cmp_epu8_mask, _mm512_cvtepi32_epi8, _mm512_dpbusd_epi32, _mm512_gf2p8affine_epi64_epi8,
    _mm512_inserti32x4, _mm512_loadu_si512, _mm512_mask_mov_epi8, _mm512_mask_sub_epi8,
    _mm512_max_epi32, _mm512_max_epu8, _mm512_min_epi32, _mm512_min_epu8, _mm512_permutexvar_epi8,
    _mm512_set_epi64, _mm512_set1_epi8, _mm512_set1_epi32, _mm512_set1_epi64, _mm512_setzero_si512,
    _mm512_shuffle_i64x2, _mm512_srai_epi32, _mm512_srli_epi32, _mm512_sub_epi8, _mm512_subs_epu8,
};

const QUARTER: usize = 4;
const TABLE_SIDE: usize = QUARTER + 2 * CDEF_PADDING;
const ROUND: i32 = 8;
const ROUND_SHIFT: u32 = 4;
const BYTE_MASK: i32 = 255;
const SHIFTS: [i64; 9] = {
    let mut values = [0; 9];
    let mut shift = 0;
    while shift < values.len() {
        values[shift] = shift_matrix(shift as i32, false);
        shift += 1;
    }
    values
};

/// The full tier uses the same quarter schedule for luma and chroma.
pub(crate) fn install(fns: &mut CdefFns) {
    fns.block8_interior = block8;
    fns.block4_interior = block::<4>;
    fns.block4_interior_pair = Some(block4_pair);
    fns.border = Some([border4, border8]);
}

/// Indices of the 4x4 output pixels inside the 8x8 source table. Each
/// pixel occupies a dword whose four bytes select its four taps.
const CENTRES: [i32; 16] = {
    let mut values = [0; 16];
    let mut i = 0;
    while i < values.len() {
        let index = (i / QUARTER + CDEF_PADDING) * TABLE_SIDE + i % QUARTER + CDEF_PADDING;
        values[i] = (index as i32) * 0x0101_0101;
        i += 1;
    }
    values
};

/// The two distances in one direction, each followed by its negation.
const fn offsets(dir: usize) -> i32 {
    let [first, second] = DIRECTIONS[dir];
    let a = (first.0 * TABLE_SIDE as isize + first.1) as i8;
    let b = (second.0 * TABLE_SIDE as isize + second.1) as i8;
    i32::from_le_bytes([
        a as u8,
        a.wrapping_neg() as u8,
        b as u8,
        b.wrapping_neg() as u8,
    ])
}

/// Build packed byte offsets at compile time. Computing them from isize
/// direction pairs in the hot wrapper makes LLVM generate vector packing
/// and reductions instead of the single broadcast dav1d uses.
const OFFSETS: [i32; 8] = {
    let mut values = [0; 8];
    let mut dir = 0;
    while dir < values.len() {
        values[dir] = offsets(dir);
        dir += 1;
    }
    values
};

struct Taps {
    indices: __m512i,
    weights: __m512i,
    strength: __m512i,
    shift: __m512i,
}

impl Taps {
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
    #[inline]
    unsafe fn new(
        dir: usize,
        weights: [i16; 2],
        strength: i16,
        shift: u32,
        centres: __m512i,
    ) -> Self {
        Self {
            indices: _mm512_add_epi8(centres, _mm512_set1_epi32(OFFSETS[dir])),
            weights: _mm512_set1_epi32(i32::from_le_bytes([
                weights[0] as u8,
                weights[0] as u8,
                weights[1] as u8,
                weights[1] as u8,
            ])),
            strength: _mm512_set1_epi8(strength as i8),
            shift: _mm512_set1_epi64(SHIFTS[shift.min(8) as usize]),
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512bitalg,gfni,avx512vnni")]
#[inline]
unsafe fn accumulate<const BORDER: bool>(
    table: __m512i,
    available: __m512i,
    centre: __m512i,
    taps: &Taps,
    sum: __m512i,
) -> (__m512i, __m512i) {
    let mut pixels = _mm512_permutexvar_epi8(taps.indices, table);
    if BORDER {
        let valid = _mm512_bitshuffle_epi64_mask(available, taps.indices);
        // Unavailable neighbours contribute zero and never widen the clamp.
        pixels = _mm512_mask_mov_epi8(centre, valid, pixels);
    }
    let negative = _mm512_cmp_epu8_mask::<{ _MM_CMPINT_NLE }>(centre, pixels);
    let magnitude = _mm512_mask_sub_epi8(_mm512_sub_epi8(pixels, centre), negative, centre, pixels);
    let shifted = _mm512_gf2p8affine_epi64_epi8::<0>(magnitude, taps.shift);
    let constrained = _mm512_min_epu8(magnitude, _mm512_subs_epu8(taps.strength, shifted));
    let weights =
        _mm512_mask_sub_epi8(taps.weights, negative, _mm512_setzero_si512(), taps.weights);
    // Each primary term is at most 15*4; the whole sum is bounded by
    // 12*15 + 12*4 = 228. VNNI's four-byte dot product cannot overflow.
    (_mm512_dpbusd_epi32(sum, constrained, weights), pixels)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512bitalg,gfni,avx512vnni")]
#[inline]
unsafe fn quarter<const BORDER: bool>(
    table: __m512i,
    available: __m512i,
    centres: __m512i,
    taps: &[Taps; 3],
    args: &CdefArgs,
) -> __m128i {
    // SAFETY: same full Ice Lake feature set as the generic strength path.
    unsafe { quarter_with_strengths::<BORDER, 0>(table, available, centres, taps, args) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512bitalg,gfni,avx512vnni")]
#[inline]
unsafe fn quarter_with_strengths<const BORDER: bool, const MODE: u8>(
    table: __m512i,
    available: __m512i,
    centres: __m512i,
    taps: &[Taps; 3],
    args: &CdefArgs,
) -> __m128i {
    let centre = _mm512_permutexvar_epi8(centres, table);
    let mut low = centre;
    let mut high = centre;
    let mut sum = _mm512_setzero_si512();
    // The 8×8 interior path resolves this once for all four quarters.
    let primary = if MODE == 0 {
        args.pri != 0
    } else {
        MODE & 1 != 0
    };
    let secondary = if MODE == 0 {
        args.sec != 0
    } else {
        MODE & 2 != 0
    };
    let clip = primary && secondary;
    // SAFETY: this function enables every intrinsic feature used by accumulate.
    unsafe {
        for (i, tap) in taps.iter().enumerate() {
            if (i == 0 && primary) || (i != 0 && secondary) {
                let (next, pixels) = accumulate::<BORDER>(table, available, centre, tap, sum);
                sum = next;
                if clip {
                    low = _mm512_min_epu8(low, pixels);
                    high = _mm512_max_epu8(high, pixels);
                }
            }
        }
    }
    let rounded = _mm512_srai_epi32::<ROUND_SHIFT>(_mm512_add_epi32(
        _mm512_add_epi32(sum, _mm512_srai_epi32::<31>(sum)),
        _mm512_set1_epi32(ROUND),
    ));
    let mask = _mm512_set1_epi32(BYTE_MASK);
    let mut result = _mm512_add_epi32(_mm512_and_si512(centre, mask), rounded);
    if clip {
        // Reduce the four tap bytes of each pixel, including its centre,
        // to one unsigned minimum and maximum in the dword's low byte.
        low = _mm512_min_epu8(low, _mm512_srli_epi32::<16>(low));
        low = _mm512_min_epu8(low, _mm512_srli_epi32::<8>(low));
        high = _mm512_max_epu8(high, _mm512_srli_epi32::<16>(high));
        high = _mm512_max_epu8(high, _mm512_srli_epi32::<8>(high));
        result = _mm512_max_epi32(_mm512_and_si512(low, mask), result);
        result = _mm512_min_epi32(_mm512_and_si512(high, mask), result);
    }
    // With one strength active the positive weights total <= 12/16: the
    // constrained update stays in [0,255] without neighbour clipping. With
    // both active the dword clamp above establishes the same bound.
    _mm512_cvtepi32_epi8(result)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
unsafe fn tap_table(args: &CdefArgs, centres: __m512i) -> [Taps; 3] {
    // SAFETY: this function enables all features of the register setup.
    unsafe {
        [
            Taps::new(args.dir, args.pri_taps, args.pri, args.pri_shift, centres),
            Taps::new(
                (args.dir + 2) & 7,
                SEC_TAPS,
                args.sec,
                args.sec_shift,
                centres,
            ),
            Taps::new(
                (args.dir + 6) & 7,
                SEC_TAPS,
                args.sec,
                args.sec_shift,
                centres,
            ),
        ]
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512bitalg,gfni,avx512vnni,avx2")]
pub(crate) unsafe fn block<const N: usize>(
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
    let extent = N + 2 * CDEF_PADDING;
    let window = &src[top..][..(extent - 1) * stride + extent];
    let dst = &mut dst[..(N - 1) * dst_stride + N];
    // SAFETY: CENTRES is a full unaligned 64-byte constant. Every table
    // load below reads exactly eight bytes in each of eight rows, within
    // the checked window; stores cover only each quarter's four output rows.
    unsafe {
        let centres = _mm512_loadu_si512(CENTRES.as_ptr().cast());
        let taps = tap_table(args, centres);
        for row in (0..N).step_by(QUARTER) {
            for col in (0..N).step_by(QUARTER) {
                let base = window.as_ptr().add(row * stride + col);
                let table = _mm512_set_epi64(
                    base.add(7 * stride).cast::<i64>().read_unaligned(),
                    base.add(6 * stride).cast::<i64>().read_unaligned(),
                    base.add(5 * stride).cast::<i64>().read_unaligned(),
                    base.add(4 * stride).cast::<i64>().read_unaligned(),
                    base.add(3 * stride).cast::<i64>().read_unaligned(),
                    base.add(2 * stride).cast::<i64>().read_unaligned(),
                    base.add(stride).cast::<i64>().read_unaligned(),
                    base.cast::<i64>().read_unaligned(),
                );
                let result = quarter::<false>(table, _mm512_setzero_si512(), centres, &taps, args);
                super::avx2_interior::store_4x4(
                    result,
                    dst.as_mut_ptr().add(row * dst_stride + col),
                    dst_stride,
                );
            }
        }
    }
}

/// Lay out four twelve-byte halo rows as the left and right groups of
/// four eight-byte rows. Reuse the middle group for both output halves.
const SPLIT_HALO: [u8; 64] = {
    let mut bytes = [0; 64];
    let mut i = 0;
    while i < 64 {
        bytes[i] = (16 * ((i % 32) / 8) + 4 * (i / 32) + i % 8) as u8;
        i += 1;
    }
    bytes
};

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn halo_rows(ptr: *const u8, stride: usize, shuffle: __m512i) -> __m512i {
    // SAFETY: the caller checked all four reference rows. The masked
    // loads access only the twelve bytes shared by both quarters.
    unsafe {
        let load = |row| _mm_maskz_loadu_epi8(0x0fff, ptr.add(row * stride).cast());
        let a = _mm512_castsi128_si512(load(0));
        let b = _mm512_inserti32x4::<1>(a, load(1));
        let c = _mm512_inserti32x4::<2>(b, load(2));
        let d = _mm512_inserti32x4::<3>(c, load(3));
        _mm512_permutexvar_epi8(shuffle, d)
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512bitalg,gfni,avx512vnni,avx2")]
unsafe fn block8(
    src: &[u8],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    const SIDE: usize = 8 + 2 * CDEF_PADDING;
    let top = origin
        .checked_sub(CDEF_PADDING * stride + CDEF_PADDING)
        .expect("interior source");
    let window = &src[top..][..(SIDE - 1) * stride + SIDE];
    let dst = &mut dst[..7 * dst_stride + 8];
    // SAFETY: the source spans twelve 12-byte rows and the stores are
    // entirely within the validated destination.
    unsafe {
        let ptr = window.as_ptr();
        let shuffle = _mm512_loadu_si512(SPLIT_HALO.as_ptr().cast());
        let top = halo_rows(ptr, stride, shuffle);
        let middle = halo_rows(ptr.add(4 * stride), stride, shuffle);
        let bottom = halo_rows(ptr.add(8 * stride), stride, shuffle);
        let tl = _mm512_shuffle_i64x2::<0x44>(top, middle);
        let tr = _mm512_shuffle_i64x2::<0xee>(top, middle);
        let bl = _mm512_shuffle_i64x2::<0x44>(middle, bottom);
        let br = _mm512_shuffle_i64x2::<0xee>(middle, bottom);
        let tables = [tl, tr, bl, br];
        match (args.pri != 0, args.sec != 0) {
            (true, false) => write_quarters::<1>(tables, args, dst, dst_stride),
            (false, true) => write_quarters::<2>(tables, args, dst, dst_stride),
            (true, true) => write_quarters::<3>(tables, args, dst, dst_stride),
            (false, false) => write_quarters::<0>(tables, args, dst, dst_stride),
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512bitalg,gfni,avx512vnni,avx2")]
#[inline]
unsafe fn write_quarters<const MODE: u8>(
    tables: [__m512i; 4],
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // SAFETY: the four quarter stores cover the checked eight-by-eight block.
    unsafe {
        let centres = _mm512_loadu_si512(CENTRES.as_ptr().cast());
        let taps = tap_table(args, centres);
        for (row, &[left, right]) in tables.as_chunks::<2>().0.iter().enumerate() {
            let zero = _mm512_setzero_si512();
            let left = quarter_with_strengths::<false, MODE>(left, zero, centres, &taps, args);
            let right = quarter_with_strengths::<false, MODE>(right, zero, centres, &taps, args);
            let first = _mm_unpacklo_epi32(left, right);
            let last = _mm_unpackhi_epi32(left, right);
            let dst = dst.as_mut_ptr().add(4 * row * dst_stride);
            _mm_storel_epi64(dst.cast(), first);
            _mm_storel_epi64(dst.add(dst_stride).cast(), _mm_srli_si128::<8>(first));
            _mm_storel_epi64(dst.add(2 * dst_stride).cast(), last);
            _mm_storel_epi64(dst.add(3 * dst_stride).cast(), _mm_srli_si128::<8>(last));
        }
    }
}

/// Read only the available rows and columns of one eight-by-eight table.
/// Table indices retain the full halo grid, including absent samples.
#[allow(
    clippy::inline_always,
    reason = "keep the byte table and availability in registers"
)]
#[inline(always)]
unsafe fn load_border_table(src: &CdefSource, (x, y): (usize, usize)) -> (__m512i, __m512i) {
    let reach = CDEF_PADDING;
    let (left, top) = (reach.saturating_sub(x), reach.saturating_sub(y));
    let (first_x, first_y) = (x.saturating_sub(reach), y.saturating_sub(reach));
    let (width, height) = (src.width(), src.height());
    let columns = (TABLE_SIDE - left).min(width - first_x);
    let rows = (TABLE_SIDE - top).min(height - first_y);
    let column_mask = (1_u16 << columns) - 1;
    let row_mask = u64::from(column_mask) << left;
    let start = src.index(first_x, first_y);
    let data = &src.samples()[start..][..(rows - 1) * width + columns];
    let mut bytes = [0_u64; TABLE_SIDE];
    let mut valid = 0;
    for (row, payload) in bytes.iter_mut().enumerate().skip(top).take(rows) {
        // SAFETY: the span above contains every unmasked byte of these rows.
        let values = unsafe {
            let ptr = data.as_ptr().add((row - top) * width);
            if columns == TABLE_SIDE {
                ptr.cast::<u64>().read_unaligned()
            } else {
                _mm_cvtsi128_si64(_mm_maskz_loadu_epi8(column_mask, ptr.cast())) as u64
            }
        };
        *payload = values << (8 * left);
        valid |= row_mask << (8 * row);
    }
    // SAFETY: bytes is a complete 64-byte table.
    unsafe {
        (
            _mm512_loadu_si512(bytes.as_ptr().cast()),
            _mm512_set1_epi64(valid as i64),
        )
    }
}

/// dav1d's byte table and availability masks at frame borders. Keep the
/// same per-pixel GFNI and VNNI graph as the interior kernel.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512bitalg,gfni,avx512vnni,avx2")]
unsafe fn border4(
    src: &CdefSource,
    (x, y): (usize, usize),
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    let dst = &mut dst[..3 * dst_stride + QUARTER];
    // SAFETY: dispatch checks the block and its available halo.
    unsafe {
        let centres = _mm512_loadu_si512(CENTRES.as_ptr().cast());
        let taps = tap_table(args, centres);
        let (table, available) = load_border_table(src, (x, y));
        let result = quarter::<true>(table, available, centres, &taps, args);
        super::avx2_interior::store_4x4(result, dst.as_mut_ptr(), dst_stride);
    }
}

/// Load one twelve-by-twelve window. The left and right eight columns
/// overlap by four pixels and are shared by the upper and lower quarters.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512bitalg,gfni,avx512vnni,avx2")]
unsafe fn border8(
    src: &CdefSource,
    (x, y): (usize, usize),
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    const WINDOW: usize = 8 + 2 * CDEF_PADDING;
    let reach = CDEF_PADDING;
    let (left, top) = (reach.saturating_sub(x), reach.saturating_sub(y));
    let (first_x, first_y) = (x.saturating_sub(reach), y.saturating_sub(reach));
    let (width, height) = (src.width(), src.height());
    let columns = (WINDOW - left).min(width - first_x);
    let rows = (WINDOW - top).min(height - first_y);
    let end = top + rows;
    let column_mask = (1_u16 << columns) - 1;
    let valid_columns = column_mask << left;
    let start = src.index(first_x, first_y);
    let data = &src.samples()[start..][..(rows - 1) * width + columns];
    let dst = &mut dst[..7 * dst_stride + 8];
    let mut halves = [[0_u64; WINDOW]; 2];
    // SAFETY: each source mask describes bytes in data. The corner tables
    // are eight consecutive u64s and only the four output rows are stored.
    unsafe {
        let [west, east] = &mut halves;
        let rows = west[top..end].iter_mut().zip(&mut east[top..end]);
        for (row, (west, east)) in rows.enumerate() {
            let ptr = data.as_ptr().add(row * width);
            let pixels = _mm_maskz_loadu_epi8(column_mask, ptr.cast());
            let pixels = match left {
                0 => pixels,
                1 => _mm_slli_si128::<1>(pixels),
                _ => _mm_slli_si128::<2>(pixels),
            };
            *west = _mm_cvtsi128_si64(pixels) as u64;
            *east = _mm_cvtsi128_si64(_mm_srli_si128::<4>(pixels)) as u64;
        }
        let centres = _mm512_loadu_si512(CENTRES.as_ptr().cast());
        let taps = tap_table(args, centres);
        for row in [0, QUARTER] {
            let missing_top = top.saturating_sub(row);
            let missing_bottom = (row + TABLE_SIDE).saturating_sub(end);
            let row_mask = (u64::MAX << (8 * missing_top)) & (u64::MAX >> (8 * missing_bottom));
            for (index, half) in halves.iter().enumerate() {
                let col = index * QUARTER;
                let columns = u64::from((valid_columns >> col) as u8);
                let valid = row_mask & (columns * 0x0101_0101_0101_0101);
                let table = _mm512_loadu_si512(half.as_ptr().add(row).cast());
                let available = _mm512_set1_epi64(valid as i64);
                let result = quarter::<true>(table, available, centres, &taps, args);
                super::avx2_interior::store_4x4(
                    result,
                    dst.as_mut_ptr().add(row * dst_stride + col),
                    dst_stride,
                );
            }
        }
    }
}

/// Two chroma blocks with independent directions, as in the other tiers.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512bitalg,gfni,avx512vnni,avx2")]
unsafe fn block4_pair(
    src: [&[u8]; 2],
    stride: usize,
    origins: [usize; 2],
    args: &CdefArgs,
    dirs: [usize; 2],
    dst: [*mut u8; 2],
    dst_stride: usize,
) {
    for i in 0..2 {
        // SAFETY: CdefPairFn8 guarantees four writable rows of four samples
        // at each destination, without concurrent aliases. Each source's
        // complete halo is checked by block; the full tier is enabled here.
        unsafe {
            let out = core::slice::from_raw_parts_mut(dst[i], 3 * dst_stride + QUARTER);
            block::<QUARTER>(
                src[i],
                stride,
                origins[i],
                &args.with_dir(dirs[i] as u8),
                out,
                dst_stride,
            );
        }
    }
}

/// Install AVX-512 high-bit-depth (`u16`) CDEF kernels over the AVX2 table:
/// four-row 8x8 luma passes (`I16x32`, two passes per 8x8 block) and single-pass
/// dual U+V 4x4 interior chroma blocks (adapted from dav1d's
/// `src/x86/cdef16_avx512.asm`, BSD-2-Clause; see `THIRD_PARTY.md`).
pub(crate) fn install_high(fns: &mut HighCdefFns) {
    fns.border4 = Some(super::avx512_high4::border);
    fns.edge[1] = block8_edge_u16;
    fns.interior[1] = block8_interior_u16;
    fns.interior_pair4 = Some(block4_pair_u16);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn block8_edge_u16(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
) {
    // SAFETY: AVX-512F/BW/VL is enabled (dispatch contract).
    unsafe {
        generic::filter_any_dir_u16::<I16x32, 4, 12>(src, stride, origin, args, dst, dst_stride);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn block8_interior_u16(
    src: &[u16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
) {
    // SAFETY: AVX-512F/BW/VL is enabled; valid 10/12-bit `u16` samples
    // (`0..=4095`) fit positive `i16`.
    unsafe {
        let src_i16 = core::slice::from_raw_parts(src.as_ptr().cast::<i16>(), src.len());
        block8_edge_u16(src_i16, stride, origin, args, dst, dst_stride);
    }
}

const CHROMA_SIDE: usize = 4;
const CHROMA_EXTENT: usize = CHROMA_SIDE + 2 * CDEF_PADDING;

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn load_uv_pair4(
    win_u: &[u16],
    win_v: &[u16],
    stride: usize,
    row: usize,
    col: usize,
) -> I16x32 {
    debug_assert!(row + CHROMA_SIDE <= CHROMA_EXTENT && col + CHROMA_SIDE <= CHROMA_EXTENT);
    let rows = |win: &[u16]| -> [&[i16]; 4] {
        // SAFETY: `win` covers all 8 rows of 8 samples (`7 * stride + 8`).
        unsafe {
            let base = win.as_ptr().add(row * stride + col).cast::<i16>();
            [
                core::slice::from_raw_parts(base, CHROMA_SIDE),
                core::slice::from_raw_parts(base.add(stride), CHROMA_SIDE),
                core::slice::from_raw_parts(base.add(2 * stride), CHROMA_SIDE),
                core::slice::from_raw_parts(base.add(3 * stride), CHROMA_SIDE),
            ]
        }
    };
    // SAFETY: AVX-512F/BW/VL enabled and row slices hold 4 words each.
    unsafe { I16x32::load_pair4(rows(win_u), rows(win_v)) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn tap_pair_uv(
    win: (&[u16], &[u16], usize),
    (dr, dc): (isize, isize),
    x: I16x32,
    strength: I16x32,
    shift: u32,
    range: Option<&mut generic::Range<I16x32>>,
) -> I16x32 {
    let at = |r: isize, c: isize| {
        (
            CDEF_PADDING.wrapping_add_signed(r),
            CDEF_PADDING.wrapping_add_signed(c),
        )
    };
    let (r0, c0) = at(dr, dc);
    let (r1, c1) = at(-dr, -dc);
    // SAFETY: `dr, dc` lie in `-2..=2`, inside the validated 8x8 windows.
    let (p0, p1) = unsafe {
        (
            load_uv_pair4(win.0, win.1, win.2, r0, c0),
            load_uv_pair4(win.0, win.1, win.2, r1, c1),
        )
    };
    if let Some(range) = range {
        range.take(p0);
        range.take(p1);
    }
    generic::constrain(p0.sub(x), strength, shift).add(generic::constrain(
        p1.sub(x),
        strength,
        shift,
    ))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn filter_pair4<const USE_PRI: bool, const USE_SEC: bool>(
    src: [&[u16]; 2],
    stride: usize,
    origins: [usize; 2],
    args: &CdefArgs,
    dst: [*mut u16; 2],
    dst_stride: usize,
) {
    let window = |i: usize| {
        let top = origins[i]
            .checked_sub(CDEF_PADDING * stride + CDEF_PADDING)
            .expect("interior chroma source");
        &src[i][top..][..(CHROMA_EXTENT - 1) * stride + CHROMA_EXTENT]
    };
    let win = (window(0), window(1), stride);
    let clip = USE_PRI && USE_SEC;
    let dir = args.dir & 7;
    let pri_dir = DIRECTIONS[dir];
    let sec_dirs = [DIRECTIONS[(dir + 2) & 7], DIRECTIONS[(dir + 6) & 7]];
    // SAFETY: windows cover all taps; `dst[0]` and `dst[1]` point to disjoint
    // 4x4 blocks in U and V planes validated by the band driver.
    unsafe {
        let x = load_uv_pair4(win.0, win.1, stride, CDEF_PADDING, CDEF_PADDING);
        let mut range = generic::Range { lo: x, hi: x };
        let mut sum = I16x32::splat(0);
        if USE_PRI {
            let pri = I16x32::splat(args.pri);
            for (&step, &weight) in pri_dir.iter().zip(&args.pri_taps) {
                let taps = tap_pair_uv(
                    win,
                    step,
                    x,
                    pri,
                    args.pri_shift,
                    clip.then_some(&mut range),
                );
                sum = sum.add(taps.mul(I16x32::splat(weight)));
            }
        }
        if USE_SEC {
            let sec = I16x32::splat(args.sec);
            let mut sec_sum = [I16x32::splat(0); 2];
            for (k, sec_k) in sec_sum.iter_mut().enumerate() {
                for dirs in &sec_dirs {
                    *sec_k = sec_k.add(tap_pair_uv(
                        win,
                        dirs[k],
                        x,
                        sec,
                        args.sec_shift,
                        clip.then_some(&mut range),
                    ));
                }
            }
            sum = sum.add(sec_sum[0].add(sec_sum[0])).add(sec_sum[1]);
        }
        let mut y = x.add(sum.add(I16x32::splat(8)).add(sum.sra(15)).sra(4));
        if clip {
            y = y.max(range.lo).min(range.hi);
        }
        let rows_mut = |base: *mut u16| -> [&mut [i16]; 4] {
            let base = base.cast::<i16>();
            [
                core::slice::from_raw_parts_mut(base, CHROMA_SIDE),
                core::slice::from_raw_parts_mut(base.add(dst_stride), CHROMA_SIDE),
                core::slice::from_raw_parts_mut(base.add(2 * dst_stride), CHROMA_SIDE),
                core::slice::from_raw_parts_mut(base.add(3 * dst_stride), CHROMA_SIDE),
            ]
        };
        y.store_pair4(rows_mut(dst[0]), rows_mut(dst[1]));
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn block4_pair_u16(
    src: [&[u16]; 2],
    stride: usize,
    origins: [usize; 2],
    args: &CdefArgs,
    dst: [*mut u16; 2],
    dst_stride: usize,
) {
    // SAFETY: forwards the caller's AVX-512 and slice contracts.
    unsafe {
        match (args.pri != 0, args.sec != 0) {
            (true, true) => filter_pair4::<true, true>(src, stride, origins, args, dst, dst_stride),
            (true, false) => {
                filter_pair4::<true, false>(src, stride, origins, args, dst, dst_stride);
            }
            (false, true) => {
                filter_pair4::<false, true>(src, stride, origins, args, dst, dst_stride);
            }
            (false, false) => {
                filter_pair4::<false, false>(src, stride, origins, args, dst, dst_stride);
            }
        }
    }
}
