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

use super::{CDEF_PADDING, CdefArgs, CdefFns, DIRECTIONS, HighCdefFns, SEC_TAPS, generic};
use crate::simd::I16Lanes;
use crate::simd::avx512_i16::I16x32;
use crate::simd::gfni::shift_matrix;
use core::arch::x86_64::{
    __m128i, __m512i, _MM_CMPINT_NLE, _mm512_add_epi8, _mm512_add_epi32, _mm512_and_si512,
    _mm512_cmp_epu8_mask, _mm512_cvtepi32_epi8, _mm512_dpbusd_epi32, _mm512_gf2p8affine_epi64_epi8,
    _mm512_loadu_si512, _mm512_mask_sub_epi8, _mm512_max_epi32, _mm512_max_epu8, _mm512_min_epi32,
    _mm512_min_epu8, _mm512_permutexvar_epi8, _mm512_set_epi64, _mm512_set1_epi8,
    _mm512_set1_epi32, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_srai_epi32,
    _mm512_srli_epi32, _mm512_sub_epi8, _mm512_subs_epu8,
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
    fns.block8_interior = block::<8>;
    fns.block4_interior = block::<4>;
    fns.block4_interior_pair = Some(block4_pair);
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

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,avx512vnni")]
#[inline]
unsafe fn accumulate(
    table: __m512i,
    centre: __m512i,
    taps: &Taps,
    sum: __m512i,
) -> (__m512i, __m512i) {
    let pixels = _mm512_permutexvar_epi8(taps.indices, table);
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

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,avx512vnni")]
#[inline]
unsafe fn quarter(table: __m512i, centres: __m512i, taps: &[Taps; 3], args: &CdefArgs) -> __m128i {
    let centre = _mm512_permutexvar_epi8(centres, table);
    let mut low = centre;
    let mut high = centre;
    let mut sum = _mm512_setzero_si512();
    let clip = args.pri != 0 && args.sec != 0;
    // SAFETY: this function enables every intrinsic feature used by accumulate.
    unsafe {
        for (i, tap) in taps.iter().enumerate() {
            if (i == 0 && args.pri != 0) || (i != 0 && args.sec != 0) {
                let (next, pixels) = accumulate(table, centre, tap, sum);
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

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,avx512vnni,avx2")]
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
        let taps = [
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
        ];
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
                let result = quarter(table, centres, &taps, args);
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
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,avx512vnni,avx2")]
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
