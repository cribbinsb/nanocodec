//! AVX2 pixel kernels (x86-64).
//!
//! SAD packs rows into 256-bit registers and uses `vpsadbw`. Widths 4 and 8
//! gather several rows per register, and width 64 is two width-32 columns.
//! SATD packs independent 8x8 tiles into 128-bit groups and reduces the
//! last Hadamard butterflies by their exact absolute maxima.
//!
//! Kernel contract: see the `pixel` module docs. Arithmetic intrinsics are
//! safe inside `#[target_feature(enable = "avx2")]` functions; only the
//! loads need `unsafe`.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_abs_epi16, _mm_add_epi16, _mm_add_epi32, _mm_add_epi64, _mm_castps_si128,
    _mm_castsi128_ps, _mm_cmpgt_epi16, _mm_cvtepu8_epi16, _mm_cvtsi32_si128, _mm_cvtsi128_si32,
    _mm_cvtsi128_si64, _mm_extract_epi16, _mm_hadd_epi16, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_madd_epi16, _mm_maddubs_epi16, _mm_max_epi16, _mm_minpos_epu16, _mm_movemask_epi8,
    _mm_packs_epi16, _mm_packus_epi32, _mm_sad_epu8, _mm_set1_epi16, _mm_setr_epi32,
    _mm_setzero_si128, _mm_shuffle_epi32, _mm_shuffle_ps, _mm_srli_si128, _mm_storeu_si128,
    _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi8,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm256_add_epi16, _mm256_add_epi64,
    _mm256_blendv_epi8, _mm256_broadcastsi128_si256, _mm256_castps_si256, _mm256_castsi128_si256,
    _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cmpeq_epi16, _mm256_cvtepi16_epi32,
    _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_inserti128_si256, _mm256_loadu_si256,
    _mm256_loadu2_m128i, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_maskz_loadu_epi8,
    _mm256_max_epu8, _mm256_min_epu8, _mm256_min_epu16, _mm256_mpsadbw_epu8, _mm256_packus_epi32,
    _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_permutevar8x32_epi32,
    _mm256_sad_epu8, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi32,
    _mm256_setr_epi64x, _mm256_setzero_si256, _mm256_shuffle_ps, _mm256_storeu_si256,
    _mm256_sub_epi8, _mm256_sub_epi16, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_zextsi128_si256,
};

use crate::simd::avx2::{I16x8, I16x16};

use super::{
    BlockPair, GRID_SMALL_BLOCKS, GridBest, GridCosts, MacroblockSad, PairwiseU16, PixelFns,
    normalise_satd_4x4, normalise_satd_8x8,
};

/// The AVX2 kernel table.
///
/// # Safety
///
/// The CPU must support AVX2 (`Isa::Avx2.is_supported()`).
pub(crate) unsafe fn pixel_fns() -> PixelFns {
    PixelFns {
        sad: [sad::<4>, sad::<8>, sad::<16>, sad::<32>, sad_w64],
        sad_x4: [
            sad_x4::<4>,
            sad_x4::<8>,
            sad_x4::<16>,
            sad_x4::<32>,
            sad_x4_w64,
        ],
        sad_quads_8x8_row,
        sad_scan_16x16_even_rows,
        sad_scan_bounded: Some(sad_scan_bounded),
        sad_grid_best_64x64: [sad_grid_best_64x64::<1>, sad_grid_best_64x64::<2>],
        sad_grid_best_x4: None,
        sad_quads_16x16,
        sad_4x4_macroblock,
        macroblock_sad: Some(macroblock_sad),
        satd_4x4: satd_4x4_tiles,
        satd_8x8: satd_8x8_tiles,
        satd_8x8_quads: None,
    }
}

// ---------------------------------------------------------------- loads ----

/// Rows of a `W`-wide block that fill one register (`W` = 4, 8, 16 or 32).
const fn rows_per_group(width: usize) -> usize {
    match width {
        4 | 8 => 4,
        16 => 2,
        _ => 1,
    }
}

/// Load one row group of a `W`-wide block starting at `p` (width 4 fills
/// only the low 128 bits; the rest is zero).
///
/// # Safety
///
/// `rows_per_group(W)` rows of `W` bytes, `stride` apart, must be readable at `p`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_group<const W: usize>(p: *const u8, stride: usize) -> __m256i {
    // SAFETY: the caller guarantees every row read here is in bounds; all
    // loads are unaligned-tolerant.
    unsafe {
        match W {
            4 => _mm256_zextsi128_si256(_mm_setr_epi32(
                read_i32(p),
                read_i32(p.add(stride)),
                read_i32(p.add(2 * stride)),
                read_i32(p.add(3 * stride)),
            )),
            8 => _mm256_setr_epi64x(
                read_i64(p),
                read_i64(p.add(stride)),
                read_i64(p.add(2 * stride)),
                read_i64(p.add(3 * stride)),
            ),
            16 => _mm256_loadu2_m128i(p.add(stride).cast(), p.cast()),
            _ => _mm256_loadu_si256(p.cast()),
        }
    }
}

/// # Safety
///
/// Four bytes must be readable at `p`.
#[inline]
unsafe fn read_i32(p: *const u8) -> i32 {
    // SAFETY: guaranteed by the caller.
    unsafe { p.cast::<i32>().read_unaligned() }
}

/// # Safety
///
/// Eight bytes must be readable at `p`.
#[inline]
unsafe fn read_i64(p: *const u8) -> i64 {
    // SAFETY: guaranteed by the caller.
    unsafe { p.cast::<i64>().read_unaligned() }
}

// ------------------------------------------------------------------ SAD ----

/// SAD for widths 4, 8, 16 and 32.
#[target_feature(enable = "avx2")]
unsafe fn sad<const W: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    h: usize,
) -> u32 {
    let mut acc = _mm256_setzero_si256();
    for y in (0..h).step_by(rows_per_group(W)) {
        // SAFETY: rows y..y + rows_per_group(W) are inside the block, which
        // the kernel contract guarantees both slices cover.
        let (a, b) = unsafe {
            (
                load_group::<W>(src.as_ptr().add(y * src_stride), src_stride),
                load_group::<W>(rf.as_ptr().add(y * ref_stride), ref_stride),
            )
        };
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(a, b));
    }
    sum_u64_lanes(acc)
}

/// SAD for width 64: both 32-byte halves of each row per iteration, into
/// two accumulators (one pass over the rows instead of two; measured 30%
/// faster than two width-32 calls against SVT-AV1's `sad64x64` on the 5950X).
#[target_feature(enable = "avx2")]
unsafe fn sad_w64(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize, h: usize) -> u32 {
    let (mut left, mut right) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    for y in 0..h {
        // SAFETY: row y holds 64 bytes in both slices (kernel contract).
        let (s, r) = unsafe {
            (
                src.as_ptr().add(y * src_stride),
                rf.as_ptr().add(y * ref_stride),
            )
        };
        // SAFETY: as above; unaligned 32-byte loads within the row.
        let (s0, s1, r0, r1) = unsafe {
            (
                _mm256_loadu_si256(s.cast()),
                _mm256_loadu_si256(s.add(32).cast()),
                _mm256_loadu_si256(r.cast()),
                _mm256_loadu_si256(r.add(32).cast()),
            )
        };
        left = _mm256_add_epi64(left, _mm256_sad_epu8(s0, r0));
        right = _mm256_add_epi64(right, _mm256_sad_epu8(s1, r1));
    }
    sum_u64_lanes(_mm256_add_epi64(left, right))
}

/// Four-candidate SAD for widths 4, 8, 16 and 32.
#[target_feature(enable = "avx2")]
unsafe fn sad_x4<const W: usize>(
    src: &[u8],
    src_stride: usize,
    refs: [&[u8]; 4],
    ref_stride: usize,
    h: usize,
) -> [u32; 4] {
    let mut acc = [_mm256_setzero_si256(); 4];
    for y in (0..h).step_by(rows_per_group(W)) {
        // SAFETY: rows y..y + rows_per_group(W) are inside the block.
        let a = unsafe { load_group::<W>(src.as_ptr().add(y * src_stride), src_stride) };
        for (sum, rf) in acc.iter_mut().zip(refs) {
            // SAFETY: as above, for each candidate.
            let b = unsafe { load_group::<W>(rf.as_ptr().add(y * ref_stride), ref_stride) };
            *sum = _mm256_add_epi64(*sum, _mm256_sad_epu8(a, b));
        }
    }
    // Closures do not inherit target features, so no `map` here.
    [
        sum_u64_lanes(acc[0]),
        sum_u64_lanes(acc[1]),
        sum_u64_lanes(acc[2]),
        sum_u64_lanes(acc[3]),
    ]
}

/// Four-candidate SAD for width 64: two width-32 columns.
#[target_feature(enable = "avx2")]
unsafe fn sad_x4_w64(
    src: &[u8],
    src_stride: usize,
    refs: [&[u8]; 4],
    ref_stride: usize,
    h: usize,
) -> [u32; 4] {
    // SAFETY: each 32-wide column lies inside the 64-wide block.
    let (left, right) = unsafe {
        (
            sad_x4::<32>(src, src_stride, refs, ref_stride, h),
            sad_x4::<32>(
                &src[32..],
                src_stride,
                refs.map(|r| &r[32..]),
                ref_stride,
                h,
            ),
        )
    };
    [0, 1, 2, 3].map(|i| left[i] + right[i])
}

/// [`sad_quads_8x8`] for each block along a row.
#[target_feature(enable = "avx2")]
unsafe fn sad_quads_8x8_row(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    out: &mut [[u16; 4]],
) {
    let groups = out.len() / 4;
    let (batches, tail) = out.as_chunks_mut::<4>();
    for (i, quads) in batches.iter_mut().enumerate() {
        let mut halves = [_mm256_setzero_si256(); 2];
        for (half, sum) in halves.iter_mut().enumerate() {
            for row in 0..4 {
                let y = 4 * half + row;
                // SAFETY: four adjacent 8x8 blocks cover 32 bytes of
                // every row in both slices (kernel contract).
                let (a, b) = unsafe {
                    (
                        _mm256_loadu_si256(src.as_ptr().add(32 * i + y * src_stride).cast()),
                        _mm256_loadu_si256(rf.as_ptr().add(32 * i + y * ref_stride).cast()),
                    )
                };
                let diff = _mm256_sub_epi8(_mm256_max_epu8(a, b), _mm256_min_epu8(a, b));
                // Unsigned byte differences times signed ones give pair
                // sums <= 510 per row; four rows stay <= 2040 in i16 lanes.
                *sum = _mm256_add_epi16(*sum, _mm256_maddubs_epi16(diff, _mm256_set1_epi8(1)));
            }
            // One widening sum per half: four columns' pair sums, <= 4080.
            *sum = _mm256_madd_epi16(*sum, _mm256_set1_epi16(1));
        }
        // Each half is [b0 left/right, b1 left/right | b2, b3].
        // Interleave top/bottom into four [TL, TR, BL, BR] records.
        let lo = _mm256_unpacklo_epi64(halves[0], halves[1]);
        let hi = _mm256_unpackhi_epi64(halves[0], halves[1]);
        let first = _mm256_permute2x128_si256::<0x20>(lo, hi);
        let second = _mm256_permute2x128_si256::<0x31>(lo, hi);
        // Pack to u16 (a 4x4 SAD is at most 4080): records [0 2 | 1 3],
        // then into order.
        let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi32(first, second));
        // SAFETY: the four records are 32 writable bytes.
        unsafe { _mm256_storeu_si256(quads.as_mut_ptr().cast(), packed) };
    }
    for (i, quads) in tail.iter_mut().enumerate() {
        let col = 8 * (4 * groups + i);
        // SAFETY: block i lies inside the row in both slices (kernel contract).
        *quads = unsafe { sad_quads_8x8(&src[col..], src_stride, &rf[col..], ref_stride) };
    }
}

/// Quadrant SADs of an 8x8 block. Each pair of rows is interleaved in
/// 32-bit units as [row y left, row y+1 left, row y right, row y+1 right],
/// so `psadbw` yields the left quadrant's sum in the low 64-bit lane and the
/// right quadrant's in the high lane. Rows 0..4 accumulate into the top
/// quadrants, 4..8 the bottom.
#[target_feature(enable = "avx2")]
unsafe fn sad_quads_8x8(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize) -> [u16; 4] {
    // SAFETY: the wrapper forwards the validated 8x8 slices.
    let lanes = unsafe { chroma_sad_lanes(src, src_stride, rf, ref_stride) };
    let packed = _mm_cvtsi128_si64(lanes) as u64;
    [0, 16, 32, 48].map(|shift| (packed >> shift) as u16)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn chroma_sad_lanes(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize) -> __m128i {
    let mut halves = [_mm_setzero_si128(); 2];
    for y in (0..8).step_by(2) {
        // SAFETY: rows y and y + 1 (8 bytes each) are inside the 8x8 blocks.
        let (a, b) = unsafe {
            (
                interleave_rows_4x2(src.as_ptr().add(y * src_stride), src_stride),
                interleave_rows_4x2(rf.as_ptr().add(y * ref_stride), ref_stride),
            )
        };
        halves[y / 4] = _mm_add_epi64(halves[y / 4], _mm_sad_epu8(a, b));
    }
    let ordered = pack_sad_lanes(halves[0], halves[1]);
    _mm_packus_epi32(ordered, _mm_setzero_si128())
}

/// The 4x4 SADs of a 4:2:0 macroblock in one call (the H.264 zero-residual
/// tests). Luma rows are paired like [`sad_quads_8x8`]'s, 16 bytes wide: the
/// low interleave covers columns 0..8 and the high one 8..16, so each
/// `psadbw` gives two 4x4 sums; every four rows close one row of 4x4 blocks.
/// Chroma blocks are [`sad_quads_8x8`].
#[target_feature(enable = "avx2")]
unsafe fn sad_4x4_macroblock(
    (src, src_stride, rf, ref_stride): BlockPair<'_>,
    [u, v]: [BlockPair<'_>; 2],
) -> ([[u16; 4]; 4], [[u16; 4]; 2]) {
    // SAFETY: the wrapper forwards the validated 16x16 slice pair.
    let lanes = unsafe { luma_sad_lanes((src, src_stride, rf, ref_stride)) };
    let mut luma = [[0u16; 4]; 4];
    for (lanes, pair) in lanes.into_iter().zip(luma.as_chunks_mut::<2>().0) {
        // SAFETY: each pair is sixteen writable bytes.
        unsafe { _mm_storeu_si128(pair.as_mut_ptr().cast(), lanes) };
    }
    // SAFETY: each chroma pair covers its 8x8 block.
    let chroma = unsafe {
        [
            sad_quads_8x8(u.0, u.1, u.2, u.3),
            sad_quads_8x8(v.0, v.1, v.2, v.3),
        ]
    };
    (luma, chroma)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn luma_sad_lanes((src, src_stride, rf, ref_stride): BlockPair<'_>) -> [__m128i; 2] {
    // The low and high 128-bit lanes follow the top and bottom
    // halves of the macroblock independently.
    let mut bands = [[_mm256_setzero_si256(); 2]; 2];
    for y in (0..8).step_by(2) {
        let load = |data: &[u8], stride: usize, row: usize| {
            // SAFETY: both rows lie inside the validated 16x16 pair.
            unsafe {
                _mm256_loadu2_m128i(
                    data.as_ptr().add((row + 8) * stride).cast(),
                    data.as_ptr().add(row * stride).cast(),
                )
            }
        };
        let (s0, s1) = (load(src, src_stride, y), load(src, src_stride, y + 1));
        let (r0, r1) = (load(rf, ref_stride, y), load(rf, ref_stride, y + 1));
        let row = &mut bands[y / 4];
        row[0] = _mm256_add_epi64(
            row[0],
            _mm256_sad_epu8(_mm256_unpacklo_epi32(s0, s1), _mm256_unpacklo_epi32(r0, r1)),
        );
        row[1] = _mm256_add_epi64(
            row[1],
            _mm256_sad_epu8(_mm256_unpackhi_epi32(s0, s1), _mm256_unpackhi_epi32(r0, r1)),
        );
    }
    let pack = |a, b| {
        _mm256_castps_si256(_mm256_shuffle_ps::<0x88>(
            _mm256_castsi256_ps(a),
            _mm256_castsi256_ps(b),
        ))
    };
    let packed = _mm256_packus_epi32(
        pack(bands[0][0], bands[1][0]),
        pack(bands[0][1], bands[1][1]),
    );
    [
        _mm256_castsi256_si128(packed),
        _mm256_extracti128_si256::<1>(packed),
    ]
}

/// The two low dwords of the 64-bit SAD lanes in each input.
#[inline]
#[target_feature(enable = "avx2")]
fn pack_sad_lanes(top: __m128i, bottom: __m128i) -> __m128i {
    _mm_castps_si128(_mm_shuffle_ps::<0x88>(
        _mm_castsi128_ps(top),
        _mm_castsi128_ps(bottom),
    ))
}

#[target_feature(enable = "avx2")]
unsafe fn macroblock_sad(
    luma: BlockPair<'_>,
    chroma: [BlockPair<'_>; 2],
    [luma_limit, ac_limit, dc_limit]: [u32; 3],
) -> MacroblockSad {
    // SAFETY: Dsp validated all three input pairs.
    let [top, bottom] = unsafe { luma_sad_lanes(luma) };
    let limit = _mm_set1_epi16(luma_limit.min(4080) as i16);
    let top_bad = _mm_cmpgt_epi16(top, limit);
    let bottom_bad = _mm_cmpgt_epi16(bottom, limit);
    let bad = _mm_movemask_epi8(_mm_packs_epi16(top_bad, bottom_bad)) as u32;
    // One bit for each failing quadrant, with four adjacent cells per quadrant.
    let any = bad | (bad >> 1) | (bad >> 2) | (bad >> 3);
    let luma_quadrants = !((any & 1) | ((any >> 3) & 2) | ((any >> 6) & 4) | ((any >> 9) & 8)) & 15;
    let sums = _mm_add_epi16(top, bottom);
    let sums = _mm_hadd_epi16(sums, sums);
    let sums = _mm_hadd_epi16(sums, sums);
    let luma_sad = _mm_extract_epi16::<0>(_mm_hadd_epi16(sums, sums)) as u16;
    // SAFETY: both pairs cover the validated chroma planes.
    let chroma = chroma.map(|pair| unsafe { chroma_skip(pair, ac_limit, dc_limit) });
    MacroblockSad {
        luma_sad,
        luma_quadrants: luma_quadrants as u8,
        chroma,
    }
}

/// Complete one plane before allocating registers for the other plane.
#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn chroma_skip((src, ss, pred, ps): BlockPair<'_>, ac_limit: u32, dc_limit: u32) -> bool {
    // SAFETY: the wrapper forwards one validated 8x8 plane.
    let sad = unsafe { chroma_sad_lanes(src, ss, pred, ps) };
    let limit = _mm_set1_epi16(ac_limit.min(4080) as i16);
    let bad = _mm_movemask_epi8(_mm_cmpgt_epi16(sad, limit));
    let pairs = _mm_hadd_epi16(sad, sad);
    let total = _mm_extract_epi16::<0>(_mm_hadd_epi16(pairs, pairs)) as u32;
    bad == 0 && total <= dc_limit
}

/// Rows 0 and 1 of an 8-wide block at `p`, as [row 0 cols 0..4, row 1 cols
/// 0..4, row 0 cols 4..8, row 1 cols 4..8].
///
/// # Safety
///
/// Two rows of 8 bytes, `stride` apart, must be readable at `p`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn interleave_rows_4x2(p: *const u8, stride: usize) -> __m128i {
    // SAFETY: guaranteed by the caller; unaligned 8-byte loads.
    let (r0, r1) = unsafe {
        (
            _mm_loadl_epi64(p.cast()),
            _mm_loadl_epi64(p.add(stride).cast()),
        )
    };
    _mm_unpacklo_epi32(r0, r1)
}

/// The two 64-bit lanes of a `psadbw` result, each holding a value < 2^32.
#[target_feature(enable = "avx2")]
#[inline]
fn lane_pair(v: __m128i) -> (u32, u32) {
    (
        _mm_cvtsi128_si32(v) as u32,
        _mm_cvtsi128_si32(_mm_shuffle_epi32::<0b10_10_10_10>(v)) as u32,
    )
}

/// Quadrant SADs of a 16x16 block. Each 256-bit load holds two 16-byte
/// rows; `vpsadbw` then yields one 64-bit sum per 8-byte half-row: lanes
/// [row y left, row y right, row y+1 left, row y+1 right]. Rows 0..8
/// accumulate into the top quadrants, 8..16 the bottom.
#[target_feature(enable = "avx2")]
unsafe fn sad_quads_16x16(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize) -> [u32; 4] {
    let mut halves = [_mm256_setzero_si256(); 2];
    for y in (0..16).step_by(2) {
        // SAFETY: rows y and y + 1 (16 bytes each) are inside the 16x16 blocks.
        let (a, b) = unsafe {
            (
                load_group::<16>(src.as_ptr().add(y * src_stride), src_stride),
                load_group::<16>(rf.as_ptr().add(y * ref_stride), ref_stride),
            )
        };
        halves[y / 8] = _mm256_add_epi64(halves[y / 8], _mm256_sad_epu8(a, b));
    }
    quad_sums(halves)
}

/// [top-left, top-right, bottom-left, bottom-right] from the two
/// accumulators of [`sad_quads_16x16`].
#[target_feature(enable = "avx2")]
#[inline]
fn quad_sums(halves: [__m256i; 2]) -> [u32; 4] {
    let (top_left, top_right) = left_right_sums(halves[0]);
    let (bottom_left, bottom_right) = left_right_sums(halves[1]);
    [top_left, top_right, bottom_left, bottom_right]
}

/// (left, right) half-row sums: 64-bit lanes are [left, right] for the even
/// rows (low 128 bits) and the odd rows (high 128 bits).
#[target_feature(enable = "avx2")]
#[inline]
fn left_right_sums(h: __m256i) -> (u32, u32) {
    lane_pair(_mm_add_epi64(
        _mm256_castsi256_si128(h),
        _mm256_extracti128_si256::<1>(h),
    ))
}

#[target_feature(enable = "avx2")]
#[inline]
fn sum_u64_lanes(v: __m256i) -> u32 {
    let sum = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    // Each 64-bit lane holds a value < 2^32; add the two lanes' low words.
    let high = _mm_shuffle_epi32::<0b10_10_10_10>(sum);
    _mm_cvtsi128_si32(_mm_add_epi32(sum, high)) as u32
}

/// AVX2 wrapper for the shared sliding scan.
#[target_feature(enable = "avx2")]
unsafe fn sad_scan_16x16_even_rows(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    out: &mut [u32],
) {
    // SAFETY: the dispatch validates the scan inputs and the AVX2 tier.
    unsafe {
        scan_sliding::<false, false>(src, src_stride, rf, ref_stride, out, 0);
    }
}

/// Ice Lake wrapper, using one masked 23-byte reference load.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
pub(super) unsafe fn sad_scan_16x16_even_rows_icl(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    out: &mut [u32],
) {
    // SAFETY: the dispatch validates the scan inputs and the Ice Lake tier.
    unsafe {
        scan_sliding::<true, false>(src, src_stride, rf, ref_stride, out, 0);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn sad_scan_bounded(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    out: &mut [u32],
    bound: u32,
) {
    // SAFETY: the dispatch validates the scan inputs and the AVX2 tier.
    unsafe { scan_sliding::<false, true>(src, src_stride, rf, ref_stride, out, bound) }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
pub(super) unsafe fn sad_scan_bounded_icl(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    out: &mut [u32],
    bound: u32,
) {
    // SAFETY: the dispatch validates the scan inputs and the Ice Lake tier.
    unsafe { scan_sliding::<true, true>(src, src_stride, rf, ref_stride, out, bound) }
}

/// SAD scan of a 16x16 block over rows 0, 2, ..., 14. Each 256-bit load
/// holds two of those rows (two rows apart in the plane); the four source
/// registers are loaded once and reused for every position.
#[inline(never)] // one compact short-path body shared by both x86 tiers
#[target_feature(enable = "avx2")]
unsafe fn scan_short(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize, out: &mut [u32]) {
    let mut a = [_mm256_setzero_si256(); 4];
    for (k, reg) in a.iter_mut().enumerate() {
        // SAFETY: rows 4k and 4k + 2 (16 bytes each) are inside the block.
        *reg = unsafe { load_group::<16>(src.as_ptr().add(4 * k * src_stride), 2 * src_stride) };
    }
    for (dx, sad) in out.iter_mut().enumerate() {
        let mut acc = _mm256_setzero_si256();
        for (k, &reg) in a.iter().enumerate() {
            // SAFETY: rows 4k and 4k + 2 of the candidate at offset dx are
            // inside the reference (kernel contract: 16 + n - 1 columns).
            let b = unsafe {
                load_group::<16>(rf.as_ptr().add(4 * k * ref_stride + dx), 2 * ref_stride)
            };
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(reg, b));
        }
        *sad = sum_u64_lanes(acc);
    }
}

/// Scan eight adjacent 16x16 candidates at once over rows 0, 2, ..., 14.
///
/// Four sliding four-byte SADs build the full row cost, as in SVT-AV1
/// v4.3.0 `compute_sad_intrin_avx2.c` (Intel, BSD-2-Clause and AOM patent
/// terms). The two AVX2 lanes cover opposite halves of a source row. The
/// shared x86 implementation reads only 23 reference bytes per full group.
///
/// # Safety
///
/// The scan footprints are validated by dispatch. AVX2 is available, and
/// `ICL` is set only when AVX-512BW/VL are available too.
#[allow(clippy::inline_always)] // the two backends enable different load instructions
#[inline(always)]
unsafe fn scan_sliding<const ICL: bool, const BOUNDED: bool>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    out: &mut [u32],
    bound: u32,
) {
    // SAFETY: the caller covers every whole group and tail and enables the ISA.
    unsafe {
        // Short edge windows leave too much of the sliding scan unused.
        if out.len() < 16 {
            scan_short(src, src_stride, rf, ref_stride, out);
            return;
        }
        let (groups, tail) = out.as_chunks_mut::<8>();
        let mut source = [_mm256_setzero_si256(); 8];
        for (row, reg) in source.iter_mut().enumerate() {
            let samples = _mm_loadu_si128(src.as_ptr().add(2 * row * src_stride).cast());
            *reg = _mm256_broadcastsi128_si256(samples);
        }
        'group: for (group, costs) in groups.iter_mut().enumerate() {
            let mut first = _mm256_setzero_si256();
            let mut second = _mm256_setzero_si256();
            for (row, &samples) in source.iter().enumerate() {
                // SAFETY: a full group covers 23 bytes, through the last
                // byte of candidate 7. The right load covers bytes 7..23;
                // shifting drops byte 7 and the zero byte is unused.
                let references = {
                    let p = rf.as_ptr().add(2 * row * ref_stride + 8 * group);
                    if ICL {
                        let window = _mm256_maskz_loadu_epi8((1 << 23) - 1, p.cast());
                        _mm256_permutevar8x32_epi32(
                            window,
                            _mm256_setr_epi32(0, 1, 2, 3, 2, 3, 4, 5),
                        )
                    } else {
                        let left = _mm_loadu_si128(p.cast());
                        let right = _mm_srli_si128::<1>(_mm_loadu_si128(p.add(7).cast()));
                        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(left), right)
                    }
                };
                // Each 3-bit selector chooses one source quartet and permits
                // sliding from reference byte 0 or 4 of the corresponding lane.
                first =
                    _mm256_add_epi16(first, _mm256_mpsadbw_epu8::<0b010_000>(references, samples));
                second = _mm256_add_epi16(
                    second,
                    _mm256_mpsadbw_epu8::<0b111_101>(references, samples),
                );
                if BOUNDED && row == 3 {
                    let halves = _mm256_add_epi16(first, second);
                    let partial = _mm_add_epi16(
                        _mm256_castsi256_si128(halves),
                        _mm256_extracti128_si256::<1>(halves),
                    );
                    // All remaining contributions are nonnegative.
                    if _mm_extract_epi16::<0>(_mm_minpos_epu16(partial)) as u32 >= bound {
                        costs.fill(bound);
                        continue 'group;
                    }
                }
            }
            // At most 8 * 16 * 255 = 32640 even after adding the two halves.
            let sum = _mm256_add_epi16(first, second);
            let sum = _mm_add_epi16(
                _mm256_castsi256_si128(sum),
                _mm256_extracti128_si256::<1>(sum),
            );
            _mm256_storeu_si256(costs.as_mut_ptr().cast(), _mm256_cvtepi16_epi32(sum));
        }
        if let [last] = tail {
            // Common odd windows end in one position. Reuse the source
            // samples already resident in registers for the full groups.
            let start = rf.as_ptr().add(groups.len() * 8);
            let mut sums = [_mm_setzero_si128(); 2];
            for (pair, src_rows) in source.as_chunks::<2>().0.iter().enumerate() {
                for (row, &samples) in src_rows.iter().enumerate() {
                    let reference =
                        _mm_loadu_si128(start.add((4 * pair + 2 * row) * ref_stride).cast());
                    sums[row] = _mm_add_epi32(
                        sums[row],
                        _mm_sad_epu8(_mm256_castsi256_si128(samples), reference),
                    );
                }
            }
            let total = _mm_add_epi32(sums[0], sums[1]);
            *last =
                _mm_cvtsi128_si32(_mm_add_epi32(total, _mm_shuffle_epi32::<0x4e>(total))) as u32;
        } else if !tail.is_empty() {
            scan_short(src, src_stride, &rf[groups.len() * 8..], ref_stride, tail);
        }
    }
}

/// The 8x8 SADs of a 64x64 block. A row is two 256-bit loads; `vpsadbw`
/// sums each 8-byte group, i.e. one 8x8 column, into its own 64-bit lane,
/// so eight rows accumulate four 8x8 SADs per register.
#[target_feature(enable = "avx2")]
unsafe fn sad_grid_8x8_in_64x64<const STEP: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
) -> [u16; 64] {
    let mut grid = [0u16; 64];
    for (block_row, out) in grid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut acc = [_mm256_setzero_si256(); 2];
        for y in (8 * block_row..8 * block_row + 8).step_by(STEP) {
            for (half, sums) in acc.iter_mut().enumerate() {
                // SAFETY: bytes 32h..32h + 32 of row y are inside both
                // 64x64 blocks (kernel contract).
                let (a, b) = unsafe {
                    (
                        _mm256_loadu_si256(src.as_ptr().add(y * src_stride + 32 * half).cast()),
                        _mm256_loadu_si256(rf.as_ptr().add(y * ref_stride + 32 * half).cast()),
                    )
                };
                *sums = _mm256_add_epi64(*sums, _mm256_sad_epu8(a, b));
            }
        }
        // Each 64-bit lane holds one SAD below 2^14. As u32 lanes the pack
        // gives blocks [0 1 4 5 | 2 3 6 7]; reorder, then pack to u16.
        let blocks = _mm256_permutevar8x32_epi32(
            _mm256_packus_epi32(acc[0], acc[1]),
            _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7),
        );
        let blocks = _mm_packus_epi32(
            _mm256_castsi256_si128(blocks),
            _mm256_extracti128_si256::<1>(blocks),
        );
        // SAFETY: `out` holds eight u16.
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), blocks) };
    }
    grid
}

/// [`sad_grid_8x8_in_64x64`] summed into every block size and folded into
/// `best`.
#[target_feature(enable = "avx2")]
unsafe fn sad_grid_best_64x64<const STEP: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    visible: &[u16; 64],
    best: &mut GridBest,
    tag: u32,
) {
    // SAFETY: the kernel contract covers both 64x64 blocks.
    let grid = unsafe { sad_grid_8x8_in_64x64::<STEP>(src, src_stride, rf, ref_stride) };
    keep_min(
        best,
        &super::grid_costs::<Pairs>(&grid, visible, STEP as u16),
        tag,
    );
}

/// [`PairwiseU16`] as one `phaddw` (a wrapping add: the sums fit u16).
pub(super) struct Pairs;

impl PairwiseU16 for Pairs {
    #[allow(clippy::inline_always)] // must inline into the grid kernels
    #[inline(always)]
    fn pairwise(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
        let mut out = [0; 8];
        // SAFETY: AVX2 (and AVX-512) kernels only (module docs); each array
        // holds eight u16.
        unsafe {
            let sums = _mm_hadd_epi16(
                _mm_loadu_si128(a.as_ptr().cast()),
                _mm_loadu_si128(b.as_ptr().cast()),
            );
            _mm_storeu_si128(out.as_mut_ptr().cast(), sums);
        }
        out
    }
}

/// Keep-minimum over a grid update: the u16 SADs sixteen blocks per step
/// (`vpminuw`), the tag taken from `tag` wherever the minimum changed the
/// best (the compare mask sign-extended to the u32 tags), then the five
/// large blocks.
#[target_feature(enable = "avx2")]
fn keep_min(best: &mut GridBest, cost: &GridCosts, tag: u32) {
    let fresh = _mm256_set1_epi32(tag as i32);
    for i in (0..GRID_SMALL_BLOCKS).step_by(16) {
        // SAFETY: elements i..i + 16 are inside the 80-element SAD arrays
        // and the 85-element tag array.
        unsafe {
            let b = _mm256_loadu_si256(best.small.as_ptr().add(i).cast());
            let c = _mm256_loadu_si256(cost.small.as_ptr().add(i).cast());
            let min = _mm256_min_epu16(b, c);
            let unchanged = _mm256_cmpeq_epi16(min, b);
            _mm256_storeu_si256(best.small.as_mut_ptr().add(i).cast(), min);
            let masks = [
                _mm256_cvtepi16_epi32(_mm256_castsi256_si128(unchanged)),
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(unchanged)),
            ];
            for (half, mask) in masks.into_iter().enumerate() {
                let at = best.tag.as_mut_ptr().add(i + 8 * half).cast();
                _mm256_storeu_si256(at, _mm256_blendv_epi8(fresh, _mm256_loadu_si256(at), mask));
            }
        }
    }
    super::scalar::keep_min_large(best, cost, tag);
}

// ----------------------------------------------------------------- SATD ----

#[target_feature(enable = "avx2")]
unsafe fn satd_4x4_tiles(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    if w == 4 && h == 4 {
        // SAFETY: the checked source and reference cover one full tile.
        let sum =
            unsafe { hadamard_4x4_abs_sum(src.as_ptr(), src_stride, rf.as_ptr(), ref_stride) };
        return normalise_satd_4x4(sum);
    }
    let mut total = 0;
    for ty in (0..h).step_by(4) {
        for tx in (0..w).step_by(4) {
            // SAFETY: the 4x4 tile at (tx, ty) is inside the block.
            let sum = unsafe {
                hadamard_4x4_abs_sum(
                    src.as_ptr().add(ty * src_stride + tx),
                    src_stride,
                    rf.as_ptr().add(ty * ref_stride + tx),
                    ref_stride,
                )
            };
            total += normalise_satd_4x4(sum);
        }
    }
    total
}

/// SATD on 8x8 tiles. Blocks at least 16 wide transform two horizontally
/// adjacent tiles at once, one per 128-bit lane of a 256-bit register
/// (`vphaddw` and friends work per lane, so the two tiles never mix).
#[target_feature(enable = "avx2")]
pub(super) unsafe fn satd_8x8_tiles(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    if w == 8 && h == 8 {
        // SAFETY: the checked source and reference cover one full tile.
        let sum =
            unsafe { hadamard_8x8_abs_sum(src.as_ptr(), src_stride, rf.as_ptr(), ref_stride) };
        return normalise_satd_8x8(sum);
    }
    let mut total = 0;
    if w == 8 && h >= 16 {
        for ty in (0..h).step_by(16) {
            // SAFETY: both 8x8 tiles lie within the vertically paired rows.
            let [top, bottom] = unsafe {
                hadamard_8x8_pair_abs_sums::<true>(
                    src.as_ptr().add(ty * src_stride),
                    src_stride,
                    rf.as_ptr().add(ty * ref_stride),
                    ref_stride,
                )
            };
            total += normalise_satd_8x8(top) + normalise_satd_8x8(bottom);
        }
        return total;
    }
    for ty in (0..h).step_by(8) {
        for tx in (0..w).step_by(16) {
            // SAFETY: the 16x8 region at (tx, ty) is inside the block
            // (w is a multiple of 16 here).
            let [left, right] = unsafe {
                hadamard_8x8_pair_abs_sums::<false>(
                    src.as_ptr().add(ty * src_stride + tx),
                    src_stride,
                    rf.as_ptr().add(ty * ref_stride + tx),
                    ref_stride,
                )
            };
            total += normalise_satd_8x8(left) + normalise_satd_8x8(right);
        }
    }
    total
}

/// Unnormalised Hadamard absolute sum of one 4x4 difference tile.
///
/// # Safety
///
/// Four rows of four bytes must be readable at `src` and `rf`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hadamard_4x4_abs_sum(
    src: *const u8,
    src_stride: usize,
    rf: *const u8,
    ref_stride: usize,
) -> u32 {
    let packed_rows = |first, second| {
        // SAFETY: each row of the tile has four readable bytes.
        unsafe {
            let a = _mm_unpacklo_epi32(
                _mm_cvtsi32_si128(read_i32(src.add(first * src_stride))),
                _mm_cvtsi32_si128(read_i32(src.add(second * src_stride))),
            );
            let b = _mm_unpacklo_epi32(
                _mm_cvtsi32_si128(read_i32(rf.add(first * ref_stride))),
                _mm_cvtsi32_si128(read_i32(rf.add(second * ref_stride))),
            );
            _mm_sub_epi16(_mm_cvtepu8_epi16(a), _mm_cvtepu8_epi16(b))
        }
    };
    let even = packed_rows(0, 2);
    let odd = packed_rows(1, 3);
    let sum = _mm_add_epi16(even, odd);
    let diff = _mm_sub_epi16(even, odd);
    let lo = _mm_unpacklo_epi16(sum, diff);
    let hi = _mm_unpackhi_epi16(sum, diff);
    let sum = _mm_add_epi16(lo, hi);
    let diff = _mm_sub_epi16(lo, hi);
    let lo = _mm_unpacklo_epi32(sum, diff);
    let hi = _mm_unpackhi_epi32(sum, diff);
    let sum = _mm_add_epi16(lo, hi);
    let diff = _mm_sub_epi16(lo, hi);
    let even = _mm_unpacklo_epi64(sum, diff);
    let odd = _mm_unpackhi_epi64(sum, diff);
    let half = _mm_max_epi16(_mm_abs_epi16(even), _mm_abs_epi16(odd));
    sum_i32_lanes(_mm_madd_epi16(half, _mm_set1_epi16(2)))
}

/// Unnormalised Hadamard absolute sum of one 8x8 difference tile.
///
/// # Safety
///
/// Eight rows of eight bytes must be readable at `src` and `rf`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hadamard_8x8_abs_sum(
    src: *const u8,
    src_stride: usize,
    rf: *const u8,
    ref_stride: usize,
) -> u32 {
    let mut d = [_mm_setzero_si128(); 8];
    for (y, row) in d.iter_mut().enumerate() {
        // SAFETY: row y of the tile is readable (caller contract).
        let (a, b) = unsafe {
            (
                _mm_loadl_epi64(src.add(y * src_stride).cast()),
                _mm_loadl_epi64(rf.add(y * ref_stride).cast()),
            )
        };
        *row = _mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), _mm_set1_epi16(-255));
    }
    let half = super::satd_x86::satd_half(d.map(I16x8)).0;
    sum_i32_lanes(_mm_madd_epi16(half, _mm_set1_epi16(2)))
}

/// Unnormalised Hadamard absolute sums of two adjacent 8x8 difference tiles: the same algorithm as [`hadamard_8x8_abs_sum`], with
/// the left tile in the low 128-bit lane and the right tile in the high lane.
///
/// # Safety
///
/// Each input must contain both adjacent tiles in the specified direction.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hadamard_8x8_pair_abs_sums<const VERTICAL: bool>(
    src: *const u8,
    src_stride: usize,
    rf: *const u8,
    ref_stride: usize,
) -> [u32; 2] {
    let gather = |source: *const u8, stride| {
        // SAFETY: the caller supplies both complete tiles.
        unsafe {
            if VERTICAL {
                _mm_unpacklo_epi64(
                    _mm_loadl_epi64(source.cast()),
                    _mm_loadl_epi64(source.add(8 * stride).cast()),
                )
            } else {
                _mm_loadu_si128(source.cast())
            }
        }
    };
    let mut d = [_mm256_setzero_si256(); 8];
    for (y, row) in d.iter_mut().enumerate() {
        // SAFETY: both tiles are readable (caller contract).
        let (a, b) = unsafe {
            (
                gather(src.add(y * src_stride), src_stride),
                gather(rf.add(y * ref_stride), ref_stride),
            )
        };
        *row = _mm256_sub_epi16(_mm256_cvtepu8_epi16(a), _mm256_cvtepu8_epi16(b));
    }
    let half = super::satd_x86::satd_half(d.map(I16x16)).0;
    let sum = _mm256_madd_epi16(half, _mm256_set1_epi16(2));
    [
        sum_i32_lanes(_mm256_castsi256_si128(sum)),
        sum_i32_lanes(_mm256_extracti128_si256::<1>(sum)),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn sum_i32_lanes(v: __m128i) -> u32 {
    let v = _mm_add_epi32(v, _mm_shuffle_epi32::<0b01_00_11_10>(v));
    let v = _mm_add_epi32(v, _mm_shuffle_epi32::<0b10_11_00_01>(v));
    _mm_cvtsi128_si32(v) as u32
}
