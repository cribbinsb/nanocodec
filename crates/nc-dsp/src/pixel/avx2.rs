//! AVX2 pixel kernels (x86-64).
//!
//! SAD packs rows into 256-bit registers and uses `vpsadbw`. Widths 4 and 8
//! gather several rows per register, and width 64 is two width-32 columns.
//! SATD runs the vertical Hadamard stages as butterflies across row
//! registers, then the horizontal stages with `phaddw`/`phsubw` on pairs of
//! rows. That gives the full transform without a transpose. For blocks at
//! least 16 wide, two 8x8 tiles share each 256-bit register (one per lane).
//!
//! Kernel contract: see the `pixel` module docs. Arithmetic intrinsics are
//! safe inside `#[target_feature(enable = "avx2")]` functions; only the
//! loads need `unsafe`.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_abs_epi16, _mm_add_epi16, _mm_add_epi32, _mm_add_epi64,
    _mm_cvtepu8_epi16, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_hadd_epi16, _mm_hsub_epi16,
    _mm_loadl_epi64, _mm_loadu_si128, _mm_madd_epi16, _mm_sad_epu8, _mm_set1_epi16, _mm_setr_epi32,
    _mm_setzero_si128, _mm_shuffle_epi32, _mm_sub_epi16, _mm_unpackhi_epi32, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64,
    _mm256_blendv_epi8, _mm256_castsi256_si128, _mm256_cmpeq_epi32, _mm256_cvtepu8_epi16,
    _mm256_extracti128_si256, _mm256_hadd_epi16, _mm256_hsub_epi16, _mm256_loadu_si256,
    _mm256_loadu2_m128i, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_max_epu8, _mm256_min_epu8,
    _mm256_min_epu32, _mm256_permute2x128_si256, _mm256_sad_epu8, _mm256_set1_epi8,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi64x, _mm256_setzero_si256,
    _mm256_storeu_si256, _mm256_sub_epi8, _mm256_sub_epi16, _mm256_unpackhi_epi64,
    _mm256_unpacklo_epi64, _mm256_zextsi128_si256,
};

use super::{BlockPair, GRID_BLOCKS, GridBest, PixelFns, normalise_satd_4x4, normalise_satd_8x8};

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
        sad_grid_best_64x64: [sad_grid_best_64x64::<1>, sad_grid_best_64x64::<2>],
        sad_grid_best_x4: None,
        sad_quads_16x16,
        sad_4x4_macroblock,
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
    out: &mut [[u32; 4]],
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
        // SAFETY: the four records contain 64 writable bytes. The two
        // unaligned stores cover the first and last two records.
        unsafe {
            let dst = quads.as_mut_ptr();
            _mm256_storeu_si256(dst.cast(), first);
            _mm256_storeu_si256(dst.add(2).cast(), second);
        }
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
unsafe fn sad_quads_8x8(src: &[u8], src_stride: usize, rf: &[u8], ref_stride: usize) -> [u32; 4] {
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
    let (top_left, top_right) = lane_pair(halves[0]);
    let (bottom_left, bottom_right) = lane_pair(halves[1]);
    [top_left, top_right, bottom_left, bottom_right]
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
) -> ([[u32; 4]; 4], [[u32; 4]; 2]) {
    // [block row][left 8 columns, right 8 columns]: 64-bit lanes hold the
    // sums of 4x4 blocks (2k, 2k + 1) of that half.
    let mut sums = [[_mm_setzero_si128(); 2]; 4];
    for y in (0..16).step_by(2) {
        // SAFETY: rows y and y + 1 (16 bytes each) lie inside the 16x16
        // blocks (kernel contract).
        let (s0, s1, r0, r1) = unsafe {
            (
                _mm_loadu_si128(src.as_ptr().add(y * src_stride).cast()),
                _mm_loadu_si128(src.as_ptr().add((y + 1) * src_stride).cast()),
                _mm_loadu_si128(rf.as_ptr().add(y * ref_stride).cast()),
                _mm_loadu_si128(rf.as_ptr().add((y + 1) * ref_stride).cast()),
            )
        };
        let row = &mut sums[y / 4];
        row[0] = _mm_add_epi64(
            row[0],
            _mm_sad_epu8(_mm_unpacklo_epi32(s0, s1), _mm_unpacklo_epi32(r0, r1)),
        );
        row[1] = _mm_add_epi64(
            row[1],
            _mm_sad_epu8(_mm_unpackhi_epi32(s0, s1), _mm_unpackhi_epi32(r0, r1)),
        );
    }
    // 4x4 block (bx, by) is in quadrant (bx / 2, by / 2) at (bx % 2, by % 2).
    let mut luma = [[0u32; 4]; 4];
    for (by, halves) in sums.iter().enumerate() {
        for (half, &pair) in halves.iter().enumerate() {
            let (left, right) = lane_pair(pair);
            let quadrant = &mut luma[(by / 2) * 2 + half];
            quadrant[(by % 2) * 2] = left;
            quadrant[(by % 2) * 2 + 1] = right;
        }
    }
    // SAFETY: each chroma pair covers its 8x8 block (kernel contract).
    let chroma = unsafe {
        [
            sad_quads_8x8(u.0, u.1, u.2, u.3),
            sad_quads_8x8(v.0, v.1, v.2, v.3),
        ]
    };
    (luma, chroma)
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

/// SAD scan of a 16x16 block over rows 0, 2, ..., 14. Each 256-bit load
/// holds two of those rows (four rows apart in the plane); the four source
/// registers are loaded once and reused for every position.
#[target_feature(enable = "avx2")]
unsafe fn sad_scan_16x16_even_rows(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
    out: &mut [u32],
) {
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

/// The 8x8 SADs of a 64x64 block. A row is two 256-bit loads; `vpsadbw`
/// sums each 8-byte group, i.e. one 8x8 column, into its own 64-bit lane,
/// so eight rows accumulate four 8x8 SADs per register.
#[target_feature(enable = "avx2")]
unsafe fn sad_grid_8x8_in_64x64<const STEP: usize>(
    src: &[u8],
    src_stride: usize,
    rf: &[u8],
    ref_stride: usize,
) -> [u32; 64] {
    let mut grid = [0u32; 64];
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
        for (half, &sums) in acc.iter().enumerate() {
            let mut lanes = [0u64; 4];
            // SAFETY: `lanes` is 32 writable bytes.
            unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sums) };
            for (i, &lane) in lanes.iter().enumerate() {
                out[4 * half + i] = lane as u32;
            }
        }
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
    visible: &[u32; 64],
    best: &mut GridBest,
    tag: u32,
) {
    // SAFETY: the kernel contract covers both 64x64 blocks.
    let grid = unsafe { sad_grid_8x8_in_64x64::<STEP>(src, src_stride, rf, ref_stride) };
    keep_min(best, &super::grid_costs(&grid, visible, STEP as u32), tag);
}

/// Keep-minimum over a grid update, eight blocks per step: `vpminud`, and
/// the tag taken from `tag` wherever the minimum changed the best.
#[target_feature(enable = "avx2")]
fn keep_min(best: &mut GridBest, cost: &[u32; GRID_BLOCKS], tag: u32) {
    let lanes = GRID_BLOCKS / 8 * 8;
    let fresh = _mm256_set1_epi32(tag as i32);
    for i in (0..lanes).step_by(8) {
        // SAFETY: elements i..i + 8 are inside all three 85-element arrays.
        unsafe {
            let b = _mm256_loadu_si256(best.sad.as_ptr().add(i).cast());
            let c = _mm256_loadu_si256(cost.as_ptr().add(i).cast());
            let t = _mm256_loadu_si256(best.tag.as_ptr().add(i).cast());
            let min = _mm256_min_epu32(b, c);
            let unchanged = _mm256_cmpeq_epi32(min, b);
            _mm256_storeu_si256(best.sad.as_mut_ptr().add(i).cast(), min);
            _mm256_storeu_si256(
                best.tag.as_mut_ptr().add(i).cast(),
                _mm256_blendv_epi8(fresh, t, unchanged),
            );
        }
    }
    super::scalar::keep_min(
        &mut best.sad[lanes..],
        &mut best.tag[lanes..],
        &cost[lanes..],
        tag,
    );
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
    let mut total = 0;
    for ty in (0..h).step_by(8) {
        if w == 8 {
            // SAFETY: the 8x8 tile at (0, ty) is inside the block.
            let sum = unsafe {
                hadamard_8x8_abs_sum(
                    src.as_ptr().add(ty * src_stride),
                    src_stride,
                    rf.as_ptr().add(ty * ref_stride),
                    ref_stride,
                )
            };
            total += normalise_satd_8x8(sum);
            continue;
        }
        for tx in (0..w).step_by(16) {
            // SAFETY: the 16x8 region at (tx, ty) is inside the block
            // (w is a multiple of 16 here).
            let [left, right] = unsafe {
                hadamard_8x8_pair_abs_sums(
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
    let mut d = [_mm_setzero_si128(); 4];
    for (y, row) in d.iter_mut().enumerate() {
        // SAFETY: row y of the tile is readable (caller contract).
        let (a, b) = unsafe {
            (
                read_i32(src.add(y * src_stride)),
                read_i32(rf.add(y * ref_stride)),
            )
        };
        *row = _mm_sub_epi16(
            _mm_cvtepu8_epi16(_mm_cvtsi32_si128(a)),
            _mm_cvtepu8_epi16(_mm_cvtsi32_si128(b)),
        );
    }
    butterflies(&mut d, 1);
    butterflies(&mut d, 2);
    // Two rows per register; two pairwise add/sub stages complete each row.
    let (a, b) = (
        _mm_unpacklo_epi64(d[0], d[1]),
        _mm_unpacklo_epi64(d[2], d[3]),
    );
    let (s1, d1) = (_mm_hadd_epi16(a, b), _mm_hsub_epi16(a, b));
    let (s2, d2) = (_mm_hadd_epi16(s1, d1), _mm_hsub_epi16(s1, d1));
    sum_i32_lanes(_mm_add_epi32(abs_sum_pairs(s2), abs_sum_pairs(d2)))
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
        *row = _mm_sub_epi16(_mm_cvtepu8_epi16(a), _mm_cvtepu8_epi16(b));
    }
    butterflies(&mut d, 1);
    butterflies(&mut d, 2);
    butterflies(&mut d, 4);
    // Values stay below 8 * 8 * 255 = 16320, well inside i16.
    let mut sum = _mm_setzero_si128();
    for pair in d.as_chunks::<2>().0 {
        let (a, b) = (pair[0], pair[1]);
        let (s1, d1) = (_mm_hadd_epi16(a, b), _mm_hsub_epi16(a, b));
        let (s2, d2) = (_mm_hadd_epi16(s1, d1), _mm_hsub_epi16(s1, d1));
        let (s3, d3) = (_mm_hadd_epi16(s2, d2), _mm_hsub_epi16(s2, d2));
        sum = _mm_add_epi32(sum, _mm_add_epi32(abs_sum_pairs(s3), abs_sum_pairs(d3)));
    }
    sum_i32_lanes(sum)
}

/// Unnormalised Hadamard absolute sums of two horizontally adjacent 8x8
/// difference tiles: the same algorithm as [`hadamard_8x8_abs_sum`], with
/// the left tile in the low 128-bit lane and the right tile in the high lane.
///
/// # Safety
///
/// Eight rows of sixteen bytes must be readable at `src` and `rf`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hadamard_8x8_pair_abs_sums(
    src: *const u8,
    src_stride: usize,
    rf: *const u8,
    ref_stride: usize,
) -> [u32; 2] {
    let mut d = [_mm256_setzero_si256(); 8];
    for (y, row) in d.iter_mut().enumerate() {
        // SAFETY: row y of the 16x8 region is readable (caller contract).
        let (a, b) = unsafe {
            (
                _mm_loadu_si128(src.add(y * src_stride).cast()),
                _mm_loadu_si128(rf.add(y * ref_stride).cast()),
            )
        };
        *row = _mm256_sub_epi16(_mm256_cvtepu8_epi16(a), _mm256_cvtepu8_epi16(b));
    }
    butterflies_256(&mut d, 1);
    butterflies_256(&mut d, 2);
    butterflies_256(&mut d, 4);
    let mut sum = _mm256_setzero_si256();
    for pair in d.as_chunks::<2>().0 {
        let (a, b) = (pair[0], pair[1]);
        let (s1, d1) = (_mm256_hadd_epi16(a, b), _mm256_hsub_epi16(a, b));
        let (s2, d2) = (_mm256_hadd_epi16(s1, d1), _mm256_hsub_epi16(s1, d1));
        let (s3, d3) = (_mm256_hadd_epi16(s2, d2), _mm256_hsub_epi16(s2, d2));
        let abs = _mm256_add_epi32(abs_sum_pairs_256(s3), abs_sum_pairs_256(d3));
        sum = _mm256_add_epi32(sum, abs);
    }
    [
        sum_i32_lanes(_mm256_castsi256_si128(sum)),
        sum_i32_lanes(_mm256_extracti128_si256::<1>(sum)),
    ]
}

/// [`butterflies`] on 256-bit rows.
#[target_feature(enable = "avx2")]
#[inline]
fn butterflies_256<const N: usize>(rows: &mut [__m256i; N], half: usize) {
    for start in (0..N).step_by(2 * half) {
        for i in start..start + half {
            let (a, b) = (rows[i], rows[i + half]);
            rows[i] = _mm256_add_epi16(a, b);
            rows[i + half] = _mm256_sub_epi16(a, b);
        }
    }
}

/// [`abs_sum_pairs`] on a 256-bit register.
#[target_feature(enable = "avx2")]
#[inline]
fn abs_sum_pairs_256(x: __m256i) -> __m256i {
    _mm256_madd_epi16(_mm256_abs_epi16(x), _mm256_set1_epi16(1))
}

/// One butterfly stage across row registers: rows `i` and `i + half`.
#[target_feature(enable = "avx2")]
#[inline]
fn butterflies<const N: usize>(rows: &mut [__m128i; N], half: usize) {
    for start in (0..N).step_by(2 * half) {
        for i in start..start + half {
            let (a, b) = (rows[i], rows[i + half]);
            rows[i] = _mm_add_epi16(a, b);
            rows[i + half] = _mm_sub_epi16(a, b);
        }
    }
}

/// |x| summed in pairs of i16 lanes, giving four i32 lanes.
#[target_feature(enable = "avx2")]
#[inline]
fn abs_sum_pairs(x: __m128i) -> __m128i {
    _mm_madd_epi16(_mm_abs_epi16(x), _mm_set1_epi16(1))
}

#[target_feature(enable = "avx2")]
#[inline]
fn sum_i32_lanes(v: __m128i) -> u32 {
    let v = _mm_add_epi32(v, _mm_shuffle_epi32::<0b01_00_11_10>(v));
    let v = _mm_add_epi32(v, _mm_shuffle_epi32::<0b10_11_00_01>(v));
    _mm_cvtsi128_si32(v) as u32
}
