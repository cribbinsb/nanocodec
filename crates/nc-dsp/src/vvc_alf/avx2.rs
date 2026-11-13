//! AVX2 ALF kernels: sixteen samples per vector in 16-bit lanes.
//!
//! The filter pairs taps for `vpmaddwd`: interleaving two taps' sums puts
//! the four samples of one 4x4 block in each 128-bit half, so a block's
//! coefficient pair is one broadcast 32-bit value. The Laplacians select
//! the checkerboard's rows per lane before computing, then reduce groups of
//! four columns with two horizontal adds.

use super::{
    ACTIVITY_CLASS, ACTIVITY_CLASSES, ACTIVITY_SCALE, ACTIVITY_SHIFT, BOUNDARY_ACTIVITY_SCALE,
    MAX_ACTIVITY, SAMPLE_BYTE_SOURCES, TRANSPOSE, VVC_ALF_BLOCK, VVC_ALF_DOT_LANES,
    VVC_ALF_GRAM_ALIGN, VVC_ALF_GRAM_COLUMNS, VVC_ALF_GRAM_TERMS, VVC_ALF_REACH,
    VVC_ALF_SAMPLE_LANES, VVC_ALF_SAMPLES, VVC_ALF_TAP_OFFSETS, VVC_ALF_TAPS, classify_from,
    filter_block_row_from, laplacians_from, vvc_alf_gram_index,
};
use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi32_si128, _mm_loadl_epi64, _mm_loadu_si128, _mm_setr_epi8,
    _mm_shuffle_epi8, _mm_storeu_si128, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64,
    _mm256_adds_epi16, _mm256_and_si256, _mm256_blend_epi16, _mm256_blendv_epi8,
    _mm256_broadcastsi128_si256, _mm256_castsi256_si128, _mm256_cmpgt_epi32, _mm256_cvtepu8_epi16,
    _mm256_cvtepu16_epi32, _mm256_extracti128_si256, _mm256_hadd_epi16, _mm256_loadu_si256,
    _mm256_madd_epi16, _mm256_max_epi32, _mm256_min_epi32, _mm256_min_epu32, _mm256_mul_epi32,
    _mm256_mullo_epi32, _mm256_or_si256, _mm256_packs_epi32, _mm256_packus_epi16,
    _mm256_packus_epi32, _mm256_permute2x128_si256, _mm256_permute4x64_epi64,
    _mm256_permutevar8x32_epi32, _mm256_set_m128i, _mm256_set1_epi32, _mm256_setr_epi8,
    _mm256_setr_epi32, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_shuffle_epi32,
    _mm256_slli_epi32, _mm256_sra_epi32, _mm256_srli_epi32, _mm256_srli_epi64, _mm256_storeu_si256,
    _mm256_sub_epi16, _mm256_sub_epi32, _mm256_unpackhi_epi16, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi16, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

/// Samples per vector.
const LANES: usize = 16;
/// Samples one vector of filter output reads from each row.
const WINDOW: usize = LANES + 2 * VVC_ALF_REACH;
/// `vpblendw` mask taking the odd 16-bit lanes from the second operand.
const ODD_LANES: i32 = 0b1010_1010;

/// Sixteen samples of `row` from `at`, widened to 16 bits.
#[target_feature(enable = "avx2")]
fn load(row: &[u8], at: usize) -> __m256i {
    let samples: &[u8; LANES] = row[at..at + LANES].try_into().expect("sixteen samples");
    // SAFETY: AVX2 is enabled; the load reads exactly the sixteen bytes of
    // `samples`.
    _mm256_cvtepu8_epi16(unsafe { _mm_loadu_si128(samples.as_ptr().cast()) })
}

/// Coefficient pairs `(2p, 2p + 1)` of blocks `first` (low half) and
/// `second` (high half), broadcast over each half's four 32-bit lanes.
#[target_feature(enable = "avx2")]
fn coefficient_pairs(first: &[i16; VVC_ALF_TAPS], second: &[i16; VVC_ALF_TAPS]) -> [__m256i; 6] {
    // SAFETY: AVX2 is enabled; each load reads 16 or 8 bytes within a
    // 24-byte coefficient array.
    let (pairs_0_3, pairs_4_5) = unsafe {
        (
            _mm256_set_m128i(
                _mm_loadu_si128(second.as_ptr().cast()),
                _mm_loadu_si128(first.as_ptr().cast()),
            ),
            _mm256_set_m128i(
                _mm_loadl_epi64(second[8..].as_ptr().cast()),
                _mm_loadl_epi64(first[8..].as_ptr().cast()),
            ),
        )
    };
    [
        _mm256_shuffle_epi32::<0x00>(pairs_0_3),
        _mm256_shuffle_epi32::<0x55>(pairs_0_3),
        _mm256_shuffle_epi32::<0xaa>(pairs_0_3),
        _mm256_shuffle_epi32::<0xff>(pairs_0_3),
        _mm256_shuffle_epi32::<0x00>(pairs_4_5),
        _mm256_shuffle_epi32::<0x55>(pairs_4_5),
    ]
}

/// One row's sixteen filtered samples from its `windows` (each row's
/// samples from 3 columns left of the first output), with the blocks'
/// coefficient pairs `low` (blocks 0 and 2) and `high` (blocks 1 and 3).
#[target_feature(enable = "avx2")]
fn filter_vector(
    windows: [&[u8; WINDOW]; 7],
    low: &[__m256i; 6],
    high: &[__m256i; 6],
    shift: u32,
) -> __m128i {
    let round = _mm256_set1_epi32(1 << (shift - 1));
    let shift_count = _mm_cvtsi32_si128(shift as i32);
    let x = VVC_ALF_REACH;
    let centre = load(windows[3], x);
    let twice = _mm256_add_epi16(centre, centre);
    let sums: [__m256i; VVC_ALF_TAPS] = std::array::from_fn(|k| {
        let (dx, dy) = VVC_ALF_TAP_OFFSETS[k];
        let a = load(windows[(3 + dy) as usize], (x as isize + dx) as usize);
        let b = load(windows[(3 - dy) as usize], (x as isize - dx) as usize);
        _mm256_sub_epi16(_mm256_add_epi16(a, b), twice)
    });
    let mut sum_low = round;
    let mut sum_high = round;
    for p in 0..VVC_ALF_TAPS / 2 {
        let (a, b) = (sums[2 * p], sums[2 * p + 1]);
        sum_low = _mm256_add_epi32(
            sum_low,
            _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), low[p]),
        );
        sum_high = _mm256_add_epi32(
            sum_high,
            _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), high[p]),
        );
    }
    // Saturating to 16 bits keeps every out-of-range result out of
    // range after adding the centre, so the final clamp is exact.
    let offsets = _mm256_packs_epi32(
        _mm256_sra_epi32(sum_low, shift_count),
        _mm256_sra_epi32(sum_high, shift_count),
    );
    let filtered = _mm256_adds_epi16(centre, offsets);
    let bytes = _mm256_permute4x64_epi64::<0b1000>(_mm256_packus_epi16(filtered, filtered));
    _mm256_castsi256_si128(bytes)
}

/// AVX2 [`super::FilterBlockRowFn`]: sixteen columns at a time, their
/// coefficient pairs built once for the four rows.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn filter_block_row(
    rows: &[[&[u8]; 7]; VVC_ALF_BLOCK],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    shifts: &[u32; VVC_ALF_BLOCK],
    out: &mut [&mut [u8]; VVC_ALF_BLOCK],
) {
    let vectors = out[0].len() / LANES;
    let whole = vectors * LANES;
    let (block_groups, _) = coefficients.as_chunks::<{ LANES / VVC_ALF_BLOCK }>();
    for (i, blocks) in block_groups.iter().take(vectors).enumerate() {
        // Unpacking two taps' sums puts blocks 0 and 2 in the low unpack,
        // 1 and 3 in the high one.
        let low = coefficient_pairs(&blocks[0], &blocks[2]);
        let high = coefficient_pairs(&blocks[1], &blocks[3]);
        for ((rows, &shift), out) in rows.iter().zip(shifts).zip(out.iter_mut()) {
            // Each row's window once (one bounds check); taps index it at
            // constant offsets.
            let windows: [&[u8; WINDOW]; 7] = rows.map(|row| {
                row[i * LANES..i * LANES + WINDOW]
                    .try_into()
                    .expect("window")
            });
            let filtered = filter_vector(windows, &low, &high, shift);
            let out: &mut [u8; LANES] = (&mut out[i * LANES..(i + 1) * LANES])
                .try_into()
                .expect("sixteen samples");
            // SAFETY: AVX2 is enabled; the store writes exactly `out`.
            unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), filtered) };
        }
    }
    filter_block_row_from(rows, coefficients, shifts, out, whole);
}

/// AVX2 [`super::LaplacianFn`]: four groups (sixteen columns) per step.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn laplacians(rows: &[&[u8]; 4], out: &mut [[u16; 4]]) {
    // Per 128-bit half, after the two horizontal adds: [v0, v1, h0, h1,
    // d0_0, d0_1, d1_0, d1_1] (two groups) into [v0, h0, d0_0, d1_0, v1, ...].
    let to_groups = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7,
        10, 11, 14, 15,
    );
    let (steps, _) = out.as_chunks_mut::<4>();
    let whole = steps.len() * 4;
    for (step, out) in steps.iter_mut().enumerate() {
        // Index of the first column (4q − 2, offset by the rows' 3 lead).
        let x = 16 * step + 1;
        // Per offset (−1, 0, +1): rows above, at and below each column's
        // centre: rows 0..3 for even columns, 1..4 for odd ones.
        let pick = |row: usize, at: usize| {
            _mm256_blend_epi16::<ODD_LANES>(load(rows[row], at), load(rows[row + 1], at))
        };
        let [up, centre, down] = [0, 1, 2].map(|row| [x - 1, x, x + 1].map(|at| pick(row, at)));
        let twice = _mm256_add_epi16(centre[1], centre[1]);
        let laplacian = |a, b| _mm256_abs_epi16(_mm256_sub_epi16(twice, _mm256_add_epi16(a, b)));
        let vertical = laplacian(up[1], down[1]);
        let horizontal = laplacian(centre[0], centre[2]);
        let diagonal0 = laplacian(up[0], down[2]);
        let diagonal1 = laplacian(down[0], up[2]);
        let sums = _mm256_hadd_epi16(
            _mm256_hadd_epi16(vertical, horizontal),
            _mm256_hadd_epi16(diagonal0, diagonal1),
        );
        let groups = _mm256_shuffle_epi8(sums, to_groups);
        // SAFETY: AVX2 is enabled; the store writes exactly the 32 bytes of `out`.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), groups) };
    }
    laplacians_from(rows, out, whole);
}

/// Bytes of a row a full [`super::TapSumsFn`] call reads: two 64-byte
/// deinterleaving loads, 4 bytes apart.
const SAMPLED_WINDOW: usize = 4 * VVC_ALF_SAMPLES + 4;

/// The sixteen bytes at every fourth column of `bytes`, for each phase
/// 0..4: per 16-byte chunk, a shuffle groups its bytes by phase, then a 4x4
/// transpose of 32-bit groups gathers each phase across the chunks.
#[target_feature(enable = "avx2")]
fn deinterleave4(bytes: &[u8]) -> [__m128i; 4] {
    let bytes: &[u8; 4 * VVC_ALF_SAMPLES] =
        bytes[..4 * VVC_ALF_SAMPLES].try_into().expect("64 bytes");
    let by_phase = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    let load = |c: usize| {
        let bytes: &[u8; 16] = bytes[16 * c..16 * c + 16].try_into().expect("chunk");
        // SAFETY: AVX2 is enabled; the load reads exactly `bytes`.
        _mm_shuffle_epi8(unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }, by_phase)
    };
    let chunks = [load(0), load(1), load(2), load(3)];
    let low01 = _mm_unpacklo_epi32(chunks[0], chunks[1]);
    let high01 = _mm_unpackhi_epi32(chunks[0], chunks[1]);
    let low23 = _mm_unpacklo_epi32(chunks[2], chunks[3]);
    let high23 = _mm_unpackhi_epi32(chunks[2], chunks[3]);
    [
        _mm_unpacklo_epi64(low01, low23),
        _mm_unpackhi_epi64(low01, low23),
        _mm_unpacklo_epi64(high01, high23),
        _mm_unpackhi_epi64(high01, high23),
    ]
}

/// The columns `4i + dx` of one row for `dx` −3..=4 (index `dx + 3`),
/// widened to 16 bits.
#[target_feature(enable = "avx2")]
fn row_phases(row: &[u8]) -> [__m256i; 8] {
    let row: &[u8; SAMPLED_WINDOW] = row[..SAMPLED_WINDOW].try_into().expect("window");
    let (a, b) = (deinterleave4(row), deinterleave4(&row[4..]));
    [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]].map(|v| _mm256_cvtepu8_epi16(v))
}

/// Tap `K`'s sums from the rows' phases (constant indices, so only the
/// phases some tap reads are computed).
#[target_feature(enable = "avx2")]
fn tap<const K: usize>(phases: &[[__m256i; 8]; 7], twice: __m256i) -> __m256i {
    let (dx, dy) = VVC_ALF_TAP_OFFSETS[K];
    let at = |dy: isize, dx: isize| phases[(3 + dy) as usize][(dx + 3) as usize];
    _mm256_sub_epi16(_mm256_add_epi16(at(dy, dx), at(-dy, -dx)), twice)
}

/// Per 128-bit half, the 8x8 transpose of the 16-bit lanes of `rows`.
#[target_feature(enable = "avx2")]
fn transpose8(rows: [__m256i; 8]) -> [__m256i; 8] {
    let pairs: [__m256i; 8] = std::array::from_fn(|k| {
        let (first, second) = (rows[k / 2 * 2], rows[k / 2 * 2 + 1]);
        if k % 2 == 0 {
            _mm256_unpacklo_epi16(first, second)
        } else {
            _mm256_unpackhi_epi16(first, second)
        }
    });
    // Columns 0..2 then 2..4 of rows 0..4 (b[0], b[1]); 4..6, 6..8 (b[2],
    // b[3]); the same for rows 4..8.
    let quads = [
        _mm256_unpacklo_epi32(pairs[0], pairs[2]),
        _mm256_unpackhi_epi32(pairs[0], pairs[2]),
        _mm256_unpacklo_epi32(pairs[1], pairs[3]),
        _mm256_unpackhi_epi32(pairs[1], pairs[3]),
        _mm256_unpacklo_epi32(pairs[4], pairs[6]),
        _mm256_unpackhi_epi32(pairs[4], pairs[6]),
        _mm256_unpacklo_epi32(pairs[5], pairs[7]),
        _mm256_unpackhi_epi32(pairs[5], pairs[7]),
    ];
    std::array::from_fn(|c| {
        let (first, second) = (quads[c / 2], quads[4 + c / 2]);
        if c % 2 == 0 {
            _mm256_unpacklo_epi64(first, second)
        } else {
            _mm256_unpackhi_epi64(first, second)
        }
    })
}

/// Per transpose index, the byte shuffles taking a natural-order sample
/// to coefficient order: from the sample's low half (both 128-bit halves
/// hold it) and from its high half, zeroing the bytes the other supplies.
const SAMPLE_SHUFFLES: [[[u8; 32]; 2]; 4] = sample_shuffles();

const fn sample_shuffles() -> [[[u8; 32]; 2]; 4] {
    let mut shuffles = [[[0x80; 32]; 2]; 4];
    let mut t = 0;
    while t < 4 {
        let mut byte = 0;
        while byte < 32 {
            let source = SAMPLE_BYTE_SOURCES[t][byte];
            if source < 16 {
                shuffles[t][0][byte] = source;
            } else if source < 32 {
                shuffles[t][1][byte] = source - 16;
            }
            byte += 1;
        }
        t += 1;
    }
    shuffles
}

/// AVX2 [`super::SamplesFn`]: sixteen samples per call, from columns
/// deinterleaved by four, transposed to one sample per vector and put in
/// coefficient order by two byte shuffles; partial calls take the scalar
/// code.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn samples(
    rows: &[&[u8]; 7],
    source: &[u8],
    transposes: &[u8; VVC_ALF_SAMPLES],
    count: usize,
    out: &mut [[i16; VVC_ALF_SAMPLE_LANES]; VVC_ALF_SAMPLES],
) {
    if count < VVC_ALF_SAMPLES || source.len() < 4 * VVC_ALF_SAMPLES {
        super::samples(rows, source, transposes, count, out);
        return;
    }
    let phases = [
        row_phases(rows[0]),
        row_phases(rows[1]),
        row_phases(rows[2]),
        row_phases(rows[3]),
        row_phases(rows[4]),
        row_phases(rows[5]),
        row_phases(rows[6]),
    ];
    let centre = phases[3][3];
    let twice = _mm256_add_epi16(centre, centre);
    let error = _mm256_sub_epi16(centre, _mm256_cvtepu8_epi16(deinterleave4(source)[0]));
    let zero = _mm256_setzero_si256();
    let first = transpose8([
        tap::<0>(&phases, twice),
        tap::<1>(&phases, twice),
        tap::<2>(&phases, twice),
        tap::<3>(&phases, twice),
        tap::<4>(&phases, twice),
        tap::<5>(&phases, twice),
        tap::<6>(&phases, twice),
        tap::<7>(&phases, twice),
    ]);
    let second = transpose8([
        tap::<8>(&phases, twice),
        tap::<9>(&phases, twice),
        tap::<10>(&phases, twice),
        tap::<11>(&phases, twice),
        error,
        zero,
        zero,
        zero,
    ]);
    // SAFETY: AVX2 is enabled; each load reads one 32-byte shuffle.
    let shuffles: [[__m256i; 2]; 4] = SAMPLE_SHUFFLES
        .map(|pair| pair.map(|shuffle| unsafe { _mm256_loadu_si256(shuffle.as_ptr().cast()) }));
    // Sample c of each half: taps 0..8 from `first`, 8..16 from `second`;
    // `_mm256_permute2x128_si256` copies one 128-bit half into both.
    for (c, (first, second)) in first.into_iter().zip(second).enumerate() {
        for half in 0..2 {
            let i = 8 * half + c;
            let [low, high] = if half == 0 {
                [
                    _mm256_permute2x128_si256::<0x00>(first, first),
                    _mm256_permute2x128_si256::<0x00>(second, second),
                ]
            } else {
                [
                    _mm256_permute2x128_si256::<0x11>(first, first),
                    _mm256_permute2x128_si256::<0x11>(second, second),
                ]
            };
            let [from_low, from_high] = shuffles[usize::from(transposes[i] & 3)];
            let sample = _mm256_or_si256(
                _mm256_shuffle_epi8(low, from_low),
                _mm256_shuffle_epi8(high, from_high),
            );
            // SAFETY: AVX2 is enabled; the store writes exactly the 32 bytes of `out[i]`.
            unsafe { _mm256_storeu_si256(out[i].as_mut_ptr().cast(), sample) };
        }
    }
}

/// Row `I`'s pair of values `(a_I, b_I)` broadcast: `low` interleaves the
/// pair's lanes 0..4 and 8..12, `high` lanes 4..8 and 12..16.
#[target_feature(enable = "avx2")]
fn row_pair(i: usize, low: __m256i, high: __m256i) -> __m256i {
    let (source, dword) = match i {
        0..4 => (low, i),
        4..8 => (high, i - 4),
        8..12 => (low, i - 4),
        _ => (high, i - 8),
    };
    _mm256_permutevar8x32_epi32(source, _mm256_set1_epi32(dword as i32))
}

/// Gram rows `FIRST..FIRST + COUNT` over sample pairs: per pair and row,
/// two `vpmaddwd` add both samples' products for eight columns each.
#[target_feature(enable = "avx2")]
fn gram_rows<const FIRST: usize, const COUNT: usize>(
    pairs: &[[[i16; VVC_ALF_SAMPLE_LANES]; 2]],
    out: &mut [i32; VVC_ALF_DOT_LANES],
) {
    // `low` sums columns 0..4 and 8..12 of each row, `high` 4..8 and
    // 12..16 (the pairs' unpacks).
    let mut low = [_mm256_setzero_si256(); COUNT];
    let mut high = [_mm256_setzero_si256(); COUNT];
    for [a, b] in pairs {
        // SAFETY: AVX2 is enabled; each load reads one 32-byte sample.
        let (a, b) = unsafe {
            (
                _mm256_loadu_si256(a.as_ptr().cast()),
                _mm256_loadu_si256(b.as_ptr().cast()),
            )
        };
        let (pairs_low, pairs_high) = (_mm256_unpacklo_epi16(a, b), _mm256_unpackhi_epi16(a, b));
        for r in 0..COUNT {
            let row = row_pair(FIRST + r, pairs_low, pairs_high);
            low[r] = _mm256_add_epi32(low[r], _mm256_madd_epi16(pairs_low, row));
            high[r] = _mm256_add_epi32(high[r], _mm256_madd_epi16(pairs_high, row));
        }
    }
    for (r, (low, high)) in low.into_iter().zip(high).enumerate() {
        let mut values = [[0i32; 8]; 2];
        // SAFETY: AVX2 is enabled; each store writes exactly 32 bytes of `values`.
        unsafe {
            _mm256_storeu_si256(values[0].as_mut_ptr().cast(), low);
            _mm256_storeu_si256(values[1].as_mut_ptr().cast(), high);
        }
        let mut row = [0; VVC_ALF_GRAM_COLUMNS];
        row[..4].copy_from_slice(&values[0][..4]);
        row[4..8].copy_from_slice(&values[1][..4]);
        row[8..12].copy_from_slice(&values[0][4..]);
        row[12] = values[1][4];
        let i = FIRST + r;
        let start = vvc_alf_gram_index(i, i);
        out[start..start + VVC_ALF_GRAM_COLUMNS - i].copy_from_slice(&row[i..]);
    }
}

/// AVX2 [`super::GramFn`]: rows in three groups whose accumulators fit the
/// registers; each 32-bit lane sums at most 2^12 products of at most 2^18.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn gram(
    samples: &[[i16; VVC_ALF_SAMPLE_LANES]],
    out: &mut [i32; VVC_ALF_DOT_LANES],
) {
    out[VVC_ALF_GRAM_TERMS..].fill(0);
    let (pairs, _) = samples.as_chunks::<VVC_ALF_GRAM_ALIGN>();
    gram_rows::<0, 5>(pairs, out);
    gram_rows::<5, 5>(pairs, out);
    gram_rows::<10, 3>(pairs, out);
}

/// AVX2 [`super::DotFn`]: `vpmuldq` on the even 32-bit lanes and, shifted
/// down, the odd ones; four 64-bit sums.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn dot(a: &[i32; VVC_ALF_DOT_LANES], b: &[i32; VVC_ALF_DOT_LANES]) -> i64 {
    let (a, _) = a.as_chunks::<8>();
    let (b, _) = b.as_chunks::<8>();
    let mut sum = _mm256_setzero_si256();
    for (a, b) in a.iter().zip(b) {
        // SAFETY: AVX2 is enabled; each load reads one 32-byte chunk.
        let (a, b) = unsafe {
            (
                _mm256_loadu_si256(a.as_ptr().cast()),
                _mm256_loadu_si256(b.as_ptr().cast()),
            )
        };
        let even = _mm256_mul_epi32(a, b);
        let odd = _mm256_mul_epi32(_mm256_srli_epi64::<32>(a), _mm256_srli_epi64::<32>(b));
        sum = _mm256_add_epi64(sum, _mm256_add_epi64(even, odd));
    }
    let mut lanes = [0i64; 4];
    // SAFETY: AVX2 is enabled; the store writes exactly the 32 bytes of `lanes`.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sum) };
    lanes
        .iter()
        .fold(0, |total: i64, &lane| total.wrapping_add(lane))
}

/// Four Laplacian groups (32 bytes) from `groups[at]`.
#[target_feature(enable = "avx2")]
fn load_groups(groups: &[[u16; 4]], at: usize) -> __m256i {
    let groups: &[[u16; 4]; 4] = groups[at..at + 4].try_into().expect("four groups");
    // SAFETY: AVX2 is enabled; the load reads exactly the 32 bytes of `groups`.
    unsafe { _mm256_loadu_si256(groups.as_ptr().cast()) }
}

/// `table[index]` per 32-bit lane (indexes 0..16; `table[0]` must be 0,
/// which the lanes' upper bytes look up).
#[target_feature(enable = "avx2")]
fn lookup(table: &[u8; 16], index: __m256i) -> __m256i {
    // SAFETY: AVX2 is enabled; the load reads exactly the 16 bytes of `table`.
    let table = unsafe { _mm_loadu_si128(table.as_ptr().cast()) };
    _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(table), index)
}

/// AVX2 [`super::ClassifyFn`]: eight blocks per step in 32-bit lanes; the
/// groups' interleaved sums are separated by a byte shuffle, a dword
/// permute and 64-bit unpacks.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn classify(groups: &[[u16; 4]], boundary: bool, out: &mut [[u8; 2]]) {
    let scale = _mm256_set1_epi32(if boundary {
        BOUNDARY_ACTIVITY_SCALE
    } else {
        ACTIVITY_SCALE
    } as i32);
    // Per 128-bit half: words [v0 h0 a0 c0 v1 h1 a1 c1] to [v0 v1 h0 h1 ...].
    let pair_words = _mm256_setr_epi8(
        0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12,
        13, 6, 7, 14, 15,
    );
    let quad_dwords = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let mut activity_class = [0u8; 16];
    activity_class.copy_from_slice(&ACTIVITY_CLASS);
    let mut transpose = [0u8; 16];
    transpose[..8].copy_from_slice(&TRANSPOSE);
    let (steps, _) = out.as_chunks_mut::<8>();
    let whole = steps.len() * 8;
    for (step, out) in steps.iter_mut().enumerate() {
        let first = 8 * step;
        // Block sums of blocks first..first + 4 and the next four, as
        // [v0..v4, h0..h4 | a0..a4, c0..c4] after the shuffles.
        let quads = [first, first + 4].map(|at| {
            let sums = _mm256_add_epi16(load_groups(groups, at), load_groups(groups, at + 1));
            _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(sums, pair_words), quad_dwords)
        });
        // Vertical with first diagonal, horizontal with second diagonal.
        let vertical_d0 = _mm256_unpacklo_epi64(quads[0], quads[1]);
        let horizontal_d1 = _mm256_unpackhi_epi64(quads[0], quads[1]);
        let widen = |v: __m256i, high: bool| {
            _mm256_cvtepu16_epi32(if high {
                _mm256_extracti128_si256::<1>(v)
            } else {
                _mm256_castsi256_si128(v)
            })
        };
        let (ver, hor, dg0, dg1) = (
            widen(vertical_d0, false),
            widen(horizontal_d1, false),
            widen(vertical_d0, true),
            widen(horizontal_d1, true),
        );
        let activity = _mm256_min_epu32(
            _mm256_srli_epi32::<{ ACTIVITY_SHIFT as i32 }>(_mm256_mullo_epi32(
                _mm256_add_epi32(ver, hor),
                scale,
            )),
            _mm256_set1_epi32(MAX_ACTIVITY as i32),
        );
        let vertical = _mm256_cmpgt_epi32(ver, hor);
        let (hv1, hv0) = (_mm256_max_epi32(ver, hor), _mm256_min_epi32(ver, hor));
        let hv_direction = _mm256_blendv_epi8(_mm256_set1_epi32(3), _mm256_set1_epi32(1), vertical);
        let first_diagonal = _mm256_cmpgt_epi32(dg0, dg1);
        let (d1, d0) = (_mm256_max_epi32(dg0, dg1), _mm256_min_epi32(dg0, dg1));
        let d_direction =
            _mm256_blendv_epi8(_mm256_set1_epi32(2), _mm256_setzero_si256(), first_diagonal);
        let diagonal = _mm256_cmpgt_epi32(_mm256_mullo_epi32(d1, hv0), _mm256_mullo_epi32(hv1, d0));
        let strong = _mm256_blendv_epi8(hv1, d1, diagonal);
        let weak = _mm256_blendv_epi8(hv0, d0, diagonal);
        let main = _mm256_blendv_epi8(hv_direction, d_direction, diagonal);
        let secondary = _mm256_blendv_epi8(d_direction, hv_direction, diagonal);
        // Strength 1 (strong > 2 weak) or 2 (2 strong > 9 weak, which
        // implies 1): minus the masks.
        let one = _mm256_cmpgt_epi32(strong, _mm256_add_epi32(weak, weak));
        let two = _mm256_cmpgt_epi32(
            _mm256_add_epi32(strong, strong),
            _mm256_add_epi32(_mm256_slli_epi32::<3>(weak), weak),
        );
        let strength = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_setzero_si256(), one), two);
        let step_classes = _mm256_add_epi32(
            _mm256_slli_epi32::<1>(_mm256_and_si256(main, _mm256_set1_epi32(1))),
            strength,
        );
        let directional = _mm256_and_si256(
            _mm256_mullo_epi32(step_classes, _mm256_set1_epi32(i32::from(ACTIVITY_CLASSES))),
            one,
        );
        let index = _mm256_add_epi32(lookup(&activity_class, activity), directional);
        let transposes = lookup(
            &transpose,
            _mm256_add_epi32(
                _mm256_slli_epi32::<1>(main),
                _mm256_srli_epi32::<1>(secondary),
            ),
        );
        let pairs = _mm256_or_si256(index, _mm256_slli_epi32::<8>(transposes));
        let packed = _mm256_permute4x64_epi64::<0b1000>(_mm256_packus_epi32(pairs, pairs));
        // SAFETY: AVX2 is enabled; the store writes exactly the 16 bytes of `out`.
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), _mm256_castsi256_si128(packed)) };
    }
    classify_from(groups, boundary, out, whole);
}
