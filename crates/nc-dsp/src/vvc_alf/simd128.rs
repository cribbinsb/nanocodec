//! SIMD128 ALF kernels: eight samples per vector in 16-bit lanes.
//!
//! The filter interleaves two taps' sums so that each half of the vector
//! holds one 4x4 block's four samples, then multiplies by the block's
//! coefficient pair with `i32x4.dot_i16x8_s`. The Laplacians select the
//! checkerboard's rows per lane and reduce groups of four columns with
//! even/odd shuffles.

use super::{
    ACTIVITY_CLASS, ACTIVITY_CLASSES, ACTIVITY_SCALE, ACTIVITY_SHIFT, BOUNDARY_ACTIVITY_SCALE,
    MAX_ACTIVITY, SAMPLE_BYTE_SOURCES, TRANSPOSE, VVC_ALF_BLOCK, VVC_ALF_DOT_LANES,
    VVC_ALF_GRAM_ALIGN, VVC_ALF_GRAM_COLUMNS, VVC_ALF_GRAM_TERMS, VVC_ALF_REACH,
    VVC_ALF_SAMPLE_LANES, VVC_ALF_SAMPLES, VVC_ALF_TAP_OFFSETS, VVC_ALF_TAPS, classify_from,
    filter_block_row_from, laplacians_from, vvc_alf_gram_index,
};
use core::arch::wasm32::{
    i8x16_swizzle, i16x8_abs, i16x8_add, i16x8_add_sat, i16x8_narrow_i32x4, i16x8_shuffle,
    i16x8_sub, i32x4_add, i32x4_dot_i16x8, i32x4_extract_lane, i32x4_mul, i32x4_shl, i32x4_shr,
    i32x4_shuffle, i32x4_splat, i32x4_sub, i64x2_add, i64x2_extmul_high_i32x4,
    i64x2_extmul_low_i32x4, i64x2_extract_lane, i64x2_splat, u8x16_narrow_i16x8, u8x16_shuffle,
    u16x8_extend_high_u8x16, u16x8_extend_low_u8x16, u16x8_load_extend_u8x8, u16x8_narrow_i32x4,
    u32x4_extend_high_u16x8, u32x4_extend_low_u16x8, u32x4_gt, u32x4_max, u32x4_min, u32x4_shr,
    u32x4_splat, u64x2_shuffle, v128, v128_and, v128_bitselect, v128_load, v128_or, v128_store,
    v128_store64_lane,
};

/// Samples per vector.
const LANES: usize = 8;
/// Samples one vector of filter output reads from each row.
const WINDOW: usize = LANES + 2 * VVC_ALF_REACH;

/// Eight samples of `row` from `at`, widened to 16 bits.
#[target_feature(enable = "simd128")]
fn load(row: &[u8], at: usize) -> v128 {
    let samples: &[u8; LANES] = row[at..at + LANES].try_into().expect("eight samples");
    // SAFETY: SIMD128 is enabled; the load reads exactly `samples`.
    unsafe { u16x8_load_extend_u8x8(samples.as_ptr()) }
}

/// Coefficients `k` and `k + 1` as one 32-bit pair, broadcast.
#[target_feature(enable = "simd128")]
fn pair(c: &[i16; VVC_ALF_TAPS], k: usize) -> v128 {
    i32x4_splat(i32::from(c[k] as u16) | i32::from(c[k + 1]) << 16)
}

/// A block's six coefficient pairs.
#[target_feature(enable = "simd128")]
fn pairs(c: &[i16; VVC_ALF_TAPS]) -> [v128; VVC_ALF_TAPS / 2] {
    std::array::from_fn(|p| pair(c, 2 * p))
}

/// One row's eight filtered samples (in the low half) from its `windows`
/// (each row's samples from 3 columns left of the first output), with the
/// coefficient pairs of its two blocks.
#[target_feature(enable = "simd128")]
fn filter_vector(
    windows: [&[u8; WINDOW]; 7],
    blocks: &[[v128; VVC_ALF_TAPS / 2]; 2],
    shift: u32,
) -> v128 {
    let round = i32x4_splat(1 << (shift - 1));
    let x = VVC_ALF_REACH;
    let centre = load(windows[3], x);
    let twice = i16x8_add(centre, centre);
    let sums: [v128; VVC_ALF_TAPS] = std::array::from_fn(|k| {
        let (dx, dy) = VVC_ALF_TAP_OFFSETS[k];
        let a = load(windows[(3 + dy) as usize], (x as isize + dx) as usize);
        let b = load(windows[(3 - dy) as usize], (x as isize - dx) as usize);
        i16x8_sub(i16x8_add(a, b), twice)
    });
    let mut low = round;
    let mut high = round;
    for p in 0..VVC_ALF_TAPS / 2 {
        let (a, b) = (sums[2 * p], sums[2 * p + 1]);
        // Block 0's samples interleaved, then block 1's.
        let first = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b);
        let second = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b);
        low = i32x4_add(low, i32x4_dot_i16x8(first, blocks[0][p]));
        high = i32x4_add(high, i32x4_dot_i16x8(second, blocks[1][p]));
    }
    // Saturating to 16 bits keeps every out-of-range result out of
    // range after adding the centre, so the final clamp is exact.
    let offsets = i16x8_narrow_i32x4(i32x4_shr(low, shift), i32x4_shr(high, shift));
    let filtered = i16x8_add_sat(centre, offsets);
    u8x16_narrow_i16x8(filtered, filtered)
}

/// SIMD128 [`super::FilterBlockRowFn`]: eight columns at a time, their
/// coefficient pairs built once for the four rows.
#[target_feature(enable = "simd128")]
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
        let blocks = [pairs(&blocks[0]), pairs(&blocks[1])];
        for ((rows, &shift), out) in rows.iter().zip(shifts).zip(out.iter_mut()) {
            // Each row's window once (one bounds check); taps index it at
            // constant offsets.
            let windows: [&[u8; WINDOW]; 7] = rows.map(|row| {
                row[i * LANES..i * LANES + WINDOW]
                    .try_into()
                    .expect("window")
            });
            let bytes = filter_vector(windows, &blocks, shift);
            let out: &mut [u8; LANES] = (&mut out[i * LANES..(i + 1) * LANES])
                .try_into()
                .expect("eight samples");
            // SAFETY: SIMD128 is enabled; the store writes exactly `out`.
            unsafe { v128_store64_lane::<0>(bytes, out.as_mut_ptr().cast()) };
        }
    }
    filter_block_row_from(rows, coefficients, shifts, out, whole);
}

/// Lane-wise sums of the even and odd lanes of `a` then `b`: `[a0 + a1,
/// a2 + a3, ..., b6 + b7]`.
#[target_feature(enable = "simd128")]
fn pair_sums(a: v128, b: v128) -> v128 {
    i16x8_add(
        i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(a, b),
        i16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(a, b),
    )
}

/// SIMD128 [`super::LaplacianFn`]: two groups (eight columns) per step.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn laplacians(rows: &[&[u8]; 4], out: &mut [[u16; 4]]) {
    let (steps, _) = out.as_chunks_mut::<2>();
    let whole = steps.len() * 2;
    for (step, out) in steps.iter_mut().enumerate() {
        let x = 8 * step + 1;
        // Even columns are centred on row 1, odd ones on row 2.
        let pick = |row: usize, at: usize| {
            i16x8_shuffle::<0, 9, 2, 11, 4, 13, 6, 15>(load(rows[row], at), load(rows[row + 1], at))
        };
        let [up, centre, down] = [0, 1, 2].map(|row| [x - 1, x, x + 1].map(|at| pick(row, at)));
        let twice = i16x8_add(centre[1], centre[1]);
        let laplacian = |a, b| i16x8_abs(i16x8_sub(twice, i16x8_add(a, b)));
        let straight = pair_sums(laplacian(up[1], down[1]), laplacian(centre[0], centre[2]));
        let diagonal = pair_sums(laplacian(up[0], down[2]), laplacian(down[0], up[2]));
        // [v0, v1, h0, h1, d0_0, d0_1, d1_0, d1_1] into [v0, h0, d0_0, d1_0, v1, ...].
        let sums = pair_sums(straight, diagonal);
        let groups = i16x8_shuffle::<0, 2, 4, 6, 1, 3, 5, 7>(sums, sums);
        // SAFETY: SIMD128 is enabled; the store writes exactly the 16 bytes of `out`.
        unsafe { v128_store(out.as_mut_ptr().cast(), groups) };
    }
    laplacians_from(rows, out, whole);
}

/// Bytes of a row a full [`super::TapSumsFn`] call reads: two 64-byte
/// deinterleaving loads, 4 bytes apart.
const SAMPLED_WINDOW: usize = 4 * VVC_ALF_SAMPLES + 4;

/// The sixteen bytes at every fourth column of 64-byte `chunks`, for each
/// phase 0..4: per phase, two shuffles take eight bytes from each 32-byte
/// half and a third joins them.
#[target_feature(enable = "simd128")]
fn deinterleave4(chunks: &[v128; 4]) -> [v128; 4] {
    let join = |low, high| u64x2_shuffle::<0, 2>(low, high);
    macro_rules! phase {
        ($p0:literal, $p1:literal, $p2:literal, $p3:literal, $p4:literal, $p5:literal, $p6:literal, $p7:literal) => {{
            let pick = |a, b| {
                u8x16_shuffle::<$p0, $p1, $p2, $p3, $p4, $p5, $p6, $p7, 0, 0, 0, 0, 0, 0, 0, 0>(
                    a, b,
                )
            };
            join(pick(chunks[0], chunks[1]), pick(chunks[2], chunks[3]))
        }};
    }
    [
        phase!(0, 4, 8, 12, 16, 20, 24, 28),
        phase!(1, 5, 9, 13, 17, 21, 25, 29),
        phase!(2, 6, 10, 14, 18, 22, 26, 30),
        phase!(3, 7, 11, 15, 19, 23, 27, 31),
    ]
}

/// The columns `4i + dx` of one row for `dx` −3..=4 (index `dx + 3`).
#[target_feature(enable = "simd128")]
fn row_phases(row: &[u8]) -> [v128; 8] {
    let row: &[u8; SAMPLED_WINDOW] = row[..SAMPLED_WINDOW].try_into().expect("window");
    let chunks = |base: usize| -> [v128; 4] {
        std::array::from_fn(|c| {
            let bytes: &[u8; 16] = row[base + 16 * c..base + 16 * c + 16]
                .try_into()
                .expect("chunk");
            // SAFETY: SIMD128 is enabled; the load reads exactly `bytes`.
            unsafe { v128_load(bytes.as_ptr().cast()) }
        })
    };
    let (a, b) = (deinterleave4(&chunks(0)), deinterleave4(&chunks(4)));
    [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
}

/// Tap `K`'s sums from the rows' phases, samples 0..8 and 8..16.
#[target_feature(enable = "simd128")]
fn tap<const K: usize>(phases: &[[v128; 8]; 7]) -> [v128; 2] {
    let (dx, dy) = VVC_ALF_TAP_OFFSETS[K];
    let at = |dy: isize, dx: isize| phases[(3 + dy) as usize][(dx + 3) as usize];
    let halves = |v| [u16x8_extend_low_u8x16(v), u16x8_extend_high_u8x16(v)];
    let (a, b, centre) = (halves(at(dy, dx)), halves(at(-dy, -dx)), halves(at(0, 0)));
    [0, 1].map(|h| i16x8_sub(i16x8_add(a[h], b[h]), i16x8_add(centre[h], centre[h])))
}

/// The 8x8 transpose of 16-bit lanes.
#[target_feature(enable = "simd128")]
fn transpose8(rows: [v128; 8]) -> [v128; 8] {
    let pairs: [v128; 8] = std::array::from_fn(|k| {
        let (first, second) = (rows[k / 2 * 2], rows[k / 2 * 2 + 1]);
        if k % 2 == 0 {
            i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(first, second)
        } else {
            i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(first, second)
        }
    });
    let low = |first, second| i32x4_shuffle::<0, 4, 1, 5>(first, second);
    let high = |first, second| i32x4_shuffle::<2, 6, 3, 7>(first, second);
    // Columns 0..2, 2..4, 4..6, 6..8 of rows 0..4, then of rows 4..8.
    let quads = [
        low(pairs[0], pairs[2]),
        high(pairs[0], pairs[2]),
        low(pairs[1], pairs[3]),
        high(pairs[1], pairs[3]),
        low(pairs[4], pairs[6]),
        high(pairs[4], pairs[6]),
        low(pairs[5], pairs[7]),
        high(pairs[5], pairs[7]),
    ];
    std::array::from_fn(|c| {
        let (first, second) = (quads[c / 2], quads[4 + c / 2]);
        if c % 2 == 0 {
            u64x2_shuffle::<0, 2>(first, second)
        } else {
            u64x2_shuffle::<1, 3>(first, second)
        }
    })
}

/// SIMD128 [`super::SamplesFn`]: sixteen samples per call, from columns
/// deinterleaved by four, transposed to one sample per pair of vectors and
/// put in coefficient order by swizzles; partial calls take the scalar
/// code.
#[target_feature(enable = "simd128")]
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
    let original: [v128; 4] = std::array::from_fn(|c| {
        let bytes: &[u8; 16] = source[16 * c..16 * c + 16].try_into().expect("chunk");
        // SAFETY: SIMD128 is enabled; the load reads exactly `bytes`.
        unsafe { v128_load(bytes.as_ptr().cast()) }
    });
    let original = deinterleave4(&original)[0];
    let centre = phases[3][3];
    let error = [
        i16x8_sub(
            u16x8_extend_low_u8x16(centre),
            u16x8_extend_low_u8x16(original),
        ),
        i16x8_sub(
            u16x8_extend_high_u8x16(centre),
            u16x8_extend_high_u8x16(original),
        ),
    ];
    let taps = [
        tap::<0>(&phases),
        tap::<1>(&phases),
        tap::<2>(&phases),
        tap::<3>(&phases),
        tap::<4>(&phases),
        tap::<5>(&phases),
        tap::<6>(&phases),
        tap::<7>(&phases),
        tap::<8>(&phases),
        tap::<9>(&phases),
        tap::<10>(&phases),
        tap::<11>(&phases),
    ];
    // Per transpose: for each output half, swizzles from the sample's low
    // and high halves (indexes past 15 give zero).
    let tables: [[[v128; 2]; 2]; 4] = SAMPLE_BYTE_SOURCES.map(|sources| {
        let split = |half: usize, from_high: bool| {
            let bytes: [u8; 16] = std::array::from_fn(|b| match sources[16 * half + b] {
                s if s < 16 && !from_high => s,
                s if (16..32).contains(&s) && from_high => s - 16,
                _ => 255,
            });
            // SAFETY: SIMD128 is enabled; the load reads exactly `bytes`.
            unsafe { v128_load(bytes.as_ptr().cast()) }
        };
        [
            [split(0, false), split(0, true)],
            [split(1, false), split(1, true)],
        ]
    });
    let zero = i32x4_splat(0);
    for half in 0..2 {
        let first = transpose8(std::array::from_fn(|k| taps[k][half]));
        let second = transpose8(std::array::from_fn(|k| match k {
            0..4 => taps[8 + k][half],
            4 => error[half],
            _ => zero,
        }));
        for (c, (first, second)) in first.into_iter().zip(second).enumerate() {
            let i = 8 * half + c;
            let table = &tables[usize::from(transposes[i] & 3)];
            for (h, [from_low, from_high]) in table.iter().enumerate() {
                let bytes = v128_or(
                    i8x16_swizzle(first, *from_low),
                    i8x16_swizzle(second, *from_high),
                );
                // SAFETY: SIMD128 is enabled; the store writes eight values of `out[i]`.
                unsafe { v128_store(out[i][8 * h..].as_mut_ptr().cast(), bytes) };
            }
        }
    }
}

/// Stores accumulated Gram rows `first..` (quads of columns from each
/// row's own) into the packed upper triangle.
fn store_rows(sums: &[[v128; 4]], first: usize, out: &mut [i32; VVC_ALF_DOT_LANES]) {
    for (r, sums) in sums.iter().enumerate() {
        let i = first + r;
        for (q, sum) in sums.iter().enumerate().skip(i / 4) {
            let values = [
                i32x4_extract_lane::<0>(*sum),
                i32x4_extract_lane::<1>(*sum),
                i32x4_extract_lane::<2>(*sum),
                i32x4_extract_lane::<3>(*sum),
            ];
            for (j, value) in (4 * q..)
                .zip(values)
                .filter(|&(j, _)| (i..VVC_ALF_GRAM_COLUMNS).contains(&j))
            {
                out[vvc_alf_gram_index(i, j)] = value;
            }
        }
    }
}

/// A function accumulating the listed Gram rows over sample pairs, upper
/// triangle by quads of columns: each row `(lane, first quad)` broadcasts
/// its pair from quad `lane / 4` and adds `i32x4.dot_i16x8_s` with each
/// quad from its own (constant lanes, so the rows unroll).
macro_rules! gram_rows {
    ($name:ident, $first:literal, [$(($quad:literal, $dword:literal)),+ $(,)?]) => {
        #[target_feature(enable = "simd128")]
        fn $name(
            pairs: &[[[i16; VVC_ALF_SAMPLE_LANES]; 2]],
            out: &mut [i32; VVC_ALF_DOT_LANES],
        ) {
            const ROWS: usize = [$($dword),+].len();
            let mut sums = [[i32x4_splat(0); 4]; ROWS];
            for [a, b] in pairs {
                let load = |values: &[i16]| {
                    let values: &[i16; 8] = values.try_into().expect("eight values");
                    // SAFETY: SIMD128 is enabled; the load reads exactly `values`.
                    unsafe { v128_load(values.as_ptr().cast()) }
                };
                let (a_low, a_high, b_low, b_high) =
                    (load(&a[..8]), load(&a[8..]), load(&b[..8]), load(&b[8..]));
                // Quad q interleaves both samples' columns 4q..4q + 4.
                let quads = [
                    i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a_low, b_low),
                    i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a_low, b_low),
                    i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a_high, b_high),
                    i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a_high, b_high),
                ];
                let mut row = 0;
                $(
                    let pair = i32x4_shuffle::<$dword, $dword, $dword, $dword>(
                        quads[$quad],
                        quads[$quad],
                    );
                    for q in $quad..4 {
                        sums[row][q] = i32x4_add(sums[row][q], i32x4_dot_i16x8(quads[q], pair));
                    }
                    row += 1;
                )+
                debug_assert_eq!(row, ROWS);
            }
            store_rows(&sums, $first, out);
        }
    };
}

// Rows 0..6 and 6..13: (quad of the row's own column, its dword there).
gram_rows!(
    gram_rows_0_6,
    0,
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
);
gram_rows!(
    gram_rows_6_13,
    6,
    [(1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0)]
);

/// SIMD128 [`super::GramFn`]: rows in two groups; each 32-bit lane sums at
/// most 2^12 products of at most 2^18.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn gram(
    samples: &[[i16; VVC_ALF_SAMPLE_LANES]],
    out: &mut [i32; VVC_ALF_DOT_LANES],
) {
    out[VVC_ALF_GRAM_TERMS..].fill(0);
    let (pairs, _) = samples.as_chunks::<VVC_ALF_GRAM_ALIGN>();
    gram_rows_0_6(pairs, out);
    gram_rows_6_13(pairs, out);
}

/// SIMD128 [`super::DotFn`]: extended multiplies into two 64-bit lanes.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn dot(a: &[i32; VVC_ALF_DOT_LANES], b: &[i32; VVC_ALF_DOT_LANES]) -> i64 {
    let (a, _) = a.as_chunks::<4>();
    let (b, _) = b.as_chunks::<4>();
    let mut sum = i64x2_splat(0);
    for (a, b) in a.iter().zip(b) {
        // SAFETY: SIMD128 is enabled; each load reads one 16-byte chunk.
        let (a, b) = unsafe { (v128_load(a.as_ptr().cast()), v128_load(b.as_ptr().cast())) };
        sum = i64x2_add(sum, i64x2_extmul_low_i32x4(a, b));
        sum = i64x2_add(sum, i64x2_extmul_high_i32x4(a, b));
    }
    i64x2_extract_lane::<0>(sum).wrapping_add(i64x2_extract_lane::<1>(sum))
}

/// `table[index]` per 32-bit lane (indexes 0..16; `table[0]` must be 0,
/// which the lanes' upper bytes look up).
#[target_feature(enable = "simd128")]
fn lookup(table: v128, index: v128) -> v128 {
    i8x16_swizzle(table, index)
}

/// SIMD128 [`super::ClassifyFn`]: four blocks per step in 32-bit lanes;
/// shuffles separate the groups' interleaved sums.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn classify(groups: &[[u16; 4]], boundary: bool, out: &mut [[u8; 2]]) {
    let scale = u32x4_splat(if boundary {
        BOUNDARY_ACTIVITY_SCALE
    } else {
        ACTIVITY_SCALE
    });
    let mut transpose = [0u8; 16];
    transpose[..8].copy_from_slice(&TRANSPOSE);
    // SAFETY: SIMD128 is enabled; each load reads exactly 16 table bytes.
    let (activity_class, transpose) = unsafe {
        (
            v128_load(ACTIVITY_CLASS.as_ptr().cast()),
            v128_load(transpose.as_ptr().cast()),
        )
    };
    let load = |at: usize| {
        let groups: &[[u16; 4]; 2] = groups[at..at + 2].try_into().expect("two groups");
        // SAFETY: SIMD128 is enabled; the load reads exactly `groups`.
        unsafe { v128_load(groups.as_ptr().cast()) }
    };
    let (steps, _) = out.as_chunks_mut::<4>();
    let whole = steps.len() * 4;
    for (step, out) in steps.iter_mut().enumerate() {
        let first = 4 * step;
        // Blocks first..first + 2 and the next two, as [v h a c] words.
        let pairs = [first, first + 2].map(|at| i16x8_add(load(at), load(at + 1)));
        let straight = i16x8_shuffle::<0, 4, 8, 12, 1, 5, 9, 13>(pairs[0], pairs[1]);
        let diagonals = i16x8_shuffle::<2, 6, 10, 14, 3, 7, 11, 15>(pairs[0], pairs[1]);
        let (ver, hor) = (
            u32x4_extend_low_u16x8(straight),
            u32x4_extend_high_u16x8(straight),
        );
        let (dg0, dg1) = (
            u32x4_extend_low_u16x8(diagonals),
            u32x4_extend_high_u16x8(diagonals),
        );
        let activity = u32x4_min(
            u32x4_shr(i32x4_mul(i32x4_add(ver, hor), scale), ACTIVITY_SHIFT),
            u32x4_splat(MAX_ACTIVITY),
        );
        let vertical = u32x4_gt(ver, hor);
        let (hv1, hv0) = (u32x4_max(ver, hor), u32x4_min(ver, hor));
        let hv_direction = v128_bitselect(u32x4_splat(1), u32x4_splat(3), vertical);
        let first_diagonal = u32x4_gt(dg0, dg1);
        let (d1, d0) = (u32x4_max(dg0, dg1), u32x4_min(dg0, dg1));
        let d_direction = v128_bitselect(u32x4_splat(0), u32x4_splat(2), first_diagonal);
        let diagonal = u32x4_gt(i32x4_mul(d1, hv0), i32x4_mul(hv1, d0));
        let strong = v128_bitselect(d1, hv1, diagonal);
        let weak = v128_bitselect(d0, hv0, diagonal);
        let main = v128_bitselect(d_direction, hv_direction, diagonal);
        let secondary = v128_bitselect(hv_direction, d_direction, diagonal);
        // Strength 1 (strong > 2 weak) or 2 (2 strong > 9 weak, which
        // implies 1): minus the masks.
        let one = u32x4_gt(strong, i32x4_add(weak, weak));
        let two = u32x4_gt(i32x4_add(strong, strong), i32x4_mul(weak, u32x4_splat(9)));
        let strength = i32x4_sub(i32x4_sub(u32x4_splat(0), one), two);
        let step_classes = i32x4_add(i32x4_shl(v128_and(main, u32x4_splat(1)), 1), strength);
        let directional = v128_and(
            i32x4_mul(step_classes, u32x4_splat(u32::from(ACTIVITY_CLASSES))),
            one,
        );
        let index = i32x4_add(lookup(activity_class, activity), directional);
        let transposes = lookup(
            transpose,
            i32x4_add(i32x4_shl(main, 1), u32x4_shr(secondary, 1)),
        );
        let pairs = v128_or(index, i32x4_shl(transposes, 8));
        let packed = u16x8_narrow_i32x4(pairs, pairs);
        // SAFETY: SIMD128 is enabled; the store writes exactly the 8 bytes of `out`.
        unsafe { v128_store64_lane::<0>(packed, out.as_mut_ptr().cast()) };
    }
    classify_from(groups, boundary, out, whole);
}
