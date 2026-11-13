//! NEON ALF kernels: sixteen samples per step.
//!
//! The filter keeps one 32-bit accumulator per 4x4 block (four samples)
//! and multiplies by coefficient lanes (`smlal` by element), so it needs no
//! coefficient vectors. The Laplacians select the checkerboard's rows per
//! byte, take `|2c − (a + b)|` as an unsigned absolute difference, reduce
//! with pairwise adds and store the groups interleaved (`st4`). The Gram
//! matrix transposes eight samples at a time to one vector per value.

use super::{
    ACTIVITY_CLASS, ACTIVITY_CLASSES, ACTIVITY_SCALE, ACTIVITY_SHIFT, BOUNDARY_ACTIVITY_SCALE,
    MAX_ACTIVITY, SAMPLE_BYTE_SOURCES, TRANSPOSE, VVC_ALF_BLOCK, VVC_ALF_DOT_LANES,
    VVC_ALF_GRAM_COLUMNS, VVC_ALF_REACH, VVC_ALF_SAMPLE_LANES, VVC_ALF_SAMPLES,
    VVC_ALF_TAP_OFFSETS, VVC_ALF_TAPS, classify_from, filter_block_row_from, laplacians_from,
};
use core::arch::aarch64::{
    int16x4_t, int16x8_t, int32x4_t, uint8x16_t, uint8x16x2_t, uint16x4_t, uint16x4x4_t,
    uint16x8_t, uint32x4_t, vabdq_u16, vaddl_high_u8, vaddl_u8, vaddq_u16, vaddq_u32, vaddvq_s64,
    vandq_u32, vbslq_u8, vbslq_u32, vcgtq_u32, vcombine_s16, vcombine_u8, vcombine_u16,
    vdupq_n_s32, vdupq_n_s64, vdupq_n_u16, vdupq_n_u32, vget_high_s16, vget_high_u16, vget_low_s16,
    vget_low_s32, vget_low_u8, vget_low_u16, vld1_s16, vld1q_s16, vld1q_s32, vld1q_u8, vld4q_u8,
    vld4q_u16, vmaxq_u32, vminq_u32, vmlal_high_s16, vmlal_high_s32, vmlal_lane_s16,
    vmlal_laneq_s16, vmlal_s16, vmlal_s32, vmovl_high_u8, vmovl_high_u16, vmovl_u8, vmovl_u16,
    vmovn_u32, vmulq_n_u32, vmulq_u32, vorrq_u32, vpaddq_s32, vpaddq_u16, vqaddq_s16, vqmovn_s32,
    vqmovun_s16, vqtbl1q_u8, vqtbl2q_u8, vreinterpretq_s16_u16, vreinterpretq_u8_u16,
    vreinterpretq_u8_u32, vreinterpretq_u16_s16, vreinterpretq_u16_u64, vreinterpretq_u32_u8,
    vreinterpretq_u32_u16, vreinterpretq_u64_u32, vrshlq_s32, vshll_high_n_u8, vshll_n_u8,
    vshlq_n_u32, vshrq_n_u32, vst1q_s32, vst1q_u8, vst1q_u16, vst4_u16, vsubl_high_u8, vsubl_u8,
    vsubq_u16, vsubq_u32, vtrn1q_u16, vtrn1q_u32, vtrn1q_u64, vtrn2q_u16, vtrn2q_u32, vtrn2q_u64,
};

/// Samples per step.
const LANES: usize = 16;
/// Samples one vector of filter output reads from each row.
const WINDOW: usize = LANES + 2 * VVC_ALF_REACH;

/// Sixteen samples of `row` from `at`.
#[target_feature(enable = "neon")]
fn load(row: &[u8], at: usize) -> uint8x16_t {
    let samples: &[u8; LANES] = row[at..at + LANES].try_into().expect("sixteen samples");
    // SAFETY: NEON is enabled; the load reads exactly `samples`.
    unsafe { vld1q_u8(samples.as_ptr()) }
}

/// Sixteen samples of `row` from `at`, widened to two 16-bit halves.
#[target_feature(enable = "neon")]
fn load_wide(row: &[u8], at: usize) -> [int16x8_t; 2] {
    let samples = load(row, at);
    [
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(samples))),
        vreinterpretq_s16_u16(vmovl_high_u8(samples)),
    ]
}

/// `acc + Σ_k c[k] · sums[k]` over one block's four samples.
#[target_feature(enable = "neon")]
fn block_sum(
    acc: int32x4_t,
    sums: &[int16x4_t; VVC_ALF_TAPS],
    c: &[i16; VVC_ALF_TAPS],
) -> int32x4_t {
    // SAFETY: NEON is enabled; the loads read 16 and 8 bytes of `c`.
    let (c0, c1) = unsafe { (vld1q_s16(c.as_ptr()), vld1_s16(c[8..].as_ptr())) };
    let mut acc = vmlal_laneq_s16::<0>(acc, sums[0], c0);
    acc = vmlal_laneq_s16::<1>(acc, sums[1], c0);
    acc = vmlal_laneq_s16::<2>(acc, sums[2], c0);
    acc = vmlal_laneq_s16::<3>(acc, sums[3], c0);
    acc = vmlal_laneq_s16::<4>(acc, sums[4], c0);
    acc = vmlal_laneq_s16::<5>(acc, sums[5], c0);
    acc = vmlal_laneq_s16::<6>(acc, sums[6], c0);
    acc = vmlal_laneq_s16::<7>(acc, sums[7], c0);
    acc = vmlal_lane_s16::<0>(acc, sums[8], c1);
    acc = vmlal_lane_s16::<1>(acc, sums[9], c1);
    acc = vmlal_lane_s16::<2>(acc, sums[10], c1);
    vmlal_lane_s16::<3>(acc, sums[11], c1)
}

/// One row's sixteen filtered samples from its `windows` (each row's
/// samples from 3 columns left of the first output) and its four blocks'
/// coefficients.
#[target_feature(enable = "neon")]
fn filter_vector(
    windows: [&[u8; WINDOW]; 7],
    blocks: &[[i16; VVC_ALF_TAPS]; LANES / VVC_ALF_BLOCK],
    shift: u32,
) -> uint8x16_t {
    // `vrshl` by −shift is the rounding right shift `(v + 2^(s−1)) >> s`.
    let shift_right = vdupq_n_s32(-(shift as i32));
    let x = VVC_ALF_REACH;
    let centre = load_wide(windows[3], x);
    let centre_bytes = load(windows[3], x);
    let twice = [
        vshll_n_u8::<1>(vget_low_u8(centre_bytes)),
        vshll_high_n_u8::<1>(centre_bytes),
    ];
    // a + b − 2c with widening adds, wrapping in u16: the i16 bit
    // pattern of the sum.
    let sums: [[int16x8_t; 2]; VVC_ALF_TAPS] = std::array::from_fn(|k| {
        let (dx, dy) = VVC_ALF_TAP_OFFSETS[k];
        let a = load(windows[(3 + dy) as usize], (x as isize + dx) as usize);
        let b = load(windows[(3 - dy) as usize], (x as isize - dx) as usize);
        [
            vsubq_u16(vaddl_u8(vget_low_u8(a), vget_low_u8(b)), twice[0]),
            vsubq_u16(vaddl_high_u8(a, b), twice[1]),
        ]
        .map(|v| vreinterpretq_s16_u16(v))
    });
    // Block b's four samples: half b / 2, low or high four lanes.
    let block: [int32x4_t; 4] = std::array::from_fn(|b| {
        let lanes: [int16x4_t; VVC_ALF_TAPS] = std::array::from_fn(|k| {
            let half = sums[k][b / 2];
            if b % 2 == 0 {
                vget_low_s16(half)
            } else {
                vget_high_s16(half)
            }
        });
        vrshlq_s32(block_sum(vdupq_n_s32(0), &lanes, &blocks[b]), shift_right)
    });
    let offsets =
        [0, 1].map(|h| vcombine_s16(vqmovn_s32(block[2 * h]), vqmovn_s32(block[2 * h + 1])));
    // Saturating to 16 bits keeps every out-of-range result out of
    // range after adding the centre, so the final clamp is exact.
    let filtered = [0, 1].map(|h| vqmovun_s16(vqaddq_s16(centre[h], offsets[h])));
    vcombine_u8(filtered[0], filtered[1])
}

/// NEON [`super::FilterBlockRowFn`] (without the dot-product extension;
/// `neon_dotprod` has the faster form): sixteen columns at a time.
#[target_feature(enable = "neon")]
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
        for ((rows, &shift), out) in rows.iter().zip(shifts).zip(out.iter_mut()) {
            // Each row's window once (one bounds check); taps index it at
            // constant offsets.
            let windows: [&[u8; WINDOW]; 7] = rows.map(|row| {
                row[i * LANES..i * LANES + WINDOW]
                    .try_into()
                    .expect("window")
            });
            let filtered = filter_vector(windows, blocks, shift);
            let out: &mut [u8; LANES] = (&mut out[i * LANES..(i + 1) * LANES])
                .try_into()
                .expect("sixteen samples");
            // SAFETY: NEON is enabled; the store writes exactly `out`.
            unsafe { vst1q_u8(out.as_mut_ptr(), filtered) };
        }
    }
    filter_block_row_from(rows, coefficients, shifts, out, whole);
}

/// `|2c − (a + b)|` for sixteen columns, as two 16-bit halves.
#[target_feature(enable = "neon")]
fn laplacian(centre: uint8x16_t, a: uint8x16_t, b: uint8x16_t) -> [uint16x8_t; 2] {
    [
        vabdq_u16(
            vshll_n_u8::<1>(vget_low_u8(centre)),
            vaddl_u8(vget_low_u8(a), vget_low_u8(b)),
        ),
        vabdq_u16(vshll_high_n_u8::<1>(centre), vaddl_high_u8(a, b)),
    ]
}

/// NEON [`super::LaplacianFn`]: four groups (sixteen columns) per step.
#[target_feature(enable = "neon")]
pub(super) unsafe fn laplacians(rows: &[&[u8]; 4], out: &mut [[u16; 4]]) {
    // Odd columns (centred one row down) take the next row.
    let odd: [u8; LANES] = std::array::from_fn(|k| if k % 2 == 1 { 0xff } else { 0 });
    // SAFETY: NEON is enabled; the load reads the sixteen bytes of `odd`.
    let odd = unsafe { vld1q_u8(odd.as_ptr()) };
    let (steps, _) = out.as_chunks_mut::<4>();
    let whole = steps.len() * 4;
    for (step, out) in steps.iter_mut().enumerate() {
        let x = 16 * step + 1;
        let pick =
            |row: usize, at: usize| vbslq_u8(odd, load(rows[row + 1], at), load(rows[row], at));
        let [up, centre, down] = [0, 1, 2].map(|row| [x - 1, x, x + 1].map(|at| pick(row, at)));
        let pair_sums = |[low, high]: [uint16x8_t; 2]| vpaddq_u16(low, high);
        let vertical = pair_sums(laplacian(centre[1], up[1], down[1]));
        let horizontal = pair_sums(laplacian(centre[1], centre[0], centre[2]));
        let diagonal0 = pair_sums(laplacian(centre[1], up[0], down[2]));
        let diagonal1 = pair_sums(laplacian(centre[1], down[0], up[2]));
        // [v of groups 0..4, h of groups 0..4] and the same for the diagonals.
        let straight = vpaddq_u16(vertical, horizontal);
        let diagonal = vpaddq_u16(diagonal0, diagonal1);
        let groups = uint16x4x4_t(
            vget_low_u16(straight),
            vget_high_u16(straight),
            vget_low_u16(diagonal),
            vget_high_u16(diagonal),
        );
        // SAFETY: NEON is enabled; the interleaved store writes exactly the
        // sixteen values of `out`.
        unsafe { vst4_u16(out.as_mut_ptr().cast(), groups) };
    }
    laplacians_from(rows, out, whole);
}

/// Bytes of a row a full [`super::TapSumsFn`] call reads: two 64-byte
/// deinterleaving loads, 4 bytes apart.
const SAMPLED_WINDOW: usize = 4 * VVC_ALF_SAMPLES + 4;

/// The columns `4i + dx` of one row for `dx` −3..=4 (index `dx + 3`).
#[target_feature(enable = "neon")]
fn row_phases(row: &[u8]) -> [uint8x16_t; 8] {
    let row: &[u8; SAMPLED_WINDOW] = row[..SAMPLED_WINDOW].try_into().expect("window");
    // SAFETY: NEON is enabled; each `ld4` reads 64 bytes of `row`, from
    // offsets 0 and 4.
    let (a, b) = unsafe { (vld4q_u8(row.as_ptr()), vld4q_u8(row[4..].as_ptr())) };
    [a.0, a.1, a.2, a.3, b.0, b.1, b.2, b.3]
}

/// Tap `K`'s sums from the rows' phases, as the 16-bit patterns of
/// `a + b − 2c` (wrapping in u16), samples 0..8 and 8..16.
#[target_feature(enable = "neon")]
fn tap<const K: usize>(phases: &[[uint8x16_t; 8]; 7]) -> [uint16x8_t; 2] {
    let (dx, dy) = VVC_ALF_TAP_OFFSETS[K];
    let at = |dy: isize, dx: isize| phases[(3 + dy) as usize][(dx + 3) as usize];
    let (a, b, centre) = (at(dy, dx), at(-dy, -dx), at(0, 0));
    [
        vsubq_u16(
            vaddl_u8(vget_low_u8(a), vget_low_u8(b)),
            vshll_n_u8::<1>(vget_low_u8(centre)),
        ),
        vsubq_u16(vaddl_high_u8(a, b), vshll_high_n_u8::<1>(centre)),
    ]
}

/// The 8x8 transpose of 16-bit lanes.
#[target_feature(enable = "neon")]
fn transpose8(rows: [uint16x8_t; 8]) -> [uint16x8_t; 8] {
    let pairs: [uint16x8_t; 8] = std::array::from_fn(|k| {
        let (first, second) = (rows[k / 2 * 2], rows[k / 2 * 2 + 1]);
        if k % 2 == 0 {
            vtrn1q_u16(first, second)
        } else {
            vtrn2q_u16(first, second)
        }
    });
    let pairs = pairs.map(|v| vreinterpretq_u32_u16(v));
    // Rows 0..4: columns 0 and 4 (quads[0]), 1 and 5, 2 and 6, 3 and 7;
    // quads[4..8] the same for rows 4..8.
    let quads = [
        vtrn1q_u32(pairs[0], pairs[2]),
        vtrn1q_u32(pairs[1], pairs[3]),
        vtrn2q_u32(pairs[0], pairs[2]),
        vtrn2q_u32(pairs[1], pairs[3]),
        vtrn1q_u32(pairs[4], pairs[6]),
        vtrn1q_u32(pairs[5], pairs[7]),
        vtrn2q_u32(pairs[4], pairs[6]),
        vtrn2q_u32(pairs[5], pairs[7]),
    ]
    .map(|v| vreinterpretq_u64_u32(v));
    std::array::from_fn(|c| {
        let (first, second) = (quads[c % 4], quads[4 + c % 4]);
        vreinterpretq_u16_u64(if c < 4 {
            vtrn1q_u64(first, second)
        } else {
            vtrn2q_u64(first, second)
        })
    })
}

/// NEON [`super::SamplesFn`]: sixteen samples per call, each row's
/// columns deinterleaved by four with two `ld4`, transposed to one sample
/// per pair of vectors and put in coefficient order by two-register table
/// lookups; partial calls take the scalar code.
#[target_feature(enable = "neon")]
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
    let source: &[u8; 4 * VVC_ALF_SAMPLES] = source[..4 * VVC_ALF_SAMPLES]
        .try_into()
        .expect("64 source bytes");
    // SAFETY: NEON is enabled; the `ld4` reads exactly `source`.
    let original = unsafe { vld4q_u8(source.as_ptr()) }.0;
    let centre = phases[3][3];
    // c − source, wrapping in u16: the i16 bit pattern of the error.
    let error = [
        vsubl_u8(vget_low_u8(centre), vget_low_u8(original)),
        vsubl_high_u8(centre, original),
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
    // SAFETY: NEON is enabled; each load reads 16 bytes of a 32-byte table.
    let tables: [[uint8x16_t; 2]; 4] = SAMPLE_BYTE_SOURCES
        .map(|sources| unsafe { [vld1q_u8(sources.as_ptr()), vld1q_u8(sources[16..].as_ptr())] });
    let zero = vdupq_n_u16(0);
    for half in 0..2 {
        let first = transpose8(std::array::from_fn(|k| taps[k][half]));
        let second = transpose8(std::array::from_fn(|k| match k {
            0..4 => taps[8 + k][half],
            4 => error[half],
            _ => zero,
        }));
        for (c, (first, second)) in first.into_iter().zip(second).enumerate() {
            let i = 8 * half + c;
            let sample = uint8x16x2_t(vreinterpretq_u8_u16(first), vreinterpretq_u8_u16(second));
            let [low, high] = tables[usize::from(transposes[i] & 3)];
            // SAFETY: NEON is enabled; the stores write the 16 values of `out[i]`.
            unsafe {
                vst1q_u8(out[i].as_mut_ptr().cast(), vqtbl2q_u8(sample, low));
                vst1q_u8(out[i][8..].as_mut_ptr().cast(), vqtbl2q_u8(sample, high));
            }
        }
    }
}

/// Samples one Gram step transposes: one 16-bit vector per value.
const GRAM_STEP: usize = 8;

/// The values of up to [`GRAM_STEP`] samples (zeros past the last), one
/// vector per value, sample `s` in lane `s`.
#[target_feature(enable = "neon")]
fn gram_columns(samples: &[[i16; VVC_ALF_SAMPLE_LANES]]) -> [int16x8_t; 2 * GRAM_STEP] {
    let mut halves = [[vdupq_n_u16(0); GRAM_STEP]; 2];
    for (s, sample) in samples.iter().enumerate() {
        // SAFETY: NEON is enabled; the loads read the sample's 16 values.
        unsafe {
            halves[0][s] = vreinterpretq_u16_s16(vld1q_s16(sample.as_ptr()));
            halves[1][s] = vreinterpretq_u16_s16(vld1q_s16(sample[8..].as_ptr()));
        }
    }
    let [low, high] = halves.map(|half| transpose8(half));
    std::array::from_fn(|i| {
        vreinterpretq_s16_u16(if i < GRAM_STEP {
            low[i]
        } else {
            high[i - GRAM_STEP]
        })
    })
}

/// NEON [`super::GramFn`], transposed: each step puts eight samples' values
/// in one vector per value, so each Gram entry is two widening
/// multiply-adds of two value vectors per eight samples (one for a tail of
/// at most four), with no lanes outside the upper triangle; the entries'
/// accumulators reduce across lanes four entries at a time at the end.
/// Each 32-bit lane sums at most 2^12 products of at most 2^18.
///
/// Why: the multiply-adds issue on one vector pipe of the Cortex-A78, and
/// broadcasting one value against quads of another (the AVX2 and SIMD128
/// form) spends 148 lane products per sample on the 91 entries, this form
/// 91. Jetson (`nc-checkasm --bench`, 512 samples): 11.2 → 8.1 µs.
#[target_feature(enable = "neon")]
pub(super) unsafe fn gram(
    samples: &[[i16; VVC_ALF_SAMPLE_LANES]],
    out: &mut [i32; VVC_ALF_DOT_LANES],
) {
    let mut sums = [vdupq_n_s32(0); VVC_ALF_DOT_LANES];
    for step in samples.chunks(GRAM_STEP) {
        let columns = gram_columns(step);
        let mut entry = 0;
        for i in 0..VVC_ALF_GRAM_COLUMNS {
            for j in i..VVC_ALF_GRAM_COLUMNS {
                let (a, b) = (columns[i], columns[j]);
                let mut sum = vmlal_s16(sums[entry], vget_low_s16(a), vget_low_s16(b));
                if step.len() > GRAM_STEP / 2 {
                    sum = vmlal_high_s16(sum, a, b);
                }
                sums[entry] = sum;
                entry += 1;
            }
        }
    }
    let (quads, _) = sums.as_chunks::<4>();
    let (outputs, _) = out.as_chunks_mut::<4>();
    for (out, [a, b, c, d]) in outputs.iter_mut().zip(quads) {
        let reduced = vpaddq_s32(vpaddq_s32(*a, *b), vpaddq_s32(*c, *d));
        // SAFETY: NEON is enabled; the store writes exactly `out`.
        unsafe { vst1q_s32(out.as_mut_ptr(), reduced) };
    }
}

/// NEON [`super::DotFn`]: widening multiply-adds into two 64-bit lanes.
#[target_feature(enable = "neon")]
pub(super) unsafe fn dot(a: &[i32; VVC_ALF_DOT_LANES], b: &[i32; VVC_ALF_DOT_LANES]) -> i64 {
    let (a, _) = a.as_chunks::<4>();
    let (b, _) = b.as_chunks::<4>();
    let mut sum = vdupq_n_s64(0);
    for (a, b) in a.iter().zip(b) {
        // SAFETY: NEON is enabled; each load reads one 16-byte chunk.
        let (a, b) = unsafe { (vld1q_s32(a.as_ptr()), vld1q_s32(b.as_ptr())) };
        sum = vmlal_high_s32(vmlal_s32(sum, vget_low_s32(a), vget_low_s32(b)), a, b);
    }
    vaddvq_s64(sum)
}

/// `table[index]` per 32-bit lane (indexes 0..16; `table[0]` must be 0,
/// which the lanes' upper bytes look up).
#[target_feature(enable = "neon")]
fn lookup(table: uint8x16_t, index: uint32x4_t) -> uint32x4_t {
    vreinterpretq_u32_u8(vqtbl1q_u8(table, vreinterpretq_u8_u32(index)))
}

/// `[class index, transpose index]` of four blocks as 16-bit lanes, from
/// their Laplacian sums.
#[target_feature(enable = "neon")]
fn classes4(
    [ver, hor, dg0, dg1]: [uint32x4_t; 4],
    scale: u32,
    tables: [uint8x16_t; 2],
) -> uint16x4_t {
    let activity = vminq_u32(
        vshrq_n_u32::<{ ACTIVITY_SHIFT as i32 }>(vmulq_n_u32(vaddq_u32(ver, hor), scale)),
        vdupq_n_u32(MAX_ACTIVITY),
    );
    let vertical = vcgtq_u32(ver, hor);
    let (hv1, hv0) = (vmaxq_u32(ver, hor), vminq_u32(ver, hor));
    let hv_direction = vbslq_u32(vertical, vdupq_n_u32(1), vdupq_n_u32(3));
    let first_diagonal = vcgtq_u32(dg0, dg1);
    let (d1, d0) = (vmaxq_u32(dg0, dg1), vminq_u32(dg0, dg1));
    let d_direction = vbslq_u32(first_diagonal, vdupq_n_u32(0), vdupq_n_u32(2));
    let diagonal = vcgtq_u32(vmulq_u32(d1, hv0), vmulq_u32(hv1, d0));
    let strong = vbslq_u32(diagonal, d1, hv1);
    let weak = vbslq_u32(diagonal, d0, hv0);
    let main = vbslq_u32(diagonal, d_direction, hv_direction);
    let secondary = vbslq_u32(diagonal, hv_direction, d_direction);
    // Strength 1 (strong > 2 weak) or 2 (2 strong > 9 weak, which implies
    // 1): minus the masks.
    let one = vcgtq_u32(strong, vaddq_u32(weak, weak));
    let two = vcgtq_u32(vaddq_u32(strong, strong), vmulq_n_u32(weak, 9));
    let strength = vsubq_u32(vsubq_u32(vdupq_n_u32(0), one), two);
    let step_classes = vaddq_u32(vshlq_n_u32::<1>(vandq_u32(main, vdupq_n_u32(1))), strength);
    let directional = vandq_u32(vmulq_n_u32(step_classes, u32::from(ACTIVITY_CLASSES)), one);
    let index = vaddq_u32(lookup(tables[0], activity), directional);
    let transposes = lookup(
        tables[1],
        vaddq_u32(vshlq_n_u32::<1>(main), vshrq_n_u32::<1>(secondary)),
    );
    vmovn_u32(vorrq_u32(index, vshlq_n_u32::<8>(transposes)))
}

/// NEON [`super::ClassifyFn`]: eight blocks per step; `ld4` separates the
/// groups' four sums.
#[target_feature(enable = "neon")]
pub(super) unsafe fn classify(groups: &[[u16; 4]], boundary: bool, out: &mut [[u8; 2]]) {
    let scale = if boundary {
        BOUNDARY_ACTIVITY_SCALE
    } else {
        ACTIVITY_SCALE
    };
    let mut transpose = [0u8; 16];
    transpose[..8].copy_from_slice(&TRANSPOSE);
    // SAFETY: NEON is enabled; each load reads exactly 16 table bytes.
    let tables = unsafe {
        [
            vld1q_u8(ACTIVITY_CLASS.as_ptr()),
            vld1q_u8(transpose.as_ptr()),
        ]
    };
    let (steps, _) = out.as_chunks_mut::<8>();
    let whole = steps.len() * 8;
    for (step, out) in steps.iter_mut().enumerate() {
        let load = |at: usize| {
            let groups: &[[u16; 4]; 8] = groups[at..at + 8].try_into().expect("eight groups");
            // SAFETY: NEON is enabled; the load reads exactly `groups`.
            unsafe { vld4q_u16(groups.as_ptr().cast()) }
        };
        let (left, right) = (load(8 * step), load(8 * step + 1));
        let sums = [
            vaddq_u16(left.0, right.0),
            vaddq_u16(left.1, right.1),
            vaddq_u16(left.2, right.2),
            vaddq_u16(left.3, right.3),
        ];
        let low = classes4(sums.map(|s| vmovl_u16(vget_low_u16(s))), scale, tables);
        let high = classes4(sums.map(|s| vmovl_high_u16(s)), scale, tables);
        // SAFETY: NEON is enabled; the store writes exactly the 16 bytes of `out`.
        unsafe { vst1q_u16(out.as_mut_ptr().cast(), vcombine_u16(low, high)) };
    }
    classify_from(groups, boundary, out, whole);
}
