//! NEON ALF filter with the dot-product extension (`sdot`), installed after
//! runtime detection.
//!
//! The filter is `Σ_k c_k (a_k + b_k − 2·centre)` over the 7x7 diamond's
//! twelve symmetric pairs. Offsetting every sample by −128 (an `eor` with
//! 0x80) leaves each pair term unchanged and makes the samples signed
//! bytes, and the coefficients are bytes too (§7.4.3.18), so the 24
//! neighbours of a sample are six `sdot` groups of four products, each
//! group one byte per tap in a 32-bit lane of the sample. The centre
//! weight `−2 Σ c_k` does not fit a byte and takes one `smull` per four
//! samples. A table lookup (`tbl`) gathers each group's bytes for four
//! samples from one to three row registers; a block's coefficient bytes,
//! repeated per sample, are built once per 4x4 block for all four rows.
//!
//! Why: on the Cortex-A78 widening multiply-accumulates (`smlal`) issue on
//! one vector pipe, while `sdot` and `tbl` issue on both; the `smlal` form
//! (`neon`) needs 48 of them per 16 samples, this form 24 `sdot`.
//!
//! Jetson (`nc-checkasm --bench`, 64x4 block row): 1120 ns (`smlal`, one
//! row per call) → 700 ns; VVC Default encode cycles −2.1% crowd, −2.6%
//! bodycam, −1.7% static (jperf, CBR 600, 150 frames). Two accumulators
//! per half step (shorter `sdot` chains) measured the same; seven groups
//! without the three-register lookup were slower (732 ns).
//!
//! Kernel contract: see the `vvc_alf` module docs; coefficients must lie
//! within −128..=127.

use super::{
    VVC_ALF_BLOCK, VVC_ALF_REACH, VVC_ALF_TAP_OFFSETS, VVC_ALF_TAPS, filter_block_row_from,
};
use crate::simd::neon_dot::sdot;
use core::arch::aarch64::{
    int8x16_t, int8x16x2_t, int8x16x3_t, int16x8_t, int32x4_t, uint8x16_t, vaddq_s16, vcombine_s8,
    vcombine_s16, vdup_n_s16, vdupq_n_s16, vdupq_n_s32, vdupq_n_u8, veorq_u8, vextq_s8,
    vget_high_s16, vget_low_s8, vget_low_s16, vld1_s16, vld1q_s16, vld1q_u8, vmovl_s8, vmull_n_s16,
    vqaddq_s16, vqmovn_s16, vqmovn_s32, vqmovun_s16, vqtbl1q_s8, vqtbl2q_s8, vqtbl3q_s8,
    vreinterpretq_s8_u8, vrshlq_s32, vst1_u8,
};

/// Samples per step: two 4x4 blocks.
const LANES: usize = 2 * VVC_ALF_BLOCK;
/// Bytes loaded from each row per step: columns −3..13 of the step, of
/// which the filter reads −3..11.
const WINDOW: usize = 16;
/// Taps per `sdot` lane.
const GROUP_TAPS: usize = 4;
/// `sdot` groups covering the 24 neighbours.
const GROUPS: usize = 6;

/// One `sdot` group: the rows (`dy`) its table registers hold, in order,
/// and its four taps as `(register, dx)`.
struct Group {
    rows: &'static [isize],
    taps: [(usize, isize); GROUP_TAPS],
}

/// The groups: single-row windows where a row has four neighbours left
/// (rows 0, ±1), the ±2 rows with their ±3 neighbour, and the four rest.
const GROUP_TAPS_BY_ROW: [Group; GROUPS] = [
    Group {
        rows: &[0],
        taps: [(0, -3), (0, -2), (0, -1), (0, 1)],
    },
    Group {
        rows: &[-1],
        taps: [(0, -2), (0, -1), (0, 0), (0, 1)],
    },
    Group {
        rows: &[1],
        taps: [(0, -2), (0, -1), (0, 0), (0, 1)],
    },
    Group {
        rows: &[-2, -3],
        taps: [(0, -1), (0, 0), (0, 1), (1, 0)],
    },
    Group {
        rows: &[2, 3],
        taps: [(0, -1), (0, 0), (0, 1), (1, 0)],
    },
    Group {
        rows: &[0, -1, 1],
        taps: [(0, 2), (0, 3), (1, 2), (2, 2)],
    },
];

/// The coefficient of neighbour `(dx, dy)`: the tap whose pair holds it.
const fn tap_of(dx: isize, dy: isize) -> u8 {
    let mut k = 0;
    while k < VVC_ALF_TAPS {
        let (tx, ty) = VVC_ALF_TAP_OFFSETS[k];
        if (tx == dx && ty == dy) || (tx == -dx && ty == -dy) {
            return k as u8;
        }
        k += 1;
    }
    panic!("not a diamond neighbour");
}

/// Per group, the coefficient byte each byte of an `sdot` weight vector
/// takes: the group's four taps, repeated for the four samples.
const WEIGHT_SOURCES: [[u8; 16]; GROUPS] = {
    let mut sources = [[0; 16]; GROUPS];
    let mut g = 0;
    while g < GROUPS {
        let group = &GROUP_TAPS_BY_ROW[g];
        let mut byte = 0;
        while byte < 16 {
            let (register, dx) = group.taps[byte % GROUP_TAPS];
            sources[g][byte] = tap_of(dx, group.rows[register]);
            byte += 1;
        }
        g += 1;
    }
    sources
};

/// Per group and half step (samples 0..4, 4..8), the table byte each byte
/// of the gathered samples takes: register `r`'s column `c` (from −3) is
/// byte `16 r + c + 3`.
const SAMPLE_SOURCES: [[[u8; 16]; 2]; GROUPS] = {
    let mut sources = [[[0; 16]; 2]; GROUPS];
    let mut g = 0;
    while g < GROUPS {
        let mut half = 0;
        while half < 2 {
            let mut byte = 0;
            while byte < 16 {
                let sample = GROUP_TAPS * half + byte / GROUP_TAPS;
                let (register, dx) = GROUP_TAPS_BY_ROW[g].taps[byte % GROUP_TAPS];
                let column = sample as isize + dx + VVC_ALF_REACH as isize;
                sources[g][half][byte] = (16 * register as isize + column) as u8;
                byte += 1;
            }
            half += 1;
        }
        g += 1;
    }
    sources
};

/// Sixteen bytes of `table`.
#[target_feature(enable = "neon")]
fn load_table(table: &[u8; 16]) -> uint8x16_t {
    // SAFETY: NEON is enabled; the load reads exactly `table`.
    unsafe { vld1q_u8(table.as_ptr()) }
}

/// A block's `sdot` weights per group, and its centre weight `−2 Σ c_k`
/// (at most 2·12·128 in magnitude, so 16 bits).
#[target_feature(enable = "neon")]
fn block_weights(c: &[i16; VVC_ALF_TAPS]) -> ([int8x16_t; GROUPS], i16) {
    // SAFETY: NEON is enabled; the loads read the 24 bytes of `c`.
    let (first, last) = unsafe { (vld1q_s16(c.as_ptr()), vld1_s16(c[8..].as_ptr())) };
    // Saturating narrows are exact for coefficients in the contract's range.
    let bytes = vcombine_s8(
        vqmovn_s16(first),
        vqmovn_s16(vcombine_s16(last, vdup_n_s16(0))),
    );
    let weights = std::array::from_fn(|g| vqtbl1q_s8(bytes, load_table(&WEIGHT_SOURCES[g])));
    let sum: i32 = c.iter().map(|&c| i32::from(c)).sum();
    (weights, (-2 * sum) as i16)
}

/// The window of `row` for step `i`, offset to signed bytes (−128).
#[target_feature(enable = "neon")]
fn load_signed(row: &[u8], i: usize) -> int8x16_t {
    let window: &[u8; WINDOW] = row[i * LANES..i * LANES + WINDOW]
        .try_into()
        .expect("window");
    // SAFETY: NEON is enabled; the load reads exactly `window`.
    let samples = unsafe { vld1q_u8(window.as_ptr()) };
    vreinterpretq_s8_u8(veorq_u8(samples, vdupq_n_u8(0x80)))
}

/// `acc` plus the six groups' dot products for half step `half` (samples
/// `4 half..4 half + 4`), `windows` indexed by `dy + 3`.
#[target_feature(enable = "neon,dotprod")]
fn neighbours(
    mut acc: int32x4_t,
    windows: &[int8x16_t; 7],
    weights: &[int8x16_t; GROUPS],
    half: usize,
) -> int32x4_t {
    let row = |dy: isize| windows[(3 + dy) as usize];
    let sources = |g: usize| load_table(&SAMPLE_SOURCES[g][half]);
    let gathered = [
        vqtbl1q_s8(row(0), sources(0)),
        vqtbl1q_s8(row(-1), sources(1)),
        vqtbl1q_s8(row(1), sources(2)),
        vqtbl2q_s8(int8x16x2_t(row(-2), row(-3)), sources(3)),
        vqtbl2q_s8(int8x16x2_t(row(2), row(3)), sources(4)),
        vqtbl3q_s8(int8x16x3_t(row(0), row(-1), row(1)), sources(5)),
    ];
    for (samples, weights) in gathered.into_iter().zip(weights) {
        acc = sdot(acc, samples, *weights);
    }
    acc
}

/// NEON dot-product [`super::FilterBlockRowFn`]: eight columns (two
/// blocks) at a time.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn filter_block_row(
    rows: &[[&[u8]; 7]; VVC_ALF_BLOCK],
    coefficients: &[[i16; VVC_ALF_TAPS]],
    shifts: &[u32; VVC_ALF_BLOCK],
    out: &mut [&mut [u8]; VVC_ALF_BLOCK],
) {
    // Steps whose 16-byte windows lie inside every row (the contract
    // guarantees only the 14 bytes the filter reads).
    let shortest = rows.as_flattened().iter().map(|row| row.len()).min();
    let loadable = shortest.map_or(0, |len| (len + LANES).saturating_sub(WINDOW) / LANES);
    let vectors = (out[0].len() / LANES).min(loadable);
    let (block_pairs, _) = coefficients.as_chunks::<2>();
    for (i, blocks) in block_pairs.iter().take(vectors).enumerate() {
        let [(first, first_centre), (second, second_centre)] = blocks.map(|c| block_weights(&c));
        for ((rows, &shift), out) in rows.iter().zip(shifts).zip(out.iter_mut()) {
            let windows = rows.map(|row| load_signed(row, i));
            let centre_bytes = vextq_s8::<{ VVC_ALF_REACH as i32 }>(windows[3], windows[3]);
            let centre: int16x8_t = vmovl_s8(vget_low_s8(centre_bytes));
            let shift_right = vdupq_n_s32(-(shift as i32));
            let low = neighbours(
                vmull_n_s16(vget_low_s16(centre), first_centre),
                &windows,
                &first,
                0,
            );
            let high = neighbours(
                vmull_n_s16(vget_high_s16(centre), second_centre),
                &windows,
                &second,
                1,
            );
            // `vrshl` by −shift is the rounding right shift. Saturating to
            // 16 bits keeps every out-of-range result out of range after
            // adding the centre, so the final clamp is exact.
            let offsets = vcombine_s16(
                vqmovn_s32(vrshlq_s32(low, shift_right)),
                vqmovn_s32(vrshlq_s32(high, shift_right)),
            );
            let unsigned_centre = vaddq_s16(centre, vdupq_n_s16(128));
            let filtered = vqmovun_s16(vqaddq_s16(unsigned_centre, offsets));
            let out: &mut [u8; LANES] = (&mut out[i * LANES..(i + 1) * LANES])
                .try_into()
                .expect("eight samples");
            // SAFETY: NEON is enabled; the store writes exactly `out`.
            unsafe { vst1_u8(out.as_mut_ptr(), filtered) };
        }
    }
    filter_block_row_from(rows, coefficients, shifts, out, vectors * LANES);
}
