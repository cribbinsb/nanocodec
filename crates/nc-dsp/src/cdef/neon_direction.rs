//! dav1d's one-pass NEON CDEF direction search, shared by AV1 and AV2.
//! Reimplemented from dav1d 1.5.4 `src/arm/64/cdef_tmpl.S`, BSD-2-Clause,
//! Copyright 2018 VideoLAN/dav1d authors and 2020 Martin Storsjo; see
//! `THIRD_PARTY.md`. Rows accumulate both diagonals, both axes and all four
//! skew directions directly; no transposed second pass is required.

use core::arch::aarch64::{
    int16x8_t, int32x4_t, vaddq_s16, vcombine_s16, vdup_n_s16, vdup_n_u8, vdupq_n_s16, vdupq_n_s32,
    vextq_s16, vget_high_s16, vget_low_s16, vld1_s16, vld1_u8, vld1q_s16, vld1q_u32, vmaxvq_u32,
    vmlal_high_s16, vmlal_s16, vmlaq_s32, vmovl_high_s16, vmovl_s16, vmull_high_s16, vmull_s16,
    vmulq_s32, vorrq_u32, vpadd_s16, vpaddq_s16, vpaddq_s32, vreinterpretq_s16_u16,
    vreinterpretq_u32_s32, vrev64q_s16, vshlq_n_u32, vst1q_s32, vsubl_u8,
};

const SIDE: usize = 8;
const SAMPLE_BIAS: u8 = 128;
/// 840 is lcm(1..=8): the diagonal's reciprocal line-length weights.
const DIAGONAL_WEIGHTS: [i16; SIDE] = [840, 420, 280, 210, 168, 140, 120, 105];
/// Twelve lines of each skew direction: lengths 2,4,6, then five 8s,6,4,2.
const SKEW_WEIGHTS: [i16; 12] = [420, 210, 140, 105, 105, 105, 105, 105, 140, 210, 420, 0];
const AXIS_WEIGHT: i32 = 105;
const COST_LANES: usize = 4;
const INDEX_BITS: i32 = 2;
const LANE_MASK: usize = COST_LANES - 1;
const LANE_PRIORITY: [u32; COST_LANES] = [3, 2, 1, 0];

#[derive(Clone, Copy)]
struct Lines {
    lo: int16x8_t,
    hi: int16x8_t,
}

/// Prepend K zero lanes. All immediate EXT operands are valid, including
/// the K=0/8 cases, which do not need an EXT instruction.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn up<const K: usize>(row: int16x8_t) -> int16x8_t {
    let zero = vdupq_n_s16(0);
    match K {
        0 => row,
        1 => vextq_s16::<7>(zero, row),
        2 => vextq_s16::<6>(zero, row),
        3 => vextq_s16::<5>(zero, row),
        4 => vextq_s16::<4>(zero, row),
        5 => vextq_s16::<3>(zero, row),
        6 => vextq_s16::<2>(zero, row),
        7 => vextq_s16::<1>(zero, row),
        8 => zero,
        _ => unreachable!("CDEF line offset"),
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn down<const K: usize>(row: int16x8_t) -> int16x8_t {
    let zero = vdupq_n_s16(0);
    match K {
        0 => row,
        1 => vextq_s16::<1>(row, zero),
        2 => vextq_s16::<2>(row, zero),
        3 => vextq_s16::<3>(row, zero),
        4 => vextq_s16::<4>(row, zero),
        5 => vextq_s16::<5>(row, zero),
        6 => vextq_s16::<6>(row, zero),
        7 => vextq_s16::<7>(row, zero),
        8 => zero,
        _ => unreachable!("CDEF line offset"),
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn axis_cost(sum: int16x8_t) -> int32x4_t {
    let low = vget_low_s16(sum);
    let squares = vmlal_high_s16(vmull_s16(low, low), sum, sum);
    vmulq_s32(squares, vdupq_n_s32(AXIS_WEIGHT))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn diagonal_cost(lines: Lines, weights: [int32x4_t; 2]) -> int32x4_t {
    let reverse = vrev64q_s16(lines.hi);
    let mirror = vextq_s16::<5>(reverse, reverse);
    let a = vget_low_s16(lines.lo);
    let b = vget_low_s16(mirror);
    let first = vmlal_s16(vmull_s16(a, a), b, b);
    let second = vmlal_high_s16(vmull_high_s16(lines.lo, lines.lo), mirror, mirror);
    vmlaq_s32(vmulq_s32(first, weights[0]), second, weights[1])
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn skew_cost(lines: Lines, weights: [int32x4_t; 3]) -> int32x4_t {
    let a = vget_low_s16(lines.lo);
    let b = vget_high_s16(lines.lo);
    let c = vget_low_s16(lines.hi);
    let sum = vmlaq_s32(
        vmulq_s32(vmull_s16(a, a), weights[0]),
        vmull_s16(b, b),
        weights[1],
    );
    vmlaq_s32(sum, vmull_s16(c, c), weights[2])
}

/// Full 8-bit direction search. Eight centered samples sum to at most 1024;
/// line sums fit i16, while squared weighted costs need i32 (below 2^30).
#[target_feature(enable = "neon")]
pub(super) unsafe fn find_dir(src: &[u8], stride: usize) -> (u8, i32) {
    assert!(
        stride >= SIDE && (SIDE - 1) * stride + SIDE <= src.len(),
        "CDEF direction window"
    );
    let zero = vdupq_n_s16(0);
    let empty = Lines { lo: zero, hi: zero };
    let mut diagonal = [empty; 2];
    let mut skew = [empty; 4];
    let mut vertical = zero;
    // Literal row numbers let lane inserts and EXT offsets remain immediate.
    macro_rules! add {
        ($lines:expr, $row:expr, $offset:expr) => {{
            $lines.lo = vaddq_s16($lines.lo, up::<{ $offset }>($row));
            $lines.hi = vaddq_s16($lines.hi, down::<{ SIDE - $offset }>($row));
        }};
    }
    macro_rules! row {
        ($r:literal) => {{
            let samples = vld1_u8(src.as_ptr().add($r * stride));
            let row = vreinterpretq_s16_u16(vsubl_u8(samples, vdup_n_u8(SAMPLE_BIAS)));
            let pairs4 = vpadd_s16(vget_low_s16(row), vget_high_s16(row));
            let pairs = vcombine_s16(pairs4, vdup_n_s16(0));
            add!(diagonal[0], row, $r);
            // Reversing every line index leaves its squared cost and
            // reciprocal-length weight unchanged. Shift the forward row
            // by 7-r instead of reversing eight samples per row.
            add!(diagonal[1], row, SIDE - 1 - $r);
            add!(skew[0], pairs, $r);
            // The eleven skew weights are symmetric too: reflect the
            // row offset instead of reversing its four adjacent pairs.
            add!(skew[1], pairs, SIDE - 1 - $r);
            row
        }};
    }
    // SAFETY: NEON is enabled; each eight-byte row lies in the checked
    // source. All remaining operations are bounded register arithmetic.
    unsafe {
        // The two remaining skew directions group adjacent rows. Sum
        // them before shifting: a pair stays within [-256,254] and uses
        // half as many EXT/add sequences as accumulating individual rows.
        macro_rules! pair {
            ($even:literal, $odd:literal) => {{
                let a = row!($even);
                let b = row!($odd);
                let pair = vaddq_s16(a, b);
                vertical = vaddq_s16(vertical, pair);
                add!(skew[2], pair, 3 - $odd / 2);
                add!(skew[3], pair, $odd / 2);
                vpaddq_s16(a, b)
            }};
        }
        // Seven packed pairwise adds reduce all eight rows. This replaces
        // eight scalar ADDV/FMOV reductions and integer square operations;
        // each row sum is in [-1024,1016], so all i16 stages are exact.
        let first = vpaddq_s16(pair!(0, 1), pair!(2, 3));
        let second = vpaddq_s16(pair!(4, 5), pair!(6, 7));
        let horizontal = vpaddq_s16(first, second);
        let weights = vld1q_s16(DIAGONAL_WEIGHTS.as_ptr());
        let diagonal_weights = [vmovl_s16(vget_low_s16(weights)), vmovl_high_s16(weights)];
        let skew_weights = [
            vmovl_s16(vld1_s16(SKEW_WEIGHTS.as_ptr())),
            vmovl_s16(vld1_s16(SKEW_WEIGHTS.as_ptr().add(4))),
            vmovl_s16(vld1_s16(SKEW_WEIGHTS.as_ptr().add(8))),
        ];
        // Three pairwise adds reduce four costs together, avoiding four
        // ADDV-to-scalar transfers followed by lane inserts.
        let low = vpaddq_s32(
            vpaddq_s32(
                diagonal_cost(diagonal[0], diagonal_weights),
                skew_cost(skew[0], skew_weights),
            ),
            vpaddq_s32(axis_cost(horizontal), skew_cost(skew[1], skew_weights)),
        );
        let high = vpaddq_s32(
            vpaddq_s32(
                diagonal_cost(diagonal[1], diagonal_weights),
                skew_cost(skew[2], skew_weights),
            ),
            vpaddq_s32(axis_cost(vertical), skew_cost(skew[3], skew_weights)),
        );
        // Costs are below 2^30. Two low priority bits therefore fit in
        // u32 without losing cost precision: larger tags prefer the first
        // lane. Choose each half's winner, then compare their costs with
        // strict > so every tie prefers directions 0..3 before 4..7.
        let priority = vld1q_u32(LANE_PRIORITY.as_ptr());
        let low_tag = vmaxvq_u32(vorrq_u32(
            vshlq_n_u32::<INDEX_BITS>(vreinterpretq_u32_s32(low)),
            priority,
        ));
        let high_tag = vmaxvq_u32(vorrq_u32(
            vshlq_n_u32::<INDEX_BITS>(vreinterpretq_u32_s32(high)),
            priority,
        ));
        let use_high = (high_tag >> INDEX_BITS) > (low_tag >> INDEX_BITS);
        let tag = if use_high { high_tag } else { low_tag };
        let best =
            (((tag as usize) & LANE_MASK) ^ LANE_MASK) | (usize::from(use_high) * COST_LANES);
        let best_cost = (tag >> INDEX_BITS) as i32;
        let mut costs = [0i32; SIDE];
        vst1q_s32(costs.as_mut_ptr(), low);
        vst1q_s32(costs.as_mut_ptr().add(COST_LANES), high);
        (best as u8, (best_cost - costs[best ^ 4]) >> 10)
    }
}
