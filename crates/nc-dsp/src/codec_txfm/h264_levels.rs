//! H.264 4×4 level costs and macroblock luma elimination.
//! The encoder supplies both thresholds. Each tier uses the same
//! scan order and updates the same coded pattern.

/// H.264 frame zig-zag scan for a 4×4 coefficient block (§8.5.6, Table 8-13).
pub const H264_ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Raster indices of the four 4×4 blocks in one 8×8 macroblock quadrant.
#[must_use]
pub const fn h264_blocks_of_8x8(quadrant: usize) -> [usize; 4] {
    let first = 8 * (quadrant / 2) + 2 * (quadrant % 2);
    [first, first + 1, first + 4, first + 5]
}

/// Cost of a lone ±1 level by the preceding zero run in scan order
/// (JM `COEFF_COST4x4`).
const RUN_COST: [u32; 16] = [3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const LARGE_LEVEL_COST: u32 = 9;

/// A 4×4 block's elimination cost, ignoring the first `first` scan entries.
///
/// # Panics
///
/// The number of ignored entries must be at most 16.
#[must_use]
#[inline]
pub fn h264_coefficient_cost(levels: &[i16; 16], first: usize) -> u32 {
    assert!(first <= H264_ZIGZAG_4X4.len());
    scalar_cost(levels, first).unwrap_or(0)
}

/// The cost is absent only for an all-zero block. A nonzero level late in
/// the scan can have zero cost and still set the coded block pattern.
#[inline]
fn scalar_cost(levels: &[i16; 16], first: usize) -> Option<u32> {
    if levels.iter().fold(0, |bits, &level| bits | level) == 0 {
        return None;
    }
    let mut cost = 0;
    let mut run = 0;
    for &position in &H264_ZIGZAG_4X4[first..] {
        match levels[position].unsigned_abs() {
            0 => run += 1,
            1 => {
                cost += RUN_COST[run];
                run = 0;
            }
            _ => return Some(LARGE_LEVEL_COST),
        }
    }
    Some(cost)
}

#[inline]
fn eliminate(
    levels: &mut [[i16; 16]; 16],
    (quadrant_limit, macroblock_limit): (u32, u32),
    mut block_cost: impl FnMut(&[i16; 16]) -> Option<u32>,
) -> u8 {
    let mut pattern = 0;
    let mut total = 0;
    for quadrant in 0..4 {
        let blocks = h264_blocks_of_8x8(quadrant);
        let mut cost = 0;
        let mut nonzero = false;
        for block in blocks {
            if let Some(value) = block_cost(&levels[block]) {
                cost += value;
                nonzero = true;
            }
        }
        if cost < quadrant_limit {
            for block in blocks {
                levels[block] = [0; 16];
            }
        } else {
            total += cost;
            pattern |= u8::from(nonzero) << quadrant;
        }
    }
    if total < macroblock_limit {
        *levels = [[0; 16]; 16];
        0
    } else {
        pattern
    }
}

pub(super) fn scalar(levels: &mut [[i16; 16]; 16], limits: (u32, u32)) -> u8 {
    eliminate(levels, limits, |block| scalar_cost(block, 0))
}

#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const SCAN: [u8; 16] = {
    let mut order = [0; 16];
    let mut i = 0;
    while i < order.len() {
        order[i] = H264_ZIGZAG_4X4[i] as u8;
        i += 1;
    }
    order
};

/// Mark each nonzero with a predecessor within 1, 3 or 6 positions,
/// corresponding to zero runs of 0, 0–2 and 0–5. Bit zero is the implicit
/// predecessor of the first coefficient.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline]
fn run_matches(mask: u32) -> [u32; 3] {
    let bits = (mask << 1) | 1;
    let adjacent = bits << 1;
    let short = adjacent | (adjacent << 1) | (adjacent << 2);
    let medium = short | (short << 3);
    [bits & adjacent, bits & short, bits & medium]
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::{
        _mm_abs_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_packs_epi16, _mm_set1_epi8,
        _mm_shuffle_epi8, _mm_slli_epi16, _mm_testz_si128,
    };

    use super::{LARGE_LEVEL_COST, SCAN, eliminate, run_matches};

    #[target_feature(enable = "avx2,popcnt")]
    #[inline]
    fn run_cost(mask: u32) -> u32 {
        let [adjacent, short, medium] = run_matches(mask);
        adjacent.count_ones() + short.count_ones() + medium.count_ones()
    }

    #[target_feature(enable = "avx2,popcnt")]
    #[inline]
    fn block_cost(levels: &[i16; 16]) -> Option<u32> {
        // SAFETY: the two loads consume exactly sixteen levels.
        let values = unsafe {
            _mm_packs_epi16(
                _mm_loadu_si128(levels.as_ptr().cast()),
                _mm_loadu_si128(levels.as_ptr().add(8).cast()),
            )
        };
        if _mm_testz_si128(values, values) != 0 {
            return None;
        }
        // Signed saturation preserves whether |level| is 0, 1 or larger.
        // Absolute values are treated as unsigned so that −128 is large.
        let magnitudes = _mm_abs_epi8(values);
        if _mm_testz_si128(magnitudes, _mm_set1_epi8(-2)) == 0 {
            return Some(LARGE_LEVEL_COST);
        }
        // SAFETY: the scan vector has sixteen indices in 0..16.
        let scan = unsafe { _mm_loadu_si128(SCAN.as_ptr().cast()) };
        let ordered = _mm_shuffle_epi8(magnitudes, scan);
        let mask = _mm_movemask_epi8(_mm_slli_epi16::<7>(ordered)) as u32;
        Some(run_cost(mask))
    }

    #[target_feature(enable = "avx2,popcnt")]
    pub(in crate::codec_txfm) unsafe fn avx2(
        levels: &mut [[i16; 16]; 16],
        limits: (u32, u32),
    ) -> u8 {
        eliminate(levels, limits, |block| block_cost(block))
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) use x86::avx2;

#[cfg(target_arch = "aarch64")]
mod arm {
    use core::arch::aarch64::{
        vabsq_s8, vaddv_u8, vget_high_u8, vget_low_u8, vld1q_s16, vld1q_u8, vmaxvq_u8, vmulq_u8,
        vqmovn_high_s16, vqmovn_s16, vqtbl1q_u8, vreinterpretq_u8_s8,
    };

    use super::{LARGE_LEVEL_COST, SCAN, eliminate, run_matches};

    #[target_feature(enable = "neon")]
    #[inline]
    fn block_cost(levels: &[i16; 16]) -> Option<u32> {
        // SAFETY: the two loads consume exactly sixteen levels.
        let values = unsafe {
            let low = vqmovn_s16(vld1q_s16(levels.as_ptr()));
            vqmovn_high_s16(low, vld1q_s16(levels.as_ptr().add(8)))
        };
        let magnitudes = vreinterpretq_u8_s8(vabsq_s8(values));
        match vmaxvq_u8(magnitudes) {
            0 => return None,
            1 => (),
            _ => return Some(LARGE_LEVEL_COST),
        }
        const WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        // SAFETY: both tables have sixteen entries and every magnitude is 0 or 1.
        let bits = unsafe {
            let ordered = vqtbl1q_u8(magnitudes, vld1q_u8(SCAN.as_ptr()));
            vmulq_u8(ordered, vld1q_u8(WEIGHTS.as_ptr()))
        };
        let mask =
            u32::from(vaddv_u8(vget_low_u8(bits))) | (u32::from(vaddv_u8(vget_high_u8(bits))) << 8);
        let [adjacent, short, medium] = run_matches(mask);
        // The first bit and sixteen coefficients fit each field in 17 bits.
        // One count avoids three vector round trips from scalar registers.
        let combined = u64::from(adjacent) | (u64::from(short) << 17) | (u64::from(medium) << 34);
        Some(combined.count_ones())
    }

    #[target_feature(enable = "neon")]
    pub(in crate::codec_txfm) unsafe fn neon(
        levels: &mut [[i16; 16]; 16],
        limits: (u32, u32),
    ) -> u8 {
        eliminate(levels, limits, |block| block_cost(block))
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) use arm::neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{
        i8x16_abs, i8x16_bitmask, i8x16_narrow_i16x8, i8x16_splat, i8x16_swizzle, i16x8_shl,
        v128_and, v128_any_true, v128_load,
    };

    use super::{LARGE_LEVEL_COST, SCAN, eliminate, run_matches};

    #[target_feature(enable = "simd128")]
    #[inline]
    fn block_cost(levels: &[i16; 16]) -> Option<u32> {
        // SAFETY: the two loads consume exactly sixteen levels.
        let values = unsafe {
            i8x16_narrow_i16x8(
                v128_load(levels.as_ptr().cast()),
                v128_load(levels.as_ptr().add(8).cast()),
            )
        };
        if !v128_any_true(values) {
            return None;
        }
        let magnitudes = i8x16_abs(values);
        if v128_any_true(v128_and(magnitudes, i8x16_splat(-2))) {
            return Some(LARGE_LEVEL_COST);
        }
        // SAFETY: the scan vector has sixteen entries in 0..16.
        let scan = unsafe { v128_load(SCAN.as_ptr().cast()) };
        let ordered = i8x16_swizzle(magnitudes, scan);
        let mask = u32::from(i8x16_bitmask(i16x8_shl(ordered, 7)));
        let [adjacent, short, medium] = run_matches(mask);
        Some(adjacent.count_ones() + short.count_ones() + medium.count_ones())
    }

    #[target_feature(enable = "simd128")]
    pub(in crate::codec_txfm) unsafe fn simd128(
        levels: &mut [[i16; 16]; 16],
        limits: (u32, u32),
    ) -> u8 {
        eliminate(levels, limits, |block| block_cost(block))
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) use wasm::simd128;
