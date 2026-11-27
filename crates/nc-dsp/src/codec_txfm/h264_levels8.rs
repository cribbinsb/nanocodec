//! H.264 8×8 inter luma coefficient costs and macroblock elimination.
//! The quantizer's counts identify live quadrants. Their bit masks
//! encode the same scan-run costs as the scalar traversal.

/// Frame zig-zag scan of an 8x8 block (H.264 §8.5.7, Table 8-14): scan
/// position → raster index (`row * 8 + column`).
pub const H264_ZIGZAG_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, //
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, //
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, //
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

const EXPENSIVE_COEFF_COST: u32 = 9;

/// Cost of a lone ±1 level in an 8x8 block by the number of zeros before it
/// in scan order (JM `COEFF_COST8x8`, frame scan).
const COEFF_COST_8X8_BY_RUN: [u32; 24] = [
    3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

/// The elimination cost of one 8x8 block's levels (raster).
fn scalar_cost(levels: &[i16; 64], nonzero: u32) -> u32 {
    // The quantiser counted the non-zero levels: the scan stops after the
    // last one instead of walking all 64 positions (most coded blocks hold
    // a handful of low-frequency levels).
    let mut remaining = nonzero;
    let mut cost = 0;
    let mut run = 0usize;
    for &pos in &H264_ZIGZAG_8X8 {
        if remaining == 0 {
            break;
        }
        match levels[pos].unsigned_abs() {
            0 => run += 1,
            1 => {
                cost += COEFF_COST_8X8_BY_RUN.get(run).copied().unwrap_or(0);
                run = 0;
                remaining -= 1;
            }
            _ => return EXPENSIVE_COEFF_COST,
        }
    }
    cost
}

#[inline]
fn eliminate(
    levels: &mut [[i16; 64]; 4],
    mut nonzero: [u32; 4],
    (quadrant_limit, macroblock_limit): (u32, u32),
    mut block_cost: impl FnMut(&[i16; 64], u32) -> u32,
) -> u8 {
    let mut total = 0;
    for (block, count) in levels.iter_mut().zip(nonzero.iter_mut()) {
        if *count == 0 {
            continue;
        }
        let cost = block_cost(block, *count);
        if cost < quadrant_limit {
            *block = [0; 64];
            *count = 0;
        } else {
            total += cost;
        }
    }
    if total < macroblock_limit {
        if nonzero.iter().any(|&c| c != 0) {
            *levels = [[0; 64]; 4];
        }
        return 0;
    }
    (0..4).fold(0u8, |cbp, b8| cbp | (u8::from(nonzero[b8] != 0) << b8))
}

pub(super) fn scalar(levels: &mut [[i16; 64]; 4], counts: [u32; 4], limits: (u32, u32)) -> u8 {
    eliminate(levels, counts, limits, scalar_cost)
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const SCAN: [u8; 64] = {
    let mut order = [0; 64];
    let mut i = 0;
    while i < order.len() {
        order[i] = H264_ZIGZAG_8X8[i] as u8;
        i += 1;
    }
    order
};

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[inline]
fn run_distances(mask: u64) -> (u64, u64, u64) {
    // Include a virtual nonzero immediately before scan position zero.
    // Each set bit is within 4, 12 or 24 positions of the preceding nonzero.
    let previous = (mask << 1) | 1;
    let near2 = previous | (previous << 1);
    let near4 = near2 | (near2 << 2);
    let near12 = near4 | (near4 << 4) | (near4 << 8);
    let near24 = near12 | (near12 << 12);
    (near4, near12, near24)
}

#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
const INDEX: [u8; 64] = {
    let mut index = [0; 64];
    let mut i = 0;
    while i < index.len() {
        index[H264_ZIGZAG_8X8[i]] = i as u8;
        i += 1;
    }
    index
};

#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[inline]
fn raster_to_scan(mut raster: u64) -> u64 {
    let mut scan = 0u64;
    while raster != 0 {
        scan |= 1u64 << INDEX[raster.trailing_zeros() as usize];
        raster &= raster - 1;
    }
    scan
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::{
        _mm256_abs_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_packs_epi16,
        _mm256_permute4x64_epi64, _mm256_set1_epi8, _mm256_slli_epi16, _mm256_testz_si256,
        _mm512_abs_epi8, _mm512_castsi256_si512, _mm512_cvtsepi16_epi8, _mm512_inserti64x4,
        _mm512_loadu_si512, _mm512_movepi8_mask, _mm512_permutexvar_epi8, _mm512_set1_epi8,
        _mm512_slli_epi16, _mm512_test_epi8_mask,
    };

    use super::{EXPENSIVE_COEFF_COST, SCAN, eliminate, raster_to_scan, run_distances};

    #[target_feature(enable = "popcnt")]
    #[inline]
    fn run_cost(mask: u64) -> u32 {
        let (near4, near12, near24) = run_distances(mask);
        (mask & near4).count_ones() + (mask & near12).count_ones() + (mask & near24).count_ones()
    }

    #[target_feature(enable = "avx2,popcnt")]
    #[inline]
    fn avx2_cost(levels: &[i16; 64], _count: u32) -> u32 {
        let mut raster = 0u64;
        for (i, block) in levels.as_chunks::<32>().0.iter().enumerate() {
            // SAFETY: each chunk is exactly 32 coefficients, 16 per load.
            let (low, high) = unsafe {
                (
                    _mm256_loadu_si256(block.as_ptr().cast()),
                    _mm256_loadu_si256(block.as_ptr().add(16).cast()),
                )
            };
            let magnitudes = _mm256_abs_epi8(_mm256_packs_epi16(low, high));
            // Signed saturation preserves the classes 0, 1 and greater than 1.
            if _mm256_testz_si256(magnitudes, _mm256_set1_epi8(-2)) == 0 {
                return EXPENSIVE_COEFF_COST;
            }
            let ordered = _mm256_permute4x64_epi64::<0xd8>(magnitudes);
            let present = _mm256_movemask_epi8(_mm256_slli_epi16::<7>(ordered)) as u32;
            raster |= u64::from(present) << (32 * i);
        }
        run_cost(raster_to_scan(raster))
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,popcnt")]
    #[inline]
    fn icl_cost(levels: &[i16; 64], _count: u32) -> u32 {
        // SAFETY: the inputs and the index table occupy two complete word
        // vectors and one byte vector, respectively.
        let (low, high, scan) = unsafe {
            (
                _mm512_cvtsepi16_epi8(_mm512_loadu_si512(levels.as_ptr().cast())),
                _mm512_cvtsepi16_epi8(_mm512_loadu_si512(levels.as_ptr().add(32).cast())),
                _mm512_loadu_si512(SCAN.as_ptr().cast()),
            )
        };
        let bytes = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(low), high);
        let magnitudes = _mm512_abs_epi8(bytes);
        if _mm512_test_epi8_mask(magnitudes, _mm512_set1_epi8(-2)) != 0 {
            return EXPENSIVE_COEFF_COST;
        }
        let ordered = _mm512_permutexvar_epi8(scan, magnitudes);
        run_cost(_mm512_movepi8_mask(_mm512_slli_epi16::<7>(ordered)))
    }

    #[target_feature(enable = "avx2,popcnt")]
    pub(in crate::codec_txfm) unsafe fn avx2(
        levels: &mut [[i16; 64]; 4],
        counts: [u32; 4],
        limits: (u32, u32),
    ) -> u8 {
        eliminate(levels, counts, limits, |block, count| {
            avx2_cost(block, count)
        })
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,popcnt")]
    pub(in crate::codec_txfm) unsafe fn icl(
        levels: &mut [[i16; 64]; 4],
        counts: [u32; 4],
        limits: (u32, u32),
    ) -> u8 {
        eliminate(levels, counts, limits, |block, count| {
            icl_cost(block, count)
        })
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) use x86::{avx2, icl};

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[inline]
fn early_sparse_cost(levels: &[i16; 64], count: u32) -> Option<u32> {
    if count > 2 {
        return None;
    }
    // The first eight scan positions can only have run cost 3 or 2.
    // Stop here if all levels are present in this prefix.
    let mut left = count;
    let mut cost = 0;
    let mut next = 0;
    for (position, &index) in H264_ZIGZAG_8X8[..8].iter().enumerate() {
        match levels[index].unsigned_abs() {
            0 => {}
            1 => {
                cost += 2 + u32::from(position - next < 4);
                left -= 1;
                if left == 0 {
                    return Some(cost);
                }
                next = position + 1;
            }
            _ => return Some(EXPENSIVE_COEFF_COST),
        }
    }
    None
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use core::arch::aarch64::{
        uint8x16_t, uint8x16x4_t, vabsq_s8, vadd_u8, vaddv_u8, vcntq_u8, vcombine_u64, vcreate_u64,
        vget_high_u8, vget_lane_u64, vget_low_u8, vld1q_s16, vld1q_u8, vmaxvq_u8, vmulq_u8,
        vorrq_u8, vpadd_u8, vpaddq_u8, vqmovn_high_s16, vqmovn_s16, vqtbl4q_u8,
        vreinterpret_u64_u8, vreinterpretq_u8_s8, vreinterpretq_u8_u64, vshl_n_u8,
    };

    use super::{EXPENSIVE_COEFF_COST, SCAN, early_sparse_cost, eliminate, run_distances};

    const BIT_WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn pack(levels: *const i16) -> uint8x16_t {
        // SAFETY: each caller supplies 16 consecutive levels.
        unsafe {
            let low = vld1q_s16(levels);
            let high = vld1q_s16(levels.add(8));
            vreinterpretq_u8_s8(vabsq_s8(vqmovn_high_s16(vqmovn_s16(low), high)))
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn block_cost(levels: &[i16; 64], count: u32) -> u32 {
        if let Some(cost) = early_sparse_cost(levels, count) {
            return cost;
        }
        // SAFETY: the level and index arrays cover all four 16-lane loads,
        // and the weight table contains one full vector.
        unsafe {
            let first = pack(levels.as_ptr());
            let second = pack(levels.as_ptr().add(16));
            let third = pack(levels.as_ptr().add(32));
            let fourth = pack(levels.as_ptr().add(48));
            let combined = vorrq_u8(vorrq_u8(first, second), vorrq_u8(third, fourth));
            if vmaxvq_u8(combined) > 1 {
                return EXPENSIVE_COEFF_COST;
            }
            let bytes = uint8x16x4_t(first, second, third, fourth);
            let weights = vld1q_u8(BIT_WEIGHTS.as_ptr());
            let weighted_scan = |offset| {
                let order = vld1q_u8(SCAN.as_ptr().add(offset));
                vmulq_u8(vqtbl4q_u8(bytes, order), weights)
            };
            let low = vpaddq_u8(weighted_scan(0), weighted_scan(16));
            let high = vpaddq_u8(weighted_scan(32), weighted_scan(48));
            let pairs = vpaddq_u8(low, high);
            let mask = vpadd_u8(vget_low_u8(pairs), vget_high_u8(pairs));
            let mask = vget_lane_u64::<0>(vreinterpret_u64_u8(mask));
            let (near4, near12, near24) = run_distances(mask);
            // The nested distances give each coefficient a weight from 0
            // to 3. Count its low bit and high bit in parallel.
            let odd = mask & (near4 ^ near12 ^ near24);
            let twice = mask & near12;
            let fields = vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(odd), vcreate_u64(twice)));
            let counts = vcntq_u8(fields);
            let weighted = vadd_u8(vget_low_u8(counts), vshl_n_u8::<1>(vget_high_u8(counts)));
            u32::from(vaddv_u8(weighted))
        }
    }

    #[target_feature(enable = "neon")]
    pub(in crate::codec_txfm) unsafe fn neon(
        levels: &mut [[i16; 64]; 4],
        counts: [u32; 4],
        limits: (u32, u32),
    ) -> u8 {
        eliminate(levels, counts, limits, |block, count| {
            block_cost(block, count)
        })
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) use arm::neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{
        i8x16_abs, i8x16_bitmask, i8x16_narrow_i16x8, i8x16_splat, i16x8_shl, v128_and,
        v128_any_true, v128_load, v128_or,
    };

    use super::{
        EXPENSIVE_COEFF_COST, early_sparse_cost, eliminate, raster_to_scan, run_distances,
    };

    #[target_feature(enable = "simd128")]
    #[inline]
    fn block_cost(levels: &[i16; 64], count: u32) -> u32 {
        if let Some(cost) = early_sparse_cost(levels, count) {
            return cost;
        }
        // SAFETY: each offset covers 16 i16 levels in the block.
        let pack = |offset| unsafe {
            let src = levels.as_ptr().add(offset);
            let low = v128_load(src.cast());
            let high = v128_load(src.add(8).cast());
            i8x16_abs(i8x16_narrow_i16x8(low, high))
        };
        let first = pack(0);
        let second = pack(16);
        let third = pack(32);
        let fourth = pack(48);
        let combined = v128_or(v128_or(first, second), v128_or(third, fourth));
        if v128_any_true(v128_and(combined, i8x16_splat(-2))) {
            return EXPENSIVE_COEFF_COST;
        }
        let bits = |bytes| u64::from(i8x16_bitmask(i16x8_shl(bytes, 7)));
        let raster =
            bits(first) | (bits(second) << 16) | (bits(third) << 32) | (bits(fourth) << 48);
        let scan = raster_to_scan(raster);
        let (near4, near12, near24) = run_distances(scan);
        (scan & near4).count_ones() + (scan & near12).count_ones() + (scan & near24).count_ones()
    }

    #[target_feature(enable = "simd128")]
    pub(in crate::codec_txfm) unsafe fn simd128(
        levels: &mut [[i16; 64]; 4],
        counts: [u32; 4],
        limits: (u32, u32),
    ) -> u8 {
        eliminate(levels, counts, limits, |block, count| {
            block_cost(block, count)
        })
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) use wasm::simd128;
