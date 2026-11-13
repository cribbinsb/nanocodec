//! AVX-512F/BW/VL motion costs. A 64-byte row supplies eight independent
//! 8x8 SAD columns; four candidates reuse each source load. Hierarchy
//! aggregation stays in `grid_costs`, and masked strict-less updates retain
//! the original candidate on ties, including the five-element final group.
//!
//! Layout ideas: SVT-AV1 v4.3.0 `compute_sad_intrin_avx512.c` (Intel,
//! BSD-2-Clause/AOM patent licence) and libaom v3.15.3 `sad_hwy.h`
//! (Alliance for Open Media, BSD-2-Clause/AOM patent licence). Reimplemented
//! with Rust intrinsics and our existing block/visibility contracts.

use super::{GRID_BLOCKS, GridBest, PixelFns};
use core::arch::x86_64::{
    _mm_loadu_si128, _mm256_storeu_si256, _mm512_add_epi64, _mm512_broadcast_i32x4,
    _mm512_cmplt_epu32_mask, _mm512_cvtepi64_epi32, _mm512_inserti32x4, _mm512_loadu_si512,
    _mm512_mask_storeu_epi32, _mm512_maskz_loadu_epi32, _mm512_reduce_add_epi64, _mm512_sad_epu8,
    _mm512_set1_epi32, _mm512_setzero_si512, _mm512_storeu_si512, _mm512_zextsi128_si512,
};

/// Override only implemented kernels, retaining AVX2 for narrower shapes.
pub(crate) fn install(table: &mut PixelFns) {
    table.satd_8x8 = super::avx512_satd::tiles;
    table.satd_8x8_quads = Some(super::avx512_satd::quads);
    table.sad[4] = sad64;
    // Hardware timing: 8-wide x4 loses on SPR (0.63x); retain AVX2.
    table.sad_x4[0] = sad_narrow_x4::<4>;
    table.sad_x4[2..].copy_from_slice(&[sad_narrow_x4::<16>, sad32_x4, sad64_x4]);
    table.sad_grid_best_64x64 = [grid::<1>, grid::<2>];
    table.sad_grid_best_x4 = Some([grid_x4::<1>, grid_x4::<2>]);
    // Keep AVX2 quads: measured AVX-512 speedup was only 0.55x.
    table.sad_scan_16x16_even_rows = scan;
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn sad64(src: &[u8], ss: usize, rf: &[u8], rs: usize, h: usize) -> u32 {
    let mut sum = _mm512_setzero_si512();
    for y in 0..h {
        // SAFETY: the dispatcher validates all 64 bytes of each row.
        let (a, b) = unsafe {
            (
                _mm512_loadu_si512(src.as_ptr().add(y * ss).cast()),
                _mm512_loadu_si512(rf.as_ptr().add(y * rs).cast()),
            )
        };
        sum = _mm512_add_epi64(sum, _mm512_sad_epu8(a, b));
    }
    _mm512_reduce_add_epi64(sum) as u32
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn sad64_x4(src: &[u8], ss: usize, refs: [&[u8]; 4], rs: usize, h: usize) -> [u32; 4] {
    let mut sums = [_mm512_setzero_si512(); 4];
    for y in 0..h {
        // SAFETY: source and all references cover 64 bytes per row.
        let a = unsafe { _mm512_loadu_si512(src.as_ptr().add(y * ss).cast()) };
        for (sum, rf) in sums.iter_mut().zip(refs) {
            // SAFETY: as above, for this reference.
            let b = unsafe { _mm512_loadu_si512(rf.as_ptr().add(y * rs).cast()) };
            *sum = _mm512_add_epi64(*sum, _mm512_sad_epu8(a, b));
        }
    }
    let mut out = [0; 4];
    for (dst, sum) in out.iter_mut().zip(sums) {
        *dst = _mm512_reduce_add_epi64(sum) as u32;
    }
    out
}

/// Produce one or four grids in source-row order. Each SAD lane is at most
/// 8*8*255, so narrowing the completed 64-bit sums to u32 is exact.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn grids<const STEP: usize, const C: usize>(
    src: &[u8],
    ss: usize,
    refs: [&[u8]; C],
    rs: usize,
) -> [[u32; 64]; C] {
    let mut result = [[0; 64]; C];
    for by in 0..8 {
        let mut sums = [_mm512_setzero_si512(); C];
        for y in (by * 8..by * 8 + 8).step_by(STEP) {
            // SAFETY: both blocks cover the complete 64x64 grid.
            let a = unsafe { _mm512_loadu_si512(src.as_ptr().add(y * ss).cast()) };
            for (sum, rf) in sums.iter_mut().zip(refs) {
                // SAFETY: each reference covers row y's 64 bytes.
                let b = unsafe { _mm512_loadu_si512(rf.as_ptr().add(y * rs).cast()) };
                *sum = _mm512_add_epi64(*sum, _mm512_sad_epu8(a, b));
            }
        }
        for (dst, sum) in result.iter_mut().zip(sums) {
            // SAFETY: eight u32 outputs at the start of this block row.
            unsafe {
                _mm256_storeu_si256(
                    dst.as_mut_ptr().add(by * 8).cast(),
                    _mm512_cvtepi64_epi32(sum),
                );
            };
        }
    }
    result
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn grid<const STEP: usize>(
    src: &[u8],
    ss: usize,
    rf: &[u8],
    rs: usize,
    visible: &[u32; 64],
    best: &mut GridBest,
    tag: u32,
) {
    // SAFETY: the dispatcher validates both grids and this ISA.
    let [grid] = unsafe { grids::<STEP, 1>(src, ss, [rf], rs) };
    keep_min(best, &super::grid_costs(&grid, visible, STEP as u32), tag);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn grid_x4<const STEP: usize>(
    src: &[u8],
    ss: usize,
    refs: [&[u8]; 4],
    rs: usize,
    visible: &[u32; 64],
    best: &mut GridBest,
    tags: [u32; 4],
) {
    // SAFETY: the dispatcher validates all grids and this ISA.
    let costs = unsafe { grids::<STEP, 4>(src, ss, refs, rs) };
    // Fold in exactly the single-candidate order. No winner selection is
    // reordered across candidates; lowest index wins equal costs.
    for (grid, tag) in costs.iter().zip(tags) {
        keep_min(best, &super::grid_costs(grid, visible, STEP as u32), tag);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn keep_min(best: &mut GridBest, cost: &[u32; GRID_BLOCKS], tag: u32) {
    const LANES: usize = 16;
    for i in (0..GRID_BLOCKS).step_by(LANES) {
        let mask = (u32::MAX >> (32 - (GRID_BLOCKS - i).min(LANES))) as u16;
        // SAFETY: the mask covers only the remaining elements of all arrays.
        unsafe {
            let old = _mm512_maskz_loadu_epi32(mask, best.sad.as_ptr().add(i).cast());
            let new = _mm512_maskz_loadu_epi32(mask, cost.as_ptr().add(i).cast());
            let better = _mm512_cmplt_epu32_mask(new, old) & mask;
            _mm512_mask_storeu_epi32(best.sad.as_mut_ptr().add(i).cast(), better, new);
            _mm512_mask_storeu_epi32(
                best.tag.as_mut_ptr().add(i).cast(),
                better,
                _mm512_set1_epi32(tag as i32),
            );
        }
    }
}

/// Four adjacent 16x16 candidates occupy the four 128-bit sublanes.
/// The source row is broadcast once, then shared by all candidates.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn scan(src: &[u8], ss: usize, rf: &[u8], rs: usize, out: &mut [u32]) {
    let mut source = [_mm512_setzero_si512(); 8];
    for (y, a) in source.iter_mut().enumerate() {
        // SAFETY: row 2y contains 16 source pixels.
        *a =
            unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(src.as_ptr().add(2 * y * ss).cast())) };
    }
    for (batch, dst) in out.chunks_mut(4).enumerate() {
        let mut sum = _mm512_setzero_si512();
        for (y, &a) in source.iter().enumerate() {
            // SAFETY: candidate zero always exists in a nonempty chunk;
            // other row loads are guarded by that chunk's length.
            let b = unsafe {
                let p = rf.as_ptr().add(2 * y * rs + 4 * batch);
                let mut b = _mm512_zextsi128_si512(_mm_loadu_si128(p.cast()));
                if dst.len() > 1 {
                    b = _mm512_inserti32x4::<1>(b, _mm_loadu_si128(p.add(1).cast()));
                }
                if dst.len() > 2 {
                    b = _mm512_inserti32x4::<2>(b, _mm_loadu_si128(p.add(2).cast()));
                }
                if dst.len() > 3 {
                    b = _mm512_inserti32x4::<3>(b, _mm_loadu_si128(p.add(3).cast()));
                }
                b
            };
            sum = _mm512_add_epi64(sum, _mm512_sad_epu8(a, b));
        }
        let mut lanes = [0u64; 8];
        // SAFETY: lanes contains 64 writable bytes.
        unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast(), sum) };
        for (i, value) in dst.iter_mut().enumerate() {
            *value = (lanes[2 * i] + lanes[2 * i + 1]) as u32;
        }
    }
}

/// Four narrow candidates occupy independent 128-bit sublanes. Masked
/// row loads read exactly W bytes, including 4/8-wide blocks at allocation
/// boundaries; zero padding contributes no SAD.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn sad_narrow_x4<const W: usize>(
    src: &[u8],
    ss: usize,
    refs: [&[u8]; 4],
    rs: usize,
    h: usize,
) -> [u32; 4] {
    let mask = (u32::MAX >> (32 - W)) as u16;
    let mut sum = _mm512_setzero_si512();
    for y in 0..h {
        // SAFETY: the mask reads W bytes from each validated source row.
        let a = unsafe {
            _mm512_broadcast_i32x4(core::arch::x86_64::_mm_maskz_loadu_epi8(
                mask,
                src.as_ptr().add(y * ss).cast(),
            ))
        };
        // SAFETY: every reference has W valid bytes on row y; masks omit padding.
        let b = unsafe {
            let b = _mm512_zextsi128_si512(core::arch::x86_64::_mm_maskz_loadu_epi8(
                mask,
                refs[0].as_ptr().add(y * rs).cast(),
            ));
            let b = _mm512_inserti32x4::<1>(
                b,
                core::arch::x86_64::_mm_maskz_loadu_epi8(mask, refs[1].as_ptr().add(y * rs).cast()),
            );
            let b = _mm512_inserti32x4::<2>(
                b,
                core::arch::x86_64::_mm_maskz_loadu_epi8(mask, refs[2].as_ptr().add(y * rs).cast()),
            );
            _mm512_inserti32x4::<3>(
                b,
                core::arch::x86_64::_mm_maskz_loadu_epi8(mask, refs[3].as_ptr().add(y * rs).cast()),
            )
        };
        sum = _mm512_add_epi64(sum, _mm512_sad_epu8(a, b));
    }
    let mut lanes = [0u64; 8];
    // SAFETY: eight writable u64 lanes.
    unsafe {
        _mm512_storeu_si512(lanes.as_mut_ptr().cast(), sum);
    }
    [
        (lanes[0] + lanes[1]) as u32,
        (lanes[2] + lanes[3]) as u32,
        (lanes[4] + lanes[5]) as u32,
        (lanes[6] + lanes[7]) as u32,
    ]
}

/// Two width-32 candidates per register, with a shared source broadcast.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn sad32_x4(src: &[u8], ss: usize, refs: [&[u8]; 4], rs: usize, h: usize) -> [u32; 4] {
    use core::arch::x86_64::{
        _mm256_loadu_si256, _mm512_broadcast_i64x4, _mm512_inserti64x4, _mm512_zextsi256_si512,
    };
    let mut sums = [_mm512_setzero_si512(); 2];
    for y in 0..h {
        // SAFETY: 32 readable source bytes per row.
        let a =
            unsafe { _mm512_broadcast_i64x4(_mm256_loadu_si256(src.as_ptr().add(y * ss).cast())) };
        for (pair, sum) in sums.iter_mut().enumerate() {
            // SAFETY: both references cover 32 readable bytes per row.
            let b = unsafe {
                _mm512_inserti64x4::<1>(
                    _mm512_zextsi256_si512(_mm256_loadu_si256(
                        refs[2 * pair].as_ptr().add(y * rs).cast(),
                    )),
                    _mm256_loadu_si256(refs[2 * pair + 1].as_ptr().add(y * rs).cast()),
                )
            };
            *sum = _mm512_add_epi64(*sum, _mm512_sad_epu8(a, b));
        }
    }
    let mut out = [0; 4];
    for (pair, sum) in sums.into_iter().enumerate() {
        let mut lanes = [0u64; 8];
        // SAFETY: eight writable u64 lanes.
        unsafe {
            _mm512_storeu_si512(lanes.as_mut_ptr().cast(), sum);
        }
        out[2 * pair] = lanes[..4].iter().sum::<u64>() as u32;
        out[2 * pair + 1] = lanes[4..].iter().sum::<u64>() as u32;
    }
    out
}
