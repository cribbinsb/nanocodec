//! NEON AV1 grain rows. The scaling lookup is four 64-byte `tbl`/`tbx`
//! lookups per 16 samples on the byte table, as dav1d's `filmgrain.S`
//! (BSD-2-Clause, `VideoLAN` and dav1d authors; see `THIRD_PARTY.md`) scales
//! from registers, instead of a load per lane; the arithmetic is the shared
//! Q15 multiply of `generic` (exact for shifts 8..=11). Rows shorter than
//! 16 samples, and the last partial group, take the scalar reference.
#![allow(unsafe_code)]
use super::{
    Av1FilmGrainParams, Av1GrainScaling, GrainBlocks, Kernels, PlaneView, PlaneViewMut, Template,
    blocks, generic,
};
use crate::simd::neon::I16x8;
use core::arch::aarch64::{
    int8x16_t, int16x8_t, uint8x16_t, uint8x16x4_t, vaddq_s16, vaddq_s32, vcombine_u8,
    vcombine_u16, vcreate_s8, vdupq_n_s8, vdupq_n_s16, vdupq_n_s32, vdupq_n_u8, vget_high_s16,
    vget_high_u8, vget_lane_s16, vget_low_s8, vget_low_s16, vget_low_u8, vld1_s8, vld1q_s8,
    vld1q_u8, vld1q_u8_x4, vld2q_u8, vmaxq_s16, vmaxq_u8, vminq_s16, vminq_u8, vmlal_high_s8,
    vmlal_s8, vmlal_s16, vmovl_high_s8, vmovl_high_u8, vmovl_s8, vmovl_u8, vmull_high_s8, vmull_s8,
    vmull_s16, vmulq_s16, vqmovn_u16, vqmovun_high_s16, vqmovun_s16, vqmovun_s32, vqrdmulhq_s16,
    vqrshrn_high_n_s16, vqrshrn_n_s16, vqtbl4q_u8, vqtbx4q_u8, vreinterpret_s16_s8,
    vreinterpretq_s8_s16, vreinterpretq_s16_s8, vreinterpretq_s16_u16, vrhaddq_u8, vrshlq_s16,
    vsetq_lane_s16, vshlq_s16, vshrq_n_s32, vst1q_u8, vsubq_u8,
};

pub(super) const KERNELS: Kernels = Kernels {
    luma,
    chroma,
    blend,
    luma_plane: Some(luma_plane),
};

/// Samples per vector group.
const GROUP: usize = 16;
/// Bytes one `tbl` with four registers looks up.
const TABLE_QUARTER: u8 = 64;

/// `table[index]` for 16 byte indices: one 64-byte quarter of the table per
/// `tbl` / `tbx` (out-of-range indices keep the previous lookup).
#[inline]
#[target_feature(enable = "neon")]
fn lookup(table: &[uint8x16x4_t; 4], index: uint8x16_t) -> uint8x16_t {
    let quarter = vdupq_n_u8(TABLE_QUARTER);
    let mut out = vqtbl4q_u8(table[0], index);
    let mut shifted = vsubq_u8(index, quarter);
    out = vqtbx4q_u8(out, table[1], shifted);
    shifted = vsubq_u8(shifted, quarter);
    out = vqtbx4q_u8(out, table[2], shifted);
    shifted = vsubq_u8(shifted, quarter);
    vqtbx4q_u8(out, table[3], shifted)
}

/// The byte table in registers.
#[inline]
#[target_feature(enable = "neon")]
fn table(scaling: &Av1GrainScaling) -> [uint8x16x4_t; 4] {
    // SAFETY: each quarter is 64 readable bytes of the 256-byte table.
    core::array::from_fn(|q| unsafe { vld1q_u8_x4(scaling.lut[q * 64..].as_ptr()) })
}

/// Add the grain of 16 samples (`source` bytes `x`, scaled by `scales`) and
/// clip; the Q15 multiply by `scale << (15 - shift)` is AV1's round-shift.
#[inline]
#[target_feature(enable = "neon")]
fn add16(
    source: uint8x16_t,
    grain: &[i8],
    scales: uint8x16_t,
    up: int16x8_t,
    (low, high): (int16x8_t, int16x8_t),
) -> uint8x16_t {
    let grain: &[i8; GROUP] = grain[..GROUP].try_into().expect("16 grain values");
    let half = |samples: core::arch::aarch64::uint8x8_t,
                scale: core::arch::aarch64::uint8x8_t,
                g: &[i8]| {
        let scale = vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(scale)), up);
        // SAFETY: `g` holds 8 values.
        let noise = vqrdmulhq_s16(vmovl_s8(unsafe { vld1_s8(g.as_ptr()) }), scale);
        let sum = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(samples)), noise);
        vqmovun_s16(vminq_s16(vmaxq_s16(sum, low), high))
    };
    vcombine_u8(
        half(vget_low_u8(source), vget_low_u8(scales), &grain[..8]),
        half(vget_high_u8(source), vget_high_u8(scales), &grain[8..]),
    )
}

#[target_feature(enable = "neon")]
unsafe fn luma(
    source: &[u8],
    grain: &[i8],
    scaling: &Av1GrainScaling,
    bounds: [u8; 2],
    out: &mut [u8],
) {
    let table = table(scaling);
    let up = vdupq_n_s16(i16::from(15 - scaling.shift));
    let clip = (
        vdupq_n_s16(i16::from(bounds[0])),
        vdupq_n_s16(i16::from(bounds[1])),
    );
    let end = source.len() / GROUP * GROUP;
    for x in (0..end).step_by(GROUP) {
        let samples: &[u8; GROUP] = source[x..x + GROUP].try_into().expect("16 samples");
        // SAFETY: 16 readable source bytes and 16 writable output bytes.
        unsafe {
            let samples = vld1q_u8(samples.as_ptr());
            let result = add16(samples, &grain[x..], lookup(&table, samples), up, clip);
            vst1q_u8(out[x..x + GROUP].as_mut_ptr(), result);
        }
    }
    super::luma(
        &source[end..],
        &grain[end..],
        scaling,
        bounds,
        &mut out[end..],
    );
}

#[target_feature(enable = "neon")]
unsafe fn chroma(
    source: &[u8],
    luma: &[u8],
    grain: &[i8],
    scaling: &Av1GrainScaling,
    bounds: [u8; 2],
    coefficients: [i16; 3],
    out: &mut [u8],
) {
    let table = table(scaling);
    let up = vdupq_n_s16(i16::from(15 - scaling.shift));
    let clip = (
        vdupq_n_s16(i16::from(bounds[0])),
        vdupq_n_s16(i16::from(bounds[1])),
    );
    let from_luma = coefficients == [0, 64, 0];
    let [mult, luma_mult, offset] = coefficients;
    // Groups whose luma pairs are both inside the row.
    let end = source.len().min(luma.len() / 2) / GROUP * GROUP;
    for x in (0..end).step_by(GROUP) {
        let samples: &[u8; GROUP] = source[x..x + GROUP].try_into().expect("16 samples");
        let pairs: &[u8; 2 * GROUP] = luma[2 * x..2 * (x + GROUP)].try_into().expect("16 pairs");
        // SAFETY: 16 source bytes, 32 luma bytes and 16 output bytes.
        unsafe {
            let samples = vld1q_u8(samples.as_ptr());
            let pairs = vld2q_u8(pairs.as_ptr());
            // (a + b + 1) >> 1 of each pair.
            let average = vrhaddq_u8(pairs.0, pairs.1);
            let index = if from_luma {
                average
            } else {
                // ((average * luma_mult + sample * mult) >> 6) + offset in
                // 32 bits, clamped to 0..=255 by the saturating narrows.
                let wide = |a: core::arch::aarch64::uint8x8_t,
                            s: core::arch::aarch64::uint8x8_t| {
                    let a = vreinterpretq_s16_u16(vmovl_u8(a));
                    let s = vreinterpretq_s16_u16(vmovl_u8(s));
                    let part = |a: core::arch::aarch64::int16x4_t,
                                s: core::arch::aarch64::int16x4_t| {
                        let sum = vmlal_s16(
                            vmull_s16(a, core::arch::aarch64::vdup_n_s16(luma_mult)),
                            s,
                            core::arch::aarch64::vdup_n_s16(mult),
                        );
                        vqmovun_s32(vaddq_s32(
                            vshrq_n_s32::<6>(sum),
                            vdupq_n_s32(i32::from(offset)),
                        ))
                    };
                    vqmovn_u16(vcombine_u16(
                        part(vget_low_s16(a), vget_low_s16(s)),
                        part(vget_high_s16(a), vget_high_s16(s)),
                    ))
                };
                vcombine_u8(
                    wide(vget_low_u8(average), vget_low_u8(samples)),
                    wide(vget_high_u8(average), vget_high_u8(samples)),
                )
            };
            let result = add16(samples, &grain[x..], lookup(&table, index), up, clip);
            vst1q_u8(out[x..x + GROUP].as_mut_ptr(), result);
        }
    }
    if end < source.len() {
        super::chroma(
            &source[end..],
            &luma[2 * end..],
            &grain[end..],
            scaling,
            bounds,
            coefficients,
            &mut out[end..],
        );
    }
}

#[target_feature(enable = "neon")]
unsafe fn blend(old: &[i8], new: &mut [i8], weights: [i16; 2]) {
    // SAFETY: this wrapper enables the lane ISA.
    unsafe { generic::blend::<I16x8>(old, new, weights) }
}

#[target_feature(enable = "neon")]
unsafe fn luma_plane(
    (params, layout): (&Av1FilmGrainParams, GrainBlocks),
    template: &Template<i8>,
    scaling: &Av1GrainScaling,
    bounds: [u8; 2],
    views: (&PlaneView<'_>, &mut PlaneViewMut<'_>),
    size: (usize, usize),
) -> bool {
    if !blocks::whole_blocks(layout, size) {
        return false;
    }
    let scales = table(scaling);
    let shift = vdupq_n_s16(-i16::from(scaling.shift));
    let clip = bounds.map(|value| vdupq_n_u8(value));
    blocks::visit((params, layout), template, views, size, |block| {
        // SAFETY: synthesis and whole_blocks checked the views. The
        // shared block offsets keep each 32-column row within the template.
        unsafe { apply_block(block, &scales, shift, clip) };
    });
    true
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn apply_block(
    block: blocks::LumaBlock,
    table: &[uint8x16x4_t; 4],
    shift: int16x8_t,
    [lower, upper]: [uint8x16_t; 2],
) {
    let (src, ss) = block.source;
    let (dst, ds) = block.output;
    let (grain, gs) = block.noise;
    let add = |pixels: uint8x16_t, noise: int8x16_t| {
        let scales = lookup(table, pixels);
        let lo = vmulq_s16(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(scales))),
            vmovl_s8(vget_low_s8(noise)),
        );
        let hi = vmulq_s16(
            vreinterpretq_s16_u16(vmovl_high_u8(scales)),
            vmovl_high_s8(noise),
        );
        let lo = vaddq_s16(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pixels))),
            vrshlq_s16(lo, shift),
        );
        let hi = vaddq_s16(
            vreinterpretq_s16_u16(vmovl_high_u8(pixels)),
            vrshlq_s16(hi, shift),
        );
        let pixels = vqmovun_high_s16(vqmovun_s16(lo), hi);
        vminq_u8(vmaxq_u8(pixels, lower), upper)
    };
    // SAFETY: every visited row has 32 visible bytes. Overlap reads two
    // valid bytes from left and two valid 32-byte rows from above.
    unsafe {
        let write = |y, a, b| {
            let s = src.add(y * ss);
            let d = dst.add(y * ds);
            vst1q_u8(d, add(vld1q_u8(s), a));
            vst1q_u8(d.add(GROUP), add(vld1q_u8(s.add(GROUP)), b));
        };
        let first = if let Some((top, top_left)) = block.above {
            for y in 0..2 {
                let mut cur = vld1q_s8(grain.add(y * gs));
                let mut prev = vld1q_s8(top.add(y * gs));
                if let Some(left) = block.left {
                    cur = blend_left(cur, left.add(y * gs));
                }
                if let Some(left) = top_left {
                    prev = blend_left(prev, left.add(y * gs));
                }
                let w0 = vdupq_n_s8(if y == 0 { 27 } else { 17 });
                let w1 = vdupq_n_s8(if y == 0 { 17 } else { 27 });
                let a = blend_above(prev, cur, w0, w1);
                let b = blend_above(
                    vld1q_s8(top.add(y * gs + GROUP)),
                    vld1q_s8(grain.add(y * gs + GROUP)),
                    w0,
                    w1,
                );
                write(y, a, b);
            }
            2
        } else {
            0
        };
        if let Some(left) = block.left {
            for y in first..block.rows {
                let a = blend_left(vld1q_s8(grain.add(y * gs)), left.add(y * gs));
                write(y, a, vld1q_s8(grain.add(y * gs + GROUP)));
            }
        } else {
            for y in first..block.rows {
                write(
                    y,
                    vld1q_s8(grain.add(y * gs)),
                    vld1q_s8(grain.add(y * gs + GROUP)),
                );
            }
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn blend_left(current: int8x16_t, left: *const i8) -> int8x16_t {
    // SAFETY: the scheduler leaves two overlap samples from the left block.
    let old = u16::from_le(unsafe { left.cast::<u16>().read_unaligned() });
    let old = vcreate_s8(u64::from(old));
    let previous = vmull_s8(old, vcreate_s8(0x111b));
    let mixed = vmlal_s8(previous, vget_low_s8(current), vcreate_s8(0x1b11));
    let mixed = vreinterpret_s16_s8(vqrshrn_n_s16::<5>(mixed));
    vreinterpretq_s8_s16(vsetq_lane_s16::<0>(
        vget_lane_s16::<0>(mixed),
        vreinterpretq_s16_s8(current),
    ))
}

#[target_feature(enable = "neon")]
#[inline]
fn blend_above(
    old: int8x16_t,
    new: int8x16_t,
    previous: int8x16_t,
    current: int8x16_t,
) -> int8x16_t {
    let low = vmlal_s8(
        vmull_s8(vget_low_s8(old), vget_low_s8(previous)),
        vget_low_s8(new),
        vget_low_s8(current),
    );
    let high = vmlal_high_s8(vmull_high_s8(old, previous), new, current);
    vqrshrn_high_n_s16::<5>(vqrshrn_n_s16::<5>(low), high)
}
