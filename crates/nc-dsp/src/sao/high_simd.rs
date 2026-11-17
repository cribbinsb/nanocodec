//! Multi-tier `u16` SIMD SAO edge-offset and band-offset kernels over
//! [`I16FilterLanes`], shared by HEVC Main 10 (H.265 §8.7.3) and VVC Main 10
//! (H.266 §8.8.4). Edge offset is the depth-generic graph of
//! [`super::generic::apply_at`] (one implementation for 8 and 16 bits); band
//! offset is here.
//!
//! For all bit depths `8..=12`, samples lie in `0..=4095`, so differences
//! `c - a` and `c - b` fit signed `i16` without overflow, `edge(c, a, b)`
//! matches `(c - a).signum() + (c - b).signum()` lane-for-lane, and saturating
//! addition followed by clamping to `0..=depth.max()` matches `Clip1`.

#![allow(unsafe_code, clippy::inline_always)]

use nc_common::BitDepth;

use crate::simd::I16FilterLanes;

use super::Rect;
use super::scalar::band_sample_at;

const BAND_BITS: u32 = 5;

#[inline(always)]
fn band_step<V: I16FilterLanes>(
    c: V,
    shift: u32,
    pos_vec: V,
    offset_vecs: &[V; 4],
    zero: V,
    max_val: V,
) -> V {
    let band = c.srl(shift).sub(pos_vec).and(V::splat(31));
    let mut delta = offset_vecs[0].and(band.eq_mask(zero));
    delta = delta.add(offset_vecs[1].and(band.eq_mask(V::splat(1))));
    delta = delta.add(offset_vecs[2].and(band.eq_mask(V::splat(2))));
    delta = delta.add(offset_vecs[3].and(band.eq_mask(V::splat(3))));
    c.add_sat(delta).max(zero).min(max_val)
}

#[inline(always)]
unsafe fn band_lanes<V: I16FilterLanes>(
    plane: &mut [u16],
    stride: usize,
    rect: Rect,
    position: u8,
    offsets: [i16; 4],
    exempt: &[bool],
    depth: BitDepth,
) {
    let zero = V::splat(0);
    let max_val = V::splat(depth.max() as i16);
    let shift = depth.bits() - BAND_BITS;
    let pos_vec = V::splat(i16::from(position));
    let offset_vecs = [
        V::splat(offsets[0]),
        V::splat(offsets[1]),
        V::splat(offsets[2]),
        V::splat(offsets[3]),
    ];
    let width = rect.x1 - rect.x0;
    let whole = (width / V::LANES) * V::LANES;
    for y in rect.y0..rect.y1 {
        let line = &mut plane[y * stride + rect.x0..y * stride + rect.x1];
        let ptr = line.as_mut_ptr();
        if exempt.is_empty() {
            let mut x = 0;
            while x < whole {
                // SAFETY: `line` has length `width >= whole`.
                unsafe {
                    let c = V::load_u16_at(ptr.add(x));
                    band_step(c, shift, pos_vec, &offset_vecs, zero, max_val)
                        .store_u16_at(ptr.add(x));
                }
                x += V::LANES;
            }
            for sample in &mut line[whole..] {
                *sample = band_sample_at(*sample, position, offsets, depth);
            }
        } else {
            let row_exempt = &exempt[(y - rect.y0) * width..(y - rect.y0 + 1) * width];
            let mut x = 0;
            while x < whole {
                if row_exempt[x..x + V::LANES].contains(&true) {
                    for dx in 0..V::LANES {
                        if !row_exempt[x + dx] {
                            line[x + dx] = band_sample_at(line[x + dx], position, offsets, depth);
                        }
                    }
                } else {
                    // SAFETY: `line` has length `width >= whole`.
                    unsafe {
                        let c = V::load_u16_at(ptr.add(x));
                        band_step(c, shift, pos_vec, &offset_vecs, zero, max_val)
                            .store_u16_at(ptr.add(x));
                    }
                }
                x += V::LANES;
            }
            for dx in whole..width {
                if !row_exempt[dx] {
                    line[dx] = band_sample_at(line[dx], position, offsets, depth);
                }
            }
        }
    }
}

macro_rules! isa_sao_u16 {
    ($mod_name:ident, $lanes:ty, $feature:literal) => {
        pub(crate) mod $mod_name {
            use nc_common::BitDepth;

            use super::super::Rect;
            use super::band_lanes;

            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn apply(
                input: &[u16],
                in_stride: usize,
                out: &mut [u16],
                out_stride: usize,
                offs: (isize, isize),
                offsets: [i16; 4],
                rect: Rect,
                depth: BitDepth,
            ) {
                // SAFETY: caller enables `$feature` and checked rectangle/neighbours.
                unsafe {
                    super::super::generic::apply_at::<$lanes, u16>(
                        input, in_stride, out, out_stride, offs, offsets, rect, depth,
                    );
                }
            }

            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn band(
                plane: &mut [u16],
                stride: usize,
                rect: Rect,
                position: u8,
                offsets: [i16; 4],
                exempt: &[bool],
                depth: BitDepth,
            ) {
                // SAFETY: caller enables `$feature` and checked rectangle/mask.
                unsafe {
                    band_lanes::<$lanes>(plane, stride, rect, position, offsets, exempt, depth);
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
isa_sao_u16!(avx2, crate::simd::avx2::I16x16, "avx2");

#[cfg(target_arch = "x86_64")]
isa_sao_u16!(
    avx512,
    crate::simd::avx512_i16::I16x32,
    "avx512f,avx512bw,avx512vl"
);

#[cfg(target_arch = "aarch64")]
isa_sao_u16!(neon, crate::simd::neon::I16x8, "neon");

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
isa_sao_u16!(simd128, crate::simd::simd128::I16x8, "simd128");
