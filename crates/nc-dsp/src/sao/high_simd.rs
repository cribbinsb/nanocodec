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

trait BandSelection<V: I16FilterLanes> {
    type Table;

    fn table(offsets: [i16; 4]) -> Self::Table;
    fn select(index: V, table: &Self::Table) -> V;
}

#[cfg(target_arch = "x86_64")]
struct CompareBand;

#[cfg(target_arch = "x86_64")]
impl<V: I16FilterLanes> BandSelection<V> for CompareBand {
    type Table = [V; 4];

    #[inline(always)]
    fn table(offsets: [i16; 4]) -> [V; 4] {
        [
            V::splat(offsets[0]),
            V::splat(offsets[1]),
            V::splat(offsets[2]),
            V::splat(offsets[3]),
        ]
    }

    #[inline(always)]
    fn select(index: V, offsets: &[V; 4]) -> V {
        let mut delta = offsets[0].and(index.eq_mask(V::splat(0)));
        delta = delta.add(offsets[1].and(index.eq_mask(V::splat(1))));
        delta = delta.add(offsets[2].and(index.eq_mask(V::splat(2))));
        delta.add(offsets[3].and(index.eq_mask(V::splat(3))))
    }
}

struct LookupBand;

impl<V: SaoWordLanes> BandSelection<V> for LookupBand {
    type Table = V;

    #[inline(always)]
    fn table([a, b, c, d]: [i16; 4]) -> V {
        V::offset_table([a, b, c, d, 0, 0, 0, 0])
    }

    #[inline(always)]
    fn select(index: V, offsets: &V) -> V {
        index.min_u16(V::splat(4)).select_offset::<0>(*offsets)
    }
}

#[inline(always)]
fn band_step<V: I16FilterLanes, O: BandSelection<V>>(
    c: V,
    shift: u32,
    pos_vec: V,
    offsets: &O::Table,
    zero: V,
    max_val: V,
) -> V {
    let band = c.srl(shift).sub(pos_vec).and(V::splat(31));
    c.add_sat(O::select(band, offsets)).max(zero).min(max_val)
}

#[inline(always)]
unsafe fn band_lanes<V: I16FilterLanes, O: BandSelection<V>>(
    plane: &mut [u16],
    stride: usize,
    rect: Rect,
    position: u8,
    offsets: [i16; 4],
    exempt: &[bool],
    depth: BitDepth,
) {
    if depth.bits() >= i16::BITS {
        super::scalar::band_at(plane, stride, rect, position, offsets, exempt, depth);
        return;
    }
    let zero = V::splat(0);
    let max_val = V::splat(depth.max() as i16);
    let shift = depth.bits() - BAND_BITS;
    let pos_vec = V::splat(i16::from(position));
    let offset_vecs = O::table(offsets);
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
                    band_step::<V, O>(c, shift, pos_vec, &offset_vecs, zero, max_val)
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
                        band_step::<V, O>(c, shift, pos_vec, &offset_vecs, zero, max_val)
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

/// libhevc's packed edge-offset lookup indexed by the sum of signs.
pub(crate) struct WordOffsets;

pub(crate) trait SaoWordLanes: I16FilterLanes {
    fn offset_table(offsets: [i16; 8]) -> Self;
    fn select_offset<const BIAS: i16>(self, table: Self) -> Self;
}

impl<V: SaoWordLanes> super::generic::OffsetSelection<V> for WordOffsets {
    type Table = V;
    const ROLL_VERTICAL: bool = true;

    #[inline(always)]
    fn table([a, b, c, d]: [i16; 4]) -> V {
        V::offset_table([a, b, 0, c, d, 0, 0, 0])
    }

    #[inline(always)]
    fn adjust<T: super::generic::SaoSample>(value: V, class: V, table: &V) -> V {
        T::offset_add(value, class.select_offset::<2>(*table))
    }
}

macro_rules! isa_sao_u16 {
    ($mod_name:ident, $lanes:ty, $feature:literal, $selection:ty, $bands:ty) => {
        pub(crate) mod $mod_name {
            use nc_common::BitDepth;

            use super::super::Rect;
            use super::band_lanes;

            #[target_feature(enable = $feature)]
            #[inline(never)]
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
                    super::super::generic::apply_at::<$lanes, u16, $selection>(
                        input, in_stride, out, out_stride, offs, offsets, rect, depth,
                    );
                }
            }

            #[target_feature(enable = $feature)]
            #[inline(never)]
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
                    band_lanes::<$lanes, $bands>(
                        plane, stride, rect, position, offsets, exempt, depth,
                    );
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
isa_sao_u16!(
    avx2,
    crate::simd::avx2::I16x16,
    "avx2",
    crate::sao::high_simd::WordOffsets,
    crate::sao::high_simd::LookupBand
);

#[cfg(target_arch = "x86_64")]
isa_sao_u16!(
    avx512,
    crate::simd::avx512_i16::I16x32,
    "avx2,avx512f,avx512bw,avx512vl",
    crate::sao::generic::CompareOffsets,
    crate::sao::high_simd::CompareBand
);

#[cfg(target_arch = "aarch64")]
isa_sao_u16!(
    neon,
    crate::simd::neon::I16x8,
    "neon",
    crate::sao::high_simd::WordOffsets,
    crate::sao::high_simd::LookupBand
);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
isa_sao_u16!(
    simd128,
    crate::simd::simd128::I16x8,
    "simd128",
    crate::sao::high_simd::WordOffsets,
    crate::sao::high_simd::LookupBand
);

/// Keep full 32-lane loops separate from the 16-lane tail schedule.
///
/// # Safety
/// The caller has selected Ice Lake and checked both source neighbours
/// and the destination rectangle.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn apply_icl(
    input: &[u16],
    in_stride: usize,
    out: &mut [u16],
    out_stride: usize,
    offs: (isize, isize),
    offsets: [i16; 4],
    rect: Rect,
    depth: BitDepth,
) {
    let width = rect.x1 - rect.x0;
    let rest = width % 32;
    // SAFETY: splitting the valid rectangle preserves all of its checked
    // neighbours and each ISA is available on the selected tier.
    unsafe {
        if rest >= 16 && depth.bits() < i16::BITS {
            let middle = rect.x1 - rest;
            if middle != rect.x0 {
                avx512::apply(
                    input,
                    in_stride,
                    out,
                    out_stride,
                    offs,
                    offsets,
                    Rect { x1: middle, ..rect },
                    depth,
                );
            }
            avx2::apply(
                input,
                in_stride,
                out,
                out_stride,
                offs,
                offsets,
                Rect { x0: middle, ..rect },
                depth,
            );
        } else {
            avx512::apply(
                input, in_stride, out, out_stride, offs, offsets, rect, depth,
            );
        }
    }
}

/// Preserve the complete 16-sample rows and cropped halves on Ice Lake.
///
/// # Safety
/// The caller selected Ice Lake and validated the rectangle and exemption mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn band_icl(
    plane: &mut [u16],
    stride: usize,
    rect: Rect,
    position: u8,
    offsets: [i16; 4],
    exempt: &[bool],
    depth: BitDepth,
) {
    // SAFETY: both kernels use the same checked rectangle and exemption mask.
    unsafe {
        if (rect.x1 - rect.x0) % 32 >= 16 {
            avx2::band(plane, stride, rect, position, offsets, exempt, depth);
        } else {
            avx512::band(plane, stride, rect, position, offsets, exempt, depth);
        }
    }
}
