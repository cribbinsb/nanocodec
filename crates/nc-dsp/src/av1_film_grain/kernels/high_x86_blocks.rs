//! High-depth AV1 luma from the grain template, following dav1d's two-row
//! block schedule in `filmgrain16_avx2.asm` and `filmgrain16_avx512.asm`.

use core::arch::x86_64::{
    __m512i, _mm_cvtsi32_si128, _mm256_blend_epi32, _mm256_castsi128_si256, _mm512_add_epi16,
    _mm512_adds_epu8, _mm512_and_si512, _mm512_cmp_epu16_mask, _mm512_loadu_si512,
    _mm512_maddubs_epi16, _mm512_mask_blend_epi8, _mm512_mask_set1_epi32, _mm512_maskz_mov_epi16,
    _mm512_movepi8_mask, _mm512_mullo_epi16, _mm512_permutex2var_epi8, _mm512_set1_epi16,
    _mm512_sllv_epi16, _mm512_srlv_epi16,
};
use nc_common::{PlaneView, PlaneViewMut};

use crate::av1_film_grain::kernels::blocks::{self, LumaBlock};
use crate::av1_film_grain::{
    Av1FilmGrainParams, BLOCK, GrainBlocks, Template, blend_overlap, grain_range,
};
use crate::simd::{HIGH_LOOKUP_TABLE, I16Lanes, avx2::I16x16, avx512_i16::I16x32};

use super::{HighGrainScaling, blend_row_lanes};

macro_rules! luma_plane {
    ($name:ident, $lanes:ty, $features:literal) => {
        #[target_feature(enable = $features)]
        pub(super) unsafe fn $name(
            grain: (&Av1FilmGrainParams, GrainBlocks),
            template: &Template<i16>,
            scaling: &HighGrainScaling,
            bounds: [i32; 2],
            views: (&PlaneView<'_, u16>, &mut PlaneViewMut<'_, u16>),
            size: (usize, usize),
        ) -> bool {
            if scaling.depth.bits() > 12 || !blocks::whole_blocks(grain.1, size) {
                return false;
            }
            let scales = <$lanes as GrainLanes>::prepare(scaling);
            blocks::visit(grain, template, views, size, |block| {
                // SAFETY: synthesis validated the views, and the block
                // visitor stays inside the template and its overlap rows.
                unsafe { apply_block::<$lanes>(&block, &scales, scaling.depth, bounds) }
            });
            true
        }
    };
}

luma_plane!(avx2, I16x16, "avx2");
luma_plane!(avx512, I16x32, "avx512f,avx512bw,avx512vl,avx512vbmi");

#[inline(always)]
#[allow(clippy::inline_always, reason = "inline into the selected ISA")]
unsafe fn apply_block<V: GrainLanes>(
    block: &LumaBlock<u16, i16>,
    scaling: &V::Scales<'_>,
    depth: nc_common::BitDepth,
    bounds: [i32; 2],
) {
    let (src, ss) = block.source;
    let (dst, ds) = block.output;
    let (grain, gs) = block.noise;
    let clip = bounds.map(|b| V::splat(b as i16));
    let range = grain_range(depth);
    // SAFETY: the block visitor makes all source/destination rows and
    // every overlap sample available.
    unsafe {
        let row = |p: *const i16, y: usize| &*p.add(y * gs).cast::<[i16; BLOCK]>();
        let left = |p: *const i16, y: usize, next: &mut [i16]| {
            let old = p.add(y * gs);
            for (x, value) in next.iter_mut().take(2).enumerate() {
                *value =
                    blend_overlap(i32::from(*old.add(x)), i32::from(*value), (x, 0), range) as i16;
            }
        };
        let first = if let Some((above, top_left)) = block.above {
            for y in 0..2 {
                let input = core::slice::from_raw_parts(src.add(y * ss), BLOCK);
                let output = core::slice::from_raw_parts_mut(dst.add(y * ds), BLOCK);
                let mut current = *row(grain, y);
                let mut previous = *row(above, y);
                if let Some(p) = block.left {
                    left(p, y, &mut current);
                }
                if let Some(p) = top_left {
                    left(p, y, &mut previous);
                }
                let weights = if y == 0 { [27, 17] } else { [17, 27] };
                blend_row_lanes::<V>(&previous, &mut current, weights, range);
                apply_row::<V>(input, &current, output, scaling, clip, None);
            }
            2
        } else {
            0
        };
        for y in first..block.rows {
            let input = core::slice::from_raw_parts(src.add(y * ss), BLOCK);
            let output = core::slice::from_raw_parts_mut(dst.add(y * ds), BLOCK);
            let noise = row(grain, y);
            let edge = block.left.map(|p| {
                let mut edge = [noise[0], noise[1]];
                left(p, y, &mut edge);
                edge
            });
            apply_row::<V>(input, noise, output, scaling, clip, edge);
        }
    }
}

#[inline(always)]
#[allow(clippy::inline_always, reason = "avoid outlining SIMD intrinsics")]
fn apply_row<V: GrainLanes>(
    source: &[u16],
    values: &[i16],
    output: &mut [u16],
    scaling: &V::Scales<'_>,
    [lo, hi]: [V; 2],
    edge: Option<[i16; 2]>,
) {
    for x in (0..BLOCK).step_by(V::LANES) {
        let orig = V::load_u16(&source[x..]);
        let mut noise = V::load(&values[x..]);
        if x == 0
            && let Some(edge) = edge
        {
            noise = noise.replace_first_pair(edge);
        }
        let noise = noise.mul_q15_round(orig.grain_scales(scaling));
        orig.add(noise).max(lo).min(hi).store_u16(&mut output[x..]);
    }
}

trait GrainLanes: I16Lanes {
    type Scales<'a>;
    fn prepare(scaling: &HighGrainScaling) -> Self::Scales<'_>;
    fn replace_first_pair(self, pair: [i16; 2]) -> Self;
    fn grain_scales(self, table: &Self::Scales<'_>) -> Self;
}

fn pack(pair: [i16; 2]) -> i32 {
    (u32::from(pair[0] as u16) | (u32::from(pair[1] as u16) << 16)) as i32
}

impl GrainLanes for I16x16 {
    type Scales<'a> = &'a [i16; HIGH_LOOKUP_TABLE];

    #[inline]
    fn prepare(scaling: &HighGrainScaling) -> Self::Scales<'_> {
        &scaling.q15
    }

    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn replace_first_pair(self, pair: [i16; 2]) -> Self {
        // SAFETY: the AVX2 luma entry selected these lanes.
        unsafe {
            Self(_mm256_blend_epi32::<1>(
                self.0,
                _mm256_castsi128_si256(_mm_cvtsi32_si128(pack(pair))),
            ))
        }
    }

    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn grain_scales(self, table: &Self::Scales<'_>) -> Self {
        self.lookup_u12(table)
    }
}

struct ByteScales {
    table: [__m512i; 4],
    bits: i16,
    remainder: i16,
    weight: i16,
    round: i16,
    up: i16,
    depth_max: i16,
}

impl GrainLanes for I16x32 {
    type Scales<'a> = ByteScales;

    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn prepare(scaling: &HighGrainScaling) -> ByteScales {
        let bits = scaling.depth.above_eight() as i16;
        let ptr = scaling.base.lut.as_ptr();
        // SAFETY: called from the Ice Lake entry, and the four vectors
        // cover exactly the byte scaling table.
        let table = unsafe {
            [
                _mm512_loadu_si512(ptr.cast()),
                _mm512_loadu_si512(ptr.add(64).cast()),
                _mm512_loadu_si512(ptr.add(128).cast()),
                _mm512_loadu_si512(ptr.add(192).cast()),
            ]
        };
        ByteScales {
            table,
            bits,
            remainder: (1 << bits) - 1,
            weight: 1 << bits,
            round: if bits == 0 { 0 } else { 1 << (bits - 1) },
            up: i16::from(15 - scaling.base.shift),
            depth_max: scaling.depth.max() as i16,
        }
    }

    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn replace_first_pair(self, pair: [i16; 2]) -> Self {
        // SAFETY: the Ice Lake luma entry selected these lanes.
        unsafe { Self(_mm512_mask_set1_epi32(self.0, 1, pack(pair))) }
    }

    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn grain_scales(self, scaling: &ByteScales) -> Self {
        // SAFETY: the direct luma kernel enables Ice Lake byte permutes.
        // The high-depth scaling is the linear interpolation between two
        // byte indices, rounded in the same order as `scale_lut`.
        unsafe {
            let pixels = self.max(Self::splat(0)).min(Self::splat(4095)).0;
            let bits = _mm512_set1_epi16(scaling.bits);
            let indices = _mm512_srlv_epi16(pixels, bits);
            let repeated = _mm512_mullo_epi16(indices, _mm512_set1_epi16(257));
            let indices = _mm512_adds_epu8(repeated, _mm512_set1_epi16(0x0100));
            let scales = lookup(&scaling.table, indices);
            let fraction = _mm512_and_si512(pixels, _mm512_set1_epi16(scaling.remainder));
            // 255*r + weight packs (weight-r, r) into one byte pair.
            let weight = _mm512_add_epi16(
                _mm512_mullo_epi16(fraction, _mm512_set1_epi16(255)),
                _mm512_set1_epi16(scaling.weight),
            );
            let weighted = _mm512_maddubs_epi16(scales, weight);
            let rounded = _mm512_add_epi16(weighted, _mm512_set1_epi16(scaling.round));
            let scale = _mm512_sllv_epi16(
                _mm512_srlv_epi16(rounded, bits),
                _mm512_set1_epi16(scaling.up),
            );
            let valid = _mm512_cmp_epu16_mask::<2>(pixels, _mm512_set1_epi16(scaling.depth_max));
            Self(_mm512_maskz_mov_epi16(valid, scale))
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
fn lookup(table: &[__m512i; 4], indices: __m512i) -> __m512i {
    let a = _mm512_permutex2var_epi8(table[0], indices, table[1]);
    let b = _mm512_permutex2var_epi8(table[2], indices, table[3]);
    _mm512_mask_blend_epi8(_mm512_movepi8_mask(indices), a, b)
}
