//! Shared 4-tier `u16` (and batched 8/16/32-line `u8`/`u16`) HEVC deblocking
//! kernels (H.265 §8.7.2.5.3–§8.7.2.5.7) over [`I16x8Tiles`] and
//! [`I16x8Shuffles`].
//!
//! Luma decisions and short filters reuse [`super::batch::filter`] (shared
//! with VVC through [`super::short_luma`]) and [`super::samples`]: up to 11
//! bits every intermediate (`12 * 2047 + 8 = 24_572`) fits signed `i16`
//! without overflow; deeper streams fall back to [`super::scalar`]. Chroma
//! (`4 * (q0 - p0) + p1 - q1 + 4 <= 20_479` at 12 bits) runs in signed `i16`
//! up to 12 bits.

#![allow(clippy::inline_always)]

use nc_common::{BitDepth, PlaneViewMut};

use super::{HEVC_LINES, HevcEdge, batch, samples, scalar};
use crate::dispatch::hevc_edge_at;
use crate::high_depth::Edge16Fn;
use crate::simd::{I16FilterLanes, I16x8Shuffles, I16x8Tiles};
use crate::{Dsp, Isa};

/// Maximum bit depth whose luma deblocking intermediates fit signed `i16`.
const MAX_LUMA_SIMD_BITS: u32 = 11;
/// Maximum bit depth whose chroma deblocking intermediates fit signed `i16`.
const MAX_CHROMA_SIMD_BITS: u32 = 12;
/// Samples read on each side of a chroma edge (`p1, p0, q0, q1`).
const CHROMA_REACH: usize = 2;
/// Total taps across a chroma edge.
const CHROMA_TAPS: usize = 4;
/// Rounding constant and `q0 - p0` scale in `((q0 - p0) * 4 + p1 - q1 + 4) >> 3`.
const CHROMA_ROUND: i16 = 4;
/// Right shift of the chroma filter delta.
const CHROMA_SHIFT: u32 = 3;
/// Chroma lines of a single 8x8 piece (4:2:0).
const SHORT_CHROMA_LINES: usize = 4;
/// Chroma lines of two joined 8x8 pieces.
const FULL_CHROMA_LINES: usize = 8;
/// Luma positions across an edge (`p3..=q3`).
const LUMA_POSITIONS: usize = 8;
/// Maximum pieces batched in one AVX-512 call (`32 / 8`).
pub const MAX_BATCH_PIECES: usize = 4;

/// One 8-line HEVC luma edge piece without PCM/lossless exemptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HevcLumaPiece {
    /// Coordinates `(x, y)` of the piece's first `q0` sample.
    pub at: (usize, usize),
    /// `tC` of each 4-line segment (`0` leaves that segment unfiltered).
    pub tc: [i16; 2],
    /// Smoothness threshold `β`.
    pub beta: i16,
}

#[inline(always)]
fn sample_edge(e: &HevcEdge) -> samples::Edge {
    samples::Edge {
        base: e.base,
        across: e.across,
        along: e.along,
        lines: e.lines,
    }
}

/// Single 8-line `u16` HEVC luma edge on `V` (`V::LANES == 8`).
///
/// # Safety
/// `V`'s ISA must be enabled and `edge`'s 8x8 footprint must lie in `plane`.
#[inline(always)]
unsafe fn luma_edge_lanes<V: I16x8Tiles + I16x8Shuffles>(
    plane: &mut [u16],
    edge: &HevcEdge,
    max_val: i16,
) {
    let se = sample_edge(edge);
    let values = samples::load_u16::<V>(plane, &se);
    let tc = V::splat_segments(&[edge.tc[0], edge.tc[1], 0, 0, 0, 0, 0, 0]);
    if let Some(out) = batch::filter::<V, false>(values, V::splat(edge.beta), tc) {
        samples::store_u16(plane, &se, out, (1, LUMA_POSITIONS - 1), max_val);
    }
}

/// 4- or 8-line `u16` HEVC chroma edge on `V` (`V::LANES == 8`).
///
/// # Safety
/// `V`'s ISA must be enabled and `edge`'s `p1..q1` footprint must lie in `plane`.
#[inline(always)]
#[allow(clippy::similar_names)]
unsafe fn chroma_edge_lanes<V: I16x8Shuffles + I16FilterLanes>(
    plane: &mut [u16],
    edge: &HevcEdge,
    max_val: i16,
) {
    let origin = edge
        .base
        .checked_sub(CHROMA_REACH * edge.across)
        .expect("chroma deblocking sample inside the plane");
    let last = origin + (edge.lines - 1) * edge.along + (CHROMA_TAPS - 1) * edge.across;
    assert!(
        last < plane.len(),
        "chroma deblocking sample inside the plane"
    );

    let (p1, p0, q0, q1) = if edge.along == 1 {
        let mut taps = [V::splat(0); CHROMA_TAPS];
        for (k, tap) in taps.iter_mut().enumerate() {
            let at = origin + k * edge.across;
            if edge.lines == FULL_CHROMA_LINES {
                *tap = V::load_u16(&plane[at..at + FULL_CHROMA_LINES]);
            } else {
                let mut words = [0u16; FULL_CHROMA_LINES];
                words[..SHORT_CHROMA_LINES].copy_from_slice(&plane[at..at + SHORT_CHROMA_LINES]);
                *tap = V::load_u16(&words);
            }
        }
        (taps[0], taps[1], taps[2], taps[3])
    } else {
        let mut rows = [V::splat(0); FULL_CHROMA_LINES];
        for (r, row) in rows.iter_mut().enumerate().take(edge.lines) {
            let at = origin + r * edge.along;
            let mut words = [0u16; FULL_CHROMA_LINES];
            words[..CHROMA_TAPS].copy_from_slice(&plane[at..at + CHROMA_TAPS]);
            *row = V::load_u16(&words);
        }
        let cols = samples::transpose(rows);
        (cols[0], cols[1], cols[2], cols[3])
    };

    let zero = V::splat(0);
    let hi = V::splat(max_val);
    let tc = V::load(&[
        edge.tc[0], edge.tc[0], edge.tc[1], edge.tc[1], edge.tc[2], edge.tc[2], edge.tc[3],
        edge.tc[3],
    ]);
    let active = tc.gt_mask(zero);
    if !active.any_nonzero() {
        return;
    }
    let four = V::splat(CHROMA_ROUND);
    let raw = q0
        .sub(p0)
        .mul(four)
        .add(p1)
        .sub(q1)
        .add(four)
        .sra(CHROMA_SHIFT);
    let delta = raw.max(zero.sub(tc)).min(tc);
    let np0 = active.blend(p0.add(delta).max(zero).min(hi), p0);
    let nq0 = active.blend(q0.sub(delta).max(zero).min(hi), q0);

    if edge.along == 1 {
        let p_at = edge.base - edge.across;
        let q_at = edge.base;
        if edge.lines == FULL_CHROMA_LINES {
            np0.store_u16(&mut plane[p_at..p_at + FULL_CHROMA_LINES]);
            nq0.store_u16(&mut plane[q_at..q_at + FULL_CHROMA_LINES]);
        } else {
            let (mut pw, mut qw) = ([0u16; FULL_CHROMA_LINES], [0u16; FULL_CHROMA_LINES]);
            np0.store_u16(&mut pw);
            nq0.store_u16(&mut qw);
            plane[p_at..p_at + SHORT_CHROMA_LINES].copy_from_slice(&pw[..SHORT_CHROMA_LINES]);
            plane[q_at..q_at + SHORT_CHROMA_LINES].copy_from_slice(&qw[..SHORT_CHROMA_LINES]);
        }
    } else {
        let mut lo = [0u16; FULL_CHROMA_LINES];
        np0.zip_lo(nq0).store_u16(&mut lo);
        for r in 0..SHORT_CHROMA_LINES {
            let at = edge.base - 1 + r * edge.along;
            plane[at..at + 2].copy_from_slice(&lo[2 * r..2 * r + 2]);
        }
        if edge.lines == FULL_CHROMA_LINES {
            let mut hi_words = [0u16; FULL_CHROMA_LINES];
            np0.zip_hi(nq0).store_u16(&mut hi_words);
            for r in 0..SHORT_CHROMA_LINES {
                let at = edge.base - 1 + (SHORT_CHROMA_LINES + r) * edge.along;
                plane[at..at + 2].copy_from_slice(&hi_words[2 * r..2 * r + 2]);
            }
        }
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! isa_hevc_deblock {
    ($mod_name:ident, $feature:literal, $v8:ty) => {
        pub(crate) mod $mod_name {
            use nc_common::BitDepth;

            use super::super::{HevcEdge, scalar};
            use super::{
                MAX_CHROMA_SIMD_BITS, MAX_LUMA_SIMD_BITS, chroma_edge_lanes, luma_edge_lanes,
            };

            /// # Safety
            /// `$feature` must be enabled and `edge` validated inside `plane`.
            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn luma_u16(plane: &mut [u16], edge: &HevcEdge, depth: BitDepth) {
                if depth.bits() > MAX_LUMA_SIMD_BITS {
                    scalar::hevc_luma_edge_at(plane, edge, depth);
                    return;
                }
                // SAFETY: forwarded ISA and validated footprint.
                unsafe { luma_edge_lanes::<$v8>(plane, edge, depth.max() as i16) }
            }

            /// # Safety
            /// `$feature` must be enabled and `edge` validated inside `plane`.
            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn chroma_u16(plane: &mut [u16], edge: &HevcEdge, depth: BitDepth) {
                if depth.bits() > MAX_CHROMA_SIMD_BITS {
                    scalar::hevc_chroma_edge_at(plane, edge, depth);
                    return;
                }
                // SAFETY: forwarded ISA and validated footprint.
                unsafe { chroma_edge_lanes::<$v8>(plane, edge, depth.max() as i16) }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
isa_hevc_deblock!(avx2, "avx2", crate::simd::avx2::I16x8);

#[cfg(target_arch = "x86_64")]
isa_hevc_deblock!(
    avx512,
    "avx512f,avx512bw,avx512vl",
    crate::simd::avx2::I16x8
);

#[cfg(target_arch = "aarch64")]
isa_hevc_deblock!(neon, "neon", crate::simd::neon::I16x8);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
isa_hevc_deblock!(simd128, "simd128", crate::simd::simd128::I16x8);

pub(crate) fn luma_kernel(isa: Isa) -> Edge16Fn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::luma_u16,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => avx512::luma_u16,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::luma_u16,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::luma_u16,
        _ => scalar::hevc_luma_edge_at::<u16>,
    }
}

pub(crate) fn chroma_kernel(isa: Isa) -> Edge16Fn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::chroma_u16,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => avx512::chroma_u16,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::chroma_u16,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::chroma_u16,
        _ => scalar::hevc_chroma_edge_at::<u16>,
    }
}

/// Validate up to `limit` disjoint pieces starting at `pieces[cursor]`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn next_disjoint_group<T: nc_common::Sample>(
    dst: &PlaneViewMut<'_, T>,
    vertical: bool,
    pieces: &[HevcLumaPiece],
    cursor: &mut usize,
    limit: usize,
) -> ([batch::Geometry; 4], [i16; 8], [i16; 8], usize, HevcEdge) {
    let first_piece = pieces[*cursor];
    let first_edge = hevc_edge_at(
        dst,
        first_piece.at,
        vertical,
        HEVC_LINES,
        [first_piece.tc[0], first_piece.tc[1], 0, 0],
        first_piece.beta,
    );
    let init = batch::Geometry {
        base: first_edge.base,
        across: first_edge.across,
        along: first_edge.along,
    };
    let mut tiles = [init; 4];
    let mut coords = [first_piece.at; 4];
    let mut betas = [first_piece.beta; 8];
    let mut tcs = [0i16; 8];
    tcs[0] = first_piece.tc[0];
    tcs[1] = first_piece.tc[1];
    *cursor += 1;
    let mut count = 1;
    while count < limit && *cursor < pieces.len() {
        let cand = pieces[*cursor];
        let mut disjoint = true;
        for &(px, py) in &coords[..count] {
            if px.abs_diff(cand.at.0) < HEVC_LINES && py.abs_diff(cand.at.1) < HEVC_LINES {
                disjoint = false;
                break;
            }
        }
        if !disjoint {
            break;
        }
        let edge = hevc_edge_at(
            dst,
            cand.at,
            vertical,
            HEVC_LINES,
            [cand.tc[0], cand.tc[1], 0, 0],
            cand.beta,
        );
        tiles[count] = batch::Geometry {
            base: edge.base,
            across: edge.across,
            along: edge.along,
        };
        coords[count] = cand.at;
        betas[2 * count] = cand.beta;
        betas[2 * count + 1] = cand.beta;
        tcs[2 * count] = cand.tc[0];
        tcs[2 * count + 1] = cand.tc[1];
        *cursor += 1;
        count += 1;
    }
    (tiles, betas, tcs, count, first_edge)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_u8_avx2(
    data: &mut [u8],
    tiles: &[batch::Geometry; 4],
    betas: &[i16; 8],
    tcs: &[i16; 8],
    count: usize,
    first_edge: &HevcEdge,
) {
    // SAFETY: AVX2 is enabled; caller validated disjoint footprints.
    unsafe {
        if count == 2 {
            batch::short_tiles::<crate::simd::avx2::I16x16, false>(data, tiles, betas, tcs);
        } else {
            super::avx2::hevc_luma(data, first_edge);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn batch_u8_avx512(
    data: &mut [u8],
    tiles: &[batch::Geometry; 4],
    betas: &[i16; 8],
    tcs: &[i16; 8],
    count: usize,
    first_edge: &HevcEdge,
) {
    // SAFETY: AVX-512F/BW/VL enabled; caller validated disjoint footprints.
    unsafe {
        if count == 4 {
            batch::short_tiles::<crate::simd::avx512_i16::I16x32, false>(data, tiles, betas, tcs);
        } else if count >= 2 {
            batch::short_tiles::<crate::simd::avx2::I16x16, false>(data, tiles, betas, tcs);
            if count == 3 {
                let tail = HevcEdge {
                    base: tiles[2].base,
                    across: tiles[2].across,
                    along: tiles[2].along,
                    lines: HEVC_LINES,
                    tc: [tcs[4], tcs[5], 0, 0],
                    beta: betas[4],
                };
                super::avx2::hevc_luma(data, &tail);
            }
        } else {
            super::avx2::hevc_luma(data, first_edge);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_u16_avx2(
    data: &mut [u16],
    tiles: &[batch::Geometry; 4],
    betas: &[i16; 8],
    tcs: &[i16; 8],
    count: usize,
    first_edge: &HevcEdge,
    depth: BitDepth,
) {
    // SAFETY: AVX2 is enabled; caller validated disjoint footprints and depth <= 11.
    unsafe {
        if count == 2 {
            batch::short_tiles_u16::<crate::simd::avx2::I16x16, false>(
                data,
                tiles,
                betas,
                tcs,
                depth.max() as i16,
            );
        } else {
            luma_edge_lanes::<crate::simd::avx2::I16x8>(data, first_edge, depth.max() as i16);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn batch_u16_avx512(
    data: &mut [u16],
    tiles: &[batch::Geometry; 4],
    betas: &[i16; 8],
    tcs: &[i16; 8],
    count: usize,
    first_edge: &HevcEdge,
    depth: BitDepth,
) {
    let max_val = depth.max() as i16;
    // SAFETY: AVX-512F/BW/VL enabled; caller validated disjoint footprints and depth <= 11.
    unsafe {
        if count == 4 {
            batch::short_tiles_u16::<crate::simd::avx512_i16::I16x32, false>(
                data, tiles, betas, tcs, max_val,
            );
        } else if count >= 2 {
            batch::short_tiles_u16::<crate::simd::avx2::I16x16, false>(
                data, tiles, betas, tcs, max_val,
            );
            if count == 3 {
                let tail = HevcEdge {
                    base: tiles[2].base,
                    across: tiles[2].across,
                    along: tiles[2].along,
                    lines: HEVC_LINES,
                    tc: [tcs[4], tcs[5], 0, 0],
                    beta: betas[4],
                };
                luma_edge_lanes::<crate::simd::avx2::I16x8>(data, &tail, max_val);
            }
        } else {
            luma_edge_lanes::<crate::simd::avx2::I16x8>(data, first_edge, max_val);
        }
    }
}

/// Filter `pieces` on an 8-bit luma plane, batching disjoint 8-line tiles on
/// wide ISAs (`16` lines on AVX2, `32` on AVX-512).
pub(crate) fn filter_luma_pieces_u8(
    dsp: &Dsp,
    dst: &mut PlaneViewMut<'_, u8>,
    vertical: bool,
    pieces: &[HevcLumaPiece],
) {
    #[cfg(target_arch = "x86_64")]
    {
        let limit = match dsp.isa() {
            Isa::Avx512 => 4,
            Isa::Avx2 => 2,
            _ => 1,
        };
        if limit > 1 {
            let mut cursor = 0;
            while cursor < pieces.len() {
                let (tiles, betas, tcs, count, first_edge) =
                    next_disjoint_group(dst, vertical, pieces, &mut cursor, limit);
                let data = dst.data_mut();
                // SAFETY: `dsp.isa()` checked CPU support; `next_disjoint_group`
                // validated every piece's bounds and pairwise 8x8 disjointness.
                unsafe {
                    if dsp.isa() == Isa::Avx512 {
                        batch_u8_avx512(data, &tiles, &betas, &tcs, count, &first_edge);
                    } else {
                        batch_u8_avx2(data, &tiles, &betas, &tcs, count, &first_edge);
                    }
                }
            }
            return;
        }
    }
    for p in pieces {
        dsp.hevc_deblock_luma_edge_i16(dst, p.at, vertical, p.tc, p.beta);
    }
}

/// Filter `pieces` on a `u16` luma plane at `depth`, batching disjoint 8-line
/// tiles on wide ISAs (`16` lines on AVX2, `32` on AVX-512).
pub(crate) fn filter_luma_pieces_u16(
    dsp: &Dsp,
    depth: BitDepth,
    dst: &mut PlaneViewMut<'_, u16>,
    vertical: bool,
    pieces: &[HevcLumaPiece],
) {
    #[cfg(target_arch = "x86_64")]
    if depth.bits() <= MAX_LUMA_SIMD_BITS {
        let limit = match dsp.isa() {
            Isa::Avx512 => 4,
            Isa::Avx2 => 2,
            _ => 1,
        };
        if limit > 1 {
            let mut cursor = 0;
            while cursor < pieces.len() {
                let (tiles, betas, tcs, count, first_edge) =
                    next_disjoint_group(dst, vertical, pieces, &mut cursor, limit);
                let data = dst.data_mut();
                // SAFETY: `dsp.isa()` checked CPU support; `next_disjoint_group`
                // validated every piece's bounds and pairwise 8x8 disjointness.
                unsafe {
                    if dsp.isa() == Isa::Avx512 {
                        batch_u16_avx512(data, &tiles, &betas, &tcs, count, &first_edge, depth);
                    } else {
                        batch_u16_avx2(data, &tiles, &betas, &tcs, count, &first_edge, depth);
                    }
                }
            }
            return;
        }
    }
    let single = dsp.high.hevc_deblock_luma;
    for p in pieces {
        let edge = hevc_edge_at(
            dst,
            p.at,
            vertical,
            HEVC_LINES,
            [p.tc[0], p.tc[1], 0, 0],
            p.beta,
        );
        // SAFETY: `hevc_edge_at` validated `edge`; `dsp.high` matches a supported ISA.
        unsafe { single(dst.data_mut(), &edge, depth) };
    }
}
