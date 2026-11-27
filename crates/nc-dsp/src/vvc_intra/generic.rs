//! H.266 planar/DC/axis prediction (§8.4.5.2.11-12, §8.4.5.2.15): the base
//! prediction, then PDPC. Planar is the shared HEVC planar kernel
//! (`crate::hevc_intra::lanes::planar`: H.266 planar equals H.265's, eq.
//! 8-48); DC and the axes are fills and copies. PDPC then runs over the
//! rows and columns its weights reach, on i16 lanes: each term is at most
//! `32 · 255` in magnitude and their sum below 2^15, so the 16-bit
//! arithmetic is exact.
#![allow(unsafe_code, clippy::inline_always)]
use super::VvcIntraMode;
use crate::hevc_intra::PlanarEdges;
use crate::simd::I16Lanes;

/// Largest side of the square kernels.
pub(super) const MAX_SIDE: usize = 32;
/// PDPC weights are `32 >> ((2 d) >> scale)`: zero from `d = 3 << scale`.
const PDPC_REACH: usize = 3;
/// PDPC rounding: `(w · d + 32) >> 6`.
const PDPC_SHIFT: u32 = 6;

/// # Safety
/// Called only with V's ISA enabled and a supported side divisible by its
/// lanes, no wider than CAPACITY.
#[inline(always)]
pub(super) unsafe fn predict<V: I16Lanes, const CAPACITY: usize>(
    n: usize,
    mode: VvcIntraMode,
    top: &[i32],
    left: &[i32],
    corner: i32,
    out: &mut [u8],
) {
    let mode = match mode {
        VvcIntraMode::Angular { direction: 18, .. } => VvcIntraMode::Horizontal,
        VvcIntraMode::Angular { direction: 50, .. } => VvcIntraMode::Vertical,
        mode => mode,
    };
    // References are 8-bit samples (`check_square`): narrowing is exact.
    let mut above = [0u8; CAPACITY];
    let mut side = [0u8; CAPACITY];
    for (narrow, &wide) in above.iter_mut().zip(&top[..n]) {
        *narrow = wide as u8;
    }
    for (narrow, &wide) in side.iter_mut().zip(&left[..n]) {
        *narrow = wide as u8;
    }
    let out = &mut out[..n * n];
    match mode {
        VvcIntraMode::Angular { .. } => unreachable!("angular directions use angular_generic"),
        VvcIntraMode::Planar => {
            let edges = PlanarEdges {
                above: &above[..n],
                left: &side[..n],
                top_right: top[n] as u8,
                bottom_left: left[n] as u8,
            };
            // SAFETY: forwarded contract.
            unsafe { crate::hevc_intra::lanes::planar::<V, CAPACITY>(&edges, n, out) };
        }
        VvcIntraMode::Dc => {
            let sum: u32 = above[..n]
                .iter()
                .chain(&side[..n])
                .map(|&v| u32::from(v))
                .sum();
            out.fill(((sum + n as u32) / (2 * n as u32)) as u8);
        }
        VvcIntraMode::Vertical => {
            for line in out.chunks_exact_mut(n) {
                line.copy_from_slice(&above[..n]);
            }
        }
        VvcIntraMode::Horizontal => {
            for (line, &value) in out.chunks_exact_mut(n).zip(&side[..n]) {
                line.fill(value);
            }
        }
    }
    // SAFETY: forwarded contract.
    unsafe { pdpc::<V, CAPACITY>(n, mode, &above, &side, corner as i16, out) };
}

/// PDPC (§8.4.5.2.15) over the predicted `out`: `v + ((wL · (left − v) +
/// wT · (top − v) + 32) >> 6)` for planar and DC, `v + ((wL · (left −
/// corner) + 32) >> 6)` for vertical, `v + ((wT · (top − corner) + 32) >> 6)`
/// for horizontal, where the weights reach (`PDPC_REACH << scale`).
///
/// # Safety
/// V's ISA is enabled and its lane count divides `n`.
#[inline(always)]
unsafe fn pdpc<V: I16Lanes, const CAPACITY: usize>(
    n: usize,
    mode: VvcIntraMode,
    above: &[u8],
    side: &[u8],
    corner: i16,
    out: &mut [u8],
) {
    let scale = (2 * n.ilog2() - 2) >> 2;
    let reach = (PDPC_REACH << scale).min(n);
    let mut weights = [0i16; CAPACITY];
    for (d, weight) in weights[..reach].iter_mut().enumerate() {
        *weight = 32 >> ((2 * d) >> scale);
    }
    let mut top = [0i16; CAPACITY];
    for (wide, &narrow) in top.iter_mut().zip(&above[..n]) {
        *wide = i16::from(narrow);
    }
    let round = V::splat(1 << (PDPC_SHIFT - 1));
    // Rows past the reach have a zero top weight: only the left term's
    // columns change (none for horizontal).
    let reached_columns = reach.div_ceil(V::LANES) * V::LANES;
    for (y, line) in out.chunks_exact_mut(n).enumerate() {
        let wt = V::splat(weights.get(y).copied().unwrap_or(0));
        let left = V::splat(i16::from(side[y]));
        let columns = match mode {
            VvcIntraMode::Horizontal if y >= reach => 0,
            _ if y >= reach => reached_columns,
            _ => n,
        };
        for x in (0..columns).step_by(V::LANES) {
            let value = V::load_u8(&line[x..]);
            let wl = V::load(&weights[x..]);
            let above = V::load(&top[x..]);
            let correction = match mode {
                VvcIntraMode::Angular { .. } => {
                    unreachable!("angular directions use angular_generic")
                }
                VvcIntraMode::Planar | VvcIntraMode::Dc => {
                    wl.mul(left.sub(value)).add(wt.mul(above.sub(value)))
                }
                VvcIntraMode::Horizontal => wt.mul(above.sub(V::splat(corner))),
                VvcIntraMode::Vertical => wl.mul(left.sub(V::splat(corner))),
            };
            value
                .add(correction.add(round).sra(PDPC_SHIFT))
                .store_u8_sat(&mut line[x..x + V::LANES]);
        }
    }
}

/// The ISA's `predict`: angular directions through the ISA's shared runs
/// (`$runs`), every mode's arithmetic on i16 lanes as wide as the block
/// side allows (`$a4`, `$a8`, `$a16`).
macro_rules! kernels {
    ($(#[$attr:meta])*; $runs:ty; $a4:ty, $a8:ty, $a16:ty, $wide:ty) => {
        $(#[$attr])*
        pub(super) unsafe fn predict(n: usize, mode: super::VvcIntraMode, top: &[i32], left: &[i32], corner: i32, transposed_ok: bool, out: &mut [u8]) -> bool {
            use super::VvcIntraMode::Angular;
            // SAFETY: target feature enabled; every supported side is a
            // multiple of four and a lane count of at most the side is used.
            unsafe {
                match mode {
                    Angular { direction, luma } if direction != 18 && direction != 50 => match n {
                        4 => super::angular_generic::predict::<$a4, $runs>(n, direction, luma, top, left, corner, transposed_ok, out),
                        8 => super::angular_generic::predict::<$a8, $runs>(n, direction, luma, top, left, corner, transposed_ok, out),
                        _ => super::angular_generic::predict::<$a16, $runs>(n, direction, luma, top, left, corner, transposed_ok, out),
                    },
                    _ => {
                        match n {
                            4 => super::generic::predict::<$a4, { super::generic::MAX_SIDE }>(n, mode, top, left, corner, out),
                            8 => super::generic::predict::<$a8, { super::generic::MAX_SIDE }>(n, mode, top, left, corner, out),
                            64 => super::generic::predict::<$wide, 64>(64, mode, top, left, corner, out),
                            _ => super::generic::predict::<$a16, { super::generic::MAX_SIDE }>(n, mode, top, left, corner, out),
                        }
                        false
                    }
                }
            }
        }
    };
}
pub(crate) use kernels;
