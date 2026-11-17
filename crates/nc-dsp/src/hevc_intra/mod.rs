//! HEVC intra prediction kernels (HEVC §8.4.4.2.4, §8.4.4.2.6): planar, and
//! the line projection of the angular modes.
//!
//! The codec gathers and filters the references and builds the angular
//! modes' main reference array (extended with projected side samples for
//! negative angles); the kernels compute the samples. An angular line `j`
//! (a row for vertical modes, a column written as a row for horizontal
//! ones) is the 2-tap blend
//! `((32 − f) · ref[i + 1] + f · ref[i + 2] + 16) >> 5` of the main
//! reference at a fixed offset, so each line is one call of the shared
//! directional run (`crate::intra_runs`, also AV1's and AV2's), through
//! [`angular_lines`]. [`reference`] is the scalar oracle.

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
pub(crate) mod lanes;
pub(crate) mod reference;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128;

use crate::Isa;

/// Largest HEVC transform (and intra prediction) side.
pub(crate) const MAX_SIDE: usize = 32;
/// Fractional bits of an angular position (1/32 sample).
pub(crate) const ANGLE_FRACTION_BITS: u32 = 5;
/// Largest angle magnitude (HEVC Table 8-5: intraPredAngle in −32..=32).
pub(crate) const MAX_ANGLE: i32 = 32;

/// Angular lines of an `n`x`n` block: `refs[n + i]` is the main reference
/// `ref[i]` for `i` in `−n..=2n` (`refs.len() == 3n + 1`); `out` gets `n`
/// lines of `n` samples (`angle` in −32..=32).
pub(crate) type AngularFn = unsafe fn(refs: &[u8], n: usize, angle: i32, out: &mut [u8]);

/// Planar prediction of an `n`x`n` block from the `n` samples above, the `n`
/// to the left, and the top-right and bottom-left samples.
pub(crate) type PlanarFn = unsafe fn(edges: &PlanarEdges<'_>, n: usize, out: &mut [u8]);

/// The references planar prediction reads (§8.4.4.2.4).
/// Samples are `u8`, or `u16` for more than 8 bits.
#[derive(Debug, Clone, Copy)]
pub struct PlanarEdges<'a, T: nc_common::Sample = u8> {
    /// `p[x][−1]` for `x` in `0..n`.
    pub above: &'a [T],
    /// `p[−1][y]` for `y` in `0..n`.
    pub left: &'a [T],
    /// `p[n][−1]`.
    pub top_right: T,
    /// `p[−1][n]`.
    pub bottom_left: T,
}

/// Angular lines ([`AngularFn`]) with `R`'s [`crate::intra_runs`] run:
/// each line's blend of the main reference at its offset. A
/// whole-sample line (`f = 0`) runs at weight 32 from one sample earlier,
/// which copies it: for the steepest angle the sample after the last
/// line's run is past `ref[2n]`, so a weight-0 window would overrun.
///
/// # Safety
///
/// `R`'s ISA must be enabled in the caller.
#[allow(clippy::inline_always, unsafe_code)] // must inline into `#[target_feature]` kernels
#[inline(always)]
pub(crate) unsafe fn angular_lines<R: crate::intra_runs::Runs>(
    refs: &[u8],
    n: usize,
    angle: i32,
    out: &mut [u8],
) {
    // SAFETY: caller contract.
    unsafe {
        match n {
            4 => lines::<R, 4>(refs, angle, out),
            8 => lines::<R, 8>(refs, angle, out),
            16 => lines::<R, 16>(refs, angle, out),
            _ => lines::<R, MAX_SIDE>(refs, angle, out),
        }
    }
}

/// [`angular_lines`] of an `N`x`N` block: each line a fixed-length run.
///
/// # Safety
///
/// As [`angular_lines`].
#[allow(clippy::inline_always, unsafe_code)] // must inline into `#[target_feature]` kernels
#[inline(always)]
unsafe fn lines<R: crate::intra_runs::Runs, const N: usize>(
    refs: &[u8],
    angle: i32,
    out: &mut [u8],
) {
    let one = 1 << ANGLE_FRACTION_BITS;
    for (j, line) in out[..N * N].as_chunks_mut::<N>().0.iter_mut().enumerate() {
        let pos = (j as i32 + 1) * angle;
        let (idx, fact) = (pos >> ANGLE_FRACTION_BITS, pos & (one - 1));
        let start = (N as i32 + idx + 1) as usize;
        // SAFETY: caller contract; the windows hold N + 1 samples.
        unsafe {
            if fact == 0 {
                R::run2_fixed::<N>(&refs[start - 1..start + N], one as usize, line);
            } else {
                R::run2_fixed::<N>(&refs[start..=start + N], fact as usize, line);
            }
        }
    }
}

/// One ISA's HEVC intra kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HevcIntraFns {
    pub angular: AngularFn,
    pub planar: PlanarFn,
}

/// The kernels for `isa` (the scalar reference where it has none).
pub(crate) fn kernels(isa: Isa) -> HevcIntraFns {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => avx2::KERNELS,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::KERNELS,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::KERNELS,
        _ => reference::KERNELS,
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code, clippy::inline_always)]
pub(crate) mod high_simd {
    use super::{ANGLE_FRACTION_BITS, MAX_SIDE, PlanarEdges, lanes};
    use crate::intra_runs::high::run2_u16;
    use crate::simd::I16Lanes;

    #[inline(always)]
    unsafe fn angular_u16<A4: I16Lanes, A8: I16Lanes, A16: I16Lanes, A32: I16Lanes>(
        refs: &[u16],
        n: usize,
        angle: i32,
        out: &mut [u16],
    ) {
        let one = 1 << ANGLE_FRACTION_BITS;
        for (j, line) in out[..n * n].chunks_exact_mut(n).enumerate() {
            let pos = (j as i32 + 1) * angle;
            let (idx, fact) = (pos >> ANGLE_FRACTION_BITS, pos & (one - 1));
            let start = (n as i32 + idx + 1) as usize;
            let (win, w) = if fact == 0 {
                (&refs[start - 1..start + n], one as usize)
            } else {
                (&refs[start..=start + n], fact as usize)
            };
            // SAFETY: caller enables lane ISAs; `n` is 4, 8, 16, or 32.
            unsafe {
                match n {
                    4 => run2_u16::<A4>(win, w, line),
                    8 => run2_u16::<A8>(win, w, line),
                    16 => run2_u16::<A16>(win, w, line),
                    _ => run2_u16::<A32>(win, w, &mut line[..MAX_SIDE]),
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn planar_dispatch<A4: I16Lanes, A8: I16Lanes, A16: I16Lanes, A32: I16Lanes>(
        edges: &PlanarEdges<'_, u16>,
        n: usize,
        out: &mut [u16],
    ) {
        // SAFETY: caller enables lane ISAs; `n` is 4, 8, 16, or 32.
        unsafe {
            match n {
                4 => lanes::planar_u16::<A4>(edges, n, out),
                8 => lanes::planar_u16::<A8>(edges, n, out),
                16 => lanes::planar_u16::<A16>(edges, n, out),
                _ => lanes::planar_u16::<A32>(edges, n, out),
            }
        }
    }

    macro_rules! isa_hevc_intra_u16 {
        ($mod_name:ident, $a4:ty, $a8:ty, $a16:ty, $a32:ty, $feature:literal) => {
            pub(crate) mod $mod_name {
                use super::super::PlanarEdges;
                use super::{angular_u16, planar_dispatch};

                #[target_feature(enable = $feature)]
                pub(crate) unsafe fn angular(refs: &[u16], n: usize, angle: i32, out: &mut [u16]) {
                    // SAFETY: `$feature` enabled; caller validated `refs`, `n`, `out`.
                    unsafe { angular_u16::<$a4, $a8, $a16, $a32>(refs, n, angle, out) }
                }

                #[target_feature(enable = $feature)]
                pub(crate) unsafe fn planar(
                    edges: &PlanarEdges<'_, u16>,
                    n: usize,
                    out: &mut [u16],
                ) {
                    // SAFETY: `$feature` enabled; caller validated `edges`, `n`, `out`.
                    unsafe { planar_dispatch::<$a4, $a8, $a16, $a32>(edges, n, out) }
                }
            }
        };
    }

    #[cfg(target_arch = "x86_64")]
    isa_hevc_intra_u16!(
        avx2,
        crate::simd::avx2::I16x4,
        crate::simd::avx2::I16x8,
        crate::simd::avx2::I16x16,
        crate::simd::avx2::I16x16,
        "avx2"
    );
    #[cfg(target_arch = "x86_64")]
    isa_hevc_intra_u16!(
        avx512,
        crate::simd::avx2::I16x4,
        crate::simd::avx2::I16x8,
        crate::simd::avx2::I16x16,
        crate::simd::avx512_i16::I16x32,
        "avx512f,avx512bw,avx512vl"
    );
    #[cfg(target_arch = "aarch64")]
    isa_hevc_intra_u16!(
        neon,
        crate::simd::neon::I16x4,
        crate::simd::neon::I16x8,
        crate::simd::neon::I16x8,
        crate::simd::neon::I16x8,
        "neon"
    );
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    isa_hevc_intra_u16!(
        simd128,
        crate::simd::simd128::I16x4,
        crate::simd::simd128::I16x8,
        crate::simd::simd128::I16x8,
        crate::simd::simd128::I16x8,
        "simd128"
    );
}
