//! The `u16` kernels of the vector ISAs that the compiler vectorises from
//! the shared generic references: each instance below is the reference
//! (`mc::scalar::{separable_at, hevc_prep_at, hevc_bi_at}`,
//! `hevc_weight::reference::weight_at`, all written as whole-row
//! multiply-adds and always inlined) compiled under the ISA's target
//! features. Bit-exact by construction (the same source); the tests in
//! `high_depth::tests` compare every ISA with the scalar table.
//!
//! Main 10 decode (`WP_A_MAIN10_Toshiba_3`, callgrind): MC and weighted
//! prediction were 50 % of the instructions on the scalar table. The VP9
//! prediction (`vp9_mc::rows_at`) and deblocking
//! (`vp9_deblock::filter_lines_at`) are compiled the same way.

/// One ISA's vectorised MC and weighting kernels (`$feature`: the target
/// feature string).
macro_rules! vectorised {
    ($feature:literal) => {
        use nc_common::BitDepth;

        use crate::high_depth::{Bi16Fn, Convolve16Fn, Prep16Fn, Vp9Mc16Fn};
        use crate::mc::SeparableTaps;
        use crate::mc::scalar::{hevc_bi_at, hevc_prep_at, separable_at};

        /// # Safety
        /// The ISA must be available; the slices cover the block.
        #[target_feature(enable = $feature)]
        pub(crate) unsafe fn predict<T: SeparableTaps<N>, const N: usize, const W: usize>(
            src: &[u16],
            ss: usize,
            dst: &mut [u16],
            ds: usize,
            h: usize,
            frac: (usize, usize),
            depth: BitDepth,
        ) {
            separable_at::<u16, T, N, W>(src, ss, dst, ds, h, frac, depth);
        }

        /// # Safety
        /// As [`predict`].
        #[target_feature(enable = $feature)]
        pub(crate) unsafe fn prep<T: SeparableTaps<N>, const N: usize, const W: usize>(
            src: &[u16],
            ss: usize,
            dst: &mut [i16],
            ds: usize,
            h: usize,
            frac: (usize, usize),
            depth: BitDepth,
        ) {
            hevc_prep_at::<u16, T, N, W>(src, ss, dst, ds, h, frac, depth);
        }

        /// # Safety
        /// As [`predict`].
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $feature)]
        pub(crate) unsafe fn bi<T: SeparableTaps<N>, const N: usize, const W: usize>(
            src: &[u16],
            ss: usize,
            first: (&[i16], usize),
            dst: &mut [u16],
            ds: usize,
            h: usize,
            frac: (usize, usize),
            depth: BitDepth,
        ) {
            hevc_bi_at::<u16, T, N, W>(src, ss, first, dst, ds, h, frac, depth);
        }

        /// # Safety
        /// As [`predict`].
        #[target_feature(enable = $feature)]
        pub(crate) unsafe fn vp9_mc<const W: usize>(
            window: (&[u16], usize),
            dst: (&mut [u16], usize),
            h: usize,
            taps: (&[i16; 8], &[i16; 8]),
            average: bool,
            depth: BitDepth,
        ) {
            crate::vp9_mc::rows_at::<u16, W>(window, dst, h, taps, average, depth);
        }

        /// # Safety
        /// The ISA must be available; the footprint lies in `data`.
        #[target_feature(enable = $feature)]
        pub(crate) unsafe fn vp9_filter(
            data: &mut [u16],
            at: (usize, usize, usize),
            lines: usize,
            width: crate::Vp9FilterWidth,
            limits: crate::Vp9FilterLimits,
            depth: BitDepth,
        ) {
            crate::vp9_deblock::filter_lines_at::<u16>(data, at, lines, width, limits, depth);
        }

        /// The kernels for [`super::super::HighFns`].
        pub(crate) const PREDICT: [[Convolve16Fn; 5]; 5] = crate::high_depth::families!(predict);
        pub(crate) const PREP: [[Prep16Fn; 5]; 5] = crate::high_depth::families!(prep);
        pub(crate) const BI: [[Bi16Fn; 5]; 5] = crate::high_depth::families!(bi);
        pub(crate) const VP9_MC: [Vp9Mc16Fn; 5] = crate::high_depth::av1_widths!(vp9_mc);
    };
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    vectorised!("avx2");
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512 {
    vectorised!("avx512f,avx512bw,avx512vl");
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    vectorised!("neon");
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod simd128 {
    vectorised!("simd128");
}
