//! The `u16` kernels of the vector ISAs that the compiler vectorises from
//! the shared generic references: each instance below is the reference
//! (`mc::scalar::{separable_at, hevc_prep_at, hevc_bi_at}`,
//! `hevc_weight::reference::weight_at`, all written as whole-row
//! multiply-adds and always inlined) compiled under the ISA's target
//! features. Bit-exact by construction (the same source); the tests in
//! `high_depth::tests` compare every ISA with the scalar table.
//!
//! Main 10 decode (`WP_A_MAIN10_Toshiba_3`, callgrind): MC and weighted
//! prediction were 50 % of the instructions on the scalar table. The AV1
//! unscaled prediction (`av1_mc::convolve::{unscaled_put_at,
//! unscaled_prep_at}`) and the CDEF of interior blocks
//! (`cdef::scalar::interior_rows`) are compiled the same way.

/// One ISA's vectorised MC and weighting kernels (`$feature`: the target
/// feature string).
macro_rules! vectorised {
    ($feature:literal) => {
        use nc_common::BitDepth;

        use crate::av1_mc::convolve::{unscaled_prep_at, unscaled_put_at};
        use crate::cdef::CdefArgs;
        use crate::hevc_weight::HevcWeights;
        use crate::high_depth::{
            Av1Prep16Fn, Av1Put16Fn, Av1Taps, Bi16Fn, CdefInterior16Fn, Convolve16Fn, Prep16Fn,
            Weight16Fn,
        };
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
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $feature)]
        pub(crate) unsafe fn weight(
            weights: &HevcWeights,
            a: &[i16],
            b: &[i16],
            w: usize,
            h: usize,
            dst: &mut [u16],
            ds: usize,
            depth: BitDepth,
        ) {
            crate::hevc_weight::reference::weight_at(weights, a, b, w, h, dst, ds, depth);
        }

        /// # Safety
        /// As [`predict`].
        #[target_feature(enable = $feature)]
        pub(crate) unsafe fn av1_put<const W: usize>(
            src: (&[u16], usize),
            dst: (&mut [u16], usize),
            h: usize,
            taps: Av1Taps<'_>,
            depth: BitDepth,
        ) {
            unscaled_put_at::<u16, W>(src, dst, h, taps, depth);
        }

        /// # Safety
        /// As [`predict`].
        #[target_feature(enable = $feature)]
        pub(crate) unsafe fn av1_prep<const W: usize>(
            src: (&[u16], usize),
            dst: (&mut [i16], usize),
            h: usize,
            taps: Av1Taps<'_>,
            depth: BitDepth,
        ) {
            unscaled_prep_at::<u16, W>(src, dst, h, taps, depth);
        }

        /// # Safety
        /// As [`predict`].
        #[target_feature(enable = $feature)]
        pub(crate) unsafe fn cdef_interior<const N: usize>(
            src: &[u16],
            stride: usize,
            origin: usize,
            args: &CdefArgs,
            dst: &mut [u16],
            dst_stride: usize,
        ) {
            crate::cdef::scalar::interior_rows::<u16, N>(
                src, stride, origin, args, dst, dst_stride,
            );
        }

        /// The kernels for [`super::super::HighFns`].
        pub(crate) const WEIGHT: Weight16Fn = weight;
        pub(crate) const PREDICT: [[Convolve16Fn; 5]; 5] = crate::high_depth::families!(predict);
        pub(crate) const PREP: [[Prep16Fn; 5]; 5] = crate::high_depth::families!(prep);
        pub(crate) const BI: [[Bi16Fn; 5]; 5] = crate::high_depth::families!(bi);
        pub(crate) const AV1_PUT: [Av1Put16Fn; 5] = crate::high_depth::av1_widths!(av1_put);
        pub(crate) const AV1_PREP: [Av1Prep16Fn; 5] = crate::high_depth::av1_widths!(av1_prep);
        pub(crate) const CDEF_INTERIOR: [CdefInterior16Fn; 2] =
            [cdef_interior::<4>, cdef_interior::<8>];
    };
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    vectorised!("avx2");
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    vectorised!("neon");
}
