//! The `u16` kernels of the vector ISAs that the compiler vectorises from
//! the shared generic references: each instance below is the reference
//! (`mc::scalar::{separable_at, hevc_prep_at, hevc_bi_at}`, all written as
//! whole-row multiply-adds and always inlined) compiled under the ISA's
//! target features. Bit-exact by construction (the same source); the tests in
//! `high_depth::tests` compare every ISA with the scalar table.

/// One ISA's vectorised MC kernels (`$feature`: the target feature string).
macro_rules! vectorised {
    ($feature:literal, $w:ty, $m:ty, $v:ty, $h:ty) => {
        use nc_common::BitDepth;

        use crate::high_depth::{Bi16Fn, Convolve16Fn, Prep16Fn};
        use crate::mc::SeparableTaps;
        use crate::mc::separable_u16_simd::{bi_lanes, predict_lanes, prep_lanes};

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
            predict_lanes::<T, N, W, $w, $m, $v, $h>(src, ss, dst, ds, h, frac, depth);
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
            prep_lanes::<T, N, W, $w, $m, $v, $h>(src, ss, dst, ds, h, frac, depth);
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
            bi_lanes::<T, N, W, $w, $m, $v, $h>(src, ss, first, dst, ds, h, frac, depth);
        }

        /// The kernels for [`super::super::HighFns`].
        pub(crate) const PREDICT: [[Convolve16Fn; 5]; 5] = crate::high_depth::families!(predict);
        pub(crate) const PREP: [[Prep16Fn; 5]; 5] = crate::high_depth::families!(prep);
        pub(crate) const BI: [[Bi16Fn; 5]; 5] = crate::high_depth::families!(bi);
    };
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    vectorised!(
        "avx2",
        crate::simd::avx2::I16x16,
        crate::simd::avx2::I16x16,
        crate::simd::avx2::I16x8,
        crate::simd::avx2::I16x4
    );
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512 {
    vectorised!(
        "avx512f,avx512bw,avx512vl",
        crate::simd::avx512_i16::I16x32,
        crate::simd::avx2::I16x16,
        crate::simd::avx2::I16x8,
        crate::simd::avx2::I16x4
    );
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    vectorised!(
        "neon",
        crate::simd::neon::I16x8,
        crate::simd::neon::I16x8,
        crate::simd::neon::I16x8,
        crate::simd::neon::I16x4
    );
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
pub(super) mod sve2 {
    use nc_common::BitDepth;

    use crate::high_depth::Prep16Fn;
    use crate::mc::HevcLumaTaps;
    use crate::mc::separable_u16_simd::prep_lanes;
    use crate::simd::neon::{I16x4, I16x8Sve2};

    #[target_feature(enable = "neon")]
    #[inline(never)]
    unsafe fn prep_horizontal<const W: usize>(
        src: &[u16],
        ss: usize,
        dst: &mut [i16],
        ds: usize,
        h: usize,
        frac: (usize, usize),
        depth: BitDepth,
    ) {
        prep_lanes::<HevcLumaTaps, 8, W, I16x8Sve2, I16x8Sve2, I16x8Sve2, I16x4>(
            src, ss, dst, ds, h, frac, depth,
        );
    }

    unsafe fn prep<const W: usize>(
        src: &[u16],
        ss: usize,
        dst: &mut [i16],
        ds: usize,
        h: usize,
        frac: (usize, usize),
        depth: BitDepth,
    ) {
        // SAFETY: the dispatcher checks SVE2 and the caller checks the footprint.
        // The dot expects signed halfword samples with sums fitting in i32.
        unsafe {
            if frac.0 != 0 && depth.bits() <= 12 {
                prep_horizontal::<W>(src, ss, dst, ds, h, frac, depth);
            } else {
                super::neon::prep::<HevcLumaTaps, 8, W>(src, ss, dst, ds, h, frac, depth);
            }
        }
    }

    pub(in crate::high_depth) const HEVC_LUMA_PREP: [Prep16Fn; 5] = [
        super::neon::prep::<HevcLumaTaps, 8, 4>,
        prep::<8>,
        prep::<16>,
        prep::<32>,
        prep::<64>,
    ];
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod simd128 {
    vectorised!(
        "simd128",
        crate::simd::simd128::I16x8,
        crate::simd::simd128::I16x8,
        crate::simd::simd128::I16x8,
        crate::simd::simd128::I16x4
    );
}
