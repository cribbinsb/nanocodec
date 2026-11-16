//! The HEVC inverse transform at any bit depth (H.265 §8.6.4.2): the
//! shared butterflies of `codec_txfm::generic` with the depth's second
//! shift, `bdShift = 20 − BitDepth`.
//!
//! Width of the result: the first pass clamps to 16 bits (`coeffMin` /
//! `coeffMax`); the second pass's sum reaches `32 · 90 · 2^15 ≈ 2^26.5`, so
//! after `>> 12` (8-bit) it fits 16 bits but after `>> 10` (10-bit) it can
//! reach `2^16.5` for a stream that drives every coefficient to the limit.
//! The residual is therefore clamped to 16 bits as well, as HM stores it
//! (`Pel` residuals, `TComTrQuant::xIT`); conforming streams never reach the
//! clamp, and at 8 bits it is never active.

use nc_common::BitDepth;

use crate::codec_txfm::generic::{
    HevcIdct4, HevcIdct8, HevcIdct16, HevcIdct32, HevcIdst4, Line, Pass, columns_then_rows,
};
use crate::cpu::Isa;

use super::Inverse16Fn;

/// Coefficients of the largest HEVC transform (32x32).
pub(crate) const MAX_AREA: usize = 32 * 32;
/// `bdShift` of the second pass is `RESIDUAL_BITS − BitDepth`.
const RESIDUAL_BITS: u32 = 20;
/// The first pass's shift (`bdShift = 7`, then the 16-bit clamp).
const FIRST_SHIFT: u32 = 7;

/// The scalar table by [`crate::HevcTransform`] (DST 4, DCT 4..32).
pub(crate) const SCALAR: [Inverse16Fn; 5] = [
    inverse::<HevcIdst4, 4>,
    inverse::<HevcIdct4, 4>,
    inverse::<HevcIdct8, 8>,
    inverse::<HevcIdct16, 16>,
    inverse::<HevcIdct32, 32>,
];

/// The table for `isa`: the shared butterflies on its vector lanes, the
/// passes bounded by the coefficients' extent (the 8-bit tiers' scheme).
pub(crate) fn kernels(isa: Isa) -> [Inverse16Fn; 5] {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => avx2::KERNELS,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::KERNELS,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::KERNELS,
        _ => SCALAR,
    }
}

/// The two passes at `depth`.
pub(super) fn passes(depth: BitDepth) -> (Pass, Pass) {
    (
        Pass {
            shift: FIRST_SHIFT,
            clamp: true,
        },
        Pass {
            shift: RESIDUAL_BITS - depth.bits(),
            clamp: true,
        },
    )
}

/// The whole-block inverse `transform` of `coeffs` into `residual` at
/// `depth` with this table's kernel ([`crate::Dsp::hevc_inverse`] at a
/// depth): VVC's square DCT-II blocks, whose bases and stage clips are
/// HEVC's (H.266 §8.7.4.1, second stage `20 − BitDepth`).
///
/// # Panics
/// If `coeffs` or `residual` is shorter than the block.
pub(crate) fn whole_block(
    dsp: &crate::Dsp,
    transform: crate::HevcTransform,
    coeffs: &[i16],
    residual: &mut [i16],
    depth: BitDepth,
) {
    let n = transform.side();
    assert!(coeffs.len() >= n * n && residual.len() >= n * n);
    // SAFETY: both slices cover the block (checked above); the extent is
    // the whole block; the table matches a supported ISA.
    unsafe { (dsp.high.hevc_inverse[transform as usize])(coeffs, residual, (n, n), depth) }
}

/// The `N`x`N` inverse transform `T` of `coeffs` into `residual` at
/// `depth`, on one scalar lane (the whole block: the reference).
fn inverse<T: Line<i32>, const N: usize>(
    coeffs: &[i16],
    residual: &mut [i16],
    _extent: (usize, usize),
    depth: BitDepth,
) {
    let (first, second) = passes(depth);
    // SAFETY: one `i32` lane needs no ISA; the callers pass `N * N`
    // coefficients and residuals (`DepthKernels::hevc_inverse_add`).
    unsafe { columns_then_rows::<i32, T, N>(coeffs, first, second, residual) }
}

/// One ISA's bounded kernels: `$v4` lanes for the 4-point transforms, `$v`
/// for the others.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! bounded_kernels {
    ($feature:literal, $v4:ty, $v:ty) => {
        use nc_common::BitDepth;

        use crate::codec_txfm::generic::{
            BoundedLine, HevcIdct4, HevcIdct8, HevcIdct16, HevcIdct32, HevcIdst4,
            hevc_inverse_extent_passes,
        };
        use crate::high_depth::Inverse16Fn;

        pub(crate) const KERNELS: [Inverse16Fn; 5] = [
            bounded::<$v4, HevcIdst4, 4>,
            bounded::<$v4, HevcIdct4, 4>,
            bounded::<$v, HevcIdct8, 8>,
            bounded::<$v, HevcIdct16, 16>,
            bounded::<$v, HevcIdct32, 32>,
        ];

        /// # Safety
        /// The ISA must be available; the slices cover `N * N`; `extent`
        /// covers every nonzero coefficient.
        #[target_feature(enable = $feature)]
        unsafe fn bounded<V: crate::simd::I32Lanes, T: BoundedLine, const N: usize>(
            coeffs: &[i16],
            residual: &mut [i16],
            extent: (usize, usize),
            depth: BitDepth,
        ) {
            let passes = super::passes(depth);
            // SAFETY: forwarded contract.
            unsafe { hevc_inverse_extent_passes::<V, T, N>(coeffs, residual, extent, passes) }
        }
    };
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use crate::simd::avx2::{I32x4, I32x8};
    bounded_kernels!("avx2", I32x4, I32x8);
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use crate::simd::neon::I32x4;
    bounded_kernels!("neon", I32x4, I32x4);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128 {
    use crate::simd::simd128::I32x4;
    bounded_kernels!("simd128", I32x4, I32x4);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dsp, HevcTransform, Isa};

    /// At 8 bits the depth's passes are the 8-bit inverse's.
    #[test]
    fn eight_bit_depth_matches_the_eight_bit_inverse() {
        let dsp = Dsp::with_isa(Isa::Scalar).unwrap();
        for transform in HevcTransform::ALL {
            let n = transform.side();
            let coeffs: Vec<i16> = (0..n * n)
                .map(|i| ((i * 7919 + 13) % 4001) as i16 - 2000)
                .collect();
            let mut want = vec![0i16; n * n];
            dsp.hevc_inverse(transform, &coeffs, &mut want);
            let mut got = vec![0i16; n * n];
            // SAFETY: scalar kernels; slices cover the block.
            unsafe { SCALAR[transform as usize](&coeffs, &mut got, (n, n), BitDepth::EIGHT) };
            assert_eq!(got, want, "{transform:?}");
        }
    }

    /// Every ISA's bounded kernel equals the scalar reference at 10 bits,
    /// for every extent that covers the nonzero coefficients.
    #[test]
    fn bounded_kernels_match_the_reference() {
        for isa in [Isa::Avx2, Isa::Avx512, Isa::Neon]
            .into_iter()
            .filter(|i| i.is_supported())
        {
            let table = kernels(isa);
            for transform in HevcTransform::ALL {
                let n = transform.side();
                for (rows, columns) in [(1, 1), (2, 3), (n / 2, n), (n, n)] {
                    let coeffs: Vec<i16> = (0..n * n)
                        .map(|i| {
                            let (r, c) = (i / n, i % n);
                            if r < rows && c < columns {
                                ((i * 7919 + 13) % 30001) as i16 - 15000
                            } else {
                                0
                            }
                        })
                        .collect();
                    let mut want = vec![0i16; n * n];
                    let mut got = vec![0i16; n * n];
                    // SAFETY: the ISA is supported; slices cover the block.
                    unsafe {
                        SCALAR[transform as usize](&coeffs, &mut want, (n, n), BitDepth::TEN);
                        table[transform as usize](
                            &coeffs,
                            &mut got,
                            (rows, columns),
                            BitDepth::TEN,
                        );
                    }
                    assert_eq!(got, want, "{isa:?} {transform:?} {rows}x{columns}");
                }
            }
        }
    }
}
