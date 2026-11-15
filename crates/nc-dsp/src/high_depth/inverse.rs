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

/// The two passes at `depth`.
fn passes(depth: BitDepth) -> (Pass, Pass) {
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

/// The `N`x`N` inverse transform `T` of `coeffs` into `residual` at
/// `depth`, on one scalar lane.
fn inverse<T: Line<i32>, const N: usize>(coeffs: &[i16], residual: &mut [i16], depth: BitDepth) {
    let (first, second) = passes(depth);
    // SAFETY: one `i32` lane needs no ISA; the callers pass `N * N`
    // coefficients and residuals (`DepthKernels::hevc_inverse_add`).
    unsafe { columns_then_rows::<i32, T, N>(coeffs, first, second, residual) }
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
            unsafe { SCALAR[transform as usize](&coeffs, &mut got, BitDepth::EIGHT) };
            assert_eq!(got, want, "{transform:?}");
        }
    }
}
