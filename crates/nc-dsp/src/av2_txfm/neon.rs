//! AV2 forward transforms in the same column/transpose/row family as the
//! AV1 and VP9 NEON forward kernels. Large DCTs use the eight-column layout
//! of AV2 AVX2 and widen to i32 for products and the row pass. The shared
//! integer butterfly supplies AV2's own basis and rounding.
//!
//! The eight-input layout and split wide row pass also appear in libaom's
//! AV1 `av1_lowbd_fwd_txfm2d_32x32_neon` (BSD-2-Clause with AOM terms,
//! see `THIRD_PARTY.md`). AV2 uses its own integer matrix.

use core::mem::MaybeUninit;

use super::generic::Av2Basis;
use crate::codec_txfm::butterfly::{DctBasis, ForwardLanes, forward_dct};
use crate::codec_txfm::generic::written;
use crate::simd::neon::{I16x8, I32x4};
use crate::simd::{I16Lanes, I32Lanes};
use crate::txfm::generic::store_pass;

super::generic::av2_txfm_kernels!(#[target_feature(enable = "neon")]; I32x4, I32x4);
super::typed_lanes::typed_fns!(av2_typed_fns, #[target_feature(enable = "neon")]; I32x4, I32x4);

/// Select the eight-column forward passes for the large shapes.
pub(crate) fn install(table: &mut Av2TxfmFns) {
    table.forward[2..].copy_from_slice(&[
        forward_wide::<16, 16>,
        forward_wide::<32, 32>,
        forward_wide::<64, 32>,
    ]);
}

/// AV2's coefficients are all in [-90, 90].
const fn narrow<const H: usize>(matrix: [[i32; H]; H]) -> [[i16; H]; H] {
    let mut result = [[0; H]; H];
    let mut r = 0;
    while r < H {
        let mut c = 0;
        while c < H {
            assert!(matrix[r][c] >= i16::MIN as i32 && matrix[r][c] <= i16::MAX as i32);
            result[r][c] = matrix[r][c] as i16;
            c += 1;
        }
        r += 1;
    }
    result
}

// At every first-pass level, each fold combines disjoint subsets of a
// column. Its magnitude is at most 64 * 255 = 16320.
impl ForwardLanes<Av2Basis> for I16x8 {
    type Output = [I32x4; 2];
    type Weight = i16;
    const ODD2: [[i16; 1]; 1] = narrow(Av2Basis::ODD2);
    const ODD4: [[i16; 2]; 2] = narrow(Av2Basis::ODD4);
    const ODD8: [[i16; 4]; 4] = narrow(Av2Basis::ODD8);
    const ODD16: [[i16; 8]; 8] = narrow(Av2Basis::ODD16);
    const ODD32: [[i16; 16]; 16] = narrow(Av2Basis::ODD32);
    const ODD64: Option<[[i16; 32]; 32]> = Some(narrow(Av2Basis::ODD64.unwrap()));

    #[inline]
    fn zero() -> Self {
        Self::splat(0)
    }

    #[inline]
    fn fold(self, other: Self) -> (Self, Self) {
        (self.add(other), self.sub(other))
    }

    #[inline]
    fn dot<const H: usize>(input: &[Self; H], weights: &[i16; H]) -> Self::Output {
        Self::matrix_dot(input, weights)
    }

    #[inline]
    fn dc(self) -> Self::Output {
        self.mul_wide(Av2Basis::DC as i16)
    }
}

#[target_feature(enable = "neon")]
unsafe fn forward_wide<const N: usize, const C: usize>(
    residual: &[i16],
    stride: usize,
    out: &mut [i32],
) {
    let (first, second) = Av2TxSize::from_side(N).unwrap().forward_shifts();
    let mut x = [I16x8::splat(0); N];
    let mut y = [[I32x4::splat(0); 2]; N];
    let mut g = [MaybeUninit::<i32>::uninit(); super::MAX_POINTS * super::MAX_CODED_SIDE];

    for x0 in (0..N).step_by(I16x8::LANES) {
        for (r, lane) in x.iter_mut().enumerate() {
            *lane = I16x8::load(&residual[r * stride + x0..]);
        }
        forward_dct::<Av2Basis, I16x8, N>(&x, &mut y, C);
        #[expect(
            clippy::needless_range_loop,
            reason = "the two column halves index every row"
        )]
        for half in 0..2 {
            let values: [I32x4; C] = core::array::from_fn(|i| y[i][half].round_shift(first));
            store_pass::<I32x4>(&values, x0 + half * I32x4::LANES, C, &mut g);
        }
    }
    // SAFETY: the exact 8-column tiles wrote every intermediate coefficient.
    let g = unsafe { written(&g, N * C) };
    let mut x = [I32x4::splat(0); N];
    let mut y = [I32x4::splat(0); N];
    g::forward_rows::<I32x4, N, C>(g, &mut x, &mut y, second, out);
}
