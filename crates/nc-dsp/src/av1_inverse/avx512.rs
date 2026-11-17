//! The AVX-512 tier's AV1 decoder inverses. Every size keeps the AVX2 tier's
//! guarded 16-bit graphs (`txfm::avx2_inverse`), except 16x16 transforms
//! with ADST (or flipped ADST) on both sides, which run the generic driver
//! on sixteen 32-bit lanes. On Zen 5 (c8a, `nc-checkasm --bench`,
//! 2026-11-17) the 32-bit lanes ran 16x16 `ADST_ADST` in 82.5 ns against
//! 95.0 ns, but every DCT and identity shape from 16x16 to 64x64 1.1-1.8x
//! slower than the 16-bit graphs. Mixed DCT/ADST pairs were not measured
//! and keep the 16-bit graphs.
use super::{Av1InverseFns, Kernel1d};
use crate::simd::avx512::I32x16;
use crate::txfm::avx2_inverse;
use crate::txfm::narrow_inverse::NarrowCoefficient;

/// The 16x16 entry of `Av1TxSize::ALL`.
const TX_16X16: usize = 2;

pub(crate) fn install(table: &mut Av1InverseFns) {
    table.add[TX_16X16] = add_16x16::<i32>;
    table.add_i16[TX_16X16] = add_16x16::<i16>;
}

/// ADST or flipped ADST: the kernels sixteen 32-bit lanes run faster.
const fn is_adst(kernel: Kernel1d) -> bool {
    matches!(kernel, Kernel1d::Adst | Kernel1d::FlipAdst)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn add_16x16<C: NarrowCoefficient>(
    vertical: Kernel1d,
    horizontal: Kernel1d,
    coeffs: &[C],
    nonzero_rows: usize,
    dst: &mut [u8],
    stride: usize,
) {
    // SAFETY: Dsp validates the buffers and ISA (which includes AVX2);
    // sixteen lanes divide both sides.
    unsafe {
        if is_adst(vertical) && is_adst(horizontal) {
            super::generic::inverse_add::<I32x16, C, 16, 16>(
                vertical,
                horizontal,
                coeffs,
                nonzero_rows,
                dst,
                stride,
            );
        } else {
            avx2_inverse::av1_inverse_add::<C, 16, 16>(
                (vertical, horizontal),
                coeffs,
                nonzero_rows,
                dst,
                stride,
            );
        }
    }
}
