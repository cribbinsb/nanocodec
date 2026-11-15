//! Sixteen-lane AV1 decoder inverses through the shared DCT/ADST/identity
//! driver. Shapes with a side below 16 keep the inherited AVX2 entry.
use super::{Av1InverseFns, Kernel1d};
use crate::simd::avx512::I32x16;
use crate::txfm::narrow_inverse::NarrowCoefficient;

/// The sixteen-lane entries of one coefficient type's table.
macro_rules! install_sizes {
    ($add:expr, $coefficient:ty) => {
        $add[2] = add::<$coefficient, 16, 16>;
        $add[3] = add::<$coefficient, 32, 32>;
        $add[4] = add::<$coefficient, 64, 64>;
        $add[9] = add::<$coefficient, 16, 32>;
        $add[10] = add::<$coefficient, 32, 16>;
        $add[11] = add::<$coefficient, 32, 64>;
        $add[12] = add::<$coefficient, 64, 32>;
        $add[17] = add::<$coefficient, 16, 64>;
        $add[18] = add::<$coefficient, 64, 16>;
    };
}

pub(crate) fn install(table: &mut Av1InverseFns) {
    install_sizes!(table.add, i32);
    install_sizes!(table.add_i16, i16);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn add<C: NarrowCoefficient, const W: usize, const H: usize>(
    vertical: Kernel1d,
    horizontal: Kernel1d,
    coeffs: &[C],
    nonzero_rows: usize,
    dst: &mut [u8],
    stride: usize,
) {
    // SAFETY: Dsp validates the buffers and ISA; sixteen lanes divide both
    // dimensions of every table entry installed above.
    unsafe {
        super::generic::inverse_add::<I32x16, C, W, H>(
            vertical,
            horizontal,
            coeffs,
            nonzero_rows,
            dst,
            stride,
        );
    }
}
