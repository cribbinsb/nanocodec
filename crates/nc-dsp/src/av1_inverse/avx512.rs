//! Sixteen-lane AV1 decoder inverses through the shared DCT/ADST/identity
//! driver. Shapes with a side below 16 keep the inherited AVX2 entry.
use super::{Av1InverseFns, Kernel1d};
use crate::simd::avx512::I32x16;

pub(crate) fn install(table: &mut Av1InverseFns) {
    table.add[2] = add::<16, 16>;
    table.add[3] = add::<32, 32>;
    table.add[4] = add::<64, 64>;
    table.add[9] = add::<16, 32>;
    table.add[10] = add::<32, 16>;
    table.add[11] = add::<32, 64>;
    table.add[12] = add::<64, 32>;
    table.add[17] = add::<16, 64>;
    table.add[18] = add::<64, 16>;
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn add<const W: usize, const H: usize>(
    vertical: Kernel1d,
    horizontal: Kernel1d,
    coeffs: &[i32],
    nonzero_rows: usize,
    dst: &mut [u8],
    stride: usize,
) {
    // SAFETY: Dsp validates the buffers and ISA; sixteen lanes divide both
    // dimensions of every table entry installed above.
    unsafe {
        super::generic::inverse_add::<I32x16, W, H>(
            vertical,
            horizontal,
            coeffs,
            nonzero_rows,
            dst,
            stride,
        );
    }
}
