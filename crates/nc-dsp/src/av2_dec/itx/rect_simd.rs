//! Shared AV2 butterflies on the decoder's column-major rectangles.

use nc_common::PlaneViewMut;

use super::{RectTxSize, TxClass, first_pass_rows};
use crate::av2_txfm::generic;
use crate::simd::I32Lanes;
#[cfg(target_arch = "x86_64")]
use crate::simd::avx2::{I32x4, I32x8 as Wide};
#[cfg(target_arch = "aarch64")]
use crate::simd::neon::I32x4;
#[cfg(target_arch = "wasm32")]
use crate::simd::simd128::I32x4;
use crate::{BlockSize, Dsp, Isa};

#[cfg(not(target_arch = "x86_64"))]
type Wide = I32x4;

pub(super) fn try_add(
    dsp: &Dsp,
    dst: &mut [u8],
    (offset, stride): (usize, usize),
    coefficients: &mut [i16],
    size: RectTxSize,
    eob: u16,
) -> bool {
    if dsp.isa() == Isa::Scalar {
        return false;
    }
    let (width, height) = (size.width(), size.height());
    if width == height {
        return false;
    }
    let view = &mut PlaneViewMut::new(&mut dst[offset..], stride, width, height)
        .expect("AV2 transform block outside its plane");
    let rows = first_pass_rows(size, TxClass::TwoD, eob);
    let [first, second] = size.info().shifts.map(u32::from);
    let shifts = (first, second);
    macro_rules! shape {
        ($v:ty, $w:literal, $h:literal, $cw:literal, $ch:literal) => {
            add::<
                $v,
                $cw,
                $ch,
                { $cw * $ch },
                { $w * $h },
                { (($w as usize).ilog2() + ($h as usize).ilog2()) & 1 == 1 },
            >(dsp, view, coefficients, rows, shifts)
        };
    }
    // SAFETY: the destination view and scan bounds are validated, and the
    // dispatch selected an ISA covering the lane width for each shape.
    unsafe {
        match size {
            RectTxSize::Tx4x8 => shape!(I32x4, 4, 8, 4, 8),
            RectTxSize::Tx8x4 => shape!(I32x4, 8, 4, 8, 4),
            RectTxSize::Tx8x16 => shape!(Wide, 8, 16, 8, 16),
            RectTxSize::Tx16x8 => shape!(Wide, 16, 8, 16, 8),
            RectTxSize::Tx16x32 => shape!(Wide, 16, 32, 16, 32),
            RectTxSize::Tx32x16 => shape!(Wide, 32, 16, 32, 16),
            RectTxSize::Tx32x64 => shape!(Wide, 32, 64, 32, 32),
            RectTxSize::Tx64x32 => shape!(Wide, 64, 32, 32, 32),
            RectTxSize::Tx4x16 => shape!(I32x4, 4, 16, 4, 16),
            RectTxSize::Tx16x4 => shape!(I32x4, 16, 4, 16, 4),
            RectTxSize::Tx8x32 => shape!(Wide, 8, 32, 8, 32),
            RectTxSize::Tx32x8 => shape!(Wide, 32, 8, 32, 8),
            RectTxSize::Tx16x64 => shape!(Wide, 16, 64, 16, 32),
            RectTxSize::Tx64x16 => shape!(Wide, 64, 16, 32, 16),
            RectTxSize::Tx4x32 => shape!(I32x4, 4, 32, 4, 32),
            RectTxSize::Tx32x4 => shape!(I32x4, 32, 4, 32, 4),
            RectTxSize::Tx8x64 => shape!(Wide, 8, 64, 8, 32),
            RectTxSize::Tx64x8 => shape!(Wide, 64, 8, 32, 8),
            RectTxSize::Tx4x64 => shape!(I32x4, 4, 64, 4, 32),
            RectTxSize::Tx64x4 => shape!(I32x4, 64, 4, 32, 4),
            _ => unreachable!("checked non-square transform"),
        }
    }
    true
}

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn add<
    V: I32Lanes,
    const W: usize,
    const H: usize,
    const AREA: usize,
    const OUTPUT: usize,
    const NORMALIZE: bool,
>(
    dsp: &Dsp,
    dst: &mut PlaneViewMut<'_>,
    coefficients: &mut [i16],
    rows: usize,
    shifts: (u32, u32),
) {
    let coefficients = coefficients
        .first_chunk_mut::<AREA>()
        .expect("coded coefficient block");
    let mut residual = [0; AREA];
    // SAFETY: V's enabled lane count divides both dimensions.
    unsafe {
        generic::inverse_rect_from_columns::<V, W, H, AREA, NORMALIZE>(
            coefficients,
            rows,
            shifts,
            &mut residual,
        );
    };
    coefficients.fill(0);
    if AREA == OUTPUT {
        let size = BlockSize::new(W, H).expect("AV2 rectangle");
        dsp.add_residual(size, dst, &residual, W);
    } else {
        let (width, height) = (dst.width(), dst.height());
        let size = BlockSize::new(width, height).expect("AV2 rectangle");
        let mut expanded = [0; OUTPUT];
        for (row, out) in expanded.chunks_exact_mut(width).enumerate() {
            let row = if height == H { row } else { row / 2 };
            let input = &residual[row * W..][..W];
            if width == W {
                out.copy_from_slice(input);
            } else {
                for (pair, &value) in out.as_chunks_mut::<2>().0.iter_mut().zip(input) {
                    *pair = [value; 2];
                }
            }
        }
        dsp.add_residual(size, dst, &expanded, width);
    }
}
