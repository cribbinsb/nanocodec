//! ICL forward32 uses guarded word lanes (including the low16 square).
//! Sixteen/64-point and inverse graphs retain the original wide lanes.
//! Sparse eight-coefficient prefixes retain AVX2. Forward32 table entries
//! point directly to ISA kernels, avoiding an extra tail-jump wrapper.
use super::{TxfmFns, generic};
use crate::simd::{avx2::I32x8, avx512::I32x16};

pub(crate) fn install(table: &mut TxfmFns) {
    super::avx512_quant::install(table);
    table.forward_dct[2..]
        .copy_from_slice(&[forward_16, super::avx512_forward::forward32::<false>]);
    table.inverse_dct_add[2..].copy_from_slice(&[inverse_16, inverse_32]);
    table.forward_dct_64 = forward_64;
    table.inverse_dct_add_64 = inverse_64;
    table.inverse_dct_add_sparse[1..].copy_from_slice(&[inverse_32_sparse, inverse_64_sparse]);
    table.forward_dct_raster[2..]
        .copy_from_slice(&[forward_16_raster, super::avx512_forward::forward32::<true>]);
    table.forward_dct_low[1..].copy_from_slice(&[
        Some(super::avx512_forward::low32::<false>),
        Some(forward_64_low),
    ]);
    table.forward_dct_raster_low[1] = Some(super::avx512_forward::low32::<true>);
    table.forward_dct_raster_bounded = [forward_16_bounded, super::avx512_forward::bounded32];
}

macro_rules! sizes {
    ($($f:ident,$i:ident,$layout:ident,$raster:ident,$bounded:ident);*)=>{$(
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $f(src:&[i16],stride:usize,out:&mut [i16]) {
            // SAFETY: supported ISA and validated buffers from Dsp.
            unsafe { generic::$f::<I32x16>(src,stride,out) }
        }
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $i(coeffs:&[i16],dst:&mut [u8],stride:usize) {
            // SAFETY: supported ISA and validated buffers from Dsp.
            unsafe { generic::$i::<I32x16>(coeffs,dst,stride) }
        }
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $raster(src:&[i16],stride:usize,out:&mut [i16]) {
            // SAFETY: supported ISA and validated buffers from Dsp.
            unsafe { generic::$layout::<I32x16,true>(src,stride,out) }
        }
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $bounded(src:&[i16],stride:usize,limit:u32,out:&mut [i16])->bool {
            // SAFETY: supported ISA and validated buffers from Dsp.
            unsafe { generic::$bounded::<I32x16>(src,stride,limit,out) }
        }
    )*};
}
sizes!(
    forward_16,
    inverse_16,
    forward_16_layout,
    forward_16_raster,
    forward_16_bounded
);

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn inverse_32(coeffs: &[i16], dst: &mut [u8], stride: usize) {
    // SAFETY: supported ISA and validated buffers from Dsp.
    unsafe { generic::inverse_32::<I32x16>(coeffs, dst, stride) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn forward_64(src: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: supported ISA and validated buffers from Dsp.
    unsafe { generic::forward_64_pruned::<I32x16>(src, stride, out) }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn inverse_64(coeffs: &[i16], dst: &mut [u8], stride: usize) {
    // SAFETY: supported ISA and validated buffers from Dsp.
    unsafe { generic::inverse_64::<I32x16>(coeffs, dst, stride) }
}
macro_rules! sparse {
    ($($name:ident);*)=>{$(
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $name(coeffs:&[i16],dst:&mut [u8],stride:usize,limit:usize) {
            // SAFETY: supported ISA and validated buffers from Dsp; the
            // selected lane count divides the initialized sparse prefix.
            unsafe {
                if limit<16 { generic::$name::<I32x8>(coeffs,dst,stride,limit); }
                else { generic::$name::<I32x16>(coeffs,dst,stride,limit); }
            }
        }
    )*};
}
sparse!(inverse_32_sparse;inverse_64_sparse);

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn forward_64_low(src: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: supported ISA; sixteen retained outputs match the lane count.
    unsafe {
        generic::forward_64_low::<I32x16, 16>(src, stride, out);
    }
}
