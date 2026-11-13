//! AVX-512 instantiations of the shared AV1/VP9 DCT graphs. Sixteen lanes
//! serve 16/32/64-point passes. An eight-coefficient sparse prefix retains
//! AVX2, so no pass reads or writes beyond its retained square.
use super::{TxfmFns, generic};
use crate::simd::{avx2::I32x8, avx512::I32x16};

pub(crate) fn install(table: &mut TxfmFns) {
    super::avx512_quant::install(table);
    table.forward_dct[2..].copy_from_slice(&[forward_16, forward_32]);
    table.inverse_dct_add[2..].copy_from_slice(&[inverse_16, inverse_32]);
    table.forward_dct_64 = forward_64;
    table.inverse_dct_add_64 = inverse_64;
    table.inverse_dct_add_sparse = [inverse_32_sparse, inverse_64_sparse];
    table.forward_dct_raster[2..]
        .copy_from_slice(&[Some(forward_16_raster), Some(forward_32_raster)]);
    table.forward_dct_low[1..].copy_from_slice(&[Some(forward_32_low), Some(forward_64_low)]);
    table.forward_dct_raster_low[1] = Some(forward_32_low_raster);
    table.forward_dct_raster_bounded = [forward_16_bounded, forward_32_bounded];
}

macro_rules! sizes {
    ($($f:ident,$i:ident,$layout:ident,$raster:ident,$bounded:ident);*)=>{$(
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $f(src:&[i16],stride:usize,out:&mut [i32]) {
            // SAFETY: supported ISA and validated buffers from Dsp.
            unsafe { generic::$f::<I32x16>(src,stride,out) }
        }
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $i(coeffs:&[i32],dst:&mut [u8],stride:usize) {
            // SAFETY: supported ISA and validated buffers from Dsp.
            unsafe { generic::$i::<I32x16>(coeffs,dst,stride) }
        }
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $raster(src:&[i16],stride:usize,out:&mut [i32]) {
            // SAFETY: supported ISA and validated buffers from Dsp.
            unsafe { generic::$layout::<I32x16,true>(src,stride,out) }
        }
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $bounded(src:&[i16],stride:usize,limit:u32,out:&mut [i32])->bool {
            // SAFETY: supported ISA and validated buffers from Dsp.
            unsafe { generic::$bounded::<I32x16>(src,stride,limit,out) }
        }
    )*};
}
sizes!(forward_16,inverse_16,forward_16_layout,forward_16_raster,forward_16_bounded;
       forward_32,inverse_32,forward_32_layout,forward_32_raster,forward_32_bounded);

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn forward_64(src: &[i16], stride: usize, out: &mut [i32]) {
    // SAFETY: supported ISA and validated buffers from Dsp.
    unsafe { generic::forward_64_pruned::<I32x16>(src, stride, out) }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn inverse_64(coeffs: &[i32], dst: &mut [u8], stride: usize) {
    // SAFETY: supported ISA and validated buffers from Dsp.
    unsafe { generic::inverse_64::<I32x16>(coeffs, dst, stride) }
}
macro_rules! sparse {
    ($($name:ident);*)=>{$(
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $name(coeffs:&[i32],dst:&mut [u8],stride:usize,limit:usize) {
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
unsafe fn forward_32_low(src: &[i16], stride: usize, out: &mut [i32]) {
    // SAFETY: supported ISA; sixteen retained outputs match the lane count.
    unsafe {
        generic::forward_32_low::<I32x16, 16, false>(src, stride, out);
    }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn forward_32_low_raster(src: &[i16], stride: usize, out: &mut [i32]) {
    // SAFETY: supported ISA; sixteen retained outputs match the lane count.
    unsafe {
        generic::forward_32_low::<I32x16, 16, true>(src, stride, out);
    }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn forward_64_low(src: &[i16], stride: usize, out: &mut [i32]) {
    // SAFETY: supported ISA; sixteen retained outputs match the lane count.
    unsafe {
        generic::forward_64_low::<I32x16, 16>(src, stride, out);
    }
}
