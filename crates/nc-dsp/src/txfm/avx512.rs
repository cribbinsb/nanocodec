//! ICL forward32 uses guarded word lanes (including the low16 square).
//! Sixteen/64-point forward graphs retain the original wide lanes. Forward32
//! table entries point directly to ISA kernels, avoiding an extra tail-jump
//! wrapper.
//!
//! Every inverse keeps the AVX2 tier's guarded 16-bit graphs
//! (`avx2_inverse`): on Zen 5 (c8a, `nc-checkasm --bench`, 2026-11-17) the
//! sixteen 32-bit lanes installed here before were up to 1.65x slower than
//! them (sparse low-8 32x32: 235 vs 142 ns) and never faster on coefficients
//! inside the 16-bit guard. Only blocks above the guard, which run the
//! guard's 32-bit fallback, are slower (dense synthetic 32x32: 0.71 vs
//! 0.26 us); no 1080p AV1 encode or decode we profiled reached it.
use super::{TxfmFns, generic};
use crate::simd::avx512::I32x16;

pub(crate) fn install(table: &mut TxfmFns) {
    super::avx512_quant::install(table);
    table.forward_dct[2..]
        .copy_from_slice(&[forward_16, super::avx512_forward::forward32::<false>]);
    table.forward_dct_64 = forward_64;
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
    ($($f:ident,$layout:ident,$raster:ident,$bounded:ident);*)=>{$(
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $f(src:&[i16],stride:usize,out:&mut [i16]) {
            // SAFETY: supported ISA and validated buffers from Dsp.
            unsafe { generic::$f::<I32x16>(src,stride,out) }
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
    forward_16_layout,
    forward_16_raster,
    forward_16_bounded
);

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn forward_64(src: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: supported ISA and validated buffers from Dsp.
    unsafe { generic::forward_64_pruned::<I32x16>(src, stride, out) }
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn forward_64_low(src: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: supported ISA; sixteen retained outputs match the lane count.
    unsafe {
        generic::forward_64_low::<I32x16, 16>(src, stride, out);
    }
}
