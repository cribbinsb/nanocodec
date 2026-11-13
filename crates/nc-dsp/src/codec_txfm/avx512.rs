//! HEVC/VVC 16/32-point transforms on the shared drivers. Compact forward
//! lines reuse AVX2's packed matrix weights and range proof (see
//! `avx2_paired`); inverses retain the shared sparse butterflies. Staged rows
//! keep the existing packed AVX2 graph, followed by sixteen-lane columns.

use super::generic::{self as g, Line};
use super::{CodecTxfmFns, HevcForwardMode, HevcForwardStaging};
use crate::simd::{
    I32Lanes,
    avx512::{I32x16, WordPairs16},
};
use core::mem::MaybeUninit;

pub(crate) fn install(table: &mut CodecTxfmFns) {
    super::avx512_h264::install(table);
    table.hevc_forward[3..].copy_from_slice(&[forward16, forward32]);
    table.hevc_inverse[3..].copy_from_slice(&[inverse16, inverse32]);
    table.hevc_forward_bounded[2..].copy_from_slice(&[bounded16, bounded32]);
}

/// One compact matrix loop per size, rather than an unrolled copy per pass.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline(never)]
unsafe fn forward<const N: usize>(x: &[I32x16], y: &mut [I32x16], outputs: usize) {
    let zero = I32x16::splat(0).word_pairs(I32x16::splat(0));
    let mut pairs: [WordPairs16; 16] = [zero; 16];
    for (i, pair) in pairs[..N / 2].iter_mut().enumerate() {
        *pair = x[2 * i].word_pairs(x[2 * i + 1]);
    }
    for (k, out) in y[..outputs].iter_mut().enumerate() {
        let weights = &super::avx2_paired::FORWARD[k * 32 / N];
        let mut sum = I32x16::splat(0);
        for (pair, &weight) in pairs[..N / 2].iter().zip(weights) {
            sum = sum.add(pair.dot(weight));
        }
        *out = sum;
    }
}

macro_rules! sizes {
    ($($line:ident,$f:ident,$i:ident,$b:ident,$inv:ty,$n:literal,$log:literal,$fallback:ident);*)=>{$(
        struct $line;
        impl Line<I32x16> for $line {
            #[inline]
            fn apply(x:&[I32x16],y:&mut [I32x16]) {
                // SAFETY: private lane values only exist in this ISA's kernels.
                unsafe { forward::<$n>(x,y,$n); }
            }
            #[inline]
            fn apply_low(x:&[I32x16],y:&mut [I32x16]) {
                // SAFETY: as above, retaining only the requested outputs.
                unsafe { forward::<$n>(x,y,$n/2); }
            }
        }
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $f(src:&[i16],stride:usize,out:&mut [i32]) {
            // SAFETY: validated block and supported ISA from Dsp.
            unsafe { g::hevc_forward::<I32x16,$line,$n,$log>(src,stride,out); }
        }
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $i(coeffs:&[i32],out:&mut [i32]) {
            // SAFETY: validated block and supported ISA from Dsp.
            unsafe { g::hevc_inverse_bounded::<I32x16,$inv,$n>(coeffs,out); }
        }
        #[target_feature(enable="avx512f,avx512bw,avx512vl")]
        unsafe fn $b(src:&[i16],stride:usize,limit:u32,out:&mut [i32],mode:HevcForwardMode)->bool {
            // SAFETY: the retained 8-column prefix needs eight-lane stores.
            if $n==16 && mode.low_frequencies {
                // SAFETY: same validated block and AVX2 is available.
                return unsafe { super::avx2::$fallback(src,stride,limit,out,mode) };
            }
            // SAFETY: shared exact driver, same validated block contract.
            if mode.staging==HevcForwardStaging::Exact {
                // SAFETY: same validated block and lane count divides N.
                return unsafe { g::hevc_forward_bounded::<I32x16,$line,$n,$log>(src,stride,limit,out,mode) };
            }
            let mut scratch=[MaybeUninit::<i32>::uninit();$n*$n];
            // SAFETY: AVX2 is included in this ISA; rows initialize scratch.
            unsafe { crate::txfm::avx2_forward::hevc_rows::<$n>(src,stride,&mut scratch); }
            // SAFETY: the complete intermediate was written above.
            let rows=unsafe { g::written(&scratch,$n*$n) };
            // SAFETY: shared column driver, same validated block contract.
            unsafe { g::hevc_forward_columns::<I32x16,$line,$n,$log>(rows,limit,out,mode) }
        }
    )*};
}
sizes!(Fdct16,forward16,inverse16,bounded16,g::HevcIdct16,16,4,staged16_bounded;
       Fdct32,forward32,inverse32,bounded32,g::HevcIdct32,32,5,staged32_bounded);
