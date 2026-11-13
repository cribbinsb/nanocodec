//! The AV2 directional interpolation runs generic over [`I16Lanes`]
//! (AVX2 and SIMD128; NEON has byte kernels in `neon.rs`): `V::LANES`
//! outputs per step, products accumulated in 32 bits.
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs).

#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use crate::intra_prediction::av2::{FOUR_TAP, FOUR_TAP_BITS, TWO_TAP_BITS};
use crate::simd::I16Lanes;

/// The 2-tap filter's weights sum to this.
const TWO_TAP_SUM: i16 = 1 << TWO_TAP_BITS;

/// The 4-tap run over the first whole vectors of `out`; returns how many
/// outputs it wrote (the caller finishes the rest).
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(crate) unsafe fn run4<V: I16Lanes>(window: &[u8], shift: usize, out: &mut [u8]) -> usize {
    let taps = FOUR_TAP[shift].map(|tap| tap as i16);
    let round = 1 << (FOUR_TAP_BITS - 1);
    let mut done = 0;
    while done + V::LANES <= out.len() {
        let samples = &window[done..];
        let acc = V::acc_mul_add2(
            V::acc_zero(),
            V::load_u8(samples),
            taps[0],
            V::load_u8(&samples[1..]),
            taps[1],
        );
        let acc = V::acc_mul_add2(
            acc,
            V::load_u8(&samples[2..]),
            taps[2],
            V::load_u8(&samples[3..]),
            taps[3],
        );
        V::acc_round_shift_store_u8(acc, round, FOUR_TAP_BITS, &mut out[done..]);
        done += V::LANES;
    }
    done
}

/// The 2-tap run over the first whole vectors of `out` (as [`run4`]).
///
/// # Safety
///
/// `V`'s ISA must be available.
#[inline(always)]
pub(crate) unsafe fn run2<V: I16Lanes>(window: &[u8], shift: usize, out: &mut [u8]) -> usize {
    let (first, second) = (TWO_TAP_SUM - shift as i16, shift as i16);
    let round = 1 << (TWO_TAP_BITS - 1);
    let mut done = 0;
    while done + V::LANES <= out.len() {
        let samples = &window[done..];
        let acc = V::acc_mul_add2(
            V::acc_zero(),
            V::load_u8(samples),
            first,
            V::load_u8(&samples[1..]),
            second,
        );
        V::acc_round_shift_store_u8(acc, round, TWO_TAP_BITS, &mut out[done..]);
        done += V::LANES;
    }
    done
}

/// Instantiates one ISA's table: runs on `$wide` lanes first, then
/// `$narrow`, then `$v4`, then the scalar code for the remainder; smooth
/// as [`av2_smooth_kernel`]. NEON has its own byte runs (`neon.rs`).
macro_rules! av2_intra_kernels {
    ($(#[$attr:meta])*; $v4:ty, $narrow:ty, $wide:ty) => {
        use $crate::av2_intra::{runs as r, Av2IntraFns};
        use $crate::intra_prediction::av2 as reference;

        /// This ISA's AV2 intra table.
        pub(crate) fn av2_intra_fns() -> Av2IntraFns {
            Av2IntraFns { run4, run2, smooth }
        }

        $(#[$attr])*
        unsafe fn run4(window: &[u8], shift: usize, out: &mut [u8]) {
            let mut done = 0;
            // SAFETY: see the module docs.
            unsafe {
                done += r::run4::<$wide>(window, shift, out);
                done += r::run4::<$narrow>(&window[done..], shift, &mut out[done..]);
                done += r::run4::<$v4>(&window[done..], shift, &mut out[done..]);
            }
            reference::run4(&window[done..], shift, &mut out[done..]);
        }

        $(#[$attr])*
        unsafe fn run2(window: &[u8], shift: usize, out: &mut [u8]) {
            let mut done = 0;
            // SAFETY: see the module docs.
            unsafe {
                done += r::run2::<$wide>(window, shift, out);
                done += r::run2::<$narrow>(&window[done..], shift, &mut out[done..]);
                done += r::run2::<$v4>(&window[done..], shift, &mut out[done..]);
            }
            reference::run2(&window[done..], shift, &mut out[done..]);
        }

        $crate::av2_intra::generic::av2_smooth_kernel!($(#[$attr])*; $v4, $narrow, $wide);
    };
}

pub(crate) use av2_intra_kernels;
