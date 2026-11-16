//! The interpolation run of directional intra prediction, shared by every
//! codec whose angular predictor blends two neighbouring samples of a
//! projected reference line (`docs/analysis/encode-kernel-sharing.md`
//! section 4):
//!
//! `out[k] = (window[k] · (32 − w) + window[k + 1] · w + 16) >> 5`, with
//! `window.len() == out.len() + 1` and `w <= 32` (`w = 32` copies
//! `window[k + 1]`: whole-sample lines run through the same kernel).
//!
//! - HEVC angular (§8.4.4.2.6): one run per line, `w = iFact`.
//! - AV1 directional (§7.11.2.4): positions in 1/64 sample with
//!   `w = (pos >> 1) & 31`, so the same blend.
//! - AV2 chroma directional (its 2-tap filter, `TWO_TAP_BITS = 5`).
//! - VVC chroma angular (H.266 §8.4.5.2.13, the linear filter `fT`).
//!
//! [`run4`] is the 4-tap form of the same pattern, VVC luma's cubic and
//! Gaussian filters with 6-bit taps:
//! `out[k] = Clip1((Σ t[i] · window[k + i] + 32) >> 6)`. AV2's luma 4-tap filter has 7-bit taps (up to 152),
//! which fit neither the byte weights nor the 16-bit pair sums used here,
//! so it keeps its own kernel (`crate::av2_intra`).
//!
//! The codecs keep their reference preparation (substitution, smoothing,
//! edge upsampling, projection of the side edge) and their angle tables;
//! only the per-line blend is this kernel. Every ISA multiplies interleaved
//! sample pairs by byte weights (x86 `pmaddubsw` and a `pmulhrsw` rounding
//! shift, NEON `umull`/`umlal` and a rounding narrow, SIMD128 widening
//! `extmul`): every `a · (32 − w) + b · w` fits 16 bits, and so does every
//! partial sum of [`run4`] (its positive taps sum to at most 66, its
//! negative ones to at most −12).

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

use crate::Isa;
use crate::intra_prediction::directional::interpolate_run;

/// A run ([`run2`]'s contract). `unsafe`: it may execute its ISA's
/// instructions, so only a table built for a supported ISA calls it.
pub(crate) type RunFn = unsafe fn(window: &[u8], weight: usize, out: &mut [u8]);

/// Shortest run the vector kernels compute with a 4-sample step.
pub(crate) const QUARTER_STEP: usize = 4;

/// The scalar reference run (module docs).
pub(crate) fn run2(window: &[u8], weight: usize, out: &mut [u8]) {
    interpolate_run(window, weight, out);
}

/// Fractional bits of [`run4`]'s taps (VVC `fC` / `fG`, sum 64).
pub(crate) const FOUR_TAP_BITS: u32 = 6;

/// A 4-tap run: `out[k] = Clip1((Σ taps[i] · window[k + i] + 32) >> 6)`,
/// with `window.len() == out.len() + 3`. The taps are VVC's (§8.4.5.2.13):
/// the inner two non-negative, the outer two of one sign (cubic: both at
/// most 0; Gaussian: both at least 0), summing to 64. The scalar reference.
pub(crate) fn run4(window: &[u8], taps: [i8; 4], out: &mut [u8]) {
    assert_eq!(window.len(), out.len() + 3);
    let round = 1 << (FOUR_TAP_BITS - 1);
    for (sample, samples) in out.iter_mut().zip(window.windows(4)) {
        let sum: i32 = (0..4)
            .map(|tap| i32::from(taps[tap]) * i32::from(samples[tap]))
            .sum();
        *sample = ((sum + round) >> FOUR_TAP_BITS).clamp(0, i32::from(u8::MAX)) as u8;
    }
}

/// One ISA's runs, inlined into that ISA's predictor code.
#[allow(unsafe_code)]
pub(crate) trait Runs {
    /// [`run2`] on this ISA.
    ///
    /// # Safety
    ///
    /// The ISA must be enabled in the caller.
    unsafe fn run2(window: &[u8], weight: usize, out: &mut [u8]);
    /// [`run4`] on this ISA.
    ///
    /// # Safety
    ///
    /// The ISA must be enabled in the caller.
    unsafe fn run4(window: &[u8], taps: [i8; 4], out: &mut [u8]);
}

/// The run of `isa`.
pub(crate) fn run2_kernel(isa: Isa) -> RunFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => avx2::run2_entry,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::run2_entry,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::run2_entry,
        _ => run2,
    }
}
