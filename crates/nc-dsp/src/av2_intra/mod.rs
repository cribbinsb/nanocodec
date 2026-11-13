//! AV2 intra prediction kernels with SIMD versions: the interpolation runs
//! of directional prediction and the smooth predictors.
//!
//! The codec-side logic (edge availability, zones, padding) and the scalar
//! references live in [`crate::intra_prediction::av2`]; that code calls
//! these kernels through [`Av2IntraKernels`], which [`Av2IntraFns`]
//! implements with one ISA's table ([`crate::Dsp::av2_intra_predict`]).
//!
//! # Kernel contract
//!
//! Every SIMD kernel is an `unsafe fn` that executes its ISA's
//! instructions, so it may only be called through a table built after
//! [`crate::Isa::is_supported`]. Loads and stores index their slices, so a
//! wrong length panics rather than reading out of bounds.

use crate::cpu::Isa;
use crate::intra_prediction::av2::{self, Av2IntraKernels, SmoothEdges};

pub(crate) mod generic;

#[cfg(not(target_arch = "aarch64"))]
pub(crate) mod runs;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

/// A directional interpolation run ([`Av2IntraKernels::run4`] /
/// [`Av2IntraKernels::run2`]).
pub(crate) type Av2RunFn = unsafe fn(window: &[u8], shift: usize, out: &mut [u8]);
/// A smooth prediction ([`Av2IntraKernels::smooth`]).
pub(crate) type Av2SmoothFn = unsafe fn(edges: &SmoothEdges<'_>, dst: &mut [u8]);

/// One ISA's AV2 intra kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Av2IntraFns {
    pub run4: Av2RunFn,
    pub run2: Av2RunFn,
    pub smooth: Av2SmoothFn,
}

/// The table of `isa`. Calling its kernels needs `isa`'s support, which
/// `Dsp::with_isa` checks before building a table.
pub(crate) fn dispatch_fns(isa: Isa) -> Av2IntraFns {
    match isa {
        Isa::Scalar => scalar_fns(),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::av2_intra_fns(),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::av2_intra_fns(),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::av2_intra_fns(),
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_supported() is false for ISAs not compiled in"),
    }
}

/// The scalar reference table.
fn scalar_fns() -> Av2IntraFns {
    Av2IntraFns {
        run4: av2::run4,
        run2: av2::run2,
        smooth: av2::smooth,
    }
}

#[allow(unsafe_code)]
impl Av2IntraKernels for Av2IntraFns {
    fn run4(&self, window: &[u8], shift: usize, out: &mut [u8]) {
        assert!(window.len() == out.len() + 3 && shift < av2::FOUR_TAP.len());
        // SAFETY: the table was built for a supported ISA (module docs);
        // the kernel indexes its slices.
        unsafe { (self.run4)(window, shift, out) }
    }

    fn run2(&self, window: &[u8], shift: usize, out: &mut [u8]) {
        assert!(window.len() == out.len() + 1 && shift < av2::FOUR_TAP.len());
        // SAFETY: as for `run4`.
        unsafe { (self.run2)(window, shift, out) }
    }

    fn smooth(&self, edges: &SmoothEdges<'_>, dst: &mut [u8]) {
        let side = edges.side();
        assert!(edges.left.len() == side + 1 && dst.len() == side * side);
        // SAFETY: as for `run4`.
        unsafe { (self.smooth)(edges, dst) }
    }
}
