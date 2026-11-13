//! Per-instance SIMD dispatch for shared VVC decoder motion refinements.
//! Scalar references retain the BSD-Clear `VVdeC` arithmetic and captured fixtures.
#![allow(unsafe_code)]
use crate::Isa;
mod bdof;
mod generic;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;
#[cfg(target_arch = "x86_64")]
mod x86;

type Bilinear = unsafe fn((usize, usize), &nc_common::PlaneView<'_>, (usize, usize), &mut [i16]);
type Dmvr = unsafe fn((usize, usize), [&[i16]; 2]) -> ([i32; 2], bool);
type Bdof = unsafe fn((usize, usize), [&[i16]; 2], &mut [u8]);
type Prof = unsafe fn(&[i16; 36], &[[i32; 2]; 16], &mut [i16; 16]);
#[derive(Debug, Clone, Copy)]
pub(crate) struct Kernels {
    pub bilinear: Bilinear,
    pub dmvr: Dmvr,
    pub bdof: Bdof,
    pub prof: Prof,
}
pub(crate) fn kernels(isa: Isa) -> Kernels {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => x86::KERNELS,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::KERNELS,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => wasm::KERNELS,
        _ => Kernels {
            bilinear: crate::vvc_dmvr::bilinear,
            dmvr: crate::vvc_dmvr::scalar,
            bdof: crate::vvc_bdof::scalar,
            prof: crate::vvc_optical_flow::scalar,
        },
    }
}
