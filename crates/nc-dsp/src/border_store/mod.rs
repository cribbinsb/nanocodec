//! Stores for picture borders ([`nc_common::Plane::extend_borders_with`]):
//! fills and row copies that bypass the cache where the ISA allows.
//!
//! Every frame the encoder replicates the edges of its source, pyramid and
//! reconstruction into wide borders (the source's luma border is 196
//! samples, for the motion search's reach): about 1.9 MB of stores per 720p
//! frame, read again only by blocks at the picture edge. Through the cache
//! each store first reads its line (write-allocate) and evicts working data.
//! On x86 the AVX2 kernels write them with non-temporal stores, which go
//! straight to memory. HEVC Faster x86, PGO, paired on the reserved cores
//! (10 rounds, 150 f): CPU time -5.7% (lowlight -7.9%, crowd -5.7%,
//! bodycam -3.4%); streams unchanged. Cachegrind with a 4 MiB last-level
//! cache put the border stores at about a third of the encoder's last-level
//! write misses. Other ISAs use ordinary stores (no measured gain).
//!
//! # Kernel contract
//!
//! The kernels take slices and are `unsafe` only because the CPU must
//! support their ISA, which [`crate::Dsp`] guarantees.

use crate::Isa;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;

/// Set every byte of the slice to the value.
pub(crate) type FillFn = unsafe fn(&mut [u8], u8);
/// Copy the second slice into the first (same length).
pub(crate) type CopyFn = unsafe fn(&mut [u8], &[u8]);
/// Order the stores made so far before any later store.
pub(crate) type FenceFn = unsafe fn();

/// The border store kernels of one ISA.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BorderStoreFns {
    pub fill: FillFn,
    pub copy: CopyFn,
    pub fence: FenceFn,
}

pub(crate) fn border_store_fns(isa: Isa) -> BorderStoreFns {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => BorderStoreFns {
            fill: avx2::fill,
            copy: avx2::copy,
            fence: avx2::fence,
        },
        _ => BorderStoreFns { fill, copy, fence },
    }
}

/// Scalar reference: ordinary stores.
fn fill(dst: &mut [u8], value: u8) {
    dst.fill(value);
}

fn copy(dst: &mut [u8], src: &[u8]) {
    dst.copy_from_slice(src);
}

/// Ordinary stores are ordered already.
fn fence() {}
