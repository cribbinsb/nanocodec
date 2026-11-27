//! VP9 byte-clipped separable interpolation, with ISA-specific row kernels.
//! The unchanged `vp9_mc::vp9_convolve` is the numerical oracle. Unlike AV1,
//! every pass rounds by seven bits and clips to bytes, including intermediates.

#![allow(
    clippy::inline_always,
    reason = "row operations must inline into ISA entry points"
)]

use crate::Isa;
use crate::vp9_mc::Vp9Filter;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128;
#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
mod simd128_dot;
#[cfg(target_arch = "aarch64")]
mod vertical_i8mm;

pub(crate) const MAX_SIDE: usize = 64;

pub(crate) type ConvolveFn =
    unsafe fn(Vp9Filter, &[u8], usize, (usize, usize), (usize, usize), &mut [u8], usize, bool);

pub(crate) fn kernel(isa: Isa) -> ConvolveFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::convolve,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => avx512::convolve,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => {
            if std::arch::is_aarch64_feature_detected!("i8mm")
                && std::arch::is_aarch64_feature_detected!("dotprod")
            {
                neon::convolve_i8mm
            } else if std::arch::is_aarch64_feature_detected!("dotprod") {
                neon::convolve_dotprod::<false>
            } else {
                neon::convolve
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::convolve,
        _ => crate::vp9_mc::vp9_convolve_into,
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod driver;

/// One ISA's VP9 entry point as a type, so that the shared
/// [`crate::mc::ConvolveFn`] wrappers below can name it at compile time (a
/// function pointer cannot carry it).
trait Entry {
    const CONVOLVE: ConvolveFn;
}

struct ScalarEntry;
impl Entry for ScalarEntry {
    const CONVOLVE: ConvolveFn = crate::vp9_mc::vp9_convolve_into;
}

#[cfg(target_arch = "x86_64")]
struct Avx2Entry;
#[cfg(target_arch = "x86_64")]
impl Entry for Avx2Entry {
    const CONVOLVE: ConvolveFn = avx2::convolve;
}

#[cfg(target_arch = "x86_64")]
struct Avx512Entry;
#[cfg(target_arch = "x86_64")]
impl Entry for Avx512Entry {
    const CONVOLVE: ConvolveFn = avx512::convolve;
}

#[cfg(target_arch = "aarch64")]
struct NeonEntry;
#[cfg(target_arch = "aarch64")]
impl Entry for NeonEntry {
    const CONVOLVE: ConvolveFn = neon::convolve;
}

#[cfg(target_arch = "aarch64")]
struct NeonI8mmEntry;
#[cfg(target_arch = "aarch64")]
impl Entry for NeonI8mmEntry {
    const CONVOLVE: ConvolveFn = neon::convolve_i8mm;
}

#[cfg(target_arch = "aarch64")]
struct NeonDotprodEntry;
#[cfg(target_arch = "aarch64")]
impl Entry for NeonDotprodEntry {
    const CONVOLVE: ConvolveFn = neon::convolve_dotprod::<false>;
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
struct Simd128Entry;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl Entry for Simd128Entry {
    const CONVOLVE: ConvolveFn = simd128::convolve;
}

/// The VP9 kernel of entry `E` behind the shared [`crate::mc::ConvolveFn`]
/// contract (`W` wide, filter `F` as an index of [`FAMILIES`]), so that
/// `crate::Filter::Vp9` predicts through [`crate::Dsp::predict`] as the
/// other codecs' filters do (the engine's exact candidate ranking).
///
/// # Safety
///
/// The [`crate::mc::ConvolveFn`] contract: `src` covers the `W + 7` by
/// `h + 7` window, `dst` the `W` by `h` block, offsets are below 16, and
/// the CPU runs `E`'s ISA.
unsafe fn shared<E: Entry, const F: usize, const W: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    // SAFETY: the caller's `ConvolveFn` contract is this entry's window,
    // destination and phase contract at size `W` by `h`.
    unsafe {
        (E::CONVOLVE)(
            FAMILIES[F],
            src,
            src_stride,
            (frac_x, frac_y),
            (W, h),
            dst,
            dst_stride,
            false,
        );
    }
}

/// The filters of [`crate::mc::McFns::vp9`], in its order.
const FAMILIES: [Vp9Filter; 3] = [Vp9Filter::Regular, Vp9Filter::Smooth, Vp9Filter::Sharp];

fn family<E: Entry, const F: usize>() -> [crate::mc::ConvolveFn; 5] {
    [
        shared::<E, F, 4>,
        shared::<E, F, 8>,
        shared::<E, F, 16>,
        shared::<E, F, 32>,
        shared::<E, F, 64>,
    ]
}

fn families_of<E: Entry>() -> [[crate::mc::ConvolveFn; 5]; 3] {
    [family::<E, 0>(), family::<E, 1>(), family::<E, 2>()]
}

/// The VP9 kernels of `isa` for [`crate::mc::McFns::vp9`]: the same entry
/// points [`kernel`] selects.
pub(crate) fn families(isa: Isa) -> [[crate::mc::ConvolveFn; 5]; 3] {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => families_of::<Avx2Entry>(),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => families_of::<Avx512Entry>(),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => {
            if std::arch::is_aarch64_feature_detected!("i8mm")
                && std::arch::is_aarch64_feature_detected!("dotprod")
            {
                families_of::<NeonI8mmEntry>()
            } else if std::arch::is_aarch64_feature_detected!("dotprod") {
                families_of::<NeonDotprodEntry>()
            } else {
                families_of::<NeonEntry>()
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => families_of::<Simd128Entry>(),
        _ => families_of::<ScalarEntry>(),
    }
}
