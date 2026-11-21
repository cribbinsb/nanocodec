//! End of block: one past the scan index of the last nonzero level, for any
//! codec's scan (`docs/analysis/encode-kernel-sharing.md` section 8, the
//! scan-statistics family). The codec supplies its scan as an end-of-block
//! table ([`eob_table`]: for each raster position, its scan index plus
//! one); the kernel is then the maximum of that table over the nonzero
//! levels, with no walk along the scan (libaom's and SVT-AV1's quantisers
//! compute it the same way, from their `iscan` tables).
//!
//! Every ISA masks the table by `level != 0` and keeps a running unsigned
//! 16-bit maximum: AVX2 16 levels per step, AVX-512 32 (masked maximum),
//! NEON and SIMD128 8. The 32-bit-level form ([`WideEobFn`], AV2 and VVC
//! levels) narrows its nonzero masks to 16 bits and also returns the OR
//! of the magnitudes when requested (it exceeds an all-ones limit exactly
//! when one magnitude does).

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx512;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128;

use crate::Isa;

/// The end-of-block table of `scan` (raster positions in scan order): entry
/// `scan[i]` is `i + 1`. Scans of up to `u16::MAX` positions.
#[must_use]
pub const fn eob_table<const N: usize>(scan: &[u16; N]) -> [u16; N] {
    assert!(N < u16::MAX as usize);
    let mut table = [0u16; N];
    let mut i = 0;
    while i < N {
        table[scan[i] as usize] = (i + 1) as u16;
        i += 1;
    }
    table
}

/// End of block of `levels` under `table` (same length). `unsafe`: it may
/// run its ISA's instructions, so only a table built for a supported ISA
/// calls it.
pub(crate) type EobFn = unsafe fn(levels: &[i16], table: &[u16]) -> usize;

/// End of block of 32-bit `levels` and the OR of their magnitudes, or zero
/// in the second element for a scan specialized to just the end position.
pub(crate) type WideEobFn = unsafe fn(levels: &[i32], table: &[u16]) -> (usize, u32);

/// The scalar reference of [`WideEobFn`].
pub(crate) fn scalar_wide<const MAGNITUDES: bool>(levels: &[i32], table: &[u16]) -> (usize, u32) {
    levels
        .iter()
        .zip(table)
        .fold((0, 0), |(end, bits), (&level, &position)| {
            let coded = if level == 0 { 0 } else { usize::from(position) };
            let bits = if MAGNITUDES {
                bits | level.unsigned_abs()
            } else {
                0
            };
            (end.max(coded), bits)
        })
}

/// The scalar reference.
pub(crate) fn scalar(levels: &[i16], table: &[u16]) -> usize {
    levels
        .iter()
        .zip(table)
        .filter(|&(&level, _)| level != 0)
        .map(|(_, &end)| usize::from(end))
        .max()
        .unwrap_or(0)
}

/// The 32-bit-level kernel of `isa`.
pub(crate) fn wide_kernel<const MAGNITUDES: bool>(isa: Isa) -> WideEobFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => avx2::end_of_block_wide::<MAGNITUDES>,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::end_of_block_wide::<MAGNITUDES>,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::end_of_block_wide::<MAGNITUDES>,
        _ => scalar_wide::<MAGNITUDES>,
    }
}

/// The kernel of `isa`.
pub(crate) fn kernel(isa: Isa) -> EobFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::end_of_block,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => avx512::end_of_block,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::end_of_block,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::end_of_block,
        _ => scalar,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_inverts_the_scan() {
        let scan = [0u16, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15];
        let table = eob_table(&scan);
        for (i, &pos) in scan.iter().enumerate() {
            assert_eq!(usize::from(table[usize::from(pos)]), i + 1);
        }
        let mut levels = [0i16; 16];
        assert_eq!(scalar(&levels, &table), 0);
        levels[6] = -3;
        assert_eq!(scalar(&levels, &table), 9);
    }
}
