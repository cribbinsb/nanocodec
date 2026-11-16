//! GFNI byte shifts for the existing eight- and sixteen-line lane layouts.
//! All other operations delegate to AVX2: the filter equations and word sums
//! stay shared. dav1d 1.5.4 `loopfilter_avx512.asm` inspired affine byte
//! shifts (BSD-2, VideoLAN/dav1d authors and Two Orioles; `THIRD_PARTY.md`).
#![allow(clippy::inline_always)]
use super::{Bytes, PairedBytes, QuadBytes};
use core::arch::x86_64::{
    _mm_gf2p8affine_epi64_epi8, _mm_set1_epi64x, _mm256_gf2p8affine_epi64_epi8, _mm256_set1_epi64x,
};

/// GFNI's matrix stores the row producing bit i in byte 7-i. A right shift
/// copies input bit i+shift, extending with zero or the sign bit.
pub(super) const fn shift_matrix(shift: i32, signed: bool) -> i64 {
    assert!(shift >= 0 && shift < 8);
    let mut matrix = 0u64;
    let mut bit = 0;
    while bit < 8 {
        let source = bit + shift;
        let row = if source < 8 {
            1u8 << source
        } else if signed {
            128
        } else {
            0
        };
        matrix |= (row as u64) << ((7 - bit) * 8);
        bit += 1;
    }
    matrix as i64
}

macro_rules! binary {
    ($($op:ident),*) => { $(
        #[inline(always)]
        fn $op(self, b: Self) -> Self { Self(self.0.$op(b.0)) }
    )* };
}
macro_rules! adapter {
    ($name:ident, $inner:path, $affine:ident, $splat:ident) => {
        #[derive(Clone, Copy)]
        pub(crate) struct $name($inner);
        impl Bytes for $name {
            type W = <$inner as Bytes>::W;
            #[inline(always)]
            fn splat(v: u8) -> Self {
                Self(<$inner>::splat(v))
            }
            #[inline(always)]
            unsafe fn load(p: *const u8) -> Self {
                // SAFETY: forwarded readable eight-byte span and ISA contract.
                unsafe { Self(<$inner>::load(p)) }
            }
            #[inline(always)]
            unsafe fn load4(p: *const u8) -> Self {
                // SAFETY: forwarded readable four-byte span and ISA contract.
                unsafe { Self(<$inner>::load4(p)) }
            }
            #[inline(always)]
            unsafe fn store(self, p: *mut u8) {
                // SAFETY: forwarded writable eight-byte span and ISA contract.
                unsafe {
                    self.0.store(p);
                }
            }
            #[inline(always)]
            unsafe fn store4(self, p: *mut u8) {
                // SAFETY: forwarded writable four-byte span and ISA contract.
                unsafe {
                    self.0.store4(p);
                }
            }
            #[inline(always)]
            unsafe fn store2(self, p: *mut u8) {
                // SAFETY: forwarded writable two-byte span and ISA contract.
                unsafe {
                    self.0.store2(p);
                }
            }
            binary!(
                max, sub_sat, add_sat, signed_add, signed_sub, and, xor, eq, zip_lo, zip_hi
            );
            #[inline(always)]
            fn shr<const N: i32>(self) -> Self {
                // SAFETY: the caller's GFNI entry point enables the ISA.
                unsafe {
                    Self($inner($affine::<0>(
                        self.0.0,
                        $splat(const { shift_matrix(N, false) }),
                    )))
                }
            }
            #[inline(always)]
            fn signed_shr<const N: i32>(self) -> Self {
                // SAFETY: as shr; the matrix copies the sign bit into high bits.
                unsafe {
                    Self($inner($affine::<0>(
                        self.0.0,
                        $splat(const { shift_matrix(N, true) }),
                    )))
                }
            }
            #[inline(always)]
            fn any(self) -> bool {
                self.0.any()
            }
            #[inline(always)]
            fn widen(self) -> Self::W {
                self.0.widen()
            }
            #[inline(always)]
            fn narrow(v: Self::W) -> Self {
                Self(<$inner>::narrow(v))
            }
        }
    };
}
adapter!(
    Narrow,
    super::avx2::B,
    _mm_gf2p8affine_epi64_epi8,
    _mm_set1_epi64x
);
adapter!(
    Pair,
    super::avx2_wide::B,
    _mm256_gf2p8affine_epi64_epi8,
    _mm256_set1_epi64x
);

impl PairedBytes for Pair {
    #[inline(always)]
    unsafe fn load_pair(first: *const u8, second: *const u8) -> Self {
        // SAFETY: caller validates both spans and enables AVX2/GFNI.
        unsafe { Self(super::avx2_wide::B::load_pair(first, second)) }
    }
    #[inline(always)]
    unsafe fn store_pair(self, first: *mut u8, second: *mut u8) {
        // SAFETY: caller validates both spans and enables AVX2/GFNI.
        unsafe {
            self.0.store_pair(first, second);
        }
    }
}
impl QuadBytes for Narrow {
    #[inline(always)]
    unsafe fn load_lines4(src: *const u8) -> [Self; 4] {
        // SAFETY: caller validates 32 readable bytes and enables AVX2/GFNI.
        unsafe {
            let rows = super::avx2::B::load_lines4(src);
            [Self(rows[0]), Self(rows[1]), Self(rows[2]), Self(rows[3])]
        }
    }
    #[inline(always)]
    unsafe fn store_lines4(lines: [Self; 4], dst: *mut u8) {
        // SAFETY: caller validates 32 writable bytes and enables AVX2/GFNI.
        unsafe {
            super::avx2::B::store_lines4([lines[0].0, lines[1].0, lines[2].0, lines[3].0], dst);
        }
    }
    #[inline(always)]
    unsafe fn load_halves(low: *const u8, high: *const u8) -> Self {
        // SAFETY: caller validates both four-byte spans and enables AVX2/GFNI.
        unsafe { Self(super::avx2::B::load_halves(low, high)) }
    }
}
