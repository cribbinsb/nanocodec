//! Full byte lanes for two/four independent sixteen-line H.264 edges.
//! Byte decisions and updates stay narrow; delta and strong sums widen only
//! while needed. The shared narrow filter graph also serves NEON/SIMD128.
//! Per-edge SSE transposes keep the proven footprint and layout; wide
//! arithmetic combines their results in independent 128-bit lanes.
#![allow(clippy::inline_always)]
use super::H264Math;
use super::x86::B as Narrow;
use core::arch::x86_64::{
    __m256i, __m512i, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_set_m128i,
    _mm512_castsi128_si512, _mm512_castsi512_si128, _mm512_extracti32x4_epi32, _mm512_inserti32x4,
};

pub(crate) trait BatchBytes: H264Math {
    const EDGES: usize;
    type Pair: BatchBytes;
    fn assemble(values: [Narrow; 4]) -> Self;
    fn split(self) -> [Narrow; 4];
}
#[derive(Clone, Copy)]
pub(crate) struct Pair(__m256i);
#[derive(Clone, Copy)]
pub(crate) struct Quad(__m512i);
impl BatchBytes for Pair {
    type Pair = Pair;
    const EDGES: usize = 2;
    #[inline(always)]
    fn assemble(v: [Narrow; 4]) -> Self {
        // SAFETY: AVX2 enabled by the kernel.
        unsafe { Self(_mm256_set_m128i(v[1].0, v[0].0)) }
    }
    #[inline(always)]
    fn split(self) -> [Narrow; 4] {
        // SAFETY: AVX2 enabled by the kernel.
        unsafe {
            [
                Narrow(_mm256_castsi256_si128(self.0)),
                Narrow(_mm256_extracti128_si256::<1>(self.0)),
                Narrow::splat(0),
                Narrow::splat(0),
            ]
        }
    }
}
impl BatchBytes for Quad {
    type Pair = IclPair;
    const EDGES: usize = 4;
    #[inline(always)]
    fn assemble(v: [Narrow; 4]) -> Self {
        // SAFETY: AVX-512 enabled by the kernel.
        unsafe {
            let a = _mm512_castsi128_si512(v[0].0);
            let a = _mm512_inserti32x4::<1>(a, v[1].0);
            let a = _mm512_inserti32x4::<2>(a, v[2].0);
            Self(_mm512_inserti32x4::<3>(a, v[3].0))
        }
    }
    #[inline(always)]
    fn split(self) -> [Narrow; 4] {
        // SAFETY: AVX-512 enabled by the kernel.
        unsafe {
            [
                Narrow(_mm512_castsi512_si128(self.0)),
                Narrow(_mm512_extracti32x4_epi32::<1>(self.0)),
                Narrow(_mm512_extracti32x4_epi32::<2>(self.0)),
                Narrow(_mm512_extracti32x4_epi32::<3>(self.0)),
            ]
        }
    }
}

macro_rules! word_equations {
    () => {
        #[inline(always)]
        pub(super) fn delta<const HIGH: bool>(
            p1: Source,
            p0: Source,
            q0: Source,
            q1: Source,
        ) -> Reg {
            // SAFETY: enabled ISA; the signed numerator is within +/-1279.
            unsafe {
                let d = shl::<2>(subw(widen::<HIGH>(q0), widen::<HIGH>(p0)));
                let e = subw(widen::<HIGH>(p1), widen::<HIGH>(q1));
                ashr::<3>(addw(addw(d, e), splatw(4)))
            }
        }
        #[inline(always)]
        pub(super) fn strong<const HIGH: bool>(v: [Source; 6]) -> [Reg; 3] {
            let [x3, x2, x1, x0, y0, y1] = [
                widen::<HIGH>(v[0]),
                widen::<HIGH>(v[1]),
                widen::<HIGH>(v[2]),
                widen::<HIGH>(v[3]),
                widen::<HIGH>(v[4]),
                widen::<HIGH>(v[5]),
            ];
            // SAFETY: enabled ISA; sums are at most 2040.
            unsafe {
                let s = addw(addw(x1, x0), y0);
                let s2 = addw(s, x2);
                let (two, four) = (splatw(2), splatw(4));
                [
                    lshr::<3>(addw(addw(shl::<1>(addw(x3, x2)), s2), four)),
                    lshr::<2>(addw(s2, two)),
                    lshr::<3>(addw(addw(addw(s, s2), y1), four)),
                ]
            }
        }
    };
}
macro_rules! word_helpers {
    () => {
        #[inline(always)]
        fn widen<const HIGH: bool>(v: Reg) -> Reg {
            // SAFETY: the caller enables this namespace's ISA.
            unsafe {
                if HIGH {
                    widenhi(v, zero())
                } else {
                    widenlo(v, zero())
                }
            }
        }
        word_equations!();
    };
}
// Intrinsic spellings differ, but the packed layout and arithmetic graph do not.
macro_rules! h264_math {
    ($ty:ty, $ops:ident) => {
        impl H264Math for $ty {
            #[inline(always)]
            fn splat(v: u8) -> Self {
                // SAFETY: these adapters only run inside enabled ISA kernels.
                unsafe { Self($ops::splat(v as i8)) }
            }
            #[inline(always)]
            fn abs_diff(self, b: Self) -> Self {
                // SAFETY: enabled ISA, register operations only.
                unsafe { Self($ops::or($ops::subs(self.0, b.0), $ops::subs(b.0, self.0))) }
            }
            #[inline(always)]
            fn less(self, b: Self) -> Self {
                Self($ops::less(self.0, b.0))
            }
            #[inline(always)]
            fn and(self, b: Self) -> Self {
                // SAFETY: enabled ISA, register operations only.
                unsafe { Self($ops::and(self.0, b.0)) }
            }
            #[inline(always)]
            fn and_not(self, b: Self) -> Self {
                // SAFETY: enabled ISA, register operations only.
                unsafe { Self($ops::andnot(b.0, self.0)) }
            }
            #[inline(always)]
            fn select(self, a: Self, b: Self) -> Self {
                Self($ops::select(self.0, a.0, b.0))
            }
            #[inline(always)]
            fn sub(self, b: Self) -> Self {
                // SAFETY: enabled ISA, register operations only.
                unsafe { Self($ops::sub(self.0, b.0)) }
            }
            #[inline(always)]
            fn add_sat(self, b: Self) -> Self {
                // SAFETY: enabled ISA, register operations only.
                unsafe { Self($ops::adds(self.0, b.0)) }
            }
            #[inline(always)]
            fn sub_sat(self, b: Self) -> Self {
                // SAFETY: enabled ISA, register operations only.
                unsafe { Self($ops::subs(self.0, b.0)) }
            }
            #[inline(always)]
            fn min(self, b: Self) -> Self {
                // SAFETY: enabled ISA, register operations only.
                unsafe { Self($ops::minu(self.0, b.0)) }
            }
            #[inline(always)]
            fn max(self, b: Self) -> Self {
                // SAFETY: enabled ISA, register operations only.
                unsafe { Self($ops::maxu(self.0, b.0)) }
            }
            #[inline(always)]
            fn avg_round(self, b: Self) -> Self {
                // SAFETY: enabled ISA, register operations only.
                unsafe { Self($ops::avg(self.0, b.0)) }
            }
            #[inline(always)]
            fn avg_floor(self, b: Self) -> Self {
                // SAFETY: enabled ISA, register operations only.
                unsafe {
                    let odd = $ops::and($ops::xor(self.0, b.0), $ops::splat(1));
                    Self($ops::sub($ops::avg(self.0, b.0), odd))
                }
            }
            #[inline(always)]
            fn any(self) -> bool {
                $ops::any(self.0)
            }
            #[inline(always)]
            fn normal_delta(p1: Self, p0: Self, q0: Self, q1: Self) -> Self {
                // SAFETY: enabled ISA; widening each half and packing restores byte order.
                unsafe {
                    Self($ops::packs(
                        $ops::delta::<false>(p1.0, p0.0, q0.0, q1.0),
                        $ops::delta::<true>(p1.0, p0.0, q0.0, q1.0),
                    ))
                }
            }
            #[inline(always)]
            fn clip_signed(self, t: Self) -> Self {
                // SAFETY: enabled ISA; tC is at most 27 for H.264 eight-bit.
                unsafe {
                    Self($ops::mins(
                        $ops::maxs(self.0, $ops::sub($ops::zero(), t.0)),
                        t.0,
                    ))
                }
            }
            #[inline(always)]
            fn add_signed(self, delta: Self) -> Self {
                // SAFETY: enabled ISA; positive and negative parts are disjoint.
                unsafe {
                    let z = $ops::zero();
                    let up = $ops::maxs(delta.0, z);
                    let down = $ops::maxs($ops::sub(z, delta.0), z);
                    Self($ops::subs($ops::adds(self.0, up), down))
                }
            }
            #[inline(always)]
            fn sub_signed(self, delta: Self) -> Self {
                // SAFETY: delta is above -128, so negation is exact.
                let delta = unsafe { Self($ops::sub($ops::zero(), delta.0)) };
                H264Math::add_signed(self, delta)
            }
            #[inline(always)]
            fn strong_side(
                x3: Self,
                x2: Self,
                x1: Self,
                x0: Self,
                y0: Self,
                y1: Self,
            ) -> [Self; 3] {
                // SAFETY: enabled ISA; sums <= 2040, narrowed results <= 255.
                unsafe {
                    let low = $ops::strong::<false>([x3.0, x2.0, x1.0, x0.0, y0.0, y1.0]);
                    let high = $ops::strong::<true>([x3.0, x2.0, x1.0, x0.0, y0.0, y1.0]);
                    [
                        Self($ops::packu(low[0], high[0])),
                        Self($ops::packu(low[1], high[1])),
                        Self($ops::packu(low[2], high[2])),
                    ]
                }
            }
        }
    };
}

mod ymm {
    pub(super) use core::arch::x86_64::{
        __m256i as Reg, _mm256_add_epi16 as addw, _mm256_adds_epu8 as adds,
        _mm256_and_si256 as and, _mm256_andnot_si256 as andnot, _mm256_avg_epu8 as avg,
        _mm256_blendv_epi8, _mm256_cmpeq_epi8, _mm256_max_epi8 as maxs, _mm256_max_epu8 as maxu,
        _mm256_min_epi8 as mins, _mm256_min_epu8 as minu, _mm256_movemask_epi8,
        _mm256_or_si256 as or, _mm256_packs_epi16 as packs, _mm256_packus_epi16 as packu,
        _mm256_set1_epi8 as splat, _mm256_set1_epi16 as splatw, _mm256_setzero_si256 as zero,
        _mm256_slli_epi16 as shl, _mm256_srai_epi16 as ashr, _mm256_srli_epi16 as lshr,
        _mm256_sub_epi8 as sub, _mm256_sub_epi16 as subw, _mm256_subs_epu8 as subs,
        _mm256_unpackhi_epi8 as widenhi, _mm256_unpacklo_epi8 as widenlo, _mm256_xor_si256 as xor,
    };
    #[inline(always)]
    pub(super) fn less(a: Reg, b: Reg) -> Reg {
        // SAFETY: called only by the enabled ISA kernel.
        unsafe { xor(_mm256_cmpeq_epi8(maxu(a, b), a), splat(-1)) }
    }
    #[inline(always)]
    pub(super) fn select(m: Reg, a: Reg, b: Reg) -> Reg {
        // SAFETY: called only by the enabled ISA kernel.
        unsafe { _mm256_blendv_epi8(b, a, m) }
    }
    #[inline(always)]
    pub(super) fn any(a: Reg) -> bool {
        // SAFETY: called only by the enabled ISA kernel.
        unsafe { _mm256_movemask_epi8(a) != 0 }
    }
    type Source = Reg;
    word_helpers!();
}

mod zmm {
    pub(super) use core::arch::x86_64::{
        __m512i as Reg, _mm512_add_epi16 as addw, _mm512_adds_epu8 as adds,
        _mm512_and_si512 as and, _mm512_andnot_si512 as andnot, _mm512_avg_epu8 as avg,
        _mm512_cmp_epu8_mask, _mm512_mask_blend_epi8, _mm512_max_epi8 as maxs,
        _mm512_max_epu8 as maxu, _mm512_min_epi8 as mins, _mm512_min_epu8 as minu,
        _mm512_movepi8_mask, _mm512_movm_epi8, _mm512_or_si512 as or, _mm512_packs_epi16 as packs,
        _mm512_packus_epi16 as packu, _mm512_set1_epi8 as splat, _mm512_set1_epi16 as splatw,
        _mm512_setzero_si512 as zero, _mm512_slli_epi16 as shl, _mm512_srai_epi16 as ashr,
        _mm512_srli_epi16 as lshr, _mm512_sub_epi8 as sub, _mm512_sub_epi16 as subw,
        _mm512_subs_epu8 as subs, _mm512_unpackhi_epi8 as widenhi, _mm512_unpacklo_epi8 as widenlo,
        _mm512_xor_si512 as xor,
    };
    #[inline(always)]
    pub(super) fn less(a: Reg, b: Reg) -> Reg {
        // SAFETY: called only by the enabled ISA kernel.
        unsafe { _mm512_movm_epi8(_mm512_cmp_epu8_mask::<1>(a, b)) }
    }
    #[inline(always)]
    pub(super) fn select(m: Reg, a: Reg, b: Reg) -> Reg {
        // SAFETY: called only by the enabled ISA kernel.
        unsafe { _mm512_mask_blend_epi8(_mm512_movepi8_mask(m), b, a) }
    }
    #[inline(always)]
    pub(super) fn any(a: Reg) -> bool {
        // SAFETY: called only by the enabled ISA kernel.
        unsafe { _mm512_movepi8_mask(a) != 0 }
    }
    type Source = Reg;
    word_helpers!();
}

h264_math!(Pair, ymm);
h264_math!(Quad, zmm);

/// Two edges' byte state remains 256 bits. Only the widening operations
/// use all 32 AVX-512 word lanes, so real macroblock pairs use the ICL tier.
#[derive(Clone, Copy)]
pub(crate) struct IclPair(__m256i);
impl BatchBytes for IclPair {
    const EDGES: usize = 2;
    type Pair = Self;
    #[inline(always)]
    fn assemble(v: [Narrow; 4]) -> Self {
        Self(Pair::assemble(v).0)
    }
    #[inline(always)]
    fn split(self) -> [Narrow; 4] {
        Pair(self.0).split()
    }
}
macro_rules! delegate_bytes {
    ($($name:ident),*) => { $(
        #[inline(always)]
        fn $name(self,b:Self)->Self { Self(H264Math::$name(Pair(self.0),Pair(b.0)).0) }
    )* };
}
impl H264Math for IclPair {
    #[inline(always)]
    fn splat(v: u8) -> Self {
        Self(Pair::splat(v).0)
    }
    delegate_bytes!(
        abs_diff,
        less,
        and,
        and_not,
        sub,
        add_sat,
        sub_sat,
        min,
        max,
        avg_round,
        avg_floor,
        clip_signed,
        add_signed,
        sub_signed
    );
    #[inline(always)]
    fn select(self, a: Self, b: Self) -> Self {
        Self(Pair(self.0).select(Pair(a.0), Pair(b.0)).0)
    }
    #[inline(always)]
    fn any(self) -> bool {
        Pair(self.0).any()
    }
    #[inline(always)]
    fn normal_delta(p1: Self, p0: Self, q0: Self, q1: Self) -> Self {
        Self(icl_words::normal(p1.0, p0.0, q0.0, q1.0))
    }
    #[inline(always)]
    fn strong_side(x3: Self, x2: Self, x1: Self, x0: Self, y0: Self, y1: Self) -> [Self; 3] {
        let [a, b, c] = icl_words::strong_bytes([x3.0, x2.0, x1.0, x0.0, y0.0, y1.0]);
        [Self(a), Self(b), Self(c)]
    }
}
mod icl_words {
    use core::arch::x86_64::{
        __m256i as Source, __m512i as Reg, _mm512_add_epi16 as addw, _mm512_cvtepu8_epi16,
        _mm512_cvtsepi16_epi8, _mm512_cvtusepi16_epi8, _mm512_set1_epi16 as splatw,
        _mm512_slli_epi16 as shl, _mm512_srai_epi16 as ashr, _mm512_srli_epi16 as lshr,
        _mm512_sub_epi16 as subw,
    };
    #[inline(always)]
    fn widen<const HIGH: bool>(v: Source) -> Reg {
        // SAFETY: the ICL entry enables AVX-512BW; all 32 bytes widen together.
        unsafe { _mm512_cvtepu8_epi16(v) }
    }
    word_equations!();
    #[inline(always)]
    pub(super) fn normal(p1: Source, p0: Source, q0: Source, q1: Source) -> Source {
        // SAFETY: enabled ISA; signed saturation precedes the exact tC clamp.
        unsafe { _mm512_cvtsepi16_epi8(delta::<false>(p1, p0, q0, q1)) }
    }
    #[inline(always)]
    pub(super) fn strong_bytes(v: [Source; 6]) -> [Source; 3] {
        let [a, b, c] = strong::<false>(v);
        // SAFETY: enabled ISA; shifted strong sums are in 0..=255.
        unsafe {
            [
                _mm512_cvtusepi16_epi8(a),
                _mm512_cvtusepi16_epi8(b),
                _mm512_cvtusepi16_epi8(c),
            ]
        }
    }
}
