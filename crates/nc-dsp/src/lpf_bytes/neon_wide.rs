//! Sixteen NEON AV1 filter lines in byte lanes with paired widened words.
//! A three-stage 8x16 byte transpose processes both vertical tiles together;
//! masks and smoothing reuse the shared AV1 filter arithmetic.
//!
//! Layout reference: dav1d 1.5.4 `src/arm/64/loopfilter.S` (sixteen byte
//! masks with paired eight-word smoothing lanes) and `src/arm/64/util.S`
//! (`transpose_8x16b`, byte/word/dword transpose stages), BSD-2-Clause.
//! Copyright VideoLAN/dav1d authors (2018), Martin Storsjo (2015, 2019),
//! and Janne Grunau (2015). This Rust adapter was reviewed against those
//! layouts; it retains our shared arithmetic, with no assembly copied.
//! See `THIRD_PARTY.md` for the source and attribution.
#![allow(clippy::inline_always)]
use super::{Bytes, LANES, PairedBytes, Words};
use core::arch::aarch64::{
    int16x8_t, uint8x16_t, vabdq_u8, vaddq_s16, vandq_u8, vceqq_u8, vcleq_u8, vcombine_u8,
    vcreate_u8, vdupq_n_s16, vdupq_n_u8, veorq_u8, vget_high_u8, vget_low_u8, vgetq_lane_u16,
    vgetq_lane_u32, vld1_u8, vmaxq_u8, vmaxvq_u8, vmovl_high_u8, vmovl_u8, vqaddq_s8, vqaddq_u8,
    vqmovun_high_s16, vqmovun_s16, vqsubq_s8, vqsubq_u8, vreinterpretq_s8_u8,
    vreinterpretq_s16_u16, vreinterpretq_u8_s8, vreinterpretq_u8_u32, vreinterpretq_u16_u8,
    vreinterpretq_u32_u8, vreinterpretq_u32_u16, vshrq_n_s8, vshrq_n_s16, vshrq_n_u8, vst1_u8,
    vsubq_s16, vtrn1q_u8, vtrn1q_u16, vtrn1q_u32, vtrn2q_u8, vtrn2q_u16, vtrn2q_u32, vzip1_u8,
    vzip2_u8,
};
#[derive(Clone, Copy)]
pub(crate) struct B(uint8x16_t);
#[derive(Clone, Copy)]
pub(crate) struct W([int16x8_t; 2]);
impl Bytes for B {
    type W = W;
    #[inline(always)]
    fn splat(v: u8) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vdupq_n_u8(v)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: caller supplies eight readable bytes and enables NEON.
        unsafe { Self::load_pair(p, p) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: caller supplies eight writable bytes and enables NEON.
        unsafe {
            vst1_u8(p, vget_low_u8(self.0));
        }
    }
    #[inline(always)]
    unsafe fn load4(p: *const u8) -> Self {
        // SAFETY: caller supplies four readable bytes and enables NEON.
        unsafe {
            let v = vcreate_u8(u64::from(p.cast::<u32>().read_unaligned()));
            Self(vcombine_u8(v, v))
        }
    }
    #[inline(always)]
    unsafe fn store4(self, p: *mut u8) {
        // SAFETY: caller supplies four writable bytes and enables NEON.
        unsafe {
            p.cast::<u32>()
                .write_unaligned(vgetq_lane_u32::<0>(vreinterpretq_u32_u8(self.0)));
        }
    }
    #[inline(always)]
    unsafe fn store2(self, p: *mut u8) {
        // SAFETY: caller supplies two writable bytes and enables NEON.
        unsafe {
            p.cast::<u16>()
                .write_unaligned(vgetq_lane_u16::<0>(vreinterpretq_u16_u8(self.0)));
        }
    }
    #[inline(always)]
    fn difference(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables NEON.
        unsafe { Self(vabdq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn within(self, limit: u8) -> Self {
        // SAFETY: the ISA entry point enables NEON.
        unsafe { Self(vcleq_u8(self.0, vdupq_n_u8(limit))) }
    }
    #[inline(always)]
    fn max(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vmaxq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub_sat(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vqsubq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn add_sat(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vqaddq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn signed_add(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            Self(vreinterpretq_u8_s8(vqaddq_s8(
                vreinterpretq_s8_u8(self.0),
                vreinterpretq_s8_u8(b.0),
            )))
        }
    }
    #[inline(always)]
    fn signed_sub(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            Self(vreinterpretq_u8_s8(vqsubq_s8(
                vreinterpretq_s8_u8(self.0),
                vreinterpretq_s8_u8(b.0),
            )))
        }
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vandq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn xor(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(veorq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn eq(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vceqq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vshrq_n_u8::<N>(self.0)) }
    }
    #[inline(always)]
    fn signed_shr<const N: i32>(self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            Self(vreinterpretq_u8_s8(vshrq_n_s8::<N>(vreinterpretq_s8_u8(
                self.0,
            ))))
        }
    }
    #[inline(always)]
    fn zip_lo(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            Self(vcombine_u8(
                vzip1_u8(vget_low_u8(self.0), vget_low_u8(b.0)),
                vzip1_u8(vget_high_u8(self.0), vget_high_u8(b.0)),
            ))
        }
    }
    #[inline(always)]
    fn zip_hi(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            Self(vcombine_u8(
                vzip2_u8(vget_low_u8(self.0), vget_low_u8(b.0)),
                vzip2_u8(vget_high_u8(self.0), vget_high_u8(b.0)),
            ))
        }
    }
    #[inline(always)]
    fn any(self) -> bool {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { vmaxvq_u8(self.0) != 0 }
    }
    #[inline(always)]
    fn widen(self) -> W {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            W([
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(self.0))),
                vreinterpretq_s16_u16(vmovl_high_u8(self.0)),
            ])
        }
    }
    #[inline(always)]
    fn narrow(v: W) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vqmovun_high_s16(vqmovun_s16(v.0[0]), v.0[1])) }
    }
    #[inline(always)]
    fn transpose(rows: [Self; LANES]) -> [Self; LANES] {
        // SAFETY: instantiated only inside NEON entry points. Three stages
        // transpose both 8x8 byte groups together, without extracting halves.
        unsafe {
            let a = [
                vtrn1q_u8(rows[0].0, rows[1].0),
                vtrn2q_u8(rows[0].0, rows[1].0),
                vtrn1q_u8(rows[2].0, rows[3].0),
                vtrn2q_u8(rows[2].0, rows[3].0),
                vtrn1q_u8(rows[4].0, rows[5].0),
                vtrn2q_u8(rows[4].0, rows[5].0),
                vtrn1q_u8(rows[6].0, rows[7].0),
                vtrn2q_u8(rows[6].0, rows[7].0),
            ];
            let b = [
                vtrn1q_u16(vreinterpretq_u16_u8(a[0]), vreinterpretq_u16_u8(a[2])),
                vtrn2q_u16(vreinterpretq_u16_u8(a[0]), vreinterpretq_u16_u8(a[2])),
                vtrn1q_u16(vreinterpretq_u16_u8(a[1]), vreinterpretq_u16_u8(a[3])),
                vtrn2q_u16(vreinterpretq_u16_u8(a[1]), vreinterpretq_u16_u8(a[3])),
                vtrn1q_u16(vreinterpretq_u16_u8(a[4]), vreinterpretq_u16_u8(a[6])),
                vtrn2q_u16(vreinterpretq_u16_u8(a[4]), vreinterpretq_u16_u8(a[6])),
                vtrn1q_u16(vreinterpretq_u16_u8(a[5]), vreinterpretq_u16_u8(a[7])),
                vtrn2q_u16(vreinterpretq_u16_u8(a[5]), vreinterpretq_u16_u8(a[7])),
            ];
            [
                Self(vreinterpretq_u8_u32(vtrn1q_u32(
                    vreinterpretq_u32_u16(b[0]),
                    vreinterpretq_u32_u16(b[4]),
                ))),
                Self(vreinterpretq_u8_u32(vtrn1q_u32(
                    vreinterpretq_u32_u16(b[2]),
                    vreinterpretq_u32_u16(b[6]),
                ))),
                Self(vreinterpretq_u8_u32(vtrn1q_u32(
                    vreinterpretq_u32_u16(b[1]),
                    vreinterpretq_u32_u16(b[5]),
                ))),
                Self(vreinterpretq_u8_u32(vtrn1q_u32(
                    vreinterpretq_u32_u16(b[3]),
                    vreinterpretq_u32_u16(b[7]),
                ))),
                Self(vreinterpretq_u8_u32(vtrn2q_u32(
                    vreinterpretq_u32_u16(b[0]),
                    vreinterpretq_u32_u16(b[4]),
                ))),
                Self(vreinterpretq_u8_u32(vtrn2q_u32(
                    vreinterpretq_u32_u16(b[2]),
                    vreinterpretq_u32_u16(b[6]),
                ))),
                Self(vreinterpretq_u8_u32(vtrn2q_u32(
                    vreinterpretq_u32_u16(b[1]),
                    vreinterpretq_u32_u16(b[5]),
                ))),
                Self(vreinterpretq_u8_u32(vtrn2q_u32(
                    vreinterpretq_u32_u16(b[3]),
                    vreinterpretq_u32_u16(b[7]),
                ))),
            ]
        }
    }
}
impl Words for W {
    #[inline(always)]
    fn splat(v: i16) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self([vdupq_n_s16(v); 2]) }
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self([vaddq_s16(self.0[0], b.0[0]), vaddq_s16(self.0[1], b.0[1])]) }
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self([vsubq_s16(self.0[0], b.0[0]), vsubq_s16(self.0[1], b.0[1])]) }
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self([vshrq_n_s16::<N>(self.0[0]), vshrq_n_s16::<N>(self.0[1])]) }
    }
}

impl PairedBytes for B {
    #[inline(always)]
    unsafe fn load_pair(first: *const u8, second: *const u8) -> Self {
        // SAFETY: caller supplies two readable eight-byte spans and NEON.
        unsafe { Self(vcombine_u8(vld1_u8(first), vld1_u8(second))) }
    }
    #[inline(always)]
    unsafe fn store_pair(self, first: *mut u8, second: *mut u8) {
        // SAFETY: caller supplies two writable eight-byte spans and NEON.
        unsafe {
            vst1_u8(first, vget_low_u8(self.0));
            vst1_u8(second, vget_high_u8(self.0));
        }
    }
}
