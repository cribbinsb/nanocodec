//! NEON byte and word lanes of the packed loop filters (`lpf_bytes`).

use super::{Bytes, LANES, MaskWords, QuadBytes, WordTable, Words};
use core::arch::aarch64::{
    int16x8_t, uint8x8_t, vabd_u8, vaddq_s16, vand_u8, vceq_u8, vcle_u8, vdup_n_u8, vdupq_n_s16,
    veor_u8, vld1_u8, vmax_u8, vmaxv_u8, vmovl_u8, vqadd_s8, vqadd_u8, vqmovun_s16, vqsub_s8,
    vqsub_u8, vreinterpret_s8_u8, vreinterpret_u8_s8, vreinterpretq_s16_u16, vshr_n_s8, vshr_n_u8,
    vshrq_n_s16, vst1_u8, vsubq_s16, vzip1_u8, vzip2_u8,
};
use core::arch::aarch64::{
    uint8x8x4_t, vabsq_s16, vandq_s16, vbicq_s16, vbslq_s16, vceqq_s16, vcgtq_s16, vcreate_u8,
    vld1q_s16, vld4_u8, vmaxq_s16, vmaxvq_u16, vminq_s16, vmulq_s16, vorrq_s16, vqrdmulhq_s16,
    vreinterpretq_u16_s16, vst1q_s16, vst4_u8,
};
#[derive(Clone, Copy)]
pub(crate) struct B(uint8x8_t);
#[derive(Clone, Copy)]
pub(crate) struct W(int16x8_t);
impl Bytes for B {
    type W = W;
    #[inline(always)]
    fn splat(v: u8) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vdup_n_u8(v)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: the caller provides eight readable bytes and enables the ISA.
        unsafe { Self(vld1_u8(p)) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: the caller provides eight writable bytes and enables the ISA.
        unsafe {
            vst1_u8(p, self.0);
        }
    }
    #[inline(always)]
    unsafe fn store2(self, p: *mut u8) {
        // SAFETY: caller supplies two writable bytes and enables this ISA.
        unsafe {
            p.cast::<u16>()
                .write_unaligned(core::arch::aarch64::vget_lane_u16::<0>(
                    core::arch::aarch64::vreinterpret_u16_u8(self.0),
                ));
        }
    }
    #[inline(always)]
    unsafe fn load4(p: *const u8) -> Self {
        // SAFETY: caller supplies four readable bytes and enables this ISA.
        unsafe { Self(vcreate_u8(u64::from(p.cast::<u32>().read_unaligned()))) }
    }
    #[inline(always)]
    unsafe fn store4(self, p: *mut u8) {
        // SAFETY: caller supplies four writable bytes and enables this ISA.
        unsafe {
            p.cast::<u32>()
                .write_unaligned(core::arch::aarch64::vget_lane_u32::<0>(
                    core::arch::aarch64::vreinterpret_u32_u8(self.0),
                ));
        }
    }
    #[inline(always)]
    fn difference(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables NEON.
        unsafe { Self(vabd_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn within(self, limit: u8) -> Self {
        // SAFETY: the ISA entry point enables NEON.
        unsafe { Self(vcle_u8(self.0, vdup_n_u8(limit))) }
    }
    #[inline(always)]
    fn max(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vmax_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub_sat(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vqsub_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn add_sat(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vqadd_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn signed_add(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            Self(vreinterpret_u8_s8(vqadd_s8(
                vreinterpret_s8_u8(self.0),
                vreinterpret_s8_u8(b.0),
            )))
        }
    }
    #[inline(always)]
    fn signed_sub(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            Self(vreinterpret_u8_s8(vqsub_s8(
                vreinterpret_s8_u8(self.0),
                vreinterpret_s8_u8(b.0),
            )))
        }
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vand_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn xor(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(veor_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn eq(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vceq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vshr_n_u8::<N>(self.0)) }
    }
    #[inline(always)]
    fn signed_shr<const N: i32>(self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe {
            Self(vreinterpret_u8_s8(vshr_n_s8::<N>(vreinterpret_s8_u8(
                self.0,
            ))))
        }
    }
    #[inline(always)]
    fn zip_lo(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vzip1_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn zip_hi(self, b: Self) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vzip2_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { vmaxv_u8(self.0) != 0 }
    }
    #[inline(always)]
    fn widen(self) -> W {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { W(vreinterpretq_s16_u16(vmovl_u8(self.0))) }
    }
    #[inline(always)]
    fn narrow(v: W) -> Self {
        // SAFETY: the ISA entry point enables these instructions.
        unsafe { Self(vqmovun_s16(v.0)) }
    }
}
impl Words for W {
    #[inline(always)]
    fn splat(v: i16) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vdupq_n_s16(v)) }
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vaddq_s16(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vsubq_s16(self.0, b.0)) }
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vshrq_n_s16::<N>(self.0)) }
    }
}
impl MaskWords for W {
    #[inline(always)]
    fn lookup(self, table: &WordTable) -> Self {
        // SAFETY: both byte tables contain sixteen bytes; caller enables the ISA.
        unsafe {
            let index = core::arch::aarch64::vmovn_u16(vreinterpretq_u16_s16(self.0));
            let low = core::arch::aarch64::vqtbl1_u8(
                core::arch::aarch64::vld1q_u8(table.low.as_ptr()),
                index,
            );
            let high = core::arch::aarch64::vqtbl1_u8(
                core::arch::aarch64::vld1q_u8(table.high.as_ptr()),
                index,
            );
            Self(core::arch::aarch64::vreinterpretq_s16_u8(
                core::arch::aarch64::vcombine_u8(vzip1_u8(low, high), vzip2_u8(low, high)),
            ))
        }
    }
    #[inline(always)]
    fn load(src: &[i16; LANES]) -> Self {
        // SAFETY: `src` holds eight values; only reached through the ISA
        // entry point.
        unsafe { Self(vld1q_s16(src.as_ptr())) }
    }
    #[inline(always)]
    fn store(self, dst: &mut [i16; LANES]) {
        // SAFETY: as `load`.
        unsafe { vst1q_s16(dst.as_mut_ptr(), self.0) }
    }
    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vabsq_s16(self.0)) }
    }
    #[inline(always)]
    fn min(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vminq_s16(self.0, b.0)) }
    }
    #[inline(always)]
    fn max(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vmaxq_s16(self.0, b.0)) }
    }
    #[inline(always)]
    fn mul(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vmulq_s16(self.0, b.0)) }
    }
    #[inline(always)]
    fn mul_q15_round(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vqrdmulhq_s16(self.0, b.0)) }
    }
    #[inline(always)]
    fn gt(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vreinterpretq_s16_u16(vcgtq_s16(self.0, b.0))) }
    }
    #[inline(always)]
    fn eq(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vreinterpretq_s16_u16(vceqq_s16(self.0, b.0))) }
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vandq_s16(self.0, b.0)) }
    }
    #[inline(always)]
    fn or(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vorrq_s16(self.0, b.0)) }
    }
    #[inline(always)]
    fn and_not(self, b: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe { Self(vbicq_s16(self.0, b.0)) }
    }
    #[inline(always)]
    fn select(self, if_set: Self, if_clear: Self) -> Self {
        // SAFETY: only reached through the ISA entry point.
        unsafe {
            Self(vbslq_s16(
                vreinterpretq_u16_s16(self.0),
                if_set.0,
                if_clear.0,
            ))
        }
    }
    #[inline(always)]
    fn any(self) -> bool {
        // SAFETY: only reached through the ISA entry point.
        unsafe { vmaxvq_u16(vreinterpretq_u16_s16(self.0)) != 0 }
    }
}
impl QuadBytes for B {
    #[inline(always)]
    unsafe fn load_lines4(src: *const u8) -> [Self; 4] {
        // SAFETY: the caller provides 32 readable bytes and the ISA.
        let lines = unsafe { vld4_u8(src) };
        [Self(lines.0), Self(lines.1), Self(lines.2), Self(lines.3)]
    }
    #[inline(always)]
    unsafe fn store_lines4(lines: [Self; 4], dst: *mut u8) {
        let [a, b, c, d] = lines.map(|line| line.0);
        // SAFETY: the caller provides 32 writable bytes and the ISA.
        unsafe { vst4_u8(dst, uint8x8x4_t(a, b, c, d)) }
    }
    #[inline(always)]
    unsafe fn load_halves(low: *const u8, high: *const u8) -> Self {
        // SAFETY: the caller provides four readable bytes at each pointer
        // and the ISA.
        unsafe {
            let half = |bytes: *const u8| u64::from(bytes.cast::<u32>().read_unaligned());
            Self(vcreate_u8(half(low) | half(high) << 32))
        }
    }
}
