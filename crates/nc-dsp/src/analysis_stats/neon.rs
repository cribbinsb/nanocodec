//! Four independent blocks per NEON vector; structure loads deinterleave
//! moments and quadrant SADs directly from their existing row buffers.

#![allow(
    clippy::inline_always,
    reason = "ISA primitives and the shared graph must inline into their target-feature wrapper"
)]

use super::generic::{self, Lanes, MAX_LANES};
use super::{AnalysisBlock, AnalysisChanges, AnalysisSummary, Moments};
use core::arch::aarch64::{
    int32x4_t, vaddl_u16, vaddq_s32, vaddq_u32, vdupq_n_s32, vget_low_u16, vld1q_u16, vld2q_u32,
    vld4_u16, vmax_u16, vmaxq_s32, vmaxq_u16, vmaxq_u32, vmovl_u16, vmulq_s32,
    vreinterpretq_s32_u32, vreinterpretq_u16_u32, vreinterpretq_u32_u16, vshrq_n_s32, vst1q_s32,
    vsubq_s32, vuzp1q_u32, vuzp2q_u32,
};

#[derive(Clone, Copy)]
struct Neon(int32x4_t);

impl Lanes for Neon {
    const LANES: usize = 4;
    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: AArch64 NEON, reached only from its wrapper.
        Self(unsafe { vdupq_n_s32(value) })
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: AArch64 NEON, reached only from its wrapper.
        Self(unsafe { vaddq_s32(self.0, other.0) })
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: AArch64 NEON, reached only from its wrapper.
        Self(unsafe { vsubq_s32(self.0, other.0) })
    }
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: AArch64 NEON, reached only from its wrapper.
        Self(unsafe { vmulq_s32(self.0, other.0) })
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: AArch64 NEON, reached only from its wrapper.
        Self(unsafe { vmaxq_s32(self.0, other.0) })
    }
    #[inline(always)]
    fn shr6(self) -> Self {
        // SAFETY: AArch64 NEON; all shifted values are nonnegative.
        Self(unsafe { vshrq_n_s32::<6>(self.0) })
    }
    #[inline(always)]
    fn store(self, out: &mut [i32; MAX_LANES]) {
        // SAFETY: the output array holds at least four lanes.
        unsafe { vst1q_s32(out.as_mut_ptr(), self.0) }
    }
    #[inline(always)]
    unsafe fn moments(src: *const Moments) -> (Self, Self) {
        // SAFETY: a complete batch of four repr(C) u32 moment pairs.
        unsafe {
            let m = vld2q_u32(src.cast());
            (
                Self(vreinterpretq_s32_u32(m.0)),
                Self(vreinterpretq_s32_u32(m.1)),
            )
        }
    }
    #[inline(always)]
    unsafe fn changes(changes: AnalysisChanges<'_>, start: usize) -> (Self, Self) {
        // SAFETY: four luma tuples and two chroma tuples per plane fit.
        unsafe {
            let q = vld4_u16(changes.luma.as_ptr().add(start).cast());
            let sum = vaddq_u32(vaddl_u16(q.0, q.1), vaddl_u16(q.2, q.3));
            let busiest = vmovl_u16(vmax_u16(vmax_u16(q.0, q.1), vmax_u16(q.2, q.3)));
            let chroma_u = vld1q_u16(changes.chroma[0].as_ptr().add(start / 2).cast());
            let chroma_v = vld1q_u16(changes.chroma[1].as_ptr().add(start / 2).cast());
            let uv = vreinterpretq_u32_u16(vmaxq_u16(chroma_u, chroma_v));
            let chosen = if changes.bottom {
                vuzp2q_u32(uv, uv)
            } else {
                vuzp1q_u32(uv, uv)
            };
            let chroma = vmovl_u16(vget_low_u16(vreinterpretq_u16_u32(chosen)));
            (
                Self(vreinterpretq_s32_u32(sum)),
                Self(vreinterpretq_s32_u32(vmaxq_u32(busiest, chroma))),
            )
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn stats(
    m: &[Moments],
    changes: Option<AnalysisChanges<'_>>,
    out: &mut [AnalysisBlock],
) -> AnalysisSummary {
    // SAFETY: the table validates NEON and the caller checks row extents.
    unsafe { generic::run::<Neon>(m, changes, out) }
}
