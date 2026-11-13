//! NEON loop restoration kernels. The Wiener rows port dav1d 1.5.4
//! `src/arm/64/looprestoration.S` (`wiener_filter7_h/v_8bpc_neon`,
//! BSD-2-Clause, see `THIRD_PARTY.md`): the horizontal pass widens the
//! source to 16 bits, takes the shifted rows with `ext`, multiplies the
//! symmetric pair sums with wrapping 16-bit `mul`/`mla` (exact: see
//! `avx2.rs` for the bound) and adds `128 · centre` with `sqadd`, whose
//! bounds are the intermediate clamp; the vertical pass accumulates with
//! widening `smull`/`smlal` and narrows with saturating shifts. The
//! self-guided rows are [`super::sgr_lanes`] on two `int32x4_t`.

#![allow(
    clippy::many_single_char_names,
    reason = "the spec's and dav1d's sample, tap and statistic names"
)]
#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use core::arch::aarch64::{
    int32x4_t, uint16x8_t, vaddq_s16, vaddq_s32, vceqq_s32, vcombine_s16, vcombine_u8,
    vcombine_u16, vcvtq_f32_s32, vcvtq_s32_f32, vdivq_f32, vdupq_n_s16, vdupq_n_s32, vextq_s16,
    vget_high_u8, vget_high_u16, vget_low_s16, vget_low_u8, vget_low_u16, vld1_u8, vld1q_s16,
    vld1q_s32, vld1q_u8, vmaxq_s32, vminq_s32, vmlal_high_n_s16, vmlal_n_s16, vmlaq_n_s16,
    vmovl_u8, vmovl_u16, vmull_high_n_s16, vmull_n_s16, vmulq_n_s16, vmulq_s32, vqaddq_s16,
    vqmovn_s32, vqmovn_u16, vqmovun_s16, vqshrun_n_s32, vreinterpretq_s16_u16,
    vreinterpretq_s32_u32, vreinterpretq_u32_s32, vshlq_n_s16, vshlq_n_s32, vshrq_n_s16,
    vshrq_n_s32, vshrq_n_u32, vst1_u8, vst1q_s16, vst1q_s32, vst1q_u8, vsubq_s32,
};

use super::kernels::{Kernels, WIENER_ROWS};
use super::sgr_lanes::{SgrLanes, sgr_kernels};
use super::wiener::{ROUND1, SATURATION_BIAS, SATURATION_OFFSET, WienerTaps};

/// This ISA's table.
pub(super) const KERNELS: Kernels = Kernels {
    wiener_stripe: None,
    wiener_h,
    wiener_v,
    box_sums3,
    box_sums5,
    box_ab3,
    box_ab5,
    between,
    on,
    around,
    blend,
};

/// Outputs per Wiener step.
const STEP: usize = 8;

#[target_feature(enable = "neon")]
unsafe fn wiener_h(src: &[u8], taps: WienerTaps, out: &mut [i16], n: usize) {
    let blocks = n.div_ceil(STEP);
    // Every load and store below stays within these bounds.
    assert!(src.len() >= blocks * STEP + STEP && out.len() >= blocks * STEP);
    let [c0, c1, c2, c3] = taps;
    let offset = vdupq_n_s16(SATURATION_OFFSET);
    let bias = vdupq_n_s16(SATURATION_BIAS);
    for block in 0..blocks {
        let j = block * STEP;
        // SAFETY: NEON is enabled; bytes j .. j + 16 lie in `src` (checked
        // above).
        let x = unsafe { vld1q_u8(src.as_ptr().add(j)) };
        let low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(x)));
        let high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(x)));
        let p3 = vextq_s16::<3>(low, high);
        let a = vaddq_s16(low, vextq_s16::<6>(low, high));
        let b = vaddq_s16(vextq_s16::<1>(low, high), vextq_s16::<5>(low, high));
        let c = vaddq_s16(vextq_s16::<2>(low, high), vextq_s16::<4>(low, high));
        let sum = vmlaq_n_s16(
            vmlaq_n_s16(vmlaq_n_s16(vmulq_n_s16(p3, c3), a, c0), b, c1),
            c,
            c2,
        );
        let saturated = vqaddq_s16(sum, vaddq_s16(vshlq_n_s16::<7>(p3), offset));
        let result = vaddq_s16(vshrq_n_s16::<3>(saturated), bias);
        // SAFETY: NEON is enabled; values j .. j + 8 lie in `out`.
        unsafe { vst1q_s16(out.as_mut_ptr().add(j), result) };
    }
}

/// Outputs per vertical Wiener step: two 8-lane halves.
const V_STEP: usize = 2 * STEP;

/// Eight vertical Wiener outputs at `rows[k] + j`, saturated to 16 bits
/// unsigned (`clamp(Σ >> 11, 0, 65535)`).
///
/// # Safety
///
/// NEON must be available and every `rows[k]` readable for 8 values at
/// `j`.
#[inline(always)]
unsafe fn vertical8(rows: &[*const i16; WIENER_ROWS], j: usize, taps: WienerTaps) -> uint16x8_t {
    let [c0, c1, c2, c3] = taps;
    // SAFETY: the caller's contract.
    unsafe {
        let row = |k: usize| vld1q_s16(rows[k].add(j));
        let a = vaddq_s16(row(0), row(6));
        let b = vaddq_s16(row(1), row(5));
        let c = vaddq_s16(row(2), row(4));
        let d = row(3);
        let low = vmlal_n_s16(
            vmlal_n_s16(
                vmlal_n_s16(vmull_n_s16(vget_low_s16(a), c0), vget_low_s16(b), c1),
                vget_low_s16(c),
                c2,
            ),
            vget_low_s16(d),
            c3,
        );
        let high = vmlal_high_n_s16(
            vmlal_high_n_s16(vmlal_high_n_s16(vmull_high_n_s16(a, c0), b, c1), c, c2),
            d,
            c3,
        );
        vcombine_u16(
            vqshrun_n_s32::<{ ROUND1 as i32 }>(low),
            vqshrun_n_s32::<{ ROUND1 as i32 }>(high),
        )
    }
}

#[target_feature(enable = "neon")]
unsafe fn wiener_v(rows: &[&[i16]; WIENER_ROWS], taps: WienerTaps, dst: &mut [u8]) {
    let n = dst.len();
    let blocks = n.div_ceil(V_STEP);
    // Every load below stays within these bounds.
    assert!(rows.iter().all(|row| row.len() >= blocks * V_STEP));
    let pointers = rows.map(<[i16]>::as_ptr);
    for block in 0..blocks {
        let j = block * V_STEP;
        // SAFETY: NEON is enabled; values j .. j + 16 of every row lie in
        // it (checked above).
        let bytes = unsafe {
            vcombine_u8(
                vqmovn_u16(vertical8(&pointers, j, taps)),
                vqmovn_u16(vertical8(&pointers, j + STEP, taps)),
            )
        };
        if j + V_STEP <= n {
            // SAFETY: NEON is enabled; bytes j .. j + 16 lie in `dst`.
            unsafe { vst1q_u8(dst.as_mut_ptr().add(j), bytes) };
        } else {
            let mut tail = [0u8; V_STEP];
            // SAFETY: NEON is enabled; the store covers `tail`.
            unsafe { vst1q_u8(tail.as_mut_ptr(), bytes) };
            dst[j..].copy_from_slice(&tail[..n - j]);
        }
    }
}

/// Eight `i32` lanes for the self-guided kernels.
#[derive(Clone, Copy)]
struct Lanes(int32x4_t, int32x4_t);

impl Lanes {
    /// `f` on both halves.
    #[inline(always)]
    fn map(self, f: impl Fn(int32x4_t) -> int32x4_t) -> Self {
        Self(f(self.0), f(self.1))
    }

    /// `f` on both halves of `self` and `other`.
    #[inline(always)]
    fn zip(self, other: Self, f: impl Fn(int32x4_t, int32x4_t) -> int32x4_t) -> Self {
        Self(f(self.0, other.0), f(self.1, other.1))
    }
}

// SAFETY (every unsafe block): these values exist only inside the NEON
// kernels of this module; loads and stores cover checked subslices.
impl SgrLanes for Lanes {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: see the impl comment.
        let v = unsafe { vdupq_n_s32(value) };
        Self(v, v)
    }
    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        let src = &src[..8];
        // SAFETY: see the impl comment.
        unsafe { Self(vld1q_s32(src.as_ptr()), vld1q_s32(src[4..].as_ptr())) }
    }
    #[inline(always)]
    fn load_u8(src: &[u8]) -> Self {
        let src = &src[..8];
        // SAFETY: see the impl comment.
        unsafe {
            let words = vmovl_u8(vld1_u8(src.as_ptr()));
            Self(
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(words))),
                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(words))),
            )
        }
    }
    #[inline(always)]
    fn store(self, dst: &mut [i32]) {
        let dst = &mut dst[..8];
        // SAFETY: see the impl comment.
        unsafe {
            vst1q_s32(dst.as_mut_ptr(), self.0);
            vst1q_s32(dst[4..].as_mut_ptr(), self.1);
        }
    }
    #[inline(always)]
    fn store_u8_sat(self, dst: &mut [u8]) {
        let dst = &mut dst[..8];
        // SAFETY: see the impl comment.
        unsafe {
            let words = vcombine_s16(vqmovn_s32(self.0), vqmovn_s32(self.1));
            vst1_u8(dst.as_mut_ptr(), vqmovun_s16(words));
        }
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: see the impl comment.
        self.zip(other, |a, b| unsafe { vaddq_s32(a, b) })
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: see the impl comment.
        self.zip(other, |a, b| unsafe { vsubq_s32(a, b) })
    }
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: see the impl comment.
        self.zip(other, |a, b| unsafe { vmulq_s32(a, b) })
    }
    #[inline(always)]
    fn shl<const BITS: i32>(self) -> Self {
        // SAFETY: see the impl comment.
        self.map(|a| unsafe { vshlq_n_s32::<BITS>(a) })
    }
    #[inline(always)]
    fn shr<const BITS: i32>(self) -> Self {
        // SAFETY: see the impl comment.
        self.map(|a| unsafe {
            vreinterpretq_s32_u32(vshrq_n_u32::<BITS>(vreinterpretq_u32_s32(a)))
        })
    }
    #[inline(always)]
    fn sra<const BITS: i32>(self) -> Self {
        // SAFETY: see the impl comment.
        self.map(|a| unsafe { vshrq_n_s32::<BITS>(a) })
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        // SAFETY: see the impl comment.
        self.zip(other, |a, b| unsafe { vminq_s32(a, b) })
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // SAFETY: see the impl comment.
        self.zip(other, |a, b| unsafe { vmaxq_s32(a, b) })
    }
    #[inline(always)]
    fn eq(self, other: Self) -> Self {
        // SAFETY: see the impl comment.
        self.zip(other, |a, b| unsafe {
            vreinterpretq_s32_u32(vceqq_s32(a, b))
        })
    }
    #[inline(always)]
    fn div_floor(self, den: Self) -> Self {
        // SAFETY: see the impl comment.
        self.zip(den, |a, b| unsafe {
            vcvtq_s32_f32(vdivq_f32(vcvtq_f32_s32(a), vcvtq_f32_s32(b)))
        })
    }
}

sgr_kernels!(#[target_feature(enable = "neon")]; Lanes);
