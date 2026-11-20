//! NEON loop restoration kernels. The Wiener rows port dav1d 1.5.4
//! `src/arm/64/looprestoration.S` (`wiener_filter7_h/v_8bpc_neon`,
//! BSD-2-Clause, see `THIRD_PARTY.md`): the horizontal pass widens the
//! source to 16 bits, takes the shifted rows with `ext`, multiplies the
//! symmetric pair sums with wrapping 16-bit `mul`/`mla` (exact: see
//! `avx2.rs` for the bound) and adds `128 · centre` with `sqadd`, whose
//! bounds are the intermediate clamp; the vertical pass accumulates with
//! widening `smull`/`smlal` and narrows with saturating shifts; complete
//! stripes run them in the shared schedule ([`super::wiener_stripe`],
//! with five-tap forms where an outer tap is zero). 8-bit
//! self-guided stripes run whole ([`super::sgr_neon`] in the shared
//! [`super::sgr_stripe`] schedule); the self-guided rows, which the other
//! callers use, are [`super::sgr_lanes`] on two `int32x4_t`.

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
    vld1q_s32, vld1q_u8, vld1q_u16, vmaxq_s32, vminq_s32, vmlal_high_n_s16, vmlal_n_s16,
    vmlaq_n_s16, vmovl_high_s16, vmovl_s16, vmovl_u8, vmovl_u16, vmull_high_n_s16, vmull_n_s16,
    vmulq_n_s16, vmulq_s32, vqaddq_s16, vqmovn_s32, vqmovn_u16, vqmovun_s16, vqmovun_s32,
    vqshrun_n_s32, vreinterpretq_s16_u16, vreinterpretq_s32_u32, vreinterpretq_u32_s32,
    vshlq_n_s16, vshlq_n_s32, vshrq_n_s16, vshrq_n_s32, vshrq_n_u32, vst1_u8, vst1q_s16, vst1q_s32,
    vst1q_u8, vst1q_u16, vsubq_s32,
};

use super::kernels::{Kernels, WIENER_ROWS};
use super::sgr_lanes::{SgrLanes, sgr_kernels};
use super::wiener::{ROUND1, SATURATION_BIAS, SATURATION_OFFSET, WienerTaps};
use super::wiener_stripe::WienerLanes;

/// This ISA's table.
pub(super) const KERNELS: Kernels = Kernels {
    wiener_stripe: Some(wiener_stripe),
    sgr_stripe: Some(super::sgr_neon::sgr_stripe),
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

/// `blocks` horizontal steps of the row at `src` into `out`: widen to 16
/// bits, take the shifted rows with `ext`, multiply the symmetric pair
/// sums (the outer pair skipped when `FIVE`: its tap is zero).
///
/// # Safety
///
/// NEON must be enabled in the caller; `src` holds `blocks · 8 + 8`
/// readable bytes and `out` `blocks · 8` writable values.
#[inline(always)]
unsafe fn h_row<const FIVE: bool>(taps: WienerTaps, src: *const u8, out: *mut i16, blocks: usize) {
    let [c0, c1, c2, c3] = taps;
    // SAFETY: the contract.
    unsafe {
        let offset = vdupq_n_s16(SATURATION_OFFSET);
        let bias = vdupq_n_s16(SATURATION_BIAS);
        for block in 0..blocks {
            let j = block * STEP;
            let x = vld1q_u8(src.add(j));
            let low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(x)));
            let high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(x)));
            let p3 = vextq_s16::<3>(low, high);
            let b = vaddq_s16(vextq_s16::<1>(low, high), vextq_s16::<5>(low, high));
            let c = vaddq_s16(vextq_s16::<2>(low, high), vextq_s16::<4>(low, high));
            let centre = vmulq_n_s16(p3, c3);
            let centre = if FIVE {
                centre
            } else {
                vmlaq_n_s16(centre, vaddq_s16(low, vextq_s16::<6>(low, high)), c0)
            };
            let sum = vmlaq_n_s16(vmlaq_n_s16(centre, b, c1), c, c2);
            let saturated = vqaddq_s16(sum, vaddq_s16(vshlq_n_s16::<7>(p3), offset));
            vst1q_s16(out.add(j), vaddq_s16(vshrq_n_s16::<3>(saturated), bias));
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn wiener_h(src: &[u8], taps: WienerTaps, out: &mut [i16], n: usize) {
    let blocks = n.div_ceil(STEP);
    // Every load and store stays within these bounds.
    assert!(src.len() >= blocks * STEP + STEP && out.len() >= blocks * STEP);
    // SAFETY: NEON is enabled; both extents checked above.
    unsafe { h_row::<false>(taps, src.as_ptr(), out.as_mut_ptr(), blocks) };
}

/// Outputs per vertical Wiener step: two 8-lane halves.
const V_STEP: usize = 2 * STEP;

/// Eight vertical Wiener outputs at `rows[k] + j`, saturated to 16 bits
/// unsigned (`clamp(Σ >> 11, 0, 65535)`); rows 0 and 6 are not read when
/// `FIVE` (their tap is zero).
///
/// # Safety
///
/// NEON must be available and every `rows[k]` readable for 8 values at
/// `j`.
#[inline(always)]
unsafe fn vertical8<const FIVE: bool>(
    rows: &[*const i16; WIENER_ROWS],
    j: usize,
    taps: WienerTaps,
) -> uint16x8_t {
    let [c0, c1, c2, c3] = taps;
    // SAFETY: the caller's contract.
    unsafe {
        let row = |k: usize| vld1q_s16(rows[k].add(j));
        let b = vaddq_s16(row(1), row(5));
        let c = vaddq_s16(row(2), row(4));
        let d = row(3);
        let mut low = vmlal_n_s16(
            vmlal_n_s16(vmull_n_s16(vget_low_s16(b), c1), vget_low_s16(c), c2),
            vget_low_s16(d),
            c3,
        );
        let mut high = vmlal_high_n_s16(vmlal_high_n_s16(vmull_high_n_s16(b, c1), c, c2), d, c3);
        if !FIVE {
            let a = vaddq_s16(row(0), row(6));
            low = vmlal_n_s16(low, vget_low_s16(a), c0);
            high = vmlal_high_n_s16(high, a, c0);
        }
        vcombine_u16(
            vqshrun_n_s32::<{ ROUND1 as i32 }>(low),
            vqshrun_n_s32::<{ ROUND1 as i32 }>(high),
        )
    }
}

/// `n` vertical output bytes at `dst` from the seven rows.
///
/// # Safety
///
/// NEON must be enabled in the caller; every row holds `ceil(n / 16) · 16`
/// readable values and `dst` `n` writable bytes.
#[inline(always)]
unsafe fn v_row<const FIVE: bool>(
    rows: &[*const i16; WIENER_ROWS],
    taps: WienerTaps,
    dst: *mut u8,
    n: usize,
) {
    // SAFETY: the contract; a partial last step goes through `tail`.
    unsafe {
        for block in 0..n.div_ceil(V_STEP) {
            let j = block * V_STEP;
            let bytes = vcombine_u8(
                vqmovn_u16(vertical8::<FIVE>(rows, j, taps)),
                vqmovn_u16(vertical8::<FIVE>(rows, j + STEP, taps)),
            );
            if j + V_STEP <= n {
                vst1q_u8(dst.add(j), bytes);
            } else {
                let mut tail = [0u8; V_STEP];
                vst1q_u8(tail.as_mut_ptr(), bytes);
                core::ptr::copy_nonoverlapping(tail.as_ptr(), dst.add(j), n - j);
            }
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn wiener_v(rows: &[&[i16]; WIENER_ROWS], taps: WienerTaps, dst: &mut [u8]) {
    let blocks = dst.len().div_ceil(V_STEP);
    // Every load below stays within these bounds.
    assert!(rows.iter().all(|row| row.len() >= blocks * V_STEP));
    let pointers = rows.map(<[i16]>::as_ptr);
    // SAFETY: NEON is enabled; input extents checked, `dst` is its slice.
    unsafe { v_row::<false>(&pointers, taps, dst.as_mut_ptr(), dst.len()) };
}

/// The NEON rows of [`super::wiener_stripe`].
struct WienerNeon;

impl WienerLanes for WienerNeon {
    type Horizontal = WienerTaps;
    type Vertical = WienerTaps;
    const H_STEP: usize = STEP;
    const V_STEP: usize = V_STEP;

    #[inline(always)]
    unsafe fn horizontal(taps: WienerTaps) -> WienerTaps {
        taps
    }

    #[inline(always)]
    unsafe fn vertical(taps: WienerTaps) -> WienerTaps {
        taps
    }

    #[inline(always)]
    unsafe fn h_row<const FIVE: bool>(
        taps: WienerTaps,
        src: *const u8,
        out: *mut i16,
        blocks: usize,
    ) {
        // SAFETY: the trait's contract.
        unsafe { h_row::<FIVE>(taps, src, out, blocks) }
    }

    #[inline(always)]
    unsafe fn v_row<const FIVE: bool>(
        rows: &[*const i16; WIENER_ROWS],
        taps: WienerTaps,
        dst: *mut u8,
        n: usize,
    ) {
        // SAFETY: the trait's contract.
        unsafe { v_row::<FIVE>(rows, taps, dst, n) }
    }
}

/// A complete Wiener stripe ([`super::wiener_stripe::run`]).
#[target_feature(enable = "neon")]
unsafe fn wiener_stripe(
    horizontal: WienerTaps,
    vertical: WienerTaps,
    source: (&[u8], usize),
    size: (usize, usize),
    ring: &mut [i16],
    output: (&mut nc_common::PlaneViewMut<'_>, usize, usize),
) {
    // SAFETY: NEON is enabled (this function's target feature).
    unsafe {
        super::wiener_stripe::run::<WienerNeon>(horizontal, vertical, source, size, ring, output);
    }
}

/// Eight `i32` lanes for the self-guided kernels.
#[derive(Clone, Copy)]
pub(super) struct Lanes(int32x4_t, int32x4_t);

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
    fn load_i16(src: &[i16]) -> Self {
        let src = &src[..8];
        // SAFETY: see the impl comment.
        unsafe {
            let words = vld1q_s16(src.as_ptr());
            Self(vmovl_s16(vget_low_s16(words)), vmovl_high_s16(words))
        }
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
    fn load_u16(src: &[u16]) -> Self {
        let src = &src[..8];
        // SAFETY: see the impl comment.
        unsafe {
            let words = vld1q_u16(src.as_ptr());
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
    fn store_i16(self, dst: &mut [i16]) {
        let dst = &mut dst[..8];
        // SAFETY: see the impl comment.
        unsafe {
            vst1q_s16(
                dst.as_mut_ptr(),
                vcombine_s16(vqmovn_s32(self.0), vqmovn_s32(self.1)),
            );
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
    fn store_u16_clip(self, max_val: i32, dst: &mut [u16]) {
        let dst = &mut dst[..8];
        // SAFETY: see the impl comment; `vqmovun_s32` clamps negatives to 0.
        unsafe {
            let cap = vdupq_n_s32(max_val);
            let words = vcombine_u16(
                vqmovun_s32(vminq_s32(self.0, cap)),
                vqmovun_s32(vminq_s32(self.1, cap)),
            );
            vst1q_u16(dst.as_mut_ptr(), words);
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
