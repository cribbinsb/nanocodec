//! NEON lanes of the byte-lane H.264 deblocking filter (`h264_bytes`).
//!
//! `delta` widens with `usubl`, shifts and narrows with the saturating
//! rounding `sqrshrn` (its rounding is the filter's `+ 4 >> 3`); `p0 +
//! delta` is the unsigned-saturating signed accumulate `usqadd`. The strong
//! sums widen to u16 (`uaddl`/`uaddw`) and narrow with rounding (`rshrn`).
#![allow(clippy::inline_always)] // lane operations inline into the kernels

use super::H264Bytes;
use core::arch::aarch64::{
    uint8x8_t, uint8x16_t, uint32x4_t, vabdq_u8, vaddl_high_u8, vaddl_u8, vaddq_s16, vaddq_u16,
    vaddw_high_u8, vaddw_u8, vandq_u8, vbicq_u8, vbslq_u8, vcltq_u8, vcombine_s8, vcombine_u8,
    vcreate_u8, vdupq_n_u8, vget_high_u8, vget_low_u8, vhaddq_u8, vld1_u8, vld1q_lane_u32,
    vld1q_u8, vmaxq_s8, vmaxq_u8, vmaxvq_u8, vminq_s8, vminq_u8, vnegq_s8, vqaddq_u8,
    vqrshrn_n_s16, vqsubq_u8, vreinterpretq_s8_u8, vreinterpretq_s16_u16, vreinterpretq_u8_s8,
    vreinterpretq_u8_u32, vreinterpretq_u16_u8, vreinterpretq_u32_u8, vreinterpretq_u32_u16,
    vrhaddq_u8, vrshrn_n_u16, vshlq_n_s16, vshlq_n_u16, vsqaddq_u8, vst1_u8, vst1q_lane_u32,
    vst1q_u8, vsubl_high_u8, vsubl_u8, vsubq_u8, vtrn1q_u8, vtrn1q_u16, vtrn1q_u32, vtrn2q_u8,
    vtrn2q_u16, vtrn2q_u32, vuzp1q_u8, vuzp2q_u8, vzip1q_u8, vzip2q_u8,
};

/// Sixteen byte lanes in one NEON register.
#[derive(Clone, Copy)]
pub(crate) struct B(uint8x16_t);

// SAFETY (every intrinsic below): NEON is baseline on AArch64 and the
// kernels using these lanes enable it; register-only intrinsics have no
// further precondition. Memory accesses state theirs.
impl H264Bytes for B {
    #[inline(always)]
    fn splat(v: u8) -> Self {
        // SAFETY: register-only (see above).
        unsafe { Self(vdupq_n_u8(v)) }
    }
    #[inline(always)]
    fn from_halves(lo: u64, hi: u64) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vcombine_u8(vcreate_u8(lo), vcreate_u8(hi))) }
    }
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: the caller provides 16 readable bytes.
        unsafe { Self(vld1q_u8(p)) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: the caller provides 16 writable bytes.
        unsafe { vst1q_u8(p, self.0) }
    }
    #[inline(always)]
    unsafe fn load_halves(lo: *const u8, hi: *const u8) -> Self {
        // SAFETY: the caller provides 8 readable bytes at each pointer.
        unsafe { Self(vcombine_u8(vld1_u8(lo), vld1_u8(hi))) }
    }
    #[inline(always)]
    unsafe fn store_halves(self, lo: *mut u8, hi: *mut u8) {
        // SAFETY: the caller provides 8 writable bytes at each pointer.
        unsafe {
            vst1_u8(lo, vget_low_u8(self.0));
            vst1_u8(hi, vget_high_u8(self.0));
        }
    }
    #[inline(always)]
    unsafe fn load_rows_8(rows: [*const u8; 16]) -> [Self; 8] {
        // Rows `i` and `i + 8` share a vector, so one pass of trn stages
        // transposes two 8x8 tiles.
        let mut pairs = [Self::splat(0); 8];
        for (i, pair) in pairs.iter_mut().enumerate() {
            // SAFETY: the caller provides 8 readable bytes at each row.
            *pair = unsafe { Self::load_halves(rows[i], rows[i + 8]) };
        }
        transpose_8x8_pairs(pairs)
    }
    #[inline(always)]
    unsafe fn store_rows_8(columns: [Self; 8], rows: [*mut u8; 16]) {
        // The pair transpose is its own inverse.
        for (i, pair) in transpose_8x8_pairs(columns).into_iter().enumerate() {
            // SAFETY: the caller provides 8 writable bytes at each row.
            unsafe { pair.store_halves(rows[i], rows[i + 8]) };
        }
    }
    #[inline(always)]
    unsafe fn load_rows_4(rows: [*const u8; 16]) -> [Self; 4] {
        let mut quads = [Self::splat(0); 4];
        for (k, quad) in quads.iter_mut().enumerate() {
            // SAFETY: the caller provides 4 readable bytes at each row
            // (`ld1` lane loads need no alignment).
            unsafe {
                let mut v = vreinterpretq_u32_u8(vdupq_n_u8(0));
                v = vld1q_lane_u32::<0>(rows[4 * k].cast(), v);
                v = vld1q_lane_u32::<1>(rows[4 * k + 1].cast(), v);
                v = vld1q_lane_u32::<2>(rows[4 * k + 2].cast(), v);
                v = vld1q_lane_u32::<3>(rows[4 * k + 3].cast(), v);
                *quad = Self(vreinterpretq_u8_u32(v));
            }
        }
        transpose_16x4(quads)
    }
    #[inline(always)]
    unsafe fn store_rows_4(columns: [Self; 4], rows: [*mut u8; 16]) {
        for (k, quad) in untranspose_16x4(columns).into_iter().enumerate() {
            // SAFETY: the caller provides 4 writable bytes at each row.
            unsafe {
                let v = vreinterpretq_u32_u8(quad.0);
                vst1q_lane_u32::<0>(rows[4 * k].cast(), v);
                vst1q_lane_u32::<1>(rows[4 * k + 1].cast(), v);
                vst1q_lane_u32::<2>(rows[4 * k + 2].cast(), v);
                vst1q_lane_u32::<3>(rows[4 * k + 3].cast(), v);
            }
        }
    }
    #[inline(always)]
    fn abs_diff(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vabdq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn less(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vcltq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn and(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vandq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn and_not(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vbicq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn select(self, a: Self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vbslq_u8(self.0, a.0, b.0)) }
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vsubq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn add_sat(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vqaddq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn sub_sat(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vqsubq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn min(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vminq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn max(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vmaxq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn avg_round(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vrhaddq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn avg_floor(self, b: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vhaddq_u8(self.0, b.0)) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        // SAFETY: register-only.
        unsafe { vmaxvq_u8(self.0) != 0 }
    }
    #[inline(always)]
    fn normal_delta(p1: Self, p0: Self, q0: Self, q1: Self) -> Self {
        // SAFETY: register-only. The u16 differences reinterpret as the
        // exact signed differences (|x| ≤ 255); 4·255 + 255 fits i16.
        unsafe {
            let low = vaddq_s16(
                vshlq_n_s16::<2>(vreinterpretq_s16_u16(vsubl_u8(
                    vget_low_u8(q0.0),
                    vget_low_u8(p0.0),
                ))),
                vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(p1.0), vget_low_u8(q1.0))),
            );
            let high = vaddq_s16(
                vshlq_n_s16::<2>(vreinterpretq_s16_u16(vsubl_high_u8(q0.0, p0.0))),
                vreinterpretq_s16_u16(vsubl_high_u8(p1.0, q1.0)),
            );
            Self(vreinterpretq_u8_s8(vcombine_s8(
                vqrshrn_n_s16::<3>(low),
                vqrshrn_n_s16::<3>(high),
            )))
        }
    }
    #[inline(always)]
    fn clip_signed(self, t: Self) -> Self {
        // SAFETY: register-only.
        unsafe {
            let t = vreinterpretq_s8_u8(t.0);
            Self(vreinterpretq_u8_s8(vminq_s8(
                vmaxq_s8(vreinterpretq_s8_u8(self.0), vnegq_s8(t)),
                t,
            )))
        }
    }
    #[inline(always)]
    fn add_signed(self, delta: Self) -> Self {
        // SAFETY: register-only.
        unsafe { Self(vsqaddq_u8(self.0, vreinterpretq_s8_u8(delta.0))) }
    }
    #[inline(always)]
    fn sub_signed(self, delta: Self) -> Self {
        // SAFETY: register-only; `delta > −128`, so its negation is exact.
        unsafe { Self(vsqaddq_u8(self.0, vnegq_s8(vreinterpretq_s8_u8(delta.0)))) }
    }
    #[inline(always)]
    fn strong_side(x3: Self, x2: Self, x1: Self, x0: Self, y0: Self, y1: Self) -> [Self; 3] {
        // SAFETY: register-only. Sums are at most 8 · 255, inside u16.
        unsafe {
            let low = strong_half([
                vget_low_u8(x3.0),
                vget_low_u8(x2.0),
                vget_low_u8(x1.0),
                vget_low_u8(x0.0),
                vget_low_u8(y0.0),
                vget_low_u8(y1.0),
            ]);
            // `vaddl_high_u8` / `vaddw_high_u8` read the high halves directly.
            let s = vaddw_high_u8(vaddl_high_u8(x1.0, x0.0), y0.0);
            let s2 = vaddw_high_u8(s, x2.0);
            let high = [
                vrshrn_n_u16::<3>(vaddq_u16(vshlq_n_u16::<1>(vaddl_high_u8(x3.0, x2.0)), s2)),
                vrshrn_n_u16::<2>(s2),
                vrshrn_n_u16::<3>(vaddw_high_u8(vaddq_u16(s, s2), y1.0)),
            ];
            [
                Self(vcombine_u8(low[0], high[0])),
                Self(vcombine_u8(low[1], high[1])),
                Self(vcombine_u8(low[2], high[2])),
            ]
        }
    }
}

/// Two 8x8 byte transposes at once: byte `j` of the low (high) half of
/// output `i` is byte `i` of the low (high) half of input `j`. Its own
/// inverse.
#[inline(always)]
fn transpose_8x8_pairs(r: [B; 8]) -> [B; 8] {
    // SAFETY: register-only. `trn` pairs elements within each 64-bit
    // half for 8- and 16-bit elements, and the 32-bit stage pairs
    // element 0 with 0 and 2 with 2: the two halves never mix.
    unsafe {
        let b: [uint8x16_t; 8] = [
            vtrn1q_u8(r[0].0, r[1].0),
            vtrn2q_u8(r[0].0, r[1].0),
            vtrn1q_u8(r[2].0, r[3].0),
            vtrn2q_u8(r[2].0, r[3].0),
            vtrn1q_u8(r[4].0, r[5].0),
            vtrn2q_u8(r[4].0, r[5].0),
            vtrn1q_u8(r[6].0, r[7].0),
            vtrn2q_u8(r[6].0, r[7].0),
        ];
        // Columns {0, 4}, {2, 6}, {1, 5}, {3, 7} of rows 0..4, then of
        // rows 4..8.
        let w = [
            trn_u16::<true>(b[0], b[2]),
            trn_u16::<false>(b[0], b[2]),
            trn_u16::<true>(b[1], b[3]),
            trn_u16::<false>(b[1], b[3]),
            trn_u16::<true>(b[4], b[6]),
            trn_u16::<false>(b[4], b[6]),
            trn_u16::<true>(b[5], b[7]),
            trn_u16::<false>(b[5], b[7]),
        ];
        [
            B(vreinterpretq_u8_u32(vtrn1q_u32(w[0], w[4]))),
            B(vreinterpretq_u8_u32(vtrn1q_u32(w[2], w[6]))),
            B(vreinterpretq_u8_u32(vtrn1q_u32(w[1], w[5]))),
            B(vreinterpretq_u8_u32(vtrn1q_u32(w[3], w[7]))),
            B(vreinterpretq_u8_u32(vtrn2q_u32(w[0], w[4]))),
            B(vreinterpretq_u8_u32(vtrn2q_u32(w[2], w[6]))),
            B(vreinterpretq_u8_u32(vtrn2q_u32(w[1], w[5]))),
            B(vreinterpretq_u8_u32(vtrn2q_u32(w[3], w[7]))),
        ]
    }
}

/// Rows of 4 bytes, four per vector (rows `4k..4k + 4` in vector `k`), to
/// one vector per byte position holding all 16 rows.
#[inline(always)]
fn transpose_16x4(rows: [B; 4]) -> [B; 4] {
    // SAFETY: register-only. Rows are [x0 x1 x2 x3]: the first unzip
    // splits bytes {0, 2} from {1, 3} of each row, the second splits
    // those pairs, leaving rows in order in every output.
    unsafe {
        let even_low = vuzp1q_u8(rows[0].0, rows[1].0);
        let odd_low = vuzp2q_u8(rows[0].0, rows[1].0);
        let even_high = vuzp1q_u8(rows[2].0, rows[3].0);
        let odd_high = vuzp2q_u8(rows[2].0, rows[3].0);
        [
            B(vuzp1q_u8(even_low, even_high)),
            B(vuzp1q_u8(odd_low, odd_high)),
            B(vuzp2q_u8(even_low, even_high)),
            B(vuzp2q_u8(odd_low, odd_high)),
        ]
    }
}

/// Inverse of [`transpose_16x4`].
#[inline(always)]
fn untranspose_16x4(p: [B; 4]) -> [B; 4] {
    // SAFETY: register-only; the zips undo `transpose_16x4`'s unzips in
    // reverse order.
    unsafe {
        let (even_low, even_high) = (vzip1q_u8(p[0].0, p[2].0), vzip2q_u8(p[0].0, p[2].0));
        let (odd_low, odd_high) = (vzip1q_u8(p[1].0, p[3].0), vzip2q_u8(p[1].0, p[3].0));
        [
            B(vzip1q_u8(even_low, odd_low)),
            B(vzip2q_u8(even_low, odd_low)),
            B(vzip1q_u8(even_high, odd_high)),
            B(vzip2q_u8(even_high, odd_high)),
        ]
    }
}

/// The strong outputs `[x2', x1', x0']` of eight lines (one register half).
#[inline(always)]
fn strong_half([x3, x2, x1, x0, y0, y1]: [uint8x8_t; 6]) -> [uint8x8_t; 3] {
    // SAFETY: register-only.
    unsafe {
        // s = x1 + x0 + y0, s2 = s + x2.
        let s = vaddw_u8(vaddl_u8(x1, x0), y0);
        let s2 = vaddw_u8(s, x2);
        [
            // (2·x3 + 3·x2 + x1 + x0 + y0 + 4) >> 3
            vrshrn_n_u16::<3>(vaddq_u16(vshlq_n_u16::<1>(vaddl_u8(x3, x2)), s2)),
            // (x2 + x1 + x0 + y0 + 2) >> 2
            vrshrn_n_u16::<2>(s2),
            // (x2 + 2·x1 + 2·x0 + 2·y0 + y1 + 4) >> 3
            vrshrn_n_u16::<3>(vaddw_u8(vaddq_u16(s, s2), y1)),
        ]
    }
}

/// `trn1` (`FIRST`) or `trn2` of the 16-bit elements of `a` and `b`, as
/// 32-bit elements.
#[inline(always)]
fn trn_u16<const FIRST: bool>(a: uint8x16_t, b: uint8x16_t) -> uint32x4_t {
    // SAFETY: register-only.
    unsafe {
        let (a, b) = (vreinterpretq_u16_u8(a), vreinterpretq_u16_u8(b));
        vreinterpretq_u32_u16(if FIRST {
            vtrn1q_u16(a, b)
        } else {
            vtrn2q_u16(a, b)
        })
    }
}
