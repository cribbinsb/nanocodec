//! Signed-byte NEON dot-product row helpers for the shared AV1/HEVC/VVC
//! eight-tap MC family (`av1_neon`). Coefficients sum to 64; flipping pixel high bits
//! with 128 is undone by the 8192 accumulator bias. Exact 11/15-byte
//! windows preserve the final-row contract without dav1d's padded reads.
//! The by-element SDOT structure follows dav1d 1.5.4 `mc_dotprod.S`, BSD-2;
//! see `THIRD_PARTY.md` and `docs/licenses/dav1d/COPYING`.

use super::{ConvolveFn, SeparableTaps};
use core::arch::aarch64::{
    int8x8_t, int8x16_t, int16x8_t, int32x4_t, uint8x16_t, vcombine_u8, vdupq_n_s32, vdupq_n_u8,
    veorq_u8, vext_u8, vld1_u8, vld1q_s16, vld1q_u8, vmovn_s16, vqtbl1q_s8, vreinterpretq_s8_u8,
    vreinterpretq_s16_s32, vshrq_n_s16, vuzp1q_s16,
};

/// Unsigned-to-signed pixel compensation: coefficients sum to 64.
pub(super) const DOT_PIXEL_BIAS: i32 = 128 * 64;

/// Contiguous four-pixel dot groups, shared with the H-only schedule.
pub(super) const GATHER: [[u8; 16]; 3] = [
    [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6],
    [4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10],
    [8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14],
];

/// By-element SDOT, retaining dav1d's single packed coefficient register.
/// The intrinsic postdates our Rust 1.90 MSRV, so isolate its register-only
/// instruction here until the minimum compiler supports the intrinsic.
#[inline]
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn dot_lane<const LANE: usize>(
    mut sum: int32x4_t,
    a: int8x16_t,
    b: core::arch::aarch64::int8x8_t,
) -> int32x4_t {
    // SAFETY: dotprod is enabled; the literal lane is zero or one. This
    // register-only instruction has no memory, stack or flag effects.
    unsafe {
        core::arch::asm!(
            "sdot {sum:v}.4s, {a:v}.16b, {b:v}.4b[{lane}]",
            sum = inout(vreg) sum, a = in(vreg) a,
            b = in(vreg_low16) b, lane = const LANE,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    sum
}

/// The kernels of 8-tap table `T` by width class (HEVC's and VVC's
/// eight-tap luma coefficients fit signed bytes and sum to 64).
pub(crate) fn separable_fns<T: SeparableTaps<8>>() -> [ConvolveFn; 5] {
    [
        super::av1_neon::separable::<T, 4>,
        super::av1_neon::separable::<T, 8>,
        super::av1_neon::separable::<T, 16>,
        super::av1_neon::separable::<T, 32>,
        super::av1_neon::separable::<T, 64>,
    ]
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DotFilter {
    coefficients: int8x8_t,
    gather0: uint8x16_t,
    gather4: uint8x16_t,
    gather8: uint8x16_t,
}

impl DotFilter {
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn new(taps: &[i16; 8]) -> Self {
        // SAFETY: the table contains eight coefficients and each mask sixteen bytes.
        unsafe {
            Self {
                coefficients: vmovn_s16(vld1q_s16(taps.as_ptr())),
                gather0: vld1q_u8(GATHER[0].as_ptr()),
                gather4: vld1q_u8(GATHER[1].as_ptr()),
                gather8: vld1q_u8(GATHER[2].as_ptr()),
            }
        }
    }

    /// Only the final allocation row needs exact 11/15-byte loads. Other
    /// rows have a complete stride after them, covering unused vector lanes.
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn load<const W: usize, const PADDED: bool>(row: *const u8) -> uint8x16_t {
        // SAFETY: PADDED promises sixteen readable bytes. Otherwise the
        // caller supplies exactly W+7 bytes (W=4 or 8); overlapping D loads
        // form continuous bytes 0..10/14, with only unused tail lanes rotated.
        unsafe {
            if PADDED {
                vld1q_u8(row)
            } else {
                let tail = vld1_u8(row.add(if W == 4 { 3 } else { 7 }));
                let upper = if W == 4 {
                    vext_u8::<5>(tail, tail)
                } else {
                    vext_u8::<1>(tail, tail)
                };
                vcombine_u8(vld1_u8(row), upper)
            }
        }
    }

    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn sum(self, a: int8x16_t, b: int8x16_t, av1: bool) -> int32x4_t {
        // SAFETY: dotprod is enabled; coefficients sum to 64. Folding the
        // AV1 +2 bias permits one word-level nonrounding shift after UZP.
        unsafe {
            dot_lane::<1>(
                dot_lane::<0>(
                    vdupq_n_s32(DOT_PIXEL_BIAS + if av1 { 2 } else { 0 }),
                    a,
                    self.coefficients,
                ),
                b,
                self.coefficients,
            )
        }
    }

    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn words<const AV1: bool>(a: int32x4_t, b: int32x4_t) -> int16x8_t {
        let words = vuzp1q_s16(vreinterpretq_s16_s32(a), vreinterpretq_s16_s32(b));
        if AV1 { vshrq_n_s16::<2>(words) } else { words }
    }

    /// One four/eight-output horizontal row for the rolling HV window.
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn row<const W: usize, const AV1: bool, const PADDED: bool>(
        self,
        row: *const u8,
    ) -> int16x8_t {
        // SAFETY: the parent checks the complete source window. PADDED is
        // used only before its final row, or with a trimmed six-tap window.
        unsafe {
            let a = vreinterpretq_s8_u8(veorq_u8(Self::load::<W, PADDED>(row), vdupq_n_u8(128)));
            let first = vqtbl1q_s8(a, self.gather0);
            let middle = vqtbl1q_s8(a, self.gather4);
            let lo = self.sum(first, middle, AV1);
            let hi = if W == 4 {
                lo
            } else {
                self.sum(middle, vqtbl1q_s8(a, self.gather8), AV1)
            };
            Self::words::<AV1>(lo, hi)
        }
    }
    /// Sixteen outputs share five shuffles; exact tails use two narrow rows.
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn pair<const AV1: bool, const PADDED: bool>(self, row: *const u8) -> [int16x8_t; 2] {
        // SAFETY: either two exact 15-byte windows in 23 readable bytes, or
        // the caller promises a following stride covering unused Q lanes.
        unsafe {
            if !PADDED {
                return [
                    self.row::<8, AV1, false>(row),
                    self.row::<8, AV1, false>(row.add(8)),
                ];
            }
            let sign = vdupq_n_u8(128);
            let a = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(row), sign));
            let b = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(row.add(12)), sign));
            let first = vqtbl1q_s8(a, self.gather0);
            let middle = vqtbl1q_s8(a, self.gather4);
            let last = vqtbl1q_s8(a, self.gather8);
            let next = vqtbl1q_s8(b, self.gather0);
            let end = vqtbl1q_s8(b, self.gather4);
            [
                Self::words::<AV1>(self.sum(first, middle, AV1), self.sum(middle, last, AV1)),
                Self::words::<AV1>(self.sum(last, next, AV1), self.sum(next, end, AV1)),
            ]
        }
    }

    /// One or two vectors of horizontal intermediates for an HV strip.
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn rows<
        const W: usize,
        const G: usize,
        const AV1: bool,
        const PADDED: bool,
    >(
        self,
        row: *const u8,
    ) -> [int16x8_t; G] {
        assert!(G == 1 || G == 2);
        // SAFETY: the parent validates W+7 samples and the tail policy.
        unsafe {
            if G == 2 {
                let pair = self.pair::<AV1, PADDED>(row);
                let mut result = [pair[0]; G];
                result[1] = pair[1];
                result
            } else {
                [self.row::<W, AV1, PADDED>(row); G]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_phases_match_scalar_with_minimal_unaligned_windows() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let scalar = super::super::scalar::mc_fns();
        for (reference, candidate, phases) in [
            (
                scalar.av1,
                super::super::av1_neon::put_fns::<super::super::Regular>(),
                16,
            ),
            (
                scalar.av1_smooth,
                super::super::av1_neon::put_fns::<super::super::Smooth>(),
                16,
            ),
            (
                scalar.av1_sharp,
                super::super::av1_neon::put_fns::<super::super::Sharp>(),
                16,
            ),
            (
                scalar.hevc_luma,
                separable_fns::<super::super::HevcLumaTaps>(),
                4,
            ),
            (
                scalar.vvc_luma,
                separable_fns::<super::super::VvcLumaTaps>(),
                16,
            ),
            (
                scalar.vvc_affine_luma,
                separable_fns::<super::super::VvcAffineLumaTaps>(),
                16,
            ),
        ] {
            for (index, width) in [4, 8, 16, 32, 64].into_iter().enumerate() {
                for height in [1, 2, 3, 4, 8, 16, 64] {
                    let ss = width + 12;
                    let source_len = (height + 6) * ss + width + 7;
                    let source: Vec<u8> = (0..=source_len)
                        .map(|i| (i * 17 + (i / ss) * 29) as u8)
                        .collect();
                    let ds = width + 11;
                    let length = (height - 1) * ds + width;
                    for fx in 0..phases {
                        for fy in 0..phases {
                            let mut expected = vec![0xa5; length + 2];
                            let mut actual = expected.clone();
                            // SAFETY: feature detection above, exact input/output windows,
                            // supported sizes and every valid phase; guard bytes excluded.
                            unsafe {
                                (reference[index])(
                                    &source[1..],
                                    ss,
                                    &mut expected[1..=length],
                                    ds,
                                    height,
                                    fx,
                                    fy,
                                );
                                (candidate[index])(
                                    &source[1..],
                                    ss,
                                    &mut actual[1..=length],
                                    ds,
                                    height,
                                    fx,
                                    fy,
                                );
                            }
                            assert_eq!(actual, expected, "{width}x{height} phase({fx},{fy})");
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn compound_phases_match_scalar_with_minimal_unaligned_windows() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let scalar = super::super::scalar::mc_fns();
        for (reference, candidate, phases) in [
            (
                scalar.compound.prep[0],
                super::super::av1_neon::prep_fns::<super::super::Regular>(),
                16,
            ),
            (
                scalar.compound.prep[1],
                super::super::av1_neon::prep_fns::<super::super::Smooth>(),
                16,
            ),
            (
                scalar.compound.prep[2],
                super::super::av1_neon::prep_fns::<super::super::Sharp>(),
                16,
            ),
            (
                scalar.compound.hevc_prep[0],
                super::super::av1_neon::separable_prep_fns::<super::super::HevcLumaTaps>(),
                4,
            ),
        ] {
            for (index, width) in [4, 8, 16, 32, 64].into_iter().enumerate() {
                for height in [1, 2, 3, 4, 8, 16, 64] {
                    let ss = width + 12;
                    let source_len = (height + 6) * ss + width + 7;
                    let source: Vec<u8> = (0..=source_len)
                        .map(|i| (i * 17 + (i / ss) * 29) as u8)
                        .collect();
                    let ds = width + 11;
                    let length = (height - 1) * ds + width;
                    for fx in 0..phases {
                        for fy in 0..phases {
                            let mut expected = vec![0x5a5ai16; length + 2];
                            let mut actual = expected.clone();
                            // SAFETY: feature detection above, exact input/output windows,
                            // supported sizes and every valid phase; guard bytes excluded.
                            unsafe {
                                (reference[index])(
                                    &source[1..],
                                    ss,
                                    &mut expected[1..=length],
                                    ds,
                                    height,
                                    fx,
                                    fy,
                                );
                                (candidate[index])(
                                    &source[1..],
                                    ss,
                                    &mut actual[1..=length],
                                    ds,
                                    height,
                                    fx,
                                    fy,
                                );
                            }
                            assert_eq!(actual, expected, "{width}x{height} phase({fx},{fy})");
                        }
                    }
                }
            }
        }
    }
}
