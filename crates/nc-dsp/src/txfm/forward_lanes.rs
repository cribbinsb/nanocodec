//! Minimal forward-butterfly operations. Storage width may be narrower than
//! 32 bits when the caller bounds every stage (8-bit 64-point columns).

#![allow(clippy::inline_always)]

use crate::simd::I32Lanes;

/// Integer-overflow safety envelope for the guarded AV1 row graph, not an
/// RD control. The outward-rounded interval test in `range_probe` checks
/// every stage.
#[cfg(any(target_arch = "aarch64", test))]
pub(super) const NARROW_ROW_L1_LIMIT: i32 = 32000;

/// Largest 16-point forward column input: an 8-bit residual (`|r| <= 255`)
/// after libaom's input shift of 2.
#[cfg(test)]
pub(super) const FORWARD_16_COLUMN_INPUT_MAX: i128 = 255 << 2;

pub(crate) trait ForwardLanes: Copy {
    fn splat(value: i32) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self;
}

impl<V: I32Lanes> ForwardLanes for V {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        V::splat(value)
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        I32Lanes::add(self, other)
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        I32Lanes::sub(self, other)
    }
    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        V::half_btf(w0, a, w1, b, bit)
    }
}

#[cfg(test)]
mod tests {
    use super::ForwardLanes;
    use crate::txfm::{cospi, dct};

    #[derive(Clone, Copy)]
    struct Checked(i16);

    impl ForwardLanes for Checked {
        fn splat(v: i32) -> Self {
            Self(v.try_into().unwrap())
        }
        fn add(self, rhs: Self) -> Self {
            Self(self.0.checked_add(rhs.0).unwrap())
        }
        fn sub(self, rhs: Self) -> Self {
            Self(self.0.checked_sub(rhs.0).unwrap())
        }
        fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
            // The NEON equal-weight shortcut also requires its pre-add or
            // pre-subtract to fit, not just the rounded butterfly output.
            if w0 == w1 {
                a.0.checked_add(b.0).unwrap();
            } else if w0 == -w1 {
                a.0.checked_sub(b.0).unwrap();
            }
            let value = (i64::from(w0) * i64::from(a.0)
                + i64::from(w1) * i64::from(b.0)
                + (1_i64 << (bit - 1)))
                >> bit;
            Self(value.try_into().unwrap())
        }
    }

    #[test]
    fn low32_forward64_equals_full_retained_coefficients() {
        let mut rng = 0x9182_3311_u32;
        for (bits, amplitude) in [(13, 255), (10, 4095), (12, 1023)] {
            for case in 0..512 {
                let input = core::array::from_fn(|i| match case {
                    0 => amplitude,
                    1 => -amplitude,
                    2..=129 => {
                        if i == (case - 2) / 2 {
                            if case % 2 == 0 { amplitude } else { -amplitude }
                        } else {
                            0
                        }
                    }
                    130..=193 => {
                        if (i * (case - 129)) % 64 < 32 {
                            amplitude
                        } else {
                            -amplitude
                        }
                    }
                    _ => {
                        rng ^= rng << 13;
                        rng ^= rng >> 17;
                        rng ^= rng << 5;
                        (rng % (2 * amplitude + 1) as u32) as i32 - amplitude
                    }
                });
                let mut full = [0_i32; 64];
                let mut pruned = [0x1234_i32; 64];
                dct::fdct64(&input, &mut full, cospi(bits), bits);
                dct::fdct64_low32(&input, &mut pruned, cospi(bits), bits);
                assert_eq!(pruned[..32], full[..32], "bits{bits} case{case}");
            }
        }
    }

    #[test]
    fn lowbd_64_columns_fit_narrow_stages_and_match_wide_graph() {
        let mut rng = 0x42a1_u32;
        for case in 0..512 {
            let input = core::array::from_fn(|i| match case {
                0 => 255,
                1 => -255,
                2..=129 => {
                    if i == (case - 2) / 2 {
                        if case % 2 == 0 { 255 } else { -255 }
                    } else {
                        0
                    }
                }
                130..=193 => {
                    if (i * (case - 129)) % 64 < 32 {
                        255
                    } else {
                        -255
                    }
                }
                _ => {
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    if case % 2 == 0 {
                        (rng % 511) as i32 - 255
                    } else if rng & 1 == 0 {
                        255
                    } else {
                        -255
                    }
                }
            });
            let mut wide = [0_i32; 64];
            let mut narrow = [Checked(0); 64];
            dct::fdct64(&input, &mut wide, cospi(13), 13);
            dct::fdct64(&input.map(Checked::splat), &mut narrow, cospi(13), 13);
            assert_eq!(wide, narrow.map(|v| i32::from(v.0)), "case {case}");
        }
    }
    #[test]
    fn lowbd_16_32_graphs_fit_narrow_stages_and_match_wide() {
        let mut rng = 0x42a1_u32;
        for n in [16usize, 32] {
            for case in 0..512 {
                let input: [i32; 32] = core::array::from_fn(|i| {
                    if i >= n {
                        return 0;
                    }
                    match case {
                        0 => 255,
                        1 => -255,
                        2..=65 => {
                            if i == (case - 2) / 2 {
                                if case % 2 == 0 { 255 } else { -255 }
                            } else {
                                0
                            }
                        }
                        66..=97 => {
                            if (i * (case - 65)) % n < n / 2 {
                                255
                            } else {
                                -255
                            }
                        }
                        _ => {
                            rng ^= rng << 13;
                            rng ^= rng >> 17;
                            rng ^= rng << 5;
                            if case % 2 == 0 {
                                (rng % 511) as i32 - 255
                            } else if rng & 1 == 0 {
                                255
                            } else {
                                -255
                            }
                        }
                    }
                });
                let mut wide = [0_i32; 32];
                let mut narrow = [Checked(0); 32];
                let wide_input = input.map(|v| v * 4);
                let narrow_input = input.map(|v| Checked::splat(v * 4));
                if n == 16 {
                    dct::fdct16(
                        &wide_input[..16].try_into().unwrap(),
                        (&mut wide[..16]).try_into().unwrap(),
                        cospi(12),
                        12,
                    );
                    dct::fdct16(
                        &narrow_input[..16].try_into().unwrap(),
                        (&mut narrow[..16]).try_into().unwrap(),
                        cospi(12),
                        12,
                    );
                } else {
                    dct::fdct32(&wide_input, &mut wide, cospi(12), 12);
                    dct::fdct32(&narrow_input, &mut narrow, cospi(12), 12);
                }
                assert_eq!(wide, narrow.map(|v| i32::from(v.0)), "n={n}, case={case}");
            }
        }
    }
}
