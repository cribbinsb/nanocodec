//! Motion vectors and reference pairs as the motion store keeps them, the
//! quantized vectors of the temporal field, and the projection of vectors
//! by display-order distance (dav2d `levels.h` `mv`, `refpair`;
//! `refmvs.h` `tmv`, `qmv`, `scale_tmv`, `quantize_mv`; `refmvs.c`
//! `scale_mv`, `div_mult`, `dav2d_mv_projection`, `tmv_projection`,
//! `tmv2_projection`, `dequantize_mv`; `env.h` `fix_int_mv_precision`,
//! `mv_reduce_prec`; dav2d commit 237229e, BSD-2-Clause, see
//! `docs/licenses/dav2d`).

/// A motion vector in eighth samples (dav2d `mv`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mv {
    /// Down, in eighth samples.
    pub y: i32,
    /// Right, in eighth samples.
    pub x: i32,
}

/// Vector components stay within this magnitude.
pub const MAX_COMPONENT: i32 = 0xffff;
/// Eighth-sample vector units per sample, as a shift.
pub const FRACTION_BITS: u32 = 3;

impl Mv {
    /// No vector: an intra unit, or one not decoded yet (dav2d
    /// `INVALID_MV`, which no vector in range equals).
    pub const INVALID: Self = Self { y: 0, x: 0x20_0000 };

    /// Rounded to whole samples (dav2d `fix_int_mv_precision`): to the
    /// nearest, halves toward zero.
    #[must_use]
    pub fn to_integer(self) -> Self {
        let round = |v: i32| (v.wrapping_sub(v >> 15).wrapping_add(3)) & !7;
        Self {
            y: round(self.y),
            x: round(self.x),
        }
    }

    /// Rounded to precision `precision` (0..6: eight samples to eighth
    /// samples; dav2d `mv_reduce_prec`): to the nearest, positive halves
    /// down.
    #[must_use]
    pub fn reduce_precision(self, precision: u8) -> Self {
        const EIGHTH: u8 = 6;
        if precision >= EIGHTH {
            return self;
        }
        let round = 32_i32 >> precision;
        let mask = !(round * 2 - 1);
        let reduce = |v: i32| (v + round - i32::from(v > 0)) & mask;
        Self {
            y: reduce(self.y),
            x: reduce(self.x),
        }
    }

    /// Scaled by `num / den` display-order distances (dav2d
    /// `dav2d_mv_projection`): `den` below 32, `num` within +-31.
    #[must_use]
    pub fn project(self, num: i32, den: i32) -> Self {
        scale(self, scale_factor(num, den as u8))
    }

    /// The sum, each component clipped to [`MAX_COMPONENT`].
    #[must_use]
    pub fn clipped_add(self, other: Self) -> Self {
        let add = |a: i32, b: i32| a.wrapping_add(b).clamp(-MAX_COMPONENT, MAX_COMPONENT);
        Self {
            y: add(self.y, other.y),
            x: add(self.x, other.x),
        }
    }
}

/// No reference (intra, or a block copy's "none" pair).
pub const NO_REFERENCE: i8 = -1;
/// The temporal interpolated (TIP) reference (dav2d `TIP_FRAME`).
pub const TIP_REFERENCE: i8 = 7;
/// The references a block may name besides TIP.
pub const REFERENCES: usize = 7;

/// A block's reference pair: ranks 0..6, [`TIP_REFERENCE`], or
/// [`NO_REFERENCE`] (dav2d `refpair`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefPair(pub [i8; 2]);

impl RefPair {
    /// Intra, or a block copy.
    pub const NONE: Self = Self([NO_REFERENCE; 2]);

    /// One reference.
    #[must_use]
    pub fn single(reference: i8) -> Self {
        Self([reference, NO_REFERENCE])
    }

    /// Both entries `reference` (dav2d's `0x101 * ref`).
    #[must_use]
    pub fn both(reference: i8) -> Self {
        Self([reference; 2])
    }

    /// Two references.
    #[must_use]
    pub fn is_compound(self) -> bool {
        self.0[1] != NO_REFERENCE
    }

    /// Reference `n`.
    #[must_use]
    pub fn get(self, n: usize) -> i8 {
        self.0[n]
    }

    /// A rank 0..6 (not TIP, not none) as an index.
    #[must_use]
    pub fn rank(reference: i8) -> Option<usize> {
        (0..TIP_REFERENCE)
            .contains(&reference)
            .then_some(reference as usize)
    }
}

/// A vector of the projected and trajectory fields, in eighth samples
/// (dav2d `tmv`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TemporalMv {
    /// Down.
    pub y: i16,
    /// Right.
    pub x: i16,
}

/// Projected and trajectory vectors stay within this magnitude.
const TEMPORAL_MAX: i32 = 2047;

impl TemporalMv {
    /// No vector (dav2d `INVALID_TMV`).
    pub const INVALID: Self = Self { y: 0, x: 0x2000 };

    /// Whether it holds a vector.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self != Self::INVALID
    }

    /// The vector, or zero when none (dav2d's `if (tmv.n == INVALID_TMV)
    /// tmv.n = 0`).
    #[must_use]
    pub fn or_zero(self) -> Self {
        if self.is_valid() {
            self
        } else {
            Self::default()
        }
    }

    /// A full vector clipped to the field's range.
    #[must_use]
    pub fn clipped(mv: Mv) -> Self {
        Self {
            y: mv.y.clamp(-TEMPORAL_MAX, TEMPORAL_MAX) as i16,
            x: mv.x.clamp(-TEMPORAL_MAX, TEMPORAL_MAX) as i16,
        }
    }

    /// As a full vector.
    #[must_use]
    pub fn widen(self) -> Mv {
        Mv {
            y: i32::from(self.y),
            x: i32::from(self.x),
        }
    }
}

/// A vector of the temporal field kept for later frames, each component
/// in a log-like 8-bit code (dav2d `qmv`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QuantizedMv {
    /// Right.
    pub x: i8,
    /// Down.
    pub y: i8,
}

/// Components up to this magnitude are quantized; larger vectors are not
/// kept.
const QUANTIZABLE: u32 = 2048;
/// Each code octave holds 16 steps.
const OCTAVE_LOG2: u32 = 4;

// The reference's code expansion depends on one byte and has a 512-byte
// working set. Each projected unit otherwise expands both components.
const DEQUANTIZED_COMPONENT: [i16; 256] = {
    let mut values = [0; 256];
    let mut index = 0;
    while index < values.len() {
        let code = index as i8;
        let magnitude = code.unsigned_abs() as i32;
        let bits = (magnitude >> OCTAVE_LOG2) - (magnitude >= 16) as i32;
        let value = ((magnitude - (bits << OCTAVE_LOG2)) << bits) as i16;
        values[index] = if code < 0 { -value } else { value };
        index += 1;
    }
    values
};

impl QuantizedMv {
    /// No vector (dav2d `INVALID_TRAJ`).
    pub const INVALID: Self = Self { x: -128, y: -128 };

    /// Whether it holds a vector.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self != Self::INVALID
    }

    /// Quantize a vector (dav2d `quantize_mv`): none when a component
    /// reaches 2048 eighth samples.
    #[must_use]
    pub fn quantize(mv: Mv) -> Self {
        let (y, x) = (mv.y.unsigned_abs(), mv.x.unsigned_abs());
        if y.max(x) >= QUANTIZABLE {
            return Self::INVALID;
        }
        let component = |magnitude: u32, sign: i32| {
            let code = if magnitude == 0 {
                0
            } else {
                let bits = magnitude.ilog2().saturating_sub(OCTAVE_LOG2);
                (magnitude >> bits) + (bits << OCTAVE_LOG2)
            } as i8;
            if sign < 0 { -code } else { code }
        };
        Self {
            x: component(x, mv.x),
            y: component(y, mv.y),
        }
    }

    /// The vector a code stands for (dav2d `dequantize_mv`).
    #[must_use]
    pub fn dequantize(self) -> TemporalMv {
        if !self.is_valid() {
            return TemporalMv::INVALID;
        }
        TemporalMv {
            y: DEQUANTIZED_COMPONENT[self.y as u8 as usize],
            x: DEQUANTIZED_COMPONENT[self.x as u8 as usize],
        }
    }
}

/// `16384 / d` for distances `d` 1..31 (dav2d `div_mult`).
pub const DIVISION_MULTIPLIERS: [u16; 32] = [
    0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092,
    1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
];
/// Scale factors are Q14.
const SCALE_BITS: u32 = 14;
const SCALE_HALF: i64 = 1 << (SCALE_BITS - 1);

/// The Q14 factor `num / den` (dav2d `num * div_mult[den]`).
#[must_use]
pub fn scale_factor(num: i32, den: u8) -> i32 {
    num * i32::from(DIVISION_MULTIPLIERS[usize::from(den)])
}

/// A Q14 product rounded to the nearest, halves away from zero (dav2d's
/// `(v + 0x2000 - (v < 0)) >> 14`).
fn scale_component(v: i64) -> i64 {
    (v + SCALE_HALF - i64::from(v < 0)) >> SCALE_BITS
}

/// A vector scaled by a Q14 factor and clipped (dav2d `scale_mv`).
#[must_use]
pub fn scale(mv: Mv, factor: i32) -> Mv {
    let s = |v: i32| {
        (scale_component(i64::from(v) * i64::from(factor)) as i32)
            .clamp(-MAX_COMPONENT, MAX_COMPONENT)
    };
    Mv {
        y: s(mv.y),
        x: s(mv.x),
    }
}

/// A field vector scaled by a Q14 factor (dav2d `scale_tmv`: the product
/// in 32 bits).
#[must_use]
pub fn scale_temporal(mv: TemporalMv, factor: i32) -> Mv {
    let s = |v: i16| {
        (scale_component(i64::from(i32::from(v).wrapping_mul(factor))) as i32)
            .clamp(-MAX_COMPONENT, MAX_COMPONENT)
    };
    Mv {
        y: s(mv.y),
        x: s(mv.x),
    }
}

/// A field vector projected likewise (dav2d `tmv_projection`).
#[must_use]
pub fn project_temporal(mv: TemporalMv, num: i32, den: u8) -> Mv {
    scale_temporal(mv, scale_factor(num, den))
}

/// A field vector projected and kept in the field's range (dav2d
/// `tmv2_projection`).
#[must_use]
pub fn project_in_field(mv: TemporalMv, num: i32, den: u8) -> TemporalMv {
    let factor = scale_factor(num, den);
    let s = |v: i16| {
        let p = i32::from(v).wrapping_mul(factor);
        ((p + 8192 + (p >> 31)) >> SCALE_BITS).clamp(-TEMPORAL_MAX, TEMPORAL_MAX) as i16
    };
    TemporalMv {
        y: s(mv.y),
        x: s(mv.x),
    }
}

/// The whole-sample part of a vector component in eighth samples (or of
/// a field component in 1/64 of an 8x8 unit: `shift` 6), rounded toward
/// zero (dav2d `apply_sign(abs(v) >> shift, v)`).
#[must_use]
pub fn whole(v: i32, shift: u32) -> i32 {
    let magnitude = (v.unsigned_abs() >> shift) as i32;
    if v < 0 { -magnitude } else { magnitude }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_rounding() {
        let mv = Mv { y: -12, x: 13 }.to_integer();
        assert_eq!((mv.y, mv.x), (-8, 16));
    }

    #[test]
    fn precision_reduction() {
        let mv = Mv { y: 5, x: -6 }.reduce_precision(4);
        assert_eq!((mv.y, mv.x), (4, -4));
        assert_eq!(Mv { y: 3, x: -3 }.reduce_precision(6), Mv { y: 3, x: -3 });
    }

    /// Quantizing is exact below 32 and loses low bits above, keeping the
    /// sign; the round trip never grows a vector.
    #[test]
    fn quantize_round_trip() {
        for v in -2047..2048 {
            let q = QuantizedMv::quantize(Mv { y: v, x: -v });
            let back = q.dequantize();
            assert!(i32::from(back.y).abs() <= v.abs());
            assert_eq!(i32::from(back.y).signum(), v.signum());
            assert_eq!(back.x, -back.y);
            if v.abs() < 32 {
                assert_eq!(i32::from(back.y), v);
            }
        }
        assert_eq!(
            QuantizedMv::quantize(Mv { y: 2048, x: 0 }),
            QuantizedMv::INVALID
        );
        assert_eq!(QuantizedMv::INVALID.dequantize(), TemporalMv::INVALID);
    }

    #[test]
    fn projections() {
        let mv = Mv { y: 100, x: -33 };
        assert_eq!(mv.project(2, 1), Mv { y: 200, x: -66 });
        assert_eq!(mv.project(-1, 2), Mv { y: -50, x: 17 });
        let t = TemporalMv { y: 100, x: -33 };
        assert_eq!(project_temporal(t, 1, 3), Mv { y: 33, x: -11 });
        assert_eq!(project_in_field(TemporalMv { y: 2000, x: 0 }, 2, 1).y, 2047);
        assert_eq!(whole(-9, 3), -1);
    }
}
