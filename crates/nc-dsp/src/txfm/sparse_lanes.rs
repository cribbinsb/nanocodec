//! Structural zeros for the generated inverse butterflies: [`Sparse`] wraps
//! any [`InverseLanes`] type and marks inputs known to be zero, so one full
//! graph from `dct.rs` also yields its partial variants. A transform whose
//! nonzero coefficients lie in a top-left corner (libvpx's eob classes,
//! `vpx_idct16x16_10_add`, `vpx_idct32x32_34_add`, ...; dav1d's `eob`
//! sub-regions) runs the same graph with the other inputs [`Sparse::Zero`]:
//! after inlining, every operation on a known zero folds away, and a
//! butterfly with one zero input becomes a single multiply
//! ([`InverseLanes::mul_round`]), exactly as libvpx's hand-written partial
//! transforms do (libvpx BSD-3-Clause, dav1d BSD-2-Clause; `THIRD_PARTY.md`).
//!
//! The results are bit-identical to the full graph under every lane policy:
//! adding or subtracting zero is exact, `half_btf(w0, a, w1, 0)` is
//! `round_shift(w0 * a)` by definition, and a stage clamp of zero is zero.

#![allow(
    clippy::inline_always,
    reason = "the zero tests must inline into the butterfly graphs to fold away"
)]

use super::inverse_lanes::InverseLanes;

/// Lanes that are either all structurally zero or held in `V`. The
/// discriminant must be known at compile time (constant inputs) for the
/// zero operations to vanish; a runtime mix is still exact, only slower.
#[derive(Clone, Copy)]
pub(crate) enum Sparse<V> {
    Zero,
    Lanes(V),
}

impl<V: InverseLanes> Sparse<V> {
    /// The lanes, with a structural zero materialised.
    #[inline(always)]
    pub(crate) fn get(self) -> V {
        match self {
            Self::Zero => V::splat(0),
            Self::Lanes(v) => v,
        }
    }
}

impl<V: InverseLanes> InverseLanes for Sparse<V> {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        if value == 0 {
            Self::Zero
        } else {
            Self::Lanes(V::splat(value))
        }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        match (self, other) {
            (Self::Zero, x) | (x, Self::Zero) => x,
            (Self::Lanes(a), Self::Lanes(b)) => Self::Lanes(a.add(b)),
        }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        match (self, other) {
            (x, Self::Zero) => x,
            (Self::Zero, Self::Lanes(b)) => Self::Lanes(V::splat(0).sub(b)),
            (Self::Lanes(a), Self::Lanes(b)) => Self::Lanes(a.sub(b)),
        }
    }

    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        match (a, b) {
            (Self::Zero, Self::Zero) => Self::Zero,
            (Self::Lanes(a), Self::Zero) => Self::Lanes(V::mul_round(w0, a, bit)),
            (Self::Zero, Self::Lanes(b)) => Self::Lanes(V::mul_round(w1, b, bit)),
            (Self::Lanes(a), Self::Lanes(b)) => Self::Lanes(V::half_btf(w0, a, w1, b, bit)),
        }
    }

    #[inline(always)]
    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        match a {
            Self::Zero => Self::Zero,
            Self::Lanes(a) => Self::Lanes(V::mul_round(w, a, bit)),
        }
    }

    #[inline(always)]
    fn clamp16(self) -> Self {
        match self {
            Self::Zero => Self::Zero,
            Self::Lanes(v) => Self::Lanes(v.clamp16()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Sparse;
    use crate::txfm::{COSPI_12, dct};

    /// Every corner size of the 8/16/32-point graphs on one exact i32 lane:
    /// the sparse run equals the full graph with explicit zeros.
    #[test]
    fn sparse_inputs_match_the_full_graph() {
        let mut state = 0x2545_f491_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0xffff) as i32 - 32768
        };
        for _ in 0..50 {
            let values: [i32; 32] = core::array::from_fn(|_| next());
            for k in [1, 4, 8, 16, 32] {
                let full: [i32; 32] = core::array::from_fn(|i| if i < k { values[i] } else { 0 });
                let sparse: [Sparse<i32>; 32] = core::array::from_fn(|i| {
                    if i < k {
                        Sparse::Lanes(values[i])
                    } else {
                        Sparse::Zero
                    }
                });
                let mut expected = [0; 32];
                dct::idct32(&full, &mut expected, &COSPI_12, 12);
                let mut actual = [Sparse::Zero; 32];
                dct::idct32(&sparse, &mut actual, &COSPI_12, 12);
                assert_eq!(expected, actual.map(Sparse::get), "idct32 k={k}");
                if k <= 16 {
                    let mut expected = [0; 16];
                    dct::idct16(full.first_chunk().unwrap(), &mut expected, &COSPI_12, 12);
                    let mut actual = [Sparse::Zero; 16];
                    dct::idct16(sparse.first_chunk().unwrap(), &mut actual, &COSPI_12, 12);
                    assert_eq!(expected, actual.map(Sparse::get), "idct16 k={k}");
                }
                if k <= 8 {
                    let mut expected = [0; 8];
                    dct::idct8(full.first_chunk().unwrap(), &mut expected, &COSPI_12, 12);
                    let mut actual = [Sparse::Zero; 8];
                    dct::idct8(sparse.first_chunk().unwrap(), &mut actual, &COSPI_12, 12);
                    assert_eq!(expected, actual.map(Sparse::get), "idct8 k={k}");
                }
            }
        }
    }
}
