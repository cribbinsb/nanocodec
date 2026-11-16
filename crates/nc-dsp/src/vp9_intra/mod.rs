//! VP9 intra prediction of one transform block from its gathered edges
//! (VP9 §8.5.1; libvpx `vpx_dsp/intrapred.c`, BSD-3-Clause, see
//! `THIRD_PARTY.md`): all ten modes, 4x4 to 32x32.
//!
//! The codec gathers the edges (availability, VP9's fixed 127/129 values,
//! the above-right rule) into [`Vp9IntraEdges`]; [`reference`] is the
//! scalar oracle, a direct transcription of libvpx's per-sample loops.
//! [`rows`] computes the same predictions as every directional mode really
//! is: a short smoothed border (2- and 3-tap averages of the edges), of
//! which every output row is a contiguous slice at a fixed offset. Its
//! loops are over byte arrays, so each ISA's instantiation vectorises them.

mod reference;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod rows;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128;

use nc_common::BitDepth;

use crate::Isa;

pub(crate) use reference::predict_at;

/// Largest transform side.
pub(crate) const MAX_SIDE: usize = 32;

/// VP9 intra modes (`intra_mode`, VP9 §9.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vp9IntraMode {
    /// Mean of the available edges.
    Dc,
    /// Vertical: the row above.
    V,
    /// Horizontal: the left column.
    H,
    /// 45° down-left.
    D45,
    /// 135° down-right.
    D135,
    /// 117° steep down-right.
    D117,
    /// 153° shallow down-right.
    D153,
    /// 207° up-right from the left column.
    D207,
    /// 63° steep down-left.
    D63,
    /// `TrueMotion`: `left + above − above_left`, clipped.
    Tm,
}

impl Vp9IntraMode {
    /// Every mode, in `intra_mode` order.
    pub const ALL: [Vp9IntraMode; 10] = [
        Self::Dc,
        Self::V,
        Self::H,
        Self::D45,
        Self::D135,
        Self::D117,
        Self::D153,
        Self::D207,
        Self::D63,
        Self::Tm,
    ];
}

/// The edges of one transform block: `above[0]` is the above-left sample,
/// `above[1..=2 · side]` the row above extended to twice the side,
/// `left[..side]` the left column, with unavailable samples already set to
/// VP9's fixed values (127 and 129 at 8 bits, `mid − 1` and `mid + 1` at
/// any depth). `has_above`/`has_left` select the DC average.
///
/// Contract: for sides above 4 the above-right samples repeat the last
/// above one (`above[side + 1..=2 · side] == above[side]`), as VP9's
/// `have_right` rule makes them; [`rows`] relies on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp9IntraEdges<T = u8> {
    /// Above-left, then the row above (twice the side).
    pub above: [T; 2 * MAX_SIDE + 1],
    /// The left column.
    pub left: [T; MAX_SIDE],
    /// The row above is available.
    pub has_above: bool,
    /// The left column is available.
    pub has_left: bool,
}

/// Predict a `side`x`side` block (side 4, 8, 16 or 32) into `out` (row
/// stride `side`).
pub(crate) type PredictFn = unsafe fn(Vp9IntraMode, usize, &Vp9IntraEdges, &mut [u8]);

/// The kernel of `isa`: the scalar reference, or the border-and-rows form
/// compiled for the ISA.
pub(crate) fn kernel(isa: Isa) -> PredictFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::predict,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::predict,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::predict,
        _ => reference::predict,
    }
}

/// The rounded 2-tap average.
#[inline]
fn avg2(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

/// The rounded [1, 2, 1] average.
#[inline]
fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8
}

/// The DC value: the rounded mean of the available edges, `mid` (128 at 8
/// bits) if none (`dc_pred[left][top]`); shared by both forms.
fn dc_value<T: nc_common::Sample>(side: usize, edges: &Vp9IntraEdges<T>, depth: BitDepth) -> T {
    let sum = |edge: &[T]| edge.iter().map(|&s| s.to_i32() as u32).sum::<u32>();
    let (above, left) = (&edges.above[1..=side], &edges.left[..side]);
    let (total, count) = match (edges.has_above, edges.has_left) {
        (true, true) => (sum(above) + sum(left), 2 * side as u32),
        (true, false) => (sum(above), side as u32),
        (false, true) => (sum(left), side as u32),
        (false, false) => return T::from_i32(depth.mid()),
    };
    T::from_i32(((total + count / 2) / count) as i32)
}
