//! General H.266 inverse primary transforms in the shared DSP layer.
//! Integer bases follow BSD-Clear `VVdeC` (see `THIRD_PARTY.md`); ordinary
//! DCT-II blocks retain the existing SIMD-backed HEVC transform path.
use super::{DCT32, DST4};
use crate::Dsp;

#[path = "vvc_primary_tables.rs"]
pub(crate) mod tables;

const MAX_FREQUENCIES: usize = 32;

/// H.266 primary transform basis for one dimension.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VvcTransformKind {
    /// Discrete cosine transform, type II.
    #[default]
    Dct2,
    /// Discrete sine transform, type VII.
    Dst7,
    /// Discrete cosine transform, type VIII.
    Dct8,
}
impl VvcTransformKind {
    /// Whether this kind has an `size`-point transform.
    pub(crate) fn valid(self, size: usize) -> bool {
        match self {
            Self::Dct2 => size.is_power_of_two() && size <= 64,
            Self::Dst7 | Self::Dct8 => [4, 8, 16, 32].contains(&size),
        }
    }
    /// The frequencies an `size`-point transform of this kind keeps (the
    /// 32-point DST-VII / DCT-VIII and the 64-point DCT-II zero the rest).
    pub(crate) fn frequencies(self, size: usize) -> usize {
        if self != Self::Dct2 && size == 32 {
            16
        } else {
            size.min(MAX_FREQUENCIES)
        }
    }
    fn basis(self, size: usize, frequency: usize, sample: usize) -> i32 {
        if self == Self::Dct2 {
            return if size == 64 {
                i32::from(tables::DCT64[frequency][sample])
            } else {
                i32::from(DCT32[frequency * 32 / size][sample])
            };
        }
        let (sample, sign) = if self == Self::Dct8 {
            (size - 1 - sample, if frequency & 1 == 0 { 1 } else { -1 })
        } else {
            (sample, 1)
        };
        let value = match size {
            4 => DST4[frequency][sample],
            8 => i32::from(tables::DST7_8[frequency][sample]),
            16 => i32::from(tables::DST7_16[frequency][sample]),
            32 => i32::from(tables::DST7_32[frequency][sample]),
            _ => unreachable!(),
        };
        sign * value
    }
}

/// Basis entry `(frequency, sample)` of an `size`-point transform `kind`.
#[cfg(test)]
pub(crate) fn basis_entry(
    kind: VvcTransformKind,
    size: usize,
    frequency: usize,
    sample: usize,
) -> i32 {
    kind.basis(size, frequency, sample)
}

/// Invert already dequantized coefficients; each stage clips to 16 bits.
/// The one-dimensional ISP case has one rounding stage with shift thirteen.
pub(crate) fn inverse(
    size: (usize, usize),
    kinds: [VvcTransformKind; 2],
    coefficients: &[i32],
    out: &mut [i32],
) {
    let (width, height) = size;
    let rounded = |sum: i32, shift: u32| ((sum + (1 << (shift - 1))) >> shift).clamp(-32768, 32767);
    if width == 1 || height == 1 {
        let (length, kind) = if width == 1 {
            (height, kinds[1])
        } else {
            (width, kinds[0])
        };
        for (x, target) in out[..length].iter_mut().enumerate() {
            let sum = (0..kind.frequencies(length))
                .map(|k| kind.basis(length, k, x) * coefficients[k])
                .sum();
            *target = rounded(sum, 13);
        }
        return;
    }
    let mut intermediate = [0; 4096];
    let [horizontal, vertical] = kinds;
    let horizontal_count = horizontal.frequencies(width);
    let vertical_count = vertical.frequencies(height);
    let mut weights = [0; MAX_FREQUENCIES];
    // Each output coordinate uses the same basis across every orthogonal
    // line. Resolve the transform kind and size once per weight, rather than
    // repeating that lookup for every multiply in every line.
    for y in 0..height {
        for (k, weight) in weights[..vertical_count].iter_mut().enumerate() {
            *weight = vertical.basis(height, k, y);
        }
        for x in 0..horizontal_count {
            let sum = weights[..vertical_count]
                .iter()
                .enumerate()
                .map(|(k, &weight)| weight * coefficients[k * width + x])
                .sum();
            intermediate[y * width + x] = rounded(sum, 7);
        }
    }
    for x in 0..width {
        for (k, weight) in weights[..horizontal_count].iter_mut().enumerate() {
            *weight = horizontal.basis(width, k, x);
        }
        for y in 0..height {
            let sum = weights[..horizontal_count]
                .iter()
                .zip(&intermediate[y * width..y * width + horizontal_count])
                .map(|(&weight, &coefficient)| weight * coefficient)
                .sum();
            out[y * width + x] = rounded(sum, 12);
        }
    }
}

impl Dsp {
    /// Inverse primary transform with flat scaling for an 8-bit VVC block.
    /// `kinds` selects horizontal then vertical bases. DCT-II dimensions are
    /// powers of two from 1 to 64; alternative bases support 4, 8, 16 and 32.
    /// # Panics
    /// If the shape, kinds, QP or buffer sizes are invalid.
    pub fn vvc_inverse_primary(
        &self,
        size: (usize, usize),
        kinds: [VvcTransformKind; 2],
        levels: &[i32],
        qp: u8,
        dependent: bool,
        out: &mut [i32],
    ) {
        let (width, height) = size;
        assert!(kinds[0].valid(width) && kinds[1].valid(height) && width * height >= 4);
        assert!(qp <= 63 && levels.len() >= width * height && out.len() >= width * height);
        if kinds == [VvcTransformKind::Dct2; 2]
            && (2..=32).contains(&width)
            && (2..=32).contains(&height)
        {
            self.vvc_inverse_dct(width, height, levels, qp, dependent, out);
            return;
        }
        let length = width * height;
        let scaling = super::vvc_inverse::scaling(size, qp, dependent);
        crate::vvc_mts::with_scratch(length, |scaled| {
            self.hevc_dequantize(&levels[..length], &scaling, &mut scaled[..length]);
            if (4..=32).contains(&width) && (4..=32).contains(&height) {
                self.vvc_inverse_kinds(size, kinds, &scaled[..length], out);
            } else {
                inverse(size, kinds, &scaled[..length], out);
            }
        });
    }
}

#[cfg(test)]
#[path = "vvc_primary_tests.rs"]
mod tests;
