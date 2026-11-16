//! H.266 general intra reconstruction adapters to the shared residual kernels.
use crate::{BlockSize, Dsp};
use nc_common::{BitDepth, PlaneViewMut, Sample};

/// The largest residual magnitude [`Dsp::add_residual`] takes: its 16-bit
/// add of an 8-bit sample must not wrap.
const ADD_RESIDUAL_LIMIT: i16 = i16::MAX - u8::MAX as i16;

/// H.266 joint `CbCr` inverse mixing (§8.7.2; mask 1: Cr coded, 2/3: Cb
/// coded) of a 16-bit inverse-transform output. Negation saturates, so
/// `CoeffMin` = −32768 becomes 32767 rather than the spec's 32768 (and
/// 16383 rather than 16384 after the halving): any magnitude above 255
/// saturates the 8-bit reconstruction alike, and LMCS chroma scaling
/// clips its input to −256..255 first (§8.7.5), so the picture is the
/// spec's.
/// # Panics
/// If the mask is invalid or outputs are short.
pub fn vvc_joint_chroma(coded: &[i16], mask: u8, negative: bool, cb: &mut [i16], cr: &mut [i16]) {
    assert!((1..=3).contains(&mask) && cb.len() >= coded.len() && cr.len() >= coded.len());
    for ((&value, cb), cr) in coded.iter().zip(cb).zip(cr) {
        let signed = if negative {
            value.saturating_neg()
        } else {
            value
        };
        (*cb, *cr) = match mask {
            1 => (signed >> 1, value),
            2 => (value, signed >> 1),
            3 => (value, signed),
            _ => unreachable!(),
        };
    }
}

impl Dsp {
    /// Add an inverse VVC residual to a packed prediction. Rectangles whose
    /// edges are at least four use the existing dispatched residual adder
    /// when every value is within its contract (it is for transformed
    /// blocks); narrow ISP/chroma edges, and the rare transform-skip or
    /// joint-chroma residual at the 16-bit limits, take a scalar fallback.
    /// # Panics
    /// If dimensions are not powers of two in 1..=64 or buffers are short.
    pub fn vvc_add_residual(
        &self,
        (width, height): (usize, usize),
        prediction: &mut [u8],
        residual: &[i16],
    ) {
        let length = check_add_residual((width, height), prediction.len(), residual.len());
        let residual = &residual[..length];
        let dispatched = BlockSize::new(width, height).filter(|_| {
            residual
                .iter()
                .all(|r| (-ADD_RESIDUAL_LIMIT..=ADD_RESIDUAL_LIMIT).contains(r))
        });
        if let Some(size) = dispatched {
            let mut view = PlaneViewMut::new(prediction, width, width, height).unwrap();
            self.add_residual(size, &mut view, residual, width);
        } else {
            add_clipped(&mut prediction[..length], residual, BitDepth::EIGHT);
        }
    }
}

/// Check the inputs of [`Dsp::vvc_add_residual`] and return the block's
/// sample count.
///
/// # Panics
/// If a dimension is not a power of two in 1..=64 or a buffer is short.
pub(crate) fn check_add_residual(
    (width, height): (usize, usize),
    prediction_len: usize,
    residual_len: usize,
) -> usize {
    let supported = |n: usize| n.is_power_of_two() && (1..=64).contains(&n);
    assert!(supported(width) && supported(height));
    let length = width * height;
    assert!(prediction_len >= length && residual_len >= length);
    length
}

/// `recSamples = Clip1(predSamples + resSamples)` (H.266 §8.7.5.1) at
/// `depth`, sample by sample: the scalar reference of the residual add.
pub(crate) fn add_clipped<T: Sample>(prediction: &mut [T], residual: &[i16], depth: BitDepth) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    for (dst, &src) in prediction.iter_mut().zip(residual) {
        *dst = depth.clip_to(dst.to_i32() + i32::from(src));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn joint_negative_odd_rounding() {
        let coded = [-7, -1, 0, 1, 7];
        let (mut cb, mut cr) = ([0; 5], [0; 5]);
        vvc_joint_chroma(&coded, 2, true, &mut cb, &mut cr);
        assert_eq!(cb, coded);
        assert_eq!(cr, [3, 0, 0, -1, -4]);
        vvc_joint_chroma(&coded, 1, false, &mut cb, &mut cr);
        assert_eq!(cb, [-4, -1, 0, 0, 3]);
        assert_eq!(cr, coded);
    }
    /// The 16-bit minimum is a legal inverse output (§8.7.4.1 clips to
    /// CoeffMin..CoeffMax); it once tripped an assertion. Its negation
    /// saturates (beyond any 8-bit reconstruction difference).
    #[test]
    fn joint_mixing_accepts_the_16_bit_minimum() {
        let coded = [-32768, 32767];
        let (mut cb, mut cr) = ([0; 2], [0; 2]);
        vvc_joint_chroma(&coded, 3, true, &mut cb, &mut cr);
        assert_eq!(cb, coded);
        assert_eq!(cr, [32767, -32767]);
        vvc_joint_chroma(&coded, 2, true, &mut cb, &mut cr);
        assert_eq!(cr, [16383, -16384]);
    }
    #[test]
    fn reconstruction_matches_wide_integer_clipping() {
        let dsp = Dsp::new();
        for width in [2, 4, 8, 16, 32] {
            for height in [2, 4, 8, 16, 32] {
                let length = width * height;
                let residual: Vec<_> = (0..length)
                    .map(|i| [-32768, -32513, -32512, -128, 0, 127, 32512, 32513, 32767][i % 9])
                    .collect();
                let mut actual: Vec<_> = (0..length).map(|i| (i * 53) as u8).collect();
                let expected: Vec<_> = actual
                    .iter()
                    .zip(&residual)
                    .map(|(&p, &r)| (i64::from(p) + i64::from(r)).clamp(0, 255) as u8)
                    .collect();
                dsp.vvc_add_residual((width, height), &mut actual, &residual);
                assert_eq!(actual, expected);
            }
        }
    }
}
