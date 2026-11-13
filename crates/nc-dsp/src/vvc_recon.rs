//! H.266 general intra reconstruction adapters to the shared residual kernels.
use crate::{BlockSize, Dsp};
use nc_common::PlaneViewMut;

/// H.266 joint `CbCr` inverse mixing (mask 1: Cr coded, 2/3: Cb coded).
/// # Panics
/// If the mask is invalid, outputs are short, or an input exceeds the normative
/// inverse-transform range (signed 16-bit, excluding its asymmetric minimum).
pub fn vvc_joint_chroma(coded: &[i32], mask: u8, negative: bool, cb: &mut [i32], cr: &mut [i32]) {
    assert!((1..=3).contains(&mask) && cb.len() >= coded.len() && cr.len() >= coded.len());
    let sign = if negative { -1 } else { 1 };
    for ((&value, cb), cr) in coded.iter().zip(cb).zip(cr) {
        assert!((-32767..=32767).contains(&value));
        (*cb, *cr) = match mask {
            1 => ((sign * value) >> 1, value),
            2 => (value, (sign * value) >> 1),
            3 => (value, sign * value),
            _ => unreachable!(),
        };
    }
}

impl Dsp {
    /// Add an inverse VVC residual to a packed prediction. Rectangles whose
    /// edges are at least four use the existing dispatched residual adder;
    /// narrow ISP/chroma edges have a small scalar fallback.
    /// # Panics
    /// If dimensions are not powers of two in 1..=64 or buffers are short.
    pub fn vvc_add_residual(
        &self,
        (width, height): (usize, usize),
        prediction: &mut [u8],
        residual: &[i32],
    ) {
        let supported = |n: usize| n.is_power_of_two() && (1..=64).contains(&n);
        assert!(supported(width) && supported(height));
        let length = width * height;
        assert!(prediction.len() >= length && residual.len() >= length);
        if let Some(size) = BlockSize::new(width, height) {
            let mut scratch = [0i16; 64 * 64];
            // Values beyond this bound already saturate any byte prediction;
            // clamping also satisfies the SIMD adder's signed-add contract.
            for (dst, &src) in scratch[..length].iter_mut().zip(residual) {
                *dst = src.clamp(-32512, 32512) as i16;
            }
            let mut view = PlaneViewMut::new(prediction, width, width, height).unwrap();
            self.add_residual(size, &mut view, &scratch[..length], width);
        } else {
            for (dst, &src) in prediction[..length].iter_mut().zip(residual) {
                *dst = (i64::from(*dst) + i64::from(src)).clamp(0, 255) as u8;
            }
        }
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
    #[test]
    fn reconstruction_matches_wide_integer_clipping() {
        let dsp = Dsp::new();
        for width in [2, 4, 8, 16, 32] {
            for height in [2, 4, 8, 16, 32] {
                let length = width * height;
                let residual: Vec<_> = (0..length)
                    .map(|i| [i32::MIN, -32768, -128, 0, 127, 32767, i32::MAX][i % 7])
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
