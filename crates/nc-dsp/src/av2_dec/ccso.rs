//! AV2 cross-component sample offset on a padded luma span.
//!
//! The AVX2 byte classifier and packed offset lookup follow dav2d 237229e's
//! `ccso_prep_*_8bpc_avx2` and `ccso_add_8bpc_avx2`. The caller supplies the
//! padded source and separates lossless and disabled spans.

use crate::Dsp;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

/// Edge classes and threshold for an 8-bit luma sample.
#[derive(Debug, Clone, Copy)]
pub enum CcsoMode {
    /// Classify by band alone.
    Bands,
    /// Below the negative threshold or at/above it.
    Two(u8),
    /// Below, between or above the negative and positive thresholds.
    Three(u8),
}

/// One plane's CCSO lookup, reused across all active spans of that plane.
#[derive(Debug)]
pub struct CcsoFilter {
    mode: CcsoMode,
    band_shift: u8,
    codes: [u8; 64],
    offsets: [i8; 16],
}

impl CcsoFilter {
    /// Build the lookup. Each byte of `codes` packs the offset codes at
    /// consecutive indices, with the even code in the low nibble. Band-only
    /// indices are band numbers and edge indices are `band << 4 | a << 2 | b`.
    ///
    /// # Panics
    /// If the band shift would produce indices outside this mode's lookup.
    #[must_use]
    pub fn new(mode: CcsoMode, band_shift: u8, codes: [u8; 64], offsets: [i8; 16]) -> Self {
        let first_shift = match mode {
            CcsoMode::Bands => 2,
            CcsoMode::Two(_) | CcsoMode::Three(_) => 5,
        };
        assert!((first_shift..=8).contains(&band_shift));
        Self {
            mode,
            band_shift,
            codes,
            offsets,
        }
    }

    /// Apply the filter to a span with the selected SIMD tier. The source
    /// spans start at the luma centre and its two neighbours. Chroma rows
    /// select every second source byte.
    ///
    /// # Panics
    /// If a source span is shorter than the destination's luma footprint.
    #[allow(unsafe_code)]
    pub fn apply(&self, dsp: &Dsp, dst: &mut [u8], source: [&[u8]; 3], chroma: bool) {
        let step = 1 + usize::from(chroma);
        let source_len = dst.len() * step;
        let source = source.map(|row| &row[..source_len]);
        #[cfg(target_arch = "x86_64")]
        let processed =
            if dst.len() >= 32 && matches!(dsp.isa(), crate::Isa::Avx2 | crate::Isa::Avx512) {
                // SAFETY: both tiers support AVX2 and each source span covers
                // every destination lane, including its chroma spacing.
                unsafe { avx2::apply(self, dst, source, chroma) }
            } else {
                0
            };
        #[cfg(target_arch = "aarch64")]
        let processed = if dst.len() >= 16 && dsp.isa() != crate::Isa::Scalar {
            // SAFETY: the Arm SIMD tiers include NEON and the source spans
            // cover every destination lane, including its chroma spacing.
            unsafe { neon::apply(self, dst, source, chroma) }
        } else {
            0
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let processed = {
            let _ = dsp;
            0
        };

        let [center, first, second] = source.map(|row| &row[processed * step..]);
        let center = center.iter().step_by(step);
        let first = first.iter().step_by(step);
        let second = second.iter().step_by(step);
        for (sample, ((&center, &first), &second)) in dst[processed..]
            .iter_mut()
            .zip(center.zip(first).zip(second))
        {
            let band = (usize::from(center)) >> self.band_shift;
            let difference = |edge| i16::from(edge) - i16::from(center);
            let index = match self.mode {
                CcsoMode::Bands => band,
                CcsoMode::Two(threshold) => {
                    let threshold = -i16::from(threshold);
                    let first = usize::from(difference(first) >= threshold);
                    let second = usize::from(difference(second) >= threshold);
                    band << 4 | first << 2 | second
                }
                CcsoMode::Three(threshold) => {
                    let threshold = i16::from(threshold);
                    let class = |edge| {
                        let diff = difference(edge);
                        usize::from(diff >= -threshold) + usize::from(diff > threshold)
                    };
                    band << 4 | class(first) << 2 | class(second)
                }
            };
            let code = self.codes[index / 2] >> (4 * (index & 1)) & 15;
            *sample = sample.saturating_add_signed(self.offsets[usize::from(code)]);
        }
    }
}
