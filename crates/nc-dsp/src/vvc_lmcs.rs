//! H.266 luma mapping and chroma-scaling tables (LMCS) at any bit depth.
//! Piecewise-linear arithmetic follows BSD-Clear `VVdeC` Reshape; see `THIRD_PARTY.md`.
//!
//! One model, [`VvcLmcs`], generic over the sample type: `VvcLmcs<u8>` (the
//! default) is the 8-bit model of the encoder and 8-bit decoding, the
//! `u16` instance the deep decoder's, its tables `1 << BitDepth` entries
//! long. Depth enters as `OrgCW = (1 << BitDepth) / 16` (H.266 §7.4.3.19),
//! the bin-width and pivot bounds derived from it, and the chroma
//! residual clip (§8.7.5.3).
use crate::Dsp;
use crate::simd::HIGH_LOOKUP_TABLE;
use nc_common::{BitDepth, PlaneViewMut, Sample};

/// Pieces of the mapping (H.266 §7.4.3.19: 16 equal input bins).
const BINS: usize = 16;
/// Fraction bits of `ScaleCoeff`, `InvScaleCoeff` and `ChromaScaleCoeff`
/// (H.266 §7.4.3.19).
const PRECISION: u32 = 11;
/// `lmcsCW` lies in `OrgCW >> 3 ..= (OrgCW << 3) − 1` (H.266 §7.4.3.19).
const WIDTH_RANGE_SHIFT: u32 = 3;
/// Pivots inside a bin are aligned to `1 << (BitDepth − 5)` or the bin
/// spans an alignment boundary (H.266 §7.4.3.19).
const PIVOT_ALIGNMENT_BITS: u32 = 5;

/// Validated LMCS mapping at one depth, independent of codec header storage.
#[derive(Debug)]
pub struct VvcLmcs<T: Sample = u8> {
    forward: Box<[T]>,
    inverse: Box<[T]>,
    forward_wide: Option<Box<[i16; HIGH_LOOKUP_TABLE]>>,
    inverse_wide: Option<Box<[i16; HIGH_LOOKUP_TABLE]>>,
    chroma: Box<[i32]>,
    depth: BitDepth,
}

impl VvcLmcs {
    /// Derive an 8-bit model from the active bin interval and signed
    /// bin-width deltas. Returns `None` for a model outside the normative
    /// 8-bit bounds.
    #[must_use]
    pub fn new(bins: [u8; 2], deltas: [i16; BINS], chroma_offset: i16) -> Option<Self> {
        Self::new_at(bins, deltas, chroma_offset, BitDepth::EIGHT)
    }
}

/// The piecewise-linear model of H.266 §7.4.3.19 at one depth.
struct Pieces {
    first: usize,
    last: usize,
    /// `OrgCW`: the input width of every bin.
    bin_width: i32,
    /// `Log2(OrgCW)`.
    bin_bits: u32,
    /// `lmcsCW`.
    widths: [i32; BINS],
    /// `LmcsPivot`.
    pivots: [i32; BINS + 1],
}

impl Pieces {
    /// The model of `bins`, `deltas` and `chroma_offset`, or `None` when it
    /// breaks a bound of H.266 §7.4.3.19.
    fn new(
        bins: [u8; 2],
        deltas: [i16; BINS],
        chroma_offset: i16,
        depth: BitDepth,
    ) -> Option<Self> {
        let [first, last] = bins.map(usize::from);
        if first > last || last >= BINS {
            return None;
        }
        let bin_bits = depth.bits() - BINS.ilog2();
        let bin_width = 1 << bin_bits;
        let allowed = (bin_width >> WIDTH_RANGE_SHIFT)..(bin_width << WIDTH_RANGE_SHIFT);
        let alignment = 1 << (depth.bits() - PIVOT_ALIGNMENT_BITS);
        let mut widths = [0; BINS];
        let mut pivots = [0; BINS + 1];
        for i in 0..BINS {
            if (first..=last).contains(&i) {
                widths[i] = bin_width + i32::from(deltas[i]);
                if !allowed.contains(&widths[i])
                    || !allowed.contains(&(widths[i] + i32::from(chroma_offset)))
                {
                    return None;
                }
            }
            pivots[i + 1] = pivots[i] + widths[i];
            if (first..=last).contains(&i)
                && pivots[i] % alignment != 0
                && pivots[i] / alignment == pivots[i + 1] / alignment
            {
                return None;
            }
        }
        if pivots[BINS] > depth.max() {
            return None;
        }
        Some(Self {
            first,
            last,
            bin_width,
            bin_bits,
            widths,
            pivots,
        })
    }

    /// `FwdMap` of a prediction sample (H.266 §8.7.5.2), with `ScaleCoeff
    /// = (lmcsCW · 2^11 + (1 << (Log2(OrgCW) − 1))) >> Log2(OrgCW)`
    /// (§7.4.3.19).
    fn forward(&self, sample: i32) -> i32 {
        let bin = (sample >> self.bin_bits) as usize;
        let scale = ((self.widths[bin] << PRECISION) + (1 << (self.bin_bits - 1))) >> self.bin_bits;
        self.pivots[bin]
            + ((scale * (sample & (self.bin_width - 1)) + (1 << (PRECISION - 1))) >> PRECISION)
    }

    /// The bin of a mapped sample (`idxYInv`, H.266 §8.8.2.2 / §8.7.5.3).
    fn mapped_bin(&self, sample: i32) -> usize {
        (self.first..=self.last)
            .find(|&i| sample < self.pivots[i + 1])
            .unwrap_or((self.last + 1).min(BINS - 1))
    }

    /// `InvMap` of a mapped sample in `bin` (H.266 §8.8.2.2,
    /// `InvScaleCoeff = OrgCW · 2^11 / lmcsCW`).
    fn inverse(&self, sample: i32, bin: usize) -> i32 {
        let width = self.widths[bin];
        let scale = if width == 0 {
            0
        } else {
            (self.bin_width << PRECISION) / width
        };
        self.bin_width * bin as i32
            + ((scale * (sample - self.pivots[bin]) + (1 << (PRECISION - 1))) >> PRECISION)
    }

    /// `ChromaScaleCoeff` of `bin` (H.266 §7.4.3.19).
    fn chroma_scale(&self, bin: usize, chroma_offset: i16) -> i32 {
        let width = self.widths[bin];
        if width == 0 {
            1 << PRECISION
        } else {
            (self.bin_width << PRECISION) / (width + i32::from(chroma_offset))
        }
    }
}

impl<T: Sample> VvcLmcs<T> {
    /// Derive a model at `depth` from the active bin interval and signed
    /// bin-width deltas (`lmcs_delta_cw`, whose range grows with the depth).
    /// Returns `None` for a model outside the normative bounds at that depth.
    ///
    /// # Panics
    /// If `T` cannot store `depth`.
    #[must_use]
    pub fn new_at(
        bins: [u8; 2],
        deltas: [i16; BINS],
        chroma_offset: i16,
        depth: BitDepth,
    ) -> Option<Self> {
        assert!(
            depth.fits::<T>(),
            "{depth:?} LMCS in {}-bit samples",
            T::MAX_BIT_DEPTH
        );
        let pieces = Pieces::new(bins, deltas, chroma_offset, depth)?;
        let samples = 0..=depth.max();
        let bins: Vec<usize> = samples.clone().map(|v| pieces.mapped_bin(v)).collect();
        let forward: Box<[T]> = samples
            .clone()
            .map(|v| depth.clip_to(pieces.forward(v)))
            .collect();
        let inverse: Box<[T]> = samples
            .zip(&bins)
            .map(|(v, &bin)| depth.clip_to(pieces.inverse(v, bin)))
            .collect();
        let wide_table = |src: &[T]| -> Option<Box<[i16; HIGH_LOOKUP_TABLE]>> {
            if T::MAX_BIT_DEPTH <= u8::MAX_BIT_DEPTH || src.len() > HIGH_LOOKUP_TABLE - 1 {
                return None;
            }
            let mut tbl = Box::new([0i16; HIGH_LOOKUP_TABLE]);
            let mask = src.len() - 1;
            for (i, slot) in tbl.iter_mut().enumerate() {
                *slot = src[i.min(mask)].to_i32() as i16;
            }
            Some(tbl)
        };
        Some(Self {
            forward_wide: wide_table(&forward),
            inverse_wide: wide_table(&inverse),
            forward,
            inverse,
            chroma: bins
                .iter()
                .map(|&bin| pieces.chroma_scale(bin, chroma_offset))
                .collect(),
            depth,
        })
    }

    /// Eleven-fraction-bit chroma residual multiplier at the given (mapped)
    /// luma level.
    #[must_use]
    pub fn chroma_scale(&self, luma: T) -> i32 {
        self.chroma[luma.to_i32() as usize]
    }

    /// The depth the model maps.
    #[must_use]
    pub fn depth(&self) -> BitDepth {
        self.depth
    }

    /// Entries of each table: `1 << BitDepth`, the constant 256 in the
    /// `u8` instance (whose depth is always 8).
    fn table_len(&self) -> usize {
        if T::MAX_BIT_DEPTH == u8::MAX_BIT_DEPTH {
            1 << u8::MAX_BIT_DEPTH
        } else {
            1 << self.depth.bits()
        }
    }

    /// Replace every sample by its entry in `table`, one of the mappings.
    /// `table_len` is `1 << BitDepth` (a power of two), so masking by
    /// `table.len() - 1` keeps every lookup statically in bounds.
    pub(crate) fn map_slice(table: &[T], len: usize, samples: &mut [T]) {
        let table = &table[..len];
        let mask = len - 1;
        for value in samples {
            *value = table[(value.to_i32() as usize) & mask];
        }
    }

    pub(crate) fn forward_wide_table(&self) -> Option<&[i16; HIGH_LOOKUP_TABLE]> {
        self.forward_wide.as_deref()
    }

    pub(crate) fn inverse_wide_table(&self) -> Option<&[i16; HIGH_LOOKUP_TABLE]> {
        self.inverse_wide.as_deref()
    }

    /// Map an inter prediction into the LMCS reconstruction domain
    /// (`FwdMap`, H.266 §8.7.5.2).
    pub(crate) fn forward_samples(&self, samples: &mut [T]) {
        Self::map_slice(&self.forward, self.table_len(), samples);
    }

    /// Return reconstructed luma to the original domain (`InvMap`, H.266
    /// §8.8.2.2).
    pub(crate) fn inverse_plane(&self, plane: &mut PlaneViewMut<'_, T>) {
        let len = self.table_len();
        for y in 0..plane.height() {
            Self::map_slice(&self.inverse, len, plane.row_mut(y));
        }
    }
}

/// Scale chroma residuals with the LMCS eleven-fraction-bit multiplier at
/// `depth`: the input is clipped to `−(1 << BitDepth) ..= (1 << BitDepth)
/// − 1` before the symmetric rounding (H.266 §8.7.5.3).
///
/// # Panics
/// If `scale` is outside `0..=16384` (`ChromaScaleCoeff` at most `OrgCW ·
/// 2^11 / (OrgCW >> 3)`).
pub(crate) fn scale_chroma_residual(scale: i32, residual: &mut [i16], depth: BitDepth) {
    assert!((0..=MAX_CHROMA_SCALE).contains(&scale));
    let limit = 1 << depth.bits();
    for value in residual {
        let input = i32::from(*value).clamp(-limit, limit - 1);
        // At most 2^BitDepth · 2^14 >> 11 = 2^(BitDepth + 3) in magnitude:
        // fits 16 bits up to 12-bit samples.
        *value =
            (input.signum() * ((input.abs() * scale + (1 << (PRECISION - 1))) >> PRECISION)) as i16;
    }
}

/// The largest `ChromaScaleCoeff`: `OrgCW · 2^11 / (OrgCW >> 3)`.
const MAX_CHROMA_SCALE: i32 = 1 << (PRECISION + WIDTH_RANGE_SHIFT);

impl Dsp {
    /// Scale 8-bit chroma residuals with the LMCS eleven-fraction-bit multiplier.
    /// Input clipping precedes symmetric rounding, as required by H.266 §8.7.5.
    pub fn vvc_scale_chroma_residual(&self, scale: i32, residual: &mut [i16]) {
        scale_chroma_residual(scale, residual, BitDepth::EIGHT);
    }

    /// Map an inter prediction into the LMCS reconstruction domain.
    pub fn vvc_forward_lmcs(&self, model: &VvcLmcs, samples: &mut [u8]) {
        model.forward_samples(samples);
    }
    /// Return reconstructed luma to the original domain before in-loop filtering.
    pub fn vvc_inverse_lmcs(&self, model: &VvcLmcs, plane: &mut PlaneViewMut<'_>) {
        model.inverse_plane(plane);
    }
}

pub(crate) fn map_u16_scalar(table: &[i16; HIGH_LOOKUP_TABLE], samples: &mut [u16]) {
    for s in samples {
        *s = table[usize::from(*s).min(HIGH_LOOKUP_TABLE - 2)] as u16;
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code, clippy::inline_always)]
pub(crate) mod high_simd {
    use crate::simd::HIGH_LOOKUP_TABLE;

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn map_lanes<W: crate::simd::I16Lanes, H: crate::simd::I16Lanes>(
        table: &[i16; HIGH_LOOKUP_TABLE],
        samples: &mut [u16],
    ) {
        let len = samples.len();
        let mut x = 0;
        while x + W::LANES <= len {
            W::load_u16(&samples[x..])
                .lookup_u12(table)
                .store_u16(&mut samples[x..]);
            x += W::LANES;
        }
        while x + H::LANES <= len {
            H::load_u16(&samples[x..])
                .lookup_u12(table)
                .store_u16(&mut samples[x..]);
            x += H::LANES;
        }
        while x < len {
            samples[x] = table[usize::from(samples[x]).min(HIGH_LOOKUP_TABLE - 2)] as u16;
            x += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn map_avx2(table: &[i16; HIGH_LOOKUP_TABLE], samples: &mut [u16]) {
        map_lanes::<crate::simd::avx2::I16x16, crate::simd::avx2::I16x4>(table, samples);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub(crate) unsafe fn map_avx512(table: &[i16; HIGH_LOOKUP_TABLE], samples: &mut [u16]) {
        map_lanes::<crate::simd::avx512_i16::I16x32, crate::simd::avx2::I16x4>(table, samples);
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn map_neon(table: &[i16; HIGH_LOOKUP_TABLE], samples: &mut [u16]) {
        super::map_u16_scalar(table, samples);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn map_simd128(table: &[i16; HIGH_LOOKUP_TABLE], samples: &mut [u16]) {
        super::map_u16_scalar(table, samples);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chroma_scaling_clips_before_symmetric_rounding() {
        let mut residual = [-32768, -257, -256, -3, -1, 0, 1, 3, 255, 256, 32767];
        Dsp::new().vvc_scale_chroma_residual(3072, &mut residual);
        assert_eq!(residual, [-384, -384, -384, -5, -2, 0, 2, 5, 383, 383, 383]);
    }
    #[test]
    fn shifted_identity_preserves_rounding_and_clamps_inactive_bins() {
        let model = VvcLmcs::new([1, 15], [0; 16], 0).unwrap();
        let dsp = Dsp::new();
        let mut samples: Vec<u8> = (0..=255).collect();
        dsp.vvc_forward_lmcs(&model, &mut samples);
        assert_eq!(
            samples,
            (0..=255u8)
                .map(|v| v.saturating_sub(16))
                .collect::<Vec<_>>()
        );
        dsp.vvc_inverse_lmcs(
            &model,
            &mut PlaneViewMut::new(&mut samples, 256, 256, 1).unwrap(),
        );
        assert_eq!(samples, (0..=255u8).map(|v| v.max(16)).collect::<Vec<_>>());
        assert!((0..=255).all(|v| model.chroma_scale(v) == 2048));
    }
    #[test]
    fn model_rejects_invalid_bin_widths_and_total_range() {
        assert!(VvcLmcs::new([0, 15], [0; 16], 0).is_none());
        assert!(VvcLmcs::new([15, 1], [0; 16], 0).is_none());
        assert!(VvcLmcs::new([1, 15], [-15; 16], 0).is_none());
        assert!(VvcLmcs::new([1, 15], [0; 16], -15).is_none());
    }
}
