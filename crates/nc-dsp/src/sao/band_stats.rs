//! SAO band-offset statistics (HEVC §8.7.3.2, H.266 §8.8.4.3): per band
//! (`sample >> (BitDepth − 5)`, 32 bands), the sum of `source −
//! reconstruction` and the sample count over a rectangle, from which an
//! encoder chooses the band position and four offsets.
//!
//! One kernel serves every ISA. A band is a data-dependent bin, so the
//! statistics are a histogram: SIMD lanes cannot add into different bins
//! without a gather/scatter that costs more than the scalar loop, and a
//! per-band compare pass costs one vector step per band present. The
//! scalar loop's limit is the store-to-load dependency when neighbouring
//! samples fall in the same band (flat areas, the common case), so it
//! keeps [`LANES`] interleaved histograms and packs each bin's sum and
//! count into one `i64` (one read-modify-write per sample).

/// Number of bands (§8.7.3.2: `bandTable` has 32 entries).
pub const SAO_BANDS: usize = 32;

/// `bandShift` of 8-bit samples (`BitDepth − 5`).
const BAND_SHIFT: u32 = 3;

/// Interleaved histograms: consecutive samples update different arrays, so
/// a run of samples in one band does not serialise on one memory slot.
const LANES: usize = 4;

/// The count's position in a packed bin: `count << COUNT_SHIFT` plus the
/// signed sum. A sum's magnitude stays below `2^31` (the rectangle bound in
/// [`band_stats`]), so the two never mix.
const COUNT_SHIFT: u32 = 32;

/// Per band (0..31), the sum of `source − reconstruction` and the number of
/// samples of the reconstruction in that band.
///
/// `i32` holds both: a sample adds at most 255 in magnitude, and
/// [`band_stats`] accepts at most [`MAX_BAND_SAMPLES`] samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaoBandStats {
    /// Sum of `source − reconstruction` per band.
    pub sum: [i32; SAO_BANDS],
    /// Number of samples per band.
    pub count: [i32; SAO_BANDS],
}

impl Default for SaoBandStats {
    fn default() -> Self {
        Self {
            sum: [0; SAO_BANDS],
            count: [0; SAO_BANDS],
        }
    }
}

/// The most samples one call may cover (after row weighting): their sum
/// stays below `2^31` at `|source − reconstruction| <= 255`, as does the
/// packed sum of [`COUNT_SHIFT`].
pub const MAX_BAND_SAMPLES: usize = (i32::MAX as usize) / 255;

/// Band statistics of the rows `y0, y0 + row_step, ...` below `y1` of the
/// samples `x0..x1`, each sampled row standing for `row_step` rows (the
/// last for the rows left before `y1`), as the sampled edge statistics.
/// `src` and `rec` are whole planes with their strides.
///
/// # Panics
///
/// If the rectangle lies outside either plane or covers more than
/// [`MAX_BAND_SAMPLES`] samples.
#[must_use]
pub(crate) fn band_stats(
    (src, src_stride): (&[u8], usize),
    (rec, rec_stride): (&[u8], usize),
    (x0, y0, x1, y1): (usize, usize, usize, usize),
    row_step: usize,
) -> SaoBandStats {
    assert!(x0 <= x1 && y0 <= y1 && row_step >= 1);
    assert!(
        (x1 - x0) * (y1 - y0 + row_step) <= MAX_BAND_SAMPLES,
        "SAO band statistics rectangle too large"
    );
    let mut bins = [[0i64; SAO_BANDS]; LANES];
    let mut add_row = |y: usize, weight: i64| {
        let source = &src[y * src_stride + x0..y * src_stride + x1];
        let recon = &rec[y * rec_stride + x0..y * rec_stride + x1];
        let unit = weight << COUNT_SHIFT;
        let (source_chunks, _) = source.as_chunks::<LANES>();
        let (recon_chunks, _) = recon.as_chunks::<LANES>();
        for (s, r) in source_chunks.iter().zip(recon_chunks) {
            for ((bins, &s), &r) in bins.iter_mut().zip(s).zip(r) {
                let diff = i64::from(s) - i64::from(r);
                bins[usize::from(r >> BAND_SHIFT)] += unit + weight * diff;
            }
        }
        let tail = source_chunks.len() * LANES;
        for (&s, &r) in source[tail..].iter().zip(&recon[tail..]) {
            let diff = i64::from(s) - i64::from(r);
            bins[0][usize::from(r >> BAND_SHIFT)] += unit + weight * diff;
        }
    };
    for y in (y0..y1).step_by(row_step) {
        add_row(y, (y1 - y).min(row_step) as i64);
    }
    let mut out = SaoBandStats::default();
    for band in 0..SAO_BANDS {
        let packed: i64 = bins.iter().map(|lane| lane[band]).sum();
        // Round the count so a negative sum borrows back from it.
        let count = (packed + (1 << (COUNT_SHIFT - 1))) >> COUNT_SHIFT;
        out.count[band] = count as i32;
        out.sum[band] = (packed - (count << COUNT_SHIFT)) as i32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-sample reference: every row, weighted as the sampled kernel.
    fn reference(
        src: &[u8],
        rec: &[u8],
        stride: usize,
        (x0, y0, x1, y1): (usize, usize, usize, usize),
        step: usize,
    ) -> SaoBandStats {
        let mut out = SaoBandStats::default();
        for y in (y0..y1).step_by(step) {
            let weight = (y1 - y).min(step) as i32;
            for x in x0..x1 {
                let band = usize::from(rec[y * stride + x] >> 3);
                out.sum[band] +=
                    weight * (i32::from(src[y * stride + x]) - i32::from(rec[y * stride + x]));
                out.count[band] += weight;
            }
        }
        out
    }

    #[test]
    fn matches_the_per_sample_reference() {
        let (w, h) = (71, 37);
        let src: Vec<u8> = (0..w * h).map(|i| ((i * 73 + 19) % 256) as u8).collect();
        // Runs of one band (flat areas) and every band, with large errors.
        let rec: Vec<u8> = (0..w * h)
            .map(|i| {
                if i % 97 < 40 {
                    3
                } else {
                    ((i * 41 + i / 13) % 256) as u8
                }
            })
            .collect();
        for step in 1..=4 {
            for rect in [(0, 0, w, h), (1, 2, 66, 35), (5, 5, 6, 9), (3, 3, 3, 3)] {
                assert_eq!(
                    band_stats((&src, w), (&rec, w), rect, step),
                    reference(&src, &rec, w, rect, step),
                    "step {step} rect {rect:?}"
                );
            }
        }
    }

    #[test]
    fn negative_sums_unpack_exactly() {
        let src = [0u8; 64];
        let rec = [255u8; 64];
        let stats = band_stats((&src, 8), (&rec, 8), (0, 0, 8, 8), 1);
        assert_eq!((stats.sum[31], stats.count[31]), (-255 * 64, 64));
    }
}
