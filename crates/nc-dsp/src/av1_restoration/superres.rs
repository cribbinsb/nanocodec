//! Super-resolution upscaling (AV1 §7.16): each row of a frame coded at a
//! reduced width is resampled to the upscaled width with an 8-tap filter
//! at 1/64-sample phases, stepping a fixed 1/16384-sample distance per
//! output sample.
//!
//! libaom upscales tile column by tile column, but carries the position
//! across columns and reads real neighbouring samples inside the frame, so
//! the result equals this whole-row form.
//!
//! `Upscale_Filter`: libaom v3.15.3 `av1/common/resize.c`
//! (`av1_resize_filter_normative`, BSD-2-Clause, see `THIRD_PARTY.md`).

use nc_common::{PlaneView, PlaneViewMut};

/// `SUPERRES_SCALE_BITS`: positions are in 1/16384 samples.
const SCALE_BITS: u32 = 14;
/// `SUPERRES_FILTER_BITS`: the filter has 64 phases.
const FILTER_PHASE_BITS: u32 = 6;
/// `SUPERRES_EXTRA_BITS`: position bits below the filter phase.
const EXTRA_BITS: u32 = SCALE_BITS - FILTER_PHASE_BITS;
/// Mask of a position's fraction of a sample.
const SCALE_MASK: i64 = (1 << SCALE_BITS) - 1;
/// Filter taps.
const UPSCALE_TAPS: usize = 8;
/// Filter phases.
const UPSCALE_POSITIONS: usize = 1 << FILTER_PHASE_BITS;
/// `SUPERRES_FILTER_OFFSET`: taps before the centre sample.
const TAPS_BEFORE: i64 = 3;
/// `FILTER_BITS`: precision of the taps.
const FILTER_BITS: u32 = 7;

/// `Upscale_Filter` (AV1 §7.16).
#[rustfmt::skip]
const UPSCALE_FILTERS: [[i16; UPSCALE_TAPS]; UPSCALE_POSITIONS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, -1, 128, 2, -1, 0, 0],
    [0, 1, -3, 127, 4, -2, 1, 0],
    [0, 1, -4, 127, 6, -3, 1, 0],
    [0, 2, -6, 126, 8, -3, 1, 0],
    [0, 2, -7, 125, 11, -4, 1, 0],
    [-1, 2, -8, 125, 13, -5, 2, 0],
    [-1, 3, -9, 124, 15, -6, 2, 0],
    [-1, 3, -10, 123, 18, -6, 2, -1],
    [-1, 3, -11, 122, 20, -7, 3, -1],
    [-1, 4, -12, 121, 22, -8, 3, -1],
    [-1, 4, -13, 120, 25, -9, 3, -1],
    [-1, 4, -14, 118, 28, -9, 3, -1],
    [-1, 4, -15, 117, 30, -10, 4, -1],
    [-1, 5, -16, 116, 32, -11, 4, -1],
    [-1, 5, -16, 114, 35, -12, 4, -1],
    [-1, 5, -17, 112, 38, -12, 4, -1],
    [-1, 5, -18, 111, 40, -13, 5, -1],
    [-1, 5, -18, 109, 43, -14, 5, -1],
    [-1, 6, -19, 107, 45, -14, 5, -1],
    [-1, 6, -19, 105, 48, -15, 5, -1],
    [-1, 6, -19, 103, 51, -16, 5, -1],
    [-1, 6, -20, 101, 53, -16, 6, -1],
    [-1, 6, -20, 99, 56, -17, 6, -1],
    [-1, 6, -20, 97, 58, -17, 6, -1],
    [-1, 6, -20, 95, 61, -18, 6, -1],
    [-2, 7, -20, 93, 64, -18, 6, -2],
    [-2, 7, -20, 91, 66, -19, 6, -1],
    [-2, 7, -20, 88, 69, -19, 6, -1],
    [-2, 7, -20, 86, 71, -19, 6, -1],
    [-2, 7, -20, 84, 74, -20, 7, -2],
    [-2, 7, -20, 81, 76, -20, 7, -1],
    [-2, 7, -20, 79, 79, -20, 7, -2],
    [-1, 7, -20, 76, 81, -20, 7, -2],
    [-2, 7, -20, 74, 84, -20, 7, -2],
    [-1, 6, -19, 71, 86, -20, 7, -2],
    [-1, 6, -19, 69, 88, -20, 7, -2],
    [-1, 6, -19, 66, 91, -20, 7, -2],
    [-2, 6, -18, 64, 93, -20, 7, -2],
    [-1, 6, -18, 61, 95, -20, 6, -1],
    [-1, 6, -17, 58, 97, -20, 6, -1],
    [-1, 6, -17, 56, 99, -20, 6, -1],
    [-1, 6, -16, 53, 101, -20, 6, -1],
    [-1, 5, -16, 51, 103, -19, 6, -1],
    [-1, 5, -15, 48, 105, -19, 6, -1],
    [-1, 5, -14, 45, 107, -19, 6, -1],
    [-1, 5, -14, 43, 109, -18, 5, -1],
    [-1, 5, -13, 40, 111, -18, 5, -1],
    [-1, 4, -12, 38, 112, -17, 5, -1],
    [-1, 4, -12, 35, 114, -16, 5, -1],
    [-1, 4, -11, 32, 116, -16, 5, -1],
    [-1, 4, -10, 30, 117, -15, 4, -1],
    [-1, 3, -9, 28, 118, -14, 4, -1],
    [-1, 3, -9, 25, 120, -13, 4, -1],
    [-1, 3, -8, 22, 121, -12, 4, -1],
    [-1, 3, -7, 20, 122, -11, 3, -1],
    [-1, 2, -6, 18, 123, -10, 3, -1],
    [0, 2, -6, 15, 124, -9, 3, -1],
    [0, 2, -5, 13, 125, -8, 2, -1],
    [0, 1, -4, 11, 125, -7, 2, 0],
    [0, 1, -3, 8, 126, -6, 2, 0],
    [0, 1, -3, 6, 127, -4, 1, 0],
    [0, 1, -2, 4, 127, -3, 1, 0],
    [0, 0, -1, 2, 128, -1, 0, 0],
];

/// The resampling of one plane's rows: the step and first position
/// (AV1 §7.16, `stepX` and `initialSubpelX`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1Superres {
    step: i64,
    initial: i64,
    upscaled_width: usize,
}

impl Av1Superres {
    /// The resampling from `downscaled_width` to `upscaled_width` samples
    /// (the plane widths `Round2(FrameWidth, subX)` and
    /// `Round2(UpscaledWidth, subX)`).
    ///
    /// # Panics
    ///
    /// If either width is zero.
    #[must_use]
    pub fn new(downscaled_width: usize, upscaled_width: usize) -> Self {
        assert!(
            downscaled_width > 0 && upscaled_width > 0,
            "empty superres plane"
        );
        let (down, up) = (downscaled_width as i64, upscaled_width as i64);
        let step = ((down << SCALE_BITS) + up / 2) / up;
        let error = up * step - (down << SCALE_BITS);
        let initial = ((-((up - down) << (SCALE_BITS - 1)) + up / 2) / up
            + (1 << (EXTRA_BITS - 1))
            - error / 2)
            & SCALE_MASK;
        Self {
            step,
            initial,
            upscaled_width,
        }
    }

    /// Width of the upscaled rows.
    #[must_use]
    pub fn upscaled_width(&self) -> usize {
        self.upscaled_width
    }

    /// Upscale one row. `src` holds every sample the decoder reconstructed
    /// in the row (its width rounded up to whole 4x4 blocks, `MiCols`);
    /// positions beyond it read its last sample. Writes
    /// [`Self::upscaled_width`] samples to `dst`.
    ///
    /// # Panics
    ///
    /// If `src` is empty or `dst` is too short.
    pub fn upscale_row(&self, src: &[u8], dst: &mut [u8]) {
        assert!(!src.is_empty(), "empty superres source row");
        let last = src.len() as i64 - 1;
        for (x, out) in dst[..self.upscaled_width].iter_mut().enumerate() {
            let position = -(1 << SCALE_BITS) + self.initial + x as i64 * self.step;
            let sample = position >> SCALE_BITS;
            let phase = ((position & SCALE_MASK) >> EXTRA_BITS) as usize;
            let sum: i32 = UPSCALE_FILTERS[phase]
                .iter()
                .enumerate()
                .map(|(k, &tap)| {
                    let at = (sample + k as i64 - TAPS_BEFORE).clamp(0, last) as usize;
                    i32::from(tap) * i32::from(src[at])
                })
                .sum();
            *out = ((sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS).clamp(0, 255) as u8;
        }
    }

    /// Upscale every row of `src` (as for [`Self::upscale_row`]) into
    /// `dst`.
    ///
    /// # Panics
    ///
    /// If `dst` is narrower than the upscaled width or shorter than `src`.
    pub fn upscale_plane(&self, src: &PlaneView<'_>, dst: &mut PlaneViewMut<'_>) {
        assert!(
            dst.width() >= self.upscaled_width && dst.height() >= src.height(),
            "superres destination too small"
        );
        for y in 0..src.height() {
            self.upscale_row(&src.row(y)[..src.width()], dst.row_mut(y));
        }
    }
}
