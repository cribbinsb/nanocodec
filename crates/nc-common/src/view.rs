//! Borrowed rectangular windows into 8-bit sample data.
//!
//! A view is `height` rows of `width` samples, `stride` bytes apart, starting
//! at `data[0]`. The constructor checks once that the slice covers every row,
//! so code taking a view (including SIMD kernels) can rely on
//! `data.len() >= (height - 1) * stride + width` without re-checking.

/// Copy one block row: `dst.copy_from_slice(src)`, with the block widths
/// (4 to 64 samples) as fixed-size copies.
///
/// A slice copy of unknown length is a `memcpy` call, and block rows are a
/// few bytes, so the call costs more than the copy. aarch64 glibc uses a
/// generic, non-SIMD `memcpy`, which made these copies 6.6% of AV2 encode
/// time on the Jetson (perf, 2026-11-09). A match on the length lets the
/// compiler inline each width as a few vector loads and stores.
///
/// # Panics
/// If the lengths differ, like `copy_from_slice`.
#[allow(
    clippy::inline_always,
    reason = "called per block row from other crates; out of line it cost ~37 instructions per call, more than the memcpy it replaces (callgrind)"
)]
#[inline(always)]
pub fn copy_row(dst: &mut [u8], src: &[u8]) {
    fn fixed<const N: usize>(dst: &mut [u8], src: &[u8]) {
        let src: &[u8; N] = src.try_into().expect("copy_row: source length");
        let dst: &mut [u8; N] = dst.try_into().expect("copy_row: lengths differ");
        *dst = *src;
    }
    match src.len() {
        4 => fixed::<4>(dst, src),
        8 => fixed::<8>(dst, src),
        16 => fixed::<16>(dst, src),
        32 => fixed::<32>(dst, src),
        64 => fixed::<64>(dst, src),
        _ => dst.copy_from_slice(src),
    }
}

/// Bytes a `width` x `height` window with the given stride occupies.
#[allow(
    clippy::inline_always,
    reason = "called per kernel invocation; out of line it measured ~65 instructions per call (callgrind)"
)]
#[inline(always)]
pub(crate) const fn required_len(width: usize, height: usize, stride: usize) -> usize {
    if width == 0 || height == 0 {
        0
    } else {
        (height - 1) * stride + width
    }
}

/// Validate untrusted geometry before views expose it to DSP kernels.
/// Trusted, allocated planes retain the existing `required_len` fast path.
#[inline]
fn checked_required_len(width: usize, height: usize, stride: usize) -> Option<usize> {
    if width == 0 || height == 0 {
        Some(0)
    } else {
        (height - 1).checked_mul(stride)?.checked_add(width)
    }
}

/// An immutable window into sample data.
#[derive(Debug, Clone, Copy)]
pub struct PlaneView<'a> {
    data: &'a [u8],
    stride: usize,
    width: usize,
    height: usize,
}

impl<'a> PlaneView<'a> {
    /// A view over `data`, or `None` if the geometry overflows, `data` is too
    /// short, or `stride < width`.
    #[must_use]
    #[allow(
        clippy::inline_always,
        reason = "called per kernel invocation; out of line it measured ~65 instructions per call (callgrind)"
    )]
    #[inline(always)]
    pub fn new(data: &'a [u8], stride: usize, width: usize, height: usize) -> Option<Self> {
        let valid = stride >= width && data.len() >= checked_required_len(width, height, stride)?;
        valid.then_some(Self {
            data,
            stride,
            width,
            height,
        })
    }

    /// Samples per row.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of rows.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Distance between the starts of consecutive rows, in bytes.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The underlying data, starting at the view's top-left sample. It covers
    /// at least `(height - 1) * stride + width` bytes.
    #[must_use]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Row `y` (exactly `width` samples).
    // Per-row accessors measured out of line on AArch64 (3% of HEVC encode
    // samples on low-light content); inlining lets the bounds check fold.
    #[must_use]
    #[inline]
    pub fn row(&self, y: usize) -> &'a [u8] {
        assert!(
            y < self.height,
            "row {y} outside view of height {}",
            self.height
        );
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }

    /// All rows, top to bottom.
    pub fn rows(&self) -> impl Iterator<Item = &'a [u8]> + '_ {
        (0..self.height).map(|y| self.row(y))
    }

    /// The `width` x `height` window at (`x`, `y`) inside this view, or `None`
    /// if it does not fit.
    #[must_use]
    pub fn sub_view(&self, x: usize, y: usize, width: usize, height: usize) -> Option<Self> {
        if x + width > self.width || y + height > self.height {
            return None;
        }
        let start = y * self.stride + x;
        Self::new(self.data.get(start..)?, self.stride, width, height)
    }
}

/// A mutable window into sample data.
#[derive(Debug)]
pub struct PlaneViewMut<'a> {
    data: &'a mut [u8],
    stride: usize,
    width: usize,
    height: usize,
}

impl<'a> PlaneViewMut<'a> {
    /// A view over `data`, or `None` if the geometry overflows, `data` is too
    /// short, or `stride < width`.
    #[must_use]
    #[allow(
        clippy::inline_always,
        reason = "called per kernel invocation; out of line it measured ~65 instructions per call (callgrind)"
    )]
    #[inline(always)]
    pub fn new(data: &'a mut [u8], stride: usize, width: usize, height: usize) -> Option<Self> {
        let valid = stride >= width && data.len() >= checked_required_len(width, height, stride)?;
        valid.then_some(Self {
            data,
            stride,
            width,
            height,
        })
    }

    /// Samples per row.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of rows.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Distance between the starts of consecutive rows, in bytes.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The underlying data, starting at the view's top-left sample. It covers
    /// at least `(height - 1) * stride + width` bytes; bytes between rows
    /// belong to the view's owner and must not be changed.
    #[must_use]
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.data
    }

    /// A read-only view of the same window.
    #[must_use]
    pub fn as_view(&self) -> PlaneView<'_> {
        PlaneView {
            data: self.data,
            stride: self.stride,
            width: self.width,
            height: self.height,
        }
    }

    /// Row `y` (exactly `width` samples), writable.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [u8] {
        assert!(
            y < self.height,
            "row {y} outside view of height {}",
            self.height
        );
        let start = y * self.stride;
        &mut self.data[start..start + self.width]
    }

    /// Set every sample in the window to `value`.
    pub fn fill(&mut self, value: u8) {
        for y in 0..self.height {
            self.row_mut(y).fill(value);
        }
    }

    /// The `width` x `height` window at (`x`, `y`) inside this view,
    /// writable, or `None` if it does not fit.
    #[must_use]
    pub fn sub_view_mut(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Option<PlaneViewMut<'_>> {
        if x + width > self.width || y + height > self.height {
            return None;
        }
        let start = y * self.stride + x;
        PlaneViewMut::new(self.data.get_mut(start..)?, self.stride, width, height)
    }

    /// Copy `src` into this window. Both must have the same size.
    pub fn copy_from(&mut self, src: &PlaneView<'_>) {
        assert_eq!(
            (self.width, self.height),
            (src.width(), src.height()),
            "copy between views of different sizes"
        );
        for y in 0..self.height {
            self.row_mut(y).copy_from_slice(src.row(y));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PlaneView, PlaneViewMut, copy_row};

    #[test]
    fn copy_row_copies_every_length() {
        for len in 0..=80 {
            let src: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
            let mut dst = vec![0u8; len];
            copy_row(&mut dst, &src);
            assert_eq!(dst, src, "length {len}");
        }
    }

    #[test]
    #[should_panic(expected = "lengths differ")]
    fn copy_row_rejects_mismatched_lengths() {
        copy_row(&mut [0u8; 7], &[0u8; 8]);
    }

    #[test]
    fn rejects_short_data_and_narrow_stride() {
        let data = [0u8; 20];
        assert!(PlaneView::new(&data, 8, 4, 3).is_some()); // needs 2*8+4 = 20
        assert!(PlaneView::new(&data, 8, 5, 3).is_none()); // needs 21
        assert!(PlaneView::new(&data, 3, 4, 1).is_none()); // stride < width
        assert!(PlaneView::new(&[], 0, 0, 0).is_some());
    }

    #[test]
    fn immutable_view_rejects_overflowing_geometry() {
        let data = [0u8; 64];
        // The row offset wraps to zero, so unchecked arithmetic accepts 64 bytes.
        let height = usize::MAX / 64 + 2;
        assert!(PlaneView::new(&data, 64, 64, height).is_none());
        // The row offset fits, but adding the final row width wraps to 32.
        assert!(PlaneView::new(&data, usize::MAX - 31, 64, 2).is_none());
    }

    #[test]
    fn mutable_view_rejects_overflowing_geometry() {
        let mut data = [0u8; 64];
        let height = usize::MAX / 64 + 2;
        assert!(PlaneViewMut::new(&mut data, 64, 64, height).is_none());
        assert!(PlaneViewMut::new(&mut data, usize::MAX - 31, 64, 2).is_none());
    }

    #[test]
    fn empty_views_preserve_geometry_semantics() {
        for (width, height) in [(0, usize::MAX), (usize::MAX, 0)] {
            assert!(PlaneView::new(&[], usize::MAX, width, height).is_some());
            assert!(PlaneViewMut::new(&mut [], usize::MAX, width, height).is_some());
        }
        // An empty height does not waive the stride constraint.
        assert!(PlaneView::new(&[], 0, 1, 0).is_none());
        assert!(PlaneViewMut::new(&mut [], 0, 1, 0).is_none());
    }

    #[test]
    fn rows_and_sub_views_index_correctly() {
        let data: Vec<u8> = (0..40).collect();
        let view = PlaneView::new(&data, 10, 10, 4).unwrap();
        assert_eq!(view.row(2), &data[20..30]);
        let sub = view.sub_view(3, 1, 4, 2).unwrap();
        assert_eq!(sub.row(0), &[13, 14, 15, 16]);
        assert_eq!(sub.row(1), &[23, 24, 25, 26]);
        assert!(view.sub_view(7, 0, 4, 1).is_none());
        assert!(view.sub_view(0, 3, 1, 2).is_none());
    }

    #[test]
    fn mutable_view_fill_and_copy() {
        let mut data = vec![0u8; 30];
        let mut dst = PlaneViewMut::new(&mut data, 10, 3, 3).unwrap();
        dst.fill(9);
        assert_eq!(dst.as_view().row(2), &[9, 9, 9]);

        let src_data: Vec<u8> = (1..=9).collect();
        let src = PlaneView::new(&src_data, 3, 3, 3).unwrap();
        dst.copy_from(&src);
        assert_eq!(dst.as_view().row(1), &[4, 5, 6]);
        assert_eq!(data[3], 0, "bytes between rows are untouched");
    }
}
