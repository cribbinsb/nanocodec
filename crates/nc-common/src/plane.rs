//! An owned plane of 8-bit samples with aligned rows and a padded border.
//!
//! Motion compensation reads blocks that hang off the picture edge, and
//! sub-pixel filters read a few samples beyond every block. Rather than
//! clamping coordinates in inner loops, every plane keeps a border around the
//! visible area. [`Plane::extend_borders`] fills it with the nearest edge
//! sample, which is exactly the edge behaviour all three codecs specify for
//! out-of-picture references.
//!
//! Memory layout (one allocation, `stride` bytes per row):
//!
//! ```text
//!   <- pad_x -> <------ width ------> <- pad_x (+ alignment slack) ->
//!   +---------+---------------------+---------------------------------+  ^
//!   |                     border rows (top)                          |  border
//!   +---------+---------------------+---------------------------------+  v
//!   |  left   |   visible samples   |  right                          |
//!   +---------+---------------------+---------------------------------+
//!   |                     border rows (bottom)                       |
//!   +----------------------------------------------------------------+
//! ```
//!
//! `pad_x` is the requested border rounded up to [`ROW_ALIGNMENT`], so the
//! first visible sample of every row is aligned.

use crate::aligned::AlignedBuf;
use crate::error::{Error, Result};
use crate::view::{PlaneView, PlaneViewMut, required_len};

/// Alignment of the first visible sample of every row, in bytes. One cache
/// line, and a multiple of every SIMD register width we use.
pub const ROW_ALIGNMENT: usize = 64;

/// Largest supported plane dimension: the largest VP9 and AV1 frame side.
const MAX_DIMENSION: usize = 1 << 16;
/// Largest plane area in samples, padding included; with
/// [`MAX_DIMENSION`] it keeps all offset arithmetic far from overflow on
/// 32-bit targets (wasm32).
const MAX_AREA: usize = 1 << 28;

/// How [`Plane::extend_borders_with`] stores the border samples. The
/// samples are the same whatever the stores; only how they reach memory
/// differs (`nc_dsp::Dsp` streams them past the cache where that pays).
pub trait BorderStores {
    /// Set every byte of `dst` to `value`.
    fn fill(&self, dst: &mut [u8], value: u8);
    /// Copy `src` into `dst` (both the same length).
    fn copy(&self, dst: &mut [u8], src: &[u8]);
    /// Called once after the last store of an extension (e.g. to order
    /// weakly ordered stores before the plane is shared).
    fn finish(&self) {}
}

/// Ordinary stores, through the cache.
#[derive(Debug, Clone, Copy, Default)]
pub struct CachedStores;

impl BorderStores for CachedStores {
    fn fill(&self, dst: &mut [u8], value: u8) {
        dst.fill(value);
    }

    fn copy(&self, dst: &mut [u8], src: &[u8]) {
        dst.copy_from_slice(src);
    }
}

/// An owned plane of 8-bit samples. See the module documentation for layout.
#[derive(Debug, Clone)]
pub struct Plane {
    buf: AlignedBuf,
    width: usize,
    height: usize,
    stride: usize,
    pad_x: usize,
    pad_y: usize,
}

impl Plane {
    /// A zeroed `width` x `height` plane with at least `border` samples of
    /// padding on every side.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidDimensions`] if a dimension is zero, or a dimension or
    /// the border exceeds the supported maximum.
    pub fn new(width: usize, height: usize, border: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions {
                width,
                height,
                reason: "plane dimensions must be non-zero",
            });
        }
        if width > MAX_DIMENSION || height > MAX_DIMENSION || border > MAX_DIMENSION {
            return Err(Error::InvalidDimensions {
                width,
                height,
                reason: "plane dimension or border too large",
            });
        }
        let pad_x = border.next_multiple_of(ROW_ALIGNMENT);
        let pad_y = border;
        let stride = conflict_free_stride(width + 2 * pad_x);
        let rows = height + 2 * pad_y;
        if stride.checked_mul(rows).is_none_or(|area| area > MAX_AREA) {
            return Err(Error::InvalidDimensions {
                width,
                height,
                reason: "plane area too large",
            });
        }
        Ok(Self {
            buf: AlignedBuf::zeroed(stride * rows, ROW_ALIGNMENT),
            width,
            height,
            stride,
            pad_x,
            pad_y,
        })
    }

    /// Visible samples per row.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Visible rows.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Bytes between the starts of consecutive rows.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Padding available on the left and right of every row (≥ the requested border).
    #[must_use]
    pub fn border_x(&self) -> usize {
        self.pad_x
    }

    /// Padding rows available above and below the visible area.
    #[must_use]
    pub fn border_y(&self) -> usize {
        self.pad_y
    }

    /// The visible area.
    // Backends take this view per transform block; out of line it measured
    // ~66 instructions per call (callgrind, HEVC code_residual).
    #[must_use]
    #[inline]
    pub fn view(&self) -> PlaneView<'_> {
        self.region(0, 0, self.width, self.height)
            .expect("the visible area is always inside the plane")
    }

    /// The visible area, writable.
    #[inline]
    pub fn view_mut(&mut self) -> PlaneViewMut<'_> {
        let (width, height) = (self.width, self.height);
        self.region_mut(0, 0, width, height)
            .expect("the visible area is always inside the plane")
    }

    /// The whole padded area as one view: `width + 2 * border_x` by
    /// `height + 2 * border_y` samples, with visible sample (0, 0) at
    /// (`border_x`, `border_y`). Motion searches index candidate blocks in
    /// this view directly instead of creating one view per candidate.
    #[must_use]
    pub fn padded_view(&self) -> PlaneView<'_> {
        let (width, height) = (self.width + 2 * self.pad_x, self.height + 2 * self.pad_y);
        self.region(
            -(self.pad_x as isize),
            -(self.pad_y as isize),
            width,
            height,
        )
        .expect("the padded area is the whole plane")
    }

    /// The `width` x `height` window whose top-left sample is at (`x`, `y`)
    /// in visible coordinates. Negative and past-the-edge coordinates reach
    /// into the border. `None` if the window leaves the padded area.
    #[must_use]
    #[allow(
        clippy::inline_always,
        reason = "called per kernel invocation; out of line it measured ~65 instructions per call (callgrind)"
    )]
    #[inline(always)]
    pub fn region(&self, x: isize, y: isize, width: usize, height: usize) -> Option<PlaneView<'_>> {
        let start = self.offset_of(x, y, width, height)?;
        let len = required_len(width, height, self.stride);
        PlaneView::new(
            &self.buf.as_slice()[start..start + len],
            self.stride,
            width,
            height,
        )
    }

    /// Writable version of [`Plane::region`].
    #[allow(
        clippy::inline_always,
        reason = "called per kernel invocation; out of line it measured ~65 instructions per call (callgrind)"
    )]
    #[inline(always)]
    pub fn region_mut(
        &mut self,
        x: isize,
        y: isize,
        width: usize,
        height: usize,
    ) -> Option<PlaneViewMut<'_>> {
        let start = self.offset_of(x, y, width, height)?;
        let len = required_len(width, height, self.stride);
        let stride = self.stride;
        PlaneViewMut::new(
            &mut self.buf.as_mut_slice()[start..start + len],
            stride,
            width,
            height,
        )
    }

    /// The samples from the top-left of the `width` x `height` window at
    /// (`x`, `y`) (as [`Plane::region`]) to the end of the plane, rows
    /// [`Plane::stride`] apart: for kernels that check their footprint
    /// themselves, without building a view per call. `None` if the window
    /// leaves the padded area.
    #[must_use]
    #[inline]
    pub fn samples_at(&self, x: isize, y: isize, width: usize, height: usize) -> Option<&[u8]> {
        let start = self.offset_of(x, y, width, height)?;
        Some(&self.buf.as_slice()[start..])
    }

    /// Writable version of [`Plane::samples_at`].
    #[inline]
    pub fn samples_at_mut(
        &mut self,
        x: isize,
        y: isize,
        width: usize,
        height: usize,
    ) -> Option<&mut [u8]> {
        let start = self.offset_of(x, y, width, height)?;
        Some(&mut self.buf.as_mut_slice()[start..])
    }

    /// Fill the border by replicating the nearest visible sample.
    ///
    /// Call after the visible area changes (e.g. after reconstructing a
    /// reference frame) and before motion compensation reads from it.
    pub fn extend_borders(&mut self) {
        self.extend_borders_with(&CachedStores);
    }

    /// [`Self::extend_borders`], writing the border through `stores`.
    pub fn extend_borders_with(&mut self, stores: &impl BorderStores) {
        self.extend_border_rows_with(0..self.height, stores);
    }

    /// Replicate edge samples into the border beside the visible `rows`,
    /// and above or below them when they include the first or last row: a
    /// picture whose rows become final in bands extends each band once
    /// (every row of the picture, in any order, gives
    /// [`Self::extend_borders`]).
    ///
    /// # Panics
    ///
    /// If `rows` reaches past the plane's height.
    pub fn extend_border_rows(&mut self, rows: core::ops::Range<usize>) {
        self.extend_border_rows_with(rows, &CachedStores);
    }

    /// [`Self::extend_border_rows`], writing the border through `stores`.
    pub fn extend_border_rows_with(
        &mut self,
        rows: core::ops::Range<usize>,
        stores: &impl BorderStores,
    ) {
        assert!(rows.end <= self.height, "rows beyond the plane");
        if rows.is_empty() {
            return;
        }
        let (stride, pad_x, pad_y) = (self.stride, self.pad_x, self.pad_y);
        let (width, height) = (self.width, self.height);
        let data = self.buf.as_mut_slice();

        for y in pad_y + rows.start..pad_y + rows.end {
            let row = &mut data[y * stride..(y + 1) * stride];
            let (left_value, right_value) = (row[pad_x], row[pad_x + width - 1]);
            stores.fill(&mut row[..pad_x], left_value);
            stores.fill(&mut row[pad_x + width..], right_value);
        }

        if rows.start == 0 {
            let (above, rest) = data.split_at_mut(pad_y * stride);
            let first_row = &rest[..stride];
            for row in above.chunks_exact_mut(stride) {
                stores.copy(row, first_row);
            }
        }

        if rows.end == height {
            let last_visible = (pad_y + height - 1) * stride;
            let (upper, below) = data.split_at_mut(last_visible + stride);
            let last_row = &upper[last_visible..];
            for row in below.chunks_exact_mut(stride) {
                stores.copy(row, last_row);
            }
        }
        stores.finish();
    }

    /// Byte offset of visible coordinate (`x`, `y`), if a `width` x `height`
    /// window there stays inside the padded area.
    #[allow(
        clippy::inline_always,
        reason = "called per kernel invocation; out of line it measured ~65 instructions per call (callgrind)"
    )]
    #[inline(always)]
    fn offset_of(&self, x: isize, y: isize, width: usize, height: usize) -> Option<usize> {
        let col = usize::try_from(x + self.pad_x as isize).ok()?;
        let row = usize::try_from(y + self.pad_y as isize).ok()?;
        let fits = col + width <= self.stride && row + height <= self.height + 2 * self.pad_y;
        fits.then_some(row * self.stride + col)
    }
}

/// The row stride for `row_bytes`: a multiple of [`ROW_ALIGNMENT`] that is
/// an odd number of alignment units. A stride that is a multiple of a large
/// power of two (1920 + 2 * 64 = 2048) maps the rows of a block column onto
/// a few L1 sets, so the 16-21 rows a motion-compensation window or a
/// deblocking edge touch evict each other (callgrind, H.264 1080p decode:
/// 2.5x the D1 misses of ffmpeg). An odd multiple spreads consecutive rows
/// over every set, for at most one alignment unit more per row.
fn conflict_free_stride(row_bytes: usize) -> usize {
    let stride = row_bytes.next_multiple_of(ROW_ALIGNMENT);
    if (stride / ROW_ALIGNMENT).is_multiple_of(2) {
        stride + ROW_ALIGNMENT
    } else {
        stride
    }
}

#[cfg(test)]
mod tests {
    use super::{Plane, ROW_ALIGNMENT};

    #[test]
    fn visible_rows_are_aligned() {
        let plane = Plane::new(1280, 720, 40).unwrap();
        assert_eq!(plane.stride() % ROW_ALIGNMENT, 0);
        assert!(plane.border_x() >= 40);
        let view = plane.view();
        for y in 0..view.height() {
            assert_eq!(view.row(y).as_ptr() as usize % ROW_ALIGNMENT, 0);
        }
    }

    #[test]
    fn strides_are_odd_multiples_of_the_alignment() {
        for width in [64, 1280, 1920, 3840] {
            let plane = Plane::new(width, 16, 64).unwrap();
            assert_eq!(plane.stride() % (2 * ROW_ALIGNMENT), ROW_ALIGNMENT);
        }
    }

    #[test]
    fn rejects_bad_dimensions() {
        assert!(Plane::new(0, 10, 8).is_err());
        assert!(Plane::new(10, 1 << 20, 8).is_err());
        // The largest VP9 test vector (20400 wide); area bounds the rest.
        assert!(Plane::new(20_400, 120, 0).is_ok());
        assert!(Plane::new(1 << 16, 1 << 16, 0).is_err());
    }

    #[test]
    fn region_reaches_into_border_but_not_beyond() {
        let plane = Plane::new(16, 8, 8).unwrap();
        let bx = plane.border_x() as isize;
        let by = plane.border_y() as isize;
        assert!(plane.region(-bx, -by, 4, 4).is_some());
        assert!(plane.region(-bx - 1, 0, 4, 4).is_none());
        assert!(plane.region(0, -by - 1, 4, 4).is_none());
        assert!(plane.region(0, 8 + by - 4, 4, 4).is_some());
        assert!(plane.region(0, 8 + by - 3, 4, 4).is_none());
    }

    #[test]
    fn extend_borders_replicates_edges() {
        let mut plane = Plane::new(4, 3, 4).unwrap();
        {
            let mut v = plane.view_mut();
            for y in 0..3 {
                let row = v.row_mut(y);
                for (x, s) in row.iter_mut().enumerate() {
                    *s = (10 * y + x) as u8;
                }
            }
        }
        plane.extend_borders();
        let border = plane.border_y() as isize;
        let at = |x: isize, y: isize| plane.region(x, y, 1, 1).unwrap().row(0)[0];
        assert_eq!(at(-1, 0), 0); // left of (0,0)
        assert_eq!(at(4, 1), 13); // right of (3,1)
        assert_eq!(at(2, -border), 2); // above (2,0)
        assert_eq!(at(1, 2 + border), 21); // below (1,2)
        assert_eq!(at(-1, -1), 0); // corner takes the corner sample
        assert_eq!(at(5, 3), 23);
    }

    #[test]
    fn banded_extension_equals_whole_extension() {
        let filled = || {
            let mut plane = Plane::new(6, 7, 4).unwrap();
            let mut v = plane.view_mut();
            for y in 0..7 {
                for (x, s) in v.row_mut(y).iter_mut().enumerate() {
                    *s = (13 * y + 3 * x) as u8;
                }
            }
            plane
        };
        let mut whole = filled();
        whole.extend_borders();
        let mut banded = filled();
        for rows in [2..5, 0..2, 5..7] {
            banded.extend_border_rows(rows);
        }
        assert_eq!(whole.padded_view().data(), banded.padded_view().data());
    }
}
