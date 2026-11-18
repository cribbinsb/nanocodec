//! AV1 CDEF block filter (AV1 §7.15.2), bit-exact with libaom's
//! `cdef_filter_block_internal` for 8-bit video.
//!
//! Blocks at least [`CDEF_PADDING`] samples from every edge read an 8-bit
//! copy of the unfiltered plane ([`CdefSource`]); no tap is unavailable there.
//! NEON filters these bytes with split byte accumulators. Blocks near an
//! edge read a **padded 16-bit
//! window** of it instead: every sample outside the frame holds
//! [`CDEF_UNAVAILABLE`], `i16::MIN` (`0x8000`; libaom uses
//! `CDEF_VERY_LARGE` and tests for it). Like dav1d's padding, that value
//! can never influence the result, with no test:
//!
//! - `constrain(p - x)` is zero for it: the 16-bit difference wraps to
//!   `0x8000 - x`, whose magnitude shifted by `shift <= 6` is at least 500
//!   as an unsigned value, so the saturating `strength - (|d| >> shift)` is
//!   zero;
//! - as an unsigned value (32768) it never wins the clipping minimum;
//! - as a signed value it never wins the clipping maximum.
//!
//! So the 16-bit kernels need no per-sample bounds tests, and every tap of a
//! row is one contiguous load.
//!
//! Padded SIMD kernels specialize on the block's direction: every tap is a
//! fixed row and column of a `(n + 4)`-square window checked once per block.
//! Interior kernels constrain byte differences before widening: AVX2 packs
//! four 8-sample rows, NEON and SIMD128 two. AVX512ICL instead selects four
//! taps per pixel from an 8x8 byte register table and accumulates them with
//! GFNI shifts and VNNI dot products (`avx512`), for both luma and chroma.
//!
//! The direction search (AV1 §7.15.1, libaom `cdef_find_dir`) reads the
//! 8-bit plane directly; see the `direction` module.
//!
//! # Kernel contract
//!
//! Kernels are `unsafe fn`s called only from [`crate::Dsp`], which checks the
//! CPU, that the padded source covers the block plus [`CDEF_PADDING`] on every
//! side, and that the destination covers the block. The NEON entry checks
//! complete footprints before pointer-based lane loads/stores. It borrows
//! the caller's interior and sentinel-padded windows; uncommon edge pitches
//! use compact scratch with every accessed payload initialized. Unused
//! trailing scratch columns remain uninitialized. Other kernels index
//! checked slices/windows.

use nc_common::Sample;

#[allow(unsafe_code)]
pub(crate) mod band;
// Shared word-lane direction search: high-depth NEON and SIMD128.
// Eight-bit NEON uses the full dav1d one-pass port in neon_direction;
// x86 has direction_avx2.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code)]
pub(crate) mod direction;
#[allow(unsafe_code)]
pub(crate) mod generic;
pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod direction_avx2;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod neon;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon_direction;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon_filter;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon_padding;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128;

/// Value of padded samples outside the frame (module docs).
pub const CDEF_UNAVAILABLE: i16 = i16::MIN;
/// Samples of padding the 16-bit source needs on every side of a block: the
/// filter taps reach two rows and two columns away.
pub const CDEF_PADDING: usize = 2;

/// Filter directions (AV1 `Cdef_Directions`): (row, column) steps of the two
/// primary taps for each of the 8 directions.
pub(crate) const DIRECTIONS: [[(isize, isize); 2]; 8] = [
    [(-1, 1), (-2, 2)],
    [(0, 1), (-1, 2)],
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2)],
    [(1, 1), (2, 2)],
    [(1, 0), (2, 1)],
    [(1, 0), (2, 0)],
    [(1, 0), (2, -1)],
];
/// Primary taps by the parity of the (adjusted) primary strength.
pub(crate) const PRI_TAPS: [[i16; 2]; 2] = [[4, 2], [3, 3]];
/// Secondary taps.
pub(crate) const SEC_TAPS: [i16; 2] = [2, 1];

/// One block's filter, resolved to what the kernels need: strengths, the
/// constraint shifts, the primary taps and the direction.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CdefArgs {
    /// Primary strength (already adjusted for luma); 0 disables the
    /// primary taps.
    pub pri: i16,
    /// Secondary strength (0, 1, 2 or 4); 0 disables the secondary taps.
    pub sec: i16,
    /// `max(0, damping - floor(log2(pri)))`.
    pub pri_shift: u32,
    /// `max(0, damping - floor(log2(sec)))`.
    pub sec_shift: u32,
    /// Primary taps for this strength's parity.
    pub pri_taps: [i16; 2],
    /// Filter direction (0..=7): primary taps along it, secondary taps
    /// along `dir + 2` and `dir + 6`.
    pub dir: usize,
}

/// One block of a [`crate::Dsp::cdef_filter_blocks`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdefBlock {
    /// Column of the block's top-left sample.
    pub x: usize,
    /// Row of the block's top-left sample.
    pub y: usize,
    /// Primary strength, already adjusted for luma (0..=15).
    pub pri: u8,
    /// Filter direction (0..=7).
    pub dir: u8,
}

impl CdefArgs {
    /// These arguments with direction `dir`.
    pub(crate) fn with_dir(self, dir: u8) -> Self {
        Self {
            dir: usize::from(dir & 7),
            ..self
        }
    }

    /// Resolve a block's parameters.
    pub(crate) fn new(pri: u8, sec: u8, dir: u8, damping: u8) -> Self {
        Self::scaled(u16::from(pri), u16::from(sec), dir, (damping, 0))
    }

    /// Resolve a block's parameters at a depth `bits_above` bits over 8
    /// (libaom `av1_cdef_filter_fb`): `pri` and `sec` already scaled by
    /// `1 << bits_above` (and `pri` adjusted after that), the damping one
    /// bit larger per bit; the primary taps by the parity of the unscaled
    /// strength.
    pub(crate) fn scaled(pri: u16, sec: u16, dir: u8, (damping, bits_above): (u8, u32)) -> Self {
        let damping = u32::from(damping) + bits_above;
        let shift = |strength: u16| -> u32 {
            if strength == 0 {
                0
            } else {
                damping.saturating_sub(strength.ilog2())
            }
        };
        Self {
            pri: pri as i16,
            sec: sec as i16,
            pri_shift: shift(pri),
            sec_shift: shift(sec),
            pri_taps: PRI_TAPS[usize::from((pri >> bits_above) & 1)],
            dir: usize::from(dir & 7),
        }
    }
}

/// A CDEF kernel: filter the `n`x`n` block whose top-left sample is at
/// `src[origin]` (row stride `stride`) into `dst` (row stride `dst_stride`).
pub(crate) type CdefFn = unsafe fn(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
);

/// An 8-bit CDEF kernel: filter the `n`x`n` block whose top-left sample is
/// at `src[origin]` (row stride `stride`) into `dst` (row stride
/// `dst_stride`), reading the unfiltered 8-bit plane directly. Every tap
/// must lie inside the plane (the block is at least [`CDEF_PADDING`]
/// samples from each edge), so no sample is unavailable.
pub(crate) type CdefFn8 = unsafe fn(
    src: &[u8],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
);

/// An 8-bit kernel filtering two 4x4 blocks in one pass ([`CdefFn8`] for
/// each), of one plane or of two (the U and V blocks of one 8x8 unit):
/// shared strengths and damping (`args`, whose direction is unused), block
/// `i` read from `src[i]` at `origins[i]` (row stride `stride`) with
/// direction `dirs[i]` and written to the 4x4 block starting at `dst[i]`
/// (row stride `dst_stride`). Only ISAs whose vectors hold more than one 4x4
/// block have one.
///
/// The destinations are pointers because two blocks of one plane interleave
/// their rows: the caller guarantees that each points at four writable rows
/// of four samples that nothing else references during the call.
pub(crate) type CdefPairFn8 = unsafe fn(
    src: [&[u8]; 2],
    stride: usize,
    origins: [usize; 2],
    args: &CdefArgs,
    dirs: [usize; 2],
    dst: [*mut u8; 2],
    dst_stride: usize,
);

/// A direction-search kernel: libaom `cdef_find_dir` on the 8x8 block of
/// 8-bit samples at the start of `src` (row stride `stride`), giving the
/// direction (0..=7) and the directional contrast (`var`).
pub(crate) type FindDirFn = unsafe fn(src: &[u8], stride: usize) -> (u8, i32);

/// High-bit-depth (`u16`) direction search on an 8x8 luma block (`bits_above`
/// bits over 8).
pub(crate) type FindDir16Fn = unsafe fn(src: &[u16], stride: usize, bits_above: u32) -> (u8, i32);

/// High-bit-depth (`u16`) CDEF kernel reading a padded `i16` window (edge blocks).
pub(crate) type CdefEdge16Fn = unsafe fn(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
);

/// High-bit-depth (`u16`) CDEF kernel reading an unpadded `u16` plane (interior blocks).
pub(crate) type CdefInterior16Fn = unsafe fn(
    src: &[u16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u16],
    dst_stride: usize,
);

/// High-bit-depth (`u16`) dual 4x4 interior kernel (U and V blocks of one 8x8 unit).
pub(crate) type CdefPair16Fn = unsafe fn(
    src: [&[u16]; 2],
    stride: usize,
    origins: [usize; 2],
    args: &CdefArgs,
    dst: [*mut u16; 2],
    dst_stride: usize,
);

/// High-bit-depth (`u16`) CDEF kernels of one ISA.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HighCdefFns {
    /// Direction search of 8x8 luma blocks.
    pub find_dir: FindDir16Fn,
    /// Padded-window edge kernels for `[4x4, 8x8]`.
    pub edge: [CdefEdge16Fn; 2],
    /// Interior kernels for `[4x4, 8x8]`.
    pub interior: [CdefInterior16Fn; 2],
    /// Dual U+V 4x4 interior kernel where vectors hold 32 `i16` lanes.
    pub interior_pair4: Option<CdefPair16Fn>,
}

impl HighCdefFns {
    /// Plain scalar reference table.
    pub const SCALAR: Self = Self {
        find_dir: scalar::find_dir_u16,
        edge: [scalar::block4_edge_u16, scalar::block8_edge_u16],
        interior: [
            scalar::interior_rows::<u16, 4>,
            scalar::interior_rows::<u16, 8>,
        ],
        interior_pair4: None,
    };
}

/// The CDEF kernels of one ISA.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CdefFns {
    /// Direction search of 8x8 luma blocks.
    pub find_dir: FindDirFn,
    /// 8x8 blocks (luma).
    pub block8: CdefFn,
    /// 4x4 blocks (4:2:0 chroma).
    pub block4: CdefFn,
    /// 8x8 blocks away from the plane's edges, read from 8-bit samples.
    pub block8_interior: CdefFn8,
    /// 4x4 blocks away from the plane's edges, read from 8-bit samples.
    pub block4_interior: CdefFn8,
    /// Two such 4x4 blocks at once, where the vectors are wide enough.
    pub block4_interior_pair: Option<CdefPairFn8>,
}

/// Samples in the largest padded window [`CdefSource::padded_window`]
/// builds: an 8x8 block plus [`CDEF_PADDING`] on every side.
pub(crate) const MAX_WINDOW: usize = (8 + 2 * CDEF_PADDING) * (8 + 2 * CDEF_PADDING);

/// A plane's unfiltered samples (8-bit by default): the input [`crate::Dsp::cdef_filter`]
/// reads while the plane itself is overwritten. Blocks away from the edges
/// read it directly; the others get a padded 16-bit window of it (module
/// docs). Reusable across frames.
///
/// The rows can be copied in a band at a time ([`Self::start`], then
/// [`Self::fill_rows_to`]), just ahead of the blocks being filtered in place,
/// so each band is copied while it is still in cache. A band filter that
/// moves down the plane also drops the rows no later block reads
/// ([`Self::drop_rows_before`]): the source then holds a line buffer of one
/// band plus its padding rows rather than a copy of the plane.
#[derive(Debug, Clone, Default)]
pub struct CdefSource<T: Sample = u8> {
    data: Vec<T>,
    width: usize,
    height: usize,
    /// Plane row of the first row still held (earlier rows are logically dropped).
    first_row: usize,
    /// Plane row after the last one `data` holds (kept, not divided out of
    /// the length: the drivers ask for it per block).
    end_row: usize,
}

impl<T: Sample> CdefSource<T> {
    /// An empty source; fill it with [`Self::fill_from`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            width: 0,
            height: 0,
            first_row: 0,
            end_row: 0,
        }
    }

    /// Copy `plane` in. Storage only grows, for reuse across planes.
    pub fn fill_from(&mut self, plane: &nc_common::PlaneView<'_, T>) {
        self.start(plane.width(), plane.height());
        self.fill_rows_to(plane, plane.height());
    }

    /// Begin a `width` x `height` plane with no rows copied yet.
    pub fn start(&mut self, width: usize, height: usize) {
        self.data.clear();
        self.data.reserve(width * height);
        (self.width, self.height, self.first_row, self.end_row) = (width, height, 0, 0);
    }

    /// Copy the rows of `plane` from the first not yet copied up to `end`
    /// (exclusive, clamped to the plane). Rows already copied are kept: the
    /// caller may have filtered the plane's own copy of them since.
    ///
    /// # Panics
    ///
    /// If `plane` is not the size given to [`Self::start`].
    pub fn fill_rows_to(&mut self, plane: &nc_common::PlaneView<'_, T>, end: usize) {
        assert_eq!(
            (plane.width(), plane.height()),
            (self.width, self.height),
            "CDEF source and plane sizes differ"
        );
        for y in self.end_row..end.min(self.height) {
            self.data.extend_from_slice(plane.row(y));
        }
        self.end_row = self.end_row.max(end.min(self.height));
    }

    /// Forget the rows above plane row `row` (no later block reads them),
    /// keeping the copied rows from `row` on. Rows not copied yet stay
    /// uncopied.
    pub fn drop_rows_before(&mut self, row: usize) {
        self.first_row = self.first_row.max(row.min(self.end_row));
    }

    /// The plane row after the last one copied (the plane's height after
    /// [`Self::fill_from`]).
    #[must_use]
    pub fn filled_rows(&self) -> usize {
        self.end_row
    }

    /// The first plane row still held (0 unless rows were dropped).
    #[must_use]
    pub fn first_row(&self) -> usize {
        self.first_row
    }

    /// Plane width in samples.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Plane height in samples.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The unfiltered sample at (`x`, `y`), which must lie inside the plane
    /// and the rows held.
    #[must_use]
    pub fn sample(&self, x: usize, y: usize) -> T {
        assert!(
            x < self.width && y < self.height,
            "sample outside the plane"
        );
        self.samples()[self.index(x, y)]
    }

    /// The rows held, as a plane view (all of the plane after
    /// [`Self::fill_from`]).
    #[must_use]
    pub fn view(&self) -> nc_common::PlaneView<'_, T> {
        nc_common::PlaneView::new(
            &self.data[self.first_row * self.width..],
            self.width,
            self.width,
            self.filled_rows() - self.first_row,
        )
        .expect("whole rows of the plane's width")
    }

    /// Backing buffer (stride = width); plane sample (`x`, `y`) is at
    /// [`Self::index`].
    #[inline]
    pub(crate) fn samples(&self) -> &[T] {
        &self.data
    }

    /// Index in [`Self::samples`] of plane sample (`x`, `y`), a row held.
    ///
    /// # Panics
    ///
    /// If row `y` was dropped.
    #[inline]
    pub(crate) fn index(&self, x: usize, y: usize) -> usize {
        assert!(y >= self.first_row, "CDEF source row dropped");
        y * self.width + x
    }

    /// The `size`-square block at (`x`, `y`) plus [`CDEF_PADDING`] samples
    /// on every side, as 16-bit values with [`CDEF_UNAVAILABLE`] outside the
    /// plane (samples up to 12 bits fit); row stride `size + 2 *
    /// CDEF_PADDING`. Every row of the window inside the plane must be
    /// held.
    pub(crate) fn padded_window(&self, x: usize, y: usize, size: usize) -> [i16; MAX_WINDOW] {
        let side = size + 2 * CDEF_PADDING;
        let mut window = [CDEF_UNAVAILABLE; MAX_WINDOW];
        // Window columns `inside` are plane columns `x + c − CDEF_PADDING`
        // within the plane; the same for every row.
        let inside =
            CDEF_PADDING.saturating_sub(x)..side.min((self.width + CDEF_PADDING).saturating_sub(x));
        let held = self.samples();
        for (r, out) in window[..side * side].chunks_exact_mut(side).enumerate() {
            let Some(row) = (y + r)
                .checked_sub(CDEF_PADDING)
                .filter(|&row| row < self.height)
            else {
                continue;
            };
            if inside.is_empty() {
                continue;
            }
            let first = self.index(x + inside.start - CDEF_PADDING, row);
            let line = &held[first..first + inside.len()];
            for (value, &sample) in out[inside.clone()].iter_mut().zip(line) {
                *value = sample.to_i32() as i16;
            }
        }
        window
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::PlaneView;

    #[test]
    fn padded_windows_mark_samples_outside_the_plane() {
        let (width, height) = (10, 6);
        let pixels: Vec<u8> = (0..width * height).map(|i| (i * 31) as u8).collect();
        let mut source = CdefSource::new();
        source.fill_from(&PlaneView::new(&pixels, width, width, height).unwrap());
        for size in [4, 8] {
            let side = size + 2 * CDEF_PADDING;
            for (x, y) in [(0, 0), (2, 0), (width - size, height - 4)] {
                let window = source.padded_window(x, y, size);
                for r in 0..side {
                    for c in 0..side {
                        let (px, py) = ((x + c) as isize - 2, (y + r) as isize - 2);
                        let inside =
                            (0..width as isize).contains(&px) && (0..height as isize).contains(&py);
                        let expected = if inside {
                            i16::from(pixels[py as usize * width + px as usize])
                        } else {
                            CDEF_UNAVAILABLE
                        };
                        assert_eq!(window[r * side + c], expected);
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx2_interior;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod simd128_interior;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod avx512;
