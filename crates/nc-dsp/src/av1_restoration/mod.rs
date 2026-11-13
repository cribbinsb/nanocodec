//! AV1 loop restoration and super-resolution (AV1 §7.16, §7.17), 8-bit.
//!
//! - [`superres`]: horizontal upscaling of frames coded at a reduced width.
//! - [`Av1LoopRestoration`]: one plane's restoration, in place: units of
//!   [`Av1LrFilter`] ([`wiener`] or [`self_guided`]) over 64-row stripes
//!   (luma; offset 8 rows up), each reading the CDEF output inside its
//!   stripe and, across internal stripe edges, the deblocked rows kept in
//!   [`Av1StripeBoundaries`] (the spec's `get_source_sample`, §7.17.6).
//!
//! Units, stripes and boundary rows follow libaom v3.15.3
//! `av1/common/restoration.c` (BSD-2-Clause, see `THIRD_PARTY.md`). The
//! in-place stripe schedule and the row kernels ([`kernels`]: scalar
//! references, AVX2, NEON, SIMD128) follow dav1d 1.5.4
//! `src/lr_apply_tmpl.c` and `src/looprestoration_tmpl.c` (BSD-2-Clause):
//! each stripe is copied once into a padded window holding exactly the
//! samples `get_source_sample` returns, so every unit filters straight from
//! it into the plane and the kernels need no edge cases.

#[cfg(target_arch = "x86_64")]
mod avx2;
pub(crate) mod kernels;
#[cfg(target_arch = "aarch64")]
mod neon;
mod self_guided;
mod sgr_lanes;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128;
pub mod superres;
mod wiener;

use nc_common::{PlaneView, PlaneViewMut};

use crate::Dsp;

pub use superres::Av1Superres;

/// Source samples each filter reads beyond its region on every side
/// (`RESTORATION_BORDER`).
pub const AV1_LR_BORDER: usize = 3;
/// Values the row kernels may read (and, in scratch rows, write) past the
/// end of a row ([`kernels`] contract): a few vectors.
const AV1_LR_OVERREAD: usize = 32;
/// Luma rows per stripe (`RESTORATION_PROC_UNIT_SIZE`).
const STRIPE_HEIGHT: usize = 64;
/// Luma rows the stripes (and unit rows) are shifted up
/// (`RESTORATION_UNIT_OFFSET`).
const STRIPE_OFFSET: usize = 8;
/// Deblocked rows kept on each side of an internal stripe edge
/// (`RESTORATION_CTX_VERT`).
const CONTEXT_ROWS: usize = 2;

/// The Wiener coefficients of a unit (`LrWiener`): the three outer taps
/// of each direction (the centre tap is derived).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Av1Wiener {
    /// Vertical taps (pass 0 of the syntax).
    pub vertical: [i8; 3],
    /// Horizontal taps (pass 1 of the syntax).
    pub horizontal: [i8; 3],
}

/// The self-guided parameters of a unit (`LrSgrSet`, `LrSgrXqd`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Av1SgrProj {
    /// Parameter set, 0 to 15.
    pub set: u8,
    /// The two coded projection weights.
    pub xqd: [i8; 2],
}

impl Av1SgrProj {
    /// Whether each of the two passes of parameter set `set` (0 to 15)
    /// runs; the syntax codes a projection weight only for a pass that
    /// runs (AV1 §5.11.58).
    ///
    /// # Panics
    ///
    /// If `set` is 16 or more.
    #[must_use]
    pub fn passes(set: u8) -> [bool; 2] {
        self_guided::passes(set)
    }
}

/// The filter of one restoration unit (`LrType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Av1LrFilter {
    /// `RESTORE_NONE`: the unit keeps its CDEF output.
    #[default]
    None,
    /// `RESTORE_WIENER`.
    Wiener(Av1Wiener),
    /// `RESTORE_SGRPROJ`.
    SgrProj(Av1SgrProj),
}

/// Units along one plane dimension of `size` samples for units of
/// `unit_size` (`count_units_in_frame`): rounded to nearest, at least one;
/// the last unit takes the remainder.
#[must_use]
pub fn av1_lr_unit_count(unit_size: usize, size: usize) -> usize {
    ((size + unit_size / 2) / unit_size).max(1)
}

/// Deblocked samples around each internal stripe edge of one plane:
/// for the edge at row `y`, rows `y - 2 .. y + 2` (clamped to the plane),
/// at the upscaled width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1StripeBoundaries {
    width: usize,
    rows: Vec<u8>,
}

/// First row of stripe `index` of a plane subsampled vertically by
/// `sub_y` (the first stripe is shorter by the offset).
fn stripe_start(index: usize, sub_y: bool) -> usize {
    (index * STRIPE_HEIGHT).saturating_sub(STRIPE_OFFSET) >> u32::from(sub_y)
}

/// Rows of the 2 + 2 context lines of one stripe edge.
const EDGE_ROWS: usize = 2 * CONTEXT_ROWS;

impl Av1StripeBoundaries {
    /// Keep the boundary rows of `deblocked` (the plane after deblocking,
    /// before CDEF), upscaling them by `superres` when the frame uses it.
    /// `deblocked` is the plane at its coded width (for superres, the
    /// readable width rounded up to whole 4x4 blocks) and visible height.
    #[must_use]
    pub fn save(deblocked: &PlaneView<'_>, sub_y: bool, superres: Option<&Av1Superres>) -> Self {
        let height = deblocked.height();
        let width = superres.map_or(deblocked.width(), Av1Superres::upscaled_width);
        let mut rows = Vec::new();
        let mut row = vec![0u8; width];
        for index in 1.. {
            let edge = stripe_start(index, sub_y);
            if edge >= height {
                break;
            }
            for y in edge - CONTEXT_ROWS..edge + CONTEXT_ROWS {
                let source = &deblocked.row(y.min(height - 1))[..deblocked.width()];
                match superres {
                    Some(s) => s.upscale_row(source, &mut row),
                    None => row.copy_from_slice(source),
                }
                rows.extend_from_slice(&row);
            }
        }
        Self { width, rows }
    }

    /// Deblocked row `y` (clamped to the edge's context) next to the
    /// edge of stripe `index` (at least 1).
    fn row(&self, index: usize, offset: usize) -> &[u8] {
        let start = ((index - 1) * EDGE_ROWS + offset) * self.width;
        &self.rows[start..start + self.width]
    }
}

/// One plane's loop restoration parameters: units of `unit_size`
/// samples, `columns` x `rows` of them, filters in raster order.
#[derive(Debug, Clone, Copy)]
pub struct Av1LoopRestoration<'a> {
    /// `LoopRestorationSize` of the plane.
    pub unit_size: usize,
    /// Units per row ([`av1_lr_unit_count`] of the upscaled width).
    pub columns: usize,
    /// Unit rows ([`av1_lr_unit_count`] of the height).
    pub rows: usize,
    /// The filter of each unit, row by row.
    pub filters: &'a [Av1LrFilter],
}

/// The span `[start, end)` of unit `index` of `count` along a dimension
/// of `size` samples (columns: units are not shifted).
fn unit_span(index: usize, count: usize, unit_size: usize, size: usize) -> (usize, usize) {
    let start = index * unit_size;
    let end = if index + 1 == count {
        size
    } else {
        (index + 1) * unit_size
    };
    (start, end)
}

/// The first `len` values of `buffer`, growing it (zeroed) when it is
/// shorter but never shrinking it, so reused scratch is not cleared again.
fn grow<T: Copy + Default>(buffer: &mut Vec<T>, len: usize) -> &mut [T] {
    if buffer.len() < len {
        buffer.resize(len, T::default());
    }
    &mut buffer[..len]
}

/// Loop restoration's working buffers (the stripe window and the filters'
/// row rings), kept by the caller across planes and frames so that they
/// are allocated and zeroed once, as dav1d keeps its line buffers in its
/// frame context.
#[derive(Debug, Default)]
pub struct Av1LrScratch {
    /// The current stripe with its border ([`StripeWindow`]).
    window: Vec<u8>,
    /// The Wiener intermediate rows.
    wiener: Vec<i16>,
    /// The self-guided rings.
    self_guided: self_guided::SgrScratch,
}

/// The rows `top .. end` of stripe `index` of a plane, plus
/// [`AV1_LR_BORDER`] rows and columns around them, as `get_source_sample`
/// (AV1 §7.17.6) reads them: columns and rows clamped to the plane; above
/// the stripe (past an internal edge) the two deblocked rows above it, the
/// upper one repeated; below the stripe the two deblocked rows below it,
/// the lower one repeated. Rows are `stride` apart, [`AV1_LR_OVERREAD`]
/// samples longer than the bordered width.
struct StripeWindow {
    index: usize,
    top: usize,
    end: usize,
    stride: usize,
}

impl StripeWindow {
    /// Copy the window of the stripe into `out` from `plane` (the CDEF
    /// output, not yet restored in this stripe) and `boundaries`.
    fn gather(
        &self,
        plane: &PlaneView<'_>,
        boundaries: &Av1StripeBoundaries,
        sub_y: bool,
        out: &mut Vec<u8>,
    ) {
        let (width, height) = (plane.width(), plane.height());
        let border = AV1_LR_BORDER;
        let next_top = stripe_start(self.index + 1, sub_y);
        let rows = self.end - self.top + 2 * border;
        let out = grow(out, rows * self.stride);
        for (t, row) in out.chunks_exact_mut(self.stride).enumerate() {
            let y = (self.top + t).saturating_sub(border).min(height - 1);
            let source: &[u8] = if y < self.top {
                // Rows top - 2 and - 1 are context rows 0 and 1 of this
                // stripe's top edge; row top - 3 repeats top - 2.
                boundaries.row(self.index, (y + CONTEXT_ROWS).saturating_sub(self.top))
            } else if y >= next_top {
                // Rows next_top and + 1 are context rows 2 and 3 of the next
                // stripe's edge; row next_top + 2 repeats next_top + 1.
                let offset = CONTEXT_ROWS + (y - next_top).min(CONTEXT_ROWS - 1);
                boundaries.row(self.index + 1, offset)
            } else {
                &plane.row(y)[..width]
            };
            row[border..border + width].copy_from_slice(source);
            row[..border].fill(source[0]);
            row[border + width..2 * border + width].fill(source[width - 1]);
        }
    }
}

impl Av1LoopRestoration<'_> {
    /// Restore one plane in place (AV1 §7.17): `plane` holds the CDEF
    /// output (after superres), `boundaries` its [`Av1StripeBoundaries`];
    /// units without a filter keep their samples. `scratch` may hold
    /// anything; reusing it saves its allocations.
    ///
    /// # Panics
    ///
    /// If the unit grid does not match `filters`, or the plane is empty.
    pub fn apply(
        &self,
        dsp: &Dsp,
        plane: &mut PlaneViewMut<'_>,
        boundaries: &Av1StripeBoundaries,
        sub_y: bool,
        scratch: &mut Av1LrScratch,
    ) {
        assert_eq!(
            self.filters.len(),
            self.columns * self.rows,
            "restoration unit grid"
        );
        let (width, height) = (plane.width(), plane.height());
        assert!(width > 0 && height > 0, "empty restoration plane");
        let kernels = dsp.av1_lr();
        let offset = STRIPE_OFFSET >> u32::from(sub_y);
        let stride = width + 2 * AV1_LR_BORDER + AV1_LR_OVERREAD;
        for index in 0.. {
            let top = stripe_start(index, sub_y);
            if top >= height {
                break;
            }
            let end = stripe_start(index + 1, sub_y).min(height);
            // Units start on stripe edges, so a stripe lies in one unit row.
            let unit_row = ((top + offset) / self.unit_size).min(self.rows - 1);
            let filters = &self.filters[unit_row * self.columns..][..self.columns];
            if filters.iter().all(|&f| f == Av1LrFilter::None) {
                continue;
            }
            let stripe = StripeWindow {
                index,
                top,
                end,
                stride,
            };
            stripe.gather(&plane.as_view(), boundaries, sub_y, &mut scratch.window);
            for (column, &filter) in filters.iter().enumerate() {
                let (x0, x1) = unit_span(column, self.columns, self.unit_size, width);
                let window = (&scratch.window[x0..], stride);
                let (size, dst) = ((x1 - x0, end - top), (&mut *plane, x0, top));
                match filter {
                    Av1LrFilter::None => {}
                    Av1LrFilter::Wiener(coefficients) => {
                        wiener::wiener(
                            kernels,
                            coefficients,
                            window,
                            size,
                            &mut scratch.wiener,
                            dst,
                        );
                    }
                    Av1LrFilter::SgrProj(params) => {
                        self_guided::self_guided(
                            kernels,
                            params,
                            window,
                            size,
                            &mut scratch.self_guided,
                            dst,
                        );
                    }
                }
            }
        }
    }
}
