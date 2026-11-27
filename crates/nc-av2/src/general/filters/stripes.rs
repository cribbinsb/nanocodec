//! Loop-restoration stripes (dav2d `dav2d_lr_slice`, `lr_sbrow`,
//! `lr_stripe`; AVM `av2_loop_restoration_filter_unit`'s stripes): each
//! tile row's rows in stripes of 64, the first 8 shorter, so that each
//! stripe boundary inside a tile row sits 8 rows above a 64-row band
//! boundary (AVM `save_tile_row_boundary_lines`). Across a boundary
//! between tile rows the filters read the frame after CDEF itself (dav2d's
//! `LR_HAVE_TOP_INTEGRATED` and `LR_HAVE_BOTTOM_INTEGRATED` stripes, the
//! rows above from `lr_cdef_line`). Across other stripe boundaries
//! restoration filters (GDF, the Wiener filters) read beyond their stripe
//! only the
//! [`STRIPE_CONTEXT_ROWS`] rows next to it, and those from the deblocked
//! frame (the rows dav2d saves before CDEF changes them); further rows
//! repeat the last of them.
//!
//! A sequence with `disable_loopfilters_across_tiles` keeps the filters
//! within each tile: tile edges are frame edges to them (AVM v1.0.0
//! `save_tile_row_boundary_lines`' CDEF lines at each tile's top and
//! bottom, `av2_loop_restoration_filter_unit`'s and GDF's
//! `tile_boundary_left`/`_right`; dav2d ignores the flag).

use crate::general::frame_header::FrameHeader;

/// Rows of the deblocked frame a stripe sees above and below it.
pub const STRIPE_CONTEXT_ROWS: usize = 2;
/// Height of a stripe, and how much shorter the first is.
const STRIPE_HEIGHT: usize = 64;
const STRIPE_OFFSET: usize = 8;

/// One stripe: rows `top..bottom` of a frame `frame_height` rows tall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stripe {
    /// First row.
    pub top: usize,
    /// Row after the last.
    pub bottom: usize,
    /// The frame's height.
    pub frame_height: usize,
    /// The luma row whose 64-row band's restoration unit and GDF flag the
    /// stripe takes: 8 rows below its top, except for the first stripe of
    /// a tile row and the 8-row stripe ending a tile row above another
    /// (dav2d `lr_sbrow`'s `row_y`, `lr_stripe`'s `inc`).
    pub band_top: usize,
    /// What lies above its top.
    pub above: Boundary,
    /// What lies below its bottom.
    pub below: Boundary,
}

/// What a filter finds across one side of a stripe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// Another stripe of its tile row: the deblocked context rows.
    Stripe,
    /// Another tile row: the frame after CDEF itself.
    TileRow,
    /// An edge to the filters, the frame's or (in a sequence that keeps
    /// loop filters within tiles) its tile row's: its own edge row repeats.
    Edge,
}

impl Stripe {
    /// Rows in the stripe.
    #[must_use]
    pub fn height(self) -> usize {
        self.bottom - self.top
    }

    /// The frame row a filter reads for the row `offset` rows below the
    /// stripe's top (negative: above it), when it reads at most `context`
    /// rows beyond the stripe: the row and whether it is the deblocked
    /// frame's. Inside the stripe, and across a boundary between tile
    /// rows, it is the row itself; next to another stripe boundary inside
    /// the frame, the deblocked rows; at the frame's top and bottom (and
    /// tile rows' within isolated tiles), the stripe's edge row.
    #[must_use]
    pub fn context_row(self, offset: isize, context: usize) -> (usize, bool) {
        let height = self.height() as isize;
        if (0..height).contains(&offset)
            || (offset < 0 && self.above == Boundary::TileRow)
            || (offset >= height && self.below == Boundary::TileRow)
        {
            (self.top.wrapping_add_signed(offset), false)
        } else if offset < 0 {
            if self.above == Boundary::Edge {
                (self.top, false)
            } else {
                (self.top - offset.unsigned_abs().min(context), true)
            }
        } else if self.below == Boundary::Edge {
            (self.bottom - 1, false)
        } else {
            let below = (offset - height) as usize;
            (self.bottom + below.min(context - 1), true)
        }
    }
}

/// The stripes of a frame `height` rows tall, top to bottom, in tile rows
/// starting at the luma rows `tile_rows` (the first 0); with `isolated`,
/// the filters stay within each tile row.
pub fn stripes(
    height: usize,
    tile_rows: &[usize],
    isolated: bool,
) -> impl Iterator<Item = Stripe> + '_ {
    tile_rows
        .iter()
        .enumerate()
        .flat_map(move |(index, &start)| {
            let end = tile_rows
                .get(index + 1)
                .copied()
                .unwrap_or(height)
                .min(height);
            let mut top = start;
            let boundary = move |frame_edge: bool, tile_edge: bool| {
                if frame_edge || (isolated && tile_edge) {
                    Boundary::Edge
                } else if tile_edge {
                    Boundary::TileRow
                } else {
                    Boundary::Stripe
                }
            };
            std::iter::from_fn(move || {
                if top >= end {
                    return None;
                }
                let bottom = if top == start {
                    start + STRIPE_HEIGHT - STRIPE_OFFSET
                } else {
                    top + STRIPE_HEIGHT
                }
                .min(end);
                let band_top = if top == start || (bottom == end && end < height) {
                    top
                } else {
                    top + STRIPE_OFFSET
                };
                let stripe = Stripe {
                    top,
                    bottom,
                    frame_height: height,
                    band_top,
                    above: boundary(top == 0, top == start),
                    below: boundary(bottom == height, bottom == end),
                };
                top = bottom;
                Some(stripe)
            })
        })
}

/// The tiles' extents in luma samples of a frame `size` samples wide and
/// tall (its coded size): column and row starts, each list ending with the
/// frame's extent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileSpans {
    /// Column starts, then the width.
    pub columns: Vec<usize>,
    /// Row starts, then the height.
    pub rows: Vec<usize>,
    /// The sequence keeps loop filters within tiles
    /// (`disable_loopfilters_across_tiles`).
    pub isolated: bool,
}

impl TileSpans {
    /// The spans of `header`'s tiles; `isolated` when loop filters stay
    /// within them.
    #[must_use]
    pub fn new(header: &FrameHeader, (width, height): (usize, usize), isolated: bool) -> Self {
        let superblock = 1_usize << header.superblock_log2;
        let spans = |starts: &[u16], extent: usize| {
            let mut spans: Vec<usize> = starts[..starts.len() - 1]
                .iter()
                .map(|&start| usize::from(start) * superblock)
                .collect();
            spans.push(extent);
            spans
        };
        Self {
            columns: spans(&header.tiles.column_starts, width),
            rows: spans(&header.tiles.row_starts, height),
            isolated,
        }
    }

    /// The luma columns a filter at luma column `x` may read, as
    /// `start..end`: the frame's, or with isolated tiles its tile's.
    #[must_use]
    pub fn column_span(&self, x: usize) -> (usize, usize) {
        let width = self.columns[self.columns.len() - 1];
        if !self.isolated {
            return (0, width);
        }
        let starts = &self.columns[..self.columns.len() - 1];
        let tile = starts.partition_point(|&start| start <= x) - 1;
        (self.columns[tile], self.columns[tile + 1])
    }

    /// The tile starts of the rows (without the height).
    #[must_use]
    pub fn row_starts(&self) -> &[usize] {
        &self.rows[..self.rows.len() - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stripes_start_eight_rows_above_each_band() {
        let tops: Vec<_> = stripes(128, &[0], false)
            .map(|s| (s.top, s.bottom))
            .collect();
        assert_eq!(tops, [(0, 56), (56, 120), (120, 128)]);
        let short: Vec<_> = stripes(40, &[0], false)
            .map(|s| (s.top, s.bottom))
            .collect();
        assert_eq!(short, [(0, 40)]);
    }

    #[test]
    fn stripes_restart_at_each_tile_row() {
        let stripes: Vec<_> = stripes(256, &[0, 192], false)
            .map(|s| (s.top, s.bottom, s.band_top))
            .collect();
        assert_eq!(
            stripes,
            [
                (0, 56, 0),
                (56, 120, 64),
                (120, 184, 128),
                (184, 192, 184),
                (192, 248, 192),
                (248, 256, 256)
            ]
        );
    }

    #[test]
    fn isolated_tile_rows_are_edges() {
        let stripes: Vec<_> = stripes(256, &[0, 192], true).collect();
        let last_above = stripes[3];
        assert_eq!((last_above.top, last_above.bottom), (184, 192));
        assert_eq!(last_above.below, Boundary::Edge);
        assert_eq!(last_above.context_row(9, 2), (191, false));
        let first_below = stripes[4];
        assert_eq!(first_below.above, Boundary::Edge);
        assert_eq!(first_below.context_row(-1, 2), (192, false));
    }

    #[test]
    fn context_rows_come_from_the_deblocked_frame() {
        let stripe = Stripe {
            top: 56,
            bottom: 120,
            frame_height: 128,
            band_top: 64,
            above: Boundary::Stripe,
            below: Boundary::Stripe,
        };
        assert_eq!(stripe.context_row(-1, 2), (55, true));
        assert_eq!(stripe.context_row(-6, 2), (54, true));
        assert_eq!(stripe.context_row(64, 2), (120, true));
        assert_eq!(stripe.context_row(69, 2), (121, true));
        let first = Stripe {
            top: 0,
            bottom: 56,
            frame_height: 128,
            band_top: 0,
            above: Boundary::Edge,
            below: Boundary::Stripe,
        };
        assert_eq!(first.context_row(-3, 2), (0, false));
        let last = Stripe {
            top: 120,
            bottom: 128,
            frame_height: 128,
            band_top: 128,
            above: Boundary::Stripe,
            below: Boundary::Edge,
        };
        assert_eq!(last.context_row(9, 2), (127, false));
    }
}
