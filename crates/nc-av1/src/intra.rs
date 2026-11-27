//! The luma intra prediction of one block as AV1 signals it (a `YMode`, its
//! angle delta, and filter intra), and which edge samples beyond the block
//! a directional prediction may read. Shared by the encoder and decoder.

use nc_common::{BitDepth, Plane, PlaneViewMut};
use nc_dsp::Dsp;
use nc_dsp::intra_prediction::av1_block::{BlockEdges, EdgeAvail, EdgeFilterCtx, IntraKind};

use crate::block_size::BlockSize;
use crate::predict::{EdgeExtension, FilterIntraMode, IntraEdges, IntraMode};

/// Largest `|angle_delta_y|` (AV1 `MAX_ANGLE_DELTA`).
pub const MAX_ANGLE_DELTA: i8 = 3;
/// Degrees per angle-delta step (AV1 `ANGLE_STEP`).
const ANGLE_STEP: i16 = 3;
/// `filter_intra_mode` symbols.
pub const FILTER_INTRA_MODES: usize = FilterIntraMode::ALL.len();
/// Largest block side (log2) that may use filter intra (32x32).
const FILTER_INTRA_MAX_LOG2: u8 = 5;
/// Superblock side in 4x4 units: availability is decided within one.
const SB_UNITS_LOG2: u32 = 4;
const SB_UNITS: usize = 1 << SB_UNITS_LOG2;
/// 4x4 units per sample, log2.
const UNIT_LOG2: u32 = 2;

/// One block's luma intra prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LumaIntra {
    /// The `YMode`.
    pub mode: IntraMode,
    /// `angle_delta_y` in −3..=3 (directional modes only, else 0).
    pub angle_delta: i8,
    /// Filter intra (DC blocks of at most 32x32 only).
    pub filter: Option<FilterIntraMode>,
}

impl LumaIntra {
    /// DC prediction without filter intra.
    pub const DC: Self = Self::plain(IntraMode::Dc);

    /// `mode` at its nominal angle, without filter intra.
    #[must_use]
    pub const fn plain(mode: IntraMode) -> Self {
        Self {
            mode,
            angle_delta: 0,
            filter: None,
        }
    }

    /// The prediction angle in degrees of a directional mode.
    #[must_use]
    pub fn angle(self) -> Option<u16> {
        self.mode
            .nominal_angle()
            .map(|nominal| (nominal as i16 + ANGLE_STEP * i16::from(self.angle_delta)) as u16)
    }

    /// The `YMode` symbol whose transform-type CDFs the block's luma
    /// transform uses (AV1 §5.11.47: filter intra maps to a direction).
    #[must_use]
    pub fn intra_dir(self) -> usize {
        self.filter
            .map_or(self.mode.symbol(), FilterIntraMode::intra_dir)
    }

    /// The `angle_delta_y` symbol (0..=6).
    #[must_use]
    pub fn angle_delta_symbol(self) -> usize {
        (self.angle_delta + MAX_ANGLE_DELTA) as usize
    }

    /// Whether this prediction reads samples beyond the block's own edges.
    #[must_use]
    pub fn needs_extension(self) -> bool {
        self.angle()
            .is_some_and(|angle| angle != 90 && angle != 180)
    }

    /// The edges of the `size` square at (`x`, `y`) of the luma `plane`
    /// this prediction reads.
    #[must_use]
    pub fn edges(self, plane: &Plane, x: usize, y: usize, size: usize) -> IntraEdges {
        let extension = if self.needs_extension() {
            edge_extension(x, y, (size, size))
        } else {
            EdgeExtension::default()
        };
        IntraEdges::extended(plane, x, y, size, extension)
    }

    /// Predict the block from its [`Self::edges`], with `dsp`'s kernels.
    pub fn predict(self, dsp: &Dsp, edges: &IntraEdges, dst: &mut PlaneViewMut<'_>) {
        match (self.filter, self.angle()) {
            (Some(filter), _) => dsp.av1_intra_predict_filter(edges, filter, dst),
            (None, Some(angle)) if self.needs_extension() => {
                dsp.av1_intra_predict_angle(edges, angle, dst);
            }
            _ => dsp.av1_intra_predict(edges, self.mode, dst),
        }
    }
}

impl LumaIntra {
    /// The edges of the `width` x `height` block at (`x`, `y`) of the
    /// 8-bit `plane` (any shape the encoder codes), as the decoder reads
    /// them (`BlockEdges`, AV1 §7.11.2), with the availability of
    /// [`edge_extension`].
    #[must_use]
    pub fn block_edges(plane: &Plane, (x, y): (usize, usize), size: (usize, usize)) -> BlockEdges {
        let extension = edge_extension(x, y, size);
        let avail = EdgeAvail {
            have_above: y > 0,
            have_left: x > 0,
            have_above_right: extension.above_right,
            have_below_left: extension.below_left,
        };
        BlockEdges::read(plane, (x, y), size, avail, BitDepth::EIGHT)
    }

    /// Predict a block of any shape from its [`Self::block_edges`], with
    /// the decoder's kernels (the intra edge filter is off in our
    /// sequences).
    pub fn predict_block(self, dsp: &Dsp, edges: &BlockEdges, dst: &mut PlaneViewMut<'_>) {
        match self.filter {
            Some(filter) => dsp.av1_intra_predict_filter_block(edges, filter, dst),
            None => dsp.av1_intra_predict_block(
                edges,
                intra_kind(self.mode, self.angle_delta, None),
                dst,
            ),
        }
    }
}

/// The prediction of `mode` with angle delta `delta` (§7.11.2.4: `pAngle`
/// is the nominal angle plus `3 * delta`).
pub(crate) fn intra_kind(
    mode: IntraMode,
    delta: i8,
    edge_filter: Option<EdgeFilterCtx>,
) -> IntraKind {
    match mode {
        IntraMode::Dc => IntraKind::Dc,
        IntraMode::Smooth => IntraKind::Smooth,
        IntraMode::SmoothVertical => IntraKind::SmoothVertical,
        IntraMode::SmoothHorizontal => IntraKind::SmoothHorizontal,
        IntraMode::Paeth => IntraKind::Paeth,
        directional => {
            let nominal = directional.nominal_angle().expect("a directional mode");
            IntraKind::Directional {
                angle: (nominal as i16 + ANGLE_STEP * i16::from(delta)) as u16,
                edge_filter,
            }
        }
    }
}

/// Whether filter intra may be signalled on a `1 << log2_width` x
/// `1 << log2_height` block (AV1 §5.11.24: both sides at most 32).
#[must_use]
pub fn filter_intra_allowed(log2_width: u8, log2_height: u8) -> bool {
    log2_width.max(log2_height) <= FILTER_INTRA_MAX_LOG2
}

/// The `filter_intra_cdfs` index of a block: its `MiSize`.
#[must_use]
pub fn filter_intra_size_context(log2_width: u8, log2_height: u8) -> usize {
    BlockSize::new(log2_width, log2_height)
        .expect("an AV1 block size")
        .index()
}

/// Which luma samples beyond the `width` x `height` block at (`x`, `y`) are decoded
/// when it is predicted (AV1 §5.11.35, `BlockDecoded` as set by
/// `clear_block_decoded_flags` and each transform block; §7.11.2
/// `haveAboveRt`: the unit right of the row above, `haveBelowLft`: the unit
/// below the column to the left).
///
/// Our blocks are squares of a quadtree, or the two halves of one of its
/// squares (`PARTITION_HORZ` / `VERT`), with one transform per intra block,
/// so blocks complete in Z order within each 64x64 superblock: a 4x4 unit
/// inside the superblock is decoded exactly when its Z-order index is below
/// the block's. That holds for a half too: the two units asked about lie
/// outside its square, whose units are one contiguous Z-order range (so the
/// half's first unit and the square's compare alike with any unit outside).
/// Above the superblock everything is decoded; to its left everything down
/// to its bottom edge; to its right and below nothing.
/// Units beyond the picture edge count as decoded here (the spec leaves
/// them unset), which reads the same samples: a block never crosses the
/// picture edge, so such a unit is only asked about by a block that ends at
/// that edge, where the edge reads clamp to the last sample either way.
#[must_use]
pub fn edge_extension(x: usize, y: usize, (width, height): (usize, usize)) -> EdgeExtension {
    let unit = |v: usize| (v >> UNIT_LOG2) % SB_UNITS;
    let (col, row) = (unit(x), unit(y));
    let (width, height) = (width >> UNIT_LOG2, height >> UNIT_LOG2);
    let own = z_order(row, col);
    let above_right = y > 0
        && match (row, col + width) {
            (0, _) => true,
            (_, c) if c >= SB_UNITS => false,
            (r, c) => z_order(r - 1, c) < own,
        };
    let below_left = x > 0
        && match (row + height, col) {
            (r, _) if r >= SB_UNITS => false,
            (_, 0) => true,
            (r, c) => z_order(r, c - 1) < own,
        };
    EdgeExtension {
        above_right,
        below_left,
    }
}

/// Z-order (Morton) index of a 4x4 unit in its superblock: row bits above
/// column bits at each level, so quadrants run top-left, top-right,
/// bottom-left, bottom-right.
fn z_order(row: usize, col: usize) -> usize {
    (0..SB_UNITS_LOG2).fold(0, |z, bit| {
        z | (((col >> bit) & 1) << (2 * bit)) | (((row >> bit) & 1) << (2 * bit + 1))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ext(above_right: bool, below_left: bool) -> EdgeExtension {
        EdgeExtension {
            above_right,
            below_left,
        }
    }

    #[test]
    fn quadtree_neighbours_follow_decode_order() {
        // Top-left 32x32 of a superblock below the first row: the row above
        // is decoded, the left superblock too.
        assert_eq!(edge_extension(64, 64, (32, 32)), ext(true, true));
        // Top-right 32x32: its above-right is the next superblock's part of
        // the row above (decoded), its below-left the bottom-left quadrant
        // (not yet).
        assert_eq!(edge_extension(96, 64, (32, 32)), ext(true, false));
        assert_eq!(edge_extension(96, 0, (32, 32)), ext(false, false));
        // Bottom-left 32x32: above-right is the top-right quadrant (done),
        // below-left is the next superblock row (not).
        assert_eq!(edge_extension(64, 96, (32, 32)), ext(true, false));
        // Bottom-right 32x32: neither.
        assert_eq!(edge_extension(96, 96, (32, 32)), ext(false, false));
        // The first 8x8 of the bottom-left 16x16 of the top-left 32x32.
        assert_eq!(edge_extension(64, 80, (8, 8)), ext(true, true));
        // A 64x64 block reads on along the row above; below it is not
        // decoded.
        assert_eq!(edge_extension(64, 64, (64, 64)), ext(true, false));
        // Picture edges: nothing above or to the left.
        assert_eq!(edge_extension(0, 0, (8, 8)), ext(false, false));
    }

    /// The halves of quad-tree squares (`PARTITION_HORZ` / `VERT`): what
    /// the Z-order rule says equals the decoded units a decoder marks block
    /// by block in coding order (the spec's `BlockDecoded`), on every half
    /// of every square of a superblock coded with its neighbours as squares.
    #[test]
    fn halves_of_squares_see_what_a_decoder_has_decoded() {
        let sb = SB_UNITS as isize;
        for log2 in 3..=6_u8 {
            let side = 1usize << log2;
            for y in (0..SB_UNITS << UNIT_LOG2).step_by(side) {
                for x in (0..SB_UNITS << UNIT_LOG2).step_by(side) {
                    // The square's halves, in coding order.
                    let half = side / 2;
                    let shapes = [
                        [(x, y, side, half), (x, y + half, side, half)],
                        [(x, y, half, side), (x + half, y, half, side)],
                    ];
                    for parts in shapes {
                        for (k, &(px, py, w, h)) in parts.iter().enumerate() {
                            // Decoded before the part: every unit of the
                            // superblock earlier in Z order than the
                            // square, and the parts before it.
                            let square_start = z_order(y >> UNIT_LOG2, x >> UNIT_LOG2);
                            let decoded = |row: isize, col: isize| {
                                // The row above the superblock and the
                                // superblock to its left are decoded; to
                                // its right and below nothing.
                                if row < 0 || row >= sb {
                                    return row < 0;
                                }
                                if col < 0 || col >= sb {
                                    return col < 0;
                                }
                                let (row, col) = (row as usize, col as usize);
                                let inside = |&(qx, qy, qw, qh): &(usize, usize, usize, usize)| {
                                    (qy >> UNIT_LOG2..(qy + qh) >> UNIT_LOG2).contains(&row)
                                        && (qx >> UNIT_LOG2..(qx + qw) >> UNIT_LOG2).contains(&col)
                                };
                                if parts.iter().any(inside) {
                                    parts[..k].iter().any(inside)
                                } else {
                                    z_order(row, col) < square_start
                                }
                            };
                            // Superblock (1, 1) of a large frame: the
                            // row above and the column left are decoded.
                            let origin = SB_UNITS << UNIT_LOG2;
                            let got = edge_extension(origin + px, origin + py, (w, h));
                            let (r, c) = ((py >> UNIT_LOG2) as isize, (px >> UNIT_LOG2) as isize);
                            let (wu, hu) = ((w >> UNIT_LOG2) as isize, (h >> UNIT_LOG2) as isize);
                            let want = ext(decoded(r - 1, c + wu), decoded(r + hu, c - 1));
                            assert_eq!(got, want, "{w}x{h} at ({px}, {py}) of the {side} square");
                        }
                    }
                }
            }
        }
    }

    /// The any-shape path ([`LumaIntra::block_edges`],
    /// [`LumaIntra::predict_block`]) predicts every square exactly as the
    /// square path does, for every mode, angle delta and filter intra mode,
    /// at positions with each combination of decoded neighbours.
    #[test]
    fn any_shape_prediction_equals_the_square_path_on_squares() {
        use crate::predict::FilterIntraMode;
        let dsp = Dsp::new();
        let (width, height) = (192, 160);
        let mut plane = Plane::new(width, height, 0).unwrap();
        let mut view = plane.view_mut();
        for row in 0..height {
            for (col, sample) in view.row_mut(row)[..width].iter_mut().enumerate() {
                *sample = ((col * 37 + row * 11 + (col * row) % 7) % 251) as u8;
            }
        }
        let mut lumas: Vec<LumaIntra> = Vec::new();
        for mode in IntraMode::ALL {
            let deltas = if mode.nominal_angle().is_some() {
                -3..=3
            } else {
                0..=0
            };
            for angle_delta in deltas {
                lumas.push(LumaIntra {
                    angle_delta,
                    ..LumaIntra::plain(mode)
                });
            }
        }
        for filter in FilterIntraMode::ALL {
            lumas.push(LumaIntra {
                filter: Some(filter),
                ..LumaIntra::DC
            });
        }
        for size in [8_usize, 16, 32] {
            for (x, y) in [
                (0, 0),
                (64, 0),
                (0, 64),
                (64, 64),
                (96, 64),
                (64, 96),
                (64 + size, 64),
            ] {
                for luma in &lumas {
                    if luma.filter.is_some()
                        && !filter_intra_allowed(size.ilog2() as u8, size.ilog2() as u8)
                    {
                        continue;
                    }
                    let mut square = vec![0; size * size];
                    let mut any = vec![0; size * size];
                    let edges = luma.edges(&plane, x, y, size);
                    luma.predict(
                        &dsp,
                        &edges,
                        &mut PlaneViewMut::new(&mut square, size, size, size).unwrap(),
                    );
                    let edges = LumaIntra::block_edges(&plane, (x, y), (size, size));
                    luma.predict_block(
                        &dsp,
                        &edges,
                        &mut PlaneViewMut::new(&mut any, size, size, size).unwrap(),
                    );
                    assert_eq!(any, square, "{luma:?} {size} at ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn filter_intra_mapping_and_sizes() {
        assert_eq!(filter_intra_size_context(3, 3), 3);
        assert_eq!(filter_intra_size_context(5, 5), 9);
        assert_eq!(filter_intra_size_context(4, 3), 5);
        assert!(filter_intra_allowed(5, 4) && !filter_intra_allowed(6, 5));
        let filtered = LumaIntra {
            filter: Some(FilterIntraMode::D157),
            ..LumaIntra::DC
        };
        assert_eq!(filtered.intra_dir(), IntraMode::D157.symbol());
        let delta = LumaIntra {
            angle_delta: -2,
            ..LumaIntra::plain(IntraMode::D203)
        };
        assert_eq!(delta.angle(), Some(197));
        assert_eq!(delta.angle_delta_symbol(), 1);
    }
}
