//! The warp models of causal and extended warp blocks (dav2d `decode.c`
//! `derive_warpmv`, `extend_warpmv` and `recon_b`'s `MM_WARP_EXTEND`
//! neighbour choice; dav2d commit 237229e, BSD-2-Clause, see
//! `docs/licenses/dav2d`).
use super::Tile;
use super::motion::candidates::Candidate;
use super::motion::vector::TIP_REFERENCE;
use super::motion::{MotionStore, MotionUnit, Mv};
use super::size::BlockSize;
use crate::general::sample::Av2Sample;
use crate::general::warp::{IDENTITY, OFF_DIAGONAL_MAX, UNITY, WarpModel, kind};

/// Causal warps fit at most eight neighbour samples.
const MAX_SAMPLES: usize = 8;
/// Blocks this many units wide or less fit their above-right neighbour.
const CORNER_MAX_UNITS: u32 = 16;
/// Extending a neighbour's model rounds to multiples of 64 within these
/// diagonal bounds.
const DIAGONAL_MIN: i32 = UNITY - OFF_DIAGONAL_MAX;
const DIAGONAL_MAX: i32 = UNITY + OFF_DIAGONAL_MAX;
/// Eighth-sample vectors to Q16 translations.
const TRANSLATION_LOG2: u32 = 13;
/// Sample positions in Q16.
const POSITION_LOG2: u32 = 16;

/// `v` rounded to a multiple of 64 (halves away from zero below `pivot`)
/// and clipped.
fn round_entry(v: i32, pivot: i32, (min, max): (i32, i32)) -> i32 {
    (v.wrapping_add(0x20 - i32::from(v < pivot)) & !0x3f).clamp(min, max)
}

/// The neighbour samples of a causal warp: each a position beside the
/// block and where its vector for `reference` moves it, in eighth samples
/// from the block's top-left.
struct Samples {
    list: Vec<[[i32; 2]; 2]>,
    reference: i8,
}

impl Samples {
    fn full(&self) -> bool {
        self.list.len() >= MAX_SAMPLES
    }

    /// The neighbour `b` at unit offset (`dx`, `dy`) from the block, on
    /// side (`sx`, `sy`) of it (dav2d `add_sample`).
    fn add(&mut self, (dx, dy): (i32, i32), (sx, sy): (i32, i32), b: &MotionUnit) {
        let vectors = if b.warped { b.warp_mv } else { b.mv };
        let (bw, bh) = b.size.units();
        for (&r, v) in b.refs.0.iter().zip(vectors) {
            if r != self.reference {
                continue;
            }
            let x = 16 * (2 * dx + sx * bw as i32) - 8;
            let y = 16 * (2 * dy + sy * bh as i32) - 8;
            self.list.push([[x, y], [x + v.x, y + v.y]]);
            if self.full() {
                break;
            }
        }
    }
}

impl<T: Av2Sample> Tile<'_, '_, T> {
    /// A causal warp: the model fitted to the vectors of the neighbours
    /// that use `reference` (dav2d `derive_warpmv`).
    pub(super) fn derive_warp(&self, size: BlockSize, reference: i8, mv: Mv) -> WarpModel {
        let Some(store) = &self.motion else {
            return WarpModel::default();
        };
        let (bw4, bh4) = size.units();
        let (row, column) = (self.row, self.column);
        let h4 = bh4.min(self.geometry.rows - row);
        let side = self.geometry.superblock_units;
        let inside = row & (side - 1) != 0;
        let mut samples = Samples {
            list: Vec::with_capacity(MAX_SAMPLES),
            reference,
        };
        let (mut have_top_left, have_top_right) = if row > self.bounds.row_start {
            self.top_samples(store, &mut samples, (bw4, inside))
        } else {
            (false, false)
        };
        if !samples.full() && column > self.bounds.column_start {
            let first = store.unit(row, column - 1);
            let mut off = first.row as i32 - row as i32;
            have_top_left &= off == 0;
            loop {
                let b = store.unit((row as i32 + off) as u32, column - 1);
                samples.add((0, off), (-1, 1), b);
                off += b.size.units().1 as i32;
                if off >= h4 as i32 || samples.full() {
                    break;
                }
            }
        } else {
            have_top_left = false;
        }
        let corners = [
            have_top_left
                .then(|| self.top_left_corner(store, inside))
                .flatten(),
            have_top_right
                .then(|| self.top_right_corner(store, bw4, inside))
                .flatten(),
        ];
        for (corner, dx) in corners.into_iter().zip([0, bw4 as i32]) {
            if let Some(b) = corner
                && !samples.full()
            {
                samples.add((dx, 0), (if dx == 0 { -1 } else { 1 }, -1), b);
            }
        }
        let mut model = WarpModel::of_matrix(kind::AFFINE, IDENTITY);
        model.fit(&samples.list, (bw4, bh4), mv, (row, column));
        model.finish();
        model
    }

    /// The samples along the block's top, and whether the above-left and
    /// above-right corners may add theirs. `inside`: the top lies inside
    /// the superblock (else the 8x8 row above is read).
    fn top_samples(
        &self,
        store: &MotionStore,
        samples: &mut Samples,
        (bw4, inside): (u32, bool),
    ) -> (bool, bool) {
        let (row, column) = (self.row, self.column);
        let w4 = bw4.min(self.geometry.columns - column) as i32;
        let side = self.geometry.superblock_units;
        let start = column as i32;
        let right = column + bw4;
        let (have_top_left, mut have_top_right);
        if inside {
            let mut off = store.unit(row - 1, column).column as i32 - start;
            have_top_left = off == 0;
            loop {
                let b = store.unit(row - 1, (start + off) as u32);
                samples.add((off, 0), (1, -1), b);
                off += b.size.units().0 as i32;
                if off >= w4 || samples.full() {
                    break;
                }
            }
            have_top_right = off <= bw4 as i32;
        } else {
            have_top_left = true;
            // A block crossing the top-left corner is the corner sample;
            // one crossing the top-right corner joins here.
            let first = store.above(column >> 1);
            let mut off = i32::from(first.column + first.size.units().0 <= column);
            let extends = right & (side - 1) != 0
                && right < self.bounds.column_end
                && store.above(right >> 1).column < right;
            loop {
                let b = store.above(((start + off) >> 1) as u32);
                let ioff = b.column as i32 - start;
                samples.add((ioff, 0), (1, -1), b);
                // Past the block (the 8x8 rounding would revisit it).
                off = ioff + b.size.units().0 as i32 + 1;
                if off >= w4 + i32::from(extends) || samples.full() {
                    break;
                }
            }
            have_top_right = true;
        }
        have_top_right &= bw4 <= CORNER_MAX_UNITS
            && (right + u32::from(!inside)) < self.bounds.column_end
            && (!inside || (right & (side - 1) != 0 && !store.unit(row - 1, right).is_intra()));
        (have_top_left, have_top_right)
    }

    /// The above-left corner block, at a superblock top only when it ends
    /// at the block's column.
    fn top_left_corner<'s>(&self, store: &'s MotionStore, inside: bool) -> Option<&'s MotionUnit> {
        let (row, column) = (self.row, self.column);
        if inside {
            return Some(store.unit(row - 1, column - 1));
        }
        let b = if column & (self.geometry.superblock_units - 1) != 0 {
            store.above((column - 1) >> 1)
        } else {
            store.above_left()
        };
        (b.size.units().0 + b.column == column).then_some(b)
    }

    /// The above-right corner block, at a superblock top only when it
    /// starts at the block's right edge.
    fn top_right_corner<'s>(
        &self,
        store: &'s MotionStore,
        bw4: u32,
        inside: bool,
    ) -> Option<&'s MotionUnit> {
        let (row, right) = (self.row, self.column + bw4);
        if inside {
            return Some(store.unit(row - 1, right));
        }
        let b = store.above((right + 1) >> 1);
        (b.column == right).then_some(b)
    }

    /// The neighbour unit at offsets (`dy`, `dx`) from the block: across a
    /// superblock top, from the 8x8 row above.
    fn warp_neighbour<'s>(&self, store: &'s MotionStore, (dy, dx): (i32, i32)) -> &'s MotionUnit {
        let (row, column) = (self.row as i32, self.column as i32);
        let side = self.geometry.superblock_units as i32;
        if dy == -1 && row & (side - 1) == 0 {
            if column + dx < (column & !(side - 1)) {
                store.above_left()
            } else {
                store.above(((column + dx) >> 1) as u32)
            }
        } else {
            store.unit((row + dy) as u32, (column + dx) as u32)
        }
    }

    /// Which neighbour an extended warp starts from (dav2d `recon_b`'s
    /// `MM_WARP_EXTEND`): the chosen candidate's left or above neighbour
    /// (not a TIP block), else the first of bottom-left, right-most above,
    /// top-left, left-most above using `reference`; offsets (`dy`, `dx`).
    fn extend_source(
        &self,
        store: &MotionStore,
        size: BlockSize,
        reference: i8,
        candidate: &Candidate,
    ) -> Option<(i32, i32)> {
        let (bw4, bh4) = size.units();
        let (bw4, bh4) = (bw4 as i32, bh4 as i32);
        let (row, column) = (self.row as i32, self.column as i32);
        let side = self.geometry.superblock_units as i32;
        let top = i32::from(row & (side - 1) == 0);
        if candidate.x_off == -1 || candidate.y_off == -1 {
            let (dy, dx) = (i32::from(candidate.y_off), i32::from(candidate.x_off));
            // dav2d tests for TIP with its own choice of the unit above a
            // superblock top (column, not the left edge, decides).
            let b = if top != 0 && dy == -1 {
                if column & (side - 1) != 0 || dx >= 0 {
                    store.above(((column + dx) >> 1) as u32)
                } else {
                    store.above_left()
                }
            } else {
                store.unit((row + dy) as u32, (column + dx) as u32)
            };
            if b.refs.get(0) != TIP_REFERENCE {
                return Some((dy, dx));
            }
        }
        let have_top = row > self.bounds.row_start as i32;
        let have_left = column > self.bounds.column_start as i32;
        let uses = |b: &MotionUnit| b.refs.get(0) == reference || b.refs.get(1) == reference;
        // The units above at column offset `dx`: at a superblock top the
        // 8x8 row's, from the even column.
        let above = |dx: i32| {
            if top != 0 {
                store.above((((column & !1) + dx) >> 1) as u32)
            } else {
                store.unit((row - 1) as u32, (column + dx) as u32)
            }
        };
        let bottom_left = (have_left && row + bh4 <= self.bounds.row_end as i32)
            .then(|| store.unit((row + bh4 - 1) as u32, (column - 1) as u32));
        let right_most = (have_top && column + bw4 <= self.bounds.column_end as i32)
            .then(|| above(if top != 0 { bw4 - 2 } else { bw4 - 1 }));
        let top_left = have_left.then(|| store.unit(row as u32, (column - 1) as u32));
        let left_most = have_top.then(|| above(0));
        [
            (bottom_left, (bh4 - 1, -1)),
            (right_most, (-1, -(column & top) + bw4 - (1 + top))),
            (top_left, (0, -1)),
            (left_most, (-1, -(column & top))),
        ]
        .into_iter()
        .find_map(|(unit, offsets)| unit.filter(|&b| uses(b)).map(|_| offsets))
    }

    /// An extended warp: a neighbour's model (or its vector as a
    /// translation) bent to pass through the block's vector (dav2d
    /// `extend_warpmv`); invalid without a neighbour to extend.
    pub(super) fn extend_warp(
        &self,
        size: BlockSize,
        reference: i8,
        mv: Mv,
        candidate: &Candidate,
    ) -> WarpModel {
        let invalid = WarpModel::of_matrix(kind::INVALID, IDENTITY);
        let Some(store) = &self.motion else {
            return invalid;
        };
        let Some((y_off, x_off)) = self.extend_source(store, size, reference, candidate) else {
            return invalid;
        };
        let b = self.warp_neighbour(store, (y_off, x_off));
        let mut m = if b.warped {
            b.matrix
        } else if b.global {
            store.frame.global[reference as usize].matrix
        } else {
            let v = b.mv[usize::from(b.refs.get(0) != reference)];
            let mut m = IDENTITY;
            m[0] = v.x.wrapping_shl(TRANSLATION_LOG2);
            m[1] = v.y.wrapping_shl(TRANSLATION_LOG2);
            m
        };
        self.bend(&mut m, size, mv, x_off >= 0);
        let mut model = WarpModel::of_matrix(kind::AFFINE, m);
        model.set_translation(size.units(), mv, (self.row, self.column));
        model.finish();
        model
    }

    /// Re-solve the model's column (`from_above`: the neighbour lies
    /// above, so the vertical entries) so its centre moves by `mv`.
    fn bend(&self, m: &mut [i32; 6], size: BlockSize, mv: Mv, from_above: bool) {
        let (bw4, bh4) = size.units();
        let (w_log2, h_log2) = size.log2_units();
        let (row, column) = (i64::from(self.row), i64::from(self.column));
        let sx = column * 4 + 2 * i64::from(bw4) - 1;
        let sy = row * 4 + 2 * i64::from(bh4) - 1;
        let px = (sx << POSITION_LOG2) + (i64::from(mv.x) << TRANSLATION_LOG2);
        let py = (sy << POSITION_LOG2) + (i64::from(mv.y) << TRANSLATION_LOG2);
        let wide = |v: i32| i64::from(v);
        let solve = |p: i64, a: i64, extent: u32, shift: u32| {
            ((p - a + i64::from(extent) - i64::from(p < a)) >> shift) as i32
        };
        let off_diagonal = (-OFF_DIAGONAL_MAX, OFF_DIAGONAL_MAX);
        let diagonal = (DIAGONAL_MIN, DIAGONAL_MAX);
        if from_above {
            let ay = row * 4 - 1;
            let shift = 1 + u32::from(h_log2);
            let apx = wide(m[2]) * sx + wide(m[3]) * ay + wide(m[0]);
            let apy = wide(m[4]) * sx + wide(m[5]) * ay + wide(m[1]);
            m[3] = round_entry(solve(px, apx, bh4, shift), 0, off_diagonal);
            m[5] = round_entry(solve(py, apy, bh4, shift), UNITY, diagonal);
        } else {
            let ax = column * 4 - 1;
            let shift = 1 + u32::from(w_log2);
            let lpx = wide(m[2]) * ax + wide(m[3]) * sy + wide(m[0]);
            let lpy = wide(m[4]) * ax + wide(m[5]) * sy + wide(m[1]);
            m[2] = round_entry(solve(px, lpx, bw4, shift), UNITY, diagonal);
            m[4] = round_entry(solve(py, lpy, bw4, shift), 0, off_diagonal);
        }
    }
}
