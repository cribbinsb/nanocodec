//! Collocated motion for temporal motion vector prediction (TMVP, HEVC
//! §8.5.3.2.8) when the reference is not always the previous picture.
//!
//! The collocated picture is the current picture's one reference. Its
//! blocks point back over that picture's own reference distance, which in
//! a hierarchical-P structure differs from the current one (an L1T3 T2
//! frame one picture after its T0, whose vectors span four pictures), or
//! point at that picture's second reference (`refs.second`), so collocated
//! vectors are scaled by POC distance (eq. 8-202..8-206). In a flat chain
//! both distances are 1 and vectors are used as they are.
//!
//! The encoder keeps one motion grid per reference slot ([`MotionStore`]);
//! the decoder keeps each reference picture's grid with it in its DPB.

use nc_encoder::{MAX_REFERENCE_SLOTS, Mv, SlotMask};

use crate::grid::Grid;

/// Limits of `td`/`tb` (eq. 8-203/8-204, 8-181/8-182) and of the scale
/// factor (8-205, 8-183).
const POC_DIFF_LIMIT: i32 = 128;
const SCALE_FACTOR_MIN: i32 = -4096;
const SCALE_FACTOR_MAX: i32 = 4095;

/// Reference indices of a P slice: `ref_idx_l0` 0 (the structure's
/// reference) and 1 (the second reference).
pub const MAX_REF_IDX: usize = 2;

/// `mv`, which spans POC distance `td`, scaled to span `tb` (HEVC eq.
/// 8-179..8-183 for spatial AMVP candidates, 8-202..8-206 for collocated
/// ones: the same arithmetic).
///
/// Equal distances keep the vector, as HM (`xGetDistScaleFactor`) and
/// ffmpeg (`mv_mp_mode_mx_lt`) do: the formula's factor is not exactly 256
/// for every distance (`td = tb = 72` gives 257). A zero `td` (a malformed
/// stream's reference with the current POC) also keeps it.
#[must_use]
pub fn scale_mv(mv: Mv, td: i32, tb: i32) -> Mv {
    if td == tb || td == 0 {
        return mv;
    }
    let td = td.clamp(-POC_DIFF_LIMIT, POC_DIFF_LIMIT - 1);
    let tb = tb.clamp(-POC_DIFF_LIMIT, POC_DIFF_LIMIT - 1);
    let tx = (16384 + (td.abs() >> 1)) / td;
    let factor = ((tb * tx + 32) >> 6).clamp(SCALE_FACTOR_MIN, SCALE_FACTOR_MAX);
    let component = |v: i16| {
        let product = factor * i32::from(v);
        let scaled = product.signum() * ((product.abs() + 127) >> 8);
        scaled.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
    };
    Mv {
        x: component(mv.x),
        y: component(mv.y),
    }
}

/// The collocated picture's motion and how its vectors map to the current
/// picture.
#[derive(Debug, Clone, Copy)]
pub struct Collocated<'a> {
    /// Motion of the collocated picture.
    pub grid: &'a Grid,
    /// POC distance from the collocated picture to each of its references,
    /// by `ref_idx_l0` (`colPocDiff` of a block pointing there).
    pub col_distances: [i32; MAX_REF_IDX],
}

impl Collocated<'_> {
    /// The collocated vector `mv`, which points at the collocated picture's
    /// reference `ref_idx`, scaled to span `cur_distance` (`currPocDiff`;
    /// §8.5.3.2.8, eq. 8-202..8-206).
    #[must_use]
    pub fn scale(&self, mv: Mv, ref_idx: u8, cur_distance: i32) -> Mv {
        scale_mv(mv, self.col_distances[usize::from(ref_idx)], cur_distance)
    }
}

/// One saved motion grid and the distances its vectors span, by
/// `ref_idx_l0`.
#[derive(Debug)]
struct SavedMotion {
    grid: Grid,
    distances: [i32; MAX_REF_IDX],
}

/// The encoder's collocated motion per reference slot.
#[derive(Debug)]
pub struct MotionStore {
    pool: Vec<SavedMotion>,
    /// The pool entry each slot holds.
    slots: [usize; MAX_REFERENCE_SLOTS],
    slot_count: usize,
}

impl MotionStore {
    /// A store for `slot_count` slots of `width` x `height` pictures.
    #[must_use]
    pub fn new(width: usize, height: usize, slot_count: usize) -> Self {
        let pool = (0..slot_count)
            .map(|_| SavedMotion {
                grid: Grid::new(width, height),
                distances: [1; MAX_REF_IDX],
            })
            .collect();
        Self {
            pool,
            slots: [0; MAX_REFERENCE_SLOTS],
            slot_count,
        }
    }

    /// Keep `grid` (a finished picture whose vectors span `distances`, by
    /// `ref_idx_l0`) as the collocated motion of the slots in `refresh`,
    /// swapping a grid no remaining slot needs into `grid` for reuse. A flat
    /// chain swaps with its single entry.
    pub fn save(&mut self, grid: &mut Grid, distances: [i32; MAX_REF_IDX], refresh: SlotMask) {
        if refresh.is_empty() {
            return;
        }
        let kept = |entry: usize| {
            (0..self.slot_count).any(|s| !refresh.contains(s) && self.slots[s] == entry)
        };
        let free = (0..self.pool.len())
            .find(|&entry| !kept(entry))
            .expect("refreshed slots release at least one entry");
        std::mem::swap(grid, &mut self.pool[free].grid);
        self.pool[free].distances = distances;
        for slot in 0..self.slot_count {
            if refresh.contains(slot) {
                self.slots[slot] = free;
            }
        }
    }

    /// The collocated motion of the picture in `slot`.
    #[must_use]
    pub fn collocated(&self, slot: usize) -> Collocated<'_> {
        let saved = &self.pool[self.slots[slot]];
        Collocated {
            grid: &saved.grid,
            col_distances: saved.distances,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scaled(col: i32, cur: i32, mv: Mv) -> Mv {
        let grid = Grid::new(16, 16);
        Collocated {
            grid: &grid,
            col_distances: [col, 1],
        }
        .scale(mv, 0, cur)
    }

    #[test]
    fn equal_distances_keep_vectors() {
        let mv = Mv { x: -37, y: 101 };
        assert_eq!(scaled(1, 1, mv), mv);
        assert_eq!(scaled(4, 4, mv), mv);
    }

    #[test]
    fn scaling_follows_the_specification_rounding() {
        // td = 4, tb = 1: tx = 4096, factor = (4096 + 32) >> 6 = 64.
        assert_eq!(scaled(4, 1, Mv { x: 40, y: -6 }), Mv { x: 10, y: -1 });
        // td = 1, tb = 4: factor = 1024.
        assert_eq!(scaled(1, 4, Mv { x: 3, y: -5 }), Mv { x: 12, y: -20 });
        // td = 2, tb = 1: factor 128, 7 -> (896 + 127) >> 8 = 3.
        assert_eq!(scaled(2, 1, Mv { x: 7, y: -7 }), Mv { x: 3, y: -3 });
    }

    #[test]
    fn second_reference_vectors_scale_by_their_own_distance() {
        let grid = Grid::new(16, 16);
        let col = Collocated {
            grid: &grid,
            col_distances: [1, 4],
        };
        let mv = Mv { x: 40, y: -8 };
        assert_eq!(col.scale(mv, 0, 1), mv);
        assert_eq!(col.scale(mv, 1, 1), Mv { x: 10, y: -2 });
        assert_eq!(col.scale(mv, 1, 4), mv);
    }

    #[test]
    fn equal_long_distances_keep_vectors() {
        // The formula alone would give a factor of 257 here.
        let mv = Mv { x: 1000, y: -1000 };
        assert_eq!(scale_mv(mv, 72, 72), mv);
    }

    #[test]
    fn store_keeps_slots_apart() {
        let mut store = MotionStore::new(64, 64, 2);
        let mut grid = Grid::new(64, 64);
        grid.fill(
            0,
            0,
            64,
            crate::grid::Cell {
                mv: Mv { x: 1, y: 0 },
                ..crate::grid::Cell::default()
            },
        );
        store.save(&mut grid, [4, 9], SlotMask::ALL);
        grid.reset();
        grid.fill(
            0,
            0,
            64,
            crate::grid::Cell {
                mv: Mv { x: 2, y: 0 },
                ..crate::grid::Cell::default()
            },
        );
        store.save(&mut grid, [2, 1], SlotMask::only(1));
        assert_eq!(store.collocated(0).grid.at(0, 0).mv.x, 1);
        assert_eq!(store.collocated(0).col_distances, [4, 9]);
        assert_eq!(store.collocated(1).grid.at(0, 0).mv.x, 2);
        assert_eq!(store.collocated(1).col_distances, [2, 1]);
    }
}
