//! Temporal motion vectors: the vectors each frame keeps for later frames
//! (AV1 §7.19 motion field motion vector storage), their projection onto
//! the current frame (§7.9 motion field estimation), and the projection
//! of a projected vector to one of the current block's references
//! (§7.9.3, `get_mv_projection`).
//!
//! Both work on 8x8 units. The projection follows dav1d's `refmvs`
//! (`src/refmvs.c` `load_tmvs`, BSD-2-Clause, Copyright (c) 2020,
//! `VideoLAN` and dav1d authors; see `THIRD_PARTY.md`), reimplemented: a
//! vector only lands in its own 64-sample row band, so the field is built
//! one superblock row at a time, just before that row is decoded, into a
//! buffer of one superblock row (16 rows of units, which stays in cache)
//! instead of a whole frame. A run of equal kept vectors is projected once.
//! Which references project, and in what order, follows libaom
//! `av1_setup_motion_field` (`av1/common/mvref_common.c`, BSD-2-Clause).
use std::ops::Range;

use super::frame_header::{
    ALTREF_FRAME, FrameType, GOLDEN_FRAME, LAST_FRAME, SlotHeader, TOTAL_REFS, relative_distance,
};
use super::motion::{INTRA_FRAME, MotionCell};
use super::sequence::Sequence;
use crate::backend::Av1Mv;

/// `REFMVS_LIMIT`: larger vector components are not kept.
const REFMVS_LIMIT: i32 = (1 << 12) - 1;
/// `MAX_FRAME_DISTANCE`: projection distances are clamped to this.
const MAX_FRAME_DISTANCE: i32 = 31;
/// [`project_mv`] numerators lie within this distance.
pub(crate) const MAX_PROJECTION_DISTANCE: i32 = MAX_FRAME_DISTANCE;
/// `Div_Mult[]` (§7.9.3): `2^14 / d`.
const DIV_MULT: [i32; 32] = [
    0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092,
    1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
];
/// Fraction bits of `Div_Mult`.
const DIV_MULT_BITS: u32 = 14;
/// Bounds of a projected vector component (`MV_LOW + 1`, `MV_UPP - 1`).
const PROJECTED_MV_MAX: i32 = (1 << 14) - 1;
/// 1/8-sample vector units per 8x8 unit, log2.
const UNIT_SHIFT: u32 = 6;
/// A projection stays in its source's band of 8 rows of units (64
/// samples, `MAX_OFFSET_HEIGHT` 0) and moves at most 8 units left or right
/// of its source's band of 8 columns (`MAX_OFFSET_WIDTH`).
const BAND_UNITS: usize = 8;
const MAX_OFFSET_COLUMNS: usize = 8;
/// Rows of units the projected field holds: one 128x128 superblock row
/// (two 64x64 ones share it, each in its own half).
const FIELD_ROWS: usize = 16;
/// `MFMV_STACK_SIZE`: references projected at most.
const MFMV_STACK_SIZE: usize = 3;
const LAST2_FRAME: usize = 2;
const BWDREF_FRAME: usize = 5;
const ALTREF2_FRAME: usize = 6;

/// One 8x8 unit's kept vector: its reference (`MfRefFrames`, or
/// `INTRA_FRAME` for none) and vector, packed in one word: a block's units
/// are filled with one 64-bit store each, and the projection finds runs of
/// equal units with one comparison each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SavedMv(u64);

/// Bit offsets of [`SavedMv`]'s fields.
const SAVED_COL_SHIFT: u32 = 16;
const SAVED_REFERENCE_SHIFT: u32 = 32;

impl SavedMv {
    const NONE: Self = Self::new(INTRA_FRAME, Av1Mv::ZERO);

    const fn new(reference: i8, mv: Av1Mv) -> Self {
        Self(
            (mv.row as u16 as u64)
                | (mv.col as u16 as u64) << SAVED_COL_SHIFT
                | (reference as u8 as u64) << SAVED_REFERENCE_SHIFT,
        )
    }

    fn reference(self) -> i8 {
        (self.0 >> SAVED_REFERENCE_SHIFT) as u8 as i8
    }

    fn mv(self) -> Av1Mv {
        Av1Mv {
            row: self.0 as u16 as i16,
            col: (self.0 >> SAVED_COL_SHIFT) as u16 as i16,
        }
    }
}

/// The vectors a frame keeps (§7.19), one per 8x8 unit.
#[derive(Debug, Clone, Default)]
pub(crate) struct SavedMotion {
    units: Vec<SavedMv>,
    columns: usize,
    rows: usize,
}

impl SavedMotion {
    /// The store of a frame of `mi_cols` x `mi_rows` 4x4 units, empty.
    pub(crate) fn new(mi_cols: usize, mi_rows: usize) -> Self {
        let (columns, rows) = (mi_cols.div_ceil(2), mi_rows.div_ceil(2));
        Self {
            units: vec![SavedMv::NONE; columns * rows],
            columns,
            rows,
        }
    }

    /// Keep the vectors of `cell`, the block at 4x4 unit (`mi_col`,
    /// `mi_row`) covering `w4` x `h4` units inside the frame: the last of
    /// its references that is not ahead of the frame (`side` false) with a
    /// small enough vector.
    pub(crate) fn record(
        &mut self,
        (mi_col, mi_row): (usize, usize),
        (w4, h4): (usize, usize),
        cell: &MotionCell,
        side: [bool; TOTAL_REFS],
    ) {
        let mut kept = SavedMv::NONE;
        for list in 0..2 {
            let reference = cell.refs[list];
            let mv = cell.mv[list];
            let small =
                i32::from(mv.row).abs() <= REFMVS_LIMIT && i32::from(mv.col).abs() <= REFMVS_LIMIT;
            if reference > INTRA_FRAME && !side[reference as usize] && small {
                kept = SavedMv::new(reference, mv);
            }
        }
        let (x8, y8) = (mi_col >> 1, mi_row >> 1);
        for row in y8..y8 + h4.div_ceil(2) {
            let start = row * self.columns;
            self.units[start + x8..start + x8 + w4.div_ceil(2)].fill(kept);
        }
    }

    /// The kept vectors of row `y8` of units.
    fn row(&self, y8: usize) -> &[SavedMv] {
        &self.units[y8 * self.columns..(y8 + 1) * self.columns]
    }
}

/// The references whose order hint is not behind the frame's
/// (`ref_frame_side` nonzero): their vectors are not kept.
pub(crate) fn reference_sides(
    seq: &Sequence,
    order_hint: u32,
    ref_order_hints: &[u32; TOTAL_REFS],
) -> [bool; TOTAL_REFS] {
    std::array::from_fn(|name| {
        name >= LAST_FRAME
            && seq.enable_order_hint
            && relative_distance(seq, ref_order_hints[name], order_hint) >= 0
    })
}

/// One 8x8 unit of the projected field: a reference frame's kept vector
/// (`MotionFieldMvs`) and the distance it spans (`MotionFieldRefOffset`,
/// 1 to 31; 0 marks a unit nothing projected to).
#[derive(Debug, Clone, Copy)]
#[repr(C, align(8))]
pub(crate) struct ProjectedMv {
    pub mv: Av1Mv,
    pub offset: i32,
}

impl ProjectedMv {
    const NONE: Self = Self {
        mv: Av1Mv::ZERO,
        offset: 0,
    };
}

/// A reference frame the field projects from.
pub(crate) struct ProjectionSource<'a> {
    pub header: &'a SlotHeader,
    pub motion: &'a SavedMotion,
}

/// One reference frame's kept vectors as the current frame projects them
/// (§7.9.2).
struct Projection<'a> {
    motion: &'a SavedMotion,
    /// Distance from the reference to the current frame, negated for a
    /// reference behind it (dav1d `mfmv_ref2cur`).
    to_current: i32,
    /// A reference behind the current frame: projections move the other
    /// way.
    backward: bool,
    /// By the reference a kept vector points to: the distance it spans, 0
    /// when that vector does not project (dav1d `mfmv_ref2ref`).
    spans: [i32; TOTAL_REFS],
}

/// The motion field of the current frame (§7.9), one entry per 8x8 unit,
/// held one superblock row at a time. The buffer outlives frames.
#[derive(Debug, Default)]
pub(crate) struct MotionField {
    units: Vec<ProjectedMv>,
    columns: usize,
    rows: usize,
}

impl MotionField {
    /// Row `y8` of the field (8x8 units; [`ProjectedMv::offset`] 0 where
    /// nothing projected); the row's superblock row must be the one last
    /// projected.
    pub(crate) fn row(&self, y8: usize) -> &[ProjectedMv] {
        let start = (y8 % FIELD_ROWS) * self.columns;
        &self.units[start..start + self.columns]
    }
}

/// The current frame's motion field and the references it projects from:
/// [`TemporalField::project_rows`] builds each superblock row before it
/// is decoded.
pub(crate) struct TemporalField<'a> {
    field: &'a mut MotionField,
    projections: [Option<Projection<'a>>; MFMV_STACK_SIZE],
}

impl<'a> TemporalField<'a> {
    /// The field of a frame of `mi_cols` x `mi_rows` 4x4 units with order
    /// hint `order_hint`, projected from its references `sources` (by
    /// reference name) as §7.9.1 orders them, in the buffer `field`.
    pub(crate) fn new(
        field: &'a mut MotionField,
        seq: &Sequence,
        (mi_cols, mi_rows): (usize, usize),
        order_hint: u32,
        sources: &[Option<ProjectionSource<'a>>; TOTAL_REFS],
    ) -> Self {
        field.columns = mi_cols >> 1;
        field.rows = mi_rows >> 1;
        field.units.clear();
        field
            .units
            .resize(FIELD_ROWS * field.columns, ProjectedMv::NONE);
        let mut projections: [Option<Projection<'a>>; MFMV_STACK_SIZE] = Default::default();
        let mut count = 0;
        let mut add = |source: &ProjectionSource<'a>, backward: bool| {
            // A reference with no vectors to give does not take a place.
            let usable = !matches!(
                source.header.frame_type,
                FrameType::Key | FrameType::IntraOnly
            ) && (source.header.size.mi_cols(), source.header.size.mi_rows())
                == (mi_cols, mi_rows);
            if usable && count < MFMV_STACK_SIZE {
                projections[count] = Projection::new(seq, order_hint, source, backward, field);
                count += 1;
            }
            usable
        };
        let hint = |name: usize| sources[name].as_ref().map_or(0, |s| s.header.order_hint);
        let ahead = |name: usize| relative_distance(seq, hint(name), order_hint) > 0;
        let mut stamp = MFMV_STACK_SIZE as i32 - 1;
        if let Some(last) = &sources[LAST_FRAME] {
            // LAST is an overlay of the frame GOLDEN shows when its
            // ALTREF is GOLDEN.
            if last.header.ref_order_hints[ALTREF_FRAME] != hint(GOLDEN_FRAME) {
                add(last, true);
            }
            stamp -= 1;
        }
        for name in [BWDREF_FRAME, ALTREF2_FRAME, ALTREF_FRAME] {
            if ahead(name)
                && (name != ALTREF_FRAME || stamp >= 0)
                && let Some(source) = &sources[name]
                && add(source, false)
            {
                stamp -= 1;
            }
        }
        if stamp >= 0
            && let Some(source) = &sources[LAST2_FRAME]
        {
            add(source, true);
        }
        Self { field, projections }
    }

    /// The field, for reading.
    pub(crate) fn field(&self) -> &MotionField {
        self.field
    }

    /// Build the field's rows `rows8` of units, columns `columns8` (a
    /// tile's part of one superblock row), from the references' rows
    /// `rows8` (§7.9.2; dav1d `load_tmvs`).
    pub(crate) fn project_rows(&mut self, columns8: Range<usize>, rows8: Range<usize>) {
        let field = &mut *self.field;
        let rows8 = rows8.start..rows8.end.min(field.rows);
        let columns8 = columns8.start..columns8.end.min(field.columns);
        for y8 in rows8.clone() {
            let start = (y8 % FIELD_ROWS) * field.columns;
            field.units[start + columns8.start..start + columns8.end].fill(ProjectedMv::NONE);
        }
        // Units this far outside the columns may project into them.
        let sources = columns8.start.saturating_sub(MAX_OFFSET_COLUMNS)
            ..(columns8.end + MAX_OFFSET_COLUMNS).min(field.columns);
        for projection in self.projections.iter().flatten() {
            for y8 in rows8.clone() {
                let band = y8 & !(BAND_UNITS - 1);
                let targets = band.max(rows8.start)..(band + BAND_UNITS).min(rows8.end);
                projection.project_row(field, y8, &sources, &targets, &columns8);
            }
        }
    }
}

impl<'a> Projection<'a> {
    /// The projection of `source`'s kept vectors onto the current frame of
    /// order hint `order_hint`; none when it is too far away to project.
    fn new(
        seq: &Sequence,
        order_hint: u32,
        source: &ProjectionSource<'a>,
        backward: bool,
        field: &MotionField,
    ) -> Option<Self> {
        let header = source.header;
        let distance = relative_distance(seq, header.order_hint, order_hint);
        let to_current = if backward { -distance } else { distance };
        let sized = (source.motion.columns, source.motion.rows) == (field.columns, field.rows);
        (to_current.abs() <= MAX_FRAME_DISTANCE && sized).then(|| Self {
            motion: source.motion,
            to_current,
            backward,
            spans: std::array::from_fn(|name| {
                let span = relative_distance(seq, header.order_hint, header.ref_order_hints[name]);
                if name > INTRA_FRAME as usize && (1..=MAX_FRAME_DISTANCE).contains(&span) {
                    span
                } else {
                    0
                }
            }),
        })
    }

    /// Project row `y8` of the kept vectors, units `sources`, into the
    /// field's rows `targets` and columns `columns`: each unit lands at
    /// most `MAX_OFFSET_COLUMNS` outside its band of columns. A run of
    /// equal units is projected once (they move alike).
    fn project_row(
        &self,
        field: &mut MotionField,
        y8: usize,
        sources: &Range<usize>,
        targets: &Range<usize>,
        columns: &Range<usize>,
    ) {
        let saved = self.motion.row(y8);
        let mut x8 = sources.start;
        while x8 < sources.end {
            let unit = saved[x8];
            let run_end = saved[x8 + 1..sources.end]
                .iter()
                .position(|&next| next != unit)
                .map_or(sources.end, |n| x8 + 1 + n);
            let span = self.spans[unit.reference() as usize];
            if span != 0 {
                let offset = project_mv(unit.mv(), self.to_current, span);
                let target_y = y8.checked_add_signed(self.units(offset.row));
                if let Some(target_y) = target_y.filter(|y| targets.contains(y)) {
                    let row = (target_y % FIELD_ROWS) * field.columns;
                    let shift = self.units(offset.col);
                    let projected = ProjectedMv {
                        mv: unit.mv(),
                        offset: span,
                    };
                    // The run's units of one band of columns land on one
                    // span of targets, clipped to that band's reach: one
                    // fill per band rather than a test per unit.
                    let mut x = x8;
                    while x < run_end {
                        let band = x & !(BAND_UNITS - 1);
                        let band_end = (band + BAND_UNITS).min(run_end);
                        let reach = band.saturating_sub(MAX_OFFSET_COLUMNS).max(columns.start)
                            ..(band + BAND_UNITS + MAX_OFFSET_COLUMNS).min(columns.end);
                        let first = (x as isize + shift).max(reach.start as isize);
                        let end = (band_end as isize + shift).min(reach.end as isize);
                        if first < end {
                            field.units[row + first as usize..row + end as usize].fill(projected);
                        }
                        x = band_end;
                    }
                }
            }
            x8 = run_end;
        }
    }

    /// Whole units a projected vector component moves (rounded toward
    /// zero), in the direction of the current frame.
    fn units(&self, v: i16) -> isize {
        let units = (i32::from(v).abs() >> UNIT_SHIFT) as isize;
        if (v < 0) == self.backward {
            units
        } else {
            -units
        }
    }
}

/// `get_mv_projection(mv, numerator, denominator)` (§7.9.3) for distances
/// already within range (the spec's clamps): `numerator` within
/// [`MAX_PROJECTION_DISTANCE`] and `denominator` 1 to it, as the field and
/// the MV stack keep them, so a temporal candidate pays no clamp.
#[inline]
pub(crate) fn project_mv(mv: Av1Mv, numerator: i32, denominator: i32) -> Av1Mv {
    debug_assert!(numerator.abs() <= MAX_FRAME_DISTANCE);
    debug_assert!((1..=MAX_FRAME_DISTANCE).contains(&denominator));
    let fraction = numerator * DIV_MULT[denominator as usize & (DIV_MULT.len() - 1)];
    let scale = |v: i16| {
        // `Round2Signed(product, 14)` without a branch: for a negative
        // product, rounding the magnitude half up equals `(product + half
        // - 1) >> 14` (as dav1d's `mv_projection`).
        let product = i32::from(v) * fraction;
        let half = 1 << (DIV_MULT_BITS - 1);
        let rounded = (product + half + (product >> 31)) >> DIV_MULT_BITS;
        rounded.clamp(-PROJECTED_MV_MAX, PROJECTED_MV_MAX) as i16
    };
    Av1Mv {
        row: scale(mv.row),
        col: scale(mv.col),
    }
}
