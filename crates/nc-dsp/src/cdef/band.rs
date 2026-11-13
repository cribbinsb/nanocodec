//! In-place CDEF of a picture, one band of rows at a time (AV1 §7.15, AVM
//! `av2_cdef_frame` for 8-bit 4:2:0): the picture driver shared by the AV1
//! decoder and any codec with AV1's 8x8 units, 64x64 strength sets and
//! luma-derived directions (AV2).
//!
//! The structure is dav1d's `cdef_brow` (`src/cdef_apply_tmpl.c`,
//! BSD-2-Clause, Copyright © 2018 `VideoLAN` and dav1d authors, Copyright ©
//! 2018 Two Orioles, LLC; see `THIRD_PARTY.md`), reimplemented:
//!
//! - the picture is filtered in place, by bands of rows from the top; the
//!   unfiltered rows a band reads live in per-plane line buffers
//!   ([`CdefLines`]) that hold the band plus [`CDEF_PADDING`] rows on each
//!   side, not a copy of the picture (dav1d keeps two saved rows above and
//!   two saved columns to the left and reads the rest from the picture; one
//!   line buffer per plane gives every kernel a single strided 8-bit source
//!   instead, for about one copied byte per sample);
//! - every 8x8 unit is finished before the next: its direction is searched
//!   on the unfiltered luma, then its luma block and both chroma blocks are
//!   filtered while the samples are in cache;
//! - the caller lists only the units to filter: filter blocks without a
//!   strength set and units whose blocks all skipped their residual are
//!   never visited, and a unit whose strengths come to zero is not filtered;
//! - the direction is searched only when a primary strength uses it; the
//!   luma primary strength is scaled by the unit's directional contrast
//!   ([`cdef_adjust_strength`]), and a plane without a primary strength
//!   filters in direction 0, as the specification's `CdefDirection` use
//!   implies.
//!
//! Where the ISA has a two-block 4x4 kernel, a unit's U and V blocks share
//! one call: same position, strengths, damping and direction.

use super::{CDEF_PADDING, CdefArgs, CdefSource};
use crate::Dsp;
use core::ops::Range;
use nc_common::{Picture, PlaneIndex, PlaneViewMut};

/// Side of a luma CDEF unit, and of its 4:2:0 chroma blocks.
const LUMA_SIZE: usize = 8;
const CHROMA_SIZE: usize = 4;
/// Largest primary strength.
const MAX_PRIMARY: u8 = 15;

/// One strength set of a frame (`cdef_y_*_strength[i]`,
/// `cdef_uv_*_strength[i]`), as the filter uses them: primary strengths
/// 0..=15, secondary strengths 0, 1, 2 or 4.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CdefStrengths {
    /// Luma primary strength, before [`cdef_adjust_strength`].
    pub luma_primary: u8,
    /// Luma secondary strength.
    pub luma_secondary: u8,
    /// Chroma primary strength.
    pub chroma_primary: u8,
    /// Chroma secondary strength.
    pub chroma_secondary: u8,
}

impl CdefStrengths {
    /// Whether this set changes any sample.
    #[must_use]
    pub fn filters(self) -> bool {
        self.filters_luma() || self.filters_chroma()
    }

    fn filters_luma(self) -> bool {
        self.luma_primary | self.luma_secondary != 0
    }

    fn filters_chroma(self) -> bool {
        self.chroma_primary | self.chroma_secondary != 0
    }

    /// Whether a unit with this set needs its direction.
    fn needs_direction(self) -> bool {
        self.luma_primary | self.chroma_primary != 0
    }
}

/// An 8x8 luma unit to filter: its top-left luma sample and the index of
/// its strength set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdefUnit {
    /// Luma column of the unit's top-left sample.
    pub x: usize,
    /// Luma row of the unit's top-left sample.
    pub y: usize,
    /// Index of the unit's strength set.
    pub set: u8,
}

/// The luma primary strength scaled by the unit's directional contrast
/// `var` (libaom/dav1d `adjust_strength`, AV1 §7.15.1): zero for a flat
/// unit, otherwise `strength * (4 + min(floor(log2(var >> 6)), 12))`
/// sixteenths, rounded.
#[must_use]
pub fn cdef_adjust_strength(strength: u8, var: i32) -> u8 {
    /// Contrast below `1 << VAR_SHIFT` is the lowest class.
    const VAR_SHIFT: u32 = 6;
    /// Highest contrast class.
    const MAX_CLASS: u32 = 12;
    /// Scale of the lowest class, in sixteenths.
    const BASE_SCALE: u32 = 4;
    if var == 0 {
        return 0;
    }
    let class = match (var as u32) >> VAR_SHIFT {
        0 => 0,
        high => high.ilog2().min(MAX_CLASS),
    };
    ((u32::from(strength) * (BASE_SCALE + class) + 8) >> 4) as u8
}

/// The unfiltered rows of each plane that the band being filtered reads:
/// per-plane line buffers, reused across bands and pictures.
#[derive(Debug, Clone, Default)]
pub struct CdefLines {
    planes: [CdefSource; 3],
}

impl CdefLines {
    /// Line buffers with nothing copied yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin filtering `picture` from its top band.
    pub fn start(&mut self, picture: &Picture) {
        for (source, index) in self.planes.iter_mut().zip(PlaneIndex::ALL) {
            let plane = picture.plane(index);
            source.start(plane.width(), plane.height());
        }
    }
}

/// Most strength sets a frame codes (AV1 `cdef_bits` up to 3, AV2
/// `nb_cdef_strengths` up to 8).
pub const CDEF_MAX_SETS: usize = 8;
/// Primary strengths 0..=15.
const PRIMARY_STRENGTHS: usize = MAX_PRIMARY as usize + 1;

/// A frame's strength sets and damping, resolved once to the kernels'
/// arguments: per set, the chroma arguments and the luma arguments for
/// every adjusted primary strength (all for direction 0).
#[derive(Debug, Clone)]
pub struct CdefSets {
    strengths: [CdefStrengths; CDEF_MAX_SETS],
    luma: [[CdefArgs; PRIMARY_STRENGTHS]; CDEF_MAX_SETS],
    chroma: [CdefArgs; CDEF_MAX_SETS],
    count: usize,
}

impl CdefSets {
    /// Resolve `sets` with the frame's `damping` (`CdefDamping`; chroma
    /// uses one less).
    ///
    /// # Panics
    ///
    /// If there are more than [`CDEF_MAX_SETS`] sets or a strength is out
    /// of range.
    #[must_use]
    pub fn new(sets: &[CdefStrengths], damping: u8) -> Self {
        assert!(sets.len() <= CDEF_MAX_SETS, "too many CDEF strength sets");
        let unused = CdefArgs::new(0, 0, 0, damping);
        let mut resolved = Self {
            strengths: [CdefStrengths::default(); CDEF_MAX_SETS],
            luma: [[unused; PRIMARY_STRENGTHS]; CDEF_MAX_SETS],
            chroma: [unused; CDEF_MAX_SETS],
            count: sets.len(),
        };
        for (i, set) in sets.iter().enumerate() {
            let secondary = |s: u8| matches!(s, 0 | 1 | 2 | 4);
            assert!(
                set.luma_primary <= MAX_PRIMARY
                    && set.chroma_primary <= MAX_PRIMARY
                    && secondary(set.luma_secondary)
                    && secondary(set.chroma_secondary),
                "CDEF strength out of range"
            );
            resolved.strengths[i] = *set;
            for (primary, args) in (0..).zip(&mut resolved.luma[i]) {
                *args = CdefArgs::new(primary, set.luma_secondary, 0, damping);
            }
            resolved.chroma[i] = CdefArgs::new(
                set.chroma_primary,
                set.chroma_secondary,
                0,
                damping.saturating_sub(1),
            );
        }
        resolved
    }

    /// The sets, in index order.
    fn sets(&self) -> &[CdefStrengths] {
        &self.strengths[..self.count]
    }

    /// Whether any set changes a sample.
    #[must_use]
    pub fn filters(&self) -> bool {
        self.sets().iter().any(|set| set.filters())
    }
}

impl Dsp {
    /// CDEF of the luma rows `rows` of `picture`, in place: every unit of
    /// `units` (in raster order, each with its top row in `rows`) with its
    /// strength set from `sets`. Bands go from the top of the picture down,
    /// each starting where the previous ended; [`CdefLines::start`] begins
    /// a picture. `rows` must be a multiple of 8 rows except at the bottom.
    ///
    /// # Panics
    ///
    /// If a unit lies outside the picture or outside `rows`, a set index is
    /// out of range, or the bands are out of order.
    pub fn cdef_filter_band(
        &self,
        lines: &mut CdefLines,
        picture: &mut Picture,
        rows: Range<usize>,
        units: &[CdefUnit],
        sets: &CdefSets,
    ) {
        let filters_chroma = sets.sets().iter().any(|set| set.filters_chroma());
        let [luma_plane, u_plane, v_plane] = picture.planes_mut();
        let [luma_lines, u_lines, v_lines] = &mut lines.planes;
        // Directions read the luma lines too, so they are copied whenever a
        // set filters anything.
        if sets.filters() {
            luma_lines.drop_rows_before(rows.start.saturating_sub(CDEF_PADDING));
            luma_lines.fill_rows_to(&luma_plane.view(), rows.end + CDEF_PADDING);
        }
        if filters_chroma {
            let chroma_rows = (rows.start / 2, rows.end.div_ceil(2));
            for (source, plane) in [(&mut *u_lines, &*u_plane), (&mut *v_lines, &*v_plane)] {
                source.drop_rows_before(chroma_rows.0.saturating_sub(CDEF_PADDING));
                source.fill_rows_to(&plane.view(), chroma_rows.1 + CDEF_PADDING);
            }
        }
        let mut luma = luma_plane.view_mut();
        let mut chroma = [u_plane.view_mut(), v_plane.view_mut()];
        // The U and V blocks of a unit go through one kernel call with one
        // stride for both line buffers and one for both planes.
        assert!(
            u_lines.width() == v_lines.width()
                && chroma[0].stride() == chroma[1].stride()
                && chroma
                    .iter()
                    .all(|p| p.width() >= u_lines.width() && p.height() >= u_lines.height()),
            "CDEF chroma planes differ"
        );
        let (width, height) = (luma_lines.width(), luma_lines.height());
        for unit in units {
            assert!(
                rows.contains(&unit.y)
                    && unit.x + LUMA_SIZE <= width
                    && unit.y + LUMA_SIZE <= height,
                "CDEF unit outside the band being filtered"
            );
            let index = usize::from(unit.set);
            let set = sets.sets()[index];
            let (dir, var) = if set.needs_direction() {
                self.cdef_find_dir_in(luma_lines, (unit.x, unit.y))
            } else {
                (0, 0)
            };
            if set.filters_luma() {
                let primary = if set.luma_primary == 0 {
                    0
                } else {
                    cdef_adjust_strength(set.luma_primary, var)
                };
                if primary | set.luma_secondary != 0 {
                    let dir = if set.luma_primary == 0 { 0 } else { dir };
                    let args = sets.luma[index][usize::from(primary)].with_dir(dir);
                    self.cdef_filter_block_at(
                        LUMA_SIZE,
                        luma_lines,
                        (unit.x, unit.y),
                        &args,
                        &mut luma,
                    );
                }
            }
            if set.filters_chroma() {
                let dir = if set.chroma_primary == 0 { 0 } else { dir };
                self.cdef_filter_chroma(
                    &sets.chroma[index].with_dir(dir),
                    [u_lines, v_lines],
                    (unit.x / 2, unit.y / 2),
                    &mut chroma,
                );
            }
        }
    }

    /// The direction and contrast of the 8x8 luma unit at (`x`, `y`) from
    /// its unfiltered rows in `lines`.
    fn cdef_find_dir_in(&self, lines: &CdefSource, (x, y): (usize, usize)) -> (u8, i32) {
        assert!(
            x + LUMA_SIZE <= lines.width() && y + LUMA_SIZE <= lines.filled_rows(),
            "CDEF unit outside the picture"
        );
        let start = lines.index(x, y);
        // SAFETY: the table matches this CPU (built after `is_supported`);
        // the kernel indexes its slice, which starts at the unit and holds
        // its eight rows (checked above).
        unsafe { (self.cdef_kernels().find_dir)(&lines.samples()[start..], lines.width()) }
    }

    /// Filter the U and V blocks at (`x`, `y`) of `planes`, two at a time
    /// where the ISA can and the blocks are interior.
    fn cdef_filter_chroma(
        &self,
        args: &CdefArgs,
        lines: [&CdefSource; 2],
        (x, y): (usize, usize),
        planes: &mut [PlaneViewMut<'_>; 2],
    ) {
        let pair = self.cdef_kernels().block4_interior_pair;
        let source = lines[0];
        match pair {
            Some(pair) if is_interior(source, (x, y), CHROMA_SIZE) => {
                let [u, v] = planes;
                let stride = u.stride();
                // The block's footprint in each plane (the caller checked
                // that both planes share the stride).
                let block = y * stride + x..(y + CHROMA_SIZE - 1) * stride + x + CHROMA_SIZE;
                let dst = [
                    u.data_mut()[block.clone()].as_mut_ptr(),
                    v.data_mut()[block].as_mut_ptr(),
                ];
                // SAFETY: the table matches this CPU; both blocks are
                // interior, and the kernel checks each source window
                // against its line buffer (`index` panics on rows already
                // dropped); each destination starts a 4x4 block of stride
                // `stride` inside its own plane's slice (sliced above), and
                // the two planes are distinct buffers.
                unsafe {
                    pair(
                        [lines[0].samples(), lines[1].samples()],
                        source.width(),
                        [source.index(x, y), lines[1].index(x, y)],
                        args,
                        [args.dir; 2],
                        dst,
                        stride,
                    );
                }
            }
            _ => {
                for (source, plane) in lines.into_iter().zip(planes.iter_mut()) {
                    self.cdef_filter_block_at(CHROMA_SIZE, source, (x, y), args, plane);
                }
            }
        }
    }
}

/// Whether every tap of the `size` block at (`x`, `y`) lies inside the
/// plane of `src`.
pub(crate) fn is_interior(src: &CdefSource, (x, y): (usize, usize), size: usize) -> bool {
    x >= CDEF_PADDING
        && y >= CDEF_PADDING
        && x + size + CDEF_PADDING <= src.width()
        && y + size + CDEF_PADDING <= src.height()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjusted_strengths_match_the_formula() {
        assert_eq!(cdef_adjust_strength(10, 0), 0);
        // Class 0 below 64: strength * 4 / 16.
        assert_eq!(cdef_adjust_strength(10, 63), (10 * 4 + 8) >> 4);
        // 64..128 is class 0 too (log2 of 1); 1 << 20 caps at 12.
        assert_eq!(cdef_adjust_strength(15, 100), (15 * 4 + 8) >> 4);
        assert_eq!(cdef_adjust_strength(15, 1 << 20), (15 * 16 + 8) >> 4);
        assert_eq!(cdef_adjust_strength(7, 1000), (7 * (4 + 3) + 8) >> 4);
    }
}
