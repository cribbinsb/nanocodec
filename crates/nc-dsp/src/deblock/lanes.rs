//! The AV1 loop filters (AV1 §7.14.6) in dav1d's lane layout: one byte
//! lane per line across the edge and a whole vector of lines (32 on AVX2)
//! per group, each 4-line unit with its own thresholds and filter width, so
//! the units of one edge line (a column of vertical edges or a row of
//! horizontal ones) share vectors whatever their transform sizes. A group
//! branches only on its widest unit; narrower units are masked per lane, as
//! dav1d 1.5.4 `src/x86/loopfilter_avx2.asm` (VideoLAN/dav1d authors and Two
//! Orioles; BSD-2-Clause, see `THIRD_PARTY.md`) does. Its techniques carried
//! here: the lane layout and transposes, thresholds and widths unpacked from
//! per-unit entries, running smoothing sums updated by byte-pair
//! multiply-adds, and `mulhrs` rounding.
//!
//! One family with the packed eight-line kernels (`av1_packed`, and VP9's
//! `vp9_loop_filter::packed`): the masks and narrow filter are
//! [`crate::lpf_bytes`]'s, and the smoothing filters are the same box sums
//! (`av1_packed` module docs); luma and chroma differ only in their
//! smoothing widths (`kernel::LanePolicy`: 8 and 14 taps, or 6). VP9 stays
//! on its packed kernel: run through this layout (VP9's 15-tap wide filter,
//! `flat2` over p4..p7, and libvpx `filter4`'s mask on p3..q3 as a policy)
//! the VP9 decoder's filter kernels cost the same (2.13 G instructions on
//! `vp90-2-02-size-lf-1920x1080`, callgrind), since its edges already come
//! as long 8-line runs, while queueing its edges per superblock pass added
//! 0.41 G: whole decode +2.3 %, so it was not kept (2026-11-19).
//!
//! Callers fill [`LoopFilterLanes`]: an entry per 4-line unit of every edge
//! line of one pass, and the kernel clears each entry it filters, so the
//! table never needs a full reset. The AV1 decoder writes entries as its
//! edge walk finds them ([`LoopFilterLanes::set_av1`], after
//! `Dsp::av1_lane_pass`); other callers hand over run lists
//! (`Dsp::av1_loop_filter_runs`, converted here).
//!
//! ISA tiers: AVX2 only; the other tiers keep their runs kernels
//! (`Dsp::av1_loop_filter_runs` takes the runs kernel when the table has no
//! lane kernel), because the lane layout measured slower there (2026-11-19,
//! frame MD5s identical in every variant; `docs/handoff/av1-decoder.md`
//! "Lane layout on the other ISA tiers"):
//!
//! - AVX-512 (ICL grouped kernels kept). Whole AV1 decode instructions
//!   (Ice Lake `perf stat`) on `aom720_good4_1000k` / `svt720_p8_1000k`:
//!   grouped 1186.8 M / 2952.5 M; these kernels at 32 lines with AVX-512
//!   `vpternlogd` selects, mask-register compares and GFNI shifts
//!   1179.1 M / 2951.3 M (−0.64 % / −0.04 %), but on Zen 5 (c8a, the
//!   published machine) 0.3–1.0 % slower in wall time (two sessions, 9 and
//!   12 paired rounds); at 64 lines a vector 1227.9 M (+3.5 %: the
//!   16-vector tiles stayed loops over stack arrays, and a group widens to
//!   its widest unit over 64 lines).
//! - NEON (Jetson, 16 lines a vector, dav1d `loopfilter.S`'s `uabd`,
//!   `bsl`, widening adds and rounding narrows): +0.40 % / +0.07 %
//!   instructions, +1.47 % / +0.51 % cycles. The kernel itself costs what
//!   the sixteen-line runs kernel does (9.9 % against 9.8 % of samples);
//!   filling the table adds 1.1 %. With the decoder writing entries
//!   directly (2026-11-20): −0.31 % / −0.07 % instructions but still
//!   +1.52 % / +0.80 % cycles. Not landed: branch
//!   `av1-lpf-neon-experiment`.
//! - SIMD128: not tried; its runs kernels are sixteen-line like NEON's.

#[cfg(target_arch = "x86_64")]
pub(crate) mod kernel;

/// Lines of one entry (a 4x4 transform edge).
pub const UNIT_LINES: usize = 4;

/// The narrow 4-tap filter only.
pub(crate) const WIDTH_NARROW: u8 = 1;
/// Up to the policy's middle smoothing filter (6 or 8 taps).
pub(crate) const WIDTH_MID: u8 = 2;
/// Up to the policy's wide smoothing filter (14 or 16 taps).
pub(crate) const WIDTH_WIDE: u8 = 3;

/// One 4-line unit of an edge line: the level's thresholds (AV1 §7.14.4)
/// and the widest filter allowed, in dav1d's
/// 4-byte entry so one vector load covers a group's units.
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LaneEdge {
    /// Largest step inside either side (`limit`).
    pub limit: u8,
    /// Largest weighted step across the edge (`blimit`).
    pub blimit: u8,
    /// High edge variance threshold (`thresh`).
    pub thresh: u8,
    /// 0 (no edge in the unit) ..= [`WIDTH_WIDE`].
    pub width: u8,
}

/// Lines of a group of the widest lane type: planes with a narrower
/// stride take the runs kernel (a group's rows would overlap).
const GROUP_LINES: usize = GROUP_UNITS * UNIT_LINES;

/// An ISA's AV1 passes: `(plane, stride, vertical, chroma, lanes)`.
pub(crate) type Av1LanesFn = unsafe fn(&mut [u8], usize, bool, bool, &mut LoopFilterLanes);

/// Units per group of the widest lane type (AVX2: 32 lines); every
/// edge line's entries are padded to a multiple of it.
const GROUP_UNITS: usize = 8;

/// The entries of one filter pass over a plane (all vertical or all
/// horizontal edges): [`LaneEdge`] per 4-line unit of each edge line,
/// lines 4 samples apart. Kept between passes and frames: the kernel
/// clears what it filters.
#[derive(Debug, Default)]
pub struct LoopFilterLanes {
    entries: Vec<LaneEdge>,
    /// Units per edge line (padded to [`GROUP_UNITS`]).
    units: usize,
    /// Edge lines holding entries, and each one's range of groups.
    touched: Vec<u32>,
    groups: Vec<(u32, u32)>,
    /// Edge lines and units along them of the current pass.
    lines: usize,
    along_units: usize,
    /// The pass is chroma's (6-tap middle filter).
    pub(crate) chroma: bool,
}

/// A run's thresholds and widest filter as an entry.
fn lane_edge(f: super::av1::Av1EdgeFilter) -> LaneEdge {
    use super::av1::Av1FilterLength;
    let width = match f.length {
        Av1FilterLength::Four => WIDTH_NARROW,
        Av1FilterLength::Six | Av1FilterLength::Eight => WIDTH_MID,
        Av1FilterLength::Fourteen => WIDTH_WIDE,
    };
    LaneEdge {
        limit: f.limit,
        blimit: f.blimit,
        thresh: f.thresh,
        width,
    }
}

impl LoopFilterLanes {
    /// Size for `lines` edge lines of `along` samples.
    fn reset(&mut self, lines: usize, along: usize) {
        // Between passes every entry is zero (the kernels clear what they
        // filter), so the table is reinterpreted, never cleared.
        self.lines = lines;
        self.along_units = along.div_ceil(UNIT_LINES);
        self.units = self.along_units.next_multiple_of(GROUP_UNITS);
        if self.entries.len() < lines * self.units {
            self.entries.resize(lines * self.units, LaneEdge::default());
        }
        if self.groups.len() < lines {
            self.groups.resize(lines, (u32::MAX, 0));
        }
    }

    /// Set the units `first..first + count` of edge line `line`.
    #[inline]
    fn set(&mut self, line: usize, first: usize, count: usize, edge: LaneEdge) {
        let start = line * self.units + first;
        self.entries[start..start + count].fill(edge);
        let (lo, hi) = &mut self.groups[line];
        if *lo == u32::MAX {
            self.touched.push(line as u32);
        }
        *lo = (*lo).min((first / GROUP_UNITS) as u32);
        *hi = (*hi).max((first + count).div_ceil(GROUP_UNITS) as u32);
    }

    /// Start an AV1 pass over a `width` x `height` plane of `stride` whose
    /// entries the caller sets ([`LoopFilterLanes::set_av1`]); `false`,
    /// leaving the table unchanged, for a plane narrower than a group (its
    /// rows would overlap) or not a whole number of units (a run's last
    /// unit would leave it).
    pub(crate) fn start_av1(
        &mut self,
        (width, height, stride): (usize, usize, usize),
        vertical: bool,
        chroma: bool,
    ) -> bool {
        if stride < GROUP_LINES
            || !width.is_multiple_of(UNIT_LINES)
            || !height.is_multiple_of(UNIT_LINES)
        {
            return false;
        }
        let (lines, along) = if vertical {
            (width / UNIT_LINES, height)
        } else {
            (height / UNIT_LINES, width)
        };
        self.reset(lines, along);
        self.chroma = chroma;
        true
    }

    /// Filter units `first..first + count` of edge line `line` (units: 4
    /// samples across and along) with `filter`, in a pass begun by
    /// `Dsp::av1_lane_pass`. A pass's filters are chroma's 4 and 6 taps or
    /// luma's 4, 8 and 14 (as AV1 §7.14.3 chooses them per plane).
    ///
    /// # Panics
    ///
    /// If the units are not inside the plane past its first edge line.
    #[inline]
    pub fn set_av1(
        &mut self,
        line: usize,
        first: usize,
        count: usize,
        filter: super::av1::Av1EdgeFilter,
    ) {
        assert!(
            (1..self.lines).contains(&line) && first + count <= self.along_units,
            "AV1 lane entry outside the plane"
        );
        self.set(line, first, count, lane_edge(filter));
    }

    /// The AV1 runs of one pass: `Some(chroma)`, whether they are
    /// chroma's (6-tap), or `None`, leaving the table unchanged, for a
    /// plane narrower than a group (its rows would overlap) or runs it
    /// cannot hold: off the 4x4 grid or outside the plane (the runs
    /// kernels take partial units and panic on the rest), or one pass
    /// mixing chroma's and luma's longer filters (no AV1 plane does).
    pub(crate) fn fill_av1(
        &mut self,
        plane: &super::runs::Plane,
        runs: &[super::runs::EdgeRun<super::av1::Av1EdgeFilter>],
    ) -> Option<bool> {
        use super::av1::Av1FilterLength;
        // A group's rows (horizontal edges) or tiles must not share bytes.
        if plane.stride < GROUP_LINES {
            return None;
        }
        let (lines, along) = if plane.vertical {
            (plane.width.div_ceil(UNIT_LINES), plane.height)
        } else {
            (plane.height.div_ceil(UNIT_LINES), plane.width)
        };
        let position = |run: &super::runs::EdgeRun<_>| {
            if plane.vertical {
                (run.x, run.y)
            } else {
                (run.y, run.x)
            }
        };
        let (mut six, mut longer) = (false, false);
        for run in runs.iter().filter(|run| run.lines != 0) {
            let (across, first) = position(run);
            let on_grid = across.is_multiple_of(UNIT_LINES)
                && first.is_multiple_of(UNIT_LINES)
                && run.lines.is_multiple_of(UNIT_LINES);
            let inside = across > 0
                && across / UNIT_LINES < lines
                && first.checked_add(run.lines).is_some_and(|end| end <= along);
            if !on_grid || !inside {
                return None;
            }
            six |= run.filter.length == Av1FilterLength::Six;
            longer |= matches!(
                run.filter.length,
                Av1FilterLength::Eight | Av1FilterLength::Fourteen
            );
        }
        if six && longer {
            return None;
        }
        self.reset(lines, along);
        self.chroma = six;
        for run in runs.iter().filter(|run| run.lines != 0) {
            let (across, first) = position(run);
            let edge = lane_edge(run.filter);
            self.set(
                across / UNIT_LINES,
                first / UNIT_LINES,
                run.lines / UNIT_LINES,
                edge,
            );
        }
        Some(six)
    }
}
