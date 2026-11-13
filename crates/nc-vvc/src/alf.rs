//! H.266 §8.8.5 adaptive loop filter (ALF), luma: each 4x4 block is
//! classified by its local Laplacian activity and direction (§8.8.5.3) into
//! one of 25 classes and filtered with a 7x7 diamond (12 symmetric taps,
//! §8.8.5.2) from its CTB's filter set, transposed by the block's direction.
//! Every CTB chooses none, one of the 16 fixed sets, or the filters the
//! slice's APS signals (`alf_aps.rs`). At the ALF virtual boundary four luma
//! rows above each CTU row's bottom, rows across it are not read (symmetric
//! padding).
//!
//! The filter rows, the classification Laplacians and the statistics' Gram
//! matrices are `nc-dsp` kernels; this module picks rows (padding, virtual
//! boundaries), classes and coefficients.
//!
//! Tables and arithmetic follow `VVenC` `AdaptiveLoopFilter` (BSD-3-Clause-Clear;
//! `THIRD_PARTY.md`).
mod clipped;
pub(crate) use clipped::ClippedFilters;

use crate::alf_aps::{CLASSES, Gram, SignalledFilters, Units};
use crate::{coeff::Bins, contexts::Contexts};
use nc_common::{Picture, PlaneIndex, PlaneView, decode::Result};
use nc_dsp::{
    Dsp, Moments, VVC_ALF_BLOCK as BLOCK, VVC_ALF_DOT_LANES, VVC_ALF_GRAM_ALIGN, VVC_ALF_REACH,
    VVC_ALF_SAMPLE_LANES, VVC_ALF_SAMPLE_STEP, VVC_ALF_SAMPLES, VVC_ALF_TAPS as TAPS,
    VVC_ALF_TRANSPOSES, vvc_alf_gram_index,
};

/// Fixed filter sets a CTB may choose (`alf_luma_fixed_filter_idx`, 4 bypass bits).
pub(crate) const FIXED_SETS: usize = 16;
const FIXED_SET_BITS: u32 = 4;
const FIXED_FILTER_COUNT: usize = 64;
/// The virtual boundary lies this many luma rows above each CTU row's bottom.
const VIRTUAL_BOUNDARY_ABOVE: usize = 4;
/// Coefficient precision (`AlfNumBits` 8, sum scaled by 2^7) and, on the two
/// rows next to the virtual boundary, three more bits of attenuation.
const SHIFT: u32 = 7;
const BOUNDARY_SHIFT: u32 = 10;

#[rustfmt::skip]
const FIXED_FILTERS: [[i8; TAPS]; FIXED_FILTER_COUNT] = [
    [0, 0, 2, -3, 1, -4, 1, 7, -1, 1, -1, 5],
    [0, 0, 0, 0, 0, -1, 0, 1, 0, 0, -1, 2],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
    [2, 2, -7, -3, 0, -5, 13, 22, 12, -3, -3, 17],
    [-1, 0, 6, -8, 1, -5, 1, 23, 0, 2, -5, 10],
    [0, 0, -1, -1, 0, -1, 2, 1, 0, 0, -1, 4],
    [0, 0, 3, -11, 1, 0, -1, 35, 5, 2, -9, 9],
    [0, 0, 8, -8, -2, -7, 4, 4, 2, 1, -1, 25],
    [0, 0, 1, -1, 0, -3, 1, 3, -1, 1, -1, 3],
    [0, 0, 3, -3, 0, -6, 5, -1, 2, 1, -4, 21],
    [-7, 1, 5, 4, -3, 5, 11, 13, 12, -8, 11, 12],
    [-5, -3, 6, -2, -3, 8, 14, 15, 2, -7, 11, 16],
    [2, -1, -6, -5, -2, -2, 20, 14, -4, 0, -3, 25],
    [3, 1, -8, -4, 0, -8, 22, 5, -3, 2, -10, 29],
    [2, 1, -7, -1, 2, -11, 23, -5, 0, 2, -10, 29],
    [-6, -3, 8, 9, -4, 8, 9, 7, 14, -2, 8, 9],
    [2, 1, -4, -7, 0, -8, 17, 22, 1, -1, -4, 23],
    [3, 0, -5, -7, 0, -7, 15, 18, -5, 0, -5, 27],
    [2, 0, 0, -7, 1, -10, 13, 13, -4, 2, -7, 24],
    [3, 3, -13, 4, -2, -5, 9, 21, 25, -2, -3, 12],
    [-5, -2, 7, -3, -7, 9, 8, 9, 16, -2, 15, 12],
    [0, -1, 0, -7, -5, 4, 11, 11, 8, -6, 12, 21],
    [3, -2, -3, -8, -4, -1, 16, 15, -2, -3, 3, 26],
    [2, 1, -5, -4, -1, -8, 16, 4, -2, 1, -7, 33],
    [2, 1, -4, -2, 1, -10, 17, -2, 0, 2, -11, 33],
    [1, -2, 7, -15, -16, 10, 8, 8, 20, 11, 14, 11],
    [2, 2, 3, -13, -13, 4, 8, 12, 2, -3, 16, 24],
    [1, 4, 0, -7, -8, -4, 9, 9, -2, -2, 8, 29],
    [1, 1, 2, -4, -1, -6, 6, 3, -1, -1, -3, 30],
    [-7, 3, 2, 10, -2, 3, 7, 11, 19, -7, 8, 10],
    [0, -2, -5, -3, -2, 4, 20, 15, -1, -3, -1, 22],
    [3, -1, -8, -4, -1, -4, 22, 8, -4, 2, -8, 28],
    [0, 3, -14, 3, 0, 1, 19, 17, 8, -3, -7, 20],
    [0, 2, -1, -8, 3, -6, 5, 21, 1, 1, -9, 13],
    [-4, -2, 8, 20, -2, 2, 3, 5, 21, 4, 6, 1],
    [2, -2, -3, -9, -4, 2, 14, 16, 3, -6, 8, 24],
    [2, 1, 5, -16, -7, 2, 3, 11, 15, -3, 11, 22],
    [1, 2, 3, -11, -2, -5, 4, 8, 9, -3, -2, 26],
    [0, -1, 10, -9, -1, -8, 2, 3, 4, 0, 0, 29],
    [1, 2, 0, -5, 1, -9, 9, 3, 0, 1, -7, 20],
    [-2, 8, -6, -4, 3, -9, -8, 45, 14, 2, -13, 7],
    [1, -1, 16, -19, -8, -4, -3, 2, 19, 0, 4, 30],
    [1, 1, -3, 0, 2, -11, 15, -5, 1, 2, -9, 24],
    [0, 1, -2, 0, 1, -4, 4, 0, 0, 1, -4, 7],
    [0, 1, 2, -5, 1, -6, 4, 10, -2, 1, -4, 10],
    [3, 0, -3, -6, -2, -6, 14, 8, -1, -1, -3, 31],
    [0, 1, 0, -2, 1, -6, 5, 1, 0, 1, -5, 13],
    [3, 1, 9, -19, -21, 9, 7, 6, 13, 5, 15, 21],
    [2, 4, 3, -12, -13, 1, 7, 8, 3, 0, 12, 26],
    [3, 1, -8, -2, 0, -6, 18, 2, -2, 3, -10, 23],
    [1, 1, -4, -1, 1, -5, 8, 1, -1, 2, -5, 10],
    [0, 1, -1, 0, 0, -2, 2, 0, 0, 1, -2, 3],
    [1, 1, -2, -7, 1, -7, 14, 18, 0, 0, -7, 21],
    [0, 1, 0, -2, 0, -7, 8, 1, -2, 0, -3, 24],
    [0, 1, 1, -2, 2, -10, 10, 0, -2, 1, -7, 23],
    [0, 2, 2, -11, 2, -4, -3, 39, 7, 1, -10, 9],
    [1, 0, 13, -16, -5, -6, -1, 8, 6, 0, 6, 29],
    [1, 3, 1, -6, -4, -7, 9, 6, -3, -2, 3, 33],
    [4, 0, -17, -1, -1, 5, 26, 8, -2, 3, -15, 30],
    [0, 1, -2, 0, 2, -8, 12, -6, 1, 1, -6, 16],
    [0, 0, 0, -1, 1, -4, 4, 0, 0, 0, -3, 11],
    [0, 1, 2, -8, 2, -6, 5, 15, 0, 2, -7, 9],
    [1, -1, 12, -15, -7, -2, 3, 6, 6, -1, 7, 30],
];
#[rustfmt::skip]
const CLASS_TO_FILTER: [[u8; CLASSES]; FIXED_SETS] = [
    [8, 2, 2, 2, 3, 4, 53, 9, 9, 52, 4, 4, 5, 9, 2, 8, 10, 9, 1, 3, 39, 39, 10, 9, 52],
    [11, 12, 13, 14, 15, 30, 11, 17, 18, 19, 16, 20, 20, 4, 53, 21, 22, 23, 14, 25, 26, 26, 27, 28, 10],
    [16, 12, 31, 32, 14, 16, 30, 33, 53, 34, 35, 16, 20, 4, 7, 16, 21, 36, 18, 19, 21, 26, 37, 38, 39],
    [35, 11, 13, 14, 43, 35, 16, 4, 34, 62, 35, 35, 30, 56, 7, 35, 21, 38, 24, 40, 16, 21, 48, 57, 39],
    [11, 31, 32, 43, 44, 16, 4, 17, 34, 45, 30, 20, 20, 7, 5, 21, 22, 46, 40, 47, 26, 48, 63, 58, 10],
    [12, 13, 50, 51, 52, 11, 17, 53, 45, 9, 30, 4, 53, 19, 0, 22, 23, 25, 43, 44, 37, 27, 28, 10, 55],
    [30, 33, 62, 51, 44, 20, 41, 56, 34, 45, 20, 41, 41, 56, 5, 30, 56, 38, 40, 47, 11, 37, 42, 57, 8],
    [35, 11, 23, 32, 14, 35, 20, 4, 17, 18, 21, 20, 20, 20, 4, 16, 21, 36, 46, 25, 41, 26, 48, 49, 58],
    [12, 31, 59, 59, 3, 33, 33, 59, 59, 52, 4, 33, 17, 59, 55, 22, 36, 59, 59, 60, 22, 36, 59, 25, 55],
    [31, 25, 15, 60, 60, 22, 17, 19, 55, 55, 20, 20, 53, 19, 55, 22, 46, 25, 43, 60, 37, 28, 10, 55, 52],
    [12, 31, 32, 50, 51, 11, 33, 53, 19, 45, 16, 4, 4, 53, 5, 22, 36, 18, 25, 43, 26, 27, 27, 28, 10],
    [5, 2, 44, 52, 3, 4, 53, 45, 9, 3, 4, 56, 5, 0, 2, 5, 10, 47, 52, 3, 63, 39, 10, 9, 52],
    [12, 34, 44, 44, 3, 56, 56, 62, 45, 9, 56, 56, 7, 5, 0, 22, 38, 40, 47, 52, 48, 57, 39, 10, 9],
    [35, 11, 23, 14, 51, 35, 20, 41, 56, 62, 16, 20, 41, 56, 7, 16, 21, 38, 24, 40, 26, 26, 42, 57, 39],
    [33, 34, 51, 51, 52, 41, 41, 34, 62, 0, 41, 41, 56, 7, 5, 56, 38, 38, 40, 44, 37, 42, 57, 39, 10],
    [16, 31, 32, 15, 60, 30, 4, 17, 19, 25, 22, 20, 4, 53, 19, 21, 22, 46, 25, 55, 26, 48, 63, 58, 55],
];

/// The filters of a CTB whose luma ALF is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CtbFilter {
    /// Fixed set `alf_luma_fixed_filter_idx`.
    Fixed(u8),
    /// The slice's APS filters (`alf_use_aps_flag`).
    Signalled,
}

/// One CTB's luma ALF: off, or its filters.
pub(crate) type CtbAlf = Option<CtbFilter>;

/// Rounds of the encoder's APS derivation: the first from every candidate
/// CTB's statistics, each later one from the CTBs the previous round's bank
/// was chosen for.
const SIGNALLED_ROUNDS: usize = 2;

/// A 4x4 block's class (0..25) and transpose index (0..4).
#[derive(Debug, Clone, Copy, Default)]
struct Class {
    index: u8,
    transpose: u8,
}

/// Samples beyond each picture edge that classification (three before a
/// block, six after) and the filter (three) read, replicated from the edge
/// (§8.8.5.2 clamps sample positions to the picture).
const PAD: usize = 8;
/// Columns left of column 0 that the Laplacian kernel's rows start at.
const LAPLACIAN_LEAD: usize = 3;
/// Sampling steps (rows, columns) of the encoder's filter statistics: the
/// row step is `vvc.alf_stats_row_step`, by default this.
pub(crate) const STATS_ROW_STEP: usize = 2;
/// (The tap-sum kernel samples every `VVC_ALF_SAMPLE_STEP`th column.)
const STATS_COLUMN_STEP: usize = VVC_ALF_SAMPLE_STEP;

/// The deblocked, SAO-filtered luma of a picture, edge-padded, and the
/// class of each of its 4x4 blocks: what filtering and the encoder's
/// decision read.
pub(crate) struct Analysis<'a> {
    dsp: &'a Dsp,
    width: usize,
    height: usize,
    stride: usize,
    samples: Vec<u8>,
    ctu: usize,
    block_columns: usize,
    /// Each 4x4 block's `[class index, transpose index]`, raster order.
    classes: Vec<[u8; 2]>,
}

impl<'a> Analysis<'a> {
    /// Analyse the luma of `picture` (`ctu`-sample CTUs): classify the
    /// blocks of the CTBs (raster index) `wanted` names: those that may be
    /// filtered (encoder) or are (decoder).
    pub(crate) fn new(
        dsp: &'a Dsp,
        picture: &Picture,
        ctu: usize,
        wanted: impl Fn(usize) -> bool,
    ) -> Self {
        let view = picture.plane(PlaneIndex::Y).view();
        let (width, height) = (view.width(), view.height());
        let stride = width + 2 * PAD;
        // Built row by row: every sample is written, so none is zeroed first.
        let mut samples = Vec::with_capacity(stride * (height + 2 * PAD));
        for y in 0..height + 2 * PAD {
            let source = view.row(y.saturating_sub(PAD).min(height - 1));
            let (first, last) = (source[0], source[width - 1]);
            samples.extend_from_slice(&[first; PAD]);
            samples.extend_from_slice(source);
            samples.extend_from_slice(&[last; PAD]);
        }
        let block_columns = width.div_ceil(BLOCK);
        let mut analysis = Self {
            dsp,
            width,
            height,
            stride,
            samples,
            ctu,
            block_columns,
            classes: Vec::new(),
        };
        analysis.classes = analysis.classify(wanted);
        analysis
    }

    /// Padded row `y`, starting [`PAD`] samples left of the picture.
    fn row(&self, y: isize) -> &[u8] {
        let start = (y + PAD as isize) as usize * self.stride;
        &self.samples[start..start + self.stride]
    }

    /// The row position within its CTU row relative to the virtual boundary.
    fn boundary_row(&self, y: usize) -> isize {
        (y % self.ctu) as isize - (self.ctu - VIRTUAL_BOUNDARY_ABOVE) as isize
    }

    /// The Laplacian group sums (`nc_dsp` `vvc_alf_laplacians`) of the
    /// classification grid row at picture row `y`; rows across the virtual
    /// boundary are replaced by the nearest on the row's side.
    fn laplacian_row(&self, y: isize, first_block: usize, out: &mut [[u16; 4]]) {
        let boundary = if y > 0 {
            self.boundary_row(y as usize)
        } else {
            isize::MIN
        };
        let row = |dy: isize| {
            let row = self.row(match (boundary, dy) {
                (-2, 2) => y + 1,
                (0, -1) => y,
                _ => y + dy,
            });
            &row[PAD + first_block * BLOCK - LAPLACIAN_LEAD..]
        };
        self.dsp
            .vvc_alf_laplacians(&[row(-1), row(0), row(1), row(2)], out);
    }

    /// The class of every block of the CTBs (raster index) that `needed`
    /// selects, others zero. Per CTB row, each run of selected CTBs takes
    /// the Laplacians on its 2x2 grid once: grid row j is at y = 2j − 2,
    /// group q of a row sums the points at x = 4q − 2 and 4q, and the block
    /// at (x, y) sums groups x/4 and x/4 + 1 of grid rows y/2 .. y/2 + 4.
    fn classify(&self, needed: impl Fn(usize) -> bool) -> Vec<[u8; 2]> {
        let block_rows = self.height.div_ceil(BLOCK);
        let mut classes = vec![[0; 2]; block_rows * self.block_columns];
        let ctb_columns = self.width.div_ceil(self.ctu);
        let ctb_blocks = self.ctu / BLOCK;
        let mut grid = Vec::new();
        let mut column_sums = Vec::new();
        for (ctb_row, y0) in (0..self.height).step_by(self.ctu).enumerate() {
            let block_rows = y0 / BLOCK..(y0 + self.ctu).min(self.height).div_ceil(BLOCK);
            let mut column = 0;
            while column < ctb_columns {
                if !needed(ctb_row * ctb_columns + column) {
                    column += 1;
                    continue;
                }
                let first = column;
                while column < ctb_columns && needed(ctb_row * ctb_columns + column) {
                    column += 1;
                }
                let blocks = first * ctb_blocks..(column * ctb_blocks).min(self.block_columns);
                let groups = blocks.len() + 1;
                // Grid rows 2 · block_rows.start .. 2 · block_rows.end + 2.
                let grid_rows = 2 * block_rows.start..2 * block_rows.end + 2;
                grid.resize(groups * grid_rows.len(), [0; 4]);
                for (j, row) in grid_rows.clone().zip(grid.chunks_exact_mut(groups)) {
                    self.laplacian_row(2 * j as isize - 2, blocks.start, row);
                }
                column_sums.resize(groups, [0; 4]);
                for by in block_rows.clone() {
                    let y = by * BLOCK;
                    // Laplacian rows y − 2 .. y + 4; next to the virtual
                    // boundary the three on the block's side.
                    let rows = match self.boundary_row(y) {
                        -4 => 0..3,
                        0 => 1..4,
                        _ => 0..4,
                    };
                    // Each group summed over the rows (at most 4 · 2040); the
                    // kernel classifies each block from its two groups.
                    column_sums.fill([0; 4]);
                    let boundary = rows.len() == 3;
                    for j in rows {
                        let row = &grid[(y / 2 + j - grid_rows.start) * groups..][..groups];
                        // Flat, so the adds vectorise.
                        for (sum, &l) in column_sums
                            .as_flattened_mut()
                            .iter_mut()
                            .zip(row.as_flattened())
                        {
                            *sum += l;
                        }
                    }
                    let start = by * self.block_columns;
                    self.dsp.vvc_alf_classify(
                        &column_sums,
                        boundary,
                        &mut classes[start + blocks.start..start + blocks.end],
                    );
                }
            }
        }
        classes
    }
}

impl Analysis<'_> {
    fn class(&self, x: usize, y: usize) -> Class {
        let [index, transpose] = self.classes[y / BLOCK * self.block_columns + x / BLOCK];
        Class { index, transpose }
    }

    /// How far row `y`'s taps may reach vertically: rows across the virtual
    /// boundary are not read (a tap's pair is mirrored at the nearer edge).
    fn reach(&self, y: usize) -> isize {
        match self.boundary_row(y) {
            b @ -4..=-1 => -b - 1,
            b @ 0..=3 => b,
            _ => 3,
        }
    }

    /// The tap rows of row `y` (for dy -3..=3) and its rounding shift.
    fn tap_rows(&self, y: usize) -> ([&[u8]; 7], u32) {
        let reach = self.reach(y);
        let rows =
            std::array::from_fn(|i| self.row(y as isize + (i as isize - 3).clamp(-reach, reach)));
        (rows, if reach == 0 { BOUNDARY_SHIFT } else { SHIFT })
    }

    /// Filter the CTBs of rectangle `ctb` (whole CTBs of one CTB row) with
    /// the set whose coefficients by class and transpose are `set` into
    /// `output`; `coefficients` holds at least the rectangle's blocks of a row.
    fn filter_ctb(
        &self,
        (x0, y0, w, h): Ctb,
        set: &SetCoefficients,
        coefficients: &mut [[i16; TAPS]],
        output: &mut nc_common::PlaneViewMut<'_>,
    ) {
        let coefficients = &mut coefficients[..w.div_ceil(BLOCK)];
        let stride = output.stride();
        // Picture heights are multiples of the minimum coding block (8), so
        // every block row is whole.
        for y in (y0..y0 + h).step_by(BLOCK) {
            // Blocks keep their classes for four rows.
            for (block, c) in coefficients.iter_mut().enumerate() {
                let class = self.class(x0 + block * BLOCK, y);
                *c = set[usize::from(class.index)][usize::from(class.transpose)];
            }
            let taps: [_; BLOCK] = std::array::from_fn(|r| self.tap_rows(y + r));
            let rows = taps.map(|(rows, _)| rows.map(|row| &row[x0 + PAD - VVC_ALF_REACH..]));
            let shifts = taps.map(|(_, shift)| shift);
            let mut lines = output.data_mut()[y * stride..]
                .chunks_mut(stride)
                .map(|line| &mut line[x0..x0 + w]);
            let mut out: [&mut [u8]; BLOCK] =
                std::array::from_fn(|_| lines.next().expect("four rows in the picture"));
            self.dsp
                .vvc_alf_filter_block_row(&rows, coefficients, &shifts, &mut out);
        }
    }

    /// Apply each CTB's decision (raster order) to the luma of `picture`;
    /// `signalled` is the slice's APS bank, if it has one.
    pub(crate) fn apply(
        &self,
        picture: &mut Picture,
        decisions: &[CtbAlf],
        signalled: Option<&SignalledFilters>,
    ) {
        let mut output = picture.plane_mut(PlaneIndex::Y).view_mut();
        let sets: Vec<SetCoefficients> = (0..FIXED_SETS as u8).map(set_coefficients).collect();
        let signalled = signalled.map(|bank| class_coefficients(|class| bank.filter(class)));
        let mut coefficients = vec![[0; TAPS]; self.block_columns];
        for (run, decision) in self.runs(decisions) {
            let set = match decision {
                None => continue,
                Some(CtbFilter::Fixed(set)) => &sets[usize::from(set)],
                Some(CtbFilter::Signalled) => signalled
                    .as_ref()
                    .expect("a signalled CTB has its slice's bank"),
            };
            self.filter_ctb(run, set, &mut coefficients, &mut output);
        }
    }

    /// The runs of horizontally adjacent CTBs whose `items` (raster order)
    /// are equal, as rectangles with their item: one filter call per row of
    /// a run rather than of each CTB.
    fn runs<'i, T: PartialEq + Copy>(&self, items: &'i [T]) -> impl Iterator<Item = (Ctb, T)> + 'i {
        let (ctu, width, height) = (self.ctu, self.width, self.height);
        let columns = width.div_ceil(ctu);
        items
            .chunks(columns)
            .enumerate()
            .flat_map(move |(row, items)| {
                let y0 = row * ctu;
                let mut first = 0;
                std::iter::from_fn(move || {
                    (first < items.len()).then(|| {
                        let item = items[first];
                        let end = items[first..]
                            .iter()
                            .position(|&i| i != item)
                            .map_or(items.len(), |n| first + n);
                        let x0 = first * ctu;
                        let run = (x0, y0, (end * ctu).min(width) - x0, ctu.min(height - y0));
                        first = end;
                        (run, item)
                    })
                })
            })
    }

    /// Choose each CTB's filters (or none) by luma squared error against
    /// `source` plus `lambda` times the signalling bits, and with
    /// `options.signalled` whether the slice signals an APS bank (derived
    /// from these statistics, `alf_aps::derive`) and its bits pay. The error
    /// change of every filter is estimated from per-class statistics of the
    /// tap sums (sampled every `row_step`th row and [`STATS_COLUMN_STEP`]th
    /// column), as a quadratic in the coefficients: filtering is linear in
    /// them, so one pass serves all filters. The scaled changes are integers
    /// far below 2^53, so their conversion to `f64` is exact.
    ///
    /// CTBs `skipped` names (raster index; their blocks need not be
    /// classified) stay off without statistics.
    ///
    /// With `reuse` (an earlier picture's bank; the caller passes one only
    /// with `options.exact` and without fixed sets), no statistics are
    /// gathered and no bank is derived: the CTBs are decided on their exact
    /// error change with `reuse` (`VVenC`'s temporal APS reuse,
    /// `vvc.alf_bank_interval`).
    pub(crate) fn decide(
        &self,
        source: &Picture,
        lambda: f32,
        contexts: &Contexts,
        skipped: impl Fn(usize) -> bool,
        options: DecideOptions,
        reuse: Option<&SignalledFilters>,
        scratch: &mut DecideScratch,
    ) -> Decision {
        let original = source.plane(PlaneIndex::Y).view();
        let plan = FilterPlan::new();
        debug_assert!(reuse.is_none() || (options.exact && !options.fixed));
        // Kept between pictures: each CTB's entry is overwritten in place.
        let count = ctbs(self.width, self.height, self.ctu).count();
        scratch.ctbs.resize_with(count, CtbStatistics::default);
        for ((index, ctb), statistics) in ctbs(self.width, self.height, self.ctu)
            .enumerate()
            .zip(&mut scratch.ctbs)
        {
            statistics.candidate = !skipped(index);
            if !statistics.candidate || reuse.is_some() {
                continue;
            }
            self.statistics(
                ctb,
                &original,
                options.row_step,
                &mut scratch.samples,
                &mut statistics.classes,
            );
            if options.fixed {
                statistics.fixed =
                    plan.set_changes(self.dsp, &statistics.classes, &scratch.weights);
            }
        }
        let chooser = Chooser {
            dsp: self.dsp,
            columns: self.width.div_ceil(self.ctu),
            units: Units {
                error_scale: (options.row_step * STATS_COLUMN_STEP) as f64
                    / f64::from(1 << (2 * SHIFT)),
                lambda: f64::from(lambda),
            },
            contexts,
            ctbs: &scratch.ctbs,
        };
        if let Some(bank) = reuse {
            return self.exact_decision(
                &original,
                bank.clone(),
                &chooser,
                options,
                &mut scratch.filtered,
            );
        }
        let (ctbs, cost) = chooser.choose(None, options.fixed, None);
        let mut best = (
            cost,
            Decision {
                ctbs,
                signalled: None,
            },
        );
        if !options.signalled {
            return best.1;
        }
        let mut chosen: Vec<bool> = scratch.ctbs.iter().map(|s| s.candidate).collect();
        for _ in 0..SIGNALLED_ROUNDS {
            let Some(bank) = alf_aps_bank(&scratch.ctbs, &chosen, chooser.units) else {
                break;
            };
            let (ctbs, cost) = chooser.choose(Some(&bank), options.fixed, None);
            let cost = cost + chooser.units.lambda * f64::from(bank.bits());
            chosen = ctbs
                .iter()
                .map(|d| *d == Some(CtbFilter::Signalled))
                .collect();
            if cost < best.0 {
                best = (
                    cost,
                    Decision {
                        ctbs,
                        signalled: Some(bank),
                    },
                );
            }
        }
        if !options.exact {
            return best.1;
        }
        let Some(bank) = best.1.signalled.take() else {
            return best.1;
        };
        // The estimate chose a bank: decide its CTBs again on their exact
        // squared-error change. Sampled statistics alone turned filtering
        // on where it hurt (crowd, pan: −0.03..−0.08 dB luma at 4-row
        // sampling).
        self.exact_decision(&original, bank, &chooser, options, &mut scratch.filtered)
    }

    /// Each CTB's choice between `bank` (and the fixed sets, if searched)
    /// and off on its exact squared-error change with `bank`, filtering
    /// every candidate CTB into `filtered` (which `apply_decided` then
    /// copies); every CTB off if that costs less.
    fn exact_decision(
        &self,
        original: &PlaneView<'_>,
        bank: SignalledFilters,
        chooser: &Chooser<'_>,
        options: DecideOptions,
        filtered: &mut Vec<u8>,
    ) -> Decision {
        let changes = self.exact_changes(
            original,
            &bank,
            chooser.ctbs,
            f64::from(options.detail_weight),
            filtered,
        );
        let (ctbs, cost) = chooser.choose(Some(&bank), options.fixed, Some(&changes));
        let cost = cost + chooser.units.lambda * f64::from(bank.bits());
        let (fallback, fallback_cost) = chooser.choose(None, options.fixed, None);
        if cost < fallback_cost {
            Decision {
                ctbs,
                signalled: Some(bank),
            }
        } else {
            Decision {
                ctbs: fallback,
                signalled: None,
            }
        }
    }

    /// Each CTB's exact luma squared-error change (against `original`) when
    /// filtered with `bank`, for the candidate CTBs (others 0); the filtered
    /// samples are left in `filtered` (a picture-sized luma plane).
    fn exact_changes(
        &self,
        original: &PlaneView<'_>,
        bank: &SignalledFilters,
        statistics: &[CtbStatistics],
        detail_weight: f64,
        filtered: &mut Vec<u8>,
    ) -> Vec<f64> {
        filtered.resize(self.width * self.height, 0);
        let mut output =
            nc_common::PlaneViewMut::new(filtered, self.width, self.width, self.height)
                .expect("a picture-sized plane");
        let set = class_coefficients(|class| bank.filter(class));
        let mut coefficients = vec![[0; TAPS]; self.block_columns];
        let candidates: Vec<bool> = statistics.iter().map(|s| s.candidate).collect();
        for (run, candidate) in self.runs(&candidates) {
            if candidate {
                self.filter_ctb(run, &set, &mut coefficients, &mut output);
            }
        }
        let output = output.as_view();
        let unfiltered = PlaneView::new(
            &self.samples[PAD * self.stride + PAD..],
            self.stride,
            self.width,
            self.height,
        )
        .expect("the padded picture");
        let mut moments = Vec::new();
        ctbs(self.width, self.height, self.ctu)
            .zip(statistics)
            .map(|(ctb, statistics)| {
                if !statistics.candidate {
                    return 0.;
                }
                let after = region_sse(self.dsp, original, &output, ctb);
                let before = region_sse(self.dsp, original, &unfiltered, ctb);
                // Detail the filter removes (AC energy lost), weighed as
                // squared error: VMAF NEG fell where Wiener filtering gained
                // PSNR by smoothing (bodycam_antare_0815 at 105 kbps: Y
                // +0.3..0.8 dB, NEG −2..−6 per frame at equal bits).
                let detail = if detail_weight > 0. {
                    let lost = ac_energy(self.dsp, &unfiltered, ctb, &mut moments)
                        .saturating_sub(ac_energy(self.dsp, &output, ctb, &mut moments));
                    detail_weight * lost as f64
                } else {
                    0.
                };
                after as f64 - before as f64 + detail
            })
            .collect()
    }

    /// Apply an encoder decision: CTBs on the bank take the samples
    /// [`Self::decide`] filtered into `scratch`; fixed sets are filtered here.
    pub(crate) fn apply_decided(
        &self,
        picture: &mut Picture,
        decision: &Decision,
        options: DecideOptions,
        scratch: &DecideScratch,
    ) {
        if !options.exact || decision.signalled.is_none() {
            self.apply(picture, &decision.ctbs, decision.signalled.as_ref());
            return;
        }
        let fixed: Vec<CtbAlf> = decision
            .ctbs
            .iter()
            .map(|d| d.filter(|f| matches!(f, CtbFilter::Fixed(_))))
            .collect();
        self.apply(picture, &fixed, None);
        let mut output = picture.plane_mut(PlaneIndex::Y).view_mut();
        for ((x0, y0, w, h), decision) in
            ctbs(self.width, self.height, self.ctu).zip(&decision.ctbs)
        {
            if *decision == Some(CtbFilter::Signalled) {
                for y in y0..y0 + h {
                    output.row_mut(y)[x0..x0 + w]
                        .copy_from_slice(&scratch.filtered[y * self.width + x0..][..w]);
                }
            }
        }
    }

    /// Per-class tap-sum statistics of CTB `ctb` against `original` into
    /// `stats` (the Gram matrix only of classes with samples): the sampled
    /// tap sums and errors are written into per-class runs of `samples`
    /// (counted first, so each sample is written once), whose Gram matrices
    /// are the statistics.
    fn statistics(
        &self,
        (x0, y0, w, h): Ctb,
        original: &PlaneView<'_>,
        row_step: usize,
        samples: &mut Vec<[i16; VVC_ALF_SAMPLE_LANES]>,
        stats: &mut [ClassStatistics; CLASSES],
    ) {
        for stats in stats.iter_mut() {
            stats.count = 0;
        }
        let rows = (y0..y0 + h).step_by(row_step);
        for y in rows.clone() {
            for x in sampled_columns(x0, w, y, row_step).step_by(STATS_COLUMN_STEP) {
                stats[usize::from(self.class(x, y).index)].count += 1;
            }
        }
        // Each class a run of samples padded with zeros to whole kernel steps.
        let mut starts = [0; CLASSES + 1];
        for class in 0..CLASSES {
            starts[class + 1] =
                starts[class] + (stats[class].count as usize).next_multiple_of(VVC_ALF_GRAM_ALIGN);
        }
        // Every sample is written below; only the padding needs clearing.
        samples.resize(starts[CLASSES], [0; VVC_ALF_SAMPLE_LANES]);
        for (class, stats) in stats.iter().enumerate() {
            samples[starts[class] + stats.count as usize..starts[class + 1]]
                .fill([0; VVC_ALF_SAMPLE_LANES]);
        }
        let mut next = starts;
        let mut sums = [[0; VVC_ALF_SAMPLE_LANES]; VVC_ALF_SAMPLES];
        for y in rows {
            let (taps, _) = self.tap_rows(y);
            let source = original.row(y);
            // Runs of up to `VVC_ALF_SAMPLES` sampled columns per kernel call.
            let sampled = sampled_columns(x0, w, y, row_step);
            for start in sampled.clone().step_by(STATS_COLUMN_STEP * VVC_ALF_SAMPLES) {
                let count = (sampled.end - start)
                    .div_ceil(STATS_COLUMN_STEP)
                    .min(VVC_ALF_SAMPLES);
                let classes: [Class; VVC_ALF_SAMPLES] = std::array::from_fn(|i| {
                    // Past `count` any class does: those samples are unused.
                    self.class(start + i.min(count - 1) * STATS_COLUMN_STEP, y)
                });
                let windows = taps.map(|row| &row[start + PAD - VVC_ALF_REACH..]);
                // Tap sums (at most 510) in the order of the untransposed
                // coefficients, then the error.
                self.dsp.vvc_alf_samples(
                    &windows,
                    &source[start..],
                    &classes.map(|class| class.transpose),
                    count,
                    &mut sums,
                );
                for (class, sample) in classes.iter().zip(&sums).take(count) {
                    let at = &mut next[usize::from(class.index)];
                    samples[*at] = *sample;
                    *at += 1;
                }
            }
        }
        for (class, stats) in stats.iter_mut().enumerate() {
            if stats.count > 0 {
                let run = &samples[starts[class]..starts[class + 1]];
                self.dsp.vvc_alf_gram(run, &mut stats.gram);
            }
        }
    }
}

/// The span of CTB columns `x0..x0 + w` from which statistics sample
/// every [`STATS_COLUMN_STEP`]th column on row `y` (of those every
/// `row_step`th row samples): staggered over sampled rows so that every
/// column phase is sampled.
fn sampled_columns(x0: usize, w: usize, y: usize, row_step: usize) -> std::ops::Range<usize> {
    let row = y / row_step;
    let stagger = row % 2 * 2 + row / 2 % 2;
    x0 + stagger..x0 + w
}

/// The Gram matrix of a class's samples (tap sums, then the
/// reconstruction error): the tap-sum products `A` (the first [`TAPS`]
/// rows and columns) and the tap sums times the error `b` (the last
/// column). The squared-error change of filtering with `c` is
/// `2 c·b / 2^7 + cᵀ A c / 2^14` (rounding and clipping aside).
#[derive(Debug, Clone, Copy)]
struct ClassStatistics {
    count: u32,
    /// The packed upper triangle (`nc_dsp::vvc_alf_gram_index`).
    gram: [i32; VVC_ALF_DOT_LANES],
}

impl Default for ClassStatistics {
    fn default() -> Self {
        Self {
            count: 0,
            gram: [0; VVC_ALF_DOT_LANES],
        }
    }
}

/// The weights of filter `c` on a class's packed Gram matrix whose dot
/// product is the squared-error change scaled by 2^14:
/// `cᵀ A c + 2^8 c·b` (`A` the tap products, `b` the tap-error products;
/// the error's own square weighs nothing).
fn filter_weights(c: &[i8; TAPS]) -> [i32; VVC_ALF_DOT_LANES] {
    let c = c.map(i32::from);
    let mut weights = [0; VVC_ALF_DOT_LANES];
    for i in 0..TAPS {
        for j in i..=TAPS {
            weights[vvc_alf_gram_index(i, j)] = match (i == j, j == TAPS) {
                (true, _) => c[i] * c[i],
                // 2^14 · 2 c·b / 2^7.
                (_, true) => c[i] << (SHIFT + 1),
                _ => 2 * c[i] * c[j],
            };
        }
    }
    weights
}

/// The distinct filters each class uses across the fixed sets (sets share
/// filters), so a CTB evaluates each once per class.
struct FilterPlan {
    /// Per class, its distinct filters, and how many there are.
    filters: [([u8; FIXED_SETS], usize); CLASSES],
    /// Per set and class, the index of the set's filter in the class's list.
    slots: [[u8; CLASSES]; FIXED_SETS],
}

impl FilterPlan {
    fn new() -> Self {
        let mut filters = [([0; FIXED_SETS], 0); CLASSES];
        let mut slots = [[0; CLASSES]; FIXED_SETS];
        for (class, (list, len)) in filters.iter_mut().enumerate() {
            for (set, slots) in slots.iter_mut().enumerate() {
                let filter = CLASS_TO_FILTER[set][class];
                let slot = list[..*len]
                    .iter()
                    .position(|&f| f == filter)
                    .unwrap_or_else(|| {
                        list[*len] = filter;
                        *len += 1;
                        *len - 1
                    });
                slots[class] = slot as u8;
            }
        }
        Self { filters, slots }
    }

    /// Each fixed set's squared-error change over the classes of `stats`,
    /// scaled by 2^14, in exact integers.
    fn set_changes(
        &self,
        dsp: &Dsp,
        stats: &[ClassStatistics; CLASSES],
        weights: &[[i32; VVC_ALF_DOT_LANES]; FIXED_FILTER_COUNT],
    ) -> [i64; FIXED_SETS] {
        let mut changes = [0; FIXED_SETS];
        for (class, stats) in stats.iter().enumerate().filter(|(_, s)| s.count > 0) {
            let (list, len) = &self.filters[class];
            let mut evaluated = [0; FIXED_SETS];
            for (value, &filter) in evaluated.iter_mut().zip(&list[..*len]) {
                *value = dsp.vvc_alf_dot(&weights[usize::from(filter)], &stats.gram);
            }
            for (change, slots) in changes.iter_mut().zip(&self.slots) {
                *change += evaluated[usize::from(slots[class])];
            }
        }
        changes
    }
}

/// A CTB as a luma rectangle (x, y, width, height).
type Ctb = (usize, usize, usize, usize);

/// A filter set's coefficients by class and transpose index.
type SetCoefficients = [[[i16; TAPS]; 4]; CLASSES];

/// The transposed coefficients of the filter `filter` gives each class.
fn class_coefficients<'f>(filter: impl Fn(usize) -> &'f [i8; TAPS]) -> SetCoefficients {
    std::array::from_fn(|class| {
        let filter = filter(class);
        VVC_ALF_TRANSPOSES.map(|order| order.map(|i| i16::from(filter[i])))
    })
}

/// The transposed coefficients of fixed set `set` for every class.
fn set_coefficients(set: u8) -> SetCoefficients {
    class_coefficients(|class| {
        &FIXED_FILTERS[usize::from(CLASS_TO_FILTER[usize::from(set)][class])]
    })
}

/// The CTB grid of a `width` x `height` luma plane with `ctu`-sample CTUs,
/// in raster order.
fn ctbs(width: usize, height: usize, ctu: usize) -> impl Iterator<Item = Ctb> {
    (0..height).step_by(ctu).flat_map(move |y| {
        (0..width)
            .step_by(ctu)
            .map(move |x| (x, y, ctu.min(width - x), ctu.min(height - y)))
    })
}

/// The `alf_ctb_flag` context of a CTB whose left and upper CTBs (where
/// they exist) chose `left` and `up` (§9.3.4.2.2: the count of enabled ones).
fn flag_context(left: Option<CtbAlf>, up: Option<CtbAlf>) -> usize {
    usize::from(left.flatten().is_some()) + usize::from(up.flatten().is_some())
}

/// One CTB's luma ALF syntax (§7.3.11.3): `alf_ctb_flag`, then, when the
/// slice names an APS (`signalled`), `alf_use_aps_flag` (one APS: no
/// `alf_luma_prev_filter_idx`), else or when that is 0
/// `alf_luma_fixed_filter_idx`. Returns the decision as coded.
pub(crate) fn code(
    io: &mut impl Bins,
    c: &mut Contexts,
    decision: CtbAlf,
    (left, up): (Option<CtbAlf>, Option<CtbAlf>),
    signalled: bool,
) -> Result<CtbAlf> {
    if !io.bin(&mut c.alf_ctb[flag_context(left, up)], decision.is_some())? {
        return Ok(None);
    }
    if signalled && io.bin(&mut c.alf_use_aps, decision == Some(CtbFilter::Signalled))? {
        return Ok(Some(CtbFilter::Signalled));
    }
    let set = match decision {
        Some(CtbFilter::Fixed(set)) => set,
        _ => 0,
    };
    let set = io.bypass(u32::from(set), FIXED_SET_BITS)?;
    Ok(Some(CtbFilter::Fixed(set as u8)))
}

/// How [`Analysis::decide`] searches.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DecideOptions {
    /// Statistics sample every this many rows (`vvc.alf_stats_row_step`).
    pub(crate) row_step: usize,
    /// Whether the slice may signal an APS bank (`vvc.alf_aps`).
    pub(crate) signalled: bool,
    /// Whether CTBs may choose the fixed sets (`vvc.alf_fixed_sets`).
    pub(crate) fixed: bool,
    /// Whether the CTBs of a chosen bank are decided on their exact error
    /// change (`vvc.alf_exact_decision`).
    pub(crate) exact: bool,
    /// With `exact`, the squared-error weight of the AC energy filtering
    /// removes from a CTB (`vvc.alf_detail_weight`).
    pub(crate) detail_weight: f32,
}

/// A picture's ALF decision: each CTB's filters (raster order) and the APS
/// bank the slice signals, if any.
#[derive(Debug, Clone, Default)]
pub(crate) struct Decision {
    pub(crate) ctbs: Vec<CtbAlf>,
    pub(crate) signalled: Option<SignalledFilters>,
}

impl Decision {
    /// The slice header's ALF syntax for this decision, its bank (if any)
    /// signalled as APS `aps`.
    pub(crate) fn slice_syntax(&self, aps: u8) -> crate::headers::SliceAlf {
        if self.ctbs.iter().any(Option::is_some) {
            crate::headers::SliceAlf::On {
                aps: self.signalled.is_some().then_some(aps),
            }
        } else {
            crate::headers::SliceAlf::Off
        }
    }
}

/// One CTB's statistics: per class, and every fixed set's squared-error
/// change (scaled by 2^14, per sampled position); none unless `candidate`.
#[derive(Debug, Clone, Copy)]
struct CtbStatistics {
    candidate: bool,
    classes: [ClassStatistics; CLASSES],
    fixed: [i64; FIXED_SETS],
}

impl Default for CtbStatistics {
    fn default() -> Self {
        Self {
            candidate: false,
            classes: [ClassStatistics::default(); CLASSES],
            fixed: [0; FIXED_SETS],
        }
    }
}

/// The encoder decision's buffers, kept between pictures.
#[derive(Debug)]
pub(crate) struct DecideScratch {
    ctbs: Vec<CtbStatistics>,
    samples: Vec<[i16; VVC_ALF_SAMPLE_LANES]>,
    /// The luma plane filtered with the bank (`Analysis::exact_changes`).
    filtered: Vec<u8>,
    /// Each fixed filter's weights on a Gram matrix (`filter_weights`).
    weights: [[i32; VVC_ALF_DOT_LANES]; FIXED_FILTER_COUNT],
}

impl Default for DecideScratch {
    fn default() -> Self {
        Self {
            ctbs: Vec::new(),
            samples: Vec::new(),
            filtered: Vec::new(),
            weights: FIXED_FILTERS.map(|filter| filter_weights(&filter)),
        }
    }
}

/// The per-CTB choice, given the CTBs' statistics and the bank (if any).
struct Chooser<'a> {
    dsp: &'a Dsp,
    columns: usize,
    units: Units,
    contexts: &'a Contexts,
    ctbs: &'a [CtbStatistics],
}

impl Chooser<'_> {
    /// Each CTB's cheapest filters (raster order, so that the flag contexts
    /// follow the left and upper choices) with the slice's APS bank
    /// `signalled`, and their summed cost (the bank's own bits aside).
    /// `exact`, when given, is each CTB's squared-error change with the bank
    /// (in place of the statistics' estimate).
    fn choose(
        &self,
        signalled: Option<&SignalledFilters>,
        fixed: bool,
        exact: Option<&[f64]>,
    ) -> (Vec<CtbAlf>, f64) {
        let lambda = self.units.lambda;
        let bits = |c: crate::contexts::Context, value: bool| f64::from(c.bits(value));
        let use_aps = |value: bool| {
            if signalled.is_some() {
                bits(self.contexts.alf_use_aps, value)
            } else {
                0.
            }
        };
        let weights: Option<Vec<[i32; VVC_ALF_DOT_LANES]>> = signalled.map(|bank| {
            (0..CLASSES)
                .map(|class| filter_weights(bank.filter(class)))
                .collect()
        });
        let mut decisions: Vec<CtbAlf> = Vec::with_capacity(self.ctbs.len());
        let mut total = 0.;
        for (index, statistics) in self.ctbs.iter().enumerate() {
            let left = (index % self.columns > 0).then(|| decisions[index - 1]);
            let up = (index >= self.columns).then(|| decisions[index - self.columns]);
            let flag = self.contexts.alf_ctb[flag_context(left, up)];
            let mut best = (None, lambda * bits(flag, false));
            if statistics.candidate {
                let on = lambda * bits(flag, true);
                let fixed_bits = lambda * (f64::from(FIXED_SET_BITS) + use_aps(false));
                for (set, &change) in (0..).zip(&statistics.fixed).filter(|_| fixed) {
                    let cost = change as f64 * self.units.error_scale + on + fixed_bits;
                    if cost < best.1 {
                        best = (Some(CtbFilter::Fixed(set)), cost);
                    }
                }
                if let Some(weights) = &weights {
                    let change = exact.map_or_else(
                        || {
                            let change: i64 = statistics
                                .classes
                                .iter()
                                .zip(weights)
                                .filter(|(s, _)| s.count > 0)
                                .map(|(s, w)| self.dsp.vvc_alf_dot(w, &s.gram))
                                .sum();
                            change as f64 * self.units.error_scale
                        },
                        |exact| exact[index],
                    );
                    let cost = change + on + lambda * use_aps(true);
                    if cost < best.1 {
                        best = (Some(CtbFilter::Signalled), cost);
                    }
                }
            }
            total += best.1;
            decisions.push(best.0);
        }
        (decisions, total)
    }
}

/// The APS bank derived from the class statistics of the CTBs `chosen`
/// selects.
fn alf_aps_bank(ctbs: &[CtbStatistics], chosen: &[bool], units: Units) -> Option<SignalledFilters> {
    let mut grams: Vec<Option<Gram>> = vec![None; CLASSES];
    for (statistics, _) in ctbs.iter().zip(chosen).filter(|&(s, &c)| s.candidate && c) {
        for (sum, class) in grams.iter_mut().zip(&statistics.classes) {
            if class.count > 0 {
                let sum = sum.get_or_insert([0; VVC_ALF_DOT_LANES]);
                for (s, &g) in sum.iter_mut().zip(&class.gram) {
                    *s += i64::from(g);
                }
            }
        }
    }
    crate::alf_aps::derive(&grams, units)
}

/// Squared error between `first` and `second` over the rectangle
/// `(x0, y0, width, height)` (sides multiples of 4), in the largest
/// power-of-two blocks the SSE kernels take.
fn region_sse(
    dsp: &Dsp,
    first: &PlaneView<'_>,
    second: &PlaneView<'_>,
    (x0, y0, width, height): Ctb,
) -> u64 {
    let tiles = |length: usize| {
        let mut at = 0;
        std::iter::from_fn(move || {
            (at < length).then(|| {
                let side = 1 << (length - at).min(MAX_SSE_BLOCK).ilog2();
                at += side;
                (at - side, side)
            })
        })
    };
    let mut sse = 0;
    for (dy, rows) in tiles(height) {
        for (dx, columns) in tiles(width) {
            let size = nc_dsp::BlockSize::new(columns, rows).expect("a power-of-two block");
            let first = first
                .sub_view(x0 + dx, y0 + dy, columns, rows)
                .expect("inside the picture");
            let second = second
                .sub_view(x0 + dx, y0 + dy, columns, rows)
                .expect("inside the picture");
            sse += u64::from(dsp.variance(size, &first, &second).sse);
        }
    }
    sse
}

/// The AC energy (`n · variance` of each 8x8 block, summed) of `view` over
/// the rectangle `(x0, y0, width, height)` (sides multiples of 8), from the
/// blocks' moments one row of blocks per kernel call.
fn ac_energy(
    dsp: &Dsp,
    view: &PlaneView<'_>,
    (x0, y0, width, height): Ctb,
    moments: &mut Vec<Moments>,
) -> u64 {
    moments.resize(width / ENERGY_BLOCK, Moments::default());
    let mut energy = 0;
    for y in (y0..y0 + height).step_by(ENERGY_BLOCK) {
        let row = view
            .sub_view(x0, y, width, ENERGY_BLOCK)
            .expect("inside the picture");
        dsp.moments_8x8_row(&row, moments);
        // `sum_sq − ⌊sum² / 64⌋`, as `Variance::n_variance` of the block.
        energy += moments
            .iter()
            .map(|m| u64::from(m.sum_sq) - u64::from(m.sum) * u64::from(m.sum) / ENERGY_AREA)
            .sum::<u64>();
    }
    energy
}

/// Samples of an [`ac_energy`] block.
const ENERGY_AREA: u64 = (ENERGY_BLOCK * ENERGY_BLOCK) as u64;

/// The block side of [`ac_energy`].
const ENERGY_BLOCK: usize = 8;

/// The largest block side of the SSE kernels.
const MAX_SSE_BLOCK: usize = 64;
