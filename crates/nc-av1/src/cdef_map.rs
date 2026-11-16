//! Which 8x8 units of a frame CDEF filters and with which strength set,
//! and the in-place application of CDEF from them (AV1 §7.15): shared by
//! the encoder ([`crate::cdef`]) and the decoder. The block layer records
//! each block coded with a residual ([`CdefMap::coded_block`], libaom
//! `is_8x8_block_skip`) as it codes or reads its `skip`, so no pass over
//! the mode-info grid is needed to find the units; each 64x64 filter
//! block carries its `cdef_idx` (§5.11.56, -1 where no block of it codes
//! one). Any number of strength sets (`cdef_bits` up to 3) is supported;
//! the filter and direction search are nc-dsp's.

use nc_common::Picture;
use nc_dsp::{CdefLines, CdefSets, CdefStrengths, CdefUnit};
use nc_dsp::{DepthKernels, Pixel};

/// 4x4 units per 8x8 CDEF unit and per 64x64 filter block, log2.
const UNIT_MI_LOG2: usize = 1;
const FILTER_BLOCK_MI_LOG2: usize = 4;
/// Luma samples per 8x8 unit and per filter block (whose rows are a
/// band).
const UNIT_SIZE: usize = 8;
const FILTER_BLOCK_SIZE: usize = 64;
/// 8x8 units per filter block row.
const UNITS_PER_BLOCK: usize = FILTER_BLOCK_SIZE / UNIT_SIZE;
/// `cdef_idx` of a filter block without a coded index: not filtered.
const NO_INDEX: i8 = -1;

/// The per-frame CDEF state the block layer fills.
#[derive(Debug, Default)]
pub(crate) struct CdefMap {
    /// Filter blocks per row.
    columns: usize,
    /// `cdef_idx` per filter block.
    index: Vec<i8>,
    /// 8x8 units per row.
    unit_columns: usize,
    /// Units with a non-skipped 4x4 block (dav1d's `noskip_mask`): per row
    /// of units, one byte per filter block, bit `i` for its unit column `i`.
    coded: Vec<u8>,
}

impl CdefMap {
    /// Clear the map for a frame of `mi_cols` x `mi_rows` 4x4 units.
    pub(crate) fn start_frame(&mut self, mi_cols: usize, mi_rows: usize) {
        let blocks = |mi: usize| mi.div_ceil(1 << FILTER_BLOCK_MI_LOG2);
        let units = |mi: usize| mi.div_ceil(1 << UNIT_MI_LOG2);
        self.columns = blocks(mi_cols);
        self.index.clear();
        self.index.resize(self.columns * blocks(mi_rows), NO_INDEX);
        self.unit_columns = units(mi_cols);
        self.coded.clear();
        self.coded.resize(self.columns * units(mi_rows), 0);
    }
    /// Whether the filter block holding 4x4 unit (`mi_col`, `mi_row`)
    /// still needs its `cdef_idx`.
    pub(crate) fn needs_index(&self, mi_col: usize, mi_row: usize) -> bool {
        self.index[self.block(mi_col, mi_row)] == NO_INDEX
    }
    /// Set `cdef_idx` of every filter block a block of `w4` x `h4` units
    /// at (`mi_col`, `mi_row`) covers (a 128-wide block covers several).
    pub(crate) fn set_index(
        &mut self,
        (mi_col, mi_row): (usize, usize),
        (w4, h4): (usize, usize),
        index: i8,
    ) {
        let step = 1 << FILTER_BLOCK_MI_LOG2;
        for row in (mi_row..mi_row + h4).step_by(step) {
            for col in (mi_col..mi_col + w4).step_by(step) {
                if let Some(i) = self.block_inside(col, row) {
                    self.index[i] = index;
                }
            }
        }
    }
    /// Record a non-skipped block of `w4` x `h4` units at (`mi_col`,
    /// `mi_row`).
    #[allow(
        clippy::extra_unused_type_parameters,
        reason = "one copy per decoder instance, kept inlined (docs/handoff/high-bit-depth.md)"
    )]
    pub(crate) fn coded_block<T>(
        &mut self,
        (mi_col, mi_row): (usize, usize),
        (w4, h4): (usize, usize),
    ) {
        let first = mi_col >> UNIT_MI_LOG2;
        let last = ((mi_col + w4 - 1) >> UNIT_MI_LOG2).min(self.unit_columns - 1);
        let rows = self.coded.len() / self.columns;
        let unit_rows =
            (mi_row >> UNIT_MI_LOG2)..=((mi_row + h4 - 1) >> UNIT_MI_LOG2).min(rows - 1);
        for column in first..=last {
            let (block, bit) = (column / UNITS_PER_BLOCK, column % UNITS_PER_BLOCK);
            for row in unit_rows.clone() {
                self.coded[row * self.columns + block] |= 1 << bit;
            }
        }
    }
    /// Give filter block `i` (raster order) the index `index(i)`: an
    /// encoder that decides its indices after coding the blocks.
    pub(crate) fn set_block_indices(&mut self, mut index: impl FnMut(usize) -> i8) {
        for (block, value) in self.index.iter_mut().enumerate() {
            *value = index(block);
        }
    }
    /// The top-left luma samples of the coded 8x8 units, in raster order.
    pub(crate) fn coded_units(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.coded
            .chunks(self.columns.max(1))
            .enumerate()
            .flat_map(|(unit_row, row)| {
                row.iter().enumerate().flat_map(move |(block, &mask)| {
                    BitPositions(mask).map(move |bit| {
                        let column = block * UNITS_PER_BLOCK + bit;
                        (column * UNIT_SIZE, unit_row * UNIT_SIZE)
                    })
                })
            })
    }
    fn block(&self, mi_col: usize, mi_row: usize) -> usize {
        (mi_row >> FILTER_BLOCK_MI_LOG2) * self.columns + (mi_col >> FILTER_BLOCK_MI_LOG2)
    }
    fn block_inside(&self, mi_col: usize, mi_row: usize) -> Option<usize> {
        let i = self.block(mi_col, mi_row);
        ((mi_col >> FILTER_BLOCK_MI_LOG2) < self.columns && i < self.index.len()).then_some(i)
    }
}

/// The set bits of a unit mask, lowest first.
struct BitPositions(u8);

impl Iterator for BitPositions {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        (self.0 != 0).then(|| {
            let bit = self.0.trailing_zeros() as usize;
            self.0 &= self.0 - 1;
            bit
        })
    }
}

/// Reusable buffers of [`apply`]: the line buffers of the band being
/// filtered and its unit list.
#[derive(Debug, Default)]
pub(crate) struct CdefScratch<T: Pixel> {
    lines: CdefLines<T>,
    units: Vec<CdefUnit>,
}

/// Apply CDEF to `recon` in place with the strength sets `strengths`
/// (damping `damping`), each filter block taking the set its index in
/// `map` names, one 64-row band of filter blocks at a time (nc-dsp's
/// shared band driver, dav1d's `cdef_brow` structure). A band lists only
/// the coded units of its filter blocks whose strength set filters
/// anything.
pub(crate) fn apply<T: Pixel>(
    kernels: DepthKernels<'_>,
    recon: &mut Picture<T>,
    map: &CdefMap,
    strengths: &[CdefStrengths],
    damping: u8,
    scratch: &mut CdefScratch<T>,
) {
    let sets = CdefSets::new(strengths, damping);
    if !sets.filters() {
        return;
    }
    let CdefScratch { lines, units } = scratch;
    lines.start(recon);
    let unit_rows = map.coded.len() / map.columns;
    for (band, indices) in map.index.chunks(map.columns).enumerate() {
        units.clear();
        let first_row = band * UNITS_PER_BLOCK;
        for unit_row in first_row..(first_row + UNITS_PER_BLOCK).min(unit_rows) {
            let coded = &map.coded[unit_row * map.columns..][..map.columns];
            for (block, (&index, &mask)) in indices.iter().zip(coded).enumerate() {
                if index == NO_INDEX || !strengths[index as usize].filters() {
                    continue;
                }
                for bit in BitPositions(mask) {
                    let column = block * UNITS_PER_BLOCK + bit;
                    units.push(CdefUnit {
                        x: column * UNIT_SIZE,
                        y: unit_row * UNIT_SIZE,
                        set: index as u8,
                    });
                }
            }
        }
        let rows = band * FILTER_BLOCK_SIZE..((band + 1) * FILTER_BLOCK_SIZE).min(recon.height());
        kernels.cdef_filter_band(lines, recon, rows, units, &sets);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_cover_large_blocks_and_skips_are_per_8x8() {
        let mut map = CdefMap::default();
        // 160x72 luma: 3x2 filter blocks, 20x9 units.
        map.start_frame(40, 18);
        assert!(map.needs_index(17, 3));
        map.set_index((0, 0), (32, 32), 2);
        assert_eq!(map.index, vec![2, 2, -1, 2, 2, -1]);
        let units = |map: &CdefMap| map.coded.iter().map(|m| m.count_ones()).sum::<u32>();
        map.coded_block::<u8>((3, 5), (1, 1));
        // Unit row 2, column 1: filter block 0, bit 1.
        assert_eq!(map.coded[2 * 3], 1 << 1);
        assert_eq!(units(&map), 1);
        // Units 18..=19 of row 8 (the last): filter block 2, bits 2 and 3.
        map.coded_block::<u8>((36, 16), (8, 4));
        assert_eq!(map.coded[8 * 3 + 2], 0b1100);
        assert_eq!(units(&map), 3);
    }
}
