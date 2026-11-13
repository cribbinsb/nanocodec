//! CDEF of decoded frames (AV1 §7.15) from the decoder's own state: the
//! strength set each 64x64 filter block reads (`cdef_idx`, §5.11.56, -1
//! where no block of it codes one) and which 8x8 units have a non-skipped
//! 4x4 (libaom `is_8x8_block_skip`). Any number of strength sets
//! (`cdef_bits` up to 3) is supported; the filter and direction search are
//! nc-dsp's, as the encoder's [`crate::cdef`] uses them.
use super::frame_header::CdefParams;
use nc_common::Picture;
use nc_dsp::{CdefLines, CdefSets, CdefStrengths, CdefUnit, Dsp};

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
pub(super) struct CdefMap {
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
    pub(super) fn start_frame(&mut self, mi_cols: usize, mi_rows: usize) {
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
    pub(super) fn needs_index(&self, mi_col: usize, mi_row: usize) -> bool {
        self.index[self.block(mi_col, mi_row)] == NO_INDEX
    }
    /// Set `cdef_idx` of every filter block a block of `w4` x `h4` units
    /// at (`mi_col`, `mi_row`) covers (a 128-wide block covers several).
    pub(super) fn set_index(
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
    pub(super) fn coded_block(
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
    fn block(&self, mi_col: usize, mi_row: usize) -> usize {
        (mi_row >> FILTER_BLOCK_MI_LOG2) * self.columns + (mi_col >> FILTER_BLOCK_MI_LOG2)
    }
    fn block_inside(&self, mi_col: usize, mi_row: usize) -> Option<usize> {
        let i = self.block(mi_col, mi_row);
        ((mi_col >> FILTER_BLOCK_MI_LOG2) < self.columns && i < self.index.len()).then_some(i)
    }
}

/// Reusable buffers of [`apply`]: the line buffers of the band being
/// filtered and its unit list.
#[derive(Debug, Default)]
pub(super) struct CdefScratch {
    lines: CdefLines,
    units: Vec<CdefUnit>,
}

/// Apply CDEF to `recon` in place with the frame's strength sets, one
/// 64-row band of filter blocks at a time (nc-dsp's shared band driver,
/// dav1d's `cdef_brow` structure). A band lists only the coded units of
/// its filter blocks whose strength set filters anything.
pub(super) fn apply(
    dsp: &Dsp,
    recon: &mut Picture,
    map: &CdefMap,
    params: &CdefParams,
    scratch: &mut CdefScratch,
) {
    let strengths: Vec<CdefStrengths> = (0..1usize << params.bits)
        .map(|s| CdefStrengths {
            luma_primary: params.y_pri[s],
            luma_secondary: params.y_sec[s],
            chroma_primary: params.uv_pri[s],
            chroma_secondary: params.uv_sec[s],
        })
        .collect();
    let sets = CdefSets::new(&strengths, params.damping);
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
                let mut mask = mask;
                while mask != 0 {
                    let column = block * UNITS_PER_BLOCK + mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    units.push(CdefUnit {
                        x: column * UNIT_SIZE,
                        y: unit_row * UNIT_SIZE,
                        set: index as u8,
                    });
                }
            }
        }
        let rows = band * FILTER_BLOCK_SIZE..((band + 1) * FILTER_BLOCK_SIZE).min(recon.height());
        dsp.cdef_filter_band(lines, recon, rows, units, &sets);
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
        map.coded_block((3, 5), (1, 1));
        // Unit row 2, column 1: filter block 0, bit 1.
        assert_eq!(map.coded[2 * 3], 1 << 1);
        assert_eq!(units(&map), 1);
        // Units 18..=19 of row 8 (the last): filter block 2, bits 2 and 3.
        map.coded_block((36, 16), (8, 4));
        assert_eq!(map.coded[8 * 3 + 2], 0b1100);
        assert_eq!(units(&map), 3);
    }
}
