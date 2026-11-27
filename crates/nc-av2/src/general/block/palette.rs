//! Luma palettes (AV2 has no chroma palette): the colours, partly reused
//! from the above and left blocks' palettes, the colour index map, and the
//! prediction from it.
//!
//! Ported from dav2d (commit 237229e, BSD-2-Clause, see
//! `docs/licenses/dav2d`): `read_pal_plane` and `copy_pal_block_y`
//! (`src/recon_tmpl.c`), `read_pal_indices` (`src/decode.c`),
//! `pal_idx_finish` (`src/pal.c`); AVM `av2_get_palette_cache`
//! (`av2/common/pred_common.c`) for the cache order.
use super::Tile;
use super::context;
use super::size::BlockSize;
use crate::general::sample::Av2Sample;
use nc_common::decode::{DecodeError, Result};

/// Palettes hold 2..=8 colours.
pub const MAX_COLORS: usize = 8;
const MIN_COLORS: usize = 2;
/// The first delta's width is the colour's (the sample depth) less this,
/// plus a coded extra of this many bits (dav2d `read_pal_plane`'s `bpc - 3`).
const DELTA_BITS_BELOW_DEPTH: u32 = 3;
const DELTA_EXTRA_BITS: u32 = 2;
/// The above palette is reused only inside the same 64x64 row band (AVM
/// `MIN_SB_SIZE_LOG2`), in 4x4 units.
const REUSE_ROW_UNITS: u32 = 16;
/// Blocks whose larger side is below this many samples may code their
/// index map column by column.
const DIRECTION_MAX_SIDE: usize = 64;
/// The largest palette block side in samples, and the packed index map
/// (two indices per byte).
pub const MAX_SIDE: usize = 64;
pub const PACKED_INDICES: usize = MAX_SIDE * MAX_SIDE / 2;
/// Identity row symbols past "each index coded" (0): one index repeated
/// across the row, a copy of the row before; the first row's context.
const ROW_REPEATED: usize = 1;
const ROW_COPIED: usize = 2;
const FIRST_ROW_CONTEXT: usize = 3;
/// The index of a packed byte's second sample.
const INDEX_BITS: u32 = 4;

/// A block's palette (dav2d `b->pal_sz`, `scratch.pal`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Palette {
    /// 0 without a palette, else 2..=8.
    pub size: u8,
    /// The colours in ascending order, samples of the sequence's depth.
    pub colors: [u16; MAX_COLORS],
}

impl<T: Av2Sample> Tile<'_, '_, T> {
    /// A luma palette's size and colours (dav2d `read_pal_plane`): reuse
    /// flags over the neighbours' colours, then new colours as ascending
    /// deltas; the palette is the sorted union.
    pub(super) fn palette_colors(&mut self, depth: usize) -> Result<Palette> {
        let size = self.symbol(|m| &mut m.palette_size)? + MIN_COLORS;
        let column = self.column as usize;
        let row = context::left_index(self.row);
        let above_count = if self.row.is_multiple_of(REUSE_ROW_UNITS) {
            0
        } else {
            usize::from(self.contexts.above.palette_size[column])
        };
        let left_count = usize::from(self.contexts.left.palette_size[row]);
        let above = self.contexts.above.palette[column];
        let left = self.contexts.left.palette[row];
        // AVM's cache: above and left interleaved, then the longer one's rest.
        let mut cache = Vec::with_capacity(2 * MAX_COLORS);
        for i in 0..above_count.max(left_count) {
            if i < above_count {
                cache.push(above[i]);
            }
            if i < left_count {
                cache.push(left[i]);
            }
        }
        let mut colors = [0; MAX_COLORS];
        let mut reuse_mask = 0_u32;
        let (mut read, mut used) = (0, 0);
        let mut count = cache.len().min(size);
        while count != 0 {
            let flags = self.reader.literal(count as u32)?;
            for bit in (0..count).rev() {
                if flags & (1 << bit) != 0 {
                    colors[used] = cache[read + count - 1 - bit];
                    used += 1;
                }
            }
            reuse_mask = (reuse_mask << count) | flags;
            read += count;
            count = (cache.len() - read).min(size - used);
        }
        if used < size {
            self.new_colors(&mut colors[used..size])?;
        }
        colors[..size].sort_unstable();
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-ypal[sz={size},cache_sz={},mask={reuse_mask:0read$b}|{used}]: r={range}, cache={}, pal={}",
            cache.len(),
            hex_list(&cache),
            hex_list(&colors[..size])
        );
        Ok(Palette {
            size: size as u8,
            colors,
        })
    }

    /// Colours not taken from the cache: the first as a literal, each
    /// next as a delta above it, its width shrinking with the room left.
    fn new_colors(&mut self, colors: &mut [u16]) -> Result<()> {
        let color_bits = self.depth().bits();
        let mut previous = self.reader.literal(color_bits)?;
        colors[0] = previous as u16;
        if colors.len() == 1 {
            return Ok(());
        }
        let mut bits =
            color_bits - DELTA_BITS_BELOW_DEPTH + self.reader.literal(DELTA_EXTRA_BITS)?;
        let max = self.depth().max() as u32;
        for i in 1..colors.len() {
            let delta = self.reader.literal(bits)?;
            previous = (previous + delta + 1).min(max);
            colors[i] = previous as u16;
            if previous + 1 >= max {
                colors[i..].fill(max as u16);
                break;
            }
            bits = bits.min(1 + (max - previous - 1).ilog2());
        }
        Ok(())
    }

    /// The colour index map of a palette block (dav2d `read_pal_indices`,
    /// then `pal_idx_finish`), packed into `self.palette_indices`.
    pub(super) fn palette_index_map(&mut self, size: BlockSize, colors: usize) -> Result<()> {
        let (bw4, bh4) = size.units();
        let block = ((bw4 as usize) << 2, (bh4 as usize) << 2);
        let visible = (
            ((bw4.min(self.geometry.columns - self.column)) as usize) << 2,
            ((bh4.min(self.geometry.rows - self.row)) as usize) << 2,
        );
        let mut map = std::mem::take(&mut self.palette_map);
        map.clear();
        map.resize(block.0 * block.1, 0);
        let result = self.read_index_map(&mut map, colors, block, visible);
        if result.is_ok() {
            pack_indices(&mut self.palette_indices, &map, block, visible);
        }
        self.palette_map = map;
        result
    }

    /// The visible part of the index map, row-major `bw` apart: lines
    /// (rows, or columns when signalled) are each coded, one index
    /// repeated, or copied from the line before; coded indices after the
    /// first take a context from their left, top and top-left neighbours
    /// along the coding direction.
    fn read_index_map(
        &mut self,
        map: &mut [u8],
        colors: usize,
        (bw, bh): (usize, usize),
        (w, h): (usize, usize),
    ) -> Result<()> {
        let by_columns = bw.max(bh) < DIRECTION_MAX_SIDE && self.bypass()?;
        // Lines along the coding direction, positions along each line.
        let (line_step, step) = if by_columns { (1, bw) } else { (bw, 1) };
        let (lines, length) = if by_columns { (w, h) } else { (h, w) };
        let mut identity = self.symbol(|m| &mut m.palette_identity[FIRST_ROW_CONTEXT])?;
        if identity == ROW_COPIED {
            return Err(DecodeError::Malformed("AV2 palette copies a missing row"));
        }
        // A palette has at most 8 colours.
        let mut first = self.uniform(colors as u32)? as u8;
        map[0] = first;
        if identity == ROW_REPEATED {
            for m in 1..length {
                map[m * step] = first;
            }
        } else {
            let mut previous = first;
            for m in 1..length {
                previous = self.index_beside(colors, previous)?;
                map[m * step] = previous;
            }
        }
        for n in 1..lines {
            let line = n * line_step;
            identity = self.symbol(|m| &mut m.palette_identity[identity])?;
            if identity == ROW_COPIED {
                for m in 0..length {
                    map[line + m * step] = map[line - line_step + m * step];
                }
                continue;
            }
            let next = self.index_beside(colors, first)?;
            map[line] = next;
            if identity == ROW_REPEATED {
                for m in 1..length {
                    map[line + m * step] = next;
                }
            } else {
                let (mut top_left, mut left) = (first, next);
                for m in 1..length {
                    let top = map[line - line_step + m * step];
                    let index = self.coded_index(colors, by_columns, [left, top, top_left])?;
                    map[line + m * step] = index;
                    (left, top_left) = (index, top);
                }
            }
            first = next;
        }
        Ok(())
    }

    /// An index coded against one neighbour (context 0): 0 repeats it,
    /// the rest name the other colours in order.
    fn index_beside(&mut self, colors: usize, neighbour: u8) -> Result<u8> {
        let v = self.palette_index(colors, 0)?;
        let index = if v == 0 {
            neighbour
        } else {
            v - u8::from(v <= neighbour)
        };
        if usize::from(index) >= colors {
            return Err(DecodeError::Malformed("AV2 palette index"));
        }
        Ok(index)
    }

    /// An index after the first of a coded line, from its context (dav2d
    /// `read_pal_indices`' `ctx` switch): the symbols name the likeliest
    /// neighbours first, then the remaining colours in order.
    fn coded_index(
        &mut self,
        colors: usize,
        by_columns: bool,
        [left, top, top_left]: [u8; 3],
    ) -> Result<u8> {
        let context = if top == left {
            3 + usize::from(top_left == left)
        } else {
            1 + usize::from(top == top_left || left == top_left)
        };
        let v = self.palette_index(colors, context)?;
        let below = |value: u8, bound: u8| u8::from(value <= bound);
        let index = match context {
            1 => match v {
                0 | 1 => {
                    if (v == 1) == by_columns {
                        left
                    } else {
                        top
                    }
                }
                2 => top_left,
                _ => {
                    let s1 = u8::from(left < top);
                    let s2 = u8::from(left < top_left);
                    let s3 = u8::from(top < top_left);
                    v.wrapping_sub(below(v, left + s1 + s2))
                        .wrapping_sub(below(v, top + s3 + (1 - s1)))
                        .wrapping_sub(below(v, top_left + (1 - s2) + (1 - s3)))
                }
            },
            2 => {
                let left_or_top = (left + top).wrapping_sub(top_left);
                match v {
                    0 => top_left,
                    1 => left_or_top,
                    _ => {
                        let s = u8::from(left_or_top < top_left);
                        v.wrapping_sub(below(v, left_or_top.wrapping_add(s)))
                            .wrapping_sub(below(v, top_left + (1 - s)))
                    }
                }
            }
            3 => match v {
                0 => left,
                1 => top_left,
                _ => {
                    let s = u8::from(left < top_left);
                    v - below(v, left + s) - below(v, top_left + (1 - s))
                }
            },
            _ => {
                if v == 0 {
                    left
                } else {
                    v - below(v, left)
                }
            }
        };
        if usize::from(index) >= colors {
            return Err(DecodeError::Malformed("AV2 palette index"));
        }
        Ok(index)
    }

    /// A palette index symbol of a `colors`-colour palette.
    fn palette_index(&mut self, colors: usize, context: usize) -> Result<u8> {
        let symbol = match colors {
            2 => self.symbol(|m| &mut m.palette_index2[context])?,
            3 => self.symbol(|m| &mut m.palette_index3[context])?,
            4 => self.symbol(|m| &mut m.palette_index4[context])?,
            5 => self.symbol(|m| &mut m.palette_index5[context])?,
            6 => self.symbol(|m| &mut m.palette_index6[context])?,
            7 => self.symbol(|m| &mut m.palette_index7[context])?,
            _ => self.symbol(|m| &mut m.palette_index8[context])?,
        };
        Ok(symbol as u8)
    }

    /// Predict a palette block from its colours and packed index map
    /// (dav2d `recon_b`'s `pal_pred`).
    pub(super) fn predict_palette(&mut self, size: BlockSize, palette: &Palette) {
        let (bw4, bh4) = size.units();
        let offset = self.planes.offset(0, self.row, self.column);
        let stride = self.planes.stride[0];
        let colors = palette.colors.map(|color| T::from_i32(i32::from(color)));
        nc_dsp::av2_dec::ipred::pal_pred(
            &mut self.planes.data[0],
            offset,
            stride,
            &colors,
            &self.palette_indices,
            (bw4 as usize) << 2,
            (bh4 as usize) << 2,
        );
    }
}

/// Pack a `bw` x `bh` index map two per byte, repeating the last visible
/// column and row (`w` x `h` visible) over the rest (dav2d
/// `pal_idx_finish`).
fn pack_indices(
    packed: &mut Vec<u8>,
    map: &[u8],
    (bw, bh): (usize, usize),
    (w, h): (usize, usize),
) {
    let (row_bytes, visible_bytes) = (bw / 2, w / 2);
    packed.clear();
    for line in map.chunks_exact(bw).take(h) {
        packed.extend(
            line[..w]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[left, right]| left | (right << INDEX_BITS)),
        );
        let edge = line[w - 1] | (line[w - 1] << INDEX_BITS);
        packed.extend(std::iter::repeat_n(edge, row_bytes - visible_bytes));
    }
    let last = packed.len() - row_bytes;
    for _ in h..bh {
        packed.extend_from_within(last..last + row_bytes);
    }
}

/// `[aa,bb,...]` in two-digit hex, as dav2d's trace prints colours.
fn hex_list(values: &[u16]) -> String {
    let items: Vec<String> = values.iter().map(|v| format!("{v:02x}")).collect();
    format!("[{}]", items.join(","))
}

#[cfg(test)]
mod tests {
    use super::pack_indices;

    #[test]
    fn packing_repeats_the_visible_edge() {
        // A 8x4 block with 4x2 visible: indices 0..=3 in the first row.
        let mut map = vec![0; 8 * 4];
        map[..4].copy_from_slice(&[0, 1, 2, 3]);
        map[8..12].copy_from_slice(&[3, 2, 1, 0]);
        let mut packed = Vec::new();
        pack_indices(&mut packed, &map, (8, 4), (4, 2));
        assert_eq!(
            packed,
            [
                0x10, 0x32, 0x33, 0x33, 0x23, 0x01, 0x00, 0x00, 0x23, 0x01, 0x00, 0x00, 0x23, 0x01,
                0x00, 0x00
            ]
        );
    }
}
