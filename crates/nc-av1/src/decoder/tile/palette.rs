//! Palette mode (AV1 §5.11.46 `palette_mode_info`, §5.11.49
//! `palette_tokens`, §5.11.50 `get_palette_color_context`, §7.11.4
//! palette prediction): a block of a screen-content frame may predict its
//! luma, and its chroma, from up to eight colours, coded against a cache
//! of the colours of the blocks above and left, and a colour index per
//! sample coded in wavefront order.
//!
//! Follows libaom `read_palette_mode_info`, `av1_get_palette_cache`,
//! `decode_color_map_tokens` and `av1_get_palette_color_index_context`
//! (`av1/decoder/decodemv.c`, `av1/common/pred_common.c`,
//! `av1/decoder/detokenize.c`, `av1/common/entropymode.c`; BSD-2-Clause,
//! see `THIRD_PARTY.md`).
use super::super::block_size::BlockSize;
use super::super::cdf::Cdfs;
use super::BlockPos;
use crate::decoder_range::{Cdf, SymbolReader};
use nc_common::{PlaneIndex, PlaneViewMut};

/// `PALETTE_MAX_SIZE`, `PALETTE_MIN_SIZE`: colours per palette.
pub(super) const PALETTE_MAX_SIZE: usize = 8;
const PALETTE_MIN_SIZE: usize = 2;
/// `palette_size_y_minus_2` alphabet.
const PALETTE_SIZES: usize = 7;
/// Largest side of a palette block (`MAX_PALETTE_BLOCK_WIDTH`).
const PALETTE_MAX_BLOCK: usize = 64;
/// Samples are 8-bit (`BitDepth`).
const BIT_DEPTH: u32 = 8;
/// Bits of `palette_num_extra_bits_*` and its base below `BitDepth`.
const EXTRA_BITS: u32 = 2;
const DELTA_BITS_BELOW_DEPTH: u32 = 3;
const V_DELTA_BITS_BELOW_DEPTH: u32 = 4;
/// The cache skips the block above across a 64-sample row boundary
/// (4x4 rows per such row).
const CACHE_ROW_UNITS: usize = 16;
/// 4x4 units of a 128x128 superblock side (the left context's reach).
const MAX_SUPERBLOCK_MI: usize = 32;
/// The colour-context neighbours (left, above-left, above), their score
/// weights and hash multipliers, and the context of each hash.
const NEIGHBOURS: usize = 3;
const NEIGHBOUR_WEIGHTS: [u8; NEIGHBOURS] = [2, 1, 2];
const HASH_MULTIPLIERS: [u8; NEIGHBOURS] = [1, 2, 2];
const CONTEXT_OF_HASH: [Option<usize>; 9] = [
    None,
    None,
    Some(0),
    None,
    None,
    Some(4),
    Some(3),
    Some(2),
    Some(1),
];

/// One plane's palette: its colours, ascending for luma and U.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Palette {
    pub size: u8,
    pub colors: [u8; PALETTE_MAX_SIZE],
}

impl Palette {
    fn colors(&self) -> &[u8] {
        &self.colors[..usize::from(self.size)]
    }
}

/// A block's palettes: luma, U and V (`PaletteSizeY`, `PaletteSizeUV`,
/// `palette_colors_y/u/v`).
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct BlockPalette {
    pub y: Palette,
    pub u: Palette,
    pub v: Palette,
}

impl BlockPalette {
    /// The palette `plane` predicts from, if it has one.
    fn of(&self, plane: PlaneIndex) -> Option<&Palette> {
        let palette = match plane {
            PlaneIndex::Y => &self.y,
            PlaneIndex::U => &self.u,
            PlaneIndex::V => &self.v,
        };
        (palette.size > 0).then_some(palette)
    }
}

/// The palette state of a screen-content frame: the luma and U palettes
/// of the blocks above and left (what the caches and the `palette_y_mode`
/// context read), the current block's palettes, and its colour maps.
#[derive(Debug)]
pub(super) struct PaletteState {
    above: Vec<[Palette; 2]>,
    left: [[Palette; 2]; MAX_SUPERBLOCK_MI],
    /// The block being decoded.
    pub block: BlockPalette,
    /// `ColorMapY`, `ColorMapUV`: one index per sample of the plane
    /// block, `stride` apart, from the plane block's top-left sample.
    maps: [Vec<u8>; 2],
    stride: [usize; 2],
    origin: [(usize, usize); 2],
}

impl PaletteState {
    /// The state of a frame `mi_cols` 4x4 units wide (with a superblock of
    /// spare units, so blocks at the edge write whole).
    pub(super) fn new(mi_cols: usize) -> Self {
        Self {
            above: vec![[Palette::default(); 2]; mi_cols + MAX_SUPERBLOCK_MI],
            left: [[Palette::default(); 2]; MAX_SUPERBLOCK_MI],
            block: BlockPalette::default(),
            maps: [
                vec![0; PALETTE_MAX_BLOCK * PALETTE_MAX_BLOCK],
                vec![0; PALETTE_MAX_BLOCK * PALETTE_MAX_BLOCK / 4],
            ],
            stride: [0; 2],
            origin: [(0, 0); 2],
        }
    }

    /// Record the current block's palettes over its 4x4 units, for the
    /// blocks after it.
    pub(super) fn record(&mut self, block: BlockPos) {
        let entry = [self.block.y, self.block.u];
        let (w4, h4) = block.units();
        self.above[block.mi_col..block.mi_col + w4].fill(entry);
        let row = block.mi_row % MAX_SUPERBLOCK_MI;
        self.left[row..(row + h4).min(MAX_SUPERBLOCK_MI)].fill(entry);
    }

    /// Record a block without palettes (inter and intra block copy
    /// blocks).
    // Out of line: our own streams never allow screen content tools.
    #[cold]
    #[inline(never)]
    pub(super) fn record_none(&mut self, block: BlockPos) {
        self.block = BlockPalette::default();
        self.record(block);
    }

    /// The above and left neighbours' palettes of `plane_kind` (0 luma, 1
    /// chroma) where available.
    fn neighbours(&self, block: BlockPos, plane_kind: usize) -> [Option<Palette>; 2] {
        let (above, left) = block.available(PlaneIndex::Y);
        [
            above.then(|| self.above[block.mi_col][plane_kind]),
            left.then(|| self.left[block.mi_row % MAX_SUPERBLOCK_MI][plane_kind]),
        ]
    }

    /// `get_palette_cache()`: the colours of the block above (unless
    /// across a 64-sample row) and left, merged ascending without
    /// repeats.
    fn cache(&self, block: BlockPos, plane_kind: usize) -> Vec<u8> {
        let [above, left] = self.neighbours(block, plane_kind);
        let above = above.filter(|_| !block.mi_row.is_multiple_of(CACHE_ROW_UNITS));
        let (above, left) = (
            above.as_ref().map_or(&[][..], Palette::colors),
            left.as_ref().map_or(&[][..], Palette::colors),
        );
        let mut cache: Vec<u8> = Vec::with_capacity(2 * PALETTE_MAX_SIZE);
        let (mut a, mut l) = (0, 0);
        while a < above.len() || l < left.len() {
            let take_left = a == above.len() || (l < left.len() && left[l] < above[a]);
            let value = if take_left {
                l += 1;
                left[l - 1]
            } else {
                // Equal colours advance both lists.
                if l < left.len() && left[l] == above[a] {
                    l += 1;
                }
                a += 1;
                above[a - 1]
            };
            if cache.last() != Some(&value) {
                cache.push(value);
            }
        }
        cache
    }

    /// `palette_mode_info()` of `block` (§5.11.46), whose luma mode is DC
    /// (`y_dc`) and chroma mode DC (`uv_dc`, only where it carries
    /// chroma), into [`Self::block`].
    pub(super) fn read_mode_info(
        &mut self,
        r: &mut SymbolReader<'_>,
        cdfs: &mut Cdfs,
        block: BlockPos,
        y_dc: bool,
        uv_dc: bool,
    ) {
        let size_ctx = usize::from(block.size.width_log2() + block.size.height_log2())
            - usize::from(2 * BlockSize::BLOCK_8X8.width_log2());
        self.block = BlockPalette::default();
        if y_dc {
            let context = self
                .neighbours(block, 0)
                .iter()
                .filter(|n| n.is_some_and(|p| p.size > 0))
                .count();
            if r.boolean(&mut cdfs.palette_y_mode[size_ctx][context]) {
                let size = read_size(r, &mut cdfs.palette_y_size[size_ctx]);
                let cache = self.cache(block, 0);
                self.block.y = read_colors(r, size, &cache, ColorCoding::Luma);
            }
        }
        if uv_dc && block.has_chroma() {
            let context = usize::from(self.block.y.size > 0);
            if r.boolean(&mut cdfs.palette_uv_mode[context]) {
                let size = read_size(r, &mut cdfs.palette_uv_size[size_ctx]);
                let cache = self.cache(block, 1);
                self.block.u = read_colors(r, size, &cache, ColorCoding::U);
                self.block.v = read_v_colors(r, size);
            }
        }
    }

    /// `palette_tokens()` (§5.11.49) of the current block's palettes:
    /// the colour map of each plane that has one. `frame_mi` is the
    /// frame's size in 4x4 units.
    pub(super) fn read_tokens(
        &mut self,
        r: &mut SymbolReader<'_>,
        cdfs: &mut Cdfs,
        block: BlockPos,
        frame_mi: (usize, usize),
    ) {
        let (bw, bh) = (block.size.width(), block.size.height());
        let onscreen = (
            bw.min((frame_mi.0 - block.mi_col) * 4),
            bh.min((frame_mi.1 - block.mi_row) * 4),
        );
        if self.block.y.size > 0 {
            let size = usize::from(self.block.y.size);
            self.stride[0] = bw;
            self.origin[0] = (block.mi_col * 4, block.mi_row * 4);
            read_map(
                r,
                &mut cdfs.palette_y_color_index[size - PALETTE_MIN_SIZE],
                &mut self.maps[0],
                size,
                (bw, bh),
                onscreen,
            );
        }
        if self.block.u.size > 0 {
            let size = usize::from(self.block.u.size);
            // A 4-sample-wide (high) block's chroma covers its 8x8 area,
            // which starts a unit left (up) (§5.11.49: `+ 2`).
            let widen = |luma_side: usize| if luma_side < 8 { 2 } else { 0 };
            let (cw, ch) = (bw / 2 + widen(bw), bh / 2 + widen(bh));
            let on = (onscreen.0 / 2 + widen(bw), onscreen.1 / 2 + widen(bh));
            let col = block.mi_col - usize::from(block.size.width_mi() == 1);
            let row = block.mi_row - usize::from(block.size.height_mi() == 1);
            self.stride[1] = cw;
            self.origin[1] = (col * 2, row * 2);
            read_map(
                r,
                &mut cdfs.palette_uv_color_index[size - PALETTE_MIN_SIZE],
                &mut self.maps[1],
                size,
                (cw, ch),
                on,
            );
        }
    }

    /// Predict the transform block of `plane` at plane sample `origin`
    /// into `dst` from the current block's palette (§7.11.4), if `plane`
    /// has one; returns whether it did.
    pub(super) fn predict(
        &self,
        plane: PlaneIndex,
        (x, y): (usize, usize),
        dst: &mut PlaneViewMut<'_>,
    ) -> bool {
        let Some(palette) = self.block.of(plane) else {
            return false;
        };
        let kind = usize::from(plane != PlaneIndex::Y);
        let (ox, oy) = self.origin[kind];
        let stride = self.stride[kind];
        let (w, h) = (dst.width(), dst.height());
        for r in 0..h {
            let start = (y - oy + r) * stride + (x - ox);
            let indices = &self.maps[kind][start..start + w];
            for (sample, &index) in dst.row_mut(r).iter_mut().zip(indices) {
                *sample = palette.colors[usize::from(index)];
            }
        }
        true
    }
}

/// `palette_size_*_minus_2` + 2.
fn read_size(r: &mut SymbolReader<'_>, cdf: &mut Cdf<7>) -> u8 {
    (r.symbol(cdf, PALETTE_SIZES) + PALETTE_MIN_SIZE) as u8
}

/// How a palette's coded colours step: luma by at least one, U by at
/// least zero.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColorCoding {
    Luma,
    U,
}

/// The colours of a luma or U palette of `size` (§5.11.46): flags for
/// the cached colours used, then the rest ascending as a literal and
/// deltas, merged with the cached ones in order.
fn read_colors(r: &mut SymbolReader<'_>, size: u8, cache: &[u8], coding: ColorCoding) -> Palette {
    let n = usize::from(size);
    let mut cached = Vec::with_capacity(n);
    for &color in cache {
        if cached.len() == n {
            break;
        }
        if r.literal(1) == 1 {
            cached.push(color);
        }
    }
    let mut coded = Vec::with_capacity(n);
    if cached.len() < n {
        let max = (1 << BIT_DEPTH) - 1;
        let first = r.literal(BIT_DEPTH);
        coded.push(first);
        if cached.len() + 1 < n {
            let (step, mut range) = match coding {
                ColorCoding::Luma => (1, max - first),
                ColorCoding::U => (0, max + 1 - first),
            };
            let mut bits = BIT_DEPTH - DELTA_BITS_BELOW_DEPTH + r.literal(EXTRA_BITS);
            while cached.len() + coded.len() < n {
                let previous = *coded.last().expect("one coded colour");
                let color = (previous + r.literal(bits) + step).min(max);
                range -= color - previous;
                bits = bits.min(ceil_log2(range));
                coded.push(color);
            }
        }
    }
    // Both lists ascend: merge them.
    let mut palette = Palette {
        size,
        colors: [0; PALETTE_MAX_SIZE],
    };
    let (mut c, mut k) = (0, 0);
    for slot in &mut palette.colors[..n] {
        let take_cached =
            c < cached.len() && (k == coded.len() || u32::from(cached[c]) <= coded[k]);
        *slot = if take_cached {
            c += 1;
            cached[c - 1]
        } else {
            k += 1;
            coded[k - 1] as u8
        };
    }
    palette
}

/// The V colours of a chroma palette of `size` (§5.11.46): signed deltas
/// wrapping around the sample range, or literals.
fn read_v_colors(r: &mut SymbolReader<'_>, size: u8) -> Palette {
    let mut palette = Palette {
        size,
        colors: [0; PALETTE_MAX_SIZE],
    };
    let colors = &mut palette.colors[..usize::from(size)];
    if r.literal(1) == 1 {
        let max = 1i32 << BIT_DEPTH;
        let bits = BIT_DEPTH - V_DELTA_BITS_BELOW_DEPTH + r.literal(EXTRA_BITS);
        colors[0] = r.literal(BIT_DEPTH) as u8;
        for i in 1..colors.len() {
            let mut delta = r.literal(bits) as i32;
            if delta != 0 && r.literal(1) == 1 {
                delta = -delta;
            }
            colors[i] = (i32::from(colors[i - 1]) + delta).rem_euclid(max) as u8;
        }
    } else {
        for color in colors.iter_mut() {
            *color = r.literal(BIT_DEPTH) as u8;
        }
    }
    palette
}

/// `CeilLog2(x)`.
fn ceil_log2(x: u32) -> u32 {
    if x < 2 {
        0
    } else {
        32 - (x - 1).leading_zeros()
    }
}

/// `NS(n)` (§4.10.10) read with the symbol reader's literal bits.
fn read_uniform(r: &mut SymbolReader<'_>, n: usize) -> usize {
    let bits = usize::BITS - n.leading_zeros();
    let short_values = (1 << bits) - n;
    let value = r.literal(bits - 1) as usize;
    if value < short_values {
        return value;
    }
    (value << 1) - short_values + r.literal(1) as usize
}

/// One plane's colour map (§5.11.49): the first index, then the rest in
/// wavefront order (anti-diagonals, each up and right) with the contexts
/// of §5.11.50; columns and rows beyond the frame repeat the last coded
/// ones.
fn read_map(
    r: &mut SymbolReader<'_>,
    cdfs: &mut [Cdf<8>; 5],
    map: &mut [u8],
    size: usize,
    (width, height): (usize, usize),
    (cols, rows): (usize, usize),
) {
    map[0] = read_uniform(r, size) as u8;
    for diagonal in 1..rows + cols - 1 {
        for col in (diagonal.saturating_sub(rows - 1)..=diagonal.min(cols - 1)).rev() {
            let row = diagonal - col;
            let (context, order) = color_context(map, width, (row, col), size);
            let index = r.symbol(&mut cdfs[context], size);
            map[row * width + col] = order[index];
        }
    }
    for row in 0..rows {
        let last = map[row * width + cols - 1];
        map[row * width + cols..(row + 1) * width].fill(last);
    }
    for row in rows..height {
        map.copy_within((rows - 1) * width..rows * width, row * width);
    }
}

/// `get_palette_color_context()` (§5.11.50) of the sample at (`row`,
/// `col`): the colour order (neighbour colours by score first) and the
/// context of its score hash.
fn color_context(
    map: &[u8],
    stride: usize,
    (row, col): (usize, usize),
    size: usize,
) -> (usize, [u8; PALETTE_MAX_SIZE]) {
    let neighbours = [
        (col > 0).then(|| map[row * stride + col - 1]),
        (col > 0 && row > 0).then(|| map[(row - 1) * stride + col - 1]),
        (row > 0).then(|| map[(row - 1) * stride + col]),
    ];
    let mut scores = [0u8; PALETTE_MAX_SIZE];
    for (neighbour, weight) in neighbours.iter().zip(NEIGHBOUR_WEIGHTS) {
        if let Some(index) = neighbour {
            scores[usize::from(*index)] += weight;
        }
    }
    let mut order: [u8; PALETTE_MAX_SIZE] = std::array::from_fn(|i| i as u8);
    // Move the best-scoring colours to the front, keeping the order of
    // equal scores (a stable partial selection sort).
    for i in 0..NEIGHBOURS {
        let mut best = i;
        for j in i + 1..size {
            if scores[j] > scores[best] {
                best = j;
            }
        }
        scores[i..=best].rotate_right(1);
        order[i..=best].rotate_right(1);
    }
    let hash: usize = scores
        .iter()
        .zip(HASH_MULTIPLIERS)
        .map(|(&score, multiplier)| usize::from(score * multiplier))
        .sum();
    let context = CONTEXT_OF_HASH[hash].expect("a valid colour context hash");
    (context, order)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §5.11.50: one left neighbour (score 2) gives hash 2, context 0; all
    /// three the same colour (score 5) gives hash 5, context 4; that colour
    /// moves to the front.
    #[test]
    fn color_contexts_follow_the_neighbour_scores() {
        let map = [3, 3, 3, 0];
        assert_eq!(
            color_context(&map, 2, (0, 1), 4),
            (0, [3, 0, 1, 2, 4, 5, 6, 7])
        );
        assert_eq!(
            color_context(&map, 2, (1, 1), 4),
            (4, [3, 0, 1, 2, 4, 5, 6, 7])
        );
    }

    #[test]
    fn ceil_log2_rounds_up() {
        assert_eq!(
            [0, 1, 2, 3, 4, 5, 255].map(ceil_log2),
            [0, 0, 1, 2, 2, 3, 8]
        );
    }
}
