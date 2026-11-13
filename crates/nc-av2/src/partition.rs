//! Regular AV2 shared-tree partition traversal for SB64 and square leaves
//! from 64×64 to 8×8.
//!
//! Adapted from pinned AVM v1.0.0 `write_partition`, `partition_plane_context`,
//! and `is_partition_implied_at_boundary`. Below 128×128, AV2 uses binary
//! rectangular partitions: a four-way square split is not legal. Extended partitions, SDP, and screen-content modes must be disabled.
use crate::entropy::{Reader, Writer};
use nc_common::decode::{DecodeError, Result};
use nc_encoder::symbol_cost::symbol_bits;

const SPLIT: [u16; 16] = [
    4684, 9013, 9134, 13_400, 7807, 17_827, 16_614, 26_863, 10_834, 22_328, 20_784, 29_294, 12_276,
    25_805, 24_669, 31_239,
];
const RECT: [u16; 28] = [
    18_124, 22_595, 14_239, 16_697, 12_505, 19_955, 6156, 9491, 22_174, 25_768, 12_766, 19_879,
    18_914, 22_018, 14_388, 15_263, 18_338, 21_214, 12_690, 13_671, 17_490, 22_631, 10_847, 18_147,
    13_438, 16_847, 6550, 8450,
];

const MODE_SET: &[u16] = &[3905, 1746, 1044, 0];
const Y_DC: &[u16] = &[17_593, 12_693, 11_040, 8670, 6363, 5113, 3908, 0];
const UV_DC: &[u16] = &[23_405, 11_811, 9903, 8015, 6357, 4785, 2340, 0];
const DPCM: &[u16] = &[16_384, 0];

/// Write DC luma/chroma modes for a shared-tree chroma-reference block.
/// All previously coded neighbors must also use non-directional modes.
/// # Errors
/// Propagates entropy coding errors.
pub fn write_dc_modes(writer: &mut Writer, lossless: bool) -> Result<()> {
    if lossless {
        writer.model(0, DPCM, |m| &mut m.dpcm_y)?;
    }
    writer.model(0, MODE_SET, |m| &mut m.mode_set)?;
    writer.model(0, Y_DC, |m| &mut m.y_mode[0])?;
    if lossless {
        writer.model(0, DPCM, |m| &mut m.dpcm_uv)?;
    }
    writer.model(0, UV_DC, |m| &mut m.uv_mode[0])
}

/// Read the restricted DC modes emitted by [`write_dc_modes`].
/// # Errors
/// Rejects non-DC modes, DPCM, or a truncated entropy stream.
pub fn read_dc_modes(reader: &mut Reader<'_>, lossless: bool) -> Result<()> {
    if lossless {
        require(reader.model(DPCM, |m| &mut m.dpcm_y)?, 0)?;
    }
    require(reader.model(MODE_SET, |m| &mut m.mode_set)?, 0)?;
    require(reader.model(Y_DC, |m| &mut m.y_mode[0])?, 0)?;
    if lossless {
        require(reader.model(DPCM, |m| &mut m.dpcm_uv)?, 0)?;
    }
    require(reader.model(UV_DC, |m| &mut m.uv_mode[0])?, 0)
}

fn require(actual: usize, symbol: usize) -> Result<()> {
    if actual != symbol {
        return Err(DecodeError::Malformed(
            "AV2 unsupported regular partition/mode",
        ));
    }
    Ok(())
}

/// One partition node: a square, or a 2:1 rectangle produced by a binary split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Node {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

/// The entropy-coded part of one node's partition syntax.
#[derive(Debug, Clone, Copy)]
struct NodeSyntax {
    /// Split direction implied by the frame boundary (true: vertical).
    forced: Option<bool>,
    /// `do_split` context, when that symbol is coded.
    split_context: Option<usize>,
    /// `rect_type` context, when a split's direction is coded.
    rect_context: usize,
}

/// Smallest coding block; every node at this size is a leaf.
const MIN_LEAF: usize = 8;
/// Superblock side in luma samples (64x64; 128x128 superblocks are disabled).
pub const SUPERBLOCK: usize = 64;
/// Partition context units are 4x4 luma samples.
const CONTEXT_UNIT: usize = 4;
/// Left partition contexts cover one superblock height.
const LEFT_UNITS: usize = SUPERBLOCK / CONTEXT_UNIT;

/// Above/left partition contexts of one tile (AVM `partition_plane_context`,
/// `update_partition_context`), shared by encoder and decoder.
#[derive(Debug, Clone)]
struct Contexts {
    width: usize,
    height: usize,
    above: Vec<u8>,
    left: [u8; LEFT_UNITS],
}

impl Contexts {
    fn new(width: usize, height: usize) -> Result<Self> {
        if width == 0
            || height == 0
            || !width.is_multiple_of(MIN_LEAF)
            || !height.is_multiple_of(MIN_LEAF)
            || width > 65_536
            || height > 65_536
        {
            return Err(DecodeError::Malformed("AV2 regular partition dimensions"));
        }
        Ok(Self {
            width,
            height,
            above: vec![0; width / CONTEXT_UNIT],
            left: [0; LEFT_UNITS],
        })
    }

    fn contains(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    /// Boundary inference and contexts for `node` (AVM `read_partition`,
    /// `is_partition_implied_at_boundary`). Boundary inference precedes all
    /// partition symbols; for square and 2:1 nodes the inferred splits always
    /// lead back to the same square children.
    fn syntax(&self, node: Node) -> NodeSyntax {
        let Node { x, y, w, h } = node;
        let has_rows = y + h / 2 < self.height;
        let has_cols = x + w / 2 < self.width;
        let forced = if has_rows && has_cols {
            None
        } else if w == h {
            Some(has_rows && !has_cols)
        } else if h > w && (!has_rows || !has_cols && x + w / 4 >= self.width) {
            Some(false)
        } else if w > h && (!has_cols || !has_rows && y + h / 4 >= self.height) {
            Some(true)
        } else {
            None
        };
        let bw_log = (w / CONTEXT_UNIT).ilog2() as usize;
        let bh_log = (h / CONTEXT_UNIT).ilog2() as usize;
        let above = (self.above[x / CONTEXT_UNIT] >> bw_log.saturating_sub(1)) & 1;
        let left = (self.left[(y / CONTEXT_UNIT) % LEFT_UNITS] >> bh_log.saturating_sub(1)) & 1;
        let raw = usize::from(left * 2 + above);
        let size_ctx = bw_log.max(bh_log).saturating_sub(1);
        // Even when a split gives square children, the other rectangular
        // direction remains legal (it can split again).
        let rect_size = if w == h {
            match w {
                8 | 16 => 0,
                32 => 3,
                64 => 6,
                _ => unreachable!("superblocks are 64x64"),
            }
        } else if w > h {
            if w == 64 { 5 } else { 2 }
        } else if h == 64 {
            4
        } else {
            1
        };
        NodeSyntax {
            forced,
            split_context: (forced.is_none() && has_rows && has_cols).then_some(size_ctx * 4 + raw),
            rect_context: rect_size * 4 + raw,
        }
    }

    /// Record a square leaf (AVM `partition_context_lookup`: a square of side
    /// `s` stores `64 - s / 4` above and left).
    fn record_leaf(&mut self, x: usize, y: usize, size: usize) {
        let value = (64 - size / CONTEXT_UNIT) as u8;
        let units = size / CONTEXT_UNIT;
        let x4 = x / CONTEXT_UNIT;
        let y4 = (y / CONTEXT_UNIT) % LEFT_UNITS;
        let above_end = (x4 + units).min(self.above.len());
        self.above[x4..above_end].fill(value);
        self.left[y4..y4 + units].fill(value);
    }
}

/// A partition symbol: `do_split` (with its context) or `rect_type`.
#[derive(Debug, Clone, Copy)]
pub enum Symbol {
    /// `do_split`, with its context index.
    Split(usize),
    /// `rect_type` (true: vertical), with its context index.
    Rect(usize),
}

/// Destination of partition symbols: the entropy writer, or a test trace.
pub trait SymbolSink {
    /// Code one binary partition symbol.
    /// # Errors
    /// Propagates entropy errors.
    fn put(&mut self, symbol: Symbol, value: bool) -> Result<()>;
}

impl SymbolSink for Writer {
    fn put(&mut self, symbol: Symbol, value: bool) -> Result<()> {
        match symbol {
            Symbol::Split(index) => self.model(usize::from(value), &[SPLIT[index], 0], |m| {
                &mut m.split[index]
            }),
            Symbol::Rect(index) => self.model(usize::from(value), &[RECT[index], 0], |m| {
                &mut m.rect[index]
            }),
        }
    }
}

fn read_symbol(reader: &mut Reader<'_>, symbol: Symbol) -> Result<bool> {
    let value = match symbol {
        Symbol::Split(index) => reader.model(&[SPLIT[index], 0], |m| &mut m.split[index])?,
        Symbol::Rect(index) => reader.model(&[RECT[index], 0], |m| &mut m.rect[index])?,
    };
    Ok(value != 0)
}

/// Writes the engine's square quadtree as AV2 binary partitions.
///
/// Below 128x128 AV2 has no four-way split: a quad split of a square is a
/// horizontal split whose two halves each split vertically (or the
/// boundary-implied transpose of that), which visits the four squares in the
/// same raster order. The engine reports nodes depth first; each half's own
/// symbols are written when the engine enters its first square, so symbols
/// stay interleaved with block syntax exactly as AVM reads them.
#[derive(Debug, Clone)]
pub struct PartitionWriter {
    contexts: Contexts,
    pending: Vec<Node>,
}

impl PartitionWriter {
    /// Partition state for one `width` x `height` tile.
    /// # Errors
    /// Requires positive dimensions divisible by eight.
    pub fn new(width: usize, height: usize) -> Result<Self> {
        Ok(Self {
            contexts: Contexts::new(width, height)?,
            pending: Vec::new(),
        })
    }

    /// Start a superblock row when `x` is zero (left contexts reset).
    pub fn begin_superblock(&mut self, x: usize) {
        if x == 0 {
            self.contexts.left.fill(0);
        }
    }

    /// Code the decision for the square node at `(x, y)` of side `size`:
    /// leaf, or quad split into four squares.
    /// # Errors
    /// Rejects a leaf the syntax cannot express (a boundary-forced split) and
    /// propagates entropy errors.
    pub fn write(
        &mut self,
        writer: &mut impl SymbolSink,
        x: usize,
        y: usize,
        size: usize,
        split: bool,
    ) -> Result<()> {
        if let Some(index) = self.pending.iter().position(|n| (n.x, n.y) == (x, y)) {
            let half = self.pending.swap_remove(index);
            // A wide half splits vertically, a tall one horizontally.
            self.write_node(writer, half, true, half.w > half.h)?;
        }
        let node = Node {
            x,
            y,
            w: size,
            h: size,
        };
        let vertical = self.write_node(writer, node, split, false)?;
        if !split {
            self.contexts.record_leaf(x, y, size);
            return Ok(());
        }
        let half = size / 2;
        let halves = if vertical {
            [(x, y, half, size), (x + half, y, half, size)]
        } else {
            [(x, y, size, half), (x, y + half, size, half)]
        };
        for (x, y, w, h) in halves {
            if self.contexts.contains(x, y) {
                self.pending.push(Node { x, y, w, h });
            }
        }
        Ok(())
    }

    /// Approximate bits of the decision for the square node at `(x, y)` of
    /// side `size` under the default CDFs and the current contexts: its own
    /// symbols plus, for a split, those of the two halves it splits into.
    #[must_use]
    pub fn bits(&self, x: usize, y: usize, size: usize, split: bool) -> f32 {
        let node = Node {
            x,
            y,
            w: size,
            h: size,
        };
        let (mut bits, vertical) = self.node_bits(node, split, false);
        if split {
            let half = size / 2;
            let halves = if vertical {
                [(x, y, half, size), (x + half, y, half, size)]
            } else {
                [(x, y, size, half), (x, y + half, size, half)]
            };
            for (x, y, w, h) in halves {
                if self.contexts.contains(x, y) {
                    bits += self.node_bits(Node { x, y, w, h }, true, w > h).0;
                }
            }
        }
        bits
    }

    /// Bits of one node's symbols and the split direction they choose.
    fn node_bits(&self, node: Node, split: bool, preferred_vertical: bool) -> (f32, bool) {
        let syntax = self.contexts.syntax(node);
        if let Some(vertical) = syntax.forced {
            return (0.0, vertical);
        }
        let mut bits = 0.0;
        if let Some(index) = syntax.split_context {
            bits += symbol_bits(&[SPLIT[index], 0], usize::from(split));
        }
        if split || syntax.split_context.is_none() {
            bits += symbol_bits(
                &[RECT[syntax.rect_context], 0],
                usize::from(preferred_vertical),
            );
        }
        (bits, preferred_vertical)
    }

    /// Write `node`'s symbols, returning the split direction used.
    fn write_node(
        &mut self,
        writer: &mut impl SymbolSink,
        node: Node,
        split: bool,
        preferred_vertical: bool,
    ) -> Result<bool> {
        let syntax = self.contexts.syntax(node);
        if let Some(vertical) = syntax.forced {
            if !split || vertical != preferred_vertical && node.w != node.h {
                return Err(DecodeError::Unsupported(
                    "AV2 partition implied at the boundary differs from the decision",
                ));
            }
            return Ok(vertical);
        }
        let must_split = syntax.split_context.is_none();
        match syntax.split_context {
            Some(index) => writer.put(Symbol::Split(index), split)?,
            None if !split => {
                return Err(DecodeError::Unsupported(
                    "AV2 partition split implied at the boundary",
                ));
            }
            None => {}
        }
        if split || must_split {
            writer.put(Symbol::Rect(syntax.rect_context), preferred_vertical)?;
        }
        Ok(preferred_vertical)
    }
}

/// Reads the partition subset written by [`PartitionWriter`]: square leaves
/// from 64x64 down to 8x8, reached through pairs of binary splits.
#[derive(Debug, Clone)]
pub struct PartitionReader {
    contexts: Contexts,
}

impl PartitionReader {
    /// Partition state for one `width` x `height` tile.
    /// # Errors
    /// Requires positive dimensions divisible by eight.
    pub fn new(width: usize, height: usize) -> Result<Self> {
        Ok(Self {
            contexts: Contexts::new(width, height)?,
        })
    }

    /// Read one superblock, calling `leaf(reader, x, y, size)` for each square
    /// leaf in coding order, before the next partition symbol.
    /// # Errors
    /// Rejects rectangular leaves (outside the subset) and callback errors.
    pub fn read_superblock(
        &mut self,
        reader: &mut Reader<'_>,
        x: usize,
        y: usize,
        leaf: &mut impl FnMut(&mut Reader<'_>, usize, usize, usize) -> Result<()>,
    ) -> Result<()> {
        if x == 0 {
            self.contexts.left.fill(0);
        }
        self.read_square(reader, x, y, SUPERBLOCK, leaf)
    }

    fn read_square(
        &mut self,
        reader: &mut Reader<'_>,
        x: usize,
        y: usize,
        size: usize,
        leaf: &mut impl FnMut(&mut Reader<'_>, usize, usize, usize) -> Result<()>,
    ) -> Result<()> {
        if !self.contexts.contains(x, y) {
            return Ok(());
        }
        let node = Node {
            x,
            y,
            w: size,
            h: size,
        };
        let Some(vertical) = self.read_node(reader, node)? else {
            self.contexts.record_leaf(x, y, size);
            return leaf(reader, x, y, size);
        };
        let half = size / 2;
        let halves = if vertical {
            [(x, y, half, size), (x + half, y, half, size)]
        } else {
            [(x, y, size, half), (x, y + half, size, half)]
        };
        for (hx, hy, w, h) in halves {
            if !self.contexts.contains(hx, hy) {
                continue;
            }
            let rect = Node { x: hx, y: hy, w, h };
            if self.read_node(reader, rect)? != Some(w > h) {
                return Err(DecodeError::Unsupported("AV2 rectangular partition leaf"));
            }
            let squares = if w > h {
                [(hx, hy), (hx + half, hy)]
            } else {
                [(hx, hy), (hx, hy + half)]
            };
            for (sx, sy) in squares {
                self.read_square(reader, sx, sy, half, leaf)?;
            }
        }
        Ok(())
    }

    /// Read `node`'s split: `None` for a leaf, else the direction.
    fn read_node(&mut self, reader: &mut Reader<'_>, node: Node) -> Result<Option<bool>> {
        if node.w == MIN_LEAF && node.h == MIN_LEAF {
            // Square 8x8 nodes code do_split; sub-8x8 blocks are outside the subset.
            let syntax = self.contexts.syntax(node);
            if let Some(index) = syntax.split_context
                && read_symbol(reader, Symbol::Split(index))?
            {
                return Err(DecodeError::Unsupported("AV2 sub-8x8 partition"));
            }
            return Ok(None);
        }
        let syntax = self.contexts.syntax(node);
        if let Some(vertical) = syntax.forced {
            return Ok(Some(vertical));
        }
        let split = match syntax.split_context {
            Some(index) => read_symbol(reader, Symbol::Split(index))?,
            None => true,
        };
        if !split {
            return Ok(None);
        }
        Ok(Some(read_symbol(
            reader,
            Symbol::Rect(syntax.rect_context),
        )?))
    }
}

/// Write a regular shared-tree partition and invoke `leaf` after each 8×8 node.
/// Coordinates are luma pixels; each leaf owns two 4×4 chroma blocks. The caller
/// writes modes and residuals in the callback, before the next partition node.
/// # Errors
/// Requires positive dimensions divisible by eight; propagates callback errors.
pub fn write_8x8<W: SymbolSink>(
    writer: &mut W,
    width: usize,
    height: usize,
    mut leaf: impl FnMut(&mut W, usize, usize) -> Result<()>,
) -> Result<()> {
    let mut partitions = PartitionWriter::new(width, height)?;
    for y in (0..height).step_by(SUPERBLOCK) {
        for x in (0..width).step_by(SUPERBLOCK) {
            partitions.begin_superblock(x);
            write_uniform(&mut partitions, writer, x, y, SUPERBLOCK, &mut leaf)?;
        }
    }
    Ok(())
}

fn write_uniform<W: SymbolSink>(
    partitions: &mut PartitionWriter,
    writer: &mut W,
    x: usize,
    y: usize,
    size: usize,
    leaf: &mut impl FnMut(&mut W, usize, usize) -> Result<()>,
) -> Result<()> {
    if !partitions.contexts.contains(x, y) {
        return Ok(());
    }
    let split = size > MIN_LEAF;
    partitions.write(writer, x, y, size, split)?;
    if !split {
        return leaf(writer, x, y);
    }
    let half = size / 2;
    for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
        write_uniform(partitions, writer, x + dx, y + dy, half, leaf)?;
    }
    Ok(())
}

/// Read the partition subset emitted by [`write_8x8`].
/// # Errors
/// Rejects other partitions, invalid dimensions, and callback errors.
pub fn read_8x8(
    reader: &mut Reader<'_>,
    width: usize,
    height: usize,
    mut leaf: impl FnMut(&mut Reader<'_>, usize, usize) -> Result<()>,
) -> Result<()> {
    let mut partitions = PartitionReader::new(width, height)?;
    for y in (0..height).step_by(SUPERBLOCK) {
        for x in (0..width).step_by(SUPERBLOCK) {
            partitions.read_superblock(reader, x, y, &mut |r: &mut Reader<'_>, x, y, size| {
                if size != MIN_LEAF {
                    return Err(DecodeError::Unsupported("AV2 non-8x8 leaf"));
                }
                leaf(r, x, y)
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const REFERENCE_TRACE: [(u64, usize); 6] = [
        (11_177_593_530_534_780_299, 1),
        (12_533_257_701_386_276_712, 4),
        (8_206_196_605_830_944_025, 6),
        (12_307_546_906_268_168_638, 190),
        (13_929_457_352_609_380_318, 124),
        (8_107_322_983_209_155_998, 796),
    ];

    /// FNV-1a over each coded symbol and its fixed CDF, plus a symbol count.
    struct Trace((u64, usize));

    impl SymbolSink for Trace {
        fn put(&mut self, symbol: Symbol, value: bool) -> Result<()> {
            let cdf = match symbol {
                Symbol::Split(index) => [SPLIT[index], 0],
                Symbol::Rect(index) => [RECT[index], 0],
            };
            for v in std::iter::once(u64::from(value)).chain(cdf.iter().map(|&v| u64::from(v))) {
                self.0.0 = (self.0.0 ^ v).wrapping_mul(1_099_511_628_211);
            }
            self.0.1 += 1;
            Ok(())
        }
    }

    #[test]
    fn original_avm_partition_symbol_traces() {
        for ((width, height), expected) in
            [(8, 8), (16, 8), (8, 24), (64, 64), (72, 40), (128, 136)]
                .into_iter()
                .zip(REFERENCE_TRACE)
        {
            let mut trace = Trace((14_695_981_039_346_656_037_u64, 0));
            write_8x8(&mut trace, width, height, |_, _, _| Ok(())).unwrap();
            assert_eq!(trace.0, expected, "{width}x{height}");
        }
    }

    #[test]
    fn coverage_and_entropy_with_interleaved_modes() {
        for (width, height) in [(8, 8), (16, 8), (8, 24), (64, 64), (72, 40), (128, 136)] {
            let mut writer = Writer::new();
            let mut encoded = Vec::new();
            write_8x8(&mut writer, width, height, |w, x, y| {
                encoded.push((x, y));
                write_dc_modes(w, true)
            })
            .unwrap();
            let packet = writer.finish();
            let mut reader = Reader::new(&packet).unwrap();
            let mut decoded = Vec::new();
            read_8x8(&mut reader, width, height, |r, x, y| {
                decoded.push((x, y));
                read_dc_modes(r, true)
            })
            .unwrap();
            assert_eq!(encoded, decoded);
            encoded.sort_unstable();
            encoded.dedup();
            assert_eq!(encoded.len(), width * height / 64);
            assert!(encoded.iter().all(|&(x, y)| x < width
                && y < height
                && x.is_multiple_of(8)
                && y.is_multiple_of(8)));
        }
    }
}
