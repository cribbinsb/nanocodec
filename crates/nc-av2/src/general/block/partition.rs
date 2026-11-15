//! The partition tree of a superblock (dav2d `decode_sb`, AVM
//! `read_partition` and `decode_partition`): binary, three-way and uneven
//! four-way splits, the four-way square split of 128x128 and 256x256
//! blocks, sizes implied at the frame edge, separate luma and chroma trees
//! (key and intra-only frames split them at 64x64; inter frames may start
//! an intra-only region with its own chroma tree).
//!
//! `luma` and `chroma` name the block size of each tree a call covers:
//! both for a shared tree, one for a separate tree's (dav2d `lbs`, `cbs`,
//! with `None` for its `BS_INVALID`). The direction word follows dav2d:
//! bits 0..7 the direction read (0xff none, 1 horizontal, 2 vertical),
//! bits 8..15 the partition code, bits 16..23 the children's directions,
//! bit 24 that the children may not split again.
use super::size::{BlockSize, Partition};
use super::{Tile, samples};
use nc_common::decode::{DecodeError, Result};

/// Direction-word fields (see the module docs).
const DIRECTION_NONE: u32 = 0xff;
const DIRECTION_MASK: u32 = 0xff;
const PARTITION_SHIFT: u32 = 8;
const CHILDREN_SHIFT: u32 = 16;
/// The luma split one way and its children only the other way: chroma
/// copies the first split (dav2d F164).
const ONE_WAY_THEN_OTHER_MASK: u32 = 0x30003;
const VERTICAL_THEN_HORIZONTAL: u32 = 0x10002;
const HORIZONTAL_THEN_VERTICAL: u32 = 0x20001;
const NO_RECURSIVE_SPLIT: u32 = 1 << 24;
/// Blocks of at most 16 units a side may split three or four ways, and
/// region types are read below 16 units.
const SMALL_BLOCK_UNITS: u32 = 16;
/// Blocks of 32 units or more along their long side split only in halves.
const LARGE_RECTANGLE_UNITS: u32 = 32;
/// The context offset (in four-context units) for each plane tree:
/// contexts are `plane context + offset * 4`.
const CONTEXTS_PER_OFFSET: usize = 4;

impl Tile<'_, '_> {
    /// Decode the partition tree of the block at the current position.
    /// # Errors
    /// Propagates the blocks' errors.
    pub(super) fn superblock(
        &mut self,
        depth: usize,
        luma: Option<BlockSize>,
        chroma: Option<BlockSize>,
        direction_word: &mut u32,
    ) -> Result<()> {
        let size = luma
            .or(chroma)
            .ok_or(DecodeError::Malformed("AV2 partition without a tree"))?;
        // Key and intra-only frames apply separate trees from 64x64.
        if luma == Some(BlockSize::B64x64)
            && chroma == Some(BlockSize::B64x64)
            && self.sequence.partition.separate_trees
            && self.intra_frame()
        {
            let mut direction = 0;
            self.superblock(depth, luma, None, &mut direction)?;
            // With a 64x64 root, the chroma restoration units are read here,
            // between the trees (dav2d `read_sb_restoration(t, 1, 3)`); the
            // tile rejects restoration until the loop-filter group reads
            // units.
            return self.superblock(depth, None, chroma, &mut direction);
        }
        let range = self.reader.range();
        let plane_name = match (luma, chroma) {
            (Some(_), None) => "y",
            (None, Some(_)) => "uv",
            _ => "yuv",
        };
        let (w4, h4) = size.units();
        trace!(
            self,
            depth - 1,
            "decode_sb[y={},x={},bs={}x{},plane={plane_name}]: r={range}",
            self.row,
            self.column,
            samples(w4),
            samples(h4)
        );
        let original_chroma = chroma;
        let (partition, chroma) =
            self.read_partition(depth, size, (luma, chroma), direction_word)?;
        if chroma.is_some() && chroma == Some(size) {
            self.chroma_row = self.row;
            self.chroma_column = self.column;
        }
        let (limit_w, limit_h) = partition.child_limit();
        let mut children = if w4 <= limit_w || h4 <= limit_h {
            NO_RECURSIVE_SPLIT
        } else {
            0
        };
        self.split(depth, size, partition, (luma, chroma), &mut children)?;
        *direction_word |= (children & DIRECTION_MASK) << CHILDREN_SHIFT;
        if self.intra_region && original_chroma.is_some() {
            // The intra region's chroma block follows its luma tree.
            self.chroma_row = self.row;
            self.chroma_column = self.column;
            self.block(depth + 1, None, original_chroma)?;
            self.intra_region = false;
        }
        Ok(())
    }

    /// Decode the children of a block of `size` split by `partition`, in
    /// coding order, skipping those wholly outside the frame (dav2d
    /// `decode_sb`'s partition cases). A chroma tree's children carry only
    /// chroma sizes; a shared tree's last child of a split too small for
    /// 4:2:0 chroma carries the parent's chroma block.
    fn split(
        &mut self,
        depth: usize,
        size: BlockSize,
        partition: Partition,
        (luma, chroma): (Option<BlockSize>, Option<BlockSize>),
        children: &mut u32,
    ) -> Result<()> {
        let parts = size.subdivision().parts;
        let chroma_tree = luma.is_none();
        let (w4, h4) = size.units();
        let (columns, rows) = (self.geometry.columns, self.geometry.rows);
        // The chroma block follows this block's own tree.
        let own_chroma = chroma == Some(size);
        let luma_part = |part: Option<BlockSize>| if chroma_tree { None } else { part };
        match partition {
            Partition::None => {
                self.block(depth + 1, luma, chroma)?;
                self.set_partition_context(
                    size,
                    luma.is_some() && chroma.is_some(),
                    usize::from(chroma_tree),
                );
            }
            Partition::Vertical | Partition::Horizontal => {
                let vertical = partition == Partition::Vertical;
                let half = parts[usize::from(vertical)][0];
                let step = if vertical { w4 / 2 } else { h4 / 2 };
                let sub = own_chroma && (step >> 1) > 0;
                self.child(
                    depth,
                    (luma_part(half), if sub { half } else { None }),
                    children,
                )?;
                if !self.advance(vertical, step, (columns, rows)) {
                    return Ok(());
                }
                self.child(
                    depth,
                    (luma_part(half), if sub { half } else { chroma }),
                    children,
                )?;
                self.retreat(vertical, step);
            }
            Partition::Split => {
                let quarter = parts[0][3];
                let have_h_split = columns > self.column + w4 / 2;
                self.child(depth, (quarter, quarter), children)?;
                self.column += w4 / 2;
                if have_h_split {
                    self.child(depth, (quarter, quarter), children)?;
                }
                self.column -= w4 / 2;
                self.row += h4 / 2;
                self.child(depth, (quarter, quarter), children)?;
                self.column += w4 / 2;
                if have_h_split {
                    self.child(depth, (quarter, quarter), children)?;
                }
                self.column -= w4 / 2;
                self.row -= h4 / 2;
            }
            Partition::Vertical3 | Partition::Horizontal3 => {
                self.three_way(
                    depth,
                    size,
                    partition == Partition::Vertical3,
                    (luma, chroma),
                    children,
                )?;
            }
            Partition::Vertical4A
            | Partition::Vertical4B
            | Partition::Horizontal4A
            | Partition::Horizontal4B => {
                let vertical = matches!(partition, Partition::Vertical4A | Partition::Vertical4B);
                let wide_second =
                    matches!(partition, Partition::Vertical4B | Partition::Horizontal4B);
                self.four_way(
                    depth,
                    size,
                    (vertical, wide_second),
                    (luma, chroma),
                    children,
                )?;
            }
        }
        Ok(())
    }

    /// A three-way split: a quarter strip, two half-length quarter blocks
    /// side by side across the middle half, a quarter strip.
    fn three_way(
        &mut self,
        depth: usize,
        size: BlockSize,
        vertical: bool,
        (luma, chroma): (Option<BlockSize>, Option<BlockSize>),
        children: &mut u32,
    ) -> Result<()> {
        let parts = size.subdivision().parts[usize::from(vertical)];
        let chroma_tree = luma.is_none();
        let (w4, h4) = size.units();
        let (columns, rows) = (self.geometry.columns, self.geometry.rows);
        // Along the split: the quarter step; across it: the half step.
        let (quarter, half) = if vertical {
            (w4 / 4, h4 / 2)
        } else {
            (h4 / 4, w4 / 2)
        };
        let sub = chroma == Some(size) && (quarter >> 1) > 0 && (half >> 1) > 0;
        let thin = if vertical {
            BlockSize::B32x8
        } else {
            BlockSize::B8x32
        };
        let strips_only = chroma.is_none() || (!sub && size != thin);
        let luma_part = |part: Option<BlockSize>| if chroma_tree { None } else { part };
        let strip = parts[1];
        let middle = parts[3];
        self.child(
            depth,
            (luma_part(strip), if strips_only { None } else { strip }),
            children,
        )?;
        if !self.advance(vertical, quarter, (columns, rows)) {
            return Ok(());
        }
        if !strips_only {
            self.chroma_row = self.row;
            self.chroma_column = self.column;
        }
        self.child(
            depth,
            (luma_part(middle), if sub { middle } else { None }),
            children,
        )?;
        // The second middle block, across the split.
        if self.advance(!vertical, half, (columns, rows)) {
            let chroma_part = if strips_only {
                None
            } else if sub {
                middle
            } else {
                parts[0]
            };
            self.child(depth, (luma_part(middle), chroma_part), children)?;
            self.retreat(!vertical, half);
        }
        let middle_extent = if vertical { w4 / 2 } else { h4 / 2 };
        if !self.advance(vertical, middle_extent, (columns, rows)) {
            self.retreat(vertical, quarter);
            return Ok(());
        }
        let last_chroma = if strips_only { chroma } else { strip };
        self.child(depth, (luma_part(strip), last_chroma), children)?;
        self.retreat(vertical, 3 * quarter);
        Ok(())
    }

    /// An uneven four-way split: strips of 1/8, then 2/8 and 4/8 (A) or
    /// 4/8 and 2/8 (B), then 1/8.
    fn four_way(
        &mut self,
        depth: usize,
        size: BlockSize,
        (vertical, wide_second): (bool, bool),
        (luma, chroma): (Option<BlockSize>, Option<BlockSize>),
        children: &mut u32,
    ) -> Result<()> {
        let parts = size.subdivision().parts[usize::from(vertical)];
        let chroma_tree = luma.is_none();
        let (w4, h4) = size.units();
        let (columns, rows) = (self.geometry.columns, self.geometry.rows);
        let extent = if vertical { w4 } else { h4 };
        let (eighth, quarter, half) = (extent / 8, extent / 4, extent / 2);
        let sub = chroma == Some(size) && (eighth >> 1) > 0;
        let luma_part = |part: Option<BlockSize>| if chroma_tree { None } else { part };
        let chroma_part = |part: Option<BlockSize>| if sub { part } else { None };
        let variant = usize::from(wide_second);
        // Strip sizes: parts[2] the eighth, parts[1] the quarter, parts[0]
        // the half.
        let (second, third) = (parts[1 - variant], parts[variant]);
        let (second_extent, third_extent) = if wide_second {
            (half, quarter)
        } else {
            (quarter, half)
        };
        self.child(
            depth,
            (luma_part(parts[2]), chroma_part(parts[2])),
            children,
        )?;
        if !self.advance(vertical, eighth, (columns, rows)) {
            return Ok(());
        }
        self.child(depth, (luma_part(second), chroma_part(second)), children)?;
        if !self.advance(vertical, second_extent, (columns, rows)) {
            self.retreat(vertical, eighth);
            return Ok(());
        }
        self.child(depth, (luma_part(third), chroma_part(third)), children)?;
        if !self.advance(vertical, third_extent, (columns, rows)) {
            self.retreat(vertical, eighth + second_extent);
            return Ok(());
        }
        let last_chroma = if sub { parts[2] } else { chroma };
        self.child(depth, (luma_part(parts[2]), last_chroma), children)?;
        self.retreat(vertical, 7 * eighth);
        Ok(())
    }

    /// Decode one child's tree.
    fn child(
        &mut self,
        depth: usize,
        (luma, chroma): (Option<BlockSize>, Option<BlockSize>),
        children: &mut u32,
    ) -> Result<()> {
        self.superblock(depth + 1, luma, chroma, children)
    }

    /// Move `step` units right (`vertical` split) or down; `false`, without
    /// moving, when that is outside the frame.
    fn advance(&mut self, vertical: bool, step: u32, (columns, rows): (u32, u32)) -> bool {
        if vertical {
            if self.column + step >= columns {
                return false;
            }
            self.column += step;
        } else {
            if self.row + step >= rows {
                return false;
            }
            self.row += step;
        }
        true
    }

    /// Move back `step` units left or up.
    fn retreat(&mut self, vertical: bool, step: u32) {
        if vertical {
            self.column -= step;
        } else {
            self.row -= step;
        }
    }

    /// Record a block of `size` in the partition contexts: of both trees
    /// when `shared`, else of `plane`'s.
    fn set_partition_context(&mut self, size: BlockSize, shared: bool, plane: usize) {
        let (w4, h4) = size.units();
        let column = self.column as usize;
        let row = super::context::left_index(self.row);
        let above_value = !((w4 - 1) as u8);
        let left_value = !((h4 - 1) as u8);
        let planes = if shared { 0..2 } else { plane..plane + 1 };
        for plane in planes {
            fill(
                &mut self.contexts.above.partition[plane],
                column,
                w4,
                above_value,
            );
            fill(
                &mut self.contexts.left.partition[plane],
                row,
                h4,
                left_value,
            );
        }
    }

    /// The partition of a block of `size` and the chroma size its children
    /// carry (none once an intra region starts its own chroma tree).
    fn read_partition(
        &mut self,
        depth: usize,
        size: BlockSize,
        (luma, chroma): (Option<BlockSize>, Option<BlockSize>),
        direction_word: &mut u32,
    ) -> Result<(Partition, Option<BlockSize>)> {
        let (w4, h4) = size.units();
        let (log2_w, log2_h) = size.log2_units();
        let subdivision = size.subdivision();
        let chroma_tree = luma.is_none();
        let plane = usize::from(chroma_tree);
        let shared = luma.is_some() && luma == chroma;
        // The block in the tree's own samples (chroma trees subsample).
        let own = (w4 >> u32::from(chroma_tree), h4 >> u32::from(chroma_tree));
        let geometry = self.geometry;
        let have_h_split = geometry.columns > self.column + w4 / 2;
        let have_v_split = geometry.rows > self.row + h4 / 2;
        let mixed_inter = !self.frame.is_intra() && !self.intra_region;
        let mut direction: Option<bool> = None;
        let mut partition = None;
        if own.0.max(own.1) == 1
            || (subdivision.parts[0][0].is_none() && subdivision.parts[1][0].is_none())
            || (mixed_inter && log2_w + log2_h == 1)
        {
            partition = Some(Partition::None);
        } else if !have_h_split || !have_v_split {
            if w4 == h4 {
                direction = Some(have_v_split);
                partition = Some(if have_v_split {
                    Partition::Vertical
                } else {
                    Partition::Horizontal
                });
            } else if w4 > h4 {
                if !have_h_split || geometry.rows <= self.row + h4 / 4 {
                    direction = Some(true);
                    partition = Some(Partition::Vertical);
                }
            } else if !have_v_split || geometry.columns <= self.column + w4 / 4 {
                direction = Some(false);
                partition = Some(Partition::Horizontal);
            }
        }
        let partition = match partition {
            Some(partition) => partition,
            None if chroma == Some(BlockSize::B64x64) && luma.is_none() => {
                match Self::inferred_chroma_partition(*direction_word, &mut direction) {
                    Some(partition) => partition,
                    None => self.coded_partition(size, (plane, shared), &mut direction)?,
                }
            }
            None => self.coded_partition(size, (plane, shared), &mut direction)?,
        };
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "read_partition[y={},x={},bs={}x{},bp={}|{}]: r={range}",
            self.row,
            self.column,
            samples(w4),
            samples(h4),
            partition.code(),
            partition_name(partition)
        );
        let direction_code = match direction {
            None => DIRECTION_NONE,
            Some(vertical) => 1 + u32::from(vertical),
        };
        if luma.is_none() && chroma == Some(BlockSize::B64x64) {
            self.sdp_cfl_disallowed =
                direction.is_some() && direction_code != (*direction_word & 3);
        }
        *direction_word |= direction_code | (partition.code() << PARTITION_SHIFT);
        let chroma = if self.reads_region_type(size, partition, (luma, chroma), *direction_word) {
            self.region_type(depth, size)?;
            if self.intra_region { None } else { chroma }
        } else {
            chroma
        };
        Ok((partition, chroma))
    }

    /// Whether a split block of an inter frame codes whether it starts an
    /// intra-only region with its own chroma tree (`enable_extended_sdp`).
    fn reads_region_type(
        &self,
        size: BlockSize,
        partition: Partition,
        (luma, chroma): (Option<BlockSize>, Option<BlockSize>),
        direction_word: u32,
    ) -> bool {
        let (w4, h4) = size.units();
        !self.frame.is_intra()
            && self.sequence.partition.extended_separate_trees
            && luma.is_some()
            && chroma.is_some()
            && matches!(
                partition,
                Partition::Horizontal
                    | Partition::Vertical
                    | Partition::Horizontal3
                    | Partition::Vertical3
            )
            && direction_word & NO_RECURSIVE_SPLIT == 0
            && w4.min(h4) >= 2
            && size != self.geometry.root
            && w4.max(h4) <= SMALL_BLOCK_UNITS
    }

    /// Read whether the block starts an intra-only region.
    fn region_type(&mut self, depth: usize, size: BlockSize) -> Result<()> {
        let (log2_w, log2_h) = size.log2_units();
        let sum = u32::from(log2_w + log2_h);
        let context = (sum.saturating_sub(4).min(3) + u32::from(sum == 4)) as usize;
        self.intra_region = !self.bool(|m| &mut m.region_type[context])?;
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "is_mixed_region[ctx={context},{}]: r={range}",
            u8::from(!self.intra_region)
        );
        Ok(())
    }

    /// The chroma partition of a 64x64 separate chroma tree the luma tree
    /// implies: none when luma did not split, luma's first split when luma
    /// split one way and its children only the other (dav2d F164).
    fn inferred_chroma_partition(word: u32, direction: &mut Option<bool>) -> Option<Partition> {
        if word & DIRECTION_MASK == DIRECTION_NONE {
            return Some(Partition::None);
        }
        let pattern = word & ONE_WAY_THEN_OTHER_MASK;
        if pattern == VERTICAL_THEN_HORIZONTAL || pattern == HORIZONTAL_THEN_VERTICAL {
            *direction = Some(pattern == VERTICAL_THEN_HORIZONTAL);
            return Partition::of_code((word >> PARTITION_SHIFT) & DIRECTION_MASK);
        }
        None
    }

    /// Read a partition: whether the block splits, then how.
    fn coded_partition(
        &mut self,
        size: BlockSize,
        (plane, shared): (usize, bool),
        direction: &mut Option<bool>,
    ) -> Result<Partition> {
        let (w4, h4) = size.units();
        let (log2_w, log2_h) = size.log2_units();
        let subdivision = size.subdivision();
        let geometry = self.geometry;
        let have_h_split = geometry.columns > self.column + w4 / 2;
        let have_v_split = geometry.rows > self.row + h4 / 2;
        let chroma_tree = plane == 1;
        let own = (w4 >> u32::from(chroma_tree), h4 >> u32::from(chroma_tree));
        let aspect = 1_u32 << self.sequence.partition.max_aspect_ratio_log2;
        let (long, short) = (w4.max(h4), w4.min(h4));
        let aspect_requires_split = long > aspect * short;
        let context = self.partition_context(plane, log2_w, log2_h);
        let split_offset = usize::from(subdivision.context[0].unwrap_or(0));
        let split = !have_h_split
            || !have_v_split
            || aspect_requires_split
            || self
                .bool(|m| &mut m.part_split[plane][context + split_offset * CONTEXTS_PER_OFFSET])?;
        if !split {
            return Ok(Partition::None);
        }
        if matches!(size, BlockSize::B128x128 | BlockSize::B256x256) && have_v_split && have_h_split
        {
            let square_context =
                context + usize::from(size == BlockSize::B256x256) * CONTEXTS_PER_OFFSET;
            if self.bool(|m| &mut m.part_square[square_context])? {
                return Ok(Partition::Split);
            }
        } else if w4.max(h4) >= LARGE_RECTANGLE_UNITS {
            return Ok(if w4 > h4 {
                Partition::Vertical
            } else {
                Partition::Horizontal
            });
        }
        let not_eight = aspect != 8;
        let vertical = if own.0.min(own.1) == 1 {
            own.0 > own.1
        } else if aspect_requires_split || short * 8 < long * 2 + u32::from(not_eight) {
            w4 > h4
        } else {
            let direction_offset = usize::from(subdivision.context[1].unwrap_or(0));
            self.bool(|m| &mut m.part_dir[plane][context + direction_offset * CONTEXTS_PER_OFFSET])?
        };
        *direction = Some(vertical);
        let mut partition = if vertical {
            Partition::Vertical
        } else {
            Partition::Horizontal
        };
        if w4.max(h4) <= SMALL_BLOCK_UNITS {
            partition = self.extended_partition(size, (plane, shared), vertical, partition)?;
        }
        Ok(partition)
    }

    /// Whether a binary split of a small block is three-way or uneven
    /// four-way instead.
    fn extended_partition(
        &mut self,
        size: BlockSize,
        (plane, shared): (usize, bool),
        vertical: bool,
        binary: Partition,
    ) -> Result<Partition> {
        let (w4, h4) = size.units();
        let (log2_w, log2_h) = size.log2_units();
        let dims = [w4, h4];
        let geometry = self.geometry;
        let have_h_split = geometry.columns > self.column + w4 / 2;
        let have_v_split = geometry.rows > self.row + h4 / 2;
        let chroma_tree = plane == 1;
        let own = [w4 >> u32::from(chroma_tree), h4 >> u32::from(chroma_tree)];
        // The chroma block of a shared tree in 4:2:0.
        let subsampled = [w4 >> 1, h4 >> 1];
        // Only a shared tree's split must also suit its 4:2:0 chroma block.
        let across = usize::from(!vertical);
        let along = usize::from(vertical);
        let not_eight = u32::from(self.sequence.partition.max_aspect_ratio_log2 != 3);
        let (qw4, qh4) = (w4 / 4, h4 / 4);
        let three_way = self.sequence.partition.extended
            && own[across] >= 4
            && own[along] >= 2
            && dims[across] * 8 >= dims[along] * 4 + not_eight
            && (!shared
                || (subsampled[across] >= 4 && subsampled[along] >= 2)
                || if vertical {
                    if size == BlockSize::B32x8 {
                        have_v_split
                    } else {
                        self.column + qw4 * 3 < geometry.columns
                    }
                } else if size == BlockSize::B8x32 {
                    have_h_split
                } else {
                    self.row + qh4 * 3 < geometry.rows
                });
        let four_way = own[across] >= 8
            && self.sequence.partition.uneven_four_way
            && dims[across] * 8 >= dims[along] * 8 + not_eight
            && (!shared
                || subsampled[across] >= 8
                || if vertical {
                    self.column + (qw4 >> 1) * 7 < geometry.columns
                } else {
                    self.row + (qh4 >> 1) * 7 < geometry.rows
                });
        if !three_way && !four_way {
            return Ok(binary);
        }
        let context = self.partition2_context(plane, vertical, log2_w, log2_h)
            + usize::from(size.subdivision().context[0].unwrap_or(0)) * CONTEXTS_PER_OFFSET;
        if !self.bool(|m| &mut m.part_ext[plane][context])? {
            return Ok(binary);
        }
        let mut partition = if vertical {
            Partition::Vertical3
        } else {
            Partition::Horizontal3
        };
        if four_way && (!three_way || self.bool(|m| &mut m.part_4way[plane][context])?) {
            let b = self.bypass()?;
            partition = match (vertical, b) {
                (false, false) => Partition::Horizontal4A,
                (false, true) => Partition::Horizontal4B,
                (true, false) => Partition::Vertical4A,
                (true, true) => Partition::Vertical4B,
            };
        }
        Ok(partition)
    }

    /// The split context of a block: whether the neighbours above and left
    /// are narrower / shorter than half of it (dav2d `get_partition_ctx`).
    fn partition_context(&self, plane: usize, log2_w: u8, log2_h: u8) -> usize {
        let above = self.contexts.above.partition[plane][self.column as usize];
        let left = self.contexts.left.partition[plane][super::context::left_index(self.row)];
        let bit = |value: u8, log2: u8| usize::from((value >> log2.saturating_sub(1)) & 1);
        bit(above, log2_w) + (bit(left, log2_h) << 1)
    }

    /// The three-way context: the neighbours along the split at its middle
    /// and start (dav2d `get_partition2_ctx`).
    fn partition2_context(&self, plane: usize, vertical: bool, log2_w: u8, log2_h: u8) -> usize {
        let bit = |value: u8, log2: u8| usize::from((value >> (log2 - 2)) & 1);
        if vertical {
            let above = &self.contexts.above.partition[plane];
            let column = self.column as usize;
            let half = 1_usize << (log2_w - 1);
            bit(above[column + half], log2_w) + (bit(above[column], log2_w) << 1)
        } else {
            let left = &self.contexts.left.partition[plane];
            let row = super::context::left_index(self.row);
            let half = 1_usize << (log2_h - 1);
            bit(left[row + half], log2_h) + (bit(left[row], log2_h) << 1)
        }
    }
}

/// Set `count` entries of `edge` from `start` to `value`, as far as the edge
/// goes (blocks may overhang the frame).
fn fill(edge: &mut [u8], start: usize, count: u32, value: u8) {
    let end = (start + count as usize).min(edge.len());
    if start < end {
        edge[start..end].fill(value);
    }
}

/// dav2d's trace name of a partition.
fn partition_name(partition: Partition) -> &'static str {
    match partition {
        Partition::None => "none",
        Partition::Horizontal => "h",
        Partition::Vertical => "v",
        Partition::Horizontal3 => "h3",
        Partition::Vertical3 => "v3",
        Partition::Horizontal4A => "h4a",
        Partition::Horizontal4B => "h4b",
        Partition::Vertical4A => "v4a",
        Partition::Vertical4B => "v4b",
        Partition::Split => "s",
    }
}
