//! The block record: what the general block layer knows of a coding block
//! once its common syntax is read (dav2d `Av2Block`'s shared part). Each
//! tool group takes the record, reads its own syntax into its own part
//! (added here as the group is built: intra modes, inter motion, the
//! transform partition and coefficients), and the layer then records the
//! block in the contexts ([`super::context`]).
use super::size::BlockSize;

/// One coding block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools, reason = "the block's flags")]
pub struct BlockRecord {
    /// The luma tree size it belongs to (absent: a chroma tree's block).
    pub luma: Option<BlockSize>,
    /// The chroma tree size (absent: a luma tree's block, or any block of a
    /// monochrome frame).
    pub chroma: Option<BlockSize>,
    /// The whole block's chroma size: pieces of blocks above 64x64 keep it
    /// for the chroma coefficient contexts (dav2d `b->cbs`).
    pub block_chroma: Option<BlockSize>,
    /// Luma and chroma share the block's tree, also in a monochrome frame
    /// that codes no chroma (dav2d `lbs == cbs_tree`).
    pub shared_tree: bool,
    /// Top-left luma row in 4x4 units.
    pub row: u32,
    /// Top-left luma column in 4x4 units.
    pub column: u32,
    /// Its chroma block's top-left, in luma 4x4 units.
    pub chroma_row: u32,
    /// See `chroma_row`.
    pub chroma_column: u32,
    /// Skip mode.
    pub skip_mode: bool,
    /// Intra (not inter).
    pub intra: bool,
    /// Intra block copy.
    pub intrabc: bool,
    /// No residual.
    pub skip_txfm: bool,
    /// The superblock quantizer in force.
    pub qindex: u16,
}
