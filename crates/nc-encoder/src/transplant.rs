//! The partition transplant oracle (`docs/analysis/distill-fast-2026-11-22.md`):
//! one encode records the partition it coded in every inter superblock, another
//! codes those partitions in place of its own plan.
//!
//! What it measures: the ceiling of a distilled partition guide. A fast rung that
//! codes a slow rung's partitions (or the partitions its own RD search would
//! have chosen) at the fast rung's cost is what a perfect predictor of those
//! decisions would give. It is an oracle, not a control: nothing in the encoder
//! records or replays partitions unless a caller asks
//! ([`crate::Encoder::record_partitions`], [`crate::Encoder::replay_partitions`];
//! `nanocodec encode --partition-record / --partition-replay`), and streams are
//! unchanged without it.
//!
//! The log: per (input frame index, superblock position) the coded quadtree in
//! pre-order, one byte per node ([`NodeCode`]). Key frames are not recorded;
//! a replayed superblock whose frame or tree the replaying encode cannot code
//! (a key frame, a shape its backend does not allow there) keeps its own plan.

use std::collections::HashMap;

use crate::partition::Block;
use crate::partition_plan::Partition;
use crate::shape::{MAX_SHAPE_PARTS, PartitionShape, ShapeSet};

/// The file's first bytes, and its version.
const MAGIC: &[u8; 8] = b"NCPLOG01";
/// Bytes of an entry's header: frame index (8), x and y (4 each), tree length (2).
const ENTRY_HEADER: usize = 18;

/// One node of a recorded tree, as its byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeCode {
    Leaf = 0,
    Split = 1,
    /// `PartitionShape::ALL[i]` is code `ShapeBase + i`.
    ShapeBase = 2,
}

/// What an encode does with partitions (`Encoder::record_partitions`,
/// `Encoder::replay_partitions`).
#[derive(Debug)]
pub(crate) enum Transplant {
    /// File every inter superblock's coded partition here.
    Record(PartitionLog),
    /// Code these partitions in place of the plan where they apply.
    Replay(PartitionLog),
}

/// Where a superblock's tree is filed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    frame: u64,
    x: u32,
    y: u32,
}

impl Key {
    fn of(frame: u64, sb: &Block) -> Self {
        Self {
            frame,
            x: u32::try_from(sb.x).expect("picture coordinates fit 32 bits"),
            y: u32::try_from(sb.y).expect("picture coordinates fit 32 bits"),
        }
    }
}

/// The recorded partitions of a stream's inter superblocks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PartitionLog {
    trees: HashMap<Key, Vec<u8>>,
}

/// Why a partition log could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionLogError(pub String);

impl std::fmt::Display for PartitionLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "partition log: {}", self.0)
    }
}

impl std::error::Error for PartitionLogError {}

impl PartitionLog {
    /// File `partition` as superblock `sb`'s of input frame `frame`; a
    /// re-coded frame's last coding wins.
    pub(crate) fn record(&mut self, frame: u64, sb: &Block, partition: &Partition) {
        let mut bytes = Vec::new();
        write_tree(partition, &mut bytes);
        self.trees.insert(Key::of(frame, sb), bytes);
    }

    /// Superblock `sb`'s recorded partition in input frame `frame`, where one was
    /// recorded and every shape in it is one `legal` allows at its node.
    pub(crate) fn replay(
        &self,
        frame: u64,
        sb: &Block,
        legal: impl Fn(&Block) -> ShapeSet,
    ) -> Option<Partition> {
        let bytes = self.trees.get(&Key::of(frame, sb))?;
        let mut at = 0;
        let tree = read_tree(bytes, &mut at, sb, &legal)?;
        (at == bytes.len()).then_some(tree)
    }

    /// Superblocks recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.trees.len()
    }

    /// Whether nothing is recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    /// The log as a file: [`MAGIC`], then per superblock (in frame, row,
    /// column order) its key, tree length and tree, little-endian.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut keys: Vec<&Key> = self.trees.keys().collect();
        keys.sort_by_key(|k| (k.frame, k.y, k.x));
        let mut out = MAGIC.to_vec();
        for key in keys {
            let tree = &self.trees[key];
            out.extend_from_slice(&key.frame.to_le_bytes());
            out.extend_from_slice(&key.x.to_le_bytes());
            out.extend_from_slice(&key.y.to_le_bytes());
            let len = u16::try_from(tree.len()).expect("a superblock's tree has under 64k nodes");
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(tree);
        }
        out
    }

    /// Read a log written by [`Self::to_bytes`].
    ///
    /// # Errors
    ///
    /// A file without the magic, or cut inside an entry.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PartitionLogError> {
        let body = bytes
            .strip_prefix(MAGIC.as_slice())
            .ok_or_else(|| PartitionLogError("not a partition log (magic)".into()))?;
        let mut trees = HashMap::new();
        let mut rest = body;
        while !rest.is_empty() {
            let (header, after) = rest
                .split_at_checked(ENTRY_HEADER)
                .ok_or_else(|| PartitionLogError("cut inside an entry header".into()))?;
            let (frame, xy) = header.split_at(8);
            let (x, y_len) = xy.split_at(4);
            let (y, len) = y_len.split_at(4);
            let key = Key {
                frame: u64::from_le_bytes(frame.try_into().expect("8 bytes")),
                x: u32::from_le_bytes(x.try_into().expect("4 bytes")),
                y: u32::from_le_bytes(y.try_into().expect("4 bytes")),
            };
            let len = usize::from(u16::from_le_bytes(len.try_into().expect("2 bytes")));
            let (tree, after) = after
                .split_at_checked(len)
                .ok_or_else(|| PartitionLogError("cut inside a tree".into()))?;
            trees.insert(key, tree.to_vec());
            rest = after;
        }
        Ok(Self { trees })
    }
}

fn write_tree(partition: &Partition, out: &mut Vec<u8>) {
    match partition {
        Partition::Leaf(_) => out.push(NodeCode::Leaf as u8),
        Partition::Split(children) => {
            out.push(NodeCode::Split as u8);
            for child in children {
                write_tree(child, out);
            }
        }
        Partition::Shaped(shape, _) => {
            let index = PartitionShape::ALL
                .iter()
                .position(|s| s == shape)
                .expect("every shape is in ALL");
            out.push(NodeCode::ShapeBase as u8 + index as u8);
        }
    }
}

/// The tree at `bytes[*at..]` for `node`, or `None` where it is malformed or
/// names a shape `legal` does not allow at its node.
fn read_tree(
    bytes: &[u8],
    at: &mut usize,
    node: &Block,
    legal: &impl Fn(&Block) -> ShapeSet,
) -> Option<Partition> {
    let code = *bytes.get(*at)?;
    *at += 1;
    match code {
        c if c == NodeCode::Leaf as u8 => Some(Partition::Leaf(None)),
        c if c == NodeCode::Split as u8 => {
            let quadrants = node.quadrants();
            let mut children: [Partition; 4] = std::array::from_fn(|_| Partition::Leaf(None));
            for (child, quadrant) in children.iter_mut().zip(quadrants.iter()) {
                *child = read_tree(bytes, at, quadrant, legal)?;
            }
            Some(Partition::Split(Box::new(children)))
        }
        c => {
            let shape = *PartitionShape::ALL.get(usize::from(c - NodeCode::ShapeBase as u8))?;
            legal(node)
                .contains(shape)
                .then(|| Partition::Shaped(shape, Box::new([None; MAX_SHAPE_PARTS])))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> Partition {
        Partition::Split(Box::new([
            Partition::Leaf(None),
            Partition::Shaped(PartitionShape::Vertical, Box::new([None; MAX_SHAPE_PARTS])),
            Partition::Split(Box::new(std::array::from_fn(|_| Partition::Leaf(None)))),
            Partition::Leaf(None),
        ]))
    }

    #[test]
    fn a_log_survives_its_file() {
        let sb = Block::square(64, 128, 6);
        let mut log = PartitionLog::default();
        log.record(7, &sb, &tree());
        log.record(9, &Block::square(0, 0, 6), &Partition::Leaf(None));
        let read = PartitionLog::from_bytes(&log.to_bytes()).unwrap();
        assert_eq!(read, log);
        let all = |_: &Block| ShapeSet::of(&PartitionShape::ALL);
        assert_eq!(read.replay(7, &sb, all), Some(tree()));
        assert_eq!(read.replay(8, &sb, all), None);
    }

    #[test]
    fn a_shape_the_replaying_encode_cannot_code_keeps_the_plan() {
        let sb = Block::square(0, 0, 6);
        let mut log = PartitionLog::default();
        log.record(1, &sb, &tree());
        assert_eq!(log.replay(1, &sb, |_| ShapeSet::EMPTY), None);
    }

    #[test]
    fn a_cut_file_is_an_error() {
        let mut log = PartitionLog::default();
        log.record(1, &Block::square(0, 0, 6), &tree());
        let bytes = log.to_bytes();
        assert!(PartitionLog::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        assert!(PartitionLog::from_bytes(b"nope").is_err());
    }
}
