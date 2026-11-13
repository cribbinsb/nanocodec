//! OBU framing (AV1 §5.3): split a temporal unit into its OBUs with their
//! layer ids, and tile groups into their tiles (§5.11.1).
use super::frame_header::{LayerIds, TileInfo};
use nc_common::decode::{BitReader, DecodeError};

/// OBU types (AV1 §6.2.2).
pub(crate) const OBU_SEQUENCE_HEADER: u8 = 1;
pub(crate) const OBU_TEMPORAL_DELIMITER: u8 = 2;
pub(crate) const OBU_FRAME_HEADER: u8 = 3;
pub(crate) const OBU_TILE_GROUP: u8 = 4;
pub(crate) const OBU_FRAME: u8 = 6;
pub(crate) const OBU_REDUNDANT_FRAME_HEADER: u8 = 7;
pub(crate) const OBU_TILE_LIST: u8 = 8;

/// `obu_header()` bits.
const FORBIDDEN_BIT: u8 = 0x80;
const TYPE_SHIFT: u8 = 3;
const TYPE_MASK: u8 = 15;
const EXTENSION_FLAG: u8 = 0b100;
const HAS_SIZE_FIELD: u8 = 0b10;
/// `obu_extension_header()`: `temporal_id` (3 bits), `spatial_id` (2 bits).
const TEMPORAL_ID_SHIFT: u8 = 5;
const SPATIAL_ID_SHIFT: u8 = 3;
const SPATIAL_ID_MASK: u8 = 3;
/// Bound on OBUs per temporal unit: memory for untrusted input.
const MAX_OBUS: usize = 1024;

/// One OBU: type, layer ids (zero without an extension header) and payload.
pub(crate) struct Obu<'a> {
    pub kind: u8,
    pub layer: LayerIds,
    /// `obu_extension_flag`: OBUs without one belong to every layer.
    pub has_extension: bool,
    pub payload: &'a [u8],
    /// Complete raw OBU slice (header, optional extension, size and payload).
    pub raw: &'a [u8],
}

/// Split `data` (a temporal unit in low-overhead format, §5.2) into OBUs.
///
/// # Errors
/// Malformed headers or sizes running past the data.
pub(crate) fn split(mut data: &[u8]) -> Result<Vec<Obu<'_>>, DecodeError> {
    let mut out = Vec::new();
    while !data.is_empty() {
        let obu_start = data;
        let (&header, rest) = data.split_first().ok_or(DecodeError::Truncated)?;
        data = rest;
        if header & FORBIDDEN_BIT != 0 {
            return Err(DecodeError::Malformed("AV1 obu_forbidden_bit"));
        }
        let mut layer = LayerIds::default();
        let has_extension = header & EXTENSION_FLAG != 0;
        if has_extension {
            let (&extension, rest) = data.split_first().ok_or(DecodeError::Truncated)?;
            layer.temporal = extension >> TEMPORAL_ID_SHIFT;
            layer.spatial = (extension >> SPATIAL_ID_SHIFT) & SPATIAL_ID_MASK;
            data = rest;
        }
        let payload;
        if header & HAS_SIZE_FIELD != 0 {
            let mut reader = BitReader::new(data);
            let size = usize::try_from(reader.read_leb128()?)
                .map_err(|_| DecodeError::Malformed("AV1 OBU size"))?;
            data = &data[reader.position() / 8..];
            payload = data.get(..size).ok_or(DecodeError::Truncated)?;
            data = &data[size..];
        } else {
            payload = data;
            data = &[];
        }
        let raw_len = obu_start.len() - data.len();
        if out.len() == MAX_OBUS {
            return Err(DecodeError::Unsupported("AV1 OBUs per temporal unit"));
        }
        out.push(Obu {
            kind: (header >> TYPE_SHIFT) & TYPE_MASK,
            layer,
            has_extension,
            payload,
            raw: &obu_start[..raw_len],
        });
    }
    Ok(out)
}

/// The tiles one tile group OBU carries: `tg_start..=tg_end` and the data
/// after its header.
pub(crate) struct TileGroup<'a> {
    pub start: usize,
    pub end: usize,
    pub data: &'a [u8],
}

/// `tile_group_obu()` header (§5.11.1); `data` starts at the tile group.
///
/// # Errors
/// Truncated data or tile numbers outside the frame.
pub(crate) fn tile_group<'a>(
    data: &'a [u8],
    tiles: &TileInfo,
) -> Result<TileGroup<'a>, DecodeError> {
    let count = tiles.count();
    let mut r = BitReader::new(data);
    let (mut start, mut end) = (0, count - 1);
    if count > 1 && r.read_bit()? {
        let bits = tiles.cols_log2 + tiles.rows_log2;
        start = r.read_bits(bits)? as usize;
        end = r.read_bits(bits)? as usize;
    }
    r.align_zero()?;
    if start > end || end >= count {
        return Err(DecodeError::Malformed("AV1 tile group range"));
    }
    Ok(TileGroup {
        start,
        end,
        data: r.remaining_bytes()?,
    })
}

impl<'a> TileGroup<'a> {
    /// The coded tiles of the group, in order (§5.11.1): each but the last
    /// follows its `tile_size_minus_1`, `tile_size_bytes` little-endian
    /// bytes; the last takes the rest.
    ///
    /// # Errors
    /// A tile size beyond the data.
    pub(crate) fn tiles(&self, tile_size_bytes: usize) -> Result<Vec<&'a [u8]>, DecodeError> {
        const TRUNCATED: DecodeError = DecodeError::Malformed("AV1 tile size");
        let mut rest = self.data;
        let mut tiles = Vec::with_capacity(self.end - self.start + 1);
        for _ in self.start..self.end {
            let (size, after) = rest.split_at_checked(tile_size_bytes).ok_or(TRUNCATED)?;
            let size = size
                .iter()
                .rev()
                .fold(0, |value, &byte| (value << 8) | usize::from(byte))
                + 1;
            let (tile, after) = after.split_at_checked(size).ok_or(TRUNCATED)?;
            tiles.push(tile);
            rest = after;
        }
        tiles.push(rest);
        Ok(tiles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiles 1..=3 of a group: two led by 2-byte little-endian
    /// `tile_size_minus_1`, the last taking the rest; a size past the end
    /// is malformed.
    #[test]
    fn tile_group_splits_into_sized_tiles() {
        let data = [0x01, 0x00, 0xa, 0xb, 0x00, 0x00, 0xc, 0xd, 0xe];
        let group = TileGroup {
            start: 1,
            end: 3,
            data: &data,
        };
        let tiles = group.tiles(2).unwrap();
        assert_eq!(tiles, [&[0xa, 0xb][..], &[0xc], &[0xd, 0xe]]);
        let truncated = TileGroup {
            data: &data[..3],
            ..group
        };
        assert!(truncated.tiles(2).is_err());
    }
}
