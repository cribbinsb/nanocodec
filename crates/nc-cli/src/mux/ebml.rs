//! EBML element coding for Matroska (RFC 8794 §4-§7, RFC 9559): element
//! IDs as written, variable-size integers, and the element kinds the
//! writer uses.

/// A variable-size integer of `value` in the fewest bytes that hold it
/// (RFC 8794 §4: 7 value bits per length byte; all ones is reserved).
fn vint(value: u64) -> Vec<u8> {
    let length = (1..=8u32)
        .find(|&n| value < (1u64 << (7 * n)) - 1)
        .expect("EBML sizes are below 2^56 - 1");
    let marked = value | (1u64 << (7 * length));
    marked.to_be_bytes()[8 - length as usize..].to_vec()
}

/// `size` as an 8-byte variable-size integer, for a size patched later.
pub(super) fn vint8(size: u64) -> [u8; 8] {
    let mut bytes = size.to_be_bytes();
    bytes[0] = 0x01;
    bytes
}

/// The ID bytes of `id` (its marker bits included, as IDs are written).
fn id_bytes(id: u32) -> Vec<u8> {
    let bytes = id.to_be_bytes();
    let skip = bytes.iter().take_while(|&&b| b == 0).count();
    bytes[skip..].to_vec()
}

/// A master or binary element: `id`, size, `body`.
pub(super) fn element(id: u32, body: &[u8]) -> Vec<u8> {
    let mut out = id_bytes(id);
    out.extend(vint(body.len() as u64));
    out.extend_from_slice(body);
    out
}

/// An unsigned integer element in the fewest bytes.
pub(super) fn uint(id: u32, value: u64) -> Vec<u8> {
    let bytes = value.to_be_bytes();
    let skip = bytes.iter().take_while(|&&b| b == 0).count().min(7);
    element(id, &bytes[skip..])
}

/// An unsigned integer element always 8 bytes long, for a value patched
/// later; its value is the last 8 bytes.
pub(super) fn uint8(id: u32, value: u64) -> Vec<u8> {
    element(id, &value.to_be_bytes())
}

/// A string element.
pub(super) fn string(id: u32, value: &str) -> Vec<u8> {
    element(id, value.as_bytes())
}

/// An 8-byte float element; its value is the last 8 bytes.
pub(super) fn float(id: u32, value: f64) -> Vec<u8> {
    element(id, &value.to_be_bytes())
}

/// A `Void` element (0xEC) of exactly `total` bytes, `total` >= 2.
pub(super) fn void(total: usize) -> Vec<u8> {
    // One ID byte and a one-byte size hold up to 126 payload bytes.
    assert!((2..=128).contains(&total), "Void spans 2-128 bytes here");
    let mut out = vec![0xEC];
    out.extend(vint(total as u64 - 2));
    out.resize(total, 0);
    out
}

/// The header (ID and 8-byte size placeholder) of a master element
/// whose size is patched when it ends.
pub(super) fn open(id: u32) -> Vec<u8> {
    let mut out = id_bytes(id);
    out.extend_from_slice(&vint8(0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_ids() {
        assert_eq!(vint(0), [0x80]);
        assert_eq!(vint(126), [0xFE]);
        assert_eq!(vint(127), [0x40, 0x7F]);
        assert_eq!(uint(0xD7, 1), [0xD7, 0x81, 1]);
        assert_eq!(uint(0x002A_D7B1, 1_000_000)[..4], [0x2A, 0xD7, 0xB1, 0x83]);
        assert_eq!(void(5), [0xEC, 0x83, 0, 0, 0]);
        assert_eq!(vint8(5), [1, 0, 0, 0, 0, 0, 0, 5]);
    }
}
