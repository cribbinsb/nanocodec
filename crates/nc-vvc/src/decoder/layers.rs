//! Which NAL units the decoder decodes: the base layer of a stream, without
//! the units a decoder must ignore (H.266 §7.4.2.2).
//!
//! A Main 10 decoder decodes output layer set 0 (§C.6 sub-bitstream
//! extraction with no external means), which holds only the first layer the
//! VPS lists (`vps_layer_id[0]`, §7.4.3.3); a stream without a VPS has one
//! layer. Units of other layers are dropped, as are reserved and unspecified
//! NAL unit types, `nuh_reserved_zero_bit` equal to 1 and `nuh_layer_id`
//! above 55, which §7.4.2.2 tells decoders to ignore.
use nc_common::decode::{BitReader, DecodeError, Result, unescape_rbsp};

/// `nal_unit_type` of video parameter sets (H.266 Table 5).
pub(super) const VPS_NUT: u8 = 14;
/// The highest `nuh_layer_id` in use; larger values are reserved (§7.4.2.2).
const MAX_LAYER_ID: u8 = 55;
/// `nuh_layer_id` width (§7.3.1.2).
const LAYER_ID_BITS: u32 = 6;

/// Whether `nal_unit_type` is reserved or unspecified (H.266 Table 5:
/// `RSV_VCL_4..6`, `RSV_IRAP_11`, `RSV_NVCL_26..27`, `UNSPEC_28..31`).
fn reserved_type(kind: u8) -> bool {
    matches!(kind, 4..=6 | 11 | 26..=31)
}

/// The layer chosen for decoding, set by the first VPS or, in a stream
/// without one, by the first unit.
#[derive(Debug, Default)]
pub(super) struct LayerFilter {
    target: Option<u8>,
}
impl LayerFilter {
    /// Whether `unit` (a NAL unit without start code) is decoded; `false`
    /// drops it. VPSs are kept (they set the target layer).
    /// # Errors
    /// A truncated header, `forbidden_zero_bit` set, `nuh_temporal_id_plus1`
    /// equal to 0, or a malformed VPS start.
    pub fn keeps(&mut self, unit: &[u8]) -> Result<bool> {
        let [first, second, ..] = *unit else {
            return Err(DecodeError::Truncated);
        };
        if first & 0x80 != 0 {
            return Err(DecodeError::Malformed("VVC forbidden_zero_bit"));
        }
        if second.trailing_zeros() >= 3 {
            return Err(DecodeError::Malformed("VVC nuh_temporal_id_plus1"));
        }
        let reserved_bit = first & 0x40 != 0;
        let layer = first & 0x3f;
        let kind = second >> 3;
        if reserved_bit || layer > MAX_LAYER_ID || reserved_type(kind) {
            return Ok(false);
        }
        if kind == VPS_NUT {
            self.target = Some(first_layer(&unescape_rbsp(&unit[2..])?)?);
            return Ok(true);
        }
        Ok(*self.target.get_or_insert(layer) == layer)
    }
}

/// `vps_layer_id[0]` of a VPS RBSP (§7.3.2.3).
fn first_layer(rbsp: &[u8]) -> Result<u8> {
    let mut r = BitReader::new(rbsp);
    r.read_bits(4)?; // vps_video_parameter_set_id
    let max_layers_minus1 = r.read_bits(6)?;
    let max_sublayers_minus1 = r.read_bits(3)?;
    if max_layers_minus1 > 0 && max_sublayers_minus1 > 0 {
        r.read_bit()?; // vps_default_ptl_dpb_hrd_max_tid_flag
    }
    if max_layers_minus1 > 0 {
        r.read_bit()?; // vps_all_independent_layers_flag
    }
    let layer = r.read_bits(LAYER_ID_BITS)? as u8;
    if layer > MAX_LAYER_ID {
        return Err(DecodeError::Malformed("VVC vps_layer_id"));
    }
    Ok(layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A NAL unit header: layer, type, temporal id plus 1.
    fn header(layer: u8, kind: u8, tid1: u8) -> [u8; 3] {
        [layer, (kind << 3) | tid1, 0x80]
    }

    #[test]
    fn reserved_types_and_bits_are_dropped() {
        let mut filter = LayerFilter::default();
        for kind in [4, 5, 6, 11, 26, 27, 28, 31] {
            assert!(!filter.keeps(&header(0, kind, 1)).unwrap(), "type {kind}");
        }
        assert!(!filter.keeps(&[0x40, (15 << 3) | 1, 0]).unwrap());
        assert!(!filter.keeps(&header(56, 0, 1)).unwrap());
        assert!(filter.keeps(&header(0, 15, 1)).unwrap());
    }

    #[test]
    fn only_the_first_layer_is_kept() {
        let mut filter = LayerFilter::default();
        assert!(filter.keeps(&header(0, 15, 1)).unwrap());
        assert!(!filter.keeps(&header(1, 15, 1)).unwrap());
        assert!(filter.keeps(&header(0, 0, 1)).unwrap());
    }

    #[test]
    fn the_vps_names_the_base_layer() {
        let mut filter = LayerFilter::default();
        // id 0, two layers, one sublayer, all independent, vps_layer_id[0] 3.
        // Bits: 0000 000001 000 1 000011 ...
        let vps = [0, VPS_NUT << 3 | 1, 0b0000_0000, 0b0100_0100, 0b0011_0000];
        assert!(filter.keeps(&vps).unwrap());
        assert!(!filter.keeps(&header(0, 15, 1)).unwrap());
        assert!(filter.keeps(&header(3, 15, 1)).unwrap());
    }

    #[test]
    fn forbidden_bit_and_zero_temporal_id_are_errors() {
        let mut filter = LayerFilter::default();
        assert!(filter.keeps(&[0x80, 1, 0]).is_err());
        assert!(filter.keeps(&[0, 0, 0]).is_err());
        assert!(filter.keeps(&[0]).is_err());
    }
}
