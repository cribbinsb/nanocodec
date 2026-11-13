//! H.266 §7.3.2 non-VCL RBSP validation for access-unit decoding.
//!
//! Supplemental messages do not change reconstructed samples. Validate their
//! framing without allocating or interpreting display/application metadata.
use nc_common::decode::{BitReader, DecodeError, Result};

pub(super) const AUD: u8 = 20;
pub(super) const PREFIX_SEI: u8 = 23;
pub(super) const SUFFIX_SEI: u8 = 24;
pub(super) const FILLER: u8 = 25;

pub(super) fn validate(kind: u8, bytes: &[u8]) -> Result<()> {
    match kind {
        AUD => {
            let mut reader = BitReader::new(bytes);
            reader.read_bit()?; // aud_irap_or_gdr_flag
            if reader.read_bits(3)? > 2 {
                return Err(DecodeError::Malformed("VVC AUD picture type"));
            }
            reader.trailing_bits()?;
            if reader.remaining_bits() != 0 {
                return Err(DecodeError::Malformed("VVC AUD trailing bytes"));
            }
            Ok(())
        }
        FILLER => {
            let (&last, body) = bytes.split_last().ok_or(DecodeError::Truncated)?;
            if last != 0x80 || body.iter().any(|&byte| byte != 0xff) {
                return Err(DecodeError::Malformed("VVC filler data"));
            }
            Ok(())
        }
        PREFIX_SEI | SUFFIX_SEI => sei(bytes),
        _ => Err(DecodeError::Unsupported("VVC metadata NAL type")),
    }
}

fn sei(mut bytes: &[u8]) -> Result<()> {
    // SEI payload headers are byte-aligned. At least one message precedes
    // rbsp_trailing_bits; the terminating 0x80 cannot be mistaken for a header.
    loop {
        extended_value(&mut bytes)?; // payloadType, unknown types are ignored
        let size = extended_value(&mut bytes)?;
        bytes = bytes.get(size..).ok_or(DecodeError::Truncated)?;
        if bytes == [0x80] {
            return Ok(());
        }
        if bytes.is_empty() {
            return Err(DecodeError::Truncated);
        }
    }
}

fn extended_value(bytes: &mut &[u8]) -> Result<usize> {
    let mut value = 0usize;
    loop {
        let (&byte, rest) = bytes.split_first().ok_or(DecodeError::Truncated)?;
        *bytes = rest;
        value = value
            .checked_add(usize::from(byte))
            .ok_or(DecodeError::Malformed("VVC SEI length overflow"))?;
        if byte != 0xff {
            return Ok(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_sei_messages_and_extensions() {
        assert!(validate(PREFIX_SEI, &[5, 1, 9, 0xff, 1, 0, 0x80]).is_ok());
        let mut extended = vec![5, 0xff, 1];
        extended.extend([0; 256]);
        extended.push(0x80);
        assert!(validate(SUFFIX_SEI, &extended).is_ok());
        for end in 0..extended.len() {
            assert!(validate(SUFFIX_SEI, &extended[..end]).is_err());
        }
        assert!(validate(PREFIX_SEI, &[0x80]).is_err());
        assert!(validate(PREFIX_SEI, &[5, 0, 0x80, 0]).is_err());
    }

    #[test]
    fn aud_and_filler_require_complete_trailing_bits() {
        for pic_type in 0..=2 {
            assert!(validate(AUD, &[pic_type << 4 | 8]).is_ok());
        }
        assert!(validate(AUD, &[0x38]).is_err());
        assert!(validate(AUD, &[8, 0]).is_err());
        assert!(validate(FILLER, &[0xff, 0xff, 0x80]).is_ok());
        assert!(validate(FILLER, &[0xff, 0]).is_err());
        assert!(validate(FILLER, &[]).is_err());
    }
}

#[cfg(test)]
mod access_unit_tests {
    use super::*;
    use nc_common::{Picture, yuv};
    use nc_encoder::{Codec, Encoder, EncoderConfig};

    #[test]
    fn metadata_preserves_reconstruction_and_reference_state() {
        let config = EncoderConfig {
            codec: Codec::Vvc,
            width: 32,
            height: 32,
            intra_period: 30,
            quantizer: 32,
            frame_rate: nc_encoder::FrameRate { num: 30, den: 1 },
            temporal_layers: 1,
            block_log2: 5,
            key_qp_offset: None,
            signal: nc_encoder::StreamSignal::default(),
            params: nc_encoder::ParamOverrides::default().coding_every_frame(),
        };
        let mut encoder = Encoder::<crate::VvcBackend>::new(config).unwrap();
        let source = Picture::new(32, 32, 0).unwrap();
        let mut decoder = crate::Decoder::new();
        let mut streaming = crate::Decoder::new();
        for _ in 0..3 {
            let packet = encoder.encode(&source);
            let mut decorated = Vec::new();
            crate::headers::nal(AUD, &[0x18], &mut decorated);
            crate::headers::nal(PREFIX_SEI, &[5, 0, 0x80], &mut decorated);
            decorated.extend_from_slice(&packet.data);
            crate::headers::nal(FILLER, &[0xff, 0x80], &mut decorated);
            crate::headers::nal(SUFFIX_SEI, &[5, 0, 0x80], &mut decorated);
            let decoded = decoder.decode(&decorated).unwrap();
            let mut shown = None;
            for nal in nc_common::decode::annex_b_units(&decorated).unwrap() {
                if let Some(picture) = streaming.push_nal(nal).unwrap() {
                    assert!(shown.is_none());
                    shown = Some(picture);
                }
            }
            streaming.flush().unwrap();
            let mut streamed = Vec::new();
            yuv::write_frame(&mut streamed, &shown.unwrap()).unwrap();
            let mut actual = Vec::new();
            let mut expected = Vec::new();
            yuv::write_frame(&mut actual, &decoded).unwrap();
            yuv::write_frame(&mut expected, encoder.reconstruction()).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(streamed, expected);
        }
    }
}
