//! H.266 §7.3.3 profile/tier/level and general constraint information.
use super::{DecodeError, Reader, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Profile {
    pub id: u8,
    pub high_tier: bool,
    pub frame_only: bool,
    pub constraints: Vec<bool>,
    pub levels: Vec<u8>,
    pub subprofiles: Vec<u32>,
}
impl Profile {
    pub fn parse(r: &mut Reader<'_>, max_layer: usize) -> Result<Self> {
        let id = r.bits(7)? as u8;
        if !matches!(id, 1 | 65) {
            return Err(DecodeError::Unsupported(
                "VVC profile outside Main 10/Still Picture",
            ));
        }
        let high_tier = r.flag()?;
        let level = r.bits(8)? as u8;
        let frame_only = r.flag()?;
        if r.flag()? {
            return Err(DecodeError::Unsupported("VVC multilayer profile"));
        }
        let constraints = if r.flag()? {
            // GCI has 71 fixed bits followed by a bounded reserved-bit count.
            let bits = r.flags(71)?;
            let depth_constraint = bits[3..7].iter().fold(0, |v, &b| (v << 1) | u8::from(b));
            if depth_constraint > 8 {
                return Err(DecodeError::Malformed("VVC GCI bit depth"));
            }
            let reserved = r.bits(8)?;
            r.zeroes(reserved)?;
            bits
        } else {
            Vec::new()
        };
        r.align()?;
        let present = r.flags(max_layer)?;
        r.align()?;
        let mut levels = vec![level; max_layer + 1];
        for (layer, &present) in (0..max_layer).rev().zip(&present) {
            levels[layer] = if present {
                r.bits(8)? as u8
            } else {
                levels[layer + 1]
            };
        }
        let count = r.bits(8)?;
        let subprofiles = (0..count).map(|_| r.bits(32)).collect::<Result<_>>()?;
        Ok(Self {
            id,
            high_tier,
            frame_only,
            constraints,
            levels,
            subprofiles,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::BitWriter;
    #[test]
    fn general_constraints_sublayer_levels_and_subprofiles() {
        let mut w = BitWriter::new();
        w.put_bits(1, 7);
        w.put_bit(false);
        w.put_bits(120, 8);
        w.put_bit(true);
        w.put_bit(false);
        w.put_bit(true);
        for _ in 0..71 {
            w.put_bit(false);
        }
        w.put_bits(3, 8);
        w.put_bits(0, 3);
        w.byte_align_zero();
        w.put_bit(false);
        w.put_bit(true);
        w.byte_align_zero();
        w.put_bits(90, 8);
        w.put_bits(1, 8);
        w.put_bits(0x1234_5678, 32);
        let bytes = w.into_bytes();
        let profile = Profile::parse(&mut Reader::new(&bytes), 2).unwrap();
        assert_eq!(profile.levels, [90, 120, 120]);
        assert_eq!(profile.constraints.len(), 71);
        assert_eq!(profile.subprofiles, [0x1234_5678]);
        for end in 0..bytes.len() {
            assert!(Profile::parse(&mut Reader::new(&bytes[..end]), 2).is_err());
        }
    }
}
