//! Restricted native 64×64 intra tile with a DC-only luma residual.
//!
//! This diagnostic subset uses Q255, one 64×64 luma transform and two all-zero
//! 32×32 chroma transforms. It establishes coefficient syntax and reconstruction
//! before general block partitioning and AC coefficient support. It is not a
//! general image encoder. Default CDFs follow AVM v1.0.0 (see `THIRD_PARTY.md`).
use crate::entropy::{Reader, Writer};
use nc_common::decode::{DecodeError, Result};

const PARTITION: &[u16] = &[12276, 0];
const MODE_SET: &[u16] = &[3905, 1746, 1044, 0];
const Y_MODE: &[u16] = &[17593, 12693, 11040, 8670, 6363, 5113, 3908, 0];
const UV_MODE: &[u16] = &[23405, 11811, 9903, 8015, 6357, 4785, 2340, 0];
const Y_SKIP: &[u16] = &[85, 0];
const U_SKIP: &[u16] = &[29245, 0];
const V_SKIP: &[u16] = &[32162, 0];
const EOB: &[u16] = &[26930, 23847, 18314, 11606, 4880, 493, 344, 0];
const BASE: &[u16] = &[4545, 1120, 457, 241, 0];
const SIGN: &[u16] = &[12660, 0];

/// Encode a Q255 DC-only tile, with the frame header supplied separately.
/// # Errors
/// Rejects levels outside the currently supported base-magnitude range.
pub fn encode(level: i32) -> Result<Vec<u8>> {
    if !(-4..=4).contains(&level) {
        return Err(DecodeError::Malformed("AV2 DC probe level outside -4..=4"));
    }
    let mut writer = Writer::new();
    for cdf in [PARTITION, MODE_SET, Y_MODE, UV_MODE] {
        writer.fixed(0, cdf)?;
    }
    writer.fixed(usize::from(level == 0), Y_SKIP)?;
    if level != 0 {
        writer.fixed(0, EOB)?; // EOB one implies DCT_DCT for an intra block.
        writer.fixed(level.unsigned_abs() as usize - 1, BASE)?;
        writer.fixed(usize::from(level < 0), SIGN)?;
    }
    writer.fixed(1, U_SKIP)?;
    writer.fixed(1, V_SKIP)?;
    Ok(writer.finish())
}

fn expect(reader: &mut Reader<'_>, cdf: &[u16], symbol: usize) -> Result<()> {
    if reader.fixed(cdf)? != symbol {
        return Err(DecodeError::Malformed("AV2 tile exceeds DC probe subset"));
    }
    Ok(())
}

/// Decode the restricted tile to planar 8-bit 4:2:0 pixels (6144 bytes).
/// # Errors
/// Rejects unsupported modes, AC coefficients, chroma residuals and truncation.
pub fn decode(tile: &[u8]) -> Result<Vec<u8>> {
    let mut reader = Reader::new(tile)?;
    for cdf in [PARTITION, MODE_SET, Y_MODE, UV_MODE] {
        expect(&mut reader, cdf, 0)?;
    }
    let mut level = 0;
    if reader.fixed(Y_SKIP)? == 0 {
        expect(&mut reader, EOB, 0)?;
        level = reader.fixed(BASE)? as i32 + 1;
        if level > 4 {
            return Err(DecodeError::Malformed(
                "AV2 DC probe range extension unsupported",
            ));
        }
        if reader.fixed(SIGN)? != 0 {
            level = -level;
        }
    }
    expect(&mut reader, U_SKIP, 1)?;
    expect(&mut reader, V_SKIP, 1)?;
    // Q255 = 60<<10, Q3 dequantization then TX64 scale>>2 gives 1920*level.
    // DC-only inverse DCT64: first pass (64*x+32)>>6, second
    // (64*x+4096)>>13. Thus the constant residual is exactly 15*level.
    let mut pixels = vec![(128 + 15 * level) as u8; 4096];
    pixels.resize(6144, 128);
    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_supported_dc_residuals() {
        for level in -4..=4 {
            let tile = encode(level).unwrap();
            let pixels = decode(&tile).unwrap();
            assert_eq!(&pixels[..4096], vec![(128 + 15 * level) as u8; 4096]);
            assert_eq!(&pixels[4096..], vec![128; 2048]);
        }
        assert!(encode(5).is_err());
        assert!(decode(&[]).is_err());
    }
}
