//! Header codes of AV2 beyond the shared [`BitReader`]'s: the uniform
//! (truncated binary) code and signed literals (AV2 §4.11; dav2d
//! `getbits.c`).
use nc_common::decode::{BitReader, Result};

/// A value in `0..count` (`count > 1`), the low values one bit shorter
/// (dav2d `dav2d_get_uniform`, AVM `avm_rb_read_primitive_quniform`).
/// # Errors
/// Truncation.
pub fn uniform(bits: &mut BitReader<'_>, count: u32) -> Result<u32> {
    debug_assert!(count > 1);
    let length = count.ilog2() + 1;
    let short = (1 << length) - count;
    let prefix = bits.read_bits(length - 1)?;
    Ok(if prefix < short {
        prefix
    } else {
        (prefix << 1) - short + bits.read_bits(1)?
    })
}

/// `value` coded relative to `reference` in `0..count` (AV2 §4.11
/// `inverse_recenter`).
fn inverse_recenter(reference: u32, value: u32) -> u32 {
    if value > reference << 1 {
        value
    } else if value & 1 == 0 {
        (value >> 1) + reference
    } else {
        reference - ((value + 1) >> 1)
    }
}

/// A value in `0..count` coded as a sub-exponential code with first group
/// width `k`, recentred on `reference` (dav2d `dav2d_get_bits_subexp_u`,
/// AVM `avm_rb_read_primitive_refsubexpfin`).
/// # Errors
/// Truncation.
pub fn subexp(bits: &mut BitReader<'_>, reference: u32, count: u32, k: u32) -> Result<u32> {
    let mut value = 0;
    for group in 0.. {
        let width = if group == 0 { k } else { k + group - 1 };
        let size = 1 << width;
        if count <= value + 3 * size {
            let remaining = count - value;
            if remaining > 1 {
                value += uniform(bits, remaining)?;
            }
            break;
        }
        if !bits.read_bit()? {
            value += bits.read_bits(width)?;
            break;
        }
        value += size;
    }
    Ok(if reference * 2 <= count {
        inverse_recenter(reference, value)
    } else {
        count - 1 - inverse_recenter(count - 1 - reference, value)
    })
}

/// A signed value in `-(count - 1)..count` coded with [`subexp`] around
/// `reference` (dav2d `dav2d_get_bits_subexp`).
/// # Errors
/// Truncation.
pub fn signed_subexp(bits: &mut BitReader<'_>, reference: i32, count: u32) -> Result<i32> {
    let offset = count as i32 - 1;
    let value = subexp(
        bits,
        (reference + offset) as u32,
        count + offset as u32,
        SIGNED_SUBEXP_K,
    )?;
    Ok(value as i32 - offset)
}

/// First group width of [`signed_subexp`].
const SIGNED_SUBEXP_K: u32 = 3;

/// A two's-complement literal of `width` bits (dav2d `dav2d_get_sbits`).
/// # Errors
/// Truncation.
pub fn signed(bits: &mut BitReader<'_>, width: u32) -> Result<i32> {
    let value = bits.read_bits(width)?;
    let shift = 32 - width;
    Ok(((value << shift) as i32) >> shift)
}

/// One flag as 0 or 1.
/// # Errors
/// Truncation.
pub fn flag(bits: &mut BitReader<'_>) -> Result<u8> {
    Ok(u8::from(bits.read_bit()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_and_signed_codes() {
        // count 5: values 0..2 take two bits, 3 and 4 three.
        let mut bits = BitReader::new(&[0b0001_1011, 0b0111_0000]);
        assert_eq!(uniform(&mut bits, 5).unwrap(), 0);
        assert_eq!(uniform(&mut bits, 5).unwrap(), 1);
        assert_eq!(uniform(&mut bits, 5).unwrap(), 2);
        assert_eq!(uniform(&mut bits, 5).unwrap(), 3);
        assert_eq!(uniform(&mut bits, 5).unwrap(), 4);
        let mut bits = BitReader::new(&[0b1110_0111]);
        assert_eq!(signed(&mut bits, 4).unwrap(), -2);
        assert_eq!(signed(&mut bits, 4).unwrap(), 7);
        assert!(signed(&mut bits, 1).is_err());
    }
}
