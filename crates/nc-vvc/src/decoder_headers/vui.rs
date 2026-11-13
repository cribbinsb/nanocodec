//! H.266 VUI and HRD metadata; bounded parsing without pixel-path state.
use super::{DecodeError, Reader, Result};
use nc_common::ColourInfo;

pub(super) fn parse(r: &mut Reader<'_>) -> Result<ColourInfo> {
    let size = r.ue(1023, "VVC VUI payload size")? as usize + 1;
    r.align()?;
    let payload =
        r.0.remaining_bytes()?
            .get(..size)
            .ok_or(DecodeError::Truncated)?;
    let mut v = Reader::new(payload);
    let progressive = v.flag()?;
    let interlaced = v.flag()?;
    v.flag()?;
    v.flag()?;
    if v.flag()? {
        v.flag()?;
        if v.bits(8)? == 255 {
            v.bits(16)?;
            v.bits(16)?;
        }
    }
    if v.flag()? {
        v.flag()?;
    }
    let mut colour = ColourInfo::default();
    if v.flag()? {
        colour.description = Some(nc_common::colour::read_code_points(&mut v.0)?);
        colour.full_range = v.flag()?;
    }
    if v.flag()? {
        v.ue(6, "VVC chroma sample location")?;
        if !progressive || interlaced {
            v.ue(6, "VVC bottom chroma sample location")?;
        }
    }
    // Payload extension bits may be reserved for future VUI definitions.
    if v.0.remaining_bits() != 0 {
        v.finish(true)?;
    }
    r.0.skip_bits(size * 8)?;
    Ok(colour)
}

pub(super) fn hrd(r: &mut Reader<'_>, max_layer: usize) -> Result<()> {
    if r.bits(32)? == 0 || r.bits(32)? == 0 {
        return Err(DecodeError::Malformed("VVC HRD time base"));
    }
    let present = [r.flag()?, r.flag()?];
    let any = present.contains(&true);
    let mut du = false;
    let mut cpb = 1;
    if any {
        r.flag()?;
        du = r.flag()?;
        if du {
            r.bits(8)?;
        }
        r.bits(4)?;
        r.bits(4)?;
        if du {
            r.bits(4)?;
        }
        cpb = r.ue(31, "VVC HRD CPB count")? + 1;
    }
    let all = max_layer > 0 && r.flag()?;
    for _ in if all { 0 } else { max_layer }..=max_layer {
        let fixed = r.flag()? || r.flag()?;
        if fixed {
            r.ue(2047, "VVC HRD elemental duration")?;
        } else if any && cpb == 1 {
            r.flag()?;
        }
        for enabled in present {
            if !enabled {
                continue;
            }
            let mut previous = None;
            for _ in 0..cpb {
                let rate = r.ue(u32::MAX - 1, "VVC HRD bit rate")?;
                let size = r.ue(u32::MAX - 1, "VVC HRD CPB size")?;
                if previous.is_some_and(|(old_rate, old_size)| rate <= old_rate || size > old_size)
                {
                    return Err(DecodeError::Malformed("VVC HRD CPB order"));
                }
                previous = Some((rate, size));
                if du {
                    r.ue(u32::MAX - 1, "VVC HRD DU size")?;
                    r.ue(u32::MAX - 1, "VVC HRD DU rate")?;
                }
                r.flag()?;
            }
        }
    }
    Ok(())
}
