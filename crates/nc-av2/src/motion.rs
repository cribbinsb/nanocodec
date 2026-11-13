//! Final AV2 quarter-pel motion-vector difference syntax.
//!
//! Uses AVM v1.0.0 shell/pair magnitude coding and the separate row/column
//! bypass signs from `write_inter_mode`. Optional tile-local adaptation; no adaptive MVD or sign
//! derivation. Inputs are differences in eighth-pel units, not absolute MVs.
//! AVM terms: BSD-3-Clause-Clear and separate AOM patent license.
use crate::entropy::{Reader, Writer};
use nc_common::decode::{DecodeError, Result};

/// Largest supported even eighth-pel component (signed 16-bit AVM difference).
pub const MAX_COMPONENT: i32 = 32766;
const CLASS_LOW: &[u16] = &[25271, 15467, 8920, 5330, 3373, 1889, 765, 0];
const CLASS_HIGH: &[u16] = &[14299, 5341, 1206, 116, 44, 40, 36, 0];
const OFFSET: [u16; 16] = [
    14825, 13834, 13840, 14072, 13724, 12406, 12342, 10205, 10578, 9310, 6541, 2003, 16384, 16384,
    16384, 16384,
];
const LOW_OFFSET: [u16; 2] = [18181, 11802];
const COL_GREATER: [u16; 2] = [27105, 27912];
const COL_INDEX: [u16; 4] = [19323, 19227, 18723, 19880];

/// Approximate bits of a vector difference component: class prefix plus
/// offset bits.
const COMPONENT_BASE_BITS: f32 = 2.0;
/// Eighth-sample difference units per quarter sample.
const EIGHTHS_PER_QUARTER: u32 = 2;

/// Approximate bits of coding `vector` (eighth-pel row, column) as a
/// difference from `predictor`, for ranking predictors and modes.
#[must_use]
pub fn difference_bits(vector: (i32, i32), predictor: (i32, i32)) -> f32 {
    [vector.0 - predictor.0, vector.1 - predictor.1]
        .iter()
        .map(|delta| {
            COMPONENT_BASE_BITS
                + 2.0 * ((delta.unsigned_abs() / EIGHTHS_PER_QUARTER) as f32 + 1.0).log2()
        })
        .sum()
}

fn write_uniform(writer: &mut Writer, count: u32, value: u32) -> Result<()> {
    if count <= 1 {
        return Ok(());
    }
    let bits = count.ilog2() + 1;
    let split = (1 << bits) - count;
    if value < split {
        writer.literal(value, bits - 1)
    } else {
        writer.literal(split + ((value - split) >> 1), bits - 1)?;
        writer.literal((value - split) & 1, 1)
    }
}

fn read_uniform(reader: &mut Reader<'_>, count: u32) -> Result<u32> {
    if count <= 1 {
        return Ok(0);
    }
    let bits = count.ilog2() + 1;
    let split = (1 << bits) - count;
    let prefix = reader.literal(bits - 1)?;
    if prefix < split {
        Ok(prefix)
    } else {
        Ok(split + 2 * (prefix - split) + reader.literal(1)?)
    }
}

/// Encode one signed quarter-pel vector difference using fixed default CDFs.
/// Signs follow both magnitudes, row first. Zero is representable; the caller
/// must enforce any mode-specific nonzero or reference-vector restrictions.
/// # Errors
/// Rejects odd components and magnitudes above [`MAX_COMPONENT`] before writing.
pub fn write_mv(writer: &mut Writer, row: i32, col: i32) -> Result<()> {
    if [row, col]
        .iter()
        .any(|&v| !(-MAX_COMPONENT..=MAX_COMPONENT).contains(&v) || v & 1 != 0)
    {
        return Err(DecodeError::Malformed(
            "AV2 quarter-pel MVD component range",
        ));
    }
    let row_abs = row.unsigned_abs() / 2;
    let col_abs = col.unsigned_abs() / 2;
    let shell = row_abs + col_abs;
    let class = if shell < 2 { 0 } else { shell.ilog2() as usize };
    let offset = shell - if class == 0 { 0 } else { 1 << class };
    writer.model(usize::from(class >= 8), &[1189, 0], |m| &mut m.mv_set)?;
    writer.model(
        class % 8,
        if class < 8 { CLASS_LOW } else { CLASS_HIGH },
        |m| {
            if class < 8 {
                &mut m.mv_class_low
            } else {
                &mut m.mv_class_high
            }
        },
    )?;
    match class {
        0 | 1 => writer.model(offset as usize, &[LOW_OFFSET[class], 0], |m| {
            &mut m.mv_low_offset[class]
        })?,
        2 => {
            for index in 0..3 {
                let bit = offset != index;
                if index == 0 {
                    writer.model(usize::from(bit), &[19579, 0], |m| &mut m.mv_class2_offset)?;
                } else {
                    writer.literal(u32::from(bit), 1)?;
                }
                if !bit {
                    break;
                }
            }
        }
        _ => {
            for (index, &probability) in OFFSET[..class].iter().enumerate() {
                writer.model(((offset >> index) & 1) as usize, &[probability, 0], |m| {
                    &mut m.mv_offset[index]
                })?;
            }
        }
    }
    if shell != 0 {
        let max_pair = shell / 2;
        let pair = col_abs.min(shell - col_abs);
        for index in 0..max_pair.min(2) {
            let bit = pair != index;
            writer.model(usize::from(bit), &[COL_GREATER[index as usize], 0], |m| {
                &mut m.mv_col_greater[index as usize]
            })?;
            if !bit {
                break;
            }
        }
        if max_pair > 2 && pair >= 2 {
            write_uniform(writer, max_pair - 1, pair - 2)?;
        }
        if pair != max_pair || shell & 1 != 0 {
            writer.model(
                usize::from(col_abs > max_pair),
                &[COL_INDEX[class.min(3)], 0],
                |m| &mut m.mv_col_index[class.min(3)],
            )?;
        }
    }
    for component in [row, col] {
        if component != 0 {
            writer.literal(u32::from(component < 0), 1)?;
        }
    }
    Ok(())
}

/// Decode one signed quarter-pel vector difference in eighth-pel units.
/// # Errors
/// Rejects truncation or decoded values outside the signed 16-bit subset.
pub fn read_mv(reader: &mut Reader<'_>) -> Result<(i32, i32)> {
    let set = reader.model(&[1189, 0], |m| &mut m.mv_set)?;
    let class = reader.model(if set == 0 { CLASS_LOW } else { CLASS_HIGH }, |m| {
        if set == 0 {
            &mut m.mv_class_low
        } else {
            &mut m.mv_class_high
        }
    })? + set * 8;
    let offset = match class {
        0 | 1 => reader.model(&[LOW_OFFSET[class], 0], |m| &mut m.mv_low_offset[class])? as u32,
        2 => {
            let mut value = 0;
            for index in 0..3 {
                let bit = if index == 0 {
                    reader.model(&[19579, 0], |m| &mut m.mv_class2_offset)? as u32
                } else {
                    reader.literal(1)?
                };
                if bit == 0 {
                    break;
                }
                value += 1;
            }
            value
        }
        _ => {
            let mut value = 0;
            for (index, &probability) in OFFSET[..class].iter().enumerate() {
                value |=
                    (reader.model(&[probability, 0], |m| &mut m.mv_offset[index])? as u32) << index;
            }
            value
        }
    };
    let shell = offset + if class == 0 { 0 } else { 1 << class };
    let mut col = 0;
    if shell != 0 {
        let max_pair = shell / 2;
        let mut pair = 0;
        for index in 0..max_pair.min(2) {
            if reader.model(&[COL_GREATER[index as usize], 0], |m| {
                &mut m.mv_col_greater[index as usize]
            })? == 0
            {
                break;
            }
            pair += 1;
        }
        if max_pair > 2 && pair >= 2 {
            pair += read_uniform(reader, max_pair - 1)?;
        }
        col = pair;
        if (pair != max_pair || shell & 1 != 0)
            && reader.model(&[COL_INDEX[class.min(3)], 0], |m| {
                &mut m.mv_col_index[class.min(3)]
            })? != 0
        {
            col = shell - pair;
        }
    }
    let mut components = [((shell - col) * 2) as i32, (col * 2) as i32];
    if components.iter().any(|&v| v > MAX_COMPONENT) {
        return Err(DecodeError::Malformed("AV2 decoded MVD component range"));
    }
    for value in &mut components {
        if *value != 0 && reader.literal(1)? != 0 {
            *value = -*value;
        }
    }
    Ok((components[0], components[1]))
}

#[cfg(test)]
mod tests {
    use super::*;
    const VECTORS: [(i32, i32); 32] = [
        (0, 0),
        (0, 2),
        (2, 0),
        (-2, 2),
        (2, -2),
        (-2, -2),
        (0, 4),
        (2, 4),
        (4, 4),
        (0, 8),
        (2, 8),
        (4, 8),
        (6, 8),
        (8, 8),
        (10, 8),
        (12, 8),
        (14, 8),
        (16, 8),
        (0, 30),
        (2, 30),
        (30, 32),
        (64, -64),
        (126, 128),
        (-254, 256),
        (510, -512),
        (-1022, -1024),
        (2046, 2048),
        (-4094, 4096),
        (8190, -8192),
        (16382, 16384),
        (32766, 0),
        (-32766, 32766),
    ];
    const REFERENCE_BYTES: &[u8] = &[
        3, 206, 1, 31, 144, 52, 175, 118, 137, 100, 215, 185, 59, 23, 224, 21, 143, 77, 81, 97,
        227, 67, 23, 127, 168, 231, 63, 235, 123, 127, 252, 61, 53, 255, 84, 59, 159, 255, 255,
        200, 191, 255, 200, 226, 175, 255, 255, 240, 52, 207, 255, 255, 255, 252, 76, 123, 255,
        255, 255, 255, 226, 101, 235, 255, 255, 188, 174, 188, 165, 127, 255, 255, 242,
    ];

    const ADAPTIVE_REFERENCE_BYTES: &[u8] = &[
        4, 245, 73, 227, 178, 159, 197, 18, 234, 54, 115, 32, 118, 49, 180, 79, 60, 149, 199, 177,
        13, 133, 242, 66, 129, 40, 191, 245, 72, 255, 255, 123, 212, 7, 253, 233, 97, 255, 255,
        253, 191, 255, 246, 244, 155, 255, 255, 222, 74, 127, 255, 255, 255, 7, 139, 255, 255, 255,
        243, 120, 215, 202, 57, 58, 181, 255, 255, 244,
    ];

    #[test]
    fn adaptive_motion_matches_original_c() {
        let mut writer = Writer::new();
        writer.enable_adaptation(100);
        for (row, col) in VECTORS {
            write_mv(&mut writer, row, col).unwrap();
        }
        let packet = writer.finish();
        assert_eq!(packet, ADAPTIVE_REFERENCE_BYTES);
        let mut reader = Reader::new(&packet).unwrap();
        reader.enable_adaptation(100);
        for vector in VECTORS {
            assert_eq!(read_mv(&mut reader).unwrap(), vector);
        }
    }

    #[test]
    fn matches_original_avm_motion_writer() {
        let mut writer = Writer::new();
        for (row, col) in VECTORS {
            write_mv(&mut writer, row, col).unwrap();
        }
        assert_eq!(writer.finish(), REFERENCE_BYTES);
        let mut reader = Reader::new(REFERENCE_BYTES).unwrap();
        for vector in VECTORS {
            assert_eq!(read_mv(&mut reader).unwrap(), vector);
        }
    }

    #[test]
    fn mixed_full_range_roundtrip_and_bounds() {
        let mut vectors = VECTORS.to_vec();
        let mut state = 91_u32;
        for _ in 0..4096 {
            let pair = std::array::from_fn::<_, 2, _>(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state % 32767) as i32 - 16383) * 2
            });
            vectors.push((pair[0], pair[1]));
        }
        let mut writer = Writer::new();
        for &(row, col) in &vectors {
            write_mv(&mut writer, row, col).unwrap();
        }
        let packet = writer.finish();
        let mut reader = Reader::new(&packet).unwrap();
        for vector in vectors {
            assert_eq!(read_mv(&mut reader).unwrap(), vector);
        }
        for component in [i32::MIN, -32768, -3, 1, 32768, i32::MAX] {
            let mut writer = Writer::new();
            assert!(write_mv(&mut writer, component, 0).is_err());
            assert_eq!(writer.finish(), Writer::new().finish());
        }
    }
}
