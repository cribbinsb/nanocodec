//! H.266 §7.3.2.6 adaptation sets, checked against BSD-licensed `VVdeC`.
//! Coefficients remain decoder state; parsing does not enable reconstruction tools.
use super::{DecodeError, Reader, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Adaptation {
    pub id: u8,
    pub chroma: bool,
    pub data: Data,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Data {
    Alf(Box<Alf>),
    Lmcs(Lmcs),
    Scaling(Box<Scaling>),
}
impl Adaptation {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let kind = r.bits(3)?;
        let id = r.bits(5)? as u8;
        let chroma = r.flag()?;
        if id > if kind == 1 { 3 } else { 7 } {
            return Err(DecodeError::Malformed("VVC APS id"));
        }
        let data = match kind {
            0 => Data::Alf(Box::new(Alf::parse(&mut r, chroma)?)),
            1 => Data::Lmcs(Lmcs::parse(&mut r, chroma)?),
            2 => Data::Scaling(Box::new(Scaling::parse(&mut r, chroma)?)),
            _ => return Err(DecodeError::Unsupported("VVC APS type")),
        };
        let extension = r.flag()?;
        r.finish(extension)?;
        Ok(Self { id, chroma, data })
    }
    pub fn kind(&self) -> usize {
        match self.data {
            Data::Alf(_) => 0,
            Data::Lmcs(_) => 1,
            Data::Scaling(_) => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Filter<const N: usize> {
    pub coefficients: [i16; N],
    pub clip: [u8; N],
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Alf {
    pub classes: [u8; 25],
    pub luma: Vec<Filter<12>>,
    pub chroma: Vec<Filter<6>>,
    pub cross: [Vec<[i16; 7]>; 2],
}
impl Alf {
    fn parse(r: &mut Reader<'_>, chroma: bool) -> Result<Self> {
        let present = [
            r.flag()?,
            chroma && r.flag()?,
            chroma && r.flag()?,
            chroma && r.flag()?,
        ];
        if !present.contains(&true) {
            return Err(DecodeError::Malformed("VVC empty ALF APS"));
        }
        let mut classes = [0; 25];
        let luma = if present[0] {
            let nonlinear = r.flag()?;
            let count = r.ue(24, "VVC luma ALF count")? + 1;
            if count > 1 {
                let bits = u32::BITS - (count - 1).leading_zeros();
                for class in &mut classes {
                    let index = r.bits(bits)?;
                    if index >= count {
                        return Err(DecodeError::Malformed("VVC ALF class index"));
                    }
                    *class = index as u8;
                }
            }
            filters(r, count, nonlinear)?
        } else {
            Vec::new()
        };
        let mut chroma_filters = Vec::new();
        if present[1] {
            let nonlinear = r.flag()?;
            let count = r.ue(7, "VVC chroma ALF count")? + 1;
            // Chroma clipping follows each alternative, unlike the luma bank.
            for _ in 0..count {
                chroma_filters.extend(filters(r, 1, nonlinear)?);
            }
        }
        let mut cross: [Vec<[i16; 7]>; 2] = std::array::from_fn(|_| Vec::new());
        for (component, bank) in cross.iter_mut().enumerate() {
            if !present[component + 2] {
                continue;
            }
            let count = r.ue(3, "VVC cross-component ALF count")? + 1;
            for _ in 0..count {
                let mut coefficients = [0; 7];
                for value in &mut coefficients {
                    let code = r.bits(3)?;
                    if code != 0 {
                        *value = signed(r, 1 << (code - 1))?;
                    }
                }
                bank.push(coefficients);
            }
        }
        Ok(Self {
            classes,
            luma,
            chroma: chroma_filters,
            cross,
        })
    }
}
fn signed(r: &mut Reader<'_>, magnitude: i16) -> Result<i16> {
    Ok(if magnitude != 0 && r.flag()? {
        -magnitude
    } else {
        magnitude
    })
}
fn filters<const N: usize>(
    r: &mut Reader<'_>,
    count: u32,
    nonlinear: bool,
) -> Result<Vec<Filter<N>>> {
    let mut bank = Vec::new();
    for _ in 0..count {
        let mut coefficients = [0; N];
        for value in &mut coefficients {
            let magnitude = r.ue(128, "VVC ALF magnitude")? as i16;
            *value = signed(r, magnitude)?;
            if *value > 127 {
                return Err(DecodeError::Malformed("VVC ALF coefficient"));
            }
        }
        bank.push(Filter {
            coefficients,
            clip: [0; N],
        });
    }
    if nonlinear {
        for filter in &mut bank {
            for clip in &mut filter.clip {
                *clip = r.bits(2)? as u8;
            }
        }
    }
    Ok(bank)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Lmcs {
    pub bins: [u8; 2],
    pub width_delta: [i16; 16],
    pub chroma_offset: i16,
}
impl Lmcs {
    fn parse(r: &mut Reader<'_>, chroma: bool) -> Result<Self> {
        let min = r.ue(15, "VVC LMCS first bin")? as usize;
        let max = 15 - r.ue(15, "VVC LMCS last bin")? as usize;
        if min > max {
            return Err(DecodeError::Malformed("VVC LMCS bin range"));
        }
        let precision = r.ue(14, "VVC LMCS precision")? + 1;
        let mut width_delta = [0; 16];
        for delta in &mut width_delta[min..=max] {
            let magnitude = r.bits(precision)? as i16;
            *delta = signed(r, magnitude)?;
        }
        let chroma_offset = if chroma {
            let magnitude = r.bits(3)? as i16;
            signed(r, magnitude)?
        } else {
            0
        };
        Ok(Self {
            bins: [min as u8, max as u8],
            width_delta,
            chroma_offset,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Scaling {
    pub matrices: [Vec<u8>; 28],
    pub dc: [u8; 28],
}
impl Scaling {
    fn parse(r: &mut Reader<'_>, chroma: bool) -> Result<Self> {
        let mut result = Self {
            matrices: std::array::from_fn(|id| vec![16; size(id).pow(2)]),
            dc: [16; 28],
        };
        for id in 0..28 {
            if !chroma && id % 3 != 2 && id != 27 {
                continue;
            }
            let copy = r.flag()?;
            let prediction = copy || r.flag()?;
            let base = if id < 2 {
                0
            } else if id < 8 {
                2
            } else {
                8
            };
            let delta = if prediction && id != base {
                r.ue((id - base) as u32, "VVC scaling predictor")? as usize
            } else {
                0
            };
            let mut matrix = if !prediction {
                vec![8; size(id).pow(2)]
            } else if delta == 0 {
                vec![16; size(id).pow(2)]
            } else {
                result.matrices[id - delta].clone()
            };
            let dc_prediction = if !prediction {
                8
            } else if delta == 0 {
                16
            } else if id - delta > 13 {
                result.dc[id - delta]
            } else {
                matrix[0]
            };
            let mut next = 0;
            if id > 13 {
                if !copy {
                    next = r.se(-128, 127, "VVC scaling DC")?;
                }
                result.dc[id] = coefficient(i32::from(dc_prediction) + next)?;
            }
            if !copy {
                let side = size(id);
                for diagonal in 0..side * 2 - 1 {
                    for x in diagonal.saturating_sub(side - 1)..=diagonal.min(side - 1) {
                        let y = diagonal - x;
                        if !(id > 25 && x >= 4 && y >= 4) {
                            next += r.se(-128, 127, "VVC scaling delta")?;
                        }
                        let pos = y * side + x;
                        matrix[pos] = coefficient(i32::from(matrix[pos]) + next)?;
                    }
                }
            }
            result.matrices[id] = matrix;
        }
        Ok(result)
    }
}
fn size(id: usize) -> usize {
    if id < 2 {
        2
    } else if id < 8 {
        4
    } else {
        8
    }
}
fn coefficient(value: i32) -> Result<u8> {
    let value = (value & 255) as u8;
    if value == 0 {
        Err(DecodeError::Malformed("VVC zero scaling coefficient"))
    } else {
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::BitWriter;

    fn aps(kind: u32, id: u32, chroma: bool, payload: impl FnOnce(&mut BitWriter)) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.put_bits(kind, 3);
        w.put_bits(id, 5);
        w.put_bit(chroma);
        payload(&mut w);
        w.put_bit(false);
        w.trailing_bits();
        w.into_bytes()
    }
    #[test]
    fn alf_signed_limits_and_class_bounds() {
        for (magnitude, negative, valid) in
            [(128, true, true), (128, false, false), (129, true, false)]
        {
            let bytes = aps(0, 7, false, |w| {
                w.put_bit(true); // Luma bank.
                w.put_bit(true); // Nonlinear clipping.
                w.put_ue(0); // One filter.
                w.put_ue(magnitude);
                w.put_bit(negative);
                for _ in 1..12 {
                    w.put_ue(0);
                }
                for _ in 0..12 {
                    w.put_bits(3, 2);
                }
            });
            let result = Adaptation::parse(&bytes);
            assert_eq!(result.is_ok(), valid);
            if let Ok(Adaptation {
                data: Data::Alf(alf),
                ..
            }) = result
            {
                assert_eq!(alf.luma[0].coefficients[0], -128);
                assert_eq!(alf.luma[0].clip, [3; 12]);
                for end in 0..bytes.len() {
                    assert!(Adaptation::parse(&bytes[..end]).is_err());
                }
            }
        }
        let invalid = aps(0, 0, false, |w| {
            w.put_bit(true);
            w.put_bit(false);
            w.put_ue(2); // Three filters: two-bit index 3 is reserved.
            w.put_bits(3, 2);
        });
        assert!(Adaptation::parse(&invalid).is_err());
        assert!(Adaptation::parse(&aps(0, 0, false, |w| w.put_bit(false))).is_err());
    }
    #[test]
    fn lmcs_ids_bins_and_signed_deltas() {
        let payload = |w: &mut BitWriter| {
            w.put_ue(2);
            w.put_ue(12); // Bins 2 and 3.
            w.put_ue(2); // Three-bit magnitudes.
            w.put_bits(3, 3);
            w.put_bit(true);
            w.put_bits(0, 3);
            w.put_bits(2, 3);
            w.put_bit(false);
        };
        let bytes = aps(1, 3, true, payload);
        let Adaptation {
            data: Data::Lmcs(lmcs),
            ..
        } = Adaptation::parse(&bytes).unwrap()
        else {
            panic!()
        };
        assert_eq!(lmcs.bins, [2, 3]);
        assert_eq!(lmcs.width_delta[2], -3);
        assert_eq!(lmcs.chroma_offset, 2);
        assert!(Adaptation::parse(&aps(1, 4, true, payload)).is_err());
        assert!(
            Adaptation::parse(&aps(1, 0, false, |w| {
                w.put_ue(15);
                w.put_ue(1);
            }))
            .is_err()
        );
    }
    #[test]
    fn scaling_prediction_and_zero_rejection() {
        let bytes = aps(2, 0, true, |w| {
            for id in 0..28 {
                w.put_bit(true);
                if !matches!(id, 0 | 2 | 8) {
                    w.put_ue(0);
                }
            }
        });
        let Adaptation {
            data: Data::Scaling(s),
            ..
        } = Adaptation::parse(&bytes).unwrap()
        else {
            panic!()
        };
        assert_eq!(s.dc, [16; 28]);
        assert!(s.matrices.iter().flatten().all(|&v| v == 16));
        let zero = aps(2, 0, true, |w| {
            w.put_bit(false);
            w.put_bit(false);
            w.put_se(-8); // Default prediction 8 plus delta -8 is forbidden.
        });
        assert!(Adaptation::parse(&zero).is_err());
    }
}
