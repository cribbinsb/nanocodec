//! H.266 §7.3.8 weighted prediction tables for 8-bit pictures.
use super::{DecodeError, PictureParameters, Reader, Result, Sequence};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Weights {
    pub denominators: [u8; 3],
    pub lists: [Vec<[[i16; 2]; 3]>; 2],
}
impl Weights {
    pub(super) fn parse(
        r: &mut Reader<'_>,
        s: &Sequence,
        p: &PictureParameters,
        counts: [u8; 2],
    ) -> Result<Self> {
        let luma = r.ue(7, "VVC weight denominator")? as i32;
        let chroma = if s.chroma {
            luma + r.se(-luma, 7 - luma, "VVC chroma weight denominator")?
        } else {
            0
        };
        let denominators = [luma as u8, chroma as u8, chroma as u8];
        let mut lists = [Vec::new(), Vec::new()];
        let mut total_flags = 0;
        for (index, list) in lists.iter_mut().enumerate() {
            let count = if index == 1 && (!p.weighted[1] || counts[1] == 0) {
                0
            } else if p.location.weights_in_ph {
                let count = r.ue(u32::from(counts[index].min(15)), "VVC weight count")?;
                if count == 0 {
                    return Err(DecodeError::Malformed("VVC empty weight table"));
                }
                count as usize
            } else {
                usize::from(counts[index])
            };
            let luma_present = r.flags(count)?;
            let chroma_present = if s.chroma {
                r.flags(count)?
            } else {
                vec![false; count]
            };
            for (has_luma, has_chroma) in luma_present.into_iter().zip(chroma_present) {
                total_flags += u32::from(has_luma) + 2 * u32::from(has_chroma);
                let mut components = denominators.map(|d| [1 << d, 0]);
                if has_luma {
                    components[0][0] += r.se(-128, 127, "VVC luma weight delta")? as i16;
                    components[0][1] = r.se(-128, 127, "VVC luma weight offset")? as i16;
                }
                if has_chroma {
                    for component in &mut components[1..] {
                        component[0] += r.se(-128, 127, "VVC chroma weight delta")? as i16;
                        let delta = r.se(-512, 508, "VVC chroma weight offset")?;
                        component[1] = (128 + delta - ((128 * i32::from(component[0])) >> chroma))
                            .clamp(-128, 127) as i16;
                    }
                }
                list.push(components);
            }
        }
        if total_flags > 24 {
            return Err(DecodeError::Malformed("VVC weighted component count"));
        }
        Ok(Self {
            denominators,
            lists,
        })
    }
}
