//! H.266 §7.3.10 reference-picture-list structures, with bounded list sizes.
use super::{DecodeError, PictureParameters, Reader, Result, Sequence};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Entry {
    Short(i32),
    Long(Option<u32>),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct List {
    pub long_in_header: bool,
    pub entries: Vec<Entry>,
}
impl List {
    pub fn parse(
        r: &mut Reader<'_>,
        long_term: bool,
        weighted: bool,
        poc_bits: u32,
        local: bool,
    ) -> Result<Self> {
        // Annex A MaxDpbSize (16) + 13 permits at most 29 entries.
        let count = r.ue(29, "VVC reference list length")?;
        let long_in_header = long_term && (local || count == 0 || r.flag()?);
        let mut entries = Vec::with_capacity(count as usize);
        let mut previous = 0i32;
        for index in 0..count {
            if !long_term || r.flag()? {
                let magnitude =
                    r.ue(32767, "VVC short-term POC delta")? + u32::from(!weighted || index == 0);
                let negative = magnitude != 0 && r.flag()?;
                previous += if negative {
                    -(magnitude as i32)
                } else {
                    magnitude as i32
                };
                entries.push(Entry::Short(previous));
            } else {
                entries.push(Entry::Long(if long_in_header {
                    None
                } else {
                    Some(r.bits(poc_bits)?)
                }));
            }
        }
        Ok(Self {
            long_in_header,
            entries,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Selected {
    pub entries: Vec<Entry>,
    pub msb_cycles: Vec<Option<u32>>,
}
pub(super) fn select(
    r: &mut Reader<'_>,
    s: &Sequence,
    p: &PictureParameters,
) -> Result<[Selected; 2]> {
    let mut output = [Selected::default(), Selected::default()];
    let mut from_sps = false;
    let mut previous_index = 0;
    for (index, selected) in output.iter_mut().enumerate() {
        let available = &s.tools.rpls[index];
        from_sps = if available.is_empty() {
            false
        } else if index == 0 || p.rpl1_index {
            r.flag()?
        } else {
            from_sps
        };
        let mut list = if from_sps {
            let selected_index = if available.len() == 1 {
                0
            } else if index == 0 || p.rpl1_index {
                r.bits(usize::BITS - (available.len() - 1).leading_zeros())? as usize
            } else {
                previous_index
            };
            previous_index = selected_index;
            available
                .get(selected_index)
                .ok_or(DecodeError::Malformed("VVC RPL index"))?
                .clone()
        } else {
            List::parse(
                r,
                s.tools.long_term,
                s.tools.weighted.contains(&true),
                u32::from(s.poc_bits),
                true,
            )?
        };
        let mut cumulative = 0u32;
        let limit = 1u32 << (32 - s.poc_bits);
        for entry in &mut list.entries {
            let mut msb = None;
            if let Entry::Long(poc) = entry {
                if list.long_in_header {
                    *poc = Some(r.bits(u32::from(s.poc_bits))?);
                }
                if r.flag()? {
                    cumulative = cumulative
                        .checked_add(r.ue(limit, "VVC long-term MSB delta")?)
                        .filter(|&v| v <= limit)
                        .ok_or(DecodeError::Malformed("VVC long-term MSB sum"))?;
                    msb = Some(cumulative);
                }
            }
            selected.msb_cycles.push(msb);
        }
        selected.entries = list.entries;
    }
    Ok(output)
}
