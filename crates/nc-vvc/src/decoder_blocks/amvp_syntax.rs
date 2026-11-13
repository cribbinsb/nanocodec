//! Both-list AMVP syntax, affine control points, symmetric MVD and AMVR.
//! Reuses the existing MVD arithmetic reader; BSD-Clear `VVdeC` syntax reference.
use super::{DecodeError, Inter, Parser, Reader, Result, Unit, inter_syntax::Prediction};
use crate::decoder_headers::{ReferenceEntry, SliceHeader};
impl Parser<'_> {
    pub(super) fn amvp_syntax(&mut self, r: &mut Reader<'_>, unit: &mut Unit) -> Result<Inter> {
        let n = unit.node;
        let direction = if self.header.kind == 1 {
            1
        } else if n.width + n.height > 12
            && r.bin(
                &mut self.inter_models.direction
                    [7 - ((n.width.ilog2() + n.height.ilog2() + 1) >> 1) as usize],
            )?
        {
            3
        } else {
            1 + u8::from(r.bin(&mut self.inter_models.direction[5])?)
        };
        let context = self.affine_context(unit);
        let points = if self.sequence.tools.affine
            && n.width >= 16
            && n.height >= 16
            && r.bin(&mut self.inter_models.affine[context])?
        {
            2 + u8::from(
                self.sequence.tools.affine_six && r.bin(&mut self.inter_models.affine_type)?,
            )
        } else {
            1
        };
        let symmetric = if direction == 3
            && points == 1
            && self.sequence.tools.smvd
            && !self.header.picture.motion.mvd_l1_zero
        {
            if let Some(indices) = symmetric_references(self.header) {
                if r.bin(&mut self.inter_models.symmetric)? {
                    Some(indices)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        unit.smvd = symmetric.is_some();
        let mut lists: [Option<Prediction>; 2] = [None; 2];
        for list in 0..2 {
            if direction & (1 << list) == 0 {
                continue;
            }
            let reference = if let Some(indices) = symmetric {
                indices[list]
            } else {
                self.reference_index(r, list)?
            };
            let mut mvd = [[0; 2]; 3];
            if list == 1 && symmetric.is_some() {
                mvd[0] = lists[0].unwrap().mvd[0].map(|v| -v);
            } else if !(list == 1 && direction == 3 && self.header.picture.motion.mvd_l1_zero) {
                for vector in &mut mvd[..usize::from(points)] {
                    *vector = crate::decoder_syntax::read_mvd(r, &mut self.contexts)?;
                }
            }
            lists[list] = Some(Prediction {
                reference,
                mvp: r.bin(&mut self.contexts.mvp)?,
                mvd,
            });
        }
        let nonzero = lists
            .iter()
            .flatten()
            .any(|p| p.mvd.iter().flatten().any(|&v| v != 0));
        let precision = self.precision(r, points > 1, nonzero)?;
        let weight = if direction == 3 && self.sequence.tools.bcw && n.width * n.height >= 256 {
            self.bipred_weight(r)?
        } else {
            0
        };
        Ok(Inter::Amvp {
            lists,
            points,
            precision,
            weight,
        })
    }
    fn reference_index(&mut self, r: &mut Reader<'_>, list: usize) -> Result<u8> {
        let count = self.header.active[list];
        if count == 0 {
            return Err(DecodeError::Malformed("VVC empty active reference list"));
        }
        let mut reference = 0;
        if count > 1 && r.bin(&mut self.ref_index[0])? {
            reference = 1;
            if count > 2 && r.bin(&mut self.ref_index[1])? {
                reference = 2;
                while reference + 1 < count && r.bypass()? {
                    reference += 1;
                }
            }
        }
        Ok(reference)
    }
    fn precision(&mut self, r: &mut Reader<'_>, affine: bool, nonzero: bool) -> Result<u8> {
        if !nonzero {
            return Ok(0);
        }
        let c = &mut self.inter_models.precision;
        if affine {
            return Ok(if self.sequence.tools.affine_amvr && r.bin(&mut c[2])? {
                1 + u8::from(r.bin(&mut c[3])?)
            } else {
                0
            });
        }
        Ok(if self.sequence.tools.amvr && r.bin(&mut c[0])? {
            if r.bin(&mut c[4])? {
                1 + u8::from(r.bin(&mut c[1])?)
            } else {
                3
            }
        } else {
            0
        })
    }
    fn bipred_weight(&mut self, r: &mut Reader<'_>) -> Result<u8> {
        if !r.bin(&mut self.inter_models.weight)? {
            return Ok(0);
        }
        let low_delay =
            self.header
                .references
                .iter()
                .zip(self.header.active)
                .all(|(list, count)| {
                    list.entries
                        .iter()
                        .take(usize::from(count))
                        .all(|r| matches!(r, ReferenceEntry::Short(d) if *d <= 0))
                });
        let count = if low_delay { 5 } else { 3 };
        let mut index = 1;
        while index + 1 < count && r.bypass()? {
            index += 1;
        }
        Ok([0, 3, 2, 4, 1][index])
    }
}
fn symmetric_references(header: &SliceHeader) -> Option<[u8; 2]> {
    for signs in [[-1, 1], [1, -1]] {
        let selected = std::array::from_fn::<_, 2, _>(|list| {
            header.references[list]
                .entries
                .iter()
                .take(usize::from(header.active[list]))
                .enumerate()
                .filter_map(|(index, entry)| match entry {
                    ReferenceEntry::Short(delta) if delta.signum() == signs[list] => {
                        Some((delta.unsigned_abs(), index as u8))
                    }
                    _ => None,
                })
                .min_by_key(|&(distance, _)| distance)
                .map(|(_, index)| index)
        });
        if let [Some(a), Some(b)] = selected {
            return Some([a, b]);
        }
    }
    None
}
