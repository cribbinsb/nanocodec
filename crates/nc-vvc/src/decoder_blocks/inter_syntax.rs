//! General P-CU motion signalling; arithmetic decoding is shared with our subset.
use super::{Constraint, DecodeError, Parser, Reader, Result, Unit};

#[derive(Debug, Clone, Copy)]
pub(super) struct Prediction {
    pub reference: u8,
    pub mvp: bool,
    pub mvd: [[i32; 2]; 3],
}
#[derive(Debug, Clone, Copy)]
pub(super) enum Inter {
    Merge(u8),
    Subblock(u8),
    Mmvd {
        base: u8,
        step: u8,
        direction: u8,
    },
    Ciip(u8),
    Geometric {
        split: u8,
        indices: [u8; 2],
    },
    Amvp {
        lists: [Option<Prediction>; 2],
        points: u8,
        precision: u8,
        weight: u8,
    },
}
impl Inter {
    pub fn affine(self) -> bool {
        matches!(self, Self::Subblock(_) | Self::Amvp { points: 2.., .. })
    }
}
impl Parser<'_> {
    pub(super) fn prediction(
        &mut self,
        r: &mut Reader<'_>,
        unit: &mut Unit,
        constraint: Constraint,
    ) -> Result<bool> {
        if self.header.kind == 2
            || constraint == Constraint::Intra
            || unit.channel == 1
            || unit.node.width * unit.node.height == 16
        {
            return Ok(true);
        }
        let n = unit.node;
        let neighbours = [
            self.neighbour(n.x.wrapping_sub(1), n.y, 0),
            self.neighbour(n.x, n.y.wrapping_sub(1), 0),
        ];
        let skip_context = neighbours
            .iter()
            .filter(|u| u.is_some_and(|u| u.skip))
            .count();
        let intra_context = usize::from(
            neighbours
                .iter()
                .any(|u| u.is_some_and(|u| u.inter.is_none())),
        );
        unit.skip = r.bin(&mut self.contexts.skip[skip_context])?;
        if !unit.skip
            && constraint != Constraint::Inter
            && r.bin(&mut self.contexts.intra[intra_context])?
        {
            return Ok(true);
        }
        unit.inter = Some(if unit.skip || r.bin(&mut self.contexts.merge)? {
            self.merge_syntax(r, unit)?
        } else {
            self.amvp_syntax(r, unit)?
        });
        if matches!(unit.inter, Some(Inter::Ciip(_))) {
            unit.ciip_weight = 1 + [
                self.neighbour(n.x.wrapping_sub(1), n.y + n.height - 1, 0),
                self.neighbour(n.x + n.width - 1, n.y.wrapping_sub(1), 0),
            ]
            .into_iter()
            .filter(|u| u.is_some_and(|u| u.inter.is_none()))
            .count() as u8;
        }
        Ok(false)
    }
    pub(super) fn affine_context(&self, unit: &Unit) -> usize {
        let n = unit.node;
        [
            self.neighbour(n.x.wrapping_sub(1), n.y, 0),
            self.neighbour(n.x, n.y.wrapping_sub(1), 0),
        ]
        .into_iter()
        .filter(|u| u.and_then(|u| u.inter).is_some_and(Inter::affine))
        .count()
    }
}

pub(super) fn index(r: &mut Reader<'_>, context: &mut super::Context, count: u8) -> Result<u8> {
    if count == 0 {
        return Err(DecodeError::Malformed("VVC empty candidate list"));
    }
    let mut index = 0;
    if count > 1 && r.bin(context)? {
        index = 1;
        while index + 1 < count && r.bypass()? {
            index += 1;
        }
    }
    Ok(index)
}
