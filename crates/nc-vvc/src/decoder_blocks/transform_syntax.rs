//! H.266 CU-level MTS/LFNST selection through the shared CABAC reader.
//! Context initialization and coefficient constraints follow BSD-Clear `VVdeC`.
use super::{Context, Parser, Reader, Result, Unit, coeff, initialization::bank};

pub(super) struct Models {
    lfnst: [Context; 3],
    mts: [Context; 4],
}
impl Models {
    pub fn new(qp: u32, kind: u8) -> Self {
        Self {
            lfnst: bank(
                [[52, 37, 27], [37, 45, 27], [28, 52, 42]],
                [9, 9, 10],
                qp,
                kind,
            ),
            mts: bank(
                [[45, 25, 27, 0], [45, 40, 27, 0], [29, 0, 28, 0]],
                [8, 0, 9, 0],
                qp,
                kind,
            ),
        }
    }
}
#[derive(Default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent normative coefficient constraints"
)]
struct Constraints {
    skip: bool,
    lfnst_outside: bool,
    lfnst_ac: bool,
    mts_outside: bool,
    mts_ac: bool,
}
impl Constraints {
    fn collect(unit: &Unit) -> Self {
        let mut result = Self::default();
        for transform in &unit.transforms {
            for (plane, levels) in transform
                .levels
                .iter()
                .enumerate()
                .filter(|(_, v)| !v.is_empty())
            {
                if transform.skip[plane] {
                    result.skip = true;
                    continue;
                }
                let channel = usize::from(plane != 0);
                let block = if plane == 0 {
                    transform.block
                } else {
                    transform.chroma_block.unwrap()
                };
                let (width, height) = (block.width() >> channel, block.height() >> channel);
                let Some(last) = coeff::last_scan(levels, width, height) else {
                    continue;
                };
                if width >= 4 && height >= 4 {
                    let limit = if width == height && [4, 8].contains(&width) {
                        7
                    } else {
                        15
                    };
                    result.lfnst_outside |= last > limit;
                    result.lfnst_ac |= last >= 1;
                }
                if plane == 0 {
                    result.mts_ac |= last >= 1;
                    result.mts_outside |= levels
                        .iter()
                        .enumerate()
                        .any(|(p, &v)| v != 0 && (p % width >= 16 || p / width >= 16));
                }
            }
        }
        result
    }
}
impl Parser<'_> {
    pub(super) fn transform_selection(
        &mut self,
        reader: &mut Reader<'_>,
        unit: &mut Unit,
    ) -> Result<()> {
        let tools = &self.sequence.transforms;
        if !tools.lfnst && !tools.mts[0] {
            return Ok(());
        }
        let constraints = Constraints::collect(unit);
        if self.lfnst_allowed(unit, &constraints) {
            let separate = self.sequence.chroma && !unit.combined;
            if reader.bin(&mut self.transform_models.lfnst[usize::from(separate)])? {
                unit.lfnst = 1 + u8::from(reader.bin(&mut self.transform_models.lfnst[2])?);
            }
        }
        let explicit = self.sequence.transforms.mts[if unit.inter.is_none() { 1 } else { 2 }];
        let first = &unit.transforms[0];
        if explicit
            && unit.channel == 0
            && unit.isp.is_none()
            && unit.sbt.is_none()
            && unit.bdpcm[0] == 0
            && unit.node.width <= 32
            && unit.node.height <= 32
            && unit.lfnst == 0
            && !first.skip[0]
            && !constraints.mts_outside
            && constraints.mts_ac
            && reader.bin(&mut self.transform_models.mts[0])?
        {
            unit.mts = 2;
            for context in &mut self.transform_models.mts[1..] {
                if !reader.bin(context)? {
                    break;
                }
                unit.mts += 1;
            }
        }
        Ok(())
    }
    fn lfnst_allowed(&self, unit: &Unit, constraints: &Constraints) -> bool {
        if !self.sequence.transforms.lfnst
            || unit.inter.is_some()
            || constraints.skip
            || constraints.lfnst_outside
            || (!constraints.lfnst_ac && unit.isp.is_none())
        {
            return false;
        }
        let max_size = 1 << self.sequence.max_transform_log2;
        if unit.node.width > max_size
            || unit.node.height > max_size
            || (unit.mip.is_some() && (unit.node.width < 16 || unit.node.height < 16))
        {
            return false;
        }
        let first = unit.transforms[0].block;
        first.width() >> unit.channel >= 4 && first.height() >> unit.channel >= 4
    }
}
