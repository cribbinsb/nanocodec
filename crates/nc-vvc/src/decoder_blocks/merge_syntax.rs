//! General merge syntax (regular, MMVD, subblock, CIIP and geometric partitions).
//! H.266 §7.3.11, checked against BSD-Clear `VVdeC` `CABACReader`.
use super::{Inter, Parser, Reader, Result, Unit, inter_syntax::index};
impl Parser<'_> {
    pub(super) fn merge_syntax(&mut self, r: &mut Reader<'_>, unit: &Unit) -> Result<Inter> {
        let tools = &self.sequence.tools;
        let n = unit.node;
        let context = self.affine_context(unit);
        let subblocks = if tools.affine {
            tools.max_subblock_merge
        } else {
            u8::from(tools.subblock_mvp && self.header.picture.motion.temporal)
        };
        if subblocks > 0
            && n.width >= 8
            && n.height >= 8
            && r.bin(&mut self.inter_models.subblock[context])?
        {
            return Ok(Inter::Subblock(index(
                r,
                &mut self.inter_models.subblock_index,
                subblocks,
            )?));
        }
        let ciip =
            tools.ciip && !unit.skip && n.width < 128 && n.height < 128 && n.width * n.height >= 64;
        let geo = tools.gpm
            && self.header.kind == 0
            && (8..=64).contains(&n.width)
            && (8..=64).contains(&n.height)
            && n.width < 8 * n.height
            && n.height < 8 * n.width;
        let regular =
            !(ciip || geo) || r.bin(&mut self.inter_models.regular[usize::from(!unit.skip)])?;
        if regular {
            if tools.mmvd && r.bin(&mut self.inter_models.mmvd)? {
                return Ok(Inter::Mmvd {
                    base: index(r, &mut self.inter_models.mmvd_base, tools.max_merge.min(2))?,
                    step: index(r, &mut self.inter_models.mmvd_step, 8)?,
                    direction: r.bypass_bits(2)? as u8,
                });
            }
        } else if ciip && (!geo || r.bin(&mut self.inter_models.ciip)?) {
            return Ok(Inter::Ciip(index(
                r,
                &mut self.contexts.merge_idx,
                tools.max_merge,
            )?));
        } else {
            let split = r.bypass_bits(6)? as u8;
            let first = index(r, &mut self.contexts.merge_idx, tools.max_gpm)?;
            let second = index(r, &mut self.contexts.merge_idx, tools.max_gpm - 1)?;
            return Ok(Inter::Geometric {
                split,
                indices: [first, second + u8::from(second >= first)],
            });
        }
        Ok(Inter::Merge(index(
            r,
            &mut self.contexts.merge_idx,
            tools.max_merge,
        )?))
    }
}
