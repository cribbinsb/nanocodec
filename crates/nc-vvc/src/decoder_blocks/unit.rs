//! General CU modes and residual syntax using the existing CABAC helpers.
use super::{Constraint, DecodeError, Inter, Node, Parser, Reader, Result, Unit, intra};
impl Parser<'_> {
    pub(super) fn unit(
        &mut self,
        r: &mut Reader<'_>,
        node: Node,
        channel: usize,
        combined: bool,
        constraint: Constraint,
    ) -> Result<()> {
        if node.outside(self.dimensions) {
            return Err(DecodeError::Malformed("VVC coding unit outside picture"));
        }
        let mut unit = Unit {
            node,
            channel,
            combined,
            inter: None,
            motion: None,
            affine_models: [None; 2],
            geometric: None,
            subblocks: Vec::new(),
            skip: false,
            ciip_weight: 0,
            bdpcm: [0; 2],
            mip: None,
            reference_line: 0,
            isp: None,
            sbt: None,
            prof: self.sequence.tools.prof && !self.header.picture.motion.prof_disabled,
            smvd: false,
            bdof: false,
            dmvr: Vec::new(),
            lfnst: 0,
            mts: 0,
            luma_mode: intra::PLANAR,
            chroma_mode: intra::PLANAR,
            transforms: super::transform_units::empty_units(
                node,
                channel,
                self.sequence.max_transform_log2,
            )?,
        };
        let intra = self.prediction(r, &mut unit, constraint)?;
        if unit.skip
            || (matches!(unit.inter, Some(Inter::Amvp { .. }))
                && !r.bin(&mut self.contexts.root_cbf)?)
        {
            self.finish_unit(unit);
            return Ok(());
        }
        if intra && channel == 0 {
            self.luma_prediction(r, &mut unit)?;
        }
        if intra && (channel != 0 || combined) {
            self.chroma_prediction(r, &mut unit)?;
        }
        self.sbt_mode(r, &mut unit)?;
        self.residuals(r, &mut unit)?;
        self.transform_selection(r, &mut unit)?;
        self.finish_unit(unit);
        Ok(())
    }
    fn finish_unit(&mut self, unit: Unit) {
        #[cfg(test)]
        if std::env::var_os("NC_VVC_TRACE_UNITS").is_some() {
            eprintln!(
                "NC_UNIT {} {} {} {} {} {} {} {} {}",
                unit.node.x,
                unit.node.y,
                unit.node.width,
                unit.node.height,
                unit.channel,
                if unit.channel == 0 {
                    unit.luma_mode
                } else {
                    unit.chroma_mode
                },
                match unit.isp {
                    None => 0,
                    Some(crate::isp::Split::Horizontal) => 1,
                    Some(crate::isp::Split::Vertical) => 2,
                },
                unit.lfnst,
                if unit.transforms[0].skip[0] {
                    1
                } else {
                    unit.mts
                }
            );
        }
        let node = unit.node;
        let index = self.units.len();
        for y in (node.y..node.y + node.height).step_by(4) {
            for x in (node.x..node.x + node.width).step_by(4) {
                for channel in if unit.combined {
                    0..2
                } else {
                    unit.channel..unit.channel + 1
                } {
                    self.grid[channel][y / 4 * self.stride + x / 4] = Some(index);
                }
            }
        }
        self.units.push(unit);
    }
}
