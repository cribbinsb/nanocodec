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
            affine_fields: [None; 2],
            geometric: None,
            block_vector: None,
            subblocks: Vec::new(),
            skip: false,
            ciip_weight: 0,
            bdpcm: [0; 2],
            mip: None,
            reference_line: 0,
            isp: None,
            sbt: None,
            palette: None,
            prof: self.sequence.tools.prof && !self.header.picture.motion.prof_disabled,
            smvd: false,
            bdof: false,
            dmvr: Vec::new(),
            lfnst: 0,
            mts: 0,
            luma_mode: intra::PLANAR,
            chroma_mode: intra::PLANAR,
            chroma_mip: false,
            act: false,
            qp: 0,
            chroma_offsets: [0; 3],
            tile: self.ctu.tile,
            slice: self.slice,
            transforms: Vec::new(),
        };
        let luma_qp = self.begin_unit_qp(&mut unit);
        let intra = self.prediction(r, &mut unit, constraint)?;
        if intra && self.palette_flag(r, &unit)? {
            return self.palette_cu(r, unit, luma_qp);
        }
        if intra {
            self.colour_transform_flag(r, &mut unit)?;
        }
        if unit.skip
            || (matches!(
                unit.inter,
                Some(Inter::Amvp { .. } | Inter::Ibc(super::ibc::BlockVector::Predicted { .. }))
            ) && !r.bin(&mut self.contexts.root_cbf)?)
        {
            unit.transforms = super::transform_units::empty_units(
                node,
                channel,
                self.sequence.max_transform_log2,
            )?;
            self.end_unit_qp(&mut unit, luma_qp);
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
        if !intra {
            self.colour_transform_flag(r, &mut unit)?;
        }
        if unit.transforms.is_empty() {
            unit.transforms = super::transform_units::empty_units(
                node,
                channel,
                self.sequence.max_transform_log2,
            )?;
        }
        self.residuals(r, &mut unit)?;
        self.end_unit_qp(&mut unit, luma_qp);
        self.transform_selection(r, &mut unit)?;
        self.finish_unit(unit);
        Ok(())
    }
    /// The rest of a palette CU, which codes no prediction modes or residual
    /// (§7.3.11.5); out of line, as most CUs are not palette CUs.
    #[inline(never)]
    fn palette_cu(
        &mut self,
        r: &mut Reader<'_>,
        mut unit: Unit,
        luma_qp: Option<i8>,
    ) -> Result<()> {
        unit.transforms = super::transform_units::empty_units(
            unit.node,
            unit.channel,
            self.sequence.max_transform_log2,
        )?;
        self.palette_unit(r, &mut unit)?;
        self.end_unit_qp(&mut unit, luma_qp);
        self.finish_unit(unit);
        Ok(())
    }
    /// `cu_act_enabled_flag` (§7.3.11.5): single-tree CUs only, after the
    /// palette flag of an intra CU or the coded flag and SBT of another.
    fn colour_transform_flag(&mut self, r: &mut Reader<'_>, unit: &mut Unit) -> Result<()> {
        if self.sequence.tools.act && unit.combined {
            unit.act = r.bin(&mut self.act)?;
        }
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
        let index = self.units.len() as u32;
        let cols = node.width.div_ceil(4);
        for channel in if unit.combined {
            0..2
        } else {
            unit.channel..unit.channel + 1
        } {
            for y in (node.y..node.y + node.height).step_by(4) {
                let base = (y >> 2) * self.stride + (node.x >> 2);
                self.grid[channel][base..base + cols].fill(index);
            }
        }
        self.units.push(unit);
    }
}
