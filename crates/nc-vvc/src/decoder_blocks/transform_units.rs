//! General transform-unit geometry and residual syntax, using shared block/tiling helpers.
use super::{DecodeError, Node, Parser, Reader, Result, Unit, coeff};
use nc_encoder::Block;

#[derive(Debug)]
pub(super) struct Transform {
    pub block: Block,
    pub chroma_block: Option<Block>,
    pub joint: u8,
    pub skip: [bool; 3],
    pub levels: [Vec<i32>; 3],
}
impl Transform {
    fn empty(block: Block) -> Self {
        Self {
            block,
            chroma_block: Some(block),
            joint: 0,
            skip: [false; 3],
            levels: std::array::from_fn(|_| Vec::new()),
        }
    }
}

/// Predict the full CU, then overlay its one coded SBT region through the
/// ordinary reconstruction loop. The first entry consumes no residual syntax.
pub(super) fn sbt_units(node: Node, selection: super::sbt::Sbt) -> Vec<Transform> {
    vec![
        Transform::empty(Block::rect(
            node.x,
            node.y,
            node.width.ilog2() as u8,
            node.height.ilog2() as u8,
        )),
        Transform::empty(selection.coded(node)),
    ]
}

/// ISP luma subpartitions share the existing geometry/flag helpers. Chroma
/// belongs to the final subpartition and retains the full CU dimensions.
pub(super) fn isp_units(node: Node, split: crate::isp::Split) -> Vec<Transform> {
    let cu = Block::rect(
        node.x,
        node.y,
        node.width.ilog2() as u8,
        node.height.ilog2() as u8,
    );
    let mut units: Vec<_> = crate::isp::general_parts(&cu, split)
        .map(|block| {
            let mut transform = Transform::empty(block);
            transform.chroma_block = None;
            transform
        })
        .collect();
    units.last_mut().unwrap().chroma_block = Some(cu);
    units
}

/// Preserve CU geometry for contexts while tiling coefficient storage separately.
/// Up to 64x64 uses the same transform partitioner as our encoder/subset decoder.
pub(super) fn empty_units(node: Node, _channel: usize, max_log2: u8) -> Result<Vec<Transform>> {
    let block = Block::rect(
        node.x,
        node.y,
        node.width.ilog2() as u8,
        node.height.ilog2() as u8,
    );
    if ![5, 6].contains(&max_log2) {
        return Err(DecodeError::Unsupported("VVC general transform size"));
    }
    let mut units = Vec::new();
    let region_log2 = max_log2 + 1;
    let region_size = 1 << region_log2;
    for y in (node.y..node.y + node.height).step_by(region_size) {
        for x in (node.x..node.x + node.width).step_by(region_size) {
            let region = Block::rect(
                x,
                y,
                block.log2_width.min(region_log2),
                block.log2_height.min(region_log2),
            );
            units.extend(crate::transform_tree::units(&region, max_log2).map(Transform::empty));
        }
    }
    Ok(units)
}

impl Parser<'_> {
    pub(super) fn residuals(&mut self, reader: &mut Reader<'_>, unit: &mut Unit) -> Result<()> {
        let mut transforms = std::mem::take(&mut unit.transforms);
        let split = transforms.len() != 1;
        let mut earlier = Vec::new();
        let count = transforms.len();
        for (index, transform) in transforms.iter_mut().enumerate() {
            if unit.sbt.is_some() && index == 0 {
                continue;
            }
            let isp = unit.isp.map(|_| (index, count, earlier.as_slice()));
            self.residual_unit(reader, unit, transform, split, isp)?;
            earlier.push(!transform.levels[0].is_empty());
        }
        unit.transforms = transforms;
        Ok(())
    }
    fn residual_unit(
        &mut self,
        reader: &mut Reader<'_>,
        unit: &Unit,
        transform: &mut Transform,
        split: bool,
        isp: Option<(usize, usize, &[bool])>,
    ) -> Result<()> {
        let intra = unit.inter.is_none();
        let mut cbf = [false; 3];
        if (unit.channel != 0 || unit.combined) && transform.chroma_block.is_some() {
            cbf[1] = self.coded_flag(reader, unit, 1, false)?;
            cbf[2] = self.coded_flag(reader, unit, 2, cbf[1])?;
        }
        if unit.channel == 0 {
            cbf[0] = if let Some((index, count, earlier)) = isp {
                crate::isp::code_luma_flag_count(
                    reader,
                    &mut self.contexts,
                    index,
                    count,
                    earlier,
                    false,
                )?
            } else {
                (!intra && (!split || unit.sbt.is_some()) && !cbf[1] && !cbf[2])
                    || self.coded_flag(reader, unit, 0, false)?
            };
        }
        let mask = 2 * u8::from(cbf[1]) + u8::from(cbf[2]);
        if self.sequence.transforms.joint_cbcr
            && ((intra && mask != 0) || mask == 3)
            && reader.bin(&mut self.joint[usize::from(mask - 1)])?
        {
            transform.joint = mask;
        }
        for (plane, present) in cbf.into_iter().enumerate() {
            if present && !(plane == 2 && transform.joint == 3) {
                self.transform_skip(reader, unit, transform, plane)?;
                let channel = usize::from(plane != 0);
                let block = if plane == 0 {
                    transform.block
                } else {
                    transform.chroma_block.unwrap()
                };
                transform.levels[plane] =
                    if transform.skip[plane] && !self.header.ts_residual_disabled {
                        super::skip_coeff::read(
                            reader,
                            &mut self.skip_coefficients,
                            (block.width() >> channel, block.height() >> channel),
                            unit.bdpcm[channel] != 0,
                        )?
                    } else {
                        coeff::read(
                            reader,
                            &mut self.coefficients,
                            block.width() >> channel,
                            block.height() >> channel,
                            channel,
                            self.header.dependent_quant,
                            self.header.sign_hiding,
                            plane == 0
                                && unit.sbt.is_some()
                                && self.sequence.transforms.mts[0]
                                && block.width() <= 32
                                && block.height() <= 32,
                        )?
                    };
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transforms_visit_64_regions_in_normative_order() {
        let units = empty_units(Node::root(0, 0, 128), 0, 5).unwrap();
        let positions: Vec<_> = units
            .iter()
            .map(|u| (u.block.x / 32, u.block.y / 32))
            .collect();
        assert_eq!(
            positions,
            [
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 0),
                (3, 0),
                (2, 1),
                (3, 1),
                (0, 2),
                (1, 2),
                (0, 3),
                (1, 3),
                (2, 2),
                (3, 2),
                (2, 3),
                (3, 3)
            ]
        );
        assert!(
            units
                .iter()
                .all(|u| u.block.width() == 32 && u.block.height() == 32)
        );
    }
    #[test]
    fn chroma_splits_follow_luma_transform_limit() {
        let node = Node::root(0, 0, 64);
        assert_eq!(empty_units(node, 1, 5).unwrap().len(), 4);
        assert_eq!(empty_units(node, 1, 6).unwrap().len(), 1);
        assert_eq!(empty_units(node, 0, 6).unwrap()[0].block.width(), 64);
    }
}
