//! H.266 sub-block transforms using shared CU geometry, CABAC and primary transforms.
//! Conditional syntax follows BSD-Clear `VVdeC` `CABACReader`/`UnitTools`.
use super::{Context, Inter, Node, Parser, Reader, Result, Unit, initialization::bank};
use nc_encoder::Block;

#[derive(Debug, Clone, Copy)]
pub(super) struct Sbt {
    horizontal: bool,
    quarter: bool,
    last: bool,
}
impl Sbt {
    pub fn coded(self, node: Node) -> Block {
        let shift = if self.quarter { 2 } else { 1 };
        let width = node.width >> if self.horizontal { 0 } else { shift };
        let height = node.height >> if self.horizontal { shift } else { 0 };
        let x = node.x + if self.last { node.width - width } else { 0 };
        let y = node.y + if self.last { node.height - height } else { 0 };
        Block::rect(x, y, width.ilog2() as u8, height.ilog2() as u8)
    }
    /// The uncoded region may span three quarters of a CU. Its extent is
    /// filter metadata; only the coded power-of-two region enters transforms.
    pub fn uncoded(self, node: Node) -> (usize, usize, usize, usize) {
        let coded = self.coded(node);
        if self.horizontal {
            (
                node.x,
                node.y + if self.last { 0 } else { coded.height() },
                node.width,
                node.height - coded.height(),
            )
        } else {
            (
                node.x + if self.last { 0 } else { coded.width() },
                node.y,
                node.width - coded.width(),
                node.height,
            )
        }
    }
    pub fn primary(self, node: Node) -> [nc_dsp::VvcTransformKind; 2] {
        use nc_dsp::VvcTransformKind::{Dct2, Dct8, Dst7};
        let block = self.coded(node);
        if block.width() > 32 || block.height() > 32 {
            return [Dct2; 2];
        }
        if self.last {
            [Dst7; 2]
        } else if self.horizontal {
            [Dst7, Dct8]
        } else {
            [Dct8, Dst7]
        }
    }
}
pub(super) struct Models {
    flag: [Context; 2],
    quarter: Context,
    horizontal: [Context; 3],
    position: Context,
}
impl Models {
    pub fn new(qp: u32, kind: u8) -> Self {
        Self {
            flag: bank([[41, 57], [56, 57], [35, 35]], [1, 5], qp, kind),
            quarter: bank::<1, 1>([[42], [42], [35]], [10], qp, kind)[0],
            horizontal: bank([[35, 51, 27], [20, 43, 12], [35; 3]], [8, 4, 1], qp, kind),
            position: bank::<1, 1>([[28], [28], [35]], [13], qp, kind)[0],
        }
    }
}
impl Parser<'_> {
    pub(super) fn sbt_mode(&mut self, r: &mut Reader<'_>, unit: &mut Unit) -> Result<()> {
        let node = unit.node;
        let max = 1 << self.sequence.max_transform_log2;
        if !self.sequence.tools.sbt
            || unit.inter.is_none()
            || matches!(unit.inter, Some(Inter::Ciip(_)))
            || node.width > max
            || node.height > max
            || node.width.max(node.height) < 8
        {
            return Ok(());
        }
        let models = &mut self.sbt_models;
        if !r.bin(&mut models.flag[usize::from(node.width * node.height <= 256)])? {
            return Ok(());
        }
        let quarter = node.width.max(node.height) >= 16 && r.bin(&mut models.quarter)?;
        let minimum = if quarter { 16 } else { 8 };
        let horizontal = if node.width >= minimum && node.height >= minimum {
            let context = match node.width.cmp(&node.height) {
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Less => 1,
                std::cmp::Ordering::Greater => 2,
            };
            r.bin(&mut models.horizontal[context])?
        } else {
            node.height >= minimum
        };
        let selection = Sbt {
            horizontal,
            quarter,
            last: r.bin(&mut models.position)?,
        };
        unit.sbt = Some(selection);
        unit.transforms = super::transform_units::sbt_units(node, selection);
        Ok(())
    }
}
