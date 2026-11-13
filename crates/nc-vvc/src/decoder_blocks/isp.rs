//! General ISP prediction state, reusing shared geometry, edge continuation and DSP.
use super::{
    Unit,
    references::{Availability, References},
};
use crate::isp::{Split, edges_for_part};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::Dsp;
use nc_encoder::Block;

pub(super) struct Prediction {
    cu: Block,
    split: Split,
    references: References,
    cached: Option<(Block, Vec<u8>)>,
}
impl Prediction {
    pub fn prepare(unit: &Unit, picture: &Picture, available: &Availability) -> Option<Self> {
        let split = unit.isp?;
        let node = unit.node;
        let cu = Block::rect(
            node.x,
            node.y,
            node.width.ilog2() as u8,
            node.height.ilog2() as u8,
        );
        Some(Self {
            cu,
            split,
            references: References::gather(
                picture,
                available,
                PlaneIndex::Y,
                (node.x, node.y, node.width, node.height),
                false,
            ),
            cached: None,
        })
    }
    pub fn predict(
        &mut self,
        block: Block,
        direction: u8,
        picture: &Picture,
        available: &Availability,
        dsp: &Dsp,
    ) -> Vec<u8> {
        // H.266 prediction regions are at least four samples wide, even when
        // the residual is split into one/two-wide vertical transform blocks.
        let region = Block::rect(
            block.x & !3,
            block.y,
            block.log2_width.max(2),
            block.log2_height,
        );
        if self
            .cached
            .as_ref()
            .is_none_or(|(cached, _)| *cached != region)
        {
            let view = picture.plane(PlaneIndex::Y).view();
            let first = region.x == self.cu.x && region.y == self.cu.y;
            let previous = if first {
                Vec::new()
            } else {
                match self.split {
                    Split::Horizontal => {
                        view.row(region.y - 1)[region.x..region.x + region.width()].to_vec()
                    }
                    Split::Vertical => (region.y..region.y + region.height())
                        .map(|y| view.row(y)[region.x - 1])
                        .collect(),
                }
            };
            let other_available = match self.split {
                Split::Horizontal => available.has(0, self.cu.x as isize - 1, region.y as isize),
                Split::Vertical => available.has(0, region.x as isize, self.cu.y as isize - 1),
            };
            let refs = &self.references;
            let (top, left, corner) = edges_for_part(
                (&refs.top, &refs.left, refs.corner),
                &self.cu,
                &region,
                self.split,
                &previous,
                other_available,
            );
            let mut prediction = vec![0; region.area()];
            dsp.vvc_intra_predict_isp(
                (self.cu.width(), self.cu.height()),
                (region.width(), region.height()),
                direction,
                &top,
                &left,
                corner,
                &mut prediction,
            );
            self.cached = Some((region, prediction));
        }
        let (_, prediction) = self.cached.as_ref().unwrap();
        let offset = block.x - region.x;
        prediction
            .chunks_exact(region.width())
            .flat_map(|row| row[offset..offset + block.width()].iter().copied())
            .collect()
    }
}
