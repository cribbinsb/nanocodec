//! LMCS model selection from bounded adaptation sets; sample mapping lives in DSP.
use super::{DecodeError, Result, Sequence, SliceHeader};
use crate::decoder_headers::{Adaptation, AdaptationData};
use nc_dsp::VvcLmcs;

pub(crate) fn model(
    _sequence: &Sequence,
    header: &SliceHeader,
    banks: &[Option<Box<Adaptation>>; 8],
) -> Result<Option<VvcLmcs>> {
    if !header.lmcs {
        return Ok(None);
    }
    let id = header
        .picture
        .lmcs
        .ok_or(DecodeError::Malformed("VVC missing LMCS selection"))?;
    let Some(AdaptationData::Lmcs(parameters)) = banks
        .get(usize::from(id))
        .and_then(Option::as_deref)
        .map(|a| &a.data)
    else {
        return Err(DecodeError::Malformed("VVC missing LMCS APS"));
    };
    VvcLmcs::new(
        parameters.bins,
        parameters.width_delta,
        parameters.chroma_offset,
    )
    .map(Some)
    .ok_or(DecodeError::Malformed("VVC LMCS model bounds"))
}

/// VPDU scales are derived once from already reconstructed luma neighbours.
/// In a dual tree, the luma CU containing the VPDU origin defines that origin.
pub(super) struct ChromaScale<'a> {
    model: &'a VvcLmcs,
    side: usize,
    columns: usize,
    origins: Vec<(usize, usize)>,
    scales: Vec<Option<i32>>,
}
impl<'a> ChromaScale<'a> {
    pub fn new(
        model: &'a VvcLmcs,
        units: &[super::Unit],
        dimensions: [usize; 2],
        ctu_log2: u8,
    ) -> Self {
        let side = (1usize << ctu_log2).min(64);
        let columns = dimensions[0].div_ceil(side);
        let count = columns * dimensions[1].div_ceil(side);
        let origins = (0..count)
            .map(|index| {
                let (x, y) = (index % columns * side, index / columns * side);
                let unit = units
                    .iter()
                    .find(|u| {
                        u.channel == 0
                            && u.node.x <= x
                            && x < u.node.x + u.node.width
                            && u.node.y <= y
                            && y < u.node.y + u.node.height
                    })
                    .unwrap();
                (unit.node.x, unit.node.y)
            })
            .collect();
        Self {
            model,
            side,
            columns,
            origins,
            scales: vec![None; count],
        }
    }
    pub fn get(
        &mut self,
        position: (usize, usize),
        picture: &nc_common::Picture,
        available: &super::references::Availability,
    ) -> i32 {
        let index = position.1 / self.side * self.columns + position.0 / self.side;
        if let Some(scale) = self.scales[index] {
            return scale;
        }
        let (x, y) = self.origins[index];
        let view = picture.plane(nc_common::PlaneIndex::Y).view();
        let mut sum = 0u32;
        let mut count = 0u32;
        if available.has(0, x as isize - 1, y as isize) {
            for dy in 0..self.side {
                sum += u32::from(view.row((y + dy).min(view.height() - 1))[x - 1]);
            }
            count += self.side as u32;
        }
        if available.has(0, x as isize, y as isize - 1) {
            for dx in 0..self.side {
                sum += u32::from(view.row(y - 1)[(x + dx).min(view.width() - 1)]);
            }
            count += self.side as u32;
        }
        let average = (sum + count / 2).checked_div(count).unwrap_or(128) as u8;
        let scale = self.model.chroma_scale(average);
        self.scales[index] = Some(scale);
        scale
    }
}
