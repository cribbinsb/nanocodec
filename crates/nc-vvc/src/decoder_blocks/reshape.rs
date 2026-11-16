//! LMCS model selection from bounded adaptation sets; sample mapping lives in DSP.
use super::{DecodeError, Result, Sequence, SliceHeader};
use crate::decoder_headers::{Adaptation, AdaptationData};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::{Pixel, VvcLmcs};

/// The LMCS model of slice `header` at the sequence's depth (H.266
/// §7.4.3.19: the bins split `0 ..= (1 << BitDepth) − 1`).
pub(crate) fn model<T: Pixel>(
    sequence: &Sequence,
    header: &SliceHeader,
    banks: &[Option<Box<Adaptation>>; 8],
) -> Result<Option<VvcLmcs<T>>> {
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
    VvcLmcs::new_at(
        parameters.bins,
        parameters.width_delta,
        parameters.chroma_offset,
        sequence.depth.of::<T>(),
    )
    .map(Some)
    .ok_or(DecodeError::Malformed("VVC LMCS model bounds"))
}

/// VPDU scales are derived once from already reconstructed luma neighbours.
/// In a dual tree, the luma CU containing the VPDU origin defines that origin.
pub(super) struct ChromaScale<'a, T: Pixel> {
    model: &'a VvcLmcs<T>,
    side: usize,
    columns: usize,
    /// The origin of the luma CU at each VPDU's origin, for the VPDUs of
    /// the CUs given (one slice's).
    origins: Vec<Option<(usize, usize)>>,
    scales: Vec<Option<i32>>,
}
impl<'a, T: Pixel> ChromaScale<'a, T> {
    pub fn new(
        model: &'a VvcLmcs<T>,
        units: &[super::Unit],
        dimensions: [usize; 2],
        ctu_log2: u8,
    ) -> Self {
        let side = (1usize << ctu_log2).min(64);
        let columns = dimensions[0].div_ceil(side);
        let count = columns * dimensions[1].div_ceil(side);
        let mut origins = vec![None; count];
        for unit in units.iter().filter(|u| u.channel == 0) {
            let n = unit.node;
            for y in (n.y.div_ceil(side) * side..n.y + n.height).step_by(side) {
                for x in (n.x.div_ceil(side) * side..n.x + n.width).step_by(side) {
                    origins[y / side * columns + x / side] = Some((n.x, n.y));
                }
            }
        }
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
        picture: &Picture<T>,
        available: &super::references::Availability,
    ) -> i32 {
        let index = position.1 / self.side * self.columns + position.0 / self.side;
        if let Some(scale) = self.scales[index] {
            return scale;
        }
        // Every chroma block lies in a VPDU whose origin a luma CU of its
        // slice covers; without one, no neighbour is available.
        let (x, y) = self.origins[index].unwrap_or((0, 0));
        let view = picture.plane(PlaneIndex::Y).view();
        let mut sum = 0u32;
        let mut count = 0u32;
        if available.has(0, x as isize - 1, y as isize) {
            for dy in 0..self.side {
                sum += view.row((y + dy).min(view.height() - 1))[x - 1].to_i32() as u32;
            }
            count += self.side as u32;
        }
        if available.has(0, x as isize, y as isize - 1) {
            for dx in 0..self.side {
                sum += view.row(y - 1)[(x + dx).min(view.width() - 1)].to_i32() as u32;
            }
            count += self.side as u32;
        }
        // Without neighbours, the mid-grey of the depth (§8.7.5.3).
        let average = (sum + count / 2)
            .checked_div(count)
            .map_or(self.model.depth().mid(), |mean| mean as i32);
        let scale = self.model.chroma_scale(T::from_i32(average));
        self.scales[index] = Some(scale);
        scale
    }
}
