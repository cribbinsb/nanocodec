//! Bounded reference retention and POC derivation for general VVC pictures
//! of samples `T` (8-bit `u8`, deeper `u16`). Each picture keeps its
//! geometry, from which a later picture's slices derive the scaling of
//! their references (reference picture resampling, H.266 §8.3.2).
use crate::{
    decoder_blocks::{Geometry, Scaling},
    decoder_headers::{ReferenceEntry, Sequence, SliceHeader},
};
use nc_common::{
    Picture, Sample,
    decode::{DecodeError, Result},
};

#[derive(Debug)]
pub(crate) struct References<T: Sample = u8> {
    /// The reference pictures: POC, samples and geometry.
    pictures: Vec<(i32, Picture<T>, Geometry)>,
    current: Option<i32>,
    previous_tid0: Option<i32>,
    pending_output: Vec<(i32, Picture<T>)>,
    ready_output: std::collections::VecDeque<Picture<T>>,
    output_enabled: bool,
    reorder: usize,
    motion: Vec<crate::decoder_blocks::MotionField>,
}
impl<T: Sample> Default for References<T> {
    fn default() -> Self {
        Self {
            pictures: Vec::new(),
            current: None,
            previous_tid0: None,
            pending_output: Vec::new(),
            ready_output: std::collections::VecDeque::new(),
            output_enabled: false,
            reorder: 0,
            motion: Vec::new(),
        }
    }
}
impl<T: Sample> References<T> {
    pub fn prepare(
        &mut self,
        previous: Option<(Picture<T>, Geometry)>,
        sequence: &Sequence,
        header: &SliceHeader,
        (kind, starts_sequence): (u8, bool),
        temporal_id: u8,
        filler: Option<(&Picture<T>, Geometry)>,
    ) -> Result<(i32, [Vec<i32>; 2])> {
        if let (Some(poc), Some((mut picture, geometry))) = (self.current, previous) {
            // Plane regions include alignment padding even with a zero border.
            // Shared interpolation requires those samples to replicate the edge.
            picture.extend_borders();
            self.pictures.push((poc, picture, geometry));
        }
        if matches!(kind, 7..=10) {
            if header.no_prior_output {
                self.pending_output.clear();
            } else {
                self.flush_output();
            }
        }
        // An IDR picture, or an IRAP picture that starts a sequence
        // (`NoOutputBeforeRecoveryFlag`), marks every reference unused
        // (H.266 §8.3.2) and restarts the POC (§8.3.1).
        if matches!(kind, 7 | 8) || starts_sequence {
            self.pictures.clear();
            self.motion.clear();
            self.current = None;
            self.previous_tid0 = None;
        }
        let lsb = header.picture.order.lsb as i32;
        let range = 1i32 << sequence.poc_bits;
        let poc = if let Some(cycle) = header.picture.order.msb_cycle {
            i32::try_from(i64::from(cycle) * i64::from(range) + i64::from(lsb))
                .map_err(|_| DecodeError::Unsupported("VVC POC range"))?
        } else if let Some(previous) = self.previous_tid0 {
            let previous_lsb = previous.rem_euclid(range);
            let mut msb = previous - previous_lsb;
            if lsb < previous_lsb && previous_lsb - lsb >= range / 2 {
                msb = msb
                    .checked_add(range)
                    .ok_or(DecodeError::Malformed("VVC POC overflow"))?;
            } else if lsb > previous_lsb && lsb - previous_lsb > range / 2 {
                msb = msb
                    .checked_sub(range)
                    .ok_or(DecodeError::Malformed("VVC POC overflow"))?;
            }
            msb.checked_add(lsb)
                .ok_or(DecodeError::Malformed("VVC POC overflow"))?
        } else {
            lsb
        };
        let retained = self.entries(poc, header, sequence.poc_bits)?;
        // The first slice's lists mark the other pictures unused (§8.3.3).
        self.pictures
            .retain(|(poc, ..)| retained.iter().any(|list| list.contains(poc)));
        // While recovering from a GDR picture that started the sequence,
        // listed pictures the DPB lacks are generated (§8.3.4).
        if let Some((filler, geometry)) = filler {
            for &listed in retained.iter().flatten() {
                if !self.pictures.iter().any(|(stored, ..)| *stored == listed) {
                    self.pictures.push((listed, filler.clone(), geometry));
                }
            }
        }
        // Inactive RPL entries may become collocated references in a later
        // picture, so retain their motion alongside their decoded samples.
        self.motion
            .retain(|field| retained.iter().any(|list| list.contains(&field.poc)));
        let retained = self.active(retained, header)?;
        // The selected 8-bit profile's DPB is bounded independently of RPL length.
        if self.pictures.len() > 16 {
            return Err(DecodeError::Unsupported("VVC DPB capacity"));
        }
        self.current = Some(poc);
        if temporal_id == 1 && !header.picture.non_reference && !matches!(kind, 2 | 3) {
            self.previous_tid0 = Some(poc);
        }
        self.output_enabled = header.picture.output;
        self.reorder = sequence.dpb.last().unwrap()[1] as usize;
        Ok((poc, retained))
    }
    /// The active reference POCs of a later slice `header` of picture `poc`
    /// (§8.3.2), each in the DPB.
    pub fn slice_lists(
        &self,
        poc: i32,
        header: &SliceHeader,
        poc_bits: u8,
    ) -> Result<[Vec<i32>; 2]> {
        self.active(self.entries(poc, header, poc_bits)?, header)
    }

    /// The POCs of every entry of the two reference picture lists of slice
    /// `header` of picture `poc` (H.266 §8.3.2): a short-term entry by its
    /// POC delta; a long-term one the picture whose POC is
    /// `PicOrderCntVal − DeltaPocMsbCycleLt · MaxPicOrderCntLsb −
    /// (PicOrderCntVal & (MaxPicOrderCntLsb − 1)) + PocLsbLt` with its MSB
    /// cycle, else the picture whose POC LSBs are `PocLsbLt`.
    fn entries(&self, poc: i32, header: &SliceHeader, poc_bits: u8) -> Result<[Vec<i32>; 2]> {
        let range = 1i64 << poc_bits;
        let mut lists: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
        for (list, selected) in header.references.iter().enumerate() {
            for (entry, msb_cycle) in selected.entries.iter().zip(&selected.msb_cycles) {
                let reference = match (entry, msb_cycle) {
                    (ReferenceEntry::Short(delta), _) => i64::from(poc) + i64::from(*delta),
                    (ReferenceEntry::Long(lsb), Some(cycle)) => {
                        let lsb = i64::from(lsb.ok_or(DecodeError::Malformed("VVC LTRP LSB"))?);
                        i64::from(poc)
                            - i64::from(*cycle) * range
                            - i64::from(poc).rem_euclid(range)
                            + lsb
                    }
                    (ReferenceEntry::Long(lsb), None) => {
                        let lsb = i64::from(lsb.ok_or(DecodeError::Malformed("VVC LTRP LSB"))?);
                        self.pictures
                            .iter()
                            .map(|(stored, ..)| i64::from(*stored))
                            .find(|stored| stored.rem_euclid(range) == lsb)
                            .unwrap_or(i64::from(poc) - i64::from(poc).rem_euclid(range) + lsb)
                    }
                };
                lists[list].push(
                    i32::try_from(reference)
                        .map_err(|_| DecodeError::Malformed("VVC reference POC overflow"))?,
                );
            }
        }
        Ok(lists)
    }

    /// The first `header.active` POCs of each of `lists`, each in the DPB.
    fn active(&self, mut lists: [Vec<i32>; 2], header: &SliceHeader) -> Result<[Vec<i32>; 2]> {
        for (list, &count) in lists.iter_mut().zip(&header.active) {
            if usize::from(count) > list.len() {
                return Err(DecodeError::Malformed("VVC active reference count"));
            }
            list.truncate(usize::from(count));
            if list
                .iter()
                .any(|poc| !self.pictures.iter().any(|(stored, ..)| stored == poc))
            {
                return Err(DecodeError::Malformed("VVC missing reference picture"));
            }
        }
        Ok(lists)
    }

    /// The current picture is not output (a picture recovering from a GDR
    /// picture that started the sequence, §8.1.1 `PictureOutputFlag`).
    pub fn withhold_output(&mut self) {
        self.output_enabled = false;
    }

    pub fn queue_output(&mut self, picture: Picture<T>) {
        if self.output_enabled {
            self.pending_output.push((self.current.unwrap(), picture));
        }
        while self.pending_output.len() > self.reorder {
            self.bump_output();
        }
    }
    fn bump_output(&mut self) {
        let index = self
            .pending_output
            .iter()
            .enumerate()
            .min_by_key(|(i, (poc, _))| (*poc, *i))
            .unwrap()
            .0;
        self.ready_output
            .push_back(self.pending_output.remove(index).1);
    }
    pub fn next_output(&mut self) -> Option<Picture<T>> {
        self.ready_output.pop_front()
    }
    pub fn flush_output(&mut self) {
        while !self.pending_output.is_empty() {
            self.bump_output();
        }
    }
    pub fn record_motion(&mut self, field: crate::decoder_blocks::MotionField) {
        self.motion.push(field);
    }
    pub fn temporal<'a>(
        &'a self,
        poc: i32,
        lists: [&'a [i32]; 2],
        header: &SliceHeader,
        ctu_log2: u8,
    ) -> Option<crate::decoder_blocks::Temporal<'a>> {
        if !header.picture.motion.temporal || header.kind == 2 {
            return None;
        }
        let [list, index] = header.collocated.map(usize::from);
        let selected = *lists[list].get(index)?;
        let field = self.motion.iter().find(|f| f.poc == selected)?;
        Some(crate::decoder_blocks::Temporal::new(
            field, poc, lists, header, ctu_log2,
        ))
    }
    pub fn pictures(&self, pocs: &[i32]) -> Vec<&Picture<T>> {
        pocs.iter().map(|&poc| &self.stored(poc).1).collect()
    }

    /// The scaling of each of the references `pocs` onto a picture of
    /// geometry `current`, `None` for an unscaled one (§8.3.2).
    pub fn scalings(
        &self,
        pocs: &[i32],
        current: Geometry,
        sequence: &Sequence,
    ) -> Result<Vec<Option<Scaling>>> {
        pocs.iter()
            .map(|&poc| {
                Scaling::between(
                    current,
                    self.stored(poc).2,
                    sequence.tools.chroma_collocated,
                )
            })
            .collect()
    }

    /// The stored reference picture `poc`.
    fn stored(&self, poc: i32) -> &(i32, Picture<T>, Geometry) {
        self.pictures
            .iter()
            .find(|(stored, ..)| *stored == poc)
            .expect("validated active reference")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_b_pictures_are_bumped_in_display_order_and_flushed() {
        let mut references = References::<u8> {
            output_enabled: true,
            reorder: 1,
            ..References::default()
        };
        for poc in [0, 2, 1] {
            let mut picture = Picture::new(2, 2, 0).unwrap();
            picture
                .plane_mut(nc_common::PlaneIndex::Y)
                .view_mut()
                .fill(poc as u8);
            references.current = Some(poc);
            references.queue_output(picture);
        }
        let value = |p: Picture| p.plane(nc_common::PlaneIndex::Y).view().row(0)[0];
        assert_eq!(references.next_output().map(value), Some(0));
        assert_eq!(references.next_output().map(value), Some(1));
        assert!(references.next_output().is_none());
        references.flush_output();
        assert_eq!(references.next_output().map(value), Some(2));
        assert!(references.next_output().is_none());
    }
}
