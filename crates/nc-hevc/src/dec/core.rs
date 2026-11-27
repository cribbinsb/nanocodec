//! The decoding process at one sample type: NAL units, POC (§8.3.1),
//! reference picture sets (§8.3.2, §8.3.3), reference picture lists
//! (§8.3.4) and the DPB (C.5.2), written once over `T: Pixel` (`u8` for
//! 8-bit streams, `u16` for Main 10; `docs/handoff/high-bit-depth.md`).

use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;

#[cfg(test)]
use nc_common::PlaneIndex;
use nc_common::decode::{DecodeError, unescape_rbsp};
#[cfg(feature = "analysis")]
use nc_common::stream_stats::StreamStats;
use nc_common::{BitDepths, ColourInfo, OutputPicture, Picture, PictureRecycler, ShownPicture};
use nc_dsp::{Dsp, Pixel};

use super::ctu::{self, Carry, Scratch, SliceContext};
use super::loop_filter::LoopFilter;
use super::mvpred::Collocated;
use super::params::{MAX_PPS, MAX_SPS, Pps, Sps};
use super::picture::{Layout, Motion, PictureState, Pool, RefPic, SliceInfo};
use super::slice::{self, SliceHeader, SliceType, nal};
#[cfg(feature = "analysis")]
use super::stats;
use crate::scan::Scans;

/// Bytes of a slice segment NAL unit that hold its first flags and PPS id:
/// the two-byte header, then at most 2 + 13 bits (a PPS id below 64 is an
/// Exp-Golomb code of at most 13 bits), with room for an emulation
/// prevention byte.
const HEAD_BYTES: usize = 8;

/// How a DPB picture is used for reference (§8.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marking {
    Unused,
    ShortTerm,
    LongTerm,
}

/// A conformance window: top-left luma sample and luma size.
type Window = ((usize, usize), (usize, usize));

/// A decoded picture in the DPB.
#[derive(Debug)]
struct Frame<T: Pixel> {
    id: u64,
    poc: i32,
    /// Shared with the output queue until the caller is done with it.
    samples: Arc<Picture<T>>,
    depths: BitDepths,
    motion: Motion,
    marking: Marking,
    needed_for_output: bool,
    /// Conformance window (left, right, top, bottom).
    crop: [usize; 4],
}

impl<T: Pixel> Frame<T> {
    fn reference(&self) -> bool {
        self.marking != Marking::Unused
    }

    fn info(&self) -> RefPic {
        RefPic {
            id: self.id,
            poc: self.poc,
            long_term: self.marking == Marking::LongTerm,
        }
    }

    /// The conformance window: top-left luma sample and size.
    fn window(&self) -> Result<Window, DecodeError> {
        let [left, right, top, bottom] = self.crop;
        let (w, h) = (self.samples.width(), self.samples.height());
        let size = (w.checked_sub(left + right), h.checked_sub(top + bottom));
        let (Some(width), Some(height)) = size else {
            return Err(DecodeError::Malformed("conformance window"));
        };
        Ok(((left, top), (width, height)))
    }

    /// The samples inside the conformance window.
    fn visible(&self) -> Result<ShownPicture<'_, T>, DecodeError> {
        let (origin, size) = self.window()?;
        ShownPicture::window(&self.samples, origin, size)
            .ok_or(DecodeError::Malformed("conformance window"))
    }

    /// The picture for output order, shared with the DPB (no copy).
    fn shared(&self) -> Result<OutputPicture<T>, DecodeError> {
        self.visible()?;
        let (origin, size) = self.window()?;
        Ok(OutputPicture {
            picture: Arc::clone(&self.samples),
            origin,
            size,
            depths: self.depths,
        })
    }
}

/// The reference picture set of the current picture, as DPB indices
/// (§8.3.2): `RefPicSetStCurrBefore`, `StCurrAfter`, `LtCurr`.
#[derive(Debug, Default)]
struct ReferenceSet {
    before: Vec<usize>,
    after: Vec<usize>,
    long_term: Vec<usize>,
}

/// The picture being decoded.
struct Current<T: Pixel> {
    sps: Rc<Sps>,
    pps: Rc<Pps>,
    state: PictureState<T>,
    carry: Carry,
    id: u64,
    poc: i32,
    output: bool,
    set: ReferenceSet,
    /// The header of the last independent slice segment, for dependent ones.
    header: Option<Rc<SliceHeader>>,
}

impl<T: Pixel> Current<T> {
    /// The active parameter sets.
    #[cfg(feature = "analysis")]
    fn sets(&self) -> (&Sps, &Pps) {
        (&self.sps, &self.pps)
    }
}

/// POC and random access state carried between pictures.
#[derive(Debug, Default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent random access flags"
)]
struct Sequence {
    /// `PicOrderCntVal` of the previous `TemporalId` 0 anchor (§8.3.1).
    prev_tid0_poc: i32,
    /// The next IRAP picture starts a new coded video sequence with
    /// `NoRaslOutputFlag` 1: the first picture, or the first after EOS.
    new_cvs: bool,
    /// `NoRaslOutputFlag` of the last IRAP picture: its RASL pictures are
    /// skipped.
    skip_rasl: bool,
    /// Pictures are skipped until the first IRAP picture.
    seen_irap: bool,
    /// A picture has been decoded (the first one never discards others).
    decoded_any: bool,
}

/// The decoding process at one sample type (`u8` for 8-bit streams, `u16`
/// beyond): everything of the decoder but the choice of that type
/// ([`super::Decoder`]).
pub(super) struct Core<T: Pixel> {
    sps: Vec<Option<Rc<Sps>>>,
    pps: Vec<Option<Rc<Pps>>>,
    dpb: Vec<Frame<T>>,
    current: Option<Current<T>>,
    /// Pictures bumped out of the DPB, in output order.
    output: VecDeque<OutputPicture<T>>,
    /// The picture [`super::Decoder::shown`] shows: in per-access-unit use
    /// ([`super::Decoder::decode_shown`]), the first picture the access unit
    /// output, shared with the DPB.
    shown: Option<OutputPicture<T>>,
    sequence: Sequence,
    next_id: u64,
    /// Buffers of pictures that left the DPB, for new ones.
    pool: Pool<T>,
    /// The CTB layout of the last SPS/PPS pair.
    layout: Option<(Rc<Sps>, Rc<Pps>, Rc<Layout>)>,
    /// Output pictures handed back by the caller ([`Core::recycle`]).
    outputs: PictureRecycler<T>,
    /// Working memory of the in-loop filters, kept between pictures.
    filter: LoopFilter<T>,
    /// Working memory of block decoding.
    scratch: Scratch<T>,
    colour: Option<ColourInfo>,
    /// Whether the last activated SPS codes luma only (4:0:0).
    monochrome: bool,
    dsp: Dsp,
    scans: Scans,
}

impl<T: Pixel> std::fmt::Debug for Core<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Core")
            .field("dpb", &self.dpb.len())
            .field("output", &self.output.len())
            .finish_non_exhaustive()
    }
}

impl<T: Pixel> Core<T> {
    /// An empty decoder; the stream must start with parameter sets and an
    /// IRAP picture (pictures before the first IRAP are skipped).
    pub fn new() -> Self {
        Self {
            sps: vec![None; MAX_SPS],
            pps: vec![None; MAX_PPS],
            dpb: Vec::new(),
            current: None,
            output: VecDeque::new(),
            shown: None,
            sequence: Sequence {
                new_cvs: true,
                ..Sequence::default()
            },
            next_id: 0,
            pool: Pool::default(),
            layout: None,
            outputs: PictureRecycler::default(),
            filter: LoopFilter::default(),
            scratch: Scratch::default(),
            colour: None,
            monochrome: false,
            dsp: Dsp::new(),
            scans: Scans::new(),
        }
    }

    /// A decoder at this sample type continuing `previous`'s stream: its
    /// parameter sets, sequence state and analysis carry over; its DPB was
    /// flushed ([`super::Decoder`] changes the sample type only at an IRAP
    /// picture whose SPS changes the bit depth, which starts a new coded
    /// video sequence).
    pub fn succeed<U: Pixel>(mut previous: Core<U>) -> Self {
        let mut next = Self::new();
        next.sps = std::mem::take(&mut previous.sps);
        next.pps = std::mem::take(&mut previous.pps);
        next.sequence = std::mem::take(&mut previous.sequence);
        next.next_id = previous.next_id;
        next.colour = previous.colour;
        next.monochrome = previous.monochrome;
        #[cfg(feature = "analysis")]
        {
            next.scratch.stats = previous.scratch.stats.take();
        }
        next
    }

    /// Whether the slice segment NAL unit `nal` starts a picture whose SPS
    /// needs the other sample type (`u8` holds 8-bit luma and chroma
    /// only); `false` for anything else or anything unreadable (decoding
    /// reports that).
    pub fn needs_other_sample_type(&self, nal: &[u8]) -> bool {
        if nal.len() < 3 || nal[0] & 0x80 != 0 {
            return false;
        }
        let nal_type = (nal[0] >> 1) & 63;
        let layer = ((nal[0] & 1) << 5) | (nal[1] >> 3);
        let irap = (nal::BLA_W_LP..=nal::RSV_IRAP_23).contains(&nal_type);
        // Only an IRAP picture activates a new SPS (§7.4.3.2.1); its first
        // slice segment header begins with the flag and the PPS id.
        if layer != 0 || !irap {
            return false;
        }
        // The flag, `no_output_of_prior_pics_flag` and a PPS id below 64
        // fit the first bytes; unescape only those.
        let head = &nal[2..nal.len().min(HEAD_BYTES)];
        let Ok(rbsp) = unescape_rbsp(head) else {
            return false;
        };
        let Ok((true, pps_id)) = slice::first_and_pps(&rbsp, nal_type) else {
            return false;
        };
        let sps = self.pps[pps_id]
            .as_ref()
            .and_then(|pps| self.sps[pps.sps_id].as_ref());
        sps.is_some_and(|sps| sps.depths.is_eight() != (T::MAX_BIT_DEPTH == 8))
    }

    /// Start collecting coding-tool analysis into `stats` (block records,
    /// and per-picture tallies unless [`StreamStats::records_only`]) from
    /// the next picture on. Decoding is unchanged.
    #[cfg(feature = "analysis")]
    pub fn collect_stats(&mut self, stats: StreamStats) {
        self.scratch.stats = Some(stats);
    }

    /// The analysis collected so far, if enabled: the block records of
    /// the last completed picture and the tallies of every completed one.
    #[cfg(feature = "analysis")]
    pub fn stats(&self) -> Option<&StreamStats> {
        self.scratch.stats.as_ref()
    }

    /// Complete the analysis of the last picture and hand it over (after
    /// [`Core::flush`]); collection stops.
    #[cfg(feature = "analysis")]
    pub fn take_stats(&mut self) -> Option<StreamStats> {
        let mut stats = self.scratch.stats.take()?;
        stats.finish();
        Some(stats)
    }

    /// Colour signalling of the last activated SPS (`None` before the first).
    pub fn colour(&self) -> Option<ColourInfo> {
        self.colour
    }

    /// Whether the last activated SPS codes luma only (4:0:0): its
    /// pictures' chroma planes are mid-grey, and the reference decoder
    /// writes luma alone.
    pub fn is_monochrome(&self) -> bool {
        self.monochrome
    }

    /// Bit depths of the last decoded picture (`None` before the first).
    pub fn depths(&self) -> Option<BitDepths> {
        self.dpb.last().map(|f| f.depths)
    }

    /// Test hook: overwrite every DPB picture with `value`, as a decoder
    /// that lost a frame would hold wrong references.
    #[cfg(test)]
    pub fn corrupt_references(&mut self, value: u8) {
        // The shown and queued pictures share the references' buffers.
        self.shown = None;
        self.output.clear();
        for frame in &mut self.dpb {
            let samples = Arc::get_mut(&mut frame.samples).expect("no picture shown or queued");
            for plane in PlaneIndex::ALL {
                samples
                    .plane_mut(plane)
                    .view_mut()
                    .fill(T::from_i32(i32::from(value)));
            }
            samples.extend_borders();
        }
    }

    /// Start an access unit of per-access-unit use
    /// ([`super::Decoder::decode_shown`]): nothing is shown until it ends.
    pub fn begin_shown(&mut self) {
        self.shown = None;
    }

    /// Finish the access unit begun with [`Self::begin_shown`] and fed with
    /// [`Self::push_nal`], and show the first picture it output (C.5.2:
    /// pictures with `PicOutputFlag` 0 or discarded by
    /// `NoOutputOfPriorPicsFlag` are never output). A stream without
    /// reordering outputs each picture as it is decoded; an access unit that
    /// outputs none (a skipped RASL picture, a reserved NAL type, a picture
    /// held for reordering) shows nothing, and one that outputs several
    /// leaves the rest for [`Self::next_shown`].
    ///
    /// # Errors
    /// As [`Self::push_nal`]; [`DecodeError::MissingKey`] when nothing has
    /// been decoded because no IRAP picture has arrived yet.
    pub fn end_shown(&mut self) -> Result<(), DecodeError> {
        self.finish_picture()?;
        if !self.sequence.seen_irap {
            // Pictures before the first IRAP are not decodable (§8.1.3):
            // the key frame that starts the stream was lost.
            return Err(DecodeError::MissingKey);
        }
        self.next_shown();
        Ok(())
    }

    /// Show the next picture output but not shown yet; `false` (nothing
    /// shown) when there is none.
    pub fn next_shown(&mut self) -> bool {
        self.shown = self.output.pop_front();
        self.shown.is_some()
    }

    /// The shown picture ([`Self::end_shown`]) copied into a buffer of its
    /// own.
    ///
    /// # Errors
    /// If no picture is shown.
    pub fn shown_copy(&mut self) -> Result<Picture<T>, DecodeError> {
        let shown = self
            .shown
            .as_ref()
            .ok_or(DecodeError::Malformed("packet has no picture"))?
            .view();
        let mut out = self.outputs.take(shown.width(), shown.height(), 0)?;
        shown.copy_to(&mut out);
        Ok(out)
    }

    /// The visible samples of the shown picture ([`Self::end_shown`]).
    pub fn shown(&self) -> Option<ShownPicture<'_, T>> {
        self.shown.as_ref().map(OutputPicture::view)
    }

    /// Signal the end of the stream: the last picture is finished and every
    /// picture still waiting is queued for output.
    /// # Errors
    /// As [`Self::push_nal`] for the last picture.
    pub fn flush(&mut self) -> Result<(), DecodeError> {
        self.finish_picture()?;
        self.bump_all();
        Ok(())
    }

    /// Hand back a picture this decoder returned, once the caller is done
    /// with it: later output pictures of its size reuse the buffer instead
    /// of allocating and clearing one.
    pub fn recycle(&mut self, picture: Picture<T>) {
        self.outputs.give(picture);
    }

    /// The next picture in output order, if one is ready.
    pub fn next_output(&mut self) -> Option<OutputPicture<T>> {
        self.output.pop_front()
    }

    /// Decode one NAL unit (without start code).
    /// # Errors
    /// Rejects malformed or truncated data and syntax outside the
    /// supported profiles; the decoder should then be reset before further
    /// use.
    pub fn push_nal(&mut self, nal: &[u8]) -> Result<(), DecodeError> {
        if nal.len() < 2 {
            return Err(DecodeError::Truncated);
        }
        if nal[0] & 0x80 != 0 {
            return Err(DecodeError::Malformed("forbidden_zero_bit"));
        }
        let nal_type = (nal[0] >> 1) & 63;
        let layer = ((nal[0] & 1) << 5) | (nal[1] >> 3);
        let temporal_id_plus1 = nal[1] & 7;
        if temporal_id_plus1 == 0 {
            return Err(DecodeError::Malformed("nuh_temporal_id_plus1"));
        }
        if layer != 0 {
            // Other layers are not part of the Main profile base layer.
            return Ok(());
        }
        let rbsp = unescape_rbsp(&nal[2..])?;
        match nal_type {
            0..=nal::RSV_IRAP_23 => {
                if nal_type <= nal::RSV_VCL_N14 && nal_type > nal::RASL_R || nal_type > nal::CRA {
                    // Reserved VCL types: ignored (§7.4.2.2).
                    return Ok(());
                }
                let result = self.slice_segment(&rbsp, nal_type, temporal_id_plus1 - 1);
                if result.is_err() {
                    // A picture with a bad slice is never finished: it
                    // would otherwise enter the DPB with its POC, and the
                    // pictures predicting from it would decode from garbage.
                    self.current = None;
                }
                result
            }
            nal::SPS => {
                self.finish_picture()?;
                let (id, sps) = Sps::parse(&rbsp)?;
                self.sps[id] = Some(Rc::new(sps));
                Ok(())
            }
            nal::PPS => {
                self.finish_picture()?;
                let (id, pps) = Pps::parse(&rbsp)?;
                self.pps[id] = Some(Rc::new(pps));
                Ok(())
            }
            nal::EOS | nal::EOB => {
                // The next picture starts a new coded video sequence; what
                // happens to pictures still waiting is decided by its
                // NoOutputOfPriorPicsFlag (C.5.2.2).
                self.finish_picture()?;
                self.sequence.new_cvs = true;
                Ok(())
            }
            // VPS, AUD, SEI, filler and reserved / unspecified units.
            _ => Ok(()),
        }
    }

    /// Decode one slice segment NAL unit.
    #[allow(
        clippy::too_many_lines,
        reason = "header, picture start and the slice's whole decoding context in order"
    )]
    fn slice_segment(
        &mut self,
        rbsp: &[u8],
        nal_type: u8,
        temporal_id: u8,
    ) -> Result<(), DecodeError> {
        let (first, pps_id) = slice::first_and_pps(rbsp, nal_type)?;
        if first {
            self.finish_picture()?;
        }
        let irap = (nal::BLA_W_LP..=nal::RSV_IRAP_23).contains(&nal_type);
        if first {
            if irap {
                self.sequence.seen_irap = true;
                self.sequence.skip_rasl = self.sequence.new_cvs || nal_type != nal::CRA;
            }
            let rasl = matches!(nal_type, nal::RASL_N | nal::RASL_R);
            if !self.sequence.seen_irap || (rasl && self.sequence.skip_rasl) {
                // Not decodable: before the first IRAP, or a RASL picture
                // of an IRAP that starts a sequence (§8.1.3, C.5.2.2).
                return Ok(());
            }
        } else if self.current.is_none() {
            // A later segment of a skipped picture.
            return Ok(());
        }
        let pps = self.pps[pps_id]
            .clone()
            .ok_or(DecodeError::Malformed("missing PPS"))?;
        let sps = self.sps[pps.sps_id]
            .clone()
            .ok_or(DecodeError::Malformed("missing SPS"))?;
        let segment = slice::parse(rbsp, nal_type, &sps, &pps)?;
        if first {
            let header = segment
                .header
                .as_ref()
                .ok_or(DecodeError::Malformed("dependent first slice segment"))?;
            self.start_picture(sps, pps, header, nal_type, temporal_id)?;
        }
        let current = self
            .current
            .as_mut()
            .ok_or(DecodeError::Malformed("slice outside a picture"))?;
        let header = match segment.header.clone() {
            Some(h) => {
                let h = Rc::new(h);
                current.header = Some(h.clone());
                let lists = reference_lists(&current.set, &h)?;
                current.state.slices.push(SliceInfo {
                    address: segment.address,
                    deblock_disabled: h.deblock_disabled,
                    beta_offset_div2: h.beta_offset_div2,
                    tc_offset_div2: h.tc_offset_div2,
                    loop_filter_across_slices: h.loop_filter_across_slices,
                    lists: lists.map(|l| l.iter().map(|&i| self.dpb[i].info()).collect()),
                });
                h
            }
            None => current.header.clone().ok_or(DecodeError::Malformed(
                "dependent slice segment without a slice",
            ))?,
        };
        #[cfg(feature = "analysis")]
        if let Some(s) = &mut self.scratch.stats {
            stats::segment(s, first.then_some(current.sets()), &segment, rbsp.len());
        }
        let slice_index = current.state.slices.len();
        let lists = reference_lists(&current.set, &header)?;
        let refs = lists
            .clone()
            .map(|l| l.iter().map(|&i| &*self.dpb[i].samples).collect());
        let col = (header.temporal_mvp && header.slice_type != SliceType::I)
            .then(|| {
                let list = usize::from(!header.collocated_from_l0);
                lists[list]
                    .get(header.collocated_ref_idx)
                    .map(|&i| Collocated {
                        motion: &self.dpb[i].motion,
                        poc: self.dpb[i].poc,
                    })
            })
            .flatten();
        let no_backward_pred = lists
            .iter()
            .flatten()
            .all(|&i| self.dpb[i].poc <= current.poc);
        let cx = SliceContext {
            sps: &current.sps,
            pps: &current.pps,
            header: &header,
            slice: slice_index as u16,
            refs,
            col,
            poc: current.poc,
            no_backward_pred,
            dsp: &self.dsp,
            scans: &self.scans,
            depths: current.sps.depths.of::<T>(),
            residual: ctu::ResidualPath::of(&current.sps, &current.pps),
            transform_skip_log2: if current.pps.transform_skip {
                current.pps.range.log2_max_transform_skip
            } else {
                0
            },
            wide: ctu::WideRange::of(&current.sps),
            chroma: ctu::ChromaLayout::of(&current.sps),
        };
        ctu::decode_segment(
            &cx,
            &mut current.state,
            &mut current.carry,
            &mut self.scratch,
            &mut self.filter,
            rbsp,
            &segment,
        )
    }

    /// Begin a picture: POC, reference picture set, DPB output and removal
    /// (§8.1.3, C.5.2.2).
    fn start_picture(
        &mut self,
        sps: Rc<Sps>,
        pps: Rc<Pps>,
        header: &SliceHeader,
        nal_type: u8,
        temporal_id: u8,
    ) -> Result<(), DecodeError> {
        let irap = (nal::BLA_W_LP..=nal::RSV_IRAP_23).contains(&nal_type);
        let no_rasl_output = irap && self.sequence.skip_rasl;
        let poc = self.picture_order_count(&sps, header, nal_type, no_rasl_output);
        // §8.3.1: the anchor for the next POC is the previous TemporalId 0
        // picture that is not RASL, RADL or a sub-layer non-reference one.
        let sub_layer_non_reference = nal_type <= nal::RSV_VCL_N14 && nal_type.is_multiple_of(2);
        let leading = (nal::RADL_N..=nal::RASL_R).contains(&nal_type);
        if temporal_id == 0 && !leading && !sub_layer_non_reference {
            self.sequence.prev_tid0_poc = poc;
        }
        if irap && no_rasl_output {
            // NoOutputOfPriorPicsFlag: always 1 for CRA (C.5.2.2).
            let discard = self.sequence.decoded_any
                && (nal_type == nal::CRA || header.no_output_of_prior_pics);
            if discard {
                self.release_where(|_| true);
            } else {
                self.bump_all();
            }
            for f in &mut self.dpb {
                f.marking = Marking::Unused;
            }
        }
        self.sequence.new_cvs = false;
        self.sequence.decoded_any = true;
        let ids = if matches!(nal_type, nal::IDR_W_RADL | nal::IDR_N_LP) {
            for f in &mut self.dpb {
                f.marking = Marking::Unused;
            }
            IdSet::default()
        } else {
            self.reference_set(&sps, header, poc)?
        };
        // Remove pictures neither waiting for output nor referenced, then
        // bump while the DPB is over its reorder or size limits (C.5.2.2).
        // The latency limit is not applied: it only makes output earlier,
        // never changes its order.
        self.release_where(|f| !f.needed_for_output && !f.reference());
        let (buffering, reorder, _) = sps.dpb[sps.max_sub_layers - 1];
        while self.waiting() > reorder || self.dpb.len() >= buffering {
            if !self.bump() {
                break;
            }
        }
        let set = ids.indices(&self.dpb)?;
        self.colour = Some(sps.colour);
        self.monochrome = sps.chroma.is_none();
        let layout = match &self.layout {
            Some((s, p, layout)) if Rc::ptr_eq(s, &sps) && Rc::ptr_eq(p, &pps) => layout.clone(),
            _ => {
                let layout = Rc::new(Layout::new(&sps, &pps)?);
                self.layout = Some((sps.clone(), pps.clone(), layout.clone()));
                layout
            }
        };
        let state = PictureState::new(&sps, layout, &mut self.pool)?;
        let id = self.next_id;
        self.next_id += 1;
        self.current = Some(Current {
            sps,
            pps,
            state,
            carry: Carry::default(),
            id,
            poc,
            output: header.pic_output,
            set,
            header: None,
        });
        Ok(())
    }

    /// `PicOrderCntVal` (§8.3.1).
    fn picture_order_count(
        &self,
        sps: &Sps,
        header: &SliceHeader,
        nal_type: u8,
        no_rasl_output: bool,
    ) -> i32 {
        let max_lsb = 1i32 << sps.log2_max_poc_lsb;
        let lsb = header.poc_lsb as i32;
        let msb = if no_rasl_output && (nal::BLA_W_LP..=nal::RSV_IRAP_23).contains(&nal_type) {
            0
        } else {
            let prev = self.sequence.prev_tid0_poc;
            let prev_lsb = prev & (max_lsb - 1);
            let prev_msb = prev - prev_lsb;
            if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                prev_msb + max_lsb
            } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                prev_msb - max_lsb
            } else {
                prev_msb
            }
        };
        msb + lsb
    }

    /// Mark the DPB from the slice's reference picture set (§8.3.2).
    /// Returns the sets by picture id.
    ///
    /// # Errors
    ///
    /// [`DecodeError::MissingReference`] when a reference the picture uses
    /// is not in the DPB: "no reference picture", an unintentional picture
    /// loss (§8.3.2 NOTE). The marking still applies, so the DPB matches
    /// the encoder's for the pictures that follow; stand-ins (§8.3.3) are
    /// only for the skipped leading pictures of a CRA or BLA, which this
    /// decoder never decodes.
    fn reference_set(
        &mut self,
        sps: &Sps,
        header: &SliceHeader,
        poc: i32,
    ) -> Result<IdSet, DecodeError> {
        let max_lsb = 1i32 << sps.log2_max_poc_lsb;
        let mut set = IdSet::default();
        let mut keep: Vec<u64> = Vec::new();
        let mut missing = false;
        // Long-term entries first: any reference picture may become one.
        let mut long_term = Vec::new();
        for lt in &header.long_term {
            let full = if lt.msb_present {
                Some(poc - (poc & (max_lsb - 1)) + lt.poc)
            } else {
                None
            };
            let found = self.dpb.iter().position(|f| {
                f.reference()
                    && match full {
                        Some(p) => f.poc == p,
                        None => f.poc & (max_lsb - 1) == lt.poc,
                    }
            });
            let Some(index) = found else {
                missing |= lt.used;
                continue;
            };
            long_term.push(self.dpb[index].id);
            keep.push(self.dpb[index].id);
            if lt.used {
                set.long_term.push(self.dpb[index].id);
            }
        }
        for f in &mut self.dpb {
            if long_term.contains(&f.id) {
                f.marking = Marking::LongTerm;
            }
        }
        for (entries, target) in [
            (&header.short_term.negative, &mut set.before),
            (&header.short_term.positive, &mut set.after),
        ] {
            for &(delta, used) in entries {
                let p = poc + delta;
                let found = self
                    .dpb
                    .iter()
                    .position(|f| f.marking == Marking::ShortTerm && f.poc == p);
                let Some(index) = found else {
                    missing |= used;
                    continue;
                };
                keep.push(self.dpb[index].id);
                if used {
                    target.push(self.dpb[index].id);
                }
            }
        }
        for f in &mut self.dpb {
            if !keep.contains(&f.id) {
                f.marking = Marking::Unused;
            }
        }
        if missing {
            return Err(DecodeError::MissingReference);
        }
        Ok(set)
    }

    /// Pictures waiting for output.
    fn waiting(&self) -> usize {
        self.dpb.iter().filter(|f| f.needed_for_output).count()
    }

    /// The bumping process (C.5.2.4): output the waiting picture with the
    /// smallest POC and drop it if no longer referenced. Returns whether a
    /// picture was output.
    fn bump(&mut self) -> bool {
        let Some(index) = self
            .dpb
            .iter()
            .enumerate()
            .filter(|(_, f)| f.needed_for_output)
            .min_by_key(|(_, f)| f.poc)
            .map(|(i, _)| i)
        else {
            return false;
        };
        if let Ok(picture) = self.dpb[index].shared() {
            self.output.push_back(picture);
        }
        self.dpb[index].needed_for_output = false;
        if !self.dpb[index].reference() {
            let frame = self.dpb.remove(index);
            self.pool.recycle(frame.samples, frame.motion);
        }
        true
    }

    fn bump_all(&mut self) {
        while self.bump() {}
        self.release_where(|f| !f.reference());
    }

    /// Remove the DPB pictures matching `remove`, keeping their buffers.
    fn release_where(&mut self, remove: impl Fn(&Frame<T>) -> bool) {
        let mut i = 0;
        while i < self.dpb.len() {
            if remove(&self.dpb[i]) {
                let frame = self.dpb.swap_remove(i);
                self.pool.recycle(frame.samples, frame.motion);
            } else {
                i += 1;
            }
        }
    }

    /// Finish the current picture: in-loop filters, DPB storage and output
    /// (C.5.2.3).
    fn finish_picture(&mut self) -> Result<(), DecodeError> {
        let Some(current) = self.current.take() else {
            return Ok(());
        };
        #[cfg(feature = "analysis")]
        if let Some(s) = &mut self.scratch.stats {
            // Its records become the completed picture's.
            s.finish();
        }
        let Current {
            sps,
            pps,
            mut state,
            id,
            poc,
            output,
            ..
        } = current;
        if !state.progress.complete() {
            return Err(DecodeError::Malformed("picture not completely decoded"));
        }
        self.filter.advance(&self.dsp, &pps, &mut state, true);
        let (samples, motion) = state.into_motion(&mut self.pool);
        let frame = Frame {
            id,
            poc,
            samples: Arc::new(samples),
            depths: sps.depths,
            motion,
            marking: Marking::ShortTerm,
            needed_for_output: output,
            crop: sps.crop,
        };
        self.dpb.push(frame);
        let (_, reorder, _) = sps.dpb[sps.max_sub_layers - 1];
        while self.waiting() > reorder {
            self.bump();
        }
        Ok(())
    }
}

/// A reference picture set by picture id (ids survive DPB removals).
#[derive(Debug, Default)]
struct IdSet {
    before: Vec<u64>,
    after: Vec<u64>,
    long_term: Vec<u64>,
}

impl IdSet {
    /// The set as DPB indices.
    fn indices<T: Pixel>(&self, dpb: &[Frame<T>]) -> Result<ReferenceSet, DecodeError> {
        let index = |ids: &[u64]| -> Result<Vec<usize>, DecodeError> {
            ids.iter()
                .map(|id| {
                    dpb.iter()
                        .position(|f| f.id == *id)
                        .ok_or(DecodeError::Malformed("reference picture removed"))
                })
                .collect()
        };
        Ok(ReferenceSet {
            before: index(&self.before)?,
            after: index(&self.after)?,
            long_term: index(&self.long_term)?,
        })
    }
}

/// `RefPicList0` / `RefPicList1` as DPB indices (§8.3.4).
fn reference_lists(
    set: &ReferenceSet,
    header: &SliceHeader,
) -> Result<[Vec<usize>; 2], DecodeError> {
    let mut lists = [Vec::new(), Vec::new()];
    if header.slice_type == SliceType::I {
        return Ok(lists);
    }
    let total = set.before.len() + set.after.len() + set.long_term.len();
    if total == 0 {
        return Err(DecodeError::Malformed(
            "inter slice without current references",
        ));
    }
    let count = if header.slice_type == SliceType::B {
        2
    } else {
        1
    };
    for (l, list) in lists.iter_mut().enumerate().take(count) {
        let order: [&[usize]; 3] = if l == 0 {
            [&set.before, &set.after, &set.long_term]
        } else {
            [&set.after, &set.before, &set.long_term]
        };
        let n = header.num_ref_idx[l];
        let mut temp = Vec::with_capacity(n.max(total));
        while temp.len() < n.max(total) {
            for part in order {
                for &i in part {
                    if temp.len() < n.max(total) {
                        temp.push(i);
                    }
                }
            }
        }
        *list = (0..n)
            .map(|i| match &header.list_entries[l] {
                Some(entries) => temp[entries[i]],
                None => temp[i],
            })
            .collect();
    }
    Ok(lists)
}
