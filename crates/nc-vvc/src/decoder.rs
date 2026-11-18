//! Bounded decoder for our H.266 intra/inter subset; rejects unsupported tools.
//!
//! Canonical parameter sets retain the own-stream fast path (8-bit). General
//! Main 10 streams (8 to 10 bits) use bounded syntax parsing and the general
//! decoding process at the sequence's sample type (`general::Instance`);
//! unsupported reconstruction fails explicitly.
use crate::quant_group::{QpMap, QuantGroup};
use crate::{
    chroma_qp::ChromaQpTable, coding_tree, contexts::Contexts, decoder_cabac::Reader,
    decoder_syntax, headers, motion::MotionGrid, recon,
};
use nc_common::{
    DecodedPicture, Picture, PictureRecycler, PlaneIndex, ShownPicture,
    decode::{BitReader, DecodeError, Result, annex_b_units, unescape_rbsp},
};
use nc_dsp::Dsp;
use std::collections::VecDeque;

mod general;
use general::{Instance, with_general};

// Four luma samples cover the 8-tap interpolation footprint (two for chroma).
// Shared Plane border extension lets inter prediction borrow edge windows rather
// than allocate and gather the same clamped samples for every boundary CU.
const REFERENCE_BORDER: usize = 4;
const MAX_PENDING_BYTES: usize = 64 * 1024 * 1024;
/// ALF APS ids (`aps_adaptation_parameter_set_id` of ALF APSs: 0..=7).
const ALF_APS_IDS: usize = 8;
/// SPS ids (`sps_seq_parameter_set_id`: 0..=15) and PPS ids
/// (`pps_pic_parameter_set_id`: 0..=63).
const SPS_IDS: usize = 16;
const PPS_IDS: usize = 64;
/// Our SPS's fixed prefix (`headers::sps_head`): IDs, chroma format, CTU
/// size and PTL without GCI or sub-profiles; the CTU size follows the IDs
/// and chroma format (§7.3.2.4).
const SPS_PREFIX_BITS: usize = 48;
const SPS_BITS_BEFORE_CTU_SIZE: usize = 13;

/// `nal_unit_type` of end of sequence and end of bitstream NAL units (H.266
/// Table 5): the next IRAP picture starts a new coded video sequence.
const EOS_NUT: u8 = 21;
const EOB_NUT: u8 = 22;
/// `nal_unit_type` of video parameter sets.
const VPS_NUT: u8 = 14;

/// What [`Decoder::push_unit`] made of one NAL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pushed {
    /// Parameters or metadata: no picture yet.
    Nothing,
    /// A picture shown in decoding order, by [`Decoder::shown`].
    Shown,
    /// A general stream's picture, queued for display-order output.
    Queued,
}

/// The error of 8-bit-only calls on a deeper stream.
const EIGHT_BIT_ONLY: DecodeError = DecodeError::Unsupported(
    "decoding-order output of more than 8 bits (use push_nal and next_output)",
);

/// Decoder for this crate's VVC streams and general Main 10 streams. Our own
/// streams' pictures show in decoding order; general streams' come out in
/// display order from [`Decoder::next_output`], 8-bit ones as `u8` samples
/// and deeper ones as `u16` ([`DecodedPicture`]).
#[derive(Debug, Default)]
pub struct Decoder {
    geometry: Option<headers::Geometry>,
    /// Whether the PPS enables `cu_qp_delta` (`quant_group.rs`); `None`
    /// until one of our PPSs arrives after the SPS.
    pps_block_qp: Option<bool>,
    colour: nc_common::ColourInfo,
    /// The SPS's chroma QP map; `None` until an SPS arrives.
    chroma_qp: Option<ChromaQpTable>,
    /// The SPS's optional tools.
    tools: headers::Tools,
    /// The coded motion of the picture being decoded and, once it is
    /// decoded, of the latest picture (the next one's collocated motion);
    /// `None` until a picture of the SPS's geometry starts.
    motion: Option<MotionGrid>,
    /// Our own streams' latest decoded picture (coded size): the next P
    /// picture's reference, and what [`Self::shown`] shows.
    reference: Option<Picture>,
    reference_border_ready: bool,
    /// The other pictures the latest picture's RPL0 keeps in the DPB
    /// (§8.3.2: the second reference, `references.rs`), with their POCs.
    retained: Vec<(u8, Picture)>,
    /// Whether the last decode succeeded, so `reference` is its picture.
    shown: bool,
    /// Replaced references, reused as later reconstructions and output copies.
    pictures: PictureRecycler,
    next_poc: Option<u8>,
    dsp: Dsp,
    /// Parameter NALs awaiting the next slice in the streaming API.
    pending: Vec<u8>,
    adaptation: [[Option<Box<crate::decoder_headers::Adaptation>>; 8]; 3],
    /// Our own streams' luma ALF APS banks, by APS id.
    alf_banks: [Option<Box<crate::alf_aps::SignalledFilters>>; ALF_APS_IDS],
    picture_header: Option<Box<crate::decoder_headers::PictureHeader>>,
    /// Whether the latest SPS is a general one (not our own streams').
    general: bool,
    /// General SPSs by `sps_seq_parameter_set_id`.
    general_sequences: [Option<Box<crate::decoder_headers::Sequence>>; SPS_IDS],
    /// General PPS RBSPs by `pps_pic_parameter_set_id`, parsed when a
    /// picture activates them (their syntax depends on their SPS).
    pps_rbsp: Vec<Option<Vec<u8>>>,
    /// The active SPS and PPS (§7.4.2.4.2), of the current picture.
    general_sequence: Option<Box<crate::decoder_headers::Sequence>>,
    general_picture: Option<Box<crate::decoder_headers::PictureParameters>>,
    /// The general decoding process of the current sequence.
    instance: Instance,
    /// Field parity from the current access unit's `frame_field_info` SEI.
    pending_field: Option<crate::decoder_metadata::FieldParity>,
    /// General pictures output before the sequence changed sample type,
    /// ahead of the current instance's.
    earlier: VecDeque<DecodedPicture>,
    /// Coding-tool analysis, collected only when enabled.
    #[cfg(feature = "analysis")]
    stats: Option<nc_common::stream_stats::StreamStats>,
}
impl Decoder {
    /// An empty decoder, awaiting SPS and PPS.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Start collecting coding-tool analysis into `stats` (block records,
    /// and per-picture tallies unless `records_only`) from the next picture
    /// on. Decoding is unchanged.
    #[cfg(feature = "analysis")]
    pub fn collect_stats(&mut self, stats: nc_common::stream_stats::StreamStats) {
        self.stats = Some(stats);
    }

    /// The analysis collected so far, if enabled: the block records of the
    /// last decoded picture and the tallies of every completed one.
    #[cfg(feature = "analysis")]
    #[must_use]
    pub fn stats(&self) -> Option<&nc_common::stream_stats::StreamStats> {
        self.stats.as_ref()
    }

    /// Complete the analysis and hand it over; collection stops.
    #[cfg(feature = "analysis")]
    pub fn take_stats(&mut self) -> Option<nc_common::stream_stats::StreamStats> {
        let mut stats = self.stats.take()?;
        stats.finish();
        Some(stats)
    }

    /// Feed one NAL without its Annex-B start code. Metadata emits no picture.
    ///
    /// Our own streams' pictures are returned as they complete. General
    /// pictures are output once their last slice arrives, reordered using
    /// the SPS DPB limits, by [`Self::next_output`]: call it after each NAL
    /// and after `flush`. Unsupported coding tools fail explicitly.
    /// # Errors
    /// Rejects malformed metadata, oversized pending headers and unsupported NALs.
    pub fn push_nal(&mut self, nal: &[u8]) -> Result<Option<Picture>> {
        match self.push_unit(nal)? {
            Pushed::Shown => self.output().map(Some),
            Pushed::Nothing | Pushed::Queued => Ok(None),
        }
    }

    /// [`Self::push_nal`] without the output copy of pictures shown in
    /// decoding order (our own streams): `true` when this NAL completed such
    /// a picture, which [`Self::shown`] shows until the next call. General
    /// streams' pictures are reordered and still come from `next_output`.
    /// # Errors
    /// As [`Self::push_nal`].
    pub fn push_nal_shown(&mut self, nal: &[u8]) -> Result<bool> {
        Ok(self.push_unit(nal)? == Pushed::Shown)
    }

    fn push_unit(&mut self, nal: &[u8]) -> Result<Pushed> {
        if nal.len() < 2 {
            return Err(DecodeError::Truncated);
        }
        if nal[0] != 0 || nal[1].trailing_zeros() >= 3 {
            return Err(DecodeError::Unsupported("VVC layer/temporal id"));
        }
        let kind = nal[1] >> 3;
        if matches!(
            kind,
            super::decoder_metadata::OPI
                | super::decoder_metadata::DCI
                | super::decoder_metadata::AUD
                | super::decoder_metadata::PREFIX_SEI
                | super::decoder_metadata::SUFFIX_SEI
                | super::decoder_metadata::FILLER
        ) {
            if let Some(parity) =
                super::decoder_metadata::validate(kind, &unescape_rbsp(&nal[2..])?)?
            {
                self.pending_field = Some(parity);
            }
            return Ok(Pushed::Nothing);
        }
        if matches!(kind, EOS_NUT | EOB_NUT) {
            self.end_sequence();
            return Ok(Pushed::Nothing);
        }
        if !matches!(
            kind,
            14..=19 | 0..=3 | 7..=10
        ) {
            return Err(DecodeError::Unsupported("VVC NAL type"));
        }
        // Bound accumulation independently of any outer transport/framer.
        if nal.len()
            > MAX_PENDING_BYTES
                .saturating_sub(self.pending.len())
                .saturating_sub(4)
        {
            return Err(DecodeError::Unsupported("VVC pending access unit size"));
        }
        if matches!(kind, 14..=19) {
            self.pending.extend_from_slice(&[0, 0, 0, 1]);
            self.pending.extend_from_slice(nal);
            return Ok(Pushed::Nothing);
        }
        let mut parameters = std::mem::take(&mut self.pending);
        // The transport already framed the slice: borrow it directly instead
        // of copying and scanning its CABAC payload for Annex-B start codes.
        let result = if parameters.is_empty() {
            self.decode_units(std::iter::once(nal))
        } else {
            annex_b_units(&parameters)
                .and_then(|units| self.decode_units(units.into_iter().chain(std::iter::once(nal))))
        };
        parameters.clear();
        self.pending = parameters;
        if !result? {
            return Ok(Pushed::Nothing);
        }
        if !self.general {
            return Ok(Pushed::Shown);
        }
        with_general!(&mut self.instance, general => general.queue_output())?;
        Ok(Pushed::Queued)
    }

    /// Take another general picture released by the streaming decoder, in
    /// display order.
    pub fn next_output(&mut self) -> Option<DecodedPicture> {
        self.earlier
            .pop_front()
            .or_else(|| self.instance.next_output())
    }

    /// Complete a NAL stream and release delayed pictures through `next_output`.
    /// # Errors
    /// Rejects parameter sets left without a following picture.
    pub fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            with_general!(&mut self.instance, general => general.flush_output());
            Ok(())
        } else {
            Err(DecodeError::Malformed("VVC headers without a picture"))
        }
    }

    /// Bit depths of the active general sequence (`None` before one, and
    /// for our own streams, which are 8-bit).
    #[must_use]
    pub fn depths(&self) -> Option<nc_common::BitDepths> {
        self.general_sequence
            .as_ref()
            .map(|s| nc_common::BitDepths {
                luma: s.depth,
                chroma: s.depth,
            })
    }

    /// Colour description carried by the most recently accepted SPS.
    #[must_use]
    pub fn colour(&self) -> nc_common::ColourInfo {
        self.colour
    }

    /// Test hook: overwrite every reference picture with `value`, as a
    /// decoder that lost a picture would hold wrong references.
    #[cfg(test)]
    pub(crate) fn corrupt_references(&mut self, value: u8) {
        let retained = self.retained.iter_mut().map(|(_, picture)| picture);
        for picture in self.reference.iter_mut().chain(retained) {
            for plane in nc_common::PlaneIndex::ALL {
                picture.plane_mut(plane).view_mut().fill(value);
            }
        }
        self.reference_border_ready = false;
    }

    /// Decode one Annex-B access unit: one picture, with all its slices.
    /// # Errors
    /// Rejects truncated/malformed data and syntax outside the emitted subset.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Picture> {
        self.decode_shown(packet)?;
        self.output()
    }

    /// [`Self::decode`] without the output copy: the picture is shown by
    /// [`Self::shown`] until the next call.
    /// # Errors
    /// As [`Self::decode`].
    pub fn decode_shown(&mut self, packet: &[u8]) -> Result<()> {
        // A packet that cannot even be split shows nothing either.
        self.shown = false;
        if !self.decode_units(annex_b_units(packet)?)? {
            with_general!(&mut self.instance, general => general.abandon());
            return Err(DecodeError::Malformed("VVC access unit without all slices"));
        }
        match self.instance {
            Instance::High(_) => Err(EIGHT_BIT_ONLY),
            Instance::Eight(_) => Ok(()),
        }
    }

    /// Decode one access unit at either sample depth without an output copy.
    /// The picture is borrowed through [`Self::shown_high`] (deep) or [`Self::shown`] (8-bit).
    /// # Errors
    /// As [`Self::decode_shown`], except that supported deep streams are accepted.
    pub fn decode_shown_any_depth(&mut self, packet: &[u8]) -> Result<()> {
        // The original call completes a valid deep picture before reporting
        // its eight-bit return-type restriction. Reuse that exact eight-bit
        // call graph, and lift only this specific API restriction.
        match self.decode_shown(packet) {
            Err(EIGHT_BIT_ONLY) => Ok(()),
            result => result,
        }
    }

    /// Whether the active sequence stores its samples in u16.
    #[must_use]
    pub fn is_high_depth(&self) -> bool {
        matches!(self.instance, Instance::High(_))
    }

    /// Borrow deep output only; the original eight-bit shown path stays separate.
    #[must_use]
    pub fn shown_high(&self) -> Option<(ShownPicture<'_, u16>, nc_common::BitDepths)> {
        match &self.instance {
            Instance::High(core) if self.shown => Some((core.shown()?, core.depths())),
            _ => None,
        }
    }

    /// Whether the active general sequence codes only the luma plane.
    #[must_use]
    pub fn is_monochrome(&self) -> bool {
        self.general_sequence.as_ref().is_some_and(|s| !s.chroma)
    }

    /// The visible samples (the SPS conformance window) of the picture the
    /// last successful [`Self::decode_shown`] decoded.
    #[must_use]
    pub fn shown(&self) -> Option<ShownPicture<'_>> {
        if !self.shown {
            return None;
        }
        if self.general {
            return match &self.instance {
                Instance::Eight(general) => general.shown(),
                Instance::High(_) => None,
            };
        }
        let picture = self.reference.as_ref()?;
        let geometry = self.geometry?;
        ShownPicture::window(picture, (0, 0), (geometry.width, geometry.height))
    }

    /// A copy of the shown picture, in a recycled buffer when one fits.
    fn output(&mut self) -> Result<Picture> {
        let (width, height) = self
            .shown()
            .map(|shown| (shown.width(), shown.height()))
            .ok_or(DecodeError::Malformed("missing VVC picture"))?;
        let mut output = self.pictures.take(width, height, 0)?;
        if let Some(shown) = self.shown() {
            shown.copy_to(&mut output);
        }
        Ok(output)
    }

    fn accept_sps(&mut self, rbsp: &[u8]) -> Result<()> {
        self.pps_block_qp = None;
        self.general_picture = None;
        self.picture_header = None;
        if let Ok(info) = parse_sps(rbsp) {
            let changed_profile = self.general;
            if changed_profile {
                self.general = false;
                self.general_sequence = None;
                self.general_sequences = Default::default();
                self.pps_rbsp.clear();
                self.instance = Instance::default();
            }
            self.colour = info.colour;
            self.chroma_qp = Some(info.chroma_qp);
            if self.tools != info.tools {
                self.motion = None;
            }
            self.tools = info.tools;
            if changed_profile || self.geometry != Some(info.geometry) {
                self.motion = None;
                self.reference = None;
                self.next_poc = None;
            }
            self.geometry = Some(info.geometry);
        } else {
            let sequence = crate::decoder_headers::Sequence::parse(rbsp)?;
            self.motion = None;
            self.general = true;
            let id = usize::from(sequence.id);
            self.general_sequences[id] = Some(Box::new(sequence));
        }
        Ok(())
    }

    fn accept_pps(&mut self, rbsp: &[u8]) -> Result<()> {
        if self.general {
            let id = usize::from(rbsp.first().ok_or(DecodeError::Truncated)? >> 2);
            if self.pps_rbsp.len() < PPS_IDS {
                self.pps_rbsp.resize(PPS_IDS, None);
            }
            self.pps_rbsp[id] = Some(rbsp.to_vec());
            return Ok(());
        }
        let geometry = self
            .geometry
            .ok_or(DecodeError::Malformed("PPS before SPS"))?;
        let block_qp = match rbsp {
            _ if rbsp == headers::pps(&geometry, false) => false,
            _ if rbsp == headers::pps(&geometry, true) => true,
            _ => return Err(DecodeError::Unsupported("VVC PPS tools")),
        };
        self.pps_block_qp = Some(block_qp);
        Ok(())
    }

    /// Store one of our own streams' luma ALF APS banks (`alf_aps.rs`).
    fn accept_alf_aps(&mut self, rbsp: &[u8]) -> Result<()> {
        let aps = crate::decoder_headers::Adaptation::parse(rbsp)?;
        let bank = crate::alf_aps::SignalledFilters::from_adaptation(&aps)
            .ok_or(DecodeError::Unsupported("VVC APS outside the subset"))?;
        *self
            .alf_banks
            .get_mut(usize::from(aps.id))
            .ok_or(DecodeError::Malformed("VVC ALF APS id"))? = Some(Box::new(bank));
        Ok(())
    }

    /// Decode one of our own streams' pictures (`predicted`: a P picture)
    /// on the canonical path; returns it and its POC.
    #[allow(
        clippy::let_and_return,
        reason = "the analysis build calls `decode_slice` with its records"
    )]
    fn decode_own(&mut self, rbsp: &[u8], predicted: bool) -> Result<OwnPicture> {
        let geometry = self
            .geometry
            .ok_or(DecodeError::Malformed("slice before SPS"))?;
        if predicted && !self.reference_border_ready {
            if let Some(reference) = &mut self.reference {
                reference.extend_borders();
            }
            self.reference_border_ready = true;
        }
        let chroma_qp = self
            .chroma_qp
            .ok_or(DecodeError::Malformed("slice before SPS"))?;
        let motion = self.motion.get_or_insert_with(|| {
            MotionGrid::new(
                geometry.coded_width,
                geometry.coded_height,
                geometry.ctu_log2,
                self.tools.merge_candidates,
            )
        });
        // The pictures a P picture's RPL0 may list: the latest and those it
        // kept.
        let latest = self
            .next_poc
            .map(|next| next.wrapping_sub(1))
            .zip(self.reference.as_ref());
        let dpb: Vec<(u8, &Picture)> = latest
            .into_iter()
            .chain(self.retained.iter().map(|(poc, picture)| (*poc, picture)))
            .collect();
        // The analysis build passes its records along (one more argument).
        #[cfg(feature = "analysis")]
        let decoded = decode_slice(
            rbsp,
            &geometry,
            &chroma_qp,
            (self.tools, self.pps_block_qp == Some(true)),
            predicted,
            (&dpb, motion),
            &self.alf_banks,
            self.next_poc,
            &self.dsp,
            &mut self.pictures,
            self.stats.as_mut(),
        );
        #[cfg(feature = "analysis")]
        if let Some(s) = &mut self.stats {
            s.finish();
        }
        #[cfg(not(feature = "analysis"))]
        let decoded = decode_slice(
            rbsp,
            &geometry,
            &chroma_qp,
            (self.tools, self.pps_block_qp == Some(true)),
            predicted,
            (&dpb, motion),
            &self.alf_banks,
            self.next_poc,
            &self.dsp,
            &mut self.pictures,
        );
        decoded
    }

    /// Keep, of the retained pictures, those the new picture's RPL0 lists
    /// (`listed` POCs); the others leave the DPB (§8.3.2).
    fn retain(&mut self, listed: &[u8]) {
        let (kept, released): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retained)
            .into_iter()
            .partition(|(poc, _)| listed.contains(poc));
        self.retained = kept;
        for (_, picture) in released {
            self.pictures.give(picture);
        }
    }

    /// Decode `units`; `false` when they end inside a general picture whose
    /// other slices are still to come.
    /// Activate PPS `id` and its SPS for the picture that starts (§7.4.2.4.2):
    /// a different SPS starts a new sequence, without the old references.
    fn activate(&mut self, id: u8) -> Result<()> {
        let rbsp = self
            .pps_rbsp
            .get(usize::from(id))
            .and_then(Option::as_deref)
            .ok_or(DecodeError::Malformed("VVC PH missing PPS"))?;
        // pps_pic_parameter_set_id u(6), then pps_seq_parameter_set_id u(4).
        let sequence_id = match rbsp {
            [first, second, ..] => usize::from((first & 3) << 2 | second >> 6),
            _ => return Err(DecodeError::Truncated),
        };
        let sequence = self.general_sequences[sequence_id]
            .as_deref()
            .ok_or(DecodeError::Malformed("VVC PPS references missing SPS"))?;
        let parameters = crate::decoder_headers::PictureParameters::parse(rbsp, sequence)?;
        if self.general_sequence.as_deref() != Some(sequence) {
            let depth = sequence.depth;
            self.colour = sequence.colour;
            self.general_sequence = Some(Box::new(sequence.clone()));
            self.start_sequence(depth);
        }
        self.general_picture = Some(Box::new(parameters));
        Ok(())
    }

    /// A new coded video sequence of samples of `depth` starts (a different
    /// SPS): the pictures the last one still holds for output go out first,
    /// then its references are dropped and decoding continues in a fresh
    /// process of the sequence's sample type.
    fn start_sequence(&mut self, depth: nc_common::BitDepth) {
        with_general!(&mut self.instance, general => general.flush_output());
        while let Some(picture) = self.instance.next_output() {
            self.earlier.push_back(picture);
        }
        self.instance = Instance::for_depth(depth);
        self.reference = None;
        self.next_poc = None;
    }

    /// The PPS id of the picture a slice NAL starts: its embedded picture
    /// header's, or the picture header NAL's before it.
    fn slice_pps_id(&self, rbsp: &[u8]) -> Result<u8> {
        if crate::decoder_headers::PictureHeader::embedded(rbsp)? {
            return crate::decoder_headers::PictureHeader::peek_pps_id(
                crate::decoder_headers::Reader::after_flag(rbsp)?,
            );
        }
        self.picture_header
            .as_ref()
            .map(|header| header.pps_id)
            .ok_or(DecodeError::Malformed("VVC missing PH"))
    }

    /// A picture header NAL: activate its PPS and keep it for the slices.
    fn accept_picture_header(&mut self, rbsp: &[u8]) -> Result<()> {
        let id = crate::decoder_headers::PictureHeader::peek_pps_id(
            crate::decoder_headers::Reader::new(rbsp),
        )?;
        self.activate(id)?;
        let mut reader = crate::decoder_headers::Reader::new(rbsp);
        let header = crate::decoder_headers::PictureHeader::parse(
            &mut reader,
            self.general_sequence.as_ref().unwrap(),
            self.general_picture.as_ref().unwrap(),
        )?;
        reader.finish(false)?;
        self.picture_header = Some(Box::new(header));
        Ok(())
    }

    /// A general slice NAL (activating its PPS when it starts a picture);
    /// `true` when it completes its picture.
    fn general_slice(&mut self, rbsp: &[u8], kind: u8, temporal_id: u8) -> Result<bool> {
        let field = if self.assembling() {
            None
        } else {
            self.activate(self.slice_pps_id(rbsp)?)?;
            self.pending_field.take()
        };
        let mut active = general::Active {
            sequence: self.general_sequence.as_deref().unwrap(),
            parameters: self.general_picture.as_deref().unwrap(),
            picture_header: self.picture_header.as_deref(),
            adaptation: &self.adaptation,
            field,
            dsp: &self.dsp,
            #[cfg(feature = "analysis")]
            stats: self.stats.as_mut(),
        };
        with_general!(
            &mut self.instance,
            general => general.decode_slice(&mut active, rbsp, kind, temporal_id)
        )
    }

    /// An end of sequence (or bitstream) NAL unit (H.266 §7.4.3.10).
    fn end_sequence(&mut self) {
        self.pending_field = None;
        with_general!(&mut self.instance, general => general.end_sequence());
    }

    /// Whether a general picture's slices are still arriving.
    fn assembling(&self) -> bool {
        with_general!(&self.instance, general => general.assembling())
    }

    fn decode_units<'a>(&mut self, units: impl IntoIterator<Item = &'a [u8]>) -> Result<bool> {
        self.shown = false;
        let mut picture = None;
        let mut general_done = false;
        for unit in units {
            if unit.len() < 2 {
                return Err(DecodeError::Truncated);
            }
            if unit[0] != 0 || unit[1].trailing_zeros() >= 3 {
                return Err(DecodeError::Unsupported("VVC layer/temporal id"));
            }
            let rbsp = unescape_rbsp(&unit[2..])?;
            let kind = unit[1] >> 3;
            // Only slices and APSs may come between the slices of a picture.
            if self.assembling() && !matches!(kind, 0..=3 | 7..=10 | 17 | 18 | 23 | 24 | 25) {
                with_general!(&mut self.instance, general => general.abandon());
                return Err(DecodeError::Malformed("VVC picture with missing slices"));
            }
            match kind {
                kind @ (super::decoder_metadata::OPI
                | super::decoder_metadata::DCI
                | super::decoder_metadata::AUD
                | super::decoder_metadata::PREFIX_SEI
                | super::decoder_metadata::SUFFIX_SEI
                | super::decoder_metadata::FILLER) => {
                    if let Some(parity) = super::decoder_metadata::validate(kind, &rbsp)? {
                        self.pending_field = Some(parity);
                    }
                }
                EOS_NUT | EOB_NUT => self.end_sequence(),
                // Single-layer streams need nothing from a VPS (§7.4.3.3).
                VPS_NUT if picture.is_none() => {}
                headers::SPS_NUT if picture.is_none() => {
                    self.accept_sps(&rbsp)?;
                }
                headers::PPS_NUT if picture.is_none() => {
                    self.accept_pps(&rbsp)?;
                }
                19 if self.general && !self.assembling() => {
                    self.accept_picture_header(&rbsp)?;
                }
                headers::PREFIX_APS_NUT if picture.is_none() && !self.general => {
                    self.accept_alf_aps(&rbsp)?;
                }
                17 | 18 if self.general => {
                    let parameters = crate::decoder_headers::Adaptation::parse(&rbsp)?;
                    let kind = parameters.kind();
                    let id = usize::from(parameters.id);
                    self.adaptation[kind][id] = Some(Box::new(parameters));
                }
                kind @ (0..=3 | 7..=10) if self.general => {
                    if general_done {
                        return Err(DecodeError::Unsupported(
                            "VVC multiple pictures per access unit",
                        ));
                    }
                    general_done = self.general_slice(&rbsp, kind, unit[1] & 7)?;
                }
                kind @ (headers::IDR_N_LP | headers::TRAIL_NUT)
                    if picture.is_none() && self.pps_block_qp.is_some() =>
                {
                    if unit[1] & 7 != 1 {
                        return Err(DecodeError::Unsupported("VVC subset temporal layer"));
                    }
                    picture = Some(self.decode_own(&rbsp, kind == headers::TRAIL_NUT)?);
                }
                _ => {
                    return Err(DecodeError::Unsupported(
                        "VVC NAL type/order or missing PPS",
                    ));
                }
            }
        }
        if general_done {
            self.shown = true;
            return Ok(true);
        }
        // A general picture still assembling, or none (skipped RASL
        // pictures), completes nothing.
        if picture.is_none() && (self.general || self.assembling()) {
            return Ok(false);
        }
        let OwnPicture {
            picture: coded,
            poc,
            listed,
        } = picture.ok_or(DecodeError::Malformed("missing VVC picture"))?;
        self.retain(&listed);
        // Extend only when a later inter picture actually uses this reference.
        // All-intra streams and the final output need no border work.
        if let Some(replaced) = self.reference.replace(coded) {
            let previous = self.next_poc.map(|next| next.wrapping_sub(1));
            // A stream with a second reference keeps every picture the new
            // one lists (§8.3.2), which a later picture may list again; its
            // borders were extended when the new picture predicted from it.
            // Other streams list only the latest picture, never again.
            match previous.filter(|poc| self.tools.second_reference && listed.contains(poc)) {
                Some(poc) => self.retained.push((poc, replaced)),
                None => self.pictures.give(replaced),
            }
        }
        if let Some(motion) = self.motion.as_mut().filter(|_| self.tools.temporal_mvp) {
            motion.commit_motion_field();
        }
        self.reference_border_ready = false;
        self.next_poc = Some(poc.wrapping_add(1));
        self.shown = true;
        Ok(true)
    }
}

/// The inter slices' coding-tree constraints of an SPS whose fields before
/// them are those of `geometry` (§7.3.2.4); only our configurations
/// (`Partitioning::of_mode`) are accepted.
fn parse_partitioning(bytes: &[u8], geometry: &headers::Geometry) -> Result<headers::Partitioning> {
    // Each configuration fixes the fields before its multi-type tree fields
    // (`MinCbSizeY` among them); the caller compares the whole SPS.
    for mode in 0..=coding_tree::MAX_MODE {
        let candidate = headers::Partitioning::of_mode(mode, geometry.ctu_log2);
        let mut r = BitReader::new(bytes);
        r.skip_bits(headers::sps_bits_before_partitioning(
            &geometry.with_partitioning(candidate),
        ))?;
        let mtt_depth = r.read_ue()?;
        let (bt_diff, tt_diff) = if mtt_depth > 0 {
            (r.read_ue()?, r.read_ue()?)
        } else {
            (0, 0)
        };
        let fields = |p: headers::Partitioning| {
            (
                u32::from(p.mtt_depth),
                u32::from(p.bt_diff),
                u32::from(p.tt_diff),
            )
        };
        if fields(candidate) == (mtt_depth, bt_diff, tt_diff) {
            return Ok(candidate);
        }
    }
    Err(DecodeError::Unsupported("VVC SPS partition constraints"))
}

fn parse_sps(bytes: &[u8]) -> Result<SequenceInfo> {
    let mut r = BitReader::new(bytes);
    // The subset has a fixed 48-bit prefix (IDs, PTL without GCI/subprofiles),
    // which holds `sps_log2_ctu_size_minus5` after its first 13 bits.
    r.skip_bits(SPS_BITS_BEFORE_CTU_SIZE)?;
    let ctu_log2 = r.read_bits(2)? as u8 + headers::MIN_CTU_LOG2;
    if !headers::CTU_LOG2_RANGE.contains(&ctu_log2) {
        return Err(DecodeError::Unsupported("VVC CTU size"));
    }
    r.skip_bits(SPS_PREFIX_BITS - SPS_BITS_BEFORE_CTU_SIZE - 2)?;
    r.skip_bits(2)?; // GDR and resampling
    let cw = r.read_ue()? as usize;
    let ch = r.read_ue()? as usize;
    // CTUs crossing the right or bottom edge split implicitly (§7.3.11.4).
    if cw == 0
        || ch == 0
        || cw > headers::MAX_DIMENSION
        || ch > headers::MAX_DIMENSION
        || !cw.is_multiple_of(headers::CODED_SIZE_MULTIPLE)
        || !ch.is_multiple_of(headers::CODED_SIZE_MULTIPLE)
    {
        return Err(DecodeError::Unsupported("VVC coded dimensions"));
    }
    let (mut width, mut height) = (cw, ch);
    if r.read_bit()? {
        let left = r.read_ue()?;
        let right = r.read_ue()? as usize;
        let top = r.read_ue()?;
        let bottom = r.read_ue()? as usize;
        if left != 0 || top != 0 || right >= cw / 2 || bottom >= ch / 2 {
            return Err(DecodeError::Malformed("VVC conformance window"));
        }
        width -= 2 * right;
        height -= 2 * bottom;
    }
    let geometry = headers::Geometry {
        width,
        height,
        coded_width: cw,
        coded_height: ch,
        ctu_log2,
        partitioning: headers::Partitioning::QUADTREE,
    };
    let geometry = geometry.with_partitioning(parse_partitioning(bytes, &geometry)?);
    let mut r = BitReader::new(bytes);
    r.skip_bits(headers::sps_bits_before_mts(&geometry))?;
    let transforms = headers::Tools {
        mts: headers::read_mts(&mut r)?,
        large_transforms: headers::read_large_transforms(bytes, &geometry)?,
        ..headers::Tools::default()
    };
    let mut r = BitReader::new(bytes);
    r.skip_bits(headers::sps_head(&geometry, transforms).bit_len())?;
    let chroma_qp = ChromaQpTable::parse(&mut r)?;
    let tools = headers::Tools {
        mts: transforms.mts,
        large_transforms: transforms.large_transforms,
        ..headers::read_sps_tools(&mut r)?
    };
    let timing = headers::read_sps_timing(&mut r)?;
    let colour = read_colour(&mut r)?;
    // The DPB size says whether the stream keeps a second reference; the
    // other tool flags were read above.
    let tools = [false, true]
        .into_iter()
        .map(|second_reference| headers::Tools {
            second_reference,
            ..tools
        })
        .find(|&tools| bytes == headers::sps(&geometry, colour, &chroma_qp, tools, timing))
        .ok_or(DecodeError::Unsupported("VVC SPS tools"))?;
    Ok(SequenceInfo {
        geometry,
        colour,
        chroma_qp,
        tools,
    })
}

/// The colour description of our SPS's VUI, `r` at `sps_vui_parameters_present_flag`.
fn read_colour(r: &mut BitReader<'_>) -> Result<nc_common::ColourInfo> {
    let mut colour = nc_common::ColourInfo::default();
    if r.read_bit()? {
        // The emitted colour-only VUI is exactly five bytes. Bound before reads.
        if r.read_ue()? != 4 {
            return Err(DecodeError::Unsupported("VVC VUI payload"));
        }
        r.align_zero()?;
        r.skip_bits(7)?; // fixed progressive/aspect/overscan/description flags
        colour.description = Some(nc_common::colour::read_code_points(r)?);
        colour.full_range = r.read_bit()?;
    }
    Ok(colour)
}

/// The slice header's ALF syntax (`None` without ALF in the SPS) and the
/// luma APS bank it names, if any.
fn slice_alf<'b>(
    r: &mut BitReader<'_>,
    tools: headers::Tools,
    alf_banks: &'b [Option<Box<crate::alf_aps::SignalledFilters>>; ALF_APS_IDS],
) -> Result<(
    Option<headers::SliceAlf>,
    Option<&'b crate::alf_aps::SignalledFilters>,
)> {
    let alf = if tools.alf {
        Some(if r.read_bit()? {
            let aps = match r.read_bits(headers::APS_ID_BITS)? {
                0 => None,
                1 => Some(r.read_bits(headers::APS_ID_BITS)? as u8),
                _ => return Err(DecodeError::Unsupported("VVC ALF APS count")),
            };
            r.skip_bits(2)?; // Cb and Cr flags, validated by the canonical header
            headers::SliceAlf::On { aps }
        } else {
            headers::SliceAlf::Off
        })
    } else {
        None
    };
    let bank = match alf {
        Some(headers::SliceAlf::On { aps: Some(id) }) => Some(
            alf_banks[usize::from(id)]
                .as_deref()
                .ok_or(DecodeError::Malformed("missing VVC ALF APS"))?,
        ),
        _ => None,
    };
    Ok((alf, bank))
}

/// What the decoder keeps from an accepted SPS.
struct SequenceInfo {
    geometry: headers::Geometry,
    colour: nc_common::ColourInfo,
    chroma_qp: ChromaQpTable,
    tools: headers::Tools,
}

fn decode_slice(
    bytes: &[u8],
    geometry: &headers::Geometry,
    chroma_qp: &ChromaQpTable,
    (tools, block_qp): (headers::Tools, bool),
    inter_frame: bool,
    (dpb, motion): (&[(u8, &Picture)], &mut MotionGrid),
    alf_banks: &[Option<Box<crate::alf_aps::SignalledFilters>>; ALF_APS_IDS],
    next_poc: Option<u8>,
    dsp: &Dsp,
    pictures: &mut PictureRecycler,
    #[cfg(feature = "analysis")] mut stats: Option<&mut nc_common::stream_stats::StreamStats>,
) -> Result<OwnPicture> {
    let (header, data) = parse_slice_header(bytes, (tools, block_qp), inter_frame, alf_banks)?;
    #[cfg(feature = "analysis")]
    if let Some(s) = stats.as_deref_mut() {
        crate::analysis::picture_facts(s, inter_frame, header.qp, data.len());
    }
    let (listed, references) = slice_references(&header, inter_frame, next_poc, dpb)?;
    let OwnSliceHeader {
        poc,
        qp,
        alf,
        bank,
        temporal_mvp,
        references: rpl,
    } = header;
    let mut reader = Reader::new(data)?;
    let mut contexts = Contexts::new(qp, inter_frame).with_dependent_quantization(tools.dep_quant);
    let (cw, ch) = (geometry.coded_width, geometry.coded_height);
    let ctu = geometry.ctu_size();
    // Keep all-intra decoding on the original unpadded allocation path.
    // A recycled picture's stale samples are safe: every coded sample is
    // reconstructed before it is read, and the border is extended before
    // this picture serves as a reference.
    let border = if inter_frame { REFERENCE_BORDER } else { 0 };
    let mut picture = pictures.take(cw, ch, border)?;
    motion.select_references(0, rpl.active_references());
    motion.reset(temporal_mvp == Some(true));
    let mut sao = Vec::new();
    let mut alf_decisions: Vec<crate::alf::CtbAlf> = Vec::new();
    let mut slice = SliceDecoder {
        reader: &mut reader,
        contexts: &mut contexts,
        motion: &mut *motion,
        picture: &mut picture,
        inter_frame,
        references,
        active_references: rpl.active,
        dsp,
        qp: [qp, chroma_qp.map(qp)],
        partitioning: geometry.partitioning,
        tools,
        block_qp: block_qp.then(|| {
            let mut map = QpMap::default();
            map.reset(cw, ch, qp);
            (QuantGroup::new(qp), map)
        }),
        slice_qp: qp,
        chroma_qp,
        #[cfg(feature = "analysis")]
        stats,
    };
    let columns = cw.div_ceil(ctu);
    for y in (0..ch).step_by(ctu) {
        for x in (0..cw).step_by(ctu) {
            let left = (x > 0).then(|| sao[sao.len() - 1]);
            let up = (y > 0).then(|| sao[sao.len() - columns]);
            sao.push(crate::sao::code(
                slice.reader,
                slice.contexts,
                crate::sao::CtuSao::default(),
                left,
                up,
            )?);
            if matches!(alf, Some(headers::SliceAlf::On { .. })) {
                let index = alf_decisions.len();
                let left = (x > 0).then(|| alf_decisions[index - 1]);
                let up = (y > 0).then(|| alf_decisions[index - columns]);
                alf_decisions.push(crate::alf::code(
                    slice.reader,
                    slice.contexts,
                    None,
                    (left, up),
                    bank.is_some(),
                )?);
            }
            slice.motion.begin_ctu(x);
            if let Some((group, map)) = slice.block_qp.as_mut() {
                group.begin_group(crate::quant_group::row_start_prediction(map, x, y));
            }
            slice.node(coding_tree::Node {
                block: nc_encoder::Block::square(x, y, geometry.ctu_log2),
                mtt_depth: 0,
            })?;
        }
    }
    let block_qp_map = slice.block_qp.map(|(_, map)| map);
    if !reader.terminate()? {
        return Err(DecodeError::Malformed("missing VVC end of slice"));
    }
    let deblock_qp = crate::filter::DeblockQp {
        frame: [qp, chroma_qp.map(qp)],
        blocks: block_qp_map.as_ref().map(|map| (map, chroma_qp)),
    };
    crate::filter::apply(dsp, &mut picture, motion, &deblock_qp);
    crate::sao::apply(dsp, &mut picture, &sao, ctu);
    if alf_decisions.iter().any(Option::is_some) {
        crate::alf::Analysis::new((dsp, nc_common::BitDepth::EIGHT), &picture, ctu, |ctb| {
            alf_decisions[ctb].is_some()
        })
        .apply(&mut picture, &alf_decisions, bank);
    }
    Ok(OwnPicture::new(picture, poc, listed))
}

/// The POCs of the pictures the RPL0 of slice `header` lists (none for an
/// intra picture), each found in `dpb` (§8.3.2), and those pictures by
/// reference index; a P picture must follow `next_poc`.
#[allow(clippy::type_complexity, reason = "the POCs and their pictures")]
fn slice_references<'a>(
    header: &OwnSliceHeader<'_>,
    inter_frame: bool,
    next_poc: Option<u8>,
    dpb: &[(u8, &'a Picture)],
) -> Result<(
    Vec<u8>,
    [Option<&'a Picture>; crate::motion::MAX_ACTIVE_REFERENCES],
)> {
    let mut references = [None; crate::motion::MAX_ACTIVE_REFERENCES];
    if !inter_frame {
        return Ok((Vec::new(), references));
    }
    if next_poc != Some(header.poc) {
        return Err(DecodeError::Malformed("out-of-order VVC picture"));
    }
    let rpl = &header.references;
    let listed: Vec<u8> = rpl.distances[..rpl.count]
        .iter()
        .map(|&distance| header.poc.wrapping_sub(distance as u8))
        .collect();
    for (index, &listed_poc) in listed.iter().enumerate() {
        let picture = dpb
            .iter()
            .find(|(held, _)| *held == listed_poc)
            .map(|(_, picture)| *picture)
            .ok_or(DecodeError::Malformed("missing VVC reference"))?;
        if let Some(reference) = references.get_mut(index) {
            *reference = Some(picture);
        }
    }
    Ok((listed, references))
}

/// What a slice of our own streams signals in its header.
struct OwnSliceHeader<'b> {
    poc: u8,
    qp: u32,
    /// `None` without ALF in the SPS.
    alf: Option<headers::SliceAlf>,
    /// The luma ALF APS bank the slice names, if any.
    bank: Option<&'b crate::alf_aps::SignalledFilters>,
    /// `ph_temporal_mvp_enabled_flag`; `None` without TMVP in the SPS.
    temporal_mvp: Option<bool>,
    /// RPL0 of a P picture (`references.rs`).
    references: crate::references::Syntax,
}

/// A picture of our own streams: its samples, POC and the POCs its RPL0
/// lists.
struct OwnPicture {
    picture: Picture,
    poc: u8,
    listed: Vec<u8>,
}

impl OwnPicture {
    fn new(picture: Picture, poc: u8, listed: Vec<u8>) -> Self {
        Self {
            picture,
            poc,
            listed,
        }
    }
}

/// Parse the slice header (with its embedded picture header) of one of our
/// pictures, validated by regenerating it canonically; returns it and the
/// slice data after it.
fn parse_slice_header<'a, 'b>(
    bytes: &'a [u8],
    (tools, block_qp): (headers::Tools, bool),
    inter_frame: bool,
    alf_banks: &'b [Option<Box<crate::alf_aps::SignalledFilters>>; ALF_APS_IDS],
) -> Result<(OwnSliceHeader<'b>, &'a [u8])> {
    let mut r = BitReader::new(bytes);
    r.skip_bits(6)?; // embedded PH flags and PPS id
    let poc = r.read_bits(8)? as u8;
    if block_qp {
        // ph_cu_qp_delta_subdiv_intra_slice (and _inter_slice), ue(v) 0:
        // one bit each, validated by the canonical header below.
        r.skip_bits(1 + usize::from(inter_frame))?;
    }
    let temporal_mvp = (inter_frame && tools.temporal_mvp)
        .then(|| r.read_bit())
        .transpose()?;
    if inter_frame {
        r.skip_bits(4)?; // L1-zero flag, P slice type
    } else if poc != 0 {
        return Err(DecodeError::Malformed("VVC IDR POC"));
    } else {
        r.skip_bits(1)?; // no-output flag
    }
    let (alf, bank) = slice_alf(&mut r, tools, alf_banks)?;
    let references = if inter_frame {
        read_reference_lists(&mut r, temporal_mvp == Some(true))?
    } else {
        crate::references::Syntax::FLAT
    };
    let qp = 26i64 + i64::from(r.read_se()?);
    if !(0..=i64::from(headers::MAX_QP)).contains(&qp) {
        return Err(DecodeError::Malformed("VVC QP"));
    }
    let qp = qp as u32;
    r.skip_bits(3)?; // SAO luma/chroma and sign hiding flags, validated by canonical header
    r.trailing_bits()?;
    let prefix = headers::slice_layered(
        qp,
        inter_frame,
        poc,
        alf,
        block_qp,
        temporal_mvp,
        &references,
    );
    if !bytes.starts_with(&prefix) {
        return Err(DecodeError::Unsupported("VVC slice header"));
    }
    Ok((
        OwnSliceHeader {
            poc,
            qp,
            alf,
            bank,
            temporal_mvp,
            references,
        },
        r.remaining_bytes()?,
    ))
}

/// A P slice's local RPL0 (one or two short-term entries), empty RPL1,
/// active-reference override and collocated reference (§7.3.7.1,
/// §7.3.10), as `headers::slice_layered` writes them; the caller checks
/// the whole header against its canonical form.
fn read_reference_lists(
    r: &mut BitReader<'_>,
    temporal_mvp: bool,
) -> Result<crate::references::Syntax> {
    let mut syntax = crate::references::Syntax::FLAT;
    let count = r.read_ue()? as usize;
    if !(1..=syntax.distances.len()).contains(&count) {
        return Err(DecodeError::Unsupported("VVC RPL0 size"));
    }
    let mut previous = 0;
    for distance in &mut syntax.distances[..count] {
        // Cumulative delta, then strp_entry_sign_flag (earlier pictures).
        *distance = previous + r.read_ue_max(u32::from(u8::MAX), "abs_delta_poc_st")? + 1;
        r.skip_bits(1)?;
        previous = *distance;
    }
    syntax.count = count;
    r.read_ue()?; // RPL1, empty
    if count > 1 && r.read_bit()? {
        syntax.active = r.read_ue_max(1, "sh_num_ref_idx_active_minus1")? as usize + 1;
    }
    if temporal_mvp && syntax.active > 1 {
        r.read_ue()?; // sh_collocated_ref_idx
    }
    Ok(syntax)
}

struct SliceDecoder<'a, 'b> {
    reader: &'a mut Reader<'b>,
    contexts: &'a mut Contexts,
    motion: &'a mut MotionGrid,
    picture: &'a mut Picture,
    inter_frame: bool,
    /// The pictures RPL0 lists (the first `active_references` active).
    references: [Option<&'a Picture>; crate::motion::MAX_ACTIVE_REFERENCES],
    /// `NumRefIdxActive[0]`.
    active_references: usize,
    dsp: &'a Dsp,
    /// Luma and chroma QP.
    qp: [u32; 2],
    /// The inter slices' coding-tree constraints.
    partitioning: coding_tree::Partitioning,
    /// The SPS's optional tools.
    tools: headers::Tools,
    /// The quantization groups and `QpY` map, when the PPS enables
    /// `cu_qp_delta`.
    block_qp: Option<(QuantGroup, QpMap)>,
    slice_qp: u32,
    chroma_qp: &'a ChromaQpTable,
    /// Where analysis records go, when they are collected.
    #[cfg(feature = "analysis")]
    stats: Option<&'a mut nc_common::stream_stats::StreamStats>,
}
impl SliceDecoder<'_, '_> {
    /// §7.3.11.4 `coding_tree` of `node`: nodes crossing the picture's
    /// right or bottom edge split (by quadtree in our streams), nodes wholly
    /// outside it are absent; a shape's leaves are coding units.
    fn node(&mut self, node: coding_tree::Node) -> Result<()> {
        let block = node.block;
        let (width, height) = (self.picture.width(), self.picture.height());
        if block.x >= width || block.y >= height {
            return Ok(());
        }
        let limits = if self.inter_frame {
            self.partitioning
        } else {
            self.partitioning.intra()
        };
        let allowed = coding_tree::Allowed::of(&node, limits, (width, height));
        let contexts = self.motion.split_contexts(&node, allowed);
        let split = coding_tree::code(
            self.reader,
            &mut self.contexts.tree,
            allowed,
            contexts,
            coding_tree::TreeSplit::None,
        )?;
        match split {
            coding_tree::TreeSplit::None => self.coding_unit(block),
            coding_tree::TreeSplit::Quad => {
                for child in block.quadrants() {
                    self.node(coding_tree::Node {
                        block: child,
                        mtt_depth: 0,
                    })?;
                }
                Ok(())
            }
            coding_tree::TreeSplit::Shape(shape) => {
                // Our streams split across the picture edge by quadtree,
                // and never into leaves below 8 a side (which would bring
                // §7.3.11.4's mode constraints).
                if !block.is_inside(width, height)
                    || shape.log2_min_side(block.log2_size()) < headers::MIN_QT_LOG2
                {
                    return Err(DecodeError::Unsupported("VVC multi-type split"));
                }
                let parts = shape.parts(&block);
                for part in parts.into_iter().take(shape.part_count()) {
                    self.node(coding_tree::Node {
                        block: part,
                        mtt_depth: node.mtt_depth + 1,
                    })?;
                }
                Ok(())
            }
        }
    }

    /// Decode and reconstruct coding unit `block`.
    fn coding_unit(&mut self, block: nc_encoder::Block) -> Result<()> {
        let p = self.motion.predictors(&block, 0);
        let pending = self
            .block_qp
            .as_ref()
            .is_some_and(|(group, _)| group.delta_pending());
        let coding = decoder_syntax::block(
            self.reader,
            self.contexts,
            (self.inter_frame, self.active_references, self.tools),
            (self.motion, &p),
            block,
            pending,
        )?;
        let qp_y = match self.block_qp.as_mut() {
            Some((group, map)) => {
                let qp_y = group.finish_decoded(coding.qp_delta);
                map.record(&block, qp_y);
                qp_y
            }
            None => self.slice_qp,
        };
        self.qp = [qp_y, self.chroma_qp.map(qp_y)];
        for plane in PlaneIndex::ALL {
            match coding.isp {
                Some(split) if plane == PlaneIndex::Y => {
                    self.reconstruct_isp(&block, &coding, split);
                }
                _ => self.reconstruct(&block, &coding, plane)?,
            }
        }
        let units = crate::transform_tree::coded_units(
            &block,
            coding.isp,
            &coding.levels,
            self.tools.max_tb_log2(),
        );
        #[cfg(feature = "analysis")]
        if let Some(s) = self.stats.as_deref_mut() {
            crate::analysis::coding_unit(s, &block, &coding, self.qp[0], &units);
        }
        let motion = coding.inter.map(|inter| inter.motion);
        let ref_idx = coding.inter.map_or(0, |inter| inter.ref_idx);
        self.motion.record(
            &block,
            motion.as_ref(),
            ref_idx,
            coding.skip,
            coding.intra_mode,
            &units,
        );
        Ok(())
    }

    /// Predict and reconstruct the luma of CU `cu` coded with intra
    /// sub-partitions `split`, one after the other.
    fn reconstruct_isp(
        &mut self,
        cu: &nc_encoder::Block,
        coding: &crate::syntax::Coding,
        split: crate::isp::Split,
    ) {
        let references =
            crate::intra::References::new(self.picture, Some(self.motion), PlaneIndex::Y, cu, cu);
        let mut predictor = crate::isp::Predictor::new(&references, cu, split, coding.intra_mode);
        for (index, part) in crate::isp::parts(cu, split).into_iter().enumerate() {
            let pred = predictor.predict(self.dsp, index, |region| {
                crate::isp::previous_in(self.picture, cu, split, region)
            });
            recon::store(
                self.dsp,
                self.picture,
                PlaneIndex::Y,
                (part.x, part.y, part.width()),
                &pred,
                crate::transform_tree::unit_levels(&coding.levels[0], index, part.area()),
                recon::Scaling {
                    qp: self.qp[0],
                    dependent: self.tools.dep_quant,
                    transform: crate::mts::luma(self.tools.mts, true, true, 0),
                },
            );
        }
    }

    /// Predict and reconstruct `plane` of CU `cu`, transform unit by
    /// transform unit (intra prediction reads the units before it).
    fn reconstruct(
        &mut self,
        cu: &nc_encoder::Block,
        coding: &crate::syntax::Coding,
        plane: PlaneIndex,
    ) -> Result<()> {
        let shift = usize::from(plane != PlaneIndex::Y);
        let scaling = recon::Scaling {
            qp: self.qp[shift],
            dependent: self.tools.dep_quant,
            transform: if plane == PlaneIndex::Y {
                crate::mts::luma(self.tools.mts, coding.inter.is_none(), false, coding.mts)
            } else {
                crate::mts::Selection::Dct2
            },
        };
        let levels = &coding.levels[plane as usize];
        let units = crate::transform_tree::units(cu, self.tools.max_tb_log2());
        let Some(inter) = coding.inter else {
            for (index, tu) in units.enumerate() {
                let samples = crate::transform_tree::plane_samples(&tu, plane as usize);
                let pred = crate::intra::prediction(
                    self.dsp,
                    self.picture,
                    Some(self.motion),
                    plane,
                    (&tu, cu),
                    if plane == PlaneIndex::Y {
                        coding.intra_mode
                    } else {
                        coding.chroma_mode
                    },
                );
                recon::store(
                    self.dsp,
                    self.picture,
                    plane,
                    (tu.x >> shift, tu.y >> shift, tu.width() >> shift),
                    &pred,
                    crate::transform_tree::unit_levels(levels, index, samples),
                    scaling,
                );
            }
            return Ok(());
        };
        // Inter prediction reads only the reference: the whole CU is
        // predicted straight into the picture, then each unit's residual is
        // added in place.
        let reference = self
            .references
            .get(usize::from(inter.ref_idx))
            .copied()
            .flatten()
            .ok_or(DecodeError::Malformed("VVC inter CU without reference"))?;
        let mut prediction = block_region(self.picture, plane, cu);
        crate::inter::predict_cu_view(
            self.dsp,
            reference,
            plane,
            cu,
            &inter.motion,
            &mut prediction,
        );
        if levels.is_empty() {
            return Ok(());
        }
        for (index, tu) in units.enumerate() {
            let samples = crate::transform_tree::plane_samples(&tu, plane as usize);
            let unit = crate::transform_tree::unit_levels(levels, index, samples);
            recon::add(
                self.dsp,
                &mut block_region(self.picture, plane, &tu),
                unit,
                scaling,
            );
        }
        Ok(())
    }
}

/// The samples of `plane` of `picture` that luma block `block` covers.
fn block_region<'a>(
    picture: &'a mut Picture,
    plane: PlaneIndex,
    block: &nc_encoder::Block,
) -> nc_common::PlaneViewMut<'a> {
    let shift = usize::from(plane != PlaneIndex::Y);
    let (x, y) = (block.x >> shift, block.y >> shift);
    let (w, h) = (block.width() >> shift, block.height() >> shift);
    picture
        .plane_mut(plane)
        .region_mut(x as isize, y as isize, w, h)
        .expect("coded blocks lie inside the picture")
}
