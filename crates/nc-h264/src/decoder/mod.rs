//! H.264 decoder of frames for the High and High 10 profiles (Annex
//! A.2.4, A.2.5): I, P and B slices of 8- to 10-bit 4:2:0 or 4:0:0 video,
//! CAVLC and CABAC, 8x8 transforms and intra prediction, scaling matrices,
//! spatial and temporal direct prediction, explicit and implicit weighted
//! prediction, any number of slices per picture in any order, slice groups
//! (FMO), redundant pictures (discarded), long-term references, memory
//! management operations and `frame_num` gaps. That covers the Baseline,
//! Main, High and High 10 streams of frames. Everything else (field and
//! MBAFF pictures, SP/SI slices, data partitioning, other chroma formats,
//! deeper samples) is rejected as unsupported.
//!
//! Module map:
//! - parameter sets and headers: `sps`, `pps`, `scaling`, `slice`, `poc`,
//!   `slice_groups` (FMO);
//! - the decoding process of one sample type: `core` (`u8` for 8-bit
//!   streams, `u16` deeper ones, `sample` names what differs);
//! - reference handling: `dpb` (marking, lists) and `output` (output order);
//! - slice data: `slice_data` (macroblock loop), `mb` (macroblock layer),
//!   `syntax` (the entropy interface) with `cavlc` / `vlc` and
//!   `cabac_syntax` / `cabac`, `mbstate` (neighbour state), `mvpred`,
//!   `direct` (B direct prediction);
//! - reconstruction: `recon`, `intra`, `inter`, `transform`, `deblock`;
//! - `reader`: the RBSP bit reader;
//! - `stats`: coding-tool analysis (feature `analysis`;
//!   `Decoder::collect_stats`).
//!
//! [`Decoder::push_nal`] takes NAL units one at a time; the pictures come
//! out of [`Decoder::next_output`] in output order, 8-bit ones as `u8`
//! samples and deeper ones as `u16` ([`DecodedPicture`]).
//! [`Decoder::decode`] takes one access unit of an 8-bit stream per call
//! and returns the first picture it outputs (low delay for streams that do
//! not reorder); [`Decoder::decode_shown`] does the same without the output
//! copy, showing the picture through [`Decoder::shown`] until the next call.

mod cabac;
mod cabac_syntax;
mod cavlc;
mod core;
mod deblock;
mod direct;
mod dpb;
mod inter;
mod intra;
mod mb;
mod mbstate;
mod mvpred;
mod output;
mod poc;
mod pool;
mod pps;
mod reader;
mod recon;
mod sample;
mod scaling;
mod slice;
mod slice_data;
mod slice_groups;
mod sps;
#[cfg(feature = "analysis")]
mod stats;
mod syntax;
#[cfg(test)]
mod tests;
mod transform;
mod vlc;

use std::collections::VecDeque;
use std::sync::Arc;

pub use nc_common::decode::DecodeError;
use nc_common::decode::{annex_b_units, unescape_rbsp};
#[cfg(feature = "analysis")]
use nc_common::stream_stats::StreamStats;
use nc_common::{
    BitDepths, ColourInfo, DecodedPicture, OutputPicture, Picture, PlaneViewMut, ShownPicture,
};

use self::core::{Core, Finished};
#[cfg(test)]
pub(crate) use cavlc::block_for_tests as cavlc_block_for_tests;
use output::DecodedFrame;
use pps::Pps;
use sample::H264Sample;
use sps::Sps;

type Result<T> = std::result::Result<T, DecodeError>;

/// Largest packet [`Decoder::decode`] accepts.
const MAX_PACKET_BYTES: usize = 64 * 1024 * 1024;
/// NAL unit header fields (§7.3.1).
const FORBIDDEN_ZERO_BIT: u8 = 0x80;
const NAL_REF_IDC_SHIFT: u8 = 5;
const NAL_REF_IDC_MASK: u8 = 0x03;
const NAL_TYPE_MASK: u8 = 0x1f;
/// `nal_unit_type` values (Table 7-1).
const NAL_SLICE: u8 = 1;
const NAL_PARTITION_A: u8 = 2;
const NAL_PARTITION_C: u8 = 4;
const NAL_IDR_SLICE: u8 = 5;
const NAL_SPS: u8 = 7;
const NAL_PPS: u8 = 8;
/// Number of SPS and PPS ids (§7.4.2.1.1, §7.4.2.2).
const SPS_IDS: usize = sps::MAX_SPS_ID as usize + 1;
const PPS_IDS: usize = pps::MAX_PPS_ID as usize + 1;
/// Writable views of a picture's visible Y, Cb and Cr samples, made once per
/// slice (or deblocking pass) rather than per macroblock.
type Planes<'a, T = u8> = [PlaneViewMut<'a, T>; 3];

/// The decoding core of the current sequence's sample type.
#[derive(Debug)]
enum Instance {
    /// 8-bit luma and chroma.
    Eight(Box<Core<u8>>),
    /// Luma or chroma deeper than 8 bits, in `u16` samples.
    High(Box<Core<u16>>),
}

/// Run `$body` on the current instance's core, bound to `$core`.
macro_rules! with_core {
    ($instance:expr, $core:ident => $body:expr) => {
        match $instance {
            Instance::Eight($core) => $body,
            Instance::High($core) => $body,
        }
    };
}

/// The error of the 8-bit picture APIs ([`Decoder::decode`],
/// [`Decoder::decode_shown`]) for a deeper stream.
const EIGHT_BIT_ONLY: DecodeError = DecodeError::Unsupported("H264 bit depth above 8");

/// An H.264 High / High 10 decoder of frames.
#[derive(Debug)]
pub struct Decoder {
    sps: Vec<Option<Arc<Sps>>>,
    pps: Vec<Option<Arc<Pps>>>,
    instance: Instance,
    /// Pictures due for output, in output order.
    ready: VecDeque<DecodedPicture>,
    /// The picture [`Decoder::shown`] / [`Decoder::shown_high`] show: the
    /// first one the last [`Decoder::decode_shown`] access unit output,
    /// sharing its samples with the DPB; its buffer goes back to the pool
    /// once the next call drops this.
    shown: Option<DecodedPicture>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// A due output frame as a [`DecodedPicture`].
fn output_picture<T: H264Sample>(frame: DecodedFrame<T>) -> OutputPicture<T> {
    OutputPicture {
        picture: frame.picture,
        origin: frame.origin,
        size: frame.size,
        depths: frame.depths,
    }
}

impl Decoder {
    /// An empty decoder; the stream must start with parameter sets and an
    /// IDR picture.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sps: vec![None; SPS_IDS],
            pps: vec![None; PPS_IDS],
            instance: Instance::Eight(Box::new(Core::new())),
            ready: VecDeque::new(),
            shown: None,
        }
    }

    /// Start collecting coding-tool analysis into `stats` (block records,
    /// and per-picture tallies unless [`StreamStats::records_only`]) from
    /// the next picture on. Decoding is unchanged.
    #[cfg(feature = "analysis")]
    pub fn collect_stats(&mut self, stats: StreamStats) {
        with_core!(&mut self.instance, core => core.stats = Some(stats));
    }

    /// The analysis collected so far, if enabled: the block records of
    /// the last completed picture and the tallies of every completed one.
    #[cfg(feature = "analysis")]
    #[must_use]
    pub fn stats(&self) -> Option<&StreamStats> {
        with_core!(&self.instance, core => core.stats.as_ref())
    }

    /// Complete the analysis of the last picture and hand it over;
    /// collection stops.
    #[cfg(feature = "analysis")]
    pub fn take_stats(&mut self) -> Option<StreamStats> {
        let mut stats = with_core!(&mut self.instance, core => core.stats.take())?;
        stats.finish();
        Some(stats)
    }

    /// Test hook: overwrite every reference picture with `value`, as a
    /// decoder that lost a frame would hold wrong references.
    #[cfg(test)]
    pub(crate) fn corrupt_references(&mut self, value: u8) {
        if let Instance::Eight(core) = &mut self.instance {
            core.corrupt_references(value);
        }
    }

    /// Hand back a picture this decoder returned, once the caller is done
    /// with it: a later picture reuses its buffer.
    pub fn recycle(&mut self, picture: Picture) {
        if let Instance::Eight(core) = &mut self.instance {
            core.pool.give_back(picture);
        }
    }

    /// Colour signalling of the active SPS (`None` before the first one).
    #[must_use]
    pub fn colour(&self) -> Option<ColourInfo> {
        with_core!(&self.instance, core => core.active_sps.as_ref().map(|sps| sps.colour))
    }

    /// Decode one access unit (Annex-B, one picture) of an 8-bit stream and
    /// return a copy of the first picture it outputs (§C.4.5.3 bumping; the
    /// low-delay use: a stream that never reorders outputs each picture as
    /// it is decoded). Further pictures it outputs wait for
    /// [`Decoder::next_shown`] or [`Decoder::next_output`].
    ///
    /// # Errors
    ///
    /// Rejects truncated, malformed, unsupported or excessively large input,
    /// packets that do not complete exactly one picture, an access unit that
    /// outputs no picture, and deeper streams ([`Decoder::push_nal`] decodes
    /// those).
    pub fn decode(&mut self, packet: &[u8]) -> Result<Picture> {
        self.decode_shown(packet)?;
        let (Some(DecodedPicture::Eight(shown)), Instance::Eight(core)) =
            (&self.shown, &mut self.instance)
        else {
            return Err(DecodeError::Malformed("packet outputs no picture"));
        };
        let view = shown.view();
        let mut picture = core.pool.take(view.width(), view.height(), 0)?;
        view.copy_to(&mut picture);
        Ok(picture)
    }

    /// [`Decoder::decode`] without the output copy: the picture is shown
    /// by [`Decoder::shown`] until the next call; an access unit that
    /// outputs none (its picture is held for reordering) shows nothing.
    ///
    /// # Errors
    ///
    /// As [`Decoder::decode`], except for an access unit without output.
    pub fn decode_shown(&mut self, packet: &[u8]) -> Result<()> {
        self.decode_shown_any_depth(packet)?;
        if self.is_high_depth() {
            return Err(EIGHT_BIT_ONLY);
        }
        Ok(())
    }

    /// [`Self::decode_shown`] for streams of any supported depth: the shown
    /// picture is borrowed through [`Self::shown`] (8-bit) or
    /// [`Self::shown_high`] (deep).
    /// # Errors
    /// As [`Self::decode_shown`], except that supported deep streams are accepted.
    pub fn decode_shown_any_depth(&mut self, packet: &[u8]) -> Result<()> {
        // Release the previous picture first, so this one can reuse it.
        self.shown = None;
        if packet.len() > MAX_PACKET_BYTES {
            return Err(DecodeError::Unsupported("H264 packet allocation limit"));
        }
        let mut completed = 0;
        for unit in annex_b_units(packet)? {
            completed += usize::from(self.decode_unit(unit)?);
        }
        match completed {
            0 => return Err(DecodeError::Malformed("packet does not complete a picture")),
            1 => {}
            _ => return Err(DecodeError::Unsupported("several pictures in one packet")),
        }
        self.next_shown();
        Ok(())
    }

    /// Show the next picture output but not shown yet (a stream that
    /// reorders can output several at an IDR picture); `false` (nothing
    /// shown) when there is none.
    pub fn next_shown(&mut self) -> bool {
        self.shown = self.ready.pop_front();
        self.shown.is_some()
    }

    /// Whether the active sequence stores its samples in u16.
    #[must_use]
    pub fn is_high_depth(&self) -> bool {
        matches!(self.instance, Instance::High(_))
    }

    /// The shown picture ([`Self::decode_shown_any_depth`]) of a deep stream.
    #[must_use]
    pub fn shown_high(&self) -> Option<(ShownPicture<'_, u16>, BitDepths)> {
        match self.shown.as_ref()? {
            DecodedPicture::High(picture) => Some((picture.view(), picture.depths)),
            DecodedPicture::Eight(_) => None,
        }
    }

    /// The shown picture ([`Decoder::decode_shown`]) of an 8-bit stream.
    #[must_use]
    pub fn shown(&self) -> Option<ShownPicture<'_>> {
        match self.shown.as_ref()? {
            DecodedPicture::Eight(picture) => Some(picture.view()),
            DecodedPicture::High(_) => None,
        }
    }

    /// Decode one NAL unit (header and payload, without start code) of a
    /// stream of any supported depth; the pictures it completes become due
    /// for [`Decoder::next_output`] in output order.
    ///
    /// # Errors
    ///
    /// Malformed or unsupported syntax. A picture that fails is dropped.
    pub fn push_nal(&mut self, nal: &[u8]) -> Result<()> {
        self.decode_unit(nal)?;
        Ok(())
    }

    /// The next picture in output order, if one is due.
    pub fn next_output(&mut self) -> Option<DecodedPicture> {
        self.ready.pop_front()
    }

    /// Make every pending picture due (end of stream).
    pub fn flush(&mut self) {
        let ready = &mut self.ready;
        match &mut self.instance {
            Instance::Eight(core) => ready.extend(
                core.queue
                    .flush()
                    .into_iter()
                    .map(|f| DecodedPicture::Eight(output_picture(f))),
            ),
            Instance::High(core) => ready.extend(
                core.queue
                    .flush()
                    .into_iter()
                    .map(|f| DecodedPicture::High(output_picture(f))),
            ),
        }
    }

    /// One NAL unit; returns whether it completed a picture, which then
    /// goes through the output queue.
    fn decode_unit(&mut self, nal: &[u8]) -> Result<bool> {
        let (&header, payload) = nal.split_first().ok_or(DecodeError::Truncated)?;
        if header & FORBIDDEN_ZERO_BIT != 0 {
            return Err(DecodeError::Malformed("forbidden_zero_bit"));
        }
        let nal_ref_idc = u32::from((header >> NAL_REF_IDC_SHIFT) & NAL_REF_IDC_MASK);
        match header & NAL_TYPE_MASK {
            NAL_SLICE | NAL_IDR_SLICE => {
                let idr = header & NAL_TYPE_MASK == NAL_IDR_SLICE;
                let rbsp = unescape_rbsp(payload)?;
                self.slice(&rbsp, (idr, nal_ref_idc))
            }
            NAL_SPS => {
                let sps = sps::parse(&unescape_rbsp(payload)?)?;
                let id = sps.id as usize;
                self.sps[id] = Some(Arc::new(sps));
                Ok(false)
            }
            NAL_PPS => {
                let pps = pps::parse(&unescape_rbsp(payload)?)?;
                let id = pps.id as usize;
                self.pps[id] = Some(Arc::new(pps));
                Ok(false)
            }
            NAL_PARTITION_A..=NAL_PARTITION_C => {
                Err(DecodeError::Unsupported("H264 data partitioning"))
            }
            // SEI, delimiters, end of sequence/stream, filler, SPS
            // extensions, auxiliary pictures and the SVC/MVC extension NAL
            // units do not affect the primary pictures of these profiles.
            _ => Ok(false),
        }
    }

    /// One slice NAL unit's RBSP, decoded by the core of its SPS's depth:
    /// an IDR picture of the other depth starts a new core (the previous
    /// one's pictures are output first). Returns whether it completed a
    /// picture.
    fn slice(&mut self, rbsp: &[u8], (idr, nal_ref_idc): (bool, u32)) -> Result<bool> {
        if idr && nal_ref_idc == 0 {
            return Err(DecodeError::Malformed("IDR picture with nal_ref_idc 0"));
        }
        let pps_id = slice::parse_pps_id(rbsp)? as usize;
        // Missing parameter sets: the IDR access unit carrying them was lost.
        let pps = self.pps[pps_id].clone().ok_or(DecodeError::MissingKey)?;
        let sps = self.sps[pps.sps_id as usize]
            .clone()
            .ok_or(DecodeError::MissingKey)?;
        let deep = !sps.depths.is_eight();
        let switch = match &self.instance {
            Instance::Eight(_) => deep,
            Instance::High(_) => !deep,
        };
        if switch {
            if !idr {
                // The core of this depth has no sequence: its IDR was lost.
                return Err(DecodeError::MissingKey);
            }
            self.flush();
            self.instance = if deep {
                Instance::High(Box::new(Core::new()))
            } else {
                Instance::Eight(Box::new(Core::new()))
            };
        }
        let params = (&sps, &pps);
        let ready = &mut self.ready;
        Ok(match &mut self.instance {
            Instance::Eight(core) => slice_on(core, rbsp, (idr, nal_ref_idc), params)?
                .map(|f| {
                    let due = core.queue.push(f.frame());
                    ready.extend(
                        due.into_iter()
                            .map(|f| DecodedPicture::Eight(output_picture(f))),
                    );
                })
                .is_some(),
            Instance::High(core) => slice_on(core, rbsp, (idr, nal_ref_idc), params)?
                .map(|f| {
                    let due = core.queue.push(f.frame());
                    ready.extend(
                        due.into_iter()
                            .map(|f| DecodedPicture::High(output_picture(f))),
                    );
                })
                .is_some(),
        })
    }
}

/// One slice on `core`; the picture in progress is dropped if it fails.
fn slice_on<T: H264Sample>(
    core: &mut Core<T>,
    rbsp: &[u8],
    nal: (bool, u32),
    params: (&Arc<Sps>, &Arc<Pps>),
) -> Result<Option<Finished<T>>> {
    let result = core.slice(rbsp, nal, params);
    if result.is_err() {
        core.drop_current();
    }
    result
}
