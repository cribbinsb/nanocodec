//! The public AV2 decoder: the decoding process ([`Core`]) instantiated
//! for the current sequence's samples, `u8` for 8-bit sequences and `u16`
//! for 10- and 12-bit ones (docs/handoff/high-bit-depth.md).
//!
//! The instance is chosen per sequence: before a temporal unit whose
//! sequence header needs the other sample type is decoded, the decoder
//! continues in a new instance of that type (a new sequence empties every
//! reference slot, so nothing else carries over), keeping the caller's
//! settings. Pictures the old instance still had queued are shown first.
//!
//! The picture-returning and browser calls ([`Decoder::decode`],
//! [`Decoder::decode_ref`], [`Decoder::decode_shown`], [`Decoder::shown`])
//! are 8-bit and refuse deeper sequences; [`Decoder::shown_frame`] shows a
//! picture of either depth.

use nc_common::decode::{DecodeError, Result};
#[cfg(feature = "analysis")]
use nc_common::stream_stats::StreamStats;
use nc_common::{BitDepths, Picture, ShownFrame, ShownPicture};

use super::decoder::{BlockSink, Core, other_obu};
use super::sequence;
use crate::obu;

/// What the 8-bit-only calls return in a deeper sequence.
const EIGHT_BIT_API: DecodeError =
    DecodeError::Unsupported("AV2 picture of more than 8 bits through an 8-bit call");

/// The decoding process of one sequence's sample type.
#[derive(Debug)]
enum Instance {
    Eight(Box<Core<u8>>),
    High(Box<Core<u16>>),
}

/// Run `$body` on an instance's core, bound to `$core`.
macro_rules! with_core {
    ($instance:expr, $core:ident => $body:expr) => {
        match $instance {
            Instance::Eight($core) => $body,
            Instance::High($core) => $body,
        }
    };
}

impl Instance {
    /// The picture this instance shows, of either depth.
    fn shown_frame(&self) -> Option<ShownFrame<'_>> {
        match self {
            Self::Eight(core) => core.shown().map(ShownFrame::Eight),
            Self::High(core) => {
                let depths = BitDepths {
                    luma: core.depth(),
                    chroma: core.depth(),
                };
                core.shown().map(|shown| ShownFrame::High(shown, depths))
            }
        }
    }
}

/// An AV2 decoder: one temporal unit in, the pictures it shows queued in
/// display order. A temporal unit that fails leaves it as it was.
#[derive(Debug)]
pub struct Decoder {
    instance: Instance,
    /// The instance of the previous sequence while it still has pictures
    /// queued; they are shown before the current instance's.
    draining: Option<Instance>,
    /// Whether the picture shown is the draining instance's.
    showing_draining: bool,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            instance: Instance::Eight(Box::default()),
            draining: None,
            showing_draining: false,
        }
    }
}

impl Decoder {
    /// Continue in the instance `unit`'s sequence header needs, if it has
    /// one of the other sample type.
    fn prepare(&mut self, unit: &[u8]) -> Result<()> {
        let Some(payload) = sequence_header(unit)? else {
            return Ok(());
        };
        let deep = sequence::parse(payload)?.high_bit_depth != 0;
        let next = match &mut self.instance {
            Instance::Eight(core) if deep => Instance::High(Box::new(core.successor())),
            Instance::High(core) if !deep => Instance::Eight(Box::new(core.successor())),
            _ => return Ok(()),
        };
        let previous = std::mem::replace(&mut self.instance, next);
        let queued = with_core!(&previous, core => core.has_output());
        self.draining = queued.then_some(previous);
        self.showing_draining = false;
        Ok(())
    }

    /// Start collecting coding-tool analysis into `stats` (block records,
    /// and per-frame tallies unless `records_only`) from the next frame on.
    /// Decoding is unchanged.
    #[cfg(feature = "analysis")]
    pub fn collect_stats(&mut self, stats: StreamStats) {
        with_core!(&mut self.instance, core => core.collect_stats(stats));
    }

    /// The analysis collected so far, if enabled: the block records of the
    /// last decoded frame and the tallies of every completed one.
    #[cfg(feature = "analysis")]
    #[must_use]
    pub fn stats(&self) -> Option<&StreamStats> {
        with_core!(&self.instance, core => core.stats())
    }

    /// Complete the analysis and hand it over; collection stops.
    #[cfg(feature = "analysis")]
    pub fn take_stats(&mut self) -> Option<StreamStats> {
        with_core!(&mut self.instance, core => core.take_stats())
    }

    /// Decode one temporal unit: its pictures to show join the output
    /// queue.
    /// # Errors
    /// Rejects missing references, unsupported or malformed syntax; the
    /// decoder is then as before the unit.
    pub fn decode_unit(&mut self, unit: &[u8]) -> Result<()> {
        self.prepare(unit)?;
        with_core!(&mut self.instance, core => core.decode_unit(unit))
    }

    /// Show the next queued picture ([`Decoder::shown_frame`]); `false`
    /// when none is queued.
    pub fn next_output(&mut self) -> bool {
        if let Some(draining) = &mut self.draining {
            if with_core!(draining, core => core.next_output()) {
                self.showing_draining = true;
                return true;
            }
            self.draining = None;
        }
        self.showing_draining = false;
        with_core!(&mut self.instance, core => core.next_output())
    }

    /// End of stream: queue the hidden frames still due for output.
    pub fn flush(&mut self) {
        with_core!(&mut self.instance, core => core.flush());
    }

    /// The instance whose picture is shown.
    fn showing(&self) -> &Instance {
        match &self.draining {
            Some(draining) if self.showing_draining => draining,
            _ => &self.instance,
        }
    }

    /// The visible samples (the sequence's conformance window) of the
    /// picture shown, of either depth.
    #[must_use]
    pub fn shown_frame(&self) -> Option<ShownFrame<'_>> {
        self.showing().shown_frame()
    }

    /// Whether the shown picture's sequence is monochrome: its chroma
    /// planes are mid-grey, and avmdec writes its luma alone.
    #[must_use]
    pub fn is_monochrome(&self) -> bool {
        with_core!(self.showing(), core => core.is_monochrome())
    }

    /// [`Decoder::shown_frame`] of an 8-bit picture; `None` for a deeper
    /// one.
    #[must_use]
    pub fn shown(&self) -> Option<ShownPicture<'_>> {
        match self.showing() {
            Instance::Eight(core) => core.shown(),
            Instance::High(_) => None,
        }
    }

    /// Decode a temporal unit that shows one picture (as our encoder's
    /// do) and show it.
    /// # Errors
    /// As [`Decoder::decode_unit`], units that show nothing, and pictures
    /// of more than 8 bits.
    pub fn decode_shown(&mut self, packet: &[u8]) -> Result<()> {
        self.prepare(packet)?;
        self.draining = None;
        match &mut self.instance {
            Instance::Eight(core) => core.decode_shown(packet),
            Instance::High(_) => Err(EIGHT_BIT_API),
        }
    }

    /// Decode a packet, returning the borrowed visible picture (the
    /// reconstruction inside the sequence's conformance window) to avoid a
    /// frame copy when nothing is cropped. The returned picture remains
    /// valid until the next mutable decoder operation.
    /// # Errors
    /// As [`Decoder::decode_shown`].
    pub fn decode_ref(&mut self, packet: &[u8]) -> Result<&Picture> {
        self.prepare(packet)?;
        self.draining = None;
        match &mut self.instance {
            Instance::Eight(core) => core.decode_ref(packet),
            Instance::High(_) => Err(EIGHT_BIT_API),
        }
    }

    /// Decode a packet and return an owned copy; use `decode_ref` to avoid
    /// the copy.
    /// # Errors
    /// As [`Decoder::decode_shown`].
    pub fn decode(&mut self, packet: &[u8]) -> Result<Picture> {
        self.decode_ref(packet).cloned()
    }

    /// Diagnosis: output general-path frames before their loop filters
    /// (dav2d `--inloopfilters none`), which are then not applied.
    pub fn skip_loop_filters(&mut self) {
        with_core!(&mut self.instance, core => core.skip_loop_filters());
    }

    /// Diagnosis: also output each frame not shown when decoded, when it
    /// is decoded (dav2d `--outputinvisible 1`), to hold hidden frames
    /// against dav2d.
    pub fn output_invisible_frames(&mut self) {
        with_core!(&mut self.instance, core => core.output_invisible_frames());
    }

    /// Send the general block layer's trace (dav2d's `DEBUG_BLOCK_INFO`
    /// lines, `tools/dev/av2_foreign/block_trace.py`) to `sink`.
    pub fn trace_blocks(&mut self, sink: BlockSink) {
        with_core!(&mut self.instance, core => core.trace_blocks(sink));
    }
}

/// The payload of the first sequence header OBU of `unit`, if it has one
/// before its first frame OBU (a sequence header follows the temporal
/// delimiter).
fn sequence_header(unit: &[u8]) -> Result<Option<&[u8]>> {
    let mut remaining = unit;
    while !remaining.is_empty() {
        let (obu, rest) = obu::parse(remaining)?;
        match obu.header.obu_type {
            other_obu::SEQUENCE_HEADER => return Ok(Some(obu.payload)),
            kind if super::frame_header::obu_type::is_frame(kind) => return Ok(None),
            _ => remaining = rest,
        }
    }
    Ok(None)
}
