//! AV2 deblocking filter (AVM v1.0.0 `av2/common/av2_loopfilter.c`): the
//! frame-header parameters and their syntax (`setup_loopfilter` in
//! `av2/decoder/decodeframe.c`, its writer in `av2/encoder/bitstream.c`),
//! the encoder's choice of them, and the picture filter shared by the
//! encoder reconstruction and our decoder.
//!
//! Each of four channels (luma vertical edges, luma horizontal edges, U, V)
//! is on or off and, when on, carries a small qindex offset that strengthens
//! or weakens its thresholds. Lossless frames have no deblocking syntax.
//! Thresholds follow the blocks' qindex ([`thresholds`]); which edges are
//! filtered follows the blocks' transforms ([`edges`]); the per-segment
//! filter is nc-dsp's `av2_deblock_edge`.

mod edges;
mod thresholds;

pub(crate) use edges::{BlockEdges, EdgeGrid, EdgeMap, filter_picture};

use nc_common::decode::{BitReader, DecodeError, Result};
use nc_common::{BitWriter, PlaneIndex};

/// `df_par_bits_minus2` of our sequence header (zero): two-bit offsets.
const DELTA_BITS: u32 = 2;
/// Offsets are coded as `delta + DELTA_OFFSET` (`df_par_offset`).
const DELTA_OFFSET: i8 = 1 << (DELTA_BITS - 1);
/// Smallest channel offset, in `DF_DELTA_SCALE` qindex units.
pub const MIN_DELTA: i8 = -DELTA_OFFSET;
/// Largest channel offset.
pub const MAX_DELTA: i8 = DELTA_OFFSET - 1;

/// One independently controlled part of the filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterChannel {
    /// Luma vertical edges (`apply_deblocking_filter[0]`).
    LumaVertical,
    /// Luma horizontal edges (`apply_deblocking_filter[1]`).
    LumaHorizontal,
    /// Both directions of U (`apply_deblocking_filter_u`).
    U,
    /// Both directions of V (`apply_deblocking_filter_v`).
    V,
}

impl FilterChannel {
    /// Every channel, in syntax order.
    pub const ALL: [Self; 4] = [Self::LumaVertical, Self::LumaHorizontal, Self::U, Self::V];

    /// The channel filtering `plane`'s edges of direction `vertical`.
    #[must_use]
    pub fn of(plane: PlaneIndex, vertical: bool) -> Self {
        match plane {
            PlaneIndex::Y if vertical => Self::LumaVertical,
            PlaneIndex::Y => Self::LumaHorizontal,
            PlaneIndex::U => Self::U,
            PlaneIndex::V => Self::V,
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// The frame's deblocking parameters: per channel, off (`None`) or on with
/// a qindex offset in `MIN_DELTA..=MAX_DELTA` units of eight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeblockParams {
    deltas: [Option<i8>; 4],
}

impl DeblockParams {
    /// No filtering.
    pub const OFF: Self = Self { deltas: [None; 4] };

    /// Parameters with the given per-channel settings (in
    /// [`FilterChannel::ALL`] order).
    ///
    /// # Errors
    ///
    /// Rejects offsets outside the two-bit range and chroma filtering
    /// without luma filtering (the syntax cannot express it).
    pub fn new(deltas: [Option<i8>; 4]) -> Result<Self> {
        let params = Self { deltas };
        if deltas
            .iter()
            .flatten()
            .any(|d| !(MIN_DELTA..=MAX_DELTA).contains(d))
        {
            return Err(DecodeError::Malformed("AV2 deblocking offset out of range"));
        }
        if !params.luma_on()
            && (params.delta(FilterChannel::U).is_some()
                || params.delta(FilterChannel::V).is_some())
        {
            return Err(DecodeError::Malformed("AV2 chroma deblocking without luma"));
        }
        Ok(params)
    }

    /// The offset of `channel`, or `None` if it is off.
    #[must_use]
    pub fn delta(self, channel: FilterChannel) -> Option<i8> {
        self.deltas[channel.index()]
    }

    /// These parameters with both chroma channels off.
    #[must_use]
    pub fn luma_only(self) -> Self {
        let mut deltas = self.deltas;
        deltas[FilterChannel::U.index()] = None;
        deltas[FilterChannel::V.index()] = None;
        Self { deltas }
    }

    /// Whether any edge is filtered.
    #[must_use]
    pub fn is_on(self) -> bool {
        self.luma_on()
    }

    fn luma_on(self) -> bool {
        self.delta(FilterChannel::LumaVertical).is_some()
            || self.delta(FilterChannel::LumaHorizontal).is_some()
    }

    /// `setup_loopfilter` syntax of a lossy frame (no multi-frame header,
    /// no TIP, no sub-PU filtering).
    pub fn write(self, bits: &mut BitWriter) {
        bits.put_bit(self.delta(FilterChannel::LumaVertical).is_some());
        bits.put_bit(self.delta(FilterChannel::LumaHorizontal).is_some());
        if self.luma_on() {
            bits.put_bit(self.delta(FilterChannel::U).is_some());
            bits.put_bit(self.delta(FilterChannel::V).is_some());
        }
        for channel in FilterChannel::ALL {
            if let Some(delta) = self.delta(channel) {
                let implied = self.implied_delta(channel);
                bits.put_bit(delta != implied);
                if delta != implied {
                    bits.put_bits((delta + DELTA_OFFSET) as u32, DELTA_BITS);
                }
            }
        }
    }

    /// Parse what [`DeblockParams::write`] writes.
    ///
    /// # Errors
    ///
    /// Propagates truncation.
    pub fn read(bits: &mut BitReader<'_>) -> Result<Self> {
        let mut params = Self::OFF;
        let luma = [bits.read_bit()?, bits.read_bit()?];
        let chroma = if luma.contains(&true) {
            [bits.read_bit()?, bits.read_bit()?]
        } else {
            [false; 2]
        };
        let on = [luma[0], luma[1], chroma[0], chroma[1]];
        for (channel, on) in FilterChannel::ALL.into_iter().zip(on) {
            if on {
                let implied = params.implied_delta(channel);
                let delta = if bits.read_bit()? {
                    bits.read_bits(DELTA_BITS)? as i8 - DELTA_OFFSET
                } else {
                    implied
                };
                params.deltas[channel.index()] = Some(delta);
            }
        }
        Ok(params)
    }

    /// The offset a channel takes when its delta flag is zero: horizontal
    /// luma repeats vertical luma (zero when that is off), others zero.
    fn implied_delta(self, channel: FilterChannel) -> i8 {
        match channel {
            FilterChannel::LumaHorizontal => self.delta(FilterChannel::LumaVertical).unwrap_or(0),
            _ => 0,
        }
    }

    /// AVM's `LPF_PICK_FROM_Q` choice for a frame at `qindex`: every
    /// channel on with zero offset, except that a channel whose thresholds
    /// are zero at the frame qindex is switched off (as AVM's search does),
    /// since it would filter nothing. Lossless frames are never filtered.
    #[must_use]
    pub fn from_qindex(qindex: u8) -> Self {
        let active = qindex != 0
            && thresholds::q_threshold(i32::from(qindex)) != 0
            && thresholds::side_threshold(i32::from(qindex)) != 0;
        if active {
            Self {
                deltas: [Some(0); 4],
            }
        } else {
            Self::OFF
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syntax_round_trips_every_channel_combination() {
        let choices = [None, Some(MIN_DELTA), Some(-1), Some(0), Some(MAX_DELTA)];
        for &lv in &choices {
            for &lh in &choices {
                for &u in &choices {
                    for &v in &choices {
                        let Ok(params) = DeblockParams::new([lv, lh, u, v]) else {
                            assert!(lv.is_none() && lh.is_none());
                            continue;
                        };
                        let mut writer = BitWriter::new();
                        params.write(&mut writer);
                        writer.byte_align_zero();
                        let bytes = writer.into_bytes();
                        let mut reader = BitReader::new(&bytes);
                        assert_eq!(DeblockParams::read(&mut reader).unwrap(), params);
                    }
                }
            }
        }
    }

    #[test]
    fn off_is_two_zero_bits_and_default_choice_follows_thresholds() {
        let mut writer = BitWriter::new();
        DeblockParams::OFF.write(&mut writer);
        assert_eq!(writer.bit_len(), 2);
        assert_eq!(DeblockParams::from_qindex(0), DeblockParams::OFF);
        assert_eq!(DeblockParams::from_qindex(60), DeblockParams::OFF);
        assert!(DeblockParams::from_qindex(160).is_on());
        assert!(DeblockParams::new([Some(2), None, None, None]).is_err());
        assert!(DeblockParams::new([None, None, Some(0), None]).is_err());
    }
}
