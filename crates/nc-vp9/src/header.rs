//! VP9 Profile 0 frame headers: the uncompressed header (VP9 §6.2 / §7.2)
//! and the compressed header (§6.3) as the encoder writes them. Field order
//! follows libvpx `vp9_decodeframe.c` `read_uncompressed_header` /
//! `read_compressed_header` (BSD-3-Clause; see `THIRD_PARTY.md`); the
//! decoder parses every Profile 0 header (`decoder/header.rs`).
//!
//! Written subset: 8-bit 4:2:0 BT.601 studio range, backward probability
//! adaptation into frame context 0 (except error-resilient frames, which
//! decode from default probabilities without previous-frame vectors), one interpolation filter per frame (any of the four) or per block (switchable), one reference (all three logical references name one slot),
//! one tile, segmentation with an explicit map and per-segment filter
//! levels only, no loop-filter deltas, per-block transform
//! sizes (`TX_MODE_SELECT`; lossless: 4x4), and a compressed header whose
//! only probability updates are coefficient models.

use nc_common::BitWriter;

use crate::bool_coder::BoolWriter;
use crate::probs::CoefUpdates;
use crate::segmentation::Segmentation;
use crate::subexp::write_update;
use crate::tables::BandProbs;

const FRAME_MARKER: u32 = 2;
const SYNC_CODE: u32 = 0x49_83_42;
/// `color_space` codes (VP9 §7.2, Table 7-1) of the colour matrices the
/// encoder signals.
mod color_space {
    /// `CS_BT_601` (BT.470 B/G; matrix code point 5), the default.
    pub(super) const BT_601: u32 = 1;
    /// `CS_BT_709`.
    pub(super) const BT_709: u32 = 2;
    /// `CS_SMPTE_170` (BT.601 525-line; matrix code point 6).
    pub(super) const SMPTE_170: u32 = 3;
}

/// A key frame's `color_space` and `color_range` (VP9 §6.2, profile 0):
/// signalling only, the decoding process ignores them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ColourSpace {
    space: u32,
    full_range: bool,
}

impl Default for ColourSpace {
    /// BT.601, studio range: what streams without a colour description say.
    fn default() -> Self {
        Self {
            space: color_space::BT_601,
            full_range: false,
        }
    }
}

impl ColourSpace {
    /// The VP9 form of `colour`, or `None` for a matrix VP9 cannot name in
    /// this subset (anything but BT.601 and BT.709). Primaries and transfer
    /// have no syntax of their own: `color_space` implies them.
    pub(crate) fn of(colour: nc_common::ColourInfo) -> Option<Self> {
        use nc_common::colour::matrix;
        let space = match colour.description.map(|d| d.matrix) {
            None | Some(matrix::BT601) => color_space::BT_601,
            Some(matrix::BT709) => color_space::BT_709,
            Some(matrix::BT601_525) => color_space::SMPTE_170,
            Some(_) => return None,
        };
        Some(Self {
            space,
            full_range: colour.full_range,
        })
    }
}
/// Logical references of an inter frame (LAST, GOLDEN, ALTREF).
const REFS_PER_FRAME: usize = 3;
/// `literal_to_filter`: the frame filter's 2-bit code.
const FILTER_LITERALS: [nc_dsp::Vp9Filter; 4] = [
    nc_dsp::Vp9Filter::Smooth,
    nc_dsp::Vp9Filter::Regular,
    nc_dsp::Vp9Filter::Sharp,
    nc_dsp::Vp9Filter::Bilinear,
];
/// Every slot: key frames refresh all eight.
pub(crate) const ALL_SLOTS: u8 = 0xff;
/// Largest picture side this one-tile subset codes.
pub(crate) const MAX_DIMENSION: usize = 4096;
/// Probability of every "update" flag (`DIFF_UPDATE_PROB`, `MV_UPDATE_PROB`).
const UPDATE_PROB: u8 = 252;
/// Probability-update groups of an inter frame's compressed header, in
/// order: inter mode (7x3), intra/inter (4), single reference (5x2),
/// y mode (4x9), partition (16x3), motion vectors (joints 3, two
/// components of 22 + 9).
const INTER_UPDATE_FLAGS: usize = 21 + 4 + 10 + 36 + 48 + (3 + 2 * 22 + 2 * 9);
/// Switchable filter probability update flags (4 contexts x 2 nodes).
const SWITCHABLE_UPDATE_FLAGS: usize = 8;
/// Skip probability update flags (both frame types).
const SKIP_UPDATE_FLAGS: usize = 3;
/// `read_tx_mode_probs` flags: 2 contexts x (1 + 2 + 3) nodes.
const TX_PROB_UPDATE_FLAGS: usize = 12;
/// Transform sizes whose coefficient models may be updated.
const TX_SIZES: usize = 4;

/// Fields of one coded frame.
#[allow(
    clippy::struct_excessive_bools,
    reason = "VP9 uncompressed header flags"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameHeader {
    pub key: bool,
    /// `show_frame` (`VP9 §6.2`): `false` on `S0` when `S1` follows in the
    /// same Annex B superframe, `true` otherwise.
    pub show_frame: bool,
    /// `error_resilient_mode`: decodable without the previous frame's
    /// vectors or adapted probabilities (default contexts, no adaptation).
    pub error_resilient: bool,
    /// `frame_context_idx`: the probability context the frame starts from
    /// (VP9 §7.2; the decoder reads context 0 on key and error-resilient
    /// frames whatever is written).
    pub context_index: u8,
    /// `refresh_frame_context` of a frame that is not error resilient: the
    /// frame's backward-adapted probabilities replace its context. A frame
    /// that does not refresh codes `frame_parallel_decoding_mode` = 1 (the
    /// adaptation would be discarded).
    pub refresh_context: bool,
    pub width: usize,
    pub height: usize,
    /// Slots `LAST_FRAME`, `GOLDEN_FRAME` and `ALTREF_FRAME` read; ignored on
    /// key frames.
    pub ref_slots: [u8; REFS_PER_FRAME],
    /// The reference (`0..3` in `ref_slots`) an inter frame takes its size
    /// from (`found_ref`, `VP9 §6.2.6`), or `None` to write it explicitly
    /// (`width - 1`, `height - 1`): a spatial layer's frame whose
    /// references all have the other layer's size.
    pub size_from_ref: Option<usize>,
    /// Slots this frame refreshes (all on key frames).
    pub refresh: u8,
    pub filter_level: u8,
    /// `sharpness_level` of the loop filter (0..=7).
    pub sharpness: u8,
    /// The interpolation filter of every inter block (inter frames),
    /// unless `switchable` lets each block code its own.
    pub interp_filter: Option<nc_dsp::Vp9Filter>,
    /// `allow_high_precision_mv` (inter frames).
    pub allow_hp: bool,
    pub base_q_idx: u8,
    /// `segmentation_params`.
    pub segmentation: Segmentation,
    /// `delta_q_uv_dc` and `delta_q_uv_ac` (equal in this subset; -15..=15).
    pub chroma_delta_q: i8,
    /// Written on key frames.
    pub colour: ColourSpace,
}

impl FrameHeader {
    /// Lossless frames (qindex 0, no deltas) use the WHT and 4x4 transforms.
    pub(crate) fn lossless(&self) -> bool {
        self.base_q_idx == 0 && self.chroma_delta_q == 0
    }

    /// Quantizer steps of `plane` at the frame quantizer.
    pub(crate) fn steps(&self, plane: nc_common::PlaneIndex) -> crate::transform::Steps {
        self.steps_at(self.base_q_idx, plane)
    }

    /// Quantizer steps of `plane` for a block coded at `qindex` (its
    /// segment's, VP9 §8.6.1 `get_qindex`); the chroma deltas apply on top.
    pub(crate) fn steps_at(
        &self,
        qindex: u8,
        plane: nc_common::PlaneIndex,
    ) -> crate::transform::Steps {
        let delta = if plane == nc_common::PlaneIndex::Y {
            0
        } else {
            self.chroma_delta_q
        };
        crate::transform::Steps::new(qindex, delta, delta)
    }

    /// Write the uncompressed header for a compressed header of
    /// `compressed_size` bytes.
    pub(crate) fn write(&self, compressed_size: usize) -> Vec<u8> {
        let mut bits = BitWriter::new();
        bits.put_bits(FRAME_MARKER, 2);
        bits.put_bits(0, 2); // profile 0
        bits.put_bit(false); // show_existing_frame
        bits.put_bit(!self.key); // frame_type
        bits.put_bit(self.show_frame); // show_frame
        bits.put_bit(self.error_resilient);
        if self.key {
            bits.put_bits(SYNC_CODE, 24);
            bits.put_bits(self.colour.space, 3);
            bits.put_bit(self.colour.full_range); // color_range
            write_size(&mut bits, self.width, self.height);
        } else {
            if !self.show_frame {
                bits.put_bit(false); // intra_only = 0 (VP9 §6.2)
            }
            if !self.error_resilient {
                bits.put_bits(0, 2); // reset_frame_context: none
            }
            bits.put_bits(u32::from(self.refresh), 8);
            for &slot in &self.ref_slots {
                bits.put_bits(u32::from(slot), 3);
                bits.put_bit(false); // sign bias: no compound prediction
            }
            // `frame_size_with_refs` (VP9 §6.2.6): `found_ref` until the
            // reference with the frame's size, else the size itself.
            if let Some(reference) = self.size_from_ref {
                for _ in 0..reference {
                    bits.put_bit(false);
                }
                bits.put_bit(true);
                bits.put_bit(false); // render size equals frame size
            } else {
                for _ in 0..REFS_PER_FRAME {
                    bits.put_bit(false);
                }
                write_size(&mut bits, self.width, self.height);
            }
            bits.put_bit(self.allow_hp);
            bits.put_bit(self.interp_filter.is_none()); // is_filter_switchable
            if let Some(filter) = self.interp_filter {
                let literal = FILTER_LITERALS
                    .iter()
                    .position(|&f| f == filter)
                    .expect("every filter has a literal");
                bits.put_bits(literal as u32, 2);
            }
        }
        if !self.error_resilient {
            bits.put_bit(self.refresh_context);
            bits.put_bit(!self.refresh_context); // frame_parallel_decoding_mode
        }
        bits.put_bits(u32::from(self.context_index), 2);
        bits.put_bits(u32::from(self.filter_level), 6);
        bits.put_bits(u32::from(self.sharpness), 3);
        bits.put_bit(false); // mode_ref_delta_enabled
        bits.put_bits(u32::from(self.base_q_idx), 8);
        bits.put_bit(false); // delta_q_y_dc
        for _ in 0..2 {
            // delta_q_uv_dc, delta_q_uv_ac
            let delta = self.chroma_delta_q;
            bits.put_bit(delta != 0);
            if delta != 0 {
                bits.put_bits(u32::from(delta.unsigned_abs()), 4);
                bits.put_bit(delta < 0);
            }
        }
        self.segmentation.write(&mut bits);
        if has_tile_column_bit(self.width) {
            bits.put_bit(false); // one tile column
        }
        bits.put_bit(false); // one tile row
        bits.put_bits(compressed_size as u32, 16);
        bits.byte_align_zero();
        bits.into_bytes()
    }

    /// "No update" flags of an inter frame's mode/vector probabilities
    /// (all of them, since none is updated): switchable frames also have
    /// the filter probabilities (`read_switchable_interp_probs`).
    fn inter_update_flags(&self) -> usize {
        match (self.key, self.interp_filter.is_none()) {
            (true, _) => 0,
            (false, false) => INTER_UPDATE_FLAGS,
            (false, true) => INTER_UPDATE_FLAGS + SWITCHABLE_UPDATE_FLAGS,
        }
    }

    /// The compressed header: transform mode and "no update" flags.
    /// `updates` are the coefficient tables chosen for the frame against
    /// the context's `coef` (forward probability updates).
    #[cfg(test)]
    pub(crate) fn write_compressed(
        &self,
        coef: &[[[BandProbs; 2]; 2]; 4],
        updates: &CoefUpdates,
    ) -> Vec<u8> {
        self.write_compressed_with_modes(coef, updates, None)
    }

    /// The compressed header with optional forward mode/skip/partition/tx
    /// probability updates (`(old_mode_probs, new_mode_probs)`).
    pub(crate) fn write_compressed_with_modes(
        &self,
        coef: &[[[BandProbs; 2]; 2]; 4],
        updates: &CoefUpdates,
        mode_updates: Option<(
            &[u8; crate::recorder::MODE_SLOTS],
            &[u8; crate::recorder::MODE_SLOTS],
        )>,
    ) -> Vec<u8> {
        let mut writer = BoolWriter::new();
        if !self.lossless() {
            writer.write(true, 128);
            writer.write(true, 128); // tx_mode literal 3 (ALLOW_32X32) ...
            writer.write(true, 128); // ... plus one: TX_MODE_SELECT
            if let Some((old, new)) = mode_updates {
                for slot in 0..TX_PROB_UPDATE_FLAGS {
                    write_update(&mut writer, old[slot], new[slot]);
                }
            } else {
                for _ in 0..TX_PROB_UPDATE_FLAGS {
                    writer.write(false, UPDATE_PROB); // tx_probs: no update
                }
            }
        }
        let tables = coef.iter().zip(&updates.coef).zip(updates.sent);
        for ((old, new), sent) in tables.take(self.coefficient_sizes()) {
            writer.write(sent, 128);
            if sent {
                for_each_node(|p, r, b, c, node| {
                    write_update(&mut writer, old[p][r][b][c][node], new[p][r][b][c][node]);
                });
            }
        }
        if let Some((old, new)) = mode_updates {
            use crate::recorder::{
                INTER_MODE_SLOT, INTRA_INTER_SLOT, MODE_SLOTS, MV_HP_BASE_SLOT, MV_JOINT_SLOT,
                MV_UPDATE_LITERAL_BITS, SKIP_SLOT, SWITCHABLE_INTERP_SLOT,
            };
            // The slots follow the compressed header's order (VP9 §6.3):
            // skip, then on inter frames inter modes, switchable filters,
            // intra/inter through partitions, and the vector probabilities.
            let updates = |writer: &mut BoolWriter, slots: std::ops::Range<usize>| {
                for slot in slots {
                    write_update(writer, old[slot], new[slot]);
                }
            };
            updates(&mut writer, SKIP_SLOT..INTER_MODE_SLOT);
            if !self.key {
                updates(&mut writer, INTER_MODE_SLOT..SWITCHABLE_INTERP_SLOT);
                if self.interp_filter.is_none() {
                    updates(&mut writer, SWITCHABLE_INTERP_SLOT..INTRA_INTER_SLOT);
                }
                updates(&mut writer, INTRA_INTER_SLOT..MV_JOINT_SLOT);
                let mv_end = if self.allow_hp {
                    MODE_SLOTS
                } else {
                    MV_HP_BASE_SLOT
                };
                // `update_mv_prob` (VP9 §6.3.17): a flag, then 7 literal bits.
                for slot in MV_JOINT_SLOT..mv_end {
                    if old[slot] == new[slot] {
                        writer.write(false, UPDATE_PROB);
                    } else {
                        writer.write(true, UPDATE_PROB);
                        for bit in (0..MV_UPDATE_LITERAL_BITS).rev() {
                            writer.write(((new[slot] >> 1) >> bit) & 1 != 0, 128);
                        }
                    }
                }
            }
        } else {
            let flags = SKIP_UPDATE_FLAGS + self.inter_update_flags();
            for _ in 0..flags {
                writer.write(false, UPDATE_PROB);
            }
        }
        writer.finish()
    }

    /// Coefficient models present: up to the largest allowed transform.
    fn coefficient_sizes(&self) -> usize {
        if self.lossless() { 1 } else { TX_SIZES }
    }
}

/// Every updatable node of one transform size's coefficient table, in
/// bitstream order (`read_coef_probs_common`).
fn for_each_node(mut f: impl FnMut(usize, usize, usize, usize, usize)) {
    for p in 0..2 {
        for r in 0..2 {
            for b in 0..6 {
                for c in 0..if b == 0 { 3 } else { 6 } {
                    for node in 0..3 {
                        f(p, r, b, c, node);
                    }
                }
            }
        }
    }
}

fn write_size(bits: &mut BitWriter, width: usize, height: usize) {
    bits.put_bits(width as u32 - 1, 16);
    bits.put_bits(height as u32 - 1, 16);
    bits.put_bit(false); // render size equals frame size
}

/// Whether the tile-column increment bit is present: from 8 superblock
/// columns more than one tile column is allowed (`get_tile_n_bits`), and
/// a zero bit keeps one.
fn has_tile_column_bit(width: usize) -> bool {
    width.div_ceil(64) >= 8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segmentation::{SegmentFeatures, Segmentation};
    use nc_common::{ColourDescription, ColourInfo};

    /// Byte 4 of a key frame's uncompressed header: after the frame marker,
    /// profile, flags and sync code, `color_space` (3 bits) then
    /// `color_range`.
    fn colour_byte(colour: ColourSpace) -> u8 {
        let header = FrameHeader {
            key: true,
            show_frame: true,
            error_resilient: false,
            context_index: 0,
            refresh_context: true,
            width: 64,
            height: 64,
            ref_slots: [0; 3],
            size_from_ref: None,
            refresh: ALL_SLOTS,
            filter_level: 0,
            sharpness: 0,
            interp_filter: None,
            allow_hp: false,
            base_q_idx: 60,
            segmentation: Segmentation::disabled(SegmentFeatures::default()),
            chroma_delta_q: 0,
            colour,
        };
        header.write(1)[4]
    }

    #[test]
    fn key_frames_signal_the_colour_matrix_and_range() {
        let unsignalled = ColourSpace::of(ColourInfo::default()).unwrap();
        assert_eq!(unsignalled, ColourSpace::default());
        assert_eq!(colour_byte(unsignalled) >> 4, 0b0010); // BT.601, studio
        let hd_full = ColourInfo {
            description: Some(ColourDescription::BT709),
            full_range: true,
        };
        assert_eq!(colour_byte(ColourSpace::of(hd_full).unwrap()) >> 4, 0b0101);
        let sd = ColourInfo {
            description: Some(ColourDescription::BT601_525),
            full_range: false,
        };
        assert_eq!(colour_byte(ColourSpace::of(sd).unwrap()) >> 4, 0b0110);
        let identity = ColourInfo {
            description: Some(ColourDescription {
                matrix: nc_common::colour::matrix::IDENTITY,
                ..ColourDescription::BT709
            }),
            full_range: false,
        };
        assert_eq!(ColourSpace::of(identity), None);
    }
}
