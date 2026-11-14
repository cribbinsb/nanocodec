//! Per-receiver VP9 spatial-SVC stream extraction: what a selective
//! forwarding relay sends one receiver of an L2T*/L3T* or K-SVC (`_KEY`)
//! stream. VP9 carries no layer ids (§6.2), so the extractor follows each
//! superframe's frames (Annex B §B.1–§B.3) through the eight reference slots
//! (`refresh_frame_flags`, `ref_frame_idx`, §6.2 / §7.2) and keeps, per
//! temporal unit, the frames the target layer's frame reaches: `[S0..=Sk]`
//! in full SVC, and in K-SVC `[S0..=Sk]` on key units and `[Sk]` alone (with
//! its `show_existing_frame`) otherwise. `nanocodec extract
//! --operating-point` and `Decoder::set_operating_point` use it.

use nc_common::decode::{BitReader, DecodeError, Result};

use crate::superframe::{
    FRAME_MARKER, REFS_PER_FRAME, SYNC_CODE, pack_coded_layers, read_color_config_profile0,
    split_superframe, trim_escape_padding,
};

/// Maximum spatial operating point index for VP9 (`0` = highest spatial layer,
/// `1` = middle of 3 or base of 2, `2` = lowest of 3).
pub const MAX_VP9_OPERATING_POINT: u8 = 2;

/// Target spatial view selected on [`Vp9StreamExtractor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp9ExtractTarget {
    /// Operating point `0..=2` (`0` = highest spatial layer, `1` = one spatial
    /// step down, `2` = two spatial steps down).
    OperatingPoint(u8),
    /// Lowest spatial layer (`spatial_id = 0`) regardless of stream layer count.
    BaseOnly,
}

#[derive(Debug, Clone, Copy)]
struct FramePrefix {
    show_existing_slot: Option<u8>,
    key_frame: bool,
    show_frame: bool,
    refresh_mask: u8,
    needed_mask: u8,
    size_ref_slot: Option<u8>,
    explicit_size: Option<(usize, usize)>,
}

fn parse_frame_prefix(frame: &[u8]) -> Result<FramePrefix> {
    let mut bits = BitReader::new(frame);
    if bits.read_bits(2)? != FRAME_MARKER {
        return Err(DecodeError::Malformed("VP9 frame marker"));
    }
    let profile = u8::from(bits.read_bit()?) | (u8::from(bits.read_bit()?) << 1);
    if profile == 3 && bits.read_bit()? {
        return Err(DecodeError::Unsupported("VP9 reserved profile"));
    }
    if profile != 0 {
        return Err(DecodeError::Unsupported("VP9 profile other than 0"));
    }
    if bits.read_bit()? {
        let slot = bits.read_bits(3)? as u8;
        return Ok(FramePrefix {
            show_existing_slot: Some(slot),
            key_frame: false,
            show_frame: true,
            refresh_mask: 0,
            needed_mask: 1u8 << slot,
            size_ref_slot: Some(slot),
            explicit_size: None,
        });
    }
    let key_frame = !bits.read_bit()?;
    let show_frame = bits.read_bit()?;
    let error_resilient = bits.read_bit()?;
    if key_frame {
        if bits.read_bits(24)? != SYNC_CODE {
            return Err(DecodeError::Malformed("VP9 sync code"));
        }
        read_color_config_profile0(&mut bits)?;
        let w = bits.read_bits(16)? as usize + 1;
        let h = bits.read_bits(16)? as usize + 1;
        return Ok(FramePrefix {
            show_existing_slot: None,
            key_frame: true,
            show_frame,
            refresh_mask: u8::MAX,
            needed_mask: 0,
            size_ref_slot: None,
            explicit_size: Some((w, h)),
        });
    }
    parse_inter_prefix(&mut bits, show_frame, error_resilient)
}

fn parse_inter_prefix(
    bits: &mut BitReader<'_>,
    show_frame: bool,
    error_resilient: bool,
) -> Result<FramePrefix> {
    let intra_only = !show_frame && bits.read_bit()?;
    if !error_resilient {
        let _reset_ctx = bits.read_bits(2)?;
    }
    if intra_only {
        if bits.read_bits(24)? != SYNC_CODE {
            return Err(DecodeError::Malformed("VP9 intra-only sync code"));
        }
        let refresh_mask = bits.read_bits(8)? as u8;
        let w = bits.read_bits(16)? as usize + 1;
        let h = bits.read_bits(16)? as usize + 1;
        return Ok(FramePrefix {
            show_existing_slot: None,
            key_frame: false,
            show_frame,
            refresh_mask,
            needed_mask: 0,
            size_ref_slot: None,
            explicit_size: Some((w, h)),
        });
    }
    let refresh_mask = bits.read_bits(8)? as u8;
    let mut ref_slots = [0u8; REFS_PER_FRAME];
    let mut needed_mask = 0u8;
    for slot in &mut ref_slots {
        *slot = bits.read_bits(3)? as u8;
        let _sign_bias = bits.read_bit()?;
        needed_mask |= 1u8 << *slot;
    }
    let mut size_ref_slot = None;
    for &slot in &ref_slots {
        if bits.read_bit()? {
            size_ref_slot = Some(slot);
            break;
        }
    }
    let explicit_size = if size_ref_slot.is_none() {
        let w = bits.read_bits(16)? as usize + 1;
        let h = bits.read_bits(16)? as usize + 1;
        Some((w, h))
    } else {
        None
    };
    Ok(FramePrefix {
        show_existing_slot: None,
        key_frame: false,
        show_frame,
        refresh_mask,
        needed_mask,
        size_ref_slot,
        explicit_size,
    })
}

struct ScannedFrame<'a> {
    raw: &'a [u8],
    prefix: FramePrefix,
    uid: u64,
    size: Option<(usize, usize)>,
}

/// Extracted VP9 packet bytes paired with the shown frame's `(width, height)`.
pub type ExtractedVp9Packet = (Vec<u8>, Option<(usize, usize)>);

/// Stateful VP9 Annex B superframe selective-forwarding extractor (`VP9 §6.2`,
/// `Annex B §B.1–§B.3`).
///
/// Tracks the 8 DPB reference slots across temporal units and walks backward
/// slot reachability from the target spatial layer:
/// - In full SVC (`L2T*`, `L3T*`), retains `[S0..=Sk]` on every temporal unit.
/// - In K-SVC (`L2T*_KEY`, `L3T*_KEY`), retains `[S0..=Sk]` on key units and
///   `[Sk, show_existing_frame]` alone on non-key units (and standalone repairs),
///   omitting non-key units where `Sk` was dropped.
#[derive(Debug, Clone)]
pub struct Vp9StreamExtractor {
    target: Vp9ExtractTarget,
    num_layers: Option<u8>,
    layer_sizes: Vec<(usize, usize)>,
    next_uid: u64,
    stream_uids: [Option<u64>; 8],
    kept_uids: [Option<u64>; 8],
    slot_sizes: [Option<(usize, usize)>; 8],
    stripped_lower: bool,
}

impl Vp9StreamExtractor {
    /// Extractor for spatial `operating_point` (`0..=2`).
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::Unsupported`] if `operating_point > MAX_VP9_OPERATING_POINT`.
    pub fn new(operating_point: u8) -> Result<Self> {
        Self::with_target(Vp9ExtractTarget::OperatingPoint(operating_point))
    }

    /// Extractor targeting the lowest spatial layer (`spatial_id = 0`).
    #[must_use]
    pub fn base_only() -> Self {
        Self {
            target: Vp9ExtractTarget::BaseOnly,
            num_layers: None,
            layer_sizes: Vec::new(),
            next_uid: 1,
            stream_uids: [None; 8],
            kept_uids: [None; 8],
            slot_sizes: [None; 8],
            stripped_lower: false,
        }
    }

    /// Extractor for `target`.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::Unsupported`] if an `OperatingPoint` exceeds `2`.
    pub fn with_target(target: Vp9ExtractTarget) -> Result<Self> {
        if let Vp9ExtractTarget::OperatingPoint(op) = target
            && op > MAX_VP9_OPERATING_POINT
        {
            return Err(DecodeError::Unsupported("VP9 operating point out of range"));
        }
        Ok(Self {
            target,
            ..Self::base_only()
        })
    }

    /// Visible `(width, height)` per spatial layer observed on the latest key unit.
    #[must_use]
    pub fn layer_sizes(&self) -> &[(usize, usize)] {
        &self.layer_sizes
    }

    /// Advance stream DPB tracking across a packet dropped by temporal filtering.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] if any header in `packet` is malformed.
    pub fn skip_packet(&mut self, packet: &[u8]) -> Result<()> {
        let _ = self.scan_packet(packet)?;
        Ok(())
    }

    fn scan_packet<'a>(&mut self, packet: &'a [u8]) -> Result<(Vec<ScannedFrame<'a>>, Option<u8>)> {
        let raw_frames =
            split_superframe(packet).unwrap_or_else(|| vec![trim_escape_padding(packet)]);
        let mut coded = Vec::with_capacity(raw_frames.len());
        let mut lone_sef_slot = None;
        for raw in raw_frames {
            let prefix = parse_frame_prefix(raw)?;
            if let Some(slot) = prefix.show_existing_slot {
                lone_sef_slot = Some(slot);
                continue;
            }
            let size = prefix.explicit_size.or_else(|| {
                prefix
                    .size_ref_slot
                    .and_then(|s| self.slot_sizes[usize::from(s)])
            });
            let uid = self.next_uid;
            self.next_uid += 1;
            for s in 0..8 {
                if (prefix.refresh_mask & (1u8 << s)) != 0 {
                    self.stream_uids[s] = Some(uid);
                    if let Some(sz) = size {
                        self.slot_sizes[s] = Some(sz);
                    }
                }
            }
            coded.push(ScannedFrame {
                raw,
                prefix,
                uid,
                size,
            });
        }
        Ok((coded, lone_sef_slot))
    }

    fn resolve_target_sid(&self, coded_len: usize, override_sid: Option<u8>) -> Result<u8> {
        if let Some(sid) = override_sid {
            return Ok(sid);
        }
        let n = self.num_layers.unwrap_or(coded_len as u8).max(1);
        match self.target {
            Vp9ExtractTarget::BaseOnly => Ok(0),
            Vp9ExtractTarget::OperatingPoint(op) => {
                if op >= n {
                    return Err(DecodeError::Unsupported("VP9 operating point out of range"));
                }
                Ok(n - 1 - op)
            }
        }
    }

    fn select_top_index(&self, coded: &[ScannedFrame<'_>], target_sid: u8) -> Option<usize> {
        let distinct_sizes = self
            .num_layers
            .is_some_and(|n| self.layer_sizes.len() == usize::from(n))
            && self.layer_sizes.windows(2).all(|w| w[0] != w[1]);
        if distinct_sizes {
            let mut sids = Vec::with_capacity(coded.len());
            for c in coded {
                let sid = c
                    .size
                    .and_then(|sz| self.layer_sizes.iter().position(|&lsz| lsz == sz))
                    .map(|p| p as u8);
                sids.push(sid);
            }
            if sids.iter().all(Option::is_some) {
                if let Some(idx) = sids.iter().position(|&s| s == Some(target_sid)) {
                    return Some(idx);
                }
                if self.stripped_lower {
                    return None;
                }
                return sids
                    .iter()
                    .rposition(|&s| s.is_some_and(|sid| sid < target_sid));
            }
        }
        if usize::from(target_sid) < coded.len() {
            Some(usize::from(target_sid))
        } else if self.stripped_lower {
            None
        } else {
            Some(coded.len() - 1)
        }
    }

    /// Filter `packet` to the frames required by this extractor's target view.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] on malformed headers or an out-of-range operating point.
    pub fn extract_packet(&mut self, packet: &[u8]) -> Result<Option<ExtractedVp9Packet>> {
        self.extract_inner(packet, None)
    }

    /// Filter `packet` to a specific `target_spatial_id` (used when an external
    /// layer sidecar has already selected the temporal unit and target spatial id).
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] on malformed headers.
    pub fn extract_spatial_layer(
        &mut self,
        packet: &[u8],
        target_spatial_id: u8,
    ) -> Result<Option<ExtractedVp9Packet>> {
        self.extract_inner(packet, Some(target_spatial_id))
    }

    fn extract_inner(
        &mut self,
        packet: &[u8],
        override_sid: Option<u8>,
    ) -> Result<Option<ExtractedVp9Packet>> {
        let stream_uids_before = self.stream_uids;
        let (coded, lone_sef_slot) = self.scan_packet(packet)?;
        if coded.is_empty() {
            let Some(slot) = lone_sef_slot else {
                return Ok(None);
            };
            let idx = usize::from(slot);
            if self.kept_uids[idx] != stream_uids_before[idx] {
                return Ok(None);
            }
            return Ok(Some((packet.to_vec(), self.slot_sizes[idx])));
        }
        if coded[0].prefix.key_frame {
            self.num_layers = Some(coded.len() as u8);
            self.stripped_lower = false;
            let sizes: Vec<(usize, usize)> = coded.iter().filter_map(|c| c.size).collect();
            if sizes.len() == coded.len() {
                self.layer_sizes = sizes;
            }
        }
        let target_sid = self.resolve_target_sid(coded.len(), override_sid)?;
        let Some(top_idx) = self.select_top_index(&coded, target_sid) else {
            return Ok(None);
        };

        // Stripping unreferenced lower layers is safe whenever the selected
        // frame is hidden (`show_frame = 0`, displayed via `show_existing_frame`
        // so `last_show_frame` stays `0`, covering every non-top spatial layer
        // and all K-SVC layers). A shown top frame (`inter_layer = 0`) keeps
        // its preceding hidden layer so `use_prev_frame_mvs = 0` (`VP9 §7.2`).
        let can_strip_lower = !coded[top_idx].prefix.show_frame;
        let mut keep = vec![!can_strip_lower; top_idx + 1];
        keep[top_idx] = true;
        let needed = if can_strip_lower {
            let mut mask = coded[top_idx].prefix.needed_mask;
            for i in (0..top_idx).rev() {
                let refreshed = coded[i].prefix.refresh_mask;
                if (refreshed & mask) != 0 {
                    keep[i] = true;
                    mask = (mask & !refreshed) | coded[i].prefix.needed_mask;
                }
            }
            mask
        } else {
            coded.iter().take(top_idx + 1).rev().fold(0u8, |acc, c| {
                (acc & !c.prefix.refresh_mask) | c.prefix.needed_mask
            })
        };
        let missing_ref = (0..8).any(|slot| {
            (needed & (1u8 << slot)) != 0 && self.kept_uids[slot] != stream_uids_before[slot]
        });
        if missing_ref {
            return Ok(None);
        }
        if keep.iter().any(|&k| !k) {
            self.stripped_lower = true;
        }
        let mut kept_slices = Vec::with_capacity(top_idx + 1);
        for (c, &k) in coded[..=top_idx].iter().zip(&keep) {
            if k {
                for (slot, dst) in self.kept_uids.iter_mut().enumerate() {
                    if (c.prefix.refresh_mask & (1u8 << slot)) != 0 {
                        *dst = Some(c.uid);
                    }
                }
                kept_slices.push(c.raw);
            }
        }
        let out = if top_idx + 1 == coded.len() && keep.iter().all(|&k| k) {
            packet.to_vec()
        } else {
            pack_coded_layers(&kept_slices)?
        };
        Ok(Some((out, coded[top_idx].size)))
    }
}
