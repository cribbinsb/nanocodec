//! What the own-stream fast path keeps so that a stream it cannot finish
//! can be decoded by the general path instead.
//!
//! The fast path is chosen when an SPS is byte-identical to one our encoder
//! writes, but a stream with such an SPS may still use a PPS, APS, picture
//! header, slice type or coding tool outside our encoder's subset. The fast
//! path then fails with `Unsupported`. Its references are in its own form,
//! so the general path cannot simply carry on: it decodes again, from the
//! parameter sets active at the last IDR picture and every unit the fast
//! path accepted since, and its outputs of the pictures the fast path
//! already showed are dropped (the fast path shows in decoding order and
//! codes no reordering, so they are the first ones out).
//!
//! The record holds the coded bytes since the last IDR picture, bounded by
//! [`MAX_RECORD_BYTES`]; a stream that codes more between IDR pictures
//! keeps the fast path's error instead.

/// Coded bytes kept between IDR pictures (a minute of 8 Mbit/s video).
const MAX_RECORD_BYTES: usize = 64 * 1024 * 1024;
/// `nal_unit_type` of the units the record keeps (H.266 Table 5).
const IDR_N_LP: u8 = 8;
const SPS_NUT: u8 = 15;
const PPS_NUT: u8 = 16;
const PREFIX_APS_NUT: u8 = 17;
const TRAIL_NUT: u8 = 0;
/// ALF APS ids (`aps_adaptation_parameter_set_id`: 0..=7).
const APS_IDS: usize = 8;

/// The fast path's accepted NAL units (each with its two-byte header).
#[derive(Debug, Default)]
pub(super) struct OwnRecord {
    /// The latest SPS, PPS and ALF APS by id: what an IDR picture starts from.
    sps: Option<Vec<u8>>,
    pps: Option<Vec<u8>>,
    aps: [Option<Vec<u8>>; APS_IDS],
    /// The parameter sets active at the last IDR picture, then every unit
    /// accepted since, in decoding order.
    since_key: Vec<Vec<u8>>,
    bytes: usize,
    /// Pictures shown since the last IDR picture.
    shown: usize,
    /// Whether the record outgrew its bound (no fallback until the next IDR).
    overflowed: bool,
    /// Set once the fast path has failed: an SPS like ours no longer
    /// selects it for this stream.
    disabled: bool,
}

impl OwnRecord {
    /// The record after a fallback: empty, and the fast path off.
    pub fn disabled_for_stream() -> Self {
        Self {
            disabled: true,
            ..Self::default()
        }
    }

    /// Whether the fast path is off for this stream.
    pub fn disabled(&self) -> bool {
        self.disabled
    }

    /// Keep the units of one call the fast path decoded without error.
    pub fn accepted(&mut self, units: &[&[u8]]) {
        for &unit in units {
            let kind = unit[1] >> 3;
            match kind {
                SPS_NUT => {
                    *self = Self {
                        sps: Some(unit.to_vec()),
                        ..Self::default()
                    };
                }
                PPS_NUT => {
                    self.pps = Some(unit.to_vec());
                    self.push(unit);
                }
                PREFIX_APS_NUT => {
                    // aps_params_type u(3), then aps_adaptation_parameter_set_id u(5).
                    if let Some(&byte) = unit.get(2) {
                        self.aps[usize::from(byte & 0x1f) % APS_IDS] = Some(unit.to_vec());
                    }
                    self.push(unit);
                }
                IDR_N_LP => {
                    let start = [&self.sps, &self.pps].into_iter().chain(&self.aps);
                    self.since_key = start.flatten().cloned().collect();
                    self.bytes = self.since_key.iter().map(Vec::len).sum();
                    self.overflowed = false;
                    self.shown = 1;
                    self.push(unit);
                }
                TRAIL_NUT => {
                    self.shown += 1;
                    self.push(unit);
                }
                _ => {}
            }
        }
    }

    fn push(&mut self, unit: &[u8]) {
        self.bytes += unit.len();
        if self.bytes > MAX_RECORD_BYTES {
            self.overflowed = true;
            self.since_key = Vec::new();
        }
        if !self.overflowed {
            self.since_key.push(unit.to_vec());
        }
    }

    /// The units to decode again, grouped one picture per group (each ends
    /// with its slice), and the number of pictures already shown; `None`
    /// when the record outgrew its bound.
    pub fn replay(&self) -> Option<(Vec<Vec<&[u8]>>, usize)> {
        if self.overflowed {
            return None;
        }
        let mut pictures = vec![Vec::new()];
        for unit in &self.since_key {
            pictures
                .last_mut()
                .expect("never empty")
                .push(unit.as_slice());
            if matches!(unit[1] >> 3, IDR_N_LP | TRAIL_NUT) {
                pictures.push(Vec::new());
            }
        }
        // Parameter sets after the last picture go with the failing call.
        Some((pictures, self.shown))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(kind: u8, payload: u8) -> Vec<u8> {
        vec![0, kind << 3 | 1, payload]
    }

    #[test]
    fn replay_starts_at_the_last_idr_picture() {
        let mut record = OwnRecord::default();
        let (sps, pps, aps) = (unit(SPS_NUT, 0), unit(PPS_NUT, 0), unit(PREFIX_APS_NUT, 3));
        let (idr, trail) = (unit(IDR_N_LP, 0), unit(TRAIL_NUT, 0));
        record.accepted(&[&sps, &pps, &aps, &idr]);
        record.accepted(&[&trail]);
        let newer_aps = unit(PREFIX_APS_NUT, 3 | 0x20);
        record.accepted(&[&newer_aps, &idr]);
        record.accepted(&[&trail]);
        let (pictures, shown) = record.replay().unwrap();
        assert_eq!(shown, 2);
        let expected: Vec<Vec<&[u8]>> =
            vec![vec![&sps, &pps, &newer_aps, &idr], vec![&trail], vec![]];
        assert_eq!(pictures, expected);
    }
}
