//! `--spatial-stats` JSON summary of per-layer byte counts, rate-control
//! actual bitrates and inter-layer reference usage.

use std::io::Write;
use std::path::Path;

use nc_encoder::{CodecBackend, EncodedFrame, Encoder, FrameType, SpatialReferenceStats};

use crate::stdio;

/// Collector for `--spatial-stats` per-layer byte and reference-hit summaries.
#[derive(Default)]
pub(crate) struct SpatialTelemetry {
    coded_tus: u64,
    key_tus: u64,
    s0_frames: u64,
    s1_frames: u64,
    s2_frames: u64,
    total_bytes: u64,
    s0_obu_bytes: u64,
    s1_obu_bytes: u64,
    s2_obu_bytes: u64,
    s0_key_bytes: u64,
    s1_key_bytes: u64,
    s2_key_bytes: u64,
    peak_tu_bytes: u64,
}

impl SpatialTelemetry {
    /// Record one coded temporal unit's byte distribution across `S0`, `S1` and `S2`.
    pub(crate) fn observe(&mut self, frame: &EncodedFrame) {
        self.coded_tus += 1;
        let is_key = frame.metadata.frame_type == FrameType::Key;
        if is_key {
            self.key_tus += 1;
        }
        let tu_bytes = frame.data.len() as u64;
        self.total_bytes += tu_bytes;
        self.peak_tu_bytes = self.peak_tu_bytes.max(tu_bytes);
        for layer in &frame.layers {
            let bytes = layer.byte_len as u64;
            match layer.spatial_id {
                0 => {
                    self.s0_frames += 1;
                    self.s0_obu_bytes += bytes;
                    if is_key {
                        self.s0_key_bytes += bytes;
                    }
                }
                1 => {
                    self.s1_frames += 1;
                    self.s1_obu_bytes += bytes;
                    if is_key {
                        self.s1_key_bytes += bytes;
                    }
                }
                _ => {
                    self.s2_frames += 1;
                    self.s2_obu_bytes += bytes;
                    if is_key {
                        self.s2_key_bytes += bytes;
                    }
                }
            }
        }
    }

    /// Write the collected spatial statistics and rate-control summary to `path`.
    pub(crate) fn write<B: CodecBackend>(
        &self,
        path: &Path,
        encoder: &Encoder<B>,
    ) -> Result<(), String> {
        let mut out = stdio::create(path)?;
        let SpatialReferenceStats {
            temporal_blocks,
            inter_layer_blocks,
            intra_blocks,
            temporal_area,
            inter_layer_area,
            intra_area,
        } = encoder.spatial_reference_stats().unwrap_or_default();
        let s1_ref = encoder.layer_reference_stats(1).unwrap_or_default();
        let s2_ref = encoder.layer_reference_stats(2).unwrap_or_default();
        let total_rc = encoder.rate_control_stats();
        let s0_rc = encoder.layer_rate_control_stats(0);
        let s1_rc = encoder.layer_rate_control_stats(1);
        let s2_rc = encoder.layer_rate_control_stats(2);
        writeln!(
            out,
            concat!(
                "{{\"coded_tus\":{},\"key_tus\":{},\"s0_frames\":{},\"s1_frames\":{},\"s2_frames\":{},",
                "\"total_bytes\":{},\"s0_obu_bytes\":{},\"s1_obu_bytes\":{},\"s2_obu_bytes\":{},",
                "\"s0_key_bytes\":{},\"s1_key_bytes\":{},\"s2_key_bytes\":{},\"peak_tu_bytes\":{},",
                "\"temporal_blocks\":{},\"inter_layer_blocks\":{},\"intra_blocks\":{},",
                "\"temporal_area\":{},\"inter_layer_area\":{},\"intra_area\":{},",
                "\"s1_temporal_blocks\":{},\"s1_inter_layer_blocks\":{},\"s1_intra_blocks\":{},",
                "\"s2_temporal_blocks\":{},\"s2_inter_layer_blocks\":{},\"s2_intra_blocks\":{},",
                "\"total_actual_bps\":{:.1},\"s0_actual_bps\":{:.1},\"s1_actual_bps\":{:.1},\"s2_actual_bps\":{:.1},",
                "\"buffer_overflows\":{},\"max_tu_bits\":{}}}"
            ),
            self.coded_tus,
            self.key_tus,
            self.s0_frames,
            self.s1_frames,
            self.s2_frames,
            self.total_bytes,
            self.s0_obu_bytes,
            self.s1_obu_bytes,
            self.s2_obu_bytes,
            self.s0_key_bytes,
            self.s1_key_bytes,
            self.s2_key_bytes,
            self.peak_tu_bytes,
            temporal_blocks,
            inter_layer_blocks,
            intra_blocks,
            temporal_area,
            inter_layer_area,
            intra_area,
            s1_ref.temporal_blocks,
            s1_ref.inter_layer_blocks,
            s1_ref.intra_blocks,
            s2_ref.temporal_blocks,
            s2_ref.inter_layer_blocks,
            s2_ref.intra_blocks,
            total_rc.map_or(0.0, |s| s.actual_bps),
            s0_rc.map_or(0.0, |s| s.actual_bps),
            s1_rc.map_or(0.0, |s| s.actual_bps),
            s2_rc.map_or(0.0, |s| s.actual_bps),
            total_rc.map_or(0, |s| s.buffer_overflows),
            total_rc.map_or(0, |s| s.max_frame_bits),
        )
        .map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())
    }
}
