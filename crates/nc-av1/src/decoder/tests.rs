//! Decoder round trips against the encoder's reconstruction.
use super::*;
use crate::Av1Backend;
use crate::backend::Block;
use crate::cdef::{self, CdefParams};
use nc_encoder::{
    BlockDecision, Codec, CodecBackend, Encoder, EncoderConfig, FramePlan, FrameRate, FrameType,
    ParamOverrides, PredictionMode,
};
fn config(width: usize, height: usize, quantizer: u32, block_log2: u8) -> EncoderConfig {
    EncoderConfig {
        codec: Codec::Av1,
        width,
        height,
        frame_rate: FrameRate { num: 30, den: 1 },
        quantizer,
        intra_period: 3,
        block_log2,
        key_qp_offset: None,
        temporal_layers: 1,
        signal: nc_encoder::StreamSignal::default(),
        params: ParamOverrides::default(),
    }
}
fn source(width: usize, height: usize, frame: usize) -> Picture {
    let mut p = Picture::new(width, height, 64).unwrap();
    for plane in PlaneIndex::ALL {
        let mut v = p.plane_mut(plane).view_mut();
        for y in 0..v.height() {
            for x in 0..v.width() {
                v.row_mut(y)[x] =
                    ((x * 13 + y * 7 + (x ^ y) * 3 + frame * 11 + plane as usize * 29) % 251) as u8;
            }
        }
    }
    p.extend_borders();
    p
}
#[test]
fn colour_signalling_reaches_the_decoder() {
    use nc_encoder::{ColourDescription, ColourInfo};
    for colour in [
        ColourInfo::default(),
        ColourInfo {
            description: Some(ColourDescription::BT709),
            full_range: true,
        },
        ColourInfo {
            description: None,
            full_range: true,
        },
    ] {
        let mut cfg = config(64, 64, 120, 4);
        cfg.signal.colour = colour;
        let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
        let mut decoder = Decoder::new();
        for frame in 0..3 {
            let packet = encoder.encode(&source(64, 64, frame));
            assert_picture(
                &decoder.decode(&packet.data).unwrap(),
                encoder.reconstruction(),
            );
        }
        assert_eq!(decoder.colour(), Some(colour));
    }
}
/// Packets of `frames` frames coded with `settings`, each checked
/// against our decoder.
fn overlay_packets(settings: &[(&str, f32)], frames: usize) -> Vec<Vec<u8>> {
    let mut cfg = config(256, 128, 160, 4);
    cfg.intra_period = 300;
    for &(name, value) in settings {
        cfg.params.set(name, value);
    }
    let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
    let mut decoder = Decoder::new();
    (0..frames)
        .map(|frame| {
            let packet = encoder.encode(&source(256, 128, frame));
            assert_picture(
                &decoder.decode(&packet.data).unwrap(),
                encoder.reconstruction(),
            );
            packet.data
        })
        .collect()
}
#[test]
fn nonbase_overlay_changes_only_nonbase_frames() {
    const FRAMES: usize = 6;
    let plain = overlay_packets(&[], FRAMES);
    // Non-base values equal to the ordinary ones change nothing.
    let same = overlay_packets(
        &[
            ("nonbase.av1.loop_filter", 1.0),
            ("nonbase.decision.max_cheap_vectors", 16.0),
        ],
        FRAMES,
    );
    assert_eq!(plain, same);
    let cheap = overlay_packets(
        &[
            ("nonbase.av1.loop_filter", 0.0),
            ("nonbase.decision.max_cheap_vectors", 1.0),
        ],
        FRAMES,
    );
    // Frame 0 is key and base; frame 1 is the first non-base frame.
    assert_eq!(plain[0], cheap[0]);
    assert_ne!(plain[1], cheap[1]);
}
#[test]
fn disposable_only_overlay_leaves_referenced_frames_unchanged() {
    const FRAMES: usize = 6;
    let layered = |settings: &[(&str, f32)]| {
        let mut cfg = config(256, 128, 160, 4);
        cfg.intra_period = 300;
        cfg.temporal_layers = 2;
        cfg.params.set("layers.adaptive", 0.0);
        for &(name, value) in settings {
            cfg.params.set(name, value);
        }
        let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
        let mut decoder = Decoder::new();
        (0..FRAMES)
            .map(|frame| {
                let packet = encoder.encode(&source(256, 128, frame));
                assert_picture(
                    &decoder.decode(&packet.data).unwrap(),
                    encoder.reconstruction(),
                );
                packet
            })
            .collect::<Vec<_>>()
    };
    let plain = layered(&[]);
    let cheap = layered(&[
        ("decision.nonbase_disposable_only", 1.0),
        ("nonbase.av1.loop_filter", 0.0),
        ("nonbase.decision.max_cheap_vectors", 1.0),
    ]);
    // T0 frames reference only T0 frames, and only the disposable T1
    // frames run the cheap settings.
    for (a, b) in plain.iter().zip(&cheap) {
        assert_eq!(a.data == b.data, a.metadata.temporal_layer == 0);
    }
}
#[test]
fn frames_without_cdf_adaptation_decode() {
    const FRAMES: usize = 6;
    let adapted = overlay_packets(&[], FRAMES);
    // Every frame, key frames included, and non-base frames only.
    let frozen = overlay_packets(&[("av1.cdf_update", 0.0)], FRAMES);
    let nonbase = overlay_packets(&[("nonbase.av1.cdf_update", 0.0)], FRAMES);
    assert_ne!(adapted[0], frozen[0]);
    assert_eq!(adapted[0], nonbase[0]);
    assert_ne!(adapted[1], nonbase[1]);
}
#[test]
fn nonbase_base_period_is_rejected() {
    let mut cfg = config(64, 64, 120, 4);
    cfg.params.set("nonbase.av1.base_period", 2.0);
    assert!(Encoder::<Av1Backend>::new(cfg).is_err());
}
#[test]
fn intra_refresh_recovers_from_wrong_references() {
    const PERIOD: u32 = 3;
    let mut cfg = config(256, 64, 120, 4);
    cfg.intra_period = 4;
    let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
    encoder
        .set_intra_refresh(Some(nc_encoder::IntraRefresh { period: PERIOD }))
        .unwrap();
    let mut dec = Decoder::new();
    let corrupt_after = 5;
    let mut mismatched = false;
    for index in 0..20u64 {
        let frame = encoder.encode(&source(256, 64, index as usize + 1));
        assert_eq!(frame.metadata.frame_type == FrameType::Key, index == 0);
        let pic = dec.decode(&frame.data).unwrap();
        if index == corrupt_after {
            dec.corrupt_references(0);
        }
        let exact = PlaneIndex::ALL.iter().all(|&p| {
            let (a, b) = (
                pic.plane(p).view(),
                encoder.reconstruction().plane(p).view(),
            );
            (0..a.height()).all(|y| a.row(y) == b.row(y))
        });
        // Recovery takes at most two sweeps: one to start a new one,
        // one to complete it.
        if index > corrupt_after + 2 * u64::from(PERIOD) {
            assert!(exact, "frame {index} not recovered");
        } else if index > corrupt_after {
            mismatched |= !exact;
        } else {
            assert!(exact, "frame {index}");
        }
    }
    assert!(mismatched, "the corruption had no effect");
}
/// A long-term reference every 4 frames in `temporal_layers` fixed
/// layers; the receiver loses `lost`, reports it, and decodes every
/// other frame exactly, with no key frame after the first.
fn long_term_recovery(temporal_layers: u8, lost: &[u64], frames: u64) {
    let mut cfg = config(128, 64, 120, 4);
    cfg.intra_period = 1000;
    cfg.temporal_layers = temporal_layers;
    cfg.params.set("ltr.interval", 4.0);
    cfg.params.set("layers.adaptive", 0.0);
    let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
    let mut decoder = Decoder::new();
    for index in 0..frames {
        let packet = encoder.encode(&source(128, 64, index as usize + 1));
        assert_eq!(packet.metadata.frame_type == FrameType::Key, index == 0);
        if lost.contains(&index) {
            // The receiver never sees this frame and says so.
            encoder.report_loss(index);
            continue;
        }
        let decoded = decoder.decode(&packet.data).unwrap();
        assert_picture(&decoded, encoder.reconstruction());
        encoder.acknowledge(index);
    }
}
/// Frame 9 is lost and its report reaches the encoder only after frame
/// 11 was sent: frames 10 and 11 name a slot whose frame id is not the one
/// they expect (§7.20) and are refused, so the decoder never predicts from
/// the picture before the loss; the recovery frame from the LTR decodes
/// exactly. Frame 13 is lost too and reported at once.
#[test]
fn frame_ids_refuse_frames_after_a_loss_until_recovery() {
    let mut cfg = config(128, 64, 120, 4);
    cfg.intra_period = 1000;
    cfg.params.set("ltr.interval", 4.0);
    let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
    let mut decoder = Decoder::new();
    for index in 0..18 {
        if index == 12 {
            encoder.report_loss(9);
        }
        let packet = encoder.encode(&source(128, 64, index as usize + 1));
        match index {
            9 | 13 => {} // lost
            10 | 11 | 14 => assert_eq!(
                decoder.decode(&packet.data).unwrap_err(),
                DecodeError::MissingReference,
                "frame {index}"
            ),
            _ => {
                let decoded = decoder
                    .decode(&packet.data)
                    .unwrap_or_else(|e| panic!("frame {index}: {e}"));
                assert_picture(&decoded, encoder.reconstruction());
                encoder.acknowledge(index);
            }
        }
        if index == 14 {
            encoder.report_loss(13);
        }
    }
}
#[test]
fn resolution_changes_decode_exactly() {
    let sizes = [(128, 64), (72, 40), (128, 64)];
    let mut cfg = config(128, 64, 120, 4);
    cfg.intra_period = 1000;
    let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
    let mut decoder = Decoder::new();
    for &(width, height) in &sizes {
        encoder.set_resolution(width, height).unwrap();
        for index in 0..4 {
            let packet = encoder.encode(&source(width, height, index + 1));
            assert_eq!(packet.metadata.frame_type == FrameType::Key, index == 0);
            assert_picture(
                &decoder.decode(&packet.data).unwrap(),
                encoder.reconstruction(),
            );
        }
    }
}
#[test]
fn long_term_reference_recovers_a_lost_frame() {
    long_term_recovery(1, &[9], 16);
}
#[test]
fn long_term_reference_recovers_with_two_layers() {
    // Frame 10 is T0 (recovered from the LTR marked at 8); 14 is T1.
    long_term_recovery(2, &[10, 14], 20);
}
#[test]
fn long_term_reference_recovers_with_three_layers() {
    // Frame 10 is T1 (recovered from the LTR marked at 8); 14 is T2.
    long_term_recovery(3, &[10, 14], 20);
}
fn assert_picture(a: &Picture, b: &Picture) {
    assert_eq!((a.width(), a.height()), (b.width(), b.height()));
    for plane in PlaneIndex::ALL {
        let a = a.plane(plane).view();
        let b = b.plane(plane).view();
        for y in 0..a.height() {
            assert_eq!(a.row(y), b.row(y), "plane={plane:?} row={y}");
        }
    }
}

/// A pattern scrolling by whole and half samples, so inter frames use
/// real vectors.
fn moving(width: usize, height: usize, frame: usize) -> Picture {
    let mut picture = Picture::new(width, height, 0).unwrap();
    for plane in PlaneIndex::ALL {
        let mut view = picture.plane_mut(plane).view_mut();
        let (plane_width, plane_height) = (view.width(), view.height());
        for row in 0..plane_height {
            for (col, sample) in view.row_mut(row).iter_mut().enumerate() {
                let x = (col + 3 * frame) % plane_width;
                let y = (row + frame) % plane_height;
                *sample = ((x * 13 + y * 7 + (x * y) % 5) % 256) as u8;
            }
        }
    }
    picture
}

/// Packets and reconstructions of `frames` moving frames, under rate
/// control at `bitrate` (bits/s, 200 ms buffer) if given, with a
/// requested key at frame 9.
fn coded_sequence(
    settings: EncoderConfig,
    bitrate: Option<u64>,
    frames: usize,
) -> (
    Vec<Vec<u8>>,
    Vec<Picture>,
    Option<nc_encoder::RateControlStats>,
) {
    let (width, height) = (settings.width, settings.height);
    let mut encoder = Encoder::<Av1Backend>::new(settings).unwrap();
    if let Some(bps) = bitrate {
        let mut rate = nc_encoder::RateControlConfig::new(bps);
        rate.buffer_ms = 200;
        encoder.set_rate_control(rate).unwrap();
    }
    let (mut packets, mut recons) = (Vec::new(), Vec::new());
    for frame in 0..frames {
        if frame == 9 {
            encoder.request_keyframe();
        }
        packets.push(encoder.encode(&source(width, height, frame)).data);
        recons.push(encoder.reconstruction().clone());
    }
    (packets, recons, encoder.rate_control_stats())
}

/// The recode contract (`CodecBackend::RECODE`): coding every frame
/// twice (`debug.recode_identity`) leaves the stream and the
/// reconstruction unchanged, across keys, a requested key, layers, the
/// long-term reference, rate control and the CDEF choices reused
/// between frames.
#[test]
fn recoding_every_frame_changes_nothing() {
    let cases: [(u8, f32, Option<u64>); 4] = [
        (1, 0., None),
        (3, 0., None),
        (1, 4., Some(150_000)),
        (2, 0., Some(150_000)),
    ];
    for (layers, ltr_interval, bitrate) in cases {
        let coded = |identity: f32| {
            // A coarse quantizer, so per-superblock CDEF is on, with
            // its choices reused between searched frames.
            let mut settings = config(256, 128, 200, 4);
            settings.params.set("av1.cdef_local_search_period", 4.);
            settings.params.set("av1.cdef_uv_search_primary", 7.);
            settings.params.set("av1.cdef_local_strong_luma", 0.);
            settings.params.set("debug.recode_identity", identity);
            settings.params.set("ltr.interval", ltr_interval);
            settings.intra_period = 7;
            settings.temporal_layers = layers;
            coded_sequence(settings, bitrate, 16)
        };
        let (once, once_recon, _) = coded(0.);
        let (twice, twice_recon, _) = coded(1.);
        assert_eq!(
            once, twice,
            "layers {layers} ltr {ltr_interval} rate {bitrate:?}"
        );
        for (a, b) in once_recon.iter().zip(&twice_recon) {
            assert_picture(a, b);
        }
    }
}

/// `rate.recode_overflow`: frames that would overflow a small buffer
/// are coded again, coarser, and still decode to the reconstruction.
#[test]
fn overflowing_frames_are_recoded_and_decode_exactly() {
    let mut settings = config(64, 64, 120, 4);
    settings.params.set("rate.recode_overflow", 1.);
    let (packets, recons, stats) = coded_sequence(settings, Some(20_000), 16);
    assert!(stats.unwrap().recoded_frames > 0);
    let mut decoder = Decoder::new();
    for (packet, recon) in packets.iter().zip(&recons) {
        assert_picture(&decoder.decode(packet).unwrap(), recon);
    }
}

#[test]
fn layered_streams_and_their_operating_points_decode_exactly() {
    for layers in [2u8, 3] {
        let mut settings = config(128, 64, 120, 4);
        settings.intra_period = 11;
        settings.temporal_layers = layers;
        // The fixed patterns (adaptive structure choice off).
        settings.params.set("layers.adaptive", 0.0);
        let mut encoder = Encoder::<Av1Backend>::new(settings).unwrap();
        let mut packets = Vec::new();
        for frame in 0..30 {
            if frame == 17 {
                encoder.request_keyframe();
            }
            let packet = encoder.encode(&moving(128, 64, frame));
            let layer = temporal_unit_layer(&packet.data).unwrap();
            assert_eq!(layer, packet.metadata.temporal_layer);
            packets.push((packet, encoder.reconstruction().clone()));
        }
        for max_layer in (0..layers).rev() {
            let mut decoder = Decoder::new();
            for (packet, recon) in &packets {
                if packet.metadata.temporal_layer <= max_layer {
                    let decoded = decoder.decode(&packet.data).unwrap_or_else(|e| {
                        panic!(
                            "L{layers} max {max_layer} frame {}: {e}",
                            packet.metadata.index
                        )
                    });
                    assert_picture(&decoded, recon);
                }
            }
        }
    }
}

#[test]
fn changing_noise_chroma_offsets_roundtrip_and_match_frame_headers() {
    let mut cfg = config(64, 64, 190, 6);
    cfg.params.set("av1.chroma_delta_q", -24.0);
    cfg.params.set("av1.chroma_noise_scale", 2.0);
    // The unscaled key offset, so the key frame carries the noise offset.
    cfg.params.set("av1.keyframe_chroma_scaled", 0.0);
    // This test drives blocks by hand and never calls `filter_frame`, so
    // per-superblock CDEF (whose choices are made there) stays off.
    cfg.params.set("av1.cdef_per_superblock", 0.0);
    let mut encoder = Av1Backend::new(&cfg).unwrap();
    let mut decoder = Decoder::new();
    let mut previous = None;
    for (index, sigma, delta) in [(0, 0.0, -24), (1, 1.0, -12), (2, 5.0, 0)] {
        encoder.set_noise_sigma(sigma);
        encoder.begin_frame(&FramePlan {
            index,
            frame_type: if index == 0 {
                FrameType::Key
            } else {
                FrameType::Inter
            },
            quantizer: cfg.quantizer,
            references: nc_encoder::FrameReferences::FLAT,
            lambda_scale: 1.0,
            block_qp: false,
        });
        let input = source(64, 64, index as usize);
        let mut expected = Picture::new(64, 64, 64).unwrap();
        let block = Block::square(0, 0, 6);
        encoder.begin_superblock(0, 0);
        encoder.write_partition(&block, false);
        encoder.encode_block(
            &block,
            &BlockDecision::new(PredictionMode::IntraDc),
            &input,
            previous.as_ref(),
            &mut expected,
        );
        let packet = encoder.end_frame();
        encoder.commit_frame();
        assert_eq!(decoder.peek_header(&packet).quant_deltas[1].1, delta);
        let decoded = decoder.decode(&packet).unwrap();
        assert_picture(&decoded, &expected);
        expected.extend_borders();
        previous = Some(expected);
    }
}

#[test]
fn optional_intra_cfl_local_cdef_and_fast_controls_roundtrip() {
    for variant in 0..6 {
        let mut cfg = config(128, 72, 190, 5);
        if variant >= 4 {
            // Searched chroma and a strong candidate with a secondary strength.
            cfg.params.set("av1.cdef_uv_search_primary", 6.0);
            cfg.params.set("av1.cdef_local_strong_secondary", 2.0);
        }
        cfg.params.set("av1.keyframe_intra_modes", 2.0);
        cfg.params.set("av1.keyframe_cfl", 1.0);
        cfg.params.set(
            "av1.chroma_delta_q",
            if variant % 2 == 0 { -12.0 } else { 11.0 },
        );
        cfg.params.set("av1.cdef_per_superblock", 1.0);
        if variant >= 2 {
            cfg.params.set("av1.pre_tx_skip", 1.0);
            cfg.params.set("av1.pre_tx_skip_score", 70.0);
            cfg.params.set("av1.transform_rd_skip", 2.0);
            cfg.params.set("av1.transform_rd_skip_chroma", 1.0);
        }
        let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
        let mut decoder = Decoder::new();
        for frame in 0..6 {
            let mut input = source(128, 72, frame);
            if variant % 2 == 0 {
                for plane in [PlaneIndex::U, PlaneIndex::V] {
                    let y = input.plane(PlaneIndex::Y).view();
                    let mut chroma = Vec::new();
                    for r in 0..36 {
                        for c in 0..64 {
                            chroma.push(y.row(2 * r)[2 * c]);
                        }
                    }
                    let mut p = input.plane_mut(plane).view_mut();
                    for r in 0..36 {
                        p.row_mut(r).copy_from_slice(&chroma[r * 64..(r + 1) * 64]);
                    }
                }
                input.extend_borders();
            }
            let packet = encoder.encode(&input);
            let decoded = decoder
                .decode(&packet.data)
                .unwrap_or_else(|e| panic!("variant{variant} frame{frame}: {e}"));
            assert_picture(&decoded, encoder.reconstruction());
        }
    }
}
#[test]
fn full_key_intra_set_roundtrips() {
    // Oblique directions, angle deltas and filter intra on key frames
    // (filter intra also makes inter-frame intra blocks code the flag),
    // at a size whose last superblock row and column are partial.
    for (deltas, candidates) in [(1.0, 3.0), (2.0, 16.0)] {
        let mut cfg = config(200, 136, 150, 5);
        cfg.params.set("av1.keyframe_directional", 1.0);
        cfg.params.set("av1.keyframe_angle_deltas", deltas);
        cfg.params.set("av1.keyframe_filter_intra", 1.0);
        cfg.params.set("av1.keyframe_cfl", 1.0);
        cfg.params
            .set("av1.keyframe_intra_rd_candidates", candidates);
        let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
        let mut decoder = Decoder::new();
        for frame in 0..3 {
            let packet = encoder.encode(&moving(200, 136, frame));
            let decoded = decoder
                .decode(&packet.data)
                .unwrap_or_else(|e| panic!("deltas {deltas} frame {frame}: {e}"));
            assert_picture(&decoded, encoder.reconstruction());
        }
    }
}

#[test]
fn mixed_cdef_indices_are_read_at_their_normative_block_position() {
    let mut cfg = config(128, 64, 240, 6);
    cfg.params.set("av1.cdef_per_superblock", 1.0);
    let mut encoder = Av1Backend::new(&cfg).unwrap();
    encoder.begin_frame(&FramePlan {
        index: 0,
        frame_type: FrameType::Key,
        quantizer: cfg.quantizer,
        references: nc_encoder::FrameReferences::FLAT,
        lambda_scale: 1.0,
        block_qp: false,
    });
    let input = source(128, 64, 0);
    let mut expected = Picture::new(128, 64, 64).unwrap();
    for x in [0, 64] {
        let block = Block::square(x, 0, 6);
        encoder.begin_superblock(x, 0);
        encoder.write_partition(&block, false);
        encoder.encode_block(
            &block,
            &BlockDecision::new(PredictionMode::IntraDc),
            &input,
            None,
            &mut expected,
        );
    }
    let filter = CdefParams {
        damping_minus_3: 2,
        y_pri: 7,
        y_sec: 2,
        uv_pri: 2,
        uv_sec: 1,
    };
    encoder.finish_cdef_indices(filter, &[false, true]);
    cdef::apply_selected(
        &mut expected,
        &encoder.grids.mode_info,
        filter,
        &[false, true],
    );
    let packet = encoder.end_frame();
    encoder.commit_frame();
    assert_eq!(Decoder::new().peek_header(&packet).cdef.bits, 1);
    assert_picture(&Decoder::new().decode(&packet).unwrap(), &expected);
}
#[test]
fn truncated_packets_fail_without_losing_reference() {
    let mut encoder = Encoder::<Av1Backend>::new(config(32, 24, 190, 4)).unwrap();
    let key = encoder.encode(&source(32, 24, 0));
    let inter = encoder.encode(&source(32, 24, 1));
    let mut decoder = Decoder::new();
    decoder.decode(&key.data).unwrap();
    for n in 0..inter.data.len() {
        assert!(decoder.decode(&inter.data[..n]).is_err(), "prefix {n}");
    }
    assert_picture(
        &decoder.decode(&inter.data).unwrap(),
        encoder.reconstruction(),
    );
    for index in (0..key.data.len()).step_by((key.data.len() / 64).max(1)) {
        let mut corrupt = key.data.clone();
        corrupt[index] ^= 0xff;
        let _ = Decoder::new().decode(&corrupt);
    }
    assert!(Decoder::new().decode(&inter.data).is_err());
}
/// Flat frames alternating black and white make every inter residual
/// ±255. At fine steps these need levels far above what a coarse step
/// would (32768 for a 64x64 DC at step 8); a fixed level cap once halved
/// them and reconstructed white as 128..255 (7 dB) at qindex 1-10.
#[test]
fn full_swing_inter_residuals_reconstruct_at_fine_steps() {
    const FRAMES: usize = 6;
    const MAX_ERROR: i32 = 2;
    let (w, h) = (128, 64);
    for q in [1, 5, 10] {
        let mut cfg = config(w, h, q, 6);
        cfg.intra_period = 300;
        let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
        let mut decoder = Decoder::new();
        for frame in 0..FRAMES {
            let luma = if frame % 2 == 0 { 0 } else { 255 };
            let mut src = Picture::new(w, h, 64).unwrap();
            for plane in PlaneIndex::ALL {
                let value = if plane == PlaneIndex::Y { luma } else { 128 };
                let mut v = src.plane_mut(plane).view_mut();
                for y in 0..v.height() {
                    v.row_mut(y).fill(value);
                }
            }
            src.extend_borders();
            let packet = encoder.encode(&src);
            let recon = encoder.reconstruction().plane(PlaneIndex::Y).view();
            for y in 0..recon.height() {
                for &sample in recon.row(y) {
                    let error = (i32::from(sample) - i32::from(luma)).abs();
                    assert!(error <= MAX_ERROR, "q{q} frame{frame}: {sample} for {luma}");
                }
            }
            let decoded = decoder.decode(&packet.data).unwrap();
            assert_picture(&decoded, encoder.reconstruction());
        }
    }
}
#[test]
fn independent_decoder_matches_key_inter_and_key_refresh() {
    for (w, h, q, block) in [(64, 64, 100, 5), (128, 72, 190, 6), (32, 24, 240, 3)] {
        let mut encoder = Encoder::<Av1Backend>::new(config(w, h, q, block)).unwrap();
        let mut decoder = Decoder::new();
        for frame in 0..5 {
            let packet = encoder.encode(&source(w, h, frame));
            let decoded = decoder
                .decode(&packet.data)
                .unwrap_or_else(|e| panic!("{w}x{h} q{q} frame{frame}: {e}"));
            assert_picture(&decoded, encoder.reconstruction());
        }
    }
}

mod rate_control_header_tests {
    use super::*;
    use crate::Av1Backend;
    use crate::obu::FrameKind;
    use nc_encoder::{Codec, Encoder, EncoderConfig, FrameRate, ParamOverrides, RateControlConfig};
    #[test]
    fn actual_frame_headers_honor_rc_bounds_and_restore_fixed_key_boost() {
        let config = EncoderConfig {
            codec: Codec::Av1,
            width: 16,
            height: 16,
            frame_rate: FrameRate { num: 15, den: 1 },
            quantizer: 180,
            intra_period: 30,
            block_log2: 4,
            key_qp_offset: None,
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal::default(),
            params: ParamOverrides::default().coding_every_frame(),
        };
        let mut encoder = Encoder::<Av1Backend>::new(config).unwrap();
        let mut decoder = crate::Decoder::new();
        let mut source = Picture::new(16, 16, 0).unwrap();
        for plane in nc_common::PlaneIndex::ALL {
            let mut view = source.plane_mut(plane).view_mut();
            for y in 0..view.height() {
                for (x, s) in view.row_mut(y).iter_mut().enumerate() {
                    *s = ((x * 13 + y * 7 + plane as usize * 31) % 256) as u8;
                }
            }
        }
        for (index, q) in [100, 140, 80].into_iter().enumerate() {
            let mut rc = RateControlConfig::new(200_000);
            rc.min_quantizer = Some(q);
            rc.max_quantizer = Some(q);
            encoder.set_rate_control(rc).unwrap();
            if index == 2 {
                encoder.request_keyframe();
            }
            let packet = encoder.encode(&source);
            let header = decoder.peek_header(&packet.data);
            assert_eq!(header.base_q_idx, q);
            assert_eq!(packet.metadata.quantizer, q);
            assert_eq!(encoder.rate_control_stats().unwrap().current_quantizer, q);
            assert_eq!(header.kind == FrameKind::Key, index != 1);
            let decoded = decoder.decode(&packet.data).unwrap();
            for plane in nc_common::PlaneIndex::ALL {
                assert!(
                    decoded
                        .plane(plane)
                        .view()
                        .rows()
                        .zip(encoder.reconstruction().plane(plane).view().rows())
                        .all(|(a, b)| a == b)
                );
            }
        }
        encoder.disable_rate_control();
        encoder.request_keyframe();
        let packet = encoder.encode(&source);
        let header = decoder.peek_header(&packet.data);
        assert_eq!(header.base_q_idx, crate::backend::scaled_qindex(180, 0.24));
        assert_eq!(packet.metadata.quantizer, 180);
        assert!(encoder.rate_control_stats().is_none());
    }
}
#[test]
fn shown_frames_equal_decoded_ones() {
    // 66x42 is cropped from the coded 72x48 (whole 8x8 units).
    for (width, height) in [(64, 64), (66, 42)] {
        let mut encoder = Encoder::<Av1Backend>::new(config(width, height, 120, 4)).unwrap();
        let (mut owned, mut borrowed) = (Decoder::new(), Decoder::new());
        for frame in 0..4 {
            let packet = encoder.encode(&source(width, height, frame));
            let decoded = owned.decode(&packet.data).unwrap();
            borrowed.decode_shown(&packet.data).unwrap();
            let shown = borrowed.shown().unwrap();
            assert_eq!((shown.width(), shown.height()), (width, height));
            for plane in PlaneIndex::ALL {
                assert!(
                    shown
                        .plane(plane)
                        .rows()
                        .eq(decoded.plane(plane).view().rows())
                );
            }
        }
    }
}

#[test]
fn superblock_quantizer_deltas_roundtrip() {
    // Two superblocks per frame: the first one whole, the second split into
    // four 32x32 blocks whose later offsets the superblock's first overrides.
    // Offsets cover the direct (|delta| < 3) and escaped `delta_q_abs` codes
    // of both signs, a clamp at qindex 1, and whole skipped superblocks
    // (a repeated frame) that code no delta.
    let mut cfg = config(128, 64, 140, 6);
    cfg.params.set("av1.cdef_per_superblock", 0.0);
    let mut encoder = Av1Backend::new(&cfg).unwrap();
    encoder.enable_block_qp();
    let mut decoder = Decoder::new();
    let mut previous: Option<Picture> = None;
    let offsets: [[i16; 2]; 4] = [[-40, 2], [0, -1], [100, -200], [-3, 3]];
    for (index, [whole, split]) in offsets.into_iter().enumerate() {
        let key = index == 0;
        encoder.begin_frame(&FramePlan {
            index: index as u64,
            frame_type: if key {
                FrameType::Key
            } else {
                FrameType::Inter
            },
            quantizer: cfg.quantizer,
            references: nc_encoder::FrameReferences::FLAT,
            lambda_scale: 1.0,
            block_qp: true,
        });
        // Frame 2 repeats its reference exactly: every block is skipped.
        let input = match (index, &previous) {
            (2, Some(reference)) => reference.clone(),
            _ => source(128, 64, index),
        };
        let mut expected = Picture::new(128, 64, 64).unwrap();
        let mode = if key {
            PredictionMode::IntraDc
        } else {
            PredictionMode::Inter(nc_encoder::Mv::ZERO)
        };
        let decision = |qp_offset| BlockDecision {
            qp_offset,
            ..BlockDecision::new(mode)
        };
        encoder.begin_superblock(0, 0);
        let sb = Block::square(0, 0, 6);
        encoder.write_partition(&sb, false);
        encoder.encode_block(
            &sb,
            &decision(whole),
            &input,
            previous.as_ref(),
            &mut expected,
        );
        encoder.begin_superblock(64, 0);
        encoder.write_partition(&Block::square(64, 0, 6), true);
        for (i, (x, y)) in [(64, 0), (96, 0), (64, 32), (96, 32)]
            .into_iter()
            .enumerate()
        {
            let leaf = Block::square(x, y, 5);
            encoder.write_partition(&leaf, false);
            let offset = if i == 0 { split } else { split + 9 * i as i16 };
            encoder.encode_block(
                &leaf,
                &decision(offset),
                &input,
                previous.as_ref(),
                &mut expected,
            );
        }
        // Deblocking reaches transform edges inside the blocks.
        encoder.filter_frame(&input, &mut expected);
        let packet = encoder.end_frame();
        encoder.commit_frame();
        assert_eq!(decoder.peek_header(&packet).delta_q_res, Some(0));
        let decoded = decoder.decode(&packet).unwrap();
        assert_picture(&decoded, &expected);
        expected.extend_borders();
        previous = Some(expected);
    }
}

#[test]
fn variance_aq_streams_roundtrip_and_differ() {
    let packets = |aq_strength: f32| {
        let mut cfg = config(192, 128, 120, 6);
        cfg.intra_period = 300;
        cfg.params.set("rate.aq_strength", aq_strength);
        let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
        let mut decoder = Decoder::new();
        (0..4)
            .map(|frame| {
                // Flat left half, the busy pattern on the right.
                let mut input = source(192, 128, frame);
                for plane in PlaneIndex::ALL {
                    let mut v = input.plane_mut(plane).view_mut();
                    for y in 0..v.height() {
                        let half = v.width() / 2;
                        v.row_mut(y)[..half].fill(100 + frame as u8);
                    }
                }
                input.extend_borders();
                let packet = encoder.encode(&input);
                let header = decoder.peek_header(&packet.data);
                assert_eq!(header.delta_q_res.is_some(), aq_strength > 0.0);
                assert_picture(
                    &decoder.decode(&packet.data).unwrap(),
                    encoder.reconstruction(),
                );
                packet.data
            })
            .collect::<Vec<_>>()
    };
    assert_ne!(packets(0.0), packets(2.0));
}

/// Superres (`av1.superres_denom`): frames coded narrower than the stream,
/// upscaled normatively, every inter block predicted from a wider
/// reference, decode to the encoder's (upscaled) reconstruction; also when
/// every frame is coded twice and across a key-frame resolution change.
#[test]
fn superres_frames_decode_exactly() {
    for (denom, recode) in [(16.0, false), (9.0, true), (10.0, false)] {
        let mut cfg = config(160, 48, 150, 4);
        cfg.intra_period = 6;
        cfg.params.set("av1.superres_denom", denom);
        cfg.params
            .set("debug.recode_identity", f32::from(u8::from(recode)));
        let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
        let mut decoder = Decoder::new();
        for (width, height) in [(160, 48), (320, 64)] {
            encoder.set_resolution(width, height).unwrap();
            for frame in 0..8 {
                let packet = encoder.encode(&moving(width, height, frame));
                let reconstruction = encoder.reconstruction();
                assert_eq!(
                    (reconstruction.width(), reconstruction.height()),
                    (width, height)
                );
                assert_picture(&decoder.decode(&packet.data).unwrap(), reconstruction);
            }
        }
    }
}

/// A denominator that would code an odd width is refused.
#[test]
fn superres_with_an_odd_coded_width_is_refused() {
    // 160 * 8 / 11 rounds to 116 (even); 150 * 8 / 11 to 109 (odd).
    let mut cfg = config(150, 48, 150, 4);
    cfg.params.set("av1.superres_denom", 11.0);
    assert!(Encoder::<Av1Backend>::new(cfg).is_err());
}

/// Coded-size switches without a key frame (`Encoder::set_coded_resolution`):
/// every frame after a switch is an inter frame predicting from references
/// of the other size (scaled, §7.11.3.3), the stream keeps its size, and
/// each picture decodes to the encoder's reconstruction at its coded size.
/// Covers a long-term reference still holding a picture two switches old,
/// layered structures, re-coding, and stream sizes whose halves round up.
#[test]
fn coded_size_switches_without_keys_decode_exactly() {
    type Case = (usize, usize, &'static [(&'static str, f32)]);
    let cases: [Case; 4] = [
        (128, 64, &[]),
        (168, 104, &[("ltr.interval", 4.0)]),
        (
            128,
            64,
            &[("layers.adaptive", 0.0), ("debug.recode_identity", 1.0)],
        ),
        (136, 72, &[("av1.still_skip", 1.0)]),
    ];
    for (case, &(width, height, params)) in cases.iter().enumerate() {
        let mut cfg = config(width, height, 120, 4);
        cfg.intra_period = 1000;
        cfg.temporal_layers = if case == 2 { 3 } else { 1 };
        for &(name, value) in params {
            cfg.params.set(name, value);
        }
        let half = (
            width.div_ceil(2).next_multiple_of(2),
            height.div_ceil(2).next_multiple_of(2),
        );
        let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
        let mut decoder = Decoder::new();
        let mut index = 0;
        for size in [
            (width, height),
            half,
            (width, height),
            half,
            (width, height),
        ] {
            encoder.set_coded_resolution(size.0, size.1).unwrap();
            for _ in 0..5 {
                let packet = encoder.encode(&moving(width, height, index));
                assert_eq!(packet.metadata.frame_type == FrameType::Key, index == 0);
                let reconstruction = encoder.reconstruction();
                assert_eq!((reconstruction.width(), reconstruction.height()), size);
                assert_picture(&decoder.decode(&packet.data).unwrap(), reconstruction);
                index += 1;
            }
        }
    }
}

/// A switch the codec cannot predict across (a quarter size, past the 2x
/// limit of §7.11.3.3) is refused and leaves the encoder working.
#[test]
fn coded_size_switch_past_twice_is_refused() {
    let mut cfg = config(128, 64, 120, 4);
    cfg.intra_period = 1000;
    let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
    let mut decoder = Decoder::new();
    let packet = encoder.encode(&moving(128, 64, 0));
    assert_picture(
        &decoder.decode(&packet.data).unwrap(),
        encoder.reconstruction(),
    );
    assert!(encoder.set_coded_resolution(32, 16).is_err());
    let packet = encoder.encode(&moving(128, 64, 1));
    assert_picture(
        &decoder.decode(&packet.data).unwrap(),
        encoder.reconstruction(),
    );
}
/// Garbage between frames is refused and changes nothing: the next frame
/// decodes exactly. With frame ids (without them these bytes are a valid
/// `show_existing_frame`).
#[test]
fn garbage_between_frames_changes_nothing() {
    let mut cfg = config(64, 64, 120, 4);
    cfg.params.set(crate::FRAME_IDS_PARAM, 1.0);
    let mut encoder = Encoder::<Av1Backend>::new(cfg).unwrap();
    let mut decoder = Decoder::new();
    for index in 0..3 {
        let packet = encoder.encode(&source(64, 64, index + 1));
        if index == 2 {
            let mut state = 3_u32;
            let noise: Vec<u8> = (0..300)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 24) as u8
                })
                .collect();
            // These bytes parse as show_existing_frame of a slot with a
            // display_frame_id that is not the slot's.
            assert_eq!(
                decoder.decode(&noise).unwrap_err(),
                DecodeError::MissingReference
            );
        }
        assert_picture(
            &decoder.decode(&packet.data).unwrap(),
            encoder.reconstruction(),
        );
    }
}
