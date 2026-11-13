//! Round trips of encoder packets through the decoder (bit-exact against
//! the encoder reconstruction), robustness and real-use stream tests.

use super::*;

mod round_trips {
    use super::*;
    use crate::HevcBackend;
    use nc_encoder::{Codec, Encoder, EncoderConfig, FrameRate, ParamOverrides};
    fn compare(a: &Picture, b: &Picture) {
        for p in PlaneIndex::ALL {
            let av = a.plane(p).view();
            let bv = b.plane(p).view();
            for (y, (x, z)) in av.rows().zip(bv.rows()).enumerate() {
                assert_eq!(x, z, "{p:?} row{y}");
            }
        }
    }
    #[test]
    fn actual_encoder_packets_reconstruct_exactly() {
        for (width, height) in [(16, 16), (64, 64), (72, 40)] {
            for qp in [0, 28, 51] {
                let config = EncoderConfig {
                    codec: Codec::Hevc,
                    width,
                    height,
                    frame_rate: FrameRate { num: 30, den: 1 },
                    quantizer: qp,
                    intra_period: 3,
                    block_log2: 6,
                    key_qp_offset: None,
                    temporal_layers: 1,
                    signal: nc_encoder::StreamSignal::default(),
                    params: ParamOverrides::default(),
                };
                let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
                let mut decoder = Decoder::new();
                for frame in 0..4 {
                    let mut source = Picture::new(width, height, 0).unwrap();
                    for p in PlaneIndex::ALL {
                        let mut v = source.plane_mut(p).view_mut();
                        for y in 0..v.height() {
                            for (x, s) in v.row_mut(y).iter_mut().enumerate() {
                                *s = ((x * 17 + y * 29 + x * y + frame * 7 + p as usize * 31) % 256)
                                    as u8;
                            }
                        }
                    }
                    let packet = encoder.encode(&source);
                    let decoded = decoder
                        .decode(&packet.data)
                        .unwrap_or_else(|e| panic!("{width}x{height} qp{qp} frame{frame}: {e}"));
                    compare(&decoded, encoder.reconstruction());
                }
            }
        }
    }

    #[cfg(feature = "analysis")]
    #[test]
    fn analysis_leaves_decoding_unchanged_and_tiles_each_picture() {
        use nc_common::stream_stats::{Pred, RecordKind, StreamStats, keys};
        let (width, height) = (72, 40);
        let config = EncoderConfig {
            codec: Codec::Hevc,
            width,
            height,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 30,
            intra_period: 3,
            block_log2: 6,
            key_qp_offset: None,
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal::default(),
            params: ParamOverrides::default(),
        };
        let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
        let mut decoder = Decoder::new();
        decoder.collect_stats(StreamStats::default());
        for frame in 0..4 {
            let packet = encoder.encode(&moving(width, height, frame));
            let decoded = decoder.decode(&packet.data).unwrap();
            compare(&decoded, encoder.reconstruction());
            // The decoded picture's records are available at once.
            let coded: u64 = decoder
                .stats()
                .unwrap()
                .records()
                .iter()
                .filter(|r| r.kind == RecordKind::Coding)
                .map(nc_common::stream_stats::BlockRecord::area)
                .sum();
            assert_eq!(coded, (width * height) as u64, "frame {frame}");
        }
        let stats = decoder.take_stats().unwrap();
        assert_eq!(stats.frames.len(), 4);
        let area = (width * height) as u64;
        for (index, frame) in stats.frames.iter().enumerate() {
            let covered: u64 = frame.tallies[keys::CB_SIZE]
                .values()
                .map(|t| t.weight)
                .sum();
            assert_eq!(
                covered, area,
                "frame {index}: coding blocks tile the picture"
            );
            let intra = frame.get(keys::CB_PRED, Pred::Intra as i64).weight;
            assert_eq!(intra == area, index % 3 == 0, "frame {index}");
        }
        let last = stats.records();
        assert!(last.iter().any(|r| r.kind == RecordKind::Prediction));
        assert!(
            last.iter()
                .all(|r| usize::from(r.x) < width && usize::from(r.y) < height)
        );
    }

    #[test]
    fn block_qp_streams_reconstruct_exactly_and_differ() {
        // Six CTBs, a flat left half and a busy right half: variance AQ
        // gives the CTBs different QPs (cu_qp_delta of both signs, up to
        // the clamps at QP 0 and 51), predicted from CTB to CTB.
        let (width, height) = (192, 128);
        for qp in [4, 30, 49] {
            let packets = |aq_strength: f32| {
                let mut params = ParamOverrides::default();
                params.set("rate.aq_strength", aq_strength);
                params.set("rate.aq_max_offset", 12.0);
                let config = EncoderConfig {
                    codec: Codec::Hevc,
                    width,
                    height,
                    frame_rate: FrameRate { num: 30, den: 1 },
                    quantizer: qp,
                    intra_period: 3,
                    block_log2: 6,
                    key_qp_offset: None,
                    temporal_layers: 1,
                    signal: nc_encoder::StreamSignal::default(),
                    params,
                };
                let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
                let mut decoder = Decoder::new();
                (0..4)
                    .map(|frame| {
                        let mut source = Picture::new(width, height, 0).unwrap();
                        for p in PlaneIndex::ALL {
                            let mut v = source.plane_mut(p).view_mut();
                            let half = v.width() / 2;
                            for y in 0..v.height() {
                                for (x, s) in v.row_mut(y).iter_mut().enumerate() {
                                    *s = if x < half {
                                        (100 + y / 8 + frame) as u8
                                    } else {
                                        ((x * 37 + y * 91 + x * y * 5 + frame * 13) % 256) as u8
                                    };
                                }
                            }
                        }
                        let packet = encoder.encode(&source);
                        let decoded = decoder
                            .decode(&packet.data)
                            .unwrap_or_else(|e| panic!("qp{qp} aq{aq_strength} frame{frame}: {e}"));
                        compare(&decoded, encoder.reconstruction());
                        packet.data
                    })
                    .collect::<Vec<_>>()
            };
            assert_ne!(packets(0.0), packets(3.0), "qp{qp}: AQ changed nothing");
        }
    }

    #[test]
    fn colour_signalling_reaches_the_decoder() {
        use nc_encoder::{ColourDescription, ColourInfo};
        for colour in [
            ColourInfo::default(),
            ColourInfo {
                description: Some(ColourDescription::BT601_525),
                full_range: false,
            },
            ColourInfo {
                description: None,
                full_range: true,
            },
        ] {
            let config = EncoderConfig {
                codec: Codec::Hevc,
                width: 64,
                height: 64,
                frame_rate: FrameRate { num: 30, den: 1 },
                quantizer: 30,
                intra_period: 3,
                block_log2: 6,
                key_qp_offset: None,
                temporal_layers: 1,
                signal: nc_encoder::StreamSignal {
                    colour,
                    ..Default::default()
                },
                params: ParamOverrides::default(),
            };
            let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
            let mut decoder = Decoder::new();
            for frame in 0..3 {
                let packet = encoder.encode(&moving(64, 64, frame));
                compare(
                    &decoder.decode(&packet.data).unwrap(),
                    encoder.reconstruction(),
                );
            }
            assert_eq!(decoder.colour(), Some(colour));
        }
    }

    #[test]
    fn streams_with_dropped_frames_decode_exactly() {
        for layers in [1u8, 2, 3] {
            let config = EncoderConfig {
                codec: Codec::Hevc,
                width: 64,
                height: 64,
                frame_rate: FrameRate { num: 30, den: 1 },
                quantizer: 30,
                intra_period: 30,
                block_log2: 6,
                key_qp_offset: None,
                temporal_layers: layers,
                signal: nc_encoder::StreamSignal::default(),
                params: ParamOverrides::default(),
            };
            let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
            let mut rate = nc_encoder::RateControlConfig::new(2_000);
            rate.peak_bps = Some(2_000);
            rate.frame_drop = true;
            encoder.set_rate_control(rate).unwrap();
            let mut decoder = Decoder::new();
            let mut dropped = 0;
            for frame in 0..16 {
                let packet = encoder.encode(&moving(64, 64, frame));
                if packet.metadata.dropped {
                    assert_eq!(packet.data, Vec::<u8>::new());
                    dropped += 1;
                    continue;
                }
                let decoded = decoder
                    .decode(&packet.data)
                    .unwrap_or_else(|e| panic!("{layers} layers, frame {frame}: {e}"));
                compare(&decoded, encoder.reconstruction());
            }
            assert!(dropped > 0, "{layers} layers: nothing dropped");
        }
    }

    /// Cheap settings on the frames nothing references (`nonbase.hevc.*`
    /// with `decision.nonbase_disposable_only`): deblocking and SAO off,
    /// a coarser pre-transform zero bound. Only the top-layer pictures change
    /// (the others only gain the slice header's deblocking override flag),
    /// and every frame decodes to the reconstruction.
    #[test]
    fn disposable_overlay_changes_only_disposable_frames_and_decodes() {
        const FRAMES: usize = 8;
        let layered = |settings: &[(&str, f32)]| {
            let mut params = ParamOverrides::default();
            params.set("layers.count", 2.0);
            params.set("layers.adaptive", 0.0);
            for &(name, value) in settings {
                params.set(name, value);
            }
            let config = EncoderConfig {
                codec: Codec::Hevc,
                width: 64,
                height: 48,
                frame_rate: FrameRate { num: 30, den: 1 },
                quantizer: 30,
                intra_period: 300,
                block_log2: 6,
                key_qp_offset: None,
                temporal_layers: 1,
                signal: nc_encoder::StreamSignal::default(),
                params,
            };
            let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
            let mut decoder = Decoder::new();
            (0..FRAMES)
                .map(|index| {
                    let packet = encoder.encode(&panning(64, 48, index));
                    compare(
                        &decoder.decode(&packet.data).unwrap(),
                        encoder.reconstruction(),
                    );
                    (
                        packet.metadata.temporal_layer,
                        encoder.reconstruction().clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let plain = layered(&[]);
        let cheap = layered(&[
            ("decision.nonbase_disposable_only", 1.0),
            ("nonbase.hevc.deblock_inter", 0.0),
            ("nonbase.hevc.sao_luma", 0.0),
            ("nonbase.hevc.sao_chroma", 0.0),
            ("nonbase.hevc.pre_transform_zero_scale", 4.0),
        ]);
        for ((layer, a), (_, b)) in plain.iter().zip(&cheap) {
            let same = PlaneIndex::ALL
                .iter()
                .all(|&p| a.plane(p).view().rows().eq(b.plane(p).view().rows()));
            assert_eq!(same, *layer == 0);
        }
    }

    #[test]
    fn stream_fixed_parameters_are_not_per_frame() {
        for name in [
            "nonbase.hevc.chroma_qp_offset",
            "nonbase.hevc.max_transform_log2",
        ] {
            let mut params = ParamOverrides::default();
            params.set(name, 4.0);
            let config = EncoderConfig {
                codec: Codec::Hevc,
                width: 64,
                height: 48,
                frame_rate: FrameRate { num: 30, den: 1 },
                quantizer: 30,
                intra_period: 300,
                block_log2: 6,
                key_qp_offset: None,
                temporal_layers: 1,
                signal: nc_encoder::StreamSignal::default(),
                params,
            };
            assert!(Encoder::<HevcBackend>::new(config).is_err(), "{name}");
        }
    }

    /// A non-repeating texture panning left by 20 samples per frame: motion
    /// vectors point right, towards intra refresh's dirty region.
    fn panning(width: usize, height: usize, frame: usize) -> Picture {
        let mut p = Picture::new(width, height, 0).unwrap();
        for plane in PlaneIndex::ALL {
            let mut view = p.plane_mut(plane).view_mut();
            let shift = if plane == PlaneIndex::Y { 20 } else { 10 };
            for y in 0..view.height() {
                for (x, v) in view.row_mut(y).iter_mut().enumerate() {
                    let gx = x + shift * frame;
                    *v = ((gx * gx * 7 + y * y * 3 + gx * y + plane as usize * 50) % 251) as u8;
                }
            }
        }
        p
    }

    /// A long-term reference every 4 frames in `temporal_layers` fixed
    /// layers; the receiver loses `lost`, reports it, and decodes every
    /// other frame exactly, with no key frame after the first.
    fn long_term_recovery(temporal_layers: u8, lost: &[u64], frames: u64) {
        let mut params = ParamOverrides::default();
        params.set("ltr.interval", 4.0);
        params.set("layers.adaptive", 0.0);
        let config = EncoderConfig {
            codec: Codec::Hevc,
            width: 128,
            height: 64,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 30,
            intra_period: 1000,
            block_log2: 4,
            key_qp_offset: None,
            temporal_layers,
            signal: nc_encoder::StreamSignal::default(),
            params,
        };
        let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
        let mut decoder = Decoder::new();
        for index in 0..frames {
            let packet = encoder.encode(&panning(128, 64, index as usize));
            assert_eq!(
                packet.metadata.frame_type == nc_encoder::FrameType::Key,
                index == 0
            );
            if lost.contains(&index) {
                encoder.report_loss(index);
                continue;
            }
            let decoded = decoder
                .decode(&packet.data)
                .unwrap_or_else(|e| panic!("frame {index}: {e}"));
            compare(&decoded, encoder.reconstruction());
            encoder.acknowledge(index);
        }
    }

    /// Frame 9 is lost and its report reaches the encoder only after frame
    /// 11 was sent: the frames between name it in their reference picture
    /// sets and must be refused ("no reference picture", §8.2.3), not
    /// predicted from a stand-in; the recovery frame then decodes exactly.
    #[test]
    fn frames_after_a_loss_are_refused_until_recovery() {
        let mut params = ParamOverrides::default();
        params.set("ltr.interval", 4.0);
        let config = EncoderConfig {
            codec: Codec::Hevc,
            width: 128,
            height: 64,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 30,
            intra_period: 1000,
            block_log2: 4,
            key_qp_offset: None,
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal::default(),
            params,
        };
        let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
        let mut decoder = Decoder::new();
        for index in 0..14 {
            if index == 12 {
                encoder.report_loss(9);
            }
            let packet = encoder.encode(&panning(128, 64, index as usize));
            match index {
                9 => {} // lost
                10 | 11 => assert_eq!(
                    decoder.decode(&packet.data).unwrap_err(),
                    DecodeError::MissingReference,
                    "frame {index}"
                ),
                _ => {
                    let decoded = decoder
                        .decode(&packet.data)
                        .unwrap_or_else(|e| panic!("frame {index}: {e}"));
                    compare(&decoded, encoder.reconstruction());
                    encoder.acknowledge(index);
                }
            }
        }
    }

    /// Key frames after the first are CRA pictures and POC keeps counting:
    /// when the key at frame 8 is lost, the pictures after it name its POC
    /// and are refused, where with a POC restart at an IDR they would name
    /// POC 0..7 pictures the receiver still holds. The next key decodes.
    #[test]
    fn frames_after_a_lost_key_are_refused() {
        let config = EncoderConfig {
            codec: Codec::Hevc,
            width: 128,
            height: 64,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 30,
            intra_period: 8,
            block_log2: 4,
            key_qp_offset: None,
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal::default(),
            params: ParamOverrides::default(),
        };
        let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
        let mut decoder = Decoder::new();
        for index in 0..20 {
            let packet = encoder.encode(&panning(128, 64, index as usize));
            assert_eq!(
                packet.metadata.frame_type == nc_encoder::FrameType::Key,
                index % 8 == 0
            );
            match index {
                8 => {} // lost
                9..=15 => assert_eq!(
                    decoder.decode(&packet.data).unwrap_err(),
                    DecodeError::MissingReference,
                    "frame {index}"
                ),
                _ => {
                    let decoded = decoder
                        .decode(&packet.data)
                        .unwrap_or_else(|e| panic!("frame {index}: {e}"));
                    compare(&decoded, encoder.reconstruction());
                }
            }
        }
    }

    #[test]
    fn resolution_changes_decode_exactly() {
        let sizes = [(128, 64), (72, 40), (128, 64)];
        let config = EncoderConfig {
            codec: Codec::Hevc,
            width: 128,
            height: 64,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 30,
            intra_period: 1000,
            block_log2: 4,
            key_qp_offset: None,
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal::default(),
            params: ParamOverrides::default(),
        };
        let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
        let mut decoder = Decoder::new();
        for &(width, height) in &sizes {
            encoder.set_resolution(width, height).unwrap();
            for index in 0..4 {
                let packet = encoder.encode(&panning(width, height, index));
                let key = packet.metadata.frame_type == nc_encoder::FrameType::Key;
                assert_eq!(key, index == 0);
                compare(
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

    #[test]
    fn intra_refresh_recovers_from_wrong_references() {
        const PERIOD: u32 = 3;
        let config = EncoderConfig {
            codec: Codec::Hevc,
            width: 256,
            height: 64,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 30,
            intra_period: 4,
            block_log2: 4,
            key_qp_offset: None,
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal::default(),
            params: ParamOverrides::default(),
        };
        let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
        encoder
            .set_intra_refresh(Some(nc_encoder::IntraRefresh { period: PERIOD }))
            .unwrap();
        let mut dec = Decoder::new();
        let corrupt_after = 5;
        let mut mismatched = false;
        for index in 0..20u64 {
            let frame = encoder.encode(&panning(256, 64, index as usize));
            assert_eq!(
                frame.metadata.frame_type == nc_encoder::FrameType::Key,
                index == 0
            );
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
        let mut encoder = Encoder::<HevcBackend>::new(settings).unwrap();
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
            packets.push(encoder.encode(&moving(width, height, frame)).data);
            recons.push(encoder.reconstruction().clone());
        }
        (packets, recons, encoder.rate_control_stats())
    }

    /// The recode contract (`CodecBackend::RECODE`): coding every frame
    /// twice (`debug.recode_identity`) leaves the stream and the
    /// reconstruction unchanged, across keys, a requested key, layers, the
    /// long-term reference and rate control.
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
                let mut settings = super::browser_tests::config(64, 64, 30, false);
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
                compare(a, b);
            }
        }
    }

    /// `rate.recode_overflow`: frames that would overflow a small buffer
    /// are coded again, coarser, and still decode to the reconstruction.
    #[test]
    fn overflowing_frames_are_recoded_and_decode_exactly() {
        let mut settings = super::browser_tests::config(64, 64, 30, false);
        settings.params.set("rate.recode_overflow", 1.);
        let (packets, recons, stats) = coded_sequence(settings, Some(20_000), 16);
        assert!(stats.unwrap().recoded_frames > 0);
        let mut decoder = Decoder::new();
        for (packet, recon) in packets.iter().zip(&recons) {
            compare(&decoder.decode(packet).unwrap(), recon);
        }
    }

    #[test]
    fn layered_streams_and_their_sub_layers_decode_exactly() {
        for layers in [2u8, 3] {
            let config = EncoderConfig {
                codec: Codec::Hevc,
                width: 128,
                height: 64,
                frame_rate: FrameRate { num: 30, den: 1 },
                quantizer: 30,
                intra_period: 11,
                block_log2: 6,
                key_qp_offset: None,
                temporal_layers: layers,
                signal: nc_encoder::StreamSignal::default(),
                // The fixed patterns (adaptive structure choice off).
                params: {
                    let mut params = ParamOverrides::default();
                    params.set("layers.adaptive", 0.0);
                    params
                },
            };
            let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
            let mut packets = Vec::new();
            for frame in 0..30 {
                if frame == 17 {
                    encoder.request_keyframe();
                }
                let packet = encoder.encode(&moving(128, 64, frame));
                let layer = crate::access_unit_layer(&packet.data).unwrap();
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
                        compare(&decoded, recon);
                    }
                }
            }
        }
    }
}

mod browser_tests {
    use super::*;
    use crate::HevcBackend;
    use nc_encoder::{Codec, Encoder, EncoderConfig, FrameRate, ParamOverrides};
    fn fast_parameters() -> ParamOverrides {
        let mut p = ParamOverrides::default();
        for line in include_str!("../../../../presets/hevc-fast.toml").lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if let Some((key, value)) = line.split_once('=') {
                p.set(key.trim(), value.trim().parse().unwrap());
            }
        }
        p
    }
    pub(super) fn config(width: usize, height: usize, qp: u32, fast: bool) -> EncoderConfig {
        EncoderConfig {
            codec: Codec::Hevc,
            width,
            height,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: qp,
            intra_period: 3,
            block_log2: 6,
            key_qp_offset: None,
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal::default(),
            params: if fast {
                fast_parameters()
            } else {
                ParamOverrides::default()
            },
        }
    }
    pub(super) fn pixels_equal(a: &Picture, b: &Picture) {
        for plane in PlaneIndex::ALL {
            for (y, (a, b)) in a
                .plane(plane)
                .view()
                .rows()
                .zip(b.plane(plane).view().rows())
                .enumerate()
            {
                assert_eq!(a, b, "{plane:?} row{y}");
            }
        }
    }
    #[test]
    fn all_qps_and_actual_fast_preset() {
        for qp in 0..=51 {
            for fast in [false, true] {
                let mut encoder = Encoder::<HevcBackend>::new(config(16, 16, qp, fast)).unwrap();
                let mut decoder = Decoder::new();
                for frame in 0..3 {
                    let mut source = Picture::new(16, 16, 0).unwrap();
                    for p in PlaneIndex::ALL {
                        let mut v = source.plane_mut(p).view_mut();
                        for y in 0..v.height() {
                            for (x, s) in v.row_mut(y).iter_mut().enumerate() {
                                *s = ((x * 17 + y * 31 + frame * 7) % 256) as u8;
                            }
                        }
                    }
                    let packet = encoder.encode(&source);
                    let decoded = decoder.decode(&packet.data).unwrap();
                    pixels_equal(&decoded, encoder.reconstruction());
                }
            }
        }
    }
    /// Planned key CUs with `PART_NxN`, intra transform splits down to 4x4
    /// (DST, chroma after the fourth block) and inter transform trees, in
    /// key and P frames (P-frame intra CUs try them with the full intra
    /// search), decode exactly at every QP band.
    #[test]
    fn intra_and_inter_coding_tools_decode_exactly() {
        let tools = [
            ("hevc.intra_leaf_cost", 1.0),
            ("hevc.intra_nxn", 1.0),
            ("hevc.intra_tu_depth", 2.0),
            ("hevc.inter_tu_split", 1.0),
            ("hevc.coarse_inter_intra", 0.0),
        ];
        let (w, h) = (96, 64);
        for qp in [4, 22, 32, 44] {
            let mut config = config(w, h, qp, false);
            for (name, value) in tools {
                config.params.set(name, value);
            }
            let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
            let mut decoder = Decoder::new();
            for frame in 0..4 {
                let mut source = Picture::new(w, h, 0).unwrap();
                for p in PlaneIndex::ALL {
                    let mut v = source.plane_mut(p).view_mut();
                    for y in 0..v.height() {
                        for (x, s) in v.row_mut(y).iter_mut().enumerate() {
                            // Fine texture with a moving edge: detail for
                            // NxN and splits, motion for inter trees.
                            let texture = ((x * 37) ^ (y * 59)) % 97;
                            let edge = if x + 3 * frame > w / 2 { 120 } else { 0 };
                            *s = ((texture + edge + y * 2) % 256) as u8;
                        }
                    }
                }
                let packet = encoder.encode(&source);
                let decoded = decoder.decode(&packet.data).unwrap();
                pixels_equal(&decoded, encoder.reconstruction());
            }
        }
    }
    /// Every quantizer extreme, default and fast, at one browser resolution.
    /// One test per resolution, so the three run in parallel (720p alone
    /// was most of the crate's test time).
    fn browser_resolution_and_fast_preset(w: usize, h: usize) {
        for qp in [0, 28, 51] {
            for fast in [false, true] {
                let mut encoder = Encoder::<HevcBackend>::new(config(w, h, qp, fast)).unwrap();
                let mut decoder = Decoder::new();
                for frame in 0..3 {
                    let mut source = Picture::new(w, h, 0).unwrap();
                    for p in PlaneIndex::ALL {
                        let mut v = source.plane_mut(p).view_mut();
                        for y in 0..v.height() {
                            for (x, s) in v.row_mut(y).iter_mut().enumerate() {
                                *s = (((x + frame) * 13
                                    + y * 7
                                    + (x / 64 + y / 64) * 19
                                    + p as usize * 31)
                                    % 256) as u8;
                            }
                        }
                    }
                    let packet = encoder.encode(&source);
                    let decoded = decoder
                        .decode(&packet.data)
                        .unwrap_or_else(|e| panic!("{w}x{h} qp{qp} fast{fast} frame{frame}: {e}"));
                    pixels_equal(&decoded, encoder.reconstruction());
                }
            }
        }
    }
    #[test]
    fn browser_resolution_216p_and_fast_preset() {
        browser_resolution_and_fast_preset(384, 216);
    }
    #[test]
    fn browser_resolution_360p_and_fast_preset() {
        browser_resolution_and_fast_preset(640, 360);
    }
    #[test]
    fn browser_resolution_720p_and_fast_preset() {
        browser_resolution_and_fast_preset(1280, 720);
    }
    #[test]
    fn shown_pictures_equal_decoded_ones() {
        // 36x20 is cropped from the coded 40x24 (conformance window).
        for (w, h) in [(64, 64), (36, 20)] {
            let mut encoder = Encoder::<HevcBackend>::new(config(w, h, 28, false)).unwrap();
            let (mut owned, mut borrowed) = (Decoder::new(), Decoder::new());
            for frame in 0..3 {
                let mut source = Picture::new(w, h, 0).unwrap();
                for p in PlaneIndex::ALL {
                    let mut v = source.plane_mut(p).view_mut();
                    for y in 0..v.height() {
                        for (x, s) in v.row_mut(y).iter_mut().enumerate() {
                            *s = ((x * 11 + y * 5 + frame * 9 + p as usize * 31) % 256) as u8;
                        }
                    }
                }
                let packet = encoder.encode(&source);
                let decoded = owned.decode(&packet.data).unwrap();
                borrowed.decode_shown(&packet.data).unwrap();
                let shown = borrowed.shown().unwrap();
                assert_eq!((shown.width(), shown.height()), (w, h));
                for p in PlaneIndex::ALL {
                    assert!(shown.plane(p).rows().eq(decoded.plane(p).view().rows()));
                }
            }
        }
    }
    #[test]
    fn truncated_and_mutated_packets_never_panic() {
        let mut encoder = Encoder::<HevcBackend>::new(config(16, 16, 28, true)).unwrap();
        let source = Picture::new(16, 16, 0).unwrap();
        let packet = encoder.encode(&source).data;
        for n in 0..packet.len() {
            let result = std::panic::catch_unwind(|| Decoder::new().decode(&packet[..n]));
            assert!(result.is_ok(), "truncation {n}");
            if n < packet.len() / 2 {
                assert!(result.unwrap().is_err());
            }
        }
        let mut random = 17u32;
        for _ in 0..500 {
            random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let mut bytes = packet.clone();
            let index = random as usize % bytes.len();
            bytes[index] ^= 1 << (random >> 29);
            assert!(std::panic::catch_unwind(|| Decoder::new().decode(&bytes)).is_ok());
        }
    }
}

mod external_fixture {
    use super::browser_tests::{config, pixels_equal};
    use super::*;
    use crate::HevcBackend;
    use nc_encoder::Encoder;
    use std::io::Write;
    #[test]
    #[ignore = "requires explicitly provided 720p TRAIN I420 fixture and output directory"]
    fn real_train_round_trip() {
        let input = std::env::var_os("NC_HEVC_DECODER_SOURCE").expect("TRAIN source");
        let output = std::path::PathBuf::from(
            std::env::var_os("NC_HEVC_DECODER_OUTPUT").expect("output directory"),
        );
        std::fs::create_dir_all(&output).unwrap();
        for fast in [false, true] {
            let label = if fast { "fast" } else { "default" };
            let mut reader = std::io::BufReader::new(std::fs::File::open(&input).unwrap());
            let mut stream = std::fs::File::create(output.join(format!("{label}.hevc"))).unwrap();
            let mut raw = std::fs::File::create(output.join(format!("{label}.yuv"))).unwrap();
            let mut encoder = Encoder::<HevcBackend>::new(config(1280, 720, 28, fast)).unwrap();
            let mut decoder = Decoder::new();
            for frame in 0..3 {
                let mut source = Picture::new(1280, 720, 0).unwrap();
                assert!(nc_common::yuv::read_frame(&mut reader, &mut source).unwrap());
                let packet = encoder.encode(&source);
                stream.write_all(&packet.data).unwrap();
                let decoded = decoder
                    .decode(&packet.data)
                    .unwrap_or_else(|e| panic!("{label} frame{frame}: {e}"));
                pixels_equal(&decoded, encoder.reconstruction());
                nc_common::yuv::write_frame(&mut raw, &decoded).unwrap();
            }
        }
    }
}

mod skipped_transform_regression {
    use super::*;
    use crate::HevcBackend;
    use nc_encoder::{Codec, Encoder, EncoderConfig, FrameRate, ParamOverrides};
    #[test]
    fn skipped_region_does_not_alias_first_coded_transform() {
        // Before the uncoded-TU sentinel this two-frame stream differs in69
        // luma samples: the first residual TU aliases the preceding skipped CU.
        let config = EncoderConfig {
            codec: Codec::Hevc,
            width: 64,
            height: 64,
            frame_rate: FrameRate { num: 30, den: 1 },
            quantizer: 39,
            intra_period: 300,
            block_log2: 4,
            key_qp_offset: None,
            temporal_layers: 1,
            signal: nc_encoder::StreamSignal::default(),
            params: ParamOverrides::default(),
        };
        let mut encoder = Encoder::<HevcBackend>::new(config).unwrap();
        let mut decoder = Decoder::new();
        let mut first = Picture::new(64, 64, 0).unwrap();
        for plane in PlaneIndex::ALL {
            let mut view = first.plane_mut(plane).view_mut();
            for y in 0..view.height() {
                view.row_mut(y).fill(128);
            }
        }
        decoder.decode(&encoder.encode(&first).data).unwrap();
        let mut second = encoder.reconstruction().clone();
        {
            let mut view = second.plane_mut(PlaneIndex::Y).view_mut();
            for y in 0..64 {
                view.row_mut(y)[16..64].fill(152);
            }
        }
        let packet = encoder.encode(&second);
        let decoded = decoder.decode(&packet.data).unwrap();
        for plane in PlaneIndex::ALL {
            for (a, b) in decoded
                .plane(plane)
                .view()
                .rows()
                .zip(encoder.reconstruction().plane(plane).view().rows())
            {
                assert_eq!(a, b);
            }
        }
    }
}
