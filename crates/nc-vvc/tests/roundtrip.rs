//! Engine integration, bounded subset decoding and unsupported configurations.
use nc_common::{Picture, PlaneIndex, yuv};
use nc_encoder::{
    Codec, CodecBackend, Encoder, EncoderConfig, FrameRate, FrameType, ParamOverrides,
    RateControlConfig, StreamSignal,
};
use nc_vvc::{Decoder, VvcBackend};

fn config(width: usize, height: usize, quantizer: u32) -> EncoderConfig {
    EncoderConfig {
        codec: Codec::Vvc,
        width,
        height,
        quantizer,
        frame_rate: FrameRate { num: 30, den: 1 },
        intra_period: 1,
        temporal_layers: 1,
        block_log2: 5,
        key_qp_offset: None,
        signal: StreamSignal::default(),
        // Some tests code repeated pictures.
        params: ParamOverrides::default().coding_every_frame(),
    }
}
fn source(w: usize, h: usize, frame: usize) -> Picture {
    let mut p = Picture::new(w, h, 0).unwrap();
    for plane in PlaneIndex::ALL {
        let mut view = p.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            for (x, s) in view.row_mut(y).iter_mut().enumerate() {
                *s = match frame {
                    0 => 0,
                    1 => 255,
                    _ => ((x * 23 + y * 7 + frame * 31 + plane as usize * 53) ^ (x * y)) as u8,
                };
            }
        }
    }
    p
}
fn bytes(p: &Picture) -> Vec<u8> {
    let mut out = Vec::new();
    yuv::write_frame(&mut out, p).unwrap();
    out
}

#[test]
fn shared_engine_roundtrips_crops_extremes_and_quantizers() {
    assert!(!std::hint::black_box(VvcBackend::OWN_PICTURE_PIPELINE));
    for (w, h) in [(2, 2), (32, 32), (34, 66), (128, 96)] {
        for qp in [0, 1, 22, 37, 51, 63] {
            let mut encoder = Encoder::<VvcBackend>::new(config(w, h, qp)).unwrap();
            let mut decoder = Decoder::new();
            for frame in 0..4 {
                let packet = encoder.encode(&source(w, h, frame));
                assert_eq!(packet.metadata.frame_type, FrameType::Key);
                let decoded = decoder.decode(&packet.data).unwrap();
                assert_eq!(
                    bytes(&decoded),
                    bytes(encoder.reconstruction()),
                    "{w}x{h} QP{qp} frame{frame}"
                );
                // Every packet is an independently decodable IDR.
                assert_eq!(
                    bytes(&decoded),
                    bytes(&Decoder::new().decode(&packet.data).unwrap())
                );
            }
        }
    }
}

#[test]
fn residuals_represent_source_and_qp() {
    let mut encoder = Encoder::<VvcBackend>::new(config(32, 32, 0)).unwrap();
    let black = source(32, 32, 0);
    let white = source(32, 32, 1);
    let a = encoder.encode(&black);
    assert_eq!(bytes(encoder.reconstruction()), bytes(&black));
    let b = encoder.encode(&white);
    assert_eq!(bytes(encoder.reconstruction()), bytes(&white));
    assert_ne!(a.data, b.data);
    let mut coarse = Encoder::<VvcBackend>::new(config(32, 32, 63)).unwrap();
    assert_ne!(a.data, coarse.encode(&black).data);
}

#[test]
fn unsupported_configurations_fail_before_encoding() {
    let mut c = config(64, 64, 64);
    assert!(Encoder::<VvcBackend>::new(c.clone()).is_err());
    c.quantizer = 32;
    c.temporal_layers = 4;
    assert!(Encoder::<VvcBackend>::new(c.clone()).is_err());
    c.temporal_layers = 1;
    c.block_log2 = 2;
    assert!(Encoder::<VvcBackend>::new(c.clone()).is_err());
    c.block_log2 = 5;
    c.width = 63;
    assert!(Encoder::<VvcBackend>::new(c.clone()).is_err());
    c.width = 4098;
    assert!(Encoder::<VvcBackend>::new(c.clone()).is_err());
    c.width = 64;
    c.codec = Codec::Hevc;
    assert!(Encoder::<VvcBackend>::new(c).is_err());
}

#[test]
fn truncated_packets_are_rejected_without_panicking() {
    let mut encoder = Encoder::<VvcBackend>::new(config(64, 64, 22)).unwrap();
    let packet = encoder.encode(&source(64, 64, 2)).data;
    for end in 0..packet.len() {
        assert!(
            Decoder::new().decode(&packet[..end]).is_err(),
            "accepted prefix {end}"
        );
    }
    for len in 0..128 {
        let garbage: Vec<_> = (0..len).map(|i| (i * 71 + len * 17) as u8).collect();
        assert!(Decoder::new().decode(&garbage).is_err());
    }
}

#[test]
fn shared_rate_control_and_resolution_change_remain_decodable() {
    let mut c = config(64, 64, 32);
    c.intra_period = 30;
    let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
    encoder
        .set_rate_control(RateControlConfig::new(30_000))
        .unwrap();
    let mut decoder = Decoder::new();
    for frame in 0..8 {
        let packet = encoder.encode(&source(64, 64, frame));
        assert_eq!(
            bytes(&decoder.decode(&packet.data).unwrap()),
            bytes(encoder.reconstruction())
        );
        assert!(encoder.rate_control_stats().is_some());
    }
    encoder.set_resolution(66, 34).unwrap();
    let packet = encoder.encode(&source(66, 34, 2));
    assert_eq!(
        bytes(&decoder.decode(&packet.data).unwrap()),
        bytes(encoder.reconstruction())
    );
}

#[test]
fn shared_inter_selection_poc_wrap_and_key_recovery() {
    let mut c = config(64, 64, 22);
    c.intra_period = 1000;
    let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
    let mut decoder = Decoder::new();
    let still = source(64, 64, 0);
    let key = encoder.encode(&still);
    decoder.decode(&key.data).unwrap();
    for frame in 1..260 {
        let packet = encoder.encode(&still);
        assert_eq!(packet.metadata.frame_type, FrameType::Inter);
        assert!(packet.data.len() < key.data.len());
        assert_eq!(
            bytes(&decoder.decode(&packet.data).unwrap()),
            bytes(encoder.reconstruction())
        );
        if frame == 1 {
            assert!(Decoder::new().decode(&packet.data).is_err());
        }
    }
    encoder.request_keyframe();
    let key = encoder.encode(&still);
    assert_eq!(key.metadata.frame_type, FrameType::Key);
    assert_eq!(
        bytes(&Decoder::new().decode(&key.data).unwrap()),
        bytes(encoder.reconstruction())
    );
    assert_eq!(
        bytes(&decoder.decode(&key.data).unwrap()),
        bytes(encoder.reconstruction())
    );
}

#[test]
fn ac_residuals_preserve_texture_at_low_qp() {
    for log2 in [3, 4, 5] {
        let mut c = config(64, 64, 0);
        c.block_log2 = log2;
        let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
        let input = source(64, 64, 2);
        let packet = encoder.encode(&input);
        let decoded = Decoder::new().decode(&packet.data).unwrap();
        let a = bytes(&input);
        let b = bytes(&decoded);
        let sse: u64 = a
            .iter()
            .zip(&b)
            .map(|(&a, &b)| {
                let e = i32::from(a) - i32::from(b);
                (e * e) as u64
            })
            .sum();
        assert!(
            sse < 10 * a.len() as u64,
            "texture distortion log2{log2}: {sse}"
        );
    }
}

#[test]
fn key_offset_matches_direct_quantizer_and_clamps() {
    for (qp, offset, effective) in [(37, 9, 28), (0, i32::MAX, 0), (63, i32::MIN, 63)] {
        let mut cfg = config(32, 32, qp);
        cfg.key_qp_offset = Some(offset);
        let mut adjusted = Encoder::<VvcBackend>::new(cfg).unwrap();
        let mut direct_cfg = config(32, 32, effective);
        direct_cfg.key_qp_offset = Some(0);
        let mut direct = Encoder::<VvcBackend>::new(direct_cfg).unwrap();
        let picture = source(32, 32, 2);
        assert_eq!(adjusted.encode(&picture).data, direct.encode(&picture).data);
    }
}

#[test]
fn angular_search_budget_is_validated_and_roundtrips() {
    for budget in [0., 8., 63.] {
        let mut cfg = config(34, 34, 37);
        cfg.params.set("vvc.angular_candidates", budget);
        cfg.intra_period = 3;
        cfg.block_log2 = 3;
        let mut encoder = Encoder::<VvcBackend>::new(cfg).unwrap();
        let mut decoder = Decoder::new();
        for frame in 2..5 {
            let packet = encoder.encode(&source(34, 34, frame));
            assert_eq!(
                bytes(&decoder.decode(&packet.data).unwrap()),
                bytes(encoder.reconstruction())
            );
        }
    }
    for invalid in [-1., 64., f32::NAN] {
        let mut cfg = config(32, 32, 37);
        cfg.params.set("vvc.angular_candidates", invalid);
        assert!(Encoder::<VvcBackend>::new(cfg).is_err());
    }
}

#[test]
fn independent_chroma_search_roundtrips_and_validates() {
    for enabled in [0., 1.] {
        let mut cfg = config(34, 34, 37);
        cfg.params.set("vvc.chroma_search", enabled);
        cfg.intra_period = 3;
        let mut encoder = Encoder::<VvcBackend>::new(cfg).unwrap();
        let mut decoder = Decoder::new();
        for frame in 0..5 {
            let packet = encoder.encode(&source(34, 34, frame));
            assert_eq!(
                bytes(&decoder.decode(&packet.data).unwrap()),
                bytes(encoder.reconstruction())
            );
        }
    }
    for invalid in [-1., 2., f32::NAN] {
        let mut cfg = config(32, 32, 37);
        cfg.params.set("vvc.chroma_search", invalid);
        assert!(Encoder::<VvcBackend>::new(cfg).is_err());
    }
}

#[test]
fn ctu_size_is_validated_and_64_codes_flat_pictures_in_fewer_bytes() {
    // A flat picture of whole 64x64 CTUs: CTU 64 codes a quarter of the
    // CUs and SAO parameter sets, so its key frame is smaller.
    let flat = source(128, 128, 0);
    let key_bytes = |ctu: f32| {
        let mut cfg = config(128, 128, 37);
        cfg.params.set("vvc.ctu_size", ctu);
        cfg.block_log2 = (ctu as usize).ilog2() as u8;
        let mut encoder = Encoder::<VvcBackend>::new(cfg).unwrap();
        let packet = encoder.encode(&flat);
        assert_eq!(
            bytes(&Decoder::new().decode(&packet.data).unwrap()),
            bytes(encoder.reconstruction())
        );
        packet.data.len()
    };
    assert!(key_bytes(64.) < key_bytes(32.));
    for invalid in [16., 48., 128.] {
        let mut cfg = config(32, 32, 37);
        cfg.params.set("vvc.ctu_size", invalid);
        assert!(Encoder::<VvcBackend>::new(cfg).is_err());
    }
}

#[test]
fn large_transforms_roundtrip_and_code_smooth_pictures_in_fewer_bytes() {
    // A smooth gradient of whole 64x64 CTUs moving by a sample a frame:
    // with `vvc.tx64` its 64x64 CUs code one 64-point transform of 32x32
    // low frequencies (and 32x32 chroma) instead of four 32x32 ones.
    let gradient = |frame: usize| {
        let mut p = Picture::new(128, 128, 0).unwrap();
        for plane in PlaneIndex::ALL {
            let mut view = p.plane_mut(plane).view_mut();
            for y in 0..view.height() {
                for (x, s) in view.row_mut(y).iter_mut().enumerate() {
                    *s = (40 + usize::midpoint(x, frame) + y / 3 + 20 * plane as usize) as u8;
                }
            }
        }
        p
    };
    let stream_bytes = |tx64: f32, ctu: f32, qp: u32| {
        let mut cfg = config(128, 128, qp);
        cfg.intra_period = 30;
        cfg.params.set("vvc.ctu_size", ctu);
        cfg.params.set("vvc.tx64", tx64);
        cfg.block_log2 = (ctu as usize).ilog2() as u8;
        let mut encoder = Encoder::<VvcBackend>::new(cfg).unwrap();
        let mut decoder = Decoder::new();
        let mut total = 0;
        for frame in 0..3 {
            let packet = encoder.encode(&gradient(frame));
            assert_eq!(
                bytes(&decoder.decode(&packet.data).unwrap()),
                bytes(encoder.reconstruction()),
                "tx64 {tx64} ctu {ctu} qp {qp} frame {frame}"
            );
            total += packet.data.len();
        }
        total
    };
    for qp in [4, 22, 37, 51] {
        assert!(
            stream_bytes(1., 64., qp) < stream_bytes(0., 64., qp),
            "qp {qp}"
        );
    }
    // 32x32 CTUs have no 64-point transforms: the parameter changes nothing.
    assert_eq!(stream_bytes(1., 32., 37), stream_bytes(0., 32., 37));
}

#[test]
fn shown_pictures_equal_decoded_ones() {
    // 66x34 crops through the conformance window; recycled pictures (the
    // replaced references) carry stale samples into every later picture.
    let mut c = config(66, 34, 30);
    c.intra_period = 6;
    let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
    let (mut owned, mut shown) = (Decoder::new(), Decoder::new());
    for frame in 2..16 {
        let packet = encoder.encode(&source(66, 34, frame));
        let decoded = owned.decode(&packet.data).unwrap();
        assert_eq!(
            bytes(&decoded),
            bytes(encoder.reconstruction()),
            "frame {frame}"
        );
        shown.decode_shown(&packet.data).unwrap();
        let view = shown.shown().unwrap();
        let mut copy = Picture::new(view.width(), view.height(), 0).unwrap();
        view.copy_to(&mut copy);
        assert_eq!(bytes(&copy), bytes(&decoded), "frame {frame}");
    }
    assert!(shown.decode_shown(&[0, 0, 1]).is_err());
    assert!(shown.shown().is_none(), "a failed decode shows nothing");
}

/// A picture whose left half is flat and right half busy, so variance AQ
/// gives its CTBs different quantizers (`cu_qp_delta`).
fn split_texture(w: usize, h: usize, frame: usize) -> Picture {
    let mut p = source(w, h, frame + 2);
    for plane in PlaneIndex::ALL {
        let mut view = p.plane_mut(plane).view_mut();
        let half = view.width() / 2;
        for y in 0..view.height() {
            view.row_mut(y)[..half].fill(100 + (y / 16) as u8);
        }
    }
    p
}

#[test]
fn block_qp_streams_reconstruct_exactly_and_differ() {
    let (w, h) = (192, 136);
    for (qp, intra_period) in [(4, 1), (30, 3), (49, 3)] {
        let mut sizes = Vec::new();
        for strength in [0., 3.] {
            let mut c = config(w, h, qp);
            c.intra_period = intra_period;
            c.params.set("rate.aq_strength", strength);
            c.params.set("rate.aq_max_offset", 12.);
            let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
            let mut decoder = Decoder::new();
            let mut total = 0;
            for frame in 0..4 {
                let packet = encoder.encode(&split_texture(w, h, frame));
                total += packet.data.len();
                let decoded = decoder.decode(&packet.data).unwrap();
                assert_eq!(
                    bytes(&decoded),
                    bytes(encoder.reconstruction()),
                    "QP{qp} AQ {strength} frame {frame}"
                );
            }
            sizes.push(total);
        }
        assert_ne!(sizes[0], sizes[1], "QP{qp}: AQ changed nothing");
    }
}

#[test]
fn temporal_layers_decode_every_extracted_substream() {
    for layers in [2, 3] {
        for period in [6, 300] {
            let mut c = config(64, 64, 32);
            c.temporal_layers = layers;
            c.intra_period = period;
            let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
            let mut decoders: Vec<_> = (0..layers).map(|_| Decoder::new()).collect();
            let frames = if period == 300 { 262 } else { 15 };
            for frame in 0..frames {
                let packet = encoder.encode(&source(64, 64, frame + 2));
                let layer = nc_vvc::access_unit_layer(&packet.data).unwrap();
                assert!(layer < layers);
                for (maximum, decoder) in decoders.iter_mut().enumerate() {
                    if usize::from(layer) <= maximum {
                        let decoded = decoder.decode(&packet.data).unwrap_or_else(|e| {
                            panic!("L{layers} period{period} frame{frame} max{maximum}: {e}")
                        });
                        assert_eq!(
                            bytes(&decoded),
                            bytes(encoder.reconstruction()),
                            "L{layers} period{period} frame{frame} max{maximum}"
                        );
                    }
                }
            }
        }
    }
}
