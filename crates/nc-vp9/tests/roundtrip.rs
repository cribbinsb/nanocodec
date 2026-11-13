//! The VP9 backend through the shared engine, decoded by the subset
//! decoder: reconstruction identity across geometries, quantizers, key and
//! inter frames, rate control and dropped frames.
use nc_common::{Picture, PlaneIndex, yuv};
use nc_encoder::{
    Codec, Encoder, EncoderConfig, FrameRate, FrameType, ParamOverrides, RateControlConfig,
    StreamSignal,
};
use nc_vp9::Vp9Backend;
use nc_vp9::stream::Decoder;

fn config(width: usize, height: usize, quantizer: u32) -> EncoderConfig {
    EncoderConfig {
        codec: Codec::Vp9,
        width,
        height,
        frame_rate: FrameRate { num: 30, den: 1 },
        quantizer,
        intra_period: 30,
        temporal_layers: 1,
        block_log2: 4,
        key_qp_offset: None,
        signal: StreamSignal::default(),
        // Some tests code repeated pictures.
        params: ParamOverrides::default().coding_every_frame(),
    }
}

fn bytes(picture: &Picture) -> Vec<u8> {
    let mut out = Vec::new();
    yuv::write_frame(&mut out, picture).unwrap();
    out
}

/// A textured picture moving `frame` samples right and half as many down.
fn moving(width: usize, height: usize, frame: usize) -> Picture {
    let mut picture = Picture::new(width, height, 0).unwrap();
    for plane in PlaneIndex::ALL {
        let shift = usize::from(plane != PlaneIndex::Y);
        let mut view = picture.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            for (x, sample) in view.row_mut(y).iter_mut().enumerate() {
                let (u, v) = (x + (frame >> shift), y + (frame >> (shift + 1)));
                *sample = ((u * 7 + v * 3) ^ (u * v / 5) ^ (plane as usize * 40)) as u8;
            }
        }
    }
    picture
}

fn assert_round_trip(
    encoder: &mut Encoder<Vp9Backend>,
    decoder: &mut Decoder,
    source: &Picture,
) -> FrameType {
    let encoded = encoder.encode(source);
    let decoded = decoder.decode(&encoded.data).unwrap();
    assert_eq!(bytes(&decoded), bytes(encoder.reconstruction()));
    encoded.metadata.frame_type
}

#[test]
fn reconstruction_matches_across_geometry_and_quantizers() {
    // 74x42 and 18x10 are not whole 8x8 units: the decoder crops.
    for (width, height) in [(8, 8), (72, 40), (74, 42), (18, 10), (128, 72), (200, 136)] {
        for quantizer in [1, 60, 140, 255] {
            let mut encoder = Encoder::<Vp9Backend>::new(config(width, height, quantizer)).unwrap();
            let mut decoder = Decoder::new();
            for frame in 0..4 {
                let kind =
                    assert_round_trip(&mut encoder, &mut decoder, &moving(width, height, frame));
                assert_eq!(kind == FrameType::Key, frame == 0);
            }
        }
    }
}

#[test]
fn shown_frames_equal_decoded_ones() {
    // 74x42 is cropped from whole 8x8 units.
    for (width, height) in [(72, 40), (74, 42)] {
        let mut encoder = Encoder::<Vp9Backend>::new(config(width, height, 100)).unwrap();
        let (mut owned, mut borrowed) = (Decoder::new(), Decoder::new());
        for frame in 0..4 {
            let encoded = encoder.encode(&moving(width, height, frame));
            let decoded = owned.decode(&encoded.data).unwrap();
            borrowed.decode_shown(&encoded.data).unwrap();
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
fn lossless_frames_are_exact() {
    let mut encoder = Encoder::<Vp9Backend>::new(config(72, 40, 0)).unwrap();
    let mut decoder = Decoder::new();
    for frame in 0..3 {
        let source = moving(72, 40, frame);
        assert_round_trip(&mut encoder, &mut decoder, &source);
        assert_eq!(bytes(encoder.reconstruction()), bytes(&source));
    }
}

#[test]
fn forced_keys_and_rate_control_round_trip() {
    let mut encoder = Encoder::<Vp9Backend>::new(config(128, 72, 120)).unwrap();
    let mut rc = RateControlConfig::new(200_000);
    rc.frame_drop = true;
    encoder.set_rate_control(rc).unwrap();
    let mut decoder = Decoder::new();
    for frame in 0..12 {
        if frame == 6 {
            encoder.request_keyframe();
        }
        let encoded = encoder.encode(&moving(128, 72, frame * 3));
        if encoded.metadata.dropped {
            continue;
        }
        let decoded = decoder.decode(&encoded.data).unwrap();
        assert_eq!(bytes(&decoded), bytes(encoder.reconstruction()));
        assert_eq!(
            encoded.metadata.frame_type == FrameType::Key,
            frame == 0 || frame == 6
        );
    }
}

#[test]
fn rejects_unsupported_configurations_and_malformed_packets() {
    let mut odd = config(72, 41, 100);
    assert!(Encoder::<Vp9Backend>::new(odd.clone()).is_err());
    odd.height = 40;
    odd.params.set("vp9.unknown_tool", 1.0);
    assert!(Encoder::<Vp9Backend>::new(odd).is_err());
    let mut encoder = Encoder::<Vp9Backend>::new(config(72, 40, 100)).unwrap();
    let key = encoder.encode(&moving(72, 40, 0)).data;
    let inter = encoder.encode(&moving(72, 40, 1)).data;
    assert!(
        Decoder::new().decode(&inter).is_err(),
        "inter frame before key frame"
    );
    for length in 0..key.len() {
        let _ = Decoder::new().decode(&key[..length]);
    }
}

/// Packets, reconstructions and rate-control statistics of a sequence.
type CodedSequence = (
    Vec<Vec<u8>>,
    Vec<Vec<u8>>,
    Option<nc_encoder::RateControlStats>,
);

/// Packets and reconstructions of 16 moving 64x48 frames at qindex 120
/// (key every 7, a requested key at frame 9) with `layers` temporal layers
/// and `params`, under rate control at `bitrate` (bits/s, 200 ms buffer)
/// if given.
fn coded_sequence(layers: u8, params: &[(&str, f32)], bitrate: Option<u64>) -> CodedSequence {
    coded_sequence_of(moving, (64, 48), layers, params, bitrate)
}

/// [`coded_sequence`] of `source(width, height, frame)` pictures.
fn coded_sequence_of(
    source: fn(usize, usize, usize) -> Picture,
    (width, height): (usize, usize),
    layers: u8,
    params: &[(&str, f32)],
    bitrate: Option<u64>,
) -> CodedSequence {
    let mut settings = config(width, height, 120);
    settings.intra_period = 7;
    settings.temporal_layers = layers;
    for &(name, value) in params {
        settings.params.set(name, value);
    }
    let mut encoder = Encoder::<Vp9Backend>::new(settings).unwrap();
    if let Some(bps) = bitrate {
        let mut rate = RateControlConfig::new(bps);
        rate.buffer_ms = 200;
        encoder.set_rate_control(rate).unwrap();
    }
    let (mut packets, mut recons) = (Vec::new(), Vec::new());
    for frame in 0..16 {
        if frame == 9 {
            encoder.request_keyframe();
        }
        packets.push(encoder.encode(&source(width, height, frame)).data);
        recons.push(bytes(encoder.reconstruction()));
    }
    (packets, recons, encoder.rate_control_stats())
}

/// The recode contract (`CodecBackend::RECODE`): coding every frame twice
/// (`debug.recode_identity`, the first time at the other end of the
/// quantizer range) leaves the stream and the reconstruction unchanged,
/// across keys, a requested key, error-resilient layered frames, the
/// long-term reference, backward adaptation, the skip segment (whose map
/// and features later frames continue from) and rate control.
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
            coded_sequence(
                layers,
                &[
                    ("debug.recode_identity", identity),
                    ("ltr.interval", ltr_interval),
                    ("vp9.skip_filter_off", 1.),
                ],
                bitrate,
            )
        };
        let (once, twice) = (coded(0.), coded(1.));
        let case = format!("layers {layers} ltr {ltr_interval} rate {bitrate:?}");
        assert_eq!(once.0, twice.0, "{case}");
        assert_eq!(once.1, twice.1, "{case}");
    }
}

/// `rate.recode_overflow`: frames that would overflow a small buffer are
/// coded again, coarser, and still decode to the reconstruction.
#[test]
fn overflowing_frames_are_recoded_and_decode_exactly() {
    let (packets, recons, stats) = coded_sequence(1, &[("rate.recode_overflow", 1.)], Some(20_000));
    assert!(stats.unwrap().recoded_frames > 0);
    let mut decoder = Decoder::new();
    for (packet, recon) in packets.iter().zip(&recons) {
        assert_eq!(&bytes(&decoder.decode(packet).unwrap()), recon);
    }
}

/// Two superblocks wide, so block AQ has a flat and a detailed square.
const AQ_SIZE: (usize, usize) = (128, 64);

/// [`moving`] with its left superblock column flat in luma: AQ gives the
/// two columns offsets a grid step apart.
fn half_flat(width: usize, height: usize, frame: usize) -> Picture {
    const FLAT_COLUMNS: usize = 64;
    const FLAT_LUMA: u8 = 128;
    let mut picture = moving(width, height, frame);
    let mut luma = picture.plane_mut(PlaneIndex::Y).view_mut();
    for y in 0..luma.height() {
        luma.row_mut(y)[..FLAT_COLUMNS].fill(FLAT_LUMA);
    }
    picture
}

/// Block quantizers through segmentation (`SEG_LVL_ALT_Q`): with AQ the
/// stream differs from frame-QP coding, every frame decodes to the
/// reconstruction across keys, layers and rate control, and coding every
/// frame twice changes nothing.
#[test]
fn block_quantizer_segments_decode_exactly() {
    let aq = [("rate.aq_strength", 3.), ("rate.aq_max_offset", 12.)];
    let coded = |params: &[(&str, f32)], layers, bitrate| {
        coded_sequence_of(half_flat, AQ_SIZE, layers, params, bitrate)
    };
    for (layers, bitrate) in [(1, None), (2, None), (1, Some(150_000))] {
        let (packets, recons, _) = coded(&aq, layers, bitrate);
        let (plain, _, _) = coded(&[], layers, bitrate);
        let case = format!("layers {layers} rate {bitrate:?}");
        assert_ne!(packets, plain, "{case}");
        let mut decoder = Decoder::new();
        for (packet, recon) in packets.iter().zip(&recons) {
            assert_eq!(&bytes(&decoder.decode(packet).unwrap()), recon, "{case}");
        }
        let recoded = coded(
            &[aq[0], aq[1], ("debug.recode_identity", 1.)],
            layers,
            bitrate,
        );
        assert_eq!(packets, recoded.0, "{case}");
    }
}

/// Exercise `Vp9Backend` spatial SVC (`L2T1`, `L2T2`, `L2T3`, `spatial.inter_layer = 0`,
/// and sparse enhancement `1.5x` / `1.25x` / `2x`), Annex B superframe packing,
/// base-layer stripping (`strip_superframe_to_base`), per-layer sidecar
/// operating-point droppability, and `Decoder` (`decode_with_layer`,
/// `decode_all_layers`, `set_base_only`, `max_frame_size`).
#[test]
fn spatial_backend_superframe_and_stripped_base_round_trip() {
    use nc_encoder::CodecBackend;
    use nc_vp9::strip_superframe_to_base;

    // Scale > 2x (e.g. base_scale = 0.25 -> 4x) is rejected by Vp9Backend (verified up to 2:1).
    let mut bad_cfg = config(128, 96, 110);
    bad_cfg.params.set("spatial.layers", 2.0);
    bad_cfg.params.set("spatial.base_scale", 0.25);
    assert!(Encoder::<Vp9Backend>::new(bad_cfg).is_err());

    let mut bad_backend = Vp9Backend::new(&config(96, 72, 100)).unwrap();
    let mut ok_cfg = config(128, 96, 100);
    ok_cfg.params.set("spatial.layers", 2.0);
    let mut bad_spatial = nc_encoder::SpatialConfig::from_encoder_config(&ok_cfg, true)
        .unwrap()
        .unwrap();
    bad_spatial.base_size = (32, 24);
    bad_spatial.full_size = (96, 72);
    assert!(bad_backend.configure_spatial(&bad_spatial).is_err());

    for (temporal_layers, scale, enh_interval, inter_layer) in [
        (1u8, 0.5f32, 1u32, 1.0f32),
        (2, 0.5, 1, 1.0),
        (3, 0.5, 1, 1.0),
        (1, 0.5, 1, 0.0),
        (1, 2.0 / 3.0, 3, 1.0),
        (1, 0.8, 2, 1.0),
    ] {
        let (full_w, full_h) = (160, 120);
        let mut cfg = config(full_w, full_h, 110);
        cfg.temporal_layers = temporal_layers;
        cfg.intra_period = 6;
        cfg.params.set("spatial.layers", 2.0);
        cfg.params.set("spatial.base_scale", scale);
        cfg.params
            .set("spatial.enhancement_interval", enh_interval as f32);
        cfg.params.set("spatial.inter_layer", inter_layer);
        cfg.params.set("layers.adaptive", 0.0);

        let mut encoder = Encoder::<Vp9Backend>::new(cfg).unwrap();
        let mut full_dec = Decoder::new();
        let mut base_dec = Decoder::new();
        base_dec.set_base_only(true);
        let mut stripped_dec = Decoder::new();
        let mut all_dec = Decoder::new();
        let mut full_packets = Vec::new();
        let mut stripped_packets = Vec::new();
        let mut expected_full_bytes = Vec::new();
        let mut expected_base_bytes = Vec::new();

        for tu in 0..8usize {
            let encoded = encoder.encode(&moving(full_w, full_h, tu * 2));
            let enh_due = (tu as u32).is_multiple_of(enh_interval);
            let s0_recon = encoder.layer_reconstruction(0).unwrap();
            assert_eq!(encoded.layers.len(), if enh_due { 2 } else { 1 });
            assert!(encoded.layers[0].included_in_operating_point(0));
            assert!(encoded.layers[0].included_in_operating_point(temporal_layers));
            if enh_due {
                assert!(encoded.layers[1].droppable);
                assert!(encoded.layers[1].included_in_operating_point(0));
                assert!(encoded.layers[1].droppable_at_operating_point(temporal_layers));
            }

            let (base_pic, base_sid) = base_dec.decode_with_layer(&encoded.data).unwrap().unwrap();
            assert_eq!(base_sid, 0);
            assert_eq!(bytes(&base_pic), bytes(s0_recon));
            assert_eq!(base_dec.max_frame_size(), Some((full_w, full_h)));

            let stripped = strip_superframe_to_base(&encoded.data).unwrap();
            let stripped_pic = stripped_dec.decode(&stripped).unwrap();
            assert_eq!(bytes(&stripped_pic), bytes(s0_recon));
            stripped_packets.push(stripped);
            expected_base_bytes.extend_from_slice(&bytes(s0_recon));

            let (full_pic, full_sid) = full_dec.decode_with_layer(&encoded.data).unwrap().unwrap();
            let all = all_dec.decode_all_layers(&encoded.data).unwrap();
            assert_eq!(all.len(), encoded.layers.len());
            assert_eq!(all[0].spatial_id, 0);
            assert_eq!(bytes(&all[0].picture), bytes(s0_recon));

            if enh_due {
                let s1_recon = encoder.layer_reconstruction(1).unwrap();
                assert_eq!(full_sid, 1);
                assert_eq!(bytes(&full_pic), bytes(s1_recon));
                assert_eq!(all[1].spatial_id, 1);
                assert_eq!(bytes(&all[1].picture), bytes(s1_recon));
                expected_full_bytes.extend_from_slice(&bytes(s1_recon));
            } else {
                assert_eq!(full_sid, 0);
                assert_eq!(bytes(&full_pic), bytes(s0_recon));
                expected_full_bytes.extend_from_slice(&bytes(s0_recon));
            }
            full_packets.push(encoded.data);
        }

        let base_w = encoder.layer_reconstruction(0).unwrap().width();
        let base_h = encoder.layer_reconstruction(0).unwrap().height();
        verify_external_vp9_decoders(&stripped_packets, (base_w, base_h), &expected_base_bytes);
        verify_external_vp9_decoders(&full_packets, (full_w, full_h), &expected_full_bytes);
    }
}

fn verify_external_vp9_decoders(
    packets: &[Vec<u8>],
    (width, height): (usize, usize),
    expected_i420: &[u8],
) {
    use std::io::Write;
    use std::path::Path;
    use std::process::{Command, Stdio};

    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"VP90");
    ivf.extend_from_slice(&(width as u16).to_le_bytes());
    ivf.extend_from_slice(&(height as u16).to_le_bytes());
    ivf.extend_from_slice(&30u32.to_le_bytes());
    ivf.extend_from_slice(&1u32.to_le_bytes());
    ivf.extend_from_slice(&(packets.len() as u32).to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    for (idx, pkt) in packets.iter().enumerate() {
        ivf.extend_from_slice(&(pkt.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&(idx as u64).to_le_bytes());
        ivf.extend_from_slice(pkt);
    }

    let vpxdec = Path::new("/home/mark/nanocodec-data/anchors/x86_64/bin/vpxdec");
    if vpxdec.is_file() {
        let mut child = Command::new(vpxdec)
            .args([
                "--codec=vp9",
                "--rawvideo",
                "--i420",
                "--threads=1",
                "-o",
                "-",
                "-",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(&ivf).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "vpxdec failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(out.stdout, expected_i420, "vpxdec output mismatch");
    }

    let ffmpeg = Path::new("/home/mark/nanocodec-data/anchors/x86_64/bin/ffmpeg-decode");
    if ffmpeg.is_file() {
        let mut child = Command::new(ffmpeg)
            .args([
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-err_detect",
                "explode",
                "-f",
                "ivf",
                "-c:v",
                "vp9",
                "-i",
                "pipe:0",
                "-fps_mode",
                "passthrough",
                "-threads",
                "1",
                "-noautoscale",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(&ivf).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "ffmpeg-decode failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(out.stdout, expected_i420, "ffmpeg-decode output mismatch");
    }
}
