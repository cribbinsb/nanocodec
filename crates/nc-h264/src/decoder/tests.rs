use super::*;
use nc_encoder::{Codec, Encoder, EncoderConfig, FrameRate, ParamOverrides};
fn config(width: usize, height: usize, qp: u32, params: ParamOverrides) -> EncoderConfig {
    EncoderConfig {
        codec: Codec::H264,
        width,
        height,
        frame_rate: FrameRate { num: 30, den: 1 },
        quantizer: qp,
        intra_period: 4,
        block_log2: 4,
        key_qp_offset: None,
        temporal_layers: 1,
        signal: nc_encoder::StreamSignal::default(),
        params,
    }
}
fn source(width: usize, height: usize, index: usize) -> Picture {
    let mut p = Picture::new(width, height, 0).unwrap();
    for plane in PlaneIndex::ALL {
        let mut view = p.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            for (x, v) in view.row_mut(y).iter_mut().enumerate() {
                *v = if index == 0 {
                    128
                } else {
                    ((x * 13 + y * 7 + (x * y) % 71 + index * 3) % 256) as u8
                };
            }
        }
    }
    p
}
fn equal(a: &Picture, b: &Picture) {
    assert_eq!((a.width(), a.height()), (b.width(), b.height()));
    for plane in PlaneIndex::ALL {
        let (av, bv) = (a.plane(plane).view(), b.plane(plane).view());
        for y in 0..av.height() {
            assert_eq!(av.row(y), bv.row(y), "plane{plane:?}row{y}");
        }
    }
}
#[test]
fn resolution_changes_decode_exactly() {
    // Down to a cropped size and back: new SPS/PPS (other id) at each key.
    let sizes = [(160, 64), (102, 50), (160, 64)];
    let cfg = EncoderConfig {
        intra_period: 1000,
        ..config(160, 64, 30, ParamOverrides::default())
    };
    let mut enc = Encoder::<crate::H264Backend>::new(cfg).unwrap();
    let mut dec = Decoder::new();
    for (segment, &(width, height)) in sizes.iter().enumerate() {
        enc.set_resolution(width, height).unwrap();
        for index in 0..4 {
            let frame = enc.encode(&panning(width, height, index));
            let key = frame.metadata.frame_type == nc_encoder::FrameType::Key;
            assert_eq!(key, index == 0, "segment {segment} frame {index}");
            equal(&dec.decode(&frame.data).unwrap(), enc.reconstruction());
        }
    }
}
#[test]
fn roundtrip_default_flat_and_textured_frames() {
    for qp in [0, 14, 32, 51] {
        let mut enc =
            Encoder::<crate::H264Backend>::new(config(32, 32, qp, ParamOverrides::default()))
                .unwrap();
        let mut dec = Decoder::new();
        for index in 0..7 {
            let frame = enc.encode(&source(32, 32, index));
            let pic = dec
                .decode(&frame.data)
                .unwrap_or_else(|e| panic!("qp{qp} frame{index}: {e}"));
            equal(&pic, enc.reconstruction());
        }
    }
}
#[test]
fn analysis_leaves_decoding_unchanged_and_tiles_each_picture() {
    use nc_common::stream_stats::{Pred, RecordKind, StreamStats, keys};
    let (width, height) = (48, 32);
    let mut enc =
        Encoder::<crate::H264Backend>::new(config(width, height, 28, ParamOverrides::default()))
            .unwrap();
    let mut dec = Decoder::new();
    dec.collect_stats(StreamStats::default());
    for index in 0..5 {
        let frame = enc.encode(&panning(width, height, index));
        equal(&dec.decode(&frame.data).unwrap(), enc.reconstruction());
    }
    let stats = dec.take_stats().unwrap();
    assert_eq!(stats.frames.len(), 5);
    let area = (width * height) as u64;
    for (index, frame) in stats.frames.iter().enumerate() {
        let covered: u64 = frame.tallies[keys::CB_SIZE]
            .values()
            .map(|t| t.weight)
            .sum();
        assert_eq!(covered, area, "frame {index}: macroblocks tile the picture");
        let intra = frame.get(keys::CB_PRED, Pred::Intra as i64).weight;
        assert_eq!(intra == area, index % 4 == 0, "frame {index}");
        let bits: u64 = frame.tallies[keys::CB_BITS]
            .values()
            .map(|t| t.weight)
            .sum();
        assert!(bits > 0, "frame {index}");
    }
    assert!(
        stats
            .records()
            .iter()
            .any(|r| r.kind == RecordKind::Prediction)
    );
}
#[test]
fn roundtrip_chroma_qp_offsets() {
    for offset in [-12.0, -3.0, 5.0] {
        let mut params = ParamOverrides::default();
        params.set("h264.chroma_qp_offset", offset);
        let mut enc = Encoder::<crate::H264Backend>::new(config(48, 32, 34, params)).unwrap();
        let mut dec = Decoder::new();
        for index in 0..6 {
            let frame = enc.encode(&source(48, 32, index));
            let pic = dec
                .decode(&frame.data)
                .unwrap_or_else(|e| panic!("offset{offset} frame{index}: {e}"));
            equal(&pic, enc.reconstruction());
        }
    }
}
#[test]
fn roundtrip_cropped_intra4_and_partition_options() {
    let mut params = ParamOverrides::default();
    params.set("h264.intra4x4_gate_satd", 0.0);
    params.set("h264.split_gate_satd", 0.0);
    let mut enc = Encoder::<crate::H264Backend>::new(config(50, 34, 22, params)).unwrap();
    let mut dec = Decoder::new();
    for index in 1..7 {
        let frame = enc.encode(&source(50, 34, index));
        let pic = dec
            .decode(&frame.data)
            .unwrap_or_else(|e| panic!("frame{index}:{e}"));
        equal(&pic, enc.reconstruction());
    }
}
#[test]
fn malformed_packets_return_errors() {
    let mut dec = Decoder::new();
    for bytes in [
        &[][..],
        &[0, 0, 0, 1][..],
        &[0, 0, 0, 1, 0xff][..],
        &[0, 0, 0, 1, 0x65][..],
    ] {
        assert!(dec.decode(bytes).is_err());
    }
}

#[test]
fn extreme_dc_and_fast_parameters_roundtrip() {
    for qp in [0, 51] {
        // The last picture repeats the one before.
        let mut params = ParamOverrides::default().coding_every_frame();
        params.set("h264.inter_zero_luma_scale", 4.0);
        params.set("h264.inter_zero_chroma_scale", 4.0);
        let mut enc = Encoder::<crate::H264Backend>::new(config(32, 32, qp, params)).unwrap();
        let mut dec = Decoder::new();
        for value in [0u8, 255, 0, 255, 255] {
            let mut p = Picture::new(32, 32, 0).unwrap();
            for plane in PlaneIndex::ALL {
                let mut view = p.plane_mut(plane).view_mut();
                for row in 0..view.height() {
                    view.row_mut(row).fill(value);
                }
            }
            let frame = enc.encode(&p);
            let decoded = dec.decode(&frame.data).unwrap();
            equal(&decoded, enc.reconstruction());
        }
    }
}

#[test]
fn packet_mutations_are_bounded_and_do_not_panic() {
    let mut enc =
        Encoder::<crate::H264Backend>::new(config(32, 32, 24, ParamOverrides::default())).unwrap();
    let original = enc.encode(&source(32, 32, 2)).data;
    for index in (0..original.len()).step_by((original.len() / 80).max(1)) {
        let mut corrupt = original.clone();
        corrupt[index] ^= 0xff;
        let _ = Decoder::new().decode(&corrupt);
    }
    for length in 0..original.len() / 2 {
        assert!(Decoder::new().decode(&original[..length]).is_err());
    }
    // Parameter sets after a picture must never change output dimensions after
    // reconstruction (even if a subsequent SPS is individually well formed).
    let mut reordered = original.clone();
    reordered.extend_from_slice(&original);
    assert!(Decoder::new().decode(&reordered).is_err());
}

#[test]
#[ignore = "requires NC_H264_DECODER_INPUT/OUTPUT retained external gate fixture"]
fn decode_external_standard_fixture() {
    use std::io::Write;
    let input = std::env::var("NC_H264_DECODER_INPUT").unwrap();
    let output = std::env::var("NC_H264_DECODER_OUTPUT").unwrap();
    let bytes = std::fs::read(input).unwrap();
    let mut out = std::fs::File::create(output).unwrap();
    let mut packet = Vec::new();
    let mut decoder = Decoder::new();
    let mut frames = 0;
    for unit in annex_b_units(&bytes).unwrap() {
        packet.extend_from_slice(&[0, 0, 0, 1]);
        packet.extend_from_slice(unit);
        if matches!(unit[0] & 31, 1 | 5) {
            let decoded = decoder
                .decode(&packet)
                .unwrap_or_else(|e| panic!("frame{frames}: {e}"));
            for plane in PlaneIndex::ALL {
                let view = decoded.plane(plane).view();
                for row in 0..view.height() {
                    out.write_all(view.row(row)).unwrap();
                }
            }
            packet.clear();
            frames += 1;
        }
    }
    assert!(frames > 0 && packet.is_empty());
}

#[test]
fn independent_cabac_decodes_decisions_bypass_and_termination() {
    use crate::cabac::{BinSink, CabacEncoder, SliceKind};
    use nc_common::BitWriter;
    let kinds = [
        (SliceKind::I, SliceType::I, 0),
        (SliceKind::P, SliceType::P, 0),
        (SliceKind::P, SliceType::P, 1),
        (SliceKind::P, SliceType::P, 2),
    ];
    for (kind, slice_type, init_idc) in kinds {
        for qp in [0, 24, 51] {
            let mut encoder = CabacEncoder::new(BitWriter::new(), kind, init_idc, qp);
            let mut bins = Vec::new();
            let mut rng = 0x1234_5678u32;
            for index in 0..4096 {
                rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let context = (rng as usize) % crate::tables::CONTEXT_COUNT;
                let value = rng & 0x8000_0000 != 0;
                if index % 5 == 0 {
                    encoder.encode_bypass(value);
                    bins.push((None, value));
                } else {
                    encoder.encode_decision(context, value);
                    bins.push((Some(context), value));
                }
            }
            encoder.encode_terminate(true);
            let bytes = encoder.finish();
            let mut decoder =
                cabac::Cabac::new(&bytes, slice_type, init_idc, i32::from(qp)).unwrap();
            for (context, value) in bins {
                let actual = if let Some(context) = context {
                    decoder.decision(context)
                } else {
                    decoder.bypass()
                };
                assert_eq!(actual, value);
            }
            assert!(decoder.terminate());
        }
    }
}

#[test]
fn continuous_reference_chain_crosses_two_frame_num_wraps() {
    for fast in [false, true] {
        let mut params = ParamOverrides::default();
        if fast {
            params.set("h264.inter_zero_luma_scale", 4.0);
            params.set("h264.inter_zero_chroma_scale", 4.0);
        }
        let mut settings = config(64, 48, 28, params);
        settings.intra_period = 1000;
        let mut encoder = Encoder::<crate::H264Backend>::new(settings).unwrap();
        let mut decoder = Decoder::new();
        for frame in 0..2 * crate::params::MAX_FRAME_NUM as usize + 8 {
            let mut moving = Picture::new(64, 48, 0).unwrap();
            for plane in PlaneIndex::ALL {
                let mut view = moving.plane_mut(plane).view_mut();
                let (width, height) = (view.width(), view.height());
                for row in 0..height {
                    for (col, value) in view.row_mut(row).iter_mut().enumerate() {
                        *value = ((((col + frame) % width) * 13 + ((row + 2 * frame) % height) * 7)
                            % 256) as u8;
                    }
                }
            }
            let packet = encoder.encode(&moving);
            // No key-frame reset hides a broken reference chain at the wraps.
            assert_eq!(
                packet.metadata.frame_type == nc_encoder::FrameType::Key,
                frame == 0
            );
            let decoded = decoder
                .decode(&packet.data)
                .unwrap_or_else(|error| panic!("fast={fast}, frame={frame}: {error}"));
            equal(&decoded, encoder.reconstruction());
        }
    }
}

/// A moving pattern, so inter frames carry real vectors.
fn moving(width: usize, height: usize, frame: usize) -> Picture {
    let mut p = Picture::new(width, height, 0).unwrap();
    for plane in PlaneIndex::ALL {
        let mut view = p.plane_mut(plane).view_mut();
        let (w, h) = (view.width(), view.height());
        for row in 0..h {
            for (col, value) in view.row_mut(row).iter_mut().enumerate() {
                *value = ((((col + 2 * frame) % w) * 13 + ((row + frame) % h) * 7 + col * row % 5)
                    % 256) as u8;
            }
        }
    }
    p
}

/// The bytes of 12 frames encoded with `temporal_layers` and `params`.
fn layered_bytes(temporal_layers: u8, params: ParamOverrides) -> Vec<u8> {
    let mut settings = config(64, 48, 30, params);
    settings.temporal_layers = temporal_layers;
    let mut encoder = Encoder::<crate::H264Backend>::new(settings).unwrap();
    (0..12)
        .flat_map(|frame| encoder.encode(&moving(64, 48, frame)).data)
        .collect()
}

#[test]
fn layer_count_parameter_takes_precedence_over_the_stream_setting() {
    // Fixed patterns (adaptive choice off), so that only the count differs.
    let fixed = |count: Option<f32>| {
        let mut params = ParamOverrides::default();
        params.set("layers.adaptive", 0.0);
        if let Some(count) = count {
            params.set("layers.count", count);
        }
        params
    };
    let flat = layered_bytes(1, fixed(Some(1.0)));
    let layered = layered_bytes(3, fixed(None));
    assert_ne!(layered, flat);
    // A preset turns layering on...
    assert_eq!(layered_bytes(1, fixed(Some(3.0))), layered);
    // ...or off, whatever the stream setting says.
    assert_eq!(layered_bytes(3, fixed(Some(1.0))), flat);
}

#[test]
fn layered_streams_and_their_lower_layers_decode_exactly() {
    for layers in [2u8, 3] {
        // The fixed patterns (adaptive structure choice off).
        let mut params = ParamOverrides::default();
        params.set("layers.adaptive", 0.0);
        let mut settings = config(64, 48, 30, params);
        settings.intra_period = 11; // keys interrupt mini-GOPs mid-way
        settings.temporal_layers = layers;
        let mut encoder = Encoder::<crate::H264Backend>::new(settings).unwrap();
        let mut packets = Vec::new();
        for frame in 0..30 {
            if frame == 17 {
                encoder.request_keyframe();
            }
            let packet = encoder.encode(&moving(64, 48, frame));
            let layer = crate::access_unit_layer(&packet.data, layers).unwrap();
            assert_eq!(layer, packet.metadata.temporal_layer);
            packets.push((packet, encoder.reconstruction().clone()));
        }
        // Every operating point: all layers, then without the top layer(s).
        for max_layer in (0..layers).rev() {
            let mut decoder = Decoder::new();
            let mut decoded = 0;
            for (packet, recon) in &packets {
                if packet.metadata.temporal_layer > max_layer {
                    continue;
                }
                let picture = decoder.decode(&packet.data).unwrap_or_else(|e| {
                    panic!(
                        "layers {layers} max {max_layer} frame {}: {e}",
                        packet.metadata.index
                    )
                });
                equal(&picture, recon);
                decoded += 1;
            }
            if max_layer + 1 == layers {
                assert_eq!(decoded, packets.len());
            } else {
                assert!((1..packets.len()).contains(&decoded));
            }
        }
    }
}
#[test]
fn roundtrip_per_macroblock_qp_offsets() {
    // Variance AQ gives flat macroblocks finer and textured ones coarser
    // QPs: mb_qp_delta with QP prediction, its modulo wrap near the QP
    // limits, per-macroblock dequantisation and deblocking QP averages.
    let half_flat = |index: usize| {
        let mut p = source(48, 32, index + 1);
        let mut view = p.plane_mut(PlaneIndex::Y).view_mut();
        for y in 0..view.height() {
            view.row_mut(y)[..24].fill(100 + index as u8);
        }
        p
    };
    for qp in [0, 26, 51] {
        let mut params = ParamOverrides::default();
        params.set("rate.aq_strength", 3.0);
        let mut enc = Encoder::<crate::H264Backend>::new(config(48, 32, qp, params)).unwrap();
        let mut plain =
            Encoder::<crate::H264Backend>::new(config(48, 32, qp, ParamOverrides::default()))
                .unwrap();
        let mut dec = Decoder::new();
        let mut differs = false;
        for index in 0..6 {
            let frame = enc.encode(&half_flat(index));
            differs |= frame.data != plain.encode(&half_flat(index)).data;
            let pic = dec
                .decode(&frame.data)
                .unwrap_or_else(|e| panic!("qp{qp} frame{index}: {e}"));
            equal(&pic, enc.reconstruction());
        }
        assert!(differs || qp == 0 || qp == 51, "qp{qp}: AQ changed nothing");
    }
}
/// `picture` as an NV12 buffer whose rows are `pad` bytes longer than needed.
fn nv12_padded(picture: &Picture, pad: usize) -> (Vec<u8>, usize, Vec<u8>, usize) {
    let (y_view, u_view, v_view) = (
        picture.plane(PlaneIndex::Y).view(),
        picture.plane(PlaneIndex::U).view(),
        picture.plane(PlaneIndex::V).view(),
    );
    let y_stride = y_view.width() + pad;
    let uv_stride = 2 * u_view.width() + pad;
    let mut y = vec![0xEE; y_stride * y_view.height()];
    let mut uv = vec![0xEE; uv_stride * u_view.height()];
    for row in 0..y_view.height() {
        y[row * y_stride..][..y_view.width()].copy_from_slice(y_view.row(row));
    }
    for row in 0..u_view.height() {
        let pairs = u_view.row(row).iter().zip(v_view.row(row));
        for (x, (&cb, &cr)) in pairs.enumerate() {
            uv[row * uv_stride + 2 * x] = cb;
            uv[row * uv_stride + 2 * x + 1] = cr;
        }
    }
    (y, y_stride, uv, uv_stride)
}
#[test]
fn nv12_input_codes_the_same_stream_as_i420() {
    let (width, height) = (48, 32);
    let new = || {
        Encoder::<crate::H264Backend>::new(config(width, height, 26, ParamOverrides::default()))
            .unwrap()
    };
    let (mut planar, mut camera) = (new(), new());
    let mut dec = Decoder::new();
    for index in 0..5 {
        let picture = source(width, height, index);
        let (y, y_stride, uv, uv_stride) = nv12_padded(&picture, 7);
        let frame = nc_common::RawFrame::nv12(
            nc_common::PlaneView::new(&y, y_stride, width, height).unwrap(),
            nc_common::PlaneView::new(&uv, uv_stride, width, height / 2).unwrap(),
        );
        let expected = planar.encode(&picture);
        let got = camera.encode_raw(&frame).unwrap();
        assert_eq!(got, expected, "frame {index}");
        equal(&dec.decode(&got.data).unwrap(), camera.reconstruction());
    }
    let wrong = nc_common::RawFrame::packed_i420(&[0; 24], 4, 4).unwrap();
    assert!(camera.encode_raw(&wrong).is_err());
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
        let mut cfg = config(32, 32, 30, ParamOverrides::default());
        cfg.signal.colour = colour;
        let mut enc = Encoder::<crate::H264Backend>::new(cfg).unwrap();
        let mut dec = Decoder::new();
        for index in 0..3 {
            let frame = enc.encode(&source(32, 32, index));
            equal(&dec.decode(&frame.data).unwrap(), enc.reconstruction());
        }
        assert_eq!(dec.colour(), Some(colour));
    }
}
#[test]
fn identity_matrix_is_rejected_for_420_output() {
    use nc_encoder::{ColourDescription, ColourInfo, ConfigError};
    let mut cfg = config(32, 32, 30, ParamOverrides::default());
    cfg.signal.colour = ColourInfo {
        description: Some(ColourDescription {
            matrix: nc_common::colour::matrix::IDENTITY,
            ..ColourDescription::BT709
        }),
        full_range: true,
    };
    assert_eq!(
        Encoder::<crate::H264Backend>::new(cfg).err(),
        Some(ConfigError::Colour)
    );
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
/// Frames after which a decoder with corrupted references must match the
/// encoder again: one sweep to start a new one, one to complete it.
const RECOVERY_SWEEPS: u64 = 2;
#[test]
fn intra_refresh_recovers_from_wrong_references() {
    let (width, height, period) = (160, 64, 4);
    let mut enc =
        Encoder::<crate::H264Backend>::new(config(width, height, 26, ParamOverrides::default()))
            .unwrap();
    enc.set_intra_refresh(Some(nc_encoder::IntraRefresh { period }))
        .unwrap();
    let mut dec = Decoder::new();
    let corrupt_after = 5;
    let mut mismatched = false;
    for index in 0..20u64 {
        let frame = enc.encode(&panning(width, height, index as usize));
        assert_eq!(
            frame.metadata.frame_type == nc_encoder::FrameType::Key,
            index == 0,
            "refresh replaces periodic keys"
        );
        let pic = dec.decode(&frame.data).unwrap();
        if index == corrupt_after {
            dec.corrupt_references(0);
        }
        let exact = (0..3).all(|p| {
            let p = nc_common::PlaneIndex::ALL[p];
            let (a, b) = (pic.plane(p).view(), enc.reconstruction().plane(p).view());
            (0..a.height()).all(|y| a.row(y) == b.row(y))
        });
        if index > corrupt_after + RECOVERY_SWEEPS * u64::from(period) {
            assert!(exact, "frame {index} not recovered");
        } else if index > corrupt_after {
            mismatched |= !exact;
        } else {
            assert!(exact, "frame {index}");
        }
    }
    assert!(mismatched, "the corruption had no effect");
}
/// Encode a panning texture with a long-term reference every 4 frames in
/// `temporal_layers` fixed layers; the receiver loses `lost` and reports
/// it, and must decode every other frame exactly, with no key frame after
/// the first.
fn long_term_recovery(temporal_layers: u8, lost: &[u64], frames: u64) {
    let (width, height) = (128, 64);
    let mut params = ParamOverrides::default();
    params.set("ltr.interval", 4.0);
    params.set("layers.adaptive", 0.0);
    let cfg = EncoderConfig {
        intra_period: 1000,
        temporal_layers,
        ..config(width, height, 30, params)
    };
    let mut enc = Encoder::<crate::H264Backend>::new(cfg).unwrap();
    let mut dec = Decoder::new();
    for index in 0..frames {
        let frame = enc.encode(&panning(width, height, index as usize));
        assert_eq!(
            frame.metadata.frame_type == nc_encoder::FrameType::Key,
            index == 0,
            "frame {index}: recovery frames are inter"
        );
        if lost.contains(&index) {
            enc.report_loss(index);
            continue;
        }
        let pic = dec
            .decode(&frame.data)
            .unwrap_or_else(|e| panic!("frame {index}: {e}"));
        equal(&pic, enc.reconstruction());
        enc.acknowledge(index);
    }
}
/// A receiver that loses marked frames and keys, and reports each loss
/// three frames late, never shows a picture decoded from the wrong
/// reference: every frame either decodes exactly or is refused
/// (`MissingReference`). A lost mark leaves no picture under the new
/// long-term index; a lost IDR leaves a `frame_num` gap.
fn lost_marks_and_keys_are_refused(temporal_layers: u8, lost: &[u64]) {
    let (width, height) = (128, 64);
    let mut params = ParamOverrides::default();
    params.set("ltr.interval", 4.0);
    params.set("layers.adaptive", 0.0);
    let cfg = EncoderConfig {
        intra_period: 16,
        temporal_layers,
        ..config(width, height, 30, params)
    };
    let mut enc = Encoder::<crate::H264Backend>::new(cfg).unwrap();
    let mut dec = Decoder::new();
    let mut refused = 0;
    let mut last_decoded = 0;
    for index in 0..48 {
        if let Some(&late) = lost.iter().find(|&&l| l + 3 == index) {
            enc.report_loss(late);
        }
        let frame = enc.encode(&panning(width, height, index as usize));
        if lost.contains(&index) {
            continue;
        }
        match dec.decode(&frame.data) {
            Ok(pic) => {
                equal(&pic, enc.reconstruction());
                enc.acknowledge(index);
                last_decoded = index;
            }
            Err(DecodeError::MissingReference) => refused += 1,
            Err(e) => panic!("frame {index}: {e}"),
        }
    }
    assert!(refused > 0, "no frame depended on a lost one");
    assert_eq!(last_decoded, 47, "the receiver recovered");
}
#[test]
fn lost_marks_and_keys_are_refused_in_a_flat_chain() {
    // 8 is a mark (every 4 frames once acknowledged), 16 a key.
    lost_marks_and_keys_are_refused(1, &[8, 16, 29]);
}
#[test]
fn lost_keys_in_layered_long_term_streams_are_refused() {
    // Keys at 16 and 32 are lost: the T0 pictures after them name the key
    // as long-term index 0, which the receiver holds from frame 0 or 16
    // unless the gap through frame_num 0 drops it.
    lost_marks_and_keys_are_refused(3, &[16, 17, 18, 32]);
}
#[test]
fn lost_marks_and_keys_are_refused_with_three_layers() {
    // 8 is a T0 mark, 16 a key, 28 a T0 frame.
    lost_marks_and_keys_are_refused(3, &[8, 16, 28]);
}
/// Frame 9 is lost and its report reaches the encoder only after frame
/// 11 was sent: the frames between arrive, predict from the "non-existing"
/// frame the `frame_num` gap infers (§8.2.5.2) and must be refused, not
/// decoded from the wrong picture; the recovery frame then decodes exactly.
#[test]
fn frames_after_a_loss_are_refused_until_recovery() {
    let (width, height) = (128, 64);
    let mut params = ParamOverrides::default();
    params.set("ltr.interval", 4.0);
    params.set("layers.adaptive", 0.0);
    // A flat chain: frame 9 is referenced (the backend default is layered).
    params.set("layers.count", 1.0);
    let cfg = EncoderConfig {
        intra_period: 1000,
        ..config(width, height, 30, params)
    };
    let mut enc = Encoder::<crate::H264Backend>::new(cfg).unwrap();
    let mut dec = Decoder::new();
    for index in 0..14 {
        if index == 12 {
            enc.report_loss(9);
        }
        let frame = enc.encode(&panning(width, height, index as usize));
        match index {
            9 => {} // lost
            10 | 11 => assert_eq!(
                dec.decode(&frame.data).unwrap_err(),
                DecodeError::MissingReference,
                "frame {index}"
            ),
            _ => {
                let pic = dec
                    .decode(&frame.data)
                    .unwrap_or_else(|e| panic!("frame {index}: {e}"));
                equal(&pic, enc.reconstruction());
                enc.acknowledge(index);
            }
        }
    }
}
#[test]
fn long_term_reference_recovers_lost_frames() {
    // Each loss after a mark was acknowledged.
    long_term_recovery(1, &[9, 17], 26);
}
#[test]
fn long_term_reference_recovers_lost_frames_with_two_layers() {
    // Frame 10 is T0 (recovered from the LTR marked at 8, restarting the
    // pattern at 11); frame 14 is T1, whose loss needs nothing.
    long_term_recovery(2, &[10, 14], 40);
}
#[test]
fn long_term_reference_recovers_lost_frames_with_three_layers() {
    // Frame 10 is T1 (a reference, recovered from the LTR marked at 8,
    // restarting the pattern at 11); frame 14 is T2, whose loss needs
    // nothing.
    long_term_recovery(3, &[10, 14], 40);
}

/// Packets and reconstructions of `frames` moving frames with `params`,
/// under rate control at `bitrate` (bits/s) if given.
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
    let mut encoder = Encoder::<crate::H264Backend>::new(settings).unwrap();
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

/// The recode contract (`CodecBackend::RECODE`): coding every frame twice
/// (`debug.recode_identity`) leaves the stream and the reconstruction
/// unchanged, across keys, a requested key, layers, the long-term
/// reference and rate control.
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
            let mut params = ParamOverrides::default();
            params.set("debug.recode_identity", identity);
            params.set("ltr.interval", ltr_interval);
            let mut settings = config(64, 48, 30, params);
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
            equal(a, b);
        }
    }
}

/// `rate.recode_overflow`: frames that would overflow a small buffer are
/// coded again, coarser, and the stream still decodes to the encoder's
/// reconstruction.
#[test]
fn overflowing_frames_are_recoded_and_decode_exactly() {
    let mut params = ParamOverrides::default();
    params.set("rate.recode_overflow", 1.);
    let (packets, recons, stats) = coded_sequence(config(64, 48, 30, params), Some(20_000), 16);
    assert!(stats.unwrap().recoded_frames > 0);
    let mut decoder = Decoder::new();
    for (packet, recon) in packets.iter().zip(&recons) {
        equal(&decoder.decode(packet).unwrap(), recon);
    }
}
#[test]
fn shown_pictures_equal_decoded_ones() {
    // 102x50 is cropped from the coded 112x64 (frame cropping).
    for (width, height) in [(32, 32), (102, 50)] {
        let cfg = config(width, height, 30, ParamOverrides::default());
        let mut enc = Encoder::<crate::H264Backend>::new(cfg).unwrap();
        let (mut owned, mut borrowed) = (Decoder::new(), Decoder::new());
        for index in 0..6 {
            let frame = enc.encode(&source(width, height, index));
            let decoded = owned.decode(&frame.data).unwrap();
            borrowed.decode_shown(&frame.data).unwrap();
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
fn disposable_overlay_changes_only_disposable_frames_and_decodes() {
    const FRAMES: usize = 8;
    let layered = |settings: &[(&str, f32)]| {
        let mut params = ParamOverrides::default();
        params.set("layers.count", 2.0);
        params.set("layers.adaptive", 0.0);
        for &(name, value) in settings {
            params.set(name, value);
        }
        let cfg = EncoderConfig {
            intra_period: 300,
            ..config(64, 48, 30, params)
        };
        let mut enc = Encoder::<crate::H264Backend>::new(cfg).unwrap();
        let mut dec = Decoder::new();
        (0..FRAMES)
            .map(|index| {
                let frame = enc.encode(&panning(64, 48, index));
                equal(&dec.decode(&frame.data).unwrap(), enc.reconstruction());
                frame
            })
            .collect::<Vec<_>>()
    };
    let plain = layered(&[]);
    let cheap = layered(&[
        ("decision.nonbase_disposable_only", 1.0),
        ("nonbase.h264.deblock_inter", 0.0),
        ("nonbase.h264.inter_zero_luma_scale", 3.0),
        ("nonbase.decision.subpel_half_only", 1.0),
    ]);
    // Only the disposable T1 frames run the cheap settings; T0 frames
    // reference only T0 frames, so they stay identical.
    for (a, b) in plain.iter().zip(&cheap) {
        assert_eq!(a.data == b.data, a.metadata.temporal_layer == 0);
    }
}
#[test]
fn stream_fixed_parameters_are_not_per_frame() {
    for name in ["nonbase.h264.chroma_qp_offset", "nonbase.h264.trellis_i4x4"] {
        let mut params = ParamOverrides::default();
        params.set(name, 1.0);
        assert!(Encoder::<crate::H264Backend>::new(config(32, 32, 30, params)).is_err());
    }
}
