//! End-to-end AV1 spatial SVC round-trip, OBU syntax and operating-point tests.

#![forbid(unsafe_code)]

use nc_av1::{Av1Backend, Decoder, filter_temporal_unit_layers, temporal_unit_layers};
use nc_common::{Picture, PlaneIndex};
use nc_encoder::{Codec, Encoder, EncoderConfig, FrameRate, ParamOverrides, StreamSignal};

/// Deterministic synthetic I420 frame with diagonal detail and frame-to-frame motion.
fn synthetic_frame(width: usize, height: usize, index: usize) -> Picture {
    let mut pic = Picture::new(width, height, 0).expect("valid dimensions");
    let shift = index * 3;
    {
        let mut luma = pic.plane_mut(PlaneIndex::Y).view_mut();
        for y in 0..height {
            let r = luma.row_mut(y);
            for (x, sample) in r[..width].iter_mut().enumerate() {
                let wave = ((x + shift) * 5 + (y + index) * 7) & 0x7f;
                let edge = if ((x + shift) / 24 + (y / 24)).is_multiple_of(2) {
                    64
                } else {
                    16
                };
                *sample = (24 + edge + wave) as u8;
            }
        }
    }
    for (plane, bias) in [(PlaneIndex::U, 96usize), (PlaneIndex::V, 144usize)] {
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let mut view = pic.plane_mut(plane).view_mut();
        for y in 0..ch {
            let r = view.row_mut(y);
            for (x, sample) in r[..cw].iter_mut().enumerate() {
                *sample = (bias + ((x * 3 + y * 5 + index * 2) & 0x3f)) as u8;
            }
        }
    }
    pic
}

fn assert_pictures_bit_exact(a: &Picture, b: &Picture, label: &str) {
    assert_eq!(
        (a.width(), a.height()),
        (b.width(), b.height()),
        "{label}: dimensions differ"
    );
    for plane in PlaneIndex::ALL {
        let ss = usize::from(plane != PlaneIndex::Y);
        let (w, h) = ((a.width() + ss) >> ss, (a.height() + ss) >> ss);
        let va = a.plane(plane).view();
        let vb = b.plane(plane).view();
        for y in 0..h {
            assert_eq!(
                &va.row(y)[..w],
                &vb.row(y)[..w],
                "{label}: plane {plane:?} row {y} mismatch"
            );
        }
    }
}

fn make_config(
    width: usize,
    height: usize,
    temporal_layers: u8,
    intra_period: u32,
    params: ParamOverrides,
) -> EncoderConfig {
    EncoderConfig {
        codec: Codec::Av1,
        width,
        height,
        frame_rate: FrameRate { num: 30, den: 1 },
        quantizer: 140,
        intra_period,
        block_log2: 4,
        key_qp_offset: None,
        temporal_layers,
        signal: StreamSignal::default(),
        params: params.coding_every_frame(),
    }
}

#[test]
fn l2t1_l2t2_l2t3_round_trip_bit_exact_across_all_operating_points() {
    let (width, height) = (160, 96);
    for temporal_layers in [1u8, 2, 3] {
        let mut params = ParamOverrides::default();
        params.set("spatial.layers", 2.0);
        let config = make_config(width, height, temporal_layers, 8, params);
        let mut encoder = Encoder::<Av1Backend>::new(config).expect("valid spatial config");

        let op_count = usize::from(temporal_layers) * 2;
        let mut decoders: Vec<Decoder> = (0..op_count)
            .map(|op| {
                let mut d = Decoder::new();
                d.set_operating_point(op as u8).expect("valid op");
                d
            })
            .collect();

        for frame_idx in 0..12 {
            let input = synthetic_frame(width, height, frame_idx);
            let encoded = encoder.encode(&input);
            let layers = temporal_unit_layers(&encoded.data).expect("valid OBU unit");
            assert_eq!(layers.len(), 2, "T={temporal_layers} frame {frame_idx}");
            assert_eq!(layers[0].1, 0);
            assert_eq!(layers[1].1, 1);
            assert_eq!(layers[0].0, layers[1].0);
            let tid = layers[0].0;

            let recon_s0 = encoder
                .layer_reconstruction(0)
                .expect("S0 reconstruction present")
                .clone();
            let recon_s1 = encoder
                .layer_reconstruction(1)
                .expect("S1 reconstruction present")
                .clone();

            for (op, decoder) in decoders.iter_mut().enumerate() {
                // Operating points are ordered highest quality first:
                // 0..temporal_layers include S0+S1 with max_tid = (T - 1 - op);
                // temporal_layers..2*temporal_layers include S0 only with max_tid = (2*T - 1 - op).
                let want_spatial = u8::from(op < usize::from(temporal_layers));
                let max_tid = if op < usize::from(temporal_layers) {
                    temporal_layers - 1 - op as u8
                } else {
                    2 * temporal_layers - 1 - op as u8
                };
                let result = decoder.decode_with_layer(&encoded.data);
                if tid <= max_tid {
                    let (pic, sid) = result
                        .unwrap_or_else(|e| {
                            panic!("T={temporal_layers} op={op} f={frame_idx}: {e}")
                        })
                        .unwrap_or_else(|| {
                            panic!(
                                "T={temporal_layers} op={op} f={frame_idx}: expected shown frame"
                            )
                        });
                    assert_eq!(sid, want_spatial);
                    assert_eq!(decoder.shown_spatial_id(), Some(want_spatial));
                    let expected = if want_spatial == 1 {
                        &recon_s1
                    } else {
                        &recon_s0
                    };
                    assert_pictures_bit_exact(
                        &pic,
                        expected,
                        &format!("L2T{temporal_layers} op={op} frame={frame_idx}"),
                    );
                    decoder.recycle(pic);
                } else {
                    let none = result.expect("dropped temporal layer should not error");
                    assert!(
                        none.is_none(),
                        "L2T{temporal_layers} op={op} frame={frame_idx} tid={tid} > {max_tid} should produce None"
                    );
                }
            }
        }
    }
}

#[test]
fn simulcast_in_stream_inter_layer_off_decodes_bit_exact_and_survives_s1_strip() {
    let (width, height) = (160, 96);
    let mut params = ParamOverrides::default();
    params.set("spatial.layers", 2.0);
    params.set("spatial.inter_layer", 0.0);
    let config = make_config(width, height, 2, 6, params);
    let mut encoder = Encoder::<Av1Backend>::new(config).expect("valid spatial config");

    let mut full = Decoder::new();
    let mut base_via_op = Decoder::new();
    base_via_op.set_operating_point(2).expect("valid op"); // S0 all temporal layers in L2T2
    let mut base_stripped = Decoder::new();
    base_stripped.set_operating_point(2).expect("valid op");

    for frame_idx in 0..10 {
        let input = synthetic_frame(width, height, frame_idx);
        let encoded = encoder.encode(&input);

        let full_pic = full.decode(&encoded.data).expect("full decode");
        assert_pictures_bit_exact(
            &full_pic,
            encoder.layer_reconstruction(1).unwrap(),
            &format!("inter_layer=0 full frame={frame_idx}"),
        );
        full.recycle(full_pic);

        let base_pic = base_via_op.decode(&encoded.data).expect("base op decode");
        let stripped =
            filter_temporal_unit_layers(&encoded.data, |_, sid| sid == 0).expect("strip S1");
        let stripped_pic = base_stripped
            .decode(&stripped)
            .expect("stripped S0 packet decode");
        assert_pictures_bit_exact(
            &base_pic,
            encoder.layer_reconstruction(0).unwrap(),
            &format!("inter_layer=0 base frame={frame_idx}"),
        );
        assert_pictures_bit_exact(
            &stripped_pic,
            &base_pic,
            &format!("inter_layer=0 stripped frame={frame_idx}"),
        );
        base_via_op.recycle(base_pic);
        base_stripped.recycle(stripped_pic);
    }
}

#[test]
fn sparse_enhancement_interval_1_5x_outputs_base_between_enhancement_tus() {
    // 192x108 enhancement (5 fps at interval=6), 128x72 base (30 fps): 1.5x ratio.
    let (width, height) = (192, 108);
    let mut params = ParamOverrides::default();
    params.set("spatial.layers", 2.0);
    params.set("spatial.base_scale", 2.0 / 3.0);
    params.set("spatial.enhancement_interval", 6.0);
    let config = make_config(width, height, 1, 12, params);
    let mut encoder = Encoder::<Av1Backend>::new(config).expect("valid sparse spatial config");

    let mut op0 = Decoder::new();
    let mut op1 = Decoder::new();
    op1.set_operating_point(1).expect("valid op");

    for frame_idx in 0..14 {
        let input = synthetic_frame(width, height, frame_idx);
        let encoded = encoder.encode(&input);
        let layers = temporal_unit_layers(&encoded.data).expect("valid OBU unit");
        let enh_due = frame_idx.is_multiple_of(6);
        assert_eq!(
            layers.len(),
            if enh_due { 2 } else { 1 },
            "frame {frame_idx}"
        );

        let (pic0, sid0) = op0
            .decode_with_layer(&encoded.data)
            .expect("op0 decode")
            .expect("shown frame on L2T1");
        let (pic1, sid1) = op1
            .decode_with_layer(&encoded.data)
            .expect("op1 decode")
            .expect("shown base frame on L2T1");

        assert_eq!(sid1, 0);
        assert_eq!((pic1.width(), pic1.height()), (128, 72));
        assert_pictures_bit_exact(
            &pic1,
            encoder.layer_reconstruction(0).unwrap(),
            &format!("sparse op1 frame={frame_idx}"),
        );

        if enh_due {
            assert_eq!(sid0, 1);
            assert_eq!((pic0.width(), pic0.height()), (192, 108));
            assert_pictures_bit_exact(
                &pic0,
                encoder.layer_reconstruction(1).unwrap(),
                &format!("sparse op0 S1 frame={frame_idx}"),
            );
        } else {
            assert_eq!(sid0, 0);
            assert_eq!((pic0.width(), pic0.height()), (128, 72));
            assert_pictures_bit_exact(
                &pic0,
                encoder.layer_reconstruction(0).unwrap(),
                &format!("sparse op0 S0-fallback frame={frame_idx}"),
            );
        }
        op0.recycle(pic0);
        op1.recycle(pic1);
    }
}

#[test]
fn decoder_operating_point_validation_and_dropped_tu_retention() {
    let (width, height) = (160, 96);
    let mut params = ParamOverrides::default();
    params.set("spatial.layers", 2.0);
    params.set("layers.adaptive", 0.0);
    let config = make_config(width, height, 2, 8, params);
    let mut encoder = Encoder::<Av1Backend>::new(config).expect("valid L2T2 config");

    // (c) Out-of-range operating points above MAX_OPERATING_POINT (31) are rejected immediately.
    let mut dec = Decoder::new();
    assert!(dec.set_operating_point(32).is_err());
    assert_eq!(dec.operating_point(), 0);

    // Encode TU 0 (key, T0: S0 + S1).
    let tu0 = encoder.encode(&synthetic_frame(width, height, 0));
    let s0_tu0 = encoder.layer_reconstruction(0).unwrap().clone();
    let s1_tu0 = encoder.layer_reconstruction(1).unwrap().clone();

    // Select op 3 (base layer T0 only in L2T2: 4 operating points 0..4).
    dec.set_operating_point(3).expect("op 3 valid before seq");
    let (pic0, sid0) = dec
        .decode_with_layer(&tu0.data)
        .expect("TU 0 decodes at op 3")
        .expect("TU 0 shown at op 3");
    assert_eq!(sid0, 0);
    assert_pictures_bit_exact(&pic0, &s0_tu0, "TU 0 at op 3");

    // (c) Once the sequence header is active (4 operating points), op 4 is rejected immediately.
    assert!(dec.set_operating_point(4).is_err());
    assert_eq!(dec.operating_point(), 3);

    // TU 1 is T1 (`layer.adaptive = 0` keeps L2T2 active from TU 1).
    let tu1 = encoder.encode(&synthetic_frame(width, height, 1));
    assert_eq!(tu1.metadata.temporal_layer, 1);

    // (a) & (b) `tu1` is T1, which op 3 drops in full; decode_optional and decode_with_layer
    // return Ok(None) (not Malformed) and retain `dec.shown()` from the previous T0 TU.
    assert!(
        dec.decode_optional(&tu1.data)
            .expect("T1 dropped cleanly")
            .is_none()
    );
    let retained = dec
        .shown()
        .expect("previous shown picture retained across dropped TU");
    assert_eq!((retained.width(), retained.height()), (80, 48));
    assert_eq!(dec.shown_spatial_id(), Some(0));

    // An S1-only TU at op 1 (base layer S0) also returns Ok(None) and retains `shown()`.
    let s1_only_tu1 =
        filter_temporal_unit_layers(&tu1.data, |_, sid| sid == 1).expect("keep S1 only");
    assert!(
        dec.decode_with_layer(&s1_only_tu1)
            .expect("S1-only TU at base op drops cleanly")
            .is_none()
    );
    assert!(dec.shown().is_some());

    // A malformed packet also retains `shown()`.
    assert!(dec.decode_with_layer(&[0xff, 0x00]).is_err());
    assert!(dec.shown().is_some());
    dec.recycle(pic0);

    // All-layers decode on TU 0 at op 0 returns both S0 and S1 in OBU order.
    let mut all_dec = Decoder::new();
    let all = all_dec
        .decode_all_layers(&tu0.data)
        .expect("all-layers decode on TU 0");
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].1, 0);
    assert_eq!(all[1].1, 1);
    assert_pictures_bit_exact(&all[0].0, &s0_tu0, "all-layers S0");
    assert_pictures_bit_exact(&all[1].0, &s1_tu0, "all-layers S1");
}

#[test]
fn compound_and_single_ref_spatial_svc_differ_and_roundtrip_bit_exact() {
    let (width, height) = (160, 96);
    let run = |compound: f32| -> Vec<Vec<u8>> {
        let mut params = ParamOverrides::default();
        params.set("spatial.layers", 2.0);
        params.set("av1.compound", compound);
        params.set("refs.second_compound_margin", 2.0);
        let config = make_config(width, height, 1, 8, params);
        let mut encoder = Encoder::<Av1Backend>::new(config).expect("valid spatial config");
        let mut dec0 = Decoder::new();
        let mut dec1 = Decoder::new();
        dec1.set_operating_point(1).expect("op 1");
        let mut packets = Vec::new();
        for idx in 0..6 {
            let input = synthetic_frame(width, height, idx);
            let encoded = encoder.encode(&input);
            let (pic0, sid0) = dec0
                .decode_with_layer(&encoded.data)
                .expect("dec0")
                .expect("shown0");
            let (pic1, sid1) = dec1
                .decode_with_layer(&encoded.data)
                .expect("dec1")
                .expect("shown1");
            assert_eq!(sid0, 1);
            assert_eq!(sid1, 0);
            assert_pictures_bit_exact(
                &pic0,
                encoder.layer_reconstruction(1).unwrap(),
                "compound op0",
            );
            assert_pictures_bit_exact(
                &pic1,
                encoder.layer_reconstruction(0).unwrap(),
                "compound op1",
            );
            dec0.recycle(pic0);
            dec1.recycle(pic1);
            packets.push(encoded.data);
        }
        packets
    };
    let with_compound = run(1.0);
    let without_compound = run(0.0);
    // Key frame TU 0 has no secondary reference, so it is identical; inter TUs
    // exercise compound prediction on S1 and produce distinct valid bitstreams.
    assert_eq!(with_compound[0], without_compound[0]);
    assert_ne!(with_compound[1..], without_compound[1..]);
}

#[test]
fn l3t1_l3t2_l3t3_round_trip_bit_exact_across_all_operating_points() {
    let (width, height) = (160, 96);
    for temporal_layers in [1u8, 2, 3] {
        let mut params = ParamOverrides::default();
        params.set("spatial.layers", 3.0);
        params.set("layers.adaptive", 0.0);
        let config = make_config(width, height, temporal_layers, 8, params);
        let mut encoder = Encoder::<Av1Backend>::new(config).expect("valid 3-layer config");

        let t = usize::from(temporal_layers);
        let op_count = t * 3;
        let mut decoders: Vec<Decoder> = (0..op_count)
            .map(|op| {
                let mut d = Decoder::new();
                d.set_operating_point(op as u8).expect("valid op");
                d
            })
            .collect();
        let mut s1_stripped_dec = Decoder::new();
        s1_stripped_dec
            .set_operating_point(t as u8)
            .expect("S1 max-fps op");
        let mut s0_stripped_dec = Decoder::new();
        s0_stripped_dec
            .set_operating_point((2 * t) as u8)
            .expect("S0 max-fps op");
        let mut all_layers_dec = Decoder::new();

        for frame_idx in 0..12 {
            let input = synthetic_frame(width, height, frame_idx);
            let encoded = encoder.encode(&input);
            let layers = temporal_unit_layers(&encoded.data).expect("valid OBU unit");
            assert_eq!(layers.len(), 3, "L3T{temporal_layers} frame {frame_idx}");
            assert_eq!(layers[0].1, 0);
            assert_eq!(layers[1].1, 1);
            assert_eq!(layers[2].1, 2);
            assert_eq!(layers[0].0, layers[1].0);
            assert_eq!(layers[1].0, layers[2].0);
            let tid = layers[0].0;

            let recons = [
                encoder.layer_reconstruction(0).expect("S0 recon").clone(),
                encoder.layer_reconstruction(1).expect("S1 recon").clone(),
                encoder.layer_reconstruction(2).expect("S2 recon").clone(),
            ];
            assert_eq!((recons[0].width(), recons[0].height()), (40, 24));
            assert_eq!((recons[1].width(), recons[1].height()), (80, 48));
            assert_eq!((recons[2].width(), recons[2].height()), (160, 96));

            let all = all_layers_dec
                .decode_all_layers(&encoded.data)
                .expect("all-layers 3-layer decode");
            assert_eq!(all.len(), 3);
            for (s, (pic, sid)) in all.into_iter().enumerate() {
                assert_eq!(usize::from(sid), s);
                assert_pictures_bit_exact(
                    &pic,
                    &recons[s],
                    &format!("L3T{temporal_layers} all-layers S{s} f={frame_idx}"),
                );
                all_layers_dec.recycle(pic);
            }

            // Physically stripping S2 leaves S0+S1 bit-exact at op `t`; stripping S1+S2
            // leaves S0 bit-exact at op `2*t`.
            let keep_s0_s1 =
                filter_temporal_unit_layers(&encoded.data, |_, sid| sid <= 1).expect("strip S2");
            let pic_s1 = s1_stripped_dec
                .decode(&keep_s0_s1)
                .expect("S0+S1 stripped decode");
            assert_pictures_bit_exact(&pic_s1, &recons[1], "stripped S0+S1");
            s1_stripped_dec.recycle(pic_s1);

            let keep_s0 =
                filter_temporal_unit_layers(&encoded.data, |_, sid| sid == 0).expect("strip S1+S2");
            let pic_s0 = s0_stripped_dec
                .decode(&keep_s0)
                .expect("S0 stripped decode");
            assert_pictures_bit_exact(&pic_s0, &recons[0], "stripped S0");
            s0_stripped_dec.recycle(pic_s0);

            for (op, decoder) in decoders.iter_mut().enumerate() {
                let want_spatial = (2 - op / t) as u8;
                let max_tid = (t - 1 - (op % t)) as u8;
                let result = decoder.decode_with_layer(&encoded.data);
                if tid <= max_tid {
                    let (pic, sid) = result
                        .unwrap_or_else(|e| {
                            panic!("L3T{temporal_layers} op={op} f={frame_idx}: {e}")
                        })
                        .expect("shown frame");
                    assert_eq!(sid, want_spatial);
                    assert_pictures_bit_exact(
                        &pic,
                        &recons[usize::from(want_spatial)],
                        &format!("L3T{temporal_layers} op={op} frame={frame_idx}"),
                    );
                    decoder.recycle(pic);
                } else {
                    assert!(
                        result.expect("dropped temporal layer clean").is_none(),
                        "L3T{temporal_layers} op={op} f={frame_idx} tid={tid} > {max_tid}"
                    );
                }
            }
        }
    }
}

#[test]
fn custom_non_dyadic_three_layer_scales_and_loss_repair_round_trip() {
    // Custom (1/3, 2/3) scales on 192x108 -> S0=64x36, S1=128x72, S2=192x108.
    let (width, height) = (192, 108);
    let mut params = ParamOverrides::default();
    params.set("spatial.layers", 3.0);
    params.set("spatial.s0_scale", 1.0 / 3.0);
    params.set("spatial.s1_scale", 2.0 / 3.0);
    params.set("layers.adaptive", 0.0);
    let config = make_config(width, height, 2, 30, params);
    let mut encoder = Encoder::<Av1Backend>::new(config).expect("valid custom 3-layer config");

    let mut dec_s2 = Decoder::new();
    let mut dec_s1 = Decoder::new();
    dec_s1.set_operating_point(2).expect("S1 op");
    let mut dec_s0 = Decoder::new();
    dec_s0.set_operating_point(4).expect("S0 op");

    for frame_idx in 0..8 {
        let input = synthetic_frame(width, height, frame_idx);
        let encoded = encoder.encode(&input);
        let s0_recon = encoder.layer_reconstruction(0).unwrap().clone();
        let s1_recon = encoder.layer_reconstruction(1).unwrap().clone();
        let s2_recon = encoder.layer_reconstruction(2).unwrap().clone();
        assert_eq!((s0_recon.width(), s0_recon.height()), (64, 36));
        assert_eq!((s1_recon.width(), s1_recon.height()), (128, 72));
        assert_eq!((s2_recon.width(), s2_recon.height()), (192, 108));

        // Simulate S2 loss on frame 2 (T0 reference TU): S0 and S1 keep decoding,
        // and once we report S2 loss, the next TU repairs S2 from S1 without a keyframe!
        if frame_idx == 2 {
            let without_s2 = filter_temporal_unit_layers(&encoded.data, |_, sid| sid <= 1).unwrap();
            let p1 = dec_s1
                .decode(&without_s2)
                .expect("S1 unaffected by S2 loss");
            let p0 = dec_s0
                .decode(&without_s2)
                .expect("S0 unaffected by S2 loss");
            let p2_fallback = dec_s2.decode(&without_s2).expect("op0 falls back to S1");
            assert_pictures_bit_exact(&p1, &s1_recon, "S1 on S2 loss");
            assert_pictures_bit_exact(&p0, &s0_recon, "S0 on S2 loss");
            assert_pictures_bit_exact(&p2_fallback, &s1_recon, "op0 fallback to S1");
            dec_s1.recycle(p1);
            dec_s0.recycle(p0);
            dec_s2.recycle(p2_fallback);
            encoder.report_layer_loss(2, 2);
            continue;
        }
        if frame_idx == 3 {
            assert_ne!(
                encoded.metadata.frame_type,
                nc_encoder::FrameType::Key,
                "S2 repair should refresh S2 from S1 without forcing an S0 keyframe"
            );
        }
        let p2 = dec_s2.decode(&encoded.data).expect("S2 decode");
        let p1 = dec_s1.decode(&encoded.data).expect("S1 decode");
        let p0 = dec_s0.decode(&encoded.data).expect("S0 decode");
        assert_pictures_bit_exact(&p2, &s2_recon, &format!("S2 f={frame_idx}"));
        assert_pictures_bit_exact(&p1, &s1_recon, &format!("S1 f={frame_idx}"));
        assert_pictures_bit_exact(&p0, &s0_recon, &format!("S0 f={frame_idx}"));
        dec_s2.recycle(p2);
        dec_s1.recycle(p1);
        dec_s0.recycle(p0);
    }
}
