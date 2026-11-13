//! Independent decoder checks of the VVC subset. No YUV files.
use nc_common::{Picture, PlaneIndex, yuv};
use nc_encoder::{Codec, Encoder, EncoderConfig, FrameRate, ParamOverrides, StreamSignal};
use nc_vvc::VvcBackend;
use std::process::Command;

/// `FFmpeg` releases before this mis-decode VVC ALF streams: they read
/// uninitialised `CC-ALF` indexes whenever ALF is on (`FFmpeg` ticket 11325).
const MIN_FFMPEG_VERSION: (u32, u32) = (8, 0);

/// The `FFmpeg` to check against: `NC_FFMPEG`, the pinned decode-only build
/// (`anchors/<arch>/bin/ffmpeg-decode`, `build_anchors.sh ffmpeg`), never one
/// on `PATH`. Panics unless it reports a release >= [`MIN_FFMPEG_VERSION`].
fn pinned_ffmpeg() -> String {
    let ffmpeg = std::env::var("NC_FFMPEG").expect("NC_FFMPEG must point to FFmpeg >= 8.0");
    let banner = Command::new(&ffmpeg)
        .arg("-version")
        .output()
        .unwrap()
        .stdout;
    let banner = String::from_utf8_lossy(&banner);
    let version = banner
        .strip_prefix("ffmpeg version ")
        .map(|v| v.trim_start_matches('n'))
        .and_then(|v| {
            let mut parts = v.split(|c: char| !c.is_ascii_digit());
            Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
        });
    assert!(
        version.is_some_and(|v: (u32, u32)| v >= MIN_FFMPEG_VERSION),
        "{ffmpeg} is not FFmpeg >= 8.0 (older ones mis-decode VVC ALF, ticket 11325): {}",
        banner.lines().next().unwrap_or("")
    );
    ffmpeg
}

/// The CTU sizes we code (`vvc.ctu_size`).
const CTU_SIZES: [usize; 2] = [32, 64];

fn config(width: usize, height: usize, qp: u32) -> EncoderConfig {
    EncoderConfig {
        codec: Codec::Vvc,
        width,
        height,
        quantizer: qp,
        frame_rate: FrameRate { num: 30, den: 1 },
        intra_period: 30,
        temporal_layers: 1,
        block_log2: 5,
        key_qp_offset: None,
        signal: StreamSignal::default(),
        params: ParamOverrides::default(),
    }
}

/// [`config`] with `ctu`-sample CTUs, partitions down from the CTU.
fn config_ctu(width: usize, height: usize, qp: u32, ctu: usize) -> EncoderConfig {
    let mut c = config(width, height, qp);
    c.params.set("vvc.ctu_size", ctu as f32);
    c.block_log2 = ctu.ilog2() as u8;
    c
}

#[test]
#[ignore = "requires NC_FFMPEG (FFmpeg >= 8.0) and NC_VVDEC pointing to vvdecapp"]
fn standard_decoders_match_reconstruction() {
    let vvdec = std::env::var("NC_VVDEC").expect("NC_VVDEC must point to vvdecapp");
    let sizes = [(2, 2), (32, 32), (64, 64), (66, 34), (128, 96), (136, 72)];
    for ((width, height), ctu) in sizes
        .into_iter()
        .flat_map(|size| CTU_SIZES.map(|ctu| (size, ctu)))
    {
        for qp in [0, 1, 2, 3, 4, 5, 22, 37, 51, 63] {
            let mut encoder =
                Encoder::<VvcBackend>::new(config_ctu(width, height, qp, ctu)).unwrap();
            let mut decoder = nc_vvc::Decoder::new();
            let mut stream = Vec::new();
            let mut expected = Vec::new();
            for frame in 0..5 {
                let mut source = Picture::new(width, height, 0).unwrap();
                for plane in PlaneIndex::ALL {
                    let mut view = source.plane_mut(plane).view_mut();
                    for y in 0..view.height() {
                        for (x, p) in view.row_mut(y).iter_mut().enumerate() {
                            *p = if frame == 0 {
                                128
                            } else if frame == 1 {
                                0
                            } else if frame == 2 {
                                255
                            } else {
                                ((x * 3 + y * 5 + frame * 43 + plane as usize * 61) % 256) as u8
                            };
                        }
                    }
                }
                let packet = encoder.encode(&source).data;
                let own = decoder.decode(&packet).unwrap();
                let mut own_bytes = Vec::new();
                let mut recon_bytes = Vec::new();
                yuv::write_frame(&mut own_bytes, &own).unwrap();
                yuv::write_frame(&mut recon_bytes, encoder.reconstruction()).unwrap();
                assert!(
                    own_bytes == recon_bytes,
                    "own mismatch {width}x{height} ctu{ctu} qp{qp} frame{frame}: {:?}",
                    own_bytes.iter().zip(&recon_bytes).position(|(a, b)| a != b)
                );
                stream.extend(packet);
                yuv::write_frame(&mut expected, encoder.reconstruction()).unwrap();
            }
            check_external(&vvdec, &stream, &expected, width, height, qp);
        }
    }
}

fn check_external(
    vvdec: &str,
    stream: &[u8],
    expected: &[u8],
    width: usize,
    height: usize,
    qp: u32,
) {
    static NEXT_STREAM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = NEXT_STREAM.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "nc-vvc-{}-{unique}-{width}-{height}-{qp}.vvc",
        std::process::id()
    ));
    std::fs::write(&path, stream).unwrap();
    let ffmpeg = Command::new(pinned_ffmpeg())
        .args(["-v", "error", "-threads", "1", "-f", "vvc", "-i"])
        .arg(&path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .output()
        .unwrap();
    let vvc = Command::new(vvdec)
        .args(["-v", "0", "-t", "1", "-b"])
        .arg(&path)
        .args(["-o", "/dev/stdout"])
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    for (name, result) in [("vvdec", vvc), ("ffmpeg", ffmpeg)] {
        assert!(
            result.status.success(),
            "{name} {width}x{height} qp{qp}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            result.stdout.len(),
            expected.len(),
            "{name} length {width}x{height} qp{qp}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        if let Some(i) = result.stdout.iter().zip(expected).position(|(a, b)| a != b) {
            panic!(
                "{name} {width}x{height} qp{qp}: sample {i} got {} expected {}",
                result.stdout[i], expected[i]
            );
        }
    }
}

#[test]
#[ignore = "requires NC_FFMPEG (FFmpeg >= 8.0) and NC_VVDEC pointing to vvdecapp"]
fn explicit_motion_and_mixed_intra_match_standard_decoders() {
    use nc_encoder::{
        Block, BlockDecision, CodecBackend, FramePlan, FrameReferences, FrameType, Mv,
        PredictionMode,
    };
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    for qp in [0, 22, 37, 63] {
        // Drives 32x32 CTUs directly.
        let mut backend = VvcBackend::new(&config_ctu(128, 96, qp, 32)).unwrap();
        let mut reference = Picture::new(128, 96, 0).unwrap();
        let mut decoder = nc_vvc::Decoder::new();
        let mut stream = Vec::new();
        let mut expected = Vec::new();
        for frame in 0..12 {
            let mut source = Picture::new(128, 96, 0).unwrap();
            for plane in PlaneIndex::ALL {
                let mut v = source.plane_mut(plane).view_mut();
                for y in 0..v.height() {
                    for (x, p) in v.row_mut(y).iter_mut().enumerate() {
                        *p = ((x / 16 * 49 + y / 16 * 71 + frame * 13 + plane as usize * 31) % 256)
                            as u8;
                    }
                }
            }
            backend.begin_frame(&FramePlan {
                index: frame as u64,
                frame_type: if frame == 0 {
                    FrameType::Key
                } else {
                    FrameType::Inter
                },
                quantizer: qp,
                references: FrameReferences::FLAT,
                lambda_scale: 1.,
                block_qp: false,
            });
            let mut recon = Picture::new(128, 96, 0).unwrap();
            for y in (0..96).step_by(32) {
                for x in (0..128).step_by(32) {
                    backend.begin_superblock(x, y);
                    let block = Block::square(x, y, 5);
                    let i = x / 32 + y / 32 * 4 + frame;
                    let mode = if frame == 0 || i % 7 == 0 {
                        PredictionMode::IntraDc
                    } else if i % 3 == 0 {
                        PredictionMode::Inter(backend.skip_vector(&block).unwrap())
                    } else {
                        PredictionMode::Inter(Mv {
                            x: ((i * 17) % 67) as i16 - 33,
                            y: ((i * 23) % 51) as i16 - 25,
                        })
                    };
                    backend.write_partition(&block, false);
                    backend.encode_block(
                        &block,
                        &BlockDecision::new(mode),
                        &source,
                        Some(&reference),
                        &mut recon,
                    );
                }
            }
            backend.filter_frame(&source, &mut recon);
            let packet = backend.end_frame();
            backend.commit_frame();
            let own = decoder.decode(&packet).unwrap();
            let mut own_bytes = Vec::new();
            let mut recon_bytes = Vec::new();
            yuv::write_frame(&mut own_bytes, &own).unwrap();
            yuv::write_frame(&mut recon_bytes, &recon).unwrap();
            assert_eq!(own_bytes, recon_bytes);
            expected.extend(recon_bytes);
            stream.extend(packet);
            recon.extend_borders();
            reference = recon;
        }
        check_external(&vvdec, &stream, &expected, 128, 96, qp);
    }
}

#[test]
#[ignore = "requires NC_FFMPEG (FFmpeg >= 8.0) and NC_VVDEC pointing to vvdecapp"]
fn skip_chain_poc_wrap_and_key_recovery_match_standard_decoders() {
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    let mut c = config(64, 64, 22);
    c.intra_period = 1000;
    let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
    let source = Picture::new(64, 64, 0).unwrap();
    let mut stream = Vec::new();
    let mut expected = Vec::new();
    for frame in 0..262 {
        if frame == 260 {
            encoder.request_keyframe();
        }
        stream.extend(encoder.encode(&source).data);
        yuv::write_frame(&mut expected, encoder.reconstruction()).unwrap();
    }
    check_external(&vvdec, &stream, &expected, 64, 64, 22);
}

#[test]
#[ignore = "requires NC_FFMPEG (FFmpeg >= 8.0) and NC_VVDEC pointing to vvdecapp"]
fn small_partitions_and_texture_match_standard_decoders() {
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    let cases = [(3, 32), (4, 32), (5, 32), (4, 64), (5, 64), (6, 64)];
    // 200x136: several 64x64 CTUs, partial ones on the right and bottom.
    for ((width, height), (log2, ctu)) in [(66, 34), (200, 136)]
        .into_iter()
        .flat_map(|size| cases.map(|case| (size, case)))
    {
        for qp in [0, 22, 37, 63] {
            let mut c = config_ctu(width, height, qp, ctu);
            c.block_log2 = log2;
            c.intra_period = 3;
            let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
            let mut decoder = nc_vvc::Decoder::new();
            let mut stream = Vec::new();
            let mut expected = Vec::new();
            for frame in 0..7 {
                let mut source = Picture::new(width, height, 0).unwrap();
                for plane in PlaneIndex::ALL {
                    let mut v = source.plane_mut(plane).view_mut();
                    for y in 0..v.height() {
                        for (x, p) in v.row_mut(y).iter_mut().enumerate() {
                            *p = match frame {
                                0 => ((x * 7 + plane as usize * 47) % 256) as u8,
                                1 => ((y * 11 + plane as usize * 47) % 256) as u8,
                                2 => ((x + y + plane as usize * 47) % 256) as u8,
                                _ => {
                                    ((x * 23 + y * 17 + frame * 13 + plane as usize * 47) ^ (x * y))
                                        as u8
                                }
                            };
                        }
                    }
                }
                let packet = encoder.encode(&source).data;
                let own = decoder.decode(&packet).unwrap();
                let mut a = Vec::new();
                let mut b = Vec::new();
                yuv::write_frame(&mut a, &own).unwrap();
                yuv::write_frame(&mut b, encoder.reconstruction()).unwrap();
                assert!(
                    a == b,
                    "{width}x{height} log2{log2} ctu{ctu} qp{qp} frame{frame}"
                );
                stream.extend(packet);
                expected.extend(b);
            }
            check_external(&vvdec, &stream, &expected, width, height, qp);
        }
    }
}

#[test]
#[ignore = "requires NC_FFMPEG (FFmpeg >= 8.0) and NC_VVDEC pointing to vvdecapp"]
fn colour_and_key_offset_match_standard_decoders() {
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    for description in [
        None,
        Some(nc_common::ColourDescription::BT709),
        Some(nc_common::ColourDescription::BT601_625),
    ] {
        for full_range in [false, true] {
            let mut cfg = config(66, 34, 37);
            cfg.signal.colour = nc_common::ColourInfo {
                description,
                full_range,
            };
            cfg.key_qp_offset = Some(9);
            let mut encoder = Encoder::<VvcBackend>::new(cfg).unwrap();
            let mut decoder = nc_vvc::Decoder::new();
            let mut stream = Vec::new();
            let mut expected = Vec::new();
            for frame in 0..3 {
                let mut source = Picture::new(66, 34, 0).unwrap();
                let mut view = source.plane_mut(PlaneIndex::Y).view_mut();
                for y in 0..view.height() {
                    view.row_mut(y).fill(40 + frame * 30);
                }
                let packet = encoder.encode(&source);
                let own = decoder.decode(&packet.data).unwrap();
                assert_eq!(decoder.colour().full_range, full_range);
                if description.is_some() {
                    assert_eq!(decoder.colour().description, description);
                }
                let mut own_bytes = Vec::new();
                yuv::write_frame(&mut own_bytes, &own).unwrap();
                let mut recon = Vec::new();
                yuv::write_frame(&mut recon, encoder.reconstruction()).unwrap();
                assert_eq!(own_bytes, recon);
                expected.extend(recon);
                stream.extend(packet.data);
            }
            check_external(&vvdec, &stream, &expected, 66, 34, 37);
        }
    }
}

#[test]
#[ignore = "requires NC_FFMPEG (FFmpeg >= 8.0) and NC_VVDEC pointing to vvdecapp"]
fn multi_type_tree_matches_standard_decoders() {
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    // Binary, then binary and ternary leaves; planned (seed 0) and random
    // shapes, modes and vectors (`debug.random_decisions`).
    for ((mtt, ctu), seed) in [1, 2]
        .into_iter()
        .flat_map(|mtt| CTU_SIZES.map(|ctu| (mtt, ctu)))
        .flat_map(|case| [0, 7, 11].map(|seed| (case, seed)))
    {
        for (width, height) in [(66, 34), (200, 136)] {
            for qp in [22, 37] {
                let mut c = config_ctu(width, height, qp, ctu);
                c.params.set("vvc.mtt", mtt as f32);
                if seed > 0 {
                    c.params.set("debug.random_decisions", seed as f32);
                }
                let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
                let mut decoder = nc_vvc::Decoder::new();
                let mut stream = Vec::new();
                let mut expected = Vec::new();
                for frame in 0..5 {
                    let mut source = Picture::new(width, height, 0).unwrap();
                    for plane in PlaneIndex::ALL {
                        let mut v = source.plane_mut(plane).view_mut();
                        for y in 0..v.height() {
                            // Top and bottom halves of every 32 rows move
                            // apart, so binary splits pay.
                            let shift = if y % 32 < 16 { frame * 3 } else { 0 };
                            for (x, p) in v.row_mut(y).iter_mut().enumerate() {
                                let u = x + shift;
                                *p = ((u * 23 + y * 17 + plane as usize * 47) ^ (u * y / 7)) as u8;
                            }
                        }
                    }
                    let packet = encoder.encode(&source).data;
                    let own = decoder.decode(&packet).unwrap();
                    let mut a = Vec::new();
                    let mut b = Vec::new();
                    yuv::write_frame(&mut a, &own).unwrap();
                    yuv::write_frame(&mut b, encoder.reconstruction()).unwrap();
                    assert!(
                        a == b,
                        "{width}x{height} mtt{mtt} ctu{ctu} seed{seed} qp{qp} frame{frame}"
                    );
                    stream.extend(packet);
                    expected.extend(b);
                }
                check_external(&vvdec, &stream, &expected, width, height, qp);
            }
        }
    }
}

#[test]
#[ignore = "requires NC_FFMPEG (FFmpeg >= 8.0) and NC_VVDEC pointing to vvdecapp"]
fn intra_sub_partitions_match_standard_decoders() {
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    // Sharp oblique edges, which sub-partitions predict well, beside
    // picture and CTU edges (the references of later sub-partitions).
    for ctu in CTU_SIZES {
        for (width, height) in [(66, 34), (200, 136)] {
            for qp in [22, 37] {
                let mut c = config_ctu(width, height, qp, ctu);
                c.params.set("vvc.isp", 1.);
                // P-picture intra CUs too (8x8 to 32x32 sub-partitions).
                c.params.set("vvc.isp_inter", 1.);
                let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
                let mut decoder = nc_vvc::Decoder::new();
                let mut stream = Vec::new();
                let mut expected = Vec::new();
                for frame in 0..3 {
                    let mut source = Picture::new(width, height, 0).unwrap();
                    for plane in PlaneIndex::ALL {
                        let mut v = source.plane_mut(plane).view_mut();
                        for y in 0..v.height() {
                            for (x, p) in v.row_mut(y).iter_mut().enumerate() {
                                let u = x + frame * 2;
                                let band = (u * 3 + y * 7 + plane as usize * 5) / 11 % 3;
                                *p = (40 + band * 80 + (u * y) % 9) as u8;
                            }
                        }
                    }
                    let packet = encoder.encode(&source).data;
                    let own = decoder.decode(&packet).unwrap();
                    let mut a = Vec::new();
                    let mut b = Vec::new();
                    yuv::write_frame(&mut a, &own).unwrap();
                    yuv::write_frame(&mut b, encoder.reconstruction()).unwrap();
                    assert!(a == b, "{width}x{height} ctu{ctu} qp{qp} frame{frame}");
                    stream.extend(packet);
                    expected.extend(b);
                }
                check_external(&vvdec, &stream, &expected, width, height, qp);
            }
        }
    }
}

#[test]
#[ignore = "requires NC_FFMPEG (FFmpeg >= 8.0) and NC_VVDEC pointing to vvdecapp"]
fn dependent_quantization_matches_standard_decoders() {
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    // Every state and both quantisers: dense texture at fine and coarse
    // QPs, moving so P pictures code inter residuals too.
    for ctu in CTU_SIZES {
        for (width, height) in [(66, 34), (200, 136)] {
            for qp in [12, 27, 42] {
                let mut c = config_ctu(width, height, qp, ctu);
                c.params.set("vvc.dep_quant", 1.);
                let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
                let mut decoder = nc_vvc::Decoder::new();
                let mut stream = Vec::new();
                let mut expected = Vec::new();
                for frame in 0..3 {
                    let mut source = Picture::new(width, height, 0).unwrap();
                    for plane in PlaneIndex::ALL {
                        let mut v = source.plane_mut(plane).view_mut();
                        for y in 0..v.height() {
                            for (x, p) in v.row_mut(y).iter_mut().enumerate() {
                                let u = x + frame * 3;
                                *p = ((u * 29 + y * 13 + plane as usize * 31) ^ (u * y / 5)) as u8;
                            }
                        }
                    }
                    let packet = encoder.encode(&source).data;
                    let own = decoder.decode(&packet).unwrap();
                    let mut a = Vec::new();
                    let mut b = Vec::new();
                    yuv::write_frame(&mut a, &own).unwrap();
                    yuv::write_frame(&mut b, encoder.reconstruction()).unwrap();
                    assert!(a == b, "{width}x{height} ctu{ctu} qp{qp} frame{frame}");
                    stream.extend(packet);
                    expected.extend(b);
                }
                check_external(&vvdec, &stream, &expected, width, height, qp);
            }
        }
    }
}

#[test]
#[ignore = "requires NC_FFMPEG (FFmpeg >= 8.0) and NC_VVDEC pointing to vvdecapp"]
fn affine_motion_matches_standard_decoders() {
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    // A smooth texture turning and zooming about the centre: affine merge
    // and AMVP CUs, translational CUs merging sixteenth-sample vectors from
    // them, sub-block deblocking, inheritance across CTU rows, and models
    // reaching past the picture edges.
    for ctu in CTU_SIZES {
        for (width, height) in [(136, 72), (200, 136)] {
            for (affine, qp) in [(1., 22), (2., 37), (1., 45)] {
                let mut c = config_ctu(width, height, qp, ctu);
                c.params.set("vvc.affine", affine);
                let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
                let mut decoder = nc_vvc::Decoder::new();
                let mut stream = Vec::new();
                let mut expected = Vec::new();
                for frame in 0..4 {
                    let source = turning_texture(width, height, frame);
                    let packet = encoder.encode(&source).data;
                    let own = decoder.decode(&packet).unwrap();
                    let mut a = Vec::new();
                    let mut b = Vec::new();
                    yuv::write_frame(&mut a, &own).unwrap();
                    yuv::write_frame(&mut b, encoder.reconstruction()).unwrap();
                    assert!(
                        a == b,
                        "{width}x{height} ctu{ctu} affine{affine} qp{qp} frame{frame}"
                    );
                    stream.extend(packet);
                    expected.extend(b);
                }
                check_external(&vvdec, &stream, &expected, width, height, qp);
            }
        }
    }
}

/// Frame `frame` of a smooth texture rotating by 0.02 rad and zooming by
/// 1.5% per frame about the picture centre.
fn turning_texture(width: usize, height: usize, frame: usize) -> Picture {
    let mut source = Picture::new(width, height, 0).unwrap();
    let (angle, zoom) = (0.02 * frame as f64, 1. + 0.015 * frame as f64);
    for plane in PlaneIndex::ALL {
        let scale = if plane == PlaneIndex::Y { 1. } else { 2. };
        let (cx, cy) = (width as f64 / 2. / scale, height as f64 / 2. / scale);
        let mut v = source.plane_mut(plane).view_mut();
        for y in 0..v.height() {
            for (x, p) in v.row_mut(y).iter_mut().enumerate() {
                let (dx, dy) = ((x as f64 - cx) * scale, (y as f64 - cy) * scale);
                let u = (dx * angle.cos() - dy * angle.sin()) / zoom;
                let w = (dx * angle.sin() + dy * angle.cos()) / zoom;
                let texture = (u * 0.21).sin() * (w * 0.17).cos() + 0.5 * ((u + w) * 0.07).sin();
                *p = (128. + 70. * texture + 9. * plane as usize as f64).clamp(0., 255.) as u8;
            }
        }
    }
    source
}

#[test]
#[ignore = "requires NC_FFMPEG (FFmpeg >= 8.0) and NC_VVDEC pointing to vvdecapp"]
fn adaptive_loop_filter_matches_standard_decoders() {
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    // Signalled (APS) filters with and without the fixed sets, and the
    // fixed sets alone; blurred moving texture so that filtering pays,
    // both CTU sizes (virtual boundaries), a picture edge inside a CTU.
    let mut signalled_pictures = 0;
    for ctu in CTU_SIZES {
        for (aps, fixed, exact) in [(1., 1., 1.), (1., 0., 1.), (1., 1., 0.), (0., 1., 1.)] {
            for qp in [27, 37, 45] {
                let (width, height) = (200, 136);
                let mut c = config_ctu(width, height, qp, ctu);
                c.params.set("vvc.alf", 1.);
                c.params.set("vvc.alf_aps", aps);
                c.params.set("vvc.alf_fixed_sets", fixed);
                c.params.set("vvc.alf_exact_decision", exact);
                // Every other P picture reuses the previous bank, only in
                // the CTBs that picture filtered.
                c.params.set("vvc.alf_bank_interval", 2.);
                c.params.set("vvc.alf_reuse_on_only", 1.);
                let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
                let mut decoder = nc_vvc::Decoder::new();
                let mut stream = Vec::new();
                let mut expected = Vec::new();
                for frame in 0..4 {
                    let mut source = Picture::new(width, height, 0).unwrap();
                    for plane in PlaneIndex::ALL {
                        let mut v = source.plane_mut(plane).view_mut();
                        for y in 0..v.height() {
                            for (x, p) in v.row_mut(y).iter_mut().enumerate() {
                                let u = (x + frame * 2) as f32;
                                let wave =
                                    (u * 0.7).sin() * 40. + (y as f32 * 0.45 + u * 0.2).cos() * 30.;
                                *p = (128. + wave + ((x * 7 + y * 3) % 11) as f32) as u8;
                            }
                        }
                    }
                    let packet = encoder.encode(&source).data;
                    // Prefix APS NAL units (type 17) in the Annex B packet.
                    signalled_pictures += usize::from(
                        packet
                            .windows(5)
                            .any(|w| w[..4] == [0, 0, 1, 0] && w[4] >> 3 == 17),
                    );
                    let own = decoder.decode(&packet).unwrap();
                    let mut a = Vec::new();
                    let mut b = Vec::new();
                    yuv::write_frame(&mut a, &own).unwrap();
                    yuv::write_frame(&mut b, encoder.reconstruction()).unwrap();
                    assert!(
                        a == b,
                        "ctu{ctu} aps{aps} fixed{fixed} exact{exact} qp{qp} frame{frame}"
                    );
                    stream.extend(packet);
                    expected.extend(b);
                }
                check_external(&vvdec, &stream, &expected, width, height, qp);
            }
        }
    }
    assert!(signalled_pictures > 0, "no picture signalled ALF filters");
}

#[test]
#[ignore = "requires FFmpeg with VVC and NC_VVDEC pointing to vvdecapp"]
fn temporal_candidates_and_merge_lists_match_standard_decoders() {
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    // Turning and zooming texture: vectors vary across the picture and
    // continue in time, so collocated, history, pairwise and zero merge
    // candidates are all chosen; a key mid-stream resets the collocated
    // motion; a picture edge inside a CTU limits the bottom-right blocks;
    // multi-type tree leaves and both CTU sizes.
    for ctu in CTU_SIZES {
        for (temporal, merge, mtt, affine) in [
            (1., 2., 0., 0.),
            (1., 4., 0., 0.),
            (0., 4., 1., 0.),
            (1., 6., 1., 0.),
            (1., 1., 0., 0.),
            (1., 5., 0., 2.),
        ] {
            for qp in [22, 37, 45] {
                let (width, height) = (200, 136);
                let mut c = config_ctu(width, height, qp, ctu);
                c.intra_period = 5;
                c.params.set("vvc.temporal_mvp", temporal);
                c.params.set("vvc.merge_candidates", merge);
                c.params.set("vvc.mtt", mtt);
                c.params.set("vvc.affine", affine);
                c.params.set(
                    "vvc.alf",
                    std::env::var("ALF").map_or(1., |v| v.parse().unwrap()),
                );
                let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
                let mut decoder = nc_vvc::Decoder::new();
                let mut stream = Vec::new();
                let mut expected = Vec::new();
                for frame in 0..7 {
                    let source = turning_texture(width, height, frame);
                    let packet = encoder.encode(&source).data;
                    let own = decoder.decode(&packet).unwrap();
                    let mut a = Vec::new();
                    let mut b = Vec::new();
                    yuv::write_frame(&mut a, &own).unwrap();
                    yuv::write_frame(&mut b, encoder.reconstruction()).unwrap();
                    let case = format!(
                        "ctu{ctu} tmvp{temporal} merge{merge} mtt{mtt} affine{affine} qp{qp}"
                    );
                    assert!(a == b, "{case} frame{frame}");
                    stream.extend(packet);
                    expected.extend(b);
                }
                check_external(&vvdec, &stream, &expected, width, height, qp);
            }
        }
    }
}

#[test]
#[ignore = "requires FFmpeg with VVC and NC_VVDEC pointing to vvdecapp"]
fn intra_refresh_switches_temporal_candidates_off() {
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    // Under intra refresh a P picture's merge index must not depend on the
    // collocated motion (a decoder that lost a picture has other motion):
    // ph_temporal_mvp_enabled_flag 0 while the SPS enables TMVP.
    let (width, height, qp) = (200, 136, 37);
    let mut c = config_ctu(width, height, qp, 64);
    c.intra_period = 1000;
    c.params.set("vvc.temporal_mvp", 1.);
    c.params.set("vvc.merge_candidates", 4.);
    let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
    let mut decoder = nc_vvc::Decoder::new();
    let mut stream = Vec::new();
    let mut expected = Vec::new();
    for frame in 0..10 {
        if frame == 4 {
            encoder
                .set_intra_refresh(Some(nc_encoder::IntraRefresh { period: 4 }))
                .unwrap();
        }
        let source = turning_texture(width, height, frame);
        let packet = encoder.encode(&source).data;
        let own = decoder.decode(&packet).unwrap();
        let mut a = Vec::new();
        let mut b = Vec::new();
        yuv::write_frame(&mut a, &own).unwrap();
        yuv::write_frame(&mut b, encoder.reconstruction()).unwrap();
        assert!(a == b, "frame{frame}");
        stream.extend(packet);
        expected.extend(b);
    }
    check_external(&vvdec, &stream, &expected, width, height, qp);
}

#[test]
#[ignore = "requires NC_FFMPEG (FFmpeg >= 8.0) and NC_VVDEC pointing to vvdecapp"]
fn large_transforms_match_standard_decoders() {
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    // 64-point transform units (`vvc.tx64`) of intra and inter 64x64 CUs,
    // with smaller CUs beside them, across picture edges (CTUs split
    // implicitly there): plain, trellis and dependent quantisation, intra
    // sub-partitions and MTS on the smaller CUs, and random decisions.
    let variants: [&[(&str, f32)]; 5] = [
        &[],
        &[("quant.trellis", 1.)],
        &[("vvc.dep_quant", 1.)],
        &[("vvc.isp", 1.), ("vvc.mts", 1.)],
        &[("debug.random_decisions", 5.)],
    ];
    for (width, height) in [(64, 64), (136, 72), (200, 136)] {
        for (variant, qp) in variants.iter().zip([4, 22, 32, 41, 51]) {
            let mut c = config_ctu(width, height, qp, 64);
            c.params.set("vvc.tx64", 1.);
            for &(name, value) in *variant {
                c.params.set(name, value);
            }
            let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
            let mut decoder = nc_vvc::Decoder::new();
            let mut stream = Vec::new();
            let mut expected = Vec::new();
            for frame in 0..4 {
                let source = turning_texture(width, height, frame);
                let packet = encoder.encode(&source).data;
                let own = decoder.decode(&packet).unwrap();
                let mut a = Vec::new();
                let mut b = Vec::new();
                yuv::write_frame(&mut a, &own).unwrap();
                yuv::write_frame(&mut b, encoder.reconstruction()).unwrap();
                assert!(a == b, "{width}x{height} {variant:?} qp{qp} frame{frame}");
                stream.extend(packet);
                expected.extend(b);
            }
            check_external(&vvdec, &stream, &expected, width, height, qp);
        }
    }
}

#[test]
#[ignore = "requires NC_FFMPEG (FFmpeg >= 8.0) and NC_VVDEC pointing to vvdecapp"]
fn multiple_transform_selection_matches_standard_decoders() {
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    // Oblique texture moving across CTU and picture edges: explicit MTS on
    // intra and inter CUs (every index, 32-point zero-out), implicit MTS
    // on intra sub-partitions and, with explicit intra off, on every intra
    // CU; with dependent quantisation, and on multi-type tree leaves.
    let variants: [&[(&str, f32)]; 6] = [
        &[("vvc.isp", 1.)],
        &[("vvc.mts_intra", 0.), ("vvc.isp", 1.)],
        &[("vvc.mts_inter", 0.), ("vvc.mts_min_levels", 1.)],
        &[("vvc.mts_predicted", 1.), ("vvc.mts_lambda_scale", 0.25)],
        &[("vvc.dep_quant", 1.)],
        &[("vvc.mtt", 2.), ("vvc.mts_candidates", 1.)],
    ];
    for ctu in CTU_SIZES {
        for (width, height) in [(66, 34), (200, 136)] {
            for (variant, qp) in variants
                .iter()
                .zip([22, 37, 45, 27, 32, 17].into_iter().cycle())
            {
                let mut c = config_ctu(width, height, qp, ctu);
                c.params.set("vvc.mts", 1.);
                for &(name, value) in *variant {
                    c.params.set(name, value);
                }
                let mut encoder = Encoder::<VvcBackend>::new(c).unwrap();
                let mut decoder = nc_vvc::Decoder::new();
                let mut stream = Vec::new();
                let mut expected = Vec::new();
                for frame in 0..3 {
                    let source = turning_texture(width, height, frame);
                    let packet = encoder.encode(&source).data;
                    let own = decoder.decode(&packet).unwrap();
                    let mut a = Vec::new();
                    let mut b = Vec::new();
                    yuv::write_frame(&mut a, &own).unwrap();
                    yuv::write_frame(&mut b, encoder.reconstruction()).unwrap();
                    assert!(a == b, "{width}x{height} ctu{ctu} {variant:?} frame{frame}");
                    stream.extend(packet);
                    expected.extend(b);
                }
                check_external(&vvdec, &stream, &expected, width, height, qp);
            }
        }
    }
}
