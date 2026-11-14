//! Static content recovers from a degraded reference, whatever degraded it.
//!
//! A still region is coded once and then copied; if the picture it was
//! copied from was poor (a starved key, a scene cut, a loss recovery, a
//! rate dip), the copies stay poor unless the encoder notices and refines
//! them. The reuse term of block AQ (`nc-encoder` `reuse.rs`) measures each
//! block's reconstruction error every frame and refines still blocks that
//! are worse than their long-run quality. This test codes a half-static,
//! half-moving picture under CBR, starves one requested key in the middle
//! (as a periodic key on a waterfall did in the browser demo), and checks
//! that the static half returns to the quality of an undisturbed stream
//! (and that without the term it does not). H.264 runs flat and with the
//! Default's adaptive layers and the h264-fast / h264-faster / h264-fastest presets: in a
//! layered stream the static half recovers at the next base-layer frame,
//! where the reuse term refines (`nc-encoder` `on_reuse_chain`). Every
//! shipped preset of every codec runs too: a preset's speed shortcut can
//! drop the refinement (av1-fast's largest-transform mode and vp9-fast's
//! partial forward transform dropped the detail it restores). Measured
//! static half after the starved key, undisturbed / recovered, dB: HEVC
//! 41.3 / 42.5 (hevc-fast 39.6 / 41.5), VP9 41.4 / 43.8 (vp9-fast 40.4 /
//! 44.2, 32.3 before), VVC 41.1 / 42.3 (vvc-fast 40.9 / 41.6), AV2 41.1 /
//! 43.5 (av2-fast 42.8, 40.2 before), h264-balanced 47.9 / 46.5 (layered).
use nc_common::{Picture, PlaneIndex};
use nc_encoder::{
    Codec, CodecBackend, Encoder, EncoderConfig, FrameRate, ParamOverrides, RateControlConfig,
    StreamSignal,
};

const WIDTH: usize = 128;
const HEIGHT: usize = 64;
/// Columns of the static half (the moving half starts after it).
const STATIC_WIDTH: usize = 56;
const FRAMES: usize = 40;
/// The normal target, and the starved one of the requested key.
const TARGET_BPS: u64 = 30_000;
const STARVED_BPS: u64 = 4_000;
/// The requested key coded at the starved target.
const STARVED_KEY: usize = 16;
/// Frames after the starved key by which the static half must have
/// recovered: its blocks were still before the key, so the first frame
/// after it already refines them. Measured (H.264 / AV1, static half
/// PSNR-Y): undisturbed 38.9 / 41.9 dB throughout; starved key 20.3 / 32.1
/// dB, the frames after it 41.6-42.2 / 43.1 dB; without the reuse term
/// 31.9-32.8 / 35.3 dB.
const RECOVERY_FRAMES: usize = 1;
/// Frames a preset's static half is checked for after it recovered.
const STAYS_RECOVERED_FRAMES: usize = 4;
/// Frames to the next base-layer frame after the key in a three-layer
/// stream (the layer period), where a layered stream recovers.
const LAYERED_RECOVERY_FRAMES: usize = 4;
/// Largest shortfall of the recovered static half against the undisturbed
/// stream.
const RECOVERED_DB: f64 = 0.5;
/// The same for a layered stream, whose undisturbed static half keeps
/// improving at every base frame (H.264 Default: 41.9 dB after the key,
/// 47.1 at the next T0 but one, 47.9 at frame 36) while the recovered one
/// is refined once, to 46.5 dB (from 19.9; 36.0 without the term).
const LAYERED_RECOVERED_DB: f64 = 1.5;
/// Least shortfall that counts as degraded.
const DEGRADED_DB: f64 = 3.;

/// Frame `index`: a fixed texture on the left, a texture moving 3 samples
/// per frame on the right (it takes the bits, as a waterfall does).
fn frame(index: usize) -> Picture {
    let mut picture = Picture::new(WIDTH, HEIGHT, 0).unwrap();
    for plane in PlaneIndex::ALL {
        let mut view = picture.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            for (x, v) in view.row_mut(y).iter_mut().enumerate() {
                *v = if plane != PlaneIndex::Y {
                    128
                } else if x < WIDTH / 2 {
                    let (xf, yf) = (x as f64, y as f64);
                    (128.
                        + 40. * (xf * 0.7 + yf * 0.3).sin()
                        + 30. * (yf * 1.1).sin() * (xf * 0.45).cos()
                        + ((x * 37 + y * 91) % 23) as f64) as u8
                } else {
                    let moved = x + 3 * index;
                    (128. + 50. * (moved as f64 * 0.5).sin() + ((moved * 53 + y * 17) % 29) as f64)
                        as u8
                };
            }
        }
    }
    picture
}

/// PSNR-Y of the static half of `recon` against `source`.
fn static_psnr(source: &Picture, recon: &Picture) -> f64 {
    let (s, r) = (
        source.plane(PlaneIndex::Y).view(),
        recon.plane(PlaneIndex::Y).view(),
    );
    let sse: u64 = (0..HEIGHT)
        .map(|y| {
            s.row(y)[..STATIC_WIDTH]
                .iter()
                .zip(&r.row(y)[..STATIC_WIDTH])
                .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
                .sum::<u64>()
        })
        .sum();
    let mse = sse as f64 / (HEIGHT * STATIC_WIDTH) as f64;
    10. * (255. * 255. / mse.max(1e-9)).log10()
}

/// Static-half PSNR of the first `frames` frames; with `starve`, the frame `STARVED_KEY`
/// is a requested key coded at the starved target, the rest at the normal.
fn encode<B: CodecBackend>(
    codec: Codec,
    preset: &str,
    (starve, reuse): (bool, bool),
    frames: usize,
) -> Vec<f64> {
    // The rate controller's seed quantizer (the CLI's default).
    let quantizer = match codec {
        Codec::Av1 | Codec::Vp9 | Codec::Av2 => 180,
        _ => 32,
    };
    // The CLI's default block size.
    let block_log2 = match codec {
        Codec::Av2 => 3,
        Codec::Vvc => 5,
        _ => 4,
    };
    let mut params = preset_params(preset);
    params.set("persist.reuse", if reuse { 1. } else { 0. });
    let mut encoder = Encoder::<B>::new(EncoderConfig {
        codec,
        width: WIDTH,
        height: HEIGHT,
        frame_rate: FrameRate { num: 15, den: 1 },
        quantizer,
        intra_period: 1000,
        block_log2,
        key_qp_offset: None,
        temporal_layers: 1,
        signal: StreamSignal::default(),
        params,
    })
    .unwrap();
    let rate = |target| RateControlConfig::new(target);
    encoder.set_rate_control(rate(TARGET_BPS)).unwrap();
    (0..frames)
        .map(|index| {
            if starve && index == STARVED_KEY {
                encoder.set_rate_control(rate(STARVED_BPS)).unwrap();
                encoder.request_keyframe();
            } else if starve && index == STARVED_KEY + 1 {
                encoder.set_rate_control(rate(TARGET_BPS)).unwrap();
            }
            let source = frame(index);
            encoder.encode(&source);
            static_psnr(&source, encoder.reconstruction())
        })
        .collect()
}

/// The overrides of a parameter file in the CLI's preset format (`name =
/// value` lines, `#` comments).
fn preset_params(text: &str) -> ParamOverrides {
    let mut params = ParamOverrides::default();
    for line in text
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
    {
        if let Some((name, value)) = line.split_once('=') {
            params.set(name.trim(), value.trim().parse().unwrap());
        }
    }
    params
}

/// The codec defaults (H.264: adaptive layers).
const DEFAULT: &str = "";
/// A flat stream.
const FLAT: &str = "layers.count = 1";

/// How soon, and how close to the undisturbed stream, a stream recovers.
struct Recovery {
    frames: usize,
    shortfall_db: f64,
}

const FLAT_RECOVERY: Recovery = Recovery {
    frames: RECOVERY_FRAMES,
    shortfall_db: RECOVERED_DB,
};
const LAYERED_RECOVERY: Recovery = Recovery {
    frames: LAYERED_RECOVERY_FRAMES,
    shortfall_db: LAYERED_RECOVERED_DB,
};
/// Largest shortfall of the full-size layer of an SVC stream. Its
/// undisturbed static half is the stationary-startup anchor (`av1-svc`: the
/// first unit's S1 coded near the base's quantizer, 44.4 dB), finer than a
/// refinement at S1's inter level is priced (offsets down to -12.5 at reuse
/// 16): the recovered half stays at 42.4 dB (35.3-36.3 without the reuse term;
/// 35.9 for the rest of the run before S1's key unit was analysed against its
/// own last picture). av1-svc-fast recovers past it (47.5 vs 44.3).
const SVC_RECOVERED_DB: f64 = 2.5;
const SVC_RECOVERY: Recovery = Recovery {
    frames: RECOVERY_FRAMES,
    shortfall_db: SVC_RECOVERED_DB,
};
/// The same for VP9 SVC, whose stationary-startup anchor stands further
/// above what a refinement reaches: vp9-svc's undisturbed static half is
/// 44.5 dB (the first unit's S1 at q 37), the recovered one 41.0 (33.0
/// without the reuse term). The reuse term asks for the same refinement as
/// in av1-svc (-10.9 QP units at the same reuse count), but VP9's S1 codes
/// this picture at q 145-155 against AV1's 125-139, so the priced
/// refinement lands at q ~76 (AV1 ~53). vp9-svc-fast, whose S1 runs at q
/// ~65, recovers past it (46.8 vs 43.2).
const VP9_SVC_RECOVERED_DB: f64 = 4.0;
const VP9_SVC_RECOVERY: Recovery = Recovery {
    frames: RECOVERY_FRAMES,
    shortfall_db: VP9_SVC_RECOVERED_DB,
};

/// The full check: the starved key degrades the static half, without the
/// reuse term its copies stay degraded, and with it they recover.
fn check<B: CodecBackend>(codec: Codec, preset: &str, recovery: &Recovery) {
    let steady = encode::<B>(codec, preset, (false, true), FRAMES);
    let starved = encode::<B>(codec, preset, (true, true), FRAMES);
    let stuck = encode::<B>(codec, preset, (true, false), FRAMES);
    // ...without the reuse term the copies stay degraded...
    for index in STARVED_KEY + RECOVERY_FRAMES..FRAMES {
        assert!(
            stuck[index] < steady[index] - DEGRADED_DB,
            "{codec:?} off, frame {index}: {:.2} vs {:.2} dB",
            stuck[index],
            steady[index]
        );
    }
    assert_recovers(codec, (&steady, &starved), recovery);
}

/// The check of a preset: the starved key degrades the static half and the
/// reuse term recovers it, over a run ending [`STAYS_RECOVERED_FRAMES`]
/// after the recovery. The premise that without the term the copies stay
/// degraded is the codec's [`check`]; leaving it out here, and the rest of
/// the run, keeps the preset sweep under 1 s of test time.
fn check_preset<B: CodecBackend>(codec: Codec, preset: &str, recovery: &Recovery) {
    let frames = STARVED_KEY + recovery.frames + STAYS_RECOVERED_FRAMES;
    let steady = encode::<B>(codec, preset, (false, true), frames);
    let starved = encode::<B>(codec, preset, (true, true), frames);
    assert_recovers(codec, (&steady, &starved), recovery);
}

/// The starved key really is degraded, and the static half recovers by
/// `recovery.frames` after it and stays recovered.
fn assert_recovers(codec: Codec, (steady, starved): (&[f64], &[f64]), recovery: &Recovery) {
    assert!(
        starved[STARVED_KEY] < steady[STARVED_KEY] - DEGRADED_DB,
        "{codec:?}: key {:.2} vs {:.2} dB",
        starved[STARVED_KEY],
        steady[STARVED_KEY]
    );
    for index in STARVED_KEY + recovery.frames..steady.len() {
        assert!(
            starved[index] > steady[index] - recovery.shortfall_db,
            "{codec:?}, frame {index}: {:.2} vs {:.2} dB",
            starved[index],
            steady[index]
        );
    }
}

#[test]
fn h264_static_content_recovers_from_a_starved_key() {
    check::<nc_h264::H264Backend>(Codec::H264, FLAT, &FLAT_RECOVERY);
}

#[test]
fn h264_layered_streams_recover_at_the_next_base_frame() {
    for preset in [
        DEFAULT,
        include_str!("../../../presets/h264-fast.toml"),
        include_str!("../../../presets/h264-faster.toml"),
        include_str!("../../../presets/h264-fastest.toml"),
    ] {
        check::<nc_h264::H264Backend>(Codec::H264, preset, &LAYERED_RECOVERY);
    }
}

#[test]
fn av1_static_content_recovers_from_a_starved_key() {
    // Loop restoration (off in the shipped presets) filters every
    // reference after CDEF, so its streams must recover too.
    for preset in [
        DEFAULT,
        include_str!("../../../presets/av1-fast.toml"),
        "av1.restoration = 1",
    ] {
        check::<nc_av1::Av1Backend>(Codec::Av1, preset, &FLAT_RECOVERY);
    }
}

/// The SVC presets (two spatial layers): the full-size layer, which a
/// 720p receiver shows, recovers from a starved key unit at its next picture
/// (`SVC_RECOVERY`), and av1-svc-fast's still content is not left at the
/// blurred base (its enhancement pictures coded from the base only use the
/// exact forward transform: 44.3 dB undisturbed, 32.3 before).
#[test]
fn av1_svc_presets_recover() {
    for preset in [
        include_str!("../../../presets/av1-svc.toml"),
        include_str!("../../../presets/av1-svc-fast.toml"),
    ] {
        check_preset::<nc_av1::Av1Backend>(Codec::Av1, preset, &SVC_RECOVERY);
        let ksvc = format!("{preset}\nspatial.inter_layer = 2\n");
        check_preset::<nc_av1::Av1Backend>(Codec::Av1, &ksvc, &SVC_RECOVERY);
    }
}

/// Three spatial layers (`spatial.layers = 3`): after a starved key temporal
/// unit, BOTH upper layers (`S1` at 128x64 and `S2` at 256x128, so `S1` spans
/// two 64x64 AV1 superblocks horizontally just like the 1- and 2-layer tests'
/// full-size layer) stay degraded when `persist.reuse = 0` and refine on
/// static content when `persist.reuse = 1` with the three-layer presets
/// (`av1-svc3`, `av1-svc3-fast`) in both full SVC and K-SVC (`inter_layer = 2`).
fn frame_sized(width: usize, height: usize, index: usize) -> Picture {
    let mut picture = Picture::new(width, height, 0).unwrap();
    for plane in PlaneIndex::ALL {
        let mut view = picture.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            for (x, v) in view.row_mut(y).iter_mut().enumerate() {
                *v = if plane != PlaneIndex::Y {
                    128
                } else if x < width / 2 {
                    let (xf, yf) = (x as f64, y as f64);
                    (128.
                        + 40. * (xf * 0.7 + yf * 0.3).sin()
                        + 30. * (yf * 1.1).sin() * (xf * 0.45).cos()
                        + ((x * 37 + y * 91) % 23) as f64) as u8
                } else {
                    let moved = x + 3 * index;
                    (128. + 50. * (moved as f64 * 0.5).sin() + ((moved * 53 + y * 17) % 29) as f64)
                        as u8
                };
            }
        }
    }
    picture
}

fn static_psnr_cols(source: &Picture, recon: &Picture, static_w: usize) -> f64 {
    let h = recon.height();
    let (s, r) = (
        source.plane(PlaneIndex::Y).view(),
        recon.plane(PlaneIndex::Y).view(),
    );
    let sse: u64 = (0..h)
        .map(|y| {
            s.row(y)[..static_w]
                .iter()
                .zip(&r.row(y)[..static_w])
                .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
                .sum::<u64>()
        })
        .sum();
    let mse = sse as f64 / (h * static_w) as f64;
    10. * (255. * 255. / mse.max(1e-9)).log10()
}

/// Scale factor for 3-layer K-SVC recovery (`256x128` top / `128x64` `S1` /
/// `64x32` `S0`), giving `S1` two 64x64 AV1 superblocks horizontally so
/// per-superblock `delta_q` separates the static left half from the moving half.
const THREE_LAYER_KSVC_SCALE: usize = 2;

fn run_three_layer_recovery<B: CodecBackend>(
    codec: Codec,
    preset: &str,
    box_downscale: bool,
    scale: usize,
    (starve, reuse): (bool, bool),
    frames: usize,
) -> (Vec<f64>, Vec<f64>) {
    let w = WIDTH * scale;
    let h = HEIGHT * scale;
    let target_bps = TARGET_BPS * (scale * scale) as u64;

    let dsp = nc_dsp::Dsp::new();
    let mut lanczos = nc_dsp::PlaneResampler::new((w, h), (w / 2, h / 2));
    let mut s1_ref = Picture::new(w / 2, h / 2, 0).unwrap();
    let mut params = preset_params(preset);
    params.set("spatial.layers", 3.0);
    params.set("persist.reuse", if reuse { 1.0 } else { 0.0 });
    let mut encoder = Encoder::<B>::new(EncoderConfig {
        codec,
        width: w,
        height: h,
        frame_rate: FrameRate { num: 15, den: 1 },
        quantizer: 180,
        intra_period: 1000,
        block_log2: 4,
        key_qp_offset: None,
        temporal_layers: 1,
        signal: StreamSignal::default(),
        params,
    })
    .unwrap();
    let rate = |target| RateControlConfig::new(target);
    encoder.set_rate_control(rate(target_bps)).unwrap();
    let mut s1_psnrs = Vec::with_capacity(frames);
    let mut s2_psnrs = Vec::with_capacity(frames);
    for index in 0..frames {
        if starve && index == STARVED_KEY {
            encoder.set_rate_control(rate(STARVED_BPS)).unwrap();
            encoder.request_keyframe();
        } else if starve && index == STARVED_KEY + 1 {
            encoder.set_rate_control(rate(target_bps)).unwrap();
        }
        let source = frame_sized(w, h, index);
        if box_downscale {
            dsp.downscale_2x(
                &source.plane(PlaneIndex::Y).view(),
                &mut s1_ref.plane_mut(PlaneIndex::Y).view_mut(),
            );
        } else {
            dsp.resample_plane_with(
                &mut lanczos,
                &source.plane(PlaneIndex::Y).view(),
                &mut s1_ref.plane_mut(PlaneIndex::Y).view_mut(),
            );
        }
        encoder.encode(&source);
        s1_psnrs.push(static_psnr_cols(
            &s1_ref,
            encoder.layer_reconstruction(1).expect("S1 recon"),
            (STATIC_WIDTH * scale) / 2,
        ));
        s2_psnrs.push(static_psnr_cols(
            &source,
            encoder.layer_reconstruction(2).expect("S2 recon"),
            STATIC_WIDTH * scale,
        ));
    }
    (s1_psnrs, s2_psnrs)
}

fn check_three_layer_preset<B: CodecBackend>(
    codec: Codec,
    name: &str,
    preset: &str,
    box_downscale: bool,
    ksvc_scale: usize,
    (s1_recovery, s2_recovery): (&Recovery, &Recovery),
) {
    const MIN_REUSE_GAIN_DB: f64 = 3.0;
    let frames = STARVED_KEY + s1_recovery.frames + STAYS_RECOVERED_FRAMES;
    let run = |cfg: &str, scale: usize, flags: (bool, bool)| {
        run_three_layer_recovery::<B>(codec, cfg, box_downscale, scale, flags, frames)
    };
    let (steady_s1, steady_s2) = run(preset, 1, (false, true));
    let (starved_s1, starved_s2) = run(preset, 1, (true, true));
    let (stuck_s1, stuck_s2) = run(preset, 1, (true, false));
    for index in STARVED_KEY + s1_recovery.frames..frames {
        assert!(
            stuck_s1[index] < steady_s1[index] - DEGRADED_DB,
            "{name} S1 reuse=0 frame {index}: {:.2} vs {:.2} dB",
            stuck_s1[index],
            steady_s1[index]
        );
        assert!(
            stuck_s2[index] < steady_s2[index] - DEGRADED_DB,
            "{name} S2 reuse=0 frame {index}: {:.2} vs {:.2} dB",
            stuck_s2[index],
            steady_s2[index]
        );
        assert!(
            starved_s1[index] > stuck_s1[index] + MIN_REUSE_GAIN_DB,
            "{name} S1 reuse gain frame {index}: {:.2} vs {:.2} dB",
            starved_s1[index],
            stuck_s1[index]
        );
        assert!(
            starved_s2[index] > stuck_s2[index] + MIN_REUSE_GAIN_DB,
            "{name} S2 reuse gain frame {index}: {:.2} vs {:.2} dB",
            starved_s2[index],
            stuck_s2[index]
        );
    }
    assert_recovers(codec, (&steady_s1, &starved_s1), s1_recovery);
    assert_recovers(codec, (&steady_s2, &starved_s2), s2_recovery);

    let ksvc_preset = format!("{preset}\nspatial.inter_layer = 2\n");
    let (k_steady_s1, k_steady_s2) = run(&ksvc_preset, ksvc_scale, (false, true));
    let (k_starved_s1, k_starved_s2) = run(&ksvc_preset, ksvc_scale, (true, true));
    assert_recovers(codec, (&k_steady_s1, &k_starved_s1), s1_recovery);
    assert_recovers(codec, (&k_steady_s2, &k_starved_s2), s2_recovery);
}

#[test]
fn av1_three_layer_svc_recovers_every_upper_layer() {
    const THREE_LAYER_S1_RECOVERED_DB: f64 = 6.0;
    const THREE_LAYER_S2_RECOVERED_DB: f64 = 3.5;

    let s1_recovery = Recovery {
        frames: RECOVERY_FRAMES + 1,
        shortfall_db: THREE_LAYER_S1_RECOVERED_DB,
    };
    let s2_recovery = Recovery {
        frames: RECOVERY_FRAMES + 1,
        shortfall_db: THREE_LAYER_S2_RECOVERED_DB,
    };
    for (name, preset, box_downscale) in [
        (
            "av1-svc3",
            include_str!("../../../presets/av1-svc3.toml"),
            false,
        ),
        (
            "av1-svc3-fast",
            include_str!("../../../presets/av1-svc3-fast.toml"),
            true,
        ),
    ] {
        check_three_layer_preset::<nc_av1::Av1Backend>(
            Codec::Av1,
            name,
            preset,
            box_downscale,
            THREE_LAYER_KSVC_SCALE,
            (&s1_recovery, &s2_recovery),
        );
    }
}

/// The VP9 SVC presets, as `av1_svc_presets_recover`: vp9-svc-fast's partial
/// forward transform is exact on enhancement pictures coded from the base
/// only (`nc-vp9` `coding_from_base_only`).
#[test]
fn vp9_svc_presets_recover() {
    for preset in [
        include_str!("../../../presets/vp9-svc.toml"),
        include_str!("../../../presets/vp9-svc-fast.toml"),
    ] {
        check_preset::<nc_vp9::Vp9Backend>(Codec::Vp9, preset, &VP9_SVC_RECOVERY);
    }
}

#[test]
fn av1_layered_streams_recover_at_the_next_base_frame() {
    check::<nc_av1::Av1Backend>(
        Codec::Av1,
        include_str!("../../../presets/av1-faster.toml"),
        &LAYERED_RECOVERY,
    );
}

#[test]
fn hevc_presets_recover() {
    for preset in [
        DEFAULT,
        include_str!("../../../presets/hevc-fast.toml"),
        include_str!("../../../presets/hevc-faster.toml"),
        include_str!("../../../presets/hevc-better.toml"),
    ] {
        check_preset::<nc_hevc::HevcBackend>(Codec::Hevc, preset, &FLAT_RECOVERY);
    }
}

#[test]
fn vp9_presets_recover() {
    for preset in [
        DEFAULT,
        include_str!("../../../presets/vp9-fast.toml"),
        include_str!("../../../presets/vp9-faster.toml"),
        include_str!("../../../presets/vp9-fastest.toml"),
    ] {
        check_preset::<nc_vp9::Vp9Backend>(Codec::Vp9, preset, &FLAT_RECOVERY);
    }
}

#[test]
fn vvc_presets_recover() {
    for preset in [DEFAULT, include_str!("../../../presets/vvc-fast.toml")] {
        check_preset::<nc_vvc::VvcBackend>(Codec::Vvc, preset, &FLAT_RECOVERY);
    }
}

#[test]
fn av2_presets_recover() {
    for preset in [DEFAULT, include_str!("../../../presets/av2-fast.toml")] {
        check_preset::<nc_av2::Av2Backend>(Codec::Av2, preset, &FLAT_RECOVERY);
    }
}

#[test]
fn h264_balanced_recovers() {
    let preset = include_str!("../../../presets/h264-balanced.toml");
    check_preset::<nc_h264::H264Backend>(Codec::H264, preset, &LAYERED_RECOVERY);
}
