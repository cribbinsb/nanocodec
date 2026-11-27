//! Grey content keeps exact grey where nothing needs coding: every codec and
//! preset, at high quantizers, codes a flat mid-grey picture back exactly in
//! every plane, and a grey zone plate (luma rings, chroma 128) with exact
//! chroma. AV2 once coded the zone plate's whole chroma plane as 127: its
//! first block has no neighbours, AV1, VP9 and AV2 fill a missing above edge
//! with 127 and a missing left one with 129, the chroma choice took a mode
//! predicting from above, the quantizer dropped the -1 residual and every
//! later block inherited it.
use std::{
    f64::consts::PI,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use nc_encoder::{Codec, Speed, speeds};

const WIDTH: usize = 256;
const HEIGHT: usize = 144;
const GREY: u8 = 128;
/// The demo's zone plate contrast (web/scene.js `ZONE.contrast`): 128 ± 110.
const ZONE_CONTRAST: f64 = 110.0;
const CODECS: [(Codec, &str); 6] = [
    (Codec::H264, "h264"),
    (Codec::Hevc, "hevc"),
    (Codec::Vvc, "vvc"),
    (Codec::Av1, "av1"),
    (Codec::Vp9, "vp9"),
    (Codec::Av2, "av2"),
];
/// The coarse end of each quantizer scale: `--qp` 0-51 for H.264, HEVC and
/// VVC, qindex 0-255 for AV1, VP9 and AV2.
fn high_quantizers(codec: &str) -> [u32; 5] {
    match codec {
        "h264" | "hevc" | "vvc" => [40, 44, 47, 49, 51],
        _ => [200, 220, 230, 240, 255],
    }
}

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("nanocodec-cli-grey-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// One I420 frame: `luma(x, y)` and grey chroma.
fn frame(luma: impl Fn(usize, usize) -> u8) -> Vec<u8> {
    let mut data: Vec<u8> = (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| (x, y)))
        .map(|(x, y)| luma(x, y))
        .collect();
    data.resize(WIDTH * HEIGHT * 3 / 2, GREY);
    data
}

/// The top-left corner of the demo's 1280x720 zone plate (web/scene.js
/// `drawZonePlate` at time zero, centre (640, 446.4)): the content whose
/// corner block took AV2's chroma to 127.
fn zone_plate(x: usize, y: usize) -> u8 {
    const SCALE: f64 = 1280.0;
    const CENTRE: (f64, f64) = (640.0, 446.4);
    let column = PI * (x as f64 - CENTRE.0).powi(2) / SCALE;
    let row = PI * (y as f64 - CENTRE.1).powi(2) / SCALE;
    let grey =
        (128.5 + ZONE_CONTRAST * (column.cos() * row.cos() - column.sin() * row.sin())) as u32;
    studio_luma(grey)
}

/// The page's RGB to YUV luma (nc-dsp `rgba_to_i420`, BT.601 limited
/// range) of a grey level: its chroma is exactly 128.
fn studio_luma(grey: u32) -> u8 {
    const WEIGHT_SUM: u32 = 66 + 129 + 25;
    (16 + ((WEIGHT_SUM * grey + 128) >> 8)) as u8
}

/// The reconstruction of one key frame of `input`.
fn reconstruct(scratch: &Path, input: &Path, codec: &str, preset: &str, qp: u32) -> Vec<u8> {
    let stream = scratch.join("stream.bin");
    let recon = scratch.join("recon.yuv");
    let size = |n: usize| n.to_string();
    let output = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
        .args(["encode", "--codec", codec, "--preset", preset])
        .args(["--input", input.to_str().unwrap()])
        .args(["--output", stream.to_str().unwrap()])
        .args(["--display-recon", recon.to_str().unwrap()])
        .args(["--width", &size(WIDTH), "--height", &size(HEIGHT)])
        .args(["--fps", "30", "--frames", "1", "--qp", &qp.to_string()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{codec} {preset} qp {qp}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::read(recon).unwrap()
}

#[test]
fn grey_stays_exact_at_high_quantizers() {
    let scratch = Scratch::new();
    let flat = scratch.0.join("flat.yuv");
    let zone = scratch.0.join("zone.yuv");
    fs::write(&flat, frame(|_, _| GREY)).unwrap();
    fs::write(&zone, frame(zone_plate)).unwrap();
    let luma = WIDTH * HEIGHT;
    let mut failures = Vec::new();
    for (id, codec) in CODECS {
        for preset in speeds(id).into_iter().map(Speed::name) {
            for qp in high_quantizers(codec) {
                let flat_recon = reconstruct(&scratch.0, &flat, codec, preset, qp);
                if flat_recon.iter().any(|&v| v != GREY) {
                    failures.push(format!("{codec} {preset} qp {qp}: flat grey"));
                }
                let zone_recon = reconstruct(&scratch.0, &zone, codec, preset, qp);
                if zone_recon[luma..].iter().any(|&v| v != GREY) {
                    failures.push(format!("{codec} {preset} qp {qp}: zone plate chroma"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "not exact grey:\n{}",
        failures.join("\n")
    );
}
