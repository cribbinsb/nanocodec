//! `--key-qp-offset D`: every codec codes its fixed-QP key frame D steps of
//! its own quantiser finer than `--qp` (`nc_encoder::offset_key_quantizer`),
//! so a finer key costs more bits and a coarser one fewer. H.264 and AV1
//! once ignored the option and VP9 and AV2 refused it.
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

const WIDTH: usize = 96;
const HEIGHT: usize = 64;
/// A one-frame stream: its size is the key frame's.
const FRAMES: usize = 1;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "nanocodec-cli-keyqp-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The size of a one-key-frame fixed-QP stream of `input` with
/// `--key-qp-offset offset`.
fn key_bytes(scratch: &Path, input: &Path, codec: &str, qp: u32, offset: i32) -> u64 {
    let stream = scratch.join(format!("{codec}{offset}.bin"));
    let size = |n: usize| n.to_string();
    let output = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
        .args([
            "encode",
            "--codec",
            codec,
            "--input",
            input.to_str().unwrap(),
        ])
        .args(["--output", stream.to_str().unwrap()])
        .args(["--width", &size(WIDTH), "--height", &size(HEIGHT)])
        .args(["--fps", "30", "--frames", &size(FRAMES)])
        .args([
            "--qp",
            &qp.to_string(),
            "--key-qp-offset",
            &offset.to_string(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{codec} --key-qp-offset {offset}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::metadata(&stream).unwrap().len()
}

#[test]
fn the_offset_changes_every_codecs_key_quantizer() {
    let scratch = Scratch::new();
    let input = scratch.0.join("source.yuv");
    let bytes: Vec<u8> = (0..FRAMES * WIDTH * HEIGHT * 3 / 2)
        .map(|i| ((i * 29 + (i / WIDTH) * 11 + i / (WIDTH * HEIGHT) * 23) % 251) as u8)
        .collect();
    fs::write(&input, bytes).unwrap();
    // (codec, inter quantizer, offset step): QP codecs move 6 QP (a step
    // ratio of 2), qindex codecs 40 qindex (about the same ratio mid-range).
    for (codec, qp, step) in [
        ("h264", 30, 6),
        ("hevc", 30, 6),
        ("vvc", 32, 6),
        ("av1", 120, 40),
        ("vp9", 120, 40),
        ("av2", 120, 40),
    ] {
        let coarser = key_bytes(&scratch.0, &input, codec, qp, -step);
        let level = key_bytes(&scratch.0, &input, codec, qp, 0);
        let finer = key_bytes(&scratch.0, &input, codec, qp, step);
        assert!(
            coarser < level && level < finer,
            "{codec}: key bytes {coarser} / {level} / {finer} at offsets -{step} / 0 / {step}"
        );
    }
}
