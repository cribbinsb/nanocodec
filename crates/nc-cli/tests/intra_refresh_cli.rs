//! `--intra-refresh N`: codecs whose decoders recover by the sweep accept
//! it; VP9 and AV2, whose entropy state a lost frame corrupts until the
//! next key (`CodecBackend::INTRA_REFRESH`), refuse it instead of coding a
//! stream that never recovers.
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const FRAMES: usize = 3;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "nanocodec-cli-refresh-{}-{stamp}",
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

#[test]
fn only_codecs_that_recover_accept_intra_refresh() {
    let scratch = Scratch::new();
    let input = scratch.0.join("source.yuv");
    let bytes: Vec<u8> = (0..FRAMES * WIDTH * HEIGHT * 3 / 2)
        .map(|i| ((i * 31 + i / WIDTH * 7) % 251) as u8)
        .collect();
    fs::write(&input, bytes).unwrap();
    let size = |n: usize| n.to_string();
    for (codec, accepted) in [
        ("h264", true),
        ("hevc", true),
        ("vvc", true),
        ("av1", true),
        ("vp9", false),
        ("av2", false),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
            .args([
                "encode",
                "--codec",
                codec,
                "--input",
                input.to_str().unwrap(),
            ])
            .args(["--output", scratch.0.join("out.bin").to_str().unwrap()])
            .args(["--width", &size(WIDTH), "--height", &size(HEIGHT)])
            .args(["--fps", "30", "--frames", &size(FRAMES)])
            .args(["--bitrate", "200", "--intra-refresh", "2"])
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.success(), accepted, "{codec}: {stderr}");
        if !accepted {
            assert!(stderr.contains("intra refresh"), "{codec}: {stderr}");
        }
    }
}
