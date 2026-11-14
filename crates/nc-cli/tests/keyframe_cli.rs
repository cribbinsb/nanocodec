//! `--keyframe FRAME`: the frame is coded as a key frame on request (as the
//! browser demo's "Request keyframes", so a debug report replays it), and
//! the stream is otherwise as without the option up to that frame.
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

const WIDTH: usize = 96;
const HEIGHT: usize = 64;
const FRAMES: usize = 12;
const REQUESTED: usize = 5;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("nc-cli-key-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The frame types of an H.264 CBR encode of `input` (its rate log), with `extra` options.
fn frame_types(scratch: &Path, input: &Path, extra: &[&str]) -> Vec<String> {
    let log = scratch.join("rate.csv");
    let size = |n: usize| n.to_string();
    let status = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
        .args([
            "encode",
            "--codec",
            "h264",
            "--input",
            input.to_str().unwrap(),
            "--output",
            scratch.join("out.264").to_str().unwrap(),
        ])
        .args([
            "--width",
            &size(WIDTH),
            "--height",
            &size(HEIGHT),
            "--fps",
            "30",
            "--frames",
            &size(FRAMES),
        ])
        .args([
            "--bitrate",
            "300",
            "--intra-period",
            "300",
            "--rate-log",
            log.to_str().unwrap(),
        ])
        .args(extra)
        .status()
        .unwrap();
    assert!(status.success());
    let text = fs::read_to_string(&log).unwrap();
    text.lines()
        .skip(1)
        .map(|line| line.split(',').nth(1).unwrap().to_owned())
        .collect()
}

#[test]
fn a_requested_key_frame_is_coded_at_its_frame() {
    let scratch = Scratch::new();
    let input = scratch.0.join("source.yuv");
    let bytes: Vec<u8> = (0..FRAMES * WIDTH * HEIGHT * 3 / 2)
        .map(|i| ((i * 29 + (i / WIDTH) * 11 + i / (WIDTH * HEIGHT) * 23) % 251) as u8)
        .collect();
    fs::write(&input, bytes).unwrap();
    let plain = frame_types(&scratch.0, &input, &[]);
    let requested = frame_types(&scratch.0, &input, &["--keyframe", &REQUESTED.to_string()]);
    assert_eq!(plain.len(), FRAMES);
    assert_ne!(plain[REQUESTED], "Key");
    assert_eq!(requested[REQUESTED], "Key");
    assert_eq!(plain[..REQUESTED], requested[..REQUESTED]);
}
