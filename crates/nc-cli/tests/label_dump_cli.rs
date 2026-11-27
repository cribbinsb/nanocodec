//! `--label-dump` leaves the stream exactly as the plain encode codes it, on
//! every codec with node trials, at medium and at a rung with the RD search's
//! context gate (`slow`: `planner.rd_split_context`), with tool trials on.
//! The context-gated rung once panicked in the tool trials' undo of the depth
//! map (2026-11-22: the map was first sized inside a trial).
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

const WIDTH: usize = 160;
const HEIGHT: usize = 96;
/// A key frame and inter frames that search.
const FRAMES: usize = 4;
/// Samples the pattern moves per frame, so inter frames have motion to code.
const MOTION: usize = 3;
/// Engine tools flipped per superblock: each on or off against the rungs'
/// settings.
const TOOLS: &[&str] = &[
    "rd_split:planner.rd_split=1",
    "key_rd_split:planner.key_rd_split=0",
    "final_stage:decision.final_stage=1",
    "intra_rd_modes:decision.intra_rd_modes=4",
];
/// The trellis, for the codecs that read it per superblock.
const TRELLIS: &str = "trellis:quant.trellis=0";
const TRELLIS_CODECS: &[&str] = &["hevc", "av1"];

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "nanocodec-cli-labels-{}-{stamp}",
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

/// Textured frames moving right by [`MOTION`] samples a frame (I420).
fn source() -> Vec<u8> {
    let luma = WIDTH * HEIGHT;
    let mut bytes = Vec::with_capacity(FRAMES * luma * 3 / 2);
    for frame in 0..FRAMES {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let u = x + frame * MOTION;
                bytes.push(((u * 7 + y * 13 + (u / 9) * (y / 7) * 31) % 251) as u8);
            }
        }
        bytes.extend((0..luma / 2).map(|i| ((i * 5 + frame) % 200) as u8));
    }
    bytes
}

/// The stream of `input` at `preset`, with the label dump into `dump` if given.
fn encode(input: &Path, codec: &str, preset: &str, output: &Path, dump: Option<&Path>) -> Vec<u8> {
    let size = |n: usize| n.to_string();
    let mut command = Command::new(env!("CARGO_BIN_EXE_nanocodec"));
    command
        .args(["encode", "--codec", codec, "--preset", preset])
        .args(["--input", input.to_str().unwrap()])
        .args(["--output", output.to_str().unwrap()])
        .args(["--width", &size(WIDTH), "--height", &size(HEIGHT)])
        .args(["--fps", "30", "--frames", &size(FRAMES), "--qp", "32"]);
    if let Some(dump) = dump {
        command.args(["--label-dump", dump.to_str().unwrap()]);
        for tool in TOOLS {
            command.args(["--label-tool", tool]);
        }
        if TRELLIS_CODECS.contains(&codec) {
            command.args(["--label-tool", TRELLIS]);
        }
    }
    let result = command.output().unwrap();
    assert!(
        result.status.success(),
        "{codec} {preset} (dump {}): {}",
        dump.is_some(),
        String::from_utf8_lossy(&result.stderr)
    );
    fs::read(output).unwrap()
}

#[test]
fn the_dump_leaves_every_stream_unchanged() {
    let scratch = Scratch::new();
    let input = scratch.0.join("source.yuv");
    fs::write(&input, source()).unwrap();
    for codec in ["hevc", "av1", "vvc", "vp9", "av2"] {
        for preset in ["medium", "slow"] {
            let plain = encode(&input, codec, preset, &scratch.0.join("plain.bin"), None);
            let dump = scratch.0.join(format!("{codec}-{preset}"));
            let dumped = encode(
                &input,
                codec,
                preset,
                &scratch.0.join("dump.bin"),
                Some(&dump),
            );
            assert!(
                plain == dumped,
                "{codec} {preset}: the dump changed the stream"
            );
            assert!(dump.join("labels.json").exists() && dump.join("node/cost.npy").exists());
        }
    }
}
