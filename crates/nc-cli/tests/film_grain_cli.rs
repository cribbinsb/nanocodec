//! `--film-grain [STRENGTH]`: turns on AV1 film grain synthesis, and is
//! refused for every other codec and for a bad strength.
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const FRAMES: usize = 3;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nc-cli-grain-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        let bytes: Vec<u8> = (0..FRAMES * WIDTH * HEIGHT * 3 / 2)
            .map(|i| ((i * 29 + (i / WIDTH) * 11) % 251) as u8)
            .collect();
        fs::write(path.join("source.yuv"), bytes).unwrap();
        Self(path)
    }

    /// Encode the source with `codec` and `extra` flags into `name`.
    fn encode(&self, codec: &str, name: &str, extra: &[&str]) -> (Output, PathBuf) {
        let output = self.0.join(name);
        let result = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
            .args(["encode", "--codec", codec, "--input"])
            .arg(self.0.join("source.yuv"))
            .arg("--output")
            .arg(&output)
            .args(["--width", "64", "--height", "64", "--fps", "30"])
            .args(["--frames", "3", "--qp", "30"])
            .args(extra)
            .output()
            .unwrap();
        (result, output)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn film_grain_changes_the_av1_stream() {
    let scratch = Scratch::new();
    let (plain, plain_path) = scratch.encode("av1", "plain.ivf", &[]);
    let (grain, grain_path) = scratch.encode("av1", "grain.ivf", &["--film-grain"]);
    let (strong, _) = scratch.encode("av1", "strong.ivf", &["--film-grain", "1.5"]);
    for run in [&plain, &grain, &strong] {
        assert!(run.status.success(), "{}", stderr(run));
    }
    // The sequence header signals `film_grain_params_present` (AV1 §5.5.1).
    assert_ne!(fs::read(plain_path).unwrap(), fs::read(grain_path).unwrap());
}

#[test]
fn film_grain_is_refused_outside_av1() {
    let scratch = Scratch::new();
    for codec in ["h264", "hevc", "vvc", "vp9", "av2"] {
        let (run, _) = scratch.encode(codec, codec, &["--film-grain"]);
        assert!(!run.status.success(), "{codec} accepted --film-grain");
        assert!(
            stderr(&run).contains("only available for AV1"),
            "{}",
            stderr(&run)
        );
    }
}

#[test]
fn film_grain_strength_must_be_positive() {
    let scratch = Scratch::new();
    for strength in ["0", "-1", "strong"] {
        let (run, _) = scratch.encode("av1", "bad.ivf", &["--film-grain", strength]);
        assert!(!run.status.success(), "accepted strength {strength}");
        assert!(
            stderr(&run).contains("bad --film-grain strength"),
            "{}",
            stderr(&run)
        );
    }
}
