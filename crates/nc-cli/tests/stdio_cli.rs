//! `-` as a path argument streams stdin/stdout and never creates a file
//! named `-` (`src/stdio.rs`).
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

const WIDTH: usize = 64;
const HEIGHT: usize = 48;
const FRAMES: usize = 4;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nc-cli-stdio-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
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

/// Run `nanocodec args` in `dir` with `stdin`; its exit success and stdout.
fn run(dir: &Path, args: &[&str], stdin: &[u8]) -> (bool, Vec<u8>) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    {
        let mut pipe = child.stdin.take().unwrap();
        // The child may refuse its arguments before reading: a broken pipe is fine.
        let _ = pipe.write_all(stdin);
    }
    let output = child.wait_with_output().unwrap();
    (output.status.success(), output.stdout)
}

fn encode_args<'a>(input: &'a str, output: &'a str) -> Vec<&'a str> {
    vec![
        "encode", "--codec", "h264", "--input", input, "--output", output, "--width", "64",
        "--height", "48", "--fps", "30", "--qp", "30", "--frames", "4",
    ]
}

#[test]
fn dash_streams_stdin_and_stdout_in_encode_and_decode() {
    let scratch = Scratch::new();
    let dir = &scratch.0;
    let source: Vec<u8> = (0..FRAMES * WIDTH * HEIGHT * 3 / 2)
        .map(|i| ((i * 17 + i / WIDTH * 5) % 251) as u8)
        .collect();
    fs::write(dir.join("source.yuv"), &source).unwrap();

    let (ok, _) = run(dir, &encode_args("source.yuv", "file.264"), &[]);
    assert!(ok);
    let (ok, piped) = run(dir, &encode_args("-", "-"), &source);
    assert!(ok);
    assert_eq!(piped, fs::read(dir.join("file.264")).unwrap());

    let decode = ["decode", "--codec", "h264", "--input", "-", "--output", "-"];
    let (ok, decoded) = run(dir, &decode, &piped);
    assert!(ok);
    assert_eq!(decoded.len(), source.len());

    let mut twice = encode_args("source.yuv", "-");
    twice.extend(["--recon", "-"]);
    assert!(
        !run(dir, &twice, &[]).0,
        "two outputs on stdout must be refused"
    );

    assert!(!dir.join("-").exists(), "a file named '-' was created");
}
